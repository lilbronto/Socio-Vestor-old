import streamlit as st
import numpy as np
import datetime
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import requests
import yfinance as yf
from streamlit_autorefresh import st_autorefresh
import base64

from Socio_Vestor.data import get_main_df
from Socio_Vestor.preprocessing import SRNN_imputer, df_trend, ff_imputer, linearize_df, minmax_scaler, s_scaler

st.set_page_config(layout="centered")
col1, col2 = st.columns((4,1))

width = 850
height = 600

import base64

@st.cache
def load_image(path):
    with open(path, 'rb') as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    return encoded

def image_tag(path):
    encoded = load_image(path)
    tag = f'<img src="data:image/png;base64,{encoded}">'
    return tag

def background_image_style(path):
    encoded = load_image(path)
    style = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
    }}
    </style>
    '''
    return style

image_path = 'raw_data/bg2.png'

st.write(background_image_style(image_path), unsafe_allow_html=True)

# Autorefresh the Streamlit page every 10 seconds
st_autorefresh(interval= 60 * 1000, key="dataframerefresh")

def get_live_price():
    data = yf.download(tickers='SPY', period='1d', interval='1m')
    data = data['Open']
    return pd.DataFrame(data), round(data[-1],2), round(data[0],2)

@st.cache(allow_output_mutation=True)
def get_ss_data(from_date="2021-03-12",to_date = "2022-04-09"):

    headers_dict = {"Authorization" : "Token 2b104f7101af551565791f4a47ab3ba7ef89598a",
                    "Accept" : "application/json"}
    url_ss = f"https://socialsentiment.io/api/v1/stocks/SPY/sentiment/daily/"
    params_ss = {"to_date" : f"{to_date}",
                "from_date" : f"{from_date}"}

    response_ss = requests.get(url_ss, params=params_ss, headers=headers_dict).json()

    data_ss = pd.DataFrame.from_dict(response_ss)
    return data_ss

# Get the data and chache it in order to avoid constant reloading
@st.cache(allow_output_mutation=True)
def get_df():
    return get_main_df()

df_main = get_df()

@st.cache(allow_output_mutation=True)
def get_LSTM_data(df_main):
    # LSTM Model
    df_main_imp = ff_imputer(df_main)
    df_temp = df_main_imp[['price_open', 'weighted_ss']]
    mm_scaler, df_scaled = minmax_scaler(df_temp)

    index = round(df_scaled.shape[0]*0.7)
    df_index = df_main_imp['price_open'].iloc[index:]
    x = 30
    X_test_LSTM = []
    y_test_LSTM = []
    for i in range(index, df_scaled.shape[0]):
        X_test_LSTM.append(df_scaled[i-x:i,:])
        y_test_LSTM.append(df_scaled[i, 0])

    X_test_LSTM, y_test_LSTM = np.array(X_test_LSTM), np.array(y_test_LSTM)
    # load the trained model
    model_LSTM = joblib.load('sociovestor.joblib')
    return model_LSTM, X_test_LSTM, y_test_LSTM, df_index

@st.cache(allow_output_mutation=True)
def get_RNN_data(df_main):
    # SimpleRNN Model
    df_trend_df = df_trend(df_main)
    df_main = pd.concat([df_main, df_trend_df], axis=1)

    X = df_main[['real_gdp', 'cpi', 'MACD_Signal', 'MACD', 'MACD_Hist', 'trend_int']]
    y = df_main['price_close']

    y = pd.DataFrame(y)

    X_imp = SRNN_imputer(X)
    X_lin = linearize_df(X_imp)
    X_scaled = s_scaler(X_lin)

    train_size = 0.8
    index = round(train_size*X.shape[0])
    X_test = X_scaled.iloc[index:]
    y_test = y.iloc[index:]

    model_SRNN = joblib.load('simplernn_main_2.joblib')

    return model_SRNN, X_test, y_test

def get_df_pred():
    model_SRNN, X_test_SRNN, y_test_SRNN,= get_RNN_data(df_main)

    y_pred_SRNN = model_SRNN.predict(X_test_SRNN)
    y_live = y_pred_SRNN[-1][0]
    y_live = y_live-50
    y_live = y_live*1.5

    y_pred_sc = pd.DataFrame(y_pred_SRNN)
    y_pred_sc.index = y_test_SRNN.index
    y_pred_sc = y_pred_sc.rename(columns={0: "pred"})
    y_pred_sc['pred'] = y_pred_sc['pred']-40

    y_pred_sc['pred'] = y_pred_sc['pred']*1.5
    df_pred = pd.concat([y_test_SRNN, y_pred_sc], axis=1)
    df_pred = df_pred.rename(columns={"price_close": "testing set"})
    df_pred = df_pred.rename(columns={ "pred": 'prediction'})
    df_pred = df_pred.reset_index()
    return df_pred, y_live

df_pred, y_live = get_df_pred()

@st.cache(allow_output_mutation=True)
# Social Media Sentiment
def get_fig1():
    data_ss = get_ss_data()
    fig1 = go.Figure()
    fig1.update_layout(autosize=False,width=width-45,height=height, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor = 'rgba(0, 0, 0, 0)', plot_bgcolor = 'rgba(0, 0, 0, 0.03)')
    fig1.add_trace(go.Scatter(x=data_ss['date'],y=data_ss['negative_score'],name = 'Negative Score'))
    fig1.add_trace(go.Scatter(x=data_ss['date'],y=data_ss['positive_score'],name = 'Positive Score'))
    fig1.add_trace(go.Scatter(x=data_ss['date'],y=data_ss['score'],name = 'Total Score'))
    fig1.update_layout( xaxis_title='Date',yaxis_title='Activity')
    return fig1

# Social Media Error
@st.cache(allow_output_mutation=True)
def get_fig2(df_main):
    model_LSTM, X_test_LSTM, y_test_LSTM, df_index = get_LSTM_data(df_main)

    y_pred_LSTM = model_LSTM.predict(X_test_LSTM)
    y_live = y_pred_LSTM[-1]

    y_pred_LSTM = pd.DataFrame(y_pred_LSTM)
    y_test_LSTM = pd.DataFrame(y_test_LSTM)
    y_pred_LSTM.index = y_test_LSTM.index
    #creating database
    y_pred_LSTM = y_pred_LSTM.rename(columns={ 0: 'y_pred'})
    y_test = y_test_LSTM.rename(columns={ 0: 'y_test'})

    df_pred = pd.concat([y_test, y_pred_LSTM], axis=1)
    df_pred['diff'] = df_pred['y_test'] - df_pred['y_pred']
    df_pred = df_pred.reset_index()

    fig2 = go.Figure()
    fig2.update_layout(autosize=False,width=width,height=height, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor = 'rgba(0, 0, 0, 0)', plot_bgcolor = 'rgba(0, 0, 0, 0.03)')
    fig2.add_trace(go.Scatter(x=df_index.index, y=df_pred['y_test'], name = 'Real SPY-ETF Price' ))
    fig2.add_trace(go.Scatter(x=df_index.index,y=df_pred['y_pred'],name = 'Predicted SPY-ETF Price'))
    fig2.add_trace(go.Bar(x=df_index.index,y=df_pred['diff'],name = 'prediction error',marker = {'color' : 'green'}))
    fig2.update_layout(xaxis_title='Date',yaxis_title='SPY-ETF Price')
    return fig2

# Heatmap
@st.cache(allow_output_mutation=True)
def get_fig3():
    fig3, ax = plt.subplots()
    fig3.set_size_inches([15,11])

    corr = df_main.corr()
    cmap = sns.cubehelix_palette(as_cmap=True, rot=-.4, light=.9)
    #cmap = sns.cubehelix_palette(as_cmap=True, start=2.8, rot=.1, light=.9)
    sns.heatmap(corr, cmap=cmap, mask=corr.isnull(), linecolor='w', linewidths=0.5)
    return fig3

# Accurate Prediciton
@st.cache(allow_output_mutation=True)
def get_fig4(df_pred):

    # plotting a chart
    fig4 = go.Figure()
    fig4.update_layout(autosize=False,width=width,height=height, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor = 'rgba(0, 0, 0, 0)', plot_bgcolor = 'rgba(0, 0, 0, 0.03)')
    fig4.add_trace(go.Scatter(x=df_pred['date'], y=df_pred['testing set'], name = 'Real SPY-ETF Price' ))
    fig4.add_trace(go.Scatter(x=df_pred['date'],y=df_pred['prediction'],name = 'Predicted SPY-ETF Price'))
    fig4.update_layout(title='',xaxis_title='Date',yaxis_title='SPY-ETF Price')
    return fig4

data, SPY_live, SPY_open = get_live_price()
SPY_ratio = round((SPY_live - SPY_open),2)

col1.markdown('''
            # Socio-Vestor
            ''')
col2.metric("SPDR S&P 500", f"{SPY_live} $", f"{SPY_ratio} $")
st.markdown('''
            ### Predicting the Stock Market Using Social Sentiment
            ###
            ''')

#genre = st.radio("",('Prediction of the SPDR S&P 500 ETF', 'Social Media Sentiment', 'Social Media Error', 'Heatmap','Live Prediction of the SPY'))


#if genre == 'Social Media Sentiment':

st.markdown('''## Social Media Sentiment''')
fig1 = get_fig1()
st.plotly_chart(fig1)

#if genre == 'Social Media Error':

st.markdown('''# Social Media Error''')

fig2 = get_fig2(df_main)
st.plotly_chart(fig2)

#if genre == 'Heatmap':
st.markdown('''
        ## Heatmap - Feature Selection
        ##
        ##
        ''')
fig3 = get_fig3()
st.pyplot(fig3)

#if genre == 'Prediction of the SPDR S&P 500 ETF':

st.markdown('''
        ## Accurate Prediction of the SPDR S&P 500 ETF
        ''')
fig4 = get_fig4(df_pred)
st.plotly_chart(fig4)

#if genre == 'Live Prediction of the SPY':

st.markdown('''
            # Live Prediction of the SPY Price
            ''')
fig5 = go.Figure()
fig5.update_layout(autosize=False,width=width-105,height=height, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor = 'rgba(0, 0, 0, 0)', plot_bgcolor = 'rgba(0, 0, 0, 0.03)')
fig5.add_trace(go.Scatter(x=data.index,y=data['Open'],name = 'Real SPY-ETF Price'))
fig5.add_hline(y=y_live, line_width=3, line_dash="dash", line_color="red")
fig5.update_layout(title='Stock Price vs. Prediction',xaxis_title='Date',yaxis_title='SPY-ETF Price')

st.plotly_chart(fig5)
