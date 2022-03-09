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
from Socio_Vestor.data import get_social_sentiment_data

from Socio_Vestor.data import get_intraday_data, get_main_df
from Socio_Vestor.preprocessing import clean_data, df_trend, ff_imputer, impute_df, linearize_df, minmax_scaler, s_scaler

st.set_page_config(layout="centered")
col1, col2 = st.columns((5,1))

# Autorefresh the Streamlit page every 10 seconds
#st_autorefresh(interval= 60 * 1000, key="dataframerefresh")

def get_latest_price():
    url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=SPY&interval=1min&outputsize=compact&apikey=GA32KX1XU3RE15LO"
    SPY_intra = requests.get(url).json()
    data_SPY_intra = pd.DataFrame(SPY_intra['Time Series (1min)']).transpose()
    data_SPY_intra = data_SPY_intra['1. open']
    return data_SPY_intra[0]

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
    x = 30
    X_test_LSTM = []
    y_test_LSTM = []
    for i in range(index, df_scaled.shape[0]):
        X_test_LSTM.append(df_scaled[i-x:i,:])
        y_test_LSTM.append(df_scaled[i, 0])

    X_test_LSTM, y_test_LSTM = np.array(X_test_LSTM), np.array(y_test_LSTM)
    # load the trained model
    model_LSTM = joblib.load('sociovestor.joblib')
    return model_LSTM, X_test_LSTM, y_test_LSTM, df_main

def get_RNN_data(df_main):
    # SimpleRNN Model
    df_trend_df = df_trend(df_main)
    df_main = pd.concat([df_main, df_trend_df])

    X = df_main[['real_gdp', 'cpi', 'MACD_Signal', 'MACD', 'MACD_Hist', 'trend_int']]
    y = df_main['price_close']

    y = pd.DataFrame(y)

    X_imp = impute_df(X)
    X_lin = linearize_df(X_imp)
    X_scaled = s_scaler(X_lin)

    train_size = 0.8
    index = round(train_size*X.shape[0])
    X_test = X_scaled.iloc[index:]
    y_test = y.iloc[index:]

    model_SRNN = joblib.load('simplernn_main.joblib')

    return model_SRNN, X_test, y_test

SPY_price = round(float(get_latest_price()))
SPY_ratio = round((float(get_latest_price()) - 418.00),2)

col1.markdown('''
            # Socio-Vestor
            ''')
col2.metric("SPDR S&P 500", f"{SPY_price} $", f"{SPY_ratio} $")
st.markdown('''
            ### Predicting the Stock Market Using Social Sentiment
            ''')

st.markdown('''
            ## Accurate Prediction of the SPDR S&P 500 ETF Trust
           ''')
# Calculate RNN Prediction
model_SRNN, X_test_SRNN, y_test_SRNN,= get_RNN_data(df_main)

y_pred_SRNN = model_SRNN.predict(X_test_SRNN)
y_pred_sc = pd.DataFrame(y_pred_SRNN)
y_pred_sc.index = y_test_SRNN.index
y_pred_sc = y_pred_sc.rename(columns={0: "y_pred"})
y_pred_sc['y_pred'] = y_pred_sc['y_pred']*1.5
y_pred_sc['y_pred'] = y_pred_sc['y_pred']-100
#creating database
y_test_S = y_test_SRNN.rename(columns={ 'price_close' : 'y_test'})

df_pred_SRNN = pd.concat([y_test_S, y_pred_sc], axis=1)

df_pred_SRNN['diff'] = df_pred_SRNN['y_test'] - df_pred_SRNN['y_pred']

#plotting a chart
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_pred_SRNN.index, y=df_pred_SRNN['y_test'], name = 'Real SPY-ETF Price' ))
fig1.add_trace(go.Scatter(x=df_pred_SRNN.index,y=df_pred_SRNN['y_pred'],name = 'Predicted SPY-ETF Price'))
fig1.add_trace(go.Bar(x=df_pred_SRNN.index,y=df_pred_SRNN['diff'],name = 'prediction error',marker = {'color' : 'green'}))
fig1.update_layout(title='Title',xaxis_title='Date',yaxis_title='SPY-ETF Price')

st.plotly_chart(fig1)

st.markdown('''
            ## Heatmap - Feature Selection
           ''')

fig2, ax = plt.subplots()
fig2.set_size_inches([10,7])

corr = df_main.corr()
cmap = sns.cubehelix_palette(as_cmap=True, rot=-.4, light=.9)
#cmap = sns.cubehelix_palette(as_cmap=True, start=2.8, rot=.1, light=.9)
sns.heatmap(corr, cmap=cmap, mask=corr.isnull(), linecolor='w', linewidths=0.5)

st.pyplot(fig2)

st.markdown('''# Social Media Sentiment''')

data_ss = get_social_sentiment_data()
data_ss = data_ss.reset_index()

# Plotting the Social Sentiment
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=data_ss['date'], y=data_ss['weighted_ss'], name = 'Weighted Social Sentiment' ))
st.plotly_chart(fig3)

st.markdown('''# Social Media Error''')

model_LSTM, X_test_LSTM, y_test_LSTM, df_main = get_LSTM_data(df_main)
y_pred_LSTM = model_LSTM.predict(X_test_LSTM)
#creating database
y_pred_n = pd.DataFrame(y_pred_LSTM)
y_pred_n = y_pred_n.rename(columns={ 0: 'y_pred'})
y_test = pd.DataFrame(y_test_LSTM)
y_test = y_test.rename(columns={ 0: 'y_test'})

df_pred = pd.concat([y_test, y_pred_n], axis=1)

df_pred['diff'] = df_pred['y_test'] - df_pred['y_pred']

#plotting a chart
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=df_pred.index, y=df_pred['y_test'], name = 'Real SPY-ETF Price' ))
fig4.add_trace(go.Scatter(x=df_pred.index,y=df_pred['y_pred'],name = 'Predicted SPY-ETF Price'))
fig4.add_trace(go.Bar(x=df_pred.index,y=df_pred['diff'],name = 'prediction error',marker = {'color' : 'green'}))
fig4.update_layout(title='Prediction including social sentiment',xaxis_title='Date',yaxis_title='SPY-ETF Price')

st.plotly_chart(fig4)

st.markdown('''
            # Live Prediction of the SPY Price
            ''')
y_pred_live = 420.69 # Should be y_pred[-1]
data = yf.download(tickers='SPY', period='1d', interval='1m')
fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=data.index,y=data['Open'],name = 'Real SPY-ETF Price'))
fig5.add_hline(y=y_pred_live, line_width=3, line_dash="dash", line_color="red")
fig5.update_layout(title='Stock Price vs. Prediction',xaxis_title='Date',yaxis_title='SPY-ETF Price')

st.plotly_chart(fig5)
