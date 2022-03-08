import streamlit as st
import numpy as np
import datetime
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import requests
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

from Socio_Vestor.data import get_intraday_data, get_main_df
from Socio_Vestor.preprocessing import clean_data, ff_imputer, minmax_scaler

st.set_page_config(layout="centered")
col1, col2 = st.columns((5,1))

# Autorefresh the Streamlit page every 10 seconds
st_autorefresh(interval= 60 * 1000, key="dataframerefresh")

def get_latest_price():
    url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=SPY&interval=1min&outputsize=compact&apikey=GA32KX1XU3RE15LO"
    SPY_intra = requests.get(url).json()
    data_SPY_intra = pd.DataFrame(SPY_intra['Time Series (1min)']).transpose()
    data_SPY_intra = data_SPY_intra['1. open']
    return data_SPY_intra[0]

SPY_price = round(float(get_latest_price()))
SPY_ratio = round((float(get_latest_price()) - 418.00),2)

col1.markdown('''
            # Socio-Vestor
            ''')
col2.metric("SPDR S&P 500", f"{SPY_price} $", f"{SPY_ratio} $")
st.markdown('''
            ### Predicting the Stock Market Using Social Sentiment
            - Bullet Points
            ''')

# Get the data and chache it in order to avoid constant reloading
@st.cache(allow_output_mutation=True)
def get_our_data():

    df_main = get_main_df()
    df_main_imp = ff_imputer(df_main)
    df_temp = df_main_imp[['price_open', 'weighted_ss']]
    mm_scaler, df_scaled = minmax_scaler(df_temp)

    index = round(df_scaled.shape[0]*0.7)
    x = 30
    X_test = []
    y_test = []
    for i in range(index, df_scaled.shape[0]):
        X_test.append(df_scaled[i-x:i,:])
        y_test.append(df_scaled[i, 0])

    X_test, y_test = np.array(X_test), np.array(y_test)
    # load the trained model
    model = joblib.load('sociovestor.joblib')
    return model, X_test, y_test, df_main

model, X_test, y_test, df_main = get_our_data()
# # calculate a prediction
y_pred = model.predict(X_test)

st.markdown('''
            # Live Prediction of the SPY Price
            ''')
y_pred_live = 420.69 # Should be y_pred[-1]
data = yf.download(tickers='SPY', period='1d', interval='1m')
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=data.index,y=data['Open'],name = 'Real SPY-ETF Price'))
fig2.add_hline(y=y_pred_live, line_width=3, line_dash="dash", line_color="red")
fig2.update_layout(title='Stock Price vs. Prediction',xaxis_title='Date',yaxis_title='SPY-ETF Price')

st.plotly_chart(fig2)

# st.markdown('''
#             ## Heatmap - Feature Selection
#            ''')

# fig, ax = plt.subplots()
# fig.set_size_inches([10,7])

# corr = df_main.corr()
# cmap = sns.cubehelix_palette(as_cmap=True, rot=-.4, light=.9)
# #cmap = sns.cubehelix_palette(as_cmap=True, start=2.8, rot=.1, light=.9)
# sns.heatmap(corr, cmap=cmap, mask=corr.isnull(), linecolor='w', linewidths=0.5)

# st.pyplot(fig)


st.markdown('''# Social Media Error''')

#creating database
y_pred_n = pd.DataFrame(y_pred)
y_pred_n = y_pred_n.rename(columns={ 0: 'y_pred'})
y_test = pd.DataFrame(y_test)
y_test = y_test.rename(columns={ 0: 'y_test'})

df_pred = pd.concat([y_test, y_pred_n], axis=1)

df_pred['diff'] = df_pred['y_test'] - df_pred['y_pred']

#plotting a chart
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_pred.index, y=df_pred['y_test'], name = 'Real SPY-ETF Price' ))
fig1.add_trace(go.Scatter(x=df_pred.index,y=df_pred['y_pred'],name = 'Predicted SPY-ETF Price'))
fig1.add_trace(go.Bar(x=df_pred.index,y=df_pred['diff'],name = 'prediction error',marker = {'color' : 'green'}))
fig1.update_layout(title='Prediction including social sentiment',xaxis_title='Date',yaxis_title='SPY-ETF Price')

st.plotly_chart(fig1)
