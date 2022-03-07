import streamlit as st
import numpy as np
import datetime
import joblib
import pandas as pd

from Socio_Vestor.data import get_intraday_data, get_main_df
from Socio_Vestor.preprocessing import clean_data, ff_imputer, minmax_scaler

'''
Get the timeframe for the Price data that will be used to make our prediction
'''
st.write(f"Pleas select the timeframe for your prediction")
start_date = st.date_input(
    "Start Date:",
    datetime.date(2022, 3, 3))
# end_date = st.date_input(
#     "End Date:",
#     datetime.date(2022, 3, 11))


'''
Make an API Call to fetch the data we need for our prediction
'''
# Get the data
#data = get_intraday_data()
#df_cleaned = clean_data(data)
# preprocess the data?
#X_pred_imp = ff_imputer(X_pred)

df_main = get_main_df()
df_main_imp = ff_imputer(df_main)
df_temp = df_main_imp[['price_open', 'weighted_ss']]
mm_scaler, df_scaled = minmax_scaler(df_temp)

X_train = []
y_train = []
x=30
index = round(df_scaled.shape[0]*0.7)
for i in range(x, index):
    X_train.append(df_scaled[i-x:i,:])
    y_train.append(df_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_test = []
y_test = []
for i in range(index, df_scaled.shape[0]):
    X_test.append(df_scaled[i-x:i,:])
    y_test.append(df_scaled[i, 0])

X_test = np.array(X_test)
# load the trained model
model = joblib.load('sociovestor.joblib')

# calculate a prediction
y_pred = model.predict(X_test)
y_pred_array = np.array(np.reshape(y_pred,y_pred.shape[0]))

# Plot the predicted values on streamlit website
st.line_chart(y_pred_array)
