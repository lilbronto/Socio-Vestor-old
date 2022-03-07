import streamlit as st
import numpy as np
import datetime
import joblib
import pandas as pd

from Socio_Vestor.data import get_intraday_data, get_main_df
from Socio_Vestor.preprocessing import clean_data, ff_imputer

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
data = get_intraday_data()
data = data.loc[start_date:]
df_cleaned = clean_data(data)
# preprocess the data?
#X_pred_imp = ff_imputer(X_pred)
#X_test = X_pred_imp['price_open'] # drop the columns we did not use
data = pd.DataFrame(df_cleaned['open'])
for i in range(1, 13):
    data[f't - {i}'] = data['open'].shift(i)
data.dropna(inplace=True)
X_test = data.drop(['open'], axis=1)
X_test = np.expand_dims(X_test, axis=2)
# load the trained model
model = joblib.load('sociovestor.joblib')

# calculate a prediction
y_pred = model.predict(X_test)
y_pred_array = np.array(np.reshape(y_pred,y_pred.shape[0]))

# Plot the predicted values on streamlit website
st.line_chart(y_pred_array)
