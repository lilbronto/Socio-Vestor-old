import streamlit as st
import numpy as np
import datetime
import joblib

from Socio_Vestor.data import get_main_df
from Socio_Vestor.preprocessing import ff_imputer

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
data = get_main_df()
X_pred = data.loc[start_date:]
# preprocess the data?
X_pred_imp = ff_imputer(X_pred)
X_test = X_pred_imp.drop(['price_high', 'price_low', 'price_close']) # drop the columns we did not use

# load the trained model
model = joblib.load('sociovestor.joblib')

# calculate a prediction
y_pred = model.predict(X_test)
y_pred_array = np.array(np.reshape(y_pred,y_pred.shape[0]))

# Plot the predicted values on streamlit website
st.line_chart(y_pred_array)
