import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import datetime
from datetime import timedelta, date
#FB Prophet Part
from fbprophet import Prophet 
import plotly.offline as py

#st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
st.title("Cryptocurrency Price Prediction")



with st.form("my_form"):
    fre = st.radio( "Select the frequency",('Day', 'Month', 'Year'))
    periods = int(st.number_input('Enter the Number of periods ',min_value=0,step=1))
 # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")

if fre=="Day":
    freq='D'
elif fre=='Month':
    freq='M'
else:
    freq='Y'

if submitted:
    request = requests.get('https://min-api.cryptocompare.com/data/histoday?fsym=BTC&tsym=CAD&limit=1000')
    data = pd.DataFrame(json.loads(request.content)['Data'])
    data['time'] = pd.to_datetime(data['time'], unit='s')

    df= data.drop(['high', 'low', 'open', 'volumefrom', 'volumeto','conversionType', 'conversionSymbol'], axis=1)

    df = df.rename(columns={'close': 'y', 'time': 'ds'})
    #df['ds'] =  pd.to_datetime(df['ds'], format='%d/%m/%Y')

    # to save a copy of the original data..you'll see why shortly. 
    df['y_orig'] = df['y'] 
    # log-transform of y
    df['y'] = np.log(df['y'])
    #instantiate Prophet
    model = Prophet() 
    model.fit(df)




    future_data = model.make_future_dataframe(periods=periods, freq = freq)  #dropdown   
    forecast_data = model.predict(future_data)

    # make sure we save the original forecast data
    forecast_data_orig = forecast_data 
    forecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])
    forecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])
    forecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])



    df['y_log']=df['y'] 
    df['y']=df['y_orig']
    final_df = pd.DataFrame(forecast_data_orig)
    actual_chart = go.Scatter(y=df["y_orig"], name= 'Actual')
    predict_chart = go.Scatter(y=final_df["yhat"], name= 'Predicted')
    predict_chart_upper = go.Scatter(y=final_df["yhat_upper"], name= 'Predicted Upper')
    predict_chart_lower = go.Scatter(y=final_df["yhat_lower"], name= 'Predicted Lower')
    #py.plot([actual_chart, predict_chart, predict_chart_upper, predict_chart_lower])
    fig=go.Figure([actual_chart, predict_chart, predict_chart_upper, predict_chart_lower])
    fig.update_layout(width=800,height=600) 
    st.plotly_chart(fig)
