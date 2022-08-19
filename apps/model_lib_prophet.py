# Machine learning
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
  
# For data manipulation
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
# To plot
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
  
# To ignore warnings
import warnings
warnings.filterwarnings("ignore")


def app():
    st.title('Modelo - Basado en la librería Prophet')

    start = '2017-07-22'
    end = '2022-07-22'

    st.title('Predicción de tendencia de acciones')

    user_input = st.text_input('Introducir cotización bursátil', 'GC=F')

    stock_data = yf.download(user_input, start, end)

    # Describiendo los datos

    st.subheader('Datos del 2017 al 2022')
    st.write(stock_data)

    stock_data = stock_data.reset_index()

    # Visualizaciones
    st.subheader('Date y price')
    # Renonbrar
    stock_data= stock_data.rename(columns={'Date':'ds', 'Close':'y'})
    st.write(stock_data)

    st.subheader('Predicción Prophet')
    fbp = Prophet(daily_seasonality= True)
    fbp.fit(stock_data)
    future = fbp.make_future_dataframe(periods=365)
    forecast = fbp.predict(future)

    st.write(plot_plotly(fbp, forecast))

    