from nbformat import write
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from tensorflow.keras.models import load_model
import streamlit as st
import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 
#import talib


def app():
    st.title('Modelo - Regresión Logística')

    start = '2017-07-22'
    end = '2022-07-22'

    st.title('Predicción de tendencia de acciones')

    user_input = st.text_input('Introducir cotización bursátil', 'GC=F')

    stock_data = yf.download(user_input, start, end)

    # Describiendo los datos

    st.subheader('Datos del 2017 al 2022')
    st.write(stock_data)

    # Visualizaciones
    st.subheader('Precio de cierre ajustado')
    stock_data['Adj Close'].pct_change() * 100
    fig = plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Adj Close'])
    plt.ylabel("Adjusted Close Prices")
    st.pyplot(fig)

    df = stock_data['Adj Close'].pct_change() * 100
    df = df.rename("Today")
    st.write(df)
    df = df.reset_index()
    st.write(df)

    for i in range(1,6):
      df['Lag '+str(i)] = df['Today'].shift(i)

    df['Volume'] = stock_data.Volume.shift(1).values/1000
    df = df.dropna() 

    df['Direction'] = [1 if i>0 else 0 for i in df['Today']]

    x_train = df[df.Date.dt.year< 2022][['Lag 1','Lag 2']]
    y_train = df[df.Date.dt.year< 2022]['Direction']
    x_test = df[df.Date.dt.year == 2022][['Lag 1','Lag 2']]
    y_test = df[df.Date.dt.year == 2022]['Direction']

    from sklearn.linear_model import LogisticRegression
    logmodel = LogisticRegression()
    logmodel.fit(x_train,y_train)

    predictions = logmodel.predict(x_test)

    st.subheader('Predicción')
    st.write('Direction')
    st.write(y_test)

    st.write('Lag 1, Lag 2')
    st.write(x_test)

    st.subheader('Predicciones')
    st.write(predictions)


    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test,predictions)

    from sklearn.metrics import classification_report

    st.text(classification_report(y_test,predictions))