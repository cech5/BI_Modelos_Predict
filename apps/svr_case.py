import numpy as np
from sklearn.svm import SVR 
import matplotlib.pyplot as plt 
import pandas as pd 
import streamlit as st
import yfinance as yf





def app():
  st.title('Learning Data Science — Predict Stock Price with Support Vector Regression (SVR)')
  start = '2019-01-01'
  end = '2019-12-31'
  st.subheader('Preparación de la data')
  user_input = st.text_input('Introducir cotización bursátil', 'TD.TO')

  df = yf.download(user_input, start, end)
  df = df.reset_index()
  #Usaremos la información de Toronto-Dominion Bank (TD.TO), correspondiente a la fecha del 01-01-2019 al 30-01-2019
  st.write(df)

  # Obtenemos toda la data con la siguiente función: get_data()
  def get_data(df):  
    data = df.copy()
    data['Date'] = data['Date'].astype(str).str.split('-').str[2]
    data['Date'] = pd.to_numeric(data['Date'])
    return [ data['Date'].tolist(), data['Close'].tolist() ] # Convierte series a listas (tolist)
  dates, prices = get_data(df)

  # Función para predecir
  def predict_prices(dates, prices, x):
    dates = np.reshape(dates,(len(dates), 1)) # Convierte a dimensión 1 x n 
    x = np.reshape(x,(len(x), 1))
    
    svr_lin  = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    
    # Ajustar modelo de regresión
    svr_lin .fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)
    
    # Función para graficar 
    

    plt.scatter(dates, prices, c='k', label='Data')
    plt.plot(dates, svr_lin.predict(dates), c='g', label='Linear model')
    plt.plot(dates, svr_rbf.predict(dates), c='r', label='RBF model')    
    plt.plot(dates, svr_poly.predict(dates), c='b', label='Polynomial model')
    

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    
    
    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

  st.subheader('Obtener la predicción y mostrar los modelos predictivos')
  predicted_price = predict_prices(dates, prices, [31])
  st.area_chart(predicted_price)