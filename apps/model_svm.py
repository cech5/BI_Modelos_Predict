from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
  
# For data manipulation
import pandas as pd
import numpy as np
import yfinance as yf
  
# To plot
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
  
# To ignore warnings
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
#import talib


def app():
    st.title('Modelo - SVM')

    start = '2017-07-22'
    end = '2022-07-22'

    st.title('Predicción de tendencia de acciones')

    user_input = st.text_input('Introducir cotización bursátil', 'GC=F')

    stock_data = yf.download(user_input, start, end)

    # Describiendo los datos

    st.subheader('Datos del 2017 al 2022')
    st.write(stock_data)

    # Visualizaciones
    # Create predictor variables
    stock_data['Open-Close'] = stock_data.Open - stock_data.Close
    stock_data['High-Low'] = stock_data.High - stock_data.Low
      
    # Store all predictor variables in a variable X
    X = stock_data[['Open-Close', 'High-Low']]
    st.write(X.head())

    y = np.where(stock_data['Close'].shift(-1) > stock_data['Close'], 1, 0)
    st.write(y)

    split_percentage = 0.8
    split = int(split_percentage*len(stock_data))
      
    # Train data set
    X_train = X[:split]
    y_train = y[:split]
      
    # Test data set
    X_test = X[split:]
    y_test = y[split:]

    cls = SVC().fit(X_train, y_train)

    stock_data['Predicted_Signal'] = cls.predict(X)
    st.write(stock_data['Predicted_Signal'])

    stock_data['Return'] = stock_data.Close.pct_change()
    st.write(stock_data['Return'])

    # Calculate strategy returns
    stock_data['Strategy_Return'] = stock_data.Return *stock_data.Predicted_Signal.shift(1)
    st.write(stock_data['Strategy_Return'])

    # Calculate Cumulutive returns
    stock_data['Cum_Ret'] = stock_data['Return'].cumsum()
    
    stock_data['Cum_Strategy'] = stock_data['Strategy_Return'].cumsum()

    import matplotlib.pyplot as plt
      
    fig = plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Cum_Ret'],color='red')
    plt.plot(stock_data['Cum_Strategy'],color='blue')
    st.pyplot(fig)
      