from pandas_datareader import data 
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
import streamlit as st
from plotly import graph_objs as go
from sklearn.preprocessing import Normalizer
from sklearn.cluster import AffinityPropagation


def app():
  st.title('Modelo - Clustering')

  companies_dict = {
      'Amazon':'AMZN',
      'Apple':'AAPL',
      'Walgreen':'WBA',
      'Northrop Grumman':'NOC',
      'Boeing':'BA',
      'Lockheed Martin':'LMT',
      'McDonalds':'MCD',
      'Intel':'INTC',
      'Navistar':'NAV',
      'IBM':'IBM',
      'Texas Instruments':'TXN',
      'MasterCard':'MA',
      'Microsoft':'MSFT',
      'General Electrics':'GE',
      'Symantec':'SYMC',
      'American Express':'AXP',
      'Pepsi':'PEP',
      'Coca Cola':'KO',
      'Johnson & Johnson':'JNJ',
      'Toyota':'TM',
      'Honda':'HMC',
      'Mistubishi':'MSBHY',
      'Sony':'SNE',
      'Exxon':'XOM',
      'Chevron':'CVX',
      'Valero Energy':'VLO',
      'Ford':'F',
      'Bank of America':'BAC'
    }
  st.write('Definimos un diccionario donde clave es el nombre de la empresa y valor es el código de acciones de la empresa')
  st.write(companies_dict)
  st.write('Extraemos los datos de mercado desde Yahoo Finance desde "2015-01-01" a "2017-12-31". El movimiento de acciones de las empresas se compararía en función de 6 parámetros: - "High", "Low", "Open", "Close", "Volume", "Adj Close".')
  st.write('‘High’ :- Precio más alto durante el día.')
  st.write('‘Low’ :- Precio más bajo durante el día.')
  st.write('‘Open’ :- Precio de apertura del día.')
  st.write('‘Close’ :- Precio de cierre del día.')
  st.write('‘Volume’ :- Número total de acciones negociadas durante el día.')
  st.write('‘Adj Close’ :- El precio de cierre se modifica para tener en cuenta cualquier acción corporativa para dar el ‘Adjusted closing’ precio.')

  data_source = 'yahoo' # Source of data is yahoo finance.
  start_date = '2015-1-1' 
  end_date = '2017-12-31'
  df = data.DataReader(list(companies_dict.values()),
  data_source,start_date,end_date)

  st.write(df)
  stock_open = np.array(df['Open']).T # stock_open es una matriz numpy de transposición de df['Open']
  stock_close = np.array(df['Close']).T # stock_close es una matriz numpy de transposición de df['Close']
  movements = stock_close - stock_open
  st.write('‘sum_of_movement’ de una empresa se define como la suma de las diferencias de precios de cierre y apertura de todos los días.')
  sum_of_movement = np.sum(movements,1)

  for i in range(len(companies_dict)):
    st.write('company:{}, Change:{}'.format(df['High'].columns[i],sum_of_movement[i]))
  
  st.header('Visualización de los datos')
  st.subheader('Variación de los precios de apertura de las empresas Amazon y Apple')
  fig = plt.figure(figsize = (20,10)) 
  plt.subplot(1,2,1) 
  plt.title('Company:Amazon',fontsize = 20)
  plt.xticks(fontsize = 10)
  plt.yticks(fontsize = 20)
  plt.xlabel('Date',fontsize = 15)
  plt.ylabel('Opening price',fontsize = 15)
  plt.plot(df['Open']['AMZN'])
  st.pyplot(fig)

  fig = plt.figure(figsize = (20,10)) 
  plt.subplot(1,2,2) 
  plt.title('Company:Apple',fontsize = 20)
  plt.xticks(fontsize = 10)
  plt.yticks(fontsize = 20)
  plt.xlabel('Date',fontsize = 15)
  plt.ylabel('Opening price',fontsize = 15)
  plt.plot(df['Open']['AAPL'])
  st.pyplot(fig)

  st.subheader('Interpretación')
  st.write('Observamos que el precio de apertura de Amazon aumentó de 200 a 1200, mientras que el precio de apertura de Apple aumentó de 110 a 180. La grafica nos muestra que Amazon tiene un aumento constante relativamente mejor al de Apple. Por lo tanto, Amazon tiene mejor crecimiento en ese periodo de tiempo')

  st.header('Trazamos los precios de apertura y cierre de Amazon en el período de tiempo de "2015-01-02" a "2015-01-23".')

  fig = plt.figure(figsize = (20,10)) # Adjusting figure size
  plt.title('Company:Amazon',fontsize = 20)
  plt.xticks(fontsize = 10)
  plt.yticks(fontsize = 20)
  plt.xlabel('Date',fontsize = 20)
  plt.ylabel('Price',fontsize = 20)
  plt.plot(df.iloc[0:30]['Open']['AMZN'],label = 'Open') # Opening prices of first 30 days are plotted against date
  plt.plot(df.iloc[0:30]['Close']['AMZN'],label = 'Close') # Closing prices of first 30 days are plotted against date
  plt.legend(loc='upper left', frameon=False,framealpha=1,prop={'size': 22}) # Properties of legend box
  st.pyplot(fig)

  st.subheader('Interpretación')
  st.write('Observamos que existe un aumento general en los precios de "Apertura" y "Cierre" durante el período de 30 días, lo que muestra a la empresa de manera positiva.')

  st.header('La variación del "movimiento" de amazon en el período de tiempo de 2015-01-02 a 2015-02-13 se representa a continuación.')
  fig = plt.figure(figsize = (20,8)) 
  plt.title('Company:Amazon',fontsize = 20)
  plt.xticks(fontsize = 18)
  plt.yticks(fontsize = 20)
  plt.xlabel('Date',fontsize = 20)
  plt.ylabel('Movement',fontsize = 20)
  plt.plot(movements[0][0:30])
  st.pyplot(fig)

  st.write('Es deseable un "movimiento" positivo que sugiera que el precio ha aumentado durante el día.')

  st.header('El "Volumen" de acciones negociadas en el período de tiempo de "2015-01-022 a "2017-12-31" se ha representado a continuación.')
  fig = plt.figure(figsize = (20,10)) 
  plt.title('Company:Amazon',fontsize = 20)
  plt.xticks(fontsize = 18)
  plt.yticks(fontsize = 20)
  plt.xlabel('Date',fontsize = 20)
  plt.ylabel('Volume',fontsize = 20)
  plt.plot(df['Volume']['AMZN'],label = 'Open')
  st.pyplot(fig)
  st.subheader('Interpretación')
  st.write('Los picos sugieren que hay grandes volúmenes de acciones negociadas en ciertos días.')

  st.header('Trazamos un gráfico de velas para los primeros 60 días de Amazon se ha trazado a continuación.')
  fig = go.Figure()
  fig.add_trace(go.Candlestick(x=df.index,
    open=df.iloc[0:60]['Open']['AMZN'],
    high=df.iloc[0:60]['High']['AMZN'],
    low=df.iloc[0:60]['Low']['AMZN'],
    close=df.iloc[0:60]['Close']['AMZN']))
  st.plotly_chart(fig, use_container_width = True)

