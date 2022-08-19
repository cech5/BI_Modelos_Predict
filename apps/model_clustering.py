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

  # st.header('Trazamos un gráfico de velas para los primeros 60 días de Amazon se ha trazado a continuación.')
  # fig = go.Figure()
  # fig.add_trace(go.Candlestick(x=df.index,
  #   open=df.iloc[0:60]['Open']['AMZN'],
  #   high=df.iloc[0:60]['High']['AMZN'],
  #   low=df.iloc[0:60]['Low']['AMZN'],
  #   close=df.iloc[0:60]['Close']['AMZN']))
  
  st.header('Necesidad de normalización')
  st.subheader('Trazamos la variación del "movimiento" de Amazon y Apple.')
  fig = plt.figure(figsize = (20,8)) 
  ax1 = plt.subplot(1,2,1)
  plt.title('Company:Amazon',fontsize = 20)
  plt.xticks(fontsize = 18)
  plt.yticks(fontsize = 20)
  plt.xlabel('Date',fontsize = 20)
  plt.ylabel('Movement',fontsize = 20)
  plt.plot(movements[0]) 
  st.pyplot(fig)
  plt.subplot(1,2,2,sharey = ax1)
  plt.title('Company:Apple',fontsize = 20)
  plt.xticks(fontsize = 18)
  plt.yticks(fontsize = 20)
  plt.xlabel('Date',fontsize = 20)
  plt.ylabel('Movement',fontsize = 20)
  plt.plot(movements[1])
  st.pyplot(fig)

  st.write('Los precios de las acciones de Amazon y Apple tienen diferentes escalas. Por lo tanto, x unidades de cambio en el precio de las acciones de Amazon no es lo mismo que x unidades de cambio en el precio de las acciones de Apple. Algunas empresas valen mucho más que otras empresas. Por lo tanto, los datos tienen que ser normalizados.')
  st.write('"norm_movements" se define como una versión normalizada de "movements". El Normalizer() vuelve a escalar cada fila de "movimientos" de forma independiente.')
  st.write('Verificamos si el conjunto de datos contiene valores NaN.  Si emparejamos esta función any (), comprobaremos si hay instancias de NaN. Podemos reemplazar los valores de NaN usando el nan_to_num ()método. ')
  np. any( np. isnan ( movements ))
  movements = np. nan_to_num ( movements )
  st.write(movements)
  st.write('"norm_movements" se define como una versión normalizada de "movements". El Normalizer() vuelve a escalar cada fila de "movimientos" de forma independiente.')
  normalizer = Normalizer() # Define a Normalizer
  norm_movements = normalizer.fit_transform(movements) # Fit and transform

  st.subheader('Se imprime el valor mínimo, máximo y medio de "norm_movements".')
  st.write(norm_movements.min())
  st.write(norm_movements.max())
  st.write(norm_movements.mean())

  st.write('Los valores mínimo, máximo y medio de "norm_movements" son -0,259, 0,26 y 0,001. Todos los valores están en el rango (-1,1) y la media es cercana a 0.')
  st.write('Volvemos a Graficar la variación de "norm_movements" de Amazon y Apple.')

  fig = plt.figure(figsize = (20,8)) 
  ax1 = plt.subplot(1,2,1)
  plt.title('Company:Amazon',fontsize = 20)
  plt.xticks(fontsize = 18)
  plt.yticks(fontsize = 20)
  plt.xlabel('Date',fontsize = 20)
  plt.ylabel('Movement',fontsize = 20)
  plt.plot(movements[0]) 
  st.pyplot(fig)
  plt.subplot(1,2,2,sharey = ax1)
  plt.title('Company:Apple',fontsize = 20)
  plt.xticks(fontsize = 18)
  plt.yticks(fontsize = 20)
  plt.xlabel('Date',fontsize = 20)
  plt.ylabel('Movement',fontsize = 20)
  plt.plot(movements[1])
  st.pyplot(fig)

  st.subheader('Interpretación')
  st.write('Los movimientos normalizados de Amazon y Apple están en un rango similar.')

  st.header('Creando una pipeline')
  st.write('Proceso comprendido en varias fases secuenciales, siendo cada salida la entrada del anterior, sin perder datos y conocimiento.')
  st.write('Ahora creamos un pipiline que normaliza los datos y luego aplica el algoritmo de agrupamiento K-Means para agrupar empresas con movimientos bursátiles diarios similares.')

  # Import the necessary packages
  from sklearn.pipeline import make_pipeline
  
  from sklearn.cluster import KMeans
  # Define a normalizer
  normalizer = Normalizer()
  # Create Kmeans model
  kmeans = KMeans(n_clusters = 10,max_iter = 1000)
  # Cree un normalizador de encadenamiento de tuberías y kmeans
  pipeline = make_pipeline(normalizer,kmeans)
  # Ajuste pipeline a los movimientos diarios de existencias
  pipeline.fit(movements)
  labels = pipeline.predict(movements)

  st.write('Mostramos las empresas y su número de clúster')
  df1 = pd.DataFrame({'labels':labels,'companies':list(companies_dict)}).sort_values(by=['labels'],axis = 0)
  st.write(df1)

  st.header('Reducción de PCA')
  st.write('A continuación se muestra la canalización (pipeline) que normaliza, reduce (reducción de PCA) y aplica el algoritmo de agrupamiento K-Means.')
  
  
  from sklearn.decomposition import PCA
  # Define a normalizer
  normalizer = Normalizer()
  # Reduce the data
  reduced_data = PCA(n_components = 2)
  # Create Kmeans model
  kmeans = KMeans(n_clusters = 10,max_iter = 1000)
  # Cree un normalizador de encadenamiento de tuberías, pca y kmeans
  pipeline = make_pipeline(normalizer,reduced_data,kmeans)
  # Adaptar la canalización a los movimientos diarios de existencias
  pipeline.fit(movements)
  # Prediction
  labels = pipeline.predict(movements)
  # Crear marco de datos para almacenar empresas y etiquetas predichas
  df2 = pd.DataFrame({'labels':labels,'companies':list(companies_dict.keys())}).sort_values(by=['labels'],axis = 0)
  st.write('Ahora echemos un vistazo a las empresas agrupadas.')

  st.write(df2)

  st.write('La formación de grupos con reducción de PCA es diferente de la formación de grupos sin reducción de PCA. La desventaja de la reducción de PCA es que se pierden algunos detalles. Los resultados no son muy precisos. Las ventajas de la reducción de PCA son menos poder computacional y fácil visualización.')
  st.write('- El límite de decisión se representa a continuación.')

  from sklearn.decomposition import PCA
  # Reduce the data
  reduced_data = PCA(n_components = 2).fit_transform(norm_movements)
  # Definir el tamaño de paso de la malla
  h = 0.01
  # Trazar el límite de decisión
  x_min,x_max = reduced_data[:,0].min()-1, reduced_data[:,0].max() + 1
  y_min,y_max = reduced_data[:,1].min()-1, reduced_data[:,1].max() + 1
  xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
  # Obtenga etiquetas para cada punto de la malla usando nuestro modelo entrenado
  Z = kmeans.predict(np.c_[xx.ravel(),yy.ravel()])
  # Poner el resultado en un diagrama de color.
  Z = Z.reshape(xx.shape)
  # Define color plot
  cmap = plt.cm.Paired
  # Figura de trazado
  plt.clf()
  fig = plt.figure(figsize=(10,10))
  plt.imshow(Z,interpolation = 'nearest',extent=(xx.min(),xx.max(),yy.min(),yy.max()),cmap = cmap,aspect = 'auto',origin = 'lower')
  plt.plot(reduced_data[:,0],reduced_data[:,1],'k.',markersize = 5)
  st.pyplot(fig)
  # Trace el centroide de cada grupo como una X blanca
  centroids = kmeans.cluster_centers_
  plt.scatter(centroids[:,0],centroids[:,1],marker = 'x',s = 169,linewidths = 3,color = 'w',zorder = 10)
  plt.title('Agrupación de K-Means en los movimientos del mercado de valores (PCA-Datos reducidos)')
  plt.xlim(x_min,x_max)
  plt.ylim(y_min,y_max)
  plt.show()
  st.pyplot(fig)

  st.subheader('Interpretación')
  st.write('Cada punto negro representa una empresa. Cada cruz blanca representa el centroide del grupo respectivo.')


