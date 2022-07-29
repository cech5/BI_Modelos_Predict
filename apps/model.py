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
    st.title('Modelo - Random Forest')

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
    stock_data['Adj Close'].plot()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Adj Close'])
    plt.ylabel("Adjusted Close Prices")
    st.pyplot(fig)

    st.subheader('Cambio porcentual de cierre ajustado de 1 día')
    fig = plt.figure(figsize=(12, 6))
    plt.hist(stock_data['Adj Close'].pct_change(), bins=50)
    plt.ylabel("Frecuencia")
    plt.xlabel("Cambio porcentual de cierre ajustado de 1 día")
    st.pyplot(fig)

    st.subheader('RSI')
    df = stock_data
    delta = df['Close'].diff()
    up = delta.clip(lower = 0)
    down = -1*delta.clip(upper=0)

    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()

    rs = ema_up/ema_down
    df['RSI'] = 100 - (100/(1+rs))
    df
    st.write(df)

    st.subheader('MACD')
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 -exp2
    df['Signal line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df=df.dropna()
    st.write(df.head())


    st.subheader('Variables predictoras')
    # Crear variables Predictoras
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low

    # Almacenar todas las variables Predictoras en una variable feature_input (que será nuestro X) 
    feature_input = df[['Open-Close', 'High-Low', 'RSI', 'MACD']]
    st.write(feature_input)

    st.subheader('Variable objetivo')
    # Variable Objetivo target (que es nuestro y)
    # Comparamos el Precio de cierre anterior con el actual
    # Almacenamos:
    st.write('1: Señal de subida próxima del precio (señal de compra)')
    st.write('0: Señal de bajada próxima del precio (no comprar, vender si se tiene)')
    #   1: Señal de subida próxima del precio (señal de compra)
    #   0: señal de bajada próxima del precio (no comprar, vender si se tiene)
    target = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    st.write(target)

    train_input, test_input, train_target, test_target = train_test_split(feature_input, target, test_size=0.2, random_state=9999)
    ##Normalize/Standardize the features before applying Machine Learning Models
    sc = StandardScaler()
    train_input = sc.fit_transform(train_input)
    test_input = sc.transform(test_input)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import KFold

    forest = RandomForestClassifier(n_estimators=10000, min_samples_split=5,
                                random_state=999, min_samples_leaf=5,
                                n_jobs=-1)

    forest.fit(train_input, train_target)
    #print(forest.score(X_test, y_test))

    #K-fold cross validation code 
    kfold = KFold(n_splits = 5, random_state=None)

    scores = []
    for train_index, test_index in kfold.split(train_input):
      X_train_kf, X_test_kf = train_input[train_index], train_input[test_index]
      y_train_kf, y_test_kf = train_target[train_index], train_target[test_index]
      forest.fit(X_train_kf, y_train_kf)
      score = forest.score(X_test_kf, y_test_kf)
      scores.append(score)
      st.write('Acc: %.3f' % (score))
    st.write('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

    from sklearn.metrics import recall_score

    st.subheader("La precisión del modelo Random Forest es: {:>7.4f}".format(forest.score(test_input, test_target)))
    pred_rf = forest.predict(test_input)
    st.subheader("La sensibilidad del modelo es: {:>7.4f}".format(recall_score(test_target, pred_rf)))
    '''
    feature_names = []
    for n in [14, 30, 50, 200]:
        stock_data['ma' +
                   str(n)] = talib.SMA(stock_data['Adj Close'].values, timeperiod=n)
        stock_data['rsi' +
                   str(n)] = talib.RSI(stock_data['Adj Close'].values, timeperiod=n)
        feature_names = feature_names + ['ma' + str(n), 'rsi' + str(n)]
    stock_data['Volume_1d_change'] = stock_data['Volume'].pct_change()
    volume_features = ['Volume_1d_change']
    feature_names.extend(volume_features)
    stock_data['5d_future_close'] = stock_data['Adj Close'].shift(-5)
    stock_data['5d_close_future_pct'] = stock_data['5d_future_close'].pct_change(
        5)
    stock_data.dropna(inplace=True)
    X = stock_data[feature_names]
    y = stock_data['5d_close_future_pct']
    train_size = int(0.85 * y.shape[0])
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    grid = {'n_estimators': [200], 'max_depth': [3],
            'max_features': [4, 8], 'random_state': [42]}
    test_scores = []
    rf_model = RandomForestRegressor()
    for g in ParameterGrid(grid):
        rf_model.set_params(**g)
        rf_model.fit(X_train, y_train)
        test_scores.append(rf_model.score(X_test, y_test))
    best_index = np.argmax(test_scores)
    rf_model = RandomForestRegressor(
        n_estimators=200, max_depth=3, max_features=4, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    st.subheader('Porcentaje de cambio de precio de cierre previsto de 5 días')
    y_pred_series = pd.Series(y_pred, index=y_test.index)
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_pred_series, 'r',
             label='Porcentaje de cambio de precio de cierre previsto de 5 días')
    plt.ylabel("Porcentaje de cambio de precio de cierre previsto de 5 días")
    plt.xlabel('Date')
    plt.legend()
    st.pyplot(fig2)
'''