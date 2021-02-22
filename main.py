import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image
import yfinance as yf
import matplotlib.pyplot as plt
import symbols_data
from scipy.stats import linregress

plt.style.use('fivethirtyeight')
st.set_option('deprecation.showPyplotGlobalUse', False)

class Finance:
    def __init__(self, symbol):
        self.symbol = symbol

    def get_history(self, date_range, period):
        try:
            history = yf.Ticker(self.symbol)
            history = history.history(period=period, start=date_range)
            history['Date'] = history.index
            history = history.set_index(pd.DatetimeIndex(history['Date'].values))
        except:
            ## IRAN STOCK
            pass
        return history

    def get_symbol_name(self):
        return self.symbol


class Prediction:
    def __init__(self, forecast, history):
        self.forecast = int(forecast)
        self.history = history

    def get_close_price(self):
        self.history = self.history[['Close']]
        self.history['Prediction'] = self.history[['Close']].shift(-self.forecast)
        close_price = np.array(self.history.drop(['Prediction'], 1))
        close_price = close_price[:-self.forecast]
        predict_price = np.array(self.history['Prediction'])
        predict_price = predict_price[:-self.forecast]

        return close_price, predict_price, self.history


class SVM_Prediction:
    def __init__(self, close_price, predict_price, history, forecast):
        self.predict_price = predict_price
        self.close_price = close_price
        self.history = history
        self.forecast = forecast

    def __train_test_split(self, close_price, predict_price, test_size=0.2):
        return train_test_split(close_price, predict_price, test_size=test_size)
    def get_accuracy(self, test_size = 0.2):
        #self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.close_price, self.predict_price, test_size=test_size)
        self.xtrain, self.xtest, self.ytrain, self.ytest = self.__train_test_split(close_price, predict_price)
        self.svr = SVR(kernel='rbf', C=1000, gamma=0.1)
        self.svr.fit(self.xtrain, self.ytrain)
        svmconf = self.svr.score(self.xtest, self.ytest)

        return svmconf

    def get_prediction(self):
        self.x_forecast = np.array(self.history.drop(['Prediction'], 1))[-self.forecast:]
        svmpred = self.svr.predict(self.x_forecast)

        return svmpred

class LR_Prediction:
    def __init__(self, svm_prediction):
        self.lr = LinearRegression()
        self.svm_prediction = svm_prediction

    def get_accuracy(self):
        self.lr.fit(self.svm_prediction.xtrain, self.svm_prediction.ytrain)
        lrconf = self.lr.score(self.svm_prediction.xtest, self.svm_prediction.ytest)

        return lrconf

    def get_prediction(self):
        return self.lr.predict(self.svm_prediction.x_forecast)


st.write('''
# Stock market software(SMS)
''')
st.header('Insert Data')
symbol = st.selectbox('Symbol: ', symbols_data.SYMBOLS)
prediction_days = int(st.text_input('Prediction days: ', 5))
st.write("## Date Range")
year = st.slider(
    'Select a range of Date',
    2000, 2021, (2015, 2021)
)
period = st.selectbox('Period: ', ['1d', '1w', '1m'])
month = day = 1
date_range = f"{year[0]}-0{month}-0{day}"

finance = Finance(symbol)
history = finance.get_history(date_range, period)

st.header('Current price: ')
st.info(history.tail(1).Close)

prediction = Prediction(prediction_days, history)
close_price, predict_price, history = prediction.get_close_price()

## SVM
svm_prediction = SVM_Prediction(close_price, predict_price, history, prediction_days)
st.header('SVM Accuracy')
st.success(svm_prediction.get_accuracy())

st.header('SVM Prediction')
st.success(str(svm_prediction.get_prediction()))

## LR
lr_prediction = LR_Prediction(svm_prediction)
st.header('LR Accuracy')
st.success(lr_prediction.get_accuracy())

st.header('LR Prediction')
st.success(lr_prediction.get_prediction())
st.write(lr_prediction.get_prediction())