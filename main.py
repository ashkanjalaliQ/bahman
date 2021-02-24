import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image
import yfinance as yf
import matplotlib.pyplot as plt
import Translations.language_names as lang_names
import Translations.translations as translates
import data.symbols_data as symbols_data
from scipy.stats import linregress
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

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


class TrendLine:
    def __init__(self, history):
        self.history = history

    def __get_data(self):
        date = st.selectbox(f'{translates.translate[lang_name]["Month"]}/{translates.translate[lang_name]["Months"]}: ', [i for i in range(1, 13)])
        days_of_month = 30

        return date, days_of_month
    def Draw(self):
        date, days_of_month = self.__get_data()
        self.history = self.history.tail(int(date) * days_of_month)
        data0 = self.history.copy()
        data0['date_id'] = ((data0.index.date - data0.index.date.min())).astype('timedelta64[D]')
        data0['date_id'] = data0['date_id'].dt.days + 1
        data1 = data0.copy()

        while len(data1) > 3:
            reg = linregress(x=data1['date_id'], y=data1['High'])
            data1 = data1.loc[data1['High'] > reg[0] * data1['date_id'] + reg[1]]

        reg = linregress(x=data1['date_id'], y=data1['High'])

        data0['high_trend'] = reg[0] * data0['date_id'] + reg[1]

        data1 = data0.copy()

        while len(data1) > 3:
            reg = linregress(x=data1['date_id'],y=data1['Low'],)
            data1 = data1.loc[data1['Low'] < reg[0] * data1['date_id'] + reg[1]]

        reg = linregress(x=data1['date_id'], y=data1['Low'],)

        data0['low_trend'] = reg[0] * data0['date_id'] + reg[1]

        self.__plot(data0)

    def __plot(self, data):
        plt.figure(figsize=(16, 8))
        data['Close'].plot()
        data['high_trend'].plot()
        data['low_trend'].plot()
        plt.show()
        st.pyplot()


class SMA:
    def __init__(self, history, symbol):
        self.history = history
        self.symbol = symbol

    def __get_signals(self, data, moving_average):
        signals = {
            'Buy': [],
            'Sell': []
        }
        f = -1
        for i in range(len(data)):
            if moving_average['30']['MA'][i] > moving_average['100']['MA'][i]:
                if f != 1:
                    signals['Buy'].append(data[self.symbol][i])
                    signals['Sell'].append(np.nan)
                    f = 1
                else:
                    signals['Buy'].append(np.nan)
                    signals['Sell'].append(np.nan)
            elif moving_average['30']['MA'][i] < moving_average['100']['MA'][i]:
                if f != 0:
                    signals['Buy'].append(np.nan)
                    signals['Sell'].append(data[self.symbol][i])
                    f = 0
                else:
                    signals['Buy'].append(np.nan)
                    signals['Sell'].append(np.nan)
            else:
                signals['Buy'].append(np.nan)
                signals['Sell'].append(np.nan)

        return signals['Buy'], signals['Sell']

    def Draw(self):
        self.moving_avarage = {
            '30': pd.DataFrame(),
            '100': pd.DataFrame()
        }
        for day in self.moving_avarage.keys():
            self.moving_avarage[day]['MA'] = self.history['Close'].rolling(window=int(day)).mean()

        self.data = pd.DataFrame()
        self.data[self.symbol] = self.history['Close']
        self.data['buy signal'], self.data['sell signal'] = self.__get_signals(self.data, self.moving_avarage)

        self.__plot()

    def __plot(self):
        plt.figure(figsize=(16, 8))
        plt.plot(self.data[self.symbol], label=self.symbol, alpha=0.3)
        plt.plot(self.moving_avarage['30']['MA'], label='MA30', alpha=0.3)
        plt.plot(self.moving_avarage['100']['MA'], label="MA100", alpha=0.3)
        plt.scatter(self.data.index, self.data['buy signal'], label='BUY', marker='^', color='g')
        plt.scatter(self.data.index, self.data['sell signal'], label='SELL', marker='v', color='r')
        plt.title('Two Moving Average Indicator')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.show()
        st.pyplot()


class MACD:
    def __init__(self, history):
        self.history = history

    def __get_signals(self, history):
        signals = {
            'Buy': [],
            'Sell': []
        }
        f = -1
        for i in range(0, len(history)):
            if history['MACD'][i] > history['signal line'][i]:
                signals['Sell'].append(np.nan)
                if f != 1:
                    signals['Buy'].append(history['Close'][i])
                    f = 1
                else:
                    signals['Buy'].append(np.nan)
            elif history['MACD'][i] < history['signal line'][i]:
                signals['Buy'].append(np.nan)
                if f != 0:
                    signals['Sell'].append(history['Close'][i])
                    f = 0
                else:
                    signals['Sell'].append(np.nan)

            else:
                signals['Buy'].append(np.nan)
                signals['Sell'].append(np.nan)

        return signals['Buy'], signals['Sell']

    def Draw(self):
        self.history = self.history.tail(220)
        shortEMA = self.history.Close.ewm(span=12, adjust=False).mean()
        longEMA = self.history.Close.ewm(span=26, adjust=False).mean()
        MACD = shortEMA - longEMA
        signal = MACD.ewm(span=9, adjust=False).mean()

        plt.plot(MACD, label='MACD')
        plt.plot(signal, label='Signal')
        plt.legend()
        plt.show()
        st.pyplot()

        self.history['MACD'], self.history['signal line'] = MACD, signal
        self.history['buy signal'], self.history['sell signal'] = self.__get_signals(self.history)

        self.__plot()

    def __plot(self):
        plt.figure(figsize=(16, 8))
        plt.scatter(self.history.index, self.history['buy signal'], color='green', label='BUY', marker='^')
        plt.scatter(self.history.index, self.history['sell signal'], color='red', label='SELL', marker='v')
        plt.plot(self.history['Close'], label='Price', alpha=0.5)
        plt.title('MACD INDICATOR')
        plt.xlabel('DATE')
        plt.ylabel('Indicator')
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()
        st.pyplot()


class DeepLearn:
    def __init__(self):
        pass

    def filter_data(self):
        pass


lang_name = st.radio('Languages', list(lang_names.languages.values()))

st.write(f'''
# {translates.translate[lang_name]['main_title']}
''')
st.header(f'{translates.translate[lang_name]["Insert_Data"]}')
symbol = st.selectbox(f'{translates.translate[lang_name]["Symbol"]}: ', symbols_data.SYMBOLS)
prediction_days = int(st.text_input(f'{translates.translate[lang_name]["Prediction_days"]}: ', 5))
st.write(f"## {translates.translate[lang_name]['Date_Range']}")
year = st.slider(
    f'{translates.translate[lang_name]["Select_a_range_of_Date"]}',
    2000, 2021
)

st.write(f'{translates.translate[lang_name]["From"]}', year, f'{translates.translate[lang_name]["until_now"]}')
period = st.selectbox(f'{translates.translate[lang_name]["Period"]}: ', ['1d', '1w', '1m'])
month = day = 1
date_range = f"{year}-0{month}-0{day}"

finance = Finance(symbol)
history = finance.get_history(date_range, period)

st.header(f'{translates.translate[lang_name]["Current_price"]}: ')
st.info(history.tail(1).Close)

prediction = Prediction(prediction_days, history)
close_price, predict_price, history = prediction.get_close_price()

plt.figure(figsize=(16, 8))
plt.plot(close_price)
plt.show()
st.pyplot()

## SVM
svm_prediction = SVM_Prediction(close_price, predict_price, history, prediction_days)
st.header(f'{translates.translate[lang_name]["SVM_Accuracy"]}')
st.success(svm_prediction.get_accuracy())

st.header(f'{translates.translate[lang_name]["SVM_Prediction"]}')
st.success(str(svm_prediction.get_prediction()))

## LR
lr_prediction = LR_Prediction(svm_prediction)
st.header(f'{translates.translate[lang_name]["LR_Accuracy"]}')
st.success(lr_prediction.get_accuracy())

st.header(f'{translates.translate[lang_name]["LR_Prediction"]}')
st.success(lr_prediction.get_prediction())
st.write(lr_prediction.get_prediction())

## TrendLine
st.header(f'{translates.translate[lang_name]["TrendLines"]}: ')
trendline = TrendLine(finance.get_history(date_range, period))
trendline.Draw()

## SMA
st.header(f'{translates.translate[lang_name]["SMA"]}: ')
sma = SMA(history, symbol)
sma.Draw()

## MACD
st.header(f'{translates.translate[lang_name]["MACD"]}: ')
macd = MACD(history)
macd.Draw()