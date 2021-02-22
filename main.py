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


class TrendLine:
    def __init__(self, history):
        self.history = history

    def __get_data(self):
        date = st.selectbox('Month/Months: ', [i for i in range(1, 13)])
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

        plt.figure(figsize=(16, 8))
        data0['Close'].plot()
        data0['high_trend'].plot()
        data0['low_trend'].plot()
        plt.show()
        st.pyplot()


class SMA:
    def __init__(self, history, symbol):
        self.history = history
        self.symbol = symbol

    def get_signals(self, data, moving_average):
        signalBuy = []
        signalSell = []
        f = -1
        for i in range(len(data)):
            if moving_average['30']['MA'][i] > moving_average['100']['MA'][i]:
                if f != 1:
                    signalBuy.append(data[self.symbol][i])
                    signalSell.append(np.nan)
                    f = 1
                else:
                    signalBuy.append(np.nan)
                    signalSell.append(np.nan)
            elif moving_average['30']['MA'][i] < moving_average['100']['MA'][i]:
                if f != 0:
                    signalBuy.append(np.nan)
                    signalSell.append(data[self.symbol][i])
                    f = 0
                else:
                    signalBuy.append(np.nan)
                    signalSell.append(np.nan)
            else:
                signalBuy.append(np.nan)
                signalSell.append(np.nan)

        return signalBuy, signalSell

    def Draw(self):
        moving_avarage = {
            '30': pd.DataFrame(),
            '100': pd.DataFrame()
        }
        for day in moving_avarage.keys():
            moving_avarage[day]['MA'] = self.history['Close'].rolling(window=int(day)).mean()

        data = pd.DataFrame()
        data[self.symbol] = self.history['Close']

        data['buy signal'], data['sell signal'] = self.get_signals(data, moving_avarage)

        plt.figure(figsize=(16, 8))
        plt.plot(data[self.symbol], label=self.symbol, alpha=0.3)
        plt.plot(moving_avarage['30']['MA'], label='MA30', alpha=0.3)
        plt.plot(moving_avarage['100']['MA'], label="MA100", alpha=0.3)
        plt.scatter(data.index, data['buy signal'], label='BUY', marker='^', color='g')
        plt.scatter(data.index, data['sell signal'], label='SELL', marker='v', color='r')
        plt.title('Two Moving Average Indicator')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.show()
        st.pyplot()

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

## TrendLine
st.header('TrendLine: ')
trendline = TrendLine(finance.get_history(date_range, period))
trendline.Draw()

## SMA
st.header('SMA: ')
sma = SMA(history, symbol)
sma.Draw()