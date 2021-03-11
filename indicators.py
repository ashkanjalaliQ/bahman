import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image
import yfinance as yf
from pytse.download import Stock_Data
from pytse.symbols_data import SYMBOLS as IRStock
import matplotlib.pyplot as plt
import Translations.language_names as lang_names
import Translations.translations as translates
import data.symbols_data as USStock
from scipy.stats import linregress
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


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

    def calculate_moving_average(self, history):
        moving_average = {
            '30': pd.DataFrame(),
            '100': pd.DataFrame()
        }
        for day in moving_average.keys():
            moving_average[day]['MA'] = history['Close'].rolling(window=int(day)).mean()

        return moving_average

    def get_data(self):
        self.moving_average = self.calculate_moving_average(self.history)

        self.data = pd.DataFrame()
        self.data[self.symbol] = self.history['Close']
        self.data['buy signal'], self.data['sell signal'] = self.__get_signals(self.data, self.moving_average)

        self.__plot()

    def __plot(self):
        plt.figure(figsize=(16, 8))
        plt.plot(self.data[self.symbol], label=self.symbol, alpha=0.3)
        plt.plot(self.moving_average['30']['MA'], label='MA30', alpha=0.3)
        plt.plot(self.moving_average['100']['MA'], label="MA100", alpha=0.3)
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
        for i in range(len(history)):
            if history['MACD Line'][i] > history['Signal Line'][i]:
                signals['Sell'].append(np.nan)
                if f != 1:
                    signals['Buy'].append(history['Close'][i])
                    f = 1
                else:
                    signals['Buy'].append(np.nan)
            elif history['MACD Line'][i] < history['Signal Line'][i]:
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

    def histogram(self):
        difference_line = []
        for i in range(len(self.history)):
            difference_line.append(self.history['MACD Line'][i] - self.history['Signal Line'][i])
        return difference_line

    def calculate_macd(self, history, days=365):
        history = history.tail(days)
        shortEMA = history.Close.ewm(span=12, adjust=False).mean()
        longEMA = history.Close.ewm(span=26, adjust=False).mean()
        MACD = shortEMA - longEMA
        Signal = MACD.ewm(span=9, adjust=False).mean()

        return MACD, Signal, history

    def Draw(self):
        MACD, Signal, self.history = self.calculate_macd(self.history)



        self.history['MACD Line'], self.history['Signal Line'] = MACD, Signal
        self.history['Histogram'] = self.histogram()
        self.history['Buy Signal'], self.history['Sell Signal'] = self.__get_signals(self.history)

        plt.plot(self.history['MACD Line'], label='MACD')
        plt.plot(self.history['Signal Line'], label='Signal')
        plt.plot(self.history['Histogram'], label='Histogram')
        plt.legend()
        plt.show()
        st.pyplot()

        self.__plot()



    def __plot(self):
        plt.figure(figsize=(16, 8))
        plt.scatter(self.history.index, self.history['Buy Signal'], color='green', label='BUY', marker='^')
        plt.scatter(self.history.index, self.history['Sell Signal'], color='red', label='SELL', marker='v')
        plt.plot(self.history['Close'], label='Price', alpha=0.5)
        plt.title('MACD INDICATOR')
        plt.xlabel('DATE')
        plt.ylabel('Indicator')
        #plt.xticks(rotation=45)
        plt.legend()
        plt.show()
        st.pyplot()


class RSI:
    def __init__(self, history):
        self.history = history

    def calculate_rsi(self, history, days=100, period=14):
        history = history.tail(days)
        history = history['Close'].diff(1)
        history.dropna()
        up = history.copy()
        down = history.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        avg_gain = up.rolling(window=period).mean()
        avg_loss = abs(down.rolling(window=period).mean())
        RSI = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
        return RSI

    def Draw(self):
        self.history['RSI'] = self.calculate_rsi(self.history)
        self.__plot()

    def __plot(self):
        plt.plot(self.history['RSI'], label='RSI')
        plt.axhline(30, linestyle='--', color='red', alpha=0.5)
        plt.axhline(70, linestyle='--', color='red', alpha=0.5)
        plt.show()
        st.pyplot()


class Plot:
    def __init__(self):
        pass

    def ploting(self, data, scatter=None, title=None, xlabel=None, ylabel=None, figsize=(16, 8), figure=True, axhline=None):
        '''
        type data: list
        type scatter: dict
        type title: str
        type xlabel: str
        type ylabel: str

        data:
        [
            {
                list:[...],
                label: '...',
                alpha: ...
            }
        ]

        scatter:
        [
            {
                index: [...],
                list: [...],
                color: '...',
                label: '...',
                marker: '...'
            }
        ]
        '''
        if figure:
            plt.figure(figsize=figsize)

        for i in range(len(data)):
            plt.plot(data[i]['list'], label=data[i]['label'], alpha=data[i]['alpha'])

        if scatter != None:
            pass



