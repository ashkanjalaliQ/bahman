
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import streamlit as st
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

plt.style.use('fivethirtyeight')
st.set_option('deprecation.showPyplotGlobalUse', False)

class Finance:
    def __init__(self, symbol):
        self.symbol = symbol

    def get_history(self, date_range, period):
        try:
            ## IRAN STOCK
            stock_data = Stock_Data(symbol_name=self.symbol)
            history = stock_data.get_data()

        except:
            history = yf.Ticker(self.symbol)
            history = history.history(period=period, start=date_range)
            history['Date'] = history.index
            history = history.set_index(pd.DatetimeIndex(history['Date'].values))

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
        self.days_of_month = 30

    def __get_data(self):
        date = st.selectbox(f'{translates.translate[lang_name]["Month"]}/{translates.translate[lang_name]["Months"]}: ', [i for i in range(1, 13)])
        return date
    def Draw(self):
        date = self.__get_data()
        self.history = self.history.tail(int(date) * self.days_of_month)
        self.history = self.history.copy()
        self.history['date_id'] = ((self.history.index.date - self.history.index.date.min())).astype('timedelta64[D]')
        self.history['date_id'] = self.history['date_id'].dt.days + 1
        history_copy = self.history.copy()

        while len(history_copy) > 3:
            reg = linregress(x=history_copy['date_id'], y=history_copy['Close'])
            history_copy = history_copy.loc[history_copy['Close'] > reg[0] * history_copy['date_id'] + reg[1]]

        reg = linregress(x=history_copy['date_id'], y=history_copy['Close'])

        self.history['high_trend'] = reg[0] * self.history['date_id'] + reg[1]

        history_copy = self.history.copy()

        while len(history_copy) > 3:
            reg = linregress(x=history_copy['date_id'],y=history_copy['Close'],)
            history_copy = history_copy.loc[history_copy['Close'] < reg[0] * history_copy['date_id'] + reg[1]]

        reg = linregress(x=history_copy['date_id'], y=history_copy['Close'])

        self.history['low_trend'] = reg[0] * self.history['date_id'] + reg[1]

        #return self.history

        self.__plot(self.history)

    def __plot(self, data):
        plt.figure(figsize=(16, 8))
        data['Close'].plot()
        data['high_trend'].plot()
        st.write(data['high_trend'])
        data['low_trend'].plot()
        st.write(data['low_trend'])
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

    def Draw(self):
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



class DeepLearn:
    def __init__(self, history):
        plt.style.use('fivethirtyeight')
        self.history = history

    def filter_by(self, history, key):
        return history.filter([key])

    def get_values(self, data, beginning=False, end=False):
        if beginning or end:
            if not end:
                return data[beginning:].values
            if not beginning:
                return data[:end].values
            return data[beginning:end].values
        return data.values

    def get_train_data_len(self):
        return math.ceil(len(self.close_price)*0.8)

    def get_train_data(self, scaled_data, t=True):
        ## Bad naming
        if t:
            return scaled_data[:self.get_train_data_len() , :]
        return self.close_data[:self.get_train_data_len()]

    def get_scaler(self):
        return MinMaxScaler(feature_range=(0, 1))

    def get_scaled_data(self):
        return self.scaler.fit_transform(self.close_price)

    def fill_trains(self, xtrain, ytrain):
        for i in range(self.n, len(self.train_data)):
            xtrain.append(self.train_data[i - self.n:i, 0])
            ytrain.append(self.train_data[i, 0])
        return self.list_to_np(xtrain, 3, reshape=True), self.list_to_np(ytrain, 4)

    def reshape_np_array(self, array, m):
        try:
            return np.reshape(array, (array.shape[0], array.shape[1], 1))
        except:
            print(m)
            print(array)

    def list_to_np(self, ls, m, reshape=False):
        ls = np.array(ls)
        if reshape:
            #return self.reshape_np_array(ls)
            ls = self.reshape_np_array(ls, m)
        return ls

    def create_dl_leyer(self, compile=True, fit=True):
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(self.xtrain.shape[1], 1)))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))

        if compile:
            self.compile_model()

        if fit:
            self.fit_model()

        return self.model

    def compile_model(self):
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def fit_model(self, epochs=5, batch_size=1):
        self.model.fit(self.xtrain, self.ytrain, epochs=epochs, batch_size=batch_size)

    def predict_model(self, xtest, inverse_transform=True):
        prediction = self.model.predict(xtest)
        if inverse_transform:
            prediction = self.inverse_transform_scaler(self.scaler, prediction)
        return prediction

    def inverse_transform_scaler(self, scaler, prediction):
        return scaler.inverse_transform(prediction)

    def get_test_data(self):
        scaled_data = self.get_scaled_data()
        return scaled_data[self.get_train_data_len() - self.n:, :]

    def fill_tests(self, xtest_reshape=True):
        xtest = []
        ytest = self.close_price[self.get_train_data_len():, :]
        for i in range(self.n, len(self.get_test_data())):
            xtest.append(self.get_test_data()[i - self.n: i, 0])

        if xtest_reshape:
            xtest = self.list_to_np(xtest, 1, reshape=True)
        return xtest, ytest

    def transform_scaler(self, data):
        return self.scaler.transform(data)

    def prediction(self):
        self.close_data = self.filter_by(self.history, 'Close')
        self.close_price = self.get_values(self.close_data)
        self.scaler = self.get_scaler()
        scaled_data = self.get_scaled_data()
        self.train_data = self.get_train_data(scaled_data)

        self.xtrain = []
        self.ytrain = []
        self.n = 60
        self.xtrain, self.ytrain = self.fill_trains(self.xtrain, self.ytrain)

        self.model = self.create_dl_leyer()

        self.xtest, self.ytest = self.fill_tests()

        self.predict = self.predict_model(self.xtest)

        self.train_data = self.get_train_data(self.get_scaled_data(), t=False)

        self.valid = self.close_data[self.get_train_data_len():]
        self.valid['Prediction'] = self.predict

        plt.figure(figsize=(16, 8))
        plt.title('Price Predictor Using DL')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.plot(self.train_data['Close'])
        plt.plot(self.valid[['Close', 'Prediction']])
        plt.legend(['Train', 'Value', 'Prediction'])
        st.pyplot()
        plt.show()


        self.dataset = self.get_values(self.close_data, beginning=-60, end=0)
        self.dataset = self.transform_scaler(self.dataset)

        self.xtest = []
        self.xtest.append(self.dataset)

        self.xtest = self.list_to_np(self.xtest, 2, reshape=True)

        self.predict = self.predict_model(self.xtest, inverse_transform=True)

        return self.predict


lang_name = st.radio('Languages', list(lang_names.languages.values()))

st.write(f'''
# {translates.translate[lang_name]['main_title']}
''')
st.header(f'{translates.translate[lang_name]["Insert_Data"]}')
print([USStock] + [list(IRStock.keys())])
symbol = st.selectbox(f'{translates.translate[lang_name]["Symbol"]}: ', [USStock.SYMBOLS][0] + [list(IRStock.keys())][0])
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

## RSI
st.header(f'{translates.translate[lang_name]["RSI"]}:')
rsi = RSI(history)
rsi.Draw()

## DeepLearning
st.header(f'{translates.translate[lang_name]["Deep_Learning"]}:')
dl = DeepLearn(history)
predict = dl.prediction()
st.header(f'{translates.translate[lang_name]["Deep_Learning_Prediction"]}: ')
#st.success(predict)
if float(history['Close'].tolist()[-1]) < float(predict):
    st.success(predict)
else:
    st.error(predict)
st.success(f'Today: {history["Close"].tolist()[-1]}')
st.success(macd.history['Signal Line'])
'''
from finance import Finance
from gui import GUI
import streamlit as st
from indicators import TrendLine
from indicators import SMA
from indicators import MACD
from indicators import RSI
from predictions import LR_Prediction
from predictions import SVM_Prediction
from predictions import DeepLearn
from predictions import Prediction


class Core(Finance):
    def __init__(self):
        self.Export_data = {
            'Current_price': None,
            'Close_Price': None,
            'Indicators': {
                'MACD': {
                    'MACD': {
                        'MACD': None,
                        'Signal': None
                    },
                    'Signals': {
                        'Buy': None,
                        'Sell': None
                    },
                    'Histo': None
                },
                'RSI': None,
                'TrendLine': None,
                'SMA': {
                    'SMA': None,
                    'Signals': {
                        'Buy': None,
                        'Sell': None
                    }
                }
            },
            'Predictions': {
                'LR': {
                    'Prediction': None,
                    'Accuracy': None
                },
                'SVM': {
                    'Prediction': None,
                    'Accuracy': None
                },
                'DeepLearning': {
                    'next_day': None,
                    'train_test': None
                }
            }
        }

    def Get_Inputs(self):
        self.GUI = GUI()
        self.symbol, self.prediction_days, self.year, self.period, self.date_range = self.GUI.Get_basic_information()
        self.history = self.get_history(date_range=self.date_range, period=self.period, symbol=self.symbol)
        #self.history = self.finance(self.date_range, self.period)

        #self.Indicators()

    def Indicators(self):
        self.Export_data['Indicators']['TrendLine'] = TrendLine(self.history).Get()
        self.Export_data['Indicators']['MACD']['MACD']['MACD'], self.Export_data['Indicators']['MACD']['MACD']['Signal'], \
            self.Export_data['Indicators']['MACD']['Histo'], self.Export_data['Indicators']['MACD']['Signals']['Buy'], \
                self.Export_data['Indicators']['MACD']['Signals']['Sell']= MACD(self.history).Get()
        self.Export_data['Indicators']['RSI'] = RSI(self.history).Get()
        self.Export_data['Indicators']['SMA']['SMA'], self.Export_data['Indicators']['SMA']['Signals']['Buy'], self.Export_data['Indicators']['SMA']['Signals']['Sell'] = SMA(self.history).Get()

        #self.Predictions()

    def Predictions(self):
        prediction = Prediction(self.prediction_days, self.history)
        self.close_price, self.predict_price, self.history = prediction.get_close_price()
        self.svm_prediction = SVM_Prediction(self.close_price, self.predict_price, self.history, self.prediction_days)
        self.Export_data['Predictions']['SVM']['Accuracy'] = self.svm_prediction.get_accuracy()
        self.Export_data['Predictions']['SVM']['Prediction'] = self.svm_prediction.get_prediction()
        self.lr_prediction = LR_Prediction(self.svm_prediction)
        self.Export_data['Predictions']['LR']['Prediction'] = self.lr_prediction.get_prediction()
        self.Export_data['Predictions']['LR']['Accuracy'] = self.lr_prediction.get_accuracy()

        #return self.Export_data
    def Get_Export(self):
        return self.Export_data

core = Core()
core.Get_Inputs()
core.Indicators()
core.Predictions()
print(core.Export_data())'''
