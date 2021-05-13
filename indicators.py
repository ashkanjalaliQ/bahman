import numpy as np
import pandas as pd
from scipy.stats import linregress


class TrendLine:
    def __init__(self, history):
        self.history = history
        self.days_of_month = 30
        self.date = 1

        #return self.Get()

    #def __get_data(self):
        #date = st.selectbox(f'{translates.translate[lang_name]["Month"]}/{translates.translate[lang_name]["Months"]}: ', [i for i in range(1, 13)])
        #return date
    def Get(self):
        #date = self.__get_data()
        self.history = self.history.tail(int(self.date) * self.days_of_month)
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

        #self.__plot(self.history)
        return self.history

    '''def __plot(self, data):
        plt.figure(figsize=(16, 8))
        data['Close'].plot()
        data['high_trend'].plot()
        st.write(data['high_trend'])
        data['low_trend'].plot()
        st.write(data['low_trend'])
        st.pyplot()'''



class SMA:
    def __init__(self, history):
        self.history = history

        #return self.Get()

    def __get_signals(self, data, moving_average):
        signals = {
            'Buy': [],
            'Sell': []
        }
        f = -1
        for i in range(len(data)):
            if moving_average['30']['MA'][i] > moving_average['100']['MA'][i]:
                if f != 1:
                    signals['Buy'].append(data['symbol'][i])
                    signals['Sell'].append(np.nan)
                    f = 1
                else:
                    signals['Buy'].append(np.nan)
                    signals['Sell'].append(np.nan)
            elif moving_average['30']['MA'][i] < moving_average['100']['MA'][i]:
                if f != 0:
                    signals['Buy'].append(np.nan)
                    signals['Sell'].append(data['symbol'][i])
                    f = 0
                else:
                    signals['Buy'].append(np.nan)
                    signals['Sell'].append(np.nan)
            else:
                signals['Buy'].append(np.nan)
                signals['Sell'].append(np.nan)

        return signals['Buy'], signals['Sell']

    def __calculate_moving_average(self, history):
        moving_average = {
            '30': pd.DataFrame(),
            '100': pd.DataFrame()
        }
        for day in moving_average.keys():
            moving_average[day]['MA'] = history['Close'].rolling(window=int(day)).mean()

        return moving_average

    def Get(self):
        self.moving_average = self.__calculate_moving_average(self.history)

        self.data = pd.DataFrame()
        self.data['symbol'] = self.history['Close']
        self.data['buy signal'], self.data['sell signal'] = self.__get_signals(self.data, self.moving_average)

        return self.data['symbol'], self.data['buy signal'], self.data['sell signal']

    '''def __plot(self):
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
        st.pyplot()'''





class MACD:
    def __init__(self, history):
        self.history = history

        #return self.Get()

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

    def histogram(self, history):
        difference_line = []
        for i in range(len(history)):
            difference_line.append(history['MACD Line'][i] - history['Signal Line'][i])
        return difference_line

    def __calculate_macd(self, history, days=365):
        history = history.tail(days)
        shortEMA = history.Close.ewm(span=12, adjust=False).mean()
        longEMA = history.Close.ewm(span=26, adjust=False).mean()
        MACD = shortEMA - longEMA
        Signal = MACD.ewm(span=9, adjust=False).mean()

        return MACD, Signal, history

    def Get(self):
        MACD, Signal, self.history = self.__calculate_macd(self.history)

        self.history['MACD Line'], self.history['Signal Line'] = MACD, Signal
        self.history['Histogram'] = self.histogram(self.history)
        self.history['Buy Signal'], self.history['Sell Signal'] = self.__get_signals(self.history)

        return self.history['MACD Line'], self.history['Signal Line'], self.history['Histogram'], self.history['Buy Signal'], self.history['Sell Signal']

        '''plt.plot(self.history['MACD Line'], label='MACD')
        plt.plot(self.history['Signal Line'], label='Signal')
        plt.plot(self.history['Histogram'], label='Histogram')
        plt.legend()
        plt.show()
        st.pyplot()

        self.__plot()'''



    '''def __plot(self):
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
        st.pyplot()'''


class RSI:
    def __init__(self, history):
        self.history = history

        #return self.Get()

    def __calculate_rsi(self, history, days=100, period=14):
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

    def Get(self):
        return self.__calculate_rsi(self.history)

        #self.__plot()

    '''def __plot(self):
        plt.plot(self.history['RSI'], label='RSI')
        plt.axhline(30, linestyle='--', color='red', alpha=0.5)
        plt.axhline(70, linestyle='--', color='red', alpha=0.5)
        plt.show()
        st.pyplot()'''



