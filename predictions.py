import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

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
        self.xtrain, self.xtest, self.ytrain, self.ytest = self.__train_test_split(self.close_price, self.predict_price)
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

        '''plt.figure(figsize=(16, 8))
        plt.title('Price Predictor Using DL')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.plot(self.train_data['Close'])
        plt.plot(self.valid[['Close', 'Prediction']])
        plt.legend(['Train', 'Value', 'Prediction'])
        st.pyplot()
        plt.show()'''


        self.dataset = self.get_values(self.close_data, beginning=-60, end=0)
        self.dataset = self.transform_scaler(self.dataset)

        self.xtest = []
        self.xtest.append(self.dataset)

        self.xtest = self.list_to_np(self.xtest, 2, reshape=True)

        self.predict = self.predict_model(self.xtest, inverse_transform=True)

        return self.predict