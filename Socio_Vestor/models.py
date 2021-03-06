from re import S
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers, metrics
from keras.layers import SimpleRNN
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.metrics import MeanSquaredError, LogCoshError

import numpy as np
import pandas as pd

from Socio_Vestor.data import get_intraday_data, get_main_df
from Socio_Vestor.preprocessing import SRNN_imputer, df_trend, linearize_df, s_scaler, clean_data, ff_imputer, minmax_scaler, standard_scaler

class SimpleRnn():

    def __init__(self):
        pass

    def get_data(self):
        df = get_intraday_data()
        # clean data
        df_cleaned = clean_data(df)
        # reduce size
        #df_reduced = df_optimized(df_cleaned)
        # set X and y
        data = pd.DataFrame(df_cleaned['open'])
        for i in range(1, 13):
            data[f't - {i}'] = data['open'].shift(i)
        data.dropna(inplace=True)
        X = data.drop(['open'], axis=1)
        y = data['open']
        # hold out
        train_size = 0.7
        index = round(train_size*X.shape[0])
        X_train = X.iloc[:index]
        y_train = y.iloc[:index]
        X_test = X.iloc[index:]
        y_test = y.iloc[index:]
        # expand dimension
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)
        return X_train, X_test, y_train, y_test

    def build_simple_rnn(self, shape):

        metric = metrics.MAPE
        opt = optimizers.RMSprop(learning_rate=0.01)

        model = Sequential()
        model.add(SimpleRNN(32, activation='relu', input_shape=(shape[1],shape[2])))
        model.add(Dense(10, activation="relu"))
        model.add(layers.Dense(1, activation="linear"))

        model.compile(loss='mse',
                    optimizer=opt,
                    metrics=[metric])
        return model

    def train_rnn(self, X_train, y_train, epochs=500):
        es = EarlyStopping(monitor='val_loss', verbose=1, patience=20, restore_best_weights=True)
        self.model = self.build_simple_rnn(X_train.shape)
        self.model.fit(X_train, y_train,
                        validation_split=0.2,
                        batch_size=8,
                        epochs=epochs,
                        callbacks=[es], verbose=1)
        return self.model


class LSTM():

    def __init__(self):
        pass

    def get_data(self):
        df_main = get_main_df()
        df_main.replace(np.nan,-42069,inplace=True)
        X = df_main.drop(['price_open', 'price_high', 'price_low', 'price_close'], axis=1)
        y = df_main[['price_open', 'price_high', 'price_low', 'price_close']]

        X_scaled = standard_scaler(X)

        train_size = 0.6
        index = round(train_size*X_scaled.shape[0])

        X_train = X_scaled.iloc[:index]
        X_test = X_scaled.iloc[index:]

        X_train_array = np.array(X_train).astype(np.float32)
        X_test_array = np.array(X_test).astype(np.float32)

        X_train_array = np.expand_dims(X_train_array, 1)
        X_test_array = np.expand_dims(X_test_array, 1)

        y_open_price = pd.DataFrame(y['price_open'])
        y_train = y_open_price.iloc[:index]
        y_test = y_open_price.iloc[index:]
        y_train_array = np.array(y_train['price_open']).astype(np.float32)
        y_test_array = np.array(y_test['price_open']).astype(np.float32)

        return X_train_array, X_test_array, y_train_array, y_test_array

    def build_LSTM(self):

        # Padding Layer (func)
        model = Sequential()
        model.add(layers.Masking(mask_value=-42069, input_shape=(1,5)))
        model.add(layers.LSTM(units=20, activation='tanh'))
        model.add(layers.Dense(1, activation="linear"))

        model.compile(loss='mse',
                      optimizer='adam',
                      metrics='accuracy')
        return model

    def train_LSTM(self, X_train_array, y_train_array, epochs=150):

        es = EarlyStopping(monitor='val_loss', verbose=1, patience=20, restore_best_weights=True)
        self.model = self.build_LSTM()
        self.model.fit(X_train_array, y_train_array,
                       batch_size=16,
                       epochs=epochs,
                       validation_split=0.2,
                       callbacks=[es],
                       verbose=1)
        return self.model

class LayerLSTM():

    def __init__(self):
        pass

    def get_data(self, x=30):
        df_main = get_main_df()
        print(df_main.shape)
        df_main_imp = ff_imputer(df_main)
        print(df_main_imp.shape)
        df_temp = df_main_imp[['price_open', 'weighted_ss']]
        print(df_temp.shape)
        mm_scaler, df_scaled = minmax_scaler(df_temp)
        print(df_scaled.shape)

        X_train = []
        y_train = []

        index = round(df_scaled.shape[0]*0.7)
        for i in range(x, index):
            X_train.append(df_scaled[i-x:i,:])
            y_train.append(df_scaled[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)

        X_test = []
        y_test = []
        for i in range(index, df_scaled.shape[0]):
            X_test.append(df_scaled[i-x:i,:])
            y_test.append(df_scaled[i, 0])

        X_test, y_test = np.array(X_test), np.array(y_test)

        return X_train, X_test, y_train, y_test

    def build_LSTM(self):

        model = Sequential()
        model.add(layers.LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))
        model.add(layers.LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))
        model.add(layers.LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))
        model.add(layers.LSTM(units = 50))
        model.add(Dropout(0.2))
        model.add(layers.Dense(units = 1))
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')

        model.compile(loss='mse',
                    optimizer='adam',
                    metrics='accuracy')
        return model

    def train_LSTM(self, X_train, y_train, epochs=500):
        es = EarlyStopping(monitor='val_loss', verbose=1, patience=15, restore_best_weights=True)
        self.model = self.build_LSTM()
        self.model.fit(X_train, y_train,
                        batch_size=32,
                        epochs=epochs,
                        validation_split=0.2,
                        callbacks=[es],
                        verbose=1)
        return self.model

class SimpleRNN_main():
    def __init__(self):
        pass

    def get_data(self):
        df_main = get_main_df()
        df_trend_df = df_trend(df_main)

        df_main = pd.concat([df_main, df_trend_df], axis=1)

        X = df_main[['real_gdp', 'cpi', 'MACD_Signal', 'MACD', 'MACD_Hist', 'trend_int']]
        y = df_main['price_close']

        y = pd.DataFrame(y)

        X_imp = SRNN_imputer(X)
        X_lin = linearize_df(X_imp)
        X_scaled = s_scaler(X_lin)

        train_size = 0.8
        index = round(train_size*X.shape[0])
        X_train = X_scaled.iloc[:index]
        y_train = y.iloc[:index]
        X_test = X_scaled.iloc[index:]
        y_test = y.iloc[index:]

        y_train = y_train.fillna(value=-2000)
        X_train = X_train.fillna(value=-2000)

        print(X_train, y_train, X_test, y_test)

        return X_train, y_train, X_test, y_test

    def Simple_RNN(self):

        metrics = MeanSquaredError(), LogCoshError()

        model = Sequential()
        model.add(layers.Masking(mask_value=-2000, input_shape=(5,1)))
        model.add(SimpleRNN(32, activation='relu'))
        model.add(Dense(10, activation="relu"))
        model.add(layers.Dense(1, activation="linear"))

        model.compile(loss='mse',
                    optimizer='adam',
                    metrics= [metrics])
        return model

    def train_SimpleRNN(self, X_train, y_train, epochs=500):
        es = EarlyStopping(monitor='val_loss', verbose=1, patience=20, restore_best_weights=True)

        self.model = self.Simple_RNN()
        self.model.fit(X_train, y_train,
                    validation_split=0.2,
                    batch_size=64,
                    epochs=100,
                    callbacks=[es], verbose=1)
        return self.model
