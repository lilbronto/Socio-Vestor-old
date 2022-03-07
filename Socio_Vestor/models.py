from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers, metrics
from keras.layers import Dense, SimpleRNN
import numpy as np
import pandas as pd

from Socio_Vestor.data import get_intraday_data, get_main_df
from Socio_Vestor.preprocessing import clean_data, scale

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

        X_scaled = scale(X)

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
