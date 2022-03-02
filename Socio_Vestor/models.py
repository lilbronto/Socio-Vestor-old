from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers, metrics
from keras.layers import Dense, SimpleRNN
import numpy as np
import pandas as pd

from Socio_Vestor.data import get_intraday_data
from Socio_Vestor.preprocessing import clean_data

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

    def build_simple_rnn(self):

        metric = metrics.MAPE
        opt = optimizers.RMSprop(learning_rate=0.01)

        model = Sequential()
        model.add(SimpleRNN(20, activation='relu', input_shape=(12,1)))
        model.add(Dense(10, activation="relu"))
        model.add(layers.Dense(1, activation="linear"))

        model.compile(loss='mse',
                    optimizer=opt,
                    metrics=[metric])
        return model

    def train_rnn(self, model, X_train, y_train):
        es = EarlyStopping(monitor='val_loss', verbose=1, patience=20, restore_best_weights=True)
        model = self.build_simple_rnn()
        history = model.fit(X_train, y_train,
                        validation_split=0.2,
                        batch_size=8,
                        epochs=300,
                        callbacks=[es], verbose=1)
        return history
