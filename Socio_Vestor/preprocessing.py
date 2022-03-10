import pandas as pd
import numpy as np
import math

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from statsmodels.tsa.seasonal import seasonal_decompose


def df_trend(df):
    df_season = pd.DataFrame(df['price_close'])

    for i in range(1, 13):
        df_season[f't - {i}'] = df_season['price_close'].shift(i)

    df_season.fillna(method='ffill', inplace=True)
    df_season.dropna(inplace=True)
    result_mul = seasonal_decompose(df_season['price_close'], model='multiplicative',  period=365)
    df_trend = pd.DataFrame(result_mul.trend)
    df_trend.index = pd.to_datetime(df_trend.index)

    df_trend['trend'].iloc[[-2]] = np.nan
    if math.isnan(df['price_close'].iloc[[-1]]):
        ### CHANGE -3 ACCORDING TO BANK HOLIDAYS AND WEEKENDS ###
        df_trend['trend'].iloc[[-1]] = df['price_close'].iloc[[-3]]
    else:
        df_trend['trend'].iloc[[-1]] = df['price_close'].iloc[[-1]]

    df_trend['trend'].iloc[[0]] = df['price_close'].iloc[[1]]

    df_trend['trend_int'] = df_trend['trend'].interpolate(method='quadratic')

    return df_trend


def impute_df(X):
    imputer = SimpleImputer()

    imputer.fit(X)
    X_imp = imputer.transform(X)
    X_imp = pd.DataFrame(X_imp)
    X_imp.columns = X.columns
    X_imp.index = X.index

    return X_imp


def linearize_df(X):
    X['real_gdp_t'] = X['real_gdp']*X['trend_int']
    X['cpi_t'] = X['cpi']*X['trend_int']
    X['MACD_Signal_t'] = X['MACD_Signal']*X['trend_int']
    X['MACD_t'] = X['MACD']*X['trend_int']
    X['MACD_Hist_t'] = X['MACD_Hist']*X['trend_int']
    X = X.drop(['real_gdp', 'cpi', 'MACD_Signal', 'MACD', 'MACD_Hist', 'trend_int'], axis=1)

    return X


def s_scaler(X):
    s_scaler = StandardScaler()

    s_scaler.fit(X)
    X_scaled = s_scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled)
    X_scaled.columns = X.columns
    X_scaled.index = X.index

    return X_scaled


def fill_nan(X, y):
    y = y.fillna(value=-2000)
    X = X.fillna(value=-2000)

    return X, y

def clean_data(df):
    ''' returns a clean dataframe tailored to our task'''
    df.replace('.','0',inplace=True)
    df = df.astype(float)

    return df


# def df_optimized(df, verbose=True, **kwargs):
#     """
#     Reduces size of dataframe by downcasting numeircal columns
#     :param df: input dataframe
#     :param verbose: print size reduction if set to True
#     :param kwargs:
#     :return: df optimized
#     """
#     in_size = df.memory_usage(index=True).sum()
#     # Optimized size here
#     for type in ["float", "integer"]:
#         l_cols = list(df.select_dtypes(include=type))
#         for col in l_cols:
#             df[col] = pd.to_numeric(df[col], downcast=type)
#             if type == "float":
#                 df[col] = pd.to_numeric(df[col], downcast="integer")
#     out_size = df.memory_usage(index=True).sum()
#     ratio = (1 - round(out_size / in_size, 2)) * 100
#     GB = out_size / 1000000000
#     if verbose:
#         print("optimized size by {} % | {} GB".format(ratio, GB))
#     return df


# No need for KNN Imputer, using simple imputer
# def knn_imputer(df=None):
#     knn_imp = KNNImputer(n_neighbors=2, missing_values=np.nan, weights='distance')

#     df_temp = knn_imp.fit_transform(df)
#     df_temp = pd.DataFrame(df_temp)
#     df_temp.columns = df.columns
#     df_temp.index = df.index
#     df = df_temp
#     return df


def ff_imputer(df_main):
    df_main_imp = df_main.astype('float32')
    df_main_imp.iloc[:,4:] = df_main_imp.iloc[:,4:].fillna(method='ffill')
    df_main_imp['weighted_ss'] = df_main_imp['weighted_ss'].fillna(df_main['weighted_ss'].mean())
    df_main_imp = df_main_imp.dropna()
    return df_main_imp

def SRNN_imputer(df_main):
    df_main_imp = df_main.astype('float32')
    df_main_imp[['real_gdp', 'cpi', 'MACD_Signal', 'MACD', 'MACD_Hist', 'trend_int']] = df_main_imp[['real_gdp', 'cpi', 'MACD_Signal', 'MACD', 'MACD_Hist', 'trend_int']].fillna(method='ffill')
    df_main_imp = df_main_imp.fillna(df_main.mean())
    df_main_imp = df_main_imp.dropna()
    return df_main_imp


# def padding(df=None):
#     df_pad = pad_sequences(df, dtype='float64', value=-42069)
#     return df_pad


def standard_scaler(X):

    ss_scaler = StandardScaler()
    X_scaled = ss_scaler.fit_transform(X)

    return ss_scaler, X_scaled


def minmax_scaler(X):

    mm_scaler = MinMaxScaler()
    X_scaled = mm_scaler.fit_transform(X)

    return mm_scaler, X_scaled
