import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from keras.preprocessing.sequence import pad_sequences

def clean_data(df):
    ''' returns a clean dataframe tailored to our task'''
    df.replace('.','0',inplace=True)
    df = df.astype(float)

    return df

def df_optimized(df, verbose=True, **kwargs):
    """
    Reduces size of dataframe by downcasting numeircal columns
    :param df: input dataframe
    :param verbose: print size reduction if set to True
    :param kwargs:
    :return: df optimized
    """
    in_size = df.memory_usage(index=True).sum()
    # Optimized size here
    for type in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=type))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=type)
            if type == "float":
                df[col] = pd.to_numeric(df[col], downcast="integer")
    out_size = df.memory_usage(index=True).sum()
    ratio = (1 - round(out_size / in_size, 2)) * 100
    GB = out_size / 1000000000
    if verbose:
        print("optimized size by {} % | {} GB".format(ratio, GB))
    return df

def knn_imputer(df=None):
    knn_imp = KNNImputer(n_neighbors=2, missing_values=np.nan, weights='distance')

    df_temp = knn_imp.fit_transform(df)
    df_temp = pd.DataFrame(df_temp)
    df_temp.columns = df.columns
    df_temp.index = df.index
    df = df_temp
    return df

def ff_imputer(df_main):
    df_main_imp = df_main.astype('float32')
    df_main.iloc[:,4:] = df_main.iloc[:,4:].fillna(method='ffill')
    df_main_imp['weighted_ss'] = df_main_imp['weighted_ss'].fillna(df_main['weighted_ss'].mean())
    df_main_imp = df_main_imp.dropna()
    return df_main_imp

def padding(df=None):
    df_pad = pad_sequences(df, dtype='float64', value=-42069)
    return df_pad

def standard_scaler(X):

    ss_scaler = StandardScaler()
    X_scaled = ss_scaler.fit_transform(X)

    return ss_scaler, X_scaled

def minmax_scaler(X):

    mm_scaler = MinMaxScaler()
    X_scaled = mm_scaler.fit_transform(X)

    return mm_scaler, X_scaled
