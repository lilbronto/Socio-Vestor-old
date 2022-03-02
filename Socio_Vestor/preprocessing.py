import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing.sequence import pad_sequences


def to_float(x):
    return float(x)

def set_pipeline():

    pipe = Pipeline([])

    return pipe

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

def impute_data(df=None):
    knn_imp = KNNImputer(n_neighbors=2, missing_values=np.nan, weights='distance')

    df_temp = knn_imp.fit_transform(df)
    df_temp = pd.DataFrame(df_temp)
    df_temp.columns = df.columns
    df_temp.index = df.index
    df = df_temp
    return df

def padding(df=None):
    df_pad = pad_sequences(df, dtype='float64', value=-42069)
    return df_pad

def scale(df=None):
    mm_scaler = MinMaxScaler()
    df_scaled = mm_scaler.fit_transform(df)
    return df_scaled
