from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
