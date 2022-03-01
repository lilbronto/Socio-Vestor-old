import requests
import pandas as pd

apikey = "GA32KX1XU3RE15LO"
url = "https://www.alphavantage.co/query?"

def get_spy_data(outputsize="compact"):

    function = "TIME_SERIES_DAILY_ADJUSTED"
    symbol = "SPY"

    params = {"function" : f"{function}",
            "symbol" : f"{symbol}",
            "outputsize" : f"{outputsize}",
            "apikey" : f"{apikey}"
            }

    response_SPY = requests.get(url, params=params).json()
    data_SPY = pd.DataFrame.from_dict(response_SPY['Time Series (Daily)']).transpose()

    return data_SPY

def get_cpi_data():

    function = "CPI"
    interval = "monthly"

    params = {"function" : f"{function}",
            "interval" : f"{interval}",
            "apikey" : f"{apikey}"
            }

    response_CPI = requests.get(url, params=params).json()
    data_CPI = pd.DataFrame.from_dict(response_CPI['data'])

    return data_CPI

def get_inflation_data(function="INFLATION_EXPECTATION"):

    params = {"function" : f"{function}",
              "apikey" : f"{apikey}"
            }

    response_inflation = requests.get(url, params=params).json()
    data_inflation = pd.DataFrame.from_dict(response_inflation['data'])

    return data_inflation

def get_consumer_sentiment_data():

    function = "CONSUMER_SENTIMENT"

    params = {"function" : f"{function}",
            "apikey" : f"{apikey}"
            }

    response_consumer_sentiment = requests.get(url, params=params).json()
    data_consumer_sentiment = pd.DataFrame.from_dict(response_consumer_sentiment['data'])

    return data_consumer_sentiment
