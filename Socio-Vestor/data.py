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
    data_CPI = data_CPI.set_index('date')

    return data_CPI

def get_inflation_data(function="INFLATION_EXPECTATION"):

    params = {"function" : f"{function}",
              "apikey" : f"{apikey}"
              }

    response_inflation = requests.get(url, params=params).json()
    data_inflation = pd.DataFrame.from_dict(response_inflation['data'])
    data_inflation = data_inflation.set_index('date')

    return data_inflation

def get_consumer_sentiment_data():

    function = "CONSUMER_SENTIMENT"

    params = {"function" : f"{function}",
            "apikey" : f"{apikey}"
            }

    response_consumer_sentiment = requests.get(url, params=params).json()
    data_consumer_sentiment = pd.DataFrame.from_dict(response_consumer_sentiment['data'])
    data_consumer_sentiment = data_consumer_sentiment.set_index('date')

    return data_consumer_sentiment

def get_social_sentiment_data(from_date="2022-02-23",to_date = "2022-02-27"):

    symbol = "SPY"

    headers_dict = {"Authorization" : "Token 2b104f7101af551565791f4a47ab3ba7ef89598a",
                    "Accept" : "application/json"}

    url_social_sent = f"https://socialsentiment.io/api/v1/stocks/{symbol}/sentiment/daily/"

    params = {"to_date" : f"{to_date}",
            "from_date" : f"{from_date}"
            }

    response = requests.get(url_social_sent, params=params, headers=headers_dict).json()

    response_df = pd.DataFrame.from_dict(response)
    response_df = response_df.set_index('date')

    return response_df

def get_intraday_data():
    ''' fetches the intraday data for SPY for a given interval for the trailing past two years'''

    function = "TIME_SERIES_INTRADAY_EXTENDED"
    symbol = "SPY"
    interval = "1min"
    slice_ = "year1month1"

    url = f"https://www.alphavantage.co/query?\
            function={function}&symbol={symbol}\
            &interval={interval}&slice={slice_}&apikey={apikey}"

    SPY_ext_hist = pd.read_csv(url)
    data_SPY_ext_hist = pd.DataFrame(SPY_ext_hist)
    data_SPY_ext_hist = data_SPY_ext_hist.set_index('time')

    return data_SPY_ext_hist
