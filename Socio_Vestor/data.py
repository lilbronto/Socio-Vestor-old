import requests
import pandas as pd
import numpy as np

apikey_av = "GA32KX1XU3RE15LO"
url_av = "https://www.alphavantage.co/query?"

def get_spy_data(outputsize="compact"):

    function_intra = "TIME_SERIES_DAILY_ADJUSTED"
    symbol = "SPY"

    params_intra = {"function" : f"{function_intra}",
            "symbol" : f"{symbol}",
            "outputsize" : f"{outputsize}",
            "apikey" : f"{apikey_av}"
            }

    response_SPY = requests.get(url_av, params=params_intra).json()
    data_SPY = pd.DataFrame.from_dict(response_SPY['Time Series (Daily)']).transpose()
    data_SPY.index.name = 'date'
    data_SPY = data_SPY.sort_values(by='date', ascending=True)
    data_SPY = data_SPY.drop(['8. split coefficient', '7. dividend amount', '5. adjusted close'], axis=1)
    data_SPY = data_SPY.rename(columns={"1. open": "price_open", "2. high": "price_high", "3. low": "price_low", "4. close": "price_close", "6. volume": "trade_volume"})

    return data_SPY

def get_cpi_data():

    function_cpi = "CPI"
    interval = "monthly"

    params_cpi = {"function" : f"{function_cpi}",
            "interval" : f"{interval}",
            "apikey" : f"{apikey_av}"
            }

    response_CPI = requests.get(url_av, params=params_cpi).json()
    data_CPI = pd.DataFrame.from_dict(response_CPI['data'])

    data_CPI = data_CPI.set_index('date')
    data_CPI = data_CPI.rename(columns={"value": "cpi"})
    data_CPI = data_CPI.sort_values(by='date', ascending=True)

    return data_CPI

def get_inflation_data(function_inf="INFLATION_EXPECTATION"):

    params_inf = {"function" : f"{function_inf}",
            "apikey" : f"{apikey_av}"
            }

    response_inflation = requests.get(url_av, params=params_inf).json()
    data_inflation = pd.DataFrame.from_dict(response_inflation['data'])
    data_inflation = data_inflation.set_index('date')

    data_inflation = data_inflation.rename(columns={"value": "inflation"})
    data_inflation = data_inflation.sort_values(by='date', ascending=True)

    return data_inflation

def get_consumer_sentiment_data():

    function_cs = "CONSUMER_SENTIMENT"

    params_cs = {"function" : f"{function_cs}",
            "apikey" : f"{apikey_av}"
            }

    response_consumer_sentiment = requests.get(url_av, params=params_cs).json()
    data_consumer_sentiment = pd.DataFrame.from_dict(response_consumer_sentiment['data'])
    data_consumer_sentiment = data_consumer_sentiment.set_index('date')

    data_consumer_sentiment = data_consumer_sentiment.rename(columns={"value": "consumer_sentiment"})
    data_consumer_sentiment = data_consumer_sentiment.sort_values(by='date', ascending=True)

    return data_consumer_sentiment

def get_social_sentiment_data(from_date="2022-02-23",to_date = "2022-02-27"):

    symbol = "SPY"

    headers_dict = {"Authorization" : "Token 2b104f7101af551565791f4a47ab3ba7ef89598a",
                    "Accept" : "application/json"}

    url_ss = f"https://socialsentiment.io/api/v1/stocks/{symbol}/sentiment/daily/"

    params_ss = {"to_date" : f"{to_date}",
            "from_date" : f"{from_date}"
            }

    response_ss = requests.get(url_ss, params=params_ss, headers=headers_dict).json()

    data_ss = pd.DataFrame.from_dict(response_ss)
    data_ss = data_ss.set_index('date')
    data_ss = data_ss.drop(['stock', 'positive_score', 'negative_score', 'avg_7_days', 'avg_14_days', 'avg_30_days'], axis=1)

    data_ss['weighted_ss'] = data_ss['score']/data_ss['activity']
    data_ss = data_ss.drop(['score', 'activity'], axis=1)

    return data_ss

def get_intraday_data():
    ''' fetches the intraday data for SPY for a given interval for the trailing past two years'''

    function = "TIME_SERIES_INTRADAY_EXTENDED"
    symbol = "SPY"
    interval = "1min"
    slice_ = "year1month1"

    url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&slice={slice_}&apikey={apikey_av}"

    SPY_ext_hist = pd.read_csv(url)
    data_SPY_ext_hist = pd.DataFrame(SPY_ext_hist)
    data_SPY_ext_hist = data_SPY_ext_hist.set_index('time')

    return data_SPY_ext_hist

def get_main_df():

    data_SPY = get_spy_data()
    data_CPI = get_cpi_data()
    data_inflation = get_inflation_data()
    data_consumer_sentiment = get_consumer_sentiment_data()
    data_ss = get_social_sentiment_data()

    df_main = pd.concat([data_SPY, data_CPI, data_inflation, data_consumer_sentiment, data_ss], axis=1)
    df_main = df_main.sort_values(by='date', ascending=True)
    df_main = df_main.loc['2000-01-01':]

    return df_main
