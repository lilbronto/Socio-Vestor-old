import requests
import pandas as pd
import numpy as np

apikey_av = "GA32KX1XU3RE15LO"
url_av = "https://www.alphavantage.co/query?"

def get_spy_data(outputsize="compact"):

    function_intra = "TIME_SERIES_DAILY_ADJUSTED"
    symbol = "SPY"
    outputsize = "full"

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

    function_inf = "INFLATION_EXPECTATION"

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

def get_social_sentiment_data(from_date="2021-03-12",to_date = "2022-04-09"):

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

def get_unemployment_data():

    function_ur = "UNEMPLOYMENT"

    params_ur = {"function" : f"{function_ur}",
                "apikey" : f"{apikey_av}"
                }

    response_unemployment_rate = requests.get(url_av, params=params_ur).json()

    data_unemployment_rate = pd.DataFrame.from_dict(response_unemployment_rate['data'])
    data_unemployment_rate = data_unemployment_rate.set_index('date')
    data_unemployment_rate = data_unemployment_rate.rename(columns={"value": "unemployment_rate"})
    data_unemployment_rate = data_unemployment_rate.sort_values(by='date', ascending=True)

    return data_unemployment_rate

def get_interest_data():

    function_ir = "FEDERAL_FUNDS_RATE"
    interval_ir = "monthly"

    params_ir = {"function" : f"{function_ir}",
                "interval" : f"{interval_ir}",
                "apikey" : f"{apikey_av}"
                }

    response_interest_rate = requests.get(url_av, params=params_ir).json()

    data_interest_rate = pd.DataFrame.from_dict(response_interest_rate['data'])
    data_interest_rate = data_interest_rate.set_index('date')
    data_interest_rate = data_interest_rate.rename(columns={"value": "interest_rate"})
    data_interest_rate = data_interest_rate.sort_values(by='date', ascending=True)

    return data_interest_rate

def get_treasury_data():

    function_ty = "TREASURY_YIELD"
    interval_ty = "monthly"

    params_iy = {"function" : f"{function_ty}",
                "interval" : f"{interval_ty}",
                "apikey" : f"{apikey_av}"
                }

    response_treasury_yield = requests.get(url_av, params=params_iy).json()

    data_treasury_yield = pd.DataFrame.from_dict(response_treasury_yield['data'])
    data_treasury_yield = data_treasury_yield.set_index('date')
    data_treasury_yield = data_treasury_yield.rename(columns={"value": "treasury_yield"})
    data_treasury_yield = data_treasury_yield.sort_values(by='date', ascending=True)

    return data_treasury_yield

def get_real_gdp_data():
    function_rgdp = "REAL_GDP"
    interval_rgdp = "quarterly"

    params_rgdp = {"function" : f"{function_rgdp}",
               "interval" : f"{interval_rgdp}",
               "apikey" : f"{apikey_av}"
              }

    response_rgdp = requests.get(url_av, params=params_rgdp).json()

    data_rgdp = pd.DataFrame.from_dict(response_rgdp['data'])
    data_rgdp = data_rgdp.set_index('date')

    data_rgdp = data_rgdp.rename(columns={"value": "real_gdp"})
    data_rgdp = data_rgdp.sort_values(by='date', ascending=True)

    return data_rgdp

def get_macd_data():
    function_macd = "MACD"
    symbol_macd = "SPY"
    interval_macd = "daily"
    series_type = "close"

    params_macd = {"function" : f"{function_macd}",
                "symbol": f"{symbol_macd}",
                "interval" : f"{interval_macd}",
                "series_type" : f"{series_type}",
                "apikey" : f"{apikey_av}"
                }

    response_macd = requests.get(url_av, params=params_macd).json()

    macd_dates_df = pd.DataFrame.from_dict(response_macd.get('Technical Analysis: MACD').keys())
    macd_df = pd.DataFrame.from_dict(response_macd.get('Technical Analysis: MACD').values())
    data_macd = pd.concat([macd_dates_df, macd_df], axis=1)
    data_macd = data_macd.rename(columns={0: "date"})
    data_macd = data_macd.set_index('date')
    data_macd = data_macd.sort_values(by='date', ascending=True)

    return data_macd

def get_main_df():

    data_SPY = get_spy_data()
    data_CPI = get_cpi_data()
    data_inflation = get_inflation_data()
    data_consumer_sentiment = get_consumer_sentiment_data()
    data_ss = get_social_sentiment_data()
    data_unemployment_rate = get_unemployment_data()
    data_interest_rate = get_interest_data()
    data_treasury_yield = get_treasury_data()
    data_real_gdp = get_real_gdp_data()
    data_macd = get_macd_data()

    df_main = pd.concat([data_SPY, data_CPI, data_inflation,
                         data_consumer_sentiment, data_ss,
                         data_unemployment_rate, data_interest_rate,
                         data_treasury_yield, data_real_gdp, data_macd], axis=1)


    df_main = df_main.sort_values(by='date', ascending=True)
    df_main = df_main.loc['2000-01-01':]
    df_main = df_main.astype('float32')
    df_main.index = pd.to_datetime(df_main.index)

    return df_main
