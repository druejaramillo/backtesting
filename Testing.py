import tda
from tda import auth
from tda.client import Client
from urllib.request import urlopen
import atexit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from Technical_Indicators import Indicators
from Portfolio_Metrics import Metrics
from Backtester import WalkForwardOptimization

'''
# Get authorization
consumer_key = 'TJCYE4T8WFVW3A21H7GGAGT8ULDE71FB@AMER.OAUTHAP'
redirect_uri = 'https://localhost'
token_path = 'token.pickle'

def make_webdriver():

	from selenium import webdriver
	option = webdriver.ChromeOptions()
	option.binary_location = '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser'
	driver = webdriver.Chrome(options = option)
	atexit.register(lambda: driver.quit())
	return driver

def resample(df, interval):

	d = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}

	return df.resample(interval, on='date').agg(d)

c = auth.easy_client(consumer_key, redirect_uri, token_path, make_webdriver)

ticker = 'AAPL'

r = c.get_price_history(ticker, period_type=Client.PriceHistory.PeriodType.YEAR, period=Client.PriceHistory.Period.FIVE_YEARS, frequency_type=Client.PriceHistory.FrequencyType.DAILY, frequency=Client.PriceHistory.Frequency.DAILY)
assert r.status_code == 200
json = r.json()
price_history = json['candles']

df = pd.json_normalize(price_history)
df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
df['datetime'] = df['datetime'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000.0).strftime('%Y-%m-%d'))
df['datetime'] = pd.to_datetime(df['datetime'])
df.rename(columns={'datetime': 'date'}, inplace=True)
df.set_index('date', inplace=True)
ohlcv = df.copy()
ohlc = ohlcv.copy()
ohlc.pop('volume')

prices = ohlc['close']

r = c.get_price_history('SPY', period_type=Client.PriceHistory.PeriodType.YEAR, period=Client.PriceHistory.Period.THREE_YEARS, frequency_type=Client.PriceHistory.FrequencyType.DAILY, frequency=Client.PriceHistory.Frequency.DAILY)
assert r.status_code == 200
json = r.json()
price_history = json['candles']

df = pd.json_normalize(price_history)
df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
df['datetime'] = df['datetime'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000.0).strftime('%Y-%m-%d'))
df['datetime'] = pd.to_datetime(df['datetime'])
df.rename(columns={'datetime': 'date'}, inplace=True)
df.set_index('date', inplace=True)

spy = df['close']

Metrics().calculate_treynor_ratio(prices, spy, 0.02)

IndicatorsDaily = Indicators(ohlcv)
rsi_14d = IndicatorsDaily.rsi(14)

ohlcv.reset_index(inplace=True)
weekly_ohlcv = resample(ohlcv, '7D')
ohlcv.set_index('date', inplace=True)
IndicatorsWeekly = Indicators(weekly_ohlcv)
rsi_14w = IndicatorsWeekly.rsi(14)

PortfolioMetrics = Metrics()
sharpe_ratio = PortfolioMetrics.calculate_sharpe_ratio(ohlc['close'], benchmark_rate=0.02)

prices = pd.DataFrame()

for ticker in ['AAPL', 'JNJ', 'TSLA', 'KO']:

    r = c.get_price_history(ticker, period_type=Client.PriceHistory.PeriodType.YEAR, period=Client.PriceHistory.Period.ONE_YEAR, frequency_type=Client.PriceHistory.FrequencyType.DAILY, frequency=Client.PriceHistory.Frequency.DAILY)
    assert r.status_code == 200
    json = r.json()
    price_history = json['candles']

    df = pd.json_normalize(price_history)
    df = df[['datetime', 'close']]
    df['datetime'] = df['datetime'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000.0).strftime('%Y-%m-%d'))
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.rename(columns={'datetime': 'date', 'close': ticker}, inplace=True)
    df.set_index('date', inplace=True)

    if prices.empty:
    	prices = df
    else:
    	prices[ticker] = df[ticker]

log_returns = prices.pct_change().apply(lambda x: np.log(1+x))
cov = log_returns.cov()
corr = log_returns.corr()

ind_er = log_returns.resample('Y').last().pct_change().mean()

p_ret = [] # Define an empty array for portfolio returns
p_vol = [] # Define an empty array for portfolio volatility
p_weights = [] # Define an empty array for asset weights

num_assets = len(log_returns.columns)
num_portfolios = 100000
for portfolio in range(num_portfolios):
    weights = np.random.rand(1, num_assets)
    weights = weights/np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its 
                                      # weights 
    p_ret.append(returns[0])
    var = cov.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
    sd = np.sqrt(var) # Daily standard deviation
    ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
    p_vol.append(ann_sd)

data = {'Returns': p_ret, 'Volatility': p_vol}

for counter, symbol in enumerate(log_returns.columns):
    #print(counter, symbol)
    data[symbol+' weight'] = [w[0][counter] for w in p_weights]

portfolios  = pd.DataFrame(data)
min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
print(min_vol_port)
rf = 0.01 # risk factor
optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
print(optimal_risky_port)

plt.subplots(figsize=(10, 10))
plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)

plt.show()

dates = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10']
tickers = ['a', 'b', 'c', 'd', 'e', 'f']
types = ['close', 'signal']
signals = np.random.random((len(dates), len(tickers)))
rand = np.random.default_rng()
prices = rand.integers(0, 100, size=(len(dates), len(tickers)))
prices_df = pd.DataFrame(prices)
prices_df.index = dates
prices_df.columns = tickers
signals_df = pd.DataFrame(signals)
signals_df.index = dates
signals_df.columns = tickers
print(prices_df)
print(signals_df)

df = pd.concat([prices_df, signals_df], keys=types)
df = df.swaplevel()
df.sort_index(inplace=True)
print(df)

for date, new_df in df.groupby(level=0):
	print(new_df)
'''

# Define the strategy parameters
rsi_buy_vals = range(10, 45, 5)
rsi_buy_ints = range(10, 20)
rsi_sell_vals = range(60, 95, 5)
rsi_sell_ints = range(10, 20)

def signal_func(params, all_ohlcv):

	# Pull out the parameters
	rsi_buy_val = params[0]
	rsi_buy_int = params[1]
	rsi_sell_val = params[2]
	rsi_sell_int = params[3]

	symbols = list(all_ohlcv.keys())

	# Get the data needed for trading signals
	rsi_buy_dict = {}
	rsi_sell_dict = {}
	for symbol in symbols:
		indicators = Indicators(all_ohlcv[symbol])
		rsi_buy_dict[symbol] = indicators.rsi(rsi_buy_int)
		rsi_sell_dict[symbol] = indicators.rsi(rsi_sell_int)

	# Generate trading signals
	signals = pd.DataFrame()
	for symbol in symbols:

		symbol_signals = {}
		for date in all_ohlcv[symbol].index:

			rsi_1 = rsi_buy_dict[symbol].loc[date]
			rsi_2 = rsi_sell_dict[symbol].loc[date]
			if rsi_1 <= rsi_buy_val:
				symbol_signals[date] = 1
			elif rsi_2 >= rsi_sell_val:
				symbol_signals[date] = -1
			else:
				symbol_signals[date] = 0

		signal_series = pd.Series(symbol_signals)
		if signals.empty:
			signals = pd.DataFrame(signal_series)
			signals.columns = [symbol]
		else:
			signals[symbol] = signal_series

	return signals

if __name__ == '__main__':

	# Set the hyperparameters
	universe = ['AAPL', 'KO', 'JNJ', 'MSFT']
	#universe = ['AAPL', 'KO', 'JNJ', 'MSFT', 'FB', 'AMZN', 'TSLA', 'WM', 'WMT', 'HD', 'LUV', 'COST', 'SNAP', 'TWTR']
	params_possible = [rsi_buy_vals, rsi_buy_ints, rsi_sell_vals, rsi_sell_ints]
	strat = [params_possible, signal_func]
	weighting_method = 'hrp'
	fitness_criterion = 'max_sharpe'
	risk_free_rate = 0.02

	# Run the backtest
	backtester = WalkForwardOptimization(universe, strat, weighting_method, fitness_criterion, risk_free_rate)
	backtester.run_backtest()











































