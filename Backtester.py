import tda
from tda import auth
from tda.client import Client
from urllib.request import urlopen
import atexit
import pandas as pd
import datetime
import time
import math
from Portfolio import PortfolioHistory
from Portfolio_Simulation import Simulator
from Genetic_Algorithm import GAOptimizer
from Execution import Executor

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

class WalkForwardOptimization():

	def __init__(self, universe, strat, weighting_method, fitness_criterion, risk_free_rate):

		'''
		Inputs:
		universe: list of ticker symbols to apply the strategy to
		strat: list of strategy components (parameters & potential values, signal function)
		weighting_method: portfolio construction method (equal ** NOT WORKING **, mean-variance (2 types), hrp, herc)
		fitness_criterion: choice of measurement of fitness (min volatility, max sharpe, max sortino, max calmar)
		risk_free_rate: current risk-free rate of return
		'''

		self.c = auth.easy_client(consumer_key, redirect_uri, token_path, make_webdriver)

		# Get OHLCV for all the tickers in our universe of choices
		self.all_ohlcv = {}

		for ticker in universe:

			r = self.c.get_price_history(ticker, period_type=Client.PriceHistory.PeriodType.YEAR, period=Client.PriceHistory.Period.THREE_YEARS, frequency_type=Client.PriceHistory.FrequencyType.DAILY, frequency=Client.PriceHistory.Frequency.DAILY)
			assert r.status_code == 200
			json = r.json()
			price_history = json['candles']

			df = pd.json_normalize(price_history)
			df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
			df['datetime'] = df['datetime'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000.0).strftime('%Y-%m-%d'))
			df['datetime'] = pd.to_datetime(df['datetime'])
			df.rename(columns={'datetime': 'date'}, inplace=True)
			df.set_index('date', inplace=True)

			self.all_ohlcv[ticker] = df

		# Get OHLCV for SPY and also pull out just closing prices
		r = self.c.get_price_history('SPY', period_type=Client.PriceHistory.PeriodType.YEAR, period=Client.PriceHistory.Period.THREE_YEARS, frequency_type=Client.PriceHistory.FrequencyType.DAILY, frequency=Client.PriceHistory.Frequency.DAILY)
		assert r.status_code == 200
		json = r.json()
		price_history = json['candles']

		df = pd.json_normalize(price_history)
		df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
		df['datetime'] = df['datetime'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000.0).strftime('%Y-%m-%d'))
		df['datetime'] = pd.to_datetime(df['datetime'])
		df.rename(columns={'datetime': 'date'}, inplace=True)
		df.set_index('date', inplace=True)

		self.spy = pd.DataFrame(df['close'])
		
		self.prices = pd.DataFrame()

		self.strat = strat
		self.weighting_method = weighting_method
		self.fitness_criterion = fitness_criterion

		self.training_dates = None
		self.testing_dates = None
		self.num_segments = None

		self.risk_free_rate = risk_free_rate

		self.executor = None
		self.optimizer = None
		self.main_simulator = None
		self.training_simulator = None

	def process_data(self):

		symbols = list(self.all_ohlcv.keys())

		# Pull out the open and close prices
		for ticker in symbols:
			close_series = self.all_ohlcv[ticker]['close']
			if self.prices.empty:
				self.prices = pd.DataFrame(close_series)
				self.prices.columns = [ticker]
			else:
				self.prices[ticker] = close_series
		
		training_size = 375 # Number of days in each training window
		testing_size = 125 # Number of days in each testing window

		total_days = len(self.all_ohlcv[symbols[0]]) # Total number of days in the data
		self.num_segments = math.floor((total_days - training_size) / testing_size) - 1 # Number of walk-forward steps (leave some early data leftover)

		# Create start and end dates for training and testing segments
		self.training_dates = []
		self.testing_dates = []
		for i in range(self.num_segments):
			start = (self.num_segments - i) * testing_size + training_size
			mid = start - training_size
			end = mid - testing_size
			self.training_dates.append((start, mid))
			self.testing_dates.append((mid, end))

	def output_results(self):
		
		# Get the final portfolio
		final_portfolio = self.main_simulator.portfolio_history

		# Output results
		final_portfolio.print_summary()
		final_portfolio.plot()
		final_portfolio.plot_benchmark_comparison()

	def run_backtest(self, **kwargs):
		'''
		Inputs:
    	- **kwargs: sector constraints for mean-variance portfolio construction (if any)
		'''

		time0 = time.time()
		
		# Split the data into open and close prices and split the close prices into walk-forward segments for training and testing
		self.process_data()

		# Create the optimization hyperparameters
		population_size = 5
		num_generations = 2
		k = 3
		p_cross = 0.8
		p_mut = 0.01

		# Create the main portfolio, optimizer, and simulators
		initial_cash = 10000
		percent_slippage = 0.0005
		self.executor = Executor(self.c, self.risk_free_rate, self.weighting_method)
		self.main_simulator = Simulator(self.executor, self.all_ohlcv, self.prices, self.spy, self.risk_free_rate, initial_cash, percent_slippage, training=False)
		self.training_simulator = Simulator(self.executor, self.all_ohlcv, self.prices, self.spy, self.risk_free_rate, initial_cash, percent_slippage, training=True)
		self.optimizer = GAOptimizer(self.strat, self.fitness_criterion, self.training_simulator, population_size, num_generations, k, p_cross, p_mut, **kwargs)

		# For each walk-forward segment:
		# - Pass the training data and strategy into the GA optimizer
		# - Run the optimized strategy on the testing data and add the performance to the portfolio history
		for i in range(len(self.training_dates)):

			print('\n\n*******************************************************************')
			print()
			print('                           Iteration {0}                           '.format(i+1))
			print()
			print('*******************************************************************')

			# Optimize the strategy on some training data
			print('\n================= Training =================')
			begin_time = time.time()
			start_idx, end_idx = self.training_dates[i]
			optimized_parameters = self.optimizer.optimize(start_idx, end_idx, **kwargs)
			optimized_strat = [optimized_parameters, self.strat[1]]
			end_time = time.time()
			print('\nTraining time: {0} seconds'.format(round(end_time-begin_time, 2)))

			# Run the new strategy on the next piece of testing data
			print('\n================= Testing =================')
			start_date, end_date = self.testing_dates[i]
			print('\n\tSimulating main portfolio')
			self.main_simulator.simulate(start_idx, end_idx, optimized_strat, **kwargs)

		# End the main simulation
		x, end_idx = self.testing_dates[-1]
		self.main_simulator.end(end_idx)

		time1 = time.time()

		print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
		print()
		print('\t Total runtime: {0} seconds'.format(round(time1-time0, 2)))
		print()
		print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')

		# Output the results
		self.output_results()


















































