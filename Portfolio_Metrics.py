import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Dict, Any, Callable

class Metrics:

	def __init__(self):
		
		self.DRAWDOWN_EVALUATORS: Dict[str, Callable] = {
		    'dollar': lambda price, peak: peak - price,
		    'percent': lambda price, peak: -((price / peak) - 1),
		    'log': lambda price, peak: np.log(peak) - np.log(price),
		}

	def calculate_return_series(self, series: pd.Series) -> pd.Series:
	    """
	    Calculates the return series of a given time series.
	    The first value will always be NaN.
	    """
	    shifted_series = series.shift(1, axis=0)
	    return series / shifted_series - 1

	def calculate_log_return_series(self, series: pd.Series) -> pd.Series:
	    """
	    Same as calculate_return_series but with log returns
	    """
	    shifted_series = series.shift(1, axis=0)
	    return pd.Series(np.log(series / shifted_series))

	def calculate_percent_return(self, series: pd.Series) -> float:
	    """
	    Takes the first and last value in a series to determine the percent return, 
	    assuming the series is in date-ascending order
	    """
	    return series.iloc[-1] / series.iloc[0] - 1

	def get_years_past(self, series: pd.Series) -> float:
	    """
	    Calculate the years past according to the index of the series for use with
	    functions that require annualization   
	    """
	    start_date = series.index[0]
	    end_date = series.index[-1]
	    return (end_date - start_date).days / 365.25

	def calculate_cagr(self, series: pd.Series) -> float:
	    """
	    Calculate compounded annual growth rate
	    """
	    start_price = series.iloc[0]
	    end_price = series.iloc[-1]
	    value_factor = end_price / start_price
	    year_past = self.get_years_past(series)
	    return (value_factor ** (1 / year_past)) - 1

	def calculate_annualized_volatility(self, return_series: pd.Series) -> float:
	    """
	    Calculates annualized volatility for a date-indexed return series. 
	    Works for any interval of date-indexed prices and returns.
	    """
	    years_past = self.get_years_past(return_series)
	    entries_per_year = return_series.shape[0] / years_past
	    return return_series.std() * np.sqrt(entries_per_year)

	def calculate_sharpe_ratio(self, price_series: pd.Series, 
	    benchmark_rate: float=0) -> float:
	    """
	    Calculates the Sharpe ratio given a price series. Defaults to benchmark_rate
	    of zero.
	    """
	    cagr = self.calculate_cagr(price_series)
	    return_series = self.calculate_return_series(price_series)
	    volatility = self.calculate_annualized_volatility(return_series)
	    return (cagr - benchmark_rate) / volatility

	def calculate_rolling_sharpe_ratio(self, price_series: pd.Series,
	    n: float=20) -> pd.Series:
	    """
	    Compute an approximation of the Sharpe ratio on a rolling basis. 
	    Intended for use as a preference value.
	    """
	    rolling_return_series = self.calculate_return_series(price_series).rolling(n)
	    return rolling_return_series.mean() / rolling_return_series.std()

	def calculate_annualized_downside_deviation(self, return_series: pd.Series,
	    benchmark_rate: float=0) -> float:
	    """
	    Calculates the downside deviation for use in the Sortino ratio.
	    Benchmark rate is assumed to be annualized. It will be adjusted according
	    to the number of periods per year seen in the data.
	    """

	    # For both de-annualizing the benchmark rate and annualizing result
	    years_past = self.get_years_past(return_series)
	    entries_per_year = return_series.shape[0] / years_past

	    adjusted_benchmark_rate = ((1+benchmark_rate) ** (1/entries_per_year)) - 1

	    downside_series = adjusted_benchmark_rate - return_series
	    downside_sum_of_squares = (downside_series[downside_series > 0] ** 2).sum()
	    denominator = return_series.shape[0] - 1
	    downside_deviation = np.sqrt(downside_sum_of_squares / denominator)

	    return downside_deviation * np.sqrt(entries_per_year)

	def calculate_sortino_ratio(self, price_series: pd.Series,
	    benchmark_rate: float=0) -> float:
	    """
	    Calculates the Sortino ratio.
	    """
	    cagr = self.calculate_cagr(price_series)
	    return_series = self.calculate_return_series(price_series)
	    downside_deviation = self.calculate_annualized_downside_deviation(return_series, benchmark_rate)
	    return (cagr - benchmark_rate) / downside_deviation

	def calculate_jensens_alpha(self, return_series: pd.Series, 
	    benchmark_return_series: pd.Series) -> float: 
	    """
	    Calculates Jensen's alpha. Prefers input series have the same index. Handles
	    NAs.
	    """

	    # Join series along date index and purge NAs
	    df = pd.concat([return_series, benchmark_return_series], sort=True, axis=1)
	    df = df.dropna()

	    # Get the appropriate data structure for scikit learn
	    clean_returns: pd.Series = df[df.columns.values[0]]
	    clean_benchmarks = pd.DataFrame(df[df.columns.values[1]])

	    # Fit a linear regression and return the alpha
	    regression = LinearRegression().fit(clean_benchmarks, y=clean_returns)
	    return regression.intercept_

	def calculate_drawdown_series(self, series: pd.Series, method: str='log') -> pd.Series:
	    """
	    Returns the drawdown series
	    """
	    assert method in self.DRAWDOWN_EVALUATORS, \
	        f'Method "{method}" must by one of {list(self.DRAWDOWN_EVALUATORS.keys())}'

	    evaluator = self.DRAWDOWN_EVALUATORS[method]
	    return evaluator(series, series.cummax())

	def calculate_max_drawdown(self, series: pd.Series, method: str='log') -> float:
	    """
	    Simply returns the max drawdown as a float
	    """
	    return self.calculate_drawdown_series(series, method).max()

	def calculate_max_drawdown_with_metadata(self, series: pd.Series, 
	    method: str='log') -> Dict[str, Any]:
	    """
	    Calculates max_drawndown and stores metadata about when and where. Returns 
	    a dictionary of the form 
	        {
	            'max_drawdown': float,
	            'peak_date': pd.Timestamp,
	            'peak_price': float,
	            'trough_date': pd.Timestamp,
	            'trough_price': float,
	        }
	    """

	    assert method in self.DRAWDOWN_EVALUATORS, \
	        f'Method "{method}" must by one of {list(self.DRAWDOWN_EVALUATORS.keys())}'

	    evaluator = self.DRAWDOWN_EVALUATORS[method]

	    max_drawdown = 0
	    local_peak_date = peak_date = trough_date = series.index[0]
	    local_peak_price = peak_price = trough_price = series.iloc[0]

	    for date, price in series.iteritems():

	        # Keep track of the rolling max
	        if price > local_peak_price:
	            local_peak_date = date
	            local_peak_price = price

	        # Compute the drawdown
	        drawdown = evaluator(price, local_peak_price)

	        # Store new max drawdown values
	        if drawdown > max_drawdown:
	            max_drawdown = drawdown

	            peak_date = local_peak_date
	            peak_price = local_peak_price

	            trough_date = date
	            trough_price = price

	    return {
	        'max_drawdown': max_drawdown,
	        'peak_date': peak_date,
	        'peak_price': peak_price,
	        'trough_date': trough_date,
	        'trough_price': trough_price
	    }

	def calculate_log_max_drawdown_ratio(self, series: pd.Series) -> float:
	    log_drawdown = self.calculate_max_drawdown(series, method='log')
	    log_return = np.log(series.iloc[-1]) - np.log(series.iloc[0])
	    return log_return - log_drawdown

	def calculate_calmar_ratio(self, series: pd.Series, years_past: int=3) -> float:
	    """
	    Return the percent max drawdown ratio over the past years_past years (default 3),
	    otherwise known as the Calmar Ratio
	    """

	    # Filter series on past three years
	    last_date = series.index[-1]
	    num_years_ago = last_date - pd.Timedelta(days=years_past*365.25)
	    series = series[series.index > num_years_ago]

	    # Compute annualized percent max drawdown ratio
	    percent_drawdown = self.calculate_max_drawdown(series, method='percent')
	    cagr = self.calculate_cagr(series)
	    return cagr / percent_drawdown

	def calculate_treynor_ratio(self, price_series: pd.Series, benchmark_series: pd.Series, risk_free_rate: float) -> float:
		'''
		Return the reward-to-volatility ratio, otherwise known as the Treynor ratio
		'''

		# Get the return series
		return_series = self.calculate_return_series(price_series)
		benchmark_return_series = self.calculate_return_series(benchmark_series)

		# Join series along date index and purge NAs
		df = pd.concat([return_series, benchmark_return_series], sort=True, axis=1)
		df.columns = ['series', 'benchmark']
		df = df.dropna()

		# Get the appropriate data structure for scikit learn
		clean_returns: pd.Series = df[df.columns.values[0]]
		clean_benchmarks = pd.DataFrame(df[df.columns.values[1]])

		# Fit a linear regression and get the beta
		regression = LinearRegression().fit(clean_benchmarks, y=clean_returns)
		beta = regression.coef_[0]

		# Compute Treynor ratio
		cagr = self.calculate_cagr(price_series)
		return (cagr - risk_free_rate) / beta