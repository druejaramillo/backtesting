from typing import Tuple, List, Dict, Callable, NewType, Any, Iterable
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import timezone
import time
from Technical_Indicators import Indicators
from Portfolio_Metrics import Metrics
from Portfolio import PortfolioHistory, Position
from Execution import Executor
from collections import OrderedDict, defaultdict, namedtuple
from pypfopt import risk_models, expected_returns, objective_functions, EfficientFrontier, DiscreteAllocation
import riskfolio.HCPortfolio as hc

class Simulator(object):
    """
    A simple trading simulator to work with the PortfolioHistory class
    """

    def __init__(self, executor: Executor, all_ohlcv: dict, prices: pd.DataFrame, spy_prices: pd.Series, risk_free_rate: float, initial_cash: float=10000, percent_slippage: float=0.0005, training: bool=False):

        ### Set simulation parameters

        # Initial cash in porfolio
        # self.cash will fluctuate
        self.initial_cash = initial_cash
        self.cash = initial_cash

        # The percentage difference between closing price and fill price for the
        # position, to simulate adverse effects of market orders
        self.percent_slippage = percent_slippage

        # Keep track of live trades
        self.active_positions_by_symbol: Dict[str, Position] = OrderedDict()

        # Keep track of portfolio history like cash, equity, and positions
        if training:
            self.portfolio_history = PortfolioHistory(None, risk_free_rate)
        else:
            self.portfolio_history = PortfolioHistory(spy_prices, risk_free_rate)
        self.training = training

        # Set the executor
        self.executor = executor

        # Set the price data
        self.risk_free_rate = risk_free_rate
        self.all_ohlcv = all_ohlcv
        self.prices = prices
        self.spy_prices = spy_prices

        # Get list of symbols
        self.all_symbols = list(set(all_ohlcv.keys()))

        # Keep track of how long since the portfolio was last updated
        self.days_since_update = 0

    def reset(self):
        '''
        Reset the simulator. Only used for the optimizer's simulator between training runs.
        '''

        self.cash = self.initial_cash
        self.active_positions_by_symbol: Dict[str, Position] = OrderedDict()
        if self.training:
            self.portfolio_history = PortfolioHistory(None, self.risk_free_rate)
        else:
            self.portfolio_history = PortfolioHistory(self.spy_prices, self.risk_free_rate)

        self.days_since_update = 0

    @property
    def active_positions_count(self):
        return len(self.active_positions_by_symbol)

    @property
    def active_symbols(self) -> List[str]:
        return list(self.active_positions_by_symbol.keys())

    def print_initial_parameters(self):
        s = f'Initial Cash: ${self.initial_cash}\n' \
            f'Percent Slippage: {100 * self.percent_slippage:.2f}%\n'
        print(s)
        return s

    def update_portfolio(self, target_portfolio, date, prices, spy_prices, total_equity, **kwargs):
        '''
        Update the portfolio to try to match the target portfolio:
        - Close positions no longer in the target portfolio
        - Open positions not yet in the current portfolio
        - Update positions still in the target portfolio

        Inputs:
        - target_portfolio: list of tickers that we want in our new portfolio
        - prices: recent price history (everything up to current date)
        - spy_prices: recent SPY price history (dates matching prices)
        - total_equity: portfolio equity
        - **kwargs: sector constraints for mean-variance portfolio construction (if any)
        '''

        # Get the new portfolio layout
        shares_dict, leftover = self.executor.get_shares(target_portfolio, prices, spy_prices, total_equity, **kwargs)
        
        # Make sure the new portfolio won't have any positions with zero shares
        shares_dict = {x:y for x,y in shares_dict.items() if y != 0}

        # Classify the current portfolio tickers
        active_set = set(self.active_positions_by_symbol.keys())
        target_set = set(target_portfolio)
        symbols_to_sell = list(active_set - target_set)
        symbols_to_buy = list(target_set - active_set)
        symbols_to_update = list(target_set & active_set)

        # Sell certain holdings
        price = 0.0
        for symbol in symbols_to_sell:
            price = prices[symbol].iloc[-1]
            self.sell_to_close(symbol, date, price)

        # Update certain holdings
        buy = []
        sell = []
        for symbol in symbols_to_update:
            current_shares = self.active_positions_by_symbol[symbol].shares
            new_shares = shares_dict[symbol]
            if new_shares > current_shares:
                buy.append(symbol)
            elif new_shares < current_shares:
                sell.append(symbol)
            else:
                pass
        for symbol in sell: # Execute sell orders before buy orders (so we don't run out of cash)
            new_shares = shares_dict[symbol]
            self.partial_buy_sell(symbol, date, price, new_shares)
        for symbol in buy:
            new_shares = shares_dict[symbol]
            self.partial_buy_sell(symbol, date, price, new_shares)

        # Buy holdings
        for symbol in symbols_to_buy:
            price = prices[symbol].iloc[-1]
            shares = shares_dict[symbol]
            self.buy_to_open(symbol, date, price, shares)

    def buy_to_open(self, symbol, date, price, shares):
        """
        Keep track of new position, make sure it isn't an existing position. 
        Verify you have cash.
        """
        
        # Calculate buy_price
        purchase_price = (1 + self.percent_slippage) * price

        # Spend the cash
        while self.cash - purchase_price * shares < 0:
            shares -= 1
        self.cash -= purchase_price * shares
        assert self.cash >= 0, 'Spent cash you do not have.'

        # Record the position
        assert not symbol in self.active_positions_by_symbol, 'Symbol already in portfolio.'        
        position = Position(symbol, date, price, shares)
        self.active_positions_by_symbol[symbol] = position

    def partial_buy_sell(self, symbol, date, price, new_shares):
        '''
        Update a position that has been modified but not closed.
        '''

        # Record the change in cash
        position = self.active_positions_by_symbol[symbol]
        old_shares = position.shares
        order_price = 0.0
        if new_shares < old_shares:
            order_price = price * (1 - self.percent_slippage)
            self.cash += order_price * (old_shares - new_shares)
        else:
            order_price = price * (1 + self.percent_slippage)
            while self.cash - order_price * (new_shares - old_shares) < 0:
                new_shares -= 1
            self.cash -= order_price * (new_shares - old_shares)
            assert self.cash >= 0, 'Spent cash you do not have'

        # Update the position
        position.buy_sell(date, price, new_shares)

    def sell_to_close(self, symbol, date, price):
        """
        Keep track of exit price, recover cash, close position, and record it in
        portfolio history.
        Will raise a KeyError if symbol isn't an active position
        """

        position = self.active_positions_by_symbol[symbol]

        # Receive the cash
        sale_value = position.shares * price * (1 - self.percent_slippage)
        self.cash += sale_value

        # Exit the position
        position.exit(date, price)

        # Record in portfolio history
        self.portfolio_history.add_to_history(position)
        del self.active_positions_by_symbol[symbol]
    
    @staticmethod
    def _assert_equal_columns(*args: Iterable[pd.DataFrame]):
        column_names = set(args[0].columns.values)
        for arg in args[1:]:
            assert set(arg.columns.values) == column_names, \
                'Found unequal column names in input dataframes.'

    def simulate(self, start_idx, end_idx, strat, **kwargs):
        """
        Runs the simulation.
        Inputs:
        - start_date: date on which to start the simulation (inclusive)
        - end_date: date on which to end the simulation (exclusive)
        - strat: list of strategy components (parameters, signal function)
        - **kwargs: sector constraints for mean-variance portfolio construction (if any)
        """

        begin_time = time.time()

        # Pull out the data needed for this simulation run
        prices = self.prices.iloc[-start_idx:-end_idx].copy()
        spy_prices = self.spy_prices.iloc[-start_idx:-end_idx].copy()

        # Get the trading signals
        strat_params = strat[0]
        signal_func = strat[1]
        reduced_ohlcv = {}
        for symbol in self.all_symbols:
            reduced_ohlcv[symbol] = self.all_ohlcv[symbol].iloc[-start_idx:-end_idx].copy()
        signals = signal_func(strat_params, reduced_ohlcv)

        # Make sure the price and signal data line up
        self._assert_equal_columns(prices, signals)

        # Get the data we'll need when updating the portfolio
        # We'll add each day's data to these
        all_past_prices = self.prices.iloc[:-start_idx].copy()
        all_past_spy_prices = self.spy_prices.iloc[:-start_idx].copy()

        # Iterating over all dates.
        # Change to itertuples()
        # Make sure that current_prices is used correctly (it's a Series)
        total_dates = len(prices)
        i = 0
        for row in prices.itertuples():

            i += 1
            if i % 30 == 0 or i == 1 or i == total_dates:
                percent_done = i / float(total_dates)
                num_bars = int(percent_done * 50)
                progress_bar = '\t\t[' + '=' * (num_bars - 1) + '>' + '-' * (50 - num_bars) + '] {0}%'.format(round(100 * percent_done, 1))
                print(progress_bar, end='\r')

            # Extract data
            r = row._asdict()
            date = r['Index']
            r.pop('Index')
            current_prices = pd.Series(r)

            # Get the current signals too
            current_signals = signals.loc[date]

            # Update the data we'll need when updating the portfolio
            all_past_prices.loc[date] = current_prices
            all_past_spy_prices.loc[date] = self.spy_prices.loc[date]

            # Determine the target portfolio
            # Start with the current portfolio
            sell_signals = set(current_signals.index[current_signals.isin([-1])].tolist())
            buy_signals = set(current_signals.index[current_signals.isin([1])].tolist())
            current_portfolio = set(self.active_positions_by_symbol.keys())
            target_portfolio = current_portfolio - sell_signals | buy_signals # take out sell signals and add buy signals

            # Calculate the total equity in the portfolio
            total_equity = 0.0
            for symbol in list(self.active_positions_by_symbol.keys()):
                position = self.active_positions_by_symbol[symbol]
                total_equity += position.shares * current_prices.loc[symbol]
            total_equity += self.cash

            # Update the portfolio
            # Unless the target portfolio has a different set of tickers than our current portfolio,
            # we only want to update the portfolio every so often
            self.days_since_update += 1
            if target_portfolio == set(self.active_positions_by_symbol.keys()):
                if self.days_since_update >= 30:
                    self.days_since_update = 0
                    self.update_portfolio(target_portfolio, date, all_past_prices, all_past_spy_prices, total_equity, **kwargs)
            else:
                self.days_since_update = 0
                self.update_portfolio(target_portfolio, date, all_past_prices, all_past_spy_prices, total_equity, **kwargs)

            # Update price data everywhere
            for symbol in list(self.active_positions_by_symbol.keys()):
                price = current_prices.loc[symbol]
                position = self.active_positions_by_symbol[symbol]
                position.record_update(date, price)

            # Record current cash
            self.portfolio_history.record_cash(date, self.cash)

        end_time = time.time()

        print()
        print('\t\tSimulation time: {0} seconds'.format(round(end_time-begin_time, 2)))

    def end(self, end_idx):
        '''
        End the current simulation run.
        '''

        # Sell all positions and mark simulation as complete
        end_date = self.prices.index[-end_idx]
        for symbol in list(self.active_positions_by_symbol.keys()):
            self.sell_to_close(symbol, end_date, self.prices.loc[end_date][symbol])
        self.portfolio_history.record_cash(end_date, self.cash)
        self.portfolio_history.finish()