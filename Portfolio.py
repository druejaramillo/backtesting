import tda
from tda.client import Client
import pandas as pd
import matplotlib.pyplot as plt
import math
from typing import Tuple, List, Dict, Callable, NewType, Any
from collections import OrderedDict, defaultdict
from Portfolio_Metrics import Metrics
from texttable import Texttable
import subprocess

DATE_FORMAT_STR = '%a %b %d, %Y'

def _pdate(date: pd.Timestamp):
    """Pretty-print a datetime with just the date"""
    return date.strftime(DATE_FORMAT_STR)

class Position(object):
    """
    A simple object to hold and manipulate data related to long stock trades.
    Allows a buy and sell operations on an asset.
    The __init__ method is equivalent to a buy-to-open operation. The exit
    method is a sell-to-close operation. The buy_sell method is an intermediate
    buy/sell operation depending on the change in number of shares.
    """

    def __init__(self, symbol: str, entry_date: pd.Timestamp,
                 entry_price: float, shares: int):
        """
        Equivalent to buying a certain number of shares of the asset
        """

        # Recorded on initialization
        self.entry_date = entry_date

        assert entry_price > 0, 'Cannot buy asset with zero or negative price.'
        self.entry_price = entry_price

        assert shares > 0, 'Cannot buy zero or negative shares.'
        self.shares = shares

        self.symbol = symbol

        # Recorded on position exit
        self.exit_date: pd.Timestamp = None
        self.exit_price: float = None

        # For easily getting current portfolio value
        self.last_date: pd.Timestamp = None
        self.last_price: float = None

        # Updated intermediately
        self.shares_series = pd.Series()
        self.price_series = pd.Series()

        self.shares_series.loc[entry_date] = self.shares
        self.price_series.loc[entry_date] = self.entry_price

    def buy_sell(self, date, price, new_shares):
        '''
        Equivalent to intermediate buy/sell order
        '''
        assert date != self.last_date, 'Bought/sold security in the same day.'
        self.shares = new_shares
        self.record_update(date, price)

    def exit(self, exit_date, exit_price):
        """
        Equivalent to selling a stock holding
        """
        assert self.entry_date != exit_date, 'Churned a position same-day.'
        assert not self.exit_date, 'Position already closed.'
        self.shares = 0
        self.record_update(exit_date, exit_price)
        self.exit_date = exit_date
        self.exit_price = exit_price

    def record_update(self, date, price):
        """
        Stateless function to record intermediate prices and values of existing positions
        """
        self.last_date = date
        self.last_price = price
        self.shares_series[date] = self.shares
        self.price_series[date] = price

    @property
    def last_value(self) -> float:
        return self.last_price * self.shares_series.iloc[-1]

    @property
    def is_active(self) -> bool:
        return self.exit_date is None

    @property
    def is_closed(self) -> bool:
        return not self.is_active

    @property
    def value_series(self) -> pd.Series:
        """
        Returns the value of the position over time. Ignores self.exit_date.
        Used in calculating the equity curve.
        """
        assert self.is_closed, 'Position must be closed to access this property'
        return self.shares_series.mul(self.price_series, fill_value=0.0)

    @property
    def percent_return(self) -> float:
        return (self.exit_price / self.entry_price) - 1

    @property
    def entry_value(self) -> float:
        return self.shares_series.iloc[0] * self.entry_price

    @property
    def exit_value(self) -> float:
        return self.shares_series.iloc[-1] * self.exit_price

    @property
    def change_in_value(self) -> float:
        return self.exit_value - self.entry_value

    @property
    def trade_length(self):
        return len(self.price_series) - 1

    def print_position_summary(self):
        _entry_date = _pdate(self.entry_date)
        _exit_date = _pdate(self.exit_date)
        _days = self.trade_length

        _entry_price = round(self.entry_price, 2)
        _exit_price = round(self.exit_price, 2)

        _entry_value = round(self.entry_value, 2)
        _exit_value = round(self.exit_value, 2)

        _return = round(100 * self.percent_return, 1)
        _diff = round(self.change_in_value, 2)

        print(f'{self.symbol:<5}     Trade summary')
        print(f'Date:     {_entry_date} -> {_exit_date} [{_days} days]')
        print(f'Price:    ${_entry_price} -> ${_exit_price} [{_return}%]')
        print(f'Value:    ${_entry_value} -> ${_exit_value} [${_diff}]')
        print()

    def __hash__(self):
        """
        A unique position will be defined by a unique combination of an 
        entry_date and symbol, in accordance with our constraints regarding 
        duplicate, variable, and compound positions
        """
        return hash((self.entry_date, self.symbol))


class PortfolioHistory(object):
    """
    Holds Position objects and keeps track of portfolio variables.
    Produces summary statistics.
    """

    def __init__(self, spy, risk_free_rate):
        # Keep track of positions, recorded in this list after close
        self.position_history: List[Position] = []
        self._logged_positions: Set[Position] = set()

        # Keep track of the last seen date
        self.last_date: pd.Timestamp = pd.Timestamp.min

        # Readonly fields
        self._cash_history: Dict[pd.Timestamp, float] = dict()
        self._simulation_finished = False

        # Set SPY data if provided
        self.spy = spy

        # Set a risk-free rate
        self.risk_free_rate = risk_free_rate

    def add_to_history(self, position: Position):
        assert not position in self._logged_positions, 'Recorded the same position twice.'
        assert position.is_closed, 'Position is not closed.'
        self._logged_positions.add(position)
        self.position_history.append(position)
        self.last_date = max(self.last_date, position.last_date)

    def record_cash(self, date, cash):
        self._cash_history[date] = cash
        self.last_date = max(self.last_date, date)

    @staticmethod
    def _as_oseries(d: Dict[pd.Timestamp, Any]) -> pd.Series:
        return pd.Series(d).sort_index()

    def _compute_cash_series(self):
        self._cash_series = self._as_oseries(self._cash_history)

    @property
    def cash_series(self) -> pd.Series:
        return self._cash_series

    def _compute_portfolio_value_series(self):
        value_by_date = defaultdict(float)
        last_date = self.last_date

        # Add up value of assets
        for position in self.position_history:
            for date, value in position.value_series.items():
                value_by_date[date] += value

        # Make sure all dates in cash_series are present
        for date in self.cash_series.index:
            value_by_date[date] += 0

        self._portfolio_value_series = self._as_oseries(value_by_date)

    @property
    def portfolio_value_series(self):
        return self._portfolio_value_series

    def _compute_equity_series(self):
        c_series = self.cash_series
        p_series = self.portfolio_value_series
        assert all(c_series.index == p_series.index), \
            'portfolio_series has dates not in cash_series'
        self._equity_series = c_series + p_series

    @property
    def equity_series(self):
        return self._equity_series

    @property
    def return_series(self):
        return Metrics().calculate_return_series(self.equity_series)

    def _assert_finished(self):
        assert self._simulation_finished, \
            'Simulation must be finished by running self.finish() in order ' + \
            'to access this method or property.'

    def finish(self):
        """
        Notate that the simulation is finished and compute readonly values
        """
        self._simulation_finished = True
        self._compute_cash_series()
        self._compute_portfolio_value_series()
        self._compute_equity_series()
        self._assert_finished()

    def compute_portfolio_size_series(self) -> pd.Series:
        size_by_date = defaultdict(int)
        for position in self.position_history:
            for date in position.value_series.index:
                size_by_date[date] += 1
        return self._as_oseries(size_by_date)

    @property
    def spy_return_series(self):
        return Metrics().calculate_return_series(self.spy['close'])
    
    @property
    def percent_return(self):
        return Metrics().calculate_percent_return(self.equity_series)

    @property
    def spy_percent_return(self):
        return Metrics().calculate_percent_return(self.spy['close'])

    @property
    def cagr(self):
        return Metrics().calculate_cagr(self.equity_series)

    @property
    def volatility(self):
        return Metrics().calculate_annualized_volatility(self.return_series)

    @property
    def sharpe_ratio(self):
        return Metrics().calculate_sharpe_ratio(self.equity_series, self.risk_free_rate)

    @property
    def spy_cagr(self):
        return Metrics().calculate_cagr(self.spy['close'])

    @property
    def spy_volatility(self):
        return Metrics().calculate_annualized_volatility(self.spy_return_series)
    
    @property
    def spy_sharpe_ratio(self):
        return Metrics().calculate_sharpe_ratio(self.spy['close'], self.risk_free_rate)

    @property
    def jensens_alpha(self):
        return Metrics().calculate_jensens_alpha(self.return_series, self.spy_return_series)

    @property
    def sortino_ratio(self):
        return Metrics().calculate_sortino_ratio(self.equity_series, self.risk_free_rate)
    
    @property
    def spy_sortino_ratio(self):
        return Metrics().calculate_sortino_ratio(self.spy['close'], self.risk_free_rate)

    @property
    def calmar_ratio(self):
        return Metrics().calculate_calmar_ratio(self.equity_series)
    
    @property
    def spy_calmar_ratio(self):
        return Metrics().calculate_calmar_ratio(self.spy['close'])

    @property
    def treynor_ratio(self):
        return Metrics().calculate_treynor_ratio(self.equity_series, self.spy['close'], self.risk_free_rate)

    @property
    def dollar_max_drawdown(self):
        return Metrics().calculate_max_drawdown(self.equity_series, 'dollar')

    @property
    def spy_dollar_max_drawdown(self):
        return Metrics().calculate_max_drawdown(self.spy['close'], 'dollar')

    @property
    def percent_max_drawdown(self):
        return Metrics().calculate_max_drawdown(self.equity_series, 'percent')

    @property
    def spy_percent_max_drawdown(self):
        return Metrics().calculate_max_drawdown(self.spy['close'], 'percent')
    
    @property
    def number_of_trades(self):
        return len(self.position_history)

    @property
    def average_active_trades(self):
        return self.compute_portfolio_size_series().mean()

    @property
    def average_trade_length(self):
        sum_length = 0.0
        for position in self.position_history:
            sum_length += position.trade_length
        avg_length = sum_length / len(self.position_history)
        return avg_length

    @property
    def final_cash(self):
        self._assert_finished()
        return self.cash_series.iloc[-1]

    @property
    def final_equity(self):
        self._assert_finished()
        return self.equity_series.iloc[-1]

    @property
    def spy_final_equity(self):
        self._assert_finished()
        initial_cash = self.cash_series.iloc[0]
        shares = math.floor(initial_cash / self.spy['close'].iloc[0])
        remaining_cash = initial_cash - shares * self.spy['close'].iloc[0]
        return shares * self.spy['close'].iloc[-1] + remaining_cash
    
    _PERFORMANCE_METRICS_PROPS = [
        'percent_return',
        'spy_percent_return',
        'cagr',
        'volatility',
        'spy_cagr',
        'spy_volatility',
        'sharpe_ratio',
        'spy_sharpe_ratio',
        'jensens_alpha',
        'sortino_ratio',
        'spy_sortino_ratio',
        'calmar_ratio',
        'spy_calmar_ratio',
        'treynor_ratio',
        'dollar_max_drawdown',
        'spy_dollar_max_drawdown',
        'percent_max_drawdown',
        'spy_percent_max_drawdown',
        'number_of_trades',
        'average_active_trades',
        'average_trade_length',
        'final_cash',
        'final_equity',
        'spy_final_equity',
    ]

    PerformancePayload = NewType('PerformancePayload', Dict[str, float])

    def get_performance_metric_data(self) -> PerformancePayload:
        props = self._PERFORMANCE_METRICS_PROPS
        return {prop: getattr(self, prop) for prop in props}

    def print_position_summaries(self):
        for position in self.position_history:
            position.print_position_summary()

    def print_summary(self):

        self._assert_finished()

        header = ['', 'Portfolio', 'SPY']

        final_equity = ['Final Equity', f'${self.final_equity:.2f}', f'${self.spy_final_equity:.2f}']
        final_cash = ['Final Cash', f'${self.final_cash:.2f}', 'N/A']
        percent_return = ['% Return', f'{100 * self.percent_return:.2f}%', f'{100 * self.spy_percent_return:.2f}%']
        cagr = ['CAGR', f'{100 * self.cagr:.2f}%', f'{100 * self.spy_cagr:.2f}%']
        volatility = ['Ann. Volatility', f'{100 * self.volatility:.2f}%', f'{100 * self.spy_volatility:.2f}%']
        sharpe_ratio = ['Sharpe Ratio', f'{self.sharpe_ratio:.2f}', f'{self.spy_sharpe_ratio:.2f}']
        sortino_ratio = ['Sortino Ratio', f'{self.sortino_ratio:.2f}', f'{self.spy_sortino_ratio:.2f}']
        calmar_ratio = ['Calmar Ratio', f'{self.calmar_ratio:.2f}', f'{self.spy_calmar_ratio:.2f}']
        treynor_ratio = ['Treynor Ratio', f'{self.treynor_ratio:.2f}', 'N/A']
        jensens_alpha = ['Jensen\'s Alpha', f'{self.jensens_alpha:.6f}', 'N/A']
        dollar_max_drawdown = ['Max DD ($)', f'${self.dollar_max_drawdown:.2f}', f'${self.spy_dollar_max_drawdown:.2f}']
        percent_max_drawdown = ['Max DD (%)', f'{100 * self.percent_max_drawdown:.2f}%', f'{100 * self.spy_percent_max_drawdown:.2f}%']
        number_of_trades = ['# of Trades', f'{self.number_of_trades}', 'N/A']
        average_active_trades = ['Average # of Active Trades', f'{self.average_active_trades}', 'N/A']
        average_trade_length = ['Average Trade Length', f'{self.average_trade_length} days', 'N/A']

        t = Texttable(max_width = 0)
        t.add_rows([header, final_equity, final_cash, percent_return, cagr, volatility, sharpe_ratio, sortino_ratio, calmar_ratio, treynor_ratio, jensens_alpha, dollar_max_drawdown, percent_max_drawdown, number_of_trades, average_active_trades, average_trade_length])
        file = 'portfolio_metrics.txt'
        with open(file, "w") as writer:
            writer.write(t.draw())
        subprocess.call(['open', '-a', 'TextEdit', file])

    def plot(self, show=True) -> plt.Figure:
        """
        Plots equity, cash and portfolio value curves.
        """
        self._assert_finished()

        figure, axes = plt.subplots(nrows=3, ncols=1)
        figure.tight_layout(pad=3.0)
        axes[0].plot(self.equity_series)
        axes[0].set_title('Equity')
        axes[0].grid()

        axes[1].plot(self.cash_series)
        axes[1].set_title('Cash')
        axes[1].grid()

        axes[2].plot(self.portfolio_value_series)
        axes[2].set_title('Portfolio Value')
        axes[2].grid()

        if show:
            plt.show()

        return figure

    def plot_benchmark_comparison(self, show=True) -> plt.Figure:
        """
        Plot comparable investment in the S&P 500.
        """
        self._assert_finished()

        equity_curve = self.equity_series
        ax = equity_curve.plot()

        spy_closes = self.spy['close']
        initial_cash = self.cash_series[0]
        initial_spy = spy_closes[0]

        scaled_spy = spy_closes * (initial_cash / initial_spy)
        scaled_spy.plot()

        baseline = pd.Series(initial_cash, index=equity_curve.index)
        ax = baseline.plot(color='black')
        ax.grid()

        ax.legend(['Equity curve', 'S&P 500 portfolio'])

        if show:
            plt.show()