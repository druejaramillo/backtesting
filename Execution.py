import pandas as pd
from Portfolio_Metrics import Metrics
from pypfopt import risk_models, expected_returns, objective_functions, EfficientFrontier, DiscreteAllocation
import riskfolio as rp

class Executor(object):

    def __init__(self, c, risk_free_rate, method):

        self.c = c
        self.risk_free_rate = risk_free_rate
        self.method = method
        self.target_portfolio = None
        self.prices = None
        self.spy_prices = None
        self.returns = None
        self.num_tickers = None
        self.weights = None

    # Compute the number of whole shares to buy
    def get_shares(self, target_portfolio, prices, spy_prices, total_equity, **kwargs):

        # Set the optimal weights
        self.set_weights(target_portfolio, prices, spy_prices, **kwargs)

        # Get the number of shares
        latest_prices = self.prices.iloc[-1]
        weights_dict = self.weights.to_dict()
        da = DiscreteAllocation(weights_dict, latest_prices, total_portfolio_value=total_equity)
        if self.num_tickers < 15: # If we don't have too many tickers, we can use the better (but slower) method
            shares, leftover = da.lp_portfolio()
        else:
            shares, leftover = da.greedy_portfolio()

        return shares, leftover

    # Get the optimal weights for a given target portfolio using specified method
    def set_weights(self, target_portfolio, prices, spy_prices, **kwargs):
        
        # Set the target portfolio and tda-api client
        self.target_portfolio = target_portfolio

        # Get some initial data
        self.prices = prices

        self.spy_prices = spy_prices

        self.returns = Metrics().calculate_return_series(self.prices)

        self.num_tickers = len(target_portfolio)

        # Set the weights
        if self.method == 'equal':
            self.equal_weighting()
        elif self.method == 'min_vol' or self.method == 'max_sharpe':
            self.mean_variance_weighting(**kwargs)
        elif self.method == 'hrp':
            self.hrp_weighting()
        else:
            self.herc_weighting()

    # Compute the weights for equal portfolio weighting
    def equal_weighting(self):

        universal_weight = 1.0 / self.num_tickers
        target_list = list(self.target_portfolio)
        weights_dict = {}
        for i in range(self.num_tickers):
            weights_dict[target_list[i]] = universal_weight
        self.weights = pd.Series(weights_dict)

    # Compute the weights for mean-variance weighting
    def mean_variance_weighting(self, sector_constraints=None):

        cov = risk_models.CovarianceShrinkage(self.prices).ledoit_wolf()
        mu = expected_returns.capm_return(self.prices, market_prices=self.spy_prices, risk_free_rate=self.risk_free_rate)

        # Building the optimizer
        ef = EfficientFrontier(mu, cov)
        if sector_constraints != None: # Set sector constraints if any
            sector_mapper = sector_constraints[0]
            sector_lower = sector_constraints[1]
            sector_upper = sector_constraints[2]
            ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)

        # Choosing the optimum weights
        if self.method == 'min_vol':
            ef.add_objective(objective_functions.L2_reg, gamma=0.1) # Perform L2 regularization on weights
            ef.min_volatility()
        else:
            ef.max_sharpe()

        self.weights = pd.Series(ef.clean_weights()) # clean_weights() returns a dictionary

    # Compute the weights for Hierarchical Risk Parity weighting
    def hrp_weighting(self):

        # Building the optimizer
        portfolio = rp.HCPortfolio(returns=pd.DataFrame(self.returns).dropna())

        # Choosing the optimum weights
        weights = portfolio.optimization(model='HRP', covariance='ledoit', rf=self.risk_free_rate)

        self.weights = weights['weights'] # weights is a pandas DataFrame

    # Compute the weights for Hierarchical Equal Risk Contribution weighting
    def herc_weighting(self):

        # Building the optimizer
        portfolio = rp.HCPortfolio(returns=pd.DataFrame(self.returns).dropna())

        # Choosing the optimum weights
        weights = portfolio.optimization(model='HERC', covariance='ledoit', rf=self.risk_free_rate)

        self.weights = weights['weights'] # weights is a pandas DataFrame