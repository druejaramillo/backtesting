# Backtester

## Overview

This project is a robust, versatile, and performance-driven backtesting framework designed specifically for algorithmic trading strategies. It facilitates comprehensive analysis and optimization of your trading models, providing you with essential insights and confidence to proceed with a live trading environment.

## Algotrading Backtesting Framework

- **Single Backtest**: This feature enables you to perform a single pass backtest on your chosen trading strategy over a specified historical data period. This allows you to determine the strategy's effectiveness and profitability retrospectively, with the option to review detailed performance metrics.
- **Walk-Forward Optimization**: With the Walk-Forward Optimization feature, the framework splits the historical data into in-sample and out-of-sample periods. It optimizes the model parameters during the in-sample period, and subsequently validates the model on the out-of-sample period. This iterative process significantly helps in understanding the robustness of the trading strategy and its ability to adapt to new market conditions.
- **Genetic Algorithm Optimization**: This feature leverages evolutionary concepts of genetic algorithms to optimize your trading strategies. It manipulates the strategy parameters through processes mirroring natural selection, including mutation, crossover, and selection, to find an optimal or near-optimal set of strategy parameters.
- **Bayesian Optimization**: AlgoTradeBacktester also incorporates Bayesian Optimization, a sequential design strategy for global optimization of black-box functions. This probabilistic approach models the objective function, uses the model to select promising query points, and updates the model based on the new sample. This methodology effectively handles the trade-off between exploration and exploitation.

## How It Works

1. **Input Your Strategy**: Define your algorithmic trading strategy by specifying your entry, exit, risk management rules, and parameters that you would like to optimize.
2. **Choose Your Optimization Method**: Choose between Genetic Algorithm Optimization or Bayesian Optimization for your strategy. You can also choose not to optimize and perform a simple backtest.
3. **Backtesting/Walk-forward**: Depending on your needs, perform a single backtest or a walk-forward optimization.
4. **Review Your Results**: After the backtest or walk-forward optimization, the framework will provide you with detailed reports about the performance of your trading strategy, along with the optimized parameters if optimization was selected.

## Main Scripts
1. Backtester
	- This script is the "orchestrator" of the project.
	- It manages the other processes, including:
		- Gathering & processing data
		- Running the backtest
		- Outputting results

2. Execution
	- This script is responsible for portfolio management throughout the backtest.
	- It does the following two things:
		- Calculates how many shares to buy in a given trade
		- Calculates optimal portfolio weights each time we rebalance

3. Genetic_Algorithm
	- This script is responsible for optimizing a strategy's parameters using a genetic algorithm.
	- After creating an initial population of potential strategies, it does the following for several generations:
		1. Perform tournament selection
		2. Perform crossover & mutation
	- After several generations, it chooses the "fittest" strategy.

4. Portfolio_Metrics
	- This script contains the methods to calculate many different portfolio metrics, including:
		- CAGR
		- Annualized volatility
		- Annualized downside deviation
		- Max drawdown
		- Sharpe, Sortino, Calmar, & Treynor ratios
		- Jensen's alpha

5. Portfolio_Simulation
	- This script is responsible for taking a particular strategy and running a simulation on historical data in order to evaluate its performance.

6. Portfolio
	- This script contains the classes used to track individual positions and an overall portfolio during a simulation.

7. Technical_Indicators
	- This class contains the methods to calculate many different technical indicators on historical data, including:
		- SMAs & EMAs
		- VWAP
		- MACD
		- RSI
		- ATR
		- Bollinger Bands & Percent B
		- ADX
		- Stochastic RSI
		- Williams %R
		- Ultimate Oscillator
		- MFI
		- CCI

8. Testing
	- This is just a script running an example strategy.

## Getting Started

This project requires the following non-standard Python libraries:
1. `tda-api`
2. `pandas`
3. `pypfopt`
4. `riskfolio`
5. `numpy`
6. `sklearn`
7. `matplotlib`
8. `finta`