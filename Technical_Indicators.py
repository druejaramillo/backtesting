import pandas as pd
from finta import TA

class Indicators:

	def __init__(self, ohlcv):

		self.ohlcv = ohlcv
		self.ohlc = self.ohlcv.copy()
		self.ohlc.pop('volume')

	# Simple Moving Average 'SMA'
	def sma(self, period):

		return TA.SMA(self.ohlc, period)

	# Smoothed Simple Moving Average 'SSMA'
	def ssma(self, period):

		return TA.SSMA(self.ohlc, period)

	# Exponential Moving Average 'EMA'
	def ema(self, period):

		return TA.EMA(self.ohlc, period)

	# Double Exponential Moving Average 'DEMA'
	def dema(self, period):

		return TA.DEMA(self.ohlc, period)

	# Triple Exponential Moving Average 'TEMA'
	def tema(self, period):

		return TA.TEMA(self.ohlc, period)

	# Triangular Moving Average 'TRIMA'
	def trima(self, period):

		return TA.TRIMA(self.ohlc, period)

	# Volume Adjusted Moving Average 'VAMA'
	def vama(self, period):

		return TA.VAMA(self.ohlcv, period)

	# Kaufman's Adaptive Moving Average 'KAMA'
	def kama(self, er, ema_fast, ema_slow, period):

		return TA.KAMA(self.ohlc, er, ema_fast, ema_slow, period)

	# Zero Lag Exponential Moving Average 'ZLEMA'
	def zlema(self, period):

		return TA.ZLEMA(self.ohlc, period)

	# Weighted Moving Average 'WMA'
	def wma(self, period):

		return TA.WMA(self.ohlc, period)

	# Hull Moving Average 'HMA'
	def hma(self, period):

		return TA.HMA(self.ohlc, period)

	# Elastic Volume Moving Average 'EVWMA'
	def evwma(self, period):

		return TA.EVWMA(self.ohlcv, period)

	# Volume Weighted Average Price 'VWAP'
	def vwap(self):

		return TA.VWAP(self.ohlcv)

	# Fractal Adaptive Moving Average 'FRAMA'
	def frama(self, period, batch):

		return TA.FRAMA(self.ohlc, period, batch)

	# Moving Average Convergence Divergence 'MACD'
	def macd(self, period_fast, period_slow, signal):

		return TA.MACD(self.ohlc, period_fast, period_slow, signal)

	# Percentage Price Oscillator 'PPO'
	def ppo(self, period_fast, period_slow, signal):

		return TA.PPO(self.ohlc, period_fast, period_slow, signal)

	# Volume-Weighted MACD 'VW_MACD'
	def vw_macd(self, period_fast, period_slow, signal):

		return TA.VW_MACD(self.ohlc, period_fast, period_slow, signal)

	# Elastic-Volume weighted MACD 'EV_MACD'
	def ev_macd(self, period_fast, period_slow, signal):

		return TA.EV_MACD(self.ohlc, period_fast, period_slow, signal)

	# Rate-of-Change 'ROC'
	def roc(self, period):

		return TA.ROC(self.ohlc, period)

	# Volatility-Based Momentum 'VBM'
	def vbm(self, roc_period, atr_period):

		return TA.VBM(self.ohlc, roc_period, atr_period)

	# Relative Strenght Index 'RSI'
	def rsi(self, period):

		return TA.RSI(self.ohlc, period)

	# Dynamic Momentum Index 'DYMI'
	def dymi(self):

		return TA.DYMI(self.ohlc)

	# Average True Range 'ATR'
	def atr(self, period):

		return TA.ATR(self.ohlc, period)

	# Bollinger Bands 'BBANDS'
	def bbands(self, period, MA):

		return TA.BBANDS(self.ohlc, period, MA=MA)

	# Bollinger Bands Width 'BBWIDTH'
	def bbwidth(self, period, MA):

		return TA.BBWIDTH(self.ohlc, period, MA=MA)

	# Percent B 'PERCENT_B'
	def percent_b(self, period, MA):

		return TA.PERCENT_B(self.ohlc, period, MA=MA)

	# Directional Movement Indicator 'DMI'
	def dmi(self, period):

		return TA.DMI(self.ohlc, period)

	# Average Directional Index 'ADX'
	def adx(self, period):

		return TA.ADX(self.ohlc, period)

	# Fibonacci Pivot Points 'PIVOT_FIB'
	def pivot_fib(self):

		return TA.PIVOT_FIB(self.ohlc)

	# Stochastic Oscillator %K 'STOCH'
	def stoch(self, period):

		return TA.STOCH(self.ohlc, period)

	# Stochastic oscillator %D 'STOCHD'
	def stochd(self, period, stoch_period):

		return TA.STOCHD(self.ohlc, period, stoch_period)

	# Stochastic RSI 'STOCHRSI'
	def stochrsi(self, rsi_period, stoch_period):

		return TA.STOCHRSI(self.ohlc, rsi_period, stoch_period)

	# Williams %R 'WILLIAMS'
	def williams(self, period):

		return TA.WILLIAMS(self.ohlc, period)

	# Ultimate Oscillator 'UO'
	def uo(self):

		return TA.UO(self.ohlc)

	# Awesome Oscillator 'AO'
	def ao(self, slow_period, fast_period):

		return TA.AO(self.ohlc, slow_period, fast_period)

	# Vortex Indicator 'VORTEX'
	def vortex(self, period):

		return TA.VORTEX(self.ohlc, period)

	# Accumulation-Distribution Line 'ADL'
	def adl(self):

		return TA.ADL(self.ohlcv)

	# Chaikin Oscillator 'CHAIKIN'
	def chaikin(self):

		return TA.CHAIKIN(self.ohlcv)

	# Money Flow Index 'MFI'
	def mfi(self, period):

		return TA.MFI(self.ohlcv, period)

	# On Balance Volume 'OBV'
	def obv(self):

		return TA.OBV(self.ohlcv)

	# Volume Zone Oscillator 'VZO'
	def vzo(self, period):

		return TA.VZO(self.ohlcv, period)

	# Commodity Channel Index 'CCI'
	def cci(self, period):

		return TA.CCI(self.ohlc, period)

	# Chande Momentum Oscillator 'CMO'
	def cmo(self, period):

		return TA.CMO(self.ohlc, period)

	# Wave Trend Oscillator 'WTO'
	def wto(self, channel_length, average_length):

		return TA.WTO(self.ohlc, channel_length, average_length)

	# Fisher Transform 'FISH'
	def fish(self, period):

		return TA.FISH(self.ohlc, period)

	# Ichimoku Cloud 'ICHIMOKU'
	def ichimoku(self, tenkan_period, kijun_period, senkou_period, chikou_period):

		return TA.ICHIMOKU(self.ohlc, tenkan_period, kijun_period, senkou_period, chikou_period)

	# Volume Flow Indicator 'VFI'
	def vfi(self, peirod):

		return TA.VFI(self.ohlcv, period)

	# Moving Standard deviation 'MSD'
	def msd(self, period):

		return TA.MSD(self.ohlc, period)

	# Mark Whistler's WAVE PM 'WAVEPM'
	def wavepm(self, period, lookback_period):

		return TA.WAVEPM(self.ohlc, period, lookback_period)