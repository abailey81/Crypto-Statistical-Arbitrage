"""
Trading Strategies Package
==========================

Implementation of crypto statistical arbitrage strategies.

Strategies
----------
1. Altcoin Pairs Trading (pairs_trading)
   - Cointegration-based mean reversion
   - Z-score signal generation
   - CEX and DEX pair support
   - Regime-aware enhancements

2. BTC Futures Curve Trading (futures_curve)
   - Term structure analysis
   - Calendar spread strategies
   - Cross-venue basis arbitrage
   - Roll optimization

3. Options Volatility Surface (vol_surface)
   - IV surface construction (SVI fitting)
   - Volatility arbitrage
   - Greeks-based strategies

Performance Targets
-------------------
- Sharpe Ratio: 1.5-2.5+
- Maximum Drawdown: <20%
- Correlation to BTC: <0.3
- Scalability: $5-50M AUM

Example Usage
-------------
>>> from strategies.pairs_trading import CointegrationAnalyzer, BaselinePairsStrategy
>>> from strategies.futures_curve import TermStructureAnalyzer, CalendarSpreadStrategy
>>> 
>>> # Pairs trading
>>> analyzer = CointegrationAnalyzer(significance_level=0.05)
>>> result = analyzer.engle_granger_test(series1, series2)
>>> 
>>> strategy = BaselinePairsStrategy(lookback=90, entry_z_cex=2.0)
>>> trades = strategy.generate_signals(prices_a, prices_b, hedge_ratio, pair)
>>> 
>>> # Futures curve
>>> ts_analyzer = TermStructureAnalyzer()
>>> curve = ts_analyzer.build_curve(futures_data, spot_price, '2024-01-01')
"""

__all__ = [
    'pairs_trading',
    'futures_curve',
    'vol_surface',
]

__version__ = '1.0.0'
