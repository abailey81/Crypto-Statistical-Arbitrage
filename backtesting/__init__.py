"""
Backtesting Engine Package
==========================

Comprehensive backtesting framework for crypto statistical arbitrage strategies.

Modules
-------
backtest_engine : Core backtesting logic with venue-aware execution
walk_forward : Walk-forward optimization framework
performance_metrics : Sharpe, Sortino, drawdown, and custom metrics
crisis_analysis : Event-specific performance analysis
visualization : Charts, plots, and performance reports

Key Features
------------
1. Walk-Forward Framework:
   - Train period: 2022-01-01 to 2023-06-30
   - Test period: 2023-07-01 to 2024-12-31
   - Rolling optimization support

2. Venue-Aware Execution:
   - CEX vs DEX cost models
   - Realistic slippage simulation
   - Gas cost integration for DEX
   - Multi-venue routing

3. Performance Metrics:
   - Sharpe Ratio (annualized)
   - Sortino Ratio
   - Maximum Drawdown
   - Win Rate
   - Average Trade PnL
   - Transaction Cost Drag
   - Turnover Analysis

4. Crisis Analysis:
   - UST/Luna collapse (May 2022)
   - FTX bankruptcy (Nov 2022)
   - Bank crisis/USDC depeg (Mar 2023)
   - SEC lawsuits (Jun 2023)

5. Capacity Analysis:
   - Slippage impact modeling
   - Volume participation constraints
   - Venue-specific capacity limits

Benchmarks
----------
1. Buy-and-hold BTC spot
2. Naive roll (3 days before expiry)
3. Perpetual hold (Binance perp)
4. Strategy performance

Example Usage
-------------
>>> from backtesting import BacktestEngine, PerformanceMetrics
>>> 
>>> # Initialize engine
>>> engine = BacktestEngine(initial_capital=1_000_000)
>>> 
>>> # Run backtest
>>> result = engine.run(
...     trades=strategy_trades,
...     price_data=ohlcv_df,
...     test_start='2023-07-01',
...     test_end='2024-12-31'
... )
>>> 
>>> # Analyze results
>>> print(f"Sharpe: {result.sharpe_ratio:.2f}")
>>> print(f"Max DD: {result.max_drawdown:.2f}%")
"""

__all__ = [
    'BacktestEngine',
    'WalkForwardOptimizer',
    'PerformanceMetrics',
    'CrisisAnalyzer',
    'BacktestVisualization',
    'CapacityAnalyzer',
]
