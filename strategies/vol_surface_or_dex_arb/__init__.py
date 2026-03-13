"""
Options Volatility Surface Strategy
====================================

Volatility surface analysis and arbitrage strategies for crypto options.

Strategy Overview
-----------------
Exploit mispricing in the implied volatility surface by comparing:
- Cross-strike arbitrage (butterfly/condor opportunities)
- Cross-expiry arbitrage (calendar vol arbitrage)
- IV vs realized vol relationships
- CEX vs DEX options pricing differences

Modules
-------
surface_construction : SVI fitting and IV surface building
vol_arbitrage : Volatility arbitrage strategies
greeks_analysis : Greeks-based portfolio management
rv_analysis : Realized volatility modeling

Key Features
------------
1. IV Surface Construction:
   - SVI (Stochastic Volatility Inspired) parameterization
   - No-arbitrage constraints
   - Multi-expiry surface interpolation

2. Arbitrage Detection:
   - Butterfly arbitrage (convexity violations)
   - Calendar arbitrage (forward vol violations)
   - Put-call parity violations
   - Cross-venue mispricing

3. Greeks Management:
   - Delta hedging
   - Vega neutral portfolios
   - Gamma scalping

4. DVOL Analysis:
   - Deribit Volatility Index (30-day forward IV)
   - DVOL vs realized vol relationship
   - Mean reversion in IV

Data Requirements
-----------------
- Deribit options chain (all strikes, all expiries)
- Implied volatility and Greeks
- Underlying BTC/ETH prices
- DVOL index history

Alternative Strategies
----------------------
If options data too sparse, substitute with:
1. Cross-DEX Arbitrage
2. Stablecoin Depeg Trading
3. DEX Liquidity Provision Analysis

Example Usage
-------------
>>> from strategies.vol_surface import (
...     VolatilitySurface,
...     SVIFitter,
...     VolArbitrage,
... )
>>> 
>>> # Build volatility surface
>>> surface = VolatilitySurface()
>>> surface.fit(options_chain_df, underlying_price)
>>> 
>>> # Find arbitrage opportunities
>>> arb = VolArbitrage(surface)
>>> opportunities = arb.scan_butterflies()
"""

__all__ = [
    'VolatilitySurface',
    'SVIFitter',
    'VolArbitrage',
    'GreeksAnalyzer',
    'RealizedVolEstimator',
]
