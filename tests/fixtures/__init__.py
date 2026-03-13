"""
Test Fixtures Package
=====================

Professional-quality mock data generators and test utilities for the
crypto statistical arbitrage data collection and strategy pipeline.

Package Overview
----------------
This package provides comprehensive synthetic data generation for testing
all components of the crypto statistical arbitrage system without requiring
live API connections or real market data.

Core Capabilities
-----------------
1. **Funding Rate Generation**:
   - Mean-reverting Ornstein-Uhlenbeck process
   - Regime-dependent parameters (bull/bear/neutral/crisis)
   - Cross-venue correlation structure
   - Asset-specific funding premiums
   - 8-hour (CEX) and 1-hour (hybrid) intervals

2. **OHLCV Generation**:
   - Geometric Brownian Motion with GARCH volatility
   - Jump-diffusion for extreme moves
   - U-shaped intraday volume patterns
   - Day-of-week seasonality effects

3. **Options Data Generation**:
   - Black-Scholes pricing
   - Realistic IV smile/skew
   - Term structure effects
   - Complete Greeks calculation

4. **DEX Pool Generation**:
   - Multi-chain support
   - Wash trading detection flags
   - TVL and volume distributions
   - Fee tier variety

5. **On-Chain Metrics**:
   - TVL time series
   - Liquidation events
   - Gas price patterns

Mathematical Framework
----------------------
Price Process (GBM with Jumps):

    dS/S = μdt + σdW + J × dN
    
    Where:
        μ = drift (regime-dependent)
        σ = volatility (GARCH clustered)
        J = jump size ~ LogNormal
        N = Poisson process

Funding Rate Process (Ornstein-Uhlenbeck):

    dF = κ(θ - F)dt + σ_F × dW
    
    Where:
        κ = 0.3 (mean reversion speed)
        θ = regime-dependent mean
        σ_F = funding volatility

Cross-Venue Correlation:

    F_venue = ρ × F_base + √(1-ρ²) × ε
    
    Where ρ ∈ [0.85, 0.95] for major venues

Volume Pattern (Intraday):

    V(h) = V_base × f_U(h) × f_dow(d) × f_regime(r) × ε
    
    f_U(h) = U-shaped function of hour
    f_dow(d) = day-of-week multiplier
    f_regime(r) = regime volume multiplier

Quality Scenarios
-----------------
The generators support injecting various data quality issues for
testing validation pipelines:

- MISSING_DATA: Random gaps in time series
- STALE_PRICE: Unchanged prices for extended periods
- OUTLIER: Extreme values outside normal range
- DUPLICATE: Repeated records
- NULL_VALUE: Missing values in fields
- WASH_TRADING: Suspicious volume/TVL patterns

Usage Examples
--------------
Quick Data Generation:

>>> from tests.fixtures import quick_funding_data, quick_ohlcv_data
>>> 
>>> # 30 days of funding rates
>>> funding_df = quick_funding_data(n_days=30, symbols=['BTC', 'ETH'])
>>> 
>>> # 30 days of hourly OHLCV
>>> ohlcv_df = quick_ohlcv_data(n_days=30, timeframe='1h')

Custom Configuration:

>>> from tests.fixtures import MockDataConfig, create_mock_dataset
>>> 
>>> config = MockDataConfig(
...     start_date=datetime(2024, 1, 1),
...     end_date=datetime(2024, 6, 1),
...     symbols=['BTC', 'ETH', 'SOL', 'ARB'],
...     venues=['binance', 'bybit', 'hyperliquid'],
...     seed=42,
...     enable_regime_switching=True,
...     inject_issues=False,
... )
>>> 
>>> funding_df = create_mock_dataset('funding', config)
>>> ohlcv_df = create_mock_dataset('ohlcv', config, timeframe='4h')

With Quality Issues for Validation Testing:

>>> config = MockDataConfig(
...     inject_issues=True,
...     issue_probability=0.02,
...     issues_to_inject=[
...         DataQualityIssue.MISSING_DATA,
...         DataQualityIssue.OUTLIER,
...     ],
... )
>>> df_with_issues = create_mock_dataset('funding', config)

Generator Classes:

>>> from tests.fixtures import (
...     MockFundingRateGenerator,
...     MockOHLCVGenerator,
...     MockOptionsDataGenerator,
...     MockDEXPoolGenerator,
... )
>>> 
>>> config = MockDataConfig()
>>> 
>>> funding_gen = MockFundingRateGenerator(config)
>>> funding_df = funding_gen.generate(include_open_interest=True)
>>> stats = funding_gen.get_stats(funding_df)
>>> 
>>> ohlcv_gen = MockOHLCVGenerator(config)
>>> ohlcv_df = ohlcv_gen.generate(timeframe='1h', include_vwap=True)

Module Structure
----------------
mock_data.py
├── Enumerations
│   ├── MarketRegime      - Market state classification
│   ├── VenueType         - Venue type properties
│   ├── DataQualityIssue  - Quality issue types
│   ├── AssetClass        - Asset classification
│   └── TimeframeType     - OHLCV timeframes
│
├── Configuration
│   ├── MockDataConfig    - Main configuration class
│   └── GeneratedDataStats - Statistics about generated data
│
├── Generators
│   ├── RegimeGenerator   - Market regime sequences
│   ├── PriceGenerator    - Price series with GARCH
│   ├── MockFundingRateGenerator  - Funding rates
│   ├── MockOHLCVGenerator        - OHLCV candles
│   ├── MockOptionsDataGenerator  - Options chains
│   ├── MockDEXPoolGenerator      - DEX pool data
│   └── MockOnChainGenerator      - On-chain metrics
│
├── Factory Functions
│   └── create_mock_dataset - Unified data creation
│
└── Convenience Functions
    ├── quick_funding_data  - Fast funding generation
    ├── quick_ohlcv_data    - Fast OHLCV generation
    ├── quick_options_data  - Fast options generation
    └── quick_pool_data     - Fast pool generation

Asset Parameters
----------------
Pre-configured parameters for 25+ assets including:

Majors: BTC, ETH
Large Cap: SOL, BNB, XRP, ADA, AVAX, DOT
DeFi: LINK, UNI, AAVE, MKR, CRV, LDO, GMX, DYDX
L2: ARB, OP, MATIC, IMX
Meme: DOGE, SHIB, PEPE, WIF, BONK

Each asset has:
- Base price and volatility
- Volume and open interest baselines
- Funding rate sensitivity
- Asset class classification

Venue Configurations
--------------------
Pre-configured parameters for venues:

CEX: binance, bybit, okx (8h funding)
Hybrid: hyperliquid, dydx, vertex (1h funding)
DEX: uniswap_v3, curve (no funding, pool-based)

Each venue has:
- Venue type and funding interval
- Supported symbols
- Volume multiplier
- Price offset (basis)

Author: Crypto StatArb Quantitative Research
Version: 2.0.0
"""

from __future__ import annotations

# =============================================================================
# CORE IMPORTS
# =============================================================================

from .mock_data import (
    # Enumerations
    MarketRegime,
    VenueType,
    DataQualityIssue,
    AssetClass,
    TimeframeType,
    
    # Configuration Classes
    MockDataConfig,
    GeneratedDataStats,
    
    # Generator Classes
    RegimeGenerator,
    PriceGenerator,
    MockFundingRateGenerator,
    MockOHLCVGenerator,
    MockOptionsDataGenerator,
    MockDEXPoolGenerator,
    MockOnChainGenerator,
    
    # Factory Function
    create_mock_dataset,
    
    # Convenience Functions
    quick_funding_data,
    quick_ohlcv_data,
    quick_options_data,
    quick_pool_data,
    
    # Constants
    ASSET_PARAMETERS,
    VENUE_CONFIGS,
)


# =============================================================================
# PACKAGE METADATA
# =============================================================================

__version__ = "2.0.0"
__author__ = "Crypto StatArb Quantitative Research"

__all__ = [
    # Version info
    '__version__',
    '__author__',
    
    # Enumerations
    'MarketRegime',
    'VenueType',
    'DataQualityIssue',
    'AssetClass',
    'TimeframeType',
    
    # Configuration
    'MockDataConfig',
    'GeneratedDataStats',
    
    # Generators
    'RegimeGenerator',
    'PriceGenerator',
    'MockFundingRateGenerator',
    'MockOHLCVGenerator',
    'MockOptionsDataGenerator',
    'MockDEXPoolGenerator',
    'MockOnChainGenerator',
    
    # Factory
    'create_mock_dataset',
    
    # Convenience
    'quick_funding_data',
    'quick_ohlcv_data',
    'quick_options_data',
    'quick_pool_data',
    
    # Constants
    'ASSET_PARAMETERS',
    'VENUE_CONFIGS',
]


# =============================================================================
# PACKAGE-LEVEL UTILITIES
# =============================================================================

def get_available_symbols() -> list:
    """Get list of all available symbols with parameters."""
    return list(ASSET_PARAMETERS.keys())


def get_available_venues() -> list:
    """Get list of all available venues with configs."""
    return list(VENUE_CONFIGS.keys())


def get_asset_info(symbol: str) -> dict:
    """Get parameter info for a specific asset."""
    if symbol not in ASSET_PARAMETERS:
        raise ValueError(f"Unknown symbol: {symbol}. Available: {get_available_symbols()}")
    return ASSET_PARAMETERS[symbol].copy()


def get_venue_info(venue: str) -> dict:
    """Get configuration info for a specific venue."""
    if venue not in VENUE_CONFIGS:
        raise ValueError(f"Unknown venue: {venue}. Available: {get_available_venues()}")
    return VENUE_CONFIGS[venue].copy()


def list_data_types() -> list:
    """List available data types for generation."""
    return ['funding', 'ohlcv', 'options', 'pools', 'tvl', 'liquidations']


def list_quality_issues() -> list:
    """List available quality issues for injection."""
    return [issue.value for issue in DataQualityIssue]


def list_market_regimes() -> list:
    """List available market regimes."""
    return [regime.value for regime in MarketRegime]


def list_timeframes() -> list:
    """List available OHLCV timeframes."""
    return [tf.value for tf in TimeframeType]


# Add utility functions to exports
__all__.extend([
    'get_available_symbols',
    'get_available_venues',
    'get_asset_info',
    'get_venue_info',
    'list_data_types',
    'list_quality_issues',
    'list_market_regimes',
    'list_timeframes',
])