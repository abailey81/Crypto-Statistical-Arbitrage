"""
Data Cleaning Module for Crypto Statistical Arbitrage Systems.

This module provides professional-quality data cleaning infrastructure
specifically designed for multi-venue cryptocurrency market data. Unlike
validation modules that DETECT issues, this module FIXES them.

==============================================================================
CRITICAL DISTINCTION
==============================================================================

    VALIDATION (data_validator.py, quality_checks.py):
        - DETECTS issues
        - Returns ValidationResult with issues list
        - Does NOT modify data
        
    CLEANING (this module):
        - FIXES issues detected by validation
        - Returns cleaned DataFrame + audit trail
        - Actually modifies data with full transparency

==============================================================================
CLEANING PHILOSOPHY FOR STATISTICAL ARBITRAGE
==============================================================================

Why Cleaning Matters for Stat Arb:
    
    1. SIGNAL INTEGRITY: Dirty data â†’ false signals â†’ bad trades
       - Outlier funding rate might look like arbitrage opportunity
       - In reality: data error, not opportunity
       
    2. BACKTEST ACCURACY: Garbage in â†’ misleading performance
       - Missing data causes look-ahead bias
       - Duplicates inflate volume/correlation metrics
       
    3. CROSS-VENUE CONSISTENCY: Compare apples to apples
       - Binance uses BTCUSDT, Hyperliquid uses BTC
       - Without normalization, pairs analysis fails

Pipeline Philosophy:
    
    RAW DATA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–º CLEAN DATA
         â”‚ â”‚
         â–¼ â–¼
    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”
    â”‚ Schema â”‚â”€â”€â–ºâ”‚ Dedup â”‚â”€â”€â–ºâ”‚ Temporalâ”‚â”€â”€â–ºâ”‚ Outlier â”‚â”€â”€â–ºâ”‚ Missing â”‚
    â”‚ Enforce â”‚ â”‚ Remove â”‚ â”‚ Align â”‚ â”‚ Treat â”‚ â”‚ Impute â”‚
    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
         â”‚ â”‚ â”‚ â”‚ â”‚
         â–¼ â–¼ â–¼ â–¼ â–¼
      LOG ALL LOG ALL LOG ALL LOG ALL LOG ALL
      CHANGES CHANGES CHANGES CHANGES CHANGES

==============================================================================
CLEANING STAGES REFERENCE
==============================================================================

Stage 1 - Schema Enforcement:
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ Operation â”‚ Description â”‚ Impact â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ Type Coercion â”‚ Convert columns to expected dtypes â”‚ Computation â”‚
â”‚ Column Mapping â”‚ Rename venue-specific columns â”‚ Consistency â”‚
â”‚ Required Fields â”‚ Add missing columns with NaN â”‚ Schema Valid â”‚
â”‚ Timestamp Parse â”‚ Convert to datetime64[ns, UTC] â”‚ Time Series â”‚
â”‚ Encoding Fix â”‚ Handle UTF-8 issues in symbols â”‚ String Ops â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

Stage 2 - Deduplication:
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ Strategy â”‚ When to Use â”‚ Behavior â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ keep_first â”‚ Time-ordered data â”‚ Drop later â”‚
â”‚ keep_last â”‚ Corrections supersede originals â”‚ Drop earlier â”‚
â”‚ keep_best â”‚ Quality-based selection â”‚ Keep most data â”‚
â”‚ aggregate â”‚ Multiple valid points per period â”‚ Mean/median â”‚
â”‚ remove_all â”‚ Cannot determine which is correct â”‚ Conservative â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

Stage 3 - Temporal Alignment:
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ Method â”‚ Description â”‚ Use Case â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ floor â”‚ Round down to interval boundary â”‚ No look-ahead â”‚
â”‚ ceil â”‚ Round up to interval boundary â”‚ Forward fill â”‚
â”‚ round â”‚ Round to nearest interval â”‚ Balanced â”‚
â”‚ snap_settlement â”‚ Snap to exchange settlement times â”‚ Funding rates â”‚
â”‚ resample â”‚ Aggregate to new frequency â”‚ Downsampling â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

Stage 4 - Outlier Treatment:
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ Action â”‚ Description â”‚ When to Use â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ flag â”‚ Add is_outlier column, keep data â”‚ Analysis/debug â”‚
â”‚ cap â”‚ Winsorize to percentile bounds â”‚ reliable signals â”‚
â”‚ remove â”‚ Delete outlier rows â”‚ Clean dataset â”‚
â”‚ interpolate â”‚ Replace with interpolated value â”‚ Continuity â”‚
â”‚ median_replace â”‚ Replace with rolling median â”‚ Smoothing â”‚
â”‚ cross_venue â”‚ Use value from correlated venue â”‚ Multi-source â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

Stage 5 - Missing Data Handling:
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ Method â”‚ Description â”‚ Limitations â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ forward_fill â”‚ Propagate last valid observation â”‚ Max N periods â”‚
â”‚ backward_fill â”‚ Use next valid observation â”‚ Look-ahead bias â”‚
â”‚ interpolate â”‚ Linear/spline interpolation â”‚ Gap size limit â”‚
â”‚ cross_venue â”‚ Fill from correlated venue â”‚ Correlation req â”‚
â”‚ seasonal â”‚ Use same period previous day/week â”‚ Seasonality req â”‚
â”‚ model_based â”‚ Predict from related features â”‚ Feature avail â”‚
â”‚ drop â”‚ Remove rows with missing values â”‚ Data loss â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

==============================================================================
CRYPTO-SPECIFIC CLEANING RULES
==============================================================================

Funding Rates:
    - Valid range: [-10%, +10%] per period (extreme but possible)
    - Typical range: [-0.1%, +0.1%] per period 
    - Outlier threshold: |rate| > 1% per 8h period â†’ flag for review
    - Zero is valid (market equilibrium)
    - Sign convention: positive = longs pay shorts
    
OHLCV Invariants:
    - high >= max(open, close) ALWAYS
    - low <= min(open, close) ALWAYS
    - high >= low ALWAYS
    - All prices > 0 ALWAYS
    - volume >= 0 ALWAYS
    - If violated â†’ FIX using OHLC relationships
    
Options Greeks:
    - delta âˆˆ [-1, 1] for standard options
    - gamma >= 0 always
    - vega >= 0 always 
    - theta typically negative for long options
    - IV âˆˆ [1%, 500%] typical, [0.1%, 1000%] extreme

DEX Pools:
    - TVL > 0 for active pools
    - volume_24h / TVL < 50 (else likely wash trading)
    - fee_tier âˆˆ {0.01%, 0.05%, 0.3%, 1%} for Uniswap V3
    
Open Interest:
    - OI >= 0 always
    - OI changes > 50% in single period â†’ flag for review
    - OI should correlate with volume

==============================================================================
USAGE EXAMPLES
==============================================================================

Basic Cleaning:
    >>> from data_cleaner import DataCleaner
    >>> cleaner = DataCleaner()
    >>> cleaned_df, report = cleaner.clean(df, data_type='funding_rates', venue='binance')
    >>> print(report.summary())
    
Pipeline with Custom Configuration:
    >>> from data_cleaner import CleaningPipeline, OutlierAction
    >>> pipeline = CleaningPipeline()
    >>> pipeline.configure('outlier_treatment', action=OutlierAction.CAP, threshold=3.0)
    >>> pipeline.configure('missing_data', method='forward_fill', max_periods=3)
    >>> result = pipeline.execute(df)
    
Cross-Venue Cleaning:
    >>> from data_cleaner import CrossVenueCleaner
    >>> cleaner = CrossVenueCleaner()
    >>> cleaned, reports = cleaner.clean_and_align(
    ... {'binance': df_binance, 'hyperliquid': df_hyperliquid},
    ... data_type='funding_rates',
    ... reference_venue='binance'
    ... )

Version: 1.0.0
"""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set, Type
import hashlib

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# GPU ACCELERATION (Optional - falls back to CPU automatically)
# =============================================================================
try:
    from .gpu_accelerator import (
        GPUAccelerator,
        get_accelerator,
        is_gpu_available,
    )
    GPU_AVAILABLE = is_gpu_available()
    if GPU_AVAILABLE:
        logger.info("GPU acceleration available for data cleaning")
except ImportError:
    GPU_AVAILABLE = False
    GPUAccelerator = None
    get_accelerator = None
    is_gpu_available = lambda: False

# =============================================================================
# ENUMERATIONS
# =============================================================================

class DataType(Enum):
    """
    Type of market data being cleaned.

    Matches data_validator.py DataType for seamless integration.
    Each type has specific cleaning rules and validation constraints.

    IMPORTANT: Keep synchronized with base_collector.py and data_validator.py DataType enums.
    """
    FUNDING_RATES = 'funding_rates'
    OHLCV = 'ohlcv'
    OPEN_INTEREST = 'open_interest'
    LIQUIDATIONS = 'liquidations'
    ORDERBOOK = 'orderbook'
    TRADES = 'trades'
    OPTIONS = 'options'
    OPTIONS_CHAIN = 'options_chain' # Alias for OPTIONS
    DEX_POOLS = 'dex_pools'
    DEX_SWAPS = 'dex_swaps'
    TVL = 'tvl'
    ONCHAIN_FLOWS = 'onchain_flows'
    SOCIAL_METRICS = 'social_metrics'
    SOCIAL_SENTIMENT = 'social_sentiment'
    TERM_STRUCTURE = 'term_structure'
    ON_CHAIN = 'on_chain'
    
    @property
    def required_columns(self) -> List[str]:
        """Minimum required columns for this data type."""
        return {
            DataType.FUNDING_RATES: ['timestamp', 'symbol', 'funding_rate'],
            DataType.OHLCV: ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'],
            DataType.OPEN_INTEREST: ['timestamp', 'symbol', 'open_interest'],
            DataType.LIQUIDATIONS: ['timestamp', 'symbol', 'side', 'size'],
            DataType.ORDERBOOK: ['timestamp', 'symbol', 'bids', 'asks'],
            DataType.TRADES: ['timestamp', 'symbol', 'price', 'size', 'side'],
            DataType.OPTIONS: ['timestamp', 'symbol', 'strike', 'expiry', 'option_type'],
            DataType.DEX_POOLS: ['timestamp', 'pool_address', 'tvl_usd'],
            DataType.TERM_STRUCTURE: ['timestamp', 'symbol', 'expiry', 'mark_price'],
            DataType.ON_CHAIN: ['timestamp', 'address', 'value'],
        }.get(self, ['timestamp'])
    
    @property
    def optional_columns(self) -> List[str]:
        """Optional but recommended columns."""
        return {
            DataType.FUNDING_RATES: ['mark_price', 'index_price', 'open_interest', 'venue'],
            DataType.OHLCV: ['volume_usd', 'trade_count', 'vwap', 'venue'],
            DataType.OPEN_INTEREST: ['mark_price', 'open_interest_usd', 'venue'],
            DataType.OPTIONS: ['mark_price', 'mark_iv', 'delta', 'gamma', 'vega', 'theta', 'venue'],
            DataType.DEX_POOLS: ['token0', 'token1', 'volume_24h_usd', 'fee_tier', 'chain'],
            DataType.TERM_STRUCTURE: ['index_price', 'basis', 'basis_pct', 'open_interest', 'venue'],
        }.get(self, ['venue'])
    
    @property
    def numeric_columns(self) -> List[str]:
        """Columns that must be numeric."""
        return {
            DataType.FUNDING_RATES: ['funding_rate', 'mark_price', 'index_price', 'open_interest'],
            DataType.OHLCV: ['open', 'high', 'low', 'close', 'volume', 'volume_usd'],
            DataType.OPEN_INTEREST: ['open_interest', 'mark_price', 'open_interest_usd'],
            DataType.LIQUIDATIONS: ['size', 'price', 'size_usd'],
            DataType.TRADES: ['price', 'size', 'size_usd'],
            DataType.OPTIONS: ['strike', 'mark_price', 'mark_iv', 'delta', 'gamma', 'vega', 'theta'],
            DataType.DEX_POOLS: ['tvl_usd', 'volume_24h_usd', 'fee_tier', 'reserve0', 'reserve1'],
            DataType.TERM_STRUCTURE: ['mark_price', 'index_price', 'basis', 'basis_pct', 'open_interest'],
        }.get(self, [])
    
    @property
    def primary_key(self) -> List[str]:
        """Columns that form the natural primary key."""
        return {
            DataType.FUNDING_RATES: ['timestamp', 'symbol', 'venue'],
            DataType.OHLCV: ['timestamp', 'symbol', 'venue'],
            DataType.OPEN_INTEREST: ['timestamp', 'symbol', 'venue'],
            DataType.LIQUIDATIONS: ['timestamp', 'symbol', 'venue', 'trade_id'],
            DataType.ORDERBOOK: ['timestamp', 'symbol', 'venue'],
            DataType.TRADES: ['timestamp', 'symbol', 'venue', 'trade_id'],
            DataType.OPTIONS: ['timestamp', 'instrument_name', 'venue'],
            DataType.DEX_POOLS: ['timestamp', 'pool_address', 'chain'],
            DataType.TERM_STRUCTURE: ['timestamp', 'symbol', 'expiry', 'venue'],
        }.get(self, ['timestamp', 'symbol'])
    
    @property
    def default_interval(self) -> Optional[str]:
        """Expected data frequency."""
        return {
            DataType.FUNDING_RATES: '8h',
            DataType.OHLCV: '1h',
            DataType.OPEN_INTEREST: '1h',
            DataType.LIQUIDATIONS: None, # Event-based
            DataType.ORDERBOOK: None, # Snapshot
            DataType.TRADES: None, # Event-based
            DataType.OPTIONS: '1h',
            DataType.DEX_POOLS: '1h',
            DataType.TERM_STRUCTURE: '1h',
        }.get(self)

class CleaningAction(Enum):
    """Actions taken during cleaning, for audit trail."""
    # Schema enforcement
    COLUMN_RENAMED = auto()
    COLUMN_ADDED = auto()
    COLUMN_DROPPED = auto()
    TYPE_COERCED = auto()
    ENCODING_FIXED = auto()
    
    # Deduplication
    DUPLICATE_REMOVED = auto()
    DUPLICATE_AGGREGATED = auto()
    
    # Temporal
    TIMESTAMP_PARSED = auto()
    TIMESTAMP_ALIGNED = auto()
    TIMESTAMP_FILLED = auto()
    GAP_DETECTED = auto()
    
    # Outlier treatment
    OUTLIER_DETECTED = auto() # Outlier identified but no action taken
    OUTLIER_FLAGGED = auto()
    OUTLIER_CAPPED = auto()
    OUTLIER_REMOVED = auto()
    OUTLIER_INTERPOLATED = auto()
    OUTLIER_REPLACED = auto()
    
    # Missing data
    MISSING_FORWARD_FILLED = auto()
    MISSING_BACKWARD_FILLED = auto()
    MISSING_INTERPOLATED = auto()
    MISSING_CROSS_VENUE_FILLED = auto()
    MISSING_DROPPED = auto()
    
    # Value corrections
    VALUE_CORRECTED = auto()
    OHLCV_FIXED = auto()
    RATE_NORMALIZED = auto()
    SYMBOL_NORMALIZED = auto()
    GREEKS_CORRECTED = auto()
    
    # Row operations
    ROW_REMOVED = auto()
    ROW_FLAGGED = auto()

class DeduplicationStrategy(Enum):
    """Strategy for handling duplicate records."""
    KEEP_FIRST = 'keep_first'
    KEEP_LAST = 'keep_last'
    KEEP_BEST = 'keep_best' # Based on data completeness
    AGGREGATE_MEAN = 'aggregate_mean'
    AGGREGATE_MEDIAN = 'aggregate_median'
    REMOVE_ALL = 'remove_all'

class OutlierAction(Enum):
    """Action to take when outliers are detected."""
    FLAG_ONLY = 'flag'
    CAP_WINSORIZE = 'cap'
    REMOVE = 'remove'
    INTERPOLATE = 'interpolate'
    MEDIAN_REPLACE = 'median_replace'
    CROSS_VENUE_REPLACE = 'cross_venue'

class OutlierMethod(Enum):
    """Statistical method for outlier detection."""
    IQR = 'iqr' # Interquartile range
    ZSCORE = 'zscore' # Standard deviations from mean
    MODIFIED_ZSCORE = 'mad' # Median absolute deviation
    PERCENTILE = 'percentile' # Fixed percentile bounds
    DOMAIN_SPECIFIC = 'domain' # Data-type specific rules
    ENSEMBLE = 'ensemble' # Multiple methods voting (IQR + ZScore + MAD)
    LOF = 'lof' # Local Outlier Factor (sklearn)
    ISOLATION_FOREST = 'iforest' # Isolation Forest (sklearn)
    ROLLING_MAD = 'rolling_mad' # Rolling median absolute deviation
    ADVANCED_ENSEMBLE = 'advanced_ensemble' # LOF + IsolationForest + Rolling MAD

class MissingDataMethod(Enum):
    """Method for handling missing data."""
    FORWARD_FILL = 'ffill'
    BACKWARD_FILL = 'bfill'
    LINEAR_INTERPOLATE = 'linear'
    SPLINE_INTERPOLATE = 'spline'
    SEASONAL = 'seasonal'
    CROSS_VENUE = 'cross_venue'
    MEAN = 'mean'
    MEDIAN = 'median'
    ZERO = 'zero'
    DROP = 'drop'

class TimestampAlignment(Enum):
    """Method for aligning timestamps."""
    FLOOR = 'floor'
    CEIL = 'ceil'
    ROUND = 'round'
    SNAP_SETTLEMENT = 'snap_settlement'

# =============================================================================
# DATA CLASSES FOR REPORTING
# =============================================================================

@dataclass
class CleaningLogEntry:
    """
    Single cleaning operation log entry.
    
    Provides full audit trail of every data modification.
    """
    stage: str
    action: CleaningAction
    column: Optional[str]
    rows_affected: int
    rows_before: int
    rows_after: int
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def rows_changed(self) -> int:
        """Net row change from this operation."""
        return self.rows_after - self.rows_before
    
    @property
    def is_destructive(self) -> bool:
        """Whether this operation removed data."""
        return self.rows_after < self.rows_before
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'stage': self.stage,
            'action': self.action.name,
            'column': self.column,
            'rows_affected': self.rows_affected,
            'rows_before': self.rows_before,
            'rows_after': self.rows_after,
            'rows_changed': self.rows_changed,
            'is_destructive': self.is_destructive,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class DataFingerprint:
    """
    Cryptographic fingerprint of data state.
    
    Used to verify data integrity and track transformations.
    """
    row_count: int
    column_count: int
    checksum: str # MD5 of sorted column names + shape
    numeric_checksum: Optional[str] # Hash of numeric data
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'DataFingerprint':
        """Create fingerprint from DataFrame."""
        cols_str = ','.join(sorted(df.columns))
        shape_str = f"{len(df)}x{len(df.columns)}"
        checksum = hashlib.md5(f"{cols_str}|{shape_str}".encode()).hexdigest()[:12]
        
        numeric_checksum = None
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_data = df[numeric_cols].fillna(0).values.tobytes()
            numeric_checksum = hashlib.md5(numeric_data).hexdigest()[:12]
        
        return cls(
            row_count=len(df),
            column_count=len(df.columns),
            checksum=checksum,
            numeric_checksum=numeric_checksum
        )

@dataclass
class CleaningReport:
    """
    Comprehensive cleaning operation report.
    
    Provides complete audit trail, statistics, and reversibility information.
    """
    data_type: DataType
    venue: str
    
    # Row statistics
    original_rows: int
    final_rows: int
    
    # Column statistics 
    original_columns: int
    final_columns: int
    
    # Data fingerprints for integrity verification
    original_fingerprint: Optional[DataFingerprint] = None
    final_fingerprint: Optional[DataFingerprint] = None
    
    # Detailed log
    log_entries: List[CleaningLogEntry] = field(default_factory=list)
    
    # Issues
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Stage tracking
    stages_completed: List[str] = field(default_factory=list)
    stages_skipped: List[str] = field(default_factory=list)
    stages_failed: List[str] = field(default_factory=list)
    
    # Timing
    execution_time_seconds: float = 0.0
    stage_timings: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    config_used: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics (post-cleaning)
    quality_score_before: Optional[float] = None
    quality_score_after: Optional[float] = None
    
    @property
    def rows_removed(self) -> int:
        return self.original_rows - self.final_rows
    
    @property
    def rows_removed_pct(self) -> float:
        if self.original_rows == 0:
            return 0.0
        return (self.rows_removed / self.original_rows) * 100
    
    @property
    def rows_modified(self) -> int:
        """Total rows affected by any modification."""
        return sum(e.rows_affected for e in self.log_entries)
    
    @property
    def columns_added(self) -> int:
        return max(0, self.final_columns - self.original_columns)
    
    @property
    def total_operations(self) -> int:
        return len(self.log_entries)
    
    @property
    def destructive_operations(self) -> int:
        return sum(1 for e in self.log_entries if e.is_destructive)
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and len(self.stages_failed) == 0
    
    @property
    def quality_improvement(self) -> Optional[float]:
        """Quality score improvement from cleaning."""
        if self.quality_score_before is not None and self.quality_score_after is not None:
            return self.quality_score_after - self.quality_score_before
        return None
    
    def add_entry(self, entry: CleaningLogEntry):
        """Add a log entry."""
        self.log_entries.append(entry)
    
    def add_warning(self, msg: str):
        """Add a warning message."""
        self.warnings.append(msg)
        logger.warning(f"[{self.venue}] {msg}")
    
    def add_error(self, msg: str):
        """Add an error message."""
        self.errors.append(msg)
        logger.error(f"[{self.venue}] {msg}")
    
    def get_actions_by_type(self, action: CleaningAction) -> List[CleaningLogEntry]:
        """Get all log entries for a specific action type."""
        return [e for e in self.log_entries if e.action == action]
    
    def get_stage_entries(self, stage: str) -> List[CleaningLogEntry]:
        """Get all log entries for a specific stage."""
        return [e for e in self.log_entries if e.stage == stage]
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "DATA CLEANING REPORT",
            "=" * 70,
            f"Data Type: {self.data_type.value}",
            f"Venue: {self.venue}",
            f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Duration: {self.execution_time_seconds:.2f}s",
            "",
            "â”€" * 70,
            "ROW STATISTICS",
            "â”€" * 70,
            f" Original rows: {self.original_rows:>12,}",
            f" Final rows: {self.final_rows:>12,}",
            f" Rows removed: {self.rows_removed:>12,} ({self.rows_removed_pct:.2f}%)",
            f" Rows modified: {self.rows_modified:>12,}",
            "",
            "â”€" * 70,
            "COLUMN STATISTICS", 
            "â”€" * 70,
            f" Original columns: {self.original_columns:>12}",
            f" Final columns: {self.final_columns:>12}",
            f" Columns added: {self.columns_added:>12}",
            "",
            "â”€" * 70,
            "OPERATIONS",
            "â”€" * 70,
            f" Total operations: {self.total_operations:>12}",
            f" Destructive ops: {self.destructive_operations:>12}",
            "",
            "â”€" * 70,
            "STAGES",
            "â”€" * 70,
        ]
        
        for stage in self.stages_completed:
            timing = self.stage_timings.get(stage, 0)
            lines.append(f" âœ“ {stage:<30} ({timing:.3f}s)")
        
        for stage in self.stages_skipped:
            lines.append(f" â—‹ {stage:<30} (skipped)")
        
        for stage in self.stages_failed:
            lines.append(f" âœ— {stage:<30} (FAILED)")
        
        if self.quality_score_before is not None and self.quality_score_after is not None:
            lines.extend([
                "",
                "â”€" * 70,
                "QUALITY IMPROVEMENT",
                "â”€" * 70,
                f" Before cleaning: {self.quality_score_before:>12.1f}",
                f" After cleaning: {self.quality_score_after:>12.1f}",
                f" Improvement: {self.quality_improvement:>+12.1f}",
            ])
        
        if self.warnings:
            lines.extend([
                "",
                "â”€" * 70,
                f"WARNINGS ({len(self.warnings)})",
                "â”€" * 70,
            ])
            for w in self.warnings[:10]:
                lines.append(f" âš  {w}")
            if len(self.warnings) > 10:
                lines.append(f" ... and {len(self.warnings) - 10} more")
        
        if self.errors:
            lines.extend([
                "",
                "â”€" * 70,
                f"ERRORS ({len(self.errors)})",
                "â”€" * 70,
            ])
            for e in self.errors:
                lines.append(f" âœ— {e}")
        
        lines.extend([
            "",
            "=" * 70,
            f"STATUS: {'SUCCESS' if self.success else 'FAILED'}",
            "=" * 70,
        ])
        
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'data_type': self.data_type.value,
            'venue': self.venue,
            'original_rows': self.original_rows,
            'final_rows': self.final_rows,
            'rows_removed': self.rows_removed,
            'rows_removed_pct': round(self.rows_removed_pct, 4),
            'rows_modified': self.rows_modified,
            'original_columns': self.original_columns,
            'final_columns': self.final_columns,
            'columns_added': self.columns_added,
            'total_operations': self.total_operations,
            'destructive_operations': self.destructive_operations,
            'stages_completed': self.stages_completed,
            'stages_skipped': self.stages_skipped,
            'stages_failed': self.stages_failed,
            'stage_timings': self.stage_timings,
            'warnings': self.warnings,
            'errors': self.errors,
            'success': self.success,
            'execution_time_seconds': round(self.execution_time_seconds, 4),
            'quality_score_before': self.quality_score_before,
            'quality_score_after': self.quality_score_after,
            'quality_improvement': self.quality_improvement,
            'timestamp': self.timestamp.isoformat(),
            'config_used': self.config_used,
            'log_entries': [e.to_dict() for e in self.log_entries],
        }

# =============================================================================
# CLEANING STAGE BASE CLASS
# =============================================================================

class CleaningStage(ABC):
    """
    Abstract base class for all cleaning stages.
    
    Each stage:
    1. Receives a DataFrame
    2. Performs specific cleaning operations
    3. Logs all changes to the report
    4. Returns the modified DataFrame
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize cleaning stage.
        
        Args:
            config: Stage-specific configuration options
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @property
    @abstractmethod
    def stage_name(self) -> str:
        """Unique identifier for this stage."""
        pass
    
    @property
    def stage_description(self) -> str:
        """Human-readable description."""
        return self.__class__.__doc__ or self.stage_name
    
    @abstractmethod
    def execute(
        self,
        df: pd.DataFrame,
        report: CleaningReport,
        data_type: DataType,
        **kwargs
    ) -> pd.DataFrame:
        """
        Execute the cleaning stage.
        
        Args:
            df: Input DataFrame
            report: CleaningReport to log changes
            data_type: Type of data being cleaned
            **kwargs: Additional stage-specific options
            
        Returns:
            Cleaned DataFrame
        """
        pass
    
    def _log_action(
        self,
        report: CleaningReport,
        action: CleaningAction,
        column: Optional[str] = None,
        rows_affected: int = 0,
        rows_before: int = 0,
        rows_after: int = 0,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log a cleaning action to the report.
        
        Args:
            report: CleaningReport to add entry to
            action: Type of action performed
            column: Column affected (if applicable)
            rows_affected: Number of rows affected
            rows_before: Row count before operation
            rows_after: Row count after operation
            details: Additional details about the operation
        """
        entry = CleaningLogEntry(
            stage=self.stage_name,
            action=action,
            column=column,
            rows_affected=rows_affected,
            rows_before=rows_before,
            rows_after=rows_after,
            details=details or {}
        )
        report.add_entry(entry)
        
        if rows_affected > 0:
            self.logger.debug(
                f"{action.name}: {rows_affected} rows, "
                f"column={column}, {rows_before}â†’{rows_after} rows"
            )

# =============================================================================
# STAGE 1: SCHEMA ENFORCEMENT
# =============================================================================

class SchemaEnforcementStage(CleaningStage):
    """
    Enforce schema consistency across venues.
    
    Operations:
    - Standardize column names (venue-specific â†’ canonical)
    - Coerce data types (string â†’ numeric, int â†’ datetime)
    - Add missing required columns with appropriate defaults
    - Handle encoding issues in string columns
    - Validate and normalize timestamp format
    
    This stage ensures all data has a consistent structure
    regardless of source venue.
    """
    
    # Comprehensive column name mapping
    # Maps various venue-specific names to canonical names
    COLUMN_ALIASES: Dict[str, str] = {
        # Timestamp variants
        'time': 'timestamp',
        'ts': 'timestamp', 
        't': 'timestamp',
        'datetime': 'timestamp',
        'date': 'timestamp',
        'Time': 'timestamp',
        'Timestamp': 'timestamp',
        'fundingTime': 'timestamp',
        'funding_time': 'timestamp',
        'calcTime': 'timestamp',
        'created_at': 'timestamp',
        'block_timestamp': 'timestamp',
        
        # Symbol variants
        'token': 'symbol',
        'asset': 'symbol',
        'coin': 'symbol',
        'pair': 'symbol',
        'market': 'symbol',
        'instrument': 'symbol',
        'Symbol': 'symbol',
        'name': 'symbol',
        'ticker': 'symbol',
        
        # Funding rate variants
        'funding': 'funding_rate',
        'rate': 'funding_rate',
        'fundingRate': 'funding_rate',
        'funding_rate_8h': 'funding_rate',
        'lastFundingRate': 'funding_rate',
        'fundingRateCurrent': 'funding_rate',
        
        # Price variants
        'price': 'close',
        'last': 'close',
        'lastPrice': 'close',
        'last_price': 'close',
        'markPrice': 'mark_price',
        'mark': 'mark_price',
        'indexPrice': 'index_price',
        'index': 'index_price',
        'spotPrice': 'index_price',
        
        # Volume variants
        'vol': 'volume',
        'qty': 'volume',
        'amount': 'volume',
        'baseVolume': 'volume',
        'quoteVolume': 'volume_usd',
        'volumeUsd': 'volume_usd',
        'volume_usd_24h': 'volume_24h_usd',
        
        # Venue variants
        'exchange': 'venue',
        'source': 'venue',
        'platform': 'venue',
        'Exchange': 'venue',
        
        # Open interest variants
        'oi': 'open_interest',
        'openInterest': 'open_interest',
        'open_Interest': 'open_interest',
        'sumOpenInterest': 'open_interest',
        'oiValue': 'open_interest_usd',
        'openInterestValue': 'open_interest_usd',
        
        # Options-specific
        'iv': 'mark_iv',
        'impliedVolatility': 'mark_iv',
        'implied_volatility': 'mark_iv',
        'strikePrice': 'strike',
        'strike_price': 'strike',
        'expirationDate': 'expiry',
        'expiration': 'expiry',
        'expiry_date': 'expiry',
        'optionType': 'option_type',
        'type': 'option_type',
        
        # DEX-specific
        'tvl': 'tvl_usd',
        'totalValueLocked': 'tvl_usd',
        'liquidity': 'tvl_usd',
        'liquidityUSD': 'tvl_usd',
        'feeTier': 'fee_tier',
        'fee': 'fee_tier',
        'poolAddress': 'pool_address',
        'pool': 'pool_address',
    }
    
    @property
    def stage_name(self) -> str:
        return "schema_enforcement"
    
    def execute(
        self,
        df: pd.DataFrame,
        report: CleaningReport,
        data_type: DataType,
        venue: str = 'unknown',
        **kwargs
    ) -> pd.DataFrame:
        """Execute schema enforcement."""
        import time
        start_time = time.time()
        
        df = df.copy()
        rows_before = len(df)
        cols_before = len(df.columns)
        
        # Step 1: Standardize column names
        df = self._standardize_columns(df, report, rows_before)
        
        # Step 2: Add missing required columns
        df = self._add_required_columns(df, data_type, report, rows_before)
        
        # Step 3: Convert timestamp
        df = self._convert_timestamp(df, report, rows_before)
        
        # Step 4: Coerce numeric types
        df = self._coerce_numeric_types(df, data_type, report, rows_before)
        
        # Step 5: Add venue column if missing
        if 'venue' not in df.columns:
            df['venue'] = venue
            self._log_action(
                report, CleaningAction.COLUMN_ADDED,
                column='venue',
                rows_affected=len(df),
                rows_before=rows_before,
                rows_after=len(df),
                details={'default_value': venue}
            )
        
        # Step 6: Fix string encoding issues
        df = self._fix_encoding(df, report, rows_before)
        
        report.stages_completed.append(self.stage_name)
        report.stage_timings[self.stage_name] = time.time() - start_time
        
        return df
    
    def _standardize_columns(
        self,
        df: pd.DataFrame,
        report: CleaningReport,
        rows_before: int
    ) -> pd.DataFrame:
        """Standardize column names to canonical format."""
        rename_map = {}
        
        for col in df.columns:
            # Check exact match first
            if col in self.COLUMN_ALIASES:
                standard = self.COLUMN_ALIASES[col]
                if standard not in df.columns and standard not in rename_map.values():
                    rename_map[col] = standard
            # Check case-insensitive match
            elif col.lower() in {k.lower(): k for k in self.COLUMN_ALIASES}:
                for alias, standard in self.COLUMN_ALIASES.items():
                    if col.lower() == alias.lower():
                        if standard not in df.columns and standard not in rename_map.values():
                            rename_map[col] = standard
                        break
        
        if rename_map:
            df = df.rename(columns=rename_map)
            self._log_action(
                report, CleaningAction.COLUMN_RENAMED,
                rows_affected=len(df),
                rows_before=rows_before,
                rows_after=len(df),
                details={'mapping': rename_map, 'count': len(rename_map)}
            )
        
        return df
    
    def _add_required_columns(
        self,
        df: pd.DataFrame,
        data_type: DataType,
        report: CleaningReport,
        rows_before: int
    ) -> pd.DataFrame:
        """Add missing required columns with appropriate defaults."""
        required = data_type.required_columns
        added = []
        
        for col in required:
            if col not in df.columns:
                # Determine appropriate default
                if col in ['timestamp']:
                    df[col] = pd.NaT
                elif col in ['venue', 'symbol', 'side', 'option_type']:
                    df[col] = 'unknown'
                elif col in data_type.numeric_columns:
                    df[col] = np.nan
                else:
                    df[col] = None
                
                added.append(col)
        
        if added:
            self._log_action(
                report, CleaningAction.COLUMN_ADDED,
                rows_affected=len(df),
                rows_before=rows_before,
                rows_after=len(df),
                details={'columns': added, 'count': len(added)}
            )
        
        return df
    
    def _convert_timestamp(
        self,
        df: pd.DataFrame,
        report: CleaningReport,
        rows_before: int
    ) -> pd.DataFrame:
        """Convert timestamp to datetime64[ns, UTC]."""
        if 'timestamp' not in df.columns:
            return df
        
        ts = df['timestamp']
        original_dtype = str(ts.dtype)
        converted = False
        
        try:
            # Already datetime with timezone
            if pd.api.types.is_datetime64_any_dtype(ts):
                if ts.dt.tz is None:
                    df['timestamp'] = ts.dt.tz_localize('UTC')
                    converted = True
                elif str(ts.dt.tz) != 'UTC':
                    df['timestamp'] = ts.dt.tz_convert('UTC')
                    converted = True
            
            # Integer (Unix timestamp)
            elif pd.api.types.is_integer_dtype(ts) or pd.api.types.is_float_dtype(ts):
                # Determine unit based on magnitude
                sample = ts.dropna().iloc[0] if len(ts.dropna()) > 0 else 0
                if sample > 1e15: # Nanoseconds
                    df['timestamp'] = pd.to_datetime(ts, unit='ns', utc=True)
                elif sample > 1e12: # Milliseconds
                    df['timestamp'] = pd.to_datetime(ts, unit='ms', utc=True)
                else: # Seconds
                    df['timestamp'] = pd.to_datetime(ts, unit='s', utc=True)
                converted = True
            
            # String
            else:
                df['timestamp'] = pd.to_datetime(ts, utc=True, errors='coerce')
                converted = True
            
            if converted:
                new_dtype = str(df['timestamp'].dtype)
                self._log_action(
                    report, CleaningAction.TIMESTAMP_PARSED,
                    column='timestamp',
                    rows_affected=len(df),
                    rows_before=rows_before,
                    rows_after=len(df),
                    details={
                        'from_dtype': original_dtype,
                        'to_dtype': new_dtype
                    }
                )
        
        except Exception as e:
            report.add_warning(f"Timestamp conversion failed: {e}")
        
        return df
    
    def _coerce_numeric_types(
        self,
        df: pd.DataFrame,
        data_type: DataType,
        report: CleaningReport,
        rows_before: int
    ) -> pd.DataFrame:
        """Coerce columns to numeric types."""
        numeric_cols = data_type.numeric_columns
        coerced = []
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(df[col]):
                original_dtype = str(df[col].dtype)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                coerced.append({'column': col, 'from': original_dtype, 'to': str(df[col].dtype)})
        
        if coerced:
            self._log_action(
                report, CleaningAction.TYPE_COERCED,
                rows_affected=len(df),
                rows_before=rows_before,
                rows_after=len(df),
                details={'columns': coerced, 'count': len(coerced)}
            )
        
        # Ensure float64 precision for all numeric
        for col in numeric_cols:
            if col in df.columns:
                try:
                    if df[col].dtype not in [np.float64, np.float32]:
                        df[col] = df[col].astype(np.float64)
                except (ValueError, TypeError):
                    pass
        
        return df
    
    def _fix_encoding(
        self,
        df: pd.DataFrame,
        report: CleaningReport,
        rows_before: int
    ) -> pd.DataFrame:
        """Fix encoding issues in string columns."""
        # Pandas 4+: Must explicitly include 'string' dtype alongside 'object'
        string_cols = df.select_dtypes(include=['object', 'string']).columns
        fixed_count = 0
        
        for col in string_cols:
            try:
                # Try to encode/decode to fix issues
                original = df[col].copy()
                df[col] = df[col].astype(str).str.encode('utf-8', errors='replace').str.decode('utf-8')
                if not df[col].equals(original):
                    fixed_count += 1
            except Exception:
                pass
        
        if fixed_count > 0:
            self._log_action(
                report, CleaningAction.ENCODING_FIXED,
                rows_affected=len(df),
                rows_before=rows_before,
                rows_after=len(df),
                details={'columns_fixed': fixed_count}
            )
        
        return df
# =============================================================================
# STAGE 2: DEDUPLICATION
# =============================================================================

class DeduplicationStage(CleaningStage):
    """
    Remove duplicate records with configurable strategies.
    
    Duplicates can arise from:
    - Multiple API calls returning overlapping data
    - Data source corrections/updates
    - Collection errors
    
    Strategies determine which record to keep when duplicates exist.
    """
    
    @property
    def stage_name(self) -> str:
        return "deduplication"
    
    def execute(
        self,
        df: pd.DataFrame,
        report: CleaningReport,
        data_type: DataType,
        strategy: DeduplicationStrategy = DeduplicationStrategy.KEEP_LAST,
        key_columns: List[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Execute deduplication."""
        import time
        start_time = time.time()
        
        df = df.copy()
        rows_before = len(df)
        
        # Determine key columns
        if key_columns:
            keys = [c for c in key_columns if c in df.columns]
        else:
            keys = [c for c in data_type.primary_key if c in df.columns]
        
        if not keys:
            report.add_warning("No key columns found for deduplication, using all columns")
            keys = list(df.columns)
        
        # Find duplicates
        dup_mask = df.duplicated(subset=keys, keep=False)
        dup_count = dup_mask.sum()
        
        if dup_count == 0:
            report.stages_completed.append(self.stage_name)
            report.stage_timings[self.stage_name] = time.time() - start_time
            return df
        
        # Apply strategy
        if strategy == DeduplicationStrategy.KEEP_FIRST:
            df = df.drop_duplicates(subset=keys, keep='first')
        
        elif strategy == DeduplicationStrategy.KEEP_LAST:
            df = df.drop_duplicates(subset=keys, keep='last')
        
        elif strategy == DeduplicationStrategy.REMOVE_ALL:
            df = df[~dup_mask]
        
        elif strategy == DeduplicationStrategy.KEEP_BEST:
            df = self._keep_best(df, keys, data_type)
        
        elif strategy in [DeduplicationStrategy.AGGREGATE_MEAN, DeduplicationStrategy.AGGREGATE_MEDIAN]:
            df = self._aggregate_duplicates(df, keys, data_type, strategy)
        
        rows_after = len(df)
        removed = rows_before - rows_after
        
        self._log_action(
            report, CleaningAction.DUPLICATE_REMOVED,
            rows_affected=removed,
            rows_before=rows_before,
            rows_after=rows_after,
            details={
                'strategy': strategy.value,
                'key_columns': keys,
                'duplicates_found': int(dup_count),
                'duplicates_removed': removed,
                'duplicate_pct': round(dup_count / rows_before * 100, 2) if rows_before > 0 else 0
            }
        )
        
        report.stages_completed.append(self.stage_name)
        report.stage_timings[self.stage_name] = time.time() - start_time
        
        return df
    
    def _keep_best(
        self,
        df: pd.DataFrame,
        keys: List[str],
        data_type: DataType
    ) -> pd.DataFrame:
        """Keep the most complete record among duplicates."""
        numeric_cols = [c for c in data_type.numeric_columns if c in df.columns]
        
        if not numeric_cols:
            return df.drop_duplicates(subset=keys, keep='first')
        
        # Calculate completeness score
        df['_completeness'] = df[numeric_cols].notna().sum(axis=1)
        
        # Sort by completeness descending, then by timestamp (latest first)
        sort_cols = ['_completeness']
        sort_ascending = [False]
        
        if 'timestamp' in df.columns:
            sort_cols.append('timestamp')
            sort_ascending.append(False)
        
        df = df.sort_values(sort_cols, ascending=sort_ascending)
        df = df.drop_duplicates(subset=keys, keep='first')
        df = df.drop(columns=['_completeness'])
        
        return df
    
    def _aggregate_duplicates(
        self,
        df: pd.DataFrame,
        keys: List[str],
        data_type: DataType,
        strategy: DeduplicationStrategy
    ) -> pd.DataFrame:
        """Aggregate duplicate records."""
        numeric_cols = [c for c in data_type.numeric_columns if c in df.columns]
        
        agg_func = 'mean' if strategy == DeduplicationStrategy.AGGREGATE_MEAN else 'median'
        
        # Build aggregation dict
        agg_dict = {}
        for col in df.columns:
            if col in keys:
                continue
            elif col in numeric_cols:
                agg_dict[col] = agg_func
            else:
                agg_dict[col] = 'first'
        
        return df.groupby(keys, as_index=False).agg(agg_dict)

# =============================================================================
# STAGE 3: TEMPORAL ALIGNMENT
# =============================================================================

class TemporalAlignmentStage(CleaningStage):
    """
    Align timestamps to regular intervals.
    
    Critical for:
    - Cross-venue comparison (different settlement times)
    - Time series analysis (regular frequency required)
    - Backtest accuracy (align to trading signals)
    
    Handles venue-specific settlement times for funding rates.
    """
    
    # Funding rate settlement times by venue (UTC hours)
    SETTLEMENT_TIMES: Dict[str, List[int]] = {
        'binance': [0, 8, 16],
        'bybit': [0, 8, 16],
        'okx': [0, 8, 16],
        'deribit': [8],
        'kraken': [0, 4, 8, 12, 16, 20],
        'hyperliquid': list(range(24)), # Hourly
        'dydx': list(range(24)),
        'dydx_v4': list(range(24)),
        'gmx': list(range(24)),
        'vertex': list(range(24)),
        'synthetix': list(range(24)),
    }
    
    # Interval string to pandas frequency mapping (use lowercase h for hours)
    INTERVAL_MAP: Dict[str, str] = {
        '1m': 'min', '5m': '5min', '15m': '15min', '30m': '30min',
        '1h': 'h', '2h': '2h', '4h': '4h', '8h': '8h', '12h': '12h',
        '1d': 'D', '1w': 'W', '1M': 'MS',
    }
    
    @property
    def stage_name(self) -> str:
        return "temporal_alignment"
    
    def execute(
        self,
        df: pd.DataFrame,
        report: CleaningReport,
        data_type: DataType,
        method: TimestampAlignment = TimestampAlignment.FLOOR,
        interval: str = None,
        venue: str = None,
        **kwargs
    ) -> pd.DataFrame:
        """Execute temporal alignment."""
        import time
        start_time = time.time()
        
        if 'timestamp' not in df.columns:
            report.add_warning("No timestamp column found for alignment")
            report.stages_skipped.append(self.stage_name)
            return df
        
        df = df.copy()
        rows_before = len(df)
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        
        # Determine interval
        interval = interval or data_type.default_interval
        if not interval:
            report.stages_skipped.append(self.stage_name)
            return df
        
        freq = self.INTERVAL_MAP.get(interval.lower(), interval)
        original_ts = df['timestamp'].copy()
        
        # Apply alignment method
        if method == TimestampAlignment.FLOOR:
            df['timestamp'] = df['timestamp'].dt.floor(freq)
        
        elif method == TimestampAlignment.CEIL:
            df['timestamp'] = df['timestamp'].dt.ceil(freq)
        
        elif method == TimestampAlignment.ROUND:
            df['timestamp'] = df['timestamp'].dt.round(freq)
        
        elif method == TimestampAlignment.SNAP_SETTLEMENT:
            venue = venue or df.get('venue', pd.Series(['binance'])).iloc[0]
            if isinstance(venue, pd.Series):
                venue = venue.iloc[0] if len(venue) > 0 else 'binance'
            df['timestamp'] = df['timestamp'].apply(
                lambda ts: self._snap_to_settlement(ts, str(venue).lower())
            )
        
        # Count changes
        changed = (df['timestamp'] != original_ts).sum()
        
        if changed > 0:
            self._log_action(
                report, CleaningAction.TIMESTAMP_ALIGNED,
                column='timestamp',
                rows_affected=int(changed),
                rows_before=rows_before,
                rows_after=len(df),
                details={
                    'method': method.value,
                    'interval': interval,
                    'frequency': freq,
                    'timestamps_modified': int(changed),
                    'modification_pct': round(changed / rows_before * 100, 2) if rows_before > 0 else 0
                }
            )
        
        report.stages_completed.append(self.stage_name)
        report.stage_timings[self.stage_name] = time.time() - start_time
        
        return df
    
    def _snap_to_settlement(self, ts: pd.Timestamp, venue: str) -> pd.Timestamp:
        """Snap timestamp to nearest settlement time for venue."""
        if pd.isna(ts):
            return ts
        
        settlement_hours = self.SETTLEMENT_TIMES.get(venue, [0, 8, 16])
        current_hour = ts.hour
        
        # Find nearest settlement hour
        distances = [(h, abs(h - current_hour)) for h in settlement_hours]
        # Also consider wraparound (23:00 is close to 00:00)
        for h in settlement_hours:
            distances.append((h, 24 - abs(h - current_hour)))
        
        nearest_hour = min(distances, key=lambda x: x[1])[0]
        
        return ts.replace(hour=nearest_hour, minute=0, second=0, microsecond=0)

# =============================================================================
# STAGE 4: OUTLIER TREATMENT
# =============================================================================

class OutlierTreatmentStage(CleaningStage):
    """
    Detect and treat statistical outliers.
    
    Methods:
    - IQR: reliable to existing outliers
    - Z-score: Assumes normality
    - Modified Z-score (MAD): reliable to outliers, non-normal data
    - Domain-specific: Uses crypto market knowledge
    
    Actions:
    - Flag: Add indicator column, keep data for analysis
    - Cap: Winsorize to bounds (preserves row count)
    - Remove: Delete outlier rows (reduces row count)
    - Interpolate: Replace with estimated value
    """
    
    # Domain-specific bounds for crypto data
    DOMAIN_BOUNDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
        # Funding rates
        'funding_rate': (-0.10, 0.10), # -10% to +10% per period (extreme)
        'funding_rate_8h': (-0.10, 0.10),
        'funding_rate_annualized': (-100.0, 100.0), # -10000% to +10000% annual
        
        # Options Greeks
        'mark_iv': (0.01, 10.0), # 1% to 1000% IV
        'delta': (-1.0, 1.0),
        'gamma': (0.0, 100.0),
        'vega': (-1000.0, 1000.0),
        'theta': (-1000.0, 1000.0),
        
        # Prices (must be positive)
        'open': (0.0, None),
        'high': (0.0, None),
        'low': (0.0, None),
        'close': (0.0, None),
        'mark_price': (0.0, None),
        'index_price': (0.0, None),
        'strike': (0.0, None),
        
        # Volume (non-negative)
        'volume': (0.0, None),
        'volume_usd': (0.0, None),
        'volume_24h_usd': (0.0, None),
        
        # Open interest (non-negative)
        'open_interest': (0.0, None),
        'open_interest_usd': (0.0, None),
        
        # DEX metrics
        'tvl_usd': (0.0, None),
        'fee_tier': (0.0, 0.10), # 0% to 10%
    }
    
    @property
    def stage_name(self) -> str:
        return "outlier_treatment"
    
    def execute(
        self,
        df: pd.DataFrame,
        report: CleaningReport,
        data_type: DataType,
        action: OutlierAction = OutlierAction.CAP_WINSORIZE,
        method: OutlierMethod = OutlierMethod.IQR,
        threshold: float = 3.0,
        columns: List[str] = None,
        percentile_bounds: Tuple[float, float] = (0.01, 0.99),
        **kwargs
    ) -> pd.DataFrame:
        """Execute outlier treatment."""
        import time
        start_time = time.time()
        
        df = df.copy()
        rows_before = len(df)
        
        # Determine columns to check
        if columns is None:
            columns = [c for c in data_type.numeric_columns if c in df.columns]
        else:
            columns = [c for c in columns if c in df.columns]
        
        if not columns:
            report.add_warning("No numeric columns found for outlier treatment")
            report.stages_skipped.append(self.stage_name)
            return df
        
        total_outliers = 0
        
        for col in columns:
            # Skip if column has no valid data
            valid_count = df[col].notna().sum()
            if valid_count < 10:
                continue
            
            # Detect outliers
            outlier_mask = self._detect_outliers(
                df[col], method, threshold, col, percentile_bounds
            )
            outlier_count = outlier_mask.sum()
            
            if outlier_count == 0:
                continue
            
            total_outliers += outlier_count
            
            # Apply action
            if action == OutlierAction.FLAG_ONLY:
                flag_col = f'{col}_is_outlier'
                df[flag_col] = outlier_mask
                action_taken = CleaningAction.OUTLIER_FLAGGED
            
            elif action == OutlierAction.CAP_WINSORIZE:
                df = self._cap_outliers(df, col, method, threshold, percentile_bounds)
                action_taken = CleaningAction.OUTLIER_CAPPED
            
            elif action == OutlierAction.REMOVE:
                df = df[~outlier_mask]
                action_taken = CleaningAction.OUTLIER_REMOVED
            
            elif action == OutlierAction.INTERPOLATE:
                df.loc[outlier_mask, col] = np.nan
                df[col] = df[col].interpolate(method='linear', limit=5)
                action_taken = CleaningAction.OUTLIER_INTERPOLATED
            
            elif action == OutlierAction.MEDIAN_REPLACE:
                rolling_median = df[col].rolling(
                    window=24, min_periods=1, center=True
                ).median()
                df.loc[outlier_mask, col] = rolling_median[outlier_mask]
                action_taken = CleaningAction.OUTLIER_REPLACED
            
            self._log_action(
                report, action_taken,
                column=col,
                rows_affected=int(outlier_count),
                rows_before=rows_before,
                rows_after=len(df),
                details={
                    'method': method.value,
                    'threshold': threshold,
                    'action': action.value,
                    'outliers_found': int(outlier_count),
                    'outlier_pct': round(outlier_count / valid_count * 100, 2) if valid_count > 0 else 0
                }
            )
        
        report.stages_completed.append(self.stage_name)
        report.stage_timings[self.stage_name] = time.time() - start_time
        
        return df
    
    def _detect_outliers(
        self,
        series: pd.Series,
        method: OutlierMethod,
        threshold: float,
        col_name: str,
        percentile_bounds: Tuple[float, float]
    ) -> pd.Series:
        """Detect outliers using specified method."""
        data = series.dropna()
        
        if len(data) < 10:
            return pd.Series(False, index=series.index)
        
        # Domain-specific bounds check first
        if col_name in self.DOMAIN_BOUNDS:
            lower, upper = self.DOMAIN_BOUNDS[col_name]
            mask = pd.Series(False, index=series.index)
            if lower is not None:
                mask |= (series < lower)
            if upper is not None:
                mask |= (series > upper)
            if mask.any():
                return mask
        
        # Statistical methods
        if method == OutlierMethod.IQR:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            return (series < lower) | (series > upper)
        
        elif method == OutlierMethod.ZSCORE:
            mean = data.mean()
            std = data.std()
            if std == 0 or pd.isna(std):
                return pd.Series(False, index=series.index)
            z_scores = np.abs((series - mean) / std)
            return z_scores > threshold
        
        elif method == OutlierMethod.MODIFIED_ZSCORE:
            median = data.median()
            mad = np.median(np.abs(data - median))
            if mad == 0:
                mad = data.std() * 0.6745 # Fallback
            if mad == 0 or pd.isna(mad):
                return pd.Series(False, index=series.index)
            modified_z = 0.6745 * np.abs(series - median) / mad
            return modified_z > threshold
        
        elif method == OutlierMethod.PERCENTILE:
            lower = data.quantile(percentile_bounds[0])
            upper = data.quantile(percentile_bounds[1])
            return (series < lower) | (series > upper)
        
        elif method == OutlierMethod.ENSEMBLE:
            # Majority voting across multiple methods
            methods = [OutlierMethod.IQR, OutlierMethod.ZSCORE, OutlierMethod.MODIFIED_ZSCORE]
            votes = pd.DataFrame(index=series.index)
            for m in methods:
                votes[m.value] = self._detect_outliers(series, m, threshold, col_name, percentile_bounds)
            return votes.sum(axis=1) >= 2 # At least 2 methods agree

        elif method == OutlierMethod.LOF:
            # Local Outlier Factor - detects local anomalies
            return self._detect_lof_outliers(series, threshold)

        elif method == OutlierMethod.ISOLATION_FOREST:
            # Isolation Forest - fast global anomaly detection
            return self._detect_isolation_forest_outliers(series, threshold)

        elif method == OutlierMethod.ROLLING_MAD:
            # Rolling median absolute deviation - reliable for time series
            return self._detect_rolling_mad_outliers(series, threshold)

        elif method == OutlierMethod.ADVANCED_ENSEMBLE:
            # detailed ensemble: LOF + Isolation Forest + Rolling MAD + IQR
            votes = pd.DataFrame(index=series.index)
            votes['lof'] = self._detect_lof_outliers(series, threshold)
            votes['iforest'] = self._detect_isolation_forest_outliers(series, threshold)
            votes['rolling_mad'] = self._detect_rolling_mad_outliers(series, threshold)
            votes['iqr'] = self._detect_outliers(series, OutlierMethod.IQR, threshold, col_name, percentile_bounds)
            # Require at least 2 methods to agree (majority voting)
            return votes.sum(axis=1) >= 2

        return pd.Series(False, index=series.index)

    def _detect_lof_outliers(
        self,
        series: pd.Series,
        threshold: float = 3.0,
        n_neighbors: int = 20,
        contamination: float = 0.05
    ) -> pd.Series:
        """
        Detect outliers using Local Outlier Factor.

        LOF measures the local density deviation of a point relative to its neighbors.
        Best for detecting local anomalies in non-uniformly distributed data.
        """
        try:
            from sklearn.neighbors import LocalOutlierFactor
        except ImportError:
            logger.warning("sklearn not available for LOF, falling back to IQR")
            return self._detect_outliers(
                series, OutlierMethod.IQR, threshold, series.name, (0.01, 0.99)
            )

        data = series.dropna()
        if len(data) < n_neighbors + 1:
            return pd.Series(False, index=series.index)

        # LOF requires 2D input
        X = data.values.reshape(-1, 1)

        try:
            lof = LocalOutlierFactor(
                n_neighbors=min(n_neighbors, len(data) - 1),
                contamination=min(contamination, 0.5),
                novelty=False
            )
            # -1 for outliers, 1 for inliers
            predictions = lof.fit_predict(X)

            # Map back to original index
            outlier_mask = pd.Series(False, index=series.index)
            outlier_mask.loc[data.index] = (predictions == -1)
            return outlier_mask

        except Exception as e:
            logger.warning(f"LOF detection failed: {e}, falling back to IQR")
            return self._detect_outliers(
                series, OutlierMethod.IQR, threshold, series.name, (0.01, 0.99)
            )

    def _detect_isolation_forest_outliers(
        self,
        series: pd.Series,
        threshold: float = 3.0,
        contamination: float = 0.05,
        n_estimators: int = 100
    ) -> pd.Series:
        """
        Detect outliers using Isolation Forest.

        Isolation Forest isolates anomalies by randomly selecting a feature
        and a split value. Outliers require fewer splits to isolate.
        Best for fast, global anomaly detection.
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.warning("sklearn not available for IsolationForest, falling back to IQR")
            return self._detect_outliers(
                series, OutlierMethod.IQR, threshold, series.name, (0.01, 0.99)
            )

        data = series.dropna()
        if len(data) < 10:
            return pd.Series(False, index=series.index)

        # IsolationForest requires 2D input
        X = data.values.reshape(-1, 1)

        try:
            iforest = IsolationForest(
                contamination=min(contamination, 0.5),
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            )
            # -1 for outliers, 1 for inliers
            predictions = iforest.fit_predict(X)

            # Map back to original index
            outlier_mask = pd.Series(False, index=series.index)
            outlier_mask.loc[data.index] = (predictions == -1)
            return outlier_mask

        except Exception as e:
            logger.warning(f"IsolationForest detection failed: {e}, falling back to IQR")
            return self._detect_outliers(
                series, OutlierMethod.IQR, threshold, series.name, (0.01, 0.99)
            )

    def _detect_rolling_mad_outliers(
        self,
        series: pd.Series,
        threshold: float = 3.0,
        window: int = 24
    ) -> pd.Series:
        """
        Detect outliers using rolling median absolute deviation.

        Rolling MAD is reliable to outliers and handles non-stationarity
        in time series data better than global statistics.
        """
        data = series.copy()

        if len(data.dropna()) < window:
            # Fallback to global MAD if not enough data
            return self._detect_outliers(
                series, OutlierMethod.MODIFIED_ZSCORE, threshold, series.name, (0.01, 0.99)
            )

        # Calculate rolling median and MAD
        rolling_median = data.rolling(window=window, min_periods=1, center=True).median()
        rolling_mad = data.rolling(window=window, min_periods=1, center=True).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        )

        # Handle zero MAD (constant regions)
        rolling_mad = rolling_mad.replace(0, data.std() * 0.6745)

        # Calculate modified z-score
        modified_z = 0.6745 * np.abs(data - rolling_median) / rolling_mad

        return modified_z > threshold
    
    def _cap_outliers(
        self,
        df: pd.DataFrame,
        col: str,
        method: OutlierMethod,
        threshold: float,
        percentile_bounds: Tuple[float, float]
    ) -> pd.DataFrame:
        """Cap outliers to boundary values (winsorization)."""
        data = df[col].dropna()
        
        # Domain-specific bounds
        if col in self.DOMAIN_BOUNDS:
            lower, upper = self.DOMAIN_BOUNDS[col]
            if lower is not None or upper is not None:
                df[col] = df[col].clip(lower=lower, upper=upper)
                return df
        
        # Statistical bounds
        if method == OutlierMethod.IQR:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
        elif method == OutlierMethod.PERCENTILE:
            lower = data.quantile(percentile_bounds[0])
            upper = data.quantile(percentile_bounds[1])
        else:
            # Default to percentile for capping
            lower = data.quantile(0.01)
            upper = data.quantile(0.99)
        
        df[col] = df[col].clip(lower=lower, upper=upper)
        return df

# =============================================================================
# STAGE 5: MISSING DATA HANDLING 
# =============================================================================

class MissingDataStage(CleaningStage):
    """
    Handle missing data with multiple imputation strategies.
    
    Missing data causes:
    - Look-ahead bias if using future values
    - Incomplete time series for backtesting
    - Failed calculations (NaN propagation)
    
    Strategies are chosen based on:
    - Gap size (small gaps can be filled, large gaps should be flagged)
    - Data type (prices vs rates vs volumes)
    - Use case (backtest vs live trading)
    """
    
    @property
    def stage_name(self) -> str:
        return "missing_data_handling"
    
    def execute(
        self,
        df: pd.DataFrame,
        report: CleaningReport,
        data_type: DataType,
        method: MissingDataMethod = MissingDataMethod.FORWARD_FILL,
        max_fill_periods: int = 3,
        columns: List[str] = None,
        fill_limit_pct: float = 10.0,
        **kwargs
    ) -> pd.DataFrame:
        """Execute missing data handling."""
        import time
        start_time = time.time()
        
        df = df.copy()
        rows_before = len(df)
        
        # Determine columns to fill
        if columns is None:
            columns = [c for c in data_type.numeric_columns if c in df.columns]
        else:
            columns = [c for c in columns if c in df.columns]
        
        if not columns:
            report.stages_skipped.append(self.stage_name)
            return df
        
        # Sort by timestamp if present
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        total_filled = 0
        
        for col in columns:
            missing_before = df[col].isna().sum()
            
            if missing_before == 0:
                continue
            
            # Check fill limit
            missing_pct = (missing_before / len(df)) * 100
            if missing_pct > fill_limit_pct and method != MissingDataMethod.DROP:
                report.add_warning(
                    f"Column '{col}' has {missing_pct:.1f}% missing (>{fill_limit_pct}%), "
                    f"filling may introduce significant bias"
                )
            
            # Apply fill method
            if method == MissingDataMethod.FORWARD_FILL:
                df[col] = df[col].ffill(limit=max_fill_periods)
                action = CleaningAction.MISSING_FORWARD_FILLED
            
            elif method == MissingDataMethod.BACKWARD_FILL:
                df[col] = df[col].bfill(limit=max_fill_periods)
                action = CleaningAction.MISSING_BACKWARD_FILLED
            
            elif method == MissingDataMethod.LINEAR_INTERPOLATE:
                df[col] = df[col].interpolate(method='linear', limit=max_fill_periods)
                action = CleaningAction.MISSING_INTERPOLATED
            
            elif method == MissingDataMethod.SPLINE_INTERPOLATE:
                try:
                    df[col] = df[col].interpolate(method='spline', order=3, limit=max_fill_periods)
                except:
                    df[col] = df[col].interpolate(method='linear', limit=max_fill_periods)
                action = CleaningAction.MISSING_INTERPOLATED
            
            elif method == MissingDataMethod.MEAN:
                df[col] = df[col].fillna(df[col].mean())
                action = CleaningAction.MISSING_INTERPOLATED
            
            elif method == MissingDataMethod.MEDIAN:
                df[col] = df[col].fillna(df[col].median())
                action = CleaningAction.MISSING_INTERPOLATED
            
            elif method == MissingDataMethod.ZERO:
                df[col] = df[col].fillna(0)
                action = CleaningAction.MISSING_INTERPOLATED
            
            elif method == MissingDataMethod.DROP:
                df = df.dropna(subset=[col])
                action = CleaningAction.MISSING_DROPPED
            
            elif method == MissingDataMethod.SEASONAL:
                df = self._seasonal_fill(df, col, max_fill_periods)
                action = CleaningAction.MISSING_INTERPOLATED
            
            missing_after = df[col].isna().sum()
            filled = missing_before - missing_after
            total_filled += filled
            
            if filled > 0 or (method == MissingDataMethod.DROP and rows_before != len(df)):
                self._log_action(
                    report, action,
                    column=col,
                    rows_affected=int(filled) if method != MissingDataMethod.DROP else rows_before - len(df),
                    rows_before=rows_before,
                    rows_after=len(df),
                    details={
                        'method': method.value,
                        'max_fill_periods': max_fill_periods,
                        'missing_before': int(missing_before),
                        'missing_after': int(missing_after),
                        'filled': int(filled),
                        'fill_pct': round(filled / missing_before * 100, 2) if missing_before > 0 else 0
                    }
                )
        
        report.stages_completed.append(self.stage_name)
        report.stage_timings[self.stage_name] = time.time() - start_time
        
        return df
    
    def _seasonal_fill(
        self,
        df: pd.DataFrame,
        col: str,
        max_periods: int
    ) -> pd.DataFrame:
        """Fill using same time period from previous day/week."""
        if 'timestamp' not in df.columns:
            return df
        
        df = df.copy()
        
        # Try daily seasonality first
        for shift in [24, 24*7]: # 1 day, 1 week
            if df[col].isna().sum() == 0:
                break
            
            missing_mask = df[col].isna()
            
            # Create shifted version
            df_shifted = df.set_index('timestamp')[col].shift(shift, freq='H')
            
            # Fill from shifted
            for idx in df[missing_mask].index:
                ts = df.loc[idx, 'timestamp']
                if ts - pd.Timedelta(hours=shift) in df_shifted.index:
                    fill_val = df_shifted[ts - pd.Timedelta(hours=shift)]
                    if pd.notna(fill_val):
                        df.loc[idx, col] = fill_val
        
        return df

# =============================================================================
# STAGE 6: OHLCV CORRECTION
# =============================================================================

class OHLCVCorrectionStage(CleaningStage):
    """
    Correct OHLCV data integrity violations.
    
    OHLCV invariants that MUST hold:
    - high >= max(open, close)
    - low <= min(open, close)
    - high >= low
    - All prices > 0
    - volume >= 0
    
    Violations indicate data errors that must be fixed
    before backtesting.
    """
    
    @property
    def stage_name(self) -> str:
        return "ohlcv_correction"
    
    def execute(
        self,
        df: pd.DataFrame,
        report: CleaningReport,
        data_type: DataType,
        remove_invalid: bool = False,
        fix_strategy: str = 'adjust', # 'adjust' or 'remove'
        **kwargs
    ) -> pd.DataFrame:
        """Execute OHLCV correction."""
        import time
        start_time = time.time()
        
        # Only applies to OHLCV data
        if data_type != DataType.OHLCV:
            report.stages_skipped.append(self.stage_name)
            return df
        
        required = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required):
            report.add_warning("Missing OHLCV columns, skipping correction")
            report.stages_skipped.append(self.stage_name)
            return df
        
        df = df.copy()
        rows_before = len(df)
        total_fixed = 0
        
        # 1. Fix negative prices
        for col in required:
            neg_mask = df[col] <= 0
            neg_count = neg_mask.sum()
            if neg_count > 0:
                if remove_invalid:
                    df = df[~neg_mask]
                    self._log_action(
                        report, CleaningAction.ROW_REMOVED,
                        column=col,
                        rows_affected=int(neg_count),
                        rows_before=rows_before,
                        rows_after=len(df),
                        details={'issue': 'negative_price'}
                    )
                else:
                    df.loc[neg_mask, col] = np.nan
                    total_fixed += neg_count
        
        # 2. Fix high < low (swap them)
        hl_violation = df['high'] < df['low']
        if hl_violation.any():
            count = hl_violation.sum()
            df.loc[hl_violation, ['high', 'low']] = df.loc[hl_violation, ['low', 'high']].values
            total_fixed += count
            self._log_action(
                report, CleaningAction.OHLCV_FIXED,
                column='high/low',
                rows_affected=int(count),
                rows_before=rows_before,
                rows_after=len(df),
                details={'issue': 'high_less_than_low', 'action': 'swapped'}
            )
        
        # 3. Fix high < max(open, close)
        max_oc = df[['open', 'close']].max(axis=1)
        high_violation = df['high'] < max_oc
        if high_violation.any():
            count = high_violation.sum()
            df.loc[high_violation, 'high'] = max_oc[high_violation]
            total_fixed += count
            self._log_action(
                report, CleaningAction.OHLCV_FIXED,
                column='high',
                rows_affected=int(count),
                rows_before=rows_before,
                rows_after=len(df),
                details={'issue': 'high_less_than_max_oc', 'action': 'set_to_max'}
            )
        
        # 4. Fix low > min(open, close)
        min_oc = df[['open', 'close']].min(axis=1)
        low_violation = df['low'] > min_oc
        if low_violation.any():
            count = low_violation.sum()
            df.loc[low_violation, 'low'] = min_oc[low_violation]
            total_fixed += count
            self._log_action(
                report, CleaningAction.OHLCV_FIXED,
                column='low',
                rows_affected=int(count),
                rows_before=rows_before,
                rows_after=len(df),
                details={'issue': 'low_greater_than_min_oc', 'action': 'set_to_min'}
            )
        
        # 5. Fix negative volume
        if 'volume' in df.columns:
            neg_vol = df['volume'] < 0
            if neg_vol.any():
                count = neg_vol.sum()
                df.loc[neg_vol, 'volume'] = 0
                total_fixed += count
                self._log_action(
                    report, CleaningAction.VALUE_CORRECTED,
                    column='volume',
                    rows_affected=int(count),
                    rows_before=rows_before,
                    rows_after=len(df),
                    details={'issue': 'negative_volume', 'action': 'set_to_zero'}
                )
        
        report.stages_completed.append(self.stage_name)
        report.stage_timings[self.stage_name] = time.time() - start_time
        
        return df

# =============================================================================
# STAGE 7: SYMBOL NORMALIZATION
# =============================================================================

class SymbolNormalizationStage(CleaningStage):
    """
    Normalize symbol names across venues.
    
    Problem: Different venues use different naming conventions:
    - Binance: BTCUSDT
    - Bybit: BTCUSDT 
    - Hyperliquid: BTC
    - dYdX: BTC-USD
    
    Solution: Normalize to base asset only (BTC, ETH, SOL, etc.)
    """
    
    # Symbol aliases for common variations
    ALIASES: Dict[str, str] = {
        # Bitcoin variants
        'XBT': 'BTC', 'XBTC': 'BTC', 'WBTC': 'BTC', 'BTCB': 'BTC',
        'BTC.D': 'BTC',
        
        # Ethereum variants 
        'XETH': 'ETH', 'WETH': 'ETH', 'STETH': 'ETH', 'ETH2': 'ETH',
        'CBETH': 'ETH', 'RETH': 'ETH',
        
        # Terra variants
        'LUNA2': 'LUNA', 'LUNC': 'LUNA_CLASSIC',
        'USTC': 'UST_CLASSIC',
        
        # Other common
        'MATIC': 'POL', # Polygon rebrand
    }
    
    # Suffixes to strip
    STRIP_SUFFIXES: List[str] = [
        # Perpetual markers
        '-PERP', '_PERP', 'PERP', '-SWAP', '_SWAP', 'SWAP',
        '-PERPETUAL', '_PERPETUAL',
        
        # Quote currencies
        'USDT', 'BUSD', 'USDC', 'USD', 'TUSD', 'DAI', 'FDUSD', 'EUR', 'GBP',
        
        # Separators with quote
        '/USDT', '/USD', '/BUSD', '/USDC', '/EUR',
        ':USDT', ':USD', ':BUSD',
        '_USDT', '_USD', '_BUSD',
        '-USDT', '-USD', '-BUSD',
        
        # Index markers
        '.P', '-INDEX',
    ]
    
    @property
    def stage_name(self) -> str:
        return "symbol_normalization"
    
    def execute(
        self,
        df: pd.DataFrame,
        report: CleaningReport,
        data_type: DataType,
        preserve_original: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """Execute symbol normalization."""
        import time
        start_time = time.time()
        
        if 'symbol' not in df.columns:
            report.stages_skipped.append(self.stage_name)
            return df
        
        df = df.copy()
        rows_before = len(df)
        
        # Preserve original if requested
        if preserve_original and 'symbol_original' not in df.columns:
            df['symbol_original'] = df['symbol']
        
        original_symbols = df['symbol'].copy()
        
        # Normalize
        df['symbol'] = df['symbol'].apply(self._normalize)
        
        # Count changes
        changed = (df['symbol'] != original_symbols).sum()
        unique_before = original_symbols.nunique()
        unique_after = df['symbol'].nunique()
        
        if changed > 0:
            self._log_action(
                report, CleaningAction.SYMBOL_NORMALIZED,
                column='symbol',
                rows_affected=int(changed),
                rows_before=rows_before,
                rows_after=len(df),
                details={
                    'symbols_modified': int(changed),
                    'unique_before': int(unique_before),
                    'unique_after': int(unique_after),
                    'consolidation': int(unique_before - unique_after)
                }
            )
        
        report.stages_completed.append(self.stage_name)
        report.stage_timings[self.stage_name] = time.time() - start_time
        
        return df
    
    def _normalize(self, symbol: Any) -> str:
        """Normalize a single symbol."""
        if pd.isna(symbol) or not symbol:
            return 'UNKNOWN'
        
        s = str(symbol).upper().strip()
        
        # Remove suffixes (longest first to avoid partial matches)
        for suffix in sorted(self.STRIP_SUFFIXES, key=len, reverse=True):
            if s.endswith(suffix.upper()):
                s = s[:-len(suffix)]
                break
        
        # Remove common prefixes
        for prefix in ['1000', '10000', 'MINI', 'MICRO']:
            if s.startswith(prefix):
                s = s[len(prefix):]
                break
        
        # Apply aliases
        s = self.ALIASES.get(s, s)
        
        # Clean up any remaining artifacts
        s = s.strip('_-/')
        
        return s if s else 'UNKNOWN'

# =============================================================================
# STAGE 8: FUNDING RATE NORMALIZATION
# =============================================================================

class FundingRateNormalizationStage(CleaningStage):
    """
    Normalize funding rates across venues with different intervals.
    
    Critical for cross-venue arbitrage:
    - Binance: 8-hour funding
    - Hyperliquid: 1-hour funding
    
    Without normalization, comparing rates is meaningless!
    
    Conversions:
    - All rates converted to 8-hour equivalent
    - Annualized rate calculated for comparison
    """
    
    # Venue funding intervals in hours
    VENUE_INTERVALS: Dict[str, float] = {
        'binance': 8.0,
        'bybit': 8.0,
        'okx': 8.0,
        'deribit': 8.0,
        'kraken': 4.0,
        'bitmex': 8.0,
        'hyperliquid': 1.0,
        'dydx': 1.0,
        'dydx_v4': 1.0,
        'gmx': 1.0,
        'vertex': 1.0,
        'synthetix': 1.0,
        'aevo': 8.0,
        'drift': 1.0,
        'perp': 1.0,
    }
    
    TARGET_INTERVAL_HOURS = 8.0
    HOURS_PER_YEAR = 8760.0
    
    @property
    def stage_name(self) -> str:
        return "funding_rate_normalization"
    
    def execute(
        self,
        df: pd.DataFrame,
        report: CleaningReport,
        data_type: DataType,
        venue: str = None,
        target_interval_hours: float = 8.0,
        **kwargs
    ) -> pd.DataFrame:
        """Execute funding rate normalization."""
        import time
        start_time = time.time()
        
        if data_type != DataType.FUNDING_RATES:
            report.stages_skipped.append(self.stage_name)
            return df
        
        if 'funding_rate' not in df.columns:
            report.add_warning("No funding_rate column found")
            report.stages_skipped.append(self.stage_name)
            return df
        
        df = df.copy()
        rows_before = len(df)
        
        # Determine source interval
        if venue:
            source_interval = self.VENUE_INTERVALS.get(venue.lower(), 8.0)
        elif 'venue' in df.columns:
            # Use venue column, defaulting to 8h
            df['_src_interval'] = df['venue'].str.lower().map(self.VENUE_INTERVALS).fillna(8.0)
        else:
            source_interval = 8.0
            report.add_warning("Unknown venue, assuming 8-hour funding interval")
        
        # Calculate conversion factor
        if 'venue' in df.columns and '_src_interval' in df.columns:
            # Per-row conversion
            conversion_factor = target_interval_hours / df['_src_interval']
            df['funding_rate_8h'] = df['funding_rate'] * conversion_factor
            df = df.drop(columns=['_src_interval'])
            
            self._log_action(
                report, CleaningAction.RATE_NORMALIZED,
                column='funding_rate_8h',
                rows_affected=len(df),
                rows_before=rows_before,
                rows_after=len(df),
                details={
                    'target_interval_hours': target_interval_hours,
                    'per_venue_conversion': True,
                    'venue_intervals': {v: i for v, i in self.VENUE_INTERVALS.items() if v in df['venue'].str.lower().unique()}
                }
            )
        else:
            # Single venue conversion
            conversion_factor = target_interval_hours / source_interval
            
            if abs(conversion_factor - 1.0) > 0.001:
                df['funding_rate_8h'] = df['funding_rate'] * conversion_factor
                
                self._log_action(
                    report, CleaningAction.RATE_NORMALIZED,
                    column='funding_rate_8h',
                    rows_affected=len(df),
                    rows_before=rows_before,
                    rows_after=len(df),
                    details={
                        'source_interval_hours': source_interval,
                        'target_interval_hours': target_interval_hours,
                        'conversion_factor': round(conversion_factor, 4),
                        'venue': venue
                    }
                )
            else:
                df['funding_rate_8h'] = df['funding_rate']
        
        # Calculate annualized rate
        periods_per_year = self.HOURS_PER_YEAR / target_interval_hours
        df['funding_rate_annualized'] = df['funding_rate_8h'] * periods_per_year
        
        self._log_action(
            report, CleaningAction.COLUMN_ADDED,
            column='funding_rate_annualized',
            rows_affected=len(df),
            rows_before=rows_before,
            rows_after=len(df),
            details={'periods_per_year': periods_per_year}
        )
        
        report.stages_completed.append(self.stage_name)
        report.stage_timings[self.stage_name] = time.time() - start_time
        
        return df
# =============================================================================
# MAIN DATA CLEANER CLASS
# =============================================================================

class DataCleaner:
    """
    Main data cleaning orchestrator with sensible defaults.
    
    Provides a simple interface for cleaning data while allowing
    full customization when needed.
    
    Example:
        >>> cleaner = DataCleaner()
        >>> cleaned_df, report = cleaner.clean(
        ... df,
        ... data_type='funding_rates',
        ... venue='binance'
        ... )
        >>> print(report.summary())
        
        >>> # With custom options
        >>> cleaned_df, report = cleaner.clean(
        ... df,
        ... data_type='ohlcv',
        ... venue='bybit',
        ... outlier_action=OutlierAction.CAP_WINSORIZE,
        ... missing_method=MissingDataMethod.FORWARD_FILL,
        ... max_fill_periods=5
        ... )
    """
    
    def __init__(self, config: Dict[str, Any] = None, use_gpu: bool = True):
        """
        Initialize DataCleaner.

        Args:
            config: Global configuration options
            use_gpu: Whether to use GPU acceleration if available (default: True)
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize GPU accelerator if available and requested
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.gpu_accelerator = None
        if self.use_gpu:
            try:
                self.gpu_accelerator = get_accelerator()
                self.logger.info("GPU acceleration enabled for DataCleaner")
            except Exception as e:
                self.logger.warning(f"GPU initialization failed, using CPU: {e}")
                self.use_gpu = False
    
    def clean(
        self,
        df: pd.DataFrame,
        data_type: Union[str, DataType] = DataType.FUNDING_RATES,
        venue: str = 'unknown',
        # Stage options
        dedup_strategy: DeduplicationStrategy = DeduplicationStrategy.KEEP_LAST,
        timestamp_method: TimestampAlignment = TimestampAlignment.FLOOR,
        outlier_action: OutlierAction = OutlierAction.CAP_WINSORIZE,
        outlier_method: OutlierMethod = OutlierMethod.IQR,
        outlier_threshold: float = 3.0,
        missing_method: MissingDataMethod = MissingDataMethod.FORWARD_FILL,
        max_fill_periods: int = 3,
        # Stage control
        skip_stages: List[str] = None,
        # Data options
        interval: str = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """
        Clean DataFrame with configurable pipeline.
        
        Args:
            df: Raw DataFrame to clean
            data_type: Type of data ('funding_rates', 'ohlcv', etc.)
            venue: Source venue name
            dedup_strategy: How to handle duplicates
            timestamp_method: How to align timestamps
            outlier_action: What to do with outliers
            outlier_method: How to detect outliers
            outlier_threshold: Sensitivity for outlier detection
            missing_method: How to handle missing data
            max_fill_periods: Maximum gaps to fill
            skip_stages: List of stage names to skip
            interval: Expected data interval (e.g., '8h', '1h')
            
        Returns:
            Tuple of (cleaned DataFrame, CleaningReport)
        """
        import time
        start_time = time.time()
        
        # Normalize data_type
        if isinstance(data_type, str):
            try:
                data_type = DataType(data_type)
            except ValueError:
                self.logger.warning(f"Unknown data type '{data_type}', using FUNDING_RATES")
                data_type = DataType.FUNDING_RATES
        
        # Initialize report
        report = CleaningReport(
            data_type=data_type,
            venue=venue,
            original_rows=len(df),
            final_rows=len(df),
            original_columns=len(df.columns),
            final_columns=len(df.columns),
            original_fingerprint=DataFingerprint.from_dataframe(df),
            config_used={
                'data_type': data_type.value,
                'venue': venue,
                'dedup_strategy': dedup_strategy.value,
                'timestamp_method': timestamp_method.value,
                'outlier_action': outlier_action.value,
                'outlier_method': outlier_method.value,
                'outlier_threshold': outlier_threshold,
                'missing_method': missing_method.value,
                'max_fill_periods': max_fill_periods,
                'skip_stages': skip_stages or [],
            }
        )
        
        skip_stages = set(skip_stages or [])
        
        # Build and execute pipeline
        stages = self._build_pipeline(data_type, skip_stages)
        
        cleaned_df = df.copy()
        
        for stage in stages:
            if stage.stage_name in skip_stages:
                report.stages_skipped.append(stage.stage_name)
                continue
            
            try:
                cleaned_df = stage.execute(
                    cleaned_df,
                    report,
                    data_type,
                    venue=venue,
                    # Stage-specific options
                    strategy=dedup_strategy,
                    method=timestamp_method,
                    action=outlier_action,
                    threshold=outlier_threshold,
                    max_fill_periods=max_fill_periods,
                    interval=interval,
                    **kwargs
                )
            except Exception as e:
                report.add_error(f"Stage {stage.stage_name} failed: {str(e)}")
                report.stages_failed.append(stage.stage_name)
                self.logger.exception(f"Stage {stage.stage_name} failed")
        
        # Finalize report
        report.final_rows = len(cleaned_df)
        report.final_columns = len(cleaned_df.columns)
        report.final_fingerprint = DataFingerprint.from_dataframe(cleaned_df)
        report.execution_time_seconds = time.time() - start_time
        
        return cleaned_df, report
    
    def _build_pipeline(
        self,
        data_type: DataType,
        skip_stages: Set[str]
    ) -> List[CleaningStage]:
        """Build cleaning pipeline for data type."""
        stages = [
            SchemaEnforcementStage(),
            DeduplicationStage(),
            TemporalAlignmentStage(),
        ]
        
        # OHLCV-specific
        if data_type == DataType.OHLCV:
            stages.append(OHLCVCorrectionStage())
        
        # Common stages
        stages.extend([
            OutlierTreatmentStage(),
            MissingDataStage(),
            SymbolNormalizationStage(),
        ])
        
        # Funding rate specific
        if data_type == DataType.FUNDING_RATES:
            stages.append(FundingRateNormalizationStage())
        
        return stages
    
    def validate_and_clean(
        self,
        df: pd.DataFrame,
        data_type: Union[str, DataType] = DataType.FUNDING_RATES,
        venue: str = 'unknown',
        **kwargs
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """
        Validate data, clean it, then validate again.
        
        Provides quality scores before and after cleaning.
        """
        # Import validator (optional dependency)
        try:
            from data_validator import DataValidator, DataType as ValidatorDataType
            has_validator = True
        except ImportError:
            has_validator = False
        
        if isinstance(data_type, str):
            data_type = DataType(data_type)
        
        # Pre-cleaning validation
        if has_validator:
            try:
                validator = DataValidator()
                validator_dtype = ValidatorDataType(data_type.value)
                pre_result = validator.validate(df, validator_dtype)
                pre_score = pre_result.overall_score
            except Exception as e:
                self.logger.warning(f"Pre-validation failed: {e}")
                pre_score = None
        else:
            pre_score = None
        
        # Clean
        cleaned_df, report = self.clean(df, data_type, venue, **kwargs)
        report.quality_score_before = pre_score
        
        # Post-cleaning validation
        if has_validator and len(cleaned_df) > 0:
            try:
                post_result = validator.validate(cleaned_df, validator_dtype)
                report.quality_score_after = post_result.overall_score
            except Exception as e:
                self.logger.warning(f"Post-validation failed: {e}")
        
        return cleaned_df, report

# =============================================================================
# CONFIGURABLE CLEANING PIPELINE
# =============================================================================

class CleaningPipeline:
    """
    Fully configurable cleaning pipeline.
    
    Allows fine-grained control over:
    - Which stages to run
    - Stage ordering
    - Per-stage configuration
    
    Example:
        >>> pipeline = CleaningPipeline()
        >>> pipeline.add_stage(SchemaEnforcementStage())
        >>> pipeline.add_stage(DeduplicationStage())
        >>> pipeline.configure('deduplication', strategy=DeduplicationStrategy.KEEP_BEST)
        >>> pipeline.configure('outlier_treatment', action=OutlierAction.REMOVE, threshold=5.0)
        >>> cleaned_df, report = pipeline.execute(df, data_type='funding_rates')
    """
    
    def __init__(self):
        """Initialize empty pipeline."""
        self.stages: List[CleaningStage] = []
        self.stage_configs: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_stage(
        self,
        stage: CleaningStage,
        position: int = None
    ) -> 'CleaningPipeline':
        """
        Add a stage to the pipeline.
        
        Args:
            stage: CleaningStage instance
            position: Position in pipeline (None = append)
            
        Returns:
            Self for chaining
        """
        if position is None:
            self.stages.append(stage)
        else:
            self.stages.insert(position, stage)
        return self
    
    def remove_stage(self, stage_name: str) -> 'CleaningPipeline':
        """Remove a stage by name."""
        self.stages = [s for s in self.stages if s.stage_name != stage_name]
        return self
    
    def configure(self, stage_name: str, **kwargs) -> 'CleaningPipeline':
        """
        Configure a stage with specific parameters.
        
        Args:
            stage_name: Name of stage to configure
            **kwargs: Configuration parameters
            
        Returns:
            Self for chaining
        """
        self.stage_configs[stage_name] = kwargs
        return self
    
    def get_stage(self, stage_name: str) -> Optional[CleaningStage]:
        """Get a stage by name."""
        for stage in self.stages:
            if stage.stage_name == stage_name:
                return stage
        return None
    
    def list_stages(self) -> List[str]:
        """List all stage names in order."""
        return [s.stage_name for s in self.stages]
    
    def build_default(
        self,
        data_type: DataType = DataType.FUNDING_RATES
    ) -> 'CleaningPipeline':
        """
        Build default pipeline for data type.
        
        Args:
            data_type: Type of data to clean
            
        Returns:
            Self for chaining
        """
        self.stages = [
            SchemaEnforcementStage(),
            DeduplicationStage(),
            TemporalAlignmentStage(),
            OutlierTreatmentStage(),
            MissingDataStage(),
            SymbolNormalizationStage(),
        ]
        
        if data_type == DataType.OHLCV:
            self.stages.insert(3, OHLCVCorrectionStage())
        
        if data_type == DataType.FUNDING_RATES:
            self.stages.append(FundingRateNormalizationStage())
        
        return self
    
    def execute(
        self,
        df: pd.DataFrame,
        data_type: Union[str, DataType] = DataType.FUNDING_RATES,
        venue: str = 'unknown',
        **kwargs
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """
        Execute the configured pipeline.
        
        Args:
            df: DataFrame to clean
            data_type: Type of data
            venue: Source venue
            **kwargs: Additional options passed to all stages
            
        Returns:
            Tuple of (cleaned DataFrame, CleaningReport)
        """
        import time
        start_time = time.time()
        
        if isinstance(data_type, str):
            data_type = DataType(data_type)
        
        report = CleaningReport(
            data_type=data_type,
            venue=venue,
            original_rows=len(df),
            final_rows=len(df),
            original_columns=len(df.columns),
            final_columns=len(df.columns),
            original_fingerprint=DataFingerprint.from_dataframe(df),
            config_used={
                'stages': self.list_stages(),
                'stage_configs': self.stage_configs,
            }
        )
        
        cleaned_df = df.copy()
        
        for stage in self.stages:
            # Merge global and stage-specific configs
            stage_config = self.stage_configs.get(stage.stage_name, {})
            merged_kwargs = {**kwargs, **stage_config}
            
            try:
                cleaned_df = stage.execute(
                    cleaned_df,
                    report,
                    data_type,
                    venue=venue,
                    **merged_kwargs
                )
            except Exception as e:
                report.add_error(f"Stage {stage.stage_name} failed: {str(e)}")
                report.stages_failed.append(stage.stage_name)
                self.logger.exception(f"Stage {stage.stage_name} failed")
        
        report.final_rows = len(cleaned_df)
        report.final_columns = len(cleaned_df.columns)
        report.final_fingerprint = DataFingerprint.from_dataframe(cleaned_df)
        report.execution_time_seconds = time.time() - start_time
        
        return cleaned_df, report

# =============================================================================
# CROSS-VENUE CLEANER
# =============================================================================

class CrossVenueCleaner:
    """
    Clean and align data across multiple venues.
    
    Essential for cross-venue arbitrage strategies where
    data from different sources must be:
    1. Cleaned individually
    2. Aligned to common timestamps
    3. Normalized for comparison
    
    Example:
        >>> cleaner = CrossVenueCleaner()
        >>> cleaned, reports = cleaner.clean_and_align(
        ... {
        ... 'binance': df_binance,
        ... 'hyperliquid': df_hyperliquid,
        ... 'dydx': df_dydx
        ... },
        ... data_type='funding_rates',
        ... reference_venue='binance'
        ... )
    """
    
    def __init__(self, cleaner_config: Dict[str, Any] = None):
        """
        Initialize CrossVenueCleaner.
        
        Args:
            cleaner_config: Configuration for individual venue cleaners
        """
        self.cleaner = DataCleaner(cleaner_config)
        self.logger = logging.getLogger(__name__)
    
    def clean_and_align(
        self,
        venue_dataframes: Dict[str, pd.DataFrame],
        data_type: Union[str, DataType] = DataType.FUNDING_RATES,
        reference_venue: str = None,
        align_timestamps: bool = True,
        common_symbols_only: bool = False,
        **kwargs
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, CleaningReport]]:
        """
        Clean multiple venues and optionally align.
        
        Args:
            venue_dataframes: Dict of {venue_name: DataFrame}
            data_type: Type of data
            reference_venue: Venue to use as timestamp reference
            align_timestamps: Whether to align timestamps across venues
            common_symbols_only: Filter to symbols present in all venues
            **kwargs: Options passed to individual cleaners
            
        Returns:
            Tuple of (cleaned DataFrames dict, reports dict)
        """
        if isinstance(data_type, str):
            data_type = DataType(data_type)
        
        cleaned_dfs: Dict[str, pd.DataFrame] = {}
        reports: Dict[str, CleaningReport] = {}
        
        # Step 1: Clean each venue individually
        for venue, df in venue_dataframes.items():
            self.logger.info(f"Cleaning {venue}: {len(df)} rows")
            
            cleaned_df, report = self.cleaner.clean(
                df,
                data_type=data_type,
                venue=venue,
                **kwargs
            )
            
            cleaned_dfs[venue] = cleaned_df
            reports[venue] = report
            
            self.logger.info(
                f"Cleaned {venue}: {report.original_rows} â†’ {report.final_rows} rows "
                f"({report.rows_removed_pct:.1f}% removed)"
            )
        
        # Step 2: Filter to common symbols if requested
        if common_symbols_only and len(cleaned_dfs) > 1:
            cleaned_dfs = self._filter_common_symbols(cleaned_dfs)
        
        # Step 3: Align timestamps if requested
        if align_timestamps and len(cleaned_dfs) > 1:
            cleaned_dfs = self._align_timestamps(cleaned_dfs, reference_venue)
        
        return cleaned_dfs, reports
    
    def _filter_common_symbols(
        self,
        venue_dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Filter to symbols present in all venues."""
        all_symbols = None
        
        for venue, df in venue_dfs.items():
            if 'symbol' in df.columns:
                symbols = set(df['symbol'].unique())
                if all_symbols is None:
                    all_symbols = symbols
                else:
                    all_symbols &= symbols
        
        if not all_symbols:
            self.logger.warning("No common symbols found across venues")
            return venue_dfs
        
        self.logger.info(f"Filtering to {len(all_symbols)} common symbols")
        
        filtered_dfs = {}
        for venue, df in venue_dfs.items():
            if 'symbol' in df.columns:
                filtered_dfs[venue] = df[df['symbol'].isin(all_symbols)].copy()
            else:
                filtered_dfs[venue] = df.copy()
        
        return filtered_dfs
    
    def _align_timestamps(
        self,
        venue_dfs: Dict[str, pd.DataFrame],
        reference_venue: str = None
    ) -> Dict[str, pd.DataFrame]:
        """Align timestamps across venues."""
        if not venue_dfs:
            return venue_dfs
        
        # Determine reference timestamps
        if reference_venue and reference_venue in venue_dfs:
            ref_df = venue_dfs[reference_venue]
            if 'timestamp' in ref_df.columns:
                reference_timestamps = set(ref_df['timestamp'].dropna().unique())
            else:
                self.logger.warning("Reference venue has no timestamp column")
                return venue_dfs
        else:
            # Use intersection of all timestamps
            reference_timestamps = None
            for df in venue_dfs.values():
                if 'timestamp' in df.columns:
                    ts = set(df['timestamp'].dropna().unique())
                    if reference_timestamps is None:
                        reference_timestamps = ts
                    else:
                        reference_timestamps &= ts
        
        if not reference_timestamps:
            self.logger.warning("No common timestamps found")
            return venue_dfs
        
        self.logger.info(f"Aligning to {len(reference_timestamps)} common timestamps")
        
        aligned_dfs = {}
        for venue, df in venue_dfs.items():
            if 'timestamp' in df.columns:
                aligned_dfs[venue] = df[df['timestamp'].isin(reference_timestamps)].copy()
            else:
                aligned_dfs[venue] = df.copy()
        
        return aligned_dfs
    
    def merge_venues(
        self,
        venue_dataframes: Dict[str, pd.DataFrame],
        merge_on: List[str] = None,
        suffixes: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        Merge cleaned venue data into single DataFrame.
        
        Args:
            venue_dataframes: Dict of {venue: DataFrame}
            merge_on: Columns to merge on (default: timestamp, symbol)
            suffixes: Venue suffixes for columns (default: _venuename)
            
        Returns:
            Merged DataFrame
        """
        if not venue_dataframes:
            return pd.DataFrame()
        
        merge_on = merge_on or ['timestamp', 'symbol']
        
        venues = list(venue_dataframes.keys())
        
        # Start with first venue
        merged = venue_dataframes[venues[0]].copy()
        
        # Add suffix to non-key columns
        suffix = suffixes.get(venues[0], f'_{venues[0]}') if suffixes else f'_{venues[0]}'
        for col in merged.columns:
            if col not in merge_on:
                merged = merged.rename(columns={col: f'{col}{suffix}'})
        
        # Merge remaining venues
        for venue in venues[1:]:
            df = venue_dataframes[venue].copy()
            suffix = suffixes.get(venue, f'_{venue}') if suffixes else f'_{venue}'
            
            for col in df.columns:
                if col not in merge_on:
                    df = df.rename(columns={col: f'{col}{suffix}'})
            
            merged = merged.merge(
                df,
                on=[c for c in merge_on if c in merged.columns and c in df.columns],
                how='outer'
            )
        
        return merged

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def clean_funding_rates(
    df: pd.DataFrame,
    venue: str = 'unknown',
    outlier_action: OutlierAction = OutlierAction.CAP_WINSORIZE,
    missing_method: MissingDataMethod = MissingDataMethod.FORWARD_FILL,
    max_fill_periods: int = 3
) -> Tuple[pd.DataFrame, CleaningReport]:
    """
    Clean funding rate data with recommended settings.
    
    Args:
        df: Raw funding rate data
        venue: Source venue
        outlier_action: How to handle outliers
        missing_method: How to fill missing data
        max_fill_periods: Maximum periods to fill
        
    Returns:
        Tuple of (cleaned DataFrame, CleaningReport)
    """
    cleaner = DataCleaner()
    return cleaner.clean(
        df,
        data_type=DataType.FUNDING_RATES,
        venue=venue,
        outlier_action=outlier_action,
        missing_method=missing_method,
        max_fill_periods=max_fill_periods
    )

def clean_ohlcv(
    df: pd.DataFrame,
    venue: str = 'unknown',
    fix_violations: bool = True,
    remove_invalid: bool = False,
    outlier_action: OutlierAction = OutlierAction.CAP_WINSORIZE
) -> Tuple[pd.DataFrame, CleaningReport]:
    """
    Clean OHLCV data with recommended settings.
    
    Args:
        df: Raw OHLCV data
        venue: Source venue
        fix_violations: Whether to fix OHLCV violations
        remove_invalid: Whether to remove unfixable rows
        outlier_action: How to handle outliers
        
    Returns:
        Tuple of (cleaned DataFrame, CleaningReport)
    """
    cleaner = DataCleaner()
    return cleaner.clean(
        df,
        data_type=DataType.OHLCV,
        venue=venue,
        outlier_action=outlier_action,
        remove_invalid=remove_invalid
    )

def clean_options(
    df: pd.DataFrame,
    venue: str = 'deribit',
    validate_greeks: bool = True
) -> Tuple[pd.DataFrame, CleaningReport]:
    """Clean options data."""
    cleaner = DataCleaner()
    return cleaner.clean(
        df,
        data_type=DataType.OPTIONS,
        venue=venue
    )

def clean_dex_pools(
    df: pd.DataFrame,
    chain: str = 'ethereum',
    min_tvl: float = 0.0
) -> Tuple[pd.DataFrame, CleaningReport]:
    """Clean DEX pool data."""
    cleaner = DataCleaner()
    cleaned_df, report = cleaner.clean(
        df,
        data_type=DataType.DEX_POOLS,
        venue=chain
    )
    
    # Additional TVL filter
    if min_tvl > 0 and 'tvl_usd' in cleaned_df.columns:
        cleaned_df = cleaned_df[cleaned_df['tvl_usd'] >= min_tvl]
    
    return cleaned_df, report

def clean_multi_venue(
    venue_dataframes: Dict[str, pd.DataFrame],
    data_type: Union[str, DataType] = DataType.FUNDING_RATES,
    align: bool = True,
    reference_venue: str = None
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, CleaningReport]]:
    """
    Clean multiple venue DataFrames.
    
    Args:
        venue_dataframes: Dict of {venue: DataFrame}
        data_type: Type of data
        align: Whether to align timestamps
        reference_venue: Reference venue for alignment
        
    Returns:
        Tuple of (cleaned dfs dict, reports dict)
    """
    cleaner = CrossVenueCleaner()
    return cleaner.clean_and_align(
        venue_dataframes,
        data_type=data_type,
        align_timestamps=align,
        reference_venue=reference_venue
    )

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'DataType',
    'CleaningAction',
    'DeduplicationStrategy',
    'OutlierAction',
    'OutlierMethod',
    'MissingDataMethod',
    'TimestampAlignment',
    
    # Data classes
    'CleaningLogEntry',
    'CleaningReport',
    'DataFingerprint',
    
    # Stage base class
    'CleaningStage',
    
    # Cleaning stages
    'SchemaEnforcementStage',
    'DeduplicationStage',
    'TemporalAlignmentStage',
    'OutlierTreatmentStage',
    'MissingDataStage',
    'OHLCVCorrectionStage',
    'SymbolNormalizationStage',
    'FundingRateNormalizationStage',
    
    # Main classes
    'DataCleaner',
    'CleaningPipeline',
    'CrossVenueCleaner',
    
    # Convenience functions
    'clean_funding_rates',
    'clean_ohlcv',
    'clean_options',
    'clean_dex_pools',
    'clean_multi_venue',
]
