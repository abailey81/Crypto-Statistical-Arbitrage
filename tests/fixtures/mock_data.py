"""
Mock Data Generation for Crypto Statistical Arbitrage Testing
=============================================================

Professional-quality synthetic data generation for testing data collection,
validation, storage, and strategy components without API calls.

Mathematical Framework
----------------------
Price Generation (Geometric Brownian Motion with Jumps):

    dS/S = μdt + σdW + J × dN
    
    Where:
        μ = drift coefficient
        σ = volatility (with GARCH clustering)
        W = Wiener process
        J = jump size (log-normal)
        N = Poisson process for jump arrivals

Funding Rate Generation (Mean-Reverting with Regime Switching):

    dF = κ(θ_regime - F)dt + σ_F × dW + ε_venue
    
    Where:
        κ = mean reversion speed (~0.3 for 8h funding)
        θ_regime = regime-dependent mean
        σ_F = funding rate volatility
        ε_venue = venue-specific noise

Volume Generation (U-Shaped Intraday with Day-of-Week Effects):

    V(t) = V_base × f_hour(h) × f_dow(d) × f_regime(r) × ε
    
    Where:
        f_hour = U-shaped intraday pattern
        f_dow = day-of-week seasonality
        f_regime = regime multiplier
        ε = log-normal noise

Cross-Venue Correlation Structure:

    F_venue_i = ρ × F_base + √(1-ρ²) × ε_i
    
    Where ρ ~ 0.85-0.95 for major CEX venues

Key Features
------------
1. Realistic Market Dynamics:
   - GARCH volatility clustering
   - Jump-diffusion price processes
   - Regime-switching funding rates
   - Cross-venue correlation structure

2. Comprehensive Data Types:
   - Funding rates (8h CEX, 1h hybrid)
   - OHLCV with microstructure effects
   - Options chains with volatility smile
   - DEX pools with wash trading patterns
   - On-chain metrics (TVL, gas, liquidations)

3. Quality Control Scenarios:
   - Configurable data gaps
   - Outlier injection
   - Stale price detection
   - Survivorship bias simulation

4. Testing Utilities:
   - Quick data generation functions
   - Parameterized fixtures
   - Reproducible random states

Author: Crypto StatArb Quantitative Research
Version: 2.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import random
import string
import hashlib
import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS WITH PROPERTIES
# =============================================================================

class MarketRegime(Enum):
    """
    Market regime classification for realistic data generation.
    
    Each regime has specific statistical properties that affect
    funding rates, volatility, and volume patterns.
    """
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    
    @property
    def funding_mean(self) -> float:
        """Mean funding rate for regime (per 8h)."""
        means = {
            self.BULL: 0.0003,      # 0.03% per 8h = ~0.33% daily
            self.BEAR: -0.0001,     # Negative in bear markets
            self.NEUTRAL: 0.0001,   # Slight positive bias
            self.HIGH_VOL: 0.0002,  # Higher variance around mean
            self.LOW_VOL: 0.00005,  # Very stable
            self.CRISIS: -0.0005,   # Extreme negative
            self.RECOVERY: 0.0002,  # Post-crisis recovery
        }
        return means.get(self, 0.0001)
    
    @property
    def funding_std(self) -> float:
        """Standard deviation of funding rate."""
        stds = {
            self.BULL: 0.0002,
            self.BEAR: 0.0003,
            self.NEUTRAL: 0.00015,
            self.HIGH_VOL: 0.0005,
            self.LOW_VOL: 0.00005,
            self.CRISIS: 0.001,
            self.RECOVERY: 0.0003,
        }
        return stds.get(self, 0.0002)
    
    @property
    def volatility_multiplier(self) -> float:
        """Multiplier for asset volatility."""
        multipliers = {
            self.BULL: 1.0,
            self.BEAR: 1.3,
            self.NEUTRAL: 0.8,
            self.HIGH_VOL: 1.8,
            self.LOW_VOL: 0.5,
            self.CRISIS: 2.5,
            self.RECOVERY: 1.2,
        }
        return multipliers.get(self, 1.0)
    
    @property
    def volume_multiplier(self) -> float:
        """Multiplier for trading volume."""
        multipliers = {
            self.BULL: 1.3,
            self.BEAR: 1.1,
            self.NEUTRAL: 0.9,
            self.HIGH_VOL: 1.5,
            self.LOW_VOL: 0.6,
            self.CRISIS: 2.0,
            self.RECOVERY: 1.2,
        }
        return multipliers.get(self, 1.0)
    
    @property
    def drift(self) -> float:
        """Price drift coefficient (annualized)."""
        drifts = {
            self.BULL: 0.50,    # 50% annual drift
            self.BEAR: -0.30,   # -30% annual drift
            self.NEUTRAL: 0.05,
            self.HIGH_VOL: 0.10,
            self.LOW_VOL: 0.02,
            self.CRISIS: -0.80,
            self.RECOVERY: 0.40,
        }
        return drifts.get(self, 0.0)
    
    @property
    def min_duration_days(self) -> int:
        """Minimum regime duration in days."""
        durations = {
            self.BULL: 30,
            self.BEAR: 21,
            self.NEUTRAL: 14,
            self.HIGH_VOL: 7,
            self.LOW_VOL: 21,
            self.CRISIS: 3,
            self.RECOVERY: 7,
        }
        return durations.get(self, 14)
    
    @property
    def transition_probs(self) -> Dict['MarketRegime', float]:
        """Transition probabilities to other regimes."""
        probs = {
            self.BULL: {self.BULL: 0.7, self.NEUTRAL: 0.15, self.HIGH_VOL: 0.1, self.BEAR: 0.05},
            self.BEAR: {self.BEAR: 0.6, self.NEUTRAL: 0.2, self.CRISIS: 0.1, self.RECOVERY: 0.1},
            self.NEUTRAL: {self.NEUTRAL: 0.5, self.BULL: 0.25, self.BEAR: 0.15, self.LOW_VOL: 0.1},
            self.HIGH_VOL: {self.HIGH_VOL: 0.4, self.CRISIS: 0.2, self.NEUTRAL: 0.3, self.BULL: 0.1},
            self.LOW_VOL: {self.LOW_VOL: 0.6, self.NEUTRAL: 0.3, self.BULL: 0.1},
            self.CRISIS: {self.CRISIS: 0.3, self.RECOVERY: 0.5, self.BEAR: 0.2},
            self.RECOVERY: {self.RECOVERY: 0.4, self.BULL: 0.3, self.NEUTRAL: 0.3},
        }
        return probs.get(self, {self.NEUTRAL: 1.0})
    
    @property
    def color_code(self) -> str:
        """Color for visualization."""
        colors = {
            self.BULL: "#00FF00",
            self.BEAR: "#FF0000",
            self.NEUTRAL: "#808080",
            self.HIGH_VOL: "#FFA500",
            self.LOW_VOL: "#0000FF",
            self.CRISIS: "#8B0000",
            self.RECOVERY: "#90EE90",
        }
        return colors.get(self, "#808080")


class VenueType(Enum):
    """
    Venue type classification affecting data characteristics.
    """
    CEX = "CEX"
    HYBRID = "HYBRID"
    DEX = "DEX"
    OPTIONS = "OPTIONS"
    ON_CHAIN = "ON_CHAIN"
    
    @property
    def funding_interval_hours(self) -> int:
        """Funding rate interval in hours."""
        intervals = {
            self.CEX: 8,
            self.HYBRID: 1,
            self.DEX: 0,  # No funding
            self.OPTIONS: 0,
            self.ON_CHAIN: 0,
        }
        return intervals.get(self, 8)
    
    @property
    def typical_latency_ms(self) -> float:
        """Typical data latency in milliseconds."""
        latencies = {
            self.CEX: 50,
            self.HYBRID: 200,
            self.DEX: 2000,
            self.OPTIONS: 100,
            self.ON_CHAIN: 12000,  # Block time
        }
        return latencies.get(self, 100)
    
    @property
    def price_precision(self) -> int:
        """Price decimal precision."""
        precisions = {
            self.CEX: 2,
            self.HYBRID: 4,
            self.DEX: 8,
            self.OPTIONS: 2,
            self.ON_CHAIN: 18,
        }
        return precisions.get(self, 4)
    
    @property
    def volume_noise_factor(self) -> float:
        """Volume noise factor (higher = noisier)."""
        noise = {
            self.CEX: 0.2,
            self.HYBRID: 0.3,
            self.DEX: 0.5,
            self.OPTIONS: 0.4,
            self.ON_CHAIN: 0.1,
        }
        return noise.get(self, 0.3)
    
    @property
    def cross_venue_correlation(self) -> float:
        """Correlation with reference venue (Binance)."""
        correlations = {
            self.CEX: 0.95,
            self.HYBRID: 0.90,
            self.DEX: 0.80,
            self.OPTIONS: 0.85,
            self.ON_CHAIN: 0.70,
        }
        return correlations.get(self, 0.85)


class DataQualityIssue(Enum):
    """
    Data quality issues for testing validation pipelines.
    """
    NONE = "none"
    MISSING_DATA = "missing_data"
    STALE_PRICE = "stale_price"
    OUTLIER = "outlier"
    DUPLICATE = "duplicate"
    NEGATIVE_VALUE = "negative_value"
    FUTURE_TIMESTAMP = "future_timestamp"
    NULL_VALUE = "null_value"
    WRONG_TYPE = "wrong_type"
    OUT_OF_RANGE = "out_of_range"
    WASH_TRADING = "wash_trading"
    
    @property
    def description(self) -> str:
        """Issue description."""
        descriptions = {
            self.NONE: "No issues",
            self.MISSING_DATA: "Missing records in time series",
            self.STALE_PRICE: "Price unchanged for extended period",
            self.OUTLIER: "Value outside expected range",
            self.DUPLICATE: "Duplicate timestamp/record",
            self.NEGATIVE_VALUE: "Negative value where positive expected",
            self.FUTURE_TIMESTAMP: "Timestamp in the future",
            self.NULL_VALUE: "Null/NaN value",
            self.WRONG_TYPE: "Incorrect data type",
            self.OUT_OF_RANGE: "Value outside valid range",
            self.WASH_TRADING: "Suspected wash trading pattern",
        }
        return descriptions.get(self, "Unknown issue")
    
    @property
    def severity(self) -> str:
        """Issue severity level."""
        severities = {
            self.NONE: "info",
            self.MISSING_DATA: "warning",
            self.STALE_PRICE: "warning",
            self.OUTLIER: "warning",
            self.DUPLICATE: "error",
            self.NEGATIVE_VALUE: "error",
            self.FUTURE_TIMESTAMP: "error",
            self.NULL_VALUE: "warning",
            self.WRONG_TYPE: "error",
            self.OUT_OF_RANGE: "warning",
            self.WASH_TRADING: "warning",
        }
        return severities.get(self, "warning")
    
    @property
    def auto_fixable(self) -> bool:
        """Whether issue can be auto-fixed."""
        fixable = {
            self.NONE: True,
            self.MISSING_DATA: False,
            self.STALE_PRICE: False,
            self.OUTLIER: True,
            self.DUPLICATE: True,
            self.NEGATIVE_VALUE: False,
            self.FUTURE_TIMESTAMP: True,
            self.NULL_VALUE: True,
            self.WRONG_TYPE: True,
            self.OUT_OF_RANGE: True,
            self.WASH_TRADING: False,
        }
        return fixable.get(self, False)


class AssetClass(Enum):
    """
    Asset class classification for realistic parameter selection.
    """
    MAJOR = "major"           # BTC, ETH
    LARGE_CAP = "large_cap"   # SOL, BNB, ADA, XRP
    MID_CAP = "mid_cap"       # LINK, UNI, AAVE
    SMALL_CAP = "small_cap"   # Lower liquidity alts
    MEME = "meme"             # DOGE, SHIB, PEPE
    DEFI = "defi"             # DeFi tokens
    L2 = "l2"                 # Layer 2 tokens
    
    @property
    def base_volatility(self) -> float:
        """Annualized base volatility."""
        vols = {
            self.MAJOR: 0.60,
            self.LARGE_CAP: 0.80,
            self.MID_CAP: 1.00,
            self.SMALL_CAP: 1.20,
            self.MEME: 1.50,
            self.DEFI: 1.00,
            self.L2: 0.90,
        }
        return vols.get(self, 1.0)
    
    @property
    def base_volume_usd(self) -> float:
        """Base 24h volume in USD."""
        volumes = {
            self.MAJOR: 20_000_000_000,
            self.LARGE_CAP: 1_000_000_000,
            self.MID_CAP: 200_000_000,
            self.SMALL_CAP: 50_000_000,
            self.MEME: 500_000_000,
            self.DEFI: 100_000_000,
            self.L2: 150_000_000,
        }
        return volumes.get(self, 100_000_000)
    
    @property
    def btc_correlation(self) -> float:
        """Correlation with BTC."""
        correlations = {
            self.MAJOR: 0.95,
            self.LARGE_CAP: 0.85,
            self.MID_CAP: 0.75,
            self.SMALL_CAP: 0.60,
            self.MEME: 0.50,
            self.DEFI: 0.70,
            self.L2: 0.75,
        }
        return correlations.get(self, 0.70)
    
    @property
    def funding_premium(self) -> float:
        """Funding rate premium over BTC."""
        premiums = {
            self.MAJOR: 0.0,
            self.LARGE_CAP: 0.00005,
            self.MID_CAP: 0.0001,
            self.SMALL_CAP: 0.00015,
            self.MEME: 0.0003,
            self.DEFI: 0.0001,
            self.L2: 0.0001,
        }
        return premiums.get(self, 0.0001)


class TimeframeType(Enum):
    """
    Timeframe classification for OHLCV generation.
    """
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    H8 = "8h"
    D1 = "1d"
    W1 = "1w"
    
    @property
    def minutes(self) -> int:
        """Timeframe in minutes."""
        minutes_map = {
            self.M1: 1,
            self.M5: 5,
            self.M15: 15,
            self.M30: 30,
            self.H1: 60,
            self.H4: 240,
            self.H8: 480,
            self.D1: 1440,
            self.W1: 10080,
        }
        return minutes_map.get(self, 60)
    
    @property
    def hours(self) -> float:
        """Timeframe in hours."""
        return self.minutes / 60
    
    @property
    def periods_per_day(self) -> float:
        """Number of periods per day."""
        return 1440 / self.minutes
    
    @property
    def vol_scaling_factor(self) -> float:
        """Volatility scaling factor (sqrt of time)."""
        return np.sqrt(self.hours / 24)


# =============================================================================
# ASSET PARAMETER DATABASE
# =============================================================================

ASSET_PARAMETERS: Dict[str, Dict[str, Any]] = {
    # Majors
    'BTC': {
        'class': AssetClass.MAJOR,
        'base_price': 40000,
        'daily_vol': 0.03,
        'base_volume': 20_000_000_000,
        'funding_sensitivity': 1.0,
        'oi_base': 15_000_000_000,
    },
    'ETH': {
        'class': AssetClass.MAJOR,
        'base_price': 2500,
        'daily_vol': 0.04,
        'base_volume': 10_000_000_000,
        'funding_sensitivity': 1.1,
        'oi_base': 8_000_000_000,
    },
    
    # Large Cap
    'SOL': {
        'class': AssetClass.LARGE_CAP,
        'base_price': 100,
        'daily_vol': 0.06,
        'base_volume': 2_000_000_000,
        'funding_sensitivity': 1.3,
        'oi_base': 1_500_000_000,
    },
    'BNB': {
        'class': AssetClass.LARGE_CAP,
        'base_price': 300,
        'daily_vol': 0.04,
        'base_volume': 800_000_000,
        'funding_sensitivity': 0.9,
        'oi_base': 500_000_000,
    },
    'XRP': {
        'class': AssetClass.LARGE_CAP,
        'base_price': 0.50,
        'daily_vol': 0.05,
        'base_volume': 1_500_000_000,
        'funding_sensitivity': 1.0,
        'oi_base': 400_000_000,
    },
    'ADA': {
        'class': AssetClass.LARGE_CAP,
        'base_price': 0.40,
        'daily_vol': 0.055,
        'base_volume': 400_000_000,
        'funding_sensitivity': 1.1,
        'oi_base': 200_000_000,
    },
    'AVAX': {
        'class': AssetClass.LARGE_CAP,
        'base_price': 30,
        'daily_vol': 0.06,
        'base_volume': 300_000_000,
        'funding_sensitivity': 1.2,
        'oi_base': 150_000_000,
    },
    'DOT': {
        'class': AssetClass.LARGE_CAP,
        'base_price': 7,
        'daily_vol': 0.055,
        'base_volume': 200_000_000,
        'funding_sensitivity': 1.0,
        'oi_base': 100_000_000,
    },
    
    # Mid Cap
    'LINK': {
        'class': AssetClass.MID_CAP,
        'base_price': 15,
        'daily_vol': 0.05,
        'base_volume': 300_000_000,
        'funding_sensitivity': 1.0,
        'oi_base': 150_000_000,
    },
    'UNI': {
        'class': AssetClass.DEFI,
        'base_price': 8,
        'daily_vol': 0.07,
        'base_volume': 150_000_000,
        'funding_sensitivity': 1.2,
        'oi_base': 80_000_000,
    },
    'AAVE': {
        'class': AssetClass.DEFI,
        'base_price': 100,
        'daily_vol': 0.06,
        'base_volume': 100_000_000,
        'funding_sensitivity': 1.1,
        'oi_base': 50_000_000,
    },
    'MKR': {
        'class': AssetClass.DEFI,
        'base_price': 1500,
        'daily_vol': 0.055,
        'base_volume': 80_000_000,
        'funding_sensitivity': 1.0,
        'oi_base': 40_000_000,
    },
    'CRV': {
        'class': AssetClass.DEFI,
        'base_price': 0.50,
        'daily_vol': 0.08,
        'base_volume': 100_000_000,
        'funding_sensitivity': 1.3,
        'oi_base': 50_000_000,
    },
    
    # Layer 2
    'ARB': {
        'class': AssetClass.L2,
        'base_price': 1.0,
        'daily_vol': 0.07,
        'base_volume': 300_000_000,
        'funding_sensitivity': 1.2,
        'oi_base': 200_000_000,
    },
    'OP': {
        'class': AssetClass.L2,
        'base_price': 2.0,
        'daily_vol': 0.07,
        'base_volume': 200_000_000,
        'funding_sensitivity': 1.2,
        'oi_base': 120_000_000,
    },
    'MATIC': {
        'class': AssetClass.L2,
        'base_price': 0.80,
        'daily_vol': 0.065,
        'base_volume': 300_000_000,
        'funding_sensitivity': 1.1,
        'oi_base': 150_000_000,
    },
    'IMX': {
        'class': AssetClass.L2,
        'base_price': 1.50,
        'daily_vol': 0.08,
        'base_volume': 80_000_000,
        'funding_sensitivity': 1.3,
        'oi_base': 40_000_000,
    },
    
    # Meme
    'DOGE': {
        'class': AssetClass.MEME,
        'base_price': 0.10,
        'daily_vol': 0.08,
        'base_volume': 500_000_000,
        'funding_sensitivity': 1.5,
        'oi_base': 300_000_000,
    },
    'SHIB': {
        'class': AssetClass.MEME,
        'base_price': 0.00001,
        'daily_vol': 0.10,
        'base_volume': 200_000_000,
        'funding_sensitivity': 1.8,
        'oi_base': 100_000_000,
    },
    'PEPE': {
        'class': AssetClass.MEME,
        'base_price': 0.000001,
        'daily_vol': 0.15,
        'base_volume': 300_000_000,
        'funding_sensitivity': 2.0,
        'oi_base': 150_000_000,
    },
    'WIF': {
        'class': AssetClass.MEME,
        'base_price': 2.0,
        'daily_vol': 0.12,
        'base_volume': 400_000_000,
        'funding_sensitivity': 1.8,
        'oi_base': 200_000_000,
    },
    'BONK': {
        'class': AssetClass.MEME,
        'base_price': 0.00002,
        'daily_vol': 0.12,
        'base_volume': 200_000_000,
        'funding_sensitivity': 1.7,
        'oi_base': 80_000_000,
    },
    
    # Small Cap / Other
    'LDO': {
        'class': AssetClass.DEFI,
        'base_price': 2.0,
        'daily_vol': 0.07,
        'base_volume': 100_000_000,
        'funding_sensitivity': 1.2,
        'oi_base': 50_000_000,
    },
    'GMX': {
        'class': AssetClass.DEFI,
        'base_price': 40,
        'daily_vol': 0.065,
        'base_volume': 50_000_000,
        'funding_sensitivity': 1.1,
        'oi_base': 30_000_000,
    },
    'DYDX': {
        'class': AssetClass.DEFI,
        'base_price': 2.0,
        'daily_vol': 0.08,
        'base_volume': 80_000_000,
        'funding_sensitivity': 1.2,
        'oi_base': 40_000_000,
    },
}

# Venue configurations
VENUE_CONFIGS: Dict[str, Dict[str, Any]] = {
    'binance': {
        'type': VenueType.CEX,
        'funding_interval': 8,
        'symbols': list(ASSET_PARAMETERS.keys()),
        'volume_multiplier': 1.0,
        'price_offset_bps': 0,
    },
    'bybit': {
        'type': VenueType.CEX,
        'funding_interval': 8,
        'symbols': list(ASSET_PARAMETERS.keys()),
        'volume_multiplier': 0.6,
        'price_offset_bps': 1,
    },
    'okx': {
        'type': VenueType.CEX,
        'funding_interval': 8,
        'symbols': list(ASSET_PARAMETERS.keys()),
        'volume_multiplier': 0.5,
        'price_offset_bps': 1,
    },
    'hyperliquid': {
        'type': VenueType.HYBRID,
        'funding_interval': 1,
        'symbols': ['BTC', 'ETH', 'SOL', 'ARB', 'OP', 'AVAX', 'LINK', 'DOGE', 'WIF'],
        'volume_multiplier': 0.15,
        'price_offset_bps': 2,
    },
    'dydx': {
        'type': VenueType.HYBRID,
        'funding_interval': 1,
        'symbols': ['BTC', 'ETH', 'SOL', 'AVAX', 'LINK', 'MATIC'],
        'volume_multiplier': 0.08,
        'price_offset_bps': 3,
    },
    'vertex': {
        'type': VenueType.HYBRID,
        'funding_interval': 1,
        'symbols': ['BTC', 'ETH', 'ARB', 'SOL'],
        'volume_multiplier': 0.03,
        'price_offset_bps': 5,
    },
    'uniswap_v3': {
        'type': VenueType.DEX,
        'funding_interval': 0,
        'symbols': ['ETH', 'UNI', 'LINK', 'AAVE', 'LDO', 'CRV', 'MKR'],
        'volume_multiplier': 0.10,
        'price_offset_bps': 10,
    },
    'curve': {
        'type': VenueType.DEX,
        'funding_interval': 0,
        'symbols': ['CRV', 'LDO'],
        'volume_multiplier': 0.05,
        'price_offset_bps': 5,
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MockDataConfig:
    """
    Comprehensive configuration for mock data generation.
    
    Includes all parameters needed for realistic synthetic data
    generation across multiple venues and asset types.
    """
    # Time range
    start_date: datetime = field(default_factory=lambda: datetime(2024, 1, 1))
    end_date: datetime = field(default_factory=lambda: datetime(2024, 3, 1))
    
    # Asset configuration
    symbols: List[str] = field(default_factory=lambda: ['BTC', 'ETH', 'SOL', 'ARB', 'OP'])
    venues: List[str] = field(default_factory=lambda: ['binance', 'bybit', 'hyperliquid'])
    
    # Reproducibility
    seed: int = 42
    
    # Regime configuration
    initial_regime: MarketRegime = MarketRegime.NEUTRAL
    enable_regime_switching: bool = True
    regime_switch_probability: float = 0.05
    
    # Quality injection
    inject_issues: bool = False
    issue_probability: float = 0.01
    issues_to_inject: List[DataQualityIssue] = field(
        default_factory=lambda: [DataQualityIssue.MISSING_DATA, DataQualityIssue.OUTLIER]
    )
    
    # Extended options
    enable_cross_venue_correlation: bool = True
    correlation_strength: float = 0.90
    enable_volatility_clustering: bool = True
    garch_persistence: float = 0.85
    enable_jumps: bool = False
    jump_intensity: float = 0.01
    jump_mean: float = 0.0
    jump_std: float = 0.05
    
    def __post_init__(self):
        """Initialize random state and validate config."""
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # Filter symbols to those with parameters
        self.symbols = [s for s in self.symbols if s in ASSET_PARAMETERS]
        
        # Filter venues to those configured
        self.venues = [v for v in self.venues if v in VENUE_CONFIGS]
    
    @property
    def n_days(self) -> int:
        """Number of days in date range."""
        return (self.end_date - self.start_date).days
    
    @property
    def date_range(self) -> pd.DatetimeIndex:
        """Pandas date range."""
        return pd.date_range(self.start_date, self.end_date, freq='D')
    
    def get_asset_params(self, symbol: str) -> Dict[str, Any]:
        """Get parameters for an asset."""
        return ASSET_PARAMETERS.get(symbol, {
            'class': AssetClass.SMALL_CAP,
            'base_price': 1.0,
            'daily_vol': 0.10,
            'base_volume': 10_000_000,
            'funding_sensitivity': 1.0,
            'oi_base': 5_000_000,
        })
    
    def get_venue_config(self, venue: str) -> Dict[str, Any]:
        """Get configuration for a venue."""
        return VENUE_CONFIGS.get(venue, {
            'type': VenueType.CEX,
            'funding_interval': 8,
            'symbols': self.symbols,
            'volume_multiplier': 0.1,
            'price_offset_bps': 5,
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'symbols': self.symbols,
            'venues': self.venues,
            'seed': self.seed,
            'n_days': self.n_days,
            'initial_regime': self.initial_regime.value,
            'inject_issues': self.inject_issues,
        }


@dataclass
class GeneratedDataStats:
    """
    Statistics about generated data for validation.
    """
    data_type: str
    n_records: int
    n_symbols: int
    n_venues: int
    date_range: Tuple[datetime, datetime]
    columns: List[str]
    
    # Quality metrics
    null_count: int = 0
    duplicate_count: int = 0
    injected_issues: Dict[str, int] = field(default_factory=dict)
    
    # Value statistics
    value_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Regime information
    regime_distribution: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'data_type': self.data_type,
            'n_records': self.n_records,
            'n_symbols': self.n_symbols,
            'n_venues': self.n_venues,
            'date_range': [d.isoformat() for d in self.date_range],
            'columns': self.columns,
            'null_count': self.null_count,
            'duplicate_count': self.duplicate_count,
            'injected_issues': self.injected_issues,
        }


# =============================================================================
# REGIME GENERATOR
# =============================================================================

class RegimeGenerator:
    """
    Generate market regime sequences with Markov transitions.
    
    Uses regime-specific transition probabilities to create
    realistic sequences of market conditions.
    """
    
    def __init__(self, config: MockDataConfig):
        """Initialize regime generator."""
        self.config = config
        self.current_regime = config.initial_regime
        self.days_in_regime = 0
        self.regime_history: List[Tuple[datetime, MarketRegime]] = []
    
    def generate_sequence(self, n_days: int) -> List[MarketRegime]:
        """Generate regime sequence for n days."""
        regimes = []
        
        for day in range(n_days):
            self.days_in_regime += 1
            
            # Check for regime transition
            if self.config.enable_regime_switching:
                if self.days_in_regime >= self.current_regime.min_duration_days:
                    if np.random.random() < self.config.regime_switch_probability:
                        self._transition()
            
            regimes.append(self.current_regime)
        
        return regimes
    
    def _transition(self):
        """Transition to new regime based on probabilities."""
        probs = self.current_regime.transition_probs
        regimes = list(probs.keys())
        probabilities = list(probs.values())
        
        # Normalize probabilities
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        
        # Sample new regime
        new_regime = np.random.choice(regimes, p=probabilities)
        
        if new_regime != self.current_regime:
            self.current_regime = new_regime
            self.days_in_regime = 0
    
    def get_regime_for_timestamp(
        self,
        timestamp: datetime,
        regime_sequence: List[MarketRegime]
    ) -> MarketRegime:
        """Get regime for a specific timestamp."""
        day_idx = (timestamp - self.config.start_date).days
        if 0 <= day_idx < len(regime_sequence):
            return regime_sequence[day_idx]
        return MarketRegime.NEUTRAL


# =============================================================================
# PRICE GENERATOR
# =============================================================================

class PriceGenerator:
    """
    Generate realistic price series with GARCH volatility and jumps.
    
    Implements:
    - Geometric Brownian Motion base
    - GARCH(1,1) volatility clustering
    - Optional jump-diffusion
    - Cross-asset correlation
    """
    
    def __init__(self, config: MockDataConfig):
        """Initialize price generator."""
        self.config = config
        self.cached_btc_returns: Optional[np.ndarray] = None
    
    def generate_returns(
        self,
        n_periods: int,
        daily_vol: float,
        timeframe: TimeframeType,
        regime_sequence: List[MarketRegime],
        asset_class: AssetClass
    ) -> np.ndarray:
        """
        Generate return series with GARCH and regime effects.
        
        Args:
            n_periods: Number of periods to generate
            daily_vol: Base daily volatility
            timeframe: Timeframe for scaling
            regime_sequence: Regime for each day
            asset_class: Asset class for correlation
        """
        # Scale volatility for timeframe
        period_vol = daily_vol * timeframe.vol_scaling_factor
        
        # Initialize volatility series (GARCH)
        vols = np.zeros(n_periods)
        vols[0] = period_vol
        
        # Generate innovations
        innovations = np.random.randn(n_periods)
        
        # Apply GARCH dynamics
        omega = period_vol ** 2 * (1 - self.config.garch_persistence)
        alpha = 0.05
        beta = self.config.garch_persistence - alpha
        
        returns = np.zeros(n_periods)
        
        for i in range(n_periods):
            # Get regime for this period
            day_idx = int(i / timeframe.periods_per_day)
            if day_idx < len(regime_sequence):
                regime = regime_sequence[day_idx]
            else:
                regime = MarketRegime.NEUTRAL
            
            # Adjust volatility for regime
            regime_vol = period_vol * regime.volatility_multiplier
            
            # GARCH volatility update
            if i > 0 and self.config.enable_volatility_clustering:
                vols[i] = np.sqrt(
                    omega +
                    alpha * returns[i-1] ** 2 +
                    beta * vols[i-1] ** 2
                )
            else:
                vols[i] = regime_vol
            
            # Generate return
            base_return = innovations[i] * vols[i]
            
            # Add drift from regime
            drift = regime.drift / 365 / timeframe.periods_per_day
            base_return += drift
            
            # Add jumps if enabled
            if self.config.enable_jumps:
                if np.random.random() < self.config.jump_intensity / timeframe.periods_per_day:
                    jump = np.random.normal(
                        self.config.jump_mean,
                        self.config.jump_std
                    )
                    base_return += jump
            
            returns[i] = base_return
        
        # Apply cross-asset correlation
        if self.config.enable_cross_venue_correlation:
            if self.cached_btc_returns is not None and len(self.cached_btc_returns) == n_periods:
                rho = asset_class.btc_correlation
                returns = rho * self.cached_btc_returns + np.sqrt(1 - rho ** 2) * returns
        
        return returns
    
    def generate_prices(
        self,
        symbol: str,
        timestamps: List[datetime],
        timeframe: TimeframeType,
        regime_sequence: List[MarketRegime]
    ) -> np.ndarray:
        """Generate price series from returns."""
        params = self.config.get_asset_params(symbol)
        base_price = params['base_price']
        daily_vol = params['daily_vol']
        asset_class = params['class']
        
        returns = self.generate_returns(
            len(timestamps),
            daily_vol,
            timeframe,
            regime_sequence,
            asset_class
        )
        
        # Cache BTC returns for correlation
        if symbol == 'BTC':
            self.cached_btc_returns = returns.copy()
        
        # Convert returns to prices
        prices = base_price * np.cumprod(1 + returns)
        
        return prices
    
    def generate_ohlc(
        self,
        close_prices: np.ndarray,
        daily_vol: float,
        timeframe: TimeframeType
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate OHLC from close prices."""
        n = len(close_prices)
        
        # Open is previous close (with small gap)
        opens = np.roll(close_prices, 1)
        opens[0] = close_prices[0] * (1 + np.random.normal(0, 0.001))
        
        # Generate intrabar volatility
        intrabar_vol = daily_vol * timeframe.vol_scaling_factor * 0.5
        intrabar_range = np.abs(np.random.randn(n)) * intrabar_vol * close_prices
        
        # High and low
        highs = np.maximum(opens, close_prices) + intrabar_range * np.random.uniform(0.3, 0.7, n)
        lows = np.minimum(opens, close_prices) - intrabar_range * np.random.uniform(0.3, 0.7, n)
        
        # Ensure valid OHLC relationship
        highs = np.maximum(highs, np.maximum(opens, close_prices))
        lows = np.minimum(lows, np.minimum(opens, close_prices))
        lows = np.maximum(lows, 0.001)  # Prevent negative prices
        
        return opens, highs, lows, close_prices


# =============================================================================
# FUNDING RATE GENERATOR
# =============================================================================

class MockFundingRateGenerator:
    """
    Generate realistic mock funding rate data.
    
    Implements:
    - Mean-reverting Ornstein-Uhlenbeck process
    - Regime-dependent parameters
    - Cross-venue correlation
    - Asset-specific premiums
    
    Mathematical Model:
    
        dF = κ(θ - F)dt + σdW
        
    Where:
        κ = mean reversion speed (~0.3)
        θ = regime-dependent long-term mean
        σ = funding rate volatility
        W = Wiener process
    """
    
    def __init__(self, config: MockDataConfig):
        """Initialize funding rate generator."""
        self.config = config
        self.regime_generator = RegimeGenerator(config)
        self.price_generator = PriceGenerator(config)
        
        # Mean reversion parameters
        self.kappa = 0.3  # Mean reversion speed
        
        # Cache for cross-venue correlation
        self._base_funding_cache: Dict[str, np.ndarray] = {}
    
    def _generate_timestamps(
        self,
        venue: str
    ) -> List[datetime]:
        """Generate funding timestamps for a venue."""
        venue_config = self.config.get_venue_config(venue)
        interval_hours = venue_config['funding_interval']
        
        if interval_hours == 0:
            return []  # No funding for spot/DEX
        
        timestamps = []
        current = self.config.start_date
        
        while current < self.config.end_date:
            timestamps.append(current)
            current += timedelta(hours=interval_hours)
        
        return timestamps
    
    def _generate_base_funding(
        self,
        symbol: str,
        timestamps: List[datetime],
        regime_sequence: List[MarketRegime]
    ) -> np.ndarray:
        """Generate base funding rate series using OU process."""
        n = len(timestamps)
        if n == 0:
            return np.array([])
        
        params = self.config.get_asset_params(symbol)
        funding_sensitivity = params['funding_sensitivity']
        asset_class = params['class']
        
        # Initialize
        funding = np.zeros(n)
        funding[0] = MarketRegime.NEUTRAL.funding_mean
        
        for i in range(1, n):
            # Get regime
            day_idx = (timestamps[i] - self.config.start_date).days
            if day_idx < len(regime_sequence):
                regime = regime_sequence[day_idx]
            else:
                regime = MarketRegime.NEUTRAL
            
            # Regime parameters
            theta = regime.funding_mean * funding_sensitivity + asset_class.funding_premium
            sigma = regime.funding_std * funding_sensitivity
            
            # OU dynamics (discrete approximation)
            dt = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600 / 8  # Normalized to 8h
            
            drift = self.kappa * (theta - funding[i-1]) * dt
            diffusion = sigma * np.sqrt(dt) * np.random.randn()
            
            funding[i] = funding[i-1] + drift + diffusion
        
        # Clip to realistic bounds
        funding = np.clip(funding, -0.01, 0.01)
        
        return funding
    
    def _apply_venue_effects(
        self,
        base_funding: np.ndarray,
        venue: str,
        symbol: str
    ) -> np.ndarray:
        """Apply venue-specific effects to funding rates."""
        if len(base_funding) == 0:
            return base_funding
        
        venue_config = self.config.get_venue_config(venue)
        venue_type = venue_config['type']
        
        # Add venue-specific noise
        noise_std = 0.00005 * (1 + venue_type.volume_noise_factor)
        venue_noise = np.random.normal(0, noise_std, len(base_funding))
        
        # Apply correlation with base venue
        if self.config.enable_cross_venue_correlation and venue != 'binance':
            if symbol in self._base_funding_cache:
                base = self._base_funding_cache[symbol]
                if len(base) == len(base_funding):
                    rho = venue_type.cross_venue_correlation
                    return rho * base + np.sqrt(1 - rho ** 2) * (base_funding + venue_noise)
        
        return base_funding + venue_noise
    
    def _inject_issues(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Inject data quality issues for testing."""
        if not self.config.inject_issues:
            return df
        
        df = df.copy()
        n = len(df)
        
        for issue in self.config.issues_to_inject:
            n_issues = int(n * self.config.issue_probability)
            
            if n_issues == 0:
                continue
            
            indices = np.random.choice(n, min(n_issues, n // 10), replace=False)
            
            if issue == DataQualityIssue.MISSING_DATA:
                # Drop rows
                df = df.drop(df.index[indices[:len(indices)//2]])
            
            elif issue == DataQualityIssue.OUTLIER:
                # Inject extreme values
                for idx in indices:
                    if idx < len(df):
                        df.iloc[idx, df.columns.get_loc('funding_rate')] = \
                            np.random.choice([-1, 1]) * np.random.uniform(0.05, 0.10)
            
            elif issue == DataQualityIssue.NULL_VALUE:
                # Inject nulls
                for idx in indices:
                    if idx < len(df):
                        df.iloc[idx, df.columns.get_loc('funding_rate')] = np.nan
            
            elif issue == DataQualityIssue.DUPLICATE:
                # Add duplicates
                dup_rows = df.iloc[indices[:min(5, len(indices))]].copy()
                df = pd.concat([df, dup_rows], ignore_index=True)
        
        return df.sort_values('timestamp').reset_index(drop=True)
    
    def generate(
        self,
        symbols: Optional[List[str]] = None,
        venues: Optional[List[str]] = None,
        include_open_interest: bool = True,
        include_mark_price: bool = True
    ) -> pd.DataFrame:
        """
        Generate mock funding rate DataFrame.
        
        Args:
            symbols: Symbols to generate (default: config symbols)
            venues: Venues to generate (default: config venues)
            include_open_interest: Include OI column
            include_mark_price: Include mark/index prices
            
        Returns:
            DataFrame with funding rate data
        """
        symbols = symbols or self.config.symbols
        venues = venues or self.config.venues
        
        # Generate regime sequence
        regime_sequence = self.regime_generator.generate_sequence(self.config.n_days)
        
        records = []
        
        for venue in venues:
            venue_config = self.config.get_venue_config(venue)
            venue_type = venue_config['type']
            
            # Skip venues without funding
            if venue_config['funding_interval'] == 0:
                continue
            
            # Filter symbols for this venue
            venue_symbols = [s for s in symbols if s in venue_config['symbols']]
            
            # Generate timestamps
            timestamps = self._generate_timestamps(venue)
            
            if len(timestamps) == 0:
                continue
            
            for symbol in venue_symbols:
                params = self.config.get_asset_params(symbol)
                
                # Generate base funding
                base_funding = self._generate_base_funding(symbol, timestamps, regime_sequence)
                
                # Cache for correlation (first venue only)
                if venue == 'binance' and symbol not in self._base_funding_cache:
                    self._base_funding_cache[symbol] = base_funding.copy()
                
                # Apply venue effects
                funding_rates = self._apply_venue_effects(base_funding, venue, symbol)
                
                # Generate prices
                prices = self.price_generator.generate_prices(
                    symbol, timestamps, TimeframeType.H8, regime_sequence
                )
                
                # Generate open interest
                if include_open_interest:
                    oi_base = params.get('oi_base', 100_000_000)
                    oi_noise = np.random.lognormal(0, 0.2, len(timestamps))
                    open_interest = oi_base * oi_noise * venue_config['volume_multiplier']
                
                for i, ts in enumerate(timestamps):
                    record = {
                        'timestamp': ts,
                        'symbol': symbol,
                        'funding_rate': funding_rates[i],
                        'venue': venue,
                        'venue_type': venue_type.value,
                    }
                    
                    if include_mark_price:
                        price_offset = venue_config['price_offset_bps'] / 10000
                        record['mark_price'] = prices[i] * (1 + price_offset)
                        record['index_price'] = prices[i]
                    
                    if include_open_interest:
                        record['open_interest'] = open_interest[i]
                    
                    records.append(record)
        
        df = pd.DataFrame(records)
        
        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.sort_values(['timestamp', 'symbol', 'venue']).reset_index(drop=True)
            df = self._inject_issues(df)
        
        return df
    
    def get_stats(self, df: pd.DataFrame) -> GeneratedDataStats:
        """Get statistics about generated data."""
        return GeneratedDataStats(
            data_type='funding_rates',
            n_records=len(df),
            n_symbols=df['symbol'].nunique() if len(df) > 0 else 0,
            n_venues=df['venue'].nunique() if len(df) > 0 else 0,
            date_range=(
                df['timestamp'].min().to_pydatetime() if len(df) > 0 else self.config.start_date,
                df['timestamp'].max().to_pydatetime() if len(df) > 0 else self.config.end_date
            ),
            columns=list(df.columns),
            null_count=int(df.isnull().sum().sum()),
            duplicate_count=int(df.duplicated().sum()),
            value_stats={
                'funding_rate': {
                    'mean': float(df['funding_rate'].mean()) if len(df) > 0 else 0,
                    'std': float(df['funding_rate'].std()) if len(df) > 0 else 0,
                    'min': float(df['funding_rate'].min()) if len(df) > 0 else 0,
                    'max': float(df['funding_rate'].max()) if len(df) > 0 else 0,
                }
            }
        )


# =============================================================================
# OHLCV GENERATOR
# =============================================================================

class MockOHLCVGenerator:
    """
    Generate realistic mock OHLCV data.
    
    Implements:
    - GARCH volatility clustering
    - U-shaped intraday volume pattern
    - Day-of-week effects
    - Regime-dependent behavior
    """
    
    def __init__(self, config: MockDataConfig):
        """Initialize OHLCV generator."""
        self.config = config
        self.regime_generator = RegimeGenerator(config)
        self.price_generator = PriceGenerator(config)
    
    def _generate_timestamps(
        self,
        timeframe: TimeframeType
    ) -> List[datetime]:
        """Generate timestamps for timeframe."""
        timestamps = []
        current = self.config.start_date
        delta = timedelta(minutes=timeframe.minutes)
        
        while current < self.config.end_date:
            timestamps.append(current)
            current += delta
        
        return timestamps
    
    def _generate_volume(
        self,
        timestamps: List[datetime],
        base_volume: float,
        regime_sequence: List[MarketRegime],
        venue_multiplier: float
    ) -> np.ndarray:
        """Generate realistic volume series."""
        n = len(timestamps)
        volumes = np.zeros(n)
        
        for i, ts in enumerate(timestamps):
            # U-shaped intraday pattern
            hour = ts.hour
            if hour < 4:
                hour_factor = 1.3 - 0.075 * hour
            elif hour < 12:
                hour_factor = 0.8 + 0.025 * (hour - 8) ** 2
            elif hour < 20:
                hour_factor = 0.9 + 0.02 * (hour - 12)
            else:
                hour_factor = 1.1 + 0.05 * (hour - 20)
            
            # Day of week effect
            dow = ts.weekday()
            dow_factors = {0: 1.1, 1: 1.0, 2: 1.0, 3: 1.05, 4: 1.15, 5: 0.7, 6: 0.65}
            dow_factor = dow_factors.get(dow, 1.0)
            
            # Regime effect
            day_idx = (ts - self.config.start_date).days
            if day_idx < len(regime_sequence):
                regime = regime_sequence[day_idx]
                regime_factor = regime.volume_multiplier
            else:
                regime_factor = 1.0
            
            # Random noise (log-normal for positive skew)
            noise = np.random.lognormal(0, 0.3)
            
            # Combine factors
            volumes[i] = (
                base_volume *
                hour_factor *
                dow_factor *
                regime_factor *
                venue_multiplier *
                noise /
                (24 * 60 / self.config.n_days)  # Scale to per-period
            )
        
        return volumes
    
    def generate(
        self,
        symbols: Optional[List[str]] = None,
        venues: Optional[List[str]] = None,
        timeframe: Union[str, TimeframeType] = TimeframeType.H1,
        include_vwap: bool = False,
        include_trades: bool = False
    ) -> pd.DataFrame:
        """
        Generate mock OHLCV DataFrame.
        
        Args:
            symbols: Symbols to generate
            venues: Venues to generate
            timeframe: Candle timeframe
            include_vwap: Include VWAP column
            include_trades: Include trade count
        """
        symbols = symbols or self.config.symbols
        venues = venues or self.config.venues
        
        # Parse timeframe
        if isinstance(timeframe, str):
            tf_map = {'1m': TimeframeType.M1, '5m': TimeframeType.M5,
                     '15m': TimeframeType.M15, '1h': TimeframeType.H1,
                     '4h': TimeframeType.H4, '1d': TimeframeType.D1}
            timeframe = tf_map.get(timeframe, TimeframeType.H1)
        
        # Generate regime sequence
        regime_sequence = self.regime_generator.generate_sequence(self.config.n_days)
        
        # Generate timestamps
        timestamps = self._generate_timestamps(timeframe)
        
        records = []
        
        for venue in venues:
            venue_config = self.config.get_venue_config(venue)
            venue_type = venue_config['type']
            
            # Filter symbols
            venue_symbols = [s for s in symbols if s in venue_config['symbols']]
            
            for symbol in venue_symbols:
                params = self.config.get_asset_params(symbol)
                
                # Generate prices
                close_prices = self.price_generator.generate_prices(
                    symbol, timestamps, timeframe, regime_sequence
                )
                
                # Generate OHLC
                opens, highs, lows, closes = self.price_generator.generate_ohlc(
                    close_prices, params['daily_vol'], timeframe
                )
                
                # Generate volume
                volumes = self._generate_volume(
                    timestamps,
                    params['base_volume'],
                    regime_sequence,
                    venue_config['volume_multiplier']
                )
                
                for i, ts in enumerate(timestamps):
                    record = {
                        'timestamp': ts,
                        'symbol': symbol,
                        'open': opens[i],
                        'high': highs[i],
                        'low': lows[i],
                        'close': closes[i],
                        'volume': volumes[i],
                        'volume_usd': volumes[i] * closes[i],
                        'venue': venue,
                        'venue_type': venue_type.value,
                        'timeframe': timeframe.value,
                    }
                    
                    if include_vwap:
                        record['vwap'] = (highs[i] + lows[i] + closes[i]) / 3
                    
                    if include_trades:
                        record['trades'] = int(volumes[i] / np.random.uniform(100, 1000))
                    
                    records.append(record)
        
        df = pd.DataFrame(records)
        
        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.sort_values(['timestamp', 'symbol', 'venue']).reset_index(drop=True)
        
        return df


# =============================================================================
# OPTIONS DATA GENERATOR
# =============================================================================

class MockOptionsDataGenerator:
    """
    Generate mock options chain data with realistic IV smile.
    
    Implements:
    - Black-Scholes pricing
    - Volatility smile/skew
    - Term structure
    - Greeks calculation
    """
    
    def __init__(self, config: MockDataConfig):
        """Initialize options generator."""
        self.config = config
    
    def _calculate_d1_d2(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> Tuple[float, float]:
        """Calculate Black-Scholes d1 and d2."""
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    def _norm_cdf(self, x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1 + math.erf(x / np.sqrt(2)))
    
    def _norm_pdf(self, x: float) -> float:
        """Standard normal PDF."""
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
    
    def _black_scholes_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str
    ) -> float:
        """Calculate Black-Scholes option price."""
        if T <= 0:
            if option_type == 'call':
                return max(0, S - K)
            return max(0, K - S)
        
        d1, d2 = self._calculate_d1_d2(S, K, T, r, sigma)
        
        if option_type == 'call':
            price = S * self._norm_cdf(d1) - K * np.exp(-r * T) * self._norm_cdf(d2)
        else:
            price = K * np.exp(-r * T) * self._norm_cdf(-d2) - S * self._norm_cdf(-d1)
        
        return max(0, price)
    
    def _calculate_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str
    ) -> Dict[str, float]:
        """Calculate option Greeks."""
        if T <= 0 or sigma <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        d1, d2 = self._calculate_d1_d2(S, K, T, r, sigma)
        sqrt_T = np.sqrt(T)
        
        # Delta
        if option_type == 'call':
            delta = self._norm_cdf(d1)
        else:
            delta = self._norm_cdf(d1) - 1
        
        # Gamma
        gamma = self._norm_pdf(d1) / (S * sigma * sqrt_T)
        
        # Theta (per day)
        theta_term1 = -S * self._norm_pdf(d1) * sigma / (2 * sqrt_T)
        if option_type == 'call':
            theta_term2 = -r * K * np.exp(-r * T) * self._norm_cdf(d2)
        else:
            theta_term2 = r * K * np.exp(-r * T) * self._norm_cdf(-d2)
        theta = (theta_term1 + theta_term2) / 365
        
        # Vega (per 1% change)
        vega = S * sqrt_T * self._norm_pdf(d1) / 100
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * self._norm_cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * self._norm_cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
        }
    
    def _generate_iv_smile(
        self,
        moneyness: float,
        T: float,
        option_type: str,
        base_iv: float = 0.60
    ) -> float:
        """Generate implied volatility with smile/skew."""
        # Base ATM IV
        atm_iv = base_iv * (1 + 0.1 * (1 / np.sqrt(T) - 1))  # Term structure
        
        # Smile: higher IV for OTM
        log_m = np.log(moneyness)
        smile_component = 0.15 * log_m ** 2
        
        # Skew: higher IV for OTM puts
        if option_type == 'put':
            skew_component = -0.08 * log_m
        else:
            skew_component = -0.04 * log_m
        
        iv = atm_iv + smile_component + skew_component
        
        # Add noise
        iv *= (1 + np.random.normal(0, 0.02))
        
        return max(0.1, min(3.0, iv))  # Clip to reasonable range
    
    def generate(
        self,
        underlying: str = 'BTC',
        spot_price: Optional[float] = None,
        n_expiries: int = 8,
        strikes_per_expiry: int = 20,
        include_greeks: bool = True
    ) -> pd.DataFrame:
        """
        Generate mock options chain.
        
        Args:
            underlying: Underlying symbol
            spot_price: Spot price (default from params)
            n_expiries: Number of expiry dates
            strikes_per_expiry: Strikes per expiry
            include_greeks: Include Greeks columns
        """
        params = self.config.get_asset_params(underlying)
        spot_price = spot_price or params['base_price']
        base_vol = params['daily_vol'] * np.sqrt(365)
        
        # Risk-free rate
        r = 0.05
        
        # Generate expiry dates
        base_date = self.config.start_date
        expiry_days = [7, 14, 30, 60, 90, 120, 180, 365][:n_expiries]
        expiries = [base_date + timedelta(days=d) for d in expiry_days]
        
        records = []
        
        for expiry in expiries:
            T = (expiry - base_date).days / 365
            
            # Generate strikes (more strikes near ATM)
            strike_range = np.concatenate([
                np.linspace(0.5, 0.9, strikes_per_expiry // 3),
                np.linspace(0.9, 1.1, strikes_per_expiry // 3),
                np.linspace(1.1, 1.5, strikes_per_expiry // 3),
            ])
            strikes = [round(spot_price * m, -2) for m in strike_range]
            strikes = sorted(set(strikes))
            
            for strike in strikes:
                moneyness = strike / spot_price
                
                for option_type in ['call', 'put']:
                    # Generate IV
                    iv = self._generate_iv_smile(moneyness, T, option_type, base_vol)
                    
                    # Calculate price
                    price = self._black_scholes_price(spot_price, strike, T, r, iv, option_type)
                    
                    # Bid-ask spread (wider for OTM)
                    spread = 0.02 * (1 + 0.5 * abs(np.log(moneyness)))
                    
                    record = {
                        'timestamp': base_date,
                        'underlying': underlying,
                        'strike': strike,
                        'expiry': expiry,
                        'expiry_days': (expiry - base_date).days,
                        'option_type': option_type,
                        'mark_price': price,
                        'mark_iv': iv,
                        'bid_price': price * (1 - spread / 2),
                        'ask_price': price * (1 + spread / 2),
                        'underlying_price': spot_price,
                        'moneyness': moneyness,
                        'volume': int(np.random.lognormal(5, 2)),
                        'open_interest': int(np.random.lognormal(7, 2)),
                        'venue': 'deribit',
                        'venue_type': 'OPTIONS',
                    }
                    
                    if include_greeks:
                        greeks = self._calculate_greeks(spot_price, strike, T, r, iv, option_type)
                        record.update(greeks)
                    
                    records.append(record)
        
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['expiry'] = pd.to_datetime(df['expiry'], utc=True)
        
        return df


# =============================================================================
# DEX POOL GENERATOR
# =============================================================================

class MockDEXPoolGenerator:
    """
    Generate mock DEX pool/liquidity data.
    
    Includes:
    - Pool TVL and volume
    - Fee tiers
    - Wash trading detection flags
    - Multi-chain support
    """
    
    # Common token pairs
    TOKEN_PAIRS = [
        ('WETH', 'USDC'), ('WETH', 'USDT'), ('WBTC', 'WETH'),
        ('ARB', 'WETH'), ('OP', 'WETH'), ('LINK', 'WETH'),
        ('UNI', 'WETH'), ('AAVE', 'WETH'), ('GMX', 'WETH'),
        ('LDO', 'WETH'), ('RPL', 'WETH'), ('CRV', 'WETH'),
        ('MKR', 'WETH'), ('SNX', 'WETH'), ('COMP', 'WETH'),
        ('SUSHI', 'WETH'), ('YFI', 'WETH'), ('1INCH', 'WETH'),
        ('WETH', 'DAI'), ('USDC', 'USDT'), ('WBTC', 'USDC'),
        ('ARB', 'USDC'), ('OP', 'USDC'), ('GMX', 'USDC'),
    ]
    
    # Fee tiers
    FEE_TIERS = [0.0001, 0.0005, 0.003, 0.01]
    
    # Chains
    CHAINS = ['ethereum', 'arbitrum', 'optimism', 'base', 'polygon']
    
    def __init__(self, config: MockDataConfig):
        """Initialize DEX pool generator."""
        self.config = config
    
    def generate_pools(
        self,
        n_pools: int = 100,
        chains: Optional[List[str]] = None,
        include_historical: bool = False
    ) -> pd.DataFrame:
        """
        Generate mock pool metadata.
        
        Args:
            n_pools: Number of pools to generate
            chains: Chains to include
            include_historical: Include historical TVL/volume
        """
        chains = chains or self.CHAINS
        records = []
        
        for i in range(n_pools):
            # Random token pair
            token0, token1 = random.choice(self.TOKEN_PAIRS)
            
            # Random chain and fee tier
            chain = random.choice(chains)
            fee_tier = random.choice(self.FEE_TIERS)
            
            # TVL (log-normal distribution)
            tvl = np.random.lognormal(15, 2)  # Median ~$3M
            
            # Volume/TVL ratio (higher = more suspicious)
            volume_tvl_ratio = np.random.lognormal(0, 1)
            volume = tvl * volume_tvl_ratio
            
            # Transaction count
            avg_trade_size = np.random.uniform(500, 5000)
            tx_count = int(volume / avg_trade_size)
            
            # Wash trading detection
            wash_score = 0.0
            if volume_tvl_ratio > 5:
                wash_score += 0.3
            if volume_tvl_ratio > 10:
                wash_score += 0.3
            if tx_count > 10000 and volume_tvl_ratio > 3:
                wash_score += 0.2
            wash_score += np.random.uniform(0, 0.2)
            
            wash_flag = wash_score > 0.5
            
            # Pool ID
            pool_id = '0x' + hashlib.md5(
                f"{token0}{token1}{chain}{fee_tier}{i}".encode()
            ).hexdigest()[:40]
            
            record = {
                'pool_id': pool_id,
                'token0_symbol': token0,
                'token1_symbol': token1,
                'fee_tier': fee_tier,
                'fee_tier_bps': fee_tier * 10000,
                'tvl_usd': tvl,
                'volume_24h_usd': volume,
                'volume_7d_usd': volume * 7 * np.random.uniform(0.8, 1.2),
                'tx_count_24h': tx_count,
                'volume_tvl_ratio': volume_tvl_ratio,
                'wash_trading_score': wash_score,
                'wash_trading_flag': wash_flag,
                'chain': chain,
                'venue': 'uniswap_v3' if chain != 'polygon' else 'quickswap',
                'venue_type': 'DEX',
                'created_at': self.config.start_date - timedelta(days=random.randint(1, 365)),
            }
            
            records.append(record)
        
        df = pd.DataFrame(records)
        df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
        
        return df.sort_values('tvl_usd', ascending=False).reset_index(drop=True)


# =============================================================================
# ON-CHAIN METRICS GENERATOR
# =============================================================================

class MockOnChainGenerator:
    """
    Generate mock on-chain metrics data.
    
    Includes:
    - TVL time series
    - Gas prices
    - Liquidation events
    - Whale transactions
    """
    
    def __init__(self, config: MockDataConfig):
        """Initialize on-chain generator."""
        self.config = config
        self.regime_generator = RegimeGenerator(config)
    
    def generate_tvl_timeseries(
        self,
        protocols: Optional[List[str]] = None,
        chains: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate TVL time series for protocols."""
        protocols = protocols or ['aave', 'compound', 'uniswap', 'curve', 'lido']
        chains = chains or ['ethereum', 'arbitrum', 'optimism']
        
        # Generate daily timestamps
        timestamps = pd.date_range(
            self.config.start_date,
            self.config.end_date,
            freq='D'
        )
        
        # Regime sequence
        regime_sequence = self.regime_generator.generate_sequence(len(timestamps))
        
        records = []
        
        for protocol in protocols:
            for chain in chains:
                # Base TVL (varies by protocol)
                base_tvl_map = {
                    'aave': 10_000_000_000,
                    'compound': 3_000_000_000,
                    'uniswap': 5_000_000_000,
                    'curve': 4_000_000_000,
                    'lido': 30_000_000_000,
                }
                base_tvl = base_tvl_map.get(protocol, 1_000_000_000)
                
                # Chain multiplier
                chain_mult = {'ethereum': 1.0, 'arbitrum': 0.1, 'optimism': 0.05}.get(chain, 0.05)
                
                # Generate TVL series
                tvl = base_tvl * chain_mult
                tvl_series = np.zeros(len(timestamps))
                tvl_series[0] = tvl
                
                for i in range(1, len(timestamps)):
                    regime = regime_sequence[i]
                    
                    # TVL changes with regime
                    if regime == MarketRegime.BULL:
                        drift = 0.002
                    elif regime == MarketRegime.BEAR:
                        drift = -0.001
                    elif regime == MarketRegime.CRISIS:
                        drift = -0.02
                    else:
                        drift = 0.0
                    
                    change = drift + np.random.normal(0, 0.01)
                    tvl_series[i] = tvl_series[i-1] * (1 + change)
                
                for i, ts in enumerate(timestamps):
                    records.append({
                        'timestamp': ts,
                        'protocol': protocol,
                        'chain': chain,
                        'tvl_usd': tvl_series[i],
                        'tvl_change_24h': (tvl_series[i] / tvl_series[max(0, i-1)] - 1) if i > 0 else 0,
                    })
        
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        return df
    
    def generate_liquidations(
        self,
        protocols: Optional[List[str]] = None,
        n_events: int = 1000
    ) -> pd.DataFrame:
        """Generate liquidation events."""
        protocols = protocols or ['aave', 'compound', 'maker']
        
        records = []
        
        for _ in range(n_events):
            # Random timestamp
            ts = self.config.start_date + timedelta(
                seconds=random.randint(0, int((self.config.end_date - self.config.start_date).total_seconds()))
            )
            
            # Liquidation size (log-normal)
            size_usd = np.random.lognormal(10, 2)  # Median ~$20k
            
            # Collateral and debt tokens
            collateral = random.choice(['ETH', 'WBTC', 'stETH', 'wstETH'])
            debt = random.choice(['USDC', 'USDT', 'DAI', 'FRAX'])
            
            records.append({
                'timestamp': ts,
                'protocol': random.choice(protocols),
                'collateral_token': collateral,
                'debt_token': debt,
                'collateral_amount_usd': size_usd,
                'debt_amount_usd': size_usd * 0.9,  # ~90% LTV
                'liquidation_bonus_pct': random.uniform(0.05, 0.15),
                'tx_hash': '0x' + hashlib.md5(str(ts).encode()).hexdigest(),
            })
        
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        return df.sort_values('timestamp').reset_index(drop=True)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_mock_dataset(
    data_type: str,
    config: Optional[MockDataConfig] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Factory function to create mock datasets.
    
    Args:
        data_type: One of 'funding', 'ohlcv', 'options', 'pools', 'tvl', 'liquidations'
        config: MockDataConfig instance
        **kwargs: Additional arguments for specific generators
        
    Returns:
        Generated DataFrame
    """
    config = config or MockDataConfig()
    
    generators = {
        'funding': MockFundingRateGenerator,
        'ohlcv': MockOHLCVGenerator,
        'options': MockOptionsDataGenerator,
        'pools': MockDEXPoolGenerator,
        'tvl': MockOnChainGenerator,
        'liquidations': MockOnChainGenerator,
    }
    
    if data_type not in generators:
        raise ValueError(
            f"Unknown data type: {data_type}. "
            f"Choose from: {list(generators.keys())}"
        )
    
    generator = generators[data_type](config)
    
    if data_type == 'funding':
        return generator.generate(**kwargs)
    elif data_type == 'ohlcv':
        return generator.generate(**kwargs)
    elif data_type == 'options':
        return generator.generate(**kwargs)
    elif data_type == 'pools':
        return generator.generate_pools(**kwargs)
    elif data_type == 'tvl':
        return generator.generate_tvl_timeseries(**kwargs)
    elif data_type == 'liquidations':
        return generator.generate_liquidations(**kwargs)
    
    return pd.DataFrame()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_funding_data(
    n_days: int = 30,
    symbols: Optional[List[str]] = None,
    venues: Optional[List[str]] = None,
    seed: int = 42
) -> pd.DataFrame:
    """Generate quick funding rate test data."""
    config = MockDataConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 1) + timedelta(days=n_days),
        symbols=symbols or ['BTC', 'ETH'],
        venues=venues or ['binance', 'hyperliquid'],
        seed=seed
    )
    return create_mock_dataset('funding', config)


def quick_ohlcv_data(
    n_days: int = 30,
    symbols: Optional[List[str]] = None,
    venues: Optional[List[str]] = None,
    timeframe: str = '1h',
    seed: int = 42
) -> pd.DataFrame:
    """Generate quick OHLCV test data."""
    config = MockDataConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 1) + timedelta(days=n_days),
        symbols=symbols or ['BTC', 'ETH'],
        venues=venues or ['binance'],
        seed=seed
    )
    return create_mock_dataset('ohlcv', config, timeframe=timeframe)


def quick_options_data(
    underlying: str = 'BTC',
    spot_price: Optional[float] = None,
    seed: int = 42
) -> pd.DataFrame:
    """Generate quick options chain test data."""
    config = MockDataConfig(seed=seed)
    return create_mock_dataset('options', config, underlying=underlying, spot_price=spot_price)


def quick_pool_data(
    n_pools: int = 50,
    chains: Optional[List[str]] = None,
    seed: int = 42
) -> pd.DataFrame:
    """Generate quick DEX pool test data."""
    config = MockDataConfig(seed=seed)
    return create_mock_dataset('pools', config, n_pools=n_pools, chains=chains)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'MarketRegime',
    'VenueType',
    'DataQualityIssue',
    'AssetClass',
    'TimeframeType',
    
    # Data classes
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