"""
Altcoin Statistical Arbitrage Strategy
======================================

Comprehensive cointegration-based pairs trading for cryptocurrency
markets across CEX, DEX, and hybrid venues.

Strategy Overview
-----------------
Similar to commodity spread trading (e.g., grain spreads), altcoins within
sectors often move together but occasionally diverge, creating mean-reversion
opportunities. Crypto exhibits higher volatility and shorter half-lives.

Mathematical Framework
----------------------
Spread Construction:

    Spread_t = Price_B_t - β × Price_A_t - α
    
    Where:
        β = hedge ratio (from OLS regression or Johansen)
        α = intercept (typically mean-adjusted out)

Z-Score Calculation:

    Z_t = (Spread_t - μ_spread) / σ_spread
    
    Where μ and σ are rolling statistics with lookback window

Entry/Exit Rules:

    LONG_SPREAD:  Entry when Z < -entry_threshold
                  Exit when Z > -exit_threshold
    
    SHORT_SPREAD: Entry when Z > +entry_threshold
                  Exit when Z < +exit_threshold

Half-Life Estimation (Ornstein-Uhlenbeck):

    dS = κ(μ - S)dt + σdW
    Discretized: ΔS_t = θ × S_{t-1} + ε_t
    Half-life = -ln(2) / θ

Cointegration Testing (Engle-Granger):

    Step 1: Y_t = α + β × X_t + ε_t (OLS)
    Step 2: Δε_t = γ × ε_{t-1} + Σ(δ_i × Δε_{t-i}) + u_t (ADF)
    H₀: γ = 0 (non-stationary), H₁: γ < 0 (cointegrated)

Position Sizing:

    Size = Base_Size × Signal_Strength × Vol_Adj × Tier_Mult × Venue_Mult
    
    Where:
        Signal_Strength = min(|Z| / entry_z, 1.0)
        Vol_Adj = Target_Vol / Realized_Vol
        Tier_Mult = based on pair quality tier
        Venue_Mult = based on venue characteristics

Hidden Markov Model (Regime Detection):

    P(O|λ) = Σ P(O|Q,λ) × P(Q|λ)
    
    States: LOW_VOL, MEDIUM_VOL, HIGH_VOL, TRENDING, CRISIS

Modules
-------
universe_construction : Build CEX + DEX trading universes
    - Multi-venue token filtering
    - Sector classification
    - Survivorship bias tracking
    - Pair candidate generation

cointegration : Engle-Granger and Johansen cointegration tests
    - Half-life estimation
    - Hurst exponent calculation
    - Rolling cointegration stability
    - Pair quality tiering

baseline_strategy : Z-score mean reversion strategy
    - Signal generation
    - Trade execution
    - Portfolio management
    - Transaction cost modeling

regime_detection : HMM-based market regime detection
    - Feature engineering
    - Regime classification
    - Regime-aware parameter adjustment
    - Transition tracking

ml_enhancement : Machine learning spread prediction
    - Walk-forward validation
    - Feature engineering
    - XGBoost/LightGBM models
    - Signal combination

position_sizing : Volatility-weighted position sizing
    - Kelly criterion
    - Risk parity
    - Venue-adjusted sizing
    - Portfolio constraints

Architecture
------------
    ┌─────────────────────────────────────────────────────────────┐
    │              Universe Construction                           │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
    │  │ CEX Universe│  │DEX Universe │  │ Hybrid Universe     │  │
    │  │  Tokens     │  │   Tokens    │  │    Tokens           │  │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
                              │
    ┌─────────────────────────────────────────────────────────────┐
    │                    Pair Selection                            │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
    │  │Cointegration│  │  Half-Life  │  │   Pair Quality      │  │
    │  │    Test     │  │ Estimation  │  │    Tiering          │  │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
                              │
    ┌─────────────────────────────────────────────────────────────┐
    │                   Signal Generation                          │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
    │  │   Z-Score   │  │   Regime    │  │  ML Enhancement     │  │
    │  │   Compute   │  │   Detect    │  │    (Optional)       │  │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
                              │
    ┌─────────────────────────────────────────────────────────────┐
    │                  Execution Layer                             │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
    │  │  Position   │  │    Cost     │  │   Portfolio         │  │
    │  │   Sizing    │  │   Model     │  │   Constraints       │  │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘

Key Features
------------
1. Dual-Universe Support:
    - CEX Universe: High liquidity, standard execution
    - DEX Universe: Long-tail tokens, unique opportunities
    - Mixed pairs: CEX-DEX cross-venue opportunities

2. Sector Classification:
    - L1/L2 blockchains (ETH, SOL, ARB, OP)
    - DeFi protocols (lending, DEX, perps)
    - Gaming/Metaverse (AXS, SAND, GALA)
    - Infrastructure (oracles, indexing)

3. Venue-Aware Execution:
    - CEX: Entry z < ±2.0, exit z = 0
    - DEX: Entry z < ±2.5, exit z < ±1.0 (account for gas)

4. Pair Tiering:
    - Tier 1: Both CEX, high liquidity (70% allocation)
    - Tier 2: Mixed CEX/DEX (25% allocation)
    - Tier 3: DEX-only, speculative (5% allocation)

5. Regime-Adaptive Trading:
    - LOW_VOL: Full positions, all tiers
    - MEDIUM_VOL: Standard parameters
    - HIGH_VOL: Reduced positions, Tier 1 only
    - CRISIS: Defensive mode, minimal exposure

Transaction Costs
-----------------
CEX (Binance/Bybit):
    - Maker: 0.02% / Taker: 0.05%
    - Round-trip (4 legs): ~0.20%
    - No gas costs

Hybrid (Hyperliquid):
    - Maker: -0.02% / Taker: 0.05%
    - Round-trip: ~0.12%
    - Minimal gas (~$0.50)

DEX (Arbitrum L2):
    - Swap fee: 0.30%
    - Slippage: 0.10-0.50%
    - Round-trip: ~1.00%
    - Gas: $1-8

DEX (Ethereum L1):
    - Swap fee: 0.30%
    - Slippage: 0.10-0.50%
    - Round-trip: ~1.50%
    - Gas: $40-200

Example Usage
-------------
>>> from strategies.pairs_trading import (
...     UniverseBuilder,
...     CointegrationAnalyzer,
...     BaselinePairsStrategy,
...     CryptoRegimeDetector,
...     PositionSizer,
... )
>>> 
>>> # Build universe
>>> builder = UniverseBuilder()
>>> builder.build_cex_universe(cex_data)
>>> builder.build_dex_universe(dex_data)
>>> universe = builder.combine_universes()
>>> pairs = builder.generate_pair_candidates()
>>> 
>>> # Test for cointegration
>>> analyzer = CointegrationAnalyzer()
>>> result = analyzer.engle_granger_test(token_a_prices, token_b_prices)
>>> 
>>> if result.is_cointegrated:
...     # Generate signals
...     strategy = BaselinePairsStrategy()
...     signals = strategy.generate_signals(
...         token_a_prices, token_b_prices,
...         result.hedge_ratio,
...         ('TOKEN_A', 'TOKEN_B'),
...         venue_type='CEX'
...     )
...     
...     # Size positions
...     sizer = PositionSizer(total_capital=1_000_000)
...     sizes = sizer.size_pair(pair_metrics, price_a, price_b)

Author: Crypto StatArb Quantitative Research
Version: 2.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CORE ENUMERATIONS
# =============================================================================

class Position(Enum):
    """
    Position state in a pairs trade.
    
    A pairs trade involves two legs:
    - LONG_SPREAD: Long token B, Short token A (expecting spread to increase)
    - SHORT_SPREAD: Short token B, Long token A (expecting spread to decrease)
    """
    FLAT = "flat"
    LONG_SPREAD = "long_spread"
    SHORT_SPREAD = "short_spread"
    
    @property
    def is_active(self) -> bool:
        """True if position is active."""
        return self != self.FLAT
    
    @property
    def direction(self) -> int:
        """Direction as integer: 1 for long spread, -1 for short, 0 for flat."""
        if self == self.LONG_SPREAD:
            return 1
        elif self == self.SHORT_SPREAD:
            return -1
        return 0
    
    @property
    def opposite(self) -> 'Position':
        """Get opposite position."""
        if self == self.LONG_SPREAD:
            return self.SHORT_SPREAD
        elif self == self.SHORT_SPREAD:
            return self.LONG_SPREAD
        return self.FLAT


class ExitReason(Enum):
    """
    Reason for exiting a pairs trade.

    Tracks why positions were closed for strategy analysis.
    """
    SIGNAL = "signal"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TIME_STOP = "time_stop"
    MAX_HOLD = "max_hold"              # Maximum holding period exceeded
    REGIME_CHANGE = "regime_change"
    REBALANCE = "rebalance"
    DELISTING = "delisting"
    LIQUIDITY = "liquidity"
    MANUAL = "manual"
    END_OF_DATA = "end_of_data"
    MEAN_REVERSION = "mean_reversion"
    
    @property
    def is_forced(self) -> bool:
        """True if exit was forced (not signal-based)."""
        return self in [
            self.STOP_LOSS, self.TIME_STOP, self.MAX_HOLD, self.REGIME_CHANGE,
            self.DELISTING, self.LIQUIDITY, self.END_OF_DATA
        ]
    
    @property
    def is_profitable_indicator(self) -> bool:
        """True if this exit type typically indicates profit."""
        return self in [self.SIGNAL, self.TAKE_PROFIT, self.MEAN_REVERSION]

    @property
    def priority(self) -> int:
        """Priority for exit (higher = more urgent)."""
        priorities = {
            self.DELISTING: 10,
            self.LIQUIDITY: 9,
            self.STOP_LOSS: 8,
            self.REGIME_CHANGE: 7,
            self.TIME_STOP: 6,
            self.TAKE_PROFIT: 5,
            self.MEAN_REVERSION: 4,
            self.SIGNAL: 4,
            self.REBALANCE: 3,
            self.MANUAL: 2,
            self.END_OF_DATA: 1,
        }
        return priorities.get(self, 0)


class VenueType(Enum):
    """
    Classification of trading venue types.
    
    Affects execution costs, position sizing, and risk parameters.
    """
    CEX = "CEX"
    HYBRID = "HYBRID"
    DEX = "DEX"
    
    @property
    def description(self) -> str:
        """Venue type description."""
        descriptions = {
            self.CEX: "Centralized Exchange",
            self.HYBRID: "Hybrid (Decentralized Orderbook)",
            self.DEX: "Decentralized Exchange (AMM)",
        }
        return descriptions.get(self, "Unknown")
    
    @property
    def position_multiplier(self) -> float:
        """Base position size multiplier."""
        multipliers = {
            self.CEX: 1.0,
            self.HYBRID: 0.7,
            self.DEX: 0.3,
        }
        return multipliers.get(self, 0.5)

    @property
    def capacity_multiplier(self) -> float:
        """Venue capacity multiplier (alias for position_multiplier)."""
        return self.position_multiplier

    @property
    def max_position_usd(self) -> float:
        """Maximum single position size."""
        limits = {
            self.CEX: 500_000,
            self.HYBRID: 200_000,
            self.DEX: 100_000,
        }
        return limits.get(self, 50_000)

    @property
    def min_position_usd(self) -> float:
        """Minimum position size to account for fixed costs."""
        minimums = {
            self.CEX: 1_000,
            self.HYBRID: 1_000,
            self.DEX: 5_000,  # Higher for DEX due to gas costs
        }
        return minimums.get(self, 1_000)

    @property
    def typical_round_trip_bps(self) -> float:
        """Typical round-trip cost in basis points."""
        costs = {
            self.CEX: 20.0,
            self.HYBRID: 12.0,
            self.DEX: 150.0,
        }
        return costs.get(self, 50.0)
    
    @property
    def recommended_entry_z(self) -> float:
        """Recommended z-score entry threshold."""
        thresholds = {
            self.CEX: 2.0,
            self.HYBRID: 2.2,
            self.DEX: 2.5,
        }
        return thresholds.get(self, 2.0)
    
    @property
    def recommended_exit_z(self) -> float:
        """Recommended z-score exit threshold."""
        thresholds = {
            self.CEX: 0.0,
            self.HYBRID: 0.25,
            self.DEX: 1.0,
        }
        return thresholds.get(self, 0.0)
    
    @property
    def has_funding_rate(self) -> bool:
        """True if venue has perpetual funding rates."""
        return self in [self.CEX, self.HYBRID]
    
    @property
    def has_gas_costs(self) -> bool:
        """True if venue has gas costs."""
        return self in [self.DEX, self.HYBRID]


class SignalStrength(Enum):
    """
    Signal strength classification for trade sizing.
    """
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"
    
    @classmethod
    def from_z_score(cls, z_score: float, entry_threshold: float = 2.0) -> 'SignalStrength':
        """Classify signal strength from z-score."""
        ratio = abs(z_score) / entry_threshold
        
        if ratio < 1.0:
            return cls.WEAK
        elif ratio < 1.25:
            return cls.MODERATE
        elif ratio < 1.5:
            return cls.STRONG
        return cls.VERY_STRONG
    
    @property
    def position_multiplier(self) -> float:
        """Position size multiplier."""
        multipliers = {
            self.WEAK: 0.5,
            self.MODERATE: 0.75,
            self.STRONG: 1.0,
            self.VERY_STRONG: 1.25,
        }
        return multipliers.get(self, 1.0)
    
    @property
    def min_z_ratio(self) -> float:
        """Minimum z-score ratio for this strength."""
        minimums = {
            self.WEAK: 0.0,
            self.MODERATE: 1.0,
            self.STRONG: 1.25,
            self.VERY_STRONG: 1.5,
        }
        return minimums.get(self, 1.0)


class PairTier(Enum):
    """
    Pair quality tier classification.
    
    Based on liquidity, cointegration stability, and venue availability.
    """
    TIER_1 = 1
    TIER_2 = 2
    TIER_3 = 3
    
    @property
    def description(self) -> str:
        """Tier description."""
        descriptions = {
            self.TIER_1: "High liquidity, CEX-CEX, stable cointegration",
            self.TIER_2: "Medium liquidity, mixed venues",
            self.TIER_3: "Lower liquidity, DEX-only, speculative",
        }
        return descriptions.get(self, "Unknown tier")
    
    @property
    def allocation_pct(self) -> float:
        """Target portfolio allocation."""
        allocations = {
            self.TIER_1: 0.70,
            self.TIER_2: 0.25,
            self.TIER_3: 0.05,
        }
        return allocations.get(self, 0.05)
    
    @property
    def position_multiplier(self) -> float:
        """Position size multiplier."""
        multipliers = {
            self.TIER_1: 1.0,
            self.TIER_2: 0.7,
            self.TIER_3: 0.4,
        }
        return multipliers.get(self, 0.4)
    
    @property
    def max_position_usd(self) -> float:
        """Maximum position size."""
        limits = {
            self.TIER_1: 500_000,
            self.TIER_2: 200_000,
            self.TIER_3: 50_000,
        }
        return limits.get(self, 50_000)
    
    @property
    def max_holding_days(self) -> int:
        """Maximum holding period."""
        days = {
            self.TIER_1: 30,
            self.TIER_2: 21,
            self.TIER_3: 14,
        }
        return days.get(self, 14)
    
    @property
    def stop_loss_z(self) -> float:
        """Default stop loss z-score."""
        stops = {
            self.TIER_1: 4.0,
            self.TIER_2: 3.5,
            self.TIER_3: 3.0,
        }
        return stops.get(self, 3.5)


class Sector(Enum):
    """
    Token sector classification (simplified for package-level use).
    
    For comprehensive sector classification, use TokenSector from
    universe_construction module.
    """
    L1 = "L1"
    L2 = "L2"
    DEFI = "DeFi"
    DEFI_LENDING = "DeFi_Lending"
    DEFI_DEX = "DeFi_DEX"
    MEME = "Meme"
    AI = "AI"
    GAMING = "Gaming"
    INFRA = "Infrastructure"
    ORACLE = "Oracle"
    LST = "Liquid_Staking"
    CEX_TOKEN = "CEX_Token"
    PRIVACY = "Privacy"
    RWA = "RWA"
    OTHER = "Other"
    
    @property
    def correlation_expectation(self) -> str:
        """Expected intra-sector correlation."""
        high = [self.L1, self.L2, self.DEFI_DEX, self.MEME, self.AI]
        return "high" if self in high else "medium"
    
    @property
    def volatility_profile(self) -> str:
        """Expected volatility profile."""
        very_high = [self.MEME, self.AI, self.GAMING]
        high = [self.L2, self.DEFI]
        
        if self in very_high:
            return "very_high"
        elif self in high:
            return "high"
        return "moderate"


class Chain(Enum):
    """
    Supported blockchain networks (simplified for package-level use).
    
    For comprehensive chain data, use Chain from universe_construction module.
    """
    ETHEREUM = "ethereum"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    BASE = "base"
    POLYGON = "polygon"
    SOLANA = "solana"
    AVALANCHE = "avalanche"
    BSC = "bsc"
    
    @property
    def is_l2(self) -> bool:
        """True if chain is an L2."""
        return self in [self.ARBITRUM, self.OPTIMISM, self.BASE]
    
    @property
    def typical_gas_usd(self) -> float:
        """Typical gas cost per transaction."""
        gas = {
            self.ETHEREUM: 25.0,
            self.ARBITRUM: 0.50,
            self.OPTIMISM: 0.30,
            self.BASE: 0.10,
            self.POLYGON: 0.05,
            self.SOLANA: 0.01,
            self.AVALANCHE: 0.50,
            self.BSC: 0.20,
        }
        return gas.get(self, 1.0)


# =============================================================================
# CONFIGURATION DATA CLASSES
# =============================================================================

@dataclass
class PairConfig:
    """
    Configuration for a specific trading pair.

    Contains all parameters needed to trade a pair including
    symbols, venue info, and trading thresholds.
    """
    symbol_a: str
    symbol_b: str
    venue_type: VenueType = VenueType.CEX
    tier: PairTier = PairTier.TIER_1
    sector: Sector = Sector.OTHER

    # Cointegration parameters
    hedge_ratio: float = 1.0
    intercept: float = 0.0  # Regression intercept
    half_life: float = 7.0
    spread_mean: float = 0.0
    spread_std: float = 1.0

    # Trading thresholds
    entry_z: float = 2.0
    exit_z: float = 0.0
    stop_z: float = 4.0

    # Position limits
    max_position_usd: float = 100_000
    max_holding_hours: int = 504  # 21 days

    # Costs
    estimated_round_trip_bps: float = 20.0

    # Aliases for compatibility
    @property
    def token_a(self) -> str:
        """Alias for symbol_a."""
        return self.symbol_a

    @property
    def token_b(self) -> str:
        """Alias for symbol_b."""
        return self.symbol_b

    @property
    def venue_enum(self) -> 'VenueType':
        """Alias for venue_type for strategy compatibility."""
        return self.venue_type

    @property
    def chain(self) -> str:
        """Default chain based on venue type."""
        if self.venue_type in [VenueType.DEX, VenueType.HYBRID]:
            return "ethereum"
        return "cex"

    @property
    def sector_enum(self) -> 'Sector':
        """Alias for sector for strategy compatibility."""
        return self.sector

    @property
    def tier_enum(self) -> 'PairTier':
        """Alias for tier for strategy compatibility."""
        return self.tier

    @property
    def pair_id(self) -> str:
        """Unique pair identifier."""
        tokens = sorted([self.symbol_a, self.symbol_b])
        return f"{tokens[0]}_{tokens[1]}"
    
    @property
    def pair_name(self) -> str:
        """Human-readable pair name."""
        return f"{self.symbol_a}/{self.symbol_b}"
    
    @property
    def position_multiplier(self) -> float:
        """Combined position multiplier."""
        return self.venue_type.position_multiplier * self.tier.position_multiplier
    
    @property
    def breakeven_half_life_hours(self) -> float:
        """Minimum half-life to be profitable."""
        return self.estimated_round_trip_bps / 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pair_id': self.pair_id,
            'symbol_a': self.symbol_a,
            'symbol_b': self.symbol_b,
            'venue_type': self.venue_type.value,
            'tier': self.tier.value,
            'hedge_ratio': self.hedge_ratio,
            'half_life': self.half_life,
            'entry_z': self.entry_z,
            'exit_z': self.exit_z,
            'stop_z': self.stop_z,
            'max_position_usd': self.max_position_usd,
            'estimated_round_trip_bps': self.estimated_round_trip_bps,
        }


@dataclass
class CostConfig:
    """
    Transaction cost configuration for a venue.
    
    Includes all cost components for accurate P&L calculation.
    """
    venue: str
    venue_type: VenueType = VenueType.CEX
    
    # Fee structure
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    
    # Execution costs
    slippage_bps: float = 2.0
    
    # On-chain costs (DEX)
    gas_cost_usd: float = 0.0
    mev_cost_bps: float = 0.0
    
    @property
    def one_way_cost_bps(self) -> float:
        """One-way cost in basis points (taker + slippage + MEV)."""
        return self.taker_fee_bps + self.slippage_bps + self.mev_cost_bps
    
    @property
    def round_trip_cost_bps(self) -> float:
        """Round-trip cost for pairs trade (4 legs)."""
        return self.one_way_cost_bps * 4
    
    @property
    def round_trip_gas_usd(self) -> float:
        """Round-trip gas cost (4 transactions)."""
        return self.gas_cost_usd * 4
    
    def calculate_cost(
        self,
        notional_usd: float,
        is_maker: bool = False
    ) -> float:
        """Calculate total cost for a trade."""
        fee_bps = self.maker_fee_bps if is_maker else self.taker_fee_bps
        bps_cost = (fee_bps + self.slippage_bps + self.mev_cost_bps) / 10000
        return notional_usd * bps_cost + self.gas_cost_usd
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'venue': self.venue,
            'venue_type': self.venue_type.value,
            'maker_fee_bps': self.maker_fee_bps,
            'taker_fee_bps': self.taker_fee_bps,
            'slippage_bps': self.slippage_bps,
            'gas_cost_usd': self.gas_cost_usd,
            'round_trip_cost_bps': round(self.round_trip_cost_bps, 1),
        }


@dataclass
class StrategyConfig:
    """
    Configuration for pairs trading strategy.
    """
    # Signal parameters
    z_lookback: int = 20
    entry_z: float = 2.0
    exit_z: float = 0.0
    stop_z: float = 4.0
    
    # Position sizing
    target_volatility: float = 0.15
    max_position_pct: float = 0.10
    max_sector_pct: float = 0.30
    
    # Holding limits
    max_holding_days: int = 21
    min_holding_hours: int = 4
    
    # Rebalancing
    rebalance_threshold: float = 0.20
    hedge_ratio_update_frequency: int = 5
    
    # Risk management
    max_drawdown_pct: float = 0.15
    max_correlation: float = 0.70
    
    # Regime awareness
    regime_aware: bool = True
    reduce_in_crisis: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'z_lookback': self.z_lookback,
            'entry_z': self.entry_z,
            'exit_z': self.exit_z,
            'stop_z': self.stop_z,
            'target_volatility': self.target_volatility,
            'max_position_pct': self.max_position_pct,
            'max_holding_days': self.max_holding_days,
            'regime_aware': self.regime_aware,
        }


@dataclass
class PortfolioConstraints:
    """
    Portfolio-level constraints for pairs trading.
    """
    # Capital allocation
    max_gross_exposure: float = 2.0
    max_net_exposure: float = 0.30
    min_cash_reserve: float = 0.20
    
    # Position limits (PDF: CEX 5-8, DEX 2-3, Total 8-10)
    max_positions: int = 10
    max_positions_per_sector: int = 3
    max_cex_positions: int = 8
    max_dex_positions: int = 3
    
    # Tier allocation
    max_tier1_allocation: float = 0.70
    max_tier2_allocation: float = 0.25
    max_tier3_allocation: float = 0.10
    
    # Venue allocation (PDF: 60% CEX, 20% T3/DEX)
    max_cex_allocation: float = 0.60
    max_hybrid_allocation: float = 0.30
    max_dex_allocation: float = 0.20
    
    # Correlation limits
    max_portfolio_correlation: float = 0.50
    max_pair_correlation: float = 0.70

    # Drawdown limits
    portfolio_stop_loss: float = 0.15
    daily_loss_limit: float = 0.05

    # Sector exposure (percentage of max_positions)
    max_sector_exposure: float = 0.25  # 25% of portfolio per sector

    @property
    def max_total_positions(self) -> int:
        """Alias for max_positions for compatibility."""
        return self.max_positions

    def get_max_positions(self, venue_type: 'VenueType') -> int:
        """Get max positions for a venue type."""
        if venue_type == VenueType.CEX:
            return self.max_cex_positions
        elif venue_type == VenueType.DEX:
            return self.max_dex_positions
        else:
            return self.max_positions  # Default for hybrid

    def check_position_allowed(
        self,
        current_positions: int,
        sector_positions: int,
        venue_type: VenueType
    ) -> Tuple[bool, str]:
        """Check if a new position is allowed."""
        if current_positions >= self.max_positions:
            return False, f"Max positions ({self.max_positions}) reached"
        
        if sector_positions >= self.max_positions_per_sector:
            return False, f"Max sector positions ({self.max_positions_per_sector}) reached"
        
        return True, "Position allowed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'max_gross_exposure': self.max_gross_exposure,
            'max_net_exposure': self.max_net_exposure,
            'max_positions': self.max_positions,
            'max_tier1_allocation': self.max_tier1_allocation,
            'max_tier2_allocation': self.max_tier2_allocation,
            'max_tier3_allocation': self.max_tier3_allocation,
            'portfolio_stop_loss': self.portfolio_stop_loss,
        }


# =============================================================================
# DEFAULT CONFIGURATIONS
# =============================================================================

DEFAULT_COST_CONFIGS: Dict[str, CostConfig] = {
    'binance': CostConfig(
        venue='binance',
        venue_type=VenueType.CEX,
        maker_fee_bps=2.0,
        taker_fee_bps=5.0,
        slippage_bps=2.0,
    ),
    'bybit': CostConfig(
        venue='bybit',
        venue_type=VenueType.CEX,
        maker_fee_bps=2.0,
        taker_fee_bps=5.5,
        slippage_bps=2.0,
    ),
    'okx': CostConfig(
        venue='okx',
        venue_type=VenueType.CEX,
        maker_fee_bps=2.0,
        taker_fee_bps=5.0,
        slippage_bps=2.0,
    ),
    'hyperliquid': CostConfig(
        venue='hyperliquid',
        venue_type=VenueType.HYBRID,
        maker_fee_bps=-2.0,
        taker_fee_bps=5.0,
        slippage_bps=2.0,
        gas_cost_usd=0.50,
    ),
    'dydx_v4': CostConfig(
        venue='dydx_v4',
        venue_type=VenueType.HYBRID,
        maker_fee_bps=-1.1,
        taker_fee_bps=5.0,
        slippage_bps=2.0,
        gas_cost_usd=0.10,
    ),
    'uniswap_v3': CostConfig(
        venue='uniswap_v3',
        venue_type=VenueType.DEX,
        maker_fee_bps=30.0,
        taker_fee_bps=30.0,
        slippage_bps=5.0,
        gas_cost_usd=25.0,
        mev_cost_bps=3.0,
    ),
    'uniswap_arbitrum': CostConfig(
        venue='uniswap_arbitrum',
        venue_type=VenueType.DEX,
        maker_fee_bps=30.0,
        taker_fee_bps=30.0,
        slippage_bps=5.0,
        gas_cost_usd=0.50,
        mev_cost_bps=1.0,
    ),
    'curve': CostConfig(
        venue='curve',
        venue_type=VenueType.DEX,
        maker_fee_bps=4.0,
        taker_fee_bps=4.0,
        slippage_bps=2.0,
        gas_cost_usd=30.0,
        mev_cost_bps=2.0,
    ),
}

DEFAULT_STRATEGY_CONFIG = StrategyConfig()
DEFAULT_PORTFOLIO_CONSTRAINTS = PortfolioConstraints()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Core Enums
    'Position',
    'ExitReason',
    'VenueType',
    'SignalStrength',
    'PairTier',
    'Sector',
    'Chain',
    
    # Configuration Classes
    'PairConfig',
    'CostConfig',
    'StrategyConfig',
    'PortfolioConstraints',
    
    # Default Configurations
    'DEFAULT_COST_CONFIGS',
    'DEFAULT_STRATEGY_CONFIG',
    'DEFAULT_PORTFOLIO_CONSTRAINTS',
    
    # From universe_construction
    'UniverseBuilder',
    'TokenInfo',
    'UniverseConfig',
    'PairCandidate',
    'TokenTier',
    'TokenSector',
    'FilterReason',
    'PairType',
    
    # From cointegration
    'CointegrationAnalyzer',
    'CointegrationResult',
    'PairRanking',
    'RollingCointegrationResult',
    'CointegrationMethod',
    'PairQuality',
    'StabilityStatus',
    'RejectionReason',
    
    # From baseline_strategy
    'BaselinePairsStrategy',
    'Trade',
    'BacktestMetrics',
    'TransactionCostModel',
    'PortfolioManager',
    
    # From regime_detection
    'CryptoRegimeDetector',
    'RegimeAwareStrategy',
    'RegimeState',
    'RegimeHistory',
    'RegimeFeatureEngineer',
    'MarketRegime',
    'RegimeTransition',
    'DetectorType',
    
    # From ml_enhancement
    'MLEnhancedStrategy',
    'FeatureEngineer',
    'WalkForwardValidator',
    'WalkForwardResult',
    'ValidationSummary',
    'ModelType',
    'PredictionTarget',
    'FeatureSet',
    'SignalConfidence',
    'EnsemblePredictor',
    'TradingSpecificLoss',
    'LSTMPredictor',
    'SharpeOptimizedModel',
    'FeatureConfig',
    'MLConfig',

    # From dynamic_pair_selection
    'DynamicPairSelector',
    'SelectionConfig',
    'SelectionAction',
    'TierLevel',
    'PairStatus',
    'RebalanceSummary',
    'RebalanceAction',

    # From position_sizing
    'PositionSizer',
    'PairMetrics',
    'PositionSize',
    'PortfolioState',
    'SizingMethod',
    'ConstraintType',
]


def __getattr__(name: str):
    """
    Lazy import for heavy modules.
    
    Imports are deferred until first use to reduce package load time.
    """
    # Universe Construction
    if name in ['UniverseBuilder', 'TokenInfo', 'UniverseConfig', 'PairCandidate',
                'TokenTier', 'TokenSector', 'FilterReason', 'PairType',
                'UniverseSnapshot', 'get_token_sector', 'is_stablecoin',
                'is_wrapped_token', 'is_leveraged_token', 'filter_token']:
        from .universe_construction import (
            UniverseBuilder, TokenInfo, UniverseConfig, PairCandidate,
            TokenTier, TokenSector, FilterReason, PairType, UniverseSnapshot,
            get_token_sector, is_stablecoin, is_wrapped_token,
            is_leveraged_token, filter_token
        )
        return locals()[name]
    
    # Cointegration
    if name in ['CointegrationAnalyzer', 'CointegrationResult', 'PairRanking',
                'RollingCointegrationResult', 'CointegrationMethod',
                'PairQuality', 'StabilityStatus', 'RejectionReason']:
        from .cointegration import (
            CointegrationAnalyzer, CointegrationResult, PairRanking,
            RollingCointegrationResult, CointegrationMethod,
            PairQuality, StabilityStatus, RejectionReason
        )
        return locals()[name]
    
    # Baseline Strategy
    if name in ['BaselinePairsStrategy', 'Trade', 'BacktestMetrics',
                'TransactionCostModel', 'PortfolioManager']:
        from .baseline_strategy import (
            BaselinePairsStrategy, Trade, BacktestMetrics,
            TransactionCostModel, PortfolioManager
        )
        return locals()[name]
    
    # Regime Detection
    if name in ['CryptoRegimeDetector', 'RegimeAwareStrategy', 'RegimeState',
                'RegimeHistory', 'RegimeFeatureEngineer', 'MarketRegime',
                'RegimeTransition', 'DetectorType', 'RegimeConfig',
                'FeatureCategory', 'detect_regime_simple']:
        from .regime_detection import (
            CryptoRegimeDetector, RegimeAwareStrategy, RegimeState,
            RegimeHistory, RegimeFeatureEngineer, MarketRegime,
            RegimeTransition, DetectorType, RegimeConfig,
            FeatureCategory, detect_regime_simple
        )
        return locals()[name]
    
    # ML Enhancement
    if name in ['MLEnhancedStrategy', 'FeatureEngineer', 'WalkForwardValidator',
                'WalkForwardResult', 'ValidationSummary', 'ModelType',
                'PredictionTarget', 'FeatureSet', 'SignalConfidence',
                'FeatureConfig', 'MLConfig', 'XGBoostModel', 'LightGBMModel',
                'create_model', 'EnsemblePredictor', 'TradingSpecificLoss',
                'LSTMPredictor', 'SharpeOptimizedModel']:
        from .ml_enhancement import (
            MLEnhancedStrategy, FeatureEngineer, WalkForwardValidator,
            WalkForwardResult, ValidationSummary, ModelType,
            PredictionTarget, FeatureSet, SignalConfidence,
            FeatureConfig, MLConfig, XGBoostModel, LightGBMModel,
            create_model, EnsemblePredictor, TradingSpecificLoss,
            LSTMPredictor, SharpeOptimizedModel
        )
        return locals()[name]

    # Dynamic Pair Selection
    if name in ['DynamicPairSelector', 'SelectionConfig', 'SelectionAction',
                'TierLevel', 'PairStatus', 'RebalanceSummary', 'RebalanceAction']:
        from .dynamic_pair_selection import (
            DynamicPairSelector, SelectionConfig, SelectionAction,
            TierLevel, PairStatus, RebalanceSummary, RebalanceAction
        )
        return locals()[name]

    # Position Sizing
    if name in ['PositionSizer', 'PairMetrics', 'PositionSize', 'PortfolioState',
                'SizingMethod', 'ConstraintType', 'VenueSizingConfig',
                'calculate_spread_volatility', 'create_pair_metrics']:
        from .position_sizing import (
            PositionSizer, PairMetrics, PositionSize, PortfolioState,
            SizingMethod, ConstraintType, VenueSizingConfig,
            calculate_spread_volatility, create_pair_metrics
        )
        return locals()[name]
    
    raise AttributeError(f"module 'pairs_trading' has no attribute '{name}'")


def __dir__():
    """List available names for tab-completion."""
    return __all__