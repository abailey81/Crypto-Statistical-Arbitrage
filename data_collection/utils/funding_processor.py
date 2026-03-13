"""
Funding Rate Processing and Cross-Venue Normalization Engine

validated funding rate alignment, normalization, spread analysis,
and signal generation for professional-quality statistical arbitrage.

===============================================================================
OVERVIEW
===============================================================================

This module solves the critical challenge of comparing funding rates across
venues with different payment intervals. Without proper normalization,
arbitrage signals are meaningless.

Key Capabilities:
    - Interval detection and normalization (hourly, 4h, 8h, daily)
    - Cross-venue spread calculation with transaction cost awareness
    - Carry trade analytics with Sharpe ratio
    - Funding regime classification
    - Anomaly detection (rate spikes, staleness, manipulation)
    - Position sizing based on funding economics

===============================================================================
FUNDING INTERVAL REFERENCE
===============================================================================

Critical: Venues use DIFFERENT intervals. Direct comparison is WRONG.

    ============== ============ ============== =================
    Venue Interval Payments/Day Annualization
    ============== ============ ============== =================
    Binance 8 hours 3 x 1095
    Bybit 8 hours 3 x 1095
    OKX 8 hours 3 x 1095
    Deribit 8 hours 3 x 1095
    CME Daily 1 x 365
    Hyperliquid 1 hour 24 x 8760
    dYdX V4 1 hour 24 x 8760
    GMX Hourly* ~24 x 8760
    Vertex 1 hour 24 x 8760
    ============== ============ ============== =================

    * GMX uses borrow fee model, converted to funding-equivalent

Normalization Formula:
    8h_equivalent = hourly_rate x 8
    daily_equivalent = 8h_rate x 3
    annualized = rate x periods_per_year

===============================================================================
SPREAD ECONOMICS
===============================================================================

Funding Spread = Venue_A_rate - Venue_B_rate

Profitable Arbitrage Requirements:
    1. Spread > Transaction Costs (maker/taker fees x 2)
    2. Spread > Funding Cost on Both Legs
    3. Spread persistence > Execution time
    4. Liquidity sufficient on both venues

Break-Even Calculation:
    min_spread_bps = (entry_fee + exit_fee) x 2 + slippage_estimate

Typical Thresholds:
    - CEX vs CEX: >5bp annualized spread
    - CEX vs Hybrid: >10bp annualized spread
    - Hybrid vs DEX: >15bp annualized spread

===============================================================================
CARRY TRADE ANALYTICS
===============================================================================

Carry Return = Position x Funding Rate x Holding Period
Risk = Position x Price Volatility x sqrt(Holding Period)
Sharpe = E[Carry Return] / Std[Carry Return] x sqrt(periods)

Key Metrics:
    - Expected carry (annualized)
    - Carry volatility
    - Funding Sharpe ratio
    - Maximum adverse carry
    - Funding regime persistence

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

1. Cross-Venue Funding Arbitrage:
   - Long on negative funding venue (receive funding)
   - Short on positive funding venue (pay less or receive)
   - Net: Capture spread while delta-neutral

2. Basis Trading:
   - Funding rate -> implied basis
   - Compare to futures basis
   - Trade convergence

3. Regime-Based Positioning:
   - Extreme positive funding -> contrarian short
   - Extreme negative funding -> contrarian long
   - Use as sentiment indicator

4. Carry Optimization:
   - Rank venues by carry
   - Adjust position sizing by funding volatility
   - Optimize holding period vs costs

Version: 3.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

STANDARD_FUNDING_TIMES_UTC = [0, 8, 16]

VENUE_CONFIGURATIONS = {
    'binance': {'interval': 8, 'type': 'cex', 'fees_bps': 4, 'slippage_bps': 1},
    'bybit': {'interval': 8, 'type': 'cex', 'fees_bps': 6, 'slippage_bps': 1},
    'okx': {'interval': 8, 'type': 'cex', 'fees_bps': 5, 'slippage_bps': 1},
    'deribit': {'interval': 8, 'type': 'cex', 'fees_bps': 5, 'slippage_bps': 2},
    'kraken': {'interval': 8, 'type': 'cex', 'fees_bps': 10, 'slippage_bps': 2},
    'coinbase': {'interval': 8, 'type': 'cex', 'fees_bps': 10, 'slippage_bps': 2},
    'hyperliquid': {'interval': 1, 'type': 'hybrid', 'fees_bps': 2.5, 'slippage_bps': 1},
    'dydx': {'interval': 1, 'type': 'hybrid', 'fees_bps': 5, 'slippage_bps': 2},
    'vertex': {'interval': 1, 'type': 'hybrid', 'fees_bps': 3, 'slippage_bps': 2},
    'aevo': {'interval': 1, 'type': 'hybrid', 'fees_bps': 5, 'slippage_bps': 3},
    'gmx': {'interval': 1, 'type': 'dex', 'fees_bps': 10, 'slippage_bps': 5},
    'synthetix': {'interval': 1, 'type': 'dex', 'fees_bps': 10, 'slippage_bps': 5},
}

# =============================================================================
# ENUMS
# =============================================================================

class FundingInterval(Enum):
    """
    Funding rate payment interval classification.

    Critical for normalization - venues pay at different intervals.
    """
    CONTINUOUS = 'continuous'
    HOURLY = 'hourly'
    FOUR_HOUR = '4h'
    EIGHT_HOUR = '8h'
    DAILY = 'daily'

    @property
    def hours(self) -> float:
        """Interval duration in hours."""
        return {
            FundingInterval.CONTINUOUS: 1/60,
            FundingInterval.HOURLY: 1.0,
            FundingInterval.FOUR_HOUR: 4.0,
            FundingInterval.EIGHT_HOUR: 8.0,
            FundingInterval.DAILY: 24.0,
        }.get(self, 8.0)

    @property
    def periods_per_day(self) -> float:
        """Number of funding periods per day."""
        return 24.0 / self.hours

    @property
    def periods_per_year(self) -> float:
        """Number of funding periods per year (for annualization)."""
        return 365.0 * self.periods_per_day

    @property
    def timedelta(self) -> timedelta:
        """Interval as timedelta."""
        return timedelta(hours=self.hours)

    @property
    def normalization_to_8h(self) -> float:
        """Multiplier to convert to 8-hour equivalent."""
        return 8.0 / self.hours

    @property
    def normalization_to_annual(self) -> float:
        """Multiplier to annualize."""
        return self.periods_per_year

    @classmethod
    def from_hours(cls, hours: float, tolerance: float = 0.2) -> 'FundingInterval':
        """Detect interval from hour duration."""
        if hours < 0.5:
            return cls.CONTINUOUS
        elif abs(hours - 1.0) <= tolerance:
            return cls.HOURLY
        elif abs(hours - 4.0) <= tolerance:
            return cls.FOUR_HOUR
        elif abs(hours - 8.0) <= tolerance:
            return cls.EIGHT_HOUR
        elif abs(hours - 24.0) <= tolerance:
            return cls.DAILY
        distances = {
            cls.HOURLY: abs(hours - 1.0),
            cls.FOUR_HOUR: abs(hours - 4.0),
            cls.EIGHT_HOUR: abs(hours - 8.0),
            cls.DAILY: abs(hours - 24.0),
        }
        return min(distances, key=distances.get)

    @classmethod
    def detect_from_timestamps(cls, timestamps: pd.Series) -> 'FundingInterval':
        """Detect interval from timestamp series."""
        if len(timestamps) < 2:
            return cls.EIGHT_HOUR
        ts = pd.to_datetime(timestamps).sort_values()
        median_diff = ts.diff().dropna().median()
        hours = median_diff.total_seconds() / 3600
        return cls.from_hours(hours)

class VenueType(Enum):
    """
    Venue classification for funding characteristics.
    """
    CEX_PERP = 'cex_perp'
    CEX_FUTURES = 'cex_futures'
    HYBRID_PERP = 'hybrid_perp'
    DEX_PERP = 'dex_perp'
    DEX_AMM = 'dex_amm'

    @property
    def typical_interval(self) -> FundingInterval:
        """Typical funding interval for this venue type."""
        return {
            VenueType.CEX_PERP: FundingInterval.EIGHT_HOUR,
            VenueType.CEX_FUTURES: FundingInterval.DAILY,
            VenueType.HYBRID_PERP: FundingInterval.HOURLY,
            VenueType.DEX_PERP: FundingInterval.HOURLY,
            VenueType.DEX_AMM: FundingInterval.CONTINUOUS,
        }.get(self, FundingInterval.EIGHT_HOUR)

    @property
    def typical_fee_bps(self) -> float:
        """Typical round-trip fee in basis points."""
        return {
            VenueType.CEX_PERP: 8,
            VenueType.CEX_FUTURES: 4,
            VenueType.HYBRID_PERP: 6,
            VenueType.DEX_PERP: 15,
            VenueType.DEX_AMM: 30,
        }.get(self, 10)

    @property
    def typical_slippage_bps(self) -> float:
        """Typical slippage in basis points (for $100K order)."""
        return {
            VenueType.CEX_PERP: 2,
            VenueType.CEX_FUTURES: 1,
            VenueType.HYBRID_PERP: 3,
            VenueType.DEX_PERP: 10,
            VenueType.DEX_AMM: 20,
        }.get(self, 5)

    @property
    def execution_cost_bps(self) -> float:
        """Total execution cost estimate (fees + slippage)."""
        return self.typical_fee_bps + self.typical_slippage_bps

    @property
    def counterparty_risk(self) -> str:
        """Counterparty risk classification."""
        return {
            VenueType.CEX_PERP: 'high',
            VenueType.CEX_FUTURES: 'medium',
            VenueType.HYBRID_PERP: 'low',
            VenueType.DEX_PERP: 'none',
            VenueType.DEX_AMM: 'none',
        }.get(self, 'unknown')

class FundingRegime(Enum):
    """
    Funding rate regime classification for market sentiment.
    """
    EXTREME_POSITIVE = 'extreme_positive'
    STRONGLY_POSITIVE = 'strongly_positive'
    MODERATELY_POSITIVE = 'moderately_positive'
    SLIGHTLY_POSITIVE = 'slightly_positive'
    NEUTRAL = 'neutral'
    SLIGHTLY_NEGATIVE = 'slightly_negative'
    MODERATELY_NEGATIVE = 'moderately_negative'
    STRONGLY_NEGATIVE = 'strongly_negative'
    EXTREME_NEGATIVE = 'extreme_negative'

    @property
    def annualized_threshold_pct(self) -> Tuple[float, float]:
        """(min, max) thresholds in percentage for this regime."""
        return {
            FundingRegime.EXTREME_POSITIVE: (100, float('inf')),
            FundingRegime.STRONGLY_POSITIVE: (50, 100),
            FundingRegime.MODERATELY_POSITIVE: (20, 50),
            FundingRegime.SLIGHTLY_POSITIVE: (5, 20),
            FundingRegime.NEUTRAL: (-5, 5),
            FundingRegime.SLIGHTLY_NEGATIVE: (-20, -5),
            FundingRegime.MODERATELY_NEGATIVE: (-50, -20),
            FundingRegime.STRONGLY_NEGATIVE: (-100, -50),
            FundingRegime.EXTREME_NEGATIVE: (float('-inf'), -100),
        }.get(self, (-5, 5))

    @classmethod
    def from_annualized_pct(cls, rate_pct: float) -> 'FundingRegime':
        """Classify from annualized rate percentage."""
        if rate_pct > 100:
            return cls.EXTREME_POSITIVE
        elif rate_pct > 50:
            return cls.STRONGLY_POSITIVE
        elif rate_pct > 20:
            return cls.MODERATELY_POSITIVE
        elif rate_pct > 5:
            return cls.SLIGHTLY_POSITIVE
        elif rate_pct >= -5:
            return cls.NEUTRAL
        elif rate_pct >= -20:
            return cls.SLIGHTLY_NEGATIVE
        elif rate_pct >= -50:
            return cls.MODERATELY_NEGATIVE
        elif rate_pct >= -100:
            return cls.STRONGLY_NEGATIVE
        return cls.EXTREME_NEGATIVE

    @property
    def is_positive(self) -> bool:
        """Funding favors shorts (longs pay)."""
        return 'positive' in self.value

    @property
    def is_negative(self) -> bool:
        """Funding favors longs (shorts pay)."""
        return 'negative' in self.value

    @property
    def is_extreme(self) -> bool:
        """Extreme regime (>100% or <-100% annualized)."""
        return 'extreme' in self.value

    @property
    def is_neutral(self) -> bool:
        """Near-zero funding."""
        return self == FundingRegime.NEUTRAL

    @property
    def market_sentiment(self) -> str:
        """Implied market sentiment."""
        if self.is_positive:
            return 'overleveraged_longs'
        elif self.is_negative:
            return 'overleveraged_shorts'
        return 'balanced'

    @property
    def contrarian_signal(self) -> str:
        """Contrarian trading signal."""
        if self.is_extreme:
            return 'strong_fade' if self.is_positive else 'strong_follow'
        elif self.is_positive:
            return 'mild_fade'
        elif self.is_negative:
            return 'mild_follow'
        return 'no_signal'

    @property
    def carry_trade_direction(self) -> str:
        """Preferred carry trade direction."""
        if self.is_positive:
            return 'short_perp_long_spot'
        elif self.is_negative:
            return 'long_perp_short_spot'
        return 'neutral'

    @property
    def risk_level(self) -> str:
        """Risk level for carry trades in this regime."""
        if self.is_extreme:
            return 'very_high'
        elif 'strongly' in self.value:
            return 'high'
        elif 'moderately' in self.value:
            return 'medium'
        return 'low'

class SpreadSignal(Enum):
    """
    Cross-venue spread signal classification.
    """
    STRONG_LONG_A = 'strong_long_a'
    LONG_A = 'long_a'
    WEAK_LONG_A = 'weak_long_a'
    NO_SIGNAL = 'no_signal'
    WEAK_LONG_B = 'weak_long_b'
    LONG_B = 'long_b'
    STRONG_LONG_B = 'strong_long_b'

    @classmethod
    def from_spread_bps(cls, spread_bps: float, breakeven_bps: float = 10) -> 'SpreadSignal':
        """
        Classify from spread in basis points.
        Positive spread (A > B) = short A, long B
        """
        net_spread = abs(spread_bps) - breakeven_bps
        if spread_bps > 0:
            if net_spread > 30:
                return cls.STRONG_LONG_B
            elif net_spread > 15:
                return cls.LONG_B
            elif net_spread > 0:
                return cls.WEAK_LONG_B
        elif spread_bps < 0:
            if net_spread > 30:
                return cls.STRONG_LONG_A
            elif net_spread > 15:
                return cls.LONG_A
            elif net_spread > 0:
                return cls.WEAK_LONG_A
        return cls.NO_SIGNAL

    @property
    def is_actionable(self) -> bool:
        """Check if signal is actionable."""
        return self != SpreadSignal.NO_SIGNAL

    @property
    def direction(self) -> str:
        """Direction of the signal."""
        if 'long_a' in self.value:
            return 'long_venue_a'
        elif 'long_b' in self.value:
            return 'long_venue_b'
        return 'none'

    @property
    def strength(self) -> float:
        """Signal strength (0-1)."""
        return {
            SpreadSignal.STRONG_LONG_A: 1.0,
            SpreadSignal.LONG_A: 0.7,
            SpreadSignal.WEAK_LONG_A: 0.4,
            SpreadSignal.NO_SIGNAL: 0.0,
            SpreadSignal.WEAK_LONG_B: 0.4,
            SpreadSignal.LONG_B: 0.7,
            SpreadSignal.STRONG_LONG_B: 1.0,
        }.get(self, 0.0)

    @property
    def position_size_multiplier(self) -> float:
        """Suggested position size multiplier based on signal strength."""
        return {
            SpreadSignal.STRONG_LONG_A: 1.0,
            SpreadSignal.LONG_A: 0.6,
            SpreadSignal.WEAK_LONG_A: 0.3,
            SpreadSignal.NO_SIGNAL: 0.0,
            SpreadSignal.WEAK_LONG_B: 0.3,
            SpreadSignal.LONG_B: 0.6,
            SpreadSignal.STRONG_LONG_B: 1.0,
        }.get(self, 0.0)

class AnomalyType(Enum):
    """
    Funding rate anomaly classification.
    """
    RATE_SPIKE = 'rate_spike'
    RATE_COLLAPSE = 'rate_collapse'
    STALE_DATA = 'stale_data'
    MISSING_PERIOD = 'missing_period'
    REGIME_SHIFT = 'regime_shift'
    CROSS_VENUE_DIVERGENCE = 'cross_venue_divergence'
    MANIPULATION_SUSPECTED = 'manipulation_suspected'

    @property
    def severity(self) -> str:
        """Anomaly severity."""
        return {
            AnomalyType.RATE_SPIKE: 'high',
            AnomalyType.RATE_COLLAPSE: 'high',
            AnomalyType.STALE_DATA: 'critical',
            AnomalyType.MISSING_PERIOD: 'medium',
            AnomalyType.REGIME_SHIFT: 'low',
            AnomalyType.CROSS_VENUE_DIVERGENCE: 'medium',
            AnomalyType.MANIPULATION_SUSPECTED: 'critical',
        }.get(self, 'unknown')

    @property
    def action_required(self) -> str:
        """Recommended action."""
        return {
            AnomalyType.RATE_SPIKE: 'reduce_position',
            AnomalyType.RATE_COLLAPSE: 'reduce_position',
            AnomalyType.STALE_DATA: 'pause_trading',
            AnomalyType.MISSING_PERIOD: 'investigate',
            AnomalyType.REGIME_SHIFT: 'reassess',
            AnomalyType.CROSS_VENUE_DIVERGENCE: 'investigate',
            AnomalyType.MANIPULATION_SUSPECTED: 'avoid',
        }.get(self, 'investigate')

    @property
    def affects_signal_quality(self) -> bool:
        """Check if anomaly affects signal quality."""
        return self in [
            AnomalyType.STALE_DATA,
            AnomalyType.MANIPULATION_SUSPECTED,
            AnomalyType.RATE_SPIKE,
        ]

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class FundingSnapshot:
    """
    Single funding rate observation with full analytics.

    Stores raw rate and provides normalized/annualized calculations.
    """
    timestamp: datetime
    symbol: str
    venue: str
    funding_rate: float
    interval: FundingInterval = FundingInterval.EIGHT_HOUR
    mark_price: Optional[float] = None
    index_price: Optional[float] = None
    open_interest: Optional[float] = None

    @property
    def funding_rate_pct(self) -> float:
        """Funding rate as percentage."""
        return self.funding_rate * 100

    @property
    def funding_rate_bps(self) -> float:
        """Funding rate in basis points."""
        return self.funding_rate * 10000

    @property
    def funding_rate_8h(self) -> float:
        """8-hour equivalent rate (decimal)."""
        return self.funding_rate * self.interval.normalization_to_8h

    @property
    def funding_rate_8h_pct(self) -> float:
        """8-hour equivalent rate (percentage)."""
        return self.funding_rate_8h * 100

    @property
    def funding_rate_8h_bps(self) -> float:
        """8-hour equivalent rate (basis points)."""
        return self.funding_rate_8h * 10000

    @property
    def funding_rate_daily(self) -> float:
        """Daily equivalent rate (decimal)."""
        return self.funding_rate * self.interval.normalization_to_8h * 3

    @property
    def funding_rate_daily_pct(self) -> float:
        """Daily equivalent rate (percentage)."""
        return self.funding_rate_daily * 100

    @property
    def funding_rate_annualized(self) -> float:
        """Annualized rate (decimal)."""
        return self.funding_rate * self.interval.normalization_to_annual

    @property
    def funding_rate_annualized_pct(self) -> float:
        """Annualized rate (percentage)."""
        return self.funding_rate_annualized * 100

    @property
    def regime(self) -> FundingRegime:
        """Funding regime classification."""
        return FundingRegime.from_annualized_pct(self.funding_rate_annualized_pct)

    @property
    def is_positive_funding(self) -> bool:
        """Longs pay shorts."""
        return self.funding_rate > 0

    @property
    def is_negative_funding(self) -> bool:
        """Shorts pay longs."""
        return self.funding_rate < 0

    @property
    def is_extreme(self) -> bool:
        """Extreme funding (>100% or <-100% annualized)."""
        return self.regime.is_extreme

    @property
    def basis_bps(self) -> Optional[float]:
        """Mark-index basis in basis points."""
        if self.mark_price and self.index_price and self.index_price > 0:
            return (self.mark_price - self.index_price) / self.index_price * 10000
        return None

    @property
    def is_contango(self) -> Optional[bool]:
        """Mark > index (contango/premium)."""
        if self.mark_price and self.index_price:
            return self.mark_price > self.index_price
        return None

    @property
    def daily_carry_cost_pct(self) -> float:
        """Daily carry cost as percentage (positive = cost to long)."""
        return self.funding_rate_daily_pct

    @property
    def annual_carry_cost_pct(self) -> float:
        """Annual carry cost as percentage."""
        return self.funding_rate_annualized_pct

    @property
    def carry_direction(self) -> str:
        """Direction to receive carry (short if positive funding)."""
        return 'short' if self.is_positive_funding else 'long'

    @property
    def is_arbitrage_candidate(self) -> bool:
        """Rate significant enough for arbitrage consideration (>10bp 8h)."""
        return abs(self.funding_rate_8h_bps) > 10

    @property
    def signal_strength(self) -> float:
        """Signal strength based on annualized rate (0-1 scale)."""
        abs_rate = abs(self.funding_rate_annualized_pct)
        if abs_rate >= 100:
            return 1.0
        elif abs_rate >= 50:
            return 0.8
        elif abs_rate >= 20:
            return 0.5
        elif abs_rate >= 5:
            return 0.3
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with all computed fields."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'symbol': self.symbol,
            'venue': self.venue,
            'interval': self.interval.value,
            'funding_rate': self.funding_rate,
            'funding_rate_pct': round(self.funding_rate_pct, 6),
            'funding_rate_bps': round(self.funding_rate_bps, 2),
            'funding_rate_8h': self.funding_rate_8h,
            'funding_rate_8h_bps': round(self.funding_rate_8h_bps, 2),
            'funding_rate_annualized_pct': round(self.funding_rate_annualized_pct, 2),
            'regime': self.regime.value,
            'is_positive': self.is_positive_funding,
            'is_extreme': self.is_extreme,
            'mark_price': self.mark_price,
            'index_price': self.index_price,
            'basis_bps': round(self.basis_bps, 2) if self.basis_bps else None,
            'is_contango': self.is_contango,
            'carry_direction': self.carry_direction,
            'daily_carry_cost_pct': round(self.daily_carry_cost_pct, 4),
            'is_arbitrage_candidate': self.is_arbitrage_candidate,
            'signal_strength': round(self.signal_strength, 2),
        }

@dataclass
class FundingSpread:
    """
    Cross-venue funding rate spread with arbitrage analytics.
    """
    timestamp: datetime
    symbol: str
    venue_a: str
    venue_b: str
    rate_a: FundingSnapshot
    rate_b: FundingSnapshot
    execution_cost_bps: float = 20.0

    @property
    def spread_raw(self) -> float:
        """Raw spread (A - B) in decimal."""
        return self.rate_a.funding_rate_8h - self.rate_b.funding_rate_8h

    @property
    def spread_bps(self) -> float:
        """Spread in basis points (8h normalized)."""
        return self.spread_raw * 10000

    @property
    def spread_annualized_pct(self) -> float:
        """Annualized spread in percentage."""
        return self.spread_raw * 1095 * 100

    @property
    def abs_spread_bps(self) -> float:
        """Absolute spread in basis points."""
        return abs(self.spread_bps)

    @property
    def abs_spread_annualized_pct(self) -> float:
        """Absolute annualized spread percentage."""
        return abs(self.spread_annualized_pct)

    @property
    def net_spread_bps(self) -> float:
        """Spread net of execution costs."""
        return self.abs_spread_bps - self.execution_cost_bps

    @property
    def is_profitable(self) -> bool:
        """Check if spread exceeds execution costs."""
        return self.net_spread_bps > 0

    @property
    def expected_pnl_bps_per_8h(self) -> float:
        """Expected P&L per 8-hour period in bps."""
        if self.is_profitable:
            return self.net_spread_bps
        return 0.0

    @property
    def expected_pnl_daily_bps(self) -> float:
        """Expected daily P&L in bps (assumes 3 funding periods)."""
        return self.expected_pnl_bps_per_8h * 3

    @property
    def expected_pnl_annual_pct(self) -> float:
        """Expected annual P&L percentage."""
        return self.expected_pnl_daily_bps * 365 / 100

    @property
    def breakeven_periods(self) -> float:
        """Periods to breakeven on execution costs."""
        if self.abs_spread_bps > 0:
            return self.execution_cost_bps / self.abs_spread_bps
        return float('inf')

    @property
    def breakeven_hours(self) -> float:
        """Hours to breakeven."""
        return self.breakeven_periods * 8

    @property
    def signal(self) -> SpreadSignal:
        """Trading signal from spread."""
        return SpreadSignal.from_spread_bps(self.spread_bps, self.execution_cost_bps)

    @property
    def arbitrage_direction(self) -> str:
        """
        Direction of arbitrage trade.
        Positive spread (A > B): Short A, Long B
        Negative spread (B > A): Long A, Short B
        """
        if self.spread_bps > 0:
            return f"short_{self.venue_a}_long_{self.venue_b}"
        elif self.spread_bps < 0:
            return f"long_{self.venue_a}_short_{self.venue_b}"
        return "no_trade"

    @property
    def long_venue(self) -> str:
        """Venue to go long on (lower funding)."""
        return self.venue_b if self.spread_bps > 0 else self.venue_a

    @property
    def short_venue(self) -> str:
        """Venue to go short on (higher funding)."""
        return self.venue_a if self.spread_bps > 0 else self.venue_b

    @property
    def regime_alignment(self) -> str:
        """Check if both venues are in same regime."""
        if self.rate_a.regime == self.rate_b.regime:
            return 'aligned'
        elif self.rate_a.regime.is_positive == self.rate_b.regime.is_positive:
            return 'partially_aligned'
        return 'divergent'

    @property
    def execution_risk(self) -> str:
        """Execution risk assessment."""
        if self.breakeven_hours < 8:
            return 'low'
        elif self.breakeven_hours < 24:
            return 'medium'
        return 'high'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'symbol': self.symbol,
            'venue_a': self.venue_a,
            'venue_b': self.venue_b,
            'rate_a_8h_bps': round(self.rate_a.funding_rate_8h_bps, 2),
            'rate_b_8h_bps': round(self.rate_b.funding_rate_8h_bps, 2),
            'spread_bps': round(self.spread_bps, 2),
            'spread_annualized_pct': round(self.spread_annualized_pct, 2),
            'execution_cost_bps': self.execution_cost_bps,
            'net_spread_bps': round(self.net_spread_bps, 2),
            'is_profitable': self.is_profitable,
            'expected_pnl_annual_pct': round(self.expected_pnl_annual_pct, 2),
            'signal': self.signal.value,
            'signal_strength': self.signal.strength,
            'arbitrage_direction': self.arbitrage_direction,
            'long_venue': self.long_venue,
            'short_venue': self.short_venue,
            'breakeven_hours': round(self.breakeven_hours, 1),
            'execution_risk': self.execution_risk,
        }

@dataclass
class FundingStatistics:
    """
    Aggregated funding rate statistics for analysis.
    """
    symbol: str
    venue: str
    period_start: datetime
    period_end: datetime
    num_observations: int
    mean_rate: float
    median_rate: float
    std_rate: float
    min_rate: float
    max_rate: float
    positive_pct: float = 0.0
    negative_pct: float = 0.0
    extreme_pct: float = 0.0

    @property
    def mean_rate_8h_bps(self) -> float:
        """Mean rate in 8h bps."""
        return self.mean_rate * 10000

    @property
    def mean_annualized_pct(self) -> float:
        """Mean annualized rate percentage."""
        return self.mean_rate * 1095 * 100

    @property
    def std_annualized_pct(self) -> float:
        """Std annualized rate percentage."""
        return self.std_rate * 1095 * 100

    @property
    def volatility_daily_bps(self) -> float:
        """Daily volatility in bps."""
        return self.std_rate * 10000 * np.sqrt(3)

    @property
    def sharpe_ratio(self) -> float:
        """
        Funding Sharpe ratio (annualized).
        Measures risk-adjusted carry return.
        """
        if self.std_rate > 0:
            return (self.mean_rate / self.std_rate) * np.sqrt(1095)
        return 0.0

    @property
    def is_stable_regime(self) -> bool:
        """Check if funding regime is stable."""
        return max(self.positive_pct, self.negative_pct) > 0.7

    @property
    def dominant_direction(self) -> str:
        """Dominant funding direction."""
        if self.positive_pct > 0.6:
            return 'positive'
        elif self.negative_pct > 0.6:
            return 'negative'
        return 'mixed'

    @property
    def regime_stability(self) -> float:
        """Regime stability score (0-1)."""
        return max(self.positive_pct, self.negative_pct)

    @property
    def range_bps(self) -> float:
        """Range in 8h bps."""
        return (self.max_rate - self.min_rate) * 10000

    @property
    def carry_quality(self) -> str:
        """Quality assessment for carry trading."""
        if self.sharpe_ratio > 2 and self.is_stable_regime:
            return 'excellent'
        elif self.sharpe_ratio > 1 and self.regime_stability > 0.6:
            return 'good'
        elif self.sharpe_ratio > 0.5:
            return 'moderate'
        return 'poor'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'venue': self.venue,
            'period_start': self.period_start.isoformat() if isinstance(self.period_start, datetime) else str(self.period_start),
            'period_end': self.period_end.isoformat() if isinstance(self.period_end, datetime) else str(self.period_end),
            'num_observations': self.num_observations,
            'mean_rate_8h_bps': round(self.mean_rate_8h_bps, 2),
            'mean_annualized_pct': round(self.mean_annualized_pct, 2),
            'std_annualized_pct': round(self.std_annualized_pct, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'min_rate_8h_bps': round(self.min_rate * 10000, 2),
            'max_rate_8h_bps': round(self.max_rate * 10000, 2),
            'range_bps': round(self.range_bps, 2),
            'positive_pct': round(self.positive_pct * 100, 1),
            'negative_pct': round(self.negative_pct * 100, 1),
            'extreme_pct': round(self.extreme_pct * 100, 1),
            'is_stable_regime': self.is_stable_regime,
            'dominant_direction': self.dominant_direction,
            'carry_quality': self.carry_quality,
        }

@dataclass
class FundingAnomaly:
    """
    Detected funding rate anomaly.
    """
    timestamp: datetime
    symbol: str
    venue: str
    anomaly_type: AnomalyType
    severity: str
    current_value: float
    expected_value: Optional[float] = None
    deviation_sigma: Optional[float] = None
    description: str = ""

    @property
    def is_critical(self) -> bool:
        """Check if anomaly is critical."""
        return self.severity == 'critical'

    @property
    def action(self) -> str:
        """Recommended action."""
        return self.anomaly_type.action_required

    @property
    def deviation_pct(self) -> Optional[float]:
        """Deviation as percentage from expected."""
        if self.expected_value and self.expected_value != 0:
            return ((self.current_value - self.expected_value) / abs(self.expected_value)) * 100
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'symbol': self.symbol,
            'venue': self.venue,
            'anomaly_type': self.anomaly_type.value,
            'severity': self.severity,
            'current_value': self.current_value,
            'expected_value': self.expected_value,
            'deviation_sigma': round(self.deviation_sigma, 2) if self.deviation_sigma else None,
            'deviation_pct': round(self.deviation_pct, 2) if self.deviation_pct else None,
            'description': self.description,
            'action': self.action,
            'is_critical': self.is_critical,
        }

# =============================================================================
# FUNDING PROCESSOR CLASS
# =============================================================================

class FundingProcessor:
    """
    Cross-venue funding rate processing and analysis engine.

    Provides:
        - Interval normalization
        - Cross-venue spread analysis
        - Carry trade analytics
        - Anomaly detection
        - Signal generation

    Example:
        >>> processor = FundingProcessor()
        >>> normalized = processor.normalize_rates(df, target_interval='8h')
        >>> spread = processor.calculate_spread(df, 'binance', 'hyperliquid')
        >>> stats = processor.calculate_statistics(df)
    """

    def __init__(
        self,
        default_execution_cost_bps: float = 20.0,
        anomaly_threshold_sigma: float = 3.0,
        min_spread_for_signal_bps: float = 10.0,
    ):
        """
        Initialize funding processor.

        Args:
            default_execution_cost_bps: Default round-trip execution cost
            anomaly_threshold_sigma: Sigma threshold for anomaly detection
            min_spread_for_signal_bps: Minimum spread for generating signals
        """
        self.default_execution_cost_bps = default_execution_cost_bps
        self.anomaly_threshold_sigma = anomaly_threshold_sigma
        self.min_spread_for_signal_bps = min_spread_for_signal_bps

    def detect_interval(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> FundingInterval:
        """Detect funding interval from DataFrame."""
        if df.empty or timestamp_col not in df.columns:
            return FundingInterval.EIGHT_HOUR
        return FundingInterval.detect_from_timestamps(df[timestamp_col])

    def normalize_rates(
        self,
        df: pd.DataFrame,
        target_interval: str = '8h',
        rate_col: str = 'funding_rate',
        timestamp_col: str = 'timestamp',
        venue_col: str = 'venue',
    ) -> pd.DataFrame:
        """
        Normalize funding rates to target interval.

        Args:
            df: DataFrame with funding rates
            target_interval: Target interval ('8h', 'hourly', 'daily', 'annual')
            rate_col: Column containing funding rates
            timestamp_col: Timestamp column
            venue_col: Venue column

        Returns:
            DataFrame with normalized rates added
        """
        if df.empty:
            return df

        df = df.copy()
        target = target_interval.lower()

        if venue_col in df.columns:
            for venue in df[venue_col].unique():
                mask = df[venue_col] == venue
                source_interval = self.detect_interval(df[mask], timestamp_col)

                if target == '8h':
                    multiplier = source_interval.normalization_to_8h
                elif target == 'hourly':
                    multiplier = 1.0 / source_interval.normalization_to_8h * 8
                elif target == 'daily':
                    multiplier = source_interval.normalization_to_8h * 3
                elif target == 'annual':
                    multiplier = source_interval.normalization_to_annual
                else:
                    multiplier = 1.0

                df.loc[mask, f'{rate_col}_normalized'] = df.loc[mask, rate_col] * multiplier
                df.loc[mask, 'source_interval'] = source_interval.value
        else:
            source_interval = self.detect_interval(df, timestamp_col)
            if target == '8h':
                multiplier = source_interval.normalization_to_8h
            elif target == 'annual':
                multiplier = source_interval.normalization_to_annual
            else:
                multiplier = 1.0

            df[f'{rate_col}_normalized'] = df[rate_col] * multiplier
            df['source_interval'] = source_interval.value

        df['target_interval'] = target

        return df

    def calculate_spread(
        self,
        df: pd.DataFrame,
        venue_a: str,
        venue_b: str,
        symbol: str = 'BTCUSDT',
        rate_col: str = 'funding_rate',
        timestamp_col: str = 'timestamp',
        venue_col: str = 'venue',
        symbol_col: str = 'symbol',
    ) -> pd.DataFrame:
        """
        Calculate funding spread between two venues.

        Args:
            df: DataFrame with funding rates
            venue_a: First venue name
            venue_b: Second venue name
            symbol: Trading pair symbol

        Returns:
            DataFrame with spread calculations
        """
        df_a = df[(df[venue_col] == venue_a) & (df[symbol_col] == symbol)].copy()
        df_b = df[(df[venue_col] == venue_b) & (df[symbol_col] == symbol)].copy()

        if df_a.empty or df_b.empty:
            logger.warning(f"Insufficient data for {venue_a} or {venue_b}")
            return pd.DataFrame()

        df_a = self.normalize_rates(df_a, '8h', rate_col, timestamp_col, venue_col)
        df_b = self.normalize_rates(df_b, '8h', rate_col, timestamp_col, venue_col)

        merged = pd.merge(
            df_a[[timestamp_col, f'{rate_col}_normalized']].rename(columns={f'{rate_col}_normalized': 'rate_a'}),
            df_b[[timestamp_col, f'{rate_col}_normalized']].rename(columns={f'{rate_col}_normalized': 'rate_b'}),
            on=timestamp_col,
            how='inner'
        )

        if merged.empty:
            return pd.DataFrame()

        merged['spread_raw'] = merged['rate_a'] - merged['rate_b']
        merged['spread_bps'] = merged['spread_raw'] * 10000
        merged['spread_annualized_pct'] = merged['spread_raw'] * 1095 * 100
        merged['abs_spread_bps'] = merged['spread_bps'].abs()

        cost = self._get_execution_cost(venue_a, venue_b)
        merged['execution_cost_bps'] = cost
        merged['net_spread_bps'] = merged['abs_spread_bps'] - cost
        merged['is_profitable'] = merged['net_spread_bps'] > 0

        merged['long_venue'] = merged['spread_bps'].apply(
            lambda x: venue_b if x > 0 else venue_a
        )
        merged['short_venue'] = merged['spread_bps'].apply(
            lambda x: venue_a if x > 0 else venue_b
        )

        merged['signal'] = merged['spread_bps'].apply(
            lambda x: SpreadSignal.from_spread_bps(x, cost).value
        )
        merged['signal_strength'] = merged['spread_bps'].apply(
            lambda x: SpreadSignal.from_spread_bps(x, cost).strength
        )

        merged['venue_a'] = venue_a
        merged['venue_b'] = venue_b
        merged['symbol'] = symbol

        return merged

    def _get_execution_cost(self, venue_a: str, venue_b: str) -> float:
        """Get combined execution cost for two venues."""
        config_a = VENUE_CONFIGURATIONS.get(venue_a.lower(), {})
        config_b = VENUE_CONFIGURATIONS.get(venue_b.lower(), {})

        fees_a = config_a.get('fees_bps', 10)
        fees_b = config_b.get('fees_bps', 10)
        slippage_a = config_a.get('slippage_bps', 2)
        slippage_b = config_b.get('slippage_bps', 2)

        return (fees_a + slippage_a + fees_b + slippage_b)

    def calculate_statistics(
        self,
        df: pd.DataFrame,
        rate_col: str = 'funding_rate',
        timestamp_col: str = 'timestamp',
        venue_col: str = 'venue',
        symbol_col: str = 'symbol',
    ) -> List[FundingStatistics]:
        """
        Calculate funding statistics per venue/symbol.

        Returns:
            List of FundingStatistics objects
        """
        results = []

        if df.empty:
            return results

        df = self.normalize_rates(df, '8h', rate_col, timestamp_col, venue_col)
        rate_norm = f'{rate_col}_normalized'

        for (venue, symbol), group in df.groupby([venue_col, symbol_col]):
            if len(group) < 2:
                continue

            rates = group[rate_norm].dropna()

            stats = FundingStatistics(
                symbol=symbol,
                venue=venue,
                period_start=group[timestamp_col].min(),
                period_end=group[timestamp_col].max(),
                num_observations=len(rates),
                mean_rate=float(rates.mean()),
                median_rate=float(rates.median()),
                std_rate=float(rates.std()),
                min_rate=float(rates.min()),
                max_rate=float(rates.max()),
                positive_pct=float((rates > 0).mean()),
                negative_pct=float((rates < 0).mean()),
                extreme_pct=float((rates.abs() > 0.001).mean()),
            )
            results.append(stats)

        return results

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        rate_col: str = 'funding_rate',
        timestamp_col: str = 'timestamp',
        venue_col: str = 'venue',
        symbol_col: str = 'symbol',
    ) -> List[FundingAnomaly]:
        """
        Detect anomalies in funding rate data.

        Returns:
            List of FundingAnomaly objects
        """
        anomalies = []

        if df.empty:
            return anomalies

        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values([venue_col, symbol_col, timestamp_col])

        for (venue, symbol), group in df.groupby([venue_col, symbol_col]):
            rates = group[rate_col].values
            timestamps = group[timestamp_col].values

            if len(rates) < 10:
                continue

            mean_rate = np.mean(rates)
            std_rate = np.std(rates)

            if std_rate > 0:
                z_scores = (rates - mean_rate) / std_rate

                for i, (z, ts, rate) in enumerate(zip(z_scores, timestamps, rates)):
                    if abs(z) > self.anomaly_threshold_sigma:
                        anomaly_type = AnomalyType.RATE_SPIKE if z > 0 else AnomalyType.RATE_COLLAPSE
                        anomalies.append(FundingAnomaly(
                            timestamp=pd.Timestamp(ts).to_pydatetime(),
                            symbol=symbol,
                            venue=venue,
                            anomaly_type=anomaly_type,
                            severity=anomaly_type.severity,
                            current_value=rate,
                            expected_value=mean_rate,
                            deviation_sigma=abs(z),
                            description=f"Rate {z:.1f} sigma from mean",
                        ))

            time_diffs = np.diff(timestamps.astype('datetime64[h]').astype(int))
            expected_diff = 8

            for i, diff in enumerate(time_diffs):
                if diff > expected_diff * 2:
                    anomalies.append(FundingAnomaly(
                        timestamp=pd.Timestamp(timestamps[i+1]).to_pydatetime(),
                        symbol=symbol,
                        venue=venue,
                        anomaly_type=AnomalyType.MISSING_PERIOD,
                        severity='medium',
                        current_value=float(diff),
                        expected_value=float(expected_diff),
                        description=f"Missing {int(diff - expected_diff)}h of data",
                    ))

        return anomalies

    def generate_signals(
        self,
        df: pd.DataFrame,
        rate_col: str = 'funding_rate',
        timestamp_col: str = 'timestamp',
        venue_col: str = 'venue',
        symbol_col: str = 'symbol',
    ) -> pd.DataFrame:
        """
        Generate trading signals from funding rates.

        Returns:
            DataFrame with signals
        """
        if df.empty:
            return pd.DataFrame()

        df = self.normalize_rates(df, '8h', rate_col, timestamp_col, venue_col)
        rate_norm = f'{rate_col}_normalized'

        signals = []

        latest = df.sort_values(timestamp_col).groupby([venue_col, symbol_col]).last().reset_index()

        for _, row in latest.iterrows():
            rate_8h = row[rate_norm]
            rate_annual_pct = rate_8h * 1095 * 100
            regime = FundingRegime.from_annualized_pct(rate_annual_pct)

            signal_dict = {
                'timestamp': row[timestamp_col],
                'symbol': row[symbol_col],
                'venue': row[venue_col],
                'funding_rate_8h_bps': rate_8h * 10000,
                'funding_rate_annual_pct': rate_annual_pct,
                'regime': regime.value,
                'carry_direction': regime.carry_trade_direction,
                'contrarian_signal': regime.contrarian_signal,
                'risk_level': regime.risk_level,
                'signal_strength': 1.0 if regime.is_extreme else 0.5 if 'strongly' in regime.value else 0.2,
            }
            signals.append(signal_dict)

        return pd.DataFrame(signals)

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def normalize_funding_rate(
    rate: float,
    source_interval: str,
    target_interval: str = '8h'
) -> float:
    """
    Normalize a single funding rate.

    Args:
        rate: Raw funding rate (decimal)
        source_interval: Source interval ('hourly', '4h', '8h', 'daily')
        target_interval: Target interval

    Returns:
        Normalized rate
    """
    interval_map = {
        'hourly': FundingInterval.HOURLY,
        '1h': FundingInterval.HOURLY,
        '4h': FundingInterval.FOUR_HOUR,
        '8h': FundingInterval.EIGHT_HOUR,
        'daily': FundingInterval.DAILY,
    }
    source = interval_map.get(source_interval.lower(), FundingInterval.EIGHT_HOUR)

    if target_interval.lower() == '8h':
        return rate * source.normalization_to_8h
    elif target_interval.lower() == 'annual':
        return rate * source.normalization_to_annual
    elif target_interval.lower() == 'daily':
        return rate * source.normalization_to_8h * 3
    return rate

def calculate_carry_pnl(
    position_usd: float,
    funding_rate_8h: float,
    holding_periods: int = 1,
) -> float:
    """
    Calculate carry P&L for a position.

    Args:
        position_usd: Position size in USD
        funding_rate_8h: 8-hour funding rate (decimal)
        holding_periods: Number of 8-hour periods

    Returns:
        Expected P&L in USD
    """
    return position_usd * funding_rate_8h * holding_periods

def annualize_funding_rate(rate: float, interval: str = '8h') -> float:
    """
    Annualize a funding rate.

    Args:
        rate: Funding rate (decimal)
        interval: Rate interval

    Returns:
        Annualized rate (decimal)
    """
    interval_map = {
        'hourly': 1, '1h': 1, '4h': 4, '8h': 8, 'daily': 24
    }
    hours = interval_map.get(interval.lower(), 8)
    fi = FundingInterval.from_hours(hours)

    return rate * fi.normalization_to_annual
