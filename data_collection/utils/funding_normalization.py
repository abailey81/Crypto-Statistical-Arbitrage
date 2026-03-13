"""
Funding Rate Normalization Module for Crypto Statistical Arbitrage.

This module provides professional-quality funding rate normalization, alignment,
and cross-venue comparison specifically designed for statistical arbitrage
strategies exploiting funding rate differentials across perpetual swap venues.

==============================================================================
FUNDING INTERVAL REFERENCE
==============================================================================

Venue funding intervals and annualization factors:
+----------------+----------+-------------+---------------+------------------+
| Venue | Interval | Periods/Day | Periods/Year | Settlement Time |
+----------------+----------+-------------+---------------+------------------+
| Binance | 8 hours | 3 | 1,095 | 00:00, 08:00 UTC |
| Bybit | 8 hours | 3 | 1,095 | 00:00, 08:00 UTC |
| OKX | 8 hours | 3 | 1,095 | 00:00, 08:00 UTC |
| Deribit | 8 hours | 3 | 1,095 | 08:00 UTC |
| Kraken | 4 hours | 6 | 2,190 | Every 4h UTC |
| Hyperliquid | 1 hour | 24 | 8,760 | Every hour |
| dYdX v4 | 1 hour | 24 | 8,760 | Every hour |
| GMX | 1 hour | 24 | 8,760 | Continuous |
| Vertex | 1 hour | 24 | 8,760 | Every hour |
| Synthetix | 1 hour | 24 | 8,760 | Every hour |
+----------------+----------+-------------+---------------+------------------+

==============================================================================
NORMALIZATION METHODOLOGY
==============================================================================

Rate Conversion Formula:
    rate_target = rate_source Ã— (source_periods_per_day / target_periods_per_day)

Example: Convert hourly rate to 8-hour equivalent:
    rate_8h = rate_1h Ã— (24 / 3) = rate_1h Ã— 8

Annualization Formula:
    rate_annual = rate_period Ã— periods_per_year

    For 8-hour funding: rate_annual = rate_8h Ã— 1,095
    For hourly funding: rate_annual = rate_1h Ã— 8,760

APR vs APY:
    APR (Simple): rate_annual = rate_period Ã— periods
    APY (Compound): rate_annual = (1 + rate_period)^periods - 1

    For small rates, APR â‰ˆ APY. Use APR for simplicity in funding analysis.

==============================================================================
TIMESTAMP ALIGNMENT
==============================================================================

Cross-venue funding rate analysis requires aligned timestamps:

Alignment Strategies:
+----------------+----------------------------------------+------------------+
| Strategy | Description | Best For |
+----------------+----------------------------------------+------------------+
| NEAREST | Match to closest timestamp | Real-time spread |
| FORWARD | Match to next available timestamp | Conservative |
| BACKWARD | Match to previous timestamp | Historical |
| INTERPOLATE | Linear interpolation between points | Smooth analysis |
| RESAMPLE | Resample to common frequency | Backtesting |
+----------------+----------------------------------------+------------------+

Tolerance Guidelines:
    - Real-time trading: Â±5 minutes maximum
    - Intraday analysis: Â±30 minutes
    - Daily aggregation: Â±1 hour
    - Historical backtest: Exact match preferred

==============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
==============================================================================

1. CROSS-VENUE FUNDING SPREAD
   - Normalize rates to common interval before comparison
   - Account for venue-specific settlement times
   - Calculate net spread after execution costs

2. FUNDING RATE PREDICTION
   - Use normalized historical rates for time-series analysis
   - Detect regime changes across venues
   - Build predictive models with aligned features

3. CARRY TRADE OPTIMIZATION
   - Compare annualized returns across venues
   - Factor in execution costs and slippage
   - Optimize venue selection for carry strategies

4. RISK-ADJUSTED ANALYSIS
   - Calculate volatility on normalized rates
   - Compute Sharpe ratios with consistent intervals
   - Assess correlation structure across venues

==============================================================================
USAGE EXAMPLES
==============================================================================

Basic Normalization:
    >>> normalizer = FundingNormalizer()
    >>> normalized_df = normalizer.normalize_to_interval(
    ... df, target_interval=NormalizationInterval.EIGHT_HOUR
    ... )

Cross-Venue Alignment:
    >>> aligned = normalizer.align_venues(
    ... binance_df, hyperliquid_df,
    ... strategy=AlignmentStrategy.NEAREST,
    ... tolerance_minutes=30
    ... )

Spread Calculation:
    >>> spread = normalizer.calculate_normalized_spread(
    ... df, venue_a='binance', venue_b='hyperliquid',
    ... output_interval=NormalizationInterval.EIGHT_HOUR
    ... )

Version: 2.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Settlement times by venue (UTC hour)
VENUE_SETTLEMENT_TIMES = {
    'binance': [0, 8, 16],
    'bybit': [0, 8, 16],
    'okx': [0, 8, 16],
    'deribit': [8], # Single daily settlement at 08:00 UTC
    'kraken': [0, 4, 8, 12, 16, 20],
    'hyperliquid': list(range(24)), # Every hour
    'dydx': list(range(24)),
    'dydx_v4': list(range(24)),
    'gmx': list(range(24)),
    'vertex': list(range(24)),
    'synthetix': list(range(24)),
    'aevo': list(range(24)),
}

# Typical funding rate ranges by venue (annualized %)
VENUE_TYPICAL_RANGES = {
    'binance': (-50.0, 100.0),
    'bybit': (-50.0, 100.0),
    'okx': (-50.0, 100.0),
    'deribit': (-30.0, 80.0),
    'kraken': (-40.0, 90.0),
    'hyperliquid': (-100.0, 200.0), # More volatile
    'dydx': (-80.0, 150.0),
    'dydx_v4': (-80.0, 150.0),
    'gmx': (-60.0, 120.0),
    'vertex': (-80.0, 150.0),
}

# =============================================================================
# ENUMS
# =============================================================================

class NormalizationInterval(Enum):
    """
    Target intervals for funding rate normalization.

    Each interval has associated conversion factors and
    typical use cases in statistical arbitrage.
    """

    CONTINUOUS = "continuous" # Per-second rate (theoretical)
    HOURLY = "hourly" # 1-hour intervals
    FOUR_HOUR = "four_hour" # 4-hour intervals
    EIGHT_HOUR = "eight_hour" # 8-hour intervals (CEX standard)
    DAILY = "daily" # 24-hour intervals
    ANNUAL = "annual" # Annualized rate

    @property
    def hours(self) -> float:
        """Hours per interval."""
        hours_map = {
            NormalizationInterval.CONTINUOUS: 1 / 3600,
            NormalizationInterval.HOURLY: 1.0,
            NormalizationInterval.FOUR_HOUR: 4.0,
            NormalizationInterval.EIGHT_HOUR: 8.0,
            NormalizationInterval.DAILY: 24.0,
            NormalizationInterval.ANNUAL: 8760.0,
        }
        return hours_map.get(self, 8.0)

    @property
    def periods_per_day(self) -> float:
        """Number of periods per day."""
        if self == NormalizationInterval.CONTINUOUS:
            return 86400.0
        elif self == NormalizationInterval.ANNUAL:
            return 1 / 365.0
        return 24.0 / self.hours

    @property
    def periods_per_year(self) -> float:
        """Number of periods per year (365 days)."""
        return self.periods_per_day * 365

    @property
    def pandas_freq(self) -> str:
        """Pandas frequency string."""
        freq_map = {
            NormalizationInterval.CONTINUOUS: '1s',
            NormalizationInterval.HOURLY: '1h',
            NormalizationInterval.FOUR_HOUR: '4h',
            NormalizationInterval.EIGHT_HOUR: '8h',
            NormalizationInterval.DAILY: '1D',
            NormalizationInterval.ANNUAL: '365D',
        }
        return freq_map.get(self, '8h')

    @property
    def typical_use_case(self) -> str:
        """Typical use case for this interval."""
        use_cases = {
            NormalizationInterval.CONTINUOUS: "Theoretical continuous rate",
            NormalizationInterval.HOURLY: "DEX/hybrid venue comparison",
            NormalizationInterval.FOUR_HOUR: "Kraken-specific analysis",
            NormalizationInterval.EIGHT_HOUR: "CEX standard, cross-venue spreads",
            NormalizationInterval.DAILY: "Daily P&L reporting",
            NormalizationInterval.ANNUAL: "Return comparison, Sharpe calculation",
        }
        return use_cases.get(self, "General analysis")

    def conversion_factor_to(self, target: 'NormalizationInterval') -> float:
        """Get conversion factor from this interval to target."""
        return self.hours / target.hours

    @classmethod
    def from_hours(cls, hours: float) -> 'NormalizationInterval':
        """Determine interval from hours value."""
        if hours <= 1 / 3600:
            return cls.CONTINUOUS
        elif hours <= 1.5:
            return cls.HOURLY
        elif hours <= 5:
            return cls.FOUR_HOUR
        elif hours <= 12:
            return cls.EIGHT_HOUR
        elif hours <= 36:
            return cls.DAILY
        return cls.ANNUAL

    @classmethod
    def detect_from_timestamps(
        cls,
        timestamps: pd.Series,
        tolerance_minutes: int = 30
    ) -> 'NormalizationInterval':
        """Auto-detect interval from timestamp series."""
        if len(timestamps) < 2:
            return cls.EIGHT_HOUR

        timestamps = pd.to_datetime(timestamps)
        diffs = timestamps.diff().dropna()
        median_diff = diffs.median()

        hours = median_diff.total_seconds() / 3600
        return cls.from_hours(hours)

class AlignmentStrategy(Enum):
    """
    Timestamp alignment strategies for cross-venue comparison.

    Different strategies are optimal for different use cases
    in statistical arbitrage analysis.
    """

    NEAREST = "nearest" # Match to closest timestamp
    FORWARD = "forward" # Match to next available
    BACKWARD = "backward" # Match to previous available
    INTERPOLATE = "interpolate" # Linear interpolation
    RESAMPLE = "resample" # Resample to common frequency
    EXACT = "exact" # Exact match only

    @property
    def pandas_direction(self) -> str:
        """Direction for pandas merge_asof."""
        directions = {
            AlignmentStrategy.NEAREST: 'nearest',
            AlignmentStrategy.FORWARD: 'forward',
            AlignmentStrategy.BACKWARD: 'backward',
        }
        return directions.get(self, 'nearest')

    @property
    def suitable_for_trading(self) -> bool:
        """Whether strategy is suitable for live trading decisions."""
        return self in {AlignmentStrategy.BACKWARD, AlignmentStrategy.EXACT}

    @property
    def introduces_lookahead(self) -> bool:
        """Whether strategy can introduce look-ahead bias."""
        return self in {AlignmentStrategy.FORWARD, AlignmentStrategy.INTERPOLATE}

    @property
    def recommended_tolerance_minutes(self) -> int:
        """Recommended tolerance for this strategy."""
        tolerances = {
            AlignmentStrategy.NEAREST: 30,
            AlignmentStrategy.FORWARD: 60,
            AlignmentStrategy.BACKWARD: 60,
            AlignmentStrategy.INTERPOLATE: 120,
            AlignmentStrategy.RESAMPLE: 0, # Not applicable
            AlignmentStrategy.EXACT: 1,
        }
        return tolerances.get(self, 30)

    @property
    def description(self) -> str:
        """Human-readable description."""
        descriptions = {
            AlignmentStrategy.NEAREST: "Match to closest timestamp within tolerance",
            AlignmentStrategy.FORWARD: "Match to next available (conservative)",
            AlignmentStrategy.BACKWARD: "Match to previous available (no lookahead)",
            AlignmentStrategy.INTERPOLATE: "Linear interpolation between points",
            AlignmentStrategy.RESAMPLE: "Resample both series to common frequency",
            AlignmentStrategy.EXACT: "Require exact timestamp match",
        }
        return descriptions.get(self, "Unknown strategy")

class AggregationMethod(Enum):
    """
    Methods for aggregating funding rates across time periods.

    Different methods have different implications for
    carry trade P&L calculation.
    """

    SUM = "sum" # Sum rates (cumulative funding)
    MEAN = "mean" # Average rate
    LAST = "last" # Last rate in period
    FIRST = "first" # First rate in period
    TWAP = "twap" # Time-weighted average
    VWAP = "vwap" # Volume-weighted (if volume available)
    COMPOUND = "compound" # Compound returns

    @property
    def pandas_agg(self) -> str:
        """Pandas aggregation function name."""
        agg_map = {
            AggregationMethod.SUM: 'sum',
            AggregationMethod.MEAN: 'mean',
            AggregationMethod.LAST: 'last',
            AggregationMethod.FIRST: 'first',
            AggregationMethod.TWAP: 'mean',
        }
        return agg_map.get(self, 'sum')

    @property
    def is_cumulative(self) -> bool:
        """Whether method produces cumulative value."""
        return self in {AggregationMethod.SUM, AggregationMethod.COMPOUND}

    @property
    def best_for(self) -> str:
        """Best use case for this method."""
        use_cases = {
            AggregationMethod.SUM: "P&L calculation over period",
            AggregationMethod.MEAN: "Average rate comparison",
            AggregationMethod.LAST: "Current rate snapshot",
            AggregationMethod.FIRST: "Period opening rate",
            AggregationMethod.TWAP: "Fair average rate",
            AggregationMethod.VWAP: "Liquidity-adjusted rate",
            AggregationMethod.COMPOUND: "Accurate long-term returns",
        }
        return use_cases.get(self, "General use")

class VenueIntervalType(Enum):
    """
    Classification of venue funding interval types.

    Helps determine appropriate normalization strategies.
    """

    CEX_STANDARD = "cex_standard" # 8-hour CEX intervals
    CEX_FREQUENT = "cex_frequent" # 4-hour intervals (Kraken)
    HYBRID_HOURLY = "hybrid_hourly" # 1-hour hybrid venues
    DEX_CONTINUOUS = "dex_continuous" # Continuous/per-block
    DEX_HOURLY = "dex_hourly" # Hourly DEX settlements

    @property
    def typical_interval(self) -> NormalizationInterval:
        """Typical interval for this venue type."""
        intervals = {
            VenueIntervalType.CEX_STANDARD: NormalizationInterval.EIGHT_HOUR,
            VenueIntervalType.CEX_FREQUENT: NormalizationInterval.FOUR_HOUR,
            VenueIntervalType.HYBRID_HOURLY: NormalizationInterval.HOURLY,
            VenueIntervalType.DEX_CONTINUOUS: NormalizationInterval.CONTINUOUS,
            VenueIntervalType.DEX_HOURLY: NormalizationInterval.HOURLY,
        }
        return intervals.get(self, NormalizationInterval.EIGHT_HOUR)

    @property
    def aggregation_method(self) -> AggregationMethod:
        """Recommended aggregation method when upsampling."""
        methods = {
            VenueIntervalType.CEX_STANDARD: AggregationMethod.SUM,
            VenueIntervalType.CEX_FREQUENT: AggregationMethod.SUM,
            VenueIntervalType.HYBRID_HOURLY: AggregationMethod.SUM,
            VenueIntervalType.DEX_CONTINUOUS: AggregationMethod.MEAN,
            VenueIntervalType.DEX_HOURLY: AggregationMethod.SUM,
        }
        return methods.get(self, AggregationMethod.SUM)

    @classmethod
    def from_venue(cls, venue: str) -> 'VenueIntervalType':
        """Determine interval type from venue name."""
        venue_lower = venue.lower()

        if venue_lower in ['binance', 'bybit', 'okx', 'deribit', 'coinbase']:
            return cls.CEX_STANDARD
        elif venue_lower in ['kraken']:
            return cls.CEX_FREQUENT
        elif venue_lower in ['hyperliquid', 'dydx', 'dydx_v4', 'vertex', 'aevo']:
            return cls.HYBRID_HOURLY
        elif venue_lower in ['gmx', 'synthetix']:
            return cls.DEX_HOURLY

        return cls.CEX_STANDARD

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class VenueInterval:
    """
    Venue-specific funding interval configuration.

    Contains all parameters needed to normalize rates from this venue.
    """

    venue: str
    interval: NormalizationInterval
    settlement_times_utc: List[int]
    typical_range_annual_pct: Tuple[float, float]
    interval_type: VenueIntervalType

    @property
    def hours(self) -> float:
        """Hours per funding period."""
        return self.interval.hours

    @property
    def periods_per_day(self) -> float:
        """Funding periods per day."""
        return self.interval.periods_per_day

    @property
    def periods_per_year(self) -> float:
        """Funding periods per year."""
        return self.interval.periods_per_year

    @property
    def annualization_factor(self) -> float:
        """Factor to annualize a single period rate."""
        return self.periods_per_year

    @property
    def next_settlement_utc(self) -> datetime:
        """Next settlement time from now."""
        now = datetime.now(timezone.utc)
        current_hour = now.hour

        for hour in sorted(self.settlement_times_utc):
            if hour > current_hour:
                return now.replace(hour=hour, minute=0, second=0, microsecond=0)

        # Next day first settlement
        next_day = now + timedelta(days=1)
        return next_day.replace(
            hour=self.settlement_times_utc[0],
            minute=0, second=0, microsecond=0
        )

    @property
    def time_to_next_settlement_hours(self) -> float:
        """Hours until next settlement."""
        now = datetime.now(timezone.utc)
        delta = self.next_settlement_utc - now
        return delta.total_seconds() / 3600

    @property
    def is_near_settlement(self) -> bool:
        """Whether within 30 minutes of settlement."""
        return self.time_to_next_settlement_hours < 0.5

    def rate_within_typical_range(self, annualized_rate_pct: float) -> bool:
        """Check if rate is within typical range."""
        min_rate, max_rate = self.typical_range_annual_pct
        return min_rate <= annualized_rate_pct <= max_rate

    def conversion_factor_to(self, target: NormalizationInterval) -> float:
        """Get conversion factor to target interval."""
        return self.interval.conversion_factor_to(target)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'venue': self.venue,
            'interval': self.interval.value,
            'hours': self.hours,
            'periods_per_day': self.periods_per_day,
            'periods_per_year': self.periods_per_year,
            'annualization_factor': self.annualization_factor,
            'settlement_times_utc': self.settlement_times_utc,
            'typical_range_annual_pct': self.typical_range_annual_pct,
            'interval_type': self.interval_type.value,
            'next_settlement_utc': self.next_settlement_utc.isoformat(),
            'time_to_next_settlement_hours': round(self.time_to_next_settlement_hours, 2),
            'is_near_settlement': self.is_near_settlement,
        }

@dataclass
class NormalizedRate:
    """
    Funding rate normalized to a specific interval.

    Contains original and normalized values with full conversion metadata.
    """

    timestamp: datetime
    symbol: str
    venue: str
    original_rate: float
    original_interval: NormalizationInterval
    normalized_rate: float
    target_interval: NormalizationInterval

    @property
    def conversion_factor(self) -> float:
        """Factor used for conversion."""
        return self.original_interval.conversion_factor_to(self.target_interval)

    @property
    def annualized_rate(self) -> float:
        """Annualized rate from normalized rate."""
        return self.normalized_rate * self.target_interval.periods_per_year

    @property
    def annualized_rate_pct(self) -> float:
        """Annualized rate as percentage."""
        return self.annualized_rate * 100

    @property
    def daily_rate(self) -> float:
        """Daily equivalent rate."""
        return self.normalized_rate * self.target_interval.periods_per_day

    @property
    def daily_rate_bps(self) -> float:
        """Daily rate in basis points."""
        return self.daily_rate * 10000

    @property
    def is_positive(self) -> bool:
        """Whether rate is positive (longs pay shorts)."""
        return self.normalized_rate > 0

    @property
    def carry_direction(self) -> str:
        """Direction to earn carry."""
        return "short" if self.is_positive else "long"

    @property
    def rate_8h_equivalent(self) -> float:
        """Rate converted to 8-hour equivalent."""
        factor = self.target_interval.conversion_factor_to(NormalizationInterval.EIGHT_HOUR)
        return self.normalized_rate * factor

    @property
    def rate_8h_bps(self) -> float:
        """8-hour equivalent rate in basis points."""
        return self.rate_8h_equivalent * 10000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'venue': self.venue,
            'original_rate': self.original_rate,
            'original_interval': self.original_interval.value,
            'normalized_rate': self.normalized_rate,
            'target_interval': self.target_interval.value,
            'conversion_factor': round(self.conversion_factor, 4),
            'annualized_rate': round(self.annualized_rate, 6),
            'annualized_rate_pct': round(self.annualized_rate_pct, 2),
            'daily_rate_bps': round(self.daily_rate_bps, 2),
            'rate_8h_bps': round(self.rate_8h_bps, 2),
            'is_positive': self.is_positive,
            'carry_direction': self.carry_direction,
        }

@dataclass
class AlignedRatePair:
    """
    Pair of aligned funding rates from two venues.

    Contains rates aligned to the same timestamp with spread analytics.
    """

    timestamp: datetime
    symbol: str
    venue_a: str
    venue_b: str
    rate_a: float
    rate_b: float
    interval: NormalizationInterval
    alignment_strategy: AlignmentStrategy
    time_diff_seconds: float = 0.0

    @property
    def spread(self) -> float:
        """Raw spread (A - B)."""
        return self.rate_a - self.rate_b

    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        return self.spread * 10000

    @property
    def spread_annualized(self) -> float:
        """Annualized spread."""
        return self.spread * self.interval.periods_per_year

    @property
    def spread_annualized_pct(self) -> float:
        """Annualized spread as percentage."""
        return self.spread_annualized * 100

    @property
    def abs_spread_bps(self) -> float:
        """Absolute spread in basis points."""
        return abs(self.spread_bps)

    @property
    def long_venue(self) -> str:
        """Venue to go long on (lower funding rate)."""
        return self.venue_a if self.rate_a < self.rate_b else self.venue_b

    @property
    def short_venue(self) -> str:
        """Venue to go short on (higher funding rate)."""
        return self.venue_b if self.rate_a < self.rate_b else self.venue_a

    @property
    def is_arbitrage_opportunity(self) -> bool:
        """Whether spread indicates potential arbitrage (>10bps)."""
        return self.abs_spread_bps > 10.0

    @property
    def alignment_quality(self) -> str:
        """Quality assessment of timestamp alignment."""
        if self.time_diff_seconds <= 60:
            return "excellent"
        elif self.time_diff_seconds <= 300:
            return "good"
        elif self.time_diff_seconds <= 900:
            return "acceptable"
        elif self.time_diff_seconds <= 1800:
            return "poor"
        return "unreliable"

    @property
    def is_alignment_reliable(self) -> bool:
        """Whether alignment is reliable for trading decisions."""
        return self.alignment_quality in ["excellent", "good", "acceptable"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'venue_a': self.venue_a,
            'venue_b': self.venue_b,
            'rate_a': round(self.rate_a, 8),
            'rate_b': round(self.rate_b, 8),
            'interval': self.interval.value,
            'spread': round(self.spread, 8),
            'spread_bps': round(self.spread_bps, 2),
            'spread_annualized_pct': round(self.spread_annualized_pct, 2),
            'long_venue': self.long_venue,
            'short_venue': self.short_venue,
            'is_arbitrage_opportunity': self.is_arbitrage_opportunity,
            'alignment_strategy': self.alignment_strategy.value,
            'time_diff_seconds': round(self.time_diff_seconds, 1),
            'alignment_quality': self.alignment_quality,
            'is_alignment_reliable': self.is_alignment_reliable,
        }

@dataclass
class NormalizationResult:
    """
    Complete result from funding rate normalization.

    Contains normalized DataFrame with comprehensive metadata.
    """

    df: pd.DataFrame
    source_venue: str
    source_interval: NormalizationInterval
    target_interval: NormalizationInterval
    aggregation_method: AggregationMethod
    rows_input: int
    rows_output: int
    timestamp_range: Tuple[datetime, datetime]
    processing_time_ms: float

    @property
    def compression_ratio(self) -> float:
        """Ratio of output to input rows."""
        if self.rows_input == 0:
            return 0.0
        return self.rows_output / self.rows_input

    @property
    def was_aggregated(self) -> bool:
        """Whether data was aggregated (downsampled)."""
        return self.source_interval.hours < self.target_interval.hours

    @property
    def was_interpolated(self) -> bool:
        """Whether data was interpolated (upsampled)."""
        return self.source_interval.hours > self.target_interval.hours

    @property
    def coverage_hours(self) -> float:
        """Total hours of data coverage."""
        delta = self.timestamp_range[1] - self.timestamp_range[0]
        return delta.total_seconds() / 3600

    @property
    def expected_rows(self) -> int:
        """Expected rows based on coverage and interval."""
        return int(self.coverage_hours / self.target_interval.hours)

    @property
    def completeness_pct(self) -> float:
        """Data completeness percentage."""
        if self.expected_rows == 0:
            return 0.0
        return min(100.0, (self.rows_output / self.expected_rows) * 100)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary (excludes DataFrame)."""
        return {
            'source_venue': self.source_venue,
            'source_interval': self.source_interval.value,
            'target_interval': self.target_interval.value,
            'aggregation_method': self.aggregation_method.value,
            'rows_input': self.rows_input,
            'rows_output': self.rows_output,
            'compression_ratio': round(self.compression_ratio, 4),
            'was_aggregated': self.was_aggregated,
            'was_interpolated': self.was_interpolated,
            'timestamp_range': [
                self.timestamp_range[0].isoformat(),
                self.timestamp_range[1].isoformat(),
            ],
            'coverage_hours': round(self.coverage_hours, 2),
            'expected_rows': self.expected_rows,
            'completeness_pct': round(self.completeness_pct, 2),
            'processing_time_ms': round(self.processing_time_ms, 2),
        }

@dataclass
class AlignmentResult:
    """
    Result from cross-venue timestamp alignment.

    Contains aligned DataFrames with alignment quality metrics.
    """

    df_aligned: pd.DataFrame
    venue_a: str
    venue_b: str
    strategy: AlignmentStrategy
    tolerance_seconds: float
    matched_rows: int
    unmatched_a: int
    unmatched_b: int
    mean_time_diff_seconds: float
    max_time_diff_seconds: float

    @property
    def match_rate_pct(self) -> float:
        """Percentage of rows successfully matched."""
        total = self.matched_rows + max(self.unmatched_a, self.unmatched_b)
        if total == 0:
            return 0.0
        return (self.matched_rows / total) * 100

    @property
    def alignment_quality(self) -> str:
        """Overall alignment quality assessment."""
        if self.mean_time_diff_seconds <= 60 and self.match_rate_pct >= 95:
            return "excellent"
        elif self.mean_time_diff_seconds <= 300 and self.match_rate_pct >= 85:
            return "good"
        elif self.mean_time_diff_seconds <= 600 and self.match_rate_pct >= 70:
            return "acceptable"
        elif self.match_rate_pct >= 50:
            return "poor"
        return "unreliable"

    @property
    def is_backtest_reliable(self) -> bool:
        """Whether alignment is reliable for backtesting."""
        return self.alignment_quality in ["excellent", "good"]

    @property
    def is_trading_reliable(self) -> bool:
        """Whether alignment is reliable for live trading."""
        return self.alignment_quality == "excellent" and not self.strategy.introduces_lookahead

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'venue_a': self.venue_a,
            'venue_b': self.venue_b,
            'strategy': self.strategy.value,
            'tolerance_seconds': self.tolerance_seconds,
            'matched_rows': self.matched_rows,
            'unmatched_a': self.unmatched_a,
            'unmatched_b': self.unmatched_b,
            'match_rate_pct': round(self.match_rate_pct, 2),
            'mean_time_diff_seconds': round(self.mean_time_diff_seconds, 1),
            'max_time_diff_seconds': round(self.max_time_diff_seconds, 1),
            'alignment_quality': self.alignment_quality,
            'is_backtest_reliable': self.is_backtest_reliable,
            'is_trading_reliable': self.is_trading_reliable,
        }

# =============================================================================
# VENUE CONFIGURATIONS
# =============================================================================

VENUE_INTERVALS: Dict[str, VenueInterval] = {
    'binance': VenueInterval(
        venue='binance',
        interval=NormalizationInterval.EIGHT_HOUR,
        settlement_times_utc=[0, 8, 16],
        typical_range_annual_pct=(-50.0, 100.0),
        interval_type=VenueIntervalType.CEX_STANDARD,
    ),
    'bybit': VenueInterval(
        venue='bybit',
        interval=NormalizationInterval.EIGHT_HOUR,
        settlement_times_utc=[0, 8, 16],
        typical_range_annual_pct=(-50.0, 100.0),
        interval_type=VenueIntervalType.CEX_STANDARD,
    ),
    'okx': VenueInterval(
        venue='okx',
        interval=NormalizationInterval.EIGHT_HOUR,
        settlement_times_utc=[0, 8, 16],
        typical_range_annual_pct=(-50.0, 100.0),
        interval_type=VenueIntervalType.CEX_STANDARD,
    ),
    'deribit': VenueInterval(
        venue='deribit',
        interval=NormalizationInterval.EIGHT_HOUR,
        settlement_times_utc=[8],
        typical_range_annual_pct=(-30.0, 80.0),
        interval_type=VenueIntervalType.CEX_STANDARD,
    ),
    'kraken': VenueInterval(
        venue='kraken',
        interval=NormalizationInterval.FOUR_HOUR,
        settlement_times_utc=[0, 4, 8, 12, 16, 20],
        typical_range_annual_pct=(-40.0, 90.0),
        interval_type=VenueIntervalType.CEX_FREQUENT,
    ),
    'hyperliquid': VenueInterval(
        venue='hyperliquid',
        interval=NormalizationInterval.HOURLY,
        settlement_times_utc=list(range(24)),
        typical_range_annual_pct=(-100.0, 200.0),
        interval_type=VenueIntervalType.HYBRID_HOURLY,
    ),
    'dydx': VenueInterval(
        venue='dydx',
        interval=NormalizationInterval.HOURLY,
        settlement_times_utc=list(range(24)),
        typical_range_annual_pct=(-80.0, 150.0),
        interval_type=VenueIntervalType.HYBRID_HOURLY,
    ),
    'dydx_v4': VenueInterval(
        venue='dydx_v4',
        interval=NormalizationInterval.HOURLY,
        settlement_times_utc=list(range(24)),
        typical_range_annual_pct=(-80.0, 150.0),
        interval_type=VenueIntervalType.HYBRID_HOURLY,
    ),
    'gmx': VenueInterval(
        venue='gmx',
        interval=NormalizationInterval.HOURLY,
        settlement_times_utc=list(range(24)),
        typical_range_annual_pct=(-60.0, 120.0),
        interval_type=VenueIntervalType.DEX_HOURLY,
    ),
    'vertex': VenueInterval(
        venue='vertex',
        interval=NormalizationInterval.HOURLY,
        settlement_times_utc=list(range(24)),
        typical_range_annual_pct=(-80.0, 150.0),
        interval_type=VenueIntervalType.HYBRID_HOURLY,
    ),
    'aevo': VenueInterval(
        venue='aevo',
        interval=NormalizationInterval.HOURLY,
        settlement_times_utc=list(range(24)),
        typical_range_annual_pct=(-80.0, 150.0),
        interval_type=VenueIntervalType.HYBRID_HOURLY,
    ),
    'synthetix': VenueInterval(
        venue='synthetix',
        interval=NormalizationInterval.HOURLY,
        settlement_times_utc=list(range(24)),
        typical_range_annual_pct=(-60.0, 120.0),
        interval_type=VenueIntervalType.DEX_HOURLY,
    ),
}

def get_venue_interval(venue: str) -> VenueInterval:
    """Get venue interval configuration."""
    return VENUE_INTERVALS.get(
        venue.lower(),
        VENUE_INTERVALS['binance'] # Default to Binance config
    )

# =============================================================================
# NORMALIZER CLASS
# =============================================================================

class FundingNormalizer:
    """
    professional-quality funding rate normalizer.

    Provides comprehensive normalization, alignment, and comparison
    of funding rates across venues for statistical arbitrage.
    """

    def __init__(
        self,
        default_target: NormalizationInterval = NormalizationInterval.EIGHT_HOUR,
        default_alignment: AlignmentStrategy = AlignmentStrategy.NEAREST,
        default_aggregation: AggregationMethod = AggregationMethod.SUM,
    ):
        """
        Initialize normalizer.

        Args:
            default_target: Default target interval for normalization
            default_alignment: Default alignment strategy
            default_aggregation: Default aggregation method
        """
        self.default_target = default_target
        self.default_alignment = default_alignment
        self.default_aggregation = default_aggregation

    def normalize_rate(
        self,
        rate: float,
        source_venue: str,
        target_interval: Optional[NormalizationInterval] = None,
    ) -> NormalizedRate:
        """
        Normalize a single funding rate.

        Args:
            rate: Original funding rate
            source_venue: Source venue name
            target_interval: Target interval (default: class default)

        Returns:
            NormalizedRate with conversion details
        """
        target = target_interval or self.default_target
        venue_config = get_venue_interval(source_venue)

        factor = venue_config.conversion_factor_to(target)
        normalized = rate * factor

        return NormalizedRate(
            timestamp=datetime.now(timezone.utc),
            symbol='',
            venue=source_venue,
            original_rate=rate,
            original_interval=venue_config.interval,
            normalized_rate=normalized,
            target_interval=target,
        )

    def normalize_dataframe(
        self,
        df: pd.DataFrame,
        venue: str,
        target_interval: Optional[NormalizationInterval] = None,
        aggregation_method: Optional[AggregationMethod] = None,
        timestamp_col: str = 'timestamp',
        funding_col: str = 'funding_rate',
        symbol_col: str = 'symbol',
    ) -> NormalizationResult:
        """
        Normalize funding rates in a DataFrame.

        Args:
            df: DataFrame with funding rates
            venue: Source venue name
            target_interval: Target interval
            aggregation_method: Aggregation method for downsampling
            timestamp_col: Timestamp column name
            funding_col: Funding rate column name
            symbol_col: Symbol column name

        Returns:
            NormalizationResult with normalized DataFrame
        """
        import time
        start_time = time.time()

        target = target_interval or self.default_target
        agg_method = aggregation_method or self.default_aggregation
        venue_config = get_venue_interval(venue)

        if df.empty:
            return NormalizationResult(
                df=pd.DataFrame(),
                source_venue=venue,
                source_interval=venue_config.interval,
                target_interval=target,
                aggregation_method=agg_method,
                rows_input=0,
                rows_output=0,
                timestamp_range=(datetime.min, datetime.min),
                processing_time_ms=0,
            )

        result_df = df.copy()

        # Ensure timestamp is datetime
        result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col], utc=True)

        # Detect source interval if not known
        detected = NormalizationInterval.detect_from_timestamps(result_df[timestamp_col])
        source_interval = venue_config.interval if venue_config else detected

        # Calculate conversion factor
        factor = source_interval.conversion_factor_to(target)

        if source_interval.hours < target.hours:
            # Aggregate (e.g., hourly -> 8-hour)
            result_df = self._aggregate_rates(
                result_df, target, agg_method, timestamp_col, funding_col, symbol_col
            )
        elif source_interval.hours > target.hours:
            # Interpolate (e.g., 8-hour -> hourly)
            result_df = self._interpolate_rates(
                result_df, target, timestamp_col, funding_col, symbol_col
            )
            factor = 1.0 # Already handled in interpolation

        # Apply conversion factor
        result_df[funding_col] = result_df[funding_col] * factor

        # Add metadata columns
        result_df['source_interval'] = source_interval.value
        result_df['target_interval'] = target.value
        result_df['conversion_factor'] = factor

        # Calculate annualized rate
        result_df['funding_rate_annualized'] = result_df[funding_col] * target.periods_per_year

        processing_time = (time.time() - start_time) * 1000

        ts_range = (
            result_df[timestamp_col].min().to_pydatetime(),
            result_df[timestamp_col].max().to_pydatetime(),
        )

        return NormalizationResult(
            df=result_df,
            source_venue=venue,
            source_interval=source_interval,
            target_interval=target,
            aggregation_method=agg_method,
            rows_input=len(df),
            rows_output=len(result_df),
            timestamp_range=ts_range,
            processing_time_ms=processing_time,
        )

    def normalize_to_interval(
        self,
        df: pd.DataFrame,
        venue: str,
        target_interval: Union[str, NormalizationInterval] = '8h',
        timestamp_col: str = 'timestamp',
        funding_col: str = 'funding_rate',
        symbol_col: str = 'symbol',
    ) -> pd.DataFrame:
        """
        Convenience method to normalize funding rates and return just the DataFrame.

        This is a wrapper around normalize_dataframe that converts string intervals
        and returns the DataFrame directly for easier pipeline integration.

        Args:
            df: DataFrame with funding rates
            venue: Source venue name
            target_interval: Target interval ('8h', '1h', etc.) or NormalizationInterval
            timestamp_col: Timestamp column name
            funding_col: Funding rate column name
            symbol_col: Symbol column name

        Returns:
            DataFrame with normalized funding rates
        """
        # Convert string interval to enum
        if isinstance(target_interval, str):
            interval_map = {
                '8h': NormalizationInterval.EIGHT_HOUR,
                '1h': NormalizationInterval.HOURLY,
                '4h': NormalizationInterval.FOUR_HOUR,
                '1d': NormalizationInterval.DAILY,
                'daily': NormalizationInterval.DAILY,
                'hourly': NormalizationInterval.HOURLY,
            }
            target = interval_map.get(target_interval.lower(), NormalizationInterval.EIGHT_HOUR)
        else:
            target = target_interval

        result = self.normalize_dataframe(
            df=df,
            venue=venue,
            target_interval=target,
            timestamp_col=timestamp_col,
            funding_col=funding_col,
            symbol_col=symbol_col,
        )
        return result.df

    def _aggregate_rates(
        self,
        df: pd.DataFrame,
        target: NormalizationInterval,
        method: AggregationMethod,
        timestamp_col: str,
        funding_col: str,
        symbol_col: str,
    ) -> pd.DataFrame:
        """Aggregate rates to larger interval while preserving metadata columns."""
        df = df.copy()
        df = df.set_index(timestamp_col)

        # Identify metadata columns to preserve (non-numeric, take first value)
        metadata_cols = [
            col for col in df.columns
            if col not in [funding_col, symbol_col]
            and df[col].dtype == 'object'
        ]

        # Build aggregation dict
        agg_dict = {}
        if method == AggregationMethod.COMPOUND:
            agg_dict[funding_col] = lambda x: np.prod(1 + x) - 1 if len(x) > 0 else 0
        else:
            agg_dict[funding_col] = method.pandas_agg

        # Preserve metadata columns by taking first value
        for col in metadata_cols:
            agg_dict[col] = 'first'

        # Group by symbol and resample
        if symbol_col in df.columns:
            grouped = df.groupby(symbol_col)
            resampled = grouped.resample(target.pandas_freq)
        else:
            resampled = df.resample(target.pandas_freq)

        # Apply aggregation with all columns
        result_df = resampled.agg(agg_dict).reset_index()

        return result_df

    def _interpolate_rates(
        self,
        df: pd.DataFrame,
        target: NormalizationInterval,
        timestamp_col: str,
        funding_col: str,
        symbol_col: str,
    ) -> pd.DataFrame:
        """Interpolate rates to smaller interval."""
        df = df.copy()
        df = df.set_index(timestamp_col)

        # Determine split factor
        source_interval = NormalizationInterval.detect_from_timestamps(df.index.to_series())
        periods_per_source = int(source_interval.hours / target.hours)

        result_frames = []

        symbols = df[symbol_col].unique() if symbol_col in df.columns else [None]

        for symbol in symbols:
            if symbol:
                symbol_df = df[df[symbol_col] == symbol]
            else:
                symbol_df = df

            # Resample and forward fill, then divide rate
            resampled = symbol_df.resample(target.pandas_freq).ffill()
            resampled[funding_col] = resampled[funding_col] / periods_per_source

            result_frames.append(resampled)

        result_df = pd.concat(result_frames).reset_index()
        return result_df

    def align_venues(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        strategy: Optional[AlignmentStrategy] = None,
        tolerance_minutes: int = 30,
        timestamp_col: str = 'timestamp',
        funding_col: str = 'funding_rate',
        symbol_col: str = 'symbol',
        venue_col: str = 'venue',
    ) -> AlignmentResult:
        """
        Align funding rates from two venues.

        Args:
            df_a: First venue DataFrame
            df_b: Second venue DataFrame
            strategy: Alignment strategy
            tolerance_minutes: Maximum time difference for matching
            timestamp_col: Timestamp column name
            funding_col: Funding rate column name
            symbol_col: Symbol column name
            venue_col: Venue column name

        Returns:
            AlignmentResult with aligned DataFrame
        """
        strategy = strategy or self.default_alignment
        tolerance = pd.Timedelta(minutes=tolerance_minutes)

        # Prepare DataFrames
        df_a = df_a.copy()
        df_b = df_b.copy()

        df_a[timestamp_col] = pd.to_datetime(df_a[timestamp_col], utc=True)
        df_b[timestamp_col] = pd.to_datetime(df_b[timestamp_col], utc=True)

        df_a = df_a.sort_values(timestamp_col)
        df_b = df_b.sort_values(timestamp_col)

        # Get venue names
        venue_a = df_a[venue_col].iloc[0] if venue_col in df_a.columns else 'venue_a'
        venue_b = df_b[venue_col].iloc[0] if venue_col in df_b.columns else 'venue_b'

        # Rename columns for merge
        df_a = df_a.rename(columns={
            funding_col: f'{funding_col}_a',
            venue_col: 'venue_a_name',
        })
        df_b = df_b.rename(columns={
            funding_col: f'{funding_col}_b',
            venue_col: 'venue_b_name',
            timestamp_col: f'{timestamp_col}_b',
        })

        if strategy == AlignmentStrategy.RESAMPLE:
            # Resample both to common frequency
            interval = NormalizationInterval.detect_from_timestamps(df_a[timestamp_col])
            df_a = df_a.set_index(timestamp_col).resample(interval.pandas_freq).last().reset_index()
            df_b = df_b.set_index(f'{timestamp_col}_b').resample(interval.pandas_freq).last().reset_index()
            df_b = df_b.rename(columns={f'{timestamp_col}_b': timestamp_col})

            aligned = pd.merge(df_a, df_b, on=[timestamp_col, symbol_col], how='inner')
            aligned['time_diff_seconds'] = 0

        elif strategy == AlignmentStrategy.EXACT:
            aligned = pd.merge(
                df_a, df_b,
                left_on=[timestamp_col, symbol_col],
                right_on=[f'{timestamp_col}_b', symbol_col],
                how='inner'
            )
            aligned['time_diff_seconds'] = 0

        else:
            # Use merge_asof for nearest/forward/backward
            aligned = pd.merge_asof(
                df_a,
                df_b,
                left_on=timestamp_col,
                right_on=f'{timestamp_col}_b',
                by=symbol_col if symbol_col in df_a.columns and symbol_col in df_b.columns else None,
                direction=strategy.pandas_direction,
                tolerance=tolerance,
            )

            # Calculate time differences
            aligned['time_diff_seconds'] = (
                aligned[timestamp_col] - aligned[f'{timestamp_col}_b']
            ).abs().dt.total_seconds()

        # Drop rows with no match
        matched_mask = aligned[f'{funding_col}_b'].notna()
        matched_rows = matched_mask.sum()
        unmatched_a = len(df_a) - matched_rows
        unmatched_b = len(df_b) - matched_rows

        aligned = aligned[matched_mask].copy()

        # Calculate spread
        aligned['spread'] = aligned[f'{funding_col}_a'] - aligned[f'{funding_col}_b']
        aligned['spread_bps'] = aligned['spread'] * 10000

        # Statistics
        mean_diff = aligned['time_diff_seconds'].mean() if len(aligned) > 0 else float('inf')
        max_diff = aligned['time_diff_seconds'].max() if len(aligned) > 0 else float('inf')

        return AlignmentResult(
            df_aligned=aligned,
            venue_a=venue_a,
            venue_b=venue_b,
            strategy=strategy,
            tolerance_seconds=tolerance_minutes * 60,
            matched_rows=int(matched_rows),
            unmatched_a=int(unmatched_a),
            unmatched_b=int(unmatched_b),
            mean_time_diff_seconds=float(mean_diff),
            max_time_diff_seconds=float(max_diff),
        )

    def calculate_normalized_spread(
        self,
        df: pd.DataFrame,
        venue_a: str,
        venue_b: str,
        output_interval: Optional[NormalizationInterval] = None,
        timestamp_col: str = 'timestamp',
        funding_col: str = 'funding_rate',
        venue_col: str = 'venue',
        symbol_col: str = 'symbol',
    ) -> pd.DataFrame:
        """
        Calculate normalized spread between two venues.

        Both venues are normalized to the same interval before
        spread calculation.

        Args:
            df: DataFrame with funding rates from multiple venues
            venue_a: First venue name
            venue_b: Second venue name
            output_interval: Target interval for normalization
            timestamp_col: Timestamp column name
            funding_col: Funding rate column name
            venue_col: Venue column name
            symbol_col: Symbol column name

        Returns:
            DataFrame with normalized spread
        """
        output = output_interval or self.default_target

        # Filter venues
        df_a = df[df[venue_col] == venue_a].copy()
        df_b = df[df[venue_col] == venue_b].copy()

        # Normalize both to target interval
        result_a = self.normalize_dataframe(
            df_a, venue_a, output,
            timestamp_col=timestamp_col, funding_col=funding_col, symbol_col=symbol_col
        )
        result_b = self.normalize_dataframe(
            df_b, venue_b, output,
            timestamp_col=timestamp_col, funding_col=funding_col, symbol_col=symbol_col
        )

        # Align and calculate spread
        alignment = self.align_venues(
            result_a.df, result_b.df,
            timestamp_col=timestamp_col, funding_col=funding_col,
            symbol_col=symbol_col, venue_col=venue_col,
        )

        return alignment.df_aligned

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def normalize_funding_rate(
    rate: float,
    source_venue: str,
    target_interval: Union[NormalizationInterval, str] = NormalizationInterval.EIGHT_HOUR,
) -> float:
    """
    Quick normalization of a single funding rate.

    Args:
        rate: Original funding rate
        source_venue: Source venue name
        target_interval: Target interval

    Returns:
        Normalized rate
    """
    if isinstance(target_interval, str):
        target_interval = NormalizationInterval(target_interval)

    normalizer = FundingNormalizer()
    result = normalizer.normalize_rate(rate, source_venue, target_interval)
    return result.normalized_rate

def annualize_funding_rate(
    rate: float,
    source_venue: str,
) -> float:
    """
    Annualize a funding rate based on venue interval.

    Args:
        rate: Funding rate (per period)
        source_venue: Venue name

    Returns:
        Annualized rate
    """
    venue_config = get_venue_interval(source_venue)
    return rate * venue_config.annualization_factor

def get_conversion_factor(
    source_venue: str,
    target_interval: NormalizationInterval,
) -> float:
    """
    Get conversion factor from venue to target interval.

    Args:
        source_venue: Source venue name
        target_interval: Target interval

    Returns:
        Conversion factor
    """
    venue_config = get_venue_interval(source_venue)
    return venue_config.conversion_factor_to(target_interval)

def normalize_funding_rates(
    df: pd.DataFrame,
    target_interval: Union[str, NormalizationInterval] = '8h',
    timestamp_col: str = 'timestamp',
    funding_col: str = 'funding_rate',
    venue_col: str = 'venue',
    symbol_col: str = 'symbol',
) -> pd.DataFrame:
    """
    Normalize funding rates in a DataFrame to a target interval.

    This function handles multiple venues with different funding intervals,
    normalizing all to a common target interval for comparison.

    Args:
        df: DataFrame with funding rates
        target_interval: Target interval ('8h', '1h', etc.) or NormalizationInterval
        timestamp_col: Timestamp column name
        funding_col: Funding rate column name
        venue_col: Venue column name
        symbol_col: Symbol column name

    Returns:
        DataFrame with normalized funding rates
    """
    if isinstance(target_interval, str):
        interval_map = {
            '8h': NormalizationInterval.EIGHT_HOUR,
            '1h': NormalizationInterval.HOURLY,
            '4h': NormalizationInterval.FOUR_HOUR,
            '1d': NormalizationInterval.DAILY,
            'daily': NormalizationInterval.DAILY,
            'hourly': NormalizationInterval.HOURLY,
        }
        target = interval_map.get(target_interval.lower(), NormalizationInterval.EIGHT_HOUR)
    else:
        target = target_interval

    if df.empty:
        return df.copy()

    normalizer = FundingNormalizer(default_target=target)
    result_frames = []

    # Process each venue separately
    for venue in df[venue_col].unique():
        venue_df = df[df[venue_col] == venue].copy()

        try:
            result = normalizer.normalize_dataframe(
                venue_df,
                venue=venue,
                target_interval=target,
                timestamp_col=timestamp_col,
                funding_col=funding_col,
                symbol_col=symbol_col,
            )
            result_frames.append(result.df)
        except Exception as e:
            logger.warning(f"Failed to normalize {venue}: {e}")
            result_frames.append(venue_df)

    if not result_frames:
        return df.copy()

    return pd.concat(result_frames, ignore_index=True)

def annualize_funding(
    df: pd.DataFrame,
    funding_col: str = 'funding_rate',
    venue_col: str = 'venue',
) -> pd.DataFrame:
    """
    Annualize funding rates in a DataFrame based on venue intervals.

    Args:
        df: DataFrame with funding rates
        funding_col: Funding rate column name
        venue_col: Venue column name

    Returns:
        DataFrame with added 'funding_rate_annualized' column
    """
    if df.empty:
        return df.copy()

    result = df.copy()
    result['funding_rate_annualized'] = 0.0

    for venue in df[venue_col].unique():
        mask = result[venue_col] == venue
        venue_config = get_venue_interval(venue)
        result.loc[mask, 'funding_rate_annualized'] = (
            result.loc[mask, funding_col] * venue_config.annualization_factor
        )

    return result

__all__ = [
    # Enums
    'NormalizationInterval',
    'AlignmentStrategy',
    'AggregationMethod',
    'VenueIntervalType',
    # Dataclasses
    'VenueInterval',
    'NormalizedRate',
    'AlignedRatePair',
    'NormalizationResult',
    'AlignmentResult',
    # Configuration
    'VENUE_INTERVALS',
    'VENUE_SETTLEMENT_TIMES',
    # Classes
    'FundingNormalizer',
    # Functions
    'get_venue_interval',
    'normalize_funding_rate',
    'annualize_funding_rate',
    'get_conversion_factor',
    'normalize_funding_rates', # DataFrame version
    'annualize_funding', # DataFrame version
]
