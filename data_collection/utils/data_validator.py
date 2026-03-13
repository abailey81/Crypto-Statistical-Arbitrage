"""
Data Validation Engine for Crypto Market Data

validated data framework for multi-venue crypto data with
comprehensive quality scoring, anomaly detection, and cross-venue consistency
checks optimized for statistical arbitrage applications.

===============================================================================
OVERVIEW
===============================================================================

Data quality directly impacts strategy performance. This module provides:
    - Schema validation (columns, types, ranges)
    - Completeness analysis (missing data detection)
    - Accuracy checks (outliers, anomalies, relationships)
    - Consistency validation (cross-venue, time series)
    - Quality scoring with trading-specific thresholds
    - Backtest reliability assessment

===============================================================================
VALIDATION STANDARDS
===============================================================================

Funding Rates:
    =================== ==================== ================
    Metric Threshold Impact
    =================== ==================== ================
    Missing Data <5% Signal quality
    Outliers (5 sigma) <1% False signals
    Cross-Venue Corr >0.95 Arb validity
    Time Gaps <8 hours Backtest bias
    =================== ==================== ================

OHLCV Data:
    =================== ==================== ================
    Metric Threshold Impact
    =================== ==================== ================
    High >= Low 100% Data integrity
    High >= Open, Close 100% Data integrity
    Volume >= 0 100% Volume analysis
    Price Gaps <10% jumps Regime detection
    =================== ==================== ================

Open Interest:
    =================== ==================== ================
    Metric Threshold Impact
    =================== ==================== ================
    OI >= 0 100% Data integrity
    Large OI Changes Flag >50%/day Risk events
    OI vs Volume Correlated Manipulation
    =================== ==================== ================

===============================================================================
QUALITY SCORING
===============================================================================

Score Components (weighted):
    - Completeness: 35% (missing data, coverage)
    - Accuracy: 25% (outliers, relationships)
    - Consistency: 25% (cross-venue, temporal)
    - Timeliness: 15% (data freshness)

Quality Levels:
    - EXCELLENT (95-100): Validated, full confidence
    - GOOD (85-94): Usable with minor caveats
    - ACCEPTABLE (70-84): Usable with caution
    - POOR (50-69): Significant issues, reduced confidence
    - CRITICAL (25-49): Major problems, limited use
    - UNUSABLE (0-24): Do not use

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

Signal Quality:
    - Quality score as confidence multiplier
    - Anomaly flags for risk management
    - Cross-venue validation for arbitrage

Backtest Reliability:
    - Survivorship bias detection
    - Gap impact assessment
    - Look-ahead bias prevention

Position Sizing:
    - Reduce size with lower quality
    - Increase size with high confidence

Version: 3.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Set

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# ENUMS
# =============================================================================

class ValidationSeverity(Enum):
    """
    Validation issue severity classification.

    Determines how issues affect data usability.
    """
    CRITICAL = 'critical' # Data unusable, blocks processing
    ERROR = 'error' # Significant issues, reduces confidence
    WARNING = 'warning' # Minor issues, proceed with caution
    INFO = 'info' # Informational, no action needed

    @property
    def score_impact(self) -> float:
        """Impact on quality score (0-1, higher = worse)."""
        return {
            ValidationSeverity.CRITICAL: 1.0,
            ValidationSeverity.ERROR: 0.5,
            ValidationSeverity.WARNING: 0.2,
            ValidationSeverity.INFO: 0.05,
        }.get(self, 0.0)

    @property
    def blocks_usage(self) -> bool:
        """Check if severity blocks data usage."""
        return self in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]

    @property
    def requires_action(self) -> bool:
        """Check if immediate action is required."""
        return self in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]

    @property
    def color_code(self) -> str:
        """Color code for visualization."""
        return {
            ValidationSeverity.CRITICAL: 'red',
            ValidationSeverity.ERROR: 'orange',
            ValidationSeverity.WARNING: 'yellow',
            ValidationSeverity.INFO: 'blue',
        }.get(self, 'gray')

class DataQualityLevel(Enum):
    """
    Overall data quality classification.

    Used for decision-making on data usability.
    """
    EXCELLENT = 6
    GOOD = 5
    ACCEPTABLE = 4
    POOR = 3
    CRITICAL = 2
    UNUSABLE = 1

    @classmethod
    def from_score(cls, score: float) -> 'DataQualityLevel':
        """Classify from quality score (0-100)."""
        if score >= 95:
            return cls.EXCELLENT
        elif score >= 85:
            return cls.GOOD
        elif score >= 70:
            return cls.ACCEPTABLE
        elif score >= 50:
            return cls.POOR
        elif score >= 25:
            return cls.CRITICAL
        return cls.UNUSABLE

    @property
    def is_usable(self) -> bool:
        """Check if data is usable for analysis."""
        return self.value >= 4

    @property
    def is_backtest_reliable(self) -> bool:
        """Check if data is reliable for backtesting."""
        return self.value >= 5

    @property
    def is_production_ready(self) -> bool:
        """Check if data is ready for production trading."""
        return self.value >= 5

    @property
    def confidence_multiplier(self) -> float:
        """Confidence multiplier for signal weighting."""
        return {
            DataQualityLevel.EXCELLENT: 1.0,
            DataQualityLevel.GOOD: 0.9,
            DataQualityLevel.ACCEPTABLE: 0.7,
            DataQualityLevel.POOR: 0.4,
            DataQualityLevel.CRITICAL: 0.2,
            DataQualityLevel.UNUSABLE: 0.0,
        }.get(self, 0.0)

    @property
    def position_size_multiplier(self) -> float:
        """Multiplier for position sizing based on quality."""
        return {
            DataQualityLevel.EXCELLENT: 1.0,
            DataQualityLevel.GOOD: 0.8,
            DataQualityLevel.ACCEPTABLE: 0.5,
            DataQualityLevel.POOR: 0.2,
            DataQualityLevel.CRITICAL: 0.0,
            DataQualityLevel.UNUSABLE: 0.0,
        }.get(self, 0.0)

    @property
    def description(self) -> str:
        """Human-readable description."""
        return {
            DataQualityLevel.EXCELLENT: "Validated with full confidence",
            DataQualityLevel.GOOD: "Suitable for trading with minor caveats",
            DataQualityLevel.ACCEPTABLE: "Usable with caution, reduced confidence",
            DataQualityLevel.POOR: "Significant issues, limited applications",
            DataQualityLevel.CRITICAL: "Major problems, research use only",
            DataQualityLevel.UNUSABLE: "Do not use - data integrity compromised",
        }.get(self, "Unknown quality level")

class ValidationCheckType(Enum):
    """
    Type of validation check performed.
    """
    SCHEMA = 'schema' # Column existence and types
    COMPLETENESS = 'completeness' # Missing data
    RANGE = 'range' # Value range validation
    RELATIONSHIP = 'relationship' # OHLCV relationships
    TEMPORAL = 'temporal' # Time series consistency
    CONSISTENCY = 'consistency' # Cross-venue consistency
    UNIQUENESS = 'uniqueness' # Duplicate detection
    OUTLIER = 'outlier' # Statistical outliers
    FRESHNESS = 'freshness' # Data timeliness

    @property
    def weight(self) -> float:
        """Weight in overall quality score."""
        return {
            ValidationCheckType.SCHEMA: 0.15,
            ValidationCheckType.COMPLETENESS: 0.20,
            ValidationCheckType.RANGE: 0.10,
            ValidationCheckType.RELATIONSHIP: 0.15,
            ValidationCheckType.TEMPORAL: 0.10,
            ValidationCheckType.CONSISTENCY: 0.10,
            ValidationCheckType.UNIQUENESS: 0.05,
            ValidationCheckType.OUTLIER: 0.10,
            ValidationCheckType.FRESHNESS: 0.05,
        }.get(self, 0.10)

class DataType(Enum):
    """
    Type of market data being validated.

    Different data types have different validation rules.

    IMPORTANT: Keep synchronized with base_collector.py and data_cleaner.py DataType enums.
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
        """Required columns for this data type."""
        return {
            DataType.FUNDING_RATES: ['timestamp', 'symbol', 'funding_rate'],
            DataType.OHLCV: ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'],
            DataType.OPEN_INTEREST: ['timestamp', 'symbol', 'open_interest'],
            DataType.LIQUIDATIONS: ['timestamp', 'symbol', 'side', 'size'],
            DataType.ORDERBOOK: ['timestamp', 'symbol', 'bids', 'asks'],
            DataType.TRADES: ['timestamp', 'symbol', 'price', 'size', 'side'],
            DataType.OPTIONS: ['timestamp', 'symbol', 'strike', 'expiry', 'option_type'],
            DataType.DEX_POOLS: ['timestamp', 'pool_address', 'token0', 'token1', 'tvl_usd'],
            DataType.SOCIAL_SENTIMENT: ['timestamp', 'token', 'sentiment_score'],
        }.get(self, ['timestamp'])

    @property
    def numeric_columns(self) -> List[str]:
        """Expected numeric columns for this data type."""
        return {
            DataType.FUNDING_RATES: ['funding_rate', 'mark_price', 'index_price'],
            DataType.OHLCV: ['open', 'high', 'low', 'close', 'volume'],
            DataType.OPEN_INTEREST: ['open_interest', 'mark_price'],
            DataType.LIQUIDATIONS: ['size', 'price'],
            DataType.TRADES: ['price', 'size'],
            DataType.OPTIONS: ['strike', 'premium', 'delta', 'gamma', 'iv'],
            DataType.DEX_POOLS: ['tvl_usd', 'volume_24h', 'fee_tier'],
            DataType.SOCIAL_SENTIMENT: ['sentiment_score', 'volume'],
        }.get(self, [])

    @property
    def expected_frequency(self) -> str:
        """Expected data frequency."""
        return {
            DataType.FUNDING_RATES: '8H',
            DataType.OHLCV: '1H',
            DataType.OPEN_INTEREST: '1H',
            DataType.LIQUIDATIONS: '1T', # Tick
            DataType.ORDERBOOK: '1T',
            DataType.TRADES: '1T',
            DataType.OPTIONS: '1H',
            DataType.DEX_POOLS: '1H',
            DataType.SOCIAL_SENTIMENT: '1H',
        }.get(self, '1H')

class OutlierMethod(Enum):
    """
    Method for outlier detection.
    """
    ZSCORE = 'zscore' # Standard deviations from mean
    IQR = 'iqr' # Interquartile range method
    MAD = 'mad' # Median absolute deviation
    ISOLATION_FOREST = 'isolation_forest' # ML-based
    PERCENTILE = 'percentile' # Fixed percentile thresholds

    @property
    def description(self) -> str:
        """Method description."""
        return {
            OutlierMethod.ZSCORE: "Flag values > N standard deviations from mean",
            OutlierMethod.IQR: "Flag values outside 1.5x IQR from quartiles",
            OutlierMethod.MAD: "Flag values > N MAD from median (reliable)",
            OutlierMethod.ISOLATION_FOREST: "ML-based anomaly detection",
            OutlierMethod.PERCENTILE: "Flag values outside percentile bounds",
        }.get(self, "Unknown method")

    @property
    def is_robust(self) -> bool:
        """Check if method is reliable to existing outliers."""
        return self in [OutlierMethod.IQR, OutlierMethod.MAD, OutlierMethod.PERCENTILE]

class GapSeverity(Enum):
    """
    Time series gap severity classification.
    """
    MINOR = 'minor' # <1 expected period
    MODERATE = 'moderate' # 1-3 expected periods
    MAJOR = 'major' # 3-10 expected periods
    CRITICAL = 'critical' # >10 expected periods

    @classmethod
    def from_periods(cls, missing_periods: int) -> 'GapSeverity':
        """Classify from number of missing periods."""
        if missing_periods <= 0:
            return cls.MINOR
        elif missing_periods <= 3:
            return cls.MODERATE
        elif missing_periods <= 10:
            return cls.MAJOR
        return cls.CRITICAL

    @property
    def score_penalty(self) -> float:
        """Quality score penalty (0-1)."""
        return {
            GapSeverity.MINOR: 0.01,
            GapSeverity.MODERATE: 0.05,
            GapSeverity.MAJOR: 0.15,
            GapSeverity.CRITICAL: 0.30,
        }.get(self, 0.10)

    @property
    def fill_recommendation(self) -> str:
        """Recommended gap handling."""
        return {
            GapSeverity.MINOR: "forward_fill",
            GapSeverity.MODERATE: "interpolate_with_flag",
            GapSeverity.MAJOR: "exclude_or_interpolate",
            GapSeverity.CRITICAL: "exclude_period",
        }.get(self, "investigate")

    @property
    def backtest_impact(self) -> str:
        """Impact on backtest reliability."""
        return {
            GapSeverity.MINOR: "negligible",
            GapSeverity.MODERATE: "minor",
            GapSeverity.MAJOR: "significant",
            GapSeverity.CRITICAL: "severe",
        }.get(self, "unknown")

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class ValidationIssue:
    """
    Single validation issue with full context.
    """
    check_type: ValidationCheckType
    severity: ValidationSeverity
    message: str
    column: Optional[str] = None
    affected_rows: int = 0
    affected_pct: float = 0.0
    sample_values: Optional[List[Any]] = None
    recommendation: str = ""

    @property
    def is_blocking(self) -> bool:
        """Check if issue blocks data usage."""
        return self.severity.blocks_usage

    @property
    def score_impact(self) -> float:
        """Impact on quality score."""
        return self.severity.score_impact * (self.affected_pct / 100)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'check_type': self.check_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'column': self.column,
            'affected_rows': self.affected_rows,
            'affected_pct': round(self.affected_pct, 2),
            'sample_values': self.sample_values[:5] if self.sample_values else None,
            'recommendation': self.recommendation,
            'is_blocking': self.is_blocking,
        }

@dataclass
class GapInfo:
    """
    Detailed information about a data gap.
    """
    start: datetime
    end: datetime
    expected_periods: int
    actual_periods: int = 0
    symbol: Optional[str] = None
    venue: Optional[str] = None

    @property
    def missing_periods(self) -> int:
        """Number of missing periods."""
        return max(0, self.expected_periods - self.actual_periods)

    @property
    def duration(self) -> timedelta:
        """Gap duration."""
        return self.end - self.start

    @property
    def duration_hours(self) -> float:
        """Gap duration in hours."""
        return self.duration.total_seconds() / 3600

    @property
    def severity(self) -> GapSeverity:
        """Gap severity classification."""
        return GapSeverity.from_periods(self.missing_periods)

    @property
    def fill_recommendation(self) -> str:
        """Recommended handling."""
        return self.severity.fill_recommendation

    @property
    def can_interpolate(self) -> bool:
        """Check if interpolation is acceptable."""
        return self.severity in [GapSeverity.MINOR, GapSeverity.MODERATE]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'start': self.start.isoformat() if isinstance(self.start, datetime) else str(self.start),
            'end': self.end.isoformat() if isinstance(self.end, datetime) else str(self.end),
            'duration_hours': round(self.duration_hours, 2),
            'expected_periods': self.expected_periods,
            'actual_periods': self.actual_periods,
            'missing_periods': self.missing_periods,
            'severity': self.severity.value,
            'recommendation': self.fill_recommendation,
            'symbol': self.symbol,
            'venue': self.venue,
        }

@dataclass
class ValidationResult:
    """
    Comprehensive validation result with scoring.
    """
    data_type: DataType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_records: int = 0
    valid_records: int = 0

    # Issues by severity
    issues: List[ValidationIssue] = field(default_factory=list)

    # Component scores (0-100)
    completeness_score: float = 100.0
    accuracy_score: float = 100.0
    consistency_score: float = 100.0
    timeliness_score: float = 100.0

    # Gaps
    gaps: List[GapInfo] = field(default_factory=list)

    # Metadata
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    symbols_count: int = 0
    venues_count: int = 0

    @property
    def invalid_records(self) -> int:
        """Number of invalid records."""
        return self.total_records - self.valid_records

    @property
    def validity_pct(self) -> float:
        """Percentage of valid records."""
        if self.total_records == 0:
            return 0.0
        return (self.valid_records / self.total_records) * 100

    @property
    def overall_score(self) -> float:
        """Calculate overall quality score (0-100)."""
        weights = {
            'completeness': 0.35,
            'accuracy': 0.25,
            'consistency': 0.25,
            'timeliness': 0.15,
        }

        score = (
            self.completeness_score * weights['completeness'] +
            self.accuracy_score * weights['accuracy'] +
            self.consistency_score * weights['consistency'] +
            self.timeliness_score * weights['timeliness']
        )

        # Apply issue penalties
        for issue in self.issues:
            score -= issue.score_impact * 100

        return max(0, min(100, score))

    @property
    def quality_level(self) -> DataQualityLevel:
        """Overall quality classification."""
        return DataQualityLevel.from_score(self.overall_score)

    @property
    def is_valid(self) -> bool:
        """Check if data passes validation."""
        return self.quality_level.is_usable

    @property
    def is_backtest_reliable(self) -> bool:
        """Check if reliable for backtesting."""
        return self.quality_level.is_backtest_reliable

    @property
    def confidence_multiplier(self) -> float:
        """Confidence multiplier for signals."""
        return self.quality_level.confidence_multiplier

    @property
    def critical_issues(self) -> List[ValidationIssue]:
        """List of critical issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.CRITICAL]

    @property
    def error_issues(self) -> List[ValidationIssue]:
        """List of error issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warning_issues(self) -> List[ValidationIssue]:
        """List of warning issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def has_blocking_issues(self) -> bool:
        """Check if any blocking issues exist."""
        return len(self.critical_issues) > 0 or len(self.error_issues) > 0

    @property
    def critical_gap_count(self) -> int:
        """Number of critical gaps."""
        return sum(1 for g in self.gaps if g.severity == GapSeverity.CRITICAL)

    @property
    def total_gap_hours(self) -> float:
        """Total hours of gaps."""
        return sum(g.duration_hours for g in self.gaps)

    @property
    def recommendations(self) -> List[str]:
        """Generate recommendations based on issues."""
        recs = []

        if self.completeness_score < 95:
            recs.append(f"Completeness at {self.completeness_score:.1f}%. Consider gap filling or sourcing additional data.")

        if self.accuracy_score < 95:
            recs.append(f"Accuracy at {self.accuracy_score:.1f}%. Review outlier handling and data validation.")

        if self.consistency_score < 95:
            recs.append(f"Consistency at {self.consistency_score:.1f}%. Cross-venue alignment may be needed.")

        if self.critical_gap_count > 0:
            recs.append(f"{self.critical_gap_count} critical gaps detected. Consider excluding affected periods.")

        for issue in self.critical_issues[:3]:
            recs.append(f"CRITICAL: {issue.message}")

        return recs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'data_type': self.data_type.value,
            'timestamp': self.timestamp.isoformat(),
            'total_records': self.total_records,
            'valid_records': self.valid_records,
            'validity_pct': round(self.validity_pct, 2),
            'overall_score': round(self.overall_score, 2),
            'quality_level': self.quality_level.name,
            'is_valid': self.is_valid,
            'is_backtest_reliable': self.is_backtest_reliable,
            'confidence_multiplier': round(self.confidence_multiplier, 2),
            'completeness_score': round(self.completeness_score, 2),
            'accuracy_score': round(self.accuracy_score, 2),
            'consistency_score': round(self.consistency_score, 2),
            'timeliness_score': round(self.timeliness_score, 2),
            'issue_count': len(self.issues),
            'critical_issues': len(self.critical_issues),
            'error_issues': len(self.error_issues),
            'warning_issues': len(self.warning_issues),
            'gap_count': len(self.gaps),
            'critical_gaps': self.critical_gap_count,
            'total_gap_hours': round(self.total_gap_hours, 2),
            'symbols_count': self.symbols_count,
            'venues_count': self.venues_count,
            'recommendations': self.recommendations,
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Data Validation Report",
            f"**Generated**: {self.timestamp.isoformat()}",
            f"**Data Type**: {self.data_type.value}",
            "",
            "## Quality Score",
            f"**Overall**: {self.quality_level.name} ({self.overall_score:.1f}/100)",
            "",
            "| Component | Score |",
            "|-----------|-------|",
            f"| Completeness | {self.completeness_score:.1f} |",
            f"| Accuracy | {self.accuracy_score:.1f} |",
            f"| Consistency | {self.consistency_score:.1f} |",
            f"| Timeliness | {self.timeliness_score:.1f} |",
            "",
            "## Summary",
            f"- **Total Records**: {self.total_records:,}",
            f"- **Valid Records**: {self.valid_records:,} ({self.validity_pct:.1f}%)",
            f"- **Symbols**: {self.symbols_count}",
            f"- **Venues**: {self.venues_count}",
            f"- **Gaps**: {len(self.gaps)} ({self.total_gap_hours:.1f} hours)",
        ]

        if self.critical_issues:
            lines.extend(["", "## Critical Issues"])
            for issue in self.critical_issues[:5]:
                lines.append(f"- {issue.message}")

        if self.recommendations:
            lines.extend(["", "## Recommendations"])
            for rec in self.recommendations[:5]:
                lines.append(f"- {rec}")

        return "\n".join(lines)

@dataclass
class CrossValidationResult:
    """
    Cross-venue consistency validation result.
    """
    venue_a: str
    venue_b: str
    column: str
    matched_records: int = 0
    correlation: float = 0.0
    mean_deviation: float = 0.0
    max_deviation: float = 0.0
    std_deviation: float = 0.0
    rmse: float = 0.0

    @property
    def mean_deviation_bps(self) -> float:
        """Mean deviation in basis points."""
        return self.mean_deviation * 10000

    @property
    def max_deviation_bps(self) -> float:
        """Max deviation in basis points."""
        return self.max_deviation * 10000

    @property
    def is_consistent(self) -> bool:
        """Check if venues are consistent (correlation > 0.95)."""
        return self.correlation > 0.95

    @property
    def consistency_level(self) -> str:
        """Consistency level classification."""
        if self.correlation >= 0.99:
            return 'perfect'
        elif self.correlation >= 0.95:
            return 'high'
        elif self.correlation >= 0.90:
            return 'moderate'
        elif self.correlation >= 0.80:
            return 'low'
        return 'inconsistent'

    @property
    def arbitrage_viable(self) -> bool:
        """Check if cross-venue arbitrage is viable."""
        return self.is_consistent and self.max_deviation_bps > 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'venue_a': self.venue_a,
            'venue_b': self.venue_b,
            'column': self.column,
            'matched_records': self.matched_records,
            'correlation': round(self.correlation, 4),
            'consistency_level': self.consistency_level,
            'mean_deviation_bps': round(self.mean_deviation_bps, 2),
            'max_deviation_bps': round(self.max_deviation_bps, 2),
            'rmse': round(self.rmse, 6),
            'is_consistent': self.is_consistent,
            'arbitrage_viable': self.arbitrage_viable,
        }

# =============================================================================
# DATA VALIDATOR CLASS
# =============================================================================

class DataValidator:
    """
    Comprehensive data validation engine for crypto market data.

    Provides:
        - Schema validation
        - Completeness checks
        - Outlier detection
        - Cross-venue consistency
        - Quality scoring

    Example:
        >>> validator = DataValidator(max_missing_pct=5.0, outlier_threshold=5.0)
        >>> result = validator.validate_funding_rates(df)
        >>> print(f"Quality: {result.quality_level.name} ({result.overall_score:.1f})")
    """

    # Default value ranges
    FUNDING_RATE_RANGE = (-0.01, 0.01) # +-1% per period
    PRICE_MIN = 0.0001 # Minimum valid price
    VOLUME_MIN = 0 # Minimum volume
    OI_MIN = 0 # Minimum OI

    def __init__(
        self,
        max_missing_pct: float = 5.0,
        outlier_threshold: float = 5.0,
        outlier_method: OutlierMethod = OutlierMethod.ZSCORE,
        min_correlation: float = 0.95,
        max_staleness_hours: float = 24.0,
    ):
        """
        Initialize validator.

        Args:
            max_missing_pct: Maximum acceptable missing data percentage
            outlier_threshold: Threshold for outlier detection (e.g., 5 sigma)
            outlier_method: Method for outlier detection
            min_correlation: Minimum correlation for cross-venue consistency
            max_staleness_hours: Maximum acceptable data staleness
        """
        self.max_missing_pct = max_missing_pct
        self.outlier_threshold = outlier_threshold
        self.outlier_method = outlier_method
        self.min_correlation = min_correlation
        self.max_staleness_hours = max_staleness_hours

    def validate_funding_rates(
        self,
        df: pd.DataFrame,
        rate_col: str = 'funding_rate',
        timestamp_col: str = 'timestamp',
        symbol_col: str = 'symbol',
        venue_col: Optional[str] = 'venue',
    ) -> ValidationResult:
        """
        Validate funding rate data.

        Args:
            df: DataFrame with funding rates
            rate_col: Column containing funding rates
            timestamp_col: Timestamp column
            symbol_col: Symbol column
            venue_col: Venue column (optional)

        Returns:
            ValidationResult with quality metrics
        """
        result = ValidationResult(data_type=DataType.FUNDING_RATES)

        if df.empty:
            result.issues.append(ValidationIssue(
                check_type=ValidationCheckType.SCHEMA,
                severity=ValidationSeverity.CRITICAL,
                message="Empty dataset",
                recommendation="Provide data for validation"
            ))
            return result

        result.total_records = len(df)
        df = df.copy()

        # Schema validation
        self._validate_schema(df, result, DataType.FUNDING_RATES)

        # Completeness
        self._check_completeness(df, result, [rate_col, timestamp_col, symbol_col])

        # Range validation for funding rates
        if rate_col in df.columns:
            self._validate_range(df, result, rate_col, self.FUNDING_RATE_RANGE)
            self._detect_outliers(df, result, rate_col)

        # Temporal consistency
        if timestamp_col in df.columns:
            self._check_temporal_consistency(df, result, timestamp_col, symbol_col, '8H')

        # Calculate scores
        self._calculate_scores(df, result, rate_col, timestamp_col)

        # Metadata
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            result.date_range_start = df[timestamp_col].min()
            result.date_range_end = df[timestamp_col].max()

        if symbol_col in df.columns:
            result.symbols_count = df[symbol_col].nunique()

        if venue_col and venue_col in df.columns:
            result.venues_count = df[venue_col].nunique()

        result.valid_records = result.total_records - sum(i.affected_rows for i in result.issues)

        logger.info(f"Funding validation: {result.quality_level.name} ({result.overall_score:.1f})")
        return result

    def validate_ohlcv(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        symbol_col: str = 'symbol',
        venue_col: Optional[str] = 'venue',
    ) -> ValidationResult:
        """
        Validate OHLCV price data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            ValidationResult with quality metrics
        """
        result = ValidationResult(data_type=DataType.OHLCV)

        if df.empty:
            result.issues.append(ValidationIssue(
                check_type=ValidationCheckType.SCHEMA,
                severity=ValidationSeverity.CRITICAL,
                message="Empty dataset",
            ))
            return result

        result.total_records = len(df)
        df = df.copy()

        # Schema validation
        self._validate_schema(df, result, DataType.OHLCV)

        # Completeness
        required = ['open', 'high', 'low', 'close', 'volume']
        self._check_completeness(df, result, required)

        # OHLCV relationships
        self._validate_ohlcv_relationships(df, result)

        # Price range validation
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                self._validate_range(df, result, col, (self.PRICE_MIN, None))

        if 'volume' in df.columns:
            self._validate_range(df, result, 'volume', (self.VOLUME_MIN, None))

        # Outlier detection on returns
        if 'close' in df.columns:
            df['return_pct'] = df['close'].pct_change() * 100
            extreme_returns = df['return_pct'].abs() > 20 # >20% moves
            if extreme_returns.any():
                result.issues.append(ValidationIssue(
                    check_type=ValidationCheckType.OUTLIER,
                    severity=ValidationSeverity.WARNING,
                    message=f"Extreme price moves (>20%) detected",
                    column='close',
                    affected_rows=int(extreme_returns.sum()),
                    affected_pct=extreme_returns.mean() * 100,
                    recommendation="Review for data errors vs legitimate volatility"
                ))

        # Temporal consistency
        if timestamp_col in df.columns:
            self._check_temporal_consistency(df, result, timestamp_col, symbol_col, '1H')

        # Calculate scores
        self._calculate_scores(df, result, 'close', timestamp_col)

        # Metadata
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            result.date_range_start = df[timestamp_col].min()
            result.date_range_end = df[timestamp_col].max()

        if symbol_col in df.columns:
            result.symbols_count = df[symbol_col].nunique()

        if venue_col and venue_col in df.columns:
            result.venues_count = df[venue_col].nunique()

        result.valid_records = result.total_records - sum(i.affected_rows for i in result.issues)

        logger.info(f"OHLCV validation: {result.quality_level.name} ({result.overall_score:.1f})")
        return result

    def cross_validate(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        venue_a: str,
        venue_b: str,
        value_col: str = 'funding_rate',
        timestamp_col: str = 'timestamp',
        symbol_col: str = 'symbol',
    ) -> CrossValidationResult:
        """
        Cross-validate data between two venues.

        Args:
            df_a: First venue data
            df_b: Second venue data
            venue_a: First venue name
            venue_b: Second venue name
            value_col: Column to compare

        Returns:
            CrossValidationResult with consistency metrics
        """
        result = CrossValidationResult(venue_a=venue_a, venue_b=venue_b, column=value_col)

        if df_a.empty or df_b.empty:
            return result

        # Merge on timestamp and symbol
        merged = pd.merge(
            df_a[[timestamp_col, symbol_col, value_col]].rename(columns={value_col: 'val_a'}),
            df_b[[timestamp_col, symbol_col, value_col]].rename(columns={value_col: 'val_b'}),
            on=[timestamp_col, symbol_col],
            how='inner'
        )

        if merged.empty:
            return result

        result.matched_records = len(merged)

        # Calculate metrics
        val_a = merged['val_a'].values
        val_b = merged['val_b'].values

        result.correlation = float(np.corrcoef(val_a, val_b)[0, 1])

        deviations = np.abs(val_a - val_b)
        result.mean_deviation = float(deviations.mean())
        result.max_deviation = float(deviations.max())
        result.std_deviation = float(deviations.std())
        result.rmse = float(np.sqrt(np.mean((val_a - val_b) ** 2)))

        logger.info(f"Cross-validation {venue_a} vs {venue_b}: correlation={result.correlation:.4f}")
        return result

    def cross_validate_venues(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        on: str = 'funding_rate',
        timestamp_col: str = 'timestamp',
        symbol_col: str = 'symbol',
    ) -> CrossValidationResult:
        """
        Cross-validate data between two venue DataFrames.

        Alias for cross_validate with simplified parameters.

        Args:
            df_a: First venue data
            df_b: Second venue data
            on: Column to compare
            timestamp_col: Timestamp column name
            symbol_col: Symbol column name

        Returns:
            CrossValidationResult with consistency metrics
        """
        # Extract venue names from data
        venue_a = df_a['venue'].iloc[0] if 'venue' in df_a.columns and len(df_a) > 0 else 'venue_a'
        venue_b = df_b['venue'].iloc[0] if 'venue' in df_b.columns and len(df_b) > 0 else 'venue_b'

        return self.cross_validate(
            df_a=df_a,
            df_b=df_b,
            venue_a=venue_a,
            venue_b=venue_b,
            value_col=on,
            timestamp_col=timestamp_col,
            symbol_col=symbol_col,
        )

    def _validate_schema(self, df: pd.DataFrame, result: ValidationResult, data_type: DataType) -> None:
        """Validate schema (required columns)."""
        required = data_type.required_columns
        missing = [col for col in required if col not in df.columns]

        if missing:
            result.issues.append(ValidationIssue(
                check_type=ValidationCheckType.SCHEMA,
                severity=ValidationSeverity.CRITICAL,
                message=f"Missing required columns: {missing}",
                recommendation=f"Add columns: {missing}"
            ))

    def _check_completeness(self, df: pd.DataFrame, result: ValidationResult, columns: List[str]) -> None:
        """Check for missing data."""
        for col in columns:
            if col not in df.columns:
                continue

            missing_count = df[col].isna().sum()
            missing_pct = missing_count / len(df) * 100

            if missing_pct > self.max_missing_pct:
                severity = ValidationSeverity.ERROR if missing_pct > 20 else ValidationSeverity.WARNING
                result.issues.append(ValidationIssue(
                    check_type=ValidationCheckType.COMPLETENESS,
                    severity=severity,
                    message=f"Missing data in '{col}': {missing_pct:.1f}%",
                    column=col,
                    affected_rows=int(missing_count),
                    affected_pct=missing_pct,
                    recommendation=f"Impute or source missing {col} data"
                ))
                result.completeness_score -= missing_pct * 2

    def _validate_range(self, df: pd.DataFrame, result: ValidationResult,
                        col: str, valid_range: Tuple[Optional[float], Optional[float]]) -> None:
        """Validate value range."""
        if col not in df.columns:
            return

        values = df[col].dropna()
        min_val, max_val = valid_range

        out_of_range = pd.Series(False, index=values.index)
        if min_val is not None:
            out_of_range |= values < min_val
        if max_val is not None:
            out_of_range |= values > max_val

        if out_of_range.any():
            count = out_of_range.sum()
            pct = count / len(values) * 100
            severity = ValidationSeverity.ERROR if pct > 5 else ValidationSeverity.WARNING

            result.issues.append(ValidationIssue(
                check_type=ValidationCheckType.RANGE,
                severity=severity,
                message=f"Values out of range in '{col}': {count} ({pct:.1f}%)",
                column=col,
                affected_rows=int(count),
                affected_pct=pct,
                sample_values=values[out_of_range].head(5).tolist(),
                recommendation=f"Review extreme values in {col}"
            ))

    def _validate_ohlcv_relationships(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate OHLCV relationships (high >= low, etc.)."""
        required = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required):
            return

        # High >= Low
        violation = df['high'] < df['low']
        if violation.any():
            result.issues.append(ValidationIssue(
                check_type=ValidationCheckType.RELATIONSHIP,
                severity=ValidationSeverity.CRITICAL,
                message=f"High < Low in {violation.sum()} records",
                affected_rows=int(violation.sum()),
                affected_pct=violation.mean() * 100,
                recommendation="Data integrity issue - high must be >= low"
            ))

        # High >= Open, Close
        for col in ['open', 'close']:
            violation = df['high'] < df[col]
            if violation.any():
                result.issues.append(ValidationIssue(
                    check_type=ValidationCheckType.RELATIONSHIP,
                    severity=ValidationSeverity.ERROR,
                    message=f"High < {col.capitalize()} in {violation.sum()} records",
                    affected_rows=int(violation.sum()),
                    affected_pct=violation.mean() * 100,
                ))

        # Low <= Open, Close
        for col in ['open', 'close']:
            violation = df['low'] > df[col]
            if violation.any():
                result.issues.append(ValidationIssue(
                    check_type=ValidationCheckType.RELATIONSHIP,
                    severity=ValidationSeverity.ERROR,
                    message=f"Low > {col.capitalize()} in {violation.sum()} records",
                    affected_rows=int(violation.sum()),
                    affected_pct=violation.mean() * 100,
                ))

    def _detect_outliers(self, df: pd.DataFrame, result: ValidationResult, col: str) -> None:
        """Detect statistical outliers."""
        if col not in df.columns:
            return

        values = df[col].dropna()
        if len(values) < 10:
            return

        if self.outlier_method == OutlierMethod.ZSCORE:
            z_scores = np.abs((values - values.mean()) / values.std())
            outliers = z_scores > self.outlier_threshold
        elif self.outlier_method == OutlierMethod.IQR:
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
        elif self.outlier_method == OutlierMethod.MAD:
            median = values.median()
            mad = np.abs(values - median).median()
            outliers = np.abs(values - median) / mad > self.outlier_threshold
        else:
            return

        if outliers.any():
            count = outliers.sum()
            pct = count / len(values) * 100
            severity = ValidationSeverity.WARNING if pct < 5 else ValidationSeverity.ERROR

            result.issues.append(ValidationIssue(
                check_type=ValidationCheckType.OUTLIER,
                severity=severity,
                message=f"Outliers in '{col}': {count} ({pct:.1f}%)",
                column=col,
                affected_rows=int(count),
                affected_pct=pct,
                recommendation=f"Review outliers using {self.outlier_method.value} method"
            ))
            result.accuracy_score -= pct * 2

    def _check_temporal_consistency(self, df: pd.DataFrame, result: ValidationResult,
                                    timestamp_col: str, symbol_col: str, expected_freq: str) -> None:
        """Check for temporal gaps."""
        if timestamp_col not in df.columns:
            return

        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Get expected timedelta
        freq_map = {'1H': timedelta(hours=1), '8H': timedelta(hours=8), '1D': timedelta(days=1)}
        expected_delta = freq_map.get(expected_freq, timedelta(hours=8))
        tolerance = expected_delta * 1.5

        # Check for gaps per symbol
        if symbol_col in df.columns:
            for symbol, group in df.groupby(symbol_col):
                timestamps = group[timestamp_col].sort_values()
                if len(timestamps) < 2:
                    continue

                diffs = timestamps.diff().dropna()
                gaps = diffs[diffs > tolerance]

                for idx, gap_duration in gaps.items():
                    gap_info = GapInfo(
                        start=timestamps.loc[timestamps.index[timestamps.index.get_loc(idx) - 1]],
                        end=timestamps.loc[idx],
                        expected_periods=int(gap_duration / expected_delta),
                        actual_periods=0,
                        symbol=symbol,
                    )
                    result.gaps.append(gap_info)

        if result.gaps:
            critical = sum(1 for g in result.gaps if g.severity == GapSeverity.CRITICAL)
            if critical > 0:
                result.issues.append(ValidationIssue(
                    check_type=ValidationCheckType.TEMPORAL,
                    severity=ValidationSeverity.ERROR,
                    message=f"{critical} critical gaps detected (>10 periods)",
                    affected_rows=critical,
                    recommendation="Consider excluding periods with critical gaps"
                ))
                result.completeness_score -= critical * 5

    def _calculate_scores(self, df: pd.DataFrame, result: ValidationResult,
                          value_col: str, timestamp_col: str) -> None:
        """Calculate component quality scores."""
        # Completeness already calculated in checks
        result.completeness_score = max(0, min(100, result.completeness_score))

        # Accuracy - based on outliers
        result.accuracy_score = max(0, min(100, result.accuracy_score))

        # Consistency - would need cross-venue data
        if result.venues_count <= 1:
            result.consistency_score = 100.0 # Single venue, no comparison needed

        # Timeliness
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            latest = df[timestamp_col].max()
            if pd.notna(latest):
                now = datetime.now(timezone.utc)
                staleness_hours = (now - latest.to_pydatetime().replace(tzinfo=timezone.utc)).total_seconds() / 3600

                if staleness_hours <= 1:
                    result.timeliness_score = 100.0
                elif staleness_hours <= 8:
                    result.timeliness_score = 90.0
                elif staleness_hours <= 24:
                    result.timeliness_score = 70.0
                elif staleness_hours <= 72:
                    result.timeliness_score = 50.0
                else:
                    result.timeliness_score = 25.0

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_dataframe(
    df: pd.DataFrame,
    data_type: str = 'funding_rates',
    **kwargs
) -> ValidationResult:
    """
    Convenience function to validate a DataFrame.

    Args:
        df: DataFrame to validate
        data_type: Type of data ('funding_rates', 'ohlcv', etc.)
        **kwargs: Additional arguments for DataValidator

    Returns:
        ValidationResult
    """
    validator = DataValidator(**kwargs)

    if data_type == 'funding_rates':
        return validator.validate_funding_rates(df)
    elif data_type == 'ohlcv':
        return validator.validate_ohlcv(df)
    else:
        return validator.validate_funding_rates(df)

def quick_quality_check(df: pd.DataFrame, value_col: str = 'funding_rate') -> Dict[str, Any]:
    """
    Quick quality check returning key metrics.

    Args:
        df: DataFrame to check
        value_col: Primary value column

    Returns:
        Dictionary with quality metrics
    """
    if df.empty:
        return {'quality': 'UNUSABLE', 'score': 0, 'records': 0}

    missing_pct = df[value_col].isna().mean() * 100 if value_col in df.columns else 100

    outlier_pct = 0
    if value_col in df.columns:
        values = df[value_col].dropna()
        if len(values) > 10 and values.std() > 0:
            z_scores = np.abs((values - values.mean()) / values.std())
            outlier_pct = (z_scores > 5).mean() * 100

    score = 100 - (missing_pct * 2) - (outlier_pct * 5)
    score = max(0, min(100, score))

    level = DataQualityLevel.from_score(score)

    return {
        'quality': level.name,
        'score': round(score, 1),
        'records': len(df),
        'missing_pct': round(missing_pct, 2),
        'outlier_pct': round(outlier_pct, 2),
        'is_usable': level.is_usable,
        'confidence': level.confidence_multiplier,
    }
