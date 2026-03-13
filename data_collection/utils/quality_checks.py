"""
Quality Checks Module for Crypto Statistical Arbitrage System.

This module provides comprehensive data quality assessment specifically designed
for statistical arbitrage trading systems. It implements professional-quality
quality metrics, pipeline health monitoring, and production readiness evaluation.

==============================================================================
QUALITY METRICS REFERENCE
==============================================================================

Data Quality Dimensions (Weighted Scoring):
+---------------------+--------+----------------------------------------+
| Dimension | Weight | Description |
+---------------------+--------+----------------------------------------+
| Completeness | 0.30 | Coverage of expected data points |
| Accuracy | 0.25 | Value correctness and range validity |
| Consistency | 0.20 | Cross-venue/cross-time agreement |
| Timeliness | 0.15 | Data freshness and update frequency |
| Uniqueness | 0.10 | Duplicate detection and handling |
+---------------------+--------+----------------------------------------+

Quality Thresholds by Use Case:
+---------------------+----------+----------+----------+----------+
| Use Case | Minimum | Target | Excellent| Critical |
+---------------------+----------+----------+----------+----------+
| Live Trading | 95% | 98% | 99%+ | <90% |
| Backtesting | 85% | 92% | 95%+ | <80% |
| Research | 75% | 85% | 90%+ | <70% |
| Data Exploration | 60% | 75% | 85%+ | <50% |
+---------------------+----------+----------+----------+----------+

Pipeline Health Indicators:
+---------------------+----------------------------------------+----------+
| Metric | Description | Target |
+---------------------+----------------------------------------+----------+
| Collection Rate | Records per minute | >100/min |
| Error Rate | Failed requests / total requests | <1% |
| Latency P50 | Median collection latency | <500ms |
| Latency P99 | 99th percentile latency | <2000ms |
| Gap Frequency | Gaps per 1000 expected periods | <5 |
| Staleness | Time since last successful update | <5min |
+---------------------+----------------------------------------+----------+

==============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
==============================================================================

1. SIGNAL QUALITY ASSESSMENT
   - Evaluate funding rate signal reliability
   - Cross-venue consistency for spread signals
   - Historical signal accuracy correlation with data quality

2. BACKTEST RELIABILITY
   - Quantify survivorship bias risk
   - Gap impact on strategy returns
   - Look-ahead bias detection in timestamps

3. PRODUCTION MONITORING
   - Real-time quality degradation alerts
   - Venue-specific health dashboards
   - SLA compliance tracking

4. RISK-ADJUSTED SIZING
   - Quality-weighted position sizing
   - Confidence intervals for expected returns
   - Dynamic exposure based on data reliability

==============================================================================
USAGE EXAMPLES
==============================================================================

Basic Quality Check:
    >>> checker = QualityChecker()
    >>> result = checker.comprehensive_check(df, DataCategory.FUNDING_RATES)
    >>> print(f"Quality Score: {result.overall_score:.1f}%")
    >>> print(f"Validated: {result.is_production_ready}")

Pipeline Health:
    >>> health = checker.assess_pipeline_health(metrics_dict)
    >>> print(f"Pipeline Status: {health.status.name}")
    >>> for alert in health.alerts:
    ... print(f" [{alert.severity}] {alert.message}")

Cross-Venue Consistency:
    >>> consistency = checker.check_cross_venue_consistency(
    ... venue_data_dict, symbol='BTC-PERP'
    ... )
    >>> print(f"Consistency Score: {consistency.score:.1f}%")
    >>> print(f"Arbitrage Viable: {consistency.is_arbitrage_viable}")

Version: 2.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Set, Union
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings

# =============================================================================
# ENUMS
# =============================================================================

class DataCategory(Enum):
    """
    Data categories with specific quality requirements and validation rules.

    Each category has different tolerance thresholds and validation strategies
    based on its characteristics and trading system requirements.
    """

    FUNDING_RATES = "funding_rates"
    OHLCV = "ohlcv"
    OPEN_INTEREST = "open_interest"
    LIQUIDATIONS = "liquidations"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    OPTIONS = "options"
    DEX_POOLS = "dex_pools"
    ON_CHAIN = "on_chain"
    SOCIAL_SENTIMENT = "social_sentiment"

    @property
    def completeness_weight(self) -> float:
        """Weight for completeness in overall score calculation."""
        weights = {
            DataCategory.FUNDING_RATES: 0.35, # High - gaps affect carry calculations
            DataCategory.OHLCV: 0.30, # Standard
            DataCategory.OPEN_INTEREST: 0.25, # Lower - can interpolate
            DataCategory.LIQUIDATIONS: 0.20, # Event-based, gaps expected
            DataCategory.ORDERBOOK: 0.25, # Snapshot-based
            DataCategory.TRADES: 0.20, # High volume, sampling OK
            DataCategory.OPTIONS: 0.30, # Important for vol arb
            DataCategory.DEX_POOLS: 0.25, # Block-based collection
            DataCategory.ON_CHAIN: 0.20, # Block confirmations vary
            DataCategory.SOCIAL_SENTIMENT: 0.15, # Supplementary data
        }
        return weights.get(self, 0.30)

    @property
    def accuracy_weight(self) -> float:
        """Weight for accuracy in overall score calculation."""
        weights = {
            DataCategory.FUNDING_RATES: 0.30, # Critical for carry trades
            DataCategory.OHLCV: 0.25, # Standard
            DataCategory.OPEN_INTEREST: 0.25, # Position sizing
            DataCategory.LIQUIDATIONS: 0.30, # Volume accuracy important
            DataCategory.ORDERBOOK: 0.35, # Execution depends on accuracy
            DataCategory.TRADES: 0.25, # Standard
            DataCategory.OPTIONS: 0.35, # Greeks very sensitive
            DataCategory.DEX_POOLS: 0.30, # TVL/reserves accuracy
            DataCategory.ON_CHAIN: 0.25, # Standard
            DataCategory.SOCIAL_SENTIMENT: 0.20, # Noisy by nature
        }
        return weights.get(self, 0.25)

    @property
    def consistency_weight(self) -> float:
        """Weight for consistency in overall score calculation."""
        weights = {
            DataCategory.FUNDING_RATES: 0.20, # Cross-venue spreads
            DataCategory.OHLCV: 0.25, # OHLC relationships
            DataCategory.OPEN_INTEREST: 0.20, # Standard
            DataCategory.LIQUIDATIONS: 0.15, # Event timing varies
            DataCategory.ORDERBOOK: 0.20, # Snapshot timing
            DataCategory.TRADES: 0.20, # Standard
            DataCategory.OPTIONS: 0.20, # Chain consistency
            DataCategory.DEX_POOLS: 0.20, # Block consistency
            DataCategory.ON_CHAIN: 0.30, # Chain reorgs
            DataCategory.SOCIAL_SENTIMENT: 0.25, # Source consistency
        }
        return weights.get(self, 0.20)

    @property
    def timeliness_weight(self) -> float:
        """Weight for timeliness in overall score calculation."""
        weights = {
            DataCategory.FUNDING_RATES: 0.10, # 8h updates OK
            DataCategory.OHLCV: 0.10, # Standard
            DataCategory.OPEN_INTEREST: 0.15, # Position changes
            DataCategory.LIQUIDATIONS: 0.20, # Time-sensitive events
            DataCategory.ORDERBOOK: 0.15, # Execution timing
            DataCategory.TRADES: 0.25, # Very time-sensitive
            DataCategory.OPTIONS: 0.10, # Less time-critical
            DataCategory.DEX_POOLS: 0.15, # Block times
            DataCategory.ON_CHAIN: 0.15, # Confirmation times
            DataCategory.SOCIAL_SENTIMENT: 0.25, # Trend detection
        }
        return weights.get(self, 0.15)

    @property
    def uniqueness_weight(self) -> float:
        """Weight for uniqueness in overall score calculation."""
        return 1.0 - (self.completeness_weight + self.accuracy_weight +
                      self.consistency_weight + self.timeliness_weight)

    @property
    def expected_frequency_minutes(self) -> Optional[int]:
        """Expected data frequency in minutes for gap detection."""
        frequencies = {
            DataCategory.FUNDING_RATES: 480, # 8 hours
            DataCategory.OHLCV: 1, # 1 minute
            DataCategory.OPEN_INTEREST: 60, # Hourly
            DataCategory.LIQUIDATIONS: None, # Event-based
            DataCategory.ORDERBOOK: 1, # Snapshot per minute
            DataCategory.TRADES: None, # Continuous
            DataCategory.OPTIONS: 60, # Hourly
            DataCategory.DEX_POOLS: 15, # ~1 block
            DataCategory.ON_CHAIN: 15, # Block time
            DataCategory.SOCIAL_SENTIMENT: 60, # Hourly aggregates
        }
        return frequencies.get(self)

    @property
    def outlier_threshold_sigma(self) -> float:
        """Standard deviations for outlier detection."""
        thresholds = {
            DataCategory.FUNDING_RATES: 4.0, # Rates can spike
            DataCategory.OHLCV: 5.0, # Prices volatile
            DataCategory.OPEN_INTEREST: 4.0, # Can change rapidly
            DataCategory.LIQUIDATIONS: 5.0, # Cascades happen
            DataCategory.ORDERBOOK: 3.0, # More stable
            DataCategory.TRADES: 5.0, # High volatility
            DataCategory.OPTIONS: 4.0, # Vol spikes
            DataCategory.DEX_POOLS: 4.0, # Pool imbalances
            DataCategory.ON_CHAIN: 5.0, # Gas spikes
            DataCategory.SOCIAL_SENTIMENT: 3.0, # More stable
        }
        return thresholds.get(self, 4.0)

    @property
    def required_columns(self) -> List[str]:
        """Required columns for this data category."""
        columns = {
            DataCategory.FUNDING_RATES: ['timestamp', 'symbol', 'venue', 'funding_rate'],
            DataCategory.OHLCV: ['timestamp', 'symbol', 'venue', 'open', 'high', 'low', 'close', 'volume'],
            DataCategory.OPEN_INTEREST: ['timestamp', 'symbol', 'venue', 'open_interest'],
            DataCategory.LIQUIDATIONS: ['timestamp', 'symbol', 'venue', 'side', 'quantity', 'price'],
            DataCategory.ORDERBOOK: ['timestamp', 'symbol', 'venue', 'bids', 'asks'],
            DataCategory.TRADES: ['timestamp', 'symbol', 'venue', 'price', 'quantity', 'side'],
            DataCategory.OPTIONS: ['timestamp', 'symbol', 'venue', 'strike', 'expiry', 'type', 'price'],
            DataCategory.DEX_POOLS: ['timestamp', 'pool_address', 'chain', 'token0_reserve', 'token1_reserve'],
            DataCategory.ON_CHAIN: ['timestamp', 'chain', 'block_number', 'tx_hash'],
            DataCategory.SOCIAL_SENTIMENT: ['timestamp', 'symbol', 'source', 'sentiment_score'],
        }
        return columns.get(self, ['timestamp'])

    @property
    def numeric_columns(self) -> List[str]:
        """Numeric columns requiring range validation."""
        columns = {
            DataCategory.FUNDING_RATES: ['funding_rate'],
            DataCategory.OHLCV: ['open', 'high', 'low', 'close', 'volume'],
            DataCategory.OPEN_INTEREST: ['open_interest'],
            DataCategory.LIQUIDATIONS: ['quantity', 'price'],
            DataCategory.OPTIONS: ['strike', 'price', 'iv', 'delta', 'gamma'],
            DataCategory.DEX_POOLS: ['token0_reserve', 'token1_reserve', 'tvl_usd'],
            DataCategory.SOCIAL_SENTIMENT: ['sentiment_score', 'volume'],
        }
        return columns.get(self, [])

class QualityLevel(Enum):
    """
    Quality level classification with trading implications.

    Each level corresponds to specific use case restrictions and
    position sizing recommendations.
    """

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    MARGINAL = "marginal"
    POOR = "poor"
    UNUSABLE = "unusable"

    @property
    def score_range(self) -> Tuple[float, float]:
        """Score range (min, max) for this quality level."""
        ranges = {
            QualityLevel.EXCELLENT: (95.0, 100.0),
            QualityLevel.GOOD: (85.0, 95.0),
            QualityLevel.ACCEPTABLE: (75.0, 85.0),
            QualityLevel.MARGINAL: (60.0, 75.0),
            QualityLevel.POOR: (40.0, 60.0),
            QualityLevel.UNUSABLE: (0.0, 40.0),
        }
        return ranges.get(self, (0.0, 100.0))

    @classmethod
    def from_score(cls, score: float) -> 'QualityLevel':
        """Determine quality level from numeric score."""
        if score >= 95.0:
            return cls.EXCELLENT
        elif score >= 85.0:
            return cls.GOOD
        elif score >= 75.0:
            return cls.ACCEPTABLE
        elif score >= 60.0:
            return cls.MARGINAL
        elif score >= 40.0:
            return cls.POOR
        else:
            return cls.UNUSABLE

    @property
    def is_production_ready(self) -> bool:
        """Whether data at this level is suitable for live trading."""
        return self in {QualityLevel.EXCELLENT, QualityLevel.GOOD}

    @property
    def is_backtest_reliable(self) -> bool:
        """Whether data at this level is suitable for backtesting."""
        return self in {QualityLevel.EXCELLENT, QualityLevel.GOOD, QualityLevel.ACCEPTABLE}

    @property
    def is_research_usable(self) -> bool:
        """Whether data at this level is suitable for research."""
        return self not in {QualityLevel.UNUSABLE}

    @property
    def position_size_multiplier(self) -> float:
        """Recommended position size multiplier based on data quality."""
        multipliers = {
            QualityLevel.EXCELLENT: 1.0,
            QualityLevel.GOOD: 0.8,
            QualityLevel.ACCEPTABLE: 0.5,
            QualityLevel.MARGINAL: 0.25,
            QualityLevel.POOR: 0.1,
            QualityLevel.UNUSABLE: 0.0,
        }
        return multipliers.get(self, 0.0)

    @property
    def confidence_interval_multiplier(self) -> float:
        """Multiplier for widening confidence intervals."""
        multipliers = {
            QualityLevel.EXCELLENT: 1.0,
            QualityLevel.GOOD: 1.2,
            QualityLevel.ACCEPTABLE: 1.5,
            QualityLevel.MARGINAL: 2.0,
            QualityLevel.POOR: 3.0,
            QualityLevel.UNUSABLE: float('inf'),
        }
        return multipliers.get(self, float('inf'))

    @property
    def description(self) -> str:
        """Human-readable description of quality level."""
        descriptions = {
            QualityLevel.EXCELLENT: "Validated, high-confidence signals",
            QualityLevel.GOOD: "Validated with minor caveats",
            QualityLevel.ACCEPTABLE: "Suitable for backtesting, not live trading",
            QualityLevel.MARGINAL: "Research use only, significant gaps",
            QualityLevel.POOR: "Limited use, major quality issues",
            QualityLevel.UNUSABLE: "Do not use for any trading decisions",
        }
        return descriptions.get(self, "Unknown")

    @property
    def color_code(self) -> str:
        """Color code for dashboard visualization."""
        colors = {
            QualityLevel.EXCELLENT: "#00FF00", # Green
            QualityLevel.GOOD: "#90EE90", # Light green
            QualityLevel.ACCEPTABLE: "#FFFF00", # Yellow
            QualityLevel.MARGINAL: "#FFA500", # Orange
            QualityLevel.POOR: "#FF6347", # Tomato
            QualityLevel.UNUSABLE: "#FF0000", # Red
        }
        return colors.get(self, "#FFFFFF")

class AlertSeverity(Enum):
    """
    Alert severity levels for quality monitoring.

    Determines notification urgency and escalation procedures.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def requires_immediate_action(self) -> bool:
        """Whether alert requires immediate human intervention."""
        return self in {AlertSeverity.CRITICAL, AlertSeverity.HIGH}

    @property
    def escalation_minutes(self) -> int:
        """Minutes before escalation if unacknowledged."""
        minutes = {
            AlertSeverity.CRITICAL: 5,
            AlertSeverity.HIGH: 15,
            AlertSeverity.MEDIUM: 60,
            AlertSeverity.LOW: 240,
            AlertSeverity.INFO: 1440,
        }
        return minutes.get(self, 60)

    @property
    def notification_channels(self) -> List[str]:
        """Channels to notify for this severity."""
        channels = {
            AlertSeverity.CRITICAL: ["pagerduty", "slack", "email", "sms"],
            AlertSeverity.HIGH: ["slack", "email"],
            AlertSeverity.MEDIUM: ["slack"],
            AlertSeverity.LOW: ["dashboard"],
            AlertSeverity.INFO: ["log"],
        }
        return channels.get(self, ["log"])

    @property
    def icon(self) -> str:
        """Icon for dashboard display."""
        icons = {
            AlertSeverity.CRITICAL: "ðŸš¨",
            AlertSeverity.HIGH: "âš ï¸",
            AlertSeverity.MEDIUM: "âš¡",
            AlertSeverity.LOW: "â„¹ï¸",
            AlertSeverity.INFO: "ðŸ“",
        }
        return icons.get(self, "â€¢")

class PipelineStatus(Enum):
    """
    Overall pipeline health status.

    Aggregated status for dashboard display and alerting.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"

    @property
    def is_operational(self) -> bool:
        """Whether pipeline is operational (may be degraded)."""
        return self in {PipelineStatus.HEALTHY, PipelineStatus.DEGRADED, PipelineStatus.WARNING}

    @property
    def allows_trading(self) -> bool:
        """Whether trading should continue at this status."""
        return self in {PipelineStatus.HEALTHY, PipelineStatus.DEGRADED}

    @property
    def color_code(self) -> str:
        """Color code for dashboard."""
        colors = {
            PipelineStatus.HEALTHY: "#00FF00",
            PipelineStatus.DEGRADED: "#FFFF00",
            PipelineStatus.WARNING: "#FFA500",
            PipelineStatus.CRITICAL: "#FF0000",
            PipelineStatus.OFFLINE: "#808080",
        }
        return colors.get(self, "#FFFFFF")

    @property
    def description(self) -> str:
        """Human-readable status description."""
        descriptions = {
            PipelineStatus.HEALTHY: "All systems operational",
            PipelineStatus.DEGRADED: "Minor issues, trading continues with reduced capacity",
            PipelineStatus.WARNING: "Significant issues, review recommended",
            PipelineStatus.CRITICAL: "Major failure, trading should halt",
            PipelineStatus.OFFLINE: "System offline, no data collection",
        }
        return descriptions.get(self, "Unknown")

class CheckType(Enum):
    """
    Types of quality checks performed.

    Each check type contributes to a specific quality dimension.
    """

    SCHEMA = "schema"
    COMPLETENESS = "completeness"
    RANGE = "range"
    OUTLIER = "outlier"
    TEMPORAL = "temporal"
    CONSISTENCY = "consistency"
    UNIQUENESS = "uniqueness"
    FRESHNESS = "freshness"
    CROSS_VENUE = "cross_venue"
    RELATIONSHIP = "relationship"

    @property
    def dimension(self) -> str:
        """Quality dimension this check contributes to."""
        dimensions = {
            CheckType.SCHEMA: "accuracy",
            CheckType.COMPLETENESS: "completeness",
            CheckType.RANGE: "accuracy",
            CheckType.OUTLIER: "accuracy",
            CheckType.TEMPORAL: "consistency",
            CheckType.CONSISTENCY: "consistency",
            CheckType.UNIQUENESS: "uniqueness",
            CheckType.FRESHNESS: "timeliness",
            CheckType.CROSS_VENUE: "consistency",
            CheckType.RELATIONSHIP: "accuracy",
        }
        return dimensions.get(self, "accuracy")

    @property
    def is_blocking(self) -> bool:
        """Whether failure blocks data usage."""
        return self in {CheckType.SCHEMA, CheckType.RANGE}

    @property
    def description(self) -> str:
        """Description of check purpose."""
        descriptions = {
            CheckType.SCHEMA: "Validates required columns and data types",
            CheckType.COMPLETENESS: "Checks for missing data and gaps",
            CheckType.RANGE: "Validates values within expected ranges",
            CheckType.OUTLIER: "Detects statistical outliers",
            CheckType.TEMPORAL: "Validates timestamp ordering and gaps",
            CheckType.CONSISTENCY: "Checks internal data consistency",
            CheckType.UNIQUENESS: "Detects duplicate records",
            CheckType.FRESHNESS: "Validates data recency",
            CheckType.CROSS_VENUE: "Compares data across venues",
            CheckType.RELATIONSHIP: "Validates inter-column relationships",
        }
        return descriptions.get(self, "Unknown check")

class IssueCategory(Enum):
    """
    Categories of quality issues for classification and handling.
    """

    MISSING_DATA = "missing_data"
    INVALID_VALUE = "invalid_value"
    OUTLIER = "outlier"
    DUPLICATE = "duplicate"
    STALE_DATA = "stale_data"
    SCHEMA_MISMATCH = "schema_mismatch"
    TEMPORAL_ISSUE = "temporal_issue"
    CROSS_VENUE_DIVERGENCE = "cross_venue_divergence"
    RELATIONSHIP_VIOLATION = "relationship_violation"

    @property
    def default_severity(self) -> AlertSeverity:
        """Default severity for issues in this category."""
        severities = {
            IssueCategory.MISSING_DATA: AlertSeverity.MEDIUM,
            IssueCategory.INVALID_VALUE: AlertSeverity.HIGH,
            IssueCategory.OUTLIER: AlertSeverity.LOW,
            IssueCategory.DUPLICATE: AlertSeverity.LOW,
            IssueCategory.STALE_DATA: AlertSeverity.MEDIUM,
            IssueCategory.SCHEMA_MISMATCH: AlertSeverity.CRITICAL,
            IssueCategory.TEMPORAL_ISSUE: AlertSeverity.MEDIUM,
            IssueCategory.CROSS_VENUE_DIVERGENCE: AlertSeverity.HIGH,
            IssueCategory.RELATIONSHIP_VIOLATION: AlertSeverity.HIGH,
        }
        return severities.get(self, AlertSeverity.MEDIUM)

    @property
    def remediation_action(self) -> str:
        """Recommended remediation action."""
        actions = {
            IssueCategory.MISSING_DATA: "Fill with interpolation or exclude period",
            IssueCategory.INVALID_VALUE: "Replace with NaN and investigate source",
            IssueCategory.OUTLIER: "Review for validity, cap if confirmed outlier",
            IssueCategory.DUPLICATE: "Keep first occurrence, remove duplicates",
            IssueCategory.STALE_DATA: "Trigger collection refresh",
            IssueCategory.SCHEMA_MISMATCH: "Update schema or fix collection",
            IssueCategory.TEMPORAL_ISSUE: "Reorder data or fill gaps",
            IssueCategory.CROSS_VENUE_DIVERGENCE: "Flag for manual review",
            IssueCategory.RELATIONSHIP_VIOLATION: "Correct values or exclude record",
        }
        return actions.get(self, "Manual review required")

    @property
    def affects_signal_quality(self) -> bool:
        """Whether this issue type affects trading signal quality."""
        return self in {
            IssueCategory.MISSING_DATA,
            IssueCategory.INVALID_VALUE,
            IssueCategory.STALE_DATA,
            IssueCategory.CROSS_VENUE_DIVERGENCE,
        }

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class QualityIssue:
    """
    Represents a single quality issue detected during validation.

    Contains full context for investigation and remediation.
    """

    check_type: CheckType
    category: IssueCategory
    severity: AlertSeverity
    message: str
    affected_rows: int = 0
    affected_columns: List[str] = field(default_factory=list)
    sample_values: List[Any] = field(default_factory=list)
    timestamp_range: Optional[Tuple[datetime, datetime]] = None
    venue: Optional[str] = None
    symbol: Optional[str] = None
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_blocking(self) -> bool:
        """Whether this issue blocks data usage."""
        return self.severity in {AlertSeverity.CRITICAL, AlertSeverity.HIGH}

    @property
    def requires_immediate_action(self) -> bool:
        """Whether immediate action is required."""
        return self.severity.requires_immediate_action

    @property
    def score_impact(self) -> float:
        """Impact on quality score (0-100 scale)."""
        base_impact = {
            AlertSeverity.CRITICAL: 25.0,
            AlertSeverity.HIGH: 15.0,
            AlertSeverity.MEDIUM: 8.0,
            AlertSeverity.LOW: 3.0,
            AlertSeverity.INFO: 0.5,
        }
        return base_impact.get(self.severity, 5.0)

    @property
    def affects_trading(self) -> bool:
        """Whether this issue affects trading decisions."""
        return self.category.affects_signal_quality

    @property
    def remediation(self) -> str:
        """Recommended remediation action."""
        return self.category.remediation_action

    @property
    def summary(self) -> str:
        """Brief summary for logging."""
        location = []
        if self.venue:
            location.append(f"venue={self.venue}")
        if self.symbol:
            location.append(f"symbol={self.symbol}")
        loc_str = f" ({', '.join(location)})" if location else ""
        return f"[{self.severity.name}] {self.category.name}: {self.message}{loc_str}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'check_type': self.check_type.value,
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'affected_rows': self.affected_rows,
            'affected_columns': self.affected_columns,
            'sample_values': [str(v) for v in self.sample_values[:5]],
            'timestamp_range': [
                self.timestamp_range[0].isoformat() if self.timestamp_range else None,
                self.timestamp_range[1].isoformat() if self.timestamp_range else None,
            ],
            'venue': self.venue,
            'symbol': self.symbol,
            'detected_at': self.detected_at.isoformat(),
            'is_blocking': self.is_blocking,
            'score_impact': self.score_impact,
            'remediation': self.remediation,
        }

@dataclass
class QualityMetrics:
    """
    Comprehensive quality metrics for a dataset.

    Implements a 9-dimension quality framework based on industry best practices:
    1. Completeness (20%) - Data presence and coverage
    2. Accuracy (20%) - Match to authoritative sources
    3. Uniqueness (10%) - No unintended duplicates
    4. Consistency (15%) - Cross-venue agreement
    5. Validity (10%) - Values in acceptable ranges
    6. Timeliness (10%) - Data freshness
    7. Age (5%) - Temporal relevance
    8. Granularity (5%) - Appropriate detail level
    9. Structure (5%) - Schema compliance
    """

    total_rows: int
    valid_rows: int
    null_count: int
    duplicate_count: int
    outlier_count: int
    gap_count: int
    gap_total_periods: int
    freshness_seconds: float
    completeness_pct: float
    accuracy_pct: float
    consistency_pct: float
    timeliness_pct: float
    uniqueness_pct: float
    overall_score: float
    data_category: DataCategory
    start_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    venues: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    calculation_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Additional 9-dimension framework metrics
    validity_pct: float = 100.0 # Values within valid ranges
    age_pct: float = 100.0 # Temporal relevance (older data may be less relevant)
    granularity_pct: float = 100.0 # Appropriate detail level
    structure_pct: float = 100.0 # Schema compliance score
    cross_venue_correlation: float = 0.0 # Cross-venue consistency correlation

    @property
    def quality_level(self) -> QualityLevel:
        """Quality level classification."""
        return QualityLevel.from_score(self.overall_score)

    @property
    def is_production_ready(self) -> bool:
        """Whether data is suitable for production."""
        return self.quality_level.is_production_ready

    @property
    def is_backtest_reliable(self) -> bool:
        """Whether data is suitable for backtesting."""
        return self.quality_level.is_backtest_reliable

    @property
    def position_size_multiplier(self) -> float:
        """Recommended position size multiplier."""
        return self.quality_level.position_size_multiplier

    @property
    def invalid_rows(self) -> int:
        """Number of invalid rows."""
        return self.total_rows - self.valid_rows

    @property
    def invalid_pct(self) -> float:
        """Percentage of invalid rows."""
        if self.total_rows == 0:
            return 0.0
        return (self.invalid_rows / self.total_rows) * 100

    @property
    def null_pct(self) -> float:
        """Percentage of null values."""
        if self.total_rows == 0:
            return 0.0
        return (self.null_count / self.total_rows) * 100

    @property
    def duplicate_pct(self) -> float:
        """Percentage of duplicate records."""
        if self.total_rows == 0:
            return 0.0
        return (self.duplicate_count / self.total_rows) * 100

    @property
    def outlier_pct(self) -> float:
        """Percentage of outlier values."""
        if self.total_rows == 0:
            return 0.0
        return (self.outlier_count / self.total_rows) * 100

    @property
    def data_coverage_hours(self) -> Optional[float]:
        """Total hours of data coverage."""
        if self.start_timestamp and self.end_timestamp:
            delta = self.end_timestamp - self.start_timestamp
            return delta.total_seconds() / 3600
        return None

    @property
    def gap_coverage_pct(self) -> float:
        """Percentage of expected periods with data."""
        if not self.data_category.expected_frequency_minutes:
            return 100.0
        if not self.data_coverage_hours:
            return 0.0
        expected_periods = (self.data_coverage_hours * 60) / self.data_category.expected_frequency_minutes
        if expected_periods == 0:
            return 100.0
        actual_periods = expected_periods - self.gap_total_periods
        return max(0.0, (actual_periods / expected_periods) * 100)

    @property
    def freshness_status(self) -> str:
        """Human-readable freshness status."""
        if self.freshness_seconds < 60:
            return "Real-time"
        elif self.freshness_seconds < 300:
            return "Fresh"
        elif self.freshness_seconds < 900:
            return "Recent"
        elif self.freshness_seconds < 3600:
            return "Aging"
        else:
            return "Stale"

    @property
    def dimension_breakdown(self) -> Dict[str, float]:
        """Quality dimension scores breakdown (5 core dimensions)."""
        return {
            'completeness': self.completeness_pct,
            'accuracy': self.accuracy_pct,
            'consistency': self.consistency_pct,
            'timeliness': self.timeliness_pct,
            'uniqueness': self.uniqueness_pct,
        }

    @property
    def nine_dimension_breakdown(self) -> Dict[str, float]:
        """
        Full 9-dimension quality framework breakdown.

        Industry-standard dimensions with weights:
        1. Completeness (20%) - Data presence and coverage
        2. Accuracy (20%) - Match to authoritative sources
        3. Uniqueness (10%) - No unintended duplicates
        4. Consistency (15%) - Cross-venue agreement
        5. Validity (10%) - Values in acceptable ranges
        6. Timeliness (10%) - Data freshness
        7. Age (5%) - Temporal relevance
        8. Granularity (5%) - Appropriate detail level
        9. Structure (5%) - Schema compliance
        """
        return {
            'completeness': self.completeness_pct,
            'accuracy': self.accuracy_pct,
            'uniqueness': self.uniqueness_pct,
            'consistency': self.consistency_pct,
            'validity': self.validity_pct,
            'timeliness': self.timeliness_pct,
            'age': self.age_pct,
            'granularity': self.granularity_pct,
            'structure': self.structure_pct,
        }

    @property
    def nine_dimension_score(self) -> float:
        """
        Calculate weighted 9-dimension quality score.

        Weights based on importance for crypto trading:
        - Completeness: 20% - Gaps affect backtest accuracy
        - Accuracy: 20% - Wrong values cause signal errors
        - Uniqueness: 10% - Duplicates inflate metrics
        - Consistency: 15% - Cross-venue discrepancies indicate errors
        - Validity: 10% - Out-of-range values are outliers
        - Timeliness: 10% - Fresh data for live trading
        - Age: 5% - Historical relevance
        - Granularity: 5% - Appropriate frequency
        - Structure: 5% - Schema compliance
        """
        weights = {
            'completeness': 0.20,
            'accuracy': 0.20,
            'uniqueness': 0.10,
            'consistency': 0.15,
            'validity': 0.10,
            'timeliness': 0.10,
            'age': 0.05,
            'granularity': 0.05,
            'structure': 0.05,
        }
        dimensions = self.nine_dimension_breakdown
        return sum(weights[k] * dimensions[k] for k in weights)

    @property
    def weakest_dimension(self) -> Tuple[str, float]:
        """Identify weakest quality dimension."""
        breakdown = self.dimension_breakdown
        weakest = min(breakdown.items(), key=lambda x: x[1])
        return weakest

    @property
    def improvement_priority(self) -> List[str]:
        """Ordered list of dimensions to improve."""
        breakdown = self.dimension_breakdown
        sorted_dims = sorted(breakdown.items(), key=lambda x: x[1])
        return [dim for dim, score in sorted_dims if score < 90]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_rows': self.total_rows,
            'valid_rows': self.valid_rows,
            'invalid_rows': self.invalid_rows,
            'null_count': self.null_count,
            'duplicate_count': self.duplicate_count,
            'outlier_count': self.outlier_count,
            'gap_count': self.gap_count,
            'gap_total_periods': self.gap_total_periods,
            'freshness_seconds': self.freshness_seconds,
            'freshness_status': self.freshness_status,
            'completeness_pct': round(self.completeness_pct, 2),
            'accuracy_pct': round(self.accuracy_pct, 2),
            'consistency_pct': round(self.consistency_pct, 2),
            'timeliness_pct': round(self.timeliness_pct, 2),
            'uniqueness_pct': round(self.uniqueness_pct, 2),
            'validity_pct': round(self.validity_pct, 2),
            'age_pct': round(self.age_pct, 2),
            'granularity_pct': round(self.granularity_pct, 2),
            'structure_pct': round(self.structure_pct, 2),
            'overall_score': round(self.overall_score, 2),
            'nine_dimension_score': round(self.nine_dimension_score, 2),
            'quality_level': self.quality_level.value,
            'is_production_ready': self.is_production_ready,
            'is_backtest_reliable': self.is_backtest_reliable,
            'position_size_multiplier': self.position_size_multiplier,
            'data_category': self.data_category.value,
            'start_timestamp': self.start_timestamp.isoformat() if self.start_timestamp else None,
            'end_timestamp': self.end_timestamp.isoformat() if self.end_timestamp else None,
            'data_coverage_hours': self.data_coverage_hours,
            'gap_coverage_pct': round(self.gap_coverage_pct, 2),
            'venues': self.venues,
            'symbols': self.symbols,
            'cross_venue_correlation': round(self.cross_venue_correlation, 4),
            'weakest_dimension': self.weakest_dimension[0],
            'improvement_priority': self.improvement_priority,
            'nine_dimension_breakdown': {k: round(v, 2) for k, v in self.nine_dimension_breakdown.items()},
            'calculation_timestamp': self.calculation_timestamp.isoformat(),
        }

@dataclass
class PipelineAlert:
    """
    Alert generated by pipeline health monitoring.

    Contains context for notification routing and escalation.
    """

    severity: AlertSeverity
    message: str
    metric_name: str
    metric_value: float
    threshold_value: float
    venue: Optional[str] = None
    symbol: Optional[str] = None
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    @property
    def is_active(self) -> bool:
        """Whether alert is still active (unacknowledged)."""
        return not self.acknowledged

    @property
    def deviation_pct(self) -> float:
        """Percentage deviation from threshold."""
        if self.threshold_value == 0:
            return float('inf')
        return abs((self.metric_value - self.threshold_value) / self.threshold_value) * 100

    @property
    def notification_channels(self) -> List[str]:
        """Channels to notify."""
        return self.severity.notification_channels

    @property
    def escalation_deadline(self) -> datetime:
        """Deadline for escalation if unacknowledged."""
        return self.triggered_at + timedelta(minutes=self.severity.escalation_minutes)

    @property
    def should_escalate(self) -> bool:
        """Whether alert should be escalated."""
        return self.is_active and datetime.now(timezone.utc) > self.escalation_deadline

    def acknowledge(self, acknowledged_by: str) -> None:
        """Acknowledge the alert."""
        self.acknowledged = True
        self.acknowledged_by = acknowledged_by
        self.acknowledged_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'severity': self.severity.value,
            'message': self.message,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold_value': self.threshold_value,
            'deviation_pct': round(self.deviation_pct, 2),
            'venue': self.venue,
            'symbol': self.symbol,
            'triggered_at': self.triggered_at.isoformat(),
            'is_active': self.is_active,
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'notification_channels': self.notification_channels,
            'should_escalate': self.should_escalate,
        }

@dataclass
class PipelineHealth:
    """
    Comprehensive pipeline health assessment.

    Aggregates metrics and alerts for dashboard display.
    """

    status: PipelineStatus
    overall_score: float
    collection_rate_per_min: float
    error_rate_pct: float
    latency_p50_ms: float
    latency_p99_ms: float
    active_venues: int
    total_venues: int
    active_symbols: int
    stale_venue_count: int
    alerts: List[PipelineAlert] = field(default_factory=list)
    venue_health: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_successful_collection: Optional[datetime] = None
    assessment_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_operational(self) -> bool:
        """Whether pipeline is operational."""
        return self.status.is_operational

    @property
    def allows_trading(self) -> bool:
        """Whether trading should continue."""
        return self.status.allows_trading

    @property
    def venue_availability_pct(self) -> float:
        """Percentage of venues operational."""
        if self.total_venues == 0:
            return 0.0
        return (self.active_venues / self.total_venues) * 100

    @property
    def critical_alerts(self) -> List[PipelineAlert]:
        """Critical severity alerts."""
        return [a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]

    @property
    def high_alerts(self) -> List[PipelineAlert]:
        """High severity alerts."""
        return [a for a in self.alerts if a.severity == AlertSeverity.HIGH]

    @property
    def active_alerts(self) -> List[PipelineAlert]:
        """All unacknowledged alerts."""
        return [a for a in self.alerts if a.is_active]

    @property
    def escalation_required(self) -> bool:
        """Whether any alerts need escalation."""
        return any(a.should_escalate for a in self.alerts)

    @property
    def staleness_seconds(self) -> Optional[float]:
        """Seconds since last successful collection."""
        if self.last_successful_collection:
            delta = datetime.now(timezone.utc) - self.last_successful_collection
            return delta.total_seconds()
        return None

    @property
    def health_summary(self) -> str:
        """Brief health summary."""
        return (
            f"{self.status.name}: {self.active_venues}/{self.total_venues} venues, "
            f"{len(self.active_alerts)} active alerts, "
            f"error rate {self.error_rate_pct:.2f}%"
        )

    @property
    def degraded_venues(self) -> List[str]:
        """List of venues with degraded health."""
        return [
            venue for venue, health in self.venue_health.items()
            if health.get('status', 'healthy') != 'healthy'
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'status': self.status.value,
            'status_description': self.status.description,
            'allows_trading': self.allows_trading,
            'overall_score': round(self.overall_score, 2),
            'collection_rate_per_min': round(self.collection_rate_per_min, 2),
            'error_rate_pct': round(self.error_rate_pct, 4),
            'latency_p50_ms': round(self.latency_p50_ms, 2),
            'latency_p99_ms': round(self.latency_p99_ms, 2),
            'active_venues': self.active_venues,
            'total_venues': self.total_venues,
            'venue_availability_pct': round(self.venue_availability_pct, 2),
            'active_symbols': self.active_symbols,
            'stale_venue_count': self.stale_venue_count,
            'critical_alert_count': len(self.critical_alerts),
            'high_alert_count': len(self.high_alerts),
            'active_alert_count': len(self.active_alerts),
            'escalation_required': self.escalation_required,
            'alerts': [a.to_dict() for a in self.alerts],
            'venue_health': self.venue_health,
            'degraded_venues': self.degraded_venues,
            'staleness_seconds': self.staleness_seconds,
            'last_successful_collection': (
                self.last_successful_collection.isoformat()
                if self.last_successful_collection else None
            ),
            'health_summary': self.health_summary,
            'assessment_timestamp': self.assessment_timestamp.isoformat(),
        }

@dataclass
class CrossVenueConsistency:
    """
    Cross-venue data consistency analysis.

    Evaluates whether data from different venues is consistent
    enough for reliable arbitrage signal generation.
    """

    symbol: str
    venues: List[str]
    timestamp_range: Tuple[datetime, datetime]
    mean_value_by_venue: Dict[str, float]
    std_by_venue: Dict[str, float]
    correlation_matrix: Dict[str, Dict[str, float]]
    max_deviation_pct: float
    mean_deviation_pct: float
    timestamp_alignment_pct: float
    overlapping_periods: int
    consistency_score: float

    @property
    def is_consistent(self) -> bool:
        """Whether data is sufficiently consistent."""
        return self.consistency_score >= 85.0

    @property
    def is_arbitrage_viable(self) -> bool:
        """Whether data is reliable enough for arbitrage."""
        return self.consistency_score >= 90.0 and self.timestamp_alignment_pct >= 95.0

    @property
    def venue_count(self) -> int:
        """Number of venues in comparison."""
        return len(self.venues)

    @property
    def coverage_hours(self) -> float:
        """Hours of overlapping coverage."""
        delta = self.timestamp_range[1] - self.timestamp_range[0]
        return delta.total_seconds() / 3600

    @property
    def most_correlated_pair(self) -> Tuple[str, str, float]:
        """Venue pair with highest correlation."""
        max_corr = -1.0
        best_pair = (None, None)
        for v1, corrs in self.correlation_matrix.items():
            for v2, corr in corrs.items():
                if v1 != v2 and corr > max_corr:
                    max_corr = corr
                    best_pair = (v1, v2)
        return (best_pair[0], best_pair[1], max_corr)

    @property
    def least_correlated_pair(self) -> Tuple[str, str, float]:
        """Venue pair with lowest correlation."""
        min_corr = 2.0
        worst_pair = (None, None)
        for v1, corrs in self.correlation_matrix.items():
            for v2, corr in corrs.items():
                if v1 != v2 and corr < min_corr:
                    min_corr = corr
                    worst_pair = (v1, v2)
        return (worst_pair[0], worst_pair[1], min_corr)

    @property
    def outlier_venue(self) -> Optional[str]:
        """Venue that deviates most from others."""
        if len(self.venues) < 3:
            return None
        avg_values = list(self.mean_value_by_venue.values())
        overall_mean = np.mean(avg_values)
        overall_std = np.std(avg_values)
        if overall_std == 0:
            return None
        for venue, mean_val in self.mean_value_by_venue.items():
            z_score = abs((mean_val - overall_mean) / overall_std)
            if z_score > 2.0:
                return venue
        return None

    @property
    def recommendation(self) -> str:
        """Trading recommendation based on consistency."""
        if self.is_arbitrage_viable:
            return "Data quality supports arbitrage trading"
        elif self.is_consistent:
            return "Data acceptable for backtesting, verify live signals"
        elif self.consistency_score >= 70:
            return "Use with caution, increased position sizing risk"
        else:
            return "Data inconsistency too high, do not trade"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'venues': self.venues,
            'venue_count': self.venue_count,
            'timestamp_range': [
                self.timestamp_range[0].isoformat(),
                self.timestamp_range[1].isoformat(),
            ],
            'coverage_hours': round(self.coverage_hours, 2),
            'mean_value_by_venue': {k: round(v, 8) for k, v in self.mean_value_by_venue.items()},
            'std_by_venue': {k: round(v, 8) for k, v in self.std_by_venue.items()},
            'correlation_matrix': {
                k: {k2: round(v2, 4) for k2, v2 in v.items()}
                for k, v in self.correlation_matrix.items()
            },
            'max_deviation_pct': round(self.max_deviation_pct, 4),
            'mean_deviation_pct': round(self.mean_deviation_pct, 4),
            'timestamp_alignment_pct': round(self.timestamp_alignment_pct, 2),
            'overlapping_periods': self.overlapping_periods,
            'consistency_score': round(self.consistency_score, 2),
            'is_consistent': self.is_consistent,
            'is_arbitrage_viable': self.is_arbitrage_viable,
            'most_correlated_pair': self.most_correlated_pair,
            'least_correlated_pair': self.least_correlated_pair,
            'outlier_venue': self.outlier_venue,
            'recommendation': self.recommendation,
        }

@dataclass
class QualityCheckResult:
    """
    Comprehensive result from quality checking process.

    Contains all metrics, issues, and recommendations.
    """

    metrics: QualityMetrics
    issues: List[QualityIssue] = field(default_factory=list)
    cross_venue_consistency: Optional[CrossVenueConsistency] = None
    check_duration_ms: float = 0.0
    checks_performed: List[CheckType] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Overall quality score."""
        return self.metrics.overall_score

    @property
    def quality_level(self) -> QualityLevel:
        """Quality level classification."""
        return self.metrics.quality_level

    @property
    def is_valid(self) -> bool:
        """Whether data passes validation."""
        return not any(i.is_blocking for i in self.issues)

    @property
    def is_production_ready(self) -> bool:
        """Whether data passes validation."""
        return self.is_valid and self.metrics.is_production_ready

    @property
    def blocking_issues(self) -> List[QualityIssue]:
        """Issues that block data usage."""
        return [i for i in self.issues if i.is_blocking]

    @property
    def critical_issues(self) -> List[QualityIssue]:
        """Critical severity issues."""
        return [i for i in self.issues if i.severity == AlertSeverity.CRITICAL]

    @property
    def issue_count_by_severity(self) -> Dict[str, int]:
        """Issue counts grouped by severity."""
        counts = defaultdict(int)
        for issue in self.issues:
            counts[issue.severity.value] += 1
        return dict(counts)

    @property
    def issue_count_by_category(self) -> Dict[str, int]:
        """Issue counts grouped by category."""
        counts = defaultdict(int)
        for issue in self.issues:
            counts[issue.category.value] += 1
        return dict(counts)

    @property
    def total_affected_rows(self) -> int:
        """Total rows affected by any issue."""
        return sum(i.affected_rows for i in self.issues)

    @property
    def position_size_multiplier(self) -> float:
        """Recommended position size multiplier."""
        return self.metrics.position_size_multiplier

    @property
    def improvement_actions(self) -> List[str]:
        """Prioritized improvement actions."""
        actions = []
        for issue in sorted(self.issues, key=lambda x: x.score_impact, reverse=True)[:5]:
            actions.append(f"[{issue.severity.name}] {issue.remediation}")
        return actions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metrics': self.metrics.to_dict(),
            'issues': [i.to_dict() for i in self.issues],
            'cross_venue_consistency': (
                self.cross_venue_consistency.to_dict()
                if self.cross_venue_consistency else None
            ),
            'overall_score': round(self.overall_score, 2),
            'quality_level': self.quality_level.value,
            'is_valid': self.is_valid,
            'is_production_ready': self.is_production_ready,
            'blocking_issue_count': len(self.blocking_issues),
            'issue_count_by_severity': self.issue_count_by_severity,
            'issue_count_by_category': self.issue_count_by_category,
            'total_affected_rows': self.total_affected_rows,
            'position_size_multiplier': self.position_size_multiplier,
            'improvement_actions': self.improvement_actions,
            'check_duration_ms': round(self.check_duration_ms, 2),
            'checks_performed': [c.value for c in self.checks_performed],
        }

    def to_markdown(self) -> str:
        """Generate markdown quality report."""
        lines = [
            "# Data Quality Report",
            "",
            f"**Overall Score:** {self.overall_score:.1f}% ({self.quality_level.value})",
            f"**Validated:** {'Yes' if self.is_production_ready else 'No'}",
            f"**Position Size Multiplier:** {self.position_size_multiplier:.2f}x",
            "",
            "## Quality Dimensions",
            "",
            "| Dimension | Score | Status |",
            "|-----------|-------|--------|",
        ]

        for dim, score in self.metrics.dimension_breakdown.items():
            status = "OK" if score >= 85 else "Warning" if score >= 70 else "Critical"
            lines.append(f"| {dim.capitalize()} | {score:.1f}% | {status} |")

        lines.extend([
            "",
            "## Data Summary",
            "",
            f"- **Total Rows:** {self.metrics.total_rows:,}",
            f"- **Valid Rows:** {self.metrics.valid_rows:,} ({100 - self.metrics.invalid_pct:.1f}%)",
            f"- **Duplicates:** {self.metrics.duplicate_count:,}",
            f"- **Outliers:** {self.metrics.outlier_count:,}",
            f"- **Data Gaps:** {self.metrics.gap_count}",
            f"- **Freshness:** {self.metrics.freshness_status}",
        ])

        if self.issues:
            lines.extend([
                "",
                "## Issues Detected",
                "",
                f"**Total Issues:** {len(self.issues)}",
                "",
            ])
            for severity in AlertSeverity:
                count = sum(1 for i in self.issues if i.severity == severity)
                if count > 0:
                    lines.append(f"- {severity.icon} {severity.name}: {count}")

            lines.extend([
                "",
                "### Top Issues",
                "",
            ])
            for issue in sorted(self.issues, key=lambda x: x.score_impact, reverse=True)[:5]:
                lines.append(f"- {issue.summary}")

        if self.improvement_actions:
            lines.extend([
                "",
                "## Recommended Actions",
                "",
            ])
            for i, action in enumerate(self.improvement_actions, 1):
                lines.append(f"{i}. {action}")

        return "\n".join(lines)

# =============================================================================
# QUALITY CHECKER CLASS
# =============================================================================

class QualityChecker:
    """
    Comprehensive data quality checker for statistical arbitrage systems.

    Provides multi-dimensional quality assessment with trading-specific
    metrics and actionable recommendations.
    """

    # Default thresholds
    DEFAULT_THRESHOLDS = {
        'completeness_min': 85.0,
        'accuracy_min': 90.0,
        'consistency_min': 80.0,
        'timeliness_max_seconds': 300,
        'uniqueness_min': 99.0,
        'outlier_sigma': 4.0,
        'max_gap_periods': 3,
        'cross_venue_max_deviation_pct': 5.0,
        'min_correlation': 0.85,
    }

    # Pipeline health thresholds
    PIPELINE_THRESHOLDS = {
        'collection_rate_min': 100, # records/min
        'error_rate_max_pct': 1.0, # 1%
        'latency_p50_max_ms': 500,
        'latency_p99_max_ms': 2000,
        'staleness_max_seconds': 300,
        'venue_availability_min_pct': 80,
    }

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        pipeline_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize quality checker.

        Args:
            thresholds: Custom quality thresholds (merged with defaults)
            pipeline_thresholds: Custom pipeline thresholds (merged with defaults)
        """
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.pipeline_thresholds = {**self.PIPELINE_THRESHOLDS, **(pipeline_thresholds or {})}

    def comprehensive_check(
        self,
        df: pd.DataFrame,
        category: DataCategory,
        venue: Optional[str] = None,
        symbol: Optional[str] = None,
        timestamp_column: str = 'timestamp',
        check_cross_venue: bool = False,
    ) -> QualityCheckResult:
        """
        Perform comprehensive quality check on dataset.

        Args:
            df: DataFrame to validate
            category: Data category for validation rules
            venue: Optional venue filter
            symbol: Optional symbol filter
            timestamp_column: Name of timestamp column
            check_cross_venue: Whether to perform cross-venue consistency check

        Returns:
            QualityCheckResult with all metrics and issues
        """
        import time
        start_time = time.time()

        issues = []
        checks_performed = []

        # Filter data if specified
        data = df.copy()
        if venue and 'venue' in data.columns:
            data = data[data['venue'] == venue]
        if symbol and 'symbol' in data.columns:
            data = data[data['symbol'] == symbol]

        if len(data) == 0:
            return self._empty_result(category, checks_performed, time.time() - start_time)

        # 1. Schema check
        schema_issues = self._check_schema(data, category)
        issues.extend(schema_issues)
        checks_performed.append(CheckType.SCHEMA)

        # 2. Completeness check
        completeness_pct, completeness_issues = self._check_completeness(
            data, category, timestamp_column
        )
        issues.extend(completeness_issues)
        checks_performed.append(CheckType.COMPLETENESS)

        # 3. Range validation
        range_issues = self._check_ranges(data, category)
        issues.extend(range_issues)
        checks_performed.append(CheckType.RANGE)

        # 4. Outlier detection
        outlier_count, outlier_issues = self._detect_outliers(data, category)
        issues.extend(outlier_issues)
        checks_performed.append(CheckType.OUTLIER)

        # 5. Temporal consistency
        gap_count, gap_periods, temporal_issues = self._check_temporal(
            data, category, timestamp_column
        )
        issues.extend(temporal_issues)
        checks_performed.append(CheckType.TEMPORAL)

        # 6. Uniqueness check
        duplicate_count, uniqueness_issues = self._check_uniqueness(data, timestamp_column)
        issues.extend(uniqueness_issues)
        checks_performed.append(CheckType.UNIQUENESS)

        # 7. Freshness check
        freshness_seconds, freshness_issues = self._check_freshness(data, timestamp_column)
        issues.extend(freshness_issues)
        checks_performed.append(CheckType.FRESHNESS)

        # 8. Relationship validation (category-specific)
        relationship_issues = self._check_relationships(data, category)
        issues.extend(relationship_issues)
        checks_performed.append(CheckType.RELATIONSHIP)

        # Calculate dimension scores
        accuracy_pct = self._calculate_accuracy_score(data, category, issues)
        consistency_pct = self._calculate_consistency_score(issues)
        timeliness_pct = self._calculate_timeliness_score(freshness_seconds, category)
        uniqueness_pct = self._calculate_uniqueness_score(data, duplicate_count)

        # Calculate overall score
        overall_score = (
            category.completeness_weight * completeness_pct +
            category.accuracy_weight * accuracy_pct +
            category.consistency_weight * consistency_pct +
            category.timeliness_weight * timeliness_pct +
            category.uniqueness_weight * uniqueness_pct
        )

        # Extract metadata
        timestamps = pd.to_datetime(data[timestamp_column], errors='coerce')
        valid_ts = timestamps.dropna()

        venues = data['venue'].unique().tolist() if 'venue' in data.columns else []
        symbols = data['symbol'].unique().tolist() if 'symbol' in data.columns else []

        # Create metrics
        metrics = QualityMetrics(
            total_rows=len(data),
            valid_rows=len(data) - sum(i.affected_rows for i in issues if i.is_blocking),
            null_count=data.isnull().sum().sum(),
            duplicate_count=duplicate_count,
            outlier_count=outlier_count,
            gap_count=gap_count,
            gap_total_periods=gap_periods,
            freshness_seconds=freshness_seconds,
            completeness_pct=completeness_pct,
            accuracy_pct=accuracy_pct,
            consistency_pct=consistency_pct,
            timeliness_pct=timeliness_pct,
            uniqueness_pct=uniqueness_pct,
            overall_score=overall_score,
            data_category=category,
            start_timestamp=valid_ts.min() if len(valid_ts) > 0 else None,
            end_timestamp=valid_ts.max() if len(valid_ts) > 0 else None,
            venues=venues,
            symbols=symbols,
        )

        # Cross-venue consistency (if applicable)
        cross_venue = None
        if check_cross_venue and 'venue' in data.columns and len(venues) > 1:
            cross_venue = self._check_cross_venue_consistency(
                data, symbol or (symbols[0] if symbols else None), timestamp_column
            )
            checks_performed.append(CheckType.CROSS_VENUE)

        check_duration = (time.time() - start_time) * 1000

        return QualityCheckResult(
            metrics=metrics,
            issues=issues,
            cross_venue_consistency=cross_venue,
            check_duration_ms=check_duration,
            checks_performed=checks_performed,
        )

    def check(
        self,
        df: pd.DataFrame,
        data_type: str,
        venue: Optional[str] = None,
        symbol: Optional[str] = None,
        timestamp_column: str = 'timestamp',
    ) -> 'QualityCheckResult':
        """
        Convenience method for pipeline integration.

        Converts string data_type to DataCategory and performs comprehensive check.

        Args:
            df: DataFrame to validate
            data_type: Data type string ('funding_rates', 'ohlcv', etc.)
            venue: Optional venue filter
            symbol: Optional symbol filter
            timestamp_column: Name of timestamp column

        Returns:
            QualityCheckResult with all metrics and issues
        """
        # Convert string to DataCategory
        category_map = {
            'funding_rates': DataCategory.FUNDING_RATES,
            'funding': DataCategory.FUNDING_RATES,
            'ohlcv': DataCategory.OHLCV,
            'open_interest': DataCategory.OPEN_INTEREST,
            'oi': DataCategory.OPEN_INTEREST,
            'liquidations': DataCategory.LIQUIDATIONS,
            'orderbook': DataCategory.ORDERBOOK,
            'order_book': DataCategory.ORDERBOOK,
            'trades': DataCategory.TRADES,
            'options': DataCategory.OPTIONS,
        }
        category = category_map.get(data_type.lower(), DataCategory.OHLCV)

        return self.comprehensive_check(
            df=df,
            category=category,
            venue=venue,
            symbol=symbol,
            timestamp_column=timestamp_column,
        )

    def assess_pipeline_health(
        self,
        metrics: Dict[str, Any],
    ) -> PipelineHealth:
        """
        Assess overall pipeline health from collected metrics.

        Args:
            metrics: Dictionary of pipeline metrics including:
                - collection_rate_per_min: Records collected per minute
                - error_rate_pct: Error rate percentage
                - latency_p50_ms: 50th percentile latency
                - latency_p99_ms: 99th percentile latency
                - venue_status: Dict of venue -> status
                - last_collection_times: Dict of venue -> timestamp

        Returns:
            PipelineHealth assessment
        """
        alerts = []

        # Extract metrics with defaults
        collection_rate = metrics.get('collection_rate_per_min', 0)
        error_rate = metrics.get('error_rate_pct', 0)
        latency_p50 = metrics.get('latency_p50_ms', 0)
        latency_p99 = metrics.get('latency_p99_ms', 0)
        venue_status = metrics.get('venue_status', {})
        last_collection_times = metrics.get('last_collection_times', {})

        # Count venues
        total_venues = len(venue_status)
        active_venues = sum(1 for s in venue_status.values() if s == 'active')
        stale_venues = 0

        # Check staleness per venue
        now = datetime.now(timezone.utc)
        venue_health = {}
        for venue, last_time in last_collection_times.items():
            if isinstance(last_time, str):
                last_time = datetime.fromisoformat(last_time)
            staleness = (now - last_time).total_seconds()

            status = 'healthy'
            if staleness > self.pipeline_thresholds['staleness_max_seconds']:
                stale_venues += 1
                status = 'stale'
                alerts.append(PipelineAlert(
                    severity=AlertSeverity.HIGH,
                    message=f"Venue {venue} data is stale ({staleness:.0f}s)",
                    metric_name='staleness_seconds',
                    metric_value=staleness,
                    threshold_value=self.pipeline_thresholds['staleness_max_seconds'],
                    venue=venue,
                ))

            venue_health[venue] = {
                'status': status,
                'staleness_seconds': staleness,
                'last_collection': last_time.isoformat(),
            }

        # Check collection rate
        if collection_rate < self.pipeline_thresholds['collection_rate_min']:
            alerts.append(PipelineAlert(
                severity=AlertSeverity.MEDIUM,
                message=f"Low collection rate: {collection_rate:.1f}/min",
                metric_name='collection_rate_per_min',
                metric_value=collection_rate,
                threshold_value=self.pipeline_thresholds['collection_rate_min'],
            ))

        # Check error rate
        if error_rate > self.pipeline_thresholds['error_rate_max_pct']:
            severity = AlertSeverity.CRITICAL if error_rate > 5.0 else AlertSeverity.HIGH
            alerts.append(PipelineAlert(
                severity=severity,
                message=f"High error rate: {error_rate:.2f}%",
                metric_name='error_rate_pct',
                metric_value=error_rate,
                threshold_value=self.pipeline_thresholds['error_rate_max_pct'],
            ))

        # Check latency
        if latency_p99 > self.pipeline_thresholds['latency_p99_max_ms']:
            alerts.append(PipelineAlert(
                severity=AlertSeverity.MEDIUM,
                message=f"High P99 latency: {latency_p99:.0f}ms",
                metric_name='latency_p99_ms',
                metric_value=latency_p99,
                threshold_value=self.pipeline_thresholds['latency_p99_max_ms'],
            ))

        # Determine overall status
        critical_count = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
        high_count = sum(1 for a in alerts if a.severity == AlertSeverity.HIGH)

        if active_venues == 0:
            status = PipelineStatus.OFFLINE
        elif critical_count > 0:
            status = PipelineStatus.CRITICAL
        elif high_count > 0 or error_rate > 3.0:
            status = PipelineStatus.WARNING
        elif stale_venues > 0 or error_rate > 1.0:
            status = PipelineStatus.DEGRADED
        else:
            status = PipelineStatus.HEALTHY

        # Calculate overall score
        score = 100.0
        score -= min(20, error_rate * 10)
        score -= min(20, (100 - (active_venues / max(total_venues, 1) * 100)) * 0.5)
        score -= min(20, stale_venues * 5)
        score -= min(20, max(0, latency_p99 - 500) * 0.01)
        score = max(0, score)

        # Find last successful collection
        last_successful = None
        if last_collection_times:
            times = [
                datetime.fromisoformat(t) if isinstance(t, str) else t
                for t in last_collection_times.values()
            ]
            last_successful = max(times)

        return PipelineHealth(
            status=status,
            overall_score=score,
            collection_rate_per_min=collection_rate,
            error_rate_pct=error_rate,
            latency_p50_ms=latency_p50,
            latency_p99_ms=latency_p99,
            active_venues=active_venues,
            total_venues=total_venues,
            active_symbols=metrics.get('active_symbols', 0),
            stale_venue_count=stale_venues,
            alerts=alerts,
            venue_health=venue_health,
            last_successful_collection=last_successful,
        )

    def _check_schema(
        self,
        df: pd.DataFrame,
        category: DataCategory,
    ) -> List[QualityIssue]:
        """Check schema validity."""
        issues = []
        required = category.required_columns

        missing = [col for col in required if col not in df.columns]
        if missing:
            issues.append(QualityIssue(
                check_type=CheckType.SCHEMA,
                category=IssueCategory.SCHEMA_MISMATCH,
                severity=AlertSeverity.CRITICAL,
                message=f"Missing required columns: {missing}",
                affected_columns=missing,
            ))

        return issues

    def _check_completeness(
        self,
        df: pd.DataFrame,
        category: DataCategory,
        timestamp_column: str,
    ) -> Tuple[float, List[QualityIssue]]:
        """Check data completeness."""
        issues = []

        # Check for null values in required columns
        required = category.required_columns
        present_required = [c for c in required if c in df.columns]

        null_counts = df[present_required].isnull().sum()
        total_nulls = null_counts.sum()
        total_cells = len(df) * len(present_required)

        if total_nulls > 0:
            high_null_cols = null_counts[null_counts > len(df) * 0.05].index.tolist()
            if high_null_cols:
                issues.append(QualityIssue(
                    check_type=CheckType.COMPLETENESS,
                    category=IssueCategory.MISSING_DATA,
                    severity=AlertSeverity.MEDIUM,
                    message=f"High null rate in columns: {high_null_cols}",
                    affected_rows=int(null_counts[high_null_cols].max()),
                    affected_columns=high_null_cols,
                ))

        completeness_pct = ((total_cells - total_nulls) / max(total_cells, 1)) * 100
        return completeness_pct, issues

    def _check_ranges(
        self,
        df: pd.DataFrame,
        category: DataCategory,
    ) -> List[QualityIssue]:
        """Check value ranges for numeric columns."""
        issues = []

        # Category-specific range checks
        if category == DataCategory.FUNDING_RATES:
            if 'funding_rate' in df.columns:
                extreme = df[(df['funding_rate'].abs() > 0.01)] # >1% per period
                if len(extreme) > 0:
                    issues.append(QualityIssue(
                        check_type=CheckType.RANGE,
                        category=IssueCategory.OUTLIER,
                        severity=AlertSeverity.LOW,
                        message=f"Extreme funding rates detected (>1%)",
                        affected_rows=len(extreme),
                        affected_columns=['funding_rate'],
                        sample_values=extreme['funding_rate'].head().tolist(),
                    ))

        elif category == DataCategory.OHLCV:
            # Check OHLC relationships
            if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
                invalid = df[
                    (df['high'] < df['low']) |
                    (df['high'] < df['open']) |
                    (df['high'] < df['close']) |
                    (df['low'] > df['open']) |
                    (df['low'] > df['close'])
                ]
                if len(invalid) > 0:
                    issues.append(QualityIssue(
                        check_type=CheckType.RANGE,
                        category=IssueCategory.RELATIONSHIP_VIOLATION,
                        severity=AlertSeverity.HIGH,
                        message="Invalid OHLC relationships (high<low, etc.)",
                        affected_rows=len(invalid),
                        affected_columns=['open', 'high', 'low', 'close'],
                    ))

            # Check for negative prices
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    negative = df[df[col] < 0]
                    if len(negative) > 0:
                        issues.append(QualityIssue(
                            check_type=CheckType.RANGE,
                            category=IssueCategory.INVALID_VALUE,
                            severity=AlertSeverity.CRITICAL,
                            message=f"Negative prices in {col}",
                            affected_rows=len(negative),
                            affected_columns=[col],
                        ))

        return issues

    def _detect_outliers(
        self,
        df: pd.DataFrame,
        category: DataCategory,
    ) -> Tuple[int, List[QualityIssue]]:
        """Detect statistical outliers."""
        issues = []
        total_outliers = 0

        numeric_cols = category.numeric_columns
        threshold = category.outlier_threshold_sigma

        for col in numeric_cols:
            if col not in df.columns:
                continue

            values = df[col].dropna()
            if len(values) < 10:
                continue

            mean = values.mean()
            std = values.std()

            if std == 0:
                continue

            z_scores = np.abs((values - mean) / std)
            outlier_mask = z_scores > threshold
            outlier_count = outlier_mask.sum()

            if outlier_count > 0:
                total_outliers += outlier_count
                if outlier_count > len(values) * 0.01: # More than 1%
                    issues.append(QualityIssue(
                        check_type=CheckType.OUTLIER,
                        category=IssueCategory.OUTLIER,
                        severity=AlertSeverity.LOW,
                        message=f"Outliers in {col}: {outlier_count} ({outlier_count/len(values)*100:.2f}%)",
                        affected_rows=int(outlier_count),
                        affected_columns=[col],
                        sample_values=values[outlier_mask].head().tolist(),
                    ))

        return total_outliers, issues

    def _check_temporal(
        self,
        df: pd.DataFrame,
        category: DataCategory,
        timestamp_column: str,
    ) -> Tuple[int, int, List[QualityIssue]]:
        """Check temporal consistency and gaps."""
        issues = []
        gap_count = 0
        gap_total_periods = 0

        if timestamp_column not in df.columns:
            return 0, 0, issues

        timestamps = pd.to_datetime(df[timestamp_column], errors='coerce')
        timestamps = timestamps.dropna().sort_values()

        if len(timestamps) < 2:
            return 0, 0, issues

        # Check for non-monotonic timestamps
        if not timestamps.is_monotonic_increasing:
            issues.append(QualityIssue(
                check_type=CheckType.TEMPORAL,
                category=IssueCategory.TEMPORAL_ISSUE,
                severity=AlertSeverity.MEDIUM,
                message="Timestamps are not monotonically increasing",
                affected_columns=[timestamp_column],
            ))

        # Check for gaps
        expected_freq = category.expected_frequency_minutes
        if expected_freq:
            diffs = timestamps.diff().dropna()
            expected_delta = pd.Timedelta(minutes=expected_freq)
            tolerance = expected_delta * 1.5

            gaps = diffs[diffs > tolerance]
            gap_count = len(gaps)

            if gap_count > 0:
                # gaps.sum() / expected_delta gives a float ratio, not a timedelta
                gap_ratio = gaps.sum() / expected_delta
                gap_total_periods = int(gap_ratio) if isinstance(gap_ratio, (int, float)) else int(gap_ratio.total_seconds() / 60 / expected_freq)

                if gap_count > 5:
                    issues.append(QualityIssue(
                        check_type=CheckType.TEMPORAL,
                        category=IssueCategory.MISSING_DATA,
                        severity=AlertSeverity.MEDIUM if gap_count < 20 else AlertSeverity.HIGH,
                        message=f"Detected {gap_count} data gaps totaling ~{gap_total_periods} periods",
                        affected_rows=gap_count,
                        affected_columns=[timestamp_column],
                    ))

        return gap_count, gap_total_periods, issues

    def _check_uniqueness(
        self,
        df: pd.DataFrame,
        timestamp_column: str,
    ) -> Tuple[int, List[QualityIssue]]:
        """Check for duplicate records."""
        issues = []

        # Determine key columns
        key_cols = [timestamp_column]
        if 'venue' in df.columns:
            key_cols.append('venue')
        if 'symbol' in df.columns:
            key_cols.append('symbol')

        present_keys = [c for c in key_cols if c in df.columns]
        if not present_keys:
            return 0, issues

        duplicates = df.duplicated(subset=present_keys, keep='first')
        duplicate_count = duplicates.sum()

        if duplicate_count > 0:
            severity = AlertSeverity.LOW if duplicate_count < 100 else AlertSeverity.MEDIUM
            issues.append(QualityIssue(
                check_type=CheckType.UNIQUENESS,
                category=IssueCategory.DUPLICATE,
                severity=severity,
                message=f"Found {duplicate_count} duplicate records",
                affected_rows=int(duplicate_count),
                affected_columns=present_keys,
            ))

        return int(duplicate_count), issues

    def _check_freshness(
        self,
        df: pd.DataFrame,
        timestamp_column: str,
    ) -> Tuple[float, List[QualityIssue]]:
        """Check data freshness."""
        issues = []

        if timestamp_column not in df.columns:
            return float('inf'), issues

        timestamps = pd.to_datetime(df[timestamp_column], errors='coerce')
        latest = timestamps.max()

        if pd.isna(latest):
            return float('inf'), issues

        # Handle both timezone-aware and naive timestamps
        now_utc = datetime.now(timezone.utc)
        latest_dt = latest.to_pydatetime()
        if latest_dt.tzinfo is None:
            # Assume naive timestamps are UTC
            latest_dt = latest_dt.replace(tzinfo=timezone.utc)
        freshness = (now_utc - latest_dt).total_seconds()

        if freshness > self.thresholds['timeliness_max_seconds']:
            issues.append(QualityIssue(
                check_type=CheckType.FRESHNESS,
                category=IssueCategory.STALE_DATA,
                severity=AlertSeverity.MEDIUM if freshness < 600 else AlertSeverity.HIGH,
                message=f"Data is stale: {freshness:.0f} seconds old",
                affected_columns=[timestamp_column],
            ))

        return freshness, issues

    def _check_relationships(
        self,
        df: pd.DataFrame,
        category: DataCategory,
    ) -> List[QualityIssue]:
        """Check inter-column relationships."""
        issues = []

        # Category-specific relationship checks
        if category == DataCategory.OPTIONS:
            if all(c in df.columns for c in ['strike', 'price']):
                invalid = df[df['price'] > df['strike'] * 2]
                if len(invalid) > 0:
                    issues.append(QualityIssue(
                        check_type=CheckType.RELATIONSHIP,
                        category=IssueCategory.RELATIONSHIP_VIOLATION,
                        severity=AlertSeverity.MEDIUM,
                        message="Option prices exceed 2x strike (suspicious)",
                        affected_rows=len(invalid),
                        affected_columns=['strike', 'price'],
                    ))

        return issues

    def _check_cross_venue_consistency(
        self,
        df: pd.DataFrame,
        symbol: Optional[str],
        timestamp_column: str,
    ) -> Optional[CrossVenueConsistency]:
        """Check consistency across venues."""
        if 'venue' not in df.columns:
            return None

        venues = df['venue'].unique().tolist()
        if len(venues) < 2:
            return None

        # Filter by symbol if specified
        data = df if not symbol else df[df['symbol'] == symbol]

        # Get numeric column for comparison
        numeric_cols = [c for c in data.columns if data[c].dtype in ['float64', 'float32', 'int64', 'int32']]
        value_col = None
        for col in ['funding_rate', 'close', 'price', 'value']:
            if col in numeric_cols:
                value_col = col
                break

        if not value_col:
            return None

        # Calculate per-venue statistics
        mean_by_venue = {}
        std_by_venue = {}
        for venue in venues:
            venue_data = data[data['venue'] == venue][value_col].dropna()
            mean_by_venue[venue] = float(venue_data.mean()) if len(venue_data) > 0 else 0
            std_by_venue[venue] = float(venue_data.std()) if len(venue_data) > 0 else 0

        # Calculate correlations
        correlation_matrix = {}
        for v1 in venues:
            correlation_matrix[v1] = {}
            for v2 in venues:
                if v1 == v2:
                    correlation_matrix[v1][v2] = 1.0
                else:
                    # Align timestamps and calculate correlation
                    d1 = data[data['venue'] == v1][[timestamp_column, value_col]].set_index(timestamp_column)
                    d2 = data[data['venue'] == v2][[timestamp_column, value_col]].set_index(timestamp_column)
                    merged = d1.join(d2, lsuffix='_1', rsuffix='_2', how='inner')
                    if len(merged) > 10:
                        corr = merged[f'{value_col}_1'].corr(merged[f'{value_col}_2'])
                        correlation_matrix[v1][v2] = float(corr) if not np.isnan(corr) else 0.0
                    else:
                        correlation_matrix[v1][v2] = 0.0

        # Calculate deviations
        values = list(mean_by_venue.values())
        overall_mean = np.mean(values) if values else 0

        deviations = []
        for v in values:
            if overall_mean != 0:
                deviations.append(abs((v - overall_mean) / overall_mean) * 100)

        max_deviation = max(deviations) if deviations else 0
        mean_deviation = np.mean(deviations) if deviations else 0

        # Timestamp alignment
        timestamps = pd.to_datetime(data[timestamp_column], errors='coerce')
        time_range = (timestamps.min().to_pydatetime(), timestamps.max().to_pydatetime())

        # Count overlapping periods
        venue_timestamps = {}
        for venue in venues:
            venue_timestamps[venue] = set(data[data['venue'] == venue][timestamp_column].unique())

        common = set.intersection(*venue_timestamps.values()) if venue_timestamps else set()
        total_unique = set.union(*venue_timestamps.values()) if venue_timestamps else set()

        alignment_pct = (len(common) / len(total_unique) * 100) if total_unique else 0

        # Calculate consistency score
        avg_corr = np.mean([
            correlation_matrix[v1][v2]
            for v1 in venues for v2 in venues if v1 != v2
        ]) if len(venues) > 1 else 1.0

        consistency_score = (
            max(0, 100 - mean_deviation * 10) * 0.4 +
            alignment_pct * 0.3 +
            avg_corr * 100 * 0.3
        )

        return CrossVenueConsistency(
            symbol=symbol or 'ALL',
            venues=venues,
            timestamp_range=time_range,
            mean_value_by_venue=mean_by_venue,
            std_by_venue=std_by_venue,
            correlation_matrix=correlation_matrix,
            max_deviation_pct=float(max_deviation),
            mean_deviation_pct=float(mean_deviation),
            timestamp_alignment_pct=float(alignment_pct),
            overlapping_periods=len(common),
            consistency_score=float(consistency_score),
        )

    def _calculate_accuracy_score(
        self,
        df: pd.DataFrame,
        category: DataCategory,
        issues: List[QualityIssue],
    ) -> float:
        """Calculate accuracy dimension score."""
        score = 100.0

        # Deduct for accuracy-related issues
        for issue in issues:
            if issue.check_type in {CheckType.SCHEMA, CheckType.RANGE, CheckType.RELATIONSHIP}:
                score -= issue.score_impact

        return max(0, score)

    def _calculate_consistency_score(
        self,
        issues: List[QualityIssue],
    ) -> float:
        """Calculate consistency dimension score."""
        score = 100.0

        for issue in issues:
            if issue.check_type in {CheckType.TEMPORAL, CheckType.CONSISTENCY, CheckType.CROSS_VENUE}:
                score -= issue.score_impact

        return max(0, score)

    def _calculate_timeliness_score(
        self,
        freshness_seconds: float,
        category: DataCategory,
    ) -> float:
        """Calculate timeliness dimension score."""
        max_staleness = self.thresholds['timeliness_max_seconds']

        if freshness_seconds <= 60:
            return 100.0
        elif freshness_seconds <= max_staleness:
            return 100 - ((freshness_seconds - 60) / (max_staleness - 60)) * 20
        else:
            return max(0, 80 - (freshness_seconds - max_staleness) / 60)

    def _calculate_uniqueness_score(
        self,
        df: pd.DataFrame,
        duplicate_count: int,
    ) -> float:
        """Calculate uniqueness dimension score."""
        if len(df) == 0:
            return 100.0

        duplicate_pct = (duplicate_count / len(df)) * 100
        return max(0, 100 - duplicate_pct * 2)

    def _empty_result(
        self,
        category: DataCategory,
        checks_performed: List[CheckType],
        duration_ms: float,
    ) -> QualityCheckResult:
        """Create result for empty dataset."""
        metrics = QualityMetrics(
            total_rows=0,
            valid_rows=0,
            null_count=0,
            duplicate_count=0,
            outlier_count=0,
            gap_count=0,
            gap_total_periods=0,
            freshness_seconds=float('inf'),
            completeness_pct=0,
            accuracy_pct=0,
            consistency_pct=0,
            timeliness_pct=0,
            uniqueness_pct=100,
            overall_score=0,
            data_category=category,
        )

        return QualityCheckResult(
            metrics=metrics,
            issues=[QualityIssue(
                check_type=CheckType.COMPLETENESS,
                category=IssueCategory.MISSING_DATA,
                severity=AlertSeverity.CRITICAL,
                message="Dataset is empty",
            )],
            check_duration_ms=duration_ms,
            checks_performed=checks_performed,
        )

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_quality_check(
    df: pd.DataFrame,
    category: Union[DataCategory, str] = DataCategory.OHLCV,
) -> Dict[str, Any]:
    """
    Perform quick quality check and return summary.

    Args:
        df: DataFrame to check
        category: Data category (enum or string)

    Returns:
        Dictionary with quality summary
    """
    if isinstance(category, str):
        category = DataCategory(category)

    checker = QualityChecker()
    result = checker.comprehensive_check(df, category)

    return {
        'score': result.overall_score,
        'level': result.quality_level.value,
        'is_valid': result.is_valid,
        'is_production_ready': result.is_production_ready,
        'issue_count': len(result.issues),
        'blocking_issues': len(result.blocking_issues),
    }

def validate_for_trading(
    df: pd.DataFrame,
    category: DataCategory,
    min_score: float = 85.0,
) -> Tuple[bool, str]:
    """
    Validate dataset for trading use.

    Args:
        df: DataFrame to validate
        category: Data category
        min_score: Minimum required quality score

    Returns:
        Tuple of (is_valid, reason)
    """
    checker = QualityChecker()
    result = checker.comprehensive_check(df, category)

    if result.overall_score < min_score:
        return False, f"Quality score {result.overall_score:.1f}% below minimum {min_score}%"

    if result.blocking_issues:
        return False, f"Blocking issues: {result.blocking_issues[0].message}"

    if not result.is_production_ready:
        return False, f"Not validated: {result.quality_level.description}"

    return True, "Data validated for trading"

def get_position_size_adjustment(
    df: pd.DataFrame,
    category: DataCategory,
) -> float:
    """
    Get recommended position size adjustment based on data quality.

    Args:
        df: DataFrame to assess
        category: Data category

    Returns:
        Position size multiplier (0.0 to 1.0)
    """
    checker = QualityChecker()
    result = checker.comprehensive_check(df, category)
    return result.position_size_multiplier

__all__ = [
    # Enums
    'DataCategory',
    'QualityLevel',
    'AlertSeverity',
    'PipelineStatus',
    'CheckType',
    'IssueCategory',
    # Dataclasses
    'QualityIssue',
    'QualityMetrics',
    'PipelineAlert',
    'PipelineHealth',
    'CrossVenueConsistency',
    'QualityCheckResult',
    # Classes
    'QualityChecker',
    # Functions
    'quick_quality_check',
    'validate_for_trading',
    'get_position_size_adjustment',
]
