"""
Base Collector Module for Crypto Statistical Arbitrage Data Collection.

This module provides the foundation for all venue-specific data collectors with:
- Comprehensive venue type classification with trading-specific properties
- Data type enumeration with required schemas and refresh intervals
- Collection statistics with performance analytics
- Validation results with quality scoring
- Rate limiting and retry handling integration
- Parquet storage with compression options

==============================================================================
VENUE TYPE CLASSIFICATION
==============================================================================

The VenueType enum classifies data sources with properties relevant to
statistical arbitrage execution:

+------------------+------------------+------------------+------------------+
| Type | Settlement | Typical Latency | API Style |
+------------------+------------------+------------------+------------------+
| CEX | Off-chain | 50-200ms | REST/WebSocket |
| HYBRID | On-chain | 100-500ms | REST/WebSocket |
| DEX | On-chain | 500ms-2s | Subgraph/RPC |
| OPTIONS | Mixed | 100-500ms | REST/WebSocket |
| MARKET_DATA | N/A | 200-1000ms | REST |
| ONCHAIN | N/A | 1-10s | GraphQL/REST |
| ALTERNATIVE | N/A | 1-60s | REST |
+------------------+------------------+------------------+------------------+

==============================================================================
DATA TYPE SCHEMAS
==============================================================================

Each DataType has a required schema for consistency across venues:

FUNDING_RATES:
    timestamp datetime64[ns] UTC timestamp
    symbol str Trading pair (e.g., 'BTC')
    funding_rate float64 Decimal rate (0.0001 = 0.01%)
    mark_price float64 Mark price at settlement
    venue str Venue identifier
    venue_type str VenueType value

OHLCV:
    timestamp datetime64[ns] Candle open time
    symbol str Trading pair
    open float64 Open price
    high float64 High price
    low float64 Low price
    close float64 Close price
    volume float64 Volume in base currency
    venue str Venue identifier

==============================================================================
STATISTICAL ARBITRAGE INTEGRATION
==============================================================================

Key considerations for arbitrage data collection:

1. TIMING PRECISION
   - Funding rates must capture settlement times accurately
   - Cross-venue alignment requires timestamp normalization
   - Network latency affects data freshness

2. DATA QUALITY
   - Missing data impacts signal generation
   - Outliers may indicate market events or errors
   - Cross-venue validation essential for spread calculations

3. RATE LIMITS
   - CEX venues typically 60-1200 requests/minute
   - Hybrid venues 50-100 requests/minute
   - Subgraph queries subject to compute limits

Version: 2.0.0
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from enum import Enum
import pandas as pd
import numpy as np
import logging
import asyncio
import time
import hashlib
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# =============================================================================
# ENUMS
# =============================================================================

class VenueType(Enum):
    """
    Classification of data source venues with trading-specific properties.
    
    Each venue type has characteristics that affect data collection strategy,
    execution latency, and arbitrage viability.
    """
    
    CEX = "CEX"
    HYBRID = "hybrid"
    DEX = "DEX"
    OPTIONS = "options"
    MARKET_DATA = "market_data"
    ONCHAIN = "onchain"
    ALTERNATIVE = "alternative"
    INDEXER = "indexer"
    SOCIAL = "social"
    
    @property
    def settlement_type(self) -> str:
        """Settlement mechanism for this venue type."""
        settlements = {
            VenueType.CEX: "off-chain",
            VenueType.HYBRID: "on-chain",
            VenueType.DEX: "on-chain",
            VenueType.OPTIONS: "mixed",
            VenueType.MARKET_DATA: "n/a",
            VenueType.ONCHAIN: "n/a",
            VenueType.ALTERNATIVE: "n/a",
            VenueType.INDEXER: "n/a",
            VenueType.SOCIAL: "n/a",
        }
        return settlements.get(self, "unknown")
    
    @property
    def typical_latency_ms(self) -> Tuple[int, int]:
        """Typical API latency range (min, max) in milliseconds."""
        latencies = {
            VenueType.CEX: (50, 200),
            VenueType.HYBRID: (100, 500),
            VenueType.DEX: (500, 2000),
            VenueType.OPTIONS: (100, 500),
            VenueType.MARKET_DATA: (200, 1000),
            VenueType.ONCHAIN: (1000, 10000),
            VenueType.ALTERNATIVE: (1000, 60000),
            VenueType.INDEXER: (500, 5000),
            VenueType.SOCIAL: (500, 5000),
        }
        return latencies.get(self, (100, 1000))
    
    @property
    def api_style(self) -> str:
        """Primary API interface style."""
        styles = {
            VenueType.CEX: "REST/WebSocket",
            VenueType.HYBRID: "REST/WebSocket",
            VenueType.DEX: "Subgraph/RPC",
            VenueType.OPTIONS: "REST/WebSocket",
            VenueType.MARKET_DATA: "REST",
            VenueType.ONCHAIN: "GraphQL/REST",
            VenueType.ALTERNATIVE: "REST",
            VenueType.INDEXER: "GraphQL",
            VenueType.SOCIAL: "REST",
        }
        return styles.get(self, "REST")
    
    @property
    def supports_websocket(self) -> bool:
        """Whether venue typically supports WebSocket streaming."""
        return self in {VenueType.CEX, VenueType.HYBRID, VenueType.OPTIONS}
    
    @property
    def supports_historical(self) -> bool:
        """Whether venue typically supports historical data queries."""
        return self in {
            VenueType.CEX, VenueType.HYBRID, VenueType.OPTIONS,
            VenueType.MARKET_DATA, VenueType.ONCHAIN, VenueType.INDEXER
        }
    
    @property
    def arbitrage_priority(self) -> int:
        """Priority for cross-venue arbitrage (1=highest)."""
        priorities = {
            VenueType.CEX: 1,
            VenueType.HYBRID: 2,
            VenueType.OPTIONS: 3,
            VenueType.DEX: 4,
            VenueType.MARKET_DATA: 5,
            VenueType.ONCHAIN: 6,
            VenueType.ALTERNATIVE: 7,
            VenueType.INDEXER: 8,
            VenueType.SOCIAL: 9,
        }
        return priorities.get(self, 10)
    
    @property
    def default_rate_limit_per_minute(self) -> int:
        """Default rate limit assumption per minute."""
        limits = {
            VenueType.CEX: 200,
            VenueType.HYBRID: 100,
            VenueType.DEX: 60,
            VenueType.OPTIONS: 200,
            VenueType.MARKET_DATA: 100,
            VenueType.ONCHAIN: 30,
            VenueType.ALTERNATIVE: 60,
            VenueType.INDEXER: 100,
            VenueType.SOCIAL: 60,
        }
        return limits.get(self, 60)
    
    @classmethod
    def from_string(cls, value: str) -> 'VenueType':
        """Convert string to VenueType enum."""
        value_normalized = value.strip().upper()
        for member in cls:
            if member.value.upper() == value_normalized:
                return member
            if member.name == value_normalized:
                return member
        raise ValueError(f"Unknown venue type: {value}")
    
    @classmethod
    def perpetual_venues(cls) -> List['VenueType']:
        """Venue types that support perpetual swaps."""
        return [cls.CEX, cls.HYBRID]
    
    @classmethod
    def spot_venues(cls) -> List['VenueType']:
        """Venue types that support spot trading."""
        return [cls.CEX, cls.DEX]

class DataType(Enum):
    """
    Types of market data with schema definitions and collection parameters.

    IMPORTANT: Keep synchronized with data_cleaner.py and data_validator.py DataType enums.
    """

    FUNDING_RATES = "funding_rates"
    OHLCV = "ohlcv"
    OPEN_INTEREST = "open_interest"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    LIQUIDATIONS = "liquidations"
    OPTIONS = "options"
    OPTIONS_CHAIN = "options_chain" # Alias for OPTIONS
    DEX_POOLS = "dex_pools"
    DEX_SWAPS = "dex_swaps"
    TVL = "tvl"
    ONCHAIN_FLOWS = "onchain_flows"
    SOCIAL_METRICS = "social_metrics"
    SOCIAL_SENTIMENT = "social_sentiment"
    TERM_STRUCTURE = "term_structure"
    ON_CHAIN = "on_chain"
    
    @property
    def required_columns(self) -> List[str]:
        """Required columns for this data type."""
        schemas = {
            DataType.FUNDING_RATES: [
                'timestamp', 'symbol', 'funding_rate', 'venue'
            ],
            DataType.OHLCV: [
                'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'venue'
            ],
            DataType.OPEN_INTEREST: [
                'timestamp', 'symbol', 'open_interest', 'venue'
            ],
            DataType.ORDERBOOK: [
                'timestamp', 'symbol', 'bids', 'asks', 'venue'
            ],
            DataType.TRADES: [
                'timestamp', 'symbol', 'price', 'size', 'side', 'venue'
            ],
            DataType.LIQUIDATIONS: [
                'timestamp', 'symbol', 'side', 'size', 'price', 'venue'
            ],
            DataType.OPTIONS_CHAIN: [
                'timestamp', 'symbol', 'strike', 'expiry', 'type', 'bid', 'ask', 'venue'
            ],
            DataType.DEX_POOLS: [
                'pool_address', 'token0', 'token1', 'tvl', 'chain'
            ],
            DataType.DEX_SWAPS: [
                'timestamp', 'pool_address', 'token_in', 'token_out', 'amount_in', 'amount_out'
            ],
            DataType.TVL: [
                'timestamp', 'protocol', 'tvl', 'chain'
            ],
            DataType.ONCHAIN_FLOWS: [
                'timestamp', 'entity', 'flow_type', 'amount', 'chain'
            ],
            DataType.SOCIAL_METRICS: [
                'timestamp', 'symbol', 'metric_type', 'value'
            ],
        }
        return schemas.get(self, ['timestamp', 'symbol', 'venue'])
    
    @property
    def refresh_interval_seconds(self) -> int:
        """Typical refresh interval in seconds."""
        intervals = {
            DataType.FUNDING_RATES: 3600, # Hourly
            DataType.OHLCV: 60, # Per minute
            DataType.OPEN_INTEREST: 300, # 5 minutes
            DataType.ORDERBOOK: 1, # Real-time
            DataType.TRADES: 1, # Real-time
            DataType.LIQUIDATIONS: 60, # Per minute
            DataType.OPTIONS_CHAIN: 3600, # Hourly
            DataType.DEX_POOLS: 900, # 15 minutes
            DataType.DEX_SWAPS: 300, # 5 minutes
            DataType.TVL: 3600, # Hourly
            DataType.ONCHAIN_FLOWS: 3600, # Hourly
            DataType.SOCIAL_METRICS: 3600, # Hourly
        }
        return intervals.get(self, 3600)
    
    @property
    def arbitrage_priority(self) -> int:
        """Priority for arbitrage strategy (1=highest)."""
        priorities = {
            DataType.FUNDING_RATES: 1,
            DataType.OHLCV: 2,
            DataType.OPEN_INTEREST: 3,
            DataType.OPTIONS_CHAIN: 4,
            DataType.DEX_POOLS: 5,
            DataType.ORDERBOOK: 6,
            DataType.TRADES: 7,
            DataType.LIQUIDATIONS: 8,
            DataType.TVL: 9,
            DataType.ONCHAIN_FLOWS: 10,
            DataType.DEX_SWAPS: 11,
            DataType.SOCIAL_METRICS: 12,
        }
        return priorities.get(self, 10)
    
    @property
    def is_time_series(self) -> bool:
        """Whether data is time-series (vs point-in-time)."""
        return self not in {DataType.DEX_POOLS}
    
    @property
    def supports_historical(self) -> bool:
        """Whether historical queries are typically supported."""
        return self not in {DataType.ORDERBOOK}

class CollectionPhase(Enum):
    """
    Phases of data collection lifecycle.
    """
    
    INITIALIZING = "initializing"
    AUTHENTICATING = "authenticating"
    FETCHING = "fetching"
    VALIDATING = "validating"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    
    @property
    def is_terminal(self) -> bool:
        """Whether this is a terminal phase."""
        return self in {
            CollectionPhase.COMPLETED,
            CollectionPhase.FAILED,
            CollectionPhase.CANCELLED
        }
    
    @property
    def is_active(self) -> bool:
        """Whether collection is actively running."""
        return self in {
            CollectionPhase.AUTHENTICATING,
            CollectionPhase.FETCHING,
            CollectionPhase.VALIDATING,
            CollectionPhase.STORING
        }

class QualityGrade(Enum):
    """
    Quality grades for collected data.
    """
    
    EXCELLENT = "excellent" # 95-100%
    GOOD = "good" # 85-95%
    ACCEPTABLE = "acceptable" # 70-85%
    MARGINAL = "marginal" # 50-70%
    POOR = "poor" # <50%
    
    @property
    def min_score(self) -> float:
        """Minimum quality score for this grade."""
        thresholds = {
            QualityGrade.EXCELLENT: 95.0,
            QualityGrade.GOOD: 85.0,
            QualityGrade.ACCEPTABLE: 70.0,
            QualityGrade.MARGINAL: 50.0,
            QualityGrade.POOR: 0.0,
        }
        return thresholds.get(self, 0.0)
    
    @property
    def trading_viable(self) -> bool:
        """Whether data quality is sufficient for trading."""
        return self in {QualityGrade.EXCELLENT, QualityGrade.GOOD, QualityGrade.ACCEPTABLE}
    
    @property
    def backtest_viable(self) -> bool:
        """Whether data quality is sufficient for backtesting."""
        return self in {QualityGrade.EXCELLENT, QualityGrade.GOOD}
    
    @classmethod
    def from_score(cls, score: float) -> 'QualityGrade':
        """Get grade from quality score."""
        if score >= 95.0:
            return cls.EXCELLENT
        elif score >= 85.0:
            return cls.GOOD
        elif score >= 70.0:
            return cls.ACCEPTABLE
        elif score >= 50.0:
            return cls.MARGINAL
        else:
            return cls.POOR

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class CollectionStats:
    """
    Comprehensive statistics for data collection runs.
    
    Tracks performance metrics, error rates, and collection efficiency
    for monitoring and optimization.
    """
    
    records_collected: int = 0
    records_failed: int = 0
    records_skipped: int = 0
    api_calls: int = 0
    api_errors: int = 0
    rate_limit_hits: int = 0
    retries: int = 0
    bytes_downloaded: int = 0
    errors: int = 0
    warnings: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    symbols_processed: List[str] = field(default_factory=list)
    symbols_failed: List[str] = field(default_factory=list)
    phase: CollectionPhase = CollectionPhase.INITIALIZING
    
    # Computed at runtime
    _latency_samples: List[float] = field(default_factory=list, repr=False)
    
    @property
    def duration_seconds(self) -> float:
        """Collection duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return 0.0
    
    @property
    def records_per_second(self) -> float:
        """Collection rate (records/second)."""
        if self.duration_seconds > 0:
            return self.records_collected / self.duration_seconds
        return 0.0
    
    @property
    def api_calls_per_second(self) -> float:
        """API call rate (calls/second)."""
        if self.duration_seconds > 0:
            return self.api_calls / self.duration_seconds
        return 0.0
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        total = self.records_collected + self.records_failed
        if total == 0:
            return 0.0
        return (self.records_collected / total) * 100
    
    @property
    def error_rate(self) -> float:
        """API error rate as percentage."""
        if self.api_calls == 0:
            return 0.0
        return (self.api_errors / self.api_calls) * 100
    
    @property
    def average_latency_ms(self) -> float:
        """Average API latency in milliseconds."""
        if not self._latency_samples:
            return 0.0
        return sum(self._latency_samples) / len(self._latency_samples)
    
    @property
    def p95_latency_ms(self) -> float:
        """95th percentile latency in milliseconds."""
        if not self._latency_samples:
            return 0.0
        sorted_samples = sorted(self._latency_samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]
    
    @property
    def bytes_per_record(self) -> float:
        """Average bytes per record."""
        if self.records_collected == 0:
            return 0.0
        return self.bytes_downloaded / self.records_collected
    
    @property
    def estimated_completion_pct(self) -> float:
        """Estimated completion percentage."""
        total_symbols = len(self.symbols_processed) + len(self.symbols_failed)
        if total_symbols == 0:
            return 0.0
        return (len(self.symbols_processed) / total_symbols) * 100
    
    def record_latency(self, latency_ms: float) -> None:
        """Record an API latency sample."""
        self._latency_samples.append(latency_ms)
        # Keep only last 1000 samples
        if len(self._latency_samples) > 1000:
            self._latency_samples = self._latency_samples[-1000:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'records_collected': self.records_collected,
            'records_failed': self.records_failed,
            'records_skipped': self.records_skipped,
            'api_calls': self.api_calls,
            'api_errors': self.api_errors,
            'rate_limit_hits': self.rate_limit_hits,
            'retries': self.retries,
            'bytes_downloaded': self.bytes_downloaded,
            'errors': self.errors,
            'warnings': self.warnings,
            'duration_seconds': self.duration_seconds,
            'records_per_second': self.records_per_second,
            'success_rate': self.success_rate,
            'error_rate': self.error_rate,
            'average_latency_ms': self.average_latency_ms,
            'p95_latency_ms': self.p95_latency_ms,
            'symbols_processed': len(self.symbols_processed),
            'symbols_failed': len(self.symbols_failed),
            'phase': self.phase.value,
        }

@dataclass
class ValidationResult:
    """
    Comprehensive result of data validation with quality scoring.
    """
    
    valid: bool
    row_count: int
    column_count: int
    date_range: Optional[Tuple[datetime, datetime]] = None
    missing_values: Dict[str, int] = field(default_factory=dict)
    missing_pct: float = 0.0
    duplicates: int = 0
    duplicate_pct: float = 0.0
    outliers: int = 0
    outlier_pct: float = 0.0
    schema_errors: List[str] = field(default_factory=list)
    range_violations: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    
    @property
    def quality_grade(self) -> QualityGrade:
        """Quality grade based on score."""
        return QualityGrade.from_score(self.quality_score)
    
    @property
    def is_trading_viable(self) -> bool:
        """Whether data is viable for trading."""
        return self.valid and self.quality_grade.trading_viable
    
    @property
    def is_backtest_viable(self) -> bool:
        """Whether data is viable for backtesting."""
        return self.valid and self.quality_grade.backtest_viable
    
    @property
    def coverage_days(self) -> int:
        """Number of days covered by data."""
        if self.date_range:
            return (self.date_range[1] - self.date_range[0]).days + 1
        return 0
    
    @property
    def completeness_score(self) -> float:
        """Completeness score (100 - missing_pct)."""
        return max(0.0, 100.0 - self.missing_pct)
    
    @property
    def uniqueness_score(self) -> float:
        """Uniqueness score (100 - duplicate_pct)."""
        return max(0.0, 100.0 - self.duplicate_pct)
    
    @property
    def summary(self) -> str:
        """Brief summary of validation result."""
        status = "VALID" if self.valid else "INVALID"
        return (
            f"{status} | {self.row_count:,} rows | "
            f"Quality: {self.quality_score:.1f}/100 ({self.quality_grade.value}) | "
            f"Missing: {self.missing_pct:.2f}% | "
            f"Duplicates: {self.duplicate_pct:.2f}%"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            'valid': self.valid,
            'row_count': self.row_count,
            'column_count': self.column_count,
            'date_range': (
                self.date_range[0].isoformat(),
                self.date_range[1].isoformat()
            ) if self.date_range else None,
            'missing_pct': self.missing_pct,
            'duplicate_pct': self.duplicate_pct,
            'outlier_pct': self.outlier_pct,
            'quality_score': self.quality_score,
            'quality_grade': self.quality_grade.value,
            'is_trading_viable': self.is_trading_viable,
            'is_backtest_viable': self.is_backtest_viable,
            'coverage_days': self.coverage_days,
            'warnings': self.warnings,
            'errors': self.errors,
        }

@dataclass
class CollectionMetadata:
    """
    Metadata for a data collection run, used for provenance tracking.
    """
    
    collection_id: str
    venue: str
    venue_type: VenueType
    data_type: DataType
    symbols: List[str]
    start_date: str
    end_date: str
    collected_at: datetime
    collector_version: str
    config_hash: str
    stats: Optional[CollectionStats] = None
    validation: Optional[ValidationResult] = None
    output_path: Optional[str] = None
    
    @property
    def data_hash(self) -> Optional[str]:
        """Hash of collected data for integrity verification."""
        if self.output_path:
            try:
                with open(self.output_path, 'rb') as f:
                    return hashlib.sha256(f.read()).hexdigest()[:16]
            except Exception:
                return None
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'collection_id': self.collection_id,
            'venue': self.venue,
            'venue_type': self.venue_type.value,
            'data_type': self.data_type.value,
            'symbols': self.symbols,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'collected_at': self.collected_at.isoformat(),
            'collector_version': self.collector_version,
            'config_hash': self.config_hash,
            'stats': self.stats.to_dict() if self.stats else None,
            'validation': self.validation.to_dict() if self.validation else None,
            'output_path': self.output_path,
            'data_hash': self.data_hash,
        }

# =============================================================================
# DEFAULT CONFIGURATIONS
# =============================================================================

DEFAULT_COLLECTOR_CONFIG: Dict[str, Any] = {
    'rate_limit_per_minute': 100,
    'max_retries': 3,
    'retry_base_delay': 1.0,
    'retry_max_delay': 60.0,
    'timeout_seconds': 30.0,
    'max_records_per_request': 1000,
    'validate_on_collect': True,
    'compress_output': True,
    'compression_codec': 'gzip',
}

# =============================================================================
# BASE COLLECTOR CLASS
# =============================================================================

class BaseCollector(ABC):
    """
    Abstract base class for all data collectors.
    
    Provides common functionality for venue-specific data collection:
    - Rate limiting integration
    - Retry handling with exponential backoff
    - Data validation with quality scoring
    - Parquet storage with compression
    - Collection statistics and metadata tracking
    
    Subclasses must implement:
    - fetch_funding_rates(): Fetch funding rate data
    - fetch_ohlcv(): Fetch OHLCV price data
    
    Attributes
    ----------
    VENUE : str
        Venue identifier (e.g., 'binance', 'hyperliquid')
    VENUE_TYPE : VenueType
        Venue type classification
    VERSION : str
        Collector version string
    config : Dict
        Configuration dictionary
    logger : logging.Logger
        Logger instance
    stats : CollectionStats
        Collection statistics tracker
    
    Example
    -------
    >>> class BinanceCollector(BaseCollector):
    ... VENUE = 'binance'
    ... VENUE_TYPE = VenueType.CEX
    ... 
    ... async def fetch_funding_rates(self, symbols, start, end):
    ... # Implementation
    ... pass
    ... 
    ... async def fetch_ohlcv(self, symbols, timeframe, start, end):
    ... # Implementation
    ... pass
    """
    
    # Class attributes to be overridden by subclasses
    VENUE: str = "base"
    VENUE_TYPE: VenueType = VenueType.CEX
    VERSION: str = "2.0.0"

    @property
    def venue_type_str(self) -> str:
        """Get venue type as string, handling both enum and string VENUE_TYPE."""
        if hasattr(self.VENUE_TYPE, 'value'):
            return self.VENUE_TYPE.value
        return str(self.VENUE_TYPE)

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the collector.
        
        Parameters
        ----------
        config : Dict, optional
            Configuration dictionary. Merged with DEFAULT_COLLECTOR_CONFIG.
        """
        self.config = {**DEFAULT_COLLECTOR_CONFIG, **(config or {})}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.stats = CollectionStats()
        
        # Rate limiter and retry handler (initialized by subclass or externally)
        self._rate_limiter = None
        self._retry_handler = None
        
        # Session management
        self._session = None
        self._is_closed = False
        
        self.logger.debug(
            f"Initialized {self.__class__.__name__} v{self.VERSION} "
            f"for {self.VENUE} ({self.venue_type_str})"
        )
    
    # -------------------------------------------------------------------------
    # Abstract Methods
    # -------------------------------------------------------------------------
    
    @abstractmethod
    async def fetch_funding_rates(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates.
        
        Parameters
        ----------
        symbols : List[str]
            Symbols to fetch (e.g., ['BTC', 'ETH'])
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
            
        Returns
        -------
        pd.DataFrame
            Funding rate data with required columns:
            timestamp, symbol, funding_rate, venue
        """
        raise NotImplementedError
    
    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch OHLCV price data.
        
        Parameters
        ----------
        symbols : List[str]
            Symbols to fetch
        timeframe : str
            Candle interval ('1m', '5m', '15m', '1h', '4h', '1d')
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
            
        Returns
        -------
        pd.DataFrame
            OHLCV data with required columns:
            timestamp, symbol, open, high, low, close, volume, venue
        """
        raise NotImplementedError
    
    # -------------------------------------------------------------------------
    # Optional Methods (Override in subclass if supported)
    # -------------------------------------------------------------------------
    
    async def fetch_open_interest(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch open interest data (optional)."""
        self.logger.warning(f"{self.VENUE} does not support open interest data")
        return pd.DataFrame()
    
    async def fetch_liquidations(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch liquidation data (optional)."""
        self.logger.warning(f"{self.VENUE} does not support liquidation data")
        return pd.DataFrame()
    
    async def fetch_orderbook(
        self,
        symbol: str,
        depth: int = 20,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch orderbook snapshot (optional)."""
        self.logger.warning(f"{self.VENUE} does not support orderbook data")
        return pd.DataFrame()
    
    # -------------------------------------------------------------------------
    # Validation Methods
    # -------------------------------------------------------------------------
    
    def validate_data(
        self,
        df: pd.DataFrame,
        data_type: DataType,
        max_missing_pct: float = 5.0,
        max_duplicate_pct: float = 1.0
    ) -> ValidationResult:
        """
        Validate collected data with comprehensive quality scoring.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        data_type : DataType
            Expected data type for schema validation
        max_missing_pct : float
            Maximum acceptable missing data percentage
        max_duplicate_pct : float
            Maximum acceptable duplicate percentage
            
        Returns
        -------
        ValidationResult
            Validation result with quality score
        """
        if df.empty:
            return ValidationResult(
                valid=False,
                row_count=0,
                column_count=0,
                quality_score=0.0,
                errors=["Empty DataFrame"]
            )
        
        warnings = []
        errors = []
        schema_errors = []
        
        # Schema validation
        required_cols = data_type.required_columns
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            schema_errors.append(f"Missing columns: {missing_cols}")
        
        # Date range
        date_range = None
        if 'timestamp' in df.columns:
            try:
                ts = pd.to_datetime(df['timestamp'])
                date_range = (ts.min().to_pydatetime(), ts.max().to_pydatetime())
            except Exception:
                warnings.append("Could not parse timestamp column")
        
        # Missing values
        missing_values = df.isnull().sum().to_dict()
        total_cells = df.size
        total_missing = df.isnull().sum().sum()
        missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0.0
        
        if missing_pct > max_missing_pct:
            warnings.append(f"Missing data: {missing_pct:.2f}% exceeds threshold {max_missing_pct}%")
        
        # Duplicates
        duplicates = 0
        duplicate_pct = 0.0
        if 'timestamp' in df.columns and 'symbol' in df.columns:
            duplicates = df.duplicated(['timestamp', 'symbol']).sum()
            duplicate_pct = (duplicates / len(df) * 100) if len(df) > 0 else 0.0
            if duplicate_pct > max_duplicate_pct:
                warnings.append(f"Duplicates: {duplicate_pct:.2f}% exceeds threshold {max_duplicate_pct}%")
        
        # Outliers (IQR method on numeric columns)
        outliers = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == 'timestamp':
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outlier_mask = (df[col] < q1 - 3 * iqr) | (df[col] > q3 + 3 * iqr)
                outliers += outlier_mask.sum()
        
        outlier_pct = (outliers / (len(df) * len(numeric_cols)) * 100) if (len(df) * len(numeric_cols)) > 0 else 0.0
        
        # Calculate quality score
        # Weights: completeness 40%, uniqueness 30%, schema 20%, outliers 10%
        completeness_score = max(0, 100 - missing_pct * 2)
        uniqueness_score = max(0, 100 - duplicate_pct * 5)
        schema_score = 100.0 if not schema_errors else max(0, 100 - len(schema_errors) * 25)
        outlier_score = max(0, 100 - outlier_pct * 2)
        
        quality_score = (
            completeness_score * 0.40 +
            uniqueness_score * 0.30 +
            schema_score * 0.20 +
            outlier_score * 0.10
        )
        
        # Determine validity
        valid = (
            len(errors) == 0 and
            len(schema_errors) == 0 and
            missing_pct <= max_missing_pct and
            duplicate_pct <= max_duplicate_pct
        )
        
        return ValidationResult(
            valid=valid,
            row_count=len(df),
            column_count=len(df.columns),
            date_range=date_range,
            missing_values=missing_values,
            missing_pct=missing_pct,
            duplicates=duplicates,
            duplicate_pct=duplicate_pct,
            outliers=outliers,
            outlier_pct=outlier_pct,
            schema_errors=schema_errors,
            warnings=warnings,
            errors=errors,
            quality_score=quality_score
        )
    
    # -------------------------------------------------------------------------
    # Storage Methods
    # -------------------------------------------------------------------------
    
    def save_to_parquet(
        self,
        df: pd.DataFrame,
        path: Union[str, Path],
        partition_cols: Optional[List[str]] = None,
        compression: str = 'gzip'
    ) -> str:
        """
        Save DataFrame to Parquet format.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to save
        path : str or Path
            Output path
        partition_cols : List[str], optional
            Columns to partition by
        compression : str
            Compression codec ('gzip', 'snappy', 'zstd')
            
        Returns
        -------
        str
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(
            path,
            engine='pyarrow',
            compression=compression,
            partition_cols=partition_cols,
            index=False
        )
        
        self.logger.info(f"Saved {len(df):,} records to {path}")
        return str(path)
    
    @staticmethod
    def load_from_parquet(
        path: Union[str, Path],
        columns: Optional[List[str]] = None,
        filters: Optional[List[Tuple]] = None
    ) -> pd.DataFrame:
        """Load DataFrame from Parquet format."""
        return pd.read_parquet(
            path,
            engine='pyarrow',
            columns=columns,
            filters=filters
        )
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    @staticmethod
    def parse_date(date_str: str) -> datetime:
        """Parse date string (YYYY-MM-DD) to datetime."""
        return datetime.strptime(date_str, '%Y-%m-%d')
    
    @staticmethod
    def date_to_timestamp_ms(date_str: str) -> int:
        """Convert date string to millisecond timestamp."""
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    
    @staticmethod
    def timestamp_ms_to_datetime(ts: int) -> datetime:
        """Convert millisecond timestamp to datetime."""
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    
    def add_venue_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add venue and venue_type columns to DataFrame."""
        df = df.copy()
        df['venue'] = self.VENUE
        df['venue_type'] = self.venue_type_str
        return df
    
    def create_metadata(
        self,
        data_type: DataType,
        symbols: List[str],
        start_date: str,
        end_date: str,
        output_path: Optional[str] = None
    ) -> CollectionMetadata:
        """Create collection metadata for provenance tracking."""
        config_str = str(sorted(self.config.items()))
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return CollectionMetadata(
            collection_id=f"{self.VENUE}_{data_type.value}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            venue=self.VENUE,
            venue_type=self.VENUE_TYPE,
            data_type=data_type,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            collected_at=datetime.now(timezone.utc),
            collector_version=self.VERSION,
            config_hash=config_hash,
            stats=self.stats,
            output_path=output_path
        )
    
    # -------------------------------------------------------------------------
    # Statistics Methods
    # -------------------------------------------------------------------------
    
    def reset_stats(self) -> None:
        """Reset collection statistics."""
        self.stats = CollectionStats()
    
    def start_collection(self) -> None:
        """Mark start of collection run."""
        self.stats.start_time = datetime.now(timezone.utc)
        self.stats.phase = CollectionPhase.FETCHING
        self.logger.info(f"Starting data collection for {self.VENUE}")
    
    def end_collection(self, success: bool = True) -> None:
        """Mark end of collection run."""
        self.stats.end_time = datetime.now(timezone.utc)
        self.stats.phase = CollectionPhase.COMPLETED if success else CollectionPhase.FAILED
        
        self.logger.info(
            f"Collection {'complete' if success else 'failed'}: "
            f"{self.stats.records_collected:,} records in "
            f"{self.stats.duration_seconds:.2f}s "
            f"({self.stats.records_per_second:.2f} rec/s)"
        )
    
    # -------------------------------------------------------------------------
    # Context Manager and Cleanup
    # -------------------------------------------------------------------------
    
    async def close(self) -> None:
        """
        Clean up resources with proper connection draining.

        CRITICAL: This ensures aiohttp sessions close all connections properly,
        preventing event loop hanging.
        """
        if self._session and not self._is_closed:
            if hasattr(self._session, 'close'):
                await self._session.close()
                # CRITICAL: Wait for underlying connections to close
                # This prevents "Unclosed client session" warnings and event loop hanging
                await asyncio.sleep(0.25)
            self._is_closed = True
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"venue={self.VENUE}, "
            f"type={self.venue_type_str}, "
            f"version={self.VERSION})"
        )