"""
Storage Module for Crypto Statistical Arbitrage Systems.

This module provides professional-quality data storage infrastructure
specifically designed for time-series financial data with support for
efficient partitioning, compression, and trading-optimized retrieval.

==============================================================================
STORAGE ARCHITECTURE
==============================================================================

Data Organization:
    data/
    â”œâ”€â”€ hot/ # Recent data (< 7 days)
    â”‚ â”œâ”€â”€ funding_rates/
    â”‚ â”œâ”€â”€ ohlcv/
    â”‚ â””â”€â”€ orderbook/
    â”œâ”€â”€ warm/ # Medium-term (7-90 days)
    â”‚ â”œâ”€â”€ funding_rates/
    â”‚ â””â”€â”€ ohlcv/
    â”œâ”€â”€ cold/ # Historical (90+ days)
    â”‚ â””â”€â”€ archived/
    â””â”€â”€ .metadata/ # Index and catalog files

Parquet vs CSV Performance:
+------------------+------------------+------------------+------------------+
| Metric | Parquet (gzip) | Parquet (zstd) | CSV (gzip) |
+------------------+------------------+------------------+------------------+
| Compression | 10-20x | 15-25x | 3-5x |
| Read Speed | Very Fast | Fast | Slow |
| Column Pruning | Yes | Yes | No |
| Predicate Push | Yes | Yes | No |
| Schema Support | Yes | Yes | No |
+------------------+------------------+------------------+------------------+

==============================================================================
PARTITIONING STRATEGIES
==============================================================================

Strategy Selection by Data Type:
+------------------+------------------+------------------+------------------+
| Data Type | Primary Part | Secondary Part | Row Group Size |
+------------------+------------------+------------------+------------------+
| Funding Rates | venue | date | 50,000 |
| OHLCV | symbol | year_month | 100,000 |
| Open Interest | venue | date | 50,000 |
| Orderbook | symbol | hour | 10,000 |
| Trades | symbol | date | 500,000 |
+------------------+------------------+------------------+------------------+

Partition Pruning Benefits:
    - Query: SELECT * WHERE symbol='BTC' AND date='2024-01-15'
    - Without partitioning: Scan all files
    - With symbol/date partitioning: Scan only 1 file
    - Speed improvement: 10-100x for targeted queries

==============================================================================
DATA TIER MANAGEMENT
==============================================================================

Hot Tier (< 7 days):
    - Purpose: Real-time trading signals
    - Storage: SSD/NVMe
    - Compression: snappy (fast)
    - Access pattern: Frequent reads/writes

Warm Tier (7-90 days):
    - Purpose: Backtesting, signal validation
    - Storage: SSD
    - Compression: zstd level 3
    - Access pattern: Batch reads

Cold Tier (90+ days):
    - Purpose: Historical analysis, compliance
    - Storage: HDD/Object storage
    - Compression: zstd level 9
    - Access pattern: Infrequent reads

Archive Tier (1+ years):
    - Purpose: Compliance, research
    - Storage: Object storage (S3/GCS)
    - Compression: Maximum
    - Access pattern: Rare reads

==============================================================================
STATISTICAL ARBITRAGE IMPLICATIONS
==============================================================================

1. DATA FRESHNESS
   - Hot tier latency: < 10ms read
   - Signal generation requires < 100ms data access
   - Partition by time enables fast range queries

2. BACKTEST EFFICIENCY
   - Warm tier optimized for sequential reads
   - Column pruning reduces I/O by 60-80%
   - Predicate pushdown eliminates unnecessary data

3. CROSS-VENUE ANALYSIS
   - Partition by venue enables isolated venue queries
   - Merge operations benefit from sorted data
   - Time alignment requires consistent partitioning

4. STORAGE COSTS
   - Hot: 10% of data, 50% of storage cost
   - Warm: 30% of data, 35% of storage cost
   - Cold: 60% of data, 15% of storage cost

==============================================================================
USAGE EXAMPLES
==============================================================================

Basic Storage:
    >>> storage = ParquetStorage(base_path='data/')
    >>> storage.save(df, 'funding_rates/binance/btc.parquet')
    >>> df = storage.load('funding_rates/binance/btc.parquet')

Partitioned Storage:
    >>> storage.save_partitioned(
    ... df, 'funding_rates',
    ... partition_cols=['venue', 'date'],
    ... compression='zstd'
    ... )

Query with Filters:
    >>> df = storage.query(
    ... 'funding_rates',
    ... columns=['timestamp', 'symbol', 'funding_rate'],
    ... filters=[('symbol', '==', 'BTC'), ('date', '>=', '2024-01-01')]
    ... )

Tiered Storage:
    >>> tiered = TieredStorage(base_path='data/')
    >>> tiered.save(df, 'funding_rates', tier=DataTier.HOT)
    >>> tiered.migrate_to_warm(older_than_days=7)

Version: 2.0.0
"""

import gzip
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    pa = None
    pq = None
    PYARROW_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# ENUMS
# =============================================================================

class DataTier(Enum):
    """
    Data tier classification for tiered storage.

    Different tiers optimize for different access patterns
    and cost considerations.
    """

    HOT = "hot" # Recent, frequently accessed
    WARM = "warm" # Medium-term, batch access
    COLD = "cold" # Historical, infrequent access
    ARCHIVE = "archive" # Long-term, rare access

    @property
    def retention_days(self) -> int:
        """Default retention in this tier before migration."""
        retentions = {
            DataTier.HOT: 7,
            DataTier.WARM: 90,
            DataTier.COLD: 365,
            DataTier.ARCHIVE: float('inf'),
        }
        return retentions.get(self, 90)

    @property
    def compression(self) -> str:
        """Recommended compression for this tier."""
        compressions = {
            DataTier.HOT: 'snappy',
            DataTier.WARM: 'zstd',
            DataTier.COLD: 'zstd',
            DataTier.ARCHIVE: 'zstd',
        }
        return compressions.get(self, 'zstd')

    @property
    def compression_level(self) -> Optional[int]:
        """Compression level for this tier."""
        levels = {
            DataTier.HOT: None,
            DataTier.WARM: 3,
            DataTier.COLD: 9,
            DataTier.ARCHIVE: 19,
        }
        return levels.get(self, 3)

    @property
    def row_group_size(self) -> int:
        """Recommended row group size."""
        sizes = {
            DataTier.HOT: 50000,
            DataTier.WARM: 100000,
            DataTier.COLD: 200000,
            DataTier.ARCHIVE: 500000,
        }
        return sizes.get(self, 100000)

    @property
    def expected_read_latency_ms(self) -> float:
        """Expected read latency in milliseconds."""
        latencies = {
            DataTier.HOT: 5.0,
            DataTier.WARM: 20.0,
            DataTier.COLD: 100.0,
            DataTier.ARCHIVE: 500.0,
        }
        return latencies.get(self, 50.0)

    @property
    def cost_multiplier(self) -> float:
        """Relative storage cost multiplier."""
        costs = {
            DataTier.HOT: 4.0,
            DataTier.WARM: 2.0,
            DataTier.COLD: 1.0,
            DataTier.ARCHIVE: 0.25,
        }
        return costs.get(self, 1.0)

class PartitionStrategy(Enum):
    """
    Partitioning strategies for time-series data.
    """

    NONE = "none"
    DAILY = "daily"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    BY_SYMBOL = "symbol"
    BY_VENUE = "venue"
    BY_SYMBOL_DAILY = "symbol_daily"
    BY_VENUE_DAILY = "venue_daily"
    BY_SYMBOL_MONTHLY = "symbol_monthly"
    BY_VENUE_MONTHLY = "venue_monthly"

    @property
    def partition_columns(self) -> List[str]:
        """Columns used for partitioning."""
        columns = {
            PartitionStrategy.NONE: [],
            PartitionStrategy.DAILY: ['date'],
            PartitionStrategy.MONTHLY: ['year_month'],
            PartitionStrategy.YEARLY: ['year'],
            PartitionStrategy.BY_SYMBOL: ['symbol'],
            PartitionStrategy.BY_VENUE: ['venue'],
            PartitionStrategy.BY_SYMBOL_DAILY: ['symbol', 'date'],
            PartitionStrategy.BY_VENUE_DAILY: ['venue', 'date'],
            PartitionStrategy.BY_SYMBOL_MONTHLY: ['symbol', 'year_month'],
            PartitionStrategy.BY_VENUE_MONTHLY: ['venue', 'year_month'],
        }
        return columns.get(self, [])

    @property
    def requires_date_column(self) -> bool:
        """Whether strategy requires adding date partition column."""
        return self in {
            PartitionStrategy.DAILY,
            PartitionStrategy.MONTHLY,
            PartitionStrategy.YEARLY,
            PartitionStrategy.BY_SYMBOL_DAILY,
            PartitionStrategy.BY_VENUE_DAILY,
            PartitionStrategy.BY_SYMBOL_MONTHLY,
            PartitionStrategy.BY_VENUE_MONTHLY,
        }

    @property
    def description(self) -> str:
        """Strategy description."""
        descriptions = {
            PartitionStrategy.NONE: "No partitioning, single file",
            PartitionStrategy.DAILY: "Partition by date (YYYY-MM-DD)",
            PartitionStrategy.MONTHLY: "Partition by month (YYYY-MM)",
            PartitionStrategy.YEARLY: "Partition by year (YYYY)",
            PartitionStrategy.BY_SYMBOL: "Partition by trading symbol",
            PartitionStrategy.BY_VENUE: "Partition by venue",
            PartitionStrategy.BY_SYMBOL_DAILY: "Symbol + daily partitions",
            PartitionStrategy.BY_VENUE_DAILY: "Venue + daily partitions",
            PartitionStrategy.BY_SYMBOL_MONTHLY: "Symbol + monthly partitions",
            PartitionStrategy.BY_VENUE_MONTHLY: "Venue + monthly partitions",
        }
        return descriptions.get(self, "Unknown")

class CompressionLevel(Enum):
    """
    Compression level presets.
    """

    NONE = "none"
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"
    MAXIMUM = "maximum"

    @property
    def codec(self) -> Optional[str]:
        """Compression codec."""
        codecs = {
            CompressionLevel.NONE: None,
            CompressionLevel.FAST: 'snappy',
            CompressionLevel.BALANCED: 'zstd',
            CompressionLevel.HIGH: 'zstd',
            CompressionLevel.MAXIMUM: 'zstd',
        }
        return codecs.get(self, 'zstd')

    @property
    def level(self) -> Optional[int]:
        """Compression level."""
        levels = {
            CompressionLevel.NONE: None,
            CompressionLevel.FAST: None,
            CompressionLevel.BALANCED: 3,
            CompressionLevel.HIGH: 9,
            CompressionLevel.MAXIMUM: 19,
        }
        return levels.get(self, 3)

    @property
    def expected_ratio(self) -> float:
        """Expected compression ratio."""
        ratios = {
            CompressionLevel.NONE: 1.0,
            CompressionLevel.FAST: 3.0,
            CompressionLevel.BALANCED: 5.0,
            CompressionLevel.HIGH: 7.0,
            CompressionLevel.MAXIMUM: 10.0,
        }
        return ratios.get(self, 5.0)

class RetentionPolicy(Enum):
    """
    Data retention policies.
    """

    KEEP_ALL = "keep_all"
    ROLLING_7D = "rolling_7d"
    ROLLING_30D = "rolling_30d"
    ROLLING_90D = "rolling_90d"
    ROLLING_1Y = "rolling_1y"
    TIERED = "tiered"

    @property
    def retention_days(self) -> Optional[int]:
        """Retention period in days (None = forever)."""
        days = {
            RetentionPolicy.KEEP_ALL: None,
            RetentionPolicy.ROLLING_7D: 7,
            RetentionPolicy.ROLLING_30D: 30,
            RetentionPolicy.ROLLING_90D: 90,
            RetentionPolicy.ROLLING_1Y: 365,
            RetentionPolicy.TIERED: None,
        }
        return days.get(self, None)

    @property
    def description(self) -> str:
        """Policy description."""
        descriptions = {
            RetentionPolicy.KEEP_ALL: "Keep all data indefinitely",
            RetentionPolicy.ROLLING_7D: "Keep 7 days of data",
            RetentionPolicy.ROLLING_30D: "Keep 30 days of data",
            RetentionPolicy.ROLLING_90D: "Keep 90 days of data",
            RetentionPolicy.ROLLING_1Y: "Keep 1 year of data",
            RetentionPolicy.TIERED: "Migrate between tiers based on age",
        }
        return descriptions.get(self, "Unknown")

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class StorageMetrics:
    """
    Comprehensive storage metrics for monitoring.
    """

    total_files: int = 0
    total_size_bytes: int = 0
    total_rows: int = 0
    files_by_tier: Dict[str, int] = field(default_factory=dict)
    size_by_tier: Dict[str, int] = field(default_factory=dict)
    oldest_data: Optional[datetime] = None
    newest_data: Optional[datetime] = None
    last_write: Optional[datetime] = None
    last_read: Optional[datetime] = None

    @property
    def total_size_mb(self) -> float:
        """Total size in megabytes."""
        return self.total_size_bytes / (1024 * 1024)

    @property
    def total_size_gb(self) -> float:
        """Total size in gigabytes."""
        return self.total_size_bytes / (1024 * 1024 * 1024)

    @property
    def average_file_size_mb(self) -> float:
        """Average file size in MB."""
        if self.total_files == 0:
            return 0.0
        return self.total_size_mb / self.total_files

    @property
    def data_coverage_days(self) -> Optional[int]:
        """Number of days of data coverage."""
        if self.oldest_data is None or self.newest_data is None:
            return None
        delta = self.newest_data - self.oldest_data
        return delta.days

    @property
    def estimated_monthly_cost(self) -> float:
        """Estimated monthly storage cost (assuming $0.02/GB/month)."""
        return self.total_size_gb * 0.02

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_files': self.total_files,
            'total_size_bytes': self.total_size_bytes,
            'total_size_mb': round(self.total_size_mb, 2),
            'total_size_gb': round(self.total_size_gb, 4),
            'total_rows': self.total_rows,
            'average_file_size_mb': round(self.average_file_size_mb, 2),
            'files_by_tier': self.files_by_tier,
            'size_by_tier': self.size_by_tier,
            'oldest_data': self.oldest_data.isoformat() if self.oldest_data else None,
            'newest_data': self.newest_data.isoformat() if self.newest_data else None,
            'data_coverage_days': self.data_coverage_days,
            'last_write': self.last_write.isoformat() if self.last_write else None,
            'last_read': self.last_read.isoformat() if self.last_read else None,
            'estimated_monthly_cost': round(self.estimated_monthly_cost, 4),
        }

@dataclass
class FileMetadata:
    """
    Metadata for a stored file.
    """

    path: str
    size_bytes: int
    row_count: int
    columns: List[str]
    min_timestamp: Optional[datetime]
    max_timestamp: Optional[datetime]
    symbols: List[str]
    venues: List[str]
    tier: DataTier
    compression: str
    created_at: datetime
    last_accessed: Optional[datetime] = None

    @property
    def size_mb(self) -> float:
        """Size in megabytes."""
        return self.size_bytes / (1024 * 1024)

    @property
    def age_days(self) -> float:
        """Age in days since creation."""
        delta = datetime.now(timezone.utc) - self.created_at
        return delta.total_seconds() / 86400

    @property
    def data_age_days(self) -> Optional[float]:
        """Age of newest data in file."""
        if self.max_timestamp is None:
            return None
        delta = datetime.now(timezone.utc) - self.max_timestamp
        return delta.total_seconds() / 86400

    @property
    def should_migrate_to_warm(self) -> bool:
        """Whether file should migrate from hot to warm."""
        if self.tier != DataTier.HOT:
            return False
        return self.data_age_days is not None and self.data_age_days > 7

    @property
    def should_migrate_to_cold(self) -> bool:
        """Whether file should migrate from warm to cold."""
        if self.tier != DataTier.WARM:
            return False
        return self.data_age_days is not None and self.data_age_days > 90

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'path': self.path,
            'size_bytes': self.size_bytes,
            'size_mb': round(self.size_mb, 2),
            'row_count': self.row_count,
            'columns': self.columns,
            'min_timestamp': self.min_timestamp.isoformat() if self.min_timestamp else None,
            'max_timestamp': self.max_timestamp.isoformat() if self.max_timestamp else None,
            'symbols': self.symbols,
            'venues': self.venues,
            'tier': self.tier.value,
            'compression': self.compression,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'age_days': round(self.age_days, 2),
            'data_age_days': round(self.data_age_days, 2) if self.data_age_days else None,
        }

@dataclass
class QueryResult:
    """
    Result from a storage query.
    """

    df: pd.DataFrame
    query_time_ms: float
    files_scanned: int
    rows_scanned: int
    rows_returned: int
    bytes_read: int
    cache_hit: bool
    partitions_pruned: int
    columns_pruned: int

    @property
    def selectivity(self) -> float:
        """Selectivity ratio (rows returned / rows scanned)."""
        if self.rows_scanned == 0:
            return 0.0
        return self.rows_returned / self.rows_scanned

    @property
    def bytes_per_row(self) -> float:
        """Average bytes per row returned."""
        if self.rows_returned == 0:
            return 0.0
        return self.bytes_read / self.rows_returned

    @property
    def rows_per_second(self) -> float:
        """Query throughput in rows per second."""
        if self.query_time_ms == 0:
            return 0.0
        return self.rows_returned / (self.query_time_ms / 1000)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes DataFrame)."""
        return {
            'row_count': len(self.df),
            'query_time_ms': round(self.query_time_ms, 2),
            'files_scanned': self.files_scanned,
            'rows_scanned': self.rows_scanned,
            'rows_returned': self.rows_returned,
            'bytes_read': self.bytes_read,
            'cache_hit': self.cache_hit,
            'partitions_pruned': self.partitions_pruned,
            'columns_pruned': self.columns_pruned,
            'selectivity': round(self.selectivity, 4),
            'bytes_per_row': round(self.bytes_per_row, 2),
            'rows_per_second': round(self.rows_per_second, 0),
        }

# =============================================================================
# LRU CACHE
# =============================================================================

class LRUCache:
    """
    Thread-safe LRU cache for query results.
    """

    def __init__(self, max_size: int = 100, max_memory_mb: float = 500.0):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.cache: OrderedDict[str, Tuple[pd.DataFrame, float]] = OrderedDict()
        self.lock = threading.Lock()
        self._memory_usage_mb = 0.0
        self._hits = 0
        self._misses = 0

    def _compute_key(
        self,
        path: str,
        columns: Optional[List[str]],
        filters: Optional[List[Tuple]],
    ) -> str:
        """Compute cache key."""
        key_parts = [path]
        if columns:
            key_parts.append(str(sorted(columns)))
        if filters:
            key_parts.append(str(filters))
        return hashlib.md5('|'.join(key_parts).encode()).hexdigest()

    def _estimate_df_size_mb(self, df: pd.DataFrame) -> float:
        """Estimate DataFrame memory usage."""
        return df.memory_usage(deep=True).sum() / (1024 * 1024)

    def get(
        self,
        path: str,
        columns: Optional[List[str]] = None,
        filters: Optional[List[Tuple]] = None,
    ) -> Optional[pd.DataFrame]:
        """Get cached result."""
        key = self._compute_key(path, columns, filters)

        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                df, _ = self.cache[key]
                self._hits += 1
                return df.copy()
            self._misses += 1
        return None

    def put(
        self,
        path: str,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        filters: Optional[List[Tuple]] = None,
    ) -> None:
        """Add result to cache."""
        if df.empty:
            return

        key = self._compute_key(path, columns, filters)
        size_mb = self._estimate_df_size_mb(df)

        if size_mb > self.max_memory_mb * 0.5:
            return

        with self.lock:
            while (
                len(self.cache) >= self.max_size or
                self._memory_usage_mb + size_mb > self.max_memory_mb
            ):
                if not self.cache:
                    break
                _, (_, old_size) = self.cache.popitem(last=False)
                self._memory_usage_mb -= old_size

            self.cache[key] = (df.copy(), size_mb)
            self._memory_usage_mb += size_mb

    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self._memory_usage_mb = 0.0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                'items': len(self.cache),
                'max_items': self.max_size,
                'memory_usage_mb': round(self._memory_usage_mb, 2),
                'max_memory_mb': self.max_memory_mb,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': round(hit_rate, 4),
            }

# =============================================================================
# PARQUET STORAGE
# =============================================================================

class ParquetStorage:
    """
    Parquet-based storage manager for crypto data.

    Provides efficient storage with compression, partitioning,
    and query optimization for time-series financial data.
    """

    def __init__(
        self,
        base_path: Union[str, Path],
        compression: str = 'zstd',
        compression_level: Optional[int] = 3,
        row_group_size: int = 100000,
        enable_cache: bool = True,
        cache_size: int = 100,
        cache_memory_mb: float = 500.0,
    ):
        if not PYARROW_AVAILABLE:
            logger.warning(
                "PyArrow not installed. Using pandas fallback. "
                "Install pyarrow for better performance."
            )

        self.base_path = Path(base_path)
        self.compression = compression
        self.compression_level = compression_level
        self.row_group_size = row_group_size

        self.base_path.mkdir(parents=True, exist_ok=True)

        # Cache
        self.enable_cache = enable_cache
        self.cache = LRUCache(cache_size, cache_memory_mb) if enable_cache else None

        # Catalog
        self.catalog: Dict[str, FileMetadata] = {}
        self._load_catalog()

        # Metrics
        self.metrics = StorageMetrics()
        self._update_metrics()

        logger.debug(
            f"ParquetStorage initialized: base_path={self.base_path}, "
            f"compression={compression}"
        )

    def _get_full_path(self, relative_path: str) -> Path:
        """Get full path from relative path."""
        return self.base_path / relative_path

    def _load_catalog(self) -> None:
        """Load catalog from disk."""
        catalog_path = self.base_path / '.catalog.json'
        if catalog_path.exists():
            try:
                with open(catalog_path) as f:
                    data = json.load(f)
                for path, entry in data.items():
                    self.catalog[path] = FileMetadata(
                        path=entry['path'],
                        size_bytes=entry['size_bytes'],
                        row_count=entry['row_count'],
                        columns=entry['columns'],
                        min_timestamp=datetime.fromisoformat(entry['min_timestamp']) if entry.get('min_timestamp') else None,
                        max_timestamp=datetime.fromisoformat(entry['max_timestamp']) if entry.get('max_timestamp') else None,
                        symbols=entry.get('symbols', []),
                        venues=entry.get('venues', []),
                        tier=DataTier(entry.get('tier', 'warm')),
                        compression=entry.get('compression', 'zstd'),
                        created_at=datetime.fromisoformat(entry['created_at']),
                        last_accessed=datetime.fromisoformat(entry['last_accessed']) if entry.get('last_accessed') else None,
                    )
            except Exception as e:
                logger.warning(f"Error loading catalog: {e}")
                self.catalog = {}

    def _save_catalog(self) -> None:
        """Save catalog to disk."""
        catalog_path = self.base_path / '.catalog.json'
        data = {}
        for path, meta in self.catalog.items():
            data[path] = {
                'path': meta.path,
                'size_bytes': meta.size_bytes,
                'row_count': meta.row_count,
                'columns': meta.columns,
                'min_timestamp': meta.min_timestamp.isoformat() if meta.min_timestamp else None,
                'max_timestamp': meta.max_timestamp.isoformat() if meta.max_timestamp else None,
                'symbols': meta.symbols,
                'venues': meta.venues,
                'tier': meta.tier.value,
                'compression': meta.compression,
                'created_at': meta.created_at.isoformat(),
                'last_accessed': meta.last_accessed.isoformat() if meta.last_accessed else None,
            }
        with open(catalog_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _update_metrics(self) -> None:
        """Update storage metrics."""
        self.metrics = StorageMetrics()

        for meta in self.catalog.values():
            self.metrics.total_files += 1
            self.metrics.total_size_bytes += meta.size_bytes
            self.metrics.total_rows += meta.row_count

            tier_key = meta.tier.value
            self.metrics.files_by_tier[tier_key] = self.metrics.files_by_tier.get(tier_key, 0) + 1
            self.metrics.size_by_tier[tier_key] = self.metrics.size_by_tier.get(tier_key, 0) + meta.size_bytes

            if meta.min_timestamp:
                if self.metrics.oldest_data is None or meta.min_timestamp < self.metrics.oldest_data:
                    self.metrics.oldest_data = meta.min_timestamp

            if meta.max_timestamp:
                if self.metrics.newest_data is None or meta.max_timestamp > self.metrics.newest_data:
                    self.metrics.newest_data = meta.max_timestamp

    def _add_partition_columns(
        self,
        df: pd.DataFrame,
        strategy: PartitionStrategy,
    ) -> pd.DataFrame:
        """Add partition columns based on strategy."""
        if not strategy.requires_date_column:
            return df

        df = df.copy()

        if 'timestamp' not in df.columns:
            return df

        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        if strategy in [PartitionStrategy.DAILY, PartitionStrategy.BY_SYMBOL_DAILY, PartitionStrategy.BY_VENUE_DAILY]:
            df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
        elif strategy in [PartitionStrategy.MONTHLY, PartitionStrategy.BY_SYMBOL_MONTHLY, PartitionStrategy.BY_VENUE_MONTHLY]:
            df['year_month'] = df['timestamp'].dt.strftime('%Y-%m')
        elif strategy == PartitionStrategy.YEARLY:
            df['year'] = df['timestamp'].dt.strftime('%Y')

        return df

    def save(
        self,
        df: pd.DataFrame,
        relative_path: str,
        tier: DataTier = DataTier.WARM,
        partition_strategy: PartitionStrategy = PartitionStrategy.NONE,
        compression_level: Optional[CompressionLevel] = None,
    ) -> str:
        """
        Save DataFrame to Parquet.
        """
        if df.empty:
            logger.warning(f"Empty DataFrame, skipping save to {relative_path}")
            return ""

        # Add partition columns
        df = self._add_partition_columns(df, partition_strategy)

        # Sort by timestamp if present
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')

        full_path = self._get_full_path(relative_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine compression
        comp = compression_level or CompressionLevel.BALANCED
        codec = comp.codec or self.compression
        level = comp.level or self.compression_level

        partition_cols = partition_strategy.partition_columns

        if not PYARROW_AVAILABLE:
            # Pandas fallback
            df.to_parquet(
                str(full_path),
                engine='auto',
                compression=self.compression,
                index=False,
            )
        elif partition_cols:
            # Partitioned write
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_to_dataset(
                table,
                root_path=str(full_path),
                partition_cols=partition_cols,
                compression=codec,
                compression_level=level,
                existing_data_behavior='overwrite_or_ignore',
            )
        else:
            # Single file write
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(
                table,
                str(full_path),
                compression=codec,
                compression_level=level,
                row_group_size=self.row_group_size,
            )

        # Update catalog
        min_ts = df['timestamp'].min() if 'timestamp' in df.columns else None
        max_ts = df['timestamp'].max() if 'timestamp' in df.columns else None
        symbols = df['symbol'].unique().tolist() if 'symbol' in df.columns else []
        venues = df['venue'].unique().tolist() if 'venue' in df.columns else []

        # Calculate size
        if full_path.is_file():
            size_bytes = full_path.stat().st_size
        else:
            size_bytes = sum(f.stat().st_size for f in full_path.rglob('*.parquet'))

        self.catalog[relative_path] = FileMetadata(
            path=str(full_path),
            size_bytes=size_bytes,
            row_count=len(df),
            columns=list(df.columns),
            min_timestamp=min_ts if pd.notna(min_ts) else None,
            max_timestamp=max_ts if pd.notna(max_ts) else None,
            symbols=symbols,
            venues=venues,
            tier=tier,
            compression=codec or 'none',
            created_at=datetime.now(timezone.utc),
        )

        self._save_catalog()
        self._update_metrics()
        self.metrics.last_write = datetime.now(timezone.utc)

        # Clear cache
        if self.cache:
            self.cache.clear()

        logger.info(f"Saved {len(df):,} rows to {relative_path}")
        return str(full_path)

    def load(
        self,
        relative_path: str,
        columns: Optional[List[str]] = None,
        filters: Optional[List[Tuple]] = None,
    ) -> pd.DataFrame:
        """
        Load DataFrame from Parquet.
        """
        # Check cache
        if self.cache:
            cached = self.cache.get(relative_path, columns, filters)
            if cached is not None:
                logger.debug(f"Cache hit for {relative_path}")
                return cached

        full_path = self._get_full_path(relative_path)

        if not full_path.exists():
            logger.warning(f"Path does not exist: {full_path}")
            return pd.DataFrame()

        try:
            df = pd.read_parquet(
                full_path,
                columns=columns,
                filters=filters,
                engine='pyarrow' if PYARROW_AVAILABLE else 'auto',
            )

            # Update cache
            if self.cache:
                self.cache.put(relative_path, df, columns, filters)

            # Update access time
            if relative_path in self.catalog:
                self.catalog[relative_path].last_accessed = datetime.now(timezone.utc)

            self.metrics.last_read = datetime.now(timezone.utc)
            logger.debug(f"Loaded {len(df):,} rows from {relative_path}")
            return df

        except Exception as e:
            logger.error(f"Error loading {relative_path}: {e}")
            return pd.DataFrame()

    def query(
        self,
        name: str,
        columns: Optional[List[str]] = None,
        filters: Optional[List[Tuple]] = None,
        use_cache: bool = True,
    ) -> QueryResult:
        """
        Query data with metrics.
        """
        start_time = time.time()

        # Check cache
        cache_hit = False
        if use_cache and self.cache:
            cached = self.cache.get(name, columns, filters)
            if cached is not None:
                cache_hit = True
                query_time_ms = (time.time() - start_time) * 1000
                return QueryResult(
                    df=cached,
                    query_time_ms=query_time_ms,
                    files_scanned=0,
                    rows_scanned=len(cached),
                    rows_returned=len(cached),
                    bytes_read=cached.memory_usage(deep=True).sum(),
                    cache_hit=True,
                    partitions_pruned=0,
                    columns_pruned=0,
                )

        # Load data
        df = self.load(name, columns, filters)

        query_time_ms = (time.time() - start_time) * 1000

        # Estimate metrics
        files_scanned = 1
        if name in self.catalog:
            rows_scanned = self.catalog[name].row_count
        else:
            rows_scanned = len(df)

        bytes_read = df.memory_usage(deep=True).sum() if not df.empty else 0

        return QueryResult(
            df=df,
            query_time_ms=query_time_ms,
            files_scanned=files_scanned,
            rows_scanned=rows_scanned,
            rows_returned=len(df),
            bytes_read=bytes_read,
            cache_hit=cache_hit,
            partitions_pruned=0,
            columns_pruned=len(columns) if columns else 0,
        )

    def exists(self, relative_path: str) -> bool:
        """Check if path exists."""
        return self._get_full_path(relative_path).exists()

    def delete(self, relative_path: str) -> bool:
        """Delete file or directory."""
        full_path = self._get_full_path(relative_path)

        if not full_path.exists():
            return False

        try:
            if full_path.is_file():
                full_path.unlink()
            else:
                shutil.rmtree(full_path)

            if relative_path in self.catalog:
                del self.catalog[relative_path]
                self._save_catalog()
                self._update_metrics()

            if self.cache:
                self.cache.clear()

            logger.info(f"Deleted {relative_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting {relative_path}: {e}")
            return False

    def list_files(
        self,
        relative_path: str = "",
        pattern: str = "*.parquet",
    ) -> List[str]:
        """List files in directory."""
        full_path = self._get_full_path(relative_path)

        if not full_path.exists():
            return []

        files = list(full_path.glob(f"**/{pattern}"))
        return [str(f.relative_to(self.base_path)) for f in files]

    def get_info(self, relative_path: str) -> Optional[FileMetadata]:
        """Get file metadata."""
        return self.catalog.get(relative_path)

    def get_metrics(self) -> StorageMetrics:
        """Get storage metrics."""
        self._update_metrics()
        return self.metrics

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.cache:
            return self.cache.stats()
        return {'enabled': False}

    def clear_cache(self) -> None:
        """Clear query cache."""
        if self.cache:
            self.cache.clear()

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_storage(
    base_path: str = 'data',
    enable_cache: bool = True,
) -> ParquetStorage:
    """Create storage with default configuration."""
    return ParquetStorage(
        base_path=base_path,
        compression='zstd',
        compression_level=3,
        row_group_size=100000,
        enable_cache=enable_cache,
    )

def create_tiered_storage(base_path: str = 'data') -> Dict[DataTier, ParquetStorage]:
    """Create tiered storage with separate paths per tier."""
    return {
        DataTier.HOT: ParquetStorage(
            base_path=f'{base_path}/hot',
            compression='snappy',
            compression_level=None,
            row_group_size=50000,
        ),
        DataTier.WARM: ParquetStorage(
            base_path=f'{base_path}/warm',
            compression='zstd',
            compression_level=3,
            row_group_size=100000,
        ),
        DataTier.COLD: ParquetStorage(
            base_path=f'{base_path}/cold',
            compression='zstd',
            compression_level=9,
            row_group_size=200000,
        ),
    }

class OptimizedStorage(ParquetStorage):
    """
    Extended ParquetStorage with optimized save/query capabilities.

    Provides additional methods for optimized storage operations
    including automatic partitioning and query filtering.
    """

    def save_optimized(
        self,
        df: pd.DataFrame,
        relative_path: str,
        partition_strategy: PartitionStrategy = PartitionStrategy.NONE,
        tier: DataTier = DataTier.WARM,
    ) -> str:
        """
        Save DataFrame with optimized settings.

        Args:
            df: DataFrame to save
            relative_path: Relative path for storage
            partition_strategy: Partitioning strategy
            tier: Storage tier

        Returns:
            Full path to saved data
        """
        return self.save(
            df,
            relative_path,
            tier=tier,
            partition_strategy=partition_strategy,
            compression_level=CompressionLevel.BALANCED,
        )

    def query(
        self,
        name: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Union[List[Tuple], Dict[str, Any]]] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Query data with optional filtering.

        Args:
            name: Data name/path
            columns: Columns to select
            filters: Filter conditions (dict or list of tuples)
            use_cache: Whether to use cache

        Returns:
            Filtered DataFrame
        """
        # Convert dict filters to list of tuples for parquet
        filter_list = None
        if isinstance(filters, dict):
            filter_list = [(k, '==', v) for k, v in filters.items()]
        elif filters:
            filter_list = filters

        # Load data
        df = self.load(name, columns, filter_list)

        # Apply dict filters if we couldn't use them directly
        if isinstance(filters, dict) and not df.empty:
            for col, value in filters.items():
                if col in df.columns:
                    df = df[df[col] == value]

        return df

def create_optimized_storage(
    base_path: Union[str, Path] = 'data',
    enable_cache: bool = True,
) -> OptimizedStorage:
    """
    Create optimized storage with enhanced capabilities.

    Args:
        base_path: Base storage path
        enable_cache: Whether to enable query caching

    Returns:
        OptimizedStorage instance
    """
    return OptimizedStorage(
        base_path=base_path,
        compression='zstd',
        compression_level=3,
        row_group_size=100000,
        enable_cache=enable_cache,
    )

__all__ = [
    # Enums
    'DataTier',
    'PartitionStrategy',
    'CompressionLevel',
    'RetentionPolicy',
    # Dataclasses
    'StorageMetrics',
    'FileMetadata',
    'QueryResult',
    # Classes
    'LRUCache',
    'ParquetStorage',
    'OptimizedStorage',
    # Functions
    'create_storage',
    'create_tiered_storage',
    'create_optimized_storage',
]
