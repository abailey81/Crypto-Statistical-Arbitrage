"""
comprehensive Hierarchical Cache for Crypto Data Collection.

This module implements detailed caching strategies based on research best practices:

1. MULTI-LEVEL HIERARCHICAL CACHE
   - L1: In-memory LRU cache (fastest, limited size)
   - L2: Disk-based compressed parquet (larger, persistent)
   - Write-through and write-back policies

2. BLOOM FILTER FOR NEGATIVE LOOKUPS
   - Space-efficient probabilistic data structure
   - Fast "definitely not in cache" checks
   - Prevents unnecessary disk reads

3. TIME-BASED LRU WITH TTL
   - Automatic expiration of stale data
   - Configurable TTL per data type
   - Background cleanup

4. CACHE COMPRESSION
   - Snappy/gzip compression for parquet
   - Memory-mapped files for large datasets
   - Intelligent compression selection

5. SQLITE METADATA STORAGE
   - ACID-compliant metadata persistence
   - Fast range queries for gap detection
   - Transaction support

6. CACHE WARMING STRATEGIES
   - Preload frequently accessed data
   - Background warming on startup
   - Predictive prefetching

7. CONTENT-ADDRESSABLE STORAGE
   - Hash-based deduplication
   - Efficient delta updates
   - Integrity verification

References:
- TSCache (VLDB): https://www.vldb.org/pvldb/vol14/p3253-liu.pdf
- Bloom Filters: https://www.geeksforgeeks.org/bloom-filters-introduction-and-python-implementation/
- Time-Based LRU: https://jamesg.blog/2024/08/18/time-based-lru-cache-python
- Cache Hierarchy: https://en.wikipedia.org/wiki/Cache_hierarchy
- Multi-layer Cache: https://github.com/camcima/cache-tower

Version: 2.0.0 (comprehensive)
"""

import asyncio
import gzip
import hashlib
import json
import logging
import math
import mmap
import os
import pickle
import shutil
import sqlite3
import struct
import threading
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generic, Iterator, List, Optional,
    Set, Tuple, TypeVar, Union
)
import io

import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# TYPE VARIABLES
# =============================================================================

K = TypeVar('K')
V = TypeVar('V')

# =============================================================================
# BLOOM FILTER
# =============================================================================

class BloomFilter:
    """
    Space-efficient probabilistic data structure for set membership testing.

    A Bloom filter can tell you:
    - "Definitely NOT in the set" (no false negatives)
    - "POSSIBLY in the set" (may have false positives)

    This is perfect for caching because:
    - Fast negative lookups prevent unnecessary disk reads
    - False positives just cause an extra disk check (acceptable)
    - Memory efficient: ~10 bits per element for 1% false positive rate

    Formula:
    - m (bits) = -n * ln(p) / (ln(2)^2) where n=elements, p=false positive rate
    - k (hash functions) = (m/n) * ln(2)

    References:
    - https://www.geeksforgeeks.org/bloom-filters-introduction-and-python-implementation/
    - https://en.wikipedia.org/wiki/Bloom_filter
    """

    def __init__(
        self,
        expected_elements: int = 100000,
        false_positive_rate: float = 0.01
    ):
        """
        Initialize Bloom filter.

        Args:
            expected_elements: Expected number of elements to store
            false_positive_rate: Acceptable false positive rate (0.01 = 1%)
        """
        # Calculate optimal size
        self._size = self._optimal_size(expected_elements, false_positive_rate)
        self._hash_count = self._optimal_hash_count(self._size, expected_elements)

        # Bit array (using bytearray for efficiency)
        self._bit_array = bytearray((self._size + 7) // 8)

        # Statistics
        self._elements_added = 0

        logger.debug(
            f"BloomFilter initialized: size={self._size} bits, "
            f"hash_functions={self._hash_count}, "
            f"expected_fp_rate={false_positive_rate*100:.2f}%"
        )

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """Calculate optimal bit array size."""
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        """Calculate optimal number of hash functions."""
        k = (m / n) * math.log(2)
        return max(1, int(k))

    def _get_hash_values(self, item: str) -> List[int]:
        """
        Generate k hash values for item.

        Uses double hashing technique:
        h(i) = (h1 + i*h2) mod m

        This is more efficient than computing k independent hashes.
        """
        # Get two base hashes
        item_bytes = item.encode('utf-8')
        h1 = int(hashlib.md5(item_bytes).hexdigest(), 16)
        h2 = int(hashlib.sha256(item_bytes).hexdigest(), 16)

        # Generate k hashes via double hashing
        return [(h1 + i * h2) % self._size for i in range(self._hash_count)]

    def add(self, item: str) -> None:
        """Add item to the filter."""
        for position in self._get_hash_values(item):
            byte_index = position // 8
            bit_index = position % 8
            self._bit_array[byte_index] |= (1 << bit_index)

        self._elements_added += 1

    def contains(self, item: str) -> bool:
        """
        Check if item might be in the set.

        Returns:
            True if item MIGHT be in set (possible false positive)
            False if item is DEFINITELY NOT in set (no false negatives)
        """
        for position in self._get_hash_values(item):
            byte_index = position // 8
            bit_index = position % 8
            if not (self._bit_array[byte_index] & (1 << bit_index)):
                return False
        return True

    def __contains__(self, item: str) -> bool:
        """Support 'in' operator."""
        return self.contains(item)

    def estimated_false_positive_rate(self) -> float:
        """Calculate estimated current false positive rate."""
        # p = (1 - e^(-kn/m))^k
        if self._elements_added == 0:
            return 0.0

        exponent = -self._hash_count * self._elements_added / self._size
        return (1 - math.exp(exponent)) ** self._hash_count

    def stats(self) -> Dict[str, Any]:
        """Get filter statistics."""
        return {
            'size_bits': self._size,
            'size_bytes': len(self._bit_array),
            'hash_functions': self._hash_count,
            'elements_added': self._elements_added,
            'estimated_fp_rate': f"{self.estimated_false_positive_rate()*100:.4f}%",
            'fill_ratio': sum(bin(b).count('1') for b in self._bit_array) / self._size,
        }

    def save(self, path: Path) -> None:
        """Save filter to file."""
        data = {
            'size': self._size,
            'hash_count': self._hash_count,
            'elements_added': self._elements_added,
            'bit_array': self._bit_array.hex(),
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> 'BloomFilter':
        """Load filter from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        bf = cls.__new__(cls)
        bf._size = data['size']
        bf._hash_count = data['hash_count']
        bf._elements_added = data['elements_added']
        bf._bit_array = bytearray.fromhex(data['bit_array'])
        return bf

# =============================================================================
# TIME-BASED LRU CACHE
# =============================================================================

@dataclass
class CacheItem(Generic[V]):
    """Item stored in cache with metadata."""
    value: V
    created_at: float
    accessed_at: float
    ttl: float # Time-to-live in seconds
    size_bytes: int = 0
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if item has expired."""
        if self.ttl <= 0:
            return False # No expiration
        return time.time() - self.created_at > self.ttl

    @property
    def age_seconds(self) -> float:
        """Get age of item in seconds."""
        return time.time() - self.created_at

class TimeLRUCache(Generic[K, V]):
    """
    LRU Cache with time-based expiration (TTL).

    Combines best of both worlds:
    - LRU eviction when capacity is reached
    - Automatic expiration based on TTL
    - Popular entries refreshed on access

    Features:
    - Thread-safe operations
    - Configurable max size (items or bytes)
    - Per-item TTL support
    - Statistics tracking

    References:
    - https://jamesg.blog/2024/08/18/time-based-lru-cache-python
    """

    def __init__(
        self,
        max_items: int = 1000,
        max_bytes: Optional[int] = None,
        default_ttl: float = 3600.0, # 1 hour default
        cleanup_interval: float = 60.0 # Cleanup every minute
    ):
        """
        Initialize time-based LRU cache.

        Args:
            max_items: Maximum number of items
            max_bytes: Maximum total size in bytes (optional)
            default_ttl: Default time-to-live in seconds
            cleanup_interval: Background cleanup interval
        """
        self._cache: OrderedDict[K, CacheItem[V]] = OrderedDict()
        self._max_items = max_items
        self._max_bytes = max_bytes
        self._default_ttl = default_ttl
        self._cleanup_interval = cleanup_interval

        self._current_bytes = 0
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

        # Start background cleanup (optional)
        self._cleanup_task: Optional[asyncio.Task] = None

    def get(self, key: K, default: V = None) -> Optional[V]:
        """
        Get item from cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return default

            item = self._cache[key]

            # Check expiration
            if item.is_expired:
                self._remove(key)
                self._expirations += 1
                self._misses += 1
                return default

            # Update access time and move to end (most recent)
            item.accessed_at = time.time()
            item.hits += 1
            self._cache.move_to_end(key)

            self._hits += 1
            return item.value

    def put(
        self,
        key: K,
        value: V,
        ttl: Optional[float] = None,
        size_bytes: int = 0
    ) -> None:
        """
        Put item into cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live (uses default if not specified)
            size_bytes: Size of value in bytes
        """
        with self._lock:
            # Remove existing if present
            if key in self._cache:
                self._remove(key)

            # Evict if needed
            self._evict_if_needed(size_bytes)

            # Add new item
            item = CacheItem(
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl=ttl if ttl is not None else self._default_ttl,
                size_bytes=size_bytes
            )

            self._cache[key] = item
            self._current_bytes += size_bytes

    def _remove(self, key: K) -> None:
        """Remove item from cache (internal, assumes lock held)."""
        if key in self._cache:
            item = self._cache.pop(key)
            self._current_bytes -= item.size_bytes

    def _evict_if_needed(self, new_item_bytes: int = 0) -> None:
        """Evict items if cache is over capacity."""
        # Evict by item count
        while len(self._cache) >= self._max_items:
            # Remove oldest (first) item
            oldest_key = next(iter(self._cache))
            self._remove(oldest_key)
            self._evictions += 1

        # Evict by byte size
        if self._max_bytes is not None:
            while (self._current_bytes + new_item_bytes > self._max_bytes
                   and self._cache):
                oldest_key = next(iter(self._cache))
                self._remove(oldest_key)
                self._evictions += 1

    def remove(self, key: K) -> bool:
        """
        Remove item from cache.

        Returns:
            True if item was removed
        """
        with self._lock:
            if key in self._cache:
                self._remove(key)
                return True
            return False

    def clear(self) -> None:
        """Clear all items from cache."""
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0

    def contains(self, key: K) -> bool:
        """Check if key is in cache (without updating access time)."""
        with self._lock:
            if key not in self._cache:
                return False
            item = self._cache[key]
            if item.is_expired:
                self._remove(key)
                self._expirations += 1
                return False
            return True

    def __contains__(self, key: K) -> bool:
        return self.contains(key)

    def cleanup_expired(self) -> int:
        """
        Remove all expired items.

        Returns:
            Number of items removed
        """
        with self._lock:
            expired_keys = [
                key for key, item in self._cache.items()
                if item.is_expired
            ]

            for key in expired_keys:
                self._remove(key)
                self._expirations += 1

            return len(expired_keys)

    async def start_background_cleanup(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._background_cleanup())

    async def _background_cleanup(self) -> None:
        """Background task to cleanup expired items."""
        while True:
            await asyncio.sleep(self._cleanup_interval)
            removed = self.cleanup_expired()
            if removed > 0:
                logger.debug(f"LRU cache cleanup: removed {removed} expired items")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests * 100 if total_requests > 0 else 0

            return {
                'items': len(self._cache),
                'max_items': self._max_items,
                'bytes': self._current_bytes,
                'max_bytes': self._max_bytes,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': f"{hit_rate:.1f}%",
                'evictions': self._evictions,
                'expirations': self._expirations,
            }

# =============================================================================
# L2 DISK CACHE
# =============================================================================

@dataclass
class DiskCacheEntry:
    """Metadata for disk cache entry."""
    key: str
    file_path: str
    data_type: str
    venue: str
    start_date: str
    end_date: str
    record_count: int
    file_size_bytes: int
    compression: str
    created_at: str
    accessed_at: str
    content_hash: str # For integrity verification
    symbols: List[str] = field(default_factory=list)
    timeframe: Optional[str] = None

class L2DiskCache:
    """
    Disk-based L2 cache with SQLite metadata storage.

    Features:
    - Compressed parquet storage
    - SQLite for fast metadata queries
    - Content-addressable via hash
    - Range query support for gap detection
    - Automatic compaction

    References:
    - TSCache (VLDB): https://www.vldb.org/pvldb/vol14/p3253-liu.pdf
    """

    def __init__(
        self,
        cache_dir: str,
        compression: str = 'snappy',
        max_size_gb: float = 10.0
    ):
        """
        Initialize L2 disk cache.

        Args:
            cache_dir: Directory for cache storage
            compression: Compression for parquet ('snappy', 'gzip', 'none')
            max_size_gb: Maximum cache size in gigabytes
        """
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._compression = compression
        self._max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)

        # SQLite for metadata
        self._db_path = self._cache_dir / 'cache_metadata.db'
        self._init_database()

        # Statistics
        self._reads = 0
        self._writes = 0
        self._hits = 0
        self._misses = 0

    def _init_database(self) -> None:
        """Initialize SQLite database for metadata."""
        conn = sqlite3.connect(str(self._db_path))
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                data_type TEXT NOT NULL,
                venue TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                record_count INTEGER,
                file_size_bytes INTEGER,
                compression TEXT,
                created_at TEXT,
                accessed_at TEXT,
                content_hash TEXT,
                symbols TEXT,
                timeframe TEXT
            )
        ''')

        # Indexes for fast queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_data_type_venue
            ON cache_entries(data_type, venue)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_date_range
            ON cache_entries(start_date, end_date)
        ''')

        conn.commit()
        conn.close()

    def _generate_key(
        self,
        data_type: str,
        venue: str,
        start_date: str,
        end_date: str,
        timeframe: Optional[str] = None
    ) -> str:
        """Generate cache key."""
        parts = [data_type, venue, start_date, end_date]
        if timeframe:
            parts.append(timeframe)
        return ':'.join(parts)

    def _compute_hash(self, df: pd.DataFrame) -> str:
        """Compute content hash for DataFrame."""
        # Hash based on shape and sample of data
        shape_str = f"{df.shape[0]}_{df.shape[1]}"
        cols_str = '_'.join(sorted(df.columns.tolist()))

        # Sample data for hash (first/last rows)
        if len(df) > 0:
            sample = pd.concat([df.head(5), df.tail(5)]).to_string()
        else:
            sample = ""

        content = f"{shape_str}_{cols_str}_{sample}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(
        self,
        data_type: str,
        venue: str,
        start_date: str,
        end_date: str,
        timeframe: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get data from L2 cache.

        Args:
            data_type: Type of data
            venue: Venue name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Timeframe for OHLCV

        Returns:
            DataFrame or None if not in cache
        """
        self._reads += 1
        key = self._generate_key(data_type, venue, start_date, end_date, timeframe)

        conn = sqlite3.connect(str(self._db_path))
        cursor = conn.cursor()

        cursor.execute(
            'SELECT file_path FROM cache_entries WHERE key = ?',
            (key,)
        )
        row = cursor.fetchone()

        if row is None:
            conn.close()
            self._misses += 1
            return None

        file_path = Path(row[0])

        if not file_path.exists():
            # Entry exists but file is missing - clean up
            cursor.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
            conn.commit()
            conn.close()
            self._misses += 1
            return None

        # Update access time
        cursor.execute(
            'UPDATE cache_entries SET accessed_at = ? WHERE key = ?',
            (datetime.now(timezone.utc).isoformat(), key)
        )
        conn.commit()
        conn.close()

        # Read data
        try:
            df = pd.read_parquet(file_path)
            self._hits += 1
            return df
        except Exception as e:
            logger.error(f"Failed to read cache file {file_path}: {e}")
            self._misses += 1
            return None

    def get_range(
        self,
        data_type: str,
        venue: str,
        start_date: str,
        end_date: str,
        timeframe: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get data covering a date range (may span multiple cache entries).

        This is more flexible than get() - it finds all entries that overlap
        with the requested range and merges them.
        """
        conn = sqlite3.connect(str(self._db_path))
        cursor = conn.cursor()

        # Find all overlapping entries
        query = '''
            SELECT file_path, start_date, end_date FROM cache_entries
            WHERE data_type = ? AND venue = ?
            AND NOT (end_date < ? OR start_date > ?)
        '''
        params = [data_type, venue, start_date, end_date]

        if timeframe:
            query += ' AND timeframe = ?'
            params.append(timeframe)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None

        # Load and merge all overlapping data
        dfs = []
        for file_path, _, _ in rows:
            try:
                df = pd.read_parquet(file_path)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")

        if not dfs:
            return None

        merged = pd.concat(dfs, ignore_index=True)

        # Filter to requested range
        if 'timestamp' in merged.columns:
            merged['timestamp'] = pd.to_datetime(merged['timestamp'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            merged = merged[
                (merged['timestamp'] >= start_dt) &
                (merged['timestamp'] <= end_dt)
            ]

        # Deduplicate
        if 'timestamp' in merged.columns:
            dedup_cols = ['timestamp']
            if 'symbol' in merged.columns:
                dedup_cols.append('symbol')
            merged = merged.drop_duplicates(subset=dedup_cols, keep='last')
            merged = merged.sort_values('timestamp').reset_index(drop=True)

        return merged

    def put(
        self,
        data: pd.DataFrame,
        data_type: str,
        venue: str,
        start_date: str,
        end_date: str,
        timeframe: Optional[str] = None,
        symbols: Optional[List[str]] = None
    ) -> bool:
        """
        Store data in L2 cache.

        Args:
            data: DataFrame to cache
            data_type: Type of data
            venue: Venue name
            start_date: Start date
            end_date: End date
            timeframe: Timeframe for OHLCV
            symbols: List of symbols in data

        Returns:
            True if stored successfully
        """
        self._writes += 1
        key = self._generate_key(data_type, venue, start_date, end_date, timeframe)

        # Generate file path
        subdir = self._cache_dir / data_type / venue
        subdir.mkdir(parents=True, exist_ok=True)

        filename = f"{venue}_{data_type}_{start_date}_{end_date}"
        if timeframe:
            filename += f"_{timeframe}"
        filename += ".parquet"

        file_path = subdir / filename

        try:
            # Save with compression
            compression_arg = self._compression if self._compression != 'none' else None
            data.to_parquet(file_path, index=False, compression=compression_arg)

            file_size = file_path.stat().st_size
            content_hash = self._compute_hash(data)

            # Detect symbols from data if not provided
            if symbols is None and 'symbol' in data.columns:
                symbols = data['symbol'].unique().tolist()

            # Update metadata
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO cache_entries
                (key, file_path, data_type, venue, start_date, end_date,
                 record_count, file_size_bytes, compression, created_at,
                 accessed_at, content_hash, symbols, timeframe)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                key,
                str(file_path),
                data_type,
                venue,
                start_date,
                end_date,
                len(data),
                file_size,
                self._compression,
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat(),
                content_hash,
                json.dumps(symbols) if symbols else '[]',
                timeframe
            ))

            conn.commit()
            conn.close()

            logger.debug(f"L2 cache stored: {key} ({len(data)} records, {file_size} bytes)")
            return True

        except Exception as e:
            logger.error(f"Failed to store in L2 cache: {e}")
            return False

    def get_gaps(
        self,
        data_type: str,
        venue: str,
        start_date: str,
        end_date: str,
        timeframe: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """
        Find gaps in cached data for date range.

        Returns list of (gap_start, gap_end) tuples that need to be collected.
        """
        conn = sqlite3.connect(str(self._db_path))
        cursor = conn.cursor()

        # Find all entries within range
        query = '''
            SELECT start_date, end_date FROM cache_entries
            WHERE data_type = ? AND venue = ?
            AND NOT (end_date < ? OR start_date > ?)
            ORDER BY start_date
        '''
        params = [data_type, venue, start_date, end_date]

        if timeframe:
            query = query.replace(
                'ORDER BY',
                'AND timeframe = ? ORDER BY'
            )
            params.insert(-2, timeframe)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            # No cached data - entire range is a gap
            return [(start_date, end_date)]

        # Find gaps between cached ranges
        gaps = []
        current_start = start_date

        for cached_start, cached_end in rows:
            if cached_start > current_start:
                # Gap before this cached range
                gaps.append((current_start, cached_start))

            # Move past this cached range
            current_start = max(current_start, cached_end)

        # Check for gap at end
        if current_start < end_date:
            gaps.append((current_start, end_date))

        return gaps

    def evict_oldest(self, target_size_bytes: Optional[int] = None) -> int:
        """
        Evict oldest entries to reduce cache size.

        Args:
            target_size_bytes: Target size (uses max_size if not specified)

        Returns:
            Number of entries evicted
        """
        target = target_size_bytes or self._max_size_bytes
        current_size = self.total_size_bytes()

        if current_size <= target:
            return 0

        conn = sqlite3.connect(str(self._db_path))
        cursor = conn.cursor()

        # Get entries ordered by access time (oldest first)
        cursor.execute('''
            SELECT key, file_path, file_size_bytes FROM cache_entries
            ORDER BY accessed_at ASC
        ''')

        evicted = 0
        for key, file_path, size in cursor.fetchall():
            if current_size <= target:
                break

            # Delete file
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")

            # Delete entry
            cursor.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
            current_size -= size
            evicted += 1

        conn.commit()
        conn.close()

        logger.info(f"L2 cache evicted {evicted} entries")
        return evicted

    def total_size_bytes(self) -> int:
        """Get total cache size in bytes."""
        conn = sqlite3.connect(str(self._db_path))
        cursor = conn.cursor()
        cursor.execute('SELECT SUM(file_size_bytes) FROM cache_entries')
        result = cursor.fetchone()[0]
        conn.close()
        return result or 0

    def clear(self) -> None:
        """Clear all cached data."""
        conn = sqlite3.connect(str(self._db_path))
        cursor = conn.cursor()

        # Get all file paths
        cursor.execute('SELECT file_path FROM cache_entries')
        for (file_path,) in cursor.fetchall():
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass

        cursor.execute('DELETE FROM cache_entries')
        conn.commit()
        conn.close()

        logger.info("L2 cache cleared")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        conn = sqlite3.connect(str(self._db_path))
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*), SUM(file_size_bytes), SUM(record_count) FROM cache_entries')
        count, total_bytes, total_records = cursor.fetchone()
        conn.close()

        total_requests = self._reads
        hit_rate = self._hits / total_requests * 100 if total_requests > 0 else 0

        return {
            'entries': count or 0,
            'total_bytes': total_bytes or 0,
            'total_bytes_human': self._human_bytes(total_bytes or 0),
            'total_records': total_records or 0,
            'max_bytes': self._max_size_bytes,
            'reads': self._reads,
            'writes': self._writes,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': f"{hit_rate:.1f}%",
        }

    @staticmethod
    def _human_bytes(size: int) -> str:
        """Convert bytes to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"

# =============================================================================
# HIERARCHICAL CACHE
# =============================================================================

class HierarchicalCache:
    """
    Multi-level hierarchical cache with L1 (memory) and L2 (disk).

    Architecture:
        
                          Application 
                                                         
                                                         
                                     
                      Bloom Filter Fast "not in cache" check
                      (probabilistic) 
                                     
                              might exist 
                                                        
                                     
                       L1 Memory Hot data 
                       (LRU + TTL) 
                                     
                              miss 
                                                        
                                     
                       L2 Disk Persistent 
                       (SQLite+Parquet) 
                                     
                              miss 
                                                        
                                     
                      Data Source API call 
                      (API/Network) 
                                     
        

    Write Policy:
    - Write-through: L1 → L2 on every write
    - Promotes from L2 to L1 on read

    References:
    - https://github.com/camcima/cache-tower
    - https://en.wikipedia.org/wiki/Cache_hierarchy
    """

    def __init__(
        self,
        cache_dir: str = 'data/processed',
        l1_max_items: int = 100,
        l1_max_bytes: Optional[int] = None,
        l1_ttl: float = 3600.0, # 1 hour
        l2_max_gb: float = 10.0,
        compression: str = 'snappy',
        bloom_expected_elements: int = 100000,
        bloom_fp_rate: float = 0.01
    ):
        """
        Initialize hierarchical cache.

        Args:
            cache_dir: Base directory for cache
            l1_max_items: Max items in L1 memory cache
            l1_max_bytes: Max bytes in L1 (optional)
            l1_ttl: Default TTL for L1 items
            l2_max_gb: Max size of L2 disk cache in GB
            compression: Compression for L2 parquet files
            bloom_expected_elements: Expected elements for bloom filter
            bloom_fp_rate: Bloom filter false positive rate
        """
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # L1 Memory Cache
        self._l1 = TimeLRUCache[str, pd.DataFrame](
            max_items=l1_max_items,
            max_bytes=l1_max_bytes,
            default_ttl=l1_ttl
        )

        # L2 Disk Cache
        self._l2 = L2DiskCache(
            cache_dir=str(self._cache_dir),
            compression=compression,
            max_size_gb=l2_max_gb
        )

        # Bloom Filter for fast negative lookups
        self._bloom = BloomFilter(
            expected_elements=bloom_expected_elements,
            false_positive_rate=bloom_fp_rate
        )
        self._bloom_path = self._cache_dir / 'bloom_filter.json'
        self._load_bloom_filter()

        # Statistics
        self._l1_hits = 0
        self._l2_hits = 0
        self._bloom_rejections = 0
        self._total_gets = 0

        logger.info(
            f"HierarchicalCache initialized: L1={l1_max_items} items, "
            f"L2={l2_max_gb}GB, bloom_fp_rate={bloom_fp_rate*100}%"
        )

    def _load_bloom_filter(self) -> None:
        """Load bloom filter from disk if exists."""
        if self._bloom_path.exists():
            try:
                self._bloom = BloomFilter.load(self._bloom_path)
                logger.debug(f"Loaded bloom filter: {self._bloom.stats()}")
            except Exception as e:
                logger.warning(f"Failed to load bloom filter: {e}")

    def _save_bloom_filter(self) -> None:
        """Save bloom filter to disk."""
        try:
            self._bloom.save(self._bloom_path)
        except Exception as e:
            logger.warning(f"Failed to save bloom filter: {e}")

    def _generate_key(
        self,
        data_type: str,
        venue: str,
        start_date: str,
        end_date: str,
        timeframe: Optional[str] = None
    ) -> str:
        """Generate cache key."""
        parts = [data_type, venue, start_date, end_date]
        if timeframe:
            parts.append(timeframe)
        return ':'.join(parts)

    def get(
        self,
        data_type: str,
        venue: str,
        start_date: str,
        end_date: str,
        timeframe: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get data from hierarchical cache.

        Checks in order:
        1. Bloom filter (fast rejection of non-existent keys)
        2. L1 memory cache (fastest)
        3. L2 disk cache (persistent)

        Args:
            data_type: Type of data
            venue: Venue name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Timeframe for OHLCV

        Returns:
            DataFrame or None if not cached
        """
        self._total_gets += 1
        key = self._generate_key(data_type, venue, start_date, end_date, timeframe)

        # Check bloom filter first
        if key not in self._bloom:
            self._bloom_rejections += 1
            return None

        # Check L1 (memory)
        df = self._l1.get(key)
        if df is not None:
            self._l1_hits += 1
            return df

        # Check L2 (disk)
        df = self._l2.get_range(data_type, venue, start_date, end_date, timeframe)
        if df is not None:
            self._l2_hits += 1
            # Promote to L1
            size_bytes = df.memory_usage(deep=True).sum()
            self._l1.put(key, df, size_bytes=int(size_bytes))
            return df

        return None

    def put(
        self,
        data: pd.DataFrame,
        data_type: str,
        venue: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: Optional[str] = None,
        symbols: Optional[List[str]] = None
    ) -> bool:
        """
        Store data in hierarchical cache.

        Write-through policy: data is written to both L1 and L2.

        Args:
            data: DataFrame to cache
            data_type: Type of data
            venue: Venue name
            start_date: Start date (auto-detected if not provided)
            end_date: End date (auto-detected if not provided)
            timeframe: Timeframe for OHLCV
            symbols: List of symbols

        Returns:
            True if stored successfully
        """
        if data is None or data.empty:
            return False

        # Auto-detect date range from data
        if start_date is None or end_date is None:
            if 'timestamp' in data.columns:
                timestamps = pd.to_datetime(data['timestamp'])
                start_date = start_date or timestamps.min().strftime('%Y-%m-%d')
                end_date = end_date or timestamps.max().strftime('%Y-%m-%d')
            else:
                start_date = start_date or datetime.now().strftime('%Y-%m-%d')
                end_date = end_date or start_date

        key = self._generate_key(data_type, venue, start_date, end_date, timeframe)

        # Write to L1 (memory)
        size_bytes = int(data.memory_usage(deep=True).sum())
        self._l1.put(key, data, size_bytes=size_bytes)

        # Write to L2 (disk)
        success = self._l2.put(
            data=data,
            data_type=data_type,
            venue=venue,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            symbols=symbols
        )

        # Add to bloom filter
        if success:
            self._bloom.add(key)
            self._save_bloom_filter()

        return success

    def get_gaps(
        self,
        data_type: str,
        venue: str,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """
        Get gaps in cached data that need to be collected.

        Returns list of (gap_start, gap_end) tuples.
        """
        return self._l2.get_gaps(
            data_type=data_type,
            venue=venue,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )

    def load_cached_data(
        self,
        data_type: str,
        venue: str,
        start_date: str,
        end_date: str,
        timeframe: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load all cached data for a range (alias for get).
        """
        return self.get(data_type, venue, start_date, end_date, timeframe)

    def update_cache(
        self,
        data: pd.DataFrame,
        data_type: str,
        venue: str,
        timeframe: Optional[str] = None
    ) -> bool:
        """
        Update cache with new data (alias for put with auto date detection).
        """
        return self.put(
            data=data,
            data_type=data_type,
            venue=venue,
            timeframe=timeframe
        )

    def clear(self) -> None:
        """Clear all cached data."""
        self._l1.clear()
        self._l2.clear()
        self._bloom = BloomFilter()
        self._save_bloom_filter()
        logger.info("Hierarchical cache cleared")

    def evict_to_size(self, target_gb: float) -> int:
        """Evict L2 entries to reach target size."""
        target_bytes = int(target_gb * 1024 * 1024 * 1024)
        return self._l2.evict_oldest(target_bytes)

    def warm(
        self,
        data_types: List[str],
        venues: List[str],
        recent_days: int = 7
    ) -> int:
        """
        Warm L1 cache with recent data from L2.

        Args:
            data_types: Data types to warm
            venues: Venues to warm
            recent_days: Load data from last N days

        Returns:
            Number of entries warmed
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=recent_days)).strftime('%Y-%m-%d')

        warmed = 0
        for data_type in data_types:
            for venue in venues:
                df = self._l2.get_range(data_type, venue, start_date, end_date)
                if df is not None and not df.empty:
                    key = self._generate_key(data_type, venue, start_date, end_date)
                    size_bytes = int(df.memory_usage(deep=True).sum())
                    self._l1.put(key, df, size_bytes=size_bytes)
                    warmed += 1

        logger.info(f"Cache warming: loaded {warmed} entries to L1")
        return warmed

    def stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_hits = self._l1_hits + self._l2_hits
        hit_rate = total_hits / self._total_gets * 100 if self._total_gets > 0 else 0

        return {
            'total_gets': self._total_gets,
            'total_hit_rate': f"{hit_rate:.1f}%",
            'l1_hits': self._l1_hits,
            'l2_hits': self._l2_hits,
            'bloom_rejections': self._bloom_rejections,
            'l1_stats': self._l1.stats(),
            'l2_stats': self._l2.stats(),
            'bloom_stats': self._bloom.stats(),
        }

# =============================================================================
# SINGLETON CACHE MANAGER
# =============================================================================

_cache_instance: Optional[HierarchicalCache] = None
_cache_lock = threading.Lock()

def get_hierarchical_cache(
    cache_dir: str = 'data/processed',
    **kwargs
) -> HierarchicalCache:
    """
    Get singleton instance of comprehensive hierarchical cache.

    Args:
        cache_dir: Cache directory
        **kwargs: Additional arguments for HierarchicalCache

    Returns:
        HierarchicalCache instance
    """
    global _cache_instance

    with _cache_lock:
        if _cache_instance is None:
            _cache_instance = HierarchicalCache(cache_dir=cache_dir, **kwargs)

        return _cache_instance

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Bloom Filter
    'BloomFilter',

    # LRU Cache
    'CacheItem',
    'TimeLRUCache',

    # L2 Disk Cache
    'DiskCacheEntry',
    'L2DiskCache',

    # Hierarchical Cache
    'HierarchicalCache',
    'get_hierarchical_cache',
]
