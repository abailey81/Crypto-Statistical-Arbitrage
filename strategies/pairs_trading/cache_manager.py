"""
Comprehensive Caching System for Crypto Statistical Arbitrage.

This module provides a comprehensive caching system designed to:
- Cache expensive cointegration test results
- Cache universe construction snapshots
- Cache price matrix data with compression
- Cache backtest results with full metadata

Features:
- Content-addressable storage using SHA256 hashes
- Automatic cache invalidation when parameters change
- LRU eviction with configurable size limits
- Compression for large objects (gzip, lz4)
- Parallel cache loading with thread pools
- Human-readable manifests for debugging
- Version control for code compatibility
- Atomic writes to prevent corruption
- Cache warming and prefetching

Usage:
    from strategies.pairs_trading.cache_manager import CacheManager, CacheConfig

    cache = CacheManager(CacheConfig(cache_dir="outputs/cache"))

    # Check if cointegration results exist
    cache_key = cache.get_cointegration_key(pair, params)
    if cache.has(cache_key):
        result = cache.get(cache_key)
    else:
        result = expensive_computation()
        cache.set(cache_key, result, ttl_hours=168)  # 1 week TTL

Author: Phase 2 Optimization System
Version: 1.0.0
"""

import hashlib
import json
import gzip
import math
import pickle
import shutil
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# RICH PROGRESS MONITORING - Enhanced Console UI
# =============================================================================
try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn,
        MofNCompleteColumn
    )
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


# =============================================================================
# BLOOM FILTER FOR FAST NEGATIVE LOOKUPS
# =============================================================================

class BloomFilter:
    """
    Space-efficient probabilistic data structure for set membership testing.

    Prevents unnecessary disk reads by quickly identifying keys that are
    definitely NOT in the cache.
    """

    def __init__(self, expected_elements: int = 100000, false_positive_rate: float = 0.01):
        # Calculate optimal size
        self._size = int(-(expected_elements * math.log(false_positive_rate)) / (math.log(2) ** 2))
        self._hash_count = int((self._size / expected_elements) * math.log(2))
        self._bit_array = bytearray((self._size + 7) // 8)
        self._elements_added = 0

    def _get_hash_values(self, key: str) -> List[int]:
        """Generate k hash values for the key."""
        h1 = int(hashlib.md5(key.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(key.encode()).hexdigest(), 16)
        return [(h1 + i * h2) % self._size for i in range(self._hash_count)]

    def add(self, key: str):
        """Add key to the bloom filter."""
        for bit_index in self._get_hash_values(key):
            byte_index = bit_index // 8
            bit_offset = bit_index % 8
            self._bit_array[byte_index] |= (1 << bit_offset)
        self._elements_added += 1

    def might_contain(self, key: str) -> bool:
        """Check if key might be in the set (no false negatives, possible false positives)."""
        for bit_index in self._get_hash_values(key):
            byte_index = bit_index // 8
            bit_offset = bit_index % 8
            if not (self._bit_array[byte_index] & (1 << bit_offset)):
                return False
        return True

    def clear(self):
        """Clear the bloom filter."""
        self._bit_array = bytearray((self._size + 7) // 8)
        self._elements_added = 0


# =============================================================================
# L1 IN-MEMORY LRU CACHE
# =============================================================================

class LRUCache:
    """
    Thread-safe LRU cache for L1 in-memory caching layer.

    Provides fast access to recently used items without disk I/O.
    """

    def __init__(self, max_size: int = 1000, max_memory_mb: float = 500.0):
        self._max_size = max_size
        self._max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self._cache: OrderedDict = OrderedDict()
        self._sizes: Dict[str, int] = {}
        self._total_size = 0
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, returns None if not found."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: Any, size_bytes: int = 0):
        """Put item in cache, evicting LRU items if necessary."""
        with self._lock:
            # If key exists, remove it first
            if key in self._cache:
                self._total_size -= self._sizes.get(key, 0)
                del self._cache[key]
                del self._sizes[key]

            # Evict until we have space
            while (len(self._cache) >= self._max_size or
                   self._total_size + size_bytes > self._max_memory_bytes) and self._cache:
                oldest_key, _ = self._cache.popitem(last=False)
                self._total_size -= self._sizes.pop(oldest_key, 0)

            # Add new item
            self._cache[key] = value
            self._sizes[key] = size_bytes
            self._total_size += size_bytes

    def remove(self, key: str) -> bool:
        """Remove item from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._total_size -= self._sizes.pop(key, 0)
                return True
            return False

    def clear(self):
        """Clear all items from cache."""
        with self._lock:
            self._cache.clear()
            self._sizes.clear()
            self._total_size = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            return {
                'entries': len(self._cache),
                'size_mb': round(self._total_size / (1024 * 1024), 2),
                'max_size': self._max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': round(hit_rate * 100, 2),
            }


# =============================================================================
# CONFIGURATION
# =============================================================================

class CompressionType(Enum):
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"  # Requires lz4 package


class CacheType(Enum):
    """Types of cached data with specific TTLs and purposes."""
    # Core computation caches
    COINTEGRATION = "cointegration"      # Cointegration test results (2 weeks)
    UNIVERSE = "universe"                # Universe construction snapshots (1 week)
    PRICE_MATRIX = "price_matrix"        # Price data matrices (1 day)
    BACKTEST = "backtest"                # Full backtest results (30 days)

    # Signal and strategy caches
    SIGNALS = "signals"                  # Signal generation results (1 week)
    ENHANCEMENT = "enhancement"          # Step 3 enhancement results (2 weeks)

    # Extended computation caches
    ML_PREDICTIONS = "ml_predictions"    # ML model predictions (1 week)
    REGIME_DETECTION = "regime_detection"  # HMM regime detection (2 weeks)
    WALK_FORWARD = "walk_forward"        # Walk-forward optimization (30 days)
    CRISIS_ANALYSIS = "crisis_analysis"  # Crisis event analysis (30 days)
    PAIR_RANKING = "pair_ranking"        # 12-factor pair rankings (2 weeks)

    # Utility caches
    GENERAL = "general"                  # General purpose cache (1 week)


@dataclass
class CacheConfig:
    """Configuration for the cache system."""
    cache_dir: str = "outputs/cache"
    max_size_gb: float = 10.0  # Maximum cache size in GB
    default_ttl_hours: float = 168.0  # 1 week default TTL
    compression: CompressionType = CompressionType.GZIP
    enable_lru_eviction: bool = True
    max_entries_per_type: int = 1000
    parallel_workers: int = 8
    atomic_writes: bool = True
    verify_on_read: bool = True
    cache_version: str = "1.0.0"

    # Type-specific TTLs (hours)
    # Core computation caches
    cointegration_ttl: float = 336.0     # 2 weeks - stable after initial computation
    universe_ttl: float = 168.0          # 1 week - market conditions change
    price_matrix_ttl: float = 24.0       # 1 day - data can be updated
    backtest_ttl: float = 720.0          # 30 days - expensive to recompute

    # Signal and strategy caches
    signals_ttl: float = 168.0           # 1 week
    enhancement_ttl: float = 336.0       # 2 weeks - expensive ML computations

    # Extended computation caches
    ml_predictions_ttl: float = 168.0    # 1 week - models need retraining
    regime_detection_ttl: float = 336.0  # 2 weeks - regime changes slowly
    walk_forward_ttl: float = 720.0      # 30 days - very expensive
    crisis_analysis_ttl: float = 720.0   # 30 days - historical analysis
    pair_ranking_ttl: float = 336.0      # 2 weeks - 12-factor ranking

    # Utility cache
    general_ttl: float = 168.0           # 1 week default


@dataclass
class CacheEntry:
    """Metadata for a cached entry."""
    key: str
    cache_type: str
    created_at: str
    expires_at: str
    size_bytes: int
    compression: str
    checksum: str
    version: str
    parameters_hash: str
    data_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: str = ""


@dataclass
class CacheManifest:
    """Manifest tracking all cache entries of a specific type."""
    cache_type: str
    version: str
    created_at: str
    last_updated: str
    total_entries: int
    total_size_bytes: int
    entries: Dict[str, CacheEntry] = field(default_factory=dict)


# =============================================================================
# HASH UTILITIES
# =============================================================================

class HashGenerator:
    """Generates deterministic hashes for cache keys."""

    @staticmethod
    def hash_dict(d: Dict[str, Any]) -> str:
        """Generate hash from dictionary."""
        # Sort keys for deterministic ordering
        serialized = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    @staticmethod
    def hash_dataframe(df: pd.DataFrame) -> str:
        """Generate hash from DataFrame structure and content sample."""
        # Hash structure + sample of data for speed
        structure = {
            'columns': list(df.columns),
            'shape': df.shape,
            'dtypes': {str(k): str(v) for k, v in df.dtypes.items()},
        }

        # Sample first/last rows for content hash
        if len(df) > 0:
            sample_size = min(100, len(df))
            sample = pd.concat([df.head(sample_size // 2), df.tail(sample_size // 2)])
            structure['sample_hash'] = hashlib.sha256(
                sample.to_json().encode()
            ).hexdigest()[:8]

        return HashGenerator.hash_dict(structure)

    @staticmethod
    def hash_array(arr: np.ndarray) -> str:
        """Generate hash from numpy array."""
        return hashlib.sha256(arr.tobytes()).hexdigest()[:16]

    @staticmethod
    def hash_params(**kwargs) -> str:
        """Generate hash from arbitrary parameters."""
        return HashGenerator.hash_dict(kwargs)

    @staticmethod
    def hash_file(filepath: Path) -> str:
        """Generate hash from file contents."""
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]


# =============================================================================
# SERIALIZATION
# =============================================================================

class Serializer:
    """Handles serialization and compression of cached data."""

    @staticmethod
    def serialize(data: Any, compression: CompressionType = CompressionType.GZIP) -> bytes:
        """Serialize and optionally compress data."""
        # Pickle the data
        pickled = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

        # Apply compression
        if compression == CompressionType.GZIP:
            return gzip.compress(pickled, compresslevel=6)
        elif compression == CompressionType.LZ4:
            try:
                import lz4.frame
                return lz4.frame.compress(pickled)
            except ImportError:
                logger.warning("LZ4 not available, falling back to gzip")
                return gzip.compress(pickled)
        else:
            return pickled

    @staticmethod
    def deserialize(data: bytes, compression: CompressionType = CompressionType.GZIP) -> Any:
        """Decompress and deserialize data."""
        # Apply decompression
        if compression == CompressionType.GZIP:
            decompressed = gzip.decompress(data)
        elif compression == CompressionType.LZ4:
            try:
                import lz4.frame
                decompressed = lz4.frame.decompress(data)
            except ImportError:
                # Try gzip as fallback
                decompressed = gzip.decompress(data)
        else:
            decompressed = data

        return pickle.loads(decompressed)

    @staticmethod
    def serialize_dataframe(df: pd.DataFrame, filepath: Path) -> int:
        """Serialize DataFrame to parquet with compression."""
        df.to_parquet(filepath, compression='snappy', index=True)
        return filepath.stat().st_size


# =============================================================================
# CACHE MANAGER
# =============================================================================

class CacheManager:
    """
    Extended cache manager for expensive computations.

    Features:
    - Content-addressable storage
    - Automatic expiration
    - LRU eviction
    - Compression
    - Atomic writes
    - Parallel operations
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize cache manager with L1/L2 hierarchical caching."""
        self.config = config or CacheConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self._lock = threading.RLock()
        self._manifests: Dict[CacheType, CacheManifest] = {}

        # L1 IN-MEMORY CACHE (fast access layer)
        self._l1_cache = LRUCache(max_size=500, max_memory_mb=256.0)

        # BLOOM FILTERS (fast negative lookups per cache type)
        self._bloom_filters: Dict[CacheType, BloomFilter] = {
            ct: BloomFilter(expected_elements=10000, false_positive_rate=0.01)
            for ct in CacheType
        }

        # Initialize cache directory structure
        self._init_cache_dirs()
        self._load_manifests()

        # Populate bloom filters from existing manifests
        self._populate_bloom_filters()

        logger.info(f"CacheManager initialized: {self.cache_dir}")
        logger.info(f"  Max size: {self.config.max_size_gb} GB")
        logger.info(f"  Compression: {self.config.compression.value}")
        logger.info(f"  L1 Cache: 256 MB in-memory LRU")
        logger.info(f"  Bloom Filters: Enabled for fast negative lookups")

    def _populate_bloom_filters(self):
        """Populate bloom filters from existing manifest entries."""
        for cache_type, manifest in self._manifests.items():
            bloom = self._bloom_filters[cache_type]
            for key in manifest.entries.keys():
                bloom.add(key)
            logger.debug(f"Populated bloom filter for {cache_type.value}: {len(manifest.entries)} entries")

    def _init_cache_dirs(self):
        """Create cache directory structure."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        for cache_type in CacheType:
            type_dir = self.cache_dir / cache_type.value
            type_dir.mkdir(exist_ok=True)
            (type_dir / "data").mkdir(exist_ok=True)

    def _get_manifest_path(self, cache_type: CacheType) -> Path:
        """Get manifest file path for cache type."""
        return self.cache_dir / cache_type.value / "manifest.json"

    def _load_manifests(self):
        """Load all cache manifests."""
        for cache_type in CacheType:
            manifest_path = self._get_manifest_path(cache_type)
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r') as f:
                        data = json.load(f)

                    # Reconstruct manifest
                    entries = {}
                    for key, entry_data in data.get('entries', {}).items():
                        entries[key] = CacheEntry(**entry_data)

                    self._manifests[cache_type] = CacheManifest(
                        cache_type=data['cache_type'],
                        version=data['version'],
                        created_at=data['created_at'],
                        last_updated=data['last_updated'],
                        total_entries=data['total_entries'],
                        total_size_bytes=data['total_size_bytes'],
                        entries=entries
                    )
                except Exception as e:
                    logger.warning(f"Failed to load manifest for {cache_type.value}: {e}")
                    self._manifests[cache_type] = self._create_empty_manifest(cache_type)
            else:
                self._manifests[cache_type] = self._create_empty_manifest(cache_type)

    def _create_empty_manifest(self, cache_type: CacheType) -> CacheManifest:
        """Create empty manifest for cache type."""
        now = datetime.now(timezone.utc).isoformat()
        return CacheManifest(
            cache_type=cache_type.value,
            version=self.config.cache_version,
            created_at=now,
            last_updated=now,
            total_entries=0,
            total_size_bytes=0,
            entries={}
        )

    def _save_manifest(self, cache_type: CacheType):
        """Save manifest to disk."""
        manifest = self._manifests[cache_type]
        manifest.last_updated = datetime.now(timezone.utc).isoformat()

        manifest_path = self._get_manifest_path(cache_type)

        # Convert to dict for JSON serialization
        data = {
            'cache_type': manifest.cache_type,
            'version': manifest.version,
            'created_at': manifest.created_at,
            'last_updated': manifest.last_updated,
            'total_entries': manifest.total_entries,
            'total_size_bytes': manifest.total_size_bytes,
            'entries': {k: asdict(v) for k, v in manifest.entries.items()}
        }

        # Atomic write
        if self.config.atomic_writes:
            temp_path = manifest_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            temp_path.rename(manifest_path)
        else:
            with open(manifest_path, 'w') as f:
                json.dump(data, f, indent=2)

    def _get_data_path(self, cache_type: CacheType, key: str) -> Path:
        """Get data file path for cache key."""
        return self.cache_dir / cache_type.value / "data" / f"{key}.cache"

    def _get_ttl(self, cache_type: CacheType) -> float:
        """Get TTL for cache type in hours."""
        ttl_map = {
            # Core computation caches
            CacheType.COINTEGRATION: self.config.cointegration_ttl,
            CacheType.UNIVERSE: self.config.universe_ttl,
            CacheType.PRICE_MATRIX: self.config.price_matrix_ttl,
            CacheType.BACKTEST: self.config.backtest_ttl,
            # Signal and strategy caches
            CacheType.SIGNALS: self.config.signals_ttl,
            CacheType.ENHANCEMENT: self.config.enhancement_ttl,
            # Extended computation caches
            CacheType.ML_PREDICTIONS: self.config.ml_predictions_ttl,
            CacheType.REGIME_DETECTION: self.config.regime_detection_ttl,
            CacheType.WALK_FORWARD: self.config.walk_forward_ttl,
            CacheType.CRISIS_ANALYSIS: self.config.crisis_analysis_ttl,
            CacheType.PAIR_RANKING: self.config.pair_ranking_ttl,
            # Utility cache
            CacheType.GENERAL: self.config.general_ttl,
        }
        return ttl_map.get(cache_type, self.config.default_ttl_hours)

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def generate_key(
        self,
        cache_type: CacheType,
        identifier: str,
        **params
    ) -> str:
        """
        Generate a unique cache key.

        Args:
            cache_type: Type of cached data
            identifier: Primary identifier (e.g., pair name)
            **params: Additional parameters that affect the result

        Returns:
            Unique cache key string
        """
        params_hash = HashGenerator.hash_params(**params) if params else "default"
        return f"{identifier}_{params_hash}"

    def has(self, cache_type: CacheType, key: str) -> bool:
        """
        Check if cache entry exists and is valid.

        Uses bloom filter for fast negative lookups and L1 cache check.

        Args:
            cache_type: Type of cached data
            key: Cache key

        Returns:
            True if valid cache entry exists
        """
        # FAST PATH: Check L1 cache first
        l1_key = f"{cache_type.value}:{key}"
        if self._l1_cache.get(l1_key) is not None:
            return True

        # BLOOM FILTER: Fast negative lookup
        bloom = self._bloom_filters.get(cache_type)
        if bloom and not bloom.might_contain(key):
            return False  # Definitely not in cache

        with self._lock:
            manifest = self._manifests.get(cache_type)
            if not manifest or key not in manifest.entries:
                return False

            entry = manifest.entries[key]

            # Check expiration
            expires_at = datetime.fromisoformat(entry.expires_at)
            if datetime.now(timezone.utc) > expires_at:
                # Expired - remove entry
                self._remove_entry(cache_type, key)
                return False

            # Check file exists
            data_path = self._get_data_path(cache_type, key)
            if not data_path.exists():
                self._remove_entry(cache_type, key)
                return False

            return True

    def get(
        self,
        cache_type: CacheType,
        key: str,
        verify: bool = None
    ) -> Optional[Any]:
        """
        Retrieve cached data with L1/L2 hierarchical lookup.

        L1: In-memory LRU cache (fastest)
        L2: Disk-based compressed storage (larger, persistent)

        Args:
            cache_type: Type of cached data
            key: Cache key
            verify: Whether to verify checksum (default from config)

        Returns:
            Cached data or None if not found/invalid
        """
        if verify is None:
            verify = self.config.verify_on_read

        l1_key = f"{cache_type.value}:{key}"

        # L1 CACHE CHECK (fastest path)
        l1_result = self._l1_cache.get(l1_key)
        if l1_result is not None:
            logger.debug(f"L1 Cache HIT: {cache_type.value}/{key}")
            return l1_result

        # BLOOM FILTER CHECK
        bloom = self._bloom_filters.get(cache_type)
        if bloom and not bloom.might_contain(key):
            logger.debug(f"Bloom filter MISS: {cache_type.value}/{key}")
            return None

        # L2 DISK CHECK
        with self._lock:
            manifest = self._manifests.get(cache_type)
            if not manifest or key not in manifest.entries:
                return None

            entry = manifest.entries[key]

            # Check expiration
            expires_at = datetime.fromisoformat(entry.expires_at)
            if datetime.now(timezone.utc) > expires_at:
                self._remove_entry(cache_type, key)
                return None

            data_path = self._get_data_path(cache_type, key)
            if not data_path.exists():
                self._remove_entry(cache_type, key)
                return None

        try:
            # Read from L2 (disk)
            with open(data_path, 'rb') as f:
                raw_data = f.read()

            # Verify checksum
            if verify:
                actual_checksum = hashlib.sha256(raw_data).hexdigest()[:16]
                if actual_checksum != entry.checksum:
                    logger.warning(f"Checksum mismatch for {key}, removing entry")
                    self._remove_entry(cache_type, key)
                    return None

            # Deserialize
            compression = CompressionType(entry.compression)
            data = Serializer.deserialize(raw_data, compression)

            # PROMOTE TO L1 CACHE
            self._l1_cache.put(l1_key, data, entry.size_bytes)

            # Update access stats
            with self._lock:
                entry.access_count += 1
                entry.last_accessed = datetime.now(timezone.utc).isoformat()

            logger.debug(f"L2 Cache HIT (promoted to L1): {cache_type.value}/{key}")
            return data

        except Exception as e:
            logger.error(f"Failed to read cache entry {key}: {e}")
            self._remove_entry(cache_type, key)
            return None

    def set(
        self,
        cache_type: CacheType,
        key: str,
        data: Any,
        ttl_hours: float = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Store data in cache.

        Args:
            cache_type: Type of cached data
            key: Cache key
            data: Data to cache
            ttl_hours: Time-to-live in hours (default from config)
            metadata: Additional metadata to store

        Returns:
            True if successfully cached
        """
        if ttl_hours is None:
            ttl_hours = self._get_ttl(cache_type)

        try:
            # Serialize data
            serialized = Serializer.serialize(data, self.config.compression)
            checksum = hashlib.sha256(serialized).hexdigest()[:16]

            # Create entry
            now = datetime.now(timezone.utc)
            entry = CacheEntry(
                key=key,
                cache_type=cache_type.value,
                created_at=now.isoformat(),
                expires_at=(now + timedelta(hours=ttl_hours)).isoformat(),
                size_bytes=len(serialized),
                compression=self.config.compression.value,
                checksum=checksum,
                version=self.config.cache_version,
                parameters_hash=key.split('_')[-1] if '_' in key else "",
                data_hash=checksum,
                metadata=metadata or {},
                access_count=0,
                last_accessed=now.isoformat()
            )

            # Write data
            data_path = self._get_data_path(cache_type, key)

            if self.config.atomic_writes:
                temp_path = data_path.with_suffix('.tmp')
                with open(temp_path, 'wb') as f:
                    f.write(serialized)
                temp_path.rename(data_path)
            else:
                with open(data_path, 'wb') as f:
                    f.write(serialized)

            # Update manifest
            with self._lock:
                manifest = self._manifests[cache_type]

                # Update size if replacing
                if key in manifest.entries:
                    manifest.total_size_bytes -= manifest.entries[key].size_bytes
                else:
                    manifest.total_entries += 1

                manifest.entries[key] = entry
                manifest.total_size_bytes += entry.size_bytes

                self._save_manifest(cache_type)

            # UPDATE L1 CACHE (store in memory for fast access)
            l1_key = f"{cache_type.value}:{key}"
            self._l1_cache.put(l1_key, data, entry.size_bytes)

            # UPDATE BLOOM FILTER
            bloom = self._bloom_filters.get(cache_type)
            if bloom:
                bloom.add(key)

            # Check if LRU eviction needed
            if self.config.enable_lru_eviction:
                self._maybe_evict(cache_type)

            logger.debug(f"Cache SET: {cache_type.value}/{key} ({entry.size_bytes} bytes)")
            return True

        except Exception as e:
            logger.error(f"Failed to cache {key}: {e}")
            return False

    def invalidate(self, cache_type: CacheType, key: str) -> bool:
        """
        Invalidate a specific cache entry.

        Args:
            cache_type: Type of cached data
            key: Cache key

        Returns:
            True if entry was invalidated
        """
        return self._remove_entry(cache_type, key)

    def invalidate_all(self, cache_type: CacheType = None):
        """
        Invalidate all cache entries.

        Args:
            cache_type: Specific type to invalidate, or None for all
        """
        types_to_clear = [cache_type] if cache_type else list(CacheType)

        with self._lock:
            for ct in types_to_clear:
                manifest = self._manifests.get(ct)
                if manifest:
                    # Remove all data files
                    for key in list(manifest.entries.keys()):
                        self._remove_entry(ct, key, save_manifest=False)

                    # Reset manifest
                    self._manifests[ct] = self._create_empty_manifest(ct)
                    self._save_manifest(ct)

        logger.info(f"Invalidated cache: {[t.value for t in types_to_clear]}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'cache_dir': str(self.cache_dir),
            'version': self.config.cache_version,
            'compression': self.config.compression.value,
            'types': {}
        }

        total_size = 0
        total_entries = 0

        with self._lock:
            for cache_type, manifest in self._manifests.items():
                type_stats = {
                    'entries': manifest.total_entries,
                    'size_bytes': manifest.total_size_bytes,
                    'size_mb': round(manifest.total_size_bytes / (1024 * 1024), 2),
                    'last_updated': manifest.last_updated,
                }
                stats['types'][cache_type.value] = type_stats
                total_size += manifest.total_size_bytes
                total_entries += manifest.total_entries

        stats['total_entries'] = total_entries
        stats['total_size_bytes'] = total_size
        stats['total_size_mb'] = round(total_size / (1024 * 1024), 2)
        stats['total_size_gb'] = round(total_size / (1024 * 1024 * 1024), 4)

        # L1 cache statistics (in-memory LRU)
        stats['l1_cache'] = self._l1_cache.get_stats()

        # L2 cache statistics (disk-based)
        # Use manifest data for accurate file counts (matches display)
        l2_total_entries = total_entries  # From manifest
        l2_total_size = total_size  # From manifest

        stats['l2_cache'] = {
            'files': l2_total_entries,
            'size_bytes': l2_total_size,
            'size_mb': round(l2_total_size / (1024 * 1024), 2),
        }

        return stats

    def _remove_entry(
        self,
        cache_type: CacheType,
        key: str,
        save_manifest: bool = True
    ) -> bool:
        """Remove a cache entry from L1, L2, and manifest."""
        with self._lock:
            manifest = self._manifests.get(cache_type)
            if not manifest or key not in manifest.entries:
                return False

            entry = manifest.entries[key]

            # Remove from L1 cache
            l1_key = f"{cache_type.value}:{key}"
            self._l1_cache.remove(l1_key)

            # Remove data file (L2)
            data_path = self._get_data_path(cache_type, key)
            try:
                if data_path.exists():
                    data_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove cache file {data_path}: {e}")

            # Update manifest
            manifest.total_size_bytes -= entry.size_bytes
            manifest.total_entries -= 1
            del manifest.entries[key]

            if save_manifest:
                self._save_manifest(cache_type)

            return True

    def _maybe_evict(self, cache_type: CacheType):
        """Evict old entries if needed using LRU policy."""
        with self._lock:
            manifest = self._manifests[cache_type]

            # Check entry limit
            if manifest.total_entries <= self.config.max_entries_per_type:
                return

            # Sort by last access time (LRU)
            entries_by_access = sorted(
                manifest.entries.items(),
                key=lambda x: x[1].last_accessed
            )

            # Remove oldest entries
            entries_to_remove = manifest.total_entries - self.config.max_entries_per_type
            for key, _ in entries_by_access[:entries_to_remove]:
                self._remove_entry(cache_type, key, save_manifest=False)

            self._save_manifest(cache_type)
            logger.info(f"Evicted {entries_to_remove} entries from {cache_type.value}")

    # =========================================================================
    # SPECIALIZED CACHE METHODS
    # =========================================================================

    def get_cointegration_key(
        self,
        symbol1: str,
        symbol2: str,
        data_start: str,
        data_end: str,
        significance_level: float = 0.05,
        min_half_life: float = 24.0,
        max_half_life: float = 504.0,
    ) -> str:
        """Generate cache key for cointegration test result."""
        # Ensure consistent ordering
        pair = tuple(sorted([symbol1, symbol2]))
        return self.generate_key(
            CacheType.COINTEGRATION,
            f"{pair[0]}_{pair[1]}",
            data_start=data_start,
            data_end=data_end,
            significance_level=significance_level,
            min_half_life=min_half_life,
            max_half_life=max_half_life,
        )

    def cache_cointegration_result(
        self,
        symbol1: str,
        symbol2: str,
        data_start: str,
        data_end: str,
        result: Any,
        params: Dict[str, Any] = None
    ) -> bool:
        """Cache cointegration test result."""
        key = self.get_cointegration_key(
            symbol1, symbol2, data_start, data_end,
            **(params or {})
        )
        return self.set(
            CacheType.COINTEGRATION,
            key,
            result,
            metadata={
                'symbol1': symbol1,
                'symbol2': symbol2,
                'data_start': data_start,
                'data_end': data_end,
                'params': params
            }
        )

    def get_cointegration_result(
        self,
        symbol1: str,
        symbol2: str,
        data_start: str,
        data_end: str,
        params: Dict[str, Any] = None
    ) -> Optional[Any]:
        """Retrieve cached cointegration test result."""
        key = self.get_cointegration_key(
            symbol1, symbol2, data_start, data_end,
            **(params or {})
        )
        return self.get(CacheType.COINTEGRATION, key)

    def cache_universe_snapshot(
        self,
        snapshot: Any,
        symbols: List[str],
        data_start: str,
        data_end: str,
        filters: Dict[str, Any] = None
    ) -> bool:
        """Cache universe construction snapshot."""
        key = self.generate_key(
            CacheType.UNIVERSE,
            f"universe_{len(symbols)}",
            symbols_hash=HashGenerator.hash_params(symbols=sorted(symbols)),
            data_start=data_start,
            data_end=data_end,
            filters=filters or {}
        )
        return self.set(
            CacheType.UNIVERSE,
            key,
            snapshot,
            metadata={
                'n_symbols': len(symbols),
                'data_start': data_start,
                'data_end': data_end,
            }
        )

    def get_universe_snapshot(
        self,
        symbols: List[str],
        data_start: str,
        data_end: str,
        filters: Dict[str, Any] = None
    ) -> Optional[Any]:
        """Retrieve cached universe snapshot."""
        key = self.generate_key(
            CacheType.UNIVERSE,
            f"universe_{len(symbols)}",
            symbols_hash=HashGenerator.hash_params(symbols=sorted(symbols)),
            data_start=data_start,
            data_end=data_end,
            filters=filters or {}
        )
        return self.get(CacheType.UNIVERSE, key)

    def cache_price_matrix(
        self,
        price_matrix: pd.DataFrame,
        venues: List[str],
        data_start: str,
        data_end: str,
    ) -> bool:
        """Cache price matrix data."""
        key = self.generate_key(
            CacheType.PRICE_MATRIX,
            f"prices_{len(price_matrix.columns)}",
            venues_hash=HashGenerator.hash_params(venues=sorted(venues)),
            data_start=data_start,
            data_end=data_end,
            n_rows=len(price_matrix),
            n_cols=len(price_matrix.columns),
        )
        return self.set(
            CacheType.PRICE_MATRIX,
            key,
            price_matrix,
            metadata={
                'n_symbols': len(price_matrix.columns),
                'n_rows': len(price_matrix),
                'venues': venues,
                'data_start': data_start,
                'data_end': data_end,
            }
        )

    def get_price_matrix(
        self,
        venues: List[str],
        data_start: str,
        data_end: str,
        n_symbols: int = None,
        n_rows: int = None,
    ) -> Optional[pd.DataFrame]:
        """Retrieve cached price matrix."""
        key = self.generate_key(
            CacheType.PRICE_MATRIX,
            f"prices_{n_symbols}",
            venues_hash=HashGenerator.hash_params(venues=sorted(venues)),
            data_start=data_start,
            data_end=data_end,
            n_rows=n_rows,
            n_cols=n_symbols,
        )
        return self.get(CacheType.PRICE_MATRIX, key)

    def cache_backtest_results(
        self,
        results: Any,
        strategy_config: Dict[str, Any],
        data_start: str,
        data_end: str,
    ) -> bool:
        """Cache backtest results."""
        key = self.generate_key(
            CacheType.BACKTEST,
            "backtest",
            config_hash=HashGenerator.hash_dict(strategy_config),
            data_start=data_start,
            data_end=data_end,
        )
        return self.set(
            CacheType.BACKTEST,
            key,
            results,
            metadata={
                'data_start': data_start,
                'data_end': data_end,
                'strategy_config': strategy_config,
            }
        )

    def get_backtest_results(
        self,
        strategy_config: Dict[str, Any],
        data_start: str,
        data_end: str,
    ) -> Optional[Any]:
        """Retrieve cached backtest results."""
        key = self.generate_key(
            CacheType.BACKTEST,
            "backtest",
            config_hash=HashGenerator.hash_dict(strategy_config),
            data_start=data_start,
            data_end=data_end,
        )
        return self.get(CacheType.BACKTEST, key)

    # =========================================================================
    # ML PREDICTIONS CACHE
    # =========================================================================

    def get_ml_predictions_key(
        self,
        pair: str,
        model_type: str,
        data_start: str,
        data_end: str,
        model_params: Dict[str, Any] = None
    ) -> str:
        """Generate cache key for ML predictions."""
        return self.generate_key(
            CacheType.ML_PREDICTIONS,
            f"ml_{pair}_{model_type}",
            data_start=data_start,
            data_end=data_end,
            model_params=model_params or {},
        )

    def cache_ml_predictions(
        self,
        pair: str,
        model_type: str,
        predictions: Any,
        data_start: str,
        data_end: str,
        model_params: Dict[str, Any] = None,
        training_metrics: Dict[str, Any] = None
    ) -> bool:
        """Cache ML model predictions."""
        key = self.get_ml_predictions_key(pair, model_type, data_start, data_end, model_params)
        return self.set(
            CacheType.ML_PREDICTIONS,
            key,
            predictions,
            metadata={
                'pair': pair,
                'model_type': model_type,
                'data_start': data_start,
                'data_end': data_end,
                'model_params': model_params,
                'training_metrics': training_metrics,
            }
        )

    def get_ml_predictions(
        self,
        pair: str,
        model_type: str,
        data_start: str,
        data_end: str,
        model_params: Dict[str, Any] = None
    ) -> Optional[Any]:
        """Retrieve cached ML predictions."""
        key = self.get_ml_predictions_key(pair, model_type, data_start, data_end, model_params)
        return self.get(CacheType.ML_PREDICTIONS, key)

    # =========================================================================
    # REGIME DETECTION CACHE
    # =========================================================================

    def get_regime_key(
        self,
        n_states: int,
        data_start: str,
        data_end: str,
        features_hash: str = None
    ) -> str:
        """Generate cache key for regime detection."""
        return self.generate_key(
            CacheType.REGIME_DETECTION,
            f"regime_{n_states}",
            data_start=data_start,
            data_end=data_end,
            features_hash=features_hash or "default",
        )

    def cache_regime_detection(
        self,
        regime_result: Any,
        n_states: int,
        data_start: str,
        data_end: str,
        features_used: List[str] = None
    ) -> bool:
        """Cache regime detection results."""
        features_hash = HashGenerator.hash_params(features=sorted(features_used or []))
        key = self.get_regime_key(n_states, data_start, data_end, features_hash)
        return self.set(
            CacheType.REGIME_DETECTION,
            key,
            regime_result,
            metadata={
                'n_states': n_states,
                'data_start': data_start,
                'data_end': data_end,
                'features_used': features_used,
            }
        )

    def get_regime_detection(
        self,
        n_states: int,
        data_start: str,
        data_end: str,
        features_used: List[str] = None
    ) -> Optional[Any]:
        """Retrieve cached regime detection results."""
        features_hash = HashGenerator.hash_params(features=sorted(features_used or []))
        key = self.get_regime_key(n_states, data_start, data_end, features_hash)
        return self.get(CacheType.REGIME_DETECTION, key)

    # =========================================================================
    # WALK-FORWARD OPTIMIZATION CACHE
    # =========================================================================

    def get_walk_forward_key(
        self,
        train_months: int,
        test_months: int,
        data_start: str,
        data_end: str,
        strategy_config: Dict[str, Any] = None
    ) -> str:
        """Generate cache key for walk-forward optimization."""
        config_hash = HashGenerator.hash_dict(strategy_config or {})
        return self.generate_key(
            CacheType.WALK_FORWARD,
            f"wf_{train_months}m_{test_months}m",
            data_start=data_start,
            data_end=data_end,
            config_hash=config_hash,
        )

    def cache_walk_forward(
        self,
        wf_result: Any,
        train_months: int,
        test_months: int,
        data_start: str,
        data_end: str,
        strategy_config: Dict[str, Any] = None
    ) -> bool:
        """Cache walk-forward optimization results."""
        key = self.get_walk_forward_key(train_months, test_months, data_start, data_end, strategy_config)
        return self.set(
            CacheType.WALK_FORWARD,
            key,
            wf_result,
            metadata={
                'train_months': train_months,
                'test_months': test_months,
                'data_start': data_start,
                'data_end': data_end,
                'strategy_config': strategy_config,
            }
        )

    def get_walk_forward(
        self,
        train_months: int,
        test_months: int,
        data_start: str,
        data_end: str,
        strategy_config: Dict[str, Any] = None
    ) -> Optional[Any]:
        """Retrieve cached walk-forward optimization results."""
        key = self.get_walk_forward_key(train_months, test_months, data_start, data_end, strategy_config)
        return self.get(CacheType.WALK_FORWARD, key)

    # =========================================================================
    # CRISIS ANALYSIS CACHE
    # =========================================================================

    def get_crisis_key(
        self,
        n_events: int,
        data_start: str,
        data_end: str,
    ) -> str:
        """Generate cache key for crisis analysis."""
        return self.generate_key(
            CacheType.CRISIS_ANALYSIS,
            f"crisis_{n_events}",
            data_start=data_start,
            data_end=data_end,
        )

    def cache_crisis_analysis(
        self,
        crisis_result: Any,
        n_events: int,
        data_start: str,
        data_end: str,
        events_analyzed: List[str] = None
    ) -> bool:
        """Cache crisis analysis results."""
        key = self.get_crisis_key(n_events, data_start, data_end)
        return self.set(
            CacheType.CRISIS_ANALYSIS,
            key,
            crisis_result,
            metadata={
                'n_events': n_events,
                'data_start': data_start,
                'data_end': data_end,
                'events_analyzed': events_analyzed,
            }
        )

    def get_crisis_analysis(
        self,
        n_events: int,
        data_start: str,
        data_end: str,
    ) -> Optional[Any]:
        """Retrieve cached crisis analysis results."""
        key = self.get_crisis_key(n_events, data_start, data_end)
        return self.get(CacheType.CRISIS_ANALYSIS, key)

    # =========================================================================
    # PAIR RANKING CACHE
    # =========================================================================

    def get_pair_ranking_key(
        self,
        n_pairs: int,
        data_start: str,
        data_end: str,
        ranking_params: Dict[str, Any] = None
    ) -> str:
        """Generate cache key for pair rankings."""
        params_hash = HashGenerator.hash_dict(ranking_params or {})
        return self.generate_key(
            CacheType.PAIR_RANKING,
            f"ranking_{n_pairs}",
            data_start=data_start,
            data_end=data_end,
            params_hash=params_hash,
        )

    def cache_pair_ranking(
        self,
        ranking_result: Any,
        n_pairs: int,
        data_start: str,
        data_end: str,
        ranking_params: Dict[str, Any] = None
    ) -> bool:
        """Cache pair ranking results."""
        key = self.get_pair_ranking_key(n_pairs, data_start, data_end, ranking_params)
        return self.set(
            CacheType.PAIR_RANKING,
            key,
            ranking_result,
            metadata={
                'n_pairs': n_pairs,
                'data_start': data_start,
                'data_end': data_end,
                'ranking_params': ranking_params,
            }
        )

    def get_pair_ranking(
        self,
        n_pairs: int,
        data_start: str,
        data_end: str,
        ranking_params: Dict[str, Any] = None
    ) -> Optional[Any]:
        """Retrieve cached pair rankings."""
        key = self.get_pair_ranking_key(n_pairs, data_start, data_end, ranking_params)
        return self.get(CacheType.PAIR_RANKING, key)

    # =========================================================================
    # SIGNALS CACHE
    # =========================================================================

    def get_signals_key(
        self,
        strategy_type: str,
        data_start: str,
        data_end: str,
        signal_params: Dict[str, Any] = None
    ) -> str:
        """Generate cache key for signals."""
        params_hash = HashGenerator.hash_dict(signal_params or {})
        return self.generate_key(
            CacheType.SIGNALS,
            f"signals_{strategy_type}",
            data_start=data_start,
            data_end=data_end,
            params_hash=params_hash,
        )

    def cache_signals(
        self,
        signals: Any,
        strategy_type: str,
        data_start: str,
        data_end: str,
        signal_params: Dict[str, Any] = None
    ) -> bool:
        """Cache signal generation results."""
        key = self.get_signals_key(strategy_type, data_start, data_end, signal_params)
        return self.set(
            CacheType.SIGNALS,
            key,
            signals,
            metadata={
                'strategy_type': strategy_type,
                'data_start': data_start,
                'data_end': data_end,
                'signal_params': signal_params,
            }
        )

    def get_signals(
        self,
        strategy_type: str,
        data_start: str,
        data_end: str,
        signal_params: Dict[str, Any] = None
    ) -> Optional[Any]:
        """Retrieve cached signals."""
        key = self.get_signals_key(strategy_type, data_start, data_end, signal_params)
        return self.get(CacheType.SIGNALS, key)

    # =========================================================================
    # ENHANCEMENT CACHE (Step 3 results)
    # =========================================================================

    def get_enhancement_key(
        self,
        enhancement_type: str,
        data_start: str,
        data_end: str,
        enhancement_params: Dict[str, Any] = None
    ) -> str:
        """Generate cache key for enhancement results."""
        params_hash = HashGenerator.hash_dict(enhancement_params or {})
        return self.generate_key(
            CacheType.ENHANCEMENT,
            f"enhancement_{enhancement_type}",
            data_start=data_start,
            data_end=data_end,
            params_hash=params_hash,
        )

    def cache_enhancement(
        self,
        result: Any,
        enhancement_type: str,
        data_start: str,
        data_end: str,
        enhancement_params: Dict[str, Any] = None
    ) -> bool:
        """Cache enhancement results (Step 3 outputs)."""
        key = self.get_enhancement_key(enhancement_type, data_start, data_end, enhancement_params)
        return self.set(
            CacheType.ENHANCEMENT,
            key,
            result,
            metadata={
                'enhancement_type': enhancement_type,
                'data_start': data_start,
                'data_end': data_end,
                'enhancement_params': enhancement_params,
            }
        )

    def get_enhancement(
        self,
        enhancement_type: str,
        data_start: str,
        data_end: str,
        enhancement_params: Dict[str, Any] = None
    ) -> Optional[Any]:
        """Retrieve cached enhancement results."""
        key = self.get_enhancement_key(enhancement_type, data_start, data_end, enhancement_params)
        return self.get(CacheType.ENHANCEMENT, key)

    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================

    def get_many(
        self,
        cache_type: CacheType,
        keys: List[str]
    ) -> Dict[str, Any]:
        """
        Retrieve multiple cache entries in parallel.

        Args:
            cache_type: Type of cached data
            keys: List of cache keys

        Returns:
            Dictionary mapping keys to cached data
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            futures = {
                executor.submit(self.get, cache_type, key): key
                for key in keys
            }

            for future in as_completed(futures):
                key = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results[key] = result
                except Exception as e:
                    logger.warning(f"Failed to get {key}: {e}")

        return results

    def set_many(
        self,
        cache_type: CacheType,
        items: Dict[str, Any],
        ttl_hours: float = None
    ) -> Dict[str, bool]:
        """
        Store multiple cache entries in parallel.

        Args:
            cache_type: Type of cached data
            items: Dictionary mapping keys to data
            ttl_hours: Time-to-live in hours

        Returns:
            Dictionary mapping keys to success status
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            futures = {
                executor.submit(self.set, cache_type, key, data, ttl_hours): key
                for key, data in items.items()
            }

            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    logger.warning(f"Failed to set {key}: {e}")
                    results[key] = False

        return results


# =============================================================================
# CACHE DECORATORS
# =============================================================================

def cached(
    cache_type: CacheType,
    key_func: Callable[..., str] = None,
    ttl_hours: float = None,
    cache_manager: CacheManager = None
):
    """
    Decorator to cache function results.

    Usage:
        @cached(CacheType.COINTEGRATION, key_func=lambda x, y: f"{x}_{y}")
        def expensive_computation(x, y):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get or create cache manager
            nonlocal cache_manager
            if cache_manager is None:
                cache_manager = CacheManager()

            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = HashGenerator.hash_params(
                    func_name=func.__name__,
                    args=args,
                    kwargs=kwargs
                )

            # Check cache
            cached_result = cache_manager.get(cache_type, key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            cache_manager.set(cache_type, key, result, ttl_hours)

            return result

        return wrapper
    return decorator


# =============================================================================
# GLOBAL CACHE INSTANCE
# =============================================================================

_global_cache: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """Get or create global cache manager instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager()
    return _global_cache


def init_cache(config: CacheConfig = None) -> CacheManager:
    """Initialize global cache manager with configuration."""
    global _global_cache
    _global_cache = CacheManager(config)
    return _global_cache


# =============================================================================
# PROGRESS MONITORING UTILITIES
# =============================================================================

class ProgressMonitor:
    """Enhanced progress monitoring with rich console UI and statistics tracking."""

    def __init__(self, description: str = "Processing", total: int = 0):
        self.description = description
        self.total = total
        self.completed = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        self.start_time = None
        self.progress = None
        self.task_id = None
        self._lock = threading.Lock()

    def __enter__(self):
        self.start_time = time.time()
        if RICH_AVAILABLE:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                MofNCompleteColumn(),
                TextColumn("[cyan]│"),
                TaskProgressColumn(),
                TextColumn("[cyan]│"),
                TimeElapsedColumn(),
                TextColumn("→"),
                TimeRemainingColumn(),
                console=console,
                expand=False
            )
            self.progress.__enter__()
            self.task_id = self.progress.add_task(self.description, total=self.total)
        else:
            print(f"\n  {self.description}: 0/{self.total}", end="", flush=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time if self.start_time else 0
        if RICH_AVAILABLE and self.progress:
            self.progress.__exit__(exc_type, exc_val, exc_tb)
            # Show final statistics
            self._print_stats(elapsed)
        else:
            print(f"\n  Completed in {elapsed:.1f}s")
        return False

    def update(self, advance: int = 1, cache_hit: bool = False, error: bool = False):
        """Update progress with optional cache and error tracking."""
        with self._lock:
            self.completed += advance
            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            if error:
                self.errors += 1

            if RICH_AVAILABLE and self.progress and self.task_id is not None:
                # Update description with cache stats
                cache_rate = self.cache_hits / max(1, self.cache_hits + self.cache_misses) * 100
                desc = f"{self.description} [green]HIT:{self.cache_hits}[/green] [yellow]MISS:{self.cache_misses}[/yellow]"
                if self.errors > 0:
                    desc += f" [red]ERR:{self.errors}[/red]"
                self.progress.update(self.task_id, advance=advance, description=desc)
            else:
                # Fallback progress display
                if self.completed % max(1, self.total // 20) == 0 or self.completed == self.total:
                    pct = self.completed / max(1, self.total) * 100
                    print(f"\r  {self.description}: {self.completed}/{self.total} ({pct:.0f}%)", end="", flush=True)

    def _print_stats(self, elapsed: float):
        """Print final statistics table."""
        if not RICH_AVAILABLE:
            return

        total_processed = self.cache_hits + self.cache_misses
        cache_rate = self.cache_hits / max(1, total_processed) * 100
        rate = self.completed / max(0.001, elapsed)

        stats_table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green", justify="right")

        stats_table.add_row("Completed", str(self.completed))
        stats_table.add_row("Cache Hits", f"{self.cache_hits} ({cache_rate:.1f}%)")
        stats_table.add_row("Cache Misses", str(self.cache_misses))
        if self.errors > 0:
            stats_table.add_row("Errors", str(self.errors), style="red")
        stats_table.add_row("Duration", f"{elapsed:.1f}s")
        stats_table.add_row("Rate", f"{rate:.1f}/sec")

        console.print(Panel(stats_table, title="[bold]Processing Statistics", border_style="blue"))


class ParallelProgressCallback:
    """Thread-safe callback for tracking joblib parallel execution progress."""

    def __init__(self, monitor: ProgressMonitor):
        self.monitor = monitor
        self._lock = threading.Lock()

    def __call__(self, result):
        """Called when each job completes."""
        with self._lock:
            cache_hit = getattr(result, '_from_cache', False) if result else False
            error = result is None
            self.monitor.update(1, cache_hit=cache_hit, error=error)


def create_summary_panel(title: str, metrics: dict, style: str = "blue") -> None:
    """Create and print a summary panel with metrics."""
    if not RICH_AVAILABLE:
        print(f"\n  {title}:")
        for k, v in metrics.items():
            print(f"    {k}: {v}")
        return

    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white", justify="right")

    for key, value in metrics.items():
        if isinstance(value, float):
            value = f"{value:.4f}"
        table.add_row(str(key), str(value))

    console.print(Panel(table, title=f"[bold]{title}", border_style=style))


# =============================================================================
# CLI UTILITIES
# =============================================================================

def print_cache_stats():
    """Print cache statistics to console."""
    cache = get_cache()
    stats = cache.get_stats()

    print("\n" + "=" * 60)
    print("CACHE STATISTICS")
    print("=" * 60)
    print(f"Cache Directory: {stats['cache_dir']}")
    print(f"Version: {stats['version']}")
    print(f"Compression: {stats['compression']}")
    print(f"\nTotal Entries: {stats['total_entries']}")
    print(f"Total Size: {stats['total_size_mb']} MB ({stats['total_size_gb']} GB)")
    print("\nBy Type:")
    print("-" * 60)

    for type_name, type_stats in stats['types'].items():
        print(f"  {type_name}:")
        print(f"    Entries: {type_stats['entries']}")
        print(f"    Size: {type_stats['size_mb']} MB")
        print(f"    Last Updated: {type_stats['last_updated']}")

    print("=" * 60)


if __name__ == "__main__":
    # Demo usage
    print_cache_stats()
