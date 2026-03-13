"""
Incremental Data Cache Manager for Crypto Statistical Arbitrage System.

This module provides comprehensive caching infrastructure that:
1. Tracks fully processed data (cleaned, normalized, validated)
2. Determines what date ranges need to be collected
3. Merges new data with existing cached data
4. Eliminates redundant collection of historical data

CRITICAL: Only caches data AFTER the full pipeline is executed (normalized, cleaned, validated).
This ensures data quality and consistency.

Architecture:
    
                        IncrementalCacheManager 
        
                           Cache Metadata Store 
                             
         funding_rates ohlcv open_interest ... 
                                   
          binance binance binance 
          BTC: BTC: BTC: 
          2020-01 2020-01 2020-01 
          to to to 
          2026-02 2026-02 2026-02 
                                   
                             
        
                                                                              
        
                          Processed Data Store (Parquet) 
        data/processed/ 
           funding_rates/ 
              binance_funding_rates.parquet 
              hyperliquid_funding_rates.parquet 
              ... 
           ohlcv/ 
              binance_ohlcv_1h.parquet 
              ... 
           cache_metadata.json 
        
    

Usage:
    >>> cache = IncrementalCacheManager('data/processed')
    >>>
    >>> # Check what needs to be collected
    >>> gaps = cache.get_collection_gaps(
    ... data_type='funding_rates',
    ... venue='binance',
    ... symbols=['BTC', 'ETH'],
    ... start_date='2020-01-01',
    ... end_date='2026-02-03'
    ... )
    >>> # gaps = [('2026-01-15', '2026-02-03')] # Only collect new data
    >>>
    >>> # After pipeline completes, update cache
    >>> cache.update_cache(
    ... data_type='funding_rates',
    ... venue='binance',
    ... data=processed_df,
    ... start_date='2026-01-15',
    ... end_date='2026-02-03'
    ... )

Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import hashlib

import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# SYMBOL NORMALIZATION UTILITIES
# =============================================================================

def normalize_symbol(symbol: str) -> str:
    """
    Normalize a trading symbol to its base asset.

    Handles various exchange-specific formats:
    - BTC-USDT-SWAP -> BTC
    - BTCUSDT -> BTC
    - BTC/USDT -> BTC
    - BTC-PERP -> BTC
    - XBTUSD -> BTC
    - BTC -> BTC

    Args:
        symbol: Exchange-specific symbol format

    Returns:
        Normalized base symbol
    """
    if not symbol:
        return symbol

    # Common quote currencies to strip (only used for concatenated symbols)
    quotes = {'USDT', 'USD', 'USDC', 'BUSD', 'EUR', 'TUSD', 'DAI', 'FDUSD'}
    # Suffixes that should be removed from split symbols
    suffixes = {'SWAP', 'PERP', 'PERPETUAL', 'FUTURES', 'FUT', 'SPOT', 'LINEAR', 'INVERSE'}

    # Handle XBT -> BTC conversion
    result = symbol.upper().replace('XBT', 'BTC')

    # Handle separators: - / _ :
    # For symbols like BTC-USDT-SWAP, take the FIRST part (base asset)
    # For symbols like BTCUSDT, handle separately below
    for sep in ['-', '/', '_', ':']:
        if sep in result:
            parts = result.split(sep)
            # First part is typically the base asset
            # Filter out empty parts and known suffixes
            valid_parts = [p for p in parts if p and p not in suffixes]
            if valid_parts:
                # Take the first valid part as the base asset
                result = valid_parts[0]
            break # Exit after processing first separator type found

    # Handle concatenated symbols (BTCUSDT -> BTC)
    # Must do this AFTER separator handling
    for quote in sorted(quotes, key=len, reverse=True): # Try longer quotes first
        if result.endswith(quote) and len(result) > len(quote):
            result = result[:-len(quote)]
            break

    return result

def normalize_symbols(symbols: List[str]) -> Set[str]:
    """
    Normalize a list of symbols to base assets.

    Args:
        symbols: List of exchange-specific symbols

    Returns:
        Set of normalized base symbols
    """
    return {normalize_symbol(s) for s in symbols if s}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CacheEntry:
    """Metadata for a single cached data segment."""
    data_type: str
    venue: str
    symbols: List[str]
    start_date: str # YYYY-MM-DD
    end_date: str # YYYY-MM-DD
    record_count: int
    file_path: str
    created_at: str
    checksum: str
    quality_score: float = 0.0
    is_normalized: bool = False
    is_validated: bool = False
    timeframe: Optional[str] = None # For OHLCV data

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'CacheEntry':
        return cls(**d)

    def covers_range(self, start: str, end: str) -> bool:
        """Check if this entry covers the requested date range."""
        return self.start_date <= start and self.end_date >= end

    def overlaps_range(self, start: str, end: str) -> bool:
        """Check if this entry overlaps with the requested date range."""
        return not (self.end_date < start or self.start_date > end)

    def get_gap_before(self, start: str) -> Optional[Tuple[str, str]]:
        """Get the gap between requested start and cached start."""
        if start < self.start_date:
            return (start, self._day_before(self.start_date))
        return None

    def get_gap_after(self, end: str) -> Optional[Tuple[str, str]]:
        """Get the gap between cached end and requested end."""
        if end > self.end_date:
            return (self._day_after(self.end_date), end)
        return None

    @staticmethod
    def _day_before(date_str: str) -> str:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return (dt - timedelta(days=1)).strftime('%Y-%m-%d')

    @staticmethod
    def _day_after(date_str: str) -> str:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return (dt + timedelta(days=1)).strftime('%Y-%m-%d')

@dataclass
class CacheMetadata:
    """Complete cache metadata for all data types and venues."""
    version: str = "1.0.0"
    created_at: str = ""
    updated_at: str = ""
    entries: Dict[str, Dict[str, List[CacheEntry]]] = field(default_factory=dict)
    # Structure: entries[data_type][venue] = [CacheEntry, ...]

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        result = {
            'version': self.version,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'entries': {}
        }
        for data_type, venues in self.entries.items():
            result['entries'][data_type] = {}
            for venue, entries in venues.items():
                result['entries'][data_type][venue] = [e.to_dict() for e in entries]
        return result

    @classmethod
    def from_dict(cls, d: Dict) -> 'CacheMetadata':
        metadata = cls(
            version=d.get('version', '1.0.0'),
            created_at=d.get('created_at', ''),
            updated_at=d.get('updated_at', '')
        )
        for data_type, venues in d.get('entries', {}).items():
            metadata.entries[data_type] = {}
            for venue, entries in venues.items():
                metadata.entries[data_type][venue] = [
                    CacheEntry.from_dict(e) for e in entries
                ]
        return metadata

    def get_entries(self, data_type: str, venue: str) -> List[CacheEntry]:
        """Get all cache entries for a data type and venue."""
        return self.entries.get(data_type, {}).get(venue, [])

    def add_entry(self, entry: CacheEntry) -> None:
        """Add a cache entry."""
        if entry.data_type not in self.entries:
            self.entries[entry.data_type] = {}
        if entry.venue not in self.entries[entry.data_type]:
            self.entries[entry.data_type][entry.venue] = []

        # Merge overlapping entries
        self._merge_entry(entry)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def _merge_entry(self, new_entry: CacheEntry) -> None:
        """Merge a new entry with existing entries, consolidating overlaps."""
        entries = self.entries[new_entry.data_type][new_entry.venue]

        # Find overlapping entries
        overlapping = [e for e in entries if e.overlaps_range(new_entry.start_date, new_entry.end_date)]

        if not overlapping:
            entries.append(new_entry)
            return

        # Merge all overlapping entries into one
        all_entries = overlapping + [new_entry]
        min_start = min(e.start_date for e in all_entries)
        max_end = max(e.end_date for e in all_entries)
        total_records = sum(e.record_count for e in all_entries)
        all_symbols = set()
        for e in all_entries:
            all_symbols.update(e.symbols)

        # Remove old overlapping entries
        for e in overlapping:
            entries.remove(e)

        # Add merged entry
        merged = CacheEntry(
            data_type=new_entry.data_type,
            venue=new_entry.venue,
            symbols=sorted(list(all_symbols)),
            start_date=min_start,
            end_date=max_end,
            record_count=total_records,
            file_path=new_entry.file_path,
            created_at=new_entry.created_at,
            checksum=new_entry.checksum,
            quality_score=new_entry.quality_score,
            is_normalized=new_entry.is_normalized,
            is_validated=new_entry.is_validated,
            timeframe=new_entry.timeframe
        )
        entries.append(merged)

# =============================================================================
# INCREMENTAL CACHE MANAGER
# =============================================================================

class IncrementalCacheManager:
    """
    comprehensive cache manager for incremental data collection.

    Key Features:
    - Tracks fully processed data (post-normalization, post-validation)
    - Determines exact date ranges that need collection
    - Merges new data with existing cached data
    - Atomic updates to prevent corruption
    - Checksum verification for data integrity

    Usage:
        >>> cache = IncrementalCacheManager('data/processed')
        >>>
        >>> # Get gaps that need collection
        >>> gaps = cache.get_collection_gaps(
        ... 'funding_rates', 'binance', ['BTC'], '2020-01-01', '2026-02-03'
        ... )
        >>>
        >>> # Load cached data
        >>> cached_df = cache.load_cached_data('funding_rates', 'binance')
        >>>
        >>> # After processing, update cache
        >>> cache.update_cache('funding_rates', 'binance', new_df, '2026-01-01', '2026-02-03')
    """

    METADATA_FILE = 'cache_metadata.json'

    def __init__(self, cache_dir: str = 'data/processed'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.cache_dir / self.METADATA_FILE
        self.metadata = self._load_metadata()

        self._lock = asyncio.Lock()

        logger.info(f"IncrementalCacheManager initialized: {self.cache_dir}")
        self._log_cache_status()

    def _load_metadata(self) -> CacheMetadata:
        """Load cache metadata from disk."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    data = json.load(f)
                return CacheMetadata.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return CacheMetadata()

    def _save_metadata(self) -> None:
        """Save cache metadata to disk atomically."""
        temp_path = self.metadata_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(self.metadata.to_dict(), f, indent=2)
            temp_path.replace(self.metadata_path)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _log_cache_status(self) -> None:
        """Log current cache status."""
        total_entries = 0
        for data_type, venues in self.metadata.entries.items():
            for venue, entries in venues.items():
                total_entries += len(entries)

        if total_entries > 0:
            logger.info(f"Cache contains {total_entries} entries across "
                       f"{len(self.metadata.entries)} data types")
        else:
            logger.info("Cache is empty - will collect all requested data")

    def _get_file_path(
        self,
        data_type: str,
        venue: str,
        timeframe: Optional[str] = None,
        for_write: bool = False
    ) -> Path:
        """
        Get the file path for cached data.

        Handles multiple file structures:
        1. data/processed/<data_type>/<venue>_<data_type>.parquet (new structure)
        2. data/processed/<venue>/<data_type>.parquet (existing structure)
        3. data/processed/<data_type>/<venue>/<data_type>.parquet (alternate structure)
        4. data/processed/<data_type>/<venue>/<venue>_<data_type>_*.parquet (date-versioned files)
        """
        # Primary path structure
        data_dir = self.cache_dir / data_type

        if timeframe:
            primary_path = data_dir / f"{venue}_{data_type}_{timeframe}.parquet"
        else:
            primary_path = data_dir / f"{venue}_{data_type}.parquet"

        # For writing, always use primary structure
        if for_write:
            data_dir.mkdir(parents=True, exist_ok=True)
            return primary_path

        # For reading, check multiple possible locations
        alternate_paths = [
            # Structure 1: data/processed/<venue>/<data_type>.parquet
            self.cache_dir / venue / f"{data_type}.parquet",
            # Structure 2: data/processed/<venue>/<venue>_<data_type>.parquet
            self.cache_dir / venue / f"{venue}_{data_type}.parquet",
            # Structure 3: data/processed/<data_type>/<venue>/<data_type>.parquet
            self.cache_dir / data_type / venue / f"{data_type}.parquet",
            # Structure 4: data/processed/<data_type>/<venue>/ohlcv.parquet (for OHLCV)
            self.cache_dir / data_type / venue / "ohlcv.parquet" if data_type == 'ohlcv' else None,
        ]

        # Check if primary path exists first
        if primary_path.exists():
            return primary_path

        # Check alternate paths
        for alt_path in alternate_paths:
            if alt_path and alt_path.exists():
                return alt_path

        # CRITICAL: Check for date-versioned files in venue subdirectory
        # Pattern: data/processed/<data_type>/<venue>/<venue>_<data_type>_*.parquet
        venue_dir = self.cache_dir / data_type / venue
        if venue_dir.exists():
            # Find the most recent matching parquet file
            pattern = f"{venue}_{data_type}_*.parquet"
            matching_files = list(venue_dir.glob(pattern))
            if matching_files:
                # Sort by modification time, return most recent
                matching_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return matching_files[0]

        # Default to primary path (for new files)
        return primary_path

    def _compute_checksum(self, df: pd.DataFrame) -> str:
        """Compute checksum of DataFrame for integrity verification."""
        # Use hash of shape + dtypes + first/last rows
        content = f"{df.shape}_{df.dtypes.to_dict()}"
        if len(df) > 0:
            content += f"_{df.iloc[0].to_dict()}_{df.iloc[-1].to_dict()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_collection_gaps(
        self,
        data_type: str,
        venue: str,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """
        Determine what date ranges need to be collected.

        Returns a list of (start_date, end_date) tuples representing gaps
        that are not in the cache and need to be collected.

        Args:
            data_type: Type of data (funding_rates, ohlcv, etc.)
            venue: Venue name
            symbols: List of symbols
            start_date: Requested start date (YYYY-MM-DD)
            end_date: Requested end date (YYYY-MM-DD)
            timeframe: Timeframe for OHLCV data

        Returns:
            List of (start, end) date tuples that need collection
        """
        entries = self.metadata.get_entries(data_type, venue)

        if not entries:
            # No cache - need to collect everything
            return [(start_date, end_date)]

        # CRITICAL FIX: Collect ALL matching entries and merge their ranges
        # This handles multiple cache entries (e.g., multiple no-data markers)
        # that together cover the requested date range
        matching_entries = []
        no_data_entries = []

        # CRITICAL FIX: Normalize symbols for comparison
        # This handles exchange-specific formats (BTC-USDT-SWAP) vs base symbols (BTC)
        requested_symbols_normalized = normalize_symbols(symbols)

        for entry in entries:
            # Check timeframe for OHLCV first
            if timeframe and entry.timeframe and entry.timeframe != timeframe:
                continue

            # Check if entry overlaps our requested range
            if not entry.overlaps_range(start_date, end_date):
                continue

            # CRITICAL: Handle "no data available" markers
            # These entries have record_count=0 and indicate we already checked this range
            # They should bypass symbol matching since they apply to ALL symbols
            if entry.record_count == 0:
                no_data_entries.append(entry)
                continue

            # CRITICAL FIX: Date gaps should focus on DATE coverage, not symbol coverage
            # Symbol gaps are now handled separately by get_missing_symbols()
            # Accept the entry if it has ANY data for this venue/data_type
            # The executor will use get_missing_symbols() to find which symbols need collection
            cached_symbols_normalized = normalize_symbols(entry.symbols)

            # Accept any entry that has data (record_count > 0 already checked above)
            # Log the overlap for debugging purposes
            if len(requested_symbols_normalized) > 0 and len(cached_symbols_normalized) > 0:
                overlap = cached_symbols_normalized & requested_symbols_normalized
                symbol_overlap = len(overlap) / len(requested_symbols_normalized)
                if symbol_overlap < 1.0:
                    logger.debug(
                        f"[{data_type}][{venue}] Symbol coverage {symbol_overlap:.1%}: "
                        f"cached={len(cached_symbols_normalized)}, requested={len(requested_symbols_normalized)}, "
                        f"missing={len(requested_symbols_normalized) - len(overlap)}"
                    )

            # Always accept the entry - symbol gaps handled by get_missing_symbols()
            matching_entries.append(entry)

        # CRITICAL FIX: No-data markers take precedence over data entries
        # If ANY no-data marker covers a range, we don't need to collect that range
        # Merge all no-data markers first, then merge with data entries
        all_matching = no_data_entries + matching_entries

        if not all_matching:
            return [(start_date, end_date)]

        # CRITICAL FIX: Merge all entry ranges to find overall coverage
        # This prevents re-fetching when multiple entries together cover the range
        covered_ranges = []
        for entry in all_matching:
            # Clip entry range to our requested range
            entry_start = max(entry.start_date, start_date)
            entry_end = min(entry.end_date, end_date)
            if entry_start <= entry_end:
                covered_ranges.append((entry_start, entry_end))

        if not covered_ranges:
            return [(start_date, end_date)]

        # Sort and merge overlapping/adjacent ranges
        covered_ranges.sort(key=lambda x: x[0])
        merged_ranges = []
        current_start, current_end = covered_ranges[0]

        for range_start, range_end in covered_ranges[1:]:
            # Check if ranges overlap or are adjacent
            next_day = CacheEntry._day_after(current_end)
            if range_start <= next_day:
                # Merge ranges
                current_end = max(current_end, range_end)
            else:
                # Gap between ranges - save current and start new
                merged_ranges.append((current_start, current_end))
                current_start, current_end = range_start, range_end

        merged_ranges.append((current_start, current_end))

        # Calculate gaps based on merged covered ranges
        gaps = []

        # Gap before first covered range
        if start_date < merged_ranges[0][0]:
            gap_end = CacheEntry._day_before(merged_ranges[0][0])
            if gap_end >= start_date:
                gaps.append((start_date, gap_end))

        # Gaps between covered ranges
        for i in range(len(merged_ranges) - 1):
            gap_start = CacheEntry._day_after(merged_ranges[i][1])
            gap_end = CacheEntry._day_before(merged_ranges[i + 1][0])
            if gap_start <= gap_end:
                gaps.append((gap_start, gap_end))

        # Gap after last covered range
        if end_date > merged_ranges[-1][1]:
            gap_start = CacheEntry._day_after(merged_ranges[-1][1])
            if gap_start <= end_date:
                gaps.append((gap_start, end_date))

        return gaps

    def get_missing_symbols(
        self,
        data_type: str,
        venue: str,
        symbols: List[str],
        timeframe: Optional[str] = None
    ) -> List[str]:
        """
        Get symbols that are not in the cache for a venue/data_type.

        This is CRITICAL for incremental collection - it allows collecting
        only the symbols that are missing without clearing the cache.

        Args:
            data_type: Type of data (funding_rates, ohlcv, etc.)
            venue: Venue name
            symbols: List of requested symbols
            timeframe: Timeframe for OHLCV data

        Returns:
            List of symbols that are NOT in the cache and need collection
        """
        if not symbols:
            return []

        # Normalize requested symbols
        requested_normalized = {normalize_symbol(s): s for s in symbols}
        requested_set = set(requested_normalized.keys())

        # Get all entries for this venue/data_type
        entries = self.metadata.get_entries(data_type, venue)

        if not entries:
            # No cache - all symbols need collection
            return symbols

        # Collect all cached symbols across all entries
        cached_normalized = set()
        for entry in entries:
            # Filter by timeframe if specified
            if timeframe and entry.timeframe and entry.timeframe != timeframe:
                continue

            # Skip no-data markers (record_count=0)
            if entry.record_count == 0:
                continue

            # Add normalized symbols from this entry
            for sym in entry.symbols:
                cached_normalized.add(normalize_symbol(sym))

        # Find missing symbols
        missing_normalized = requested_set - cached_normalized

        # Map back to original symbol format
        missing_symbols = [
            requested_normalized[norm_sym]
            for norm_sym in missing_normalized
            if norm_sym in requested_normalized
        ]

        if missing_symbols:
            logger.info(
                f"[{data_type}][{venue}] Symbol gap detected: "
                f"{len(missing_symbols)} missing out of {len(symbols)} requested "
                f"(cached: {len(cached_normalized)})"
            )

        return missing_symbols

    def get_cached_symbols(
        self,
        data_type: str,
        venue: str,
        timeframe: Optional[str] = None
    ) -> Set[str]:
        """
        Get all cached symbols for a venue/data_type.

        Args:
            data_type: Type of data
            venue: Venue name
            timeframe: Timeframe for OHLCV data

        Returns:
            Set of normalized symbol names in cache
        """
        entries = self.metadata.get_entries(data_type, venue)

        if not entries:
            return set()

        cached_symbols = set()
        for entry in entries:
            if timeframe and entry.timeframe and entry.timeframe != timeframe:
                continue
            if entry.record_count == 0:
                continue
            for sym in entry.symbols:
                cached_symbols.add(normalize_symbol(sym))

        return cached_symbols

    def get_cached_date_range(
        self,
        data_type: str,
        venue: str,
        timeframe: Optional[str] = None
    ) -> Optional[Tuple[str, str]]:
        """
        Get the date range of cached data for a venue.

        Returns:
            Tuple of (start_date, end_date) or None if no cache
        """
        entries = self.metadata.get_entries(data_type, venue)

        if not entries:
            return None

        # Filter by timeframe if specified
        if timeframe:
            entries = [e for e in entries if e.timeframe == timeframe]

        if not entries:
            return None

        # Return the overall range
        min_start = min(e.start_date for e in entries)
        max_end = max(e.end_date for e in entries)

        return (min_start, max_end)

    def has_cached_data(
        self,
        data_type: str,
        venue: str,
        start_date: str,
        end_date: str,
        timeframe: Optional[str] = None
    ) -> bool:
        """Check if the requested range is fully cached."""
        gaps = self.get_collection_gaps(
            data_type, venue, [], start_date, end_date, timeframe
        )
        return len(gaps) == 0

    def load_cached_data(
        self,
        data_type: str,
        venue: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load cached data from disk.

        Args:
            data_type: Type of data
            venue: Venue name
            start_date: Optional filter start date
            end_date: Optional filter end date
            timeframe: Timeframe for OHLCV data

        Returns:
            DataFrame with cached data, or None if not found
        """
        file_path = self._get_file_path(data_type, venue, timeframe)

        if not file_path.exists():
            return None

        try:
            df = pd.read_parquet(file_path)

            # Apply date filters if specified
            if 'timestamp' in df.columns:
                # reliable timestamp parsing - try multiple formats
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
                except (ValueError, TypeError):
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
                    except (ValueError, TypeError):
                        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

                if start_date:
                    start_dt = pd.to_datetime(start_date, utc=True)
                    df = df[df['timestamp'] >= start_dt]

                if end_date:
                    end_dt = pd.to_datetime(end_date, utc=True) + timedelta(days=1)
                    df = df[df['timestamp'] < end_dt]

            logger.info(f"Loaded {len(df):,} cached records for {venue}/{data_type}")
            return df

        except Exception as e:
            logger.error(f"Failed to load cached data: {e}")
            return None

    def mark_no_data_available(
        self,
        data_type: str,
        venue: str,
        start_date: str,
        end_date: str,
        timeframe: Optional[str] = None
    ) -> bool:
        """
        Mark that a venue/data_type combination has no data for the given range.

        This creates a metadata entry indicating the range was checked and returned empty.
        Prevents re-collection on subsequent runs.

        Args:
            data_type: Type of data
            venue: Venue name
            start_date: Start date checked
            end_date: End date checked
            timeframe: Timeframe for OHLCV

        Returns:
            True if marked successfully
        """
        try:
            # Generate a placeholder file path (no actual file, just metadata)
            file_path = self._get_file_path(data_type, venue, timeframe, for_write=False)

            # Create timestamp and checksum for the "no data" marker
            created_at = datetime.now(timezone.utc).isoformat()
            checksum = hashlib.md5(f"no_data_{venue}_{data_type}_{start_date}_{end_date}".encode()).hexdigest()

            # Create a cache entry with 0 records
            # This tells the gap detection that this range was already checked
            entry = CacheEntry(
                data_type=data_type,
                venue=venue,
                symbols=[], # Empty list - no symbols (List, not set)
                start_date=start_date,
                end_date=end_date,
                record_count=0, # Explicitly 0 records
                file_path=str(file_path),
                created_at=created_at,
                checksum=checksum,
                quality_score=1.0, # Valid check
                is_normalized=True,
                is_validated=True,
                timeframe=timeframe
            )
            self.metadata.add_entry(entry)
            self._save_metadata() # Use the manager's save method, not metadata's

            logger.debug(
                f"Marked {venue}/{data_type} as no-data-available "
                f"for {start_date} to {end_date}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to mark no-data-available: {e}")
            return False

    def update_cache(
        self,
        data_type: str,
        venue: str,
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
        timeframe: Optional[str] = None,
        quality_score: float = 0.0,
        is_normalized: bool = True,
        is_validated: bool = True
    ) -> bool:
        """
        Update cache with new processed data.

        CRITICAL: Only call this after full pipeline processing (normalization, validation).

        Args:
            data_type: Type of data
            venue: Venue name
            data: Processed DataFrame to cache
            start_date: Start date of the data
            end_date: End date of the data
            symbols: List of symbols in the data
            timeframe: Timeframe for OHLCV data
            quality_score: Quality score from validation
            is_normalized: Whether data is normalized
            is_validated: Whether data passed validation

        Returns:
            True if successful
        """
        if data is None or data.empty:
            # For empty data, mark it as checked to prevent re-collection
            # This is critical for venues that don't support certain data types
            self.mark_no_data_available(data_type, venue, start_date, end_date, timeframe)
            return True # Success - we've marked it as checked

        file_path = self._get_file_path(data_type, venue, timeframe, for_write=True)

        try:
            # Ensure timestamps in new data are datetime objects
            if 'timestamp' in data.columns:
                data = data.copy()
                # reliable timestamp parsing - try multiple formats
                try:
                    data['timestamp'] = pd.to_datetime(data['timestamp'], format='ISO8601', utc=True)
                except (ValueError, TypeError):
                    try:
                        data['timestamp'] = pd.to_datetime(data['timestamp'], format='mixed', utc=True)
                    except (ValueError, TypeError):
                        data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True, errors='coerce')

            # Load existing data and merge
            existing_df = self.load_cached_data(data_type, venue, timeframe=timeframe)

            if existing_df is not None and not existing_df.empty:
                # Merge with existing data
                combined = pd.concat([existing_df, data], ignore_index=True)

                # Remove duplicates based on timestamp and symbol
                if 'timestamp' in combined.columns and 'symbol' in combined.columns:
                    combined = combined.drop_duplicates(
                        subset=['timestamp', 'symbol'],
                        keep='last'
                    )
                combined = combined.sort_values('timestamp').reset_index(drop=True)
            else:
                combined = data

            # Determine symbols from data
            if symbols is None:
                if 'symbol' in combined.columns:
                    symbols = combined['symbol'].unique().tolist()
                else:
                    symbols = []

            # Save to disk atomically
            temp_path = file_path.with_suffix('.tmp')
            combined.to_parquet(temp_path, index=False, compression='gzip')
            temp_path.replace(file_path)

            # Update metadata
            entry = CacheEntry(
                data_type=data_type,
                venue=venue,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                record_count=len(combined),
                file_path=str(file_path),
                created_at=datetime.now(timezone.utc).isoformat(),
                checksum=self._compute_checksum(combined),
                quality_score=quality_score,
                is_normalized=is_normalized,
                is_validated=is_validated,
                timeframe=timeframe
            )

            self.metadata.add_entry(entry)
            self._save_metadata()

            logger.info(
                f"Updated cache for {venue}/{data_type}: "
                f"{len(combined):,} total records ({start_date} to {end_date})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to update cache: {e}")
            return False

    def get_cache_summary(self) -> Dict[str, Any]:
        """Get a summary of the cache contents."""
        summary = {
            'cache_dir': str(self.cache_dir),
            'total_entries': 0,
            'total_records': 0,
            'data_types': {},
        }

        for data_type, venues in self.metadata.entries.items():
            summary['data_types'][data_type] = {
                'venues': {},
                'total_records': 0,
            }

            for venue, entries in venues.items():
                venue_records = sum(e.record_count for e in entries)
                date_range = None
                if entries:
                    min_start = min(e.start_date for e in entries)
                    max_end = max(e.end_date for e in entries)
                    date_range = f"{min_start} to {max_end}"

                summary['data_types'][data_type]['venues'][venue] = {
                    'entries': len(entries),
                    'records': venue_records,
                    'date_range': date_range,
                }
                summary['data_types'][data_type]['total_records'] += venue_records
                summary['total_entries'] += len(entries)
                summary['total_records'] += venue_records

        return summary

    def clear_cache(
        self,
        data_type: Optional[str] = None,
        venue: Optional[str] = None
    ) -> None:
        """
        Clear cache entries.

        Args:
            data_type: If specified, only clear this data type
            venue: If specified, only clear this venue
        """
        if data_type is None and venue is None:
            # Clear everything
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.metadata = CacheMetadata()
            self._save_metadata()
            logger.info("Cleared all cache")
            return

        if data_type and venue:
            # Clear specific venue for data type
            if data_type in self.metadata.entries:
                if venue in self.metadata.entries[data_type]:
                    del self.metadata.entries[data_type][venue]

            # Remove file
            for tf in [None, '1h', '4h', '1d']:
                file_path = self._get_file_path(data_type, venue, tf)
                if file_path.exists():
                    file_path.unlink()

        elif data_type:
            # Clear entire data type
            if data_type in self.metadata.entries:
                del self.metadata.entries[data_type]

            data_dir = self.cache_dir / data_type
            if data_dir.exists():
                shutil.rmtree(data_dir)

        self._save_metadata()
        logger.info(f"Cleared cache: data_type={data_type}, venue={venue}")

    def rebuild_metadata(self) -> Dict[str, int]:
        """
        Rebuild cache metadata by scanning all parquet files.

        This is useful when data files exist but metadata is missing or out of sync.

        Returns:
            Dict with counts of discovered entries by data type
        """
        logger.info("Rebuilding cache metadata from parquet files...")
        discovered = {}

        # Scan all parquet files in the cache directory
        for parquet_file in self.cache_dir.rglob('*.parquet'):
            try:
                # Load the file to extract metadata
                df = pd.read_parquet(parquet_file)
                if df.empty:
                    continue

                # Parse path to determine data_type and venue
                # Expected patterns:
                # - data/processed/funding_rates/binance_funding_rates.parquet
                # - data/processed/binance/funding_rates.parquet
                # - data/processed/ohlcv/binance_ohlcv_1h.parquet
                rel_path = parquet_file.relative_to(self.cache_dir)
                parts = list(rel_path.parts)

                data_type = None
                venue = None
                timeframe = None

                # Try to extract data_type and venue from path
                if len(parts) >= 2:
                    # Pattern: data_type/venue_data_type.parquet
                    if parts[0] in ['funding_rates', 'ohlcv', 'open_interest', 'liquidations',
                                    'trades', 'sentiment', 'social', 'on_chain_metrics', 'dvol']:
                        data_type = parts[0]
                        # Extract venue from filename
                        filename = parts[-1].replace('.parquet', '')
                        for known_venue in ['binance', 'bybit', 'okx', 'hyperliquid', 'dydx',
                                           'coinbase', 'kraken', 'deribit', 'aevo', 'gmx',
                                           'coinalyze', 'santiment', 'lunarcrush', 'coingecko']:
                            if known_venue in filename.lower():
                                venue = known_venue
                                break
                        # Try to extract timeframe from filename
                        for tf in ['1h', '4h', '1d', '8h']:
                            if tf in filename:
                                timeframe = tf
                    else:
                        # Pattern: venue/data_type.parquet
                        venue = parts[0]
                        filename = parts[-1].replace('.parquet', '')
                        # Try common data type patterns
                        for dt in ['funding_rates', 'ohlcv', 'open_interest', 'liquidations',
                                   'trades', 'sentiment', 'social', 'on_chain_metrics']:
                            if dt in filename.lower():
                                data_type = dt
                                break

                if not data_type or not venue:
                    logger.debug(f"Could not parse path: {parquet_file}")
                    continue

                # Extract metadata from DataFrame
                symbols = []
                start_date = None
                end_date = None

                if 'symbol' in df.columns:
                    symbols = sorted(df['symbol'].dropna().unique().tolist())
                    # Normalize symbols for storage
                    symbols = sorted(set(normalize_symbol(s) for s in symbols))
                elif 'slug' in df.columns:
                    symbols = sorted(df['slug'].dropna().unique().tolist())

                if 'timestamp' in df.columns:
                    # reliable timestamp parsing - try multiple formats
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
                    except (ValueError, TypeError):
                        try:
                            # Fallback: mixed format parsing
                            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
                        except (ValueError, TypeError):
                            # Last resort: infer format, coerce errors
                            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

                    # Filter out any NaT values from failed parsing
                    valid_timestamps = df['timestamp'].dropna()
                    if len(valid_timestamps) > 0:
                        start_date = valid_timestamps.min().strftime('%Y-%m-%d')
                        end_date = valid_timestamps.max().strftime('%Y-%m-%d')
                    else:
                        start_date = '2020-01-01'
                        end_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
                else:
                    # Default to wide range if no timestamp
                    start_date = '2020-01-01'
                    end_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')

                # Check if entry already exists
                existing_entries = self.metadata.get_entries(data_type, venue)
                already_exists = False
                for entry in existing_entries:
                    if entry.file_path == str(parquet_file):
                        already_exists = True
                        break

                if not already_exists:
                    # Create cache entry
                    entry = CacheEntry(
                        data_type=data_type,
                        venue=venue,
                        symbols=symbols[:100], # Limit symbol list
                        start_date=start_date,
                        end_date=end_date,
                        record_count=len(df),
                        file_path=str(parquet_file),
                        created_at=datetime.now(timezone.utc).isoformat(),
                        checksum=self._compute_checksum(df),
                        quality_score=0.0,
                        is_normalized=True,
                        is_validated=True,
                        timeframe=timeframe,
                    )
                    self.metadata.add_entry(entry)

                    if data_type not in discovered:
                        discovered[data_type] = 0
                    discovered[data_type] += 1

                    logger.info(f"Discovered cache: {data_type}/{venue} - {len(df):,} records ({start_date} to {end_date})")

            except Exception as e:
                logger.warning(f"Could not process {parquet_file}: {e}")
                continue

        # Save updated metadata
        self._save_metadata()

        total = sum(discovered.values())
        logger.info(f"Rebuilt metadata: discovered {total} new entries across {len(discovered)} data types")
        return discovered

# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_cache_manager: Optional[IncrementalCacheManager] = None

def get_cache_manager(cache_dir: str = 'data/processed') -> IncrementalCacheManager:
    """Get or create the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = IncrementalCacheManager(cache_dir)
    return _cache_manager

def reset_cache_manager() -> None:
    """Reset the global cache manager instance."""
    global _cache_manager
    _cache_manager = None

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'CacheEntry',
    'CacheMetadata',
    'IncrementalCacheManager',
    'get_cache_manager',
    'reset_cache_manager',
]
