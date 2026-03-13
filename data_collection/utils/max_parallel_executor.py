"""
Maximum Parallel Executor for Crypto Data Collection.

This module provides MAXIMUM parallelization that uses ALL available system resources:
1. All CPU cores via multiprocessing where beneficial
2. Maximum async concurrency via asyncio
3. Parallel data types, venues, AND symbols simultaneously
4. Automatic resource detection and optimization

Architecture:
    
                           MaxParallelExecutor 
        
                          Level 1: Data Type Parallelism 
                              
         funding_rates ohlcv open_interest ... 
         (parallel) (parallel) (parallel) 
                              
        
                                                                              
        
                                                                             
                          Level 2: Venue Parallelism 
                                  
         binance bybit hyperl dydx ... (all venues) 
        (parallel (parallel (parallel (parallel 
                                  
        
                                                                             
        
                                                                            
                          Level 3: Symbol Parallelism 
              (per-venue) 
        BTCETHSOL BTCETHSOL BTCETHSOL rate-limited 
                                
        
                                                                                  
      Resource Auto-Detection: 
        - CPU Cores: os.cpu_count() → max_data_type_concurrency 
        - Memory: Available RAM → batch sizes 
        - Network: Adaptive based on latency 
    

Key Features:
    - Processes ALL data types in parallel (not sequentially)
    - Processes ALL venues in parallel within each data type
    - Processes ALL symbols in parallel within each venue (rate-limited)
    - Uses ALL CPU cores for processing
    - Integrates with IncrementalCacheManager for smart collection
    - Automatic resource optimization

Version: 1.0.0
"""

import asyncio
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from .incremental_cache import IncrementalCacheManager, get_cache_manager
from .parallel_processor import (
    ParallelSymbolProcessor,
    ParallelCollectionManager,
    VENUE_PARALLEL_CONFIGS,
    get_venue_config,
    filter_symbols_for_venue, # Pre-filter symbols to avoid wasted iterations
    BatchResult,
)
from .collector_pool import get_collector_pool

logger = logging.getLogger(__name__)

# =============================================================================
# SYSTEM RESOURCE DETECTION
# =============================================================================

@dataclass
class SystemResources:
    """Detected system resources for optimization."""
    cpu_cores: int
    cpu_cores_physical: int
    memory_gb: float
    is_laptop: bool # Laptops may need thermal throttling consideration

    # Calculated optimal settings
    max_data_type_concurrency: int = 0
    max_venue_concurrency: int = 0
    max_total_connections: int = 0

    def __post_init__(self):
        # Calculate optimal concurrency based on resources
        # MODERATE - Balance speed vs rate limit compliance

        # 2 data types at a time (safe for DNS)
        self.max_data_type_concurrency = 2

        # 3 venues at a time (safe for DNS)
        self.max_venue_concurrency = 3

        # Total connections: Conservative - cap at 150
        self.max_total_connections = min(int(self.memory_gb * 20), 150)

def detect_system_resources() -> SystemResources:
    """Detect available system resources for optimization."""
    cpu_cores = os.cpu_count() or 4
    cpu_physical = cpu_cores // 2 if cpu_cores > 1 else 1 # Estimate physical cores

    # Detect memory (platform-specific)
    memory_gb = 8.0 # Default assumption
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        # Fallback for systems without psutil
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        memory_kb = int(line.split()[1])
                        memory_gb = memory_kb / (1024 ** 2)
                        break
        except:
            pass # Use default

    # Detect if laptop (simple heuristic)
    is_laptop = memory_gb < 32 and cpu_cores <= 16

    resources = SystemResources(
        cpu_cores=cpu_cores,
        cpu_cores_physical=cpu_physical,
        memory_gb=memory_gb,
        is_laptop=is_laptop
    )

    logger.info(
        f"System Resources Detected: "
        f"CPU={cpu_cores} cores ({cpu_physical} physical), "
        f"RAM={memory_gb:.1f}GB, "
        f"Laptop={is_laptop}"
    )
    logger.info(
        f"Optimal Settings: "
        f"data_type_concurrency={resources.max_data_type_concurrency}, "
        f"venue_concurrency={resources.max_venue_concurrency}, "
        f"max_connections={resources.max_total_connections}"
    )

    return resources

# Global resources (detected once)
_system_resources: Optional[SystemResources] = None

def get_system_resources() -> SystemResources:
    """Get or detect system resources."""
    global _system_resources
    if _system_resources is None:
        _system_resources = detect_system_resources()
    return _system_resources

# =============================================================================
# EXECUTION STATISTICS
# =============================================================================

@dataclass
class ExecutionStats:
    """Statistics for parallel execution."""
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None

    # Counts
    data_types_processed: int = 0
    venues_processed: int = 0
    symbols_processed: int = 0
    total_records: int = 0

    # Cache stats
    records_from_cache: int = 0
    records_collected: int = 0
    collection_gaps_found: int = 0

    # Performance
    peak_concurrency: int = 0
    total_api_calls: int = 0
    rate_limit_hits: int = 0
    errors: int = 0

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or datetime.now(timezone.utc)
        return (end - self.start_time).total_seconds()

    @property
    def records_per_second(self) -> float:
        if self.duration_seconds == 0:
            return 0
        return self.total_records / self.duration_seconds

    @property
    def cache_hit_rate(self) -> float:
        total = self.records_from_cache + self.records_collected
        if total == 0:
            return 0
        return self.records_from_cache / total * 100

    def summary(self) -> Dict[str, Any]:
        return {
            'duration_seconds': self.duration_seconds,
            'data_types_processed': self.data_types_processed,
            'venues_processed': self.venues_processed,
            'symbols_processed': self.symbols_processed,
            'total_records': self.total_records,
            'records_per_second': self.records_per_second,
            'records_from_cache': self.records_from_cache,
            'records_collected': self.records_collected,
            'cache_hit_rate': f"{self.cache_hit_rate:.1f}%",
            'peak_concurrency': self.peak_concurrency,
            'errors': self.errors,
        }

# =============================================================================
# MAX PARALLEL EXECUTOR
# =============================================================================

class MaxParallelExecutor:
    """
    Maximum parallelization executor for data collection.

    This executor runs EVERYTHING in parallel:
    - All data types simultaneously
    - All venues simultaneously (within each data type)
    - All symbols simultaneously (within each venue, rate-limited)

    Integrates with IncrementalCacheManager to only collect missing data.

    Usage:
        >>> executor = MaxParallelExecutor()
        >>>
        >>> results = await executor.execute_collection(
        ... data_types=['funding_rates', 'ohlcv', 'open_interest'],
        ... venues=['binance', 'bybit', 'okx', 'hyperliquid', 'dydx'],
        ... symbols=['BTC', 'ETH', 'SOL', ...],
        ... start_date='2020-01-01',
        ... end_date='2026-02-03'
        ... )
    """

    def __init__(
        self,
        cache_dir: str = 'data/processed',
        use_cache: bool = True,
        max_data_type_concurrency: Optional[int] = None,
        max_venue_concurrency: Optional[int] = None,
    ):
        self.resources = get_system_resources()
        self.cache = get_cache_manager(cache_dir) if use_cache else None
        self.use_cache = use_cache

        # OPTIMIZATION: Only rebuild metadata if cache is empty
        # The metadata file tracks all collected data - trust it when it exists
        # This prevents slow startup from scanning all parquet files every run
        if self.cache:
            if not self.cache.metadata.entries:
                logger.info("Cache metadata empty - rebuilding from parquet files...")
                discovered = self.cache.rebuild_metadata()
                if discovered:
                    total = sum(discovered.values())
                    logger.info(f"Cache rebuild: discovered {total} entries from existing files")
            else:
                # Count total entries for logging
                total_entries = sum(
                    len(entries)
                    for venues in self.cache.metadata.entries.values()
                    for entries in venues.values()
                )
                logger.info(f"Cache loaded with {total_entries} entries - skipping rebuild")

        # Use detected resources or override
        self.max_data_type_concurrency = (
            max_data_type_concurrency or self.resources.max_data_type_concurrency
        )
        self.max_venue_concurrency = (
            max_venue_concurrency or self.resources.max_venue_concurrency
        )

        # Semaphores for concurrency control
        self._data_type_semaphore = asyncio.Semaphore(self.max_data_type_concurrency)
        self._venue_semaphore = asyncio.Semaphore(self.max_venue_concurrency)
        self._connection_semaphore = asyncio.Semaphore(
            self.resources.max_total_connections
        )

        # Stats
        self.stats = ExecutionStats()
        self._current_concurrency = 0
        self._lock = asyncio.Lock()

        # Thread pool for CPU-bound post-processing
        self._thread_pool = ThreadPoolExecutor(
            max_workers=self.resources.cpu_cores_physical
        )

        logger.info(
            f"MaxParallelExecutor initialized: "
            f"data_type_concurrency={self.max_data_type_concurrency}, "
            f"venue_concurrency={self.max_venue_concurrency}, "
            f"use_cache={use_cache}"
        )

    async def execute_collection(
        self,
        data_types: List[str],
        venues: List[str],
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        collector_factory: Callable[[str], Any],
        timeframe: str = '1h',
        **kwargs
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Execute maximum parallel collection.

        Args:
            data_types: List of data types to collect
            venues: List of venues to collect from
            symbols: List of symbols to collect
            start_date: Start date (YYYY-MM-DD or datetime)
            end_date: End date (YYYY-MM-DD or datetime)
            collector_factory: Factory function that creates collector for venue
            timeframe: Timeframe for OHLCV data
            **kwargs: Additional arguments passed to collectors

        Returns:
            Dict[data_type][venue] = DataFrame
        """
        self.stats = ExecutionStats()

        # Normalize dates
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')

        logger.info(
            f"Starting MAXIMUM PARALLEL collection: "
            f"{len(data_types)} data types × {len(venues)} venues × {len(symbols)} symbols"
        )
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(
            f"Concurrency: {self.max_data_type_concurrency} data types, "
            f"{self.max_venue_concurrency} venues"
        )

        # =====================================================================
        # LEVEL 1: PARALLEL DATA TYPES
        # =====================================================================
        # All data types run simultaneously

        async def collect_data_type(data_type: str) -> Tuple[str, Dict[str, pd.DataFrame]]:
            """Collect all venues for a single data type."""
            async with self._data_type_semaphore:
                logger.info(f"[{data_type}] Starting parallel venue collection")
                start = time.monotonic()

                venue_results = await self._collect_all_venues_parallel(
                    data_type=data_type,
                    venues=venues,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    collector_factory=collector_factory,
                    timeframe=timeframe,
                    **kwargs
                )

                duration = time.monotonic() - start
                total_records = sum(
                    len(df) for df in venue_results.values() if df is not None
                )

                logger.info(
                    f"[{data_type}] Completed: {total_records:,} records "
                    f"from {len(venue_results)} venues in {duration:.1f}s"
                )

                self.stats.data_types_processed += 1
                return data_type, venue_results

        # Execute ALL data types in parallel
        tasks = [collect_data_type(dt) for dt in data_types]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Memory management: Run garbage collection after parallel tasks complete
        import gc
        gc.collect()

        # Compile results
        results: Dict[str, Dict[str, pd.DataFrame]] = {}
        for item in results_list:
            if isinstance(item, Exception):
                logger.error(f"Data type collection error: {item}")
                self.stats.errors += 1
                continue
            data_type, venue_results = item
            results[data_type] = venue_results

        self.stats.end_time = datetime.now(timezone.utc)
        self._log_summary()

        # Final memory cleanup
        gc.collect()

        return results

    async def _collect_all_venues_parallel(
        self,
        data_type: str,
        venues: List[str],
        symbols: List[str],
        start_date: str,
        end_date: str,
        collector_factory: Callable[[str], Any],
        timeframe: str = '1h',
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect from all venues in parallel for a single data type.
        """
        # =====================================================================
        # LEVEL 2: PARALLEL VENUES
        # =====================================================================

        async def collect_venue(venue: str) -> Tuple[str, Optional[pd.DataFrame]]:
            """Collect from a single venue with caching."""
            async with self._venue_semaphore:
                try:
                    return await self._collect_single_venue(
                        data_type=data_type,
                        venue=venue,
                        symbols=symbols,
                        start_date=start_date,
                        end_date=end_date,
                        collector_factory=collector_factory,
                        timeframe=timeframe,
                        **kwargs
                    )
                except Exception as e:
                    logger.error(f"[{data_type}][{venue}] Collection error: {e}")
                    self.stats.errors += 1
                    return venue, None

        # Execute ALL venues in parallel
        tasks = [collect_venue(v) for v in venues]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Memory management after parallel venue collection
        import gc
        gc.collect()

        # Compile results
        venue_results: Dict[str, pd.DataFrame] = {}
        for item in results_list:
            if isinstance(item, Exception):
                self.stats.errors += 1
                continue
            venue, df = item
            if df is not None and not df.empty:
                venue_results[venue] = df

        return venue_results

    async def _collect_single_venue(
        self,
        data_type: str,
        venue: str,
        symbols: List[str],
        start_date: str,
        end_date: str,
        collector_factory: Callable[[str], Any],
        timeframe: str = '1h',
        **kwargs
    ) -> Tuple[str, Optional[pd.DataFrame]]:
        """
        Collect from a single venue with incremental caching.
        """
        timeframe_param = timeframe if data_type == 'ohlcv' else None

        # =====================================================================
        # CACHE CHECK: Load existing data and determine gaps
        # =====================================================================
        cached_df = None
        gaps_to_collect = [(start_date, end_date)]

        if self.use_cache and self.cache:
            # Load any existing cached data
            cached_df = self.cache.load_cached_data(
                data_type=data_type,
                venue=venue,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe_param
            )

            if cached_df is not None and not cached_df.empty:
                self.stats.records_from_cache += len(cached_df)
                logger.debug(
                    f"[{data_type}][{venue}] Loaded {len(cached_df):,} cached records"
                )

            # Determine what gaps need to be collected
            gaps_to_collect = self.cache.get_collection_gaps(
                data_type=data_type,
                venue=venue,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe_param
            )

            if not gaps_to_collect:
                # CRITICAL FIX: Check for missing symbols even when date range is cached
                # This handles the case where new symbols were added to the universe
                missing_symbols = self.cache.get_missing_symbols(
                    data_type=data_type,
                    venue=venue,
                    symbols=symbols,
                    timeframe=timeframe_param
                )

                if not missing_symbols:
                    logger.info(
                        f"[{data_type}][{venue}] Full cache hit - no collection needed (all {len(symbols)} symbols)"
                    )
                    self.stats.venues_processed += 1
                    return venue, cached_df
                else:
                    # Date range is covered, but some symbols are missing
                    logger.info(
                        f"[{data_type}][{venue}] Date range cached, but {len(missing_symbols)} symbols missing. "
                        f"Collecting: {missing_symbols[:10]}{'...' if len(missing_symbols) > 10 else ''}"
                    )
                    # Override symbols to only collect missing ones
                    symbols = missing_symbols
                    gaps_to_collect = [(start_date, end_date)]

            self.stats.collection_gaps_found += len(gaps_to_collect)
            logger.info(
                f"[{data_type}][{venue}] Found {len(gaps_to_collect)} gap(s) to collect"
            )

        # =====================================================================
        # LEVEL 3: PARALLEL SYMBOL COLLECTION (within rate limits)
        # =====================================================================
        all_new_data = []
        timeframe_param = timeframe if data_type == 'ohlcv' else None

        # Create collector ONCE for all gaps (not per-gap) to avoid overhead
        # The collector_factory should use CollectorPool for caching
        try:
            collector = collector_factory(venue)
        except Exception as e:
            logger.error(f"[{data_type}][{venue}] Failed to create collector: {e}")
            self.stats.errors += 1
            return venue, cached_df

        # CRITICAL: Check if collector is None or doesn't support this data type
        # Mark as no-data to prevent re-checking on future runs
        if collector is None:
            logger.debug(f"[{data_type}][{venue}] Collector factory returned None")
            if self.use_cache and self.cache:
                self.cache.mark_no_data_available(data_type, venue, start_date, end_date, timeframe_param)
            self.stats.venues_processed += 1
            return venue, cached_df

        if hasattr(collector, 'supported_data_types'):
            if data_type not in collector.supported_data_types:
                logger.debug(f"[{data_type}][{venue}] Data type not supported by this collector")
                if self.use_cache and self.cache:
                    self.cache.mark_no_data_available(data_type, venue, start_date, end_date, timeframe_param)
                self.stats.venues_processed += 1
                return venue, cached_df

        # Get venue config for optimal parallelism
        venue_config = get_venue_config(venue)

        # OPTIMIZATION: Pre-filter symbols to avoid wasted iterations
        # This prevents ~52 wasted iterations for venues like Deribit (BTC/ETH/SOL only)
        filtered_symbols = venue_config.filter_symbols(symbols)
        if len(filtered_symbols) == 0:
            logger.info(f"[{data_type}][{venue}] No supported symbols - skipping")
            if self.use_cache and self.cache:
                self.cache.mark_no_data_available(data_type, venue, start_date, end_date, timeframe_param)
            self.stats.venues_processed += 1
            return venue, cached_df
        if len(filtered_symbols) < len(symbols):
            logger.info(f"[{data_type}][{venue}] Using {len(filtered_symbols)}/{len(symbols)} supported symbols")

        for gap_start, gap_end in gaps_to_collect:
            logger.info(
                f"[{data_type}][{venue}] Collecting gap: {gap_start} to {gap_end}"
            )

            try:

                # Create parallel symbol processor
                processor = ParallelSymbolProcessor(
                    venue=venue,
                    max_symbol_concurrency=venue_config.max_symbol_concurrency,
                    rate_limit_per_minute=venue_config.rate_limit_per_minute
                )

                # Define fetch function based on data type
                if data_type == 'funding_rates':
                    async def fetch_symbol(symbol: str, **kw) -> pd.DataFrame:
                        return await collector.fetch_funding_rates(
                            symbols=[symbol],
                            start_date=gap_start,
                            end_date=gap_end
                        )
                elif data_type == 'ohlcv':
                    async def fetch_symbol(symbol: str, **kw) -> pd.DataFrame:
                        return await collector.fetch_ohlcv(
                            symbols=[symbol],
                            timeframe=timeframe,
                            start_date=gap_start,
                            end_date=gap_end
                        )
                elif data_type == 'open_interest':
                    async def fetch_symbol(symbol: str, **kw) -> pd.DataFrame:
                        if hasattr(collector, 'fetch_open_interest'):
                            return await collector.fetch_open_interest(
                                symbols=[symbol],
                                start_date=gap_start,
                                end_date=gap_end
                            )
                        return pd.DataFrame()
                elif data_type in ('social', 'social_metrics'):
                    # Handle social/sentiment data collectors (Santiment, LunarCrush)
                    async def fetch_symbol(symbol: str, **kw) -> pd.DataFrame:
                        # Try multiple method names used by different collectors
                        for method_name in ['fetch_social_metrics', 'fetch_social']:
                            method = getattr(collector, method_name, None)
                            if method:
                                return await method(
                                    symbols=[symbol],
                                    start_date=gap_start,
                                    end_date=gap_end
                                )
                        return pd.DataFrame()
                elif data_type in ('sentiment', 'sentiment_signals'):
                    # Handle sentiment data collectors (LunarCrush)
                    async def fetch_symbol(symbol: str, **kw) -> pd.DataFrame:
                        # Try multiple method names used by different collectors
                        for method_name in ['fetch_sentiment_signals', 'fetch_sentiment']:
                            method = getattr(collector, method_name, None)
                            if method:
                                return await method(
                                    symbols=[symbol],
                                    start_date=gap_start,
                                    end_date=gap_end
                                )
                        return pd.DataFrame()
                elif data_type in ('on_chain_metrics', 'onchain_metrics'):
                    # Handle on-chain metrics (Santiment, Glassnode)
                    async def fetch_symbol(symbol: str, **kw) -> pd.DataFrame:
                        for method_name in ['fetch_onchain_metrics', 'fetch_on_chain_metrics', 'fetch_comprehensive_metrics']:
                            method = getattr(collector, method_name, None)
                            if method:
                                return await method(
                                    symbols=[symbol],
                                    start_date=gap_start,
                                    end_date=gap_end
                                )
                        return pd.DataFrame()
                else:
                    # Generic fetch with fallback method names
                    fetch_method = getattr(collector, f'fetch_{data_type}', None)

                    # Try underscore variant (on_chain -> onchain)
                    if fetch_method is None:
                        alt_name = f"fetch_{data_type.replace('_', '')}"
                        fetch_method = getattr(collector, alt_name, None)

                    if fetch_method:
                        async def fetch_symbol(symbol: str, **kw) -> pd.DataFrame:
                            return await fetch_method(
                                symbols=[symbol],
                                start_date=gap_start,
                                end_date=gap_end
                            )
                    else:
                        logger.debug(
                            f"[{data_type}][{venue}] No fetch method found - skipping"
                        )
                        continue

                # Execute parallel symbol collection (using filtered symbols)
                batch_result = await processor.process_symbols(
                    symbols=filtered_symbols, # Use filtered symbols to avoid wasted iterations
                    fetch_func=fetch_symbol,
                    data_type=data_type
                )

                # Combine results
                combined = batch_result.get_combined_dataframe()
                if combined is not None and not combined.empty:
                    all_new_data.append(combined)
                    self.stats.records_collected += len(combined)
                    self.stats.symbols_processed += batch_result.symbols_processed

                self.stats.total_api_calls += processor.stats.requests_made
                self.stats.peak_concurrency = max(
                    self.stats.peak_concurrency,
                    processor.stats.peak_concurrency
                )

                # NOTE: Do NOT close collector here - it's reused across gaps
                # The collector pool manages lifecycle at end of run

            except Exception as e:
                logger.error(
                    f"[{data_type}][{venue}] Gap collection error "
                    f"({gap_start} to {gap_end}): {e}"
                )
                self.stats.errors += 1

        # =====================================================================
        # MERGE NEW DATA WITH CACHED DATA
        # =====================================================================
        final_df = None
        timeframe_param = timeframe if data_type == 'ohlcv' else None

        if all_new_data:
            new_df = pd.concat(all_new_data, ignore_index=True)

            if cached_df is not None and not cached_df.empty:
                final_df = pd.concat([cached_df, new_df], ignore_index=True)
            else:
                final_df = new_df

            # Deduplicate and sort
            if 'timestamp' in final_df.columns:
                if 'symbol' in final_df.columns:
                    final_df = final_df.drop_duplicates(
                        subset=['timestamp', 'symbol'],
                        keep='last'
                    )
                final_df = final_df.sort_values('timestamp').reset_index(drop=True)

            # Update cache with new data
            if self.use_cache and self.cache:
                self.cache.update_cache(
                    data_type=data_type,
                    venue=venue,
                    data=final_df,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe_param
                )

        elif cached_df is not None:
            final_df = cached_df
        else:
            # CRITICAL FIX: Cache empty results to prevent re-collection
            # This prevents infinite loops when a venue doesn't support a data type
            # or when authentication is required but not provided
            if self.use_cache and self.cache:
                logger.debug(f"[{data_type}][{venue}] Caching empty result to prevent re-collection")
                self.cache.mark_no_data_available(
                    data_type=data_type,
                    venue=venue,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe_param
                )

        # Update stats
        if final_df is not None:
            self.stats.total_records += len(final_df)
        self.stats.venues_processed += 1

        return venue, final_df

    def _log_summary(self) -> None:
        """Log execution summary."""
        summary = self.stats.summary()

        logger.info("=" * 70)
        logger.info("MAXIMUM PARALLEL EXECUTION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Duration: {summary['duration_seconds']:.1f}s")
        logger.info(f"Data Types: {summary['data_types_processed']}")
        logger.info(f"Venues: {summary['venues_processed']}")
        logger.info(f"Total Records: {summary['total_records']:,}")
        logger.info(f"Throughput: {summary['records_per_second']:.0f} records/sec")
        logger.info(f"Cache Hit Rate: {summary['cache_hit_rate']}")
        logger.info(f" - From Cache: {summary['records_from_cache']:,}")
        logger.info(f" - Collected: {summary['records_collected']:,}")
        logger.info(f"Peak Concurrency: {summary['peak_concurrency']}")
        if summary['errors'] > 0:
            logger.warning(f"Errors: {summary['errors']}")
        logger.info("=" * 70)

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self._thread_pool.shutdown(wait=False)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def execute_max_parallel_collection(
    data_types: List[str],
    venues: List[str],
    symbols: List[str],
    start_date: str,
    end_date: str,
    collector_factory: Callable[[str], Any],
    cache_dir: str = 'data/processed',
    use_cache: bool = True,
    timeframe: str = '1h'
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Convenience function for maximum parallel collection.

    This is the simplest way to use maximum parallelization with caching.

    Args:
        data_types: Data types to collect
        venues: Venues to collect from
        symbols: Symbols to collect
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        collector_factory: Function that creates collector for venue name
        cache_dir: Directory for processed data cache
        use_cache: Whether to use incremental caching
        timeframe: Timeframe for OHLCV

    Returns:
        Dict[data_type][venue] = DataFrame
    """
    executor = MaxParallelExecutor(
        cache_dir=cache_dir,
        use_cache=use_cache
    )

    try:
        results = await executor.execute_collection(
            data_types=data_types,
            venues=venues,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            collector_factory=collector_factory,
            timeframe=timeframe
        )
        return results
    finally:
        await executor.cleanup()

def get_optimal_concurrency_settings() -> Dict[str, int]:
    """
    Get optimal concurrency settings based on system resources.

    Returns dict with recommended settings for maximum performance.
    """
    resources = get_system_resources()
    return {
        'max_data_type_concurrency': resources.max_data_type_concurrency,
        'max_venue_concurrency': resources.max_venue_concurrency,
        'max_total_connections': resources.max_total_connections,
        'cpu_cores': resources.cpu_cores,
        'memory_gb': resources.memory_gb,
    }

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SystemResources',
    'detect_system_resources',
    'get_system_resources',
    'ExecutionStats',
    'MaxParallelExecutor',
    'execute_max_parallel_collection',
    'get_optimal_concurrency_settings',
]
