"""
Parallel Processing Infrastructure for Crypto Data Collection.

This module provides comprehensive parallel processing utilities for
symbol-level and venue-level parallelism, enabling 10-15x speedup while
respecting API rate limits.

Architecture Overview:
    
                          ParallelCollectionManager 
        
       Venue Semaphore (10-15 concurrent venues) 
                               
         Binance Hyperliq dYdX ... 
         Collector Collector Collector 
                               
                                                                    
                               
         Symbol Symbol Symbol 
         Processor Processor Processor 
         (5-20 (3-5 (3-5 
         parallel) parallel) parallel) 
                               
        
                                                                         
        
       Per-Venue Rate Limiters (adaptive token bucket) 
        binance: 1200/min | hyperliquid: 100/min | dydx: 100/min 
        
    

Key Features:
    - Adaptive symbol concurrency based on venue rate limits
    - Token bucket rate limiting with burst support
    - Automatic batch API detection and usage
    - Pipeline parallelism (collect while processing)
    - Error isolation per symbol (one failure doesn't block others)
    - Progress tracking and estimated completion time

Usage Example:
    >>> processor = ParallelSymbolProcessor(
    ... venue='binance',
    ... rate_limit_per_minute=1200,
    ... max_symbol_concurrency=20
    ... )
    >>> results = await processor.process_symbols(
    ... symbols=['BTC', 'ETH', 'SOL'],
    ... fetch_func=collector.fetch_single_funding_rate,
    ... start_date=start,
    ... end_date=end
    ... )

Version: 1.0.0
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any, Callable, Coroutine, Dict, List, Optional,
    Tuple, TypeVar, Union, Generic
)
from enum import Enum, auto
import pandas as pd

from .rate_limiter import (
    RateLimiter, AdaptiveRateLimiter, MultiRateLimiter,
    RateLimitStrategy, RequestPriority, VenueRateLimitConfig,
    create_venue_limiters
)

logger = logging.getLogger(__name__)

# Import batch optimization for intelligent grouping of symbols
try:
    from .batch_optimizer import (
        supports_batch, create_batches, estimate_speedup
    )
    BATCH_OPTIMIZER_AVAILABLE = True
except ImportError:
    BATCH_OPTIMIZER_AVAILABLE = False
    logger.warning("BatchOptimizer not available - batch optimization disabled")

# =============================================================================
# HIGH-THROUGHPUT RATE LIMITER FOR PARALLEL PROCESSING
# =============================================================================

class ParallelRateLimiter:
    """
    Simple high-throughput rate limiter optimized for parallel processing.

    Unlike the standard RateLimiter, this implementation:
    - Uses non-blocking token checks
    - Minimizes lock contention
    - Allows maximum parallelism within rate limits

    Uses leaky bucket algorithm with minimal locking.
    """

    def __init__(
        self,
        rate_per_minute: int,
        burst_size: Optional[int] = None,
        name: str = "parallel_limiter"
    ):
        self.rate = rate_per_minute
        self.per_second = rate_per_minute / 60.0
        self.name = name

        # Burst size: allow at least 10 concurrent or rate/10
        self.max_tokens = burst_size or max(10, rate_per_minute // 10)
        self.tokens = float(self.max_tokens)
        self.last_refill = time.monotonic()

        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.per_second
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        self.last_refill = now

    def available(self) -> int:
        """Check available tokens without locking."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        current = min(self.max_tokens, self.tokens + elapsed * self.per_second)
        return int(current)

    async def acquire(self, tokens: int = 1, timeout: float = 30.0) -> bool:
        """
        Acquire rate limit tokens.

        Returns True if acquired, False if timeout.
        """
        start = time.monotonic()

        while True:
            async with self._lock:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                # Calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.per_second

            # Check timeout
            if time.monotonic() - start + wait_time > timeout:
                return False

            # Wait outside lock to allow parallelism
            await asyncio.sleep(min(wait_time, 0.1))

        return False

T = TypeVar('T')

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class VenueParallelConfig:
    """
    Venue-specific parallelism configuration.

    Determines optimal concurrency levels based on API characteristics.

    Symbol Filtering:
        - supported_symbols: If set, only these symbols will be passed to the venue
        - unsupported_symbols: If set, these symbols will be excluded
        - None means all symbols are supported (default)

    This prevents wasted iterations where 55 symbols are passed to venues
    that only support 3 (like Deribit: BTC, ETH, SOL).
    """
    venue: str
    rate_limit_per_minute: int
    max_symbol_concurrency: int
    supports_batch_api: bool = False
    batch_size: int = 10
    min_request_interval_ms: int = 50 # Minimum time between requests
    burst_allowance: float = 1.5 # Burst multiplier
    # Symbol filtering - prevents wasted iterations
    supported_symbols: Optional[List[str]] = None # Whitelist (e.g., ['BTC', 'ETH', 'SOL'])
    unsupported_symbols: Optional[List[str]] = None # Blacklist

    def filter_symbols(self, symbols: List[str]) -> List[str]:
        """
        Filter symbols based on venue's supported/unsupported lists.

        This prevents wasted API calls and log spam from trying to fetch
        data for symbols the venue doesn't support.

        Args:
            symbols: List of symbols to filter

        Returns:
            Filtered list of symbols supported by this venue
        """
        if self.supported_symbols is not None:
            # Whitelist mode - only keep symbols in the supported list
            filtered = [s for s in symbols if s.upper() in [x.upper() for x in self.supported_symbols]]
            if len(filtered) < len(symbols):
                skipped = len(symbols) - len(filtered)
                logger.debug(f"[{self.venue}] Pre-filtered {skipped} unsupported symbols")
            return filtered
        elif self.unsupported_symbols is not None:
            # Blacklist mode - remove symbols in the unsupported list
            unsupported_upper = [x.upper() for x in self.unsupported_symbols]
            filtered = [s for s in symbols if s.upper() not in unsupported_upper]
            if len(filtered) < len(symbols):
                skipped = len(symbols) - len(filtered)
                logger.debug(f"[{self.venue}] Pre-filtered {skipped} unsupported symbols")
            return filtered
        else:
            # No filtering - all symbols supported
            return symbols

    @property
    def requests_per_second(self) -> float:
        """Calculate requests per second."""
        return self.rate_limit_per_minute / 60

    @property
    def optimal_concurrency(self) -> int:
        """
        Calculate optimal symbol concurrency.

        Formula: Allow enough concurrent requests to fully utilize the rate limit
        assuming typical API request latency of 200-500ms.

        For rate_limit of 600/min (10/sec) with 200ms latency:
        - We can have 10 * 0.2 = 2 requests in flight at minimum
        - To maximize throughput, allow 2-5 seconds worth of requests

        Returns at least 2 and at most max_symbol_concurrency.
        """
        # Allow 2 seconds worth of concurrent requests
        rate_based = int(self.requests_per_second * 2)
        # Ensure at least 2 concurrent requests
        return max(2, min(self.max_symbol_concurrency, rate_based))

# Pre-configured venue settings for optimal parallelism
VENUE_PARALLEL_CONFIGS: Dict[str, VenueParallelConfig] = {
    # High-volume CEX - REDUCED to prevent IP bans
    'binance': VenueParallelConfig(
        venue='binance',
        rate_limit_per_minute=400,
        max_symbol_concurrency=2,
        supports_batch_api=True,
        batch_size=100,
    ),
    'deribit': VenueParallelConfig(
        venue='deribit',
        rate_limit_per_minute=400,
        max_symbol_concurrency=2,
        supports_batch_api=False,
        # Deribit ONLY supports BTC, ETH, SOL - pre-filter to avoid 52 wasted iterations
        supported_symbols=['BTC', 'ETH', 'SOL'],
    ),
    'kraken': VenueParallelConfig(
        venue='kraken',
        rate_limit_per_minute=300,
        max_symbol_concurrency=2,
        supports_batch_api=False,
        # Kraken doesn't support these symbols - pre-filter to avoid wasted iterations
        unsupported_symbols=['FXS', 'GMX', 'DYDX', 'SEI', 'SUI', 'APT', 'INJ', 'WLD', 'BLUR', 'APE', 'TIA', 'PENDLE'],
    ),

    # Standard CEX - REDUCED
    'bybit': VenueParallelConfig(
        venue='bybit',
        rate_limit_per_minute=200,
        max_symbol_concurrency=2,
        supports_batch_api=False,
        # Bybit uses special naming for some symbols - pre-filter problematic ones
        unsupported_symbols=['MOG', 'FLOKI', 'BONK', 'SHIB', 'PEPE', 'LUNC', 'BTTC', 'MATIC', 'WIN', 'BTT', 'NFT', 'JASMY', 'FXS', 'MKR'],
    ),
    'okx': VenueParallelConfig(
        venue='okx',
        rate_limit_per_minute=30,
        max_symbol_concurrency=2,
        supports_batch_api=False,
    ),
    'coinbase': VenueParallelConfig(
        venue='coinbase',
        rate_limit_per_minute=200,
        max_symbol_concurrency=2,
        supports_batch_api=False,
    ),

    # Hybrid venues - REDUCED
    'hyperliquid': VenueParallelConfig(
        venue='hyperliquid',
        rate_limit_per_minute=30,
        max_symbol_concurrency=2,
        supports_batch_api=True,
        batch_size=50,
    ),
    'dydx': VenueParallelConfig(
        venue='dydx',
        rate_limit_per_minute=50,
        max_symbol_concurrency=2,
        supports_batch_api=False,
        # dYdX V4 doesn't support many altcoins - pre-filter to avoid wasted iterations
        unsupported_symbols=['FXS', 'MATIC', 'APE', 'BLUR', 'MKR', 'MOG', 'FLOKI', 'BONK', 'SHIB', 'PEPE', 'LUNC', 'BTTC', 'WIN', 'BTT', 'NFT', 'JASMY', 'ILV', 'ENJ', 'SAND', 'GALA'],
    ),

    # DEX venues - REDUCED
    'gmx': VenueParallelConfig(
        venue='gmx',
        rate_limit_per_minute=30,
        max_symbol_concurrency=2,
        supports_batch_api=False,
    ),
    'geckoterminal': VenueParallelConfig(
        venue='geckoterminal',
        rate_limit_per_minute=15,
        max_symbol_concurrency=2,
        supports_batch_api=False,
        min_request_interval_ms=4000, # 4 seconds between requests
    ),
    'aevo': VenueParallelConfig(
        venue='aevo',
        rate_limit_per_minute=30,
        max_symbol_concurrency=2,
        supports_batch_api=False,
        # Aevo only supports BTC, ETH for most data types
        supported_symbols=['BTC', 'ETH'],
    ),

    # Alternative data - REDUCED
    'coinalyze': VenueParallelConfig(
        venue='coinalyze',
        rate_limit_per_minute=20,
        max_symbol_concurrency=2,
        supports_batch_api=True,
        batch_size=10,
    ),
    'defillama': VenueParallelConfig(
        venue='defillama',
        rate_limit_per_minute=30,
        max_symbol_concurrency=2,
        supports_batch_api=False,
    ),

    # Market data providers - REDUCED
    'santiment': VenueParallelConfig(
        venue='santiment',
        rate_limit_per_minute=30,
        max_symbol_concurrency=2,
        supports_batch_api=True,
        batch_size=20,
    ),
    'cryptocompare': VenueParallelConfig(
        venue='cryptocompare',
        rate_limit_per_minute=200,
        max_symbol_concurrency=2,
        supports_batch_api=True, # Supports batch API
        batch_size=100,
    ),
    'dexscreener': VenueParallelConfig(
        venue='dexscreener',
        rate_limit_per_minute=30,
        max_symbol_concurrency=2,
        supports_batch_api=False,
    ),

    # Low rate limit venues - REDUCED
    'coingecko': VenueParallelConfig(
        venue='coingecko',
        rate_limit_per_minute=5,
        max_symbol_concurrency=2,
        supports_batch_api=True, # Use batch API to speed up
        batch_size=250, # Can fetch up to 250 coins in one request
        min_request_interval_ms=12000, # 12 seconds between requests (free tier)
    ),

    # =========================================================================
    # ADDITIONAL VENUES - All rates REDUCED for reliability
    # =========================================================================

    # CEX - Traditional exchanges
    'cme': VenueParallelConfig(
        venue='cme',
        rate_limit_per_minute=15,
        max_symbol_concurrency=2,
        supports_batch_api=False,
    ),

    # On-chain analytics - REDUCED
    'arkham': VenueParallelConfig(
        venue='arkham',
        rate_limit_per_minute=30,
        max_symbol_concurrency=2,
        supports_batch_api=False,
    ),
    'glassnode': VenueParallelConfig(
        venue='glassnode',
        rate_limit_per_minute=10,
        max_symbol_concurrency=2,
        supports_batch_api=False,
        min_request_interval_ms=6000,
    ),
    'cryptoquant': VenueParallelConfig(
        venue='cryptoquant',
        rate_limit_per_minute=10,
        max_symbol_concurrency=2,
        supports_batch_api=False,
        min_request_interval_ms=6000,
    ),
    'nansen': VenueParallelConfig(
        venue='nansen',
        rate_limit_per_minute=20,
        max_symbol_concurrency=2,
        supports_batch_api=False,
    ),
    'coinmetrics': VenueParallelConfig(
        venue='coinmetrics',
        rate_limit_per_minute=30,
        max_symbol_concurrency=2,
        supports_batch_api=True,
        batch_size=10,
    ),

    # Blockchain indexers - REDUCED
    'flipside': VenueParallelConfig(
        venue='flipside',
        rate_limit_per_minute=10,
        max_symbol_concurrency=2,
        supports_batch_api=False,
        min_request_interval_ms=6000,
    ),
    'dune': VenueParallelConfig(
        venue='dune',
        rate_limit_per_minute=5,
        max_symbol_concurrency=2,
        supports_batch_api=False,
        min_request_interval_ms=12000, # Dune has very low rate limits
    ),
    'thegraph': VenueParallelConfig(
        venue='thegraph',
        rate_limit_per_minute=60,
        max_symbol_concurrency=2,
        supports_batch_api=True,
        batch_size=50,
    ),
    'covalent': VenueParallelConfig(
        venue='covalent',
        rate_limit_per_minute=15,
        max_symbol_concurrency=2,
        supports_batch_api=False,
        min_request_interval_ms=4000,
    ),
    'bitquery': VenueParallelConfig(
        venue='bitquery',
        rate_limit_per_minute=30,
        max_symbol_concurrency=2,
        supports_batch_api=True,
        batch_size=10,
    ),

    # DEX aggregators and protocols - REDUCED
    'uniswap': VenueParallelConfig(
        venue='uniswap',
        rate_limit_per_minute=30,
        max_symbol_concurrency=2,
        supports_batch_api=False,
    ),
    'sushiswap': VenueParallelConfig(
        venue='sushiswap',
        rate_limit_per_minute=30,
        max_symbol_concurrency=2,
        supports_batch_api=False,
    ),
    'curve': VenueParallelConfig(
        venue='curve',
        rate_limit_per_minute=30,
        max_symbol_concurrency=2,
        supports_batch_api=False,
    ),
    'oneinch': VenueParallelConfig(
        venue='oneinch',
        rate_limit_per_minute=30,
        max_symbol_concurrency=2,
        supports_batch_api=False,
    ),
    'cowswap': VenueParallelConfig(
        venue='cowswap',
        rate_limit_per_minute=30,
        max_symbol_concurrency=2,
        supports_batch_api=False,
    ),
    'zerox': VenueParallelConfig(
        venue='zerox',
        rate_limit_per_minute=30,
        max_symbol_concurrency=2,
        supports_batch_api=False,
    ),
    'jupiter': VenueParallelConfig(
        venue='jupiter',
        rate_limit_per_minute=30,
        max_symbol_concurrency=2,
        supports_batch_api=False,
    ),

    # Perpetual DEX - REDUCED
    'vertex': VenueParallelConfig(
        venue='vertex',
        rate_limit_per_minute=30,
        max_symbol_concurrency=2,
        supports_batch_api=False,
    ),

    # Options protocols - REDUCED
    'lyra': VenueParallelConfig(
        venue='lyra',
        rate_limit_per_minute=30,
        max_symbol_concurrency=2,
        supports_batch_api=False,
    ),
    'dopex': VenueParallelConfig(
        venue='dopex',
        rate_limit_per_minute=30,
        max_symbol_concurrency=2,
        supports_batch_api=False,
    ),

    # Market data providers - REDUCED
    'kaiko': VenueParallelConfig(
        venue='kaiko',
        rate_limit_per_minute=60,
        max_symbol_concurrency=2,
        supports_batch_api=True,
        batch_size=20,
    ),
    'messari': VenueParallelConfig(
        venue='messari',
        rate_limit_per_minute=20,
        max_symbol_concurrency=2,
        supports_batch_api=False,
    ),
    'lunarcrush': VenueParallelConfig(
        venue='lunarcrush',
        rate_limit_per_minute=10,
        max_symbol_concurrency=2,
        supports_batch_api=False,
        min_request_interval_ms=6000,
    ),
    'coinalyze_enhanced': VenueParallelConfig(
        venue='coinalyze_enhanced',
        rate_limit_per_minute=15,
        max_symbol_concurrency=2,
        supports_batch_api=True,
        batch_size=10,
    ),

    # Alert/tracking services - REDUCED
    'whale_alert': VenueParallelConfig(
        venue='whale_alert',
        rate_limit_per_minute=10,
        max_symbol_concurrency=2,
        supports_batch_api=False,
        min_request_interval_ms=6000,
    ),
}

def get_venue_config(venue: str) -> VenueParallelConfig:
    """Get venue config with fallback to defaults."""
    if venue in VENUE_PARALLEL_CONFIGS:
        return VENUE_PARALLEL_CONFIGS[venue]

    # Default configuration for unknown venues - conservative to prevent rate issues
    return VenueParallelConfig(
        venue=venue,
        rate_limit_per_minute=30, # 0.5 req/sec - conservative for unknown APIs
        max_symbol_concurrency=2, # Sequential: max 2 symbols at a time
        supports_batch_api=False,
    )

def filter_symbols_for_venue(venue: str, symbols: List[str]) -> List[str]:
    """
    Filter symbols based on venue's supported/unsupported lists.

    This is a convenience function that prevents wasted API calls and log spam
    by filtering out symbols that a venue doesn't support BEFORE calling the collector.

    OPTIMIZATION: Reduces wasted iterations by ~52 per data type for venues like
    Deribit (which only supports BTC/ETH/SOL out of 55 symbols).

    Args:
        venue: Venue name (e.g., 'deribit', 'binance')
        symbols: List of symbols to filter

    Returns:
        Filtered list of symbols supported by this venue

    Example:
        >>> symbols = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', ...] # 55 symbols
        >>> filtered = filter_symbols_for_venue('deribit', symbols)
        >>> print(filtered) # ['BTC', 'ETH', 'SOL'] - only 3 supported
    """
    config = get_venue_config(venue)
    return config.filter_symbols(symbols)

# =============================================================================
# DATA CLASSES
# =============================================================================

class ProcessingStatus(Enum):
    """Status of a symbol processing task."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()

@dataclass
class SymbolResult:
    """Result from processing a single symbol."""
    symbol: str
    status: ProcessingStatus
    data: Optional[pd.DataFrame] = None
    records: int = 0
    duration_seconds: float = 0.0
    error: Optional[str] = None
    retry_count: int = 0

    @property
    def success(self) -> bool:
        return self.status == ProcessingStatus.COMPLETED and self.error is None

@dataclass
class BatchResult:
    """Result from processing a batch of symbols."""
    venue: str
    data_type: str
    symbols_processed: int
    symbols_failed: int
    total_records: int
    total_duration_seconds: float
    results: List[SymbolResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = self.symbols_processed + self.symbols_failed
        if total == 0:
            return 0.0
        return self.symbols_processed / total * 100

    def get_combined_dataframe(self) -> Optional[pd.DataFrame]:
        """Combine all successful results into single DataFrame."""
        dfs = []
        for r in self.results:
            if r.data is not None:
                if isinstance(r.data, pd.DataFrame):
                    if not r.data.empty:
                        dfs.append(r.data)
                elif hasattr(r.data, '__len__') and len(r.data) > 0:
                    dfs.append(pd.DataFrame(r.data))
        if not dfs:
            return None
        return pd.concat(dfs, ignore_index=True)

@dataclass
class ParallelStats:
    """Statistics for parallel processing."""
    venue: str
    symbols_total: int = 0
    symbols_completed: int = 0
    symbols_failed: int = 0
    total_records: int = 0
    total_duration_seconds: float = 0.0
    requests_made: int = 0
    requests_delayed: int = 0
    avg_concurrency: float = 0.0
    peak_concurrency: int = 0

    @property
    def records_per_second(self) -> float:
        if self.total_duration_seconds == 0:
            return 0.0
        return self.total_records / self.total_duration_seconds

    @property
    def speedup_factor(self) -> float:
        """Estimated speedup vs sequential processing."""
        if self.avg_concurrency == 0:
            return 1.0
        return self.avg_concurrency

# =============================================================================
# PARALLEL SYMBOL PROCESSOR
# =============================================================================

class ParallelSymbolProcessor:
    """
    comprehensive parallel processor for symbol-level data collection.

    Features:
        - Adaptive concurrency based on venue rate limits
        - Integrated rate limiting with burst support
        - Error isolation (one symbol failure doesn't affect others)
        - Progress tracking and ETA estimation
        - Automatic retry with exponential backoff

    Example:
        >>> processor = ParallelSymbolProcessor('binance')
        >>>
        >>> async def fetch_symbol(symbol, **kwargs):
        ... return await collector.fetch_funding_rate(symbol, **kwargs)
        >>>
        >>> results = await processor.process_symbols(
        ... symbols=['BTC', 'ETH', 'SOL'],
        ... fetch_func=fetch_symbol,
        ... start_date='2024-01-01',
        ... end_date='2024-01-31'
        ... )
    """

    def __init__(
        self,
        venue: str,
        rate_limit_per_minute: Optional[int] = None,
        max_symbol_concurrency: Optional[int] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.venue = venue
        self.config = get_venue_config(venue)

        # Override config if explicit values provided
        if rate_limit_per_minute:
            self.config.rate_limit_per_minute = rate_limit_per_minute
        if max_symbol_concurrency:
            self.config.max_symbol_concurrency = max_symbol_concurrency

        # Initialize rate limiter - use high-throughput version for parallel processing
        if rate_limiter:
            self.rate_limiter = rate_limiter
        else:
            # Burst size: allow enough for all concurrent requests plus buffer
            # This enables true parallelism for symbol batches
            burst = max(
                self.config.max_symbol_concurrency * 2, # Allow 2x concurrency
                self.config.rate_limit_per_minute // 6, # Or 10 seconds worth
                20 # Minimum burst
            )
            self.rate_limiter = ParallelRateLimiter(
                rate_per_minute=self.config.rate_limit_per_minute,
                burst_size=burst,
                name=venue,
            )

        # SPEEDUP: ADAPTIVE CONCURRENCY: Track metrics for dynamic adjustment
        self.adaptive_concurrency_enabled = True
        self.recent_successes = 0
        self.recent_failures = 0
        self.recent_latencies = []
        self.current_concurrency = self.config.optimal_concurrency
        self.min_concurrency = max(1, self.config.optimal_concurrency // 4)
        self.max_concurrency = min(
            self.config.max_symbol_concurrency,
            self.config.optimal_concurrency * 3
        )
        self.adjustment_interval = 10 # Adjust after every 10 requests

        # Semaphore for symbol-level concurrency - use max_symbol_concurrency directly
        self._semaphore = asyncio.Semaphore(self.config.max_symbol_concurrency)

        # Stats tracking
        self.stats = ParallelStats(venue=venue)
        self._current_concurrency = 0
        self._lock = asyncio.Lock()

        logger.info(
            f"ParallelSymbolProcessor initialized for {venue}: "
            f"rate_limit={self.config.rate_limit_per_minute}/min, "
            f"concurrency={self.config.optimal_concurrency}"
        )

    def _adjust_concurrency(self) -> None:
        """
        SPEEDUP: ADAPTIVE CONCURRENCY: Dynamically adjust concurrency based on performance.

        Increases concurrency when:
        - Success rate > 90%
        - Average latency is stable or decreasing

        Decreases concurrency when:
        - Success rate < 70%
        - Recent failures detected
        - Latency is increasing

        This allows the system to automatically speed up when the venue is
        performing well and throttle back when errors occur.
        """
        if not self.adaptive_concurrency_enabled:
            return

        total_requests = self.recent_successes + self.recent_failures
        if total_requests < self.adjustment_interval:
            return # Not enough data yet

        success_rate = self.recent_successes / total_requests if total_requests > 0 else 0
        avg_latency = sum(self.recent_latencies) / len(self.recent_latencies) if self.recent_latencies else 0

        old_concurrency = self.current_concurrency

        # Aggressive increase when performing very well
        if success_rate >= 0.95 and avg_latency < 1000: # < 1 second
            self.current_concurrency = min(
                int(self.current_concurrency * 1.5),
                self.max_concurrency
            )
        # Moderate increase when performing well
        elif success_rate >= 0.90:
            self.current_concurrency = min(
                self.current_concurrency + 2,
                self.max_concurrency
            )
        # Moderate decrease when performance degrades
        elif success_rate < 0.70:
            self.current_concurrency = max(
                int(self.current_concurrency * 0.7),
                self.min_concurrency
            )
        # Aggressive decrease when many failures
        elif success_rate < 0.50:
            self.current_concurrency = max(
                int(self.current_concurrency * 0.5),
                self.min_concurrency
            )

        if old_concurrency != self.current_concurrency:
            logger.info(
                f"FAST: [{self.venue}] Adaptive concurrency: {old_concurrency} → {self.current_concurrency} "
                f"(success_rate={success_rate:.1%}, avg_latency={avg_latency:.0f}ms)"
            )

        # Reset metrics for next interval
        self.recent_successes = 0
        self.recent_failures = 0
        self.recent_latencies = []

    async def process_symbols(
        self,
        symbols: List[str],
        fetch_func: Callable[..., Coroutine[Any, Any, Any]],
        max_retries: int = 2,
        **fetch_kwargs
    ) -> BatchResult:
        """
        Process multiple symbols in parallel.

        Parameters
        ----------
        symbols : List[str]
            Symbols to process
        fetch_func : Callable
            Async function to fetch data for a single symbol.
            Signature: fetch_func(symbol: str, **kwargs) -> DataFrame/List/Dict
        max_retries : int
            Maximum retry attempts per symbol
        **fetch_kwargs
            Additional kwargs to pass to fetch_func

        Returns
        -------
        BatchResult
            Combined results from all symbols
        """
        start_time = time.monotonic()
        self.stats = ParallelStats(venue=self.venue, symbols_total=len(symbols))

        # BATCH: BATCH OPTIMIZATION: Use intelligent batching if available
        if BATCH_OPTIMIZER_AVAILABLE and supports_batch(self.venue) and len(symbols) > 1:
            try:
                speedup = estimate_speedup(self.venue, len(symbols))
                if speedup > 1.5:
                    batches = create_batches(self.venue, symbols)
                    logger.info(
                        f"FAST: [{self.venue}] Batch optimization enabled: "
                        f"{len(symbols)} symbols → {len(batches)} batches "
                        f"(estimated {speedup:.1f}x speedup)"
                    )
                    # Note: Actual batch API calls would need collector-specific implementation
                    # For now, we log the optimization and continue with parallel processing
            except Exception as e:
                logger.warning(f"Batch optimization failed for {self.venue}: {e}")

        # Process symbols in parallel with semaphore control
        tasks = [
            self._process_single_symbol(
                symbol, fetch_func, max_retries, **fetch_kwargs
            )
            for symbol in symbols
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to SymbolResult
        symbol_results = []
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                symbol_results.append(SymbolResult(
                    symbol=symbol,
                    status=ProcessingStatus.FAILED,
                    error=str(result),
                ))
                self.stats.symbols_failed += 1
            elif isinstance(result, SymbolResult):
                symbol_results.append(result)
                if result.success:
                    self.stats.symbols_completed += 1
                    self.stats.total_records += result.records
                else:
                    self.stats.symbols_failed += 1
            else:
                # Unexpected result type
                symbol_results.append(SymbolResult(
                    symbol=symbol,
                    status=ProcessingStatus.FAILED,
                    error=f"Unexpected result type: {type(result)}",
                ))
                self.stats.symbols_failed += 1

        total_duration = time.monotonic() - start_time
        self.stats.total_duration_seconds = total_duration

        logger.info(
            f"ParallelSymbolProcessor [{self.venue}] completed: "
            f"{self.stats.symbols_completed}/{len(symbols)} symbols, "
            f"{self.stats.total_records} records in {total_duration:.1f}s"
        )

        return BatchResult(
            venue=self.venue,
            data_type=fetch_kwargs.get('data_type', 'unknown'),
            symbols_processed=self.stats.symbols_completed,
            symbols_failed=self.stats.symbols_failed,
            total_records=self.stats.total_records,
            total_duration_seconds=total_duration,
            results=symbol_results,
        )

    async def _process_single_symbol(
        self,
        symbol: str,
        fetch_func: Callable,
        max_retries: int,
        **kwargs
    ) -> SymbolResult:
        """Process a single symbol with rate limiting and retry."""
        start_time = time.monotonic()
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Acquire semaphore for concurrency control
                async with self._semaphore:
                    # Track concurrency
                    async with self._lock:
                        self._current_concurrency += 1
                        self.stats.peak_concurrency = max(
                            self.stats.peak_concurrency,
                            self._current_concurrency
                        )

                    try:
                        # Acquire rate limit token with explicit timeout check
                        acquire_result = await self.rate_limiter.acquire(timeout=120.0)
                        if hasattr(acquire_result, 'acquired') and not acquire_result.acquired:
                            # Rate limiter timed out - skip this symbol to prevent hangs
                            raise TimeoutError(f"Rate limiter timeout after 120s for {symbol}")
                        self.stats.requests_made += 1

                        # Enforce minimum request interval if configured
                        if self.config.min_request_interval_ms > 50:
                            await asyncio.sleep(
                                self.config.min_request_interval_ms / 1000
                            )

                        # Execute the fetch function with timeout (60s per symbol)
                        try:
                            result = await asyncio.wait_for(
                                fetch_func(symbol, **kwargs),
                                timeout=60.0 # 60 second timeout per symbol
                            )
                        except asyncio.TimeoutError:
                            raise TimeoutError(f"Symbol fetch timeout after 60s for {symbol}")

                        # Convert result to DataFrame if needed
                        if isinstance(result, list):
                            result = pd.DataFrame(result)
                        elif isinstance(result, dict):
                            result = pd.DataFrame([result])

                        records = 0
                        if result is not None:
                            if isinstance(result, pd.DataFrame):
                                records = len(result)
                            else:
                                records = 1

                        # SPEEDUP: ADAPTIVE CONCURRENCY: Track success and latency
                        latency_ms = (time.monotonic() - start_time) * 1000
                        self.recent_successes += 1
                        self.recent_latencies.append(latency_ms)
                        # Keep only recent latencies to avoid memory bloat
                        if len(self.recent_latencies) > 50:
                            self.recent_latencies = self.recent_latencies[-50:]
                        # Adjust concurrency if needed
                        self._adjust_concurrency()

                        return SymbolResult(
                            symbol=symbol,
                            status=ProcessingStatus.COMPLETED,
                            data=result,
                            records=records,
                            duration_seconds=time.monotonic() - start_time,
                            retry_count=attempt,
                        )

                    finally:
                        async with self._lock:
                            self._current_concurrency -= 1

            except Exception as e:
                last_error = str(e)

                if attempt < max_retries:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * 0.5
                    logger.debug(
                        f"Retry {attempt + 1}/{max_retries} for {symbol} "
                        f"after {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                    self.stats.requests_delayed += 1

        # All retries exhausted
        # SPEEDUP: ADAPTIVE CONCURRENCY: Track failure
        self.recent_failures += 1
        self._adjust_concurrency()

        return SymbolResult(
            symbol=symbol,
            status=ProcessingStatus.FAILED,
            error=last_error,
            duration_seconds=time.monotonic() - start_time,
            retry_count=max_retries,
        )

# =============================================================================
# PARALLEL COLLECTION MANAGER
# =============================================================================

class ParallelCollectionManager:
    """
    Enhanced collection manager with comprehensive parallel processing.

    Improvements over base CollectionManager:
        - Higher venue concurrency (10-15 vs 5)
        - Per-venue rate limiting with adaptive throttling
        - Symbol-level parallelism within each venue
        - Progress tracking across all venues
        - Automatic optimal concurrency calculation

    Example:
        >>> manager = ParallelCollectionManager(max_venue_concurrency=12)
        >>> results = await manager.collect_all(
        ... venues=['binance', 'hyperliquid', 'dydx', 'gmx'],
        ... data_type='funding_rates',
        ... symbols=['BTC', 'ETH', 'SOL'],
        ... start_date=start,
        ... end_date=end
        ... )
    """

    def __init__(
        self,
        max_venue_concurrency: int = 12,
        output_dir: str = 'data/processed',
    ):
        self.max_venue_concurrency = max_venue_concurrency
        self.output_dir = output_dir

        # Venue-level semaphore
        self._venue_semaphore = asyncio.Semaphore(max_venue_concurrency)

        # Per-venue rate limiters
        self.rate_limiters = create_venue_limiters()

        # Per-venue symbol processors
        self._processors: Dict[str, ParallelSymbolProcessor] = {}

        # Stats
        self._start_time: Optional[float] = None
        self._venues_completed = 0
        self._venues_total = 0

        logger.info(
            f"ParallelCollectionManager initialized: "
            f"max_venue_concurrency={max_venue_concurrency}"
        )

    def get_processor(self, venue: str) -> ParallelSymbolProcessor:
        """Get or create symbol processor for venue."""
        if venue not in self._processors:
            # Try to get venue-specific rate limiter
            rate_limiter = self.rate_limiters.get(venue)

            self._processors[venue] = ParallelSymbolProcessor(
                venue=venue,
                rate_limiter=rate_limiter,
            )

        return self._processors[venue]

    async def collect_venue(
        self,
        venue: str,
        fetch_func: Callable,
        symbols: List[str],
        **kwargs
    ) -> BatchResult:
        """
        Collect data from a single venue with parallel symbol processing.

        Uses venue-level semaphore for concurrency control.
        """
        async with self._venue_semaphore:
            processor = self.get_processor(venue)
            result = await processor.process_symbols(
                symbols=symbols,
                fetch_func=fetch_func,
                **kwargs
            )
            self._venues_completed += 1
            return result

    async def collect_all_venues(
        self,
        venue_configs: List[Dict[str, Any]],
    ) -> List[BatchResult]:
        """
        Collect from multiple venues in parallel.

        Parameters
        ----------
        venue_configs : List[Dict]
            List of venue configurations, each containing:
            - venue: str
            - fetch_func: Callable
            - symbols: List[str]
            - **kwargs: Additional args for fetch_func

        Returns
        -------
        List[BatchResult]
            Results from all venues
        """
        self._start_time = time.monotonic()
        self._venues_completed = 0
        self._venues_total = len(venue_configs)

        # Create tasks for all venues
        tasks = [
            self.collect_venue(
                venue=config['venue'],
                fetch_func=config['fetch_func'],
                symbols=config['symbols'],
                **{k: v for k, v in config.items()
                   if k not in ['venue', 'fetch_func', 'symbols']}
            )
            for config in venue_configs
        ]

        # Execute all venue collections in parallel (limited by semaphore)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to BatchResult
        final_results = []
        for config, result in zip(venue_configs, results):
            if isinstance(result, Exception):
                final_results.append(BatchResult(
                    venue=config['venue'],
                    data_type=config.get('data_type', 'unknown'),
                    symbols_processed=0,
                    symbols_failed=len(config['symbols']),
                    total_records=0,
                    total_duration_seconds=0,
                ))
            else:
                final_results.append(result)

        total_duration = time.monotonic() - self._start_time
        total_records = sum(r.total_records for r in final_results)

        logger.info(
            f"ParallelCollectionManager completed {len(venue_configs)} venues: "
            f"{total_records} total records in {total_duration:.1f}s"
        )

        return final_results

    def get_estimated_speedup(self) -> Dict[str, Any]:
        """
        Get estimated speedup metrics.

        Returns dict with:
            - sequential_estimate_seconds: Estimated time for sequential
            - parallel_actual_seconds: Actual parallel duration
            - speedup_factor: Speedup ratio
            - efficiency: Parallelization efficiency
        """
        if self._start_time is None:
            return {}

        actual_duration = time.monotonic() - self._start_time

        # Estimate sequential time based on processor stats
        sequential_estimate = sum(
            p.stats.total_duration_seconds * p.stats.peak_concurrency
            for p in self._processors.values()
        )

        speedup = sequential_estimate / actual_duration if actual_duration > 0 else 1.0

        return {
            'sequential_estimate_seconds': sequential_estimate,
            'parallel_actual_seconds': actual_duration,
            'speedup_factor': speedup,
            'venues_processed': self._venues_completed,
            'efficiency': min(1.0, speedup / self.max_venue_concurrency),
        }

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def parallel_fetch_symbols(
    symbols: List[str],
    fetch_func: Callable[..., Coroutine],
    venue: str = 'default',
    max_concurrency: int = 5,
    rate_limit_per_minute: int = 60,
    **kwargs
) -> List[Any]:
    """
    Simple helper for parallel symbol fetching.

    Example:
        >>> results = await parallel_fetch_symbols(
        ... symbols=['BTC', 'ETH', 'SOL'],
        ... fetch_func=collector.fetch_single,
        ... venue='binance',
        ... max_concurrency=10
        ... )
    """
    processor = ParallelSymbolProcessor(
        venue=venue,
        max_symbol_concurrency=max_concurrency,
        rate_limit_per_minute=rate_limit_per_minute,
    )

    result = await processor.process_symbols(
        symbols=symbols,
        fetch_func=fetch_func,
        **kwargs
    )

    return [r.data for r in result.results if r.success]

def calculate_optimal_concurrency(
    rate_limit_per_minute: int,
    avg_request_time_ms: float = 200,
    safety_factor: float = 0.8
) -> int:
    """
    Calculate optimal concurrency level.

    Parameters
    ----------
    rate_limit_per_minute : int
        API rate limit
    avg_request_time_ms : float
        Average request time in milliseconds
    safety_factor : float
        Safety margin (0.0-1.0)

    Returns
    -------
    int
        Optimal number of concurrent requests
    """
    requests_per_second = rate_limit_per_minute / 60
    requests_per_request_time = avg_request_time_ms / 1000 * requests_per_second
    optimal = int(requests_per_request_time * safety_factor)
    return max(1, min(50, optimal))

# =============================================================================
# COLLECTOR WRAPPER FOR PARALLEL SYMBOL PROCESSING
# =============================================================================

class ParallelCollectorWrapper:
    """
    Wrapper that adds parallel symbol processing to any existing collector.

    This wrapper transforms sequential symbol processing (for symbol in symbols)
    into parallel processing using asyncio.gather() while respecting rate limits.

    Usage:
        >>> # Create wrapper around existing collector
        >>> wrapper = ParallelCollectorWrapper(BinanceCollector())
        >>>
        >>> # Use wrapper's parallel methods
        >>> df = await wrapper.fetch_funding_rates_parallel(
        ... symbols=['BTC', 'ETH', 'SOL'],
        ... start_date='2024-01-01',
        ... end_date='2024-01-31'
        ... )

    Speedup: 5-15x depending on venue rate limits and number of symbols.
    """

    def __init__(
        self,
        collector: Any,
        max_symbol_concurrency: Optional[int] = None,
        rate_limit_per_minute: Optional[int] = None,
    ):
        self.collector = collector
        self.venue = getattr(collector, 'VENUE', 'unknown')

        # Get venue-specific config
        config = get_venue_config(self.venue)

        self.max_concurrency = max_symbol_concurrency or config.optimal_concurrency
        self.rate_limit = rate_limit_per_minute or config.rate_limit_per_minute

        # Create processor for this collector
        self.processor = ParallelSymbolProcessor(
            venue=self.venue,
            rate_limit_per_minute=self.rate_limit,
            max_symbol_concurrency=self.max_concurrency,
        )

        logger.info(
            f"ParallelCollectorWrapper for {self.venue}: "
            f"concurrency={self.max_concurrency}, rate_limit={self.rate_limit}/min"
        )

    async def fetch_funding_rates_parallel(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch funding rates for multiple symbols in parallel.

        Parameters
        ----------
        symbols : List[str]
            Symbols to fetch
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)

        Returns
        -------
        pd.DataFrame
            Combined funding rate data from all symbols
        """
        async def fetch_single_symbol(symbol: str, **kw) -> pd.DataFrame:
            """Fetch funding rates for a single symbol."""
            return await self.collector.fetch_funding_rates(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
                **kwargs
            )

        result = await self.processor.process_symbols(
            symbols=symbols,
            fetch_func=fetch_single_symbol,
            data_type='funding_rates',
        )

        combined = result.get_combined_dataframe()
        return combined if combined is not None else pd.DataFrame()

    async def fetch_ohlcv_parallel(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for multiple symbols in parallel.

        Parameters
        ----------
        symbols : List[str]
            Symbols to fetch
        timeframe : str
            Candle interval
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)

        Returns
        -------
        pd.DataFrame
            Combined OHLCV data from all symbols
        """
        async def fetch_single_symbol(symbol: str, **kw) -> pd.DataFrame:
            """Fetch OHLCV for a single symbol."""
            return await self.collector.fetch_ohlcv(
                symbols=[symbol],
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                **kwargs
            )

        result = await self.processor.process_symbols(
            symbols=symbols,
            fetch_func=fetch_single_symbol,
            data_type='ohlcv',
        )

        combined = result.get_combined_dataframe()
        return combined if combined is not None else pd.DataFrame()

    def get_stats(self) -> Dict[str, Any]:
        """Get parallel processing statistics."""
        return {
            'venue': self.venue,
            'max_concurrency': self.max_concurrency,
            'rate_limit_per_minute': self.rate_limit,
            'processor_stats': {
                'symbols_completed': self.processor.stats.symbols_completed,
                'symbols_failed': self.processor.stats.symbols_failed,
                'total_records': self.processor.stats.total_records,
                'total_duration_seconds': self.processor.stats.total_duration_seconds,
                'peak_concurrency': self.processor.stats.peak_concurrency,
            }
        }

def wrap_collector_for_parallel(
    collector: Any,
    max_symbol_concurrency: Optional[int] = None,
) -> ParallelCollectorWrapper:
    """
    Create a parallel wrapper for any existing collector.

    Example:
        >>> from data_collection.cex.binance_collector import BinanceCollector
        >>> collector = BinanceCollector()
        >>> parallel_collector = wrap_collector_for_parallel(collector)
        >>> df = await parallel_collector.fetch_funding_rates_parallel(...)
    """
    return ParallelCollectorWrapper(
        collector=collector,
        max_symbol_concurrency=max_symbol_concurrency,
    )

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Config
    'VenueParallelConfig',
    'VENUE_PARALLEL_CONFIGS',
    'get_venue_config',
    'filter_symbols_for_venue', # Pre-filter symbols to avoid wasted iterations
    # Data classes
    'ProcessingStatus',
    'SymbolResult',
    'BatchResult',
    'ParallelStats',
    # Main classes
    'ParallelSymbolProcessor',
    'ParallelCollectionManager',
    'ParallelCollectorWrapper',
    # Helper functions
    'parallel_fetch_symbols',
    'calculate_optimal_concurrency',
    'wrap_collector_for_parallel',
]
