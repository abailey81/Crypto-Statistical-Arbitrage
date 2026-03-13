"""
Batch API Optimizer for Phase 1 Data Collection.

Automatically groups symbols and requests to leverage batch API capabilities
of venues that support them, significantly improving collection speed.

Supported Batch Venues:
- Binance: 50-100 symbols per request
- Hyperliquid: Batch market data
- Santiment: Batch metrics
- CoinGecko: Multiple coins per request
- And more...

Version: 1.0.0
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    """Batch API configuration for a venue."""
    venue: str
    supports_batch: bool
    max_batch_size: int
    optimal_batch_size: int
    batch_delay_ms: int = 0 # Delay between batches to avoid rate limits

class BatchOptimizer:
    """
    Intelligent batch request optimizer.

    Automatically detects and uses batch API capabilities to reduce
    the number of API calls and improve collection speed.
    """

    # Venue-specific batch configurations
    BATCH_CONFIGS = {
        'binance': BatchConfig(
            venue='binance',
            supports_batch=True,
            max_batch_size=100,
            optimal_batch_size=50,
            batch_delay_ms=100
        ),
        'bybit': BatchConfig(
            venue='bybit',
            supports_batch=False, # Individual requests only
            max_batch_size=1,
            optimal_batch_size=1
        ),
        'okx': BatchConfig(
            venue='okx',
            supports_batch=True,
            max_batch_size=20,
            optimal_batch_size=10,
            batch_delay_ms=50
        ),
        'coinbase': BatchConfig(
            venue='coinbase',
            supports_batch=False,
            max_batch_size=1,
            optimal_batch_size=1
        ),
        'kraken': BatchConfig(
            venue='kraken',
            supports_batch=True,
            max_batch_size=15,
            optimal_batch_size=10,
            batch_delay_ms=50
        ),
        'hyperliquid': BatchConfig(
            venue='hyperliquid',
            supports_batch=True,
            max_batch_size=30,
            optimal_batch_size=20,
            batch_delay_ms=100
        ),
        'dydx': BatchConfig(
            venue='dydx',
            supports_batch=True,
            max_batch_size=25,
            optimal_batch_size=15,
            batch_delay_ms=50
        ),
        'coingecko': BatchConfig(
            venue='coingecko',
            supports_batch=True,
            max_batch_size=100,
            optimal_batch_size=50,
            batch_delay_ms=200
        ),
        'cryptocompare': BatchConfig(
            venue='cryptocompare',
            supports_batch=True,
            max_batch_size=20,
            optimal_batch_size=10,
            batch_delay_ms=100
        ),
        'santiment': BatchConfig(
            venue='santiment',
            supports_batch=True,
            max_batch_size=30,
            optimal_batch_size=15,
            batch_delay_ms=150
        ),
        'geckoterminal': BatchConfig(
            venue='geckoterminal',
            supports_batch=True,
            max_batch_size=30,
            optimal_batch_size=20,
            batch_delay_ms=100
        ),
        'dexscreener': BatchConfig(
            venue='dexscreener',
            supports_batch=True,
            max_batch_size=20,
            optimal_batch_size=10,
            batch_delay_ms=100
        ),
    }

    @classmethod
    def get_config(cls, venue: str) -> BatchConfig:
        """Get batch configuration for venue."""
        return cls.BATCH_CONFIGS.get(
            venue.lower(),
            BatchConfig( # Default for unknown venues
                venue=venue,
                supports_batch=False,
                max_batch_size=1,
                optimal_batch_size=1
            )
        )

    @classmethod
    def supports_batch(cls, venue: str) -> bool:
        """Check if venue supports batch requests."""
        config = cls.get_config(venue)
        return config.supports_batch

    @classmethod
    def get_batch_size(cls, venue: str, num_symbols: int) -> int:
        """Get optimal batch size for venue and number of symbols."""
        config = cls.get_config(venue)

        if not config.supports_batch:
            return 1

        # Use optimal size, but don't exceed max or available symbols
        return min(config.optimal_batch_size, config.max_batch_size, num_symbols)

    @classmethod
    def create_batches(cls, venue: str, symbols: List[str]) -> List[List[str]]:
        """
        Create optimal symbol batches for a venue.

        Returns:
            List of batches, each containing up to optimal_batch_size symbols
        """
        config = cls.get_config(venue)

        if not config.supports_batch or len(symbols) <= 1:
            return [[s] for s in symbols]

        batch_size = config.optimal_batch_size
        batches = []

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batches.append(batch)

        if batches:
            logger.debug(
                f"[{venue}] Created {len(batches)} batches "
                f"(batch_size={batch_size}, total_symbols={len(symbols)})"
            )

        return batches

    @classmethod
    async def execute_batched(
        cls,
        venue: str,
        symbols: List[str],
        fetch_func: Callable,
        **kwargs
    ) -> List[Any]:
        """
        Execute batch-optimized requests.

        Args:
            venue: Venue name
            symbols: List of symbols to fetch
            fetch_func: Async function to call for each batch
                        Should accept (symbols: List[str], **kwargs)
            **kwargs: Additional arguments for fetch_func

        Returns:
            Combined results from all batches
        """
        config = cls.get_config(venue)
        batches = cls.create_batches(venue, symbols)

        if len(batches) == 1:
            # Single batch or single symbol - execute directly
            return await fetch_func(symbols, **kwargs)

        # Multiple batches - execute with delays
        all_results = []

        for i, batch in enumerate(batches):
            if i > 0 and config.batch_delay_ms > 0:
                # Add delay between batches to avoid rate limits
                await asyncio.sleep(config.batch_delay_ms / 1000.0)

            try:
                result = await fetch_func(batch, **kwargs)
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                logger.warning(
                    f"[{venue}] Batch {i+1}/{len(batches)} failed: {e}"
                )
                # Continue with next batch
                continue

        return all_results

    @classmethod
    def estimate_speedup(cls, venue: str, num_symbols: int) -> float:
        """
        Estimate speedup factor from using batch API.

        Returns:
            Speedup multiplier (e.g., 5.0 means 5x faster)
        """
        config = cls.get_config(venue)

        if not config.supports_batch or num_symbols <= 1:
            return 1.0

        # Calculate number of requests
        batch_size = min(config.optimal_batch_size, num_symbols)
        batched_requests = (num_symbols + batch_size - 1) // batch_size
        individual_requests = num_symbols

        # Speedup = individual / batched
        speedup = individual_requests / batched_requests

        return speedup

    @classmethod
    def get_batch_statistics(cls, venue: str, num_symbols: int) -> Dict[str, Any]:
        """Get detailed batch statistics."""
        config = cls.get_config(venue)

        if not config.supports_batch:
            return {
                "supports_batch": False,
                "strategy": "individual",
                "batches": num_symbols,
                "avg_batch_size": 1,
                "speedup": 1.0
            }

        batch_size = min(config.optimal_batch_size, num_symbols)
        num_batches = (num_symbols + batch_size - 1) // batch_size
        avg_batch_size = num_symbols / num_batches if num_batches > 0 else 0

        return {
            "supports_batch": True,
            "strategy": "batched",
            "batches": num_batches,
            "optimal_batch_size": config.optimal_batch_size,
            "max_batch_size": config.max_batch_size,
            "avg_batch_size": avg_batch_size,
            "speedup": cls.estimate_speedup(venue, num_symbols),
            "batch_delay_ms": config.batch_delay_ms
        }

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def supports_batch(venue: str) -> bool:
    """Check if venue supports batch API."""
    return BatchOptimizer.supports_batch(venue)

def create_batches(venue: str, symbols: List[str]) -> List[List[str]]:
    """Create optimal batches for venue."""
    return BatchOptimizer.create_batches(venue, symbols)

def estimate_speedup(venue: str, num_symbols: int) -> float:
    """Estimate speedup from batching."""
    return BatchOptimizer.estimate_speedup(venue, num_symbols)

__all__ = [
    'BatchOptimizer',
    'BatchConfig',
    'supports_batch',
    'create_batches',
    'estimate_speedup'
]
