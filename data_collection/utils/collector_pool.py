"""
Collector Pool for Efficient Resource Management.

This module provides a singleton collector pool that caches and reuses
collector instances, preventing the massive overhead of creating and
destroying collectors repeatedly.

PROBLEM SOLVED:
- Before: 527 collector instances created with 0 records collected
- Each collector creation involves: HTTP session setup, rate limiter init,
  config parsing, logger setup, etc.
- Collectors were destroyed after each use, wasting all that initialization

SOLUTION:
- Collectors are cached by venue name
- Same collector instance is reused across all data types
- Collectors are only closed at the end of the full run

Version: 1.0.0
"""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional, Type

logger = logging.getLogger(__name__)

class CollectorPool:
    """
    Singleton pool for caching and reusing collector instances.

    This eliminates the overhead of creating ~600 collector instances
    when most of them produce 0 records. Instead, collectors are created
    once per venue and reused across all data types.

    Usage:
        >>> pool = get_collector_pool()
        >>> collector = pool.get_or_create('binance', BinanceCollector, config)
        >>> # Use collector...
        >>> # Don't call collector.close() - pool manages lifecycle
        >>>
        >>> # At end of run:
        >>> await pool.close_all()
    """

    _instance: Optional['CollectorPool'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._collectors: Dict[str, Any] = {}
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._collectors: Dict[str, Any] = {}
            self._creation_count = 0
            self._reuse_count = 0
            self._initialized = True
            logger.info("CollectorPool initialized (singleton)")

    def get_or_create(
        self,
        venue: str,
        collector_class: Type,
        config: Optional[Dict] = None,
    ) -> Any:
        """
        Get existing collector or create a new one.

        Parameters
        ----------
        venue : str
            Venue identifier (e.g., 'binance', 'hyperliquid')
        collector_class : Type
            Collector class to instantiate if not cached
        config : Dict, optional
            Configuration for the collector

        Returns
        -------
        Any
            Collector instance (cached or newly created)
        """
        if venue in self._collectors:
            self._reuse_count += 1
            logger.debug(f"CollectorPool: Reusing cached collector for '{venue}'")
            return self._collectors[venue]

        # Create new collector
        try:
            collector = collector_class(config or {})
            self._collectors[venue] = collector
            self._creation_count += 1
            logger.info(f"CollectorPool: Created new collector for '{venue}' (total: {self._creation_count})")
            return collector
        except Exception as e:
            logger.error(f"CollectorPool: Failed to create collector for '{venue}': {e}")
            raise

    def get(self, venue: str) -> Optional[Any]:
        """Get a cached collector without creating."""
        return self._collectors.get(venue)

    def has(self, venue: str) -> bool:
        """Check if a collector exists in the pool."""
        return venue in self._collectors

    async def close(self, venue: str) -> None:
        """Close and remove a specific collector."""
        if venue in self._collectors:
            collector = self._collectors.pop(venue)
            if hasattr(collector, 'close'):
                try:
                    await collector.close()
                    logger.debug(f"CollectorPool: Closed collector for '{venue}'")
                except Exception as e:
                    logger.warning(f"CollectorPool: Error closing '{venue}': {e}")

    async def close_all(self) -> None:
        """Close all collectors in the pool."""
        logger.info(f"CollectorPool: Closing {len(self._collectors)} collectors...")

        close_tasks = []
        for venue, collector in list(self._collectors.items()):
            if hasattr(collector, 'close'):
                close_tasks.append(self._safe_close(venue, collector))

        if close_tasks:
            await asyncio.gather(*close_tasks)

        self._collectors.clear()
        logger.info(
            f"CollectorPool: All collectors closed. "
            f"Stats: {self._creation_count} created, {self._reuse_count} reused"
        )

    async def _safe_close(self, venue: str, collector: Any) -> None:
        """Safely close a collector with error handling."""
        try:
            await collector.close()
        except Exception as e:
            logger.warning(f"CollectorPool: Error closing '{venue}': {e}")

    @property
    def stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'active_collectors': len(self._collectors),
            'total_created': self._creation_count,
            'total_reused': self._reuse_count,
            'reuse_ratio': (
                self._reuse_count / max(1, self._creation_count + self._reuse_count) * 100
            ),
            'venues': list(self._collectors.keys()),
        }

    def reset_stats(self) -> None:
        """Reset statistics (useful for testing)."""
        self._creation_count = 0
        self._reuse_count = 0

    def __repr__(self) -> str:
        return (
            f"CollectorPool(active={len(self._collectors)}, "
            f"created={self._creation_count}, reused={self._reuse_count})"
        )

# Global pool instance
_collector_pool: Optional[CollectorPool] = None

def get_collector_pool() -> CollectorPool:
    """Get the global collector pool singleton."""
    global _collector_pool
    if _collector_pool is None:
        _collector_pool = CollectorPool()
    return _collector_pool

def create_pooled_collector_factory(
    registry_get_func: Callable[[str], Any]
) -> Callable[[str], Any]:
    """
    Create a collector factory that uses the pool.

    This wraps an existing collector factory (like CollectorRegistry.get_collector)
    to use the pool for caching.

    Parameters
    ----------
    registry_get_func : Callable
        Function that creates/gets collectors (e.g., registry.get_collector)

    Returns
    -------
    Callable
        Pooled collector factory function

    Example:
        >>> from data_collection import CollectorRegistry
        >>> registry = CollectorRegistry()
        >>> pooled_factory = create_pooled_collector_factory(registry.get_collector)
        >>> collector = pooled_factory('binance') # Uses pool
    """
    pool = get_collector_pool()

    def pooled_factory(venue: str) -> Any:
        # Check pool first
        if pool.has(venue):
            pool._reuse_count += 1
            return pool.get(venue)

        # Create via registry and cache in pool
        collector = registry_get_func(venue)
        if collector is not None:
            pool._collectors[venue] = collector
            pool._creation_count += 1
            logger.info(f"CollectorPool: Cached new collector for '{venue}'")
        return collector

    return pooled_factory

__all__ = [
    'CollectorPool',
    'get_collector_pool',
    'create_pooled_collector_factory',
]
