"""
Binance Data Collector - Primary CEX Data Source

validated collector for the world's largest cryptocurrency derivatives exchange.
Provides comprehensive data for perpetual futures, quarterly futures, and spot markets.

Supported Data Types:
    - Perpetual funding rates (8-hour intervals, 00:00/08:00/16:00 UTC)
    - Spot and futures OHLCV (klines) with multiple timeframes
    - Open interest history (contracts and notional value)
    - Long/short ratio (global accounts, positions, top traders)
    - Taker buy/sell volume (order flow analysis)
    - Top trader positions (smart money tracking)
    - Mark price, index price, and basis
    - Liquidation data streams

API Documentation:
    - Spot API: https://binance-docs.github.io/apidocs/spot/en/
    - Futures API: https://binance-docs.github.io/apidocs/futures/en/
    - Coin-M API: https://binance-docs.github.io/apidocs/delivery/en/

Rate Limits:
    - Weight-based: 1200 weight per minute (IP-based)
    - Order limits: 10 orders/second, 100,000 orders/day
    - WebSocket: 5 messages/second per connection

Contract Specifications (USDT-M Perpetuals):
    - Funding interval: 8 hours
    - Funding cap: +/- 0.75% per interval (some pairs +/- 3%)
    - Tick size: Varies by pair (BTC: $0.10)
    - Leverage: Up to 125x (varies by pair and tier)

Statistical Arbitrage Applications:
    - Funding rate arbitrage (perp vs spot, cross-exchange)
    - Basis trading (futures premium/discount)
    - Sentiment signals (long/short ratios, taker flow)
    - Smart money tracking (top trader positions)
    - Cross-venue spread detection
    - Open interest divergence signals

Version: 2.0.0
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib
import hmac
import time

from ..base_collector import BaseCollector, CollectionStats
from ..utils.rate_limiter import get_shared_rate_limiter
from ..utils.retry_handler import RetryHandler, safe_float, safe_int

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================

class ContractType(Enum):
    """Binance futures contract types."""
    PERPETUAL = 'PERPETUAL'
    CURRENT_QUARTER = 'CURRENT_QUARTER'
    NEXT_QUARTER = 'NEXT_QUARTER'
    CURRENT_MONTH = 'CURRENT_MONTH'
    NEXT_MONTH = 'NEXT_MONTH'

class MarketType(Enum):
    """Market type classification."""
    SPOT = 'spot'
    USDT_MARGINED = 'usdt_m'
    COIN_MARGINED = 'coin_m'
    QUARTERLY = 'quarterly'

class PositionSide(Enum):
    """Position side for derivatives."""
    LONG = 'long'
    SHORT = 'short'
    BOTH = 'both'

class FundingTrend(Enum):
    """Funding rate trend classification."""
    HIGHLY_POSITIVE = 'highly_positive'
    POSITIVE = 'positive'
    NEUTRAL = 'neutral'
    NEGATIVE = 'negative'
    HIGHLY_NEGATIVE = 'highly_negative'

class MarketSentiment(Enum):
    """Market sentiment based on positioning data."""
    STRONGLY_BULLISH = 'strongly_bullish'
    BULLISH = 'bullish'
    NEUTRAL = 'neutral'
    BEARISH = 'bearish'
    STRONGLY_BEARISH = 'strongly_bearish'

class TakerAggression(Enum):
    """Taker order flow aggression classification."""
    AGGRESSIVE_BUYING = 'aggressive_buying'
    MODERATE_BUYING = 'moderate_buying'
    BALANCED = 'balanced'
    MODERATE_SELLING = 'moderate_selling'
    AGGRESSIVE_SELLING = 'aggressive_selling'

class OITrend(Enum):
    """Open interest trend classification."""
    STRONG_INCREASE = 'strong_increase'
    INCREASE = 'increase'
    STABLE = 'stable'
    DECREASE = 'decrease'
    STRONG_DECREASE = 'strong_decrease'

# =============================================================================
# Symbol Listing Dates (Unix timestamp in milliseconds)
# =============================================================================

# Symbols that were listed after 2020 - used to skip unnecessary API calls
# Dates are set very conservatively (1 month after approximate listing) to avoid 400 errors
SYMBOL_LISTING_DATES = {
    'PRIMEUSDT': 1682121600000, # April 22, 2023 (very conservative: March 23 + 1 month)
    'GNSUSDT': 1675209600000, # February 1, 2023 (very conservative: Jan 1 + 1 month)
    'PYTHUSDT': 1702857600000, # December 18, 2023 (very conservative + 1 month)
    'WLDUSDT': 1692489600000, # August 20, 2023 (very conservative + 1 month)
    'SEIUSDT': 1694889600000, # September 17, 2023 (very conservative + 1 month)
    'SUIUSDT': 1685577600000, # June 1, 2023 (very conservative + 1 month)
    'APTUSDT': 1668211200000, # November 12, 2022 (very conservative + 1 month)
    'ARBUSDT': 1682121600000, # April 22, 2023 (very conservative: March 23 + 1 month)
    'OPUSDT': 1656633600000, # July 1, 2022 (very conservative + 1 month)
    'STRKUSDT': 1711584000000, # March 28, 2024 (very conservative: Feb 27 + 1 month)
    # Add more as needed
}

# Symbols to completely skip on Binance Futures (not available or no data)
# If symbols consistently fail even with conservative listing dates, add them here
SKIP_SYMBOLS = {
    'GNSUSDT', # Consistently fails with 400 - not available on Binance Futures
    'PRIMEUSDT', # Consistently fails with 400 - not available on Binance Futures
}

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class FundingRate:
    """
    Funding rate data point for perpetual futures.
    
    Funding payments occur every 8 hours at 00:00, 08:00, 16:00 UTC.
    Positive rate means longs pay shorts; negative means shorts pay longs.
    """
    timestamp: datetime
    symbol: str
    funding_rate: float
    mark_price: float
    funding_interval_hours: int = 8
    
    @property
    def annualized_rate(self) -> float:
        """Annualized funding rate (simple, not compounded)."""
        periods_per_year = 365 * 24 / self.funding_interval_hours
        return self.funding_rate * periods_per_year
    
    @property
    def annualized_rate_pct(self) -> float:
        """Annualized rate as percentage."""
        return self.annualized_rate * 100
    
    @property
    def daily_rate(self) -> float:
        """Daily funding rate (sum of 3 payments)."""
        payments_per_day = 24 / self.funding_interval_hours
        return self.funding_rate * payments_per_day
    
    @property
    def trend(self) -> FundingTrend:
        """Classify funding trend."""
        rate_pct = self.funding_rate * 100
        if rate_pct > 0.05:
            return FundingTrend.HIGHLY_POSITIVE
        elif rate_pct > 0.01:
            return FundingTrend.POSITIVE
        elif rate_pct > -0.01:
            return FundingTrend.NEUTRAL
        elif rate_pct > -0.05:
            return FundingTrend.NEGATIVE
        else:
            return FundingTrend.HIGHLY_NEGATIVE
    
    @property
    def is_arbitrage_opportunity(self) -> bool:
        """Check if rate presents arbitrage opportunity (> 0.03% or < -0.03%)."""
        return abs(self.funding_rate) > 0.0003
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'funding_rate': self.funding_rate,
            'funding_rate_pct': self.funding_rate * 100,
            'mark_price': self.mark_price,
            'annualized_rate': self.annualized_rate,
            'annualized_rate_pct': self.annualized_rate_pct,
            'daily_rate': self.daily_rate,
            'trend': self.trend.value,
            'is_arbitrage_opportunity': self.is_arbitrage_opportunity,
        }

@dataclass
class LongShortRatio:
    """
    Long/short ratio data from Binance.
    
    Available types:
    - Global accounts: Percentage of accounts long vs short
    - Global positions: Percentage of position value long vs short
    - Top traders (accounts): Top 20% of traders by margin balance
    - Top traders (positions): Top 20% of traders by position value
    """
    timestamp: datetime
    symbol: str
    long_short_ratio: float
    long_account_pct: float
    short_account_pct: float
    ratio_type: str
    
    @property
    def sentiment(self) -> MarketSentiment:
        """Market sentiment classification."""
        if self.long_short_ratio > 1.5:
            return MarketSentiment.STRONGLY_BULLISH
        elif self.long_short_ratio > 1.1:
            return MarketSentiment.BULLISH
        elif self.long_short_ratio > 0.9:
            return MarketSentiment.NEUTRAL
        elif self.long_short_ratio > 0.67:
            return MarketSentiment.BEARISH
        else:
            return MarketSentiment.STRONGLY_BEARISH
    
    @property
    def imbalance_pct(self) -> float:
        """Position imbalance as percentage (positive = more longs)."""
        return self.long_account_pct - self.short_account_pct
    
    @property
    def is_extreme(self) -> bool:
        """Check if positioning is extreme (potential reversal signal)."""
        return self.long_short_ratio > 2.0 or self.long_short_ratio < 0.5
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'long_short_ratio': self.long_short_ratio,
            'long_account_pct': self.long_account_pct,
            'short_account_pct': self.short_account_pct,
            'ratio_type': self.ratio_type,
            'sentiment': self.sentiment.value,
            'imbalance_pct': self.imbalance_pct,
            'is_extreme': self.is_extreme,
        }

@dataclass
class TakerVolume:
    """
    Taker buy/sell volume data for order flow analysis.
    
    Taker volume represents aggressive orders that cross the spread.
    Buy volume = market buys hitting asks; Sell volume = market sells hitting bids.
    """
    timestamp: datetime
    symbol: str
    buy_volume: float
    sell_volume: float
    buy_value: float
    sell_value: float
    
    @property
    def buy_sell_ratio(self) -> float:
        """Buy/sell volume ratio."""
        return self.buy_volume / self.sell_volume if self.sell_volume > 0 else float('inf')
    
    @property
    def net_taker_flow(self) -> float:
        """Net taker flow in quote currency (positive = buying pressure)."""
        return self.buy_value - self.sell_value
    
    @property
    def total_volume(self) -> float:
        """Total taker volume."""
        return self.buy_volume + self.sell_volume
    
    @property
    def total_value(self) -> float:
        """Total taker value in quote currency."""
        return self.buy_value + self.sell_value
    
    @property
    def aggression(self) -> TakerAggression:
        """Classify taker aggression."""
        ratio = self.buy_sell_ratio
        if ratio > 1.3:
            return TakerAggression.AGGRESSIVE_BUYING
        elif ratio > 1.1:
            return TakerAggression.MODERATE_BUYING
        elif ratio > 0.9:
            return TakerAggression.BALANCED
        elif ratio > 0.77:
            return TakerAggression.MODERATE_SELLING
        else:
            return TakerAggression.AGGRESSIVE_SELLING
    
    @property
    def buy_pressure_pct(self) -> float:
        """Buy pressure as percentage of total."""
        total = self.buy_value + self.sell_value
        return (self.buy_value / total * 100) if total > 0 else 50.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'buy_volume': self.buy_volume,
            'sell_volume': self.sell_volume,
            'buy_value': self.buy_value,
            'sell_value': self.sell_value,
            'buy_sell_ratio': self.buy_sell_ratio,
            'net_taker_flow': self.net_taker_flow,
            'total_volume': self.total_volume,
            'aggression': self.aggression.value,
            'buy_pressure_pct': self.buy_pressure_pct,
        }

@dataclass
class OpenInterestData:
    """
    Open interest data for futures markets.

    OI represents total outstanding contracts not yet settled.
    Rising OI with price = trend confirmation; Divergence = potential reversal.
    """
    timestamp: datetime
    symbol: str
    open_interest: float
    open_interest_value: float

    @property
    def average_position_size(self) -> float:
        """Implied average position size in USD."""
        return self.open_interest_value / self.open_interest if self.open_interest > 0 else 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'open_interest': self.open_interest,
            'open_interest_value': self.open_interest_value,
            'average_position_size': self.average_position_size,
        }

@dataclass
class AggregateTradeData:
    """
    Aggregated trade data from Binance futures.

    Aggregate trades combine individual trades executed at the same price
    in the same order into a single record, reducing data volume.
    """
    timestamp: datetime
    symbol: str
    agg_trade_id: int
    price: float
    quantity: float
    first_trade_id: int
    last_trade_id: int
    is_buyer_maker: bool

    @property
    def trade_value(self) -> float:
        """Trade value in quote currency (USDT)."""
        return self.price * self.quantity

    @property
    def trade_count(self) -> int:
        """Number of individual trades aggregated."""
        return self.last_trade_id - self.first_trade_id + 1

    @property
    def side(self) -> str:
        """Trade side from taker perspective."""
        return 'sell' if self.is_buyer_maker else 'buy'

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'agg_trade_id': self.agg_trade_id,
            'price': self.price,
            'quantity': self.quantity,
            'first_trade_id': self.first_trade_id,
            'last_trade_id': self.last_trade_id,
            'is_buyer_maker': self.is_buyer_maker,
            'trade_value': self.trade_value,
            'trade_count': self.trade_count,
            'side': self.side,
        }

@dataclass
class MarketSnapshot:
    """
    Complete market snapshot combining multiple data points.
    
    Provides comprehensive view for quick analysis and screening.
    """
    timestamp: datetime
    symbol: str
    mark_price: float
    index_price: float
    last_funding_rate: float
    next_funding_time: datetime
    open_interest: float
    open_interest_value: float
    volume_24h: float
    volume_quote_24h: float
    price_change_24h_pct: float
    high_24h: float
    low_24h: float
    
    @property
    def basis(self) -> float:
        """Mark price premium/discount to index (basis)."""
        return self.mark_price - self.index_price
    
    @property
    def basis_pct(self) -> float:
        """Basis as percentage of index price."""
        return (self.basis / self.index_price * 100) if self.index_price > 0 else 0
    
    @property
    def funding_annualized(self) -> float:
        """Annualized funding rate."""
        return self.last_funding_rate * 3 * 365
    
    @property
    def range_24h_pct(self) -> float:
        """24h price range as percentage."""
        if self.low_24h > 0:
            return (self.high_24h - self.low_24h) / self.low_24h * 100
        return 0
    
    @property
    def turnover_ratio(self) -> float:
        """Volume/OI ratio (turnover indicator)."""
        return self.volume_quote_24h / self.open_interest_value if self.open_interest_value > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'mark_price': self.mark_price,
            'index_price': self.index_price,
            'basis': self.basis,
            'basis_pct': self.basis_pct,
            'last_funding_rate': self.last_funding_rate,
            'funding_annualized': self.funding_annualized,
            'next_funding_time': self.next_funding_time,
            'open_interest': self.open_interest,
            'open_interest_value': self.open_interest_value,
            'volume_24h': self.volume_24h,
            'volume_quote_24h': self.volume_quote_24h,
            'price_change_24h_pct': self.price_change_24h_pct,
            'high_24h': self.high_24h,
            'low_24h': self.low_24h,
            'range_24h_pct': self.range_24h_pct,
            'turnover_ratio': self.turnover_ratio,
        }

# =============================================================================
# Global Circuit Breaker for Binance IP Bans
# =============================================================================

class BinanceCircuitBreaker:
    """
    Global circuit breaker for Binance IP bans (HTTP 418).

    When Binance returns 418 (IP banned), this circuit breaker trips and
    prevents ALL subsequent Binance requests from even attempting, saving
    bandwidth and preventing further ban extensions.

    This is a class-level singleton shared across all BinanceCollector instances.
    """
    _instance: Optional['BinanceCircuitBreaker'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._is_tripped = False
            cls._instance._trip_time: Optional[datetime] = None
            cls._instance._trip_count = 0
            cls._instance._requests_blocked = 0
        return cls._instance

    def trip(self) -> None:
        """Trip the circuit breaker (called when 418 is received)."""
        if not self._is_tripped:
            self._is_tripped = True
            self._trip_time = datetime.utcnow()
            self._trip_count += 1
            logger.error(
                f" BINANCE CIRCUIT BREAKER TRIPPED: IP banned (418). "
                f"All subsequent Binance requests will be blocked. "
                f"Consider using VPN or waiting 24+ hours."
            )

    def is_open(self) -> bool:
        """Check if circuit breaker is open (requests should be blocked)."""
        return self._is_tripped

    def block_request(self) -> None:
        """Record a blocked request for stats."""
        self._requests_blocked += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            'is_tripped': self._is_tripped,
            'trip_time': self._trip_time.isoformat() if self._trip_time else None,
            'trip_count': self._trip_count,
            'requests_blocked': self._requests_blocked,
        }

    def reset(self) -> None:
        """
        Reset the circuit breaker (for testing or manual recovery).

        WARNING: Only call this if you're sure the IP ban has been lifted.
        """
        self._is_tripped = False
        self._trip_time = None
        logger.info("Binance circuit breaker reset manually")

def get_binance_circuit_breaker() -> BinanceCircuitBreaker:
    """Get the global Binance circuit breaker singleton."""
    return BinanceCircuitBreaker()

# =============================================================================
# Collector Class
# =============================================================================

class BinanceCollector(BaseCollector):
    """
    Binance data collector for spot and derivatives markets.

    validated implementation with:
    - Weight-based rate limiting (1200/min)
    - Automatic retry with exponential backoff
    - Response caching for expensive queries
    - Comprehensive error handling
    - Collection statistics tracking
    - Global circuit breaker for IP bans (HTTP 418)

    Attributes:
        VENUE: Exchange identifier ('binance')
        VENUE_TYPE: Exchange type ('CEX')
        BASE_URL_SPOT: Spot API endpoint
        BASE_URL_FUTURES: USDT-M futures endpoint
        FUNDING_INTERVAL_HOURS: Hours between funding payments (8)

    Example:
        >>> config = {'rate_limit': 400, 'timeout': 30}
        >>> async with BinanceCollector(config) as collector:
        ... funding = await collector.fetch_funding_rates(['BTC', 'ETH'], '2024-01-01', '2024-01-31')
        ... snapshot = await collector.fetch_market_snapshot(['BTC', 'ETH'])
    """

    VENUE = 'binance'
    VENUE_TYPE = 'CEX'

    # Collection manager compatibility attributes
    supported_data_types = ['funding_rates', 'ohlcv', 'open_interest', 'trades', 'liquidations']
    venue = 'binance'
    requires_auth = False # Public endpoints available for all data types

    BASE_URL_SPOT = 'https://api.binance.com'
    BASE_URL_FUTURES = 'https://fapi.binance.com'
    BASE_URL_DELIVERY = 'https://dapi.binance.com'

    FUNDING_INTERVAL_HOURS = 8
    FUNDINGS_PER_DAY = 3

    MAX_FUNDING_RECORDS = 1000
    MAX_KLINES_RECORDS = 1500
    MAX_OI_RECORDS = 500
    MAX_RATIO_RECORDS = 500

    ENDPOINT_WEIGHTS = {
        'funding_rate': 1, 'klines': 5, 'open_interest': 5,
        'long_short_ratio': 1, 'taker_volume': 1, 'top_traders': 1,
        'ticker': 1, 'exchange_info': 10, 'premium_index': 1,
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Binance collector."""
        super().__init__(config)

        rate_limit = config.get('rate_limit', 400)
        # Use shared rate limiter to avoid re-initialization overhead
        self.rate_limiter = get_shared_rate_limiter('binance', rate=rate_limit, per=60.0, burst=min(20, rate_limit // 20))
        # OPTIMIZATION: Reduced max_delay from 60s to 30s to avoid long stalls
        # Also reduced max_retries from 5 to 3 for faster fail-through on persistent errors
        self.retry_handler = RetryHandler(max_retries=config.get('max_retries', 3), base_delay=1.0, max_delay=30.0)

        # Global circuit breaker for IP bans (shared across all instances)
        self._circuit_breaker = get_binance_circuit_breaker()
        
        self.timeout = aiohttp.ClientTimeout(total=config.get('timeout', 30))
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Load API keys from config or environment variables
        self.api_key = config.get('api_key') or config.get('binance_api_key') or os.getenv('BINANCE_API_KEY', '')
        self.api_secret = config.get('api_secret') or config.get('binance_api_secret') or os.getenv('BINANCE_SECRET_KEY', '')
        
        # SPEEDUP: SPEEDUP: Intelligent caching with data-type-specific TTLs
        # Different data types have different update frequencies
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl_map = {
            'exchange_info': 86400, # 24 hours (changes rarely)
            'ticker_24h': 300, # 5 minutes (batch endpoint, high volume)
            'premium_index': 60, # 1 minute (updates frequently)
            'funding_rate': 3600, # 1 hour (8h interval, cache longer)
            'ohlcv': 300, # 5 minutes (OHLCV 1h can be cached)
            'snapshot': 60, # 1 minute (live data)
            'default': config.get('cache_ttl', 60)
        }
        self._cache_ttl = config.get('cache_ttl', 60) # Keep for backward compat
        
        self.collection_stats = {
            'records_collected': 0, 'api_calls': 0, 'weight_used': 0,
            'cache_hits': 0, 'errors': 0, 'rate_limit_hits': 0
        }
        
        # Semaphore to limit concurrent HTTP requests (prevents rate limit exhaustion)
        self._request_sem = asyncio.Semaphore(2)

        logger.info(f"Initialized Binance collector (rate_limit={rate_limit} weight/min)")

    async def __aenter__(self):
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            # SPEEDUP: SPEEDUP: Optimized headers with compression support
            headers = {
                'User-Agent': 'CryptoStatArb/2.0',
                'Accept-Encoding': 'gzip, deflate', # Enable response compression (2-3x bandwidth reduction)
            }
            if self.api_key:
                headers['X-MBX-APIKEY'] = self.api_key

            # SPEEDUP: SPEEDUP: Aggressive connection pooling for high-throughput collection
            # - limit=100 (up from 20): Support 5+ venues × 20 symbols concurrently
            # - limit_per_host=30 (up from 10): Binance rate limit is 1200/min = 20/sec
            # - ttl_dns_cache=300: Cache DNS lookups for 5 minutes (saves 50-100ms per symbol)
            # - force_close=False: Keep-alive connections (reuse TCP, save TLS handshake)
            # - enable_cleanup_closed=True: Clean up stale connections
            # Expected speedup: 2-3x
            connector = aiohttp.TCPConnector(
                limit=100, # Total connection pool size (was 20)
                limit_per_host=30, # Per-host connections (was 10)
                ttl_dns_cache=300, # DNS cache TTL in seconds
                force_close=False, # Keep-alive connections
                enable_cleanup_closed=True # Clean up closed connections
            )
            self.session = aiohttp.ClientSession(timeout=self.timeout, headers=headers, connector=connector)
        return self.session
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
        self._cache.clear()
        cb_stats = self._circuit_breaker.get_stats()
        if cb_stats['is_tripped']:
            logger.warning(f"Binance collector closed. Circuit breaker was TRIPPED - blocked {cb_stats['requests_blocked']} requests")
        logger.info(f"Binance collector closed. Stats: {self.collection_stats}")
    
    def _get_cached(self, key: str, data_type: str = 'default') -> Optional[Any]:
        """Get cached value with intelligent TTL based on data type."""
        if key in self._cache:
            timestamp, value = self._cache[key]
            # SPEEDUP: SPEEDUP: Use data-type-specific TTL for better cache efficiency
            ttl = self._cache_ttl_map.get(data_type, self._cache_ttl)
            if (datetime.utcnow() - timestamp).total_seconds() < ttl:
                self.collection_stats['cache_hits'] += 1
                return value
            del self._cache[key] # Remove expired entry
        return None

    def _set_cached(self, key: str, value: Any):
        """Set cached value with timestamp."""
        self._cache[key] = (datetime.utcnow(), value)
    
    # Binance 1000x tokens - these trade as 1000{SYMBOL}USDT on futures
    # Due to their low price, Binance uses 1000x denomination
    BINANCE_1000X_TOKENS = {
        'SHIB', 'PEPE', 'FLOKI', 'BONK', 'LUNC', 'XEC', 'BTT', 'BTTC',
        'LADYS', 'SATS', 'RATS', 'ORDI', 'DOGE2', 'BABYDOGE'
    }

    def _format_symbol_futures(self, symbol: str) -> str:
        """Convert to Binance futures format (BTCUSDT).

        Handles 1000x tokens (SHIB, PEPE, etc.) that trade as 1000SHIBUSDT.
        """
        symbol = symbol.upper()
        for suffix in ['/USDT', '-USDT', ':USDT', '/USD', '-USD', 'USDT', 'PERP']:
            symbol = symbol.replace(suffix, '')

        # Handle 1000x tokens - Binance uses 1000SHIBUSDT format
        if symbol in self.BINANCE_1000X_TOKENS:
            return f"1000{symbol}USDT"

        return f"{symbol}USDT"
    
    def _format_symbol_spot(self, symbol: str) -> str:
        """Convert to Binance spot format."""
        symbol = symbol.upper().replace('/', '').replace('-', '')
        if not symbol.endswith(('USDT', 'BTC', 'ETH', 'BNB')):
            symbol = f"{symbol}USDT"
        return symbol
    
    def _parse_symbol(self, binance_symbol: str) -> str:
        """Convert Binance symbol back to base symbol.

        Handles 1000x tokens (1000SHIBUSDT → SHIB).
        """
        symbol = binance_symbol.replace('USDT', '').replace('USD', '')

        # Handle 1000x tokens - convert 1000SHIB back to SHIB
        if symbol.startswith('1000'):
            base = symbol[4:] # Remove '1000' prefix
            if base in self.BINANCE_1000X_TOKENS:
                return base

        return symbol
    
    async def _make_request(self, url: str, params: Optional[Dict] = None, weight: int = 1, use_cache: bool = False) -> Any:
        """Make rate-limited HTTP request with circuit breaker protection."""
        async with self._request_sem:
            # CIRCUIT BREAKER CHECK: Block all requests if IP is banned
            if self._circuit_breaker.is_open():
                self._circuit_breaker.block_request()
                self.collection_stats['errors'] += 1
                return None # Fail fast without making HTTP request

            if use_cache:
                cache_key = f"{url}_{hash(frozenset((params or {}).items()))}"
                cached = self._get_cached(cache_key)
                if cached is not None:
                    return cached

            session = await self._get_session()

            async def _request():
                # Double-check circuit breaker (may have tripped during wait)
                if self._circuit_breaker.is_open():
                    self._circuit_breaker.block_request()
                    return None

                for _ in range(weight):
                    acquire_result = await self.rate_limiter.acquire(timeout=120.0)
                    if hasattr(acquire_result, 'acquired') and not acquire_result.acquired:
                        raise TimeoutError("Rate limiter timeout after 120s")

                async with session.get(url, params=params) as response:
                    self.collection_stats['weight_used'] = int(response.headers.get('X-MBX-USED-WEIGHT-1M', weight))

                    if response.status == 429:
                        self.collection_stats['rate_limit_hits'] += 1
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Binance rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        raise aiohttp.ClientResponseError(response.request_info, response.history, status=429)

                    if response.status == 418:
                        # 418 = IP banned - TRIP THE CIRCUIT BREAKER
                        # This will block ALL subsequent Binance requests globally
                        self._circuit_breaker.trip()
                        self.collection_stats['errors'] += 1
                        return None # Return None instead of raising to allow graceful degradation

                    if response.status == 401:
                        # 401 = Unauthorized - endpoint requires authentication
                        if not hasattr(self, '_auth_warning_logged'):
                            logger.warning("Binance endpoint requires authentication - skipping")
                            self._auth_warning_logged = True
                        return None # Skip silently

                    if response.status == 403:
                        # 403 = Forbidden - may be geo-blocked or API key issue
                        logger.debug(f"Binance 403 Forbidden for {url}")
                        return None

                    if response.status == 400:
                        # 400 = Bad Request - often means symbol doesn't exist or date range invalid
                        # Common causes:
                        # - Symbol not listed on Binance Futures
                        # - Start time before symbol listing
                        # - End time in the future (Binance server time)
                        text = await response.text()
                        error_msg = text[:200] if text else "Unknown error"
                        logger.debug(f"Binance 400 Bad Request: {error_msg}")
                        return None

                    response.raise_for_status()
                    return await response.json()

            try:
                result = await self.retry_handler.execute(_request)
                self.collection_stats['api_calls'] += 1
                if use_cache and result is not None:
                    self._set_cached(cache_key, result)
                return result
            except Exception as e:
                self.collection_stats['errors'] += 1
                raise
    
    async def _fetch_funding_rate_single(self, symbol: str, start_ts: int, end_ts: int) -> List[Dict]:
        """SPEEDUP: SPEEDUP: Helper to fetch funding rates for a single symbol (for parallel processing)."""
        binance_symbol = self._format_symbol_futures(symbol)
        logger.info(f"Fetching Binance funding rates: {binance_symbol}")

        all_data = []
        current_ts = start_ts
        symbol_records = 0

        while current_ts < end_ts:
            url = f"{self.BASE_URL_FUTURES}/fapi/v1/fundingRate"
            params = {'symbol': binance_symbol, 'startTime': current_ts, 'endTime': end_ts, 'limit': self.MAX_FUNDING_RECORDS}

            try:
                data = await self._make_request(url, params, weight=1)
                if not data:
                    break

                for record in data:
                    # Safe float conversion - handle empty strings and None values
                    funding_rate_val = record.get('fundingRate', 0)
                    funding_rate_float = float(funding_rate_val) if funding_rate_val not in ('', None) else 0.0
                    mark_price_val = record.get('markPrice', 0)
                    mark_price_float = float(mark_price_val) if mark_price_val not in ('', None) else 0.0

                    fr = FundingRate(
                        timestamp=pd.to_datetime(record['fundingTime'], unit='ms', utc=True),
                        symbol=symbol.upper(),
                        funding_rate=funding_rate_float,
                        mark_price=mark_price_float,
                        funding_interval_hours=self.FUNDING_INTERVAL_HOURS
                    )
                    all_data.append({**fr.to_dict(), 'funding_interval_hours': self.FUNDING_INTERVAL_HOURS, 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE})

                symbol_records += len(data)
                self.collection_stats['records_collected'] += len(data)
                current_ts = data[-1]['fundingTime'] + 1
            except Exception as e:
                logger.error(f"Error fetching {binance_symbol} funding: {e}")
                break

        logger.info(f"Completed {symbol}: {symbol_records} records")
        return all_data

    async def fetch_funding_rates(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical funding rates for USDT-margined perpetuals.

        SPEEDUP: SPEEDUP: Now uses parallel processing for multiple symbols.
        Expected speedup: 5-10x for 10+ symbols (limited by rate limits, not by sequential processing)
        """
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        # SPEEDUP: SPEEDUP: Process all symbols in parallel instead of sequentially
        # This allows up to limit_per_host=30 concurrent requests
        # Rate limit (1200/min = 20/sec) will control actual throughput
        tasks = [self._fetch_funding_rate_single(symbol, start_ts, end_ts) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and filter out exceptions
        all_data = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch funding rates for {symbols[i]}: {result}")
            elif isinstance(result, list):
                all_data.extend(result)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        return df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
    
    async def fetch_current_funding_info(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch current funding rate and premium index data."""
        url = f"{self.BASE_URL_FUTURES}/fapi/v1/premiumIndex"
        data = await self._make_request(url, weight=1, use_cache=True)
        
        records = []
        symbol_filter = {self._format_symbol_futures(s) for s in symbols} if symbols else None
        
        for item in data:
            if symbol_filter and item['symbol'] not in symbol_filter:
                continue
            
            funding_rate = float(item['lastFundingRate'])
            records.append({
                'timestamp': pd.to_datetime(item['time'], unit='ms', utc=True),
                'symbol': self._parse_symbol(item['symbol']),
                'mark_price': float(item['markPrice']),
                'index_price': float(item['indexPrice']),
                'last_funding_rate': funding_rate,
                'last_funding_rate_pct': funding_rate * 100,
                'funding_annualized': funding_rate * self.FUNDINGS_PER_DAY * 365,
                'next_funding_time': pd.to_datetime(item['nextFundingTime'], unit='ms', utc=True),
                'venue': self.VENUE, 'venue_type': self.VENUE_TYPE
            })
        
        return pd.DataFrame(records)
    
    async def _fetch_ohlcv_single(self, symbol: str, timeframe: str, start_ts: int, end_ts: int, base_url: str, endpoint: str, format_func, market_type: str) -> List[Dict]:
        """SPEEDUP: SPEEDUP: Helper to fetch OHLCV for a single symbol (for parallel processing)."""
        binance_symbol = format_func(symbol)
        logger.info(f"Fetching Binance {market_type} OHLCV: {binance_symbol} ({timeframe})")

        all_data = []
        current_ts = start_ts
        symbol_records = 0

        while current_ts < end_ts:
            url = f"{base_url}{endpoint}"
            params = {'symbol': binance_symbol, 'interval': timeframe, 'startTime': current_ts, 'endTime': end_ts, 'limit': self.MAX_KLINES_RECORDS}

            try:
                data = await self._make_request(url, params, weight=5)
                if not data:
                    break

                for candle in data:
                    volume = float(candle[5])
                    taker_buy_vol = float(candle[9])
                    all_data.append({
                        'timestamp': pd.to_datetime(candle[0], unit='ms', utc=True),
                        'symbol': symbol.upper(),
                        'open': float(candle[1]), 'high': float(candle[2]),
                        'low': float(candle[3]), 'close': float(candle[4]),
                        'volume': volume, 'volume_quote': float(candle[7]),
                        'trades': int(candle[8]),
                        'taker_buy_volume': taker_buy_vol,
                        'taker_sell_volume': volume - taker_buy_vol,
                        'taker_buy_ratio': taker_buy_vol / volume if volume > 0 else 0.5,
                        'market_type': market_type.upper(),
                        'venue': self.VENUE, 'venue_type': self.VENUE_TYPE
                    })

                symbol_records += len(data)
                self.collection_stats['records_collected'] += len(data)
                current_ts = data[-1][6] + 1
            except Exception as e:
                logger.error(f"Error fetching {binance_symbol} OHLCV: {e}")
                break

        logger.info(f"Completed {symbol} OHLCV: {symbol_records} records")
        return all_data

    async def fetch_ohlcv(self, symbols: List[str], timeframe: str, start_date: str, end_date: str, market_type: str = 'futures') -> pd.DataFrame:
        """
        Fetch OHLCV data for spot or futures markets.

        SPEEDUP: SPEEDUP: Now uses parallel processing for multiple symbols.
        Expected speedup: 5-10x for 10+ symbols
        """
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        if market_type == 'futures':
            base_url, endpoint = self.BASE_URL_FUTURES, '/fapi/v1/klines'
            format_func = self._format_symbol_futures
        else:
            base_url, endpoint = self.BASE_URL_SPOT, '/api/v3/klines'
            format_func = self._format_symbol_spot

        # SPEEDUP: SPEEDUP: Process all symbols in parallel
        tasks = [self._fetch_ohlcv_single(symbol, timeframe, start_ts, end_ts, base_url, endpoint, format_func, market_type) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and filter out exceptions
        all_data = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch OHLCV for {symbols[i]}: {result}")
            elif isinstance(result, list):
                all_data.extend(result)

        if not all_data:
            return pd.DataFrame()
        return pd.DataFrame(all_data).sort_values(['timestamp', 'symbol']).reset_index(drop=True)
    
    async def _fetch_open_interest_single(self, symbol: str, timeframe: str, start_ts: int, end_ts: int) -> List[Dict]:
        """SPEEDUP: Helper to fetch open interest for a single symbol (for parallel processing)."""
        binance_symbol = self._format_symbol_futures(symbol)

        all_data = []
        current_ts = start_ts
        symbol_records = 0

        while current_ts < end_ts:
            url = f"{self.BASE_URL_FUTURES}/futures/data/openInterestHist"
            params = {'symbol': binance_symbol, 'period': timeframe, 'startTime': current_ts, 'endTime': end_ts, 'limit': self.MAX_OI_RECORDS}

            try:
                data = await self._make_request(url, params, weight=5)
                if not data:
                    break

                for record in data:
                    oi = OpenInterestData(
                        timestamp=pd.to_datetime(safe_int(record.get('timestamp', 0)), unit='ms', utc=True),
                        symbol=symbol.upper(),
                        open_interest=safe_float(record.get('sumOpenInterest', 0)),
                        open_interest_value=safe_float(record.get('sumOpenInterestValue', 0))
                    )
                    all_data.append({**oi.to_dict(), 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE})

                symbol_records += len(data)
                self.collection_stats['records_collected'] += len(data)
                current_ts = data[-1]['timestamp'] + 1
            except Exception as e:
                logger.debug(f"Binance OI {binance_symbol}: {e}")
                break

        if symbol_records > 0:
            logger.debug(f"Binance OI {symbol}: {symbol_records} records")
        return all_data

    async def fetch_open_interest(self, symbols: List[str], timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical open interest data.

        SPEEDUP: Now uses parallel processing for multiple symbols.
        Expected speedup: 5-10x for 10+ symbols

        NOTE: Binance openInterestHist endpoint only supports recent data (last 30 days).
        Requesting older data returns 400 errors. We limit to last 30 days regardless of start_date.
        """
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        # Binance openInterestHist endpoint only supports last 30 days
        # Limit start_ts to max 30 days ago, even if user requests older data
        max_lookback_days = 30
        max_start_ts = end_ts - (max_lookback_days * 24 * 60 * 60 * 1000)
        requested_start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        start_ts = max(requested_start_ts, max_start_ts)

        if requested_start_ts < max_start_ts:
            logger.info(f"Binance OI: Requested data from {start_date}, but endpoint only supports last {max_lookback_days} days. Adjusting to recent data only.")

        # SPEEDUP: Process all symbols in parallel
        tasks = [self._fetch_open_interest_single(symbol, timeframe, start_ts, end_ts) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and filter out exceptions
        all_data = []
        symbols_with_data = 0
        symbols_without_data = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.debug(f"Binance OI {symbols[i]}: {result}")
                symbols_without_data += 1
            elif isinstance(result, list):
                if result:
                    all_data.extend(result)
                    symbols_with_data += 1
                else:
                    symbols_without_data += 1

        # Log summary once (not per-symbol spam)
        logger.info(f"Binance OI: {symbols_with_data}/{len(symbols)} symbols returned data")

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data).sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        df['oi_change'] = df.groupby('symbol')['open_interest'].diff()
        df['oi_change_pct'] = df.groupby('symbol')['open_interest'].pct_change() * 100
        return df
    
    async def _fetch_long_short_ratio_single(self, symbol: str, timeframe: str, start_ts: int, end_ts: int, endpoint: str, ratio_type: str) -> List[Dict]:
        """SPEEDUP: Helper to fetch L/S ratio for a single symbol (for parallel processing)."""
        binance_symbol = self._format_symbol_futures(symbol)
        logger.info(f"Fetching Binance {ratio_type} L/S ratio: {binance_symbol}")

        all_data = []
        current_ts = start_ts
        symbol_records = 0

        while current_ts < end_ts:
            url = f"{self.BASE_URL_FUTURES}{endpoint}"
            params = {'symbol': binance_symbol, 'period': timeframe, 'startTime': current_ts, 'endTime': end_ts, 'limit': self.MAX_RATIO_RECORDS}

            try:
                data = await self._make_request(url, params, weight=1)
                if not data:
                    break

                for record in data:
                    ls_ratio = safe_float(record.get('longShortRatio', 0))
                    long_pct = safe_float(record.get('longAccount') or record.get('longPosition', 0)) * 100
                    short_pct = safe_float(record.get('shortAccount') or record.get('shortPosition', 0)) * 100

                    ls_data = LongShortRatio(
                        timestamp=pd.to_datetime(safe_int(record.get('timestamp', 0)), unit='ms', utc=True),
                        symbol=symbol.upper(), long_short_ratio=ls_ratio,
                        long_account_pct=long_pct, short_account_pct=short_pct, ratio_type=ratio_type
                    )
                    all_data.append({**ls_data.to_dict(), 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE})

                symbol_records += len(data)
                self.collection_stats['records_collected'] += len(data)
                current_ts = data[-1]['timestamp'] + 1
            except Exception as e:
                logger.error(f"Error fetching {binance_symbol} L/S ratio: {e}")
                break

        logger.info(f"Completed {symbol} L/S ratio: {symbol_records} records")
        return all_data

    async def fetch_long_short_ratio(self, symbols: List[str], timeframe: str, start_date: str, end_date: str, ratio_type: str = 'accounts') -> pd.DataFrame:
        """
        Fetch long/short ratio history.

        SPEEDUP: Now uses parallel processing for multiple symbols.
        Expected speedup: 5-10x for 10+ symbols
        """
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        endpoint_map = {
            'accounts': '/futures/data/globalLongShortAccountRatio',
            'positions': '/futures/data/topLongShortPositionRatio',
            'top_accounts': '/futures/data/topLongShortAccountRatio',
            'top_positions': '/futures/data/topLongShortPositionRatio'
        }
        endpoint = endpoint_map.get(ratio_type, endpoint_map['accounts'])

        # SPEEDUP: Process all symbols in parallel
        tasks = [self._fetch_long_short_ratio_single(symbol, timeframe, start_ts, end_ts, endpoint, ratio_type) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and filter out exceptions
        all_data = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch L/S ratio for {symbols[i]}: {result}")
            elif isinstance(result, list):
                all_data.extend(result)

        if not all_data:
            return pd.DataFrame()
        return pd.DataFrame(all_data).sort_values(['timestamp', 'symbol']).reset_index(drop=True)
    
    async def _fetch_taker_volume_single(self, symbol: str, timeframe: str, start_ts: int, end_ts: int) -> List[Dict]:
        """SPEEDUP: Helper to fetch taker volume for a single symbol (for parallel processing)."""
        binance_symbol = self._format_symbol_futures(symbol)
        logger.info(f"Fetching Binance taker volume: {binance_symbol}")

        all_data = []
        current_ts = start_ts
        symbol_records = 0

        while current_ts < end_ts:
            url = f"{self.BASE_URL_FUTURES}/futures/data/takerlongshortRatio"
            params = {'symbol': binance_symbol, 'period': timeframe, 'startTime': current_ts, 'endTime': end_ts, 'limit': self.MAX_RATIO_RECORDS}

            try:
                data = await self._make_request(url, params, weight=1)
                if not data:
                    break

                for record in data:
                    taker = TakerVolume(
                        timestamp=pd.to_datetime(safe_int(record.get('timestamp', 0)), unit='ms', utc=True),
                        symbol=symbol.upper(),
                        buy_volume=safe_float(record.get('buyVol', 0)), sell_volume=safe_float(record.get('sellVol', 0)),
                        buy_value=safe_float(record.get('buyVol', 0)), sell_value=safe_float(record.get('sellVol', 0))
                    )
                    all_data.append({**taker.to_dict(), 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE})

                symbol_records += len(data)
                self.collection_stats['records_collected'] += len(data)
                current_ts = data[-1]['timestamp'] + 1
            except Exception as e:
                logger.error(f"Error fetching {binance_symbol} taker volume: {e}")
                break

        logger.info(f"Completed {symbol} taker volume: {symbol_records} records")
        return all_data

    async def fetch_taker_volume(self, symbols: List[str], timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch taker buy/sell volume history.

        SPEEDUP: Now uses parallel processing for multiple symbols.
        Expected speedup: 5-10x for 10+ symbols
        """
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        # SPEEDUP: Process all symbols in parallel
        tasks = [self._fetch_taker_volume_single(symbol, timeframe, start_ts, end_ts) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and filter out exceptions
        all_data = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch taker volume for {symbols[i]}: {result}")
            elif isinstance(result, list):
                all_data.extend(result)

        if not all_data:
            return pd.DataFrame()
        return pd.DataFrame(all_data).sort_values(['timestamp', 'symbol']).reset_index(drop=True)
    
    async def fetch_market_snapshot(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch comprehensive market snapshot."""
        url = f"{self.BASE_URL_FUTURES}/fapi/v1/ticker/24hr"
        ticker_data = await self._make_request(url, weight=5, use_cache=True)
        
        premium_data = await self.fetch_current_funding_info(symbols)
        premium_map = {row['symbol']: row for _, row in premium_data.iterrows()} if not premium_data.empty else {}
        
        records = []
        symbol_filter = {self._format_symbol_futures(s) for s in symbols} if symbols else None
        
        for item in ticker_data:
            if symbol_filter and item['symbol'] not in symbol_filter:
                continue
            
            symbol = self._parse_symbol(item['symbol'])
            premium = premium_map.get(symbol, {})
            
            snapshot = MarketSnapshot(
                timestamp=pd.to_datetime(item['closeTime'], unit='ms', utc=True),
                symbol=symbol,
                mark_price=premium.get('mark_price', float(item['lastPrice'])),
                index_price=premium.get('index_price', 0),
                last_funding_rate=premium.get('last_funding_rate', 0),
                next_funding_time=premium.get('next_funding_time'),
                open_interest=0, open_interest_value=0,
                volume_24h=float(item['volume']),
                volume_quote_24h=float(item['quoteVolume']),
                price_change_24h_pct=float(item['priceChangePercent']),
                high_24h=float(item['highPrice']),
                low_24h=float(item['lowPrice'])
            )
            records.append({**snapshot.to_dict(), 'last_price': float(item['lastPrice']), 'trades_24h': int(item['count']), 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE})
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('volume_quote_24h', ascending=False).reset_index(drop=True)
            df['volume_rank'] = range(1, len(df) + 1)
        return df
    
    async def fetch_comprehensive_data(self, symbols: List[str], start_date: str, end_date: str, timeframe: str = '1h') -> Dict[str, pd.DataFrame]:
        """
        Fetch comprehensive derivatives data.

        SPEEDUP: SPEEDUP: Now fetches all 6 data types in parallel instead of sequentially.
        Expected speedup: 2-4x (limited by slowest data type, not by sum of all)
        """
        logger.info(f"Fetching comprehensive Binance data for {len(symbols)} symbols...")

        # SPEEDUP: SPEEDUP: Fetch all data types in parallel using asyncio.gather
        # This allows multiple HTTP requests to run concurrently
        # BEFORE: Sum of all durations (~120s for 6 data types × 20s each)
        # AFTER: Max of all durations (~30-40s, limited by slowest data type)
        (
            funding_rates,
            ohlcv,
            open_interest,
            long_short_ratio,
            taker_volume,
            snapshot
        ) = await asyncio.gather(
            self.fetch_funding_rates(symbols, start_date, end_date),
            self.fetch_ohlcv(symbols, timeframe, start_date, end_date, 'futures'),
            self.fetch_open_interest(symbols, timeframe, start_date, end_date),
            self.fetch_long_short_ratio(symbols, timeframe, start_date, end_date),
            self.fetch_taker_volume(symbols, timeframe, start_date, end_date),
            self.fetch_market_snapshot(symbols),
            return_exceptions=True # Don't fail all if one fails
        )

        # Build results dict, handling exceptions
        results = {}
        data_names = ['funding_rates', 'ohlcv', 'open_interest', 'long_short_ratio', 'taker_volume', 'snapshot']
        data_values = [funding_rates, ohlcv, open_interest, long_short_ratio, taker_volume, snapshot]

        for name, value in zip(data_names, data_values):
            if isinstance(value, Exception):
                logger.error(f"Failed to fetch {name}: {value}")
                results[name] = pd.DataFrame() # Return empty DataFrame on error
            else:
                results[name] = value

        return results
    
    async def fetch_exchange_info(self) -> Dict:
        """Fetch exchange information."""
        # SPEEDUP: SPEEDUP: Check intelligent cache first (24h TTL for exchange_info)
        cached = self._get_cached('exchange_info', data_type='exchange_info')
        if cached:
            return cached

        url = f"{self.BASE_URL_FUTURES}/fapi/v1/exchangeInfo"
        data = await self._make_request(url, weight=10, use_cache=False) # Don't double-cache

        # Cache with intelligent TTL (24 hours)
        self._set_cached('exchange_info', data)
        return data
    
    async def get_available_symbols(self, min_volume_usd: float = 10_000_000, contract_type: str = 'PERPETUAL') -> List[str]:
        """
        Get available futures symbols filtered by volume and contract type.

        Args:
            min_volume_usd: Minimum 24h volume in USD
            contract_type: 'PERPETUAL', 'CURRENT_QUARTER', 'NEXT_QUARTER', 'CURRENT_MONTH', 'NEXT_MONTH', or 'ALL'

        Returns:
            List of symbol names
        """
        exchange_info = await self.fetch_exchange_info()

        # Filter by contract type
        if contract_type == 'ALL':
            symbols = [s['symbol'].replace('USDT', '').replace('_', '-') for s in exchange_info.get('symbols', [])
                       if s['symbol'].endswith('USDT') and s['status'] == 'TRADING']
        else:
            symbols = [s['symbol'].replace('USDT', '').replace('_', '-') for s in exchange_info.get('symbols', [])
                       if s['symbol'].endswith('USDT') and s['contractType'] == contract_type and s['status'] == 'TRADING']

        # Get volume data
        url = f"{self.BASE_URL_FUTURES}/fapi/v1/ticker/24hr"
        tickers = await self._make_request(url, weight=5, use_cache=True)
        if not tickers:
            return sorted(symbols)

        volume_map = {t['symbol'].replace('USDT', '').replace('_', '-'): float(t['quoteVolume']) for t in tickers}

        filtered = [s for s in symbols if volume_map.get(s, 0) >= min_volume_usd]
        logger.info(f"Found {len(filtered)} {contract_type} symbols with volume >= ${min_volume_usd:,.0f}")
        return sorted(filtered)
    
    async def _fetch_quarterly_futures_single_symbol(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        all_symbols_info: List[Dict]
    ) -> Optional[List[Dict]]:
        """Helper to fetch quarterly futures for a single symbol."""
        try:
            # Find all contract types for this symbol
            base_symbol = self._format_symbol_futures(symbol)

            # Get quarterly contracts
            quarterly_contracts = [
                s for s in all_symbols_info
                if s['symbol'].startswith(base_symbol.replace('USDT', '')) and
                s['contractType'] in ['CURRENT_QUARTER', 'NEXT_QUARTER'] and
                s['status'] == 'TRADING'
            ]

            all_contract_data = []

            for contract_info in quarterly_contracts:
                contract_symbol = contract_info['symbol']
                contract_type = contract_info['contractType']

                logger.info(f"Fetching Binance quarterly futures: {contract_symbol} ({contract_type})")

                # Fetch OHLCV data for this specific contract
                start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
                end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

                current_ts = start_ts
                contract_data = []

                while current_ts < end_ts:
                    url = f"{self.BASE_URL_FUTURES}/fapi/v1/klines"
                    params = {
                        'symbol': contract_symbol,
                        'interval': timeframe,
                        'startTime': current_ts,
                        'endTime': end_ts,
                        'limit': self.MAX_KLINES_RECORDS
                    }

                    try:
                        data = await self._make_request(url, params, weight=5)
                        if not data:
                            break

                        for candle in data:
                            volume = float(candle[5])
                            taker_buy_vol = float(candle[9])
                            contract_data.append({
                                'timestamp': pd.to_datetime(candle[0], unit='ms', utc=True),
                                'symbol': symbol.upper(),
                                'contract_symbol': contract_symbol,
                                'contract_type': contract_type,
                                'open': float(candle[1]),
                                'high': float(candle[2]),
                                'low': float(candle[3]),
                                'close': float(candle[4]),
                                'volume': volume,
                                'volume_quote': float(candle[7]),
                                'trades': int(candle[8]),
                                'taker_buy_volume': taker_buy_vol,
                                'taker_sell_volume': volume - taker_buy_vol,
                                'market_type': 'QUARTERLY',
                                'venue': self.VENUE,
                                'venue_type': self.VENUE_TYPE
                            })

                        self.collection_stats['records_collected'] += len(data)
                        current_ts = data[-1][6] + 1
                    except Exception as e:
                        logger.error(f"Error fetching {contract_symbol} quarterly futures: {e}")
                        break

                all_contract_data.extend(contract_data)
                logger.info(f"Completed {contract_symbol}: {len(contract_data)} records")

            return all_contract_data

        except Exception as e:
            logger.error(f"Error processing quarterly futures for {symbol}: {e}")
            return None

    async def fetch_quarterly_futures(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch quarterly futures contract data for term structure analysis.

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH'])
            timeframe: Timeframe for OHLCV data
            start_date: Start date (YYYY-MM-DD), defaults to last 7 days
            end_date: End date (YYYY-MM-DD), defaults to now

        Returns:
            DataFrame with quarterly futures data including contract type
        """
        if start_date is None:
            start_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')

        exchange_info = await self.fetch_exchange_info()
        all_symbols_info = exchange_info.get('symbols', [])

        # Parallelize symbol processing
        tasks = [
            self._fetch_quarterly_futures_single_symbol(symbol, timeframe, start_date, end_date, all_symbols_info)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and filter out exceptions/None
        all_data = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch quarterly futures for {symbols[i]}: {result}")
            elif result is not None:
                all_data.extend(result)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data).sort_values(['timestamp', 'symbol', 'contract_type']).reset_index(drop=True)
        return df

    async def _fetch_term_structure_single_symbol(
        self,
        symbol: str,
        all_symbols_info: List[Dict],
        ticker_map: Dict[str, Dict]
    ) -> Optional[List[Dict]]:
        """Helper to fetch term structure for a single symbol."""
        try:
            base_symbol = self._format_symbol_futures(symbol)

            # Find all contracts for this symbol (perpetual + quarterly)
            contracts = [
                s for s in all_symbols_info
                if s['symbol'].startswith(base_symbol.replace('USDT', '')) and
                s['contractType'] in ['PERPETUAL', 'CURRENT_QUARTER', 'NEXT_QUARTER'] and
                s['status'] == 'TRADING'
            ]

            if not contracts:
                return None

            # Build term structure
            term_structure = []
            for contract_info in contracts:
                contract_symbol = contract_info['symbol']
                if contract_symbol not in ticker_map:
                    continue

                ticker = ticker_map[contract_symbol]

                term_structure.append({
                    'symbol': symbol.upper(),
                    'contract_symbol': contract_symbol,
                    'contract_type': contract_info['contractType'],
                    'last_price': float(ticker['lastPrice']),
                    'mark_price': float(ticker.get('markPrice', ticker['lastPrice'])),
                    'volume_24h': float(ticker['volume']),
                    'volume_quote_24h': float(ticker['quoteVolume']),
                    'price_change_24h_pct': float(ticker['priceChangePercent']),
                    'timestamp': pd.to_datetime(ticker['closeTime'], unit='ms', utc=True),
                })

            if not term_structure:
                return None

            # Sort by contract type priority
            contract_priority = {'PERPETUAL': 0, 'CURRENT_QUARTER': 1, 'NEXT_QUARTER': 2}
            term_structure.sort(key=lambda x: contract_priority.get(x['contract_type'], 99))

            # Calculate spreads if we have multiple contracts
            if len(term_structure) >= 2:
                perp_price = term_structure[0]['mark_price'] if term_structure[0]['contract_type'] == 'PERPETUAL' else None

                for i, contract in enumerate(term_structure):
                    # Calculate basis to perpetual
                    if perp_price and contract['contract_type'] != 'PERPETUAL':
                        contract['basis_to_perp'] = contract['mark_price'] - perp_price
                        contract['basis_to_perp_pct'] = (contract['basis_to_perp'] / perp_price) * 100
                    else:
                        contract['basis_to_perp'] = 0
                        contract['basis_to_perp_pct'] = 0

                    # Calculate calendar spread to next contract
                    if i < len(term_structure) - 1:
                        next_price = term_structure[i + 1]['mark_price']
                        contract['calendar_spread'] = next_price - contract['mark_price']
                        contract['calendar_spread_pct'] = (contract['calendar_spread'] / contract['mark_price']) * 100
                    else:
                        contract['calendar_spread'] = 0
                        contract['calendar_spread_pct'] = 0

                    contract['venue'] = self.VENUE
                    contract['venue_type'] = self.VENUE_TYPE

            return term_structure

        except Exception as e:
            logger.error(f"Error processing term structure for {symbol}: {e}")
            return None

    async def fetch_term_structure(
        self,
        symbols: List[str],
        date: str = None
    ) -> pd.DataFrame:
        """
        Fetch futures term structure (perpetual + quarterly contracts).

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH'])
            date: Date for term structure (YYYY-MM-DD), defaults to now

        Returns:
            DataFrame with term structure data (basis, calendar spreads)
        """
        if date is None:
            date = datetime.now(timezone.utc).strftime('%Y-%m-%d')

        exchange_info = await self.fetch_exchange_info()
        all_symbols_info = exchange_info.get('symbols', [])

        # Get current prices for all contracts (shared across all symbols)
        url = f"{self.BASE_URL_FUTURES}/fapi/v1/ticker/24hr"
        tickers = await self._make_request(url, weight=5, use_cache=True)
        if not tickers:
            return pd.DataFrame()

        ticker_map = {t['symbol']: t for t in tickers}

        # Parallelize symbol processing
        tasks = [
            self._fetch_term_structure_single_symbol(symbol, all_symbols_info, ticker_map)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and filter out exceptions/None
        all_data = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch term structure for {symbols[i]}: {result}")
            elif result is not None:
                all_data.extend(result)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        logger.info(f"Fetched term structure for {len(symbols)} symbols, {len(df)} contracts total")
        return df

    async def collect_funding_rates(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect funding rates - wraps fetch_funding_rates().

        Standardized method name for collection manager compatibility.
        """
        try:
            # Convert dates to string format if needed
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            return await self.fetch_funding_rates(
                symbols=symbols,
                start_date=start_str,
                end_date=end_str
            )
        except Exception as e:
            logger.error(f"Binance collect_funding_rates error: {e}")
            return pd.DataFrame()

    async def collect_ohlcv(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect OHLCV data - wraps fetch_ohlcv().

        Standardized method name for collection manager compatibility.
        """
        try:
            timeframe = kwargs.get('timeframe', '1h')
            market_type = kwargs.get('market_type', 'futures')

            # Convert dates to string format if needed
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            return await self.fetch_ohlcv(
                symbols=symbols,
                timeframe=timeframe,
                start_date=start_str,
                end_date=end_str,
                market_type=market_type
            )
        except Exception as e:
            logger.error(f"Binance collect_ohlcv error: {e}")
            return pd.DataFrame()

    async def collect_open_interest(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect open interest - wraps fetch_open_interest().

        Standardized method name for collection manager compatibility.
        """
        try:
            timeframe = kwargs.get('timeframe', '1h')

            # Convert dates to string format if needed
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            return await self.fetch_open_interest(
                symbols=symbols,
                timeframe=timeframe,
                start_date=start_str,
                end_date=end_str
            )
        except Exception as e:
            logger.error(f"Binance collect_open_interest error: {e}")
            return pd.DataFrame()

    async def _fetch_trades_single_symbol(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        limit_per_request: int
    ) -> Optional[List[Dict]]:
        """Helper to fetch trades for a single symbol."""
        try:
            binance_symbol = self._format_symbol_futures(symbol)

            # Check if symbol should be skipped entirely
            if binance_symbol in SKIP_SYMBOLS:
                logger.debug(f"Skipping {binance_symbol} (in SKIP_SYMBOLS list)")
                return []

            # Check if symbol was listed after the start date (skip if too early)
            listing_ts = SYMBOL_LISTING_DATES.get(binance_symbol)
            if listing_ts and start_ts < listing_ts:
                logger.info(f"[{binance_symbol}] Adjusting start date from {datetime.fromtimestamp(start_ts/1000).strftime('%Y-%m-%d')} to listing date ~{datetime.fromtimestamp(listing_ts/1000).strftime('%Y-%m-%d')}")
                # Adjust start_ts to listing date if end_ts is after listing
                if end_ts > listing_ts:
                    start_ts = listing_ts
                else:
                    logger.debug(f"Skipping {binance_symbol} - entire range before listing date")
                    return [] # Entire range is before listing

            # Calculate date range duration
            range_days = (end_ts - start_ts) / (1000 * 60 * 60 * 24)

            # For very large date ranges (>30 days), use daily chunks instead of hourly
            # This reduces API calls from 720 per month to 30 per month
            if range_days > 30:
                chunk_ms = 86400000 # 24 hours in ms
                chunk_name = "day"
            else:
                chunk_ms = 3600000 # 1 hour in ms
                chunk_name = "hour"

            logger.info(f"Fetching Binance trades: {binance_symbol} ({int(range_days)} days, {chunk_name}ly chunks)")

            all_data = []
            current_ts = start_ts
            symbol_records = 0
            max_records_per_symbol = 50000 # Safety limit
            max_api_calls = 500 # Safety limit on API calls per symbol
            api_calls = 0

            while current_ts < end_ts and symbol_records < max_records_per_symbol and api_calls < max_api_calls:
                url = f"{self.BASE_URL_FUTURES}/fapi/v1/aggTrades"
                params = {
                    'symbol': binance_symbol,
                    'startTime': current_ts,
                    'endTime': min(current_ts + chunk_ms, end_ts),
                    'limit': min(limit_per_request, 1000)
                }

                try:
                    api_calls += 1
                    data = await self._make_request(url, params, weight=20)
                    if not data:
                        # Move to next chunk if no data
                        current_ts += chunk_ms
                        continue

                    for trade in data:
                        trade_data = AggregateTradeData(
                            timestamp=pd.to_datetime(trade['T'], unit='ms', utc=True),
                            symbol=symbol.upper(),
                            agg_trade_id=int(trade['a']),
                            price=float(trade['p']),
                            quantity=float(trade['q']),
                            first_trade_id=int(trade['f']),
                            last_trade_id=int(trade['l']),
                            is_buyer_maker=trade['m']
                        )
                        all_data.append({
                            **trade_data.to_dict(),
                            'venue': self.VENUE,
                            'venue_type': self.VENUE_TYPE
                        })

                    symbol_records += len(data)
                    self.collection_stats['records_collected'] += len(data)

                    # Advance to next time window
                    if data:
                        last_ts = data[-1]['T']
                        current_ts = last_ts + 1
                    else:
                        current_ts += chunk_ms

                except Exception as e:
                    logger.error(f"Error fetching {binance_symbol} trades: {e}")
                    current_ts += chunk_ms # Skip to next chunk on error
                    continue

            if api_calls >= max_api_calls:
                logger.warning(f"[{binance_symbol}] Hit API call limit ({max_api_calls}), collected {symbol_records} records")
            else:
                logger.info(f"Completed {symbol} trades: {symbol_records} records in {api_calls} API calls")
            return all_data

        except Exception as e:
            logger.error(f"Error processing trades for {symbol}: {e}")
            return None

    async def fetch_trades(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        limit_per_request: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch aggregated trades from Binance Futures.

        Uses /fapi/v1/aggTrades endpoint which returns compressed/aggregate trades.
        Weight: 20 per request.

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH'])
            start_date: Start date string 'YYYY-MM-DD'
            end_date: End date string 'YYYY-MM-DD'
            limit_per_request: Max trades per request (max 1000)

        Returns:
            DataFrame with aggregated trade data
        """
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        # Parallelize symbol processing
        tasks = [
            self._fetch_trades_single_symbol(symbol, start_ts, end_ts, limit_per_request)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and filter out exceptions/None
        all_data = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch trades for {symbols[i]}: {result}")
            elif result is not None:
                all_data.extend(result)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = df.sort_values(['timestamp', 'symbol', 'agg_trade_id']).reset_index(drop=True)
        return df

    async def collect_trades(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect aggregated trades - wraps fetch_trades().

        Standardized method name for collection manager compatibility.
        """
        try:
            # Convert dates to string format if needed
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            limit = kwargs.get('limit_per_request', 1000)

            return await self.fetch_trades(
                symbols=symbols,
                start_date=start_str,
                end_date=end_str,
                limit_per_request=limit
            )
        except Exception as e:
            logger.error(f"Binance collect_trades error: {e}")
            return pd.DataFrame()

    async def _fetch_liquidations_single_symbol(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        limit_per_request: int
    ) -> Optional[List[Dict]]:
        """Helper to fetch liquidations for a single symbol."""
        try:
            binance_symbol = self._format_symbol_futures(symbol)
            logger.info(f"Fetching Binance liquidations: {binance_symbol}")

            # Check if we have API keys - liquidations endpoint requires authentication
            if not self.api_key or not self.api_secret:
                if not hasattr(self, '_liq_auth_logged'):
                    logger.debug("Binance liquidations requires API authentication - skipping")
                    self._liq_auth_logged = True
                return None

            all_data = []
            current_ts = start_ts
            symbol_records = 0
            max_records_per_symbol = 10000 # Safety limit

            while current_ts < end_ts and symbol_records < max_records_per_symbol:
                url = f"{self.BASE_URL_FUTURES}/fapi/v1/forceOrders"
                params = {
                    'symbol': binance_symbol,
                    'startTime': current_ts,
                    'endTime': min(current_ts + 86400000, end_ts), # 24 hour chunks
                    'limit': min(limit_per_request, 1000)
                }

                try:
                    data = await self._make_request(url, params, weight=20)
                    if data is None: # Includes 401/418 responses
                        # Move to next day if no data
                        current_ts += 86400000
                        continue

                    for liq in data:
                        liq_time = int(liq['time'])
                        all_data.append({
                            'timestamp': pd.to_datetime(liq_time, unit='ms', utc=True),
                            'symbol': symbol.upper(),
                            'side': liq['side'].lower(),
                            'order_type': liq['type'],
                            'time_in_force': liq.get('timeInForce', ''),
                            'orig_qty': float(liq['origQty']),
                            'price': float(liq['price']),
                            'avg_price': float(liq['averagePrice']),
                            'status': liq['status'],
                            'executed_qty': float(liq['executedQty']),
                            'liquidation_value': float(liq['averagePrice']) * float(liq['executedQty']),
                            'venue': self.VENUE,
                            'venue_type': self.VENUE_TYPE
                        })

                    symbol_records += len(data)
                    self.collection_stats['records_collected'] += len(data)

                    # Advance to next time window
                    if data:
                        last_ts = max(int(l['time']) for l in data)
                        current_ts = last_ts + 1
                    else:
                        current_ts += 86400000

                except Exception as e:
                    logger.error(f"Error fetching {binance_symbol} liquidations: {e}")
                    current_ts += 86400000 # Skip to next day on error
                    continue

            logger.info(f"Completed {symbol} liquidations: {symbol_records} records")
            return all_data

        except Exception as e:
            logger.error(f"Error processing liquidations for {symbol}: {e}")
            return None

    async def fetch_liquidations(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        limit_per_request: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch forced liquidation orders from Binance Futures.

        NOTE: The /fapi/v1/forceOrders endpoint REQUIRES authentication for historical data.
        Without API keys, this will return empty DataFrame.
        Weight: 20 per request with symbol, 50 without.

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH'])
            start_date: Start date string 'YYYY-MM-DD'
            end_date: End date string 'YYYY-MM-DD'
            limit_per_request: Max records per request (max 1000)

        Returns:
            DataFrame with liquidation order data
        """
        # CRITICAL: Check authentication FIRST before any processing
        # This prevents wasting time iterating through symbols when we can't get data
        if not self.api_key or not self.api_secret:
            if not hasattr(self, '_liq_auth_warn_logged'):
                logger.info("Binance liquidations: Requires API authentication - returning empty (not an error)")
                self._liq_auth_warn_logged = True
            return pd.DataFrame()

        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        # Parallelize symbol processing
        tasks = [
            self._fetch_liquidations_single_symbol(symbol, start_ts, end_ts, limit_per_request)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and filter out exceptions/None
        all_data = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch liquidations for {symbols[i]}: {result}")
            elif result is not None:
                all_data.extend(result)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        return df

    async def collect_liquidations(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect liquidation orders - wraps fetch_liquidations().

        Standardized method name for collection manager compatibility.
        """
        try:
            # Convert dates to string format if needed
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            limit = kwargs.get('limit_per_request', 1000)

            return await self.fetch_liquidations(
                symbols=symbols,
                start_date=start_str,
                end_date=end_str,
                limit_per_request=limit
            )
        except Exception as e:
            logger.error(f"Binance collect_liquidations error: {e}")
            return pd.DataFrame()

    def get_collection_stats(self) -> Dict:
        stats = self.collection_stats.copy()
        # Include circuit breaker stats for visibility
        stats['circuit_breaker'] = self._circuit_breaker.get_stats()
        return stats
    
    def reset_collection_stats(self):
        self.collection_stats = {'records_collected': 0, 'api_calls': 0, 'weight_used': 0, 'cache_hits': 0, 'errors': 0, 'rate_limit_hits': 0}

async def test_binance_collector():
    """Test Binance collector functionality."""
    config = {'rate_limit': 400, 'timeout': 30}
    
    async with BinanceCollector(config) as collector:
        print("=" * 60)
        print("Binance Collector Test")
        print("=" * 60)
        
        symbols = await collector.get_available_symbols(min_volume_usd=100_000_000)
        print(f"\n1. Found {len(symbols)} high-volume symbols: {symbols[:10]}")
        
        funding_info = await collector.fetch_current_funding_info(['BTC', 'ETH'])
        if not funding_info.empty:
            print("\n2. Current funding rates:")
            for _, row in funding_info.iterrows():
                print(f" {row['symbol']}: {row['last_funding_rate_pct']:.4f}%")
        
        snapshot = await collector.fetch_market_snapshot(['BTC', 'ETH'])
        if not snapshot.empty:
            print("\n3. Market snapshot:")
            for _, row in snapshot.iterrows():
                print(f" {row['symbol']}: ${row['mark_price']:,.2f}, basis: {row['basis_pct']:.4f}%")
        
        print(f"\nStats: {collector.get_collection_stats()}")

if __name__ == '__main__':
    asyncio.run(test_binance_collector())