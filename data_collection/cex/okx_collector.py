"""
OKX Data Collector - Tertiary CEX for Cross-Validation

validated collector for OKX spot and derivatives markets.
Third largest crypto exchange with strong Asian market presence.

Supported Data Types:
    - USDT perpetual funding rates (8-hour intervals)
    - Spot and futures OHLCV data
    - Open interest history
    - Mark and index prices
    - Long/short ratio (account-based)
    - Position tiers and margin information
    - Instrument specifications

API Documentation: https://www.okx.com/docs-v5/en/

Rate Limits:
    - Public endpoints: 20 requests per 2 seconds
    - Private endpoints: 60 requests per 2 seconds
    - WebSocket: 480 messages per second

Contract Specifications (USDT-M Perpetuals):
    - Funding interval: 8 hours (00:00, 08:00, 16:00 UTC)
    - Contract value: Varies by asset
    - Leverage: Up to 125x (varies by tier)
    - Settlement: USDT

Data Quality Highlights:
    - Different trading population than Binance
    - Strong Asian liquidity
    - Comprehensive derivatives data
    - Good for cross-exchange validation

Statistical Arbitrage Applications:
    - Cross-exchange funding arbitrage
    - Asian market sentiment analysis
    - Retail positioning signals (L/S ratio)
    - Open interest divergence detection
    - Market microstructure comparison

Version: 2.0.0
"""

import asyncio
import os
import random
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import hmac
import hashlib
import base64
import time

from ..base_collector import BaseCollector, CollectionStats
from ..utils.rate_limiter import get_shared_rate_limiter
from ..utils.retry_handler import RetryHandler, safe_float, safe_int

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================

class InstType(Enum):
    """OKX instrument types."""
    SWAP = 'SWAP'
    FUTURES = 'FUTURES'
    SPOT = 'SPOT'
    OPTION = 'OPTION'
    MARGIN = 'MARGIN'

class TdMode(Enum):
    """Trading modes."""
    CROSS = 'cross'
    ISOLATED = 'isolated'
    CASH = 'cash'

class ContractState(Enum):
    """Contract trading state."""
    LIVE = 'live'
    SUSPEND = 'suspend'
    PREOPEN = 'preopen'
    TEST = 'test'

class FundingTrend(Enum):
    """Funding rate trend classification."""
    HIGHLY_POSITIVE = 'highly_positive'
    POSITIVE = 'positive'
    NEUTRAL = 'neutral'
    NEGATIVE = 'negative'
    HIGHLY_NEGATIVE = 'highly_negative'

class MarketSentiment(Enum):
    """Market sentiment based on positioning."""
    STRONGLY_BULLISH = 'strongly_bullish'
    BULLISH = 'bullish'
    NEUTRAL = 'neutral'
    BEARISH = 'bearish'
    STRONGLY_BEARISH = 'strongly_bearish'

class OITrend(Enum):
    """Open interest trend classification."""
    STRONG_INCREASE = 'strong_increase'
    INCREASE = 'increase'
    STABLE = 'stable'
    DECREASE = 'decrease'
    STRONG_DECREASE = 'strong_decrease'

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class OKXFundingRate:
    """OKX perpetual funding rate data."""
    timestamp: datetime
    symbol: str
    funding_rate: float
    realized_rate: float
    next_funding_time: Optional[datetime]
    funding_interval_hours: int = 8
    
    @property
    def annualized_rate(self) -> float:
        """Annualized funding rate."""
        periods_per_year = 365 * 24 / self.funding_interval_hours
        return self.funding_rate * periods_per_year
    
    @property
    def annualized_rate_pct(self) -> float:
        """Annualized rate as percentage."""
        return self.annualized_rate * 100
    
    @property
    def daily_rate(self) -> float:
        """Daily funding rate (3 payments)."""
        return self.funding_rate * (24 / self.funding_interval_hours)
    
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
    def realized_vs_expected_pct(self) -> float:
        """Realized rate deviation from expected."""
        if self.funding_rate == 0:
            return 0
        return ((self.realized_rate - self.funding_rate) / abs(self.funding_rate)) * 100
    
    @property
    def is_arbitrage_opportunity(self) -> bool:
        """Check if rate presents arbitrage opportunity."""
        return abs(self.funding_rate) > 0.0003
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'funding_rate': self.funding_rate,
            'funding_rate_pct': self.funding_rate * 100,
            'realized_rate': self.realized_rate,
            'realized_rate_pct': self.realized_rate * 100,
            'annualized_rate': self.annualized_rate,
            'annualized_rate_pct': self.annualized_rate_pct,
            'daily_rate': self.daily_rate,
            'next_funding_time': self.next_funding_time,
            'trend': self.trend.value,
            'realized_vs_expected_pct': self.realized_vs_expected_pct,
            'is_arbitrage_opportunity': self.is_arbitrage_opportunity,
        }

@dataclass
class OKXTicker:
    """OKX ticker/market snapshot data."""
    timestamp: datetime
    symbol: str
    last_price: float
    mark_price: float
    index_price: float
    open_24h: float
    high_24h: float
    low_24h: float
    volume_24h: float
    volume_ccy_24h: float
    
    @property
    def basis(self) -> float:
        """Mark price basis to index."""
        return self.mark_price - self.index_price
    
    @property
    def basis_pct(self) -> float:
        """Basis as percentage."""
        return (self.basis / self.index_price * 100) if self.index_price > 0 else 0
    
    @property
    def change_24h(self) -> float:
        """24h price change."""
        return self.last_price - self.open_24h
    
    @property
    def change_24h_pct(self) -> float:
        """24h price change percentage."""
        return (self.change_24h / self.open_24h * 100) if self.open_24h > 0 else 0
    
    @property
    def range_24h_pct(self) -> float:
        """24h range as percentage."""
        return ((self.high_24h - self.low_24h) / self.low_24h * 100) if self.low_24h > 0 else 0
    
    @property
    def price_vs_mark_pct(self) -> float:
        """Last price deviation from mark."""
        return ((self.last_price - self.mark_price) / self.mark_price * 100) if self.mark_price > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'last_price': self.last_price,
            'mark_price': self.mark_price,
            'index_price': self.index_price,
            'basis': self.basis,
            'basis_pct': self.basis_pct,
            'open_24h': self.open_24h,
            'high_24h': self.high_24h,
            'low_24h': self.low_24h,
            'change_24h': self.change_24h,
            'change_24h_pct': self.change_24h_pct,
            'range_24h_pct': self.range_24h_pct,
            'volume_24h': self.volume_24h,
            'volume_ccy_24h': self.volume_ccy_24h,
            'price_vs_mark_pct': self.price_vs_mark_pct,
        }

@dataclass
class OKXLongShortRatio:
    """OKX long/short account ratio data."""
    timestamp: datetime
    symbol: str
    long_short_ratio: float
    
    @property
    def long_account_pct(self) -> float:
        """Long accounts percentage."""
        return (self.long_short_ratio / (1 + self.long_short_ratio)) * 100 if self.long_short_ratio > 0 else 50
    
    @property
    def short_account_pct(self) -> float:
        """Short accounts percentage."""
        return (1 / (1 + self.long_short_ratio)) * 100 if self.long_short_ratio > 0 else 50
    
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
        """Position imbalance percentage."""
        return self.long_account_pct - self.short_account_pct
    
    @property
    def is_extreme(self) -> bool:
        """Check for extreme positioning (potential reversal)."""
        return self.long_short_ratio > 2.0 or self.long_short_ratio < 0.5
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'long_short_ratio': self.long_short_ratio,
            'long_account_pct': self.long_account_pct,
            'short_account_pct': self.short_account_pct,
            'sentiment': self.sentiment.value,
            'imbalance_pct': self.imbalance_pct,
            'is_extreme': self.is_extreme,
        }

@dataclass
class OKXOpenInterest:
    """OKX open interest data."""
    timestamp: datetime
    symbol: str
    open_interest: float
    open_interest_usd: float
    
    @property
    def average_position_size(self) -> float:
        """Average position size in USD."""
        return self.open_interest_usd / self.open_interest if self.open_interest > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'open_interest': self.open_interest,
            'open_interest_usd': self.open_interest_usd,
            'average_position_size': self.average_position_size,
        }

@dataclass
class OKXInstrument:
    """OKX instrument specification."""
    inst_id: str
    inst_type: str
    base_ccy: str
    quote_ccy: str
    settle_ccy: str
    ct_val: float
    min_sz: float
    lot_sz: float
    tick_sz: float
    state: str
    
    @property
    def is_tradeable(self) -> bool:
        """Check if instrument is tradeable."""
        return self.state == 'live'
    
    @property
    def is_usdt_settled(self) -> bool:
        """Check if USDT settled."""
        return self.settle_ccy == 'USDT'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'inst_id': self.inst_id,
            'inst_type': self.inst_type,
            'base_ccy': self.base_ccy,
            'quote_ccy': self.quote_ccy,
            'settle_ccy': self.settle_ccy,
            'ct_val': self.ct_val,
            'min_sz': self.min_sz,
            'lot_sz': self.lot_sz,
            'tick_sz': self.tick_sz,
            'state': self.state,
            'is_tradeable': self.is_tradeable,
            'is_usdt_settled': self.is_usdt_settled,
        }

# =============================================================================
# Collector Class
# =============================================================================

class OKXCollector(BaseCollector):
    """
    OKX data collector for spot and derivatives markets.

    validated implementation for cross-validation with other exchanges.

    Features:
    - USDT-M perpetual funding rates
    - Spot and futures OHLCV
    - Open interest history
    - Mark and index prices
    - Long/short ratio (account-based)
    - Instrument specifications

    Attributes:
        VENUE: Exchange identifier ('okx')
        VENUE_TYPE: Exchange type ('CEX')
        BASE_URL: API endpoint
        FUNDING_INTERVAL_HOURS: Funding payment interval (8)

    Example:
        >>> config = {'rate_limit': 120}
        >>> async with OKXCollector(config) as collector:
        ... funding = await collector.fetch_funding_rates(['BTC', 'ETH'], '2024-01-01', '2024-01-31')
        ... snapshot = await collector.fetch_current_funding_rate(['BTC', 'ETH'])
    """

    VENUE = 'okx'
    VENUE_TYPE = 'CEX'
    BASE_URL = 'https://www.okx.com'

    # Collection manager compatibility attributes
    supported_data_types = ['funding_rates', 'ohlcv', 'open_interest', 'trades', 'liquidations']
    venue = 'okx'
    requires_auth = False # Public endpoints available for all data types
    
    FUNDING_INTERVAL_HOURS = 8
    FUNDINGS_PER_DAY = 3
    MAX_FUNDING_RECORDS = 100
    MAX_KLINES_RECORDS = 300
    
    VALID_BARS = {'1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '6H', '12H', '1D', '1W', '1M'}

    # Symbols truly unavailable on OKX (delisted or never listed)
    # Minimal list - let the API return errors for borderline cases
    UNSUPPORTED_SWAP_SYMBOLS = {
        'MATIC', # Migrated to POL on OKX
        'BTTC',  # Not listed on OKX
        'WIN',   # Not listed on OKX
        'BTT',   # Not listed on OKX
        'NFT',   # Not listed on OKX
    }

    # Symbols that need the 1000 prefix on OKX for perpetual swaps
    THOUSANDX_SYMBOLS = {
        'FLOKI', 'BONK', 'SHIB', 'PEPE', 'LUNC',
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize OKX collector."""
        super().__init__(config)

        # Load API keys from config or environment variables
        self.api_key = config.get('api_key') or config.get('okx_api_key') or os.getenv('OKX_API_KEY', '')
        self.api_secret = config.get('api_secret') or config.get('okx_api_secret') or os.getenv('OKX_SECRET_KEY', '')
        self.passphrase = config.get('passphrase') or config.get('okx_passphrase') or os.getenv('OKX_PASSPHRASE', '')

        # OKX rate limit: 20 req/2s = 10 req/s = 600 req/min
        # CONSERVATIVE: 20% utilization to avoid rate limiting
        rate_limit = config.get('rate_limit', 120) # 2 req/s (20% utilization)

        # Use shared rate limiter to avoid re-initialization overhead
        self.rate_limiter = get_shared_rate_limiter('okx', rate=rate_limit, per=60.0, burst=4) # Small burst (kept at 4)
        self.retry_handler = RetryHandler(max_retries=config.get('max_retries', 5), base_delay=3.0, max_delay=90.0)

        # Semaphore to limit concurrent requests
        self._concurrency_limit = asyncio.Semaphore(5) # Max 5 concurrent

        self.timeout = aiohttp.ClientTimeout(total=config.get('timeout', 30))
        self.session: Optional[aiohttp.ClientSession] = None

        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl = config.get('cache_ttl', 60)

        # Semaphore to limit concurrent HTTP requests (prevents rate limit exhaustion)
        self._request_sem = asyncio.Semaphore(2)

        self.collection_stats = {'records_collected': 0, 'api_calls': 0, 'errors': 0, 'cache_hits': 0}
        logger.info(f"Initialized OKX collector (rate_limit={rate_limit}/min, burst=4, max_concurrent=5, CONSERVATIVE)")
    
    async def __aenter__(self):
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            # SPEEDUP: Aggressive connection pooling for high-throughput collection
            # OKX rate limit: 20/2sec = 600/min, higher pool supports concurrent symbols
            connector = aiohttp.TCPConnector(
                limit=80, # Total connection pool size (was 20)
                limit_per_host=25, # Per-host connections (was 10)
                ttl_dns_cache=300, # DNS cache TTL in seconds
                force_close=False, # Keep-alive connections
                enable_cleanup_closed=True
            )
            self.session = aiohttp.ClientSession(timeout=self.timeout, connector=connector)
        return self.session
    
    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
        self._cache.clear()
        logger.info(f"OKX collector closed. Stats: {self.collection_stats}")
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            timestamp, value = self._cache[key]
            if (datetime.utcnow() - timestamp).total_seconds() < self._cache_ttl:
                self.collection_stats['cache_hits'] += 1
                return value
            del self._cache[key]
        return None
    
    def _set_cached(self, key: str, value: Any):
        """Set cache value."""
        self._cache[key] = (datetime.utcnow(), value)
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = '') -> str:
        """Generate HMAC-SHA256 signature."""
        message = timestamp + method + request_path + body
        signature = hmac.new(self.api_secret.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).digest()
        return base64.b64encode(signature).decode('utf-8')
    
    def _get_headers(self, method: str, request_path: str, body: str = '') -> Dict[str, str]:
        """Generate headers for authenticated requests."""
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        headers = {'Content-Type': 'application/json'}
        
        if self.api_key and self.api_secret:
            headers.update({
                'OK-ACCESS-KEY': self.api_key,
                'OK-ACCESS-SIGN': self._generate_signature(timestamp, method, request_path, body),
                'OK-ACCESS-TIMESTAMP': timestamp,
                'OK-ACCESS-PASSPHRASE': self.passphrase
            })
        
        return headers
    
    def _format_symbol(self, symbol: str, inst_type: str = 'SWAP') -> str:
        """Convert to OKX symbol format."""
        symbol = symbol.upper().replace('/USDT', '').replace('-USDT', '').replace(':USDT', '')
        symbol = symbol.replace('USDT', '').replace('PERP', '').replace('/USD', '')

        # Handle 1000x symbols on OKX (e.g., FLOKI -> 1000FLOKI)
        if inst_type == 'SWAP' and symbol in self.THOUSANDX_SYMBOLS:
            symbol = f"1000{symbol}"

        if inst_type == 'SWAP':
            return f"{symbol}-USDT-SWAP"
        elif inst_type == 'SPOT':
            return f"{symbol}-USDT"
        else:
            return symbol
    
    def _parse_symbol(self, okx_symbol: str) -> str:
        """Parse OKX symbol to standard format."""
        parts = okx_symbol.split('-')
        return parts[0] if parts else okx_symbol
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None, authenticated: bool = False, use_cache: bool = False) -> Optional[Any]:
        """Make rate-limited request with retry."""
        async with self._request_sem:
            if use_cache:
                cache_key = f"{endpoint}_{hash(frozenset((params or {}).items()))}"
                cached = self._get_cached(cache_key)
                if cached is not None:
                    return cached

            session = await self._get_session()
            url = f"{self.BASE_URL}{endpoint}"

            query_string = ''
            if params:
                query_string = '?' + '&'.join(f"{k}={v}" for k, v in params.items() if v is not None)
                url += query_string

            headers = {}
            if authenticated:
                request_path = endpoint + query_string
                headers = self._get_headers('GET', request_path)

            async def _request():
                acquire_result = await self.rate_limiter.acquire(timeout=120.0)
                if hasattr(acquire_result, 'acquired') and not acquire_result.acquired:
                    raise TimeoutError("Rate limiter timeout after 120s")
                # BALANCED: Moderate jitter for speed without clustering
                # Base 200ms + random 0-400ms jitter = 200-600ms distribution
                await asyncio.sleep(0.2 + random.uniform(0, 0.4))

                async with session.get(url, headers=headers) as response:
                    if response.status == 429:
                        logger.warning("OKX rate limited - backing off")
                        await asyncio.sleep(5) # Longer backoff on rate limit
                        raise aiohttp.ClientResponseError(response.request_info, response.history, status=429)

                    data = await response.json()

                    if data.get('code') != '0':
                        error_msg = data.get('msg', 'Unknown error')
                        error_code = data.get('code', '0')

                        # Handle known non-critical errors gracefully
                        if error_code == '51001' or "Instrument ID doesn't exist" in error_msg:
                            logger.debug(f"OKX instrument not found: {error_msg}")
                            return None # Return None instead of raising
                        elif error_code == '51014' or "Index doesn't exist" in error_msg:
                            logger.debug(f"OKX index doesn't exist (symbol not available): {error_msg}")
                            return None # Non-retryable validation error
                        elif error_code == '51000' or 'parameter' in error_msg.lower():
                            logger.debug(f"OKX parameter error: {error_msg}")
                            return None
                        elif error_code == '50011': # Rate limit
                            logger.warning(f"OKX rate limited: {error_msg}")
                            await asyncio.sleep(2)
                            raise ValueError(f"OKX rate limit: {error_msg}") # Allow retry
                        elif error_code == '50013': # System busy
                            logger.debug(f"OKX system busy: {error_msg}")
                            raise ValueError(f"OKX busy: {error_msg}") # Allow retry
                        else:
                            logger.warning(f"OKX API error ({error_code}): {error_msg}")
                            raise ValueError(f"OKX API error: {error_msg}")

                    self.collection_stats['api_calls'] += 1
                    return data.get('data', [])

            try:
                result = await self.retry_handler.execute(_request)
                if use_cache and result is not None:
                    self._set_cached(cache_key, result)
                return result
            except Exception as e:
                logger.error(f"Request failed: {endpoint} - {e}")
                self.collection_stats['errors'] += 1
                return None
    
    async def _fetch_funding_rate_single(
        self, symbol: str, start_ts: int, end_ts: int
    ) -> List[Dict]:
        """Fetch funding rates for a single symbol (internal helper for parallelization)."""
        async with self._concurrency_limit:
            # Skip symbols that don't have perpetual swaps on OKX
            base_symbol = symbol.upper().replace('USDT', '').replace('PERP', '').replace('/USD', '')
            if base_symbol in self.UNSUPPORTED_SWAP_SYMBOLS:
                logger.debug(f"Skipping {symbol} - no OKX perpetual swap available")
                return []

            okx_symbol = self._format_symbol(symbol, 'SWAP')
            logger.info(f"Fetching OKX funding rates for {okx_symbol}")

            symbol_data = []
            pagination_ts = None # Start with no pagination to get most recent

            while True:
                # Build params - start without 'after' to get most recent data
                params = {'instId': okx_symbol, 'limit': str(self.MAX_FUNDING_RECORDS)}
                if pagination_ts is not None:
                    params['after'] = str(pagination_ts)

                data = await self._make_request('/api/v5/public/funding-rate-history', params)

                if not data:
                    break

                oldest_ts = None

                for record in data:
                    record_ts = int(record['fundingTime'])

                    # Track oldest for pagination
                    if oldest_ts is None or record_ts < oldest_ts:
                        oldest_ts = record_ts

                    # Filter records within the requested date range
                    if record_ts < start_ts or record_ts > end_ts:
                        continue

                    funding_rate = safe_float(record.get('fundingRate', 0))
                    fr = OKXFundingRate(
                        timestamp=pd.to_datetime(record_ts, unit='ms', utc=True),
                        symbol=symbol.upper(),
                        funding_rate=funding_rate,
                        realized_rate=safe_float(record.get('realizedRate', funding_rate)),
                        next_funding_time=pd.to_datetime(safe_int(record.get('nextFundingTime', record_ts + 28800000)), unit='ms', utc=True),
                        funding_interval_hours=self.FUNDING_INTERVAL_HOURS
                    )

                    symbol_data.append({**fr.to_dict(), 'funding_interval_hours': self.FUNDING_INTERVAL_HOURS, 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE})

                # Check if we should continue paginating
                if oldest_ts is None or oldest_ts <= start_ts:
                    break # Reached start date

                if len(data) < self.MAX_FUNDING_RECORDS:
                    break # No more data available

                if pagination_ts is not None and oldest_ts >= pagination_ts:
                    break # No progress - avoid infinite loop

                pagination_ts = oldest_ts

            logger.info(f"Completed {symbol}: {len(symbol_data)} records")
            return symbol_data

    async def fetch_funding_rates(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical funding rates from OKX.

        CRITICAL: OKX API has limited historical data (~3 months window).
        Start from most recent data and paginate backward until reaching start_date
        or hitting the API's historical limit.

        Args:
            symbols: List of symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with funding rate data

        Note:
            OKX pagination: 'after' returns records older than the given timestamp.
            Always start without pagination to get most recent data first.
        """
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        # SPEEDUP: Process all symbols in parallel instead of sequentially
        # Rate limiting will control actual throughput
        tasks = [self._fetch_funding_rate_single(symbol, start_ts, end_ts) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine all results
        all_data = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error fetching funding rates: {result}")
                continue
            if result:
                all_data.extend(result)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        df = df.drop_duplicates(subset=['timestamp', 'symbol'], keep='first')
        self.collection_stats['records_collected'] += len(df)
        return df
    
    async def _fetch_current_funding_rate_single(self, symbol: str) -> Optional[List[Dict]]:
        """Helper to fetch current funding rate for a single symbol."""
        async with self._concurrency_limit:
            try:
                # Skip unsupported symbols
                base_symbol = symbol.upper().replace('USDT', '').replace('PERP', '').replace('/USD', '')
                if base_symbol in self.UNSUPPORTED_SWAP_SYMBOLS:
                    return None

                okx_symbol = self._format_symbol(symbol, 'SWAP')
                params = {'instId': okx_symbol}
                data = await self._make_request('/api/v5/public/funding-rate', params, use_cache=True)

                if not data:
                    return None

                records = []
                for item in data:
                    if not item['instId'].endswith('-USDT-SWAP'):
                        continue

                    funding_rate = float(item.get('fundingRate', 0) or 0)
                    fr = OKXFundingRate(
                        timestamp=pd.to_datetime(int(item.get('fundingTime', 0)), unit='ms', utc=True) if item.get('fundingTime') else datetime.utcnow().replace(tzinfo=timezone.utc),
                        symbol=self._parse_symbol(item['instId']),
                        funding_rate=funding_rate,
                        realized_rate=funding_rate,
                        next_funding_time=pd.to_datetime(int(item.get('nextFundingTime', 0)), unit='ms', utc=True) if item.get('nextFundingTime') else None
                    )

                    records.append({**fr.to_dict(), 'next_funding_rate': float(item.get('nextFundingRate', 0) or 0), 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE})

                return records
            except Exception as e:
                logger.error(f"Error fetching current funding rate for {symbol}: {e}")
                return None

    async def fetch_current_funding_rate(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch current funding rates.

        Note: OKX requires instId parameter for funding-rate endpoint.
        We fetch individually for each symbol.
        """
        # If no symbols provided, use common ones
        if symbols is None:
            symbols = ['BTC', 'ETH', 'SOL']

        tasks = [self._fetch_current_funding_rate_single(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine all results
        all_records = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in fetch_current_funding_rate: {result}")
                continue
            if result:
                all_records.extend(result)

        return pd.DataFrame(all_records)
    
    async def _fetch_ohlcv_single(self, symbol: str, okx_bar: str, start_ts: int, end_ts: int, inst_type: str, contract_type: str) -> Optional[List[Dict]]:
        """Helper to fetch OHLCV for a single symbol."""
        async with self._concurrency_limit:
            try:
                # Skip symbols that don't have perpetual swaps on OKX (for SWAP inst_type only)
                if inst_type == 'SWAP':
                    base_symbol = symbol.upper().replace('USDT', '').replace('PERP', '').replace('/USD', '')
                    if base_symbol in self.UNSUPPORTED_SWAP_SYMBOLS:
                        logger.debug(f"Skipping {symbol} OHLCV - no OKX perpetual swap available")
                        return None

                okx_symbol = self._format_symbol(symbol, inst_type)
                logger.info(f"Fetching OKX {contract_type} OHLCV for {okx_symbol} ({okx_bar})")

                symbol_data = []
                # Use 'after' to get records older than end_ts, then paginate backwards
                current_ts = end_ts
                symbol_records = 0

                while current_ts > start_ts:
                    # 'after' returns records with timestamp < current_ts
                    params = {'instId': okx_symbol, 'bar': okx_bar, 'after': str(current_ts), 'limit': str(self.MAX_KLINES_RECORDS)}

                    data = await self._make_request('/api/v5/market/history-candles', params)

                    if not data:
                        break

                    for candle in data:
                        record_ts = int(candle[0])
                        # Filter records within the requested date range
                        if record_ts < start_ts:
                            continue

                        symbol_data.append({
                            'timestamp': pd.to_datetime(record_ts, unit='ms', utc=True),
                            'symbol': symbol.upper(),
                            'open': float(candle[1]),
                            'high': float(candle[2]),
                            'low': float(candle[3]),
                            'close': float(candle[4]),
                            'volume': float(candle[5]),
                            'volume_ccy': float(candle[6]) if candle[6] else 0,
                            'volume_quote': float(candle[7]) if len(candle) > 7 and candle[7] else 0,
                            'contract_type': contract_type.upper(),
                            'venue': self.VENUE,
                            'venue_type': self.VENUE_TYPE
                        })
                        symbol_records += 1

                    oldest_ts = min(int(c[0]) for c in data)
                    if oldest_ts >= current_ts:
                        break
                    current_ts = oldest_ts

                logger.info(f"Completed {symbol} OHLCV: {symbol_records} records")
                return symbol_data
            except Exception as e:
                logger.error(f"Error fetching OHLCV for {symbol}: {e}")
                return None

    async def fetch_ohlcv(self, symbols: List[str], timeframe: str, start_date: str, end_date: str, contract_type: str = 'perpetual') -> pd.DataFrame:
        """
        Fetch OHLCV data.

        Args:
            symbols: List of symbols
            timeframe: Bar interval ('1H', '4H', '1D')
            start_date: Start date
            end_date: End date
            contract_type: 'spot' or 'perpetual'

        Returns:
            DataFrame with OHLCV data

        Note:
            OKX pagination: 'after' returns records older than the given timestamp.
        """
        tf_map = {'1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1H', '4h': '4H', '1d': '1D'}
        okx_bar = tf_map.get(timeframe.lower(), '1H')

        inst_type = 'SPOT' if contract_type == 'spot' else 'SWAP'

        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        tasks = [self._fetch_ohlcv_single(symbol, okx_bar, start_ts, end_ts, inst_type, contract_type) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine all results
        all_data = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in fetch_ohlcv: {result}")
                continue
            if result:
                all_data.extend(result)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        df = df.drop_duplicates(subset=['timestamp', 'symbol'], keep='first')
        df['volume_usd'] = df['volume_quote']
        self.collection_stats['records_collected'] += len(df)
        return df
    
    async def _fetch_open_interest_single(self, symbol: str, start_ts: int, end_ts: int) -> Optional[List[Dict]]:
        """Helper to fetch open interest for a single symbol."""
        async with self._concurrency_limit:
            try:
                okx_symbol = self._format_symbol(symbol, 'SWAP')
                logger.info(f"Fetching OKX open interest for {okx_symbol}")

                symbol_data = []
                current_ts = end_ts
                symbol_records = 0

                while current_ts > start_ts:
                    # 'after' returns records with timestamp < current_ts
                    params = {'instId': okx_symbol, 'period': '1H', 'after': str(current_ts), 'limit': '100'}

                    data = await self._make_request('/api/v5/rubik/stat/contracts/open-interest-history', params)

                    if not data:
                        break

                    for record in data:
                        record_ts = int(record[0])
                        if record_ts < start_ts:
                            continue

                        oi = OKXOpenInterest(
                            timestamp=pd.to_datetime(record_ts, unit='ms', utc=True),
                            symbol=symbol.upper(),
                            open_interest=safe_float(record[1] if len(record) > 1 else 0),
                            open_interest_usd=safe_float(record[2] if len(record) > 2 else 0)
                        )

                        symbol_data.append({**oi.to_dict(), 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE})
                        symbol_records += 1

                    oldest_ts = min(int(r[0]) for r in data)
                    if oldest_ts >= current_ts:
                        break
                    current_ts = oldest_ts

                logger.info(f"Completed {symbol} OI: {symbol_records} records")
                return symbol_data
            except Exception as e:
                logger.error(f"Error fetching open interest for {symbol}: {e}")
                return None

    async def fetch_open_interest(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch open interest history.

        Note:
            OKX pagination: 'after' returns records older than the given timestamp.
        """
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        tasks = [self._fetch_open_interest_single(symbol, start_ts, end_ts) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine all results
        all_data = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in fetch_open_interest: {result}")
                continue
            if result:
                all_data.extend(result)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        df = df.drop_duplicates(subset=['timestamp', 'symbol'], keep='first')
        df['oi_change'] = df.groupby('symbol')['open_interest'].diff()
        df['oi_change_pct'] = df.groupby('symbol')['open_interest'].pct_change() * 100
        self.collection_stats['records_collected'] += len(df)
        return df
    
    async def _fetch_long_short_ratio_single(self, symbol: str, start_ts: int, end_ts: int) -> Optional[List[Dict]]:
        """Helper to fetch long/short ratio for a single symbol."""
        async with self._concurrency_limit:
            try:
                logger.info(f"Fetching OKX L/S ratio for {symbol}")

                params = {'ccy': symbol.upper(), 'period': '1H', 'begin': str(start_ts), 'end': str(end_ts)}

                data = await self._make_request('/api/v5/rubik/stat/contracts/long-short-account-ratio', params)

                if not data:
                    return None

                symbol_data = []
                for record in data:
                    ts = safe_int(record[0] if len(record) > 0 else 0)
                    ratio = safe_float(record[1] if len(record) > 1 else 0)

                    ls = OKXLongShortRatio(
                        timestamp=pd.to_datetime(ts, unit='ms', utc=True),
                        symbol=symbol.upper(),
                        long_short_ratio=ratio
                    )

                    symbol_data.append({**ls.to_dict(), 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE})

                logger.info(f"Completed {symbol} L/S ratio: {len(data)} records")
                return symbol_data
            except Exception as e:
                logger.error(f"Error fetching L/S ratio for {symbol}: {e}")
                return None

    async def fetch_long_short_ratio(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch long/short account ratio."""
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        tasks = [self._fetch_long_short_ratio_single(symbol, start_ts, end_ts) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine all results
        all_data = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in fetch_long_short_ratio: {result}")
                continue
            if result:
                all_data.extend(result)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        self.collection_stats['records_collected'] += len(df)
        return df
    
    async def fetch_mark_price(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch current mark prices."""
        params = {'instType': 'SWAP'}
        data = await self._make_request('/api/v5/public/mark-price', params, use_cache=True)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        symbol_set = {self._format_symbol(s, 'SWAP') for s in symbols}
        
        for item in data:
            if item['instId'] not in symbol_set:
                continue
            
            records.append({
                'timestamp': pd.to_datetime(int(item['ts']), unit='ms', utc=True),
                'symbol': self._parse_symbol(item['instId']),
                'mark_price': float(item['markPx']),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE
            })
        
        return pd.DataFrame(records)
    
    async def fetch_instruments(self, inst_type: str = 'SWAP') -> pd.DataFrame:
        """Fetch available trading instruments."""
        params = {'instType': inst_type}
        data = await self._make_request('/api/v5/public/instruments', params, use_cache=True)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        for inst in data:
            i = OKXInstrument(
                inst_id=inst['instId'],
                inst_type=inst_type,
                base_ccy=inst.get('baseCcy', ''),
                quote_ccy=inst.get('quoteCcy', 'USDT'),
                settle_ccy=inst.get('settleCcy', 'USDT'),
                ct_val=float(inst.get('ctVal', 1)),
                min_sz=float(inst.get('minSz', 0)),
                lot_sz=float(inst.get('lotSz', 0)),
                tick_sz=float(inst.get('tickSz', 0)),
                state=inst.get('state', 'live')
            )
            records.append({**i.to_dict(), 'venue': self.VENUE})
        
        return pd.DataFrame(records)
    
    async def get_available_symbols(self, min_volume_usd: float = 10_000_000) -> List[str]:
        """Get available symbols filtered by volume."""
        instruments = await self.fetch_instruments('SWAP')
        
        if instruments.empty:
            return []
        
        usdt_perps = instruments[instruments['is_usdt_settled'] & instruments['is_tradeable']]['inst_id'].tolist()
        
        params = {'instType': 'SWAP'}
        tickers = await self._make_request('/api/v5/market/tickers', params, use_cache=True)
        
        if not tickers:
            return sorted([self._parse_symbol(s) for s in usdt_perps])
        
        volume_map = {t['instId']: float(t.get('volCcy24h', 0) or 0) for t in tickers}
        
        filtered = []
        for inst_id in usdt_perps:
            if volume_map.get(inst_id, 0) >= min_volume_usd:
                filtered.append(self._parse_symbol(inst_id))
        
        logger.info(f"Found {len(filtered)} OKX symbols with volume >= ${min_volume_usd:,.0f}")
        return sorted(filtered)
    
    async def fetch_comprehensive_data(self, symbols: List[str], start_date: str, end_date: str, timeframe: str = '1h') -> Dict[str, pd.DataFrame]:
        """Fetch comprehensive data for multiple symbols."""
        results = {}
        
        logger.info(f"Fetching comprehensive OKX data for {len(symbols)} symbols")
        results['funding_rates'] = await self.fetch_funding_rates(symbols, start_date, end_date)
        results['ohlcv'] = await self.fetch_ohlcv(symbols, timeframe, start_date, end_date)
        results['open_interest'] = await self.fetch_open_interest(symbols, start_date, end_date)
        results['long_short_ratio'] = await self.fetch_long_short_ratio(symbols, start_date, end_date)
        results['snapshot'] = await self.fetch_current_funding_rate(symbols)
        
        return results

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
            logger.error(f"OKX collect_funding_rates error: {e}")
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
            contract_type = kwargs.get('contract_type', 'perpetual')

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
                contract_type=contract_type
            )
        except Exception as e:
            logger.error(f"OKX collect_ohlcv error: {e}")
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
                start_date=start_str,
                end_date=end_str
            )
        except Exception as e:
            logger.error(f"OKX collect_open_interest error: {e}")
            return pd.DataFrame()

    async def _fetch_trades_single(self, symbol: str, limit_per_request: int) -> Optional[List[Dict]]:
        """Helper to fetch trades for a single symbol."""
        async with self._concurrency_limit:
            try:
                # Skip unsupported symbols
                base_symbol = symbol.upper().replace('USDT', '').replace('PERP', '').replace('/USD', '')
                if base_symbol in self.UNSUPPORTED_SWAP_SYMBOLS:
                    logger.debug(f"Skipping {symbol} trades - no OKX perpetual swap available")
                    return None

                okx_symbol = self._format_symbol(symbol, 'SWAP')
                logger.info(f"Fetching OKX trades: {okx_symbol}")

                params = {
                    'instId': okx_symbol,
                    'limit': str(min(limit_per_request, 500))
                }

                data = await self._make_request('/api/v5/market/trades', params)
                if not data:
                    return None

                symbol_data = []
                for trade in data:
                    trade_time = int(trade['ts'])
                    symbol_data.append({
                        'timestamp': pd.to_datetime(trade_time, unit='ms', utc=True),
                        'symbol': symbol.upper(),
                        'trade_id': trade.get('tradeId', ''),
                        'price': float(trade['px']),
                        'size': float(trade['sz']),
                        'side': trade['side'].lower(),
                        'trade_value': float(trade['px']) * float(trade['sz']),
                        'venue': self.VENUE,
                        'venue_type': self.VENUE_TYPE
                    })

                logger.info(f"Completed {symbol} trades: {len(data)} records")
                return symbol_data
            except Exception as e:
                logger.error(f"Error fetching trades for {symbol}: {e}")
                return None

    async def fetch_trades(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        limit_per_request: int = 100
    ) -> pd.DataFrame:
        """
        Fetch recent trades from OKX.

        Uses /api/v5/market/trades endpoint (public, no auth required).

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH'])
            start_date: Start date string 'YYYY-MM-DD' (for reference)
            end_date: End date string 'YYYY-MM-DD'
            limit_per_request: Max trades per request (max 500)

        Returns:
            DataFrame with recent trade data
        """
        tasks = [self._fetch_trades_single(symbol, limit_per_request) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine all results
        all_data = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in fetch_trades: {result}")
                continue
            if result:
                all_data.extend(result)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        self.collection_stats['records_collected'] += len(df)
        return df

    async def collect_trades(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect trades - wraps fetch_trades().

        Standardized method name for collection manager compatibility.
        """
        try:
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            limit = kwargs.get('limit_per_request', 100)

            return await self.fetch_trades(
                symbols=symbols,
                start_date=start_str,
                end_date=end_str,
                limit_per_request=limit
            )
        except Exception as e:
            logger.error(f"OKX collect_trades error: {e}")
            return pd.DataFrame()

    async def _fetch_liquidations_single(self, symbol: str, limit_per_request: int) -> Optional[List[Dict]]:
        """Helper to fetch liquidations for a single symbol."""
        async with self._concurrency_limit:
            try:
                # Skip unsupported symbols
                base_symbol = symbol.upper().replace('USDT', '').replace('PERP', '').replace('/USD', '')
                if base_symbol in self.UNSUPPORTED_SWAP_SYMBOLS:
                    logger.debug(f"Skipping {symbol} liquidations - no OKX perpetual swap available")
                    return None

                # OKX requires 'uly' (underlying) for SWAP liquidations
                # Format: BTC-USDT (underlying), not BTC-USDT-SWAP (instId)
                underlying = f"{base_symbol}-USDT"
                logger.info(f"Fetching OKX liquidations: {underlying}")

                params = {
                    'instType': 'SWAP',
                    'uly': underlying, # Required: underlying asset
                    'state': 'filled',
                    'limit': str(min(limit_per_request, 100))
                }

                data = await self._make_request('/api/v5/public/liquidation-orders', params)
                if not data:
                    return None

                symbol_data = []
                for liq in data:
                    details = liq.get('details', [{}])
                    for detail in details:
                        liq_time = int(detail.get('ts', liq.get('ts', 0)))
                        if liq_time == 0:
                            continue

                        symbol_data.append({
                            'timestamp': pd.to_datetime(liq_time, unit='ms', utc=True),
                            'symbol': symbol.upper(),
                            'side': detail.get('side', liq.get('side', '')).lower(),
                            'pos_side': detail.get('posSide', ''),
                            'price': float(detail.get('bkPx', 0) or 0), # Bankruptcy price
                            'size': float(detail.get('sz', 0) or 0),
                            'loss': float(detail.get('bkLoss', 0) or 0), # Bankruptcy loss
                            'liquidation_value': float(detail.get('bkPx', 0) or 0) * float(detail.get('sz', 0) or 0),
                            'venue': self.VENUE,
                            'venue_type': self.VENUE_TYPE
                        })

                logger.info(f"Completed {symbol} liquidations: {len(symbol_data)} records")
                return symbol_data
            except Exception as e:
                logger.error(f"Error fetching liquidations for {symbol}: {e}")
                return None

    async def fetch_liquidations(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        limit_per_request: int = 100
    ) -> pd.DataFrame:
        """
        Fetch liquidation orders from OKX.

        Uses /api/v5/public/liquidation-orders endpoint (public, no auth required).
        Requires 'uly' (underlying) parameter for SWAP instruments.

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH'])
            start_date: Start date string 'YYYY-MM-DD'
            end_date: End date string 'YYYY-MM-DD'
            limit_per_request: Max records per request (max 100)

        Returns:
            DataFrame with liquidation order data
        """
        tasks = [self._fetch_liquidations_single(symbol, limit_per_request) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine all results
        all_data = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in fetch_liquidations: {result}")
                continue
            if result:
                all_data.extend(result)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        self.collection_stats['records_collected'] += len(df)
        return df

    async def collect_liquidations(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect liquidations - wraps fetch_liquidations().

        Standardized method name for collection manager compatibility.
        """
        try:
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            limit = kwargs.get('limit_per_request', 100)

            return await self.fetch_liquidations(
                symbols=symbols,
                start_date=start_str,
                end_date=end_str,
                limit_per_request=limit
            )
        except Exception as e:
            logger.error(f"OKX collect_liquidations error: {e}")
            return pd.DataFrame()

    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.collection_stats.copy()

async def test_okx_collector():
    """Test OKX collector functionality."""
    config = {'rate_limit': 120}
    
    async with OKXCollector(config) as collector:
        print("=" * 60)
        print("OKX Collector Test")
        print("=" * 60)
        
        symbols = await collector.get_available_symbols(min_volume_usd=50_000_000)
        print(f"\n1. Found {len(symbols)} high-volume symbols: {symbols[:10]}")
        
        funding = await collector.fetch_current_funding_rate(['BTC', 'ETH', 'SOL'])
        if not funding.empty:
            print("\n2. Current funding rates:")
            for _, row in funding.iterrows():
                print(f" {row['symbol']}: {row['funding_rate_pct']:.4f}% ({row['trend']})")
        
        print(f"\nStats: {collector.get_collection_stats()}")

if __name__ == '__main__':
    asyncio.run(test_okx_collector())