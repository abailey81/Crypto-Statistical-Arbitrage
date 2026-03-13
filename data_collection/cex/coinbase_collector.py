"""
Coinbase Data Collector - Regulated US CEX for Spot Data

validated collector for Coinbase detailed Trade API.
Primary US-regulated exchange for institutional spot cryptocurrency trading.

Supported Data Types:
    - Spot OHLCV data (candlesticks) with multiple timeframes
    - Trading pairs information and specifications
    - Market trades (recent fills)
    - Order book snapshots (L2 data)
    - Product ticker data
    - 24h statistics

API Documentation: https://docs.cloud.coinbase.com/detailed-trade-api/

Rate Limits:
    - Public endpoints: 10 requests/second
    - Private endpoints: 15 requests/second
    - detailed Trade API: 30 requests/second (authenticated)

Important Notes:
    - Coinbase does NOT offer perpetual futures (US regulatory restrictions)
    - Used primarily for spot price validation and altcoin data
    - High regulatory compliance (SEC, FinCEN registered)
    - Strong institutional presence

Authentication:
    - JWT-based authentication with EC private key
    - API key format: "organizations/{org_id}/apiKeys/{key_id}"

Statistical Arbitrage Applications:
    - Spot price validation (reference rate)
    - US market hours analysis
    - Regulatory-compliant data source
    - Institutional flow proxy
    - Spot-futures basis calculation (with CME)
    - Cross-exchange spot arbitrage

Version: 2.0.0
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import jwt
import time
import secrets
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

from ..base_collector import BaseCollector, CollectionStats
from ..utils.rate_limiter import get_shared_rate_limiter
from ..utils.retry_handler import RetryHandler

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================

class ProductType(Enum):
    """Coinbase product types."""
    SPOT = 'SPOT'
    FUTURE = 'FUTURE' # Limited availability

class ProductStatus(Enum):
    """Product trading status."""
    ONLINE = 'online'
    OFFLINE = 'offline'
    INTERNAL = 'internal'
    DELISTED = 'delisted'

class Granularity(Enum):
    """OHLCV candle granularity options."""
    ONE_MINUTE = 'ONE_MINUTE'
    FIVE_MINUTE = 'FIVE_MINUTE'
    FIFTEEN_MINUTE = 'FIFTEEN_MINUTE'
    THIRTY_MINUTE = 'THIRTY_MINUTE'
    ONE_HOUR = 'ONE_HOUR'
    TWO_HOUR = 'TWO_HOUR'
    SIX_HOUR = 'SIX_HOUR'
    ONE_DAY = 'ONE_DAY'

class TradeSide(Enum):
    """Trade execution side."""
    BUY = 'BUY'
    SELL = 'SELL'
    UNKNOWN = 'UNKNOWN'

class PriceTrend(Enum):
    """Price trend classification."""
    STRONG_UP = 'strong_up' # > 5% change
    UP = 'up' # 1-5% change
    FLAT = 'flat' # -1% to 1%
    DOWN = 'down' # -5% to -1%
    STRONG_DOWN = 'strong_down' # < -5%

class VolumeTier(Enum):
    """Volume classification tiers."""
    VERY_HIGH = 'very_high' # Top 10%
    HIGH = 'high' # Top 25%
    MEDIUM = 'medium' # Top 50%
    LOW = 'low' # Bottom 50%
    VERY_LOW = 'very_low' # Bottom 25%

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class CoinbaseProduct:
    """Coinbase trading product specification."""
    product_id: str
    base_currency: str
    quote_currency: str
    base_min_size: float
    base_max_size: float
    quote_increment: float
    base_increment: float
    status: str
    trading_disabled: bool
    cancel_only: bool
    limit_only: bool
    post_only: bool
    auction_mode: bool
    
    @property
    def is_tradeable(self) -> bool:
        """Check if product is actively tradeable."""
        return not self.trading_disabled and self.status == 'online'
    
    @property
    def is_usd_pair(self) -> bool:
        """Check if USD quote currency."""
        return self.quote_currency == 'USD'
    
    @property
    def is_usdc_pair(self) -> bool:
        """Check if USDC quote currency."""
        return self.quote_currency == 'USDC'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'product_id': self.product_id,
            'base_currency': self.base_currency,
            'quote_currency': self.quote_currency,
            'base_min_size': self.base_min_size,
            'base_max_size': self.base_max_size,
            'quote_increment': self.quote_increment,
            'base_increment': self.base_increment,
            'status': self.status,
            'trading_disabled': self.trading_disabled,
            'cancel_only': self.cancel_only,
            'limit_only': self.limit_only,
            'is_tradeable': self.is_tradeable,
            'is_usd_pair': self.is_usd_pair,
        }

@dataclass
class CoinbaseTicker:
    """Coinbase ticker/market data."""
    timestamp: datetime
    product_id: str
    price: float
    size: float
    bid: float
    ask: float
    volume_24h: float
    low_24h: float
    high_24h: float
    open_24h: float
    
    @property
    def symbol(self) -> str:
        """Extract base symbol."""
        return self.product_id.split('-')[0] if '-' in self.product_id else self.product_id
    
    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid
    
    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid price."""
        mid = (self.ask + self.bid) / 2
        return (self.spread / mid * 100) if mid > 0 else 0
    
    @property
    def change_24h(self) -> float:
        """24h price change."""
        return self.price - self.open_24h
    
    @property
    def change_24h_pct(self) -> float:
        """24h price change percentage."""
        return (self.change_24h / self.open_24h * 100) if self.open_24h > 0 else 0
    
    @property
    def range_24h(self) -> float:
        """24h price range."""
        return self.high_24h - self.low_24h
    
    @property
    def range_24h_pct(self) -> float:
        """24h range as percentage."""
        return (self.range_24h / self.low_24h * 100) if self.low_24h > 0 else 0
    
    @property
    def price_trend(self) -> PriceTrend:
        """Classify price trend."""
        change = self.change_24h_pct
        if change > 5:
            return PriceTrend.STRONG_UP
        elif change > 1:
            return PriceTrend.UP
        elif change > -1:
            return PriceTrend.FLAT
        elif change > -5:
            return PriceTrend.DOWN
        else:
            return PriceTrend.STRONG_DOWN
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'product_id': self.product_id,
            'symbol': self.symbol,
            'price': self.price,
            'size': self.size,
            'bid': self.bid,
            'ask': self.ask,
            'spread': self.spread,
            'spread_pct': self.spread_pct,
            'volume_24h': self.volume_24h,
            'low_24h': self.low_24h,
            'high_24h': self.high_24h,
            'open_24h': self.open_24h,
            'change_24h': self.change_24h,
            'change_24h_pct': self.change_24h_pct,
            'range_24h_pct': self.range_24h_pct,
            'price_trend': self.price_trend.value,
        }

@dataclass
class CoinbaseTrade:
    """Individual trade/fill data."""
    timestamp: datetime
    product_id: str
    trade_id: str
    price: float
    size: float
    side: str
    
    @property
    def value(self) -> float:
        """Trade value in quote currency."""
        return self.price * self.size
    
    @property
    def is_buy(self) -> bool:
        """Check if taker was buyer."""
        return self.side.upper() == 'BUY'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'product_id': self.product_id,
            'trade_id': self.trade_id,
            'price': self.price,
            'size': self.size,
            'side': self.side,
            'value': self.value,
            'is_buy': self.is_buy,
        }

@dataclass
class CoinbaseCandle:
    """OHLCV candle data."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @property
    def typical_price(self) -> float:
        """Typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3
    
    @property
    def range(self) -> float:
        """Candle range (high - low)."""
        return self.high - self.low
    
    @property
    def range_pct(self) -> float:
        """Range as percentage of low."""
        return (self.range / self.low * 100) if self.low > 0 else 0
    
    @property
    def body(self) -> float:
        """Candle body (close - open)."""
        return self.close - self.open
    
    @property
    def body_pct(self) -> float:
        """Body as percentage of open."""
        return (self.body / self.open * 100) if self.open > 0 else 0
    
    @property
    def is_bullish(self) -> bool:
        """Check if bullish candle."""
        return self.close > self.open
    
    @property
    def upper_wick(self) -> float:
        """Upper wick size."""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> float:
        """Lower wick size."""
        return min(self.open, self.close) - self.low
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'typical_price': self.typical_price,
            'range_pct': self.range_pct,
            'body_pct': self.body_pct,
            'is_bullish': self.is_bullish,
        }

# =============================================================================
# Collector Class
# =============================================================================

class CoinbaseCollector(BaseCollector):
    """
    Coinbase detailed Trade API data collector.
    
    validated implementation for US-regulated spot data.
    
    Features:
    - Spot OHLCV (candlesticks)
    - Product information
    - Market trades
    - Ticker data with 24h stats
    - Order book snapshots
    
    Note: No perpetual futures support (US regulatory restrictions).
    
    Attributes:
        VENUE: Exchange identifier ('coinbase')
        VENUE_TYPE: Exchange type ('CEX')
        BASE_URL: API endpoint
        MAX_CANDLES: Maximum candles per request (300)
    
    Example:
        >>> config = {'api_key': 'YOUR_KEY', 'api_secret': 'YOUR_EC_PRIVATE_KEY'}
        >>> async with CoinbaseCollector(config) as collector:
        ... ohlcv = await collector.fetch_ohlcv(['BTC', 'ETH'], '1h', '2024-01-01', '2024-01-31')
        ... ticker = await collector.fetch_ticker(['BTC', 'ETH'])
    """
    
    VENUE = 'coinbase'
    VENUE_TYPE = 'CEX'
    BASE_URL = 'https://api.coinbase.com'
    
    MAX_CANDLES = 300
    
    GRANULARITY_MAP = {
        '1m': 'ONE_MINUTE', '5m': 'FIVE_MINUTE', '15m': 'FIFTEEN_MINUTE',
        '30m': 'THIRTY_MINUTE', '1h': 'ONE_HOUR', '2h': 'TWO_HOUR',
        '6h': 'SIX_HOUR', '1d': 'ONE_DAY'
    }
    
    GRANULARITY_SECONDS = {
        '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
        '1h': 3600, '2h': 7200, '6h': 21600, '1d': 86400
    }

    # Cache for dynamically fetched available products
    # This replaces the static UNSUPPORTED_PRODUCTS approach
    _available_products_cache: Optional[set] = None
    _products_cache_time: Optional[datetime] = None
    PRODUCTS_CACHE_TTL = 3600 # 1 hour cache TTL for available products

    def __init__(self, config: Dict[str, Any]):
        """Initialize Coinbase collector."""
        super().__init__(config)

        # Initialize dynamic products cache
        self._available_products_cache: Optional[set] = None
        self._products_cache_time: Optional[datetime] = None

        # Load API keys from config or environment variables
        self.api_key = config.get('api_key') or config.get('coinbase_api_key') or os.getenv('COINBASE_API_KEY', '')
        self.api_secret = config.get('api_secret') or config.get('coinbase_api_secret') or os.getenv('COINBASE_PRIVATE_KEY', '') or os.getenv('COINBASE_SECRET_KEY', '')
        self.key_name = self.api_key
        
        self._private_key = None
        if self.api_secret and HAS_CRYPTOGRAPHY:
            try:
                pem_string = self.api_secret.replace('\\n', '\n')
                self._private_key = serialization.load_pem_private_key(
                    pem_string.encode('utf-8'), password=None, backend=default_backend()
                )
                logger.info("Coinbase EC private key loaded")
            except Exception as e:
                logger.warning(f"Failed to load Coinbase private key: {e}")
        
        rate_limit = config.get('rate_limit', 15)  # Conservative: Coinbase ~10/sec
        self.rate_limiter = get_shared_rate_limiter('coinbase', rate=rate_limit, per=60.0, burst=3)
        self.retry_handler = RetryHandler(max_retries=5, base_delay=3.0, max_delay=60.0)
        self._semaphore = asyncio.Semaphore(5)  # Max 5 concurrent symbol fetches
        
        self.timeout = aiohttp.ClientTimeout(total=config.get('timeout', 30))
        self.session: Optional[aiohttp.ClientSession] = None
        
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl = config.get('cache_ttl', 60)
        
        # Semaphore to limit concurrent HTTP requests (prevents rate limit exhaustion)
        self._request_sem = asyncio.Semaphore(2)

        self.collection_stats = {'records_collected': 0, 'api_calls': 0, 'errors': 0, 'cache_hits': 0}
        self._auth_warning_logged = False # Track if auth warning already logged
        logger.info(f"Initialized Coinbase collector (rate_limit={rate_limit}/min)")
    
    async def __aenter__(self):
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            # SPEEDUP: Aggressive connection pooling for high-throughput collection
            # Coinbase rate limit: 10/sec public, higher pool supports concurrent symbols
            connector = aiohttp.TCPConnector(
                limit=60, # Total connection pool size (was 10)
                limit_per_host=20, # Per-host connections (was 5)
                ttl_dns_cache=300, # DNS cache TTL in seconds
                force_close=False, # Keep-alive connections
                enable_cleanup_closed=True
            )
            self.session = aiohttp.ClientSession(timeout=self.timeout, connector=connector)
        return self.session
    
    async def close(self):
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
        self._cache.clear()
        logger.info(f"Coinbase collector closed. Stats: {self.collection_stats}")
    
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
    
    def _generate_jwt(self, request_method: str, request_path: str) -> Optional[str]:
        """Generate JWT token for authenticated requests."""
        if not self._private_key:
            return None
        
        uri = f"{request_method} api.coinbase.com{request_path}"
        
        payload = {
            'sub': self.key_name,
            'iss': 'coinbase-cloud',
            'nbf': int(time.time()),
            'exp': int(time.time()) + 120,
            'aud': ['retail_rest_api_proxy'],
            'uri': uri
        }
        
        headers = {'kid': self.key_name, 'nonce': secrets.token_hex(16)}
        
        return jwt.encode(payload, self._private_key, algorithm='ES256', headers=headers)
    
    def _get_headers(self, method: str, path: str) -> Dict[str, str]:
        """Generate headers for requests."""
        headers = {'Content-Type': 'application/json', 'User-Agent': 'CryptoStatArb/2.0'}
        if self._private_key:
            token = self._generate_jwt(method, path)
            if token:
                headers['Authorization'] = f'Bearer {token}'
        return headers
    
    def _format_symbol(self, symbol: str) -> str:
        """Convert to Coinbase format (XXX-USD)."""
        symbol = symbol.upper()
        for suffix in ['/USD', 'USD', '/USDT', 'USDT', '-USD']:
            symbol = symbol.replace(suffix, '')
        return f"{symbol}-USD"
    
    def _parse_symbol(self, cb_symbol: str) -> str:
        """Parse Coinbase symbol to internal format."""
        return cb_symbol.split('-')[0] if '-' in cb_symbol else cb_symbol

    async def _get_available_products(self) -> set:
        """
        Dynamically fetch and cache available USD trading products.

        This replaces the static UNSUPPORTED_PRODUCTS approach with
        dynamic validation against actual Coinbase product listings.
        """
        now = datetime.utcnow()

        # Return cached products if still valid
        if (self._available_products_cache is not None and
            self._products_cache_time is not None and
            (now - self._products_cache_time).total_seconds() < self.PRODUCTS_CACHE_TTL):
            return self._available_products_cache

        # Fetch available products from API
        try:
            data = await self._make_request('/api/v3/brokerage/products', use_cache=True)

            if data and 'products' in data:
                available = set()
                for product in data.get('products', []):
                    # Only include USD pairs that are tradeable
                    if product.get('quote_currency_id') == 'USD':
                        status = product.get('status', '')
                        trading_disabled = product.get('trading_disabled', True)
                        if status == 'online' and not trading_disabled:
                            base = product.get('base_currency_id', '')
                            if base:
                                available.add(base.upper())

                self._available_products_cache = available
                self._products_cache_time = now
                logger.info(f"Coinbase: Cached {len(available)} available USD products")
                return available
            else:
                logger.warning("Coinbase: Failed to fetch products, using fallback")
                # Fallback to known major coins if API fails
                return {'BTC', 'ETH', 'SOL', 'AVAX', 'LINK', 'DOT', 'UNI', 'AAVE', 'CRV',
                        'ATOM', 'ADA', 'XRP', 'LTC', 'BCH', 'ETC', 'ALGO', 'XLM', 'NEAR'}
        except Exception as e:
            logger.warning(f"Coinbase: Error fetching products: {e}, using fallback")
            return {'BTC', 'ETH', 'SOL', 'AVAX', 'LINK', 'DOT', 'UNI', 'AAVE', 'CRV'}

    async def _is_product_available(self, symbol: str) -> bool:
        """Check if a product is available for trading on Coinbase."""
        available = await self._get_available_products()
        base_symbol = symbol.upper().replace('-USD', '').replace('USDT', '').replace('USD', '')
        return base_symbol in available
    
    async def _make_request(self, endpoint: str, params: Dict = None, authenticated: bool = True, use_cache: bool = False) -> Optional[Dict]:
        """Make rate-limited request with retry."""
        async with self._request_sem:
            if use_cache:
                cache_key = f"{endpoint}_{hash(frozenset((params or {}).items()))}"
                cached = self._get_cached(cache_key)
                if cached is not None:
                    return cached

            session = await self._get_session()
            url = f"{self.BASE_URL}{endpoint}"

            if params:
                query_string = '&'.join(f"{k}={v}" for k, v in params.items() if v is not None)
                url += f"?{query_string}"

            # JWT uri should NOT include query parameters - only the path
            headers = self._get_headers('GET', endpoint) if authenticated else {}

            async def _request():
                await self.rate_limiter.acquire()
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        self.collection_stats['api_calls'] += 1
                        return await response.json()
                    elif response.status == 401:
                        # Log only once to avoid spamming
                        if not self._auth_warning_logged:
                            logger.warning("Coinbase authentication failed - check API credentials")
                            self._auth_warning_logged = True
                        return None
                    elif response.status == 429:
                        logger.warning("Coinbase rate limited - backing off 15s")
                        await asyncio.sleep(15)
                        raise aiohttp.ClientResponseError(response.request_info, response.history, status=429)
                    else:
                        text = await response.text()
                        logger.error(f"Coinbase error {response.status}: {text[:200]}")
                        return None

            try:
                result = await self.retry_handler.execute(_request)
                if use_cache and result is not None:
                    self._set_cached(cache_key, result)
                return result
            except Exception as e:
                logger.error(f"Coinbase request error: {e}")
                self.collection_stats['errors'] += 1
                return None
    
    async def fetch_funding_rates(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Coinbase does not have perpetual futures - returns empty DataFrame."""
        logger.info("Coinbase does not support perpetual futures - no funding rates")
        return pd.DataFrame()
    
    async def _fetch_single_ohlcv(
        self, symbol: str, timeframe: str, start_ts: int, end_ts: int,
        available_products: set, granularity: str, seconds_per_candle: int,
        window_seconds: int
    ) -> List[Dict]:
        """Helper to fetch OHLCV data for a single symbol."""
        try:
            # Dynamically validate against available Coinbase products
            base_symbol = symbol.upper().replace('-USD', '').replace('USDT', '').replace('USD', '')
            if base_symbol not in available_products:
                logger.debug(f"Skipping {symbol} - not available on Coinbase (dynamic check)")
                return []

            cb_symbol = self._format_symbol(symbol)
            logger.info(f"Fetching Coinbase OHLCV: {cb_symbol} ({timeframe})")

            symbol_data = []
            current_start = start_ts
            symbol_records = 0

            while current_start < end_ts:
                current_end = min(current_start + window_seconds, end_ts)

                params = {'start': str(current_start), 'end': str(current_end), 'granularity': granularity}
                endpoint = f'/api/v3/brokerage/products/{cb_symbol}/candles'

                data = await self._make_request(endpoint, params)

                if not data:
                    current_start = current_end
                    continue

                candles = data.get('candles', [])

                for candle in candles:
                    c = CoinbaseCandle(
                        timestamp=pd.to_datetime(int(candle['start']), unit='s', utc=True),
                        symbol=symbol.upper(),
                        open=float(candle['open']),
                        high=float(candle['high']),
                        low=float(candle['low']),
                        close=float(candle['close']),
                        volume=float(candle['volume'])
                    )

                    symbol_data.append({
                        **c.to_dict(),
                        'volume_usd': c.volume * c.close,
                        'contract_type': 'SPOT',
                        'venue': self.VENUE,
                        'venue_type': self.VENUE_TYPE
                    })
                    symbol_records += 1

                current_start = current_end

            logger.info(f"Completed {symbol} OHLCV: {symbol_records} records")
            return symbol_data
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return []

    async def fetch_ohlcv(self, symbols: List[str], timeframe: str, start_date: str, end_date: str, contract_type: str = 'spot') -> pd.DataFrame:
        """
        Fetch OHLCV data for spot markets (parallelized).

        Args:
            symbols: List of symbols
            timeframe: Interval ('1m', '5m', '15m', '30m', '1h', '2h', '6h', '1d')
            start_date: Start date
            end_date: End date
            contract_type: Only 'spot' supported

        Returns:
            DataFrame with OHLCV data and derived metrics
        """
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        granularity = self.GRANULARITY_MAP.get(timeframe.lower(), 'ONE_HOUR')
        seconds_per_candle = self.GRANULARITY_SECONDS.get(timeframe.lower(), 3600)
        window_seconds = self.MAX_CANDLES * seconds_per_candle

        # Fetch available products once for all symbols
        available_products = await self._get_available_products()

        # Parallelize symbol fetching with concurrency limit to avoid 429s
        async def _limited_fetch(symbol):
            async with self._semaphore:
                return await self._fetch_single_ohlcv(
                    symbol, timeframe, start_ts, end_ts,
                    available_products, granularity, seconds_per_candle, window_seconds
                )
        tasks = [_limited_fetch(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results
        all_data = []
        for result in results:
            if isinstance(result, list):
                all_data.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"OHLCV fetch error: {result}")

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        df = df.drop_duplicates(subset=['timestamp', 'symbol'], keep='first')
        self.collection_stats['records_collected'] += len(df)
        return df
    
    async def fetch_products(self) -> pd.DataFrame:
        """Fetch available trading products."""
        data = await self._make_request('/api/v3/brokerage/products', use_cache=True)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        for product in data.get('products', []):
            if product.get('quote_currency_id') in ['USD', 'USDC']:
                p = CoinbaseProduct(
                    product_id=product['product_id'],
                    base_currency=product['base_currency_id'],
                    quote_currency=product['quote_currency_id'],
                    base_min_size=float(product.get('base_min_size', 0)),
                    base_max_size=float(product.get('base_max_size', 0)),
                    quote_increment=float(product.get('quote_increment', 0)),
                    base_increment=float(product.get('base_increment', 0)),
                    status=product.get('status', 'online'),
                    trading_disabled=product.get('trading_disabled', False),
                    cancel_only=product.get('cancel_only', False),
                    limit_only=product.get('limit_only', False),
                    post_only=product.get('post_only', False),
                    auction_mode=product.get('auction_mode', False)
                )
                records.append({**p.to_dict(), 'venue': self.VENUE})
        
        return pd.DataFrame(records)
    
    async def _fetch_single_ticker(self, symbol: str) -> Optional[Dict]:
        """Helper to fetch ticker data for a single symbol."""
        try:
            cb_symbol = self._format_symbol(symbol)
            endpoint = f'/api/v3/brokerage/products/{cb_symbol}'
            data = await self._make_request(endpoint)

            if not data:
                return None

            t = CoinbaseTicker(
                timestamp=datetime.utcnow().replace(tzinfo=timezone.utc),
                product_id=cb_symbol,
                price=float(data.get('price', 0) or 0),
                size=0,
                bid=float(data.get('bid', 0) or 0),
                ask=float(data.get('ask', 0) or 0),
                volume_24h=float(data.get('volume_24h', 0) or 0),
                low_24h=float(data.get('low_24h', 0) or 0),
                high_24h=float(data.get('high_24h', 0) or 0),
                open_24h=float(data.get('open_24h', 0) or 0)
            )

            return {**t.to_dict(), 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE}
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None

    async def fetch_ticker(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch current ticker data with 24h stats (parallelized).

        Args:
            symbols: List of symbols

        Returns:
            DataFrame with ticker data and derived metrics
        """
        tasks = [self._fetch_single_ticker(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        records = [r for r in results if isinstance(r, dict)]
        return pd.DataFrame(records)
    
    async def fetch_trades(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch recent trades for a symbol.
        
        Args:
            symbol: Symbol to fetch
            limit: Number of trades (max 100)
            
        Returns:
            DataFrame with trade data
        """
        cb_symbol = self._format_symbol(symbol)
        endpoint = f'/api/v3/brokerage/products/{cb_symbol}/ticker'
        
        data = await self._make_request(endpoint)
        
        if not data or 'trades' not in data:
            return pd.DataFrame()
        
        records = []
        for trade in data.get('trades', [])[:limit]:
            t = CoinbaseTrade(
                timestamp=pd.to_datetime(trade['time']),
                product_id=cb_symbol,
                trade_id=trade.get('trade_id', ''),
                price=float(trade['price']),
                size=float(trade['size']),
                side=trade['side']
            )
            records.append({**t.to_dict(), 'venue': self.VENUE})
        
        return pd.DataFrame(records)
    
    async def fetch_market_snapshot(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch comprehensive market snapshot.
        
        Args:
            symbols: List of symbols
            
        Returns:
            DataFrame with complete market data
        """
        return await self.fetch_ticker(symbols)
    
    async def get_available_symbols(self, min_volume_usd: float = 1_000_000) -> List[str]:
        """Get available USD trading pairs filtered by implied volume."""
        products = await self.fetch_products()
        
        if products.empty:
            return ['BTC', 'ETH', 'SOL', 'AVAX', 'LINK', 'MATIC', 'DOT', 'UNI', 'AAVE', 'CRV']
        
        # Filter for USD pairs that are tradeable
        usd_products = products[
            (products['is_usd_pair']) & 
            (products['is_tradeable'])
        ]
        
        symbols = sorted(usd_products['base_currency'].unique().tolist())
        logger.info(f"Found {len(symbols)} available Coinbase USD pairs")
        return symbols[:100]
    
    async def fetch_comprehensive_data(self, symbols: List[str], start_date: str, end_date: str, timeframe: str = '1h') -> Dict[str, pd.DataFrame]:
        """
        Fetch comprehensive spot data.
        
        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            timeframe: OHLCV timeframe
            
        Returns:
            Dictionary with DataFrames for each data type
        """
        results = {}
        
        logger.info(f"Fetching comprehensive Coinbase data for {len(symbols)} symbols")
        
        results['ohlcv'] = await self.fetch_ohlcv(symbols, timeframe, start_date, end_date)
        results['ticker'] = await self.fetch_ticker(symbols)
        results['products'] = await self.fetch_products()
        
        return results
    
    async def _collect_single_trade(self, symbol: str, limit: int) -> Optional[pd.DataFrame]:
        """Helper to collect trades for a single symbol."""
        try:
            df = await self.fetch_trades(symbol=symbol, limit=limit)
            if not df.empty:
                return df
            return None
        except Exception as e:
            logger.error(f"Error collecting trades for {symbol}: {e}")
            return None

    async def collect_trades(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect recent trades - wraps fetch_trades() (parallelized).

        Args:
            symbols: List of symbols
            start_date: Start date (ignored - returns recent trades)
            end_date: End date (ignored - returns recent trades)
            **kwargs: Additional arguments (limit)

        Returns:
            DataFrame with recent trade data
        """
        try:
            limit = kwargs.get('limit', 100)

            tasks = [self._collect_single_trade(symbol, limit) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_records = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

            if all_records:
                result = pd.concat(all_records, ignore_index=True)
                # Add venue_type if not present
                if 'venue_type' not in result.columns:
                    result['venue_type'] = self.VENUE_TYPE
                return result

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Coinbase collect_trades error: {e}")
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
            contract_type = kwargs.get('contract_type', 'spot')

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
            logger.error(f"Coinbase collect_ohlcv error: {e}")
            return pd.DataFrame()

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
            logger.error(f"Coinbase collect_funding_rates error: {e}")
            return pd.DataFrame()

    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.collection_stats.copy()
    
    def reset_collection_stats(self):
        """Reset collection statistics."""
        self.collection_stats = {'records_collected': 0, 'api_calls': 0, 'errors': 0, 'cache_hits': 0}

async def test_coinbase_collector():
    """Test Coinbase collector functionality."""
    config = {'rate_limit': 15}
    
    async with CoinbaseCollector(config) as collector:
        print("=" * 60)
        print("Coinbase Collector Test")
        print("=" * 60)
        
        symbols = await collector.get_available_symbols()
        print(f"\n1. Found {len(symbols)} available symbols: {symbols[:10]}")
        
        ticker = await collector.fetch_ticker(['BTC', 'ETH', 'SOL'])
        if not ticker.empty:
            print("\n2. Current prices:")
            for _, row in ticker.iterrows():
                print(f" {row['symbol']}: ${row['price']:,.2f} ({row['change_24h_pct']:.2f}%)")
        
        products = await collector.fetch_products()
        if not products.empty:
            print(f"\n3. Products: {len(products)} USD/USDC pairs available")
        
        print(f"\nStats: {collector.get_collection_stats()}")

if __name__ == '__main__':
    asyncio.run(test_coinbase_collector())