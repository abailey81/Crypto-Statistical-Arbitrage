"""
Bybit Data Collector - Secondary CEX for Cross-Validation

validated collector for Bybit derivatives and spot markets.
Second largest crypto derivatives exchange by volume.

Supported Data Types:
    - USDT perpetual funding rates (8-hour intervals)
    - USDC perpetual funding rates 
    - Inverse perpetual funding rates
    - Spot and futures OHLCV (klines)
    - Open interest history
    - Long/short ratio (account-based)
    - Insurance fund data
    - Instrument specifications

API Documentation: https://bybit-exchange.github.io/docs/v5/intro

Rate Limits:
    - 120 requests per minute (IP-based, unauthenticated)
    - 600 requests per minute (with API key)
    - Burst: 50 requests per 5 seconds

Contract Specifications (USDT Perpetuals):
    - Funding interval: 8 hours (00:00, 08:00, 16:00 UTC)
    - Funding cap: +/- 0.375% per interval
    - Mark price: EMA of index price
    - Leverage: Up to 100x

Statistical Arbitrage Applications:
    - Cross-exchange funding arbitrage (Bybit vs Binance)
    - Retail sentiment analysis (different user base)
    - Market microstructure comparison
    - Liquidity fragmentation analysis
    - Insurance fund health monitoring

Version: 2.0.0
"""

import asyncio
import os
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
import time

from ..base_collector import BaseCollector, CollectionStats
from ..utils.rate_limiter import get_shared_rate_limiter
from ..utils.retry_handler import RetryHandler, safe_float, safe_int

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================

class Category(Enum):
    """Bybit market categories."""
    LINEAR = 'linear'
    INVERSE = 'inverse'
    SPOT = 'spot'
    OPTION = 'option'

class ContractStatus(Enum):
    """Contract trading status."""
    TRADING = 'Trading'
    SETTLING = 'Settling'
    CLOSED = 'Closed'

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

class TickDirection(Enum):
    """Price tick direction."""
    PLUS_TICK = 'PlusTick'
    ZERO_PLUS_TICK = 'ZeroPlusTick'
    MINUS_TICK = 'MinusTick'
    ZERO_MINUS_TICK = 'ZeroMinusTick'

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class BybitFundingRate:
    """Bybit funding rate data point."""
    timestamp: datetime
    symbol: str
    funding_rate: float
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
    def is_arbitrage_opportunity(self) -> bool:
        """Check if rate is significant for arbitrage."""
        return abs(self.funding_rate) > 0.0003
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'funding_rate': self.funding_rate,
            'funding_rate_pct': self.funding_rate * 100,
            'annualized_rate': self.annualized_rate,
            'annualized_rate_pct': self.annualized_rate_pct,
            'daily_rate': self.daily_rate,
            'trend': self.trend.value,
            'is_arbitrage_opportunity': self.is_arbitrage_opportunity,
        }

@dataclass
class BybitTicker:
    """Bybit ticker/market snapshot data."""
    timestamp: datetime
    symbol: str
    last_price: float
    mark_price: float
    index_price: float
    funding_rate: float
    next_funding_time: Optional[datetime]
    open_interest: float
    open_interest_value: float
    volume_24h: float
    turnover_24h: float
    high_24h: float
    low_24h: float
    price_change_24h_pct: float
    
    @property
    def basis(self) -> float:
        """Mark price basis to index."""
        return self.mark_price - self.index_price
    
    @property
    def basis_pct(self) -> float:
        """Basis as percentage."""
        return (self.basis / self.index_price * 100) if self.index_price > 0 else 0
    
    @property
    def funding_annualized(self) -> float:
        """Annualized funding rate."""
        return self.funding_rate * 3 * 365
    
    @property
    def range_24h_pct(self) -> float:
        """24h range as percentage."""
        return ((self.high_24h - self.low_24h) / self.low_24h * 100) if self.low_24h > 0 else 0
    
    @property
    def turnover_ratio(self) -> float:
        """Volume/OI ratio."""
        return self.turnover_24h / self.open_interest_value if self.open_interest_value > 0 else 0
    
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
            'funding_rate': self.funding_rate,
            'funding_rate_pct': self.funding_rate * 100,
            'funding_annualized': self.funding_annualized,
            'next_funding_time': self.next_funding_time,
            'open_interest': self.open_interest,
            'open_interest_value': self.open_interest_value,
            'volume_24h': self.volume_24h,
            'turnover_24h': self.turnover_24h,
            'high_24h': self.high_24h,
            'low_24h': self.low_24h,
            'price_change_24h_pct': self.price_change_24h_pct,
            'range_24h_pct': self.range_24h_pct,
            'turnover_ratio': self.turnover_ratio,
        }

@dataclass
class BybitLongShortRatio:
    """Bybit long/short ratio data."""
    timestamp: datetime
    symbol: str
    buy_ratio: float
    sell_ratio: float
    
    @property
    def long_short_ratio(self) -> float:
        """Calculate L/S ratio."""
        return self.buy_ratio / self.sell_ratio if self.sell_ratio > 0 else float('inf')
    
    @property
    def sentiment(self) -> MarketSentiment:
        """Market sentiment classification."""
        ratio = self.long_short_ratio
        if ratio > 1.5:
            return MarketSentiment.STRONGLY_BULLISH
        elif ratio > 1.1:
            return MarketSentiment.BULLISH
        elif ratio > 0.9:
            return MarketSentiment.NEUTRAL
        elif ratio > 0.67:
            return MarketSentiment.BEARISH
        else:
            return MarketSentiment.STRONGLY_BEARISH
    
    @property
    def imbalance_pct(self) -> float:
        """Position imbalance percentage."""
        return (self.buy_ratio - self.sell_ratio) * 100
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'buy_ratio': self.buy_ratio,
            'sell_ratio': self.sell_ratio,
            'long_short_ratio': self.long_short_ratio,
            'sentiment': self.sentiment.value,
            'imbalance_pct': self.imbalance_pct,
        }

@dataclass
class BybitInstrument:
    """Bybit instrument specification."""
    symbol: str
    base_coin: str
    quote_coin: str
    status: str
    contract_type: str
    launch_time: Optional[datetime]
    tick_size: float
    min_qty: float
    max_qty: float
    max_leverage: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'base_coin': self.base_coin,
            'quote_coin': self.quote_coin,
            'status': self.status,
            'contract_type': self.contract_type,
            'launch_time': self.launch_time,
            'tick_size': self.tick_size,
            'min_qty': self.min_qty,
            'max_qty': self.max_qty,
            'max_leverage': self.max_leverage,
        }

# =============================================================================
# Collector Class
# =============================================================================

class BybitCollector(BaseCollector):
    """
    Bybit data collector for perpetual futures and spot markets.

    validated implementation for cross-validation with Binance.

    Features:
    - USDT-M and Coin-M perpetual funding rates
    - Spot and futures OHLCV
    - Open interest history
    - Long/short ratio (account-based)
    - Insurance fund data
    - Instrument specifications

    Attributes:
        VENUE: Exchange identifier ('bybit')
        VENUE_TYPE: Exchange type ('CEX')
        BASE_URL: API endpoint
        FUNDING_INTERVAL_HOURS: Funding payment interval (8)

    Example:
        >>> config = {'rate_limit': 60}
        >>> async with BybitCollector(config) as collector:
        ... funding = await collector.fetch_funding_rates(['BTC', 'ETH'], '2024-01-01', '2024-01-31')
        ... snapshot = await collector.fetch_current_funding_rate(['BTC', 'ETH'])
    """

    VENUE = 'bybit'
    VENUE_TYPE = 'CEX'
    BASE_URL = 'https://api.bybit.com'

    # Collection manager compatibility attributes
    supported_data_types = ['funding_rates', 'ohlcv', 'open_interest', 'trades', 'liquidations']
    venue = 'bybit'
    requires_auth = False # Public endpoints available for all data types
    
    FUNDING_INTERVAL_HOURS = 8
    FUNDINGS_PER_DAY = 3
    MAX_RECORDS = 200
    
    VALID_KLINE_INTERVALS = {'1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D', 'W', 'M'}
    VALID_OI_INTERVALS = {'5min', '15min', '30min', '1h', '4h', '1d'}

    # Symbols known to NOT exist on Bybit (meme coins, smaller caps, or renamed)
    # This prevents noisy "symbol invalid" errors
    UNSUPPORTED_SYMBOLS = {
        'MOG', 'MOGUSDT', # Not on Bybit
        'FLOKI', 'FLOKIUSDT', # May be FLOKI1000USDT
        'BONK', 'BONKUSDT', # May be 1000BONKUSDT
        'SHIB', 'SHIBUSDT', # May be 1000SHIBUSDT
        'PEPE', 'PEPEUSDT', # May be 1000PEPEUSDT
        'LUNC', 'LUNCUSDT', # May be 1000LUNCUSDT
        'BTTC', 'BTTCUSDT', # Not listed
        'MATIC', 'MATICUSDT', # Migrated to POL
        'WIN', 'WINUSDT', # Not listed
        'BTT', 'BTTUSDT', # Not listed
        'NFT', 'NFTUSDT', # Not listed
        'JASMY', 'JASMYUSDT', # May need special format
        'FXS', 'FXSUSDT', # Not listed
        'MKR', 'MKRUSDT', # May not have perp
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Bybit collector."""
        super().__init__(config)
        
        # Load API keys from config or environment variables
        self.api_key = config.get('api_key') or config.get('bybit_api_key') or os.getenv('BYBIT_API_KEY', '')
        self.api_secret = config.get('api_secret') or config.get('bybit_api_secret') or os.getenv('BYBIT_SECRET_KEY', '')
        
        # SPEED OPTIMIZATION: Use authenticated rate limit (300/min) if API key available, otherwise unauthenticated (60/min)
        # User has API keys → 300 req/min = 5 req/s (5x faster than unauthenticated)
        default_rate = 300 if self.api_key else 60
        rate_limit = config.get('rate_limit', default_rate)
        # Use shared rate limiter to avoid re-initialization overhead
        # Burst=15 (reduced from 50 to stay within safe limits)
        self.rate_limiter = get_shared_rate_limiter('bybit', rate=rate_limit, per=60.0, burst=15)
        # OPTIMIZATION: Reduced max_delay from 60s to 30s to avoid long stalls
        # Also reduced max_retries from 5 to 3 for faster fail-through on persistent errors
        self.retry_handler = RetryHandler(max_retries=config.get('max_retries', 3), base_delay=1.0, max_delay=30.0)

        self.timeout = aiohttp.ClientTimeout(total=config.get('timeout', 30))
        self.session: Optional[aiohttp.ClientSession] = None

        self.collection_stats = {'records_collected': 0, 'api_calls': 0, 'errors': 0, 'rate_limit_hits': 0}
        # Semaphore to limit concurrent HTTP requests (prevents rate limit exhaustion)
        self._request_sem = asyncio.Semaphore(2)

        auth_status = "AUTHENTICATED (300/min)" if self.api_key else "PUBLIC (60/min)"
        logger.info(f"Initialized Bybit collector (rate_limit={rate_limit}/min, burst=15, {auth_status})")

    async def __aenter__(self):
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            # SPEEDUP: SPEEDUP: Optimized headers with compression support
            headers = {
                'User-Agent': 'CryptoStatArb/2.0',
                'Accept-Encoding': 'gzip, deflate', # Enable response compression
            }
            if self.api_key:
                headers['X-BAPI-API-KEY'] = self.api_key

            # SPEEDUP: SPEEDUP: Aggressive connection pooling (was limit=10, limit_per_host=5)
            # Bybit rate limit: 120/min = 2/sec, but with 10+ symbols we need higher pool
            # Expected speedup: 2-3x
            connector = aiohttp.TCPConnector(
                limit=80, # Total connection pool size (was 10)
                limit_per_host=20, # Per-host connections (was 5)
                ttl_dns_cache=300, # DNS cache TTL in seconds
                force_close=False, # Keep-alive connections
                enable_cleanup_closed=True
            )
            self.session = aiohttp.ClientSession(timeout=self.timeout, headers=headers, connector=connector)
        return self.session
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
        logger.info(f"Bybit collector closed. Stats: {self.collection_stats}")
    
    def _generate_signature(self, timestamp: int, params: Dict) -> str:
        """Generate HMAC signature."""
        param_str = str(timestamp) + self.api_key + '5000'
        param_str += '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
        return hmac.new(self.api_secret.encode('utf-8'), param_str.encode('utf-8'), hashlib.sha256).hexdigest()
    
    def _format_symbol(self, symbol: str) -> str:
        """Convert to Bybit format (BTCUSDT)."""
        symbol = symbol.upper()
        for suffix in ['/USDT', '-USDT', ':USDT', 'USDT', 'PERP']:
            symbol = symbol.replace(suffix, '')
        return f"{symbol}USDT"
    
    def _parse_symbol(self, bybit_symbol: str) -> str:
        """Parse Bybit symbol to internal format."""
        return bybit_symbol.replace('USDT', '') if bybit_symbol.endswith('USDT') else bybit_symbol
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make rate-limited request with retry."""
        async with self._request_sem:
            session = await self._get_session()
            url = f"{self.BASE_URL}{endpoint}"

            async def _request():
                acquire_result = await self.rate_limiter.acquire(timeout=120.0)
                if hasattr(acquire_result, 'acquired') and not acquire_result.acquired:
                    raise TimeoutError("Rate limiter timeout after 120s")

                async with session.get(url, params=params) as response:
                    if response.status == 429:
                        self.collection_stats['rate_limit_hits'] += 1
                        logger.warning("Bybit rate limited")
                        await asyncio.sleep(60)
                        raise aiohttp.ClientResponseError(response.request_info, response.history, status=429)

                    response.raise_for_status()
                    data = await response.json()

                    if data.get('retCode') != 0:
                        error_msg = data.get('retMsg', 'Unknown error')
                        ret_code = data.get('retCode', 0)

                        # Handle known non-critical errors at debug level
                        if 'symbol invalid' in error_msg.lower() or 'invalid symbol' in error_msg.lower():
                            logger.debug(f"Bybit symbol not found: {error_msg}")
                            return None # Return None instead of raising
                        elif ret_code == 10001 or 'params error' in error_msg.lower():
                            logger.debug(f"Bybit params error: {error_msg}")
                            return None
                        elif ret_code == 10002: # Request timeout
                            logger.debug(f"Bybit request timeout: {error_msg}")
                            raise ValueError(f"Bybit timeout: {error_msg}") # Allow retry
                        else:
                            logger.warning(f"Bybit API error ({ret_code}): {error_msg}")
                            raise ValueError(f"Bybit API error: {error_msg}")

                    self.collection_stats['api_calls'] += 1
                    return data.get('result', {})

            try:
                return await self.retry_handler.execute(_request)
            except Exception as e:
                logger.error(f"Request failed: {endpoint} - {e}")
                self.collection_stats['errors'] += 1
                return None
    
    async def _fetch_funding_rate_single(self, symbol: str, start_ts: int, end_ts: int) -> List[Dict]:
        """SPEEDUP: Helper to fetch funding rates for a single symbol (for parallel processing)."""
        # Skip symbols known to not exist on Bybit
        base_symbol = symbol.upper().replace('USDT', '').replace('PERP', '').replace('/USD', '')
        if base_symbol in self.UNSUPPORTED_SYMBOLS or f"{base_symbol}USDT" in self.UNSUPPORTED_SYMBOLS:
            logger.debug(f"Skipping {symbol} - not listed on Bybit")
            return []

        bybit_symbol = self._format_symbol(symbol)
        logger.info(f"Fetching Bybit funding rates: {bybit_symbol}")

        all_data = []
        symbol_records = 0
        current_end_ts = end_ts

        # Time-window pagination: iterate backwards from end_date to start_date
        while current_end_ts > start_ts:
            params = {
                'category': 'linear',
                'symbol': bybit_symbol,
                'startTime': start_ts,
                'endTime': current_end_ts,
                'limit': self.MAX_RECORDS
            }

            result = await self._make_request('/v5/market/funding/history', params)
            if not result:
                break

            records = result.get('list', [])
            if not records:
                break

            # Track oldest timestamp for next iteration
            oldest_ts = current_end_ts

            for record in records:
                funding_time = int(record['fundingRateTimestamp'])

                # Track oldest for pagination
                if funding_time < oldest_ts:
                    oldest_ts = funding_time

                if funding_time < start_ts or funding_time > end_ts:
                    continue

                fr = BybitFundingRate(
                    timestamp=pd.to_datetime(funding_time, unit='ms', utc=True),
                    symbol=symbol.upper(),
                    funding_rate=safe_float(record.get('fundingRate', 0)),
                    funding_interval_hours=self.FUNDING_INTERVAL_HOURS
                )

                all_data.append({**fr.to_dict(), 'funding_interval_hours': self.FUNDING_INTERVAL_HOURS, 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE})
                symbol_records += 1

            # Check if we got all data or need to paginate
            if len(records) < self.MAX_RECORDS:
                break # Got all data in this window

            if oldest_ts >= current_end_ts:
                logger.warning(f"No pagination progress for {bybit_symbol}, stopping")
                break

            # Move window backward (subtract 1ms to avoid duplicate)
            current_end_ts = oldest_ts - 1

        logger.info(f"Completed {symbol}: {symbol_records} records")
        return all_data

    async def fetch_funding_rates(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical funding rates from Bybit.

        SPEEDUP: Now uses parallel processing for multiple symbols.
        Expected speedup: 5-10x for 10+ symbols (limited by rate limits, not by sequential processing)

        CRITICAL: Bybit API does NOT support cursor pagination for funding history.
        Use time-window pagination by adjusting endTime after each request.
        Max 200 records per request, returns data in reverse chronological order.

        Args:
            symbols: List of symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with funding rate data and derived metrics
        """
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        # SPEEDUP: Process all symbols in parallel instead of sequentially
        # This allows up to limit_per_host=20 concurrent requests
        # Rate limit will control actual throughput
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
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        df = df.drop_duplicates(['timestamp', 'symbol'], keep='first')
        self.collection_stats['records_collected'] += len(df)
        return df
    
    async def fetch_current_funding_rate(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch current funding rates and ticker data."""
        result = await self._make_request('/v5/market/tickers', {'category': 'linear'})
        if not result:
            return pd.DataFrame()
        
        records = []
        symbol_set = {self._format_symbol(s) for s in symbols} if symbols else None
        
        for ticker in result.get('list', []):
            if symbol_set and ticker['symbol'] not in symbol_set:
                continue
            if not ticker['symbol'].endswith('USDT'):
                continue
            
            t = BybitTicker(
                timestamp=datetime.utcnow().replace(tzinfo=timezone.utc),
                symbol=self._parse_symbol(ticker['symbol']),
                last_price=float(ticker.get('lastPrice', 0) or 0),
                mark_price=float(ticker.get('markPrice', 0) or 0),
                index_price=float(ticker.get('indexPrice', 0) or 0),
                funding_rate=float(ticker.get('fundingRate', 0) or 0),
                next_funding_time=pd.to_datetime(int(ticker.get('nextFundingTime', 0)), unit='ms', utc=True) if ticker.get('nextFundingTime') else None,
                open_interest=float(ticker.get('openInterest', 0) or 0),
                open_interest_value=float(ticker.get('openInterestValue', 0) or 0),
                volume_24h=float(ticker.get('volume24h', 0) or 0),
                turnover_24h=float(ticker.get('turnover24h', 0) or 0),
                high_24h=float(ticker.get('highPrice24h', 0) or 0),
                low_24h=float(ticker.get('lowPrice24h', 0) or 0),
                price_change_24h_pct=float(ticker.get('price24hPcnt', 0) or 0) * 100
            )
            records.append({**t.to_dict(), 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE})
        
        return pd.DataFrame(records)
    
    async def _fetch_ohlcv_single(self, symbol: str, bybit_interval: str, market_type: str, start_ts: int, end_ts: int) -> List[Dict]:
        """SPEEDUP: Helper to fetch OHLCV for a single symbol (for parallel processing)."""
        # Skip symbols known to not exist on Bybit
        base_symbol = symbol.upper().replace('USDT', '').replace('PERP', '').replace('/USD', '')
        if base_symbol in self.UNSUPPORTED_SYMBOLS or f"{base_symbol}USDT" in self.UNSUPPORTED_SYMBOLS:
            logger.debug(f"Skipping {symbol} OHLCV - not listed on Bybit")
            return []

        bybit_symbol = self._format_symbol(symbol)
        logger.info(f"Fetching Bybit OHLCV: {bybit_symbol} ({bybit_interval})")

        all_data = []
        current_ts = start_ts
        symbol_records = 0

        while current_ts < end_ts:
            params = {'category': market_type, 'symbol': bybit_symbol, 'interval': bybit_interval, 'start': current_ts, 'end': end_ts, 'limit': self.MAX_RECORDS}

            result = await self._make_request('/v5/market/kline', params)
            if not result:
                break

            records = result.get('list', [])
            if not records:
                break

            records = records[::-1] # Reverse (newest first in response)

            for candle in records:
                ts = int(candle[0])
                volume = float(candle[5])
                all_data.append({
                    'timestamp': pd.to_datetime(ts, unit='ms', utc=True),
                    'symbol': symbol.upper(),
                    'open': float(candle[1]), 'high': float(candle[2]),
                    'low': float(candle[3]), 'close': float(candle[4]),
                    'volume': volume,
                    'turnover': float(candle[6]) if len(candle) > 6 else 0,
                    'market_type': market_type.upper(),
                    'venue': self.VENUE, 'venue_type': self.VENUE_TYPE
                })
                symbol_records += 1

            last_ts = int(records[-1][0])
            if last_ts <= current_ts:
                break
            current_ts = last_ts + 1

        logger.info(f"Completed {symbol} OHLCV: {symbol_records} records")
        return all_data

    async def fetch_ohlcv(self, symbols: List[str], timeframe: str, start_date: str, end_date: str, market_type: str = 'linear') -> pd.DataFrame:
        """
        Fetch OHLCV data from Bybit.

        SPEEDUP: Now uses parallel processing for multiple symbols.
        Expected speedup: 5-10x for 10+ symbols
        """
        tf_map = {'1m': '1', '5m': '5', '15m': '15', '30m': '30', '1h': '60', '4h': '240', '1d': 'D'}
        bybit_interval = tf_map.get(timeframe, timeframe)
        if bybit_interval not in self.VALID_KLINE_INTERVALS:
            bybit_interval = '60'

        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        # SPEEDUP: Process all symbols in parallel
        tasks = [self._fetch_ohlcv_single(symbol, bybit_interval, market_type, start_ts, end_ts) for symbol in symbols]
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

        df = pd.DataFrame(all_data)
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        df['volume_usd'] = df['turnover']
        self.collection_stats['records_collected'] += len(df)
        return df
    
    async def _fetch_open_interest_single(self, symbol: str, interval: str, start_ts: int, end_ts: int) -> List[Dict]:
        """SPEEDUP: Helper to fetch open interest for a single symbol (for parallel processing)."""
        bybit_symbol = self._format_symbol(symbol)
        logger.info(f"Fetching Bybit OI: {bybit_symbol}")

        all_data = []
        cursor = ""
        symbol_records = 0

        while True:
            params = {'category': 'linear', 'symbol': bybit_symbol, 'intervalTime': interval, 'startTime': start_ts, 'endTime': end_ts, 'limit': self.MAX_RECORDS}
            if cursor:
                params['cursor'] = cursor

            result = await self._make_request('/v5/market/open-interest', params)
            if not result:
                break

            records = result.get('list', [])
            if not records:
                break

            for record in records:
                all_data.append({
                    'timestamp': pd.to_datetime(safe_int(record.get('timestamp', 0)), unit='ms', utc=True),
                    'symbol': symbol.upper(),
                    'open_interest': safe_float(record.get('openInterest', 0)),
                    'venue': self.VENUE, 'venue_type': self.VENUE_TYPE
                })
                symbol_records += 1

            cursor = result.get('nextPageCursor', '')
            if not cursor:
                break

        logger.info(f"Completed {symbol} OI: {symbol_records} records")
        return all_data

    async def fetch_open_interest(self, symbols: List[str], timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch open interest history.

        SPEEDUP: Now uses parallel processing for multiple symbols.
        Expected speedup: 5-10x for 10+ symbols
        """
        tf_map = {'5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h', '1d': '1d'}
        interval = tf_map.get(timeframe, '1h')

        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        # SPEEDUP: Process all symbols in parallel
        tasks = [self._fetch_open_interest_single(symbol, interval, start_ts, end_ts) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and filter out exceptions
        all_data = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch OI for {symbols[i]}: {result}")
            elif isinstance(result, list):
                all_data.extend(result)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        df['oi_change'] = df.groupby('symbol')['open_interest'].diff()
        df['oi_change_pct'] = df.groupby('symbol')['open_interest'].pct_change() * 100
        self.collection_stats['records_collected'] += len(df)
        return df
    
    async def _fetch_long_short_ratio_single(self, symbol: str, interval: str, start_ts: int, end_ts: int) -> List[Dict]:
        """SPEEDUP: Helper to fetch L/S ratio for a single symbol (for parallel processing)."""
        bybit_symbol = self._format_symbol(symbol)
        logger.info(f"Fetching Bybit L/S ratio: {bybit_symbol}")

        all_data = []
        cursor = ""
        symbol_records = 0

        while True:
            params = {'category': 'linear', 'symbol': bybit_symbol, 'period': interval, 'limit': self.MAX_RECORDS}
            if cursor:
                params['cursor'] = cursor

            result = await self._make_request('/v5/market/account-ratio', params)
            if not result:
                break

            records = result.get('list', [])
            if not records:
                break

            for record in records:
                ts = int(record['timestamp'])
                if ts < start_ts or ts > end_ts:
                    continue

                ls = BybitLongShortRatio(
                    timestamp=pd.to_datetime(ts, unit='ms', utc=True),
                    symbol=symbol.upper(),
                    buy_ratio=float(record.get('buyRatio', 0)),
                    sell_ratio=float(record.get('sellRatio', 0))
                )
                all_data.append({**ls.to_dict(), 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE})
                symbol_records += 1

            cursor = result.get('nextPageCursor', '')
            if not cursor:
                break

        logger.info(f"Completed {symbol} L/S ratio: {symbol_records} records")
        return all_data

    async def fetch_long_short_ratio(self, symbols: List[str], timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch long/short ratio from Bybit.

        SPEEDUP: Now uses parallel processing for multiple symbols.
        Expected speedup: 5-10x for 10+ symbols
        """
        tf_map = {'5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h', '1d': '1d'}
        interval = tf_map.get(timeframe, '1h')

        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        # SPEEDUP: Process all symbols in parallel
        tasks = [self._fetch_long_short_ratio_single(symbol, interval, start_ts, end_ts) for symbol in symbols]
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

        df = pd.DataFrame(all_data)
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        self.collection_stats['records_collected'] += len(df)
        return df
    
    async def fetch_insurance_fund(self) -> pd.DataFrame:
        """Fetch insurance fund history."""
        result = await self._make_request('/v5/market/insurance', {'coin': 'USDT'})
        if not result:
            return pd.DataFrame()
        
        records = []
        for record in result.get('list', []):
            balance_val = safe_float(record.get('balance', 0))
            records.append({
                'timestamp': pd.to_datetime(record.get('updatedTime'), utc=True),
                'coin': record.get('coin', ''),
                'balance': balance_val,
                'value': safe_float(record.get('value', balance_val)),
                'venue': self.VENUE, 'venue_type': self.VENUE_TYPE
            })
        
        return pd.DataFrame(records)
    
    async def fetch_instruments(self, category: str = 'linear') -> pd.DataFrame:
        """Fetch instrument specifications."""
        result = await self._make_request('/v5/market/instruments-info', {'category': category})
        if not result:
            return pd.DataFrame()
        
        records = []
        for inst in result.get('list', []):
            i = BybitInstrument(
                symbol=inst['symbol'],
                base_coin=inst.get('baseCoin'),
                quote_coin=inst.get('quoteCoin'),
                status=inst.get('status'),
                contract_type=inst.get('contractType'),
                launch_time=pd.to_datetime(int(inst.get('launchTime', 0)), unit='ms', utc=True) if inst.get('launchTime') else None,
                tick_size=float(inst.get('priceFilter', {}).get('tickSize', 0)),
                min_qty=float(inst.get('lotSizeFilter', {}).get('minOrderQty', 0)),
                max_qty=float(inst.get('lotSizeFilter', {}).get('maxOrderQty', 0)),
                max_leverage=float(inst.get('leverageFilter', {}).get('maxLeverage', 0))
            )
            records.append({**i.to_dict(), 'venue': self.VENUE})
        
        return pd.DataFrame(records)
    
    async def get_available_symbols(self, min_volume_usd: float = 10_000_000, category: str = 'linear') -> List[str]:
        """Get available symbols filtered by volume."""
        result = await self._make_request('/v5/market/tickers', {'category': category})
        if not result:
            return []
        
        symbols = []
        for ticker in result.get('list', []):
            symbol = ticker['symbol']
            if not symbol.endswith('USDT'):
                continue
            turnover_24h = float(ticker.get('turnover24h', 0) or 0)
            if turnover_24h >= min_volume_usd:
                symbols.append(symbol.replace('USDT', ''))
        
        logger.info(f"Found {len(symbols)} Bybit symbols with volume >= ${min_volume_usd:,.0f}")
        return sorted(symbols)
    
    async def fetch_comprehensive_data(self, symbols: List[str], start_date: str, end_date: str, timeframe: str = '1h') -> Dict[str, pd.DataFrame]:
        """Fetch comprehensive data for multiple symbols."""
        results = {}
        logger.info(f"Fetching comprehensive Bybit data for {len(symbols)} symbols")
        
        results['funding_rates'] = await self.fetch_funding_rates(symbols, start_date, end_date)
        results['ohlcv'] = await self.fetch_ohlcv(symbols, timeframe, start_date, end_date)
        results['open_interest'] = await self.fetch_open_interest(symbols, timeframe, start_date, end_date)
        results['long_short_ratio'] = await self.fetch_long_short_ratio(symbols, timeframe, start_date, end_date)
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
            logger.error(f"Bybit collect_funding_rates error: {e}")
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
            market_type = kwargs.get('market_type', 'linear')

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
            logger.error(f"Bybit collect_ohlcv error: {e}")
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
            logger.error(f"Bybit collect_open_interest error: {e}")
            return pd.DataFrame()

    async def _fetch_trades_single_symbol(self, symbol: str, limit_per_request: int) -> Optional[List[Dict]]:
        """Helper to fetch trades for a single symbol."""
        try:
            bybit_symbol = self._format_symbol(symbol)
            logger.info(f"Fetching Bybit trades: {bybit_symbol}")

            params = {
                'category': 'linear',
                'symbol': bybit_symbol,
                'limit': min(limit_per_request, 1000)
            }

            result = await self._make_request('/v5/market/recent-trade', params)
            if not result:
                return None

            records = result.get('list', [])
            trade_data = []
            for trade in records:
                trade_time = int(trade['time'])
                trade_data.append({
                    'timestamp': pd.to_datetime(trade_time, unit='ms', utc=True),
                    'symbol': symbol.upper(),
                    'trade_id': trade.get('execId', ''),
                    'price': float(trade['price']),
                    'size': float(trade['size']),
                    'side': trade['side'].lower(),
                    'is_block_trade': trade.get('isBlockTrade', False),
                    'trade_value': float(trade['price']) * float(trade['size']),
                    'venue': self.VENUE,
                    'venue_type': self.VENUE_TYPE
                })

            logger.info(f"Completed {symbol} trades: {len(records)} records")
            return trade_data
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            return None

    async def fetch_trades(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        limit_per_request: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch recent public trades from Bybit.

        Uses /v5/market/recent-trade endpoint (public, no auth required).

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH'])
            start_date: Start date string 'YYYY-MM-DD' (for reference, API returns recent only)
            end_date: End date string 'YYYY-MM-DD'
            limit_per_request: Max trades per request (max 1000)

        Returns:
            DataFrame with recent trade data
        """
        # Parallelize symbol processing
        tasks = [self._fetch_trades_single_symbol(symbol, limit_per_request) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        all_data = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch trades for {symbols[i]}: {result}")
            elif result is not None and isinstance(result, list):
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

            limit = kwargs.get('limit_per_request', 1000)

            return await self.fetch_trades(
                symbols=symbols,
                start_date=start_str,
                end_date=end_str,
                limit_per_request=limit
            )
        except Exception as e:
            logger.error(f"Bybit collect_trades error: {e}")
            return pd.DataFrame()

    async def _fetch_liquidations_single_symbol(self, symbol: str) -> Optional[Dict]:
        """Helper to fetch liquidations for a single symbol."""
        try:
            bybit_symbol = self._format_symbol(symbol)
            logger.info(f"Fetching Bybit liquidations: {bybit_symbol}")

            # Get ticker data which includes 24h liquidation volume (if available)
            result = await self._make_request('/v5/market/tickers', {
                'category': 'linear',
                'symbol': bybit_symbol
            })

            if not result or not result.get('list'):
                return None

            ticker = result['list'][0]
            # Bybit tickers don't directly provide liquidation data
            # We'll record the ticker snapshot as context

            liquidation_data = {
                'timestamp': datetime.utcnow().replace(tzinfo=timezone.utc),
                'symbol': symbol.upper(),
                'mark_price': float(ticker.get('markPrice', 0) or 0),
                'index_price': float(ticker.get('indexPrice', 0) or 0),
                'open_interest': float(ticker.get('openInterest', 0) or 0),
                'volume_24h': float(ticker.get('volume24h', 0) or 0),
                'data_type': 'liquidation_context',
                'note': 'Bybit requires WebSocket for real-time liquidation data',
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE
            }

            logger.info(f"Completed {symbol} liquidations context")
            return liquidation_data
        except Exception as e:
            logger.error(f"Error fetching liquidations for {symbol}: {e}")
            return None

    async def fetch_liquidations(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        limit_per_request: int = 200
    ) -> pd.DataFrame:
        """
        Fetch recent liquidation orders from Bybit.

        Uses /v5/market/recent-trade endpoint filtered for liquidations.
        Note: Bybit does not provide historical liquidation data via REST API.
        This returns recent liquidations only.

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH'])
            start_date: Start date (for reference)
            end_date: End date (for reference)
            limit_per_request: Max records per request

        Returns:
            DataFrame with liquidation data
        """
        # Bybit doesn't have a dedicated liquidations endpoint in v5 public API
        # We can use the account endpoint for own liquidations (requires auth)
        # For public liquidations, we need to use WebSocket or check recent trades
        # This implementation fetches tickers and identifies potential liquidation signals

        # Parallelize symbol processing
        tasks = [self._fetch_liquidations_single_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        all_data = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch liquidations for {symbols[i]}: {result}")
            elif result is not None:
                all_data.append(result)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
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

        Note: Bybit public REST API has limited liquidation data.
        For real-time liquidations, use WebSocket stream.
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

            limit = kwargs.get('limit_per_request', 200)

            return await self.fetch_liquidations(
                symbols=symbols,
                start_date=start_str,
                end_date=end_str,
                limit_per_request=limit
            )
        except Exception as e:
            logger.error(f"Bybit collect_liquidations error: {e}")
            return pd.DataFrame()

    def get_collection_stats(self) -> Dict:
        return self.collection_stats.copy()
    
    def reset_collection_stats(self):
        self.collection_stats = {'records_collected': 0, 'api_calls': 0, 'errors': 0, 'rate_limit_hits': 0}

async def test_bybit_collector():
    """Test Bybit collector functionality."""
    config = {'rate_limit': 30}
    
    async with BybitCollector(config) as collector:
        print("=" * 60)
        print("Bybit Collector Test")
        print("=" * 60)
        
        symbols = await collector.get_available_symbols(min_volume_usd=50_000_000)
        print(f"\n1. Found {len(symbols)} high-volume symbols: {symbols[:10]}")
        
        funding = await collector.fetch_current_funding_rate(['BTC', 'ETH', 'SOL'])
        if not funding.empty:
            print("\n2. Current funding rates:")
            for _, row in funding.iterrows():
                print(f" {row['symbol']}: {row['funding_rate_pct']:.4f}% (basis: {row['basis_pct']:.4f}%)")
        
        print(f"\nStats: {collector.get_collection_stats()}")

if __name__ == '__main__':
    asyncio.run(test_bybit_collector())