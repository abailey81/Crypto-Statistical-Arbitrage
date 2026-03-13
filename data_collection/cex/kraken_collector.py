"""
Kraken Exchange Data Collector - High Data Quality CEX

validated collector for Kraken spot and futures markets.
One of the oldest and most reliable cryptocurrency exchanges (founded 2011).

Supported Data Types:
    - Spot OHLCV with VWAP (since 2013 for BTC)
    - Kraken Futures funding rates (hourly settlement)
    - Order book snapshots (L2 data)
    - Trade history with timestamps
    - Asset pair specifications
    - System status

API Documentation:
    - Spot API: https://docs.kraken.com/rest/
    - Futures API: https://docs.futures.kraken.com/

Rate Limits:
    - Public: 1 request per second (burst 15)
    - Private: Tier-based (15-20 calls per minute)
    - Futures: 500 requests per 10 seconds

Contract Specifications (Kraken Futures):
    - Funding interval: 1 hour (continuous)
    - Contract sizes: BTC: 1 BTC, ETH: 1 ETH
    - Leverage: Up to 50x
    - Settlement: USD equivalent

Data Quality Highlights:
    - Extensive historical archives (10+ years)
    - Strong regulatory compliance (US/EU)
    - High uptime and reliability
    - VWAP included in OHLCV data

Statistical Arbitrage Applications:
    - Reference price validation (professional-quality)
    - Hourly vs 8-hour funding comparison
    - Cross-exchange spread monitoring
    - Historical volatility analysis
    - Liquidity benchmarking

Version: 2.0.0
"""

import asyncio
import aiohttp
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import hmac
import hashlib
import base64
import urllib.parse

from ..base_collector import BaseCollector
from ..utils.rate_limiter import get_shared_rate_limiter
from ..utils.retry_handler import RetryHandler, safe_float, safe_int

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================

class MarketType(Enum):
    """Kraken market types."""
    SPOT = 'spot'
    FUTURES = 'futures'
    MARGIN = 'margin'

class AssetStatus(Enum):
    """Asset trading status."""
    ENABLED = 'enabled'
    DEPOSIT_ONLY = 'deposit_only'
    WITHDRAWAL_ONLY = 'withdrawal_only'
    FUNDING_TEMPORARILY_DISABLED = 'funding_temporarily_disabled'

class FundingTrend(Enum):
    """Funding rate trend classification."""
    HIGHLY_POSITIVE = 'highly_positive'
    POSITIVE = 'positive'
    NEUTRAL = 'neutral'
    NEGATIVE = 'negative'
    HIGHLY_NEGATIVE = 'highly_negative'

class PriceTrend(Enum):
    """Price trend classification."""
    STRONG_UP = 'strong_up'
    UP = 'up'
    FLAT = 'flat'
    DOWN = 'down'
    STRONG_DOWN = 'strong_down'

class SpreadQuality(Enum):
    """Bid-ask spread quality classification."""
    EXCELLENT = 'excellent' # < 0.02%
    GOOD = 'good' # 0.02-0.05%
    FAIR = 'fair' # 0.05-0.1%
    POOR = 'poor' # 0.1-0.2%
    VERY_POOR = 'very_poor' # > 0.2%

class OrderBookDepth(Enum):
    """Order book depth classification."""
    DEEP = 'deep' # > $1M within 1%
    MODERATE = 'moderate' # $100K-$1M within 1%
    SHALLOW = 'shallow' # < $100K within 1%

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class KrakenFundingRate:
    """
    Kraken Futures funding rate data.
    
    Kraken uses continuous hourly funding (different from 8h perps).
    Rate is charged/paid every hour based on mark-index spread.
    """
    timestamp: datetime
    symbol: str
    funding_rate: float
    relative_funding_rate: float
    funding_interval_hours: int = 1
    
    @property
    def annualized_rate(self) -> float:
        """Annualized funding rate."""
        return self.funding_rate * 8760 # 24 * 365 hours
    
    @property
    def annualized_rate_pct(self) -> float:
        """Annualized rate as percentage."""
        return self.annualized_rate * 100
    
    @property
    def daily_rate(self) -> float:
        """Daily funding rate (24 payments)."""
        return self.funding_rate * 24
    
    @property
    def eight_hour_equivalent(self) -> float:
        """Equivalent 8-hour funding for comparison with perps."""
        return self.funding_rate * 8
    
    @property
    def trend(self) -> FundingTrend:
        """Classify funding trend."""
        rate_pct = self.funding_rate * 100
        if rate_pct > 0.01:
            return FundingTrend.HIGHLY_POSITIVE
        elif rate_pct > 0.002:
            return FundingTrend.POSITIVE
        elif rate_pct > -0.002:
            return FundingTrend.NEUTRAL
        elif rate_pct > -0.01:
            return FundingTrend.NEGATIVE
        else:
            return FundingTrend.HIGHLY_NEGATIVE
    
    @property
    def is_arbitrage_opportunity(self) -> bool:
        """Check if rate presents arbitrage opportunity."""
        return abs(self.eight_hour_equivalent) > 0.0003
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'funding_rate': self.funding_rate,
            'funding_rate_pct': self.funding_rate * 100,
            'relative_funding_rate': self.relative_funding_rate,
            'annualized_rate': self.annualized_rate,
            'annualized_rate_pct': self.annualized_rate_pct,
            'daily_rate': self.daily_rate,
            'eight_hour_equivalent': self.eight_hour_equivalent,
            'trend': self.trend.value,
            'is_arbitrage_opportunity': self.is_arbitrage_opportunity,
        }

@dataclass
class KrakenTicker:
    """Kraken ticker/market snapshot data."""
    timestamp: datetime
    symbol: str
    ask: float
    bid: float
    last: float
    volume_24h: float
    vwap_24h: float
    trades_24h: int
    low_24h: float
    high_24h: float
    open_24h: float
    
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
    def spread_quality(self) -> SpreadQuality:
        """Classify spread quality."""
        pct = self.spread_pct
        if pct < 0.02:
            return SpreadQuality.EXCELLENT
        elif pct < 0.05:
            return SpreadQuality.GOOD
        elif pct < 0.1:
            return SpreadQuality.FAIR
        elif pct < 0.2:
            return SpreadQuality.POOR
        else:
            return SpreadQuality.VERY_POOR
    
    @property
    def mid_price(self) -> float:
        """Mid price."""
        return (self.ask + self.bid) / 2
    
    @property
    def change_24h(self) -> float:
        """24h price change."""
        return self.last - self.open_24h
    
    @property
    def change_24h_pct(self) -> float:
        """24h price change percentage."""
        return (self.change_24h / self.open_24h * 100) if self.open_24h > 0 else 0
    
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
    
    @property
    def range_24h_pct(self) -> float:
        """24h range as percentage."""
        return ((self.high_24h - self.low_24h) / self.low_24h * 100) if self.low_24h > 0 else 0
    
    @property
    def vwap_deviation_pct(self) -> float:
        """Current price deviation from VWAP."""
        return ((self.last - self.vwap_24h) / self.vwap_24h * 100) if self.vwap_24h > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'ask': self.ask,
            'bid': self.bid,
            'last': self.last,
            'mid_price': self.mid_price,
            'spread': self.spread,
            'spread_pct': self.spread_pct,
            'spread_quality': self.spread_quality.value,
            'volume_24h': self.volume_24h,
            'vwap_24h': self.vwap_24h,
            'vwap_deviation_pct': self.vwap_deviation_pct,
            'trades_24h': self.trades_24h,
            'low_24h': self.low_24h,
            'high_24h': self.high_24h,
            'open_24h': self.open_24h,
            'change_24h': self.change_24h,
            'change_24h_pct': self.change_24h_pct,
            'price_trend': self.price_trend.value,
            'range_24h_pct': self.range_24h_pct,
        }

@dataclass
class KrakenCandle:
    """Kraken OHLCV candle with VWAP."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    vwap: float
    volume: float
    trades: int
    
    @property
    def typical_price(self) -> float:
        """Typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3
    
    @property
    def range_pct(self) -> float:
        """Candle range as percentage."""
        return ((self.high - self.low) / self.low * 100) if self.low > 0 else 0
    
    @property
    def body_pct(self) -> float:
        """Body as percentage of open."""
        return ((self.close - self.open) / self.open * 100) if self.open > 0 else 0
    
    @property
    def is_bullish(self) -> bool:
        """Check if bullish candle."""
        return self.close > self.open
    
    @property
    def vwap_position(self) -> float:
        """Close position relative to VWAP (-1 to 1)."""
        if self.high == self.low:
            return 0
        if self.vwap == 0:
            return 0
        return (self.close - self.vwap) / (self.high - self.low) if (self.high - self.low) > 0 else 0
    
    @property
    def volume_usd(self) -> float:
        """Estimated volume in USD."""
        return self.volume * self.vwap
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'vwap': self.vwap,
            'volume': self.volume,
            'volume_usd': self.volume_usd,
            'trades': self.trades,
            'typical_price': self.typical_price,
            'range_pct': self.range_pct,
            'body_pct': self.body_pct,
            'is_bullish': self.is_bullish,
            'vwap_position': self.vwap_position,
        }

@dataclass
class KrakenOrderBook:
    """Kraken order book snapshot."""
    timestamp: datetime
    symbol: str
    bids: List[Tuple[float, float, int]] # (price, volume, timestamp)
    asks: List[Tuple[float, float, int]]
    
    @property
    def best_bid(self) -> float:
        """Best bid price."""
        return self.bids[0][0] if self.bids else 0
    
    @property
    def best_ask(self) -> float:
        """Best ask price."""
        return self.asks[0][0] if self.asks else 0
    
    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.best_ask - self.best_bid if self.best_bid and self.best_ask else 0
    
    @property
    def spread_pct(self) -> float:
        """Spread as percentage."""
        mid = (self.best_ask + self.best_bid) / 2
        return (self.spread / mid * 100) if mid > 0 else 0
    
    @property
    def mid_price(self) -> float:
        """Mid price."""
        return (self.best_ask + self.best_bid) / 2 if self.best_bid and self.best_ask else 0
    
    @property
    def bid_depth_1pct(self) -> float:
        """Total bid volume within 1% of best bid."""
        threshold = self.best_bid * 0.99
        return sum(vol for price, vol, _ in self.bids if price >= threshold)
    
    @property
    def ask_depth_1pct(self) -> float:
        """Total ask volume within 1% of best ask."""
        threshold = self.best_ask * 1.01
        return sum(vol for price, vol, _ in self.asks if price <= threshold)
    
    @property
    def imbalance(self) -> float:
        """Order book imbalance (-1 to 1, positive = more bids)."""
        bid_vol = self.bid_depth_1pct
        ask_vol = self.ask_depth_1pct
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'mid_price': self.mid_price,
            'spread': self.spread,
            'spread_pct': self.spread_pct,
            'bid_depth_1pct': self.bid_depth_1pct,
            'ask_depth_1pct': self.ask_depth_1pct,
            'imbalance': self.imbalance,
            'bid_levels': len(self.bids),
            'ask_levels': len(self.asks),
        }

@dataclass
class KrakenAssetPair:
    """Kraken trading pair specification."""
    pair: str
    altname: str
    wsname: str
    base: str
    quote: str
    pair_decimals: int
    lot_decimals: int
    maker_fee: float
    taker_fee: float
    status: str
    
    @property
    def is_usd_pair(self) -> bool:
        """Check if USD quote."""
        return self.quote in ['ZUSD', 'USD']
    
    @property
    def is_tradeable(self) -> bool:
        """Check if pair is tradeable."""
        return self.status == 'online'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'pair': self.pair,
            'altname': self.altname,
            'wsname': self.wsname,
            'base': self.base,
            'quote': self.quote,
            'pair_decimals': self.pair_decimals,
            'lot_decimals': self.lot_decimals,
            'maker_fee': self.maker_fee,
            'taker_fee': self.taker_fee,
            'status': self.status,
            'is_usd_pair': self.is_usd_pair,
            'is_tradeable': self.is_tradeable,
        }

# =============================================================================
# Collector Class
# =============================================================================

class KrakenCollector(BaseCollector):
    """
    Kraken exchange data collector.
    
    validated implementation for institutional-quality data.
    
    Features:
    - Long historical OHLCV with VWAP (since 2013)
    - Hourly funding rates (Kraken Futures)
    - Order book snapshots
    - Trade history
    - Asset pair specifications
    
    Attributes:
        VENUE: Exchange identifier ('kraken')
        VENUE_TYPE: Exchange type ('CEX')
        SPOT_URL: Spot API endpoint
        FUTURES_URL: Futures API endpoint
        FUNDING_INTERVAL_HOURS: Funding payment interval (1)
    
    Example:
        >>> config = {'rate_limit': 30}
        >>> async with KrakenCollector(config) as collector:
        ... ohlcv = await collector.fetch_ohlcv(['BTC', 'ETH'], '1h', '2024-01-01', '2024-01-31')
        ... ticker = await collector.fetch_ticker(['BTC', 'ETH'])
    """
    
    VENUE = 'kraken'
    VENUE_TYPE = 'CEX'
    
    SPOT_URL = 'https://api.kraken.com/0/public'
    FUTURES_URL = 'https://futures.kraken.com/derivatives/api/v3'
    
    FUNDING_INTERVAL_HOURS = 1
    
    # Kraken pair mapping (legacy X/Z format)
    PAIR_MAP = {
        'BTC': 'XXBTZUSD', 'ETH': 'XETHZUSD', 'SOL': 'SOLUSD', 'AVAX': 'AVAXUSD',
        'LINK': 'LINKUSD', 'DOT': 'DOTUSD', 'ATOM': 'ATOMUSD', 'UNI': 'UNIUSD',
        'AAVE': 'AAVEUSD', 'ARB': 'ARBUSD', 'OP': 'OPUSD', 'MATIC': 'MATICUSD'
    }
    
    FUTURES_PAIRS = {'BTC': 'PI_XBTUSD', 'ETH': 'PI_ETHUSD'}

    # Symbols truly not available on Kraken (minimal list - let API reject borderline cases)
    UNSUPPORTED_SYMBOLS = {
        'FXS',  # Not on Kraken
    }

    VALID_INTERVALS = {1, 5, 15, 30, 60, 240, 1440, 10080, 21600}
    
    def __init__(self, config: Dict):
        """Initialize Kraken collector."""
        super().__init__(config)
        self.api_key = config.get('api_key') or config.get('kraken_api_key') or os.getenv('KRAKEN_API_KEY', '')
        self.api_secret = config.get('api_secret') or config.get('kraken_api_secret') or os.getenv('KRAKEN_PRIVATE_KEY', '')
        
        rate_limit = config.get('rate_limit', 30)
        # Use shared rate limiter to avoid re-initialization overhead
        self.rate_limiter = get_shared_rate_limiter('kraken', rate=rate_limit, per=60.0, burst=8)
        self.retry_handler = RetryHandler(max_retries=3, base_delay=2.0, max_delay=30.0)
        
        self.timeout = aiohttp.ClientTimeout(total=config.get('timeout', 30))
        self.session: Optional[aiohttp.ClientSession] = None
        
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl = config.get('cache_ttl', 60)
        
        # Semaphore to limit concurrent HTTP requests (prevents rate limit exhaustion)
        self._request_sem = asyncio.Semaphore(2)

        self.collection_stats = {'records_collected': 0, 'api_calls': 0, 'errors': 0, 'cache_hits': 0}
        logger.info(f"Initialized Kraken collector (rate_limit={rate_limit}/min)")
    
    async def __aenter__(self):
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            # SPEEDUP: Aggressive connection pooling for high-throughput collection
            # Kraken rate limit: 15/sec public, higher pool supports concurrent symbols
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
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
        self._cache.clear()
        logger.info(f"Kraken collector closed. Stats: {self.collection_stats}")
    
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
    
    def _get_kraken_pair(self, symbol: str) -> str:
        """Convert standard symbol to Kraken format."""
        symbol = symbol.upper().replace('/USD', '').replace('USD', '')
        return self.PAIR_MAP.get(symbol, f"{symbol}USD")
    
    def _parse_symbol(self, kraken_pair: str) -> str:
        """Parse Kraken pair to standard symbol."""
        for symbol, pair in self.PAIR_MAP.items():
            if pair == kraken_pair:
                return symbol
        return kraken_pair.replace('USD', '').replace('ZUSD', '').replace('XX', '').replace('X', '')
    
    async def _public_request(self, endpoint: str, params: Dict = None, base_url: str = None, use_cache: bool = False) -> Optional[Dict]:
        """Make public API request."""
        async with self._request_sem:
            if use_cache:
                cache_key = f"{endpoint}_{hash(frozenset((params or {}).items()))}"
                cached = self._get_cached(cache_key)
                if cached is not None:
                    return cached

            url = f"{base_url or self.SPOT_URL}/{endpoint}"
            session = await self._get_session()

            async def _request():
                await self.rate_limiter.acquire()
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('error') and len(data['error']) > 0:
                            logger.error(f"Kraken API error: {data['error']}")
                            return None
                        self.collection_stats['api_calls'] += 1
                        return data.get('result', data)
                    logger.error(f"Kraken HTTP {resp.status}")
                    return None

            try:
                result = await self.retry_handler.execute(_request)
                if use_cache and result is not None:
                    self._set_cached(cache_key, result)
                return result
            except Exception as e:
                logger.error(f"Kraken request error: {e}")
                self.collection_stats['errors'] += 1
                return None
    
    async def _futures_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make Kraken Futures API request."""
        async with self._request_sem:
            url = f"{self.FUTURES_URL}/{endpoint}"
            session = await self._get_session()

            async def _request():
                await self.rate_limiter.acquire()
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('result') == 'error':
                            logger.error(f"Kraken Futures error: {data.get('error')}")
                            return None
                        self.collection_stats['api_calls'] += 1
                        return data
                    return None

            try:
                return await self.retry_handler.execute(_request)
            except Exception as e:
                logger.error(f"Kraken Futures error: {e}")
                self.collection_stats['errors'] += 1
                return None
    
    async def _fetch_ohlcv_single(self, symbol: str, interval: int, start_ts: int, end_ts: int) -> List[Dict]:
        """SPEEDUP: Helper to fetch OHLCV for a single symbol (for parallel processing)."""
        # Skip symbols that don't have USD pairs on Kraken
        base_symbol = symbol.upper().replace('/USD', '').replace('USD', '')
        if base_symbol in self.UNSUPPORTED_SYMBOLS:
            logger.debug(f"Skipping {symbol} - not listed on Kraken")
            return []

        pair = self._get_kraken_pair(symbol)
        logger.info(f"Fetching Kraken OHLCV for {symbol} ({pair})")

        params = {'pair': pair, 'interval': interval, 'since': start_ts}
        data = await self._public_request('OHLC', params)

        if not data:
            return []

        all_data = []
        ohlcv = data.get(pair, data.get(list(data.keys())[0] if data else [], []))
        symbol_records = 0

        for candle in ohlcv:
            if len(candle) < 7:
                continue

            ts = int(candle[0])
            if ts > end_ts:
                break

            c = KrakenCandle(
                timestamp=pd.to_datetime(ts, unit='s', utc=True),
                symbol=symbol.upper(),
                open=float(candle[1]),
                high=float(candle[2]),
                low=float(candle[3]),
                close=float(candle[4]),
                vwap=float(candle[5]),
                volume=float(candle[6]),
                trades=int(candle[7]) if len(candle) > 7 else 0
            )

            all_data.append({**c.to_dict(), 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE})
            symbol_records += 1

        logger.info(f"Completed {symbol}: {symbol_records} records")
        return all_data

    async def fetch_ohlcv(self, symbols: List[str], timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch OHLCV data with VWAP from Kraken.

        SPEEDUP: Now uses parallel processing for multiple symbols.
        Expected speedup: 3-5x for 10+ symbols

        Args:
            symbols: List of symbols
            timeframe: '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data including VWAP
        """
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        interval_map = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440, '1w': 10080}
        interval = interval_map.get(timeframe, 60)

        # SPEEDUP: Process all symbols in parallel
        tasks = [self._fetch_ohlcv_single(symbol, interval, start_ts, end_ts) for symbol in symbols]
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
        self.collection_stats['records_collected'] += len(df)
        return df
    
    async def _fetch_funding_rate_single(self, symbol: str, start_ts: int, end_ts: int) -> List[Dict]:
        """SPEEDUP: Helper to fetch funding rates for a single symbol (for parallel processing)."""
        base = symbol.split('/')[0] if '/' in symbol else symbol.upper()
        futures_symbol = self.FUTURES_PAIRS.get(base)

        if not futures_symbol:
            logger.warning(f"No Kraken futures for {symbol}")
            return []

        logger.info(f"Fetching Kraken funding rates for {symbol}")

        params = {'symbol': futures_symbol}
        data = await self._futures_request('historical-funding-rates', params)

        if not data or 'rates' not in data:
            return []

        all_data = []
        symbol_records = 0
        for rate in data.get('rates', []):
            ts_str = rate.get('timestamp', '')
            if not ts_str:
                continue
            # Parse ISO format timestamp (e.g., "2018-08-31T16:00:00Z")
            ts_dt = pd.to_datetime(ts_str, utc=True)
            ts_unix = int(ts_dt.timestamp() * 1000)
            if ts_unix < start_ts or ts_unix > end_ts:
                continue

            fr = KrakenFundingRate(
                timestamp=ts_dt,
                symbol=base,
                funding_rate=safe_float(rate.get('fundingRate', 0)),
                relative_funding_rate=safe_float(rate.get('relativeFundingRate', 0)),
                funding_interval_hours=self.FUNDING_INTERVAL_HOURS
            )

            all_data.append({**fr.to_dict(), 'funding_interval_hours': self.FUNDING_INTERVAL_HOURS, 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE})
            symbol_records += 1

        logger.info(f"Completed {symbol}: {symbol_records} records")
        return all_data

    async def fetch_funding_rates(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch funding rates from Kraken Futures.

        SPEEDUP: Now uses parallel processing for multiple symbols.
        Expected speedup: 2-3x (limited by number of available futures symbols)

        Note: Kraken Futures uses hourly funding (different from 8h perps).

        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with funding rate data
        """
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp()) * 1000
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp()) * 1000

        # SPEEDUP: Process all symbols in parallel
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
        self.collection_stats['records_collected'] += len(df)
        return df
    
    async def fetch_ticker(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch current ticker data.
        
        Args:
            symbols: List of symbols
            
        Returns:
            DataFrame with ticker data and derived metrics
        """
        pairs = [self._get_kraken_pair(s) for s in symbols]
        params = {'pair': ','.join(pairs)}
        
        data = await self._public_request('Ticker', params, use_cache=True)
        if not data:
            return pd.DataFrame()
        
        records = []
        for pair, ticker in data.items():
            symbol = None
            for s in symbols:
                if self._get_kraken_pair(s) == pair:
                    symbol = s
                    break
            
            t = KrakenTicker(
                timestamp=datetime.utcnow().replace(tzinfo=timezone.utc),
                symbol=symbol or self._parse_symbol(pair),
                ask=float(ticker['a'][0]),
                bid=float(ticker['b'][0]),
                last=float(ticker['c'][0]),
                volume_24h=float(ticker['v'][1]),
                vwap_24h=float(ticker['p'][1]),
                trades_24h=int(ticker['t'][1]),
                low_24h=float(ticker['l'][1]),
                high_24h=float(ticker['h'][1]),
                open_24h=float(ticker['o'])
            )
            
            records.append({**t.to_dict(), 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE})
        
        return pd.DataFrame(records)
    
    async def fetch_orderbook(self, symbol: str, depth: int = 100) -> Dict:
        """
        Fetch current order book snapshot.
        
        Args:
            symbol: Symbol to fetch
            depth: Number of levels (default 100)
            
        Returns:
            Order book data dictionary
        """
        pair = self._get_kraken_pair(symbol)
        params = {'pair': pair, 'count': depth}
        
        data = await self._public_request('Depth', params)
        if not data:
            return {}
        
        book = data.get(pair, data.get(list(data.keys())[0] if data else {}, {}))
        
        bids = [(float(b[0]), float(b[1]), int(b[2])) for b in book.get('bids', [])]
        asks = [(float(a[0]), float(a[1]), int(a[2])) for a in book.get('asks', [])]
        
        ob = KrakenOrderBook(
            timestamp=datetime.utcnow().replace(tzinfo=timezone.utc),
            symbol=symbol,
            bids=bids,
            asks=asks
        )
        
        return {**ob.to_dict(), 'bids': bids, 'asks': asks, 'venue': self.VENUE}
    
    async def fetch_asset_pairs(self) -> pd.DataFrame:
        """Fetch all available trading pairs."""
        data = await self._public_request('AssetPairs', use_cache=True)
        if not data:
            return pd.DataFrame()
        
        records = []
        for pair_name, pair_info in data.items():
            ap = KrakenAssetPair(
                pair=pair_name,
                altname=pair_info.get('altname'),
                wsname=pair_info.get('wsname'),
                base=pair_info.get('base'),
                quote=pair_info.get('quote'),
                pair_decimals=pair_info.get('pair_decimals', 0),
                lot_decimals=pair_info.get('lot_decimals', 0),
                maker_fee=float(pair_info.get('fees_maker', [[0, 0]])[0][1]) / 100,
                taker_fee=float(pair_info.get('fees', [[0, 0]])[0][1]) / 100,
                status=pair_info.get('status', 'online')
            )
            records.append({**ap.to_dict(), 'venue': self.VENUE})
        
        return pd.DataFrame(records)
    
    async def fetch_futures_instruments(self) -> pd.DataFrame:
        """Fetch available futures instruments."""
        data = await self._futures_request('instruments')
        if not data or 'instruments' not in data:
            return pd.DataFrame()
        
        records = []
        for inst in data.get('instruments', []):
            records.append({
                'symbol': inst.get('symbol'),
                'type': inst.get('type'),
                'underlying': inst.get('underlying'),
                'tick_size': safe_float(inst.get('tickSize', 0)),
                'contract_size': safe_float(inst.get('contractSize', 0)),
                'tradeable': inst.get('tradeable'),
                'funding_rate_coeff': safe_float(inst.get('fundingRateCoefficient', 0)),
                'max_position': safe_float(inst.get('maxPositionSize', 0)),
                'venue': self.VENUE
            })
        
        return pd.DataFrame(records)
    
    async def get_available_symbols(self, min_volume_usd: float = 1_000_000) -> List[str]:
        """Get available USD trading pairs."""
        pairs = await self.fetch_asset_pairs()
        if pairs.empty:
            return list(self.PAIR_MAP.keys())
        
        usd_pairs = pairs[pairs['is_usd_pair'] & pairs['is_tradeable']]['altname'].tolist()
        return sorted([p.replace('USD', '') for p in usd_pairs if p.endswith('USD')])[:30]
    
    async def fetch_comprehensive_data(self, symbols: List[str], start_date: str, end_date: str, timeframe: str = '1h') -> Dict[str, pd.DataFrame]:
        """Fetch comprehensive Kraken data."""
        results = {}
        logger.info(f"Fetching comprehensive Kraken data for {len(symbols)} symbols")
        
        results['ohlcv'] = await self.fetch_ohlcv(symbols, timeframe, start_date, end_date)
        results['funding_rates'] = await self.fetch_funding_rates(symbols, start_date, end_date)
        results['ticker'] = await self.fetch_ticker(symbols)
        results['asset_pairs'] = await self.fetch_asset_pairs()
        
        return results

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
                end_date=end_str
            )
        except Exception as e:
            logger.error(f"Kraken collect_ohlcv error: {e}")
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
            logger.error(f"Kraken collect_funding_rates error: {e}")
            return pd.DataFrame()

    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.collection_stats.copy()

async def test_kraken_collector():
    """Test Kraken collector."""
    config = {'rate_limit': 30}
    
    async with KrakenCollector(config) as collector:
        print("=" * 60)
        print("Kraken Collector Test")
        print("=" * 60)
        
        ticker = await collector.fetch_ticker(['BTC', 'ETH'])
        if not ticker.empty:
            print("\n1. Current prices:")
            for _, row in ticker.iterrows():
                print(f" {row['symbol']}: ${row['last']:,.2f} (spread: {row['spread_pct']:.4f}%)")
        
        print(f"\nStats: {collector.get_collection_stats()}")

if __name__ == '__main__':
    asyncio.run(test_kraken_collector())