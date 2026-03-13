"""
CryptoCompare Market Data Collector - Aggregated Multi-Exchange Data

validated collector for aggregated cryptocurrency market data
across exchanges. Provides price validation, historical data, and
exchange volume metrics.

===============================================================================
OVERVIEW
===============================================================================

CryptoCompare aggregates data from 250+ exchanges to provide:
    - Aggregated price feeds (CCCAGG)
    - Exchange-specific data
    - Historical OHLCV (daily, hourly, minute)
    - Trading pair information
    - Social and on-chain metrics

The CCCAGG (CryptoCompare Aggregate) index is a volume-weighted average
price calculated from constituent exchanges, providing a reference price
that's less susceptible to single-exchange manipulation.

API Tiers:
    ============== ==================== ================ ================
    Tier Rate Limit Monthly Calls Best For
    ============== ==================== ================ ================
    Free 50 calls/second 100,000 Development
    Starter 200 calls/second 500,000 Small projects
    Professional 200 calls/second 2,500,000 Production
    Enterprise Custom Custom High-volume
    ============== ==================== ================ ================

===============================================================================
DATA TYPES COLLECTED
===============================================================================

Price Data:
    - Current prices (single and multi-currency)
    - CCCAGG aggregated prices
    - Exchange-specific prices
    - Price comparisons across exchanges

Historical OHLCV:
    - Daily candles (unlimited history)
    - Hourly candles (up to 2000 points)
    - Minute candles (up to 2000 points)
    - Volume in base and quote currencies

Exchange Data:
    - Top exchanges by volume
    - Exchange-specific OHLCV
    - Trading pair availability
    - Exchange trust/grade scores

Social & On-Chain:
    - Social media metrics
    - Code repository activity
    - On-chain transaction data
    - Network statistics

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

Price Validation:
    - Compare exchange prices to CCCAGG reference
    - Identify exchange-specific pricing anomalies
    - Detect potential arbitrage opportunities

Cross-Exchange Analysis:
    - Volume distribution across exchanges
    - Spread analysis between venues
    - Liquidity profiling

Historical Analysis:
    - Long-term price series for backtesting
    - Volume pattern analysis
    - Correlation studies

Data Enrichment:
    - Fill gaps in primary exchange data
    - Cross-reference multiple data sources
    - Validate data quality

===============================================================================
USAGE EXAMPLES
===============================================================================

Basic price fetch:

    >>> from data_collection.market_data import CryptoCompareCollector
    >>> 
    >>> collector = CryptoCompareCollector({'api_key': 'your-key'})
    >>> try:
    ... prices = await collector.get_price(['BTC', 'ETH'], ['USD', 'EUR'])
    ... print(f"BTC/USD: ${prices['BTC']['USD']:,.2f}")
    ... finally:
    ... await collector.close()

Historical daily data:

    >>> df = await collector.get_historical_daily(
    ... symbol='BTC',
    ... currency='USD',
    ... limit=365
    ... )

Exchange volume comparison:

    >>> top_exchanges = await collector.get_top_exchanges(
    ... symbol='BTC',
    ... currency='USD',
    ... limit=10
    ... )

Multi-symbol OHLCV:

    >>> ohlcv = await collector.fetch_ohlcv(
    ... symbols=['BTC', 'ETH', 'SOL'],
    ... timeframe='1d',
    ... start_date='2023-01-01',
    ... end_date='2024-01-01'
    ... )

===============================================================================
DATA QUALITY CONSIDERATIONS
===============================================================================

- CCCAGG prices may differ from specific exchange prices
- Volume data may include wash trading from some exchanges
- Historical data availability varies by trading pair
- Exchange trust scores are CryptoCompare's proprietary metric
- Minute data limited to recent history

Version: 2.0.0
API Documentation: https://min-api.cryptocompare.com/documentation
"""

import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..base_collector import BaseCollector
from ..utils.rate_limiter import get_shared_rate_limiter
from ..utils.retry_handler import RetryHandler

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================

class DataSource(Enum):
    """Data source/aggregation type."""
    CCCAGG = 'CCCAGG' # CryptoCompare aggregated
    BINANCE = 'Binance'
    COINBASE = 'Coinbase'
    KRAKEN = 'Kraken'
    BITSTAMP = 'Bitstamp'
    BITFINEX = 'Bitfinex'

class Timeframe(Enum):
    """OHLCV timeframe options."""
    MINUTE = 'minute'
    HOUR = 'hour'
    DAY = 'day'

class QuoteCurrency(Enum):
    """Common quote currencies."""
    USD = 'USD'
    EUR = 'EUR'
    GBP = 'GBP'
    JPY = 'JPY'
    BTC = 'BTC'
    ETH = 'ETH'
    USDT = 'USDT'
    USDC = 'USDC'

class ExchangeGrade(Enum):
    """CryptoCompare exchange grade."""
    AA = 'AA'
    A = 'A'
    BB = 'BB'
    B = 'B'
    CC = 'CC'
    C = 'C'
    D = 'D'
    E = 'E'

class VolumeLevel(Enum):
    """Volume classification."""
    VERY_HIGH = 'very_high' # > $1B 24h
    HIGH = 'high' # $100M - $1B
    MEDIUM = 'medium' # $10M - $100M
    LOW = 'low' # $1M - $10M
    VERY_LOW = 'very_low' # < $1M

class PriceTrend(Enum):
    """Price trend classification."""
    STRONG_UP = 'strong_up'
    UP = 'up'
    NEUTRAL = 'neutral'
    DOWN = 'down'
    STRONG_DOWN = 'strong_down'

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class CCPrice:
    """CryptoCompare price data."""
    symbol: str
    currency: str
    price: float
    source: str = 'CCCAGG'
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def pair(self) -> str:
        """Trading pair string."""
        return f"{self.symbol}/{self.currency}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'currency': self.currency,
            'pair': self.pair,
            'price': self.price,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'venue': 'cryptocompare',
        }

@dataclass
class CCOHLCVCandle:
    """CryptoCompare OHLCV candle data."""
    timestamp: datetime
    symbol: str
    currency: str
    open: float
    high: float
    low: float
    close: float
    volume: float # Volume in base currency
    volume_quote: float # Volume in quote currency
    source: str = 'CCCAGG'
    
    @property
    def pair(self) -> str:
        """Trading pair string."""
        return f"{self.symbol}/{self.currency}"
    
    @property
    def typical_price(self) -> float:
        """Typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3
    
    @property
    def range_pct(self) -> float:
        """Price range as percentage."""
        return (self.high - self.low) / self.open * 100 if self.open > 0 else 0
    
    @property
    def return_pct(self) -> float:
        """Period return percentage."""
        return (self.close - self.open) / self.open * 100 if self.open > 0 else 0
    
    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish."""
        return self.close > self.open
    
    @property
    def body_pct(self) -> float:
        """Body as percentage of range."""
        range_val = self.high - self.low
        if range_val == 0:
            return 0
        return abs(self.close - self.open) / range_val * 100
    
    @property
    def vwap(self) -> float:
        """Volume-weighted average price estimate."""
        return self.volume_quote / self.volume if self.volume > 0 else self.typical_price
    
    @property
    def trend(self) -> PriceTrend:
        """Classify price trend."""
        ret = self.return_pct
        if ret > 5:
            return PriceTrend.STRONG_UP
        elif ret > 1:
            return PriceTrend.UP
        elif ret > -1:
            return PriceTrend.NEUTRAL
        elif ret > -5:
            return PriceTrend.DOWN
        else:
            return PriceTrend.STRONG_DOWN
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'currency': self.currency,
            'pair': self.pair,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'volume_quote': self.volume_quote,
            'vwap': self.vwap,
            'typical_price': self.typical_price,
            'range_pct': self.range_pct,
            'return_pct': self.return_pct,
            'is_bullish': self.is_bullish,
            'trend': self.trend.value,
            'source': self.source,
            'venue': 'cryptocompare',
            'venue_type': 'AGGREGATOR',
        }

@dataclass
class ExchangeVolume:
    """Exchange volume data for a trading pair."""
    exchange: str
    symbol: str
    currency: str
    volume_24h: float # 24h volume in base
    volume_24h_quote: float # 24h volume in quote
    price: float # Current price on exchange
    grade: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def pair(self) -> str:
        """Trading pair string."""
        return f"{self.symbol}/{self.currency}"
    
    @property
    def volume_level(self) -> VolumeLevel:
        """Classify volume level."""
        vol = self.volume_24h_quote
        if vol > 1_000_000_000:
            return VolumeLevel.VERY_HIGH
        elif vol > 100_000_000:
            return VolumeLevel.HIGH
        elif vol > 10_000_000:
            return VolumeLevel.MEDIUM
        elif vol > 1_000_000:
            return VolumeLevel.LOW
        else:
            return VolumeLevel.VERY_LOW
    
    @property
    def exchange_grade_enum(self) -> Optional[ExchangeGrade]:
        """Exchange grade as enum."""
        if self.grade:
            try:
                return ExchangeGrade(self.grade)
            except ValueError:
                return None
        return None
    
    @property
    def is_tier1_exchange(self) -> bool:
        """Check if exchange is tier 1 (AA or A grade)."""
        return self.grade in ['AA', 'A']
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'exchange': self.exchange,
            'symbol': self.symbol,
            'currency': self.currency,
            'pair': self.pair,
            'volume_24h': self.volume_24h,
            'volume_24h_quote': self.volume_24h_quote,
            'price': self.price,
            'grade': self.grade,
            'volume_level': self.volume_level.value,
            'is_tier1_exchange': self.is_tier1_exchange,
            'timestamp': self.timestamp.isoformat(),
            'venue': 'cryptocompare',
        }

@dataclass
class CrossExchangeComparison:
    """Cross-exchange price comparison."""
    symbol: str
    currency: str
    cccagg_price: float
    exchange_prices: Dict[str, float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def pair(self) -> str:
        """Trading pair string."""
        return f"{self.symbol}/{self.currency}"
    
    @property
    def exchanges(self) -> List[str]:
        """List of exchanges."""
        return list(self.exchange_prices.keys())
    
    @property
    def deviations_from_cccagg(self) -> Dict[str, float]:
        """Price deviation from CCCAGG for each exchange (%)."""
        return {
            ex: (price - self.cccagg_price) / self.cccagg_price * 100
            for ex, price in self.exchange_prices.items()
        }
    
    @property
    def max_deviation_pct(self) -> float:
        """Maximum price deviation from CCCAGG."""
        devs = self.deviations_from_cccagg
        return max(abs(d) for d in devs.values()) if devs else 0
    
    @property
    def price_spread_pct(self) -> float:
        """Price spread across exchanges (max-min)/min."""
        prices = list(self.exchange_prices.values())
        if len(prices) < 2:
            return 0
        return (max(prices) - min(prices)) / min(prices) * 100
    
    @property
    def has_arbitrage_opportunity(self) -> bool:
        """Check if spread suggests arbitrage (> 0.1%)."""
        return self.price_spread_pct > 0.1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'currency': self.currency,
            'pair': self.pair,
            'cccagg_price': self.cccagg_price,
            'exchanges': self.exchanges,
            'max_deviation_pct': self.max_deviation_pct,
            'price_spread_pct': self.price_spread_pct,
            'has_arbitrage_opportunity': self.has_arbitrage_opportunity,
            'deviations': self.deviations_from_cccagg,
            'timestamp': self.timestamp.isoformat(),
        }

@dataclass
class SocialStats:
    """Social media and community metrics."""
    coin_id: int
    symbol: str
    twitter_followers: int
    reddit_subscribers: int
    reddit_active_users: int
    github_stars: int
    github_forks: int
    github_contributors: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def social_score(self) -> float:
        """Simple social score combining metrics."""
        return (
            self.twitter_followers / 1_000_000 +
            self.reddit_subscribers / 100_000 +
            self.github_stars / 10_000
        )
    
    @property
    def developer_activity(self) -> float:
        """Developer activity score."""
        return self.github_contributors + self.github_forks / 10
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'coin_id': self.coin_id,
            'symbol': self.symbol,
            'twitter_followers': self.twitter_followers,
            'reddit_subscribers': self.reddit_subscribers,
            'reddit_active_users': self.reddit_active_users,
            'github_stars': self.github_stars,
            'github_forks': self.github_forks,
            'github_contributors': self.github_contributors,
            'social_score': self.social_score,
            'developer_activity': self.developer_activity,
            'timestamp': self.timestamp.isoformat(),
            'venue': 'cryptocompare',
        }

# =============================================================================
# Main Collector Class
# =============================================================================

class CryptoCompareCollector(BaseCollector):
    """
    CryptoCompare market data collector.
    
    validated implementation for aggregated cryptocurrency
    market data with cross-exchange validation capabilities.
    
    Features:
        - CCCAGG aggregated price feeds
        - Historical OHLCV (daily, hourly, minute)
        - Exchange-specific data
        - Volume and liquidity metrics
        - Social and on-chain data
        - Automatic rate limiting
    
    Example:
        >>> config = {'api_key': 'your-api-key'}
        >>> collector = CryptoCompareCollector(config)
        >>> try:
        ... prices = await collector.get_price(['BTC', 'ETH'])
        ... ohlcv = await collector.get_historical_daily('BTC', limit=365)
        ... finally:
        ... await collector.close()
    
    Attributes:
        VENUE: 'cryptocompare'
        VENUE_TYPE: 'AGGREGATOR'
    """
    
    VENUE = 'cryptocompare'
    VENUE_TYPE = 'AGGREGATOR'
    BASE_URL = 'https://min-api.cryptocompare.com/data'
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize CryptoCompare collector.
        
        Args:
            config: Configuration with options:
                - api_key: API key (optional but recommended)
                - rate_limit: Requests per second (default: 25)
                - max_retries: Retry attempts (default: 3)
        """
        config = config or {}
        super().__init__(config)

        self.api_key = config.get('api_key') or config.get('cryptocompare_api_key') or os.getenv('CRYPTOCOMPARE_API_KEY', '')
        
        self.rate_limiter = get_shared_rate_limiter(
            'cryptocompare',
            rate=config.get('rate_limit', 25),
            per=1.0,
            burst=config.get('burst', 15)
        )
        
        self.retry_handler = RetryHandler(
            max_retries=config.get('max_retries', 3),
            base_delay=config.get('retry_delay', 1.0)
        )
        
        self.session: Optional[aiohttp.ClientSession] = None
        
        self.stats = {'requests': 0, 'records': 0, 'errors': 0}
        
        tier = 'Authenticated' if self.api_key else 'Free'
        logger.info(f"CryptoCompare collector initialized (tier: {tier})")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {}
            if self.api_key:
                headers['authorization'] = f'Apikey {self.api_key}'
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self.session
    
    async def _request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make API request with rate limiting."""
        await self.rate_limiter.acquire()
        session = await self._get_session()
        
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            self.stats['requests'] += 1
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('Response') == 'Error':
                        logger.warning(f"CryptoCompare error: {data.get('Message')}")
                        return None
                    return data
                else:
                    self.stats['errors'] += 1
                    logger.warning(f"CryptoCompare request failed: {resp.status}")
                    return None
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"CryptoCompare request error: {e}")
            return None
    
    # =========================================================================
    # Price Methods
    # =========================================================================
    
    async def get_price(
        self,
        symbols: List[str],
        currencies: List[str] = ['USD', 'BTC']
    ) -> Dict[str, Dict[str, float]]:
        """
        Get current prices for multiple symbols.
        
        Args:
            symbols: List of crypto symbols (e.g., ['BTC', 'ETH'])
            currencies: List of quote currencies
            
        Returns:
            Price matrix {symbol: {currency: price}}
        """
        data = await self._request(
            '/pricemulti',
            params={
                'fsyms': ','.join(symbols),
                'tsyms': ','.join(currencies)
            }
        )
        return data or {}
    
    async def get_price_full(
        self,
        symbols: List[str],
        currencies: List[str] = ['USD']
    ) -> Optional[Dict]:
        """
        Get full price data with market metrics.
        
        Returns volume, market cap, and change data.
        """
        return await self._request(
            '/pricemultifull',
            params={
                'fsyms': ','.join(symbols),
                'tsyms': ','.join(currencies)
            }
        )
    
    # =========================================================================
    # Historical Data Methods
    # =========================================================================
    
    async def get_historical_daily(
        self,
        symbol: str,
        currency: str = 'USD',
        limit: int = 365,
        exchange: str = 'CCCAGG'
    ) -> pd.DataFrame:
        """
        Get historical daily OHLCV data.
        
        Args:
            symbol: Crypto symbol
            currency: Quote currency
            limit: Number of days (no limit for daily)
            exchange: Exchange or 'CCCAGG' for aggregated
            
        Returns:
            DataFrame with daily OHLCV
        """
        data = await self._request(
            '/v2/histoday',
            params={
                'fsym': symbol,
                'tsym': currency,
                'limit': limit,
                'e': exchange
            }
        )
        
        if data and data.get('Data', {}).get('Data'):
            records = []
            for row in data['Data']['Data']:
                candle = CCOHLCVCandle(
                    timestamp=pd.to_datetime(row['time'], unit='s', utc=True),
                    symbol=symbol,
                    currency=currency,
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row.get('volumefrom', 0),
                    volume_quote=row.get('volumeto', 0),
                    source=exchange
                )
                records.append(candle.to_dict())
            
            self.stats['records'] += len(records)
            return pd.DataFrame(records)
        
        return pd.DataFrame()
    
    async def get_historical_hourly(
        self,
        symbol: str,
        currency: str = 'USD',
        limit: int = 168,
        exchange: str = 'CCCAGG'
    ) -> pd.DataFrame:
        """
        Get historical hourly OHLCV data.
        
        Args:
            symbol: Crypto symbol
            currency: Quote currency
            limit: Number of hours (max 2000)
            exchange: Exchange or 'CCCAGG'
            
        Returns:
            DataFrame with hourly OHLCV
        """
        data = await self._request(
            '/v2/histohour',
            params={
                'fsym': symbol,
                'tsym': currency,
                'limit': min(limit, 2000),
                'e': exchange
            }
        )
        
        if data and data.get('Data', {}).get('Data'):
            records = []
            for row in data['Data']['Data']:
                candle = CCOHLCVCandle(
                    timestamp=pd.to_datetime(row['time'], unit='s', utc=True),
                    symbol=symbol,
                    currency=currency,
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row.get('volumefrom', 0),
                    volume_quote=row.get('volumeto', 0),
                    source=exchange
                )
                records.append(candle.to_dict())
            
            self.stats['records'] += len(records)
            return pd.DataFrame(records)
        
        return pd.DataFrame()
    
    async def get_historical_minute(
        self,
        symbol: str,
        currency: str = 'USD',
        limit: int = 1440,
        exchange: str = 'CCCAGG'
    ) -> pd.DataFrame:
        """
        Get historical minute OHLCV data.
        
        Args:
            symbol: Crypto symbol
            currency: Quote currency
            limit: Number of minutes (max 2000)
            exchange: Exchange or 'CCCAGG'
            
        Returns:
            DataFrame with minute OHLCV
        """
        data = await self._request(
            '/v2/histominute',
            params={
                'fsym': symbol,
                'tsym': currency,
                'limit': min(limit, 2000),
                'e': exchange
            }
        )
        
        if data and data.get('Data', {}).get('Data'):
            records = []
            for row in data['Data']['Data']:
                candle = CCOHLCVCandle(
                    timestamp=pd.to_datetime(row['time'], unit='s', utc=True),
                    symbol=symbol,
                    currency=currency,
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row.get('volumefrom', 0),
                    volume_quote=row.get('volumeto', 0),
                    source=exchange
                )
                records.append(candle.to_dict())
            
            self.stats['records'] += len(records)
            return pd.DataFrame(records)
        
        return pd.DataFrame()
    
    # =========================================================================
    # Exchange Data Methods
    # =========================================================================
    
    async def get_top_exchanges(
        self,
        symbol: str,
        currency: str = 'USD',
        limit: int = 10
    ) -> List[ExchangeVolume]:
        """
        Get top exchanges by volume for a pair.
        
        Args:
            symbol: Crypto symbol
            currency: Quote currency
            limit: Number of exchanges
            
        Returns:
            List of ExchangeVolume dataclasses
        """
        data = await self._request(
            '/top/exchanges',
            params={
                'fsym': symbol,
                'tsym': currency,
                'limit': limit
            }
        )
        
        if data and data.get('Data'):
            results = []
            for ex in data['Data']:
                ev = ExchangeVolume(
                    exchange=ex.get('exchange', ''),
                    symbol=symbol,
                    currency=currency,
                    volume_24h=ex.get('volume24h', 0),
                    volume_24h_quote=ex.get('volume24hTo', 0),
                    price=ex.get('price', 0),
                    grade=ex.get('GRADE')
                )
                results.append(ev)
            return results
        return []
    
    async def compare_exchange_prices(
        self,
        symbol: str,
        currency: str = 'USD',
        exchanges: List[str] = None
    ) -> Optional[CrossExchangeComparison]:
        """
        Compare prices across exchanges vs CCCAGG.
        
        Args:
            symbol: Crypto symbol
            currency: Quote currency
            exchanges: List of exchanges to compare
            
        Returns:
            CrossExchangeComparison dataclass
        """
        if exchanges is None:
            exchanges = ['Binance', 'Coinbase', 'Kraken', 'Bitstamp']
        
        # Get CCCAGG price
        cccagg_data = await self.get_price([symbol], [currency])
        if not cccagg_data or symbol not in cccagg_data:
            return None
        
        cccagg_price = cccagg_data[symbol][currency]
        
        # Get exchange prices
        exchange_prices = {}
        for exchange in exchanges:
            data = await self._request(
                '/price',
                params={
                    'fsym': symbol,
                    'tsyms': currency,
                    'e': exchange
                }
            )
            if data and currency in data:
                exchange_prices[exchange] = data[currency]
        
        return CrossExchangeComparison(
            symbol=symbol,
            currency=currency,
            cccagg_price=cccagg_price,
            exchange_prices=exchange_prices
        )
    
    # =========================================================================
    # Social Data Methods
    # =========================================================================
    
    async def get_social_stats(self, coin_id: int, symbol: str = '') -> Optional[SocialStats]:
        """
        Get social statistics for a coin.
        
        Args:
            coin_id: CryptoCompare coin ID
            symbol: Symbol for reference
            
        Returns:
            SocialStats dataclass
        """
        data = await self._request(
            '/social/coin/latest',
            params={'coinId': coin_id}
        )
        
        if data and data.get('Data'):
            d = data['Data']
            
            twitter = d.get('Twitter', {})
            reddit = d.get('Reddit', {})
            github = d.get('CodeRepository', {}).get('List', [{}])[0] if d.get('CodeRepository', {}).get('List') else {}
            
            return SocialStats(
                coin_id=coin_id,
                symbol=symbol,
                twitter_followers=twitter.get('followers', 0),
                reddit_subscribers=reddit.get('subscribers', 0),
                reddit_active_users=reddit.get('active_users', 0),
                github_stars=github.get('stars', 0),
                github_forks=github.get('forks', 0),
                github_contributors=github.get('contributors', 0)
            )
        return None
    
    # =========================================================================
    # High-Level Collection Methods
    # =========================================================================
    
    async def _fetch_single_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        days: int,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single symbol.

        Args:
            symbol: Symbol to fetch
            timeframe: '1m', '1h', or '1d'
            days: Number of days in range
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            DataFrame with OHLCV data or empty DataFrame on error
        """
        try:
            logger.info(f"Fetching CryptoCompare {timeframe} data for {symbol}")

            if timeframe == '1d':
                df = await self.get_historical_daily(symbol=symbol, limit=min(days + 1, 2000))
            elif timeframe == '1h':
                hours = days * 24
                df = await self.get_historical_hourly(symbol=symbol, limit=min(hours, 2000))
            else: # 1m
                minutes = days * 24 * 60
                df = await self.get_historical_minute(symbol=symbol, limit=min(minutes, 2000))

            if not df.empty:
                # Filter to date range
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df[
                    (df['timestamp'] >= start_date) &
                    (df['timestamp'] <= end_date)
                ]
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()

    async def fetch_ohlcv(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of symbols
            timeframe: '1m', '1h', or '1d'
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD

        Returns:
            Combined OHLCV DataFrame
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days = (end_dt - start_dt).days

        # Parallelize using asyncio.gather
        tasks = [self._fetch_single_ohlcv(symbol, timeframe, days, start_date, end_date) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter results to only keep valid DataFrames
        all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            return combined

        return pd.DataFrame()
    
    async def fetch_funding_rates(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """CryptoCompare doesn't provide funding rates."""
        logger.info("CryptoCompare does not provide funding rates")
        return pd.DataFrame()
    
    async def fetch_pools(
        self,
        chain: str = 'ethereum',
        min_liquidity: float = 100000
    ) -> pd.DataFrame:
        """Not applicable for market data aggregator."""
        return pd.DataFrame()
    
    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
            logger.info(f"CryptoCompare session closed. Stats: {self.stats}")
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.stats.copy()

async def test_cryptocompare_collector():
    """Test CryptoCompare collector functionality."""
    config = {'rate_limit': 10}
    collector = CryptoCompareCollector(config)
    
    try:
        print("=" * 60)
        print("CryptoCompare Collector Test")
        print("=" * 60)
        print(f"\nStats: {collector.get_collection_stats()}")
    finally:
        await collector.close()

if __name__ == '__main__':
    asyncio.run(test_cryptocompare_collector())