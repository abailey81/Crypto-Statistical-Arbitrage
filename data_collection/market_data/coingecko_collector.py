"""
CoinGecko Market Data Collector - Comprehensive Cryptocurrency Market Data

validated collector for cryptocurrency market data from CoinGecko API.
Provides price data, market rankings, historical OHLCV, and global metrics
for 10,000+ cryptocurrencies.

===============================================================================
OVERVIEW
===============================================================================

CoinGecko is one of the largest cryptocurrency data aggregators, providing:
    - Real-time prices from 500+ exchanges
    - Market cap rankings and metrics
    - Historical price and OHLCV data
    - Exchange volume data
    - Social and developer metrics
    - DeFi-specific data

API Tiers:
    ============== ==================== ============== ================
    Tier Rate Limit Monthly Calls Features
    ============== ==================== ============== ================
    Demo (Free) 10-30 calls/min 10K-30K Basic endpoints
    Analyst 500 calls/min 500K All endpoints
    Lite 500 calls/min 2M Historical data
    Pro 1000 calls/min 5M Full access
    ============== ==================== ============== ================

===============================================================================
DATA TYPES COLLECTED
===============================================================================

Price Data:
    - Current prices in multiple currencies
    - 24h/7d/30d price changes
    - All-time high/low data
    - Token contract prices

Market Data:
    - Market cap and rankings
    - Trading volume (24h)
    - Circulating/total/max supply
    - Fully diluted valuation

Historical Data:
    - OHLC candlestick data (1/7/14/30/90/180/365 days)
    - Market chart data (prices, market caps, volumes)
    - Custom date range queries (Pro tier)

Exchange Data:
    - Exchange listings and rankings
    - Trading pairs and volumes
    - Spread and liquidity metrics
    - Trust scores

Global Metrics:
    - Total market cap
    - Bitcoin/Ethereum dominance
    - DeFi market cap
    - Active cryptocurrencies count

===============================================================================
RATE LIMITING
===============================================================================

The collector implements intelligent rate limiting:
    - Tracks requests per minute
    - Automatic backoff on 429 responses
    - Configurable rate limits per tier
    - Request queuing for burst protection

===============================================================================
USAGE EXAMPLES
===============================================================================

Basic price fetch:

    >>> from data_collection.market_data import CoinGeckoCollector
    >>> 
    >>> collector = CoinGeckoCollector(rate_limit=10)
    >>> try:
    ... prices = await collector.get_price(['bitcoin', 'ethereum'])
    ... print(f"BTC: ${prices['bitcoin']['usd']:,.2f}")
    ... finally:
    ... await collector.close()

Historical OHLCV:

    >>> ohlcv = await collector.fetch_ohlcv(
    ... symbols=['BTC', 'ETH', 'SOL'],
    ... vs_currency='usd',
    ... days=365
    ... )

Market rankings:

    >>> market_data = await collector.fetch_market_data(
    ... top_n=100,
    ... vs_currency='usd'
    ... )

Token price by contract:

    >>> token_prices = await collector.get_token_price(
    ... platform_id='ethereum',
    ... contract_addresses=['0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48'] # USDC
    ... )

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

Cross-Exchange Validation:
    - Compare CoinGecko aggregated prices with exchange-specific data
    - Identify pricing anomalies
    - Validate data quality

Market Structure Analysis:
    - Track market cap distribution
    - Monitor sector rotations (DeFi, L1, L2, etc.)
    - Analyze volume patterns

Sentiment Indicators:
    - Trending coins as momentum signals
    - Volume spikes detection
    - Market dominance changes

Data Enrichment:
    - Supplement exchange data with broader market context
    - Fill gaps in exchange-specific data
    - Cross-reference multiple data sources

===============================================================================
SYMBOL MAPPING
===============================================================================

CoinGecko uses unique IDs (not ticker symbols). The collector provides
automatic mapping for common tokens. For others, use the coin ID directly.

Common Mappings:
    - BTC -> 'bitcoin'
    - ETH -> 'ethereum'
    - SOL -> 'solana'
    - AVAX -> 'avalanche-2'
    - ARB -> 'arbitrum'

===============================================================================
DATA QUALITY CONSIDERATIONS
===============================================================================

- Prices are aggregated from multiple exchanges (weighted average)
- OHLC granularity depends on time range (auto-determined)
- Historical data availability varies by coin age
- Free tier has limited historical data access
- Exchange volume may include wash trading

Version: 2.0.0
API Documentation: https://www.coingecko.com/en/api/documentation
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import os

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================

class VsCurrency(Enum):
    """Supported quote currencies."""
    USD = 'usd'
    EUR = 'eur'
    GBP = 'gbp'
    JPY = 'jpy'
    BTC = 'btc'
    ETH = 'eth'

class MarketOrder(Enum):
    """Market data sort order."""
    MARKET_CAP_DESC = 'market_cap_desc'
    MARKET_CAP_ASC = 'market_cap_asc'
    VOLUME_DESC = 'volume_desc'
    VOLUME_ASC = 'volume_asc'
    ID_ASC = 'id_asc'
    ID_DESC = 'id_desc'
    GECKO_DESC = 'gecko_desc'
    GECKO_ASC = 'gecko_asc'

class PriceChange(Enum):
    """Price change timeframes."""
    ONE_HOUR = '1h'
    TWENTY_FOUR_HOUR = '24h'
    SEVEN_DAY = '7d'
    FOURTEEN_DAY = '14d'
    THIRTY_DAY = '30d'
    TWO_HUNDRED_DAY = '200d'
    ONE_YEAR = '1y'

class OHLCDays(Enum):
    """OHLC data available timeframes."""
    ONE = 1
    SEVEN = 7
    FOURTEEN = 14
    THIRTY = 30
    NINETY = 90
    ONE_EIGHTY = 180
    THREE_SIXTY_FIVE = 365
    MAX = 'max'

class MarketCapRank(Enum):
    """Market cap ranking tiers."""
    TOP_10 = 'top_10'
    TOP_50 = 'top_50'
    TOP_100 = 'top_100'
    TOP_250 = 'top_250'
    TOP_500 = 'top_500'
    OUTSIDE_500 = 'outside_500'

class VolumeLevel(Enum):
    """24h volume classification."""
    VERY_HIGH = 'very_high' # > $1B
    HIGH = 'high' # $100M - $1B
    MEDIUM = 'medium' # $10M - $100M
    LOW = 'low' # $1M - $10M
    VERY_LOW = 'very_low' # < $1M

class TrendDirection(Enum):
    """Price trend direction."""
    STRONG_UP = 'strong_up' # > 10%
    UP = 'up' # 2% - 10%
    NEUTRAL = 'neutral' # -2% to 2%
    DOWN = 'down' # -10% to -2%
    STRONG_DOWN = 'strong_down' # < -10%

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class CoinPrice:
    """Current coin price with market metrics."""
    coin_id: str
    symbol: str
    price_usd: float
    price_btc: Optional[float] = None
    market_cap_usd: Optional[float] = None
    volume_24h_usd: Optional[float] = None
    change_24h_pct: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def volume_mcap_ratio(self) -> float:
        """Volume to market cap ratio (liquidity indicator)."""
        if self.market_cap_usd and self.market_cap_usd > 0:
            return (self.volume_24h_usd or 0) / self.market_cap_usd
        return 0
    
    @property
    def trend_24h(self) -> TrendDirection:
        """Classify 24h trend direction."""
        if self.change_24h_pct is None:
            return TrendDirection.NEUTRAL
        if self.change_24h_pct > 10:
            return TrendDirection.STRONG_UP
        elif self.change_24h_pct > 2:
            return TrendDirection.UP
        elif self.change_24h_pct > -2:
            return TrendDirection.NEUTRAL
        elif self.change_24h_pct > -10:
            return TrendDirection.DOWN
        else:
            return TrendDirection.STRONG_DOWN
    
    @property
    def volume_level(self) -> VolumeLevel:
        """Classify volume level."""
        vol = self.volume_24h_usd or 0
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
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'coin_id': self.coin_id,
            'symbol': self.symbol,
            'price_usd': self.price_usd,
            'price_btc': self.price_btc,
            'market_cap_usd': self.market_cap_usd,
            'volume_24h_usd': self.volume_24h_usd,
            'change_24h_pct': self.change_24h_pct,
            'volume_mcap_ratio': self.volume_mcap_ratio,
            'trend_24h': self.trend_24h.value,
            'volume_level': self.volume_level.value,
            'timestamp': self.timestamp.isoformat(),
            'source': 'coingecko',
        }

@dataclass
class MarketData:
    """Comprehensive market data for a coin."""
    coin_id: str
    symbol: str
    name: str
    market_cap_rank: int
    price_usd: float
    market_cap_usd: float
    fully_diluted_valuation: Optional[float]
    volume_24h_usd: float
    circulating_supply: float
    total_supply: Optional[float]
    max_supply: Optional[float]
    change_1h_pct: Optional[float]
    change_24h_pct: Optional[float]
    change_7d_pct: Optional[float]
    ath_usd: float
    ath_change_pct: float
    ath_date: datetime
    atl_usd: float
    atl_change_pct: float
    atl_date: datetime
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def rank_tier(self) -> MarketCapRank:
        """Classify market cap rank tier."""
        if self.market_cap_rank <= 10:
            return MarketCapRank.TOP_10
        elif self.market_cap_rank <= 50:
            return MarketCapRank.TOP_50
        elif self.market_cap_rank <= 100:
            return MarketCapRank.TOP_100
        elif self.market_cap_rank <= 250:
            return MarketCapRank.TOP_250
        elif self.market_cap_rank <= 500:
            return MarketCapRank.TOP_500
        else:
            return MarketCapRank.OUTSIDE_500
    
    @property
    def supply_ratio(self) -> float:
        """Circulating to max supply ratio."""
        if self.max_supply and self.max_supply > 0:
            return self.circulating_supply / self.max_supply
        elif self.total_supply and self.total_supply > 0:
            return self.circulating_supply / self.total_supply
        return 1.0
    
    @property
    def fdv_mcap_ratio(self) -> float:
        """FDV to market cap ratio (dilution indicator)."""
        if self.fully_diluted_valuation and self.market_cap_usd > 0:
            return self.fully_diluted_valuation / self.market_cap_usd
        return 1.0
    
    @property
    def drawdown_from_ath(self) -> float:
        """Current drawdown from ATH (negative percentage)."""
        return self.ath_change_pct
    
    @property
    def is_near_ath(self) -> bool:
        """Check if within 10% of ATH."""
        return self.ath_change_pct > -10
    
    @property
    def is_near_atl(self) -> bool:
        """Check if within 50% of ATL."""
        return self.atl_change_pct < 50
    
    @property
    def momentum_score(self) -> float:
        """Simple momentum score from price changes."""
        changes = [
            (self.change_1h_pct or 0) * 0.1,
            (self.change_24h_pct or 0) * 0.3,
            (self.change_7d_pct or 0) * 0.6
        ]
        return sum(changes)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'coin_id': self.coin_id,
            'symbol': self.symbol,
            'name': self.name,
            'market_cap_rank': self.market_cap_rank,
            'rank_tier': self.rank_tier.value,
            'price_usd': self.price_usd,
            'market_cap_usd': self.market_cap_usd,
            'fully_diluted_valuation': self.fully_diluted_valuation,
            'volume_24h_usd': self.volume_24h_usd,
            'circulating_supply': self.circulating_supply,
            'total_supply': self.total_supply,
            'max_supply': self.max_supply,
            'supply_ratio': self.supply_ratio,
            'fdv_mcap_ratio': self.fdv_mcap_ratio,
            'change_1h_pct': self.change_1h_pct,
            'change_24h_pct': self.change_24h_pct,
            'change_7d_pct': self.change_7d_pct,
            'momentum_score': self.momentum_score,
            'ath_usd': self.ath_usd,
            'drawdown_from_ath': self.drawdown_from_ath,
            'is_near_ath': self.is_near_ath,
            'atl_usd': self.atl_usd,
            'timestamp': self.timestamp.isoformat(),
            'source': 'coingecko',
        }

@dataclass
class OHLCCandle:
    """OHLC candle data from CoinGecko."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    
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
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': 0.0, # CoinGecko OHLC API doesn't provide volume
            'range_pct': self.range_pct,
            'return_pct': self.return_pct,
            'is_bullish': self.is_bullish,
            'source': 'coingecko',
            'venue': 'coingecko',
            'venue_type': 'market_data',
        }

@dataclass
class GlobalMarketData:
    """Global cryptocurrency market metrics."""
    timestamp: datetime
    total_market_cap_usd: float
    total_volume_24h_usd: float
    btc_dominance_pct: float
    eth_dominance_pct: float
    active_cryptocurrencies: int
    markets: int
    market_cap_change_24h_pct: float
    
    @property
    def altcoin_market_cap_usd(self) -> float:
        """Market cap excluding BTC."""
        return self.total_market_cap_usd * (1 - self.btc_dominance_pct / 100)
    
    @property
    def altcoin_dominance_pct(self) -> float:
        """Altcoin dominance (100 - BTC dominance)."""
        return 100 - self.btc_dominance_pct
    
    @property
    def volume_mcap_ratio(self) -> float:
        """Global volume to market cap ratio."""
        return self.total_volume_24h_usd / self.total_market_cap_usd if self.total_market_cap_usd > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_market_cap_usd': self.total_market_cap_usd,
            'total_volume_24h_usd': self.total_volume_24h_usd,
            'btc_dominance_pct': self.btc_dominance_pct,
            'eth_dominance_pct': self.eth_dominance_pct,
            'altcoin_dominance_pct': self.altcoin_dominance_pct,
            'active_cryptocurrencies': self.active_cryptocurrencies,
            'markets': self.markets,
            'market_cap_change_24h_pct': self.market_cap_change_24h_pct,
            'volume_mcap_ratio': self.volume_mcap_ratio,
            'source': 'coingecko',
        }

# =============================================================================
# Main Collector Class
# =============================================================================

class CoinGeckoCollector:
    """
    CoinGecko market data collector.
    
    validated implementation for comprehensive cryptocurrency
    market data including prices, rankings, and historical data.
    
    Features:
        - Current and historical prices for 10,000+ coins
        - Market cap rankings and metrics
        - OHLC candlestick data
        - Global market statistics
        - Exchange data
        - Intelligent rate limiting
    
    Example:
        >>> collector = CoinGeckoCollector(rate_limit=10)
        >>> try:
        ... prices = await collector.get_price(['bitcoin', 'ethereum'])
        ... market = await collector.fetch_market_data(top_n=100)
        ... finally:
        ... await collector.close()
    
    Attributes:
        VENUE: 'coingecko'
        VENUE_TYPE: 'market_data'
    """
    
    VENUE = 'coingecko'
    VENUE_TYPE = 'market_data'
    
    BASE_URL_FREE = 'https://api.coingecko.com/api/v3'
    BASE_URL_PRO = 'https://pro-api.coingecko.com/api/v3'
    
    # Symbol to CoinGecko ID mapping
    SYMBOL_TO_ID = {
        'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana',
        'AVAX': 'avalanche-2', 'MATIC': 'matic-network', 'ARB': 'arbitrum',
        'OP': 'optimism', 'ATOM': 'cosmos', 'DOT': 'polkadot',
        'LINK': 'chainlink', 'UNI': 'uniswap', 'AAVE': 'aave',
        'MKR': 'maker', 'SNX': 'synthetix-network-token', 'CRV': 'curve-dao-token',
        'LDO': 'lido-dao', 'GMX': 'gmx', 'DYDX': 'dydx',
        'NEAR': 'near', 'FTM': 'fantom', 'APT': 'aptos',
        'SUI': 'sui', 'SEI': 'sei-network', 'INJ': 'injective-protocol',
        'TIA': 'celestia', 'PENDLE': 'pendle', 'SUSHI': 'sushi',
        'BAL': 'balancer', 'COMP': 'compound-governance-token',
        'YFI': 'yearn-finance', 'RPL': 'rocket-pool', 'FXS': 'frax-share',
        'XRP': 'ripple', 'ADA': 'cardano', 'DOGE': 'dogecoin',
        'TRX': 'tron', 'TON': 'the-open-network', 'SHIB': 'shiba-inu',
        'WLD': 'worldcoin-wld', 'BLUR': 'blur', 'APE': 'apecoin',
        'LTC': 'litecoin', 'BCH': 'bitcoin-cash', 'ETC': 'ethereum-classic',
        'POL': 'polygon-ecosystem-token', # MATIC renamed to POL
    }
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        rate_limit: int = 10,
    ):
        """
        Initialize CoinGecko collector.

        Args:
            config: Configuration dict (optional, for unified interface)
            api_key: Pro API key (optional, increases rate limits)
            rate_limit: Max requests per minute (default: 10 for free tier)
        """
        config = config or {}
        self.api_key = config.get('api_key') or api_key or os.getenv('COINGECKO_API_KEY')
        self.rate_limit = config.get('rate_limit', rate_limit)

        # CoinGecko API key types:
        # - Demo keys (CG-xxx): Use FREE URL with x-cg-demo-api-key header
        # - Pro keys: Use PRO URL with x-cg-pro-api-key header
        self.is_demo_key = self.api_key and self.api_key.startswith('CG-')

        if self.api_key and not self.is_demo_key:
            self.base_url = self.BASE_URL_PRO
        else:
            self.base_url = self.BASE_URL_FREE

        self._request_times: List[float] = []
        self._min_interval = 60.0 / rate_limit

        self.session: Optional[aiohttp.ClientSession] = None

        self.stats = {'requests': 0, 'errors': 0, 'rate_limits': 0}

        tier = 'Demo' if self.is_demo_key else ('Pro' if self.api_key else 'Free')
        logger.info(f"CoinGecko collector initialized (tier: {tier}, rate_limit: {rate_limit}/min)")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            headers = {'Accept': 'application/json'}
            if self.api_key:
                # Demo keys use x-cg-demo-api-key, Pro keys use x-cg-pro-api-key
                header_name = 'x-cg-demo-api-key' if self.is_demo_key else 'x-cg-pro-api-key'
                headers[header_name] = self.api_key

            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self.session
    
    async def _rate_limit(self):
        """Apply rate limiting."""
        now = asyncio.get_event_loop().time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        
        if len(self._request_times) >= self.rate_limit:
            wait_time = 60 - (now - self._request_times[0])
            if wait_time > 0:
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        self._request_times.append(now)
    
    async def _request(self, endpoint: str, params: Optional[Dict] = None, _retry_count: int = 0) -> Optional[Any]:
        """Make API request with rate limiting and exponential backoff."""
        await self._rate_limit()

        session = await self._get_session()
        url = f"{self.base_url}/{endpoint}"
        max_retries = 3

        try:
            self.stats['requests'] += 1

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    self.stats['rate_limits'] += 1
                    if _retry_count >= max_retries:
                        logger.error(f"CoinGecko rate limit: max retries ({max_retries}) exceeded")
                        return None
                    # Exponential backoff: 30s, 60s, 120s
                    wait_time = min(30 * (2 ** _retry_count), 120)
                    retry_after = int(response.headers.get('Retry-After', wait_time))
                    logger.warning(f"CoinGecko rate limit hit, waiting {retry_after}s (retry {_retry_count + 1}/{max_retries})")
                    await asyncio.sleep(retry_after)
                    return await self._request(endpoint, params, _retry_count=_retry_count + 1)
                else:
                    self.stats['errors'] += 1
                    text = await response.text()
                    logger.error(f"CoinGecko API error {response.status}: {text[:200]}")
                    return None
        except asyncio.TimeoutError:
            self.stats['errors'] += 1
            if _retry_count < max_retries:
                wait_time = 5 * (2 ** _retry_count)
                logger.warning(f"CoinGecko timeout, retrying in {wait_time}s (retry {_retry_count + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
                return await self._request(endpoint, params, _retry_count=_retry_count + 1)
            logger.error(f"CoinGecko request timeout after {max_retries} retries")
            return None
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"CoinGecko request error: {e}")
            return None
    
    def _symbol_to_id(self, symbol: str) -> str:
        """Convert trading symbol to CoinGecko ID."""
        symbol_upper = symbol.upper().replace('/USDT', '').replace('/USD', '').replace('-', '')
        return self.SYMBOL_TO_ID.get(symbol_upper, symbol.lower())
    
    # =========================================================================
    # Price Methods
    # =========================================================================
    
    async def get_price(
        self,
        ids: List[str],
        vs_currencies: List[str] = ['usd'],
        include_market_cap: bool = True,
        include_24hr_vol: bool = True,
        include_24hr_change: bool = True,
    ) -> Optional[Dict]:
        """Get current prices for multiple coins."""
        coin_ids = [self._symbol_to_id(id) for id in ids]
        
        params = {
            'ids': ','.join(coin_ids),
            'vs_currencies': ','.join(vs_currencies),
            'include_market_cap': str(include_market_cap).lower(),
            'include_24hr_vol': str(include_24hr_vol).lower(),
            'include_24hr_change': str(include_24hr_change).lower(),
        }
        
        return await self._request('simple/price', params)
    
    async def get_token_price(
        self,
        platform_id: str,
        contract_addresses: List[str],
        vs_currencies: List[str] = ['usd'],
    ) -> Optional[Dict]:
        """Get token prices by contract address."""
        params = {
            'contract_addresses': ','.join(contract_addresses),
            'vs_currencies': ','.join(vs_currencies),
        }
        return await self._request(f'simple/token_price/{platform_id}', params)
    
    # =========================================================================
    # Market Data Methods
    # =========================================================================
    
    async def get_coins_markets(
        self,
        vs_currency: str = 'usd',
        order: str = 'market_cap_desc',
        per_page: int = 100,
        page: int = 1,
        price_change_percentage: str = '1h,24h,7d',
    ) -> Optional[List[Dict]]:
        """Get coins market data with ranking."""
        params = {
            'vs_currency': vs_currency,
            'order': order,
            'per_page': min(per_page, 250),
            'page': page,
            'sparkline': 'false',
            'price_change_percentage': price_change_percentage,
        }
        return await self._request('coins/markets', params)
    
    async def get_global(self) -> Optional[GlobalMarketData]:
        """Get global cryptocurrency statistics."""
        data = await self._request('global')
        
        if data and 'data' in data:
            d = data['data']
            return GlobalMarketData(
                timestamp=datetime.now(timezone.utc),
                total_market_cap_usd=d.get('total_market_cap', {}).get('usd', 0),
                total_volume_24h_usd=d.get('total_volume', {}).get('usd', 0),
                btc_dominance_pct=d.get('market_cap_percentage', {}).get('btc', 0),
                eth_dominance_pct=d.get('market_cap_percentage', {}).get('eth', 0),
                active_cryptocurrencies=d.get('active_cryptocurrencies', 0),
                markets=d.get('markets', 0),
                market_cap_change_24h_pct=d.get('market_cap_change_percentage_24h_usd', 0)
            )
        return None
    
    async def get_trending(self) -> Optional[Dict]:
        """Get trending search coins."""
        return await self._request('search/trending')
    
    # =========================================================================
    # Historical Data Methods
    # =========================================================================
    
    async def get_coin_ohlc(
        self,
        coin_id: str,
        vs_currency: str = 'usd',
        days: int = 30,
    ) -> Optional[List]:
        """Get OHLC data."""
        params = {'vs_currency': vs_currency, 'days': str(days)}
        coin_id = self._symbol_to_id(coin_id)
        return await self._request(f'coins/{coin_id}/ohlc', params)
    
    async def get_coin_market_chart_range(
        self,
        coin_id: str,
        vs_currency: str = 'usd',
        from_timestamp: Optional[int] = None,
        to_timestamp: Optional[int] = None,
    ) -> Optional[Dict]:
        """Get historical market data within time range."""
        if from_timestamp is None:
            from_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())
        if to_timestamp is None:
            to_timestamp = int(datetime.now().timestamp())
        
        params = {
            'vs_currency': vs_currency,
            'from': from_timestamp,
            'to': to_timestamp,
        }
        
        coin_id = self._symbol_to_id(coin_id)
        return await self._request(f'coins/{coin_id}/market_chart/range', params)
    
    # =========================================================================
    # High-Level Collection Methods
    # =========================================================================
    
    # Valid days values for CoinGecko OHLC API
    VALID_OHLC_DAYS = [1, 7, 14, 30, 90, 180, 365]

    async def _fetch_single_ohlcv(
        self,
        symbol: str,
        vs_currency: str,
        days: int
    ) -> List[Dict]:
        """
        Fetch OHLCV data for a single symbol.

        Args:
            symbol: Symbol or CoinGecko ID
            vs_currency: Quote currency
            days: Number of days

        Returns:
            List of candle dicts or empty list on error
        """
        try:
            logger.info(f"Fetching CoinGecko OHLCV for {symbol}")

            data = await self.get_coin_ohlc(symbol, vs_currency, days)

            if data:
                records = []
                for candle in data:
                    c = OHLCCandle(
                        timestamp=pd.to_datetime(candle[0], unit='ms', utc=True),
                        symbol=symbol.upper(),
                        open=candle[1],
                        high=candle[2],
                        low=candle[3],
                        close=candle[4]
                    )
                    records.append(c.to_dict())
                return records
            return []
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return []

    async def fetch_ohlcv(
        self,
        symbols: List[str],
        timeframe: str = '1d',
        start_date: str = None,
        end_date: str = None,
        vs_currency: str = 'usd',
        days: int = 365,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of symbols or CoinGecko IDs
            timeframe: Ignored (CoinGecko determines granularity from days)
            start_date: Start date (YYYY-MM-DD) - if provided, calculates days
            end_date: End date (YYYY-MM-DD) - if provided, calculates days
            vs_currency: Quote currency
            days: Number of days (1, 7, 14, 30, 90, 180, 365, max)
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            DataFrame with OHLCV data and computed fields
        """
        # Calculate days from date range if provided
        if start_date and end_date:
            from datetime import datetime
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            days = (end - start).days

        # CoinGecko OHLC API only supports specific day values: 1, 7, 14, 30, 90, 180, 365
        # Cap days to the nearest valid value to avoid API errors
        if days > 365:
            logger.info(f"CoinGecko OHLC: Requested {days} days exceeds max (365), using 365")
            days = 365
        elif days not in self.VALID_OHLC_DAYS:
            # Find the nearest valid value that's >= requested days
            valid_days = [d for d in self.VALID_OHLC_DAYS if d >= days]
            days = valid_days[0] if valid_days else 365
            logger.debug(f"CoinGecko OHLC: Adjusted days to nearest valid value: {days}")

        # Parallelize using asyncio.gather
        tasks = [self._fetch_single_ohlcv(symbol, vs_currency, days) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter results to only keep valid data
        all_records = []
        for result in results:
            if isinstance(result, list):
                all_records.extend(result)

        df = pd.DataFrame(all_records)
        if not df.empty:
            df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        return df
    
    async def fetch_market_data(
        self,
        top_n: int = 100,
        vs_currency: str = 'usd',
    ) -> pd.DataFrame:
        """
        Fetch market data for top coins by market cap.
        
        Args:
            top_n: Number of top coins
            vs_currency: Quote currency
            
        Returns:
            DataFrame with comprehensive market data
        """
        all_records = []
        pages = (top_n + 249) // 250
        
        for page in range(1, pages + 1):
            logger.info(f"Fetching CoinGecko market data page {page}")
            
            data = await self.get_coins_markets(
                vs_currency=vs_currency,
                per_page=min(250, top_n - len(all_records)),
                page=page,
            )
            
            if data:
                for coin in data:
                    m = MarketData(
                        coin_id=coin['id'],
                        symbol=coin['symbol'].upper(),
                        name=coin['name'],
                        market_cap_rank=coin.get('market_cap_rank', 0) or 0,
                        price_usd=coin.get('current_price', 0) or 0,
                        market_cap_usd=coin.get('market_cap', 0) or 0,
                        fully_diluted_valuation=coin.get('fully_diluted_valuation'),
                        volume_24h_usd=coin.get('total_volume', 0) or 0,
                        circulating_supply=coin.get('circulating_supply', 0) or 0,
                        total_supply=coin.get('total_supply'),
                        max_supply=coin.get('max_supply'),
                        change_1h_pct=coin.get('price_change_percentage_1h_in_currency'),
                        change_24h_pct=coin.get('price_change_percentage_24h'),
                        change_7d_pct=coin.get('price_change_percentage_7d_in_currency'),
                        ath_usd=coin.get('ath', 0) or 0,
                        ath_change_pct=coin.get('ath_change_percentage', 0) or 0,
                        ath_date=pd.to_datetime(coin.get('ath_date')) if coin.get('ath_date') else datetime.now(timezone.utc),
                        atl_usd=coin.get('atl', 0) or 0,
                        atl_change_pct=coin.get('atl_change_percentage', 0) or 0,
                        atl_date=pd.to_datetime(coin.get('atl_date')) if coin.get('atl_date') else datetime.now(timezone.utc),
                    )
                    all_records.append(m.to_dict())
            
            await asyncio.sleep(0.5)
        
        return pd.DataFrame(all_records)
    
    async def _fetch_single_historical_price(
        self,
        symbol: str,
        vs_currency: str,
        from_ts: int,
        to_ts: int
    ) -> List[Dict]:
        """
        Fetch historical prices for a single symbol.

        Args:
            symbol: Symbol to fetch
            vs_currency: Quote currency
            from_ts: Start timestamp
            to_ts: End timestamp

        Returns:
            List of price records or empty list on error
        """
        try:
            logger.info(f"Fetching CoinGecko historical prices for {symbol}")

            data = await self.get_coin_market_chart_range(symbol, vs_currency, from_ts, to_ts)

            if data and 'prices' in data:
                records = []
                for price_point in data['prices']:
                    records.append({
                        'timestamp': pd.to_datetime(price_point[0], unit='ms', utc=True),
                        'symbol': symbol.upper(),
                        'price': price_point[1],
                        'source': self.VENUE,
                    })
                return records
            return []
        except Exception as e:
            logger.error(f"Error fetching historical prices for {symbol}: {e}")
            return []

    async def fetch_historical_prices(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        vs_currency: str = 'usd',
    ) -> pd.DataFrame:
        """Fetch historical prices for multiple symbols."""
        from_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        to_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        # Parallelize using asyncio.gather
        tasks = [self._fetch_single_historical_price(symbol, vs_currency, from_ts, to_ts) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter results to only keep valid data
        all_records = []
        for result in results:
            if isinstance(result, list):
                all_records.extend(result)

        df = pd.DataFrame(all_records)
        if not df.empty:
            df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        return df
    
    async def collect_market_cap(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect market cap data - wraps fetch_market_data().

        Args:
            symbols: List of symbols (used for filtering results)
            start_date: Start date (ignored - returns current snapshot)
            end_date: End date (ignored - returns current snapshot)
            **kwargs: Additional arguments (vs_currency, top_n)

        Returns:
            DataFrame with market cap data
        """
        try:
            vs_currency = kwargs.get('vs_currency', 'usd')
            top_n = kwargs.get('top_n', 250)

            # Fetch market data for top coins
            df = await self.fetch_market_data(top_n=top_n, vs_currency=vs_currency)

            if df.empty:
                return pd.DataFrame()

            # Filter to requested symbols if provided
            if symbols:
                # Support both CoinGecko IDs (bitcoin, ethereum) and symbols (BTC, ETH)
                symbols_upper = [s.upper() for s in symbols]
                symbols_lower = [s.lower() for s in symbols]

                # Try filtering by symbol first, then by coin_id
                mask = df['symbol'].isin(symbols_upper)
                if 'coin_id' in df.columns:
                    mask |= df['coin_id'].isin(symbols_lower)

                df = df[mask]

            # Extract market cap specific columns
            market_cap_cols = [
                'symbol', 'timestamp', 'coin_id', 'name',
                'market_cap_usd', 'market_cap_rank', 'rank_tier',
                'fully_diluted_valuation', 'fdv_mcap_ratio',
                'circulating_supply', 'total_supply', 'max_supply', 'supply_ratio',
                'source'
            ]

            available_cols = [col for col in market_cap_cols if col in df.columns]

            if available_cols:
                result = df[available_cols].copy()
                # Add venue info
                result['venue'] = self.VENUE
                result['venue_type'] = self.VENUE_TYPE
                return result

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"CoinGecko collect_market_cap error: {e}")
            return pd.DataFrame()

    async def collect_volume(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect volume data - wraps fetch_market_data().

        Args:
            symbols: List of symbols (used for filtering results)
            start_date: Start date (ignored - returns current snapshot)
            end_date: End date (ignored - returns current snapshot)
            **kwargs: Additional arguments (vs_currency, top_n)

        Returns:
            DataFrame with volume data
        """
        try:
            vs_currency = kwargs.get('vs_currency', 'usd')
            top_n = kwargs.get('top_n', 250)

            # Fetch market data for top coins
            df = await self.fetch_market_data(top_n=top_n, vs_currency=vs_currency)

            if df.empty:
                return pd.DataFrame()

            # Filter to requested symbols if provided
            if symbols:
                # Support both CoinGecko IDs (bitcoin, ethereum) and symbols (BTC, ETH)
                symbols_upper = [s.upper() for s in symbols]
                symbols_lower = [s.lower() for s in symbols]

                # Try filtering by symbol first, then by coin_id
                mask = df['symbol'].isin(symbols_upper)
                if 'coin_id' in df.columns:
                    mask |= df['coin_id'].isin(symbols_lower)

                df = df[mask]

            # Extract volume specific columns
            volume_cols = [
                'symbol', 'timestamp', 'coin_id', 'name',
                'volume_24h_usd', 'market_cap_usd',
                'price_usd', 'change_24h_pct',
                'source'
            ]

            available_cols = [col for col in volume_cols if col in df.columns]

            if available_cols:
                result = df[available_cols].copy()
                # Add venue info
                result['venue'] = self.VENUE
                result['venue_type'] = self.VENUE_TYPE
                return result

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"CoinGecko collect_volume error: {e}")
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
            # Convert dates to string format if needed
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date) if start_date else None

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date) if end_date else None

            return await self.fetch_ohlcv(
                symbols=symbols,
                start_date=start_str,
                end_date=end_str,
                **kwargs
            )
        except Exception as e:
            logger.error(f"CoinGecko collect_ohlcv error: {e}")
            return pd.DataFrame()

    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info(f"CoinGecko session closed. Stats: {self.stats}")
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.stats.copy()
    
    @classmethod
    def get_supported_symbols(cls) -> List[str]:
        """Get list of mapped symbols."""
        return list(cls.SYMBOL_TO_ID.keys())

async def test_coingecko_collector():
    """Test CoinGecko collector functionality."""
    collector = CoinGeckoCollector(rate_limit=10)
    
    try:
        print("=" * 60)
        print("CoinGecko Collector Test")
        print("=" * 60)
        print(f"\nSupported symbols: {len(CoinGeckoCollector.get_supported_symbols())}")
        print(f"\nStats: {collector.get_collection_stats()}")
    finally:
        await collector.close()

if __name__ == '__main__':
    asyncio.run(test_coingecko_collector())