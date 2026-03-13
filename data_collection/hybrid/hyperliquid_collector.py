"""
Hyperliquid Data Collector - Hybrid On-Chain Perpetuals

validated collector for Hyperliquid perpetual futures.
On-chain settlement with order book execution on Arbitrum L1.

===============================================================================
PROTOCOL OVERVIEW
===============================================================================

Hyperliquid is a hybrid decentralized exchange featuring:
    - On-chain perpetual futures with order book model
    - Settlement on Arbitrum L1 (Ethereum L2)
    - HOURLY funding rate payments (CRITICAL: differs from CEX 8-hour)
    - Up to 50x leverage on major pairs
    - Self-custody with smart contract settlement

Key Differentiators:
    - No counterparty risk (on-chain settlement)
    - Sub-second block times on Arbitrum
    - Lower fees than traditional CEX
    - Transparent order book and liquidations

===============================================================================
FUNDING RATE MECHANISM (CRITICAL)
===============================================================================

IMPORTANT: Hyperliquid uses HOURLY funding, NOT 8-hour like CEX.

Funding Comparison:
    ============== ================ ================= ================
    Venue Interval To Compare to CEX Annualization
    ============== ================ ================= ================
    Hyperliquid 1 hour Ã— 8 Ã— 8760
    dYdX V4 1 hour Ã— 8 Ã— 8760
    Binance 8 hours - Ã— 1095
    Bybit 8 hours - Ã— 1095
    OKX 8 hours - Ã— 1095
    ============== ================ ================= ================

Normalization Formula:
    - 8h_equivalent = hourly_rate Ã— 8
    - annualized = hourly_rate Ã— 8760 (24 Ã— 365)

This collector provides BOTH raw hourly rates AND normalized 8h equivalents.

===============================================================================
API DOCUMENTATION
===============================================================================

Base URL: https://api.hyperliquid.xyz/info

Endpoints (all POST with JSON body):
    - meta: Market metadata and current funding
    - fundingHistory: Historical funding rates (max 500 hours/request)
    - candleSnapshot: OHLCV candlestick data
    - l2Book: Order book snapshot
    - userFills: Trade history (requires authentication)

Rate Limits:
    - ~100 requests per minute
    - Max 500 hours per funding history request (~21 days)
    - No official documentation on exact limits

===============================================================================
DATA TYPES COLLECTED
===============================================================================

Funding Rates:
    - Raw hourly funding rate
    - 8-hour normalized equivalent
    - Annualized rate
    - Premium/discount component

OHLCV:
    - Open, High, Low, Close prices
    - Volume in base asset
    - Number of trades

Open Interest:
    - Current OI snapshot (no historical API)
    - Mark and index prices
    - Funding rate at snapshot time

Market Metadata:
    - Available perpetual contracts
    - Size decimals and tick size
    - Maximum leverage
    - Trading status

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

1. CEX vs Hyperliquid Funding Arbitrage:
   - Compare 8h-normalized Hyperliquid funding to CEX
   - Identify persistent funding divergences
   - Account for hourly vs 8h timing differences

2. Basis Trading:
   - Hyperliquid perp vs CEX spot basis
   - Cross-venue mark price divergence
   - Liquidation cascade opportunities

3. Liquidity Analysis:
   - Order book depth comparison
   - Price impact at various sizes
   - Maker vs taker flow

4. On-Chain Transparency:
   - Liquidation monitoring (on-chain)
   - Large position detection
   - Whale wallet tracking

===============================================================================
DATA QUALITY CONSIDERATIONS
===============================================================================

- Hourly funding must be normalized for cross-venue comparison
- API may return stale data during high congestion
- Historical OI not available via API (snapshot only)
- Order book depth varies significantly by pair
- Block time affects data freshness

Version: 2.0.0
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

from ..base_collector import BaseCollector
from ..utils.rate_limiter import get_shared_rate_limiter
from ..utils.retry_handler import RetryHandler, safe_float, safe_int

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================

class FundingInterval(Enum):
    """Funding rate payment interval."""
    HOURLY = 'hourly' # Hyperliquid, dYdX V4
    EIGHT_HOUR = '8h' # Binance, Bybit, OKX
    FOUR_HOUR = '4h' # Some venues
    CONTINUOUS = 'continuous' # GMX borrow fee model

class FundingTrend(Enum):
    """Funding rate trend classification."""
    VERY_POSITIVE = 'very_positive' # > 0.05% hourly (> 0.4% 8h equiv)
    POSITIVE = 'positive' # 0.01% - 0.05% hourly
    NEUTRAL = 'neutral' # -0.01% to 0.01%
    NEGATIVE = 'negative' # -0.05% to -0.01%
    VERY_NEGATIVE = 'very_negative' # < -0.05% hourly

class MarketSentiment(Enum):
    """Market sentiment based on funding."""
    STRONGLY_BULLISH = 'strongly_bullish' # Longs paying high funding
    BULLISH = 'bullish'
    NEUTRAL = 'neutral'
    BEARISH = 'bearish'
    STRONGLY_BEARISH = 'strongly_bearish' # Shorts paying high funding

class LiquidityDepth(Enum):
    """Order book liquidity classification."""
    VERY_DEEP = 'very_deep' # > $10M within 1%
    DEEP = 'deep' # $5M - $10M
    MODERATE = 'moderate' # $1M - $5M
    SHALLOW = 'shallow' # $500K - $1M
    THIN = 'thin' # < $500K

class MarketStatus(Enum):
    """Market trading status."""
    ACTIVE = 'active'
    PAUSED = 'paused'
    DELISTED = 'delisted'
    PENDING = 'pending'

class PositionSide(Enum):
    """Position direction."""
    LONG = 'long'
    SHORT = 'short'

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class HyperliquidFundingRate:
    """
    Hyperliquid funding rate data with normalization.
    
    IMPORTANT: Raw rate is HOURLY. Use funding_rate_8h for CEX comparison.
    """
    timestamp: datetime
    symbol: str
    funding_rate: float # Raw hourly rate
    premium: float # Funding premium component
    mark_price: Optional[float] = None
    
    @property
    def funding_rate_8h(self) -> float:
        """8-hour equivalent rate for CEX comparison."""
        return self.funding_rate * 8
    
    @property
    def funding_rate_annualized(self) -> float:
        """Annualized funding rate (hourly Ã— 8760)."""
        return self.funding_rate * 8760
    
    @property
    def funding_rate_daily(self) -> float:
        """Daily funding rate (hourly Ã— 24)."""
        return self.funding_rate * 24
    
    @property
    def trend(self) -> FundingTrend:
        """Classify funding trend."""
        rate = self.funding_rate * 100 # Convert to percentage
        if rate > 0.05:
            return FundingTrend.VERY_POSITIVE
        elif rate > 0.01:
            return FundingTrend.POSITIVE
        elif rate > -0.01:
            return FundingTrend.NEUTRAL
        elif rate > -0.05:
            return FundingTrend.NEGATIVE
        else:
            return FundingTrend.VERY_NEGATIVE
    
    @property
    def sentiment(self) -> MarketSentiment:
        """Derive market sentiment from funding."""
        rate = self.funding_rate * 100
        if rate > 0.05:
            return MarketSentiment.STRONGLY_BULLISH
        elif rate > 0.01:
            return MarketSentiment.BULLISH
        elif rate > -0.01:
            return MarketSentiment.NEUTRAL
        elif rate > -0.05:
            return MarketSentiment.BEARISH
        else:
            return MarketSentiment.STRONGLY_BEARISH
    
    @property
    def is_arbitrage_opportunity(self) -> bool:
        """Check if funding suggests arb opportunity (> 0.1% 8h equiv)."""
        return abs(self.funding_rate_8h) > 0.001
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with all computed fields."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'funding_rate': self.funding_rate,
            'funding_rate_8h': self.funding_rate_8h,
            'funding_rate_annualized': self.funding_rate_annualized,
            'funding_rate_daily': self.funding_rate_daily,
            'premium': self.premium,
            'mark_price': self.mark_price,
            'trend': self.trend.value,
            'sentiment': self.sentiment.value,
            'is_arbitrage_opportunity': self.is_arbitrage_opportunity,
            'venue': 'hyperliquid',
            'venue_type': 'hybrid',
            'funding_interval': 'hourly',
        }

@dataclass
class HyperliquidOHLCV:
    """OHLCV candle data from Hyperliquid."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int = 0
    
    @property
    def typical_price(self) -> float:
        """Typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3
    
    @property
    def range_pct(self) -> float:
        """Price range as percentage of open."""
        return (self.high - self.low) / self.open * 100 if self.open > 0 else 0
    
    @property
    def body_pct(self) -> float:
        """Candle body as percentage of range."""
        range_val = self.high - self.low
        if range_val == 0:
            return 0
        return abs(self.close - self.open) / range_val * 100
    
    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish."""
        return self.close > self.open
    
    @property
    def return_pct(self) -> float:
        """Period return percentage."""
        return (self.close - self.open) / self.open * 100 if self.open > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'trades': self.trades,
            'typical_price': self.typical_price,
            'range_pct': self.range_pct,
            'return_pct': self.return_pct,
            'is_bullish': self.is_bullish,
            'venue': 'hyperliquid',
            'venue_type': 'hybrid',
        }

@dataclass
class HyperliquidOpenInterest:
    """Open interest snapshot from Hyperliquid."""
    timestamp: datetime
    symbol: str
    open_interest: float # In contracts
    open_interest_usd: float # Notional USD value
    mark_price: float
    index_price: float
    funding_rate: float # Current hourly funding
    
    @property
    def basis_bps(self) -> float:
        """Basis in basis points (mark vs index)."""
        if self.index_price > 0:
            return (self.mark_price - self.index_price) / self.index_price * 10000
        return 0
    
    @property
    def is_contango(self) -> bool:
        """Check if mark > index (contango)."""
        return self.mark_price > self.index_price
    
    @property
    def funding_rate_8h(self) -> float:
        """8-hour equivalent funding for comparison."""
        return self.funding_rate * 8
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'open_interest': self.open_interest,
            'open_interest_usd': self.open_interest_usd,
            'mark_price': self.mark_price,
            'index_price': self.index_price,
            'funding_rate': self.funding_rate,
            'funding_rate_8h': self.funding_rate_8h,
            'basis_bps': self.basis_bps,
            'is_contango': self.is_contango,
            'venue': 'hyperliquid',
            'venue_type': 'hybrid',
        }

@dataclass
class HyperliquidMarket:
    """Market/instrument metadata from Hyperliquid."""
    symbol: str
    size_decimals: int
    max_leverage: int
    tick_size: float
    min_size: float
    status: MarketStatus = MarketStatus.ACTIVE
    
    @property
    def is_tradeable(self) -> bool:
        """Check if market is tradeable."""
        return self.status == MarketStatus.ACTIVE
    
    @property
    def leverage_tier(self) -> str:
        """Classify leverage tier."""
        if self.max_leverage >= 50:
            return 'high'
        elif self.max_leverage >= 20:
            return 'medium'
        else:
            return 'low'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'size_decimals': self.size_decimals,
            'max_leverage': self.max_leverage,
            'tick_size': self.tick_size,
            'min_size': self.min_size,
            'status': self.status.value,
            'is_tradeable': self.is_tradeable,
            'leverage_tier': self.leverage_tier,
            'venue': 'hyperliquid',
            'venue_type': 'hybrid',
        }

@dataclass
class HyperliquidOrderBook:
    """Order book snapshot from Hyperliquid."""
    timestamp: datetime
    symbol: str
    bids: List[Tuple[float, float]] # [(price, size), ...]
    asks: List[Tuple[float, float]]
    
    @property
    def best_bid(self) -> float:
        """Best bid price."""
        return self.bids[0][0] if self.bids else 0
    
    @property
    def best_ask(self) -> float:
        """Best ask price."""
        return self.asks[0][0] if self.asks else 0
    
    @property
    def mid_price(self) -> float:
        """Mid price."""
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2
        return 0
    
    @property
    def spread_bps(self) -> float:
        """Bid-ask spread in basis points."""
        if self.mid_price > 0:
            return (self.best_ask - self.best_bid) / self.mid_price * 10000
        return 0
    
    @property
    def bid_depth_usd(self) -> float:
        """Total bid depth in USD (top 20 levels)."""
        return sum(p * s for p, s in self.bids[:20])
    
    @property
    def ask_depth_usd(self) -> float:
        """Total ask depth in USD (top 20 levels)."""
        return sum(p * s for p, s in self.asks[:20])
    
    @property
    def imbalance(self) -> float:
        """Order book imbalance (-1 to 1, positive = more bids)."""
        total = self.bid_depth_usd + self.ask_depth_usd
        if total > 0:
            return (self.bid_depth_usd - self.ask_depth_usd) / total
        return 0
    
    @property
    def liquidity_depth(self) -> LiquidityDepth:
        """Classify liquidity depth."""
        total_depth = self.bid_depth_usd + self.ask_depth_usd
        if total_depth > 20_000_000:
            return LiquidityDepth.VERY_DEEP
        elif total_depth > 10_000_000:
            return LiquidityDepth.DEEP
        elif total_depth > 2_000_000:
            return LiquidityDepth.MODERATE
        elif total_depth > 1_000_000:
            return LiquidityDepth.SHALLOW
        else:
            return LiquidityDepth.THIN
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'mid_price': self.mid_price,
            'spread_bps': self.spread_bps,
            'bid_depth_usd': self.bid_depth_usd,
            'ask_depth_usd': self.ask_depth_usd,
            'imbalance': self.imbalance,
            'liquidity_depth': self.liquidity_depth.value,
            'venue': 'hyperliquid',
            'venue_type': 'hybrid',
        }

# =============================================================================
# Collector Class
# =============================================================================

class HyperliquidCollector(BaseCollector):
    """
    Hyperliquid data collector for hybrid perpetuals.
    
    validated implementation for on-chain perpetual data collection.
    Handles hourly funding rate normalization for cross-venue comparison.
    
    KEY DIFFERENCE FROM CEX:
        - Hyperliquid funding is HOURLY (every hour)
        - Binance/Bybit funding is every 8 HOURS
        - Must normalize when comparing:
            * hourly_rate Ã— 8 = equivalent 8h rate
            * hourly_rate Ã— 8760 = annualized
    
    Features:
        - Historical funding rates (max 500 hours/request)
        - OHLCV candlestick data
        - Open interest snapshots
        - Order book snapshots
        - Market metadata
        - Automatic rate limiting
        - Retry with exponential backoff
    
    Attributes:
        VENUE: 'hyperliquid'
        VENUE_TYPE: 'hybrid'
        FUNDING_INTERVAL_HOURS: 1 (hourly funding)
    
    Example:
        >>> config = {'rate_limit': 30}
        >>> async with HyperliquidCollector(config) as collector:
        ... funding = await collector.fetch_funding_rates(
        ... ['BTC', 'ETH'], '2024-01-01', '2024-03-31'
        ... )
        ... # funding['funding_rate'] = raw hourly
        ... # funding['funding_rate_8h'] = normalized for CEX comparison
    """
    
    VENUE = 'hyperliquid'
    VENUE_TYPE = 'hybrid'
    BASE_URL = 'https://api.hyperliquid.xyz/info'
    
    # CRITICAL: Hyperliquid has HOURLY funding
    FUNDING_INTERVAL_HOURS = 1
    
    # API limitations
    MAX_HOURS_PER_REQUEST = 500 # ~21 days for funding rates
    MAX_CANDLES_PER_REQUEST = 5000 # API limit per request

    # Timeframe mapping
    TIMEFRAME_MAP = {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '4h': '4h', '1d': '1d', '1D': '1d'
    }

    # Interval to milliseconds mapping for pagination calculations
    INTERVAL_MS = {
        '1m': 60_000, '5m': 300_000, '15m': 900_000, '30m': 1_800_000,
        '1h': 3_600_000, '4h': 14_400_000, '1d': 86_400_000
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Hyperliquid collector.
        
        Args:
            config: Configuration with options:
                - rate_limit: Requests per minute (default: 30, reduced to avoid 429 errors)
                - timeout: Request timeout seconds (default: 60)
                - max_retries: Retry attempts (default: 5)
        """
        config = config or {}
        super().__init__(config)
        
        rate_limit = config.get('rate_limit', 30) # REDUCED from 60 to stay well within limits
        # Use shared rate limiter to avoid re-initialization overhead
        self.rate_limiter = get_shared_rate_limiter(
            'hyperliquid',
            rate=rate_limit,
            per=60.0,
            burst=min(3, rate_limit // 10) # REDUCED burst from 6 to 3
        )
        
        self.retry_handler = RetryHandler(
            max_retries=config.get('max_retries', 5),
            base_delay=2.0,
            max_delay=120.0
        )
        
        self.timeout = aiohttp.ClientTimeout(total=config.get('timeout', 60))
        self.session: Optional[aiohttp.ClientSession] = None

        # CRITICAL: Set supported data types for dynamic routing (collection_manager)
        self.supported_data_types = ['funding_rates', 'ohlcv', 'open_interest', 'positions', 'trades']
        self.venue = 'hyperliquid'
        self.requires_auth = False # Public API endpoints

        self.collection_stats = {
            'records_collected': 0, 'api_calls': 0, 'errors': 0
        }

        self._request_sem = asyncio.Semaphore(5)

        logger.info(f"Initialized Hyperliquid collector (rate_limit={rate_limit}/min)")

    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            # SPEEDUP: Aggressive connection pooling for high-throughput collection
            # Hyperliquid rate limit: 1200/min = 20/sec, higher pool supports concurrent symbols
            connector = aiohttp.TCPConnector(
                limit=80, # Total connection pool size
                limit_per_host=25, # Per-host connections
                ttl_dns_cache=300, # DNS cache TTL in seconds
                force_close=False, # Keep-alive connections
                enable_cleanup_closed=True
            )
            self.session = aiohttp.ClientSession(
                timeout=self.timeout,
                connector=connector,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'CryptoStatArb/2.0'
                }
            )
        return self.session
    
    async def close(self):
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
        logger.info(f"Hyperliquid collector closed. Stats: {self.collection_stats}")
    
    def _format_symbol(self, symbol: str) -> str:
        """Convert to Hyperliquid symbol format (BTC, ETH)."""
        symbol = symbol.upper()
        for suffix in ['/USDT', '-USDT', ':USDT', 'USDT', '/USD', '-USD', 'PERP']:
            symbol = symbol.replace(suffix, '')
        return symbol
    
    async def _make_request(self, payload: Dict) -> Any:
        """Make POST request to Hyperliquid API."""
        async with self._request_sem:
            session = await self._get_session()

            async def _request():
                acquire_result = await self.rate_limiter.acquire(timeout=120.0)
                if hasattr(acquire_result, 'acquired') and not acquire_result.acquired:
                    raise TimeoutError("Rate limiter timeout after 120s")
                self.collection_stats['api_calls'] += 1

                async with session.post(self.BASE_URL, json=payload) as response:
                    if response.status == 429:
                        if not hasattr(self, '_hl_rate_limit_logged'):
                            logger.warning("Hyperliquid rate limited - reducing request rate")
                            self._hl_rate_limit_logged = True
                        await asyncio.sleep(30) # Shorter wait, let retry handler manage
                        raise aiohttp.ClientResponseError(
                            response.request_info, response.history, status=429
                        )

                    if response.status == 500:
                        # Server error - log at debug and let retry handler manage
                        logger.debug(f"Hyperliquid 500 Internal Server Error")
                        raise aiohttp.ClientResponseError(
                            response.request_info, response.history, status=500
                        )

                    if response.status == 502 or response.status == 503 or response.status == 504:
                        # Gateway/service unavailable - temporary, allow retry
                        logger.debug(f"Hyperliquid {response.status} - temporary error")
                        raise aiohttp.ClientResponseError(
                            response.request_info, response.history, status=response.status
                        )

                    if response.status == 400:
                        # Bad request - likely invalid symbol, don't retry
                        text = await response.text()
                        logger.debug(f"Hyperliquid 400 Bad Request: {text[:100]}")
                        return None # Return None to skip this request

                    response.raise_for_status()
                    return await response.json()

            try:
                return await self.retry_handler.execute(_request)
            except Exception as e:
                logger.error(f"Request failed: {e}")
                self.collection_stats['errors'] += 1
                raise
    
    async def _fetch_funding_rate_single(
        self, symbol: str, start_dt: datetime, end_dt: datetime
    ) -> List[Dict]:
        """Fetch funding rates for a single symbol (internal helper for parallelization)."""
        hl_symbol = self._format_symbol(symbol)
        logger.info(f" Collecting funding for {symbol} ({hl_symbol})")

        symbol_records = []
        current_end = end_dt

        while current_end > start_dt:
            current_start = max(
                start_dt,
                current_end - timedelta(hours=self.MAX_HOURS_PER_REQUEST - 1)
            )

            payload = {
                "type": "fundingHistory",
                "coin": hl_symbol,
                "startTime": int(current_start.timestamp() * 1000),
                "endTime": int(current_end.timestamp() * 1000)
            }

            try:
                data = await self._make_request(payload)

                if not data:
                    current_end = current_start - timedelta(hours=1)
                    continue

                for record in data:
                    timestamp = pd.to_datetime(record.get('time', 0), unit='ms', utc=True)
                    hourly_rate = safe_float(record.get('fundingRate', 0))
                    premium = safe_float(record.get('premium', 0))

                    fr = HyperliquidFundingRate(
                        timestamp=timestamp,
                        symbol=symbol.upper(),
                        funding_rate=hourly_rate,
                        premium=premium,
                        mark_price=None
                    )
                    symbol_records.append(fr.to_dict())

            except Exception as e:
                logger.error(f"Error fetching {hl_symbol}: {e}")
                break

            current_end = current_start - timedelta(hours=1)

        logger.info(f" {symbol}: {len(symbol_records)} hourly records")
        return symbol_records

    async def fetch_funding_rates(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates.

        IMPORTANT: Returns HOURLY funding rates with 8h normalization.

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH'])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns:
                - timestamp: UTC datetime
                - symbol: Token symbol
                - funding_rate: Raw hourly rate
                - funding_rate_8h: Normalized 8h equivalent
                - funding_rate_annualized: Annualized rate
                - premium: Funding premium
                - trend: Funding trend classification
                - sentiment: Market sentiment
                - venue: 'hyperliquid'
                - funding_interval: 'hourly'
        """
        logger.info(f"Fetching Hyperliquid funding for {len(symbols)} symbols")

        start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)

        # SPEEDUP: Process all symbols in parallel instead of sequentially
        # Rate limiting will control actual throughput
        tasks = [self._fetch_funding_rate_single(symbol, start_dt, end_dt) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine all results
        all_records = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error fetching funding rates: {result}")
                continue
            if result:
                all_records.extend(result)
                self.collection_stats['records_collected'] += len(result)

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        df = df.drop_duplicates(['timestamp', 'symbol'])

        logger.info(f"Collected {len(df)} funding records from Hyperliquid")
        return df
    
    async def _fetch_ohlcv_single(
        self, symbol: str, hl_interval: str, start_ts: int, end_ts: int
    ) -> List[Dict]:
        """
        Fetch OHLCV for a single symbol with pagination (internal helper).

        CRITICAL FIX: Hyperliquid candleSnapshot API limited to 5000 candles per request.
        This method now paginates through the date range to collect full history.

        For long historical ranges (>30 days), consider using daily interval ('1d')
        which gives 5000 days = 13+ years of history in a single request.
        """
        hl_symbol = self._format_symbol(symbol)
        interval_ms = self.INTERVAL_MS.get(hl_interval, 3_600_000) # Default to 1h

        # Calculate max time range per request based on 5000 candle limit
        max_range_ms = self.MAX_CANDLES_PER_REQUEST * interval_ms

        symbol_records = []
        current_start = start_ts

        try:
            while current_start < end_ts:
                # Calculate end of this chunk (limited by API max candles)
                chunk_end = min(current_start + max_range_ms, end_ts)

                payload = {
                    "type": "candleSnapshot",
                    "req": {
                        "coin": hl_symbol,
                        "interval": hl_interval,
                        "startTime": current_start,
                        "endTime": chunk_end
                    }
                }

                data = await self._make_request(payload)

                if not data:
                    # Move to next chunk even if this one fails
                    current_start = chunk_end
                    continue

                for candle in data:
                    ohlcv = HyperliquidOHLCV(
                        timestamp=pd.to_datetime(candle['t'], unit='ms', utc=True),
                        symbol=symbol.upper(),
                        open=float(candle['o']),
                        high=float(candle['h']),
                        low=float(candle['l']),
                        close=float(candle['c']),
                        volume=float(candle['v']),
                        trades=int(candle.get('n', 0))
                    )
                    symbol_records.append(ohlcv.to_dict())

                # Move to next chunk
                current_start = chunk_end

                # Small delay between paginated requests
                if current_start < end_ts:
                    await asyncio.sleep(0.1)

            if symbol_records:
                logger.info(f" {symbol}: {len(symbol_records)} OHLCV records")

        except Exception as e:
            logger.error(f"Error fetching {hl_symbol} OHLCV: {e}")

        return symbol_records

    async def fetch_ohlcv(
        self, symbols: List[str], timeframe: str,
        start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candlestick data with automatic pagination.

        CRITICAL FIX: For long historical ranges, this now:
        1. Uses pagination to fetch all data (5000 candles per request limit)
        2. Automatically uses daily interval for ranges > 200 days for efficiency
           (5000 daily candles = 13+ years of history)

        Args:
            symbols: List of symbols
            timeframe: Candle interval ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days_range = (end_dt - start_dt).days

        hl_interval = self.TIMEFRAME_MAP.get(timeframe, timeframe)

        # SMART INTERVAL SELECTION: For very long ranges, suggest or auto-use daily
        # 5000 hourly candles = ~208 days, 5000 daily = ~13.7 years
        effective_interval = hl_interval
        if days_range > 200 and hl_interval in ['1m', '5m', '15m', '30m', '1h']:
            # For long historical collection, daily is more efficient
            # Each request gets 5000 candles - daily gives 13+ years vs hourly ~208 days
            if timeframe == '1h':
                logger.info(f"Long range detected ({days_range} days). Using pagination for {len(symbols)} symbols.")
            # Still use requested interval but pagination handles the limit

        logger.info(f"Fetching Hyperliquid OHLCV for {len(symbols)} symbols ({effective_interval}, {days_range} days)")

        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)

        # SPEEDUP: Process all symbols in parallel instead of sequentially
        # Rate limiting will control actual throughput
        tasks = [self._fetch_ohlcv_single(symbol, effective_interval, start_ts, end_ts) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine all results
        all_records = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error fetching OHLCV: {result}")
                continue
            if result:
                all_records.extend(result)
                self.collection_stats['records_collected'] += len(result)

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        # Remove duplicates that might occur from pagination overlap
        df = df.drop_duplicates(['timestamp', 'symbol'], keep='first')

        logger.info(f"Collected {len(df)} OHLCV records from Hyperliquid")
        return df
    
    async def fetch_open_interest(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch current open interest snapshot.

        Note: Hyperliquid doesn't provide historical OI via API.
        Uses 'metaAndAssetCtxs' endpoint which includes OI data.

        Args:
            symbols: List of symbols

        Returns:
            DataFrame with current OI snapshot
        """
        # Use metaAndAssetCtxs which returns both metadata and asset contexts with OI
        payload = {"type": "metaAndAssetCtxs"}

        try:
            data = await self._make_request(payload)

            # Response is [meta, assetCtxs]
            if not isinstance(data, list) or len(data) < 2:
                logger.warning(f"Unexpected Hyperliquid response format: {type(data)}")
                return pd.DataFrame()

            meta = data[0]
            asset_ctxs = data[1]
            universe = meta.get('universe', [])

            records = []
            symbol_set = {self._format_symbol(s) for s in symbols}

            # Map symbol names to indices
            for i, perp in enumerate(universe):
                name = perp.get('name', '')
                if name not in symbol_set:
                    continue

                # Get asset context which has OI data
                if i < len(asset_ctxs):
                    ctx = asset_ctxs[i]
                    oi_raw = safe_float(ctx.get('openInterest', 0))
                    mark_px = safe_float(ctx.get('markPx', 0))
                    oracle_px = safe_float(ctx.get('oraclePx', 0))
                    funding = safe_float(ctx.get('funding', 0))

                    oi = HyperliquidOpenInterest(
                        timestamp=datetime.now(timezone.utc),
                        symbol=name,
                        open_interest=oi_raw,
                        open_interest_usd=oi_raw * mark_px,
                        mark_price=mark_px,
                        index_price=oracle_px,
                        funding_rate=funding
                    )
                    records.append(oi.to_dict())
                    self.collection_stats['records_collected'] += 1

            return pd.DataFrame(records)

        except Exception as e:
            logger.error(f"Error fetching Hyperliquid OI: {e}")
            return pd.DataFrame()
    
    async def fetch_orderbook(self, symbol: str, depth: int = 20) -> Optional[HyperliquidOrderBook]:
        """
        Fetch order book snapshot.
        
        Args:
            symbol: Symbol
            depth: Number of levels
            
        Returns:
            HyperliquidOrderBook or None
        """
        hl_symbol = self._format_symbol(symbol)
        
        payload = {
            "type": "l2Book",
            "coin": hl_symbol
        }
        
        try:
            data = await self._make_request(payload)
            levels = data.get('levels', [[], []])
            
            bids = [(float(b['px']), float(b['sz'])) for b in levels[0][:depth]]
            asks = [(float(a['px']), float(a['sz'])) for a in levels[1][:depth]]
            
            return HyperliquidOrderBook(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol.upper(),
                bids=bids,
                asks=asks
            )
            
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return None
    
    async def fetch_markets(self) -> pd.DataFrame:
        """Fetch all available markets."""
        payload = {"type": "meta"}
        
        try:
            data = await self._make_request(payload)
            records = []
            
            for perp in data.get('universe', []):
                market = HyperliquidMarket(
                    symbol=perp['name'],
                    size_decimals=int(perp.get('szDecimals', 0)),
                    max_leverage=int(perp.get('maxLeverage', 50)),
                    tick_size=float(perp.get('tickSize', 0.01)),
                    min_size=float(perp.get('minSize', 0.001)),
                    status=MarketStatus.ACTIVE
                )
                records.append(market.to_dict())
            
            return pd.DataFrame(records)
            
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return pd.DataFrame()
    
    async def get_available_symbols(self) -> List[str]:
        """Get list of available perpetual symbols."""
        df = await self.fetch_markets()
        return df['symbol'].tolist() if not df.empty else []
    
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
            logger.error(f"Hyperliquid collect_funding_rates error: {e}")
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
            logger.error(f"Hyperliquid collect_ohlcv error: {e}")
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

        Note: Hyperliquid only provides current OI snapshot, not historical.
        start_date and end_date are ignored.

        Standardized method name for collection manager compatibility.
        """
        try:
            return await self.fetch_open_interest(symbols=symbols)
        except Exception as e:
            logger.error(f"Hyperliquid collect_open_interest error: {e}")
            return pd.DataFrame()

    async def collect_positions(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect positions data - wraps fetch_open_interest().

        Note: Hyperliquid's open interest data includes aggregate position data.
        Individual positions are not publicly available.

        Standardized method name for collection manager compatibility.
        """
        try:
            oi_data = await self.fetch_open_interest(symbols=symbols)

            if not oi_data.empty:
                # Add position-specific columns
                oi_data['position_type'] = 'aggregate'
                oi_data['data_type'] = 'positions'
                return oi_data

            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Hyperliquid collect_positions error: {e}")
            return pd.DataFrame()

    async def fetch_recent_trades(
        self,
        symbol: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch recent trades for a symbol.

        Uses the public recentTrades endpoint (no authentication required).

        Args:
            symbol: Symbol (e.g., 'BTC', 'ETH')
            limit: Maximum number of trades (API returns most recent)

        Returns:
            DataFrame with trade data including:
                - timestamp: Trade time
                - symbol: Asset symbol
                - price: Trade price
                - size: Trade size
                - side: 'buy' or 'sell' (taker side)
                - trade_id: Unique trade identifier
                - notional_usd: Price * size
        """
        hl_symbol = self._format_symbol(symbol)

        payload = {
            "type": "recentTrades",
            "coin": hl_symbol
        }

        try:
            data = await self._make_request(payload)

            if not data:
                return pd.DataFrame()

            records = []
            for trade in data[:limit]:
                try:
                    # Parse trade data
                    # side: "A" = buy (ask hit), "B" = sell (bid hit)
                    side_map = {"A": "buy", "B": "sell"}
                    ts = datetime.fromtimestamp(trade['time'] / 1000, tz=timezone.utc)
                    price = float(trade['px'])
                    size = float(trade['sz'])

                    records.append({
                        'timestamp': ts.isoformat(),
                        'symbol': hl_symbol,
                        'price': price,
                        'size': size,
                        'side': side_map.get(trade.get('side', 'A'), 'unknown'),
                        'trade_id': trade.get('tid', ''),
                        'notional_usd': price * size,
                        'venue': self.VENUE,
                        'venue_type': self.VENUE_TYPE,
                    })
                except (KeyError, ValueError) as e:
                    logger.warning(f"Parse error for trade: {e}")
                    continue

            df = pd.DataFrame(records)
            self.collection_stats['records_collected'] += len(df)
            return df

        except Exception as e:
            logger.error(f"Error fetching recent trades for {symbol}: {e}")
            return pd.DataFrame()

    async def _fetch_recent_trades_single(
        self, symbol: str, limit: int
    ) -> pd.DataFrame:
        """Fetch recent trades for a single symbol (internal helper for parallelization)."""
        try:
            df = await self.fetch_recent_trades(
                symbol=symbol,
                limit=limit
            )
            if not df.empty:
                logger.info(f" {symbol}: {len(df)} recent trades")
            return df
        except Exception as e:
            logger.error(f" {symbol}: error - {e}")
            return pd.DataFrame()

    async def collect_trades(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect trade data from Hyperliquid.

        Uses the public recentTrades endpoint (no authentication required).
        Note: This returns recent trades only, not historical date-range trades.
        The start_date/end_date parameters are accepted for interface compatibility
        but the API only returns the most recent trades.

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH'])
            start_date: Start date (for interface compatibility)
            end_date: End date (for interface compatibility)
            **kwargs: Additional parameters (e.g., limit)

        Returns:
            DataFrame with trade data
        """
        try:
            limit = kwargs.get('limit', 100)

            # SPEEDUP: Process all symbols in parallel instead of sequentially
            # Rate limiting will control actual throughput
            tasks = [self._fetch_recent_trades_single(symbol, limit) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Combine all results
            all_records = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error fetching trades: {result}")
                    continue
                if isinstance(result, pd.DataFrame) and not result.empty:
                    all_records.append(result)

            if all_records:
                combined = pd.concat(all_records, ignore_index=True)
                logger.info(f"Collected {len(combined)} total trades from Hyperliquid")
                return combined

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Hyperliquid collect_trades error: {e}")
            return pd.DataFrame()

    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.collection_stats.copy()

# =============================================================================
# Utility Functions
# =============================================================================

def normalize_funding_to_8h(
    funding_df: pd.DataFrame,
    source_interval_hours: float = 1.0
) -> pd.DataFrame:
    """
    Normalize funding rates from any interval to 8-hour equivalent.
    
    Use for cross-venue comparison between Hyperliquid (hourly) and CEX (8h).
    
    Args:
        funding_df: DataFrame with 'funding_rate' column
        source_interval_hours: Hours between funding payments
            - Hyperliquid: 1 (hourly)
            - Binance/Bybit: 8
            
    Returns:
        DataFrame with 'funding_rate_8h' column added
    """
    df = funding_df.copy()
    multiplier = 8.0 / source_interval_hours
    df['funding_rate_8h'] = df['funding_rate'] * multiplier
    
    logger.info(f"Normalized funding from {source_interval_hours}h to 8h (Ã—{multiplier})")
    return df

def aggregate_hourly_to_8h(
    hourly_df: pd.DataFrame,
    method: str = 'sum'
) -> pd.DataFrame:
    """
    Aggregate hourly funding data to 8-hour periods.
    
    Produces data at 8-hour intervals (00:00, 08:00, 16:00 UTC)
    matching CEX funding timestamps.
    
    Args:
        hourly_df: Hourly funding DataFrame
        method: 'sum' (add 8 hourly rates) or 'mean' (average Ã— 8)
            
    Returns:
        DataFrame with 8-hour aggregated funding
    """
    if hourly_df.empty:
        return hourly_df
    
    df = hourly_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['period_8h'] = df['timestamp'].dt.floor('8H')
    
    if method == 'sum':
        agg_df = df.groupby(['period_8h', 'symbol']).agg({
            'funding_rate': 'sum',
            'premium': 'sum' if 'premium' in df.columns else 'first',
            'venue': 'first',
            'venue_type': 'first'
        }).reset_index()
    else:
        agg_df = df.groupby(['period_8h', 'symbol']).agg({
            'funding_rate': lambda x: x.mean() * 8,
            'premium': 'mean' if 'premium' in df.columns else 'first',
            'venue': 'first',
            'venue_type': 'first'
        }).reset_index()
    
    agg_df = agg_df.rename(columns={'period_8h': 'timestamp', 'funding_rate': 'funding_rate_8h'})
    agg_df['funding_rate_annualized'] = agg_df['funding_rate_8h'] * 1095
    agg_df['funding_interval'] = '8h_aggregated'
    
    logger.info(f"Aggregated {len(df)} hourly records to {len(agg_df)} 8h periods")
    return agg_df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

async def test_hyperliquid_collector():
    """Test Hyperliquid collector functionality."""
    config = {'rate_limit': 30}
    
    async with HyperliquidCollector(config) as collector:
        print("=" * 60)
        print("Hyperliquid Collector Test")
        print("=" * 60)
        
        markets = await collector.fetch_markets()
        if not markets.empty:
            print(f"\n1. Available markets: {len(markets)}")
            print(f" First 5: {markets['symbol'].head().tolist()}")
        
        print(f"\nStats: {collector.get_collection_stats()}")

if __name__ == '__main__':
    asyncio.run(test_hyperliquid_collector())