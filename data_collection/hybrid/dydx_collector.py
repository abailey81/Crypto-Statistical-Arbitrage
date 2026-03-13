"""
dYdX V4 Data Collector - Cosmos-Based Perpetuals

validated collector for dYdX V4 perpetual futures.
Sovereign Cosmos appchain with decentralized order book matching.

===============================================================================
PROTOCOL OVERVIEW
===============================================================================

dYdX V4 is a sovereign blockchain built on Cosmos SDK featuring:
    - Decentralized order book matching (validators run matching engine)
    - HOURLY funding rate payments (like Hyperliquid, differs from CEX)
    - Up to 20x leverage on major pairs
    - No centralized sequencer or operator
    - Native USDC as settlement currency

Key Differences from dYdX V3:
    - V3: Ethereum L2 (StarkEx) - centralized sequencer
    - V4: Cosmos appchain - fully decentralized
    - V4 uses native USDC, not bridged

Architecture:
    - Consensus: CometBFT (Tendermint)
    - Matching: Validators run order book matching
    - Settlement: On-chain, block-level
    - Data: Indexer API for historical queries

===============================================================================
FUNDING RATE MECHANISM (CRITICAL)
===============================================================================

IMPORTANT: dYdX V4 uses HOURLY funding, NOT 8-hour like CEX.

Funding Comparison:
    ============== ================ ================= ================
    Venue Interval To Compare to CEX Annualization
    ============== ================ ================= ================
    dYdX V4 1 hour Ã— 8 Ã— 8760
    Hyperliquid 1 hour Ã— 8 Ã— 8760
    Binance 8 hours - Ã— 1095
    Bybit 8 hours - Ã— 1095
    ============== ================ ================= ================

Normalization Formula:
    - 8h_equivalent = hourly_rate Ã— 8
    - annualized = hourly_rate Ã— 8760 (24 Ã— 365)

This collector provides BOTH raw hourly rates AND normalized 8h equivalents.

===============================================================================
API DOCUMENTATION
===============================================================================

Indexer Base URL: https://indexer.dydx.trade/v4

Endpoints:
    - /perpetualMarkets: Market metadata and current state
    - /historicalFunding/{ticker}: Historical funding rates
    - /candles/perpetualMarkets/{ticker}: OHLCV candlesticks
    - /orderbooks/perpetualMarket/{ticker}: Order book snapshot
    - /trades/perpetualMarket/{ticker}: Historical trades

Rate Limits:
    - ~100 requests per minute (conservative estimate)
    - No official documentation on exact limits
    - Indexer may have different limits than chain RPC

===============================================================================
DATA TYPES COLLECTED
===============================================================================

Funding Rates:
    - Raw hourly funding rate
    - 8-hour normalized equivalent
    - Annualized rate
    - Oracle price at funding time

OHLCV:
    - Open, High, Low, Close prices
    - USD volume
    - Trade count

Open Interest:
    - Current OI per market
    - Oracle and mark prices
    - Margin requirements

Market Metadata:
    - Perpetual contract specifications
    - Tick and step sizes
    - Margin fractions
    - Trading status

Order Book:
    - Bid/ask levels
    - Depth analysis
    - Spread calculation

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

1. CEX vs dYdX Funding Arbitrage:
   - Compare 8h-normalized dYdX funding to Binance/Bybit
   - Identify persistent funding divergences
   - Account for hourly vs 8h timing differences

2. dYdX vs Hyperliquid Comparison:
   - Both use hourly funding - direct comparison possible
   - Different user bases and liquidity profiles
   - Cross-hybrid venue arbitrage

3. Basis Trading:
   - dYdX perp vs CEX spot basis
   - Oracle price divergence monitoring
   - Mark-to-oracle basis analysis

4. Decentralization Alpha:
   - On-chain transparency for position analysis
   - Validator behavior patterns
   - Block-level trade timing

===============================================================================
DATA QUALITY CONSIDERATIONS
===============================================================================

- Hourly funding must be normalized for CEX comparison
- Indexer may lag during high activity
- Block finality affects data freshness (~6 seconds)
- Historical data availability varies by market
- ISO 8601 timestamps with 'Z' suffix (UTC)

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
    HOURLY = 'hourly' # dYdX V4, Hyperliquid
    EIGHT_HOUR = '8h' # CEX venues

class FundingTrend(Enum):
    """Funding rate trend classification."""
    VERY_POSITIVE = 'very_positive' # > 0.05% hourly
    POSITIVE = 'positive' # 0.01% - 0.05%
    NEUTRAL = 'neutral' # -0.01% to 0.01%
    NEGATIVE = 'negative' # -0.05% to -0.01%
    VERY_NEGATIVE = 'very_negative' # < -0.05%

class MarketSentiment(Enum):
    """Market sentiment derived from funding."""
    STRONGLY_BULLISH = 'strongly_bullish'
    BULLISH = 'bullish'
    NEUTRAL = 'neutral'
    BEARISH = 'bearish'
    STRONGLY_BEARISH = 'strongly_bearish'

class MarketStatus(Enum):
    """Market trading status."""
    ACTIVE = 'ACTIVE'
    PAUSED = 'PAUSED'
    CANCEL_ONLY = 'CANCEL_ONLY'
    POST_ONLY = 'POST_ONLY'
    INITIALIZING = 'INITIALIZING'
    FINAL_SETTLEMENT = 'FINAL_SETTLEMENT'

class LiquidityDepth(Enum):
    """Order book liquidity classification."""
    VERY_DEEP = 'very_deep' # > $10M within 1%
    DEEP = 'deep' # $5M - $10M
    MODERATE = 'moderate' # $1M - $5M
    SHALLOW = 'shallow' # $500K - $1M
    THIN = 'thin' # < $500K

class Timeframe(Enum):
    """Supported OHLCV timeframes."""
    ONE_MIN = '1MIN'
    FIVE_MIN = '5MINS'
    FIFTEEN_MIN = '15MINS'
    THIRTY_MIN = '30MINS'
    ONE_HOUR = '1HOUR'
    FOUR_HOUR = '4HOURS'
    ONE_DAY = '1DAY'

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class DYDXFundingRate:
    """
    dYdX V4 funding rate data with normalization.
    
    IMPORTANT: Raw rate is HOURLY. Use funding_rate_8h for CEX comparison.
    """
    timestamp: datetime
    symbol: str
    funding_rate: float # Raw hourly rate
    oracle_price: float # Price at funding time
    
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
        rate_pct = self.funding_rate * 100
        if rate_pct > 0.05:
            return FundingTrend.VERY_POSITIVE
        elif rate_pct > 0.01:
            return FundingTrend.POSITIVE
        elif rate_pct > -0.01:
            return FundingTrend.NEUTRAL
        elif rate_pct > -0.05:
            return FundingTrend.NEGATIVE
        else:
            return FundingTrend.VERY_NEGATIVE
    
    @property
    def sentiment(self) -> MarketSentiment:
        """Derive market sentiment from funding."""
        rate_pct = self.funding_rate * 100
        if rate_pct > 0.05:
            return MarketSentiment.STRONGLY_BULLISH
        elif rate_pct > 0.01:
            return MarketSentiment.BULLISH
        elif rate_pct > -0.01:
            return MarketSentiment.NEUTRAL
        elif rate_pct > -0.05:
            return MarketSentiment.BEARISH
        else:
            return MarketSentiment.STRONGLY_BEARISH
    
    @property
    def is_arbitrage_opportunity(self) -> bool:
        """Check if funding suggests arb (> 0.1% 8h equiv)."""
        return abs(self.funding_rate_8h) > 0.001
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with computed fields."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'funding_rate': self.funding_rate,
            'funding_rate_8h': self.funding_rate_8h,
            'funding_rate_annualized': self.funding_rate_annualized,
            'funding_rate_daily': self.funding_rate_daily,
            'oracle_price': self.oracle_price,
            'trend': self.trend.value,
            'sentiment': self.sentiment.value,
            'is_arbitrage_opportunity': self.is_arbitrage_opportunity,
            'venue': 'dydx_v4',
            'venue_type': 'hybrid',
            'funding_interval': 'hourly',
        }

@dataclass
class DYDXOHLCV:
    """OHLCV candle data from dYdX V4."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float # USD volume
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
            'venue': 'dydx_v4',
            'venue_type': 'hybrid',
            'market_type': 'perpetual',
        }

@dataclass
class DYDXMarket:
    """Market/instrument metadata from dYdX V4."""
    symbol: str
    dydx_symbol: str # BTC-USD format
    status: MarketStatus
    tick_size: float
    step_size: float
    initial_margin: float # Initial margin fraction
    maintenance_margin: float
    open_interest: float
    oracle_price: float
    
    @property
    def max_leverage(self) -> float:
        """Calculate max leverage from initial margin."""
        if self.initial_margin > 0:
            return 1 / self.initial_margin
        return 1
    
    @property
    def is_tradeable(self) -> bool:
        """Check if market is tradeable."""
        return self.status == MarketStatus.ACTIVE
    
    @property
    def open_interest_usd(self) -> float:
        """Open interest in USD notional."""
        return self.open_interest * self.oracle_price
    
    @property
    def leverage_tier(self) -> str:
        """Classify leverage tier."""
        lev = self.max_leverage
        if lev >= 20:
            return 'high'
        elif lev >= 10:
            return 'medium'
        else:
            return 'low'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'dydx_symbol': self.dydx_symbol,
            'status': self.status.value,
            'tick_size': self.tick_size,
            'step_size': self.step_size,
            'initial_margin': self.initial_margin,
            'maintenance_margin': self.maintenance_margin,
            'max_leverage': self.max_leverage,
            'open_interest': self.open_interest,
            'open_interest_usd': self.open_interest_usd,
            'oracle_price': self.oracle_price,
            'is_tradeable': self.is_tradeable,
            'leverage_tier': self.leverage_tier,
            'venue': 'dydx_v4',
            'venue_type': 'hybrid',
        }

@dataclass
class DYDXOrderBook:
    """Order book snapshot from dYdX V4."""
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
        total = self.bid_depth_usd + self.ask_depth_usd
        if total > 20_000_000:
            return LiquidityDepth.VERY_DEEP
        elif total > 10_000_000:
            return LiquidityDepth.DEEP
        elif total > 2_000_000:
            return LiquidityDepth.MODERATE
        elif total > 1_000_000:
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
            'venue': 'dydx_v4',
            'venue_type': 'hybrid',
        }

@dataclass
class DYDXTrade:
    """Trade data from dYdX V4."""
    timestamp: datetime
    symbol: str
    price: float
    size: float
    side: str # 'BUY' or 'SELL'
    
    @property
    def notional_usd(self) -> float:
        """Trade notional in USD."""
        return self.price * self.size
    
    @property
    def is_buy(self) -> bool:
        """Check if taker was buyer."""
        return self.side.upper() == 'BUY'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'price': self.price,
            'size': self.size,
            'side': self.side,
            'notional_usd': self.notional_usd,
            'is_buy': self.is_buy,
            'venue': 'dydx_v4',
            'venue_type': 'hybrid',
        }

# =============================================================================
# Collector Class
# =============================================================================

class DYDXCollector(BaseCollector):
    """
    dYdX V4 data collector for Cosmos-based perpetuals.
    
    validated implementation for decentralized perpetual data.
    Handles hourly funding rate normalization for cross-venue comparison.
    
    KEY DIFFERENCE FROM CEX:
        - dYdX V4 funding is HOURLY (every hour)
        - Binance/Bybit funding is every 8 HOURS
        - Must normalize when comparing:
            * hourly_rate Ã— 8 = equivalent 8h rate
            * hourly_rate Ã— 8760 = annualized
    
    Features:
        - Historical funding rates
        - OHLCV candlestick data
        - Market metadata with margin info
        - Order book snapshots
        - Historical trades
        - Automatic rate limiting
        - Retry with exponential backoff
    
    Attributes:
        VENUE: 'dydx_v4'
        VENUE_TYPE: 'hybrid'
        FUNDING_INTERVAL_HOURS: 1 (hourly funding)
    
    Example:
        >>> config = {'rate_limit': 50}
        >>> async with DYDXCollector(config) as collector:
        ... funding = await collector.fetch_funding_rates(
        ... ['BTC', 'ETH'], '2024-01-01', '2024-03-31'
        ... )
        ... # funding['funding_rate'] = raw hourly
        ... # funding['funding_rate_8h'] = normalized for CEX comparison
    """
    
    VENUE = 'dydx_v4'
    VENUE_TYPE = 'hybrid'
    BASE_URL = 'https://indexer.dydx.trade/v4'
    
    # CRITICAL: dYdX V4 has HOURLY funding
    FUNDING_INTERVAL_HOURS = 1
    
    # API limits
    MAX_FUNDING_RECORDS = 100
    MAX_CANDLES = 1000
    
    # Timeframe mapping
    TIMEFRAME_MAP = {
        '1m': '1MIN', '5m': '5MINS', '15m': '15MINS', '30m': '30MINS',
        '1h': '1HOUR', '4h': '4HOURS', '1d': '1DAY',
    }

    # Symbols that don't have perpetual markets on dYdX V4
    # These will be silently skipped to avoid ticker validation errors
    UNSUPPORTED_SYMBOLS = {
        'FXS', # Not listed on dYdX V4
        'MATIC', # Migrated to POL
        'APE', # Not listed
        'BLUR', # Not listed
        'MKR', # Not listed
        'MOG', # Not listed
        'FLOKI', # Not listed
        'BONK', # Not listed
        'SHIB', # Not listed
        'PEPE', # Not listed
        'LUNC', # Not listed
        'BTTC', # Not listed
        'WIN', # Not listed
        'BTT', # Not listed
        'NFT', # Not listed
        'JASMY', # Not listed
        'ILV', # Not listed
        'ENJ', # Not listed
        'SAND', # Not listed
        'GALA', # Not listed
        'CELO', # Not listed
        'KAVA', # Not listed
        'RSR', # Not listed
        'CHZ', # Not listed
        'ANKR', # Not listed
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize dYdX V4 collector.
        
        Args:
            config: Configuration with options:
                - rate_limit: Requests per minute (default: 50)
                - timeout: Request timeout seconds (default: 30)
                - max_retries: Retry attempts (default: 3)
        """
        config = config or {}
        super().__init__(config)
        
        rate_limit = config.get('rate_limit', 50)
        # Use shared rate limiter to avoid re-initialization overhead
        self.rate_limiter = get_shared_rate_limiter(
            'dydx', rate=rate_limit, per=60.0, burst=5
        )
        
        # OPTIMIZATION: Reduced max_delay from 60s to 30s to avoid long stalls
        self.retry_handler = RetryHandler(
            max_retries=config.get('max_retries', 3),
            base_delay=1.0, max_delay=30.0
        )
        
        self.timeout = aiohttp.ClientTimeout(total=config.get('timeout', 30))
        self.session: Optional[aiohttp.ClientSession] = None
        
        self.collection_stats = {
            'records_collected': 0, 'api_calls': 0, 'errors': 0
        }
        self._symbol_cache: Dict[str, Dict] = {}
        
        self._request_sem = asyncio.Semaphore(2)

        logger.info(f"Initialized dYdX V4 collector (rate_limit={rate_limit}/min)")

    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with optimized connection pooling."""
        if self.session is None or self.session.closed:
            # SPEEDUP: Aggressive connection pooling for high-throughput collection
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
        logger.info(f"dYdX V4 collector closed. Stats: {self.collection_stats}")
    
    def _format_symbol(self, symbol: str) -> str:
        """Convert to dYdX format (BTC-USD)."""
        symbol = symbol.upper()
        # Process longer suffixes first to avoid double-dash issues
        for suffix in ['-USDT', '/USDT', '-USD', '/USD', ':USDT', ':USD', 'USDT', 'USD', 'PERP', '-PERP']:
            symbol = symbol.replace(suffix, '')
        return f"{symbol}-USD"
    
    def _parse_symbol(self, dydx_symbol: str) -> str:
        """Convert dYdX symbol to internal format."""
        return dydx_symbol.replace('-USD', '').upper()
    
    async def _request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make rate-limited request to dYdX indexer API."""
        async with self._request_sem:
            if not self.session:
                await self._get_session()

            acquire_result = await self.rate_limiter.acquire(timeout=120.0)
            if hasattr(acquire_result, 'acquired') and not acquire_result.acquired:
                logger.debug(f"dYdX rate limiter timeout for {endpoint}")
                return None
            self.collection_stats['api_calls'] += 1

            url = f"{self.BASE_URL}{endpoint}"

            async def _do_request():
                async with self.session.get(url, params=params) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.debug(f"dYdX rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        raise aiohttp.ClientError("Rate limited")

                    if response.status == 404:
                        # Ticker not found - this is expected for unsupported symbols
                        logger.debug(f"dYdX ticker not found: {endpoint}")
                        return None # Return None instead of raising

                    if response.status == 400:
                        # Bad request - likely invalid ticker format
                        text = await response.text()
                        if 'ticker' in text.lower() or 'invalid' in text.lower():
                            logger.debug(f"dYdX invalid ticker: {endpoint}")
                            return None
                        raise aiohttp.ClientError(f"HTTP 400: {text[:100]}")

                    if response.status == 500 or response.status == 502 or response.status == 503:
                        # Server error - temporary, allow retry
                        logger.debug(f"dYdX {response.status} - temporary error")
                        raise aiohttp.ClientError(f"HTTP {response.status}")

                    if response.status != 200:
                        text = await response.text()
                        raise aiohttp.ClientError(f"HTTP {response.status}: {text[:200]}")

                    return await response.json()

            try:
                return await self.retry_handler.execute(_do_request)
            except Exception as e:
                logger.error(f"Request failed: {endpoint} - {e}")
                self.collection_stats['errors'] += 1
                return None
    
    async def fetch_markets(self) -> pd.DataFrame:
        """
        Fetch all available perpetual markets.
        
        Returns:
            DataFrame with market metadata
        """
        logger.info("Fetching dYdX V4 markets")
        
        data = await self._request('/perpetualMarkets')
        
        if not data or 'markets' not in data:
            return pd.DataFrame()
        
        records = []
        for ticker, market in data['markets'].items():
            try:
                status = MarketStatus(market.get('status', 'ACTIVE'))
            except ValueError:
                status = MarketStatus.ACTIVE
            
            m = DYDXMarket(
                symbol=self._parse_symbol(ticker),
                dydx_symbol=ticker,
                status=status,
                tick_size=float(market.get('tickSize', 0)),
                step_size=float(market.get('stepSize', 0)),
                initial_margin=float(market.get('initialMarginFraction', 0)),
                maintenance_margin=float(market.get('maintenanceMarginFraction', 0)),
                open_interest=float(market.get('openInterest', 0)),
                oracle_price=float(market.get('oraclePrice', 0))
            )
            records.append(m.to_dict())
            self._symbol_cache[self._parse_symbol(ticker)] = market
        
        df = pd.DataFrame(records)
        logger.info(f"Found {len(df)} dYdX V4 markets")
        return df
    
    async def get_available_symbols(self, min_oi_usd: float = 100_000) -> List[str]:
        """Get available symbols filtered by open interest."""
        markets = await self.fetch_markets()
        if markets.empty:
            return []
        
        active = markets[
            (markets['status'] == 'ACTIVE') &
            (markets['open_interest_usd'] >= min_oi_usd)
        ]
        return active['symbol'].tolist()
    
    async def _fetch_single_funding_rates(
        self, symbol: str, start_dt: datetime, end_dt: datetime
    ) -> List[Dict]:
        """
        Fetch funding rates for a single symbol with reliable pagination.

        CRITICAL FIX: Improved pagination for long historical ranges:
        - Uses effectiveBeforeOrAt for backward pagination
        - Properly handles rate limits and empty responses
        - Continues paginating until start_dt is reached
        """
        try:
            # Skip symbols that don't have perpetual markets on dYdX V4
            base_symbol = symbol.upper().replace('-USD', '').replace('USDT', '').replace('PERP', '')
            if base_symbol in self.UNSUPPORTED_SYMBOLS:
                logger.debug(f" Skipping {symbol} - not listed on dYdX V4")
                return []

            dydx_symbol = self._format_symbol(symbol)
            logger.info(f" Collecting funding for {symbol} ({dydx_symbol})")

            all_records = []
            current_end = end_dt
            empty_response_count = 0
            max_empty_responses = 3 # Stop after 3 consecutive empty responses

            while current_end > start_dt:
                params = {
                    'limit': self.MAX_FUNDING_RECORDS,
                    'effectiveBeforeOrAt': current_end.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                }

                data = await self._request(f'/historicalFunding/{dydx_symbol}', params)

                if not data or 'historicalFunding' not in data:
                    empty_response_count += 1
                    if empty_response_count >= max_empty_responses:
                        logger.debug(f" {symbol}: No more historical data available")
                        break
                    # Move back by 1 day and try again
                    current_end = current_end - timedelta(days=1)
                    continue

                funding_records = data['historicalFunding']
                if not funding_records:
                    empty_response_count += 1
                    if empty_response_count >= max_empty_responses:
                        break
                    current_end = current_end - timedelta(days=1)
                    continue

                # Reset empty count on successful fetch
                empty_response_count = 0

                for record in funding_records:
                    try:
                        effective_at = datetime.fromisoformat(
                            record['effectiveAt'].replace('Z', '+00:00')
                        )

                        if effective_at < start_dt or effective_at > end_dt:
                            continue

                        fr = DYDXFundingRate(
                            timestamp=effective_at,
                            symbol=symbol.upper(),
                            funding_rate=safe_float(record.get('rate', 0)),
                            oracle_price=safe_float(record.get('price', 0))
                        )
                        all_records.append(fr.to_dict())

                    except (KeyError, ValueError) as e:
                        logger.warning(f"Parse error: {e}")

                # Move to earlier window
                if funding_records:
                    earliest = min(
                        datetime.fromisoformat(r['effectiveAt'].replace('Z', '+00:00'))
                        for r in funding_records
                    )
                    # Move 1 second before the earliest record to get the next batch
                    current_end = earliest - timedelta(seconds=1)

                    # If we've fetched a partial page, we've reached the end
                    if len(funding_records) < self.MAX_FUNDING_RECORDS:
                        break
                else:
                    break

                await asyncio.sleep(0.1)

            logger.info(f" {symbol}: {len(all_records)} hourly records")
            return all_records

        except Exception as e:
            logger.error(f"Error fetching funding rates for {symbol}: {e}")
            return []

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
                - oracle_price: Price at funding time
                - trend: Funding trend classification
                - sentiment: Market sentiment
                - venue: 'dydx_v4'
                - funding_interval: 'hourly'
        """
        logger.info(f"Fetching dYdX V4 funding for {len(symbols)} symbols")

        start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)

        # Parallelize funding rate fetching
        tasks = [self._fetch_single_funding_rates(symbol, start_dt, end_dt) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter and flatten results
        all_records = []
        for result in results:
            if isinstance(result, list):
                all_records.extend(result)

        self.collection_stats['records_collected'] += len(all_records)

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        logger.info(f"Collected {len(df)} funding records from dYdX V4")
        return df
    
    async def _fetch_single_ohlcv(
        self, symbol: str, timeframe: str, resolution: str,
        start_dt: datetime, end_dt: datetime
    ) -> List[Dict]:
        """Fetch OHLCV data for a single symbol."""
        try:
            # Skip symbols that don't have perpetual markets on dYdX V4
            base_symbol = symbol.upper().replace('-USD', '').replace('USDT', '').replace('PERP', '')
            if base_symbol in self.UNSUPPORTED_SYMBOLS:
                return []

            dydx_symbol = self._format_symbol(symbol)
            current_start = start_dt

            all_records = []

            while current_start < end_dt:
                params = {
                    'resolution': resolution,
                    'limit': self.MAX_CANDLES,
                    'fromISO': current_start.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                    'toISO': end_dt.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                }

                data = await self._request(
                    f'/candles/perpetualMarkets/{dydx_symbol}', params
                )

                if not data or 'candles' not in data:
                    break

                candles = data['candles']
                if not candles:
                    break

                for candle in candles:
                    try:
                        ts = datetime.fromisoformat(
                            candle['startedAt'].replace('Z', '+00:00')
                        )

                        ohlcv = DYDXOHLCV(
                            timestamp=ts,
                            symbol=symbol.upper(),
                            open=float(candle['open']),
                            high=float(candle['high']),
                            low=float(candle['low']),
                            close=float(candle['close']),
                            volume=float(candle.get('usdVolume', 0)),
                            trades=int(candle.get('trades', 0))
                        )
                        all_records.append(ohlcv.to_dict())

                    except (KeyError, ValueError):
                        continue

                # Move to next window
                if candles:
                    latest = max(
                        datetime.fromisoformat(c['startedAt'].replace('Z', '+00:00'))
                        for c in candles
                    )
                    current_start = latest + timedelta(seconds=1)
                else:
                    break

                await asyncio.sleep(0.1)

            return all_records

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return []

    async def fetch_ohlcv(
        self, symbols: List[str], timeframe: str,
        start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candlestick data.

        Args:
            symbols: List of symbols
            timeframe: Candle interval ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching dYdX V4 OHLCV for {len(symbols)} symbols ({timeframe})")

        resolution = self.TIMEFRAME_MAP.get(timeframe)
        if not resolution:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)

        # Parallelize OHLCV fetching
        tasks = [
            self._fetch_single_ohlcv(symbol, timeframe, resolution, start_dt, end_dt)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter and flatten results
        all_records = []
        for result in results:
            if isinstance(result, list):
                all_records.extend(result)

        self.collection_stats['records_collected'] += len(all_records)

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        logger.info(f"Collected {len(df)} OHLCV records from dYdX V4")
        return df
    
    async def fetch_orderbook(self, symbol: str, depth: int = 20) -> Optional[DYDXOrderBook]:
        """
        Fetch order book snapshot.
        
        Args:
            symbol: Symbol
            depth: Number of levels
            
        Returns:
            DYDXOrderBook or None
        """
        dydx_symbol = self._format_symbol(symbol)
        
        data = await self._request(f'/orderbooks/perpetualMarket/{dydx_symbol}')
        
        if not data:
            return None
        
        bids = [(float(b['price']), float(b['size'])) for b in data.get('bids', [])[:depth]]
        asks = [(float(a['price']), float(a['size'])) for a in data.get('asks', [])[:depth]]
        
        return DYDXOrderBook(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol.upper(),
            bids=bids,
            asks=asks
        )
    
    async def fetch_trades(
        self, symbol: str, start_date: str, end_date: str, limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch historical trades.
        
        Args:
            symbol: Symbol
            start_date: Start date
            end_date: End date
            limit: Max records per request
            
        Returns:
            DataFrame with trade data
        """
        dydx_symbol = self._format_symbol(symbol)
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        all_trades = []
        current_end = end_dt
        
        while current_end > start_dt:
            params = {
                'limit': limit,
                'createdBeforeOrAt': current_end.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            }
            
            data = await self._request(
                f'/trades/perpetualMarket/{dydx_symbol}', params
            )
            
            if not data or 'trades' not in data:
                break
            
            trades = data['trades']
            if not trades:
                break
            
            for trade in trades:
                try:
                    ts = datetime.fromisoformat(
                        trade['createdAt'].replace('Z', '+00:00')
                    )
                    
                    if ts < start_dt:
                        continue
                    
                    t = DYDXTrade(
                        timestamp=ts,
                        symbol=symbol.upper(),
                        price=float(trade['price']),
                        size=float(trade['size']),
                        side=trade['side']
                    )
                    all_trades.append(t.to_dict())
                    
                except (KeyError, ValueError):
                    continue
            
            earliest = min(
                datetime.fromisoformat(t['createdAt'].replace('Z', '+00:00'))
                for t in trades
            )
            current_end = earliest - timedelta(seconds=1)
            
            await asyncio.sleep(0.1)
        
        if not all_trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_trades)
        return df.sort_values('timestamp').reset_index(drop=True)

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
            logger.error(f"dYdX collect_funding_rates error: {e}")
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
            logger.error(f"dYdX collect_ohlcv error: {e}")
            return pd.DataFrame()

    async def _collect_single_trades(
        self, symbol: str, start_date: str, end_date: str, limit: int
    ) -> pd.DataFrame:
        """Collect trades for a single symbol."""
        try:
            df = await self.fetch_trades(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
            await asyncio.sleep(0.1) # Small delay between symbols
            return df
        except Exception as e:
            logger.error(f"Error collecting trades for {symbol}: {e}")
            return pd.DataFrame()

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
            limit = kwargs.get('limit', 100)

            # Convert dates to string format if needed
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            # Parallelize trade fetching
            tasks = [
                self._collect_single_trades(symbol, start_str, end_str, limit)
                for symbol in symbols
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter valid dataframes
            all_records = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

            if all_records:
                return pd.concat(all_records, ignore_index=True)

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"dYdX collect_trades error: {e}")
            return pd.DataFrame()

    async def collect_open_interest(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect open interest data from dYdX V4 markets.

        dYdX provides current OI snapshot through market data, not historical OI.

        Standardized method name for collection manager compatibility.
        """
        try:
            # Fetch all markets to get OI data
            markets = await self.fetch_markets()
            if markets.empty:
                return pd.DataFrame()

            # Filter to requested symbols
            symbol_filter = {s.upper() for s in symbols}
            filtered = markets[markets['symbol'].isin(symbol_filter)]

            if filtered.empty:
                return pd.DataFrame()

            # Return OI-relevant columns
            oi_df = filtered[['symbol', 'open_interest', 'open_interest_usd', 'oracle_price', 'venue', 'venue_type']].copy()
            oi_df['timestamp'] = datetime.now(timezone.utc).isoformat()
            oi_df['data_type'] = 'snapshot'

            self.collection_stats['records_collected'] += len(oi_df)
            return oi_df

        except Exception as e:
            logger.error(f"dYdX collect_open_interest error: {e}")
            return pd.DataFrame()

    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.collection_stats.copy()

# =============================================================================
# Utility Functions
# =============================================================================

def aggregate_hourly_to_8h(
    df: pd.DataFrame, method: str = 'sum'
) -> pd.DataFrame:
    """
    Aggregate hourly dYdX funding to 8h periods for CEX comparison.
    
    Produces data at 8-hour intervals (00:00, 08:00, 16:00 UTC)
    matching CEX funding timestamps.
    
    Args:
        df: DataFrame with hourly funding rates
        method: 'sum' (add 8 hourly rates) or 'mean' (average Ã— 8)
            
    Returns:
        DataFrame with 8h aggregated funding
    """
    if df.empty:
        return df
    
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['period_8h'] = df['timestamp'].dt.floor('8H')
    
    if method == 'sum':
        agg_df = df.groupby(['symbol', 'period_8h']).agg({
            'funding_rate': 'sum',
            'oracle_price': 'mean',
            'venue': 'first',
            'venue_type': 'first',
        }).reset_index()
    else:
        agg_df = df.groupby(['symbol', 'period_8h']).agg({
            'funding_rate': lambda x: x.mean() * 8,
            'oracle_price': 'mean',
            'venue': 'first',
            'venue_type': 'first',
        }).reset_index()
    
    agg_df = agg_df.rename(columns={
        'period_8h': 'timestamp', 'funding_rate': 'funding_rate_8h'
    })
    agg_df['funding_rate_annualized'] = agg_df['funding_rate_8h'] * 1095
    agg_df['funding_interval'] = '8h_aggregated'
    
    logger.info(f"Aggregated {len(df)} hourly records to {len(agg_df)} 8h periods")
    return agg_df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

async def test_dydx_collector():
    """Test dYdX V4 collector functionality."""
    config = {'rate_limit': 30}
    
    async with DYDXCollector(config) as collector:
        print("=" * 60)
        print("dYdX V4 Collector Test")
        print("=" * 60)
        
        markets = await collector.fetch_markets()
        if not markets.empty:
            print(f"\n1. Available markets: {len(markets)}")
            active = markets[markets['status'] == 'ACTIVE']
            print(f" Active: {len(active)}")
        
        print(f"\nStats: {collector.get_collection_stats()}")

if __name__ == '__main__':
    asyncio.run(test_dydx_collector())