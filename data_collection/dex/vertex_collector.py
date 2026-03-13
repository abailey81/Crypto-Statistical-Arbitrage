"""
Vertex Protocol Data Collector - Hybrid CEX/DEX on Arbitrum

validated collector for Vertex Protocol combining order book
trading with AMM liquidity on Arbitrum.

===============================================================================
PROTOCOL OVERVIEW
===============================================================================

Vertex is a hybrid decentralized exchange on Arbitrum that combines:
    - Central Limit Order Book (CLOB): Traditional order matching
    - Automated Market Maker (AMM): Liquidity backstop
    - Cross-margin system: Unified collateral across positions
    - Integrated clearing: On-chain settlement

Key Characteristics:
    - Hourly funding rate settlements
    - Cross-margining across all positions
    - Low latency via Arbitrum sequencer
    - Insurance fund for socialized losses

===============================================================================
API DOCUMENTATION
===============================================================================

Endpoints:
    - Archive: https://archive.prod.vertexprotocol.com/v1
    - Gateway: https://gateway.prod.vertexprotocol.com/v1
    - Indexer: https://archive.prod.vertexprotocol.com/v1

Data Format:
    - Prices/amounts returned as fixed point (1e18 scale)
    - Timestamps in Unix seconds or milliseconds
    - Product IDs map to specific markets

Rate Limits:
    - Archive API: ~100 requests/minute
    - Gateway API: ~60 requests/minute
    - WebSocket: Unlimited for subscriptions

===============================================================================
PRODUCT IDS
===============================================================================

Spot Markets (even numbers):
    - 0: USDC (quote asset)
    - 1: BTC spot
    - 3: ETH spot
    - 5: ARB spot

Perpetual Markets (even numbers + 1):
    - 2: BTC-PERP
    - 4: ETH-PERP
    - 6: ARB-PERP
    - 8: SOL-PERP

===============================================================================
FUNDING RATE MECHANISM
===============================================================================

Vertex uses hourly funding with the formula:
    Funding Rate = (Mark Price - Index Price) / Index Price * 0.01

Funding payments:
    - Settled every hour
    - Long pays short when mark > index
    - Short pays long when mark < index
    - Capped at 0.1% per hour (876% APR max)

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

1. Cross-Venue Funding Arbitrage:
   - Compare Vertex hourly funding to CEX 8h funding
   - Identify funding rate convergence opportunities
   - Exploit funding rate term structure

2. Hybrid Liquidity Analysis:
   - Order book vs AMM execution quality
   - Price impact comparison
   - Optimal execution routing

3. Cross-Margin Efficiency:
   - Capital efficiency metrics
   - Margin utilization analysis
   - Liquidation risk modeling

4. L2 Arbitrage:
   - Vertex vs other Arbitrum DEXs
   - Cross-L2 price discrepancies
   - Gas optimization analysis

===============================================================================
DATA QUALITY NOTES
===============================================================================

- Fixed point conversion required (divide by 1e18)
- Timestamps vary between endpoints (s vs ms)
- Historical data availability varies by product
- Archive API may have slight delay vs real-time

Version: 2.0.0
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timezone
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

class ProductType(Enum):
    """Vertex product types."""
    SPOT = 'spot'
    PERP = 'perp'

class FundingTrend(Enum):
    """Funding rate trend classification."""
    VERY_POSITIVE = 'very_positive' # > 0.05% hourly
    POSITIVE = 'positive' # 0.01% - 0.05%
    NEUTRAL = 'neutral' # -0.01% to 0.01%
    NEGATIVE = 'negative' # -0.05% to -0.01%
    VERY_NEGATIVE = 'very_negative' # < -0.05%

class MarketSentiment(Enum):
    """Market sentiment based on OI and funding."""
    STRONGLY_BULLISH = 'strongly_bullish'
    BULLISH = 'bullish'
    NEUTRAL = 'neutral'
    BEARISH = 'bearish'
    STRONGLY_BEARISH = 'strongly_bearish'

class LiquidationSeverity(Enum):
    """Liquidation severity classification."""
    MINOR = 'minor' # < $10K
    MODERATE = 'moderate' # $10K - $100K
    SIGNIFICANT = 'significant' # $100K - $1M
    MAJOR = 'major' # > $1M

class TradeSide(Enum):
    """Trade side."""
    BUY = 'buy'
    SELL = 'sell'

class OITrend(Enum):
    """Open interest trend."""
    INCREASING = 'increasing' # > 5%
    STABLE = 'stable' # -5% to 5%
    DECREASING = 'decreasing' # < -5%

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class VertexFundingRate:
    """Vertex funding rate data with analytics."""
    timestamp: datetime
    symbol: str
    product_id: int
    funding_rate: float # Hourly rate (already converted from 1e18)
    mark_price: float
    index_price: float
    
    @property
    def annualized_rate(self) -> float:
        """Annualized funding rate (hourly * 8760)."""
        return self.funding_rate * 8760 * 100
    
    @property
    def daily_rate(self) -> float:
        """Daily funding rate (hourly * 24)."""
        return self.funding_rate * 24 * 100
    
    @property
    def eight_hour_equivalent(self) -> float:
        """8-hour equivalent rate for CEX comparison."""
        return self.funding_rate * 8 * 100
    
    @property
    def trend(self) -> FundingTrend:
        """Classify funding trend."""
        rate_pct = self.funding_rate * 100 # Convert to percentage
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
    def basis_bps(self) -> float:
        """Basis in basis points (mark vs index)."""
        if self.index_price > 0:
            return (self.mark_price - self.index_price) / self.index_price * 10000
        return 0
    
    @property
    def is_contango(self) -> bool:
        """Check if market is in contango (mark > index)."""
        return self.mark_price > self.index_price
    
    @property
    def is_arbitrage_opportunity(self) -> bool:
        """Check if funding presents arbitrage opportunity (>0.1% daily)."""
        return abs(self.daily_rate) > 0.1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'product_id': self.product_id,
            'funding_rate': self.funding_rate,
            'funding_rate_hourly_pct': self.funding_rate * 100,
            'annualized_rate': self.annualized_rate,
            'daily_rate': self.daily_rate,
            'eight_hour_equivalent': self.eight_hour_equivalent,
            'mark_price': self.mark_price,
            'index_price': self.index_price,
            'basis_bps': self.basis_bps,
            'trend': self.trend.value,
            'is_contango': self.is_contango,
            'is_arbitrage_opportunity': self.is_arbitrage_opportunity,
        }

@dataclass
class VertexOHLCV:
    """Vertex OHLCV candle data."""
    timestamp: datetime
    symbol: str
    product_id: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    volume_quote: float
    num_trades: int
    contract_type: str
    
    @property
    def typical_price(self) -> float:
        """Typical price (H+L+C)/3."""
        return (self.high + self.low + self.close) / 3
    
    @property
    def range_pct(self) -> float:
        """Price range as percentage."""
        return (self.high - self.low) / self.low * 100 if self.low > 0 else 0
    
    @property
    def body_pct(self) -> float:
        """Candle body as percentage of range."""
        total_range = self.high - self.low
        body = abs(self.close - self.open)
        return body / total_range * 100 if total_range > 0 else 0
    
    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish."""
        return self.close > self.open
    
    @property
    def avg_trade_size(self) -> float:
        """Average trade size in quote."""
        return self.volume_quote / self.num_trades if self.num_trades > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'product_id': self.product_id,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'volume_quote': self.volume_quote,
            'num_trades': self.num_trades,
            'contract_type': self.contract_type,
            'typical_price': self.typical_price,
            'range_pct': self.range_pct,
            'is_bullish': self.is_bullish,
            'avg_trade_size': self.avg_trade_size,
        }

@dataclass
class VertexOpenInterest:
    """Vertex open interest data."""
    timestamp: datetime
    symbol: str
    product_id: int
    open_interest: float
    open_interest_usd: float
    long_interest: float
    short_interest: float
    
    @property
    def long_short_ratio(self) -> float:
        """Long/short ratio."""
        return self.long_interest / self.short_interest if self.short_interest > 0 else 0
    
    @property
    def long_pct(self) -> float:
        """Long percentage of total OI."""
        total = self.long_interest + self.short_interest
        return self.long_interest / total * 100 if total > 0 else 50
    
    @property
    def short_pct(self) -> float:
        """Short percentage of total OI."""
        return 100 - self.long_pct
    
    @property
    def imbalance_pct(self) -> float:
        """Position imbalance percentage."""
        return abs(self.long_pct - 50) * 2
    
    @property
    def sentiment(self) -> MarketSentiment:
        """Market sentiment from positioning."""
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
    def is_crowded(self) -> bool:
        """Check if position is crowded (>65% one side)."""
        return self.imbalance_pct > 30
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'product_id': self.product_id,
            'open_interest': self.open_interest,
            'open_interest_usd': self.open_interest_usd,
            'long_interest': self.long_interest,
            'short_interest': self.short_interest,
            'long_short_ratio': self.long_short_ratio,
            'long_pct': self.long_pct,
            'short_pct': self.short_pct,
            'imbalance_pct': self.imbalance_pct,
            'sentiment': self.sentiment.value,
            'is_crowded': self.is_crowded,
        }

@dataclass
class VertexTrade:
    """Vertex individual trade data."""
    timestamp: datetime
    symbol: str
    product_id: int
    price: float
    size: float
    side: str
    is_liquidation: bool
    tx_hash: Optional[str]
    contract_type: str
    
    @property
    def notional(self) -> float:
        """Trade notional value."""
        return self.price * self.size
    
    @property
    def trade_side(self) -> TradeSide:
        """Trade side enum."""
        return TradeSide.BUY if self.side == 'buy' else TradeSide.SELL
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'product_id': self.product_id,
            'price': self.price,
            'size': self.size,
            'notional': self.notional,
            'side': self.side,
            'is_liquidation': self.is_liquidation,
            'tx_hash': self.tx_hash,
            'contract_type': self.contract_type,
        }

@dataclass
class VertexLiquidation:
    """Vertex liquidation event data."""
    timestamp: datetime
    symbol: str
    product_id: int
    account: str
    liquidator: str
    size: float
    price: float
    is_long: bool
    pnl: float
    tx_hash: Optional[str]
    
    @property
    def notional(self) -> float:
        """Liquidation notional value."""
        return self.size * self.price
    
    @property
    def severity(self) -> LiquidationSeverity:
        """Classify liquidation severity."""
        notional = self.notional
        if notional > 1_000_000:
            return LiquidationSeverity.MAJOR
        elif notional > 100_000:
            return LiquidationSeverity.SIGNIFICANT
        elif notional > 10_000:
            return LiquidationSeverity.MODERATE
        else:
            return LiquidationSeverity.MINOR
    
    @property
    def position_side(self) -> str:
        """Position side that was liquidated."""
        return 'long' if self.is_long else 'short'
    
    @property
    def loss_pct(self) -> float:
        """Approximate loss percentage."""
        if self.notional > 0:
            return abs(self.pnl) / self.notional * 100
        return 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'product_id': self.product_id,
            'account': self.account,
            'liquidator': self.liquidator,
            'size': self.size,
            'price': self.price,
            'notional': self.notional,
            'is_long': self.is_long,
            'position_side': self.position_side,
            'pnl': self.pnl,
            'severity': self.severity.value,
            'loss_pct': self.loss_pct,
            'tx_hash': self.tx_hash,
        }

@dataclass
class VertexMarketStats:
    """Vertex market statistics."""
    timestamp: datetime
    symbol: str
    product_id: int
    product_type: str
    oracle_price: float
    mark_price: float
    index_price: float
    open_interest: float
    volume_24h: float
    num_trades_24h: int
    funding_rate: float
    
    @property
    def mark_oracle_deviation_bps(self) -> float:
        """Mark vs oracle price deviation in bps."""
        if self.oracle_price > 0:
            return (self.mark_price - self.oracle_price) / self.oracle_price * 10000
        return 0
    
    @property
    def avg_trade_size_24h(self) -> float:
        """Average trade size in 24h."""
        return self.volume_24h / self.num_trades_24h if self.num_trades_24h > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'product_id': self.product_id,
            'product_type': self.product_type,
            'oracle_price': self.oracle_price,
            'mark_price': self.mark_price,
            'index_price': self.index_price,
            'open_interest': self.open_interest,
            'volume_24h': self.volume_24h,
            'num_trades_24h': self.num_trades_24h,
            'funding_rate': self.funding_rate,
            'mark_oracle_deviation_bps': self.mark_oracle_deviation_bps,
            'avg_trade_size_24h': self.avg_trade_size_24h,
        }

# =============================================================================
# Collector Class
# =============================================================================

class VertexCollector(BaseCollector):
    """
    Vertex Protocol data collector.
    
    validated implementation for hybrid CEX/DEX data collection
    on Arbitrum. Supports perpetual futures with hourly funding.
    
    Features:
        - Hourly funding rate data
        - OHLCV candles (spot and perp)
        - Open interest tracking
        - Trade and liquidation events
        - Market statistics
    
    Attributes:
        VENUE: Protocol identifier ('vertex')
        VENUE_TYPE: Protocol type ('HYBRID')
    
    Example:
        >>> config = {'rate_limit': 30}
        >>> async with VertexCollector(config) as collector:
        ... funding = await collector.fetch_funding_rates(
        ... ['BTC', 'ETH'], '2024-01-01', '2024-01-31'
        ... )
        ... ohlcv = await collector.fetch_ohlcv(
        ... ['BTC'], '1h', '2024-01-01', '2024-01-31'
        ... )
    """
    
    VENUE = 'vertex'
    VENUE_TYPE = 'HYBRID'
    
    ARCHIVE_URL = 'https://archive.prod.vertexprotocol.com/v1'
    GATEWAY_URL = 'https://gateway.prod.vertexprotocol.com/v1'
    
    # Product ID mapping: symbol -> {spot: id, perp: id}
    PRODUCT_MAP = {
        'BTC': {'spot': 1, 'perp': 2},
        'ETH': {'spot': 3, 'perp': 4},
        'ARB': {'spot': 5, 'perp': 6},
        'SOL': {'spot': 7, 'perp': 8},
        'MATIC': {'spot': 9, 'perp': 10},
        'OP': {'spot': 11, 'perp': 12},
        'AVAX': {'spot': 13, 'perp': 14},
        'LINK': {'spot': 15, 'perp': 16},
    }
    
    GRANULARITY_MAP = {
        '1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400, '1d': 86400
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Vertex collector."""
        config = config or {}
        super().__init__(config)
        
        rate_limit = config.get('rate_limit', 30)
        # Use shared rate limiter to avoid re-initialization overhead
        self.rate_limiter = get_shared_rate_limiter('vertex', rate=rate_limit, per=60.0, burst=5)
        self.retry_handler = RetryHandler(max_retries=3, base_delay=2.0)
        
        self.timeout = aiohttp.ClientTimeout(total=30)
        self.session: Optional[aiohttp.ClientSession] = None
        
        self.collection_stats = {
            'records_collected': 0, 'api_calls': 0, 'errors': 0
        }
        logger.info(f"Initialized Vertex collector (rate_limit={rate_limit}/min)")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self.session
    
    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
        logger.info(f"Vertex collector closed. Stats: {self.collection_stats}")
    
    async def _query_archive(
        self, endpoint: str, params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Query Vertex archive API."""
        url = f"{self.ARCHIVE_URL}{endpoint}"
        session = await self._get_session()
        await self.rate_limiter.acquire()
        
        try:
            async with session.get(url, params=params) as resp:
                self.collection_stats['api_calls'] += 1
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.error(f"HTTP {resp.status} from Vertex: {await resp.text()}")
                    return None
        except Exception as e:
            logger.error(f"Error querying Vertex archive: {e}")
            self.collection_stats['errors'] += 1
            return None
    
    async def _query_gateway(
        self, method: str, params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Query Vertex gateway API (real-time data)."""
        session = await self._get_session()
        await self.rate_limiter.acquire()
        
        payload = {'method': method, 'params': params or {}}
        
        try:
            async with session.post(self.GATEWAY_URL, json=payload) as resp:
                self.collection_stats['api_calls'] += 1
                if resp.status == 200:
                    result = await resp.json()
                    return result.get('result', result)
                else:
                    logger.error(f"HTTP {resp.status} from Vertex gateway")
                    return None
        except Exception as e:
            logger.error(f"Error querying Vertex gateway: {e}")
            self.collection_stats['errors'] += 1
            return None
    
    async def _fetch_single_symbol_funding_rates(
        self, symbol: str, start_ts: int, end_ts: int
    ) -> List[Dict]:
        """Fetch funding rates for a single symbol."""
        if symbol.upper() not in self.PRODUCT_MAP:
            logger.warning(f"Symbol {symbol} not supported on Vertex")
            return []

        product_id = self.PRODUCT_MAP[symbol.upper()]['perp']
        logger.info(f"Fetching Vertex funding for {symbol} (product_id={product_id})")

        params = {
            'product_id': product_id,
            'start_time': start_ts,
            'end_time': end_ts,
            'limit': 10000
        }

        data = await self._query_archive('/funding_rates', params)

        if not data or 'funding_rates' not in data:
            logger.warning(f"No funding data for {symbol}")
            return []

        all_data = []
        for rate in data.get('funding_rates', []):
            # Convert from 1e18 fixed point
            funding_rate = safe_float(rate.get('funding_rate', 0)) / 1e18

            vfr = VertexFundingRate(
                timestamp=datetime.fromtimestamp(safe_int(rate.get('timestamp', 0)), tz=timezone.utc),
                symbol=symbol.upper(),
                product_id=product_id,
                funding_rate=funding_rate,
                mark_price=safe_float(rate.get('mark_price', 0)) / 1e18,
                index_price=safe_float(rate.get('index_price', 0)) / 1e18
            )

            all_data.append({
                **vfr.to_dict(),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE,
                'funding_interval': 'hourly',
                'chain': 'arbitrum'
            })

        return all_data

    async def fetch_funding_rates(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates from Vertex.

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH'])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with hourly funding rate data
        """
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        # Parallelize symbol fetching
        tasks = [
            self._fetch_single_symbol_funding_rates(symbol, start_ts, end_ts)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten and filter valid results
        all_data = []
        for r in results:
            if isinstance(r, list):
                all_data.extend(r)

        self.collection_stats['records_collected'] += len(all_data)

        df = pd.DataFrame(all_data)
        if not df.empty:
            df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        return df

    async def _normalize_single_symbol(self, symbol: str) -> str:
        """Helper method to normalize a single symbol."""
        # Remove common perpetual suffixes
        normalized = symbol.upper()
        for suffix in ['-PERP', ':USDT', '/USDT', '-USD', ':USD', '/USD']:
            normalized = normalized.replace(suffix, '')
        return normalized

    async def collect_funding_rates(self, symbols: List[str], start_date: Any, end_date: Any, **kwargs) -> pd.DataFrame:
        """Standardized collect_funding_rates wrapper - wraps fetch_funding_rates()."""
        try:
            # Normalize symbols in parallel - strip '-PERP', ':USDT', '/USDT' suffixes
            tasks = [self._normalize_single_symbol(symbol) for symbol in symbols]
            normalized_symbols = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out any exceptions and keep only valid strings
            normalized_symbols = [s for s in normalized_symbols if isinstance(s, str)]

            # Convert datetime to string
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            df = await self.fetch_funding_rates(symbols=normalized_symbols, start_date=start_str, end_date=end_str)

            return df
        except Exception as e:
            logger.error(f"Vertex collect_funding_rates error: {e}")
            return pd.DataFrame()

    async def _fetch_single_symbol_ohlcv(
        self, symbol: str, contract_type: str, granularity: int, start_ts: int, end_ts: int
    ) -> List[Dict]:
        """Fetch OHLCV for a single symbol."""
        if symbol.upper() not in self.PRODUCT_MAP:
            return []

        product_id = self.PRODUCT_MAP[symbol.upper()][contract_type]
        logger.info(f"Fetching Vertex OHLCV for {symbol} ({contract_type})")

        all_data = []

        # Fetch in 30-day chunks
        current_ts = start_ts

        while current_ts < end_ts:
            chunk_end = min(current_ts + 86400 * 30, end_ts)

            params = {
                'product_id': product_id,
                'granularity': granularity,
                'start_time': current_ts,
                'end_time': chunk_end,
                'limit': 1000
            }

            data = await self._query_archive('/candlesticks', params)

            if not data or 'candlesticks' not in data:
                current_ts = chunk_end
                continue

            for candle in data.get('candlesticks', []):
                vohlcv = VertexOHLCV(
                    timestamp=datetime.fromtimestamp(candle['timestamp'], tz=timezone.utc),
                    symbol=symbol.upper(),
                    product_id=product_id,
                    open=float(candle['open']) / 1e18,
                    high=float(candle['high']) / 1e18,
                    low=float(candle['low']) / 1e18,
                    close=float(candle['close']) / 1e18,
                    volume=float(candle.get('volume', 0)) / 1e18,
                    volume_quote=float(candle.get('volume_quote', 0)) / 1e18,
                    num_trades=int(candle.get('num_trades', 0)),
                    contract_type=contract_type.upper()
                )

                all_data.append({
                    **vohlcv.to_dict(),
                    'venue': self.VENUE,
                    'venue_type': self.VENUE_TYPE,
                    'chain': 'arbitrum'
                })

            current_ts = chunk_end

        return all_data

    async def fetch_ohlcv(
        self, symbols: List[str], timeframe: str,
        start_date: str, end_date: str, contract_type: str = 'perp'
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles from Vertex.

        Args:
            symbols: List of symbols
            timeframe: '1m', '5m', '15m', '1h', '4h', '1d'
            start_date: Start date
            end_date: End date
            contract_type: 'spot' or 'perp'

        Returns:
            DataFrame with OHLCV data
        """
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        granularity = self.GRANULARITY_MAP.get(timeframe, 3600)

        # Parallelize symbol fetching
        tasks = [
            self._fetch_single_symbol_ohlcv(symbol, contract_type, granularity, start_ts, end_ts)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten and filter valid results
        all_data = []
        for r in results:
            if isinstance(r, list):
                all_data.extend(r)

        self.collection_stats['records_collected'] += len(all_data)

        df = pd.DataFrame(all_data)
        if not df.empty:
            df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        return df
    
    async def _fetch_single_symbol_open_interest(
        self, symbol: str, start_ts: int, end_ts: int
    ) -> List[Dict]:
        """Fetch open interest for a single symbol."""
        if symbol.upper() not in self.PRODUCT_MAP:
            return []

        product_id = self.PRODUCT_MAP[symbol.upper()]['perp']
        logger.info(f"Fetching Vertex OI for {symbol}")

        params = {
            'product_id': product_id,
            'start_time': start_ts,
            'end_time': end_ts,
            'granularity': 3600,
            'limit': 10000
        }

        data = await self._query_archive('/open_interest', params)

        if not data or 'open_interest' not in data:
            return []

        all_data = []
        for oi in data.get('open_interest', []):
            voi = VertexOpenInterest(
                timestamp=datetime.fromtimestamp(oi['timestamp'], tz=timezone.utc),
                symbol=symbol.upper(),
                product_id=product_id,
                open_interest=float(oi.get('open_interest', 0)) / 1e18,
                open_interest_usd=float(oi.get('open_interest_usd', 0)) / 1e18,
                long_interest=float(oi.get('long_interest', 0)) / 1e18,
                short_interest=float(oi.get('short_interest', 0)) / 1e18
            )

            all_data.append({
                **voi.to_dict(),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE,
                'chain': 'arbitrum'
            })

        return all_data

    async def fetch_open_interest(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch open interest data from Vertex."""
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        # Parallelize symbol fetching
        tasks = [
            self._fetch_single_symbol_open_interest(symbol, start_ts, end_ts)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten and filter valid results
        all_data = []
        for r in results:
            if isinstance(r, list):
                all_data.extend(r)

        self.collection_stats['records_collected'] += len(all_data)

        df = pd.DataFrame(all_data)
        if not df.empty:
            df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        return df
    
    async def _fetch_single_symbol_liquidations(
        self, symbol: str, start_ts: int, end_ts: int
    ) -> List[Dict]:
        """Fetch liquidations for a single symbol."""
        if symbol.upper() not in self.PRODUCT_MAP:
            return []

        product_id = self.PRODUCT_MAP[symbol.upper()]['perp']
        logger.info(f"Fetching Vertex liquidations for {symbol}")

        params = {
            'product_id': product_id,
            'start_time': start_ts,
            'end_time': end_ts,
            'limit': 10000
        }

        data = await self._query_archive('/liquidations', params)

        if not data or 'liquidations' not in data:
            return []

        all_data = []
        for liq in data.get('liquidations', []):
            vliq = VertexLiquidation(
                timestamp=datetime.fromtimestamp(liq['timestamp'], tz=timezone.utc),
                symbol=symbol.upper(),
                product_id=product_id,
                account=liq.get('account'),
                liquidator=liq.get('liquidator'),
                size=float(liq.get('amount', 0)) / 1e18,
                price=float(liq.get('price', 0)) / 1e18,
                is_long=liq.get('is_long'),
                pnl=float(liq.get('pnl', 0)) / 1e18,
                tx_hash=liq.get('tx_hash')
            )

            all_data.append({
                **vliq.to_dict(),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE,
                'chain': 'arbitrum'
            })

        return all_data

    async def fetch_liquidations(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch liquidation events from Vertex."""
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        # Parallelize symbol fetching
        tasks = [
            self._fetch_single_symbol_liquidations(symbol, start_ts, end_ts)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten and filter valid results
        all_data = []
        for r in results:
            if isinstance(r, list):
                all_data.extend(r)

        self.collection_stats['records_collected'] += len(all_data)

        df = pd.DataFrame(all_data)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)

        return df
    
    async def fetch_market_stats(
        self, symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Fetch current market statistics."""
        data = await self._query_gateway('get_all_products', {})
        
        if not data or 'products' not in data:
            return pd.DataFrame()
        
        records = []
        
        for product in data.get('products', []):
            product_id = product.get('product_id')
            symbol = None
            
            for sym, ids in self.PRODUCT_MAP.items():
                if product_id in [ids.get('spot'), ids.get('perp')]:
                    symbol = sym
                    break
            
            if symbols and symbol not in symbols:
                continue
            
            vms = VertexMarketStats(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol or f"product_{product_id}",
                product_id=product_id,
                product_type=product.get('product_type'),
                oracle_price=safe_float(product.get('oracle_price', 0)) / 1e18,
                mark_price=safe_float(product.get('mark_price', 0)) / 1e18,
                index_price=safe_float(product.get('index_price', 0)) / 1e18,
                open_interest=safe_float(product.get('open_interest', 0)) / 1e18,
                volume_24h=safe_float(product.get('volume_24h', 0)) / 1e18,
                num_trades_24h=safe_int(product.get('num_trades_24h', 0)),
                funding_rate=safe_float(product.get('funding_rate', 0)) / 1e18
            )
            
            records.append({
                **vms.to_dict(),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE,
                'chain': 'arbitrum'
            })
        
        return pd.DataFrame(records)
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.collection_stats.copy()