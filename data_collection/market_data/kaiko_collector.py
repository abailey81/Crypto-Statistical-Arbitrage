"""
Kaiko Collector - Professional-Quality Market Microstructure Data

validated collector for institutional market data including order book
snapshots, tick-by-tick trades, and market microstructure analytics from Kaiko.

===============================================================================
OVERVIEW
===============================================================================

Kaiko is a leading institutional cryptocurrency market data provider,
offering high-quality, normalized data across 100+ exchanges. The platform
specializes in:

    - Order book data (Level 1, 2, 3)
    - Tick-by-tick trade data
    - OHLCV aggregates
    - Market microstructure metrics
    - Cross-exchange analytics
    - Historical data archives

Target Users:
    - Quantitative trading firms
    - Asset managers
    - Research institutions
    - Risk management teams

===============================================================================
API TIERS
===============================================================================

    ============== ==================== ============== ================
    Tier Rate Limit Data Access Best For
    ============== ==================== ============== ================
    Free/Sample Limited Sample data Evaluation
    Starter 100 req/min 30 days Development
    Professional 500 req/min Full history Production
    Enterprise Custom Full + L3 HFT/MM
    ============== ==================== ============== ================

===============================================================================
DATA TYPES COLLECTED
===============================================================================

Order Book Snapshots:
    - Full depth order book (Level 2)
    - Best bid/ask (Level 1)
    - Order book aggregations
    - Historical snapshots

Trade Data:
    - Tick-by-tick trades
    - Taker side identification
    - Trade timestamps (microsecond precision)
    - Trade IDs for deduplication

OHLCV Aggregates:
    - Multiple timeframes (1m to 1w)
    - Trade count per interval
    - Volume in base and quote
    - VWAP calculations

Market Microstructure:
    - Bid-ask spreads (historical)
    - Market depth at various levels
    - Slippage estimates
    - Order imbalance metrics

Cross-Exchange Analytics:
    - Price comparison
    - Arbitrage detection
    - Liquidity comparison
    - Spread comparison

===============================================================================
SUPPORTED EXCHANGES
===============================================================================

Tier 1 (High Volume):
    - Binance, Coinbase, Kraken, Bitstamp, Gemini

Tier 2 (Major):
    - OKX, Bybit, Huobi, KuCoin, Gate.io, Bitfinex

Derivatives:
    - Deribit, BitMEX, Binance Futures, OKX Futures

Regional:
    - Bitflyer, Upbit, Bitstamp

===============================================================================
USAGE EXAMPLES
===============================================================================

Order book snapshot:

    >>> from data_collection.market_data import KaikoCollector
    >>> 
    >>> config = {'kaiko_api_key': 'your-api-key'}
    >>> collector = KaikoCollector(config)
    >>> try:
    ... ob = await collector.fetch_order_book_snapshot(
    ... symbol='btc-usd',
    ... exchange='coinbase',
    ... depth=100
    ... )
    ... print(f"Spread: {ob.spread_bps:.2f} bps")
    ... finally:
    ... await collector.close()

Cross-exchange arbitrage:

    >>> prices = await collector.fetch_cross_exchange_prices(
    ... symbol='btc-usd',
    ... exchanges=['binance', 'coinbase', 'kraken']
    ... )
    >>> if prices.has_arbitrage_opportunity:
    ... print(f"Arb: {prices.arbitrage_bps:.2f} bps")

Historical spreads:

    >>> spreads = await collector.fetch_historical_spreads(
    ... symbol='btc-usd',
    ... exchange='binance',
    ... start_date='2024-01-01',
    ... end_date='2024-01-31'
    ... )

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

Market Making:
    - Real-time spread monitoring
    - Inventory risk management
    - Quote optimization

Statistical Arbitrage:
    - Cross-exchange price discrepancies
    - Mean reversion signals
    - Cointegration analysis

Execution Analysis:
    - Slippage estimation
    - Optimal execution timing
    - Market impact modeling

Risk Management:
    - Liquidity risk assessment
    - Market depth monitoring
    - Stress testing

===============================================================================
DATA QUALITY CONSIDERATIONS
===============================================================================

- Timestamps are exchange-reported (may have clock skew)
- Order book snapshots are point-in-time (not continuous)
- Trade data includes all matched orders
- Some exchanges may have higher latency
- Historical data availability varies by exchange

Version: 2.0.0
API Documentation: https://docs.kaiko.com/
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

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================

class Exchange(Enum):
    """Supported exchanges."""
    BINANCE = 'binance'
    COINBASE = 'coinbase'
    KRAKEN = 'kraken'
    BITSTAMP = 'bitstamp'
    BITFINEX = 'bitfinex'
    OKX = 'okex'
    BYBIT = 'bybit'
    HUOBI = 'huobi'
    KUCOIN = 'kucoin'
    GEMINI = 'gemini'
    DERIBIT = 'deribit'
    BITMEX = 'bitmex'
    GATE = 'gate'

class InstrumentClass(Enum):
    """Instrument/market type."""
    SPOT = 'spot'
    FUTURE = 'future'
    PERPETUAL = 'perpetual-future'
    OPTION = 'option'

class Timeframe(Enum):
    """Supported OHLCV timeframes."""
    ONE_MIN = '1m'
    FIVE_MIN = '5m'
    FIFTEEN_MIN = '15m'
    THIRTY_MIN = '30m'
    ONE_HOUR = '1h'
    FOUR_HOUR = '4h'
    ONE_DAY = '1d'
    ONE_WEEK = '1w'

class TradeSide(Enum):
    """Trade taker side."""
    BUY = 'buy'
    SELL = 'sell'
    UNKNOWN = 'unknown'

class SpreadQuality(Enum):
    """Bid-ask spread quality classification."""
    EXCELLENT = 'excellent' # < 1 bp
    GOOD = 'good' # 1-5 bps
    FAIR = 'fair' # 5-10 bps
    POOR = 'poor' # 10-25 bps
    VERY_POOR = 'very_poor' # > 25 bps

class LiquidityLevel(Enum):
    """Market liquidity classification."""
    VERY_HIGH = 'very_high' # > $10M depth at 1%
    HIGH = 'high' # $5M-$10M
    MEDIUM = 'medium' # $1M-$5M
    LOW = 'low' # $500K-$1M
    VERY_LOW = 'very_low' # < $500K

class OrderImbalance(Enum):
    """Order book imbalance direction."""
    STRONG_BUY = 'strong_buy' # > 0.5
    BUY = 'buy' # 0.2 to 0.5
    NEUTRAL = 'neutral' # -0.2 to 0.2
    SELL = 'sell' # -0.5 to -0.2
    STRONG_SELL = 'strong_sell' # < -0.5

class ExchangeTier(Enum):
    """Exchange quality tier."""
    TIER_1 = 'tier_1' # Top liquidity
    TIER_2 = 'tier_2' # Major exchanges
    TIER_3 = 'tier_3' # Regional/smaller

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class KaikoInstrument:
    """Trading instrument metadata."""
    exchange: str
    instrument_class: str
    code: str
    base_asset: str
    quote_asset: str
    kaiko_legacy_symbol: Optional[str] = None
    trade_start_time: Optional[datetime] = None
    trade_end_time: Optional[datetime] = None
    
    @property
    def pair(self) -> str:
        """Trading pair string."""
        return f"{self.base_asset}/{self.quote_asset}".upper()
    
    @property
    def is_active(self) -> bool:
        """Check if instrument is currently active."""
        return self.trade_end_time is None
    
    @property
    def is_spot(self) -> bool:
        """Check if spot market."""
        return self.instrument_class == 'spot'
    
    @property
    def is_perpetual(self) -> bool:
        """Check if perpetual future."""
        return self.instrument_class == 'perpetual-future'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'exchange': self.exchange,
            'instrument_class': self.instrument_class,
            'code': self.code,
            'pair': self.pair,
            'base_asset': self.base_asset,
            'quote_asset': self.quote_asset,
            'kaiko_legacy_symbol': self.kaiko_legacy_symbol,
            'is_active': self.is_active,
            'is_spot': self.is_spot,
            'is_perpetual': self.is_perpetual,
        }

@dataclass
class KaikoOHLCV:
    """OHLCV candle data from Kaiko."""
    timestamp: datetime
    symbol: str
    exchange: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int = 0
    
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
    def notional_volume(self) -> float:
        """Estimated notional volume."""
        return self.volume * self.typical_price
    
    @property
    def avg_trade_size(self) -> float:
        """Average trade size."""
        return self.volume / self.trade_count if self.trade_count > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'exchange': self.exchange,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'trade_count': self.trade_count,
            'typical_price': self.typical_price,
            'range_pct': self.range_pct,
            'return_pct': self.return_pct,
            'is_bullish': self.is_bullish,
            'notional_volume': self.notional_volume,
            'venue': 'kaiko',
            'venue_type': 'market_data',
        }

@dataclass
class KaikoTrade:
    """Individual trade from Kaiko."""
    timestamp: datetime
    trade_id: str
    symbol: str
    exchange: str
    price: float
    amount: float
    side: TradeSide
    
    @property
    def notional(self) -> float:
        """Trade notional value."""
        return self.price * self.amount
    
    @property
    def is_buy(self) -> bool:
        """Check if taker was buyer."""
        return self.side == TradeSide.BUY
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'price': self.price,
            'amount': self.amount,
            'side': self.side.value,
            'notional': self.notional,
            'is_buy': self.is_buy,
            'venue': 'kaiko',
        }

@dataclass
class KaikoOrderBook:
    """Order book snapshot with computed metrics."""
    timestamp: datetime
    symbol: str
    exchange: str
    bids: List[Tuple[float, float]] # [(price, amount), ...]
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
    def spread(self) -> float:
        """Absolute spread."""
        return self.best_ask - self.best_bid
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        if self.mid_price > 0:
            return self.spread / self.mid_price * 10000
        return 0
    
    @property
    def spread_quality(self) -> SpreadQuality:
        """Classify spread quality."""
        bps = self.spread_bps
        if bps < 1:
            return SpreadQuality.EXCELLENT
        elif bps < 5:
            return SpreadQuality.GOOD
        elif bps < 10:
            return SpreadQuality.FAIR
        elif bps < 25:
            return SpreadQuality.POOR
        else:
            return SpreadQuality.VERY_POOR
    
    @property
    def total_bid_volume(self) -> float:
        """Total bid volume."""
        return sum(amt for _, amt in self.bids)
    
    @property
    def total_ask_volume(self) -> float:
        """Total ask volume."""
        return sum(amt for _, amt in self.asks)
    
    @property
    def order_imbalance(self) -> float:
        """Order book imbalance (-1 to 1, positive = more bids)."""
        total = self.total_bid_volume + self.total_ask_volume
        if total > 0:
            return (self.total_bid_volume - self.total_ask_volume) / total
        return 0
    
    @property
    def imbalance_direction(self) -> OrderImbalance:
        """Classify imbalance direction."""
        imb = self.order_imbalance
        if imb > 0.5:
            return OrderImbalance.STRONG_BUY
        elif imb > 0.2:
            return OrderImbalance.BUY
        elif imb > -0.2:
            return OrderImbalance.NEUTRAL
        elif imb > -0.5:
            return OrderImbalance.SELL
        else:
            return OrderImbalance.STRONG_SELL
    
    def depth_at_pct(self, pct: float) -> Tuple[float, float]:
        """
        Calculate bid/ask depth within X% of mid price.
        
        Args:
            pct: Percentage from mid (e.g., 1.0 for 1%)
            
        Returns:
            (bid_depth, ask_depth) in base currency
        """
        mid = self.mid_price
        if mid == 0:
            return (0, 0)
        
        threshold_bid = mid * (1 - pct / 100)
        threshold_ask = mid * (1 + pct / 100)
        
        bid_depth = sum(amt for p, amt in self.bids if p >= threshold_bid)
        ask_depth = sum(amt for p, amt in self.asks if p <= threshold_ask)
        
        return (bid_depth, ask_depth)
    
    def depth_usd_at_pct(self, pct: float) -> Tuple[float, float]:
        """Calculate depth in USD within X% of mid."""
        bid_depth, ask_depth = self.depth_at_pct(pct)
        mid = self.mid_price
        return (bid_depth * mid, ask_depth * mid)
    
    @property
    def liquidity_level(self) -> LiquidityLevel:
        """Classify liquidity level based on 1% depth."""
        bid_usd, ask_usd = self.depth_usd_at_pct(1.0)
        total_usd = bid_usd + ask_usd
        
        if total_usd > 20_000_000:
            return LiquidityLevel.VERY_HIGH
        elif total_usd > 10_000_000:
            return LiquidityLevel.HIGH
        elif total_usd > 2_000_000:
            return LiquidityLevel.MEDIUM
        elif total_usd > 1_000_000:
            return LiquidityLevel.LOW
        else:
            return LiquidityLevel.VERY_LOW
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with metrics."""
        bid_1pct, ask_1pct = self.depth_at_pct(1.0)
        bid_5pct, ask_5pct = self.depth_at_pct(5.0)
        
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'exchange': self.exchange,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'mid_price': self.mid_price,
            'spread': self.spread,
            'spread_bps': self.spread_bps,
            'spread_quality': self.spread_quality.value,
            'total_bid_volume': self.total_bid_volume,
            'total_ask_volume': self.total_ask_volume,
            'order_imbalance': self.order_imbalance,
            'imbalance_direction': self.imbalance_direction.value,
            'bid_depth_1pct': bid_1pct,
            'ask_depth_1pct': ask_1pct,
            'bid_depth_5pct': bid_5pct,
            'ask_depth_5pct': ask_5pct,
            'liquidity_level': self.liquidity_level.value,
            'venue': 'kaiko',
        }

@dataclass
class KaikoVWAP:
    """Volume Weighted Average Price data."""
    timestamp: datetime
    symbol: str
    exchange: str
    vwap: float
    volume: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'exchange': self.exchange,
            'vwap': self.vwap,
            'volume': self.volume,
            'venue': 'kaiko',
        }

@dataclass
class KaikoSpread:
    """Historical spread data."""
    timestamp: datetime
    symbol: str
    exchange: str
    bid_price_avg: float
    ask_price_avg: float
    mid_price: float
    spread_avg: float
    spread_min: float
    spread_max: float
    bid_volume_sum: float
    ask_volume_sum: float
    
    @property
    def spread_bps(self) -> float:
        """Average spread in basis points."""
        return self.spread_avg / self.mid_price * 10000 if self.mid_price > 0 else 0
    
    @property
    def spread_volatility(self) -> float:
        """Spread range as proxy for volatility."""
        return self.spread_max - self.spread_min
    
    @property
    def volume_imbalance(self) -> float:
        """Volume imbalance."""
        total = self.bid_volume_sum + self.ask_volume_sum
        if total > 0:
            return (self.bid_volume_sum - self.ask_volume_sum) / total
        return 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'exchange': self.exchange,
            'bid_price_avg': self.bid_price_avg,
            'ask_price_avg': self.ask_price_avg,
            'mid_price': self.mid_price,
            'spread_avg': self.spread_avg,
            'spread_bps': self.spread_bps,
            'spread_min': self.spread_min,
            'spread_max': self.spread_max,
            'spread_volatility': self.spread_volatility,
            'bid_volume_sum': self.bid_volume_sum,
            'ask_volume_sum': self.ask_volume_sum,
            'volume_imbalance': self.volume_imbalance,
            'venue': 'kaiko',
        }

@dataclass
class KaikoCrossExchangePrice:
    """Cross-exchange price comparison."""
    timestamp: datetime
    symbol: str
    prices: Dict[str, Dict[str, float]] # exchange -> {mid_price, spread_bps, ...}
    
    @property
    def exchanges(self) -> List[str]:
        """List of exchanges."""
        return list(self.prices.keys())
    
    @property
    def mid_prices(self) -> Dict[str, float]:
        """Mid prices by exchange."""
        return {ex: data['mid_price'] for ex, data in self.prices.items() if data.get('mid_price')}
    
    @property
    def avg_mid_price(self) -> float:
        """Average mid price across exchanges."""
        prices = list(self.mid_prices.values())
        return np.mean(prices) if prices else 0
    
    @property
    def best_bid(self) -> Tuple[str, float]:
        """Exchange with best bid."""
        best_ex, best_price = '', 0
        for ex, data in self.prices.items():
            bid = data.get('best_bid', 0)
            if bid > best_price:
                best_price = bid
                best_ex = ex
        return (best_ex, best_price)
    
    @property
    def best_ask(self) -> Tuple[str, float]:
        """Exchange with best ask."""
        best_ex, best_price = '', float('inf')
        for ex, data in self.prices.items():
            ask = data.get('best_ask', float('inf'))
            if ask < best_price and ask > 0:
                best_price = ask
                best_ex = ex
        return (best_ex, best_price)
    
    @property
    def arbitrage_bps(self) -> float:
        """Arbitrage opportunity in basis points (best bid - best ask)."""
        _, best_bid = self.best_bid
        _, best_ask = self.best_ask
        if best_ask > 0:
            return (best_bid - best_ask) / best_ask * 10000
        return 0
    
    @property
    def has_arbitrage_opportunity(self) -> bool:
        """Check if arbitrage exists (best bid > best ask)."""
        return self.arbitrage_bps > 0
    
    @property
    def price_dispersion_bps(self) -> float:
        """Price dispersion across exchanges (max-min)/avg."""
        prices = list(self.mid_prices.values())
        if len(prices) < 2:
            return 0
        avg = np.mean(prices)
        return (max(prices) - min(prices)) / avg * 10000 if avg > 0 else 0
    
    def deviation_from_avg(self, exchange: str) -> float:
        """Price deviation from average for exchange (bps)."""
        if exchange not in self.mid_prices:
            return 0
        avg = self.avg_mid_price
        return (self.mid_prices[exchange] - avg) / avg * 10000 if avg > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'exchanges': self.exchanges,
            'avg_mid_price': self.avg_mid_price,
            'best_bid_exchange': self.best_bid[0],
            'best_bid_price': self.best_bid[1],
            'best_ask_exchange': self.best_ask[0],
            'best_ask_price': self.best_ask[1],
            'arbitrage_bps': self.arbitrage_bps,
            'has_arbitrage_opportunity': self.has_arbitrage_opportunity,
            'price_dispersion_bps': self.price_dispersion_bps,
            'venue': 'kaiko',
        }

# =============================================================================
# Main Collector Class
# =============================================================================

class KaikoCollector(BaseCollector):
    """
    Kaiko market data collector for professional-quality market microstructure data.
    
    validated implementation providing high-quality market data
    including order books, trades, and cross-exchange analytics.
    
    Features:
        - Order book snapshots (Level 2)
        - Tick-by-tick trade data
        - OHLCV aggregates
        - Market microstructure metrics
        - Cross-exchange price comparison
        - Historical spread data
        - Liquidity analytics
    
    Example:
        >>> config = {'kaiko_api_key': 'your-api-key'}
        >>> collector = KaikoCollector(config)
        >>> try:
        ... ob = await collector.fetch_order_book_snapshot('btc-usd', 'coinbase')
        ... print(f"Spread: {ob.spread_bps:.2f} bps")
        ... finally:
        ... await collector.close()
    
    Attributes:
        VENUE: 'kaiko'
        VENUE_TYPE: 'market_data'
    """
    
    VENUE = 'kaiko'
    VENUE_TYPE = 'market_data'
    BASE_URL = 'https://us.market-api.kaiko.io/v2'
    REFERENCE_URL = 'https://reference-data-api.kaiko.io/v1'
    
    # Exchange tier classification
    EXCHANGE_TIERS = {
        'binance': ExchangeTier.TIER_1,
        'coinbase': ExchangeTier.TIER_1,
        'kraken': ExchangeTier.TIER_1,
        'bitstamp': ExchangeTier.TIER_1,
        'gemini': ExchangeTier.TIER_1,
        'okex': ExchangeTier.TIER_2,
        'bybit': ExchangeTier.TIER_2,
        'huobi': ExchangeTier.TIER_2,
        'kucoin': ExchangeTier.TIER_2,
        'bitfinex': ExchangeTier.TIER_2,
        'gate': ExchangeTier.TIER_2,
        'deribit': ExchangeTier.TIER_2,
        'bitmex': ExchangeTier.TIER_2,
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Kaiko collector.
        
        Args:
            config: Configuration with options:
                - kaiko_api_key: API key (required for full access)
                - rate_limit: Requests per minute (default: 60)
        """
        config = config or {}
        super().__init__(config)

        self.api_key = config.get('kaiko_api_key', config.get('api_key', ''))
        self.session: Optional[aiohttp.ClientSession] = None

        # Import here to avoid circular imports
        from ..utils.rate_limiter import get_shared_rate_limiter
        self.rate_limiter = get_shared_rate_limiter(
            'kaiko',
            rate=config.get('rate_limit', 30),
            per=60.0,
            burst=config.get('burst', 15)
        )

        self.stats = {'requests': 0, 'records': 0, 'errors': 0}

        # Track if API key is valid (not placeholder)
        self._api_key_valid = bool(self.api_key) and 'your_' not in self.api_key.lower()
        self._disabled_logged = False

        if not self._api_key_valid:
            logger.info("Kaiko: No valid API key configured - collector disabled")
        else:
            logger.info("Kaiko collector initialized with API key")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with auth headers."""
        if self.session is None or self.session.closed:
            headers = {'Accept': 'application/json', 'X-Api-Key': self.api_key}
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self.session
    
    async def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make authenticated API request."""
        await self.rate_limiter.acquire()
        session = await self._get_session()
        self.stats['requests'] += 1
        
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 429:
                    logger.debug("Kaiko rate limited, waiting 60s")
                    await asyncio.sleep(60)
                    return None
                elif resp.status == 401 or resp.status == 403:
                    # Authentication error - log only once
                    if not hasattr(self, '_auth_warning_logged') or not self._auth_warning_logged:
                        logger.warning(f"Kaiko authentication failed (HTTP {resp.status}) - check API key")
                        self._auth_warning_logged = True
                    self.stats['errors'] += 1
                    return None
                else:
                    text = await resp.text()
                    logger.debug(f"Kaiko HTTP {resp.status}: {text[:100]}")
                    self.stats['errors'] += 1
                    return None
        except Exception as e:
            logger.debug(f"Kaiko request error: {e}")
            self.stats['errors'] += 1
            return None
    
    async def fetch_available_instruments(
        self,
        exchange: str = 'binance',
        instrument_class: str = 'spot'
    ) -> pd.DataFrame:
        """
        Fetch available trading instruments on an exchange.
        
        Args:
            exchange: Exchange name
            instrument_class: 'spot', 'future', 'perpetual-future', 'option'
            
        Returns:
            DataFrame with instrument metadata
        """
        url = f"{self.REFERENCE_URL}/instruments"
        data = await self._make_request(
            url, params={'exchange': exchange, 'instrument_class': instrument_class}
        )
        
        records = []
        if data and 'data' in data:
            for inst in data['data']:
                i = KaikoInstrument(
                    exchange=inst.get('exchange', ''),
                    instrument_class=inst.get('class', ''),
                    code=inst.get('code', ''),
                    base_asset=inst.get('base_asset', ''),
                    quote_asset=inst.get('quote_asset', ''),
                    kaiko_legacy_symbol=inst.get('kaiko_legacy_symbol'),
                    trade_start_time=pd.to_datetime(inst.get('trade_start_time')) if inst.get('trade_start_time') else None,
                    trade_end_time=pd.to_datetime(inst.get('trade_end_time')) if inst.get('trade_end_time') else None
                )
                records.append(i.to_dict())
        
        self.stats['records'] += len(records)
        return pd.DataFrame(records)
    
    async def _fetch_single_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_ts: datetime,
        end_ts: datetime,
        exchange: str
    ) -> List[Dict]:
        """
        Fetch OHLCV data for a single symbol.

        Args:
            symbol: Symbol to fetch
            timeframe: Candle interval
            start_ts: Start timestamp
            end_ts: End timestamp
            exchange: Exchange name

        Returns:
            List of candle dicts or empty list on error
        """
        try:
            logger.info(f"Fetching OHLCV for {symbol} from {exchange}")

            url = f"{self.BASE_URL}/data/{exchange}.spot/{symbol}/ohlcv"
            params = {
                'start_time': start_ts.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'end_time': end_ts.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'interval': timeframe,
                'page_size': 1000
            }

            records = []
            continuation_token = None

            while True:
                if continuation_token:
                    params['continuation_token'] = continuation_token

                data = await self._make_request(url, params=params)

                if not data or 'data' not in data:
                    break

                for candle in data['data']:
                    c = KaikoOHLCV(
                        timestamp=pd.to_datetime(candle.get('timestamp'), utc=True),
                        symbol=symbol,
                        exchange=exchange,
                        open=float(candle.get('open', 0)),
                        high=float(candle.get('high', 0)),
                        low=float(candle.get('low', 0)),
                        close=float(candle.get('close', 0)),
                        volume=float(candle.get('volume', 0)),
                        trade_count=candle.get('count', 0)
                    )
                    records.append(c.to_dict())

                continuation_token = data.get('continuation_token')
                if not continuation_token:
                    break

                await asyncio.sleep(0.1)

            return records
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return []

    async def fetch_ohlcv(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        exchange: str = 'binance'
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Kaiko.

        Args:
            symbols: List of symbols (e.g., ['btc-usd', 'eth-usd'])
            timeframe: Candle interval
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            exchange: Exchange name

        Returns:
            DataFrame with OHLCV data and computed fields
        """
        # Skip if no valid API key
        if not self._api_key_valid:
            if not self._disabled_logged:
                logger.debug("Kaiko: Skipping OHLCV - no valid API key")
                self._disabled_logged = True
            return pd.DataFrame()

        start_ts = datetime.strptime(start_date, '%Y-%m-%d')
        end_ts = datetime.strptime(end_date, '%Y-%m-%d')

        # Parallelize using asyncio.gather
        tasks = [self._fetch_single_ohlcv(symbol, timeframe, start_ts, end_ts, exchange) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter results to only keep valid data
        all_records = []
        for result in results:
            if isinstance(result, list):
                all_records.extend(result)

        self.stats['records'] += len(all_records)

        df = pd.DataFrame(all_records)
        if not df.empty:
            df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        return df
    
    async def fetch_trades(
        self,
        symbol: str,
        exchange: str = 'binance',
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 10000
    ) -> pd.DataFrame:
        """
        Fetch tick-by-tick trade data.
        
        Args:
            symbol: Trading pair (e.g., 'btc-usd')
            exchange: Exchange name
            start_time: Start time (ISO format)
            end_time: End time (ISO format)
            limit: Maximum trades to fetch
            
        Returns:
            DataFrame with individual trades
        """
        url = f"{self.BASE_URL}/data/{exchange}.spot/{symbol}/trades"
        params = {'page_size': min(limit, 1000)}
        
        if start_time:
            params['start_time'] = start_time
        if end_time:
            params['end_time'] = end_time
        
        records = []
        continuation_token = None
        
        while len(records) < limit:
            if continuation_token:
                params['continuation_token'] = continuation_token
            
            data = await self._make_request(url, params=params)
            
            if not data or 'data' not in data:
                break
            
            for trade in data['data']:
                side = TradeSide.SELL if trade.get('taker_side_sell') else TradeSide.BUY
                
                t = KaikoTrade(
                    timestamp=pd.to_datetime(trade.get('timestamp'), utc=True),
                    trade_id=str(trade.get('id', '')),
                    symbol=symbol,
                    exchange=exchange,
                    price=float(trade.get('price', 0)),
                    amount=float(trade.get('amount', 0)),
                    side=side
                )
                records.append(t.to_dict())
            
            continuation_token = data.get('continuation_token')
            if not continuation_token:
                break
            
            await asyncio.sleep(0.1)
        
        self.stats['records'] += len(records)
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    async def fetch_order_book_snapshot(
        self,
        symbol: str,
        exchange: str = 'binance',
        depth: int = 100
    ) -> Optional[KaikoOrderBook]:
        """
        Fetch current order book snapshot.
        
        Args:
            symbol: Trading pair (e.g., 'btc-usd')
            exchange: Exchange name
            depth: Number of price levels
            
        Returns:
            KaikoOrderBook dataclass with metrics
        """
        url = f"{self.BASE_URL}/data/{exchange}.spot/{symbol}/ob_snapshots"
        data = await self._make_request(url, params={'depth': depth})
        
        if not data or 'data' not in data or not data['data']:
            return None
        
        snapshot = data['data'][0]
        timestamp = pd.to_datetime(snapshot.get('timestamp'), utc=True)
        
        bids = [(float(b[0]), float(b[1])) for b in snapshot.get('bids', [])]
        asks = [(float(a[0]), float(a[1])) for a in snapshot.get('asks', [])]
        
        return KaikoOrderBook(
            timestamp=timestamp,
            symbol=symbol,
            exchange=exchange,
            bids=bids,
            asks=asks
        )
    
    async def _fetch_single_exchange_orderbook(
        self,
        symbol: str,
        exchange: str
    ) -> Optional[Tuple[str, Dict]]:
        """
        Fetch order book for a single exchange.

        Args:
            symbol: Trading pair
            exchange: Exchange name

        Returns:
            Tuple of (exchange, price_dict) or None on error
        """
        try:
            ob = await self.fetch_order_book_snapshot(symbol, exchange)

            if ob:
                return (exchange, {
                    'best_bid': ob.best_bid,
                    'best_ask': ob.best_ask,
                    'mid_price': ob.mid_price,
                    'spread_bps': ob.spread_bps,
                    'order_imbalance': ob.order_imbalance,
                    'liquidity_level': ob.liquidity_level.value,
                })
            return None
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol} on {exchange}: {e}")
            return None

    async def fetch_cross_exchange_prices(
        self,
        symbol: str,
        exchanges: Optional[List[str]] = None
    ) -> Optional[KaikoCrossExchangePrice]:
        """
        Fetch current prices across multiple exchanges.

        Args:
            symbol: Trading pair (e.g., 'btc-usd')
            exchanges: List of exchanges (default: tier 1 exchanges)

        Returns:
            KaikoCrossExchangePrice with comparison metrics
        """
        if exchanges is None:
            exchanges = ['binance', 'coinbase', 'kraken', 'bitstamp', 'gemini']

        # Parallelize using asyncio.gather
        tasks = [self._fetch_single_exchange_orderbook(symbol, exchange) for exchange in exchanges]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter results to only keep valid data
        prices = {}
        for result in results:
            if isinstance(result, tuple) and result is not None:
                exchange, price_data = result
                prices[exchange] = price_data

        if not prices:
            return None

        return KaikoCrossExchangePrice(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            prices=prices
        )
    
    async def fetch_historical_spreads(
        self,
        symbol: str,
        exchange: str = 'binance',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '1h'
    ) -> pd.DataFrame:
        """
        Fetch historical bid-ask spread data.
        
        Args:
            symbol: Trading pair
            exchange: Exchange name
            start_date: Start date
            end_date: End date
            interval: Time interval
            
        Returns:
            DataFrame with historical spread data
        """
        url = f"{self.BASE_URL}/data/{exchange}.spot/{symbol}/ob_aggregations/full"
        
        params = {'interval': interval}
        if start_date:
            params['start_time'] = f"{start_date}T00:00:00Z"
        if end_date:
            params['end_time'] = f"{end_date}T23:59:59Z"
        
        records = []
        continuation_token = None
        
        while True:
            if continuation_token:
                params['continuation_token'] = continuation_token
            
            data = await self._make_request(url, params=params)
            
            if not data or 'data' not in data:
                break
            
            for point in data['data']:
                s = KaikoSpread(
                    timestamp=pd.to_datetime(point.get('timestamp'), utc=True),
                    symbol=symbol,
                    exchange=exchange,
                    bid_price_avg=float(point.get('bid_price_avg', 0)),
                    ask_price_avg=float(point.get('ask_price_avg', 0)),
                    mid_price=float(point.get('mid_price', 0)),
                    spread_avg=float(point.get('spread_avg', 0)),
                    spread_min=float(point.get('spread_min', 0)),
                    spread_max=float(point.get('spread_max', 0)),
                    bid_volume_sum=float(point.get('bid_volume_sum', 0)),
                    ask_volume_sum=float(point.get('ask_volume_sum', 0))
                )
                records.append(s.to_dict())
            
            continuation_token = data.get('continuation_token')
            if not continuation_token:
                break
            
            await asyncio.sleep(0.1)
        
        self.stats['records'] += len(records)
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    async def fetch_vwap(
        self,
        symbol: str,
        exchange: str = 'binance',
        interval: str = '1h',
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch Volume Weighted Average Price data."""
        url = f"{self.BASE_URL}/data/{exchange}.spot/{symbol}/vwap"
        
        params = {'interval': interval}
        if start_time:
            params['start_time'] = start_time
        if end_time:
            params['end_time'] = end_time
        
        data = await self._make_request(url, params=params)
        
        records = []
        if data and 'data' in data:
            for point in data['data']:
                v = KaikoVWAP(
                    timestamp=pd.to_datetime(point.get('timestamp'), utc=True),
                    symbol=symbol,
                    exchange=exchange,
                    vwap=float(point.get('price', 0)),
                    volume=float(point.get('volume', 0))
                )
                records.append(v.to_dict())
        
        self.stats['records'] += len(records)
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    async def _fetch_single_funding_rate(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> List[Dict]:
        """
        Fetch funding rates for a single symbol across all exchanges.

        Args:
            symbol: Symbol to fetch
            start_date: Start date
            end_date: End date

        Returns:
            List of funding rate records or empty list on error
        """
        try:
            logger.info(f"Fetching funding rates for {symbol}")

            records = []
            for exchange in ['binance', 'bybit', 'okex']:
                url = f"{self.BASE_URL}/data/{exchange}.perpetual-future/{symbol}/funding_rate"

                params = {
                    'start_time': f"{start_date}T00:00:00Z",
                    'end_time': f"{end_date}T23:59:59Z"
                }

                data = await self._make_request(url, params=params)

                if data and 'data' in data:
                    for point in data['data']:
                        records.append({
                            'timestamp': pd.to_datetime(point.get('timestamp'), utc=True),
                            'symbol': symbol.upper().replace('-', ''),
                            'funding_rate': float(point.get('rate', 0)),
                            'exchange': exchange,
                            'venue': self.VENUE,
                            'venue_type': self.VENUE_TYPE,
                        })

                await asyncio.sleep(0.2)

            return records
        except Exception as e:
            logger.error(f"Error fetching funding rates for {symbol}: {e}")
            return []

    async def fetch_funding_rates(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch funding rates for perpetual futures."""
        # Skip if no valid API key
        if not self._api_key_valid:
            return pd.DataFrame()

        # Parallelize using asyncio.gather
        tasks = [self._fetch_single_funding_rate(symbol, start_date, end_date) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter results to only keep valid data
        all_records = []
        for result in results:
            if isinstance(result, list):
                all_records.extend(result)

        self.stats['records'] += len(all_records)

        df = pd.DataFrame(all_records)
        if not df.empty:
            df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        return df
    
    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info(f"Kaiko session closed. Stats: {self.stats}")
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.stats.copy()
    
    @classmethod
    def get_supported_exchanges(cls) -> List[str]:
        """Get list of supported exchanges."""
        return [e.value for e in Exchange]
    
    @classmethod
    def get_tier1_exchanges(cls) -> List[str]:
        """Get tier 1 exchanges."""
        return [ex for ex, tier in cls.EXCHANGE_TIERS.items() if tier == ExchangeTier.TIER_1]

async def test_kaiko_collector():
    """Test Kaiko collector functionality."""
    config = {'rate_limit': 10}
    collector = KaikoCollector(config)
    
    try:
        print("=" * 60)
        print("Kaiko Collector Test")
        print("=" * 60)
        print(f"\nSupported exchanges: {len(KaikoCollector.get_supported_exchanges())}")
        print(f"Tier 1 exchanges: {KaikoCollector.get_tier1_exchanges()}")
        print(f"\nStats: {collector.get_collection_stats()}")
    finally:
        await collector.close()

if __name__ == '__main__':
    asyncio.run(test_kaiko_collector())