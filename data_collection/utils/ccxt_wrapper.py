"""
CCXT Unified Exchange Wrapper - Access to 100+ Cryptocurrency Exchanges

validated wrapper providing standardized access to cryptocurrency
exchange data through the CCXT library. Enables cross-venue data collection,
comparison, and arbitrage analysis with unified data formats.

===============================================================================
OVERVIEW
===============================================================================

CCXT (CryptoCurrency eXchange Trading) is an open-source library providing
unified APIs for 100+ cryptocurrency exchanges. This wrapper extends CCXT
with tested features for statistical arbitrage research.

Supported Exchange Categories:

    Major CEX (Spot + Futures):
        - Binance, Binance USDM, Binance COINM
        - OKX (OKEx)
        - Bybit
        - Kraken, Kraken Futures
        - Coinbase (detailed Trade)
        - KuCoin, KuCoin Futures

    Derivatives Specialists:
        - Deribit (Options + Futures)
        - BitMEX
        - Bitfinex

    Regional Exchanges:
        - Bitflyer (Japan)
        - Upbit (Korea)
        - Bitstamp (Europe)
        - Gate.io
        - MEXC
        - Huobi

===============================================================================
DATA TYPES COLLECTED
===============================================================================

OHLCV (Candlestick Data):
    - Open, High, Low, Close prices
    - Volume (base and quote)
    - Configurable timeframes (1m to 1M)
    - Historical data with pagination

Order Book:
    - Bid/ask levels with depth
    - Spread calculation
    - Order book imbalance
    - Liquidity analysis

Ticker Data:
    - Last price, bid, ask
    - 24h high, low, volume
    - Price change percentage
    - Real-time snapshots

Funding Rates (Perpetuals):
    - Historical funding rate data
    - Mark and index prices
    - Cross-exchange funding comparison
    - Normalized to 8h equivalent

Trade History:
    - Individual trade records
    - Taker side identification
    - Trade size and price
    - Transaction costs

Market Information:
    - Available trading pairs
    - Contract specifications
    - Margin and leverage info
    - Market status

===============================================================================
RATE LIMITING
===============================================================================

Exchange-specific rate limits with automatic backoff:

    ============== ==================== ============= ===============
    Exchange Rate Limit Burst Market Type
    ============== ==================== ============= ===============
    Binance 1200 weight/min 100 Spot + Futures
    Binance USDM 1200 weight/min 100 USDT Perpetuals
    OKX 600 requests/min 20 Unified Account
    Bybit 120 requests/min 20 Spot + Perps
    Kraken 60 requests/min 10 Spot + Futures
    Coinbase 100 requests/min 10 Spot
    KuCoin 180 requests/min 20 Spot + Futures
    Gate.io 900 requests/min 30 Spot + Perps
    Deribit 100 requests/min 10 Options + Futures
    BitMEX 60 requests/min 10 Perpetuals
    ============== ==================== ============= ===============

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

Cross-Exchange Price Comparison:
    - Identify price discrepancies across venues
    - Calculate arbitrage spreads
    - Monitor price convergence/divergence

Funding Rate Arbitrage:
    - Compare funding rates across perpetual venues
    - Identify funding arbitrage opportunities
    - Track funding rate convergence

Market Microstructure Analysis:
    - Order book depth comparison
    - Spread analysis across venues
    - Liquidity profiling

Data Validation:
    - Cross-reference OHLCV data
    - Detect exchange-specific anomalies
    - Validate price feeds

===============================================================================
USAGE EXAMPLES
===============================================================================

Basic OHLCV collection:

    >>> from data_collection.exchanges import CCXTWrapper
    >>> 
    >>> config = {'rate_limit': 10}
    >>> wrapper = CCXTWrapper(config)
    >>> 
    >>> try:
    ... ohlcv = await wrapper.fetch_ohlcv(
    ... 'binance', 'BTC/USDT', '1h',
    ... start_date='2024-01-01', end_date='2024-01-31'
    ... )
    ... print(f"Collected {len(ohlcv)} candles")
    ... finally:
    ... await wrapper.close()

Cross-exchange price comparison:

    >>> prices = await wrapper.fetch_cross_exchange_prices(
    ... 'BTC/USDT',
    ... exchanges=['binance', 'okx', 'bybit', 'kraken']
    ... )
    >>> print(f"Max price deviation: {prices['deviation_pct'].abs().max():.4f}%")

Multi-exchange funding rates:

    >>> funding = await wrapper.fetch_multi_exchange_funding(
    ... 'BTC/USDT:USDT',
    ... exchanges=['binanceusdm', 'okx', 'bybit'],
    ... start_date='2024-01-01',
    ... end_date='2024-01-31'
    ... )

===============================================================================
DATA QUALITY CONSIDERATIONS
===============================================================================

- Exchange APIs may have different data availability windows
- Symbol formats vary (BTC/USDT vs BTCUSDT vs BTC-USDT)
- Timestamp precision varies (ms vs s)
- Some exchanges require authentication for historical data
- Rate limits differ significantly between exchanges

Version: 2.0.0
Documentation: https://docs.ccxt.com/
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    import ccxt.async_support as ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    ccxt = None

from ..base_collector import BaseCollector
from .retry_handler import safe_float, safe_int

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================

class ExchangeType(Enum):
    """Classification of exchange types."""
    SPOT = 'spot'
    FUTURES = 'futures'
    SWAP = 'swap' # Perpetual
    OPTION = 'option'
    MARGIN = 'margin'

class MarketType(Enum):
    """Market/instrument type classification."""
    SPOT = 'spot'
    LINEAR = 'linear' # USDT-margined perpetual
    INVERSE = 'inverse' # Coin-margined perpetual
    FUTURE = 'future' # Dated futures
    OPTION = 'option'

class Timeframe(Enum):
    """Supported OHLCV timeframes."""
    ONE_MIN = '1m'
    FIVE_MIN = '5m'
    FIFTEEN_MIN = '15m'
    THIRTY_MIN = '30m'
    ONE_HOUR = '1h'
    TWO_HOUR = '2h'
    FOUR_HOUR = '4h'
    SIX_HOUR = '6h'
    TWELVE_HOUR = '12h'
    ONE_DAY = '1d'
    ONE_WEEK = '1w'
    ONE_MONTH = '1M'

class TradeSide(Enum):
    """Trade direction."""
    BUY = 'buy'
    SELL = 'sell'

class ExchangeTier(Enum):
    """Exchange classification by volume/reliability."""
    TIER_1 = 'tier_1' # Binance, OKX, Bybit, Coinbase
    TIER_2 = 'tier_2' # Kraken, KuCoin, Bitfinex
    TIER_3 = 'tier_3' # Regional, smaller exchanges
    SPECIALIZED = 'specialized' # Deribit, BitMEX

class LiquidityDepth(Enum):
    """Order book liquidity classification."""
    VERY_DEEP = 'very_deep' # > $10M within 1%
    DEEP = 'deep' # $5M - $10M
    MODERATE = 'moderate' # $1M - $5M
    SHALLOW = 'shallow' # $500K - $1M
    THIN = 'thin' # < $500K

class SpreadQuality(Enum):
    """Bid-ask spread quality classification."""
    EXCELLENT = 'excellent' # < 1 bp
    GOOD = 'good' # 1-5 bps
    FAIR = 'fair' # 5-10 bps
    POOR = 'poor' # 10-50 bps
    VERY_POOR = 'very_poor' # > 50 bps

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class CCXTCandle:
    """OHLCV candle data from any CCXT-supported exchange."""
    timestamp: datetime
    symbol: str
    exchange: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str = '1h'
    
    @property
    def typical_price(self) -> float:
        """Typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3
    
    @property
    def vwap_estimate(self) -> float:
        """Estimated VWAP using typical price."""
        return self.typical_price
    
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
    
    @property
    def notional_volume(self) -> float:
        """Estimated notional volume (volume Ã— typical price)."""
        return self.volume * self.typical_price
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with computed fields."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'symbol': self.symbol,
            'exchange': self.exchange,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'timeframe': self.timeframe,
            'typical_price': self.typical_price,
            'range_pct': self.range_pct,
            'return_pct': self.return_pct,
            'is_bullish': self.is_bullish,
            'notional_volume': self.notional_volume,
        }

@dataclass
class CCXTTicker:
    """Real-time ticker data from any CCXT-supported exchange."""
    timestamp: datetime
    symbol: str
    exchange: str
    bid: float
    ask: float
    last: float
    high: float
    low: float
    volume: float
    quote_volume: float
    change_pct: Optional[float] = None
    
    @property
    def mid_price(self) -> float:
        """Mid price between bid and ask."""
        return (self.bid + self.ask) / 2 if self.bid and self.ask else self.last
    
    @property
    def spread(self) -> float:
        """Absolute spread."""
        return self.ask - self.bid if self.bid and self.ask else 0
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        mid = self.mid_price
        return self.spread / mid * 10000 if mid > 0 else 0
    
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
        elif bps < 50:
            return SpreadQuality.POOR
        else:
            return SpreadQuality.VERY_POOR
    
    @property
    def range_pct(self) -> float:
        """24h range as percentage."""
        return (self.high - self.low) / self.low * 100 if self.low > 0 else 0
    
    @property
    def vwap_estimate(self) -> float:
        """Estimated VWAP from quote/base volume."""
        return self.quote_volume / self.volume if self.volume > 0 else self.last
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'symbol': self.symbol,
            'exchange': self.exchange,
            'bid': self.bid,
            'ask': self.ask,
            'last': self.last,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'quote_volume': self.quote_volume,
            'change_pct': self.change_pct,
            'mid_price': self.mid_price,
            'spread_bps': self.spread_bps,
            'spread_quality': self.spread_quality.value,
            'range_pct': self.range_pct,
        }

@dataclass
class CCXTOrderBook:
    """Order book snapshot from any CCXT-supported exchange."""
    timestamp: datetime
    symbol: str
    exchange: str
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
        """Total bid depth in USD (all levels)."""
        return sum(p * s for p, s in self.bids)
    
    @property
    def ask_depth_usd(self) -> float:
        """Total ask depth in USD (all levels)."""
        return sum(p * s for p, s in self.asks)
    
    @property
    def total_depth_usd(self) -> float:
        """Total order book depth."""
        return self.bid_depth_usd + self.ask_depth_usd
    
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
        total = self.total_depth_usd
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
    
    def depth_at_pct(self, pct: float) -> Tuple[float, float]:
        """
        Calculate bid/ask depth within X% of mid price.
        
        Args:
            pct: Percentage from mid (e.g., 1.0 for 1%)
            
        Returns:
            (bid_depth_usd, ask_depth_usd) within range
        """
        mid = self.mid_price
        if mid == 0:
            return (0, 0)
        
        threshold_bid = mid * (1 - pct / 100)
        threshold_ask = mid * (1 + pct / 100)
        
        bid_depth = sum(p * s for p, s in self.bids if p >= threshold_bid)
        ask_depth = sum(p * s for p, s in self.asks if p <= threshold_ask)
        
        return (bid_depth, ask_depth)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'symbol': self.symbol,
            'exchange': self.exchange,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'mid_price': self.mid_price,
            'spread_bps': self.spread_bps,
            'bid_depth_usd': self.bid_depth_usd,
            'ask_depth_usd': self.ask_depth_usd,
            'total_depth_usd': self.total_depth_usd,
            'imbalance': self.imbalance,
            'liquidity_depth': self.liquidity_depth.value,
            'bid_levels': len(self.bids),
            'ask_levels': len(self.asks),
        }

@dataclass
class CCXTTrade:
    """Individual trade from any CCXT-supported exchange."""
    trade_id: str
    timestamp: datetime
    symbol: str
    exchange: str
    side: TradeSide
    price: float
    amount: float
    cost: Optional[float] = None
    
    @property
    def notional(self) -> float:
        """Trade notional value."""
        return self.cost if self.cost else self.price * self.amount
    
    @property
    def is_buy(self) -> bool:
        """Check if taker was buyer."""
        return self.side == TradeSide.BUY
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'trade_id': self.trade_id,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'symbol': self.symbol,
            'exchange': self.exchange,
            'side': self.side.value,
            'price': self.price,
            'amount': self.amount,
            'notional': self.notional,
            'is_buy': self.is_buy,
        }

@dataclass
class CCXTFundingRate:
    """Funding rate from perpetual futures exchange."""
    timestamp: datetime
    symbol: str
    exchange: str
    funding_rate: float
    mark_price: Optional[float] = None
    index_price: Optional[float] = None
    funding_interval_hours: int = 8
    
    @property
    def funding_rate_annualized(self) -> float:
        """Annualized funding rate."""
        periods_per_year = 8760 / self.funding_interval_hours
        return self.funding_rate * periods_per_year
    
    @property
    def funding_rate_daily(self) -> float:
        """Daily funding rate."""
        periods_per_day = 24 / self.funding_interval_hours
        return self.funding_rate * periods_per_day
    
    @property
    def basis_bps(self) -> Optional[float]:
        """Basis in bps (mark vs index)."""
        if self.mark_price and self.index_price and self.index_price > 0:
            return (self.mark_price - self.index_price) / self.index_price * 10000
        return None
    
    @property
    def is_positive(self) -> bool:
        """Check if funding is positive (longs pay shorts)."""
        return self.funding_rate > 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'symbol': self.symbol,
            'exchange': self.exchange,
            'funding_rate': self.funding_rate,
            'funding_rate_annualized': self.funding_rate_annualized,
            'funding_rate_daily': self.funding_rate_daily,
            'mark_price': self.mark_price,
            'index_price': self.index_price,
            'basis_bps': self.basis_bps,
            'is_positive': self.is_positive,
            'funding_interval_hours': self.funding_interval_hours,
            'venue_type': 'CEX',
        }

@dataclass
class CCXTMarket:
    """Market/instrument information from any CCXT-supported exchange."""
    symbol: str
    exchange: str
    base: str
    quote: str
    market_type: MarketType
    active: bool
    contract_size: Optional[float] = None
    tick_size: Optional[float] = None
    min_amount: Optional[float] = None
    max_leverage: Optional[float] = None
    
    @property
    def is_perpetual(self) -> bool:
        """Check if market is perpetual swap."""
        return self.market_type in [MarketType.LINEAR, MarketType.INVERSE]
    
    @property
    def is_spot(self) -> bool:
        """Check if market is spot."""
        return self.market_type == MarketType.SPOT
    
    @property
    def is_derivative(self) -> bool:
        """Check if market is any derivative."""
        return self.market_type in [MarketType.LINEAR, MarketType.INVERSE, MarketType.FUTURE, MarketType.OPTION]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'base': self.base,
            'quote': self.quote,
            'market_type': self.market_type.value,
            'active': self.active,
            'contract_size': self.contract_size,
            'tick_size': self.tick_size,
            'min_amount': self.min_amount,
            'max_leverage': self.max_leverage,
            'is_perpetual': self.is_perpetual,
            'is_spot': self.is_spot,
            'is_derivative': self.is_derivative,
        }

@dataclass
class CrossExchangePrice:
    """Cross-exchange price comparison result."""
    symbol: str
    timestamp: datetime
    prices: Dict[str, float] # exchange -> price
    spreads: Dict[str, float] # exchange -> spread_bps
    
    @property
    def exchanges(self) -> List[str]:
        """List of exchanges."""
        return list(self.prices.keys())
    
    @property
    def mean_price(self) -> float:
        """Mean price across exchanges."""
        return np.mean(list(self.prices.values()))
    
    @property
    def min_price(self) -> Tuple[str, float]:
        """Exchange with minimum price."""
        ex = min(self.prices, key=self.prices.get)
        return (ex, self.prices[ex])
    
    @property
    def max_price(self) -> Tuple[str, float]:
        """Exchange with maximum price."""
        ex = max(self.prices, key=self.prices.get)
        return (ex, self.prices[ex])
    
    @property
    def price_spread_pct(self) -> float:
        """Price spread as percentage (max-min)/min."""
        min_ex, min_price = self.min_price
        max_ex, max_price = self.max_price
        return (max_price - min_price) / min_price * 100 if min_price > 0 else 0
    
    @property
    def arbitrage_opportunity_bps(self) -> float:
        """Arbitrage opportunity in basis points."""
        return self.price_spread_pct * 100
    
    @property
    def deviations(self) -> Dict[str, float]:
        """Price deviation from mean for each exchange (%)."""
        mean = self.mean_price
        return {ex: (p - mean) / mean * 100 for ex, p in self.prices.items()}
    
    @property
    def is_actionable_arb(self) -> bool:
        """Check if arbitrage is potentially actionable (> 10 bps)."""
        return self.arbitrage_opportunity_bps > 10
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'exchanges': self.exchanges,
            'mean_price': self.mean_price,
            'min_exchange': self.min_price[0],
            'min_price': self.min_price[1],
            'max_exchange': self.max_price[0],
            'max_price': self.max_price[1],
            'price_spread_pct': self.price_spread_pct,
            'arbitrage_opportunity_bps': self.arbitrage_opportunity_bps,
            'is_actionable_arb': self.is_actionable_arb,
            'deviations': self.deviations,
        }

# =============================================================================
# Main Wrapper Class
# =============================================================================

class CCXTWrapper(BaseCollector):
    """
    CCXT unified wrapper for cryptocurrency exchange data.
    
    validated implementation providing standardized access to 100+
    cryptocurrency exchanges through the CCXT library.
    
    Features:
        - Unified data format across all exchanges
        - Automatic rate limiting per exchange
        - Async support for parallel requests
        - Comprehensive error handling
        - Cross-exchange comparison utilities
        - Collection statistics tracking
    
    Supported Data Types:
        - OHLCV candlestick data
        - Real-time ticker data
        - Order book snapshots
        - Trade history
        - Funding rates (perpetuals)
        - Market information
    
    Example:
        >>> config = {'rate_limit': 10}
        >>> wrapper = CCXTWrapper(config)
        >>> try:
        ... # Single exchange
        ... ohlcv = await wrapper.fetch_ohlcv('binance', 'BTC/USDT', '1h')
        ... 
        ... # Cross-exchange comparison
        ... prices = await wrapper.fetch_cross_exchange_prices(
        ... 'BTC/USDT', ['binance', 'okx', 'bybit']
        ... )
        ... finally:
        ... await wrapper.close()
    
    Attributes:
        VENUE: 'ccxt'
        VENUE_TYPE: 'aggregator'
    """
    
    VENUE = 'ccxt'
    VENUE_TYPE = 'aggregator'
    
    # Exchange configurations with metadata
    EXCHANGE_CONFIGS: Dict[str, Dict] = {
        'binance': {
            'rate_limit': 1200,
            'rate_limit_type': 'weight',
            'has_futures': True,
            'has_funding': True,
            'funding_interval': 8,
            'default_type': 'spot',
            'tier': ExchangeTier.TIER_1,
        },
        'binanceusdm': {
            'rate_limit': 1200,
            'rate_limit_type': 'weight',
            'has_futures': True,
            'has_funding': True,
            'funding_interval': 8,
            'default_type': 'swap',
            'tier': ExchangeTier.TIER_1,
        },
        'binancecoinm': {
            'rate_limit': 1200,
            'rate_limit_type': 'weight',
            'has_futures': True,
            'has_funding': True,
            'funding_interval': 8,
            'default_type': 'swap',
            'tier': ExchangeTier.TIER_1,
        },
        'okx': {
            'rate_limit': 600,
            'rate_limit_type': 'requests',
            'has_futures': True,
            'has_funding': True,
            'funding_interval': 8,
            'default_type': 'spot',
            'tier': ExchangeTier.TIER_1,
        },
        'bybit': {
            'rate_limit': 120,
            'rate_limit_type': 'requests',
            'has_futures': True,
            'has_funding': True,
            'funding_interval': 8,
            'default_type': 'spot',
            'tier': ExchangeTier.TIER_1,
        },
        'coinbase': {
            'rate_limit': 100,
            'rate_limit_type': 'requests',
            'has_futures': False,
            'has_funding': False,
            'default_type': 'spot',
            'tier': ExchangeTier.TIER_1,
        },
        'kraken': {
            'rate_limit': 60,
            'rate_limit_type': 'requests',
            'has_futures': True,
            'has_funding': True,
            'funding_interval': 1, # Hourly
            'default_type': 'spot',
            'tier': ExchangeTier.TIER_2,
        },
        'kucoin': {
            'rate_limit': 180,
            'rate_limit_type': 'requests',
            'has_futures': True,
            'has_funding': True,
            'funding_interval': 8,
            'default_type': 'spot',
            'tier': ExchangeTier.TIER_2,
        },
        'gate': {
            'rate_limit': 900,
            'rate_limit_type': 'requests',
            'has_futures': True,
            'has_funding': True,
            'funding_interval': 8,
            'default_type': 'spot',
            'tier': ExchangeTier.TIER_2,
        },
        'huobi': {
            'rate_limit': 100,
            'rate_limit_type': 'requests',
            'has_futures': True,
            'has_funding': True,
            'funding_interval': 8,
            'default_type': 'spot',
            'tier': ExchangeTier.TIER_2,
        },
        'bitfinex': {
            'rate_limit': 90,
            'rate_limit_type': 'requests',
            'has_futures': True,
            'has_funding': True,
            'funding_interval': 8,
            'default_type': 'spot',
            'tier': ExchangeTier.TIER_2,
        },
        'deribit': {
            'rate_limit': 100,
            'rate_limit_type': 'requests',
            'has_futures': True,
            'has_funding': True,
            'has_options': True,
            'funding_interval': 8,
            'default_type': 'swap',
            'tier': ExchangeTier.SPECIALIZED,
        },
        'bitmex': {
            'rate_limit': 60,
            'rate_limit_type': 'requests',
            'has_futures': True,
            'has_funding': True,
            'funding_interval': 8,
            'default_type': 'swap',
            'tier': ExchangeTier.SPECIALIZED,
        },
        'mexc': {
            'rate_limit': 200,
            'rate_limit_type': 'requests',
            'has_futures': True,
            'has_funding': True,
            'funding_interval': 8,
            'default_type': 'spot',
            'tier': ExchangeTier.TIER_3,
        },
        'bitflyer': {
            'rate_limit': 60,
            'rate_limit_type': 'requests',
            'has_futures': False,
            'has_funding': False,
            'default_type': 'spot',
            'tier': ExchangeTier.TIER_3,
        },
        'upbit': {
            'rate_limit': 100,
            'rate_limit_type': 'requests',
            'has_futures': False,
            'has_funding': False,
            'default_type': 'spot',
            'tier': ExchangeTier.TIER_3,
        },
        'bitstamp': {
            'rate_limit': 60,
            'rate_limit_type': 'requests',
            'has_futures': False,
            'has_funding': False,
            'default_type': 'spot',
            'tier': ExchangeTier.TIER_3,
        },
    }
    
    # Timeframe mapping
    TIMEFRAME_MAP = {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '12h': '12h',
        '1d': '1d', '1w': '1w', '1M': '1M'
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize CCXT wrapper.
        
        Args:
            config: Configuration with options:
                - rate_limit: Default rate limit (requests/min)
                - api_keys: Dict of exchange -> credentials
                - timeout: Request timeout in seconds
        """
        if not CCXT_AVAILABLE:
            raise ImportError("CCXT library not available. Install with: pip install ccxt")
        
        config = config or {}
        super().__init__(config)
        
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.api_keys = config.get('api_keys', {})
        self.timeout = config.get('timeout', 30000) # ms
        
        self.collection_stats = {
            'api_calls': 0,
            'records_collected': 0,
            'errors': 0,
            'exchanges_used': set(),
        }
        
        logger.info("Initialized CCXT wrapper")
    
    async def _get_exchange(self, exchange_id: str) -> ccxt.Exchange:
        """Get or create exchange instance with configuration."""
        if exchange_id not in self.exchanges:
            exchange_class = getattr(ccxt, exchange_id, None)
            if not exchange_class:
                raise ValueError(f"Unknown exchange: {exchange_id}")
            
            config = {
                'enableRateLimit': True,
                'timeout': self.timeout,
                'options': {'adjustForTimeDifference': True}
            }
            
            # Add exchange-specific config
            if exchange_id in self.EXCHANGE_CONFIGS:
                ex_config = self.EXCHANGE_CONFIGS[exchange_id]
                config['rateLimit'] = int(1000 * 60 / ex_config['rate_limit'])
                if 'default_type' in ex_config:
                    config['options']['defaultType'] = ex_config['default_type']
            
            # Add API keys if available
            if exchange_id in self.api_keys:
                config.update(self.api_keys[exchange_id])
            
            self.exchanges[exchange_id] = exchange_class(config)
            await self.exchanges[exchange_id].load_markets()
            self.collection_stats['exchanges_used'].add(exchange_id)
            
            logger.info(f"Initialized exchange: {exchange_id}")
        
        return self.exchanges[exchange_id]
    
    async def fetch_ohlcv(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str = '1h',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from any supported exchange.
        
        Args:
            exchange_id: Exchange name (e.g., 'binance', 'okx')
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Max candles per request
            
        Returns:
            DataFrame with OHLCV data and computed fields
        """
        exchange = await self._get_exchange(exchange_id)
        
        if timeframe not in exchange.timeframes:
            logger.warning(f"{exchange_id} doesn't support {timeframe}, using 1h")
            timeframe = '1h'
        
        since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000) if start_date else None
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000) if end_date else int(datetime.utcnow().timestamp() * 1000)
        
        all_records = []
        rate_limit_retries = 0
        MAX_RATE_LIMIT_RETRIES = 3 # Max 3 rate limit retries to prevent infinite loops

        while True:
            try:
                self.collection_stats['api_calls'] += 1
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)

                if not ohlcv:
                    break

                # Reset rate limit counter on success
                rate_limit_retries = 0

                for candle in ohlcv:
                    if candle[0] > end_ts:
                        break

                    c = CCXTCandle(
                        timestamp=pd.to_datetime(candle[0], unit='ms', utc=True),
                        symbol=symbol,
                        exchange=exchange_id,
                        open=float(candle[1]),
                        high=float(candle[2]),
                        low=float(candle[3]),
                        close=float(candle[4]),
                        volume=float(candle[5]) if candle[5] else 0,
                        timeframe=timeframe
                    )
                    all_records.append(c.to_dict())

                if len(ohlcv) < limit:
                    break

                since = ohlcv[-1][0] + 1
                if since > end_ts:
                    break

            except ccxt.RateLimitExceeded:
                rate_limit_retries += 1
                if rate_limit_retries > MAX_RATE_LIMIT_RETRIES:
                    logger.warning(f"{exchange_id} rate limited {MAX_RATE_LIMIT_RETRIES} times, returning partial data")
                    break
                logger.warning(f"{exchange_id} rate limited ({rate_limit_retries}/{MAX_RATE_LIMIT_RETRIES}), waiting 30s...")
                await asyncio.sleep(30) # Reduced from 60s to 30s
            except Exception as e:
                logger.error(f"OHLCV error {exchange_id}: {e}")
                self.collection_stats['errors'] += 1
                break
        
        self.collection_stats['records_collected'] += len(all_records)
        
        df = pd.DataFrame(all_records)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    async def fetch_funding_rates(
        self,
        exchange_id: str,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch funding rate history from futures exchanges.
        
        Args:
            exchange_id: Exchange with futures (binanceusdm, okx, bybit, etc.)
            symbol: Perpetual symbol (e.g., 'BTC/USDT:USDT')
            start_date: Start date
            end_date: End date
            limit: Max records per request
            
        Returns:
            DataFrame with funding rates and computed fields
        """
        exchange = await self._get_exchange(exchange_id)
        
        if not hasattr(exchange, 'fetch_funding_rate_history'):
            logger.warning(f"{exchange_id} doesn't support funding rate history")
            return pd.DataFrame()
        
        ex_config = self.EXCHANGE_CONFIGS.get(exchange_id, {})
        funding_interval = ex_config.get('funding_interval', 8)
        
        since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000) if start_date else None
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000) if end_date else int(datetime.utcnow().timestamp() * 1000)
        
        all_records = []
        
        while True:
            try:
                self.collection_stats['api_calls'] += 1
                rates = await exchange.fetch_funding_rate_history(symbol, since=since, limit=limit)
                
                if not rates:
                    break
                
                for rate in rates:
                    if rate['timestamp'] > end_ts:
                        break
                    
                    fr = CCXTFundingRate(
                        timestamp=pd.to_datetime(rate.get('timestamp', 0), unit='ms', utc=True),
                        symbol=rate.get('symbol', symbol),
                        exchange=exchange_id,
                        funding_rate=safe_float(rate.get('fundingRate', 0)),
                        mark_price=safe_float(rate.get('markPrice')) if rate.get('markPrice') else None,
                        index_price=safe_float(rate.get('indexPrice')) if rate.get('indexPrice') else None,
                        funding_interval_hours=funding_interval
                    )
                    all_records.append(fr.to_dict())
                
                if len(rates) < limit:
                    break
                
                since = rates[-1]['timestamp'] + 1
                if since > end_ts:
                    break
                
            except Exception as e:
                logger.error(f"Funding rate error {exchange_id}: {e}")
                self.collection_stats['errors'] += 1
                break
        
        self.collection_stats['records_collected'] += len(all_records)
        
        df = pd.DataFrame(all_records)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    async def fetch_ticker(self, exchange_id: str, symbol: str) -> Optional[CCXTTicker]:
        """Fetch current ticker for a symbol."""
        exchange = await self._get_exchange(exchange_id)
        
        try:
            self.collection_stats['api_calls'] += 1
            ticker = await exchange.fetch_ticker(symbol)
            
            return CCXTTicker(
                timestamp=pd.to_datetime(ticker['timestamp'], unit='ms', utc=True) if ticker.get('timestamp') else datetime.now(timezone.utc),
                symbol=ticker['symbol'],
                exchange=exchange_id,
                bid=float(ticker['bid']) if ticker.get('bid') else 0,
                ask=float(ticker['ask']) if ticker.get('ask') else 0,
                last=float(ticker['last']) if ticker.get('last') else 0,
                high=float(ticker['high']) if ticker.get('high') else 0,
                low=float(ticker['low']) if ticker.get('low') else 0,
                volume=float(ticker['baseVolume']) if ticker.get('baseVolume') else 0,
                quote_volume=float(ticker['quoteVolume']) if ticker.get('quoteVolume') else 0,
                change_pct=float(ticker['percentage']) if ticker.get('percentage') else None
            )
        except Exception as e:
            logger.error(f"Ticker error {exchange_id}: {e}")
            self.collection_stats['errors'] += 1
            return None
    
    async def fetch_order_book(
        self, exchange_id: str, symbol: str, limit: int = 20
    ) -> Optional[CCXTOrderBook]:
        """Fetch order book snapshot."""
        exchange = await self._get_exchange(exchange_id)
        
        try:
            self.collection_stats['api_calls'] += 1
            book = await exchange.fetch_order_book(symbol, limit)
            
            return CCXTOrderBook(
                timestamp=pd.to_datetime(book['timestamp'], unit='ms', utc=True) if book.get('timestamp') else datetime.now(timezone.utc),
                symbol=symbol,
                exchange=exchange_id,
                bids=[(float(b[0]), float(b[1])) for b in book['bids'][:limit]],
                asks=[(float(a[0]), float(a[1])) for a in book['asks'][:limit]]
            )
        except Exception as e:
            logger.error(f"Order book error {exchange_id}: {e}")
            self.collection_stats['errors'] += 1
            return None
    
    async def fetch_trades(
        self, exchange_id: str, symbol: str,
        since: Optional[int] = None, limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch recent trades."""
        exchange = await self._get_exchange(exchange_id)
        
        try:
            self.collection_stats['api_calls'] += 1
            trades = await exchange.fetch_trades(symbol, since=since, limit=limit)
            
            records = []
            for trade in trades:
                t = CCXTTrade(
                    trade_id=str(trade['id']),
                    timestamp=pd.to_datetime(trade['timestamp'], unit='ms', utc=True),
                    symbol=trade['symbol'],
                    exchange=exchange_id,
                    side=TradeSide.BUY if trade['side'] == 'buy' else TradeSide.SELL,
                    price=float(trade['price']),
                    amount=float(trade['amount']),
                    cost=float(trade['cost']) if trade.get('cost') else None
                )
                records.append(t.to_dict())
            
            self.collection_stats['records_collected'] += len(records)
            return pd.DataFrame(records)
            
        except Exception as e:
            logger.error(f"Trades error {exchange_id}: {e}")
            self.collection_stats['errors'] += 1
            return pd.DataFrame()
    
    async def fetch_markets(
        self, exchange_id: str, market_type: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch available markets from exchange."""
        exchange = await self._get_exchange(exchange_id)
        
        records = []
        for symbol, market in exchange.markets.items():
            if market_type and market.get('type') != market_type:
                continue
            
            # Determine market type
            if market.get('spot'):
                mtype = MarketType.SPOT
            elif market.get('linear'):
                mtype = MarketType.LINEAR
            elif market.get('inverse'):
                mtype = MarketType.INVERSE
            elif market.get('future'):
                mtype = MarketType.FUTURE
            elif market.get('option'):
                mtype = MarketType.OPTION
            else:
                mtype = MarketType.SPOT
            
            m = CCXTMarket(
                symbol=symbol,
                exchange=exchange_id,
                base=market.get('base', ''),
                quote=market.get('quote', ''),
                market_type=mtype,
                active=market.get('active', True),
                contract_size=market.get('contractSize'),
                tick_size=market.get('precision', {}).get('price'),
                min_amount=market.get('limits', {}).get('amount', {}).get('min'),
                max_leverage=market.get('leverage', {}).get('max') if market.get('leverage') else None
            )
            records.append(m.to_dict())
        
        return pd.DataFrame(records)
    
    async def fetch_multi_exchange_ohlcv(
        self, symbol: str, exchanges: List[str],
        timeframe: str = '1h',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch OHLCV from multiple exchanges for cross-validation."""
        all_data = []
        
        for exchange_id in exchanges:
            logger.info(f"Fetching {symbol} OHLCV from {exchange_id}")
            try:
                df = await self.fetch_ohlcv(exchange_id, symbol, timeframe, start_date, end_date)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed {exchange_id}: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    async def fetch_multi_exchange_funding(
        self, symbol: str, exchanges: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch funding rates from multiple futures exchanges."""
        all_data = []
        
        for exchange_id in exchanges:
            ex_config = self.EXCHANGE_CONFIGS.get(exchange_id, {})
            if not ex_config.get('has_funding', False):
                logger.debug(f"Skipping {exchange_id}: no funding support")
                continue
            
            logger.info(f"Fetching {symbol} funding from {exchange_id}")
            try:
                df = await self.fetch_funding_rates(exchange_id, symbol, start_date, end_date)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed funding {exchange_id}: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    async def fetch_cross_exchange_prices(
        self, symbol: str, exchanges: List[str]
    ) -> CrossExchangePrice:
        """
        Fetch current prices across exchanges for arbitrage analysis.
        
        Args:
            symbol: Trading pair
            exchanges: List of exchange IDs
            
        Returns:
            CrossExchangePrice with comparison metrics
        """
        prices = {}
        spreads = {}
        
        for exchange_id in exchanges:
            ticker = await self.fetch_ticker(exchange_id, symbol)
            if ticker:
                prices[exchange_id] = ticker.last
                spreads[exchange_id] = ticker.spread_bps
        
        return CrossExchangePrice(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            prices=prices,
            spreads=spreads
        )
    
    async def close(self):
        """Close all exchange connections."""
        for exchange_id, exchange in self.exchanges.items():
            try:
                await exchange.close()
            except Exception as e:
                logger.warning(f"Error closing {exchange_id}: {e}")
        self.exchanges.clear()
        
        logger.info(f"CCXT wrapper closed. Stats: {self.get_collection_stats()}")
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        stats = self.collection_stats.copy()
        stats['exchanges_used'] = list(stats['exchanges_used'])
        return stats
    
    @classmethod
    def get_supported_exchanges(cls) -> List[str]:
        """Get list of supported exchanges with configurations."""
        return list(cls.EXCHANGE_CONFIGS.keys())
    
    @classmethod
    def get_exchanges_with_funding(cls) -> List[str]:
        """Get exchanges that support funding rates."""
        return [ex for ex, cfg in cls.EXCHANGE_CONFIGS.items() if cfg.get('has_funding')]
    
    @classmethod
    def get_tier_1_exchanges(cls) -> List[str]:
        """Get Tier 1 (highest volume) exchanges."""
        return [ex for ex, cfg in cls.EXCHANGE_CONFIGS.items() if cfg.get('tier') == ExchangeTier.TIER_1]

# =============================================================================
# Utility Functions
# =============================================================================

def get_all_ccxt_exchanges() -> List[str]:
    """Get list of all CCXT-supported exchanges."""
    if not CCXT_AVAILABLE:
        return []
    return ccxt.exchanges

def normalize_symbol(symbol: str, exchange_id: str) -> str:
    """
    Normalize symbol format for specific exchange.
    
    Different exchanges use different formats:
        - BTC/USDT (standard)
        - BTCUSDT (no separator)
        - BTC-USDT (dash separator)
        - BTC/USDT:USDT (perpetual)
    """
    # Standard CCXT format is BASE/QUOTE
    symbol = symbol.upper()
    
    # Remove common variations
    for old, new in [('BTCUSDT', 'BTC/USDT'), ('ETHUSDT', 'ETH/USDT'),
                     ('BTC-USDT', 'BTC/USDT'), ('ETH-USDT', 'ETH/USDT')]:
        symbol = symbol.replace(old, new)
    
    return symbol

async def test_ccxt_wrapper():
    """Test CCXT wrapper functionality."""
    config = {'rate_limit': 10}
    wrapper = CCXTWrapper(config)
    
    try:
        print("=" * 60)
        print("CCXT Wrapper Test")
        print("=" * 60)
        
        # List supported exchanges
        print(f"\n1. Supported exchanges: {len(CCXTWrapper.get_supported_exchanges())}")
        print(f" Tier 1: {CCXTWrapper.get_tier_1_exchanges()}")
        print(f" With funding: {CCXTWrapper.get_exchanges_with_funding()}")
        
        print(f"\nStats: {wrapper.get_collection_stats()}")
        
    finally:
        await wrapper.close()

if __name__ == '__main__':
    asyncio.run(test_ccxt_wrapper())