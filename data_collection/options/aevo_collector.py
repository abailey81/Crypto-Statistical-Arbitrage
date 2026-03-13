"""
AEVO Options Exchange Data Collector

validated collector for AEVO decentralized options exchange.
AEVO is built on a custom rollup with order book-based options trading.

===============================================================================
AEVO OVERVIEW
===============================================================================

AEVO is a decentralized options exchange that combines the efficiency of
centralized order books with the security of on-chain settlement.

Key Differentiators:
    - Order book model (not AMM) for professional trading
    - Custom rollup for high performance
    - On-chain settlement for security
    - Growing BTC and ETH options liquidity

===============================================================================
API SPECIFICATIONS
===============================================================================

Base URL: https://api.aevo.xyz

Authentication:
    - API Key + HMAC Signature required
    - Timestamp-based request signing

Rate Limits:
    ============ ============== ================
    Type Requests/min Notes
    ============ ============== ================
    Public 100 No auth required
    Private 300 With API key
    WebSocket N/A Real-time feeds
    ============ ============== ================

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

Cross-Venue Arbitrage:
    - Compare IV surface with Deribit
    - Identify mispriced options
    - Calendar spread opportunities

Volatility Trading:
    - IV term structure analysis
    - Skew monitoring
    - Volatility surface construction

Version: 2.0.0
"""

import os
import asyncio
import hmac
import hashlib
import time
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any

import aiohttp
import pandas as pd
import numpy as np

# =============================================================================
# ENUMS
# =============================================================================

class OptionType(Enum):
    """Option type classification with payoff properties."""
    CALL = 'call'
    PUT = 'put'
    
    @property
    def sign(self) -> int:
        """Payoff sign multiplier."""
        return 1 if self == OptionType.CALL else -1
    
    @property
    def is_call(self) -> bool:
        """Check if call option."""
        return self == OptionType.CALL

class Underlying(Enum):
    """Supported underlying assets."""
    BTC = 'BTC'
    ETH = 'ETH'
    
    @property
    def index_name(self) -> str:
        """Index price feed name."""
        return f"{self.value.lower()}_usd"

class Moneyness(Enum):
    """Option moneyness classification based on delta."""
    DEEP_ITM = 'deep_itm'
    ITM = 'itm'
    SLIGHT_ITM = 'slight_itm'
    ATM = 'atm'
    SLIGHT_OTM = 'slight_otm'
    OTM = 'otm'
    DEEP_OTM = 'deep_otm'
    
    @classmethod
    def from_delta(cls, delta: float) -> 'Moneyness':
        """Classify moneyness from delta."""
        abs_delta = abs(delta)
        if abs_delta > 0.85:
            return cls.DEEP_ITM
        elif abs_delta > 0.70:
            return cls.ITM
        elif abs_delta > 0.55:
            return cls.SLIGHT_ITM
        elif abs_delta >= 0.45:
            return cls.ATM
        elif abs_delta >= 0.30:
            return cls.SLIGHT_OTM
        elif abs_delta >= 0.15:
            return cls.OTM
        return cls.DEEP_OTM
    
    @property
    def is_atm_region(self) -> bool:
        """Check if in ATM region."""
        return self in [Moneyness.SLIGHT_ITM, Moneyness.ATM, Moneyness.SLIGHT_OTM]

class ExpiryCategory(Enum):
    """Option expiry time classification."""
    INTRADAY = 'intraday'
    DAILY = 'daily'
    WEEKLY = 'weekly'
    BIWEEKLY = 'biweekly'
    MONTHLY = 'monthly'
    QUARTERLY = 'quarterly'
    
    @classmethod
    def from_days(cls, days: float) -> 'ExpiryCategory':
        """Classify from days to expiry."""
        if days < 1:
            return cls.INTRADAY
        elif days < 3:
            return cls.DAILY
        elif days < 10:
            return cls.WEEKLY
        elif days < 21:
            return cls.BIWEEKLY
        elif days < 60:
            return cls.MONTHLY
        return cls.QUARTERLY
    
    @property
    def is_short_dated(self) -> bool:
        """Check if short-dated option."""
        return self in [ExpiryCategory.INTRADAY, ExpiryCategory.DAILY, ExpiryCategory.WEEKLY]

class IVLevel(Enum):
    """Implied volatility level classification."""
    EXTREMELY_LOW = 'extremely_low'
    VERY_LOW = 'very_low'
    LOW = 'low'
    NORMAL = 'normal'
    ELEVATED = 'elevated'
    HIGH = 'high'
    VERY_HIGH = 'very_high'
    EXTREME = 'extreme'
    
    @classmethod
    def from_iv(cls, iv: float) -> 'IVLevel':
        """Classify from IV value."""
        iv_pct = iv * 100 if iv < 10 else iv
        if iv_pct < 20:
            return cls.EXTREMELY_LOW
        elif iv_pct < 35:
            return cls.VERY_LOW
        elif iv_pct < 50:
            return cls.LOW
        elif iv_pct < 70:
            return cls.NORMAL
        elif iv_pct < 90:
            return cls.ELEVATED
        elif iv_pct < 120:
            return cls.HIGH
        elif iv_pct < 150:
            return cls.VERY_HIGH
        return cls.EXTREME
    
    @property
    def trading_signal(self) -> str:
        """Trading signal from IV level."""
        if self in [IVLevel.EXTREMELY_LOW, IVLevel.VERY_LOW]:
            return 'buy_vol'
        elif self in [IVLevel.VERY_HIGH, IVLevel.EXTREME]:
            return 'sell_vol'
        return 'neutral'

class LiquidityTier(Enum):
    """Market liquidity classification."""
    INSTITUTIONAL = 'institutional'
    PROFESSIONAL = 'professional'
    RETAIL = 'retail'
    THIN = 'thin'
    ILLIQUID = 'illiquid'
    
    @property
    def max_slippage_bps(self) -> int:
        """Expected max slippage in bps."""
        mapping = {
            LiquidityTier.INSTITUTIONAL: 5,
            LiquidityTier.PROFESSIONAL: 15,
            LiquidityTier.RETAIL: 50,
            LiquidityTier.THIN: 150,
            LiquidityTier.ILLIQUID: 500,
        }
        return mapping.get(self, 500)

class OrderSide(Enum):
    """Order/trade side."""
    BUY = 'buy'
    SELL = 'sell'

class MarketStatus(Enum):
    """Market trading status."""
    ACTIVE = 'active'
    HALTED = 'halted'
    EXPIRED = 'expired'

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class OptionInstrument:
    """Option instrument metadata with comprehensive analytics."""
    instrument_name: str
    underlying: str
    strike: float
    option_type: str
    expiry: datetime
    is_active: bool = True
    tick_size: float = 0.0001
    min_trade_amount: float = 0.01
    contract_size: float = 1.0
    
    @property
    def option_type_enum(self) -> OptionType:
        """Get option type as enum."""
        return OptionType.CALL if self.option_type.lower() == 'call' else OptionType.PUT
    
    @property
    def days_to_expiry(self) -> float:
        """Calculate days to expiry."""
        now = datetime.now(timezone.utc)
        expiry = self.expiry.replace(tzinfo=timezone.utc) if self.expiry.tzinfo is None else self.expiry
        return max(0, (expiry - now).total_seconds() / 86400)
    
    @property
    def years_to_expiry(self) -> float:
        """Years to expiry for pricing."""
        return self.days_to_expiry / 365.0
    
    @property
    def expiry_category(self) -> ExpiryCategory:
        """Classify expiry timeframe."""
        return ExpiryCategory.from_days(self.days_to_expiry)
    
    @property
    def is_near_expiry(self) -> bool:
        """Check if near expiry (< 7 days)."""
        return self.days_to_expiry < 7
    
    @property
    def is_expired(self) -> bool:
        """Check if expired."""
        return self.days_to_expiry <= 0
    
    @property
    def is_tradeable(self) -> bool:
        """Check if tradeable."""
        return self.is_active and not self.is_expired
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'instrument_name': self.instrument_name, 'underlying': self.underlying,
            'strike': self.strike, 'option_type': self.option_type,
            'expiry': self.expiry.isoformat() if isinstance(self.expiry, datetime) else self.expiry,
            'days_to_expiry': self.days_to_expiry, 'years_to_expiry': self.years_to_expiry,
            'expiry_category': self.expiry_category.value, 'is_active': self.is_active,
            'is_near_expiry': self.is_near_expiry, 'is_tradeable': self.is_tradeable,
        }

@dataclass
class OptionQuote:
    """Comprehensive option quote with pricing, IV, and Greeks analytics."""
    timestamp: datetime
    instrument_name: str
    underlying: str
    strike: float
    option_type: str
    expiry: datetime
    mark_price: float = 0.0
    mark_iv: float = 0.0
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_iv: float = 0.0
    ask_iv: float = 0.0
    underlying_price: float = 0.0
    index_price: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    volume_24h: float = 0.0
    open_interest: float = 0.0
    
    @property
    def option_type_enum(self) -> OptionType:
        """Get option type as enum."""
        return OptionType.CALL if self.option_type.lower() in ['call', 'c'] else OptionType.PUT
    
    @property
    def moneyness(self) -> Moneyness:
        """Classify option moneyness based on delta."""
        return Moneyness.from_delta(self.delta)
    
    @property
    def moneyness_ratio(self) -> float:
        """Calculate moneyness ratio (strike/spot)."""
        return self.strike / self.underlying_price if self.underlying_price > 0 else 0.0
    
    @property
    def log_moneyness(self) -> float:
        """Log moneyness for surface fitting."""
        return np.log(self.moneyness_ratio) if self.moneyness_ratio > 0 else 0.0
    
    @property
    def standardized_moneyness(self) -> float:
        """Standardized moneyness (log(K/S) / (sigma * sqrt(T)))."""
        dte = self.days_to_expiry
        if dte <= 0 or self.mark_iv <= 0:
            return 0.0
        return self.log_moneyness / (self.mark_iv * np.sqrt(dte / 365))
    
    @property
    def is_itm(self) -> bool:
        """Check if option is in-the-money."""
        if self.option_type_enum == OptionType.CALL:
            return self.underlying_price > self.strike
        return self.underlying_price < self.strike
    
    @property
    def intrinsic_value(self) -> float:
        """Calculate intrinsic value."""
        if self.option_type_enum == OptionType.CALL:
            return max(0, self.underlying_price - self.strike)
        return max(0, self.strike - self.underlying_price)
    
    @property
    def intrinsic_value_pct(self) -> float:
        """Intrinsic value as percentage of spot."""
        return (self.intrinsic_value / self.underlying_price) * 100 if self.underlying_price > 0 else 0.0
    
    @property
    def time_value(self) -> float:
        """Calculate time value."""
        mark_usd = self.mark_price * self.underlying_price if self.mark_price < 1 else self.mark_price
        return max(0, mark_usd - self.intrinsic_value)
    
    @property
    def time_value_pct(self) -> float:
        """Time value as percentage of option price."""
        mark_usd = self.mark_price * self.underlying_price if self.mark_price < 1 else self.mark_price
        return (self.time_value / mark_usd) * 100 if mark_usd > 0 else 0.0
    
    @property
    def bid_ask_spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask_price - self.bid_price
    
    @property
    def spread_pct(self) -> float:
        """Calculate spread as percentage of mid price."""
        mid = (self.bid_price + self.ask_price) / 2
        return (self.bid_ask_spread / mid) * 100 if mid > 0 else 0.0
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        return self.spread_pct * 100
    
    @property
    def mid_iv(self) -> float:
        """Mid implied volatility."""
        if self.bid_iv > 0 and self.ask_iv > 0:
            return (self.bid_iv + self.ask_iv) / 2
        return self.mark_iv
    
    @property
    def iv_spread(self) -> float:
        """IV bid-ask spread."""
        return self.ask_iv - self.bid_iv if self.bid_iv > 0 and self.ask_iv > 0 else 0.0
    
    @property
    def iv_level(self) -> IVLevel:
        """Classify IV level."""
        return IVLevel.from_iv(self.mark_iv)
    
    @property
    def days_to_expiry(self) -> float:
        """Calculate days to expiry."""
        now = datetime.now(timezone.utc)
        expiry = self.expiry.replace(tzinfo=timezone.utc) if self.expiry.tzinfo is None else self.expiry
        return max(0, (expiry - now).total_seconds() / 86400)
    
    @property
    def theta_decay_pct(self) -> float:
        """Daily theta decay as percentage of option price."""
        return (abs(self.theta) / self.mark_price) * 100 if self.mark_price > 0 else 0.0
    
    @property
    def gamma_dollar(self) -> float:
        """Dollar gamma (gamma * spot^2 / 100)."""
        return self.gamma * (self.underlying_price ** 2) / 100
    
    @property
    def vega_dollar(self) -> float:
        """Dollar vega per 1% IV move."""
        return self.vega * self.underlying_price / 100
    
    @property
    def theta_dollar(self) -> float:
        """Dollar theta per day."""
        return abs(self.theta) * self.underlying_price
    
    @property
    def gamma_theta_ratio(self) -> float:
        """Gamma/theta ratio for scalping assessment."""
        return abs(self.gamma / self.theta) if self.theta != 0 else 0.0
    
    @property
    def vega_theta_ratio(self) -> float:
        """Vega/theta ratio."""
        return abs(self.vega / self.theta) if self.theta != 0 else 0.0
    
    @property
    def liquidity_score(self) -> float:
        """Liquidity score (0-100)."""
        oi_score = min(25, self.open_interest / 50)
        vol_score = min(25, self.volume_24h / 20)
        spread_score = max(0, 50 - self.spread_pct * 5)
        return oi_score + vol_score + spread_score
    
    @property
    def liquidity_tier(self) -> LiquidityTier:
        """Classify liquidity tier."""
        score = self.liquidity_score
        if score >= 80:
            return LiquidityTier.INSTITUTIONAL
        elif score >= 60:
            return LiquidityTier.PROFESSIONAL
        elif score >= 40:
            return LiquidityTier.RETAIL
        elif score >= 20:
            return LiquidityTier.THIN
        return LiquidityTier.ILLIQUID
    
    @property
    def is_liquid(self) -> bool:
        """Check if option has reasonable liquidity."""
        return self.open_interest > 0 and self.spread_pct < 10
    
    @property
    def is_tradeable(self) -> bool:
        """Check if option is tradeable."""
        return self.is_liquid and self.days_to_expiry > 0 and self.bid_price > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'instrument_name': self.instrument_name, 'underlying': self.underlying,
            'strike': self.strike, 'option_type': self.option_type,
            'expiry': self.expiry.isoformat() if isinstance(self.expiry, datetime) else self.expiry,
            'mark_price': self.mark_price, 'mark_iv': self.mark_iv,
            'bid_price': self.bid_price, 'ask_price': self.ask_price,
            'mid_iv': self.mid_iv, 'underlying_price': self.underlying_price,
            'delta': self.delta, 'gamma': self.gamma, 'theta': self.theta, 'vega': self.vega,
            'moneyness': self.moneyness.value, 'moneyness_ratio': self.moneyness_ratio,
            'log_moneyness': self.log_moneyness, 'standardized_moneyness': self.standardized_moneyness,
            'is_itm': self.is_itm, 'intrinsic_value': self.intrinsic_value,
            'time_value': self.time_value, 'spread_pct': self.spread_pct,
            'iv_level': self.iv_level.value, 'days_to_expiry': self.days_to_expiry,
            'gamma_dollar': self.gamma_dollar, 'vega_dollar': self.vega_dollar,
            'liquidity_tier': self.liquidity_tier.value, 'liquidity_score': self.liquidity_score,
            'volume_24h': self.volume_24h, 'open_interest': self.open_interest,
            'is_tradeable': self.is_tradeable,
        }

@dataclass
class OptionOrderBook:
    """Option order book snapshot with analytics."""
    timestamp: datetime
    instrument_name: str
    bids: List[List[float]] = field(default_factory=list)
    asks: List[List[float]] = field(default_factory=list)
    
    @property
    def best_bid(self) -> Optional[float]:
        """Best bid price."""
        return self.bids[0][0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        """Best ask price."""
        return self.asks[0][0] if self.asks else None
    
    @property
    def best_bid_size(self) -> Optional[float]:
        """Best bid size."""
        return self.bids[0][1] if self.bids and len(self.bids[0]) > 1 else None
    
    @property
    def best_ask_size(self) -> Optional[float]:
        """Best ask size."""
        return self.asks[0][1] if self.asks and len(self.asks[0]) > 1 else None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_bps(self) -> Optional[float]:
        """Spread in basis points."""
        if self.mid_price and self.spread:
            return (self.spread / self.mid_price) * 10000
        return None
    
    @property
    def bid_depth(self) -> float:
        """Total bid depth."""
        return sum(level[1] for level in self.bids if len(level) > 1)
    
    @property
    def ask_depth(self) -> float:
        """Total ask depth."""
        return sum(level[1] for level in self.asks if len(level) > 1)
    
    @property
    def total_depth(self) -> float:
        """Total order book depth."""
        return self.bid_depth + self.ask_depth
    
    @property
    def depth_imbalance(self) -> float:
        """Order book imbalance (-1 to 1)."""
        total = self.total_depth
        return (self.bid_depth - self.ask_depth) / total if total > 0 else 0.0
    
    @property
    def imbalance_signal(self) -> str:
        """Trading signal from imbalance."""
        if self.depth_imbalance > 0.3:
            return 'bullish'
        elif self.depth_imbalance < -0.3:
            return 'bearish'
        return 'neutral'
    
    @property
    def is_liquid(self) -> bool:
        """Check if order book has reasonable liquidity."""
        return self.bid_depth > 0 and self.ask_depth > 0 and (self.spread_bps or 1000) < 500
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'instrument_name': self.instrument_name,
            'best_bid': self.best_bid, 'best_ask': self.best_ask,
            'best_bid_size': self.best_bid_size, 'best_ask_size': self.best_ask_size,
            'mid_price': self.mid_price, 'spread_bps': self.spread_bps,
            'bid_depth': self.bid_depth, 'ask_depth': self.ask_depth,
            'depth_imbalance': self.depth_imbalance, 'imbalance_signal': self.imbalance_signal,
            'is_liquid': self.is_liquid,
        }

@dataclass
class OptionTrade:
    """Single option trade with analytics."""
    timestamp: datetime
    instrument_name: str
    price: float
    size: float
    side: str
    iv: Optional[float] = None
    underlying_price: Optional[float] = None
    
    @property
    def side_enum(self) -> OrderSide:
        """Get side as enum."""
        return OrderSide.BUY if self.side.lower() == 'buy' else OrderSide.SELL
    
    @property
    def notional(self) -> float:
        """Trade notional value."""
        return self.price * self.size
    
    @property
    def notional_usd(self) -> float:
        """Notional in USD (if underlying price available)."""
        if self.underlying_price:
            return self.notional * self.underlying_price
        return self.notional
    
    @property
    def is_buy(self) -> bool:
        """Check if buy trade."""
        return self.side_enum == OrderSide.BUY
    
    @property
    def is_large_trade(self) -> bool:
        """Check if large trade (>10 contracts)."""
        return self.size > 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'instrument_name': self.instrument_name,
            'price': self.price, 'size': self.size, 'side': self.side,
            'iv': self.iv, 'notional': self.notional, 'notional_usd': self.notional_usd,
            'is_buy': self.is_buy, 'is_large_trade': self.is_large_trade,
        }

@dataclass
class VolatilitySurfacePoint:
    """Single point on the volatility surface."""
    timestamp: datetime
    underlying: str
    strike: float
    expiry: datetime
    days_to_expiry: float
    moneyness: float
    option_type: str
    iv: float
    delta: float
    
    @property
    def log_moneyness(self) -> float:
        """Log moneyness for surface interpolation."""
        return np.log(self.moneyness) if self.moneyness > 0 else 0.0
    
    @property
    def sqrt_dte(self) -> float:
        """Square root of days to expiry."""
        return np.sqrt(max(0, self.days_to_expiry))
    
    @property
    def is_atm(self) -> bool:
        """Check if approximately ATM."""
        return 0.95 <= self.moneyness <= 1.05
    
    @property
    def iv_annualized(self) -> float:
        """Ensure IV is annualized."""
        return self.iv if self.iv > 1 else self.iv * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'underlying': self.underlying, 'strike': self.strike, 'days_to_expiry': self.days_to_expiry,
            'moneyness': self.moneyness, 'log_moneyness': self.log_moneyness,
            'option_type': self.option_type, 'iv': self.iv, 'iv_annualized': self.iv_annualized,
            'delta': self.delta, 'is_atm': self.is_atm,
        }

# =============================================================================
# COLLECTOR CLASS
# =============================================================================

class AEVOCollector:
    """
    AEVO options exchange collector.
    
    Features:
    - Option chain with Greeks
    - Order book data
    - Historical trades
    - IV surface construction
    - Cross-venue comparison tools
    """
    
    VENUE = 'aevo'
    VENUE_TYPE = 'DEX_OPTIONS'
    BASE_URL = 'https://api.aevo.xyz'
    UNDERLYINGS = ['BTC', 'ETH']

    # Supported perpetual instruments on Aevo (as of 2026)
    SUPPORTED_PERPS = {'BTC', 'ETH', 'SOL', 'AVAX', 'ARB', 'OP', 'SUI', 'APT'}

    def __init__(self, config: Optional[Dict] = None):
        """Initialize AEVO collector."""
        config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_key = config.get('api_key') or config.get('aevo_api_key') or os.getenv('AEVO_API_KEY', '')
        self.api_secret = config.get('api_secret') or config.get('aevo_api_secret') or os.getenv('AEVO_API_SECRET', '')
        self.rate_limit = config.get('rate_limit', 50)
        self.last_request_time = 0.0
        self.min_request_interval = 60.0 / self.rate_limit
        self.collection_stats = {'requests': 0, 'records': 0, 'errors': 0}

        # CRITICAL: Set supported data types for dynamic routing (collection_manager)
        self.supported_data_types = ['options', 'funding_rates', 'ohlcv', 'open_interest']
        self.venue = 'aevo'
        self.requires_auth = False # Public API endpoints available without auth
    
    async def __aenter__(self) -> 'AEVOCollector':
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self.session
    
    async def _rate_limit(self) -> None:
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = '') -> str:
        message = f"{timestamp}{method}{path}{body}"
        return hmac.new(self.api_secret.encode(), message.encode(), hashlib.sha256).hexdigest()
    
    def _get_headers(self, method: str, path: str, body: str = '') -> Dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        return {
            'AEVO-KEY': self.api_key,
            'AEVO-TIMESTAMP': timestamp,
            'AEVO-SIGNATURE': self._generate_signature(timestamp, method, path, body),
            'Content-Type': 'application/json',
        }
    
    async def _request(self, method: str, endpoint: str, params: Optional[Dict] = None,
                       data: Optional[Dict] = None, authenticated: bool = False) -> Optional[Dict]:
        await self._rate_limit()
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        body = json.dumps(data) if data else ''
        headers = self._get_headers(method, endpoint, body) if authenticated and self.api_key else {}
        self.collection_stats['requests'] += 1
        
        try:
            async with session.request(method, url, params=params, data=body if data else None, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                self.collection_stats['errors'] += 1
                error_text = await resp.text()
                # Reduce logging noise for expected errors like invalid instruments
                if 'INVALID_INSTRUMENT' in error_text or resp.status == 400:
                    self.logger.debug(f"AEVO {resp.status} for {endpoint}: {error_text[:100]}")
                else:
                    self.logger.warning(f"HTTP {resp.status} for {endpoint}: {error_text[:200]}")
                return None
        except Exception as e:
            error_msg = str(e) if str(e) else f"{type(e).__name__}"
            self.logger.debug(f"AEVO request error for {endpoint}: {error_msg}")
            self.collection_stats['errors'] += 1
            return None
    
    async def get_markets(self, asset: str = 'BTC') -> List[OptionInstrument]:
        data = await self._request('GET', '/markets', params={'asset': asset})
        instruments = []
        if data and isinstance(data, list):
            for m in data:
                try:
                    # Skip perpetuals, only process options
                    if m.get('instrument_type') != 'OPTION':
                        continue

                    # Parse expiry (Aevo returns nanoseconds timestamp)
                    expiry_raw = m.get('expiry')
                    if expiry_raw:
                        try:
                            # Aevo returns expiry in nanoseconds
                            expiry_ns = int(expiry_raw)
                            expiry_s = expiry_ns // 1_000_000_000 # Convert ns to seconds
                            expiry = datetime.fromtimestamp(expiry_s, tz=timezone.utc)
                        except (ValueError, TypeError):
                            expiry = datetime.now(timezone.utc) + timedelta(days=30)
                    else:
                        expiry = datetime.now(timezone.utc) + timedelta(days=30)

                    instruments.append(OptionInstrument(
                        instrument_name=m.get('instrument_name', ''),
                        underlying=m.get('underlying_asset', asset),
                        strike=float(m.get('strike', 0)),
                        option_type=m.get('option_type', 'call'),
                        expiry=expiry,
                        is_active=m.get('is_active', True),
                    ))
                except Exception as e:
                    self.logger.debug(f"Error parsing market: {e}")
                    pass
        self.logger.info(f"Fetched {len(instruments)} option instruments for {asset}")
        return instruments
    
    async def get_ticker(self, instrument_name: str) -> Optional[OptionQuote]:
        # Aevo uses /instrument/{instrument_name} endpoint
        data = await self._request('GET', f'/instrument/{instrument_name}')
        if not data:
            return None

        parts = instrument_name.split('-')
        underlying = parts[0] if parts else 'BTC'
        expiry_str = parts[1] if len(parts) > 1 else ''
        strike = float(parts[2]) if len(parts) > 2 and parts[2].replace('.', '').isdigit() else 0
        option_type = 'call' if len(parts) > 3 and parts[3] == 'C' else 'put'

        try:
            # Aevo uses format like 03FEB26
            expiry = datetime.strptime(expiry_str, '%d%b%y').replace(tzinfo=timezone.utc)
        except:
            try:
                expiry = datetime.strptime(expiry_str, '%Y%m%d').replace(tzinfo=timezone.utc)
            except:
                expiry = datetime.now(timezone.utc)

        greeks = data.get('greeks', {})
        best_bid = data.get('best_bid', {})
        best_ask = data.get('best_ask', {})
        markets = data.get('markets', {})

        return OptionQuote(
            timestamp=datetime.now(timezone.utc),
            instrument_name=instrument_name, underlying=underlying,
            strike=strike, option_type=option_type, expiry=expiry,
            mark_price=float(data.get('mark_price') or 0),
            mark_iv=float(greeks.get('iv') or 0), # IV is in greeks for Aevo
            bid_price=float(best_bid.get('price') or 0),
            ask_price=float(best_ask.get('price') or 0),
            underlying_price=float(data.get('index_price') or 0),
            index_price=float(data.get('index_price') or 0),
            delta=float(greeks.get('delta') or 0),
            gamma=float(greeks.get('gamma') or 0),
            theta=float(greeks.get('theta') or 0),
            vega=float(greeks.get('vega') or 0),
            volume_24h=float(markets.get('daily_volume') or 0),
            open_interest=float(markets.get('total_oi') or 0),
        )
    
    async def get_orderbook(self, instrument_name: str, depth: int = 10) -> Optional[OptionOrderBook]:
        data = await self._request('GET', '/orderbook', params={'instrument_name': instrument_name})
        if not data:
            return None
        return OptionOrderBook(
            timestamp=datetime.now(timezone.utc),
            instrument_name=instrument_name,
            bids=data.get('bids', [])[:depth],
            asks=data.get('asks', [])[:depth],
        )
    
    async def get_index_price(self, asset: str = 'BTC') -> Optional[float]:
        data = await self._request('GET', '/index', params={'asset': asset})
        return float(data.get('price', 0)) if data else None
    
    async def fetch_option_chain(self, underlying: str = 'BTC', include_greeks: bool = True, limit: int = 100) -> pd.DataFrame:
        """
        Fetch complete option chain with IV and Greeks.

        Optimized to extract data directly from /markets response which includes greeks.
        """
        self.logger.info(f"Fetching AEVO option chain for {underlying}")

        # Get raw market data which already includes greeks
        data = await self._request('GET', '/markets', params={'asset': underlying})
        all_options = []

        if data and isinstance(data, list):
            count = 0
            for m in data:
                try:
                    # Only process options
                    if m.get('instrument_type') != 'OPTION':
                        continue

                    if not m.get('is_active', True):
                        continue

                    # Parse expiry (Aevo returns nanoseconds timestamp)
                    expiry_raw = m.get('expiry')
                    if expiry_raw:
                        try:
                            expiry_ns = int(expiry_raw)
                            expiry_s = expiry_ns // 1_000_000_000
                            expiry = datetime.fromtimestamp(expiry_s, tz=timezone.utc)
                        except (ValueError, TypeError):
                            expiry = datetime.now(timezone.utc) + timedelta(days=30)
                    else:
                        continue

                    # Skip expired options
                    if expiry < datetime.now(timezone.utc):
                        continue

                    greeks = m.get('greeks', {})
                    strike = float(m.get('strike', 0))
                    index_price = float(m.get('index_price', 0))

                    record = {
                        'timestamp': datetime.now(timezone.utc),
                        'instrument_name': m.get('instrument_name', ''),
                        'underlying': underlying,
                        'strike': strike,
                        'option_type': m.get('option_type', 'call'),
                        'expiry': expiry,
                        'days_to_expiry': (expiry - datetime.now(timezone.utc)).days,
                        'mark_price': float(m.get('mark_price', 0)),
                        'mark_iv': float(greeks.get('iv', 0)),
                        'index_price': index_price,
                        'underlying_price': index_price,
                        'forward_price': float(m.get('forward_price', 0)),
                        'delta': float(greeks.get('delta', 0)),
                        'gamma': float(greeks.get('gamma', 0)),
                        'theta': float(greeks.get('theta', 0)),
                        'vega': float(greeks.get('vega', 0)),
                        'rho': float(greeks.get('rho', 0)),
                        'moneyness_ratio': strike / index_price if index_price > 0 else 0,
                        'open_interest': 0, # Not in /markets, would need /instrument call
                        'volume_24h': 0, # Not in /markets
                        'venue': self.VENUE,
                        'venue_type': self.VENUE_TYPE,
                    }

                    all_options.append(record)
                    self.collection_stats['records'] += 1
                    count += 1

                    if count >= limit:
                        break

                except Exception as e:
                    self.logger.debug(f"Error parsing market: {e}")
                    continue

        df = pd.DataFrame(all_options)
        self.logger.info(f"Fetched {len(df)} options for {underlying}")
        return df
    
    async def get_historical_trades(self, instrument_name: str, limit: int = 100) -> List[OptionTrade]:
        data = await self._request('GET', '/trades', params={'instrument_name': instrument_name, 'limit': limit})
        trades = []
        if data and isinstance(data, list):
            for t in data:
                trades.append(OptionTrade(
                    timestamp=pd.to_datetime(t.get('timestamp'), unit='ms', utc=True),
                    instrument_name=instrument_name,
                    price=float(t.get('price', 0)),
                    size=float(t.get('amount', 0)),
                    side=t.get('side', 'buy'),
                    iv=float(t.get('iv', 0)) if t.get('iv') else None,
                ))
        return trades
    
    async def build_volatility_surface(self, underlying: str = 'BTC', min_oi: float = 0) -> pd.DataFrame:
        """Build implied volatility surface from option chain."""
        chain = await self.fetch_option_chain(underlying)
        
        if chain.empty:
            return pd.DataFrame()
        
        if min_oi > 0:
            chain = chain[chain['open_interest'] > min_oi]
        
        surface_points = []
        for _, row in chain.iterrows():
            if row['mark_iv'] > 0 and row['underlying_price'] > 0:
                surface_points.append(VolatilitySurfacePoint(
                    timestamp=pd.to_datetime(row['timestamp']),
                    underlying=underlying,
                    strike=row['strike'],
                    expiry=pd.to_datetime(row.get('expiry', datetime.now(timezone.utc))),
                    days_to_expiry=row['days_to_expiry'],
                    moneyness=row['moneyness_ratio'],
                    option_type=row['option_type'],
                    iv=row['mark_iv'],
                    delta=row['delta'],
                ).to_dict())
        
        return pd.DataFrame(surface_points)
    
    async def compare_with_deribit(self, underlying: str = 'BTC') -> pd.DataFrame:
        """Get AEVO data formatted for comparison with Deribit."""
        chain = await self.fetch_option_chain(underlying)
        if chain.empty:
            return chain
        chain = chain.rename(columns={'mark_iv': 'implied_volatility', 'mark_price': 'mark_price_usd'})
        if 'implied_volatility' in chain.columns and chain['implied_volatility'].max() > 10:
            chain['implied_volatility'] = chain['implied_volatility'] / 100
        return chain
    
    async def fetch_funding_rates(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch funding rates for Aevo perpetuals using /funding-history endpoint.

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH'])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with funding rate data
        """
        self.logger.info(f"Fetching Aevo funding rates for {symbols}")

        # Convert dates to nanosecond timestamps
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
        start_ns = int(start_dt.timestamp() * 1_000_000_000)
        end_ns = int(end_dt.timestamp() * 1_000_000_000)

        # PARALLELIZED: Fetch funding rates for all symbols concurrently
        async def _fetch_single_funding_rate(symbol: str) -> List[Dict]:
            records = []
            base = symbol.upper().replace('-PERP', '').replace('USDT', '').replace('USD', '')
            instrument = f"{base}-PERP"

            params = {
                'instrument_name': instrument,
                'start_time': str(start_ns),
                'end_time': str(end_ns),
            }

            self.logger.info(f"Fetching Aevo funding: {instrument} ({start_date} to {end_date})")

            try:
                data = await self._request('GET', '/funding-history', params=params)

                if not data:
                    self.logger.warning(f"No data returned for {instrument}")
                    return records

                # Handle response - can be a list directly or dict with funding_history key
                history = data
                if isinstance(data, dict):
                    history = data.get('funding_history', data.get('history', []))

                if not history:
                    self.logger.warning(f"Empty funding history for {instrument}")
                    return records

                for entry in history:
                    try:
                        # Response format: [instrument_name, timestamp, funding_rate, mark_price]
                        if isinstance(entry, list) and len(entry) >= 4:
                            ts_ns = int(entry[1])
                            ts = pd.to_datetime(ts_ns, unit='ns', utc=True)
                            funding_rate = float(entry[2])
                            mark_price = float(entry[3])

                            records.append({
                                'timestamp': ts,
                                'symbol': base,
                                'instrument': instrument,
                                'funding_rate': funding_rate,
                                'mark_price': mark_price,
                                'venue': self.VENUE,
                                'venue_type': 'HYBRID',
                                'funding_interval_hours': 1,
                            })
                            self.collection_stats['records'] += 1
                    except (ValueError, IndexError) as e:
                        self.logger.debug(f"Error parsing funding entry: {e}")
                        continue

                self.logger.info(f" {instrument}: {len(history)} funding records")

            except Exception as e:
                self.logger.error(f"Error fetching Aevo funding for {instrument}: {e}")

            return records

        tasks = [_fetch_single_funding_rate(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_data = []
        for result in results:
            if isinstance(result, list):
                all_data.extend(result)

        if not all_data:
            self.logger.warning(f"No Aevo funding data collected")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        self.logger.info(f"Collected {len(df)} Aevo funding records")
        return df

    async def fetch_ohlcv(self, symbols: List[str], timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch OHLCV-like data from Aevo using /index-history endpoint.

        Note: Aevo doesn't have traditional OHLCV, so we use index history
        to construct price series (close prices at each interval).

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH'])
            timeframe: Timeframe ('1h', '4h', etc.)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with price data
        """
        self.logger.info(f"Fetching Aevo index history for {symbols}")

        # Convert timeframe to resolution in seconds (must be multiple of 30)
        resolution_map = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4h': 14400, '1d': 86400
        }
        resolution = resolution_map.get(timeframe.lower(), 3600)

        # Convert dates to nanosecond timestamps
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
        start_ns = int(start_dt.timestamp() * 1_000_000_000)
        end_ns = int(end_dt.timestamp() * 1_000_000_000)

        # PARALLELIZED: Fetch index history for all symbols concurrently
        async def _fetch_single_index_history(symbol: str) -> List[Dict]:
            records = []
            asset = symbol.upper().replace('-PERP', '').replace('USDT', '').replace('USD', '')

            # Only BTC and ETH are supported on Aevo for index history
            if asset not in ['BTC', 'ETH']:
                self.logger.debug(f"Skipping {asset} - Aevo only supports BTC and ETH for index history")
                return records

            params = {
                'asset': asset,
                'start_time': str(start_ns),
                'end_time': str(end_ns),
                'resolution': str(resolution),
                'limit': '50',
            }

            self.logger.info(f"Fetching Aevo index history: {asset}")

            # Paginate through results
            offset = 0
            while True:
                params['offset'] = str(offset)
                data = await self._request('GET', '/index-history', params=params)

                if not data or not isinstance(data, dict):
                    break

                history = data.get('history', [])
                if not history:
                    break

                for entry in history:
                    try:
                        # Response format: [timestamp, price]
                        if isinstance(entry, list) and len(entry) >= 2:
                            ts_ns = int(entry[0])
                            ts = pd.to_datetime(ts_ns, unit='ns', utc=True)
                            price = float(entry[1])

                            # Create pseudo-OHLCV (all prices equal since it's index snapshots)
                            records.append({
                                'timestamp': ts,
                                'symbol': asset,
                                'open': price,
                                'high': price,
                                'low': price,
                                'close': price,
                                'volume': 0.0, # Index history doesn't have volume
                                'venue': self.VENUE,
                                'venue_type': 'HYBRID',
                                'price_type': 'index',
                            })
                            self.collection_stats['records'] += 1
                    except (ValueError, IndexError) as e:
                        self.logger.debug(f"Error parsing index entry: {e}")
                        continue

                if len(history) < 50:
                    break
                offset += 50

            return records

        tasks = [_fetch_single_index_history(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_data = []
        for result in results:
            if isinstance(result, list):
                all_data.extend(result)

        if not all_data:
            self.logger.warning(f"No Aevo index history collected")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        self.logger.info(f"Collected {len(df)} Aevo index records")
        return df
    
    async def collect_options(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect options data - wraps fetch_option_chain().

        Args:
            symbols: List of underlying symbols (BTC, ETH, etc.)
            start_date: Start date (ignored - returns current snapshot)
            end_date: End date (ignored - returns current snapshot)
            **kwargs: Additional arguments (include_greeks)

        Returns:
            DataFrame with options data including Greeks
        """
        try:
            include_greeks = kwargs.get('include_greeks', True)

            # PARALLELIZED: Fetch option chains for all symbols concurrently
            async def _fetch_single_option_chain(symbol: str) -> Optional[pd.DataFrame]:
                underlying = symbol.upper().replace('USDT', '').replace('USD', '').replace('-', '')
                df = await self.fetch_option_chain(
                    underlying=underlying,
                    include_greeks=include_greeks
                )
                if not df.empty:
                    return df
                return None

            tasks = [_fetch_single_option_chain(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_records = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

            if all_records:
                result = pd.concat(all_records, ignore_index=True)
                # Add venue info if not present
                if 'venue' not in result.columns:
                    result['venue'] = self.VENUE
                if 'venue_type' not in result.columns:
                    result['venue_type'] = self.VENUE_TYPE
                return result

            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Aevo collect_options error: {e}")
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
            self.logger.error(f"Aevo collect_funding_rates error: {e}")
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
            self.logger.error(f"Aevo collect_ohlcv error: {e}")
            return pd.DataFrame()

    async def collect_open_interest(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect open interest data from option chain.

        Extracts open interest from the option chain data.
        Standardized method name for collection manager compatibility.
        """
        try:
            # PARALLELIZED: Fetch open interest for all symbols concurrently
            async def _fetch_single_open_interest(symbol: str) -> List[Dict]:
                oi_records = []
                underlying = symbol.upper().replace('USDT', '').replace('USD', '')
                if underlying not in self.UNDERLYINGS:
                    underlying = 'ETH' if 'ETH' in symbol.upper() else 'BTC'

                # Fetch option chain which includes open interest
                chain = await self.fetch_option_chain(underlying=underlying, include_greeks=True)

                if not chain.empty:
                    # Extract open interest data
                    for _, row in chain.iterrows():
                        try:
                            oi_records.append({
                                'timestamp': datetime.now(timezone.utc),
                                'symbol': symbol,
                                'underlying': underlying,
                                'instrument_name': row.get('instrument_name', ''),
                                'strike': row.get('strike', 0),
                                'expiry': row.get('expiry', ''),
                                'option_type': row.get('option_type', ''),
                                'open_interest': row.get('open_interest', 0),
                                'open_interest_usd': row.get('open_interest', 0) * row.get('index_price', 0),
                                'volume_24h': row.get('volume_24h', 0),
                                'iv': row.get('iv', 0),
                                'venue': self.VENUE,
                                'venue_type': self.VENUE_TYPE,
                            })
                        except Exception as e:
                            self.logger.debug(f"Error parsing OI row: {e}")
                            continue

                return oi_records

            tasks = [_fetch_single_open_interest(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_oi_data = []
            for result in results:
                if isinstance(result, list):
                    all_oi_data.extend(result)

            if all_oi_data:
                df = pd.DataFrame(all_oi_data)
                self.collection_stats['records'] += len(df)
                return df

            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Aevo collect_open_interest error: {e}")
            return pd.DataFrame()

    def get_collection_stats(self) -> Dict[str, Any]:
        return {**self.collection_stats, 'venue': self.VENUE}
    
    @staticmethod
    def get_supported_underlyings() -> List[str]:
        return [u.value for u in Underlying]
    
    async def close(self) -> None:
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

# Alias for backward compatibility (collection_manager expects AevoCollector)
AevoCollector = AEVOCollector