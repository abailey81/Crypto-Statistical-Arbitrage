"""
Lyra Finance Options Protocol Data Collector

validated collector for Lyra Finance decentralized options AMM.
Lyra pioneered AMM-based options trading with dynamic pricing.

===============================================================================
LYRA OVERVIEW
===============================================================================

Lyra Finance is a decentralized options AMM built on Optimism and Arbitrum.
It uses a novel pricing mechanism combining Black-Scholes with dynamic
volatility adjustments based on pool exposure.

Key Innovations:
    - AMM-based options (no order book needed)
    - Dynamic IV adjustment based on skew
    - Delta hedging via Synthetix/GMX
    - Liquidity provider vault model

Deployments:
    - Lyra V1: Optimism (deprecated)
    - Lyra V2 (Newport): Arbitrum
    - Lyra V2: Base (coming)

===============================================================================
DATA SOURCES
===============================================================================

API Endpoints:
    - Lyra API: https://api.lyra.finance
    - Subgraph: The Graph (Optimism/Arbitrum)

Rate Limits:
    ============ ============== ================
    Type Requests/min Notes
    ============ ============== ================
    Public API 60 No auth required
    Subgraph 100 Query complexity
    ============ ============== ================

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

Volatility Arbitrage:
    - AMM IV vs Deribit IV comparison
    - Skew impact analysis
    - Pool utilization effects on pricing

Liquidity Analysis:
    - LP vault returns vs option buying
    - Delta hedging efficiency
    - Utilization rate impact

Version: 2.0.0
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any

import aiohttp
import pandas as pd
import numpy as np

# =============================================================================
# ENUMS
# =============================================================================

class OptionType(Enum):
    """Option type classification."""
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
    ETH = 'ETH'
    BTC = 'BTC'
    OP = 'OP'
    ARB = 'ARB'
    SOL = 'SOL'
    
    @property
    def base_iv(self) -> float:
        """Typical base IV for asset."""
        mapping = {Underlying.ETH: 0.65, Underlying.BTC: 0.55, Underlying.OP: 1.0,
                   Underlying.ARB: 0.95, Underlying.SOL: 0.85}
        return mapping.get(self, 0.7)

class Chain(Enum):
    """Deployment chains."""
    OPTIMISM = 'optimism'
    ARBITRUM = 'arbitrum'
    BASE = 'base'
    
    @property
    def chain_id(self) -> int:
        """Chain ID."""
        mapping = {Chain.OPTIMISM: 10, Chain.ARBITRUM: 42161, Chain.BASE: 8453}
        return mapping.get(self, 0)

class Moneyness(Enum):
    """Option moneyness classification."""
    DEEP_ITM = 'deep_itm'
    ITM = 'itm'
    SLIGHT_ITM = 'slight_itm'
    ATM = 'atm'
    SLIGHT_OTM = 'slight_otm'
    OTM = 'otm'
    DEEP_OTM = 'deep_otm'
    
    @classmethod
    def from_delta(cls, delta: float) -> 'Moneyness':
        """Classify from delta."""
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

class IVLevel(Enum):
    """Implied volatility level classification."""
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
        if iv_pct < 30:
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

class PoolStatus(Enum):
    """Liquidity pool status."""
    ACTIVE = 'active'
    PAUSED = 'paused'
    SETTLING = 'settling'
    CLOSED = 'closed'
    
    @property
    def is_tradeable(self) -> bool:
        """Check if pool allows trading."""
        return self == PoolStatus.ACTIVE

class SkewDirection(Enum):
    """Volatility skew direction."""
    PUT_SKEW = 'put_skew'
    NEUTRAL = 'neutral'
    CALL_SKEW = 'call_skew'
    
    @classmethod
    def from_skew(cls, skew: float) -> 'SkewDirection':
        """Classify from skew value."""
        if skew < -0.02:
            return cls.PUT_SKEW
        elif skew > 0.02:
            return cls.CALL_SKEW
        return cls.NEUTRAL

class LPHealthStatus(Enum):
    """LP vault health status."""
    HEALTHY = 'healthy'
    STRESSED = 'stressed'
    CRITICAL = 'critical'
    UNDERWATER = 'underwater'

class TradeDirection(Enum):
    """Trade direction."""
    OPEN_LONG = 'open_long'
    OPEN_SHORT = 'open_short'
    CLOSE_LONG = 'close_long'
    CLOSE_SHORT = 'close_short'

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class Market:
    """Lyra market (underlying asset pool) with analytics."""
    market_address: str
    underlying: str
    chain: str
    base_iv: float = 0.0
    skew_ratio: float = 0.0
    total_liquidity: float = 0.0
    total_open_interest: float = 0.0
    utilization_rate: float = 0.0
    is_active: bool = True
    hedging_enabled: bool = True
    spot_price: float = 0.0
    
    @property
    def underlying_enum(self) -> Optional[Underlying]:
        """Get underlying as enum."""
        try:
            return Underlying(self.underlying)
        except ValueError:
            return None
    
    @property
    def chain_enum(self) -> Optional[Chain]:
        """Get chain as enum."""
        try:
            return Chain(self.chain.lower())
        except ValueError:
            return None
    
    @property
    def base_iv_pct(self) -> float:
        """Base IV as percentage."""
        return self.base_iv * 100 if self.base_iv < 10 else self.base_iv
    
    @property
    def iv_level(self) -> IVLevel:
        """Classify IV level."""
        return IVLevel.from_iv(self.base_iv)
    
    @property
    def skew_direction(self) -> SkewDirection:
        """Classify skew direction."""
        return SkewDirection.from_skew(self.skew_ratio)
    
    @property
    def utilization_pct(self) -> float:
        """Utilization as percentage."""
        return self.utilization_rate * 100
    
    @property
    def is_well_utilized(self) -> bool:
        """Check if utilization is in healthy range."""
        return 0.2 <= self.utilization_rate <= 0.7
    
    @property
    def available_liquidity(self) -> float:
        """Available liquidity for new trades."""
        return self.total_liquidity * (1 - self.utilization_rate)
    
    @property
    def tvl_tier(self) -> str:
        """TVL tier classification."""
        if self.total_liquidity > 50_000_000:
            return 'large'
        elif self.total_liquidity > 10_000_000:
            return 'medium'
        elif self.total_liquidity > 1_000_000:
            return 'small'
        return 'micro'
    
    @property
    def oi_to_liquidity_ratio(self) -> float:
        """OI to liquidity ratio."""
        return self.total_open_interest / self.total_liquidity if self.total_liquidity > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'market_address': self.market_address, 'underlying': self.underlying, 'chain': self.chain,
            'base_iv': self.base_iv, 'base_iv_pct': self.base_iv_pct, 'iv_level': self.iv_level.value,
            'skew_ratio': self.skew_ratio, 'skew_direction': self.skew_direction.value,
            'total_liquidity': self.total_liquidity, 'total_open_interest': self.total_open_interest,
            'utilization_pct': self.utilization_pct, 'available_liquidity': self.available_liquidity,
            'tvl_tier': self.tvl_tier, 'spot_price': self.spot_price, 'is_active': self.is_active,
        }

@dataclass
class Board:
    """Option board (expiry) with strike data."""
    board_id: str
    market_address: str
    underlying: str
    expiry: datetime
    base_iv: float = 0.0
    iv_variance: float = 0.0
    total_long_calls: float = 0.0
    total_short_calls: float = 0.0
    total_long_puts: float = 0.0
    total_short_puts: float = 0.0
    is_frozen: bool = False
    spot_price: float = 0.0
    
    @property
    def days_to_expiry(self) -> float:
        """Days to expiry."""
        now = datetime.now(timezone.utc)
        expiry = self.expiry.replace(tzinfo=timezone.utc) if self.expiry.tzinfo is None else self.expiry
        return max(0, (expiry - now).total_seconds() / 86400)
    
    @property
    def years_to_expiry(self) -> float:
        """Years to expiry."""
        return self.days_to_expiry / 365.0
    
    @property
    def is_expired(self) -> bool:
        """Check if expired."""
        return self.days_to_expiry <= 0
    
    @property
    def is_near_expiry(self) -> bool:
        """Check if near expiry (<3 days)."""
        return self.days_to_expiry < 3
    
    @property
    def expiry_category(self) -> str:
        """Expiry category."""
        dte = self.days_to_expiry
        if dte < 3:
            return 'near_term'
        elif dte < 10:
            return 'weekly'
        elif dte < 35:
            return 'monthly'
        return 'quarterly'
    
    @property
    def net_call_exposure(self) -> float:
        """Net call exposure (long - short)."""
        return self.total_long_calls - self.total_short_calls
    
    @property
    def net_put_exposure(self) -> float:
        """Net put exposure (long - short)."""
        return self.total_long_puts - self.total_short_puts
    
    @property
    def total_oi(self) -> float:
        """Total open interest."""
        return self.total_long_calls + self.total_short_calls + self.total_long_puts + self.total_short_puts
    
    @property
    def put_call_ratio(self) -> float:
        """Put/call ratio based on OI."""
        total_calls = self.total_long_calls + self.total_short_calls
        total_puts = self.total_long_puts + self.total_short_puts
        return total_puts / total_calls if total_calls > 0 else 0.0
    
    @property
    def sentiment_signal(self) -> str:
        """Sentiment from put/call ratio."""
        pcr = self.put_call_ratio
        if pcr > 1.3:
            return 'bearish'
        elif pcr < 0.7:
            return 'bullish'
        return 'neutral'
    
    @property
    def base_iv_pct(self) -> float:
        """Base IV as percentage."""
        return self.base_iv * 100 if self.base_iv < 10 else self.base_iv
    
    @property
    def iv_level(self) -> IVLevel:
        """Classify IV level."""
        return IVLevel.from_iv(self.base_iv)
    
    @property
    def is_tradeable(self) -> bool:
        """Check if board is tradeable."""
        return not self.is_frozen and not self.is_expired
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'board_id': self.board_id, 'market_address': self.market_address, 'underlying': self.underlying,
            'expiry': self.expiry.isoformat() if isinstance(self.expiry, datetime) else self.expiry,
            'days_to_expiry': self.days_to_expiry, 'expiry_category': self.expiry_category,
            'base_iv': self.base_iv, 'base_iv_pct': self.base_iv_pct, 'iv_level': self.iv_level.value,
            'net_call_exposure': self.net_call_exposure, 'net_put_exposure': self.net_put_exposure,
            'total_oi': self.total_oi, 'put_call_ratio': self.put_call_ratio,
            'sentiment_signal': self.sentiment_signal, 'is_tradeable': self.is_tradeable, 'spot_price': self.spot_price,
        }

@dataclass
class Strike:
    """Strike-level option data with comprehensive analytics."""
    strike_id: str
    board_id: str
    market_address: str
    underlying: str
    strike_price: float
    expiry: datetime
    option_type: str
    iv: float = 0.0
    skew: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    long_oi: float = 0.0
    short_oi: float = 0.0
    spot_price: float = 0.0
    bid_price: float = 0.0
    ask_price: float = 0.0
    
    @property
    def option_type_enum(self) -> OptionType:
        """Get option type as enum."""
        return OptionType.CALL if self.option_type.lower() in ['call', 'c'] else OptionType.PUT
    
    @property
    def days_to_expiry(self) -> float:
        """Days to expiry."""
        now = datetime.now(timezone.utc)
        expiry = self.expiry.replace(tzinfo=timezone.utc) if self.expiry.tzinfo is None else self.expiry
        return max(0, (expiry - now).total_seconds() / 86400)
    
    @property
    def years_to_expiry(self) -> float:
        """Years to expiry."""
        return self.days_to_expiry / 365.0
    
    @property
    def moneyness_ratio(self) -> float:
        """Moneyness ratio (strike/spot)."""
        return self.strike_price / self.spot_price if self.spot_price > 0 else 0.0
    
    @property
    def log_moneyness(self) -> float:
        """Log moneyness for surface fitting."""
        return np.log(self.moneyness_ratio) if self.moneyness_ratio > 0 else 0.0
    
    @property
    def standardized_moneyness(self) -> float:
        """Standardized moneyness."""
        dte = self.days_to_expiry
        if dte <= 0 or self.iv <= 0:
            return 0.0
        return self.log_moneyness / (self.iv * np.sqrt(dte / 365))
    
    @property
    def moneyness(self) -> Moneyness:
        """Classify moneyness from delta."""
        return Moneyness.from_delta(self.delta)
    
    @property
    def is_itm(self) -> bool:
        """Check if in-the-money."""
        if self.option_type_enum == OptionType.CALL:
            return self.spot_price > self.strike_price
        return self.spot_price < self.strike_price
    
    @property
    def is_atm(self) -> bool:
        """Check if approximately ATM."""
        return 0.95 <= self.moneyness_ratio <= 1.05
    
    @property
    def intrinsic_value(self) -> float:
        """Calculate intrinsic value."""
        if self.option_type_enum == OptionType.CALL:
            return max(0, self.spot_price - self.strike_price)
        return max(0, self.strike_price - self.spot_price)
    
    @property
    def intrinsic_value_pct(self) -> float:
        """Intrinsic value as percentage of spot."""
        return (self.intrinsic_value / self.spot_price) * 100 if self.spot_price > 0 else 0.0
    
    @property
    def mid_price(self) -> float:
        """Mid price."""
        if self.bid_price > 0 and self.ask_price > 0:
            return (self.bid_price + self.ask_price) / 2
        return max(self.bid_price, self.ask_price)
    
    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask_price - self.bid_price
    
    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid."""
        return (self.spread / self.mid_price) * 100 if self.mid_price > 0 else 0.0
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        return self.spread_pct * 100
    
    @property
    def iv_pct(self) -> float:
        """IV as percentage."""
        return self.iv * 100 if self.iv < 10 else self.iv
    
    @property
    def iv_level(self) -> IVLevel:
        """Classify IV level."""
        return IVLevel.from_iv(self.iv)
    
    @property
    def total_oi(self) -> float:
        """Total open interest."""
        return self.long_oi + self.short_oi
    
    @property
    def net_oi(self) -> float:
        """Net open interest (long - short)."""
        return self.long_oi - self.short_oi
    
    @property
    def oi_imbalance(self) -> float:
        """OI imbalance ratio (-1 to 1)."""
        total = self.total_oi
        return self.net_oi / total if total > 0 else 0.0
    
    @property
    def gamma_dollar(self) -> float:
        """Dollar gamma."""
        return self.gamma * (self.spot_price ** 2) / 100
    
    @property
    def vega_dollar(self) -> float:
        """Dollar vega per 1% IV move."""
        return self.vega * self.spot_price / 100
    
    @property
    def theta_dollar(self) -> float:
        """Dollar theta per day."""
        return abs(self.theta) * self.spot_price
    
    @property
    def theta_decay_pct(self) -> float:
        """Daily theta as percentage of option price."""
        return (abs(self.theta) / self.mid_price) * 100 if self.mid_price > 0 else 0.0
    
    @property
    def gamma_theta_ratio(self) -> float:
        """Gamma/theta ratio for scalping."""
        return abs(self.gamma / self.theta) if self.theta != 0 else 0.0
    
    @property
    def is_liquid(self) -> bool:
        """Check if strike is liquid."""
        return self.total_oi > 0 and self.spread_pct < 10
    
    @property
    def skew_adjusted_iv(self) -> float:
        """Skew-adjusted IV."""
        return self.iv + self.skew
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strike_id': self.strike_id, 'board_id': self.board_id, 'market_address': self.market_address,
            'underlying': self.underlying, 'strike_price': self.strike_price, 'option_type': self.option_type,
            'expiry': self.expiry.isoformat() if isinstance(self.expiry, datetime) else self.expiry,
            'days_to_expiry': self.days_to_expiry, 'iv': self.iv, 'iv_pct': self.iv_pct,
            'iv_level': self.iv_level.value, 'skew': self.skew, 'skew_adjusted_iv': self.skew_adjusted_iv,
            'delta': self.delta, 'gamma': self.gamma, 'theta': self.theta, 'vega': self.vega,
            'moneyness': self.moneyness.value, 'moneyness_ratio': self.moneyness_ratio,
            'log_moneyness': self.log_moneyness, 'is_itm': self.is_itm, 'intrinsic_value': self.intrinsic_value,
            'bid_price': self.bid_price, 'ask_price': self.ask_price, 'mid_price': self.mid_price,
            'spread_pct': self.spread_pct, 'total_oi': self.total_oi, 'net_oi': self.net_oi,
            'gamma_dollar': self.gamma_dollar, 'vega_dollar': self.vega_dollar,
            'theta_dollar': self.theta_dollar, 'spot_price': self.spot_price, 'is_liquid': self.is_liquid,
        }

@dataclass
class Trade:
    """Single option trade from Lyra."""
    tx_hash: str
    timestamp: datetime
    market_address: str
    underlying: str
    strike_price: float
    option_type: str
    expiry: datetime
    size: float
    premium: float
    direction: str
    iv_at_trade: float = 0.0
    spot_at_trade: float = 0.0
    trader: str = ''
    
    @property
    def direction_enum(self) -> TradeDirection:
        """Get direction as enum."""
        mapping = {'open_long': TradeDirection.OPEN_LONG, 'open_short': TradeDirection.OPEN_SHORT,
                   'close_long': TradeDirection.CLOSE_LONG, 'close_short': TradeDirection.CLOSE_SHORT}
        return mapping.get(self.direction.lower(), TradeDirection.OPEN_LONG)
    
    @property
    def is_opening(self) -> bool:
        """Check if opening trade."""
        return self.direction_enum in [TradeDirection.OPEN_LONG, TradeDirection.OPEN_SHORT]
    
    @property
    def is_long(self) -> bool:
        """Check if long direction."""
        return self.direction_enum in [TradeDirection.OPEN_LONG, TradeDirection.CLOSE_LONG]
    
    @property
    def premium_per_contract(self) -> float:
        """Premium per contract."""
        return self.premium / self.size if self.size > 0 else 0.0
    
    @property
    def notional_value(self) -> float:
        """Notional value."""
        return self.size * self.spot_at_trade if self.spot_at_trade > 0 else self.size * self.strike_price
    
    @property
    def premium_pct(self) -> float:
        """Premium as percentage of notional."""
        return (self.premium / self.notional_value) * 100 if self.notional_value > 0 else 0.0
    
    @property
    def is_large_trade(self) -> bool:
        """Check if large trade (>$50k notional)."""
        return self.notional_value > 50000
    
    @property
    def iv_at_trade_pct(self) -> float:
        """IV at trade as percentage."""
        return self.iv_at_trade * 100 if self.iv_at_trade < 10 else self.iv_at_trade
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'tx_hash': self.tx_hash,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'market_address': self.market_address, 'underlying': self.underlying,
            'strike_price': self.strike_price, 'option_type': self.option_type,
            'expiry': self.expiry.isoformat() if isinstance(self.expiry, datetime) else self.expiry,
            'size': self.size, 'premium': self.premium, 'direction': self.direction,
            'is_opening': self.is_opening, 'is_long': self.is_long,
            'premium_per_contract': self.premium_per_contract, 'notional_value': self.notional_value,
            'premium_pct': self.premium_pct, 'iv_at_trade_pct': self.iv_at_trade_pct,
            'spot_at_trade': self.spot_at_trade, 'is_large_trade': self.is_large_trade,
        }

@dataclass
class LPVault:
    """Liquidity Provider vault data."""
    vault_address: str
    market_address: str
    underlying: str
    chain: str
    total_deposits: float = 0.0
    total_pending_deposits: float = 0.0
    total_pending_withdrawals: float = 0.0
    net_delta: float = 0.0
    net_std_vega: float = 0.0
    utilization: float = 0.0
    min_collateral: float = 0.0
    current_collateral: float = 0.0
    is_active: bool = True
    
    @property
    def available_liquidity(self) -> float:
        """Available liquidity."""
        return self.total_deposits * (1 - self.utilization)
    
    @property
    def utilization_pct(self) -> float:
        """Utilization as percentage."""
        return self.utilization * 100
    
    @property
    def collateral_ratio(self) -> float:
        """Collateral ratio (current/min)."""
        return self.current_collateral / self.min_collateral if self.min_collateral > 0 else 0.0
    
    @property
    def health_status(self) -> LPHealthStatus:
        """LP vault health status."""
        ratio = self.collateral_ratio
        if ratio > 1.5:
            return LPHealthStatus.HEALTHY
        elif ratio > 1.2:
            return LPHealthStatus.STRESSED
        elif ratio > 1.0:
            return LPHealthStatus.CRITICAL
        return LPHealthStatus.UNDERWATER
    
    @property
    def is_healthy(self) -> bool:
        """Check if vault is healthy."""
        return self.health_status == LPHealthStatus.HEALTHY
    
    @property
    def net_exposure(self) -> str:
        """Net exposure direction."""
        if self.net_delta > 0.1:
            return 'long'
        elif self.net_delta < -0.1:
            return 'short'
        return 'neutral'
    
    @property
    def pending_net_flow(self) -> float:
        """Net pending flow (deposits - withdrawals)."""
        return self.total_pending_deposits - self.total_pending_withdrawals
    
    @property
    def pending_flow_pct(self) -> float:
        """Pending flow as percentage of deposits."""
        return (self.pending_net_flow / self.total_deposits) * 100 if self.total_deposits > 0 else 0.0
    
    @property
    def tvl_tier(self) -> str:
        """TVL tier classification."""
        if self.total_deposits > 50_000_000:
            return 'large'
        elif self.total_deposits > 10_000_000:
            return 'medium'
        elif self.total_deposits > 1_000_000:
            return 'small'
        return 'micro'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'vault_address': self.vault_address, 'market_address': self.market_address,
            'underlying': self.underlying, 'chain': self.chain,
            'total_deposits': self.total_deposits, 'available_liquidity': self.available_liquidity,
            'utilization_pct': self.utilization_pct, 'net_delta': self.net_delta,
            'net_std_vega': self.net_std_vega, 'collateral_ratio': self.collateral_ratio,
            'health_status': self.health_status.value, 'is_healthy': self.is_healthy,
            'net_exposure': self.net_exposure, 'pending_net_flow': self.pending_net_flow,
            'pending_flow_pct': self.pending_flow_pct, 'tvl_tier': self.tvl_tier, 'is_active': self.is_active,
        }

# =============================================================================
# COLLECTOR CLASS
# =============================================================================

class LyraCollector:
    """
    Lyra Finance options AMM data collector.
    
    Features:
    - Market/board/strike data
    - IV surface construction
    - LP vault analytics
    - Trade history
    - Cross-venue comparison
    """
    
    VENUE = 'lyra'
    VENUE_TYPE = 'DEX_OPTIONS_AMM'
    API_URL = 'https://api.lyra.finance'
    SUBGRAPH_URLS = {
        'optimism': 'https://api.thegraph.com/subgraphs/name/lyra-finance/mainnet',
        'arbitrum': 'https://api.thegraph.com/subgraphs/name/lyra-finance/arbitrum',
    }
    
    MARKETS_QUERY = """
    query GetMarkets {
        markets(first: 100) {
            id
            baseAsset
            quoteAsset
            baseIv
            skewRatio
            totalLiquidity
            totalOpenInterest
            utilizationRate
            isActive
        }
    }
    """
    
    BOARDS_QUERY = """
    query GetBoards($market: String!) {
        boards(where: {market: $market}, first: 100, orderBy: expiry, orderDirection: asc) {
            id
            market { id }
            expiry
            baseIv
            ivVariance
            totalLongCalls
            totalShortCalls
            totalLongPuts
            totalShortPuts
            isFrozen
        }
    }
    """
    
    STRIKES_QUERY = """
    query GetStrikes($board: String!) {
        strikes(where: {board: $board}, first: 100) {
            id
            board { id, expiry }
            strikePrice
            skew
            longCallOpenInterest
            shortCallOpenInterest
            longPutOpenInterest
            shortPutOpenInterest
        }
    }
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Lyra collector."""
        config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        self.chain = config.get('chain', 'arbitrum')
        self.subgraph_url = self.SUBGRAPH_URLS.get(self.chain, self.SUBGRAPH_URLS['arbitrum'])
        self.rate_limit = config.get('rate_limit', 30)
        self.last_request_time = 0.0
        self.min_request_interval = 60.0 / self.rate_limit
        self.collection_stats = {'requests': 0, 'records': 0, 'errors': 0}
    
    async def __aenter__(self) -> 'LyraCollector':
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
    
    async def _query_subgraph(self, query: str, variables: Optional[Dict] = None) -> Optional[Dict]:
        """Execute GraphQL query against subgraph."""
        await self._rate_limit()
        session = await self._get_session()
        self.collection_stats['requests'] += 1
        
        try:
            async with session.post(
                self.subgraph_url,
                json={'query': query, 'variables': variables or {}},
                headers={'Content-Type': 'application/json'},
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'errors' in data:
                        self.logger.error(f"GraphQL errors: {data['errors']}")
                        self.collection_stats['errors'] += 1
                        return None
                    return data.get('data')
                self.collection_stats['errors'] += 1
                return None
        except Exception as e:
            self.logger.error(f"Subgraph query error: {e}")
            self.collection_stats['errors'] += 1
            return None
    
    async def _api_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make REST API request."""
        await self._rate_limit()
        session = await self._get_session()
        self.collection_stats['requests'] += 1
        
        try:
            async with session.get(f"{self.API_URL}{endpoint}", params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                self.collection_stats['errors'] += 1
                return None
        except Exception as e:
            self.logger.error(f"API request error: {e}")
            self.collection_stats['errors'] += 1
            return None
    
    async def fetch_markets(self) -> List[Market]:
        """Fetch all Lyra markets."""
        data = await self._query_subgraph(self.MARKETS_QUERY)
        markets = []
        
        if data and 'markets' in data:
            for m in data['markets']:
                markets.append(Market(
                    market_address=m.get('id', ''),
                    underlying=m.get('baseAsset', ''),
                    chain=self.chain,
                    base_iv=float(m.get('baseIv', 0)),
                    skew_ratio=float(m.get('skewRatio', 0)),
                    total_liquidity=float(m.get('totalLiquidity', 0)),
                    total_open_interest=float(m.get('totalOpenInterest', 0)),
                    utilization_rate=float(m.get('utilizationRate', 0)),
                    is_active=m.get('isActive', True),
                ))
                self.collection_stats['records'] += 1
        
        return markets
    
    async def fetch_boards(self, market_address: str) -> List[Board]:
        """Fetch boards (expiries) for a market."""
        data = await self._query_subgraph(self.BOARDS_QUERY, {'market': market_address})
        boards = []
        
        if data and 'boards' in data:
            for b in data['boards']:
                expiry_ts = int(b.get('expiry', 0))
                boards.append(Board(
                    board_id=b.get('id', ''),
                    market_address=market_address,
                    underlying='',
                    expiry=datetime.fromtimestamp(expiry_ts, tz=timezone.utc) if expiry_ts else datetime.now(timezone.utc),
                    base_iv=float(b.get('baseIv', 0)),
                    iv_variance=float(b.get('ivVariance', 0)),
                    total_long_calls=float(b.get('totalLongCalls', 0)),
                    total_short_calls=float(b.get('totalShortCalls', 0)),
                    total_long_puts=float(b.get('totalLongPuts', 0)),
                    total_short_puts=float(b.get('totalShortPuts', 0)),
                    is_frozen=b.get('isFrozen', False),
                ))
                self.collection_stats['records'] += 1
        
        return boards
    
    async def fetch_strikes(self, board_id: str, spot_price: float = 0.0) -> List[Strike]:
        """Fetch strikes for a board."""
        data = await self._query_subgraph(self.STRIKES_QUERY, {'board': board_id})
        strikes = []
        
        if data and 'strikes' in data:
            for s in data['strikes']:
                expiry_ts = int(s.get('board', {}).get('expiry', 0))
                long_call = float(s.get('longCallOpenInterest', 0))
                short_call = float(s.get('shortCallOpenInterest', 0))
                long_put = float(s.get('longPutOpenInterest', 0))
                short_put = float(s.get('shortPutOpenInterest', 0))
                
                for opt_type in ['call', 'put']:
                    strikes.append(Strike(
                        strike_id=f"{s.get('id', '')}_{opt_type}",
                        board_id=board_id,
                        market_address='',
                        underlying='',
                        strike_price=float(s.get('strikePrice', 0)),
                        expiry=datetime.fromtimestamp(expiry_ts, tz=timezone.utc) if expiry_ts else datetime.now(timezone.utc),
                        option_type=opt_type,
                        skew=float(s.get('skew', 0)),
                        long_oi=long_call if opt_type == 'call' else long_put,
                        short_oi=short_call if opt_type == 'call' else short_put,
                        spot_price=spot_price,
                    ))
                    self.collection_stats['records'] += 1
        
        return strikes
    
    async def _fetch_single_board_strikes(self, board: Board, underlying: str, market_address: str, spot_price: float) -> List[Dict[str, Any]]:
        """Helper to fetch strikes for a single board."""
        try:
            if not board.is_tradeable:
                return []

            strikes = await self.fetch_strikes(board.board_id, spot_price)
            strike_dicts = []
            for strike in strikes:
                strike_dict = strike.to_dict()
                strike_dict['underlying'] = underlying
                strike_dict['market_address'] = market_address
                strike_dict['base_iv'] = board.base_iv
                strike_dicts.append(strike_dict)
            return strike_dicts
        except Exception as e:
            self.logger.error(f"Error fetching strikes for board {board.board_id}: {e}")
            return []

    async def fetch_option_chain(self, underlying: str = 'ETH') -> pd.DataFrame:
        """Fetch complete option chain for underlying."""
        self.logger.info(f"Fetching Lyra option chain for {underlying}")
        markets = await self.fetch_markets()

        target_market = None
        for market in markets:
            if market.underlying.upper() == underlying.upper():
                target_market = market
                break

        if not target_market:
            return pd.DataFrame()

        boards = await self.fetch_boards(target_market.market_address)

        # Parallelize board fetching
        tasks = [
            self._fetch_single_board_strikes(board, underlying, target_market.market_address, target_market.spot_price)
            for board in boards
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and filter out errors
        all_strikes = []
        for result in results:
            if isinstance(result, list):
                all_strikes.extend(result)

        df = pd.DataFrame(all_strikes)
        if not df.empty:
            df['venue'] = self.VENUE
            df['venue_type'] = self.VENUE_TYPE
            df['chain'] = self.chain
        return df
    
    async def build_volatility_surface(self, underlying: str = 'ETH') -> pd.DataFrame:
        """Build implied volatility surface."""
        chain = await self.fetch_option_chain(underlying)
        
        if chain.empty:
            return pd.DataFrame()
        
        surface_data = chain[['underlying', 'strike_price', 'days_to_expiry', 'moneyness_ratio',
                               'log_moneyness', 'option_type', 'iv', 'iv_pct', 'delta', 'skew']].copy()
        surface_data = surface_data[surface_data['iv'] > 0]
        
        return surface_data
    
    async def compare_with_deribit(self, underlying: str = 'ETH') -> pd.DataFrame:
        """Get Lyra data formatted for Deribit comparison."""
        chain = await self.fetch_option_chain(underlying)
        if chain.empty:
            return chain
        chain = chain.rename(columns={'iv': 'implied_volatility', 'skew_adjusted_iv': 'mark_iv'})
        return chain
    
    async def fetch_funding_rates(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Options AMM doesn't have perpetual funding rates."""
        return pd.DataFrame()
    
    async def fetch_ohlcv(self, symbols: List[str], timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Options AMM doesn't provide standard OHLCV."""
        return pd.DataFrame()

    async def _fetch_single_symbol_chain(self, symbol: str) -> pd.DataFrame:
        """Helper to fetch option chain for a single symbol."""
        try:
            return await self.fetch_option_chain(underlying=symbol)
        except Exception as e:
            self.logger.error(f"Error fetching chain for {symbol}: {e}")
            return pd.DataFrame()

    async def collect_options(self, symbols: List[str], start_date: Any, end_date: Any, **kwargs) -> pd.DataFrame:
        """Standardized collect_options wrapper - wraps fetch_option_chain()."""
        try:
            # Normalize symbols (WETH -> ETH, WBTC -> BTC)
            normalized_symbols = [
                symbol.replace('WBTC', 'BTC').replace('WETH', 'ETH').upper()
                for symbol in symbols
            ]

            # Parallelize symbol fetching
            tasks = [self._fetch_single_symbol_chain(symbol) for symbol in normalized_symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out errors and empty DataFrames
            all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

            # Combine all data
            if all_data:
                return pd.concat(all_data, ignore_index=True)

            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Lyra collect_options error: {e}")
            return pd.DataFrame()

    def get_collection_stats(self) -> Dict[str, Any]:
        return {**self.collection_stats, 'venue': self.VENUE, 'chain': self.chain}
    
    @staticmethod
    def get_supported_underlyings() -> List[str]:
        return [u.value for u in Underlying]
    
    @staticmethod
    def get_supported_chains() -> List[str]:
        return [c.value for c in Chain]
    
    async def close(self) -> None:
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None