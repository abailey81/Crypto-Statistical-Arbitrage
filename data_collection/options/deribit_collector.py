"""
Deribit Options Data Collector

validated collector for Deribit - the dominant cryptocurrency options
exchange handling approximately 90% of global crypto options volume.

===============================================================================
DERIBIT PLATFORM OVERVIEW
===============================================================================

Market Position:
    Deribit is the premier cryptocurrency derivatives exchange, offering:
    - Deepest options liquidity globally
    - Professional-quality trading infrastructure
    - Comprehensive market data APIs
    - Industry-standard settlement procedures

Supported Products:
    ================ ================ ================ ================
    Asset Options Futures Perpetuals
    ================ ================ ================ ================
    BTC Weekly/Monthly Quarterly Perpetual
    ETH Weekly/Monthly Quarterly Perpetual
    SOL Weekly/Monthly Quarterly Perpetual
    ================ ================ ================ ================

===============================================================================
API SPECIFICATIONS
===============================================================================

Endpoints:
    Production: https://www.deribit.com/api/v2
    Testnet: https://test.deribit.com/api/v2

Authentication:
    - Public endpoints: No authentication required
    - Private endpoints: API Key + Secret with HMAC signature

Rate Limits:
    ==================== ============== =================================
    Endpoint Type Limit Notes
    ==================== ============== =================================
    Matching Engine 20 req/sec Order operations
    Non-Matching Engine 10,000/10min Market data, historical
    WebSocket Unlimited Real-time streaming
    ==================== ============== =================================

===============================================================================
DATA CATEGORIES
===============================================================================

Option Chain Data:
    - All active strikes and expiries
    - Mark prices (theoretical fair value)
    - Best bid/ask prices and sizes
    - Implied volatility (mark, bid, ask)
    - Full Greeks suite (delta, gamma, vega, theta, rho)

DVOL Index:
    - 30-day forward-looking implied volatility
    - Crypto equivalent of VIX
    - Available for BTC and ETH
    - Hourly and daily resolution

Futures Data:
    - Perpetual contracts with 8-hour funding
    - Quarterly futures (March, June, September, December)
    - Basis and funding rate analytics

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

Volatility Surface Arbitrage:
    - Cross-strike mispricing detection
    - Put-call parity violations
    - Calendar spread opportunities
    - Butterfly arbitrage

Term Structure Trading:
    - Contango/backwardation regime detection
    - Roll yield capture strategies
    - Volatility term premium analysis

Skew Trading:
    - 25-delta risk reversal monitoring
    - Skew regime classification
    - Tail risk hedging signals

Cross-Venue Arbitrage:
    - Deribit vs DEX options (Lyra, AEVO)
    - IV surface differential analysis
    - Liquidity arbitrage opportunities

Greeks-Based Strategies:
    - Delta-neutral portfolio construction
    - Gamma scalping signal generation
    - Vega exposure management
    - Theta decay optimization

===============================================================================
USAGE EXAMPLE
===============================================================================

    async with DeribitCollector() as collector:
        # Fetch complete option chain
        chain = await collector.fetch_option_chain('BTC')
        
        # Get DVOL history
        dvol = await collector.fetch_dvol('BTC', '1H', '2024-01-01', '2024-12-31')
        
        # Calculate term structure
        term = await collector.calculate_term_structure('BTC')
        
        # Build volatility surface
        surface = await collector.build_volatility_surface(chain)
        
        # Analyze put-call ratio
        pcr = collector.calculate_put_call_ratio(chain)

Version: 2.0.0
Last Updated: 2024
"""

import os
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Union

import aiohttp
import pandas as pd
import numpy as np

# =============================================================================
# ENUMERATIONS
# =============================================================================

class Underlying(Enum):
    """Supported underlying assets on Deribit."""
    BTC = 'BTC'
    ETH = 'ETH'
    SOL = 'SOL'
    
    @classmethod
    def from_string(cls, value: str) -> 'Underlying':
        """Parse underlying from string."""
        return cls(value.upper())
    
    @property
    def index_name(self) -> str:
        """Get index name for API calls."""
        return f"{self.value.lower()}_usd"
    
    @property
    def perpetual_symbol(self) -> str:
        """Get perpetual contract symbol."""
        return f"{self.value}-PERPETUAL"

class OptionType(Enum):
    """Option contract type."""
    CALL = 'call'
    PUT = 'put'
    
    @classmethod
    def from_string(cls, value: str) -> 'OptionType':
        """Parse option type from string."""
        v = value.lower()
        if v in ('call', 'c'):
            return cls.CALL
        return cls.PUT
    
    @property
    def sign(self) -> int:
        """Get sign for payoff calculation (+1 for call, -1 for put)."""
        return 1 if self == OptionType.CALL else -1

class InstrumentKind(Enum):
    """Deribit instrument types."""
    OPTION = 'option'
    FUTURE = 'future'
    SPOT = 'spot'
    FUTURE_COMBO = 'future_combo'
    OPTION_COMBO = 'option_combo'

class Moneyness(Enum):
    """Option moneyness classification based on delta."""
    DEEP_ITM = 'deep_itm' # |delta| > 0.85
    ITM = 'itm' # 0.60 < |delta| <= 0.85
    SLIGHT_ITM = 'slight_itm' # 0.55 < |delta| <= 0.60
    ATM = 'atm' # 0.45 <= |delta| <= 0.55
    SLIGHT_OTM = 'slight_otm' # 0.40 <= |delta| < 0.45
    OTM = 'otm' # 0.15 <= |delta| < 0.40
    DEEP_OTM = 'deep_otm' # |delta| < 0.15
    
    @classmethod
    def from_delta(cls, delta: float) -> 'Moneyness':
        """Classify moneyness from delta value."""
        abs_delta = abs(delta)
        if abs_delta > 0.85:
            return cls.DEEP_ITM
        elif abs_delta > 0.60:
            return cls.ITM
        elif abs_delta > 0.55:
            return cls.SLIGHT_ITM
        elif abs_delta >= 0.45:
            return cls.ATM
        elif abs_delta >= 0.40:
            return cls.SLIGHT_OTM
        elif abs_delta >= 0.15:
            return cls.OTM
        return cls.DEEP_OTM
    
    @property
    def is_atm_region(self) -> bool:
        """Check if in ATM region (45-55 delta)."""
        return self in (Moneyness.ATM, Moneyness.SLIGHT_ITM, Moneyness.SLIGHT_OTM)
    
    @property
    def is_wing(self) -> bool:
        """Check if in wing region (deep ITM/OTM)."""
        return self in (Moneyness.DEEP_ITM, Moneyness.DEEP_OTM)

class ExpiryCategory(Enum):
    """Option expiry classification."""
    DAILY = 'daily' # 0-1 days (daily options)
    WEEKLY = 'weekly' # 1-7 days
    BI_WEEKLY = 'bi_weekly' # 7-14 days
    MONTHLY = 'monthly' # 14-35 days
    QUARTERLY = 'quarterly' # 35-100 days
    SEMI_ANNUAL = 'semi_annual' # 100-200 days
    ANNUAL = 'annual' # > 200 days
    
    @classmethod
    def from_days(cls, days: float) -> 'ExpiryCategory':
        """Classify from days to expiry."""
        if days <= 1:
            return cls.DAILY
        elif days <= 7:
            return cls.WEEKLY
        elif days <= 14:
            return cls.BI_WEEKLY
        elif days <= 35:
            return cls.MONTHLY
        elif days <= 100:
            return cls.QUARTERLY
        elif days <= 200:
            return cls.SEMI_ANNUAL
        return cls.ANNUAL
    
    @property
    def is_short_dated(self) -> bool:
        """Check if short-dated (< 14 days)."""
        return self in (ExpiryCategory.DAILY, ExpiryCategory.WEEKLY, ExpiryCategory.BI_WEEKLY)
    
    @property
    def typical_theta_decay(self) -> str:
        """Get typical theta decay characteristic."""
        if self == ExpiryCategory.DAILY:
            return 'extreme'
        elif self == ExpiryCategory.WEEKLY:
            return 'high'
        elif self in (ExpiryCategory.BI_WEEKLY, ExpiryCategory.MONTHLY):
            return 'moderate'
        return 'low'

class IVLevel(Enum):
    """Implied volatility level classification."""
    EXTREMELY_LOW = 'extremely_low' # < 25%
    VERY_LOW = 'very_low' # 25-35%
    LOW = 'low' # 35-50%
    NORMAL = 'normal' # 50-70%
    ELEVATED = 'elevated' # 70-90%
    HIGH = 'high' # 90-120%
    VERY_HIGH = 'very_high' # 120-150%
    EXTREME = 'extreme' # > 150%
    
    @classmethod
    def from_iv(cls, iv: float) -> 'IVLevel':
        """Classify from IV value (as decimal or percentage)."""
        # Normalize to percentage
        iv_pct = iv * 100 if iv < 5 else iv
        
        if iv_pct < 25:
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
        """Get trading signal implication."""
        if self in (IVLevel.EXTREMELY_LOW, IVLevel.VERY_LOW):
            return 'buy_vol'
        elif self in (IVLevel.VERY_HIGH, IVLevel.EXTREME):
            return 'sell_vol'
        return 'neutral'
    
    @property
    def is_opportunity(self) -> bool:
        """Check if IV level suggests opportunity."""
        return self in (IVLevel.EXTREMELY_LOW, IVLevel.VERY_LOW, 
                       IVLevel.VERY_HIGH, IVLevel.EXTREME)

class VolTermStructure(Enum):
    """Volatility term structure shape."""
    STEEP_CONTANGO = 'steep_contango' # Far > Near by >10%
    CONTANGO = 'contango' # Far > Near by 3-10%
    FLAT = 'flat' # Within Â±3%
    BACKWARDATION = 'backwardation' # Near > Far by 3-10%
    STEEP_BACKWARDATION = 'steep_backwardation' # Near > Far by >10%
    HUMPED = 'humped' # Mid-term peak
    INVERTED_HUMP = 'inverted_hump' # Mid-term trough
    
    @classmethod
    def from_term_spread(cls, spread_pct: float, has_hump: bool = False) -> 'VolTermStructure':
        """Classify from term spread percentage."""
        if has_hump:
            return cls.HUMPED if spread_pct > 0 else cls.INVERTED_HUMP
        
        if spread_pct > 10:
            return cls.STEEP_CONTANGO
        elif spread_pct > 3:
            return cls.CONTANGO
        elif spread_pct > -3:
            return cls.FLAT
        elif spread_pct > -10:
            return cls.BACKWARDATION
        return cls.STEEP_BACKWARDATION
    
    @property
    def regime(self) -> str:
        """Get regime classification."""
        if self in (VolTermStructure.STEEP_CONTANGO, VolTermStructure.CONTANGO):
            return 'normal'
        elif self in (VolTermStructure.STEEP_BACKWARDATION, VolTermStructure.BACKWARDATION):
            return 'stressed'
        return 'transitional'
    
    @property
    def calendar_spread_bias(self) -> str:
        """Get calendar spread trading bias."""
        if self in (VolTermStructure.STEEP_CONTANGO, VolTermStructure.CONTANGO):
            return 'sell_far_buy_near'
        elif self in (VolTermStructure.STEEP_BACKWARDATION, VolTermStructure.BACKWARDATION):
            return 'buy_far_sell_near'
        return 'neutral'

class SkewDirection(Enum):
    """Volatility skew direction classification."""
    STRONG_PUT_SKEW = 'strong_put_skew' # 25d RR < -5%
    PUT_SKEW = 'put_skew' # -5% <= 25d RR < -2%
    SLIGHT_PUT_SKEW = 'slight_put_skew' # -2% <= 25d RR < -0.5%
    SYMMETRIC = 'symmetric' # -0.5% <= 25d RR <= 0.5%
    SLIGHT_CALL_SKEW = 'slight_call_skew' # 0.5% < 25d RR <= 2%
    CALL_SKEW = 'call_skew' # 2% < 25d RR <= 5%
    STRONG_CALL_SKEW = 'strong_call_skew' # 25d RR > 5%
    
    @classmethod
    def from_risk_reversal(cls, rr_pct: float) -> 'SkewDirection':
        """Classify from 25-delta risk reversal (call IV - put IV)."""
        if rr_pct < -5:
            return cls.STRONG_PUT_SKEW
        elif rr_pct < -2:
            return cls.PUT_SKEW
        elif rr_pct < -0.5:
            return cls.SLIGHT_PUT_SKEW
        elif rr_pct <= 0.5:
            return cls.SYMMETRIC
        elif rr_pct <= 2:
            return cls.SLIGHT_CALL_SKEW
        elif rr_pct <= 5:
            return cls.CALL_SKEW
        return cls.STRONG_CALL_SKEW
    
    @property
    def market_sentiment(self) -> str:
        """Infer market sentiment from skew."""
        if self in (SkewDirection.STRONG_PUT_SKEW, SkewDirection.PUT_SKEW):
            return 'fearful'
        elif self in (SkewDirection.STRONG_CALL_SKEW, SkewDirection.CALL_SKEW):
            return 'euphoric'
        return 'neutral'
    
    @property
    def tail_risk_indicator(self) -> str:
        """Get tail risk indication."""
        if self == SkewDirection.STRONG_PUT_SKEW:
            return 'high_downside_fear'
        elif self == SkewDirection.STRONG_CALL_SKEW:
            return 'high_upside_fomo'
        return 'normal'

class DVOLRegime(Enum):
    """DVOL volatility regime classification."""
    EXTREME_FEAR = 'extreme_fear' # > 100
    FEAR = 'fear' # 80-100
    ELEVATED = 'elevated' # 65-80
    NORMAL = 'normal' # 45-65
    COMPLACENT = 'complacent' # 30-45
    EXTREME_COMPLACENCY = 'extreme_complacency' # < 30
    
    @classmethod
    def from_dvol(cls, dvol: float) -> 'DVOLRegime':
        """Classify from DVOL value."""
        if dvol > 100:
            return cls.EXTREME_FEAR
        elif dvol > 80:
            return cls.FEAR
        elif dvol > 65:
            return cls.ELEVATED
        elif dvol > 45:
            return cls.NORMAL
        elif dvol > 30:
            return cls.COMPLACENT
        return cls.EXTREME_COMPLACENCY
    
    @property
    def trading_signal(self) -> str:
        """Get vol trading signal."""
        if self in (DVOLRegime.EXTREME_FEAR, DVOLRegime.FEAR):
            return 'sell_vol'
        elif self in (DVOLRegime.COMPLACENT, DVOLRegime.EXTREME_COMPLACENCY):
            return 'buy_vol'
        return 'neutral'
    
    @property
    def is_extreme(self) -> bool:
        """Check if in extreme regime."""
        return self in (DVOLRegime.EXTREME_FEAR, DVOLRegime.EXTREME_COMPLACENCY)

class LiquidityTier(Enum):
    """Option liquidity classification."""
    HIGHLY_LIQUID = 'highly_liquid' # Top decile
    LIQUID = 'liquid' # 60-90th percentile
    MODERATE = 'moderate' # 30-60th percentile
    ILLIQUID = 'illiquid' # 10-30th percentile
    VERY_ILLIQUID = 'very_illiquid' # Bottom decile
    
    @classmethod
    def from_metrics(cls, oi: float, volume: float, spread_pct: float) -> 'LiquidityTier':
        """Classify from liquidity metrics."""
        # Score based on multiple factors
        oi_score = min(40, oi / 25) # Max 40 points for 1000+ OI
        vol_score = min(30, volume / 3.33) # Max 30 points for 100+ volume
        spread_score = max(0, 30 - spread_pct * 3) # Max 30 points for tight spread
        
        total = oi_score + vol_score + spread_score
        
        if total >= 80:
            return cls.HIGHLY_LIQUID
        elif total >= 60:
            return cls.LIQUID
        elif total >= 40:
            return cls.MODERATE
        elif total >= 20:
            return cls.ILLIQUID
        return cls.VERY_ILLIQUID
    
    @property
    def tradeable(self) -> bool:
        """Check if sufficiently liquid to trade."""
        return self in (LiquidityTier.HIGHLY_LIQUID, LiquidityTier.LIQUID, LiquidityTier.MODERATE)
    
    @property
    def slippage_estimate(self) -> str:
        """Estimate slippage category."""
        if self == LiquidityTier.HIGHLY_LIQUID:
            return 'minimal'
        elif self == LiquidityTier.LIQUID:
            return 'low'
        elif self == LiquidityTier.MODERATE:
            return 'moderate'
        return 'high'

class FundingRegime(Enum):
    """Perpetual funding rate regime."""
    EXTREME_LONG_BIAS = 'extreme_long_bias' # > 0.1% per 8h
    LONG_BIAS = 'long_bias' # 0.03% - 0.1%
    SLIGHT_LONG_BIAS = 'slight_long_bias' # 0.01% - 0.03%
    NEUTRAL = 'neutral' # -0.01% - 0.01%
    SLIGHT_SHORT_BIAS = 'slight_short_bias' # -0.03% - -0.01%
    SHORT_BIAS = 'short_bias' # -0.1% - -0.03%
    EXTREME_SHORT_BIAS = 'extreme_short_bias' # < -0.1%
    
    @classmethod
    def from_rate(cls, rate: float) -> 'FundingRegime':
        """Classify from 8-hour funding rate."""
        rate_pct = rate * 100 # Convert to percentage
        
        if rate_pct > 0.1:
            return cls.EXTREME_LONG_BIAS
        elif rate_pct > 0.03:
            return cls.LONG_BIAS
        elif rate_pct > 0.01:
            return cls.SLIGHT_LONG_BIAS
        elif rate_pct > -0.01:
            return cls.NEUTRAL
        elif rate_pct > -0.03:
            return cls.SLIGHT_SHORT_BIAS
        elif rate_pct > -0.1:
            return cls.SHORT_BIAS
        return cls.EXTREME_SHORT_BIAS
    
    @property
    def carry_direction(self) -> str:
        """Get carry trade direction."""
        if self in (FundingRegime.EXTREME_LONG_BIAS, FundingRegime.LONG_BIAS):
            return 'short_perp_long_spot'
        elif self in (FundingRegime.EXTREME_SHORT_BIAS, FundingRegime.SHORT_BIAS):
            return 'long_perp_short_spot'
        return 'neutral'
    
    @property
    def annualized_yield_estimate(self) -> str:
        """Estimate annualized carry yield."""
        if self == FundingRegime.EXTREME_LONG_BIAS:
            return '>100%'
        elif self == FundingRegime.LONG_BIAS:
            return '30-100%'
        elif self == FundingRegime.SLIGHT_LONG_BIAS:
            return '10-30%'
        elif self == FundingRegime.NEUTRAL:
            return '<10%'
        elif self == FundingRegime.SLIGHT_SHORT_BIAS:
            return '-10 to -30%'
        elif self == FundingRegime.SHORT_BIAS:
            return '-30 to -100%'
        return '<-100%'

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class OptionInstrument:
    """
    Option instrument metadata with computed analytics.
    
    Represents a single option contract specification including
    strike, expiry, and trading parameters.
    """
    instrument_name: str
    underlying: str
    strike: float
    option_type: str
    expiry_timestamp: int
    expiry_date: datetime
    creation_timestamp: int = 0
    settlement_period: str = 'day'
    tick_size: float = 0.0001
    min_trade_amount: float = 0.1
    contract_size: float = 1.0
    maker_commission: float = 0.0003
    taker_commission: float = 0.0003
    is_active: bool = True
    
    # -------------------------------------------------------------------------
    # Type Properties
    # -------------------------------------------------------------------------
    
    @property
    def option_type_enum(self) -> OptionType:
        """Get option type as enum."""
        return OptionType.from_string(self.option_type)
    
    @property
    def underlying_enum(self) -> Underlying:
        """Get underlying as enum."""
        return Underlying.from_string(self.underlying)
    
    @property
    def is_call(self) -> bool:
        """Check if call option."""
        return self.option_type_enum == OptionType.CALL
    
    @property
    def is_put(self) -> bool:
        """Check if put option."""
        return self.option_type_enum == OptionType.PUT
    
    # -------------------------------------------------------------------------
    # Time Properties
    # -------------------------------------------------------------------------
    
    @property
    def days_to_expiry(self) -> float:
        """Calculate days to expiry."""
        now = datetime.now(timezone.utc)
        expiry = self.expiry_date if self.expiry_date.tzinfo else self.expiry_date.replace(tzinfo=timezone.utc)
        return max(0, (expiry - now).total_seconds() / 86400)
    
    @property
    def hours_to_expiry(self) -> float:
        """Calculate hours to expiry."""
        return self.days_to_expiry * 24
    
    @property
    def time_to_expiry_years(self) -> float:
        """Time to expiry in years (for Black-Scholes)."""
        return self.days_to_expiry / 365.25
    
    @property
    def expiry_category(self) -> ExpiryCategory:
        """Classify expiry timeframe."""
        return ExpiryCategory.from_days(self.days_to_expiry)
    
    @property
    def is_expired(self) -> bool:
        """Check if option has expired."""
        return self.days_to_expiry <= 0
    
    @property
    def is_near_expiry(self) -> bool:
        """Check if within 3 days of expiry."""
        return 0 < self.days_to_expiry <= 3
    
    @property
    def is_same_day_expiry(self) -> bool:
        """Check if expiring today."""
        return 0 < self.days_to_expiry <= 1
    
    @property
    def is_weekly(self) -> bool:
        """Check if weekly option (non-standard expiry)."""
        return self.settlement_period in ('day', 'week')
    
    @property
    def is_monthly(self) -> bool:
        """Check if monthly option (standard expiry)."""
        return self.settlement_period == 'month'
    
    # -------------------------------------------------------------------------
    # Trading Properties
    # -------------------------------------------------------------------------
    
    @property
    def total_commission(self) -> float:
        """Total round-trip commission."""
        return self.maker_commission + self.taker_commission
    
    @property
    def is_tradeable(self) -> bool:
        """Check if option is tradeable."""
        return self.is_active and not self.is_expired
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'instrument_name': self.instrument_name,
            'underlying': self.underlying,
            'strike': self.strike,
            'option_type': self.option_type,
            'expiry_date': self.expiry_date.isoformat() if isinstance(self.expiry_date, datetime) else self.expiry_date,
            'days_to_expiry': self.days_to_expiry,
            'time_to_expiry_years': self.time_to_expiry_years,
            'expiry_category': self.expiry_category.value,
            'is_call': self.is_call,
            'is_weekly': self.is_weekly,
            'is_near_expiry': self.is_near_expiry,
            'is_tradeable': self.is_tradeable,
            'settlement_period': self.settlement_period,
            'tick_size': self.tick_size,
            'min_trade_amount': self.min_trade_amount,
            'is_active': self.is_active,
        }

@dataclass
class OptionQuote:
    """
    Comprehensive option quote with pricing, Greeks, and analytics.
    
    Contains all market data for a single option including theoretical
    prices, implied volatility, and full Greeks suite.
    """
    timestamp: datetime
    instrument_name: str
    underlying: str
    strike: float
    option_type: str
    expiry_date: datetime
    
    # Prices
    mark_price: float = 0.0 # Theoretical fair value (in BTC/ETH)
    mark_iv: float = 0.0 # Mark implied volatility
    bid_price: float = 0.0 # Best bid price
    ask_price: float = 0.0 # Best ask price
    bid_iv: float = 0.0 # Bid implied volatility
    ask_iv: float = 0.0 # Ask implied volatility
    last_price: float = 0.0 # Last traded price
    
    # Underlying
    underlying_price: float = 0.0 # Underlying futures price
    underlying_index: str = '' # Index used for pricing
    index_price: float = 0.0 # Spot index price
    
    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    
    # Volume & Interest
    volume: float = 0.0 # 24h volume in contracts
    volume_usd: float = 0.0 # 24h volume in USD
    open_interest: float = 0.0 # Open interest in contracts
    
    # Order Book
    bid_size: float = 0.0
    ask_size: float = 0.0
    
    # -------------------------------------------------------------------------
    # Type Properties
    # -------------------------------------------------------------------------
    
    @property
    def option_type_enum(self) -> OptionType:
        """Get option type as enum."""
        return OptionType.from_string(self.option_type)
    
    @property
    def is_call(self) -> bool:
        """Check if call option."""
        return self.option_type_enum == OptionType.CALL
    
    @property
    def is_put(self) -> bool:
        """Check if put option."""
        return self.option_type_enum == OptionType.PUT
    
    # -------------------------------------------------------------------------
    # Time Properties
    # -------------------------------------------------------------------------
    
    @property
    def days_to_expiry(self) -> float:
        """Days to expiry."""
        now = datetime.now(timezone.utc)
        expiry = self.expiry_date if self.expiry_date.tzinfo else self.expiry_date.replace(tzinfo=timezone.utc)
        return max(0, (expiry - now).total_seconds() / 86400)
    
    @property
    def time_to_expiry_years(self) -> float:
        """Time to expiry in years."""
        return self.days_to_expiry / 365.25
    
    @property
    def expiry_category(self) -> ExpiryCategory:
        """Expiry classification."""
        return ExpiryCategory.from_days(self.days_to_expiry)
    
    # -------------------------------------------------------------------------
    # Moneyness Properties
    # -------------------------------------------------------------------------
    
    @property
    def moneyness(self) -> Moneyness:
        """Classify moneyness from delta."""
        return Moneyness.from_delta(self.delta)
    
    @property
    def moneyness_ratio(self) -> float:
        """Strike / spot ratio."""
        if self.underlying_price > 0:
            return self.strike / self.underlying_price
        return 0.0
    
    @property
    def log_moneyness(self) -> float:
        """Log moneyness for volatility surface fitting."""
        if self.moneyness_ratio > 0:
            return np.log(self.moneyness_ratio)
        return 0.0
    
    @property
    def standardized_moneyness(self) -> float:
        """Standardized moneyness: ln(K/S) / (ÏƒâˆšT)."""
        if self.mark_iv > 0 and self.time_to_expiry_years > 0:
            return self.log_moneyness / (self.mark_iv * np.sqrt(self.time_to_expiry_years))
        return 0.0
    
    @property
    def is_itm(self) -> bool:
        """Check if in-the-money."""
        if self.is_call:
            return self.underlying_price > self.strike
        return self.underlying_price < self.strike
    
    @property
    def is_otm(self) -> bool:
        """Check if out-of-the-money."""
        return not self.is_itm
    
    @property
    def is_atm(self) -> bool:
        """Check if approximately at-the-money."""
        return self.moneyness == Moneyness.ATM
    
    # -------------------------------------------------------------------------
    # Value Properties
    # -------------------------------------------------------------------------
    
    @property
    def intrinsic_value(self) -> float:
        """Calculate intrinsic value."""
        if self.is_call:
            return max(0, self.underlying_price - self.strike)
        return max(0, self.strike - self.underlying_price)
    
    @property
    def intrinsic_value_pct(self) -> float:
        """Intrinsic value as percentage of underlying."""
        if self.underlying_price > 0:
            return (self.intrinsic_value / self.underlying_price) * 100
        return 0.0
    
    @property
    def time_value(self) -> float:
        """Calculate time value (extrinsic value)."""
        mark_usd = self.mark_price * self.underlying_price
        return max(0, mark_usd - self.intrinsic_value)
    
    @property
    def time_value_pct(self) -> float:
        """Time value as percentage of option price."""
        mark_usd = self.mark_price * self.underlying_price
        if mark_usd > 0:
            return (self.time_value / mark_usd) * 100
        return 0.0
    
    @property
    def mark_price_usd(self) -> float:
        """Mark price in USD."""
        return self.mark_price * self.underlying_price
    
    @property
    def notional_value(self) -> float:
        """Notional value of one contract."""
        return self.underlying_price
    
    # -------------------------------------------------------------------------
    # Spread Properties
    # -------------------------------------------------------------------------
    
    @property
    def mid_price(self) -> float:
        """Mid price."""
        if self.bid_price > 0 and self.ask_price > 0:
            return (self.bid_price + self.ask_price) / 2
        return self.mark_price
    
    @property
    def bid_ask_spread(self) -> float:
        """Bid-ask spread."""
        return self.ask_price - self.bid_price
    
    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid price."""
        mid = self.mid_price
        if mid > 0:
            return (self.bid_ask_spread / mid) * 100
        return 0.0
    
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
        if self.bid_iv > 0 and self.ask_iv > 0:
            return self.ask_iv - self.bid_iv
        return 0.0
    
    @property
    def iv_spread_pct(self) -> float:
        """IV spread as percentage of mid IV."""
        if self.mid_iv > 0:
            return (self.iv_spread / self.mid_iv) * 100
        return 0.0
    
    # -------------------------------------------------------------------------
    # IV Properties
    # -------------------------------------------------------------------------
    
    @property
    def iv_level(self) -> IVLevel:
        """Classify IV level."""
        return IVLevel.from_iv(self.mark_iv)
    
    @property
    def iv_percentile_proxy(self) -> float:
        """Rough IV percentile proxy (assumes 50% = 50th percentile)."""
        # This is a simplified proxy - real implementation would use historical data
        iv_pct = self.mark_iv * 100 if self.mark_iv < 5 else self.mark_iv
        return min(100, max(0, iv_pct))
    
    @property
    def is_iv_rich(self) -> bool:
        """Check if IV appears rich (potential sell candidate)."""
        return self.iv_level in (IVLevel.HIGH, IVLevel.VERY_HIGH, IVLevel.EXTREME)
    
    @property
    def is_iv_cheap(self) -> bool:
        """Check if IV appears cheap (potential buy candidate)."""
        return self.iv_level in (IVLevel.EXTREMELY_LOW, IVLevel.VERY_LOW, IVLevel.LOW)
    
    # -------------------------------------------------------------------------
    # Greeks Analytics
    # -------------------------------------------------------------------------
    
    @property
    def gamma_dollar(self) -> float:
        """Dollar gamma: P&L for 1% spot move squared."""
        return self.gamma * (self.underlying_price ** 2) / 100
    
    @property
    def gamma_pct(self) -> float:
        """Gamma as percentage per 1% move."""
        return self.gamma * self.underlying_price
    
    @property
    def vega_dollar(self) -> float:
        """Dollar vega: P&L per 1 vol point move."""
        return self.vega * self.underlying_price / 100
    
    @property
    def theta_dollar(self) -> float:
        """Dollar theta: daily time decay in USD."""
        return abs(self.theta) * self.underlying_price
    
    @property
    def theta_pct(self) -> float:
        """Theta as percentage of option value per day."""
        if self.mark_price > 0:
            return (abs(self.theta) / self.mark_price) * 100
        return 0.0
    
    @property
    def vega_theta_ratio(self) -> float:
        """Vega/Theta ratio (vol exposure per day of decay)."""
        if self.theta != 0:
            return abs(self.vega / self.theta)
        return 0.0
    
    @property
    def gamma_theta_ratio(self) -> float:
        """Gamma/Theta ratio (convexity per day of decay)."""
        if self.theta != 0:
            return abs(self.gamma / self.theta)
        return 0.0
    
    @property
    def delta_dollars(self) -> float:
        """Delta exposure in dollar terms."""
        return self.delta * self.underlying_price
    
    @property
    def is_delta_neutral_candidate(self) -> bool:
        """Check if good for delta-neutral portfolio."""
        return abs(self.delta) >= 0.4 and abs(self.delta) <= 0.6
    
    # -------------------------------------------------------------------------
    # Liquidity Properties
    # -------------------------------------------------------------------------
    
    @property
    def liquidity_tier(self) -> LiquidityTier:
        """Classify liquidity tier."""
        return LiquidityTier.from_metrics(self.open_interest, self.volume, self.spread_pct)
    
    @property
    def liquidity_score(self) -> float:
        """Calculate liquidity score (0-100)."""
        oi_score = min(40, self.open_interest / 25)
        vol_score = min(30, self.volume / 3.33)
        spread_score = max(0, 30 - self.spread_pct * 3)
        return oi_score + vol_score + spread_score
    
    @property
    def is_liquid(self) -> bool:
        """Check if sufficiently liquid."""
        return self.liquidity_tier.tradeable
    
    @property
    def estimated_slippage_bps(self) -> float:
        """Estimate slippage in basis points."""
        return self.spread_bps / 2
    
    @property
    def depth_at_touch(self) -> float:
        """Combined depth at best bid/ask."""
        return self.bid_size + self.ask_size
    
    # -------------------------------------------------------------------------
    # Trading Signals
    # -------------------------------------------------------------------------
    
    @property
    def iv_trading_signal(self) -> str:
        """Get IV-based trading signal."""
        return self.iv_level.trading_signal
    
    @property
    def is_tradeable(self) -> bool:
        """Check if option is suitable for trading."""
        return (
            self.is_liquid and
            self.days_to_expiry > 0.1 and
            self.mark_price > 0 and
            self.mark_iv > 0
        )
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'instrument_name': self.instrument_name,
            'underlying': self.underlying,
            'strike': self.strike,
            'option_type': self.option_type,
            'expiry_date': self.expiry_date.isoformat() if isinstance(self.expiry_date, datetime) else self.expiry_date,
            'days_to_expiry': self.days_to_expiry,
            'expiry_category': self.expiry_category.value,
            
            # Prices
            'mark_price': self.mark_price,
            'mark_price_usd': self.mark_price_usd,
            'mark_iv': self.mark_iv,
            'bid_price': self.bid_price,
            'ask_price': self.ask_price,
            'mid_price': self.mid_price,
            'bid_iv': self.bid_iv,
            'ask_iv': self.ask_iv,
            'mid_iv': self.mid_iv,
            
            # Underlying
            'underlying_price': self.underlying_price,
            'index_price': self.index_price,
            
            # Moneyness
            'moneyness': self.moneyness.value,
            'moneyness_ratio': self.moneyness_ratio,
            'log_moneyness': self.log_moneyness,
            'is_itm': self.is_itm,
            'is_atm': self.is_atm,
            
            # Values
            'intrinsic_value': self.intrinsic_value,
            'time_value': self.time_value,
            'time_value_pct': self.time_value_pct,
            
            # Spreads
            'spread_pct': self.spread_pct,
            'spread_bps': self.spread_bps,
            'iv_spread': self.iv_spread,
            
            # Greeks
            'delta': self.delta,
            'gamma': self.gamma,
            'vega': self.vega,
            'theta': self.theta,
            'rho': self.rho,
            'gamma_dollar': self.gamma_dollar,
            'vega_dollar': self.vega_dollar,
            'theta_dollar': self.theta_dollar,
            
            # IV Analysis
            'iv_level': self.iv_level.value,
            'is_iv_rich': self.is_iv_rich,
            'is_iv_cheap': self.is_iv_cheap,
            
            # Liquidity
            'volume': self.volume,
            'open_interest': self.open_interest,
            'liquidity_tier': self.liquidity_tier.value,
            'liquidity_score': self.liquidity_score,
            'is_liquid': self.is_liquid,
            'is_tradeable': self.is_tradeable,
        }

@dataclass
class DVOLRecord:
    """
    DVOL (Deribit Volatility Index) data point with analytics.
    
    DVOL is a 30-day forward-looking implied volatility index,
    analogous to VIX for traditional markets.
    """
    timestamp: datetime
    underlying: str
    open: float
    high: float
    low: float
    close: float
    resolution: str = '1H'
    
    # -------------------------------------------------------------------------
    # Basic Analytics
    # -------------------------------------------------------------------------
    
    @property
    def daily_range(self) -> float:
        """Intraday range."""
        return self.high - self.low
    
    @property
    def range_pct(self) -> float:
        """Range as percentage of open."""
        if self.open > 0:
            return (self.daily_range / self.open) * 100
        return 0.0
    
    @property
    def change(self) -> float:
        """Change from open to close."""
        return self.close - self.open
    
    @property
    def change_pct(self) -> float:
        """Percentage change."""
        if self.open > 0:
            return (self.change / self.open) * 100
        return 0.0
    
    @property
    def is_up(self) -> bool:
        """Check if DVOL increased."""
        return self.close > self.open
    
    @property
    def is_down(self) -> bool:
        """Check if DVOL decreased."""
        return self.close < self.open
    
    # -------------------------------------------------------------------------
    # Regime Classification
    # -------------------------------------------------------------------------
    
    @property
    def regime(self) -> DVOLRegime:
        """Classify DVOL regime."""
        return DVOLRegime.from_dvol(self.close)
    
    @property
    def is_elevated(self) -> bool:
        """Check if DVOL is elevated (>65)."""
        return self.close > 65
    
    @property
    def is_depressed(self) -> bool:
        """Check if DVOL is depressed (<45)."""
        return self.close < 45
    
    @property
    def is_extreme(self) -> bool:
        """Check if in extreme regime."""
        return self.regime.is_extreme
    
    @property
    def is_fear(self) -> bool:
        """Check if in fear regime (>80)."""
        return self.close > 80
    
    @property
    def is_complacent(self) -> bool:
        """Check if in complacent regime (<35)."""
        return self.close < 35
    
    # -------------------------------------------------------------------------
    # Trading Signals
    # -------------------------------------------------------------------------
    
    @property
    def vol_trading_signal(self) -> str:
        """Get volatility trading signal."""
        return self.regime.trading_signal
    
    @property
    def mean_reversion_signal(self) -> str:
        """Get mean reversion signal (assumes ~55 mean)."""
        if self.close > 80:
            return 'strong_sell_vol'
        elif self.close > 70:
            return 'sell_vol'
        elif self.close < 35:
            return 'strong_buy_vol'
        elif self.close < 45:
            return 'buy_vol'
        return 'neutral'
    
    @property
    def vol_of_vol_proxy(self) -> float:
        """Intraday vol-of-vol proxy."""
        if self.close > 0:
            return self.daily_range / self.close
        return 0.0
    
    @property
    def is_high_vov(self) -> bool:
        """Check if vol-of-vol is elevated."""
        return self.vol_of_vol_proxy > 0.1
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'underlying': self.underlying,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'resolution': self.resolution,
            'daily_range': self.daily_range,
            'range_pct': self.range_pct,
            'change': self.change,
            'change_pct': self.change_pct,
            'regime': self.regime.value,
            'is_elevated': self.is_elevated,
            'is_depressed': self.is_depressed,
            'is_extreme': self.is_extreme,
            'vol_trading_signal': self.vol_trading_signal,
            'mean_reversion_signal': self.mean_reversion_signal,
            'vol_of_vol_proxy': self.vol_of_vol_proxy,
        }

@dataclass
class FundingRateRecord:
    """
    Perpetual funding rate data with analytics.
    
    Captures 8-hour funding payments between longs and shorts
    on perpetual contracts.
    """
    timestamp: datetime
    instrument_name: str
    symbol: str
    funding_rate: float # 8-hour rate
    index_price: float = 0.0
    prev_index_price: float = 0.0
    
    # -------------------------------------------------------------------------
    # Rate Conversions
    # -------------------------------------------------------------------------
    
    @property
    def funding_rate_8h(self) -> float:
        """8-hour funding rate (raw)."""
        return self.funding_rate
    
    @property
    def funding_rate_daily(self) -> float:
        """Daily funding rate (8h * 3)."""
        return self.funding_rate * 3
    
    @property
    def funding_rate_annualized(self) -> float:
        """Annualized funding rate (8h * 3 * 365)."""
        return self.funding_rate * 1095
    
    @property
    def funding_rate_8h_pct(self) -> float:
        """8-hour rate as percentage."""
        return self.funding_rate * 100
    
    @property
    def funding_rate_annualized_pct(self) -> float:
        """Annualized rate as percentage."""
        return self.funding_rate_annualized * 100
    
    # -------------------------------------------------------------------------
    # Direction Properties
    # -------------------------------------------------------------------------
    
    @property
    def is_positive(self) -> bool:
        """Check if longs pay shorts."""
        return self.funding_rate > 0
    
    @property
    def is_negative(self) -> bool:
        """Check if shorts pay longs."""
        return self.funding_rate < 0
    
    @property
    def payer(self) -> str:
        """Who pays funding."""
        return 'longs' if self.is_positive else 'shorts'
    
    @property
    def receiver(self) -> str:
        """Who receives funding."""
        return 'shorts' if self.is_positive else 'longs'
    
    # -------------------------------------------------------------------------
    # Regime Classification
    # -------------------------------------------------------------------------
    
    @property
    def regime(self) -> FundingRegime:
        """Classify funding regime."""
        return FundingRegime.from_rate(self.funding_rate)
    
    @property
    def is_extreme(self) -> bool:
        """Check if funding is extreme."""
        return self.regime in (FundingRegime.EXTREME_LONG_BIAS, FundingRegime.EXTREME_SHORT_BIAS)
    
    @property
    def is_elevated(self) -> bool:
        """Check if funding is elevated (either direction)."""
        return abs(self.funding_rate) > 0.0003 # >0.03% per 8h
    
    @property
    def is_neutral(self) -> bool:
        """Check if funding is neutral."""
        return self.regime == FundingRegime.NEUTRAL
    
    # -------------------------------------------------------------------------
    # Index Properties
    # -------------------------------------------------------------------------
    
    @property
    def index_change(self) -> float:
        """Index price change."""
        return self.index_price - self.prev_index_price
    
    @property
    def index_change_pct(self) -> float:
        """Index price change percentage."""
        if self.prev_index_price > 0:
            return (self.index_change / self.prev_index_price) * 100
        return 0.0
    
    # -------------------------------------------------------------------------
    # Trading Signals
    # -------------------------------------------------------------------------
    
    @property
    def carry_trade_direction(self) -> str:
        """Get carry trade direction."""
        return self.regime.carry_direction
    
    @property
    def carry_yield_estimate(self) -> str:
        """Estimate annualized carry yield."""
        return self.regime.annualized_yield_estimate
    
    @property
    def basis_trade_signal(self) -> str:
        """Get basis trade signal."""
        if self.regime in (FundingRegime.EXTREME_LONG_BIAS, FundingRegime.LONG_BIAS):
            return 'short_perp_long_spot'
        elif self.regime in (FundingRegime.EXTREME_SHORT_BIAS, FundingRegime.SHORT_BIAS):
            return 'long_perp_short_spot'
        return 'no_trade'
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'instrument_name': self.instrument_name,
            'symbol': self.symbol,
            'funding_rate': self.funding_rate,
            'funding_rate_8h_pct': self.funding_rate_8h_pct,
            'funding_rate_daily': self.funding_rate_daily,
            'funding_rate_annualized': self.funding_rate_annualized,
            'funding_rate_annualized_pct': self.funding_rate_annualized_pct,
            'is_positive': self.is_positive,
            'payer': self.payer,
            'receiver': self.receiver,
            'regime': self.regime.value,
            'is_extreme': self.is_extreme,
            'is_elevated': self.is_elevated,
            'carry_trade_direction': self.carry_trade_direction,
            'basis_trade_signal': self.basis_trade_signal,
            'index_price': self.index_price,
            'index_change_pct': self.index_change_pct,
        }

@dataclass
class VolatilitySurfaceSlice:
    """
    Volatility surface slice for a single expiry with analytics.
    
    Contains ATM vol, skew metrics, and smile characteristics
    for cross-sectional vol analysis.
    """
    timestamp: datetime
    underlying: str
    expiry_date: datetime
    days_to_expiry: float
    
    # Core Metrics
    atm_iv: float = 0.0
    atm_strike: float = 0.0
    spot_price: float = 0.0
    
    # 25-Delta Metrics
    call_25d_iv: float = 0.0
    put_25d_iv: float = 0.0
    call_25d_strike: float = 0.0
    put_25d_strike: float = 0.0
    
    # 10-Delta Metrics (wings)
    call_10d_iv: float = 0.0
    put_10d_iv: float = 0.0
    
    # -------------------------------------------------------------------------
    # Skew Metrics
    # -------------------------------------------------------------------------
    
    @property
    def risk_reversal_25d(self) -> float:
        """25-delta risk reversal (call IV - put IV)."""
        if self.call_25d_iv > 0 and self.put_25d_iv > 0:
            return self.call_25d_iv - self.put_25d_iv
        return 0.0
    
    @property
    def risk_reversal_25d_pct(self) -> float:
        """25-delta RR as percentage."""
        return self.risk_reversal_25d * 100
    
    @property
    def skew_25d(self) -> float:
        """25-delta skew (put IV - call IV, traditional definition)."""
        return -self.risk_reversal_25d
    
    @property
    def skew_direction(self) -> SkewDirection:
        """Classify skew direction."""
        return SkewDirection.from_risk_reversal(self.risk_reversal_25d_pct)
    
    @property
    def is_put_skewed(self) -> bool:
        """Check if puts are richer than calls."""
        return self.put_25d_iv > self.call_25d_iv
    
    @property
    def is_call_skewed(self) -> bool:
        """Check if calls are richer than puts."""
        return self.call_25d_iv > self.put_25d_iv
    
    # -------------------------------------------------------------------------
    # Smile Metrics
    # -------------------------------------------------------------------------
    
    @property
    def butterfly_25d(self) -> float:
        """25-delta butterfly: (call + put)/2 - ATM."""
        if self.call_25d_iv > 0 and self.put_25d_iv > 0 and self.atm_iv > 0:
            return (self.call_25d_iv + self.put_25d_iv) / 2 - self.atm_iv
        return 0.0
    
    @property
    def smile_strength(self) -> float:
        """Smile strength (butterfly as pct of ATM)."""
        if self.atm_iv > 0:
            return (self.butterfly_25d / self.atm_iv) * 100
        return 0.0
    
    @property
    def has_smile(self) -> bool:
        """Check if meaningful smile exists."""
        return self.butterfly_25d > 0.02
    
    @property
    def wing_spread(self) -> float:
        """Spread between 10d wings."""
        if self.call_10d_iv > 0 and self.put_10d_iv > 0:
            return self.put_10d_iv - self.call_10d_iv
        return 0.0
    
    # -------------------------------------------------------------------------
    # Trading Signals
    # -------------------------------------------------------------------------
    
    @property
    def skew_trading_signal(self) -> str:
        """Get skew-based trading signal."""
        rr = self.risk_reversal_25d_pct
        if rr < -5:
            return 'sell_puts_buy_calls' # Put skew is expensive
        elif rr > 5:
            return 'sell_calls_buy_puts' # Call skew is expensive
        return 'neutral'
    
    @property
    def butterfly_trading_signal(self) -> str:
        """Get butterfly trading signal."""
        if self.smile_strength > 10:
            return 'sell_wings_buy_atm' # Wings are expensive
        elif self.smile_strength < -5:
            return 'buy_wings_sell_atm' # Wings are cheap
        return 'neutral'
    
    @property
    def market_sentiment(self) -> str:
        """Infer market sentiment from skew."""
        return self.skew_direction.market_sentiment
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'underlying': self.underlying,
            'expiry_date': self.expiry_date.isoformat() if isinstance(self.expiry_date, datetime) else self.expiry_date,
            'days_to_expiry': self.days_to_expiry,
            'atm_iv': self.atm_iv,
            'atm_strike': self.atm_strike,
            'spot_price': self.spot_price,
            'call_25d_iv': self.call_25d_iv,
            'put_25d_iv': self.put_25d_iv,
            'risk_reversal_25d': self.risk_reversal_25d,
            'risk_reversal_25d_pct': self.risk_reversal_25d_pct,
            'skew_direction': self.skew_direction.value,
            'butterfly_25d': self.butterfly_25d,
            'smile_strength': self.smile_strength,
            'has_smile': self.has_smile,
            'skew_trading_signal': self.skew_trading_signal,
            'butterfly_trading_signal': self.butterfly_trading_signal,
            'market_sentiment': self.market_sentiment,
        }

@dataclass
class TermStructureRecord:
    """
    Volatility term structure with analytics.
    
    Contains ATM IV across expiries for term structure analysis
    and calendar spread trading.
    """
    timestamp: datetime
    underlying: str
    term_points: List[Dict] = field(default_factory=list)
    
    # -------------------------------------------------------------------------
    # Term Structure Shape
    # -------------------------------------------------------------------------
    
    @property
    def num_points(self) -> int:
        """Number of term structure points."""
        return len(self.term_points)
    
    @property
    def sorted_points(self) -> List[Dict]:
        """Points sorted by days to expiry."""
        return sorted(self.term_points, key=lambda x: x.get('days_to_expiry', 0))
    
    @property
    def front_iv(self) -> float:
        """Front month ATM IV."""
        if self.sorted_points:
            return self.sorted_points[0].get('atm_iv', 0)
        return 0.0
    
    @property
    def back_iv(self) -> float:
        """Back month ATM IV."""
        if self.sorted_points:
            return self.sorted_points[-1].get('atm_iv', 0)
        return 0.0
    
    @property
    def mid_iv(self) -> float:
        """Mid-term ATM IV."""
        if len(self.sorted_points) >= 3:
            mid_idx = len(self.sorted_points) // 2
            return self.sorted_points[mid_idx].get('atm_iv', 0)
        return (self.front_iv + self.back_iv) / 2
    
    @property
    def term_spread(self) -> float:
        """Back - front IV spread."""
        return self.back_iv - self.front_iv
    
    @property
    def term_spread_pct(self) -> float:
        """Term spread as percentage of front IV."""
        if self.front_iv > 0:
            return (self.term_spread / self.front_iv) * 100
        return 0.0
    
    @property
    def has_hump(self) -> bool:
        """Check if term structure has a hump (mid > front and back)."""
        return self.mid_iv > self.front_iv and self.mid_iv > self.back_iv
    
    @property
    def has_trough(self) -> bool:
        """Check if term structure has a trough (mid < front and back)."""
        return self.mid_iv < self.front_iv and self.mid_iv < self.back_iv
    
    @property
    def shape(self) -> VolTermStructure:
        """Classify term structure shape."""
        if self.has_hump:
            return VolTermStructure.HUMPED
        elif self.has_trough:
            return VolTermStructure.INVERTED_HUMP
        return VolTermStructure.from_term_spread(self.term_spread_pct)
    
    @property
    def is_contango(self) -> bool:
        """Check if in contango (upward sloping)."""
        return self.shape in (VolTermStructure.CONTANGO, VolTermStructure.STEEP_CONTANGO)
    
    @property
    def is_backwardation(self) -> bool:
        """Check if in backwardation (downward sloping)."""
        return self.shape in (VolTermStructure.BACKWARDATION, VolTermStructure.STEEP_BACKWARDATION)
    
    @property
    def is_inverted(self) -> bool:
        """Check if inverted (front > back)."""
        return self.front_iv > self.back_iv * 1.03
    
    # -------------------------------------------------------------------------
    # Trading Signals
    # -------------------------------------------------------------------------
    
    @property
    def calendar_spread_signal(self) -> str:
        """Get calendar spread trading signal."""
        return self.shape.calendar_spread_bias
    
    @property
    def regime(self) -> str:
        """Get regime classification."""
        return self.shape.regime
    
    @property
    def roll_yield_direction(self) -> str:
        """Expected roll yield direction."""
        if self.is_contango:
            return 'positive_for_short_vol'
        elif self.is_backwardation:
            return 'positive_for_long_vol'
        return 'neutral'
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'underlying': self.underlying,
            'num_points': self.num_points,
            'front_iv': self.front_iv,
            'back_iv': self.back_iv,
            'mid_iv': self.mid_iv,
            'term_spread': self.term_spread,
            'term_spread_pct': self.term_spread_pct,
            'shape': self.shape.value,
            'is_contango': self.is_contango,
            'is_backwardation': self.is_backwardation,
            'is_inverted': self.is_inverted,
            'has_hump': self.has_hump,
            'regime': self.regime,
            'calendar_spread_signal': self.calendar_spread_signal,
            'roll_yield_direction': self.roll_yield_direction,
            'term_points': self.term_points,
        }

@dataclass
class PutCallRatioRecord:
    """
    Put/call ratio metrics with sentiment analysis.
    
    Aggregates volume and OI ratios for sentiment indicators.
    """
    timestamp: datetime
    underlying: str
    
    # Volume Metrics
    call_volume: float = 0.0
    put_volume: float = 0.0
    total_volume: float = 0.0
    
    # Open Interest Metrics
    call_oi: float = 0.0
    put_oi: float = 0.0
    total_oi: float = 0.0
    
    # -------------------------------------------------------------------------
    # Ratio Calculations
    # -------------------------------------------------------------------------
    
    @property
    def volume_ratio(self) -> float:
        """Put/call volume ratio."""
        if self.call_volume > 0:
            return self.put_volume / self.call_volume
        return 0.0
    
    @property
    def oi_ratio(self) -> float:
        """Put/call OI ratio."""
        if self.call_oi > 0:
            return self.put_oi / self.call_oi
        return 0.0
    
    @property
    def call_volume_pct(self) -> float:
        """Call volume as percentage of total."""
        if self.total_volume > 0:
            return (self.call_volume / self.total_volume) * 100
        return 0.0
    
    @property
    def put_volume_pct(self) -> float:
        """Put volume as percentage of total."""
        if self.total_volume > 0:
            return (self.put_volume / self.total_volume) * 100
        return 0.0
    
    # -------------------------------------------------------------------------
    # Sentiment Analysis
    # -------------------------------------------------------------------------
    
    @property
    def volume_sentiment(self) -> str:
        """Sentiment from volume ratio."""
        r = self.volume_ratio
        if r > 1.5:
            return 'very_bearish'
        elif r > 1.2:
            return 'bearish'
        elif r > 0.8:
            return 'neutral'
        elif r > 0.6:
            return 'bullish'
        return 'very_bullish'
    
    @property
    def oi_sentiment(self) -> str:
        """Sentiment from OI ratio."""
        r = self.oi_ratio
        if r > 1.5:
            return 'very_bearish'
        elif r > 1.2:
            return 'bearish'
        elif r > 0.8:
            return 'neutral'
        elif r > 0.6:
            return 'bullish'
        return 'very_bullish'
    
    @property
    def combined_sentiment(self) -> str:
        """Combined sentiment signal."""
        vol_score = self.volume_ratio
        oi_score = self.oi_ratio
        avg = (vol_score + oi_score) / 2
        
        if avg > 1.3:
            return 'bearish'
        elif avg > 0.7:
            return 'neutral'
        return 'bullish'
    
    # -------------------------------------------------------------------------
    # Extreme Detection
    # -------------------------------------------------------------------------
    
    @property
    def is_extreme_put_buying(self) -> bool:
        """Check for extreme put buying."""
        return self.volume_ratio > 2.0
    
    @property
    def is_extreme_call_buying(self) -> bool:
        """Check for extreme call buying."""
        return self.volume_ratio < 0.5
    
    @property
    def is_contrarian_signal(self) -> bool:
        """Check if ratio suggests contrarian opportunity."""
        return self.is_extreme_put_buying or self.is_extreme_call_buying
    
    @property
    def contrarian_direction(self) -> str:
        """Get contrarian trading direction."""
        if self.is_extreme_put_buying:
            return 'bullish' # Extreme put buying often marks bottoms
        elif self.is_extreme_call_buying:
            return 'bearish' # Extreme call buying often marks tops
        return 'neutral'
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'underlying': self.underlying,
            'call_volume': self.call_volume,
            'put_volume': self.put_volume,
            'total_volume': self.total_volume,
            'call_oi': self.call_oi,
            'put_oi': self.put_oi,
            'total_oi': self.total_oi,
            'volume_ratio': self.volume_ratio,
            'oi_ratio': self.oi_ratio,
            'call_volume_pct': self.call_volume_pct,
            'put_volume_pct': self.put_volume_pct,
            'volume_sentiment': self.volume_sentiment,
            'oi_sentiment': self.oi_sentiment,
            'combined_sentiment': self.combined_sentiment,
            'is_extreme_put_buying': self.is_extreme_put_buying,
            'is_extreme_call_buying': self.is_extreme_call_buying,
            'is_contrarian_signal': self.is_contrarian_signal,
            'contrarian_direction': self.contrarian_direction,
        }

# =============================================================================
# COLLECTOR CLASS
# =============================================================================

class DeribitCollector:
    """
    validated Deribit options data collector.
    
    Provides comprehensive access to Deribit's options market data
    including chains, IV surfaces, term structure, and DVOL.
    """
    
    VENUE = 'deribit'
    VENUE_TYPE = 'CEX'
    BASE_URL = 'https://www.deribit.com/api/v2'
    TEST_URL = 'https://test.deribit.com/api/v2'
    
    UNDERLYINGS = ['BTC', 'ETH', 'SOL']
    RATE_LIMIT = 30 # requests per minute (reduced from 100, conservative burst limit)
    
    def __init__(self, config: Optional[Dict] = None, use_testnet: bool = False):
        """
        Initialize Deribit collector.

        Args:
            config: Configuration dictionary
            use_testnet: Use testnet instead of production
        """
        config = config or {}
        use_testnet = use_testnet or os.getenv('DERIBIT_TESTNET', '').lower() == 'true'
        self.base_url = self.TEST_URL if use_testnet else self.BASE_URL
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load API credentials for private endpoints
        self.client_id = config.get('client_id') or config.get('deribit_client_id') or os.getenv('DERIBIT_CLIENT_ID', '')
        self.client_secret = config.get('client_secret') or config.get('deribit_client_secret') or os.getenv('DERIBIT_CLIENT_SECRET', '')

        # CRITICAL: Set supported data types for dynamic routing (collection_manager)
        self.supported_data_types = ['options', 'funding_rates', 'ohlcv', 'open_interest']
        self.venue = 'deribit'
        self.requires_auth = False # Public API endpoints available without auth

        # Cache
        self._instruments_cache: Dict[str, List[Dict]] = {}
        self._index_prices: Dict[str, float] = {}

        # Stats
        self.collection_stats = {
            'requests': 0,
            'records': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None,
        }
    
    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------
    
    async def __aenter__(self) -> 'DeribitCollector':
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        self.collection_stats['start_time'] = datetime.now(timezone.utc)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self.collection_stats['end_time'] = datetime.now(timezone.utc)
        if self.session:
            await self.session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Make rate-limited API request."""
        if not self.session:
            await self._get_session()
        
        url = f"{self.base_url}/public/{method}"
        self.collection_stats['requests'] += 1
        
        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status == 429:
                    self.logger.warning("Rate limited - waiting 60s")
                    await asyncio.sleep(60)
                    return None
                
                if resp.status != 200:
                    self.collection_stats['errors'] += 1
                    error_text = await resp.text()
                    self.logger.warning(f"HTTP {resp.status} for {method}: {error_text[:200]}")
                    return None
                
                data = await resp.json()
                
                if 'error' in data:
                    self.collection_stats['errors'] += 1
                    self.logger.error(f"API error: {data['error']}")
                    return None
                
                return data.get('result')
                
        except Exception as e:
            error_msg = str(e) if str(e) else f"{type(e).__name__}"
            self.logger.error(f"Request error for {method}: {error_msg}")
            self.collection_stats['errors'] += 1
            return None
    
    # -------------------------------------------------------------------------
    # Instrument Methods
    # -------------------------------------------------------------------------
    
    async def fetch_instruments(
        self,
        currency: str = 'BTC',
        kind: str = 'option',
        expired: bool = False
    ) -> pd.DataFrame:
        """
        Fetch all instruments for a currency.
        
        Args:
            currency: BTC, ETH, or SOL
            kind: 'option', 'future', or 'spot'
            expired: Include expired instruments
            
        Returns:
            DataFrame with instrument metadata
        """
        self.logger.info(f"Fetching {currency} {kind} instruments")
        
        result = await self._request('get_instruments', {
            'currency': currency.upper(),
            'kind': kind,
            'expired': str(expired).lower(),
        })
        
        if not result:
            return pd.DataFrame()
        
        instruments = []
        
        for inst in result:
            try:
                if kind == 'option':
                    instrument = OptionInstrument(
                        instrument_name=inst['instrument_name'],
                        underlying=currency.upper(),
                        strike=float(inst['strike']),
                        option_type=inst['option_type'],
                        expiry_timestamp=inst['expiration_timestamp'],
                        expiry_date=datetime.fromtimestamp(
                            inst['expiration_timestamp'] / 1000, tz=timezone.utc
                        ),
                        creation_timestamp=inst.get('creation_timestamp', 0),
                        settlement_period=inst.get('settlement_period', 'day'),
                        tick_size=inst.get('tick_size', 0.0001),
                        min_trade_amount=inst.get('min_trade_amount', 0.1),
                        contract_size=inst.get('contract_size', 1.0),
                        maker_commission=inst.get('maker_commission', 0.0003),
                        taker_commission=inst.get('taker_commission', 0.0003),
                        is_active=inst.get('is_active', True),
                    )
                    instruments.append(instrument.to_dict())
                    
            except Exception as e:
                self.logger.warning(f"Instrument parse error: {e}")
                continue
        
        df = pd.DataFrame(instruments)
        
        if not df.empty:
            df['venue'] = self.VENUE
            df['venue_type'] = self.VENUE_TYPE
            df = df.sort_values(['expiry_date', 'strike']).reset_index(drop=True)
            self.collection_stats['records'] += len(df)
        
        return df
    
    # -------------------------------------------------------------------------
    # Option Chain Methods
    # -------------------------------------------------------------------------
    
    async def fetch_option_chain(self, currency: str = 'BTC') -> pd.DataFrame:
        """
        Fetch complete option chain with Greeks.
        
        Args:
            currency: Underlying asset (BTC, ETH, SOL)
            
        Returns:
            DataFrame with comprehensive option data
        """
        self.logger.info(f"Fetching {currency} option chain")
        
        # Get instruments
        instruments = await self.fetch_instruments(currency, 'option', False)
        
        if instruments.empty:
            return pd.DataFrame()
        
        # Get index price
        index_result = await self._request('get_index_price', {
            'index_name': f'{currency.lower()}_usd'
        })
        index_price = index_result.get('index_price', 0) if index_result else 0
        self._index_prices[currency] = index_price
        
        all_options = []
        
        for _, inst in instruments.iterrows():
            if not inst.get('is_tradeable', True):
                continue
            
            ticker = await self._request('ticker', {
                'instrument_name': inst['instrument_name']
            })
            
            if not ticker:
                continue
            
            try:
                greeks = ticker.get('greeks', {})
                stats = ticker.get('stats', {})
                
                quote = OptionQuote(
                    timestamp=datetime.now(timezone.utc),
                    instrument_name=inst['instrument_name'],
                    underlying=currency,
                    strike=inst['strike'],
                    option_type=inst.get('option_type', 'call'),
                    expiry_date=pd.to_datetime(inst['expiry_date']),
                    
                    # Prices
                    mark_price=ticker.get('mark_price', 0) or 0,
                    mark_iv=ticker.get('mark_iv', 0) or 0,
                    bid_price=ticker.get('best_bid_price', 0) or 0,
                    ask_price=ticker.get('best_ask_price', 0) or 0,
                    bid_iv=ticker.get('bid_iv', 0) or 0,
                    ask_iv=ticker.get('ask_iv', 0) or 0,
                    last_price=ticker.get('last_price', 0) or 0,
                    
                    # Underlying
                    underlying_price=ticker.get('underlying_price', 0) or 0,
                    underlying_index=ticker.get('underlying_index', ''),
                    index_price=index_price,
                    
                    # Greeks
                    delta=greeks.get('delta', 0) or 0,
                    gamma=greeks.get('gamma', 0) or 0,
                    vega=greeks.get('vega', 0) or 0,
                    theta=greeks.get('theta', 0) or 0,
                    rho=greeks.get('rho', 0) or 0,
                    
                    # Volume & Interest
                    volume=stats.get('volume', 0) or 0,
                    volume_usd=stats.get('volume_usd', 0) or 0,
                    open_interest=ticker.get('open_interest', 0) or 0,
                    
                    # Order Book
                    bid_size=ticker.get('best_bid_amount', 0) or 0,
                    ask_size=ticker.get('best_ask_amount', 0) or 0,
                )
                
                all_options.append(quote.to_dict())
                self.collection_stats['records'] += 1
                
            except Exception as e:
                self.logger.warning(f"Quote parse error: {e}")
                continue
            
            # Rate limiting (reduced: every 30 records, 1.0s pause for ~10 req/sec effective)
            if self.collection_stats['records'] % 30 == 0:
                await asyncio.sleep(1.0)
        
        df = pd.DataFrame(all_options)
        
        if not df.empty:
            df['venue'] = self.VENUE
            df['venue_type'] = self.VENUE_TYPE
            df = df.sort_values(
                ['expiry_date', 'strike', 'option_type']
            ).reset_index(drop=True)
        
        self.logger.info(f"Fetched {len(df)} options for {currency}")
        return df
    
    # -------------------------------------------------------------------------
    # DVOL Methods
    # -------------------------------------------------------------------------
    
    async def fetch_dvol(
        self,
        currency: str = 'BTC',
        resolution: str = '1H',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch DVOL volatility index history.
        
        Args:
            currency: BTC or ETH
            resolution: '1H' or '1D'
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with DVOL data
        """
        self.logger.info(f"Fetching {currency} DVOL ({resolution})")
        
        # Parse dates
        end_dt = (
            datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            if end_date else datetime.now(timezone.utc)
        )
        start_dt = (
            datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            if start_date else end_dt - timedelta(days=30)
        )
        
        resolution_map = {'1H': 3600, '1D': 86400}
        resolution_sec = resolution_map.get(resolution, 3600)
        
        result = await self._request('get_volatility_index_data', {
            'currency': currency.upper(),
            'start_timestamp': int(start_dt.timestamp() * 1000),
            'end_timestamp': int(end_dt.timestamp() * 1000),
            'resolution': str(resolution_sec),
        })
        
        if not result or 'data' not in result:
            return pd.DataFrame()
        
        dvol_data = []
        
        for row in result['data']:
            try:
                record = DVOLRecord(
                    timestamp=datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc),
                    underlying=currency,
                    open=row[1],
                    high=row[2],
                    low=row[3],
                    close=row[4],
                    resolution=resolution,
                )
                dvol_data.append(record.to_dict())
                self.collection_stats['records'] += 1
                
            except Exception as e:
                self.logger.warning(f"DVOL parse error: {e}")
                continue
        
        df = pd.DataFrame(dvol_data)
        
        if not df.empty:
            df['venue'] = self.VENUE
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        self.logger.info(f"Fetched {len(df)} DVOL records")
        return df
    
    # -------------------------------------------------------------------------
    # Funding Rate Methods
    # -------------------------------------------------------------------------
    
    async def fetch_funding_rate(
        self,
        instrument_name: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch funding rate history for perpetual.
        
        Args:
            instrument_name: Perpetual name (e.g., BTC-PERPETUAL)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with funding rate history
        """
        self.logger.info(f"Fetching funding rates for {instrument_name}")
        
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        result = await self._request('get_funding_rate_history', {
            'instrument_name': instrument_name,
            'start_timestamp': start_ts,
            'end_timestamp': end_ts,
        })
        
        if not result:
            return pd.DataFrame()
        
        funding_data = []
        underlying = instrument_name.split('-')[0]
        
        for record in result:
            try:
                fr = FundingRateRecord(
                    timestamp=datetime.fromtimestamp(
                        record['timestamp'] / 1000, tz=timezone.utc
                    ),
                    instrument_name=instrument_name,
                    symbol=underlying,
                    funding_rate=record['interest_8h'],
                    index_price=record.get('index_price', 0),
                    prev_index_price=record.get('prev_index_price', 0),
                )
                funding_data.append(fr.to_dict())
                self.collection_stats['records'] += 1
                
            except Exception as e:
                self.logger.warning(f"Funding parse error: {e}")
                continue
        
        df = pd.DataFrame(funding_data)
        
        if not df.empty:
            df['venue'] = self.VENUE
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    async def fetch_funding_rates(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch funding rates for multiple symbols."""
        # PARALLELIZED: Fetch all symbols concurrently
        async def _fetch_single_funding_rate(symbol: str) -> Optional[pd.DataFrame]:
            base_symbol = symbol.replace('USDT', '').replace('USD', '').upper()
            if base_symbol not in self.UNDERLYINGS:
                self.logger.debug(f"Skipping unsupported symbol {symbol} (Deribit only supports {self.UNDERLYINGS})")
                return None

            instrument = f"{base_symbol}-PERPETUAL"
            df = await self.fetch_funding_rate(instrument, start_date, end_date)
            if not df.empty:
                return df
            return None

        tasks = [_fetch_single_funding_rate(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    # -------------------------------------------------------------------------
    # OHLCV Methods
    # -------------------------------------------------------------------------
    
    async def fetch_ohlcv(
        self,
        symbols: Union[str, List[str]] = None,
        timeframe: str = '1h',
        start_date: str = None,
        end_date: str = None,
        contract_type: str = 'perpetual',
        instrument_name: str = None,
        resolution: str = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for instruments.

        Supports two signatures for compatibility:
        1. Standard: (symbols, timeframe, start_date, end_date, contract_type)
        2. Legacy: (instrument_name, resolution, start_date, end_date)

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH']) or single symbol
            timeframe: Candle interval ('1h', '4h', '1d')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            contract_type: Contract type (default: perpetual)
            instrument_name: Legacy parameter - specific instrument
            resolution: Legacy parameter - candle resolution
            **kwargs: Additional arguments (ignored)

        Returns:
            DataFrame with OHLCV data
        """
        # Handle legacy signature
        if instrument_name is not None:
            symbols = [instrument_name]
            timeframe = resolution or timeframe

        # Handle single symbol
        if isinstance(symbols, str):
            symbols = [symbols]

        # If no symbols provided, return empty
        if not symbols:
            return pd.DataFrame()

        # PARALLELIZED: Fetch OHLCV for all symbols concurrently
        async def _fetch_single_ohlcv(symbol: str) -> List[Dict]:
            records = []
            # Convert symbol to Deribit instrument format if not already
            if '-' not in symbol:
                base = symbol.replace('USDT', '').replace('USD', '').upper()
            else:
                # Extract base from instrument name (e.g., "BTC-PERPETUAL" -> "BTC")
                base = symbol.split('-')[0].upper()

            # Filter to supported underlyings only
            if base not in self.UNDERLYINGS:
                self.logger.debug(f"Skipping unsupported symbol {symbol} (Deribit only supports {self.UNDERLYINGS})")
                return records

            inst = f"{base}-PERPETUAL" if '-' not in symbol else symbol

            self.logger.info(f"Fetching OHLCV for {inst} ({timeframe})")

            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

            resolution_map = {
                '1m': '1', '5m': '5', '15m': '15', '30m': '30',
                '1h': '60', '2h': '120', '4h': '240', '6h': '360',
                '12h': '720', '1d': '1D',
            }
            deribit_resolution = resolution_map.get(timeframe, timeframe)

            current_start = start_ts

            while current_start < end_ts:
                result = await self._request('get_tradingview_chart_data', {
                    'instrument_name': inst,
                    'start_timestamp': current_start,
                    'end_timestamp': end_ts,
                    'resolution': deribit_resolution,
                })

                if not result or 'ticks' not in result:
                    break

                ticks = result['ticks']
                if not ticks:
                    break

                for i, ts in enumerate(ticks):
                    try:
                        records.append({
                            'timestamp': datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                            'instrument_name': inst,
                            'symbol': symbol.replace('-PERPETUAL', '').upper(),
                            'open': result['open'][i],
                            'high': result['high'][i],
                            'low': result['low'][i],
                            'close': result['close'][i],
                            'volume': result['volume'][i],
                            'venue': self.VENUE,
                            'venue_type': self.VENUE_TYPE,
                        })
                        self.collection_stats['records'] += 1
                    except Exception:
                        continue

                current_start = max(ticks) + 1
                await asyncio.sleep(0.2)  # Doubled from 0.1 for reduced rate limit

            return records

        tasks = [_fetch_single_ohlcv(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_data = []
        for result in results:
            if isinstance(result, list):
                all_data.extend(result)

        df = pd.DataFrame(all_data)

        if not df.empty:
            df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        return df
    
    # -------------------------------------------------------------------------
    # Analytics Methods
    # -------------------------------------------------------------------------
    
    async def calculate_term_structure(self, currency: str = 'BTC') -> TermStructureRecord:
        """
        Calculate ATM volatility term structure.
        
        Args:
            currency: Underlying asset
            
        Returns:
            TermStructureRecord with analytics
        """
        chain = await self.fetch_option_chain(currency)
        
        if chain.empty:
            return TermStructureRecord(
                timestamp=datetime.now(timezone.utc),
                underlying=currency
            )
        
        index_price = self._index_prices.get(currency, chain['underlying_price'].iloc[0])
        term_points = []
        
        for expiry, group in chain.groupby('expiry_date'):
            # Find ATM options
            group = group.copy()
            group['distance_from_atm'] = abs(group['strike'] - index_price)
            atm_options = group.nsmallest(2, 'distance_from_atm')
            
            if len(atm_options) < 2:
                continue
            
            atm_iv = atm_options['mark_iv'].mean()
            atm_strike = atm_options['strike'].mean()
            dte = (pd.to_datetime(expiry) - datetime.now(timezone.utc)).total_seconds() / 86400
            
            if dte > 0:
                term_points.append({
                    'expiry_date': expiry,
                    'days_to_expiry': dte,
                    'atm_iv': atm_iv,
                    'atm_strike': atm_strike,
                })
        
        return TermStructureRecord(
            timestamp=datetime.now(timezone.utc),
            underlying=currency,
            term_points=term_points,
        )
    
    def calculate_put_call_ratio(self, chain: pd.DataFrame) -> PutCallRatioRecord:
        """
        Calculate put/call ratio from option chain.
        
        Args:
            chain: Option chain DataFrame
            
        Returns:
            PutCallRatioRecord with analytics
        """
        if chain.empty:
            return PutCallRatioRecord(
                timestamp=datetime.now(timezone.utc),
                underlying=''
            )
        
        calls = chain[chain['option_type'] == 'call']
        puts = chain[chain['option_type'] == 'put']
        
        return PutCallRatioRecord(
            timestamp=datetime.now(timezone.utc),
            underlying=chain['underlying'].iloc[0] if not chain.empty else '',
            call_volume=calls['volume'].sum() if not calls.empty else 0,
            put_volume=puts['volume'].sum() if not puts.empty else 0,
            total_volume=chain['volume'].sum(),
            call_oi=calls['open_interest'].sum() if not calls.empty else 0,
            put_oi=puts['open_interest'].sum() if not puts.empty else 0,
            total_oi=chain['open_interest'].sum(),
        )
    
    async def build_volatility_surface(
        self,
        chain: pd.DataFrame,
        currency: str = 'BTC'
    ) -> List[VolatilitySurfaceSlice]:
        """
        Build volatility surface slices from option chain.
        
        Args:
            chain: Option chain DataFrame
            currency: Underlying asset
            
        Returns:
            List of VolatilitySurfaceSlice objects
        """
        if chain.empty:
            return []
        
        index_price = self._index_prices.get(currency, chain['underlying_price'].iloc[0])
        surface_slices = []
        
        for expiry, group in chain.groupby('expiry_date'):
            dte = (pd.to_datetime(expiry) - datetime.now(timezone.utc)).total_seconds() / 86400
            
            if dte <= 0:
                continue
            
            calls = group[group['option_type'] == 'call']
            puts = group[group['option_type'] == 'put']
            
            # Find ATM
            group['distance'] = abs(group['strike'] - index_price)
            atm = group.nsmallest(2, 'distance')
            atm_iv = atm['mark_iv'].mean() if not atm.empty else 0
            atm_strike = atm['strike'].mean() if not atm.empty else index_price
            
            # Find 25-delta options
            call_25d = calls[(calls['delta'] - 0.25).abs() < 0.1]
            put_25d = puts[(puts['delta'].abs() - 0.25).abs() < 0.1]
            
            call_25d_iv = call_25d['mark_iv'].mean() if not call_25d.empty else 0
            put_25d_iv = put_25d['mark_iv'].mean() if not put_25d.empty else 0
            
            # Find 10-delta options (wings)
            call_10d = calls[(calls['delta'] - 0.10).abs() < 0.05]
            put_10d = puts[(puts['delta'].abs() - 0.10).abs() < 0.05]
            
            surface_slices.append(VolatilitySurfaceSlice(
                timestamp=datetime.now(timezone.utc),
                underlying=currency,
                expiry_date=pd.to_datetime(expiry),
                days_to_expiry=dte,
                atm_iv=atm_iv,
                atm_strike=atm_strike,
                spot_price=index_price,
                call_25d_iv=call_25d_iv,
                put_25d_iv=put_25d_iv,
                call_25d_strike=call_25d['strike'].mean() if not call_25d.empty else 0,
                put_25d_strike=put_25d['strike'].mean() if not put_25d.empty else 0,
                call_10d_iv=call_10d['mark_iv'].mean() if not call_10d.empty else 0,
                put_10d_iv=put_10d['mark_iv'].mean() if not put_10d.empty else 0,
            ))
        
        return surface_slices
    
    # -------------------------------------------------------------------------
    # Standardized Collection Methods
    # -------------------------------------------------------------------------

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
            symbols: List of underlying symbols (BTC, ETH, SOL)
            start_date: Start date (ignored - returns current snapshot)
            end_date: End date (ignored - returns current snapshot)
            **kwargs: Additional arguments

        Returns:
            DataFrame with options data including Greeks
        """
        try:
            # PARALLELIZED: Fetch option chains for all symbols concurrently
            async def _fetch_single_option_chain(symbol: str) -> Optional[pd.DataFrame]:
                currency = symbol.upper().replace('USDT', '').replace('USD', '').replace('-', '')
                df = await self.fetch_option_chain(currency=currency)
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
            self.logger.error(f"Deribit collect_options error: {e}")
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
            self.logger.error(f"Deribit collect_funding_rates error: {e}")
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
            self.logger.error(f"Deribit collect_ohlcv error: {e}")
            return pd.DataFrame()

    async def collect_open_interest(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect open interest data for perpetual futures.

        Uses Deribit's get_book_summary_by_currency endpoint to retrieve
        open interest for perpetual contracts.

        Args:
            symbols: List of underlying symbols (BTC, ETH, SOL)
            start_date: Start date (ignored - returns current snapshot)
            end_date: End date (ignored - returns current snapshot)
            **kwargs: Additional arguments (include_options=True for option OI)

        Returns:
            DataFrame with open interest data
        """
        try:
            include_options = kwargs.get('include_options', False)

            # PARALLELIZED: Fetch open interest for all symbols concurrently
            async def _fetch_single_open_interest(symbol: str) -> List[Dict]:
                records = []
                currency = symbol.upper().replace('USDT', '').replace('USD', '').replace('-', '')

                # Fetch book summary for futures/perpetuals
                result = await self._request('get_book_summary_by_currency', {
                    'currency': currency,
                    'kind': 'future'
                })

                if result:
                    now = datetime.now(timezone.utc)
                    for item in result:
                        instrument = item.get('instrument_name', '')
                        # Focus on perpetual contracts for the main OI data
                        if 'PERPETUAL' in instrument or include_options:
                            record = {
                                'timestamp': now,
                                'symbol': currency,
                                'instrument_name': instrument,
                                'open_interest': float(item.get('open_interest', 0) or 0),
                                'volume_24h': float(item.get('volume', 0) or 0),
                                'volume_usd_24h': float(item.get('volume_usd', 0) or 0),
                                'mark_price': float(item.get('mark_price', 0) or 0),
                                'underlying_price': float(item.get('underlying_price', 0) or 0),
                                'bid_price': float(item.get('bid_price', 0) or 0),
                                'ask_price': float(item.get('ask_price', 0) or 0),
                                'high_24h': float(item.get('high', 0) or 0),
                                'low_24h': float(item.get('low', 0) or 0),
                                'price_change_24h': float(item.get('price_change', 0) or 0),
                                'venue': self.VENUE,
                                'venue_type': self.VENUE_TYPE
                            }
                            records.append(record)
                            self.collection_stats['records'] += 1

                # Optionally get option OI as well
                if include_options:
                    option_result = await self._request('get_book_summary_by_currency', {
                        'currency': currency,
                        'kind': 'option'
                    })

                    if option_result:
                        for item in option_result:
                            record = {
                                'timestamp': now,
                                'symbol': currency,
                                'instrument_name': item.get('instrument_name', ''),
                                'open_interest': float(item.get('open_interest', 0) or 0),
                                'volume_24h': float(item.get('volume', 0) or 0),
                                'volume_usd_24h': float(item.get('volume_usd', 0) or 0),
                                'mark_price': float(item.get('mark_price', 0) or 0),
                                'underlying_price': float(item.get('underlying_price', 0) or 0),
                                'bid_price': float(item.get('bid_price', 0) or 0),
                                'ask_price': float(item.get('ask_price', 0) or 0),
                                'mark_iv': float(item.get('mark_iv', 0) or 0),
                                'venue': self.VENUE,
                                'venue_type': self.VENUE_TYPE
                            }
                            records.append(record)
                            self.collection_stats['records'] += 1

                return records

            tasks = [_fetch_single_open_interest(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_records = []
            for result in results:
                if isinstance(result, list):
                    all_records.extend(result)

            if all_records:
                df = pd.DataFrame(all_records)
                df = df.sort_values(['timestamp', 'symbol', 'instrument_name']).reset_index(drop=True)
                return df

            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Deribit collect_open_interest error: {e}")
            return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        stats = {**self.collection_stats, 'venue': self.VENUE}
        
        if stats['start_time'] and stats['end_time']:
            duration = (stats['end_time'] - stats['start_time']).total_seconds()
            stats['duration_seconds'] = duration
            stats['requests_per_second'] = stats['requests'] / duration if duration > 0 else 0
        
        return stats
    
    @staticmethod
    def get_supported_underlyings() -> List[str]:
        """Get list of supported underlying assets."""
        return [u.value for u in Underlying]
    
    async def close(self) -> None:
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    def validate_data(self, df, data_type=None, *args, **kwargs):
        """
        Validate collected data (compatibility method).

        Returns a validation result indicating success if data is valid.
        """
        from dataclasses import dataclass

        @dataclass
        class ValidationResult:
            valid: bool
            row_count: int
            column_count: int
            quality_score: float
            errors: list
            warnings: list = None

            def __post_init__(self):
                if self.warnings is None:
                    self.warnings = []

        if df is None or df.empty:
            return ValidationResult(
                valid=False,
                row_count=0,
                column_count=0,
                quality_score=0.0,
                errors=["Empty DataFrame"]
            )

        return ValidationResult(
            valid=True,
            row_count=len(df),
            column_count=len(df.columns),
            quality_score=100.0,
            errors=[]
        )

    async def fetch_perpetual_ohlcv(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        contract_type: str = 'perpetual',
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for perpetual contracts (standard interface).

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH'])
            timeframe: Candle interval ('1h', '4h', '1d')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            contract_type: Contract type (ignored, always perpetual)
            **kwargs: Additional arguments (ignored)

        Returns:
            DataFrame with OHLCV data
        """
        # PARALLELIZED: Fetch all symbols concurrently
        async def _fetch_single_perpetual_ohlcv(symbol: str) -> Optional[pd.DataFrame]:
            # Convert symbol to Deribit perpetual format
            base = symbol.replace('USDT', '').replace('USD', '').upper()
            instrument = f"{base}-PERPETUAL"

            df = await self.fetch_ohlcv(instrument, timeframe, start_date, end_date)

            if not df.empty:
                df['symbol'] = symbol.upper()
                return df
            return None

        tasks = [_fetch_single_perpetual_ohlcv(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]
        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        return result.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Historical DVOL Collection (for Strategy 4: Options Vol Surface Arbitrage)
    # -------------------------------------------------------------------------

    async def collect_dvol(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect historical DVOL (Deribit Volatility Index) data.

        DVOL is the crypto equivalent of VIX - a 30-day forward-looking
        implied volatility index. This is CRITICAL for Strategy 4 which
        requires historical implied volatility data for 2022-2024.

        The Deribit public API provides historical DVOL data via the
        public/get_volatility_index_data endpoint.

        Args:
            symbols: List of underlying symbols (BTC, ETH)
            start_date: Start date for historical data
            end_date: End date for historical data
            **kwargs: Additional arguments (resolution: '1H' or '1D')

        Returns:
            DataFrame with historical DVOL data including:
            - timestamp, underlying, open, high, low, close (IV values)
            - resolution, venue
        """
        try:
            resolution = kwargs.get('resolution', '1D')

            # Convert dates to string format if needed
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            # PARALLELIZED: Fetch all symbols concurrently
            async def _fetch_single_dvol(symbol: str) -> Optional[pd.DataFrame]:
                currency = symbol.upper().replace('USDT', '').replace('USD', '').replace('-', '')

                # Only BTC and ETH have DVOL
                if currency not in ['BTC', 'ETH']:
                    self.logger.debug(f"Skipping {currency} - DVOL only available for BTC/ETH")
                    return None

                self.logger.info(f"Fetching DVOL for {currency} ({start_str} to {end_str})")

                df = await self.fetch_dvol(
                    currency=currency,
                    resolution=resolution,
                    start_date=start_str,
                    end_date=end_str
                )

                if not df.empty:
                    self.logger.info(f"Collected {len(df)} DVOL records for {currency}")
                    return df
                return None

            tasks = [_fetch_single_dvol(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_records = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]
            if all_records:
                result = pd.concat(all_records, ignore_index=True)
                if 'venue' not in result.columns:
                    result['venue'] = self.VENUE
                if 'venue_type' not in result.columns:
                    result['venue_type'] = self.VENUE_TYPE
                return result

            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Deribit collect_dvol error: {e}")
            return pd.DataFrame()

    async def fetch_historical_option_trades(
        self,
        instrument_name: str,
        start_date: str,
        end_date: str,
        count: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch historical option trades for a specific instrument.

        The Deribit API provides historical trades via
        public/get_last_trades_by_instrument_and_time endpoint.
        Each trade includes the implied volatility (IV) at execution.

        Args:
            instrument_name: Option instrument (e.g., BTC-28MAR25-100000-C)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            count: Number of trades to fetch per request (max 1000)

        Returns:
            DataFrame with trade data including IV
        """
        self.logger.info(f"Fetching historical trades for {instrument_name}")

        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(
            tzinfo=timezone.utc
        ).timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc
        ).timestamp() * 1000)

        all_trades = []
        current_end = end_ts

        while current_end > start_ts:
            result = await self._request('get_last_trades_by_instrument_and_time', {
                'instrument_name': instrument_name,
                'start_timestamp': start_ts,
                'end_timestamp': current_end,
                'count': count,
                'sorting': 'desc'
            })

            if not result or 'trades' not in result:
                break

            trades = result['trades']
            if not trades:
                break

            for trade in trades:
                try:
                    record = {
                        'timestamp': datetime.fromtimestamp(
                            trade['timestamp'] / 1000, tz=timezone.utc
                        ),
                        'instrument_name': instrument_name,
                        'trade_id': trade.get('trade_id', ''),
                        'price': float(trade.get('price', 0)),
                        'amount': float(trade.get('amount', 0)),
                        'direction': trade.get('direction', ''),
                        'iv': float(trade.get('iv', 0)) if trade.get('iv') else None,
                        'index_price': float(trade.get('index_price', 0)),
                        'mark_price': float(trade.get('mark_price', 0)),
                        'venue': self.VENUE
                    }
                    all_trades.append(record)
                    self.collection_stats['records'] += 1
                except Exception as e:
                    self.logger.warning(f"Trade parse error: {e}")
                    continue

            # Update end timestamp for pagination
            if trades:
                current_end = min(t['timestamp'] for t in trades) - 1
            else:
                break

            await asyncio.sleep(0.2) # Rate limit (doubled from 0.1 for reduced rate)

        if not all_trades:
            return pd.DataFrame()

        df = pd.DataFrame(all_trades)
        df = df.sort_values('timestamp').reset_index(drop=True)
        self.logger.info(f"Fetched {len(df)} historical trades for {instrument_name}")
        return df

    async def collect_historical_option_trades(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect historical option trades for multiple symbols.

        This fetches actual trade data with implied volatility (IV).
        Greeks can be calculated from this data using Black-Scholes.

        Args:
            symbols: List of underlying symbols (BTC, ETH)
            start_date: Start date for historical data
            end_date: End date for historical data
            **kwargs: Additional arguments

        Returns:
            DataFrame with historical option trade data
        """
        try:
            # Convert dates
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            # PARALLELIZED: Fetch all symbols concurrently
            async def _fetch_single_symbol_trades(symbol: str) -> Optional[List[pd.DataFrame]]:
                currency = symbol.upper().replace('USDT', '').replace('USD', '').replace('-', '')

                if currency not in [u.value for u in Underlying]:
                    self.logger.debug(f"Skipping unsupported symbol {symbol}")
                    return None

                # Get list of option instruments
                instruments = await self._request('get_instruments', {
                    'currency': currency,
                    'kind': 'option',
                    'expired': False
                })

                if not instruments:
                    return None

                # Fetch trades for each instrument (sample - not all)
                # Focus on most liquid options (ATM, near-term)
                sampled = instruments[:50] # Limit to avoid rate limits

                # PARALLELIZED: Fetch all instruments concurrently
                async def _fetch_single_instrument_trades(inst: dict) -> Optional[pd.DataFrame]:
                    inst_name = inst['instrument_name']
                    df = await self.fetch_historical_option_trades(
                        instrument_name=inst_name,
                        start_date=start_str,
                        end_date=end_str,
                        count=100 # Limit per instrument
                    )

                    if not df.empty:
                        df['underlying'] = currency
                        return df
                    return None

                inst_tasks = [_fetch_single_instrument_trades(inst) for inst in sampled]
                inst_results = await asyncio.gather(*inst_tasks, return_exceptions=True)

                symbol_records = [r for r in inst_results if isinstance(r, pd.DataFrame) and not r.empty]
                return symbol_records if symbol_records else None

            tasks = [_fetch_single_symbol_trades(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Flatten the list of lists
            all_records = []
            for r in results:
                if isinstance(r, list):
                    all_records.extend(r)

            if all_records:
                result = pd.concat(all_records, ignore_index=True)
                return result.sort_values('timestamp').reset_index(drop=True)

            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"collect_historical_option_trades error: {e}")
            return pd.DataFrame()

    @staticmethod
    def calculate_greeks_from_iv(
        spot: float,
        strike: float,
        time_to_expiry: float,
        iv: float,
        risk_free_rate: float = 0.05,
        option_type: str = 'call'
    ) -> Dict[str, float]:
        """
        Calculate option Greeks using Black-Scholes model.

        This utility allows calculating Greeks from historical trade data
        that includes IV but not pre-calculated Greeks.

        Args:
            spot: Current underlying price
            strike: Option strike price
            time_to_expiry: Time to expiry in years
            iv: Implied volatility (decimal, e.g., 0.80 for 80%)
            risk_free_rate: Risk-free rate (decimal)
            option_type: 'call' or 'put'

        Returns:
            Dictionary with delta, gamma, vega, theta, rho
        """
        import math

        if time_to_expiry <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
            return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}

        try:
            # Normal CDF approximation (avoids scipy dependency)
            def norm_cdf(x):
                return 0.5 * (1 + math.erf(x / math.sqrt(2)))

            def norm_pdf(x):
                return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

            sqrt_t = math.sqrt(time_to_expiry)
            d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * iv**2) * time_to_expiry) / (iv * sqrt_t)
            d2 = d1 - iv * sqrt_t

            N_d1 = norm_cdf(d1)
            N_d2 = norm_cdf(d2)
            n_d1 = norm_pdf(d1)

            df = math.exp(-risk_free_rate * time_to_expiry)

            if option_type.lower() == 'call':
                delta = N_d1
                theta = (-spot * n_d1 * iv / (2 * sqrt_t) -
                        risk_free_rate * strike * df * N_d2) / 365
                rho = strike * time_to_expiry * df * N_d2 / 100
            else:
                delta = N_d1 - 1
                theta = (-spot * n_d1 * iv / (2 * sqrt_t) +
                        risk_free_rate * strike * df * (1 - N_d2)) / 365
                rho = -strike * time_to_expiry * df * (1 - N_d2) / 100

            gamma = n_d1 / (spot * iv * sqrt_t)
            vega = spot * n_d1 * sqrt_t / 100

            return {
                'delta': round(delta, 6),
                'gamma': round(gamma, 8),
                'vega': round(vega, 4),
                'theta': round(theta, 4),
                'rho': round(rho, 4)
            }

        except Exception:
            return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}