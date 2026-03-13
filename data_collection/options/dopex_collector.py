"""
Dopex Single Staking Option Vaults (SSOV) Data Collector

validated collector for Dopex decentralized options protocol on Arbitrum.
Dopex pioneered the SSOV model for option writing vaults.

===============================================================================
DOPEX OVERVIEW
===============================================================================

Dopex is a decentralized options protocol featuring Single Staking Option
Vaults (SSOVs) where users can deposit assets to write covered options.

Key Innovations:
    - SSOV model: Deposit assets, earn premiums from option writing
    - Weekly/Monthly epochs with fixed strikes
    - Atlantic Options: Partial collateralization
    - rDPX rebate mechanism for losses

===============================================================================
DATA SOURCE: THE GRAPH SUBGRAPH
===============================================================================

Subgraph URL: https://api.thegraph.com/subgraphs/name/dopex-io/dopex

Rate Limits:
    ============ ============== ================
    Type Requests/min Notes
    ============ ============== ================
    Free Tier 100 Hosted service
    Decentralized Variable Query complexity
    ============ ============== ================

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

Premium Arbitrage:
    - Compare SSOV premiums vs Deribit IV
    - Identify mispriced epochs
    - Strike selection optimization

Yield Analysis:
    - Writer yield vs DeFi alternatives
    - Exercise rate prediction
    - Settlement price analysis

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

class SSOVType(Enum):
    """SSOV vault type classification."""
    CALL = 'CALL'
    PUT = 'PUT'
    
    @property
    def is_call(self) -> bool:
        """Check if call vault."""
        return self == SSOVType.CALL
    
    @property
    def collateral_type(self) -> str:
        """Expected collateral type."""
        return 'underlying' if self.is_call else 'stablecoin'

class Underlying(Enum):
    """Supported underlying assets."""
    DPX = 'DPX'
    RDPX = 'rDPX'
    ETH = 'ETH'
    GOHM = 'gOHM'
    GMX = 'GMX'
    ARB = 'ARB'
    STETH = 'stETH'
    
    @property
    def is_native_token(self) -> bool:
        """Check if Dopex native token."""
        return self in [Underlying.DPX, Underlying.RDPX]
    
    @property
    def typical_iv(self) -> float:
        """Typical IV range for asset."""
        mapping = {
            Underlying.DPX: 1.5, Underlying.RDPX: 2.0, Underlying.ETH: 0.7,
            Underlying.GOHM: 1.2, Underlying.GMX: 1.0, Underlying.ARB: 1.1, Underlying.STETH: 0.65,
        }
        return mapping.get(self, 1.0)

class EpochStatus(Enum):
    """Epoch lifecycle status."""
    PENDING = 'pending'
    ACTIVE = 'active'
    EXPIRED = 'expired'
    SETTLED = 'settled'
    
    @property
    def is_tradeable(self) -> bool:
        """Check if epoch allows trading."""
        return self == EpochStatus.ACTIVE
    
    @property
    def is_final(self) -> bool:
        """Check if epoch is finalized."""
        return self in [EpochStatus.EXPIRED, EpochStatus.SETTLED]

class StrikePosition(Enum):
    """Strike position relative to spot."""
    DEEP_ITM = 'deep_itm'
    ITM = 'itm'
    ATM = 'atm'
    OTM = 'otm'
    DEEP_OTM = 'deep_otm'
    
    @classmethod
    def from_moneyness(cls, moneyness: float, is_call: bool) -> 'StrikePosition':
        """Classify from moneyness ratio."""
        if is_call:
            if moneyness < 0.85:
                return cls.DEEP_ITM
            elif moneyness < 0.95:
                return cls.ITM
            elif moneyness <= 1.05:
                return cls.ATM
            elif moneyness <= 1.15:
                return cls.OTM
            return cls.DEEP_OTM
        else:
            if moneyness > 1.15:
                return cls.DEEP_ITM
            elif moneyness > 1.05:
                return cls.ITM
            elif moneyness >= 0.95:
                return cls.ATM
            elif moneyness >= 0.85:
                return cls.OTM
            return cls.DEEP_OTM
    
    @property
    def exercise_probability(self) -> str:
        """Expected exercise probability."""
        mapping = {
            StrikePosition.DEEP_ITM: 'very_high', StrikePosition.ITM: 'high',
            StrikePosition.ATM: 'medium', StrikePosition.OTM: 'low', StrikePosition.DEEP_OTM: 'very_low',
        }
        return mapping.get(self, 'unknown')

class YieldLevel(Enum):
    """Premium yield classification."""
    EXCEPTIONAL = 'exceptional'
    HIGH = 'high'
    ABOVE_AVERAGE = 'above_average'
    AVERAGE = 'average'
    BELOW_AVERAGE = 'below_average'
    LOW = 'low'
    
    @classmethod
    def from_apy(cls, apy: float) -> 'YieldLevel':
        """Classify from APY."""
        if apy > 100:
            return cls.EXCEPTIONAL
        elif apy > 50:
            return cls.HIGH
        elif apy > 30:
            return cls.ABOVE_AVERAGE
        elif apy > 15:
            return cls.AVERAGE
        elif apy > 5:
            return cls.BELOW_AVERAGE
        return cls.LOW

class ActivityLevel(Enum):
    """Vault activity classification."""
    VERY_HIGH = 'very_high'
    HIGH = 'high'
    MODERATE = 'moderate'
    LOW = 'low'
    DORMANT = 'dormant'
    
    @classmethod
    def from_volume(cls, volume_24h: float) -> 'ActivityLevel':
        """Classify from 24h volume."""
        if volume_24h > 1_000_000:
            return cls.VERY_HIGH
        elif volume_24h > 100_000:
            return cls.HIGH
        elif volume_24h > 10_000:
            return cls.MODERATE
        elif volume_24h > 1_000:
            return cls.LOW
        return cls.DORMANT

class ExerciseOutcome(Enum):
    """Option exercise outcome."""
    EXERCISED_PROFIT = 'exercised_profit'
    EXERCISED_LOSS = 'exercised_loss'
    EXPIRED_WORTHLESS = 'expired_worthless'
    NOT_SETTLED = 'not_settled'

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class SSOVVault:
    """SSOV vault metadata and analytics."""
    vault_address: str
    underlying: str
    ssov_type: str
    collateral_token: str
    current_epoch: int = 0
    total_deposits: float = 0.0
    total_deposits_usd: float = 0.0
    utilization_rate: float = 0.0
    is_active: bool = True
    created_at: Optional[datetime] = None
    
    @property
    def ssov_type_enum(self) -> SSOVType:
        """Get SSOV type as enum."""
        return SSOVType.CALL if self.ssov_type.upper() == 'CALL' else SSOVType.PUT
    
    @property
    def underlying_enum(self) -> Optional[Underlying]:
        """Get underlying as enum."""
        try:
            return Underlying(self.underlying)
        except ValueError:
            return None
    
    @property
    def is_call_vault(self) -> bool:
        """Check if call vault."""
        return self.ssov_type_enum.is_call
    
    @property
    def is_put_vault(self) -> bool:
        """Check if put vault."""
        return not self.is_call_vault
    
    @property
    def is_well_utilized(self) -> bool:
        """Check if vault has good utilization."""
        return 0.3 <= self.utilization_rate <= 0.8
    
    @property
    def utilization_status(self) -> str:
        """Utilization status description."""
        if self.utilization_rate < 0.1:
            return 'underutilized'
        elif self.utilization_rate < 0.5:
            return 'moderate'
        elif self.utilization_rate < 0.8:
            return 'healthy'
        elif self.utilization_rate < 0.95:
            return 'high'
        return 'maxed'
    
    @property
    def tvl_tier(self) -> str:
        """TVL tier classification."""
        if self.total_deposits_usd > 10_000_000:
            return 'large'
        elif self.total_deposits_usd > 1_000_000:
            return 'medium'
        elif self.total_deposits_usd > 100_000:
            return 'small'
        return 'micro'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'vault_address': self.vault_address, 'underlying': self.underlying,
            'ssov_type': self.ssov_type, 'collateral_token': self.collateral_token,
            'current_epoch': self.current_epoch, 'total_deposits': self.total_deposits,
            'total_deposits_usd': self.total_deposits_usd, 'utilization_rate': self.utilization_rate,
            'is_active': self.is_active, 'is_call_vault': self.is_call_vault,
            'utilization_status': self.utilization_status, 'tvl_tier': self.tvl_tier,
        }

@dataclass
class EpochData:
    """SSOV epoch data with comprehensive analytics."""
    vault_address: str
    epoch_number: int
    underlying: str
    ssov_type: str
    start_time: datetime
    expiry_time: datetime
    settlement_price: float = 0.0
    total_deposits: float = 0.0
    total_premium_collected: float = 0.0
    total_exercised: float = 0.0
    strikes: List[float] = field(default_factory=list)
    status: str = 'active'
    underlying_price_at_start: float = 0.0
    
    @property
    def ssov_type_enum(self) -> SSOVType:
        """Get SSOV type as enum."""
        return SSOVType.CALL if self.ssov_type.upper() == 'CALL' else SSOVType.PUT
    
    @property
    def status_enum(self) -> EpochStatus:
        """Get status as enum."""
        mapping = {'pending': EpochStatus.PENDING, 'active': EpochStatus.ACTIVE,
                   'expired': EpochStatus.EXPIRED, 'settled': EpochStatus.SETTLED}
        return mapping.get(self.status.lower(), EpochStatus.ACTIVE)
    
    @property
    def duration_days(self) -> float:
        """Epoch duration in days."""
        return (self.expiry_time - self.start_time).total_seconds() / 86400
    
    @property
    def time_to_expiry(self) -> float:
        """Time to expiry in days."""
        now = datetime.now(timezone.utc)
        expiry = self.expiry_time.replace(tzinfo=timezone.utc) if self.expiry_time.tzinfo is None else self.expiry_time
        return max(0, (expiry - now).total_seconds() / 86400)
    
    @property
    def is_expired(self) -> bool:
        """Check if epoch is expired."""
        return self.time_to_expiry <= 0
    
    @property
    def is_weekly(self) -> bool:
        """Check if weekly epoch."""
        return self.duration_days <= 8
    
    @property
    def is_monthly(self) -> bool:
        """Check if monthly epoch."""
        return self.duration_days > 25
    
    @property
    def epoch_type(self) -> str:
        """Epoch type classification."""
        if self.is_weekly:
            return 'weekly'
        elif self.is_monthly:
            return 'monthly'
        return 'biweekly'
    
    @property
    def premium_yield(self) -> float:
        """Premium yield (premium / deposits)."""
        return self.total_premium_collected / self.total_deposits if self.total_deposits > 0 else 0.0
    
    @property
    def premium_yield_pct(self) -> float:
        """Premium yield as percentage."""
        return self.premium_yield * 100
    
    @property
    def annualized_yield(self) -> float:
        """Annualized premium yield."""
        if self.duration_days <= 0:
            return 0.0
        return self.premium_yield * (365 / self.duration_days)
    
    @property
    def annualized_yield_pct(self) -> float:
        """Annualized yield as percentage."""
        return self.annualized_yield * 100
    
    @property
    def yield_level(self) -> YieldLevel:
        """Classify yield level."""
        return YieldLevel.from_apy(self.annualized_yield_pct)
    
    @property
    def exercise_rate(self) -> float:
        """Exercise rate (exercised / deposits)."""
        return self.total_exercised / self.total_deposits if self.total_deposits > 0 else 0.0
    
    @property
    def exercise_rate_pct(self) -> float:
        """Exercise rate as percentage."""
        return self.exercise_rate * 100
    
    @property
    def net_writer_return(self) -> float:
        """Net return for writers (premium - exercised losses)."""
        return self.total_premium_collected - self.total_exercised
    
    @property
    def net_writer_yield(self) -> float:
        """Net yield for writers."""
        return self.net_writer_return / self.total_deposits if self.total_deposits > 0 else 0.0
    
    @property
    def net_writer_yield_pct(self) -> float:
        """Net writer yield as percentage."""
        return self.net_writer_yield * 100
    
    @property
    def writer_pnl_status(self) -> str:
        """Writer P&L status."""
        if self.net_writer_yield > 0.05:
            return 'highly_profitable'
        elif self.net_writer_yield > 0:
            return 'profitable'
        elif self.net_writer_yield > -0.05:
            return 'slight_loss'
        return 'significant_loss'
    
    @property
    def num_strikes(self) -> int:
        """Number of available strikes."""
        return len(self.strikes)
    
    @property
    def strike_range(self) -> Optional[tuple]:
        """Strike range (min, max)."""
        return (min(self.strikes), max(self.strikes)) if self.strikes else None
    
    @property
    def atm_strike(self) -> Optional[float]:
        """Closest strike to ATM."""
        if not self.strikes or self.underlying_price_at_start <= 0:
            return None
        return min(self.strikes, key=lambda x: abs(x - self.underlying_price_at_start))
    
    @property
    def price_move(self) -> float:
        """Price move during epoch (settlement vs start)."""
        if self.underlying_price_at_start <= 0 or self.settlement_price <= 0:
            return 0.0
        return (self.settlement_price - self.underlying_price_at_start) / self.underlying_price_at_start
    
    @property
    def price_move_pct(self) -> float:
        """Price move as percentage."""
        return self.price_move * 100
    
    @property
    def is_favorable_for_calls(self) -> bool:
        """Check if settlement was favorable for call writers."""
        return self.settlement_price < self.underlying_price_at_start if self.settlement_price > 0 else False
    
    @property
    def is_favorable_for_puts(self) -> bool:
        """Check if settlement was favorable for put writers."""
        return self.settlement_price > self.underlying_price_at_start if self.settlement_price > 0 else False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'vault_address': self.vault_address, 'epoch_number': self.epoch_number,
            'underlying': self.underlying, 'ssov_type': self.ssov_type,
            'start_time': self.start_time.isoformat() if isinstance(self.start_time, datetime) else self.start_time,
            'expiry_time': self.expiry_time.isoformat() if isinstance(self.expiry_time, datetime) else self.expiry_time,
            'settlement_price': self.settlement_price, 'total_deposits': self.total_deposits,
            'total_premium_collected': self.total_premium_collected, 'total_exercised': self.total_exercised,
            'duration_days': self.duration_days, 'time_to_expiry': self.time_to_expiry,
            'epoch_type': self.epoch_type, 'status': self.status,
            'premium_yield_pct': self.premium_yield_pct, 'annualized_yield_pct': self.annualized_yield_pct,
            'yield_level': self.yield_level.value, 'exercise_rate_pct': self.exercise_rate_pct,
            'net_writer_yield_pct': self.net_writer_yield_pct, 'writer_pnl_status': self.writer_pnl_status,
            'num_strikes': self.num_strikes, 'price_move_pct': self.price_move_pct,
        }

@dataclass
class StrikeData:
    """Strike-level data within an epoch."""
    vault_address: str
    epoch_number: int
    strike: float
    underlying: str
    ssov_type: str
    total_deposits: float = 0.0
    total_purchased: float = 0.0
    premium_collected: float = 0.0
    exercised_amount: float = 0.0
    underlying_price: float = 0.0
    implied_volatility: float = 0.0
    
    @property
    def ssov_type_enum(self) -> SSOVType:
        """Get SSOV type as enum."""
        return SSOVType.CALL if self.ssov_type.upper() == 'CALL' else SSOVType.PUT
    
    @property
    def moneyness(self) -> float:
        """Calculate moneyness (strike / spot)."""
        return self.strike / self.underlying_price if self.underlying_price > 0 else 0.0
    
    @property
    def log_moneyness(self) -> float:
        """Log moneyness for analytics."""
        return np.log(self.moneyness) if self.moneyness > 0 else 0.0
    
    @property
    def strike_position(self) -> StrikePosition:
        """Classify strike position."""
        return StrikePosition.from_moneyness(self.moneyness, self.ssov_type_enum.is_call)
    
    @property
    def is_atm(self) -> bool:
        """Check if approximately ATM."""
        return 0.95 <= self.moneyness <= 1.05
    
    @property
    def is_itm(self) -> bool:
        """Check if in-the-money."""
        if self.ssov_type_enum.is_call:
            return self.underlying_price > self.strike
        return self.underlying_price < self.strike
    
    @property
    def is_otm(self) -> bool:
        """Check if out-of-the-money."""
        return not self.is_itm and not self.is_atm
    
    @property
    def intrinsic_value(self) -> float:
        """Calculate intrinsic value."""
        if self.ssov_type_enum.is_call:
            return max(0, self.underlying_price - self.strike)
        return max(0, self.strike - self.underlying_price)
    
    @property
    def intrinsic_value_pct(self) -> float:
        """Intrinsic value as percentage of strike."""
        return (self.intrinsic_value / self.strike) * 100 if self.strike > 0 else 0.0
    
    @property
    def utilization(self) -> float:
        """Strike utilization (purchased / deposits)."""
        return self.total_purchased / self.total_deposits if self.total_deposits > 0 else 0.0
    
    @property
    def utilization_pct(self) -> float:
        """Utilization as percentage."""
        return self.utilization * 100
    
    @property
    def avg_premium_per_contract(self) -> float:
        """Average premium per contract."""
        return self.premium_collected / self.total_purchased if self.total_purchased > 0 else 0.0
    
    @property
    def exercise_rate(self) -> float:
        """Exercise rate for this strike."""
        return self.exercised_amount / self.total_purchased if self.total_purchased > 0 else 0.0
    
    @property
    def exercise_rate_pct(self) -> float:
        """Exercise rate as percentage."""
        return self.exercise_rate * 100
    
    @property
    def writer_pnl(self) -> float:
        """Writer P&L (premium - exercise losses)."""
        return self.premium_collected - self.exercised_amount * self.intrinsic_value
    
    @property
    def writer_yield(self) -> float:
        """Writer yield on deposits."""
        return self.writer_pnl / self.total_deposits if self.total_deposits > 0 else 0.0
    
    @property
    def implied_vol_pct(self) -> float:
        """IV as percentage."""
        return self.implied_volatility * 100 if self.implied_volatility < 10 else self.implied_volatility
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'vault_address': self.vault_address, 'epoch_number': self.epoch_number,
            'strike': self.strike, 'underlying': self.underlying, 'ssov_type': self.ssov_type,
            'total_deposits': self.total_deposits, 'total_purchased': self.total_purchased,
            'premium_collected': self.premium_collected, 'exercised_amount': self.exercised_amount,
            'underlying_price': self.underlying_price, 'moneyness': self.moneyness,
            'strike_position': self.strike_position.value, 'is_itm': self.is_itm,
            'intrinsic_value': self.intrinsic_value, 'utilization_pct': self.utilization_pct,
            'avg_premium_per_contract': self.avg_premium_per_contract,
            'exercise_rate_pct': self.exercise_rate_pct, 'writer_yield': self.writer_yield,
            'implied_vol_pct': self.implied_vol_pct,
        }

@dataclass
class PurchaseEvent:
    """Option purchase event from SSOV."""
    tx_hash: str
    timestamp: datetime
    vault_address: str
    epoch_number: int
    strike: float
    amount: float
    premium: float
    buyer: str
    underlying_price: float = 0.0
    
    @property
    def premium_per_contract(self) -> float:
        """Premium per contract."""
        return self.premium / self.amount if self.amount > 0 else 0.0
    
    @property
    def premium_pct(self) -> float:
        """Premium as percentage of strike."""
        return (self.premium_per_contract / self.strike) * 100 if self.strike > 0 else 0.0
    
    @property
    def notional_value(self) -> float:
        """Notional value of purchase."""
        return self.amount * self.strike
    
    @property
    def is_large_purchase(self) -> bool:
        """Check if large purchase (>$10k notional)."""
        return self.notional_value > 10000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'tx_hash': self.tx_hash,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'vault_address': self.vault_address, 'epoch_number': self.epoch_number,
            'strike': self.strike, 'amount': self.amount, 'premium': self.premium,
            'buyer': self.buyer, 'premium_per_contract': self.premium_per_contract,
            'premium_pct': self.premium_pct, 'notional_value': self.notional_value,
            'is_large_purchase': self.is_large_purchase,
        }

@dataclass
class ExerciseEvent:
    """Option exercise event from SSOV."""
    tx_hash: str
    timestamp: datetime
    vault_address: str
    epoch_number: int
    strike: float
    amount: float
    pnl: float
    exerciser: str
    settlement_price: float = 0.0
    
    @property
    def pnl_per_contract(self) -> float:
        """P&L per contract."""
        return self.pnl / self.amount if self.amount > 0 else 0.0
    
    @property
    def pnl_pct(self) -> float:
        """P&L as percentage of strike."""
        return (self.pnl_per_contract / self.strike) * 100 if self.strike > 0 else 0.0
    
    @property
    def is_profitable(self) -> bool:
        """Check if exercise was profitable."""
        return self.pnl > 0
    
    @property
    def outcome(self) -> ExerciseOutcome:
        """Exercise outcome classification."""
        if self.pnl > 0:
            return ExerciseOutcome.EXERCISED_PROFIT
        elif self.pnl < 0:
            return ExerciseOutcome.EXERCISED_LOSS
        return ExerciseOutcome.EXPIRED_WORTHLESS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'tx_hash': self.tx_hash,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'vault_address': self.vault_address, 'epoch_number': self.epoch_number,
            'strike': self.strike, 'amount': self.amount, 'pnl': self.pnl,
            'exerciser': self.exerciser, 'settlement_price': self.settlement_price,
            'pnl_per_contract': self.pnl_per_contract, 'pnl_pct': self.pnl_pct,
            'is_profitable': self.is_profitable, 'outcome': self.outcome.value,
        }

@dataclass
class DepositEvent:
    """Deposit event to SSOV vault."""
    tx_hash: str
    timestamp: datetime
    vault_address: str
    epoch_number: int
    strike: float
    amount: float
    depositor: str
    
    @property
    def is_large_deposit(self) -> bool:
        """Check if large deposit."""
        return self.amount > 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'tx_hash': self.tx_hash,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'vault_address': self.vault_address, 'epoch_number': self.epoch_number,
            'strike': self.strike, 'amount': self.amount, 'depositor': self.depositor,
            'is_large_deposit': self.is_large_deposit,
        }

# =============================================================================
# COLLECTOR CLASS
# =============================================================================

class DopexCollector:
    """
    Dopex SSOV data collector via The Graph.
    
    Features:
    - SSOV vault data
    - Epoch analytics
    - Strike-level data
    - Purchase/exercise events
    - Premium arbitrage analysis
    """
    
    VENUE = 'dopex'
    VENUE_TYPE = 'DEX_OPTIONS'
    SUBGRAPH_URL = 'https://api.thegraph.com/subgraphs/name/dopex-io/dopex'
    CHAIN = 'arbitrum'
    
    SSOV_VAULTS_QUERY = """
    query GetSSOVVaults($first: Int!, $skip: Int!) {
        ssovs(first: $first, skip: $skip) {
            id
            underlying
            type
            collateralToken
            currentEpoch
            totalDeposits
            utilizationRate
        }
    }
    """
    
    EPOCHS_QUERY = """
    query GetEpochs($vault: String!, $first: Int!) {
        epochs(where: {ssov: $vault}, first: $first, orderBy: epoch, orderDirection: desc) {
            id
            epoch
            startTime
            expiry
            settlementPrice
            totalDeposits
            totalPremiumCollected
            totalExercised
            strikes
        }
    }
    """
    
    STRIKES_QUERY = """
    query GetStrikes($vault: String!, $epoch: Int!) {
        strikeData(where: {ssov: $vault, epoch: $epoch}) {
            id
            strike
            totalDeposits
            totalPurchased
            premiumCollected
            exercisedAmount
        }
    }
    """
    
    PURCHASES_QUERY = """
    query GetPurchases($vault: String!, $epoch: Int!, $first: Int!) {
        purchases(where: {ssov: $vault, epoch: $epoch}, first: $first, orderBy: timestamp, orderDirection: desc) {
            id
            txHash
            timestamp
            strike
            amount
            premium
            buyer
        }
    }
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Dopex collector."""
        config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        self.subgraph_url = config.get('subgraph_url', self.SUBGRAPH_URL)
        self.rate_limit = config.get('rate_limit', 50)
        self.last_request_time = 0.0
        self.min_request_interval = 60.0 / self.rate_limit
        self.collection_stats = {'requests': 0, 'records': 0, 'errors': 0}
    
    async def __aenter__(self) -> 'DopexCollector':
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
    
    async def fetch_ssov_vaults(self, first: int = 100, skip: int = 0) -> List[SSOVVault]:
        """Fetch SSOV vault information."""
        data = await self._query_subgraph(self.SSOV_VAULTS_QUERY, {'first': first, 'skip': skip})
        vaults = []
        
        if data and 'ssovs' in data:
            for v in data['ssovs']:
                vaults.append(SSOVVault(
                    vault_address=v.get('id', ''),
                    underlying=v.get('underlying', ''),
                    ssov_type=v.get('type', 'CALL'),
                    collateral_token=v.get('collateralToken', ''),
                    current_epoch=int(v.get('currentEpoch', 0)),
                    total_deposits=float(v.get('totalDeposits', 0)),
                    utilization_rate=float(v.get('utilizationRate', 0)),
                    is_active=True,
                ))
                self.collection_stats['records'] += 1
        
        return vaults
    
    async def fetch_epochs(self, vault_address: str, first: int = 20) -> List[EpochData]:
        """Fetch epoch data for a vault."""
        data = await self._query_subgraph(self.EPOCHS_QUERY, {'vault': vault_address, 'first': first})
        epochs = []
        
        if data and 'epochs' in data:
            for e in data['epochs']:
                start_ts = int(e.get('startTime', 0))
                expiry_ts = int(e.get('expiry', 0))
                epochs.append(EpochData(
                    vault_address=vault_address,
                    epoch_number=int(e.get('epoch', 0)),
                    underlying='',
                    ssov_type='CALL',
                    start_time=datetime.fromtimestamp(start_ts, tz=timezone.utc) if start_ts else datetime.now(timezone.utc),
                    expiry_time=datetime.fromtimestamp(expiry_ts, tz=timezone.utc) if expiry_ts else datetime.now(timezone.utc),
                    settlement_price=float(e.get('settlementPrice', 0)),
                    total_deposits=float(e.get('totalDeposits', 0)),
                    total_premium_collected=float(e.get('totalPremiumCollected', 0)),
                    total_exercised=float(e.get('totalExercised', 0)),
                    strikes=[float(s) for s in e.get('strikes', [])],
                ))
                self.collection_stats['records'] += 1
        
        return epochs
    
    async def fetch_strikes(self, vault_address: str, epoch: int, underlying_price: float = 0.0) -> List[StrikeData]:
        """Fetch strike-level data for an epoch."""
        data = await self._query_subgraph(self.STRIKES_QUERY, {'vault': vault_address, 'epoch': epoch})
        strikes = []
        
        if data and 'strikeData' in data:
            for s in data['strikeData']:
                strikes.append(StrikeData(
                    vault_address=vault_address,
                    epoch_number=epoch,
                    strike=float(s.get('strike', 0)),
                    underlying='',
                    ssov_type='CALL',
                    total_deposits=float(s.get('totalDeposits', 0)),
                    total_purchased=float(s.get('totalPurchased', 0)),
                    premium_collected=float(s.get('premiumCollected', 0)),
                    exercised_amount=float(s.get('exercisedAmount', 0)),
                    underlying_price=underlying_price,
                ))
                self.collection_stats['records'] += 1
        
        return strikes
    
    async def fetch_purchases(self, vault_address: str, epoch: int, first: int = 100) -> List[PurchaseEvent]:
        """Fetch purchase events for an epoch."""
        data = await self._query_subgraph(self.PURCHASES_QUERY, {'vault': vault_address, 'epoch': epoch, 'first': first})
        purchases = []
        
        if data and 'purchases' in data:
            for p in data['purchases']:
                ts = int(p.get('timestamp', 0))
                purchases.append(PurchaseEvent(
                    tx_hash=p.get('txHash', ''),
                    timestamp=datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc),
                    vault_address=vault_address,
                    epoch_number=epoch,
                    strike=float(p.get('strike', 0)),
                    amount=float(p.get('amount', 0)),
                    premium=float(p.get('premium', 0)),
                    buyer=p.get('buyer', ''),
                ))
                self.collection_stats['records'] += 1
        
        return purchases
    
    async def _fetch_single_vault_data(self, vault: SSOVVault) -> List[Dict[str, Any]]:
        """Helper to fetch all epoch data for a single vault."""
        try:
            epochs = await self.fetch_epochs(vault.vault_address, first=5)
            vault_data = []
            for epoch in epochs:
                epoch_dict = epoch.to_dict()
                epoch_dict['underlying'] = vault.underlying
                epoch_dict['ssov_type'] = vault.ssov_type
                vault_data.append(epoch_dict)
            return vault_data
        except Exception as e:
            self.logger.error(f"Error fetching data for vault {vault.vault_address}: {e}")
            return []

    async def fetch_all_ssov_data(self) -> pd.DataFrame:
        """Fetch comprehensive SSOV data across all vaults."""
        self.logger.info("Fetching all Dopex SSOV data")
        vaults = await self.fetch_ssov_vaults()

        # Parallelize vault data fetching
        tasks = [self._fetch_single_vault_data(vault) for vault in vaults]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and filter out errors
        all_data = []
        for result in results:
            if isinstance(result, list):
                all_data.extend(result)

        df = pd.DataFrame(all_data)
        if not df.empty:
            df['venue'] = self.VENUE
            df['venue_type'] = self.VENUE_TYPE
            df['chain'] = self.CHAIN
        return df
    
    async def _fetch_single_vault_premium_data(self, vault: SSOVVault) -> List[Dict[str, Any]]:
        """Helper to fetch premium comparison data for a single vault."""
        try:
            epochs = await self.fetch_epochs(vault.vault_address, first=3)
            vault_comparison_data = []

            # Process all active epochs
            for epoch in epochs:
                if epoch.status_enum == EpochStatus.ACTIVE:
                    strikes = await self.fetch_strikes(vault.vault_address, epoch.epoch_number)
                    for strike in strikes:
                        vault_comparison_data.append({
                            'venue': self.VENUE, 'underlying': vault.underlying,
                            'ssov_type': vault.ssov_type, 'strike': strike.strike,
                            'epoch': epoch.epoch_number, 'days_to_expiry': epoch.time_to_expiry,
                            'implied_vol': strike.implied_vol_pct / 100 if strike.implied_vol_pct > 0 else 0,
                            'premium_yield': epoch.premium_yield,
                            'utilization': strike.utilization,
                        })

            return vault_comparison_data
        except Exception as e:
            self.logger.error(f"Error fetching premium data for vault {vault.vault_address}: {e}")
            return []

    async def fetch_premium_comparison_data(self) -> pd.DataFrame:
        """Fetch data formatted for premium arbitrage analysis."""
        vaults = await self.fetch_ssov_vaults()

        # Parallelize vault premium data fetching
        tasks = [self._fetch_single_vault_premium_data(vault) for vault in vaults]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and filter out errors
        comparison_data = []
        for result in results:
            if isinstance(result, list):
                comparison_data.extend(result)

        return pd.DataFrame(comparison_data)
    
    async def fetch_funding_rates(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Options protocol doesn't have perpetual funding rates."""
        return pd.DataFrame()
    
    async def fetch_ohlcv(self, symbols: List[str], timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """SSOV doesn't provide standard OHLCV."""
        return pd.DataFrame()

    async def collect_options(self, symbols: List[str], start_date: Any, end_date: Any, **kwargs) -> pd.DataFrame:
        """Standardized collect_options wrapper - wraps fetch_all_ssov_data()."""
        try:
            # Fetch all SSOV data (options data)
            df = await self.fetch_all_ssov_data()

            # Filter by symbols if DataFrame is not empty
            if not df.empty and symbols and 'underlying' in df.columns:
                # Normalize symbols to match Dopex format (e.g., BTC, ETH)
                normalized_symbols = [s.replace('WBTC', 'BTC').replace('WETH', 'ETH').upper() for s in symbols]
                df = df[df['underlying'].str.upper().isin(normalized_symbols)]

            return df
        except Exception as e:
            self.logger.error(f"Dopex collect_options error: {e}")
            return pd.DataFrame()
    
    def get_collection_stats(self) -> Dict[str, Any]:
        return {**self.collection_stats, 'venue': self.VENUE, 'chain': self.CHAIN}
    
    @staticmethod
    def get_supported_underlyings() -> List[str]:
        return [u.value for u in Underlying]
    
    async def close(self) -> None:
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None