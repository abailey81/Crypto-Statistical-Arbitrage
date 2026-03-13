"""
Nansen Smart Money Tracking Collector

validated collector for smart money intelligence from Nansen.
Provides wallet labels, smart money flows, token holder analysis,
and DEX trader profitability metrics.

===============================================================================
NANSEN OVERVIEW
===============================================================================

Nansen is the premier smart money tracking platform, providing wallet labels
and behavioral analytics for on-chain intelligence. Used by funds, traders,
and researchers to follow comprehensive market participants.

Key Differentiators:
    - 100M+ labeled wallet addresses
    - Smart Money wallet identification
    - Real-time flow tracking
    - DEX trader profitability metrics
    - NFT trader analytics
    - Fund and exchange flow monitoring

===============================================================================
API SPECIFICATIONS
===============================================================================

Base URL: https://api.nansen.ai/v1

Authentication:
    - Bearer Token in Authorization header
    - API keys obtained from Nansen dashboard

Rate Limits by Tier:
    ============ ============== ================ ===============
    Tier Requests/min Daily Limit Features
    ============ ============== ================ ===============
    Starter 60 10,000 Basic labels
    Professional 120 50,000 Smart Money
    Enterprise Custom Unlimited Full access
    ============ ============== ================ ===============

===============================================================================
DATA CATEGORIES
===============================================================================

Wallet Labels:
    - Entity identification (exchanges, funds, whales)
    - Smart Money classification
    - Risk scoring
    - Historical activity

Token Analytics:
    - Holder composition with labels
    - Smart Money flows
    - Profit leaderboards
    - New buyer analysis

DEX Analytics:
    - Trader profitability
    - Win rates
    - Trade patterns

Exchange Flows:
    - Deposit/withdrawal tracking
    - Net flow analysis
    - Per-exchange breakdown

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

Alpha Generation:
    - Follow Smart Money accumulation
    - Early token detection via smart buyers
    - Copy trading signals

Risk Assessment:
    - Exchange concentration risk
    - Whale selling detection
    - Token distribution analysis

Market Timing:
    - Smart Money flow divergence
    - Institutional accumulation phases
    - Distribution pattern recognition

Version: 2.0.0
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any

import aiohttp
import pandas as pd
import numpy as np

# =============================================================================
# ENUMS
# =============================================================================

class Chain(Enum):
    """Supported blockchain networks."""
    ETHEREUM = 'ethereum'
    POLYGON = 'polygon'
    ARBITRUM = 'arbitrum'
    OPTIMISM = 'optimism'
    BSC = 'bsc'
    AVALANCHE = 'avalanche'
    BASE = 'base'
    SOLANA = 'solana'

class LabelCategory(Enum):
    """Nansen wallet label categories."""
    SMART_MONEY = 'smart_money'
    FUND = 'fund'
    EXCHANGE = 'exchange'
    DEFI_USER = 'defi_user'
    NFT_TRADER = 'nft_trader'
    WHALE = 'whale'
    EARLY_ADOPTER = 'early_adopter'
    AIRDROP_HUNTER = 'airdrop_hunter'
    BOT = 'bot'
    MINER = 'miner'
    FOUNDATION = 'foundation'
    TEAM = 'team'
    GOVERNANCE = 'governance'
    LIQUIDITY_PROVIDER = 'liquidity_provider'

class SmartMoneyType(Enum):
    """Types of smart money wallets."""
    FUND = 'fund'
    SMART_DEX_TRADER = 'smart_dex_trader'
    SMART_NFT_TRADER = 'smart_nft_trader'
    INFLUENCER = 'influencer'
    EARLY_ADOPTER = 'early_adopter'
    INSIDER = 'insider'
    MARKET_MAKER = 'market_maker'

class TraderTier(Enum):
    """DEX trader performance tiers."""
    ELITE = 'elite' # Top 1%
    EXPERT = 'expert' # Top 5%
    PROFICIENT = 'proficient' # Top 20%
    AVERAGE = 'average' # Middle 60%
    NOVICE = 'novice' # Bottom 20%

class RiskLevel(Enum):
    """Wallet risk levels."""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'

class FlowSignal(Enum):
    """Exchange flow signal types."""
    STRONG_ACCUMULATION = 'strong_accumulation'
    ACCUMULATION = 'accumulation'
    NEUTRAL = 'neutral'
    DISTRIBUTION = 'distribution'
    STRONG_DISTRIBUTION = 'strong_distribution'

class HolderType(Enum):
    """Token holder types."""
    SMART_MONEY = 'smart_money'
    INSTITUTIONAL = 'institutional'
    EXCHANGE = 'exchange'
    RETAIL = 'retail'
    CONTRACT = 'contract'

class Timeframe(Enum):
    """Analysis timeframes."""
    HOURS_24 = '24h'
    DAYS_7 = '7d'
    DAYS_30 = '30d'
    DAYS_90 = '90d'
    ALL_TIME = 'all'

class ProfitStatus(Enum):
    """Holder profit status."""
    DEEP_PROFIT = 'deep_profit' # > 100% profit
    PROFIT = 'profit' # 10-100% profit
    BREAKEVEN = 'breakeven' # -10% to 10%
    LOSS = 'loss' # -10% to -50%
    DEEP_LOSS = 'deep_loss' # > 50% loss

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class WalletProfile:
    """Comprehensive wallet profile with labels and analytics."""
    address: str
    chain: str
    labels: List[str]
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None
    balance_usd: float = 0.0
    first_tx_date: Optional[datetime] = None
    last_tx_date: Optional[datetime] = None
    tx_count: int = 0
    unique_tokens: int = 0
    nansen_score: Optional[float] = None
    risk_score: Optional[float] = None
    
    @property
    def is_smart_money(self) -> bool:
        """Check if wallet is Smart Money."""
        return any('smart' in label.lower() for label in self.labels)
    
    @property
    def is_exchange(self) -> bool:
        """Check if wallet is exchange."""
        return any('exchange' in label.lower() for label in self.labels)
    
    @property
    def is_fund(self) -> bool:
        """Check if wallet is fund/institution."""
        return any('fund' in label.lower() or 'institution' in label.lower() for label in self.labels)
    
    @property
    def is_whale(self) -> bool:
        """Check if wallet is whale."""
        return any('whale' in label.lower() for label in self.labels) or self.balance_usd >= 10_000_000
    
    @property
    def primary_label(self) -> str:
        """Get primary label."""
        priority = ['smart_money', 'fund', 'exchange', 'whale']
        for p in priority:
            for label in self.labels:
                if p in label.lower():
                    return label
        return self.labels[0] if self.labels else 'unknown'
    
    @property
    def risk_level(self) -> RiskLevel:
        """Assess wallet risk level."""
        if self.risk_score is None:
            return RiskLevel.MEDIUM
        if self.risk_score < 25:
            return RiskLevel.LOW
        elif self.risk_score < 50:
            return RiskLevel.MEDIUM
        elif self.risk_score < 75:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL
    
    @property
    def activity_level(self) -> str:
        """Assess activity level."""
        if self.tx_count > 10000:
            return 'very_high'
        elif self.tx_count > 1000:
            return 'high'
        elif self.tx_count > 100:
            return 'medium'
        elif self.tx_count > 10:
            return 'low'
        return 'minimal'
    
    @property
    def wallet_age_days(self) -> Optional[int]:
        """Calculate wallet age in days."""
        if self.first_tx_date:
            return (datetime.utcnow() - self.first_tx_date).days
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'address': self.address, 'chain': self.chain, 'labels': self.labels,
            'entity_name': self.entity_name, 'balance_usd': self.balance_usd,
            'is_smart_money': self.is_smart_money, 'is_exchange': self.is_exchange,
            'is_whale': self.is_whale, 'primary_label': self.primary_label,
            'risk_level': self.risk_level.value, 'activity_level': self.activity_level,
        }

@dataclass
class SmartMoneyWallet:
    """Smart Money wallet with performance metrics."""
    address: str
    chain: str
    labels: List[str]
    category: str
    realized_pnl_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0
    total_pnl_usd: float = 0.0
    win_rate: float = 0.0
    avg_trade_size_usd: float = 0.0
    trade_count: int = 0
    tokens_traded: int = 0
    avg_hold_time_days: Optional[float] = None
    roi_pct: float = 0.0
    nansen_score: Optional[float] = None
    timeframe: str = '30d'
    
    @property
    def trader_tier(self) -> TraderTier:
        """Classify trader tier based on performance."""
        if self.win_rate >= 0.7 and self.roi_pct >= 100:
            return TraderTier.ELITE
        elif self.win_rate >= 0.6 and self.roi_pct >= 50:
            return TraderTier.EXPERT
        elif self.win_rate >= 0.5 and self.roi_pct >= 20:
            return TraderTier.PROFICIENT
        elif self.win_rate >= 0.4:
            return TraderTier.AVERAGE
        return TraderTier.NOVICE
    
    @property
    def is_profitable(self) -> bool:
        """Check if trader is profitable."""
        return self.total_pnl_usd > 0
    
    @property
    def is_active(self) -> bool:
        """Check if trader is active."""
        return self.trade_count >= 10
    
    @property
    def avg_pnl_per_trade(self) -> float:
        """Average PnL per trade."""
        if self.trade_count > 0:
            return self.total_pnl_usd / self.trade_count
        return 0.0
    
    @property
    def trade_frequency(self) -> str:
        """Classify trade frequency."""
        if self.trade_count > 100:
            return 'very_active'
        elif self.trade_count > 50:
            return 'active'
        elif self.trade_count > 20:
            return 'moderate'
        elif self.trade_count > 5:
            return 'occasional'
        return 'rare'
    
    @property
    def follow_worthiness(self) -> str:
        """Assess if wallet is worth following."""
        if self.trader_tier in [TraderTier.ELITE, TraderTier.EXPERT] and self.is_active:
            return 'high'
        elif self.trader_tier == TraderTier.PROFICIENT and self.is_active:
            return 'medium'
        return 'low'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'address': self.address, 'chain': self.chain, 'category': self.category,
            'total_pnl_usd': self.total_pnl_usd, 'win_rate': self.win_rate,
            'roi_pct': self.roi_pct, 'trade_count': self.trade_count,
            'trader_tier': self.trader_tier.value, 'is_profitable': self.is_profitable,
            'follow_worthiness': self.follow_worthiness, 'avg_pnl_per_trade': self.avg_pnl_per_trade,
        }

@dataclass
class TokenHolder:
    """Token holder with labels and analytics."""
    token_address: str
    chain: str
    rank: int
    holder_address: str
    labels: List[str]
    entity_name: Optional[str] = None
    balance: float = 0.0
    balance_usd: float = 0.0
    pct_of_supply: float = 0.0
    cost_basis_usd: float = 0.0
    pnl_usd: float = 0.0
    first_buy_date: Optional[datetime] = None
    last_activity_date: Optional[datetime] = None
    
    @property
    def holder_type(self) -> HolderType:
        """Classify holder type."""
        for label in self.labels:
            label_lower = label.lower()
            if 'smart' in label_lower:
                return HolderType.SMART_MONEY
            elif 'fund' in label_lower or 'institution' in label_lower:
                return HolderType.INSTITUTIONAL
            elif 'exchange' in label_lower:
                return HolderType.EXCHANGE
            elif 'contract' in label_lower:
                return HolderType.CONTRACT
        return HolderType.RETAIL
    
    @property
    def is_smart_money(self) -> bool:
        """Check if Smart Money holder."""
        return self.holder_type == HolderType.SMART_MONEY
    
    @property
    def is_significant(self) -> bool:
        """Check if significant holder."""
        return self.pct_of_supply >= 1.0
    
    @property
    def profit_status(self) -> ProfitStatus:
        """Classify profit status."""
        if self.cost_basis_usd <= 0:
            return ProfitStatus.BREAKEVEN
        
        pnl_pct = (self.pnl_usd / self.cost_basis_usd) * 100
        
        if pnl_pct > 100:
            return ProfitStatus.DEEP_PROFIT
        elif pnl_pct > 10:
            return ProfitStatus.PROFIT
        elif pnl_pct > -10:
            return ProfitStatus.BREAKEVEN
        elif pnl_pct > -50:
            return ProfitStatus.LOSS
        return ProfitStatus.DEEP_LOSS
    
    @property
    def roi_pct(self) -> float:
        """Calculate ROI percentage."""
        if self.cost_basis_usd > 0:
            return (self.pnl_usd / self.cost_basis_usd) * 100
        return 0.0
    
    @property
    def holding_days(self) -> Optional[int]:
        """Days since first buy."""
        if self.first_buy_date:
            return (datetime.utcnow() - self.first_buy_date).days
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rank': self.rank, 'holder_address': self.holder_address, 'labels': self.labels,
            'balance_usd': self.balance_usd, 'pct_of_supply': self.pct_of_supply,
            'pnl_usd': self.pnl_usd, 'holder_type': self.holder_type.value,
            'is_smart_money': self.is_smart_money, 'profit_status': self.profit_status.value,
            'roi_pct': self.roi_pct,
        }

@dataclass
class SmartMoneyFlow:
    """Smart Money flow data for a token."""
    timestamp: datetime
    token_address: str
    chain: str
    smart_money_buys: float = 0.0
    smart_money_sells: float = 0.0
    smart_money_net: float = 0.0
    smart_money_buyers: int = 0
    smart_money_sellers: int = 0
    total_buys: float = 0.0
    total_sells: float = 0.0
    price_usd: float = 0.0
    sm_pct_of_volume: float = 0.0
    timeframe: str = '7d'
    
    @property
    def flow_signal(self) -> FlowSignal:
        """Classify flow signal."""
        if self.smart_money_net <= 0:
            return FlowSignal.NEUTRAL
        
        if self.sm_pct_of_volume > 30:
            return FlowSignal.STRONG_ACCUMULATION if self.smart_money_net > 0 else FlowSignal.STRONG_DISTRIBUTION
        elif self.sm_pct_of_volume > 15:
            return FlowSignal.ACCUMULATION if self.smart_money_net > 0 else FlowSignal.DISTRIBUTION
        return FlowSignal.NEUTRAL
    
    @property
    def is_accumulation(self) -> bool:
        """Check if Smart Money is accumulating."""
        return self.smart_money_net > 0 and self.smart_money_buyers > self.smart_money_sellers
    
    @property
    def is_distribution(self) -> bool:
        """Check if Smart Money is distributing."""
        return self.smart_money_net < 0 and self.smart_money_sellers > self.smart_money_buyers
    
    @property
    def smart_money_dominance(self) -> str:
        """Assess Smart Money dominance."""
        if self.sm_pct_of_volume > 30:
            return 'dominant'
        elif self.sm_pct_of_volume > 15:
            return 'significant'
        elif self.sm_pct_of_volume > 5:
            return 'moderate'
        return 'minimal'
    
    @property
    def net_buyers(self) -> int:
        """Net Smart Money buyers."""
        return self.smart_money_buyers - self.smart_money_sellers
    
    @property
    def sentiment(self) -> str:
        """Overall Smart Money sentiment."""
        if self.is_accumulation and self.sm_pct_of_volume > 10:
            return 'bullish'
        elif self.is_distribution and self.sm_pct_of_volume > 10:
            return 'bearish'
        return 'neutral'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'token_address': self.token_address, 'chain': self.chain,
            'smart_money_net': self.smart_money_net, 'smart_money_buyers': self.smart_money_buyers,
            'sm_pct_of_volume': self.sm_pct_of_volume, 'flow_signal': self.flow_signal.value,
            'is_accumulation': self.is_accumulation, 'sentiment': self.sentiment,
        }

@dataclass
class ExchangeFlow:
    """Exchange deposit/withdrawal flow data."""
    timestamp: datetime
    token_address: str
    chain: str
    exchange: str
    deposits: float = 0.0
    deposits_usd: float = 0.0
    withdrawals: float = 0.0
    withdrawals_usd: float = 0.0
    net_flow: float = 0.0
    net_flow_usd: float = 0.0
    unique_depositors: int = 0
    unique_withdrawers: int = 0
    timeframe: str = '7d'
    
    @property
    def flow_signal(self) -> FlowSignal:
        """Classify exchange flow signal."""
        if self.net_flow_usd > 1_000_000:
            return FlowSignal.STRONG_DISTRIBUTION
        elif self.net_flow_usd > 100_000:
            return FlowSignal.DISTRIBUTION
        elif self.net_flow_usd < -1_000_000:
            return FlowSignal.STRONG_ACCUMULATION
        elif self.net_flow_usd < -100_000:
            return FlowSignal.ACCUMULATION
        return FlowSignal.NEUTRAL
    
    @property
    def is_bullish(self) -> bool:
        """Check if flow is bullish (withdrawals > deposits)."""
        return self.net_flow_usd < 0
    
    @property
    def is_bearish(self) -> bool:
        """Check if flow is bearish (deposits > withdrawals)."""
        return self.net_flow_usd > 0
    
    @property
    def activity_level(self) -> str:
        """Assess activity level."""
        total_users = self.unique_depositors + self.unique_withdrawers
        if total_users > 1000:
            return 'very_high'
        elif total_users > 100:
            return 'high'
        elif total_users > 10:
            return 'moderate'
        return 'low'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'exchange': self.exchange, 'deposits_usd': self.deposits_usd,
            'withdrawals_usd': self.withdrawals_usd, 'net_flow_usd': self.net_flow_usd,
            'flow_signal': self.flow_signal.value, 'is_bullish': self.is_bullish,
        }

@dataclass
class DEXTrader:
    """DEX trader with profitability analytics."""
    address: str
    chain: str
    dex: str
    labels: List[str]
    is_smart_money: bool = False
    volume_usd: float = 0.0
    trade_count: int = 0
    realized_pnl_usd: float = 0.0
    win_rate: float = 0.0
    avg_trade_size_usd: float = 0.0
    unique_tokens: int = 0
    profitable_tokens: int = 0
    timeframe: str = '7d'
    
    @property
    def trader_tier(self) -> TraderTier:
        """Classify trader tier."""
        if self.win_rate >= 0.7 and self.realized_pnl_usd >= 100_000:
            return TraderTier.ELITE
        elif self.win_rate >= 0.6 and self.realized_pnl_usd >= 10_000:
            return TraderTier.EXPERT
        elif self.win_rate >= 0.5 and self.realized_pnl_usd >= 1_000:
            return TraderTier.PROFICIENT
        elif self.win_rate >= 0.4:
            return TraderTier.AVERAGE
        return TraderTier.NOVICE
    
    @property
    def is_profitable(self) -> bool:
        """Check if trader is profitable."""
        return self.realized_pnl_usd > 0
    
    @property
    def token_win_rate(self) -> float:
        """Win rate by tokens traded."""
        if self.unique_tokens > 0:
            return self.profitable_tokens / self.unique_tokens
        return 0.0
    
    @property
    def avg_pnl_per_trade(self) -> float:
        """Average PnL per trade."""
        if self.trade_count > 0:
            return self.realized_pnl_usd / self.trade_count
        return 0.0
    
    @property
    def trade_size_category(self) -> str:
        """Classify average trade size."""
        if self.avg_trade_size_usd >= 100_000:
            return 'whale'
        elif self.avg_trade_size_usd >= 10_000:
            return 'large'
        elif self.avg_trade_size_usd >= 1_000:
            return 'medium'
        return 'small'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'address': self.address, 'dex': self.dex, 'is_smart_money': self.is_smart_money,
            'volume_usd': self.volume_usd, 'realized_pnl_usd': self.realized_pnl_usd,
            'win_rate': self.win_rate, 'trader_tier': self.trader_tier.value,
            'avg_pnl_per_trade': self.avg_pnl_per_trade,
        }

# =============================================================================
# COLLECTOR CLASS
# =============================================================================

# Import BaseCollector
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_collector import BaseCollector

class NansenCollector(BaseCollector):
    """
    Nansen smart money tracking collector.

    Features:
    - Smart Money wallet labels
    - Token holder analysis with labels
    - DEX trader profitability
    - Exchange deposit/withdrawal tracking
    - Wallet profiling and segments
    """

    VENUE = 'nansen'
    VENUE_TYPE = 'onchain'
    BASE_URL = 'https://api.nansen.ai/v1'

    LABEL_CATEGORIES = [c.value for c in LabelCategory]
    SUPPORTED_CHAINS = [c.value for c in Chain]

    def __init__(self, config: Dict):
        """Initialize Nansen collector."""
        super().__init__(config)

        # CRITICAL: Set supported data types for dynamic routing
        self.supported_data_types = ['wallet_analytics', 'smart_money']
        self.venue = 'nansen'

        # Import VenueType from base_collector
        from ..base_collector import VenueType
        self.venue_type = VenueType.ONCHAIN
        self.requires_auth = True # Requires Nansen API key

        import os
        self.api_key = (
            config.get('nansen_api_key') or
            config.get('api_key') or
            os.getenv('NANSEN_API_KEY', '')
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.collection_stats = {'requests': 0, 'records': 0, 'errors': 0}

        if not self.api_key:
            self.logger.warning("Nansen API key not provided")
        else:
            self.logger.info(f"Initialized Nansen collector with data types: {self.supported_data_types}")
    
    async def __aenter__(self) -> 'NansenCollector':
        """Async context manager entry."""
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            headers = {'Accept': 'application/json', 'Authorization': f'Bearer {self.api_key}'}
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self.session
    
    async def _make_request(
        self, endpoint: str, params: Optional[Dict] = None, method: str = 'GET'
    ) -> Optional[Dict]:
        """Make authenticated API request."""
        session = await self._get_session()
        url = f"{self.BASE_URL}/{endpoint}"
        self.collection_stats['requests'] += 1
        
        try:
            if method == 'GET':
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        self.logger.warning("Rate limited")
                        await asyncio.sleep(15)
                    self.collection_stats['errors'] += 1
                    return None
            else:
                async with session.post(url, json=params) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    self.collection_stats['errors'] += 1
                    return None
        except Exception as e:
            self.logger.error(f"Request error: {e}")
            self.collection_stats['errors'] += 1
            return None
    
    async def fetch_wallet_labels(self, address: str, chain: str = 'ethereum') -> Optional[WalletProfile]:
        """Fetch Nansen labels for a wallet address."""
        data = await self._make_request(f'wallet/{address}', params={'chain': chain})
        
        if data:
            return WalletProfile(
                address=address, chain=chain, labels=data.get('labels', []),
                entity_name=data.get('entity_name'), entity_type=data.get('entity_type'),
                balance_usd=float(data.get('balance_usd', 0)),
                first_tx_date=pd.to_datetime(data.get('first_tx_date')) if data.get('first_tx_date') else None,
                last_tx_date=pd.to_datetime(data.get('last_tx_date')) if data.get('last_tx_date') else None,
                tx_count=data.get('tx_count', 0), unique_tokens=data.get('unique_tokens', 0),
                nansen_score=data.get('nansen_score'), risk_score=data.get('risk_score'),
            )
        return None
    
    async def fetch_smart_money_wallets(
        self, chain: str = 'ethereum', category: str = 'defi',
        timeframe: str = '30d', limit: int = 100
    ) -> pd.DataFrame:
        """Fetch list of Smart Money wallets by category."""
        data = await self._make_request('smart-money', params={
            'chain': chain, 'category': category, 'timeframe': timeframe, 'limit': limit
        })
        
        wallets = []
        if data and 'wallets' in data:
            for w in data['wallets']:
                wallet = SmartMoneyWallet(
                    address=w.get('address'), chain=chain, labels=w.get('labels', []),
                    category=category, realized_pnl_usd=float(w.get('realized_pnl', 0)),
                    unrealized_pnl_usd=float(w.get('unrealized_pnl', 0)),
                    total_pnl_usd=float(w.get('total_pnl', 0)),
                    win_rate=float(w.get('win_rate', 0)), avg_trade_size_usd=float(w.get('avg_trade_size', 0)),
                    trade_count=w.get('trade_count', 0), tokens_traded=w.get('tokens_traded', 0),
                    roi_pct=float(w.get('roi', 0)), nansen_score=w.get('nansen_score'), timeframe=timeframe,
                )
                wallets.append(wallet.to_dict())
            self.collection_stats['records'] += len(wallets)
        
        df = pd.DataFrame(wallets)
        if not df.empty:
            df['timestamp'] = datetime.utcnow()
            df['venue'] = self.VENUE
        return df
    
    async def fetch_token_holders_analysis(
        self, token_address: str, chain: str = 'ethereum', limit: int = 100
    ) -> pd.DataFrame:
        """Fetch token holders with Nansen labels and analytics."""
        data = await self._make_request(f'token/{token_address}/holders', params={'chain': chain, 'limit': limit})
        
        holders = []
        if data and 'holders' in data:
            for i, h in enumerate(data['holders']):
                holder = TokenHolder(
                    token_address=token_address, chain=chain, rank=i + 1,
                    holder_address=h.get('address'), labels=h.get('labels', []),
                    entity_name=h.get('entity_name'), balance=float(h.get('balance', 0)),
                    balance_usd=float(h.get('balance_usd', 0)), pct_of_supply=float(h.get('pct_supply', 0)),
                    cost_basis_usd=float(h.get('cost_basis', 0)), pnl_usd=float(h.get('pnl', 0)),
                    first_buy_date=pd.to_datetime(h.get('first_buy_date')) if h.get('first_buy_date') else None,
                    last_activity_date=pd.to_datetime(h.get('last_activity_date')) if h.get('last_activity_date') else None,
                )
                holders.append(holder.to_dict())
            self.collection_stats['records'] += len(holders)
        
        df = pd.DataFrame(holders)
        if not df.empty:
            df['timestamp'] = datetime.utcnow()
            df['venue'] = self.VENUE
        return df
    
    async def fetch_token_smart_money_flows(
        self, token_address: str, chain: str = 'ethereum', timeframe: str = '7d'
    ) -> pd.DataFrame:
        """Fetch Smart Money buy/sell flows for a token."""
        data = await self._make_request(f'token/{token_address}/smart-money-flows',
                                         params={'chain': chain, 'timeframe': timeframe})
        
        flows = []
        if data and 'flows' in data:
            for f in data['flows']:
                flow = SmartMoneyFlow(
                    timestamp=pd.to_datetime(f.get('date')), token_address=token_address, chain=chain,
                    smart_money_buys=float(f.get('sm_buys', 0)), smart_money_sells=float(f.get('sm_sells', 0)),
                    smart_money_net=float(f.get('sm_net', 0)), smart_money_buyers=f.get('sm_buyers', 0),
                    smart_money_sellers=f.get('sm_sellers', 0), total_buys=float(f.get('total_buys', 0)),
                    total_sells=float(f.get('total_sells', 0)), price_usd=float(f.get('price', 0)),
                    sm_pct_of_volume=float(f.get('sm_pct_volume', 0)), timeframe=timeframe,
                )
                flows.append(flow.to_dict())
            self.collection_stats['records'] += len(flows)
        
        df = pd.DataFrame(flows)
        if not df.empty:
            df['venue'] = self.VENUE
        return df.sort_values('timestamp').reset_index(drop=True) if not df.empty else df
    
    async def fetch_dex_traders(
        self, chain: str = 'ethereum', dex: str = 'uniswap',
        timeframe: str = '7d', limit: int = 100
    ) -> pd.DataFrame:
        """Fetch top DEX traders with profitability metrics."""
        data = await self._make_request('dex-traders', params={
            'chain': chain, 'dex': dex, 'timeframe': timeframe, 'limit': limit
        })
        
        traders = []
        if data and 'traders' in data:
            for t in data['traders']:
                trader = DEXTrader(
                    address=t.get('address'), chain=chain, dex=dex, labels=t.get('labels', []),
                    is_smart_money=t.get('is_smart_money', False), volume_usd=float(t.get('volume_usd', 0)),
                    trade_count=t.get('trade_count', 0), realized_pnl_usd=float(t.get('realized_pnl', 0)),
                    win_rate=float(t.get('win_rate', 0)), avg_trade_size_usd=float(t.get('avg_trade_size', 0)),
                    unique_tokens=t.get('unique_tokens', 0), profitable_tokens=t.get('profitable_tokens', 0),
                    timeframe=timeframe,
                )
                traders.append(trader.to_dict())
            self.collection_stats['records'] += len(traders)
        
        df = pd.DataFrame(traders)
        if not df.empty:
            df['timestamp'] = datetime.utcnow()
            df['venue'] = self.VENUE
        return df
    
    async def fetch_exchange_deposits_withdrawals(
        self, token_address: str, chain: str = 'ethereum', timeframe: str = '7d'
    ) -> pd.DataFrame:
        """Fetch exchange deposit/withdrawal data for a token."""
        data = await self._make_request(f'token/{token_address}/exchange-flows',
                                         params={'chain': chain, 'timeframe': timeframe})
        
        flows = []
        if data and 'flows' in data:
            for f in data['flows']:
                flow = ExchangeFlow(
                    timestamp=pd.to_datetime(f.get('date')), token_address=token_address, chain=chain,
                    exchange=f.get('exchange'), deposits=float(f.get('deposits', 0)),
                    deposits_usd=float(f.get('deposits_usd', 0)), withdrawals=float(f.get('withdrawals', 0)),
                    withdrawals_usd=float(f.get('withdrawals_usd', 0)), net_flow=float(f.get('net_flow', 0)),
                    net_flow_usd=float(f.get('net_flow_usd', 0)), unique_depositors=f.get('unique_depositors', 0),
                    unique_withdrawers=f.get('unique_withdrawers', 0), timeframe=timeframe,
                )
                flows.append(flow.to_dict())
            self.collection_stats['records'] += len(flows)
        
        df = pd.DataFrame(flows)
        if not df.empty:
            df['venue'] = self.VENUE
        return df.sort_values('timestamp').reset_index(drop=True) if not df.empty else df
    
    async def fetch_wallet_profit_leaderboard(
        self, token_address: str, chain: str = 'ethereum', limit: int = 50
    ) -> pd.DataFrame:
        """Fetch profit leaderboard for a token's holders."""
        data = await self._make_request(f'token/{token_address}/profit-leaderboard',
                                         params={'chain': chain, 'limit': limit})
        
        leaderboard = []
        if data and 'wallets' in data:
            for i, w in enumerate(data['wallets']):
                leaderboard.append({
                    'timestamp': datetime.utcnow(), 'token_address': token_address, 'chain': chain,
                    'rank': i + 1, 'address': w.get('address'), 'labels': w.get('labels', []),
                    'entity_name': w.get('entity_name'), 'total_bought': float(w.get('total_bought', 0)),
                    'total_sold': float(w.get('total_sold', 0)), 'current_holdings': float(w.get('current_holdings', 0)),
                    'cost_basis_usd': float(w.get('cost_basis', 0)), 'realized_pnl_usd': float(w.get('realized_pnl', 0)),
                    'unrealized_pnl_usd': float(w.get('unrealized_pnl', 0)), 'total_pnl_usd': float(w.get('total_pnl', 0)),
                    'roi_pct': float(w.get('roi', 0)), 'venue': self.VENUE,
                })
            self.collection_stats['records'] += len(leaderboard)
        
        return pd.DataFrame(leaderboard)
    
    async def fetch_new_token_buyers(
        self, token_address: str, chain: str = 'ethereum',
        timeframe: str = '24h', limit: int = 100
    ) -> pd.DataFrame:
        """Fetch new buyers of a token with their profiles."""
        data = await self._make_request(f'token/{token_address}/new-buyers', params={
            'chain': chain, 'timeframe': timeframe, 'limit': limit
        })
        
        buyers = []
        if data and 'buyers' in data:
            for b in data['buyers']:
                buyers.append({
                    'timestamp': pd.to_datetime(b.get('buy_time')), 'token_address': token_address, 'chain': chain,
                    'buyer_address': b.get('address'), 'labels': b.get('labels', []),
                    'is_smart_money': b.get('is_smart_money', False), 'amount_bought': float(b.get('amount', 0)),
                    'value_usd': float(b.get('value_usd', 0)), 'buy_price': float(b.get('price', 0)),
                    'historical_roi_avg': float(b.get('historical_roi', 0)), 'win_rate': float(b.get('win_rate', 0)),
                    'trade_count': b.get('trade_count', 0), 'timeframe': timeframe, 'venue': self.VENUE,
                })
            self.collection_stats['records'] += len(buyers)
        
        df = pd.DataFrame(buyers)
        return df.sort_values('timestamp', ascending=False).reset_index(drop=True) if not df.empty else df
    
    async def fetch_comprehensive_token_analysis(
        self, token_address: str, chain: str = 'ethereum'
    ) -> Dict[str, pd.DataFrame]:
        """Fetch comprehensive analysis for a token."""
        self.logger.info(f"Fetching comprehensive analysis for {token_address[:10]}...")
        
        return {
            'holders': await self.fetch_token_holders_analysis(token_address, chain),
            'smart_money_flows': await self.fetch_token_smart_money_flows(token_address, chain),
            'exchange_flows': await self.fetch_exchange_deposits_withdrawals(token_address, chain),
            'profit_leaderboard': await self.fetch_wallet_profit_leaderboard(token_address, chain),
            'new_buyers': await self.fetch_new_token_buyers(token_address, chain),
        }
    
    async def fetch_funding_rates(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Nansen doesn't provide funding rate data."""
        return pd.DataFrame()
    
    async def fetch_ohlcv(self, symbols: List[str], timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Nansen doesn't provide OHLCV data."""
        return pd.DataFrame()
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {**self.collection_stats, 'venue': self.VENUE}
    
    @staticmethod
    def get_supported_chains() -> List[str]:
        """Get list of supported chains."""
        return [c.value for c in Chain]
    
    @staticmethod
    def get_label_categories() -> List[str]:
        """Get list of label categories."""
        return [c.value for c in LabelCategory]

    # =========================================================================
    # Standardized Collection Methods (for dynamic routing in collection_manager)
    # =========================================================================

    async def collect_wallet_analytics(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect wallet analytics data (standardized interface).

        Wraps fetch_token_holders_analysis() to match collection_manager expectations.

        Args:
            symbols: List of token symbols to analyze
            start_date: Start date (not used - wallet data is current snapshot)
            end_date: End date (not used - wallet data is current snapshot)
            **kwargs: Additional parameters (chain, min_balance)

        Returns:
            DataFrame with wallet holder analytics for specified tokens
        """
        try:
            chain = kwargs.get('chain', 'ethereum')

            self.logger.info(f"Nansen: Collecting wallet_analytics for {len(symbols)} symbols on {chain}")

            # PARALLELIZED: Fetch all symbols concurrently
            async def _fetch_single_holder_analysis(symbol: str) -> Optional[pd.DataFrame]:
                try:
                    holder_data = await self.fetch_token_holders_analysis(
                        token_address=symbol,
                        chain=chain
                    )
                    if holder_data and not holder_data.empty:
                        holder_data['symbol'] = symbol
                        return holder_data
                except Exception as e:
                    self.logger.warning(f"Nansen: Failed to fetch wallet_analytics for {symbol}: {e}")
                return None

            tasks = [_fetch_single_holder_analysis(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

            if all_data:
                df = pd.concat(all_data, ignore_index=True)
                self.logger.info(f"Nansen: Collected wallet_analytics for {len(df)} holder records")
                return df

            self.logger.warning(f"Nansen: No wallet_analytics found for symbols {symbols}")
            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Nansen collect_wallet_analytics error: {e}")
            return pd.DataFrame()

    async def collect_smart_money(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect smart money data (standardized interface).

        Wraps fetch_smart_money_wallets() to match collection_manager expectations.

        Args:
            symbols: List of symbols (not used - fetches all smart money wallets)
            start_date: Start date (not used - smart money data is current snapshot)
            end_date: End date (not used - smart money data is current snapshot)
            **kwargs: Additional parameters (chain, category, min_balance)

        Returns:
            DataFrame with smart money wallet data
        """
        try:
            chain = kwargs.get('chain', 'ethereum')
            category = kwargs.get('category', 'defi')
            timeframe = kwargs.get('timeframe', '30d')
            limit = kwargs.get('limit', 100)

            self.logger.info(f"Nansen: Collecting smart_money data on {chain}")

            # Fetch smart money wallets
            smart_money_data = await self.fetch_smart_money_wallets(
                chain=chain,
                category=category,
                timeframe=timeframe,
                limit=limit
            )

            if smart_money_data is not None and not smart_money_data.empty:
                self.logger.info(f"Nansen: Collected smart_money data for {len(smart_money_data)} wallets")
                return smart_money_data

            self.logger.warning("Nansen: No smart_money data found")
            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Nansen collect_smart_money error: {e}")
            return pd.DataFrame()

    async def close(self) -> None:
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

async def test_nansen():
    """Test Nansen collector."""
    config = {'nansen_api_key': ''}
    collector = NansenCollector(config)
    try:
        print(f"Supported chains: {collector.get_supported_chains()}")
        print(f"Label categories: {collector.get_label_categories()}")
        
        # Test dataclasses
        wallet = SmartMoneyWallet(
            address='0x123', chain='ethereum', labels=['smart_money'], category='defi',
            total_pnl_usd=50000, win_rate=0.65, trade_count=100, roi_pct=80
        )
        print(f"Trader tier: {wallet.trader_tier.value}, follow: {wallet.follow_worthiness}")
        
        flow = SmartMoneyFlow(
            timestamp=datetime.utcnow(), token_address='0xabc', chain='ethereum',
            smart_money_net=100000, smart_money_buyers=50, smart_money_sellers=20, sm_pct_of_volume=25
        )
        print(f"Flow signal: {flow.flow_signal.value}, sentiment: {flow.sentiment}")
    finally:
        await collector.close()

if __name__ == '__main__':
    asyncio.run(test_nansen())