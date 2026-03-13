"""
Arkham Intelligence Collector - Wallet Tracking and Entity Intelligence

validated collector for blockchain intelligence data including
wallet tracking, entity labels, flow analysis, and whale monitoring.

===============================================================================
OVERVIEW
===============================================================================

Arkham Intelligence is a leading blockchain analytics platform providing:
    - Address labeling and entity identification
    - Wallet portfolio tracking
    - Transaction history with entity context
    - Flow analysis between entities
    - Exchange flow monitoring
    - Whale wallet tracking
    - Smart money identification

Target Users:
    - Institutional traders
    - Risk and compliance teams
    - Research analysts
    - Market makers

Key Differentiators:
    - Comprehensive address labeling (millions of addresses)
    - Real-time entity tracking
    - Cross-chain intelligence
    - Proprietary entity classification

===============================================================================
API TIERS
===============================================================================

    ============== ==================== ============== ================
    Tier Rate Limit Features Best For
    ============== ==================== ============== ================
    Free 100 req/min Basic labels Evaluation
    Analyst 500 req/min Full labels Research
    Professional 1000 req/min + Alerts Production
    Enterprise Custom + API feeds Institutional
    ============== ==================== ============== ================

===============================================================================
DATA TYPES COLLECTED
===============================================================================

Entity Information:
    - Entity name and type classification
    - Labels and tags
    - Social links (Twitter, website)
    - Risk scoring
    - Activity timestamps

Portfolio Data:
    - Token holdings with USD values
    - Portfolio composition
    - Historical balance changes

Transaction History:
    - Full transaction details
    - Sender/receiver entity labels
    - Value in native and USD
    - Gas metrics

Flow Analysis:
    - Inflows and outflows by entity
    - Cross-entity fund movements
    - Time-based aggregations

Exchange Flows:
    - Exchange deposit/withdrawal tracking
    - Reserve monitoring
    - Net flow calculations

Whale Alerts:
    - Large transaction monitoring
    - Significance scoring
    - Real-time alerts

Smart Money:
    - High-performance wallet tracking
    - PnL analysis
    - Win rate metrics

===============================================================================
SUPPORTED CHAINS
===============================================================================

    - Ethereum (primary)
    - Bitcoin
    - Polygon
    - Arbitrum
    - Optimism
    - Avalanche
    - BSC
    - Solana
    - Tron
    - Fantom

===============================================================================
USAGE EXAMPLES
===============================================================================

Entity lookup:

    >>> from data_collection.onchain import ArkhamCollector
    >>> 
    >>> collector = ArkhamCollector({'arkham_api_key': 'your-key'})
    >>> try:
    ... entity = await collector.fetch_entity_info(
    ... address='0x28C6c06298d514Db089934071355E5743bf21d60',
    ... chain='ethereum'
    ... )
    ... print(f"Entity: {entity.entity_name} ({entity.entity_type})")
    ... finally:
    ... await collector.close()

Exchange flow monitoring:

    >>> flows = await collector.fetch_exchange_flows(
    ... exchanges=['Binance', 'Coinbase'],
    ... tokens=['ETH', 'USDT'],
    ... timeframe='24h'
    ... )
    >>> for flow in flows:
    ... print(f"{flow.exchange}: Net {flow.netflow_usd:,.0f} USD")

Whale alerts:

    >>> alerts = await collector.fetch_whale_alerts(
    ... chain='ethereum',
    ... min_value_usd=5_000_000,
    ... limit=50
    ... )

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

Market Sentiment:
    - Track smart money positioning
    - Monitor exchange flows for accumulation/distribution
    - Identify whale accumulation patterns

Risk Management:
    - Monitor counterparty exposure
    - Track exchange reserves
    - Identify suspicious flow patterns

Alpha Generation:
    - Follow successful traders
    - Detect early accumulation signals
    - Monitor institutional movements

Due Diligence:
    - Verify counterparty identity
    - Assess wallet risk scores
    - Track entity transaction history

===============================================================================
DATA QUALITY CONSIDERATIONS
===============================================================================

- Entity labels are probabilistic (confidence scores provided)
- Some addresses may have multiple labels
- New addresses may lack labels initially
- Cross-chain tracking may have gaps
- Historical data availability varies

Version: 2.0.0
API Documentation: https://docs.arkhamintelligence.com/
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

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================

class Chain(Enum):
    """Supported blockchain networks."""
    ETHEREUM = 'ethereum'
    BITCOIN = 'bitcoin'
    POLYGON = 'polygon'
    ARBITRUM = 'arbitrum'
    OPTIMISM = 'optimism'
    AVALANCHE = 'avalanche'
    BSC = 'bsc'
    SOLANA = 'solana'
    TRON = 'tron'
    FANTOM = 'fantom'
    BASE = 'base'

class EntityType(Enum):
    """Entity classification types."""
    EXCHANGE = 'exchange'
    DEFI_PROTOCOL = 'defi_protocol'
    FUND = 'fund'
    WHALE = 'whale'
    SMART_MONEY = 'smart_money'
    MINER = 'miner'
    FOUNDATION = 'foundation'
    GOVERNMENT = 'government'
    MIXER = 'mixer'
    BRIDGE = 'bridge'
    CUSTODIAN = 'custodian'
    MARKET_MAKER = 'market_maker'
    VC = 'vc'
    NFT_TRADER = 'nft_trader'
    UNKNOWN = 'unknown'

class FlowDirection(Enum):
    """Flow direction for fund movements."""
    INFLOW = 'inflow'
    OUTFLOW = 'outflow'
    BOTH = 'both'

class Timeframe(Enum):
    """Timeframes for flow analysis."""
    ONE_HOUR = '1h'
    FOUR_HOUR = '4h'
    TWENTY_FOUR_HOUR = '24h'
    SEVEN_DAY = '7d'
    THIRTY_DAY = '30d'

class RiskLevel(Enum):
    """Risk classification for addresses."""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'
    UNKNOWN = 'unknown'

class AlertSignificance(Enum):
    """Whale alert significance levels."""
    NORMAL = 'normal'
    ELEVATED = 'elevated'
    HIGH = 'high'
    CRITICAL = 'critical'

class SmartMoneyCategory(Enum):
    """Smart money wallet categories."""
    DEFI = 'defi'
    NFT = 'nft'
    TRADING = 'trading'
    VC = 'vc'
    ALL = 'all'

class TransactionType(Enum):
    """Transaction type classification."""
    TRANSFER = 'transfer'
    SWAP = 'swap'
    DEPOSIT = 'deposit'
    WITHDRAWAL = 'withdrawal'
    CONTRACT_CALL = 'contract_call'
    NFT_TRANSFER = 'nft_transfer'
    BRIDGE = 'bridge'
    STAKE = 'stake'
    UNSTAKE = 'unstake'

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class ArkhamEntity:
    """Entity information from Arkham Intelligence."""
    address: str
    chain: str
    entity_name: Optional[str] = None
    entity_type: str = 'unknown'
    labels: List[str] = field(default_factory=list)
    twitter: Optional[str] = None
    website: Optional[str] = None
    description: Optional[str] = None
    risk_score: Optional[float] = None
    first_seen: Optional[datetime] = None
    last_active: Optional[datetime] = None
    total_received_usd: Optional[float] = None
    total_sent_usd: Optional[float] = None
    is_exchange: bool = False
    is_smart_contract: bool = False
    confidence_score: float = 0
    
    @property
    def entity_type_enum(self) -> EntityType:
        """Get entity type as enum."""
        try:
            return EntityType(self.entity_type.lower())
        except ValueError:
            return EntityType.UNKNOWN
    
    @property
    def risk_level(self) -> RiskLevel:
        """Classify risk based on score."""
        if self.risk_score is None:
            return RiskLevel.UNKNOWN
        if self.risk_score < 25:
            return RiskLevel.LOW
        elif self.risk_score < 50:
            return RiskLevel.MEDIUM
        elif self.risk_score < 75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    @property
    def is_labeled(self) -> bool:
        """Check if entity has been labeled."""
        return self.entity_name is not None or len(self.labels) > 0
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if label is high confidence (>80%)."""
        return self.confidence_score >= 0.8
    
    @property
    def net_flow_usd(self) -> float:
        """Net USD flow (received - sent)."""
        received = self.total_received_usd or 0
        sent = self.total_sent_usd or 0
        return received - sent
    
    @property
    def is_accumulating(self) -> bool:
        """Check if entity is net accumulating."""
        return self.net_flow_usd > 0
    
    @property
    def days_since_active(self) -> Optional[int]:
        """Days since last activity."""
        if self.last_active:
            return (datetime.now(timezone.utc) - self.last_active).days
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'address': self.address,
            'chain': self.chain,
            'entity_name': self.entity_name,
            'entity_type': self.entity_type,
            'entity_type_enum': self.entity_type_enum.value,
            'labels': self.labels,
            'twitter': self.twitter,
            'website': self.website,
            'risk_score': self.risk_score,
            'risk_level': self.risk_level.value,
            'is_labeled': self.is_labeled,
            'is_high_confidence': self.is_high_confidence,
            'is_exchange': self.is_exchange,
            'is_smart_contract': self.is_smart_contract,
            'confidence_score': self.confidence_score,
            'total_received_usd': self.total_received_usd,
            'total_sent_usd': self.total_sent_usd,
            'net_flow_usd': self.net_flow_usd,
            'is_accumulating': self.is_accumulating,
            'first_seen': self.first_seen.isoformat() if self.first_seen else None,
            'last_active': self.last_active.isoformat() if self.last_active else None,
            'days_since_active': self.days_since_active,
            'venue': 'arkham',
        }

@dataclass
class ArkhamHolding:
    """Token holding in a wallet."""
    timestamp: datetime
    address: str
    chain: str
    token_address: Optional[str]
    token_symbol: str
    token_name: Optional[str]
    balance: float
    balance_usd: float
    price_usd: float
    pct_of_portfolio: float
    
    @property
    def is_stablecoin(self) -> bool:
        """Check if holding is a stablecoin."""
        stables = {'USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'FRAX', 'LUSD'}
        return self.token_symbol.upper() in stables
    
    @property
    def is_native(self) -> bool:
        """Check if native token (ETH, BTC, etc.)."""
        natives = {'ETH', 'BTC', 'MATIC', 'AVAX', 'SOL', 'BNB', 'FTM'}
        return self.token_symbol.upper() in natives
    
    @property
    def is_significant(self) -> bool:
        """Check if holding is significant (>5% of portfolio)."""
        return self.pct_of_portfolio >= 5.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'address': self.address,
            'chain': self.chain,
            'token_address': self.token_address,
            'token_symbol': self.token_symbol,
            'token_name': self.token_name,
            'balance': self.balance,
            'balance_usd': self.balance_usd,
            'price_usd': self.price_usd,
            'pct_of_portfolio': self.pct_of_portfolio,
            'is_stablecoin': self.is_stablecoin,
            'is_native': self.is_native,
            'is_significant': self.is_significant,
            'venue': 'arkham',
        }

@dataclass
class ArkhamTransaction:
    """Transaction with entity context."""
    timestamp: datetime
    tx_hash: str
    chain: str
    from_address: str
    from_entity: Optional[str]
    from_entity_type: Optional[str]
    to_address: str
    to_entity: Optional[str]
    to_entity_type: Optional[str]
    value: float
    value_usd: float
    token_symbol: str
    token_address: Optional[str]
    gas_used: Optional[int]
    gas_price_gwei: Optional[float]
    tx_type: str = 'transfer'
    
    @property
    def tx_type_enum(self) -> TransactionType:
        """Get transaction type as enum."""
        try:
            return TransactionType(self.tx_type.lower())
        except ValueError:
            return TransactionType.TRANSFER
    
    @property
    def is_labeled_sender(self) -> bool:
        """Check if sender is labeled."""
        return self.from_entity is not None
    
    @property
    def is_labeled_receiver(self) -> bool:
        """Check if receiver is labeled."""
        return self.to_entity is not None
    
    @property
    def is_exchange_deposit(self) -> bool:
        """Check if transaction is exchange deposit."""
        if self.to_entity_type:
            return self.to_entity_type.lower() == 'exchange'
        return False
    
    @property
    def is_exchange_withdrawal(self) -> bool:
        """Check if transaction is exchange withdrawal."""
        if self.from_entity_type:
            return self.from_entity_type.lower() == 'exchange'
        return False
    
    @property
    def gas_cost_usd(self) -> float:
        """Estimated gas cost in USD (assumes ETH ~$3000)."""
        if self.gas_used and self.gas_price_gwei:
            eth_cost = (self.gas_used * self.gas_price_gwei) / 1e9
            return eth_cost * 3000 # Rough estimate
        return 0
    
    @property
    def is_whale_transaction(self) -> bool:
        """Check if whale-sized transaction (>$1M)."""
        return self.value_usd >= 1_000_000
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'tx_hash': self.tx_hash,
            'chain': self.chain,
            'from_address': self.from_address,
            'from_entity': self.from_entity,
            'from_entity_type': self.from_entity_type,
            'to_address': self.to_address,
            'to_entity': self.to_entity,
            'to_entity_type': self.to_entity_type,
            'value': self.value,
            'value_usd': self.value_usd,
            'token_symbol': self.token_symbol,
            'token_address': self.token_address,
            'tx_type': self.tx_type,
            'tx_type_enum': self.tx_type_enum.value,
            'is_labeled_sender': self.is_labeled_sender,
            'is_labeled_receiver': self.is_labeled_receiver,
            'is_exchange_deposit': self.is_exchange_deposit,
            'is_exchange_withdrawal': self.is_exchange_withdrawal,
            'is_whale_transaction': self.is_whale_transaction,
            'gas_used': self.gas_used,
            'gas_price_gwei': self.gas_price_gwei,
            'gas_cost_usd': self.gas_cost_usd,
            'venue': 'arkham',
        }

@dataclass
class ArkhamFlow:
    """Fund flow between entities."""
    timestamp: datetime
    entity: str
    chain: str
    counterparty: Optional[str]
    counterparty_type: Optional[str]
    direction: str
    token_symbol: str
    amount: float
    amount_usd: float
    tx_count: int
    timeframe: str
    
    @property
    def direction_enum(self) -> FlowDirection:
        """Get direction as enum."""
        try:
            return FlowDirection(self.direction.lower())
        except ValueError:
            return FlowDirection.BOTH
    
    @property
    def avg_tx_size(self) -> float:
        """Average transaction size in USD."""
        return self.amount_usd / self.tx_count if self.tx_count > 0 else 0
    
    @property
    def is_large_flow(self) -> bool:
        """Check if large flow (>$1M)."""
        return self.amount_usd >= 1_000_000
    
    @property
    def is_exchange_counterparty(self) -> bool:
        """Check if counterparty is an exchange."""
        if self.counterparty_type:
            return self.counterparty_type.lower() == 'exchange'
        return False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'entity': self.entity,
            'chain': self.chain,
            'counterparty': self.counterparty,
            'counterparty_type': self.counterparty_type,
            'direction': self.direction,
            'direction_enum': self.direction_enum.value,
            'token_symbol': self.token_symbol,
            'amount': self.amount,
            'amount_usd': self.amount_usd,
            'tx_count': self.tx_count,
            'avg_tx_size': self.avg_tx_size,
            'is_large_flow': self.is_large_flow,
            'is_exchange_counterparty': self.is_exchange_counterparty,
            'timeframe': self.timeframe,
            'venue': 'arkham',
        }

@dataclass
class ArkhamExchangeFlow:
    """Exchange flow metrics."""
    timestamp: datetime
    exchange: str
    chain: str
    token: str
    inflow: float
    inflow_usd: float
    outflow: float
    outflow_usd: float
    reserve: float
    reserve_usd: float
    timeframe: str
    
    @property
    def netflow(self) -> float:
        """Net flow in native token."""
        return self.inflow - self.outflow
    
    @property
    def netflow_usd(self) -> float:
        """Net flow in USD."""
        return self.inflow_usd - self.outflow_usd
    
    @property
    def is_net_inflow(self) -> bool:
        """Check if net inflow (bearish signal)."""
        return self.netflow > 0
    
    @property
    def is_net_outflow(self) -> bool:
        """Check if net outflow (bullish signal)."""
        return self.netflow < 0
    
    @property
    def flow_ratio(self) -> float:
        """Inflow/outflow ratio."""
        return self.inflow / self.outflow if self.outflow > 0 else float('inf')
    
    @property
    def reserve_change_pct(self) -> float:
        """Reserve change as percentage of reserve."""
        if self.reserve > 0:
            return self.netflow / self.reserve * 100
        return 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'exchange': self.exchange,
            'chain': self.chain,
            'token': self.token,
            'inflow': self.inflow,
            'inflow_usd': self.inflow_usd,
            'outflow': self.outflow,
            'outflow_usd': self.outflow_usd,
            'netflow': self.netflow,
            'netflow_usd': self.netflow_usd,
            'is_net_inflow': self.is_net_inflow,
            'is_net_outflow': self.is_net_outflow,
            'flow_ratio': self.flow_ratio,
            'reserve': self.reserve,
            'reserve_usd': self.reserve_usd,
            'reserve_change_pct': self.reserve_change_pct,
            'timeframe': self.timeframe,
            'venue': 'arkham',
        }

@dataclass
class ArkhamWhaleAlert:
    """Whale transaction alert."""
    timestamp: datetime
    tx_hash: str
    chain: str
    from_address: str
    from_entity: Optional[str]
    from_type: Optional[str]
    to_address: str
    to_entity: Optional[str]
    to_type: Optional[str]
    token: str
    amount: float
    value_usd: float
    alert_type: str
    significance: str
    
    @property
    def significance_enum(self) -> AlertSignificance:
        """Get significance as enum."""
        try:
            return AlertSignificance(self.significance.lower())
        except ValueError:
            return AlertSignificance.NORMAL
    
    @property
    def is_critical(self) -> bool:
        """Check if critical significance."""
        return self.significance_enum == AlertSignificance.CRITICAL
    
    @property
    def is_exchange_movement(self) -> bool:
        """Check if exchange-related movement."""
        from_ex = self.from_type and self.from_type.lower() == 'exchange'
        to_ex = self.to_type and self.to_type.lower() == 'exchange'
        return from_ex or to_ex
    
    @property
    def movement_direction(self) -> str:
        """Describe movement direction."""
        from_ex = self.from_type and self.from_type.lower() == 'exchange'
        to_ex = self.to_type and self.to_type.lower() == 'exchange'
        
        if from_ex and not to_ex:
            return 'exchange_withdrawal'
        elif to_ex and not from_ex:
            return 'exchange_deposit'
        elif from_ex and to_ex:
            return 'exchange_to_exchange'
        else:
            return 'wallet_to_wallet'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'tx_hash': self.tx_hash,
            'chain': self.chain,
            'from_address': self.from_address,
            'from_entity': self.from_entity,
            'from_type': self.from_type,
            'to_address': self.to_address,
            'to_entity': self.to_entity,
            'to_type': self.to_type,
            'token': self.token,
            'amount': self.amount,
            'value_usd': self.value_usd,
            'alert_type': self.alert_type,
            'significance': self.significance,
            'significance_enum': self.significance_enum.value,
            'is_critical': self.is_critical,
            'is_exchange_movement': self.is_exchange_movement,
            'movement_direction': self.movement_direction,
            'venue': 'arkham',
        }

@dataclass
class ArkhamSmartMoneyWallet:
    """Smart money wallet information."""
    timestamp: datetime
    address: str
    chain: str
    entity_name: Optional[str]
    category: str
    pnl_7d: float
    pnl_30d: float
    win_rate: float
    total_trades: int
    avg_hold_time_hours: float
    portfolio_value_usd: float
    followers: int
    
    @property
    def category_enum(self) -> SmartMoneyCategory:
        """Get category as enum."""
        try:
            return SmartMoneyCategory(self.category.lower())
        except ValueError:
            return SmartMoneyCategory.ALL
    
    @property
    def is_profitable_7d(self) -> bool:
        """Check if profitable in last 7 days."""
        return self.pnl_7d > 0
    
    @property
    def is_profitable_30d(self) -> bool:
        """Check if profitable in last 30 days."""
        return self.pnl_30d > 0
    
    @property
    def is_high_win_rate(self) -> bool:
        """Check if high win rate (>60%)."""
        return self.win_rate >= 0.6
    
    @property
    def is_active_trader(self) -> bool:
        """Check if active trader (>10 trades)."""
        return self.total_trades >= 10
    
    @property
    def avg_hold_time_days(self) -> float:
        """Average hold time in days."""
        return self.avg_hold_time_hours / 24
    
    @property
    def is_short_term_trader(self) -> bool:
        """Check if short-term trader (<24h avg hold)."""
        return self.avg_hold_time_hours < 24
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'address': self.address,
            'chain': self.chain,
            'entity_name': self.entity_name,
            'category': self.category,
            'category_enum': self.category_enum.value,
            'pnl_7d': self.pnl_7d,
            'pnl_30d': self.pnl_30d,
            'is_profitable_7d': self.is_profitable_7d,
            'is_profitable_30d': self.is_profitable_30d,
            'win_rate': self.win_rate,
            'is_high_win_rate': self.is_high_win_rate,
            'total_trades': self.total_trades,
            'is_active_trader': self.is_active_trader,
            'avg_hold_time_hours': self.avg_hold_time_hours,
            'avg_hold_time_days': self.avg_hold_time_days,
            'is_short_term_trader': self.is_short_term_trader,
            'portfolio_value_usd': self.portfolio_value_usd,
            'followers': self.followers,
            'venue': 'arkham',
        }

# =============================================================================
# Main Collector Class
# =============================================================================

class ArkhamCollector(BaseCollector):
    """
    Arkham Intelligence data collector for blockchain intelligence.
    
    validated implementation providing wallet tracking,
    entity labels, and flow analysis for market intelligence.
    
    Features:
        - Address labeling and entity identification
        - Wallet portfolio tracking
        - Transaction history with entity context
        - Flow analysis between entities
        - Exchange flow monitoring
        - Whale transaction alerts
        - Smart money wallet tracking
    
    Example:
        >>> collector = ArkhamCollector({'arkham_api_key': 'your-key'})
        >>> try:
        ... entity = await collector.fetch_entity_info(address, 'ethereum')
        ... flows = await collector.fetch_exchange_flows()
        ... finally:
        ... await collector.close()
    
    Attributes:
        VENUE: 'arkham'
        VENUE_TYPE: 'onchain'
    """
    
    VENUE = 'arkham'
    VENUE_TYPE = 'onchain'
    BASE_URL = 'https://api.arkhamintelligence.com/v1'
    
    # Major exchanges for flow tracking
    MAJOR_EXCHANGES = [
        'Binance', 'Coinbase', 'Kraken', 'OKX', 'Bybit',
        'Bitfinex', 'Gemini', 'KuCoin', 'Huobi', 'Gate.io'
    ]
    
    # Common tokens for flow tracking
    MAJOR_TOKENS = ['ETH', 'BTC', 'USDT', 'USDC', 'DAI']
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Arkham collector.
        
        Args:
            config: Configuration with arkham_api_key
        """
        config = config or {}
        super().__init__(config)

        self.api_key = config.get('arkham_api_key', '')
        self.session: Optional[aiohttp.ClientSession] = None

        # Initialize rate limiter (50 req/min conservative, 500 for Analyst)
        rate_limit = config.get('rate_limit', 50)
        self.rate_limiter = get_shared_rate_limiter(
            'arkham',
            rate=rate_limit,
            per=60,
            burst=config.get('burst', 20)
        )

        self.stats = {'requests': 0, 'records': 0, 'errors': 0}

        # Check if API key is valid (not a placeholder)
        self._api_key_valid = bool(self.api_key) and 'your_' not in self.api_key.lower()
        self._disabled_logged = False

        if not self._api_key_valid:
            logger.info("Arkham: No valid API key configured - collector disabled")
        else:
            logger.info("Arkham collector initialized with API key")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with auth headers."""
        if self.session is None or self.session.closed:
            headers = {'Accept': 'application/json', 'API-Key': self.api_key}
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self.session
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make authenticated API request."""
        # Skip if no valid API key
        if not self._api_key_valid:
            if not self._disabled_logged:
                logger.debug("Arkham: Skipping request - no valid API key")
                self._disabled_logged = True
            return None

        await self.rate_limiter.acquire()
        session = await self._get_session()
        self.stats['requests'] += 1

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 429:
                    logger.warning("Arkham: Rate limited, waiting 60s")
                    await asyncio.sleep(60)
                    return None
                elif resp.status == 401 or resp.status == 400:
                    # Auth error - mark as disabled
                    if not self._disabled_logged:
                        logger.info(f"Arkham: Authentication failed (HTTP {resp.status}) - requires valid API key")
                        self._disabled_logged = True
                        self._api_key_valid = False
                    return None
                else:
                    logger.debug(f"Arkham HTTP {resp.status} for {endpoint}")
                    self.stats['errors'] += 1
                    return None
        except Exception as e:
            if not self._disabled_logged:
                logger.debug(f"Arkham request error: {e}")
            self.stats['errors'] += 1
            return None
    
    async def fetch_entity_info(self, address: str, chain: str = 'ethereum') -> Optional[ArkhamEntity]:
        """Fetch entity information for an address."""
        data = await self._make_request('entity', params={'address': address, 'chain': chain})
        
        if data:
            return ArkhamEntity(
                address=address,
                chain=chain,
                entity_name=data.get('name'),
                entity_type=data.get('type', 'unknown'),
                labels=data.get('labels', []),
                twitter=data.get('twitter'),
                website=data.get('website'),
                description=data.get('description'),
                risk_score=data.get('risk_score'),
                first_seen=pd.to_datetime(data.get('first_seen')) if data.get('first_seen') else None,
                last_active=pd.to_datetime(data.get('last_active')) if data.get('last_active') else None,
                total_received_usd=data.get('total_received_usd'),
                total_sent_usd=data.get('total_sent_usd'),
                is_exchange=data.get('is_exchange', False),
                is_smart_contract=data.get('is_smart_contract', False),
                confidence_score=data.get('confidence', 0)
            )
        return None
    
    async def fetch_address_portfolio(self, address: str, chain: str = 'ethereum') -> pd.DataFrame:
        """Fetch current portfolio holdings for an address."""
        data = await self._make_request('portfolio', params={'address': address, 'chain': chain})
        
        records = []
        if data and 'tokens' in data:
            for token in data['tokens']:
                h = ArkhamHolding(
                    timestamp=datetime.now(timezone.utc),
                    address=address,
                    chain=chain,
                    token_address=token.get('contract_address'),
                    token_symbol=token.get('symbol', ''),
                    token_name=token.get('name'),
                    balance=float(token.get('balance', 0)),
                    balance_usd=float(token.get('value_usd', 0)),
                    price_usd=float(token.get('price_usd', 0)),
                    pct_of_portfolio=float(token.get('percentage', 0))
                )
                records.append(h.to_dict())
        
        self.stats['records'] += len(records)
        return pd.DataFrame(records)
    
    async def fetch_address_transactions(
        self,
        address: str,
        chain: str = 'ethereum',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch transaction history with entity context."""
        params = {'address': address, 'chain': chain, 'limit': min(limit, 1000)}
        
        if start_date:
            params['from'] = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        if end_date:
            params['to'] = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        
        records = []
        offset = 0
        
        while len(records) < limit:
            params['offset'] = offset
            data = await self._make_request('transactions', params=params)
            
            if not data or 'transactions' not in data:
                break
            
            txs = data['transactions']
            if not txs:
                break
            
            for tx in txs:
                t = ArkhamTransaction(
                    timestamp=pd.to_datetime(tx.get('timestamp'), unit='s', utc=True),
                    tx_hash=tx.get('hash', ''),
                    chain=chain,
                    from_address=tx.get('from', ''),
                    from_entity=tx.get('from_entity', {}).get('name'),
                    from_entity_type=tx.get('from_entity', {}).get('type'),
                    to_address=tx.get('to', ''),
                    to_entity=tx.get('to_entity', {}).get('name'),
                    to_entity_type=tx.get('to_entity', {}).get('type'),
                    value=float(tx.get('value', 0)),
                    value_usd=float(tx.get('value_usd', 0)),
                    token_symbol=tx.get('token_symbol', 'ETH'),
                    token_address=tx.get('token_address'),
                    gas_used=tx.get('gas_used'),
                    gas_price_gwei=tx.get('gas_price_gwei'),
                    tx_type=tx.get('type', 'transfer')
                )
                records.append(t.to_dict())
            
            offset += len(txs)
            if len(txs) < 1000:
                break
            await asyncio.sleep(0.5)
        
        self.stats['records'] += len(records)
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    async def fetch_entity_flows(
        self,
        entity_name: str,
        chain: str = 'ethereum',
        timeframe: str = '24h',
        direction: str = 'both'
    ) -> pd.DataFrame:
        """Fetch flow analysis for an entity."""
        params = {'entity': entity_name, 'chain': chain, 'timeframe': timeframe}
        data = await self._make_request('flows', params=params)
        
        records = []
        if data:
            if direction in ['both', 'inflow']:
                for flow in data.get('inflows', []):
                    f = ArkhamFlow(
                        timestamp=datetime.now(timezone.utc),
                        entity=entity_name,
                        chain=chain,
                        counterparty=flow.get('from_entity'),
                        counterparty_type=flow.get('from_type'),
                        direction='inflow',
                        token_symbol=flow.get('token', ''),
                        amount=float(flow.get('amount', 0)),
                        amount_usd=float(flow.get('amount_usd', 0)),
                        tx_count=flow.get('tx_count', 1),
                        timeframe=timeframe
                    )
                    records.append(f.to_dict())
            
            if direction in ['both', 'outflow']:
                for flow in data.get('outflows', []):
                    f = ArkhamFlow(
                        timestamp=datetime.now(timezone.utc),
                        entity=entity_name,
                        chain=chain,
                        counterparty=flow.get('to_entity'),
                        counterparty_type=flow.get('to_type'),
                        direction='outflow',
                        token_symbol=flow.get('token', ''),
                        amount=float(flow.get('amount', 0)),
                        amount_usd=float(flow.get('amount_usd', 0)),
                        tx_count=flow.get('tx_count', 1),
                        timeframe=timeframe
                    )
                    records.append(f.to_dict())
        
        self.stats['records'] += len(records)
        return pd.DataFrame(records)
    
    async def fetch_exchange_flows(
        self,
        exchanges: Optional[List[str]] = None,
        tokens: Optional[List[str]] = None,
        chain: str = 'ethereum',
        timeframe: str = '24h'
    ) -> pd.DataFrame:
        """Fetch exchange inflow/outflow data."""
        exchanges = exchanges or self.MAJOR_EXCHANGES
        tokens = tokens or self.MAJOR_TOKENS
        
        records = []
        
        for exchange in exchanges:
            data = await self._make_request(
                'exchange-flows',
                params={'exchange': exchange, 'chain': chain, 'timeframe': timeframe}
            )
            
            if data:
                for token in tokens:
                    token_data = data.get('tokens', {}).get(token, {})
                    f = ArkhamExchangeFlow(
                        timestamp=datetime.now(timezone.utc),
                        exchange=exchange,
                        chain=chain,
                        token=token,
                        inflow=float(token_data.get('inflow', 0)),
                        inflow_usd=float(token_data.get('inflow_usd', 0)),
                        outflow=float(token_data.get('outflow', 0)),
                        outflow_usd=float(token_data.get('outflow_usd', 0)),
                        reserve=float(token_data.get('reserve', 0)),
                        reserve_usd=float(token_data.get('reserve_usd', 0)),
                        timeframe=timeframe
                    )
                    records.append(f.to_dict())
            
            await asyncio.sleep(0.2)
        
        self.stats['records'] += len(records)
        return pd.DataFrame(records)
    
    async def fetch_whale_alerts(
        self,
        chain: str = 'ethereum',
        min_value_usd: float = 1_000_000,
        limit: int = 100
    ) -> pd.DataFrame:
        """Fetch recent whale transactions."""
        data = await self._make_request(
            'alerts',
            params={'chain': chain, 'min_value': min_value_usd, 'limit': limit, 'type': 'whale'}
        )
        
        records = []
        if data and 'alerts' in data:
            for alert in data['alerts']:
                a = ArkhamWhaleAlert(
                    timestamp=pd.to_datetime(alert.get('timestamp'), unit='s', utc=True),
                    tx_hash=alert.get('tx_hash', ''),
                    chain=chain,
                    from_address=alert.get('from', ''),
                    from_entity=alert.get('from_entity'),
                    from_type=alert.get('from_type'),
                    to_address=alert.get('to', ''),
                    to_entity=alert.get('to_entity'),
                    to_type=alert.get('to_type'),
                    token=alert.get('token', ''),
                    amount=float(alert.get('amount', 0)),
                    value_usd=float(alert.get('value_usd', 0)),
                    alert_type=alert.get('alert_type', ''),
                    significance=alert.get('significance', 'normal')
                )
                records.append(a.to_dict())
        
        self.stats['records'] += len(records)
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)
        return df
    
    async def fetch_smart_money_wallets(
        self,
        chain: str = 'ethereum',
        category: str = 'all',
        limit: int = 100
    ) -> pd.DataFrame:
        """Fetch list of smart money wallets."""
        params = {'chain': chain, 'limit': limit}
        if category != 'all':
            params['category'] = category
        
        data = await self._make_request('smart-money', params=params)
        
        records = []
        if data and 'wallets' in data:
            for wallet in data['wallets']:
                w = ArkhamSmartMoneyWallet(
                    timestamp=datetime.now(timezone.utc),
                    address=wallet.get('address', ''),
                    chain=chain,
                    entity_name=wallet.get('name'),
                    category=wallet.get('category', 'unknown'),
                    pnl_7d=float(wallet.get('pnl_7d', 0)),
                    pnl_30d=float(wallet.get('pnl_30d', 0)),
                    win_rate=float(wallet.get('win_rate', 0)),
                    total_trades=wallet.get('total_trades', 0),
                    avg_hold_time_hours=wallet.get('avg_hold_time', 0),
                    portfolio_value_usd=float(wallet.get('portfolio_value', 0)),
                    followers=wallet.get('followers', 0)
                )
                records.append(w.to_dict())
        
        self.stats['records'] += len(records)
        return pd.DataFrame(records)
    
    async def fetch_token_holders(
        self,
        token_address: str,
        chain: str = 'ethereum',
        limit: int = 100
    ) -> pd.DataFrame:
        """Fetch top holders of a token with entity labels."""
        data = await self._make_request(
            'token-holders',
            params={'token': token_address, 'chain': chain, 'limit': limit}
        )
        
        records = []
        if data and 'holders' in data:
            for i, holder in enumerate(data['holders']):
                records.append({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'token_address': token_address,
                    'chain': chain,
                    'rank': i + 1,
                    'holder_address': holder.get('address'),
                    'holder_entity': holder.get('entity_name'),
                    'holder_type': holder.get('entity_type', 'unknown'),
                    'balance': float(holder.get('balance', 0)),
                    'balance_usd': float(holder.get('value_usd', 0)),
                    'pct_of_supply': float(holder.get('percentage', 0)),
                    'is_contract': holder.get('is_contract', False),
                    'venue': self.VENUE,
                })
        
        self.stats['records'] += len(records)
        return pd.DataFrame(records)
    
    async def fetch_funding_rates(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Arkham doesn't provide funding rates - returns empty DataFrame."""
        logger.info("Arkham doesn't provide funding rate data")
        return pd.DataFrame()
    
    async def fetch_ohlcv(self, symbols: List[str], timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Arkham doesn't provide OHLCV data - returns empty DataFrame."""
        logger.info("Arkham doesn't provide OHLCV data")
        return pd.DataFrame()
    
    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info(f"Arkham session closed. Stats: {self.stats}")
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.stats.copy()
    
    @classmethod
    def get_supported_chains(cls) -> List[str]:
        """Get list of supported chains."""
        return [c.value for c in Chain]
    
    @classmethod
    def get_entity_types(cls) -> List[str]:
        """Get list of entity types."""
        return [e.value for e in EntityType]

async def test_arkham_collector():
    """Test Arkham collector functionality."""
    collector = ArkhamCollector({'rate_limit': 10})
    
    try:
        print("=" * 60)
        print("Arkham Collector Test")
        print("=" * 60)
        print(f"\nSupported chains: {len(ArkhamCollector.get_supported_chains())}")
        print(f"Entity types: {ArkhamCollector.get_entity_types()}")
        print(f"\nStats: {collector.get_collection_stats()}")
    finally:
        await collector.close()

if __name__ == '__main__':
    asyncio.run(test_arkham_collector())