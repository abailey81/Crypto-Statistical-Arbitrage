"""
Glassnode On-Chain Metrics Collector

validated collector for institutional on-chain data from Glassnode.
Comprehensive metrics for network health, exchange flows, supply dynamics,
and market indicators for Bitcoin, Ethereum, and major assets.

===============================================================================
GLASSNODE OVERVIEW
===============================================================================

Glassnode is the premier institutional on-chain analytics provider, offering
the most comprehensive Bitcoin and Ethereum metrics. Used by hedge funds,
trading desks, and researchers for market analysis.

Key Differentiators:
    - Deepest Bitcoin on-chain coverage (since 2009)
    - Entity-adjusted metrics (clustering analysis)
    - HODL waves and supply age bands
    - professional-quality derivatives data
    - Real-time alerts and thresholds

===============================================================================
API SPECIFICATIONS
===============================================================================

Base URL: https://api.glassnode.com/v1/metrics

Authentication:
    - API Key as query parameter
    - Keys obtained from Glassnode Studio

Rate Limits by Tier:
    ============ ============== ================ ===============
    Tier Requests/min Data Resolution Historical Data
    ============ ============== ================ ===============
    Free 10 24h only 2 years
    Starter 20 24h, 1h Full history
    Professional 60 24h, 1h, 10m Full history
    Enterprise Custom All Full history
    ============ ============== ================ ===============

Data Resolution:
    - 10m: 10-minute intervals (Professional+)
    - 1h: Hourly intervals (Starter+)
    - 24h: Daily intervals (All tiers)
    - 1w: Weekly intervals
    - 1month: Monthly intervals

===============================================================================
METRIC CATEGORIES
===============================================================================

Addresses:
    - active_count: Daily active addresses
    - new_non_zero_count: New addresses with balance
    - accumulation_count: Addresses accumulating

Blockchain:
    - block_count, block_height, utxo_count
    - block_interval_mean, block_size_mean

Distribution:
    - balance_exchanges: Exchange holdings
    - gini: Wealth concentration
    - supply_contracts: Smart contract supply

Indicators:
    - sopr, sopr_adjusted: Spent Output Profit Ratio
    - nupl: Net Unrealized Profit/Loss
    - mvrv, mvrv_z_score: Market Value to Realized Value
    - nvt, nvts: Network Value to Transactions
    - puell_multiple: Miner profitability
    - rhodl_ratio: Realized HODL Ratio

Market:
    - price_usd_close, marketcap_usd
    - realized_cap_usd, deltacap_usd
    - thermocap_usd

Mining:
    - hash_rate_mean, difficulty_latest
    - revenue_sum, thermocap_multiple

Supply:
    - current, issued, inflation_rate
    - liquid_sum, illiquid_sum
    - active_more_1y_percent (HODL waves)

Derivatives:
    - futures_open_interest_sum
    - futures_funding_rate_perpetual
    - futures_liquidated_volume_*

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

Valuation Signals:
    - MVRV z-score for cycle positioning
    - NVT for network value assessment
    - Puell Multiple for miner stress

Flow Analysis:
    - Exchange netflows for accumulation/distribution
    - Whale movements
    - Institutional holdings (Grayscale, ETFs)

Risk Management:
    - SOPR for profit-taking detection
    - NUPL for sentiment extremes
    - Supply in profit/loss ratios

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

class Asset(Enum):
    """Supported assets on Glassnode."""
    BTC = 'BTC'
    ETH = 'ETH'
    LTC = 'LTC'
    AAVE = 'AAVE'
    BNB = 'BNB'
    LINK = 'LINK'
    MKR = 'MKR'
    UNI = 'UNI'
    YFI = 'YFI'
    SNX = 'SNX'
    COMP = 'COMP'
    BAT = 'BAT'
    MANA = 'MANA'
    GRT = 'GRT'

class Resolution(Enum):
    """Data resolution options."""
    TEN_MINUTES = '10m'
    HOURLY = '1h'
    DAILY = '24h'
    WEEKLY = '1w'
    MONTHLY = '1month'

class MetricCategory(Enum):
    """Glassnode metric categories."""
    ADDRESSES = 'addresses'
    BLOCKCHAIN = 'blockchain'
    DISTRIBUTION = 'distribution'
    ENTITIES = 'entities'
    FEES = 'fees'
    INDICATORS = 'indicators'
    INSTITUTIONS = 'institutions'
    LIGHTNING = 'lightning'
    MARKET = 'market'
    MINING = 'mining'
    PROTOCOLS = 'protocols'
    SUPPLY = 'supply'
    TRANSACTIONS = 'transactions'
    DERIVATIVES = 'derivatives'

class MVRVZone(Enum):
    """MVRV valuation zones."""
    DEEP_VALUE = 'deep_value' # < 0.8
    UNDERVALUED = 'undervalued' # 0.8 - 1.0
    FAIR_VALUE = 'fair_value' # 1.0 - 2.4
    OVERVALUED = 'overvalued' # 2.4 - 3.7
    EXTREME_OVERVALUED = 'extreme' # > 3.7

class NUPLPhase(Enum):
    """NUPL market cycle phases."""
    CAPITULATION = 'capitulation' # < 0
    HOPE = 'hope' # 0 - 0.25
    OPTIMISM = 'optimism' # 0.25 - 0.5
    BELIEF = 'belief' # 0.5 - 0.75
    EUPHORIA = 'euphoria' # > 0.75

class SOPRSignal(Enum):
    """SOPR trading signals."""
    CAPITULATION = 'capitulation' # < 0.9
    LOSS_SELLING = 'loss_selling' # 0.9 - 0.97
    SUPPORT = 'support' # 0.97 - 1.03
    PROFIT_TAKING = 'profit_taking' # 1.03 - 1.1
    DISTRIBUTION = 'distribution' # > 1.1

class PuellZone(Enum):
    """Puell Multiple zones."""
    CAPITULATION = 'capitulation' # < 0.5
    UNDERVALUED = 'undervalued' # 0.5 - 0.8
    FAIR = 'fair' # 0.8 - 1.5
    OVERVALUED = 'overvalued' # 1.5 - 4.0
    EUPHORIA = 'euphoria' # > 4.0

class FlowDirection(Enum):
    """Exchange flow direction."""
    STRONG_INFLOW = 'strong_inflow'
    INFLOW = 'inflow'
    NEUTRAL = 'neutral'
    OUTFLOW = 'outflow'
    STRONG_OUTFLOW = 'strong_outflow'

class SupplyAge(Enum):
    """Supply age bands (HODL waves)."""
    ACTIVE_24H = '24h'
    ACTIVE_1D_1W = '1d_1w'
    ACTIVE_1W_1M = '1w_1m'
    ACTIVE_1M_3M = '1m_3m'
    ACTIVE_3M_6M = '3m_6m'
    ACTIVE_6M_12M = '6m_12m'
    ACTIVE_1Y_2Y = '1y_2y'
    ACTIVE_2Y_3Y = '2y_3y'
    ACTIVE_3Y_5Y = '3y_5y'
    ACTIVE_5Y_7Y = '5y_7y'
    ACTIVE_7Y_10Y = '7y_10y'
    ACTIVE_MORE_10Y = '10y_plus'

class NetworkHealth(Enum):
    """Network health assessment."""
    DECLINING = 'declining'
    WEAK = 'weak'
    STABLE = 'stable'
    GROWING = 'growing'
    BOOMING = 'booming'

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class OnChainMetric:
    """Generic on-chain metric with analytics."""
    timestamp: datetime
    asset: str
    category: str
    metric: str
    value: float
    resolution: str = '24h'
    
    @property
    def is_valid(self) -> bool:
        """Check if metric value is valid."""
        return self.value is not None and not np.isnan(self.value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'asset': self.asset, 'category': self.category, 'metric': self.metric,
            'value': self.value, 'resolution': self.resolution,
        }

@dataclass
class MVRVData:
    """MVRV (Market Value to Realized Value) data with analytics."""
    timestamp: datetime
    asset: str
    mvrv: float
    mvrv_z_score: Optional[float] = None
    market_cap: float = 0.0
    realized_cap: float = 0.0
    
    @property
    def zone(self) -> MVRVZone:
        """Classify MVRV zone."""
        if self.mvrv < 0.8:
            return MVRVZone.DEEP_VALUE
        elif self.mvrv < 1.0:
            return MVRVZone.UNDERVALUED
        elif self.mvrv < 2.4:
            return MVRVZone.FAIR_VALUE
        elif self.mvrv < 3.7:
            return MVRVZone.OVERVALUED
        return MVRVZone.EXTREME_OVERVALUED
    
    @property
    def is_undervalued(self) -> bool:
        """Check if undervalued."""
        return self.mvrv < 1.0
    
    @property
    def is_overvalued(self) -> bool:
        """Check if overvalued."""
        return self.mvrv > 2.4
    
    @property
    def is_extreme(self) -> bool:
        """Check if at extreme levels."""
        return self.mvrv < 0.8 or self.mvrv > 3.7
    
    @property
    def position_signal(self) -> str:
        """Position sizing signal."""
        if self.mvrv < 0.8:
            return 'max_long'
        elif self.mvrv < 1.0:
            return 'accumulate'
        elif self.mvrv < 2.4:
            return 'hold'
        elif self.mvrv < 3.7:
            return 'reduce'
        return 'max_short'
    
    @property
    def unrealized_profit_pct(self) -> float:
        """Percentage of unrealized profit in market."""
        return (self.mvrv - 1.0) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'asset': self.asset, 'mvrv': self.mvrv, 'mvrv_z_score': self.mvrv_z_score,
            'zone': self.zone.value, 'position_signal': self.position_signal,
            'is_undervalued': self.is_undervalued, 'is_overvalued': self.is_overvalued,
        }

@dataclass
class NUPLData:
    """NUPL (Net Unrealized Profit/Loss) data with cycle analysis."""
    timestamp: datetime
    asset: str
    nupl: float
    
    @property
    def phase(self) -> NUPLPhase:
        """Classify market cycle phase."""
        if self.nupl < 0:
            return NUPLPhase.CAPITULATION
        elif self.nupl < 0.25:
            return NUPLPhase.HOPE
        elif self.nupl < 0.5:
            return NUPLPhase.OPTIMISM
        elif self.nupl < 0.75:
            return NUPLPhase.BELIEF
        return NUPLPhase.EUPHORIA
    
    @property
    def is_capitulation(self) -> bool:
        """Check if in capitulation."""
        return self.nupl < 0
    
    @property
    def is_euphoria(self) -> bool:
        """Check if in euphoria."""
        return self.nupl > 0.75
    
    @property
    def risk_level(self) -> str:
        """Market risk level."""
        if self.nupl < 0:
            return 'low'
        elif self.nupl < 0.5:
            return 'moderate'
        elif self.nupl < 0.75:
            return 'elevated'
        return 'extreme'
    
    @property
    def cycle_position(self) -> float:
        """Normalized cycle position (0-1)."""
        return min(1.0, max(0.0, (self.nupl + 0.25) / 1.0))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'asset': self.asset, 'nupl': self.nupl, 'phase': self.phase.value,
            'risk_level': self.risk_level, 'cycle_position': self.cycle_position,
        }

@dataclass
class SOPRData:
    """SOPR (Spent Output Profit Ratio) data with trading signals."""
    timestamp: datetime
    asset: str
    sopr: float
    sopr_adjusted: Optional[float] = None
    
    @property
    def signal(self) -> SOPRSignal:
        """Classify SOPR signal."""
        v = self.sopr_adjusted if self.sopr_adjusted else self.sopr
        if v < 0.9:
            return SOPRSignal.CAPITULATION
        elif v < 0.97:
            return SOPRSignal.LOSS_SELLING
        elif v <= 1.03:
            return SOPRSignal.SUPPORT
        elif v <= 1.1:
            return SOPRSignal.PROFIT_TAKING
        return SOPRSignal.DISTRIBUTION
    
    @property
    def is_profitable(self) -> bool:
        """Check if market is profitable."""
        return self.sopr > 1.0
    
    @property
    def is_at_support(self) -> bool:
        """Check if at SOPR=1 support."""
        return 0.97 <= self.sopr <= 1.03
    
    @property
    def is_capitulating(self) -> bool:
        """Check if capitulating."""
        return self.sopr < 0.95
    
    @property
    def deviation_pct(self) -> float:
        """Deviation from breakeven."""
        return (self.sopr - 1.0) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'asset': self.asset, 'sopr': self.sopr, 'sopr_adjusted': self.sopr_adjusted,
            'signal': self.signal.value, 'is_profitable': self.is_profitable,
            'is_at_support': self.is_at_support, 'deviation_pct': self.deviation_pct,
        }

@dataclass
class ExchangeFlowData:
    """Exchange flow data with sentiment analysis."""
    timestamp: datetime
    asset: str
    inflow: float = 0.0
    outflow: float = 0.0
    netflow: float = 0.0
    balance: float = 0.0
    
    @property
    def flow_direction(self) -> FlowDirection:
        """Classify flow direction."""
        if self.netflow == 0:
            return FlowDirection.NEUTRAL
        
        pct = (self.netflow / self.balance * 100) if self.balance > 0 else 0
        
        if pct > 1.0:
            return FlowDirection.STRONG_INFLOW
        elif pct > 0.2:
            return FlowDirection.INFLOW
        elif pct < -1.0:
            return FlowDirection.STRONG_OUTFLOW
        elif pct < -0.2:
            return FlowDirection.OUTFLOW
        return FlowDirection.NEUTRAL
    
    @property
    def is_accumulation(self) -> bool:
        """Check if accumulation (outflows)."""
        return self.netflow < 0
    
    @property
    def is_distribution(self) -> bool:
        """Check if distribution (inflows)."""
        return self.netflow > 0
    
    @property
    def netflow_pct(self) -> float:
        """Netflow as percentage of balance."""
        if self.balance > 0:
            return (self.netflow / self.balance) * 100
        return 0.0
    
    @property
    def sentiment(self) -> str:
        """Market sentiment from flows."""
        if self.netflow < 0:
            return 'bullish'
        elif self.netflow > 0:
            return 'bearish'
        return 'neutral'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'asset': self.asset, 'inflow': self.inflow, 'outflow': self.outflow,
            'netflow': self.netflow, 'balance': self.balance,
            'flow_direction': self.flow_direction.value, 'sentiment': self.sentiment,
            'is_accumulation': self.is_accumulation,
        }

@dataclass
class SupplyData:
    """Supply distribution data with HODL wave analysis."""
    timestamp: datetime
    asset: str
    total_supply: float
    circulating_supply: float
    liquid_supply: float = 0.0
    illiquid_supply: float = 0.0
    supply_in_profit: float = 0.0
    supply_in_loss: float = 0.0
    supply_active_1y_plus: float = 0.0
    
    @property
    def illiquid_ratio(self) -> float:
        """Ratio of illiquid supply."""
        if self.circulating_supply > 0:
            return self.illiquid_supply / self.circulating_supply
        return 0.0
    
    @property
    def profit_ratio(self) -> float:
        """Ratio of supply in profit."""
        total = self.supply_in_profit + self.supply_in_loss
        if total > 0:
            return self.supply_in_profit / total
        return 0.5
    
    @property
    def hodler_ratio(self) -> float:
        """Ratio of long-term holders (1y+)."""
        if self.circulating_supply > 0:
            return self.supply_active_1y_plus / self.circulating_supply
        return 0.0
    
    @property
    def market_sentiment(self) -> str:
        """Market sentiment from supply metrics."""
        if self.profit_ratio > 0.9:
            return 'extreme_greed'
        elif self.profit_ratio > 0.7:
            return 'greed'
        elif self.profit_ratio > 0.5:
            return 'neutral'
        elif self.profit_ratio > 0.3:
            return 'fear'
        return 'extreme_fear'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'asset': self.asset, 'total_supply': self.total_supply,
            'illiquid_ratio': self.illiquid_ratio, 'profit_ratio': self.profit_ratio,
            'hodler_ratio': self.hodler_ratio, 'market_sentiment': self.market_sentiment,
        }

@dataclass
class MiningData:
    """Mining metrics with miner behavior analysis."""
    timestamp: datetime
    asset: str
    hash_rate: float
    difficulty: float
    miner_revenue: float = 0.0
    puell_multiple: float = 1.0
    
    @property
    def puell_zone(self) -> PuellZone:
        """Classify Puell Multiple zone."""
        if self.puell_multiple < 0.5:
            return PuellZone.CAPITULATION
        elif self.puell_multiple < 0.8:
            return PuellZone.UNDERVALUED
        elif self.puell_multiple < 1.5:
            return PuellZone.FAIR
        elif self.puell_multiple < 4.0:
            return PuellZone.OVERVALUED
        return PuellZone.EUPHORIA
    
    @property
    def is_miner_capitulation(self) -> bool:
        """Check if miners are capitulating."""
        return self.puell_multiple < 0.5
    
    @property
    def is_miner_euphoria(self) -> bool:
        """Check if miners in euphoria."""
        return self.puell_multiple > 4.0
    
    @property
    def miner_health(self) -> str:
        """Miner health assessment."""
        if self.puell_multiple < 0.5:
            return 'critical'
        elif self.puell_multiple < 0.8:
            return 'stressed'
        elif self.puell_multiple < 2.0:
            return 'healthy'
        return 'euphoric'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'asset': self.asset, 'hash_rate': self.hash_rate, 'difficulty': self.difficulty,
            'puell_multiple': self.puell_multiple, 'puell_zone': self.puell_zone.value,
            'miner_health': self.miner_health,
        }

@dataclass
class DerivativesData:
    """Derivatives market data."""
    timestamp: datetime
    asset: str
    futures_oi: float = 0.0
    futures_volume: float = 0.0
    funding_rate: float = 0.0
    liquidations_long: float = 0.0
    liquidations_short: float = 0.0
    
    @property
    def total_liquidations(self) -> float:
        """Total liquidations."""
        return self.liquidations_long + self.liquidations_short
    
    @property
    def liquidation_bias(self) -> str:
        """Liquidation direction bias."""
        if self.liquidations_long > self.liquidations_short * 1.5:
            return 'long_squeeze'
        elif self.liquidations_short > self.liquidations_long * 1.5:
            return 'short_squeeze'
        return 'balanced'
    
    @property
    def funding_sentiment(self) -> str:
        """Sentiment from funding rate."""
        if self.funding_rate > 0.01:
            return 'extreme_long'
        elif self.funding_rate > 0.001:
            return 'long_bias'
        elif self.funding_rate < -0.01:
            return 'extreme_short'
        elif self.funding_rate < -0.001:
            return 'short_bias'
        return 'neutral'
    
    @property
    def leverage_risk(self) -> str:
        """Leverage risk assessment."""
        if self.total_liquidations > 500_000_000:
            return 'extreme'
        elif self.total_liquidations > 100_000_000:
            return 'high'
        elif self.total_liquidations > 50_000_000:
            return 'elevated'
        return 'normal'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'asset': self.asset, 'futures_oi': self.futures_oi, 'funding_rate': self.funding_rate,
            'total_liquidations': self.total_liquidations, 'liquidation_bias': self.liquidation_bias,
            'funding_sentiment': self.funding_sentiment, 'leverage_risk': self.leverage_risk,
        }

@dataclass
class AddressData:
    """Address activity metrics."""
    timestamp: datetime
    asset: str
    active_count: int
    sending_count: int = 0
    receiving_count: int = 0
    new_count: int = 0
    
    @property
    def activity_ratio(self) -> float:
        """Ratio of sending to receiving."""
        if self.receiving_count > 0:
            return self.sending_count / self.receiving_count
        return 1.0
    
    @property
    def network_health(self) -> NetworkHealth:
        """Assess network health from address activity."""
        if self.new_count <= 0:
            return NetworkHealth.DECLINING
        
        growth_rate = self.new_count / max(self.active_count, 1) * 100
        
        if growth_rate > 5:
            return NetworkHealth.BOOMING
        elif growth_rate > 2:
            return NetworkHealth.GROWING
        elif growth_rate > 0.5:
            return NetworkHealth.STABLE
        elif growth_rate > 0:
            return NetworkHealth.WEAK
        return NetworkHealth.DECLINING
    
    @property
    def is_growing(self) -> bool:
        """Check if network is growing."""
        return self.new_count > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'asset': self.asset, 'active_count': self.active_count, 'new_count': self.new_count,
            'activity_ratio': self.activity_ratio, 'network_health': self.network_health.value,
        }

# =============================================================================
# COLLECTOR CLASS
# =============================================================================

class GlassnodeCollector:
    """
    Glassnode on-chain metrics collector.
    
    Features:
    - Network metrics: Active addresses, transactions, hash rate
    - Exchange metrics: Inflows, outflows, balances
    - Supply metrics: Circulating, liquid, illiquid, HODL waves
    - Market indicators: SOPR, NUPL, MVRV, NVT, Puell Multiple
    - Derivatives: Futures OI, funding rates, liquidations
    """
    
    VENUE = 'glassnode'
    VENUE_TYPE = 'onchain_analytics'
    BASE_URL = 'https://api.glassnode.com/v1/metrics'
    
    SUPPORTED_ASSETS = [a.value for a in Asset]
    RESOLUTIONS = [r.value for r in Resolution]
    
    def __init__(self, config: Dict):
        """Initialize Glassnode collector."""
        self.api_key = config.get('api_key', config.get('glassnode_api_key', ''))
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_remaining = 10
        self.logger = logging.getLogger(self.__class__.__name__)
        self.collection_stats = {'requests': 0, 'records': 0, 'errors': 0}
    
    async def __aenter__(self) -> 'GlassnodeCollector':
        """Async context manager entry."""
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def _make_request(
        self, category: str, metric: str, asset: str = 'BTC',
        resolution: str = '24h', since: Optional[int] = None,
        until: Optional[int] = None
    ) -> Optional[List[Dict]]:
        """Make authenticated API request."""
        session = await self._get_session()
        url = f"{self.BASE_URL}/{category}/{metric}"
        
        params = {'a': asset, 'i': resolution, 'api_key': self.api_key}
        if since:
            params['s'] = since
        if until:
            params['u'] = until
        
        try:
            async with session.get(url, params=params) as response:
                self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 10))
                self.collection_stats['requests'] += 1
                
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    self.logger.warning("Rate limit exceeded")
                    await asyncio.sleep(120)
                    return None
                else:
                    self.collection_stats['errors'] += 1
                    return None
        except Exception as e:
            self.logger.error(f"Request error: {e}")
            self.collection_stats['errors'] += 1
            return None
    
    async def fetch_metric(
        self, category: str, metric: str, asset: str = 'BTC',
        resolution: str = '24h', start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch a specific metric for an asset."""
        since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp()) if start_date else None
        until = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp()) if end_date else None
        
        data = await self._make_request(category, metric, asset, resolution, since, until)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if not df.empty and 't' in df.columns:
            df['timestamp'] = pd.to_datetime(df['t'], unit='s', utc=True)
            df = df.rename(columns={'v': metric})
            df['asset'] = asset
            df['resolution'] = resolution
            df['venue'] = self.VENUE
            df = df.drop(columns=['t'], errors='ignore')
            self.collection_stats['records'] += len(df)
        
        return df
    
    async def fetch_mvrv(
        self, asset: str = 'BTC', resolution: str = '24h',
        start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch MVRV with z-score and zone classifications."""
        mvrv_df = await self.fetch_metric('indicators', 'mvrv', asset, resolution, start_date, end_date)
        zscore_df = await self.fetch_metric('indicators', 'mvrv_z_score', asset, resolution, start_date, end_date)
        
        if mvrv_df.empty:
            return pd.DataFrame()
        
        if not zscore_df.empty:
            mvrv_df = mvrv_df.merge(zscore_df[['timestamp', 'mvrv_z_score']], on='timestamp', how='left')
        
        mvrv_df['zone'] = mvrv_df['mvrv'].apply(
            lambda v: MVRVZone.DEEP_VALUE.value if v < 0.8 else
                      MVRVZone.UNDERVALUED.value if v < 1.0 else
                      MVRVZone.FAIR_VALUE.value if v < 2.4 else
                      MVRVZone.OVERVALUED.value if v < 3.7 else
                      MVRVZone.EXTREME_OVERVALUED.value)
        mvrv_df['is_undervalued'] = mvrv_df['mvrv'] < 1.0
        mvrv_df['is_overvalued'] = mvrv_df['mvrv'] > 2.4
        mvrv_df['position_signal'] = mvrv_df['mvrv'].apply(
            lambda v: 'max_long' if v < 0.8 else 'accumulate' if v < 1.0 else
                      'hold' if v < 2.4 else 'reduce' if v < 3.7 else 'max_short')
        
        return mvrv_df
    
    async def fetch_nupl(
        self, asset: str = 'BTC', resolution: str = '24h',
        start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch NUPL with cycle phase classifications."""
        df = await self.fetch_metric('indicators', 'nupl', asset, resolution, start_date, end_date)
        
        if not df.empty:
            df['phase'] = df['nupl'].apply(
                lambda v: NUPLPhase.CAPITULATION.value if v < 0 else
                          NUPLPhase.HOPE.value if v < 0.25 else
                          NUPLPhase.OPTIMISM.value if v < 0.5 else
                          NUPLPhase.BELIEF.value if v < 0.75 else
                          NUPLPhase.EUPHORIA.value)
            df['is_capitulation'] = df['nupl'] < 0
            df['is_euphoria'] = df['nupl'] > 0.75
            df['risk_level'] = df['nupl'].apply(
                lambda v: 'low' if v < 0 else 'moderate' if v < 0.5 else
                          'elevated' if v < 0.75 else 'extreme')
        
        return df
    
    async def fetch_sopr(
        self, asset: str = 'BTC', resolution: str = '24h',
        start_date: Optional[str] = None, end_date: Optional[str] = None,
        adjusted: bool = True
    ) -> pd.DataFrame:
        """Fetch SOPR with trading signals."""
        metric = 'sopr_adjusted' if adjusted else 'sopr'
        df = await self.fetch_metric('indicators', metric, asset, resolution, start_date, end_date)
        
        if not df.empty:
            col = metric
            df['signal'] = df[col].apply(
                lambda v: SOPRSignal.CAPITULATION.value if v < 0.9 else
                          SOPRSignal.LOSS_SELLING.value if v < 0.97 else
                          SOPRSignal.SUPPORT.value if v <= 1.03 else
                          SOPRSignal.PROFIT_TAKING.value if v <= 1.1 else
                          SOPRSignal.DISTRIBUTION.value)
            df['is_profitable'] = df[col] > 1.0
            df['is_at_support'] = (df[col] >= 0.97) & (df[col] <= 1.03)
            df['deviation_pct'] = (df[col] - 1.0) * 100
        
        return df
    
    async def _fetch_single_exchange_metric(
        self, metric: str, asset: str, resolution: str, start_date: Optional[str], end_date: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """Fetch a single exchange flow metric."""
        try:
            df = await self.fetch_metric('distribution', metric, asset, resolution, start_date, end_date)
            if not df.empty:
                return df
        except Exception as e:
            self.logger.error(f"Error fetching exchange metric {metric}: {e}")
        return None

    async def fetch_exchange_flows(
        self, asset: str = 'BTC', resolution: str = '24h',
        start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch exchange flow metrics."""
        metrics = ['balance_exchanges', 'inflow_exchanges_sum', 'outflow_exchanges_sum', 'net_position_change_exchanges']

        # PARALLELIZED: Fetch all metrics concurrently
        tasks = [self._fetch_single_exchange_metric(metric, asset, resolution, start_date, end_date) for metric in metrics]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

        if not all_data:
            return pd.DataFrame()

        result = all_data[0]
        for df in all_data[1:]:
            col_name = [c for c in df.columns if c not in ['timestamp', 'asset', 'resolution', 'venue']]
            if col_name:
                result = result.merge(df[['timestamp'] + col_name], on='timestamp', how='outer')

        if 'net_position_change_exchanges' in result.columns:
            result['is_accumulation'] = result['net_position_change_exchanges'] < 0
            result['is_distribution'] = result['net_position_change_exchanges'] > 0
            result['sentiment'] = result['net_position_change_exchanges'].apply(
                lambda v: 'bullish' if v < 0 else 'bearish' if v > 0 else 'neutral')

        return result.sort_values('timestamp')
    
    async def _fetch_single_supply_metric(
        self, metric: str, asset: str, resolution: str, start_date: Optional[str], end_date: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """Fetch a single supply metric."""
        try:
            df = await self.fetch_metric('supply', metric, asset, resolution, start_date, end_date)
            if not df.empty:
                return df
        except Exception as e:
            self.logger.error(f"Error fetching supply metric {metric}: {e}")
        return None

    async def fetch_supply_metrics(
        self, asset: str = 'BTC', resolution: str = '24h',
        start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch supply distribution metrics."""
        metrics = ['current', 'liquid_sum', 'illiquid_sum', 'profit_relative', 'loss_relative', 'active_more_1y_percent']

        # PARALLELIZED: Fetch all metrics concurrently
        tasks = [self._fetch_single_supply_metric(metric, asset, resolution, start_date, end_date) for metric in metrics]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

        if not all_data:
            return pd.DataFrame()

        result = all_data[0]
        for df in all_data[1:]:
            col_name = [c for c in df.columns if c not in ['timestamp', 'asset', 'resolution', 'venue']]
            if col_name:
                result = result.merge(df[['timestamp'] + col_name], on='timestamp', how='outer')

        return result.sort_values('timestamp')
    
    async def _fetch_single_mining_metric(
        self, category: str, metric: str, asset: str, resolution: str, start_date: Optional[str], end_date: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """Fetch a single mining metric."""
        try:
            df = await self.fetch_metric(category, metric, asset, resolution, start_date, end_date)
            if not df.empty:
                return df
        except Exception as e:
            self.logger.error(f"Error fetching mining metric {metric}: {e}")
        return None

    async def fetch_mining_metrics(
        self, asset: str = 'BTC', resolution: str = '24h',
        start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch mining and hash rate metrics."""
        # PARALLELIZED: Fetch all metrics concurrently
        tasks = [
            self._fetch_single_mining_metric('mining', 'hash_rate_mean', asset, resolution, start_date, end_date),
            self._fetch_single_mining_metric('mining', 'difficulty_latest', asset, resolution, start_date, end_date),
            self._fetch_single_mining_metric('mining', 'revenue_sum', asset, resolution, start_date, end_date),
            self._fetch_single_mining_metric('indicators', 'puell_multiple', asset, resolution, start_date, end_date),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

        if not all_data:
            return pd.DataFrame()

        result = all_data[0]
        for df in all_data[1:]:
            col_name = [c for c in df.columns if c not in ['timestamp', 'asset', 'resolution', 'venue']]
            if col_name:
                result = result.merge(df[['timestamp'] + col_name], on='timestamp', how='outer')

        if 'puell_multiple' in result.columns:
            result['puell_zone'] = result['puell_multiple'].apply(
                lambda v: PuellZone.CAPITULATION.value if v < 0.5 else
                          PuellZone.UNDERVALUED.value if v < 0.8 else
                          PuellZone.FAIR.value if v < 1.5 else
                          PuellZone.OVERVALUED.value if v < 4.0 else
                          PuellZone.EUPHORIA.value)
            result['is_miner_capitulation'] = result['puell_multiple'] < 0.5

        return result.sort_values('timestamp')
    
    async def _fetch_single_derivatives_metric(
        self, metric: str, asset: str, resolution: str, start_date: Optional[str], end_date: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """Fetch a single derivatives metric."""
        try:
            df = await self.fetch_metric('derivatives', metric, asset, resolution, start_date, end_date)
            if not df.empty:
                return df
        except Exception as e:
            self.logger.error(f"Error fetching derivatives metric {metric}: {e}")
        return None

    async def fetch_derivatives_metrics(
        self, asset: str = 'BTC', resolution: str = '24h',
        start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch derivatives market metrics."""
        metrics = ['futures_open_interest_sum', 'futures_volume_daily_sum', 'futures_funding_rate_perpetual',
                   'futures_liquidated_volume_long_sum', 'futures_liquidated_volume_short_sum']

        # PARALLELIZED: Fetch all metrics concurrently
        tasks = [self._fetch_single_derivatives_metric(metric, asset, resolution, start_date, end_date) for metric in metrics]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

        if not all_data:
            return pd.DataFrame()

        result = all_data[0]
        for df in all_data[1:]:
            col_name = [c for c in df.columns if c not in ['timestamp', 'asset', 'resolution', 'venue']]
            if col_name:
                result = result.merge(df[['timestamp'] + col_name], on='timestamp', how='outer')

        if 'futures_funding_rate_perpetual' in result.columns:
            result['funding_sentiment'] = result['futures_funding_rate_perpetual'].apply(
                lambda v: 'extreme_long' if v > 0.01 else 'long_bias' if v > 0.001 else
                          'extreme_short' if v < -0.01 else 'short_bias' if v < -0.001 else 'neutral')

        return result.sort_values('timestamp')
    
    async def _fetch_single_address_metric(
        self, metric: str, asset: str, resolution: str, start_date: Optional[str], end_date: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """Fetch a single address metric."""
        try:
            df = await self.fetch_metric('addresses', metric, asset, resolution, start_date, end_date)
            if not df.empty:
                return df
        except Exception as e:
            self.logger.error(f"Error fetching address metric {metric}: {e}")
        return None

    async def fetch_address_metrics(
        self, asset: str = 'BTC', resolution: str = '24h',
        start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch address activity metrics."""
        metrics = ['active_count', 'sending_count', 'receiving_count', 'new_non_zero_count']

        # PARALLELIZED: Fetch all metrics concurrently
        tasks = [self._fetch_single_address_metric(metric, asset, resolution, start_date, end_date) for metric in metrics]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

        if not all_data:
            return pd.DataFrame()

        result = all_data[0]
        for df in all_data[1:]:
            col_name = [c for c in df.columns if c not in ['timestamp', 'asset', 'resolution', 'venue']]
            if col_name:
                result = result.merge(df[['timestamp'] + col_name], on='timestamp', how='outer')

        return result.sort_values('timestamp')
    
    async def _fetch_single_indicator_metric(
        self, metric: str, asset: str, resolution: str, start_date: Optional[str], end_date: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """Fetch a single indicator metric."""
        try:
            df = await self.fetch_metric('indicators', metric, asset, resolution, start_date, end_date)
            if not df.empty:
                return df
        except Exception as e:
            self.logger.error(f"Error fetching indicator metric {metric}: {e}")
        return None

    async def fetch_market_indicators(
        self, asset: str = 'BTC', resolution: str = '24h',
        start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch all market indicator metrics."""
        indicators = ['mvrv', 'sopr', 'sopr_adjusted', 'nvt', 'nvts', 'nupl', 'puell_multiple']

        # PARALLELIZED: Fetch all metrics concurrently
        tasks = [self._fetch_single_indicator_metric(metric, asset, resolution, start_date, end_date) for metric in indicators]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

        if not all_data:
            return pd.DataFrame()

        result = all_data[0]
        for df in all_data[1:]:
            col_name = [c for c in df.columns if c not in ['timestamp', 'asset', 'resolution', 'venue']]
            if col_name:
                result = result.merge(df[['timestamp'] + col_name], on='timestamp', how='outer')

        return result.sort_values('timestamp')
    
    async def fetch_comprehensive_onchain(
        self, asset: str = 'BTC', resolution: str = '24h',
        start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch comprehensive on-chain data across all categories."""
        self.logger.info(f"Fetching comprehensive Glassnode data for {asset}")
        
        return {
            'exchange_flows': await self.fetch_exchange_flows(asset, resolution, start_date, end_date),
            'supply': await self.fetch_supply_metrics(asset, resolution, start_date, end_date),
            'indicators': await self.fetch_market_indicators(asset, resolution, start_date, end_date),
            'addresses': await self.fetch_address_metrics(asset, resolution, start_date, end_date),
            'mining': await self.fetch_mining_metrics(asset, resolution, start_date, end_date),
            'derivatives': await self.fetch_derivatives_metrics(asset, resolution, start_date, end_date),
            'mvrv': await self.fetch_mvrv(asset, resolution, start_date, end_date),
            'nupl': await self.fetch_nupl(asset, resolution, start_date, end_date),
            'sopr': await self.fetch_sopr(asset, resolution, start_date, end_date),
        }
    
    async def _fetch_single_funding_rate(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch funding rate for a single symbol."""
        try:
            asset = symbol.replace('USDT', '').replace('USD', '').replace('PERP', '')
            df = await self.fetch_metric('derivatives', 'futures_funding_rate_perpetual', asset, '1h', start_date, end_date)
            if not df.empty:
                df['symbol'] = symbol
                return df
        except Exception as e:
            self.logger.error(f"Error fetching funding rate for {symbol}: {e}")
        return None

    async def fetch_funding_rates(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch aggregated funding rates."""
        # PARALLELIZED: Fetch all symbols concurrently
        tasks = [self._fetch_single_funding_rate(symbol, start_date, end_date) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    async def _fetch_single_ohlcv(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV for a single symbol."""
        try:
            resolution_map = {'1h': '1h', '4h': '24h', '1d': '24h', '1w': '1w'}
            resolution = resolution_map.get(timeframe, '24h')

            asset = symbol.replace('USDT', '').replace('USD', '')
            df = await self.fetch_metric('market', 'price_usd_close', asset, resolution, start_date, end_date)
            if not df.empty:
                df['symbol'] = symbol
                df = df.rename(columns={'price_usd_close': 'close'})
                return df
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV for {symbol}: {e}")
        return None

    async def fetch_ohlcv(self, symbols: List[str], timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch price data."""
        # PARALLELIZED: Fetch all symbols concurrently
        tasks = [self._fetch_single_ohlcv(symbol, timeframe, start_date, end_date) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {**self.collection_stats, 'venue': self.VENUE, 'rate_limit_remaining': self.rate_limit_remaining}
    
    @staticmethod
    def get_supported_assets() -> List[str]:
        """Get list of supported assets."""
        return [a.value for a in Asset]
    
    @staticmethod
    def get_metric_categories() -> List[str]:
        """Get list of metric categories."""
        return [c.value for c in MetricCategory]
    
    async def close(self) -> None:
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

async def test_glassnode():
    """Test Glassnode collector."""
    config = {'api_key': ''}
    collector = GlassnodeCollector(config)
    try:
        print(f"Supported assets: {collector.get_supported_assets()}")
        print(f"Metric categories: {collector.get_metric_categories()}")
        
        mvrv = MVRVData(timestamp=datetime.utcnow(), asset='BTC', mvrv=1.5)
        print(f"MVRV zone: {mvrv.zone.value}, signal: {mvrv.position_signal}")
        
        nupl = NUPLData(timestamp=datetime.utcnow(), asset='BTC', nupl=0.4)
        print(f"NUPL phase: {nupl.phase.value}, risk: {nupl.risk_level}")
    finally:
        await collector.close()

if __name__ == '__main__':
    asyncio.run(test_glassnode())