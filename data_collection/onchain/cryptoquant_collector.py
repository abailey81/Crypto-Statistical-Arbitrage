"""
CryptoQuant On-Chain Data Collector

validated collector for institutional on-chain analytics from CryptoQuant.
Provides exchange flows, miner metrics, network indicators, and market valuation
signals for BTC and ETH.

===============================================================================
CRYPTOQUANT OVERVIEW
===============================================================================

CryptoQuant is the leading on-chain analytics provider used by institutional
traders and funds. It aggregates data from major exchanges and blockchain
networks to provide actionable market intelligence.

Key Differentiators:
    - Real-time exchange flow data from 20+ major exchanges
    - Comprehensive miner behavior tracking
    - professional-quality market indicators (SOPR, NUPL, MVRV, Puell)
    - Fund flow tracking (Grayscale, ETF flows)
    - Stablecoin supply and flow analysis

===============================================================================
API SPECIFICATIONS
===============================================================================

Base URL: https://api.cryptoquant.com/v1

Authentication:
    - Bearer Token in Authorization header
    - API keys obtained from CryptoQuant dashboard

Rate Limits by Tier:
    ============ ============== ================ ===============
    Tier Requests/min Daily Limit Historical Data
    ============ ============== ================ ===============
    Free 10 1,000 30 days
    Starter 30 10,000 1 year
    Professional 100 Unlimited Full history
    Enterprise Custom Unlimited Full history
    ============ ============== ================ ===============

===============================================================================
DATA CATEGORIES
===============================================================================

Exchange Metrics:
    - exchange-reserve: Total holdings on exchanges
    - exchange-inflow: Deposits to exchanges (selling pressure)
    - exchange-outflow: Withdrawals from exchanges (accumulation)
    - exchange-netflow: Net flow (inflow - outflow)

Market Indicators:
    - sopr: Spent Output Profit Ratio (profit-taking)
    - nupl: Net Unrealized Profit/Loss (market sentiment)
    - mvrv: Market Value to Realized Value (valuation)
    - puell-multiple: Miner profitability indicator

Miner Metrics:
    - miner-reserve: Miner holdings
    - miner-outflow: Miner selling
    - miner-revenue: Daily mining income

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

Signal Generation:
    - Exchange inflows spike -> Short-term bearish
    - MVRV < 1 -> Historically undervalued
    - NUPL capitulation -> Potential bottom
    - Miner capitulation -> Cycle low indicator

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
    """Supported assets on CryptoQuant."""
    BTC = 'btc'
    ETH = 'eth'
    
    @classmethod
    def from_string(cls, value: str) -> 'Asset':
        """Convert string to Asset enum."""
        mapping = {'btc': cls.BTC, 'bitcoin': cls.BTC, 'eth': cls.ETH, 'ethereum': cls.ETH}
        return mapping.get(value.lower(), cls.BTC)

class MetricCategory(Enum):
    """CryptoQuant metric categories."""
    EXCHANGE = 'exchange'
    NETWORK = 'network'
    MARKET = 'market'
    MINER = 'miner'
    STABLECOIN = 'stablecoin'

class ExchangeMetric(Enum):
    """Exchange-related metrics."""
    RESERVE = 'exchange-reserve'
    INFLOW = 'exchange-inflow'
    OUTFLOW = 'exchange-outflow'
    NETFLOW = 'exchange-netflow'
    INFLOW_MEAN = 'exchange-inflow-mean'
    INFLOW_TOP10 = 'exchange-inflow-top10'

class MarketIndicator(Enum):
    """Market valuation indicators."""
    SOPR = 'sopr'
    ASOPR = 'asopr'
    NUPL = 'nupl'
    MVRV = 'mvrv'
    REALIZED_CAP = 'realized-cap'
    PUELL_MULTIPLE = 'puell-multiple'

class NetworkMetric(Enum):
    """Network health metrics."""
    HASH_RATE = 'hash-rate'
    ACTIVE_ADDRESSES = 'active-addresses'
    TRANSACTION_COUNT = 'transaction-count'
    TRANSACTION_VOLUME = 'transaction-volume'
    FEES_TOTAL = 'fees-total'
    BLOCK_SIZE = 'block-size'

class MinerMetric(Enum):
    """Miner behavior metrics."""
    RESERVE = 'miner-reserve'
    OUTFLOW = 'miner-outflow'
    REVENUE = 'miner-revenue'
    TO_EXCHANGE = 'miner-to-exchange'

class TimeWindow(Enum):
    """Data aggregation windows."""
    BLOCK = 'block'
    HOUR = 'hour'
    DAY = 'day'
    WEEK = 'week'

class SOPRSignal(Enum):
    """SOPR interpretation signals."""
    STRONG_LOSS = 'strong_loss'
    MILD_LOSS = 'mild_loss'
    BREAKEVEN = 'breakeven'
    MILD_PROFIT = 'mild_profit'
    STRONG_PROFIT = 'strong_profit'

class MVRVZone(Enum):
    """MVRV valuation zones."""
    EXTREME_UNDERVALUED = 'extreme_undervalued'
    UNDERVALUED = 'undervalued'
    FAIR = 'fair'
    OVERVALUED = 'overvalued'
    EXTREME_OVERVALUED = 'extreme_overvalued'

class NUPLPhase(Enum):
    """NUPL market cycle phases."""
    CAPITULATION = 'capitulation'
    ANXIETY = 'anxiety'
    OPTIMISM = 'optimism'
    GREED = 'greed'
    EUPHORIA = 'euphoria'

class FlowSignal(Enum):
    """Exchange flow signal classification."""
    STRONG_INFLOW = 'strong_inflow'
    MODERATE_INFLOW = 'moderate_inflow'
    NEUTRAL = 'neutral'
    MODERATE_OUTFLOW = 'moderate_outflow'
    STRONG_OUTFLOW = 'strong_outflow'

class PuellZone(Enum):
    """Puell Multiple valuation zones."""
    EXTREME_LOW = 'extreme_low'
    LOW = 'low'
    NEUTRAL = 'neutral'
    HIGH = 'high'
    EXTREME_HIGH = 'extreme_high'

class MinerBehavior(Enum):
    """Miner behavior classification."""
    CAPITULATING = 'capitulating'
    DISTRIBUTING = 'distributing'
    NEUTRAL = 'neutral'
    ACCUMULATING = 'accumulating'
    HODLING = 'hodling'

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class ExchangeFlowRecord:
    """Exchange flow data with analytical properties."""
    timestamp: datetime
    asset: str
    metric: str
    value: float
    value_usd: float = 0.0
    exchange: Optional[str] = None
    window: str = 'day'
    price_at_time: float = 0.0
    percent_of_reserve: float = 0.0
    
    @property
    def flow_signal(self) -> FlowSignal:
        """Classify flow based on magnitude."""
        if self.percent_of_reserve == 0:
            return FlowSignal.NEUTRAL
        if 'inflow' in self.metric:
            if self.percent_of_reserve > 2.0:
                return FlowSignal.STRONG_INFLOW
            elif self.percent_of_reserve > 0.5:
                return FlowSignal.MODERATE_INFLOW
        elif 'outflow' in self.metric:
            if self.percent_of_reserve > 2.0:
                return FlowSignal.STRONG_OUTFLOW
            elif self.percent_of_reserve > 0.5:
                return FlowSignal.MODERATE_OUTFLOW
        return FlowSignal.NEUTRAL
    
    @property
    def is_significant(self) -> bool:
        """Check if flow is statistically significant."""
        return abs(self.percent_of_reserve) > 0.5
    
    @property
    def is_bullish(self) -> bool:
        """Check if flow suggests bullish sentiment."""
        return 'outflow' in self.metric and self.value > 0
    
    @property
    def is_bearish(self) -> bool:
        """Check if flow suggests bearish sentiment."""
        return 'inflow' in self.metric and self.value > 0
    
    @property
    def flow_direction(self) -> str:
        """Get flow direction as string."""
        if 'inflow' in self.metric:
            return 'inflow'
        elif 'outflow' in self.metric:
            return 'outflow'
        elif 'netflow' in self.metric:
            return 'inflow' if self.value > 0 else 'outflow'
        return 'unknown'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'asset': self.asset, 'metric': self.metric, 'value': self.value,
            'value_usd': self.value_usd, 'exchange': self.exchange, 'window': self.window,
            'flow_signal': self.flow_signal.value, 'is_significant': self.is_significant,
            'is_bullish': self.is_bullish, 'is_bearish': self.is_bearish,
        }

@dataclass
class SOPRRecord:
    """SOPR data with signal classifications."""
    timestamp: datetime
    asset: str
    value: float
    adjusted: bool = False
    sma_7d: Optional[float] = None
    sma_30d: Optional[float] = None
    
    @property
    def signal(self) -> SOPRSignal:
        """Classify SOPR signal."""
        if self.value < 0.9:
            return SOPRSignal.STRONG_LOSS
        elif self.value < 0.98:
            return SOPRSignal.MILD_LOSS
        elif self.value <= 1.02:
            return SOPRSignal.BREAKEVEN
        elif self.value <= 1.1:
            return SOPRSignal.MILD_PROFIT
        return SOPRSignal.STRONG_PROFIT
    
    @property
    def is_profitable(self) -> bool:
        """Check if holders are profitable on average."""
        return self.value > 1.0
    
    @property
    def is_capitulating(self) -> bool:
        """Check if market shows capitulation."""
        return self.value < 0.95
    
    @property
    def is_profit_taking(self) -> bool:
        """Check if heavy profit-taking."""
        return self.value > 1.05
    
    @property
    def is_at_support(self) -> bool:
        """Check if SOPR at support level (1.0)."""
        return 0.98 <= self.value <= 1.02
    
    @property
    def deviation_from_breakeven(self) -> float:
        """Percentage deviation from breakeven."""
        return (self.value - 1.0) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'asset': self.asset, 'value': self.value, 'adjusted': self.adjusted,
            'signal': self.signal.value, 'is_profitable': self.is_profitable,
            'is_capitulating': self.is_capitulating, 'deviation_from_breakeven': self.deviation_from_breakeven,
        }

@dataclass
class MVRVRecord:
    """MVRV data with valuation zones."""
    timestamp: datetime
    asset: str
    value: float
    market_cap: float = 0.0
    realized_cap: float = 0.0
    z_score: Optional[float] = None
    
    @property
    def zone(self) -> MVRVZone:
        """Classify MVRV into valuation zones."""
        if self.value < 0.8:
            return MVRVZone.EXTREME_UNDERVALUED
        elif self.value < 1.2:
            return MVRVZone.UNDERVALUED
        elif self.value < 2.5:
            return MVRVZone.FAIR
        elif self.value < 3.5:
            return MVRVZone.OVERVALUED
        return MVRVZone.EXTREME_OVERVALUED
    
    @property
    def is_undervalued(self) -> bool:
        """Check if market is undervalued."""
        return self.value < 1.2
    
    @property
    def is_overvalued(self) -> bool:
        """Check if market is overvalued."""
        return self.value > 2.5
    
    @property
    def is_extreme(self) -> bool:
        """Check if at extreme valuation."""
        return self.value < 0.8 or self.value > 3.5
    
    @property
    def position_recommendation(self) -> str:
        """Trading recommendation based on MVRV."""
        if self.value < 0.8:
            return 'strong_accumulate'
        elif self.value < 1.2:
            return 'accumulate'
        elif self.value < 2.5:
            return 'hold'
        elif self.value < 3.5:
            return 'reduce'
        return 'strong_reduce'
    
    @property
    def percent_above_realized(self) -> float:
        """Percentage market cap is above realized cap."""
        return (self.value - 1.0) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'asset': self.asset, 'value': self.value, 'zone': self.zone.value,
            'is_undervalued': self.is_undervalued, 'is_overvalued': self.is_overvalued,
            'position_recommendation': self.position_recommendation,
        }

@dataclass
class NUPLRecord:
    """NUPL data with cycle phase classifications."""
    timestamp: datetime
    asset: str
    value: float
    unrealized_profit: float = 0.0
    unrealized_loss: float = 0.0
    
    @property
    def phase(self) -> NUPLPhase:
        """Classify market cycle phase."""
        if self.value < 0:
            return NUPLPhase.CAPITULATION
        elif self.value < 0.25:
            return NUPLPhase.ANXIETY
        elif self.value < 0.5:
            return NUPLPhase.OPTIMISM
        elif self.value < 0.75:
            return NUPLPhase.GREED
        return NUPLPhase.EUPHORIA
    
    @property
    def is_capitulation(self) -> bool:
        """Check if market in capitulation."""
        return self.value < 0
    
    @property
    def is_euphoria(self) -> bool:
        """Check if market in euphoria."""
        return self.value > 0.75
    
    @property
    def is_danger_zone(self) -> bool:
        """Check if in danger zone."""
        return self.value < 0 or self.value > 0.75
    
    @property
    def risk_level(self) -> str:
        """Market risk level."""
        if self.value < 0:
            return 'low'
        elif self.value < 0.5:
            return 'moderate'
        elif self.value < 0.75:
            return 'elevated'
        return 'extreme'
    
    @property
    def cycle_position(self) -> float:
        """Normalized cycle position (0-1)."""
        return min(1.0, max(0.0, (self.value + 0.5) / 1.25))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'asset': self.asset, 'value': self.value, 'phase': self.phase.value,
            'is_capitulation': self.is_capitulation, 'is_euphoria': self.is_euphoria,
            'risk_level': self.risk_level, 'cycle_position': self.cycle_position,
        }

@dataclass
class PuellMultipleRecord:
    """Puell Multiple data with miner profitability analysis."""
    timestamp: datetime
    asset: str
    value: float
    daily_revenue_usd: float = 0.0
    ma_365d_revenue: float = 0.0
    
    @property
    def zone(self) -> PuellZone:
        """Classify Puell Multiple zone."""
        if self.value < 0.5:
            return PuellZone.EXTREME_LOW
        elif self.value < 1.0:
            return PuellZone.LOW
        elif self.value < 2.5:
            return PuellZone.NEUTRAL
        elif self.value < 4.0:
            return PuellZone.HIGH
        return PuellZone.EXTREME_HIGH
    
    @property
    def is_miner_capitulation(self) -> bool:
        """Check if miners are capitulating."""
        return self.value < 0.5
    
    @property
    def is_overheated(self) -> bool:
        """Check if mining is overheated."""
        return self.value > 4.0
    
    @property
    def miner_profitability(self) -> str:
        """Miner profitability description."""
        if self.value < 0.5:
            return 'unprofitable'
        elif self.value < 1.0:
            return 'below_average'
        elif self.value < 2.0:
            return 'average'
        elif self.value < 4.0:
            return 'profitable'
        return 'extremely_profitable'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'asset': self.asset, 'value': self.value, 'zone': self.zone.value,
            'is_miner_capitulation': self.is_miner_capitulation,
            'miner_profitability': self.miner_profitability,
        }

@dataclass
class MinerDataRecord:
    """Miner behavior data record."""
    timestamp: datetime
    asset: str
    metric: str
    value: float
    value_usd: float = 0.0
    hash_rate: float = 0.0
    
    @property
    def behavior(self) -> MinerBehavior:
        """Classify miner behavior based on metric."""
        if 'outflow' in self.metric or 'to-exchange' in self.metric:
            if self.value_usd > 50_000_000:
                return MinerBehavior.CAPITULATING
            elif self.value_usd > 20_000_000:
                return MinerBehavior.DISTRIBUTING
            elif self.value_usd > 5_000_000:
                return MinerBehavior.NEUTRAL
            return MinerBehavior.HODLING
        return MinerBehavior.NEUTRAL
    
    @property
    def is_selling(self) -> bool:
        """Check if miners are net sellers."""
        return 'outflow' in self.metric or 'to-exchange' in self.metric
    
    @property
    def is_significant_outflow(self) -> bool:
        """Check if outflow is significant."""
        return self.is_selling and self.value_usd > 10_000_000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'asset': self.asset, 'metric': self.metric, 'value': self.value,
            'behavior': self.behavior.value, 'is_selling': self.is_selling,
        }

@dataclass
class WhaleMovement:
    """Whale transaction record."""
    timestamp: datetime
    asset: str
    amount: float
    amount_usd: float
    direction: str
    exchange: Optional[str] = None
    threshold_btc: float = 100.0
    
    @property
    def is_whale(self) -> bool:
        """Verify this is a whale transaction."""
        return self.amount >= self.threshold_btc
    
    @property
    def size_category(self) -> str:
        """Categorize whale size."""
        if self.amount >= 1000:
            return 'mega_whale'
        elif self.amount >= 500:
            return 'large_whale'
        elif self.amount >= 100:
            return 'whale'
        return 'large_holder'
    
    @property
    def market_impact(self) -> str:
        """Estimated market impact."""
        if self.amount_usd >= 100_000_000:
            return 'very_high'
        elif self.amount_usd >= 50_000_000:
            return 'high'
        elif self.amount_usd >= 10_000_000:
            return 'moderate'
        return 'low'
    
    @property
    def sentiment_signal(self) -> str:
        """Sentiment signal from whale movement."""
        if self.direction == 'inflow' and self.exchange:
            return 'bearish'
        elif self.direction == 'outflow' and self.exchange:
            return 'bullish'
        return 'neutral'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'asset': self.asset, 'amount': self.amount, 'amount_usd': self.amount_usd,
            'direction': self.direction, 'size_category': self.size_category,
            'market_impact': self.market_impact, 'sentiment_signal': self.sentiment_signal,
        }

# =============================================================================
# COLLECTOR CLASS
# =============================================================================

class CryptoQuantCollector:
    """
    CryptoQuant on-chain data collector.
    
    Provides comprehensive on-chain metrics:
    - Exchange flows and reserves
    - Network fundamentals
    - Miner behavior
    - Market indicators (SOPR, NUPL, MVRV, Puell)
    
    API: https://cryptoquant.com/docs/api
    """
    
    VENUE = 'cryptoquant'
    VENUE_TYPE = 'onchain'
    BASE_URL = 'https://api.cryptoquant.com/v1'
    
    EXCHANGE_METRICS = [e.value for e in ExchangeMetric]
    NETWORK_METRICS = [n.value for n in NetworkMetric]
    MARKET_METRICS = [m.value for m in MarketIndicator]
    MINER_METRICS = [m.value for m in MinerMetric]
    SUPPORTED_ASSETS = ['btc', 'eth']
    
    def __init__(self, config: Dict):
        """Initialize CryptoQuant collector."""
        self.api_key = config.get('cryptoquant_api_key', '')
        self.rate_limit = config.get('rate_limit', 5)
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.collection_stats = {'requests': 0, 'records': 0, 'errors': 0, 'last_request': None}
    
    async def __aenter__(self) -> 'CryptoQuantCollector':
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
            headers = {'Authorization': f'Bearer {self.api_key}', 'Accept': 'application/json'}
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self.session
    
    async def _rate_limit_wait(self) -> None:
        """Wait for rate limit compliance."""
        await asyncio.sleep(60.0 / self.rate_limit)
    
    async def _fetch_metric(
        self, asset: str, metric: str, window: str = 'day',
        start_date: Optional[str] = None, end_date: Optional[str] = None,
        exchange: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch a specific metric from CryptoQuant."""
        session = await self._get_session()
        url = f'{self.BASE_URL}/{asset}/{metric}'
        params = {'window': window, 'limit': 1000}
        if start_date:
            params['from'] = start_date
        if end_date:
            params['to'] = end_date
        if exchange:
            params['exchange'] = exchange
        
        await self._rate_limit_wait()
        self.collection_stats['requests'] += 1
        self.collection_stats['last_request'] = datetime.utcnow()
        
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    records = data.get('result', {}).get('data', [])
                    if not records:
                        return pd.DataFrame()
                    
                    df = pd.DataFrame(records)
                    if 'date' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['date'], utc=True)
                    elif 'datetime' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['datetime'], utc=True)
                    
                    df['metric'] = metric
                    df['asset'] = asset.upper()
                    df['window'] = window
                    df['venue'] = self.VENUE
                    df['venue_type'] = self.VENUE_TYPE
                    self.collection_stats['records'] += len(df)
                    return df
                elif resp.status == 429:
                    self.logger.warning("Rate limit exceeded")
                    await asyncio.sleep(60)
                else:
                    self.collection_stats['errors'] += 1
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error fetching {metric}: {e}")
            self.collection_stats['errors'] += 1
            return pd.DataFrame()
    
    async def fetch_exchange_flows(
        self, asset: str = 'btc', start_date: str = '2022-01-01',
        end_date: Optional[str] = None, exchange: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch exchange flow data with signal classifications."""
        all_data = []
        for metric in [ExchangeMetric.INFLOW.value, ExchangeMetric.OUTFLOW.value,
                       ExchangeMetric.NETFLOW.value, ExchangeMetric.RESERVE.value]:
            df = await self._fetch_metric(asset, metric, start_date=start_date,
                                          end_date=end_date, exchange=exchange)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        
        def classify_flow(row):
            if 'netflow' in row['metric']:
                return FlowSignal.MODERATE_INFLOW.value if row.get('value', 0) > 0 else FlowSignal.MODERATE_OUTFLOW.value
            return FlowSignal.NEUTRAL.value
        
        result['signal'] = result.apply(classify_flow, axis=1)
        return result
    
    async def fetch_sopr(
        self, asset: str = 'btc', start_date: str = '2022-01-01',
        end_date: Optional[str] = None, adjusted: bool = True
    ) -> pd.DataFrame:
        """Fetch SOPR with signal classifications."""
        metric = MarketIndicator.ASOPR.value if adjusted else MarketIndicator.SOPR.value
        df = await self._fetch_metric(asset, metric, start_date=start_date, end_date=end_date)
        
        if not df.empty:
            df['signal'] = df['value'].apply(lambda v: SOPRSignal.STRONG_LOSS.value if v < 0.9 else
                                              SOPRSignal.MILD_LOSS.value if v < 0.98 else
                                              SOPRSignal.BREAKEVEN.value if v <= 1.02 else
                                              SOPRSignal.MILD_PROFIT.value if v <= 1.1 else
                                              SOPRSignal.STRONG_PROFIT.value)
            df['is_profitable'] = df['value'] > 1.0
            df['is_capitulating'] = df['value'] < 0.95
            df = df.sort_values('timestamp')
            df['sma_7d'] = df['value'].rolling(7, min_periods=1).mean()
        return df
    
    async def fetch_mvrv(
        self, asset: str = 'btc', start_date: str = '2022-01-01',
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch MVRV with zone classifications."""
        df = await self._fetch_metric(asset, MarketIndicator.MVRV.value,
                                       start_date=start_date, end_date=end_date)
        if not df.empty:
            df['zone'] = df['value'].apply(lambda v: MVRVZone.EXTREME_UNDERVALUED.value if v < 0.8 else
                                           MVRVZone.UNDERVALUED.value if v < 1.2 else
                                           MVRVZone.FAIR.value if v < 2.5 else
                                           MVRVZone.OVERVALUED.value if v < 3.5 else
                                           MVRVZone.EXTREME_OVERVALUED.value)
            df['is_undervalued'] = df['value'] < 1.2
            df['is_overvalued'] = df['value'] > 2.5
            df['position_recommendation'] = df['value'].apply(
                lambda v: 'strong_accumulate' if v < 0.8 else 'accumulate' if v < 1.2 else
                          'hold' if v < 2.5 else 'reduce' if v < 3.5 else 'strong_reduce')
        return df
    
    async def fetch_nupl(
        self, asset: str = 'btc', start_date: str = '2022-01-01',
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch NUPL with phase classifications."""
        df = await self._fetch_metric(asset, MarketIndicator.NUPL.value,
                                       start_date=start_date, end_date=end_date)
        if not df.empty:
            df['phase'] = df['value'].apply(lambda v: NUPLPhase.CAPITULATION.value if v < 0 else
                                            NUPLPhase.ANXIETY.value if v < 0.25 else
                                            NUPLPhase.OPTIMISM.value if v < 0.5 else
                                            NUPLPhase.GREED.value if v < 0.75 else
                                            NUPLPhase.EUPHORIA.value)
            df['is_capitulation'] = df['value'] < 0
            df['is_euphoria'] = df['value'] > 0.75
            df['risk_level'] = df['value'].apply(lambda v: 'low' if v < 0 else
                                                  'moderate' if v < 0.5 else
                                                  'elevated' if v < 0.75 else 'extreme')
        return df
    
    async def fetch_puell_multiple(
        self, asset: str = 'btc', start_date: str = '2022-01-01',
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch Puell Multiple with zone classifications."""
        df = await self._fetch_metric(asset, MarketIndicator.PUELL_MULTIPLE.value,
                                       start_date=start_date, end_date=end_date)
        if not df.empty:
            df['zone'] = df['value'].apply(lambda v: PuellZone.EXTREME_LOW.value if v < 0.5 else
                                           PuellZone.LOW.value if v < 1.0 else
                                           PuellZone.NEUTRAL.value if v < 2.5 else
                                           PuellZone.HIGH.value if v < 4.0 else
                                           PuellZone.EXTREME_HIGH.value)
            df['is_miner_capitulation'] = df['value'] < 0.5
            df['is_overheated'] = df['value'] > 4.0
        return df
    
    async def fetch_miner_data(
        self, asset: str = 'btc', start_date: str = '2022-01-01',
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch miner metrics with behavior classifications."""
        all_data = []
        for metric in self.MINER_METRICS:
            df = await self._fetch_metric(asset, metric, start_date=start_date, end_date=end_date)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        result['is_selling'] = result['metric'].str.contains('outflow|to-exchange')
        return result
    
    async def fetch_network_metrics(
        self, asset: str = 'btc', start_date: str = '2022-01-01',
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch network metrics with trends."""
        all_data = []
        for metric in self.NETWORK_METRICS:
            df = await self._fetch_metric(asset, metric, start_date=start_date, end_date=end_date)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values(['metric', 'timestamp'])
        result['sma_7d'] = result.groupby('metric')['value'].transform(
            lambda x: x.rolling(7, min_periods=1).mean())
        return result
    
    async def fetch_whale_movements(
        self, asset: str = 'btc', start_date: str = '2022-01-01',
        end_date: Optional[str] = None, threshold: float = 100
    ) -> pd.DataFrame:
        """Fetch whale movement data."""
        df = await self._fetch_metric(asset, ExchangeMetric.INFLOW_TOP10.value,
                                       start_date=start_date, end_date=end_date)
        if not df.empty:
            df['whale_threshold_btc'] = threshold
            df['is_significant'] = df['value'] > threshold
            df['size_category'] = df['value'].apply(
                lambda v: 'mega_whale' if v >= 1000 else 'large_whale' if v >= 500 else
                          'whale' if v >= 100 else 'large_holder')
        return df
    
    async def fetch_all_metrics(
        self, asset: str = 'btc', start_date: str = '2022-01-01',
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch all available metrics for an asset."""
        return {
            'exchange_flows': await self.fetch_exchange_flows(asset, start_date, end_date),
            'sopr': await self.fetch_sopr(asset, start_date, end_date),
            'mvrv': await self.fetch_mvrv(asset, start_date, end_date),
            'nupl': await self.fetch_nupl(asset, start_date, end_date),
            'puell': await self.fetch_puell_multiple(asset, start_date, end_date),
            'miner': await self.fetch_miner_data(asset, start_date, end_date),
            'network': await self.fetch_network_metrics(asset, start_date, end_date),
            'whales': await self.fetch_whale_movements(asset, start_date, end_date),
        }
    
    async def fetch_funding_rates(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """CryptoQuant doesn't provide funding rates."""
        return pd.DataFrame()
    
    async def fetch_ohlcv(self, symbols: List[str], timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """CryptoQuant doesn't provide OHLCV."""
        return pd.DataFrame()
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {**self.collection_stats, 'venue': self.VENUE, 'supported_assets': self.SUPPORTED_ASSETS}
    
    @staticmethod
    def get_supported_assets() -> List[str]:
        """Get list of supported assets."""
        return CryptoQuantCollector.SUPPORTED_ASSETS
    
    @staticmethod
    def get_exchange_metrics() -> List[str]:
        """Get list of exchange metrics."""
        return [e.value for e in ExchangeMetric]
    
    @staticmethod
    def get_market_indicators() -> List[str]:
        """Get list of market indicators."""
        return [m.value for m in MarketIndicator]
    
    async def close(self) -> None:
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

async def test_collector():
    """Test CryptoQuant collector."""
    config = {'cryptoquant_api_key': '', 'rate_limit': 10}
    collector = CryptoQuantCollector(config)
    try:
        print(f"Supported assets: {collector.get_supported_assets()}")
        print(f"Market indicators: {collector.get_market_indicators()}")
        
        # Test dataclasses
        sopr = SOPRRecord(timestamp=datetime.utcnow(), asset='BTC', value=0.95)
        print(f"SOPR signal: {sopr.signal.value}, capitulating: {sopr.is_capitulating}")
        
        mvrv = MVRVRecord(timestamp=datetime.utcnow(), asset='BTC', value=1.5)
        print(f"MVRV zone: {mvrv.zone.value}, recommendation: {mvrv.position_recommendation}")
    finally:
        await collector.close()

if __name__ == '__main__':
    asyncio.run(test_collector())