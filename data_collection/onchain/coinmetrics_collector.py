"""
CoinMetrics Collector - Institutional On-Chain Fundamentals

validated collector for professional-quality blockchain metrics including
network data, market metrics, on-chain indicators, and exchange flows.

===============================================================================
OVERVIEW
===============================================================================

CoinMetrics provides comprehensive blockchain data including:
    - Network metrics (hash rate, difficulty, addresses)
    - Market metrics (OHLCV, market cap, realized cap)
    - On-chain indicators (SOPR, MVRV, NVT)
    - Exchange flows (deposits, withdrawals, reserves)
    - Miner metrics (revenue, outflows)
    - Institutional data (Grayscale, ETF flows)

Target Users:
    - Institutional investors
    - Quantitative funds
    - Research teams
    - Risk management

Key Differentiators:
    - professional-quality data quality
    - Comprehensive metric coverage
    - Standardized methodology
    - Long historical data

===============================================================================
API TIERS
===============================================================================

    ============== ==================== ============== ================
    Tier Rate Limit Assets Best For
    ============== ==================== ============== ================
    Community 10 req/min Limited Evaluation
    Starter 100 req/min 50+ Development
    Professional 500 req/min 200+ Production
    Enterprise Custom All Institutional
    ============== ==================== ============== ================

===============================================================================
METRIC CATEGORIES
===============================================================================

Network Metrics:
    - HashRate, DiffMean (hash rate, difficulty)
    - BlkCnt, BlkSizeMeanByte (block stats)
    - TxCnt, TxTfrCnt (transaction counts)
    - AdrActCnt, AdrBalCnt (address activity)

Market Metrics:
    - PriceUSD, PriceBTC (prices)
    - CapMrktCurUSD, CapRealUSD (market caps)
    - VtyDayRet30d (volatility)
    - SplyAct1d, SplyAct30d (active supply)

On-Chain Indicators:
    - NVTAdj, NVTAdj90 (Network Value to Transactions)
    - SOPR, SOPRadj (Spent Output Profit Ratio)
    - MVRV, MVRVCur (Market Value to Realized Value)
    - RevUSD, FeeTotUSD (revenue, fees)

Exchange Metrics:
    - FlowInExUSD, FlowOutExUSD (exchange flows)
    - SplyExBal (exchange balance)
    - Per-exchange flows (Binance, Coinbase, etc.)

Miner Metrics:
    - RevHashRateUSD (miner revenue per hash)
    - FlowMinerOut1HopAllUSD (miner outflows)
    - SplyMiner0HopAllNtv (miner holdings)

===============================================================================
USAGE EXAMPLES
===============================================================================

Network metrics:

    >>> from data_collection.onchain import CoinMetricsCollector
    >>> 
    >>> collector = CoinMetricsCollector({'coinmetrics_api_key': 'key'})
    >>> try:
    ... network = await collector.fetch_network_data(
    ... assets=['btc', 'eth'],
    ... start_date='2024-01-01',
    ... end_date='2024-03-31'
    ... )
    ... print(f"Records: {len(network)}")
    ... finally:
    ... await collector.close()

On-chain indicators:

    >>> indicators = await collector.fetch_onchain_indicators(
    ... assets=['btc'],
    ... start_date='2024-01-01',
    ... end_date='2024-03-31'
    ... )
    >>> # Check MVRV for valuation signals
    >>> mvrv = indicators[indicators['MVRV'].notna()]['MVRV']

Exchange flows:

    >>> flows = await collector.fetch_exchange_flows(
    ... assets=['btc', 'eth'],
    ... start_date='2024-01-01',
    ... end_date='2024-01-31'
    ... )
    >>> # Positive NetFlowExUSD = bearish (deposits > withdrawals)

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

Valuation Signals:
    - MVRV < 1 suggests undervaluation
    - NVT spikes may indicate overvaluation
    - Realized cap divergence from market cap

Flow Analysis:
    - Exchange inflows suggest selling pressure
    - Exchange outflows suggest accumulation
    - Miner outflows may indicate selling

Network Health:
    - Hash rate trends (security)
    - Active address growth
    - Transaction volume trends

Risk Assessment:
    - Volatility metrics
    - Supply concentration
    - Exchange reserve ratios

===============================================================================
DATA QUALITY CONSIDERATIONS
===============================================================================

- Metrics availability varies by asset
- Some metrics require paid tier
- Historical data depth varies
- Methodology is well-documented
- Data undergoes quality review

Version: 2.0.0
API Documentation: https://docs.coinmetrics.io/api/v4
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

class MetricCategory(Enum):
    """Metric categories."""
    NETWORK = 'network'
    MARKET = 'market'
    ONCHAIN = 'onchain'
    EXCHANGE = 'exchange'
    MINER = 'miner'
    INSTITUTION = 'institution'
    STAKING = 'staking'
    DEFI = 'defi'

class Frequency(Enum):
    """Data frequency options."""
    BLOCK = '1b'
    MINUTE = '1m'
    HOUR = '1h'
    DAY = '1d'
    WEEK = '1w'

class ValuationSignal(Enum):
    """Valuation signal based on MVRV."""
    UNDERVALUED = 'undervalued' # MVRV < 0.8
    FAIR = 'fair' # 0.8 <= MVRV < 1.5
    OVERVALUED = 'overvalued' # 1.5 <= MVRV < 3.0
    EXTREME = 'extreme' # MVRV >= 3.0
    UNKNOWN = 'unknown'

class FlowSignal(Enum):
    """Exchange flow signal."""
    STRONG_ACCUMULATION = 'strong_accumulation' # Large outflows
    ACCUMULATION = 'accumulation' # Net outflows
    NEUTRAL = 'neutral' # Balanced
    DISTRIBUTION = 'distribution' # Net inflows
    STRONG_DISTRIBUTION = 'strong_distribution' # Large inflows

class NetworkHealth(Enum):
    """Network health assessment."""
    EXCELLENT = 'excellent'
    GOOD = 'good'
    FAIR = 'fair'
    POOR = 'poor'
    CRITICAL = 'critical'

class AssetType(Enum):
    """Asset type classification."""
    POW = 'pow' # Proof of Work
    POS = 'pos' # Proof of Stake
    HYBRID = 'hybrid'
    TOKEN = 'token'

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class CoinMetricsAsset:
    """Asset metadata from CoinMetrics."""
    asset: str
    full_name: str
    metrics_count: int
    markets_count: int
    exchanges_count: int
    min_time: Optional[datetime]
    max_time: Optional[datetime]
    
    @property
    def data_years(self) -> float:
        """Years of data available."""
        if self.min_time and self.max_time:
            return (self.max_time - self.min_time).days / 365
        return 0
    
    @property
    def is_major_asset(self) -> bool:
        """Check if major asset (>100 metrics)."""
        return self.metrics_count >= 100
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'asset': self.asset,
            'full_name': self.full_name,
            'metrics_count': self.metrics_count,
            'markets_count': self.markets_count,
            'exchanges_count': self.exchanges_count,
            'min_time': self.min_time.isoformat() if self.min_time else None,
            'max_time': self.max_time.isoformat() if self.max_time else None,
            'data_years': self.data_years,
            'is_major_asset': self.is_major_asset,
            'venue': 'coinmetrics',
        }

@dataclass
class NetworkMetrics:
    """Network metrics dataclass."""
    timestamp: datetime
    asset: str
    hash_rate: Optional[float] = None
    difficulty: Optional[float] = None
    block_count: Optional[int] = None
    block_size_mean: Optional[float] = None
    tx_count: Optional[int] = None
    tx_transfer_count: Optional[int] = None
    tx_transfer_value_usd: Optional[float] = None
    active_addresses: Optional[int] = None
    addresses_with_balance: Optional[int] = None
    
    @property
    def tx_per_block(self) -> float:
        """Transactions per block."""
        if self.tx_count and self.block_count and self.block_count > 0:
            return self.tx_count / self.block_count
        return 0
    
    @property
    def avg_tx_value_usd(self) -> float:
        """Average transaction value in USD."""
        if self.tx_transfer_value_usd and self.tx_transfer_count and self.tx_transfer_count > 0:
            return self.tx_transfer_value_usd / self.tx_transfer_count
        return 0
    
    @property
    def network_health(self) -> NetworkHealth:
        """Assess network health based on activity."""
        if self.active_addresses is None:
            return NetworkHealth.FAIR
        if self.active_addresses > 1_000_000:
            return NetworkHealth.EXCELLENT
        elif self.active_addresses > 500_000:
            return NetworkHealth.GOOD
        elif self.active_addresses > 100_000:
            return NetworkHealth.FAIR
        elif self.active_addresses > 10_000:
            return NetworkHealth.POOR
        else:
            return NetworkHealth.CRITICAL
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'asset': self.asset,
            'hash_rate': self.hash_rate,
            'difficulty': self.difficulty,
            'block_count': self.block_count,
            'block_size_mean': self.block_size_mean,
            'tx_count': self.tx_count,
            'tx_transfer_count': self.tx_transfer_count,
            'tx_transfer_value_usd': self.tx_transfer_value_usd,
            'tx_per_block': self.tx_per_block,
            'avg_tx_value_usd': self.avg_tx_value_usd,
            'active_addresses': self.active_addresses,
            'addresses_with_balance': self.addresses_with_balance,
            'network_health': self.network_health.value,
            'venue': 'coinmetrics',
        }

@dataclass
class MarketMetrics:
    """Market metrics dataclass."""
    timestamp: datetime
    asset: str
    price_usd: Optional[float] = None
    price_btc: Optional[float] = None
    market_cap_usd: Optional[float] = None
    realized_cap_usd: Optional[float] = None
    volatility_30d: Optional[float] = None
    volatility_180d: Optional[float] = None
    supply_active_1d: Optional[float] = None
    supply_active_30d: Optional[float] = None
    supply_active_1y: Optional[float] = None
    
    @property
    def realized_premium(self) -> float:
        """Market cap premium over realized cap (MVRV proxy)."""
        if self.market_cap_usd and self.realized_cap_usd and self.realized_cap_usd > 0:
            return self.market_cap_usd / self.realized_cap_usd
        return 0
    
    @property
    def supply_velocity_30d(self) -> float:
        """30-day supply velocity ratio."""
        if self.supply_active_30d and self.market_cap_usd and self.market_cap_usd > 0:
            return self.supply_active_30d / self.market_cap_usd
        return 0
    
    @property
    def is_high_volatility(self) -> bool:
        """Check if high volatility (>80% annualized)."""
        if self.volatility_30d:
            return self.volatility_30d * 100 > 80
        return False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'asset': self.asset,
            'price_usd': self.price_usd,
            'price_btc': self.price_btc,
            'market_cap_usd': self.market_cap_usd,
            'realized_cap_usd': self.realized_cap_usd,
            'realized_premium': self.realized_premium,
            'volatility_30d': self.volatility_30d,
            'volatility_180d': self.volatility_180d,
            'is_high_volatility': self.is_high_volatility,
            'supply_active_1d': self.supply_active_1d,
            'supply_active_30d': self.supply_active_30d,
            'supply_active_1y': self.supply_active_1y,
            'supply_velocity_30d': self.supply_velocity_30d,
            'venue': 'coinmetrics',
        }

@dataclass
class OnChainIndicators:
    """On-chain valuation indicators."""
    timestamp: datetime
    asset: str
    nvt: Optional[float] = None
    nvt_90: Optional[float] = None
    sopr: Optional[float] = None
    sopr_adj: Optional[float] = None
    mvrv: Optional[float] = None
    mvrv_cur: Optional[float] = None
    revenue_usd: Optional[float] = None
    fees_usd: Optional[float] = None
    fee_mean_usd: Optional[float] = None
    
    @property
    def valuation_signal(self) -> ValuationSignal:
        """Get valuation signal based on MVRV."""
        if self.mvrv is None:
            return ValuationSignal.UNKNOWN
        if self.mvrv < 0.8:
            return ValuationSignal.UNDERVALUED
        elif self.mvrv < 1.5:
            return ValuationSignal.FAIR
        elif self.mvrv < 3.0:
            return ValuationSignal.OVERVALUED
        else:
            return ValuationSignal.EXTREME
    
    @property
    def is_sopr_profitable(self) -> bool:
        """Check if SOPR > 1 (profit taking)."""
        return self.sopr is not None and self.sopr > 1.0
    
    @property
    def is_nvt_elevated(self) -> bool:
        """Check if NVT is elevated (>90, overvalued signal)."""
        return self.nvt is not None and self.nvt > 90
    
    @property
    def fee_revenue_ratio(self) -> float:
        """Fees as percentage of total revenue."""
        if self.fees_usd and self.revenue_usd and self.revenue_usd > 0:
            return self.fees_usd / self.revenue_usd * 100
        return 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'asset': self.asset,
            'nvt': self.nvt,
            'nvt_90': self.nvt_90,
            'is_nvt_elevated': self.is_nvt_elevated,
            'sopr': self.sopr,
            'sopr_adj': self.sopr_adj,
            'is_sopr_profitable': self.is_sopr_profitable,
            'mvrv': self.mvrv,
            'mvrv_cur': self.mvrv_cur,
            'valuation_signal': self.valuation_signal.value,
            'revenue_usd': self.revenue_usd,
            'fees_usd': self.fees_usd,
            'fee_mean_usd': self.fee_mean_usd,
            'fee_revenue_ratio': self.fee_revenue_ratio,
            'venue': 'coinmetrics',
        }

@dataclass
class ExchangeFlowMetrics:
    """Exchange flow metrics."""
    timestamp: datetime
    asset: str
    flow_in_usd: Optional[float] = None
    flow_out_usd: Optional[float] = None
    flow_in_native: Optional[float] = None
    flow_out_native: Optional[float] = None
    exchange_balance: Optional[float] = None
    
    @property
    def net_flow_usd(self) -> float:
        """Net flow in USD (positive = inflows dominate)."""
        inflow = self.flow_in_usd or 0
        outflow = self.flow_out_usd or 0
        return inflow - outflow
    
    @property
    def net_flow_native(self) -> float:
        """Net flow in native token."""
        inflow = self.flow_in_native or 0
        outflow = self.flow_out_native or 0
        return inflow - outflow
    
    @property
    def flow_signal(self) -> FlowSignal:
        """Get flow signal."""
        net = self.net_flow_usd
        if net < -100_000_000:
            return FlowSignal.STRONG_ACCUMULATION
        elif net < -10_000_000:
            return FlowSignal.ACCUMULATION
        elif net < 10_000_000:
            return FlowSignal.NEUTRAL
        elif net < 100_000_000:
            return FlowSignal.DISTRIBUTION
        else:
            return FlowSignal.STRONG_DISTRIBUTION
    
    @property
    def is_net_inflow(self) -> bool:
        """Check if net inflows (bearish)."""
        return self.net_flow_usd > 0
    
    @property
    def is_net_outflow(self) -> bool:
        """Check if net outflows (bullish)."""
        return self.net_flow_usd < 0
    
    @property
    def flow_ratio(self) -> float:
        """Inflow/outflow ratio."""
        if self.flow_out_usd and self.flow_out_usd > 0:
            return (self.flow_in_usd or 0) / self.flow_out_usd
        return float('inf') if self.flow_in_usd else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'asset': self.asset,
            'flow_in_usd': self.flow_in_usd,
            'flow_out_usd': self.flow_out_usd,
            'flow_in_native': self.flow_in_native,
            'flow_out_native': self.flow_out_native,
            'net_flow_usd': self.net_flow_usd,
            'net_flow_native': self.net_flow_native,
            'flow_signal': self.flow_signal.value,
            'is_net_inflow': self.is_net_inflow,
            'is_net_outflow': self.is_net_outflow,
            'flow_ratio': self.flow_ratio,
            'exchange_balance': self.exchange_balance,
            'venue': 'coinmetrics',
        }

@dataclass
class MinerMetrics:
    """Miner-specific metrics for PoW chains."""
    timestamp: datetime
    asset: str
    revenue_per_hash_usd: Optional[float] = None
    revenue_per_hash_native: Optional[float] = None
    miner_outflow_usd: Optional[float] = None
    miner_outflow_native: Optional[float] = None
    miner_supply: Optional[float] = None
    
    @property
    def is_miner_selling(self) -> bool:
        """Check if miners are selling (high outflows)."""
        if self.miner_outflow_usd:
            return self.miner_outflow_usd > 10_000_000 # >$10M daily
        return False
    
    @property
    def miner_capitulation_risk(self) -> bool:
        """Check for miner capitulation signals."""
        if self.revenue_per_hash_usd and self.is_miner_selling:
            return True
        return False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'asset': self.asset,
            'revenue_per_hash_usd': self.revenue_per_hash_usd,
            'revenue_per_hash_native': self.revenue_per_hash_native,
            'miner_outflow_usd': self.miner_outflow_usd,
            'miner_outflow_native': self.miner_outflow_native,
            'miner_supply': self.miner_supply,
            'is_miner_selling': self.is_miner_selling,
            'miner_capitulation_risk': self.miner_capitulation_risk,
            'venue': 'coinmetrics',
        }

# =============================================================================
# Main Collector Class
# =============================================================================

class CoinMetricsCollector(BaseCollector):
    """
    CoinMetrics data collector for on-chain fundamentals.
    
    validated implementation providing professional-quality
    blockchain metrics for valuation and risk analysis.
    
    Features:
        - Network metrics (hash rate, addresses, transactions)
        - Market metrics (price, market cap, volatility)
        - On-chain indicators (MVRV, NVT, SOPR)
        - Exchange flows (deposits, withdrawals, reserves)
        - Miner metrics (revenue, outflows)
    
    Example:
        >>> collector = CoinMetricsCollector({'coinmetrics_api_key': 'key'})
        >>> try:
        ... indicators = await collector.fetch_onchain_indicators(
        ... assets=['btc', 'eth'],
        ... start_date='2024-01-01',
        ... end_date='2024-03-31'
        ... )
        ... finally:
        ... await collector.close()
    
    Attributes:
        VENUE: 'coinmetrics'
        VENUE_TYPE: 'analytics'
    """
    
    VENUE = 'coinmetrics'
    VENUE_TYPE = 'analytics'
    BASE_URL = 'https://api.coinmetrics.io/v4'
    
    # Core metric lists
    NETWORK_METRICS = [
        'HashRate', 'DiffMean', 'BlkCnt', 'BlkSizeMeanByte',
        'TxCnt', 'TxTfrCnt', 'TxTfrValAdjUSD', 'TxTfrValMeanUSD',
        'AdrActCnt', 'AdrBalCnt', 'AdrBal1in100KCnt', 'AdrBal1in1MCnt'
    ]
    
    MARKET_METRICS = [
        'PriceUSD', 'PriceBTC', 'CapMrktCurUSD', 'CapMrktFFUSD',
        'CapRealUSD', 'VtyDayRet30d', 'VtyDayRet180d',
        'SplyAct1d', 'SplyAct30d', 'SplyAct1yr'
    ]
    
    ONCHAIN_METRICS = [
        'NVTAdj', 'NVTAdj90', 'SOPR', 'SOPRadj',
        'MVRV', 'MVRVCur', 'RevUSD', 'RevNtv',
        'FeeMeanUSD', 'FeeMedUSD', 'FeeTotUSD'
    ]
    
    EXCHANGE_METRICS = [
        'FlowInExUSD', 'FlowOutExUSD', 'FlowInExNtv', 'FlowOutExNtv', 'SplyExBal'
    ]
    
    MINER_METRICS = [
        'RevHashRateUSD', 'RevHashRateNtv', 'FlowMinerOut1HopAllUSD',
        'FlowMinerOut1HopAllNtv', 'SplyMiner0HopAllNtv'
    ]
    
    # Major assets with comprehensive data
    MAJOR_ASSETS = [
        'btc', 'eth', 'sol', 'avax', 'matic', 'arb', 'op',
        'link', 'uni', 'aave', 'mkr', 'snx', 'crv', 'comp',
        'ada', 'dot', 'atom', 'near', 'ftm', 'algo'
    ]
    
    # PoW assets for miner metrics
    POW_ASSETS = ['btc', 'ltc', 'etc', 'doge', 'bch', 'zec']
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize CoinMetrics collector.
        
        Args:
            config: Configuration with coinmetrics_api_key
        """
        config = config or {}
        super().__init__(config)

        self.api_key = config.get('coinmetrics_api_key', '')
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = get_shared_rate_limiter(
            'coinmetrics',
            rate=config.get('rate_limit', 5),
            per=60,
            burst=config.get('burst', 3)
        )

        self.stats = {'requests': 0, 'records': 0, 'errors': 0}

        # Check if API key is valid (not a placeholder)
        self._api_key_valid = bool(self.api_key) and 'your_' not in self.api_key.lower()
        self._disabled_logged = False
        self._auth_failed = False

        if self._api_key_valid:
            logger.info("CoinMetrics collector initialized with API key")
        else:
            logger.info("CoinMetrics: No valid API key - collector disabled (requires paid plan)")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self.session
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make API request."""
        # Skip if already know auth failed or no valid key
        if self._auth_failed or not self._api_key_valid:
            if not self._disabled_logged:
                logger.debug("CoinMetrics: Skipping request - no valid API key")
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
                elif resp.status == 401 or resp.status == 403:
                    # Auth error - mark as disabled and stop spamming
                    if not self._auth_failed:
                        logger.info("CoinMetrics: Authentication failed (requires paid API key)")
                        self._auth_failed = True
                        self._api_key_valid = False
                    return None
                elif resp.status == 429:
                    logger.warning("CoinMetrics: Rate limited, waiting 60s")
                    await asyncio.sleep(60)
                    return None
                else:
                    logger.debug(f"CoinMetrics HTTP {resp.status}")
                    self.stats['errors'] += 1
                    return None
        except Exception as e:
            logger.error(f"CoinMetrics request error: {e}")
            self.stats['errors'] += 1
            return None
    
    async def fetch_catalog_assets(self) -> pd.DataFrame:
        """Fetch available assets and their supported metrics."""
        data = await self._make_request('catalog/assets')
        
        if not data or 'data' not in data:
            return pd.DataFrame()
        
        records = []
        for asset in data['data']:
            a = CoinMetricsAsset(
                asset=asset.get('asset', ''),
                full_name=asset.get('full_name', ''),
                metrics_count=len(asset.get('metrics', [])),
                markets_count=len(asset.get('markets', [])),
                exchanges_count=len(asset.get('exchanges', [])),
                min_time=pd.to_datetime(asset.get('min_time')) if asset.get('min_time') else None,
                max_time=pd.to_datetime(asset.get('max_time')) if asset.get('max_time') else None
            )
            records.append(a.to_dict())
        
        self.stats['records'] += len(records)
        return pd.DataFrame(records)
    
    async def fetch_asset_metrics(
        self,
        assets: List[str],
        metrics: List[str],
        start_date: str,
        end_date: str,
        frequency: str = '1d'
    ) -> pd.DataFrame:
        """Fetch time series metrics for assets."""
        all_data = []
        metrics_per_request = 10
        
        for i in range(0, len(metrics), metrics_per_request):
            metric_batch = metrics[i:i + metrics_per_request]
            
            params = {
                'assets': ','.join(assets),
                'metrics': ','.join(metric_batch),
                'start_time': start_date,
                'end_time': end_date,
                'frequency': frequency,
                'page_size': 10000
            }
            
            data = await self._make_request('timeseries/asset-metrics', params)
            
            if data and 'data' in data:
                for record in data['data']:
                    row = {
                        'timestamp': pd.to_datetime(record.get('time')),
                        'asset': record.get('asset')
                    }
                    for metric in metric_batch:
                        value = record.get(metric)
                        if value is not None:
                            try:
                                row[metric] = float(value)
                            except (ValueError, TypeError):
                                row[metric] = value
                    all_data.append(row)
                
                # Handle pagination
                next_page = data.get('next_page_url')
                while next_page:
                    async with (await self._get_session()).get(next_page) as resp:
                        if resp.status == 200:
                            page_data = await resp.json()
                            for record in page_data.get('data', []):
                                row = {
                                    'timestamp': pd.to_datetime(record.get('time')),
                                    'asset': record.get('asset')
                                }
                                for metric in metric_batch:
                                    value = record.get(metric)
                                    if value is not None:
                                        try:
                                            row[metric] = float(value)
                                        except (ValueError, TypeError):
                                            row[metric] = value
                                all_data.append(row)
                            next_page = page_data.get('next_page_url')
                        else:
                            break
        
        self.stats['records'] += len(all_data)
        
        df = pd.DataFrame(all_data)
        if not df.empty:
            df = df.sort_values(['timestamp', 'asset']).reset_index(drop=True)
        return df
    
    async def fetch_network_data(
        self,
        assets: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch network metrics (hash rate, difficulty, addresses)."""
        return await self.fetch_asset_metrics(
            assets=assets,
            metrics=self.NETWORK_METRICS,
            start_date=start_date,
            end_date=end_date
        )
    
    async def fetch_market_data(
        self,
        assets: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch market metrics (price, cap, volatility)."""
        return await self.fetch_asset_metrics(
            assets=assets,
            metrics=self.MARKET_METRICS,
            start_date=start_date,
            end_date=end_date
        )
    
    async def fetch_onchain_indicators(
        self,
        assets: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch on-chain valuation indicators (SOPR, MVRV, NVT)."""
        return await self.fetch_asset_metrics(
            assets=assets,
            metrics=self.ONCHAIN_METRICS,
            start_date=start_date,
            end_date=end_date
        )
    
    async def fetch_exchange_flows(
        self,
        assets: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch exchange flow metrics."""
        df = await self.fetch_asset_metrics(
            assets=assets,
            metrics=self.EXCHANGE_METRICS,
            start_date=start_date,
            end_date=end_date
        )
        
        if not df.empty:
            if 'FlowInExUSD' in df.columns and 'FlowOutExUSD' in df.columns:
                df['NetFlowExUSD'] = df['FlowInExUSD'] - df['FlowOutExUSD']
            if 'FlowInExNtv' in df.columns and 'FlowOutExNtv' in df.columns:
                df['NetFlowExNtv'] = df['FlowInExNtv'] - df['FlowOutExNtv']
        
        return df
    
    async def fetch_miner_data(
        self,
        assets: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch miner metrics (revenue, outflows)."""
        pow_assets = [a for a in assets if a in self.POW_ASSETS]
        if not pow_assets:
            logger.info("No PoW assets provided for miner data")
            return pd.DataFrame()
        
        return await self.fetch_asset_metrics(
            assets=pow_assets,
            metrics=self.MINER_METRICS,
            start_date=start_date,
            end_date=end_date
        )
    
    async def fetch_comprehensive_fundamentals(
        self,
        assets: Optional[List[str]] = None,
        start_date: str = '2022-01-01',
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch comprehensive fundamental data for assets."""
        if assets is None:
            assets = self.MAJOR_ASSETS
        if end_date is None:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')

        logger.info(f"Fetching CoinMetrics fundamentals for {len(assets)} assets")

        # PARALLELIZED: Fetch all fundamental data concurrently
        pow_assets = [a for a in assets if a in self.POW_ASSETS]

        tasks = [
            self.fetch_network_data(assets, start_date, end_date),
            self.fetch_market_data(assets, start_date, end_date),
            self.fetch_onchain_indicators(assets, start_date, end_date),
            self.fetch_exchange_flows(assets, start_date, end_date),
        ]

        # Add miner data task only if there are PoW assets
        if pow_assets:
            tasks.append(self.fetch_miner_data(pow_assets, start_date, end_date))

        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build results dictionary
        results = {}
        result_keys = ['network', 'market', 'onchain', 'exchange_flows']
        if pow_assets:
            result_keys.append('miner')

        for i, key in enumerate(result_keys):
            if i < len(task_results) and isinstance(task_results[i], pd.DataFrame):
                results[key] = task_results[i]
                logger.info(f"{key.capitalize()} metrics: {len(results[key])} records")
            else:
                results[key] = pd.DataFrame()
                logger.warning(f"{key.capitalize()} metrics: Failed to fetch")

        return results
    
    async def fetch_funding_rates(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """CoinMetrics doesn't provide funding rates - returns empty DataFrame."""
        logger.info("CoinMetrics doesn't provide funding rate data directly")
        return pd.DataFrame()
    
    async def fetch_ohlcv(self, symbols: List[str], timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch OHLCV via market metrics."""
        return await self.fetch_market_data(symbols, start_date, end_date)
    
    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info(f"CoinMetrics session closed. Stats: {self.stats}")
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.stats.copy()
    
    @classmethod
    def get_major_assets(cls) -> List[str]:
        """Get list of major assets."""
        return cls.MAJOR_ASSETS.copy()
    
    @classmethod
    def get_pow_assets(cls) -> List[str]:
        """Get list of PoW assets."""
        return cls.POW_ASSETS.copy()
    
    @classmethod
    def get_metric_categories(cls) -> Dict[str, List[str]]:
        """Get metrics by category."""
        return {
            'network': cls.NETWORK_METRICS,
            'market': cls.MARKET_METRICS,
            'onchain': cls.ONCHAIN_METRICS,
            'exchange': cls.EXCHANGE_METRICS,
            'miner': cls.MINER_METRICS,
        }

async def test_coinmetrics_collector():
    """Test CoinMetrics collector functionality."""
    collector = CoinMetricsCollector({'rate_limit': 5})
    
    try:
        print("=" * 60)
        print("CoinMetrics Collector Test")
        print("=" * 60)
        print(f"\nMajor assets: {CoinMetricsCollector.get_major_assets()}")
        print(f"PoW assets: {CoinMetricsCollector.get_pow_assets()}")
        print(f"Metric categories: {list(CoinMetricsCollector.get_metric_categories().keys())}")
        print(f"\nStats: {collector.get_collection_stats()}")
    finally:
        await collector.close()

if __name__ == '__main__':
    asyncio.run(test_coinmetrics_collector())