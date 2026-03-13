"""
Curve Finance Data Collector

validated collector for Curve Finance - the dominant stablecoin and pegged asset DEX.
Curve specializes in low-slippage swaps between similarly-priced assets using its StableSwap
invariant, making it critical for stablecoin arbitrage and depeg monitoring.

Protocol Features:
    - StableSwap: Optimized AMM for pegged assets (low slippage)
    - CryptoSwap: V2 pools for volatile assets
    - Virtual Price: LP token appreciation tracking
    - Gauge System: CRV emission distribution

Data Categories:
    - Pool Liquidity: TVL, token balances, virtual prices
    - Swap Volume: Trading volume and fees by pool
    - Virtual Price: LP token appreciation (yield + fees)
    - Depeg Detection: Price deviation monitoring
    - Gauge Metrics: CRV emissions and voting

Supported Chains:
    - Ethereum Mainnet (primary)
    - Arbitrum
    - Optimism
    - Polygon
    - Avalanche
    - Base

API Sources:
    - The Graph Subgraphs (historical data)
    - Curve API (real-time pool data)

Statistical Arbitrage Applications:
    - Stablecoin depeg arbitrage (virtual price deviations)
    - Cross-pool arbitrage (3pool vs specialized pools)
    - LP token yield farming optimization
    - Gauge weight arbitrage (CRV emissions)

Version: 2.0.0
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..base_collector import BaseCollector
from ..utils.rate_limiter import get_shared_rate_limiter
from ..utils.retry_handler import RetryHandler

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================

class CurveChain(Enum):
    """Supported Curve chains."""
    ETHEREUM = 'ethereum'
    ARBITRUM = 'arbitrum'
    OPTIMISM = 'optimism'
    POLYGON = 'polygon'
    AVALANCHE = 'avalanche'
    BASE = 'base'
    FANTOM = 'fantom'
    GNOSIS = 'gnosis'

class PoolType(Enum):
    """Curve pool types."""
    STABLESWAP = 'stableswap' # Original stable pools (3pool, FRAX, etc.)
    CRYPTOSWAP = 'cryptoswap' # V2 volatile asset pools
    TRICRYPTO = 'tricrypto' # BTC/ETH/USDT style pools
    FACTORY = 'factory' # Permissionless factory pools
    LENDING = 'lending' # Pools with lending integration (aave, compound)
    META = 'meta' # Metapools using 3pool LP

class PegStatus(Enum):
    """Stablecoin peg status classification."""
    STABLE = 'stable' # Within 0.1% of peg
    MINOR_DEVIATION = 'minor' # 0.1-0.5% deviation
    MODERATE_DEVIATION = 'moderate' # 0.5-2% deviation
    SIGNIFICANT_DEPEG = 'significant' # 2-5% deviation
    SEVERE_DEPEG = 'severe' # >5% deviation

class VirtualPriceTrend(Enum):
    """Virtual price trend classification."""
    GROWING = 'growing' # Normal fee accumulation
    STABLE = 'stable' # No significant change
    DECLINING = 'declining' # Potential IL or withdrawal
    ANOMALY = 'anomaly' # Unusual movement (exploit?)

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class CurvePool:
    """Curve pool data with analytics."""
    timestamp: datetime
    pool_id: str
    pool_name: str
    pool_address: str
    pool_type: str
    chain: str
    
    # Liquidity
    tvl_usd: float
    token_count: int
    tokens: List[str]
    token_balances: List[float]
    
    # Performance
    cumulative_volume_usd: float
    daily_volume_usd: float
    protocol_revenue_usd: float
    lp_revenue_usd: float
    
    # LP metrics
    virtual_price: float
    lp_token_supply: float
    
    # Pool parameters
    amplification: Optional[int] # A parameter for StableSwap
    fee_pct: float
    admin_fee_pct: float
    
    @property
    def utilization_rate(self) -> float:
        """Daily volume / TVL ratio."""
        return self.daily_volume_usd / self.tvl_usd if self.tvl_usd > 0 else 0
    
    @property
    def lp_apy_estimate(self) -> float:
        """Estimated LP APY from fees (annualized)."""
        if self.tvl_usd > 0:
            daily_fees = self.daily_volume_usd * (self.fee_pct / 100) * (1 - self.admin_fee_pct / 100)
            return (daily_fees / self.tvl_usd) * 365 * 100
        return 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'pool_id': self.pool_id,
            'pool_name': self.pool_name,
            'pool_address': self.pool_address,
            'pool_type': self.pool_type,
            'chain': self.chain,
            'tvl_usd': self.tvl_usd,
            'token_count': self.token_count,
            'tokens': self.tokens,
            'token_balances': self.token_balances,
            'cumulative_volume_usd': self.cumulative_volume_usd,
            'daily_volume_usd': self.daily_volume_usd,
            'protocol_revenue_usd': self.protocol_revenue_usd,
            'lp_revenue_usd': self.lp_revenue_usd,
            'virtual_price': self.virtual_price,
            'lp_token_supply': self.lp_token_supply,
            'amplification': self.amplification,
            'fee_pct': self.fee_pct,
            'admin_fee_pct': self.admin_fee_pct,
            'utilization_rate': self.utilization_rate,
            'lp_apy_estimate': self.lp_apy_estimate,
        }

@dataclass
class PoolSnapshot:
    """Daily pool snapshot data."""
    timestamp: datetime
    date: datetime
    pool_id: str
    pool_name: str
    chain: str
    
    # Liquidity
    tvl_usd: float
    tvl_change_1d: float
    
    # Volume
    daily_volume_usd: float
    cumulative_volume_usd: float
    
    # Revenue
    daily_lp_revenue_usd: float
    daily_protocol_revenue_usd: float
    
    # LP metrics
    virtual_price: float
    virtual_price_change_1d: float
    lp_supply: float
    
    @property
    def fee_apy(self) -> float:
        """Fee-based APY estimate."""
        if self.tvl_usd > 0:
            return (self.daily_lp_revenue_usd / self.tvl_usd) * 365 * 100
        return 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'date': self.date,
            'pool_id': self.pool_id,
            'pool_name': self.pool_name,
            'chain': self.chain,
            'tvl_usd': self.tvl_usd,
            'tvl_change_1d': self.tvl_change_1d,
            'daily_volume_usd': self.daily_volume_usd,
            'cumulative_volume_usd': self.cumulative_volume_usd,
            'daily_lp_revenue_usd': self.daily_lp_revenue_usd,
            'daily_protocol_revenue_usd': self.daily_protocol_revenue_usd,
            'virtual_price': self.virtual_price,
            'virtual_price_change_1d': self.virtual_price_change_1d,
            'lp_supply': self.lp_supply,
            'fee_apy': self.fee_apy,
        }

@dataclass
class CurveSwap:
    """Individual swap event data."""
    timestamp: datetime
    tx_hash: str
    pool_id: str
    pool_name: str
    chain: str
    
    # Swap details
    token_in_symbol: str
    token_in_address: str
    token_out_symbol: str
    token_out_address: str
    
    amount_in: float
    amount_in_usd: float
    amount_out: float
    amount_out_usd: float
    
    # Execution
    exchange_rate: float
    slippage_pct: float
    fee_usd: float
    
    @property
    def pair(self) -> str:
        """Trading pair string."""
        return f"{self.token_in_symbol}/{self.token_out_symbol}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'tx_hash': self.tx_hash,
            'pool_id': self.pool_id,
            'pool_name': self.pool_name,
            'chain': self.chain,
            'token_in_symbol': self.token_in_symbol,
            'token_in_address': self.token_in_address,
            'token_out_symbol': self.token_out_symbol,
            'token_out_address': self.token_out_address,
            'amount_in': self.amount_in,
            'amount_in_usd': self.amount_in_usd,
            'amount_out': self.amount_out,
            'amount_out_usd': self.amount_out_usd,
            'exchange_rate': self.exchange_rate,
            'slippage_pct': self.slippage_pct,
            'fee_usd': self.fee_usd,
            'pair': self.pair,
        }

@dataclass
class VirtualPriceData:
    """Virtual price tracking for depeg detection."""
    timestamp: datetime
    pool_address: str
    pool_name: str
    chain: str
    
    virtual_price: float
    virtual_price_raw: int # Raw 18 decimal value
    tvl_usd: float
    volume_24h: float
    apy: float
    
    # Peg analysis
    expected_price: float # Expected virtual price based on time
    price_deviation_pct: float
    peg_status: str
    
    @property
    def is_anomaly(self) -> bool:
        """Check if virtual price shows anomalous behavior."""
        # Virtual price should generally only go up (from fees)
        # Significant drops may indicate exploit or manipulation
        return self.price_deviation_pct < -0.5
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'pool_address': self.pool_address,
            'pool_name': self.pool_name,
            'chain': self.chain,
            'virtual_price': self.virtual_price,
            'tvl_usd': self.tvl_usd,
            'volume_24h': self.volume_24h,
            'apy': self.apy,
            'expected_price': self.expected_price,
            'price_deviation_pct': self.price_deviation_pct,
            'peg_status': self.peg_status,
            'is_anomaly': self.is_anomaly,
        }

# =============================================================================
# Collector Class
# =============================================================================

class CurveCollector(BaseCollector):
    """
    Curve Finance data collector using The Graph subgraphs and Curve API.
    
    Provides comprehensive stablecoin and pegged asset pool data:
    - Pool liquidity and TVL tracking
    - Virtual price monitoring (depeg detection)
    - Swap volume and fee revenue
    - LP token metrics and yields
    
    Key Curve Concepts:
    - Virtual Price: LP token value in underlying, grows with fees
    - Amplification (A): Concentration parameter for StableSwap
    - Gauge: CRV emission distribution mechanism
    
    Attributes:
        VENUE: Venue identifier ('curve')
        VENUE_TYPE: Venue classification ('DEX')
        SUBGRAPH_URLS: Chain-specific subgraph endpoints
        API_URLS: Chain-specific Curve API endpoints
    
    Example:
        >>> config = {'rate_limit': 15}
        >>> async with CurveCollector(config) as collector:
        ... pools = await collector.fetch_pools(min_tvl=1_000_000)
        ... virtual_prices = await collector.fetch_virtual_prices()
        ... swaps = await collector.fetch_swaps(pool_id, '2024-01-01', '2024-01-31')
    """
    
    VENUE = 'curve'
    VENUE_TYPE = 'DEX'
    
    # Subgraph URLs by chain
    SUBGRAPH_URLS = {
        'ethereum': 'https://api.thegraph.com/subgraphs/name/curvefi/curve',
        'arbitrum': 'https://api.thegraph.com/subgraphs/name/curvefi/curve-arbitrum',
        'optimism': 'https://api.thegraph.com/subgraphs/name/curvefi/curve-optimism',
        'polygon': 'https://api.thegraph.com/subgraphs/name/curvefi/curve-polygon',
        'avalanche': 'https://api.thegraph.com/subgraphs/name/curvefi/curve-avalanche',
        'base': 'https://api.thegraph.com/subgraphs/name/curvefi/curve-base',
        'fantom': 'https://api.thegraph.com/subgraphs/name/curvefi/curve-fantom',
    }
    
    # Curve API URLs for real-time data
    API_URLS = {
        'ethereum': 'https://api.curve.fi/api/getPools/ethereum/main',
        'arbitrum': 'https://api.curve.fi/api/getPools/arbitrum/main',
        'optimism': 'https://api.curve.fi/api/getPools/optimism/main',
        'polygon': 'https://api.curve.fi/api/getPools/polygon/main',
        'avalanche': 'https://api.curve.fi/api/getPools/avalanche/main',
        'base': 'https://api.curve.fi/api/getPools/base/main',
    }
    
    # Notable pools for monitoring
    NOTABLE_POOLS = {
        'ethereum': {
            '3pool': '0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7',
            'steth': '0xdc24316b9ae028f1497c275eb9192a3ea0f67022',
            'fraxusdc': '0xdcef968d416a41cdac0ed8702fac8128a64241a2',
            'tricrypto2': '0xd51a44d3fae010294c616388b506acda1bfaae46',
        },
        'arbitrum': {
            '2pool': '0x7f90122bf0700f9e7e1f688fe926940e8839f353',
            'tricrypto': '0x960ea3e3c7fb317332d990873d354e18d7645590',
        }
    }
    
    # Default rate limit
    DEFAULT_RATE_LIMIT = 15
    
    def __init__(self, config: Dict):
        """
        Initialize Curve collector.

        Args:
            config: Configuration dictionary with optional keys:
                - rate_limit: Requests per minute (default: 15)
                - timeout: Request timeout in seconds (default: 60)
                - graph_api_key: Optional Graph Protocol API key
        """
        super().__init__(config)

        # CRITICAL: Set supported data types for dynamic routing
        self.supported_data_types = ['pool_data', 'swaps']
        self.venue = 'curve'

        # Import VenueType from base_collector
        from ..base_collector import VenueType
        self.venue_type = VenueType.DEX
        self.requires_auth = True # Requires The Graph API key

        self.session: Optional[aiohttp.ClientSession] = None
        self.api_key = config.get('graph_api_key', '')
        # Use shared rate limiter to avoid re-initialization overhead
        self.rate_limiter = get_shared_rate_limiter(
            'curve',
            rate=config.get('rate_limit', self.DEFAULT_RATE_LIMIT),
            per=60,
            burst=5
        )
        self.retry_handler = RetryHandler(
            max_retries=config.get('max_retries', 3),
            base_delay=1.0,
            max_delay=30.0
        )
        self._timeout = config.get('timeout', 60)

        # Collection statistics
        self.collection_stats = {
            'records_collected': 0,
            'api_calls': 0,
            'subgraph_calls': 0,
            'errors': 0,
        }
        logger.info(f"Initialized Curve collector with data types: {self.supported_data_types}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            # SPEEDUP: Aggressive connection pooling for high-throughput collection
            connector = aiohttp.TCPConnector(
                limit=50, # Total connection pool size (was 10)
                limit_per_host=15, # Per-host connections
                ttl_dns_cache=300, # DNS cache TTL in seconds
                force_close=False, # Keep-alive connections
                enable_cleanup_closed=True
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'Accept': 'application/json'}
            )
        return self.session
    
    async def _subgraph_query(
        self,
        query: str,
        variables: Optional[Dict] = None,
        chain: str = 'ethereum'
    ) -> Dict:
        """
        Query Curve subgraph.
        
        Args:
            query: GraphQL query
            variables: Query variables
            chain: Target chain
            
        Returns:
            Query result data
        """
        url = self.SUBGRAPH_URLS.get(chain)
        if not url:
            logger.error(f"Unsupported chain for subgraph: {chain}")
            return {}
        
        session = await self._get_session()
        await self.rate_limiter.acquire()
        self.collection_stats['subgraph_calls'] += 1
        
        payload = {'query': query}
        if variables:
            payload['variables'] = variables
        
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    if 'errors' in result:
                        errors = result.get('errors', [])
                        # Check if this is a deprecated endpoint error
                        if any('removed' in str(e.get('message', '')).lower() for e in errors):
                            logger.debug(f"Curve subgraph deprecated for {chain}, using REST API fallback")
                        else:
                            logger.debug(f"Curve GraphQL errors: {errors[:1]}")
                        return {}
                    return result.get('data', {})
                else:
                    logger.debug(f"Curve subgraph error {resp.status} for {chain}")
                    return {}
        except Exception as e:
            logger.debug(f"Curve subgraph query failed: {e}")
            self.collection_stats['errors'] += 1
            return {}
    
    async def _api_request(
        self,
        url: str
    ) -> Dict:
        """
        Make request to Curve API.
        
        Args:
            url: Full API URL
            
        Returns:
            JSON response data
        """
        session = await self._get_session()
        await self.rate_limiter.acquire()
        self.collection_stats['api_calls'] += 1
        
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.warning(f"Curve API error {resp.status}")
                    return {}
        except Exception as e:
            logger.error(f"Curve API request failed: {e}")
            self.collection_stats['errors'] += 1
            return {}
    
    # =========================================================================
    # Pool Methods
    # =========================================================================

    async def _process_single_pool(
        self, pool: Dict, chain: str, pool_type: Optional[str]
    ) -> Optional[Dict]:
        """Process a single pool entry (helper for parallelization)."""
        try:
            tokens = pool.get('inputTokens', [])
            token_symbols = [t.get('symbol', '') for t in tokens]

            # Determine pool type
            detected_type = self._classify_pool_type(pool.get('name', ''), token_symbols)

            if pool_type and detected_type != pool_type:
                return None

            balances = pool.get('inputTokenBalances', [])
            token_balances = [float(b) / 1e18 for b in balances] if balances else []

            tvl = float(pool.get('totalValueLockedUSD', 0))
            cum_volume = float(pool.get('cumulativeVolumeUSD', 0))

            return {
                'timestamp': datetime.utcnow(),
                'pool_id': pool['id'],
                'pool_name': pool.get('name', ''),
                'pool_symbol': pool.get('symbol', ''),
                'pool_address': pool['id'],
                'pool_type': detected_type,
                'tvl_usd': tvl,
                'cumulative_volume_usd': cum_volume,
                'protocol_revenue_usd': float(pool.get('cumulativeProtocolSideRevenueUSD', 0)),
                'lp_revenue_usd': float(pool.get('cumulativeSupplySideRevenueUSD', 0)),
                'token_count': len(tokens),
                'tokens': ','.join(token_symbols),
                'token_list': token_symbols,
                'token_balances': token_balances,
                'lp_token_supply': float(pool.get('outputTokenSupply', 0) or 0) / 1e18,
                'created_timestamp': int(pool.get('createdTimestamp', 0)),
                'chain': chain,
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE,
            }
        except Exception as e:
            logger.debug(f"Error processing pool: {e}")
            return None

    async def fetch_pools(
        self,
        chain: str = 'ethereum',
        min_tvl: float = 100_000,
        pool_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch Curve pool data from subgraph.
        
        Args:
            chain: Target chain
            min_tvl: Minimum TVL filter in USD
            pool_type: Filter by pool type (stableswap, crypto, etc.)
            
        Returns:
            DataFrame with pool information
        """
        logger.info(f"Fetching Curve pools from {chain} (min TVL: ${min_tvl:,.0f})")
        
        query = """
        query GetPools($minTvl: BigDecimal!, $skip: Int!) {
            liquidityPools(
                first: 1000,
                skip: $skip,
                orderBy: totalValueLockedUSD,
                orderDirection: desc,
                where: { totalValueLockedUSD_gte: $minTvl }
            ) {
                id
                name
                symbol
                totalValueLockedUSD
                cumulativeVolumeUSD
                cumulativeProtocolSideRevenueUSD
                cumulativeSupplySideRevenueUSD
                inputTokens {
                    id
                    symbol
                    decimals
                }
                inputTokenBalances
                outputToken {
                    id
                    symbol
                    decimals
                }
                outputTokenSupply
                createdTimestamp
                createdBlockNumber
            }
        }
        """
        
        all_pools = []
        skip = 0
        
        while True:
            variables = {'minTvl': str(min_tvl), 'skip': skip}
            data = await self._subgraph_query(query, variables, chain)
            
            pools = data.get('liquidityPools', [])
            if not pools:
                break

            # PARALLELIZED: Process all pools concurrently
            tasks = [
                self._process_single_pool(pool, chain, pool_type)
                for pool in pools
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out None results and errors
            for result in results:
                if isinstance(result, dict):
                    all_pools.append(result)

            skip += len(pools)
            logger.debug(f"Fetched {skip} pools")
            
            if len(pools) < 1000:
                break
        
        df = pd.DataFrame(all_pools)
        
        if not df.empty:
            df = df.sort_values('tvl_usd', ascending=False).reset_index(drop=True)
            df['tvl_rank'] = df.index + 1
            
            # Calculate utilization
            df['volume_tvl_ratio'] = df['cumulative_volume_usd'] / df['tvl_usd'].replace(0, np.nan)
        
        self.collection_stats['records_collected'] += len(df)
        logger.info(f"Fetched {len(df)} Curve pools")

        # Fallback to REST API if The Graph returns no data
        if df.empty:
            logger.warning("The Graph returned no pools, trying Curve REST API...")
            df = await self._fetch_pools_from_api(chain, min_tvl)

        return df

    async def _fetch_pools_from_api(self, chain: str = 'ethereum', min_tvl: float = 100_000) -> pd.DataFrame:
        """Fetch pools from Curve's REST API (fallback when The Graph is down)."""
        try:
            api_url = self.API_URLS.get(chain)
            if not api_url:
                logger.warning(f"No Curve API URL for chain: {chain}")
                return pd.DataFrame()

            session = await self._get_session()
            async with session.get(api_url) as resp:
                if resp.status != 200:
                    logger.error(f"Curve API error {resp.status}")
                    return pd.DataFrame()

                data = await resp.json()

                if not data or 'data' not in data:
                    return pd.DataFrame()

                all_pools = []
                pools_data = data['data'].get('poolData', [])

                for pool in pools_data:
                    try:
                        tvl = float(pool.get('usdTotal', 0))
                        if tvl < min_tvl:
                            continue

                        coins = pool.get('coins', [])
                        token_symbols = [c.get('symbol', '') for c in coins]

                        all_pools.append({
                            'timestamp': datetime.utcnow(),
                            'pool_id': pool.get('address', ''),
                            'pool_name': pool.get('name', ''),
                            'pool_symbol': pool.get('symbol', ''),
                            'pool_address': pool.get('address', ''),
                            'pool_type': 'curve',
                            'tvl_usd': tvl,
                            'cumulative_volume_usd': float(pool.get('volumeUSD', 0)),
                            'token_count': len(coins),
                            'tokens': ','.join(token_symbols),
                            'token_list': token_symbols,
                            'chain': chain,
                            'venue': self.VENUE,
                            'venue_type': self.VENUE_TYPE,
                            'source': 'curve_api'
                        })
                    except Exception as e:
                        logger.debug(f"Error parsing Curve pool: {e}")
                        continue

                df = pd.DataFrame(all_pools)
                if not df.empty:
                    df = df.sort_values('tvl_usd', ascending=False).reset_index(drop=True)
                    logger.info(f"Fetched {len(df)} Curve pools from REST API")

                return df

        except Exception as e:
            logger.error(f"Curve API fetch error: {e}")
            return pd.DataFrame()
    
    async def fetch_pool_daily_snapshots(
        self,
        pool_id: str,
        chain: str = 'ethereum',
        start_date: str = '2022-01-01',
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch daily snapshots for a specific pool.
        
        Args:
            pool_id: Pool identifier/address
            chain: Target chain
            start_date: Start date
            end_date: End date (default: today)
            
        Returns:
            DataFrame with daily pool metrics
        """
        logger.info(f"Fetching daily snapshots for pool {pool_id[:20]}...")
        
        if end_date is None:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')
        
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        
        query = """
        query GetPoolDailySnapshots($poolId: String!, $startTime: Int!, $endTime: Int!, $skip: Int!) {
            liquidityPoolDailySnapshots(
                first: 1000,
                skip: $skip,
                orderBy: timestamp,
                where: {
                    pool: $poolId,
                    timestamp_gte: $startTime,
                    timestamp_lte: $endTime
                }
            ) {
                id
                timestamp
                totalValueLockedUSD
                cumulativeVolumeUSD
                dailyVolumeUSD
                dailySupplySideRevenueUSD
                dailyProtocolSideRevenueUSD
                inputTokenBalances
                outputTokenSupply
                rewardTokenEmissionsAmount
                rewardTokenEmissionsUSD
            }
        }
        """
        
        all_snapshots = []
        skip = 0
        
        while True:
            variables = {
                'poolId': pool_id,
                'startTime': start_ts,
                'endTime': end_ts,
                'skip': skip
            }
            
            data = await self._subgraph_query(query, variables, chain)
            snapshots = data.get('liquidityPoolDailySnapshots', [])
            
            if not snapshots:
                break
            
            for snap in snapshots:
                ts = pd.to_datetime(int(snap['timestamp']), unit='s', utc=True)
                
                all_snapshots.append({
                    'timestamp': ts,
                    'date': ts.date(),
                    'pool_id': pool_id,
                    'tvl_usd': float(snap.get('totalValueLockedUSD', 0)),
                    'cumulative_volume_usd': float(snap.get('cumulativeVolumeUSD', 0)),
                    'daily_volume_usd': float(snap.get('dailyVolumeUSD', 0)),
                    'daily_lp_revenue_usd': float(snap.get('dailySupplySideRevenueUSD', 0)),
                    'daily_protocol_revenue_usd': float(snap.get('dailyProtocolSideRevenueUSD', 0)),
                    'lp_supply': float(snap.get('outputTokenSupply', 0) or 0) / 1e18,
                    'reward_emissions_usd': float(snap.get('rewardTokenEmissionsUSD', 0) or 0),
                    'chain': chain,
                    'venue': self.VENUE,
                })
            
            skip += len(snapshots)
            
            if len(snapshots) < 1000:
                break
        
        df = pd.DataFrame(all_snapshots)
        
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate changes
            df['tvl_change_1d'] = df['tvl_usd'].pct_change() * 100
            df['volume_change_1d'] = df['daily_volume_usd'].pct_change() * 100
            
            # Calculate fee APY
            df['fee_apy'] = (df['daily_lp_revenue_usd'] / df['tvl_usd'].replace(0, np.nan)) * 365 * 100
            
            # Rolling metrics
            df['volume_sma_7d'] = df['daily_volume_usd'].rolling(7).mean()
            df['tvl_sma_7d'] = df['tvl_usd'].rolling(7).mean()
        
        return df
    
    # =========================================================================
    # Swap Methods
    # =========================================================================
    
    async def fetch_swaps(
        self,
        pool_id: str,
        start_date: str,
        end_date: str,
        chain: str = 'ethereum',
        limit: int = 10000
    ) -> pd.DataFrame:
        """
        Fetch swap events for a specific pool.
        
        Args:
            pool_id: Pool identifier/address
            start_date: Start date
            end_date: End date
            chain: Target chain
            limit: Maximum swaps to fetch
            
        Returns:
            DataFrame with swap events
        """
        logger.info(f"Fetching swaps for pool {pool_id[:20]}...")
        
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        
        query = """
        query GetSwaps($poolId: String!, $startTime: Int!, $endTime: Int!, $skip: Int!) {
            swaps(
                first: 1000,
                skip: $skip,
                orderBy: timestamp,
                where: {
                    pool: $poolId,
                    timestamp_gte: $startTime,
                    timestamp_lte: $endTime
                }
            ) {
                id
                hash
                timestamp
                tokenIn {
                    id
                    symbol
                    decimals
                }
                tokenOut {
                    id
                    symbol
                    decimals
                }
                amountIn
                amountInUSD
                amountOut
                amountOutUSD
            }
        }
        """
        
        all_swaps = []
        skip = 0
        
        while len(all_swaps) < limit:
            variables = {
                'poolId': pool_id,
                'startTime': start_ts,
                'endTime': end_ts,
                'skip': skip
            }
            
            data = await self._subgraph_query(query, variables, chain)
            swaps = data.get('swaps', [])
            
            if not swaps:
                break
            
            for swap in swaps:
                token_in = swap.get('tokenIn', {}) or {}
                token_out = swap.get('tokenOut', {}) or {}
                
                in_decimals = int(token_in.get('decimals', 18))
                out_decimals = int(token_out.get('decimals', 18))
                
                amount_in = float(swap.get('amountIn', 0)) / (10 ** in_decimals)
                amount_out = float(swap.get('amountOut', 0)) / (10 ** out_decimals)
                amount_in_usd = float(swap.get('amountInUSD', 0))
                amount_out_usd = float(swap.get('amountOutUSD', 0))
                
                # Calculate exchange rate
                exchange_rate = amount_out / amount_in if amount_in > 0 else 0
                
                # Estimate slippage (deviation from 1:1 for stables)
                slippage = abs(1 - exchange_rate) * 100 if exchange_rate > 0 else 0
                
                all_swaps.append({
                    'timestamp': pd.to_datetime(int(swap['timestamp']), unit='s', utc=True),
                    'tx_hash': swap.get('hash', ''),
                    'pool_id': pool_id,
                    'token_in_symbol': token_in.get('symbol', ''),
                    'token_in_address': token_in.get('id', ''),
                    'token_out_symbol': token_out.get('symbol', ''),
                    'token_out_address': token_out.get('id', ''),
                    'amount_in': amount_in,
                    'amount_in_usd': amount_in_usd,
                    'amount_out': amount_out,
                    'amount_out_usd': amount_out_usd,
                    'exchange_rate': exchange_rate,
                    'slippage_pct': slippage,
                    'chain': chain,
                    'venue': self.VENUE,
                })
            
            skip += len(swaps)
            
            if len(swaps) < 1000:
                break
        
        df = pd.DataFrame(all_swaps[:limit])
        
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
            df['pair'] = df['token_in_symbol'] + '/' + df['token_out_symbol']
        
        return df
    
    # =========================================================================
    # Virtual Price Methods
    # =========================================================================

    async def _process_single_virtual_price(
        self, pool: Dict, chain: str, pool_filter: Optional[List[str]]
    ) -> Optional[Dict]:
        """Process a single pool's virtual price (helper for parallelization)."""
        try:
            pool_address = pool.get('address', '')

            if pool_filter and pool_address.lower() not in [p.lower() for p in pool_filter]:
                return None

            virtual_price_raw = pool.get('virtualPrice', 0)
            virtual_price = float(virtual_price_raw) / 1e18 if virtual_price_raw else 1.0

            # Classify peg status
            deviation = abs(virtual_price - 1.0) * 100
            peg_status = self._classify_peg_status(deviation)

            return {
                'timestamp': datetime.utcnow(),
                'pool_address': pool_address,
                'pool_name': pool.get('name', ''),
                'pool_symbol': pool.get('symbol', ''),
                'virtual_price': virtual_price,
                'virtual_price_raw': virtual_price_raw,
                'tvl_usd': float(pool.get('usdTotal', 0)),
                'volume_24h': float(pool.get('volumeUSD', 0) or 0),
                'apy': float(pool.get('apy', 0) or 0),
                'gauge_apy': float(pool.get('gaugeRewards', 0) or 0),
                'price_deviation_pct': (virtual_price - 1.0) * 100,
                'peg_status': peg_status,
                'is_anomaly': deviation > 0.5 and virtual_price < 1.0,
                'chain': chain,
                'venue': self.VENUE,
            }
        except Exception as e:
            logger.debug(f"Error processing virtual price: {e}")
            return None

    async def fetch_virtual_prices(
        self,
        chain: str = 'ethereum',
        pool_filter: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch virtual prices for pools (real-time from Curve API).
        
        Virtual price tracks LP token appreciation from fees.
        Deviations indicate potential depeg or exploit scenarios.
        
        Args:
            chain: Target chain
            pool_filter: Optional list of pool addresses to filter
            
        Returns:
            DataFrame with virtual price data
        """
        logger.info(f"Fetching virtual prices from {chain}")
        
        api_url = self.API_URLS.get(chain)
        if not api_url:
            logger.warning(f"No API URL for chain: {chain}")
            return pd.DataFrame()
        
        data = await self._api_request(api_url)
        
        if not data:
            return pd.DataFrame()
        
        pools = data.get('data', {}).get('poolData', [])

        # PARALLELIZED: Process all pool virtual prices concurrently
        tasks = [
            self._process_single_virtual_price(pool, chain, pool_filter)
            for pool in pools
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None results and errors
        records = [r for r in results if isinstance(r, dict)]

        df = pd.DataFrame(records)
        
        if not df.empty:
            df = df.sort_values('tvl_usd', ascending=False).reset_index(drop=True)
        
        return df
    
    # =========================================================================
    # Comprehensive Methods
    # =========================================================================
    
    async def _fetch_single_chain_comprehensive(
        self, chain: str, min_tvl: float, top_n_pools: int
    ) -> Dict[str, List[pd.DataFrame]]:
        """Fetch comprehensive data for a single chain."""
        logger.info(f"Fetching comprehensive data from {chain}")

        results = {
            'pools': [],
            'virtual_prices': [],
            'snapshots': [],
        }

        # Pools
        pools = await self.fetch_pools(chain, min_tvl=min_tvl)
        if not pools.empty:
            results['pools'].append(pools)

            # Fetch snapshots for top pools in parallel
            end_date = datetime.utcnow().strftime('%Y-%m-%d')
            start_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')

            snapshot_tasks = [
                self.fetch_pool_daily_snapshots(pool['pool_id'], chain, start_date, end_date)
                for _, pool in pools.head(top_n_pools).iterrows()
            ]
            snapshot_results = await asyncio.gather(*snapshot_tasks, return_exceptions=True)

            # Add pool names to snapshots
            for idx, (_, pool) in enumerate(pools.head(top_n_pools).iterrows()):
                if idx < len(snapshot_results) and isinstance(snapshot_results[idx], pd.DataFrame):
                    snapshots = snapshot_results[idx]
                    if not snapshots.empty:
                        snapshots['pool_name'] = pool['pool_name']
                        results['snapshots'].append(snapshots)

        # Virtual prices
        virtual_prices = await self.fetch_virtual_prices(chain)
        if not virtual_prices.empty:
            results['virtual_prices'].append(virtual_prices)

        return results

    async def fetch_comprehensive_data(
        self,
        chains: Optional[List[str]] = None,
        min_tvl: float = 500_000,
        top_n_pools: int = 20
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch comprehensive Curve data across chains.

        Args:
            chains: Chains to query (default: ethereum, arbitrum, optimism)
            min_tvl: Minimum TVL filter
            top_n_pools: Number of top pools to fetch detailed data for

        Returns:
            Dict with pools, virtual_prices, and snapshots DataFrames
        """
        if chains is None:
            chains = ['ethereum', 'arbitrum', 'optimism']

        # Parallelize chain fetching
        tasks = [
            self._fetch_single_chain_comprehensive(chain, min_tvl, top_n_pools)
            for chain in chains
        ]
        chain_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        results = {
            'pools': [],
            'virtual_prices': [],
            'snapshots': [],
        }

        for chain_result in chain_results:
            if isinstance(chain_result, dict):
                for key in results.keys():
                    results[key].extend(chain_result.get(key, []))

        return {
            key: pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            for key, dfs in results.items()
        }
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    @staticmethod
    def _classify_pool_type(name: str, tokens: List[str]) -> str:
        """Classify pool type based on name and tokens."""
        name_lower = name.lower()
        
        if 'tricrypto' in name_lower:
            return PoolType.TRICRYPTO.value
        elif 'crypto' in name_lower:
            return PoolType.CRYPTOSWAP.value
        elif 'factory' in name_lower:
            return PoolType.FACTORY.value
        elif any(t in ['aDAI', 'aUSDC', 'cDAI', 'cUSDC'] for t in tokens):
            return PoolType.LENDING.value
        elif '3crv' in [t.lower() for t in tokens] or '3pool' in name_lower:
            return PoolType.META.value
        else:
            return PoolType.STABLESWAP.value
    
    @staticmethod
    def _classify_peg_status(deviation_pct: float) -> str:
        """Classify peg status based on deviation percentage."""
        if deviation_pct < 0.1:
            return PegStatus.STABLE.value
        elif deviation_pct < 0.5:
            return PegStatus.MINOR_DEVIATION.value
        elif deviation_pct < 2.0:
            return PegStatus.MODERATE_DEVIATION.value
        elif deviation_pct < 5.0:
            return PegStatus.SIGNIFICANT_DEPEG.value
        else:
            return PegStatus.SEVERE_DEPEG.value
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.collection_stats.copy()
    
    def reset_collection_stats(self):
        """Reset collection statistics."""
        self.collection_stats = {
            'records_collected': 0,
            'api_calls': 0,
            'subgraph_calls': 0,
            'errors': 0,
        }
    
    # =========================================================================
    # Required Abstract Methods
    # =========================================================================
    
    async def fetch_funding_rates(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Curve doesn't have funding rates - return empty DataFrame."""
        logger.debug("Curve: No funding rates available (spot DEX)")
        return pd.DataFrame()
    
    async def _fetch_single_pool_snapshots(
        self, symbol: str, pool: pd.Series, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch snapshots for a single pool."""
        snapshots = await self.fetch_pool_daily_snapshots(
            pool['pool_id'],
            start_date=start_date,
            end_date=end_date
        )

        if not snapshots.empty:
            snapshots['symbol'] = symbol
            snapshots['pool_name'] = pool['pool_name']
            return snapshots
        return pd.DataFrame()

    async def _process_single_symbol_ohlcv(
        self, symbol: str, pools: pd.DataFrame, start_date: str, end_date: str
    ) -> List[pd.DataFrame]:
        """Process OHLCV data for a single symbol (helper for parallelization)."""
        try:
            matching = pools[pools['tokens'].str.contains(symbol, case=False, na=False)]

            tasks = [
                self._fetch_single_pool_snapshots(symbol, pool, start_date, end_date)
                for _, pool in matching.head(3).iterrows()
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter valid DataFrames
            return [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

        except Exception as e:
            logger.debug(f"Error processing symbol {symbol}: {e}")
            return []

    async def fetch_ohlcv(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch OHLCV-equivalent data from pool snapshots.

        Uses daily pool snapshots to provide volume data.
        """
        pools = await self.fetch_pools()

        if pools.empty:
            return pd.DataFrame()

        # PARALLELIZED: Process all symbols concurrently
        tasks = [
            self._process_single_symbol_ohlcv(symbol, pools, start_date, end_date)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and filter valid DataFrames
        all_data = []
        for result in results:
            if isinstance(result, list):
                all_data.extend(result)

        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df['venue'] = self.VENUE
            df['venue_type'] = self.VENUE_TYPE
            return df

        return pd.DataFrame()
    
    async def close(self) -> None:
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.debug("Curve session closed")

    # =========================================================================
    # Standardized Collection Methods (for dynamic routing in collection_manager)
    # =========================================================================

    async def collect_pool_data(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect pool data for symbols (standardized interface).

        Wraps fetch_pools() to match collection_manager expectations.

        Args:
            symbols: List of token symbols to fetch pools for
            start_date: Start date (unused - pools are current state)
            end_date: End date (unused - pools are current state)
            **kwargs: Additional parameters

        Returns:
            DataFrame with pool data for specified symbols
        """
        try:
            # Get chain from kwargs or default to ethereum
            chain = kwargs.get('chain', 'ethereum')
            min_tvl = kwargs.get('min_tvl', 100_000)
            pool_type = kwargs.get('pool_type', None)

            # Fetch all pools
            all_pools = await self.fetch_pools(
                chain=chain,
                min_tvl=min_tvl,
                pool_type=pool_type
            )

            if all_pools.empty:
                logger.warning(f"Curve: No pools found on {chain}")
                return pd.DataFrame()

            # Filter for requested symbols if specified
            if symbols and symbols != ['BTC']: # BTC is placeholder, ignore it
                # Check if any token in the pool matches any requested symbol
                symbol_filter = all_pools[
                    all_pools['tokens'].str.contains('|'.join(symbols), case=False, na=False)
                ]
                if not symbol_filter.empty:
                    all_pools = symbol_filter

            logger.info(f"Curve: Collected {len(all_pools)} pools from {chain}")
            return all_pools

        except Exception as e:
            logger.error(f"Curve collect_pool_data error: {e}")
            return pd.DataFrame()

    async def _fetch_single_pool_swaps(
        self, pool: pd.Series, start_str: str, end_str: str, chain: str, limit_per_pool: int
    ) -> pd.DataFrame:
        """Fetch swaps for a single pool."""
        pool_id = pool['pool_id']
        swaps = await self.fetch_swaps(
            pool_id=pool_id,
            start_date=start_str,
            end_date=end_str,
            chain=chain,
            limit=limit_per_pool
        )

        if not swaps.empty:
            # Add pool context
            swaps['pool_name'] = pool.get('pool_name', '')
            swaps['pool_type'] = pool.get('pool_type', '')
            return swaps
        return pd.DataFrame()

    async def collect_swaps(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect swap data for symbols (standardized interface).

        Wraps fetch_swaps() to match collection_manager expectations.

        Args:
            symbols: List of token symbols to fetch swaps for
            start_date: Start date (string or datetime)
            end_date: End date (string or datetime)
            **kwargs: Additional parameters

        Returns:
            DataFrame with swap events for specified symbols
        """
        try:
            # Convert dates to strings if needed
            if isinstance(start_date, datetime):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if isinstance(end_date, datetime):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            # Get chain from kwargs
            chain = kwargs.get('chain', 'ethereum')
            limit = kwargs.get('limit', 10000)

            # First fetch pools to find pool IDs for requested symbols
            pools = await self.collect_pool_data(symbols, start_date, end_date, **kwargs)

            if pools.empty:
                logger.warning(f"Curve: No pools found for symbols {symbols}")
                return pd.DataFrame()

            # Collect swaps for top 3 pools by TVL in parallel
            top_pools = pools.nlargest(3, 'tvl_usd')
            limit_per_pool = limit // len(top_pools)

            tasks = [
                self._fetch_single_pool_swaps(pool, start_str, end_str, chain, limit_per_pool)
                for _, pool in top_pools.iterrows()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter valid DataFrames
            all_swaps = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

            if all_swaps:
                result = pd.concat(all_swaps, ignore_index=True)
                logger.info(f"Curve: Collected {len(result)} swaps from {chain}")
                return result

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Curve collect_swaps error: {e}")
            return pd.DataFrame()

    async def collect_liquidity(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect liquidity data - wraps collect_pool_data().

        Liquidity data is part of pool data for Curve.
        Standardized method name for collection manager compatibility.
        """
        try:
            # Just call collect_pool_data which includes liquidity metrics (TVL)
            return await self.collect_pool_data(symbols, start_date, end_date, **kwargs)

        except Exception as e:
            logger.error(f"Curve collect_liquidity error: {e}")
            return pd.DataFrame()

# =============================================================================
# Test Function
# =============================================================================

async def test_curve_collector():
    """Test Curve collector functionality."""
    config = {'rate_limit': 20}
    
    async with CurveCollector(config) as collector:
        print("=" * 60)
        print("Curve Finance Collector Test")
        print("=" * 60)
        
        # Test pools
        print("\n1. Testing pool fetch...")
        pools = await collector.fetch_pools(min_tvl=1_000_000)
        print(f" Pools (>$1M TVL): {len(pools)}")
        if not pools.empty:
            top = pools.iloc[0]
            print(f" Top pool: {top['pool_name']} - ${top['tvl_usd']:,.0f}")
        
        # Test virtual prices
        print("\n2. Testing virtual prices...")
        vp = await collector.fetch_virtual_prices()
        print(f" Pools with virtual price: {len(vp)}")
        if not vp.empty:
            anomalies = vp[vp['is_anomaly']]
            print(f" Anomalies detected: {len(anomalies)}")
        
        # Test pool snapshots
        if not pools.empty:
            print("\n3. Testing daily snapshots...")
            pool_id = pools.iloc[0]['pool_id']
            end = datetime.utcnow().strftime('%Y-%m-%d')
            start = (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            snapshots = await collector.fetch_pool_daily_snapshots(
                pool_id, start_date=start, end_date=end
            )
            print(f" Snapshots: {len(snapshots)}")
            if not snapshots.empty:
                print(f" Avg daily volume: ${snapshots['daily_volume_usd'].mean():,.0f}")
        
        # Collection stats
        print("\n" + "=" * 60)
        stats = collector.get_collection_stats()
        print(f"Collection Stats:")
        print(f" Records: {stats['records_collected']}")
        print(f" API Calls: {stats['api_calls']}")
        print(f" Subgraph Calls: {stats['subgraph_calls']}")
        print(f" Errors: {stats['errors']}")
        print("=" * 60)

if __name__ == '__main__':
    asyncio.run(test_curve_collector())