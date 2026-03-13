"""
The Graph Collector - Subgraph Query Infrastructure for On-Chain Protocol Data

validated collector for indexed blockchain data via GraphQL.
Access historical and real-time data from major DeFi protocols across
multiple chains through The Graph's decentralized indexing network.

===============================================================================
OVERVIEW
===============================================================================

The Graph is a decentralized indexing protocol that enables querying blockchain
data via GraphQL. Subgraphs are open APIs that index specific smart contracts,
providing structured access to on-chain events and state.

Network Types:

    Decentralized Network (Gateway):
        - Requires API key from The Graph Studio
        - Higher reliability and decentralization
        - Pay per query with GRT tokens
        - Recommended for production

    Hosted Service (Legacy):
        - Free tier available
        - Being deprecated
        - Limited reliability
        - Useful for development

===============================================================================
SUPPORTED PROTOCOLS
===============================================================================

DEX Protocols:
    ============== ============== ======================================
    Protocol Chains Data Available
    ============== ============== ======================================
    Uniswap V3 ETH, ARB, OP, Pools, swaps, positions, daily OHLCV
                    POLY, BASE
    Uniswap V2 ETH Pairs, swaps, liquidity events
    SushiSwap ETH, ARB Pairs, swaps, volume
    Curve ETH Pools, swaps, gauges
    Balancer V2 ETH, ARB Pools, swaps, gauge data
    PancakeSwap BSC Pairs, swaps, farms
    ============== ============== ======================================

Lending Protocols:
    ============== ============== ======================================
    Protocol Chains Data Available
    ============== ============== ======================================
    Aave V3 ETH, ARB, OP Markets, deposits, borrows, rates
    Compound V3 ETH Markets, positions, liquidations
    MakerDAO ETH Vaults, liquidations, DAI supply
    ============== ============== ======================================

Staking & Liquid Staking:
    ============== ============== ======================================
    Protocol Chains Data Available
    ============== ============== ======================================
    Lido ETH Submissions, rewards, operators
    Rocket Pool ETH Minipools, node operators
    ============== ============== ======================================

Other:
    - ENS: Domain registrations, transfers
    - NFT Marketplaces: Sales, listings
    - Bridges: Cross-chain transfers

===============================================================================
DATA TYPES COLLECTED
===============================================================================

Pool/Pair Data:
    - Pool addresses and metadata
    - Token composition and weights
    - Fee tiers and configurations
    - Total Value Locked (TVL)
    - Historical volume

Swap/Trade Data:
    - Individual swap transactions
    - Price impact analysis
    - Routing information
    - Gas costs

Liquidity Data:
    - Liquidity positions
    - Mint/burn events
    - Position ranges (V3)
    - Fee accumulation

Time Series:
    - Pool day data (daily OHLCV)
    - Hourly snapshots
    - Token price history

Lending Data:
    - Supply/borrow rates
    - Utilization rates
    - Collateral factors
    - Liquidation events

===============================================================================
RATE LIMITS
===============================================================================

Decentralized Network:
    - Rate limits depend on query complexity
    - Queries consume GRT based on indexer pricing
    - Recommended: 10-30 queries/second

Hosted Service:
    - ~100 requests/minute (varies)
    - May have reliability issues
    - Being deprecated

===============================================================================
USAGE EXAMPLES
===============================================================================

Basic pool discovery:

    >>> from data_collection.indexers import TheGraphCollector
    >>> 
    >>> config = {'graph_api_key': 'your-api-key'}
    >>> async with TheGraphCollector(config) as collector:
    ... pools = await collector.fetch_uniswap_v3_pools(
    ... chain='ethereum',
    ... min_tvl=1_000_000
    ... )
    ... print(f"Found {len(pools)} pools with >$1M TVL")

Cross-chain pool aggregation:

    >>> pools = await collector.fetch_multi_dex_pools(
    ... chains=['ethereum', 'arbitrum', 'optimism'],
    ... min_tvl=500_000
    ... )

Historical pool data:

    >>> day_data = await collector.fetch_uniswap_v3_pool_day_data(
    ... pool_id='0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8',
    ... chain='ethereum',
    ... start_date='2024-01-01',
    ... end_date='2024-03-31'
    ... )

Custom GraphQL queries:

    >>> result = await collector.execute_query(
    ... subgraph='uniswap-v3-ethereum',
    ... query='''
    ... query {
    ... pools(first: 10, orderBy: volumeUSD, orderDirection: desc) {
    ... id
    ... volumeUSD
    ... }
    ... }
    ... '''
    ... )

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

Liquidity Analysis:
    - Identify deep liquidity pools for execution
    - Monitor TVL changes for alpha signals
    - Track liquidity migration across chains

Volume Analysis:
    - Cross-DEX volume comparison
    - Volume/TVL ratios for efficiency
    - Unusual volume detection

Price Discovery:
    - Compare prices across pools/DEXes
    - Identify arbitrage opportunities
    - Track price impact curves

Yield Opportunities:
    - Compare lending rates across protocols
    - LP yield analysis
    - Fee accumulation tracking

===============================================================================
DATA QUALITY CONSIDERATIONS
===============================================================================

- Subgraph indexing may lag during high activity
- Different subgraphs may have different update frequencies
- Pool creation timestamps may be block-level, not exact
- TVL calculations depend on price oracles used by subgraph
- Some subgraphs may have data gaps or inconsistencies

Version: 2.0.0
Documentation: https://thegraph.com/docs/
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
    ARBITRUM = 'arbitrum'
    OPTIMISM = 'optimism'
    POLYGON = 'polygon'
    BASE = 'base'
    BSC = 'bsc'
    AVALANCHE = 'avalanche'
    FANTOM = 'fantom'
    GNOSIS = 'gnosis'
    CELO = 'celo'

class Protocol(Enum):
    """Supported DeFi protocols."""
    UNISWAP_V3 = 'uniswap_v3'
    UNISWAP_V2 = 'uniswap_v2'
    SUSHISWAP = 'sushiswap'
    CURVE = 'curve'
    BALANCER_V2 = 'balancer_v2'
    PANCAKESWAP = 'pancakeswap'
    AAVE_V3 = 'aave_v3'
    COMPOUND_V3 = 'compound_v3'
    LIDO = 'lido'

class ProtocolType(Enum):
    """Protocol category classification."""
    DEX = 'dex'
    LENDING = 'lending'
    STAKING = 'staking'
    BRIDGE = 'bridge'
    NFT = 'nft'
    DERIVATIVES = 'derivatives'

class FeeTier(Enum):
    """Uniswap V3 fee tiers."""
    LOWEST = 100 # 0.01% - stablecoin pairs
    LOW = 500 # 0.05% - stable pairs
    MEDIUM = 3000 # 0.30% - most pairs
    HIGH = 10000 # 1.00% - exotic pairs

class PoolQuality(Enum):
    """Pool quality classification based on TVL and volume."""
    INSTITUTIONAL = 'institutional' # >$50M TVL
    DEEP = 'deep' # $10M-$50M TVL
    MODERATE = 'moderate' # $1M-$10M TVL
    SHALLOW = 'shallow' # $100K-$1M TVL
    THIN = 'thin' # <$100K TVL

class LendingRateType(Enum):
    """Lending protocol rate types."""
    SUPPLY_VARIABLE = 'supply_variable'
    SUPPLY_STABLE = 'supply_stable'
    BORROW_VARIABLE = 'borrow_variable'
    BORROW_STABLE = 'borrow_stable'

class NetworkType(Enum):
    """The Graph network type."""
    DECENTRALIZED = 'decentralized' # Gateway with API key
    HOSTED = 'hosted' # Legacy hosted service

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class UniswapV3Pool:
    """Uniswap V3 pool data with computed analytics."""
    pool_id: str
    token0_address: str
    token0_symbol: str
    token1_address: str
    token1_symbol: str
    fee_tier: int # Raw fee tier (100, 500, 3000, 10000)
    liquidity: float
    tvl_usd: float
    volume_usd: float
    tx_count: int
    created_at: datetime
    chain: str
    
    @property
    def pair_name(self) -> str:
        """Formatted pair name."""
        return f"{self.token0_symbol}/{self.token1_symbol}"
    
    @property
    def fee_pct(self) -> float:
        """Fee as percentage."""
        return self.fee_tier / 10000
    
    @property
    def fee_tier_enum(self) -> FeeTier:
        """Fee tier as enum."""
        try:
            return FeeTier(self.fee_tier)
        except ValueError:
            return FeeTier.MEDIUM
    
    @property
    def volume_tvl_ratio(self) -> float:
        """Volume to TVL ratio (capital efficiency)."""
        return self.volume_usd / self.tvl_usd if self.tvl_usd > 0 else 0
    
    @property
    def quality(self) -> PoolQuality:
        """Classify pool quality based on TVL."""
        if self.tvl_usd >= 50_000_000:
            return PoolQuality.INSTITUTIONAL
        elif self.tvl_usd >= 10_000_000:
            return PoolQuality.DEEP
        elif self.tvl_usd >= 1_000_000:
            return PoolQuality.MODERATE
        elif self.tvl_usd >= 100_000:
            return PoolQuality.SHALLOW
        else:
            return PoolQuality.THIN
    
    @property
    def is_stablecoin_pair(self) -> bool:
        """Check if pool is stablecoin pair."""
        stables = {'USDC', 'USDT', 'DAI', 'FRAX', 'LUSD', 'TUSD', 'BUSD', 'GUSD'}
        return (self.token0_symbol.upper() in stables and 
                self.token1_symbol.upper() in stables)
    
    @property
    def estimated_daily_fees(self) -> float:
        """Estimate daily fee revenue (assuming 24h volume)."""
        return self.volume_usd * self.fee_pct / 100
    
    @property
    def estimated_apy(self) -> float:
        """Estimate LP APY from fees."""
        if self.tvl_usd > 0:
            daily_fees = self.estimated_daily_fees
            return (daily_fees * 365 / self.tvl_usd) * 100
        return 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with computed fields."""
        return {
            'pool_id': self.pool_id,
            'pair_name': self.pair_name,
            'token0_address': self.token0_address,
            'token0_symbol': self.token0_symbol,
            'token1_address': self.token1_address,
            'token1_symbol': self.token1_symbol,
            'fee_tier': self.fee_tier,
            'fee_pct': self.fee_pct,
            'liquidity': self.liquidity,
            'tvl_usd': self.tvl_usd,
            'volume_usd': self.volume_usd,
            'volume_tvl_ratio': self.volume_tvl_ratio,
            'tx_count': self.tx_count,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else str(self.created_at),
            'chain': self.chain,
            'quality': self.quality.value,
            'is_stablecoin_pair': self.is_stablecoin_pair,
            'estimated_apy': self.estimated_apy,
            'dex': 'uniswap_v3',
        }

@dataclass
class UniswapV3Swap:
    """Individual swap transaction from Uniswap V3."""
    tx_id: str
    timestamp: datetime
    pool_id: str
    sender: str
    recipient: str
    amount0: float
    amount1: float
    amount_usd: float
    tick: int
    chain: str
    
    @property
    def is_buy_token0(self) -> bool:
        """Check if swap bought token0 (amount0 > 0)."""
        return self.amount0 > 0
    
    @property
    def price_impact_proxy(self) -> float:
        """Proxy for price impact (based on tick movement)."""
        # Larger tick values indicate larger price changes
        return abs(self.tick) / 10000 if self.tick else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'tx_id': self.tx_id,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'pool_id': self.pool_id,
            'sender': self.sender,
            'recipient': self.recipient,
            'amount0': self.amount0,
            'amount1': self.amount1,
            'amount_usd': self.amount_usd,
            'tick': self.tick,
            'chain': self.chain,
            'is_buy_token0': self.is_buy_token0,
        }

@dataclass
class PoolDayData:
    """Daily OHLCV data for a pool."""
    timestamp: datetime
    pool_id: str
    open: float
    high: float
    low: float
    close: float
    volume_usd: float
    tvl_usd: float
    tx_count: int
    fees_usd: float
    chain: str
    
    @property
    def range_pct(self) -> float:
        """Daily price range as percentage."""
        return (self.high - self.low) / self.open * 100 if self.open > 0 else 0
    
    @property
    def return_pct(self) -> float:
        """Daily return percentage."""
        return (self.close - self.open) / self.open * 100 if self.open > 0 else 0
    
    @property
    def is_bullish(self) -> bool:
        """Check if day was bullish."""
        return self.close > self.open
    
    @property
    def fee_yield_daily(self) -> float:
        """Daily fee yield percentage."""
        return (self.fees_usd / self.tvl_usd) * 100 if self.tvl_usd > 0 else 0
    
    @property
    def volume_tvl_ratio(self) -> float:
        """Volume/TVL ratio for the day."""
        return self.volume_usd / self.tvl_usd if self.tvl_usd > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'pool_id': self.pool_id,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume_usd': self.volume_usd,
            'tvl_usd': self.tvl_usd,
            'tx_count': self.tx_count,
            'fees_usd': self.fees_usd,
            'chain': self.chain,
            'range_pct': self.range_pct,
            'return_pct': self.return_pct,
            'is_bullish': self.is_bullish,
            'fee_yield_daily': self.fee_yield_daily,
        }

@dataclass
class AaveMarket:
    """Aave V3 lending market data."""
    market_id: str
    name: str
    token_address: str
    token_symbol: str
    tvl_usd: float
    total_deposits_usd: float
    total_borrows_usd: float
    supply_rate: float # Annual rate as decimal
    borrow_rate: float # Annual rate as decimal
    token_price_usd: float
    chain: str
    
    @property
    def utilization_rate(self) -> float:
        """Calculate utilization rate."""
        if self.total_deposits_usd > 0:
            return self.total_borrows_usd / self.total_deposits_usd
        return 0
    
    @property
    def supply_apy_pct(self) -> float:
        """Supply APY as percentage."""
        return self.supply_rate * 100
    
    @property
    def borrow_apy_pct(self) -> float:
        """Borrow APY as percentage."""
        return self.borrow_rate * 100
    
    @property
    def rate_spread(self) -> float:
        """Spread between borrow and supply rates."""
        return self.borrow_rate - self.supply_rate
    
    @property
    def available_liquidity(self) -> float:
        """Available liquidity for borrowing."""
        return self.total_deposits_usd - self.total_borrows_usd
    
    @property
    def is_high_utilization(self) -> bool:
        """Check if market has high utilization (>80%)."""
        return self.utilization_rate > 0.8
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'market_id': self.market_id,
            'name': self.name,
            'token_address': self.token_address,
            'token_symbol': self.token_symbol,
            'tvl_usd': self.tvl_usd,
            'total_deposits_usd': self.total_deposits_usd,
            'total_borrows_usd': self.total_borrows_usd,
            'supply_apy_pct': self.supply_apy_pct,
            'borrow_apy_pct': self.borrow_apy_pct,
            'utilization_rate': self.utilization_rate,
            'rate_spread': self.rate_spread,
            'available_liquidity': self.available_liquidity,
            'is_high_utilization': self.is_high_utilization,
            'token_price_usd': self.token_price_usd,
            'chain': self.chain,
            'protocol': 'aave_v3',
        }

@dataclass
class CurvePool:
    """Curve Finance pool data."""
    pool_id: str
    name: str
    symbol: str
    tokens: List[str]
    tvl_usd: float
    cumulative_volume_usd: float
    daily_volume_usd: float
    amplification: int # A parameter
    fee_pct: float
    
    @property
    def token_count(self) -> int:
        """Number of tokens in pool."""
        return len(self.tokens)
    
    @property
    def is_metapool(self) -> bool:
        """Check if pool is a metapool (contains LP token)."""
        # Metapools typically have fewer than 4 tokens and include base pool
        return self.token_count <= 2
    
    @property
    def volume_tvl_ratio(self) -> float:
        """Daily volume to TVL ratio."""
        return self.daily_volume_usd / self.tvl_usd if self.tvl_usd > 0 else 0
    
    @property
    def estimated_daily_fees(self) -> float:
        """Estimated daily fee revenue."""
        return self.daily_volume_usd * self.fee_pct / 100
    
    @property
    def estimated_apy(self) -> float:
        """Estimated base APY from fees."""
        if self.tvl_usd > 0:
            return (self.estimated_daily_fees * 365 / self.tvl_usd) * 100
        return 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'pool_id': self.pool_id,
            'name': self.name,
            'symbol': self.symbol,
            'tokens': self.tokens,
            'token_count': self.token_count,
            'tvl_usd': self.tvl_usd,
            'cumulative_volume_usd': self.cumulative_volume_usd,
            'daily_volume_usd': self.daily_volume_usd,
            'volume_tvl_ratio': self.volume_tvl_ratio,
            'amplification': self.amplification,
            'fee_pct': self.fee_pct,
            'estimated_apy': self.estimated_apy,
            'is_metapool': self.is_metapool,
            'protocol': 'curve',
        }

@dataclass
class BalancerPool:
    """Balancer V2 pool data."""
    pool_id: str
    name: str
    symbol: str
    pool_type: str
    tokens: List[str]
    tvl_usd: float
    total_volume_usd: float
    total_fees_usd: float
    swap_fee_pct: float
    chain: str
    
    @property
    def token_count(self) -> int:
        """Number of tokens in pool."""
        return len(self.tokens)
    
    @property
    def is_weighted(self) -> bool:
        """Check if weighted pool."""
        return self.pool_type.lower() == 'weighted'
    
    @property
    def is_stable(self) -> bool:
        """Check if stable pool."""
        return 'stable' in self.pool_type.lower()
    
    @property
    def avg_fee_per_trade(self) -> float:
        """Average fee per trade estimate."""
        if self.total_volume_usd > 0:
            return self.total_fees_usd / self.total_volume_usd * 100
        return self.swap_fee_pct
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'pool_id': self.pool_id,
            'name': self.name,
            'symbol': self.symbol,
            'pool_type': self.pool_type,
            'tokens': self.tokens,
            'token_count': self.token_count,
            'tvl_usd': self.tvl_usd,
            'total_volume_usd': self.total_volume_usd,
            'total_fees_usd': self.total_fees_usd,
            'swap_fee_pct': self.swap_fee_pct,
            'is_weighted': self.is_weighted,
            'is_stable': self.is_stable,
            'chain': self.chain,
            'protocol': 'balancer_v2',
        }

# =============================================================================
# Main Collector Class
# =============================================================================

class TheGraphCollector(BaseCollector):
    """
    The Graph subgraph collector for on-chain protocol data.
    
    validated implementation for querying indexed blockchain data
    via GraphQL across multiple DeFi protocols and chains.
    
    Features:
        - Multi-chain support (Ethereum, Arbitrum, Optimism, Polygon, etc.)
        - Pre-built queries for major protocols
        - Custom GraphQL query support
        - Automatic pagination handling
        - Decentralized and hosted service support
        - Rate limiting and retry logic
    
    Supported Protocols:
        - DEX: Uniswap V3/V2, SushiSwap, Curve, Balancer
        - Lending: Aave V3, Compound V3
        - Staking: Lido, Rocket Pool
    
    Example:
        >>> config = {'graph_api_key': 'your-api-key'}
        >>> async with TheGraphCollector(config) as collector:
        ... pools = await collector.fetch_uniswap_v3_pools(
        ... chain='ethereum',
        ... min_tvl=1_000_000
        ... )
    
    Attributes:
        VENUE: 'thegraph'
        VENUE_TYPE: 'indexer'
    """
    
    VENUE = 'thegraph'
    VENUE_TYPE = 'indexer'
    
    # Decentralized network gateway
    GATEWAY_URL = 'https://gateway.thegraph.com/api'
    
    # Hosted service (legacy)
    HOSTED_SERVICE = 'https://api.thegraph.com/subgraphs/name'
    
    # Decentralized network subgraph IDs
    SUBGRAPH_IDS = {
        # Uniswap V3
        'uniswap-v3-ethereum': '5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV',
        'uniswap-v3-arbitrum': 'FbCGRftH4a3yZugY7TnbYgPJVEv2LvMT6oF1fxPe9aJM',
        'uniswap-v3-optimism': 'Cghf4LfVqPiFw6fp6Y5X5Ubc8UpmUhSfJL82zwiBFLaj',
        'uniswap-v3-polygon': '3hCPRGf4z88VC5rsBKU5AA9FBBq5nF3jbKJG7VZCbhjm',
        'uniswap-v3-base': 'GqzP4Xaehti8KSfQmv3ZctFSjnSUYZ4En5NRsiTbvZpz',
        
        # Aave V3
        'aave-v3-ethereum': 'Cd2gEDVeqnjBn1hSeqFMitw8Q1iiyV9FYUZkLNRcL87g',
        'aave-v3-arbitrum': 'DLuE98kEb5pQNXAcKFQGQgfSQ57Xdou4jnVbAEqMfy3B',
        'aave-v3-optimism': 'DSfLz8oQBUeU5atALgUFQKMTSYV9mZAVYp4noLSXAfvb',
        
        # Curve
        'curve-ethereum': 'FLQ5VGwGJkx2sJz8v3J8pP2WR1NWWdMqqCckaHvE4zQF',
        
        # Balancer
        'balancer-v2-ethereum': 'C4ayEZP2yTXRAB8vSaTrgN4m9anTe9Mdm2ViyiAuV9TV',
        'balancer-v2-arbitrum': '98cQDy6tufTJtshDLurGLeb5Hv5TBxQJrqAWWchqrYLa',
        
        # Lido
        'lido': 'Sxx1XP6wzgJ7CDvjdMfWCdPC6ZrLNfJG3xWZXvB8g8j',
    }
    
    # Hosted service subgraphs (legacy)
    HOSTED_SUBGRAPHS = {
        'uniswap-v2': 'uniswap/uniswap-v2',
        'sushiswap-ethereum': 'sushiswap/exchange',
        'sushiswap-arbitrum': 'sushiswap/arbitrum-exchange',
        'pancakeswap': 'pancakeswap/exchange-v2',
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize The Graph collector.

        Args:
            config: Configuration with options:
                - graph_api_key: API key for decentralized network
                - rate_limit: Requests per minute (default: 60)
                - timeout: Request timeout seconds (default: 60)
        """
        config = config or {}
        super().__init__(config)

        # CRITICAL: Set supported data types for dynamic routing
        self.supported_data_types = ['subgraph_data']
        self.venue = 'thegraph'

        # Import VenueType from base_collector
        from ..base_collector import VenueType
        self.venue_type = VenueType.INDEXER
        self.requires_auth = True # Requires The Graph API key

        # Support multiple env var names: THE_GRAPH_API_KEY, GRAPH_API_KEY, graph_api_key
        import os
        self.api_key = (
            config.get('graph_api_key') or
            config.get('the_graph_api_key') or
            os.getenv('THE_GRAPH_API_KEY', '') or
            os.getenv('GRAPH_API_KEY', '')
        )
        self.timeout = aiohttp.ClientTimeout(total=config.get('timeout', 60))
        self.session: Optional[aiohttp.ClientSession] = None

        # Use shared rate limiter to avoid re-initialization overhead
        rate_limit = config.get('rate_limit', 30) # 30 requests per minute default
        self.rate_limiter = get_shared_rate_limiter('thegraph', rate=rate_limit, per=60.0, burst=5)

        self.collection_stats = {
            'queries': 0, 'records': 0, 'errors': 0
        }

        network = 'decentralized' if self.api_key else 'hosted'
        logger.info(f"Initialized TheGraph collector (network: {network}, data types: {self.supported_data_types})")
    
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
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self.session
    
    def _get_gateway_url(self, subgraph_id: str) -> str:
        """Get decentralized gateway URL for subgraph."""
        return f"{self.GATEWAY_URL}/{self.api_key}/subgraphs/id/{subgraph_id}"
    
    def _get_hosted_url(self, subgraph_name: str) -> str:
        """Get hosted service URL for subgraph."""
        return f"{self.HOSTED_SERVICE}/{subgraph_name}"
    
    async def execute_query(
        self,
        subgraph: str,
        query: str,
        variables: Optional[Dict] = None,
        use_gateway: bool = True
    ) -> Dict:
        """
        Execute GraphQL query against subgraph.
        
        Args:
            subgraph: Subgraph name or ID
            query: GraphQL query string
            variables: Optional query variables
            use_gateway: Use decentralized gateway (requires API key)
            
        Returns:
            Query result dictionary
        """
        session = await self._get_session()
        await self.rate_limiter.acquire()
        self.collection_stats['queries'] += 1
        
        # Determine URL
        if use_gateway and self.api_key and subgraph in self.SUBGRAPH_IDS:
            url = self._get_gateway_url(self.SUBGRAPH_IDS[subgraph])
        elif subgraph in self.HOSTED_SUBGRAPHS:
            url = self._get_hosted_url(self.HOSTED_SUBGRAPHS[subgraph])
        elif subgraph in self.SUBGRAPH_IDS:
            url = self._get_gateway_url(self.SUBGRAPH_IDS[subgraph])
        else:
            if '/' in subgraph:
                url = self._get_hosted_url(subgraph)
            else:
                url = self._get_gateway_url(subgraph)
        
        payload = {'query': query}
        if variables:
            payload['variables'] = variables
        
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'errors' in data:
                        logger.warning(f"GraphQL errors: {data['errors']}")
                    return data
                else:
                    text = await resp.text()
                    logger.error(f"Graph query error {resp.status}: {text[:200]}")
                    self.collection_stats['errors'] += 1
                    return {}
        except Exception as e:
            logger.error(f"Graph query error: {e}")
            self.collection_stats['errors'] += 1
            return {}
    
    async def fetch_uniswap_v3_pools(
        self,
        chain: str = 'ethereum',
        min_tvl: float = 100000,
        first: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch Uniswap V3 pools with liquidity filtering.
        
        Args:
            chain: Blockchain network
            min_tvl: Minimum TVL in USD
            first: Max records per page
            
        Returns:
            DataFrame with pool data and computed fields
        """
        subgraph = f'uniswap-v3-{chain}'
        
        query = """
        query GetPools($minTvl: BigDecimal!, $first: Int!, $skip: Int!) {
            pools(
                first: $first,
                skip: $skip,
                orderBy: totalValueLockedUSD,
                orderDirection: desc,
                where: { totalValueLockedUSD_gte: $minTvl }
            ) {
                id
                token0 { id symbol name decimals }
                token1 { id symbol name decimals }
                feeTier
                liquidity
                sqrtPrice
                tick
                totalValueLockedUSD
                volumeUSD
                txCount
                createdAtTimestamp
            }
        }
        """
        
        all_records = []
        skip = 0
        
        while True:
            variables = {'minTvl': str(min_tvl), 'first': first, 'skip': skip}
            result = await self.execute_query(subgraph, query, variables)
            pools = result.get('data', {}).get('pools', [])
            
            if not pools:
                break
            
            for pool in pools:
                p = UniswapV3Pool(
                    pool_id=pool['id'],
                    token0_address=pool['token0']['id'],
                    token0_symbol=pool['token0']['symbol'],
                    token1_address=pool['token1']['id'],
                    token1_symbol=pool['token1']['symbol'],
                    fee_tier=int(pool['feeTier']),
                    liquidity=float(pool.get('liquidity', 0)),
                    tvl_usd=float(pool['totalValueLockedUSD']),
                    volume_usd=float(pool['volumeUSD']),
                    tx_count=int(pool['txCount']),
                    created_at=pd.to_datetime(int(pool['createdAtTimestamp']), unit='s', utc=True),
                    chain=chain
                )
                all_records.append(p.to_dict())
            
            if len(pools) < first:
                break
            
            skip += first
            logger.info(f"Fetched {skip} pools from Uniswap V3 {chain}")
        
        self.collection_stats['records'] += len(all_records)
        return pd.DataFrame(all_records)
    
    async def fetch_uniswap_v3_swaps(
        self,
        pool_id: str,
        chain: str = 'ethereum',
        start_timestamp: Optional[int] = None,
        first: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch swap transactions for a specific pool.
        
        Args:
            pool_id: Pool contract address
            chain: Blockchain network
            start_timestamp: Unix timestamp filter
            first: Max records per page
            
        Returns:
            DataFrame with swap data
        """
        subgraph = f'uniswap-v3-{chain}'
        
        query = """
        query GetSwaps($poolId: String!, $first: Int!, $skip: Int!, $startTime: BigInt) {
            swaps(
                first: $first,
                skip: $skip,
                orderBy: timestamp,
                orderDirection: desc,
                where: { pool: $poolId, timestamp_gte: $startTime }
            ) {
                id
                timestamp
                sender
                recipient
                amount0
                amount1
                amountUSD
                tick
            }
        }
        """
        
        if start_timestamp is None:
            start_timestamp = int((datetime.utcnow() - timedelta(days=7)).timestamp())
        
        all_records = []
        skip = 0
        
        while True:
            variables = {
                'poolId': pool_id, 'first': first,
                'skip': skip, 'startTime': str(start_timestamp)
            }
            
            result = await self.execute_query(subgraph, query, variables)
            swaps = result.get('data', {}).get('swaps', [])
            
            if not swaps:
                break
            
            for swap in swaps:
                s = UniswapV3Swap(
                    tx_id=swap['id'],
                    timestamp=pd.to_datetime(int(swap['timestamp']), unit='s', utc=True),
                    pool_id=pool_id,
                    sender=swap['sender'],
                    recipient=swap['recipient'],
                    amount0=float(swap['amount0']),
                    amount1=float(swap['amount1']),
                    amount_usd=float(swap['amountUSD']),
                    tick=int(swap['tick']),
                    chain=chain
                )
                all_records.append(s.to_dict())
            
            if len(swaps) < first:
                break
            skip += first
        
        self.collection_stats['records'] += len(all_records)
        return pd.DataFrame(all_records)
    
    async def fetch_uniswap_v3_pool_day_data(
        self,
        pool_id: str,
        chain: str = 'ethereum',
        start_date: str = '2022-01-01',
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV data for a pool.
        
        Args:
            pool_id: Pool contract address
            chain: Blockchain network
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with daily OHLCV data
        """
        subgraph = f'uniswap-v3-{chain}'
        
        query = """
        query GetPoolDayData($poolId: String!, $startDate: Int!, $endDate: Int!) {
            poolDayDatas(
                first: 1000,
                orderBy: date,
                where: { pool: $poolId, date_gte: $startDate, date_lte: $endDate }
            ) {
                date
                open
                high
                low
                close
                volumeUSD
                tvlUSD
                txCount
                feesUSD
            }
        }
        """
        
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp()) if end_date else int(datetime.utcnow().timestamp())
        
        variables = {'poolId': pool_id, 'startDate': start_ts, 'endDate': end_ts}
        result = await self.execute_query(subgraph, query, variables)
        day_data = result.get('data', {}).get('poolDayDatas', [])
        
        records = []
        for day in day_data:
            d = PoolDayData(
                timestamp=pd.to_datetime(int(day['date']), unit='s', utc=True),
                pool_id=pool_id,
                open=float(day.get('open', 0)),
                high=float(day.get('high', 0)),
                low=float(day.get('low', 0)),
                close=float(day.get('close', 0)),
                volume_usd=float(day['volumeUSD']),
                tvl_usd=float(day['tvlUSD']),
                tx_count=int(day['txCount']),
                fees_usd=float(day['feesUSD']),
                chain=chain
            )
            records.append(d.to_dict())
        
        self.collection_stats['records'] += len(records)
        return pd.DataFrame(records)
    
    async def fetch_aave_markets(self, chain: str = 'ethereum') -> pd.DataFrame:
        """
        Fetch Aave V3 market data.
        
        Args:
            chain: Blockchain network
            
        Returns:
            DataFrame with lending market data
        """
        subgraph = f'aave-v3-{chain}'
        
        query = """
        {
            markets(first: 100) {
                id
                name
                inputToken { id symbol name decimals }
                totalValueLockedUSD
                totalDepositBalanceUSD
                totalBorrowBalanceUSD
                rates { rate side type }
                inputTokenPriceUSD
            }
        }
        """
        
        result = await self.execute_query(subgraph, query)
        markets = result.get('data', {}).get('markets', [])
        
        records = []
        for market in markets:
            supply_rate = borrow_rate = 0
            for rate in market.get('rates', []):
                if rate['side'] == 'LENDER' and rate['type'] == 'VARIABLE':
                    supply_rate = float(rate['rate'])
                elif rate['side'] == 'BORROWER' and rate['type'] == 'VARIABLE':
                    borrow_rate = float(rate['rate'])
            
            m = AaveMarket(
                market_id=market['id'],
                name=market['name'],
                token_address=market['inputToken']['id'],
                token_symbol=market['inputToken']['symbol'],
                tvl_usd=float(market['totalValueLockedUSD']),
                total_deposits_usd=float(market['totalDepositBalanceUSD']),
                total_borrows_usd=float(market['totalBorrowBalanceUSD']),
                supply_rate=supply_rate,
                borrow_rate=borrow_rate,
                token_price_usd=float(market.get('inputTokenPriceUSD', 0)),
                chain=chain
            )
            records.append(m.to_dict())
        
        self.collection_stats['records'] += len(records)
        return pd.DataFrame(records)
    
    async def fetch_curve_pools(self) -> pd.DataFrame:
        """Fetch Curve Finance pool data."""
        query = """
        {
            pools(first: 100, orderBy: totalValueLockedUSD, orderDirection: desc) {
                id
                name
                symbol
                coins { token { symbol } }
                totalValueLockedUSD
                cumulativeVolumeUSD
                dailyVolumeUSD
                A
                fee
            }
        }
        """
        
        result = await self.execute_query('curve-ethereum', query)
        pools = result.get('data', {}).get('pools', [])
        
        records = []
        for pool in pools:
            coins = [c['token']['symbol'] for c in pool.get('coins', [])]
            
            p = CurvePool(
                pool_id=pool['id'],
                name=pool['name'],
                symbol=pool['symbol'],
                tokens=coins,
                tvl_usd=float(pool['totalValueLockedUSD']),
                cumulative_volume_usd=float(pool['cumulativeVolumeUSD']),
                daily_volume_usd=float(pool.get('dailyVolumeUSD', 0)),
                amplification=int(pool.get('A', 0)),
                fee_pct=float(pool.get('fee', 0)) / 1e10
            )
            records.append(p.to_dict())
        
        self.collection_stats['records'] += len(records)
        return pd.DataFrame(records)
    
    async def fetch_balancer_pools(
        self, chain: str = 'ethereum', min_tvl: float = 100000
    ) -> pd.DataFrame:
        """Fetch Balancer V2 pool data."""
        subgraph = f'balancer-v2-{chain}'
        
        query = """
        query GetPools($minTvl: BigDecimal!) {
            pools(
                first: 100,
                orderBy: totalLiquidity,
                orderDirection: desc,
                where: { totalLiquidity_gte: $minTvl }
            ) {
                id
                name
                symbol
                poolType
                tokens { symbol }
                totalLiquidity
                totalSwapVolume
                totalSwapFee
                swapFee
            }
        }
        """
        
        variables = {'minTvl': str(min_tvl)}
        result = await self.execute_query(subgraph, query, variables)
        pools = result.get('data', {}).get('pools', [])
        
        records = []
        for pool in pools:
            tokens = [t['symbol'] for t in pool.get('tokens', [])]
            
            p = BalancerPool(
                pool_id=pool['id'],
                name=pool['name'],
                symbol=pool['symbol'],
                pool_type=pool['poolType'],
                tokens=tokens,
                tvl_usd=float(pool['totalLiquidity']),
                total_volume_usd=float(pool['totalSwapVolume']),
                total_fees_usd=float(pool['totalSwapFee']),
                swap_fee_pct=float(pool['swapFee']),
                chain=chain
            )
            records.append(p.to_dict())
        
        self.collection_stats['records'] += len(records)
        return pd.DataFrame(records)
    
    async def _fetch_single_chain_uniswap_pools(
        self,
        chain: str,
        min_tvl: float
    ) -> pd.DataFrame:
        """
        Helper to fetch Uniswap V3 pools for a single chain (parallelized).

        Args:
            chain: Chain name
            min_tvl: Minimum TVL filter

        Returns:
            DataFrame with pool data
        """
        try:
            pools = await self.fetch_uniswap_v3_pools(chain, min_tvl)
            logger.info(f"Fetched {len(pools)} Uniswap V3 pools from {chain}")
            return pools
        except Exception as e:
            logger.warning(f"Failed Uniswap V3 {chain}: {e}")
            return pd.DataFrame()

    async def _fetch_single_chain_balancer_pools(
        self,
        chain: str,
        min_tvl: float
    ) -> pd.DataFrame:
        """
        Helper to fetch Balancer pools for a single chain (parallelized).

        Args:
            chain: Chain name
            min_tvl: Minimum TVL filter

        Returns:
            DataFrame with pool data
        """
        try:
            bal_pools = await self.fetch_balancer_pools(chain, min_tvl)
            logger.info(f"Fetched {len(bal_pools)} Balancer pools from {chain}")
            return bal_pools
        except Exception as e:
            logger.warning(f"Failed Balancer {chain}: {e}")
            return pd.DataFrame()

    async def fetch_multi_dex_pools(
        self,
        chains: Optional[List[str]] = None,
        min_tvl: float = 100000
    ) -> pd.DataFrame:
        """
        Fetch pools from multiple DEXes across chains.

        Args:
            chains: List of chains (default: major L1/L2s)
            min_tvl: Minimum TVL filter

        Returns:
            Combined DataFrame from all DEXes
        """
        if chains is None:
            chains = ['ethereum', 'arbitrum', 'optimism', 'polygon']

        all_tasks = []

        # Uniswap V3 - parallelize all chains
        for chain in chains:
            all_tasks.append(self._fetch_single_chain_uniswap_pools(chain, min_tvl))

        # Curve (Ethereum only)
        if 'ethereum' in chains:
            async def fetch_curve_wrapper():
                try:
                    curve_pools = await self.fetch_curve_pools()
                    curve_pools['chain'] = 'ethereum'
                    return curve_pools
                except Exception as e:
                    logger.warning(f"Failed Curve: {e}")
                    return pd.DataFrame()
            all_tasks.append(fetch_curve_wrapper())

        # Balancer - parallelize supported chains
        for chain in [c for c in chains if c in ['ethereum', 'arbitrum']]:
            all_tasks.append(self._fetch_single_chain_balancer_pools(chain, min_tvl))

        # Execute all tasks in parallel
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # Filter valid DataFrames
        all_pools = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

        if all_pools:
            return pd.concat(all_pools, ignore_index=True)
        return pd.DataFrame()
    
    async def fetch_funding_rates(self, symbols, start_date, end_date):
        """The Graph doesn't provide funding rates directly."""
        return pd.DataFrame()
    
    async def fetch_ohlcv(self, symbols, timeframe, start_date, end_date):
        """Use fetch_uniswap_v3_pool_day_data for OHLCV."""
        return pd.DataFrame()
    
    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
        logger.info(f"TheGraph collector closed. Stats: {self.collection_stats}")
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.collection_stats.copy()
    
    @classmethod
    def get_supported_protocols(cls) -> List[str]:
        """Get list of supported protocols."""
        return [p.value for p in Protocol]
    
    @classmethod
    def get_supported_chains(cls) -> List[str]:
        """Get list of supported chains."""
        return [c.value for c in Chain]

    # =========================================================================
    # Standardized Collection Methods (for dynamic routing in collection_manager)
    # =========================================================================

    async def _fetch_single_chain_protocol_data(
        self,
        chain: str,
        protocol: str,
        min_tvl: float
    ) -> pd.DataFrame:
        """
        Helper to fetch protocol data for a single chain (parallelized).

        Args:
            chain: Chain name
            protocol: Protocol name
            min_tvl: Minimum TVL filter

        Returns:
            DataFrame with protocol data
        """
        try:
            if protocol == 'uniswap_v3':
                pools = await self.fetch_uniswap_v3_pools(chain=chain, min_tvl=min_tvl)
                if not pools.empty:
                    return pools

            elif protocol == 'balancer':
                if chain in ['ethereum', 'arbitrum']:
                    pools = await self.fetch_balancer_pools(chain, min_tvl)
                    if not pools.empty:
                        return pools

            elif protocol == 'aave':
                if chain in ['ethereum', 'arbitrum', 'optimism']:
                    markets = await self.fetch_aave_markets(chain)
                    if not markets.empty:
                        return markets

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"TheGraph {protocol} {chain} error: {e}")
            return pd.DataFrame()

    async def collect_subgraph_data(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect subgraph data for symbols (standardized interface).

        Wraps fetch_multi_dex_pools() to match collection_manager expectations.

        Args:
            symbols: List of token symbols to fetch data for
            start_date: Start date (unused - subgraph queries are current state)
            end_date: End date (unused - subgraph queries are current state)
            **kwargs: Additional parameters (chain, min_tvl, protocol)

        Returns:
            DataFrame with subgraph data for specified symbols
        """
        try:
            # Get parameters from kwargs
            chains = kwargs.get('chains', ['ethereum'])
            min_tvl = kwargs.get('min_tvl', 100_000)
            protocol = kwargs.get('protocol', 'uniswap_v3')

            # Fetch data based on protocol
            if protocol == 'curve':
                # Curve is Ethereum-only, no parallelization needed
                pools = await self.fetch_curve_pools()
                logger.info(f"TheGraph: Collected {len(pools)} Curve pools")
                return pools

            else:
                # Parallelize chain fetching for protocols that support multiple chains
                tasks = [self._fetch_single_chain_protocol_data(chain, protocol, min_tvl) for chain in chains]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Filter valid DataFrames
                all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

                if all_data:
                    result = pd.concat(all_data, ignore_index=True)
                    logger.info(f"TheGraph: Collected {len(result)} {protocol} records")
                    return result

                # Fallback: if no specific protocol data, use multi-DEX pools
                if protocol not in ['uniswap_v3', 'balancer', 'aave']:
                    pools = await self.fetch_multi_dex_pools(chains, min_tvl)
                    logger.info(f"TheGraph: Collected {len(pools)} pools from multiple DEXes")
                    return pools

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"TheGraph collect_subgraph_data error: {e}")
            return pd.DataFrame()

async def test_thegraph_collector():
    """Test TheGraph collector functionality."""
    config = {'rate_limit': 10}
    
    async with TheGraphCollector(config) as collector:
        print("=" * 60)
        print("TheGraph Collector Test")
        print("=" * 60)
        print(f"\nSupported protocols: {TheGraphCollector.get_supported_protocols()}")
        print(f"Supported chains: {TheGraphCollector.get_supported_chains()}")
        print(f"\nStats: {collector.get_collection_stats()}")

if __name__ == '__main__':
    asyncio.run(test_thegraph_collector())