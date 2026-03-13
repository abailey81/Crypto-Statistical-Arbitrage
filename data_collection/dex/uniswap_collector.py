"""
Uniswap V3 Data Collector via The Graph

validated collector for Uniswap V3 concentrated liquidity AMM.
Supports multi-chain deployment with comprehensive pool analytics.

===============================================================================
PROTOCOL OVERVIEW
===============================================================================

Uniswap V3 introduced concentrated liquidity, allowing LPs to provide
liquidity within specific price ranges. This creates:
    - Higher capital efficiency (up to 4000x vs V2)
    - Multiple fee tiers per pair
    - Position NFTs for LP tracking
    - Tick-based price discretization

Key Formula:
    L = sqrt(x * y) (virtual liquidity)
    Price moves within tick ranges defined by LPs

===============================================================================
FEE TIERS
===============================================================================

Uniswap V3 offers multiple fee tiers:
    - 0.01% (1 bps): Stable pairs (USDC/USDT)
    - 0.05% (5 bps): Stable pairs, high volume
    - 0.30% (30 bps): Standard pairs (ETH/USDC)
    - 1.00% (100 bps): Exotic/volatile pairs

Fee selection impacts:
    - LP returns vs impermanent loss
    - Arbitrage profitability
    - Trade routing decisions

===============================================================================
SUPPORTED CHAINS
===============================================================================

The Graph Subgraphs:
    - ethereum: Original deployment, highest TVL
    - arbitrum: L2, lower fees
    - optimism: L2, OP incentives
    - polygon: Alt-L1, high volume
    - base: Coinbase L2

Decentralized Network IDs:
    - ethereum: 5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV
    - arbitrum: FbCGRftH4a3yZugY7TnbYgPJVEv2LvMT6oF1fxPe9aJM
    - optimism: Cghf4LfVqPiFw6fp6Y5X5Ubc8UpmUhSfJL82zwiBFLaj
    - polygon: 3hCPRGf4z88VC5rsBKU5AA9FBBq5nF3jbKJG7VZCbhjm

===============================================================================
DATA TYPES COLLECTED
===============================================================================

1. Pool Metadata:
   - Token pair and fee tier
   - TVL and volume metrics
   - Tick spacing and current tick
   - sqrtPriceX96 for price calculation

2. Pool Day Data:
   - Daily OHLCV equivalent
   - Fee revenue metrics
   - TVL snapshots

3. Swap Events:
   - Individual trade data
   - Price impact analysis
   - MEV detection signals

4. Position Data:
   - LP position analytics
   - Tick range analysis
   - Liquidity concentration

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

1. Cross-Fee-Tier Arbitrage:
   - Price discrepancies between fee tiers
   - Optimal fee tier for trade size
   - Fee tier migration signals

2. Concentrated Liquidity Analysis:
   - Tick range liquidity mapping
   - Slippage estimation
   - Optimal trade sizing

3. MEV Detection:
   - Sandwich attack identification
   - Just-in-time liquidity
   - Backrun opportunity detection

4. LP Analytics:
   - Position profitability
   - Impermanent loss tracking
   - Fee revenue analysis

===============================================================================
DATA QUALITY CONSIDERATIONS
===============================================================================

- sqrtPriceX96 requires careful conversion
- Tick math can overflow
- Historical positions require subgraph support
- Some chains have delayed indexing

Version: 2.0.0
"""

import asyncio
import aiohttp
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..base_collector import BaseCollector, VenueType
from ..utils.rate_limiter import get_shared_rate_limiter
from ..utils.retry_handler import RetryHandler

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

class FeeTier(Enum):
    """Uniswap V3 fee tiers in basis points."""
    LOWEST = 100 # 0.01% - Stable pairs
    LOW = 500 # 0.05% - Stable pairs
    MEDIUM = 3000 # 0.30% - Standard pairs
    HIGH = 10000 # 1.00% - Exotic pairs

class PoolQuality(Enum):
    """Pool quality classification."""
    PREMIUM = 'premium' # High TVL, high volume, trusted tokens
    STANDARD = 'standard' # Good TVL/volume
    EMERGING = 'emerging' # Growing pool
    RISKY = 'risky' # Low metrics or wash trading flags
    SUSPICIOUS = 'suspicious' # Strong wash trading indicators

class LiquidityDepth(Enum):
    """Liquidity depth classification."""
    VERY_DEEP = 'very_deep' # > $50M TVL
    DEEP = 'deep' # $10M - $50M TVL
    MODERATE = 'moderate' # $1M - $10M TVL
    SHALLOW = 'shallow' # $100K - $1M TVL
    THIN = 'thin' # < $100K TVL

class PriceMovement(Enum):
    """Price movement classification."""
    STRONG_UP = 'strong_up' # > 5%
    UP = 'up' # 1% - 5%
    STABLE = 'stable' # -1% to 1%
    DOWN = 'down' # -5% to -1%
    STRONG_DOWN = 'strong_down' # < -5%

class TradeSize(Enum):
    """Trade size classification."""
    WHALE = 'whale' # > $1M
    LARGE = 'large' # $100K - $1M
    MEDIUM = 'medium' # $10K - $100K
    SMALL = 'small' # $1K - $10K
    MICRO = 'micro' # < $1K

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class V3Pool:
    """Uniswap V3 pool data with computed analytics."""
    pool_id: str
    chain: str
    token0_address: str
    token0_symbol: str
    token0_name: str
    token0_decimals: int
    token1_address: str
    token1_symbol: str
    token1_name: str
    token1_decimals: int
    fee_tier_bps: int
    tvl_usd: float
    volume_usd: float
    tx_count: int
    liquidity: str
    sqrt_price: str
    tick: int
    created_timestamp: int

    @property
    def pair_name(self) -> str:
        """Formatted pair name with fee tier."""
        fee_pct = self.fee_tier_bps / 10000
        return f"{self.token0_symbol}/{self.token1_symbol} ({fee_pct}%)"

    @property
    def fee_percent(self) -> float:
        """Fee tier as percentage."""
        return self.fee_tier_bps / 10000

    @property
    def fee_tier(self) -> FeeTier:
        """Get fee tier enum."""
        tier_map = {100: FeeTier.LOWEST, 500: FeeTier.LOW,
                    3000: FeeTier.MEDIUM, 10000: FeeTier.HIGH}
        return tier_map.get(self.fee_tier_bps, FeeTier.MEDIUM)

    @property
    def volume_tvl_ratio(self) -> float:
        """Volume to TVL ratio (efficiency metric)."""
        return self.volume_usd / self.tvl_usd if self.tvl_usd > 0 else 0

    @property
    def is_wash_trading_suspect(self) -> bool:
        """Flag potential wash trading."""
        # High volume/TVL with low tx count is suspicious
        if self.volume_tvl_ratio > 10:
            return True
        if self.tx_count > 0:
            tx_per_volume = self.tx_count / self.volume_usd if self.volume_usd > 0 else 0
            return tx_per_volume < 0.0001 and self.volume_usd > 100_000
        return False

    @property
    def liquidity_depth(self) -> LiquidityDepth:
        """Classify liquidity depth."""
        if self.tvl_usd > 50_000_000:
            return LiquidityDepth.VERY_DEEP
        elif self.tvl_usd > 10_000_000:
            return LiquidityDepth.DEEP
        elif self.tvl_usd > 1_000_000:
            return LiquidityDepth.MODERATE
        elif self.tvl_usd > 100_000:
            return LiquidityDepth.SHALLOW
        else:
            return LiquidityDepth.THIN

    @property
    def pool_quality(self) -> PoolQuality:
        """Assess overall pool quality."""
        if self.is_wash_trading_suspect:
            return PoolQuality.SUSPICIOUS
        if self.tvl_usd > 10_000_000 and self.volume_usd > 1_000_000:
            return PoolQuality.PREMIUM
        if self.tvl_usd > 1_000_000 and self.volume_usd > 100_000:
            return PoolQuality.STANDARD
        if self.tvl_usd > 100_000:
            return PoolQuality.EMERGING
        return PoolQuality.RISKY

    @property
    def is_stable_pair(self) -> bool:
        """Check if pool is for stablecoin pair."""
        stables = ['USDC', 'USDT', 'DAI', 'FRAX', 'LUSD', 'BUSD']
        return (self.token0_symbol.upper() in stables and
                self.token1_symbol.upper() in stables)

    @property
    def expected_fee_tier(self) -> FeeTier:
        """Determine expected fee tier based on pair type."""
        if self.is_stable_pair:
            return FeeTier.LOWEST
        return FeeTier.MEDIUM

    @property
    def has_optimal_fee_tier(self) -> bool:
        """Check if pool uses optimal fee tier for pair type."""
        if self.is_stable_pair:
            return self.fee_tier_bps in [100, 500]
        return True

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'pool_id': self.pool_id,
            'chain': self.chain,
            'pair_name': self.pair_name,
            'token0_address': self.token0_address,
            'token0_symbol': self.token0_symbol,
            'token1_address': self.token1_address,
            'token1_symbol': self.token1_symbol,
            'fee_tier_bps': self.fee_tier_bps,
            'fee_percent': self.fee_percent,
            'tvl_usd': self.tvl_usd,
            'volume_usd': self.volume_usd,
            'tx_count': self.tx_count,
            'volume_tvl_ratio': self.volume_tvl_ratio,
            'is_wash_trading_suspect': self.is_wash_trading_suspect,
            'liquidity_depth': self.liquidity_depth.value,
            'pool_quality': self.pool_quality.value,
            'is_stable_pair': self.is_stable_pair,
            'tick': self.tick,
            'created_timestamp': self.created_timestamp,
        }

@dataclass
class PoolDayData:
    """Daily pool metrics (OHLCV equivalent)."""
    timestamp: datetime
    pool_id: str
    chain: str
    token0_symbol: str
    token1_symbol: str
    fee_tier_bps: int
    open: float
    high: float
    low: float
    close: float
    volume_usd: float
    volume_token0: float
    volume_token1: float
    tvl_usd: float
    tx_count: int
    fees_usd: float

    @property
    def pair_name(self) -> str:
        """Formatted pair name."""
        return f"{self.token0_symbol}/{self.token1_symbol}"

    @property
    def daily_return(self) -> float:
        """Daily return percentage."""
        return (self.close - self.open) / self.open * 100 if self.open > 0 else 0

    @property
    def price_movement(self) -> PriceMovement:
        """Classify price movement."""
        ret = self.daily_return
        if ret > 5:
            return PriceMovement.STRONG_UP
        elif ret > 1:
            return PriceMovement.UP
        elif ret > -1:
            return PriceMovement.STABLE
        elif ret > -5:
            return PriceMovement.DOWN
        else:
            return PriceMovement.STRONG_DOWN

    @property
    def daily_range_pct(self) -> float:
        """Daily price range as percentage."""
        return (self.high - self.low) / self.low * 100 if self.low > 0 else 0

    @property
    def fee_yield_daily(self) -> float:
        """Daily fee yield on TVL."""
        return self.fees_usd / self.tvl_usd * 100 if self.tvl_usd > 0 else 0

    @property
    def fee_yield_annualized(self) -> float:
        """Annualized fee yield."""
        return self.fee_yield_daily * 365

    @property
    def avg_trade_size(self) -> float:
        """Average trade size in USD."""
        return self.volume_usd / self.tx_count if self.tx_count > 0 else 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'pool_id': self.pool_id,
            'chain': self.chain,
            'pair_name': self.pair_name,
            'fee_tier_bps': self.fee_tier_bps,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume_usd': self.volume_usd,
            'tvl_usd': self.tvl_usd,
            'tx_count': self.tx_count,
            'fees_usd': self.fees_usd,
            'daily_return': self.daily_return,
            'price_movement': self.price_movement.value,
            'daily_range_pct': self.daily_range_pct,
            'fee_yield_daily': self.fee_yield_daily,
            'fee_yield_annualized': self.fee_yield_annualized,
            'avg_trade_size': self.avg_trade_size,
        }

@dataclass
class V3Swap:
    """Individual swap event with analytics."""
    timestamp: datetime
    pool_id: str
    chain: str
    swap_id: str
    tx_hash: str
    block_number: int
    sender: str
    recipient: str
    amount0: float
    amount1: float
    amount_usd: float
    tick: int
    gas_used: int
    gas_price: int

    @property
    def is_buy(self) -> bool:
        """Check if swap is buying token0."""
        return self.amount0 > 0

    @property
    def trade_size(self) -> TradeSize:
        """Classify trade size."""
        usd = abs(self.amount_usd)
        if usd > 1_000_000:
            return TradeSize.WHALE
        elif usd > 100_000:
            return TradeSize.LARGE
        elif usd > 10_000:
            return TradeSize.MEDIUM
        elif usd > 1_000:
            return TradeSize.SMALL
        else:
            return TradeSize.MICRO

    @property
    def execution_price(self) -> float:
        """Calculate execution price (token1/token0)."""
        if self.amount0 != 0:
            return abs(self.amount1 / self.amount0)
        return 0

    @property
    def gas_cost_eth(self) -> float:
        """Gas cost in ETH."""
        return self.gas_used * self.gas_price / 1e18

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'pool_id': self.pool_id,
            'chain': self.chain,
            'swap_id': self.swap_id,
            'tx_hash': self.tx_hash,
            'block_number': self.block_number,
            'sender': self.sender,
            'recipient': self.recipient,
            'amount0': self.amount0,
            'amount1': self.amount1,
            'amount_usd': self.amount_usd,
            'tick': self.tick,
            'is_buy': self.is_buy,
            'trade_size': self.trade_size.value,
            'execution_price': self.execution_price,
            'gas_used': self.gas_used,
            'gas_cost_eth': self.gas_cost_eth,
        }

# =============================================================================
# Collector Class
# =============================================================================

class UniswapV3Collector(BaseCollector):
    """
    Uniswap V3 data collector via The Graph.

    Collects concentrated liquidity AMM data including pools,
    daily metrics, swaps, and position analytics.

    Features:
        - Multi-chain support (5 chains)
        - Pool discovery with quality filters
        - Historical daily data
        - Swap event tracking
        - Wash trading detection
        - Fee tier analysis

    Attributes:
        VENUE: Protocol identifier ('uniswap_v3')
        VENUE_TYPE: Protocol type ('DEX')

    Example:
        >>> config = {'rate_limit': 15}
        >>> async with UniswapV3Collector(config) as collector:
        ... pools = await collector.fetch_pools('ethereum')
        ... day_data = await collector.fetch_pool_day_data(
        ... pool_ids, 'ethereum', '2024-01-01', '2024-01-31'
        ... )
    """

    VENUE = 'uniswap_v3'
    VENUE_TYPE = 'DEX'

    # Decentralized Network Deployment IDs (requires THE_GRAPH_API_KEY)
    # Free tier: 100,000 queries/month at thegraph.com/studio
    DEPLOYMENT_IDS = {
        'ethereum': '5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV',
        'arbitrum': 'FbCGRftH4a3yZugY7TnbYgPJVEv2LvMT6oF1fxPe9aJM',
        'optimism': 'Cghf4LfVqPiFw6fp6Y5X5Ubc8UpmUhSfJL82zwiBFLaj',
        'polygon': '3hCPRGf4z88VC5rsBKU5AA9FBBq5nF3jbKJG7VZCbhjm',
        'base': '43Hwfi3dJSoGpyas9VwNoDAv55yjgGrPpNSmbQZArzMG',
    }

    # Goldsky public endpoints (NO API KEY REQUIRED, rate limited: 50 req/10s)
    # https://docs.goldsky.com/subgraphs/graphql-endpoints
    GOLDSKY_URLS = {
        'ethereum': 'https://api.goldsky.com/api/public/project_clfy6g6nu0f5w01tg0hhe0nre/subgraphs/uniswap-v3/1.0.0/gn',
        'arbitrum': 'https://api.goldsky.com/api/public/project_clfy6g6nu0f5w01tg0hhe0nre/subgraphs/uniswap-v3-arbitrum/1.0.0/gn',
        'base': 'https://api.goldsky.com/api/public/project_cl8ylkiw00krx0hvza0qw17vn/subgraphs/uniswap-v3-base/prod/gn',
        'optimism': 'https://api.goldsky.com/api/public/project_clfy6g6nu0f5w01tg0hhe0nre/subgraphs/uniswap-v3-optimism/1.0.0/gn',
        'polygon': 'https://api.goldsky.com/api/public/project_clfy6g6nu0f5w01tg0hhe0nre/subgraphs/uniswap-v3-polygon/1.0.0/gn',
    }

    # Legacy hosted service URLs (DEPRECATED - service shut down)
    SUBGRAPH_URLS = {
        'ethereum': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
        'arbitrum': 'https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-arbitrum-one',
        'optimism': 'https://api.thegraph.com/subgraphs/name/ianlapham/optimism-post-regenesis',
        'polygon': 'https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-v3-polygon',
        'base': 'https://api.thegraph.com/subgraphs/name/lynnshaoyu/uniswap-v3-base',
    }

    FEE_TIERS = {100: 0.01, 500: 0.05, 3000: 0.30, 10000: 1.00}
    MAX_RESULTS = 1000

    def __init__(self, config: Optional[Dict] = None, api_key: Optional[str] = None):
        """Initialize Uniswap V3 collector."""
        config = config or {}
        super().__init__(config)

        # CRITICAL: Set supported data types for dynamic routing
        self.supported_data_types = ['pool_data', 'swaps', 'liquidity', 'dex_trades']
        self.venue = 'uniswap'
        self.venue_type = VenueType.DEX
        self.requires_auth = False # Has Goldsky public fallback (no API key required)

        self.api_key = api_key or os.getenv('THE_GRAPH_API_KEY')
        rate_limit = config.get('rate_limit', 15)
        # Use shared rate limiter to avoid re-initialization overhead
        self.rate_limiter = get_shared_rate_limiter('uniswap', rate=rate_limit, per=60.0, burst=3)
        self.retry_handler = RetryHandler(max_retries=3, base_delay=2.0)

        self.timeout = aiohttp.ClientTimeout(total=60)
        self.session: Optional[aiohttp.ClientSession] = None

        self._pool_cache: Dict[str, V3Pool] = {}
        self.collection_stats = {
            'records_collected': 0, 'api_calls': 0, 'errors': 0
        }
        self._request_sem = asyncio.Semaphore(2)

        logger.info(f"Initialized Uniswap V3 collector with supported types: {self.supported_data_types}")

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info(f"Uniswap V3 collector closed. Stats: {self.collection_stats}")

    async def _search_single_token_dexscreener(
        self, collector, token: str, min_tvl: float
    ) -> pd.DataFrame:
        """Search for a single token on DexScreener."""
        try:
            await asyncio.sleep(1) # Rate limiting
            pairs = await collector.search_pairs(query=token, min_liquidity=min_tvl)
            if not pairs.empty:
                # Filter for Uniswap only
                uni_pairs = pairs[pairs['dex_id'].str.lower().str.contains('uniswap', na=False)]
                if not uni_pairs.empty:
                    return uni_pairs
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"DexScreener search error for {token}: {e}")
            return pd.DataFrame()

    async def _normalize_single_symbol(
        self, symbol: str, token_mappings: Dict[str, str]
    ) -> str:
        """Normalize a single symbol using token mappings."""
        symbol_upper = symbol.upper()
        return token_mappings.get(symbol_upper, symbol_upper)

    async def _process_single_pool(
        self, pool: Dict, chain: str, min_volume: float, token_filter: Optional[List[str]]
    ) -> Optional[Dict]:
        """Process a single pool and return formatted data."""
        try:
            tvl = float(pool['totalValueLockedUSD'])
            volume = float(pool['volumeUSD'])

            if volume < min_volume:
                return None

            token0 = pool['token0']
            token1 = pool['token1']
            t0_sym = token0['symbol'].upper()
            t1_sym = token1['symbol'].upper()

            if token_filter:
                if t0_sym not in token_filter and t1_sym not in token_filter:
                    return None

            v3_pool = V3Pool(
                pool_id=pool['id'],
                chain=chain,
                token0_address=token0['id'],
                token0_symbol=t0_sym,
                token0_name=token0['name'],
                token0_decimals=int(token0['decimals']),
                token1_address=token1['id'],
                token1_symbol=t1_sym,
                token1_name=token1['name'],
                token1_decimals=int(token1['decimals']),
                fee_tier_bps=int(pool['feeTier']),
                tvl_usd=tvl,
                volume_usd=volume,
                tx_count=int(pool['txCount']),
                liquidity=pool['liquidity'],
                sqrt_price=pool['sqrtPrice'],
                tick=int(pool['tick']) if pool.get('tick') else 0,
                created_timestamp=int(pool['createdAtTimestamp'])
            )

            self._pool_cache[pool['id']] = v3_pool

            return {
                **v3_pool.to_dict(),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE
            }

        except (KeyError, ValueError) as e:
            logger.warning(f"Pool parse error: {e}")
            return None

    async def _query(
        self, chain: str, query: str, variables: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Execute GraphQL query against The Graph or Goldsky fallback."""
        async with self._request_sem:
            # Use decentralized network if API key is available
            if self.api_key and chain in self.DEPLOYMENT_IDS:
                deployment_id = self.DEPLOYMENT_IDS[chain]
                url = f"https://gateway-arbitrum.thegraph.com/api/{self.api_key}/subgraphs/id/{deployment_id}"
                logger.debug(f"Using The Graph decentralized network for {chain}")
            elif chain in self.GOLDSKY_URLS:
                # Use Goldsky public endpoint (no API key required)
                url = self.GOLDSKY_URLS[chain]
                logger.debug(f"Using Goldsky public endpoint for {chain}")
            else:
                logger.error(f"Unsupported chain: {chain}")
                return None

            if not self.session:
                self.session = aiohttp.ClientSession(timeout=self.timeout)

            await self.rate_limiter.acquire()

            try:
                async with self.session.post(
                    url, json={'query': query, 'variables': variables or {}}
                ) as resp:
                    self.collection_stats['api_calls'] += 1

                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"HTTP {resp.status} from subgraph: {error_text[:200]}")
                        return None

                    result = await resp.json()

                    if 'errors' in result:
                        logger.error(f"GraphQL errors: {result['errors']}")
                        return None

                    return result.get('data', {})

            except Exception as e:
                logger.error(f"Query error: {e}")
                self.collection_stats['errors'] += 1
                return None

    async def fetch_pools(
        self, chain: str = 'ethereum',
        min_tvl: float = 100_000,
        min_volume: float = 10_000,
        token_filter: Optional[List[str]] = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch pools with liquidity and volume filters.

        Args:
            chain: Blockchain network
            min_tvl: Minimum TVL in USD
            min_volume: Minimum volume in USD
            token_filter: Optional token symbols to filter
            limit: Maximum pools to fetch

        Returns:
            DataFrame with pool data and quality metrics
        """
        query = """
        query GetPools($first: Int!, $skip: Int!, $minTvl: BigDecimal!) {
            pools(
                first: $first, skip: $skip,
                orderBy: totalValueLockedUSD, orderDirection: desc,
                where: { totalValueLockedUSD_gte: $minTvl }
            ) {
                id
                token0 { id symbol name decimals }
                token1 { id symbol name decimals }
                feeTier liquidity sqrtPrice tick
                totalValueLockedUSD volumeUSD txCount
                createdAtTimestamp createdAtBlockNumber
            }
        }
        """

        all_pools = []
        skip = 0

        while len(all_pools) < limit:
            batch_size = min(self.MAX_RESULTS, limit - len(all_pools))

            data = await self._query(
                chain, query,
                variables={'first': batch_size, 'skip': skip, 'minTvl': str(min_tvl)}
            )

            if not data or 'pools' not in data:
                break

            pools = data['pools']
            if not pools:
                break

            # Parallelize pool processing
            pool_tasks = [
                self._process_single_pool(pool, chain, min_volume, token_filter)
                for pool in pools
            ]
            pool_results = await asyncio.gather(*pool_tasks, return_exceptions=True)

            # Filter valid results
            for result in pool_results:
                if isinstance(result, dict):
                    all_pools.append(result)

            skip += batch_size

            if len(pools) < batch_size:
                break

            await asyncio.sleep(0.5)

        self.collection_stats['records_collected'] += len(all_pools)

        df = pd.DataFrame(all_pools)
        if not df.empty:
            df = df.sort_values('tvl_usd', ascending=False).reset_index(drop=True)

            logger.info(f"Found {len(df)} V3 pools on {chain}")
            suspicious = df['is_wash_trading_suspect'].sum()
            if suspicious > 0:
                logger.warning(f" {suspicious} pools flagged for wash trading")
        else:
            # Fallback: Use DexScreener when The Graph fails
            logger.warning(f"The Graph returned no pools, trying DexScreener fallback...")
            df = await self._fetch_from_dexscreener(token_filter, chain, min_tvl)

        return df

    async def _fetch_from_dexscreener(self, token_filter: Optional[List[str]] = None, chain: str = 'ethereum', min_tvl: float = 100_000) -> pd.DataFrame:
        """Fallback: Fetch Uniswap pools from DexScreener when The Graph is down."""
        try:
            from data_collection.dex.dexscreener_collector import DexScreenerCollector

            logger.info("Fetching Uniswap pools from DexScreener...")
            collector = DexScreenerCollector()

            search_terms = token_filter if token_filter else ['WETH', 'USDC', 'USDT', 'WBTC']

            # Parallelize token searches
            token_tasks = [
                self._search_single_token_dexscreener(collector, token, min_tvl)
                for token in search_terms
            ]
            all_pools_results = await asyncio.gather(*token_tasks, return_exceptions=True)

            await collector.close()

            # Filter valid results
            all_pools = [r for r in all_pools_results if isinstance(r, pd.DataFrame) and not r.empty]

            if all_pools:
                df = pd.concat(all_pools, ignore_index=True)
                # Map DexScreener columns to Uniswap format
                df = df.rename(columns={
                    'pair_address': 'pool_id',
                    'base_token_symbol': 'token0_symbol',
                    'quote_token_symbol': 'token1_symbol',
                    'base_token_address': 'token0_address',
                    'quote_token_address': 'token1_address',
                    'liquidity_usd': 'tvl_usd',
                    'volume_24h': 'volume_usd',
                    'price_usd': 'token0_price'
                })
                df['chain'] = chain
                df['venue'] = self.VENUE
                df['venue_type'] = self.VENUE_TYPE
                df['source'] = 'dexscreener'
                logger.info(f"Fetched {len(df)} Uniswap pools from DexScreener")
                return df

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"DexScreener fallback error: {e}")
            return pd.DataFrame()

    async def _fetch_single_pool_day_data(
        self, pool_id: str, chain: str, start_ts: int, end_ts: int, query: str
    ) -> List[Dict]:
        """Fetch day data for a single pool."""
        all_records = []
        skip = 0

        while True:
            data = await self._query(
                chain, query,
                variables={
                    'poolId': pool_id.lower(), 'startTime': start_ts,
                    'endTime': end_ts, 'first': self.MAX_RESULTS, 'skip': skip
                }
            )

            if not data or 'poolDayDatas' not in data:
                break

            day_datas = data['poolDayDatas']
            if not day_datas:
                break

            for day in day_datas:
                try:
                    pool_info = day['pool']

                    pdd = PoolDayData(
                        timestamp=datetime.fromtimestamp(int(day['date']), tz=timezone.utc),
                        pool_id=pool_id,
                        chain=chain,
                        token0_symbol=pool_info['token0']['symbol'].upper(),
                        token1_symbol=pool_info['token1']['symbol'].upper(),
                        fee_tier_bps=int(pool_info['feeTier']),
                        open=float(day.get('open', 0)),
                        high=float(day.get('high', 0)),
                        low=float(day.get('low', 0)),
                        close=float(day.get('close', 0)),
                        volume_usd=float(day['volumeUSD']),
                        volume_token0=float(day['volumeToken0']),
                        volume_token1=float(day['volumeToken1']),
                        tvl_usd=float(day['tvlUSD']),
                        tx_count=int(day['txCount']),
                        fees_usd=float(day['feesUSD'])
                    )

                    all_records.append({
                        **pdd.to_dict(),
                        'venue': self.VENUE,
                        'venue_type': self.VENUE_TYPE
                    })

                except (KeyError, ValueError) as e:
                    logger.warning(f"Day data parse error: {e}")
                    continue

            skip += len(day_datas)

            if len(day_datas) < self.MAX_RESULTS:
                break

            await asyncio.sleep(0.3)

        return all_records

    async def fetch_pool_day_data(
        self, pool_ids: List[str], chain: str,
        start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV data for pools.

        Args:
            pool_ids: List of pool addresses
            chain: Blockchain network
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with daily pool metrics
        """
        query = """
        query GetPoolDayData($poolId: String!, $startTime: Int!, $endTime: Int!, $first: Int!, $skip: Int!) {
            poolDayDatas(
                first: $first, skip: $skip, orderBy: date, orderDirection: asc,
                where: { pool: $poolId, date_gte: $startTime, date_lte: $endTime }
            ) {
                date
                pool { id token0 { symbol } token1 { symbol } feeTier }
                open high low close
                volumeUSD volumeToken0 volumeToken1
                tvlUSD txCount feesUSD
            }
        }
        """

        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        # Parallelize pool fetching
        tasks = [
            self._fetch_single_pool_day_data(pool_id, chain, start_ts, end_ts, query)
            for pool_id in pool_ids
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten and filter valid results
        all_records = []
        for r in results:
            if isinstance(r, list):
                all_records.extend(r)

        self.collection_stats['records_collected'] += len(all_records)

        df = pd.DataFrame(all_records)
        if not df.empty:
            df = df.sort_values(['timestamp', 'pool_id']).reset_index(drop=True)
        return df

    async def fetch_swaps(
        self, pool_id: str, chain: str,
        start_date: str, end_date: str, limit: int = 10000
    ) -> pd.DataFrame:
        """Fetch individual swap transactions for a pool."""
        query = """
        query GetSwaps($poolId: String!, $startTime: BigInt!, $first: Int!, $skip: Int!) {
            swaps(
                first: $first, skip: $skip, orderBy: timestamp, orderDirection: asc,
                where: { pool: $poolId, timestamp_gte: $startTime }
            ) {
                id timestamp sender recipient
                amount0 amount1 amountUSD
                sqrtPriceX96 tick logIndex
                transaction { id blockNumber gasUsed gasPrice }
            }
        }
        """

        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        all_swaps = []
        skip = 0

        while len(all_swaps) < limit:
            batch_size = min(self.MAX_RESULTS, limit - len(all_swaps))

            data = await self._query(
                chain, query,
                variables={
                    'poolId': pool_id.lower(), 'startTime': str(start_ts),
                    'first': batch_size, 'skip': skip
                }
            )

            if not data or 'swaps' not in data:
                break

            swaps = data['swaps']
            if not swaps:
                break

            for swap in swaps:
                try:
                    ts = int(swap['timestamp'])
                    if ts > end_ts:
                        break

                    tx = swap.get('transaction', {})

                    v3_swap = V3Swap(
                        timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
                        pool_id=pool_id,
                        chain=chain,
                        swap_id=swap['id'],
                        tx_hash=tx.get('id', ''),
                        block_number=int(tx.get('blockNumber', 0)),
                        sender=swap['sender'],
                        recipient=swap['recipient'],
                        amount0=float(swap['amount0']),
                        amount1=float(swap['amount1']),
                        amount_usd=float(swap['amountUSD']),
                        tick=int(swap['tick']),
                        gas_used=int(tx.get('gasUsed', 0)),
                        gas_price=int(tx.get('gasPrice', 0))
                    )

                    all_swaps.append({
                        **v3_swap.to_dict(),
                        'venue': self.VENUE,
                        'venue_type': self.VENUE_TYPE
                    })

                except (KeyError, ValueError) as e:
                    logger.warning(f"Swap parse error: {e}")
                    continue

            skip += len(swaps)

            if len(swaps) < batch_size:
                break

            await asyncio.sleep(0.3)

        self.collection_stats['records_collected'] += len(all_swaps)

        df = pd.DataFrame(all_swaps)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    async def _fetch_single_chain_pools(
        self, chain: str, min_tvl: float, min_volume: float,
        token_filter: Optional[List[str]]
    ) -> pd.DataFrame:
        """Fetch pools for a single chain."""
        try:
            chain_pools = await self.fetch_pools(
                chain=chain, min_tvl=min_tvl, min_volume=min_volume,
                token_filter=token_filter
            )

            if not chain_pools.empty:
                logger.info(f" {chain}: {len(chain_pools)} pools")
                return chain_pools
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to fetch {chain} pools: {e}")
            return pd.DataFrame()

    async def fetch_multi_chain_pools(
        self, chains: Optional[List[str]] = None,
        min_tvl: float = 100_000, min_volume: float = 10_000,
        token_filter: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Fetch pools across multiple chains."""
        if chains is None:
            chains = list(self.SUBGRAPH_URLS.keys())

        # Parallelize chain fetching
        tasks = [
            self._fetch_single_chain_pools(chain, min_tvl, min_volume, token_filter)
            for chain in chains
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter valid DataFrames
        all_pools = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

        if not all_pools:
            return pd.DataFrame()

        combined = pd.concat(all_pools, ignore_index=True)
        combined = combined.sort_values('tvl_usd', ascending=False).reset_index(drop=True)

        logger.info(f"Total: {len(combined)} V3 pools across {len(chains)} chains")
        return combined

    async def fetch_funding_rates(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """V3 AMM doesn't have funding rates."""
        logger.info("Uniswap V3: No funding rates (spot AMM)")
        return pd.DataFrame()

    async def _fetch_single_symbol_ohlcv(
        self, symbol: str, pools: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch OHLCV for a single symbol."""
        mask = (pools['token0_symbol'] == symbol) | (pools['token1_symbol'] == symbol)
        symbol_pools = pools[mask]
        if symbol_pools.empty:
            return pd.DataFrame()

        pool_id = symbol_pools.iloc[0]['pool_id']
        day_data = await self.fetch_pool_day_data([pool_id], 'ethereum', start_date, end_date)
        return day_data

    async def fetch_ohlcv(
        self, symbols: List[str], timeframe: str,
        start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch OHLCV via best pools per token."""
        pools = await self.fetch_pools(token_filter=symbols)
        if pools.empty:
            return pd.DataFrame()

        # Parallelize symbol OHLCV fetching
        tasks = [
            self._fetch_single_symbol_ohlcv(symbol, pools, start_date, end_date)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter valid DataFrames
        all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    # -------------------------------------------------------------------------
    # Standardized Data Collection Methods (for dynamic routing)
    # -------------------------------------------------------------------------

    async def collect_pool_data(
        self, symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        **kwargs
    ) -> pd.DataFrame:
        """Collect pool data for symbols (standardized interface)."""
        # Convert datetime to string if needed
        start_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else start_date
        end_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else end_date

        chain = kwargs.get('chain', 'ethereum')

        # Common token mappings to match Uniswap pool symbols
        token_mappings = {
            'BTC': 'WBTC',
            'BITCOIN': 'WBTC',
            'ETH': 'WETH',
            'ETHEREUM': 'WETH',
        }

        # Normalize symbols (parallelized for consistency)
        symbol_tasks = [
            self._normalize_single_symbol(symbol, token_mappings)
            for symbol in symbols
        ]
        normalized_symbols_results = await asyncio.gather(*symbol_tasks, return_exceptions=True)

        # Filter valid results
        normalized_symbols = [
            r for r in normalized_symbols_results
            if isinstance(r, str)
        ]

        # Fetch pools filtered by symbols
        pools = await self.fetch_pools(
            chain=chain,
            min_tvl=100000,
            token_filter=normalized_symbols
        )

        if pools.empty:
            logger.warning(f"No Uniswap pools found for symbols: {symbols}")
            return pd.DataFrame()

        # Handle both 'id' and 'pool_id' columns (The Graph uses 'id', local uses 'pool_id')
        id_column = 'id' if 'id' in pools.columns else 'pool_id'

        if id_column not in pools.columns:
            logger.error(f"Uniswap pools DataFrame missing ID column. Columns: {pools.columns.tolist()}")
            return pd.DataFrame()

        # Get pool day data - if The Graph is down, return current pool data instead
        pool_ids = pools[id_column].tolist()[:50] # Limit to top 50 pools
        pool_data = await self.fetch_pool_day_data(pool_ids, chain, start_str, end_str)

        if not pool_data.empty:
            pool_data['venue'] = 'uniswap'
            pool_data['data_type'] = 'pool_data'
            logger.info(f"Collected {len(pool_data)} pool day records for {len(pool_ids)} pools")
            return pool_data

        # Fallback: If historical data unavailable (The Graph shutdown), return current pool snapshot
        if not pools.empty:
            pools['venue'] = 'uniswap'
            pools['data_type'] = 'pool_data'
            logger.info(f"Returning {len(pools)} current pool snapshots (historical data unavailable)")
            return pools

        return pd.DataFrame()

    async def collect_swaps(
        self, symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        **kwargs
    ) -> pd.DataFrame:
        """Collect swap data for symbols (standardized interface)."""
        # Convert datetime to string if needed
        start_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else start_date
        end_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else end_date

        chain = kwargs.get('chain', 'ethereum')

        # Get pools first
        pools = await self.fetch_pools(chain=chain, min_tvl=100000)

        if pools.empty:
            return pd.DataFrame()

        # Handle both 'id' and 'pool_id' columns
        id_column = 'id' if 'id' in pools.columns else 'pool_id'

        # Collect swaps from top pool
        pool_id = pools.iloc[0][id_column]
        swaps = await self.fetch_swaps(pool_id, chain, start_str, end_str, limit=1000)

        if not swaps.empty:
            swaps['venue'] = 'uniswap'
            swaps['data_type'] = 'swaps'

        return swaps

    async def collect_liquidity(
        self, symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        **kwargs
    ) -> pd.DataFrame:
        """Collect liquidity data for symbols (standardized interface)."""
        # For now, return pool data which includes liquidity metrics
        return await self.collect_pool_data(symbols, start_date, end_date, **kwargs)

    async def collect_dex_trades(
        self, symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect DEX trade data for symbols (standardized interface).

        This is an alias for collect_swaps() for DEX data type compatibility.
        """
        swaps = await self.collect_swaps(symbols, start_date, end_date, **kwargs)
        if not swaps.empty:
            swaps['data_type'] = 'dex_trades'
        return swaps

    async def close(self):
        """Close session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.collection_stats.copy()

# Alias for backwards compatibility with tests
UniswapCollector = UniswapV3Collector
