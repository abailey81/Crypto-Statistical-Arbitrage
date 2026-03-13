"""
SushiSwap V2 / Uniswap V2 Data Collector via The Graph

validated collector for constant product AMM (x*y=k) protocols.
Supports SushiSwap and Uniswap V2 across multiple chains.

===============================================================================
PROTOCOL OVERVIEW
===============================================================================

V2 AMMs (Uniswap V2, SushiSwap) use the constant product formula:
    x * y = k
    
Where x and y are token reserves. This creates a hyperbolic price curve
with unlimited price range but concentrated liquidity around current price.

Key Characteristics:
    - 0.30% swap fee (distributed to LPs)
    - No concentrated liquidity (V3 feature)
    - Simpler LP position management
    - Lower gas costs than V3
    - Multi-chain deployment

===============================================================================
SUPPORTED CHAINS
===============================================================================

SushiSwap Subgraphs:
    - ethereum: Main deployment
    - arbitrum: L2 with low fees
    - polygon: High volume alt-L1
    - avalanche: C-Chain deployment
    - fantom: Opera network
    - bsc: BNB Smart Chain

Uniswap V2 Subgraphs:
    - ethereum: Original deployment
    - arbitrum: L2 deployment

===============================================================================
DATA TYPES COLLECTED
===============================================================================

1. Pool/Pair Data:
   - Token pair information
   - Reserve levels (TVL)
   - Historical volume
   - Transaction counts

2. Daily Metrics (PairDayData):
   - Daily OHLCV-equivalent
   - Volume in both tokens
   - Reserve snapshots
   - Transaction counts

3. Swap Events:
   - Individual trade data
   - Direction detection
   - USD volume
   - Counterparty addresses

4. Token Prices:
   - Derived from best pool
   - ETH-denominated
   - USD conversion via ETH price

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

1. Cross-DEX Arbitrage:
   - Compare V2 pools to V3 concentrated liquidity
   - Identify price discrepancies between protocols
   - Monitor reserve imbalances

2. Pairs Trading:
   - Track correlated token pairs
   - Identify mean-reversion opportunities
   - Measure pair divergence

3. Liquidity Analysis:
   - Reserve depth comparison
   - Volume/TVL ratio for efficiency
   - LP concentration metrics

4. Wash Trading Detection:
   - High volume/TVL ratios (>10x suspicious)
   - Low tx count per volume
   - Same-address round trips

===============================================================================
DATA QUALITY NOTES
===============================================================================

- The Graph hosted service being deprecated
- May need to migrate to decentralized network
- Subgraph indexing can lag during high activity
- Historical data quality varies by chain
- Some pools may have stale reserves

Version: 2.0.0
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import os

from ..base_collector import BaseCollector
from ..utils.rate_limiter import get_shared_rate_limiter
from ..utils.retry_handler import RetryHandler

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================

class Protocol(Enum):
    """Supported V2 protocols."""
    SUSHISWAP = 'sushiswap'
    UNISWAP_V2 = 'uniswap_v2'

class Chain(Enum):
    """Supported blockchain networks."""
    ETHEREUM = 'ethereum'
    ARBITRUM = 'arbitrum'
    POLYGON = 'polygon'
    AVALANCHE = 'avalanche'
    FANTOM = 'fantom'
    BSC = 'bsc'

class PairQuality(Enum):
    """Pair quality classification."""
    EXCELLENT = 'excellent' # High TVL, high volume, no wash flags
    GOOD = 'good' # Decent TVL/volume, minor concerns
    FAIR = 'fair' # Lower TVL/volume, some concerns
    POOR = 'poor' # Low metrics, potential issues
    SUSPICIOUS = 'suspicious' # Wash trading indicators

class LiquidityTier(Enum):
    """Liquidity tier classification."""
    DEEP = 'deep' # > $10M TVL
    MODERATE = 'moderate' # $1M - $10M TVL
    SHALLOW = 'shallow' # $100K - $1M TVL
    THIN = 'thin' # < $100K TVL

class SwapDirection(Enum):
    """Swap direction in a pair."""
    TOKEN0_TO_TOKEN1 = 'token0_to_token1'
    TOKEN1_TO_TOKEN0 = 'token1_to_token0'

class VolumeCategory(Enum):
    """Daily volume categorization."""
    VERY_HIGH = 'very_high' # > $10M
    HIGH = 'high' # $1M - $10M
    MODERATE = 'moderate' # $100K - $1M
    LOW = 'low' # $10K - $100K
    VERY_LOW = 'very_low' # < $10K

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class V2Pair:
    """V2 AMM pair/pool data with computed metrics."""
    pair_id: str
    token0_address: str
    token0_symbol: str
    token0_name: str
    token0_decimals: int
    token1_address: str
    token1_symbol: str
    token1_name: str
    token1_decimals: int
    reserve0: float
    reserve1: float
    reserve_usd: float
    volume_usd: float
    tx_count: int
    token0_price: float
    token1_price: float
    created_timestamp: int
    chain: str
    protocol: str
    
    @property
    def pair_name(self) -> str:
        """Formatted pair name."""
        return f"{self.token0_symbol}/{self.token1_symbol}"
    
    @property
    def volume_tvl_ratio(self) -> float:
        """Volume to TVL ratio (liquidity efficiency)."""
        return self.volume_usd / self.reserve_usd if self.reserve_usd > 0 else 0
    
    @property
    def is_wash_trading_suspect(self) -> bool:
        """Flag potential wash trading (volume > 10x TVL)."""
        return self.volume_tvl_ratio > 10
    
    @property
    def avg_trade_size(self) -> float:
        """Average trade size in USD."""
        return self.volume_usd / self.tx_count if self.tx_count > 0 else 0
    
    @property
    def liquidity_tier(self) -> LiquidityTier:
        """Classify liquidity depth."""
        if self.reserve_usd > 10_000_000:
            return LiquidityTier.DEEP
        elif self.reserve_usd > 1_000_000:
            return LiquidityTier.MODERATE
        elif self.reserve_usd > 100_000:
            return LiquidityTier.SHALLOW
        else:
            return LiquidityTier.THIN
    
    @property
    def pair_quality(self) -> PairQuality:
        """Assess overall pair quality."""
        if self.is_wash_trading_suspect:
            return PairQuality.SUSPICIOUS
        if self.reserve_usd > 1_000_000 and self.volume_usd > 100_000:
            return PairQuality.EXCELLENT
        if self.reserve_usd > 100_000 and self.volume_usd > 10_000:
            return PairQuality.GOOD
        if self.reserve_usd > 10_000:
            return PairQuality.FAIR
        return PairQuality.POOR
    
    @property
    def constant_product(self) -> float:
        """Calculate k in x*y=k formula."""
        return self.reserve0 * self.reserve1
    
    @property
    def implied_price_token0(self) -> float:
        """Price of token0 in terms of token1."""
        return self.reserve1 / self.reserve0 if self.reserve0 > 0 else 0
    
    @property
    def implied_price_token1(self) -> float:
        """Price of token1 in terms of token0."""
        return self.reserve0 / self.reserve1 if self.reserve1 > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'pair_id': self.pair_id,
            'pair_name': self.pair_name,
            'token0_address': self.token0_address,
            'token0_symbol': self.token0_symbol,
            'token1_address': self.token1_address,
            'token1_symbol': self.token1_symbol,
            'reserve0': self.reserve0,
            'reserve1': self.reserve1,
            'reserve_usd': self.reserve_usd,
            'volume_usd': self.volume_usd,
            'tx_count': self.tx_count,
            'volume_tvl_ratio': self.volume_tvl_ratio,
            'is_wash_trading_suspect': self.is_wash_trading_suspect,
            'liquidity_tier': self.liquidity_tier.value,
            'pair_quality': self.pair_quality.value,
            'constant_product': self.constant_product,
            'chain': self.chain,
            'protocol': self.protocol,
        }

@dataclass
class PairDayData:
    """Daily pair metrics (OHLCV equivalent)."""
    timestamp: datetime
    pair_id: str
    token0_symbol: str
    token1_symbol: str
    daily_volume_token0: float
    daily_volume_token1: float
    daily_volume_usd: float
    daily_txns: int
    reserve0: float
    reserve1: float
    reserve_usd: float
    price: float # token1/token0
    chain: str
    protocol: str
    
    @property
    def pair_name(self) -> str:
        """Formatted pair name."""
        return f"{self.token0_symbol}/{self.token1_symbol}"
    
    @property
    def volume_category(self) -> VolumeCategory:
        """Categorize daily volume."""
        if self.daily_volume_usd > 10_000_000:
            return VolumeCategory.VERY_HIGH
        elif self.daily_volume_usd > 1_000_000:
            return VolumeCategory.HIGH
        elif self.daily_volume_usd > 100_000:
            return VolumeCategory.MODERATE
        elif self.daily_volume_usd > 10_000:
            return VolumeCategory.LOW
        else:
            return VolumeCategory.VERY_LOW
    
    @property
    def utilization_rate(self) -> float:
        """Volume as percentage of TVL."""
        return (self.daily_volume_usd / self.reserve_usd * 100) if self.reserve_usd > 0 else 0
    
    @property
    def avg_trade_size(self) -> float:
        """Average trade size in USD."""
        return self.daily_volume_usd / self.daily_txns if self.daily_txns > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'pair_id': self.pair_id,
            'pair_name': self.pair_name,
            'token0_symbol': self.token0_symbol,
            'token1_symbol': self.token1_symbol,
            'daily_volume_token0': self.daily_volume_token0,
            'daily_volume_token1': self.daily_volume_token1,
            'daily_volume_usd': self.daily_volume_usd,
            'daily_txns': self.daily_txns,
            'reserve0': self.reserve0,
            'reserve1': self.reserve1,
            'reserve_usd': self.reserve_usd,
            'price': self.price,
            'volume_category': self.volume_category.value,
            'utilization_rate': self.utilization_rate,
            'avg_trade_size': self.avg_trade_size,
            'chain': self.chain,
            'protocol': self.protocol,
        }

@dataclass
class V2Swap:
    """Individual swap event data."""
    timestamp: datetime
    pair_id: str
    tx_hash: str
    block_number: int
    sender: str
    to: str
    amount0_in: float
    amount1_in: float
    amount0_out: float
    amount1_out: float
    amount_usd: float
    chain: str
    protocol: str
    
    @property
    def direction(self) -> SwapDirection:
        """Determine swap direction."""
        if self.amount0_in > 0:
            return SwapDirection.TOKEN0_TO_TOKEN1
        else:
            return SwapDirection.TOKEN1_TO_TOKEN0
    
    @property
    def is_buy(self) -> bool:
        """Check if this is a buy (token1 -> token0)."""
        return self.direction == SwapDirection.TOKEN1_TO_TOKEN0
    
    @property
    def execution_price(self) -> float:
        """Calculate execution price (token1/token0)."""
        if self.direction == SwapDirection.TOKEN0_TO_TOKEN1:
            return self.amount1_out / self.amount0_in if self.amount0_in > 0 else 0
        else:
            return self.amount0_out / self.amount1_in if self.amount1_in > 0 else 0
    
    @property
    def trade_size_usd(self) -> float:
        """Trade size in USD."""
        return self.amount_usd
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'pair_id': self.pair_id,
            'tx_hash': self.tx_hash,
            'block_number': self.block_number,
            'sender': self.sender,
            'to': self.to,
            'amount0_in': self.amount0_in,
            'amount1_in': self.amount1_in,
            'amount0_out': self.amount0_out,
            'amount1_out': self.amount1_out,
            'amount_usd': self.amount_usd,
            'direction': self.direction.value,
            'is_buy': self.is_buy,
            'execution_price': self.execution_price,
            'chain': self.chain,
            'protocol': self.protocol,
        }

@dataclass
class TokenPrice:
    """Token price derived from pool data."""
    timestamp: datetime
    address: str
    symbol: str
    name: str
    price_usd: float
    price_eth: float
    volume_usd: float
    liquidity: float
    chain: str
    protocol: str
    
    @property
    def has_liquidity(self) -> bool:
        """Check if token has meaningful liquidity."""
        return self.liquidity > 10_000
    
    @property
    def is_tradeable(self) -> bool:
        """Check if token is tradeable (has price and liquidity)."""
        return self.price_usd > 0 and self.has_liquidity
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'address': self.address,
            'symbol': self.symbol,
            'name': self.name,
            'price_usd': self.price_usd,
            'price_eth': self.price_eth,
            'volume_usd': self.volume_usd,
            'liquidity': self.liquidity,
            'has_liquidity': self.has_liquidity,
            'is_tradeable': self.is_tradeable,
            'chain': self.chain,
            'protocol': self.protocol,
        }

# =============================================================================
# Collector Class 
# =============================================================================

class SushiswapV2Collector(BaseCollector):
    """
    SushiSwap / Uniswap V2 data collector via The Graph.
    
    Collects constant product AMM data including pools, swaps,
    daily metrics, and derived token prices.
    
    Features:
        - Multi-chain support (6 chains)
        - Pool discovery with quality filters
        - Historical daily data
        - Individual swap events
        - Wash trading detection
        - Token price derivation
    
    Attributes:
        VENUE: Protocol identifier
        VENUE_TYPE: Protocol type ('DEX')
        protocol: 'sushiswap' or 'uniswap_v2'
    
    Example:
        >>> config = {'rate_limit': 15}
        >>> async with SushiswapV2Collector(config) as collector:
        ... pairs = await collector.fetch_pairs('ethereum')
        ... day_data = await collector.fetch_pair_day_data(
        ... pair_id, 'ethereum', '2024-01-01', '2024-01-31'
        ... )
    """
    
    VENUE = 'sushiswap_v2'
    VENUE_TYPE = 'DEX'

    # The Graph decentralized network subgraph IDs (requires THE_GRAPH_API_KEY)
    # Format: https://gateway-arbitrum.network.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}
    SUSHISWAP_SUBGRAPH_IDS = {
        'ethereum': '8faFtzhUW3vdQjxX3LjxR8mMPppSbr3oo6djCQwbhp2R', # Sushiswap Exchange
        'arbitrum': 'AoDCfVHZfL9jcmtP7JgfMc3LdPiSqf7e9XTGV6P5xGtG', # Sushiswap Arbitrum
        'polygon': 'B5XPLmjZnBHwWJe7EcfP4L9QXJR8u6PDh8xz2wGVV5yR', # Sushiswap Polygon
    }

    UNISWAP_V2_SUBGRAPH_IDS = {
        'ethereum': 'A3Np3RQbaBA6oKJgiwDJeo5T3zrYfGHPWFYayMwtNDum', # Uniswap V2
    }

    # Legacy hosted service URLs (deprecated - may still work temporarily)
    SUSHISWAP_URLS_LEGACY = {
        'ethereum': 'https://api.thegraph.com/subgraphs/name/sushiswap/exchange',
        'arbitrum': 'https://api.thegraph.com/subgraphs/name/sushiswap/arbitrum-exchange',
        'polygon': 'https://api.thegraph.com/subgraphs/name/sushiswap/matic-exchange',
        'avalanche': 'https://api.thegraph.com/subgraphs/name/sushiswap/avalanche-exchange',
        'fantom': 'https://api.thegraph.com/subgraphs/name/sushiswap/fantom-exchange',
        'bsc': 'https://api.thegraph.com/subgraphs/name/sushiswap/bsc-exchange',
    }

    UNISWAP_V2_URLS_LEGACY = {
        'ethereum': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2',
        'arbitrum': 'https://api.thegraph.com/subgraphs/name/ianlapham/arbitrum-minimal',
    }
    
    def __init__(self, config: Optional[Dict] = None, protocol: str = 'sushiswap'):
        """
        Initialize V2 collector.

        Args:
            config: Configuration dictionary
            protocol: 'sushiswap' or 'uniswap_v2'
        """
        config = config or {}
        super().__init__(config)

        # CRITICAL: Set supported data types for dynamic routing
        self.supported_data_types = ['pool_data', 'swaps', 'liquidity']
        self.venue = 'sushiswap' if protocol == 'sushiswap' else 'uniswap_v2'

        # Import VenueType from base_collector
        from ..base_collector import VenueType
        self.venue_type = VenueType.DEX
        self.requires_auth = False # Has fallback to DexScreener

        self.protocol = protocol

        # Graph API key for decentralized network (optional but recommended)
        self.graph_api_key = (
            config.get('graph_api_key') or
            config.get('the_graph_api_key') or
            os.getenv('THE_GRAPH_API_KEY', '') or
            os.getenv('GRAPH_API_KEY', '')
        )

        # Store subgraph IDs and legacy URLs
        self.subgraph_ids = (
            self.SUSHISWAP_SUBGRAPH_IDS if protocol == 'sushiswap'
            else self.UNISWAP_V2_SUBGRAPH_IDS
        )
        self.subgraph_urls_legacy = (
            self.SUSHISWAP_URLS_LEGACY if protocol == 'sushiswap'
            else self.UNISWAP_V2_URLS_LEGACY
        )

        if protocol == 'uniswap_v2':
            self.VENUE = 'uniswap_v2'

        rate_limit = config.get('rate_limit', 15)
        # Use shared rate limiter to avoid re-initialization overhead
        self.rate_limiter = get_shared_rate_limiter('sushiswap', rate=rate_limit, per=60.0, burst=3)
        self.retry_handler = RetryHandler(max_retries=3, base_delay=2.0)

        self.timeout = aiohttp.ClientTimeout(total=60)
        self.session: Optional[aiohttp.ClientSession] = None

        # Track subgraph availability
        self._subgraph_available = None

        self.collection_stats = {
            'records_collected': 0, 'api_calls': 0, 'subgraph_calls': 0, 'dexscreener_calls': 0, 'errors': 0
        }
        logger.info(f"Initialized {protocol} V2 collector with data types: {self.supported_data_types}")
    
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
    
    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
        logger.info(f"{self.protocol} V2 collector closed. Stats: {self.collection_stats}")

    async def _process_single_pair(self, pair: Dict, chain: str, min_volume: float) -> Optional[Dict]:
        """Helper method to process a single pair."""
        try:
            volume_usd = float(pair.get('volumeUSD', 0))
            if volume_usd < min_volume:
                return None

            token0 = pair.get('token0', {})
            token1 = pair.get('token1', {})

            v2_pair = V2Pair(
                pair_id=pair['id'],
                token0_address=token0.get('id', ''),
                token0_symbol=token0.get('symbol', ''),
                token0_name=token0.get('name', ''),
                token0_decimals=int(token0.get('decimals', 18)),
                token1_address=token1.get('id', ''),
                token1_symbol=token1.get('symbol', ''),
                token1_name=token1.get('name', ''),
                token1_decimals=int(token1.get('decimals', 18)),
                reserve0=float(pair.get('reserve0', 0)),
                reserve1=float(pair.get('reserve1', 0)),
                reserve_usd=float(pair.get('reserveUSD', 0)),
                volume_usd=volume_usd,
                tx_count=int(pair.get('txCount', 0)),
                token0_price=float(pair.get('token0Price', 0)),
                token1_price=float(pair.get('token1Price', 0)),
                created_timestamp=int(pair.get('createdAtTimestamp', 0)),
                chain=chain,
                protocol=self.protocol
            )

            return {
                **v2_pair.to_dict(),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE
            }
        except Exception as e:
            logger.debug(f"Error processing pair {pair.get('id', 'unknown')}: {e}")
            return None

    async def _process_single_day_data(self, day: Dict, pair_id: str, chain: str, end_ts: int) -> Optional[Dict]:
        """Helper method to process a single day data entry."""
        try:
            ts = int(day['date'])
            if ts > end_ts:
                return None

            r0 = float(day.get('reserve0', 0))
            r1 = float(day.get('reserve1', 0))
            price = r1 / r0 if r0 > 0 else 0

            pdd = PairDayData(
                timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
                pair_id=pair_id,
                token0_symbol=day.get('token0', {}).get('symbol', ''),
                token1_symbol=day.get('token1', {}).get('symbol', ''),
                daily_volume_token0=float(day.get('dailyVolumeToken0', 0)),
                daily_volume_token1=float(day.get('dailyVolumeToken1', 0)),
                daily_volume_usd=float(day.get('dailyVolumeUSD', 0)),
                daily_txns=int(day.get('dailyTxns', 0)),
                reserve0=r0,
                reserve1=r1,
                reserve_usd=float(day.get('reserveUSD', 0)),
                price=price,
                chain=chain,
                protocol=self.protocol
            )

            return {
                **pdd.to_dict(),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE
            }
        except Exception as e:
            logger.debug(f"Error processing day data: {e}")
            return None

    async def _process_single_swap(self, swap: Dict, pair_id: str, chain: str) -> Optional[Dict]:
        """Helper method to process a single swap."""
        try:
            tx = swap.get('transaction', {})

            v2_swap = V2Swap(
                timestamp=datetime.fromtimestamp(int(swap['timestamp']), tz=timezone.utc),
                pair_id=pair_id,
                tx_hash=tx.get('id', ''),
                block_number=int(tx.get('blockNumber', 0)),
                sender=swap.get('sender', ''),
                to=swap.get('to', ''),
                amount0_in=float(swap.get('amount0In', 0)),
                amount1_in=float(swap.get('amount1In', 0)),
                amount0_out=float(swap.get('amount0Out', 0)),
                amount1_out=float(swap.get('amount1Out', 0)),
                amount_usd=float(swap.get('amountUSD', 0)),
                chain=chain,
                protocol=self.protocol
            )

            return {
                **v2_swap.to_dict(),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE
            }
        except Exception as e:
            logger.debug(f"Error processing swap: {e}")
            return None

    async def _process_single_ticker(self, ticker: Dict, chain: str) -> Optional[Dict]:
        """Helper method to process a single ticker from CoinGecko."""
        try:
            volume_usd = ticker.get('converted_volume', {}).get('usd', 0) or 0
            last_price = ticker.get('last', 0) or 0

            return {
                'timestamp': datetime.now(timezone.utc),
                'pair_id': f"{ticker.get('base', '')}_{ticker.get('target', '')}",
                'token0_address': ticker.get('base', ''),
                'token0_symbol': ticker.get('coin_id', '').upper().replace('-', '_'),
                'token1_address': ticker.get('target', ''),
                'token1_symbol': ticker.get('target_coin_id', '').upper().replace('-', '_'),
                'reserve_usd': 0, # Not provided by CoinGecko
                'volume_usd': volume_usd,
                'tx_count': 0, # Not provided
                'token0_price': last_price,
                'token1_price': 1 / last_price if last_price > 0 else 0,
                'chain': chain,
                'protocol': self.protocol,
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE,
                'trust_score': ticker.get('trust_score', ''),
                'trade_url': ticker.get('trade_url', ''),
                'data_source': 'coingecko',
            }
        except Exception as e:
            logger.debug(f"Error parsing ticker: {e}")
            return None

    async def _search_dexscreener_symbol(self, collector, symbol: str, protocol: str) -> pd.DataFrame:
        """Helper method to search DexScreener for a single symbol."""
        try:
            # Map common tokens
            search_symbol = symbol.upper().replace('BTC', 'WBTC').replace('ETH', 'WETH')

            pairs = await collector.search_pairs(query=search_symbol, min_liquidity=100000)

            if pairs is not None and not pairs.empty:
                # Filter for Sushiswap only (or Uniswap V2 for uniswap_v2 protocol)
                dex_filter = 'sushi' if protocol == 'sushiswap' else 'uniswap'
                filtered_pairs = pairs[pairs['dex_id'].str.lower().str.contains(dex_filter, na=False)]
                if not filtered_pairs.empty:
                    await asyncio.sleep(1) # Rate limiting
                    return filtered_pairs

            return pd.DataFrame()
        except Exception as e:
            logger.debug(f"Error searching DexScreener for {symbol}: {e}")
            return pd.DataFrame()

    async def _fetch_pool_swaps(self, pool: pd.Series, chain: str, start_str: str, limit: int) -> pd.DataFrame:
        """Helper method to fetch swaps for a single pool."""
        try:
            pair_id = pool['pair_id']
            swaps = await self.fetch_swaps(
                pair_id=pair_id,
                chain=chain,
                start_date=start_str,
                limit=limit
            )

            if not swaps.empty:
                # Add pool context
                swaps['token0_symbol'] = pool.get('token0_symbol', '')
                swaps['token1_symbol'] = pool.get('token1_symbol', '')
                return swaps

            return pd.DataFrame()
        except Exception as e:
            logger.debug(f"Error fetching swaps for pool {pool.get('pair_id', 'unknown')}: {e}")
            return pd.DataFrame()
    
    def _get_subgraph_url(self, chain: str) -> Optional[str]:
        """
        Get the appropriate subgraph URL for the chain.

        Uses decentralized network if API key is available, otherwise falls back to legacy.

        Args:
            chain: Target chain

        Returns:
            Subgraph URL string or None if chain not supported
        """
        if self.graph_api_key and chain in self.subgraph_ids:
            subgraph_id = self.subgraph_ids[chain]
            return f"https://gateway-arbitrum.network.thegraph.com/api/{self.graph_api_key}/subgraphs/id/{subgraph_id}"

        # Fallback to legacy hosted service (deprecated but may still work)
        return self.subgraph_urls_legacy.get(chain)

    async def _query(
        self, chain: str, query: str, variables: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Execute GraphQL query against The Graph."""
        url = self._get_subgraph_url(chain)
        if not url:
            logger.warning(f"No subgraph URL available for chain {chain} for {self.protocol}")
            return None

        session = await self._get_session()
        await self.rate_limiter.acquire()

        try:
            async with session.post(
                url, json={'query': query, 'variables': variables or {}}
            ) as resp:
                self.collection_stats['api_calls'] += 1
                self.collection_stats['subgraph_calls'] += 1

                if resp.status == 402:
                    # Payment required - need API key
                    logger.warning(f"Subgraph requires API key (402). Set THE_GRAPH_API_KEY env var.")
                    self._subgraph_available = False
                    return None

                if resp.status != 200:
                    logger.warning(f"HTTP {resp.status} from The Graph")
                    return None

                result = await resp.json()

                if 'errors' in result:
                    error_msgs = [e.get('message', '') for e in result['errors']]
                    if any('removed' in msg.lower() for msg in error_msgs):
                        logger.warning(f"Subgraph endpoint deprecated: {error_msgs}")
                        self._subgraph_available = False
                    else:
                        logger.warning(f"GraphQL errors: {result['errors']}")
                    return None

                self._subgraph_available = True
                return result.get('data', {})

        except Exception as e:
            logger.error(f"Query error: {e}")
            self.collection_stats['errors'] += 1
            return None
    
    async def fetch_pairs(
        self, chain: str = 'ethereum',
        min_liquidity: float = 100_000,
        min_volume: float = 10_000,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch trading pairs with liquidity and volume filters.
        
        Args:
            chain: Blockchain network
            min_liquidity: Minimum TVL in USD
            min_volume: Minimum 24h volume in USD
            limit: Maximum pairs to fetch
            
        Returns:
            DataFrame with pair information and quality metrics
        """
        query = """
        query GetPairs($minLiquidity: BigDecimal!, $skip: Int!) {
            pairs(
                first: 1000, skip: $skip,
                orderBy: reserveUSD, orderDirection: desc,
                where: { reserveUSD_gte: $minLiquidity }
            ) {
                id
                token0 { id symbol name decimals }
                token1 { id symbol name decimals }
                reserve0 reserve1 reserveUSD
                volumeUSD txCount
                token0Price token1Price
                createdAtTimestamp createdAtBlockNumber
            }
        }
        """
        
        all_pairs = []
        skip = 0
        
        while len(all_pairs) < limit:
            data = await self._query(
                chain, query,
                variables={'minLiquidity': str(min_liquidity), 'skip': skip}
            )
            
            if not data or 'pairs' not in data:
                break
            
            pairs = data['pairs']
            if not pairs:
                break
            
            # Parallelize pair processing
            tasks = [self._process_single_pair(pair, chain, min_volume) for pair in pairs]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, dict):
                    all_pairs.append(result)
                elif isinstance(result, Exception):
                    logger.debug(f"Error processing pair: {result}")
                # None results are filtered pairs (volume too low)
            
            skip += 1000
            logger.info(f"Fetched {skip} {self.protocol} pairs from {chain}")
            
            if len(pairs) < 1000:
                break
        
        self.collection_stats['records_collected'] += len(all_pairs)
        
        df = pd.DataFrame(all_pairs[:limit])
        if not df.empty:
            df = df.sort_values('reserve_usd', ascending=False).reset_index(drop=True)
        return df
    
    async def fetch_pair_day_data(
        self, pair_id: str, chain: str = 'ethereum',
        start_date: str = '2024-01-01', end_date: str = '2024-12-31'
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV-equivalent data for a specific pair.
        
        Args:
            pair_id: Pair address/ID
            chain: Blockchain network
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with daily metrics
        """
        query = """
        query GetPairDayData($pairId: String!, $startDate: Int!, $skip: Int!) {
            pairDayDatas(
                first: 1000, skip: $skip, orderBy: date,
                where: { pairAddress: $pairId, date_gte: $startDate }
            ) {
                id date
                dailyVolumeToken0 dailyVolumeToken1 dailyVolumeUSD
                dailyTxns reserve0 reserve1 reserveUSD
                token0 { symbol }
                token1 { symbol }
            }
        }
        """
        
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        
        all_data = []
        skip = 0
        
        while True:
            data = await self._query(
                chain, query,
                variables={'pairId': pair_id.lower(), 'startDate': start_ts, 'skip': skip}
            )
            
            if not data or 'pairDayDatas' not in data:
                break
            
            day_data = data['pairDayDatas']
            if not day_data:
                break
            
            # Parallelize day data processing
            tasks = [self._process_single_day_data(day, pair_id, chain, end_ts) for day in day_data]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, dict):
                    all_data.append(result)
                elif isinstance(result, Exception):
                    logger.debug(f"Error processing day data: {result}")
                # None results are filtered (timestamp out of range)
            
            skip += 1000
            if len(day_data) < 1000:
                break
        
        self.collection_stats['records_collected'] += len(all_data)
        
        df = pd.DataFrame(all_data)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    async def fetch_swaps(
        self, pair_id: str, chain: str = 'ethereum',
        start_date: str = '2024-01-01', limit: int = 5000
    ) -> pd.DataFrame:
        """
        Fetch individual swap events for a pair.
        
        Args:
            pair_id: Pair address
            chain: Blockchain network
            start_date: Start date
            limit: Maximum swaps to fetch
            
        Returns:
            DataFrame with swap events
        """
        query = """
        query GetSwaps($pairId: String!, $startTime: Int!, $skip: Int!) {
            swaps(
                first: 1000, skip: $skip, orderBy: timestamp,
                where: { pair: $pairId, timestamp_gte: $startTime }
            ) {
                id timestamp sender to
                amount0In amount1In amount0Out amount1Out amountUSD
                transaction { id blockNumber }
            }
        }
        """
        
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        
        all_swaps = []
        skip = 0
        
        while len(all_swaps) < limit:
            data = await self._query(
                chain, query,
                variables={'pairId': pair_id.lower(), 'startTime': start_ts, 'skip': skip}
            )
            
            if not data or 'swaps' not in data:
                break
            
            swaps = data['swaps']
            if not swaps:
                break
            
            # Parallelize swap processing
            tasks = [self._process_single_swap(swap, pair_id, chain) for swap in swaps]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, dict):
                    all_swaps.append(result)
                elif isinstance(result, Exception):
                    logger.debug(f"Error processing swap: {result}")
            
            skip += 1000
            if len(swaps) < 1000:
                break
        
        self.collection_stats['records_collected'] += len(all_swaps)
        return pd.DataFrame(all_swaps[:limit])
    
    async def fetch_token_prices(
        self, token_address: str, chain: str = 'ethereum'
    ) -> Optional[TokenPrice]:
        """Fetch token price derived from best liquidity pair."""
        query = """
        query GetToken($tokenId: ID!) {
            token(id: $tokenId) {
                id symbol name derivedETH tradeVolumeUSD totalLiquidity
            }
            bundle(id: "1") { ethPrice }
        }
        """
        
        data = await self._query(
            chain, query, variables={'tokenId': token_address.lower()}
        )
        
        if not data:
            return None
        
        token = data.get('token', {})
        bundle = data.get('bundle', {})
        
        if not token:
            return None
        
        eth_price = float(bundle.get('ethPrice', 0))
        derived_eth = float(token.get('derivedETH', 0))
        
        return TokenPrice(
            timestamp=datetime.now(timezone.utc),
            address=token_address,
            symbol=token.get('symbol', ''),
            name=token.get('name', ''),
            price_usd=derived_eth * eth_price,
            price_eth=derived_eth,
            volume_usd=float(token.get('tradeVolumeUSD', 0)),
            liquidity=float(token.get('totalLiquidity', 0)),
            chain=chain,
            protocol=self.protocol
        )
    
    async def fetch_funding_rates(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """V2 AMMs don't have funding rates."""
        logger.info(f"{self.protocol}: No funding rates (spot AMM)")
        return pd.DataFrame()
    
    async def _fetch_single_symbol_ohlcv(
        self, symbol: str, pairs: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch OHLCV for a single symbol."""
        matching = pairs[
            (pairs['token0_symbol'] == symbol) |
            (pairs['token1_symbol'] == symbol)
        ].sort_values('reserve_usd', ascending=False)

        if matching.empty:
            return pd.DataFrame()

        best_pair = matching.iloc[0]

        day_data = await self.fetch_pair_day_data(
            best_pair['pair_id'],
            start_date=start_date,
            end_date=end_date
        )

        if not day_data.empty:
            day_data['symbol'] = symbol
            return day_data
        return pd.DataFrame()

    async def fetch_ohlcv(
        self, symbols: List[str], timeframe: str,
        start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch OHLCV via best liquidity pairs."""
        pairs = await self.fetch_pairs()
        if pairs.empty:
            return pd.DataFrame()

        # Parallelize symbol OHLCV fetching
        tasks = [
            self._fetch_single_symbol_ohlcv(symbol, pairs, start_date, end_date)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter valid DataFrames
        all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.collection_stats.copy()

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

        Wraps fetch_pairs() to match collection_manager expectations.

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
            min_liquidity = kwargs.get('min_liquidity', 100_000)
            min_volume = kwargs.get('min_volume', 10_000)
            limit = kwargs.get('limit', 500)

            # Fetch all pairs
            all_pairs = await self.fetch_pairs(
                chain=chain,
                min_liquidity=min_liquidity,
                min_volume=min_volume,
                limit=limit
            )

            if all_pairs.empty:
                logger.warning(f"{self.protocol}: No pools found on {chain} via The Graph")
                # Fallback: Use DexScreener to get Sushiswap pools
                all_pairs = await self._fetch_from_dexscreener(symbols, chain)
                if all_pairs.empty:
                    return pd.DataFrame()

            # Filter for requested symbols if specified
            if symbols and symbols != ['BTC']: # BTC is placeholder, ignore it
                symbol_filter = all_pairs[
                    all_pairs['token0_symbol'].isin(symbols) |
                    all_pairs['token1_symbol'].isin(symbols)
                ]
                if not symbol_filter.empty:
                    all_pairs = symbol_filter

            logger.info(f"{self.protocol}: Collected {len(all_pairs)} pools from {chain}")
            return all_pairs

        except Exception as e:
            logger.error(f"{self.protocol} collect_pool_data error: {e}")
            return pd.DataFrame()

    async def _fetch_from_coingecko(self, chain: str = 'ethereum') -> pd.DataFrame:
        """Fallback: Fetch Sushiswap pools from CoinGecko exchange API."""
        try:
            logger.info(f"Fetching {self.protocol} pools from CoinGecko (fallback)...")

            # CoinGecko exchange identifiers
            exchange_id = 'sushiswap' if self.protocol == 'sushiswap' else 'uniswap'

            session = await self._get_session()

            # CoinGecko exchange tickers endpoint (free, no API key required)
            url = f"https://api.coingecko.com/api/v3/exchanges/{exchange_id}/tickers"

            all_tickers = []
            page = 1

            while page <= 5: # Max 5 pages (500 tickers)
                try:
                    params = {'page': page, 'depth': 'false', 'include_exchange_logo': 'false'}
                    await self.rate_limiter.acquire()

                    async with session.get(url, params=params) as resp:
                        self.collection_stats['api_calls'] += 1

                        if resp.status == 429:
                            logger.warning("CoinGecko rate limited, waiting...")
                            await asyncio.sleep(60)
                            continue

                        if resp.status != 200:
                            logger.warning(f"CoinGecko API error {resp.status}")
                            break

                        data = await resp.json()
                        tickers = data.get('tickers', [])

                        if not tickers:
                            break

                        all_tickers.extend(tickers)
                        page += 1

                        await asyncio.sleep(1.5) # CoinGecko rate limit

                except Exception as e:
                    logger.warning(f"CoinGecko request error: {e}")
                    break

            if not all_tickers:
                return pd.DataFrame()

            # Parse tickers into DataFrame - parallelize processing
            tasks = [self._process_single_ticker(ticker, chain) for ticker in all_tickers]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            records = []
            for result in results:
                if isinstance(result, dict):
                    records.append(result)
                elif isinstance(result, Exception):
                    logger.debug(f"Error processing ticker: {result}")

            df = pd.DataFrame(records)

            if not df.empty:
                df = df.drop_duplicates(subset=['pair_id']).sort_values('volume_usd', ascending=False).reset_index(drop=True)

            self.collection_stats['records_collected'] += len(df)
            logger.info(f"Fetched {len(df)} {self.protocol} pools from CoinGecko")
            return df

        except Exception as e:
            logger.error(f"CoinGecko fallback error: {e}")
            return pd.DataFrame()

    async def _fetch_from_dexscreener(self, symbols: List[str], chain: str = 'ethereum') -> pd.DataFrame:
        """Fallback: Fetch Sushiswap pools from DexScreener when The Graph is down."""
        # First try CoinGecko (more reliable for SushiSwap)
        coingecko_result = await self._fetch_from_coingecko(chain)
        if not coingecko_result.empty:
            return coingecko_result

        # Fall back to DexScreener
        try:
            try:
                from data_collection.dex.dexscreener_collector import DexScreenerCollector
            except ImportError:
                from ..dex.dexscreener_collector import DexScreenerCollector

            logger.info(f"Fetching {self.protocol} pools from DexScreener (fallback)...")
            collector = DexScreenerCollector()

            # Use default symbols if none provided or if placeholder
            search_symbols = symbols if symbols and symbols != ['BTC'] else ['WETH', 'USDC', 'USDT', 'DAI']

            # Search for token pairs - parallelize symbol searches
            tasks = [
                self._search_dexscreener_symbol(collector, symbol, self.protocol)
                for symbol in search_symbols
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_pools = []
            for result in results:
                if isinstance(result, pd.DataFrame) and not result.empty:
                    all_pools.append(result)
                    self.collection_stats['dexscreener_calls'] += 1
                elif isinstance(result, Exception):
                    logger.debug(f"Error in DexScreener search: {result}")

            await collector.close()

            if all_pools:
                df = pd.concat(all_pools, ignore_index=True).drop_duplicates(subset=['pair_address'] if 'pair_address' in pd.concat(all_pools).columns else None)
                # Rename columns to match Sushiswap format
                column_mapping = {
                    'base_token_symbol': 'token0_symbol',
                    'quote_token_symbol': 'token1_symbol',
                    'liquidity_usd': 'reserve_usd',
                    'volume_24h': 'volume_usd',
                    'pair_address': 'pair_id',
                }
                for old_col, new_col in column_mapping.items():
                    if old_col in df.columns:
                        df = df.rename(columns={old_col: new_col})

                # Add missing columns with defaults
                if 'reserve_usd' not in df.columns and 'liquidity' in df.columns:
                    df['reserve_usd'] = df['liquidity']

                df['venue'] = self.VENUE
                df['venue_type'] = self.VENUE_TYPE
                df['data_source'] = 'dexscreener'

                self.collection_stats['records_collected'] += len(df)
                logger.info(f"Fetched {len(df)} {self.protocol} pools from DexScreener")
                return df

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"DexScreener fallback error: {e}")
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
            from datetime import datetime
            if isinstance(start_date, datetime):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            # Get chain from kwargs
            chain = kwargs.get('chain', 'ethereum')
            limit = kwargs.get('limit', 5000)

            # First fetch pools to find pair IDs for requested symbols
            pools = await self.collect_pool_data(symbols, start_date, end_date, **kwargs)

            if pools.empty:
                logger.warning(f"{self.protocol}: No pools found for symbols {symbols}")
                return pd.DataFrame()

            # Collect swaps for top 3 pools by liquidity - parallelize
            top_pools = pools.nlargest(3, 'reserve_usd')

            tasks = [
                self._fetch_pool_swaps(pool, chain, start_str, limit // len(top_pools))
                for _, pool in top_pools.iterrows()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_swaps = []
            for result in results:
                if isinstance(result, pd.DataFrame) and not result.empty:
                    all_swaps.append(result)
                elif isinstance(result, Exception):
                    logger.debug(f"Error fetching pool swaps: {result}")

            if all_swaps:
                result = pd.concat(all_swaps, ignore_index=True)
                logger.info(f"{self.protocol}: Collected {len(result)} swaps from {chain}")
                return result

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"{self.protocol} collect_swaps error: {e}")
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

        Liquidity data is part of pool data for Sushiswap.
        Standardized method name for collection manager compatibility.
        """
        try:
            # Just call collect_pool_data which includes liquidity metrics
            return await self.collect_pool_data(symbols, start_date, end_date, **kwargs)

        except Exception as e:
            logger.error(f"{self.protocol} collect_liquidity error: {e}")
            return pd.DataFrame()

class UniswapV2Collector(SushiswapV2Collector):
    """Uniswap V2 collector - same interface as SushiSwap."""

    VENUE = 'uniswap_v2'

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config, protocol='uniswap_v2')

# Alias for backward compatibility (collection_manager expects SushiSwapV2Collector)
SushiSwapV2Collector = SushiswapV2Collector