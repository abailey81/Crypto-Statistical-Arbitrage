"""
CowSwap (CoW Protocol) Data Collector

validated collector for CoW Protocol - a MEV-protected DEX using batch auctions.
CoW (Coincidence of Wants) matches traders peer-to-peer when possible, otherwise routes
through DEX aggregation for optimal execution.

Key Protocol Features:
    - MEV Protection: Batch auctions prevent front-running and sandwich attacks
    - Gasless Trading: Protocol pays gas, users sign off-chain orders
    - Peer-to-Peer: CoW matching eliminates LP fees when orders overlap
    - Surplus Optimization: Solvers compete to find best prices

Data Categories:
    - Trade Data: Individual trade execution with MEV savings
    - Settlements: Batch auction results with solver attribution
    - Solver Competition: Solver performance and rankings
    - Token Volumes: Aggregate trading volume by token
    - Daily Statistics: Protocol-level daily metrics

Supported Chains:
    - Ethereum Mainnet
    - Gnosis Chain
    - Arbitrum One

API Documentation: https://docs.cow.fi/
Subgraph: https://thegraph.com/hosted-service/subgraph/cowprotocol/cow

Statistical Arbitrage Applications:
    - MEV-protected execution for arb strategies
    - Cross-DEX price comparison (CowSwap vs direct DEX)
    - Solver efficiency analysis
    - Order flow toxicity measurement

Version: 2.0.0
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
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

class CowChain(Enum):
    """Supported CoW Protocol chains."""
    ETHEREUM = 'ethereum'
    GNOSIS = 'gnosis'
    ARBITRUM = 'arbitrum'

class OrderKind(Enum):
    """Order type classification."""
    SELL = 'sell'
    BUY = 'buy'

class OrderClass(Enum):
    """Order class (affects fee structure)."""
    MARKET = 'market'
    LIMIT = 'limit'
    LIQUIDITY = 'liquidity'

class SettlementStatus(Enum):
    """Settlement status."""
    PENDING = 'pending'
    EXECUTED = 'executed'
    CANCELLED = 'cancelled'
    EXPIRED = 'expired'

class TradeType(Enum):
    """Trade execution type."""
    COW = 'cow' # Peer-to-peer match (Coincidence of Wants)
    AMM = 'amm' # Routed through AMM
    PARTIAL = 'partial' # Partially filled

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class CowTrade:
    """Individual trade execution data."""
    timestamp: datetime
    trade_id: str
    tx_hash: str
    settlement_id: str
    solver: str
    
    # Token details
    buy_token_symbol: str
    buy_token_address: str
    sell_token_symbol: str
    sell_token_address: str
    
    # Amounts
    buy_amount: float
    sell_amount: float
    execution_price: float
    fee_amount: float
    
    # MEV protection
    surplus_amount: float # Positive = user got better price than expected
    surplus_pct: float
    
    # Metadata
    chain: str
    gas_price: int
    trade_type: str # COW, AMM, etc.
    mev_protected: bool = True
    
    @property
    def effective_price(self) -> float:
        """Price including surplus (actual execution)."""
        return self.execution_price
    
    @property
    def pair(self) -> str:
        """Trading pair string."""
        return f"{self.sell_token_symbol}/{self.buy_token_symbol}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'trade_id': self.trade_id,
            'tx_hash': self.tx_hash,
            'settlement_id': self.settlement_id,
            'solver': self.solver,
            'buy_token_symbol': self.buy_token_symbol,
            'buy_token_address': self.buy_token_address,
            'sell_token_symbol': self.sell_token_symbol,
            'sell_token_address': self.sell_token_address,
            'buy_amount': self.buy_amount,
            'sell_amount': self.sell_amount,
            'execution_price': self.execution_price,
            'fee_amount': self.fee_amount,
            'surplus_amount': self.surplus_amount,
            'surplus_pct': self.surplus_pct,
            'chain': self.chain,
            'gas_price': self.gas_price,
            'trade_type': self.trade_type,
            'mev_protected': self.mev_protected,
            'pair': self.pair,
        }

@dataclass
class CowSettlement:
    """Batch settlement data."""
    timestamp: datetime
    settlement_id: str
    tx_hash: str
    solver_id: str
    solver_address: Optional[str]
    
    # Settlement metrics
    num_trades: int
    num_orders: int
    total_volume_usd: float
    
    # Solver performance
    profitability: float # Solver profit in ETH
    gas_used: int
    gas_price: int
    
    # Efficiency metrics
    cow_volume_pct: float # % of volume matched peer-to-peer
    surplus_generated: float
    
    chain: str
    
    @property
    def avg_trade_size(self) -> float:
        """Average trade size in settlement."""
        return self.total_volume_usd / self.num_trades if self.num_trades > 0 else 0
    
    @property
    def gas_efficiency(self) -> float:
        """Gas per trade (lower is better)."""
        return self.gas_used / self.num_trades if self.num_trades > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'settlement_id': self.settlement_id,
            'tx_hash': self.tx_hash,
            'solver_id': self.solver_id,
            'solver_address': self.solver_address,
            'num_trades': self.num_trades,
            'num_orders': self.num_orders,
            'total_volume_usd': self.total_volume_usd,
            'profitability': self.profitability,
            'gas_used': self.gas_used,
            'gas_price': self.gas_price,
            'cow_volume_pct': self.cow_volume_pct,
            'surplus_generated': self.surplus_generated,
            'chain': self.chain,
            'avg_trade_size': self.avg_trade_size,
            'gas_efficiency': self.gas_efficiency,
        }

@dataclass
class SolverStats:
    """Solver competition statistics."""
    solver_id: str
    solver_address: str
    solver_name: Optional[str]
    
    # Performance metrics
    total_settlements: int
    total_trades: int
    total_volume_usd: float
    
    # Efficiency
    avg_trades_per_settlement: float
    avg_surplus_per_trade: float
    win_rate: float # % of auctions won
    
    # Rankings
    rank: int
    settlements_rank: int
    volume_rank: int
    
    chain: str
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'solver_id': self.solver_id,
            'solver_address': self.solver_address,
            'solver_name': self.solver_name,
            'total_settlements': self.total_settlements,
            'total_trades': self.total_trades,
            'total_volume_usd': self.total_volume_usd,
            'avg_trades_per_settlement': self.avg_trades_per_settlement,
            'avg_surplus_per_trade': self.avg_surplus_per_trade,
            'win_rate': self.win_rate,
            'rank': self.rank,
            'settlements_rank': self.settlements_rank,
            'volume_rank': self.volume_rank,
            'chain': self.chain,
            'timestamp': self.timestamp,
        }

@dataclass
class DailyVolume:
    """Daily protocol statistics."""
    date: datetime
    timestamp: datetime
    
    # Volume metrics
    total_volume_usd: float
    volume_eth: float
    fees_usd: float
    
    # Activity metrics
    num_trades: int
    num_orders: int
    num_settlements: int
    unique_tokens: int
    unique_traders: int
    
    # Efficiency
    cow_volume_usd: float # Peer-to-peer matched volume
    cow_volume_pct: float
    avg_trade_size: float
    
    chain: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'date': self.date,
            'timestamp': self.timestamp,
            'total_volume_usd': self.total_volume_usd,
            'volume_eth': self.volume_eth,
            'fees_usd': self.fees_usd,
            'num_trades': self.num_trades,
            'num_orders': self.num_orders,
            'num_settlements': self.num_settlements,
            'unique_tokens': self.unique_tokens,
            'unique_traders': self.unique_traders,
            'cow_volume_usd': self.cow_volume_usd,
            'cow_volume_pct': self.cow_volume_pct,
            'avg_trade_size': self.avg_trade_size,
            'chain': self.chain,
        }

# =============================================================================
# Collector Class
# =============================================================================

class CowSwapCollector(BaseCollector):
    """
    CowSwap (CoW Protocol) data collector.
    
    Collects MEV-protected DEX data via batch auction mechanism:
    - Trade execution data with surplus tracking
    - Settlement/batch auction results
    - Solver competition metrics
    - Protocol-level volume statistics
    
    CoW Protocol Key Concepts:
    - Batch Auctions: Orders collected and settled together
    - Solvers: Compete to find optimal trade execution
    - CoW: Coincidence of Wants - peer-to-peer matching
    - Surplus: Price improvement vs. quoted price
    
    Attributes:
        VENUE: Venue identifier ('cowswap')
        VENUE_TYPE: Venue classification ('DEX')
        API_URLS: Chain-specific API endpoints
        SUBGRAPH_URLS: Chain-specific subgraph URLs
    
    Example:
        >>> config = {'rate_limit': 15}
        >>> async with CowSwapCollector(config) as collector:
        ... trades = await collector.fetch_trades('2024-01-01', '2024-01-31')
        ... settlements = await collector.fetch_settlements('2024-01-01', '2024-01-31')
        ... solvers = await collector.fetch_solver_stats()
    """
    
    VENUE = 'cowswap'
    VENUE_TYPE = 'DEX'

    # Safety limit to prevent infinite pagination loops
    MAX_PAGINATION_ITERATIONS = 100

    # API endpoints by chain (REST API - primary data source)
    API_URLS = {
        'ethereum': 'https://api.cow.fi/mainnet',
        'gnosis': 'https://api.cow.fi/xdai',
        'arbitrum': 'https://api.cow.fi/arbitrum_one',
        'base': 'https://api.cow.fi/base',
    }

    # The Graph decentralized network subgraph IDs (requires THE_GRAPH_API_KEY)
    # Format: https://gateway-arbitrum.network.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}
    # Note: These are the subgraph IDs from The Graph's decentralized network
    SUBGRAPH_IDS = {
        'ethereum': 'EeYqsFQJyLmE2R2CeDTKoMBrDNAoT5FMvjL3SdrLy2Ja',
        'gnosis': 'EYSsJ5qLxHdQPYVZE9rDKHJxbZVzBd38FqCWi7NQKP8N',
        'arbitrum': 'AH6ZNHPL8SQpjQFhyJmPLu8Yt7FbqJDMk7aAPRQyYhdb',
    }

    # Fallback: Old hosted service URLs (deprecated but may still work for some queries)
    SUBGRAPH_URLS_LEGACY = {
        'ethereum': 'https://api.thegraph.com/subgraphs/name/cowprotocol/cow',
        'gnosis': 'https://api.thegraph.com/subgraphs/name/cowprotocol/cow-gc',
        'arbitrum': 'https://api.thegraph.com/subgraphs/name/cowprotocol/cow-arbitrum-one',
    }

    # Well-known trader addresses for REST API fallback (when subgraph unavailable)
    # These are active traders/protocols that interact with CowSwap frequently
    WELL_KNOWN_TRADERS = {
        'ethereum': [
            '0x9008D19f58AAbD9eD0D60971565AA8510560ab41', # CoW Protocol GPv2Settlement
            '0x40A50cf069e992AA4536211B23F286eF88752187', # CoW Protocol VaultRelayer
            '0x6810e776880C02933D47DB1b9fc05908e5386b96', # Gnosis Safe Proxy Factory
            '0x3328F7f4A1D1C57c35df56bBf0c9dCAFCA309C49', # Balancer Vault
        ],
        'gnosis': [
            '0x9008D19f58AAbD9eD0D60971565AA8510560ab41', # CoW Protocol GPv2Settlement
        ],
        'arbitrum': [
            '0x9008D19f58AAbD9eD0D60971565AA8510560ab41', # CoW Protocol GPv2Settlement
        ],
    }
    
    # Common token addresses by chain
    TOKEN_MAP = {
        'ethereum': {
            'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
            'USDC': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
            'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
            'DAI': '0x6B175474E89094C44Da98b954EescdeCB5B0B0d',
            'WBTC': '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599',
            'COW': '0xDEf1CA1fb7FBcDC777520aa7f396b4E015F497aB',
        },
        'gnosis': {
            'WXDAI': '0xe91D153E0b41518A2Ce8Dd3D7944Fa863463a97d',
            'USDC': '0xDDAfbb505ad214D7b80b1f830fcCc89B60fb7A83',
            'COW': '0x177127622c4A00F3d409B75571e12cB3c8973d3c',
        },
        'arbitrum': {
            'WETH': '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',
            'USDC': '0xaf88d065e77c8cC2239327C5EDb3A432268e5831',
            'ARB': '0x912CE59144191C1204E64559FE8253a0e49E6548',
        }
    }
    
    # Default rate limit
    DEFAULT_RATE_LIMIT = 15
    
    def __init__(self, config: Dict = None):
        """
        Initialize CowSwap collector.

        Args:
            config: Configuration dictionary with optional keys:
                - rate_limit: Requests per minute (default: 15)
                - timeout: Request timeout in seconds (default: 30)
                - graph_api_key: Optional Graph Protocol API key for decentralized network
                - the_graph_api_key: Alternative key name for Graph API key
        """
        config = config or {}
        super().__init__(config)

        # CRITICAL: Set supported data types for dynamic routing
        self.supported_data_types = ['swaps', 'orders', 'trades']
        self.venue = 'cowswap'

        # Import VenueType from base_collector
        from ..base_collector import VenueType
        self.venue_type = VenueType.DEX
        self.requires_auth = False # CowSwap REST API is public, subgraph may need key

        # Graph API key for decentralized network (optional but recommended)
        self.graph_api_key = (
            config.get('graph_api_key') or
            config.get('the_graph_api_key') or
            os.getenv('THE_GRAPH_API_KEY', '') or
            os.getenv('GRAPH_API_KEY', '')
        )

        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = get_shared_rate_limiter(
            'cowswap',
            rate=config.get('rate_limit', self.DEFAULT_RATE_LIMIT),
            per=60,
            burst=5
        )
        self.retry_handler = RetryHandler(
            max_retries=config.get('max_retries', 3),
            base_delay=1.0,
            max_delay=30.0
        )
        self._timeout = config.get('timeout', 30)

        # Track which data source is available
        self._subgraph_available = None # Will be tested on first query

        # Collection statistics
        self.collection_stats = {
            'records_collected': 0,
            'api_calls': 0,
            'subgraph_calls': 0,
            'rest_api_calls': 0,
            'errors': 0,
        }
    
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
            connector = aiohttp.TCPConnector(limit=10)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'Accept': 'application/json'}
            )
        return self.session
    
    async def _api_request(
        self,
        endpoint: str,
        chain: str = 'ethereum',
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Make request to CowSwap REST API.
        
        Args:
            endpoint: API endpoint path
            chain: Target chain
            params: Query parameters
            
        Returns:
            JSON response or empty dict on error
        """
        base_url = self.API_URLS.get(chain, self.API_URLS['ethereum'])
        url = f"{base_url}{endpoint}"
        
        session = await self._get_session()
        await self.rate_limiter.acquire()
        self.collection_stats['api_calls'] += 1
        
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.warning(f"CowSwap API error {resp.status}: {endpoint}")
                    return {}
        except Exception as e:
            logger.error(f"CowSwap API request failed: {e}")
            self.collection_stats['errors'] += 1
            return {}
    
    def _get_subgraph_url(self, chain: str = 'ethereum') -> str:
        """
        Get the appropriate subgraph URL for the chain.

        Uses decentralized network if API key is available, otherwise falls back to legacy.

        Args:
            chain: Target chain

        Returns:
            Subgraph URL string
        """
        if self.graph_api_key and chain in self.SUBGRAPH_IDS:
            subgraph_id = self.SUBGRAPH_IDS[chain]
            return f"https://gateway-arbitrum.network.thegraph.com/api/{self.graph_api_key}/subgraphs/id/{subgraph_id}"

        # Fallback to legacy hosted service (deprecated but may still work)
        return self.SUBGRAPH_URLS_LEGACY.get(chain, self.SUBGRAPH_URLS_LEGACY.get('ethereum', ''))

    async def _subgraph_query(
        self,
        query: str,
        variables: Optional[Dict] = None,
        chain: str = 'ethereum'
    ) -> Dict:
        """
        Query CowSwap subgraph via The Graph.

        Tries decentralized network first (if API key available), then legacy hosted service.

        Args:
            query: GraphQL query string
            variables: Query variables
            chain: Target chain

        Returns:
            Query result data or empty dict on error
        """
        url = self._get_subgraph_url(chain)

        if not url:
            logger.warning(f"No subgraph URL available for chain {chain}")
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
                        logger.warning(f"GraphQL errors: {result['errors']}")
                        # Mark subgraph as unavailable for this session
                        self._subgraph_available = False
                        return {}
                    self._subgraph_available = True
                    return result.get('data', {})
                elif resp.status == 402:
                    # Payment required - need API key
                    logger.warning(f"Subgraph requires API key (402). Set THE_GRAPH_API_KEY env var.")
                    self._subgraph_available = False
                    return {}
                else:
                    logger.warning(f"Subgraph error {resp.status}")
                    return {}
        except Exception as e:
            logger.error(f"Subgraph query failed: {e}")
            self.collection_stats['errors'] += 1
            return {}

    # =========================================================================
    # REST API Trade Methods (fallback when subgraph unavailable)
    # =========================================================================

    async def _fetch_trades_rest_api(
        self,
        chain: str = 'ethereum',
        owner: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Fetch trades via REST API for a specific owner address.

        Args:
            chain: Target chain
            owner: Trader address (required by CoW API)
            limit: Maximum trades to return

        Returns:
            List of trade dictionaries
        """
        if not owner:
            return []

        base_url = self.API_URLS.get(chain, self.API_URLS['ethereum'])
        url = f"{base_url}/api/v1/trades"

        session = await self._get_session()
        await self.rate_limiter.acquire()
        self.collection_stats['rest_api_calls'] += 1

        try:
            params = {'owner': owner}
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    trades = await resp.json()
                    return trades[:limit] if trades else []
                else:
                    logger.warning(f"REST API trades error {resp.status}: {owner[:10]}...")
                    return []
        except Exception as e:
            logger.error(f"REST API trades request failed: {e}")
            self.collection_stats['errors'] += 1
            return []

    async def _fetch_orders_rest_api(
        self,
        chain: str = 'ethereum',
        owner: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Fetch orders via REST API for a specific owner address.

        Args:
            chain: Target chain
            owner: Trader address (required by CoW API)
            limit: Maximum orders to return

        Returns:
            List of order dictionaries
        """
        if not owner:
            return []

        base_url = self.API_URLS.get(chain, self.API_URLS['ethereum'])
        url = f"{base_url}/api/v1/account/{owner}/orders"

        session = await self._get_session()
        await self.rate_limiter.acquire()
        self.collection_stats['rest_api_calls'] += 1

        try:
            params = {'limit': limit, 'offset': 0}
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    orders = await resp.json()
                    return orders[:limit] if orders else []
                else:
                    logger.warning(f"REST API orders error {resp.status}: {owner[:10]}...")
                    return []
        except Exception as e:
            logger.error(f"REST API orders request failed: {e}")
            self.collection_stats['errors'] += 1
            return []

    async def _fetch_single_trader_trades(
        self,
        trader_addr: str,
        chain: str,
        per_address_limit: int
    ) -> List[Dict]:
        """
        Helper to fetch trades for a single trader address (parallelized).

        Args:
            trader_addr: Trader address
            chain: Target chain
            per_address_limit: Limit per address

        Returns:
            List of trade dictionaries
        """
        try:
            trades = await self._fetch_trades_rest_api(chain, trader_addr, per_address_limit)

            all_trades = []
            for trade in trades:
                try:
                    all_trades.append({
                        'timestamp': pd.to_datetime(trade.get('executionTime', datetime.utcnow().isoformat())),
                        'trade_id': f"{trade.get('orderUid', '')}_{trade.get('logIndex', 0)}",
                        'tx_hash': trade.get('txHash', ''),
                        'order_uid': trade.get('orderUid', ''),
                        'owner': trade.get('owner', ''),
                        'buy_token_address': trade.get('buyToken', ''),
                        'sell_token_address': trade.get('sellToken', ''),
                        'buy_amount': float(trade.get('buyAmount', 0)),
                        'sell_amount': float(trade.get('sellAmount', 0)),
                        'sell_amount_before_fees': float(trade.get('sellAmountBeforeFees', 0)),
                        'block_number': int(trade.get('blockNumber', 0)),
                        'log_index': int(trade.get('logIndex', 0)),
                        'chain': chain,
                        'venue': self.VENUE,
                        'venue_type': self.VENUE_TYPE,
                        'mev_protected': True,
                        'data_source': 'rest_api',
                    })
                except Exception as e:
                    logger.debug(f"Error parsing trade: {e}")
                    continue

            return all_trades

        except Exception as e:
            logger.warning(f"Error fetching trades for {trader_addr}: {e}")
            return []

    async def fetch_trades_via_rest_api(
        self,
        chain: str = 'ethereum',
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch trades via REST API using well-known trader addresses.

        This is a fallback method when the subgraph is unavailable.
        Fetches trades from known active traders like CoW Protocol settlement contract.

        Args:
            chain: Target chain
            limit: Total trades to collect across all addresses

        Returns:
            DataFrame with trade data
        """
        logger.info(f"Fetching CowSwap trades via REST API for {chain}")

        traders = self.WELL_KNOWN_TRADERS.get(chain, [])
        if not traders:
            logger.warning(f"No well-known traders for chain {chain}")
            return pd.DataFrame()

        per_address_limit = max(50, limit // len(traders))

        # Parallelize trader fetching using asyncio.gather
        tasks = [self._fetch_single_trader_trades(trader_addr, chain, per_address_limit) for trader_addr in traders]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten and filter valid results
        all_trades = []
        for result in results:
            if isinstance(result, list):
                all_trades.extend(result)
                if len(all_trades) >= limit:
                    break

        df = pd.DataFrame(all_trades[:limit])

        if not df.empty:
            df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)

        self.collection_stats['records_collected'] += len(df)
        logger.info(f"Fetched {len(df)} CowSwap trades via REST API")
        return df

    # =========================================================================
    # Trade Data Methods
    # =========================================================================

    async def fetch_trades(
        self,
        start_date: str,
        end_date: str,
        chain: str = 'ethereum',
        limit: int = 1000,
        token_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical trades from CowSwap.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            chain: Target chain
            limit: Maximum trades to fetch
            token_filter: Optional token symbol filter
            
        Returns:
            DataFrame with trade execution data
        """
        logger.info(f"Fetching CowSwap trades from {chain}: {start_date} to {end_date}")
        
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        
        query = """
        query GetTrades($startTime: Int!, $endTime: Int!, $skip: Int!, $limit: Int!) {
            trades(
                first: $limit,
                skip: $skip,
                where: {
                    timestamp_gte: $startTime,
                    timestamp_lte: $endTime
                },
                orderBy: timestamp,
                orderDirection: asc
            ) {
                id
                timestamp
                txHash
                settlement {
                    id
                    solver {
                        id
                        address
                    }
                }
                buyAmount
                sellAmount
                buyToken {
                    id
                    symbol
                    decimals
                }
                sellToken {
                    id
                    symbol
                    decimals
                }
                order {
                    id
                    owner {
                        id
                    }
                }
                gasPrice
                feeAmount
                executedSurplus
            }
        }
        """
        
        all_trades = []
        skip = 0
        
        while skip < limit:
            batch_limit = min(1000, limit - skip)
            
            variables = {
                'startTime': start_ts,
                'endTime': end_ts,
                'skip': skip,
                'limit': batch_limit
            }
            
            data = await self._subgraph_query(query, variables, chain)
            trades = data.get('trades', [])
            
            if not trades:
                break
            
            for trade in trades:
                buy_token = trade.get('buyToken', {})
                sell_token = trade.get('sellToken', {})
                
                # Apply token filter
                if token_filter:
                    if (buy_token.get('symbol', '').upper() != token_filter.upper() and
                        sell_token.get('symbol', '').upper() != token_filter.upper()):
                        continue
                
                buy_decimals = int(buy_token.get('decimals', 18))
                sell_decimals = int(sell_token.get('decimals', 18))
                
                buy_amount = float(trade.get('buyAmount', 0)) / (10 ** buy_decimals)
                sell_amount = float(trade.get('sellAmount', 0)) / (10 ** sell_decimals)
                fee_amount = float(trade.get('feeAmount', 0)) / (10 ** sell_decimals)
                
                # Calculate execution price
                exec_price = sell_amount / buy_amount if buy_amount > 0 else 0
                
                # Surplus calculation
                surplus = float(trade.get('executedSurplus', 0)) / (10 ** buy_decimals)
                surplus_pct = (surplus / buy_amount * 100) if buy_amount > 0 else 0
                
                settlement = trade.get('settlement', {}) or {}
                solver = settlement.get('solver', {}) or {}
                
                all_trades.append({
                    'timestamp': pd.to_datetime(int(trade['timestamp']), unit='s', utc=True),
                    'trade_id': trade.get('id'),
                    'tx_hash': trade.get('txHash'),
                    'settlement_id': settlement.get('id'),
                    'solver_id': solver.get('id'),
                    'solver_address': solver.get('address'),
                    'buy_token_symbol': buy_token.get('symbol'),
                    'buy_token_address': buy_token.get('id'),
                    'sell_token_symbol': sell_token.get('symbol'),
                    'sell_token_address': sell_token.get('id'),
                    'buy_amount': buy_amount,
                    'sell_amount': sell_amount,
                    'execution_price': exec_price,
                    'fee_amount': fee_amount,
                    'surplus_amount': surplus,
                    'surplus_pct': surplus_pct,
                    'gas_price': int(trade.get('gasPrice', 0)),
                    'chain': chain,
                    'venue': self.VENUE,
                    'venue_type': self.VENUE_TYPE,
                    'mev_protected': True,
                })
            
            skip += len(trades)
            logger.debug(f"Fetched {skip} trades")
            
            if len(trades) < batch_limit:
                break
        
        df = pd.DataFrame(all_trades)

        # If subgraph returned no data, fall back to REST API
        if df.empty and self._subgraph_available is False:
            logger.info("Subgraph unavailable, falling back to REST API for trades")
            df = await self.fetch_trades_via_rest_api(chain=chain, limit=limit)
            if not df.empty:
                # Filter by date range if possible (REST API may return recent trades only)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                    # Note: REST API may not have data for historical date range
                return df

        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
            if 'sell_token_symbol' in df.columns and 'buy_token_symbol' in df.columns:
                df['pair'] = df['sell_token_symbol'] + '/' + df['buy_token_symbol']
            df['data_source'] = df.get('data_source', 'subgraph')

        self.collection_stats['records_collected'] += len(df)
        logger.info(f"Fetched {len(df)} CowSwap trades")
        return df

    # =========================================================================
    # Settlement Methods
    # =========================================================================
    
    async def fetch_settlements(
        self,
        start_date: str,
        end_date: str,
        chain: str = 'ethereum',
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch batch settlement data.
        
        Settlements represent batched orders settled together in a single tx.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            chain: Target chain
            limit: Maximum settlements to fetch
            
        Returns:
            DataFrame with settlement data
        """
        logger.info(f"Fetching CowSwap settlements from {chain}")
        
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        
        query = """
        query GetSettlements($startTime: Int!, $endTime: Int!, $skip: Int!, $limit: Int!) {
            settlements(
                first: $limit,
                skip: $skip,
                where: {
                    firstTradeTimestamp_gte: $startTime,
                    firstTradeTimestamp_lte: $endTime
                },
                orderBy: firstTradeTimestamp,
                orderDirection: asc
            ) {
                id
                txHash
                firstTradeTimestamp
                solver {
                    id
                    address
                    numberOfSettlements
                    totalTrades
                }
                trades {
                    id
                    buyAmount
                    sellAmount
                }
                profitability
            }
        }
        """
        
        all_settlements = []
        skip = 0
        
        while skip < limit:
            batch_limit = min(1000, limit - skip)
            
            variables = {
                'startTime': start_ts,
                'endTime': end_ts,
                'skip': skip,
                'limit': batch_limit
            }
            
            data = await self._subgraph_query(query, variables, chain)
            settlements = data.get('settlements', [])
            
            if not settlements:
                break
            
            for settlement in settlements:
                solver = settlement.get('solver', {}) or {}
                trades = settlement.get('trades', [])
                
                # Calculate total volume (rough estimate in native units)
                total_volume = sum(
                    float(t.get('sellAmount', 0)) / 1e18
                    for t in trades
                )
                
                all_settlements.append({
                    'timestamp': pd.to_datetime(
                        int(settlement['firstTradeTimestamp']), unit='s', utc=True
                    ),
                    'settlement_id': settlement.get('id'),
                    'tx_hash': settlement.get('txHash'),
                    'solver_id': solver.get('id'),
                    'solver_address': solver.get('address'),
                    'solver_total_settlements': int(solver.get('numberOfSettlements', 0)),
                    'solver_total_trades': int(solver.get('totalTrades', 0)),
                    'num_trades': len(trades),
                    'total_volume_native': total_volume,
                    'profitability': float(settlement.get('profitability', 0)) / 1e18,
                    'chain': chain,
                    'venue': self.VENUE,
                    'venue_type': self.VENUE_TYPE,
                })
            
            skip += len(settlements)
            
            if len(settlements) < batch_limit:
                break
        
        df = pd.DataFrame(all_settlements)
        
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate derived metrics
            df['avg_trades_per_settlement'] = df['num_trades']
            df['is_profitable'] = df['profitability'] > 0
        
        self.collection_stats['records_collected'] += len(df)
        return df
    
    # =========================================================================
    # Solver Methods
    # =========================================================================
    
    async def fetch_solver_stats(
        self,
        chain: str = 'ethereum',
        limit: int = 50
    ) -> pd.DataFrame:
        """
        Fetch solver competition statistics.
        
        Solvers compete in auctions to settle batches optimally.
        
        Args:
            chain: Target chain
            limit: Maximum solvers to fetch
            
        Returns:
            DataFrame with solver performance metrics
        """
        logger.info(f"Fetching CowSwap solver stats from {chain}")
        
        query = """
        query GetSolvers($limit: Int!) {
            solvers(
                first: $limit,
                orderBy: numberOfSettlements,
                orderDirection: desc
            ) {
                id
                address
                numberOfSettlements
                totalTrades
                solverAddress
            }
        }
        """
        
        data = await self._subgraph_query(query, {'limit': limit}, chain)
        solvers = data.get('solvers', [])
        
        records = []
        total_settlements = sum(int(s.get('numberOfSettlements', 0)) for s in solvers)
        
        for i, solver in enumerate(solvers):
            settlements = int(solver.get('numberOfSettlements', 0))
            trades = int(solver.get('totalTrades', 0))
            
            records.append({
                'timestamp': datetime.utcnow(),
                'solver_id': solver.get('id'),
                'solver_address': solver.get('solverAddress') or solver.get('address'),
                'total_settlements': settlements,
                'total_trades': trades,
                'avg_trades_per_settlement': trades / settlements if settlements > 0 else 0,
                'market_share_pct': (settlements / total_settlements * 100) if total_settlements > 0 else 0,
                'rank': i + 1,
                'chain': chain,
                'venue': self.VENUE,
            })
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            # Add volume-based rankings
            df['settlements_rank'] = df['total_settlements'].rank(ascending=False).astype(int)
            df['trades_rank'] = df['total_trades'].rank(ascending=False).astype(int)
        
        return df
    
    # =========================================================================
    # Token Volume Methods
    # =========================================================================
    
    async def fetch_token_volume(
        self,
        chain: str = 'ethereum',
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch trading volume by token.
        
        Args:
            chain: Target chain
            limit: Maximum tokens to fetch
            
        Returns:
            DataFrame with token volume data
        """
        logger.info(f"Fetching CowSwap token volumes from {chain}")
        
        query = """
        query GetTokenVolume($limit: Int!) {
            tokens(
                first: $limit,
                orderBy: totalVolumeUsd,
                orderDirection: desc
            ) {
                id
                symbol
                name
                decimals
                totalVolumeUsd
                totalTrades
                priceUsd
            }
        }
        """
        
        data = await self._subgraph_query(query, {'limit': limit}, chain)
        tokens = data.get('tokens', [])
        
        records = []
        total_volume = sum(float(t.get('totalVolumeUsd', 0) or 0) for t in tokens)
        
        for token in tokens:
            volume = float(token.get('totalVolumeUsd', 0) or 0)
            trades = int(token.get('totalTrades', 0))
            
            records.append({
                'timestamp': datetime.utcnow(),
                'token_address': token.get('id'),
                'symbol': token.get('symbol'),
                'name': token.get('name'),
                'decimals': int(token.get('decimals', 18)),
                'total_volume_usd': volume,
                'total_trades': trades,
                'price_usd': float(token.get('priceUsd', 0) or 0),
                'avg_trade_size': volume / trades if trades > 0 else 0,
                'volume_share_pct': (volume / total_volume * 100) if total_volume > 0 else 0,
                'chain': chain,
                'venue': self.VENUE,
            })
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            df = df.sort_values('total_volume_usd', ascending=False).reset_index(drop=True)
            df['volume_rank'] = df.index + 1
        
        return df
    
    # =========================================================================
    # Daily Statistics Methods
    # =========================================================================
    
    async def fetch_daily_volume(
        self,
        start_date: str,
        end_date: str,
        chain: str = 'ethereum'
    ) -> pd.DataFrame:
        """
        Fetch daily protocol statistics.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            chain: Target chain
            
        Returns:
            DataFrame with daily volume metrics
        """
        logger.info(f"Fetching CowSwap daily volume from {chain}")
        
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        
        query = """
        query GetDailyVolume($startTime: Int!, $endTime: Int!, $skip: Int!) {
            dailyTotals(
                first: 1000,
                skip: $skip,
                where: {
                    timestamp_gte: $startTime,
                    timestamp_lte: $endTime
                },
                orderBy: timestamp,
                orderDirection: asc
            ) {
                id
                timestamp
                totalVolumeUsd
                numberOfTrades
                tokens
                orders
                settlements
                volumeEth
                volumeUsd
                feesUsd
            }
        }
        """
        
        all_data = []
        skip = 0

        # Bounded iteration to prevent infinite loops
        for _ in range(self.MAX_PAGINATION_ITERATIONS):
            variables = {
                'startTime': start_ts,
                'endTime': end_ts,
                'skip': skip
            }

            data = await self._subgraph_query(query, variables, chain)
            totals = data.get('dailyTotals', [])

            if not totals:
                break

            for total in totals:
                ts = pd.to_datetime(int(total['timestamp']), unit='s', utc=True)
                volume = float(total.get('totalVolumeUsd', 0) or 0)
                trades = int(total.get('numberOfTrades', 0))

                all_data.append({
                    'date': ts.date(),
                    'timestamp': ts,
                    'total_volume_usd': volume,
                    'volume_eth': float(total.get('volumeEth', 0) or 0) / 1e18,
                    'fees_usd': float(total.get('feesUsd', 0) or 0),
                    'num_trades': trades,
                    'num_orders': int(total.get('orders', 0)),
                    'num_settlements': int(total.get('settlements', 0)),
                    'unique_tokens': int(total.get('tokens', 0)),
                    'avg_trade_size': volume / trades if trades > 0 else 0,
                    'chain': chain,
                    'venue': self.VENUE,
                    'venue_type': self.VENUE_TYPE,
                })

            skip += len(totals)

            if len(totals) < 1000:
                break
        
        df = pd.DataFrame(all_data)
        
        if not df.empty:
            df = df.sort_values('date').reset_index(drop=True)
            
            # Calculate rolling metrics
            df['volume_change_1d'] = df['total_volume_usd'].pct_change() * 100
            df['volume_sma_7d'] = df['total_volume_usd'].rolling(7).mean()
            df['trades_sma_7d'] = df['num_trades'].rolling(7).mean()
        
        self.collection_stats['records_collected'] += len(df)
        return df
    
    # =========================================================================
    # Quote Methods
    # =========================================================================
    
    async def fetch_quote(
        self,
        sell_token: str,
        buy_token: str,
        sell_amount: int,
        chain: str = 'ethereum'
    ) -> Dict:
        """
        Get a quote from CowSwap API.
        
        Note: This is for real-time quoting, not historical data.
        
        Args:
            sell_token: Sell token symbol or address
            buy_token: Buy token symbol or address
            sell_amount: Amount to sell in wei
            chain: Target chain
            
        Returns:
            Quote data dict
        """
        # Resolve token addresses
        tokens = self.TOKEN_MAP.get(chain, {})
        sell_addr = tokens.get(sell_token.upper(), sell_token)
        buy_addr = tokens.get(buy_token.upper(), buy_token)
        
        payload = {
            'sellToken': sell_addr,
            'buyToken': buy_addr,
            'sellAmountBeforeFee': str(sell_amount),
            'kind': 'sell',
            'from': '0x0000000000000000000000000000000000000000'
        }
        
        base_url = self.API_URLS.get(chain, self.API_URLS['ethereum'])
        url = f"{base_url}/api/v1/quote"
        
        session = await self._get_session()
        await self.rate_limiter.acquire()
        
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    quote = data.get('quote', {})
                    
                    return {
                        'timestamp': datetime.utcnow(),
                        'sell_token': sell_token,
                        'buy_token': buy_token,
                        'sell_amount': sell_amount,
                        'buy_amount': int(quote.get('buyAmount', 0)),
                        'fee_amount': int(quote.get('feeAmount', 0)),
                        'valid_to': quote.get('validTo'),
                        'execution_price': (
                            sell_amount / int(quote.get('buyAmount', 1))
                            if quote.get('buyAmount') else 0
                        ),
                        'chain': chain,
                        'venue': self.VENUE,
                        'mev_protected': True,
                    }
                else:
                    logger.warning(f"Quote request failed: {resp.status}")
                    return {}
        except Exception as e:
            logger.error(f"Quote request error: {e}")
            return {}
    
    # =========================================================================
    # Comprehensive Data Methods
    # =========================================================================
    
    async def _fetch_single_chain_data(
        self,
        chain: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Helper to fetch data for a single chain (parallelized).

        Args:
            chain: Chain to query
            start_date: Start date
            end_date: End date

        Returns:
            Dict with data for this chain
        """
        try:
            logger.info(f"Fetching comprehensive data from {chain}")

            result = {
                'trades': pd.DataFrame(),
                'settlements': pd.DataFrame(),
                'solvers': pd.DataFrame(),
                'daily_volume': pd.DataFrame(),
                'token_volume': pd.DataFrame(),
            }

            # Trades
            trades = await self.fetch_trades(start_date, end_date, chain, limit=5000)
            if not trades.empty:
                result['trades'] = trades

            # Settlements
            settlements = await self.fetch_settlements(start_date, end_date, chain)
            if not settlements.empty:
                result['settlements'] = settlements

            # Solvers
            solvers = await self.fetch_solver_stats(chain)
            if not solvers.empty:
                result['solvers'] = solvers

            # Daily volume
            daily = await self.fetch_daily_volume(start_date, end_date, chain)
            if not daily.empty:
                result['daily_volume'] = daily

            # Token volume
            tokens = await self.fetch_token_volume(chain)
            if not tokens.empty:
                result['token_volume'] = tokens

            return result

        except Exception as e:
            logger.error(f"Error fetching data for {chain}: {e}")
            return {
                'trades': pd.DataFrame(),
                'settlements': pd.DataFrame(),
                'solvers': pd.DataFrame(),
                'daily_volume': pd.DataFrame(),
                'token_volume': pd.DataFrame(),
            }

    async def fetch_comprehensive_data(
        self,
        start_date: str,
        end_date: str,
        chains: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch comprehensive CowSwap data across chains.

        Args:
            start_date: Start date
            end_date: End date
            chains: Chains to query (default: ethereum only)

        Returns:
            Dict with trades, settlements, solvers, daily_volume DataFrames
        """
        if chains is None:
            chains = ['ethereum']

        # Parallelize chain fetching using asyncio.gather
        tasks = [self._fetch_single_chain_data(chain, start_date, end_date) for chain in chains]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results from all chains
        combined = {
            'trades': [],
            'settlements': [],
            'solvers': [],
            'daily_volume': [],
            'token_volume': [],
        }

        for result in results:
            if isinstance(result, dict):
                for key in combined.keys():
                    df = result.get(key, pd.DataFrame())
                    if not df.empty:
                        combined[key].append(df)

        # Combine results
        return {
            key: pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            for key, dfs in combined.items()
        }
    
    # =========================================================================
    # Standardized Collection Methods
    # =========================================================================

    async def collect_swaps(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect swap data - wraps fetch_trades().

        Standardized method name for collection manager compatibility.
        """
        try:
            # Convert dates to string format (YYYY-MM-DD) as fetch_trades expects strings
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            # CowSwap fetch_trades doesn't accept 'symbols', only 'token_filter' (single token)
            token_filter = symbols[0] if symbols else None
            return await self.fetch_trades(
                start_date=start_str,
                end_date=end_str,
                token_filter=token_filter
            )

        except Exception as e:
            logger.error(f"CowSwap collect_swaps error: {e}")
            return pd.DataFrame()

    async def collect_orders(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect orders data - wraps fetch_trades().

        CowSwap uses order-based model so this is same as collect_swaps.
        Standardized method name for collection manager compatibility.
        """
        try:
            return await self.collect_swaps(symbols, start_date, end_date, **kwargs)

        except Exception as e:
            logger.error(f"CowSwap collect_orders error: {e}")
            return pd.DataFrame()

    async def collect_trades(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect trades data - wraps fetch_trades().

        Standardized method name for collection manager compatibility.
        """
        try:
            return await self.collect_swaps(symbols, start_date, end_date, **kwargs)

        except Exception as e:
            logger.error(f"CowSwap collect_trades error: {e}")
            return pd.DataFrame()

    async def collect_funding_rates(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect funding rates - wraps fetch_funding_rates().

        Standardized method name for collection manager compatibility.
        """
        try:
            # Convert dates to string format if needed
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            return await self.fetch_funding_rates(
                symbols=symbols,
                start_date=start_str,
                end_date=end_str
            )
        except Exception as e:
            logger.error(f"CowSwap collect_funding_rates error: {e}")
            return pd.DataFrame()

    async def collect_ohlcv(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect OHLCV data - wraps fetch_ohlcv().

        Standardized method name for collection manager compatibility.
        """
        try:
            timeframe = kwargs.get('timeframe', '1h')

            # Convert dates to string format if needed
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            return await self.fetch_ohlcv(
                symbols=symbols,
                timeframe=timeframe,
                start_date=start_str,
                end_date=end_str
            )
        except Exception as e:
            logger.error(f"CowSwap collect_ohlcv error: {e}")
            return pd.DataFrame()

    # =========================================================================
    # Helper Methods
    # =========================================================================

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
        """CowSwap is a spot DEX - no funding rates."""
        logger.debug("CowSwap: No funding rates available (spot DEX)")
        return pd.DataFrame()
    
    async def fetch_ohlcv(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch OHLCV by aggregating trade data.
        
        Note: Aggregated from individual trades, may be sparse.
        """
        trades = await self.fetch_trades(start_date, end_date, limit=50000)

        if trades.empty:
            return pd.DataFrame()

        # Check if required columns exist
        if 'buy_token_symbol' not in trades.columns or 'sell_token_symbol' not in trades.columns:
            logger.warning("CowSwap trades missing token symbol columns - cannot aggregate to OHLCV")
            return pd.DataFrame()

        # Map timeframe to pandas frequency
        freq_map = {'1h': 'H', '4h': '4H', '1d': 'D'}
        freq = freq_map.get(timeframe, 'H')

        # Filter by symbols
        if symbols:
            symbol_upper = [s.upper() for s in symbols]
            trades = trades[
                (trades['buy_token_symbol'].isin(symbol_upper)) |
                (trades['sell_token_symbol'].isin(symbol_upper))
            ]
        
        if trades.empty:
            return pd.DataFrame()
        
        # Aggregate by time period
        trades['period'] = trades['timestamp'].dt.floor(freq)
        
        ohlcv = trades.groupby(['period', 'buy_token_symbol']).agg({
            'execution_price': ['first', 'max', 'min', 'last'],
            'buy_amount': 'sum',
            'sell_amount': 'sum',
            'trade_id': 'count'
        }).reset_index()
        
        ohlcv.columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close',
                        'volume_buy', 'volume_sell', 'trades']
        
        ohlcv['volume'] = ohlcv['volume_sell']
        ohlcv['venue'] = self.VENUE
        ohlcv['venue_type'] = self.VENUE_TYPE
        
        return ohlcv.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
    
    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.debug("CowSwap session closed")

# =============================================================================
# Test Function
# =============================================================================

async def test_cowswap_collector():
    """Test CowSwap collector functionality."""
    config = {'rate_limit': 20}

    async with CowSwapCollector(config) as collector:
        print("=" * 60)
        print("CowSwap Collector Test")
        print("=" * 60)

        # Check API key status
        if collector.graph_api_key:
            print(f"\n Graph API key configured (decentralized network)")
        else:
            print(f"\n No Graph API key - using legacy endpoints + REST API fallback")

        # Test REST API trades (always works, no API key needed)
        print("\n1. Testing REST API trades...")
        rest_trades = await collector.fetch_trades_via_rest_api(chain='ethereum', limit=50)
        print(f" REST API trades: {len(rest_trades)}")
        if not rest_trades.empty:
            print(f" Latest trade: {rest_trades.iloc[0]['tx_hash'][:20]}...")

        # Test subgraph trades (may require API key)
        end = datetime.utcnow()
        start = end - timedelta(days=7)

        print("\n2. Testing subgraph trades...")
        trades = await collector.fetch_trades(
            start.strftime('%Y-%m-%d'),
            end.strftime('%Y-%m-%d'),
            limit=100
        )
        print(f" Subgraph trades: {len(trades)}")
        if not trades.empty:
            print(f" Data source: {trades['data_source'].iloc[0] if 'data_source' in trades.columns else 'subgraph'}")

        # Test collect_trades (standardized method)
        print("\n3. Testing collect_trades (standardized method)...")
        collected = await collector.collect_trades(
            symbols=['ETH'],
            start_date=start,
            end_date=end
        )
        print(f" Collected trades: {len(collected)}")

        # Test solver stats
        print("\n4. Testing solver stats...")
        solvers = await collector.fetch_solver_stats(limit=10)
        print(f" Top solvers: {len(solvers)}")
        if not solvers.empty:
            top = solvers.iloc[0]
            solver_id = top['solver_id'] if top['solver_id'] else 'Unknown'
            print(f" #1: {solver_id[:20]}... - {top['total_settlements']} settlements")

        # Test token volumes
        print("\n5. Testing token volumes...")
        tokens = await collector.fetch_token_volume(limit=20)
        print(f" Tokens: {len(tokens)}")
        if not tokens.empty:
            print(f" Top: {tokens.iloc[0]['symbol']} - ${tokens.iloc[0]['total_volume_usd']:,.0f}")

        # Collection stats
        print("\n" + "=" * 60)
        stats = collector.get_collection_stats()
        print(f"Collection Stats:")
        print(f" Records: {stats['records_collected']}")
        print(f" API Calls: {stats['api_calls']}")
        print(f" Subgraph Calls: {stats['subgraph_calls']}")
        print(f" REST API Calls: {stats['rest_api_calls']}")
        print(f" Errors: {stats['errors']}")
        print("=" * 60)

if __name__ == '__main__':
    asyncio.run(test_cowswap_collector())