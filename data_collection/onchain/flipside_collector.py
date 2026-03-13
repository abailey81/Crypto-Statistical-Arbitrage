"""
Flipside Crypto SQL-Based Blockchain Data Collector

validated SQL interface for on-chain blockchain data analysis.
Execute custom SQL queries against indexed multi-chain blockchain data.

===============================================================================
FLIPSIDE CRYPTO OVERVIEW
===============================================================================

Flipside Crypto provides SQL-based access to indexed blockchain data across
multiple chains. Unlike REST APIs, Flipside allows custom analytical queries
enabling comprehensive on-chain analysis.

Key Features:
    - SQL interface to blockchain data (Snowflake backend)
    - Pre-indexed tables for DEX swaps, transfers, lending, NFTs
    - Multi-chain support (Ethereum, Arbitrum, Polygon, Solana, etc.)
    - Historical data since chain genesis
    - Real-time data with ~5 minute delay

===============================================================================
API SPECIFICATIONS
===============================================================================

Base URL: https://api-v2.flipsidecrypto.xyz

Authentication:
    - API Key in x-api-key header
    - Keys obtained from Flipside dashboard

Rate Limits by Tier:
    ============ ============== ================ ===============
    Tier Queries/day Concurrent Result Size
    ============ ============== ================ ===============
    Free 100 1 1M rows
    Developer 1,000 3 10M rows
    Team 10,000 10 100M rows
    Enterprise Unlimited Custom Unlimited
    ============ ============== ================ ===============

Query Execution:
    1. Submit query via createQueryRun
    2. Poll getQueryRunResults for status
    3. Retrieve results when QUERY_STATE_SUCCESS

===============================================================================
SUPPORTED CHAINS
===============================================================================

    Chain Schema Prefix Coverage
    --------------- ---------------- ---------------
    Ethereum ethereum 2015+
    Arbitrum arbitrum 2021+
    Avalanche avalanche_c 2020+
    BSC bsc 2020+
    Gnosis gnosis 2020+
    Optimism optimism 2021+
    Polygon polygon 2020+
    Solana solana 2020+
    NEAR near 2020+
    Flow flow 2020+
    Base base 2023+

===============================================================================
DATA TABLES
===============================================================================

DeFi Tables:
    - {chain}.defi.ez_dex_swaps: DEX swap transactions
    - {chain}.defi.ez_lending_deposits: Lending deposits
    - {chain}.defi.ez_lending_borrows: Lending borrows
    - {chain}.defi.ez_lp_actions: Liquidity pool actions
    - {chain}.defi.ez_bridge_activity: Cross-chain bridges

Core Tables:
    - {chain}.core.ez_token_transfers: Token transfers
    - {chain}.core.ez_native_transfers: Native token transfers
    - {chain}.core.fact_transactions: All transactions
    - {chain}.core.fact_blocks: Block data

Price Tables:
    - {chain}.price.ez_prices_hourly: Hourly token prices

NFT Tables:
    - {chain}.nft.ez_nft_sales: NFT sales
    - {chain}.nft.ez_nft_mints: NFT mints

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

DEX Analytics:
    - Cross-DEX price comparison
    - Liquidity depth analysis
    - Slippage estimation
    - MEV detection

Flow Analysis:
    - Whale transaction tracking
    - Smart money following
    - Token accumulation patterns
    - Exchange deposit/withdrawal flows

Protocol Analytics:
    - TVL changes
    - Utilization rates
    - Liquidation risk assessment

===============================================================================
USAGE EXAMPLES
===============================================================================

DEX swap analysis:

    >>> collector = FlipsideCollector({'api_key': 'key'})
    >>> swaps = await collector.fetch_dex_swaps(
    ... chain='ethereum',
    ... start_date='2024-01-01',
    ... platform='uniswap'
    ... )

Custom SQL query:

    >>> custom_sql = '''
    ... SELECT DATE_TRUNC('day', block_timestamp) as date,
    ... SUM(amount_in_usd) as volume
    ... FROM ethereum.defi.ez_dex_swaps
    ... WHERE platform = 'uniswap-v3'
    ... GROUP BY 1 ORDER BY 1
    ... '''
    >>> results = await collector.execute_query(custom_sql)

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

# =============================================================================
# ENUMS
# =============================================================================

class Chain(Enum):
    """Supported blockchain networks."""
    ETHEREUM = 'ethereum'
    ARBITRUM = 'arbitrum'
    AVALANCHE = 'avalanche_c'
    BSC = 'bsc'
    GNOSIS = 'gnosis'
    OPTIMISM = 'optimism'
    POLYGON = 'polygon'
    SOLANA = 'solana'
    NEAR = 'near'
    FLOW = 'flow'
    BASE = 'base'
    
    @classmethod
    def from_string(cls, value: str) -> 'Chain':
        """Convert string to Chain enum."""
        mapping = {
            'ethereum': cls.ETHEREUM, 'eth': cls.ETHEREUM,
            'arbitrum': cls.ARBITRUM, 'arb': cls.ARBITRUM,
            'avalanche': cls.AVALANCHE, 'avax': cls.AVALANCHE,
            'bsc': cls.BSC, 'bnb': cls.BSC,
            'gnosis': cls.GNOSIS, 'xdai': cls.GNOSIS,
            'optimism': cls.OPTIMISM, 'op': cls.OPTIMISM,
            'polygon': cls.POLYGON, 'matic': cls.POLYGON,
            'solana': cls.SOLANA, 'sol': cls.SOLANA,
            'near': cls.NEAR,
            'flow': cls.FLOW,
            'base': cls.BASE,
        }
        return mapping.get(value.lower(), cls.ETHEREUM)

class DEXPlatform(Enum):
    """Supported DEX platforms."""
    UNISWAP_V2 = 'uniswap-v2'
    UNISWAP_V3 = 'uniswap-v3'
    SUSHISWAP = 'sushiswap'
    CURVE = 'curve'
    BALANCER = 'balancer'
    PANCAKESWAP = 'pancakeswap'
    QUICKSWAP = 'quickswap'
    TRADER_JOE = 'trader-joe'
    CAMELOT = 'camelot'
    GMX = 'gmx'

class LendingPlatform(Enum):
    """Supported lending platforms."""
    AAVE_V2 = 'aave-v2'
    AAVE_V3 = 'aave-v3'
    COMPOUND_V2 = 'compound-v2'
    COMPOUND_V3 = 'compound-v3'
    MAKER = 'maker'
    SPARK = 'spark'
    MORPHO = 'morpho'

class QueryState(Enum):
    """Flipside query execution states."""
    PENDING = 'QUERY_STATE_PENDING'
    RUNNING = 'QUERY_STATE_RUNNING'
    SUCCESS = 'QUERY_STATE_SUCCESS'
    FAILED = 'QUERY_STATE_FAILED'
    CANCELLED = 'QUERY_STATE_CANCELLED'

class ActionType(Enum):
    """DeFi action types."""
    SWAP = 'swap'
    DEPOSIT = 'deposit'
    WITHDRAW = 'withdraw'
    BORROW = 'borrow'
    REPAY = 'repay'
    LIQUIDATE = 'liquidate'
    ADD_LIQUIDITY = 'add_liquidity'
    REMOVE_LIQUIDITY = 'remove_liquidity'
    BRIDGE = 'bridge'

class TransferSize(Enum):
    """Transfer size classification."""
    MICRO = 'micro' # < $1K
    SMALL = 'small' # $1K - $10K
    MEDIUM = 'medium' # $10K - $100K
    LARGE = 'large' # $100K - $1M
    WHALE = 'whale' # $1M - $10M
    MEGA_WHALE = 'mega_whale' # > $10M

class VolumeCategory(Enum):
    """Volume category classification."""
    VERY_LOW = 'very_low' # < $10K
    LOW = 'low' # $10K - $100K
    MEDIUM = 'medium' # $100K - $1M
    HIGH = 'high' # $1M - $10M
    VERY_HIGH = 'very_high' # > $10M

class TradeDirection(Enum):
    """Trade direction."""
    BUY = 'buy'
    SELL = 'sell'
    UNKNOWN = 'unknown'

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class DEXSwap:
    """DEX swap transaction with analytics."""
    timestamp: datetime
    tx_hash: str
    chain: str
    platform: str
    sender: str
    token_in_symbol: str
    token_in_amount: float
    token_out_symbol: str
    token_out_amount: float
    amount_in_usd: float
    amount_out_usd: float
    
    @property
    def pair(self) -> str:
        """Trading pair."""
        return f"{self.token_in_symbol}/{self.token_out_symbol}"
    
    @property
    def avg_usd_value(self) -> float:
        """Average USD value of swap."""
        return (self.amount_in_usd + self.amount_out_usd) / 2
    
    @property
    def size_category(self) -> TransferSize:
        """Classify swap size."""
        usd = self.avg_usd_value
        if usd < 1_000:
            return TransferSize.MICRO
        elif usd < 10_000:
            return TransferSize.SMALL
        elif usd < 100_000:
            return TransferSize.MEDIUM
        elif usd < 1_000_000:
            return TransferSize.LARGE
        elif usd < 10_000_000:
            return TransferSize.WHALE
        return TransferSize.MEGA_WHALE
    
    @property
    def is_whale_trade(self) -> bool:
        """Check if whale-sized trade."""
        return self.avg_usd_value >= 1_000_000
    
    @property
    def is_stablecoin_swap(self) -> bool:
        """Check if stablecoin swap."""
        stables = {'USDC', 'USDT', 'DAI', 'FRAX', 'BUSD', 'TUSD', 'USDP', 'GUSD'}
        return (self.token_in_symbol.upper() in stables and
                self.token_out_symbol.upper() in stables)
    
    @property
    def implied_price(self) -> Optional[float]:
        """Implied price from swap."""
        if self.token_out_amount > 0:
            return self.token_in_amount / self.token_out_amount
        return None
    
    @property
    def slippage_estimate(self) -> float:
        """Estimated slippage percentage."""
        if self.amount_in_usd > 0:
            return abs(self.amount_out_usd - self.amount_in_usd) / self.amount_in_usd * 100
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'tx_hash': self.tx_hash, 'chain': self.chain, 'platform': self.platform,
            'sender': self.sender, 'pair': self.pair,
            'token_in_symbol': self.token_in_symbol, 'token_in_amount': self.token_in_amount,
            'token_out_symbol': self.token_out_symbol, 'token_out_amount': self.token_out_amount,
            'amount_in_usd': self.amount_in_usd, 'amount_out_usd': self.amount_out_usd,
            'size_category': self.size_category.value, 'is_whale_trade': self.is_whale_trade,
            'slippage_estimate': self.slippage_estimate,
        }

@dataclass
class TokenTransfer:
    """Token transfer with analytics."""
    timestamp: datetime
    tx_hash: str
    chain: str
    token_symbol: str
    token_address: str
    from_address: str
    to_address: str
    amount: float
    amount_usd: float
    
    @property
    def size_category(self) -> TransferSize:
        """Classify transfer size."""
        if self.amount_usd < 1_000:
            return TransferSize.MICRO
        elif self.amount_usd < 10_000:
            return TransferSize.SMALL
        elif self.amount_usd < 100_000:
            return TransferSize.MEDIUM
        elif self.amount_usd < 1_000_000:
            return TransferSize.LARGE
        elif self.amount_usd < 10_000_000:
            return TransferSize.WHALE
        return TransferSize.MEGA_WHALE
    
    @property
    def is_whale_transfer(self) -> bool:
        """Check if whale-sized transfer."""
        return self.amount_usd >= 1_000_000
    
    @property
    def is_significant(self) -> bool:
        """Check if transfer is significant."""
        return self.amount_usd >= 100_000
    
    @property
    def is_stablecoin(self) -> bool:
        """Check if stablecoin transfer."""
        stables = {'USDC', 'USDT', 'DAI', 'FRAX', 'BUSD'}
        return self.token_symbol.upper() in stables
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'tx_hash': self.tx_hash, 'chain': self.chain,
            'token_symbol': self.token_symbol, 'from_address': self.from_address,
            'to_address': self.to_address, 'amount': self.amount, 'amount_usd': self.amount_usd,
            'size_category': self.size_category.value, 'is_whale_transfer': self.is_whale_transfer,
        }

@dataclass
class LendingAction:
    """Lending protocol action with analytics."""
    timestamp: datetime
    tx_hash: str
    chain: str
    platform: str
    action_type: str
    user_address: str
    token_symbol: str
    token_amount: float
    amount_usd: float
    
    @property
    def action_enum(self) -> ActionType:
        """Get action type enum."""
        mapping = {
            'deposit': ActionType.DEPOSIT, 'supply': ActionType.DEPOSIT,
            'withdraw': ActionType.WITHDRAW,
            'borrow': ActionType.BORROW,
            'repay': ActionType.REPAY,
            'liquidate': ActionType.LIQUIDATE,
        }
        return mapping.get(self.action_type.lower(), ActionType.DEPOSIT)
    
    @property
    def is_deposit(self) -> bool:
        """Check if deposit action."""
        return self.action_type.lower() in ['deposit', 'supply']
    
    @property
    def is_withdrawal(self) -> bool:
        """Check if withdrawal action."""
        return self.action_type.lower() == 'withdraw'
    
    @property
    def is_borrow(self) -> bool:
        """Check if borrow action."""
        return self.action_type.lower() == 'borrow'
    
    @property
    def is_significant(self) -> bool:
        """Check if action is significant."""
        return self.amount_usd >= 100_000
    
    @property
    def size_category(self) -> TransferSize:
        """Classify action size."""
        if self.amount_usd < 1_000:
            return TransferSize.MICRO
        elif self.amount_usd < 10_000:
            return TransferSize.SMALL
        elif self.amount_usd < 100_000:
            return TransferSize.MEDIUM
        elif self.amount_usd < 1_000_000:
            return TransferSize.LARGE
        return TransferSize.WHALE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'tx_hash': self.tx_hash, 'chain': self.chain, 'platform': self.platform,
            'action_type': self.action_type, 'user_address': self.user_address,
            'token_symbol': self.token_symbol, 'amount_usd': self.amount_usd,
            'size_category': self.size_category.value, 'is_significant': self.is_significant,
        }

@dataclass
class LiquidityAction:
    """Liquidity pool action with analytics."""
    timestamp: datetime
    tx_hash: str
    chain: str
    platform: str
    pool_address: str
    pool_name: str
    action: str
    amount0: float
    amount1: float
    amount0_usd: float
    amount1_usd: float
    lp_token_amount: float
    provider: str
    
    @property
    def total_usd(self) -> float:
        """Total USD value of action."""
        return self.amount0_usd + self.amount1_usd
    
    @property
    def is_add(self) -> bool:
        """Check if adding liquidity."""
        return self.action.lower() in ['add', 'mint', 'increase']
    
    @property
    def is_remove(self) -> bool:
        """Check if removing liquidity."""
        return self.action.lower() in ['remove', 'burn', 'decrease']
    
    @property
    def size_category(self) -> TransferSize:
        """Classify action size."""
        usd = self.total_usd
        if usd < 1_000:
            return TransferSize.MICRO
        elif usd < 10_000:
            return TransferSize.SMALL
        elif usd < 100_000:
            return TransferSize.MEDIUM
        elif usd < 1_000_000:
            return TransferSize.LARGE
        return TransferSize.WHALE
    
    @property
    def is_whale_action(self) -> bool:
        """Check if whale-sized action."""
        return self.total_usd >= 1_000_000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'tx_hash': self.tx_hash, 'chain': self.chain, 'platform': self.platform,
            'pool_name': self.pool_name, 'action': self.action,
            'total_usd': self.total_usd, 'size_category': self.size_category.value,
            'is_add': self.is_add, 'is_remove': self.is_remove,
        }

@dataclass
class BridgeTransaction:
    """Cross-chain bridge transaction."""
    timestamp: datetime
    tx_hash: str
    source_chain: str
    destination_chain: str
    platform: str
    sender: str
    receiver: str
    token_symbol: str
    amount: float
    amount_usd: float
    
    @property
    def bridge_direction(self) -> str:
        """Bridge direction description."""
        return f"{self.source_chain} -> {self.destination_chain}"
    
    @property
    def is_to_l2(self) -> bool:
        """Check if bridging to L2."""
        l2s = {'arbitrum', 'optimism', 'polygon', 'base'}
        return self.destination_chain.lower() in l2s
    
    @property
    def is_from_l2(self) -> bool:
        """Check if bridging from L2."""
        l2s = {'arbitrum', 'optimism', 'polygon', 'base'}
        return self.source_chain.lower() in l2s
    
    @property
    def size_category(self) -> TransferSize:
        """Classify bridge size."""
        if self.amount_usd < 1_000:
            return TransferSize.MICRO
        elif self.amount_usd < 10_000:
            return TransferSize.SMALL
        elif self.amount_usd < 100_000:
            return TransferSize.MEDIUM
        elif self.amount_usd < 1_000_000:
            return TransferSize.LARGE
        return TransferSize.WHALE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'tx_hash': self.tx_hash, 'platform': self.platform,
            'bridge_direction': self.bridge_direction, 'token_symbol': self.token_symbol,
            'amount_usd': self.amount_usd, 'size_category': self.size_category.value,
            'is_to_l2': self.is_to_l2, 'is_from_l2': self.is_from_l2,
        }

@dataclass
class QueryResult:
    """Flipside query result metadata."""
    query_run_id: str
    state: str
    row_count: int
    execution_time_ms: int
    credits_used: float
    
    @property
    def state_enum(self) -> QueryState:
        """Get query state enum."""
        return QueryState(self.state)
    
    @property
    def is_success(self) -> bool:
        """Check if query succeeded."""
        return self.state == QueryState.SUCCESS.value
    
    @property
    def is_failed(self) -> bool:
        """Check if query failed."""
        return self.state == QueryState.FAILED.value
    
    @property
    def execution_time_seconds(self) -> float:
        """Execution time in seconds."""
        return self.execution_time_ms / 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query_run_id': self.query_run_id, 'state': self.state,
            'row_count': self.row_count, 'execution_time_seconds': self.execution_time_seconds,
            'credits_used': self.credits_used, 'is_success': self.is_success,
        }

# =============================================================================
# QUERY TEMPLATES
# =============================================================================

QUERY_TEMPLATES = {
    'dex_swaps': """
        SELECT 
            block_timestamp,
            tx_hash,
            platform,
            token_in_symbol,
            token_in_amount,
            token_out_symbol,
            token_out_amount,
            amount_in_usd,
            amount_out_usd,
            sender
        FROM {chain}.defi.ez_dex_swaps
        WHERE block_timestamp >= '{start_date}'
        AND block_timestamp < '{end_date}'
        {filters}
        ORDER BY block_timestamp DESC
        LIMIT {limit}
    """,
    
    'token_transfers': """
        SELECT 
            block_timestamp,
            tx_hash,
            contract_address,
            symbol,
            from_address,
            to_address,
            amount,
            amount_usd
        FROM {chain}.core.ez_token_transfers
        WHERE block_timestamp >= '{start_date}'
        AND block_timestamp < '{end_date}'
        {filters}
        ORDER BY block_timestamp DESC
        LIMIT {limit}
    """,
    
    'lending_deposits': """
        SELECT 
            block_timestamp,
            tx_hash,
            platform,
            depositor_address,
            token_symbol,
            token_amount,
            amount_usd,
            'deposit' as action_type
        FROM {chain}.defi.ez_lending_deposits
        WHERE block_timestamp >= '{start_date}'
        AND block_timestamp < '{end_date}'
        {filters}
        ORDER BY block_timestamp DESC
        LIMIT {limit}
    """,
    
    'lending_borrows': """
        SELECT 
            block_timestamp,
            tx_hash,
            platform,
            depositor_address,
            token_symbol,
            token_amount,
            amount_usd,
            'borrow' as action_type
        FROM {chain}.defi.ez_lending_borrows
        WHERE block_timestamp >= '{start_date}'
        AND block_timestamp < '{end_date}'
        {filters}
        ORDER BY block_timestamp DESC
        LIMIT {limit}
    """,
    
    'liquidity_actions': """
        SELECT 
            block_timestamp,
            tx_hash,
            platform,
            pool_address,
            pool_name,
            action,
            amount0,
            amount1,
            amount0_usd,
            amount1_usd,
            lp_token_amount,
            liquidity_provider
        FROM {chain}.defi.ez_lp_actions
        WHERE block_timestamp >= '{start_date}'
        AND block_timestamp < '{end_date}'
        {filters}
        ORDER BY block_timestamp DESC
        LIMIT {limit}
    """,
    
    'bridge_activity': """
        SELECT 
            block_timestamp,
            tx_hash,
            platform,
            sender,
            receiver,
            destination_chain,
            token_symbol,
            amount,
            amount_usd
        FROM {chain}.defi.ez_bridge_activity
        WHERE block_timestamp >= '{start_date}'
        AND block_timestamp < '{end_date}'
        {filters}
        ORDER BY block_timestamp DESC
        LIMIT {limit}
    """,
    
    'whale_transfers': """
        SELECT 
            block_timestamp,
            tx_hash,
            symbol,
            from_address,
            to_address,
            amount,
            amount_usd
        FROM {chain}.core.ez_token_transfers
        WHERE block_timestamp >= '{start_date}'
        AND block_timestamp < '{end_date}'
        AND amount_usd >= {min_usd}
        {filters}
        ORDER BY amount_usd DESC
        LIMIT {limit}
    """,
}

# =============================================================================
# COLLECTOR CLASS
# =============================================================================

class FlipsideCollector:
    """
    Flipside Crypto SQL-based blockchain data collector.
    
    Features:
    - Execute custom SQL queries against blockchain data
    - Pre-built queries for common DeFi analytics
    - Multi-chain support
    - DEX trades, lending, liquidity, bridges
    """
    
    VENUE = 'flipside'
    VENUE_TYPE = 'blockchain_sql'
    BASE_URL = 'https://api-v2.flipsidecrypto.xyz'
    
    SUPPORTED_CHAINS = {c.name.lower(): c.value for c in Chain}
    
    def __init__(self, config: Dict):
        """Initialize Flipside collector."""
        self.api_key = config.get('api_key', config.get('flipside_api_key', ''))
        self.query_timeout = config.get('query_timeout', 300)
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.collection_stats = {'queries': 0, 'records': 0, 'errors': 0, 'credits_used': 0.0}
    
    async def __aenter__(self) -> 'FlipsideCollector':
        """Async context manager entry."""
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.query_timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    def _get_chain_schema(self, chain: str) -> str:
        """Get chain schema prefix."""
        chain_enum = Chain.from_string(chain)
        return chain_enum.value
    
    async def execute_query(
        self,
        sql: str,
        ttl_minutes: int = 60,
        max_age_minutes: Optional[int] = None
    ) -> pd.DataFrame:
        """Execute a SQL query against Flipside data."""
        session = await self._get_session()
        headers = {'Content-Type': 'application/json', 'x-api-key': self.api_key}
        
        create_payload = {
            "jsonrpc": "2.0",
            "method": "createQueryRun",
            "params": [{
                "resultTTLHours": ttl_minutes // 60,
                "maxAgeMinutes": max_age_minutes or 0,
                "sql": sql,
                "tags": {"source": "crypto-statarb"},
                "dataSource": "snowflake-default",
                "dataProvider": "flipside"
            }],
            "id": 1
        }
        
        try:
            async with session.post(f"{self.BASE_URL}/json-rpc", headers=headers, json=create_payload) as response:
                if response.status != 200:
                    self.collection_stats['errors'] += 1
                    return pd.DataFrame()
                
                result = await response.json()
                if 'error' in result:
                    self.logger.error(f"Query error: {result['error']}")
                    self.collection_stats['errors'] += 1
                    return pd.DataFrame()
                
                query_run_id = result.get('result', {}).get('queryRun', {}).get('id')
                if not query_run_id:
                    return pd.DataFrame()
            
            self.collection_stats['queries'] += 1
            return await self._poll_query_results(query_run_id, headers)
            
        except Exception as e:
            self.logger.error(f"Query execution error: {e}")
            self.collection_stats['errors'] += 1
            return pd.DataFrame()
    
    async def _poll_query_results(
        self, query_run_id: str, headers: Dict,
        max_attempts: int = 60, poll_interval: float = 10.0
    ) -> pd.DataFrame:
        """Poll for query results until complete."""
        session = await self._get_session()
        
        for attempt in range(max_attempts):
            status_payload = {
                "jsonrpc": "2.0",
                "method": "getQueryRunResults",
                "params": [{"queryRunId": query_run_id, "format": "json", "page": {"number": 1, "size": 10000}}],
                "id": 1
            }
            
            try:
                async with session.post(f"{self.BASE_URL}/json-rpc", headers=headers, json=status_payload) as response:
                    if response.status != 200:
                        await asyncio.sleep(poll_interval)
                        continue
                    
                    result = await response.json()
                    query_run = result.get('result', {}).get('queryRun', {})
                    state = query_run.get('state')
                    
                    if state == QueryState.SUCCESS.value:
                        rows = result.get('result', {}).get('rows', [])
                        columns = result.get('result', {}).get('columnNames', [])
                        
                        if rows and columns:
                            df = pd.DataFrame(rows, columns=columns)
                            df['venue'] = self.VENUE
                            self.collection_stats['records'] += len(df)
                            return df
                        return pd.DataFrame()
                    
                    elif state == QueryState.FAILED.value:
                        self.logger.error(f"Query failed: {query_run.get('errorMessage')}")
                        self.collection_stats['errors'] += 1
                        return pd.DataFrame()
                    
                    await asyncio.sleep(poll_interval)
                    
            except Exception as e:
                self.logger.error(f"Polling error: {e}")
                await asyncio.sleep(poll_interval)
        
        self.logger.error(f"Query timed out after {max_attempts} attempts")
        return pd.DataFrame()
    
    async def fetch_dex_swaps(
        self, chain: str = 'ethereum', start_date: Optional[str] = None,
        end_date: Optional[str] = None, platform: Optional[str] = None,
        token_symbol: Optional[str] = None, limit: int = 10000
    ) -> pd.DataFrame:
        """Fetch DEX swap transactions."""
        filters = []
        if platform:
            filters.append(f"AND platform ILIKE '%{platform}%'")
        if token_symbol:
            filters.append(f"AND (token_in_symbol = '{token_symbol}' OR token_out_symbol = '{token_symbol}')")
        
        sql = QUERY_TEMPLATES['dex_swaps'].format(
            chain=self._get_chain_schema(chain),
            start_date=start_date or '2024-01-01',
            end_date=end_date or datetime.now().strftime('%Y-%m-%d'),
            filters=' '.join(filters),
            limit=limit
        )
        
        df = await self.execute_query(sql)
        if not df.empty and 'amount_in_usd' in df.columns:
            df['avg_usd_value'] = (df['amount_in_usd'].fillna(0) + df['amount_out_usd'].fillna(0)) / 2
            df['is_whale_trade'] = df['avg_usd_value'] >= 1_000_000
            df['chain'] = chain
        return df
    
    async def fetch_token_transfers(
        self, chain: str = 'ethereum', start_date: Optional[str] = None,
        end_date: Optional[str] = None, token_symbol: Optional[str] = None,
        min_amount_usd: Optional[float] = None, limit: int = 10000
    ) -> pd.DataFrame:
        """Fetch token transfer events."""
        filters = []
        if token_symbol:
            filters.append(f"AND symbol = '{token_symbol}'")
        if min_amount_usd:
            filters.append(f"AND amount_usd >= {min_amount_usd}")
        
        sql = QUERY_TEMPLATES['token_transfers'].format(
            chain=self._get_chain_schema(chain),
            start_date=start_date or '2024-01-01',
            end_date=end_date or datetime.now().strftime('%Y-%m-%d'),
            filters=' '.join(filters),
            limit=limit
        )
        
        df = await self.execute_query(sql)
        if not df.empty and 'amount_usd' in df.columns:
            df['is_whale_transfer'] = df['amount_usd'] >= 1_000_000
            df['chain'] = chain
        return df
    
    async def fetch_lending_activity(
        self, chain: str = 'ethereum', start_date: Optional[str] = None,
        end_date: Optional[str] = None, platform: Optional[str] = None,
        limit: int = 10000
    ) -> pd.DataFrame:
        """Fetch lending protocol activity (deposits + borrows)."""
        filters = []
        if platform:
            filters.append(f"AND platform ILIKE '%{platform}%'")
        filter_str = ' '.join(filters)
        
        deposits_sql = QUERY_TEMPLATES['lending_deposits'].format(
            chain=self._get_chain_schema(chain),
            start_date=start_date or '2024-01-01',
            end_date=end_date or datetime.now().strftime('%Y-%m-%d'),
            filters=filter_str, limit=limit
        )
        
        borrows_sql = QUERY_TEMPLATES['lending_borrows'].format(
            chain=self._get_chain_schema(chain),
            start_date=start_date or '2024-01-01',
            end_date=end_date or datetime.now().strftime('%Y-%m-%d'),
            filters=filter_str, limit=limit
        )
        
        deposits_df = await self.execute_query(deposits_sql)
        borrows_df = await self.execute_query(borrows_sql)
        
        if deposits_df.empty and borrows_df.empty:
            return pd.DataFrame()
        
        result = pd.concat([deposits_df, borrows_df], ignore_index=True)
        result['chain'] = chain
        return result
    
    async def fetch_liquidity_actions(
        self, chain: str = 'ethereum', start_date: Optional[str] = None,
        end_date: Optional[str] = None, platform: Optional[str] = None,
        limit: int = 10000
    ) -> pd.DataFrame:
        """Fetch liquidity pool add/remove events."""
        filters = []
        if platform:
            filters.append(f"AND platform ILIKE '%{platform}%'")
        
        sql = QUERY_TEMPLATES['liquidity_actions'].format(
            chain=self._get_chain_schema(chain),
            start_date=start_date or '2024-01-01',
            end_date=end_date or datetime.now().strftime('%Y-%m-%d'),
            filters=' '.join(filters), limit=limit
        )
        
        df = await self.execute_query(sql)
        if not df.empty:
            df['chain'] = chain
            if 'amount0_usd' in df.columns and 'amount1_usd' in df.columns:
                df['total_usd'] = df['amount0_usd'].fillna(0) + df['amount1_usd'].fillna(0)
        return df
    
    async def fetch_bridge_activity(
        self, chain: str = 'ethereum', start_date: Optional[str] = None,
        end_date: Optional[str] = None, destination_chain: Optional[str] = None,
        limit: int = 10000
    ) -> pd.DataFrame:
        """Fetch cross-chain bridge transactions."""
        filters = []
        if destination_chain:
            filters.append(f"AND destination_chain ILIKE '%{destination_chain}%'")
        
        sql = QUERY_TEMPLATES['bridge_activity'].format(
            chain=self._get_chain_schema(chain),
            start_date=start_date or '2024-01-01',
            end_date=end_date or datetime.now().strftime('%Y-%m-%d'),
            filters=' '.join(filters), limit=limit
        )
        
        df = await self.execute_query(sql)
        if not df.empty:
            df['source_chain'] = chain
        return df
    
    async def fetch_whale_transactions(
        self, chain: str = 'ethereum', token_symbol: str = 'ETH',
        min_usd_value: float = 1000000, start_date: Optional[str] = None,
        end_date: Optional[str] = None, limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch large whale transactions."""
        filters = []
        if token_symbol:
            filters.append(f"AND symbol = '{token_symbol}'")
        
        sql = QUERY_TEMPLATES['whale_transfers'].format(
            chain=self._get_chain_schema(chain),
            start_date=start_date or '2024-01-01',
            end_date=end_date or datetime.now().strftime('%Y-%m-%d'),
            min_usd=min_usd_value,
            filters=' '.join(filters), limit=limit
        )
        
        df = await self.execute_query(sql)
        if not df.empty:
            df['chain'] = chain
            df['is_mega_whale'] = df.get('amount_usd', 0) >= 10_000_000
        return df
    
    async def fetch_custom_query(self, sql: str) -> pd.DataFrame:
        """Execute a custom SQL query."""
        return await self.execute_query(sql)
    
    async def fetch_funding_rates(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Flipside doesn't have funding rate data."""
        return pd.DataFrame()
    
    async def fetch_ohlcv(
        self, symbols: List[str], timeframe: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch price data via SQL queries."""
        # Would need to construct custom SQL for price aggregation
        return pd.DataFrame()
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {**self.collection_stats, 'venue': self.VENUE, 'supported_chains': list(self.SUPPORTED_CHAINS.keys())}
    
    @staticmethod
    def get_supported_chains() -> List[str]:
        """Get list of supported chains."""
        return [c.name.lower() for c in Chain]
    
    @staticmethod
    def get_dex_platforms() -> List[str]:
        """Get list of DEX platforms."""
        return [p.value for p in DEXPlatform]
    
    @staticmethod
    def get_lending_platforms() -> List[str]:
        """Get list of lending platforms."""
        return [p.value for p in LendingPlatform]
    
    async def close(self) -> None:
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

async def test_flipside():
    """Test Flipside collector."""
    config = {'api_key': '', 'query_timeout': 300}
    collector = FlipsideCollector(config)
    try:
        print(f"Supported chains: {collector.get_supported_chains()}")
        print(f"DEX platforms: {collector.get_dex_platforms()}")
        
        # Test dataclasses
        swap = DEXSwap(
            timestamp=datetime.utcnow(), tx_hash='0x123', chain='ethereum',
            platform='uniswap-v3', sender='0xabc',
            token_in_symbol='ETH', token_in_amount=10.0,
            token_out_symbol='USDC', token_out_amount=25000.0,
            amount_in_usd=25000.0, amount_out_usd=25000.0
        )
        print(f"Swap pair: {swap.pair}, size: {swap.size_category.value}")
    finally:
        await collector.close()

if __name__ == '__main__':
    asyncio.run(test_flipside())