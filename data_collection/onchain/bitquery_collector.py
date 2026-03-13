"""
Bitquery On-Chain Data Collector - GraphQL Multi-Chain Analytics

validated collector for blockchain data across 40+ chains using
GraphQL queries. Provides DEX trades, token transfers, smart contract
events, and cross-chain analytics.

===============================================================================
OVERVIEW
===============================================================================

Bitquery provides blockchain data through a powerful GraphQL API supporting:
    - DEX trades across major protocols
    - Token transfers and balances
    - Smart contract events
    - NFT data
    - Cross-chain analytics

Target Users:
    - DeFi researchers
    - Trading firms
    - Protocol analytics
    - Compliance teams

Key Differentiators:
    - GraphQL flexibility
    - 40+ blockchain support
    - Real-time and historical data
    - Unified query interface

===============================================================================
API TIERS
===============================================================================

    ============== ==================== ============== ================
    Tier Rate Limit Credits/Month Best For
    ============== ==================== ============== ================
    Free 10 queries/min 100K Evaluation
    Developer 60 queries/min 500K Development
    Team 120 queries/min 2M Production
    Enterprise Custom Custom Institutional
    ============== ==================== ============== ================

===============================================================================
SUPPORTED NETWORKS
===============================================================================

EVM Chains:
    - Ethereum, Polygon, Arbitrum, Optimism, Base
    - BSC, Avalanche, Fantom, Gnosis, Celo

Non-EVM:
    - Solana, Bitcoin, Tron, Cosmos, others

DEX Protocols:
    - Uniswap V2/V3, SushiSwap, PancakeSwap
    - Curve, Balancer, QuickSwap, TraderJoe

===============================================================================
DATA TYPES COLLECTED
===============================================================================

DEX Trades:
    - Trade timestamp and block
    - Exchange and pool information
    - Base and quote currencies
    - Trade amounts and prices
    - Transaction details

Token Transfers:
    - Sender and receiver addresses
    - Token details and amounts
    - USD values
    - Transaction hashes

Token Holders:
    - Holder addresses
    - Balance amounts
    - Percentage of supply
    - Historical balances

Smart Contract Events:
    - Event name and signature
    - Contract address
    - Block and timestamp
    - Transaction context

Pool Statistics:
    - Pool addresses
    - Token pairs
    - Volume metrics
    - Trade counts

===============================================================================
USAGE EXAMPLES
===============================================================================

DEX trades:

    >>> from data_collection.onchain import BitqueryCollector
    >>> 
    >>> collector = BitqueryCollector({'access_token': 'your-token'})
    >>> try:
    ... trades = await collector.get_dex_trades(
    ... network='ethereum',
    ... start_date='2024-01-01',
    ... limit=1000
    ... )
    ... print(f"Fetched {len(trades)} trades")
    ... finally:
    ... await collector.close()

Token transfers:

    >>> transfers = await collector.get_token_transfers(
    ... network='ethereum',
    ... token_address='0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48', # USDC
    ... start_date='2024-01-01'
    ... )

Whale movements:

    >>> whales = await collector.fetch_whale_transfers(
    ... token_address='0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
    ... min_amount_usd=1_000_000
    ... )

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

DEX Analytics:
    - Cross-DEX price comparison
    - Volume analysis
    - Liquidity assessment
    - Arbitrage opportunity detection

Flow Analysis:
    - Whale movement tracking
    - Smart money following
    - Token distribution analysis
    - Accumulation/distribution patterns

Market Structure:
    - Pool liquidity comparison
    - Trading activity metrics
    - Protocol market share
    - Fee analysis

Risk Assessment:
    - Token holder concentration
    - Large transfer monitoring
    - Contract interaction patterns

===============================================================================
GRAPHQL QUERY STRUCTURE
===============================================================================

Bitquery uses GraphQL for flexible data queries:

    query {
      ethereum(network: ethereum) {
        dexTrades(options: {limit: 100}) {
          block { timestamp { time } }
          exchange { name }
          baseCurrency { symbol }
          quoteCurrency { symbol }
          tradeAmount(in: USD)
        }
      }
    }

===============================================================================
DATA QUALITY CONSIDERATIONS
===============================================================================

- GraphQL queries have complexity limits
- Some chains have delayed indexing
- Historical data may have gaps
- Token metadata varies by chain
- USD values use CoinGecko/CMC prices

Version: 2.0.0
API Documentation: https://docs.bitquery.io/
"""

import os
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

class Network(Enum):
    """Supported blockchain networks."""
    ETHEREUM = 'ethereum'
    BSC = 'bsc'
    POLYGON = 'matic'
    ARBITRUM = 'arbitrum'
    OPTIMISM = 'optimism'
    AVALANCHE = 'avalanche'
    FANTOM = 'fantom'
    BASE = 'base'
    GNOSIS = 'xdai'
    CELO = 'celo'
    SOLANA = 'solana'
    BITCOIN = 'bitcoin'

class DEXProtocol(Enum):
    """DEX protocol names."""
    UNISWAP_V2 = 'Uniswap'
    UNISWAP_V3 = 'Uniswap V3'
    SUSHISWAP = 'SushiSwap'
    PANCAKESWAP = 'PancakeSwap'
    CURVE = 'Curve'
    BALANCER = 'Balancer'
    QUICKSWAP = 'QuickSwap'
    TRADER_JOE = 'TraderJoe'
    CAMELOT = 'Camelot'

class TradeSide(Enum):
    """Trade side classification."""
    BUY = 'buy'
    SELL = 'sell'
    UNKNOWN = 'unknown'

class TransferType(Enum):
    """Token transfer type."""
    TRANSFER = 'transfer'
    MINT = 'mint'
    BURN = 'burn'

class VolumeLevel(Enum):
    """Volume classification."""
    VERY_HIGH = 'very_high' # > $10M
    HIGH = 'high' # $1M - $10M
    MEDIUM = 'medium' # $100K - $1M
    LOW = 'low' # $10K - $100K
    VERY_LOW = 'very_low' # < $10K

class HolderTier(Enum):
    """Token holder tier classification."""
    WHALE = 'whale' # > 1% supply
    LARGE = 'large' # 0.1% - 1%
    MEDIUM = 'medium' # 0.01% - 0.1%
    SMALL = 'small' # < 0.01%

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class BitqueryDEXTrade:
    """DEX trade from Bitquery."""
    timestamp: datetime
    block_height: int
    exchange: str
    exchange_full: Optional[str]
    base_symbol: str
    base_address: str
    quote_symbol: str
    quote_address: str
    side: str
    trade_amount_usd: float
    base_amount: float
    quote_amount: float
    price: float
    tx_hash: str
    gas_price: float
    network: str
    
    @property
    def side_enum(self) -> TradeSide:
        """Get side as enum."""
        try:
            return TradeSide(self.side.lower())
        except ValueError:
            return TradeSide.UNKNOWN
    
    @property
    def pair(self) -> str:
        """Trading pair string."""
        return f"{self.base_symbol}/{self.quote_symbol}"
    
    @property
    def volume_level(self) -> VolumeLevel:
        """Classify trade volume."""
        vol = self.trade_amount_usd
        if vol > 10_000_000:
            return VolumeLevel.VERY_HIGH
        elif vol > 1_000_000:
            return VolumeLevel.HIGH
        elif vol > 100_000:
            return VolumeLevel.MEDIUM
        elif vol > 10_000:
            return VolumeLevel.LOW
        else:
            return VolumeLevel.VERY_LOW
    
    @property
    def is_whale_trade(self) -> bool:
        """Check if whale-sized trade (>$100K)."""
        return self.trade_amount_usd >= 100_000
    
    @property
    def is_stablecoin_pair(self) -> bool:
        """Check if pair includes stablecoin."""
        stables = {'USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'FRAX'}
        return self.base_symbol.upper() in stables or self.quote_symbol.upper() in stables
    
    @property
    def gas_cost_estimate(self) -> float:
        """Estimated gas cost in USD (assumes ETH ~$3000)."""
        if self.gas_price:
            # Approximate swap gas usage
            return (self.gas_price * 150000) / 1e9 * 3000
        return 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'block_height': self.block_height,
            'exchange': self.exchange,
            'exchange_full': self.exchange_full,
            'pair': self.pair,
            'base_symbol': self.base_symbol,
            'base_address': self.base_address,
            'quote_symbol': self.quote_symbol,
            'quote_address': self.quote_address,
            'side': self.side,
            'side_enum': self.side_enum.value,
            'trade_amount_usd': self.trade_amount_usd,
            'base_amount': self.base_amount,
            'quote_amount': self.quote_amount,
            'price': self.price,
            'volume_level': self.volume_level.value,
            'is_whale_trade': self.is_whale_trade,
            'is_stablecoin_pair': self.is_stablecoin_pair,
            'tx_hash': self.tx_hash,
            'gas_price': self.gas_price,
            'gas_cost_estimate': self.gas_cost_estimate,
            'network': self.network,
            'venue': 'bitquery',
        }

@dataclass
class BitqueryTransfer:
    """Token transfer from Bitquery."""
    timestamp: datetime
    block_height: int
    from_address: str
    to_address: str
    token_symbol: str
    token_address: str
    amount: float
    amount_usd: float
    tx_hash: str
    network: str
    
    @property
    def is_whale_transfer(self) -> bool:
        """Check if whale-sized transfer (>$100K)."""
        return self.amount_usd >= 100_000
    
    @property
    def is_large_transfer(self) -> bool:
        """Check if large transfer (>$1M)."""
        return self.amount_usd >= 1_000_000
    
    @property
    def volume_level(self) -> VolumeLevel:
        """Classify transfer volume."""
        vol = self.amount_usd
        if vol > 10_000_000:
            return VolumeLevel.VERY_HIGH
        elif vol > 1_000_000:
            return VolumeLevel.HIGH
        elif vol > 100_000:
            return VolumeLevel.MEDIUM
        elif vol > 10_000:
            return VolumeLevel.LOW
        else:
            return VolumeLevel.VERY_LOW
    
    @property
    def is_mint(self) -> bool:
        """Check if mint (from zero address)."""
        return self.from_address == '0x0000000000000000000000000000000000000000'
    
    @property
    def is_burn(self) -> bool:
        """Check if burn (to zero/dead address)."""
        burn_addresses = {
            '0x0000000000000000000000000000000000000000',
            '0x000000000000000000000000000000000000dEaD'
        }
        return self.to_address.lower() in {a.lower() for a in burn_addresses}
    
    @property
    def transfer_type(self) -> TransferType:
        """Determine transfer type."""
        if self.is_mint:
            return TransferType.MINT
        elif self.is_burn:
            return TransferType.BURN
        return TransferType.TRANSFER
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'block_height': self.block_height,
            'from_address': self.from_address,
            'to_address': self.to_address,
            'token_symbol': self.token_symbol,
            'token_address': self.token_address,
            'amount': self.amount,
            'amount_usd': self.amount_usd,
            'volume_level': self.volume_level.value,
            'is_whale_transfer': self.is_whale_transfer,
            'is_large_transfer': self.is_large_transfer,
            'transfer_type': self.transfer_type.value,
            'tx_hash': self.tx_hash,
            'network': self.network,
            'venue': 'bitquery',
        }

@dataclass
class BitqueryHolder:
    """Token holder from Bitquery."""
    rank: int
    address: str
    balance: float
    balance_usd: float
    token_address: str
    network: str
    pct_of_supply: Optional[float] = None
    
    @property
    def holder_tier(self) -> HolderTier:
        """Classify holder tier based on supply percentage."""
        if self.pct_of_supply is None:
            return HolderTier.SMALL
        if self.pct_of_supply >= 1:
            return HolderTier.WHALE
        elif self.pct_of_supply >= 0.1:
            return HolderTier.LARGE
        elif self.pct_of_supply >= 0.01:
            return HolderTier.MEDIUM
        else:
            return HolderTier.SMALL
    
    @property
    def is_whale(self) -> bool:
        """Check if whale holder (>1% supply)."""
        return self.holder_tier == HolderTier.WHALE
    
    @property
    def is_top_holder(self) -> bool:
        """Check if top 10 holder."""
        return self.rank <= 10
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'rank': self.rank,
            'address': self.address,
            'balance': self.balance,
            'balance_usd': self.balance_usd,
            'pct_of_supply': self.pct_of_supply,
            'holder_tier': self.holder_tier.value,
            'is_whale': self.is_whale,
            'is_top_holder': self.is_top_holder,
            'token_address': self.token_address,
            'network': self.network,
            'venue': 'bitquery',
        }

@dataclass
class BitqueryPoolStats:
    """DEX pool statistics from Bitquery."""
    pool_address: Optional[str]
    base_symbol: str
    base_address: str
    quote_symbol: str
    quote_address: str
    volume_usd: float
    trade_count: int
    exchange: str
    network: str
    
    @property
    def pair(self) -> str:
        """Trading pair string."""
        return f"{self.base_symbol}/{self.quote_symbol}"
    
    @property
    def avg_trade_size(self) -> float:
        """Average trade size in USD."""
        return self.volume_usd / self.trade_count if self.trade_count > 0 else 0
    
    @property
    def volume_level(self) -> VolumeLevel:
        """Classify pool volume."""
        vol = self.volume_usd
        if vol > 10_000_000:
            return VolumeLevel.VERY_HIGH
        elif vol > 1_000_000:
            return VolumeLevel.HIGH
        elif vol > 100_000:
            return VolumeLevel.MEDIUM
        elif vol > 10_000:
            return VolumeLevel.LOW
        else:
            return VolumeLevel.VERY_LOW
    
    @property
    def is_active_pool(self) -> bool:
        """Check if actively traded (>100 trades)."""
        return self.trade_count >= 100
    
    @property
    def is_high_volume(self) -> bool:
        """Check if high volume (>$1M)."""
        return self.volume_usd >= 1_000_000
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'pool_address': self.pool_address,
            'pair': self.pair,
            'base_symbol': self.base_symbol,
            'base_address': self.base_address,
            'quote_symbol': self.quote_symbol,
            'quote_address': self.quote_address,
            'volume_usd': self.volume_usd,
            'trade_count': self.trade_count,
            'avg_trade_size': self.avg_trade_size,
            'volume_level': self.volume_level.value,
            'is_active_pool': self.is_active_pool,
            'is_high_volume': self.is_high_volume,
            'exchange': self.exchange,
            'network': self.network,
            'venue': 'bitquery',
        }

@dataclass
class BitqueryEvent:
    """Smart contract event from Bitquery."""
    timestamp: datetime
    block_height: int
    contract_address: str
    event_name: str
    event_signature: str
    tx_hash: str
    network: str
    
    @property
    def is_swap_event(self) -> bool:
        """Check if swap-related event."""
        swap_events = {'Swap', 'TokenSwap', 'Exchange'}
        return self.event_name in swap_events
    
    @property
    def is_liquidity_event(self) -> bool:
        """Check if liquidity event."""
        liq_events = {'Mint', 'Burn', 'AddLiquidity', 'RemoveLiquidity'}
        return self.event_name in liq_events
    
    @property
    def is_transfer_event(self) -> bool:
        """Check if transfer event."""
        return self.event_name == 'Transfer'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'block_height': self.block_height,
            'contract_address': self.contract_address,
            'event_name': self.event_name,
            'event_signature': self.event_signature,
            'is_swap_event': self.is_swap_event,
            'is_liquidity_event': self.is_liquidity_event,
            'is_transfer_event': self.is_transfer_event,
            'tx_hash': self.tx_hash,
            'network': self.network,
            'venue': 'bitquery',
        }

@dataclass
class BitqueryOHLCV:
    """OHLCV data aggregated from DEX trades."""
    timestamp: datetime
    network: str
    base_token: str
    quote_token: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int
    
    @property
    def pair(self) -> str:
        """Trading pair string."""
        return f"{self.base_token}/{self.quote_token}"
    
    @property
    def typical_price(self) -> float:
        """Typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3
    
    @property
    def range_pct(self) -> float:
        """Price range as percentage."""
        return (self.high - self.low) / self.open * 100 if self.open > 0 else 0
    
    @property
    def return_pct(self) -> float:
        """Period return percentage."""
        return (self.close - self.open) / self.open * 100 if self.open > 0 else 0
    
    @property
    def is_bullish(self) -> bool:
        """Check if bullish candle."""
        return self.close > self.open
    
    @property
    def avg_trade_size(self) -> float:
        """Average trade size."""
        return self.volume / self.trades if self.trades > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'network': self.network,
            'pair': self.pair,
            'base_token': self.base_token,
            'quote_token': self.quote_token,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'trades': self.trades,
            'typical_price': self.typical_price,
            'range_pct': self.range_pct,
            'return_pct': self.return_pct,
            'is_bullish': self.is_bullish,
            'avg_trade_size': self.avg_trade_size,
            'venue': 'bitquery',
        }

# =============================================================================
# Global Circuit Breaker for Bitquery Quota Exhaustion
# =============================================================================

class BitqueryQuotaCircuitBreaker:
    """
    Global circuit breaker for Bitquery quota exhaustion (HTTP 402).

    When Bitquery returns 402 (Payment Required - points exhausted), this
    circuit breaker trips and prevents ALL subsequent Bitquery requests from
    even attempting, saving bandwidth and avoiding repeated 402 errors.

    This is a class-level singleton shared across all BitqueryCollector instances.
    The quota typically resets monthly based on your subscription plan.
    """
    _instance: Optional['BitqueryQuotaCircuitBreaker'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._is_tripped = False
            cls._instance._trip_time: Optional[datetime] = None
            cls._instance._trip_count = 0
            cls._instance._requests_blocked = 0
        return cls._instance

    def trip(self) -> None:
        """Trip the circuit breaker (called when 402 is received)."""
        if not self._is_tripped:
            self._is_tripped = True
            self._trip_time = datetime.utcnow()
            self._trip_count += 1
            logger.error(
                f" BITQUERY CIRCUIT BREAKER TRIPPED: Quota exhausted (402). "
                f"All subsequent Bitquery requests will be blocked. "
                f"Quota resets monthly - consider upgrading your plan or waiting."
            )

    def is_open(self) -> bool:
        """Check if circuit breaker is open (requests should be blocked)."""
        return self._is_tripped

    def block_request(self) -> None:
        """Record a blocked request for stats."""
        self._requests_blocked += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            'is_tripped': self._is_tripped,
            'trip_time': self._trip_time.isoformat() if self._trip_time else None,
            'trip_count': self._trip_count,
            'requests_blocked': self._requests_blocked,
        }

    def reset(self) -> None:
        """
        Reset the circuit breaker (for testing or manual recovery).

        WARNING: Only call this if you're sure the quota has been reset/restored.
        """
        self._is_tripped = False
        self._trip_time = None
        logger.info("Bitquery quota circuit breaker reset manually")

def get_bitquery_circuit_breaker() -> BitqueryQuotaCircuitBreaker:
    """Get the global Bitquery circuit breaker singleton."""
    return BitqueryQuotaCircuitBreaker()

# =============================================================================
# Main Collector Class
# =============================================================================

class BitqueryCollector(BaseCollector):
    """
    Bitquery GraphQL-based blockchain data collector.
    
    validated implementation providing DEX trades, token transfers,
    and cross-chain analytics through GraphQL queries.
    
    Features:
        - DEX trades across multiple protocols
        - Token transfers and balances
        - Smart contract events
        - Token holder analysis
        - Pool statistics
        - OHLCV aggregation from trades
    
    Example:
        >>> collector = BitqueryCollector({'access_token': 'your-token'})
        >>> try:
        ... trades = await collector.get_dex_trades('ethereum', limit=1000)
        ... transfers = await collector.get_token_transfers('ethereum', token)
        ... finally:
        ... await collector.close()
    
    Attributes:
        VENUE: 'bitquery'
        VENUE_TYPE: 'onchain'
    """
    
    VENUE = 'bitquery'
    VENUE_TYPE = 'onchain'

    # V2 API endpoint (V1 was sunset - use streaming API)
    BASE_URL = 'https://streaming.bitquery.io/graphql'
    BASE_URL_V1 = 'https://graphql.bitquery.io' # Fallback for V1 queries

    # Network name mapping for V2 API
    NETWORKS = {
        'ethereum': 'eth',
        'bsc': 'bsc',
        'polygon': 'matic',
        'arbitrum': 'arbitrum',
        'optimism': 'optimism',
        'avalanche': 'avalanche_c',
        'fantom': 'fantom',
        'base': 'base',
        'gnosis': 'gnosis',
        'celo': 'celo',
    }

    # V1 network names (legacy fallback)
    NETWORKS_V1 = {
        'ethereum': 'ethereum',
        'bsc': 'bsc',
        'polygon': 'matic',
        'arbitrum': 'arbitrum',
        'optimism': 'optimism',
        'avalanche': 'avalanche',
        'fantom': 'fantom',
        'base': 'base',
        'gnosis': 'xdai',
        'celo': 'celo',
    }
    
    # Major DEX factory addresses
    DEX_FACTORIES = {
        'ethereum': {
            'uniswap_v2': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',
            'uniswap_v3': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
            'sushiswap': '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac',
        },
        'bsc': {
            'pancakeswap_v2': '0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73',
        },
        'polygon': {
            'quickswap': '0x5757371414417b8C6CAad45bAeF941aBc7d3Ab32',
        },
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Bitquery collector.

        Args:
            config: Configuration with access_token for authentication
        """
        config = config or {}
        super().__init__(config)

        # CRITICAL: Set supported data types for dynamic routing
        self.supported_data_types = ['on_chain_metrics', 'dex_trades']
        self.venue = 'bitquery'

        # Import VenueType from base_collector
        from ..base_collector import VenueType
        self.venue_type = VenueType.ONCHAIN
        self.requires_auth = True # Requires Bitquery access token

        self.access_token = config.get('access_token') or config.get('bitquery_access_token') or os.getenv('BITQUERY_ACCESS_TOKEN', '')
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = get_shared_rate_limiter(
            'bitquery',
            rate=config.get('rate_limit', 30),
            per=60,
            burst=config.get('burst', 15)
        )

        # Global circuit breaker for quota exhaustion (shared across all instances)
        self._circuit_breaker = get_bitquery_circuit_breaker()

        # Track networks where V2 API has failed (to avoid repeated fallback log spam)
        self._v2_failed_networks: set = set()
        self._v1_no_data_logged: set = set() # Track networks where V1 also returned no data

        self.stats = {'requests': 0, 'records': 0, 'errors': 0}

        if not self.access_token:
            logger.warning("Bitquery access token not provided")
        else:
            logger.info(f"Initialized Bitquery collector with data types: {self.supported_data_types}")
            logger.info("Bitquery collector initialized with access token")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with authentication."""
        if self.session is None or self.session.closed:
            headers = {'Content-Type': 'application/json'}
            if self.access_token:
                headers['Authorization'] = f'Bearer {self.access_token}'
            
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self.session
    
    async def _execute_query(self, query: str, variables: Optional[Dict] = None, use_v2: bool = True) -> Dict:
        """Execute GraphQL query against Bitquery API with circuit breaker protection.

        Args:
            query: GraphQL query string
            variables: Optional query variables
            use_v2: Use V2 API (default True, falls back to V1 if needed)

        Returns:
            Query result dictionary
        """
        # CIRCUIT BREAKER CHECK: Block all requests if quota is exhausted
        if self._circuit_breaker.is_open():
            self._circuit_breaker.block_request()
            self.stats['errors'] += 1
            return {} # Fail fast without making HTTP request

        session = await self._get_session()
        await self.rate_limiter.acquire()
        self.stats['requests'] += 1

        payload = {'query': query}
        if variables:
            payload['variables'] = variables

        # Try V2 first, fallback to V1
        url = self.BASE_URL if use_v2 else self.BASE_URL_V1

        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'errors' in result:
                        errors = result['errors']
                        logger.warning(f"Bitquery GraphQL errors: {errors}")
                        # If V2 fails with schema error, try V1
                        if use_v2 and any('not supported' in str(e) or 'unknown' in str(e).lower() for e in errors):
                            logger.info("Falling back to Bitquery V1 API")
                            return await self._execute_query(query, variables, use_v2=False)
                        self.stats['errors'] += 1
                        return {}
                    return result.get('data', {})
                elif response.status == 401:
                    logger.error("Bitquery authentication failed - check BITQUERY_ACCESS_TOKEN")
                    self.stats['errors'] += 1
                    return {}
                elif response.status == 402:
                    # 402 = Payment Required / Quota Exhausted - TRIP THE CIRCUIT BREAKER
                    # This will block ALL subsequent Bitquery requests globally
                    self._circuit_breaker.trip()
                    self.stats['errors'] += 1
                    return {} # Return empty instead of raising to allow graceful degradation
                elif response.status == 429:
                    logger.warning("Bitquery rate limit hit, waiting 60s")
                    await asyncio.sleep(60)
                    self.stats['errors'] += 1
                    return {}
                else:
                    text = await response.text()
                    logger.error(f"Bitquery HTTP {response.status}: {text[:200]}")
                    self.stats['errors'] += 1
                    return {}
        except Exception as e:
            logger.error(f"Bitquery request error: {e}")
            self.stats['errors'] += 1
            return {}
    
    def _get_network(self, network: str) -> str:
        """Convert network name to Bitquery format."""
        return self.NETWORKS.get(network.lower(), network)
    
    async def get_dex_trades(
        self,
        network: str = 'ethereum',
        base_currency: Optional[str] = None,
        quote_currency: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch DEX trades from specified network using V2 API."""
        net = self._get_network(network)

        since = start_date or (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d')
        till = end_date or datetime.utcnow().strftime('%Y-%m-%d')

        # V2 API query format
        query = """
        {
            EVM(network: %s, dataset: combined) {
                DEXTrades(
                    limit: {count: %d}
                    orderBy: {descending: Block_Time}
                    where: {Block: {Date: {since: "%s", till: "%s"}}}
                ) {
                    Block {
                        Number
                        Time
                    }
                    Transaction {
                        Hash
                    }
                    Trade {
                        Dex {
                            ProtocolName
                            ProtocolFamily
                        }
                        Buy {
                            Amount
                            Currency {
                                Symbol
                                SmartContract
                            }
                            Price
                        }
                        Sell {
                            Amount
                            Currency {
                                Symbol
                                SmartContract
                            }
                            Price
                        }
                    }
                }
            }
        }
        """ % (net, limit, since, till)

        # If V2 already known to fail for this network, skip directly to V1
        if network.lower() in self._v2_failed_networks:
            return await self._get_dex_trades_v1(network, base_currency, quote_currency, start_date, end_date, limit)

        data = await self._execute_query(query)
        trades = data.get('EVM', {}).get('DEXTrades', [])

        # If V2 returns no data, try V1 format as fallback (silent after first attempt per network)
        if not trades:
            if network.lower() not in self._v2_failed_networks:
                self._v2_failed_networks.add(network.lower())
            return await self._get_dex_trades_v1(network, base_currency, quote_currency, start_date, end_date, limit)

        records = []
        for trade in trades:
            try:
                block = trade.get('Block', {})
                tx = trade.get('Transaction', {})
                trade_data = trade.get('Trade', {})
                buy = trade_data.get('Buy', {})
                sell = trade_data.get('Sell', {})
                dex = trade_data.get('Dex', {})

                # Calculate trade value from buy amount and price
                buy_amount = float(buy.get('Amount', 0) or 0)
                buy_price = float(buy.get('Price', 0) or 0)
                trade_value_usd = buy_amount * buy_price if buy_price > 0 else 0

                t = BitqueryDEXTrade(
                    timestamp=pd.to_datetime(block.get('Time')),
                    block_height=int(block.get('Number', 0)),
                    exchange=dex.get('ProtocolName', 'Unknown'),
                    exchange_full=dex.get('ProtocolFamily', 'Unknown'),
                    base_symbol=buy.get('Currency', {}).get('Symbol', ''),
                    base_address=buy.get('Currency', {}).get('SmartContract', ''),
                    quote_symbol=sell.get('Currency', {}).get('Symbol', ''),
                    quote_address=sell.get('Currency', {}).get('SmartContract', ''),
                    side='buy',
                    trade_amount_usd=trade_value_usd,
                    base_amount=buy_amount,
                    quote_amount=float(sell.get('Amount', 0) or 0),
                    price=buy_price,
                    tx_hash=tx.get('Hash', ''),
                    gas_price=0, # V2 doesn't include gas in this query
                    network=network
                )
                records.append(t.to_dict())
            except Exception as e:
                logger.warning(f"Failed to parse DEX trade: {e}")
                continue

        self.stats['records'] += len(records)
        logger.info(f"Bitquery: Fetched {len(records)} DEX trades from {network}")
        return pd.DataFrame(records)

    async def _get_dex_trades_v1(
        self,
        network: str = 'ethereum',
        base_currency: Optional[str] = None,
        quote_currency: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch DEX trades using V1 API format (fallback)."""
        net = self.NETWORKS_V1.get(network.lower(), network)

        since = start_date or (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d')
        till = end_date or datetime.utcnow().strftime('%Y-%m-%d')

        query = """
        query ($network: EthereumNetwork!, $limit: Int!, $since: ISO8601DateTime, $till: ISO8601DateTime) {
            ethereum(network: $network) {
                dexTrades(
                    options: {limit: $limit, desc: "block.timestamp.time"}
                    date: {since: $since, till: $till}
                ) {
                    block { timestamp { time(format: "%Y-%m-%d %H:%M:%S") } height }
                    exchange { name fullName }
                    baseCurrency { symbol address decimals }
                    quoteCurrency { symbol address decimals }
                    side
                    tradeAmount(in: USD)
                    baseAmount
                    quoteAmount
                    quotePrice
                    transaction { hash gasPrice gasValue }
                }
            }
        }
        """

        variables = {'network': net, 'limit': limit, 'since': since, 'till': till}

        data = await self._execute_query(query, variables, use_v2=False)
        trades = data.get('ethereum', {}).get('dexTrades', [])

        if not trades:
            # Only log once per network to reduce spam - use debug level
            if network.lower() not in self._v1_no_data_logged:
                logger.debug(f"Bitquery: No DEX trades available for {network}")
                self._v1_no_data_logged.add(network.lower())
            return pd.DataFrame()

        records = []
        for trade in trades:
            try:
                t = BitqueryDEXTrade(
                    timestamp=pd.to_datetime(trade['block']['timestamp']['time']),
                    block_height=trade['block']['height'],
                    exchange=trade['exchange']['name'],
                    exchange_full=trade['exchange']['fullName'],
                    base_symbol=trade['baseCurrency']['symbol'],
                    base_address=trade['baseCurrency']['address'],
                    quote_symbol=trade['quoteCurrency']['symbol'],
                    quote_address=trade['quoteCurrency']['address'],
                    side=trade['side'] or 'unknown',
                    trade_amount_usd=float(trade['tradeAmount']) if trade['tradeAmount'] else 0,
                    base_amount=float(trade['baseAmount']) if trade['baseAmount'] else 0,
                    quote_amount=float(trade['quoteAmount']) if trade['quoteAmount'] else 0,
                    price=float(trade['quotePrice']) if trade['quotePrice'] else 0,
                    tx_hash=trade['transaction']['hash'],
                    gas_price=float(trade['transaction']['gasPrice']) if trade['transaction']['gasPrice'] else 0,
                    network=network
                )
                records.append(t.to_dict())
            except Exception as e:
                logger.warning(f"Failed to parse V1 DEX trade: {e}")
                continue

        self.stats['records'] += len(records)
        logger.info(f"Bitquery V1: Fetched {len(records)} DEX trades from {network}")
        return pd.DataFrame(records)
    
    async def get_token_transfers(
        self,
        network: str = 'ethereum',
        token_address: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch token transfers on specified network."""
        net = self._get_network(network)
        since = start_date or (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d')
        till = end_date or datetime.utcnow().strftime('%Y-%m-%d')
        
        query = """
        query ($network: EthereumNetwork!, $limit: Int!, $since: ISO8601DateTime, $till: ISO8601DateTime, $token: String) {
            ethereum(network: $network) {
                transfers(
                    options: {limit: $limit, desc: "block.timestamp.time"}
                    date: {since: $since, till: $till}
                    currency: {is: $token}
                ) {
                    block { timestamp { time(format: "%Y-%m-%d %H:%M:%S") } height }
                    sender { address }
                    receiver { address }
                    currency { symbol address decimals }
                    amount
                    amountInUSD: amount(in: USD)
                    transaction { hash }
                }
            }
        }
        """
        
        variables = {'network': net, 'limit': limit, 'since': since, 'till': till, 'token': token_address}
        
        data = await self._execute_query(query, variables)
        transfers = data.get('ethereum', {}).get('transfers', [])
        
        if not transfers:
            return pd.DataFrame()
        
        records = []
        for transfer in transfers:
            t = BitqueryTransfer(
                timestamp=pd.to_datetime(transfer['block']['timestamp']['time']),
                block_height=transfer['block']['height'],
                from_address=transfer['sender']['address'],
                to_address=transfer['receiver']['address'],
                token_symbol=transfer['currency']['symbol'],
                token_address=transfer['currency']['address'],
                amount=float(transfer['amount']) if transfer['amount'] else 0,
                amount_usd=float(transfer['amountInUSD']) if transfer['amountInUSD'] else 0,
                tx_hash=transfer['transaction']['hash'],
                network=network
            )
            records.append(t.to_dict())
        
        self.stats['records'] += len(records)
        return pd.DataFrame(records)
    
    async def get_token_holders(
        self,
        network: str = 'ethereum',
        token_address: str = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """Fetch top token holders."""
        net = self._get_network(network)
        
        query = """
        query ($network: EthereumNetwork!, $limit: Int!, $token: String!) {
            ethereum(network: $network) {
                address(
                    balances: {currency: {is: $token}}
                    options: {limit: $limit, desc: "balance"}
                ) {
                    address
                    balance
                    balanceInUSD: balance(in: USD)
                }
            }
        }
        """
        
        variables = {'network': net, 'limit': limit, 'token': token_address}
        
        data = await self._execute_query(query, variables)
        holders = data.get('ethereum', {}).get('address', [])
        
        if not holders:
            return pd.DataFrame()
        
        records = []
        for i, holder in enumerate(holders):
            h = BitqueryHolder(
                rank=i + 1,
                address=holder['address'],
                balance=float(holder['balance']) if holder['balance'] else 0,
                balance_usd=float(holder['balanceInUSD']) if holder['balanceInUSD'] else 0,
                token_address=token_address,
                network=network
            )
            records.append(h.to_dict())
        
        self.stats['records'] += len(records)
        return pd.DataFrame(records)
    
    async def get_dex_pool_stats(
        self,
        network: str = 'ethereum',
        exchange: str = 'Uniswap',
        limit: int = 100
    ) -> pd.DataFrame:
        """Fetch DEX pool statistics."""
        net = self._get_network(network)
        
        query = """
        query ($network: EthereumNetwork!, $limit: Int!, $exchange: String!) {
            ethereum(network: $network) {
                dexTrades(
                    options: {limit: $limit, desc: "tradeAmount"}
                    exchangeName: {is: $exchange}
                ) {
                    smartContract { address { address } }
                    baseCurrency { symbol address }
                    quoteCurrency { symbol address }
                    tradeAmount(in: USD)
                    count
                }
            }
        }
        """
        
        variables = {'network': net, 'limit': limit, 'exchange': exchange}
        
        data = await self._execute_query(query, variables)
        pools = data.get('ethereum', {}).get('dexTrades', [])
        
        if not pools:
            return pd.DataFrame()
        
        records = []
        for pool in pools:
            p = BitqueryPoolStats(
                pool_address=pool['smartContract']['address']['address'] if pool.get('smartContract') else None,
                base_symbol=pool['baseCurrency']['symbol'],
                base_address=pool['baseCurrency']['address'],
                quote_symbol=pool['quoteCurrency']['symbol'],
                quote_address=pool['quoteCurrency']['address'],
                volume_usd=float(pool['tradeAmount']) if pool['tradeAmount'] else 0,
                trade_count=int(pool['count']) if pool['count'] else 0,
                exchange=exchange,
                network=network
            )
            records.append(p.to_dict())
        
        self.stats['records'] += len(records)
        return pd.DataFrame(records)
    
    async def get_smart_contract_events(
        self,
        network: str = 'ethereum',
        contract_address: Optional[str] = None,
        event_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch smart contract events."""
        net = self._get_network(network)
        since = start_date or (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d')
        till = end_date or datetime.utcnow().strftime('%Y-%m-%d')
        
        query = """
        query ($network: EthereumNetwork!, $limit: Int!, $since: ISO8601DateTime, $till: ISO8601DateTime, $contract: String) {
            ethereum(network: $network) {
                smartContractEvents(
                    options: {limit: $limit, desc: "block.timestamp.time"}
                    date: {since: $since, till: $till}
                    smartContractAddress: {is: $contract}
                ) {
                    block { timestamp { time(format: "%Y-%m-%d %H:%M:%S") } height }
                    smartContract { address { address } }
                    smartContractEvent { name signature }
                    transaction { hash }
                }
            }
        }
        """
        
        variables = {'network': net, 'limit': limit, 'since': since, 'till': till, 'contract': contract_address}
        
        data = await self._execute_query(query, variables)
        events = data.get('ethereum', {}).get('smartContractEvents', [])
        
        if not events:
            return pd.DataFrame()
        
        records = []
        for event in events:
            e = BitqueryEvent(
                timestamp=pd.to_datetime(event['block']['timestamp']['time']),
                block_height=event['block']['height'],
                contract_address=event['smartContract']['address']['address'],
                event_name=event['smartContractEvent']['name'],
                event_signature=event['smartContractEvent']['signature'],
                tx_hash=event['transaction']['hash'],
                network=network
            )
            records.append(e.to_dict())
        
        self.stats['records'] += len(records)
        return pd.DataFrame(records)
    
    async def fetch_dex_ohlcv(
        self,
        base_token: str,
        quote_token: str,
        network: str = 'ethereum',
        interval: str = '1h',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a DEX pair by aggregating trades."""
        trades_df = await self.get_dex_trades(
            network=network,
            base_currency=base_token,
            quote_currency=quote_token,
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )
        
        if trades_df.empty:
            return pd.DataFrame()
        
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df.set_index('timestamp', inplace=True)
        
        freq_map = {'1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W'}
        freq = freq_map.get(interval, '1H')
        
        ohlcv = trades_df['price'].resample(freq).ohlc()
        ohlcv['volume'] = trades_df['trade_amount_usd'].resample(freq).sum()
        ohlcv['trades'] = trades_df['price'].resample(freq).count()
        ohlcv = ohlcv.dropna()
        
        records = []
        for ts, row in ohlcv.iterrows():
            o = BitqueryOHLCV(
                timestamp=ts.to_pydatetime(),
                network=network,
                base_token=base_token,
                quote_token=quote_token,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                trades=int(row['trades'])
            )
            records.append(o.to_dict())
        
        return pd.DataFrame(records)
    
    async def fetch_whale_transfers(
        self,
        token_address: str,
        network: str = 'ethereum',
        min_amount_usd: float = 100000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch large token transfers (whale movements)."""
        transfers = await self.get_token_transfers(
            network=network,
            token_address=token_address,
            start_date=start_date,
            end_date=end_date,
            limit=5000
        )
        
        if transfers.empty:
            return pd.DataFrame()
        
        whales = transfers[transfers['amount_usd'] >= min_amount_usd].copy()
        return whales.sort_values('amount_usd', ascending=False)
    
    async def fetch_funding_rates(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Bitquery doesn't provide funding rates - returns empty DataFrame."""
        logger.warning("Bitquery does not provide funding rate data")
        return pd.DataFrame()
    
    async def _fetch_single_ohlcv(
        self, symbol: str, timeframe: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV for a single symbol."""
        try:
            df = await self.fetch_dex_ohlcv(
                base_token=symbol,
                quote_token='USDT',
                interval=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            if not df.empty:
                df['symbol'] = symbol
                return df
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
        return None

    async def fetch_ohlcv(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch OHLCV by aggregating DEX trades for common pairs."""
        # PARALLELIZED: Fetch all symbols concurrently
        tasks = [
            self._fetch_single_ohlcv(symbol, timeframe, start_date, end_date)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]
        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)
    
    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            cb_stats = self._circuit_breaker.get_stats()
            if cb_stats['is_tripped']:
                logger.warning(f"Bitquery session closed. Circuit breaker was TRIPPED - blocked {cb_stats['requests_blocked']} requests")
            logger.info(f"Bitquery session closed. Stats: {self.stats}")
    
    async def collect_dex_trades(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect DEX trades - wraps get_dex_trades().

        Standardized method name for collection manager compatibility.
        """
        # Skip if circuit breaker is tripped (quota exhausted)
        if self._circuit_breaker.is_open():
            return pd.DataFrame()

        try:
            network = kwargs.get('network', 'ethereum')
            dex_name = kwargs.get('dex_name', None)
            limit = kwargs.get('limit', 1000)

            # Convert dates to datetime if needed
            if hasattr(start_date, 'strftime'):
                start_dt = start_date
            else:
                from datetime import datetime
                start_dt = datetime.strptime(str(start_date), '%Y-%m-%d')

            if hasattr(end_date, 'strftime'):
                end_dt = end_date
            else:
                from datetime import datetime
                end_dt = datetime.strptime(str(end_date), '%Y-%m-%d')

            # PARALLELIZED: Collect trades for all symbols concurrently
            async def _collect_single_symbol_trades(symbol: str) -> Optional[pd.DataFrame]:
                df = await self.get_dex_trades(
                    base_currency=symbol,
                    network=network,
                    start_date=start_dt,
                    end_date=end_dt,
                    dex_name=dex_name,
                    limit=limit
                )
                if not df.empty:
                    return df
                return None

            tasks = [_collect_single_symbol_trades(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_records = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]
            if all_records:
                return pd.concat(all_records, ignore_index=True)

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Bitquery collect_dex_trades error: {e}")
            return pd.DataFrame()

    async def collect_on_chain_metrics(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect on-chain metrics - wraps get_token_holders().

        Standardized method name for collection manager compatibility.
        """
        # Skip if circuit breaker is tripped (quota exhausted)
        if self._circuit_breaker.is_open():
            return pd.DataFrame()

        try:
            network = kwargs.get('network', 'ethereum')

            # PARALLELIZED: Collect holder data for all symbols concurrently
            async def _collect_single_symbol_holders(symbol: str) -> Optional[pd.DataFrame]:
                df = await self.get_token_holders(
                    token_address=symbol,
                    network=network
                )
                if not df.empty:
                    df['symbol'] = symbol
                    return df
                return None

            tasks = [_collect_single_symbol_holders(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_records = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]
            if all_records:
                return pd.concat(all_records, ignore_index=True)

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Bitquery collect_on_chain_metrics error: {e}")
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
            logger.error(f"Bitquery collect_funding_rates error: {e}")
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
        # Skip if circuit breaker is tripped (quota exhausted)
        if self._circuit_breaker.is_open():
            return pd.DataFrame()

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
            logger.error(f"Bitquery collect_ohlcv error: {e}")
            return pd.DataFrame()

    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        stats = self.stats.copy()
        # Include circuit breaker stats for visibility
        stats['circuit_breaker'] = self._circuit_breaker.get_stats()
        return stats
    
    @classmethod
    def get_supported_networks(cls) -> List[str]:
        """Get list of supported networks."""
        return list(cls.NETWORKS.keys())
    
    @classmethod
    def get_dex_protocols(cls) -> List[str]:
        """Get list of DEX protocols."""
        return [p.value for p in DEXProtocol]

async def test_bitquery_collector():
    """Test Bitquery collector functionality."""
    collector = BitqueryCollector({'rate_limit': 5})
    
    try:
        print("=" * 60)
        print("Bitquery Collector Test")
        print("=" * 60)
        print(f"\nSupported networks: {BitqueryCollector.get_supported_networks()}")
        print(f"DEX protocols: {BitqueryCollector.get_dex_protocols()}")
        print(f"\nStats: {collector.get_collection_stats()}")
    finally:
        await collector.close()

if __name__ == '__main__':
    asyncio.run(test_bitquery_collector())