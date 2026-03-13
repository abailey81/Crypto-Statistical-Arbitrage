"""
Covalent (GoldenRush) Collector - Multi-Chain Blockchain Data

validated collector for comprehensive blockchain data across 100+ chains.
Provides token balances, transaction history, DEX data, and NFT holdings.

===============================================================================
OVERVIEW
===============================================================================

Covalent (now GoldenRush) provides unified blockchain data including:
    - Token balances across 100+ chains
    - Transaction history with decoded logs
    - DEX swaps and liquidity
    - NFT holdings and metadata
    - Historical prices
    - Block data

Target Users:
    - DeFi applications
    - Portfolio trackers
    - Analytics platforms
    - Research teams

Key Differentiators:
    - Unified API across all chains
    - Decoded transaction logs
    - Historical state snapshots
    - Comprehensive NFT support

===============================================================================
API TIERS
===============================================================================

    ============== ==================== ============== ================
    Tier Rate Limit Credits/Month Best For
    ============== ==================== ============== ================
    Free 5 req/sec 100K Evaluation
    Premium 50 req/sec 1M Development
    Growth 100 req/sec 5M Production
    Enterprise Custom Custom Institutional
    ============== ==================== ============== ================

===============================================================================
SUPPORTED CHAINS
===============================================================================

Major EVM Chains:
    - Ethereum (1)
    - Polygon (137)
    - Arbitrum (42161)
    - Optimism (10)
    - Base (8453)
    - BSC (56)
    - Avalanche (43114)

Layer 2 / Sidechains:
    - zkSync (324)
    - Scroll (534352)
    - Linea (59144)
    - Fantom (250)
    - Gnosis (100)

===============================================================================
DATA TYPES COLLECTED
===============================================================================

Token Balances:
    - Current holdings
    - Historical snapshots
    - USD valuations
    - Token metadata

Transactions:
    - Full transaction history
    - Decoded event logs
    - Gas metrics
    - Contract interactions

DEX Data:
    - Pool information
    - Swap history
    - Liquidity positions
    - TVL metrics

NFT Data:
    - NFT holdings
    - Collection metadata
    - Transfer history

Block Data:
    - Block details
    - Block heights by date
    - Timestamp mapping

===============================================================================
USAGE EXAMPLES
===============================================================================

Token balances:

    >>> from data_collection.onchain import CovalentCollector
    >>> 
    >>> collector = CovalentCollector(api_key='your-key')
    >>> try:
    ... balances = await collector.get_token_balances(
    ... chain='ethereum',
    ... address='0x...'
    ... )
    ... print(f"Holdings: {len(balances['items'])} tokens")
    ... finally:
    ... await collector.close()

DEX pools:

    >>> pools = await collector.fetch_dex_pool_data(
    ... chain='ethereum',
    ... dex_name='uniswap_v3',
    ... top_n=100
    ... )

Wallet analysis:

    >>> analysis = await collector.fetch_wallet_analysis(
    ... chain='ethereum',
    ... addresses=['0x...', '0x...']
    ... )

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

Portfolio Tracking:
    - Multi-chain balance aggregation
    - Historical portfolio snapshots
    - Profit/loss calculation
    - Token concentration analysis

DEX Analytics:
    - Pool TVL comparison
    - Volume analysis
    - Fee tracking
    - Liquidity depth

Wallet Intelligence:
    - Smart money tracking
    - Whale identification
    - Token holder analysis
    - Transaction patterns

Cross-Chain Analysis:
    - Bridge flow tracking
    - Chain activity comparison
    - Token distribution across chains

===============================================================================
AUTHENTICATION
===============================================================================

Covalent uses Basic Auth with API key as username:

    Authorization: Basic base64(API_KEY:)

The collector handles this automatically.

===============================================================================
DATA QUALITY CONSIDERATIONS
===============================================================================

- Data freshness varies by chain (1-30 blocks)
- Some chains have limited historical data
- Token prices use CoinGecko/CMC
- Decoded logs depend on ABI availability
- NFT metadata may require IPFS resolution

Version: 2.0.0
API Documentation: https://www.covalenthq.com/docs/api/
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
import os
import base64

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================

class Chain(Enum):
    """Supported blockchain networks."""
    ETHEREUM = '1'
    POLYGON = '137'
    ARBITRUM = '42161'
    OPTIMISM = '10'
    BASE = '8453'
    BSC = '56'
    AVALANCHE = '43114'
    FANTOM = '250'
    GNOSIS = '100'
    ZKSYNC = '324'
    SCROLL = '534352'
    LINEA = '59144'
    CELO = '42220'

class DEXProtocol(Enum):
    """Supported DEX protocols."""
    UNISWAP_V2 = 'uniswap_v2'
    UNISWAP_V3 = 'uniswap_v3'
    SUSHISWAP = 'sushiswap'
    PANCAKESWAP_V2 = 'pancakeswap_v2'
    QUICKSWAP = 'quickswap'
    TRADER_JOE = 'traderjoe'
    CURVE = 'curve'
    BALANCER_V2 = 'balancer_v2'

class TokenType(Enum):
    """Token classification."""
    NATIVE = 'native'
    ERC20 = 'erc20'
    ERC721 = 'erc721'
    ERC1155 = 'erc1155'

class PortfolioTier(Enum):
    """Portfolio value tier."""
    WHALE = 'whale' # > $10M
    LARGE = 'large' # $1M - $10M
    MEDIUM = 'medium' # $100K - $1M
    SMALL = 'small' # $10K - $100K
    MICRO = 'micro' # < $10K

class HoldingSignificance(Enum):
    """Holding significance level."""
    DOMINANT = 'dominant' # > 50% of portfolio
    MAJOR = 'major' # 20-50%
    SIGNIFICANT = 'significant' # 5-20%
    MINOR = 'minor' # 1-5%
    DUST = 'dust' # < 1%

class PoolLiquidity(Enum):
    """Pool liquidity classification."""
    DEEP = 'deep' # > $10M TVL
    HIGH = 'high' # $1M - $10M
    MEDIUM = 'medium' # $100K - $1M
    LOW = 'low' # $10K - $100K
    THIN = 'thin' # < $10K

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class CovalentTokenBalance:
    """Token balance from Covalent."""
    timestamp: datetime
    chain: str
    address: str
    token_address: Optional[str]
    token_symbol: str
    token_name: str
    token_decimals: int
    balance_raw: str
    balance: float
    balance_usd: float
    price_usd: Optional[float]
    token_type: str = 'erc20'
    
    @property
    def token_type_enum(self) -> TokenType:
        """Get token type as enum."""
        try:
            return TokenType(self.token_type.lower())
        except ValueError:
            return TokenType.ERC20
    
    @property
    def is_native(self) -> bool:
        """Check if native token."""
        return self.token_type_enum == TokenType.NATIVE or self.token_address is None
    
    @property
    def is_stablecoin(self) -> bool:
        """Check if stablecoin."""
        stables = {'USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'FRAX', 'LUSD', 'USDD'}
        return self.token_symbol.upper() in stables
    
    @property
    def is_nft(self) -> bool:
        """Check if NFT token."""
        return self.token_type_enum in [TokenType.ERC721, TokenType.ERC1155]
    
    @property
    def has_value(self) -> bool:
        """Check if holding has USD value."""
        return self.balance_usd > 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'chain': self.chain,
            'address': self.address,
            'token_address': self.token_address,
            'token_symbol': self.token_symbol,
            'token_name': self.token_name,
            'token_decimals': self.token_decimals,
            'token_type': self.token_type,
            'token_type_enum': self.token_type_enum.value,
            'is_native': self.is_native,
            'is_stablecoin': self.is_stablecoin,
            'is_nft': self.is_nft,
            'balance': self.balance,
            'balance_usd': self.balance_usd,
            'price_usd': self.price_usd,
            'has_value': self.has_value,
            'venue': 'covalent',
        }

@dataclass
class CovalentPortfolio:
    """Aggregated portfolio analysis."""
    timestamp: datetime
    address: str
    chain: str
    total_value_usd: float
    token_count: int
    top_holding_symbol: Optional[str]
    top_holding_pct: Optional[float]
    stablecoin_pct: float = 0
    native_pct: float = 0
    
    @property
    def portfolio_tier(self) -> PortfolioTier:
        """Classify portfolio tier by value."""
        val = self.total_value_usd
        if val > 10_000_000:
            return PortfolioTier.WHALE
        elif val > 1_000_000:
            return PortfolioTier.LARGE
        elif val > 100_000:
            return PortfolioTier.MEDIUM
        elif val > 10_000:
            return PortfolioTier.SMALL
        else:
            return PortfolioTier.MICRO
    
    @property
    def is_whale(self) -> bool:
        """Check if whale portfolio."""
        return self.portfolio_tier == PortfolioTier.WHALE
    
    @property
    def is_concentrated(self) -> bool:
        """Check if portfolio is concentrated (>50% in one asset)."""
        return self.top_holding_pct is not None and self.top_holding_pct > 50
    
    @property
    def is_stablecoin_heavy(self) -> bool:
        """Check if >50% in stablecoins."""
        return self.stablecoin_pct > 50
    
    @property
    def diversification_score(self) -> float:
        """Simple diversification score (inverse of concentration)."""
        if self.top_holding_pct:
            return 100 - self.top_holding_pct
        return 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'address': self.address,
            'chain': self.chain,
            'total_value_usd': self.total_value_usd,
            'portfolio_tier': self.portfolio_tier.value,
            'is_whale': self.is_whale,
            'token_count': self.token_count,
            'top_holding_symbol': self.top_holding_symbol,
            'top_holding_pct': self.top_holding_pct,
            'is_concentrated': self.is_concentrated,
            'stablecoin_pct': self.stablecoin_pct,
            'is_stablecoin_heavy': self.is_stablecoin_heavy,
            'native_pct': self.native_pct,
            'diversification_score': self.diversification_score,
            'venue': 'covalent',
        }

@dataclass
class CovalentTransaction:
    """Transaction from Covalent."""
    timestamp: datetime
    chain: str
    tx_hash: str
    block_height: int
    from_address: str
    to_address: Optional[str]
    value: float
    value_usd: Optional[float]
    gas_used: int
    gas_price: float
    gas_quote: Optional[float]
    successful: bool
    
    @property
    def gas_cost_usd(self) -> float:
        """Gas cost in USD."""
        return self.gas_quote or 0
    
    @property
    def is_contract_creation(self) -> bool:
        """Check if contract creation."""
        return self.to_address is None
    
    @property
    def is_high_value(self) -> bool:
        """Check if high value transaction (>$10K)."""
        return self.value_usd is not None and self.value_usd > 10_000
    
    @property
    def is_whale_tx(self) -> bool:
        """Check if whale transaction (>$100K)."""
        return self.value_usd is not None and self.value_usd > 100_000
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'chain': self.chain,
            'tx_hash': self.tx_hash,
            'block_height': self.block_height,
            'from_address': self.from_address,
            'to_address': self.to_address,
            'value': self.value,
            'value_usd': self.value_usd,
            'gas_used': self.gas_used,
            'gas_price': self.gas_price,
            'gas_cost_usd': self.gas_cost_usd,
            'successful': self.successful,
            'is_contract_creation': self.is_contract_creation,
            'is_high_value': self.is_high_value,
            'is_whale_tx': self.is_whale_tx,
            'venue': 'covalent',
        }

@dataclass
class CovalentDEXPool:
    """DEX pool from Covalent."""
    timestamp: datetime
    chain: str
    dex: str
    pool_address: str
    token0_symbol: str
    token0_address: Optional[str]
    token1_symbol: str
    token1_address: Optional[str]
    tvl_usd: float
    volume_24h_usd: float
    fee_24h_usd: float
    
    @property
    def pair(self) -> str:
        """Trading pair string."""
        return f"{self.token0_symbol}/{self.token1_symbol}"
    
    @property
    def liquidity_level(self) -> PoolLiquidity:
        """Classify pool liquidity."""
        tvl = self.tvl_usd
        if tvl > 10_000_000:
            return PoolLiquidity.DEEP
        elif tvl > 1_000_000:
            return PoolLiquidity.HIGH
        elif tvl > 100_000:
            return PoolLiquidity.MEDIUM
        elif tvl > 10_000:
            return PoolLiquidity.LOW
        else:
            return PoolLiquidity.THIN
    
    @property
    def is_deep_liquidity(self) -> bool:
        """Check if deep liquidity."""
        return self.liquidity_level == PoolLiquidity.DEEP
    
    @property
    def volume_tvl_ratio(self) -> float:
        """Volume to TVL ratio (capital efficiency)."""
        return self.volume_24h_usd / self.tvl_usd if self.tvl_usd > 0 else 0
    
    @property
    def fee_apy(self) -> float:
        """Estimated fee APY."""
        return (self.fee_24h_usd * 365 / self.tvl_usd * 100) if self.tvl_usd > 0 else 0
    
    @property
    def is_stablecoin_pair(self) -> bool:
        """Check if stablecoin pair."""
        stables = {'USDT', 'USDC', 'DAI', 'BUSD'}
        return self.token0_symbol.upper() in stables or self.token1_symbol.upper() in stables
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'chain': self.chain,
            'dex': self.dex,
            'pool_address': self.pool_address,
            'pair': self.pair,
            'token0_symbol': self.token0_symbol,
            'token0_address': self.token0_address,
            'token1_symbol': self.token1_symbol,
            'token1_address': self.token1_address,
            'tvl_usd': self.tvl_usd,
            'liquidity_level': self.liquidity_level.value,
            'is_deep_liquidity': self.is_deep_liquidity,
            'volume_24h_usd': self.volume_24h_usd,
            'volume_tvl_ratio': self.volume_tvl_ratio,
            'fee_24h_usd': self.fee_24h_usd,
            'fee_apy': self.fee_apy,
            'is_stablecoin_pair': self.is_stablecoin_pair,
            'venue': 'covalent',
        }

@dataclass
class CovalentTokenHolder:
    """Token holder from Covalent."""
    timestamp: datetime
    chain: str
    token_address: str
    rank: int
    address: str
    balance: float
    balance_usd: float
    pct_of_supply: Optional[float] = None
    
    @property
    def holding_significance(self) -> HoldingSignificance:
        """Classify holding significance."""
        if self.pct_of_supply is None:
            return HoldingSignificance.MINOR
        if self.pct_of_supply > 50:
            return HoldingSignificance.DOMINANT
        elif self.pct_of_supply > 20:
            return HoldingSignificance.MAJOR
        elif self.pct_of_supply > 5:
            return HoldingSignificance.SIGNIFICANT
        elif self.pct_of_supply > 1:
            return HoldingSignificance.MINOR
        else:
            return HoldingSignificance.DUST
    
    @property
    def is_whale(self) -> bool:
        """Check if whale holder (>1% supply or >$1M)."""
        is_pct_whale = self.pct_of_supply is not None and self.pct_of_supply > 1
        is_value_whale = self.balance_usd > 1_000_000
        return is_pct_whale or is_value_whale
    
    @property
    def is_top_10(self) -> bool:
        """Check if top 10 holder."""
        return self.rank <= 10
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'chain': self.chain,
            'token_address': self.token_address,
            'rank': self.rank,
            'address': self.address,
            'balance': self.balance,
            'balance_usd': self.balance_usd,
            'pct_of_supply': self.pct_of_supply,
            'holding_significance': self.holding_significance.value,
            'is_whale': self.is_whale,
            'is_top_10': self.is_top_10,
            'venue': 'covalent',
        }

# =============================================================================
# Main Collector Class
# =============================================================================

# Import BaseCollector
import sys
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from base_collector import BaseCollector

class CovalentCollector(BaseCollector):
    """
    Covalent (GoldenRush) on-chain data collector.

    validated implementation providing comprehensive blockchain
    data across 100+ chains including balances, transactions, and DEX data.

    Features:
        - Token balances across all chains
        - Transaction history
        - DEX pools and swaps
        - NFT holdings
        - Historical prices
        - Token holders

    Example:
        >>> collector = CovalentCollector(api_key='your-key')
        >>> try:
        ... balances = await collector.get_token_balances('ethereum', '0x...')
        ... pools = await collector.fetch_dex_pool_data('ethereum', 'uniswap_v3')
        ... finally:
        ... await collector.close()
    
    Attributes:
        VENUE: 'covalent'
        VENUE_TYPE: 'onchain'
    """
    
    VENUE = 'covalent'
    VENUE_TYPE = 'onchain'
    BASE_URL = 'https://api.covalenthq.com/v1'
    
    # Chain name mapping (GoldRush API uses these chain names)
    # Note: xy=k DEX endpoints may have limited availability
    CHAINS = {
        'ethereum': 'eth-mainnet',
        'polygon': 'matic-mainnet',
        'arbitrum': 'arbitrum-mainnet',
        'optimism': 'optimism-mainnet',
        'base': 'base-mainnet',
        'bsc': 'bsc-mainnet',
        'avalanche': 'avalanche-mainnet',
        'fantom': 'fantom-mainnet',
        'gnosis': 'gnosis-mainnet',
        'zksync': 'zksync-mainnet',
        'scroll': 'scroll-mainnet',
        'linea': 'linea-mainnet',
    }
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        rate_limit: int = 2,
    ):
        """
        Initialize Covalent collector.

        Args:
            config: Configuration dictionary (for compatibility with collection_manager)
            api_key: Covalent API key
            rate_limit: Max requests per second
        """
        config = config or {}
        super().__init__(config)

        # CRITICAL: Set supported data types for dynamic routing
        self.supported_data_types = ['wallet_analytics', 'token_balances', 'on_chain_metrics']
        self.venue = 'covalent'

        # Import VenueType from base_collector
        from ..base_collector import VenueType
        self.venue_type = VenueType.ONCHAIN
        self.requires_auth = True # Requires Covalent API key

        self.api_key = api_key or config.get('api_key') or os.getenv('COVALENT_API_KEY')
        self.rate_limit = config.get('rate_limit', rate_limit)

        if not self.api_key:
            logger.warning("Covalent API key not provided - requests will fail")
        else:
            logger.info("Covalent collector initialized with API key")
        
        self._last_request = 0
        self._min_interval = 1.0 / rate_limit
        
        self.session: Optional[aiohttp.ClientSession] = None
        
        self.stats = {'requests': 0, 'records': 0, 'errors': 0, 'rate_limits': 0}
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            auth_string = base64.b64encode(f"{self.api_key}:".encode()).decode()
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Basic {auth_string}',
            }
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self.session
    
    async def _rate_limit(self):
        """Apply rate limiting."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request
        
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        
        self._last_request = asyncio.get_event_loop().time()
    
    async def _request(self, endpoint: str, params: Optional[Dict] = None, _retry_count: int = 0) -> Optional[Any]:
        """Make API request with rate limiting and error handling."""
        await self._rate_limit()
        
        session = await self._get_session()
        url = f"{self.BASE_URL}/{endpoint}"
        
        self.stats['requests'] += 1
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', data)
                elif response.status == 429:
                    self.stats['rate_limits'] += 1
                    if _retry_count >= 3:
                        logger.error("Covalent rate limit: max retries (3) exceeded")
                        return None
                    wait_time = min(30 * (2 ** _retry_count), 120)
                    logger.warning(f"Covalent rate limit hit, waiting {wait_time}s (retry {_retry_count + 1}/3)")
                    await asyncio.sleep(wait_time)
                    return await self._request(endpoint, params, _retry_count=_retry_count + 1)
                elif response.status == 401:
                    logger.error("Covalent API key invalid")
                    self.stats['errors'] += 1
                    return None
                else:
                    self.stats['errors'] += 1
                    text = await response.text()
                    logger.error(f"Covalent API error {response.status}: {text[:200]}")
                    return None
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Covalent request error: {e}")
            return None
    
    def _get_chain_id(self, chain: str) -> str:
        """Convert chain name to chain ID."""
        return self.CHAINS.get(chain.lower(), chain)
    
    async def get_token_balances(
        self,
        chain: str,
        address: str,
        quote_currency: str = 'USD',
        nft: bool = False,
    ) -> Optional[Dict]:
        """Get token balances for an address."""
        chain_id = self._get_chain_id(chain)
        params = {'quote-currency': quote_currency, 'nft': str(nft).lower()}
        return await self._request(f'{chain_id}/address/{address}/balances_v2/', params)
    
    async def get_transactions(
        self,
        chain: str,
        address: str,
        page_size: int = 100,
        page_number: int = 0,
        quote_currency: str = 'USD',
    ) -> Optional[Dict]:
        """Get transactions for an address."""
        chain_id = self._get_chain_id(chain)
        params = {
            'quote-currency': quote_currency,
            'page-size': page_size,
            'page-number': page_number,
        }
        return await self._request(f'{chain_id}/address/{address}/transactions_v3/', params)
    
    async def get_token_holders(
        self,
        chain: str,
        token_address: str,
        page_size: int = 100,
        page_number: int = 0,
    ) -> Optional[Dict]:
        """Get token holders."""
        chain_id = self._get_chain_id(chain)
        params = {'page-size': page_size, 'page-number': page_number}
        return await self._request(f'{chain_id}/tokens/{token_address}/token_holders_v2/', params)
    
    async def get_dex_pools(
        self,
        chain: str,
        dex_name: str = 'uniswap_v3',
        page_size: int = 100,
        page_number: int = 0,
    ) -> Optional[Dict]:
        """Get DEX liquidity pools."""
        chain_id = self._get_chain_id(chain)
        params = {'page-size': page_size, 'page-number': page_number}
        return await self._request(f'{chain_id}/xy=k/{dex_name}/pools/', params)
    
    async def get_dex_ecosystem_stats(self, chain: str, dex_name: str) -> Optional[Dict]:
        """Get DEX ecosystem statistics."""
        chain_id = self._get_chain_id(chain)
        return await self._request(f'{chain_id}/xy=k/{dex_name}/ecosystem/')
    
    async def get_historical_prices(
        self,
        chain: str,
        quote_currency: str,
        contract_address: str,
        from_date: str,
        to_date: str,
    ) -> Optional[Dict]:
        """Get historical token prices."""
        chain_id = self._get_chain_id(chain)
        params = {'from': from_date, 'to': to_date}
        return await self._request(
            f'pricing/historical_by_addresses_v2/{chain_id}/{quote_currency}/{contract_address}/',
            params,
        )
    
    async def _fetch_single_wallet_analysis(self, chain: str, address: str) -> Optional[Dict]:
        """Analyze a single wallet address."""
        try:
            logger.info(f"Analyzing wallet {address[:10]}...")

            balances = await self.get_token_balances(chain, address)

            if balances and 'items' in balances:
                total_value = 0
                tokens = []
                stablecoin_value = 0
                native_value = 0

                for item in balances['items']:
                    token_value = item.get('quote', 0) or 0
                    total_value += token_value

                    symbol = item.get('contract_ticker_symbol', '')

                    if symbol.upper() in {'USDT', 'USDC', 'DAI', 'BUSD'}:
                        stablecoin_value += token_value

                    if item.get('native_token', False):
                        native_value += token_value

                    if token_value > 0:
                        tokens.append({
                            'symbol': symbol,
                            'value': token_value,
                            'pct': (token_value / total_value * 100) if total_value > 0 else 0,
                        })

                tokens.sort(key=lambda x: x['value'], reverse=True)

                p = CovalentPortfolio(
                    timestamp=datetime.now(timezone.utc),
                    address=address,
                    chain=chain,
                    total_value_usd=total_value,
                    token_count=len(tokens),
                    top_holding_symbol=tokens[0]['symbol'] if tokens else None,
                    top_holding_pct=tokens[0]['pct'] if tokens else None,
                    stablecoin_pct=(stablecoin_value / total_value * 100) if total_value > 0 else 0,
                    native_pct=(native_value / total_value * 100) if total_value > 0 else 0
                )
                return p.to_dict()
        except Exception as e:
            logger.error(f"Error analyzing wallet {address[:10]}: {e}")
        return None

    async def fetch_wallet_analysis(self, chain: str, addresses: List[str]) -> pd.DataFrame:
        """Analyze multiple wallet addresses."""
        # PARALLELIZED: Analyze all wallets concurrently
        tasks = [self._fetch_single_wallet_analysis(chain, address) for address in addresses]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        records = [r for r in results if isinstance(r, dict)]
        self.stats['records'] += len(records)
        return pd.DataFrame(records)
    
    async def fetch_dex_pool_data(
        self,
        chain: str,
        dex_name: str = 'uniswap_v3',
        top_n: int = 50,
    ) -> pd.DataFrame:
        """Fetch top DEX pools."""
        records = []
        
        logger.info(f"Fetching {dex_name} pools on {chain}")
        
        pools = await self.get_dex_pools(chain, dex_name, page_size=top_n)
        
        if pools and 'items' in pools:
            for pool in pools['items']:
                p = CovalentDEXPool(
                    timestamp=datetime.now(timezone.utc),
                    chain=chain,
                    dex=dex_name,
                    pool_address=pool.get('exchange', ''),
                    token0_symbol=pool.get('token_0', {}).get('contract_ticker_symbol', ''),
                    token0_address=pool.get('token_0', {}).get('contract_address'),
                    token1_symbol=pool.get('token_1', {}).get('contract_ticker_symbol', ''),
                    token1_address=pool.get('token_1', {}).get('contract_address'),
                    tvl_usd=pool.get('total_liquidity_quote', 0) or 0,
                    volume_24h_usd=pool.get('volume_24h_quote', 0) or 0,
                    fee_24h_usd=pool.get('fee_24h_quote', 0) or 0
                )
                records.append(p.to_dict())
        
        self.stats['records'] += len(records)
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('tvl_usd', ascending=False).reset_index(drop=True)
        
        return df
    
    async def fetch_token_holders_analysis(
        self,
        chain: str,
        token_address: str,
        top_n: int = 100
    ) -> pd.DataFrame:
        """Fetch and analyze token holders."""
        data = await self.get_token_holders(chain, token_address, page_size=top_n)
        
        records = []
        if data and 'items' in data:
            for i, holder in enumerate(data['items']):
                h = CovalentTokenHolder(
                    timestamp=datetime.now(timezone.utc),
                    chain=chain,
                    token_address=token_address,
                    rank=i + 1,
                    address=holder.get('address', ''),
                    balance=float(holder.get('balance', 0)),
                    balance_usd=float(holder.get('quote', 0) or 0),
                    pct_of_supply=holder.get('percent_supply')
                )
                records.append(h.to_dict())
        
        self.stats['records'] += len(records)
        return pd.DataFrame(records)
    
    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info(f"Covalent session closed. Stats: {self.stats}")
    
    # Well-known addresses for token balance tracking (whales, protocols, exchanges)
    WELL_KNOWN_ADDRESSES = {
        'ethereum': [
            '0x28C6c06298d514Db089934071355E5743bf21d60', # Binance Hot Wallet
            '0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549', # Binance 15
            '0xDFd5293D8e347dFe59E90eFd55b2956a1343963d', # Bitfinex
            '0x1DB92e2EeBe7429B948F6C26b7AdAdE79E73eB6E', # Coinbase Custody
        ],
        'polygon': [
            '0x28C6c06298d514Db089934071355E5743bf21d60', # Binance
        ],
        'arbitrum': [
            '0x28C6c06298d514Db089934071355E5743bf21d60', # Binance
        ],
    }

    async def collect_token_balances(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect token balances - wraps get_token_balances().

        Standardized method name for collection manager compatibility.
        If no addresses provided, uses well-known whale/exchange addresses.
        """
        try:
            chain = kwargs.get('chain', 'ethereum')
            addresses = kwargs.get('addresses', [])

            # Use well-known addresses if none provided
            if not addresses:
                addresses = self.WELL_KNOWN_ADDRESSES.get(chain, self.WELL_KNOWN_ADDRESSES.get('ethereum', []))
                logger.info(f"Covalent: Using {len(addresses)} well-known addresses for {chain}")

            if not addresses:
                logger.warning("Covalent: No addresses available for token balance collection")
                return pd.DataFrame()

            # PARALLELIZED: Collect balances for all addresses concurrently
            async def _collect_single_address_balances(address: str) -> List[Dict]:
                try:
                    result = await self.get_token_balances(chain=chain, address=address)
                    if result and 'items' in result:
                        records = []
                        for item in result['items']:
                            # Filter to requested symbols if provided (or include all)
                            symbol = item.get('contract_ticker_symbol', '')
                            if symbols and symbol and symbol.upper() not in [s.upper() for s in symbols]:
                                continue

                            # Handle None decimals safely
                            decimals = item.get('contract_decimals')
                            if decimals is None:
                                decimals = 18 # Default to 18 decimals
                            balance_raw = item.get('balance', 0) or 0
                            try:
                                balance = float(balance_raw) / (10 ** int(decimals))
                            except (TypeError, ValueError):
                                balance = 0

                            records.append({
                                'address': address,
                                'chain': chain,
                                'symbol': symbol,
                                'contract_address': item.get('contract_address'),
                                'balance': balance,
                                'quote_usd': item.get('quote'),
                                'timestamp': datetime.now(timezone.utc),
                                'venue': self.VENUE,
                                'venue_type': self.VENUE_TYPE
                            })
                        return records
                except Exception as e:
                    logger.warning(f"Covalent: Failed to get balances for {address[:10]}...: {e}")
                return []

            tasks = [_collect_single_address_balances(address) for address in addresses[:5]]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_records = []
            for result in results:
                if isinstance(result, list):
                    all_records.extend(result)

            logger.info(f"Covalent: Collected {len(all_records)} token balance records")
            return pd.DataFrame(all_records)

        except Exception as e:
            logger.error(f"Covalent collect_token_balances error: {e}")
            return pd.DataFrame()

    async def collect_on_chain_metrics(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect on-chain metrics - aggregates token holder and transaction data.

        Standardized method name for collection manager compatibility.
        """
        try:
            chain = kwargs.get('chain', 'ethereum')

            logger.info(f"Covalent: Collecting on_chain_metrics for {chain}")

            # Collect token holder counts for major tokens
            token_addresses = {
                'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
                'USDC': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
                'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
                'DAI': '0x6B175474E89094C44Da98b954EesdffdD3A92Ca',
                'UNI': '0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984',
                'LINK': '0x514910771AF9Ca656af840dff83E8264EcF986CA',
            }

            # PARALLELIZED: Collect holder counts for all symbols concurrently
            async def _collect_single_symbol_holders(symbol: str) -> Optional[Dict]:
                token_addr = token_addresses.get(symbol.upper())
                if not token_addr:
                    return None

                try:
                    holders = await self.get_token_holders(chain=chain, token_address=token_addr)
                    if holders and 'items' in holders:
                        holder_count = len(holders['items'])
                        return {
                            'symbol': symbol.upper(),
                            'chain': chain,
                            'metric': 'holder_count',
                            'value': holder_count,
                            'token_address': token_addr,
                            'timestamp': datetime.now(timezone.utc),
                            'venue': self.VENUE,
                            'venue_type': self.VENUE_TYPE
                        }
                except Exception as e:
                    logger.warning(f"Covalent: Failed to get holders for {symbol}: {e}")
                return None

            tasks = [_collect_single_symbol_holders(symbol) for symbol in symbols[:5]]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_records = [r for r in results if isinstance(r, dict)]

            logger.info(f"Covalent: Collected {len(all_records)} on_chain_metrics records")
            return pd.DataFrame(all_records)

        except Exception as e:
            logger.error(f"Covalent collect_on_chain_metrics error: {e}")
            return pd.DataFrame()

    async def collect_wallet_analytics(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect wallet analytics - wraps fetch_wallet_analysis().

        Standardized method name for collection manager compatibility.
        """
        try:
            chain = kwargs.get('chain', 'ethereum')
            addresses = kwargs.get('addresses', [])

            if not addresses:
                logger.warning("Covalent collect_wallet_analytics requires 'addresses' kwarg")
                return pd.DataFrame()

            return await self.fetch_wallet_analysis(chain=chain, addresses=addresses)

        except Exception as e:
            logger.error(f"Covalent collect_wallet_analytics error: {e}")
            return pd.DataFrame()

    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.stats.copy()

    async def fetch_funding_rates(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """On-chain data doesn't have funding rates."""
        logger.info("Covalent: No funding rates (on-chain data)")
        return pd.DataFrame()

    async def fetch_ohlcv(
        self, symbols: List[str], timeframe: str,
        start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Covalent doesn't provide historical OHLCV directly."""
        logger.info("Covalent: fetch_ohlcv not supported - use get_historical_prices() instead")
        return pd.DataFrame()

    @classmethod
    def get_supported_chains(cls) -> List[str]:
        """Get list of supported chains."""
        return list(cls.CHAINS.keys())
    
    @classmethod
    def get_supported_dexes(cls) -> List[str]:
        """Get list of supported DEX protocols."""
        return [d.value for d in DEXProtocol]

async def test_covalent_collector():
    """Test Covalent collector functionality."""
    collector = CovalentCollector(rate_limit=3)
    
    try:
        print("=" * 60)
        print("Covalent Collector Test")
        print("=" * 60)
        print(f"\nSupported chains: {CovalentCollector.get_supported_chains()}")
        print(f"Supported DEXes: {CovalentCollector.get_supported_dexes()}")
        print(f"\nStats: {collector.get_collection_stats()}")
    finally:
        await collector.close()

if __name__ == '__main__':
    asyncio.run(test_covalent_collector())