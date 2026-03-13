"""
DEXScreener Data Collector

validated collector for DEXScreener - a multi-chain DEX aggregator
providing real-time pair discovery, prices, and liquidity across 80+ chains.

Key Features:
    - Token/pair search across all supported chains
    - Real-time prices and liquidity data
    - Trending tokens and new pair detection
    - Cross-chain price comparison for arbitrage
    - Pool statistics and transaction metrics

Supported Chains (80+):
    Ethereum, BSC, Polygon, Arbitrum, Optimism, Avalanche, Base, Solana,
    Fantom, zkSync, Linea, Scroll, Mantle, Blast, Manta, and many more.

API Documentation: https://docs.dexscreener.com/
Rate Limits: ~300 requests/minute (FREE, no API key required)

Statistical Arbitrage Applications:
    - Cross-chain price arbitrage detection
    - New pair discovery for early opportunities
    - Liquidity depth analysis for execution
    - Token momentum via buy/sell ratio tracking

Version: 2.0.0
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
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

class Chain(Enum):
    """Supported blockchain networks."""
    ETHEREUM = 'ethereum'
    BSC = 'bsc'
    POLYGON = 'polygon'
    ARBITRUM = 'arbitrum'
    OPTIMISM = 'optimism'
    AVALANCHE = 'avalanche'
    BASE = 'base'
    SOLANA = 'solana'
    FANTOM = 'fantom'
    CRONOS = 'cronos'
    GNOSIS = 'gnosis'
    ZKSYNC = 'zksync'
    LINEA = 'linea'
    SCROLL = 'scroll'
    MANTLE = 'mantle'
    BLAST = 'blast'
    MODE = 'mode'
    MANTA = 'manta'
    SEI = 'sei'
    SUI = 'sui'
    APTOS = 'aptos'
    TON = 'ton'
    NEAR = 'near'
    TRON = 'tron'

class PriceChangeTimeframe(Enum):
    """Price change timeframes."""
    M5 = 'm5' # 5 minutes
    H1 = 'h1' # 1 hour
    H6 = 'h6' # 6 hours
    H24 = 'h24' # 24 hours

class MomentumSignal(Enum):
    """Token momentum classification."""
    STRONG_BUY = 'strong_buy' # Buy/sell ratio > 2.0
    BUY = 'buy' # Buy/sell ratio 1.5-2.0
    NEUTRAL = 'neutral' # Buy/sell ratio 0.7-1.5
    SELL = 'sell' # Buy/sell ratio 0.5-0.7
    STRONG_SELL = 'strong_sell' # Buy/sell ratio < 0.5

class LiquidityTier(Enum):
    """Liquidity classification tiers."""
    WHALE = 'whale' # > $10M liquidity
    HIGH = 'high' # $1M - $10M
    MEDIUM = 'medium' # $100K - $1M
    LOW = 'low' # $10K - $100K
    MICRO = 'micro' # < $10K

class VolumeActivity(Enum):
    """Volume activity classification."""
    VERY_HIGH = 'very_high' # > 100% of liquidity
    HIGH = 'high' # 50-100% of liquidity
    MODERATE = 'moderate' # 10-50% of liquidity
    LOW = 'low' # 1-10% of liquidity
    DORMANT = 'dormant' # < 1% of liquidity

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class DEXPair:
    """DEX trading pair data."""
    timestamp: datetime
    pair_address: str
    chain: str
    dex_id: str
    
    # Token information
    base_token_address: str
    base_token_symbol: str
    base_token_name: str
    quote_token_address: str
    quote_token_symbol: str
    quote_token_name: str
    
    # Prices
    price_native: float
    price_usd: float
    
    # Liquidity
    liquidity_usd: float
    liquidity_base: float
    liquidity_quote: float
    
    # Volume (24h)
    volume_24h_usd: float
    volume_6h_usd: float
    volume_1h_usd: float
    volume_5m_usd: float
    
    # Price changes
    price_change_24h: float
    price_change_6h: float
    price_change_1h: float
    price_change_5m: float
    
    # Transaction counts
    txns_24h_buys: int
    txns_24h_sells: int
    
    # Market metrics
    fdv: Optional[float] = None
    market_cap: Optional[float] = None
    created_at: Optional[datetime] = None
    
    @property
    def pair_symbol(self) -> str:
        """Trading pair symbol."""
        return f"{self.base_token_symbol}/{self.quote_token_symbol}"
    
    @property
    def buy_sell_ratio(self) -> float:
        """Calculate buy/sell transaction ratio."""
        if self.txns_24h_sells > 0:
            return self.txns_24h_buys / self.txns_24h_sells
        return float('inf') if self.txns_24h_buys > 0 else 1.0
    
    @property
    def momentum_signal(self) -> MomentumSignal:
        """Classify momentum based on buy/sell ratio."""
        ratio = self.buy_sell_ratio
        if ratio > 2.0:
            return MomentumSignal.STRONG_BUY
        elif ratio > 1.5:
            return MomentumSignal.BUY
        elif ratio > 0.7:
            return MomentumSignal.NEUTRAL
        elif ratio > 0.5:
            return MomentumSignal.SELL
        else:
            return MomentumSignal.STRONG_SELL
    
    @property
    def liquidity_tier(self) -> LiquidityTier:
        """Classify liquidity tier."""
        if self.liquidity_usd >= 10_000_000:
            return LiquidityTier.WHALE
        elif self.liquidity_usd >= 1_000_000:
            return LiquidityTier.HIGH
        elif self.liquidity_usd >= 100_000:
            return LiquidityTier.MEDIUM
        elif self.liquidity_usd >= 10_000:
            return LiquidityTier.LOW
        else:
            return LiquidityTier.MICRO
    
    @property
    def volume_activity(self) -> VolumeActivity:
        """Classify volume activity relative to liquidity."""
        if self.liquidity_usd <= 0:
            return VolumeActivity.DORMANT
        
        ratio = self.volume_24h_usd / self.liquidity_usd
        
        if ratio > 1.0:
            return VolumeActivity.VERY_HIGH
        elif ratio > 0.5:
            return VolumeActivity.HIGH
        elif ratio > 0.1:
            return VolumeActivity.MODERATE
        elif ratio > 0.01:
            return VolumeActivity.LOW
        else:
            return VolumeActivity.DORMANT
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'pair_address': self.pair_address,
            'chain': self.chain,
            'dex_id': self.dex_id,
            'pair_symbol': self.pair_symbol,
            'base_token_address': self.base_token_address,
            'base_token_symbol': self.base_token_symbol,
            'base_token_name': self.base_token_name,
            'quote_token_address': self.quote_token_address,
            'quote_token_symbol': self.quote_token_symbol,
            'quote_token_name': self.quote_token_name,
            'price_native': self.price_native,
            'price_usd': self.price_usd,
            'liquidity_usd': self.liquidity_usd,
            'liquidity_base': self.liquidity_base,
            'liquidity_quote': self.liquidity_quote,
            'volume_24h_usd': self.volume_24h_usd,
            'volume_6h_usd': self.volume_6h_usd,
            'volume_1h_usd': self.volume_1h_usd,
            'volume_5m_usd': self.volume_5m_usd,
            'price_change_24h': self.price_change_24h,
            'price_change_6h': self.price_change_6h,
            'price_change_1h': self.price_change_1h,
            'price_change_5m': self.price_change_5m,
            'txns_24h_buys': self.txns_24h_buys,
            'txns_24h_sells': self.txns_24h_sells,
            'buy_sell_ratio': self.buy_sell_ratio,
            'momentum_signal': self.momentum_signal.value,
            'liquidity_tier': self.liquidity_tier.value,
            'volume_activity': self.volume_activity.value,
            'fdv': self.fdv,
            'market_cap': self.market_cap,
            'created_at': self.created_at,
        }

@dataclass
class TokenInfo:
    """Detailed token information."""
    timestamp: datetime
    address: str
    name: str
    symbol: str
    chain: str
    
    # Pricing
    price_usd: float
    price_native: float
    
    # Market data
    liquidity_usd: float
    volume_24h: float
    price_change_24h: float
    
    # Valuation
    fdv: Optional[float] = None
    market_cap: Optional[float] = None
    
    # Pair data
    num_pairs: int = 0
    top_pair_address: Optional[str] = None
    top_pair_dex: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'address': self.address,
            'name': self.name,
            'symbol': self.symbol,
            'chain': self.chain,
            'price_usd': self.price_usd,
            'price_native': self.price_native,
            'liquidity_usd': self.liquidity_usd,
            'volume_24h': self.volume_24h,
            'price_change_24h': self.price_change_24h,
            'fdv': self.fdv,
            'market_cap': self.market_cap,
            'num_pairs': self.num_pairs,
            'top_pair_address': self.top_pair_address,
            'top_pair_dex': self.top_pair_dex,
        }

@dataclass
class ArbitrageOpportunity:
    """Cross-chain arbitrage opportunity."""
    timestamp: datetime
    symbol: str
    
    # Buy side (lower price)
    buy_chain: str
    buy_price: float
    buy_liquidity: float
    buy_dex: str
    buy_pair_address: str
    
    # Sell side (higher price)
    sell_chain: str
    sell_price: float
    sell_liquidity: float
    sell_dex: str
    sell_pair_address: str
    
    # Opportunity metrics
    spread_pct: float
    spread_usd: float
    max_trade_size_usd: float # Limited by lower liquidity
    
    @property
    def is_executable(self) -> bool:
        """Check if opportunity is worth executing (accounting for fees)."""
        # Typically need >1% spread to cover bridge + swap fees
        return self.spread_pct > 1.0 and self.max_trade_size_usd > 1000
    
    @property
    def estimated_profit_pct(self) -> float:
        """Estimate profit after typical fees (~0.6% total)."""
        return self.spread_pct - 0.6 # 0.3% swap + 0.3% bridge (rough estimate)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'buy_chain': self.buy_chain,
            'buy_price': self.buy_price,
            'buy_liquidity': self.buy_liquidity,
            'buy_dex': self.buy_dex,
            'buy_pair_address': self.buy_pair_address,
            'sell_chain': self.sell_chain,
            'sell_price': self.sell_price,
            'sell_liquidity': self.sell_liquidity,
            'sell_dex': self.sell_dex,
            'sell_pair_address': self.sell_pair_address,
            'spread_pct': self.spread_pct,
            'spread_usd': self.spread_usd,
            'max_trade_size_usd': self.max_trade_size_usd,
            'is_executable': self.is_executable,
            'estimated_profit_pct': self.estimated_profit_pct,
        }

# =============================================================================
# Collector Class
# =============================================================================

class DEXScreenerCollector(BaseCollector):
    """
    DEXScreener data collector for multi-chain DEX pair discovery.
    
    Provides comprehensive DEX data across 80+ chains:
    - Token and pair search functionality
    - Real-time prices and liquidity
    - Transaction metrics (buys/sells)
    - Cross-chain price comparison
    - New pair and trending token detection
    
    FREE API - No authentication required.
    
    Attributes:
        VENUE: Venue identifier ('dexscreener')
        VENUE_TYPE: Venue classification ('DEX')
        BASE_URL: API endpoint
        SUPPORTED_CHAINS: Dict of supported chain identifiers
    
    Example:
        >>> config = {'rate_limit': 60}
        >>> async with DEXScreenerCollector(config) as collector:
        ... pairs = await collector.search_pairs('PEPE')
        ... prices = await collector.get_multi_chain_prices('ETH')
        ... arbs = await collector.find_arbitrage_opportunities(['UNI', 'AAVE'])
    """
    
    VENUE = 'dexscreener'
    VENUE_TYPE = 'DEX'
    BASE_URL = 'https://api.dexscreener.com'
    
    # Default rate limit (conservative: 2 req/sec = 120/min)
    DEFAULT_RATE_LIMIT = 60
    
    # Supported chains mapping
    SUPPORTED_CHAINS = {
        'ethereum': 'ethereum', 'bsc': 'bsc', 'polygon': 'polygon',
        'arbitrum': 'arbitrum', 'optimism': 'optimism', 'avalanche': 'avalanche',
        'base': 'base', 'solana': 'solana', 'fantom': 'fantom',
        'cronos': 'cronos', 'gnosis': 'gnosis', 'zksync': 'zksync',
        'linea': 'linea', 'scroll': 'scroll', 'mantle': 'mantle',
        'blast': 'blast', 'mode': 'mode', 'manta': 'manta',
        'sei': 'sei', 'sui': 'sui', 'aptos': 'aptos',
        'ton': 'ton', 'near': 'near', 'tron': 'tron',
        'aurora': 'aurora', 'moonbeam': 'moonbeam', 'celo': 'celo',
        'metis': 'metis', 'boba': 'boba', 'harmony': 'harmony',
        'okc': 'okc', 'kcc': 'kcc', 'moonriver': 'moonriver',
        'fuse': 'fuse', 'evmos': 'evmos', 'klaytn': 'klaytn',
    }
    
    # Major chains for default queries
    MAJOR_CHAINS = ['ethereum', 'arbitrum', 'optimism', 'base', 'polygon', 'bsc']
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DEXScreener collector.
        
        Args:
            config: Configuration dictionary with optional keys:
                - rate_limit: Requests per minute (default: 60)
                - timeout: Request timeout in seconds (default: 30)
        """
        super().__init__(config or {})
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = get_shared_rate_limiter(
            'dexscreener',
            rate=self.config.get('rate_limit', self.DEFAULT_RATE_LIMIT),
            per=60.0,
            burst=30
        )
        self.retry_handler = RetryHandler(
            max_retries=self.config.get('max_retries', 3),
            base_delay=1.0,
            max_delay=30.0
        )
        self._timeout = self.config.get('timeout', 30)
        
        # Collection statistics
        self.collection_stats = {
            'records_collected': 0,
            'api_calls': 0,
            'errors': 0,
            'rate_limit_hits': 0,
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with connection pooling."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            # SPEEDUP: Aggressive connection pooling for high-throughput collection
            connector = aiohttp.TCPConnector(
                limit=50, # Total connection pool size (was 10)
                limit_per_host=15, # Per-host connections (was 5)
                ttl_dns_cache=300, # DNS cache TTL in seconds
                force_close=False, # Keep-alive connections
                enable_cleanup_closed=True
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    'Accept': 'application/json',
                    'User-Agent': 'CryptoStatArb/2.0'
                }
            )
        return self.session
    
    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Make request to DEXScreener API with rate limiting and retry.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            JSON response or None on error
        """
        await self.rate_limiter.acquire()
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        
        self.collection_stats['api_calls'] += 1
        
        async def _execute_request():
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 429:
                    self.collection_stats['rate_limit_hits'] += 1
                    logger.warning("DEXScreener rate limited, backing off")
                    raise aiohttp.ClientResponseError(
                        resp.request_info, resp.history,
                        status=429, message="Rate limited"
                    )
                else:
                    logger.warning(f"DEXScreener API error: {resp.status}")
                    return None
        
        try:
            return await self.retry_handler.execute(_execute_request)
        except Exception as e:
            logger.error(f"DEXScreener request failed: {e}")
            self.collection_stats['errors'] += 1
            return None
    
    # =========================================================================
    # Pair Search Methods
    # =========================================================================
    
    async def search_pairs(
        self,
        query: str,
        min_liquidity: float = 0,
        chains: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Search for pairs by token name, symbol, or address.
        
        Args:
            query: Search query (token name, symbol, or address)
            min_liquidity: Minimum liquidity filter in USD
            chains: Optional list of chains to filter
            
        Returns:
            DataFrame with matching pairs sorted by liquidity
        """
        logger.info(f"Searching pairs for: {query}")
        
        data = await self._request("/latest/dex/search", {"q": query})
        
        if not data or 'pairs' not in data:
            return pd.DataFrame()
        
        records = []
        for pair in data['pairs']:
            parsed = self._parse_pair(pair)
            
            # Apply filters
            if parsed['liquidity_usd'] < min_liquidity:
                continue
            if chains:
                chain_ids = [self.SUPPORTED_CHAINS.get(c, c) for c in chains]
                if parsed['chain'] not in chain_ids:
                    continue
            
            records.append(parsed)
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            df = df.sort_values('liquidity_usd', ascending=False).reset_index(drop=True)
            df['liquidity_rank'] = df.index + 1
        
        self.collection_stats['records_collected'] += len(df)
        return df
    
    async def get_pairs_by_token(
        self,
        token_addresses: List[str],
        min_liquidity: float = 0
    ) -> pd.DataFrame:
        """
        Get all pairs for specific token addresses.
        
        Args:
            token_addresses: List of token contract addresses (max 30)
            min_liquidity: Minimum liquidity filter
            
        Returns:
            DataFrame with all pairs containing these tokens
        """
        logger.info(f"Fetching pairs for {len(token_addresses)} tokens")
        
        # API accepts comma-separated addresses (max 30)
        addresses = ','.join(token_addresses[:30])
        
        data = await self._request(f"/latest/dex/tokens/{addresses}")
        
        if not data or 'pairs' not in data:
            return pd.DataFrame()
        
        records = []
        for pair in data['pairs']:
            parsed = self._parse_pair(pair)
            if parsed['liquidity_usd'] >= min_liquidity:
                records.append(parsed)
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            df = df.sort_values('liquidity_usd', ascending=False).reset_index(drop=True)
        
        return df
    
    async def get_pairs_by_chain(
        self,
        chain: str,
        pair_addresses: List[str]
    ) -> pd.DataFrame:
        """
        Get specific pairs on a chain.
        
        Args:
            chain: Chain identifier
            pair_addresses: List of pair addresses (max 30)
            
        Returns:
            DataFrame with pair data
        """
        if chain not in self.SUPPORTED_CHAINS:
            logger.warning(f"Unsupported chain: {chain}")
            return pd.DataFrame()
        
        chain_id = self.SUPPORTED_CHAINS[chain]
        addresses = ','.join(pair_addresses[:30])
        
        data = await self._request(f"/latest/dex/pairs/{chain_id}/{addresses}")
        
        if not data or 'pairs' not in data:
            return pd.DataFrame()
        
        records = [self._parse_pair(pair) for pair in data['pairs']]
        return pd.DataFrame(records)
    
    # =========================================================================
    # Token Information Methods
    # =========================================================================
    
    async def get_token_info(
        self,
        chain: str,
        token_address: str
    ) -> Optional[TokenInfo]:
        """
        Get detailed information about a token.
        
        Args:
            chain: Chain identifier
            token_address: Token contract address
            
        Returns:
            TokenInfo dataclass or None if not found
        """
        chain_id = self.SUPPORTED_CHAINS.get(chain, chain)
        
        data = await self._request(f"/latest/dex/tokens/{token_address}")
        
        if not data or 'pairs' not in data:
            return None
        
        # Filter pairs on specified chain
        chain_pairs = [p for p in data['pairs'] if p.get('chainId') == chain_id]
        
        if not chain_pairs:
            return None
        
        # Get most liquid pair
        best_pair = max(chain_pairs, key=lambda p: p.get('liquidity', {}).get('usd', 0))
        
        # Determine which token is our target
        base_token = best_pair.get('baseToken', {})
        quote_token = best_pair.get('quoteToken', {})
        
        if base_token.get('address', '').lower() == token_address.lower():
            token = base_token
        else:
            token = quote_token
        
        return TokenInfo(
            timestamp=datetime.utcnow(),
            address=token.get('address', ''),
            name=token.get('name', ''),
            symbol=token.get('symbol', ''),
            chain=chain_id,
            price_usd=float(best_pair.get('priceUsd', 0) or 0),
            price_native=float(best_pair.get('priceNative', 0) or 0),
            liquidity_usd=float(best_pair.get('liquidity', {}).get('usd', 0) or 0),
            volume_24h=float(best_pair.get('volume', {}).get('h24', 0) or 0),
            price_change_24h=float(best_pair.get('priceChange', {}).get('h24', 0) or 0),
            fdv=float(best_pair.get('fdv', 0) or 0) or None,
            market_cap=float(best_pair.get('marketCap', 0) or 0) or None,
            num_pairs=len(chain_pairs),
            top_pair_address=best_pair.get('pairAddress'),
            top_pair_dex=best_pair.get('dexId'),
        )
    
    # =========================================================================
    # Cross-Chain Price Methods
    # =========================================================================
    
    async def get_multi_chain_prices(
        self,
        symbol: str,
        chains: Optional[List[str]] = None,
        min_liquidity: float = 10000
    ) -> pd.DataFrame:
        """
        Get prices for a token across multiple chains.
        
        Useful for finding cross-chain arbitrage opportunities.
        
        Args:
            symbol: Token symbol to search
            chains: List of chains to check (default: major chains)
            min_liquidity: Minimum liquidity filter
            
        Returns:
            DataFrame with best price per chain
        """
        logger.info(f"Getting multi-chain prices for {symbol}")
        
        if chains is None:
            chains = self.MAJOR_CHAINS
        
        df = await self.search_pairs(symbol, min_liquidity=min_liquidity)
        
        if df.empty:
            return pd.DataFrame()
        
        # Filter by chains
        chain_ids = [self.SUPPORTED_CHAINS.get(c, c) for c in chains]
        df = df[df['chain'].isin(chain_ids)]
        
        if df.empty:
            return pd.DataFrame()
        
        # Get best (highest liquidity) price per chain
        prices = []
        for chain in df['chain'].unique():
            chain_df = df[df['chain'] == chain]
            best = chain_df.nlargest(1, 'liquidity_usd')
            
            if not best.empty:
                row = best.iloc[0]
                prices.append({
                    'timestamp': datetime.utcnow(),
                    'symbol': symbol,
                    'chain': chain,
                    'price_usd': row['price_usd'],
                    'liquidity_usd': row['liquidity_usd'],
                    'volume_24h_usd': row['volume_24h_usd'],
                    'dex_id': row['dex_id'],
                    'pair_address': row['pair_address'],
                    'venue': self.VENUE,
                })
        
        result = pd.DataFrame(prices)
        
        if not result.empty and len(result) > 1:
            result = result.sort_values('price_usd', ascending=False)
            
            # Calculate spread metrics
            max_price = result['price_usd'].max()
            min_price = result['price_usd'].min()
            
            if min_price > 0:
                result['spread_from_min_pct'] = (
                    (result['price_usd'] - min_price) / min_price * 100
                )
                result['max_spread_pct'] = (max_price - min_price) / min_price * 100
        
        return result
    
    async def _find_single_symbol_arbitrage(
        self, symbol: str, chains: List[str], min_spread_pct: float, min_liquidity: float
    ) -> Optional[Dict]:
        """Find arbitrage opportunity for a single symbol."""
        prices = await self.get_multi_chain_prices(
            symbol, chains=chains, min_liquidity=min_liquidity
        )

        if prices.empty or len(prices) < 2:
            return None

        max_row = prices.loc[prices['price_usd'].idxmax()]
        min_row = prices.loc[prices['price_usd'].idxmin()]

        max_price = max_row['price_usd']
        min_price = min_row['price_usd']

        if min_price <= 0:
            return None

        spread_pct = (max_price - min_price) / min_price * 100

        if spread_pct >= min_spread_pct:
            # Max trade size limited by lower liquidity
            max_trade = min(min_row['liquidity_usd'], max_row['liquidity_usd']) * 0.1

            return {
                'timestamp': datetime.utcnow(),
                'symbol': symbol,
                'buy_chain': min_row['chain'],
                'buy_price': min_price,
                'buy_liquidity': min_row['liquidity_usd'],
                'buy_dex': min_row['dex_id'],
                'buy_pair_address': min_row['pair_address'],
                'sell_chain': max_row['chain'],
                'sell_price': max_price,
                'sell_liquidity': max_row['liquidity_usd'],
                'sell_dex': max_row['dex_id'],
                'sell_pair_address': max_row['pair_address'],
                'spread_pct': spread_pct,
                'spread_usd': max_price - min_price,
                'max_trade_size_usd': max_trade,
                'is_executable': spread_pct > 1.0 and max_trade > 1000,
                'estimated_profit_pct': spread_pct - 0.6,
                'venue': self.VENUE,
            }
        return None

    async def find_arbitrage_opportunities(
        self,
        symbols: List[str],
        chains: Optional[List[str]] = None,
        min_spread_pct: float = 0.5,
        min_liquidity: float = 50000
    ) -> pd.DataFrame:
        """
        Find cross-chain arbitrage opportunities for multiple tokens.

        Args:
            symbols: Token symbols to check
            chains: Chains to compare (default: major chains)
            min_spread_pct: Minimum spread to report
            min_liquidity: Minimum liquidity on both sides

        Returns:
            DataFrame with arbitrage opportunities sorted by spread
        """
        logger.info(f"Scanning {len(symbols)} tokens for arbitrage opportunities")

        if chains is None:
            chains = self.MAJOR_CHAINS

        # Parallelize symbol scanning
        tasks = [
            self._find_single_symbol_arbitrage(symbol, chains, min_spread_pct, min_liquidity)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter valid opportunities
        opportunities = [r for r in results if isinstance(r, dict)]

        df = pd.DataFrame(opportunities)

        if not df.empty:
            df = df.sort_values('spread_pct', ascending=False).reset_index(drop=True)

        self.collection_stats['records_collected'] += len(df)
        return df
    
    # =========================================================================
    # Pool Discovery Methods
    # =========================================================================
    
    async def _fetch_single_token_pools(
        self, token: str, min_liquidity_usd: float, min_volume_24h: float, chains: List[str]
    ) -> pd.DataFrame:
        """Fetch pools for a single token."""
        df = await self.search_pairs(
            token,
            min_liquidity=min_liquidity_usd,
            chains=chains
        )

        if not df.empty:
            df = df[df['volume_24h_usd'] >= min_volume_24h]
            return df
        return pd.DataFrame()

    async def fetch_liquid_pools(
        self,
        chains: Optional[List[str]] = None,
        min_liquidity_usd: float = 100000,
        min_volume_24h: float = 10000,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch high-liquidity pools for statistical arbitrage.

        Args:
            chains: Chains to search (default: major chains)
            min_liquidity_usd: Minimum liquidity threshold
            min_volume_24h: Minimum 24h volume
            limit: Maximum pools to return

        Returns:
            DataFrame with liquid pools sorted by liquidity
        """
        if chains is None:
            chains = self.MAJOR_CHAINS[:4]

        logger.info(f"Fetching liquid pools on {chains}")

        # Search for DeFi blue chips
        defi_tokens = [
            'UNI', 'AAVE', 'LINK', 'CRV', 'SNX', 'COMP',
            'MKR', 'SUSHI', 'YFI', 'BAL', 'LDO', 'RPL',
            'GMX', 'GNS', 'PENDLE', 'RDNT', 'MAGIC', 'JOE'
        ]

        # Parallelize token pool fetching
        tasks = [
            self._fetch_single_token_pools(token, min_liquidity_usd, min_volume_24h, chains)
            for token in defi_tokens
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter valid DataFrames
        all_pairs = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

        if not all_pairs:
            return pd.DataFrame()

        combined = pd.concat(all_pairs, ignore_index=True)
        combined = combined.drop_duplicates(subset=['pair_address', 'chain'])
        combined = combined.sort_values('liquidity_usd', ascending=False)

        return combined.head(limit)
    
    async def _fetch_single_search_trending(
        self, search: str, chains: Optional[List[str]], min_volume_24h: float
    ) -> pd.DataFrame:
        """Fetch trending tokens for a single search term."""
        df = await self.search_pairs(search, chains=chains)

        if not df.empty:
            df = df[df['volume_24h_usd'] >= min_volume_24h]
            df = df[df['price_change_24h'].notna()]
            return df
        return pd.DataFrame()

    async def fetch_trending_tokens(
        self,
        chains: Optional[List[str]] = None,
        min_volume_24h: float = 50000,
        limit: int = 50
    ) -> pd.DataFrame:
        """
        Fetch trending tokens by volume and price momentum.

        Args:
            chains: Chains to search
            min_volume_24h: Minimum 24h volume
            limit: Maximum tokens to return

        Returns:
            DataFrame with trending tokens sorted by momentum
        """
        logger.info("Fetching trending tokens")

        # Search popular categories
        searches = ['meme', 'ai', 'defi', 'gaming', 'layer2', 'rwa']

        # Parallelize search fetching
        tasks = [
            self._fetch_single_search_trending(search, chains, min_volume_24h)
            for search in searches
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter valid DataFrames
        all_pairs = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

        if not all_pairs:
            return pd.DataFrame()

        combined = pd.concat(all_pairs, ignore_index=True)
        combined = combined.drop_duplicates(subset=['base_token_address', 'chain'])

        # Sort by momentum (combination of volume and price change)
        combined['momentum_score'] = (
            combined['volume_24h_usd'].rank(pct=True) * 0.5 +
            combined['price_change_24h'].rank(pct=True) * 0.5
        )

        combined = combined.sort_values('momentum_score', ascending=False)

        return combined.head(limit)
    
    # =========================================================================
    # Pair Parsing Helper
    # =========================================================================
    
    def _parse_pair(self, pair: Dict) -> Dict[str, Any]:
        """Parse pair data from API response into standardized format."""
        base_token = pair.get('baseToken', {})
        quote_token = pair.get('quoteToken', {})
        liquidity = pair.get('liquidity', {})
        volume = pair.get('volume', {})
        price_change = pair.get('priceChange', {})
        txns = pair.get('txns', {})
        
        # Transaction metrics
        h24_txns = txns.get('h24', {})
        buys = h24_txns.get('buys', 0)
        sells = h24_txns.get('sells', 0)
        buy_sell_ratio = buys / sells if sells > 0 else (float('inf') if buys > 0 else 1.0)
        
        # Liquidity classification
        liq_usd = float(liquidity.get('usd', 0) or 0)
        if liq_usd >= 10_000_000:
            liq_tier = 'whale'
        elif liq_usd >= 1_000_000:
            liq_tier = 'high'
        elif liq_usd >= 100_000:
            liq_tier = 'medium'
        elif liq_usd >= 10_000:
            liq_tier = 'low'
        else:
            liq_tier = 'micro'
        
        # Momentum signal
        if buy_sell_ratio > 2.0:
            momentum = 'strong_buy'
        elif buy_sell_ratio > 1.5:
            momentum = 'buy'
        elif buy_sell_ratio > 0.7:
            momentum = 'neutral'
        elif buy_sell_ratio > 0.5:
            momentum = 'sell'
        else:
            momentum = 'strong_sell'
        
        return {
            'timestamp': datetime.utcnow(),
            'pair_address': pair.get('pairAddress'),
            'chain': pair.get('chainId'),
            'dex_id': pair.get('dexId'),
            'url': pair.get('url'),
            
            # Base token
            'base_token_address': base_token.get('address'),
            'base_token_name': base_token.get('name'),
            'base_token_symbol': base_token.get('symbol'),
            
            # Quote token
            'quote_token_address': quote_token.get('address'),
            'quote_token_name': quote_token.get('name'),
            'quote_token_symbol': quote_token.get('symbol'),
            
            # Pair symbol
            'pair_symbol': f"{base_token.get('symbol', '')}/{quote_token.get('symbol', '')}",
            
            # Prices
            'price_native': float(pair.get('priceNative', 0) or 0),
            'price_usd': float(pair.get('priceUsd', 0) or 0),
            
            # Liquidity
            'liquidity_usd': liq_usd,
            'liquidity_base': float(liquidity.get('base', 0) or 0),
            'liquidity_quote': float(liquidity.get('quote', 0) or 0),
            'liquidity_tier': liq_tier,
            
            # Volume
            'volume_5m_usd': float(volume.get('m5', 0) or 0),
            'volume_1h_usd': float(volume.get('h1', 0) or 0),
            'volume_6h_usd': float(volume.get('h6', 0) or 0),
            'volume_24h_usd': float(volume.get('h24', 0) or 0),
            
            # Price changes
            'price_change_5m': float(price_change.get('m5', 0) or 0),
            'price_change_1h': float(price_change.get('h1', 0) or 0),
            'price_change_6h': float(price_change.get('h6', 0) or 0),
            'price_change_24h': float(price_change.get('h24', 0) or 0),
            
            # Transactions
            'txns_5m_buys': txns.get('m5', {}).get('buys', 0),
            'txns_5m_sells': txns.get('m5', {}).get('sells', 0),
            'txns_1h_buys': txns.get('h1', {}).get('buys', 0),
            'txns_1h_sells': txns.get('h1', {}).get('sells', 0),
            'txns_24h_buys': buys,
            'txns_24h_sells': sells,
            'buy_sell_ratio': buy_sell_ratio,
            'momentum_signal': momentum,
            
            # Market cap
            'fdv': float(pair.get('fdv', 0) or 0),
            'market_cap': float(pair.get('marketCap', 0) or 0),
            
            # Metadata
            'created_at': pair.get('pairCreatedAt'),
            
            # Collection metadata
            'venue': self.VENUE,
            'venue_type': self.VENUE_TYPE,
        }

    # =========================================================================
    # Standardized Collection Methods
    # =========================================================================

    async def _fetch_single_symbol_pools(self, symbol: str, token_mappings: Dict[str, str]) -> pd.DataFrame:
        """Fetch pools for a single symbol."""
        try:
            # Normalize symbol
            symbol_upper = symbol.upper()
            search_term = token_mappings.get(symbol_upper, symbol_upper)

            # Check if it's an address (starts with 0x)
            if symbol.startswith('0x') and len(symbol) == 42:
                # Use get_pairs_by_token for addresses
                pairs = await self.get_pairs_by_token(
                    token_addresses=[symbol]
                )
            else:
                # Use search_pairs for symbols
                pairs = await self.search_pairs(
                    query=search_term,
                    min_liquidity=0
                )

            if not pairs.empty:
                pairs['search_symbol'] = symbol
                logger.info(f"Found {len(pairs)} pairs for {symbol}")
                return pairs
            else:
                logger.warning(f"No pairs found for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching pairs for {symbol}: {e}")
            return pd.DataFrame()

    async def collect_pool_data(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect pool data using search (symbols) or addresses.
        Handles both token symbols (BTC, ETH, WETH) and addresses (0x...).
        Uses search_pairs for symbol lookup, get_pairs_by_token for addresses.

        Standardized method name for collection manager compatibility.
        """
        try:
            # Common token mappings
            token_mappings = {
                'BTC': 'WBTC',
                'BITCOIN': 'WBTC',
                'ETH': 'WETH',
                'ETHEREUM': 'WETH',
            }

            # Parallelize symbol fetching
            tasks = [
                self._fetch_single_symbol_pools(symbol, token_mappings)
                for symbol in symbols
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter valid DataFrames
            all_records = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

            if all_records:
                return pd.concat(all_records, ignore_index=True)

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"DexScreener collect_pool_data error: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
            self.logger.error(f"DexScreener collect_ohlcv error: {e}")
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

        Note: DEX doesn't have funding rates, returns empty DataFrame.
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
            self.logger.error(f"DexScreener collect_funding_rates error: {e}")
            return pd.DataFrame()

    # =========================================================================
    # Statistics Methods
    # =========================================================================

    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.collection_stats.copy()
    
    def reset_collection_stats(self):
        """Reset collection statistics."""
        self.collection_stats = {
            'records_collected': 0,
            'api_calls': 0,
            'errors': 0,
            'rate_limit_hits': 0,
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
        """DEXScreener doesn't provide funding rates (spot DEX aggregator)."""
        logger.debug("DEXScreener: No funding rates available (spot DEX)")
        return pd.DataFrame()
    
    async def _fetch_single_symbol_current_price(self, symbol: str) -> Optional[Dict]:
        """Fetch current price for a single symbol."""
        df = await self.search_pairs(symbol, min_liquidity=10000)

        if not df.empty:
            # Get best pair by liquidity
            best = df.nlargest(1, 'liquidity_usd')
            if not best.empty:
                row = best.iloc[0]
                return {
                    'timestamp': datetime.utcnow(),
                    'symbol': symbol,
                    'open': row['price_usd'],
                    'high': row['price_usd'],
                    'low': row['price_usd'],
                    'close': row['price_usd'],
                    'volume': row['volume_24h_usd'],
                    'liquidity_usd': row['liquidity_usd'],
                    'chain': row['chain'],
                    'dex': row['dex_id'],
                    'pair_address': row['pair_address'],
                    'venue': self.VENUE,
                    'venue_type': self.VENUE_TYPE,
                }
        return None

    async def fetch_ohlcv(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch current prices (DEXScreener doesn't provide historical OHLCV).

        Note: For historical DEX OHLCV, use GeckoTerminal collector.
        """
        logger.info("DEXScreener: Fetching current prices only (no historical OHLCV)")

        # Parallelize symbol price fetching
        tasks = [
            self._fetch_single_symbol_current_price(symbol)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter valid results
        all_data = [r for r in results if isinstance(r, dict)]

        return pd.DataFrame(all_data)
    
    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.debug("DEXScreener session closed")

# =============================================================================
# Test Function
# =============================================================================

async def test_dexscreener_collector():
    """Test DEXScreener collector functionality."""
    config = {'rate_limit': 30}

    async with DEXScreenerCollector(config) as collector:
        print("=" * 60)
        print("DEXScreener Collector Test")
        print("=" * 60)
        
        # Test pair search
        print("\n1. Testing pair search...")
        pairs = await collector.search_pairs('PEPE', min_liquidity=50000)
        print(f" PEPE pairs (>$50K liq): {len(pairs)}")
        if not pairs.empty:
            top = pairs.iloc[0]
            print(f" Top: {top['pair_symbol']} on {top['chain']} - ${top['liquidity_usd']:,.0f} liq")
        
        # Test multi-chain prices
        print("\n2. Testing multi-chain prices...")
        prices = await collector.get_multi_chain_prices('UNI', min_liquidity=100000)
        print(f" UNI price across chains: {len(prices)}")
        if not prices.empty and 'max_spread_pct' in prices.columns:
            print(f" Max spread: {prices['max_spread_pct'].iloc[0]:.2f}%")
        
        # Test liquid pools
        print("\n3. Testing liquid pool discovery...")
        pools = await collector.fetch_liquid_pools(min_liquidity_usd=500000, limit=20)
        print(f" Liquid pools (>$500K): {len(pools)}")
        if not pools.empty:
            total_liq = pools['liquidity_usd'].sum()
            print(f" Total liquidity: ${total_liq:,.0f}")
        
        # Test arbitrage scan
        print("\n4. Testing arbitrage scanner...")
        arbs = await collector.find_arbitrage_opportunities(
            ['UNI', 'AAVE', 'LINK'],
            min_spread_pct=0.3,
            min_liquidity=100000
        )
        print(f" Opportunities found: {len(arbs)}")
        if not arbs.empty:
            top_arb = arbs.iloc[0]
            print(f" Best: {top_arb['symbol']} {top_arb['spread_pct']:.2f}% ({top_arb['buy_chain']} -> {top_arb['sell_chain']})")
        
        # Collection stats
        print("\n" + "=" * 60)
        stats = collector.get_collection_stats()
        print(f"Collection Stats:")
        print(f" Records: {stats['records_collected']}")
        print(f" API Calls: {stats['api_calls']}")
        print(f" Errors: {stats['errors']}")
        print(f" Rate Limit Hits: {stats['rate_limit_hits']}")
        print("=" * 60)

# Alias for backward compatibility (collection_manager expects DexScreenerCollector)
DexScreenerCollector = DEXScreenerCollector

if __name__ == '__main__':
    asyncio.run(test_dexscreener_collector())