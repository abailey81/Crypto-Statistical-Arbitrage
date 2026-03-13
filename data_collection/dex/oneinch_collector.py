"""
1inch Aggregator Data Collector - Multi-Chain DEX Aggregator

validated collector for 1inch Protocol aggregation API.
Aggregates liquidity across 100+ DEX sources for optimal execution.

===============================================================================
PROTOCOL OVERVIEW
===============================================================================

1inch is the leading DEX aggregator, routing trades across multiple liquidity
sources to find optimal execution. Unlike single-venue DEXs, 1inch splits
orders across venues to minimize slippage and maximize output.

Key Characteristics:
    - Aggregates 100+ DEX sources per chain
    - Split routing for large orders
    - Chi gas token optimization
    - Pathfinder algorithm for route discovery
    - Fusion mode for gasless swaps

===============================================================================
API DOCUMENTATION
===============================================================================

Base URL: https://api.1inch.dev
Documentation: https://docs.1inch.io/docs/aggregation-protocol/introduction

Endpoints:
    - /swap/v6.0/{chainId}/quote - Get swap quote
    - /swap/v6.0/{chainId}/swap - Get executable swap data
    - /swap/v6.0/{chainId}/liquidity-sources - Available DEXs
    - /swap/v6.0/{chainId}/tokens - Supported tokens

Rate Limits:
    - Free tier: ~1 request/second (conservative)
    - API key tier: Higher limits available
    - Recommended: 60 requests/minute with backoff

===============================================================================
SUPPORTED CHAINS
===============================================================================

Chain ID mappings:
    - 1: Ethereum Mainnet
    - 56: BNB Smart Chain 
    - 137: Polygon
    - 42161: Arbitrum One
    - 10: Optimism
    - 43114: Avalanche C-Chain
    - 8453: Base
    - 100: Gnosis Chain
    - 250: Fantom Opera
    - 324: zkSync Era

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

1. Cross-Chain Price Comparison:
   - Compare effective prices across chains
   - Identify cross-chain arbitrage opportunities
   - Account for bridge costs and latency

2. DEX Market Share Analysis:
   - Track which DEXs capture volume
   - Monitor liquidity migration patterns
   - Identify emerging venues

3. Liquidity Depth Profiling:
   - Measure price impact at various sizes
   - Compare depth across chains
   - Optimal trade sizing

4. Route Optimization Signals:
   - Complex routes indicate fragmented liquidity
   - Simple routes suggest concentrated liquidity
   - Route changes signal market structure shifts

5. Slippage Estimation:
   - Pre-trade slippage prediction
   - Execution quality analysis
   - Market impact modeling

===============================================================================
DATA QUALITY CONSIDERATIONS
===============================================================================

- Quotes are indicative, not guaranteed
- Gas estimates may vary from actual
- Route optimization changes in real-time
- Price impact calculations are estimates
- Multi-hop routes have compounding slippage

Version: 2.0.0
"""

import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timezone
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

class Chain(Enum):
    """Supported blockchain networks."""
    ETHEREUM = 1
    BSC = 56
    POLYGON = 137
    ARBITRUM = 42161
    OPTIMISM = 10
    AVALANCHE = 43114
    BASE = 8453
    GNOSIS = 100
    FANTOM = 250
    ZKSYNC = 324

class RouteComplexity(Enum):
    """Route complexity classification based on hop count."""
    DIRECT = 'direct' # Single DEX, no intermediate tokens
    SIMPLE = 'simple' # 1-2 DEXs, single hop
    MODERATE = 'moderate' # 3-4 DEXs or 2 hops
    COMPLEX = 'complex' # 5+ DEXs or 3+ hops
    HIGHLY_COMPLEX = 'highly_complex' # Split routes with many paths

class PriceImpactSeverity(Enum):
    """Price impact classification for trade sizing."""
    NEGLIGIBLE = 'negligible' # < 0.1% - No concern
    LOW = 'low' # 0.1% - 0.3% - Acceptable
    MODERATE = 'moderate' # 0.3% - 1.0% - Consider splitting
    HIGH = 'high' # 1.0% - 3.0% - Split recommended
    SEVERE = 'severe' # > 3.0% - Requires TWAP/splitting

class LiquidityTier(Enum):
    """DEX liquidity tier classification."""
    TIER_1 = 'tier_1' # Major DEXs: Uniswap, Curve, Balancer
    TIER_2 = 'tier_2' # Secondary: Kyber, Bancor, DODO
    TIER_3 = 'tier_3' # Long-tail DEXs

class QuoteConfidence(Enum):
    """Quote execution confidence level."""
    HIGH = 'high' # Deep liquidity, low slippage expected
    MEDIUM = 'medium' # Moderate liquidity, some slippage risk
    LOW = 'low' # Thin liquidity, high slippage risk

class TokenType(Enum):
    """Token classification."""
    NATIVE = 'native' # ETH, MATIC, etc.
    WRAPPED_NATIVE = 'wrapped' # WETH, WMATIC
    STABLECOIN = 'stablecoin' # USDC, USDT, DAI
    GOVERNANCE = 'governance' # UNI, AAVE
    UTILITY = 'utility' # Other tokens

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class OneInchQuote:
    """1inch swap quote with computed analytics."""
    timestamp: datetime
    chain: str
    chain_id: int
    from_token: str
    to_token: str
    from_token_symbol: str
    to_token_symbol: str
    from_amount: int
    to_amount: int
    from_decimals: int
    to_decimals: int
    protocols: List[Any]
    gas_estimate: int
    
    @property
    def effective_price(self) -> float:
        """Effective execution price (to/from in human units)."""
        from_human = self.from_amount / (10 ** self.from_decimals)
        to_human = self.to_amount / (10 ** self.to_decimals)
        return to_human / from_human if from_human > 0 else 0
    
    @property
    def inverse_price(self) -> float:
        """Inverse price (from/to)."""
        return 1 / self.effective_price if self.effective_price > 0 else 0
    
    @property
    def routes_count(self) -> int:
        """Number of distinct routing paths."""
        return len(self.protocols) if self.protocols else 0
    
    @property
    def route_complexity(self) -> RouteComplexity:
        """Classify route complexity."""
        count = self.routes_count
        if count == 0:
            return RouteComplexity.DIRECT
        elif count == 1:
            return RouteComplexity.SIMPLE
        elif count <= 3:
            return RouteComplexity.MODERATE
        elif count <= 6:
            return RouteComplexity.COMPLEX
        else:
            return RouteComplexity.HIGHLY_COMPLEX
    
    @property
    def is_split_route(self) -> bool:
        """Check if route uses split execution across DEXs."""
        return self.routes_count > 1
    
    @property
    def dex_count(self) -> int:
        """Count unique DEXs used in route."""
        dexes = set()
        for route in self.protocols or []:
            if isinstance(route, list):
                for step in route:
                    if isinstance(step, list):
                        for swap in step:
                            if isinstance(swap, dict):
                                dexes.add(swap.get('name', ''))
        return len(dexes)
    
    @property
    def gas_cost_gwei(self) -> float:
        """Gas cost in gwei (assumes 30 gwei gas price)."""
        return self.gas_estimate * 30 / 1e9
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'chain': self.chain,
            'chain_id': self.chain_id,
            'from_token': self.from_token,
            'to_token': self.to_token,
            'from_token_symbol': self.from_token_symbol,
            'to_token_symbol': self.to_token_symbol,
            'from_amount': self.from_amount,
            'to_amount': self.to_amount,
            'effective_price': self.effective_price,
            'inverse_price': self.inverse_price,
            'routes_count': self.routes_count,
            'route_complexity': self.route_complexity.value,
            'is_split_route': self.is_split_route,
            'dex_count': self.dex_count,
            'gas_estimate': self.gas_estimate,
        }

@dataclass
class LiquiditySource:
    """DEX liquidity source information."""
    id: str
    name: str
    chain: str
    chain_id: int
    img_url: Optional[str] = None
    
    @property
    def tier(self) -> LiquidityTier:
        """Classify DEX tier based on name."""
        tier_1_keywords = ['UNISWAP', 'CURVE', 'BALANCER', 'SUSHISWAP', 
                          'PANCAKESWAP', 'AAVE', 'COMPOUND', 'MAKER']
        tier_2_keywords = ['KYBER', 'BANCOR', 'DODO', 'SHIBASWAP', 
                          'QUICKSWAP', 'TRADERJOE', 'VELODROME']
        
        name_upper = self.name.upper()
        if any(kw in name_upper for kw in tier_1_keywords):
            return LiquidityTier.TIER_1
        elif any(kw in name_upper for kw in tier_2_keywords):
            return LiquidityTier.TIER_2
        else:
            return LiquidityTier.TIER_3
    
    @property
    def is_amm(self) -> bool:
        """Check if source is AMM-based."""
        amm_keywords = ['SWAP', 'CURVE', 'BALANCER', 'POOL']
        return any(kw in self.name.upper() for kw in amm_keywords)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'chain': self.chain,
            'chain_id': self.chain_id,
            'tier': self.tier.value,
            'is_amm': self.is_amm,
            'img_url': self.img_url,
        }

@dataclass
class TokenInfo:
    """Token information from 1inch."""
    address: str
    symbol: str
    name: str
    decimals: int
    chain: str
    chain_id: int
    logo_uri: Optional[str] = None
    
    @property
    def token_type(self) -> TokenType:
        """Classify token type."""
        symbol_upper = self.symbol.upper()
        
        if self.address.lower() == '0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee':
            return TokenType.NATIVE
        
        wrapped = ['WETH', 'WBNB', 'WMATIC', 'WAVAX', 'WFTM', 'WONE']
        if symbol_upper in wrapped:
            return TokenType.WRAPPED_NATIVE
        
        stables = ['USDC', 'USDT', 'DAI', 'BUSD', 'FRAX', 'LUSD', 'TUSD', 'USDD']
        if symbol_upper in stables:
            return TokenType.STABLECOIN
        
        governance = ['UNI', 'AAVE', 'COMP', 'MKR', 'CRV', 'BAL', 'SUSHI']
        if symbol_upper in governance:
            return TokenType.GOVERNANCE
        
        return TokenType.UTILITY
    
    @property
    def is_stablecoin(self) -> bool:
        """Check if likely stablecoin."""
        return self.token_type == TokenType.STABLECOIN
    
    @property
    def is_wrapped_native(self) -> bool:
        """Check if wrapped native token."""
        return self.token_type == TokenType.WRAPPED_NATIVE
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'address': self.address,
            'symbol': self.symbol,
            'name': self.name,
            'decimals': self.decimals,
            'chain': self.chain,
            'chain_id': self.chain_id,
            'token_type': self.token_type.value,
            'is_stablecoin': self.is_stablecoin,
            'is_wrapped_native': self.is_wrapped_native,
        }

@dataclass
class CrossChainComparison:
    """Cross-chain price comparison data."""
    timestamp: datetime
    token: str
    chain: str
    chain_id: int
    token_address: str
    amount_in: int
    amount_out: float
    implied_price: float
    gas_estimate: int
    routes_count: int
    
    @property
    def gas_adjusted_price(self) -> float:
        """Price adjusted for estimated gas (rough)."""
        # Approximate gas cost in USD (assumes $2000 ETH, 30 gwei)
        gas_cost_usd = self.gas_estimate * 30 * 2000 / 1e18
        return self.implied_price - gas_cost_usd if self.amount_in > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'token': self.token,
            'chain': self.chain,
            'chain_id': self.chain_id,
            'token_address': self.token_address,
            'amount_in': self.amount_in,
            'amount_out': self.amount_out,
            'implied_price': self.implied_price,
            'gas_estimate': self.gas_estimate,
            'routes_count': self.routes_count,
        }

@dataclass 
class LiquidityDepthPoint:
    """Liquidity depth analysis at specific trade size."""
    chain: str
    amount_in: int
    amount_out: int
    rate: float
    price_impact_pct: float
    gas_estimate: int
    routes_count: int
    
    @property
    def price_impact_severity(self) -> PriceImpactSeverity:
        """Classify price impact severity."""
        impact = abs(self.price_impact_pct)
        if impact < 0.1:
            return PriceImpactSeverity.NEGLIGIBLE
        elif impact < 0.3:
            return PriceImpactSeverity.LOW
        elif impact < 1.0:
            return PriceImpactSeverity.MODERATE
        elif impact < 3.0:
            return PriceImpactSeverity.HIGH
        else:
            return PriceImpactSeverity.SEVERE
    
    @property
    def is_efficient_size(self) -> bool:
        """Check if trade size is efficient (<0.5% impact)."""
        return abs(self.price_impact_pct) < 0.5
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'chain': self.chain,
            'amount_in': self.amount_in,
            'amount_out': self.amount_out,
            'rate': self.rate,
            'price_impact_pct': self.price_impact_pct,
            'price_impact_severity': self.price_impact_severity.value,
            'is_efficient_size': self.is_efficient_size,
            'gas_estimate': self.gas_estimate,
            'routes_count': self.routes_count,
        }

@dataclass
class DEXMarketShare:
    """DEX market share for a specific swap route."""
    dex: str
    volume_share: float
    percentage: float
    chain: str
    
    @property
    def is_dominant(self) -> bool:
        """Check if DEX is dominant (>50% share)."""
        return self.percentage > 50
    
    @property
    def share_category(self) -> str:
        """Categorize market share."""
        if self.percentage > 50:
            return 'dominant'
        elif self.percentage > 25:
            return 'major'
        elif self.percentage > 10:
            return 'significant'
        else:
            return 'minor'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'dex': self.dex,
            'volume_share': self.volume_share,
            'percentage': self.percentage,
            'chain': self.chain,
            'is_dominant': self.is_dominant,
            'share_category': self.share_category,
        }

# =============================================================================
# Collector Class
# =============================================================================

class OneInchCollector(BaseCollector):
    """
    1inch Aggregator data collector.
    
    validated implementation for DEX aggregation data collection.
    Supports multi-chain price comparison, liquidity analysis, and
    route optimization insights.
    
    Features:
        - Swap quotes with route optimization
        - Multi-chain support (10+ chains)
        - Liquidity source analysis
        - Price impact estimation
        - Cross-chain price comparison
        - DEX market share tracking
    
    Attributes:
        VENUE: Protocol identifier ('oneinch')
        VENUE_TYPE: Protocol type ('DEX_AGGREGATOR')
        BASE_URL: API endpoint
    
    Example:
        >>> config = {'rate_limit': 30}
        >>> async with OneInchCollector(config) as collector:
        ... quote = await collector.get_quote(
        ... 'ethereum', from_token, to_token, amount
        ... )
        ... sources = await collector.get_liquidity_sources('ethereum')
    """
    
    VENUE = 'oneinch'
    VENUE_TYPE = 'DEX_AGGREGATOR'
    BASE_URL = 'https://api.1inch.dev'
    
    CHAIN_IDS = {
        'ethereum': 1, 'bsc': 56, 'polygon': 137, 'arbitrum': 42161,
        'optimism': 10, 'avalanche': 43114, 'base': 8453, 'gnosis': 100,
        'fantom': 250, 'zksync': 324, 'aurora': 1313161554,
    }
    
    COMMON_TOKENS = {
        1: { # Ethereum
            'ETH': ('0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE', 18),
            'WETH': ('0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2', 18),
            'USDC': ('0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48', 6),
            'USDT': ('0xdAC17F958D2ee523a2206206994597C13D831ec7', 6),
            'WBTC': ('0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599', 8),
        },
        42161: { # Arbitrum
            'ETH': ('0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE', 18),
            'WETH': ('0x82aF49447D8a07e3bd95BD0d56f35241523fBab1', 18),
            'USDC': ('0xaf88d065e77c8cC2239327C5EDb3A432268e5831', 6),
            'ARB': ('0x912CE59144191C1204E64559FE8253a0e49E6548', 18),
        },
        137: { # Polygon
            'MATIC': ('0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE', 18),
            'WMATIC': ('0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270', 18),
            'USDC': ('0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174', 6),
        },
        8453: { # Base
            'ETH': ('0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE', 18),
            'WETH': ('0x4200000000000000000000000000000000000006', 18),
            'USDC': ('0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913', 6),
        },
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize 1inch collector."""
        config = config or {}
        super().__init__(config)

        # CRITICAL: Set supported data types for dynamic routing
        self.supported_data_types = ['swaps', 'routes']
        self.venue = 'oneinch'

        # Import VenueType from base_collector
        from ..base_collector import VenueType
        self.venue_type = VenueType.DEX

        # Load API key from config or environment - required for higher rate limits
        self.api_key = config.get('api_key') or config.get('oneinch_api_key') or os.getenv('ONEINCH_API_KEY', '')
        self.requires_auth = bool(self.api_key) # 1inch API works without key (with limits)

        rate_limit = config.get('rate_limit', 30)
        self.rate_limiter = get_shared_rate_limiter('oneinch', rate=rate_limit, per=60.0, burst=3)
        # OPTIMIZATION: Reduced max_delay from 60s to 30s to avoid long stalls
        self.retry_handler = RetryHandler(max_retries=3, base_delay=2.0, max_delay=30.0)

        self.timeout = aiohttp.ClientTimeout(total=config.get('timeout', 30))
        self.session: Optional[aiohttp.ClientSession] = None

        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl = config.get('cache_ttl', 30)

        self.collection_stats = {
            'records_collected': 0, 'api_calls': 0, 
            'errors': 0, 'cache_hits': 0
        }
        logger.info(f"Initialized 1inch collector (rate_limit={rate_limit}/min)")
    
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
            # SPEEDUP: Aggressive connection pooling for high-throughput collection
            connector = aiohttp.TCPConnector(
                limit=50, # Total connection pool size (was 10)
                limit_per_host=15, # Per-host connections (was 5)
                ttl_dns_cache=300, # DNS cache TTL in seconds
                force_close=False, # Keep-alive connections
                enable_cleanup_closed=True
            )
            headers = {'Accept': 'application/json', 'User-Agent': 'CryptoStatArb/2.0'}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            self.session = aiohttp.ClientSession(
                timeout=self.timeout, connector=connector, headers=headers
            )
        return self.session
    
    async def close(self):
        """Close aiohttp session and clear cache."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
        self._cache.clear()
        logger.info(f"1inch collector closed. Stats: {self.collection_stats}")
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            timestamp, value = self._cache[key]
            if (datetime.utcnow() - timestamp).total_seconds() < self._cache_ttl:
                self.collection_stats['cache_hits'] += 1
                return value
            del self._cache[key]
        return None
    
    def _set_cached(self, key: str, value: Any):
        """Set cache value with timestamp."""
        self._cache[key] = (datetime.utcnow(), value)
    
    async def _request(
        self, endpoint: str, params: Optional[Dict] = None, use_cache: bool = False
    ) -> Optional[Dict]:
        """Make request to 1inch API with rate limiting and retry."""
        if use_cache:
            cache_key = f"{endpoint}_{hash(frozenset((params or {}).items()))}"
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached
        
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        
        async def _do_request():
            await self.rate_limiter.acquire()
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    self.collection_stats['api_calls'] += 1
                    return await response.json()
                elif response.status == 429:
                    logger.warning("1inch rate limited, waiting 60s")
                    await asyncio.sleep(60)
                    raise aiohttp.ClientResponseError(
                        response.request_info, response.history, status=429
                    )
                else:
                    text = await response.text()
                    logger.warning(f"1inch API error {response.status}: {text[:200]}")
                    return None
        
        try:
            result = await self.retry_handler.execute(_do_request)
            if use_cache and result is not None:
                self._set_cached(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"1inch request error: {e}")
            self.collection_stats['errors'] += 1
            return None
    
    async def get_quote(
        self, chain: str, from_token: str, to_token: str, amount: int,
        from_decimals: int = 18, to_decimals: int = 18,
        from_symbol: str = '', to_symbol: str = ''
    ) -> Optional[OneInchQuote]:
        """
        Get swap quote from 1inch with full analytics.
        
        Args:
            chain: Chain name (ethereum, arbitrum, etc.)
            from_token: Source token address
            to_token: Destination token address
            amount: Amount in smallest units (wei)
            from_decimals: Source token decimals
            to_decimals: Destination token decimals
            from_symbol: Source token symbol
            to_symbol: Destination token symbol
            
        Returns:
            OneInchQuote dataclass with computed properties
        """
        chain_id = self.CHAIN_IDS.get(chain.lower())
        if not chain_id:
            logger.error(f"Unsupported chain: {chain}")
            return None
        
        params = {'src': from_token, 'dst': to_token, 'amount': str(amount)}
        data = await self._request(f"/swap/v6.0/{chain_id}/quote", params)
        
        if not data:
            return None
        
        return OneInchQuote(
            timestamp=datetime.now(timezone.utc),
            chain=chain,
            chain_id=chain_id,
            from_token=from_token,
            to_token=to_token,
            from_token_symbol=from_symbol,
            to_token_symbol=to_symbol,
            from_amount=int(data.get('srcAmount', amount)),
            to_amount=int(data.get('dstAmount', 0)),
            from_decimals=from_decimals,
            to_decimals=to_decimals,
            protocols=data.get('protocols', []),
            gas_estimate=int(data.get('gas', 0))
        )
    
    async def get_liquidity_sources(self, chain: str) -> pd.DataFrame:
        """Get available liquidity sources on a chain."""
        chain_id = self.CHAIN_IDS.get(chain.lower())
        if not chain_id:
            return pd.DataFrame()
        
        data = await self._request(
            f"/swap/v6.0/{chain_id}/liquidity-sources", use_cache=True
        )
        
        if not data or 'protocols' not in data:
            return pd.DataFrame()
        
        records = []
        for protocol in data['protocols']:
            source = LiquiditySource(
                id=protocol.get('id', ''),
                name=protocol.get('title', ''),
                chain=chain,
                chain_id=chain_id,
                img_url=protocol.get('img')
            )
            records.append({
                **source.to_dict(),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'venue': self.VENUE
            })
        
        self.collection_stats['records_collected'] += len(records)
        return pd.DataFrame(records)
    
    async def get_tokens(self, chain: str) -> pd.DataFrame:
        """Get supported tokens on a chain."""
        chain_id = self.CHAIN_IDS.get(chain.lower())
        if not chain_id:
            return pd.DataFrame()
        
        data = await self._request(f"/swap/v6.0/{chain_id}/tokens", use_cache=True)
        
        if not data or 'tokens' not in data:
            return pd.DataFrame()
        
        records = []
        for address, info in data['tokens'].items():
            token = TokenInfo(
                address=address,
                symbol=info.get('symbol', ''),
                name=info.get('name', ''),
                decimals=info.get('decimals', 18),
                chain=chain,
                chain_id=chain_id,
                logo_uri=info.get('logoURI')
            )
            records.append({
                **token.to_dict(),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'venue': self.VENUE
            })
        
        return pd.DataFrame(records)
    
    async def _fetch_single_amount_quote(
        self,
        amount: int,
        chain: str,
        token_in: str,
        token_out: str,
        in_decimals: int,
        out_decimals: int,
        base_rate: Optional[float]
    ) -> Optional[Dict]:
        """
        Helper to fetch quote for a single amount (parallelized).

        Args:
            amount: Amount to test
            chain: Chain name
            token_in: Input token address
            token_out: Output token address
            in_decimals: Input token decimals
            out_decimals: Output token decimals
            base_rate: Base rate for price impact calculation

        Returns:
            Quote data dictionary or None
        """
        try:
            quote = await self.get_quote(
                chain, token_in, token_out, amount,
                from_decimals=in_decimals, to_decimals=out_decimals
            )

            if not quote or quote.from_amount == 0:
                return None

            rate = quote.effective_price

            price_impact = (base_rate - rate) / base_rate * 100 if base_rate and base_rate > 0 else 0

            ldp = LiquidityDepthPoint(
                chain=chain,
                amount_in=amount,
                amount_out=quote.to_amount,
                rate=rate,
                price_impact_pct=price_impact,
                gas_estimate=quote.gas_estimate,
                routes_count=quote.routes_count
            )
            return {
                **ldp.to_dict(),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'venue': self.VENUE,
                'amount': amount,
                'rate': rate
            }

        except Exception as e:
            logger.error(f"Error fetching quote for amount {amount}: {e}")
            return None

    async def analyze_liquidity_depth(
        self, chain: str, token_in: str, token_out: str,
        amounts: List[int], in_decimals: int = 18, out_decimals: int = 18
    ) -> pd.DataFrame:
        """
        Analyze price impact at different trade sizes.

        Args:
            chain: Chain name
            token_in: Input token address
            token_out: Output token address
            amounts: List of amounts to test (in smallest units)
            in_decimals: Input token decimals
            out_decimals: Output token decimals

        Returns:
            DataFrame with price impact analysis
        """
        sorted_amounts = sorted(amounts)

        # Fetch smallest amount first to get base rate
        if sorted_amounts:
            first_quote = await self.get_quote(
                chain, token_in, token_out, sorted_amounts[0],
                from_decimals=in_decimals, to_decimals=out_decimals
            )
            base_rate = first_quote.effective_price if first_quote and first_quote.from_amount > 0 else None
        else:
            base_rate = None

        # Parallelize remaining amounts using asyncio.gather
        tasks = [
            self._fetch_single_amount_quote(amount, chain, token_in, token_out, in_decimals, out_decimals, base_rate)
            for amount in sorted_amounts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter valid results
        records = [r for r in results if isinstance(r, dict)]

        return pd.DataFrame(records)
    
    async def get_dex_market_share(
        self, chain: str, token_in: str, token_out: str, amount: int
    ) -> pd.DataFrame:
        """Analyze DEX market share for a specific swap."""
        quote = await self.get_quote(chain, token_in, token_out, amount)
        
        if not quote:
            return pd.DataFrame()
        
        dex_volumes: Dict[str, float] = {}
        
        for route in quote.protocols or []:
            if isinstance(route, list):
                for step in route:
                    if isinstance(step, list):
                        for swap in step:
                            if isinstance(swap, dict):
                                name = swap.get('name', 'Unknown')
                                part = float(swap.get('part', 0))
                                dex_volumes[name] = dex_volumes.get(name, 0) + part
        
        total = sum(dex_volumes.values())
        
        records = []
        for dex, volume in dex_volumes.items():
            ms = DEXMarketShare(
                dex=dex,
                volume_share=volume,
                percentage=volume / total * 100 if total > 0 else 0,
                chain=chain
            )
            records.append({
                **ms.to_dict(),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'venue': self.VENUE
            })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('percentage', ascending=False)
        return df
    
    async def fetch_funding_rates(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """DEX aggregator doesn't have funding rates."""
        logger.info("1inch: No funding rates (spot DEX aggregator)")
        return pd.DataFrame()
    
    async def _fetch_single_symbol_quote(
        self,
        symbol: str,
        chain: str,
        tokens: Dict,
        usdc_addr: str,
        usdc_dec: int
    ) -> Optional[Dict]:
        """
        Helper to fetch quote for a single symbol (parallelized).

        Args:
            symbol: Token symbol
            chain: Chain name
            tokens: Token mapping dictionary
            usdc_addr: USDC address
            usdc_dec: USDC decimals

        Returns:
            Quote data dictionary or None
        """
        try:
            token_info = tokens.get(symbol)
            if not token_info or token_info[0] == usdc_addr:
                return None

            token_addr, token_dec = token_info
            amount = int(1 * (10 ** token_dec))

            quote = await self.get_quote(
                chain, token_addr, usdc_addr, amount,
                from_decimals=token_dec, to_decimals=usdc_dec,
                from_symbol=symbol, to_symbol='USDC'
            )

            if quote and quote.to_amount > 0:
                price = quote.effective_price

                return {
                    'timestamp': datetime.now(timezone.utc),
                    'symbol': symbol,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': 0,
                    'chain': chain,
                    'venue': self.VENUE,
                    'venue_type': self.VENUE_TYPE
                }

            return None

        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None

    async def fetch_ohlcv(
        self, symbols: List[str], timeframe: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch current prices via quotes (no historical OHLCV)."""
        logger.info("1inch: Fetching current spot prices via quotes")

        chain = 'ethereum'
        chain_id = self.CHAIN_IDS[chain]
        tokens = self.COMMON_TOKENS.get(chain_id, {})
        usdc_info = tokens.get('USDC')

        if not usdc_info:
            return pd.DataFrame()

        usdc_addr, usdc_dec = usdc_info

        # Parallelize symbol quote fetching using asyncio.gather
        tasks = [self._fetch_single_symbol_quote(symbol, chain, tokens, usdc_addr, usdc_dec) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter valid results
        all_data = [r for r in results if isinstance(r, dict)]

        return pd.DataFrame(all_data)
    
    async def collect_routes(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect route/liquidity source data - wraps get_liquidity_sources().

        Standardized method name for collection manager compatibility.
        """
        try:
            chain = kwargs.get('chain', 'ethereum')

            # Fetch liquidity sources for the chain
            df = await self.get_liquidity_sources(chain=chain)
            return df

        except Exception as e:
            logger.error(f"1inch collect_routes error: {e}")
            return pd.DataFrame()

    async def collect_swaps(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect swap data (Note: 1inch API doesn't provide historical swap data).

        Returns current quote/price data instead.
        Standardized method name for collection manager compatibility.
        """
        try:
            chain = kwargs.get('chain', 'ethereum')

            # 1inch doesn't provide historical swap data through their API
            # Return empty DataFrame with note
            logger.warning("1inch API doesn't provide historical swap data - use subgraph or indexer instead")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"1inch collect_swaps error: {e}")
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
            logger.error(f"1inch collect_funding_rates error: {e}")
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
            logger.error(f"1inch collect_ohlcv error: {e}")
            return pd.DataFrame()

    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.collection_stats.copy()

async def test_oneinch_collector():
    """Test 1inch collector functionality."""
    config = {'rate_limit': 30}
    
    async with OneInchCollector(config) as collector:
        print("=" * 60)
        print("1inch Collector Test")
        print("=" * 60)
        
        sources = await collector.get_liquidity_sources('ethereum')
        if not sources.empty:
            print(f"\n1. Liquidity sources on Ethereum: {len(sources)}")
            tier1 = sources[sources['tier'] == 'tier_1']
            print(f" Tier 1 DEXs: {len(tier1)}")
        
        print(f"\nStats: {collector.get_collection_stats()}")

if __name__ == '__main__':
    asyncio.run(test_oneinch_collector())