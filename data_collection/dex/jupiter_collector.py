"""
Jupiter Protocol Data Collector - Solana DEX Aggregator

validated collector for Jupiter, the leading DEX aggregator on Solana.
Jupiter aggregates liquidity from all major Solana DEXs for best execution.

Supported Data Types:
    - Token prices (aggregated best price)
    - Swap quotes with route optimization
    - Liquidity source mapping
    - Token metadata and verification
    - Market statistics
    - Price impact analysis

Aggregated DEX Sources:
    - Raydium (CPMM and CLMM)
    - Orca (Whirlpools)
    - Serum/OpenBook
    - Lifinity
    - Meteora
    - Marinade
    - 20+ additional DEXs

API Documentation:
    - Quote API: https://station.jup.ag/docs/apis/swap-api
    - Price API: https://station.jup.ag/docs/apis/price-api
    - Token API: https://station.jup.ag/docs/token-list-api

Rate Limits:
    - Public endpoints: 600 requests per minute
    - Quote API: 10 requests per second

Important Notes:
    - Spot DEX aggregator only (no perpetuals/futures)
    - No funding rates (spot market)
    - Chainlink + Pyth for price feeds

Statistical Arbitrage Applications:
    - Solana spot price reference
    - DEX liquidity fragmentation analysis
    - Cross-chain spot arbitrage
    - Price impact estimation
    - Route optimization signals
    - Token liquidity scoring

Version: 2.0.0
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

from ..base_collector import BaseCollector
from ..utils.rate_limiter import get_shared_rate_limiter
from ..utils.retry_handler import RetryHandler

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================

class SwapMode(Enum):
    """Jupiter swap execution modes."""
    EXACT_IN = 'ExactIn'
    EXACT_OUT = 'ExactOut'

class PriceConfidence(Enum):
    """Price confidence level."""
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'

class TokenVerification(Enum):
    """Token verification status."""
    STRICT = 'strict' # Verified tokens only
    ALL = 'all' # All tokens including unverified

class LiquidityTier(Enum):
    """Token liquidity classification."""
    VERY_HIGH = 'very_high' # > $10M daily volume
    HIGH = 'high' # $1M - $10M
    MEDIUM = 'medium' # $100K - $1M
    LOW = 'low' # $10K - $100K
    VERY_LOW = 'very_low' # < $10K

class PriceImpactSeverity(Enum):
    """Price impact severity classification."""
    NEGLIGIBLE = 'negligible' # < 0.1%
    LOW = 'low' # 0.1% - 0.5%
    MODERATE = 'moderate' # 0.5% - 1%
    HIGH = 'high' # 1% - 3%
    SEVERE = 'severe' # > 3%

class PriceTrend(Enum):
    """Price trend classification."""
    STRONG_UP = 'strong_up'
    UP = 'up'
    FLAT = 'flat'
    DOWN = 'down'
    STRONG_DOWN = 'strong_down'

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class JupiterPrice:
    """Jupiter token price data."""
    timestamp: datetime
    symbol: str
    token_address: str
    price: float
    vs_currency: str
    confidence: str
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if price has high confidence."""
        return self.confidence == 'high'
    
    @property
    def price_formatted(self) -> str:
        """Formatted price string."""
        if self.price >= 1:
            return f"${self.price:,.2f}"
        elif self.price >= 0.01:
            return f"${self.price:.4f}"
        else:
            return f"${self.price:.8f}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'token_address': self.token_address,
            'price': self.price,
            'price_formatted': self.price_formatted,
            'vs_currency': self.vs_currency,
            'confidence': self.confidence,
            'is_high_confidence': self.is_high_confidence,
        }

@dataclass
class JupiterQuote:
    """Jupiter swap quote data."""
    timestamp: datetime
    input_token: str
    output_token: str
    input_amount: int
    output_amount: int
    price_impact_pct: float
    slippage_bps: int
    routes_count: int
    dexes_used: List[str]
    other_amount_threshold: int
    swap_mode: str
    
    @property
    def effective_price(self) -> float:
        """Effective execution price."""
        return self.output_amount / self.input_amount if self.input_amount > 0 else 0
    
    @property
    def price_impact_severity(self) -> PriceImpactSeverity:
        """Classify price impact severity."""
        impact = abs(self.price_impact_pct)
        if impact < 0.1:
            return PriceImpactSeverity.NEGLIGIBLE
        elif impact < 0.5:
            return PriceImpactSeverity.LOW
        elif impact < 1.0:
            return PriceImpactSeverity.MODERATE
        elif impact < 3.0:
            return PriceImpactSeverity.HIGH
        else:
            return PriceImpactSeverity.SEVERE
    
    @property
    def is_multi_hop(self) -> bool:
        """Check if route uses multiple hops."""
        return self.routes_count > 1
    
    @property
    def dex_count(self) -> int:
        """Number of DEXs used in route."""
        return len(self.dexes_used)
    
    @property
    def slippage_pct(self) -> float:
        """Slippage tolerance as percentage."""
        return self.slippage_bps / 100
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'input_token': self.input_token,
            'output_token': self.output_token,
            'input_amount': self.input_amount,
            'output_amount': self.output_amount,
            'effective_price': self.effective_price,
            'price_impact_pct': self.price_impact_pct,
            'price_impact_severity': self.price_impact_severity.value,
            'slippage_bps': self.slippage_bps,
            'slippage_pct': self.slippage_pct,
            'routes_count': self.routes_count,
            'dexes_used': self.dexes_used,
            'dex_count': self.dex_count,
            'is_multi_hop': self.is_multi_hop,
            'swap_mode': self.swap_mode,
        }

@dataclass
class JupiterToken:
    """Jupiter token metadata."""
    address: str
    symbol: str
    name: str
    decimals: int
    logo_uri: Optional[str]
    tags: List[str]
    daily_volume: Optional[float]
    verified: bool
    
    @property
    def is_stablecoin(self) -> bool:
        """Check if token is a stablecoin."""
        return 'stablecoin' in self.tags or self.symbol in ['USDC', 'USDT', 'DAI', 'BUSD']
    
    @property
    def is_wrapped(self) -> bool:
        """Check if token is wrapped."""
        return 'wrapped' in self.tags or self.symbol.startswith('W')
    
    @property
    def liquidity_tier(self) -> LiquidityTier:
        """Classify liquidity tier based on volume."""
        vol = self.daily_volume or 0
        if vol > 10_000_000:
            return LiquidityTier.VERY_HIGH
        elif vol > 1_000_000:
            return LiquidityTier.HIGH
        elif vol > 100_000:
            return LiquidityTier.MEDIUM
        elif vol > 10_000:
            return LiquidityTier.LOW
        else:
            return LiquidityTier.VERY_LOW
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'address': self.address,
            'symbol': self.symbol,
            'name': self.name,
            'decimals': self.decimals,
            'logo_uri': self.logo_uri,
            'tags': self.tags,
            'daily_volume': self.daily_volume,
            'verified': self.verified,
            'is_stablecoin': self.is_stablecoin,
            'is_wrapped': self.is_wrapped,
            'liquidity_tier': self.liquidity_tier.value,
        }

@dataclass
class JupiterMarketStats:
    """Jupiter market statistics."""
    timestamp: datetime
    period: str
    total_volume_usd: float
    total_transactions: int
    unique_users: int
    top_pairs: List[Dict]
    
    @property
    def avg_trade_size(self) -> float:
        """Average trade size in USD."""
        return self.total_volume_usd / self.total_transactions if self.total_transactions > 0 else 0
    
    @property
    def volume_per_user(self) -> float:
        """Average volume per user."""
        return self.total_volume_usd / self.unique_users if self.unique_users > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'period': self.period,
            'total_volume_usd': self.total_volume_usd,
            'total_transactions': self.total_transactions,
            'unique_users': self.unique_users,
            'avg_trade_size': self.avg_trade_size,
            'volume_per_user': self.volume_per_user,
            'top_pairs_count': len(self.top_pairs),
        }

@dataclass
class LiquiditySource:
    """DEX liquidity source information."""
    program_id: str
    name: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'program_id': self.program_id,
            'name': self.name,
        }

# =============================================================================
# Collector Class
# =============================================================================

class JupiterCollector(BaseCollector):
    """
    Jupiter Protocol data collector for Solana.
    
    validated implementation for DEX aggregator data.
    
    Features:
    - Token prices (aggregated best price)
    - Swap quotes with route optimization
    - Liquidity source mapping
    - Token metadata and verification
    - Market statistics
    - Price impact analysis
    
    Note: Spot DEX only - no perpetuals or funding rates.
    
    Attributes:
        VENUE: Protocol identifier ('jupiter')
        VENUE_TYPE: Protocol type ('DEX')
        CHAIN: Blockchain ('solana')
    
    Example:
        >>> config = {'rate_limit': 300}
        >>> async with JupiterCollector(config) as collector:
        ... prices = await collector.fetch_token_prices(['SOL', 'BONK'])
        ... quote = await collector.fetch_quote('SOL', 'USDC', 1000000000)
    """
    
    VENUE = 'jupiter'
    VENUE_TYPE = 'DEX'
    CHAIN = 'solana'
    
    # Updated API endpoints (February 2026)
    # Old endpoints (quote-api.jup.ag, api.jup.ag) are DEPRECATED / require paid key
    # Free Lite tier: no key needed, rate limited per 60s window
    # Paid Pro/Ultra: api.jup.ag with API key from portal.jup.ag
    QUOTE_API = 'https://lite-api.jup.ag/swap/v1'
    PRICE_API = 'https://lite-api.jup.ag/price/v2'
    TOKEN_API = 'https://lite-api.jup.ag/tokens/v1'
    STATS_API = 'https://stats.jup.ag/v1' # Stats API unchanged
    
    TOKEN_ADDRESSES = {
        'SOL': 'So11111111111111111111111111111111111111112',
        'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
        'USDT': 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
        'BONK': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
        'JUP': 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',
        'RAY': '4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R',
        'ORCA': 'orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE',
        'MNGO': 'MangoCzJ36AjZyKwVj3VnYU4GTonjfVEnJmvvWaxLac',
        'MSOL': 'mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So',
        'JITOSOL': 'J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn',
        'WIF': 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm',
        'PYTH': 'HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3',
        'W': '85VBFQZC9TZkfaptBWjvUw7YbZjy52A6mjtPGjstQAmQ',
        'RENDER': 'rndrizKT3MK1iimdxRdWabcF7Zg7AR5T4nud4EkHBof',
    }
    
    ADDRESS_TO_SYMBOL = {v: k for k, v in TOKEN_ADDRESSES.items()}
    
    def __init__(self, config: Dict = None):
        """Initialize Jupiter collector."""
        config = config or {}
        super().__init__(config)

        # CRITICAL: Set supported data types for dynamic routing
        self.supported_data_types = ['swaps', 'routes']
        self.venue = 'jupiter'

        # Import VenueType from base_collector
        from ..base_collector import VenueType
        self.venue_type = VenueType.DEX

        # Jupiter API now requires API key (free tier: 60 req/min)
        # Get key from portal.jup.ag
        self.api_key = config.get('api_key') or os.getenv('JUPITER_API_KEY', '')
        self.requires_auth = bool(self.api_key)

        # Rate limit: 60/min for free tier, higher for paid
        rate_limit = config.get('rate_limit', 30 if not self.api_key else 300)
        # Use shared rate limiter to avoid re-initialization overhead
        self.rate_limiter = get_shared_rate_limiter('jupiter', rate=rate_limit, per=60.0, burst=10)
        self.retry_handler = RetryHandler(max_retries=3, base_delay=1.0, max_delay=30.0)

        self.timeout = aiohttp.ClientTimeout(total=config.get('timeout', 30))
        self.session: Optional[aiohttp.ClientSession] = None

        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl = config.get('cache_ttl', 30)

        self.collection_stats = {'records_collected': 0, 'api_calls': 0, 'errors': 0, 'cache_hits': 0}

        if not self.api_key:
            logger.warning("No JUPITER_API_KEY found - API access may be limited. Get free key at portal.jup.ag")
        logger.info(f"Initialized Jupiter collector (rate_limit={rate_limit}/min, data types: {self.supported_data_types})")
    
    async def __aenter__(self):
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            # SPEEDUP: Aggressive connection pooling for high-throughput collection
            connector = aiohttp.TCPConnector(
                limit=60, # Total connection pool size (was 20)
                limit_per_host=20, # Per-host connections (was 10)
                ttl_dns_cache=300, # DNS cache TTL in seconds
                force_close=False, # Keep-alive connections
                enable_cleanup_closed=True
            )
            headers = {}
            if self.api_key:
                headers['x-api-key'] = self.api_key
            self.session = aiohttp.ClientSession(
                timeout=self.timeout,
                connector=connector,
                headers=headers
            )
        return self.session
    
    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
        self._cache.clear()
        logger.info(f"Jupiter collector closed. Stats: {self.collection_stats}")
    
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
        """Set cache value."""
        self._cache[key] = (datetime.utcnow(), value)
    
    async def _get_request(self, url: str, params: Dict = None, use_cache: bool = False) -> Dict:
        """Make GET request to Jupiter API."""
        if use_cache:
            cache_key = f"{url}_{hash(frozenset((params or {}).items()))}"
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached
        
        session = await self._get_session()
        
        async def _request():
            await self.rate_limiter.acquire()
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    self.collection_stats['api_calls'] += 1
                    return await resp.json()
                elif resp.status == 429:
                    logger.warning("Jupiter rate limited")
                    await asyncio.sleep(2)
                    raise aiohttp.ClientResponseError(resp.request_info, resp.history, status=429)
                else:
                    text = await resp.text()
                    logger.error(f"HTTP {resp.status} from Jupiter: {text[:200]}")
                    return {}
        
        try:
            result = await self.retry_handler.execute(_request)
            if use_cache and result:
                self._set_cached(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Error querying Jupiter: {e}")
            self.collection_stats['errors'] += 1
            return {}
    
    def _get_token_address(self, symbol: str) -> Optional[str]:
        """Get token address from symbol."""
        return self.TOKEN_ADDRESSES.get(symbol.upper())
    
    async def fetch_token_prices(self, symbols: List[str], vs_currency: str = 'USDC') -> pd.DataFrame:
        """
        Fetch current token prices from Jupiter Price API.

        Args:
            symbols: List of token symbols (e.g., ['SOL', 'BONK'])
            vs_currency: Quote currency (default USDC)

        Returns:
            DataFrame with price data
        """
        records = []

        addresses = []
        symbol_map = {}

        for sym in symbols:
            addr = self._get_token_address(sym)
            if addr:
                addresses.append(addr)
                symbol_map[addr] = sym
            else:
                logger.warning(f"Unknown token symbol: {sym}")

        if not addresses:
            return pd.DataFrame()

        ids = ','.join(addresses)
        vs_addr = self._get_token_address(vs_currency) or vs_currency

        # New v3 API: https://api.jup.ag/price/v3?ids=mint1,mint2
        url = self.PRICE_API
        params = {'ids': ids, 'vsToken': vs_addr}

        data = await self._get_request(url, params, use_cache=True)

        if not data or 'data' not in data:
            return pd.DataFrame()

        timestamp = datetime.utcnow().replace(tzinfo=timezone.utc)

        for addr, price_data in data.get('data', {}).items():
            symbol = symbol_map.get(addr, self.ADDRESS_TO_SYMBOL.get(addr, addr[:8]))

            jp = JupiterPrice(
                timestamp=timestamp,
                symbol=symbol,
                token_address=addr,
                price=float(price_data.get('price', 0)),
                vs_currency=vs_currency,
                confidence=price_data.get('confidence', 'high')
            )

            records.append({**jp.to_dict(), 'chain': self.CHAIN, 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE})

        self.collection_stats['records_collected'] += len(records)
        return pd.DataFrame(records)
    
    async def fetch_quote(self, input_token: str, output_token: str, amount: int, slippage_bps: int = 50) -> Dict:
        """
        Fetch a swap quote from Jupiter.
        
        Args:
            input_token: Input token symbol or address
            output_token: Output token symbol or address
            amount: Amount in smallest unit (lamports for SOL)
            slippage_bps: Slippage tolerance in basis points
            
        Returns:
            Quote data dictionary
        """
        input_addr = self._get_token_address(input_token) or input_token
        output_addr = self._get_token_address(output_token) or output_token
        
        url = f"{self.QUOTE_API}/quote"
        params = {'inputMint': input_addr, 'outputMint': output_addr, 'amount': amount, 'slippageBps': slippage_bps}
        
        data = await self._get_request(url, params)
        
        if not data:
            return {}
        
        routes = data.get('routePlan', [])
        dexes_used = []
        
        for route in routes:
            swap_info = route.get('swapInfo', {})
            dex = swap_info.get('label', 'Unknown')
            if dex not in dexes_used:
                dexes_used.append(dex)
        
        quote = JupiterQuote(
            timestamp=datetime.utcnow().replace(tzinfo=timezone.utc),
            input_token=input_token,
            output_token=output_token,
            input_amount=amount,
            output_amount=int(data.get('outAmount', 0)),
            price_impact_pct=float(data.get('priceImpactPct', 0)),
            slippage_bps=slippage_bps,
            routes_count=len(routes),
            dexes_used=dexes_used,
            other_amount_threshold=int(data.get('otherAmountThreshold', 0)),
            swap_mode=data.get('swapMode', 'ExactIn')
        )
        
        return {**quote.to_dict(), 'chain': self.CHAIN, 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE}
    
    async def fetch_funding_rates(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Jupiter is a spot DEX aggregator - no funding rates."""
        logger.info("Jupiter is a spot DEX - no funding rates available")
        return pd.DataFrame()
    
    async def fetch_ohlcv(self, symbols: List[str], timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch price data (snapshot only).
        
        Note: Jupiter doesn't provide historical OHLCV.
        Returns current prices formatted as OHLCV snapshot.
        """
        logger.warning("Jupiter doesn't provide historical OHLCV. Fetching current prices only.")
        
        df = await self.fetch_token_prices(symbols)
        
        if not df.empty:
            df['open'] = df['price']
            df['high'] = df['price']
            df['low'] = df['price']
            df['close'] = df['price']
            df['volume'] = 0
        
        return df
    
    async def fetch_token_list(self, mode: str = 'strict') -> pd.DataFrame:
        """
        Fetch list of all tokens known to Jupiter.
        
        Args:
            mode: 'strict' (verified tokens) or 'all' (all tokens)
            
        Returns:
            DataFrame with token metadata
        """
        url = f"{self.TOKEN_API}/{mode}"
        data = await self._get_request(url, use_cache=True)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        
        for token in data:
            jt = JupiterToken(
                address=token.get('address'),
                symbol=token.get('symbol'),
                name=token.get('name'),
                decimals=token.get('decimals', 0),
                logo_uri=token.get('logoURI'),
                tags=token.get('tags', []),
                daily_volume=token.get('daily_volume'),
                verified=mode == 'strict'
            )
            records.append({**jt.to_dict(), 'chain': self.CHAIN, 'venue': self.VENUE})
        
        self.collection_stats['records_collected'] += len(records)
        return pd.DataFrame(records)
    
    async def fetch_market_stats(self, period: str = '24h') -> Dict:
        """
        Fetch overall Jupiter market statistics.
        
        Args:
            period: '24h', '7d', '30d'
            
        Returns:
            Market statistics dictionary
        """
        url = f"{self.STATS_API}/volume"
        params = {'period': period}
        
        data = await self._get_request(url, params, use_cache=True)
        
        if not data:
            return {}
        
        stats = JupiterMarketStats(
            timestamp=datetime.utcnow().replace(tzinfo=timezone.utc),
            period=period,
            total_volume_usd=data.get('totalVolume', 0),
            total_transactions=data.get('totalTransactions', 0),
            unique_users=data.get('uniqueUsers', 0),
            top_pairs=data.get('topPairs', [])
        )
        
        return {**stats.to_dict(), 'chain': self.CHAIN, 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE}
    
    async def fetch_liquidity_sources(self) -> pd.DataFrame:
        """Fetch available liquidity sources (DEXs) aggregated by Jupiter."""
        url = f"{self.QUOTE_API}/program-id-to-label"
        data = await self._get_request(url, use_cache=True)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        
        for program_id, label in data.items():
            ls = LiquiditySource(program_id=program_id, name=label)
            records.append({**ls.to_dict(), 'timestamp': datetime.utcnow().replace(tzinfo=timezone.utc), 'chain': self.CHAIN, 'venue': self.VENUE})
        
        return pd.DataFrame(records)
    
    async def _fetch_single_quote_comparison(
        self,
        amount: int,
        input_token: str,
        output_token: str
    ) -> Optional[Dict]:
        """
        Helper to fetch quote for a single amount (parallelized).

        Args:
            amount: Amount to quote
            input_token: Input token symbol
            output_token: Output token symbol

        Returns:
            Quote comparison data or None
        """
        try:
            quote = await self.fetch_quote(input_token, output_token, amount)

            if quote:
                return {
                    'input_amount': amount,
                    'output_amount': quote.get('output_amount', 0),
                    'price_impact_pct': quote.get('price_impact_pct', 0),
                    'price_impact_severity': quote.get('price_impact_severity', ''),
                    'routes_count': quote.get('routes_count', 0),
                    'dex_count': quote.get('dex_count', 0),
                    'dexes_used': quote.get('dexes_used', []),
                    'effective_price': quote.get('effective_price', 0),
                    'timestamp': quote.get('timestamp'),
                    'input_token': input_token,
                    'output_token': output_token
                }

            return None

        except Exception as e:
            logger.error(f"Error fetching quote for amount {amount}: {e}")
            return None

    async def compare_quotes(self, input_token: str, output_token: str, amounts: List[int]) -> pd.DataFrame:
        """
        Compare quotes for different amounts to analyze price impact.

        Args:
            input_token: Input token symbol
            output_token: Output token symbol
            amounts: List of amounts to quote

        Returns:
            DataFrame with quote comparison
        """
        # Parallelize quote fetching using asyncio.gather
        tasks = [self._fetch_single_quote_comparison(amount, input_token, output_token) for amount in amounts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter valid results
        records = [r for r in results if isinstance(r, dict)]

        df = pd.DataFrame(records)

        if not df.empty and len(df) > 1:
            df['marginal_impact'] = df['price_impact_pct'].diff()

        return df
    
    async def fetch_top_tokens(self, limit: int = 50) -> pd.DataFrame:
        """Fetch top tokens by volume/usage on Jupiter."""
        url = f"{self.STATS_API}/top-tokens"
        params = {'limit': limit}
        
        data = await self._get_request(url, params, use_cache=True)
        
        if not data or 'tokens' not in data:
            return pd.DataFrame()
        
        records = []
        
        for i, token in enumerate(data.get('tokens', [])[:limit]):
            symbol = self.ADDRESS_TO_SYMBOL.get(token.get('mint'), token.get('symbol', 'UNKNOWN'))
            
            records.append({
                'rank': i + 1,
                'address': token.get('mint'),
                'symbol': symbol,
                'name': token.get('name'),
                'volume_24h': token.get('volume24h'),
                'trades_24h': token.get('trades24h'),
                'price_usd': token.get('priceUsd'),
                'price_change_24h': token.get('priceChange24h'),
                'liquidity': token.get('liquidity'),
                'timestamp': datetime.utcnow().replace(tzinfo=timezone.utc),
                'chain': self.CHAIN,
                'venue': self.VENUE
            })
        
        return pd.DataFrame(records)
    
    async def fetch_comprehensive_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch comprehensive Jupiter data."""
        results = {}
        logger.info(f"Fetching comprehensive Jupiter data for {len(symbols)} symbols")
        
        results['prices'] = await self.fetch_token_prices(symbols)
        results['token_list'] = await self.fetch_token_list('strict')
        results['top_tokens'] = await self.fetch_top_tokens(50)
        results['liquidity_sources'] = await self.fetch_liquidity_sources()
        
        return results
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.collection_stats.copy()

    # =========================================================================
    # Standardized Collection Methods (for dynamic routing in collection_manager)
    # =========================================================================

    async def collect_swaps(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect swap data for symbols (standardized interface).

        Wraps fetch_token_prices() to match collection_manager expectations.
        Note: Jupiter is a price aggregator, not a historical swap tracker.

        Args:
            symbols: List of token symbols to fetch swap/price data for
            start_date: Start date (not used - Jupiter provides current prices)
            end_date: End date (not used - Jupiter provides current prices)
            **kwargs: Additional parameters (vs_currency)

        Returns:
            DataFrame with current price/swap data for specified symbols
        """
        try:
            vs_currency = kwargs.get('vs_currency', 'USDC')

            logger.info(f"Jupiter: Collecting swap prices for {len(symbols)} symbols")

            # Fetch current token prices (best available swap rates)
            df = await self.fetch_token_prices(symbols, vs_currency)

            if not df.empty:
                logger.info(f"Jupiter: Collected swap prices for {len(df)} symbols")
                return df

            logger.warning(f"Jupiter: No swap data found for symbols {symbols}")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Jupiter collect_swaps error: {e}")
            return pd.DataFrame()

    async def collect_routes(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect route optimization data for symbols (standardized interface).

        Wraps fetch_liquidity_sources() to match collection_manager expectations.

        Args:
            symbols: List of symbols (not used - fetches all liquidity sources)
            start_date: Start date (not used - sources are current state)
            end_date: End date (not used - sources are current state)
            **kwargs: Additional parameters

        Returns:
            DataFrame with available liquidity sources/routes
        """
        try:
            logger.info("Jupiter: Collecting route/liquidity sources data")

            # Fetch available liquidity sources (DEXs Jupiter routes through)
            df = await self.fetch_liquidity_sources()

            if not df.empty:
                logger.info(f"Jupiter: Collected {len(df)} liquidity sources for routing")
                return df

            logger.warning("Jupiter: No route data found")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Jupiter collect_routes error: {e}")
            return pd.DataFrame()

async def test_jupiter_collector():
    """Test Jupiter collector."""
    config = {'rate_limit': 300}
    
    async with JupiterCollector(config) as collector:
        print("=" * 60)
        print("Jupiter Collector Test")
        print("=" * 60)
        
        prices = await collector.fetch_token_prices(['SOL', 'BONK', 'JUP'])
        if not prices.empty:
            print("\n1. Current prices:")
            for _, row in prices.iterrows():
                print(f" {row['symbol']}: {row['price_formatted']}")
        
        sources = await collector.fetch_liquidity_sources()
        if not sources.empty:
            print(f"\n2. Liquidity sources: {len(sources)} DEXs integrated")
        
        print(f"\nStats: {collector.get_collection_stats()}")

if __name__ == '__main__':
    asyncio.run(test_jupiter_collector())