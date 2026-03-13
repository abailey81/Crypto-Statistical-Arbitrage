"""
0x Protocol Aggregator Data Collector - Multi-Chain DEX Aggregation

validated collector for 0x Protocol swap/aggregation API.
Aggregates liquidity across multiple DEX sources for optimal execution.

===============================================================================
PROTOCOL OVERVIEW
===============================================================================

0x Protocol is a decentralized exchange infrastructure providing:
    - DEX aggregation across multiple venues
    - Professional market maker (RFQ) liquidity
    - Gasless trading via 0x API
    - Limit order protocol

Key Characteristics:
    - Aggregates 100+ liquidity sources
    - Professional market maker integration
    - Gasless swap support
    - Multi-chain deployment

===============================================================================
API DOCUMENTATION
===============================================================================

Base URL (v2 - unified for all chains):
    - https://api.0x.org (chain specified via chainId query parameter)

Required Headers:
    - 0x-api-key: Your API key from dashboard.0x.org
    - 0x-version: v2

Endpoints (v2):
    - /swap/allowance-holder/price - Get indicative price
    - /swap/allowance-holder/quote - Get executable quote
    - /sources - Available liquidity sources

Note: v1 API was sunset on April 11, 2025. This collector uses v2 API.

Rate Limits:
    - Free tier: ~100 requests/minute
    - API key tier: Higher limits

===============================================================================
SUPPORTED CHAINS
===============================================================================

Chain configurations:
    - ethereum (1): Highest liquidity, most sources
    - polygon (137): High volume, low fees
    - bsc (56): Large retail volume
    - arbitrum (42161): Growing L2
    - optimism (10): OP incentives
    - avalanche (43114): C-Chain
    - base (8453): Coinbase L2
    - fantom (250): Opera network

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

1. Cross-Chain Price Comparison:
   - Compare prices across chains for same token
   - Identify cross-chain arbitrage opportunities
   - Account for bridge costs and latency

2. DEX Market Share Analysis:
   - Track source utilization by trade size
   - Monitor liquidity migration
   - Identify emerging venues

3. Liquidity Depth Profiling:
   - Measure price impact at various sizes
   - Compare depth across chains
   - Optimal trade sizing

4. Arbitrage Detection:
   - Forward/reverse quote comparison
   - Identify round-trip opportunities
   - Gas-adjusted profitability

5. Execution Quality:
   - Slippage estimation
   - Route complexity analysis
   - Gas optimization

===============================================================================
DATA QUALITY CONSIDERATIONS
===============================================================================

- Quotes are indicative, not guaranteed
- Gas estimates may vary from actual execution
- Price impact is estimated
- RFQ quotes may have short validity windows
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
    ETHEREUM = 'ethereum'
    POLYGON = 'polygon'
    BSC = 'bsc'
    ARBITRUM = 'arbitrum'
    OPTIMISM = 'optimism'
    AVALANCHE = 'avalanche'
    BASE = 'base'
    FANTOM = 'fantom'

class PriceImpactSeverity(Enum):
    """Price impact classification."""
    NEGLIGIBLE = 'negligible' # < 0.1%
    LOW = 'low' # 0.1% - 0.5%
    MODERATE = 'moderate' # 0.5% - 1%
    HIGH = 'high' # 1% - 3%
    SEVERE = 'severe' # > 3%

class QuoteConfidence(Enum):
    """Quote execution confidence level."""
    HIGH = 'high' # Direct route, deep liquidity
    MEDIUM = 'medium' # Multi-hop, moderate liquidity
    LOW = 'low' # Complex route, thin liquidity

class SourceTier(Enum):
    """Liquidity source tier classification."""
    TIER_1 = 'tier_1' # Major DEXs: Uniswap, Curve, Balancer
    TIER_2 = 'tier_2' # Secondary DEXs
    TIER_3 = 'tier_3' # Long-tail sources
    RFQ = 'rfq' # Professional market makers

class ArbitrageViability(Enum):
    """Arbitrage opportunity viability."""
    PROFITABLE = 'profitable' # > gas costs
    MARGINAL = 'marginal' # Near breakeven
    UNPROFITABLE = 'unprofitable' # Below gas costs

class RouteType(Enum):
    """Route type classification."""
    DIRECT = 'direct' # Single source
    SPLIT = 'split' # Multiple sources, same hop
    MULTI_HOP = 'multi_hop' # Multiple intermediate tokens

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class ZeroXPrice:
    """0x price quote data with analytics."""
    timestamp: datetime
    chain: str
    sell_token: str
    buy_token: str
    sell_amount: int
    buy_amount: int
    price: float
    gas_price: int
    estimated_gas: int
    sources: List[Dict]
    
    @property
    def effective_price(self) -> float:
        """Effective execution price."""
        return self.buy_amount / self.sell_amount if self.sell_amount > 0 else 0
    
    @property
    def sources_count(self) -> int:
        """Number of liquidity sources used."""
        return len([s for s in self.sources if float(s.get('proportion', 0)) > 0])
    
    @property
    def route_type(self) -> RouteType:
        """Classify route type."""
        active_sources = self.sources_count
        if active_sources == 1:
            return RouteType.DIRECT
        elif active_sources > 1:
            return RouteType.SPLIT
        else:
            return RouteType.DIRECT
    
    @property
    def gas_cost_wei(self) -> int:
        """Estimated gas cost in wei."""
        return self.estimated_gas * self.gas_price
    
    @property
    def gas_cost_eth(self) -> float:
        """Estimated gas cost in ETH."""
        return self.gas_cost_wei / 1e18
    
    @property
    def dominant_source(self) -> Optional[str]:
        """Get dominant liquidity source (>50%)."""
        for source in self.sources:
            if float(source.get('proportion', 0)) > 0.5:
                return source.get('name')
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'chain': self.chain,
            'sell_token': self.sell_token,
            'buy_token': self.buy_token,
            'sell_amount': self.sell_amount,
            'buy_amount': self.buy_amount,
            'price': self.price,
            'effective_price': self.effective_price,
            'gas_price': self.gas_price,
            'estimated_gas': self.estimated_gas,
            'gas_cost_eth': self.gas_cost_eth,
            'sources_count': self.sources_count,
            'route_type': self.route_type.value,
            'dominant_source': self.dominant_source,
        }

@dataclass
class ZeroXQuote:
    """0x executable quote with full details."""
    timestamp: datetime
    chain: str
    sell_token: str
    buy_token: str
    sell_amount: int
    buy_amount: int
    price: float
    guaranteed_price: float
    estimated_price_impact: Optional[float]
    gas_price: int
    estimated_gas: int
    sources: List[Dict]
    to_address: str
    value: str
    
    @property
    def slippage_tolerance_pct(self) -> float:
        """Implied slippage tolerance from guaranteed vs spot price."""
        if self.price > 0:
            return (self.price - self.guaranteed_price) / self.price * 100
        return 0
    
    @property
    def price_impact_severity(self) -> PriceImpactSeverity:
        """Classify price impact severity."""
        impact = abs(self.estimated_price_impact or 0)
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
    def confidence(self) -> QuoteConfidence:
        """Assess quote confidence level."""
        sources_count = len([s for s in self.sources if float(s.get('proportion', 0)) > 0])
        impact = abs(self.estimated_price_impact or 0)
        
        if sources_count == 1 and impact < 0.5:
            return QuoteConfidence.HIGH
        elif sources_count <= 3 and impact < 1.0:
            return QuoteConfidence.MEDIUM
        else:
            return QuoteConfidence.LOW
    
    @property
    def is_executable(self) -> bool:
        """Check if quote has execution data."""
        return bool(self.to_address and self.value is not None)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'chain': self.chain,
            'sell_token': self.sell_token,
            'buy_token': self.buy_token,
            'sell_amount': self.sell_amount,
            'buy_amount': self.buy_amount,
            'price': self.price,
            'guaranteed_price': self.guaranteed_price,
            'estimated_price_impact': self.estimated_price_impact,
            'price_impact_severity': self.price_impact_severity.value,
            'slippage_tolerance_pct': self.slippage_tolerance_pct,
            'gas_price': self.gas_price,
            'estimated_gas': self.estimated_gas,
            'confidence': self.confidence.value,
            'is_executable': self.is_executable,
        }

@dataclass
class LiquiditySource:
    """DEX liquidity source information."""
    name: str
    chain: str
    proportion: float
    
    @property
    def tier(self) -> SourceTier:
        """Classify source tier."""
        tier_1 = ['Uniswap', 'Curve', 'Balancer', 'SushiSwap', 'PancakeSwap']
        tier_2 = ['Kyber', 'Bancor', 'DODO', 'QuickSwap', 'TraderJoe']
        rfq = ['0x RFQ', 'Hashflow', 'Native']
        
        name_lower = self.name.lower()
        
        if any(t.lower() in name_lower for t in rfq):
            return SourceTier.RFQ
        elif any(t.lower() in name_lower for t in tier_1):
            return SourceTier.TIER_1
        elif any(t.lower() in name_lower for t in tier_2):
            return SourceTier.TIER_2
        else:
            return SourceTier.TIER_3
    
    @property
    def percentage(self) -> float:
        """Proportion as percentage."""
        return self.proportion * 100
    
    @property
    def is_significant(self) -> bool:
        """Check if source has significant share (>10%)."""
        return self.proportion > 0.1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'chain': self.chain,
            'proportion': self.proportion,
            'percentage': self.percentage,
            'tier': self.tier.value,
            'is_significant': self.is_significant,
        }

@dataclass
class CrossChainComparison:
    """Cross-chain price comparison data."""
    timestamp: datetime
    token: str
    chain: str
    buy_amount: int
    price: float
    estimated_gas: int
    sources_count: int
    
    @property
    def gas_cost_usd_estimate(self) -> float:
        """Rough gas cost estimate in USD (assumes $2000 ETH, 30 gwei)."""
        return self.estimated_gas * 30 * 2000 / 1e9
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'token': self.token,
            'chain': self.chain,
            'buy_amount': self.buy_amount,
            'price': self.price,
            'estimated_gas': self.estimated_gas,
            'sources_count': self.sources_count,
            'gas_cost_usd_estimate': self.gas_cost_usd_estimate,
        }

@dataclass
class DEXMarketShare:
    """DEX market share analysis for a trade."""
    dex: str
    proportion: float
    chain: str
    sell_amount: int
    
    @property
    def percentage(self) -> float:
        """Share as percentage."""
        return self.proportion * 100
    
    @property
    def share_category(self) -> str:
        """Categorize market share."""
        pct = self.percentage
        if pct > 50:
            return 'dominant'
        elif pct > 25:
            return 'major'
        elif pct > 10:
            return 'significant'
        else:
            return 'minor'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'dex': self.dex,
            'proportion': self.proportion,
            'percentage': self.percentage,
            'chain': self.chain,
            'sell_amount': self.sell_amount,
            'share_category': self.share_category,
        }

@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity detection."""
    timestamp: datetime
    chain: str
    token_a: str
    token_b: str
    forward_price: float
    reverse_price: float
    spread_bps: float
    estimated_gas_cost: float
    
    @property
    def round_trip_profit_bps(self) -> float:
        """Round trip profit in basis points (before gas)."""
        return self.spread_bps
    
    @property
    def viability(self) -> ArbitrageViability:
        """Assess arbitrage viability."""
        # Rough profitability threshold (10 bps minimum after gas)
        if self.spread_bps > 50:
            return ArbitrageViability.PROFITABLE
        elif self.spread_bps > 10:
            return ArbitrageViability.MARGINAL
        else:
            return ArbitrageViability.UNPROFITABLE
    
    @property
    def is_actionable(self) -> bool:
        """Check if opportunity is actionable."""
        return self.viability == ArbitrageViability.PROFITABLE
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'chain': self.chain,
            'token_a': self.token_a,
            'token_b': self.token_b,
            'forward_price': self.forward_price,
            'reverse_price': self.reverse_price,
            'spread_bps': self.spread_bps,
            'viability': self.viability.value,
            'is_actionable': self.is_actionable,
        }

# =============================================================================
# Collector Class
# =============================================================================

class ZeroXCollector(BaseCollector):
    """
    0x Protocol aggregator collector.
    
    validated implementation for DEX aggregation data collection.
    Supports multi-chain price comparison, liquidity analysis, and
    arbitrage detection.
    
    Features:
        - Price and quote fetching
        - Multi-chain support (8 chains)
        - Liquidity source analysis
        - Cross-chain comparison
        - Arbitrage detection
        - Market share tracking
    
    Attributes:
        VENUE: Protocol identifier ('0x')
        VENUE_TYPE: Protocol type ('DEX_AGGREGATOR')
    
    Example:
        >>> config = {'rate_limit': 30}
        >>> async with ZeroXCollector(config) as collector:
        ... price = await collector.get_price(
        ... sell_token, buy_token, amount, 'ethereum'
        ... )
        ... sources = await collector.get_liquidity_sources('ethereum')
    """
    
    VENUE = '0x'
    VENUE_TYPE = 'DEX_AGGREGATOR'
    
    # v2 API uses a single base URL with chainId query parameter
    BASE_URL = 'https://api.0x.org'

    # Chain IDs for v2 API
    CHAIN_IDS = {
        'ethereum': 1,
        'polygon': 137,
        'bsc': 56,
        'arbitrum': 42161,
        'optimism': 10,
        'avalanche': 43114,
        'base': 8453,
        'fantom': 250,
    }
    
    COMMON_TOKENS = {
        'ethereum': {
            'WETH': ('0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2', 18),
            'USDC': ('0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48', 6),
            'USDT': ('0xdAC17F958D2ee523a2206206994597C13D831ec7', 6),
            'DAI': ('0x6B175474E89094C44Da98b954EesdffdD3A92Ca', 18),
            'WBTC': ('0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599', 8),
        },
        'arbitrum': {
            'WETH': ('0x82aF49447D8a07e3bd95BD0d56f35241523fBab1', 18),
            'USDC': ('0xaf88d065e77c8cC2239327C5EDb3A432268e5831', 6),
            'USDT': ('0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9', 6),
            'ARB': ('0x912CE59144191C1204E64559FE8253a0e49E6548', 18),
        },
        'polygon': {
            'WMATIC': ('0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270', 18),
            'USDC': ('0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174', 6),
            'USDT': ('0xc2132D05D31c914a87C6611C10748AEb04B58e8F', 6),
            'WETH': ('0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619', 18),
        },
        'optimism': {
            'WETH': ('0x4200000000000000000000000000000000000006', 18),
            'USDC': ('0x7F5c764cBc14f9669B88837ca1490cCa17c31607', 6),
            'OP': ('0x4200000000000000000000000000000000000042', 18),
        },
        'base': {
            'WETH': ('0x4200000000000000000000000000000000000006', 18),
            'USDC': ('0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913', 6),
        },
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize 0x collector."""
        config = config or {}
        super().__init__(config)

        # CRITICAL: Set supported data types for dynamic routing
        self.supported_data_types = ['swaps', 'routes']
        self.venue = 'zerox'

        # Import VenueType from base_collector
        from ..base_collector import VenueType
        self.venue_type = VenueType.DEX

        # Load API key from config or environment - required for higher rate limits
        self.api_key = config.get('api_key') or config.get('zerox_api_key') or os.getenv('ZEROX_API_KEY', '')
        self.requires_auth = bool(self.api_key) # 0x API works without key (with limits)

        rate_limit = config.get('rate_limit', 30)
        self.rate_limiter = get_shared_rate_limiter('zerox', rate=rate_limit, per=60.0, burst=5)
        self.retry_handler = RetryHandler(max_retries=3, base_delay=1.0)

        self.timeout = aiohttp.ClientTimeout(total=30)
        self.session: Optional[aiohttp.ClientSession] = None

        self.collection_stats = {
            'records_collected': 0, 'api_calls': 0, 'errors': 0
        }
        logger.info(f"Initialized 0x collector (rate_limit={rate_limit}/min, data types: {self.supported_data_types})")
    
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
            headers = {
                'Accept': 'application/json',
                '0x-version': 'v2', # Required for v2 API
            }
            if self.api_key:
                headers['0x-api-key'] = self.api_key
            self.session = aiohttp.ClientSession(timeout=self.timeout, headers=headers)
        return self.session
    
    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
        logger.info(f"0x collector closed. Stats: {self.collection_stats}")
    
    def _get_chain_id(self, chain: str) -> int:
        """Get chain ID for chain name."""
        return self.CHAIN_IDS.get(chain.lower(), self.CHAIN_IDS['ethereum'])
    
    async def get_price(
        self, sell_token: str, buy_token: str,
        sell_amount: str, chain: str = 'ethereum'
    ) -> Optional[ZeroXPrice]:
        """
        Get price quote for a swap.

        Args:
            sell_token: Token address to sell
            buy_token: Token address to buy
            sell_amount: Amount in base units (wei)
            chain: Blockchain network

        Returns:
            ZeroXPrice dataclass with quote details
        """
        await self.rate_limiter.acquire()
        session = await self._get_session()

        # v2 API: unified URL with chainId parameter
        url = f"{self.BASE_URL}/swap/allowance-holder/price"

        params = {
            'chainId': self._get_chain_id(chain),
            'sellToken': sell_token,
            'buyToken': buy_token,
            'sellAmount': sell_amount,
        }

        try:
            async with session.get(url, params=params) as resp:
                self.collection_stats['api_calls'] += 1

                if resp.status == 200:
                    data = await resp.json()

                    # v2 response structure: extract sources from route.fills
                    sources = []
                    route = data.get('route', {})
                    fills = route.get('fills', [])
                    for fill in fills:
                        sources.append({
                            'name': fill.get('source', 'Unknown'),
                            'proportion': float(fill.get('proportionBps', 0)) / 10000
                        })

                    # v2 response: transaction details are in 'transaction' object
                    transaction = data.get('transaction', {})

                    return ZeroXPrice(
                        timestamp=datetime.now(timezone.utc),
                        chain=chain,
                        sell_token=sell_token,
                        buy_token=buy_token,
                        sell_amount=int(sell_amount),
                        buy_amount=int(data.get('buyAmount', 0)),
                        price=float(data.get('price', 0)) if data.get('price') else 0,
                        gas_price=int(transaction.get('gasPrice', 0)),
                        estimated_gas=int(transaction.get('gas', 0)),
                        sources=sources
                    )
                else:
                    error_text = await resp.text()
                    logger.warning(f"0x price error: {resp.status} - {error_text[:200]}")
                    return None

        except Exception as e:
            logger.error(f"Error getting 0x price: {e}")
            self.collection_stats['errors'] += 1
            return None
    
    async def get_quote(
        self, sell_token: str, buy_token: str,
        sell_amount: str, chain: str = 'ethereum',
        taker_address: Optional[str] = None
    ) -> Optional[ZeroXQuote]:
        """
        Get executable quote for a swap.

        Args:
            sell_token: Token address to sell
            buy_token: Token address to buy
            sell_amount: Amount in base units
            chain: Blockchain network
            taker_address: Address executing the swap

        Returns:
            ZeroXQuote dataclass with execution details
        """
        await self.rate_limiter.acquire()
        session = await self._get_session()

        # v2 API: unified URL with chainId parameter
        url = f"{self.BASE_URL}/swap/allowance-holder/quote"

        params = {
            'chainId': self._get_chain_id(chain),
            'sellToken': sell_token,
            'buyToken': buy_token,
            'sellAmount': sell_amount,
        }

        # v2 uses 'taker' instead of 'takerAddress'
        if taker_address:
            params['taker'] = taker_address

        try:
            async with session.get(url, params=params) as resp:
                self.collection_stats['api_calls'] += 1

                if resp.status == 200:
                    data = await resp.json()

                    # v2 response structure: extract sources from route.fills
                    sources = []
                    route = data.get('route', {})
                    fills = route.get('fills', [])
                    for fill in fills:
                        sources.append({
                            'name': fill.get('source', 'Unknown'),
                            'proportion': float(fill.get('proportionBps', 0)) / 10000
                        })

                    # v2 response: transaction details are in 'transaction' object
                    transaction = data.get('transaction', {})

                    return ZeroXQuote(
                        timestamp=datetime.now(timezone.utc),
                        chain=chain,
                        sell_token=sell_token,
                        buy_token=buy_token,
                        sell_amount=int(sell_amount),
                        buy_amount=int(data.get('buyAmount', 0)),
                        price=float(data.get('price', 0)) if data.get('price') else 0,
                        guaranteed_price=float(data.get('minBuyAmount', 0)) / int(sell_amount) if data.get('minBuyAmount') else 0,
                        estimated_price_impact=None, # v2 doesn't provide this directly
                        gas_price=int(transaction.get('gasPrice', 0)),
                        estimated_gas=int(transaction.get('gas', 0)),
                        sources=sources,
                        to_address=transaction.get('to', ''),
                        value=transaction.get('value', '0')
                    )
                else:
                    error_text = await resp.text()
                    logger.warning(f"0x quote error: {resp.status} - {error_text[:200]}")
                    return None

        except Exception as e:
            logger.error(f"Error getting 0x quote: {e}")
            self.collection_stats['errors'] += 1
            return None
    
    async def get_liquidity_sources(self, chain: str = 'ethereum') -> pd.DataFrame:
        """
        Get available liquidity sources for a chain.

        Args:
            chain: Blockchain network

        Returns:
            DataFrame with available DEX sources
        """
        await self.rate_limiter.acquire()
        session = await self._get_session()

        # v2 API: /sources endpoint with chainId parameter
        url = f"{self.BASE_URL}/sources"
        params = {'chainId': self._get_chain_id(chain)}

        try:
            async with session.get(url, params=params) as resp:
                self.collection_stats['api_calls'] += 1

                if resp.status == 200:
                    data = await resp.json()

                    # v2 response: sources may be in 'sources' array or directly
                    source_list = data.get('sources', []) if isinstance(data, dict) else data

                    # If it's a dict with source names as keys, extract names
                    if isinstance(source_list, dict):
                        source_list = list(source_list.keys())

                    sources = []
                    for item in source_list:
                        # Handle both string names and dict objects
                        if isinstance(item, str):
                            name = item
                        elif isinstance(item, dict):
                            name = item.get('name', item.get('source', str(item)))
                        else:
                            name = str(item)

                        ls = LiquiditySource(
                            name=name,
                            chain=chain,
                            proportion=0 # No proportion data in sources endpoint
                        )
                        sources.append({
                            **ls.to_dict(),
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'venue': self.VENUE
                        })

                    self.collection_stats['records_collected'] += len(sources)
                    logger.info(f"0x sources for {chain}: {len(sources)} DEXs found")
                    return pd.DataFrame(sources)
                else:
                    error_text = await resp.text()
                    logger.warning(f"0x sources error: {resp.status} - {error_text[:200]}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error getting liquidity sources: {e}")
            self.collection_stats['errors'] += 1
            return pd.DataFrame()
    
    async def _fetch_single_chain_price(
        self,
        chain: str,
        token_symbol: str,
        sell_amount_usd: float
    ) -> Optional[Dict]:
        """
        Helper to fetch price for a single chain (parallelized).

        Args:
            chain: Chain name
            token_symbol: Token symbol
            sell_amount_usd: Amount in USD

        Returns:
            Price comparison data or None
        """
        try:
            tokens = self.COMMON_TOKENS.get(chain)
            if not tokens or token_symbol not in tokens:
                return None

            token_info = tokens[token_symbol]
            usdc_info = tokens.get('USDC')

            if not usdc_info:
                return None

            token_addr, token_dec = token_info
            usdc_addr, usdc_dec = usdc_info

            # Sell USDC to buy token
            sell_amount = str(int(sell_amount_usd * (10 ** usdc_dec)))

            price = await self.get_price(usdc_addr, token_addr, sell_amount, chain)

            if price and price.buy_amount > 0:
                ccc = CrossChainComparison(
                    timestamp=datetime.now(timezone.utc),
                    token=token_symbol,
                    chain=chain,
                    buy_amount=price.buy_amount,
                    price=price.price,
                    estimated_gas=price.estimated_gas,
                    sources_count=price.sources_count
                )
                return {
                    **ccc.to_dict(),
                    'venue': self.VENUE
                }

            return None

        except Exception as e:
            logger.error(f"Error fetching price for {token_symbol} on {chain}: {e}")
            return None

    async def compare_prices_across_chains(
        self, token_symbol: str, sell_amount_usd: float = 1000.0
    ) -> pd.DataFrame:
        """
        Compare token prices across different chains.

        Args:
            token_symbol: Token symbol (e.g., 'WETH', 'USDC')
            sell_amount_usd: Amount in USD for comparison

        Returns:
            DataFrame with price comparison
        """
        # Get all chains that have this token
        chains = [chain for chain, tokens in self.COMMON_TOKENS.items() if token_symbol in tokens]

        # Parallelize chain price fetching using asyncio.gather
        tasks = [self._fetch_single_chain_price(chain, token_symbol, sell_amount_usd) for chain in chains]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter valid results
        records = [r for r in results if isinstance(r, dict)]

        return pd.DataFrame(records)
    
    async def _fetch_single_amount_market_share(
        self,
        amount: str,
        sell_token: str,
        buy_token: str,
        chain: str
    ) -> List[Dict]:
        """
        Helper to fetch market share for a single amount (parallelized).

        Args:
            amount: Amount to test
            sell_token: Token address to sell
            buy_token: Token address to buy
            chain: Blockchain network

        Returns:
            List of market share data dictionaries
        """
        try:
            price = await self.get_price(sell_token, buy_token, amount, chain)

            records = []
            if price and price.sources:
                for source in price.sources:
                    proportion = float(source.get('proportion', 0))
                    if proportion > 0:
                        dms = DEXMarketShare(
                            dex=source.get('name', 'Unknown'),
                            proportion=proportion,
                            chain=chain,
                            sell_amount=int(amount)
                        )
                        records.append({
                            **dms.to_dict(),
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'venue': self.VENUE
                        })

            return records

        except Exception as e:
            logger.error(f"Error fetching market share for amount {amount}: {e}")
            return []

    async def analyze_dex_market_share(
        self, sell_token: str, buy_token: str,
        amounts: List[str], chain: str = 'ethereum'
    ) -> pd.DataFrame:
        """
        Analyze DEX market share at different trade sizes.

        Args:
            sell_token: Token address to sell
            buy_token: Token address to buy
            amounts: List of sell amounts to test
            chain: Blockchain network

        Returns:
            DataFrame with DEX utilization by trade size
        """
        # Parallelize amount fetching using asyncio.gather
        tasks = [self._fetch_single_amount_market_share(amount, sell_token, buy_token, chain) for amount in amounts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten and filter valid results
        records = []
        for result in results:
            if isinstance(result, list):
                records.extend(result)

        return pd.DataFrame(records)
    
    async def _check_single_pair_arbitrage(
        self,
        token_a: str,
        token_b: str,
        chain: str,
        min_spread_bps: float
    ) -> Optional[ArbitrageOpportunity]:
        """
        Helper to check arbitrage for a single pair (parallelized).

        Args:
            token_a: Token A address
            token_b: Token B address
            chain: Blockchain network
            min_spread_bps: Minimum spread to report

        Returns:
            ArbitrageOpportunity or None
        """
        try:
            # Forward: A -> B
            forward = await self.get_price(token_a, token_b, str(int(1e18)), chain)

            if not forward or not forward.buy_amount:
                return None

            # Reverse: B -> A using forward output
            reverse = await self.get_price(
                token_b, token_a, str(forward.buy_amount), chain
            )

            if not reverse or not reverse.buy_amount:
                return None

            # Calculate round-trip
            initial = 1e18
            final = reverse.buy_amount

            if final > 0:
                spread_bps = ((final - initial) / initial) * 10000

                if spread_bps > min_spread_bps:
                    arb = ArbitrageOpportunity(
                        timestamp=datetime.now(timezone.utc),
                        chain=chain,
                        token_a=token_a,
                        token_b=token_b,
                        forward_price=forward.price,
                        reverse_price=reverse.price,
                        spread_bps=spread_bps,
                        estimated_gas_cost=forward.gas_cost_eth + reverse.gas_cost_eth
                    )
                    return arb

            return None

        except Exception as e:
            logger.error(f"Error checking arbitrage for {token_a}/{token_b}: {e}")
            return None

    async def find_arbitrage_opportunities(
        self, token_pairs: List[Tuple[str, str]],
        chain: str = 'ethereum', min_spread_bps: float = 10.0
    ) -> List[ArbitrageOpportunity]:
        """
        Find potential arbitrage via forward/reverse quotes.

        Args:
            token_pairs: List of (token_a, token_b) address tuples
            chain: Blockchain network
            min_spread_bps: Minimum spread to report

        Returns:
            List of ArbitrageOpportunity objects
        """
        # Parallelize pair checking using asyncio.gather
        tasks = [self._check_single_pair_arbitrage(token_a, token_b, chain, min_spread_bps) for token_a, token_b in token_pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter valid opportunities
        opportunities = [r for r in results if isinstance(r, ArbitrageOpportunity)]

        return opportunities
    
    async def fetch_funding_rates(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """DEX aggregator doesn't have funding rates."""
        logger.info("0x aggregator: No funding rates (spot DEX)")
        return pd.DataFrame()
    
    async def fetch_ohlcv(
        self, symbols: List[str], timeframe: str,
        start_date: str, end_date: str
    ) -> pd.DataFrame:
        """DEX aggregator doesn't provide historical OHLCV."""
        logger.info("0x aggregator: No historical OHLCV")
        return pd.DataFrame()
    
    async def fetch_pools(
        self, chain: str = 'ethereum', min_liquidity: float = 100000
    ) -> pd.DataFrame:
        """Fetch liquidity sources as pool proxies."""
        return await self.get_liquidity_sources(chain)
    
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
            logger.error(f"0x collect_routes error: {e}")
            return pd.DataFrame()

    async def collect_swaps(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect swap data (Note: 0x API doesn't provide historical swap data).

        Returns current price/quote data instead.
        Standardized method name for collection manager compatibility.
        """
        try:
            chain = kwargs.get('chain', 'ethereum')

            # 0x doesn't provide historical swap data through their API
            # Return empty DataFrame with note
            logger.warning("0x API doesn't provide historical swap data - use subgraph or indexer instead")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"0x collect_swaps error: {e}")
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
            logger.error(f"0x collect_funding_rates error: {e}")
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
            logger.error(f"0x collect_ohlcv error: {e}")
            return pd.DataFrame()

    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.collection_stats.copy()

async def test_zerox_collector():
    """Test 0x collector functionality."""
    config = {'rate_limit': 30}
    
    async with ZeroXCollector(config) as collector:
        print("=" * 60)
        print("0x Protocol Collector Test")
        print("=" * 60)
        
        sources = await collector.get_liquidity_sources('ethereum')
        if not sources.empty:
            print(f"\n1. Liquidity sources on Ethereum: {len(sources)}")
            tier1 = sources[sources['tier'] == 'tier_1']
            print(f" Tier 1 DEXs: {len(tier1)}")
        
        print(f"\nStats: {collector.get_collection_stats()}")

if __name__ == '__main__':
    asyncio.run(test_zerox_collector())