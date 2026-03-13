"""
GMX Perpetual DEX Collector - Decentralized Perps on Arbitrum/Avalanche

validated collector for GMX via The Graph subgraphs.
GMX uses a unique GLP (GMX Liquidity Provider) pool model for perpetual trading.

Supported Data Types:
    - Position data (longs/shorts by asset)
    - Funding rates and borrow fees
    - GLP pool composition and pricing
    - Trading volume and fees
    - Liquidation events
    - Open interest by asset

API Documentation:
    - GMX Stats: https://stats.gmx.io/
    - Subgraph: https://thegraph.com/hosted-service/subgraph/gmx-io/gmx-stats

Key Differences from Traditional Perps:
    - No order book - trades against GLP pool
    - Dynamic pricing based on Chainlink oracles
    - Fees vary based on pool utilization
    - GLP holders are counterparty to all trades
    - Hourly funding (not 8h intervals)

Contract Specifications:
    - Maximum leverage: 50x
    - No funding rate cap
    - Position fee: 0.1% (open/close)
    - Borrow fee: Dynamic based on utilization

Statistical Arbitrage Applications:
    - DEX vs CEX funding arbitrage
    - GLP pool imbalance signals
    - Liquidation cascade detection
    - Oracle price deviation analysis
    - Cross-chain arbitrage (Arbitrum vs Avalanche)

Version: 2.0.0
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

from ..base_collector import BaseCollector
from ..utils.rate_limiter import get_shared_rate_limiter
from ..utils.retry_handler import RetryHandler, safe_float, safe_int

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================

class Chain(Enum):
    """Supported chains for GMX."""
    ARBITRUM = 'arbitrum'
    AVALANCHE = 'avalanche'

class GMXVersion(Enum):
    """GMX protocol versions."""
    V1 = 'v1'
    V2 = 'v2'

class PositionSide(Enum):
    """Position side."""
    LONG = 'long'
    SHORT = 'short'

class FundingMechanism(Enum):
    """Funding mechanism type."""
    BORROW_FEE = 'borrow_fee'
    FUNDING_RATE = 'funding_rate'

class PoolHealthStatus(Enum):
    """GLP pool health status."""
    HEALTHY = 'healthy' # Balanced utilization
    IMBALANCED = 'imbalanced' # Significant long/short skew
    STRESSED = 'stressed' # High utilization
    CRITICAL = 'critical' # Near capacity

class LiquidationSeverity(Enum):
    """Liquidation event severity."""
    MINOR = 'minor' # < $100K
    MODERATE = 'moderate' # $100K - $500K
    SIGNIFICANT = 'significant' # $500K - $1M
    MAJOR = 'major' # > $1M

class OITrend(Enum):
    """Open interest trend."""
    STRONG_INCREASE = 'strong_increase'
    INCREASE = 'increase'
    STABLE = 'stable'
    DECREASE = 'decrease'
    STRONG_DECREASE = 'strong_decrease'

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class GMXFundingRate:
    """
    GMX funding/borrow rate data.
    
    GMX uses borrow fees instead of traditional funding:
    - Longs pay borrow fee based on utilization
    - Shorts pay borrow fee based on utilization
    - Rate = (assets borrowed / total assets in pool) * base rate
    """
    timestamp: datetime
    symbol: str
    chain: str
    funding_rate_long: float
    funding_rate_short: float
    borrow_rate_long: float
    borrow_rate_short: float
    
    @property
    def net_funding_rate(self) -> float:
        """Net funding rate (long - short)."""
        return (self.funding_rate_long - self.funding_rate_short) / 2
    
    @property
    def annualized_rate(self) -> float:
        """Annualized net funding rate (hourly snapshots)."""
        return self.net_funding_rate * 8760
    
    @property
    def annualized_rate_pct(self) -> float:
        """Annualized rate as percentage."""
        return self.annualized_rate * 100
    
    @property
    def total_borrow_cost_long(self) -> float:
        """Total cost for longs (funding + borrow)."""
        return self.funding_rate_long + self.borrow_rate_long
    
    @property
    def total_borrow_cost_short(self) -> float:
        """Total cost for shorts (funding + borrow)."""
        return self.funding_rate_short + self.borrow_rate_short
    
    @property
    def long_short_spread(self) -> float:
        """Spread between long and short costs."""
        return self.total_borrow_cost_long - self.total_borrow_cost_short
    
    @property
    def eight_hour_equivalent(self) -> float:
        """8-hour equivalent for comparison with CEX perps."""
        return self.net_funding_rate * 8
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'chain': self.chain,
            'funding_rate': self.net_funding_rate, # Standard column for cross-venue compatibility
            'funding_rate_long': self.funding_rate_long,
            'funding_rate_short': self.funding_rate_short,
            'borrow_rate_long': self.borrow_rate_long,
            'borrow_rate_short': self.borrow_rate_short,
            'net_funding_rate': self.net_funding_rate,
            'annualized_rate': self.annualized_rate,
            'annualized_rate_pct': self.annualized_rate_pct,
            'total_borrow_cost_long': self.total_borrow_cost_long,
            'total_borrow_cost_short': self.total_borrow_cost_short,
            'long_short_spread': self.long_short_spread,
            'eight_hour_equivalent': self.eight_hour_equivalent,
            'funding_mechanism': FundingMechanism.BORROW_FEE.value,
            'funding_interval_hours': 1, # GMX uses continuous/hourly funding
            'venue': 'gmx',
            'venue_type': 'dex',
        }

@dataclass
class GMXOpenInterest:
    """GMX open interest data."""
    timestamp: datetime
    symbol: str
    chain: str
    long_oi: float
    short_oi: float
    long_oi_usd: float
    short_oi_usd: float
    
    @property
    def total_oi(self) -> float:
        """Total open interest in contracts."""
        return self.long_oi + self.short_oi
    
    @property
    def total_oi_usd(self) -> float:
        """Total open interest in USD."""
        return self.long_oi_usd + self.short_oi_usd
    
    @property
    def long_short_ratio(self) -> float:
        """Long/short ratio by USD value."""
        return self.long_oi_usd / self.short_oi_usd if self.short_oi_usd > 0 else float('inf')
    
    @property
    def long_pct(self) -> float:
        """Long percentage of total OI."""
        total = self.long_oi_usd + self.short_oi_usd
        return (self.long_oi_usd / total * 100) if total > 0 else 50
    
    @property
    def short_pct(self) -> float:
        """Short percentage of total OI."""
        return 100 - self.long_pct
    
    @property
    def imbalance_pct(self) -> float:
        """Position imbalance percentage (positive = more longs)."""
        return self.long_pct - self.short_pct
    
    @property
    def is_heavily_skewed(self) -> bool:
        """Check if positions are heavily skewed (>70% one side)."""
        return self.long_pct > 70 or self.short_pct > 70
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'chain': self.chain,
            'long_oi': self.long_oi,
            'short_oi': self.short_oi,
            'long_oi_usd': self.long_oi_usd,
            'short_oi_usd': self.short_oi_usd,
            'total_oi_usd': self.total_oi_usd,
            'long_short_ratio': self.long_short_ratio,
            'long_pct': self.long_pct,
            'short_pct': self.short_pct,
            'imbalance_pct': self.imbalance_pct,
            'is_heavily_skewed': self.is_heavily_skewed,
        }

@dataclass
class GLPStats:
    """
    GLP (GMX Liquidity Provider) pool statistics.
    
    GLP is the counterparty to all trades on GMX.
    Understanding GLP composition helps predict:
    - Available liquidity
    - Fee rates
    - Pool imbalances
    """
    timestamp: datetime
    chain: str
    aum_usd: float
    glp_supply: float
    distributed_eth: float
    distributed_eth_cumulative: float
    distributed_usd: float
    distributed_usd_cumulative: float
    
    @property
    def glp_price(self) -> float:
        """GLP token price."""
        return self.aum_usd / self.glp_supply if self.glp_supply > 0 else 0
    
    @property
    def daily_yield_eth(self) -> float:
        """Daily ETH yield per GLP."""
        return self.distributed_eth / self.glp_supply if self.glp_supply > 0 else 0
    
    @property
    def daily_yield_usd(self) -> float:
        """Daily USD yield per GLP."""
        return self.distributed_usd / self.glp_supply if self.glp_supply > 0 else 0
    
    @property
    def apy_estimate(self) -> float:
        """Estimated APY based on recent distributions."""
        if self.glp_price > 0:
            daily_yield_pct = (self.daily_yield_usd / self.glp_price) * 100
            return daily_yield_pct * 365
        return 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'chain': self.chain,
            'aum_usd': self.aum_usd,
            'glp_supply': self.glp_supply,
            'glp_price': self.glp_price,
            'distributed_eth': self.distributed_eth,
            'distributed_eth_cumulative': self.distributed_eth_cumulative,
            'distributed_usd': self.distributed_usd,
            'distributed_usd_cumulative': self.distributed_usd_cumulative,
            'daily_yield_eth': self.daily_yield_eth,
            'daily_yield_usd': self.daily_yield_usd,
            'apy_estimate': self.apy_estimate,
        }

@dataclass
class GMXLiquidation:
    """GMX liquidation event data."""
    timestamp: datetime
    chain: str
    account: str
    collateral_token: str
    index_token: str
    is_long: bool
    size_usd: float
    collateral_usd: float
    realized_pnl: float
    mark_price: float
    
    @property
    def leverage_at_liquidation(self) -> float:
        """Estimated leverage at liquidation."""
        return self.size_usd / self.collateral_usd if self.collateral_usd > 0 else 0
    
    @property
    def loss_pct(self) -> float:
        """Loss percentage of collateral."""
        return (abs(self.realized_pnl) / self.collateral_usd * 100) if self.collateral_usd > 0 else 0
    
    @property
    def severity(self) -> LiquidationSeverity:
        """Classify liquidation severity."""
        if self.size_usd < 100_000:
            return LiquidationSeverity.MINOR
        elif self.size_usd < 500_000:
            return LiquidationSeverity.MODERATE
        elif self.size_usd < 1_000_000:
            return LiquidationSeverity.SIGNIFICANT
        else:
            return LiquidationSeverity.MAJOR
    
    @property
    def position_side(self) -> str:
        """Position side string."""
        return 'long' if self.is_long else 'short'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'chain': self.chain,
            'account': self.account,
            'collateral_token': self.collateral_token,
            'index_token': self.index_token,
            'is_long': self.is_long,
            'position_side': self.position_side,
            'size_usd': self.size_usd,
            'collateral_usd': self.collateral_usd,
            'realized_pnl': self.realized_pnl,
            'mark_price': self.mark_price,
            'leverage_at_liquidation': self.leverage_at_liquidation,
            'loss_pct': self.loss_pct,
            'severity': self.severity.value,
        }

@dataclass
class GMXVolumeStats:
    """GMX daily volume statistics."""
    timestamp: datetime
    chain: str
    margin_volume_usd: float
    swap_volume_usd: float
    liquidation_volume_usd: float
    mint_volume_usd: float
    burn_volume_usd: float
    
    @property
    def total_volume_usd(self) -> float:
        """Total trading volume."""
        return self.margin_volume_usd + self.swap_volume_usd
    
    @property
    def margin_pct(self) -> float:
        """Margin trading percentage of total."""
        total = self.total_volume_usd
        return (self.margin_volume_usd / total * 100) if total > 0 else 0
    
    @property
    def net_glp_flow(self) -> float:
        """Net GLP flow (mints - burns)."""
        return self.mint_volume_usd - self.burn_volume_usd
    
    @property
    def is_net_outflow(self) -> bool:
        """Check if net GLP outflow (redemptions > mints)."""
        return self.net_glp_flow < 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'chain': self.chain,
            'margin_volume_usd': self.margin_volume_usd,
            'swap_volume_usd': self.swap_volume_usd,
            'liquidation_volume_usd': self.liquidation_volume_usd,
            'mint_volume_usd': self.mint_volume_usd,
            'burn_volume_usd': self.burn_volume_usd,
            'total_volume_usd': self.total_volume_usd,
            'margin_pct': self.margin_pct,
            'net_glp_flow': self.net_glp_flow,
            'is_net_outflow': self.is_net_outflow,
        }

# =============================================================================
# Collector Class
# =============================================================================

class GMXCollector(BaseCollector):
    """
    GMX perpetual DEX data collector via The Graph.

    validated implementation for decentralized perpetuals.

    Features:
    - Arbitrum (GMX V1 and V2)
    - Avalanche (GMX V1)
    - GLP pool model (no order book)
    - Borrow fee funding mechanism
    - Oracle-based pricing (Chainlink)

    Attributes:
        VENUE: Protocol identifier ('gmx')
        VENUE_TYPE: Protocol type ('DEX')
        SUBGRAPH_URLS: Chain-specific subgraph endpoints

    Example:
        >>> config = {'graph_api_key': 'YOUR_KEY'}
        >>> async with GMXCollector(config) as collector:
        ... funding = await collector.fetch_funding_rates(['BTC', 'ETH'], '2024-01-01', '2024-01-31')
        ... glp = await collector.fetch_glp_stats('2024-01-01', '2024-01-31')
    """

    VENUE = 'gmx'
    VENUE_TYPE = 'DEX'

    # CRITICAL: Short timeouts to prevent hanging on DNS/connection failures
    CONNECTION_TIMEOUT = 10 # 10 seconds for connection (DNS resolution + TCP)
    TOTAL_TIMEOUT = 30 # 30 seconds total per request
    DNS_FAILURE_SKIP = True # Skip endpoint immediately on DNS failure (no retries)
    MAX_PAGINATION_ITERATIONS = 100 # Safety limit for pagination loops to prevent infinite loops

    # REST API endpoints for current market data (with fallbacks)
    REST_API_URLS = {
        'arbitrum': 'https://arbitrum-api.gmxinfra.io',
        'avalanche': 'https://avalanche-api.gmxinfra.io',
    }

    # REST API fallback endpoints (if primary fails)
    REST_API_URLS_FALLBACK = {
        'arbitrum': [
            'https://arbitrum-api-fallback.gmxinfra.io',
            'https://arbitrum-api-fallback.gmxinfra2.io',
        ],
        'avalanche': [
            'https://avalanche-api-fallback.gmxinfra.io',
            'https://avalanche-api-fallback.gmxinfra2.io',
        ],
    }

    # CRITICAL UPDATE 2026-02: Working subgraph endpoints
    # Priority: 1) Subsquid (official GMX), 2) The Graph Network, 3) Graph Studio
    SUBGRAPH_URLS = {
        # GMX V2 Synthetics on Arbitrum - Subsquid (PREFERRED - official GMX)
        'arbitrum_v2': 'https://gmx.squids.live/gmx-synthetics-arbitrum:prod/api/graphql',
        # GMX V1 Stats - The Graph decentralized network (requires API key)
        # Will be dynamically constructed with API key at runtime
        'arbitrum_v1': None, # Set in __init__ if API key available
        # Avalanche GMX Stats
        'avalanche_v1': None, # Set in __init__ if API key available
    }

    # The Graph subgraph IDs for decentralized network queries
    THEGRAPH_SUBGRAPH_IDS = {
        'arbitrum_v1': 'DiR5cWwB3pwXXQWWdus7fDLR2mnFRQLiBFsVmHAH9VAs', # GMX Arbitrum (Messari)
        'avalanche_v1': '6pXgnXcL6mkXBjKX7NyHN7tCudv2JGFnXZ8wf8WbjPXv', # GMX Avalanche
    }

    # Fallback: Graph Studio endpoints (rate-limited, no API key needed)
    SUBGRAPH_URLS_FALLBACK = {
        'arbitrum_v1': 'https://api.studio.thegraph.com/query/44700/gmx-arbitrum-stats/version/latest',
        'arbitrum_v2': 'https://api.studio.thegraph.com/query/44700/synthetics-arbitrum-stats/version/latest',
        'avalanche_v1': 'https://api.studio.thegraph.com/query/44700/gmx-avalanche-stats/version/latest',
    }

    GMX_ASSETS = {
        'arbitrum': ['BTC', 'ETH', 'LINK', 'UNI', 'ARB', 'SOL', 'DOGE', 'AVAX', 'OP', 'ATOM', 'NEAR', 'AAVE', 'LTC', 'GMX'],
        'avalanche': ['BTC', 'ETH', 'AVAX'],
    }
    
    FUNDING_INTERVAL_HOURS = 1
    
    def __init__(self, config: Dict):
        """Initialize GMX collector."""
        super().__init__(config)

        # Load Graph API key from config or environment
        import os
        self.graph_api_key = config.get('graph_api_key', '') or os.getenv('THE_GRAPH_API_KEY', '')

        # CRITICAL: Set supported data types for dynamic routing (collection_manager)
        self.supported_data_types = ['funding_rates', 'ohlcv', 'open_interest', 'liquidations', 'pool_data', 'positions']
        self.venue = 'gmx'
        self.requires_auth = False # Public API endpoints

        rate_limit = config.get('rate_limit', 15)
        # Use shared rate limiter to avoid re-initialization overhead
        self.rate_limiter = get_shared_rate_limiter('gmx', rate=rate_limit, per=60.0, burst=5)
        # CRITICAL: Reduced retries to prevent hanging on failing endpoints
        self.retry_handler = RetryHandler(max_retries=2, base_delay=1.0, max_delay=10.0)

        # CRITICAL: Short timeouts to prevent hanging on DNS/connection failures
        self.timeout = aiohttp.ClientTimeout(
            total=self.TOTAL_TIMEOUT,
            connect=self.CONNECTION_TIMEOUT,
            sock_connect=self.CONNECTION_TIMEOUT,
            sock_read=self.TOTAL_TIMEOUT
        )
        self.session: Optional[aiohttp.ClientSession] = None
        # Track failed endpoints to avoid repeated failures
        self._failed_endpoints: set = set()

        self.collection_stats = {'records_collected': 0, 'api_calls': 0, 'errors': 0}

        # Build The Graph Network URLs if API key is available
        self._subgraph_urls = dict(self.SUBGRAPH_URLS) # Copy class-level URLs
        if self.graph_api_key:
            # Construct decentralized network URLs with API key
            for key, subgraph_id in self.THEGRAPH_SUBGRAPH_IDS.items():
                if subgraph_id:
                    self._subgraph_urls[key] = (
                        f"https://gateway-arbitrum.network.thegraph.com/api/{self.graph_api_key}"
                        f"/subgraphs/id/{subgraph_id}"
                    )
            logger.info(f"GMX collector initialized with Graph API key (decentralized network enabled)")
        else:
            logger.info(f"GMX collector initialized (no Graph API key - using Subsquid + Graph Studio)")
    
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
                limit=50, # Total connection pool size (was 10)
                limit_per_host=15, # Per-host connections
                ttl_dns_cache=300, # DNS cache TTL in seconds
                force_close=False, # Keep-alive connections
                enable_cleanup_closed=True
            )
            self.session = aiohttp.ClientSession(timeout=self.timeout, connector=connector)
        return self.session
    
    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
        logger.info(f"GMX collector closed. Stats: {self.collection_stats}")

    async def _fetch_rest_api(self, chain: str, endpoint: str, params: Dict = None) -> Any:
        """Fetch data from GMX REST API with graceful error handling and fallbacks."""
        # Build list of URLs to try (primary + fallbacks)
        urls_to_try = []

        base_url = self.REST_API_URLS.get(chain)
        if base_url and base_url not in self._failed_endpoints:
            urls_to_try.append(base_url)

        # Add fallback URLs
        fallbacks = self.REST_API_URLS_FALLBACK.get(chain, [])
        for fb_url in fallbacks:
            if fb_url not in self._failed_endpoints:
                urls_to_try.append(fb_url)

        if not urls_to_try:
            logger.debug(f"GMX REST API: All endpoints for {chain} unavailable")
            return None

        session = await self._get_session()

        for base_url in urls_to_try:
            url = f"{base_url}{endpoint}"

            try:
                acquire_result = await self.rate_limiter.acquire(timeout=120.0)
                if hasattr(acquire_result, 'acquired') and not acquire_result.acquired:
                    logger.debug(f"GMX rate limiter timeout for {endpoint}")
                    continue # Try next URL
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        self.collection_stats['api_calls'] += 1
                        return await resp.json()
                    else:
                        logger.debug(f"HTTP {resp.status} from GMX REST API ({base_url[:30]}...)")
                        continue # Try next URL

            except aiohttp.ClientConnectorError as e:
                # DNS/connection failures - mark endpoint as failed, try next
                error_str = str(e).lower()
                if 'dns' in error_str or 'getaddrinfo' in error_str or 'name resolution' in error_str:
                    logger.info(f"GMX REST API DNS failure for {base_url[:40]}... - trying fallback")
                    self._failed_endpoints.add(base_url)
                continue # Try next URL

            except asyncio.TimeoutError:
                logger.debug(f"GMX REST API timeout for {base_url[:40]}...")
                continue # Try next URL

            except Exception as e:
                logger.debug(f"GMX REST API error ({base_url[:30]}...): {e}")
                continue # Try next URL

        # All URLs failed
        self.collection_stats['errors'] += 1
        return None

    async def _query_subgraph(self, chain: str, query: str, variables: Dict = None, version: str = 'v1') -> Dict:
        """
        Execute GraphQL query against GMX subgraph with fallback support.

        CRITICAL FIX: Uses updated subgraph endpoints with automatic fallback.
        Priority: 1) Subsquid (official), 2) The Graph Network, 3) Graph Studio
        Handles DNS failures gracefully by skipping to next endpoint immediately.
        """
        url_key = f"{chain}_{version}"
        urls_to_try = []

        # Primary URL from instance (includes dynamically constructed Graph Network URLs)
        if url_key in self._subgraph_urls:
            url = self._subgraph_urls[url_key]
            if url and url not in self._failed_endpoints:
                urls_to_try.append(url)

        # Fallback URL (Graph Studio - rate limited but no API key needed)
        if url_key in self.SUBGRAPH_URLS_FALLBACK:
            url = self.SUBGRAPH_URLS_FALLBACK[url_key]
            if url and url not in self._failed_endpoints:
                urls_to_try.append(url)

        if not urls_to_try:
            # All endpoints have failed - log once and return empty
            logger.debug(f"GMX subgraph: All endpoints for {url_key} unavailable (DNS/connection failures)")
            return {}

        session = await self._get_session()
        payload = {'query': query}
        if variables:
            payload['variables'] = variables

        # Add Graph API key header if available (better rate limits)
        headers = {}
        if self.graph_api_key:
            headers['Authorization'] = f'Bearer {self.graph_api_key}'

        for url in urls_to_try:
            try:
                acquire_result = await self.rate_limiter.acquire(timeout=120.0)
                if hasattr(acquire_result, 'acquired') and not acquire_result.acquired:
                    logger.debug(f"GMX rate limiter timeout for GraphQL query")
                    continue # Try next URL

                async with session.post(url, json=payload, headers=headers if headers else None) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if 'errors' in result:
                            error_msg = str(result['errors'])[:200]
                            logger.debug(f"GraphQL errors from {url}: {error_msg}")
                            continue # Try next URL
                        self.collection_stats['api_calls'] += 1
                        return result.get('data', {})
                    else:
                        logger.debug(f"HTTP {resp.status} from GMX subgraph: {url}")
                        continue # Try next URL

            except aiohttp.ClientConnectorError as e:
                # CRITICAL: DNS failures and connection errors - skip immediately, no retries
                error_str = str(e).lower()
                if 'dns' in error_str or 'getaddrinfo' in error_str or 'name resolution' in error_str:
                    logger.warning(f"GMX subgraph DNS failure for {url[:50]}... - marking as unavailable")
                    self._failed_endpoints.add(url)
                else:
                    logger.debug(f"GMX subgraph connection error for {url[:50]}...: {e}")
                continue # Try next URL immediately

            except asyncio.TimeoutError:
                # Timeout - skip to next endpoint
                logger.debug(f"GMX subgraph timeout for {url[:50]}...")
                continue

            except Exception as e:
                logger.debug(f"GMX subgraph query failed for {url[:50]}...: {e}")
                continue

        # All URLs failed - don't spam logs, just return empty
        self.collection_stats['errors'] += 1
        return {}
    
    async def _fetch_single_chain_funding_rates(
        self, chain: str, symbols: List[str], now: datetime
    ) -> List[Dict]:
        """Fetch funding rates for a single chain from REST API (current snapshot)."""
        logger.info(f"Fetching GMX funding rates from {chain} via REST API")

        all_data = []

        # Fetch current market info (includes funding/borrowing rates)
        data = await self._fetch_rest_api(chain, '/markets/info')
        if not data:
            return []

        # Parse markets data
        markets = data if isinstance(data, list) else data.get('markets', [])
        for market in markets:
            # Extract token symbol from market name (format: "LINK/USD [ETH-USDC]")
            market_name = market.get('name', '')
            token_symbol = ''

            # Parse symbol from name like "BTC/USD [WBTC.b-USDC]" -> "BTC"
            if '/' in market_name:
                token_symbol = market_name.split('/')[0].strip().upper()

            # Also try alternative field names as fallback
            if not token_symbol:
                index_token = market.get('indexToken', {})
                if isinstance(index_token, dict):
                    token_symbol = index_token.get('symbol', '').upper()
            if not token_symbol:
                token_symbol = market.get('indexTokenSymbol', '').upper()

            if not token_symbol:
                continue

            # Filter by requested symbols
            if not any(s.upper() in token_symbol or token_symbol in s.upper() for s in symbols):
                continue

            # Extract funding/borrowing rates (values are typically in basis points or percentages)
            funding_rate_long = safe_float(market.get('fundingRateLong', 0))
            funding_rate_short = safe_float(market.get('fundingRateShort', 0))
            borrow_rate_long = safe_float(market.get('borrowingRateLong') or market.get('borrowRateLong', 0))
            borrow_rate_short = safe_float(market.get('borrowingRateShort') or market.get('borrowRateShort', 0))

            # Normalize if values are in raw format (1e30) - check absolute value for negative rates
            if abs(funding_rate_long) > 1e10:
                funding_rate_long /= 1e30
            if abs(funding_rate_short) > 1e10:
                funding_rate_short /= 1e30
            if abs(borrow_rate_long) > 1e10:
                borrow_rate_long /= 1e30
            if abs(borrow_rate_short) > 1e10:
                borrow_rate_short /= 1e30

            fr = GMXFundingRate(
                timestamp=now,
                symbol=token_symbol,
                chain=chain,
                funding_rate_long=funding_rate_long,
                funding_rate_short=funding_rate_short,
                borrow_rate_long=borrow_rate_long,
                borrow_rate_short=borrow_rate_short
            )

            all_data.append({
                **fr.to_dict(),
                'funding_interval_hours': self.FUNDING_INTERVAL_HOURS,
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE
            })

        return all_data

    async def _fetch_historical_funding_subgraph(
        self, chain: str, symbols: List[str], start_ts: int, end_ts: int
    ) -> List[Dict]:
        """
        Fetch historical funding rates from GMX subgraph.

        CRITICAL FIX: Uses updated subgraph endpoints for historical data.
        The REST API only provides current snapshots - subgraph has history.
        """
        all_data = []

        # GraphQL query for historical funding rates
        query = """
        query GetFundingRates($startTime: Int!, $endTime: Int!, $skip: Int!) {
            fundingRates(
                first: 1000, skip: $skip,
                where: { timestamp_gte: $startTime, timestamp_lte: $endTime },
                orderBy: timestamp, orderDirection: asc
            ) {
                id timestamp token fundingRateLong fundingRateShort
            }
        }
        """

        skip = 0
        max_iterations = 100 # Safety limit

        for _ in range(max_iterations):
            variables = {'startTime': start_ts, 'endTime': end_ts, 'skip': skip}

            data = await self._query_subgraph(chain, query, variables, version='v2')
            funding_data = data.get('fundingRates', [])

            if not funding_data:
                break

            for record in funding_data:
                try:
                    token = record.get('token', '').upper()

                    # Filter by requested symbols
                    if symbols and not any(s.upper() in token or token in s.upper() for s in symbols):
                        continue

                    timestamp = pd.to_datetime(int(record['timestamp']), unit='s', utc=True)
                    funding_long = safe_float(record.get('fundingRateLong', 0))
                    funding_short = safe_float(record.get('fundingRateShort', 0))

                    # Normalize if in raw format
                    if abs(funding_long) > 1e10:
                        funding_long /= 1e30
                    if abs(funding_short) > 1e10:
                        funding_short /= 1e30

                    fr = GMXFundingRate(
                        timestamp=timestamp,
                        symbol=token,
                        chain=chain,
                        funding_rate_long=funding_long,
                        funding_rate_short=funding_short,
                        borrow_rate_long=0, # Not in this query
                        borrow_rate_short=0
                    )

                    all_data.append({
                        **fr.to_dict(),
                        'funding_interval_hours': self.FUNDING_INTERVAL_HOURS,
                        'venue': self.VENUE,
                        'venue_type': self.VENUE_TYPE,
                        'data_source': 'subgraph'
                    })

                except Exception as e:
                    logger.debug(f"Error parsing funding record: {e}")
                    continue

            skip += 1000
            if len(funding_data) < 1000:
                break

        return all_data

    async def fetch_funding_rates(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch GMX funding/borrow rates.

        CRITICAL FIX: Now attempts to fetch historical data from subgraph first,
        then falls back to current snapshot from REST API if no historical data.

        Note: GMX uses borrow fees instead of traditional funding rates.

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH'])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with funding rate data
        """
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        now = datetime.now(timezone.utc)

        all_data = []
        chains = ['arbitrum', 'avalanche']

        # First try to get historical data from subgraph
        logger.info(f"Fetching GMX historical funding rates from subgraph...")
        for chain in chains:
            try:
                historical = await self._fetch_historical_funding_subgraph(
                    chain, symbols, start_ts, end_ts
                )
                if historical:
                    all_data.extend(historical)
                    logger.info(f" {chain}: {len(historical)} historical funding records from subgraph")
            except Exception as e:
                logger.debug(f" {chain}: Subgraph historical fetch failed: {e}")

        # If no historical data, fall back to current snapshot from REST API
        if not all_data:
            logger.info("No historical data from subgraph, fetching current snapshot from REST API...")
            tasks = [
                self._fetch_single_chain_funding_rates(chain, symbols, now)
                for chain in chains
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, list):
                    all_data.extend(r)

        if not all_data:
            logger.warning("No GMX funding rate data collected from any source")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = df.sort_values(['timestamp', 'symbol', 'chain']).reset_index(drop=True)
        self.collection_stats['records_collected'] += len(df)

        logger.info(f"Collected {len(df)} total GMX funding rate records")
        return df
    
    async def _fetch_single_chain_symbol_ohlcv(
        self, chain: str, symbol: str, period: str, limit: int
    ) -> List[Dict]:
        """Fetch OHLCV for a single symbol on a single chain."""
        # Get supported assets for this chain
        chain_assets = self.GMX_ASSETS.get(chain, [])

        token = symbol.upper()

        # Check if token is supported on this chain
        if token not in chain_assets:
            return []

        logger.info(f"Fetching GMX OHLCV for {token} on {chain} via REST API")

        params = {
            'tokenSymbol': token,
            'period': period,
            'limit': limit
        }

        data = await self._fetch_rest_api(chain, '/prices/candles', params)
        if not data:
            return []

        all_data = []

        # Parse candles - data structure may be a list or dict with 'candles' key
        candles = data if isinstance(data, list) else data.get('candles', [])

        for candle in candles:
            try:
                # Handle different candle formats:
                # - List format: [timestamp, open, high, low, close, volume]
                # - Dict format: {timestamp, open, high, low, close, volume}
                if isinstance(candle, list):
                    if len(candle) >= 5:
                        ts = candle[0]
                        o = float(candle[1]) if candle[1] is not None else 0
                        h = float(candle[2]) if candle[2] is not None else 0
                        l = float(candle[3]) if candle[3] is not None else 0
                        c = float(candle[4]) if candle[4] is not None else 0
                        v = float(candle[5]) if len(candle) > 5 and candle[5] is not None else 0
                    else:
                        continue
                else:
                    # Dict format
                    ts = candle.get('timestamp') or candle.get('t')
                    o = float(candle.get('open', candle.get('o', 0)))
                    h = float(candle.get('high', candle.get('h', 0)))
                    l = float(candle.get('low', candle.get('l', 0)))
                    c = float(candle.get('close', candle.get('c', 0)))
                    v = float(candle.get('volume', candle.get('v', 0)))

                # Handle timestamp formats
                if isinstance(ts, str):
                    ts = int(ts)
                ts_dt = pd.to_datetime(ts, unit='s', utc=True) if ts > 1e9 else pd.to_datetime(ts, unit='ms', utc=True)

                # Normalize if values are in raw format (1e30)
                if o > 1e10:
                    o, h, l, c = o / 1e30, h / 1e30, l / 1e30, c / 1e30
                if v > 1e20:
                    v /= 1e30

                all_data.append({
                    'timestamp': ts_dt,
                    'symbol': token,
                    'chain': chain,
                    'open': o,
                    'high': h,
                    'low': l,
                    'close': c,
                    'volume': v,
                    'price_source': 'chainlink_oracle',
                    'venue': self.VENUE,
                    'venue_type': self.VENUE_TYPE
                })
            except Exception as e:
                logger.warning(f"Error parsing GMX candle: {e}")
                continue

        return all_data

    async def fetch_ohlcv(self, symbols: List[str], timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch OHLCV data from GMX REST API (oracle prices).

        Note: GMX uses Chainlink oracle prices, not order book.
        """
        # Map timeframe to API period format
        period_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '4h': '4h', '1d': '1d'}
        period = period_map.get(timeframe, '1h')

        # Calculate limit based on date range
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days = (end_dt - start_dt).days + 1
        limit_map = {'1m': days * 1440, '5m': days * 288, '15m': days * 96, '1h': days * 24, '4h': days * 6, '1d': days}
        limit = min(limit_map.get(period, days * 24), 1000) # API max is 10000

        # Parallelize fetching across chains and symbols
        chains = ['arbitrum', 'avalanche']
        tasks = [
            self._fetch_single_chain_symbol_ohlcv(chain, symbol, period, limit)
            for chain in chains
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten and filter valid results
        all_data = []
        for r in results:
            if isinstance(r, list):
                all_data.extend(r)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = df.sort_values(['timestamp', 'symbol', 'chain']).reset_index(drop=True)
        return df
    
    async def fetch_open_interest(self, symbols: List[str], start_date: str, end_date: str, chain: str = 'arbitrum') -> pd.DataFrame:
        """
        Fetch open interest data from GMX REST API.

        Uses /markets/info endpoint which includes current open interest per market.
        Note: This returns current snapshot, not historical data.
        """
        all_data = []
        now = datetime.now(timezone.utc)

        logger.info(f"Fetching GMX open interest from {chain} via REST API")

        # Use REST API which has current open interest
        data = await self._fetch_rest_api(chain, '/markets/info')

        if data:
            markets = data if isinstance(data, list) else data.get('markets', [])

            for market in markets:
                try:
                    # Extract token symbol from market name (format: "BTC/USD [WBTC.b-USDC]")
                    market_name = market.get('name', '')
                    if '/' not in market_name:
                        continue

                    token = market_name.split('/')[0].strip().upper()

                    # Filter by requested symbols - check if market starts with symbol/
                    # This prevents matching "LDO/USD [ETH-USDC]" when searching for ETH
                    if symbols:
                        symbol_matches = any(
                            market_name.upper().startswith(f"{s.upper()}/")
                            for s in symbols
                        )
                        if not symbol_matches:
                            continue

                    # Extract open interest data (values are in 30 decimals)
                    long_oi_raw = market.get('openInterestLong', 0) or 0
                    short_oi_raw = market.get('openInterestShort', 0) or 0

                    long_oi_usd = float(long_oi_raw) / 1e30
                    short_oi_usd = float(short_oi_raw) / 1e30
                    total_oi_usd = long_oi_usd + short_oi_usd

                    record = {
                        'timestamp': now,
                        'symbol': token,
                        'chain': chain,
                        'long_open_interest': long_oi_usd,
                        'short_open_interest': short_oi_usd,
                        'total_open_interest_usd': total_oi_usd,
                        'open_interest': total_oi_usd, # Alias for compatibility
                        'market_name': market_name,
                        'venue': self.VENUE,
                        'venue_type': self.VENUE_TYPE,
                    }
                    all_data.append(record)
                    self.collection_stats['records_collected'] += 1

                except Exception as e:
                    logger.debug(f"Error parsing OI for market: {e}")
                    continue

        if all_data:
            df = pd.DataFrame(all_data)
            logger.info(f"Fetched {len(df)} GMX open interest records")
            return df

        # Fallback: Try deprecated subgraph (may not work)
        logger.warning("REST API returned no OI data, trying deprecated subgraph...")
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        query = """
        query GetOpenInterest($startTime: Int!, $endTime: Int!, $skip: Int!) {
            openInterestStats(
                first: 1000, skip: $skip,
                where: { timestamp_gte: $startTime, timestamp_lte: $endTime },
                orderBy: timestamp, orderDirection: asc
            ) {
                id timestamp token longOI shortOI longOIUsd shortOIUsd
            }
        }
        """

        skip = 0

        for _ in range(self.MAX_PAGINATION_ITERATIONS):
            variables = {'startTime': start_ts, 'endTime': end_ts, 'skip': skip}

            data = await self._query_subgraph(chain, query, variables)
            oi_data = data.get('openInterestStats', [])

            if not oi_data:
                break

            for oi in oi_data:
                token = oi.get('token', '').upper()

                if not any(s.upper() in token for s in symbols):
                    continue

                oi_record = GMXOpenInterest(
                    timestamp=pd.to_datetime(int(oi['timestamp']), unit='s', utc=True),
                    symbol=token,
                    chain=chain,
                    long_oi=float(oi.get('longOI', 0)) / 1e30,
                    short_oi=float(oi.get('shortOI', 0)) / 1e30,
                    long_oi_usd=float(oi.get('longOIUsd', 0)) / 1e30,
                    short_oi_usd=float(oi.get('shortOIUsd', 0)) / 1e30
                )

                all_data.append({**oi_record.to_dict(), 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE})

            skip += 1000
            if len(oi_data) < 1000:
                break
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        self.collection_stats['records_collected'] += len(df)
        return df
    
    async def fetch_glp_stats(self, start_date: str, end_date: str, chain: str = 'arbitrum') -> pd.DataFrame:
        """Fetch GLP (GMX Liquidity Provider) pool statistics."""
        all_data = []
        
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        
        query = """
        query GetGLPStats($startTime: Int!, $endTime: Int!, $skip: Int!) {
            glpStats(
                first: 1000, skip: $skip,
                where: { timestamp_gte: $startTime, timestamp_lte: $endTime },
                orderBy: timestamp, orderDirection: asc
            ) {
                id timestamp aumInUsdg glpSupply
                distributedEth distributedEthCumulative
                distributedUsd distributedUsdCumulative
            }
        }
        """
        
        logger.info(f"Fetching GLP stats from {chain}")
        skip = 0

        for _ in range(self.MAX_PAGINATION_ITERATIONS):
            variables = {'startTime': start_ts, 'endTime': end_ts, 'skip': skip}

            data = await self._query_subgraph(chain, query, variables)
            stats = data.get('glpStats', [])

            if not stats:
                break

            for stat in stats:
                glp = GLPStats(
                    timestamp=pd.to_datetime(int(stat['timestamp']), unit='s', utc=True),
                    chain=chain,
                    aum_usd=float(stat.get('aumInUsdg', 0)) / 1e18,
                    glp_supply=float(stat.get('glpSupply', 0)) / 1e18,
                    distributed_eth=float(stat.get('distributedEth', 0)) / 1e18,
                    distributed_eth_cumulative=float(stat.get('distributedEthCumulative', 0)) / 1e18,
                    distributed_usd=float(stat.get('distributedUsd', 0)) / 1e30,
                    distributed_usd_cumulative=float(stat.get('distributedUsdCumulative', 0)) / 1e30
                )

                all_data.append({**glp.to_dict(), 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE})

            skip += 1000
            if len(stats) < 1000:
                break
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    async def fetch_liquidations(self, symbols: List[str], start_date: str, end_date: str, chain: str = 'arbitrum') -> pd.DataFrame:
        """Fetch liquidation events from GMX."""
        all_data = []
        
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        
        query = """
        query GetLiquidations($startTime: Int!, $endTime: Int!, $skip: Int!) {
            liquidatePositions(
                first: 1000, skip: $skip,
                where: { timestamp_gte: $startTime, timestamp_lte: $endTime },
                orderBy: timestamp, orderDirection: asc
            ) {
                id timestamp key account
                collateralToken indexToken isLong
                size collateral reserveAmount
                realisedPnl markPrice
            }
        }
        """
        
        logger.info(f"Fetching GMX liquidations from {chain}")
        skip = 0

        for _ in range(self.MAX_PAGINATION_ITERATIONS):
            variables = {'startTime': start_ts, 'endTime': end_ts, 'skip': skip}

            data = await self._query_subgraph(chain, query, variables)
            liquidations = data.get('liquidatePositions', [])

            if not liquidations:
                break

            for liq in liquidations:
                index_token = liq.get('indexToken', '').upper()

                if symbols and not any(s.upper() in index_token for s in symbols):
                    continue

                liq_record = GMXLiquidation(
                    timestamp=pd.to_datetime(int(liq['timestamp']), unit='s', utc=True),
                    chain=chain,
                    account=liq.get('account'),
                    collateral_token=liq.get('collateralToken'),
                    index_token=index_token,
                    is_long=liq.get('isLong'),
                    size_usd=float(liq.get('size', 0)) / 1e30,
                    collateral_usd=float(liq.get('collateral', 0)) / 1e30,
                    realized_pnl=float(liq.get('realisedPnl', 0)) / 1e30,
                    mark_price=float(liq.get('markPrice', 0)) / 1e30
                )

                all_data.append({**liq_record.to_dict(), 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE})

            skip += 1000
            if len(liquidations) < 1000:
                break
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    async def fetch_trading_volume(self, start_date: str, end_date: str, chain: str = 'arbitrum') -> pd.DataFrame:
        """Fetch daily trading volume and fees from GMX."""
        all_data = []
        
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        
        query = """
        query GetVolumeStats($startTime: Int!, $endTime: Int!, $skip: Int!) {
            volumeStats(
                first: 1000, skip: $skip,
                where: { timestamp_gte: $startTime, timestamp_lte: $endTime, period: "daily" },
                orderBy: timestamp, orderDirection: asc
            ) {
                id timestamp margin swap liquidation mint burn
            }
        }
        """
        
        logger.info(f"Fetching GMX volume stats from {chain}")
        skip = 0

        for _ in range(self.MAX_PAGINATION_ITERATIONS):
            variables = {'startTime': start_ts, 'endTime': end_ts, 'skip': skip}

            data = await self._query_subgraph(chain, query, variables)
            stats = data.get('volumeStats', [])

            if not stats:
                break

            for stat in stats:
                vol = GMXVolumeStats(
                    timestamp=pd.to_datetime(int(stat['timestamp']), unit='s', utc=True),
                    chain=chain,
                    margin_volume_usd=float(stat.get('margin', 0)) / 1e30,
                    swap_volume_usd=float(stat.get('swap', 0)) / 1e30,
                    liquidation_volume_usd=float(stat.get('liquidation', 0)) / 1e30,
                    mint_volume_usd=float(stat.get('mint', 0)) / 1e30,
                    burn_volume_usd=float(stat.get('burn', 0)) / 1e30
                )

                all_data.append({**vol.to_dict(), 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE})

            skip += 1000
            if len(stats) < 1000:
                break
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    async def fetch_comprehensive_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch comprehensive GMX data."""
        results = {}
        logger.info(f"Fetching comprehensive GMX data for {len(symbols)} symbols")
        
        results['funding_rates'] = await self.fetch_funding_rates(symbols, start_date, end_date)
        results['open_interest'] = await self.fetch_open_interest(symbols, start_date, end_date)
        results['glp_stats'] = await self.fetch_glp_stats(start_date, end_date)
        results['liquidations'] = await self.fetch_liquidations(symbols, start_date, end_date)
        results['volume'] = await self.fetch_trading_volume(start_date, end_date)
        
        return results

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
            logger.error(f"GMX collect_funding_rates error: {e}")
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
            logger.error(f"GMX collect_ohlcv error: {e}")
            return pd.DataFrame()

    async def collect_open_interest(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect open interest - wraps fetch_open_interest().

        Standardized method name for collection manager compatibility.
        """
        try:
            chain = kwargs.get('chain', 'arbitrum')

            # Convert dates to string format if needed
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            return await self.fetch_open_interest(
                symbols=symbols,
                start_date=start_str,
                end_date=end_str,
                chain=chain
            )
        except Exception as e:
            logger.error(f"GMX collect_open_interest error: {e}")
            return pd.DataFrame()

    async def collect_liquidations(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect liquidations - wraps fetch_liquidations().

        Standardized method name for collection manager compatibility.
        """
        try:
            chain = kwargs.get('chain', 'arbitrum')

            # Convert dates to string format if needed
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            return await self.fetch_liquidations(
                symbols=symbols,
                start_date=start_str,
                end_date=end_str,
                chain=chain
            )
        except Exception as e:
            logger.error(f"GMX collect_liquidations error: {e}")
            return pd.DataFrame()

    async def collect_pool_data(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect pool data (GLP stats) - wraps fetch_glp_stats().

        Standardized method name for collection manager compatibility.
        """
        try:
            chain = kwargs.get('chain', 'arbitrum')

            # Convert dates to string format if needed
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            return await self.fetch_glp_stats(
                start_date=start_str,
                end_date=end_str,
                chain=chain
            )
        except Exception as e:
            logger.error(f"GMX collect_pool_data error: {e}")
            return pd.DataFrame()

    async def collect_positions(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect positions data from GMX.

        Returns open interest and position data for each market.
        Note: GMX doesn't expose individual position data publicly,
        so this returns aggregate open interest per market.

        Args:
            symbols: List of asset symbols (BTC, ETH, etc.)
            start_date: Start date (ignored - returns current snapshot)
            end_date: End date (ignored - returns current snapshot)
            **kwargs: Additional arguments (chain='arbitrum')

        Returns:
            DataFrame with position/open interest data
        """
        try:
            chain = kwargs.get('chain', 'arbitrum')

            # Use collect_open_interest which gives aggregate position data
            oi_data = await self.collect_open_interest(symbols, start_date, end_date, **kwargs)

            if not oi_data.empty:
                # Rename columns to match positions format
                oi_data['position_type'] = 'aggregate'
                oi_data['long_open_interest'] = oi_data.get('long_open_interest', 0)
                oi_data['short_open_interest'] = oi_data.get('short_open_interest', 0)
                oi_data['total_positions'] = oi_data.get('open_interest', 0)
                oi_data['data_type'] = 'positions'
                return oi_data

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"GMX collect_positions error: {e}")
            return pd.DataFrame()

    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.collection_stats.copy()

async def test_gmx_collector():
    """Test GMX collector."""
    config = {'rate_limit': 20}
    
    async with GMXCollector(config) as collector:
        print("=" * 60)
        print("GMX Collector Test")
        print("=" * 60)
        print(f"\nSupported assets: {collector.GMX_ASSETS}")
        print(f"Stats: {collector.get_collection_stats()}")

if __name__ == '__main__':
    asyncio.run(test_gmx_collector())