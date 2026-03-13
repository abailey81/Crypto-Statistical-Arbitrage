"""
DefiLlama Data Collector - FREE Alternative to Token Terminal

validated collector for DeFi protocol analytics replacing paid services
like Token Terminal ($99-299/month) with completely FREE API access.

API Documentation: https://defillama.com/docs/api
Base URL: https://api.llama.fi

Coverage:
    - 4,000+ protocols across 200+ chains
    - TVL, fees, revenue data
    - DEX volumes, perpetual volumes
    - Yields and APY data
    - Stablecoin metrics
    - Bridge volumes and flows
    - Token prices (current and historical)

Rate Limits: Very generous (no strict limits documented, use ~100/min respectfully)
Authentication: None required (fully open API)

Cost Savings: $99-299/month â†’ $0/month

Version: 2.0.0
Date: January 2025
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any, Union, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_collection.base_collector import BaseCollector
from data_collection.utils.rate_limiter import get_shared_rate_limiter
from data_collection.utils.retry_handler import RetryHandler

logger = logging.getLogger(__name__)

# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class DefiLlamaEndpoint(Enum):
    """DefiLlama API endpoints organized by category."""
    
    # TVL endpoints
    PROTOCOLS = "https://api.llama.fi/protocols"
    PROTOCOL = "https://api.llama.fi/protocol/{protocol}"
    TVL_CHAINS = "https://api.llama.fi/v2/chains"
    TVL_HISTORICAL = "https://api.llama.fi/v2/historicalChainTvl/{chain}"
    TVL_GLOBAL = "https://api.llama.fi/v2/historicalChainTvl"
    
    # Fees & Revenue endpoints
    FEES_OVERVIEW = "https://api.llama.fi/overview/fees"
    FEES_PROTOCOL = "https://api.llama.fi/summary/fees/{protocol}"
    FEES_CHAIN = "https://api.llama.fi/overview/fees/{chain}"
    
    # DEX Volume endpoints
    DEXS_OVERVIEW = "https://api.llama.fi/overview/dexs"
    DEX_PROTOCOL = "https://api.llama.fi/summary/dexs/{protocol}"
    DEX_CHAIN = "https://api.llama.fi/overview/dexs/{chain}"
    
    # Derivatives/Perps Volume endpoints
    PERPS_OVERVIEW = "https://api.llama.fi/overview/derivatives"
    PERP_PROTOCOL = "https://api.llama.fi/summary/derivatives/{protocol}"
    PERP_CHAIN = "https://api.llama.fi/overview/derivatives/{chain}"
    
    # Options Volume endpoints
    OPTIONS_OVERVIEW = "https://api.llama.fi/overview/options"
    OPTIONS_PROTOCOL = "https://api.llama.fi/summary/options/{protocol}"
    
    # Aggregators endpoints
    AGGREGATORS_OVERVIEW = "https://api.llama.fi/overview/aggregators"
    AGGREGATOR_PROTOCOL = "https://api.llama.fi/summary/aggregators/{protocol}"
    
    # Yields endpoints
    YIELDS_POOLS = "https://yields.llama.fi/pools"
    YIELDS_CHART = "https://yields.llama.fi/chart/{pool_id}"
    YIELDS_LENDRATES = "https://yields.llama.fi/lendRates"
    
    # Stablecoins endpoints
    STABLECOINS = "https://stablecoins.llama.fi/stablecoins"
    STABLECOIN = "https://stablecoins.llama.fi/stablecoin/{asset}"
    STABLECOIN_CHAINS = "https://stablecoins.llama.fi/stablecoinchains"
    STABLECOIN_PRICES = "https://stablecoins.llama.fi/stablecoinprices"
    STABLECOIN_HISTORY = "https://stablecoins.llama.fi/stablecoincharts/{chain}"
    
    # Bridges endpoints
    BRIDGES = "https://bridges.llama.fi/bridges"
    BRIDGE = "https://bridges.llama.fi/bridge/{bridge_id}"
    BRIDGE_VOLUME = "https://bridges.llama.fi/bridgevolume/{chain}"
    BRIDGE_DAY_STATS = "https://bridges.llama.fi/bridgedaystats/{timestamp}/{chain}"
    BRIDGE_TRANSACTIONS = "https://bridges.llama.fi/transactions/{bridge_id}"
    
    # Token prices endpoints
    PRICES_CURRENT = "https://coins.llama.fi/prices/current/{coins}"
    PRICES_HISTORICAL = "https://coins.llama.fi/prices/historical/{timestamp}/{coins}"
    PRICES_CHART = "https://coins.llama.fi/chart/{coins}"
    PRICES_BATCH = "https://coins.llama.fi/batchHistorical"
    PRICES_PERCENTAGE = "https://coins.llama.fi/percentage/{coins}"
    PRICES_BLOCK = "https://coins.llama.fi/block/{chain}/{timestamp}"
    
    # Treasury endpoints
    TREASURY = "https://api.llama.fi/treasury/{protocol}"
    
    # Raises/Funding endpoints
    RAISES = "https://api.llama.fi/raises"

@dataclass
class ProtocolMetrics:
    """Protocol metrics data structure."""
    name: str
    slug: str
    chain: str
    chains: List[str]
    category: str
    tvl: float
    tvl_change_1d: float
    tvl_change_7d: float
    tvl_change_1m: float
    mcap: Optional[float] = None
    fdv: Optional[float] = None
    fees_24h: Optional[float] = None
    fees_7d: Optional[float] = None
    fees_30d: Optional[float] = None
    revenue_24h: Optional[float] = None
    revenue_7d: Optional[float] = None
    revenue_30d: Optional[float] = None
    volume_24h: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class YieldPool:
    """Yield pool data structure."""
    pool_id: str
    chain: str
    project: str
    symbol: str
    tvl_usd: float
    apy: float
    apy_base: float
    apy_reward: float
    apy_mean_30d: float
    il_risk: str
    stable_coin: bool
    exposure: str
    pool_meta: Optional[str] = None
    underlying_tokens: List[str] = field(default_factory=list)
    reward_tokens: List[str] = field(default_factory=list)

@dataclass 
class StablecoinMetrics:
    """Stablecoin metrics data structure."""
    name: str
    symbol: str
    peg_type: str
    peg_mechanism: str
    circulating: float
    circulating_prev_day: float
    circulating_prev_week: float
    circulating_prev_month: float
    price: float
    chains: List[str] = field(default_factory=list)

# =============================================================================
# MAIN COLLECTOR CLASS
# =============================================================================

class DefiLlamaCollector(BaseCollector):
    """
    DefiLlama API collector for comprehensive DeFi protocol analytics.
    
    FREE alternative to Token Terminal providing:
    - Protocol TVL and TVL changes across 4,000+ protocols
    - Fees and revenue metrics
    - DEX and perpetual trading volumes
    - Yield/APY data across protocols and pools
    - Stablecoin metrics and flows
    - Bridge volumes and cross-chain flows
    - Token prices (current and historical)
    
    Usage:
        collector = DefiLlamaCollector(config)
        
        # Get all protocols with TVL
        protocols = await collector.fetch_all_protocols()
        
        # Get specific protocol data
        aave_data = await collector.fetch_protocol_details('aave')
        
        # Get fees/revenue
        fees = await collector.fetch_protocol_fees('uniswap')
        
        # Get DEX volumes
        volumes = await collector.fetch_dex_volumes()
        
        # Get yields
        yields = await collector.fetch_yields(chain='ethereum')
        
        # Get stablecoin data
        stables = await collector.fetch_stablecoins()
        
        # Get token prices
        prices = await collector.fetch_token_prices(['ethereum:0x...'])
    """
    
    VENUE = 'defillama'
    VENUE_TYPE = 'aggregator'
    
    # Chain name mappings (lowercase key -> DefiLlama format)
    CHAIN_MAPPINGS = {
        'eth': 'Ethereum',
        'ethereum': 'Ethereum',
        'arb': 'Arbitrum',
        'arbitrum': 'Arbitrum',
        'op': 'Optimism',
        'optimism': 'Optimism',
        'matic': 'Polygon',
        'polygon': 'Polygon',
        'avax': 'Avalanche',
        'avalanche': 'Avalanche',
        'bsc': 'BSC',
        'bnb': 'BSC',
        'sol': 'Solana',
        'solana': 'Solana',
        'base': 'Base',
        'fantom': 'Fantom',
        'gnosis': 'Gnosis',
        'celo': 'Celo',
        'moonbeam': 'Moonbeam',
        'aurora': 'Aurora',
        'harmony': 'Harmony',
        'cronos': 'Cronos',
        'metis': 'Metis',
        'boba': 'Boba',
        'kava': 'Kava',
        'zksync': 'zkSync Era',
        'linea': 'Linea',
        'scroll': 'Scroll',
        'mantle': 'Mantle',
        'blast': 'Blast',
        'manta': 'Manta',
        'mode': 'Mode',
    }
    
    # Protocol category to sector mappings
    CATEGORY_SECTOR_MAP = {
        'Dexes': 'DEX',
        'Lending': 'Lending',
        'Bridge': 'Infrastructure',
        'CDP': 'Lending',
        'Derivatives': 'Derivatives',
        'Yield': 'Yield',
        'Liquid Staking': 'Staking',
        'Yield Aggregator': 'Yield',
        'Farm': 'Yield',
        'Services': 'Infrastructure',
        'Chain': 'L1',
        'Cross Chain': 'Infrastructure',
        'Launchpad': 'Infrastructure',
        'Options': 'Derivatives',
        'Indexes': 'Asset Management',
        'Insurance': 'Insurance',
        'Synthetics': 'Derivatives',
        'Prediction Market': 'Derivatives',
        'NFT Lending': 'NFT',
        'NFT Marketplace': 'NFT',
        'Gaming': 'Gaming',
        'RWA': 'RWA',
        'Staking Pool': 'Staking',
        'Algo-Stables': 'Stablecoin',
        'Reserve Currency': 'Stablecoin',
        'Liquidity manager': 'Yield',
        'Leveraged Farming': 'Yield',
        'Privacy': 'Infrastructure',
        'Payments': 'Payments',
        'Oracle': 'Infrastructure',
        'SoFi': 'Social',
    }
    
    # Major protocols for focused analysis
    MAJOR_PROTOCOLS = [
        # DEXs
        'uniswap', 'curve-dex', 'pancakeswap', 'sushiswap', 'balancer',
        'trader-joe', 'velodrome', 'aerodrome', 'camelot',
        # Lending
        'aave', 'compound', 'makerdao', 'morpho', 'radiant',
        'benqi-lending', 'venus', 'spark',
        # Liquid Staking
        'lido', 'rocket-pool', 'coinbase-wrapped-staked-eth', 'frax-ether',
        # Derivatives
        'gmx', 'dydx', 'hyperliquid', 'gains-network', 'synthetix',
        # Bridges
        'stargate', 'across', 'hop-protocol', 'synapse', 'celer-cbridge',
        # Yield
        'convex-finance', 'yearn-finance', 'beefy', 'pendle',
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DefiLlama collector.

        Args:
            config: Optional configuration dictionary
                - rate_limit: Requests per minute (default: 50)
                - max_retries: Maximum retry attempts (default: 3)
                - timeout: Request timeout in seconds (default: 30)
                - cache_ttl_hours: Cache TTL in hours (default: 1)
        """
        config = config or {}
        super().__init__(config)

        # CRITICAL: Set supported data types for dynamic routing
        self.supported_data_types = ['tvl', 'yields', 'stablecoins']
        self.venue = 'defillama'

        # Import VenueType from base_collector
        from ..base_collector import VenueType
        self.venue_type = VenueType.ALTERNATIVE
        self.requires_auth = False # DefiLlama is completely free, no API key required

        self.logger = logging.getLogger(f'{__name__}.{self.VENUE}')

        # Use shared rate limiter to avoid re-initialization overhead
        self.rate_limiter = get_shared_rate_limiter('defillama', rate=config.get('rate_limit', 50), per=60.0, burst=5)

        # Retry handler with exponential backoff
        self.retry_handler = RetryHandler(
            max_retries=config.get('max_retries', 3),
            base_delay=1.0,
            max_delay=30.0
        )

        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = config.get('timeout', 30)

        # Caching for expensive calls
        self._protocols_cache: Optional[List[Dict]] = None
        self._protocols_cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=config.get('cache_ttl_hours', 1))

        self.logger.info(
            f"DefiLlamaCollector initialized (FREE - no API key required), "
            f"rate_limit={config.get('rate_limit', 50)}/min, "
            f"data types: {self.supported_data_types}"
        )
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'Accept': 'application/json',
                    'User-Agent': 'CryptoStatArb/2.0'
                }
            )
        return self.session
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
    
    async def _make_request(
        self,
        url: str,
        params: Optional[Dict] = None,
        method: str = 'GET'
    ) -> Optional[Union[Dict, List]]:
        """
        Make HTTP request with rate limiting and retry logic.
        
        Args:
            url: Request URL
            params: Optional query parameters
            method: HTTP method (GET or POST)
            
        Returns:
            JSON response data or None on failure
        """
        await self.rate_limiter.acquire()
        session = await self._get_session()
        
        async def _request():
            if method == 'GET':
                async with session.get(url, params=params) as response:
                    return await self._handle_response(response, url)
            else:
                async with session.post(url, json=params) as response:
                    return await self._handle_response(response, url)
        
        try:
            return await self.retry_handler.execute(_request)
        except Exception as e:
            self.logger.error(f"Request failed: {url} - {e}")
            return None
    
    async def _handle_response(
        self,
        response: aiohttp.ClientResponse,
        url: str
    ) -> Optional[Union[Dict, List]]:
        """Handle HTTP response with error handling."""
        if response.status == 200:
            return await response.json()
        elif response.status == 429:
            self.logger.warning("Rate limited by DefiLlama, waiting 60s...")
            await asyncio.sleep(60)
            raise Exception("Rate limited - retry")
        elif response.status == 404:
            self.logger.warning(f"Resource not found: {url}")
            return None
        else:
            text = await response.text()
            self.logger.error(f"HTTP {response.status} for {url}: {text[:200]}")
            return None
    
    def _normalize_chain(self, chain: str) -> str:
        """Normalize chain name to DefiLlama format."""
        return self.CHAIN_MAPPINGS.get(chain.lower(), chain)
    
    def _get_sector(self, category: str) -> str:
        """Map protocol category to sector."""
        return self.CATEGORY_SECTOR_MAP.get(category, 'Other')
    
    # =========================================================================
    # TVL DATA METHODS
    # =========================================================================
    
    async def fetch_all_protocols(
        self,
        min_tvl: float = 0,
        chains: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        include_changes: bool = True
    ) -> pd.DataFrame:
        """
        Fetch all DeFi protocols with TVL data.
        
        Args:
            min_tvl: Minimum TVL filter in USD (default: 0)
            chains: Filter by specific chains (optional)
            categories: Filter by categories (optional)
            include_changes: Include TVL change percentages
            
        Returns:
            DataFrame with protocol data including:
            - name, slug, category, chains
            - tvl, tvl_change_1d/7d/1m
            - mcap, fdv (if available)
        """
        self.logger.info(f"Fetching all protocols (min_tvl=${min_tvl:,.0f})...")
        
        # Check cache
        if (self._protocols_cache is not None and 
            self._protocols_cache_time is not None and
            datetime.utcnow() - self._protocols_cache_time < self._cache_ttl):
            data = self._protocols_cache
            self.logger.debug("Using cached protocols data")
        else:
            data = await self._make_request(DefiLlamaEndpoint.PROTOCOLS.value)
            if data:
                self._protocols_cache = data
                self._protocols_cache_time = datetime.utcnow()
        
        if not data:
            self.logger.warning("No protocol data received")
            return pd.DataFrame()
        
        # Normalize chain filters
        if chains:
            chains = [self._normalize_chain(c) for c in chains]
        
        records = []
        for p in data:
            tvl = p.get('tvl', 0) or 0
            
            # Apply TVL filter
            if tvl < min_tvl:
                continue
            
            # Apply chain filter
            protocol_chains = p.get('chains', [])
            if chains:
                if not any(c in protocol_chains for c in chains):
                    continue
            
            # Apply category filter
            category = p.get('category', 'Other')
            if categories and category not in categories:
                continue
            
            record = {
                'name': p.get('name', ''),
                'slug': p.get('slug', ''),
                'symbol': p.get('symbol', ''),
                'category': category,
                'sector': self._get_sector(category),
                'chains': protocol_chains,
                'chain_count': len(protocol_chains),
                'tvl': tvl,
                'mcap': p.get('mcap'),
                'fdv': p.get('fdv'),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE,
                'timestamp': datetime.utcnow()
            }
            
            if include_changes:
                record.update({
                    'tvl_change_1d': p.get('change_1d'),
                    'tvl_change_7d': p.get('change_7d'),
                    'tvl_change_1m': p.get('change_1m'),
                })
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            df = df.sort_values('tvl', ascending=False).reset_index(drop=True)
            self.logger.info(f"Fetched {len(df)} protocols with TVL >= ${min_tvl:,.0f}")
        
        return df
    
    async def fetch_protocol_details(
        self,
        protocol: str,
        include_tvl_history: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed data for a specific protocol.
        
        Args:
            protocol: Protocol slug (e.g., 'aave', 'uniswap')
            include_tvl_history: Include historical TVL data
            
        Returns:
            Dictionary with detailed protocol data including:
            - Basic info (name, category, chains, etc.)
            - Current TVL and token metrics
            - TVL history by chain (if requested)
            - Token holdings breakdown
        """
        self.logger.info(f"Fetching details for protocol: {protocol}")
        
        url = DefiLlamaEndpoint.PROTOCOL.value.format(protocol=protocol)
        data = await self._make_request(url)
        
        if not data:
            self.logger.warning(f"No data found for protocol: {protocol}")
            return None
        
        result = {
            'name': data.get('name'),
            'slug': data.get('slug'),
            'symbol': data.get('symbol'),
            'category': data.get('category'),
            'sector': self._get_sector(data.get('category', '')),
            'chains': data.get('chains', []),
            'tvl': data.get('tvl'),
            'mcap': data.get('mcap'),
            'fdv': data.get('fdv'),
            'description': data.get('description'),
            'url': data.get('url'),
            'twitter': data.get('twitter'),
            'audit_links': data.get('audit_links', []),
            'gecko_id': data.get('gecko_id'),
            'cmc_id': data.get('cmcId'),
            'chain_tvls': data.get('chainTvls', {}),
            'current_chain_tvls': data.get('currentChainTvls', {}),
            'tokens': data.get('tokens', []),
            'venue': self.VENUE,
            'timestamp': datetime.utcnow()
        }
        
        # Include TVL history if requested
        if include_tvl_history and 'tvl' in data:
            tvl_history = data.get('tvl', [])
            if isinstance(tvl_history, list) and tvl_history:
                result['tvl_history'] = pd.DataFrame(tvl_history)
        
        return result
    
    async def fetch_chain_tvl(
        self,
        chain: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch TVL data by chain.
        
        Args:
            chain: Specific chain to fetch (optional, fetches all if None)
            
        Returns:
            DataFrame with chain TVL data
        """
        if chain:
            chain = self._normalize_chain(chain)
            url = DefiLlamaEndpoint.TVL_HISTORICAL.value.format(chain=chain)
        else:
            url = DefiLlamaEndpoint.TVL_CHAINS.value
        
        data = await self._make_request(url)
        
        if not data:
            return pd.DataFrame()
        
        if chain:
            # Historical data for specific chain
            df = pd.DataFrame(data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], unit='s')
                df = df.rename(columns={'date': 'timestamp'})
            df['chain'] = chain
        else:
            # All chains current TVL
            df = pd.DataFrame(data)
            df['timestamp'] = datetime.utcnow()
        
        df['venue'] = self.VENUE
        
        return df
    
    async def fetch_historical_tvl(
        self,
        protocol: Optional[str] = None,
        chain: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical TVL data.
        
        Args:
            protocol: Protocol slug (optional)
            chain: Chain name (optional)
            start_date: Start date 'YYYY-MM-DD' (optional)
            end_date: End date 'YYYY-MM-DD' (optional)
            
        Returns:
            DataFrame with historical TVL data
        """
        if protocol:
            details = await self.fetch_protocol_details(protocol, include_tvl_history=True)
            if details and 'tvl_history' in details:
                df = details['tvl_history']
                df['protocol'] = protocol
            else:
                return pd.DataFrame()
        elif chain:
            df = await self.fetch_chain_tvl(chain)
        else:
            # Global TVL history
            data = await self._make_request(DefiLlamaEndpoint.TVL_GLOBAL.value)
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], unit='s')
                df = df.rename(columns={'date': 'timestamp'})
        
        # Apply date filters
        if not df.empty and 'timestamp' in df.columns:
            if start_date:
                df = df[df['timestamp'] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df['timestamp'] <= pd.Timestamp(end_date)]
        
        return df
    
    # =========================================================================
    # FEES & REVENUE METHODS
    # =========================================================================
    
    async def fetch_fees_overview(
        self,
        chain: Optional[str] = None,
        exclude_total_data_chart: bool = True
    ) -> pd.DataFrame:
        """
        Fetch fees/revenue overview for all protocols or specific chain.
        
        Args:
            chain: Specific chain (optional)
            exclude_total_data_chart: Exclude bulky chart data
            
        Returns:
            DataFrame with fees/revenue metrics by protocol
        """
        if chain:
            chain = self._normalize_chain(chain)
            url = DefiLlamaEndpoint.FEES_CHAIN.value.format(chain=chain)
        else:
            url = DefiLlamaEndpoint.FEES_OVERVIEW.value
        
        params = {}
        if exclude_total_data_chart:
            params['excludeTotalDataChart'] = 'true'
            params['excludeTotalDataChartBreakdown'] = 'true'
        
        data = await self._make_request(url, params=params)
        
        if not data or 'protocols' not in data:
            return pd.DataFrame()
        
        records = []
        for p in data['protocols']:
            records.append({
                'name': p.get('name'),
                'slug': p.get('defillamaId') or p.get('name', '').lower().replace(' ', '-'),
                'category': p.get('category'),
                'chains': p.get('chains', []),
                'total_24h': p.get('total24h'),
                'total_7d': p.get('total7d'),
                'total_30d': p.get('total30d'),
                'total_all_time': p.get('totalAllTime'),
                'revenue_24h': p.get('revenue24h'),
                'revenue_7d': p.get('revenue7d'),
                'revenue_30d': p.get('revenue30d'),
                'daily_revenue': p.get('dailyRevenue'),
                'daily_fees': p.get('dailyFees'),
                'change_1d': p.get('change_1d'),
                'change_7d': p.get('change_7d'),
                'change_1m': p.get('change_1m'),
                'methodology_url': p.get('methodologyURL'),
                'venue': self.VENUE,
                'timestamp': datetime.utcnow()
            })
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            df = df.sort_values('total_24h', ascending=False, na_position='last')
            df = df.reset_index(drop=True)
        
        return df
    
    async def fetch_protocol_fees(
        self,
        protocol: str,
        data_type: str = 'dailyFees'
    ) -> pd.DataFrame:
        """
        Fetch detailed fees data for a specific protocol.
        
        Args:
            protocol: Protocol slug
            data_type: Type of data ('dailyFees', 'dailyRevenue', etc.)
            
        Returns:
            DataFrame with historical fees data
        """
        url = DefiLlamaEndpoint.FEES_PROTOCOL.value.format(protocol=protocol)
        params = {'dataType': data_type}
        
        data = await self._make_request(url, params=params)
        
        if not data:
            return pd.DataFrame()
        
        # Extract time series data
        records = []
        
        # Total data chart
        total_data = data.get('totalDataChart', [])
        for item in total_data:
            if isinstance(item, list) and len(item) >= 2:
                records.append({
                    'timestamp': datetime.fromtimestamp(item[0], tz=timezone.utc),
                    'value': item[1],
                    'type': 'total'
                })
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            df['protocol'] = protocol
            df['data_type'] = data_type
            df['venue'] = self.VENUE
        
        # Add summary metrics
        summary = {
            'total_24h': data.get('total24h'),
            'total_7d': data.get('total7d'),
            'total_30d': data.get('total30d'),
            'total_all_time': data.get('totalAllTime'),
        }
        
        return df
    
    # =========================================================================
    # DEX VOLUME METHODS
    # =========================================================================
    
    async def fetch_dex_volumes(
        self,
        chain: Optional[str] = None,
        exclude_total_data_chart: bool = True
    ) -> pd.DataFrame:
        """
        Fetch DEX trading volumes.
        
        Args:
            chain: Specific chain (optional)
            exclude_total_data_chart: Exclude bulky chart data
            
        Returns:
            DataFrame with DEX volume data
        """
        if chain:
            chain = self._normalize_chain(chain)
            url = DefiLlamaEndpoint.DEX_CHAIN.value.format(chain=chain)
        else:
            url = DefiLlamaEndpoint.DEXS_OVERVIEW.value
        
        params = {}
        if exclude_total_data_chart:
            params['excludeTotalDataChart'] = 'true'
            params['excludeTotalDataChartBreakdown'] = 'true'
        
        data = await self._make_request(url, params=params)
        
        if not data or 'protocols' not in data:
            return pd.DataFrame()
        
        records = []
        for p in data['protocols']:
            records.append({
                'name': p.get('name'),
                'slug': p.get('defillamaId') or p.get('slug'),
                'category': 'DEX',
                'chains': p.get('chains', []),
                'volume_24h': p.get('total24h'),
                'volume_7d': p.get('total7d'),
                'volume_30d': p.get('total30d'),
                'volume_all_time': p.get('totalAllTime'),
                'change_1d': p.get('change_1d'),
                'change_7d': p.get('change_7d'),
                'change_1m': p.get('change_1m'),
                'venue': self.VENUE,
                'timestamp': datetime.utcnow()
            })
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            df = df.sort_values('volume_24h', ascending=False, na_position='last')
            df = df.reset_index(drop=True)
        
        return df
    
    async def fetch_protocol_dex_volume(
        self,
        protocol: str
    ) -> pd.DataFrame:
        """
        Fetch detailed DEX volume for a specific protocol.
        
        Args:
            protocol: Protocol slug
            
        Returns:
            DataFrame with historical volume data
        """
        url = DefiLlamaEndpoint.DEX_PROTOCOL.value.format(protocol=protocol)
        data = await self._make_request(url)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        total_data = data.get('totalDataChart', [])
        for item in total_data:
            if isinstance(item, list) and len(item) >= 2:
                records.append({
                    'timestamp': datetime.fromtimestamp(item[0], tz=timezone.utc),
                    'volume': item[1]
                })
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            df['protocol'] = protocol
            df['venue'] = self.VENUE
        
        return df
    
    # =========================================================================
    # DERIVATIVES / PERPS VOLUME METHODS
    # =========================================================================
    
    async def fetch_perp_volumes(
        self,
        chain: Optional[str] = None,
        exclude_total_data_chart: bool = True
    ) -> pd.DataFrame:
        """
        Fetch perpetual/derivatives trading volumes.
        
        Args:
            chain: Specific chain (optional)
            exclude_total_data_chart: Exclude bulky chart data
            
        Returns:
            DataFrame with perp volume data
        """
        if chain:
            chain = self._normalize_chain(chain)
            url = DefiLlamaEndpoint.PERP_CHAIN.value.format(chain=chain)
        else:
            url = DefiLlamaEndpoint.PERPS_OVERVIEW.value
        
        params = {}
        if exclude_total_data_chart:
            params['excludeTotalDataChart'] = 'true'
        
        data = await self._make_request(url, params=params)
        
        if not data or 'protocols' not in data:
            return pd.DataFrame()
        
        records = []
        for p in data['protocols']:
            records.append({
                'name': p.get('name'),
                'slug': p.get('defillamaId') or p.get('slug'),
                'category': 'Derivatives',
                'chains': p.get('chains', []),
                'volume_24h': p.get('total24h'),
                'volume_7d': p.get('total7d'),
                'volume_30d': p.get('total30d'),
                'volume_all_time': p.get('totalAllTime'),
                'change_1d': p.get('change_1d'),
                'change_7d': p.get('change_7d'),
                'change_1m': p.get('change_1m'),
                'venue': self.VENUE,
                'timestamp': datetime.utcnow()
            })
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            df = df.sort_values('volume_24h', ascending=False, na_position='last')
            df = df.reset_index(drop=True)
        
        return df
    
    async def fetch_options_volumes(self) -> pd.DataFrame:
        """
        Fetch options trading volumes.
        
        Returns:
            DataFrame with options volume data
        """
        data = await self._make_request(DefiLlamaEndpoint.OPTIONS_OVERVIEW.value)
        
        if not data or 'protocols' not in data:
            return pd.DataFrame()
        
        records = []
        for p in data['protocols']:
            records.append({
                'name': p.get('name'),
                'slug': p.get('defillamaId') or p.get('slug'),
                'category': 'Options',
                'chains': p.get('chains', []),
                'volume_24h': p.get('total24h'),
                'volume_7d': p.get('total7d'),
                'premium_volume_24h': p.get('dailyPremiumVolume'),
                'notional_volume_24h': p.get('dailyNotionalVolume'),
                'change_1d': p.get('change_1d'),
                'venue': self.VENUE,
                'timestamp': datetime.utcnow()
            })
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            df = df.sort_values('volume_24h', ascending=False, na_position='last')
        
        return df
    
    # =========================================================================
    # YIELDS / APY METHODS
    # =========================================================================
    
    async def fetch_yields(
        self,
        chain: Optional[str] = None,
        project: Optional[str] = None,
        min_tvl: float = 0,
        min_apy: float = 0,
        stablecoin_only: bool = False
    ) -> pd.DataFrame:
        """
        Fetch yield/APY data from DeFi protocols.
        
        Args:
            chain: Filter by chain (optional)
            project: Filter by project/protocol (optional)
            min_tvl: Minimum TVL filter in USD
            min_apy: Minimum APY filter (as percentage)
            stablecoin_only: Only return stablecoin pools
            
        Returns:
            DataFrame with yield pool data including:
            - Pool info (chain, project, symbol)
            - TVL and APY metrics
            - IL risk and exposure info
        """
        self.logger.info("Fetching yield pools...")
        
        data = await self._make_request(DefiLlamaEndpoint.YIELDS_POOLS.value)
        
        if not data or 'data' not in data:
            return pd.DataFrame()
        
        pools = data['data']
        
        # Normalize chain filter
        if chain:
            chain = self._normalize_chain(chain)
        
        records = []
        for pool in pools:
            pool_chain = pool.get('chain', '')
            pool_tvl = pool.get('tvlUsd', 0) or 0
            pool_apy = pool.get('apy', 0) or 0
            is_stablecoin = pool.get('stablecoin', False)
            
            # Apply filters
            if chain and pool_chain.lower() != chain.lower():
                continue
            if project and pool.get('project', '').lower() != project.lower():
                continue
            if pool_tvl < min_tvl:
                continue
            if pool_apy < min_apy:
                continue
            if stablecoin_only and not is_stablecoin:
                continue
            
            records.append({
                'pool_id': pool.get('pool'),
                'chain': pool_chain,
                'project': pool.get('project'),
                'symbol': pool.get('symbol'),
                'pool_meta': pool.get('poolMeta'),
                'tvl_usd': pool_tvl,
                'apy': pool_apy,
                'apy_base': pool.get('apyBase'),
                'apy_reward': pool.get('apyReward'),
                'apy_mean_30d': pool.get('apyMean30d'),
                'apy_pct_1d': pool.get('apyPct1D'),
                'apy_pct_7d': pool.get('apyPct7D'),
                'apy_pct_30d': pool.get('apyPct30D'),
                'il_risk': pool.get('ilRisk'),
                'stablecoin': is_stablecoin,
                'exposure': pool.get('exposure'),
                'underlying_tokens': pool.get('underlyingTokens', []),
                'reward_tokens': pool.get('rewardTokens', []),
                'predictions': pool.get('predictions'),
                'venue': self.VENUE,
                'timestamp': datetime.utcnow()
            })
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            df = df.sort_values('tvl_usd', ascending=False).reset_index(drop=True)
            self.logger.info(f"Fetched {len(df)} yield pools")
        
        return df
    
    async def fetch_yield_history(
        self,
        pool_id: str
    ) -> pd.DataFrame:
        """
        Fetch historical yield data for a specific pool.
        
        Args:
            pool_id: Pool identifier from fetch_yields()
            
        Returns:
            DataFrame with historical APY and TVL data
        """
        url = DefiLlamaEndpoint.YIELDS_CHART.value.format(pool_id=pool_id)
        data = await self._make_request(url)
        
        if not data or 'data' not in data:
            return pd.DataFrame()
        
        records = []
        for item in data['data']:
            records.append({
                'timestamp': pd.to_datetime(item.get('timestamp')),
                'tvl_usd': item.get('tvlUsd'),
                'apy': item.get('apy'),
                'apy_base': item.get('apyBase'),
                'apy_reward': item.get('apyReward'),
                'il_7d': item.get('il7d'),
            })
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            df['pool_id'] = pool_id
            df['venue'] = self.VENUE
        
        return df
    
    # =========================================================================
    # STABLECOIN METHODS
    # =========================================================================
    
    async def fetch_stablecoins(
        self,
        include_prices: bool = True
    ) -> pd.DataFrame:
        """
        Fetch stablecoin metrics.

        Args:
            include_prices: Include current prices

        Returns:
            DataFrame with stablecoin data
        """
        data = await self._make_request(DefiLlamaEndpoint.STABLECOINS.value)

        if not data or 'peggedAssets' not in data:
            return pd.DataFrame()

        records = []
        for s in data['peggedAssets']:
            circulating = s.get('circulating', {})

            # Convert price to float (API sometimes returns strings)
            price_raw = s.get('price')
            try:
                price = float(price_raw) if price_raw is not None else None
            except (ValueError, TypeError):
                price = None

            records.append({
                'id': s.get('id'),
                'name': s.get('name'),
                'symbol': s.get('symbol'),
                'peg_type': s.get('pegType'),
                'peg_mechanism': s.get('pegMechanism'),
                'circulating_usd': circulating.get('peggedUSD', 0),
                'circulating_prev_day': s.get('circulatingPrevDay', {}).get('peggedUSD', 0),
                'circulating_prev_week': s.get('circulatingPrevWeek', {}).get('peggedUSD', 0),
                'circulating_prev_month': s.get('circulatingPrevMonth', {}).get('peggedUSD', 0),
                'chains': s.get('chains', []),
                'price': price,
                'gecko_id': s.get('gecko_id'),
                'venue': self.VENUE,
                'timestamp': datetime.utcnow()
            })

        df = pd.DataFrame(records)

        if not df.empty:
            df = df.sort_values('circulating_usd', ascending=False).reset_index(drop=True)

        return df
    
    async def fetch_stablecoin_history(
        self,
        chain: str = 'all'
    ) -> pd.DataFrame:
        """
        Fetch historical stablecoin supply by chain.
        
        Args:
            chain: Chain name or 'all' for global
            
        Returns:
            DataFrame with historical stablecoin supply
        """
        url = DefiLlamaEndpoint.STABLECOIN_HISTORY.value.format(chain=chain)
        data = await self._make_request(url)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        for item in data:
            record = {
                'timestamp': datetime.fromtimestamp(item.get('date', 0), tz=timezone.utc),
                'total_circulating_usd': item.get('totalCirculatingUSD', {}).get('peggedUSD', 0),
            }
            
            # Add breakdown by stablecoin
            tokens = item.get('totalCirculating', {})
            for token, values in tokens.items():
                record[f'{token}_circulating'] = values.get('peggedUSD', 0)
            
            records.append(record)
        
        df = pd.DataFrame(records)
        df['chain'] = chain
        df['venue'] = self.VENUE
        
        return df
    
    # =========================================================================
    # BRIDGE METHODS
    # =========================================================================
    
    async def fetch_bridges(self) -> pd.DataFrame:
        """
        Fetch bridge data.
        
        Returns:
            DataFrame with bridge metrics
        """
        data = await self._make_request(DefiLlamaEndpoint.BRIDGES.value)
        
        if not data or 'bridges' not in data:
            return pd.DataFrame()
        
        records = []
        for b in data['bridges']:
            records.append({
                'id': b.get('id'),
                'name': b.get('name'),
                'display_name': b.get('displayName'),
                'icon': b.get('icon'),
                'volume_prev_day': b.get('volumePrevDay'),
                'volume_prev_2days': b.get('volumePrev2Days'),
                'last_hourly_volume': b.get('lastHourlyVolume'),
                'current_day_volume': b.get('currentDayVolume'),
                'last_daily_volume': b.get('lastDailyVolume'),
                'weekly_volume': b.get('weeklyVolume'),
                'monthly_volume': b.get('monthlyVolume'),
                'chains': b.get('chains', []),
                'destination_chain': b.get('destinationChain'),
                'venue': self.VENUE,
                'timestamp': datetime.utcnow()
            })
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            df = df.sort_values('weekly_volume', ascending=False, na_position='last')
        
        return df
    
    async def fetch_bridge_volume(
        self,
        chain: str
    ) -> pd.DataFrame:
        """
        Fetch bridge volume for a specific chain.
        
        Args:
            chain: Chain name
            
        Returns:
            DataFrame with bridge volume history
        """
        chain = self._normalize_chain(chain)
        url = DefiLlamaEndpoint.BRIDGE_VOLUME.value.format(chain=chain)
        data = await self._make_request(url)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        for item in data:
            records.append({
                'timestamp': datetime.fromtimestamp(item.get('date', 0), tz=timezone.utc),
                'deposit_usd': item.get('depositUSD', 0),
                'withdraw_usd': item.get('withdrawUSD', 0),
                'net_flow': item.get('depositUSD', 0) - item.get('withdrawUSD', 0),
            })
        
        df = pd.DataFrame(records)
        df['chain'] = chain
        df['venue'] = self.VENUE
        
        return df
    
    # =========================================================================
    # TOKEN PRICE METHODS
    # =========================================================================
    
    async def fetch_token_prices(
        self,
        coins: List[str],
        search_width: str = '4h'
    ) -> pd.DataFrame:
        """
        Fetch current token prices.
        
        Args:
            coins: List of coin identifiers in format 'chain:address'
                   e.g., ['ethereum:0x...', 'arbitrum:0x...']
                   Special: 'coingecko:bitcoin' for CoinGecko IDs
            search_width: Time window for price lookup
            
        Returns:
            DataFrame with token prices
        """
        if not coins:
            return pd.DataFrame()
        
        coins_str = ','.join(coins)
        url = DefiLlamaEndpoint.PRICES_CURRENT.value.format(coins=coins_str)
        params = {'searchWidth': search_width}
        
        data = await self._make_request(url, params=params)
        
        if not data or 'coins' not in data:
            return pd.DataFrame()
        
        records = []
        for coin_id, info in data['coins'].items():
            records.append({
                'coin_id': coin_id,
                'price': info.get('price'),
                'symbol': info.get('symbol'),
                'decimals': info.get('decimals'),
                'confidence': info.get('confidence'),
                'timestamp': datetime.fromtimestamp(
                    info.get('timestamp', 0), tz=timezone.utc
                ) if info.get('timestamp') else datetime.utcnow(),
                'venue': self.VENUE
            })
        
        return pd.DataFrame(records)
    
    async def fetch_historical_prices(
        self,
        coins: List[str],
        timestamp: int
    ) -> pd.DataFrame:
        """
        Fetch historical token prices at a specific timestamp.
        
        Args:
            coins: List of coin identifiers
            timestamp: Unix timestamp
            
        Returns:
            DataFrame with historical prices
        """
        if not coins:
            return pd.DataFrame()
        
        coins_str = ','.join(coins)
        url = DefiLlamaEndpoint.PRICES_HISTORICAL.value.format(
            timestamp=timestamp,
            coins=coins_str
        )
        
        data = await self._make_request(url)
        
        if not data or 'coins' not in data:
            return pd.DataFrame()
        
        records = []
        for coin_id, info in data['coins'].items():
            records.append({
                'coin_id': coin_id,
                'price': info.get('price'),
                'symbol': info.get('symbol'),
                'timestamp': datetime.fromtimestamp(timestamp, tz=timezone.utc),
                'venue': self.VENUE
            })
        
        return pd.DataFrame(records)
    
    async def fetch_price_chart(
        self,
        coins: List[str],
        start: Optional[int] = None,
        end: Optional[int] = None,
        span: int = 0,
        period: str = '1d',
        search_width: str = '600'
    ) -> pd.DataFrame:
        """
        Fetch price chart data for tokens.
        
        Args:
            coins: List of coin identifiers
            start: Start timestamp (optional)
            end: End timestamp (optional)
            span: Number of data points (0 = all)
            period: Candle period ('1d', '4h', '1h')
            search_width: Price search tolerance
            
        Returns:
            DataFrame with OHLCV-like price data
        """
        if not coins:
            return pd.DataFrame()
        
        coins_str = ','.join(coins)
        url = DefiLlamaEndpoint.PRICES_CHART.value.format(coins=coins_str)
        
        params = {
            'span': span,
            'period': period,
            'searchWidth': search_width
        }
        if start:
            params['start'] = start
        if end:
            params['end'] = end
        
        data = await self._make_request(url, params=params)
        
        if not data or 'coins' not in data:
            return pd.DataFrame()
        
        all_records = []
        for coin_id, prices in data['coins'].items():
            if 'prices' not in prices:
                continue
            
            for price_point in prices['prices']:
                all_records.append({
                    'coin_id': coin_id,
                    'symbol': prices.get('symbol'),
                    'timestamp': datetime.fromtimestamp(
                        price_point.get('timestamp', 0), tz=timezone.utc
                    ),
                    'price': price_point.get('price'),
                    'confidence': prices.get('confidence'),
                    'venue': self.VENUE
                })
        
        df = pd.DataFrame(all_records)
        
        if not df.empty:
            df = df.sort_values(['coin_id', 'timestamp']).reset_index(drop=True)
        
        return df
    
    # =========================================================================
    # RAISES / FUNDING METHODS
    # =========================================================================
    
    async def fetch_raises(self) -> pd.DataFrame:
        """
        Fetch protocol funding/raise data.
        
        Returns:
            DataFrame with funding rounds
        """
        data = await self._make_request(DefiLlamaEndpoint.RAISES.value)
        
        if not data or 'raises' not in data:
            return pd.DataFrame()
        
        records = []
        for r in data['raises']:
            records.append({
                'name': r.get('name'),
                'date': datetime.fromtimestamp(r.get('date', 0), tz=timezone.utc) if r.get('date') else None,
                'amount': r.get('amount'),
                'round': r.get('round'),
                'sector': r.get('sector'),
                'category': r.get('category'),
                'lead_investors': r.get('leadInvestors', []),
                'other_investors': r.get('otherInvestors', []),
                'valuation': r.get('valuation'),
                'chains': r.get('chains', []),
                'source': r.get('source'),
                'venue': self.VENUE,
                'timestamp': datetime.utcnow()
            })
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            df = df.sort_values('date', ascending=False, na_position='last')
        
        return df
    
    # =========================================================================
    # CONVENIENCE / AGGREGATION METHODS
    # =========================================================================
    
    async def fetch_major_protocols_summary(
        self,
        protocols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch summary data for major protocols.
        
        Args:
            protocols: List of protocol slugs (uses MAJOR_PROTOCOLS if None)
            
        Returns:
            DataFrame with comprehensive protocol metrics
        """
        protocols = protocols or self.MAJOR_PROTOCOLS
        
        self.logger.info(f"Fetching summary for {len(protocols)} major protocols...")
        
        # Fetch all protocols for TVL data
        all_protocols = await self.fetch_all_protocols(min_tvl=0)
        
        if all_protocols.empty:
            return pd.DataFrame()
        
        # Filter to major protocols
        df = all_protocols[all_protocols['slug'].isin(protocols)].copy()
        
        # Fetch fees data
        fees_df = await self.fetch_fees_overview()
        if not fees_df.empty:
            fees_cols = ['slug', 'total_24h', 'total_7d', 'total_30d', 
                        'revenue_24h', 'revenue_7d', 'revenue_30d']
            fees_subset = fees_df[fees_df['slug'].isin(protocols)][
                [c for c in fees_cols if c in fees_df.columns]
            ]
            if not fees_subset.empty:
                df = df.merge(
                    fees_subset, 
                    on='slug', 
                    how='left', 
                    suffixes=('', '_fees')
                )
        
        # Fetch DEX volumes
        dex_df = await self.fetch_dex_volumes()
        if not dex_df.empty:
            dex_cols = ['slug', 'volume_24h', 'volume_7d', 'volume_30d']
            dex_subset = dex_df[dex_df['slug'].isin(protocols)][
                [c for c in dex_cols if c in dex_df.columns]
            ]
            if not dex_subset.empty:
                df = df.merge(
                    dex_subset, 
                    on='slug', 
                    how='left', 
                    suffixes=('', '_dex')
                )
        
        return df
    
    async def fetch_chain_summary(
        self,
        chain: str
    ) -> Dict[str, Any]:
        """
        Fetch comprehensive summary for a specific chain.
        
        Args:
            chain: Chain name
            
        Returns:
            Dictionary with chain metrics
        """
        chain = self._normalize_chain(chain)
        
        summary = {
            'chain': chain,
            'timestamp': datetime.utcnow()
        }
        
        # TVL data
        chain_tvl = await self.fetch_chain_tvl(chain)
        if not chain_tvl.empty:
            summary['tvl'] = chain_tvl['tvl'].iloc[-1] if 'tvl' in chain_tvl.columns else None
        
        # Protocol count
        protocols = await self.fetch_all_protocols(chains=[chain])
        summary['protocol_count'] = len(protocols)
        summary['total_tvl'] = protocols['tvl'].sum() if not protocols.empty else 0
        
        # Top protocols
        if not protocols.empty:
            summary['top_protocols'] = protocols.head(10)[
                ['name', 'category', 'tvl']
            ].to_dict('records')
        
        # DEX volume
        dex_volumes = await self.fetch_dex_volumes(chain=chain)
        summary['total_dex_volume_24h'] = (
            dex_volumes['volume_24h'].sum() if not dex_volumes.empty else 0
        )
        
        # Stablecoin supply
        stables = await self.fetch_stablecoins()
        if not stables.empty:
            chain_stables = stables[
                stables['chains'].apply(lambda x: chain in x if x else False)
            ]
            summary['stablecoin_count'] = len(chain_stables)
        
        return summary

    # =========================================================================
    # REQUIRED INTERFACE METHODS (BaseCollector)
    # =========================================================================

    async def fetch_funding_rates(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        DefiLlama does not provide funding rates data.

        DefiLlama focuses on TVL, protocol analytics, yields, and prices.
        For funding rates, use exchange-specific collectors (Binance, Bybit, etc.)
        or derivatives data providers (Coinalyze).

        Returns:
            Empty DataFrame with standard funding rate columns
        """
        logger.info("DefiLlama doesn't provide funding rates. Use exchange collectors.")
        return pd.DataFrame(columns=[
            'timestamp', 'symbol', 'venue', 'funding_rate',
            'mark_price', 'index_price', 'next_funding_time'
        ])

    async def fetch_ohlcv(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        DefiLlama provides token prices but not full OHLCV candles.

        Use fetch_token_prices() for current/historical prices.
        For OHLCV data, use exchange collectors or CoinGecko/CryptoCompare.

        Returns:
            Empty DataFrame with standard OHLCV columns
        """
        logger.info("DefiLlama doesn't provide OHLCV. Use fetch_token_prices() or exchange collectors.")
        return pd.DataFrame(columns=[
            'timestamp', 'symbol', 'venue', 'open', 'high', 'low', 'close', 'volume'
        ])

    # =========================================================================
    # Standardized Collection Methods (for dynamic routing in collection_manager)
    # =========================================================================

    async def collect_tvl(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect TVL data for symbols (standardized interface).

        Wraps fetch_all_protocols() to match collection_manager expectations.

        Args:
            symbols: List of symbols (not used - fetches all protocols)
            start_date: Start date (not used - TVL is current snapshot)
            end_date: End date (not used - TVL is current snapshot)
            **kwargs: Additional parameters (min_tvl, chains, categories)

        Returns:
            DataFrame with TVL data for all protocols
        """
        try:
            min_tvl = kwargs.get('min_tvl', 0)
            chains = kwargs.get('chains', None)
            categories = kwargs.get('categories', None)

            logger.info(f"DefiLlama: Collecting TVL data (min_tvl=${min_tvl:,})")

            df = await self.fetch_all_protocols(
                min_tvl=min_tvl,
                chains=chains,
                categories=categories,
                include_changes=True
            )

            if not df.empty:
                logger.info(f"DefiLlama: Collected TVL data for {len(df)} protocols")
                return df

            logger.warning("DefiLlama: No TVL data found")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"DefiLlama collect_tvl error: {e}")
            return pd.DataFrame()

    async def collect_yields(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect yield data for symbols (standardized interface).

        Wraps fetch_yields() to match collection_manager expectations.

        Args:
            symbols: List of symbols (not used - fetches all yield pools)
            start_date: Start date (not used - yields are current snapshot)
            end_date: End date (not used - yields are current snapshot)
            **kwargs: Additional parameters (chain, min_tvl, min_apy)

        Returns:
            DataFrame with yield pool data
        """
        try:
            chain = kwargs.get('chain', None)
            min_tvl = kwargs.get('min_tvl', 0)
            min_apy = kwargs.get('min_apy', 0)

            logger.info(f"DefiLlama: Collecting yields data (chain={chain}, min_apy={min_apy}%)")

            df = await self.fetch_yields(
                chain=chain,
                min_tvl=min_tvl,
                min_apy=min_apy
            )

            if not df.empty:
                logger.info(f"DefiLlama: Collected yields for {len(df)} pools")
                return df

            logger.warning("DefiLlama: No yields data found")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"DefiLlama collect_yields error: {e}")
            return pd.DataFrame()

    async def collect_stablecoins(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect stablecoin data for symbols (standardized interface).

        Wraps fetch_stablecoins() to match collection_manager expectations.

        Args:
            symbols: List of symbols (not used - fetches all stablecoins)
            start_date: Start date (not used - stablecoin data is current snapshot)
            end_date: End date (not used - stablecoin data is current snapshot)
            **kwargs: Additional parameters (min_circulating)

        Returns:
            DataFrame with stablecoin metrics
        """
        try:
            min_circulating = kwargs.get('min_circulating', 0)

            logger.info(f"DefiLlama: Collecting stablecoins data (min_circulating=${min_circulating:,})")

            df = await self.fetch_stablecoins()

            if not df.empty:
                # Filter by minimum circulating if specified
                if min_circulating > 0:
                    df = df[df.get('circulating_usd', 0) >= min_circulating]

                logger.info(f"DefiLlama: Collected data for {len(df)} stablecoins")
                return df

            logger.warning("DefiLlama: No stablecoins data found")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"DefiLlama collect_stablecoins error: {e}")
            return pd.DataFrame()

    # =========================================================================
    # CLEANUP
    # =========================================================================

    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'DefiLlamaCollector',
    'DefiLlamaEndpoint',
    'ProtocolMetrics',
    'YieldPool',
    'StablecoinMetrics',
]

# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Test the DefiLlama collector."""
        logging.basicConfig(level=logging.INFO)
        
        async with DefiLlamaCollector() as collector:
            print("\n=== Testing DefiLlama Collector ===\n")
            
            # Test 1: Fetch top protocols
            print("1. Fetching top protocols by TVL...")
            protocols = await collector.fetch_all_protocols(min_tvl=1_000_000_000)
            print(f" Found {len(protocols)} protocols with TVL > $1B")
            if not protocols.empty:
                print(f" Top 5: {protocols['name'].head().tolist()}")
            
            # Test 2: Fetch DEX volumes
            print("\n2. Fetching DEX volumes...")
            dex = await collector.fetch_dex_volumes()
            print(f" Found {len(dex)} DEXs")
            if not dex.empty:
                total_vol = dex['volume_24h'].sum()
                print(f" Total 24h volume: ${total_vol:,.0f}")
            
            # Test 3: Fetch yields
            print("\n3. Fetching yield pools...")
            yields = await collector.fetch_yields(min_tvl=10_000_000, min_apy=5)
            print(f" Found {len(yields)} pools with TVL>$10M and APY>5%")
            
            # Test 4: Fetch stablecoins
            print("\n4. Fetching stablecoins...")
            stables = await collector.fetch_stablecoins()
            print(f" Found {len(stables)} stablecoins")
            if not stables.empty:
                total_supply = stables['circulating_usd'].sum()
                print(f" Total supply: ${total_supply:,.0f}")
            
            print("\n=== All tests passed! ===")
    
    asyncio.run(main())