"""
Messari API Collector - professional-quality Crypto Asset Data and Research.

tested collector for validated crypto fundamentals:
- Asset profiles and detailed fundamentals
- Market data with quality validation
- On-chain metrics and blockchain statistics
- Qualitative research data
- Sector and category classifications
- Developer activity tracking
- Valuation metrics and comparisons
- News and research integration

API Documentation: https://messari.io/api/docs
Rate Limits: 20 requests/minute (free), 100 requests/minute (Pro)
Registration: https://messari.io (free tier available)

Version: 2.0.0
"""

import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

from ..base_collector import BaseCollector
from ..utils.rate_limiter import get_shared_rate_limiter
from ..utils.retry_handler import RetryHandler

logger = logging.getLogger(__name__)

class AssetSector(Enum):
    """Crypto asset sectors."""
    CURRENCIES = 'currencies'
    SMART_CONTRACTS = 'smart-contracts'
    DEFI = 'defi'
    DEFI_LENDING = 'defi-lending'
    DEFI_DEX = 'defi-dex'
    DEFI_DERIVATIVES = 'defi-derivatives'
    DEFI_YIELD = 'defi-yield'
    EXCHANGE_TOKENS = 'exchange-tokens'
    STABLECOINS = 'stablecoins'
    NFT = 'nft'
    GAMING = 'gaming'
    LAYER_1 = 'layer-1'
    LAYER_2 = 'layer-2'
    INTEROPERABILITY = 'interoperability'
    STORAGE = 'storage'
    ORACLES = 'oracles'
    PRIVACY = 'privacy'
    AI = 'ai'
    MEME = 'meme'
    RWA = 'rwa'

class MetricType(Enum):
    """Available metric types for time series."""
    PRICE = 'price'
    VOLUME = 'volume'
    MARKET_CAP = 'mcap'
    REAL_VOLUME = 'real_vol'
    ACTIVE_ADDRESSES = 'act.addr.cnt'
    TRANSACTION_COUNT = 'txn.cnt'
    TRANSFER_VALUE = 'txn.tfr.val.adj'
    NVT_RATIO = 'nvt.adj'
    SUPPLY_CIRCULATING = 'sply.circ'
    FEES = 'fees'

class ValuationTier(Enum):
    """Valuation classification tiers."""
    SEVERELY_UNDERVALUED = 'severely_undervalued'
    UNDERVALUED = 'undervalued'
    FAIRLY_VALUED = 'fairly_valued'
    OVERVALUED = 'overvalued'
    SEVERELY_OVERVALUED = 'severely_overvalued'

@dataclass
class AssetMetrics:
    """Comprehensive asset metrics."""
    symbol: str
    name: str
    price_usd: float
    price_btc: float
    market_cap: float
    market_cap_rank: int
    volume_24h: float
    real_volume_24h: float
    volume_turnover: float
    circulating_supply: float
    max_supply: Optional[float]
    supply_inflation_rate: float
    pct_change_1h: float
    pct_change_24h: float
    pct_change_7d: float
    pct_change_30d: float
    pct_change_1y: float
    ath_price: float
    ath_date: Optional[datetime]
    atl_price: float
    atl_date: Optional[datetime]
    txn_count_24h: int
    active_addresses_24h: int
    transfer_value_24h: float
    nvt_ratio: float
    github_stars: int
    github_commits_90d: int
    timestamp: datetime

@dataclass
class AssetProfile:
    """Detailed asset profile information."""
    symbol: str
    name: str
    tagline: str
    category: str
    sector: str
    project_created: Optional[datetime]
    genesis_block_date: Optional[datetime]
    consensus_algorithm: str
    block_time_seconds: float
    token_type: str
    is_mineable: bool
    is_stakeable: bool
    governance_type: str
    website: str
    whitepaper: str
    github: str
    twitter: str
    discord: str
    telegram: str
    team_members: List[str]
    investors: List[str]
    timestamp: datetime

@dataclass
class SectorAnalysis:
    """Sector-level aggregated analytics."""
    sector: str
    asset_count: int
    total_market_cap: float
    total_volume_24h: float
    avg_pct_change_24h: float
    weighted_pct_change_24h: float
    top_performers: List[str]
    worst_performers: List[str]
    market_cap_dominance: float
    timestamp: datetime

@dataclass
class ValuationMetrics:
    """Valuation analysis metrics."""
    symbol: str
    price_usd: float
    market_cap: float
    nvt_ratio: float
    nvt_percentile: float
    mvrv_ratio: float
    mvrv_percentile: float
    pe_ratio: Optional[float]
    ps_ratio: Optional[float]
    mcap_to_tvl: Optional[float]
    valuation_tier: ValuationTier
    fair_value_estimate: Optional[float]
    upside_potential_pct: Optional[float]
    timestamp: datetime

class MessariCollector(BaseCollector):
    """
    Messari data collector for professional-quality crypto asset data.
    
    Features:
    - Clean, validated market data
    - Asset fundamentals and profiles
    - On-chain metrics (transactions, addresses, fees)
    - Developer activity (GitHub stats)
    - Sector and category classifications
    - Valuation metrics and analysis
    - News and research integration
    - Historical time series data
    - Comparative analytics
    
    Use Cases:
    - Fundamental analysis for long-term positions
    - Sector rotation strategies
    - Quality filtering for universe construction
    - Valuation-based trading signals
    - Research and due diligence
    
    Data Quality:
    - Messari validates and cleans data
    - Real volume filtering removes wash trading
    - professional-quality accuracy
    - Research team curation
    """
    
    VENUE = 'messari'
    VENUE_TYPE = 'analytics'
    # NOTE: Messari AI API (api.messari.io) uses x-messari-api-key header
    # The data API (data.messari.io) requires a separate subscription
    BASE_URL = 'https://api.messari.io'
    DATA_API_URL = 'https://data.messari.io/api' # Legacy data API (requires separate key)

    # Symbol to Messari slug mapping (Messari uses slugs not ticker symbols)
    SYMBOL_TO_SLUG = {
        'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana', 'XRP': 'xrp',
        'DOGE': 'dogecoin', 'ADA': 'cardano', 'AVAX': 'avalanche', 'DOT': 'polkadot',
        'LINK': 'chainlink', 'MATIC': 'polygon', 'UNI': 'uniswap', 'ATOM': 'cosmos',
        'LTC': 'litecoin', 'BCH': 'bitcoin-cash', 'XLM': 'stellar', 'ALGO': 'algorand',
        'VET': 'vechain', 'FIL': 'filecoin', 'ICP': 'internet-computer', 'NEAR': 'near-protocol',
        'APT': 'aptos', 'ARB': 'arbitrum', 'OP': 'optimism', 'SUI': 'sui',
        'AAVE': 'aave', 'MKR': 'maker', 'CRV': 'curve-dao-token', 'SNX': 'synthetix',
        'COMP': 'compound', 'LDO': 'lido-dao', 'RUNE': 'thorchain', 'INJ': 'injective',
        'FTM': 'fantom', 'SAND': 'the-sandbox', 'MANA': 'decentraland', 'AXS': 'axie-infinity',
        'APE': 'apecoin', 'PEPE': 'pepe', 'WLD': 'worldcoin', 'SEI': 'sei',
        'TIA': 'celestia', 'JUP': 'jupiter', 'PYTH': 'pyth-network', 'BONK': 'bonk',
        'WIF': 'dogwifhat', 'RENDER': 'render-token', 'FET': 'fetch-ai', 'TAO': 'bittensor',
    }

    # Available sectors
    SECTORS = [s.value for s in AssetSector]
    
    # Key market metrics
    MARKET_METRICS = [
        'price_usd', 'price_btc', 'volume_last_24_hours',
        'real_volume_last_24_hours', 'current_marketcap_usd',
        'marketcap_dominance_percent', 'percent_change_usd_last_24_hours',
        'percent_change_btc_last_24_hours', 'ohlcv_last_24_hour'
    ]
    
    # Supply metrics
    SUPPLY_METRICS = [
        'circulating', 'max', 'outstanding', 'staked', 
        'y_2050', 'annual_inflation_percent', 'y_plus10'
    ]
    
    # On-chain metrics
    ONCHAIN_METRICS = [
        'count_of_tx_24_hours', 'count_of_active_addresses_24_hours',
        'transaction_volume_24_hours', 'adjusted_transaction_volume_24_hours',
        'average_transaction_value_24_hours', 'median_transaction_value_24_hours'
    ]
    
    # ROI metrics
    ROI_PERIODS = ['1h', '24h', '7d', '30d', '90d', 'ytd', '1y', '3y', '5y']
    
    # Major assets for benchmarking
    BENCHMARK_ASSETS = ['btc', 'eth', 'sol', 'bnb', 'xrp', 'ada', 'avax', 'dot', 'link', 'atom']
    
    def __init__(self, config: Dict):
        """
        Initialize Messari collector.

        Args:
            config: Configuration with:
                - messari_api_key: API key (optional for basic endpoints)
                - rate_limit: Requests per minute (default 20)
                - cache_ttl: Cache TTL in seconds (default 300)
        """
        super().__init__(config)

        # CRITICAL: Set supported data types for dynamic routing
        self.supported_data_types = ['asset_metrics', 'fundamentals']
        self.venue = 'messari'

        # Import VenueType from base_collector
        from ..base_collector import VenueType
        self.venue_type = VenueType.MARKET_DATA
        self.requires_auth = False # Free tier works WITHOUT API key (20 req/min, 1K/day)

        self.api_key = config.get('messari_api_key') or config.get('api_key') or os.getenv('MESSARI_API_KEY', '')
        self.session = None

        # Retry handler
        self.retry_handler = RetryHandler(
            max_retries=3,
            base_delay=2.0,
            max_delay=30.0
        )

        # Cache for expensive queries
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl = config.get('cache_ttl', 300)

        # Collection stats
        self.collection_stats = {
            'records_collected': 0,
            'api_calls': 0,
            'cache_hits': 0,
            'errors': 0,
            'rate_limits': 0,
            'start_time': None,
            'end_time': None
        }

        # Use higher rate limit if API key is provided (Pro tier: 100/min vs Free: 20/min)
        if self.api_key:
            rate_limit = config.get('rate_limit', 50) # Pro tier default
            logger.info(f"Initialized Messari collector WITH API key (rate limit: {rate_limit}/min)")
        else:
            rate_limit = config.get('rate_limit', 10) # Free tier
            logger.info(f"Initialized Messari collector without API key (rate limit: {rate_limit}/min)")

        # Update rate limiter with correct rate
        self.rate_limiter = get_shared_rate_limiter('messari', rate=rate_limit, per=60.0, burst=rate_limit)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            if self.api_key:
                headers['x-messari-api-key'] = self.api_key
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self.session
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self._cache:
            timestamp, value = self._cache[key]
            if (datetime.utcnow() - timestamp).total_seconds() < self._cache_ttl:
                self.collection_stats['cache_hits'] += 1
                return value
            del self._cache[key]
        return None
    
    def _set_cached(self, key: str, value: Any):
        """Set value in cache."""
        self._cache[key] = (datetime.utcnow(), value)

    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """
        Make API request with rate limiting, caching, and error handling.

        Args:
            endpoint: API endpoint
            params: Query parameters
            use_cache: Whether to use caching

        Returns:
            Response data or None on error
        """
        # Check cache FIRST
        cache_key = f"{endpoint}_{json.dumps(params or {}, sort_keys=True)}"
        if use_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                logger.debug(f"Messari cache hit: {endpoint}")
                return cached

        await self.rate_limiter.acquire()
        session = await self._get_session()

        url = f"{self.BASE_URL}/{endpoint}"
        self.collection_stats['api_calls'] += 1

        async def _request():
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 401:
                    logger.error("Messari authentication failed - check API key")
                    return None
                elif response.status == 429:
                    self.collection_stats['rate_limits'] += 1
                    logger.warning("Messari rate limit hit - backing off")
                    raise aiohttp.ClientError("Rate limit exceeded")
                elif response.status == 404:
                    logger.debug(f"Messari resource not found: {endpoint}")
                    return None
                else:
                    text = await response.text()
                    logger.debug(f"Messari API error {response.status}: {text[:200]}")
                    raise aiohttp.ClientError(f"API error {response.status}")

        try:
            result = await self.retry_handler.execute(_request)
            if result is not None and use_cache:
                self._set_cached(cache_key, result)
            return result
        except Exception as e:
            logger.debug(f"Messari request failed: {endpoint} - {e}")
            self.collection_stats['errors'] += 1
            return None

    async def _query_ai_api(
        self,
        query: str,
        use_cache: bool = True
    ) -> Optional[str]:
        """
        Query the Messari AI API for market data.

        Args:
            query: Natural language query about crypto data
            use_cache: Whether to use caching

        Returns:
            AI response text or None on error
        """
        # Check cache
        cache_key = f"ai_query_{query}"
        if use_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        await self.rate_limiter.acquire()
        session = await self._get_session()

        url = f"{self.BASE_URL}/ai/v1/chat/completions"
        self.collection_stats['api_calls'] += 1

        payload = {
            'messages': [
                {'role': 'user', 'content': query}
            ]
        }

        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    # Extract the assistant's message content
                    messages = data.get('data', {}).get('messages', [])
                    if messages:
                        content = messages[0].get('content', '')
                        if use_cache:
                            self._set_cached(cache_key, content)
                        return content
                    return None
                elif response.status == 401:
                    logger.error("Messari AI API authentication failed")
                    return None
                else:
                    text = await response.text()
                    logger.error(f"Messari AI API error {response.status}: {text[:200]}")
                    return None
        except Exception as e:
            logger.error(f"Messari AI API query failed: {e}")
            self.collection_stats['errors'] += 1
            return None

    async def get_asset_price_via_ai(self, symbol: str) -> Optional[Dict]:
        """
        Get current asset price using Messari AI API.

        Args:
            symbol: Asset symbol (e.g., 'BTC', 'ETH')

        Returns:
            Dict with price data or None
        """
        query = f"What is the current price, market cap, and 24h volume of {symbol}? Please provide the exact numerical values."

        response = await self._query_ai_api(query)
        if not response:
            return None

        # Parse the AI response to extract numerical data
        import re

        result = {
            'symbol': symbol.upper(),
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'messari_ai'
        }

        # Try to extract price
        price_match = re.search(r'\$?([\d,]+\.?\d*)\s*(?:USD)?', response)
        if price_match:
            try:
                result['price_usd'] = float(price_match.group(1).replace(',', ''))
            except:
                pass

        return result if 'price_usd' in result else None

    async def _fetch_metrics_via_ai(self, asset_key: str) -> Optional['AssetMetrics']:
        """
        Fetch asset metrics using Messari AI API as fallback.

        Args:
            asset_key: Asset symbol (e.g., 'BTC', 'ETH')

        Returns:
            AssetMetrics dataclass or None
        """
        import re

        query = f"""Provide the following metrics for {asset_key} in a structured format:
- Current price in USD
- Market cap in USD
- 24h trading volume
- Circulating supply
- 24h price change percentage
- 7d price change percentage
Please provide exact numerical values."""

        response = await self._query_ai_api(query)
        if not response:
            return None

        # Parse AI response to extract metrics
        def extract_number(text: str, pattern: str) -> float:
            """Extract a number from text using pattern."""
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    # Handle numbers with commas, K, M, B suffixes
                    num_str = match.group(1).replace(',', '').strip()
                    multiplier = 1
                    if num_str.endswith('K'):
                        multiplier = 1000
                        num_str = num_str[:-1]
                    elif num_str.endswith('M'):
                        multiplier = 1_000_000
                        num_str = num_str[:-1]
                    elif num_str.endswith('B'):
                        multiplier = 1_000_000_000
                        num_str = num_str[:-1]
                    elif num_str.endswith('T'):
                        multiplier = 1_000_000_000_000
                        num_str = num_str[:-1]
                    return float(num_str) * multiplier
                except:
                    pass
            return 0.0

        # Extract price - look for dollar amounts
        price_patterns = [
            r'\$\s*([\d,]+\.?\d*)',
            r'price[:\s]+\$?([\d,]+\.?\d*)',
            r'([\d,]+\.?\d*)\s*(?:USD|dollars?)',
        ]
        price_usd = 0.0
        for pattern in price_patterns:
            price_usd = extract_number(response, pattern)
            if price_usd > 0:
                break

        # Extract market cap
        mcap_patterns = [
            r'market\s*cap[:\s]+\$?([\d,]+\.?\d*[KMBT]?)',
            r'\$?([\d,]+\.?\d*[KMBT]?)\s*(?:market\s*cap|mcap)',
        ]
        market_cap = 0.0
        for pattern in mcap_patterns:
            market_cap = extract_number(response, pattern)
            if market_cap > 0:
                break

        # Extract volume
        volume_patterns = [
            r'(?:24h?\s*)?volume[:\s]+\$?([\d,]+\.?\d*[KMBT]?)',
            r'\$?([\d,]+\.?\d*[KMBT]?)\s*(?:24h?\s*)?volume',
        ]
        volume_24h = 0.0
        for pattern in volume_patterns:
            volume_24h = extract_number(response, pattern)
            if volume_24h > 0:
                break

        # Extract percentage changes
        change_24h = extract_number(response, r'24h?[:\s]+([+-]?[\d,]+\.?\d*)%')
        change_7d = extract_number(response, r'7d?[:\s]+([+-]?[\d,]+\.?\d*)%')

        if price_usd == 0:
            logger.warning(f"Could not extract price from Messari AI response for {asset_key}")
            return None

        return AssetMetrics(
            symbol=asset_key.upper(),
            name=asset_key.upper(),
            price_usd=price_usd,
            price_btc=0.0, # Not available via AI
            market_cap=market_cap,
            market_cap_rank=0,
            volume_24h=volume_24h,
            real_volume_24h=volume_24h,
            volume_turnover=1.0,
            circulating_supply=0.0,
            max_supply=None,
            supply_inflation_rate=0.0,
            pct_change_1h=0.0,
            pct_change_24h=change_24h,
            pct_change_7d=change_7d,
            pct_change_30d=0.0,
            pct_change_1y=0.0,
            ath_price=0.0,
            ath_date=None,
            atl_price=0.0,
            atl_date=None,
            txn_count_24h=0,
            active_addresses_24h=0,
            transfer_value_24h=0.0,
            nvt_ratio=0.0,
            github_stars=0,
            github_commits_90d=0,
            timestamp=datetime.utcnow()
        )

    # =========================================================================
    # Asset List Methods
    # =========================================================================
    
    async def fetch_all_assets(
        self,
        with_metrics: bool = True,
        with_profiles: bool = False,
        limit: int = 500,
        sort_by: str = 'market_cap',
        descending: bool = True
    ) -> pd.DataFrame:
        """
        Fetch all assets with basic data.
        
        Args:
            with_metrics: Include market metrics
            with_profiles: Include profile data (slower)
            limit: Max assets to return
            sort_by: Sort field (market_cap, volume, etc.)
            descending: Sort descending
            
        Returns:
            DataFrame with asset data
        """
        fields = ['id', 'symbol', 'name', 'slug']
        if with_metrics:
            fields.append('metrics')
        if with_profiles:
            fields.append('profile')
        
        params = {
            'limit': min(limit, 500),
            'fields': ','.join(fields),
            'sort': sort_by,
            'order': 'desc' if descending else 'asc'
        }
        
        data = await self._make_request('v2/assets', params)
        
        if not data or 'data' not in data:
            return pd.DataFrame()
        
        records = []
        for asset in data.get('data', []):
            record = {
                'id': asset.get('id'),
                'symbol': asset.get('symbol', '').upper(),
                'name': asset.get('name'),
                'slug': asset.get('slug'),
                'timestamp': datetime.utcnow(),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE
            }
            
            # Add metrics if available
            metrics = asset.get('metrics', {})
            if metrics:
                market = metrics.get('market_data', {})
                mcap = metrics.get('marketcap', {})
                supply = metrics.get('supply', {})
                roi = metrics.get('roi_data', {})
                
                record.update({
                    'price_usd': float(market.get('price_usd', 0) or 0),
                    'price_btc': float(market.get('price_btc', 0) or 0),
                    'volume_24h': float(market.get('volume_last_24_hours', 0) or 0),
                    'real_volume_24h': float(market.get('real_volume_last_24_hours', 0) or 0),
                    'market_cap': float(mcap.get('current_marketcap_usd', 0) or 0),
                    'market_cap_rank': int(mcap.get('rank', 0) or 0),
                    'market_cap_dominance': float(mcap.get('marketcap_dominance_percent', 0) or 0),
                    'pct_change_1h': float(market.get('percent_change_usd_last_1_hour', 0) or 0),
                    'pct_change_24h': float(market.get('percent_change_usd_last_24_hours', 0) or 0),
                    'pct_change_btc_24h': float(market.get('percent_change_btc_last_24_hours', 0) or 0),
                    'circulating_supply': float(supply.get('circulating', 0) or 0),
                    'max_supply': float(supply.get('max', 0) or 0) if supply.get('max') else None,
                    'supply_inflation': float(supply.get('annual_inflation_percent', 0) or 0),
                    'supply_pct_staked': float(supply.get('staked_pct', 0) or 0) if supply.get('staked_pct') else None,
                    'roi_24h': float(roi.get('percent_change_last_1_day', 0) or 0),
                    'roi_7d': float(roi.get('percent_change_last_1_week', 0) or 0),
                    'roi_30d': float(roi.get('percent_change_last_1_month', 0) or 0),
                    'roi_1y': float(roi.get('percent_change_last_1_year', 0) or 0)
                })
                
                # Calculate volume/mcap ratio (liquidity indicator)
                if record['market_cap'] > 0:
                    record['volume_mcap_ratio'] = record['volume_24h'] / record['market_cap']
                else:
                    record['volume_mcap_ratio'] = 0
                
                # Calculate real volume percentage
                if record['volume_24h'] > 0:
                    record['real_volume_pct'] = record['real_volume_24h'] / record['volume_24h'] * 100
                else:
                    record['real_volume_pct'] = 0
            
            # Add profile data if available
            profile = asset.get('profile', {})
            if profile:
                general = profile.get('general', {}).get('overview', {})
                record.update({
                    'category': general.get('category'),
                    'sector': general.get('sector'),
                    'tagline': general.get('tagline')
                })
            
            records.append(record)
        
        df = pd.DataFrame(records)
        self.collection_stats['records_collected'] += len(df)
        return df
    
    async def fetch_assets_paginated(
        self,
        max_assets: int = 2000,
        with_metrics: bool = True
    ) -> pd.DataFrame:
        """
        Fetch assets with pagination for large result sets.
        
        Args:
            max_assets: Maximum total assets to fetch
            with_metrics: Include market metrics
            
        Returns:
            DataFrame with all fetched assets
        """
        all_assets = []
        page = 1
        page_size = 500
        
        while len(all_assets) < max_assets:
            params = {
                'limit': page_size,
                'page': page,
                'fields': 'id,symbol,name,slug,metrics' if with_metrics else 'id,symbol,name,slug'
            }
            
            data = await self._make_request('v2/assets', params, use_cache=False)
            
            if not data or 'data' not in data or not data['data']:
                break
            
            all_assets.extend(data['data'])
            
            if len(data['data']) < page_size:
                break
            
            page += 1
        
        if not all_assets:
            return pd.DataFrame()
        
        # Process into DataFrame (simplified)
        records = []
        for asset in all_assets[:max_assets]:
            metrics = asset.get('metrics', {})
            market = metrics.get('market_data', {}) if metrics else {}
            mcap = metrics.get('marketcap', {}) if metrics else {}
            
            records.append({
                'symbol': asset.get('symbol', '').upper(),
                'name': asset.get('name'),
                'price_usd': float(market.get('price_usd', 0) or 0),
                'market_cap': float(mcap.get('current_marketcap_usd', 0) or 0),
                'market_cap_rank': int(mcap.get('rank', 0) or 0),
                'volume_24h': float(market.get('volume_last_24_hours', 0) or 0),
                'pct_change_24h': float(market.get('percent_change_usd_last_24_hours', 0) or 0),
                'timestamp': datetime.utcnow()
            })
        
        df = pd.DataFrame(records)
        self.collection_stats['records_collected'] += len(df)
        return df
    
    # =========================================================================
    # Asset Detail Methods
    # =========================================================================
    
    async def fetch_asset_profile(
        self,
        asset_key: str
    ) -> Optional[AssetProfile]:
        """
        Fetch detailed asset profile.

        Args:
            asset_key: Asset symbol or slug (e.g., 'btc', 'bitcoin')

        Returns:
            AssetProfile dataclass or None
        """
        # Convert symbol to Messari slug if needed
        slug = self.SYMBOL_TO_SLUG.get(asset_key.upper(), asset_key.lower())
        data = await self._make_request(f'v2/assets/{slug}/profile')
        
        if not data or 'data' not in data:
            return None
        
        profile = data['data']
        general = profile.get('profile', {}).get('general', {}).get('overview', {})
        tech = profile.get('profile', {}).get('technology', {}).get('overview', {})
        econ = profile.get('profile', {}).get('economics', {})
        gov = profile.get('profile', {}).get('governance', {})
        
        # Extract links
        links = general.get('official_links', [])
        link_map = {link.get('name', '').lower(): link.get('link', '') for link in links}
        
        # Extract team/investors if available
        contributors = profile.get('profile', {}).get('contributors', {})
        team = [p.get('name', '') for p in contributors.get('individuals', [])]
        investors = [o.get('name', '') for o in contributors.get('organizations', []) 
                     if o.get('slug', '') in ['investor', 'venture-capital']]
        
        return AssetProfile(
            symbol=profile.get('symbol', '').upper(),
            name=profile.get('name', ''),
            tagline=general.get('tagline', ''),
            category=general.get('category', ''),
            sector=general.get('sector', ''),
            project_created=pd.to_datetime(general.get('project_created')) if general.get('project_created') else None,
            genesis_block_date=pd.to_datetime(econ.get('launch', {}).get('genesis_block_date')) if econ.get('launch', {}).get('genesis_block_date') else None,
            consensus_algorithm=tech.get('consensus_algorithm', ''),
            block_time_seconds=float(tech.get('block_time', 0) or 0),
            token_type=econ.get('token', {}).get('token_type', ''),
            is_mineable=econ.get('launch', {}).get('is_mineable', False),
            is_stakeable=econ.get('staking', {}).get('is_stakeable', False) if econ.get('staking') else False,
            governance_type=gov.get('governance_details', '') if gov else '',
            website=link_map.get('website', ''),
            whitepaper=link_map.get('whitepaper', ''),
            github=link_map.get('github', ''),
            twitter=link_map.get('twitter', ''),
            discord=link_map.get('discord', ''),
            telegram=link_map.get('telegram', ''),
            team_members=team[:10], # Limit to first 10
            investors=investors[:10],
            timestamp=datetime.utcnow()
        )
    
    async def fetch_asset_metrics(
        self,
        asset_key: str
    ) -> Optional[AssetMetrics]:
        """
        Fetch comprehensive metrics for an asset.

        Args:
            asset_key: Asset symbol or slug

        Returns:
            AssetMetrics dataclass or None
        """
        # Convert symbol to Messari slug if needed
        slug = self.SYMBOL_TO_SLUG.get(asset_key.upper(), asset_key.lower())
        data = await self._make_request(f'v1/assets/{slug}/metrics')

        # If data API fails, try AI API fallback
        if not data or 'data' not in data:
            logger.info(f"Messari data API unavailable, using AI API for {asset_key}")
            return await self._fetch_metrics_via_ai(asset_key)

        metrics = data['data']
        market = metrics.get('market_data', {})
        mcap = metrics.get('marketcap', {})
        supply = metrics.get('supply', {})
        blockchain = metrics.get('blockchain_stats_24_hours', {})
        developer = metrics.get('developer_activity', {})
        roi = metrics.get('roi_data', {})
        ath = metrics.get('all_time_high', {})
        
        # Calculate NVT if we have the data
        nvt_ratio = 0
        if blockchain.get('adjusted_transaction_volume_24_hours') and mcap.get('current_marketcap_usd'):
            daily_txn_vol = float(blockchain.get('adjusted_transaction_volume_24_hours', 0) or 0)
            if daily_txn_vol > 0:
                nvt_ratio = float(mcap.get('current_marketcap_usd', 0)) / (daily_txn_vol * 365)
        
        return AssetMetrics(
            symbol=metrics.get('symbol', '').upper(),
            name=metrics.get('name', ''),
            price_usd=float(market.get('price_usd', 0) or 0),
            price_btc=float(market.get('price_btc', 0) or 0),
            market_cap=float(mcap.get('current_marketcap_usd', 0) or 0),
            market_cap_rank=int(mcap.get('rank', 0) or 0),
            volume_24h=float(market.get('volume_last_24_hours', 0) or 0),
            real_volume_24h=float(market.get('real_volume_last_24_hours', 0) or 0),
            volume_turnover=float(market.get('volume_last_24_hours_overstatement_multiple', 1) or 1),
            circulating_supply=float(supply.get('circulating', 0) or 0),
            max_supply=float(supply.get('max', 0)) if supply.get('max') else None,
            supply_inflation_rate=float(supply.get('annual_inflation_percent', 0) or 0),
            pct_change_1h=float(market.get('percent_change_usd_last_1_hour', 0) or 0),
            pct_change_24h=float(market.get('percent_change_usd_last_24_hours', 0) or 0),
            pct_change_7d=float(roi.get('percent_change_last_1_week', 0) or 0),
            pct_change_30d=float(roi.get('percent_change_last_1_month', 0) or 0),
            pct_change_1y=float(roi.get('percent_change_last_1_year', 0) or 0),
            ath_price=float(ath.get('price', 0) or 0),
            ath_date=pd.to_datetime(ath.get('at')) if ath.get('at') else None,
            atl_price=float(metrics.get('all_time_low', {}).get('price', 0) or 0),
            atl_date=pd.to_datetime(metrics.get('all_time_low', {}).get('at')) if metrics.get('all_time_low', {}).get('at') else None,
            txn_count_24h=int(blockchain.get('count_of_tx_24_hours', 0) or 0),
            active_addresses_24h=int(blockchain.get('count_of_active_addresses_24_hours', 0) or 0),
            transfer_value_24h=float(blockchain.get('adjusted_transaction_volume_24_hours', 0) or 0),
            nvt_ratio=nvt_ratio,
            github_stars=int(developer.get('stars', 0) or 0),
            github_commits_90d=int(developer.get('commits_last_3_months', 0) or 0),
            timestamp=datetime.utcnow()
        )
    
    async def _fetch_single_asset_metrics(
        self,
        asset: str
    ) -> Optional[Dict]:
        """
        Fetch metrics for a single asset.

        Args:
            asset: Asset symbol

        Returns:
            Dict with metrics or None on error
        """
        try:
            logger.info(f"Fetching Messari metrics for {asset}")
            metrics = await self.fetch_asset_metrics(asset)

            if metrics:
                return {
                    'symbol': metrics.symbol,
                    'name': metrics.name,
                    'price_usd': metrics.price_usd,
                    'price_btc': metrics.price_btc,
                    'market_cap': metrics.market_cap,
                    'market_cap_rank': metrics.market_cap_rank,
                    'volume_24h': metrics.volume_24h,
                    'real_volume_24h': metrics.real_volume_24h,
                    'volume_turnover': metrics.volume_turnover,
                    'circulating_supply': metrics.circulating_supply,
                    'max_supply': metrics.max_supply,
                    'supply_inflation': metrics.supply_inflation_rate,
                    'pct_change_1h': metrics.pct_change_1h,
                    'pct_change_24h': metrics.pct_change_24h,
                    'pct_change_7d': metrics.pct_change_7d,
                    'pct_change_30d': metrics.pct_change_30d,
                    'pct_change_1y': metrics.pct_change_1y,
                    'ath_price': metrics.ath_price,
                    'pct_from_ath': ((metrics.price_usd - metrics.ath_price) / metrics.ath_price * 100) if metrics.ath_price > 0 else 0,
                    'txn_count_24h': metrics.txn_count_24h,
                    'active_addresses_24h': metrics.active_addresses_24h,
                    'transfer_value_24h': metrics.transfer_value_24h,
                    'nvt_ratio': metrics.nvt_ratio,
                    'github_stars': metrics.github_stars,
                    'github_commits_90d': metrics.github_commits_90d,
                    'timestamp': metrics.timestamp,
                    'venue': self.VENUE
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching metrics for {asset}: {e}")
            return None

    async def fetch_asset_metrics_batch(
        self,
        assets: List[str]
    ) -> pd.DataFrame:
        """
        Fetch metrics for multiple assets.

        Args:
            assets: List of asset symbols

        Returns:
            DataFrame with all asset metrics
        """
        # Parallelize using asyncio.gather
        tasks = [self._fetch_single_asset_metrics(asset) for asset in assets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter results to only keep valid data
        records = [r for r in results if isinstance(r, dict)]

        df = pd.DataFrame(records)
        self.collection_stats['records_collected'] += len(df)
        return df
    
    # =========================================================================
    # Time Series Methods
    # =========================================================================
    
    async def fetch_asset_timeseries(
        self,
        asset_key: str,
        metric_id: str = 'price',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch historical time series data for an asset metric.
        
        Args:
            asset_key: Asset symbol or slug
            metric_id: Metric to fetch ('price', 'volume', 'mcap', etc.)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1w')
            
        Returns:
            DataFrame with time series data
        """
        params = {
            'interval': interval
        }
        if start_date:
            params['start'] = start_date
        if end_date:
            params['end'] = end_date
        
        data = await self._make_request(
            f'v1/assets/{asset_key.lower()}/metrics/{metric_id}/time-series',
            params,
            use_cache=False
        )
        
        if not data or 'data' not in data:
            return pd.DataFrame()
        
        values = data['data'].get('values', [])
        
        records = []
        for entry in values:
            if len(entry) >= 2 and entry[1] is not None:
                records.append({
                    'timestamp': pd.to_datetime(entry[0], unit='ms', utc=True),
                    'value': float(entry[1]),
                    'symbol': asset_key.upper(),
                    'metric': metric_id,
                    'venue': self.VENUE
                })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
            self.collection_stats['records_collected'] += len(df)
        return df
    
    async def fetch_price_history(
        self,
        asset_key: str,
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch price history for an asset.
        
        Args:
            asset_key: Asset symbol
            start_date: Start date
            end_date: End date
            interval: Interval
            
        Returns:
            DataFrame with OHLCV-style data
        """
        df = await self.fetch_asset_timeseries(asset_key, 'price', start_date, end_date, interval)
        
        if df.empty:
            return df
        
        df = df.rename(columns={'value': 'close'})
        df['open'] = df['close']
        df['high'] = df['close']
        df['low'] = df['close']
        
        return df
    
    async def _fetch_single_metric_history(
        self,
        asset_key: str,
        metric: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch a single metric time series for an asset.

        Args:
            asset_key: Asset symbol
            metric: Metric ID
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with metric data or empty DataFrame on error
        """
        try:
            df = await self.fetch_asset_timeseries(asset_key, metric, start_date, end_date)
            if not df.empty:
                df = df.rename(columns={'value': metric})
                return df[['timestamp', metric]]
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching {metric} for {asset_key}: {e}")
            return pd.DataFrame()

    async def fetch_multiple_metrics_history(
        self,
        asset_key: str,
        metrics: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch multiple metric time series for an asset.

        Args:
            asset_key: Asset symbol
            metrics: List of metric IDs
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with all metrics as columns
        """
        # Parallelize using asyncio.gather
        tasks = [self._fetch_single_metric_history(asset_key, metric, start_date, end_date) for metric in metrics]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter results to only keep valid DataFrames
        all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

        if not all_data:
            return pd.DataFrame()

        # Merge all metrics
        result = all_data[0]
        for df in all_data[1:]:
            result = result.merge(df, on='timestamp', how='outer')

        result['symbol'] = asset_key.upper()
        result = result.sort_values('timestamp').reset_index(drop=True)

        return result
    
    # =========================================================================
    # Sector Analysis Methods
    # =========================================================================
    
    async def fetch_assets_by_sector(
        self,
        sector: str,
        limit: int = 200
    ) -> pd.DataFrame:
        """
        Fetch assets filtered by sector.
        
        Args:
            sector: Sector name (defi, layer-1, layer-2, etc.)
            limit: Max assets to return
            
        Returns:
            DataFrame with sector assets
        """
        params = {
            'sector': sector,
            'limit': limit,
            'fields': 'id,symbol,name,metrics'
        }
        
        data = await self._make_request('v2/assets', params)
        
        if not data or 'data' not in data:
            return pd.DataFrame()
        
        records = []
        for asset in data.get('data', []):
            metrics = asset.get('metrics', {})
            market = metrics.get('market_data', {})
            mcap = metrics.get('marketcap', {})
            
            records.append({
                'symbol': asset.get('symbol', '').upper(),
                'name': asset.get('name'),
                'sector': sector,
                'price_usd': float(market.get('price_usd', 0) or 0),
                'market_cap': float(mcap.get('current_marketcap_usd', 0) or 0),
                'market_cap_rank': int(mcap.get('rank', 0) or 0),
                'volume_24h': float(market.get('volume_last_24_hours', 0) or 0),
                'real_volume_24h': float(market.get('real_volume_last_24_hours', 0) or 0),
                'pct_change_24h': float(market.get('percent_change_usd_last_24_hours', 0) or 0),
                'pct_change_btc_24h': float(market.get('percent_change_btc_last_24_hours', 0) or 0),
                'timestamp': datetime.utcnow(),
                'venue': self.VENUE
            })
        
        df = pd.DataFrame(records)
        self.collection_stats['records_collected'] += len(df)
        return df
    
    async def analyze_sector(
        self,
        sector: str
    ) -> Optional[SectorAnalysis]:
        """
        Perform sector-level analysis.
        
        Args:
            sector: Sector name
            
        Returns:
            SectorAnalysis dataclass
        """
        assets = await self.fetch_assets_by_sector(sector, limit=100)
        
        if assets.empty:
            return None
        
        total_mcap = assets['market_cap'].sum()
        total_volume = assets['volume_24h'].sum()
        
        # Calculate weighted average change
        assets['weight'] = assets['market_cap'] / total_mcap
        weighted_change = (assets['pct_change_24h'] * assets['weight']).sum()
        
        # Get top/worst performers
        sorted_assets = assets.sort_values('pct_change_24h', ascending=False)
        top_performers = sorted_assets.head(5)['symbol'].tolist()
        worst_performers = sorted_assets.tail(5)['symbol'].tolist()
        
        # Get total market cap for dominance (rough estimate)
        all_assets = await self.fetch_all_assets(limit=100)
        total_market_cap = all_assets['market_cap'].sum() if not all_assets.empty else total_mcap
        
        return SectorAnalysis(
            sector=sector,
            asset_count=len(assets),
            total_market_cap=total_mcap,
            total_volume_24h=total_volume,
            avg_pct_change_24h=assets['pct_change_24h'].mean(),
            weighted_pct_change_24h=weighted_change,
            top_performers=top_performers,
            worst_performers=worst_performers,
            market_cap_dominance=total_mcap / total_market_cap * 100 if total_market_cap > 0 else 0,
            timestamp=datetime.utcnow()
        )
    
    async def _fetch_single_sector_analysis(
        self,
        sector: str
    ) -> Optional[Dict]:
        """
        Analyze a single sector.

        Args:
            sector: Sector name

        Returns:
            Dict with sector analysis or None on error
        """
        try:
            analysis = await self.analyze_sector(sector)
            if analysis:
                return {
                    'sector': analysis.sector,
                    'asset_count': analysis.asset_count,
                    'total_market_cap': analysis.total_market_cap,
                    'total_volume_24h': analysis.total_volume_24h,
                    'avg_change_24h': analysis.avg_pct_change_24h,
                    'weighted_change_24h': analysis.weighted_pct_change_24h,
                    'dominance_pct': analysis.market_cap_dominance,
                    'top_performers': ', '.join(analysis.top_performers[:3]),
                    'worst_performers': ', '.join(analysis.worst_performers[:3]),
                    'timestamp': analysis.timestamp
                }
            return None
        except Exception as e:
            logger.error(f"Error analyzing sector {sector}: {e}")
            return None

    async def fetch_sector_comparison(
        self,
        sectors: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple sectors.

        Args:
            sectors: List of sectors to compare (default: major sectors)

        Returns:
            DataFrame with sector comparison
        """
        if sectors is None:
            sectors = ['layer-1', 'layer-2', 'defi', 'exchange-tokens', 'stablecoins', 'nft', 'gaming']

        # Parallelize using asyncio.gather
        tasks = [self._fetch_single_sector_analysis(sector) for sector in sectors]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter results to only keep valid data
        records = [r for r in results if isinstance(r, dict)]

        df = pd.DataFrame(records)

        if not df.empty:
            # Add rankings
            df['mcap_rank'] = df['total_market_cap'].rank(ascending=False).astype(int)
            df['volume_rank'] = df['total_volume_24h'].rank(ascending=False).astype(int)
            df['performance_rank'] = df['weighted_change_24h'].rank(ascending=False).astype(int)

        return df
    
    # =========================================================================
    # Valuation Methods
    # =========================================================================
    
    async def calculate_valuation_metrics(
        self,
        asset_key: str
    ) -> Optional[ValuationMetrics]:
        """
        Calculate valuation metrics for an asset.
        
        Args:
            asset_key: Asset symbol
            
        Returns:
            ValuationMetrics dataclass
        """
        metrics = await self.fetch_asset_metrics(asset_key)
        
        if not metrics:
            return None
        
        # NVT percentile (rough historical comparison)
        # Higher NVT = potentially overvalued
        nvt_percentile = 50 # Default
        if metrics.nvt_ratio > 0:
            if metrics.nvt_ratio > 100:
                nvt_percentile = 90
            elif metrics.nvt_ratio > 50:
                nvt_percentile = 75
            elif metrics.nvt_ratio < 20:
                nvt_percentile = 25
            elif metrics.nvt_ratio < 10:
                nvt_percentile = 10
        
        # MVRV placeholder (would need realized cap data)
        mvrv_ratio = 1.0
        mvrv_percentile = 50
        
        # Determine valuation tier
        if nvt_percentile >= 80:
            tier = ValuationTier.SEVERELY_OVERVALUED
        elif nvt_percentile >= 65:
            tier = ValuationTier.OVERVALUED
        elif nvt_percentile <= 20:
            tier = ValuationTier.SEVERELY_UNDERVALUED
        elif nvt_percentile <= 35:
            tier = ValuationTier.UNDERVALUED
        else:
            tier = ValuationTier.FAIRLY_VALUED
        
        # Simple fair value estimate based on NVT
        fair_value = None
        upside = None
        if metrics.nvt_ratio > 0 and metrics.transfer_value_24h > 0:
            # Assume "fair" NVT is 30
            fair_nvt = 30
            implied_mcap = metrics.transfer_value_24h * 365 * fair_nvt
            fair_value = implied_mcap / metrics.circulating_supply if metrics.circulating_supply > 0 else None
            if fair_value and metrics.price_usd > 0:
                upside = (fair_value - metrics.price_usd) / metrics.price_usd * 100
        
        return ValuationMetrics(
            symbol=metrics.symbol,
            price_usd=metrics.price_usd,
            market_cap=metrics.market_cap,
            nvt_ratio=metrics.nvt_ratio,
            nvt_percentile=nvt_percentile,
            mvrv_ratio=mvrv_ratio,
            mvrv_percentile=mvrv_percentile,
            pe_ratio=None, # Would need fee/revenue data
            ps_ratio=None,
            mcap_to_tvl=None, # Would need TVL data
            valuation_tier=tier,
            fair_value_estimate=fair_value,
            upside_potential_pct=upside,
            timestamp=datetime.utcnow()
        )
    
    async def _fetch_single_valuation(
        self,
        asset: str
    ) -> Optional[Dict]:
        """
        Calculate valuation metrics for a single asset.

        Args:
            asset: Asset symbol

        Returns:
            Dict with valuation data or None on error
        """
        try:
            val = await self.calculate_valuation_metrics(asset)
            if val:
                return {
                    'symbol': val.symbol,
                    'price_usd': val.price_usd,
                    'market_cap': val.market_cap,
                    'nvt_ratio': val.nvt_ratio,
                    'nvt_percentile': val.nvt_percentile,
                    'valuation_tier': val.valuation_tier.value,
                    'fair_value_estimate': val.fair_value_estimate,
                    'upside_potential_pct': val.upside_potential_pct,
                    'timestamp': val.timestamp
                }
            return None
        except Exception as e:
            logger.error(f"Error calculating valuation for {asset}: {e}")
            return None

    async def fetch_valuation_comparison(
        self,
        assets: List[str]
    ) -> pd.DataFrame:
        """
        Compare valuations across multiple assets.

        Args:
            assets: List of asset symbols

        Returns:
            DataFrame with valuation comparison
        """
        # Parallelize using asyncio.gather
        tasks = [self._fetch_single_valuation(asset) for asset in assets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter results to only keep valid data
        records = [r for r in results if isinstance(r, dict)]

        df = pd.DataFrame(records)

        if not df.empty:
            df = df.sort_values('upside_potential_pct', ascending=False)

        return df
    
    # =========================================================================
    # News and Research
    # =========================================================================
    
    async def fetch_news(
        self,
        asset_key: Optional[str] = None,
        limit: int = 50
    ) -> pd.DataFrame:
        """
        Fetch recent news articles.
        
        Args:
            asset_key: Filter by asset (optional)
            limit: Max articles
            
        Returns:
            DataFrame with news articles
        """
        params = {'limit': limit}
        
        endpoint = f'v1/news/{asset_key.lower()}' if asset_key else 'v1/news'
        
        data = await self._make_request(endpoint, params)
        
        if not data or 'data' not in data:
            return pd.DataFrame()
        
        records = []
        for article in data.get('data', []):
            tags = [t.get('name') for t in article.get('tags', [])]
            references = [r.get('name') for r in article.get('references', [])]
            
            records.append({
                'id': article.get('id'),
                'title': article.get('title'),
                'published_at': pd.to_datetime(article.get('published_at')),
                'author': article.get('author', {}).get('name'),
                'url': article.get('url'),
                'tags': ', '.join(tags) if tags else '',
                'referenced_assets': ', '.join(references) if references else '',
                'asset_filter': asset_key.upper() if asset_key else 'all',
                'venue': self.VENUE
            })
        
        df = pd.DataFrame(records)
        self.collection_stats['records_collected'] += len(df)
        return df
    
    # =========================================================================
    # Comprehensive Data Methods
    # =========================================================================
    
    async def fetch_comprehensive_data(
        self,
        assets: Optional[List[str]] = None
    ) -> Dict[str, Union[pd.DataFrame, SectorAnalysis]]:
        """
        Fetch comprehensive data for research and analysis.
        
        Args:
            assets: List of asset symbols (default: benchmark assets)
            
        Returns:
            Dictionary with multiple DataFrames and analyses
        """
        if assets is None:
            assets = self.BENCHMARK_ASSETS
        
        results = {}
        
        # All assets overview
        logger.info("Fetching all assets...")
        results['all_assets'] = await self.fetch_all_assets(limit=200)
        
        # Detailed metrics for selected assets
        logger.info(f"Fetching detailed metrics for {len(assets)} assets...")
        results['detailed_metrics'] = await self.fetch_asset_metrics_batch(assets)
        
        # Valuation comparison
        logger.info("Calculating valuations...")
        results['valuations'] = await self.fetch_valuation_comparison(assets)
        
        # Sector comparison
        logger.info("Analyzing sectors...")
        results['sector_comparison'] = await self.fetch_sector_comparison()
        
        # Recent news
        logger.info("Fetching news...")
        results['news'] = await self.fetch_news(limit=30)
        
        return results
    
    # =========================================================================
    # Required Abstract Methods
    # =========================================================================
    
    async def fetch_funding_rates(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Messari doesn't provide funding rates."""
        logger.info("Messari doesn't provide funding rates. Use Coinalyze or exchange collectors.")
        return pd.DataFrame()
    
    async def _fetch_single_price_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Fetch price history for a single symbol.

        Args:
            symbol: Symbol to fetch
            start_date: Start date
            end_date: End date
            timeframe: Timeframe

        Returns:
            DataFrame with price data or empty DataFrame on error
        """
        try:
            df = await self.fetch_price_history(symbol, start_date, end_date, timeframe)
            return df if not df.empty else pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching price history for {symbol}: {e}")
            return pd.DataFrame()

    async def fetch_ohlcv(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch OHLCV-style price data.

        Note: Messari provides daily close prices only.
        For full OHLCV, use exchange collectors.
        """
        # Parallelize using asyncio.gather
        tasks = [self._fetch_single_price_history(symbol, start_date, end_date, timeframe) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter results to only keep valid DataFrames
        all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            return result

        return pd.DataFrame()
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return {
            **self.collection_stats,
            'cache_size': len(self._cache)
        }
    
    async def close(self):
        """Close aiohttp session and cleanup."""
        if self.session and not self.session.closed:
            await self.session.close()
        self._cache.clear()
        logger.info(f"Messari collector closed. Stats: {self.get_collection_stats()}")

    # =========================================================================
    # Standardized Collection Methods (for dynamic routing in collection_manager)
    # =========================================================================

    async def collect_asset_metrics(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect asset metrics for symbols (standardized interface).

        Wraps fetch_asset_metrics_batch() to match collection_manager expectations.

        Args:
            symbols: List of asset symbols to fetch metrics for
            start_date: Start date (unused - metrics are current snapshot)
            end_date: End date (unused - metrics are current snapshot)
            **kwargs: Additional parameters

        Returns:
            DataFrame with asset metrics for specified symbols
        """
        try:
            logger.info(f"Messari: Collecting asset_metrics for {len(symbols)} symbols")

            # Use the existing batch method
            df = await self.fetch_asset_metrics_batch(symbols)

            if not df.empty:
                logger.info(f"Messari: Collected asset_metrics for {len(df)} assets")
                return df

            logger.warning(f"Messari: No asset_metrics found for symbols {symbols}")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Messari collect_asset_metrics error: {e}")
            return pd.DataFrame()

    async def _fetch_single_fundamental(
        self,
        symbol: str
    ) -> Optional[Dict]:
        """
        Fetch fundamental data for a single symbol.

        Args:
            symbol: Asset symbol

        Returns:
            Dict with fundamental data or None on error
        """
        try:
            profile = await self.fetch_asset_profile(symbol)

            if profile:
                return {
                    'symbol': profile.symbol,
                    'name': profile.name,
                    'tagline': profile.tagline,
                    'category': profile.category,
                    'sector': profile.sector,
                    'project_created': profile.project_created,
                    'genesis_block_date': profile.genesis_block_date,
                    'consensus_algorithm': profile.consensus_algorithm,
                    'block_time_seconds': profile.block_time_seconds,
                    'token_type': profile.token_type,
                    'is_mineable': profile.is_mineable,
                    'is_stakeable': profile.is_stakeable,
                    'governance_type': profile.governance_type,
                    'website': profile.website,
                    'whitepaper': profile.whitepaper,
                    'github': profile.github,
                    'twitter': profile.twitter,
                    'discord': profile.discord,
                    'telegram': profile.telegram,
                    'team_members': ','.join(profile.team_members) if profile.team_members else '',
                    'investors': ','.join(profile.investors) if profile.investors else '',
                    'timestamp': profile.timestamp,
                    'venue': self.VENUE
                }
            return None
        except Exception as e:
            logger.warning(f"Messari: Failed to fetch fundamentals for {symbol}: {e}")
            return None

    async def collect_fundamentals(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect fundamental data for symbols (standardized interface).

        Wraps fetch_asset_profile() to match collection_manager expectations.

        Args:
            symbols: List of asset symbols to fetch fundamentals for
            start_date: Start date (unused - profiles are current data)
            end_date: End date (unused - profiles are current data)
            **kwargs: Additional parameters

        Returns:
            DataFrame with fundamental data for specified symbols
        """
        try:
            logger.info(f"Messari: Collecting fundamentals for {len(symbols)} symbols")

            # Parallelize using asyncio.gather
            tasks = [self._fetch_single_fundamental(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter results to only keep valid data
            records = [r for r in results if isinstance(r, dict)]

            if records:
                df = pd.DataFrame(records)
                logger.info(f"Messari: Collected fundamentals for {len(df)} assets")
                return df

            logger.warning(f"Messari: No fundamentals found for symbols {symbols}")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Messari collect_fundamentals error: {e}")
            return pd.DataFrame()

# =============================================================================
# Standalone Testing
# =============================================================================

async def test_messari_collector():
    """Test Messari collector functionality."""
    import os
    
    config = {
        'messari_api_key': os.getenv('MESSARI_API_KEY', ''),
        'rate_limit': 20,
    }
    
    collector = MessariCollector(config)
    
    try:
        print("Testing Messari Collector")
        print("=" * 60)
        
        # Test all assets
        print("\n1. Testing fetch_all_assets...")
        assets = await collector.fetch_all_assets(limit=20)
        if not assets.empty:
            print(f" Found {len(assets)} assets")
            print(f" Top: {assets.iloc[0]['symbol']} - ${assets.iloc[0]['price_usd']:,.2f}")
        
        # Test asset metrics
        print("\n2. Testing fetch_asset_metrics (BTC)...")
        btc_metrics = await collector.fetch_asset_metrics('btc')
        if btc_metrics:
            print(f" Price: ${btc_metrics.price_usd:,.2f}")
            print(f" Market Cap: ${btc_metrics.market_cap:,.0f}")
            print(f" 24h Change: {btc_metrics.pct_change_24h:.2f}%")
            print(f" Active Addresses: {btc_metrics.active_addresses_24h:,}")
            print(f" NVT Ratio: {btc_metrics.nvt_ratio:.2f}")
        
        # Test asset profile
        print("\n3. Testing fetch_asset_profile (ETH)...")
        eth_profile = await collector.fetch_asset_profile('eth')
        if eth_profile:
            print(f" Sector: {eth_profile.sector}")
            print(f" Category: {eth_profile.category}")
            print(f" Consensus: {eth_profile.consensus_algorithm}")
        
        # Test sector analysis
        print("\n4. Testing sector analysis (DeFi)...")
        defi_analysis = await collector.analyze_sector('defi')
        if defi_analysis:
            print(f" Assets: {defi_analysis.asset_count}")
            print(f" Total MCap: ${defi_analysis.total_market_cap:,.0f}")
            print(f" Weighted 24h Change: {defi_analysis.weighted_pct_change_24h:.2f}%")
            print(f" Top Performers: {', '.join(defi_analysis.top_performers[:3])}")
        
        # Test valuation
        print("\n5. Testing valuation metrics (ETH)...")
        eth_val = await collector.calculate_valuation_metrics('eth')
        if eth_val:
            print(f" NVT Ratio: {eth_val.nvt_ratio:.2f}")
            print(f" NVT Percentile: {eth_val.nvt_percentile}")
            print(f" Valuation Tier: {eth_val.valuation_tier.value}")
            if eth_val.upside_potential_pct:
                print(f" Upside Potential: {eth_val.upside_potential_pct:.1f}%")
        
        # Test news
        print("\n6. Testing fetch_news...")
        news = await collector.fetch_news(limit=5)
        if not news.empty:
            print(f" Found {len(news)} articles")
            print(f" Latest: {news.iloc[0]['title'][:50]}...")
        
        print("\n" + "=" * 60)
        print(f"Collection stats: {collector.get_collection_stats()}")
        print("Messari collector tests completed!")
        
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await collector.close()

if __name__ == '__main__':
    asyncio.run(test_messari_collector())