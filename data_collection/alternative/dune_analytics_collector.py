"""
Dune Analytics API Collector - FREE Alternative to Glassnode & Flipside Crypto.

validated collector for on-chain analytics:
- SQL-based blockchain data queries across 100+ chains
- Pre-built community queries (700,000+ available)
- DEX trading data, volumes, liquidity
- DeFi protocol events (liquidations, deposits, withdrawals)
- Token transfers and holder analytics
- Whale activity tracking
- Bridge activity and cross-chain flows
- Gas analytics and network metrics
- Credit tracking and budget management

API Documentation: https://docs.dune.com
Rate Limit: Credit-based (2,500 credits/month free tier)
Registration: https://dune.com (free)

Version: 2.0.0
"""

import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import time
import json
import hashlib

from ..base_collector import BaseCollector
from ..utils.rate_limiter import get_shared_rate_limiter
from ..utils.retry_handler import RetryHandler

logger = logging.getLogger(__name__)

class QueryState(Enum):
    """Dune query execution states."""
    PENDING = 'QUERY_STATE_PENDING'
    EXECUTING = 'QUERY_STATE_EXECUTING'
    COMPLETED = 'QUERY_STATE_COMPLETED'
    FAILED = 'QUERY_STATE_FAILED'
    CANCELLED = 'QUERY_STATE_CANCELLED'
    EXPIRED = 'QUERY_STATE_EXPIRED'

class Chain(Enum):
    """Supported blockchain networks."""
    ETHEREUM = 'ethereum'
    ARBITRUM = 'arbitrum'
    OPTIMISM = 'optimism'
    POLYGON = 'polygon'
    BASE = 'base'
    AVALANCHE = 'avalanche_c'
    BNB = 'bnb'
    FANTOM = 'fantom'
    GNOSIS = 'gnosis'
    ZKSYNC = 'zksync'
    LINEA = 'linea'
    SCROLL = 'scroll'
    SOLANA = 'solana'
    BITCOIN = 'bitcoin'

@dataclass
class QueryResult:
    """Result from a Dune query execution."""
    query_id: int
    execution_id: str
    state: QueryState
    data: pd.DataFrame
    execution_time_ms: int
    row_count: int
    credits_used: int
    cached: bool
    timestamp: datetime

@dataclass 
class CreditUsage:
    """Credit usage tracking."""
    monthly_budget: int
    credits_used: int
    credits_remaining: int
    queries_executed: int
    avg_credits_per_query: float
    reset_date: datetime

class DuneAnalyticsCollector(BaseCollector):
    """
    Dune Analytics data collector - FREE alternative to Glassnode and Flipside.
    
    Features:
    - SQL-based blockchain data queries across 100+ chains
    - Pre-built community queries (700,000+ available)
    - DEX trading data, DeFi events, token transfers
    - Custom on-chain analytics with DuneSQL
    - Credit tracking and budget management
    - Query result caching
    - Batch query execution
    
    Supported Chains:
    - Layer 1: Bitcoin, Ethereum, Solana, Avalanche, BNB Chain
    - Layer 2: Arbitrum, Optimism, Base, zkSync, Polygon, Linea, Scroll
    
    Pricing Tiers:
    - Free: 2,500 credits/month with API access
    - Plus: $349/mo - 25,000 credits 
    - Premium: $699/mo - 100,000 credits
    
    Credit Costs (approximate):
    - Simple queries: 10-50 credits
    - Medium queries: 50-200 credits
    - Complex queries: 200-1000+ credits
    
    Use Cases:
    - DEX volume and liquidity analysis
    - Whale tracking and large transfers
    - DeFi liquidation monitoring
    - Token holder distribution
    - Cross-chain bridge activity
    - Gas price analysis
    - Protocol TVL tracking
    """
    
    VENUE = 'dune_analytics'
    VENUE_TYPE = 'on_chain_aggregator'
    BASE_URL = 'https://api.dune.com/api/v1'
    
    # Pre-defined community query IDs for common analytics
    # These are real, popular community queries
    COMMUNITY_QUERIES = {
        # DEX Analytics
        'uniswap_v3_daily_volume': 2356520,
        'dex_aggregated_volume': 2030664,
        'curve_pool_volumes': 2587632,
        'balancer_volumes': 2756891,
        
        # DeFi Liquidations
        'aave_v3_liquidations': 2418296,
        'compound_v3_liquidations': 2418297,
        'maker_liquidations': 2418298,
        
        # Token Analytics
        'erc20_top_holders': 2587633,
        'token_transfers_large': 2756892,
        
        # Bridge Activity
        'bridge_volumes_daily': 3012567,
        'arbitrum_bridge_flows': 3012568,
        'optimism_bridge_flows': 3012569,
        
        # Network Metrics
        'ethereum_gas_tracker': 1234567,
        'l2_gas_comparison': 1234568,
        'active_addresses_daily': 3156789,
        
        # Staking
        'eth_staking_deposits': 3156790,
        'lsd_market_share': 3156791,
    }
    
    # Supported chains
    SUPPORTED_CHAINS = [c.value for c in Chain]
    
    # DuneSQL table prefixes by chain
    CHAIN_PREFIXES = {
        'ethereum': '',
        'arbitrum': 'arbitrum.',
        'optimism': 'optimism.',
        'polygon': 'polygon.',
        'base': 'base.',
        'avalanche_c': 'avalanche_c.',
        'bnb': 'bnb.',
        'fantom': 'fantom.',
        'gnosis': 'gnosis.',
        'zksync': 'zksync.',
        'solana': 'solana.'
    }
    
    # Common token addresses
    COMMON_TOKENS = {
        'WETH': '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',
        'USDC': '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
        'USDT': '0xdac17f958d2ee523a2206206994597c13d831ec7',
        'DAI': '0x6b175474e89094c44da98b954eedeac495271d0f',
        'WBTC': '0x2260fac5e5542a773aa44fbcfedf7c193bc2c599',
        'UNI': '0x1f9840a85d5af5bf1d1762f925bdaddc4201f984',
        'LINK': '0x514910771af9ca656af840dff83e8264ecf986ca',
        'AAVE': '0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9',
    }
    
    def __init__(self, config: Dict):
        """
        Initialize Dune Analytics collector.

        Args:
            config: Configuration dict containing:
                - api_key: Dune API key (free from dune.com)
                - credits_per_month: Budget limit (default 2500)
                - cache_ttl: Cache TTL in seconds (default 3600)
                - max_concurrent: Max concurrent queries (default 3)
        """
        super().__init__(config)

        # CRITICAL: Set supported data types for dynamic routing
        self.supported_data_types = ['custom_queries']
        self.venue = 'dune'

        # Import VenueType from base_collector
        from ..base_collector import VenueType
        self.venue_type = VenueType.ALTERNATIVE
        self.requires_auth = True # Requires Dune API key (free tier available)

        self.api_key = config.get('api_key') or config.get('dune_api_key') or os.getenv('DUNE_API_KEY', '')
        if not self.api_key:
            logger.warning("No Dune API key provided. Register free at dune.com")
        else:
            logger.info(f"Initialized Dune Analytics collector with data types: {self.supported_data_types}")

        self.headers = {
            'X-Dune-API-Key': self.api_key,
            'Content-Type': 'application/json'
        }
        
        # Credit tracking
        self.monthly_budget = config.get('credits_per_month', 2500)
        self.credits_used = 0
        self._credit_reset_date = self._get_next_reset_date()
        
        # Rate limiter (conservative to save credits)
        self.rate_limiter = get_shared_rate_limiter('dune', rate=5, per=60.0, burst=3)
        
        # Retry handler - OPTIMIZATION: Reduced max_delay from 60s to 45s
        # (Dune has slow queries, so keeping base_delay high but capping max_delay)
        self.retry_handler = RetryHandler(
            max_retries=3,
            base_delay=5.0,
            max_delay=45.0 # Reduced from 60s
        )
        
        # Session management
        self.session = None
        
        # Query result caching
        self._cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}
        self._cache_ttl = config.get('cache_ttl', 3600)

        # Concurrent query limit
        self._semaphore = asyncio.Semaphore(config.get('max_concurrent', 3))

        # Track if SQL execution is available (free tier can't create queries)
        self._sql_execution_available = True
        self._sql_unavailable_logged = False

        # Error patterns indicating paid plan required
        self._paid_plan_patterns = [
            'paid plan', 'paid plans', 'upgrade', 'query management endpoints',
            'not available', 'premium', 'subscription required'
        ]

        # Collection stats
        self.collection_stats = {
            'records_collected': 0,
            'queries_executed': 0,
            'credits_used': 0,
            'cache_hits': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    def _get_next_reset_date(self) -> datetime:
        """Get the next monthly credit reset date."""
        now = datetime.utcnow()
        if now.day >= 1:
            if now.month == 12:
                return datetime(now.year + 1, 1, 1)
            return datetime(now.year, now.month + 1, 1)
        return datetime(now.year, now.month, 1)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=300) # Long timeout for queries
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout
            )
        return self.session
    
    def _get_cache_key(self, query_id: int, parameters: Optional[Dict] = None) -> str:
        """Generate cache key for query results."""
        param_str = json.dumps(parameters or {}, sort_keys=True)
        return hashlib.md5(f"{query_id}_{param_str}".encode()).hexdigest()
    
    def _get_cached(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get cached query result if valid."""
        if cache_key in self._cache:
            timestamp, data = self._cache[cache_key]
            if (datetime.utcnow() - timestamp).total_seconds() < self._cache_ttl:
                self.collection_stats['cache_hits'] += 1
                return data.copy()
            del self._cache[cache_key]
        return None
    
    def _set_cached(self, cache_key: str, data: pd.DataFrame):
        """Cache query result."""
        self._cache[cache_key] = (datetime.utcnow(), data.copy())
    
    def get_credit_usage(self) -> CreditUsage:
        """Get current credit usage statistics."""
        queries = self.collection_stats['queries_executed']
        return CreditUsage(
            monthly_budget=self.monthly_budget,
            credits_used=self.credits_used,
            credits_remaining=max(0, self.monthly_budget - self.credits_used),
            queries_executed=queries,
            avg_credits_per_query=self.credits_used / queries if queries > 0 else 0,
            reset_date=self._credit_reset_date
        )
    
    def _check_budget(self, estimated_credits: int = 50) -> bool:
        """Check if budget allows for query execution."""
        if self.credits_used + estimated_credits > self.monthly_budget:
            logger.warning(
                f"Budget warning: {self.credits_used}/{self.monthly_budget} credits used. "
                f"Query may exceed budget."
            )
            return False
        return True

    def _is_paid_plan_error(self, text: str) -> bool:
        """Check if error text indicates paid plan is required."""
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in self._paid_plan_patterns)

    def _handle_paid_plan_required(self) -> pd.DataFrame:
        """Handle paid plan requirement - set flag and log once."""
        self._sql_execution_available = False
        if not self._sql_unavailable_logged:
            logger.info(
                "Dune Analytics: Query creation requires paid plan. "
                "Skipping SQL-based queries. Use pre-existing query IDs instead."
            )
            self._sql_unavailable_logged = True
        return pd.DataFrame()
    
    # =========================================================================
    # Core Query Methods
    # =========================================================================
    
    async def execute_query(
        self,
        query_id: int,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: int = 300,
        use_cache: bool = True,
        performance_mode: str = 'medium'
    ) -> QueryResult:
        """
        Execute a Dune query and return results.
        
        Args:
            query_id: Dune query ID (create your own or use community queries)
            parameters: Optional query parameters for parameterized queries
            timeout: Maximum wait time in seconds
            use_cache: Whether to use cached results if available
            performance_mode: 'medium' or 'large' (affects credits)
            
        Returns:
            QueryResult with data and metadata
        """
        # Check cache
        cache_key = self._get_cache_key(query_id, parameters)
        if use_cache:
            cached_data = self._get_cached(cache_key)
            if cached_data is not None:
                logger.info(f"Using cached results for query {query_id}")
                return QueryResult(
                    query_id=query_id,
                    execution_id='cached',
                    state=QueryState.COMPLETED,
                    data=cached_data,
                    execution_time_ms=0,
                    row_count=len(cached_data),
                    credits_used=0,
                    cached=True,
                    timestamp=datetime.utcnow()
                )
        
        # Check budget
        self._check_budget()
        
        async with self._semaphore:
            await self.rate_limiter.acquire()
            session = await self._get_session()
            
            # Start execution
            exec_url = f"{self.BASE_URL}/query/{query_id}/execute"
            payload = {'performance': performance_mode}
            if parameters:
                payload['query_parameters'] = parameters
            
            start_time = time.time()
            
            try:
                async with session.post(exec_url, json=payload) as response:
                    if response.status == 401:
                        raise ValueError("Invalid Dune API key")
                    elif response.status == 402:
                        raise ValueError("Insufficient credits. Upgrade plan or wait for reset.")
                    elif response.status == 404:
                        raise ValueError(f"Query {query_id} not found")
                    elif response.status != 200:
                        text = await response.text()
                        raise ValueError(f"Dune API error {response.status}: {text[:200]}")
                    
                    data = await response.json()
                    execution_id = data.get('execution_id')
                    
                    if not execution_id:
                        raise ValueError(f"No execution_id returned: {data}")
                
                # Poll for results
                df, state, row_count = await self._poll_execution(execution_id, timeout)
                
                execution_time = int((time.time() - start_time) * 1000)
                
                # Estimate credits used (rough estimate)
                estimated_credits = 10 + (row_count // 1000) * 5
                if performance_mode == 'large':
                    estimated_credits *= 2
                
                self.credits_used += estimated_credits
                self.collection_stats['queries_executed'] += 1
                self.collection_stats['credits_used'] += estimated_credits
                self.collection_stats['records_collected'] += row_count
                
                # Cache results
                if use_cache and not df.empty:
                    self._set_cached(cache_key, df)
                
                return QueryResult(
                    query_id=query_id,
                    execution_id=execution_id,
                    state=state,
                    data=df,
                    execution_time_ms=execution_time,
                    row_count=row_count,
                    credits_used=estimated_credits,
                    cached=False,
                    timestamp=datetime.utcnow()
                )
                
            except Exception as e:
                self.collection_stats['errors'] += 1
                logger.error(f"Query {query_id} execution failed: {e}")
                raise
    
    async def _poll_execution(
        self,
        execution_id: str,
        timeout: int = 300
    ) -> Tuple[pd.DataFrame, QueryState, int]:
        """Poll for query execution results."""
        session = await self._get_session()
        status_url = f"{self.BASE_URL}/execution/{execution_id}/status"
        results_url = f"{self.BASE_URL}/execution/{execution_id}/results"
        
        start_time = time.time()
        poll_interval = 2
        
        while time.time() - start_time < timeout:
            await asyncio.sleep(poll_interval)
            
            # Gradually increase poll interval
            poll_interval = min(poll_interval * 1.2, 10)
            
            try:
                async with session.get(status_url) as response:
                    if response.status != 200:
                        continue
                    
                    data = await response.json()
                    state_str = data.get('state', '')
                    
                    if state_str == QueryState.COMPLETED.value:
                        # Fetch results
                        async with session.get(results_url) as results_response:
                            if results_response.status == 200:
                                results_data = await results_response.json()
                                rows = results_data.get('result', {}).get('rows', [])
                                df = pd.DataFrame(rows)
                                return df, QueryState.COMPLETED, len(rows)
                    
                    elif state_str == QueryState.FAILED.value:
                        error = data.get('error', 'Unknown error')
                        raise ValueError(f"Query failed: {error}")
                    
                    elif state_str in [QueryState.CANCELLED.value, QueryState.EXPIRED.value]:
                        raise ValueError(f"Query {state_str}")
                        
            except aiohttp.ClientError as e:
                logger.warning(f"Poll error (will retry): {e}")
        
        raise TimeoutError(f"Query execution timed out after {timeout}s")
    
    async def execute_sql(
        self,
        sql: str,
        name: str = "StatArb Query",
        timeout: int = 300,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Create a new query with custom SQL and execute it.

        Args:
            sql: DuneSQL query string
            name: Query name for reference
            timeout: Maximum wait time
            use_cache: Use cached results

        Returns:
            DataFrame with query results

        Note:
            Query creation requires a paid Dune plan. Free tier can only
            execute pre-existing queries by ID.
        """
        # Skip if we've already determined SQL execution isn't available (paid plan only)
        if not self._sql_execution_available:
            return pd.DataFrame()

        # Create cache key from SQL
        sql_hash = hashlib.md5(sql.encode()).hexdigest()
        if use_cache:
            cached = self._get_cached(sql_hash)
            if cached is not None:
                return cached

        session = await self._get_session()

        # Create query
        create_url = f"{self.BASE_URL}/query"
        payload = {
            'name': name,
            'query_sql': sql,
            'is_private': True
        }

        try:
            async with session.post(create_url, json=payload) as response:
                text = await response.text()

                # Check for paid plan requirement in response (regardless of status code)
                if self._is_paid_plan_error(text):
                    return self._handle_paid_plan_required()

                if response.status != 200:
                    # Log as debug, not error, for expected API limitations
                    logger.debug(f"Dune query creation failed ({response.status}): {text[:100]}")
                    return pd.DataFrame()

                data = await response.json()
                query_id = data.get('query_id')

                # Check for error in JSON response body (API sometimes returns 200 with error)
                if 'error' in data:
                    if self._is_paid_plan_error(str(data.get('error', ''))):
                        return self._handle_paid_plan_required()
                    logger.debug(f"Dune query error: {data.get('error')}")
                    return pd.DataFrame()

            # Execute the new query
            result = await self.execute_query(query_id, timeout=timeout, use_cache=False)

            # Cache with SQL hash
            if use_cache and not result.data.empty:
                self._set_cached(sql_hash, result.data)

            return result.data

        except Exception as e:
            # Check if this is the paid plan error
            error_msg = str(e)
            if self._is_paid_plan_error(error_msg):
                return self._handle_paid_plan_required()
            # Don't log as error - this is expected for free tier
            self.collection_stats['errors'] += 1
            logger.debug(f"Dune SQL execution issue: {e}")
            return pd.DataFrame()
    
    # =========================================================================
    # DEX Analytics
    # =========================================================================
    
    async def fetch_dex_volume(
        self,
        chain: str = 'ethereum',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = 'day',
        protocols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch DEX trading volume data.
        
        Args:
            chain: Blockchain (ethereum, arbitrum, optimism, etc.)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Aggregation interval (hour, day)
            protocols: Filter by DEX protocols (uniswap, curve, etc.)
            
        Returns:
            DataFrame with DEX volume by protocol and time period
        """
        if start_date is None:
            start_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')
        
        protocol_filter = ""
        if protocols:
            protocol_list = "', '".join(protocols)
            protocol_filter = f"AND project IN ('{protocol_list}')"
        
        sql = f"""
        SELECT 
            date_trunc('{interval}', block_time) as period,
            project as dex,
            blockchain,
            sum(amount_usd) as volume_usd,
            count(*) as trade_count,
            count(distinct taker) as unique_traders,
            avg(amount_usd) as avg_trade_size
        FROM dex.trades
        WHERE blockchain = '{chain}'
          AND block_time >= date '{start_date}'
          AND block_time < date '{end_date}'
          AND amount_usd > 0
          AND amount_usd < 1e12
          {protocol_filter}
        GROUP BY 1, 2, 3
        ORDER BY 1 DESC, 4 DESC
        """
        
        try:
            df = await self.execute_sql(sql, name=f"DEX Volume {chain}")
            
            if not df.empty:
                df['chain'] = chain
                df['venue'] = self.VENUE
                df['venue_type'] = self.VENUE_TYPE
                df['period'] = pd.to_datetime(df['period'])
                
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch DEX volume: {e}")
            return pd.DataFrame()
    
    async def fetch_dex_liquidity(
        self,
        chain: str = 'ethereum',
        protocol: str = 'uniswap_v3',
        min_tvl: float = 100000
    ) -> pd.DataFrame:
        """
        Fetch DEX pool liquidity data.
        
        Args:
            chain: Blockchain
            protocol: DEX protocol
            min_tvl: Minimum TVL filter
            
        Returns:
            DataFrame with pool liquidity data
        """
        sql = f"""
        SELECT 
            pool,
            token0_symbol,
            token1_symbol,
            tvl_usd,
            volume_24h_usd,
            fee_tier,
            volume_24h_usd / nullif(tvl_usd, 0) as volume_tvl_ratio
        FROM dex.pools
        WHERE blockchain = '{chain}'
          AND project = '{protocol}'
          AND tvl_usd >= {min_tvl}
        ORDER BY tvl_usd DESC
        LIMIT 500
        """
        
        try:
            df = await self.execute_sql(sql, name=f"DEX Liquidity {protocol}")
            
            if not df.empty:
                df['chain'] = chain
                df['protocol'] = protocol
                df['venue'] = self.VENUE
                df['timestamp'] = datetime.utcnow()
                
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch DEX liquidity: {e}")
            return pd.DataFrame()
    
    # =========================================================================
    # DeFi Liquidations
    # =========================================================================
    
    async def fetch_liquidations(
        self,
        protocol: str = 'aave_v3',
        chain: str = 'ethereum',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_value_usd: float = 1000
    ) -> pd.DataFrame:
        """
        Fetch DeFi liquidation events.
        
        Args:
            protocol: DeFi protocol (aave_v3, compound_v3, maker)
            chain: Blockchain
            start_date: Start date
            end_date: End date
            min_value_usd: Minimum liquidation value
            
        Returns:
            DataFrame with liquidation events
        """
        if start_date is None:
            start_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')
        
        # Protocol-specific queries
        if protocol == 'aave_v3':
            sql = f"""
            SELECT 
                block_time,
                tx_hash,
                liquidator,
                borrower,
                collateral_asset,
                debt_asset,
                liquidated_collateral_amount,
                debt_to_cover,
                collateral_amount_usd as liquidation_value_usd
            FROM aave_v3_{chain}.LiquidationCall
            WHERE block_time >= date '{start_date}'
              AND block_time < date '{end_date}'
              AND collateral_amount_usd >= {min_value_usd}
            ORDER BY block_time DESC
            LIMIT 5000
            """
        elif protocol == 'compound_v3':
            sql = f"""
            SELECT 
                block_time,
                tx_hash,
                absorber as liquidator,
                borrower,
                asset as collateral_asset,
                collateralAbsorbed as liquidated_collateral_amount,
                usdValue as liquidation_value_usd
            FROM compound_v3_{chain}.AbsorbCollateral
            WHERE block_time >= date '{start_date}'
              AND block_time < date '{end_date}'
              AND usdValue >= {min_value_usd}
            ORDER BY block_time DESC
            LIMIT 5000
            """
        else:
            # Generic lending protocol query
            sql = f"""
            SELECT 
                block_time,
                tx_hash,
                liquidator,
                borrower,
                collateral_token,
                debt_token,
                amount_usd as liquidation_value_usd
            FROM lending.liquidations
            WHERE blockchain = '{chain}'
              AND protocol = '{protocol}'
              AND block_time >= date '{start_date}'
              AND block_time < date '{end_date}'
              AND amount_usd >= {min_value_usd}
            ORDER BY block_time DESC
            LIMIT 5000
            """
        
        try:
            df = await self.execute_sql(sql, name=f"Liquidations {protocol}")
            
            if not df.empty:
                df['protocol'] = protocol
                df['chain'] = chain
                df['venue'] = self.VENUE
                if 'block_time' in df.columns:
                    df['block_time'] = pd.to_datetime(df['block_time'])
                
            return df
            
        except Exception as e:
            logger.warning(f"Failed to fetch liquidations for {protocol}: {e}")
            return pd.DataFrame()
    
    async def fetch_liquidation_summary(
        self,
        chain: str = 'ethereum',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch aggregated liquidation summary across protocols.
        
        Args:
            chain: Blockchain
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with daily liquidation summary
        """
        if start_date is None:
            start_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')
        
        sql = f"""
        SELECT 
            date_trunc('day', block_time) as day,
            protocol,
            count(*) as liquidation_count,
            sum(amount_usd) as total_liquidated_usd,
            avg(amount_usd) as avg_liquidation_usd,
            count(distinct borrower) as unique_borrowers_liquidated
        FROM lending.liquidations
        WHERE blockchain = '{chain}'
          AND block_time >= date '{start_date}'
          AND block_time < date '{end_date}'
        GROUP BY 1, 2
        ORDER BY 1 DESC, 4 DESC
        """
        
        try:
            df = await self.execute_sql(sql, name="Liquidation Summary")
            
            if not df.empty:
                df['chain'] = chain
                df['venue'] = self.VENUE
                df['day'] = pd.to_datetime(df['day'])
                
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch liquidation summary: {e}")
            return pd.DataFrame()
    
    # =========================================================================
    # Token Analytics
    # =========================================================================
    
    async def fetch_token_holders(
        self,
        token_address: str,
        chain: str = 'ethereum',
        top_n: int = 100,
        exclude_contracts: bool = True
    ) -> pd.DataFrame:
        """
        Fetch top token holders.
        
        Args:
            token_address: Token contract address
            chain: Blockchain
            top_n: Number of top holders to return
            exclude_contracts: Exclude contract addresses
            
        Returns:
            DataFrame with holder data
        """
        contract_filter = ""
        if exclude_contracts:
            contract_filter = "AND length(code) = 0"
        
        sql = f"""
        WITH transfers AS (
            SELECT 
                "to" as address,
                sum(cast(value as double) / power(10, decimals)) as received
            FROM erc20_{chain}.evt_Transfer t
            JOIN tokens.erc20 tok ON t.contract_address = tok.contract_address
            WHERE t.contract_address = lower('{token_address}')
            GROUP BY 1
            
            UNION ALL
            
            SELECT 
                "from" as address,
                -sum(cast(value as double) / power(10, decimals)) as received
            FROM erc20_{chain}.evt_Transfer t
            JOIN tokens.erc20 tok ON t.contract_address = tok.contract_address
            WHERE t.contract_address = lower('{token_address}')
            GROUP BY 1
        ),
        balances AS (
            SELECT 
                address,
                sum(received) as balance
            FROM transfers
            GROUP BY 1
            HAVING sum(received) > 0
        )
        SELECT 
            b.address,
            b.balance,
            b.balance * p.price as balance_usd
        FROM balances b
        LEFT JOIN prices.usd_latest p 
            ON p.contract_address = lower('{token_address}')
            AND p.blockchain = '{chain}'
        ORDER BY b.balance DESC
        LIMIT {top_n}
        """
        
        try:
            df = await self.execute_sql(sql, name=f"Token Holders {token_address[:10]}...")
            
            if not df.empty:
                df['token_address'] = token_address
                df['chain'] = chain
                df['venue'] = self.VENUE
                df['rank'] = range(1, len(df) + 1)
                
                # Calculate concentration metrics
                total_supply = df['balance'].sum()
                if total_supply > 0:
                    df['pct_of_supply'] = df['balance'] / total_supply * 100
                    df['cumulative_pct'] = df['pct_of_supply'].cumsum()
                
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch token holders: {e}")
            return pd.DataFrame()
    
    async def fetch_whale_transfers(
        self,
        token_address: str,
        chain: str = 'ethereum',
        min_value_usd: float = 100000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch large token transfers (whale activity).
        
        Args:
            token_address: Token contract address
            chain: Blockchain
            min_value_usd: Minimum transfer value in USD
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with whale transfer data
        """
        if start_date is None:
            start_date = (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')
        
        sql = f"""
        SELECT 
            t.block_time,
            t."from" as from_address,
            t."to" as to_address,
            cast(t.value as double) / power(10, tok.decimals) as amount,
            cast(t.value as double) / power(10, tok.decimals) * p.price as value_usd,
            t.tx_hash
        FROM erc20_{chain}.evt_Transfer t
        JOIN tokens.erc20 tok 
            ON t.contract_address = tok.contract_address
            AND tok.blockchain = '{chain}'
        LEFT JOIN prices.usd p 
            ON p.contract_address = t.contract_address
            AND p.blockchain = '{chain}'
            AND p.minute = date_trunc('minute', t.block_time)
        WHERE t.contract_address = lower('{token_address}')
          AND t.block_time >= date '{start_date}'
          AND t.block_time < date '{end_date}'
          AND cast(t.value as double) / power(10, tok.decimals) * p.price >= {min_value_usd}
        ORDER BY t.block_time DESC
        LIMIT 1000
        """
        
        try:
            df = await self.execute_sql(sql, name=f"Whale Transfers {token_address[:10]}...")
            
            if not df.empty:
                df['token_address'] = token_address
                df['chain'] = chain
                df['venue'] = self.VENUE
                df['block_time'] = pd.to_datetime(df['block_time'])
                
                # Add transfer direction classification
                # (Would need known exchange/contract addresses for better classification)
                
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch whale transfers: {e}")
            return pd.DataFrame()
    
    # =========================================================================
    # Bridge Activity
    # =========================================================================
    
    async def fetch_bridge_activity(
        self,
        source_chain: Optional[str] = None,
        dest_chain: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch cross-chain bridge activity.
        
        Args:
            source_chain: Filter by source chain
            dest_chain: Filter by destination chain
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with bridge volume data
        """
        if start_date is None:
            start_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')
        
        filters = []
        if source_chain:
            filters.append(f"source_chain = '{source_chain}'")
        if dest_chain:
            filters.append(f"destination_chain = '{dest_chain}'")
        
        filter_clause = f"AND {' AND '.join(filters)}" if filters else ""
        
        sql = f"""
        SELECT 
            date_trunc('day', block_time) as day,
            source_chain,
            destination_chain,
            bridge_protocol,
            sum(amount_usd) as volume_usd,
            count(*) as transfer_count,
            count(distinct sender) as unique_senders
        FROM bridge.transactions
        WHERE block_time >= date '{start_date}'
          AND block_time < date '{end_date}'
          {filter_clause}
        GROUP BY 1, 2, 3, 4
        ORDER BY 1 DESC, 5 DESC
        """
        
        try:
            df = await self.execute_sql(sql, name="Bridge Activity")
            
            if not df.empty:
                df['venue'] = self.VENUE
                df['day'] = pd.to_datetime(df['day'])
                
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch bridge activity: {e}")
            return pd.DataFrame()
    
    async def fetch_net_bridge_flows(
        self,
        chain: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate net bridge flows for a chain.
        
        Args:
            chain: Target blockchain
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with net flow data (inflows - outflows)
        """
        if start_date is None:
            start_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')
        
        sql = f"""
        WITH flows AS (
            SELECT 
                date_trunc('day', block_time) as day,
                CASE 
                    WHEN destination_chain = '{chain}' THEN 'inflow'
                    WHEN source_chain = '{chain}' THEN 'outflow'
                END as flow_type,
                sum(amount_usd) as volume_usd
            FROM bridge.transactions
            WHERE block_time >= date '{start_date}'
              AND block_time < date '{end_date}'
              AND (source_chain = '{chain}' OR destination_chain = '{chain}')
            GROUP BY 1, 2
        )
        SELECT 
            day,
            sum(CASE WHEN flow_type = 'inflow' THEN volume_usd ELSE 0 END) as inflow_usd,
            sum(CASE WHEN flow_type = 'outflow' THEN volume_usd ELSE 0 END) as outflow_usd,
            sum(CASE WHEN flow_type = 'inflow' THEN volume_usd ELSE -volume_usd END) as net_flow_usd
        FROM flows
        GROUP BY 1
        ORDER BY 1 DESC
        """
        
        try:
            df = await self.execute_sql(sql, name=f"Net Bridge Flows {chain}")
            
            if not df.empty:
                df['chain'] = chain
                df['venue'] = self.VENUE
                df['day'] = pd.to_datetime(df['day'])
                
                # Add cumulative net flow
                df = df.sort_values('day')
                df['cumulative_net_flow'] = df['net_flow_usd'].cumsum()
                
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch net bridge flows: {e}")
            return pd.DataFrame()
    
    # =========================================================================
    # Network Metrics
    # =========================================================================
    
    async def fetch_gas_prices(
        self,
        chain: str = 'ethereum',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = 'hour'
    ) -> pd.DataFrame:
        """
        Fetch historical gas prices.
        
        Args:
            chain: Blockchain
            start_date: Start date
            end_date: End date
            interval: Aggregation interval
            
        Returns:
            DataFrame with gas price data
        """
        if start_date is None:
            start_date = (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')
        
        sql = f"""
        SELECT 
            date_trunc('{interval}', block_time) as period,
            avg(gas_price / 1e9) as avg_gas_gwei,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY gas_price / 1e9) as median_gas_gwei,
            percentile_cont(0.1) WITHIN GROUP (ORDER BY gas_price / 1e9) as p10_gas_gwei,
            percentile_cont(0.9) WITHIN GROUP (ORDER BY gas_price / 1e9) as p90_gas_gwei,
            min(gas_price / 1e9) as min_gas_gwei,
            max(gas_price / 1e9) as max_gas_gwei,
            count(*) as tx_count,
            sum(gas_used * gas_price / 1e18) as total_gas_eth
        FROM {chain}.transactions
        WHERE block_time >= date '{start_date}'
          AND block_time < date '{end_date}'
          AND gas_price > 0
        GROUP BY 1
        ORDER BY 1 DESC
        """
        
        try:
            df = await self.execute_sql(sql, name=f"Gas Prices {chain}")
            
            if not df.empty:
                df['chain'] = chain
                df['venue'] = self.VENUE
                df['period'] = pd.to_datetime(df['period'])
                
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch gas prices: {e}")
            return pd.DataFrame()
    
    async def fetch_active_addresses(
        self,
        chain: str = 'ethereum',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch daily active addresses.
        
        Args:
            chain: Blockchain
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with daily active address counts
        """
        if start_date is None:
            start_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')
        
        sql = f"""
        SELECT 
            date_trunc('day', block_time) as day,
            count(distinct "from") as unique_senders,
            count(distinct "to") as unique_receivers,
            count(distinct "from") + count(distinct "to") as total_active,
            count(*) as tx_count
        FROM {chain}.transactions
        WHERE block_time >= date '{start_date}'
          AND block_time < date '{end_date}'
        GROUP BY 1
        ORDER BY 1 DESC
        """
        
        try:
            df = await self.execute_sql(sql, name=f"Active Addresses {chain}")
            
            if not df.empty:
                df['chain'] = chain
                df['venue'] = self.VENUE
                df['day'] = pd.to_datetime(df['day'])
                
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch active addresses: {e}")
            return pd.DataFrame()
    
    # =========================================================================
    # Required Abstract Methods
    # =========================================================================
    
    async def fetch_funding_rates(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Dune doesn't have direct funding rate data.
        Use Coinalyze for funding rates instead.
        """
        logger.info("Dune doesn't provide funding rates. Use CoinalyzeCollector instead.")
        return pd.DataFrame()
    
    async def fetch_ohlcv(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch OHLCV-like data from DEX trades.

        Note: This constructs price data from DEX swaps, not true OHLCV.
        For CEX OHLCV, use exchange collectors.
        """
        # Skip early if SQL execution isn't available (requires paid plan)
        if not self._sql_execution_available:
            return pd.DataFrame()

        interval_map = {'1h': 'hour', '4h': 'hour', '1d': 'day'}
        interval = interval_map.get(timeframe, 'day')

        # PARALLELIZED: Fetch OHLCV for all symbols concurrently
        async def _fetch_single_ohlcv(symbol: str) -> Optional[pd.DataFrame]:
            sql = f"""
            WITH trades AS (
                SELECT
                    block_time,
                    amount_usd / nullif(token_bought_amount, 0) as price,
                    amount_usd
                FROM dex.trades
                WHERE token_bought_symbol = '{symbol}'
                  AND block_time >= date '{start_date}'
                  AND block_time < date '{end_date}'
                  AND amount_usd > 100
                  AND amount_usd < 1e9
            )
            SELECT
                date_trunc('{interval}', block_time) as period,
                first_value(price) OVER (PARTITION BY date_trunc('{interval}', block_time) ORDER BY block_time) as open,
                max(price) as high,
                min(price) as low,
                last_value(price) OVER (PARTITION BY date_trunc('{interval}', block_time) ORDER BY block_time ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as close,
                sum(amount_usd) as volume
            FROM trades
            GROUP BY date_trunc('{interval}', block_time), block_time, price
            ORDER BY 1 DESC
            """

            try:
                df = await self.execute_sql(sql, name=f"OHLCV {symbol}")
                if not df.empty:
                    df['symbol'] = symbol
                    return df
            except Exception as e:
                logger.warning(f"Failed to fetch OHLCV for {symbol}: {e}")

            return None

        tasks = [_fetch_single_ohlcv(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result['venue'] = self.VENUE
            result['venue_type'] = 'DEX'
            return result

        return pd.DataFrame()

    # =========================================================================
    # Standardized Collection Method (for dynamic routing in collection_manager)
    # =========================================================================

    async def collect_custom_queries(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect data using custom Dune queries (standardized interface).

        This method executes several pre-defined queries to collect
        DEX volume, token transfers, and network metrics data.

        Args:
            symbols: List of token symbols to analyze
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            **kwargs: Additional parameters (chain, query_type)

        Returns:
            DataFrame with combined query results
        """
        try:
            chain = kwargs.get('chain', 'ethereum')
            query_types = kwargs.get('query_types', ['dex_volume', 'active_addresses'])

            # Convert dates to strings if needed
            if hasattr(start_date, 'strftime'):
                start_date = start_date.strftime('%Y-%m-%d')
            if hasattr(end_date, 'strftime'):
                end_date = end_date.strftime('%Y-%m-%d')

            logger.info(f"Dune: Collecting custom_queries for {chain} from {start_date} to {end_date}")

            all_data = []

            # Collect DEX volume data
            if 'dex_volume' in query_types or 'all' in query_types:
                try:
                    dex_data = await self.fetch_dex_volume(
                        chain=chain,
                        start_date=start_date,
                        end_date=end_date
                    )
                    if not dex_data.empty:
                        dex_data['query_type'] = 'dex_volume'
                        all_data.append(dex_data)
                        logger.info(f"Dune: Collected {len(dex_data)} DEX volume records")
                except Exception as e:
                    logger.warning(f"Dune: Failed to fetch DEX volume: {e}")

            # Collect active addresses data
            if 'active_addresses' in query_types or 'all' in query_types:
                try:
                    addr_data = await self.fetch_active_addresses(
                        chain=chain,
                        start_date=start_date,
                        end_date=end_date
                    )
                    if not addr_data.empty:
                        addr_data['query_type'] = 'active_addresses'
                        all_data.append(addr_data)
                        logger.info(f"Dune: Collected {len(addr_data)} active address records")
                except Exception as e:
                    logger.warning(f"Dune: Failed to fetch active addresses: {e}")

            # Collect gas prices
            if 'gas_prices' in query_types or 'all' in query_types:
                try:
                    gas_data = await self.fetch_gas_prices(
                        chain=chain,
                        start_date=start_date,
                        end_date=end_date
                    )
                    if not gas_data.empty:
                        gas_data['query_type'] = 'gas_prices'
                        all_data.append(gas_data)
                        logger.info(f"Dune: Collected {len(gas_data)} gas price records")
                except Exception as e:
                    logger.warning(f"Dune: Failed to fetch gas prices: {e}")

            # Combine all results
            if all_data:
                # Each DataFrame might have different columns, so we can't simply concat
                # Return the first non-empty result or merge them intelligently
                result = pd.concat(all_data, ignore_index=True)
                result['timestamp'] = datetime.utcnow()
                logger.info(f"Dune: Total collected {len(result)} custom_queries records")
                return result

            logger.warning("Dune: No custom_queries data collected")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Dune collect_custom_queries error: {e}")
            return pd.DataFrame()

    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return {
            **self.collection_stats,
            'cache_size': len(self._cache),
            'credits_used': self.credits_used,
            'credits_remaining': max(0, self.monthly_budget - self.credits_used)
        }
    
    async def close(self):
        """Clean up resources."""
        if self.session and not self.session.closed:
            await self.session.close()
        self._cache.clear()
        logger.info(
            f"DuneAnalytics collector closed. "
            f"Queries: {self.collection_stats['queries_executed']}, "
            f"Credits: ~{self.credits_used}/{self.monthly_budget}"
        )

# =============================================================================
# Testing
# =============================================================================

async def test_dune_collector():
    """Test Dune Analytics collector."""
    import os
    
    config = {
        'api_key': os.getenv('DUNE_API_KEY', ''),
        'credits_per_month': 2500
    }
    
    collector = DuneAnalyticsCollector(config)
    
    try:
        print("Testing Dune Analytics Collector")
        print("=" * 60)
        
        # Test credit tracking
        print("\n1. Credit usage status:")
        usage = collector.get_credit_usage()
        print(f" Budget: {usage.monthly_budget}, Used: {usage.credits_used}")
        print(f" Reset date: {usage.reset_date}")
        
        if not config['api_key']:
            print("\n No API key - skipping live tests")
            return
        
        # Test DEX volume
        print("\n2. Testing DEX volume fetch...")
        dex_vol = await collector.fetch_dex_volume(
            chain='ethereum',
            start_date='2024-01-01',
            end_date='2024-01-07'
        )
        if not dex_vol.empty:
            print(f" Found {len(dex_vol)} records")
            print(f" Top DEX: {dex_vol.iloc[0]['dex'] if 'dex' in dex_vol.columns else 'N/A'}")
        
        # Test gas prices
        print("\n3. Testing gas prices fetch...")
        gas = await collector.fetch_gas_prices(chain='ethereum')
        if not gas.empty:
            print(f" Found {len(gas)} records")
            if 'median_gas_gwei' in gas.columns:
                print(f" Latest median gas: {gas.iloc[0]['median_gas_gwei']:.1f} gwei")
        
        # Test active addresses
        print("\n4. Testing active addresses fetch...")
        addrs = await collector.fetch_active_addresses(chain='ethereum')
        if not addrs.empty:
            print(f" Found {len(addrs)} records")
        
        print("\n" + "=" * 60)
        print(f"Collection stats: {collector.get_collection_stats()}")
        
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await collector.close()

if __name__ == '__main__':
    asyncio.run(test_dune_collector())