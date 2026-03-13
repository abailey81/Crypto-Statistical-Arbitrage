"""
GeckoTerminal Data Collector
============================

Free DEX data provider across 100+ chains.
API: https://api.geckoterminal.com/api/v2

Features:
- Real-time DEX prices
- Pool information and TVL
- Trending pools
- OHLCV data for pools
- Token information

Rate Limits:
- 15 calls/minute (without API key, conservative)
- No authentication required

Version: 1.0.0
"""

import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import asyncio
import logging

from ..base_collector import BaseCollector

logger = logging.getLogger(__name__)

class GeckoTerminalCollector(BaseCollector):
    """
    GeckoTerminal data collector for DEX prices and pool data.
    
    FREE API - No authentication required.
    
    Supported Networks:
    - ethereum, arbitrum, optimism, polygon, base, avalanche
    - bsc, fantom, solana, and 100+ more
    
    Data Types:
    - Pool prices and OHLCV
    - Pool TVL and volume
    - Token prices
    - Trending pools
    """
    
    VENUE = 'geckoterminal'
    VENUE_TYPE = 'DEX'
    BASE_URL = 'https://api.geckoterminal.com/api/v2'
    
    # Supported networks with DEX activity
    SUPPORTED_NETWORKS = [
        'eth', 'arbitrum', 'optimism', 'polygon_pos', 'base',
        'avalanche', 'bsc', 'fantom', 'solana', 'celo',
        'gnosis', 'moonbeam', 'zksync', 'linea', 'scroll'
    ]
    
    # OHLCV timeframes
    TIMEFRAME_MAP = {
        '1m': 'minute',
        '5m': 'minute',
        '15m': 'minute',
        '1h': 'hour',
        '4h': 'hour',
        '1d': 'day',
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize GeckoTerminal collector.
        
        Parameters
        ----------
        config : Dict, optional
            Configuration options
        """
        config = config or {}
        config.setdefault('rate_limit', 15) # 15 calls/minute (halved)
        super().__init__(config)
        
        self.session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0
        self._min_request_interval = 4.0 # 4 seconds between requests (doubled)
        
        self.logger.info(f"Initialized {self.VENUE} collector (FREE - no API key)")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'Accept': 'application/json',
                    'User-Agent': 'CryptoStatArb/1.0'
                }
            )
        return self.session
    
    async def _rate_limit(self):
        """Implement simple rate limiting."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()
    
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Make API request with rate limiting.
        
        Parameters
        ----------
        endpoint : str
            API endpoint
        params : Dict, optional
            Query parameters
            
        Returns
        -------
        Dict or None
            Response data or None on error
        """
        await self._rate_limit()
        
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            self.stats.api_calls += 1
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                elif response.status == 429:
                    self.logger.warning("Rate limited, waiting 60s")
                    await asyncio.sleep(60)
                    return None
                else:
                    self.logger.warning(f"HTTP {response.status} for {endpoint}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Request error: {e}")
            self.stats.errors += 1
            return None
    
    async def fetch_networks(self) -> pd.DataFrame:
        """
        Fetch all supported networks.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with network information
        """
        data = await self._make_request('/networks')
        
        if not data or 'data' not in data:
            return pd.DataFrame()
        
        networks = []
        for network in data['data']:
            networks.append({
                'network_id': network['id'],
                'name': network['attributes'].get('name', ''),
                'coingecko_asset_platform_id': network['attributes'].get('coingecko_asset_platform_id')
            })
        
        return pd.DataFrame(networks)
    
    async def fetch_trending_pools(
        self,
        network: str = 'eth',
        page: int = 1
    ) -> pd.DataFrame:
        """
        Fetch trending pools on a network.
        
        Parameters
        ----------
        network : str
            Network identifier (e.g., 'eth', 'arbitrum')
        page : int
            Page number for pagination
            
        Returns
        -------
        pd.DataFrame
            DataFrame with trending pool information
        """
        endpoint = f'/networks/{network}/trending_pools'
        params = {'page': page}
        
        data = await self._make_request(endpoint, params)
        
        if not data or 'data' not in data:
            return pd.DataFrame()
        
        pools = []
        for pool in data['data']:
            attrs = pool['attributes']
            pools.append({
                'pool_address': pool['id'],
                'network': network,
                'name': attrs.get('name', ''),
                'base_token_symbol': attrs.get('base_token_price_native_currency'),
                'quote_token_symbol': attrs.get('quote_token_price_native_currency'),
                'price_usd': float(attrs.get('base_token_price_usd', 0) or 0),
                'price_change_24h': float(attrs.get('price_change_percentage', {}).get('h24', 0) or 0),
                'volume_24h_usd': float(attrs.get('volume_usd', {}).get('h24', 0) or 0),
                'reserve_usd': float(attrs.get('reserve_in_usd', 0) or 0),
                'fdv_usd': float(attrs.get('fdv_usd', 0) or 0),
                'dex_name': attrs.get('dex_id', ''),
                'timestamp': datetime.utcnow(),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE
            })
        
        return pd.DataFrame(pools)
    
    async def fetch_pool_info(
        self,
        network: str,
        pool_address: str
    ) -> Optional[Dict]:
        """
        Fetch detailed pool information.
        
        Parameters
        ----------
        network : str
            Network identifier
        pool_address : str
            Pool contract address
            
        Returns
        -------
        Dict or None
            Pool information
        """
        endpoint = f'/networks/{network}/pools/{pool_address}'
        data = await self._make_request(endpoint)
        
        if not data or 'data' not in data:
            return None
        
        pool = data['data']
        attrs = pool['attributes']
        
        return {
            'pool_address': pool['id'],
            'network': network,
            'name': attrs.get('name', ''),
            'price_usd': float(attrs.get('base_token_price_usd', 0) or 0),
            'price_native': float(attrs.get('base_token_price_native_currency', 0) or 0),
            'volume_24h_usd': float(attrs.get('volume_usd', {}).get('h24', 0) or 0),
            'volume_1h_usd': float(attrs.get('volume_usd', {}).get('h1', 0) or 0),
            'reserve_usd': float(attrs.get('reserve_in_usd', 0) or 0),
            'fdv_usd': float(attrs.get('fdv_usd', 0) or 0),
            'market_cap_usd': float(attrs.get('market_cap_usd', 0) or 0),
            'price_change_24h': float(attrs.get('price_change_percentage', {}).get('h24', 0) or 0),
            'price_change_1h': float(attrs.get('price_change_percentage', {}).get('h1', 0) or 0),
            'transactions_24h_buys': attrs.get('transactions', {}).get('h24', {}).get('buys', 0),
            'transactions_24h_sells': attrs.get('transactions', {}).get('h24', {}).get('sells', 0),
            'timestamp': datetime.utcnow(),
            'venue': self.VENUE,
            'venue_type': self.VENUE_TYPE
        }
    
    async def fetch_pool_ohlcv(
        self,
        network: str,
        pool_address: str,
        timeframe: str = '1h',
        aggregate: int = 1,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a pool.
        
        Parameters
        ----------
        network : str
            Network identifier
        pool_address : str
            Pool contract address
        timeframe : str
            Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
        aggregate : int
            Number of timeframes to aggregate
        limit : int
            Maximum number of candles
            
        Returns
        -------
        pd.DataFrame
            OHLCV data
        """
        tf = self.TIMEFRAME_MAP.get(timeframe, 'hour')
        # Strip network prefix from pool address if present (e.g., "eth_0x..." -> "0x...")
        if '_' in pool_address:
            pool_address = pool_address.split('_', 1)[1]
        endpoint = f'/networks/{network}/pools/{pool_address}/ohlcv/{tf}'
        
        params = {
            'aggregate': aggregate,
            'limit': min(limit, 1000)
        }
        
        data = await self._make_request(endpoint, params)
        
        if not data or 'data' not in data:
            return pd.DataFrame()
        
        ohlcv_data = data['data'].get('attributes', {}).get('ohlcv_list', [])
        
        if not ohlcv_data:
            return pd.DataFrame()
        
        records = []
        for candle in ohlcv_data:
            # Format: [timestamp, open, high, low, close, volume]
            if len(candle) >= 6:
                records.append({
                    'timestamp': pd.to_datetime(candle[0], unit='s', utc=True),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5]),
                    'pool_address': pool_address,
                    'network': network,
                    'venue': self.VENUE,
                    'venue_type': self.VENUE_TYPE
                })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    async def fetch_token_info(
        self,
        network: str,
        token_address: str
    ) -> Optional[Dict]:
        """
        Fetch token information.
        
        Parameters
        ----------
        network : str
            Network identifier
        token_address : str
            Token contract address
            
        Returns
        -------
        Dict or None
            Token information
        """
        endpoint = f'/networks/{network}/tokens/{token_address}'
        data = await self._make_request(endpoint)
        
        if not data or 'data' not in data:
            return None
        
        token = data['data']
        attrs = token['attributes']
        
        return {
            'token_address': token['id'],
            'network': network,
            'name': attrs.get('name', ''),
            'symbol': attrs.get('symbol', ''),
            'decimals': attrs.get('decimals', 18),
            'price_usd': float(attrs.get('price_usd', 0) or 0),
            'fdv_usd': float(attrs.get('fdv_usd', 0) or 0),
            'market_cap_usd': float(attrs.get('market_cap_usd', 0) or 0),
            'total_supply': float(attrs.get('total_supply', 0) or 0),
            'volume_24h_usd': float(attrs.get('volume_usd', {}).get('h24', 0) or 0),
            'timestamp': datetime.utcnow(),
            'venue': self.VENUE,
            'venue_type': self.VENUE_TYPE
        }
    
    async def fetch_token_pools(
        self,
        network: str,
        token_address: str,
        page: int = 1
    ) -> pd.DataFrame:
        """
        Fetch all pools for a token.
        
        Parameters
        ----------
        network : str
            Network identifier
        token_address : str
            Token contract address
        page : int
            Page number
            
        Returns
        -------
        pd.DataFrame
            DataFrame with pool information
        """
        endpoint = f'/networks/{network}/tokens/{token_address}/pools'
        params = {'page': page}
        
        data = await self._make_request(endpoint, params)
        
        if not data or 'data' not in data:
            return pd.DataFrame()
        
        pools = []
        for pool in data['data']:
            attrs = pool['attributes']
            pools.append({
                'pool_address': pool['id'],
                'network': network,
                'name': attrs.get('name', ''),
                'dex_id': attrs.get('dex_id', ''),
                'price_usd': float(attrs.get('base_token_price_usd', 0) or 0),
                'volume_24h_usd': float(attrs.get('volume_usd', {}).get('h24', 0) or 0),
                'reserve_usd': float(attrs.get('reserve_in_usd', 0) or 0),
                'timestamp': datetime.utcnow(),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE
            })
        
        return pd.DataFrame(pools)
    
    async def search_pools(
        self,
        query: str,
        network: Optional[str] = None,
        page: int = 1
    ) -> pd.DataFrame:
        """
        Search for pools by name or token.
        
        Parameters
        ----------
        query : str
            Search query
        network : str, optional
            Filter by network
        page : int
            Page number
            
        Returns
        -------
        pd.DataFrame
            Search results
        """
        endpoint = '/search/pools'
        params = {
            'query': query,
            'page': page
        }
        if network:
            params['network'] = network
        
        data = await self._make_request(endpoint, params)
        
        if not data or 'data' not in data:
            return pd.DataFrame()
        
        pools = []
        for pool in data['data']:
            attrs = pool['attributes']
            pools.append({
                'pool_address': pool['id'],
                'network': attrs.get('network', {}).get('identifier', ''),
                'name': attrs.get('name', ''),
                'dex_id': attrs.get('dex_id', ''),
                'price_usd': float(attrs.get('base_token_price_usd', 0) or 0),
                'volume_24h_usd': float(attrs.get('volume_usd', {}).get('h24', 0) or 0),
                'reserve_usd': float(attrs.get('reserve_in_usd', 0) or 0),
                'timestamp': datetime.utcnow(),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE
            })
        
        return pd.DataFrame(pools)
    
    async def _fetch_single_symbol_ohlcv(
        self,
        symbol: str,
        network: str,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Helper to fetch OHLCV for a single symbol (parallelized).

        Parameters
        ----------
        symbol : str
            Token symbol
        network : str
            Network to search on
        timeframe : str
            Timeframe

        Returns
        -------
        pd.DataFrame
            OHLCV data for this symbol
        """
        try:
            self.logger.info(f"Fetching {symbol} pools on {network}")

            # Search for pools
            pools = await self.search_pools(symbol, network=network)

            if pools.empty:
                self.logger.warning(f"No pools found for {symbol}")
                return pd.DataFrame()

            # Get top pool by volume
            top_pool = pools.sort_values('volume_24h_usd', ascending=False).iloc[0]
            pool_address = top_pool['pool_address']

            # Fetch OHLCV
            ohlcv = await self.fetch_pool_ohlcv(
                network=network,
                pool_address=pool_address,
                timeframe=timeframe
            )

            if not ohlcv.empty:
                ohlcv['symbol'] = symbol
                self.stats.records_collected += len(ohlcv)

            self.stats.symbols_processed.append(symbol)
            return ohlcv

        except Exception as e:
            self.logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()

    async def fetch_ohlcv(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        network: str = 'eth',
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for DEX tokens.

        Note: GeckoTerminal requires pool addresses, not symbols.
        This method searches for pools and fetches OHLCV for each.

        Parameters
        ----------
        symbols : List[str]
            List of token symbols to search
        timeframe : str
            Timeframe
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        network : str
            Network to search on

        Returns
        -------
        pd.DataFrame
            OHLCV data
        """
        self.start_collection()

        # Parallelize symbol fetching using asyncio.gather
        tasks = [self._fetch_single_symbol_ohlcv(symbol, network, timeframe) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter valid DataFrames
        all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

        self.end_collection()

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    async def fetch_funding_rates(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        DEX does not have funding rates.

        Returns empty DataFrame.
        """
        self.logger.info("GeckoTerminal (DEX) does not have funding rates")
        return pd.DataFrame()

    async def _fetch_single_symbol_pools(
        self,
        symbol: str,
        network: str,
        token_mappings: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Helper to fetch pools for a single symbol (parallelized).

        Parameters
        ----------
        symbol : str
            Token symbol or address
        network : str
            Network to search on
        token_mappings : Dict[str, str]
            Symbol mapping dictionary

        Returns
        -------
        pd.DataFrame
            Pool data for this symbol
        """
        try:
            # Normalize symbol
            symbol_upper = symbol.upper()
            search_term = token_mappings.get(symbol_upper, symbol_upper)

            # Check if it's an address (starts with 0x)
            if symbol.startswith('0x') and len(symbol) == 42:
                # Use fetch_token_pools for addresses
                pools = await self.fetch_token_pools(
                    network=network,
                    token_address=symbol
                )
            else:
                # Use search for symbols
                pools = await self.search_pools(
                    query=search_term,
                    network=network
                )

            if not pools.empty:
                pools['search_symbol'] = symbol
                self.logger.info(f"Found {len(pools)} pools for {symbol}")
                return pools
            else:
                self.logger.warning(f"No pools found for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error fetching pools for {symbol}: {e}")
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
        Uses search_pools for symbol lookup, fetch_token_pools for addresses.
        """
        try:
            network = kwargs.get('network', 'eth')

            # Common token mappings
            token_mappings = {
                'BTC': 'WBTC',
                'BITCOIN': 'WBTC',
                'ETH': 'WETH',
                'ETHEREUM': 'WETH',
            }

            # Parallelize symbol fetching using asyncio.gather
            tasks = [self._fetch_single_symbol_pools(symbol, network, token_mappings) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter valid DataFrames
            all_records = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

            if all_records:
                return pd.concat(all_records, ignore_index=True)

            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"GeckoTerminal collect_pool_data error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
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
            network = kwargs.get('network', 'eth')

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
                end_date=end_str,
                network=network
            )
        except Exception as e:
            self.logger.error(f"GeckoTerminal collect_ohlcv error: {e}")
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
            self.logger.error(f"GeckoTerminal collect_funding_rates error: {e}")
            return pd.DataFrame()

    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.debug("Closed aiohttp session")
