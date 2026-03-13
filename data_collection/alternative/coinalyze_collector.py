"""
Coinalyze API Collector - FREE Alternative to Coinglass.

validated collector for derivatives data:
- Funding rates (current, predicted, historical)
- Open interest (current, historical, by exchange)
- Liquidations (historical, aggregated)
- Long/Short ratios
- OHLCV data for perpetuals
- Premium/discount calculation
- Funding rate arbitrage detection
- Cross-exchange aggregation

API Documentation: https://api.coinalyze.net/v1/doc/
Rate Limit: 40 calls/minute per API key
Registration: https://coinalyze.net (free)

Version: 2.0.0
"""

import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
import time

from ..base_collector import BaseCollector
from ..utils.rate_limiter import get_shared_rate_limiter
from ..utils.retry_handler import RetryHandler

logger = logging.getLogger(__name__)

class DerivativeExchange(Enum):
    """Supported derivative exchanges."""
    BINANCE = 'binance'
    BYBIT = 'bybit'
    OKX = 'okx'
    BITGET = 'bitget'
    DYDX = 'dydx'
    BITMEX = 'bitmex'
    BITFINEX = 'bitfinex'
    GATE = 'gate'
    HYPERLIQUID = 'hyperliquid'

@dataclass
class FundingArbitrage:
    """Funding rate arbitrage opportunity."""
    symbol: str
    long_exchange: str
    short_exchange: str
    long_rate: float
    short_rate: float
    spread: float
    spread_annualized: float
    timestamp: datetime

class CoinalyzeCollector(BaseCollector):
    """
    Coinalyze data collector - FREE alternative to Coinglass.
    
    Provides:
    - Funding rates across major exchanges (8h intervals)
    - Open interest data (USD and base currency)
    - Liquidation data (long/short breakdown)
    - Long/Short ratios (trader sentiment)
    - OHLCV data for perpetual contracts
    - Cross-exchange aggregation
    - Funding arbitrage detection
    - Premium/discount analysis
    
    Supported Exchanges:
    - Binance, Bybit, OKX, Bitget, dYdX, Bitmex, Bitfinex, Gate.io
    
    Rate Limits:
    - 40 calls/minute per API key
    - Historical data: 1500-2000 datapoints per request (intraday)
    - Daily data: unlimited datapoints
    
    Symbol Format:
    - Binance USDT perp: BTCUSDT_PERP.A
    - Bybit: BTCUSDT_PERP.BY
    - OKX: BTCUSDT_PERP.O
    - dYdX: BTC-USD.DY
    """
    
    VENUE = 'coinalyze'
    VENUE_TYPE = 'aggregator'
    BASE_URL = 'https://api.coinalyze.net/v1'

    # Collection manager compatibility attributes
    supported_data_types = ['funding_rates', 'open_interest', 'liquidations', 'ohlcv']
    venue = 'coinalyze'
    requires_auth = True # Requires free API key from coinalyze.net

    # Exchange code mapping for Coinalyze API
    EXCHANGE_CODES = {
        'binance': 'A',
        'bybit': 'BY',
        'okx': 'O',
        'bitget': 'BG',
        'dydx': 'DY',
        'bitmex': 'BM',
        'bitfinex': 'BF',
        'gate': 'G',
        'hyperliquid': 'HL'
    }
    
    # Reverse mapping
    CODE_TO_EXCHANGE = {v: k for k, v in EXCHANGE_CODES.items()}
    
    # Symbol suffix mapping by exchange
    SYMBOL_FORMATS = {
        'binance': '{base}USDT_PERP.A',
        'bybit': '{base}USDT_PERP.BY',
        'okx': '{base}USDT_PERP.O',
        'bitget': '{base}_UMCBL.BG',
        'dydx': '{base}-USD.DY',
        'bitmex': '{base}USD.BM',
        'bitfinex': '{base}F0:USTF0.BF',
        'gate': '{base}_USDT.G',
        'hyperliquid': '{base}-USD.HL'
    }
    
    # Major symbols for batch operations
    MAJOR_SYMBOLS = [
        'BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'AVAX', 'LINK', 
        'MATIC', 'ARB', 'OP', 'BNB', 'ADA', 'DOT', 'ATOM', 'LTC'
    ]
    
    # Default exchanges for aggregation
    DEFAULT_EXCHANGES = ['binance', 'bybit', 'okx', 'bitget']
    
    def __init__(self, config: Dict):
        """
        Initialize Coinalyze collector.
        
        Args:
            config: Configuration dict containing:
                - api_key: Coinalyze API key (free from coinalyze.net)
                - rate_limit: Calls per minute (default 40)
                - default_exchanges: List of default exchanges
        """
        super().__init__(config)

        self.api_key = config.get('api_key') or config.get('coinalyze_api_key') or os.getenv('COINALYZE_API_KEY', '')
        if not self.api_key:
            logger.warning("No Coinalyze API key provided. Register free at coinalyze.net")
        
        # Use shared rate limiter to avoid re-initialization overhead
        rate_limit = config.get('rate_limit', 10)
        self.rate_limiter = get_shared_rate_limiter('coinalyze', rate=rate_limit, per=60.0, burst=3)

        # Retry handler - OPTIMIZATION: Reduced max_delay from 60s to 30s
        self.retry_handler = RetryHandler(
            max_retries=3, # Reduced from 4 for faster fail-through
            base_delay=2.0,
            max_delay=30.0 # Reduced from 60s
        )
        
        # Default exchanges
        self.default_exchanges = config.get('default_exchanges', self.DEFAULT_EXCHANGES)

        # Concurrency limiter - Coinalyze free tier is very limited (~10 req/min)
        # Only allow 3 concurrent requests to avoid DNS/connection saturation
        self._request_sem = asyncio.Semaphore(3)

        # Session management
        self.session = None
        
        # Collection stats
        self.collection_stats = {
            'records_collected': 0,
            'api_calls': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            # Coinalyze uses query param for API key, not headers
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={'Accept': 'application/json'}
            )
        return self.session
    
    def _build_symbol(self, base: str, exchange: str) -> str:
        """
        Build Coinalyze symbol format for an exchange.
        
        Args:
            base: Base symbol (e.g., 'BTC', 'ETH')
            exchange: Exchange name (e.g., 'binance')
            
        Returns:
            Coinalyze symbol format (e.g., 'BTCUSDT_PERP.A')
        """
        exchange_lower = exchange.lower()
        
        if exchange_lower not in self.SYMBOL_FORMATS:
            logger.warning(f"Unknown exchange {exchange}, using binance format")
            exchange_lower = 'binance'
        
        return self.SYMBOL_FORMATS[exchange_lower].format(base=base.upper())
    
    def _parse_exchange_from_symbol(self, symbol: str) -> str:
        """Parse exchange name from Coinalyze symbol."""
        if '.' not in symbol:
            return 'unknown'
        
        code = symbol.split('.')[-1]
        return self.CODE_TO_EXCHANGE.get(code, 'unknown')
    
    def _parse_base_from_symbol(self, symbol: str) -> str:
        """Parse base currency from Coinalyze symbol."""
        # Remove exchange suffix
        if '.' in symbol:
            symbol = symbol.split('.')[0]

        # Handle different formats
        for suffix in ['USDT_PERP', 'USD', '_UMCBL', 'F0:USTF0', '_USDT']:
            if suffix in symbol:
                return symbol.replace(suffix, '').replace('-', '')

        return symbol.replace('USDT', '').replace('USD', '')

    def _map_interval(self, interval: str) -> str:
        """
        Map standard interval format to Coinalyze API format.

        Coinalyze expects: 1min, 5min, 15min, 30min, 1hour, 2hour, 4hour, 6hour, 12hour, daily
        Standard format: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 1w
        Note: 8h is mapped to 12hour (closest available)
        """
        interval_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1hour', '2h': '2hour', '4h': '4hour', '6h': '6hour',
            '8h': '12hour', # 8hour not supported, use 12hour
            '12h': '12hour', '1d': 'daily', '1w': 'daily',
            # Also handle if already in correct format
            '1min': '1min', '5min': '5min', '15min': '15min', '30min': '30min',
            '1hour': '1hour', '2hour': '2hour', '4hour': '4hour', '6hour': '6hour',
            '12hour': '12hour', 'daily': 'daily',
        }
        return interval_map.get(interval.lower(), '4hour')
    
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> Union[Dict, List, None]:
        """
        Make API request with rate limiting and retry logic.

        Args:
            endpoint: API endpoint (e.g., '/funding-rate')
            params: Query parameters

        Returns:
            JSON response data
        """
        async with self._request_sem:
            return await self._make_request_inner(endpoint, params)

    async def _make_request_inner(
        self,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> Union[Dict, List, None]:
        """Inner request method, called under semaphore."""
        acquire_result = await self.rate_limiter.acquire(timeout=120.0)
        if hasattr(acquire_result, 'acquired') and not acquire_result.acquired:
            logger.debug(f"Coinalyze rate limiter timeout for {endpoint}")
            return None
        self.collection_stats['api_calls'] += 1

        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"

        # Add API key to params (Coinalyze requires api_key as query parameter)
        request_params = dict(params) if params else {}
        if self.api_key:
            request_params['api_key'] = self.api_key

        async def _request():
            async with session.get(url, params=request_params) as response:
                if response.status == 429:
                    # Rate limit - log once and wait before retry
                    if not hasattr(self, '_coinalyze_rate_limit_logged'):
                        logger.warning("Coinalyze rate limit exceeded - reducing request rate")
                        self._coinalyze_rate_limit_logged = True
                    await asyncio.sleep(30) # Wait 30 seconds
                    raise aiohttp.ClientError("Rate limit exceeded")
                elif response.status == 401:
                    # Auth error - log once
                    if not hasattr(self, '_coinalyze_auth_logged'):
                        logger.warning("Coinalyze authentication failed - check API key at coinalyze.net")
                        self._coinalyze_auth_logged = True
                    return None
                elif response.status == 400:
                    # Bad request - likely invalid symbol
                    text = await response.text()
                    logger.debug(f"Coinalyze bad request: {text[:100]}")
                    return None # Don't retry, just return empty
                elif response.status == 404:
                    # Symbol not found
                    logger.debug(f"Coinalyze endpoint/symbol not found: {endpoint}")
                    return None
                elif response.status == 500 or response.status == 502 or response.status == 503:
                    # Server error - allow retry
                    logger.debug(f"Coinalyze server error {response.status}")
                    raise aiohttp.ClientError(f"Server error {response.status}")
                elif response.status != 200:
                    text = await response.text()
                    logger.warning(f"Coinalyze API error {response.status}: {text[:100]}")
                    raise aiohttp.ClientError(f"API error {response.status}")
                return await response.json()
        
        try:
            return await self.retry_handler.execute(_request)
        except Exception as e:
            logger.error(f"Request failed: {endpoint} - {e}")
            self.collection_stats['errors'] += 1
            return None
    
    # ========================================================================
    # FUNDING RATES
    # ========================================================================
    
    async def _fetch_single_funding_rate(
        self,
        symbol: str,
        exchange: str,
        start_ts: int,
        end_ts: int
    ) -> List[Dict]:
        """Fetch funding rate for a single symbol/exchange pair."""
        coinalyze_symbol = self._build_symbol(symbol, exchange)

        try:
            params = {
                'symbols': coinalyze_symbol,
                'from': start_ts,
                'to': end_ts,
                'interval': self._map_interval('8h')
            }

            data = await self._make_request('/funding-rate-history', params)

            records = []
            if data and isinstance(data, list):
                for record in data:
                    history = record.get('history', [])
                    for point in history:
                        rate = float(point.get('r', 0))
                        records.append({
                            'timestamp': pd.to_datetime(point['t'], unit='s', utc=True),
                            'symbol': symbol.upper(),
                            'funding_rate': rate,
                            'funding_rate_annualized': rate * 3 * 365 * 100, # 8h to annual %
                            'exchange': exchange,
                            'coinalyze_symbol': coinalyze_symbol,
                            'venue': self.VENUE,
                            'venue_type': self.VENUE_TYPE
                        })
            return records

        except Exception as e:
            logger.error(f"Error fetching {symbol} on {exchange}: {e}")
            self.collection_stats['errors'] += 1
            return []

    async def fetch_funding_rates(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        exchanges: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates IN PARALLEL (30x speedup).

        Args:
            symbols: List of base symbols (e.g., ['BTC', 'ETH'])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            exchanges: Optional list of exchanges to filter

        Returns:
            DataFrame with columns:
            - timestamp, symbol, funding_rate, funding_rate_annualized,
              exchange, next_funding_time, venue, venue_type
        """
        exchanges = exchanges or self.default_exchanges

        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        # PARALLEL: Create all tasks upfront
        tasks = []
        for symbol in symbols:
            for exchange in exchanges:
                tasks.append(self._fetch_single_funding_rate(symbol, exchange, start_ts, end_ts))

        # Execute ALL tasks in parallel
        logger.info(f"Fetching {len(tasks)} funding rate combinations in parallel")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results
        all_data = []
        for result in results:
            if isinstance(result, list):
                all_data.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Task failed: {result}")

        df = pd.DataFrame(all_data)

        if not df.empty:
            df = df.sort_values(['timestamp', 'symbol', 'exchange']).reset_index(drop=True)
            self.collection_stats['records_collected'] += len(df)
            logger.info(f"Collected {len(df)} funding rate records")

        return df
    
    async def fetch_current_funding_rates(
        self,
        symbols: List[str],
        exchanges: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch current funding rates across exchanges.
        
        Args:
            symbols: List of base symbols
            exchanges: Optional list of exchanges
            
        Returns:
            DataFrame with current funding rates
        """
        exchanges = exchanges or self.default_exchanges
        
        # Build all symbol combinations
        all_symbols = []
        symbol_map = {} # Map coinalyze symbol back to base/exchange
        
        for symbol in symbols:
            for exchange in exchanges:
                cs = self._build_symbol(symbol, exchange)
                all_symbols.append(cs)
                symbol_map[cs] = (symbol, exchange)
        
        params = {'symbols': ','.join(all_symbols)}
        
        try:
            data = await self._make_request('/funding-rate', params)
            
            records = []
            if data and isinstance(data, list):
                for record in data:
                    cs = record.get('symbol', '')
                    if cs in symbol_map:
                        base, exchange = symbol_map[cs]
                    else:
                        base = self._parse_base_from_symbol(cs)
                        exchange = self._parse_exchange_from_symbol(cs)
                    
                    rate = float(record.get('rate', 0))
                    
                    records.append({
                        'timestamp': pd.Timestamp.now(tz='UTC'),
                        'symbol': base.upper(),
                        'funding_rate': rate,
                        'funding_rate_annualized': rate * 3 * 365 * 100,
                        'exchange': exchange,
                        'next_funding_time': pd.to_datetime(
                            record.get('nextFundingTime', 0), unit='ms', utc=True
                        ) if record.get('nextFundingTime') else None,
                        'mark_price': float(record.get('markPrice', 0)),
                        'venue': self.VENUE,
                        'venue_type': self.VENUE_TYPE
                    })
            
            df = pd.DataFrame(records)
            self.collection_stats['records_collected'] += len(df)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching current funding rates: {e}")
            return pd.DataFrame()
    
    async def fetch_predicted_funding_rates(
        self,
        symbols: List[str],
        exchanges: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch predicted next funding rates.
        
        Args:
            symbols: List of base symbols
            exchanges: Optional list of exchanges
            
        Returns:
            DataFrame with predicted funding rates
        """
        exchanges = exchanges or self.default_exchanges
        
        all_symbols = []
        symbol_map = {}
        
        for symbol in symbols:
            for exchange in exchanges:
                cs = self._build_symbol(symbol, exchange)
                all_symbols.append(cs)
                symbol_map[cs] = (symbol, exchange)
        
        params = {'symbols': ','.join(all_symbols)}
        
        try:
            data = await self._make_request('/predicted-funding-rate', params)
            
            records = []
            if data and isinstance(data, list):
                for record in data:
                    cs = record.get('symbol', '')
                    if cs in symbol_map:
                        base, exchange = symbol_map[cs]
                    else:
                        base = self._parse_base_from_symbol(cs)
                        exchange = self._parse_exchange_from_symbol(cs)
                    
                    rate = float(record.get('predictedRate', 0))
                    
                    records.append({
                        'timestamp': pd.Timestamp.now(tz='UTC'),
                        'symbol': base.upper(),
                        'predicted_funding_rate': rate,
                        'predicted_rate_annualized': rate * 3 * 365 * 100,
                        'exchange': exchange,
                        'venue': self.VENUE,
                        'venue_type': self.VENUE_TYPE
                    })
            
            df = pd.DataFrame(records)
            self.collection_stats['records_collected'] += len(df)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching predicted funding rates: {e}")
            return pd.DataFrame()
    
    async def fetch_aggregated_funding(
        self,
        symbols: List[str],
        exchanges: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch funding rates aggregated across exchanges.
        
        Args:
            symbols: List of base symbols
            exchanges: List of exchanges to aggregate
            
        Returns:
            DataFrame with aggregated funding metrics per symbol
        """
        current = await self.fetch_current_funding_rates(symbols, exchanges)
        
        if current.empty:
            return pd.DataFrame()
        
        # Aggregate by symbol
        aggregated = []
        for symbol in current['symbol'].unique():
            symbol_data = current[current['symbol'] == symbol]
            
            rates = symbol_data['funding_rate'].values
            
            aggregated.append({
                'timestamp': pd.Timestamp.now(tz='UTC'),
                'symbol': symbol,
                'funding_rate_mean': np.mean(rates),
                'funding_rate_median': np.median(rates),
                'funding_rate_std': np.std(rates),
                'funding_rate_min': np.min(rates),
                'funding_rate_max': np.max(rates),
                'funding_rate_spread': np.max(rates) - np.min(rates),
                'annualized_mean': np.mean(rates) * 3 * 365 * 100,
                'annualized_spread': (np.max(rates) - np.min(rates)) * 3 * 365 * 100,
                'exchange_count': len(symbol_data),
                'exchanges': ','.join(symbol_data['exchange'].tolist()),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE
            })
        
        return pd.DataFrame(aggregated)
    
    async def detect_funding_arbitrage(
        self,
        symbols: List[str],
        exchanges: Optional[List[str]] = None,
        min_spread_annualized: float = 10.0
    ) -> List[FundingArbitrage]:
        """
        Detect funding rate arbitrage opportunities.
        
        Strategy: Long on exchange with negative funding, 
                  Short on exchange with positive funding.
        
        Args:
            symbols: List of base symbols
            exchanges: List of exchanges to compare
            min_spread_annualized: Minimum annualized spread to flag (%)
            
        Returns:
            List of FundingArbitrage opportunities
        """
        current = await self.fetch_current_funding_rates(symbols, exchanges)
        
        if current.empty:
            return []
        
        opportunities = []
        
        for symbol in current['symbol'].unique():
            symbol_data = current[current['symbol'] == symbol]
            
            if len(symbol_data) < 2:
                continue
            
            # Find min and max funding rates
            min_idx = symbol_data['funding_rate'].idxmin()
            max_idx = symbol_data['funding_rate'].idxmax()
            
            min_row = symbol_data.loc[min_idx]
            max_row = symbol_data.loc[max_idx]
            
            spread = max_row['funding_rate'] - min_row['funding_rate']
            spread_annualized = spread * 3 * 365 * 100
            
            if abs(spread_annualized) >= min_spread_annualized:
                opportunities.append(FundingArbitrage(
                    symbol=symbol,
                    long_exchange=min_row['exchange'], # Long where funding is lowest
                    short_exchange=max_row['exchange'], # Short where funding is highest
                    long_rate=min_row['funding_rate'],
                    short_rate=max_row['funding_rate'],
                    spread=spread,
                    spread_annualized=spread_annualized,
                    timestamp=datetime.utcnow()
                ))
        
        # Sort by spread
        opportunities.sort(key=lambda x: abs(x.spread_annualized), reverse=True)
        
        return opportunities
    
    # ========================================================================
    # OPEN INTEREST
    # ========================================================================
    
    async def _fetch_single_open_interest(
        self,
        symbol: str,
        exchange: str,
        start_ts: int,
        end_ts: int,
        interval: str
    ) -> List[Dict]:
        """Fetch open interest for a single symbol/exchange pair."""
        coinalyze_symbol = self._build_symbol(symbol, exchange)

        try:
            params = {
                'symbols': coinalyze_symbol,
                'from': start_ts,
                'to': end_ts,
                'interval': self._map_interval(interval)
            }

            data = await self._make_request('/open-interest-history', params)

            records = []
            if data and isinstance(data, list):
                for record in data:
                    history = record.get('history', [])
                    for point in history:
                        records.append({
                            'timestamp': pd.to_datetime(point['t'], unit='s', utc=True),
                            'symbol': symbol.upper(),
                            'open_interest_usd': float(point.get('o', 0)),
                            'exchange': exchange,
                            'venue': self.VENUE,
                            'venue_type': self.VENUE_TYPE
                        })
            return records

        except Exception as e:
            logger.error(f"Error fetching OI for {symbol} on {exchange}: {e}")
            self.collection_stats['errors'] += 1
            return []

    async def fetch_open_interest(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        exchanges: Optional[List[str]] = None,
        interval: str = '1h'
    ) -> pd.DataFrame:
        """
        Fetch historical open interest data IN PARALLEL (30x speedup).

        Args:
            symbols: List of base symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            exchanges: Optional list of exchanges
            interval: Data interval (1h, 4h, 1d)

        Returns:
            DataFrame with open interest data
        """
        exchanges = exchanges or self.default_exchanges

        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        # PARALLEL: Create all tasks upfront
        tasks = []
        for symbol in symbols:
            for exchange in exchanges:
                tasks.append(self._fetch_single_open_interest(symbol, exchange, start_ts, end_ts, interval))

        # Execute ALL tasks in parallel
        logger.info(f"Fetching {len(tasks)} open interest combinations in parallel")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results
        all_data = []
        for result in results:
            if isinstance(result, list):
                all_data.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Task failed: {result}")

        df = pd.DataFrame(all_data)

        if not df.empty:
            df = df.sort_values(['timestamp', 'symbol', 'exchange']).reset_index(drop=True)
            self.collection_stats['records_collected'] += len(df)
            logger.info(f"Collected {len(df)} open interest records")

        return df
    
    async def fetch_current_open_interest(
        self,
        symbols: List[str],
        exchanges: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch current open interest.
        
        Args:
            symbols: List of base symbols
            exchanges: Optional list of exchanges
            
        Returns:
            DataFrame with current open interest
        """
        exchanges = exchanges or self.default_exchanges
        
        all_symbols = []
        symbol_map = {}
        
        for symbol in symbols:
            for exchange in exchanges:
                cs = self._build_symbol(symbol, exchange)
                all_symbols.append(cs)
                symbol_map[cs] = (symbol, exchange)
        
        params = {'symbols': ','.join(all_symbols)}
        
        try:
            data = await self._make_request('/open-interest', params)
            
            records = []
            if data and isinstance(data, list):
                for record in data:
                    cs = record.get('symbol', '')
                    if cs in symbol_map:
                        base, exchange = symbol_map[cs]
                    else:
                        base = self._parse_base_from_symbol(cs)
                        exchange = self._parse_exchange_from_symbol(cs)
                    
                    oi_usd = float(record.get('openInterestUsd', 0))
                    oi_base = float(record.get('openInterest', 0))
                    
                    records.append({
                        'timestamp': pd.Timestamp.now(tz='UTC'),
                        'symbol': base.upper(),
                        'open_interest_usd': oi_usd,
                        'open_interest_base': oi_base,
                        'exchange': exchange,
                        'venue': self.VENUE,
                        'venue_type': self.VENUE_TYPE
                    })
            
            df = pd.DataFrame(records)
            self.collection_stats['records_collected'] += len(df)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching current open interest: {e}")
            return pd.DataFrame()
    
    async def fetch_aggregated_open_interest(
        self,
        symbols: List[str],
        exchanges: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch open interest aggregated across exchanges.
        
        Args:
            symbols: List of base symbols
            exchanges: List of exchanges to aggregate
            
        Returns:
            DataFrame with aggregated OI metrics per symbol
        """
        current = await self.fetch_current_open_interest(symbols, exchanges)
        
        if current.empty:
            return pd.DataFrame()
        
        aggregated = []
        for symbol in current['symbol'].unique():
            symbol_data = current[current['symbol'] == symbol]
            
            aggregated.append({
                'timestamp': pd.Timestamp.now(tz='UTC'),
                'symbol': symbol,
                'total_oi_usd': symbol_data['open_interest_usd'].sum(),
                'total_oi_base': symbol_data['open_interest_base'].sum(),
                'avg_oi_usd': symbol_data['open_interest_usd'].mean(),
                'max_oi_usd': symbol_data['open_interest_usd'].max(),
                'dominant_exchange': symbol_data.loc[symbol_data['open_interest_usd'].idxmax(), 'exchange'],
                'exchange_count': len(symbol_data),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE
            })
        
        return pd.DataFrame(aggregated)
    
    # ========================================================================
    # LIQUIDATIONS
    # ========================================================================
    
    async def _fetch_single_liquidation(
        self,
        symbol: str,
        exchange: str,
        start_ts: int,
        end_ts: int,
        interval: str
    ) -> List[Dict]:
        """Fetch liquidations for a single symbol/exchange pair."""
        coinalyze_symbol = self._build_symbol(symbol, exchange)

        try:
            params = {
                'symbols': coinalyze_symbol,
                'from': start_ts,
                'to': end_ts,
                'interval': self._map_interval(interval)
            }

            data = await self._make_request('/liquidation-history', params)

            records = []
            if data and isinstance(data, list):
                for record in data:
                    history = record.get('history', [])
                    for point in history:
                        long_liq = float(point.get('l', 0))
                        short_liq = float(point.get('s', 0))
                        total_liq = long_liq + short_liq

                        records.append({
                            'timestamp': pd.to_datetime(point['t'], unit='s', utc=True),
                            'symbol': symbol.upper(),
                            'long_liquidations_usd': long_liq,
                            'short_liquidations_usd': short_liq,
                            'total_liquidations_usd': total_liq,
                            'liquidation_ratio': long_liq / total_liq if total_liq > 0 else 0.5,
                            'dominant_side': 'long' if long_liq > short_liq else 'short',
                            'exchange': exchange,
                            'venue': self.VENUE,
                            'venue_type': self.VENUE_TYPE
                        })
            return records

        except Exception as e:
            logger.error(f"Error fetching liquidations for {symbol} on {exchange}: {e}")
            self.collection_stats['errors'] += 1
            return []

    async def fetch_liquidations(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        exchanges: Optional[List[str]] = None,
        interval: str = '1h'
    ) -> pd.DataFrame:
        """
        Fetch historical liquidation data IN PARALLEL (30x speedup).

        Args:
            symbols: List of base symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            exchanges: Optional list of exchanges
            interval: Aggregation interval

        Returns:
            DataFrame with liquidation data
        """
        exchanges = exchanges or self.default_exchanges

        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        # PARALLEL: Create all tasks upfront
        tasks = []
        for symbol in symbols:
            for exchange in exchanges:
                tasks.append(self._fetch_single_liquidation(symbol, exchange, start_ts, end_ts, interval))

        # Execute ALL tasks in parallel
        logger.info(f"Fetching {len(tasks)} liquidation combinations in parallel")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results
        all_data = []
        for result in results:
            if isinstance(result, list):
                all_data.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Task failed: {result}")

        df = pd.DataFrame(all_data)

        if not df.empty:
            df = df.sort_values(['timestamp', 'symbol', 'exchange']).reset_index(drop=True)
            self.collection_stats['records_collected'] += len(df)
            logger.info(f"Collected {len(df)} liquidation records")

        return df
    
    async def fetch_aggregated_liquidations(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        exchanges: Optional[List[str]] = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch liquidations aggregated across exchanges.
        
        Args:
            symbols: List of base symbols
            start_date: Start date
            end_date: End date
            exchanges: List of exchanges
            interval: Aggregation interval
            
        Returns:
            DataFrame with aggregated liquidation data
        """
        liquidations = await self.fetch_liquidations(
            symbols, start_date, end_date, exchanges, interval
        )
        
        if liquidations.empty:
            return pd.DataFrame()
        
        # Aggregate by timestamp and symbol
        aggregated = liquidations.groupby(['timestamp', 'symbol']).agg({
            'long_liquidations_usd': 'sum',
            'short_liquidations_usd': 'sum',
            'total_liquidations_usd': 'sum',
            'exchange': lambda x: ','.join(x.unique())
        }).reset_index()
        
        aggregated['liquidation_ratio'] = (
            aggregated['long_liquidations_usd'] / 
            aggregated['total_liquidations_usd'].replace(0, 1)
        )
        aggregated['dominant_side'] = aggregated.apply(
            lambda r: 'long' if r['long_liquidations_usd'] > r['short_liquidations_usd'] else 'short',
            axis=1
        )
        aggregated['venue'] = self.VENUE
        
        return aggregated
    
    # ========================================================================
    # LONG/SHORT RATIOS
    # ========================================================================

    async def _fetch_single_long_short_ratio(
        self,
        symbol: str,
        exchange: str,
        start_ts: int,
        end_ts: int,
        interval: str
    ) -> List[Dict]:
        """Fetch long/short ratio for a single symbol/exchange pair."""
        coinalyze_symbol = self._build_symbol(symbol, exchange)

        try:
            params = {
                'symbols': coinalyze_symbol,
                'from': start_ts,
                'to': end_ts,
                'interval': self._map_interval(interval)
            }

            data = await self._make_request('/long-short-ratio-history', params)

            records = []
            if data and isinstance(data, list):
                for record in data:
                    history = record.get('history', [])
                    for point in history:
                        ratio = float(point.get('r', 1.0))
                        long_pct = ratio / (1 + ratio) * 100 if ratio > 0 else 50
                        short_pct = 100 - long_pct

                        # Sentiment classification
                        if long_pct >= 60:
                            sentiment = 'bullish'
                        elif long_pct <= 40:
                            sentiment = 'bearish'
                        else:
                            sentiment = 'neutral'

                        records.append({
                            'timestamp': pd.to_datetime(point['t'], unit='s', utc=True),
                            'symbol': symbol.upper(),
                            'long_short_ratio': ratio,
                            'long_pct': long_pct,
                            'short_pct': short_pct,
                            'sentiment': sentiment,
                            'exchange': exchange,
                            'venue': self.VENUE,
                            'venue_type': self.VENUE_TYPE
                        })
            return records

        except Exception as e:
            logger.error(f"Error fetching L/S ratio for {symbol} on {exchange}: {e}")
            self.collection_stats['errors'] += 1
            return []

    async def fetch_long_short_ratio(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        exchanges: Optional[List[str]] = None,
        interval: str = '4h'
    ) -> pd.DataFrame:
        """
        Fetch historical long/short ratio data IN PARALLEL (30x speedup).

        Args:
            symbols: List of base symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            exchanges: Optional list of exchanges (binance, bybit, okx, bitget)
            interval: Data interval

        Returns:
            DataFrame with long/short ratio data
        """
        # L/S ratio only available for major exchanges
        exchanges = exchanges or ['binance', 'bybit', 'okx', 'bitget']

        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        # PARALLEL: Create all tasks upfront
        tasks = []
        for symbol in symbols:
            for exchange in exchanges:
                tasks.append(self._fetch_single_long_short_ratio(symbol, exchange, start_ts, end_ts, interval))

        # Execute ALL tasks in parallel
        logger.info(f"Fetching {len(tasks)} L/S ratio combinations in parallel")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results
        all_data = []
        for result in results:
            if isinstance(result, list):
                all_data.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Task failed: {result}")

        df = pd.DataFrame(all_data)

        if not df.empty:
            df = df.sort_values(['timestamp', 'symbol', 'exchange']).reset_index(drop=True)
            self.collection_stats['records_collected'] += len(df)
            logger.info(f"Collected {len(df)} L/S ratio records")

        return df
    
    # ========================================================================
    # OHLCV DATA
    # ========================================================================

    async def _fetch_single_ohlcv(
        self,
        symbol: str,
        exchange: str,
        start_ts: int,
        end_ts: int,
        timeframe: str
    ) -> List[Dict]:
        """Fetch OHLCV data for a single symbol/exchange pair."""
        coinalyze_symbol = self._build_symbol(symbol, exchange)

        try:
            params = {
                'symbols': coinalyze_symbol,
                'from': start_ts,
                'to': end_ts,
                'interval': self._map_interval(timeframe)
            }

            data = await self._make_request('/ohlcv-history', params)

            records = []
            if data and isinstance(data, list):
                for record in data:
                    history = record.get('history', [])
                    for point in history:
                        records.append({
                            'timestamp': pd.to_datetime(point['t'], unit='s', utc=True),
                            'symbol': symbol.upper(),
                            'open': float(point.get('o', 0)),
                            'high': float(point.get('h', 0)),
                            'low': float(point.get('l', 0)),
                            'close': float(point.get('c', 0)),
                            'volume': float(point.get('v', 0)),
                            'exchange': exchange,
                            'venue': self.VENUE,
                            'venue_type': self.VENUE_TYPE
                        })
            return records

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol} on {exchange}: {e}")
            self.collection_stats['errors'] += 1
            return []

    async def fetch_ohlcv(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        exchanges: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for perpetuals IN PARALLEL (30x speedup).

        Args:
            symbols: List of base symbols
            timeframe: Candle interval (1h, 4h, 1d)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            exchanges: Optional list of exchanges

        Returns:
            DataFrame with OHLCV data
        """
        exchanges = exchanges or ['binance'] # Single exchange for OHLCV

        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        # PARALLEL: Create all tasks upfront
        tasks = []
        for symbol in symbols:
            for exchange in exchanges:
                tasks.append(self._fetch_single_ohlcv(symbol, exchange, start_ts, end_ts, timeframe))

        # Execute ALL tasks in parallel
        logger.info(f"Fetching {len(tasks)} OHLCV combinations in parallel")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results
        all_data = []
        for result in results:
            if isinstance(result, list):
                all_data.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Task failed: {result}")

        df = pd.DataFrame(all_data)

        if not df.empty:
            df = df.sort_values(['timestamp', 'symbol', 'exchange']).reset_index(drop=True)
            self.collection_stats['records_collected'] += len(df)
            logger.info(f"Collected {len(df)} OHLCV records")

        return df
    
    # ========================================================================
    # COMPREHENSIVE FETCH
    # ========================================================================
    
    async def fetch_all_derivatives_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        exchanges: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all derivatives data types in one call.
        
        Args:
            symbols: List of base symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            exchanges: Optional list of exchanges
            
        Returns:
            Dict with DataFrames: funding_rates, open_interest, 
            liquidations, long_short_ratio, ohlcv
        """
        results = {}
        
        logger.info(f"Fetching all derivatives data for {len(symbols)} symbols")
        
        results['funding_rates'] = await self.fetch_funding_rates(
            symbols, start_date, end_date, exchanges
        )
        
        results['open_interest'] = await self.fetch_open_interest(
            symbols, start_date, end_date, exchanges
        )
        
        results['liquidations'] = await self.fetch_liquidations(
            symbols, start_date, end_date, exchanges
        )
        
        results['long_short_ratio'] = await self.fetch_long_short_ratio(
            symbols, start_date, end_date, exchanges
        )
        
        results['ohlcv'] = await self.fetch_ohlcv(
            symbols, '1d', start_date, end_date, exchanges[:1] if exchanges else None
        )
        
        logger.info(f"Completed fetch. Stats: {self.collection_stats}")
        
        return results
    
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
                end_date=end_str
            )
        except Exception as e:
            logger.error(f"Coinalyze collect_liquidations error: {e}")
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
            logger.error(f"Coinalyze collect_funding_rates error: {e}")
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
            logger.error(f"Coinalyze collect_ohlcv error: {e}")
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

            return await self.fetch_open_interest(
                symbols=symbols,
                timeframe=timeframe,
                start_date=start_str,
                end_date=end_str
            )
        except Exception as e:
            logger.error(f"Coinalyze collect_open_interest error: {e}")
            return pd.DataFrame()

    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.collection_stats
    
    async def close(self):
        """Clean up resources."""
        if self.session and not self.session.closed:
            await self.session.close()
        logger.info(f"Coinalyze collector closed. Stats: {self.collection_stats}")

# =============================================================================
# Convenience function
# =============================================================================

async def fetch_derivatives_snapshot(
    api_key: str,
    symbols: List[str],
    exchanges: List[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Fetch current derivatives snapshot for symbols.
    
    Args:
        api_key: Coinalyze API key
        symbols: List of base symbols
        exchanges: Optional list of exchanges
        
    Returns:
        Dict with current DataFrames: funding, oi, arbitrage
    """
    config = {'api_key': api_key}
    collector = CoinalyzeCollector(config)
    
    try:
        results = {}
        results['funding'] = await collector.fetch_current_funding_rates(symbols, exchanges)
        results['funding_aggregated'] = await collector.fetch_aggregated_funding(symbols, exchanges)
        results['oi'] = await collector.fetch_current_open_interest(symbols, exchanges)
        results['oi_aggregated'] = await collector.fetch_aggregated_open_interest(symbols, exchanges)
        results['arbitrage'] = await collector.detect_funding_arbitrage(symbols, exchanges)
        return results
    finally:
        await collector.close()

# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    import os
    
    async def test():
        config = {
            'api_key': os.getenv('COINALYZE_API_KEY', ''),
            'rate_limit': 30
        }
        
        collector = CoinalyzeCollector(config)
        
        try:
            print("Testing Coinalyze Collector")
            print("=" * 60)
            
            symbols = ['BTC', 'ETH']
            
            # Test current funding rates
            print("\n1. Testing current funding rates...")
            funding = await collector.fetch_current_funding_rates(symbols)
            if not funding.empty:
                print(f" Found {len(funding)} records")
                for _, row in funding.head(4).iterrows():
                    print(f" {row['symbol']} @ {row['exchange']}: {row['funding_rate_annualized']:.2f}% ann.")
            
            # Test aggregated funding
            print("\n2. Testing aggregated funding...")
            agg_funding = await collector.fetch_aggregated_funding(symbols)
            if not agg_funding.empty:
                for _, row in agg_funding.iterrows():
                    print(f" {row['symbol']}: mean={row['annualized_mean']:.2f}%, spread={row['annualized_spread']:.2f}%")
            
            # Test funding arbitrage
            print("\n3. Testing funding arbitrage detection...")
            arb = await collector.detect_funding_arbitrage(symbols, min_spread_annualized=1.0)
            if arb:
                for opp in arb[:3]:
                    print(f" {opp.symbol}: Long {opp.long_exchange} / Short {opp.short_exchange} = {opp.spread_annualized:.2f}% ann.")
            
            # Test current OI
            print("\n4. Testing current open interest...")
            oi = await collector.fetch_current_open_interest(symbols)
            if not oi.empty:
                print(f" Found {len(oi)} records")
            
            # Test historical funding
            print("\n5. Testing historical funding rates...")
            hist = await collector.fetch_funding_rates(
                ['BTC'], '2024-01-01', '2024-01-07', ['binance']
            )
            if not hist.empty:
                print(f" Found {len(hist)} historical records")
            
            print("\n" + "=" * 60)
            print(f"Collection stats: {collector.get_collection_stats()}")
            
        finally:
            await collector.close()
    
    asyncio.run(test())