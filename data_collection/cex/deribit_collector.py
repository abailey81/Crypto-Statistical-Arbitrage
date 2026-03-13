"""
Deribit Options Data Collector - Strategy 4 Requirement

validated collector for Deribit cryptocurrency options exchange.
Provides comprehensive options data including full option chains, Greeks,
implied volatility, and historical options pricing.

Supported Data Types:
    - Options instruments (all strikes and expiries)
    - Options ticker data with Greeks (delta, gamma, vega, theta, rho)
    - Implied volatility (mark, bid, ask)
    - Historical options prices and Greeks
    - Options order book (bid/ask spreads)
    - Options trade history
    - Historical volatility (realized vol for underlying)
    - Options open interest

API Documentation:
    - REST API: https://docs.deribit.com/
    - WebSocket: wss://www.deribit.com/ws/api/v2

Rate Limits:
    - Public endpoints: 20 requests/second per IP
    - Burst limit: 100 requests
    - WebSocket: 50 messages/second per connection

Options Specifications:
    - BTC options: Available strikes and weekly/monthly expiries
    - ETH options: Available strikes and weekly/monthly expiries
    - Settlement: European-style, cash-settled
    - Greeks: Calculated using proprietary model (Black-Scholes variant)
    - IV calculation: Mark IV from order book

Statistical Arbitrage Applications (Strategy 4):
    - Volatility surface arbitrage (calendar spreads, butterfly spreads)
    - Put-call parity violations
    - Skew trading (volatility smile arbitrage)
    - Greeks-based hedging and delta-neutral strategies
    - Cross-exchange options arbitrage (Deribit vs others)
    - Implied vs realized volatility trading

Version: 1.0.0
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

from ..base_collector import BaseCollector, CollectionStats
from ..utils.rate_limiter import get_shared_rate_limiter
from ..utils.retry_handler import RetryHandler, safe_float, safe_int

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================

class OptionType(Enum):
    """Option type classification."""
    CALL = 'call'
    PUT = 'put'

class OptionKind(Enum):
    """Deribit instrument kinds."""
    FUTURE = 'future'
    OPTION = 'option'
    SPOT = 'spot'
    FUTURE_COMBO = 'future_combo'
    OPTION_COMBO = 'option_combo'

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class OptionGreeks:
    """
    Option Greeks for sensitivity analysis.

    Greeks measure the sensitivity of option prices to various factors:
    - Delta: Price sensitivity to underlying ($change per $1 underlying move)
    - Gamma: Delta sensitivity to underlying (delta change per $1 move)
    - Vega: Price sensitivity to volatility ($change per 1% vol change)
    - Theta: Time decay ($change per day)
    - Rho: Interest rate sensitivity ($change per 1% rate change)
    """
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    @property
    def is_delta_neutral(self, threshold: float = 0.1) -> bool:
        """Check if position is approximately delta neutral."""
        return abs(self.delta) < threshold

    @property
    def gamma_risk_category(self) -> str:
        """Classify gamma risk level."""
        abs_gamma = abs(self.gamma)
        if abs_gamma > 0.05:
            return 'high'
        elif abs_gamma > 0.01:
            return 'medium'
        else:
            return 'low'

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'delta': self.delta,
            'gamma': self.gamma,
            'vega': self.vega,
            'theta': self.theta,
            'rho': self.rho,
            'is_delta_neutral': self.is_delta_neutral,
            'gamma_risk_category': self.gamma_risk_category,
        }

@dataclass
class ImpliedVolatility:
    """
    Implied volatility data point.

    IV represents market's expectation of future volatility.
    Different IV calculations available:
    - Mark IV: Fair value from order book
    - Bid IV: Implied from best bid
    - Ask IV: Implied from best ask
    """
    mark_iv: float
    bid_iv: Optional[float] = None
    ask_iv: Optional[float] = None

    @property
    def mid_iv(self) -> Optional[float]:
        """Calculate mid IV from bid/ask."""
        if self.bid_iv is not None and self.ask_iv is not None:
            return (self.bid_iv + self.ask_iv) / 2
        return None

    @property
    def iv_spread(self) -> Optional[float]:
        """Calculate bid-ask IV spread."""
        if self.bid_iv is not None and self.ask_iv is not None:
            return self.ask_iv - self.bid_iv
        return None

    @property
    def iv_spread_pct(self) -> Optional[float]:
        """Calculate IV spread as percentage of mark."""
        if self.iv_spread is not None and self.mark_iv > 0:
            return (self.iv_spread / self.mark_iv) * 100
        return None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'mark_iv': self.mark_iv,
            'bid_iv': self.bid_iv,
            'ask_iv': self.ask_iv,
            'mid_iv': self.mid_iv,
            'iv_spread': self.iv_spread,
            'iv_spread_pct': self.iv_spread_pct,
        }

@dataclass
class OptionInstrument:
    """
    Options contract specification.

    Complete specification for a single option contract including
    strike, expiration, type, and trading status.
    """
    instrument_name: str
    currency: str
    underlying: str
    strike: float
    expiration_timestamp: datetime
    option_type: OptionType
    is_active: bool
    creation_timestamp: datetime
    settlement_period: str
    min_trade_amount: float
    tick_size: float

    @property
    def time_to_expiry_days(self) -> float:
        """Days until expiration."""
        return (self.expiration_timestamp - datetime.now(timezone.utc)).total_seconds() / 86400

    @property
    def is_weekly(self) -> bool:
        """Check if weekly expiration."""
        return 'W' in self.instrument_name or self.time_to_expiry_days < 10

    @property
    def is_monthly(self) -> bool:
        """Check if monthly expiration."""
        return not self.is_weekly

    @property
    def moneyness_label(self, underlying_price: float) -> str:
        """Calculate moneyness category."""
        if self.option_type == OptionType.CALL:
            if underlying_price > self.strike * 1.05:
                return 'ITM'
            elif underlying_price < self.strike * 0.95:
                return 'OTM'
            else:
                return 'ATM'
        else: # PUT
            if underlying_price < self.strike * 0.95:
                return 'ITM'
            elif underlying_price > self.strike * 1.05:
                return 'OTM'
            else:
                return 'ATM'

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'instrument_name': self.instrument_name,
            'currency': self.currency,
            'underlying': self.underlying,
            'strike': self.strike,
            'expiration_timestamp': self.expiration_timestamp,
            'option_type': self.option_type.value,
            'is_active': self.is_active,
            'creation_timestamp': self.creation_timestamp,
            'settlement_period': self.settlement_period,
            'min_trade_amount': self.min_trade_amount,
            'tick_size': self.tick_size,
            'time_to_expiry_days': self.time_to_expiry_days,
            'is_weekly': self.is_weekly,
            'is_monthly': self.is_monthly,
        }

@dataclass
class OptionSnapshot:
    """
    Complete option market snapshot combining all data points.

    Provides comprehensive view of option state including price,
    Greeks, IV, and liquidity metrics.
    """
    timestamp: datetime
    instrument_name: str
    underlying_price: float
    underlying_index: str
    strike: float
    option_type: OptionType
    expiration_timestamp: datetime
    mark_price: float
    last_price: Optional[float]
    best_bid_price: Optional[float]
    best_ask_price: Optional[float]
    best_bid_amount: Optional[float]
    best_ask_amount: Optional[float]
    open_interest: float
    volume_24h: float
    greeks: OptionGreeks
    iv: ImpliedVolatility

    @property
    def bid_ask_spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        if self.best_bid_price and self.best_ask_price:
            return self.best_ask_price - self.best_bid_price
        return None

    @property
    def bid_ask_spread_pct(self) -> Optional[float]:
        """Calculate spread as percentage of mark."""
        if self.bid_ask_spread and self.mark_price > 0:
            return (self.bid_ask_spread / self.mark_price) * 100
        return None

    @property
    def moneyness(self) -> float:
        """Calculate moneyness ratio."""
        if self.option_type == OptionType.CALL:
            return self.underlying_price / self.strike
        else: # PUT
            return self.strike / self.underlying_price

    @property
    def time_to_expiry_days(self) -> float:
        """Days until expiration."""
        return (self.expiration_timestamp - self.timestamp).total_seconds() / 86400

    @property
    def is_liquid(self, min_oi: float = 10, min_volume: float = 1) -> bool:
        """Check if option is liquid enough for trading."""
        return self.open_interest >= min_oi or self.volume_24h >= min_volume

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'instrument_name': self.instrument_name,
            'underlying_price': self.underlying_price,
            'underlying_index': self.underlying_index,
            'strike': self.strike,
            'option_type': self.option_type.value,
            'expiration_timestamp': self.expiration_timestamp,
            'mark_price': self.mark_price,
            'last_price': self.last_price,
            'best_bid_price': self.best_bid_price,
            'best_ask_price': self.best_ask_price,
            'best_bid_amount': self.best_bid_amount,
            'best_ask_amount': self.best_ask_amount,
            'open_interest': self.open_interest,
            'volume_24h': self.volume_24h,
            'bid_ask_spread': self.bid_ask_spread,
            'bid_ask_spread_pct': self.bid_ask_spread_pct,
            'moneyness': self.moneyness,
            'time_to_expiry_days': self.time_to_expiry_days,
            'is_liquid': self.is_liquid,
            **{f'greek_{k}': v for k, v in self.greeks.to_dict().items()},
            **{f'iv_{k}': v for k, v in self.iv.to_dict().items()},
        }

# =============================================================================
# Collector Class
# =============================================================================

class DeribitCollector(BaseCollector):
    """
    Deribit options data collector for cryptocurrency derivatives.

    validated implementation with:
    - Rate limiting (20 req/sec)
    - Automatic retry with exponential backoff
    - Comprehensive options data (Greeks, IV, chains)
    - Historical data collection
    - Collection statistics tracking

    Attributes:
        VENUE: Exchange identifier ('deribit')
        VENUE_TYPE: Exchange type ('CEX')
        BASE_URL: API endpoint

    Example:
        >>> config = {'rate_limit': 20, 'timeout': 30}
        >>> async with DeribitCollector(config) as collector:
        ... options = await collector.fetch_options_chain(['BTC'], '2024-01-01', '2024-01-31')
        ... greeks = await collector.fetch_options_greeks(['BTC'])
    """

    VENUE = 'deribit'
    VENUE_TYPE = 'CEX'

    # Collection manager compatibility attributes
    supported_data_types = ['options', 'options_greeks', 'options_iv', 'historical_volatility']
    venue = 'deribit'
    requires_auth = False # Public endpoints available for options data

    BASE_URL = 'https://www.deribit.com/api/v2'
    BASE_URL_TEST = 'https://test.deribit.com/api/v2'

    def __init__(self, config: Dict[str, Any]):
        """Initialize Deribit collector."""
        super().__init__(config)

        rate_limit = config.get('rate_limit', 20) # 20 req/sec
        self.rate_limiter = get_shared_rate_limiter('deribit', rate=rate_limit, per=1.0, burst=min(100, rate_limit * 5))
        self.retry_handler = RetryHandler(max_retries=config.get('max_retries', 3), base_delay=1.0, max_delay=30.0)

        self.timeout = aiohttp.ClientTimeout(total=config.get('timeout', 30))
        self.session: Optional[aiohttp.ClientSession] = None

        # Use test environment if specified
        self.use_test = config.get('use_test', False)
        self.base_url = self.BASE_URL_TEST if self.use_test else self.BASE_URL

        # Load API keys from config or environment (optional for public endpoints)
        self.api_key = config.get('api_key') or os.getenv('DERIBIT_API_KEY', '')
        self.api_secret = config.get('api_secret') or os.getenv('DERIBIT_API_SECRET', '')

        # Caching
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl = config.get('cache_ttl', 60)

        # Semaphore to limit concurrent HTTP requests (prevents rate limit exhaustion)
        self._request_sem = asyncio.Semaphore(2)

        self.collection_stats = {
            'records_collected': 0, 'api_calls': 0,
            'cache_hits': 0, 'errors': 0, 'rate_limit_hits': 0
        }

        logger.info(f"Initialized Deribit collector (rate_limit={rate_limit} req/sec)")

    async def __aenter__(self):
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            headers = {
                'User-Agent': 'CryptoStatArb/2.0',
                'Accept-Encoding': 'gzip, deflate',
                'Content-Type': 'application/json',
            }
            connector = aiohttp.TCPConnector(
                limit=100, limit_per_host=30,
                ttl_dns_cache=300, force_close=False
            )
            self.session = aiohttp.ClientSession(timeout=self.timeout, headers=headers, connector=connector)
        return self.session

    async def close(self) -> None:
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
        self._cache.clear()
        logger.info(f"Deribit collector closed. Stats: {self.collection_stats}")

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
        """Set cached value with timestamp."""
        self._cache[key] = (datetime.utcnow(), value)

    async def _make_request(self, method: str, params: Optional[Dict] = None) -> Any:
        """Make rate-limited HTTP request to Deribit API."""
        async with self._request_sem:
            cache_key = f"{method}_{hash(frozenset((params or {}).items()))}"
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

            session = await self._get_session()
            url = f"{self.base_url}/public/{method}"

            async def _request():
                await self.rate_limiter.acquire()

                async with session.get(url, params=params) as response:
                    if response.status == 429:
                        self.collection_stats['rate_limit_hits'] += 1
                        retry_after = int(response.headers.get('Retry-After', 5))
                        logger.warning(f"Deribit rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        raise aiohttp.ClientResponseError(response.request_info, response.history, status=429)

                    response.raise_for_status()
                    data = await response.json()

                    # Deribit returns data in 'result' field
                    if 'result' in data:
                        return data['result']
                    elif 'error' in data:
                        error_msg = data['error'].get('message', 'Unknown error')
                        logger.error(f"Deribit API error: {error_msg}")
                        return None
                    return data

            try:
                result = await self.retry_handler.execute(_request)
                self.collection_stats['api_calls'] += 1
                if result is not None:
                    self._set_cached(cache_key, result)
                return result
            except Exception as e:
                self.collection_stats['errors'] += 1
                logger.error(f"Deribit request error for {method}: {e}")
                return None

    async def fetch_instruments(self, currency: str = 'BTC', kind: str = 'option', expired: bool = False) -> List[OptionInstrument]:
        """
        Fetch all available options instruments for a currency.

        Args:
            currency: 'BTC' or 'ETH'
            kind: 'option', 'future', or 'spot'
            expired: Include expired contracts

        Returns:
            List of OptionInstrument objects
        """
        params = {'currency': currency.upper(), 'kind': kind, 'expired': expired}
        data = await self._make_request('get_instruments', params)

        if not data:
            return []

        instruments = []
        for item in data:
            if item['kind'] != 'option':
                continue

            # Parse instrument name (e.g., BTC-25DEC24-50000-C)
            parts = item['instrument_name'].split('-')
            if len(parts) != 4:
                continue

            _, exp_str, strike_str, opt_type = parts
            option_type = OptionType.CALL if opt_type == 'C' else OptionType.PUT

            instrument = OptionInstrument(
                instrument_name=item['instrument_name'],
                currency=currency.upper(),
                underlying=item['base_currency'],
                strike=float(strike_str),
                expiration_timestamp=pd.to_datetime(item['expiration_timestamp'], unit='ms', utc=True),
                option_type=option_type,
                is_active=item['is_active'],
                creation_timestamp=pd.to_datetime(item['creation_timestamp'], unit='ms', utc=True),
                settlement_period=item['settlement_period'],
                min_trade_amount=safe_float(item.get('min_trade_amount', 0)),
                tick_size=safe_float(item.get('tick_size', 0))
            )
            instruments.append(instrument)

        logger.info(f"Fetched {len(instruments)} {currency} options instruments")
        self.collection_stats['records_collected'] += len(instruments)
        return instruments

    async def fetch_ticker(self, instrument_name: str) -> Optional[OptionSnapshot]:
        """
        Fetch ticker data for a single option instrument including Greeks and IV.

        Args:
            instrument_name: Option instrument name (e.g., BTC-25DEC24-50000-C)

        Returns:
            OptionSnapshot with complete market data
        """
        params = {'instrument_name': instrument_name}
        data = await self._make_request('ticker', params)

        if not data:
            return None

        # Parse instrument name for metadata
        parts = instrument_name.split('-')
        if len(parts) != 4:
            return None

        currency, exp_str, strike_str, opt_type = parts
        option_type = OptionType.CALL if opt_type == 'C' else OptionType.PUT
        strike = float(strike_str)

        # Parse expiration date (format: 25DEC24)
        exp_timestamp = pd.to_datetime(data.get('expiration_timestamp', 0), unit='ms', utc=True) if 'expiration_timestamp' in data else datetime.now(timezone.utc)

        # Extract Greeks
        greeks_data = data.get('greeks', {})
        greeks = OptionGreeks(
            delta=safe_float(greeks_data.get('delta', 0)),
            gamma=safe_float(greeks_data.get('gamma', 0)),
            vega=safe_float(greeks_data.get('vega', 0)),
            theta=safe_float(greeks_data.get('theta', 0)),
            rho=safe_float(greeks_data.get('rho', 0))
        )

        # Extract IV
        iv = ImpliedVolatility(
            mark_iv=safe_float(data.get('mark_iv', 0)),
            bid_iv=safe_float(data.get('bid_iv')) if 'bid_iv' in data else None,
            ask_iv=safe_float(data.get('ask_iv')) if 'ask_iv' in data else None
        )

        snapshot = OptionSnapshot(
            timestamp=pd.to_datetime(data['timestamp'], unit='ms', utc=True),
            instrument_name=instrument_name,
            underlying_price=safe_float(data.get('underlying_price', 0)),
            underlying_index=data.get('underlying_index', f'{currency}_USD'),
            strike=strike,
            option_type=option_type,
            expiration_timestamp=exp_timestamp,
            mark_price=safe_float(data.get('mark_price', 0)),
            last_price=safe_float(data.get('last_price')) if data.get('last_price') else None,
            best_bid_price=safe_float(data.get('best_bid_price')) if data.get('best_bid_price') else None,
            best_ask_price=safe_float(data.get('best_ask_price')) if data.get('best_ask_price') else None,
            best_bid_amount=safe_float(data.get('best_bid_amount')) if data.get('best_bid_amount') else None,
            best_ask_amount=safe_float(data.get('best_ask_amount')) if data.get('best_ask_amount') else None,
            open_interest=safe_float(data.get('open_interest', 0)),
            volume_24h=safe_float(data.get('stats', {}).get('volume', 0)),
            greeks=greeks,
            iv=iv
        )

        self.collection_stats['records_collected'] += 1
        return snapshot

    async def _fetch_options_chain_single_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        include_expired: bool = False
    ) -> Optional[List[Dict]]:
        """Helper to process single symbol for options chain."""
        try:
            logger.info(f"Fetching Deribit options chain for {symbol}")

            # Get all instruments for this currency
            instruments = await self.fetch_instruments(currency=symbol, kind='option', expired=include_expired)

            # Filter by expiration date range
            start_dt = pd.to_datetime(start_date, utc=True)
            end_dt = pd.to_datetime(end_date, utc=True)

            filtered_instruments = [
                inst for inst in instruments
                if start_dt <= inst.expiration_timestamp <= end_dt
            ]

            logger.info(f"Found {len(filtered_instruments)} {symbol} options in date range")

            # Fetch ticker data for each instrument (with rate limiting)
            tasks = []
            for inst in filtered_instruments:
                tasks.append(self.fetch_ticker(inst.instrument_name))

            # Process in batches to avoid overwhelming the API
            symbol_data = []
            batch_size = 50
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i+batch_size]
                results = await asyncio.gather(*batch, return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception):
                        logger.debug(f"Failed to fetch ticker: {result}")
                    elif result is not None:
                        snapshot_dict = result.to_dict()
                        snapshot_dict['venue'] = self.VENUE
                        snapshot_dict['venue_type'] = self.VENUE_TYPE
                        snapshot_dict['symbol'] = symbol
                        symbol_data.append(snapshot_dict)

                # Small delay between batches
                if i + batch_size < len(tasks):
                    await asyncio.sleep(0.5)

            return symbol_data
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return None

    async def fetch_options_chain(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        include_expired: bool = False
    ) -> pd.DataFrame:
        """
        Fetch complete options chain data for symbols.

        Args:
            symbols: List of underlying assets (e.g., ['BTC', 'ETH'])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_expired: Include expired contracts

        Returns:
            DataFrame with options chain data including Greeks and IV
        """
        # Parallelize symbol processing
        tasks = [
            self._fetch_options_chain_single_symbol(symbol, start_date, end_date, include_expired)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine all data
        all_data = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch options chain: {result}")
            elif result is not None:
                all_data.extend(result)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        return df.sort_values(['timestamp', 'symbol', 'expiration_timestamp', 'strike']).reset_index(drop=True)

    async def fetch_historical_volatility(self, currency: str = 'BTC') -> pd.DataFrame:
        """
        Fetch historical volatility for underlying asset.

        Args:
            currency: 'BTC' or 'ETH'

        Returns:
            DataFrame with historical volatility data
        """
        params = {'currency': currency.upper()}
        data = await self._make_request('get_historical_volatility', params)

        if not data:
            return pd.DataFrame()

        records = []
        for item in data:
            records.append({
                'timestamp': pd.to_datetime(item[0], unit='ms', utc=True),
                'volatility': safe_float(item[1]),
                'currency': currency.upper(),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE
            })

        self.collection_stats['records_collected'] += len(records)
        return pd.DataFrame(records)

    async def collect_options(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect options data - wrapper for fetch_options_chain().

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

            include_expired = kwargs.get('include_expired', False)

            return await self.fetch_options_chain(
                symbols=symbols,
                start_date=start_str,
                end_date=end_str,
                include_expired=include_expired
            )
        except Exception as e:
            logger.error(f"Deribit collect_options error: {e}")
            return pd.DataFrame()

    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.collection_stats.copy()

    def reset_collection_stats(self):
        """Reset collection statistics."""
        self.collection_stats = {
            'records_collected': 0, 'api_calls': 0,
            'cache_hits': 0, 'errors': 0, 'rate_limit_hits': 0
        }

async def test_deribit_collector():
    """Test Deribit collector functionality."""
    config = {'rate_limit': 20, 'timeout': 30}

    async with DeribitCollector(config) as collector:
        print("=" * 60)
        print("Deribit Options Collector Test")
        print("=" * 60)

        # Test 1: Fetch instruments
        instruments = await collector.fetch_instruments(currency='BTC', kind='option')
        print(f"\n1. Found {len(instruments)} BTC options instruments")
        if instruments:
            print(f" Sample: {instruments[0].instrument_name}")

        # Test 2: Fetch ticker for one option
        if instruments:
            snapshot = await collector.fetch_ticker(instruments[0].instrument_name)
            if snapshot:
                print(f"\n2. Sample option snapshot:")
                print(f" Instrument: {snapshot.instrument_name}")
                print(f" Mark Price: ${snapshot.mark_price:.4f}")
                print(f" Mark IV: {snapshot.iv.mark_iv:.2%}")
                print(f" Delta: {snapshot.greeks.delta:.4f}")
                print(f" Gamma: {snapshot.greeks.gamma:.6f}")

        # Test 3: Fetch historical volatility
        hist_vol = await collector.fetch_historical_volatility('BTC')
        if not hist_vol.empty:
            print(f"\n3. Historical volatility: {len(hist_vol)} records")
            print(f" Latest: {hist_vol.iloc[-1]['volatility']:.2%}")

        print(f"\nStats: {collector.get_collection_stats()}")

if __name__ == '__main__':
    asyncio.run(test_deribit_collector())
