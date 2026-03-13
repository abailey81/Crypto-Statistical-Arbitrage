"""
Drift Protocol Data Collector - Solana's Largest Perpetual DEX

validated collector for Drift Protocol, a decentralized perpetual
futures exchange on Solana offering up to 20x leverage on over 50 markets.

===============================================================================
PROTOCOL OVERVIEW
===============================================================================

Drift is a decentralized exchange on Solana that combines:
    - Central Limit Order Book (CLOB): Limit order matching
    - Virtual AMM (vAMM): Liquidity backstop using virtual reserves
    - Cross-margin: Unified collateral across all positions
    - Just-In-Time (JIT) Liquidity: Makers can fill orders via auction

Key Characteristics:
    - Hourly funding rate settlements
    - Multi-collateral support (USDC, SOL, etc.)
    - Sub-second order execution (Solana TPS)
    - Insurance fund for socialized losses
    - Free public API (no authentication required)

===============================================================================
API DOCUMENTATION
===============================================================================

Endpoints (Public, No Auth Required):
    - Data API: https://data.api.drift.trade
    - Mainnet Gateway: https://mainnet-beta.api.drift.trade
    - Historical: https://data.api.drift.trade/market-history

Data Format:
    - Funding rates returned in quote/base units (divide by oracle for %)
    - Timestamps in Unix seconds
    - Market indices map to specific trading pairs

Rate Limits:
    - Data API: ~100 requests/minute
    - Gateway API: ~60 requests/minute

===============================================================================
MARKET INDICES
===============================================================================

Perpetual Markets:
    - 0: SOL-PERP
    - 1: BTC-PERP
    - 2: ETH-PERP
    - 3: APT-PERP
    - 4: 1MBONK-PERP
    - 5: MATIC-PERP (deprecated)
    - 6: ARB-PERP
    ... (50+ markets)

===============================================================================
FUNDING RATE MECHANISM
===============================================================================

Drift uses hourly funding with velocity-based adjustments:
    Base Rate = (Mark Price - Oracle Price) / Oracle Price

Funding payments:
    - Settled every hour (vs 8h on CEX)
    - Long pays short when mark > oracle
    - Short pays long when mark < oracle
    - Rate influenced by open interest imbalance

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

1. Cross-Venue Funding Arbitrage:
   - Compare Drift hourly funding to CEX 8h funding
   - Identify persistent rate differentials
   - Solana speed enables faster position adjustments

2. Multi-Chain Arbitrage:
   - Drift (Solana) vs Hyperliquid (Arbitrum) vs dYdX (Cosmos)
   - Chain-specific gas/fee considerations

3. Velocity-Based Strategy:
   - Track funding rate velocity changes
   - Position before major rate moves

Version: 1.0.0
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..base_collector import BaseCollector
from ..utils.rate_limiter import get_shared_rate_limiter
from ..utils.retry_handler import RetryHandler, safe_float, safe_int

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================

class FundingTrend(Enum):
    """Funding rate trend classification."""
    VERY_POSITIVE = 'very_positive' # > 0.05% hourly
    POSITIVE = 'positive' # 0.01% - 0.05%
    NEUTRAL = 'neutral' # -0.01% to 0.01%
    NEGATIVE = 'negative' # -0.05% to -0.01%
    VERY_NEGATIVE = 'very_negative' # < -0.05%

class MarketStatus(Enum):
    """Market trading status."""
    ACTIVE = 'active'
    PAUSED = 'paused'
    REDUCE_ONLY = 'reduce_only'
    SETTLEMENT = 'settlement'

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class DriftFundingRate:
    """Drift funding rate data with analytics."""
    timestamp: datetime
    symbol: str
    market_index: int
    funding_rate: float # Hourly rate (decimal, e.g., 0.0001 = 0.01%)
    oracle_price: float
    mark_price: float
    twap_price: float

    @property
    def annualized_rate(self) -> float:
        """Annualized funding rate (hourly * 8760 * 100 for percentage)."""
        return self.funding_rate * 8760 * 100

    @property
    def daily_rate(self) -> float:
        """Daily funding rate (hourly * 24 * 100 for percentage)."""
        return self.funding_rate * 24 * 100

    @property
    def eight_hour_equivalent(self) -> float:
        """8-hour equivalent rate for CEX comparison (%)."""
        return self.funding_rate * 8 * 100

    @property
    def trend(self) -> FundingTrend:
        """Classify funding trend."""
        rate_pct = self.funding_rate * 100
        if rate_pct > 0.05:
            return FundingTrend.VERY_POSITIVE
        elif rate_pct > 0.01:
            return FundingTrend.POSITIVE
        elif rate_pct > -0.01:
            return FundingTrend.NEUTRAL
        elif rate_pct > -0.05:
            return FundingTrend.NEGATIVE
        else:
            return FundingTrend.VERY_NEGATIVE

    @property
    def basis_bps(self) -> float:
        """Basis in basis points (mark vs oracle)."""
        if self.oracle_price > 0:
            return (self.mark_price - self.oracle_price) / self.oracle_price * 10000
        return 0

    @property
    def is_contango(self) -> bool:
        """Check if market is in contango (mark > oracle)."""
        return self.mark_price > self.oracle_price

    @property
    def is_arbitrage_opportunity(self) -> bool:
        """Check if funding presents arbitrage opportunity (>0.1% daily)."""
        return abs(self.daily_rate) > 0.1

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'market_index': self.market_index,
            'funding_rate': self.funding_rate,
            'funding_rate_hourly_pct': self.funding_rate * 100,
            'annualized_rate': self.annualized_rate,
            'daily_rate': self.daily_rate,
            'eight_hour_equivalent': self.eight_hour_equivalent,
            'oracle_price': self.oracle_price,
            'mark_price': self.mark_price,
            'twap_price': self.twap_price,
            'basis_bps': self.basis_bps,
            'trend': self.trend.value,
            'is_contango': self.is_contango,
            'is_arbitrage_opportunity': self.is_arbitrage_opportunity,
        }

@dataclass
class DriftOHLCV:
    """Drift OHLCV candle data."""
    timestamp: datetime
    symbol: str
    market_index: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    num_trades: int

    @property
    def typical_price(self) -> float:
        """Typical price (H+L+C)/3."""
        return (self.high + self.low + self.close) / 3

    @property
    def range_pct(self) -> float:
        """Price range as percentage."""
        return (self.high - self.low) / self.low * 100 if self.low > 0 else 0

    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish."""
        return self.close > self.open

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'market_index': self.market_index,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'num_trades': self.num_trades,
            'typical_price': self.typical_price,
            'range_pct': self.range_pct,
            'is_bullish': self.is_bullish,
        }

@dataclass
class DriftOpenInterest:
    """Drift open interest data."""
    timestamp: datetime
    symbol: str
    market_index: int
    open_interest: float
    open_interest_usd: float
    long_interest: float
    short_interest: float

    @property
    def long_short_ratio(self) -> float:
        """Long/short ratio."""
        return self.long_interest / self.short_interest if self.short_interest > 0 else 0

    @property
    def imbalance_pct(self) -> float:
        """Position imbalance percentage (how far from 50/50)."""
        total = self.long_interest + self.short_interest
        if total == 0:
            return 0
        long_pct = self.long_interest / total * 100
        return abs(long_pct - 50) * 2

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'market_index': self.market_index,
            'open_interest': self.open_interest,
            'open_interest_usd': self.open_interest_usd,
            'long_interest': self.long_interest,
            'short_interest': self.short_interest,
            'long_short_ratio': self.long_short_ratio,
            'imbalance_pct': self.imbalance_pct,
        }

# =============================================================================
# Collector Class
# =============================================================================

class DriftCollector(BaseCollector):
    """
    Drift Protocol data collector.

    validated implementation for Solana's largest perpetual DEX.
    Supports hourly funding rates, OHLCV candles, and open interest.

    Features:
        - Hourly funding rate data (free public API)
        - OHLCV candles for all markets
        - Open interest tracking
        - Market statistics

    Attributes:
        VENUE: Protocol identifier ('drift')
        VENUE_TYPE: Protocol type ('HYBRID')

    Example:
        >>> config = {'rate_limit': 30}
        >>> async with DriftCollector(config) as collector:
        ... funding = await collector.fetch_funding_rates(
        ... ['BTC', 'ETH', 'SOL'], '2024-01-01', '2024-01-31'
        ... )
        ... ohlcv = await collector.fetch_ohlcv(
        ... ['BTC'], '1h', '2024-01-01', '2024-01-31'
        ... )
    """

    VENUE = 'drift'
    VENUE_TYPE = 'HYBRID'

    # API endpoints
    DATA_API_URL = 'https://data.api.drift.trade'
    MAINNET_API_URL = 'https://mainnet-beta.api.drift.trade'

    # Market index mapping: symbol -> market_index
    MARKET_MAP = {
        'SOL': 0,
        'BTC': 1,
        'ETH': 2,
        'APT': 3,
        '1MBONK': 4,
        'ARB': 6,
        'DOGE': 7,
        'BNB': 8,
        'SUI': 9,
        'PEPE': 10,
        'OP': 11,
        'RNDR': 12,
        'XRP': 13,
        'HNT': 14,
        'INJ': 15,
        'LINK': 16,
        'MATIC': 17, # Deprecated
        'AVAX': 18,
        'TIA': 19,
        'JTO': 20,
        'SEI': 21,
        'PYTH': 22,
        'JUP': 23,
        'WIF': 24,
        'STRK': 25,
    }

    # Reverse mapping for convenience
    INDEX_TO_SYMBOL = {v: k for k, v in MARKET_MAP.items()}

    GRANULARITY_MAP = {
        '1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400, '1d': 86400
    }

    def __init__(self, config: Optional[Dict] = None):
        """Initialize Drift collector."""
        config = config or {}
        super().__init__(config)

        rate_limit = config.get('rate_limit', 30)
        # Use shared rate limiter to avoid re-initialization overhead
        self.rate_limiter = get_shared_rate_limiter('drift', rate=rate_limit, per=60.0, burst=5)
        self.retry_handler = RetryHandler(max_retries=5, base_delay=3.0, max_delay=90.0)

        # Concurrency limiter - higher limit for faster collection
        self._concurrency_limit = asyncio.Semaphore(5)

        self.timeout = aiohttp.ClientTimeout(total=60) # Increased timeout for slow responses
        self.session: Optional[aiohttp.ClientSession] = None

        self.collection_stats = {
            'records_collected': 0, 'api_calls': 0, 'errors': 0
        }

        # Flag to log OHLCV unavailability only once
        self._ohlcv_logged = False

        self._request_sem = asyncio.Semaphore(2)

        logger.info(f"Initialized Drift collector (rate_limit={rate_limit}/min, max_concurrent=5, timeout=60s)")

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
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self.session

    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
        logger.info(f"Drift collector closed. Stats: {self.collection_stats}")

    async def _query_api(
        self, endpoint: str, params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Query Drift Data API."""
        async with self._request_sem:
            url = f"{self.DATA_API_URL}{endpoint}"
            session = await self._get_session()
            acquire_result = await self.rate_limiter.acquire(timeout=120.0)
            if hasattr(acquire_result, 'acquired') and not acquire_result.acquired:
                logger.debug(f"Drift rate limiter timeout for {endpoint}")
                return None

            try:
                async with session.get(url, params=params) as resp:
                    self.collection_stats['api_calls'] += 1
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 404:
                        # Endpoint not found - may be deprecated or incorrect path
                        logger.debug(f"Drift API endpoint not found: {endpoint}")
                        return None
                    elif resp.status == 400 or resp.status == 500:
                        # Bad request or server error - log but don't spam errors
                        text = await resp.text()
                        logger.debug(f"Drift API {resp.status}: {endpoint} - {text[:100]}")
                        return None
                    else:
                        text = await resp.text()
                        logger.warning(f"HTTP {resp.status} from Drift: {text[:200]}")
                        return None
            except asyncio.TimeoutError:
                logger.debug(f"Drift API timeout: {endpoint}")
                return None
            except Exception as e:
                logger.debug(f"Error querying Drift API: {e}")
                self.collection_stats['errors'] += 1
                return None

    async def _get_markets(self) -> List[Dict]:
        """Get list of all available markets."""
        data = await self._query_api('/markets')
        if data and 'markets' in data:
            return data['markets']
        return []

    async def _fetch_single_funding_rates(
        self, symbol: str, start_ts: int, end_ts: int
    ) -> List[Dict]:
        """Fetch funding rates for a single symbol (internal helper for parallelization)."""
        async with self._concurrency_limit:
            try:
                symbol_upper = symbol.upper()
                if symbol_upper not in self.MARKET_MAP:
                    logger.warning(f"Symbol {symbol} not supported on Drift")
                    return []

                market_index = self.MARKET_MAP[symbol_upper]
                logger.info(f"Fetching Drift funding for {symbol} (market_index={market_index})")

                all_data = []

                # First try the detailed fundingRates endpoint
                params = {
                    'marketIndex': market_index,
                    'marketType': 'perp',
                }

                data = await self._query_api('/fundingRates', params)

                if data and 'fundingRates' in data:
                    rates = data['fundingRates']
                    for rate in rates:
                        try:
                            timestamp = int(rate.get('ts', 0))
                            if timestamp < start_ts or timestamp > end_ts:
                                continue

                            # Drift returns fundingRate in a scaled format
                            # Need to divide by oraclePriceTwap to get percentage
                            raw_funding = safe_float(rate.get('fundingRate', 0))
                            oracle_twap = safe_float(rate.get('oraclePriceTwap', 1e10), default=1e10)

                            # Convert to decimal funding rate
                            if oracle_twap > 0:
                                funding_rate = raw_funding / oracle_twap
                            else:
                                funding_rate = 0

                            dfr = DriftFundingRate(
                                timestamp=datetime.fromtimestamp(timestamp, tz=timezone.utc),
                                symbol=symbol_upper,
                                market_index=market_index,
                                funding_rate=funding_rate,
                                oracle_price=oracle_twap / 1e6, # Scale down
                                mark_price=safe_float(rate.get('markPriceTwap', 0)) / 1e6,
                                twap_price=oracle_twap / 1e6,
                            )

                            all_data.append({
                                **dfr.to_dict(),
                                'venue': self.VENUE,
                                'venue_type': self.VENUE_TYPE,
                                'funding_interval': 'hourly',
                                'chain': 'solana'
                            })
                        except Exception as e:
                            logger.debug(f"Error parsing funding rate: {e}")
                            continue
                else:
                    # Fallback to rateHistory endpoint (simpler format)
                    data = await self._query_api('/rateHistory', {'marketIndex': market_index})

                    if data and data.get('success') and 'data' in data:
                        for rate_entry in data['data']:
                            try:
                                timestamp = int(rate_entry[0])
                                if timestamp < start_ts or timestamp > end_ts:
                                    continue

                                # rateHistory returns rate as percentage string
                                funding_rate = float(rate_entry[1]) / 100 # Convert percentage to decimal

                                dfr = DriftFundingRate(
                                    timestamp=datetime.fromtimestamp(timestamp, tz=timezone.utc),
                                    symbol=symbol_upper,
                                    market_index=market_index,
                                    funding_rate=funding_rate,
                                    oracle_price=0, # Not available in this endpoint
                                    mark_price=0,
                                    twap_price=0,
                                )

                                all_data.append({
                                    **dfr.to_dict(),
                                    'venue': self.VENUE,
                                    'venue_type': self.VENUE_TYPE,
                                    'funding_interval': 'hourly',
                                    'chain': 'solana'
                                })
                            except Exception as e:
                                logger.debug(f"Error parsing rate history: {e}")
                                continue

                    return all_data

            except Exception as e:
                logger.error(f"Error fetching funding rates for {symbol}: {e}")
                return []

    async def fetch_funding_rates(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates from Drift.

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH', 'SOL'])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with hourly funding rate data
        """
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        # Parallelize funding rate fetching with timeout to prevent hangs
        tasks = [self._fetch_single_funding_rates(symbol, start_ts, end_ts) for symbol in symbols]
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=120.0 # 2 minute timeout for all funding rate fetches
            )
        except asyncio.TimeoutError:
            logger.warning("Drift funding rates gather timed out after 120s")
            results = []

        # Filter and flatten results
        all_data = []
        for result in results:
            if isinstance(result, list):
                all_data.extend(result)

        self.collection_stats['records_collected'] += len(all_data)

        df = pd.DataFrame(all_data)
        if not df.empty:
            df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        return df

    async def _fetch_single_ohlcv(
        self, symbol: str, timeframe: str, start_ts: int, end_ts: int,
        resolution: int, contract_type: str
    ) -> List[Dict]:
        """
        Fetch OHLCV data for a single symbol.

        NOTE: Drift Protocol Data API does NOT provide a public candles/OHLCV endpoint.
        The API only supports: /contracts, /fundingRates, /rateHistory, /auctionParams.
        S3 historical data was deprecated in January 2025.

        This method returns empty list as OHLCV data is not available from Drift.
        For price data, use funding rates which include oracle/mark prices.
        """
        # Drift API does not support OHLCV/candles endpoint
        # Returning empty immediately to avoid unnecessary API calls
        return []

    async def fetch_ohlcv(
        self, symbols: List[str], timeframe: str,
        start_date: str, end_date: str, contract_type: str = 'perp'
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles from Drift.

        NOTE: Drift Protocol Data API does NOT provide a public candles/OHLCV endpoint.
        This method returns an empty DataFrame. For price data, use fetch_funding_rates()
        which includes oracle_price and mark_price fields.

        The Drift Data API only supports:
        - /contracts (current market info)
        - /fundingRates (historical funding with prices)
        - /rateHistory (simplified rate history)
        - /auctionParams (auction parameters)

        Args:
            symbols: List of symbols
            timeframe: '1m', '5m', '15m', '1h', '4h', '1d'
            start_date: Start date
            end_date: End date
            contract_type: 'perp' (default)

        Returns:
            Empty DataFrame (OHLCV not supported by Drift API)
        """
        # Only log once to avoid spam
        if not self._ohlcv_logged:
            logger.info(
                "Drift OHLCV: API does not support candles endpoint. "
                "Use funding_rates for price data (oracle/mark prices)."
            )
            self._ohlcv_logged = True
        return pd.DataFrame()

    async def fetch_open_interest(
        self, symbols: List[str], start_date: str = None, end_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch open interest data from Drift via /contracts endpoint.

        NOTE: Drift API does NOT have a historical /openInterest endpoint.
        The /contracts endpoint provides CURRENT open interest only.
        Historical OI would require external data sources or on-chain indexing.

        Args:
            symbols: List of symbols to fetch OI for
            start_date: Ignored (API only returns current data)
            end_date: Ignored (API only returns current data)

        Returns:
            DataFrame with current open interest snapshot
        """
        # Drift /openInterest endpoint does NOT exist
        # Use /contracts which includes current open_interest field
        data = await self._query_api('/contracts')

        if not data or 'contracts' not in data:
            logger.debug("Drift: No contracts data available for OI")
            return pd.DataFrame()

        # Normalize symbols for matching
        symbols_upper = [s.upper() for s in symbols]

        all_data = []
        for contract in data.get('contracts', []):
            try:
                base_currency = contract.get('base_currency', '')

                # Filter by requested symbols
                if base_currency.upper() not in symbols_upper:
                    continue

                market_index = contract.get('contract_index', 0)
                oi_value = safe_float(contract.get('open_interest', 0))
                index_price = safe_float(contract.get('index_price', 0))

                # Calculate USD value
                oi_usd = oi_value * index_price if index_price > 0 else 0

                all_data.append({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'symbol': base_currency.upper(),
                    'market_index': market_index,
                    'open_interest': oi_value,
                    'open_interest_usd': oi_usd,
                    'long_interest': 0, # Not available from /contracts
                    'short_interest': 0, # Not available from /contracts
                    'long_short_ratio': 0,
                    'imbalance_pct': 0,
                    'index_price': index_price,
                    'funding_rate': safe_float(contract.get('funding_rate', 0)),
                    'venue': self.VENUE,
                    'venue_type': self.VENUE_TYPE,
                    'chain': 'solana',
                    'data_note': 'current_snapshot_only'
                })
            except Exception as e:
                logger.debug(f"Error parsing contract OI: {e}")
                continue

        self.collection_stats['records_collected'] += len(all_data)

        df = pd.DataFrame(all_data)
        if not df.empty:
            df = df.sort_values('symbol').reset_index(drop=True)
            logger.info(f"Drift OI: Fetched {len(df)} symbols from /contracts")

        return df

    async def _fetch_single_liquidations(
        self, symbol: str, start_ts: int, end_ts: int
    ) -> List[Dict]:
        """Fetch liquidation events for a single symbol (internal helper for parallelization)."""
        async with self._concurrency_limit:
            try:
                symbol_upper = symbol.upper()
                if symbol_upper not in self.MARKET_MAP:
                    return []

                market_index = self.MARKET_MAP[symbol_upper]
                logger.info(f"Fetching Drift liquidations for {symbol}")

                params = {
                    'marketIndex': market_index,
                    'marketType': 'perp',
                    'startTime': start_ts,
                    'endTime': end_ts,
                }

                data = await self._query_api('/liquidations', params)

                if not data:
                    return []

                liquidations = data.get('liquidations', data.get('data', []))

                all_data = []
                for liq in liquidations:
                    try:
                        timestamp = liq.get('ts', liq.get('timestamp', 0))
                        if timestamp > 1e12:
                            timestamp = timestamp / 1000

                        all_data.append({
                            'timestamp': datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                            'symbol': symbol_upper,
                            'market_index': market_index,
                            'user': liq.get('user', liq.get('userAccount', '')),
                            'liquidator': liq.get('liquidator', ''),
                            'base_asset_amount': float(liq.get('baseAssetAmount', 0)),
                            'quote_asset_amount': float(liq.get('quoteAssetAmount', 0)),
                            'liquidation_price': float(liq.get('liquidationPrice', liq.get('price', 0))),
                            'pnl': float(liq.get('pnl', 0)),
                            'is_long': liq.get('isLong', liq.get('direction', 'long') == 'long'),
                            'venue': self.VENUE,
                            'venue_type': self.VENUE_TYPE,
                            'chain': 'solana',
                        })
                    except Exception as e:
                        logger.debug(f"Error parsing liquidation: {e}")
                        continue

                return all_data

            except Exception as e:
                logger.error(f"Error fetching liquidations for {symbol}: {e}")
                return []

    async def fetch_liquidations(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch liquidation events from Drift."""
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        # Parallelize liquidation fetching with timeout to prevent hangs
        tasks = [self._fetch_single_liquidations(symbol, start_ts, end_ts) for symbol in symbols]
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=120.0 # 2 minute timeout for all liquidation fetches
            )
        except asyncio.TimeoutError:
            logger.warning("Drift liquidations gather timed out after 120s")
            results = []

        # Filter and flatten results
        all_data = []
        for result in results:
            if isinstance(result, list):
                all_data.extend(result)

        self.collection_stats['records_collected'] += len(all_data)

        df = pd.DataFrame(all_data)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    async def get_current_markets(self) -> pd.DataFrame:
        """Get current market statistics for all perpetual markets."""
        data = await self._query_api('/contracts')

        if not data or 'contracts' not in data:
            return pd.DataFrame()

        records = []
        for contract in data.get('contracts', []):
            try:
                market_index = contract.get('contract_index')
                ticker = contract.get('ticker_id', '')
                base_currency = contract.get('base_currency', '')
                symbol = base_currency if base_currency else ticker.replace('-PERP', '')

                records.append({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'symbol': symbol,
                    'market_index': market_index,
                    'ticker_id': ticker,
                    'last_price': safe_float(contract.get('last_price', 0)),
                    'index_price': safe_float(contract.get('index_price', 0)),
                    'high_24h': safe_float(contract.get('high', 0)),
                    'low_24h': safe_float(contract.get('low', 0)),
                    'base_volume': safe_float(contract.get('base_volume', 0)),
                    'quote_volume': safe_float(contract.get('quote_volume', 0)),
                    'open_interest': safe_float(contract.get('open_interest', 0)),
                    'funding_rate': safe_float(contract.get('funding_rate', 0)),
                    'next_funding_rate': safe_float(contract.get('next_funding_rate', 0)),
                    'next_funding_timestamp': contract.get('next_funding_rate_timestamp'),
                    'venue': self.VENUE,
                    'venue_type': self.VENUE_TYPE,
                    'chain': 'solana',
                })
            except Exception as e:
                logger.debug(f"Error parsing contract: {e}")
                continue

        return pd.DataFrame(records)

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
            logger.error(f"Drift collect_liquidations error: {e}")
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
            # Normalize symbols - strip '-PERP', ':USDT', '/USDT' suffixes
            normalized_symbols = []
            for symbol in symbols:
                # Remove common perpetual suffixes
                normalized = symbol.upper()
                for suffix in ['-PERP', ':USDT', '/USDT', '-USD', ':USD', '/USD']:
                    normalized = normalized.replace(suffix, '')
                normalized_symbols.append(normalized)

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
                symbols=normalized_symbols,
                start_date=start_str,
                end_date=end_str
            )
        except Exception as e:
            logger.error(f"Drift collect_funding_rates error: {e}")
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

        NOTE: Drift Protocol does NOT provide OHLCV data via their public API.
        This method always returns an empty DataFrame.
        For price data, use collect_funding_rates() which includes oracle/mark prices.

        Standardized method name for collection manager compatibility.
        """
        # Drift does not have OHLCV endpoint - use the flag from fetch_ohlcv
        if not self._ohlcv_logged:
            logger.info(
                "Drift OHLCV: API does not support candles endpoint. "
                "Use funding_rates for price data."
            )
            self._ohlcv_logged = True
        return pd.DataFrame()

    async def collect_open_interest(
        self,
        symbols: List[str],
        start_date: Any = None,
        end_date: Any = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect open interest - wraps fetch_open_interest().

        NOTE: Drift API only provides CURRENT open interest via /contracts.
        Historical OI is not available from the public API.
        start_date and end_date are ignored.

        Standardized method name for collection manager compatibility.
        """
        try:
            # Normalize symbols - strip suffixes
            normalized_symbols = []
            for symbol in symbols:
                normalized = symbol.upper()
                for suffix in ['-PERP', ':USDT', '/USDT', '-USD', ':USD', '/USD']:
                    normalized = normalized.replace(suffix, '')
                normalized_symbols.append(normalized)

            return await self.fetch_open_interest(symbols=normalized_symbols)
        except Exception as e:
            logger.error(f"Drift collect_open_interest error: {e}")
            return pd.DataFrame()

    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.collection_stats.copy()
