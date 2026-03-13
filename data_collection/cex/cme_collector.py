"""
CME Bitcoin Futures Data Collector - Professional Quality

validated collector for CME Group regulated Bitcoin futures.
Primary venue for institutional and regulated cryptocurrency exposure.

Supported Data Types:
    - BTC/USD monthly futures contracts
    - MicroBTC futures (1/10th contract size)
    - Continuous futures (front/second month)
    - Term structure and basis analysis
    - Settlement prices and OHLCV
    - Open interest history
    - CFTC Commitment of Traders (COT) reports
    - CME CF Bitcoin Reference Rate (BRR)

Data Sources:
    - Yahoo Finance: PRIMARY - FREE, no API key needed (ticker: BTC=F)
    - Nasdaq Data Link (formerly Quandl): Secondary historical data
    - FRED (Federal Reserve): BRR reference rate
    - CME Group: Contract specifications

API Documentation:
    - Yahoo Finance: https://pypi.org/project/yfinance/
    - Nasdaq Data Link: https://docs.data.nasdaq.com/
    - FRED API: https://fred.stlouisfed.org/docs/api/

Contract Specifications (BTC Futures):
    - Contract size: 5 BTC per contract
    - Micro contract: 0.1 BTC per contract
    - Tick size: $5 per BTC ($25 per contract)
    - Settlement: Cash settled to CME CF Bitcoin Reference Rate
    - Expiry: Last Friday of contract month, 4:00 PM London Time
    - Trading hours: Sunday-Friday 5:00pm-4:00pm CT (23 hours)
    - Margin: ~35% initial margin (varies)

Statistical Arbitrage Applications:
    - Basis trading (CME futures vs perp funding)
    - Calendar spread arbitrage (front vs back month)
    - Cash-and-carry arbitrage
    - COT-based positioning signals (smart money tracking)
    - Institutional flow analysis
    - Regulatory arbitrage (onshore vs offshore)

Version: 3.0.0 - Added FREE Yahoo Finance support
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import calendar

from ..base_collector import BaseCollector
from ..utils.rate_limiter import get_shared_rate_limiter
from ..utils.retry_handler import RetryHandler

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================

class ContractMonth(Enum):
    """CME futures contract month codes."""
    F = 1 # January
    G = 2 # February
    H = 3 # March
    J = 4 # April
    K = 5 # May
    M = 6 # June
    N = 7 # July
    Q = 8 # August
    U = 9 # September
    V = 10 # October
    X = 11 # November
    Z = 12 # December

class MarketStructure(Enum):
    """Futures term structure classification."""
    CONTANGO = 'contango' # Futures > Spot (normal)
    BACKWARDATION = 'backwardation' # Futures < Spot (inverted)
    FLAT = 'flat' # ~0 basis

class BasisTrend(Enum):
    """Basis trend classification."""
    STEEP_CONTANGO = 'steep_contango' # > 15% annualized
    MODERATE_CONTANGO = 'moderate_contango' # 5-15% annualized
    FLAT = 'flat' # -5% to 5%
    MODERATE_BACKWARDATION = 'moderate_backwardation' # -15% to -5%
    STEEP_BACKWARDATION = 'steep_backwardation' # < -15%

class TraderCategory(Enum):
    """CFTC COT trader categories."""
    DEALER = 'dealer' # Swap dealers
    ASSET_MANAGER = 'asset_manager' # Institutional investors
    LEVERAGED = 'leveraged' # Hedge funds
    OTHER_REPORTABLE = 'other_reportable'
    NON_REPORTABLE = 'non_reportable' # Small traders

class COTSentiment(Enum):
    """COT-derived sentiment classification."""
    STRONGLY_BULLISH = 'strongly_bullish' # Specs net long > 70%
    BULLISH = 'bullish' # Specs net long 55-70%
    NEUTRAL = 'neutral' # 45-55%
    BEARISH = 'bearish' # Specs net long 30-45%
    STRONGLY_BEARISH = 'strongly_bearish' # < 30%

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class FuturesContract:
    """CME futures contract data."""
    timestamp: datetime
    symbol: str
    contract_month: str
    expiry_date: datetime
    settlement_price: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    open_interest: int
    spot_price: Optional[float] = None
    
    @property
    def days_to_expiry(self) -> int:
        """Days until contract expiry."""
        return max(0, (self.expiry_date - self.timestamp).days)
    
    @property
    def basis(self) -> float:
        """Basis (futures - spot)."""
        if self.spot_price:
            return self.settlement_price - self.spot_price
        return 0.0
    
    @property
    def basis_pct(self) -> float:
        """Basis as percentage of spot."""
        if self.spot_price and self.spot_price > 0:
            return (self.basis / self.spot_price) * 100
        return 0.0
    
    @property
    def annualized_basis(self) -> float:
        """Annualized basis (carry)."""
        if self.days_to_expiry > 0:
            return self.basis_pct * (365 / self.days_to_expiry)
        return 0.0
    
    @property
    def basis_trend(self) -> BasisTrend:
        """Classify basis trend."""
        ann = self.annualized_basis
        if ann > 15:
            return BasisTrend.STEEP_CONTANGO
        elif ann > 5:
            return BasisTrend.MODERATE_CONTANGO
        elif ann > -5:
            return BasisTrend.FLAT
        elif ann > -15:
            return BasisTrend.MODERATE_BACKWARDATION
        else:
            return BasisTrend.STEEP_BACKWARDATION
    
    @property
    def market_structure(self) -> MarketStructure:
        """Determine market structure."""
        if self.basis_pct > 0.5:
            return MarketStructure.CONTANGO
        elif self.basis_pct < -0.5:
            return MarketStructure.BACKWARDATION
        else:
            return MarketStructure.FLAT
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'contract_month': self.contract_month,
            'expiry_date': self.expiry_date,
            'days_to_expiry': self.days_to_expiry,
            'settlement_price': self.settlement_price,
            'open': self.open, 'high': self.high, 'low': self.low, 'close': self.close,
            'volume': self.volume,
            'open_interest': self.open_interest,
            'spot_price': self.spot_price,
            'basis': self.basis,
            'basis_pct': self.basis_pct,
            'annualized_basis': self.annualized_basis,
            'basis_trend': self.basis_trend.value,
            'market_structure': self.market_structure.value,
        }

@dataclass
class TermStructure:
    """Futures term structure snapshot."""
    timestamp: datetime
    spot_price: float
    front_month_price: float
    front_month_expiry: datetime
    second_month_price: float
    second_month_expiry: datetime
    front_oi: int
    second_oi: int
    
    @property
    def front_basis(self) -> float:
        """Front month basis."""
        return self.front_month_price - self.spot_price
    
    @property
    def front_basis_pct(self) -> float:
        """Front month basis percentage."""
        return (self.front_basis / self.spot_price * 100) if self.spot_price > 0 else 0
    
    @property
    def front_dte(self) -> int:
        """Front month days to expiry."""
        return max(0, (self.front_month_expiry - self.timestamp).days)
    
    @property
    def front_annualized_basis(self) -> float:
        """Front month annualized basis."""
        if self.front_dte > 0:
            return self.front_basis_pct * (365 / self.front_dte)
        return 0
    
    @property
    def calendar_spread(self) -> float:
        """Calendar spread (second - front)."""
        return self.second_month_price - self.front_month_price
    
    @property
    def calendar_spread_pct(self) -> float:
        """Calendar spread as percentage."""
        return (self.calendar_spread / self.front_month_price * 100) if self.front_month_price > 0 else 0
    
    @property
    def market_structure(self) -> MarketStructure:
        """Overall market structure."""
        if self.front_basis_pct > 0.5:
            return MarketStructure.CONTANGO
        elif self.front_basis_pct < -0.5:
            return MarketStructure.BACKWARDATION
        else:
            return MarketStructure.FLAT
    
    @property
    def is_normal_contango(self) -> bool:
        """Normal contango: front < second."""
        return self.calendar_spread > 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'spot_price': self.spot_price,
            'front_month_price': self.front_month_price,
            'front_month_expiry': self.front_month_expiry,
            'front_dte': self.front_dte,
            'front_basis': self.front_basis,
            'front_basis_pct': self.front_basis_pct,
            'front_annualized_basis': self.front_annualized_basis,
            'second_month_price': self.second_month_price,
            'second_month_expiry': self.second_month_expiry,
            'calendar_spread': self.calendar_spread,
            'calendar_spread_pct': self.calendar_spread_pct,
            'front_oi': self.front_oi,
            'second_oi': self.second_oi,
            'market_structure': self.market_structure.value,
            'is_normal_contango': self.is_normal_contango,
        }

@dataclass
class COTPosition:
    """CFTC Commitment of Traders position data."""
    timestamp: datetime
    symbol: str
    dealer_long: int
    dealer_short: int
    asset_manager_long: int
    asset_manager_short: int
    leveraged_long: int
    leveraged_short: int
    other_long: int
    other_short: int
    non_reportable_long: int
    non_reportable_short: int
    total_oi: int
    
    @property
    def dealer_net(self) -> int:
        """Dealer net position."""
        return self.dealer_long - self.dealer_short
    
    @property
    def asset_manager_net(self) -> int:
        """Asset manager net position."""
        return self.asset_manager_long - self.asset_manager_short
    
    @property
    def leveraged_net(self) -> int:
        """Leveraged funds (hedge funds) net position."""
        return self.leveraged_long - self.leveraged_short
    
    @property
    def spec_long_ratio(self) -> float:
        """Speculator (leveraged + asset manager) long ratio."""
        spec_long = self.leveraged_long + self.asset_manager_long
        spec_short = self.leveraged_short + self.asset_manager_short
        total = spec_long + spec_short
        return (spec_long / total * 100) if total > 0 else 50.0
    
    @property
    def sentiment(self) -> COTSentiment:
        """COT-derived sentiment."""
        ratio = self.spec_long_ratio
        if ratio > 70:
            return COTSentiment.STRONGLY_BULLISH
        elif ratio > 55:
            return COTSentiment.BULLISH
        elif ratio > 45:
            return COTSentiment.NEUTRAL
        elif ratio > 30:
            return COTSentiment.BEARISH
        else:
            return COTSentiment.STRONGLY_BEARISH
    
    @property
    def is_extreme_positioning(self) -> bool:
        """Check for extreme positioning (potential reversal)."""
        return self.spec_long_ratio > 80 or self.spec_long_ratio < 20
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'dealer_long': self.dealer_long, 'dealer_short': self.dealer_short, 'dealer_net': self.dealer_net,
            'asset_manager_long': self.asset_manager_long, 'asset_manager_short': self.asset_manager_short, 'asset_manager_net': self.asset_manager_net,
            'leveraged_long': self.leveraged_long, 'leveraged_short': self.leveraged_short, 'leveraged_net': self.leveraged_net,
            'other_long': self.other_long, 'other_short': self.other_short,
            'non_reportable_long': self.non_reportable_long, 'non_reportable_short': self.non_reportable_short,
            'total_oi': self.total_oi,
            'spec_long_ratio': self.spec_long_ratio,
            'sentiment': self.sentiment.value,
            'is_extreme_positioning': self.is_extreme_positioning,
        }

# =============================================================================
# Collector Class
# =============================================================================

class CMECollector(BaseCollector):
    """
    CME Bitcoin Futures data collector.
    
    professional-quality data for regulated futures analysis.
    
    Features:
    - Continuous futures (front/second month)
    - Term structure and basis analysis
    - Settlement prices and OHLCV
    - Open interest history
    - CFTC COT reports
    - CME CF Bitcoin Reference Rate
    - Calendar spread analysis
    
    Attributes:
        VENUE: Exchange identifier ('cme')
        VENUE_TYPE: Exchange type ('CEX')
        CONTRACT_SIZE: BTC per standard contract (5)
        MICRO_CONTRACT_SIZE: BTC per micro contract (0.1)
    
    Example:
        >>> config = {'nasdaq_api_key': 'YOUR_KEY'}
        >>> collector = CMECollector(config)
        >>> term_structure = await collector.fetch_term_structure()
        >>> cot = await collector.fetch_cot_report('2024-01-01')
    """
    
    VENUE = 'cme'
    VENUE_TYPE = 'CEX'
    
    NASDAQ_DATA_LINK_URL = 'https://data.nasdaq.com/api/v3'
    FRED_URL = 'https://api.stlouisfed.org/fred'
    
    QUANDL_DATASETS = {
        'continuous_front': 'CHRIS/CME_BTC1',
        'continuous_second': 'CHRIS/CME_BTC2',
    }
    
    CONTRACT_SIZE = 5
    MICRO_CONTRACT_SIZE = 0.1
    TICK_SIZE = 5.0
    TICK_VALUE = 25.0
    
    MONTH_CODES = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M', 7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
    MONTH_CODES_REVERSE = {v: k for k, v in MONTH_CODES.items()}
    
    def __init__(self, config: Dict):
        """Initialize CME collector."""
        super().__init__(config)
        self.nasdaq_api_key = config.get('nasdaq_api_key', config.get('quandl_api_key', ''))
        self.fred_api_key = config.get('fred_api_key', '')

        # Use Yahoo Finance as primary source (FREE, no API key needed)
        self.use_yahoo_finance = config.get('use_yahoo_finance', True)

        rate_limit = config.get('rate_limit', 15)
        self.rate_limiter = get_shared_rate_limiter('cme', rate=rate_limit, per=60.0, burst=5)
        self.retry_handler = RetryHandler(max_retries=3, base_delay=2.0, max_delay=30.0)

        self.timeout = aiohttp.ClientTimeout(total=60)
        self.session: Optional[aiohttp.ClientSession] = None

        self.collection_stats = {'records_collected': 0, 'api_calls': 0, 'errors': 0}
        logger.info(f"Initialized CME collector (rate_limit={rate_limit}/min, yahoo_finance={self.use_yahoo_finance})")
    
    async def __aenter__(self):
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(limit=5)
            self.session = aiohttp.ClientSession(timeout=self.timeout, connector=connector)
        return self.session
    
    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
        logger.info(f"CME collector closed. Stats: {self.collection_stats}")
    
    async def _nasdaq_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make Nasdaq Data Link API request."""
        session = await self._get_session()
        url = f"{self.NASDAQ_DATA_LINK_URL}/{endpoint}"
        
        params = params or {}
        params['api_key'] = self.nasdaq_api_key
        
        await self.rate_limiter.acquire()
        
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    self.collection_stats['api_calls'] += 1
                    return await resp.json()
                elif resp.status == 403:
                    logger.error("Nasdaq Data Link: Invalid API key")
                    return None
                else:
                    logger.error(f"Nasdaq Data Link HTTP {resp.status}")
                    return None
        except Exception as e:
            logger.error(f"Nasdaq Data Link error: {e}")
            self.collection_stats['errors'] += 1
            return None
    
    async def _fred_request(self, series_id: str, params: Dict = None) -> Optional[Dict]:
        """Make FRED API request."""
        if not self.fred_api_key:
            return None
        
        session = await self._get_session()
        url = f"{self.FRED_URL}/series/observations"
        
        params = params or {}
        params['api_key'] = self.fred_api_key
        params['series_id'] = series_id
        params['file_type'] = 'json'
        
        await self.rate_limiter.acquire()
        
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    self.collection_stats['api_calls'] += 1
                    return await resp.json()
                return None
        except Exception as e:
            logger.error(f"FRED error: {e}")
            self.collection_stats['errors'] += 1
            return None
    
    def _get_expiry_date(self, year: int, month: int) -> datetime:
        """Calculate last Friday of month (CME expiry)."""
        last_day = calendar.monthrange(year, month)[1]
        last_date = datetime(year, month, last_day)
        days_since_friday = (last_date.weekday() - 4) % 7
        return (last_date - timedelta(days=days_since_friday)).replace(tzinfo=timezone.utc)
    
    async def fetch_from_yahoo_finance(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch CME BTC Futures data from Yahoo Finance (FREE, no API key).

        Uses ticker BTC=F which represents CME Bitcoin Futures front month.
        Data available from Dec 2017 to present.

        Args:
            start_date: Start date YYYY-MM-DD (default: 2020-01-01)
            end_date: End date YYYY-MM-DD (default: today)

        Returns:
            DataFrame with OHLCV data and PDF-compliant fields
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            return pd.DataFrame()

        start_date = start_date or '2020-01-01'
        end_date = end_date or datetime.utcnow().strftime('%Y-%m-%d')

        try:
            logger.info(f"Fetching CME BTC Futures from Yahoo Finance (BTC=F): {start_date} to {end_date}")

            btc_futures = yf.Ticker("BTC=F")
            df = btc_futures.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning("No data returned from Yahoo Finance")
                return pd.DataFrame()

            # Reset index to get timestamp as column
            df = df.reset_index()

            # Rename columns to standard format
            df = df.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Ensure timestamp is timezone-aware UTC
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

            # Add PDF-required fields for CME futures
            df['symbol'] = 'BTC'
            df['venue'] = self.VENUE
            df['venue_type'] = self.VENUE_TYPE
            df['contract_type'] = 'futures'
            df['settlement_method'] = 'cash'

            # Select only needed columns
            required_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                           'venue', 'venue_type', 'contract_type', 'settlement_method']
            df = df[[c for c in required_cols if c in df.columns]]

            self.collection_stats['records_collected'] += len(df)
            self.collection_stats['api_calls'] += 1
            logger.info(f"Fetched {len(df)} CME records from Yahoo Finance ({df['timestamp'].min()} to {df['timestamp'].max()})")

            return df

        except Exception as e:
            logger.error(f"Yahoo Finance error: {e}")
            self.collection_stats['errors'] += 1
            return pd.DataFrame()

    async def fetch_continuous_futures(self, contract_num: int = 1, start_date: str = '2022-01-01', end_date: str = None) -> pd.DataFrame:
        """
        Fetch continuous futures data.

        Args:
            contract_num: 1 for front month, 2 for second month
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD

        Returns:
            DataFrame with OHLCV, settlement, and open interest
        """
        # Use Yahoo Finance as primary source (FREE)
        if self.use_yahoo_finance and contract_num == 1:
            df = await self.fetch_from_yahoo_finance(start_date, end_date)
            if not df.empty:
                return df
            logger.info("Yahoo Finance returned no data, falling back to Nasdaq")

        # Fallback to Nasdaq Data Link (requires API key)
        end_date = end_date or datetime.utcnow().strftime('%Y-%m-%d')
        dataset = f'CHRIS/CME_BTC{contract_num}'

        params = {'start_date': start_date, 'end_date': end_date, 'order': 'asc'}

        data = await self._nasdaq_request(f'datasets/{dataset}.json', params)

        if not data or 'dataset' not in data:
            logger.warning(f"No data returned for {dataset}")
            return pd.DataFrame()

        dataset_data = data['dataset']
        columns = dataset_data.get('column_names', [])
        rows = dataset_data.get('data', [])

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=columns)

        column_mapping = {
            'Date': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low',
            'Last': 'close', 'Settle': 'settlement', 'Volume': 'volume',
            'Open Interest': 'open_interest', 'Change': 'change'
        }

        df = df.rename(columns=column_mapping)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        if 'settlement' in df.columns:
            df['close'] = df['close'].fillna(df['settlement'])

        df['symbol'] = 'BTC'
        df['contract_num'] = contract_num
        df['contract_type'] = 'continuous'
        df['venue'] = self.VENUE
        df['venue_type'] = self.VENUE_TYPE

        self.collection_stats['records_collected'] += len(df)
        logger.info(f"Fetched {len(df)} CME continuous futures records (contract {contract_num})")

        return df
    
    async def fetch_term_structure(self, date: str = None, spot_price: float = None) -> pd.DataFrame:
        """
        Fetch CME BTC futures term structure.
        
        Args:
            date: Date to fetch (default: latest)
            spot_price: Current spot price for basis calculation
            
        Returns:
            DataFrame with term structure data
        """
        df1 = await self.fetch_continuous_futures(1)
        df2 = await self.fetch_continuous_futures(2)
        
        if df1.empty or df2.empty:
            return pd.DataFrame()
        
        if date:
            target_date = pd.to_datetime(date, utc=True).date()
            df1 = df1[df1['timestamp'].dt.date == target_date]
            df2 = df2[df2['timestamp'].dt.date == target_date]
        else:
            df1 = df1.iloc[[-1]]
            df2 = df2.iloc[[-1]]
        
        if df1.empty or df2.empty:
            return pd.DataFrame()
        
        front_price = df1['close'].iloc[0]
        second_price = df2['close'].iloc[0]
        timestamp = df1['timestamp'].iloc[0]
        
        dte_front, dte_second = 30, 60
        
        if spot_price is None:
            spot_price = front_price * 0.99
        
        ts = TermStructure(
            timestamp=timestamp,
            spot_price=spot_price,
            front_month_price=front_price,
            front_month_expiry=timestamp + timedelta(days=dte_front),
            second_month_price=second_price,
            second_month_expiry=timestamp + timedelta(days=dte_second),
            front_oi=int(df1['open_interest'].iloc[0]) if 'open_interest' in df1.columns else 0,
            second_oi=int(df2['open_interest'].iloc[0]) if 'open_interest' in df2.columns else 0
        )
        
        result = pd.DataFrame([{**ts.to_dict(), 'venue': self.VENUE, 'venue_type': self.VENUE_TYPE}])
        return result
    
    async def fetch_funding_rates(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        CME futures don't have funding - calculate implied carry.

        Uses term structure to derive implied funding equivalent
        for comparison with perpetual swap funding.

        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with implied carry rates
        """
        df1 = await self.fetch_continuous_futures(1, start_date, end_date)
        df2 = await self.fetch_continuous_futures(2, start_date, end_date)

        if df1.empty or df2.empty:
            return pd.DataFrame()

        merged = pd.merge(
            df1[['timestamp', 'close', 'open_interest', 'volume']].rename(
                columns={'close': 'front_price', 'open_interest': 'front_oi', 'volume': 'front_volume'}
            ),
            df2[['timestamp', 'close', 'open_interest', 'volume']].rename(
                columns={'close': 'second_price', 'open_interest': 'second_oi', 'volume': 'second_volume'}
            ),
            on='timestamp', how='inner'
        )

        if merged.empty:
            return pd.DataFrame()

        merged['calendar_spread'] = merged['second_price'] - merged['front_price']
        merged['spread_pct'] = (merged['calendar_spread'] / merged['front_price']) * 100
        merged['implied_annual_carry'] = merged['spread_pct'] * 12
        merged['implied_daily_funding'] = merged['implied_annual_carry'] / 365 / 100

        result = pd.DataFrame({
            'timestamp': merged['timestamp'],
            'symbol': 'BTC',
            'funding_rate': merged['implied_daily_funding'],
            'funding_rate_pct': merged['implied_daily_funding'] * 100,
            'funding_rate_annualized': merged['implied_annual_carry'] / 100,
            'calendar_spread': merged['calendar_spread'],
            'calendar_spread_pct': merged['spread_pct'],
            'front_price': merged['front_price'],
            'second_price': merged['second_price'],
            'front_oi': merged['front_oi'],
            'second_oi': merged['second_oi'],
            'rate_type': 'implied_from_term_structure',
            'venue': self.VENUE,
            'venue_type': self.VENUE_TYPE
        })

        self.collection_stats['records_collected'] += len(result)
        return result

    async def collect_funding_rates(self, symbols: List[str], start_date: Any, end_date: Any, **kwargs) -> pd.DataFrame:
        """Standardized collect_funding_rates wrapper - wraps fetch_funding_rates()."""
        try:
            # Convert datetime to string
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            return await self.fetch_funding_rates(symbols=symbols, start_date=start_str, end_date=end_str)
        except Exception as e:
            logger.error(f"CME collect_funding_rates error: {e}")
            return pd.DataFrame()
    
    async def fetch_ohlcv(self, symbols: List[str], timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch OHLCV data for CME BTC futures (daily only via Nasdaq)."""
        all_data = []

        df_front = await self.fetch_continuous_futures(1, start_date, end_date)
        if not df_front.empty:
            df_front['contract'] = 'front_month'
            all_data.append(df_front)

        df_second = await self.fetch_continuous_futures(2, start_date, end_date)
        if not df_second.empty:
            df_second['contract'] = 'second_month'
            all_data.append(df_second)

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    async def collect_ohlcv(self, symbols: List[str], start_date: Any, end_date: Any, **kwargs) -> pd.DataFrame:
        """Standardized collect_ohlcv wrapper - wraps fetch_ohlcv()."""
        try:
            timeframe = kwargs.get('timeframe', '1d')

            # Convert datetime to string
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            return await self.fetch_ohlcv(symbols=symbols, timeframe=timeframe, start_date=start_str, end_date=end_str)
        except Exception as e:
            logger.error(f"CME collect_ohlcv error: {e}")
            return pd.DataFrame()
    
    async def fetch_open_interest(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical open interest data."""
        df = await self.fetch_continuous_futures(1, start_date, end_date)
        
        if df.empty or 'open_interest' not in df.columns:
            return pd.DataFrame()
        
        oi_df = df[['timestamp', 'open_interest', 'volume', 'close', 'venue', 'venue_type']].copy()
        oi_df['symbol'] = 'BTC'
        oi_df['oi_change'] = oi_df['open_interest'].diff()
        oi_df['oi_change_pct'] = oi_df['open_interest'].pct_change() * 100
        oi_df['oi_notional'] = oi_df['open_interest'] * self.CONTRACT_SIZE * oi_df['close']
        
        return oi_df
    
    async def fetch_cot_report(self, start_date: str = '2022-01-01') -> pd.DataFrame:
        """
        Fetch CFTC Commitment of Traders report for CME Bitcoin.
        
        Args:
            start_date: Start date
            
        Returns:
            DataFrame with COT positioning data
        """
        params = {'start_date': start_date, 'order': 'asc'}
        
        data = await self._nasdaq_request('datasets/CFTC/133741_FO_ALL.json', params)
        
        if not data or 'dataset' not in data:
            logger.warning("COT data not available")
            return pd.DataFrame()
        
        dataset_data = data['dataset']
        columns = dataset_data.get('column_names', [])
        rows = dataset_data.get('data', [])
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows, columns=columns)
        df['timestamp'] = pd.to_datetime(df['Date'], utc=True)
        df['venue'] = self.VENUE
        df['report_type'] = 'COT'
        df['symbol'] = 'BTC'
        
        if 'Noncommercial Long' in df.columns and 'Noncommercial Short' in df.columns:
            df['spec_net'] = df['Noncommercial Long'] - df['Noncommercial Short']
            total_spec = df['Noncommercial Long'] + df['Noncommercial Short']
            df['spec_long_pct'] = df['Noncommercial Long'] / total_spec * 100
        
        self.collection_stats['records_collected'] += len(df)
        logger.info(f"Fetched {len(df)} COT records")
        
        return df
    
    async def fetch_brr(self, start_date: str = '2022-01-01', end_date: str = None) -> pd.DataFrame:
        """Fetch CME CF Bitcoin Reference Rate (BRR)."""
        if not self.fred_api_key:
            logger.warning("FRED API key required for BRR data")
            return pd.DataFrame()
        
        end_date = end_date or datetime.utcnow().strftime('%Y-%m-%d')
        
        params = {'observation_start': start_date, 'observation_end': end_date}
        data = await self._fred_request('CBBTCUSD', params)
        
        if not data or 'observations' not in data:
            return pd.DataFrame()
        
        records = []
        for obs in data['observations']:
            if obs['value'] != '.':
                records.append({
                    'timestamp': pd.to_datetime(obs['date'], utc=True),
                    'brr_price': float(obs['value']),
                    'symbol': 'BTC',
                    'rate_type': 'BRR',
                    'venue': self.VENUE
                })
        
        return pd.DataFrame(records)
    
    async def fetch_comprehensive_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch comprehensive CME data."""
        results = {}
        
        logger.info("Fetching comprehensive CME data...")
        results['front_month'] = await self.fetch_continuous_futures(1, start_date, end_date)
        results['second_month'] = await self.fetch_continuous_futures(2, start_date, end_date)
        results['implied_funding'] = await self.fetch_funding_rates(['BTC'], start_date, end_date)
        results['open_interest'] = await self.fetch_open_interest(start_date, end_date)
        results['term_structure'] = await self.fetch_term_structure()
        results['cot'] = await self.fetch_cot_report(start_date)
        results['brr'] = await self.fetch_brr(start_date, end_date)
        
        return results
    
    def get_contract_specs(self) -> Dict:
        """Return CME BTC futures contract specifications."""
        return {
            'contract_name': 'CME Bitcoin Futures',
            'symbol': 'BTC',
            'underlying': 'CME CF Bitcoin Reference Rate (BRR)',
            'contract_size': self.CONTRACT_SIZE,
            'micro_contract_size': self.MICRO_CONTRACT_SIZE,
            'tick_size': self.TICK_SIZE,
            'tick_value': self.TICK_VALUE,
            'settlement': 'cash',
            'settlement_time': '4:00 PM London Time',
            'expiry_day': 'Last Friday of contract month',
            'trading_hours': 'Sunday-Friday 5:00pm-4:00pm CT',
            'position_limits': {'spot_month': 4000, 'all_months': 5000},
            'regulatory_body': 'CFTC',
            'exchange': 'CME Group',
            'venue': self.VENUE,
            'venue_type': self.VENUE_TYPE
        }
    
    def get_collection_stats(self) -> Dict:
        return self.collection_stats.copy()

async def test_cme_collector():
    """Test CME collector functionality."""
    config = {'nasdaq_api_key': '', 'rate_limit': 15}
    
    async with CMECollector(config) as collector:
        print("=" * 60)
        print("CME Collector Test")
        print("=" * 60)
        
        specs = collector.get_contract_specs()
        print(f"\n1. Contract specs: {specs['contract_name']}")
        print(f" Contract size: {specs['contract_size']} BTC")
        print(f" Tick value: ${specs['tick_value']}")
        
        if config['nasdaq_api_key']:
            df = await collector.fetch_continuous_futures(1, '2024-01-01', '2024-01-31')
            if not df.empty:
                print(f"\n2. Fetched {len(df)} front-month records")
                print(f" Latest close: ${df['close'].iloc[-1]:,.2f}")
        else:
            print("\n2. Skipping data tests (no API key)")
        
        print(f"\nStats: {collector.get_collection_stats()}")

if __name__ == '__main__':
    asyncio.run(test_cme_collector())