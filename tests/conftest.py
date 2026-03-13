"""
Pytest Configuration and Fixtures
==================================

Shared fixtures and configuration for data collection test suite.

This module provides:
- Mock data generators for all data types
- Async test support fixtures
- Temporary storage fixtures
- Rate limiter test fixtures
- Common assertion helpers

Usage:
    pytest tests/ -v
    pytest tests/test_collectors.py -v
    pytest tests/ --cov=data_collection --cov-report=html

Author: Crypto StatArb Project
Version: 1.0.0
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Generator
from dataclasses import dataclass, field
from unittest.mock import MagicMock, AsyncMock, patch
import pandas as pd
import numpy as np
import json

# ==============================================================================
# Pytest Configuration
# ==============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "asyncio: marks tests as async tests"
    )


# ==============================================================================
# Async Event Loop Fixture
# ==============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ==============================================================================
# Temporary Directory Fixtures
# ==============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def temp_data_dir(temp_dir: Path) -> Path:
    """Create a temporary data directory structure."""
    data_dir = temp_dir / "data"
    (data_dir / "raw" / "cex").mkdir(parents=True)
    (data_dir / "raw" / "hybrid").mkdir(parents=True)
    (data_dir / "raw" / "dex").mkdir(parents=True)
    (data_dir / "raw" / "options").mkdir(parents=True)
    (data_dir / "processed").mkdir(parents=True)
    (data_dir / "metadata").mkdir(parents=True)
    return data_dir


# ==============================================================================
# Mock Configuration Fixtures
# ==============================================================================

@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Standard mock configuration for collectors."""
    return {
        'rate_limit': 100,
        'timeout': 30,
        'max_retries': 3,
        'base_delay': 0.1,
        'max_delay': 1.0,
        'api_key': 'test_api_key',
        'api_secret': 'test_api_secret',
    }


@pytest.fixture
def mock_venues_config() -> Dict[str, Dict]:
    """Mock venues configuration."""
    return {
        'binance': {
            'name': 'Binance',
            'venue_type': 'CEX',
            'funding_interval': 8,
            'rate_limit': 1200,
            'enabled': True,
        },
        'hyperliquid': {
            'name': 'Hyperliquid',
            'venue_type': 'HYBRID',
            'funding_interval': 1,
            'rate_limit': 100,
            'enabled': True,
        },
        'uniswap_v3': {
            'name': 'Uniswap V3',
            'venue_type': 'DEX',
            'rate_limit': 30,
            'enabled': True,
        },
    }


# ==============================================================================
# Mock Data Generators
# ==============================================================================

@dataclass
class MockDataGenerator:
    """Generate realistic mock data for testing."""

    seed: int = 42

    def __post_init__(self):
        np.random.seed(self.seed)

    def generate_funding_rates(
        self,
        symbols: List[str] = None,
        venue: str = 'binance',
        n_periods: int = 100,
        start_date: str = '2024-01-01',
        interval_hours: int = 8
    ) -> pd.DataFrame:
        """Generate mock funding rate data."""
        symbols = symbols or ['BTC', 'ETH', 'SOL']

        records = []
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')

        for symbol in symbols:
            base_rate = np.random.uniform(-0.0001, 0.0003)

            for i in range(n_periods):
                timestamp = start_dt + timedelta(hours=i * interval_hours)

                # Random walk with mean reversion
                rate = base_rate + np.random.normal(0, 0.0002)
                rate = np.clip(rate, -0.01, 0.01)

                mark_price = np.random.uniform(1000, 50000) if symbol == 'BTC' else np.random.uniform(100, 5000)

                records.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'funding_rate': rate,
                    'mark_price': mark_price,
                    'index_price': mark_price * (1 + np.random.uniform(-0.001, 0.001)),
                    'venue': venue,
                    'venue_type': 'CEX' if venue in ['binance', 'bybit'] else 'HYBRID',
                    'funding_interval': f'{interval_hours}h',
                })

        return pd.DataFrame(records)

    def generate_ohlcv(
        self,
        symbols: List[str] = None,
        venue: str = 'binance',
        n_candles: int = 100,
        start_date: str = '2024-01-01',
        timeframe: str = '1h'
    ) -> pd.DataFrame:
        """Generate mock OHLCV data."""
        symbols = symbols or ['BTC', 'ETH']

        records = []
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')

        for symbol in symbols:
            base_price = 45000 if symbol == 'BTC' else 2500
            price = base_price

            for i in range(n_candles):
                timestamp = start_dt + timedelta(hours=i)

                # Generate realistic OHLC
                return_pct = np.random.normal(0, 0.01)
                close = price * (1 + return_pct)

                high_deviation = abs(np.random.normal(0, 0.005))
                low_deviation = abs(np.random.normal(0, 0.005))

                high = max(price, close) * (1 + high_deviation)
                low = min(price, close) * (1 - low_deviation)
                open_price = price

                volume = np.random.exponential(1000) * base_price / 1000

                records.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume,
                    'volume_usd': volume * close,
                    'venue': venue,
                    'venue_type': 'CEX',
                    'contract_type': 'PERPETUAL',
                    'timeframe': timeframe,
                })

                price = close

        return pd.DataFrame(records)

    def generate_dex_pools(
        self,
        n_pools: int = 50,
        chain: str = 'ethereum'
    ) -> pd.DataFrame:
        """Generate mock DEX pool data."""
        tokens = ['WETH', 'WBTC', 'USDC', 'USDT', 'DAI', 'UNI', 'LINK', 'AAVE']

        records = []
        for i in range(n_pools):
            token0, token1 = np.random.choice(tokens, size=2, replace=False)

            tvl = np.exp(np.random.normal(13, 2))  # Log-normal TVL
            tvl = max(10000, min(100_000_000, tvl))

            volume_ratio = np.random.lognormal(0, 1)
            volume_24h = tvl * volume_ratio

            records.append({
                'pool_address': f'0x{i:040x}',
                'chain': chain,
                'protocol': 'uniswap_v3',
                'token0_symbol': token0,
                'token1_symbol': token1,
                'fee_tier_bps': np.random.choice([100, 500, 3000, 10000]),
                'tvl_usd': tvl,
                'volume_24h_usd': volume_24h,
                'tx_count_24h': int(volume_24h / np.random.uniform(500, 5000)),
                'created_at': datetime.utcnow() - timedelta(days=np.random.randint(30, 365)),
            })

        return pd.DataFrame(records)

    def generate_options_chain(
        self,
        underlying: str = 'BTC',
        n_strikes: int = 20,
        n_expiries: int = 5,
        spot_price: float = 45000
    ) -> pd.DataFrame:
        """Generate mock options chain data."""
        records = []
        timestamp = datetime.utcnow()

        # Generate strikes around spot
        strikes = [spot_price * (0.7 + 0.03 * i) for i in range(n_strikes)]

        # Generate expiries
        expiries = [timestamp + timedelta(days=7 * (i + 1)) for i in range(n_expiries)]

        for expiry in expiries:
            days_to_expiry = (expiry - timestamp).days

            for strike in strikes:
                for option_type in ['CALL', 'PUT']:
                    moneyness = strike / spot_price

                    # Simple IV model
                    base_iv = 0.6 + 0.1 * abs(np.log(moneyness))
                    iv = base_iv + np.random.normal(0, 0.05)
                    iv = max(0.1, min(2.0, iv))

                    # Simple Greeks
                    if option_type == 'CALL':
                        delta = max(0, min(1, 1 - moneyness + np.random.normal(0, 0.1)))
                    else:
                        delta = max(-1, min(0, -moneyness + np.random.normal(0, 0.1)))

                    records.append({
                        'timestamp': timestamp,
                        'instrument_name': f'{underlying}-{expiry.strftime("%d%b%y").upper()}-{int(strike)}-{option_type[0]}',
                        'underlying': underlying,
                        'strike': strike,
                        'expiry': expiry,
                        'option_type': option_type,
                        'mark_iv': iv,
                        'delta': delta,
                        'gamma': abs(np.random.normal(0, 0.0001)),
                        'vega': abs(np.random.normal(0, 10)),
                        'theta': -abs(np.random.normal(0, 50)),
                        'underlying_price': spot_price,
                        'days_to_expiry': days_to_expiry,
                        'venue': 'deribit',
                    })

        return pd.DataFrame(records)


@pytest.fixture
def mock_data_generator() -> MockDataGenerator:
    """Provide mock data generator instance."""
    return MockDataGenerator(seed=42)


@pytest.fixture
def sample_funding_df(mock_data_generator: MockDataGenerator) -> pd.DataFrame:
    """Generate sample funding rate DataFrame."""
    return mock_data_generator.generate_funding_rates()


@pytest.fixture
def sample_ohlcv_df(mock_data_generator: MockDataGenerator) -> pd.DataFrame:
    """Generate sample OHLCV DataFrame."""
    return mock_data_generator.generate_ohlcv()


@pytest.fixture
def sample_pools_df(mock_data_generator: MockDataGenerator) -> pd.DataFrame:
    """Generate sample DEX pools DataFrame."""
    return mock_data_generator.generate_dex_pools()


@pytest.fixture
def sample_options_df(mock_data_generator: MockDataGenerator) -> pd.DataFrame:
    """Generate sample options chain DataFrame."""
    return mock_data_generator.generate_options_chain()


# ==============================================================================
# Mock HTTP Response Fixtures
# ==============================================================================

@pytest.fixture
def mock_aiohttp_session():
    """Create mock aiohttp session."""
    session = AsyncMock()

    # Default successful response
    response = AsyncMock()
    response.status = 200
    response.json = AsyncMock(return_value={'data': []})
    response.text = AsyncMock(return_value='{}')
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)

    session.get = MagicMock(return_value=response)
    session.post = MagicMock(return_value=response)
    session.close = AsyncMock()
    session.closed = False

    return session


@pytest.fixture
def mock_rate_limited_response():
    """Create mock rate limited response."""
    response = AsyncMock()
    response.status = 429
    response.headers = {'Retry-After': '60'}
    response.text = AsyncMock(return_value='Rate limit exceeded')
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    return response


@pytest.fixture
def mock_error_response():
    """Create mock error response."""
    response = AsyncMock()
    response.status = 500
    response.text = AsyncMock(return_value='Internal Server Error')
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    return response


# ==============================================================================
# Validation Helper Fixtures
# ==============================================================================

@dataclass
class ValidationHelper:
    """Helper class for test validations."""

    @staticmethod
    def assert_funding_rate_schema(df: pd.DataFrame) -> None:
        """Assert DataFrame matches funding rate schema."""
        required_columns = ['timestamp', 'symbol', 'funding_rate', 'venue']
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

        assert df['funding_rate'].between(-1, 1).all(), "Funding rate out of range"
        assert df['symbol'].str.isupper().all(), "Symbols should be uppercase"

    @staticmethod
    def assert_ohlcv_schema(df: pd.DataFrame) -> None:
        """Assert DataFrame matches OHLCV schema."""
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

        # OHLCV consistency
        assert (df['high'] >= df['low']).all(), "High must be >= Low"
        assert (df['high'] >= df['open']).all(), "High must be >= Open"
        assert (df['high'] >= df['close']).all(), "High must be >= Close"
        assert (df['low'] <= df['open']).all(), "Low must be <= Open"
        assert (df['low'] <= df['close']).all(), "Low must be <= Close"
        assert (df['volume'] >= 0).all(), "Volume must be non-negative"

    @staticmethod
    def assert_no_duplicates(df: pd.DataFrame, key_columns: List[str]) -> None:
        """Assert no duplicate rows for given key columns."""
        duplicates = df.duplicated(subset=key_columns, keep=False)
        assert not duplicates.any(), f"Found {duplicates.sum()} duplicate rows"

    @staticmethod
    def assert_date_range(
        df: pd.DataFrame,
        start_date: str,
        end_date: str,
        timestamp_col: str = 'timestamp'
    ) -> None:
        """Assert data falls within expected date range."""
        df_start = df[timestamp_col].min()
        df_end = df[timestamp_col].max()

        expected_start = pd.to_datetime(start_date)
        expected_end = pd.to_datetime(end_date)

        assert df_start >= expected_start, f"Data starts before {start_date}"
        assert df_end <= expected_end, f"Data ends after {end_date}"


@pytest.fixture
def validation_helper() -> ValidationHelper:
    """Provide validation helper instance."""
    return ValidationHelper()


# ==============================================================================
# Performance Testing Fixtures
# ==============================================================================

@pytest.fixture
def performance_timer():
    """Simple performance timer context manager."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end_time = time.perf_counter()
            self.elapsed = self.end_time - self.start_time

    return Timer


# ==============================================================================
# Environment Variable Fixtures
# ==============================================================================

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set mock environment variables for testing."""
    test_vars = {
        'BINANCE_API_KEY': 'test_binance_key',
        'BINANCE_API_SECRET': 'test_binance_secret',
        'HYPERLIQUID_API_KEY': 'test_hl_key',
        'DERIBIT_CLIENT_ID': 'test_deribit_id',
        'DERIBIT_CLIENT_SECRET': 'test_deribit_secret',
        'THEGRAPH_API_KEY': 'test_graph_key',
    }

    for key, value in test_vars.items():
        monkeypatch.setenv(key, value)

    return test_vars


# ==============================================================================
# Collector Instance Fixtures
# ==============================================================================

@pytest.fixture
def mock_base_collector(mock_config):
    """Create mock base collector for testing."""
    from unittest.mock import MagicMock

    collector = MagicMock()
    collector.VENUE = 'test_venue'
    collector.VENUE_TYPE = 'CEX'
    collector.config = mock_config
    collector.stats = MagicMock()
    collector.stats.api_calls = 0
    collector.stats.records_fetched = 0
    collector.stats.errors = []

    return collector


# ==============================================================================
# Cleanup Fixtures
# ==============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Any cleanup code here
    pass


# ==============================================================================
# Markers for Test Categories
# ==============================================================================

# Usage in tests:
# @pytest.mark.slow
# @pytest.mark.integration
# @pytest.mark.unit
# @pytest.mark.asyncio
