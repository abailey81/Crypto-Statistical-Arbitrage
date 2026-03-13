# API Reference - Data Collection Module

## Overview

The data collection module provides a unified interface for collecting cryptocurrency data from multiple venue types:

- **CEX (Centralized Exchanges)**: Binance, Bybit, OKX
- **Hybrid Venues**: Hyperliquid, dYdX V4
- **DEX (Decentralized Exchanges)**: Uniswap V3, Curve
- **Options**: Deribit

---

## Core Classes

### BaseCollector

Abstract base class that all collectors inherit from.

```python
from data_collection.base_collector import BaseCollector
```

#### Class Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `VENUE` | `str` | Venue identifier (e.g., 'binance', 'hyperliquid') |
| `VENUE_TYPE` | `str` | Category: 'CEX', 'hybrid', or 'DEX' |

#### Constructor

```python
def __init__(self, config: Dict[str, Any])
```

**Parameters:**
- `config`: Dictionary containing venue-specific configuration
  - `rate_limit`: Requests per minute (default: 100)
  - `base_url`: API base URL (optional, uses default)
  - Additional venue-specific parameters

#### Abstract Methods

##### fetch_funding_rates

```python
async def fetch_funding_rates(
    self,
    symbols: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame
```

Fetch historical funding rates.

**Parameters:**
- `symbols`: List of trading symbols (e.g., ['BTC', 'ETH'])
- `start_date`: Start date in 'YYYY-MM-DD' format
- `end_date`: End date in 'YYYY-MM-DD' format

**Returns:** DataFrame with columns:
- `timestamp`: UTC datetime
- `symbol`: Trading symbol
- `funding_rate`: Funding rate as decimal
- `mark_price`: Mark price at funding time
- `index_price`: Index/spot price
- `venue`: Venue name
- `venue_type`: Venue category

##### fetch_ohlcv

```python
async def fetch_ohlcv(
    self,
    symbols: List[str],
    timeframe: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame
```

Fetch OHLCV (candlestick) data.

**Parameters:**
- `symbols`: List of trading symbols
- `timeframe`: Candle interval ('1m', '5m', '15m', '1h', '4h', '1d')
- `start_date`: Start date
- `end_date`: End date

**Returns:** DataFrame with columns:
- `timestamp`: UTC datetime
- `symbol`: Trading symbol
- `open`, `high`, `low`, `close`: Price data
- `volume`: Trading volume in base currency
- `volume_usd`: Volume in USD
- `venue`: Venue name
- `venue_type`: Venue category

#### Instance Methods

##### validate_data

```python
def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]
```

Run validation checks on collected data.

**Returns:** Dictionary containing:
- `valid`: Boolean indicating overall validity
- `row_count`: Number of rows
- `date_range`: Tuple of (min, max) timestamps
- `missing_pct`: Percentage of missing values
- `duplicates`: Number of duplicate rows
- `warnings`: List of warning messages

##### save_to_parquet

```python
def save_to_parquet(
    self,
    df: pd.DataFrame,
    path: str,
    partition_cols: Optional[List[str]] = None
)
```

Save DataFrame to Parquet format with optional partitioning.

##### close

```python
async def close()
```

Clean up resources (connections, sessions).

---

## CEX Collectors

### BinanceCollector

```python
from data_collection.cex.binance_collector import BinanceCollector
```

Primary CEX data source with highest coverage.

#### Configuration

```yaml
binance:
  rate_limit: 1200  # requests per minute
  endpoints:
    funding: "/fapi/v1/fundingRate"
    klines: "/fapi/v1/klines"
  costs:
    maker_fee: 0.0002
    taker_fee: 0.0004
```

#### Usage

```python
import asyncio
from data_collection.cex.binance_collector import BinanceCollector

async def main():
    config = {'rate_limit': 1200}
    collector = BinanceCollector(config)
    
    # Fetch funding rates
    funding_df = await collector.fetch_funding_rates(
        symbols=['BTC', 'ETH', 'SOL'],
        start_date='2024-01-01',
        end_date='2024-06-30'
    )
    
    # Fetch OHLCV data
    ohlcv_df = await collector.fetch_ohlcv(
        symbols=['BTC', 'ETH'],
        timeframe='1h',
        start_date='2024-01-01',
        end_date='2024-06-30'
    )
    
    await collector.close()

asyncio.run(main())
```

#### Notes

- Funding rates: 8-hour intervals (00:00, 08:00, 16:00 UTC)
- Max 1000 records per funding rate request
- Max 1500 records per OHLCV request

### BybitCollector

```python
from data_collection.cex.bybit_collector import BybitCollector
```

Secondary CEX source for cross-validation.

#### Configuration

```yaml
bybit:
  rate_limit: 120
  costs:
    maker_fee: 0.0001
    taker_fee: 0.0006
```

---

## Hybrid Collectors

### HyperliquidCollector

```python
from data_collection.hybrid.hyperliquid_collector import HyperliquidCollector
```

On-chain perpetuals with order book model.

#### Key Differences from CEX

- **Hourly funding** (not 8-hour like CEX)
- Max 500 hours per request
- On-chain settlement on Arbitrum

#### Configuration

```yaml
hyperliquid:
  rate_limit: 100
  base_url: "https://api.hyperliquid.xyz/info"
  funding_interval: "hourly"
  costs:
    maker_fee: 0.0000
    taker_fee: 0.00025
```

#### Usage

```python
from data_collection.hybrid.hyperliquid_collector import HyperliquidCollector

async def main():
    config = {'rate_limit': 100}
    collector = HyperliquidCollector(config)
    
    # Note: Returns hourly funding rates
    funding_df = await collector.fetch_funding_rates(
        symbols=['BTC', 'ETH'],
        start_date='2024-01-01',
        end_date='2024-03-31'
    )
    
    # DataFrame includes 'funding_interval' = 'hourly'
    # And 'funding_rate_annualized' pre-calculated
    
    await collector.close()
```

### DYDXCollector

```python
from data_collection.hybrid.dydx_v4_collector import DYDXCollector
```

Cosmos-based decentralized perpetuals.

#### Configuration

```yaml
dydx_v4:
  rate_limit: 100
  base_url: "https://indexer.dydx.trade/v4"
  funding_interval: "hourly"
```

---

## DEX Collectors

### UniswapV3Collector

```python
from data_collection.dex.uniswap_v3_collector import UniswapV3Collector
```

Collects data from Uniswap V3 via The Graph subgraphs.

#### Supported Chains

- Ethereum mainnet
- Arbitrum
- Optimism
- Polygon

#### Methods

##### fetch_pools

```python
async def fetch_pools(
    self,
    chain: str = 'ethereum',
    min_tvl: float = 500_000,
    min_volume: float = 50_000,
    min_tx_count: int = 100
) -> pd.DataFrame
```

Fetch pool data with liquidity and wash trading filters.

**Returns:** DataFrame with columns:
- `pool_id`: Pool contract address
- `token0_symbol`, `token1_symbol`: Token symbols
- `fee_tier`: Pool fee tier (0.05%, 0.30%, 1.00%)
- `tvl_usd`: Total value locked
- `volume_usd`: 24h volume
- `tx_count`: Transaction count
- `volume_tvl_ratio`: Volume/TVL ratio
- `wash_trading_flag`: Boolean (True if ratio > 10)
- `chain`: Blockchain network

##### fetch_token_prices

```python
async def fetch_token_prices(
    self,
    pool_ids: List[str],
    start_date: str,
    end_date: str,
    chain: str = 'ethereum'
) -> pd.DataFrame
```

Fetch daily OHLCV data from DEX swaps.

#### Usage

```python
from data_collection.dex.uniswap_v3_collector import UniswapV3Collector

async def main():
    config = {'graph_api_key': 'your-api-key'}
    collector = UniswapV3Collector(config)
    
    # Fetch pools on Arbitrum with min $500k TVL
    pools_df = await collector.fetch_pools(
        chain='arbitrum',
        min_tvl=500_000,
        min_volume=50_000
    )
    
    # Filter out wash trading
    clean_pools = pools_df[~pools_df['wash_trading_flag']]
    
    await collector.close()
```

---

## Options Collectors

### DeribitCollector

```python
from data_collection.options.deribit_collector import DeribitCollector
```

Primary source for crypto options data.

#### Methods

##### fetch_instruments

```python
async def fetch_instruments(
    self,
    currency: str = 'BTC'
) -> pd.DataFrame
```

Fetch all active option instruments.

##### fetch_option_chain

```python
async def fetch_option_chain(
    self,
    currency: str = 'BTC',
    include_greeks: bool = True
) -> pd.DataFrame
```

Fetch complete options chain with IV and Greeks.

**Returns:** DataFrame with columns:
- `instrument`: Contract name
- `underlying`: BTC or ETH
- `strike`: Strike price
- `expiry`: Expiration date
- `option_type`: 'call' or 'put'
- `mark_price`: Mark price
- `mark_iv`: Implied volatility
- `bid_price`, `ask_price`: Best bid/ask
- `delta`, `gamma`, `vega`, `theta`: Greeks
- `volume`, `open_interest`: Activity metrics

##### fetch_dvol_history

```python
async def fetch_dvol_history(
    self,
    currency: str = 'BTC',
    start_date: str = '2022-01-01',
    end_date: str = '2024-12-31',
    resolution: str = '1h'
) -> pd.DataFrame
```

Fetch DVOL (Deribit Volatility Index) history.

---

## Utility Classes

### RateLimiter

```python
from data_collection.utils.rate_limiter import TokenBucketRateLimiter
```

Token bucket algorithm for rate limiting.

```python
limiter = TokenBucketRateLimiter(rate=100, per=60.0, burst=10)

# Blocks until token available
await limiter.acquire()

# Acquire multiple tokens
await limiter.acquire(5)

# Check wait time
wait_time = limiter.wait_time(3)
```

### RetryHandler

```python
from data_collection.utils.retry_handler import RetryHandler
```

Exponential backoff with jitter for retries.

```python
handler = RetryHandler(
    max_retries=5,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True
)

# Use as decorator
@handler.retry
async def fetch_data():
    # API call that might fail
    pass

# Or execute directly
result = await handler.execute(fetch_data, arg1, arg2)
```

### DataValidator

```python
from data_collection.utils.data_validator import DataValidator
```

Comprehensive data quality validation.

```python
validator = DataValidator(max_missing_pct=5.0)

# Validate funding rates
result = validator.validate_funding_rates(df)
print(f"Valid: {result['valid']}")
print(f"Warnings: {result['warnings']}")

# Validate OHLCV
result = validator.validate_ohlcv(df)

# Cross-validate venues
result = validator.cross_validate_venues(
    df1=binance_df,
    df2=bybit_df,
    on='funding_rate',
    tolerance=0.001
)
```

### OptimizedStorage

```python
from data_collection.utils.storage import OptimizedStorage
```

High-performance Parquet storage with partitioning.

```python
storage = OptimizedStorage(
    base_path='/data',
    partition_strategy='daily',  # or 'monthly', 'venue', 'symbol'
    compression='zstd',
    cache_size=100
)

# Save with auto-partitioning
storage.save(df, 'funding_rates')

# Query with predicate pushdown
result = storage.query(
    'funding_rates',
    filters={'venue': 'binance', 'symbol': 'BTC'},
    columns=['timestamp', 'funding_rate']
)

# Get query plan
plan = storage.explain('funding_rates', filters={'venue': 'binance'})
```

### QualityAnalyzer

```python
from data_collection.utils.data_validator import QualityAnalyzer
```

Detailed data quality analysis.

```python
analyzer = QualityAnalyzer()

# Detect gaps in data
gaps = analyzer.detect_gaps(
    df,
    timestamp_col='timestamp',
    expected_interval='8h'
)

# Generate quality score
score = analyzer.calculate_quality_score(df)
# Returns dict with: completeness, consistency, accuracy, timeliness

# Generate dashboard
dashboard = analyzer.generate_quality_dashboard(df)
```

---

## Collection Manager

```python
from data_collection.collection_manager import CollectionManager
```

Orchestrates multi-venue data collection.

### Basic Usage

```python
manager = CollectionManager(config_path='config/')

# Collect from single venue
df = await manager.collect_venue(
    venue='binance',
    data_type='funding',
    symbols=['BTC', 'ETH'],
    start_date='2024-01-01',
    end_date='2024-06-30'
)

# Collect from all venues
results = await manager.collect_all(
    data_type='funding',
    symbols=['BTC', 'ETH'],
    start_date='2024-01-01',
    end_date='2024-06-30'
)
```

### Checkpoint/Resume

```python
# Enable checkpointing
manager = CollectionManager(
    config_path='config/',
    checkpoint_path='checkpoints/'
)

# Collection automatically saves progress
# If interrupted, resume from checkpoint:
manager.resume_from_checkpoint('collection_20240101.json')
```

### Progress Tracking

```python
# Get collection progress
progress = manager.get_progress()
print(f"Completed: {progress.completed}/{progress.total}")
print(f"Failed: {progress.failed}")
print(f"ETA: {progress.estimated_completion}")
```

---

## Funding Rate Normalization

```python
from data_collection.utils.funding_normalization import (
    normalize_funding_rates,
    annualize_funding,
    align_timestamps,
    calculate_funding_spread
)
```

### Normalize Intervals

```python
# Convert all venues to 8-hour intervals
normalized = normalize_funding_rates(
    df,
    target_interval='8h',
    aggregation='mean'
)
```

### Annualize

```python
# Add annualized funding column
df = annualize_funding(df)
# Creates 'funding_rate_annualized' column
```

### Cross-Venue Alignment

```python
# Align timestamps between venues
aligned = align_timestamps(
    binance_df,
    hyperliquid_df,
    tolerance='1h',
    method='nearest'
)
```

### Calculate Spread

```python
# Calculate funding spread between venues
spread_df = calculate_funding_spread(
    df,
    venue1='binance',
    venue2='hyperliquid'
)
# Creates 'funding_spread' and 'funding_spread_annualized' columns
```

---

## Error Handling

All collectors raise standardized exceptions:

```python
from data_collection.base_collector import (
    CollectionError,
    RateLimitError,
    ValidationError,
    NetworkError
)

try:
    df = await collector.fetch_funding_rates(...)
except RateLimitError:
    # Handle rate limiting (auto-retried by default)
    pass
except NetworkError as e:
    # Network connectivity issues
    logger.error(f"Network error: {e}")
except ValidationError as e:
    # Data validation failed
    logger.error(f"Invalid data: {e}")
except CollectionError as e:
    # General collection error
    logger.error(f"Collection failed: {e}")
```

---

## Performance Considerations

### Rate Limiting

Each venue has specific rate limits:
- Binance: 1200 req/min
- Bybit: 120 req/min
- Hyperliquid: 100 req/min
- The Graph: Varies by tier

### Memory Management

For large date ranges:
1. Use chunked collection via `CollectionManager`
2. Enable streaming to disk
3. Use appropriate partition strategy

### Parallelization

The `CollectionManager` automatically parallelizes:
- Across venues (concurrent)
- Within venue by symbol (concurrent with rate limit)

```python
# Configure parallelism
manager = CollectionManager(
    config_path='config/',
    max_concurrent_venues=3,
    max_concurrent_symbols=5
)
```

---

## See Also

- [Jupyter Notebooks](../notebooks/) - Analysis notebooks with examples
- [Data Dictionary](data_dictionary.md) - Schema reference for all data types
- [Source Attribution](source_attribution.md) - Complete API documentation per venue
