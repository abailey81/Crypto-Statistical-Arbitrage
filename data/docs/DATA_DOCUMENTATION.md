# Data Documentation

## Crypto Statistical Arbitrage System - Phase 1

**Document Version:** 1.0.0
**Last Updated:** January 2025

---

## Table of Contents

1. [Data Dictionary](#1-data-dictionary)
2. [Source Attribution](#2-source-attribution)
3. [Preprocessing Steps](#3-preprocessing-steps)
4. [File Formats](#4-file-formats)
5. [Naming Conventions](#5-naming-conventions)

---

## 1. Data Dictionary

### 1.1 Funding Rates Schema

| Field | Type | Unit | Description | Valid Range | Required |
|-------|------|------|-------------|-------------|----------|
| `timestamp` | datetime64[ns, UTC] | - | UTC timestamp of funding rate snapshot | 2022-2024 | Yes |
| `symbol` | string | - | Normalized token symbol (e.g., BTC, ETH) | [A-Z0-9]+ | Yes |
| `funding_rate` | float64 | decimal | Funding rate per 8-hour period | [-0.10, 0.10] | Yes |
| `funding_rate_original` | float64 | decimal | Original rate before normalization | [-0.10, 0.10] | No |
| `original_interval` | string | - | Original funding interval | 1h, 8h | No |
| `funding_interval` | string | - | Normalized interval (always 8h) | 8h | Yes |
| `venue` | string | - | Source exchange/venue | Valid venues | Yes |
| `venue_type` | string | - | Venue classification | cex, dex, hybrid | Yes |
| `mark_price` | float64 | USD | Mark price at funding time | > 0 | No |
| `index_price` | float64 | USD | Index price at funding time | > 0 | No |
| `next_funding_time` | datetime64[ns, UTC] | - | Next funding settlement time | Future | No |

### 1.2 OHLCV Schema

| Field | Type | Unit | Description | Valid Range | Required |
|-------|------|------|-------------|-------------|----------|
| `timestamp` | datetime64[ns, UTC] | - | Candle open time (UTC) | 2022-2024 | Yes |
| `symbol` | string | - | Normalized token symbol | [A-Z0-9]+ | Yes |
| `open` | float64 | USD | Opening price | > 0 | Yes |
| `high` | float64 | USD | Highest price in candle | >= max(open, close) | Yes |
| `low` | float64 | USD | Lowest price in candle | <= min(open, close) | Yes |
| `close` | float64 | USD | Closing price | > 0 | Yes |
| `volume` | float64 | base | Volume in base currency | >= 0 | Yes |
| `volume_usd` | float64 | USD | Volume in USD equivalent | >= 0 | No |
| `venue` | string | - | Source exchange/venue | Valid venues | Yes |
| `venue_type` | string | - | Venue classification | cex, dex, hybrid | Yes |
| `timeframe` | string | - | Candle timeframe | 1m, 5m, 15m, 1h, 4h, 1d | Yes |
| `trades` | int64 | count | Number of trades in candle | >= 0 | No |

### 1.3 Open Interest Schema

| Field | Type | Unit | Description | Valid Range | Required |
|-------|------|------|-------------|-------------|----------|
| `timestamp` | datetime64[ns, UTC] | - | Snapshot time (UTC) | 2022-2024 | Yes |
| `symbol` | string | - | Normalized token symbol | [A-Z0-9]+ | Yes |
| `open_interest` | float64 | contracts | Total open interest | >= 0 | Yes |
| `open_interest_usd` | float64 | USD | OI in USD value | >= 0 | No |
| `long_ratio` | float64 | ratio | Ratio of long positions | [0, 1] | No |
| `short_ratio` | float64 | ratio | Ratio of short positions | [0, 1] | No |
| `venue` | string | - | Source exchange/venue | Valid venues | Yes |

### 1.4 Options Schema

| Field | Type | Unit | Description | Valid Range | Required |
|-------|------|------|-------------|-------------|----------|
| `timestamp` | datetime64[ns, UTC] | - | Quote time (UTC) | 2022-2024 | Yes |
| `symbol` | string | - | Option instrument name | format varies | Yes |
| `underlying` | string | - | Underlying asset symbol | [A-Z]+ | Yes |
| `strike` | float64 | USD | Strike price | > 0 | Yes |
| `expiry` | datetime64[ns, UTC] | - | Expiration date | Future | Yes |
| `option_type` | string | - | Call or Put | C, P | Yes |
| `bid` | float64 | USD | Best bid price | >= 0 | Yes |
| `ask` | float64 | USD | Best ask price | >= bid | Yes |
| `mark_price` | float64 | USD | Mark price | > 0 | Yes |
| `mark_iv` | float64 | decimal | Mark implied volatility | [0.01, 10.0] | Yes |
| `delta` | float64 | - | Option delta | [-1, 1] | No |
| `gamma` | float64 | - | Option gamma | >= 0 | No |
| `vega` | float64 | - | Option vega | Real | No |
| `theta` | float64 | - | Option theta | Real | No |
| `venue` | string | - | Source venue | deribit, aevo | Yes |

---

## 2. Source Attribution

### 2.1 Primary Data Sources

| Venue | API Type | Rate Limit | Auth Required | Data Types | License |
|-------|----------|------------|---------------|------------|---------|
| **Binance** | REST + WebSocket | 1200/min | No (public) | Funding, OHLCV, OI | Free tier |
| **Bybit** | REST + WebSocket | 600/min | No (public) | Funding, OHLCV, OI | Free tier |
| **OKX** | REST + WebSocket | 1000/min | No (public) | Funding, OHLCV, OI | Free tier |
| **Hyperliquid** | REST | 100/min | No | Funding, OHLCV, OI | Free tier |
| **dYdX** | REST | 100/min | No | Funding, OHLCV, OI | Free tier |
| **GMX** | Subgraph | 60/min | No | Funding, OHLCV | Free tier |

### 2.2 Secondary/Validation Sources

| Venue | API Type | Rate Limit | Auth Required | Data Types | License |
|-------|----------|------------|---------------|------------|---------|
| **Deribit** | REST + WebSocket | 1200/min | API Key | Options, Funding | Free tier |
| **CoinGecko** | REST | 30/min | Optional | OHLCV, Reference | Free tier |
| **Coinalyze** | REST | 60/min | API Key | Funding, OI, Aggregated | Paid |
| **GeckoTerminal** | REST | 60/min | No | DEX OHLCV | Free tier |
| **DexScreener** | REST | 60/min | No | DEX OHLCV | Free tier |

### 2.3 API Endpoints Used

#### Binance
```
Funding Rates: GET /fapi/v1/fundingRate
OHLCV:         GET /fapi/v1/klines
Open Interest: GET /fapi/v1/openInterest
Mark Price:    GET /fapi/v1/premiumIndex
```

#### Bybit
```
Funding Rates: GET /v5/market/funding/history
OHLCV:         GET /v5/market/kline
Open Interest: GET /v5/market/open-interest
```

#### Hyperliquid
```
Funding Rates: POST /info (action: fundingHistory)
OHLCV:         POST /info (action: candleSnapshot)
Meta Info:     POST /info (action: meta)
```

#### dYdX (v4)
```
Funding Rates: GET /v4/historicalFunding/{ticker}
OHLCV:         GET /v4/candles/perpetualMarkets/{ticker}
Markets:       GET /v4/perpetualMarkets
```

### 2.4 Data Licensing

| Source | License Type | Commercial Use | Attribution Required |
|--------|--------------|----------------|---------------------|
| Binance | Public API Terms | Yes | No |
| Bybit | Public API Terms | Yes | No |
| OKX | Public API Terms | Yes | No |
| Hyperliquid | Open API | Yes | No |
| dYdX | Open API | Yes | No |
| CoinGecko | Free Tier Terms | Research | Recommended |
| Coinalyze | Paid Subscription | Per Agreement | Per Agreement |

---

## 3. Preprocessing Steps

### 3.1 Data Pipeline Stages

```
RAW DATA --> [Schema Enforcement] --> [Deduplication] --> [Temporal Alignment]
         --> [Outlier Treatment] --> [Missing Data] --> [Symbol Normalization]
         --> [Rate Normalization] --> CLEAN DATA
```

### 3.2 Stage Descriptions

#### Stage 1: Schema Enforcement
- Validate required columns exist
- Convert data types (timestamps to datetime64[ns, UTC])
- Add missing columns with NaN values
- Handle encoding issues in symbol names

#### Stage 2: Deduplication
- Strategy: Keep last occurrence (corrections supersede originals)
- Key columns: [timestamp, symbol, venue]
- Log all duplicates removed for audit trail

#### Stage 3: Temporal Alignment
- Normalize all timestamps to UTC
- Snap to settlement times per venue:
  - CEX (Binance, Bybit, OKX): 00:00, 08:00, 16:00 UTC
  - Hybrid (Hyperliquid, dYdX): Every hour
- Resample to target frequency if needed

#### Stage 4: Outlier Treatment
- Methods: Ensemble (LOF + Isolation Forest + Rolling MAD + IQR)
- Action: Winsorize (cap at 1st/99th percentile)
- Domain bounds enforced (funding rates: [-0.10, 0.10])

#### Stage 5: Missing Data Handling
- Method: Forward fill with max 3 period limit
- Cross-venue fill for extended gaps (if correlation > 0.85)
- Flag remaining gaps in metadata

#### Stage 6: Symbol Normalization
- Remove quote currency suffix (BTCUSDT -> BTC)
- Map venue-specific naming (BTC-PERP -> BTC)
- Standardize to uppercase

#### Stage 7: Rate Normalization (Funding Rates Only)
- Convert hourly rates to 8-hour equivalent
- Hyperliquid, dYdX: rate * 8
- CEX: No conversion needed

---

## 4. File Formats

### 4.1 Storage Format

| Format | Use Case | Compression | Partitioning |
|--------|----------|-------------|--------------|
| Parquet | Primary storage | Snappy | By venue |
| CSV | Export/sharing | None | N/A |
| JSON | Metadata | None | N/A |

### 4.2 Directory Structure

```
data/
├── raw/                    # Original API responses
│   ├── binance/
│   ├── bybit/
│   └── ...
├── processed/              # Cleaned data (Parquet)
│   ├── binance/
│   │   ├── funding_rates.parquet
│   │   └── ohlcv.parquet
│   ├── hyperliquid/
│   └── ...
├── metadata/               # Collection metadata
│   ├── collection_log.json
│   └── quality_metrics.json
└── docs/                   # Documentation
    ├── DATA_ACQUISITION_PLAN.md
    ├── DATA_QUALITY_REPORT.md
    └── DATA_DOCUMENTATION.md
```

### 4.3 Parquet Schema Example

```python
import pyarrow as pa

funding_rates_schema = pa.schema([
    ('timestamp', pa.timestamp('ns', tz='UTC')),
    ('symbol', pa.string()),
    ('funding_rate', pa.float64()),
    ('funding_rate_original', pa.float64()),
    ('original_interval', pa.string()),
    ('funding_interval', pa.string()),
    ('venue', pa.string()),
    ('venue_type', pa.string()),
    ('mark_price', pa.float64()),
    ('index_price', pa.float64()),
])
```

---

## 5. Naming Conventions

### 5.1 Symbol Naming

| Original (Venue-specific) | Normalized |
|---------------------------|------------|
| BTCUSDT (Binance) | BTC |
| BTC-PERP (Hyperliquid) | BTC |
| BTC-USD (dYdX) | BTC |
| XBTUSD (BitMEX) | BTC |
| ETH-USDT (OKX) | ETH |

### 5.2 Venue Identifiers

| Venue ID | Full Name | Type |
|----------|-----------|------|
| `binance` | Binance Futures | CEX |
| `bybit` | Bybit Derivatives | CEX |
| `okx` | OKX Futures | CEX |
| `kraken` | Kraken Futures | CEX |
| `hyperliquid` | Hyperliquid | Hybrid |
| `dydx` | dYdX v4 | Hybrid |
| `vertex` | Vertex Protocol | Hybrid |
| `gmx` | GMX | DEX |
| `deribit` | Deribit | Options |
| `aevo` | Aevo | Options |
| `coingecko` | CoinGecko | Aggregator |
| `coinalyze` | Coinalyze | Aggregator |

### 5.3 Timeframe Notation

| Notation | Duration | Use Case |
|----------|----------|----------|
| `1m` | 1 minute | High-frequency analysis |
| `5m` | 5 minutes | Short-term trading |
| `15m` | 15 minutes | Intraday trading |
| `1h` | 1 hour | Standard analysis (default) |
| `4h` | 4 hours | Swing trading |
| `1d` | 1 day | Position trading |

### 5.4 File Naming

```
{venue}_{data_type}_{timeframe}_{date_range}.parquet

Examples:
- binance_funding_rates.parquet
- binance_ohlcv_1h.parquet
- hyperliquid_funding_rates_2024.parquet
```

---

## Appendix A: Venue API Documentation Links

| Venue | Documentation URL |
|-------|-------------------|
| Binance | https://binance-docs.github.io/apidocs/ |
| Bybit | https://bybit-exchange.github.io/docs/v5/intro |
| OKX | https://www.okx.com/docs-v5/en/ |
| Hyperliquid | https://hyperliquid.gitbook.io/hyperliquid-docs |
| dYdX | https://docs.dydx.exchange/ |
| GMX | https://gmx-docs.io/ |
| Deribit | https://docs.deribit.com/ |
| CoinGecko | https://www.coingecko.com/en/api/documentation |

---

## Appendix B: Data Type Mappings

| Python Type | Parquet Type | JSON Type | Description |
|-------------|--------------|-----------|-------------|
| datetime64[ns, UTC] | TIMESTAMP(ns, UTC) | string (ISO8601) | UTC timestamps |
| float64 | DOUBLE | number | Numeric values |
| int64 | INT64 | integer | Count values |
| string | STRING | string | Text values |
| bool | BOOLEAN | boolean | Flag values |

---

*Document maintained by Tamer Atesyakar*
