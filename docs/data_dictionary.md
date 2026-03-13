# Data Dictionary
## Crypto Statistical Arbitrage Multi-Venue System

**Document Version:** 3.0
**Last Updated:** January 31, 2026
**Schema Version:** 1.1.0
**Author:** Tamer Atesyakar

---

## Table of Contents

1. [Overview](#1-overview)
2. [Schema Conventions](#2-schema-conventions)
3. [Core Data Schemas](#3-core-data-schemas)
4. [Enumerated Types](#4-enumerated-types)
5. [Cross-Venue Normalization](#5-cross-venue-normalization)
6. [Data Relationships](#6-data-relationships)
7. [Quality Attributes](#7-quality-attributes)
8. [Storage Specifications](#8-storage-specifications)
9. [Derived Fields](#9-derived-fields)
10. [Changelog](#10-changelog)

---

## 1. Overview

This data dictionary defines all data schemas, field specifications, enumerated types, and normalization standards for the Crypto Statistical Arbitrage system. It serves as the authoritative reference for:

- Data collection pipeline output schemas
- Cross-venue data normalization rules
- Field validation constraints
- Storage format specifications
- Derived field calculations

### 1.1 Scope

| Data Category | Venues Covered | Primary Use Case |
|---------------|----------------|------------------|
| Funding Rates | Binance, Bybit, Hyperliquid, dYdX V4 | Cross-venue funding arbitrage |
| OHLCV (Price) | Binance, Bybit, CEX aggregated | Altcoin pairs trading, signals |
| Open Interest | Binance, Bybit | Market positioning, regime detection |
| Options Chain | Deribit | Volatility surface arbitrage |
| DEX Pools | Uniswap V3, Curve, GMX | DEX liquidity analysis |
| On-Chain Metrics | Glassnode, Arkham, Nansen | Whale tracking, flow analysis |

### 1.2 Design Principles

1. **Timestamp Standardization:** All timestamps in UTC, millisecond precision
2. **Symbol Normalization:** Unified symbol format across venues (e.g., `BTC`, not `BTCUSDT`)
3. **Numeric Precision:** Float64 for prices/rates, preserving full precision
4. **Null Handling:** Explicit null values, no magic numbers (e.g., -999)
5. **Schema Versioning:** Breaking changes increment major version

---

## 2. Schema Conventions

### 2.1 Naming Standards

| Convention | Rule | Example |
|------------|------|---------|
| Field names | snake_case, lowercase | `funding_rate`, `mark_price` |
| Timestamp fields | Suffix `_at` or `_timestamp` | `collected_at`, `timestamp` |
| Boolean fields | Prefix `is_` or `has_` | `is_valid`, `has_options` |
| Percentage fields | Suffix `_pct` for 0-100, `_rate` for decimals | `missing_pct`, `funding_rate` |
| USD values | Suffix `_usd` | `volume_usd`, `tvl_usd` |
| Annualized values | Suffix `_annualized` | `funding_rate_annualized` |

### 2.2 Data Types

| Type | Description | Python Type | Parquet Type |
|------|-------------|-------------|--------------|
| `timestamp_ms` | UTC timestamp, milliseconds | `datetime64[ms, UTC]` | `timestamp[ms, tz=UTC]` |
| `timestamp_us` | UTC timestamp, microseconds | `datetime64[us, UTC]` | `timestamp[us, tz=UTC]` |
| `symbol` | Normalized asset symbol | `str` | `string` |
| `venue` | Exchange/protocol identifier | `str` | `string` |
| `price` | USD price, full precision | `float64` | `float64` |
| `rate` | Decimal rate (e.g., 0.0001) | `float64` | `float64` |
| `quantity` | Asset quantity | `float64` | `float64` |
| `count` | Integer count | `int64` | `int64` |
| `category` | Categorical/enum value | `str` | `string` |
| `address` | Blockchain address | `str` | `string` |
| `hash` | Transaction/block hash | `str` | `string` |

### 2.3 Null Value Policy

| Scenario | Handling | Representation |
|----------|----------|----------------|
| Missing data point | Explicit null | `None` / `NaN` |
| Not applicable | Explicit null | `None` / `NaN` |
| Zero value | Actual zero | `0` / `0.0` |
| Unknown/unavailable | Explicit null with flag | `None` + `is_estimated=True` |

---

## 3. Core Data Schemas

### 3.1 Funding Rate Schema

Primary schema for perpetual swap funding rate data.

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| `timestamp` | timestamp_ms | Yes | Funding settlement time (UTC) | Non-null |
| `symbol` | symbol | Yes | Normalized asset symbol | Uppercase, no suffix |
| `funding_rate` | rate | Yes | Per-period funding rate | [-1.0, 1.0] |
| `funding_rate_annualized` | rate | No | Annualized funding rate | Derived field |
| `mark_price` | price | Yes | Mark price at funding time | > 0 |
| `index_price` | price | No | Index/spot reference price | > 0 |
| `next_funding_rate` | rate | No | Predicted next funding rate | [-1.0, 1.0] |
| `next_funding_time` | timestamp_ms | No | Next funding settlement time | > timestamp |
| `open_interest` | quantity | No | Open interest at funding time | >= 0 |
| `open_interest_usd` | price | No | Open interest in USD | >= 0 |
| `venue` | venue | Yes | Source exchange identifier | Valid venue enum |
| `venue_type` | category | Yes | Venue classification | CEX, HYBRID, DEX |
| `funding_interval` | category | Yes | Funding period duration | 1h, 8h |
| `contract_type` | category | No | Contract specification | PERPETUAL |
| `collected_at` | timestamp_ms | No | Data collection timestamp | >= timestamp |

**Primary Key:** `(timestamp, symbol, venue)`

**Validation Rules:**
- `funding_rate` typically in range [-0.01, 0.01] per period (flag outliers)
- `mark_price` and `index_price` deviation < 1% (flag larger deviations)
- No duplicate `(timestamp, symbol, venue)` combinations

**Example Record:**
```json
{
  "timestamp": "2024-01-15T08:00:00.000Z",
  "symbol": "BTC",
  "funding_rate": 0.0001,
  "funding_rate_annualized": 0.1095,
  "mark_price": 42500.50,
  "index_price": 42498.25,
  "venue": "binance",
  "venue_type": "CEX",
  "funding_interval": "8h"
}
```

---

### 3.2 OHLCV Schema

Price and volume data for spot and perpetual markets.

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| `timestamp` | timestamp_ms | Yes | Candle open time (UTC) | Non-null |
| `symbol` | symbol | Yes | Normalized asset symbol | Uppercase |
| `open` | price | Yes | Opening price | > 0 |
| `high` | price | Yes | Highest price | >= max(open, close) |
| `low` | price | Yes | Lowest price | <= min(open, close) |
| `close` | price | Yes | Closing price | > 0 |
| `volume` | quantity | Yes | Base currency volume | >= 0 |
| `volume_usd` | price | No | Notional volume in USD | >= 0 |
| `quote_volume` | quantity | No | Quote currency volume | >= 0 |
| `trade_count` | count | No | Number of trades | >= 0 |
| `taker_buy_volume` | quantity | No | Taker buy base volume | >= 0 |
| `taker_sell_volume` | quantity | No | Taker sell base volume | >= 0 |
| `venue` | venue | Yes | Source exchange | Valid venue enum |
| `venue_type` | category | Yes | Venue classification | CEX, HYBRID, DEX |
| `contract_type` | category | Yes | Instrument type | SPOT, PERPETUAL, FUTURES |
| `timeframe` | category | Yes | Candle duration | 1m, 5m, 15m, 1h, 4h, 1d |
| `is_complete` | boolean | No | Candle is closed | True/False |

**Primary Key:** `(timestamp, symbol, venue, contract_type, timeframe)`

**Validation Rules:**
- `high >= low` (always)
- `high >= max(open, close)` and `low <= min(open, close)`
- `volume >= 0`
- Price change within candle < 50% (flag outliers)
- No negative prices

**OHLCV Consistency Check:**
```
VALID: high >= open AND high >= close AND low <= open AND low <= close AND high >= low
```

---

### 3.3 Open Interest Schema

Aggregate open interest for derivatives markets.

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| `timestamp` | timestamp_ms | Yes | Snapshot time (UTC) | Non-null |
| `symbol` | symbol | Yes | Normalized asset symbol | Uppercase |
| `open_interest` | quantity | Yes | Open interest in contracts/coins | >= 0 |
| `open_interest_usd` | price | Yes | Open interest in USD | >= 0 |
| `long_short_ratio` | rate | No | Long/short account ratio | > 0 |
| `top_trader_long_ratio` | rate | No | Top traders long ratio | [0, 1] |
| `top_trader_short_ratio` | rate | No | Top traders short ratio | [0, 1] |
| `venue` | venue | Yes | Source exchange | Valid venue enum |
| `venue_type` | category | Yes | Venue classification | CEX, HYBRID |
| `contract_type` | category | Yes | Contract specification | PERPETUAL, FUTURES |

**Primary Key:** `(timestamp, symbol, venue, contract_type)`

---

### 3.4 Options Chain Schema

Options market data from Deribit and other options venues.

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| `timestamp` | timestamp_ms | Yes | Snapshot time (UTC) | Non-null |
| `instrument_name` | str | Yes | Full instrument identifier | Venue-specific format |
| `underlying` | symbol | Yes | Underlying asset | BTC, ETH, SOL |
| `strike` | price | Yes | Strike price in USD | > 0 |
| `expiry` | timestamp_ms | Yes | Expiration timestamp | > timestamp |
| `option_type` | category | Yes | Option type | CALL, PUT |
| `mark_price` | price | Yes | Option mark price (underlying units) | >= 0 |
| `mark_price_usd` | price | No | Option mark price in USD | >= 0 |
| `bid_price` | price | No | Best bid price | >= 0 |
| `ask_price` | price | No | Best ask price | >= bid_price |
| `bid_size` | quantity | No | Bid size in contracts | >= 0 |
| `ask_size` | quantity | No | Ask size in contracts | >= 0 |
| `mark_iv` | rate | Yes | Mark implied volatility (annualized) | [0.05, 5.0] |
| `bid_iv` | rate | No | Bid implied volatility | [0.05, 5.0] |
| `ask_iv` | rate | No | Ask implied volatility | [0.05, 5.0] |
| `delta` | rate | Yes | Option delta | [-1.0, 1.0] |
| `gamma` | rate | Yes | Option gamma | >= 0 |
| `vega` | rate | Yes | Option vega | >= 0 |
| `theta` | rate | Yes | Option theta | <= 0 (typically) |
| `rho` | rate | No | Option rho | Any |
| `open_interest` | quantity | No | Open interest in contracts | >= 0 |
| `volume_24h` | quantity | No | 24-hour volume | >= 0 |
| `underlying_price` | price | Yes | Underlying spot/index price | > 0 |
| `days_to_expiry` | count | No | Days until expiration | >= 0 |
| `moneyness` | rate | No | Strike / Underlying | > 0 |
| `venue` | venue | Yes | Source exchange | deribit, lyra, etc. |
| `venue_type` | category | Yes | Venue classification | CEX, DEX |

**Primary Key:** `(timestamp, instrument_name, venue)`

**Validation Rules:**
- Call delta in [0, 1], Put delta in [-1, 0]
- Gamma >= 0
- Vega >= 0
- mark_iv in [5%, 500%] (flag outliers beyond)
- Put-call parity approximately holds

**Greeks Sign Validation:**
```
CALL: 0 <= delta <= 1, gamma >= 0, vega >= 0, theta <= 0
PUT: -1 <= delta <= 0, gamma >= 0, vega >= 0, theta <= 0
```

---

### 3.5 DEX Pool Schema

Liquidity pool data from decentralized exchanges.

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| `timestamp` | timestamp_ms | Yes | Snapshot time (UTC) | Non-null |
| `pool_address` | address | Yes | Pool contract address | Valid address format |
| `pool_id` | str | No | Pool identifier (venue-specific) | Unique per venue |
| `chain` | category | Yes | Blockchain network | ethereum, arbitrum, etc. |
| `protocol` | category | Yes | DEX protocol | uniswap_v3, curve, etc. |
| `token0_address` | address | Yes | First token address | Valid address |
| `token0_symbol` | symbol | Yes | First token symbol | Uppercase |
| `token0_decimals` | count | Yes | First token decimals | [0, 18] |
| `token1_address` | address | Yes | Second token address | Valid address |
| `token1_symbol` | symbol | Yes | Second token symbol | Uppercase |
| `token1_decimals` | count | Yes | Second token decimals | [0, 18] |
| `fee_tier` | count | No | Fee tier in basis points | 1, 5, 30, 100 (Uni V3) |
| `fee_pct` | rate | No | Fee percentage | [0.0001, 0.01] |
| `tvl_usd` | price | Yes | Total value locked in USD | >= 0 |
| `tvl_token0` | quantity | No | TVL in token0 | >= 0 |
| `tvl_token1` | quantity | No | TVL in token1 | >= 0 |
| `volume_24h_usd` | price | No | 24-hour volume in USD | >= 0 |
| `volume_7d_usd` | price | No | 7-day volume in USD | >= 0 |
| `fees_24h_usd` | price | No | 24-hour fees in USD | >= 0 |
| `tx_count_24h` | count | No | 24-hour transaction count | >= 0 |
| `price_token0_usd` | price | No | Token0 price in USD | > 0 |
| `price_token1_usd` | price | No | Token1 price in USD | > 0 |
| `pool_created_at` | timestamp_ms | No | Pool creation timestamp | < timestamp |
| `is_active` | boolean | No | Pool has recent activity | True/False |
| `wash_trading_flag` | boolean | No | Suspected wash trading | True/False |
| `wash_trading_score` | rate | No | Wash trading likelihood | [0, 1] |
| `venue` | venue | Yes | Source venue | Valid venue enum |
| `venue_type` | category | Yes | Classification | DEX |

**Primary Key:** `(timestamp, pool_address, chain)`

**Validation Rules:**
- `tvl_usd >= $500` (quality filter)
- `tx_count_24h >= 10` (activity filter)
- `volume_24h_usd / tvl_usd < 10` (wash trading filter)

**Wash Trading Detection:**
```
wash_trading_flag = True IF:
  - volume_24h_usd / tvl_usd > 10 (excessive turnover)
  - tx_count_24h < 50 AND volume_24h_usd > $1M (few large trades)
  - single address > 50% of volume
```

---

### 3.6 Liquidation Schema

Liquidation events from derivatives venues.

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| `timestamp` | timestamp_ms | Yes | Liquidation time (UTC) | Non-null |
| `symbol` | symbol | Yes | Normalized asset symbol | Uppercase |
| `side` | category | Yes | Position side liquidated | LONG, SHORT |
| `quantity` | quantity | Yes | Liquidated quantity | > 0 |
| `price` | price | Yes | Liquidation price | > 0 |
| `value_usd` | price | Yes | Liquidation value in USD | > 0 |
| `order_type` | category | No | Order execution type | MARKET, LIMIT |
| `venue` | venue | Yes | Source exchange | Valid venue enum |
| `venue_type` | category | Yes | Venue classification | CEX, HYBRID |

**Primary Key:** `(timestamp, symbol, venue, side)`

---

### 3.7 On-Chain Metrics Schema

Blockchain-derived metrics from analytics providers.

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| `timestamp` | timestamp_ms | Yes | Metric timestamp (UTC) | Non-null |
| `symbol` | symbol | Yes | Asset symbol | Uppercase |
| `chain` | category | Yes | Blockchain network | bitcoin, ethereum, etc. |
| `metric_name` | str | Yes | Metric identifier | Valid metric enum |
| `metric_value` | rate | Yes | Metric value | Varies by metric |
| `metric_unit` | str | No | Value unit | USD, BTC, count, etc. |
| `source` | venue | Yes | Data provider | glassnode, arkham, etc. |
| `resolution` | category | Yes | Time resolution | 1h, 1d |

**Common Metrics:**

| Metric Name | Description | Unit | Typical Range |
|-------------|-------------|------|---------------|
| `exchange_inflow` | Tokens flowing to exchanges | Token units | Varies |
| `exchange_outflow` | Tokens leaving exchanges | Token units | Varies |
| `exchange_netflow` | Net exchange flow | Token units | Any |
| `exchange_reserve` | Total exchange holdings | Token units | > 0 |
| `active_addresses` | Daily active addresses | Count | > 0 |
| `new_addresses` | New addresses created | Count | >= 0 |
| `transaction_count` | Daily transactions | Count | > 0 |
| `transfer_volume` | On-chain transfer volume | USD | > 0 |
| `nvt_ratio` | Network Value to Transactions | Ratio | > 0 |
| `mvrv_ratio` | Market Value to Realized Value | Ratio | Any |
| `sopr` | Spent Output Profit Ratio | Ratio | Any |
| `puell_multiple` | Mining revenue vs 365d MA | Ratio | > 0 |
| `whale_transactions` | Transactions > $100k | Count | >= 0 |

---

### 3.8 BTC Futures Term Structure Schema

Dated futures contract data for term structure analysis.

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| `timestamp` | timestamp_ms | Yes | Price snapshot time (UTC) | Non-null |
| `symbol` | symbol | Yes | Base asset | BTC, ETH |
| `contract_code` | str | Yes | Contract identifier | Venue-specific |
| `expiry` | timestamp_ms | Yes | Contract expiration | > timestamp |
| `days_to_expiry` | count | Yes | Days until expiration | >= 0 |
| `mark_price` | price | Yes | Contract mark price | > 0 |
| `index_price` | price | Yes | Spot/index reference | > 0 |
| `basis` | price | No | Mark - Index price | Any |
| `basis_pct` | rate | No | Basis as percentage | Any |
| `basis_annualized` | rate | No | Annualized basis | Any |
| `open_interest` | quantity | No | Contract OI | >= 0 |
| `open_interest_usd` | price | No | OI in USD | >= 0 |
| `volume_24h` | quantity | No | 24h volume | >= 0 |
| `funding_rate` | rate | No | If perpetual component | [-1, 1] |
| `venue` | venue | Yes | Source exchange | binance, cme, deribit |
| `venue_type` | category | Yes | Classification | CEX |
| `contract_type` | category | Yes | Contract type | QUARTERLY, PERPETUAL |

**Primary Key:** `(timestamp, contract_code, venue)`

**Basis Calculation:**
```
basis = mark_price - index_price
basis_pct = basis / index_price
basis_annualized = basis_pct * (365 / days_to_expiry)
```

---

## 4. Enumerated Types

### 4.1 Venue Enumeration

| Venue Code | Full Name | Type | Funding Interval |
|------------|-----------|------|------------------|
| `binance` | Binance Futures | CEX | 8h |
| `bybit` | Bybit Derivatives | CEX | 8h |
| `okx` | OKX Futures | CEX | 8h |
| `coinbase` | Coinbase Advanced | CEX | N/A |
| `kraken` | Kraken Futures | CEX | 8h |
| `cme` | CME Group | CEX | N/A (dated) |
| `deribit` | Deribit | CEX | 8h |
| `hyperliquid` | Hyperliquid L1 | HYBRID | 1h |
| `dydx_v4` | dYdX V4 (Cosmos) | HYBRID | 1h |
| `gmx` | GMX (Arbitrum) | HYBRID | Variable |
| `vertex` | Vertex Protocol | HYBRID | 1h |
| `uniswap_v3` | Uniswap V3 | DEX | N/A |
| `curve` | Curve Finance | DEX | N/A |
| `sushiswap` | SushiSwap | DEX | N/A |
| `jupiter` | Jupiter (Solana) | DEX | N/A |

### 4.2 Venue Type Enumeration

| Type | Description | Custody | Settlement |
|------|-------------|---------|------------|
| `CEX` | Centralized Exchange | Exchange-held | Off-chain |
| `HYBRID` | Hybrid/Semi-Decentralized | Smart contract | On-chain |
| `DEX` | Decentralized Exchange | Self-custody | On-chain |

### 4.3 Contract Type Enumeration

| Type | Description | Expiry | Funding |
|------|-------------|--------|---------|
| `SPOT` | Spot market | N/A | N/A |
| `PERPETUAL` | Perpetual swap | Never | Yes |
| `FUTURES` | Dated futures | Fixed date | No |
| `QUARTERLY` | Quarterly futures | End of quarter | No |
| `OPTION` | Options contract | Fixed date | N/A |

### 4.4 Timeframe Enumeration

| Code | Duration | Candles/Day | Primary Use |
|------|----------|-------------|-------------|
| `1m` | 1 minute | 1,440 | High-frequency |
| `5m` | 5 minutes | 288 | Intraday |
| `15m` | 15 minutes | 96 | Intraday |
| `1h` | 1 hour | 24 | **Primary** |
| `4h` | 4 hours | 6 | Swing trading |
| `1d` | 1 day | 1 | Daily analysis |

### 4.5 Chain Enumeration

| Chain Code | Full Name | Type | Block Time |
|------------|-----------|------|------------|
| `ethereum` | Ethereum Mainnet | L1 | ~12s |
| `arbitrum` | Arbitrum One | L2 | ~0.25s |
| `optimism` | Optimism | L2 | ~2s |
| `polygon` | Polygon PoS | L2 | ~2s |
| `base` | Base | L2 | ~2s |
| `solana` | Solana | L1 | ~0.4s |
| `avalanche` | Avalanche C-Chain | L1 | ~2s |
| `bsc` | BNB Smart Chain | L1 | ~3s |
| `cosmos` | Cosmos Hub | L1 | ~6s |

### 4.6 Option Type Enumeration

| Type | Description | Delta Range | Exercise |
|------|-------------|-------------|----------|
| `CALL` | Right to buy | [0, 1] | At/before expiry |
| `PUT` | Right to sell | [-1, 0] | At/before expiry |

### 4.7 Side Enumeration

| Side | Description | Position | P&L Direction |
|------|-------------|----------|---------------|
| `LONG` | Buy/long position | Positive | Price increase |
| `SHORT` | Sell/short position | Negative | Price decrease |
| `BUY` | Buy order | Entry long | - |
| `SELL` | Sell order | Entry short | - |

### 4.8 Quality Score Enumeration

| Score | Name | Threshold | Description |
|-------|------|-----------|-------------|
| 5 | `EXCELLENT` | > 95% | Validated |
| 4 | `GOOD` | 85-95% | Minor issues |
| 3 | `ACCEPTABLE` | 70-85% | Usable with caveats |
| 2 | `POOR` | 50-70% | Significant issues |
| 1 | `CRITICAL` | < 50% | Not usable |

---

## 5. Cross-Venue Normalization

### 5.1 Symbol Normalization

All symbols are normalized to base asset only (no quote currency suffix).

| Source Format | Normalized |
|---------------|------------|
| `BTCUSDT` (Binance) | `BTC` |
| `BTC-USDT` (OKX) | `BTC` |
| `BTC/USDT` (Generic) | `BTC` |
| `BTC:USDT` (Hyperliquid) | `BTC` |
| `XBTUSD` (BitMEX) | `BTC` |
| `tBTCUSD` (Bitfinex) | `BTC` |

**Normalization Rules:**
1. Convert to uppercase
2. Remove quote currency suffix (USDT, USD, BUSD, USDC)
3. Remove special characters (/, -, :)
4. Map legacy symbols (XBT → BTC)

### 5.2 Funding Rate Normalization

Funding rates must be normalized for cross-venue comparison.

| Venue | Native Interval | Periods/Year | Normalization to 8h |
|-------|-----------------|--------------|---------------------|
| Binance | 8h | 1,095 | None (native) |
| Bybit | 8h | 1,095 | None (native) |
| Hyperliquid | 1h | 8,760 | Sum 8 hourly rates |
| dYdX V4 | 1h | 8,760 | Sum 8 hourly rates |

**Annualization Formulas:**
```
8h_rate_annualized = 8h_rate × 1,095
1h_rate_annualized = 1h_rate × 8,760
1h_to_8h_equivalent = SUM(8 consecutive 1h rates) OR AVG(8 rates) × 8
```

### 5.3 Timestamp Alignment

Standard funding settlement times (UTC):

| Venue | Settlement Times | Alignment Strategy |
|-------|------------------|-------------------|
| Binance | 00:00, 08:00, 16:00 | Floor to nearest 8h |
| Bybit | 00:00, 08:00, 16:00 | Floor to nearest 8h |
| Hyperliquid | Every hour | Group into 8h windows |
| dYdX V4 | Every hour | Group into 8h windows |

**Alignment Algorithm:**
```python
def align_to_8h(timestamp):
    hour = timestamp.hour
    aligned_hour = (hour // 8) * 8  # 0, 8, or 16
    return timestamp.replace(hour=aligned_hour, minute=0, second=0, microsecond=0)
```

### 5.4 Price Normalization

All prices normalized to USD for cross-venue comparison.

| Base Currency | Conversion | Source |
|---------------|------------|--------|
| USD | None | Direct |
| USDT | × USDT/USD rate | CoinGecko |
| USDC | × USDC/USD rate | CoinGecko |
| BUSD | × BUSD/USD rate | CoinGecko |
| BTC | × BTC/USD spot | Binance |
| ETH | × ETH/USD spot | Binance |

---

## 6. Data Relationships

### 6.1 Entity Relationship Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  FUNDING_RATE   │────>│     SYMBOL      │<────│     OHLCV       │
│                 │     │                 │     │                 │
│ - timestamp     │     │ - symbol (PK)   │     │ - timestamp     │
│ - symbol (FK)   │     │ - name          │     │ - symbol (FK)   │
│ - venue (FK)    │     │ - sector        │     │ - venue (FK)    │
│ - funding_rate  │     │ - market_cap    │     │ - open, high... │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                       │
         │                      │                       │
         v                      v                       v
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     VENUE       │     │  OPEN_INTEREST  │     │  LIQUIDATIONS   │
│                 │     │                 │     │                 │
│ - venue (PK)    │     │ - timestamp     │     │ - timestamp     │
│ - venue_type    │     │ - symbol (FK)   │     │ - symbol (FK)   │
│ - funding_int   │     │ - venue (FK)    │     │ - venue (FK)    │
│ - maker_fee     │     │ - open_interest │     │ - side, qty     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                v
                        ┌─────────────────┐
                        │  OPTIONS_CHAIN  │
                        │                 │
                        │ - timestamp     │
                        │ - underlying    │
                        │ - strike        │
                        │ - expiry        │
                        │ - mark_iv       │
                        │ - greeks...     │
                        └─────────────────┘
```

### 6.2 Join Keys

| Join | Left Table | Right Table | Key Columns |
|------|------------|-------------|-------------|
| Funding + OHLCV | funding_rates | ohlcv | timestamp, symbol, venue |
| Funding + OI | funding_rates | open_interest | timestamp, symbol, venue |
| Cross-venue Funding | funding_rates (A) | funding_rates (B) | timestamp, symbol |
| Options + Underlying | options_chain | ohlcv | timestamp, underlying |
| Pool + Token | dex_pools | tokens | token0_address OR token1_address |

---

## 7. Quality Attributes

### 7.1 Field-Level Quality Metrics

| Field | Completeness Target | Accuracy Check | Consistency Check |
|-------|---------------------|----------------|-------------------|
| `timestamp` | 100% | UTC validation | Monotonic increase |
| `symbol` | 100% | Enum validation | Case-normalized |
| `funding_rate` | > 95% | Range [-1, 1] | Cross-venue correlation > 0.9 |
| `mark_price` | > 98% | > 0 | Deviation from index < 1% |
| `open` | > 99% | > 0 | Within high-low range |
| `high` | > 99% | >= max(open, close) | No impossible values |
| `low` | > 99% | <= min(open, close) | No impossible values |
| `close` | > 99% | > 0 | Continuity with next open |
| `volume` | > 95% | >= 0 | No negative values |
| `mark_iv` | > 90% | [5%, 500%] | Smile consistency |
| `delta` | > 95% | Call [0,1], Put [-1,0] | Monotonic in strike |

### 7.2 Dataset-Level Quality Metrics

| Metric | Definition | Target | Calculation |
|--------|------------|--------|-------------|
| Completeness | % of expected records present | > 95% | actual / expected records |
| Coverage | % of date range covered | > 98% | days with data / total days |
| Consistency | Cross-venue correlation | > 0.95 | corr(venue_a, venue_b) |
| Timeliness | Hours since last update | < 24h | now - max(timestamp) |
| Validity | % of records passing validation | > 99% | valid / total records |
| Uniqueness | % of unique primary keys | 100% | unique PKs / total records |

### 7.3 Outlier Thresholds

| Data Type | Field | Warning Threshold | Error Threshold |
|-----------|-------|-------------------|-----------------|
| Funding | funding_rate | |rate| > 0.01 (1%) | |rate| > 0.05 (5%) |
| Funding | funding_rate_annualized | |rate| > 100% | |rate| > 500% |
| OHLCV | price_change_pct | |change| > 20% | |change| > 50% |
| OHLCV | volume_spike | > 10x avg | > 50x avg |
| Options | mark_iv | > 200% | > 500% |
| Options | delta | |delta| > 1.0 | N/A (invalid) |
| DEX | volume/tvl ratio | > 5 | > 10 (wash trading) |

---

## 8. Storage Specifications

### 8.1 File Format

**Primary Format:** Apache Parquet

| Property | Specification |
|----------|---------------|
| Compression | gzip (default), snappy (fast queries) |
| Row Group Size | 100,000 rows |
| Page Size | 1 MB |
| Dictionary Encoding | Enabled for string columns |
| Statistics | Min/max per column |

### 8.2 Partitioning Strategy

| Dataset | Partition Columns | Partition Size Target |
|---------|-------------------|----------------------|
| funding_rates | year, month | ~100K rows |
| ohlcv | year, month, symbol | ~50K rows |
| options_chain | year, month, underlying | ~1M rows |
| dex_pools | chain, date | ~10K rows |
| on_chain_metrics | source, date | ~50K rows |

### 8.3 Directory Structure

```
data/
├── raw/                          # Unprocessed collector output
│   ├── cex/
│   │   ├── binance/
│   │   │   ├── funding_rates/
│   │   │   │   ├── year=2022/
│   │   │   │   │   └── month=01/
│   │   │   │   │       └── data.parquet
│   │   │   └── ohlcv/
│   │   └── bybit/
│   ├── hybrid/
│   │   ├── hyperliquid/
│   │   └── dydx_v4/
│   ├── dex/
│   │   ├── uniswap_v3/
│   │   │   ├── ethereum/
│   │   │   └── arbitrum/
│   │   └── curve/
│   └── options/
│       └── deribit/
├── processed/                    # Normalized, validated data
│   ├── funding_rates_consolidated.parquet
│   ├── ohlcv_hourly_all_venues.parquet
│   ├── options_chain_deribit.parquet
│   └── pairs_universe.parquet
└── metadata/
    ├── data_dictionary.md        # This document
    ├── data_quality_report.md
    ├── source_attribution.md
    └── collection_logs/
```

### 8.4 Estimated Storage Requirements

| Dataset | Records (3 years) | Raw Size | Compressed |
|---------|-------------------|----------|------------|
| Funding Rates (all venues) | ~2.5M | 200 MB | 50 MB |
| OHLCV Hourly (50 symbols) | ~1.3M | 400 MB | 100 MB |
| Open Interest | ~500K | 50 MB | 15 MB |
| Options Chain | ~50M | 5 GB | 500 MB |
| DEX Pools | ~500K | 100 MB | 25 MB |
| On-Chain Metrics | ~2M | 200 MB | 50 MB |
| **Total** | **~57M** | **~6 GB** | **~750 MB** |

---

## 9. Derived Fields

### 9.1 Funding Rate Derived Fields

| Derived Field | Formula | Unit |
|---------------|---------|------|
| `funding_rate_pct` | `funding_rate × 100` | % per period |
| `funding_rate_annualized` | `funding_rate × periods_per_year` | Decimal |
| `funding_rate_annualized_pct` | `funding_rate_annualized × 100` | % |
| `funding_usd_per_100k` | `funding_rate × 100,000` | USD |
| `funding_spread` | `rate_venue_a - rate_venue_b` | Decimal |
| `funding_spread_annualized` | `funding_spread × periods_per_year` | Decimal |

### 9.2 Price Derived Fields

| Derived Field | Formula | Unit |
|---------------|---------|------|
| `return_1h` | `(close - prev_close) / prev_close` | Decimal |
| `return_1h_pct` | `return_1h × 100` | % |
| `log_return` | `ln(close / prev_close)` | Decimal |
| `range_pct` | `(high - low) / close × 100` | % |
| `body_pct` | `abs(close - open) / close × 100` | % |
| `vwap` | `volume_usd / volume` | USD |

### 9.3 Options Derived Fields

| Derived Field | Formula | Unit |
|---------------|---------|------|
| `moneyness` | `strike / underlying_price` | Ratio |
| `log_moneyness` | `ln(strike / underlying_price)` | Decimal |
| `time_value` | `mark_price - intrinsic_value` | Underlying units |
| `intrinsic_value` | `max(0, underlying - strike)` for call | Underlying units |
| `days_to_expiry` | `(expiry - timestamp) / 86400` | Days |
| `iv_percentile` | `percentile_rank(mark_iv)` | [0, 100] |

### 9.4 Basis Derived Fields

| Derived Field | Formula | Unit |
|---------------|---------|------|
| `basis` | `futures_price - spot_price` | USD |
| `basis_pct` | `basis / spot_price` | Decimal |
| `basis_annualized` | `basis_pct × (365 / days_to_expiry)` | Decimal |
| `carry` | `basis_annualized - funding_rate_annualized` | Decimal |

---

## 10. Changelog

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 3.0.0 | 2026-01-31 | Quant Research | Added survivorship bias tracking, 32 venue coverage |
| 2.0.0 | 2025-01-28 | Quant Research | Complete rewrite, added all schemas |
| 1.0.0 | 2025-01-15 | Quant Research | Initial version |

---

## Appendix A: Validation Code Reference

```python
# Schema validation example
FUNDING_RATE_SCHEMA = {
    'timestamp': {'type': 'timestamp_ms', 'required': True, 'nullable': False},
    'symbol': {'type': 'string', 'required': True, 'nullable': False, 'pattern': r'^[A-Z0-9]+$'},
    'funding_rate': {'type': 'float64', 'required': True, 'nullable': False, 'min': -1.0, 'max': 1.0},
    'mark_price': {'type': 'float64', 'required': True, 'nullable': False, 'min': 0},
    'venue': {'type': 'string', 'required': True, 'nullable': False, 'enum': VENUE_ENUM},
    'venue_type': {'type': 'string', 'required': True, 'nullable': False, 'enum': ['CEX', 'HYBRID', 'DEX']},
    'funding_interval': {'type': 'string', 'required': True, 'nullable': False, 'enum': ['1h', '8h']},
}

# OHLCV consistency validation
def validate_ohlcv_consistency(row):
    return (
        row['high'] >= row['low'] and
        row['high'] >= row['open'] and
        row['high'] >= row['close'] and
        row['low'] <= row['open'] and
        row['low'] <= row['close'] and
        row['volume'] >= 0
    )

# Options Greeks sign validation
def validate_greeks(row):
    if row['option_type'] == 'CALL':
        return 0 <= row['delta'] <= 1 and row['gamma'] >= 0 and row['vega'] >= 0
    else:  # PUT
        return -1 <= row['delta'] <= 0 and row['gamma'] >= 0 and row['vega'] >= 0
```

---

## Appendix B: Quick Reference Card

### Funding Rate Quick Reference

| Property | Value |
|----------|-------|
| Primary venues | Binance, Bybit, Hyperliquid, dYdX V4 |
| CEX interval | 8 hours (00:00, 08:00, 16:00 UTC) |
| Hybrid interval | 1 hour |
| Typical range | ±0.01% per 8h (±10.95% annualized) |
| Annualization (8h) | × 1,095 |
| Annualization (1h) | × 8,760 |

### OHLCV Quick Reference

| Property | Value |
|----------|-------|
| Primary timeframe | 1 hour |
| Contract types | SPOT, PERPETUAL, FUTURES |
| Primary venues | Binance, Bybit (CEX), Hyperliquid (HYBRID) |
| Required fields | timestamp, symbol, O, H, L, C, volume, venue |

### Options Quick Reference

| Property | Value |
|----------|-------|
| Primary venue | Deribit (~90% market share) |
| Underlyings | BTC, ETH, SOL |
| Greeks | delta, gamma, vega, theta, rho |
| IV range | 5% - 500% (flag outliers) |

---

*Document prepared for Crypto Statistical Arbitrage project Phase 1 completion.*
*This data dictionary is the authoritative reference for all data schemas in the system.*
