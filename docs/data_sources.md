# Data Sources
## Crypto Statistical Arbitrage Multi-Venue System

**Version:** 2.0
**Last Updated:** January 2025
**Author:** Tamer Atesyakar

---

## 1. Overview

This document catalogues all data sources used across the four trading strategies. Each source is evaluated for coverage, reliability, cost, and suitability for both CEX and DEX analysis.

### Design Principles

- **CEX as primary source** for price data (higher liquidity, lower noise)
- **DEX data for validation** and cross-checking, plus DEX-specific strategies
- **Cross-validation** across 3+ independent sources for all core datasets
- **Parquet storage** for all processed data (10-50x compression vs CSV)
- **UTC timestamps** throughout, millisecond precision

---

## 2. Centralized Exchange (CEX) Sources

### 2.1 Binance (Primary)

| Attribute | Detail |
|-----------|--------|
| **Data Types** | Funding rates, OHLCV (spot + futures), open interest, liquidations |
| **Coverage** | 500+ trading pairs, 2020-present |
| **Frequency** | 1h OHLCV, 8h funding rates |
| **API** | REST + WebSocket, 1200 req/min |
| **Cost** | Free |
| **Reliability** | High (99.9%+ uptime, occasional maintenance windows) |

**Used For:**
- Phase 1 (Funding Rate Arb): Primary funding rate source
- Phase 2 (Altcoin Pairs): Primary OHLCV for CEX universe
- Phase 3 (BTC Futures): Quarterly and bi-quarterly futures

**Collection Method:** CCXT unified interface with exponential backoff retry logic. Bulk historical download for OHLCV, paginated API for funding rates.

### 2.2 OKX

| Attribute | Detail |
|-----------|--------|
| **Data Types** | Funding rates, OHLCV, open interest |
| **Coverage** | 300+ pairs |
| **Frequency** | 1h OHLCV, 8h funding |
| **API** | REST, 600 req/min |
| **Cost** | Free |

**Used For:** Cross-validation source for funding rates and OHLCV. Part of 4-source cross-validation pipeline.

### 2.3 Coinbase

| Attribute | Detail |
|-----------|--------|
| **Data Types** | Spot OHLCV |
| **Coverage** | Major coins (150+ pairs) |
| **Frequency** | 1h OHLCV |
| **API** | REST |
| **Cost** | Free |

**Used For:** Cross-validation of spot prices. US-regulated exchange provides institutional-grade price reference.

### 2.4 Bybit

| Attribute | Detail |
|-----------|--------|
| **Data Types** | Funding rates, OHLCV |
| **Coverage** | 200+ perpetual pairs |
| **Frequency** | 8h funding, 1h OHLCV |
| **Cost** | Free |

**Used For:** Additional funding rate source for Phase 1 cross-venue analysis.

### 2.5 Deribit

| Attribute | Detail |
|-----------|--------|
| **Data Types** | BTC/ETH options (all strikes, expiries), futures |
| **Coverage** | 2019-present |
| **Frequency** | Hourly snapshots |
| **API** | REST, ~20 req/sec |
| **Cost** | Free |

**Used For:** Phase 3 BTC futures curve, options data reference.

---

## 3. Decentralized Exchange (DEX) Sources

### 3.1 Hyperliquid

| Attribute | Detail |
|-----------|--------|
| **Data Types** | Perpetual funding rates, OHLCV, order book |
| **Coverage** | 100+ perpetual markets |
| **Frequency** | 1h funding, real-time candles |
| **API** | REST + WebSocket (Arbitrum L1 settlement) |
| **Cost** | Free |
| **Classification** | Hybrid (on-chain settlement, order book model) |

**Used For:**
- Phase 1: Cross-venue funding rate arbitrage (vs Binance)
- Phase 2: Cross-validation source
- Phase 3: On-chain perp term structure proxy

### 3.2 The Graph (Uniswap V3 Subgraphs)

| Attribute | Detail |
|-----------|--------|
| **Data Types** | Pool swaps, liquidity, TVL, fees |
| **Coverage** | Ethereum, Arbitrum, Optimism, Polygon, Base |
| **Frequency** | Per-block (aggregated to hourly) |
| **API** | GraphQL, paginated (1000 results/query) |
| **Cost** | Free tier with rate limits |

**Used For:** DEX universe construction, TVL filtering, wash trading detection.

### 3.3 dYdX V4

| Attribute | Detail |
|-----------|--------|
| **Data Types** | Perpetual funding rates, OHLCV |
| **Coverage** | 50+ markets |
| **API** | Indexer REST API (Cosmos chain) |
| **Cost** | Free |

**Used For:** Phase 1 cross-venue funding analysis, Phase 3 term structure.

### 3.4 GeckoTerminal / DEXScreener

| Attribute | Detail |
|-----------|--------|
| **Data Types** | DEX prices, pool metrics, new listings |
| **Coverage** | 100+ chains |
| **API** | REST |
| **Cost** | Free with rate limits |

**Used For:** DEX universe discovery, long-tail token price data.

---

## 4. On-Chain & Alternative Data

### 4.1 DefiLlama

| Attribute | Detail |
|-----------|--------|
| **Data Types** | TVL, protocol fees, revenue |
| **Coverage** | 2000+ DeFi protocols |
| **Cost** | Free |

**Used For:** DEX universe filtering (TVL thresholds), sector classification.

### 4.2 Glassnode (Free Tier)

| Attribute | Detail |
|-----------|--------|
| **Data Types** | BTC/ETH on-chain fundamentals |
| **Coverage** | Bitcoin, Ethereum |
| **Cost** | Free tier (limited metrics) |

**Used For:** On-chain flow signals for regime detection (bonus: on-chain data integration).

### 4.3 CoinGecko

| Attribute | Detail |
|-----------|--------|
| **Data Types** | Market cap, volume, metadata |
| **Coverage** | 10,000+ coins |
| **Cost** | Free with rate limits |

**Used For:** Token metadata, market cap filtering for universe construction.

---

## 5. Aggregated & Derived Sources

### 5.1 CryptoCompare

| Attribute | Detail |
|-----------|--------|
| **Data Types** | Aggregated OHLCV across venues |
| **Coverage** | 5000+ pairs |
| **Cost** | Free tier (100k calls/month) |

**Used For:** Cross-venue price aggregation, data gap filling.

### 5.2 CCXT Library

| Attribute | Detail |
|-----------|--------|
| **Type** | Python library (unified API) |
| **Coverage** | 100+ exchanges |
| **Cost** | Open source |

**Used For:** Unified data collection interface across all CEX sources.

---

## 6. Cross-Validation Framework

All core datasets are validated across 3+ independent sources:

| Dataset | Primary | Secondary | Tertiary | Fourth |
|---------|---------|-----------|----------|--------|
| Funding Rates | Binance | Hyperliquid | Bybit | dYdX V4 |
| Spot OHLCV | Binance | OKX | Coinbase | CryptoCompare |
| DEX Prices | Uniswap V3 | GeckoTerminal | DEXScreener | - |
| BTC Futures | Binance | Deribit | CME (delayed) | - |

### Validation Criteria

- **Price correlation**: > 0.95 between sources
- **MAPE threshold**: < 5% mean absolute percentage error
- **Gap tolerance**: < 5% missing data for core assets
- **Outlier detection**: Flag prices > 50% single-bar moves

---

## 7. Data Quality Summary

| Metric | Target | Achieved |
|--------|--------|----------|
| Coverage period | 2022-2024 minimum | 2022-01-01 to 2025-01-31 |
| Missing data (core) | < 5% | < 2.1% |
| Cross-validation sources | 3+ | 4 (Binance, OKX, Coinbase, Hyperliquid) |
| Outlier treatment | Documented | Winsorized at 5-sigma |
| Survivorship bias | Addressed | Delisting tracker with 47 events |
| DEX wash trading | Filtered | < 10 trades/hour filter applied |

---

## 8. Known Limitations

1. **DEX historical depth**: Most DEX data starts mid-2021 (Uniswap V3 launch). Earlier periods use CEX-only analysis.
2. **Options data gaps**: Deribit data has intermittent gaps during high-volatility periods.
3. **L2 data fragmentation**: Each L2 chain requires separate subgraph queries with different schemas.
4. **CME data**: Only delayed/end-of-day available for free. Real-time requires paid feed.
5. **On-chain data costs**: Full Glassnode/Nansen access requires paid tiers. Free tier metrics used where available.
