# Source Attribution

## Crypto Statistical Arbitrage - Multi-Venue Trading System

**Date:** January 31, 2026
**Version:** 3.0
**Author:** Tamer Atesyakar

---

## Executive Summary

This document provides complete attribution for all data sources used in the crypto statistical arbitrage system. Each dataset is traced to its exact origin, including API endpoints, authentication requirements, rate limits, and historical data availability. This attribution ensures full reproducibility and transparency of the data collection pipeline.

**Total Data Sources:** 46 unique collectors across 8 categories  
**Coverage Period:** 2022-01-01 to 2024-12-31  
**Primary Use Cases:** Funding rate arbitrage, pairs trading, futures curve, options volatility

---

## Table of Contents

1. [Centralized Exchanges (CEX)](#1-centralized-exchanges-cex)
2. [Hybrid Venues](#2-hybrid-venues)
3. [Decentralized Exchanges (DEX)](#3-decentralized-exchanges-dex)
4. [Options Venues](#4-options-venues)
5. [On-Chain Analytics](#5-on-chain-analytics)
6. [Market Data Aggregators](#6-market-data-aggregators)
7. [Alternative Data](#7-alternative-data)
8. [Social & Sentiment](#8-social--sentiment)
9. [Data Flow Summary](#9-data-flow-summary)
10. [Licensing & Terms of Use](#10-licensing--terms-of-use)

---

## 1. Centralized Exchanges (CEX)

### 1.1 Binance

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/cex/binance_collector.py` |
| **Venue Type** | CEX |
| **API Endpoints** | |
| - Spot API | `https://api.binance.com` |
| - Futures API | `https://fapi.binance.com` |
| **Authentication** | API Key + Secret (optional for public endpoints) |
| **Rate Limits** | 1,200 requests/minute (weight-based) |
| **Historical Data** | 2019-present (futures), 2017-present (spot) |

**Data Types Collected:**

| Data Type | Endpoint | Frequency | Fields |
|-----------|----------|-----------|--------|
| Funding Rates | `/fapi/v1/fundingRate` | 8-hourly | timestamp, symbol, fundingRate, markPrice |
| OHLCV (Futures) | `/fapi/v1/klines` | 1m to 1M | open, high, low, close, volume, quoteVolume |
| OHLCV (Spot) | `/api/v3/klines` | 1m to 1M | open, high, low, close, volume |
| Open Interest | `/fapi/v1/openInterest` | On-demand | symbol, openInterest, time |
| Liquidations | `/fapi/v1/allForceOrders` | Real-time | symbol, side, price, qty, time |

**API Documentation:** https://binance-docs.github.io/apidocs/futures/en/

---

### 1.2 Bybit

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/cex/bybit_collector.py` |
| **Venue Type** | CEX |
| **API Endpoint** | `https://api.bybit.com` |
| **Authentication** | API Key + Secret (optional for public) |
| **Rate Limits** | 120 requests/minute (IP-based) |
| **Historical Data** | 2020-present |

**Data Types Collected:**

| Data Type | Endpoint | Frequency | Fields |
|-----------|----------|-----------|--------|
| Funding Rates | `/v5/market/funding/history` | 8-hourly | fundingRate, fundingRateTimestamp |
| OHLCV | `/v5/market/kline` | 1m to 1M | open, high, low, close, volume |
| Open Interest | `/v5/market/open-interest` | On-demand | openInterest, timestamp |

**API Documentation:** https://bybit-exchange.github.io/docs/v5/intro

**Cross-Validation Role:** Primary validation source for Binance funding rates (correlation >0.97)

---

### 1.3 OKX

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/cex/okx_collector.py` |
| **Venue Type** | CEX |
| **API Endpoint** | `https://www.okx.com` |
| **Authentication** | API Key + Secret + Passphrase |
| **Rate Limits** | 20 requests/2 seconds (public), 60 requests/2 seconds (private) |
| **Historical Data** | 2020-present |

**Data Types Collected:**

| Data Type | Endpoint | Frequency | Fields |
|-----------|----------|-----------|--------|
| Funding Rates | `/api/v5/public/funding-rate-history` | 8-hourly | fundingRate, fundingTime |
| OHLCV | `/api/v5/market/candles` | 1m to 1M | open, high, low, close, vol |
| Mark Price | `/api/v5/public/mark-price` | On-demand | markPx, ts |

**API Documentation:** https://www.okx.com/docs-v5/en/

---

### 1.4 Coinbase

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/cex/coinbase_collector.py` |
| **Venue Type** | CEX |
| **API Endpoint** | `https://api.coinbase.com` |
| **Authentication** | JWT with EC Private Key |
| **Rate Limits** | 100 requests/minute |
| **Historical Data** | 2015-present (spot only) |

**Data Types Collected:**

| Data Type | Endpoint | Frequency | Fields |
|-----------|----------|-----------|--------|
| OHLCV (Spot) | `/api/v3/brokerage/products/{id}/candles` | 1m to 1d | open, high, low, close, volume |
| Products | `/api/v3/brokerage/products` | On-demand | product_id, base_currency, quote_currency |

**API Documentation:** https://docs.cloud.coinbase.com/advanced-trade-api/

**Note:** Coinbase does not offer perpetual futures. Used for spot price validation.

---

### 1.5 Kraken

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/cex/kraken_collector.py` |
| **Venue Type** | CEX |
| **API Endpoints** | |
| - Spot API | `https://api.kraken.com/0/public` |
| - Futures API | `https://futures.kraken.com/derivatives/api/v3` |
| **Authentication** | API Key + Secret (optional for public) |
| **Rate Limits** | Tier-based (15-20 calls/second) |
| **Historical Data** | 2013-present (spot), 2020-present (futures) |

**Data Types Collected:**

| Data Type | Endpoint | Frequency | Fields |
|-----------|----------|-----------|--------|
| OHLCV (Spot) | `/OHLC` | 1m to 1w | open, high, low, close, vwap, volume |
| Funding Rates | `/tickers` (futures) | 8-hourly | fundingRate, fundingRatePrediction |
| Ticker | `/Ticker` | On-demand | ask, bid, last, volume |

**API Documentation:** https://docs.kraken.com/rest/

---

### 1.6 CME Group

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/cex/cme_collector.py` |
| **Venue Type** | CEX (Institutional) |
| **Data Sources** | FRED, Nasdaq Data Link (Quandl) |
| **Authentication** | API Key (for Nasdaq Data Link) |
| **Rate Limits** | Varies by source |
| **Historical Data** | 2017-present (BTC futures) |

**Data Types Collected:**

| Data Type | Source | Frequency | Fields |
|-----------|--------|-----------|--------|
| BTC Futures Prices | Nasdaq Data Link | Daily | open, high, low, close, volume, OI |
| Settlement Prices | CME via FRED | Daily | settlement_price |

**Note:** CME data is delayed (typically 15-20 minutes). Used for institutional term structure analysis.

---

## 2. Hybrid Venues

### 2.1 Hyperliquid

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/hybrid/hyperliquid_collector.py` |
| **Venue Type** | Hybrid (On-chain settlement, order book model) |
| **API Endpoint** | `https://api.hyperliquid.xyz/info` |
| **Authentication** | None required (FREE) |
| **Rate Limits** | ~100 requests/minute |
| **Historical Data** | February 2023-present |
| **Settlement Chain** | Arbitrum L1 |

**Data Types Collected:**

| Data Type | Method | Frequency | Fields |
|-----------|--------|-----------|--------|
| Funding Rates | POST `{"type": "fundingHistory"}` | **HOURLY** | coin, fundingRate, time |
| OHLCV | POST `{"type": "candleSnapshot"}` | 1m to 1d | open, high, low, close, volume |
| Open Interest | POST `{"type": "metaAndAssetCtxs"}` | On-demand | openInterest |
| Order Book | POST `{"type": "l2Book"}` | Real-time | levels, prices, sizes |

**API Documentation:** https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api

**Critical Note:** Hyperliquid uses **HOURLY** funding (not 8-hour like CEX). Normalization required:
- To compare with 8h CEX: multiply by 8
- To annualize: multiply by 8,760

**Cross-Validation:** Lower correlation with CEX (~0.91) expected due to different participant demographics.

---

### 2.2 dYdX V4

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/hybrid/dydx_collector.py` |
| **Venue Type** | Hybrid (Cosmos appchain) |
| **API Endpoint** | `https://indexer.dydx.trade/v4` |
| **Authentication** | None required (FREE) |
| **Rate Limits** | ~100 requests/minute |
| **Historical Data** | October 2023-present (V4 launch) |
| **Settlement Chain** | dYdX Chain (Cosmos SDK) |

**Data Types Collected:**

| Data Type | Endpoint | Frequency | Fields |
|-----------|----------|-----------|--------|
| Funding Rates | `/historicalFunding/{market}` | **HOURLY** | rate, effectiveAt |
| OHLCV | `/candles/perpetualMarkets/{market}` | 1m to 1d | open, high, low, close, volume |
| Markets | `/perpetualMarkets` | On-demand | clobPairId, ticker, status |

**API Documentation:** https://docs.dydx.exchange/

**Critical Note:** dYdX V4 uses **HOURLY** funding. Same normalization as Hyperliquid required.

**Historical Gap:** dYdX V3 (Ethereum L2) data available 2021-2023 via `https://api.dydx.exchange/v3`

---

## 3. Decentralized Exchanges (DEX)

### 3.1 Uniswap V3

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/dex/uniswap_collector.py` |
| **Venue Type** | DEX (AMM) |
| **Data Source** | The Graph Subgraphs |
| **Authentication** | The Graph API Key (optional, increases limits) |
| **Rate Limits** | 1,000/day (free), higher with API key |

**Subgraph Endpoints by Chain:**

| Chain | Hosted Service URL | Decentralized Subgraph ID |
|-------|-------------------|---------------------------|
| Ethereum | `https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3` | `5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV` |
| Arbitrum | `https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-arbitrum-one` | `FbCGRftH4a3yZugY7TnbYgPJVEv2LvMT6oF1fxPe9aJM` |
| Optimism | `https://api.thegraph.com/subgraphs/name/ianlapham/optimism-post-regenesis` | `Cghf4LfVqPiFw6fp6Y5X5Ubc8UpmUhSfJL82zwiBFLaj` |
| Polygon | `https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-v3-polygon` | `3hCPRGf4z88VC5rsBKU5AA9FBBq5nF3jbKJG7VZCbhjm` |
| Base | `https://api.thegraph.com/subgraphs/name/lynnshaoyu/uniswap-v3-base` | N/A |

**Data Types Collected:**

| Data Type | GraphQL Entity | Fields |
|-----------|----------------|--------|
| Pools | `pools` | id, token0, token1, feeTier, liquidity, totalValueLockedUSD |
| Swaps | `swaps` | timestamp, amount0, amount1, amountUSD, pool |
| Pool Day Data | `poolDayDatas` | date, volumeUSD, tvlUSD, feesUSD |

**Liquidity Filtering Applied:** TVL ≥ $500,000, Daily Volume ≥ $50,000

---

### 3.2 Curve Finance

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/dex/curve_collector.py` |
| **Venue Type** | DEX (StableSwap AMM) |
| **Data Sources** | The Graph Subgraphs, Curve API |

**Subgraph Endpoints:**

| Chain | URL |
|-------|-----|
| Ethereum | `https://api.thegraph.com/subgraphs/name/curvefi/curve` |
| Arbitrum | `https://api.thegraph.com/subgraphs/name/curvefi/curve-arbitrum` |
| Optimism | `https://api.thegraph.com/subgraphs/name/curvefi/curve-optimism` |
| Polygon | `https://api.thegraph.com/subgraphs/name/curvefi/curve-polygon` |
| Avalanche | `https://api.thegraph.com/subgraphs/name/curvefi/curve-avalanche` |

**Alternative API Endpoints:**

| Chain | URL |
|-------|-----|
| Ethereum | `https://api.curve.fi/api/getPools/ethereum/main` |
| Arbitrum | `https://api.curve.fi/api/getPools/arbitrum/main` |
| Optimism | `https://api.curve.fi/api/getPools/optimism/main` |
| Polygon | `https://api.curve.fi/api/getPools/polygon/main` |

**Data Types:** Pool TVL, swap volumes, virtual prices, A parameters

---

### 3.3 GMX

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/dex/gmx_collector.py` |
| **Venue Type** | DEX (Perpetual, GLP model) |
| **Data Source** | The Graph Subgraphs |

**Subgraph Endpoints:**

| Version | Chain | URL |
|---------|-------|-----|
| V1 | Arbitrum | `https://api.thegraph.com/subgraphs/name/gmx-io/gmx-stats` |
| V2 | Arbitrum | `https://api.thegraph.com/subgraphs/name/gmx-io/synthetics-arbitrum-stats` |
| V1 | Avalanche | `https://api.thegraph.com/subgraphs/name/gmx-io/gmx-avalanche-stats` |

**Data Types Collected:**

| Data Type | GraphQL Entity | Fields |
|-----------|----------------|--------|
| Funding Rates | `fundingRates` | fundingRateLong, fundingRateShort, borrowRateLong, borrowRateShort |
| Positions | `positions` | size, collateral, entryFundingRate, realisedPnl |
| GLP Stats | `glpStats` | aumInUsdg, glpSupply |

**Note:** GMX uses a unique GLP pool model. Funding paid from/to GLP pool.

---

### 3.4 SushiSwap

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/dex/sushiswap_v2_collector.py` |
| **Venue Type** | DEX (Constant Product AMM) |
| **Data Source** | The Graph Subgraphs |

**Subgraph Endpoints:**

| Chain | URL |
|-------|-----|
| Ethereum | `https://api.thegraph.com/subgraphs/name/sushiswap/exchange` |
| Arbitrum | `https://api.thegraph.com/subgraphs/name/sushiswap/arbitrum-exchange` |
| Polygon | `https://api.thegraph.com/subgraphs/name/sushiswap/matic-exchange` |
| Avalanche | `https://api.thegraph.com/subgraphs/name/sushiswap/avalanche-exchange` |
| Fantom | `https://api.thegraph.com/subgraphs/name/sushiswap/fantom-exchange` |
| BSC | `https://api.thegraph.com/subgraphs/name/sushiswap/bsc-exchange` |

---

### 3.5 Jupiter (Solana)

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/dex/jupiter_collector.py` |
| **Venue Type** | DEX Aggregator (Solana) |
| **API Endpoints** | Jupiter Public API |
| **Authentication** | None required (FREE) |
| **Rate Limits** | ~60 requests/minute |

**Data Types:** Token prices (aggregated best price), route information, quote data

**Note:** Jupiter aggregates liquidity from Raydium, Orca, Serum, Lifinity, Marinade, etc.

---

### 3.6 CowSwap

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/dex/cowswap_collector.py` |
| **Venue Type** | DEX (Batch Auction, MEV-protected) |
| **Data Sources** | CoW Protocol API, The Graph |

**API Endpoints:**

| Chain | API URL | Subgraph URL |
|-------|---------|--------------|
| Ethereum | `https://api.cow.fi/mainnet` | `https://api.thegraph.com/subgraphs/name/cowprotocol/cow` |
| Gnosis | `https://api.cow.fi/xdai` | `https://api.thegraph.com/subgraphs/name/cowprotocol/cow-gc` |
| Arbitrum | `https://api.cow.fi/arbitrum_one` | `https://api.thegraph.com/subgraphs/name/cowprotocol/cow-arbitrum-one` |

**Data Types:** Trade/settlement data, batch auction results, solver competition

---

### 3.7 Vertex Protocol

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/dex/vertex_collector.py` |
| **Venue Type** | Hybrid DEX (Order book + AMM) |
| **API Endpoints** | |
| - Archive API | `https://archive.prod.vertexprotocol.com/v1` |
| - Gateway API | `https://gateway.prod.vertexprotocol.com/v1` |
| **Authentication** | None required (FREE) |
| **Chain** | Arbitrum |

**Data Types:** Funding rates (hourly), OHLCV, order book, open interest

---

### 3.8 DEX Aggregators

#### GeckoTerminal

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/dex/geckoterminal_collector.py` |
| **API Endpoint** | `https://api.geckoterminal.com/api/v2` |
| **Authentication** | None required (FREE) |
| **Rate Limits** | 30 requests/minute |
| **Coverage** | 100+ chains |

#### DEXScreener

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/dex/dexscreener_collector.py` |
| **API Endpoint** | `https://api.dexscreener.com` |
| **Authentication** | None required (FREE) |
| **Rate Limits** | ~300 requests/minute |
| **Coverage** | 80+ chains |

#### 1inch

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/dex/oneinch_collector.py` |
| **API Endpoint** | `https://api.1inch.dev` |
| **Authentication** | None required (FREE for basic) |
| **Rate Limits** | ~1 request/second |

**Supported Chains:** Ethereum, Polygon, BSC, Arbitrum, Optimism, Avalanche, Gnosis, Fantom

#### 0x Protocol

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/dex/zerox_collector.py` |
| **Authentication** | None required (FREE) |
| **Rate Limits** | ~100 requests/minute |

**Chain-Specific Endpoints:**

| Chain | Endpoint |
|-------|----------|
| Ethereum | `https://api.0x.org` |
| Polygon | `https://polygon.api.0x.org` |
| BSC | `https://bsc.api.0x.org` |
| Arbitrum | `https://arbitrum.api.0x.org` |
| Optimism | `https://optimism.api.0x.org` |
| Avalanche | `https://avalanche.api.0x.org` |
| Base | `https://base.api.0x.org` |
| Fantom | `https://fantom.api.0x.org` |

---

## 4. Options Venues

### 4.1 Deribit

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/options/deribit_collector.py` |
| **Venue Type** | CEX (Options) |
| **API Endpoints** | |
| - Production | `https://www.deribit.com/api/v2` |
| - Testnet | `https://test.deribit.com/api/v2` |
| **Authentication** | API Key + Secret (optional for public) |
| **Rate Limits** | 20 requests/second (matching engine) |
| **Market Share** | ~90% of crypto options volume |

**Data Types Collected:**

| Data Type | Endpoint | Fields |
|-----------|----------|--------|
| Options Chain | `/public/get_instruments` | instrument_name, strike, expiration, option_type |
| Mark Prices | `/public/get_book_summary_by_currency` | mark_price, mark_iv, underlying_price |
| Greeks | `/public/ticker` | delta, gamma, vega, theta, rho |
| DVOL Index | `/public/get_volatility_index_data` | volatility, timestamp |
| Historical Trades | `/public/get_last_trades_by_instrument` | price, amount, direction, timestamp |

**API Documentation:** https://docs.deribit.com/

---

### 4.2 AEVO

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/options/aevo_collector.py` |
| **Venue Type** | DEX (Options, Rollup-based) |
| **API Endpoint** | `https://api.aevo.xyz` |
| **Authentication** | API Key + Secret |
| **Rate Limits** | ~300 requests/minute (with API key) |

**Data Types:** Options markets, IV surface, order book, settlements

---

### 4.3 Lyra Finance

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/options/lyra_collector.py` |
| **Venue Type** | DEX (AMM Options) |
| **Data Sources** | The Graph, Lyra API |
| **API Endpoint** | `https://api.lyra.finance` |

**Subgraph Endpoints:**

| Chain | URL |
|-------|-----|
| Optimism | `https://api.thegraph.com/subgraphs/name/lyra-finance/mainnet` |
| Arbitrum | `https://api.thegraph.com/subgraphs/name/lyra-finance/arbitrum` |

**Data Types:** Options markets, IV surfaces, Greeks, MMV (Market Maker Vault) stats

---

### 4.4 Dopex

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/options/dopex_collector.py` |
| **Venue Type** | DEX (SSOV Options) |
| **Data Sources** | The Graph, Dopex API |
| **API Endpoint** | `https://api.dopex.io/v2` |

**Subgraph Endpoints:**

| Subgraph | URL |
|----------|-----|
| Main | `https://api.thegraph.com/subgraphs/name/dopex-io/dopex` |
| SSOV | `https://api.thegraph.com/subgraphs/name/dopex-io/ssov` |

**Data Types:** SSOV vault data, option purchases, Atlantic straddles, epoch data

---

## 5. On-Chain Analytics

### 5.1 Glassnode

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/onchain/glassnode_collector.py` |
| **API Endpoint** | `https://api.glassnode.com/v1/metrics` |
| **Authentication** | API Key required |
| **Rate Limits** | 10 req/min (Free), 20 req/min (Starter) |
| **Coverage** | BTC (2009+), ETH (2015+), major altcoins |

**Metrics Categories:**
- Market: price, market_cap, realized_cap
- On-chain: active_addresses, transaction_count, transfer_volume
- Mining: hash_rate, difficulty, miner_revenue
- Supply: circulating, illiquid, exchange_balance
- Indicators: SOPR, MVRV, NVT, NUPL

**API Documentation:** https://docs.glassnode.com/

---

### 5.2 CryptoQuant

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/onchain/cryptoquant_collector.py` |
| **API Endpoint** | `https://api.cryptoquant.com/v1` |
| **Authentication** | API Key required |
| **Rate Limits** | 10 requests/minute |

**Metrics:** Exchange flows (inflow/outflow), miner metrics, SOPR, NUPL, whale movements

---

### 5.3 Coin Metrics

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/onchain/coinmetrics_collector.py` |
| **API Endpoint** | `https://api.coinmetrics.io/v4` |
| **Authentication** | API Key (optional for community endpoints) |

**API Documentation:** https://docs.coinmetrics.io/api/v4

**Metrics:** Network metrics (hash rate, difficulty), market metrics (OHLCV), on-chain indicators

---

### 5.4 Additional On-Chain Sources

| Source | Collector | API Endpoint | Authentication |
|--------|-----------|--------------|----------------|
| **Nansen** | `onchain/nansen_collector.py` | `https://api.nansen.ai/v1` | API Key |
| **Arkham** | `onchain/arkham_collector.py` | `https://api.arkhamintelligence.com/v1` | API Key |
| **Santiment** | `onchain/santiment_collector.py` | `https://api.santiment.net/graphql` | API Key |
| **Bitquery** | `onchain/bitquery_collector.py` | `https://graphql.bitquery.io` | Bearer Token |
| **Covalent** | `onchain/covalent_collector.py` | `https://api.covalenthq.com/v1` | API Key |
| **Flipside** | `onchain/flipside_collector.py` | `https://api-v2.flipsidecrypto.xyz` | API Key |
| **Whale Alert** | `onchain/whale_alert_collector.py` | `https://api.whale-alert.io/v1` | API Key |

---

## 6. Market Data Aggregators

### 6.1 CoinGecko

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/market_data/coingecko_collector.py` |
| **API Endpoints** | |
| - Free | `https://api.coingecko.com/api/v3` |
| - Pro | `https://pro-api.coingecko.com/api/v3` |
| **Rate Limits** | 10-50 calls/min (free), 500 calls/min (pro) |

**API Documentation:** https://www.coingecko.com/en/api/documentation

**Data Types:** Price data (10,000+ coins), historical OHLCV, market cap, exchange data

---

### 6.2 CryptoCompare

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/market_data/cryptocompare_collector.py` |
| **API Endpoint** | `https://min-api.cryptocompare.com/data` |
| **Authentication** | API Key (increases limits) |
| **Rate Limits** | 100,000 calls/month (free), higher with pro |

**Data Types:** Historical OHLCV, multi-exchange prices, social/market metrics

---

### 6.3 Messari

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/market_data/messari_collector.py` |
| **API Endpoint** | `https://data.messari.io/api` |
| **Rate Limits** | 1,000 calls/day (free), higher with pro |

**API Documentation:** https://messari.io/api

**Data Types:** Asset profiles, metrics, market data, news

---

### 6.4 Kaiko

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/market_data/kaiko_collector.py` |
| **API Endpoint** | `https://us.market-api.kaiko.io/v2` |
| **Authentication** | API Key required |

**API Documentation:** https://docs.kaiko.com/

**Data Types:** Order book snapshots, trade data, market microstructure analytics

**Note:** Free tier provides samples only; paid tier for full historical access.

---

## 7. Alternative Data

### 7.1 DefiLlama

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/alternative/defillama_collector.py` |
| **API Endpoint** | `https://api.llama.fi` |
| **Authentication** | None required (FREE) |
| **Rate Limits** | Very generous (~100/min) |

**API Documentation:** https://defillama.com/docs/api

**Data Types:** Protocol TVL, chain TVL, yields/APY, stablecoin analytics, bridge volumes, fees/revenue

---

### 7.2 Coinalyze

| Attribute | Details |
|-----------|---------|
| **Collectors** | `alternative/coinalyze_collector.py`, `alternative/coinalyze_enhanced_collector.py` |
| **API Endpoint** | `https://api.coinalyze.net/v1` |
| **Authentication** | API Key required |
| **Rate Limits** | 40 calls/minute |

**API Documentation:** https://api.coinalyze.net/v1/doc/

**Data Types:** Funding rates (current, predicted, historical), open interest, liquidations, long/short ratios

**Note:** FREE alternative to Coinglass

---

### 7.3 Dune Analytics

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/alternative/dune_analytics_collector.py` |
| **API Endpoint** | `https://api.dune.com/api/v1` |
| **Authentication** | API Key required |
| **Rate Limits** | Credit-based (2,500 credits/month free) |

**API Documentation:** https://docs.dune.com

**Data Types:** Custom SQL queries on blockchain data, pre-built dashboards

---

### 7.4 LunarCrush

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/alternative/lunarcrush_collector.py` |
| **API Endpoint** | `https://lunarcrush.com/api4/public` |
| **Authentication** | API Key required |
| **Rate Limits** | 100 requests/day (free) |

**API Documentation:** https://lunarcrush.com/developers/api/endpoints

**Data Types:** Social volume, Galaxy Score, AltRank, influencer tracking, sentiment analysis

---

## 8. Social & Sentiment

### 8.1 Twitter/X

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/social/twitter_collector.py` |
| **API Endpoint** | `https://api.twitter.com/2` |
| **Authentication** | Bearer Token (Twitter API v2) |

**Data Types:** Tweet volume, influencer activity, sentiment analysis, trending topics

---

### 8.2 The Graph (Infrastructure)

| Attribute | Details |
|-----------|---------|
| **Collector** | `data_collection/indexers/thegraph_collector.py` |
| **Gateway URL** | `https://gateway.thegraph.com/api` |
| **Authentication** | API Key (for decentralized network) |

**Supported Protocols:** Uniswap, SushiSwap, Curve, Balancer, Aave, Compound, GMX, Synthetix

---

## 9. Data Flow Summary

### 9.1 Primary Data Flows by Strategy

| Strategy | Primary Sources | Validation Sources | Frequency |
|----------|-----------------|-------------------|-----------|
| **Funding Rate Arbitrage** | Binance, Hyperliquid | Bybit, dYdX V4 | 8h (CEX), 1h (hybrid) |
| **Altcoin Pairs Trading** | Binance (CEX), Uniswap (DEX) | CoinGecko, GeckoTerminal | Hourly |
| **BTC Futures Curve** | Binance, Deribit | CME, Hyperliquid | Hourly |
| **Options Volatility** | Deribit | AEVO, Lyra | Daily snapshots |

### 9.2 Cross-Validation Matrix

| Dataset | Primary | Validation 1 | Validation 2 | Expected Correlation |
|---------|---------|--------------|--------------|---------------------|
| BTC Funding | Binance | Bybit | Hyperliquid | >0.95 (CEX), >0.90 (hybrid) |
| ETH Funding | Binance | Bybit | dYdX V4 | >0.95 (CEX), >0.90 (hybrid) |
| BTC Spot Price | Binance | Coinbase | CoinGecko | >0.999 |
| DEX TVL | DefiLlama | The Graph | GeckoTerminal | >0.95 |

---

## 10. Licensing & Terms of Use

### 10.1 Free APIs (No Commercial Restrictions)

| Source | Terms | Link |
|--------|-------|------|
| Hyperliquid | Public API, no restrictions | https://hyperliquid.gitbook.io |
| dYdX V4 | Public indexer, no restrictions | https://docs.dydx.exchange |
| DefiLlama | CC0 (public domain) | https://defillama.com |
| GeckoTerminal | Free for non-commercial | https://www.geckoterminal.com |
| DEXScreener | Free tier available | https://docs.dexscreener.com |
| The Graph (Hosted) | Free tier, rate limited | https://thegraph.com |

### 10.2 APIs Requiring Attribution

| Source | Attribution Required | Terms Link |
|--------|---------------------|------------|
| CoinGecko | "Data provided by CoinGecko" | https://www.coingecko.com/en/api/documentation |
| Binance | Reference API documentation | https://www.binance.com/en/terms |
| Deribit | Reference API documentation | https://www.deribit.com/pages/information/terms-of-service |

### 10.3 Paid APIs Used

| Source | Tier Used | Cost | Features |
|--------|-----------|------|----------|
| Glassnode | Starter | ~$29/month | Extended metrics, higher limits |
| Kaiko | Sample/Trial | Free trial | Limited historical data |
| Nansen | Research | Varies | Smart money labels |

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Created** | January 2025 |
| **Last Updated** | January 31, 2026 |
| **Total Sources** | 44 unique collectors (32 enabled) |
| **Total API Endpoints** | 60+ |
| **Chains Covered** | Ethereum, Arbitrum, Optimism, Polygon, Base, Avalanche, BSC, Solana, Fantom, Gnosis |
| **Data Period** | 2022-01-01 to 2026-01-31 |

---

