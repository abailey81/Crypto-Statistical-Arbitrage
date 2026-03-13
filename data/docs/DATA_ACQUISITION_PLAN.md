# Data Acquisition Plan

## Crypto Statistical Arbitrage System - Phase 1

**Document Version:** 3.0.0
**Last Updated:** January 31, 2026
**Project Phase:** Data Acquisition & Validation (20% of Total Evaluation)
**Status:** ACTIVE - Comprehensive Implementation with Extended Analytics

---

## 1. Executive Summary

This document outlines the comprehensive data acquisition strategy for the Crypto Statistical Arbitrage system. The system requires high-quality, multi-venue market data to identify and exploit pricing inefficiencies across centralized (CEX), decentralized (DEX), and hybrid venues.

**Key Objectives:**
- Collect funding rates from 12+ derivatives venues
- Aggregate OHLCV data from 17+ price sources
- Achieve <5% missing data for core assets
- Cross-validate CEX vs DEX data integrity
- Cover 2+ years of historical data (2022-2024)

---

## 2. Data Requirements Matrix

### 2.1 Strategy 1: Perpetual Funding Rate Arbitrage

| Data Type | Assets | Frequency | Critical Fields | Priority |
|-----------|--------|-----------|-----------------|----------|
| Funding Rates | BTC, ETH, 25+ altcoins | 1h-8h | rate, timestamp, venue | **Must-have** |
| Open Interest | Same | 1h | oi_long, oi_short | Nice-to-have |
| Mark Price | Same | 1h | mark_price, index_price | Nice-to-have |

**Rationale:** Funding rate arbitrage exploits persistent rate differentials between venues. Requires normalized 8h rates for cross-venue comparison.

### 2.2 Strategy 2: Altcoin Statistical Arbitrage

| Data Type | Assets | Frequency | Critical Fields | Priority |
|-----------|--------|-----------|-----------------|----------|
| OHLCV | 50+ altcoins | 1h | open, high, low, close, volume | **Must-have** |
| Correlation | Asset pairs | Daily | rolling_corr_30d | Calculated |
| Volume Profile | Same | 1h | volume_usd, venue | Nice-to-have |

**Rationale:** Pairs trading requires synchronized price data across venues to detect mean-reversion opportunities.

### 2.3 Strategy 3: BTC Futures Curve Trading

| Data Type | Assets | Frequency | Critical Fields | Priority |
|-----------|--------|-----------|-----------------|----------|
| Futures OHLCV | BTC, ETH | 1h | close, expiry, venue | **Must-have** |
| Spot Reference | BTC, ETH | 1h | close | **Must-have** |
| Funding (Perps) | BTC, ETH | 8h | funding_rate | Nice-to-have |

**Rationale:** Exploits basis (spot-futures spread) and contango/backwardation patterns.

### 2.4 Strategy 4: Options Vol Surface Arbitrage

| Data Type | Assets | Frequency | Critical Fields | Priority |
|-----------|--------|-----------|-----------------|----------|
| Options Data | BTC, ETH | 1h | strike, expiry, iv, delta | **Must-have** |
| Underlying | BTC, ETH | 1m | close | **Must-have** |
| Funding Rates | BTC, ETH | 8h | funding_rate | Nice-to-have |

**Rationale:** Combines options volatility surface analysis with funding rate carry.

---

## 3. Source Evaluation

### 3.1 CEX Sources (Primary - Highest Liquidity)

| Source | Coverage | Cost | API Limits | Quality | Decision |
|--------|----------|------|------------|---------|----------|
| Binance | Excellent | Free | 1200/min | A+ | **Primary** |
| Bybit | Excellent | Free | 600/min | A | **Secondary** |
| OKX | Good | Free | 1000/min | A | **Tertiary** |
| Kraken | Limited | Paid | 900/min | A | Backup |
| Deribit | BTC/ETH Options | Paid | 1200/min | A+ | Options Primary |

**CEX Characteristics:**
- Deep liquidity, tight spreads
- Fast order matching (~10ms)
- Primary price discovery venue
- Counterparty risk present

### 3.2 Hybrid DEX Sources (Secondary - On-chain Settlement)

| Source | Coverage | Cost | Latency | Quality | Decision |
|--------|----------|------|---------|---------|----------|
| Hyperliquid | Good | Free | 1-2 blocks | A | **Primary DEX** |
| dYdX (v4) | Good | Free | 1-2 blocks | A | **Secondary DEX** |
| Vertex | Moderate | Free | 1-2 blocks | B+ | Tertiary |
| GMX | Good | Free | 1-2 blocks | B+ | Perp DEX |

**Hybrid Characteristics:**
- On-chain settlement, off-chain matching
- No counterparty risk
- Higher latency (block time)
- 1-hour funding intervals (vs 8h for CEX)

### 3.3 DEX Aggregators (Cross-validation)

| Source | Coverage | Cost | Rate Limit | Use Case |
|--------|----------|------|------------|----------|
| GeckoTerminal | Multi-chain | Free | 60/min | DEX price aggregation |
| DexScreener | Multi-chain | Free | 60/min | DEX price discovery |
| CoinGecko | Global | Free tier | 30/min | Reference pricing |

### 3.4 CEX vs DEX Tradeoffs

| Dimension | CEX | DEX/Hybrid |
|-----------|-----|-----------|
| Liquidity | Deep (~$500M+/day) | Moderate (~$50M+/day) |
| Latency | ~10ms | ~250ms-12s (block time) |
| Counterparty Risk | Exchange custodial | None (smart contract) |
| Funding Interval | 8-hour | 1-hour (continuous) |
| Price Discovery | Primary | Secondary |
| MEV Risk | None | Present |
| Regulatory Risk | Higher | Lower |

**Cross-validation enables:**
- Detection of venue-specific anomalies
- Validation of data accuracy
- Arbitrage opportunity identification

---

## 4. Collection Strategy

### 4.1 Technical Architecture

```
                    ┌─────────────────────────────────────┐
                    │     Phase 1 Pipeline Orchestrator    │
                    │         (run_phase1.py)              │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
      ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
      │ CEX Collectors │       │Hybrid Collectors│      │ DEX Collectors │
      │  (6 venues)    │       │  (3 venues)     │      │  (5 venues)    │
      └───────┬───────┘       └───────┬───────┘       └───────┬───────┘
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │        Data Pipeline              │
                    │ (Cleaning → Validation → Storage) │
                    └─────────────────┬─────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
      ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
      │ Data Cleaner  │       │Cross-Venue    │       │Quality Checker│
      │(7 stages)     │       │Reconciler     │       │(9 dimensions) │
      └───────────────┘       └───────────────┘       └───────────────┘
```

### 4.2 Data Collection Approach

| Method | Venues | Use Case |
|--------|--------|----------|
| REST API (Historical) | All venues | Backfill 2022-2024 data |
| REST API (Polling) | Primary venues | Gap filling, validation |
| WebSocket (Future) | Binance, Bybit | Live trading (Phase 3) |
| Subgraph Queries | DEX protocols | On-chain DEX data |

### 4.3 Rate Limit Management

```
Venue Rate Limits:
├── Binance: 1200/min → 20 req/sec sustainable
├── Bybit: 600/min → 10 req/sec sustainable
├── OKX: 1000/min → 16 req/sec sustainable
├── Hyperliquid: 100/min → 1.5 req/sec sustainable
└── dYdX: 100/min → 1.5 req/sec sustainable

Strategy:
- Adaptive rate limiting with exponential backoff
- Per-venue semaphores for concurrent requests
- Parallel symbol processing within rate limits
- Automatic retry with jitter
```

### 4.4 Collection Timeline

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1 | CEX Historical Data | Binance, Bybit, OKX (2022-2024) |
| 2 | Hybrid/DEX Data | Hyperliquid, dYdX, GMX |
| 3 | Validation & Gap Fill | Cross-venue reconciliation |
| 4 | Documentation | Quality report, data dictionary |

---

## 5. Risk Mitigation

### 5.1 Data Quality Risks

| Risk | Mitigation |
|------|------------|
| API Downtime | Fallback venues, local caching |
| Rate Limiting | Adaptive throttling, queue management |
| Data Gaps | Cross-venue fill, interpolation (max 3 periods) |
| Schema Changes | Version-aware parsing, graceful degradation |
| Stale Data | Freshness monitoring, alerting |

### 5.2 Technical Risks

| Risk | Mitigation |
|------|------------|
| Network Failures | Exponential backoff, circuit breakers |
| Memory Issues | Streaming processing, batch chunking |
| Storage Limits | Parquet compression, data retention policy |
| Credential Exposure | Environment variables, no hardcoding |

### 5.3 Data Integrity Controls

1. **Schema Validation:** Enforce required columns and types
2. **Range Validation:** Flag values outside expected bounds
3. **Duplicate Detection:** Hash-based deduplication
4. **Outlier Detection:** Ensemble methods (LOF, IsolationForest, MAD)
5. **Cross-venue Correlation:** Minimum 0.85 for CEX-DEX pairs
6. **Timestamp Alignment:** Settlement time snapping per venue

---

## 6. Extended Analytics (NEW in v3.0)

### 6.1 Survivorship Bias Assessment

**Academic Foundation:**
Based on Liu et al. (2019) and Elendner et al. (2018):

| Portfolio Type | Annual Bias | Adjustment Factor |
|----------------|-------------|-------------------|
| Value-Weighted | 0.93% | ~0.99 |
| Equal-Weighted | 62.19% | ~0.62 |
| Liquidity-Weighted | ~2.5% | ~0.97 |

**Tracked Major Delistings:**
- LUNA (May 2022) - Terra collapse, -99.99%
- UST (May 2022) - Algorithmic stablecoin failure
- FTT (November 2022) - FTX exchange collapse
- CEL (July 2022) - Celsius bankruptcy

**Application:** All backtested returns are adjusted using survivorship bias factors.

### 6.2 Wash Trading Detection

**Risk Assessment by Venue Type:**
- HIGH: CEX (Binance, Bybit, OKX) - Unverifiable volume
- MEDIUM: Aggregators (GeckoTerminal, DexScreener)
- LOW: On-chain DEX - Verifiable transactions

**Detection Methods:**
1. Volume-price divergence analysis
2. Round number concentration
3. Cross-venue volume correlation

### 6.3 DEX-Specific Considerations

**MEV (Maximal Extractable Value):**
| Venue | MEV Risk | Adjustment |
|-------|----------|------------|
| Uniswap, Curve | HIGH | Apply 0.1-5% MEV cost |
| CoWSwap | LOW | MEV-protected routing |
| GMX | LOW | Oracle-based pricing |

**Sandwich Attack Risk:**
- HIGH: Traditional AMMs (Uniswap, SushiSwap)
- PROTECTED: CoWSwap (batch auctions)
- LOW: GMX (oracle pricing)

**Liquidity Fragmentation:**
- 6 Ethereum venues
- 2 Arbitrum venues
- 2 Solana venues
- 2 Custom L1 venues

### 6.4 CEX vs DEX Cross-Validation

**Expected Divergence:**
| Condition | CEX-DEX Divergence |
|-----------|-------------------|
| Normal | < 0.1% (10 bps) |
| Stressed | 0.1-1% |
| Extreme | > 1% |

**Arbitrage Signal:** Flag opportunities when divergence > 10 bps for 30+ seconds.

---

## 7. Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Venue Coverage | 12+ funding, 17+ OHLCV | Active collectors |
| Asset Coverage | BTC, ETH, 25+ alts | Symbol list |
| Date Range | 2022-01-01 to 2024-12-31 | Min/max timestamps |
| Missing Data | <5% for core assets | Gap analysis |
| Quality Score | >80/100 average | 9-dimension framework |
| Cross-venue Correlation | >0.85 for funding | CEX-DEX reconciliation |
| Data Freshness | <5 minutes lag | Timestamp analysis |

---

## 8. Appendices

### A. Target Symbol Universe

**Core Assets (Tier 1):**
BTC, ETH

**Major Altcoins (Tier 2):**
SOL, ARB, OP, AVAX, MATIC, LINK, DOGE, XRP

**Extended Universe (Tier 3):**
ADA, DOT, ATOM, UNI, AAVE, MKR, SNX, CRV, LDO, FXS, GMX, DYDX, APE, BLUR, WLD, SEI, SUI, APT, INJ

### B. API Documentation References

- Binance: https://binance-docs.github.io/apidocs/
- Bybit: https://bybit-exchange.github.io/docs/
- OKX: https://www.okx.com/docs-v5/
- Hyperliquid: https://hyperliquid.gitbook.io/
- dYdX v4: https://docs.dydx.exchange/

### C. Funding Rate Normalization

All funding rates normalized to 8-hour equivalent:
- CEX (8h native): No adjustment
- Hybrid (1h native): Multiply by 8
- GMX (continuous): Convert to 8h equivalent

---

*Document maintained by Tamer Atesyakar*
