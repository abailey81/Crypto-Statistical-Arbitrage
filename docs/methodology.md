# Methodology
## Crypto Statistical Arbitrage Multi-Venue System

**Version:** 2.0
**Last Updated:** January 2025
**Author:** Tamer Atesyakar

---

## 1. Overview

This document describes the quantitative methodology for the multi-venue crypto statistical arbitrage system. The approach spans four strategies across CEX, hybrid, and DEX venues with a unified portfolio construction framework.

### Strategy Summary

| # | Strategy | Venue Focus | Key Method |
|---|----------|------------|------------|
| 1 | Perpetual Funding Rate Arbitrage | CEX + Hybrid | Cross-venue funding differential capture |
| 2 | Altcoin Statistical Arbitrage | CEX + DEX | Cointegration-based pairs trading |
| 3 | BTC Futures Curve Trading | CEX + Hybrid | Term structure arbitrage |
| 4 | Cross-DEX Arbitrage | DEX | Price differential across DEX protocols |

---

## 2. Data Processing Methodology

### 2.1 Collection

All data is collected from free, publicly available APIs using the CCXT library (CEX) and The Graph protocol (DEX). Collection follows these principles:

- **Rate limiting**: Exponential backoff with jitter to respect API limits
- **Idempotent collection**: Re-running the pipeline produces identical results
- **Incremental updates**: Only fetch new data since last collection timestamp
- **Multi-source validation**: Each dataset validated against 3+ independent sources

### 2.2 Cleaning & Preprocessing

1. **Timestamp normalization**: All timestamps converted to UTC, rounded to nearest hour
2. **Symbol normalization**: Unified format (e.g., `BTC` not `BTCUSDT`, `BTC-USD`)
3. **Missing data handling**: Forward-fill for gaps < 4 hours, mark and exclude longer gaps
4. **Outlier detection**: Winsorize at 5-sigma from rolling 168h mean
5. **Survivorship bias**: Track 47 delisting events, include dead tokens in analysis period

### 2.3 Cross-Validation

Each core dataset is validated across multiple sources:

| Validation Check | Threshold | Action if Failed |
|-----------------|-----------|-----------------|
| Price correlation | > 0.95 | Flag token, investigate |
| MAPE | < 5% | Exclude from analysis |
| Volume consistency | Same order of magnitude | Use higher-quality source |
| Funding rate sign | Must agree | Use majority vote |

---

## 3. Strategy 1: Perpetual Funding Rate Arbitrage

### 3.1 Signal Construction

The funding rate differential between venues creates arbitrage opportunity:

```
Signal = FundingRate(Venue_A) - FundingRate(Venue_B) - TransactionCosts
```

Entry when the annualized differential exceeds transaction costs by a configurable threshold (default: 2x costs).

### 3.2 Execution Logic

- **Long funding**: Open long position on venue with lower funding (receive funding)
- **Short funding**: Open short position on venue with higher funding (pay less)
- **Delta neutral**: Maintain equal and opposite positions across venues

### 3.3 Risk Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max leverage | 2.0x | Futures already leveraged |
| Stop loss | 5% basis move | Limit counterparty exposure |
| Min holding | 8 hours | At least 1 funding payment |
| Venue exposure | 50% Binance, 30% CME, 15% Hyperliquid, 5% dYdX | Diversification |

---

## 4. Strategy 2: Altcoin Statistical Arbitrage

### 4.1 Universe Construction

**CEX Universe (30-50 tokens):**
- Filter: Average daily volume > $10M, market cap > $300M
- Exclude: Stablecoins, wrapped tokens, leveraged tokens
- Handle: Delisted tokens tracked for survivorship bias

**DEX Universe (20-30 tokens):**
- Filter: Pool TVL > $500k, daily volume > $50k, > 100 trades/day
- Check: Liquidity across multiple DEXs (Uniswap, Curve, Balancer)
- Exclude: Obvious scams, locked liquidity tokens

### 4.2 Cointegration Analysis

Three complementary tests applied to all candidate pairs:

1. **Engle-Granger**: Two-step residual-based test (ADF on spread)
2. **Johansen**: Multivariate trace and eigenvalue tests
3. **Phillips-Ouliaris**: Residual-based with Phillips-Perron correction

A pair passes if consensus score >= 0.35 (weighted average of p-values across tests).

### 4.3 Half-Life Estimation

Half-life calculated via AR(1) on the spread residuals:

```
spread_t = phi * spread_{t-1} + epsilon_t
half_life = -log(2) / log(phi)
```

Computed on hourly data (half-life in hours internally, reported in days).

| Classification | Half-Life | Score | Action |
|---------------|-----------|-------|--------|
| Preferred | 1-7 days | 1.0 | Full position |
| Acceptable | 7-14 days | 0.7 | Reduced position |
| Marginal | 14-30 days | 0.3 | Small position, close monitoring |
| Retire | > 30 days | 0.0 | Remove from portfolio |

### 4.4 Signal Generation (Z-Score)

```
z_score = (spread - mean(spread, lookback)) / std(spread, lookback)
```

| Parameter | CEX Pairs | DEX Pairs | Rationale |
|-----------|-----------|-----------|-----------|
| Entry (long spread) | z < -2.0 | z < -2.5 | Higher threshold for DEX (gas costs) |
| Entry (short spread) | z > +2.0 | z > +2.5 | Same logic |
| Exit | z crosses 0 | abs(z) < 1.0 | Tighter exit for DEX (capture profit before gas) |
| Stop loss | abs(z) > 3.0 | abs(z) > 3.5 | Wider stop for DEX (higher noise) |

### 4.5 Position Sizing

Venue-adjusted Kelly criterion with conservative fractional sizing:

```
kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
position_size = kelly_fraction * 0.25 to 0.50 * capital
```

| Venue Tier | Max Position | Kelly Fraction |
|-----------|-------------|---------------|
| Tier 1 (CEX) | $100,000 | 0.50x Kelly |
| Tier 2 (Mixed) | $50,000 | 0.35x Kelly |
| Tier 3 (DEX) | $10,000 | 0.25x Kelly |

### 4.6 ML Enhancement

Gradient Boosting and Random Forest models trained on spread features to improve entry/exit timing:

**Features:**
- Lagged z-scores (1, 2, 4, 8, 24 bars)
- Spread momentum and acceleration
- Volume ratios (token A vs B)
- BTC returns and volatility
- Sector index returns
- Correlation stability metrics

**Training:** Walk-forward validation with 18-month train / 6-month test windows. Separate models for CEX and DEX pairs.

### 4.7 Pair Ranking & Selection

Pairs ranked by composite score:

| Factor | Weight | Metric |
|--------|--------|--------|
| Cointegration strength | 25% | Consensus p-value |
| Half-life | 20% | Preference for 1-7 days |
| Liquidity | 20% | Combined volume/TVL |
| Venue accessibility | 15% | Both CEX > mixed > both DEX |
| Sector diversification | 10% | Penalty for concentrated sectors |
| Spread volatility | 10% | Sufficient movement for profitability |

Final selection: 10-15 Tier 1 pairs, 3-5 Tier 2 pairs, up to 3 Tier 3 pairs.

---

## 5. Strategy 3: BTC Futures Curve Trading

### 5.1 Term Structure Construction

**Traditional (CEX):** Basis = (Futures - Spot) / Spot, annualized by days to expiry.

**Synthetic (from funding):** Implied futures price = Spot * (1 + funding_rate * time).

Compare actual futures prices to synthetic prices derived from perpetual funding across venues.

### 5.2 Trading Strategies

**A. Calendar Spreads:** Long near-dated, short far-dated when contango > 15% annualized. Exit when basis < 5% or at expiry.

**B. Cross-Venue Basis:** Exploit CME premium over Binance, or Binance quarterly vs Hyperliquid perpetual.

**C. Synthetic Futures:** Replicate futures exposure using lower-cost perpetual funding on Hyperliquid.

**D. Roll Optimization:** Choose optimal venue for rolling expiring positions based on cost comparison.

### 5.3 Regime Classification

| Regime | Annualized Basis | Strategy Adjustment |
|--------|-----------------|-------------------|
| Steep contango | > 20% | Aggressive calendar spreads |
| Mild contango | 5-20% | Selective basis trades |
| Flat | -5% to +5% | Reduce exposure |
| Backwardation | < -5% | Reverse calendar spreads |

---

## 6. Portfolio Construction

### 6.1 Optimization Method

Hierarchical Risk Parity (HRP) is the primary allocation method, chosen for:
- No covariance matrix inversion (more stable)
- Works well with correlated strategies
- Tree-based allocation captures sector structure

### 6.2 Constraints

| Constraint | Limit | Rationale |
|-----------|-------|-----------|
| Max CEX allocation | 70% | Counterparty diversification |
| Max DEX allocation | 30% | Smart contract risk |
| Max single strategy | 25% | Concentration limit |
| Max sector | 40% | Sector diversification |
| Max cross-pair correlation | 0.70 | Avoid redundant positions |
| Leverage (pairs) | 1.0x | Conservative for altcoin pairs |
| Leverage (futures) | 2.0x max | Futures inherently leveraged |

### 6.3 Risk Management

- **VaR limit**: 3% (95%, 1-day)
- **Maximum drawdown**: 20%
- **BTC correlation**: < 0.3
- **Crisis response**: Close Tier 3 positions, reduce Tier 2, maintain Tier 1

---

## 7. Backtest Methodology

### 7.1 Walk-Forward Design

```
Training:  2022-01-01 to 2023-06-30 (18 months)
Testing:   2023-07-01 to 2024-12-31 (18 months)
```

Rolling walk-forward with 6-month refit windows to capture parameter drift.

### 7.2 Transaction Cost Model

| Component | CEX | DEX |
|-----------|-----|-----|
| Entry/exit fee | 0.05% per side | 0.30% swap fee |
| Slippage | 0.02% | 0.25% |
| MEV | $0 | 0.075% |
| Gas | $0 | $1.00 (Arbitrum) |
| **Total (pair trade)** | **~0.20%** | **~1.00%** |

### 7.3 Crisis Event Analysis

Four major events analyzed for strategy resilience:

1. **UST/Luna Collapse** (May 2022): Impact on DeFi pairs, cointegration stability
2. **FTX Bankruptcy** (November 2022): CEX counterparty risk, DEX volume surge
3. **March 2023 Banking Crisis** (USDC depeg): Stablecoin pair behavior
4. **SEC Lawsuits** (June 2023): Token delisting impact, regulatory risk

### 7.4 Capacity Analysis

| Venue | Estimated Capacity | Limiting Factor |
|-------|-------------------|----------------|
| CEX | $10-30M | Daily volume (5% rule) |
| Hybrid | $5-10M | Order book depth |
| DEX | $1-5M | Pool TVL |
| **Combined** | **$20-50M** | CEX-driven |

---

## 8. Comparison to Grain Futures

Altcoin pairs trading is analogous to grain futures spread trading (e.g., corn-soybean) with key differences:

| Dimension | Grain Futures | Crypto Pairs |
|-----------|--------------|-------------|
| Half-life | 30-90 days | 1-30 days (faster mean reversion) |
| Volatility | 15-25% annualized | 60-120% annualized |
| Cointegration stability | Very stable (decades) | Less stable (months to years) |
| Transaction costs | ~$2-5 per contract | 0.20-1.50% per trade |
| Liquidity | Deep, centralized | Fragmented across venues |
| Seasonality | Strong (harvest cycles) | Weak (halving cycles, DeFi seasons) |
| Data history | 50+ years | 3-5 years |

The higher volatility and faster mean reversion in crypto compensates for higher costs and less stable relationships, making the strategy viable despite the structural differences.

---

## 9. Statistical Validation

| Test | Purpose | Threshold |
|------|---------|-----------|
| Engle-Granger | Cointegration | p < 0.05 |
| Johansen Trace | Multivariate cointegration | Reject H0 at 5% |
| ADF | Spread stationarity | p < 0.05 |
| KPSS | Confirm stationarity | p > 0.05 (fail to reject) |
| Ljung-Box | Residual autocorrelation | p > 0.05 |
| Monte Carlo | Sharpe significance | p < 0.05 (10,000 sims) |

---

## 10. References

1. Engle, R.F. & Granger, C.W.J. (1987). Co-Integration and Error Correction. Econometrica.
2. Johansen, S. (1991). Estimation and Hypothesis Testing of Cointegration Vectors. Econometrica.
3. Lopez de Prado, M. (2016). Building Diversified Portfolios that Outperform Out-of-Sample. Journal of Portfolio Management.
4. Gatev, E., Goetzmann, W., & Rouwenhorst, K. (2006). Pairs Trading: Performance of a Relative-Value Arbitrage Rule. Review of Financial Studies.
5. Black, F. & Litterman, R. (1992). Global Portfolio Optimization. Financial Analysts Journal.
