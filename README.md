<p align="center">
  <h1 align="center">Crypto Statistical Arbitrage</h1>
  <p align="center">
    <strong>Multi-Venue Quantitative Trading System</strong>
  </p>
  <p align="center">
    <a href="#key-results"><img src="https://img.shields.io/badge/Sharpe-1.61%20%7C%205.81-brightgreen?style=for-the-badge" alt="Sharpe Ratio"></a>
    <a href="#key-results"><img src="https://img.shields.io/badge/Win%20Rate-51%25%20%7C%2095%25-blue?style=for-the-badge" alt="Win Rate"></a>
    <a href="#key-results"><img src="https://img.shields.io/badge/BTC%20Correlation--0.12%20%7C%20--0.05-orange?style=for-the-badge" alt="BTC Correlation"></a>
  </p>
  <p align="center">
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/Venues-32-purple.svg" alt="32 Venues">
    <img src="https://img.shields.io/badge/Symbols-211-red.svg" alt="211 Symbols">
    <img src="https://img.shields.io/badge/Lines%20of%20Code-226K-lightgrey.svg" alt="226K Lines">
  </p>
</p>

---

A production-grade quantitative cryptocurrency trading system implementing statistical arbitrage across **32 venues** — centralized exchanges (CEX), decentralized exchanges (DEX), and hybrid platforms. The system covers the complete quant pipeline: multi-venue data acquisition, cointegration-based pair selection, ML-enhanced signal generation, walk-forward backtesting with crisis analysis, and automated report generation with publication-quality visualizations.

## Key Results

<table>
<tr>
<td>

### Altcoin Statistical Arbitrage
| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | 1.61 |
| **Total Return** | 6.84% |
| **Max Drawdown** | 4.64% |
| **Win Rate** | 51.18% |
| **Trades** | 127 |
| **BTC Correlation** | -0.12 |
| **Profit Factor** | 1.69 |

</td>
<td>

### BTC Futures Curve Trading
| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | 5.81 |
| **Total Return** | 203.70% |
| **Max Drawdown** | 0.89% |
| **Win Rate** | 95.02% |
| **Trades** | 44,652 |
| **BTC Correlation** | -0.05 |
| **Profit Factor** | 28.53 |

</td>
</tr>
</table>

> Walk-forward out-of-sample results. Train: Jan 2022 - Jun 2023 | Test: Jul 2023 - Dec 2024. No leverage (1.0x). Transaction costs included.

---

## Architecture

```
run_arb.py                          # Master orchestrator
  |
  +-- phase1run.py --> run_phase1.py    # Phase 1: Multi-venue data collection (32 venues)
  |     +-- 7 CEX collectors            #   Binance, Bybit, OKX, Kraken, Coinbase, Deribit, CME
  |     +-- 12 DEX collectors           #   Uniswap, Curve, GMX, SushiSwap, Jupiter, ...
  |     +-- 3 Hybrid collectors         #   Hyperliquid, dYdX, Drift
  |     +-- 10 Alternative sources      #   On-chain, sentiment, social, analytics
  |
  +-- phase2run.py                      # Phase 2: Altcoin StatArb (5-step pipeline)
  |     +-- Step 1: Universe construction + cointegration testing
  |     +-- Step 2: Baseline z-score mean reversion strategy
  |     +-- Step 3: ML enhancement (Gradient Boosting + Random Forest)
  |     +-- Step 4: Walk-forward backtest + crisis analysis
  |     +-- Step 5: Report generation
  |
  +-- run_phase3.py --> phase3run.py    # Phase 3: BTC Futures curve trading
  |     +-- Funding rate term structure
  |     +-- Calendar spread signals
  |     +-- Cross-venue arbitrage
  |     +-- Walk-forward backtest
  |
  +-- generate_visualizations.py        # 34 publication-quality charts
  +-- Compliance validator              # 61 automated checks
```

---

## Project Structure

```
.
├── config/                     # Configuration
│   ├── config.yaml             #   Strategy parameters, risk limits, dates
│   ├── venues.yaml             #   32 venue configs (endpoints, costs, capacity)
│   ├── symbols.yaml            #   211 symbols across 16 sectors
│   └── api_keys_template.env   #   API key template (copy to .env)
│
├── data_collection/            # Phase 1: Data acquisition layer
│   ├── cex/                    #   CEX collectors (Binance, Bybit, OKX, ...)
│   ├── dex/                    #   DEX collectors (Uniswap, Curve, GMX, ...)
│   ├── hybrid/                 #   Hybrid collectors (Hyperliquid, dYdX, Drift)
│   ├── onchain/                #   On-chain analytics (10 providers)
│   ├── options/                #   Options data (Deribit, Aevo)
│   ├── alternative/            #   Alternative data (DeFiLlama, Coinalyze, ...)
│   ├── market_data/            #   Market data aggregators
│   ├── indexers/               #   Blockchain indexers (The Graph)
│   └── utils/                  #   Rate limiting, caching, validation, storage
│
├── strategies/                 # Trading strategies
│   ├── pairs_trading/          #   Cointegration, Kalman filter, ML signals
│   ├── futures_curve/          #   Term structure, calendar spreads, roll opt.
│   ├── funding_rate_arb/       #   Cross-venue funding rate arbitrage
│   └── vol_surface_or_dex_arb/ #   Options vol surface / DEX arbitrage
│
├── backtesting/                # Backtesting engine
│   ├── backtest_engine.py      #   Core event-driven backtester
│   ├── optimized_backtest.py   #   Vectorized fast backtester
│   └── analysis/               #   Walk-forward, crisis, capacity, attribution
│
├── portfolio/                  # Portfolio construction
│   ├── optimizer.py            #   HRP, MVO, risk parity, Black-Litterman
│   └── risk_manager.py         #   Drawdown stops, VaR limits, stress tests
│
├── reporting/                  # Report generation
│   ├── advanced_report_generator.py
│   └── strict_pdf_validator.py #   61-check compliance validator
│
├── execution/                  # Execution layer
│   ├── order_manager.py        #   Order routing and management
│   └── slippage_model.py       #   Venue-specific slippage models
│
├── notebooks/                  # Jupyter notebooks (7)
│   ├── 00_data_acquisition_plan.ipynb
│   ├── 01_cex_data_exploration.ipynb
│   ├── 02_dex_data_exploration.ipynb
│   ├── 03_venue_comparison.ipynb
│   ├── 04_strategy_development.ipynb
│   ├── 05_multi_venue_backtesting.ipynb
│   └── 06_portfolio_construction.ipynb
│
├── docs/                       # Documentation
│   ├── methodology.md          #   Statistical methodology
│   ├── data_dictionary.md      #   Data schema reference
│   ├── data_sources.md         #   Venue documentation
│   ├── api_reference.md        #   API reference
│   └── venue_comparison.md     #   Venue comparison analysis
│
├── tests/                      # Test suite
│   ├── unit/                   #   Unit tests
│   ├── integration/            #   Integration tests
│   └── performance/            #   Performance benchmarks
│
├── run_arb.py                  # Master orchestrator
├── phase1run.py                # Phase 1 entry point
├── phase2run.py                # Phase 2 engine (5,706 lines)
├── run_phase1.py               # Phase 1 engine (5,398 lines)
├── run_phase3.py               # Phase 3 entry point
├── generate_visualizations.py  # Chart generator (34 visualizations)
├── requirements.txt            # Dependencies (137 packages)
├── setup.py                    # Package configuration
└── Makefile                    # Build automation
```

---

## Strategy Overview

### Phase 2: Altcoin Statistical Arbitrage

Identifies cointegrated cryptocurrency pairs and trades mean-reverting spreads with ML-enhanced signals.

| Parameter | CEX | DEX |
|-----------|-----|-----|
| **Universe** | 50 tokens (top by volume) | 25 tokens (DeFi-native) |
| **Entry Z-Score** | +/- 2.0 | +/- 2.5 |
| **Exit Z-Score** | 0.0 (mean) | \|z\| < 1.0 |
| **Stop Z-Score** | +/- 3.0 | +/- 3.5 |
| **Max Position** | $100,000 | $50,000 |
| **Transaction Cost** | 0.20% (4-leg round trip) | 0.50 - 1.50% all-in |
| **Max Positions** | 5 - 8 | 2 - 3 |

**ML Enhancement:** Gradient Boosting + Random Forest ensemble predicts spread direction, filtering baseline z-score signals. Features include lagged spreads, volatility regimes (HMM), momentum, and cross-pair correlations.

**Risk Controls:**
- Venue-based tier classification (T1: Both CEX, T2: Mixed, T3: Both DEX)
- 40% sector concentration limit, 70% max cross-pair correlation
- Kelly criterion position sizing (0.25 - 0.5x)
- 1.0x leverage only (no leverage)

### Phase 3: BTC Futures Curve Trading

Exploits the term structure of BTC perpetual funding rates across venues.

| Component | Details |
|-----------|---------|
| **Venues** | Binance, Hyperliquid, dYdX, OKX, Bybit, GMX, Aevo |
| **Signals** | Funding rate carry, calendar spreads, cross-venue basis |
| **Frequency** | Hourly rebalancing |
| **Walk-Forward** | 6-month train / 18-month test |

---

## Getting Started

### Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.10 | 3.13+ |
| **RAM** | 8 GB | 16 GB |
| **Disk** | 5 GB | 10 GB |
| **OS** | macOS / Linux / Windows (WSL) | macOS (Apple Silicon) |

### Installation

```bash
# Clone the repository
git clone https://github.com/abailey81/Crypto-Statistical-Arbitrage.git
cd Crypto-Statistical-Arbitrage

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# macOS only (required by XGBoost)
brew install libomp
```

### Configuration

```bash
# Copy the API key template
cp config/api_keys_template.env config/.env

# Edit with your API keys (many venues work without keys)
nano config/.env

# Verify credentials
python config/verify_my_credentials.py
```

> **Note:** Many data sources (Binance, Bybit, OKX, Hyperliquid, dYdX, GeckoTerminal, DeFiLlama, etc.) work without API keys using public endpoints. See [config/api_keys_template.env](config/api_keys_template.env) for the full list.

### Quick Start

```bash
# Full pipeline: Data Collection + Altcoin StatArb + BTC Futures + Visualizations
python run_arb.py

# Skip data collection (use existing data)
python run_arb.py --skip-phase1

# Run specific phases
python run_arb.py --phase 2        # Altcoin StatArb only
python run_arb.py --phase 3        # BTC Futures only
python run_arb.py --phase 2 3      # Both strategies

# Cold run (clear all caches)
python run_arb.py --clean-cache

# Validate compliance
python run_arb.py --validate
```

---

## Data Pipeline

### Supported Venues (32)

<table>
<tr><th>Type</th><th>Venues</th><th>Data</th></tr>
<tr>
  <td><strong>CEX</strong></td>
  <td>Binance, Bybit, OKX, Kraken, Coinbase, Deribit, CME</td>
  <td>OHLCV, funding rates, open interest, liquidations, options</td>
</tr>
<tr>
  <td><strong>Hybrid</strong></td>
  <td>Hyperliquid, dYdX, Drift</td>
  <td>OHLCV, hourly funding rates, open interest</td>
</tr>
<tr>
  <td><strong>DEX</strong></td>
  <td>Uniswap, Curve, GMX, SushiSwap, Jupiter, 1inch, 0x, CoWSwap, GeckoTerminal, DexScreener</td>
  <td>Pool data, swaps, TVL, liquidity</td>
</tr>
<tr>
  <td><strong>On-Chain</strong></td>
  <td>Covalent, Bitquery, Santiment, The Graph, Nansen</td>
  <td>Wallet flows, smart money, on-chain metrics</td>
</tr>
<tr>
  <td><strong>Alternative</strong></td>
  <td>DeFiLlama, Coinalyze, LunarCrush, Dune, CoinGecko, CryptoCompare, Messari</td>
  <td>TVL, sentiment, social, fundamentals</td>
</tr>
</table>

### Symbol Universe

**211 unique symbols** across 16 sectors with full survivorship bias tracking:

| Sector | Count | Examples |
|--------|-------|---------|
| L1 Blockchains | 18 | SOL, AVAX, ADA, DOT, ATOM, TON |
| DeFi DEX | 16 | UNI, SUSHI, CRV, DYDX, GMX, JUP |
| Major Altcoins | 13 | BNB, XRP, DOGE, LTC, BCH |
| Infrastructure | 13 | LINK, GRT, FIL, AR, ENS |
| DeFi Lending | 9 | AAVE, COMP, MKR, ENA |
| Liquid Staking | 9 | LDO, RPL, EIGEN, PENDLE |
| AI/ML | 8 | FET, TAO, WLD, RNDR |
| L2 Solutions | 8 | ARB, OP, MATIC, STRK |
| Meme Tokens | 8 | PEPE, SHIB, BONK, WIF |
| Gaming | 7 | AXS, SAND, MANA, APE |

---

## Run Modes

| Mode | Command | Description |
|------|---------|-------------|
| **Full Pipeline** | `python run_arb.py` | All phases + visualizations + compliance |
| **Cold Run** | `python run_arb.py --clean-cache` | Clear caches, run from scratch |
| **Warm Run** | `python run_arb.py --skip-phase1` | Skip data collection |
| **Phase Select** | `python run_arb.py --phase 2` | Run specific phase(s) |
| **Validate** | `python run_arb.py --validate` | 61-check compliance audit |
| **Check Data** | `python run_arb.py --check-only` | Data readiness audit |
| **1-Day Test** | `python run_phase1.py --start 2026-02-08 --end 2026-02-09` | Smoke test |

---

## Testing

```bash
make test              # Run all tests
make test-unit         # Unit tests only
make test-integration  # Integration tests
make test-coverage     # With coverage report
```

## Code Quality

```bash
make lint              # Flake8 linting
make format            # Black + isort formatting
make type-check        # mypy type checking
make quality           # All checks
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/methodology.md](docs/methodology.md) | Statistical methodology (cointegration, Kalman, HMM) |
| [docs/data_dictionary.md](docs/data_dictionary.md) | Data schema and field reference |
| [docs/data_sources.md](docs/data_sources.md) | Venue documentation and capabilities |
| [docs/api_reference.md](docs/api_reference.md) | API reference for all modules |
| [docs/venue_comparison.md](docs/venue_comparison.md) | Cross-venue comparison analysis |

---

## Dependencies

137 pinned packages organized by function:

| Category | Key Packages |
|----------|-------------|
| **Scientific Computing** | numpy, pandas, scipy |
| **Data Collection** | ccxt, aiohttp, requests, websockets, httpx |
| **Econometrics** | statsmodels, arch, hmmlearn |
| **Machine Learning** | scikit-learn, lightgbm, xgboost |
| **GPU Acceleration** | numba, pyopencl, joblib |
| **Portfolio Optimization** | cvxpy |
| **Data Storage** | pyarrow, fastparquet, h5py |
| **Visualization** | matplotlib, seaborn, plotly, kaleido |
| **Configuration** | pydantic, python-dotenv, PyYAML |

---

## Disclaimer

This project is for **educational and research purposes only**. It is not financial advice. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research before making any investment decisions.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built by <strong>Tamer Atesyakar</strong></sub>
</p>
