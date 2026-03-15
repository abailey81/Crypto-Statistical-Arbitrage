<p align="center">
  <img src="https://img.shields.io/badge/Crypto-Statistical%20Arbitrage-000000?style=for-the-badge&labelColor=000000" alt="Crypto Statistical Arbitrage" height="40">
</p>

<h1 align="center">Crypto Statistical Arbitrage</h1>

<p align="center">
  <strong>Multi-venue quantitative crypto trading system &mdash; cointegration-based pair selection,<br>ML-enhanced signals, and walk-forward backtesting across 32 CEX/DEX venues.</strong>
</p>

<p align="center">
  <a href="https://github.com/abailey81/Crypto-Statistical-Arbitrage/stargazers"><img src="https://img.shields.io/github/stars/abailey81/Crypto-Statistical-Arbitrage?style=for-the-badge&logo=github&color=gold" alt="Stars"></a>&nbsp;
  <a href="https://github.com/abailey81/Crypto-Statistical-Arbitrage/network/members"><img src="https://img.shields.io/github/forks/abailey81/Crypto-Statistical-Arbitrage?style=for-the-badge&logo=github&color=blue" alt="Forks"></a>&nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">&nbsp;
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">&nbsp;
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">&nbsp;
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">&nbsp;
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy">
</p>

<p align="center">
  <a href="#-key-results">Results</a>&ensp;&bull;&ensp;
  <a href="#-architecture">Architecture</a>&ensp;&bull;&ensp;
  <a href="#-strategies">Strategies</a>&ensp;&bull;&ensp;
  <a href="#-data-pipeline">Data Pipeline</a>&ensp;&bull;&ensp;
  <a href="#-quick-start">Quick Start</a>&ensp;&bull;&ensp;
  <a href="#-project-structure">Structure</a>&ensp;&bull;&ensp;
  <a href="#-development">Development</a>
</p>

---

## Key Results

<table>
<tr>
<td width="50%" valign="top">

### Altcoin Statistical Arbitrage
| Metric | Value |
|:-------|------:|
| **Sharpe Ratio** | **1.61** |
| **Total Return** | 6.84% |
| **Max Drawdown** | 4.64% |
| **Win Rate** | 51.18% |
| **Total Trades** | 127 |
| **Profit Factor** | 1.69 |
| **BTC Correlation** | -0.12 |

</td>
<td width="50%" valign="top">

### BTC Futures Curve Trading
| Metric | Value |
|:-------|------:|
| **Sharpe Ratio** | **5.81** |
| **Total Return** | 203.70% |
| **Max Drawdown** | 0.89% |
| **Win Rate** | 95.02% |
| **Total Trades** | 44,652 |
| **Profit Factor** | 28.53 |
| **BTC Correlation** | -0.05 |

</td>
</tr>
</table>

> **Walk-forward out-of-sample results.** Train: Jan 2022 &ndash; Jun 2023 &ensp;|&ensp; Test: Jul 2023 &ndash; Dec 2024. No leverage (1.0x). Transaction costs included.

---

## Highlights

<table>
<tr>
<td align="center" width="25%">
<br>
<strong>32 Venues</strong><br>
<sub>CEX &bull; DEX &bull; Hybrid</sub><br><br>
</td>
<td align="center" width="25%">
<br>
<strong>211 Symbols</strong><br>
<sub>16 crypto sectors</sub><br><br>
</td>
<td align="center" width="25%">
<br>
<strong>226K+ Lines</strong><br>
<sub>184 Python files</sub><br><br>
</td>
<td align="center" width="25%">
<br>
<strong>137 Dependencies</strong><br>
<sub>Pinned & reproducible</sub><br><br>
</td>
</tr>
<tr>
<td align="center">
<br>
<strong>Walk-Forward</strong><br>
<sub>Out-of-sample validated</sub><br><br>
</td>
<td align="center">
<br>
<strong>ML Enhanced</strong><br>
<sub>GBM + Random Forest</sub><br><br>
</td>
<td align="center">
<br>
<strong>No Leverage</strong><br>
<sub>1.0x only</sub><br><br>
</td>
<td align="center">
<br>
<strong>61 Compliance Checks</strong><br>
<sub>Automated validation</sub><br><br>
</td>
</tr>
</table>

---

## Architecture

```
run_arb.py                              Master orchestrator
  │
  ├── phase1run.py ─► run_phase1.py     Phase 1: Multi-venue data collection
  │     ├── 7 CEX collectors                Binance, Bybit, OKX, Kraken, Coinbase, Deribit, CME
  │     ├── 12 DEX collectors               Uniswap, Curve, GMX, SushiSwap, Jupiter, ...
  │     ├── 3 Hybrid collectors             Hyperliquid, dYdX, Drift
  │     └── 10 Alternative sources          On-chain, sentiment, social, analytics
  │
  ├── phase2run.py                      Phase 2: Altcoin StatArb (5-step pipeline)
  │     ├── Step 1  Universe construction + cointegration testing
  │     ├── Step 2  Baseline z-score mean reversion strategy
  │     ├── Step 3  ML enhancement (Gradient Boosting + Random Forest)
  │     ├── Step 4  Walk-forward backtest + crisis analysis
  │     └── Step 5  Report generation
  │
  ├── run_phase3.py ─► phase3run.py     Phase 3: BTC Futures curve trading
  │     ├── Funding rate term structure
  │     ├── Calendar spread signals
  │     ├── Cross-venue arbitrage
  │     └── Walk-forward backtest
  │
  ├── generate_visualizations.py        34 publication-quality charts
  └── Compliance validator              61 automated checks
```

---

## Strategies

### Phase 2: Altcoin Statistical Arbitrage

Identifies cointegrated cryptocurrency pairs and trades mean-reverting spreads with ML-enhanced signals.

| Parameter | CEX | DEX |
|:----------|:----|:----|
| **Universe** | 50 tokens (top by volume) | 25 tokens (DeFi-native) |
| **Entry Z-Score** | &plusmn; 2.0 | &plusmn; 2.5 |
| **Exit Z-Score** | 0.0 (mean) | \|z\| < 1.0 |
| **Stop Z-Score** | &plusmn; 3.0 | &plusmn; 3.5 |
| **Max Position** | $100,000 | $50,000 |
| **Transaction Cost** | 0.20% (4-leg round trip) | 0.50 &ndash; 1.50% all-in |
| **Max Positions** | 5 &ndash; 8 | 2 &ndash; 3 |

**ML Enhancement:** Gradient Boosting + Random Forest ensemble predicts spread direction, filtering baseline z-score signals. Features include lagged spreads, volatility regimes (HMM), momentum, and cross-pair correlations.

**Risk Controls:**
- Venue-based tier classification (T1: Both CEX, T2: Mixed, T3: Both DEX)
- 40% sector concentration limit, 70% max cross-pair correlation
- Kelly criterion position sizing (0.25 &ndash; 0.5x)
- 1.0x leverage only (no leverage)

### Phase 3: BTC Futures Curve Trading

Exploits the term structure of BTC perpetual funding rates across venues.

| Component | Details |
|:----------|:--------|
| **Venues** | Binance, Hyperliquid, dYdX, OKX, Bybit, GMX, Aevo |
| **Signals** | Funding rate carry, calendar spreads, cross-venue basis |
| **Frequency** | Hourly rebalancing |
| **Walk-Forward** | 6-month train / 18-month test |

---

## Data Pipeline

### Supported Venues (32)

<table>
<tr><th>Type</th><th>Venues</th><th>Data</th></tr>
<tr>
  <td><strong>CEX</strong> (7)</td>
  <td>Binance, Bybit, OKX, Kraken, Coinbase, Deribit, CME</td>
  <td>OHLCV, funding rates, open interest, liquidations, options</td>
</tr>
<tr>
  <td><strong>Hybrid</strong> (3)</td>
  <td>Hyperliquid, dYdX, Drift</td>
  <td>OHLCV, hourly funding rates, open interest</td>
</tr>
<tr>
  <td><strong>DEX</strong> (12)</td>
  <td>Uniswap, Curve, GMX, SushiSwap, Jupiter, 1inch, 0x, CoWSwap, GeckoTerminal, DexScreener, &hellip;</td>
  <td>Pool data, swaps, TVL, liquidity</td>
</tr>
<tr>
  <td><strong>On-Chain</strong> (5)</td>
  <td>Covalent, Bitquery, Santiment, The Graph, Nansen</td>
  <td>Wallet flows, smart money, on-chain metrics</td>
</tr>
<tr>
  <td><strong>Alternative</strong> (5+)</td>
  <td>DeFiLlama, Coinalyze, LunarCrush, Dune, CoinGecko, CryptoCompare, Messari</td>
  <td>TVL, sentiment, social, fundamentals</td>
</tr>
</table>

### Symbol Universe

**211 unique symbols** across 16 sectors with full survivorship bias tracking:

<details>
<summary><strong>View full sector breakdown</strong></summary>

| Sector | Count | Examples |
|:-------|------:|:--------|
| L1 Blockchains | 18 | SOL, AVAX, ADA, DOT, ATOM, TON |
| DeFi DEX | 16 | UNI, SUSHI, CRV, DYDX, GMX, JUP |
| Major Altcoins | 13 | BNB, XRP, DOGE, LTC, BCH |
| Infrastructure | 13 | LINK, GRT, FIL, AR, ENS |
| DeFi Lending | 9 | AAVE, COMP, MKR, ENA |
| Liquid Staking | 9 | LDO, RPL, EIGEN, PENDLE |
| AI / ML | 8 | FET, TAO, WLD, RNDR |
| L2 Solutions | 8 | ARB, OP, MATIC, STRK |
| Meme Tokens | 8 | PEPE, SHIB, BONK, WIF |
| Gaming | 7 | AXS, SAND, MANA, APE |

</details>

---

## Backtesting

The system includes both an **event-driven backtester** and a **vectorized fast backtester**:

| Feature | Details |
|:--------|:--------|
| **Walk-Forward Validation** | Train: Jan 2022 &ndash; Jun 2023 &ensp;\|&ensp; Test: Jul 2023 &ndash; Dec 2024 |
| **Crisis Analysis** | UST/Luna collapse, FTX bankruptcy, Banking crisis, SEC lawsuits |
| **Capacity Analysis** | Market-impact modelling per venue |
| **Attribution** | Per-pair, per-sector, and per-regime P&L decomposition |
| **Compliance** | 61 automated checks via `run_arb.py --validate` |

---

## Quick Start

### Prerequisites

| Requirement | Minimum | Recommended |
|:------------|:--------|:------------|
| **Python** | 3.10 | 3.12 |
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
source .venv/bin/activate      # macOS / Linux
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

> **Note:** Many data sources (Binance, Bybit, OKX, Hyperliquid, dYdX, GeckoTerminal, DeFiLlama, etc.) work without API keys using public endpoints.

### Running

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

# Validate compliance (61 checks)
python run_arb.py --validate
```

<details>
<summary><strong>All run modes</strong></summary>

| Mode | Command | Description |
|:-----|:--------|:------------|
| **Full Pipeline** | `python run_arb.py` | All phases + visualizations + compliance |
| **Cold Run** | `python run_arb.py --clean-cache` | Clear caches, run from scratch |
| **Warm Run** | `python run_arb.py --skip-phase1` | Skip data collection |
| **Phase Select** | `python run_arb.py --phase 2` | Run specific phase(s) |
| **Validate** | `python run_arb.py --validate` | 61-check compliance audit |
| **Check Data** | `python run_arb.py --check-only` | Data readiness audit |
| **1-Day Test** | `python run_phase1.py --start 2026-02-08 --end 2026-02-09` | Smoke test |

</details>

---

## Project Structure

<details>
<summary><strong>View full project tree</strong></summary>

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
│   ├── futures_curve/          #   Term structure, calendar spreads
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
├── notebooks/                  # Jupyter notebooks
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
├── phase2run.py                # Phase 2 engine (~5,700 lines)
├── run_phase1.py               # Phase 1 engine (~5,400 lines)
├── run_phase3.py               # Phase 3 entry point
├── generate_visualizations.py  # Chart generator (34 visualizations)
├── requirements.txt            # Dependencies (137 packages)
├── setup.py                    # Package configuration
└── Makefile                    # Build automation
```

</details>

---

## Configuration

All strategy parameters are defined in `config/config.yaml`:

| Section | What it controls |
|:--------|:-----------------|
| `universe` | Token lists, sector mappings, venue assignments |
| `cointegration` | Half-life bounds, p-value thresholds, test window |
| `strategy` | Z-score entry/exit/stop, position sizing, Kelly fraction |
| `risk` | Drawdown limits, concentration caps, correlation thresholds |
| `backtest` | Train/test dates, transaction costs, walk-forward windows |
| `venues` | Per-venue endpoints, rate limits, fee schedules |

---

## Development

```bash
make format        # Black + isort formatting
make lint          # Flake8 linting
make type-check    # mypy type checking
make quality       # All of the above

make test              # Run all tests
make test-unit         # Unit tests only
make test-integration  # Integration tests
make test-coverage     # With coverage report
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development guidelines.

---

## Documentation

| Document | Description |
|:---------|:------------|
| [Methodology](docs/methodology.md) | Statistical methodology &mdash; cointegration, Kalman filter, HMM |
| [Data Dictionary](docs/data_dictionary.md) | Data schema and field reference |
| [Data Sources](docs/data_sources.md) | Venue documentation and capabilities |
| [API Reference](docs/api_reference.md) | Module-level API reference |
| [Venue Comparison](docs/venue_comparison.md) | Cross-venue comparison analysis |

---

## Dependencies

137 pinned packages organized by function:

<details>
<summary><strong>View dependency breakdown</strong></summary>

| Category | Key Packages |
|:---------|:-------------|
| **Scientific Computing** | numpy, pandas, scipy |
| **Data Collection** | ccxt, aiohttp, requests, websockets, httpx |
| **Econometrics** | statsmodels, arch, hmmlearn |
| **Machine Learning** | scikit-learn, lightgbm, xgboost |
| **GPU Acceleration** | numba, pyopencl, joblib |
| **Portfolio Optimization** | cvxpy |
| **Data Storage** | pyarrow, fastparquet, h5py |
| **Visualization** | matplotlib, seaborn, plotly, kaleido |
| **Configuration** | pydantic, python-dotenv, PyYAML |

</details>

---

## Disclaimer

This project is for **educational and research purposes only**. It is not financial advice. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research before making any investment decisions.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built by <a href="https://github.com/abailey81"><strong>Tamer Atesyakar</strong></a></sub>
</p>
