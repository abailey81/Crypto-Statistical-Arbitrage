# Contributing to Crypto Statistical Arbitrage

Thank you for your interest in contributing. This document provides guidelines and instructions for contributing to the project.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Project Architecture](#project-architecture)
- [Reporting Issues](#reporting-issues)

---

## Development Setup

### Prerequisites

| Requirement | Version |
|:------------|:--------|
| Python | 3.10 - 3.12 (3.13 is **not** supported) |
| RAM | 8 GB minimum, 16 GB recommended |
| Disk | 5 GB minimum |
| OS | macOS, Linux, or Windows (WSL) |

### Installation

```bash
# Clone the repository
git clone https://github.com/abailey81/Crypto-Statistical-Arbitrage.git
cd Crypto-Statistical-Arbitrage

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate    # macOS / Linux

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

---

## Code Style

This project uses the following tools for code quality:

| Tool | Purpose | Config |
|:-----|:--------|:-------|
| **Black** | Code formatting | Line length 99 |
| **Flake8** | Linting | Standard rules |
| **isort** | Import sorting | Black-compatible profile |
| **mypy** | Type checking | Strict mode |

### Formatting Commands

```bash
make format       # Run Black + isort
make lint         # Run Flake8
make type-check   # Run mypy
make quality      # All checks
```

### Style Guidelines

- Use `Union[int, str]` for parameters accepting flexible ID formats
- Follow PEP 8 naming conventions
- Add type hints to all function signatures
- Write docstrings for all public functions and classes
- Keep functions focused -- prefer many small functions over few large ones

---

## Testing

### Running Tests

```bash
make test              # Run all tests
make test-unit         # Unit tests only
make test-integration  # Integration tests
make test-coverage     # With coverage report
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Place performance benchmarks in `tests/performance/`
- Name test files with the `test_` prefix
- Use descriptive test names: `test_cointegration_rejects_nonstationary_pair`
- Mock external API calls -- tests should not require live venue connections

### Test Coverage

All pull requests should maintain or improve test coverage. Run `make test-coverage` to check.

---

## Pull Request Process

### Before Submitting

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines above.

3. **Run the full quality suite**:
   ```bash
   make quality    # Lint + format check
   make test       # All tests
   ```

4. **If you modified a strategy**, run the compliance validator:
   ```bash
   python run_arb.py --validate
   ```

### Submitting the PR

- Fill out the pull request template completely
- Keep PRs focused on a single concern
- Reference any related issues with `Closes #123`
- Include before/after backtest results if modifying strategy logic

### Review Process

- All PRs require review before merging
- Strategy changes require walk-forward backtest validation
- CI must pass (lint, format, tests)
- Keep discussion constructive and focused on the code

---

## Project Architecture

```
run_arb.py                    # Master orchestrator
  |
  +-- Phase 1: Data Collection (32 venues)
  |     data_collection/cex/      CEX collectors
  |     data_collection/dex/      DEX collectors
  |     data_collection/hybrid/   Hybrid collectors
  |     data_collection/onchain/  On-chain analytics
  |
  +-- Phase 2: Altcoin StatArb
  |     phase2run.py              5-step pipeline
  |     strategies/pairs_trading/ Cointegration + ML signals
  |
  +-- Phase 3: BTC Futures Curve
  |     run_phase3.py             Funding rate term structure
  |     strategies/futures_curve/ Calendar spreads
  |
  +-- Shared Components
        backtesting/              Event-driven + vectorized backtester
        portfolio/                HRP, MVO, risk parity
        reporting/                Charts + compliance validator
```

### Key Files

| File | Lines | Purpose |
|:-----|------:|:--------|
| `phase2run.py` | ~5,700 | Altcoin StatArb engine |
| `run_phase1.py` | ~5,400 | Data collection engine |
| `run_arb.py` | -- | Master orchestrator |
| `generate_visualizations.py` | -- | 34 publication-quality charts |

---

## Reporting Issues

- Use the [bug report template](https://github.com/abailey81/Crypto-Statistical-Arbitrage/issues/new?template=bug_report.yml) for bugs
- Use the [feature request template](https://github.com/abailey81/Crypto-Statistical-Arbitrage/issues/new?template=feature_request.yml) for enhancements
- Check [existing issues](https://github.com/abailey81/Crypto-Statistical-Arbitrage/issues) before creating a new one

---

## Security

If you discover a security vulnerability (e.g., exposed API keys, credential leaks), please **do not** open a public issue. See [SECURITY.md](SECURITY.md) for responsible disclosure instructions.

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
