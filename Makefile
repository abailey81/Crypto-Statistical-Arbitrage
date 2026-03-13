# =============================================================================
# Crypto Statistical Arbitrage - Multi-Venue Trading System
# Makefile for Common Commands
# =============================================================================

.PHONY: help install install-dev install-all clean test test-unit test-integration \
        test-coverage lint format type-check collect-all collect-binance collect-bybit \
        collect-hyperliquid collect-dydx collect-uniswap collect-deribit backtest \
        backtest-pairs backtest-futures report docs notebook

# Default target
.DEFAULT_GOAL := help

# Python interpreter
PYTHON := python
PIP := pip

# Directories
SRC_DIR := .
TEST_DIR := tests
DATA_DIR := data
REPORT_DIR := reports

# Configuration
CONFIG_FILE := config/config.yaml

# =============================================================================
# Help
# =============================================================================

help:
	@echo "============================================================================="
	@echo "Crypto Statistical Arbitrage - Multi-Venue Trading System"
	@echo "============================================================================="
	@echo ""
	@echo "Installation:"
	@echo "  make install          Install core dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo "  make install-all      Install all dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-coverage    Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run linting (flake8, pylint)"
	@echo "  make format           Format code (black, isort)"
	@echo "  make type-check       Run type checking (mypy)"
	@echo "  make quality          Run all quality checks"
	@echo ""
	@echo "Data Collection:"
	@echo "  make collect-all      Collect data from all venues"
	@echo "  make collect-binance  Collect Binance data"
	@echo "  make collect-bybit    Collect Bybit data (validation)"
	@echo "  make collect-hyperliquid  Collect Hyperliquid data"
	@echo "  make collect-dydx     Collect dYdX V4 data"
	@echo "  make collect-uniswap  Collect Uniswap V3 data"
	@echo "  make collect-deribit  Collect Deribit options data"
	@echo ""
	@echo "Backtesting:"
	@echo "  make backtest         Run full backtest suite"
	@echo "  make backtest-pairs   Run pairs trading backtest"
	@echo "  make backtest-futures Run futures curve backtest"
	@echo ""
	@echo "Reporting:"
	@echo "  make report           Generate analysis reports"
	@echo "  make docs             Build documentation"
	@echo "  make notebook         Start Jupyter notebook server"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            Clean generated files"
	@echo "  make clean-data       Clean data directory (CAUTION)"
	@echo "  make clean-all        Clean everything (CAUTION)"
	@echo ""

# =============================================================================
# Installation
# =============================================================================

install:
	@echo "Installing core dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "Installation complete."

install-dev:
	@echo "Installing development dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"
	@echo "Development installation complete."

install-all:
	@echo "Installing all dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[all]"
	@echo "Full installation complete."

# =============================================================================
# Testing
# =============================================================================

test:
	@echo "Running all tests..."
	$(PYTHON) -m pytest $(TEST_DIR) -v --tb=short

test-unit:
	@echo "Running unit tests..."
	$(PYTHON) -m pytest $(TEST_DIR)/unit -v --tb=short

test-integration:
	@echo "Running integration tests..."
	$(PYTHON) -m pytest $(TEST_DIR)/integration -v --tb=short

test-coverage:
	@echo "Running tests with coverage..."
	$(PYTHON) -m pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

# =============================================================================
# Code Quality
# =============================================================================

lint:
	@echo "Running linters..."
	$(PYTHON) -m flake8 data_collection strategies backtesting portfolio execution --max-line-length=100 --ignore=E501,W503
	@echo "Linting complete."

format:
	@echo "Formatting code..."
	$(PYTHON) -m black data_collection strategies backtesting portfolio execution tests --line-length=100
	$(PYTHON) -m isort data_collection strategies backtesting portfolio execution tests --profile black
	@echo "Formatting complete."

type-check:
	@echo "Running type checker..."
	$(PYTHON) -m mypy data_collection strategies backtesting portfolio execution --ignore-missing-imports
	@echo "Type checking complete."

quality: lint type-check
	@echo "All quality checks complete."

# =============================================================================
# Data Collection
# =============================================================================

collect-all:
	@echo "Collecting data from all venues..."
	$(PYTHON) -m data_collection.run_collection --all --config $(CONFIG_FILE)
	@echo "Data collection complete."

collect-binance:
	@echo "Collecting Binance data..."
	$(PYTHON) -m data_collection.run_collection --venue binance --config $(CONFIG_FILE)
	@echo "Binance data collection complete."

collect-bybit:
	@echo "Collecting Bybit data..."
	$(PYTHON) -m data_collection.run_collection --venue bybit --config $(CONFIG_FILE)
	@echo "Bybit data collection complete."

collect-hyperliquid:
	@echo "Collecting Hyperliquid data..."
	$(PYTHON) -m data_collection.run_collection --venue hyperliquid --config $(CONFIG_FILE)
	@echo "Hyperliquid data collection complete."

collect-dydx:
	@echo "Collecting dYdX V4 data..."
	$(PYTHON) -m data_collection.run_collection --venue dydx --config $(CONFIG_FILE)
	@echo "dYdX V4 data collection complete."

collect-uniswap:
	@echo "Collecting Uniswap V3 data..."
	$(PYTHON) -m data_collection.run_collection --venue uniswap --config $(CONFIG_FILE)
	@echo "Uniswap V3 data collection complete."

collect-deribit:
	@echo "Collecting Deribit options data..."
	$(PYTHON) -m data_collection.run_collection --venue deribit --config $(CONFIG_FILE)
	@echo "Deribit data collection complete."

# =============================================================================
# Backtesting
# =============================================================================

backtest:
	@echo "Running full backtest suite..."
	$(PYTHON) -m backtesting.backtest_engine --all --config $(CONFIG_FILE)
	@echo "Backtest complete."

backtest-pairs:
	@echo "Running pairs trading backtest..."
	$(PYTHON) -m backtesting.backtest_engine --strategy pairs_trading --config $(CONFIG_FILE)
	@echo "Pairs trading backtest complete."

backtest-futures:
	@echo "Running futures curve backtest..."
	$(PYTHON) -m backtesting.backtest_engine --strategy futures_curve --config $(CONFIG_FILE)
	@echo "Futures curve backtest complete."

# =============================================================================
# Reporting
# =============================================================================

report:
	@echo "Generating analysis reports..."
	$(PYTHON) -m backtesting.visualization --config $(CONFIG_FILE) --output $(REPORT_DIR)
	@echo "Reports generated in $(REPORT_DIR)/"

docs:
	@echo "Building documentation..."
	cd docs && $(MAKE) html
	@echo "Documentation built in docs/_build/html/"

notebook:
	@echo "Starting Jupyter notebook server..."
	$(PYTHON) -m jupyter lab --notebook-dir=notebooks

# =============================================================================
# Cleaning
# =============================================================================

clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	@echo "Clean complete."

clean-data:
	@echo "WARNING: This will delete all collected data!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] && rm -rf $(DATA_DIR)/raw/* $(DATA_DIR)/processed/* || echo "Aborted."

clean-all: clean
	@echo "WARNING: This will delete all data and generated files!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] && rm -rf $(DATA_DIR)/raw/* $(DATA_DIR)/processed/* $(REPORT_DIR)/* || echo "Aborted."

# =============================================================================
# Development Utilities
# =============================================================================

# Create a new virtual environment
venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source venv/bin/activate  (Linux/Mac)"
	@echo "  venv\\Scripts\\activate    (Windows)"

# Check configuration
check-config:
	@echo "Checking configuration..."
	$(PYTHON) -c "from data_collection.utils.config_loader import ConfigLoader; c = ConfigLoader('$(CONFIG_FILE)'); print('Configuration valid.')"

# Validate data quality
validate-data:
	@echo "Validating data quality..."
	$(PYTHON) -m data_collection.utils.data_validator --data-dir $(DATA_DIR)/raw --output $(DATA_DIR)/metadata/quality_report.md
	@echo "Validation complete. Report saved to $(DATA_DIR)/metadata/quality_report.md"
