"""
Crypto Statistical Arbitrage - Multi-Venue Trading System
=========================================================

Professional-quality package configuration for the cryptocurrency
statistical arbitrage system spanning CEX, DEX, and hybrid venues.

Installation
------------
Minimal:        pip install crypto-statarb-multiverse
With ML:        pip install crypto-statarb-multiverse[ml]
With viz:       pip install crypto-statarb-multiverse[viz]
Development:    pip install -e ".[dev]"
Full:           pip install crypto-statarb-multiverse[all]

Entry Points
------------
    crypto-collect      - Data collection CLI
    crypto-backtest     - Backtesting CLI
    crypto-validate     - Data validation CLI

Custom Commands
---------------
    python setup.py test        - Run pytest
    python setup.py clean       - Remove build artifacts
    python setup.py lint        - Run linting
    python setup.py typecheck   - Run mypy

Author: Tamer Atesyakar
Version: 2.0.0
License: MIT
"""

from __future__ import annotations

import os
import sys
import re
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict

from setuptools import setup, find_packages, Command
from setuptools.command.test import test as TestCommand
from setuptools.command.develop import develop as DevelopCommand
from setuptools.command.install import install as InstallCommand


# =============================================================================
# PACKAGE METADATA
# =============================================================================

PACKAGE_NAME = "crypto-statarb-multiverse"
VERSION = "2.0.0"
DESCRIPTION = (
    "Professional-quality multi-venue cryptocurrency statistical arbitrage system "
    "supporting CEX, DEX, and hybrid venues with 47+ data collectors"
)
AUTHOR = "Tamer Atesyakar"
AUTHOR_EMAIL = "abailey81@users.noreply.github.com"
URL = "https://github.com/abailey81/Crypto-Statistical-Arbitrage"
LICENSE = "MIT"
PYTHON_REQUIRES = ">=3.10"

THIS_DIRECTORY = Path(__file__).parent.resolve()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def read_file(filename: str) -> str:
    """Read file content."""
    filepath = THIS_DIRECTORY / filename
    if filepath.exists():
        return filepath.read_text(encoding="utf-8")
    return ""


def read_readme() -> str:
    """Read README for long description."""
    return read_file("README.md")


def read_version() -> str:
    """Read version from package."""
    patterns = [r"__version__\s*=\s*['\"]([^'\"]+)['\"]"]
    files = ["data_collection/__version__.py", "data_collection/__init__.py"]
    
    for f in files:
        filepath = THIS_DIRECTORY / f
        if filepath.exists():
            content = filepath.read_text(encoding="utf-8")
            for pattern in patterns:
                match = re.search(pattern, content)
                if match:
                    return match.group(1)
    return VERSION


# =============================================================================
# CLASSIFIERS AND KEYWORDS
# =============================================================================

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Financial and Insurance Industry",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Office/Business :: Financial :: Investment",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Typing :: Typed",
    "Framework :: AsyncIO",
    "Framework :: Pytest",
]

KEYWORDS = [
    "cryptocurrency", "trading", "statistical-arbitrage", "quantitative-finance",
    "algorithmic-trading", "backtesting", "portfolio-optimization",
    "defi", "dex", "cex", "funding-rates", "perpetual-futures",
    "binance", "hyperliquid", "dydx", "uniswap", "ccxt",
]


# =============================================================================
# DEPENDENCIES
# =============================================================================

INSTALL_REQUIRES: List[str] = [
    # Scientific computing
    "numpy>=1.24.0,<3.0.0",
    "pandas>=2.0.0,<3.0.0",
    "scipy>=1.10.0,<2.0.0",
    
    # Data collection
    "ccxt>=4.0.0",
    "aiohttp>=3.8.0",
    "requests>=2.28.0",
    "websockets>=10.0",
    "httpx>=0.23.0",
    
    # Statistical analysis
    "statsmodels>=0.14.0",
    "scikit-learn>=1.2.0",
    
    # Data storage
    "pyarrow>=12.0.0",
    
    # Configuration
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "python-dotenv>=1.0.0",
    "PyYAML>=6.0.0",
    
    # Logging & Progress
    "structlog>=23.1.0",
    "tqdm>=4.65.0",
    "rich>=13.0.0",
    
    # Utilities
    "tenacity>=8.2.0",
    "cachetools>=5.0.0",
    "python-dateutil>=2.8.0",
    "pytz>=2023.0",
    "typing-extensions>=4.5.0",
]

EXTRAS_REQUIRE: Dict[str, List[str]] = {
    "ml": [
        "lightgbm>=4.0.0",
        "xgboost>=2.0.0",
        "hmmlearn>=0.3.0",
        "arch>=6.0.0",
    ],
    "portfolio": [
        "cvxpy>=1.3.0",
    ],
    "options": [
        "py_vollib>=1.0.0",
    ],
    "viz": [
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "kaleido>=0.2.0",
    ],
    "dev": [
        "pytest>=7.3.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0.0",
        "pytest-xdist>=3.0.0",
        "pytest-timeout>=2.1.0",
        "pytest-mock>=3.10.0",
        "hypothesis>=6.75.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "pylint>=2.17.0",
        "pre-commit>=3.0.0",
        "pandas-stubs>=2.0.0",
        "types-PyYAML>=6.0.0",
        "types-requests>=2.28.0",
    ],
    "docs": [
        "sphinx>=6.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "myst-parser>=1.0.0",
    ],
    "notebook": [
        "jupyter>=1.0.0",
        "jupyterlab>=4.0.0",
        "ipywidgets>=8.0.0",
    ],
    "async": [
        "asyncio-throttle>=1.0.0",
        "aiofiles>=23.0.0",
    ],
    "storage": [
        "fastparquet>=2023.0.0",
        "h5py>=3.8.0",
        "orjson>=3.8.0",
    ],
}

# Convenience combinations
EXTRAS_REQUIRE["research"] = list(set(
    EXTRAS_REQUIRE["ml"] + EXTRAS_REQUIRE["viz"] + EXTRAS_REQUIRE["notebook"]
))

EXTRAS_REQUIRE["full"] = list(set(
    dep for key, deps in EXTRAS_REQUIRE.items()
    if key not in ["all", "full", "research"]
    for dep in deps
))

EXTRAS_REQUIRE["all"] = EXTRAS_REQUIRE["full"]


# =============================================================================
# ENTRY POINTS
# =============================================================================

ENTRY_POINTS = {
    "console_scripts": [
        "crypto-collect=data_collection.run_collection:main",
        "crypto-backtest=backtesting.backtest_engine:main",
        "crypto-validate=data_collection.utils.data_validator:main",
    ],
}


# =============================================================================
# CUSTOM COMMANDS
# =============================================================================

class PyTestCommand(TestCommand):
    """Custom pytest command."""
    
    description = "Run pytest test suite"
    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]
    
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ""
    
    def finalize_options(self):
        TestCommand.finalize_options(self)
    
    def run_tests(self):
        import shlex
        try:
            import pytest
        except ImportError:
            print("pytest required: pip install pytest")
            sys.exit(1)
        
        args = shlex.split(self.pytest_args) if self.pytest_args else []
        args.insert(0, "tests/")
        sys.exit(pytest.main(args))


class CleanCommand(Command):
    """Clean build artifacts."""
    
    description = "Remove build artifacts"
    user_options = []
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        patterns = [
            "build", "dist", "*.egg-info", ".eggs",
            ".pytest_cache", ".mypy_cache", ".coverage",
            "htmlcov", "__pycache__", "*.pyc", "*.pyo",
        ]
        
        removed = 0
        for pattern in patterns:
            for path in THIS_DIRECTORY.rglob(pattern):
                try:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                    removed += 1
                except Exception:
                    pass
        
        print(f"Removed {removed} items")


class LintCommand(Command):
    """Run linting checks."""
    
    description = "Run linting (flake8, black, isort)"
    user_options = []
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        commands = [
            ["black", "--check", "data_collection/", "strategies/"],
            ["isort", "--check-only", "data_collection/", "strategies/"],
            ["flake8", "data_collection/", "strategies/"],
        ]
        
        failed = False
        for cmd in commands:
            print(f"\nRunning: {' '.join(cmd)}")
            if subprocess.run(cmd).returncode != 0:
                failed = True
        
        sys.exit(1 if failed else 0)


class TypeCheckCommand(Command):
    """Run mypy type checking."""
    
    description = "Run mypy type checking"
    user_options = []
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        cmd = ["mypy", "--ignore-missing-imports", "data_collection/"]
        print(f"Running: {' '.join(cmd)}")
        sys.exit(subprocess.run(cmd).returncode)


# =============================================================================
# POST-INSTALL HOOKS
# =============================================================================

class PostInstallCommand(InstallCommand):
    """Post-installation hook."""
    
    def run(self):
        InstallCommand.run(self)
        print_post_install()


class PostDevelopCommand(DevelopCommand):
    """Post-develop hook."""
    
    def run(self):
        DevelopCommand.run(self)
        print_post_install()


def print_post_install():
    """Print post-installation message."""
    print("""
================================================================================
  Crypto Statistical Arbitrage - Multi-Venue Trading System v2.0.0
================================================================================

Quick Start:
  1. cp config/api_keys_template.env config/.env
  2. Edit config/.env with your API keys
  3. crypto-collect --venue binance --days 30
  4. crypto-backtest --strategy pairs_trading

Documentation: https://github.com/abailey81/Crypto-Statistical-Arbitrage
================================================================================
""")


# =============================================================================
# PROJECT URLS
# =============================================================================

PROJECT_URLS = {
    "Homepage": URL,
    "Documentation": f"{URL}#readme",
    "Repository": URL,
    "Bug Tracker": f"{URL}/issues",
}


# =============================================================================
# SETUP
# =============================================================================

setup(
    name=PACKAGE_NAME,
    version=read_version(),
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url=URL,
    project_urls=PROJECT_URLS,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "docs"]),
    package_dir={"": "."},
    include_package_data=True,
    package_data={"": ["*.yaml", "*.yml", "*.json", "*.md", "py.typed"]},
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points=ENTRY_POINTS,
    cmdclass={
        "test": PyTestCommand,
        "clean": CleanCommand,
        "lint": LintCommand,
        "typecheck": TypeCheckCommand,
        "install": PostInstallCommand,
        "develop": PostDevelopCommand,
    },
    zip_safe=False,
)