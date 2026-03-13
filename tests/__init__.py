"""
Crypto Statistical Arbitrage Test Suite
=======================================

Comprehensive testing infrastructure for the crypto statistical arbitrage
data collection and strategy pipeline.

Package Structure
-----------------
tests/
├── __init__.py              - This file (package initialization)
├── conftest.py              - Pytest configuration and shared fixtures
├── fixtures/                - Mock data generators
│   ├── __init__.py
│   └── mock_data.py         - Synthetic data generation
├── integration/             - Integration tests
│   ├── __init__.py
│   ├── test_collection_manager.py
│   └── test_pipeline.py
├── unit/                    - Unit tests (future)
│   └── __init__.py
└── performance/             - Performance benchmarks (future)
    └── __init__.py

Test Categories
---------------
1. **Unit Tests** (tests/unit/):
   - Individual component validation
   - Function-level testing
   - Isolated behavior verification

2. **Integration Tests** (tests/integration/):
   - Multi-component workflows
   - End-to-end pipeline testing
   - Cross-venue consistency

3. **Performance Tests** (tests/performance/):
   - Large dataset handling
   - I/O benchmarks
   - Query performance

4. **Fixtures** (tests/fixtures/):
   - Mock data generators
   - Test utilities
   - Synthetic dataset creation

Coverage Areas
--------------
Data Collection (47+ collectors):
- CEX: Binance, Bybit, OKX, Coinbase, Kraken, CME
- Hybrid: Hyperliquid, dYdX
- DEX: Uniswap, Curve, GMX, etc.
- Options: Deribit, Aevo, Lyra, Dopex
- On-Chain: Nansen, Glassnode, Arkham, etc.

Data Validation:
- Schema validation
- Value range checks
- Cross-venue consistency
- Quality metrics

Storage:
- Parquet I/O
- Partitioning strategies
- Query optimization

Strategies:
- Pairs trading
- Funding rate arbitrage
- Futures curve trading

Running Tests
-------------
All tests:

    pytest tests/ -v

Integration tests only:

    pytest tests/integration/ -v

With coverage:

    pytest tests/ --cov=. --cov-report=html

Specific markers:

    pytest tests/ -m "not slow" -v

Quick validation:

    python -c "from tests import run_quick_check; run_quick_check()"

Test Configuration
------------------
Configure via pytest.ini or pyproject.toml:

    [tool.pytest.ini_options]
    testpaths = ["tests"]
    python_files = ["test_*.py"]
    python_classes = ["Test*"]
    python_functions = ["test_*"]
    markers = [
        "slow: marks tests as slow",
        "integration: integration tests",
        "unit: unit tests",
    ]

Environment Variables
---------------------
For credential tests, set mock credentials:

    export BINANCE_API_KEY=test_key
    export BINANCE_SECRET_KEY=test_secret
    ...

Or use the provided fixtures that mock credentials.

Author: Crypto StatArb Quantitative Research
Version: 2.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure project paths are available
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# SUBPACKAGE IMPORTS
# =============================================================================

# Import fixtures
try:
    from .fixtures import (
        # Enums
        MarketRegime,
        VenueType,
        DataQualityIssue,
        AssetClass,
        TimeframeType,
        
        # Configuration
        MockDataConfig,
        GeneratedDataStats,
        
        # Generators
        MockFundingRateGenerator,
        MockOHLCVGenerator,
        MockOptionsDataGenerator,
        MockDEXPoolGenerator,
        MockOnChainGenerator,
        
        # Factory and convenience
        create_mock_dataset,
        quick_funding_data,
        quick_ohlcv_data,
        quick_options_data,
        quick_pool_data,
        
        # Constants
        ASSET_PARAMETERS,
        VENUE_CONFIGS,
        
        # Utilities
        get_available_symbols,
        get_available_venues,
        list_data_types,
    )
    FIXTURES_AVAILABLE = True
except ImportError as e:
    FIXTURES_AVAILABLE = False
    print(f"Warning: fixtures not available: {e}")

# Import integration tests info
try:
    from .integration import (
        TestModule,
        TestPriority,
        CollectorCategory,
        TestSuiteConfig,
        get_all_collectors,
        get_free_collectors,
        get_test_summary,
        COLLECTION_MANAGER_TESTS_AVAILABLE,
        PIPELINE_TESTS_AVAILABLE,
    )
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    INTEGRATION_AVAILABLE = False
    print(f"Warning: integration tests not available: {e}")


# =============================================================================
# PACKAGE METADATA
# =============================================================================

__version__ = "2.0.0"
__author__ = "Crypto StatArb Quantitative Research"
__description__ = "Comprehensive test suite for crypto statistical arbitrage"


# =============================================================================
# TEST DISCOVERY AND EXECUTION UTILITIES
# =============================================================================

def get_test_status() -> Dict[str, Any]:
    """
    Get status of test infrastructure.
    
    Returns:
        Dictionary with availability status of test components
    """
    return {
        'fixtures_available': FIXTURES_AVAILABLE,
        'integration_available': INTEGRATION_AVAILABLE,
        'project_root': str(PROJECT_ROOT),
        'python_path_configured': str(PROJECT_ROOT) in sys.path,
    }


def run_quick_check() -> bool:
    """
    Run quick validation that test infrastructure is working.
    
    Returns:
        True if all checks pass
    """
    print("Running quick test infrastructure check...")
    
    checks_passed = True
    
    # Check fixtures
    if FIXTURES_AVAILABLE:
        try:
            df = quick_funding_data(n_days=1)
            if len(df) > 0:
                print("  + Fixtures: Mock data generation working")
            else:
                print("  x Fixtures: Mock data generation returned empty")
                checks_passed = False
        except Exception as e:
            print(f"  x Fixtures: Error - {e}")
            checks_passed = False
    else:
        print("  x Fixtures: Not available")
        checks_passed = False
    
    # Check integration test discovery
    if INTEGRATION_AVAILABLE:
        try:
            summary = get_test_summary()
            print(f"  + Integration: {summary['estimated_tests']} tests estimated")
            print(f"    - Collectors: {summary['total_collectors']}")
            print(f"    - Free collectors: {summary['free_collectors']}")
        except Exception as e:
            print(f"  x Integration: Error - {e}")
            checks_passed = False
    else:
        print("  x Integration: Not available")
        checks_passed = False
    
    if checks_passed:
        print("\n+ All checks passed!")
    else:
        print("\nx Some checks failed")
    
    return checks_passed


def list_test_modules() -> List[str]:
    """List available test modules."""
    modules = []
    
    test_dir = Path(__file__).parent
    
    # Check integration tests
    integration_dir = test_dir / 'integration'
    if integration_dir.exists():
        for f in integration_dir.glob('test_*.py'):
            modules.append(f'integration/{f.stem}')
    
    # Check unit tests
    unit_dir = test_dir / 'unit'
    if unit_dir.exists():
        for f in unit_dir.glob('test_*.py'):
            modules.append(f'unit/{f.stem}')
    
    return modules


def get_fixture_info() -> Dict[str, Any]:
    """Get information about available fixtures."""
    if not FIXTURES_AVAILABLE:
        return {'available': False}
    
    return {
        'available': True,
        'data_types': list_data_types(),
        'symbols': get_available_symbols(),
        'venues': get_available_venues(),
        'quality_issues': [qi.value for qi in DataQualityIssue],
        'market_regimes': [mr.value for mr in MarketRegime],
        'timeframes': [tf.value for tf in TimeframeType],
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Version
    '__version__',
    '__author__',
    '__description__',
    
    # Status
    'FIXTURES_AVAILABLE',
    'INTEGRATION_AVAILABLE',
    
    # Utilities
    'get_test_status',
    'run_quick_check',
    'list_test_modules',
    'get_fixture_info',
]

# Add fixture exports if available
if FIXTURES_AVAILABLE:
    __all__.extend([
        # Enums
        'MarketRegime',
        'VenueType',
        'DataQualityIssue',
        'AssetClass',
        'TimeframeType',
        
        # Configuration
        'MockDataConfig',
        'GeneratedDataStats',
        
        # Generators
        'MockFundingRateGenerator',
        'MockOHLCVGenerator',
        'MockOptionsDataGenerator',
        'MockDEXPoolGenerator',
        'MockOnChainGenerator',
        
        # Factory and convenience
        'create_mock_dataset',
        'quick_funding_data',
        'quick_ohlcv_data',
        'quick_options_data',
        'quick_pool_data',
        
        # Constants
        'ASSET_PARAMETERS',
        'VENUE_CONFIGS',
    ])

# Add integration exports if available
if INTEGRATION_AVAILABLE:
    __all__.extend([
        'TestModule',
        'TestPriority',
        'CollectorCategory',
        'TestSuiteConfig',
        'get_all_collectors',
        'get_free_collectors',
        'get_test_summary',
    ])