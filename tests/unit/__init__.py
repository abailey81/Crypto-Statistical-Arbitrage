"""
Unit Tests Package
==================

Comprehensive unit tests for individual components of the
crypto statistical arbitrage data collection pipeline.

Package Overview
----------------
This package provides isolated unit tests for all pipeline components
with mocked external dependencies for deterministic testing.

Test Modules
------------
test_collectors.py
    Unit tests for data collectors:
    - Base collector abstract class
    - CEX collectors (Binance, Bybit, OKX)
    - Hybrid collectors (Hyperliquid, dYdX)
    - DEX collectors (Uniswap, Curve)
    - Options collectors (Deribit)
    - Collection manager orchestration

test_utils.py
    Unit tests for utilities:
    - Token bucket rate limiter
    - Venue rate limiter manager
    - Retry handler with backoff
    - Parquet storage operations
    - Optimized storage queries
    - Data validation
    - Quality analysis
    - Funding normalization

Test Coverage Areas
-------------------
Collectors:
    - Initialization and configuration
    - Funding rate fetching
    - OHLCV data fetching
    - Rate limit handling
    - Error recovery (network, auth, timeout)
    - Multi-symbol handling
    - Venue-specific formatting

Utilities:
    - Rate limiter token management
    - Burst handling
    - Exponential backoff
    - Jitter randomization
    - Storage read/write
    - Partitioning strategies
    - Compression options
    - Query filtering
    - Data validation rules
    - Quality metrics

Mock Data Strategy
------------------
All external API calls are mocked to ensure:
- Deterministic test results
- Fast execution (no network I/O)
- No external dependencies
- Comprehensive edge case coverage
- Reproducible failures

Example Mock Patterns:

    # Mock CCXT exchange
    collector.exchange = AsyncMock()
    collector.exchange.fetch_funding_rate_history = AsyncMock(
        return_value=mock_data
    )

    # Mock aiohttp session
    mock_session = AsyncMock()
    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(return_value=data)

    # Mock rate limiting
    with patch('asyncio.sleep', new_callable=AsyncMock):
        result = await collector.fetch_data()

Test Organization
-----------------
Tests are organized by component:

1. **TestBaseCollector**: Abstract class enforcement
2. **TestBinanceCollector**: CEX reference implementation
3. **TestHyperliquidCollector**: Hybrid venue (hourly funding)
4. **TestUniswapV3Collector**: DEX pool data
5. **TestDeribitCollector**: Options instruments
6. **TestTokenBucketRateLimiter**: Rate limiting
7. **TestRetryHandler**: Error recovery
8. **TestParquetStorage**: Data persistence
9. **TestDataValidator**: Data quality

Running Tests
-------------
All unit tests:

    pytest tests/unit/ -v

Specific module:

    pytest tests/unit/test_collectors.py -v
    pytest tests/unit/test_utils.py -v

Specific class:

    pytest tests/unit/test_collectors.py::TestBinanceCollector -v

With coverage:

    pytest tests/unit/ --cov=data_collection --cov-report=html

Parametrized tests:

    pytest tests/unit/ -k "parametrized" -v

Module Structure
----------------
test_collectors.py
├── Enumerations
│   ├── CollectorType         - Collector categories
│   └── MockResponseType      - Response scenarios
│
├── Mock Data Generators
│   ├── MockFundingResponse   - Funding rate mocks
│   ├── MockOHLCVResponse     - OHLCV mocks
│   ├── MockPoolResponse      - DEX pool mocks
│   └── MockOptionsResponse   - Options mocks
│
└── Test Classes
    ├── TestBaseCollector
    ├── TestBinanceCollector
    ├── TestBybitCollector
    ├── TestHyperliquidCollector
    ├── TestDydxCollector
    ├── TestUniswapV3Collector
    ├── TestDeribitCollector
    ├── TestCollectionManagerUnit
    ├── TestFundingNormalization
    ├── TestCollectorErrorHandling
    └── TestParametrizedCollectors

test_utils.py
├── Enumerations
│   ├── ValidationSeverity    - Issue severity
│   └── StorageFormat         - Storage formats
│
└── Test Classes
    ├── TestTokenBucketRateLimiter
    ├── TestVenueRateLimiterManager
    ├── TestRetryHandler
    ├── TestParquetStorage
    ├── TestOptimizedStorage
    ├── TestDataValidator
    ├── TestQualityAnalyzer
    ├── TestFundingNormalizationUtils
    └── TestStorageValidatorIntegration

Author: Crypto StatArb Quantitative Research
Version: 2.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# IMPORTS FROM TEST MODULES
# =============================================================================

# Import collector tests
try:
    from .test_collectors import (
        # Enums
        CollectorType,
        MockResponseType,
        
        # Mock data generators
        MockFundingResponse,
        MockOHLCVResponse,
        MockPoolResponse,
        MockOptionsResponse,
        
        # Test classes
        TestBaseCollector,
        TestBinanceCollector,
        TestBybitCollector,
        TestHyperliquidCollector,
        TestDydxCollector,
        TestUniswapV3Collector,
        TestDeribitCollector,
        TestCollectionManagerUnit,
        TestFundingNormalization,
        TestCollectorErrorHandling,
        TestParametrizedCollectors,
    )
    COLLECTOR_TESTS_AVAILABLE = True
except ImportError as e:
    COLLECTOR_TESTS_AVAILABLE = False
    print(f"Warning: Collector tests not available: {e}")

# Import utility tests
try:
    from .test_utils import (
        # Enums
        ValidationSeverity,
        StorageFormat,
        
        # Test classes
        TestTokenBucketRateLimiter,
        TestVenueRateLimiterManager,
        TestRetryHandler,
        TestParquetStorage,
        TestOptimizedStorage,
        TestDataValidator,
        TestQualityAnalyzer,
        TestFundingNormalizationUtils,
        TestStorageValidatorIntegration,
    )
    UTILS_TESTS_AVAILABLE = True
except ImportError as e:
    UTILS_TESTS_AVAILABLE = False
    print(f"Warning: Utility tests not available: {e}")


# =============================================================================
# PACKAGE METADATA
# =============================================================================

__version__ = "2.0.0"
__author__ = "Crypto StatArb Quantitative Research"

# Build exports list
__all__ = [
    '__version__',
    '__author__',
    'COLLECTOR_TESTS_AVAILABLE',
    'UTILS_TESTS_AVAILABLE',
]

# Add collector test exports if available
if COLLECTOR_TESTS_AVAILABLE:
    __all__.extend([
        # Enums
        'CollectorType',
        'MockResponseType',
        
        # Mock generators
        'MockFundingResponse',
        'MockOHLCVResponse',
        'MockPoolResponse',
        'MockOptionsResponse',
        
        # Test classes
        'TestBaseCollector',
        'TestBinanceCollector',
        'TestBybitCollector',
        'TestHyperliquidCollector',
        'TestDydxCollector',
        'TestUniswapV3Collector',
        'TestDeribitCollector',
        'TestCollectionManagerUnit',
        'TestFundingNormalization',
        'TestCollectorErrorHandling',
        'TestParametrizedCollectors',
    ])

# Add utility test exports if available
if UTILS_TESTS_AVAILABLE:
    __all__.extend([
        # Enums
        'ValidationSeverity',
        'StorageFormat',
        
        # Test classes
        'TestTokenBucketRateLimiter',
        'TestVenueRateLimiterManager',
        'TestRetryHandler',
        'TestParquetStorage',
        'TestOptimizedStorage',
        'TestDataValidator',
        'TestQualityAnalyzer',
        'TestFundingNormalizationUtils',
        'TestStorageValidatorIntegration',
    ])


# =============================================================================
# PACKAGE-LEVEL UTILITIES
# =============================================================================

def get_test_summary() -> Dict[str, Any]:
    """Get summary of available unit tests."""
    collector_tests = [
        'TestBaseCollector',
        'TestBinanceCollector',
        'TestBybitCollector',
        'TestHyperliquidCollector',
        'TestDydxCollector',
        'TestUniswapV3Collector',
        'TestDeribitCollector',
        'TestCollectionManagerUnit',
        'TestFundingNormalization',
        'TestCollectorErrorHandling',
        'TestParametrizedCollectors',
    ]
    
    utility_tests = [
        'TestTokenBucketRateLimiter',
        'TestVenueRateLimiterManager',
        'TestRetryHandler',
        'TestParquetStorage',
        'TestOptimizedStorage',
        'TestDataValidator',
        'TestQualityAnalyzer',
        'TestFundingNormalizationUtils',
        'TestStorageValidatorIntegration',
    ]
    
    return {
        'collector_tests': {
            'available': COLLECTOR_TESTS_AVAILABLE,
            'count': len(collector_tests),
            'classes': collector_tests,
        },
        'utility_tests': {
            'available': UTILS_TESTS_AVAILABLE,
            'count': len(utility_tests),
            'classes': utility_tests,
        },
        'total_test_classes': len(collector_tests) + len(utility_tests),
    }


def list_test_classes() -> List[str]:
    """List all available test classes."""
    classes = []
    
    if COLLECTOR_TESTS_AVAILABLE:
        classes.extend([
            'TestBaseCollector',
            'TestBinanceCollector',
            'TestBybitCollector',
            'TestHyperliquidCollector',
            'TestDydxCollector',
            'TestUniswapV3Collector',
            'TestDeribitCollector',
            'TestCollectionManagerUnit',
            'TestFundingNormalization',
            'TestCollectorErrorHandling',
            'TestParametrizedCollectors',
        ])
    
    if UTILS_TESTS_AVAILABLE:
        classes.extend([
            'TestTokenBucketRateLimiter',
            'TestVenueRateLimiterManager',
            'TestRetryHandler',
            'TestParquetStorage',
            'TestOptimizedStorage',
            'TestDataValidator',
            'TestQualityAnalyzer',
            'TestFundingNormalizationUtils',
            'TestStorageValidatorIntegration',
        ])
    
    return classes


def get_coverage_areas() -> Dict[str, List[str]]:
    """Get test coverage areas by component."""
    return {
        'collectors': [
            'Initialization and configuration',
            'Funding rate fetching',
            'OHLCV data fetching',
            'Rate limit handling',
            'Error recovery (network, auth, timeout)',
            'Multi-symbol handling',
            'Venue-specific formatting',
            'Cross-venue normalization',
        ],
        'rate_limiter': [
            'Token bucket algorithm',
            'Burst handling',
            'Rate limiting enforcement',
            'Multi-venue management',
            'Concurrent acquisition',
        ],
        'retry_handler': [
            'Exponential backoff',
            'Jitter randomization',
            'Max retries enforcement',
            'Exception filtering',
        ],
        'storage': [
            'Parquet read/write',
            'Partitioning strategies',
            'Compression options',
            'Query filtering',
            'Cache functionality',
        ],
        'validation': [
            'Schema validation',
            'Value range checks',
            'OHLC relationship checks',
            'Cross-venue consistency',
            'Outlier detection',
            'Quality scoring',
        ],
    }


# Add utilities to exports
__all__.extend([
    'get_test_summary',
    'list_test_classes',
    'get_coverage_areas',
])