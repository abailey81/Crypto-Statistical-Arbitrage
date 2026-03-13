"""
Integration Tests Package
=========================

Comprehensive end-to-end integration tests for the crypto statistical
arbitrage data collection and strategy pipeline.

Test Coverage Overview
----------------------
This package provides integration tests that verify the complete data
pipeline from collection through validation, storage, and analysis.

Test Modules
------------
test_collection_manager.py
    Tests for the CollectionManager and CollectorRegistry:
    - Collector configuration completeness (47+ collectors)
    - Credential handling and validation
    - Collector instantiation
    - Multi-venue coordination
    - Task orchestration
    - Progress tracking

test_pipeline.py
    End-to-end pipeline tests:
    - Mock data generation validation
    - Data validation pipeline
    - Multi-venue collection simulation
    - Storage round-trip integrity
    - Cross-venue consistency
    - Data quality scenarios
    - Checkpoint/resume functionality
    - Performance benchmarks

Test Categories
---------------
UNIT Tests:
    - Individual component validation
    - Configuration completeness
    - Field validation

INTEGRATION Tests:
    - Multi-component workflows
    - Data flow verification
    - Cross-venue consistency

PERFORMANCE Tests:
    - Large dataset handling
    - Storage I/O benchmarks
    - Query performance

STRESS Tests:
    - Error handling
    - Recovery scenarios
    - Edge cases

Collector Coverage
------------------
The test suite covers all 47+ data collectors:

CEX (6):
    binance, bybit, okx, coinbase, kraken, cme

Hybrid (2):
    hyperliquid, dydx

DEX (11):
    uniswap, geckoterminal, dexscreener, oneinch, zerox,
    gmx, vertex, jupiter, cowswap, curve, sushiswap

Options (4):
    deribit, aevo, lyra, dopex

Market Data (4):
    cryptocompare, coingecko, messari, kaiko

On-Chain (10):
    covalent, bitquery, santiment, cryptoquant, whale_alert,
    arkham, nansen, coinmetrics, glassnode, flipside

Indexers (1):
    thegraph

Social (1):
    twitter

Analytics (2):
    messari_analytics, defillama

Alternative (4):
    coinalyze, coinalyze_enhanced, dune, lunarcrush

Data Quality Scenarios
----------------------
Tests cover various data quality scenarios:

1. Missing Data:
   - Gap detection
   - Completeness measurement
   - Interpolation validation

2. Outliers:
   - Extreme value detection
   - Statistical bounds checking
   - Automated flagging

3. Duplicates:
   - Duplicate record detection
   - Deduplication validation

4. Consistency:
   - Cross-venue price alignment
   - Funding rate correlation
   - Timestamp synchronization

5. Survivorship Bias:
   - Late listing detection
   - Early delisting detection
   - Coverage tracking

Running Tests
-------------
Run all integration tests:

    pytest tests/integration/ -v

Run specific test module:

    pytest tests/integration/test_collection_manager.py -v
    pytest tests/integration/test_pipeline.py -v

Run with coverage:

    pytest tests/integration/ --cov=data_collection --cov-report=html

Run specific test class:

    pytest tests/integration/test_collection_manager.py::TestCollectorRegistry -v

Run with markers:

    pytest tests/integration/ -m "not slow" -v

Fixtures Available
------------------
From conftest.py and test modules:

registry:
    Fresh CollectorRegistry instance

mock_all_credentials:
    Environment with all mock credentials set

mock_collector_class:
    Generic mock collector for instantiation tests

temp_workspace:
    Temporary directory with standard structure

collection_config:
    Sample collection configuration dict

pipeline_config:
    Full pipeline configuration with YAML file

mock_config:
    MockDataConfig instance for data generation

Test Utilities
--------------
MockDataValidator:
    Fallback validator when real one unavailable

MockQualityAnalyzer:
    Fallback analyzer for quality checks

MockStorage:
    In-memory storage for testing

PipelineTestResult:
    Structured test result with metrics

Expected Test Results
---------------------
All tests should pass with:

- 47+ collectors configured
- Free collectors always available
- Credential mapping correct
- Data validation passing
- Storage round-trip intact
- Cross-venue correlation > 0.5
- Performance within bounds

Minimum Expectations:
- CEX collectors: 6+
- DEX collectors: 10+
- Hybrid collectors: 2+
- Free collectors: 13+

Performance Bounds:
- Large dataset generation: < 60s
- Storage write: < 10s
- Storage read: < 5s
- Query with filters: < 3s

Author: Crypto StatArb Quantitative Research
Version: 2.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# TEST ORGANIZATION ENUMS
# =============================================================================

class TestModule(Enum):
    """Available test modules."""
    COLLECTION_MANAGER = "test_collection_manager"
    PIPELINE = "test_pipeline"
    
    @property
    def description(self) -> str:
        """Module description."""
        descriptions = {
            self.COLLECTION_MANAGER: "CollectionManager and Registry tests",
            self.PIPELINE: "End-to-end pipeline integration tests",
        }
        return descriptions.get(self, "")
    
    @property
    def test_count_estimate(self) -> int:
        """Estimated number of tests in module."""
        counts = {
            self.COLLECTION_MANAGER: 35,
            self.PIPELINE: 40,
        }
        return counts.get(self, 0)


class TestPriority(Enum):
    """Test priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    
    @property
    def description(self) -> str:
        """Priority description."""
        descriptions = {
            self.CRITICAL: "Must pass for any functionality",
            self.HIGH: "Core functionality tests",
            self.MEDIUM: "Important but not blocking",
            self.LOW: "Nice to have coverage",
        }
        return descriptions.get(self, "")


class CollectorCategory(Enum):
    """Collector categories for organized testing."""
    CEX = "cex"
    HYBRID = "hybrid"
    DEX = "dex"
    OPTIONS = "options"
    MARKET_DATA = "market_data"
    ON_CHAIN = "on_chain"
    INDEXER = "indexer"
    SOCIAL = "social"
    ANALYTICS = "analytics"
    ALTERNATIVE = "alternative"
    
    @property
    def expected_count(self) -> int:
        """Expected collectors in category."""
        counts = {
            self.CEX: 6,
            self.HYBRID: 2,
            self.DEX: 11,
            self.OPTIONS: 4,
            self.MARKET_DATA: 4,
            self.ON_CHAIN: 10,
            self.INDEXER: 1,
            self.SOCIAL: 1,
            self.ANALYTICS: 2,
            self.ALTERNATIVE: 4,
        }
        return counts.get(self, 0)
    
    @property
    def collectors(self) -> List[str]:
        """List of collector names."""
        collector_map = {
            self.CEX: ['binance', 'bybit', 'okx', 'coinbase', 'kraken', 'cme'],
            self.HYBRID: ['hyperliquid', 'dydx'],
            self.DEX: [
                'uniswap', 'geckoterminal', 'dexscreener', 'oneinch', 'zerox',
                'gmx', 'vertex', 'jupiter', 'cowswap', 'curve', 'sushiswap'
            ],
            self.OPTIONS: ['deribit', 'aevo', 'lyra', 'dopex'],
            self.MARKET_DATA: ['cryptocompare', 'coingecko', 'messari', 'kaiko'],
            self.ON_CHAIN: [
                'covalent', 'bitquery', 'santiment', 'cryptoquant', 'whale_alert',
                'arkham', 'nansen', 'coinmetrics', 'glassnode', 'flipside'
            ],
            self.INDEXER: ['thegraph'],
            self.SOCIAL: ['twitter'],
            self.ANALYTICS: ['messari_analytics', 'defillama'],
            self.ALTERNATIVE: ['coinalyze', 'coinalyze_enhanced', 'dune', 'lunarcrush'],
        }
        return collector_map.get(self, [])


# =============================================================================
# TEST SUITE CONFIGURATION
# =============================================================================

@dataclass
class TestSuiteConfig:
    """Configuration for test suite execution."""
    # Test selection
    modules: List[TestModule] = None
    categories: List[CollectorCategory] = None
    min_priority: TestPriority = TestPriority.LOW
    
    # Execution options
    parallel: bool = False
    max_workers: int = 4
    timeout_seconds: int = 300
    
    # Reporting
    verbose: bool = True
    generate_report: bool = True
    report_format: str = "html"
    
    def __post_init__(self):
        if self.modules is None:
            self.modules = list(TestModule)
        if self.categories is None:
            self.categories = list(CollectorCategory)


# =============================================================================
# TEST IMPORTS
# =============================================================================

# Import test classes for discovery
try:
    from .test_collection_manager import (
        TestCollectorRegistry,
        TestVenueTypeMapping,
        TestCredentialHandling,
        TestCollectorInstantiation,
        TestListAvailableCollectors,
        TestCollectionManager,
        TestCollectionProgress,
        TestErrorHandling,
        TestAsyncOperations,
        TestParametrizedCollectors,
        EXPECTED_COLLECTORS,
        MOCK_CREDENTIALS,
    )
    COLLECTION_MANAGER_TESTS_AVAILABLE = True
except ImportError as e:
    COLLECTION_MANAGER_TESTS_AVAILABLE = False
    TestCollectorRegistry = None

try:
    from .test_pipeline import (
        TestFullPipeline,
        TestCrossVenueConsistency,
        TestDataQualityScenarios,
        TestCheckpointResume,
        TestPipelinePerformance,
        TestConfigurationLoading,
        TestOptionsData,
        TestDEXPoolData,
        PipelineTestResult,
        TestCategory,
        DataQualityMetric,
    )
    PIPELINE_TESTS_AVAILABLE = True
except ImportError as e:
    PIPELINE_TESTS_AVAILABLE = False
    TestFullPipeline = None


# =============================================================================
# PACKAGE METADATA
# =============================================================================

__version__ = "2.0.0"
__author__ = "Crypto StatArb Quantitative Research"

# Build __all__ based on available imports
__all__ = [
    # Version info
    '__version__',
    '__author__',
    
    # Enums
    'TestModule',
    'TestPriority',
    'CollectorCategory',
    
    # Configuration
    'TestSuiteConfig',
    
    # Availability flags
    'COLLECTION_MANAGER_TESTS_AVAILABLE',
    'PIPELINE_TESTS_AVAILABLE',
]

# Add collection manager tests if available
if COLLECTION_MANAGER_TESTS_AVAILABLE:
    __all__.extend([
        'TestCollectorRegistry',
        'TestVenueTypeMapping',
        'TestCredentialHandling',
        'TestCollectorInstantiation',
        'TestListAvailableCollectors',
        'TestCollectionManager',
        'TestCollectionProgress',
        'TestErrorHandling',
        'TestAsyncOperations',
        'TestParametrizedCollectors',
        'EXPECTED_COLLECTORS',
        'MOCK_CREDENTIALS',
    ])

# Add pipeline tests if available
if PIPELINE_TESTS_AVAILABLE:
    __all__.extend([
        'TestFullPipeline',
        'TestCrossVenueConsistency',
        'TestDataQualityScenarios',
        'TestCheckpointResume',
        'TestPipelinePerformance',
        'TestConfigurationLoading',
        'TestOptionsData',
        'TestDEXPoolData',
        'PipelineTestResult',
        'TestCategory',
        'DataQualityMetric',
    ])


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_all_collectors() -> List[str]:
    """Get list of all expected collectors across categories."""
    collectors = []
    for category in CollectorCategory:
        collectors.extend(category.collectors)
    return collectors


def get_free_collectors() -> List[str]:
    """Get list of collectors that don't require API keys."""
    return [
        'hyperliquid', 'dydx',
        'geckoterminal', 'dexscreener',
        'oneinch', 'zerox',
        'vertex', 'jupiter', 'cowswap',
        'curve', 'sushiswap',
        'gmx', 'defillama',
    ]


def get_test_summary() -> Dict[str, Any]:
    """Get summary of available tests."""
    return {
        'modules': {
            'collection_manager': COLLECTION_MANAGER_TESTS_AVAILABLE,
            'pipeline': PIPELINE_TESTS_AVAILABLE,
        },
        'total_collectors': len(get_all_collectors()),
        'free_collectors': len(get_free_collectors()),
        'categories': len(CollectorCategory),
        'estimated_tests': sum(m.test_count_estimate for m in TestModule),
    }


def run_quick_validation() -> bool:
    """Run quick validation that test infrastructure is working."""
    try:
        # Check imports
        from tests.fixtures import quick_funding_data
        
        # Generate small test dataset
        df = quick_funding_data(n_days=1)
        
        if len(df) == 0:
            return False
        
        return True
    except Exception as e:
        print(f"Quick validation failed: {e}")
        return False


# Add utility functions to exports
__all__.extend([
    'get_all_collectors',
    'get_free_collectors',
    'get_test_summary',
    'run_quick_validation',
])