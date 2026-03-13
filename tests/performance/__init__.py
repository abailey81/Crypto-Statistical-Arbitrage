"""
Performance Tests Package
=========================

Comprehensive performance benchmarks and stress tests for the
crypto statistical arbitrage data collection pipeline.

Package Overview
----------------
This package provides performance testing infrastructure for measuring
throughput, latency, memory usage, and scalability of all pipeline
components.

Benchmark Categories
--------------------
1. **Data Generation** (BenchmarkCategory.DATA_GENERATION):
   - Mock data generation throughput
   - Scaling characteristics
   - Memory efficiency

2. **Storage Write** (BenchmarkCategory.STORAGE_WRITE):
   - Parquet write performance
   - Compression benchmarks
   - Partitioning overhead

3. **Storage Read** (BenchmarkCategory.STORAGE_READ):
   - Parquet read performance
   - Memory-mapped I/O

4. **Storage Query** (BenchmarkCategory.STORAGE_QUERY):
   - Filtered query performance
   - Partition pruning efficiency
   - Cache hit rates

5. **Validation** (BenchmarkCategory.VALIDATION):
   - Funding rate validation throughput
   - OHLCV validation throughput
   - Cross-venue consistency checks

6. **Rate Limiter** (BenchmarkCategory.RATE_LIMITER):
   - Token bucket overhead
   - Concurrent acquisition

7. **Memory** (BenchmarkCategory.MEMORY):
   - Peak memory usage
   - Memory efficiency ratios
   - GC impact

8. **End-to-End** (BenchmarkCategory.END_TO_END):
   - Full pipeline benchmarks
   - Multi-stage workflows

Performance Targets
-------------------
Data Generation:
    - Small (<1K rows): < 0.5s
    - Medium (~10K rows): < 5s
    - Large (~100K rows): < 30s

Storage Operations:
    - Write: > 100,000 rows/sec
    - Read: > 200,000 rows/sec
    - Query: > 500,000 rows/sec

Validation:
    - Funding: > 50,000 rows/sec
    - OHLCV: > 30,000 rows/sec

Memory:
    - Peak per 100K rows: < 100 MB
    - Bytes per row: < 500 bytes

Metrics Captured
----------------
Timing:
    - Total time
    - Average time
    - Min/Max time
    - Standard deviation
    - Percentiles (p50, p90, p95, p99)
    - Coefficient of variation

Throughput:
    - Items per second
    - MB per second
    - Bytes per item

Memory:
    - Peak memory (MB)
    - Average memory (MB)
    - Memory per 1K items

Usage Examples
--------------
Run All Benchmarks:

    pytest tests/performance/ -v

Run Specific Category:

    pytest tests/performance/benchmarks.py::TestStorageBenchmarks -v

Run with Size Filter:

    pytest tests/performance/ -k "not xlarge" -v

Generate Report:

    from tests.performance import BenchmarkRunner
    
    runner = BenchmarkRunner(warmup=2, iterations=5)
    result = runner.run("my_benchmark", my_function, ...)
    print(runner.report())

Custom Benchmark:

    from tests.performance import (
        BenchmarkRunner, 
        BenchmarkCategory,
        BenchmarkSize
    )
    
    runner = BenchmarkRunner()
    result = runner.run(
        "custom_benchmark",
        my_function,
        arg1, arg2,
        category=BenchmarkCategory.DATA_GENERATION,
        size=BenchmarkSize.MEDIUM,
        items_per_call=10000
    )
    
    print(f"Throughput: {result.throughput_items_per_sec:,.0f} items/sec")
    print(f"Peak memory: {result.memory_peak_mb:.1f} MB")

Scaling Test:

    results = runner.run_scaling_test(
        "scaling_test",
        lambda n: lambda: generate_data(n),
        sizes=[100, 1000, 10000, 100000]
    )
    
    for r in results:
        print(f"{r.name}: {r.avg_time_s:.3f}s")

Module Structure
----------------
benchmarks.py
├── Enumerations
│   ├── BenchmarkCategory    - Test categories
│   ├── BenchmarkSize        - Dataset size classification
│   └── BenchmarkStatus      - Execution status
│
├── Data Classes
│   ├── BenchmarkResult      - Individual benchmark result
│   └── BenchmarkSuite       - Collection of results
│
├── BenchmarkRunner          - Main execution engine
│
└── Test Classes
    ├── TestDataGenerationBenchmarks
    ├── TestStorageBenchmarks
    ├── TestValidationBenchmarks
    ├── TestRateLimiterBenchmarks
    ├── TestMemoryBenchmarks
    └── TestEndToEndBenchmarks

Author: Crypto StatArb Quantitative Research
Version: 2.0.0
"""

from __future__ import annotations

# =============================================================================
# IMPORTS
# =============================================================================

from .benchmarks import (
    # Enumerations
    BenchmarkCategory,
    BenchmarkSize,
    BenchmarkStatus,
    
    # Data classes
    BenchmarkResult,
    BenchmarkSuite,
    
    # Runner
    BenchmarkRunner,
    
    # Test classes
    TestDataGenerationBenchmarks,
    TestStorageBenchmarks,
    TestValidationBenchmarks,
    TestRateLimiterBenchmarks,
    TestMemoryBenchmarks,
    TestEndToEndBenchmarks,
    
    # Functions
    run_all_benchmarks,
)


# =============================================================================
# PACKAGE METADATA
# =============================================================================

__version__ = "2.0.0"
__author__ = "Crypto StatArb Quantitative Research"

__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # Enumerations
    'BenchmarkCategory',
    'BenchmarkSize',
    'BenchmarkStatus',
    
    # Data classes
    'BenchmarkResult',
    'BenchmarkSuite',
    
    # Runner
    'BenchmarkRunner',
    
    # Test classes
    'TestDataGenerationBenchmarks',
    'TestStorageBenchmarks',
    'TestValidationBenchmarks',
    'TestRateLimiterBenchmarks',
    'TestMemoryBenchmarks',
    'TestEndToEndBenchmarks',
    
    # Functions
    'run_all_benchmarks',
]


# =============================================================================
# PACKAGE-LEVEL UTILITIES
# =============================================================================

def get_performance_targets() -> dict:
    """Get performance targets for all categories."""
    return {
        'data_generation': {
            'small_rows': 1000,
            'small_max_time_s': 0.5,
            'medium_rows': 10000,
            'medium_max_time_s': 5.0,
            'large_rows': 100000,
            'large_max_time_s': 30.0,
        },
        'storage_write': {
            'min_rows_per_sec': 100000,
            'max_time_per_100k_rows_s': 1.0,
        },
        'storage_read': {
            'min_rows_per_sec': 200000,
        },
        'storage_query': {
            'min_rows_per_sec': 500000,
        },
        'validation': {
            'funding_min_rows_per_sec': 50000,
            'ohlcv_min_rows_per_sec': 30000,
        },
        'memory': {
            'max_mb_per_100k_rows': 100,
            'max_bytes_per_row': 500,
        },
    }


def quick_benchmark(func, *args, **kwargs) -> BenchmarkResult:
    """Run a quick benchmark with default settings."""
    runner = BenchmarkRunner(warmup=1, iterations=3)
    return runner.run("quick_benchmark", func, *args, **kwargs)


# Add utilities to exports
__all__.extend([
    'get_performance_targets',
    'quick_benchmark',
])