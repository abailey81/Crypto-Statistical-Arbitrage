"""
Performance Benchmarks for Crypto Statistical Arbitrage Pipeline
================================================================

Comprehensive benchmarks for measuring throughput, latency, memory usage,
and resource efficiency across all data collection and processing components.

Benchmark Categories
--------------------
1. Data Generation Benchmarks:
   - Mock data generation throughput
   - Memory efficiency per record
   - Scaling characteristics

2. Storage Benchmarks:
   - Write throughput (rows/sec, MB/sec)
   - Read throughput
   - Query performance with filters
   - Compression comparison

3. Validation Benchmarks:
   - Funding rate validation throughput
   - OHLCV validation throughput
   - Cross-venue validation

4. Rate Limiter Benchmarks:
   - Token bucket overhead
   - Concurrent acquisition

5. Memory Benchmarks:
   - Peak memory usage
   - Memory efficiency ratios
   - Garbage collection impact

6. Async Operation Benchmarks:
   - Concurrent collection simulation
   - Network I/O patterns

Performance Metrics
-------------------
Throughput:

    T = N / Δt  (items/second or MB/second)
    
    Where:
        N = number of items processed
        Δt = elapsed time

Latency Percentiles:

    p50, p90, p95, p99 latencies for operation timing

Memory Efficiency:

    η_mem = useful_data_size / total_allocated_memory

Scaling Factor:

    α = T(2N) / T(N)
    
    Where α ≈ 1.0 indicates linear scaling

Expected Performance Targets
----------------------------
Data Generation:
    - Small dataset (<1000 rows): < 0.5s
    - Medium dataset (~50,000 rows): < 5s
    - Large dataset (~500,000 rows): < 30s

Storage Operations:
    - Parquet write: > 100,000 rows/sec
    - Parquet read: > 200,000 rows/sec
    - Filtered query: > 500,000 rows/sec

Validation:
    - Funding validation: > 50,000 rows/sec
    - OHLCV validation: > 30,000 rows/sec

Memory:
    - Peak memory per 100K rows: < 100 MB
    - Bytes per row: < 500 bytes

Author: Crypto StatArb Quantitative Research
Version: 2.0.0
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
import asyncio
import tempfile
import shutil
import time
import gc
import sys
import os
import tracemalloc
import statistics
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import (
    Dict, List, Callable, Any, Optional, Tuple, 
    TypeVar, Generic, Union, Awaitable
)
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from contextlib import contextmanager
import threading

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.fixtures.mock_data import (
    MockDataConfig, 
    create_mock_dataset, 
    quick_funding_data, 
    quick_ohlcv_data,
    quick_options_data,
    quick_pool_data,
    MarketRegime,
    ASSET_PARAMETERS,
    VENUE_CONFIGS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# BENCHMARK ENUMERATIONS
# =============================================================================

class BenchmarkCategory(Enum):
    """Categories of benchmarks for organized reporting."""
    DATA_GENERATION = "data_generation"
    STORAGE_WRITE = "storage_write"
    STORAGE_READ = "storage_read"
    STORAGE_QUERY = "storage_query"
    VALIDATION = "validation"
    RATE_LIMITER = "rate_limiter"
    MEMORY = "memory"
    ASYNC_IO = "async_io"
    END_TO_END = "end_to_end"
    
    @property
    def description(self) -> str:
        """Category description."""
        descriptions = {
            self.DATA_GENERATION: "Mock data generation throughput",
            self.STORAGE_WRITE: "Parquet/storage write performance",
            self.STORAGE_READ: "Parquet/storage read performance",
            self.STORAGE_QUERY: "Filtered query performance",
            self.VALIDATION: "Data validation throughput",
            self.RATE_LIMITER: "Rate limiter overhead",
            self.MEMORY: "Memory usage and efficiency",
            self.ASYNC_IO: "Async I/O patterns",
            self.END_TO_END: "Full pipeline benchmarks",
        }
        return descriptions.get(self, "")
    
    @property
    def priority(self) -> int:
        """Execution priority (1 = highest)."""
        priorities = {
            self.DATA_GENERATION: 1,
            self.STORAGE_WRITE: 2,
            self.STORAGE_READ: 2,
            self.VALIDATION: 3,
            self.MEMORY: 4,
            self.RATE_LIMITER: 5,
            self.STORAGE_QUERY: 3,
            self.ASYNC_IO: 4,
            self.END_TO_END: 6,
        }
        return priorities.get(self, 5)


class BenchmarkSize(Enum):
    """Dataset sizes for scalability testing."""
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"
    
    @property
    def row_count_target(self) -> int:
        """Target row count for this size."""
        counts = {
            self.TINY: 100,
            self.SMALL: 1_000,
            self.MEDIUM: 10_000,
            self.LARGE: 100_000,
            self.XLARGE: 1_000_000,
        }
        return counts.get(self, 10_000)
    
    @property
    def days(self) -> int:
        """Approximate days of data."""
        days_map = {
            self.TINY: 1,
            self.SMALL: 7,
            self.MEDIUM: 30,
            self.LARGE: 180,
            self.XLARGE: 730,
        }
        return days_map.get(self, 30)
    
    @property
    def timeout_seconds(self) -> int:
        """Max allowed time for this size."""
        timeouts = {
            self.TINY: 5,
            self.SMALL: 10,
            self.MEDIUM: 30,
            self.LARGE: 120,
            self.XLARGE: 600,
        }
        return timeouts.get(self, 60)
    
    @property
    def memory_limit_mb(self) -> int:
        """Max allowed memory for this size."""
        limits = {
            self.TINY: 50,
            self.SMALL: 100,
            self.MEDIUM: 500,
            self.LARGE: 2000,
            self.XLARGE: 8000,
        }
        return limits.get(self, 1000)


class BenchmarkStatus(Enum):
    """Status of benchmark execution."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    OOM = "out_of_memory"
    
    @property
    def is_success(self) -> bool:
        """Whether status indicates success."""
        return self == self.PASSED
    
    @property
    def color_code(self) -> str:
        """ANSI color code for terminal output."""
        colors = {
            self.PENDING: "\033[33m",   # Yellow
            self.RUNNING: "\033[34m",   # Blue
            self.PASSED: "\033[32m",    # Green
            self.FAILED: "\033[31m",    # Red
            self.SKIPPED: "\033[90m",   # Gray
            self.TIMEOUT: "\033[35m",   # Magenta
            self.OOM: "\033[31m",       # Red
        }
        return colors.get(self, "\033[0m")


# =============================================================================
# BENCHMARK DATA CLASSES
# =============================================================================

@dataclass
class BenchmarkResult:
    """
    Comprehensive result of a benchmark run.
    
    Captures timing, throughput, memory, and statistical metrics
    for detailed performance analysis.
    """
    # Identification
    name: str
    category: BenchmarkCategory
    size: BenchmarkSize
    status: BenchmarkStatus = BenchmarkStatus.PENDING
    
    # Timing metrics
    iterations: int = 0
    total_time_s: float = 0.0
    times: List[float] = field(default_factory=list)
    
    # Throughput
    throughput_items_per_sec: float = 0.0
    throughput_mb_per_sec: float = 0.0
    items_processed: int = 0
    bytes_processed: int = 0
    
    # Memory
    memory_peak_mb: float = 0.0
    memory_avg_mb: float = 0.0
    memory_samples: List[float] = field(default_factory=list)
    
    # Additional metrics
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def avg_time_s(self) -> float:
        """Average execution time."""
        return np.mean(self.times) if self.times else 0.0
    
    @property
    def min_time_s(self) -> float:
        """Minimum execution time."""
        return np.min(self.times) if self.times else 0.0
    
    @property
    def max_time_s(self) -> float:
        """Maximum execution time."""
        return np.max(self.times) if self.times else 0.0
    
    @property
    def std_time_s(self) -> float:
        """Standard deviation of execution time."""
        return np.std(self.times) if len(self.times) > 1 else 0.0
    
    @property
    def p50_time_s(self) -> float:
        """Median (p50) execution time."""
        return np.percentile(self.times, 50) if self.times else 0.0
    
    @property
    def p90_time_s(self) -> float:
        """90th percentile execution time."""
        return np.percentile(self.times, 90) if self.times else 0.0
    
    @property
    def p95_time_s(self) -> float:
        """95th percentile execution time."""
        return np.percentile(self.times, 95) if self.times else 0.0
    
    @property
    def p99_time_s(self) -> float:
        """99th percentile execution time."""
        return np.percentile(self.times, 99) if self.times else 0.0
    
    @property
    def coefficient_of_variation(self) -> float:
        """CV = std/mean, measures consistency."""
        if self.avg_time_s > 0:
            return self.std_time_s / self.avg_time_s
        return 0.0
    
    @property
    def bytes_per_item(self) -> float:
        """Average bytes per processed item."""
        if self.items_processed > 0:
            return self.bytes_processed / self.items_processed
        return 0.0
    
    @property
    def memory_per_1k_items_mb(self) -> float:
        """Memory usage per 1000 items."""
        if self.items_processed > 0:
            return (self.memory_peak_mb * 1000) / self.items_processed
        return 0.0
    
    def meets_target(self, target_throughput: float, target_memory_mb: float) -> bool:
        """Check if benchmark meets performance targets."""
        throughput_ok = self.throughput_items_per_sec >= target_throughput
        memory_ok = self.memory_peak_mb <= target_memory_mb
        return throughput_ok and memory_ok and self.status == BenchmarkStatus.PASSED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'category': self.category.value,
            'size': self.size.value,
            'status': self.status.value,
            'iterations': self.iterations,
            'timing': {
                'total_s': round(self.total_time_s, 4),
                'avg_s': round(self.avg_time_s, 4),
                'min_s': round(self.min_time_s, 4),
                'max_s': round(self.max_time_s, 4),
                'std_s': round(self.std_time_s, 4),
                'p50_s': round(self.p50_time_s, 4),
                'p90_s': round(self.p90_time_s, 4),
                'p95_s': round(self.p95_time_s, 4),
                'p99_s': round(self.p99_time_s, 4),
                'cv': round(self.coefficient_of_variation, 4),
            },
            'throughput': {
                'items_per_sec': round(self.throughput_items_per_sec, 2),
                'mb_per_sec': round(self.throughput_mb_per_sec, 2),
                'items_processed': self.items_processed,
                'bytes_processed': self.bytes_processed,
            },
            'memory': {
                'peak_mb': round(self.memory_peak_mb, 2),
                'avg_mb': round(self.memory_avg_mb, 2),
                'per_1k_items_mb': round(self.memory_per_1k_items_mb, 4),
            },
            'extra_metrics': self.extra_metrics,
            'error': self.error_message,
            'timestamp': self.timestamp.isoformat(),
        }
    
    def __str__(self) -> str:
        """Human-readable summary."""
        status_color = self.status.color_code
        reset = "\033[0m"
        
        return (
            f"{status_color}[{self.status.value.upper()}]{reset} "
            f"{self.name}: "
            f"{self.avg_time_s:.4f}s avg (±{self.std_time_s:.4f}s), "
            f"{self.throughput_items_per_sec:,.0f} items/s, "
            f"peak mem: {self.memory_peak_mb:.1f}MB"
        )


@dataclass
class BenchmarkSuite:
    """
    Collection of related benchmarks with aggregate statistics.
    """
    name: str
    description: str
    results: List[BenchmarkResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def total_duration_s(self) -> float:
        """Total suite duration."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return sum(r.total_time_s for r in self.results)
    
    @property
    def pass_count(self) -> int:
        """Number of passed benchmarks."""
        return sum(1 for r in self.results if r.status == BenchmarkStatus.PASSED)
    
    @property
    def fail_count(self) -> int:
        """Number of failed benchmarks."""
        return sum(1 for r in self.results if r.status == BenchmarkStatus.FAILED)
    
    @property
    def pass_rate(self) -> float:
        """Percentage of passed benchmarks."""
        if not self.results:
            return 0.0
        return (self.pass_count / len(self.results)) * 100
    
    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results.append(result)
    
    def get_by_category(self, category: BenchmarkCategory) -> List[BenchmarkResult]:
        """Get results by category."""
        return [r for r in self.results if r.category == category]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'summary': {
                'total_benchmarks': len(self.results),
                'passed': self.pass_count,
                'failed': self.fail_count,
                'pass_rate': round(self.pass_rate, 1),
                'total_duration_s': round(self.total_duration_s, 2),
            },
            'results': [r.to_dict() for r in self.results],
        }
    
    def generate_report(self) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 80,
            f"BENCHMARK SUITE: {self.name}",
            f"{self.description}",
            "=" * 80,
            "",
            f"Total benchmarks: {len(self.results)}",
            f"Passed: {self.pass_count} ({self.pass_rate:.1f}%)",
            f"Failed: {self.fail_count}",
            f"Total duration: {self.total_duration_s:.2f}s",
            "",
            "-" * 80,
            "RESULTS BY CATEGORY",
            "-" * 80,
        ]
        
        for category in BenchmarkCategory:
            cat_results = self.get_by_category(category)
            if cat_results:
                lines.append(f"\n{category.value.upper()}:")
                for result in cat_results:
                    lines.append(f"  {result}")
        
        lines.extend(["", "=" * 80])
        return "\n".join(lines)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class BenchmarkRunner:
    """
    Comprehensive benchmark execution engine.
    
    Features:
    - Warmup iterations
    - Memory tracking
    - Statistical analysis
    - Async support
    - Timeout handling
    """
    
    def __init__(
        self,
        warmup: int = 2,
        iterations: int = 5,
        track_memory: bool = True,
        gc_before_run: bool = True,
        timeout_seconds: Optional[int] = None
    ):
        """
        Initialize benchmark runner.
        
        Args:
            warmup: Number of warmup iterations (not counted)
            iterations: Number of measured iterations
            track_memory: Whether to track memory usage
            gc_before_run: Whether to run GC before each benchmark
            timeout_seconds: Optional timeout for benchmarks
        """
        self.warmup = warmup
        self.iterations = iterations
        self.track_memory = track_memory
        self.gc_before_run = gc_before_run
        self.timeout_seconds = timeout_seconds
        self.results: List[BenchmarkResult] = []
    
    @contextmanager
    def _memory_tracker(self):
        """Context manager for memory tracking."""
        if self.track_memory:
            gc.collect()
            tracemalloc.start()
            try:
                yield
            finally:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                self._last_memory_current = current
                self._last_memory_peak = peak
        else:
            self._last_memory_current = 0
            self._last_memory_peak = 0
            yield
    
    def run(
        self,
        name: str,
        func: Callable[..., Any],
        *args,
        category: BenchmarkCategory = BenchmarkCategory.DATA_GENERATION,
        size: BenchmarkSize = BenchmarkSize.MEDIUM,
        items_per_call: int = 1,
        bytes_per_call: int = 0,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run a synchronous benchmark.
        
        Args:
            name: Benchmark name
            func: Function to benchmark
            category: Benchmark category
            size: Dataset size category
            items_per_call: Items processed per function call
            bytes_per_call: Bytes processed per call
        """
        result = BenchmarkResult(
            name=name,
            category=category,
            size=size,
            status=BenchmarkStatus.RUNNING
        )
        
        try:
            # Warmup
            for _ in range(self.warmup):
                func(*args, **kwargs)
            
            # Measured runs
            if self.gc_before_run:
                gc.collect()
            
            times = []
            memory_samples = []
            
            with self._memory_tracker():
                for _ in range(self.iterations):
                    start = time.perf_counter()
                    func(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)
                    
                    if self.track_memory:
                        current, _ = tracemalloc.get_traced_memory()
                        memory_samples.append(current / (1024 * 1024))
            
            # Populate result
            result.times = times
            result.iterations = self.iterations
            result.total_time_s = sum(times)
            result.items_processed = items_per_call * self.iterations
            result.bytes_processed = bytes_per_call * self.iterations
            
            # Throughput
            if result.avg_time_s > 0:
                result.throughput_items_per_sec = items_per_call / result.avg_time_s
                if bytes_per_call > 0:
                    result.throughput_mb_per_sec = (bytes_per_call / (1024 * 1024)) / result.avg_time_s
            
            # Memory
            result.memory_peak_mb = self._last_memory_peak / (1024 * 1024)
            result.memory_samples = memory_samples
            result.memory_avg_mb = np.mean(memory_samples) if memory_samples else 0.0
            
            result.status = BenchmarkStatus.PASSED
            
        except MemoryError as e:
            result.status = BenchmarkStatus.OOM
            result.error_message = str(e)
        except TimeoutError as e:
            result.status = BenchmarkStatus.TIMEOUT
            result.error_message = str(e)
        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.error_message = str(e)
        
        self.results.append(result)
        return result
    
    async def run_async(
        self,
        name: str,
        func: Callable[..., Awaitable[Any]],
        *args,
        category: BenchmarkCategory = BenchmarkCategory.ASYNC_IO,
        size: BenchmarkSize = BenchmarkSize.MEDIUM,
        items_per_call: int = 1,
        **kwargs
    ) -> BenchmarkResult:
        """Run an async benchmark."""
        result = BenchmarkResult(
            name=name,
            category=category,
            size=size,
            status=BenchmarkStatus.RUNNING
        )
        
        try:
            # Warmup
            for _ in range(self.warmup):
                await func(*args, **kwargs)
            
            if self.gc_before_run:
                gc.collect()
            
            times = []
            
            with self._memory_tracker():
                for _ in range(self.iterations):
                    start = time.perf_counter()
                    await func(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)
            
            result.times = times
            result.iterations = self.iterations
            result.total_time_s = sum(times)
            result.items_processed = items_per_call * self.iterations
            
            if result.avg_time_s > 0:
                result.throughput_items_per_sec = items_per_call / result.avg_time_s
            
            result.memory_peak_mb = self._last_memory_peak / (1024 * 1024)
            result.status = BenchmarkStatus.PASSED
            
        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.error_message = str(e)
        
        self.results.append(result)
        return result
    
    def run_scaling_test(
        self,
        name: str,
        func_factory: Callable[[int], Callable],
        sizes: List[int],
        category: BenchmarkCategory = BenchmarkCategory.DATA_GENERATION
    ) -> List[BenchmarkResult]:
        """
        Run scaling test across multiple sizes.
        
        Args:
            name: Base benchmark name
            func_factory: Function that takes size and returns benchmark function
            sizes: List of sizes to test
        """
        results = []
        
        for size_val in sizes:
            func = func_factory(size_val)
            
            # Map size to BenchmarkSize
            if size_val < 1000:
                size = BenchmarkSize.TINY
            elif size_val < 10000:
                size = BenchmarkSize.SMALL
            elif size_val < 100000:
                size = BenchmarkSize.MEDIUM
            elif size_val < 1000000:
                size = BenchmarkSize.LARGE
            else:
                size = BenchmarkSize.XLARGE
            
            result = self.run(
                f"{name}_n{size_val}",
                func,
                category=category,
                size=size,
                items_per_call=size_val
            )
            results.append(result)
        
        return results
    
    def report(self) -> str:
        """Generate text report of all results."""
        lines = [
            "=" * 80,
            "BENCHMARK RESULTS",
            "=" * 80,
            "",
        ]
        
        for result in self.results:
            lines.append(str(result))
        
        lines.extend(["", "=" * 80])
        return "\n".join(lines)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        records = []
        for r in self.results:
            records.append({
                'name': r.name,
                'category': r.category.value,
                'size': r.size.value,
                'status': r.status.value,
                'avg_time_s': r.avg_time_s,
                'std_time_s': r.std_time_s,
                'p50_time_s': r.p50_time_s,
                'p95_time_s': r.p95_time_s,
                'throughput_items_s': r.throughput_items_per_sec,
                'throughput_mb_s': r.throughput_mb_per_sec,
                'memory_peak_mb': r.memory_peak_mb,
                'items_processed': r.items_processed,
            })
        return pd.DataFrame(records)
    
    def save_results(self, path: Path) -> None:
        """Save results to JSON file."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'warmup': self.warmup,
                'iterations': self.iterations,
            },
            'results': [r.to_dict() for r in self.results],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# DATA GENERATION BENCHMARKS
# =============================================================================

class TestDataGenerationBenchmarks:
    """Benchmarks for mock data generation performance."""
    
    @pytest.fixture
    def runner(self):
        """Create benchmark runner."""
        return BenchmarkRunner(warmup=1, iterations=3)
    
    def test_funding_generation_tiny(self, runner):
        """Benchmark tiny funding data generation."""
        config = MockDataConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            symbols=['BTC'],
            venues=['binance']
        )
        
        def generate():
            return create_mock_dataset('funding', config)
        
        result = runner.run(
            "funding_tiny",
            generate,
            category=BenchmarkCategory.DATA_GENERATION,
            size=BenchmarkSize.TINY,
            items_per_call=50  # Approximate rows
        )
        
        print(f"\n{result}")
        assert result.status == BenchmarkStatus.PASSED
        assert result.avg_time_s < BenchmarkSize.TINY.timeout_seconds
    
    def test_funding_generation_small(self, runner):
        """Benchmark small funding data generation."""
        config = MockDataConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 8),
            symbols=['BTC', 'ETH'],
            venues=['binance', 'hyperliquid']
        )
        
        def generate():
            return create_mock_dataset('funding', config)
        
        result = runner.run(
            "funding_small",
            generate,
            category=BenchmarkCategory.DATA_GENERATION,
            size=BenchmarkSize.SMALL,
            items_per_call=500
        )
        
        print(f"\n{result}")
        assert result.status == BenchmarkStatus.PASSED
        assert result.avg_time_s < 5.0
    
    def test_funding_generation_medium(self, runner):
        """Benchmark medium funding data generation."""
        config = MockDataConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 1),
            symbols=['BTC', 'ETH', 'SOL', 'ARB'],
            venues=['binance', 'bybit', 'hyperliquid']
        )
        
        def generate():
            return create_mock_dataset('funding', config)
        
        result = runner.run(
            "funding_medium",
            generate,
            category=BenchmarkCategory.DATA_GENERATION,
            size=BenchmarkSize.MEDIUM,
            items_per_call=5000
        )
        
        print(f"\n{result}")
        assert result.status == BenchmarkStatus.PASSED
        assert result.avg_time_s < 10.0
    
    def test_funding_generation_large(self, runner):
        """Benchmark large funding data generation."""
        config = MockDataConfig(
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2024, 1, 1),
            symbols=list(ASSET_PARAMETERS.keys())[:8],
            venues=['binance', 'bybit', 'hyperliquid']
        )
        
        def generate():
            return create_mock_dataset('funding', config)
        
        result = runner.run(
            "funding_large",
            generate,
            category=BenchmarkCategory.DATA_GENERATION,
            size=BenchmarkSize.LARGE,
            items_per_call=50000
        )
        
        print(f"\n{result}")
        assert result.status == BenchmarkStatus.PASSED
        assert result.avg_time_s < 60.0
    
    def test_ohlcv_generation(self, runner):
        """Benchmark OHLCV data generation."""
        config = MockDataConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            symbols=['BTC', 'ETH', 'SOL'],
            venues=['binance']
        )
        
        def generate():
            return create_mock_dataset('ohlcv', config, timeframe='1h')
        
        result = runner.run(
            "ohlcv_medium",
            generate,
            category=BenchmarkCategory.DATA_GENERATION,
            size=BenchmarkSize.MEDIUM,
            items_per_call=10000
        )
        
        print(f"\n{result}")
        assert result.status == BenchmarkStatus.PASSED
    
    def test_options_generation(self, runner):
        """Benchmark options chain generation."""
        def generate():
            return quick_options_data(underlying='BTC', spot_price=40000)
        
        result = runner.run(
            "options_chain",
            generate,
            category=BenchmarkCategory.DATA_GENERATION,
            size=BenchmarkSize.SMALL,
            items_per_call=500
        )
        
        print(f"\n{result}")
        assert result.status == BenchmarkStatus.PASSED
    
    def test_pool_generation(self, runner):
        """Benchmark DEX pool generation."""
        def generate():
            return quick_pool_data(n_pools=100)
        
        result = runner.run(
            "dex_pools",
            generate,
            category=BenchmarkCategory.DATA_GENERATION,
            size=BenchmarkSize.SMALL,
            items_per_call=100
        )
        
        print(f"\n{result}")
        assert result.status == BenchmarkStatus.PASSED
    
    def test_generation_scaling(self, runner):
        """Test generation scaling characteristics."""
        sizes = [7, 30, 90, 180]  # Days
        results = []
        
        for days in sizes:
            config = MockDataConfig(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 1) + timedelta(days=days),
                symbols=['BTC', 'ETH'],
                venues=['binance']
            )
            
            def generate():
                return create_mock_dataset('funding', config)
            
            result = runner.run(
                f"scaling_{days}d",
                generate,
                category=BenchmarkCategory.DATA_GENERATION,
                size=BenchmarkSize.SMALL if days < 60 else BenchmarkSize.MEDIUM,
                items_per_call=days * 6  # ~6 records per day
            )
            results.append(result)
        
        # Check roughly linear scaling
        if len(results) >= 2:
            time_ratio = results[-1].avg_time_s / results[0].avg_time_s
            size_ratio = sizes[-1] / sizes[0]
            scaling_factor = time_ratio / size_ratio
            
            print(f"\nScaling factor: {scaling_factor:.2f} (1.0 = linear)")
            # Should be roughly linear (within 2x)
            assert scaling_factor < 3.0, "Scaling worse than O(n) * 3"


# =============================================================================
# STORAGE BENCHMARKS
# =============================================================================

class TestStorageBenchmarks:
    """Benchmarks for storage operations."""
    
    @pytest.fixture
    def runner(self):
        """Create benchmark runner."""
        return BenchmarkRunner(warmup=1, iterations=5)
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield Path(temp_path)
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample datasets of various sizes."""
        return {
            'small': quick_funding_data(n_days=7),
            'medium': quick_funding_data(n_days=90),
            'large': create_mock_dataset(
                'funding',
                MockDataConfig(
                    start_date=datetime(2023, 1, 1),
                    end_date=datetime(2024, 1, 1),
                    symbols=['BTC', 'ETH', 'SOL', 'ARB'],
                    venues=['binance', 'hyperliquid']
                )
            )
        }
    
    def test_parquet_write_small(self, runner, temp_dir, sample_data):
        """Benchmark small Parquet write."""
        try:
            from data_collection.utils.storage import ParquetStorage
        except ImportError:
            pytest.skip("Storage module not available")
        
        storage = ParquetStorage(base_path=temp_dir)
        df = sample_data['small']
        counter = [0]
        
        def write_op():
            counter[0] += 1
            storage.save(df, f'bench_small_{counter[0]}')
        
        bytes_size = df.memory_usage(deep=True).sum()
        
        result = runner.run(
            "parquet_write_small",
            write_op,
            category=BenchmarkCategory.STORAGE_WRITE,
            size=BenchmarkSize.SMALL,
            items_per_call=len(df),
            bytes_per_call=int(bytes_size)
        )
        
        result.extra_metrics['rows'] = len(df)
        result.extra_metrics['rows_per_sec'] = len(df) / result.avg_time_s
        
        print(f"\n{result}")
        print(f"  Rows/sec: {result.extra_metrics['rows_per_sec']:,.0f}")
    
    def test_parquet_write_large(self, runner, temp_dir, sample_data):
        """Benchmark large Parquet write."""
        try:
            from data_collection.utils.storage import ParquetStorage
        except ImportError:
            pytest.skip("Storage module not available")
        
        storage = ParquetStorage(base_path=temp_dir)
        df = sample_data['large']
        counter = [0]
        
        def write_op():
            counter[0] += 1
            storage.save(df, f'bench_large_{counter[0]}')
        
        bytes_size = df.memory_usage(deep=True).sum()
        
        result = runner.run(
            "parquet_write_large",
            write_op,
            category=BenchmarkCategory.STORAGE_WRITE,
            size=BenchmarkSize.LARGE,
            items_per_call=len(df),
            bytes_per_call=int(bytes_size)
        )
        
        result.extra_metrics['rows'] = len(df)
        result.extra_metrics['rows_per_sec'] = len(df) / result.avg_time_s
        
        print(f"\n{result}")
        print(f"  Rows/sec: {result.extra_metrics['rows_per_sec']:,.0f}")
        
        # Should achieve > 10K rows/sec
        assert result.extra_metrics['rows_per_sec'] > 10000
    
    def test_parquet_read(self, runner, temp_dir, sample_data):
        """Benchmark Parquet read."""
        try:
            from data_collection.utils.storage import ParquetStorage
        except ImportError:
            pytest.skip("Storage module not available")
        
        storage = ParquetStorage(base_path=temp_dir)
        df = sample_data['large']
        storage.save(df, 'read_bench')
        
        result = runner.run(
            "parquet_read_large",
            storage.load,
            'read_bench',
            category=BenchmarkCategory.STORAGE_READ,
            size=BenchmarkSize.LARGE,
            items_per_call=len(df)
        )
        
        result.extra_metrics['rows'] = len(df)
        result.extra_metrics['rows_per_sec'] = len(df) / result.avg_time_s
        
        print(f"\n{result}")
        print(f"  Rows/sec: {result.extra_metrics['rows_per_sec']:,.0f}")
        
        # Read should be faster than write
        assert result.extra_metrics['rows_per_sec'] > 50000
    
    def test_filtered_query(self, runner, temp_dir, sample_data):
        """Benchmark filtered query."""
        try:
            from data_collection.utils.storage import create_optimized_storage, PartitionStrategy
        except ImportError:
            pytest.skip("Storage module not available")
        
        storage = create_optimized_storage(base_path=temp_dir)
        df = sample_data['large']
        
        storage.save_optimized(
            df, 'query_bench',
            partition_strategy=PartitionStrategy.BY_SYMBOL
        )
        
        def query_op():
            return storage.query('query_bench', filters={'symbol': 'BTC'})
        
        result = runner.run(
            "filtered_query",
            query_op,
            category=BenchmarkCategory.STORAGE_QUERY,
            size=BenchmarkSize.LARGE,
            items_per_call=len(df) // 4  # Approximate filtered size
        )
        
        print(f"\n{result}")
    
    def test_compression_comparison(self, runner, temp_dir, sample_data):
        """Benchmark different compression algorithms."""
        try:
            from data_collection.utils.storage import ParquetStorage
        except ImportError:
            pytest.skip("Storage module not available")
        
        df = sample_data['large']
        compression_results = {}
        
        for compression in ['snappy', 'gzip', 'zstd']:
            storage = ParquetStorage(base_path=temp_dir, compression=compression)
            counter = [0]
            
            def write_op():
                counter[0] += 1
                storage.save(df, f'comp_{compression}_{counter[0]}')
            
            result = runner.run(
                f"compression_{compression}",
                write_op,
                category=BenchmarkCategory.STORAGE_WRITE,
                size=BenchmarkSize.LARGE,
                items_per_call=len(df)
            )
            
            # Get file size
            file_path = temp_dir / f'comp_{compression}_1.parquet'
            if file_path.exists():
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                result.extra_metrics['file_size_mb'] = file_size_mb
                result.extra_metrics['compression_ratio'] = (
                    df.memory_usage(deep=True).sum() / (1024 * 1024)
                ) / file_size_mb
            
            compression_results[compression] = result
            print(f"\n{result}")
        
        # zstd should be faster than gzip
        assert compression_results['zstd'].avg_time_s < compression_results['gzip'].avg_time_s * 1.5


# =============================================================================
# VALIDATION BENCHMARKS
# =============================================================================

class TestValidationBenchmarks:
    """Benchmarks for data validation operations."""
    
    @pytest.fixture
    def runner(self):
        """Create benchmark runner."""
        return BenchmarkRunner(warmup=1, iterations=5)
    
    def test_funding_validation(self, runner):
        """Benchmark funding rate validation."""
        try:
            from data_collection.utils.data_validator import DataValidator
        except ImportError:
            pytest.skip("Validator not available")
        
        validator = DataValidator()
        df = create_mock_dataset(
            'funding',
            MockDataConfig(
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2024, 1, 1),
                symbols=['BTC', 'ETH', 'SOL', 'ARB'],
                venues=['binance', 'hyperliquid']
            )
        )
        
        result = runner.run(
            "funding_validation",
            validator.validate_funding_rates,
            df,
            category=BenchmarkCategory.VALIDATION,
            size=BenchmarkSize.LARGE,
            items_per_call=len(df)
        )
        
        result.extra_metrics['rows_validated'] = len(df)
        result.extra_metrics['rows_per_sec'] = len(df) / result.avg_time_s
        
        print(f"\n{result}")
        print(f"  Validation throughput: {result.extra_metrics['rows_per_sec']:,.0f} rows/sec")
    
    def test_ohlcv_validation(self, runner):
        """Benchmark OHLCV validation."""
        try:
            from data_collection.utils.data_validator import DataValidator
        except ImportError:
            pytest.skip("Validator not available")
        
        validator = DataValidator()
        df = create_mock_dataset(
            'ohlcv',
            MockDataConfig(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 6, 1),
                symbols=['BTC', 'ETH'],
                venues=['binance']
            ),
            timeframe='1h'
        )
        
        result = runner.run(
            "ohlcv_validation",
            validator.validate_ohlcv,
            df,
            category=BenchmarkCategory.VALIDATION,
            size=BenchmarkSize.MEDIUM,
            items_per_call=len(df)
        )
        
        print(f"\n{result}")
    
    def test_cross_validation(self, runner):
        """Benchmark cross-venue validation."""
        try:
            from data_collection.utils.data_validator import DataValidator
        except ImportError:
            pytest.skip("Validator not available")
        
        validator = DataValidator()
        config = MockDataConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 1),
            symbols=['BTC', 'ETH'],
            venues=['binance', 'hyperliquid']
        )
        
        df = create_mock_dataset('funding', config)
        df1 = df[df['venue'] == 'binance']
        df2 = df[df['venue'] == 'hyperliquid']
        
        def cross_val():
            return validator.cross_validate_venues(df1, df2, on='funding_rate')
        
        result = runner.run(
            "cross_validation",
            cross_val,
            category=BenchmarkCategory.VALIDATION,
            size=BenchmarkSize.MEDIUM,
            items_per_call=len(df1)
        )
        
        print(f"\n{result}")


# =============================================================================
# RATE LIMITER BENCHMARKS
# =============================================================================

class TestRateLimiterBenchmarks:
    """Benchmarks for rate limiter overhead."""
    
    @pytest.fixture
    def runner(self):
        """Create benchmark runner."""
        return BenchmarkRunner(warmup=1, iterations=5)
    
    @pytest.mark.asyncio
    async def test_rate_limiter_overhead(self, runner):
        """Benchmark rate limiter acquisition overhead."""
        try:
            from data_collection.utils.rate_limiter import TokenBucketRateLimiter
        except ImportError:
            pytest.skip("Rate limiter not available")
        
        limiter = TokenBucketRateLimiter(rate=10000, per=1.0, burst=1000)
        
        async def acquire_many():
            for _ in range(100):
                await limiter.acquire()
        
        result = await runner.run_async(
            "rate_limiter_overhead",
            acquire_many,
            category=BenchmarkCategory.RATE_LIMITER,
            size=BenchmarkSize.SMALL,
            items_per_call=100
        )
        
        print(f"\n{result}")
        
        # Overhead should be minimal (< 1ms per acquire)
        assert result.avg_time_s / 100 < 0.001
    
    @pytest.mark.asyncio
    async def test_concurrent_acquire(self, runner):
        """Benchmark concurrent rate limiter acquisition."""
        try:
            from data_collection.utils.rate_limiter import TokenBucketRateLimiter
        except ImportError:
            pytest.skip("Rate limiter not available")
        
        limiter = TokenBucketRateLimiter(rate=1000, per=1.0, burst=100)
        
        async def concurrent_acquire():
            tasks = [limiter.acquire() for _ in range(50)]
            await asyncio.gather(*tasks)
        
        result = await runner.run_async(
            "concurrent_acquire",
            concurrent_acquire,
            category=BenchmarkCategory.RATE_LIMITER,
            size=BenchmarkSize.SMALL,
            items_per_call=50
        )
        
        print(f"\n{result}")


# =============================================================================
# MEMORY BENCHMARKS
# =============================================================================

class TestMemoryBenchmarks:
    """Benchmarks for memory usage."""
    
    def test_large_dataset_memory(self):
        """Test memory usage for large dataset."""
        tracemalloc.start()
        
        config = MockDataConfig(
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2024, 1, 1),
            symbols=list(ASSET_PARAMETERS.keys())[:8],
            venues=['binance', 'bybit', 'hyperliquid', 'dydx']
        )
        
        df = create_mock_dataset('funding', config)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / (1024 * 1024)
        current_mb = current / (1024 * 1024)
        rows = len(df)
        bytes_per_row = current / rows if rows > 0 else 0
        
        print(f"\nLarge dataset memory profile:")
        print(f"  Rows: {rows:,}")
        print(f"  Current memory: {current_mb:.2f} MB")
        print(f"  Peak memory: {peak_mb:.2f} MB")
        print(f"  Bytes per row: {bytes_per_row:.1f}")
        print(f"  Memory per 1K rows: {(current_mb / rows) * 1000:.2f} MB")
        
        # Memory should be reasonable
        assert peak_mb < 2000, f"Memory too high: {peak_mb}MB"
        assert bytes_per_row < 1000, f"Too many bytes per row: {bytes_per_row}"
    
    def test_memory_efficiency_by_datatype(self):
        """Compare memory efficiency across data types."""
        tracemalloc.start()
        
        results = {}
        
        # Funding
        snapshot_before = tracemalloc.take_snapshot()
        funding_df = quick_funding_data(n_days=30)
        snapshot_after = tracemalloc.take_snapshot()
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        funding_mem = sum(stat.size for stat in top_stats) / (1024 * 1024)
        results['funding'] = {
            'rows': len(funding_df),
            'memory_mb': funding_mem,
            'bytes_per_row': funding_mem * 1024 * 1024 / len(funding_df)
        }
        
        # OHLCV
        snapshot_before = tracemalloc.take_snapshot()
        ohlcv_df = quick_ohlcv_data(n_days=30)
        snapshot_after = tracemalloc.take_snapshot()
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        ohlcv_mem = sum(stat.size for stat in top_stats) / (1024 * 1024)
        results['ohlcv'] = {
            'rows': len(ohlcv_df),
            'memory_mb': ohlcv_mem,
            'bytes_per_row': ohlcv_mem * 1024 * 1024 / len(ohlcv_df) if len(ohlcv_df) > 0 else 0
        }
        
        tracemalloc.stop()
        
        print("\nMemory efficiency by data type:")
        for dtype, stats in results.items():
            print(f"  {dtype}: {stats['rows']:,} rows, "
                  f"{stats['memory_mb']:.2f} MB, "
                  f"{stats['bytes_per_row']:.1f} bytes/row")


# =============================================================================
# END-TO-END BENCHMARKS
# =============================================================================

class TestEndToEndBenchmarks:
    """End-to-end pipeline benchmarks."""
    
    @pytest.fixture
    def runner(self):
        """Create benchmark runner."""
        return BenchmarkRunner(warmup=0, iterations=3)
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield Path(temp_path)
        shutil.rmtree(temp_path)
    
    def test_full_pipeline(self, runner, temp_dir):
        """Benchmark full generate-validate-store pipeline."""
        try:
            from data_collection.utils.storage import ParquetStorage
            from data_collection.utils.data_validator import DataValidator
        except ImportError:
            pytest.skip("Required modules not available")
        
        storage = ParquetStorage(base_path=temp_dir)
        validator = DataValidator()
        counter = [0]
        
        def full_pipeline():
            counter[0] += 1
            
            # Generate
            df = create_mock_dataset(
                'funding',
                MockDataConfig(
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 2, 1),
                    symbols=['BTC', 'ETH', 'SOL'],
                    venues=['binance', 'hyperliquid']
                )
            )
            
            # Validate
            result = validator.validate_funding_rates(df)
            assert result['valid']
            
            # Store
            storage.save(df, f'pipeline_{counter[0]}')
            
            # Reload
            loaded = storage.load(f'pipeline_{counter[0]}')
            
            return loaded
        
        result = runner.run(
            "full_pipeline",
            full_pipeline,
            category=BenchmarkCategory.END_TO_END,
            size=BenchmarkSize.MEDIUM,
            items_per_call=2000
        )
        
        print(f"\n{result}")
        assert result.status == BenchmarkStatus.PASSED


# =============================================================================
# BENCHMARK SUITE RUNNER
# =============================================================================

def run_all_benchmarks() -> BenchmarkSuite:
    """Run all benchmarks and return results."""
    import subprocess
    
    suite = BenchmarkSuite(
        name="Crypto StatArb Pipeline Benchmarks",
        description="Comprehensive performance benchmarks for data collection pipeline"
    )
    suite.start_time = datetime.now()
    
    print("Running benchmark suite...")
    print("=" * 80)
    
    # Run pytest benchmarks
    result = subprocess.run(
        ['pytest', __file__, '-v', '--tb=short', '-x', '-k', 'not xlarge'],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    suite.end_time = datetime.now()
    
    return suite


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
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


if __name__ == '__main__':
    run_all_benchmarks()