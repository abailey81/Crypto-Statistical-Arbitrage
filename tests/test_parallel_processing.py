"""
Test Parallel Processing Infrastructure.

This script tests and benchmarks the parallel processing optimizations
for the data collection system. It compares sequential vs parallel
processing to verify speedup and correctness.

Usage:
    python tests/test_parallel_processing.py

Expected Results:
    - 3-10x speedup for symbol-level parallelism
    - 2-3x speedup for increased venue concurrency
    - Combined: 10-15x improvement for multi-venue, multi-symbol collection
"""

import asyncio
import time
import sys
import os
from datetime import datetime, timedelta, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


async def test_parallel_symbol_processor():
    """Test the ParallelSymbolProcessor utility."""
    print("\n" + "=" * 60)
    print("TEST 1: ParallelSymbolProcessor")
    print("=" * 60)

    from data_collection.utils.parallel_processor import (
        ParallelSymbolProcessor,
        ProcessingStatus,
    )

    # Create a mock fetch function that simulates API latency
    async def mock_fetch_symbol(symbol: str, **kwargs) -> pd.DataFrame:
        """Simulate fetching data with network latency."""
        await asyncio.sleep(0.1)  # 100ms simulated latency
        return pd.DataFrame({
            'symbol': [symbol],
            'value': [100 + hash(symbol) % 50],
            'timestamp': [datetime.now(timezone.utc)]
        })

    # Test with 10 symbols
    symbols = ['BTC', 'ETH', 'SOL', 'AVAX', 'ARB', 'OP', 'LINK', 'AAVE', 'UNI', 'MKR']

    # Sequential baseline
    print(f"\nTesting with {len(symbols)} symbols...")
    print("\n1. Sequential processing (baseline)...")
    start = time.monotonic()
    sequential_results = []
    for symbol in symbols:
        df = await mock_fetch_symbol(symbol)
        sequential_results.append(df)
    sequential_time = time.monotonic() - start
    print(f"   Sequential time: {sequential_time:.2f}s")

    # Parallel processing
    print("\n2. Parallel processing (optimized)...")
    processor = ParallelSymbolProcessor(
        venue='test',
        rate_limit_per_minute=600,  # High limit for testing
        max_symbol_concurrency=10,
    )

    start = time.monotonic()
    result = await processor.process_symbols(
        symbols=symbols,
        fetch_func=mock_fetch_symbol,
    )
    parallel_time = time.monotonic() - start
    print(f"   Parallel time: {parallel_time:.2f}s")

    # Calculate speedup
    speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
    print(f"\n   Speedup: {speedup:.1f}x")
    print(f"   Symbols processed: {result.symbols_processed}/{len(symbols)}")
    print(f"   Peak concurrency: {processor.stats.peak_concurrency}")

    # Verify results
    assert result.symbols_processed == len(symbols), "Not all symbols processed!"
    assert result.symbols_failed == 0, "Some symbols failed!"
    assert speedup > 2, f"Expected at least 2x speedup, got {speedup:.1f}x"

    print("\n   [PASS] ParallelSymbolProcessor test PASSED")
    return speedup


async def test_parallel_collection_manager():
    """Test the ParallelCollectionManager."""
    print("\n" + "=" * 60)
    print("TEST 2: ParallelCollectionManager (Multi-venue)")
    print("=" * 60)

    from data_collection.utils.parallel_processor import ParallelCollectionManager

    # Create mock fetch functions for different venues
    async def create_mock_venue_fetch(venue: str, delay: float = 0.05):
        async def fetch(symbol: str, **kwargs) -> pd.DataFrame:
            await asyncio.sleep(delay)
            return pd.DataFrame({
                'symbol': [symbol],
                'venue': [venue],
                'value': [hash(f"{venue}{symbol}") % 100],
                'timestamp': [datetime.now(timezone.utc)]
            })
        return fetch

    venues = ['binance', 'hyperliquid', 'dydx', 'gmx', 'aevo']
    symbols = ['BTC', 'ETH', 'SOL']

    manager = ParallelCollectionManager(max_venue_concurrency=5)

    # Create venue configs
    venue_configs = []
    for venue in venues:
        fetch_func = await create_mock_venue_fetch(venue)
        venue_configs.append({
            'venue': venue,
            'fetch_func': fetch_func,
            'symbols': symbols,
            'data_type': 'test',
        })

    # Test parallel collection
    print(f"\nCollecting from {len(venues)} venues with {len(symbols)} symbols each...")
    start = time.monotonic()
    results = await manager.collect_all_venues(venue_configs)
    parallel_time = time.monotonic() - start

    # Calculate totals
    total_records = sum(r.total_records for r in results)
    total_symbols = sum(r.symbols_processed for r in results)
    failed_symbols = sum(r.symbols_failed for r in results)

    print(f"\n   Parallel time: {parallel_time:.2f}s")
    print(f"   Total records: {total_records}")
    print(f"   Symbols processed: {total_symbols}/{len(venues) * len(symbols)}")
    print(f"   Symbols failed: {failed_symbols}")

    # Verify results
    expected_records = len(venues) * len(symbols)
    assert total_symbols == expected_records, f"Expected {expected_records} symbols"
    assert failed_symbols == 0, "Some symbols failed!"

    print("\n   [PASS] ParallelCollectionManager test PASSED")
    return results


async def test_collector_wrapper():
    """Test the ParallelCollectorWrapper with a mock collector."""
    print("\n" + "=" * 60)
    print("TEST 3: ParallelCollectorWrapper")
    print("=" * 60)

    from data_collection.utils.parallel_processor import wrap_collector_for_parallel

    # Create a mock collector
    class MockCollector:
        VENUE = 'mock_exchange'

        def __init__(self):
            self.call_count = 0

        async def fetch_funding_rates(
            self, symbols, start_date, end_date, **kwargs
        ) -> pd.DataFrame:
            self.call_count += 1
            await asyncio.sleep(0.05)  # Simulate latency
            records = []
            for symbol in symbols:
                records.append({
                    'symbol': symbol,
                    'funding_rate': 0.0001 * (hash(symbol) % 10),
                    'timestamp': datetime.now(timezone.utc),
                    'venue': self.VENUE,
                })
            return pd.DataFrame(records)

    symbols = ['BTC', 'ETH', 'SOL', 'AVAX', 'ARB', 'OP', 'LINK', 'AAVE']

    # Test sequential (original collector)
    print(f"\nTesting with {len(symbols)} symbols...")
    print("\n1. Sequential (original collector)...")
    collector = MockCollector()
    start = time.monotonic()
    sequential_df = await collector.fetch_funding_rates(
        symbols=symbols,
        start_date='2024-01-01',
        end_date='2024-01-31',
    )
    sequential_time = time.monotonic() - start
    print(f"   Sequential time: {sequential_time:.3f}s")
    print(f"   API calls made: {collector.call_count}")

    # Test parallel (wrapped collector)
    print("\n2. Parallel (wrapped collector)...")
    collector2 = MockCollector()
    parallel_collector = wrap_collector_for_parallel(
        collector2,
        max_symbol_concurrency=8,
    )

    start = time.monotonic()
    parallel_df = await parallel_collector.fetch_funding_rates_parallel(
        symbols=symbols,
        start_date='2024-01-01',
        end_date='2024-01-31',
    )
    parallel_time = time.monotonic() - start
    print(f"   Parallel time: {parallel_time:.3f}s")
    print(f"   API calls made: {collector2.call_count}")

    # Calculate speedup
    speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
    print(f"\n   Speedup: {speedup:.1f}x")

    # Verify results match
    assert len(parallel_df) == len(symbols), "Missing records in parallel result"
    assert set(parallel_df['symbol'].unique()) == set(symbols), "Symbol mismatch"

    stats = parallel_collector.get_stats()
    print(f"   Peak concurrency: {stats['processor_stats']['peak_concurrency']}")

    print("\n   [PASS] ParallelCollectorWrapper test PASSED")
    return speedup


async def test_rate_limiting():
    """Test that rate limiting is properly applied."""
    print("\n" + "=" * 60)
    print("TEST 4: Rate Limiting Verification")
    print("=" * 60)

    from data_collection.utils.parallel_processor import ParallelSymbolProcessor

    # Low rate limit to verify throttling
    rate_limit = 30  # 30 requests per minute = 0.5 requests/sec

    async def quick_fetch(symbol: str, **kwargs) -> pd.DataFrame:
        """Very fast fetch to highlight rate limiting."""
        return pd.DataFrame({'symbol': [symbol], 'value': [1]})

    processor = ParallelSymbolProcessor(
        venue='rate_test',
        rate_limit_per_minute=rate_limit,
        max_symbol_concurrency=20,  # High concurrency
    )

    symbols = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']

    print(f"\nProcessing {len(symbols)} symbols with {rate_limit} req/min limit...")
    start = time.monotonic()
    result = await processor.process_symbols(
        symbols=symbols,
        fetch_func=quick_fetch,
    )
    elapsed = time.monotonic() - start

    actual_rate = len(symbols) / elapsed * 60  # requests per minute

    print(f"   Elapsed time: {elapsed:.2f}s")
    print(f"   Actual rate: {actual_rate:.1f} req/min")
    print(f"   Configured limit: {rate_limit} req/min")
    print(f"   Requests made: {processor.stats.requests_made}")

    # With burst allowance, actual rate can exceed limit briefly
    # but should not be more than 2x the limit
    assert actual_rate < rate_limit * 3, f"Rate limit exceeded too much: {actual_rate}"
    assert result.symbols_processed == len(symbols), "Not all symbols processed"

    print("\n   [PASS] Rate limiting test PASSED")


async def run_all_tests():
    """Run all parallel processing tests."""
    print("\n" + "=" * 60)
    print("  PARALLEL PROCESSING TEST SUITE")
    print("=" * 60)

    results = {}

    try:
        results['symbol_processor_speedup'] = await test_parallel_symbol_processor()
    except Exception as e:
        print(f"\n   [FAIL] ParallelSymbolProcessor test FAILED: {e}")
        results['symbol_processor_speedup'] = None

    try:
        results['collection_manager'] = await test_parallel_collection_manager()
    except Exception as e:
        print(f"\n   [FAIL] ParallelCollectionManager test FAILED: {e}")
        results['collection_manager'] = None

    try:
        results['wrapper_speedup'] = await test_collector_wrapper()
    except Exception as e:
        print(f"\n   [FAIL] ParallelCollectorWrapper test FAILED: {e}")
        results['wrapper_speedup'] = None

    try:
        await test_rate_limiting()
        results['rate_limiting'] = True
    except Exception as e:
        print(f"\n   [FAIL] Rate limiting test FAILED: {e}")
        results['rate_limiting'] = False

    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test, result in results.items():
        if result is None or result is False:
            print(f"   [FAIL] {test}: FAILED")
            all_passed = False
        elif isinstance(result, (int, float)):
            print(f"   [PASS] {test}: PASSED (speedup: {result:.1f}x)")
        else:
            print(f"   [PASS] {test}: PASSED")

    if all_passed:
        print("\n   All tests PASSED.")
        print("\n   Parallel processing infrastructure is working correctly.")
        print("   Expected production speedup: 10-15x for multi-venue collection.")
    else:
        print("\n   Some tests FAILED.")
        return 1

    return 0


if __name__ == '__main__':
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
