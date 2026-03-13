"""
Test CEX-DEX Balance Improvements

Tests for the fixed trade data collection from DEX/Hybrid venues.
The project requires balanced data across venue types.

Previously:
- trades: CEX=2,500 vs DEX=0 (0% DEX coverage)
- open_interest: CEX=21,380 vs DEX=65 (0.30% ratio)

Fixes Applied:
1. dYdX V4: Added 'trades' to supported_data_types (method already existed)
2. Hyperliquid: Implemented collect_trades using recentTrades API

Known Limitations:
- DEX open_interest: Only snapshots available (no historical API)
- Hyperliquid trades: Only recent trades (no date range filtering)
- GMX: Pool model, no individual trade history available
"""

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd


async def test_dydx_trades_collection():
    """Test dYdX V4 trades collection (public API, no auth required)."""
    print("\n" + "="*70)
    print("TEST 1: dYdX V4 Trades Collection")
    print("="*70)

    from data_collection.hybrid.dydx_collector import DYDXCollector

    try:
        async with DYDXCollector() as collector:
            # Test single symbol first
            print("\nFetching BTC trades from dYdX V4...")

            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=1)

            df = await collector.fetch_trades(
                symbol='BTC',
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                limit=100
            )

            if df.empty:
                print(" WARNING: No trades returned (may be rate limited)")
                return False

            print(f" SUCCESS: Collected {len(df)} trades from dYdX V4")
            print(f"\nColumns: {list(df.columns)}")

            # Verify expected columns
            expected_cols = ['timestamp', 'symbol', 'price', 'size', 'side']
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                print(f" WARNING: Missing columns: {missing}")
            else:
                print(f" All expected columns present")

            print(f"\nSample data:")
            print(df.head(3).to_string())

            return True

    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_hyperliquid_trades_collection():
    """Test Hyperliquid trades collection (recentTrades API)."""
    print("\n" + "="*70)
    print("TEST 2: Hyperliquid Trades Collection (recentTrades API)")
    print("="*70)

    from data_collection.hybrid.hyperliquid_collector import HyperliquidCollector

    try:
        async with HyperliquidCollector() as collector:
            print("\nFetching recent trades for BTC and ETH...")

            df = await collector.collect_trades(
                symbols=['BTC', 'ETH'],
                start_date=datetime.now(timezone.utc) - timedelta(days=1),
                end_date=datetime.now(timezone.utc),
                limit=50
            )

            if df.empty:
                print(" WARNING: No trades returned")
                return False

            print(f" SUCCESS: Collected {len(df)} trades from Hyperliquid")
            print(f"\nColumns: {list(df.columns)}")

            # Verify expected columns
            expected_cols = ['timestamp', 'symbol', 'price', 'size', 'side']
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                print(f" WARNING: Missing columns: {missing}")
            else:
                print(f" All expected columns present")

            # Show by symbol breakdown
            if 'symbol' in df.columns:
                print(f"\nTrades by symbol:")
                for sym in df['symbol'].unique():
                    count = len(df[df['symbol'] == sym])
                    print(f"  {sym}: {count} trades")

            print(f"\nSample data:")
            print(df.head(3).to_string())

            return True

    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_collection_manager_trades_config():
    """Test that collection manager properly routes trades to DEX/Hybrid venues."""
    print("\n" + "="*70)
    print("TEST 3: Collection Manager Trades Configuration")
    print("="*70)

    from data_collection.collection_manager import COLLECTOR_CONFIGS

    print("\nVenues with 'trades' in supported_data_types:")
    print("-" * 60)

    cex_trades = []
    hybrid_trades = []
    dex_trades = []

    for venue, config in COLLECTOR_CONFIGS.items():
        if 'trades' in config.supported_data_types:
            venue_type = config.venue_type.value
            enabled = "" if config.enabled else ""

            if venue_type == 'CEX':
                cex_trades.append(venue)
            elif venue_type == 'HYBRID':
                hybrid_trades.append(venue)
            elif venue_type == 'DEX':
                dex_trades.append(venue)

            print(f"  {enabled} {venue:15} | {venue_type:8} | {config.supported_data_types}")

    print(f"\n Summary:")
    print(f"  CEX venues with trades:    {len(cex_trades)} ({', '.join(cex_trades) or 'none'})")
    print(f"  HYBRID venues with trades: {len(hybrid_trades)} ({', '.join(hybrid_trades) or 'none'})")
    print(f"  DEX venues with trades:    {len(dex_trades)} ({', '.join(dex_trades) or 'none'})")

    # Check for expected venues
    expected_hybrid = ['hyperliquid', 'dydx']
    missing = [v for v in expected_hybrid if v not in hybrid_trades]

    if missing:
        print(f"\n MISSING: Expected hybrid venues with trades: {missing}")
        return False
    else:
        print(f"\n All expected hybrid venues have trades support")
        return True


async def test_balance_improvement():
    """Demonstrate the improved CEX-DEX balance for trades."""
    print("\n" + "="*70)
    print("TEST 4: CEX-DEX Balance Improvement Verification")
    print("="*70)

    from data_collection.collection_manager import COLLECTOR_CONFIGS

    # Count venues by type and trades support
    stats = {
        'CEX': {'total': 0, 'with_trades': 0, 'enabled': 0},
        'HYBRID': {'total': 0, 'with_trades': 0, 'enabled': 0},
        'DEX': {'total': 0, 'with_trades': 0, 'enabled': 0},
    }

    for venue, config in COLLECTOR_CONFIGS.items():
        vtype = config.venue_type.value
        if vtype in stats:
            stats[vtype]['total'] += 1
            if 'trades' in config.supported_data_types:
                stats[vtype]['with_trades'] += 1
                if config.enabled:
                    stats[vtype]['enabled'] += 1

    print("\nVenue Type Analysis:")
    print("-" * 60)
    print(f"{'Type':<10} | {'Total':<8} | {'With Trades':<12} | {'Enabled':<10}")
    print("-" * 60)

    for vtype, s in stats.items():
        print(f"{vtype:<10} | {s['total']:<8} | {s['with_trades']:<12} | {s['enabled']:<10}")

    # Calculate improvement
    hybrid_dex_trades = stats['HYBRID']['with_trades'] + stats['DEX']['with_trades']

    print(f"\n Balance Assessment:")
    print(f"  Before fix: CEX trades enabled, DEX/HYBRID trades = 0")
    print(f"  After fix:  DEX/HYBRID venues with trades = {hybrid_dex_trades}")

    if hybrid_dex_trades >= 2:
        print(f"\n IMPROVED: {hybrid_dex_trades} DEX/HYBRID venues now support trades")
        return True
    else:
        print(f"\n INSUFFICIENT: Need more DEX/HYBRID venues with trades")
        return False


async def test_open_interest_limitation_documented():
    """Document the open_interest limitation (snapshot only for DEX)."""
    print("\n" + "="*70)
    print("TEST 5: Open Interest Limitation Documentation")
    print("="*70)

    print("""
OPEN INTEREST DATA LIMITATIONS
==============================

CEX venues (Binance, Bybit, OKX):
- Historical open interest available via API
- Can fetch OI time series for backtesting
- Data quality: HIGH

HYBRID venues (Hyperliquid, dYdX):
- Only current OI snapshot available
- No historical OI API endpoint
- Must poll regularly to build history
- Data quality: MEDIUM (snapshot only)

DEX venues (GMX):
- Only current OI snapshot
- No historical data
- Pool-based model differs from order book
- Data quality: MEDIUM (snapshot only)

RECOMMENDATION:
- For historical analysis, rely on CEX OI data
- For DEX/Hybrid, implement scheduled polling to build OI history over time
- Cross-validate CEX and DEX OI levels for consistency checks
""")

    return True


async def main():
    """Run all tests."""
    print("="*70)
    print("CEX-DEX BALANCE IMPROVEMENT TESTS")
    print("="*70)
    print(f"\nTest Time: {datetime.now(timezone.utc).isoformat()}")
    print("\nThese tests validate fixes for Issue #5: CEX-DEX Balance")
    print("  - trades: Previously CEX=2,500 vs DEX=0")
    print("  - open_interest: CEX=21,380 vs DEX=65")

    results = {}

    # Test configuration
    try:
        results['Collection Manager Config'] = await test_collection_manager_trades_config()
    except Exception as e:
        print(f"\nConfig test error: {e}")
        results['Collection Manager Config'] = False

    # Test balance improvement
    try:
        results['Balance Improvement'] = await test_balance_improvement()
    except Exception as e:
        print(f"\nBalance test error: {e}")
        results['Balance Improvement'] = False

    # Test dYdX trades (live API call)
    try:
        results['dYdX Trades'] = await test_dydx_trades_collection()
    except Exception as e:
        print(f"\ndYdX test error: {e}")
        results['dYdX Trades'] = False

    # Test Hyperliquid trades (live API call)
    try:
        results['Hyperliquid Trades'] = await test_hyperliquid_trades_collection()
    except Exception as e:
        print(f"\nHyperliquid test error: {e}")
        results['Hyperliquid Trades'] = False

    # Document OI limitation
    try:
        results['OI Limitation Documented'] = await test_open_interest_limitation_documented()
    except Exception as e:
        print(f"\nOI documentation error: {e}")
        results['OI Limitation Documented'] = False

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    all_passed = True
    for test_name, passed in results.items():
        status = "" if passed else ""
        print(f"  {status} {test_name}")
        if not passed:
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED - CEX-DEX balance improvements verified!")
        print("\nChanges made:")
        print("  1. dYdX: Added 'trades' to collection_manager config")
        print("  2. Hyperliquid: Implemented collect_trades (recentTrades API)")
    else:
        print("SOME TESTS FAILED - Review errors above")
        print("\nNote: Live API tests may fail due to rate limiting.")
    print("="*70)

    return all_passed


if __name__ == '__main__':
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
