"""
Test Options and DVOL Data Collection

Tests for the enhanced Deribit collector that now supports:
1. Historical DVOL (Deribit Volatility Index) data collection
2. Historical option trades with IV
3. Greeks calculation from IV

These features are CRITICAL for Strategy 4: Options Vol Surface Arbitrage
which requires historical volatility data for 2022-2024.
"""

import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd


async def test_dvol_collection():
    """Test historical DVOL data collection from Deribit."""
    print("\n" + "="*70)
    print("TEST 1: Historical DVOL Collection")
    print("="*70)

    from data_collection.options.deribit_collector import DeribitCollector

    async with DeribitCollector() as collector:
        # Test for 30 days of data
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)

        print(f"\nFetching DVOL for BTC and ETH ({start_date.date()} to {end_date.date()})")

        df = await collector.collect_dvol(
            symbols=['BTC', 'ETH'],
            start_date=start_date,
            end_date=end_date,
            resolution='1D'
        )

        if df.empty:
            print(" FAILED: No DVOL data returned")
            return False

        print(f"\n SUCCESS: Collected {len(df)} DVOL records")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"\nUnique underlyings: {df['underlying'].unique().tolist()}")

        # Check BTC DVOL values (should be between 20-200 typically)
        btc_df = df[df['underlying'] == 'BTC']
        if not btc_df.empty:
            print(f"\nBTC DVOL stats:")
            print(f"  Min: {btc_df['close'].min():.2f}%")
            print(f"  Max: {btc_df['close'].max():.2f}%")
            print(f"  Mean: {btc_df['close'].mean():.2f}%")

        print("\nSample data:")
        print(df.head())

        return True


async def test_options_chain():
    """Test current options chain snapshot collection (with Greeks)."""
    print("\n" + "="*70)
    print("TEST 2: Options Chain Snapshot (with Greeks)")
    print("="*70)

    from data_collection.options.deribit_collector import DeribitCollector

    async with DeribitCollector() as collector:
        print("\nFetching current BTC option chain...")

        df = await collector.fetch_option_chain(currency='BTC')

        if df.empty:
            print(" FAILED: No options data returned")
            return False

        print(f"\n SUCCESS: Fetched {len(df)} option contracts")

        # Check for Greeks
        greeks_cols = ['delta', 'gamma', 'vega', 'theta', 'rho']
        available_greeks = [c for c in greeks_cols if c in df.columns]

        print(f"\nGreeks available: {available_greeks}")
        print(f"IV column: {'mark_iv' in df.columns}")

        if 'delta' in df.columns:
            # Sample ATM options
            atm = df[(df['delta'].abs() >= 0.4) & (df['delta'].abs() <= 0.6)]
            print(f"\nATM options (|delta| 0.4-0.6): {len(atm)}")
            if not atm.empty:
                print("\nSample ATM option:")
                print(atm[['instrument_name', 'strike', 'option_type', 'mark_iv', 'delta', 'gamma', 'vega']].head(3))

        return True


async def test_greeks_calculation():
    """Test Black-Scholes Greeks calculation utility."""
    print("\n" + "="*70)
    print("TEST 3: Greeks Calculation Utility")
    print("="*70)

    from data_collection.options.deribit_collector import DeribitCollector

    # Test parameters: BTC at $100k, $100k strike, 30 days to expiry, 80% IV
    spot = 100000
    strike = 100000
    time_to_expiry = 30 / 365  # 30 days in years
    iv = 0.80  # 80% annualized volatility

    print(f"\nTest parameters:")
    print(f"  Spot: ${spot:,}")
    print(f"  Strike: ${strike:,}")
    print(f"  Time to expiry: {30} days")
    print(f"  IV: {iv*100}%")

    # Calculate Greeks for call
    call_greeks = DeribitCollector.calculate_greeks_from_iv(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        iv=iv,
        option_type='call'
    )

    print(f"\nCall Greeks:")
    for k, v in call_greeks.items():
        print(f"  {k}: {v}")

    # Calculate Greeks for put
    put_greeks = DeribitCollector.calculate_greeks_from_iv(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        iv=iv,
        option_type='put'
    )

    print(f"\nPut Greeks:")
    for k, v in put_greeks.items():
        print(f"  {k}: {v}")

    # Validate expected values
    # ATM call delta should be ~0.5 (with risk-free rate, can be 0.5-0.6)
    if 0.40 <= call_greeks['delta'] <= 0.60:
        print(f"\n ATM Call delta in range [0.4-0.6]: {call_greeks['delta']:.4f}")
    else:
        print(f"\n Call delta unexpected: {call_greeks['delta']:.4f}")
        return False

    # ATM put delta should be ~-0.5 (with risk-free rate, can be -0.6 to -0.4)
    if -0.60 <= put_greeks['delta'] <= -0.40:
        print(f" ATM Put delta ~-0.5: {put_greeks['delta']:.4f}")
    else:
        print(f" Put delta unexpected: {put_greeks['delta']:.4f}")
        return False

    # Gamma should be same for call and put
    if abs(call_greeks['gamma'] - put_greeks['gamma']) < 0.0001:
        print(f" Gamma same for call/put: {call_greeks['gamma']:.8f}")
    else:
        print(f" Gamma mismatch")
        return False

    # Vega should be same for call and put
    if abs(call_greeks['vega'] - put_greeks['vega']) < 0.01:
        print(f" Vega same for call/put: {call_greeks['vega']:.4f}")
    else:
        print(f" Vega mismatch")
        return False

    return True


async def test_collection_manager_dvol():
    """Test DVOL collection through collection manager."""
    print("\n" + "="*70)
    print("TEST 4: Collection Manager DVOL Support")
    print("="*70)

    from data_collection.collection_manager import CollectionManager, COLLECTOR_CONFIGS

    # Check if DVOL is in Deribit's supported data types
    deribit_config = COLLECTOR_CONFIGS.get('deribit')
    if deribit_config and 'dvol' in deribit_config.supported_data_types:
        print(f"\n Deribit config includes 'dvol' in supported_data_types")
        print(f"   Supported types: {deribit_config.supported_data_types}")
    else:
        print(f"\n Deribit config does NOT include 'dvol'")
        return False

    # Test collection through manager
    manager = CollectionManager()

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=7)

    print(f"\nCollecting DVOL via manager ({start_date.date()} to {end_date.date()})...")

    results = await manager.collect_data_type(
        data_type='dvol',
        venues=['deribit'],
        symbols=['BTC', 'ETH'],
        start_date=start_date,
        end_date=end_date,
        resolution='1D'
    )

    if not results:
        print(" No results returned from manager")
        return False

    for result in results:
        from data_collection.collection_manager import CollectionStatus
        if result.status == CollectionStatus.COMPLETED:
            print(f"\n SUCCESS: {result.venue} collected {result.total_records} DVOL records")
            if result.data is not None and not result.data.empty:
                print(f"   Date range: {result.data['timestamp'].min()} to {result.data['timestamp'].max()}")
        else:
            print(f"\n FAILED: {result.venue} - {result.error}")
            return False

    return True


async def main():
    """Run all tests."""
    print("="*70)
    print("OPTIONS AND DVOL DATA COLLECTION TESTS")
    print("="*70)
    print(f"\nTest Time: {datetime.now(timezone.utc).isoformat()}")
    print("\nThese tests validate features critical for:")
    print("  Strategy 4: Options Vol Surface Arbitrage")
    print("  Project Requirement: Historical IV and Greeks 2022-2024")

    results = {}

    # Run tests
    try:
        results['Greeks Calculation'] = await test_greeks_calculation()
    except Exception as e:
        print(f"\nGreeks test error: {e}")
        results['Greeks Calculation'] = False

    try:
        results['DVOL Collection'] = await test_dvol_collection()
    except Exception as e:
        print(f"\nDVOL test error: {e}")
        results['DVOL Collection'] = False

    try:
        results['Options Chain'] = await test_options_chain()
    except Exception as e:
        print(f"\nOptions chain test error: {e}")
        results['Options Chain'] = False

    try:
        results['Collection Manager DVOL'] = await test_collection_manager_dvol()
    except Exception as e:
        print(f"\nCollection manager test error: {e}")
        results['Collection Manager DVOL'] = False

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
        print("ALL TESTS PASSED - Options/DVOL collection is working!")
    else:
        print("SOME TESTS FAILED - Review errors above")
    print("="*70)

    return all_passed


if __name__ == '__main__':
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
