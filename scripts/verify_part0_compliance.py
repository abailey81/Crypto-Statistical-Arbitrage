#!/usr/bin/env python3
"""
Part 0 Compliance Verification Script
Checks all data files against project requirements
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import json

# Configuration
DATA_DIR = Path("/Users/a_bailey8/Desktop/crypto-statarb-multiverse 7/data/processed")
REQUIRED_START_DATE = "2022-01-01"
REQUIRED_END_DATE = "2024-12-31"
MIN_YEARS_COVERAGE = 2

# Strategy requirements from the project
STRATEGY_REQUIREMENTS = {
    "Strategy 1 - Perpetual Funding Rate Arb": {
        "must_have": ["funding_rates", "ohlcv"],
        "nice_to_have": ["open_interest"],
        "min_venues": 2,
        "min_assets": ["BTC", "ETH"],
        "description": "Funding rates (8-hour intervals) for at least 3 coins, at least 2 venues"
    },
    "Strategy 2 - Altcoin Statistical Arb": {
        "must_have": ["ohlcv", "market_cap"],
        "nice_to_have": ["tvl", "volume"],
        "min_assets_count": 20,
        "description": "OHLCV data for at least 20 altcoins, hourly frequency, market cap data"
    },
    "Strategy 3 - BTC Futures Curve Trading": {
        "must_have": ["ohlcv", "funding_rates"],
        "nice_to_have": ["open_interest"],
        "min_venues": 2,
        "description": "BTC spot prices, BTC futures prices for at least 2 contract maturities"
    },
    "Strategy 4 - Vol Surface Arb (or Alternative)": {
        "must_have": ["options", "dvol", "ohlcv"],
        "alternatives": ["Cross-DEX Arb", "Stablecoin Depeg"],
        "description": "Deribit BTC options (all strikes, expiries) OR alternative strategy"
    }
}

def analyze_parquet_file(file_path):
    """Analyze a single parquet file for compliance metrics"""
    try:
        df = pd.read_parquet(file_path)

        # Find timestamp column
        timestamp_cols = ['timestamp', 'time', 'datetime', 'date', 'funding_time']
        timestamp_col = None
        for col in timestamp_cols:
            if col in df.columns:
                timestamp_col = col
                break

        result = {
            "file": str(file_path),
            "records": len(df),
            "columns": list(df.columns),
            "file_size_mb": round(os.path.getsize(file_path) / (1024*1024), 2)
        }

        if timestamp_col and len(df) > 0:
            # Convert to datetime
            ts = pd.to_datetime(df[timestamp_col])
            result["start_date"] = str(ts.min().date())
            result["end_date"] = str(ts.max().date())

            # Calculate coverage in days
            date_range = (ts.max() - ts.min()).days
            result["date_range_days"] = date_range
            result["years_coverage"] = round(date_range / 365, 2)

            # Check 2022-2024 requirement
            required_start = pd.Timestamp(REQUIRED_START_DATE)
            required_end = pd.Timestamp(REQUIRED_END_DATE)

            result["meets_2022_2024"] = ts.min() <= pd.Timestamp("2022-12-31") and date_range >= 365
            result["covers_2_years"] = date_range >= 730
        else:
            result["start_date"] = "N/A"
            result["end_date"] = "N/A"
            result["date_range_days"] = 0
            result["years_coverage"] = 0
            result["meets_2022_2024"] = False
            result["covers_2_years"] = False

        # Check for venues
        if 'venue' in df.columns:
            result["venues"] = df['venue'].unique().tolist()
            result["venue_count"] = len(result["venues"])
        elif 'exchange' in df.columns:
            result["venues"] = df['exchange'].unique().tolist()
            result["venue_count"] = len(result["venues"])
        else:
            result["venues"] = []
            result["venue_count"] = 0

        # Check for symbols/assets
        symbol_cols = ['symbol', 'asset', 'token', 'underlying']
        for col in symbol_cols:
            if col in df.columns:
                symbols = df[col].unique().tolist()
                result["symbols"] = symbols[:50]  # Limit to 50
                result["symbol_count"] = len(symbols)
                break
        else:
            result["symbols"] = []
            result["symbol_count"] = 0

        return result
    except Exception as e:
        return {
            "file": str(file_path),
            "error": str(e),
            "records": 0
        }

def categorize_data_type(file_path):
    """Categorize file by data type"""
    path_str = str(file_path).lower()

    if 'funding' in path_str:
        return 'funding_rates'
    elif 'ohlcv' in path_str or 'price' in path_str:
        return 'ohlcv'
    elif 'open_interest' in path_str:
        return 'open_interest'
    elif 'option' in path_str:
        return 'options'
    elif 'dvol' in path_str:
        return 'dvol'
    elif 'liquidation' in path_str:
        return 'liquidations'
    elif 'trade' in path_str:
        return 'trades'
    elif 'market_cap' in path_str:
        return 'market_cap'
    elif 'tvl' in path_str:
        return 'tvl'
    elif 'pool' in path_str:
        return 'pool_data'
    elif 'swap' in path_str:
        return 'swaps'
    elif 'route' in path_str:
        return 'routes'
    elif 'volume' in path_str:
        return 'volume'
    elif 'stablecoin' in path_str:
        return 'stablecoins'
    elif 'yield' in path_str:
        return 'yields'
    elif 'sentiment' in path_str:
        return 'sentiment'
    elif 'on_chain' in path_str:
        return 'on_chain_metrics'
    elif 'token_balance' in path_str:
        return 'token_balances'
    elif 'order' in path_str:
        return 'orders'
    elif 'subgraph' in path_str:
        return 'subgraph_data'
    elif 'position' in path_str:
        return 'positions'
    else:
        return 'other'

def identify_venue(file_path):
    """Identify venue from file path"""
    path_str = str(file_path).lower()
    venues = [
        'binance', 'bybit', 'okx', 'coinbase', 'kraken', 'deribit', 'cme',
        'hyperliquid', 'dydx', 'gmx', 'aevo', 'drift', 'vertex',
        'uniswap', 'curve', 'geckoterminal', 'dexscreener', 'coingecko',
        'coinalyze', 'cryptocompare', 'santiment', 'defillama', 'cowswap',
        'oneinch', 'zerox', 'jupiter', 'covalent', 'thegraph', 'lunarcrush'
    ]

    for venue in venues:
        if venue in path_str:
            return venue
    return 'unknown'

def classify_venue_type(venue):
    """Classify venue as CEX, Hybrid, or DEX"""
    cex = ['binance', 'bybit', 'okx', 'coinbase', 'kraken', 'deribit', 'cme']
    hybrid = ['hyperliquid', 'dydx', 'gmx', 'aevo', 'drift', 'vertex']
    dex = ['uniswap', 'curve', 'cowswap', 'oneinch', 'zerox', 'jupiter']
    aggregators = ['coingecko', 'coinalyze', 'cryptocompare', 'geckoterminal', 'dexscreener']
    on_chain = ['santiment', 'defillama', 'covalent', 'thegraph', 'lunarcrush']

    if venue in cex:
        return 'CEX'
    elif venue in hybrid:
        return 'Hybrid'
    elif venue in dex:
        return 'DEX'
    elif venue in aggregators:
        return 'Aggregator'
    elif venue in on_chain:
        return 'On-Chain'
    else:
        return 'Unknown'

def main():
    print("=" * 80)
    print("PART 0 COMPLIANCE VERIFICATION REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().isoformat()}")
    print(f"Required Coverage: {REQUIRED_START_DATE} to {REQUIRED_END_DATE} (minimum 2 years)")
    print("=" * 80)

    # Find all parquet files
    parquet_files = list(DATA_DIR.rglob("*.parquet"))
    print(f"\nFound {len(parquet_files)} parquet files to analyze...\n")

    # Analyze all files
    results = []
    for pf in parquet_files:
        result = analyze_parquet_file(pf)
        result["data_type"] = categorize_data_type(pf)
        result["venue"] = identify_venue(pf)
        result["venue_type"] = classify_venue_type(result["venue"])
        results.append(result)

    # Group by data type
    data_by_type = {}
    for r in results:
        dt = r["data_type"]
        if dt not in data_by_type:
            data_by_type[dt] = []
        data_by_type[dt].append(r)

    # Summary by data type
    print("\n" + "=" * 80)
    print("DATA COVERAGE BY TYPE")
    print("=" * 80)

    total_records = 0
    compliant_types = 0

    for data_type, files in sorted(data_by_type.items()):
        total_recs = sum(f.get("records", 0) for f in files)
        total_records += total_recs

        # Get date ranges
        start_dates = [f.get("start_date") for f in files if f.get("start_date") and f.get("start_date") != "N/A"]
        end_dates = [f.get("end_date") for f in files if f.get("end_date") and f.get("end_date") != "N/A"]

        earliest = min(start_dates) if start_dates else "N/A"
        latest = max(end_dates) if end_dates else "N/A"

        # Check compliance
        meets_req = any(f.get("meets_2022_2024", False) for f in files)
        covers_2yrs = any(f.get("covers_2_years", False) for f in files)

        if meets_req or covers_2yrs:
            compliant_types += 1
            status = "COMPLIANT"
        else:
            status = "NON-COMPLIANT"

        venues = list(set(f.get("venue") for f in files if f.get("venue") != "unknown"))
        venue_types = list(set(f.get("venue_type") for f in files if f.get("venue_type") != "Unknown"))

        print(f"\n{data_type.upper()}")
        print(f"  Files: {len(files)}")
        print(f"  Total Records: {total_recs:,}")
        print(f"  Date Range: {earliest} to {latest}")
        print(f"  Venues ({len(venues)}): {', '.join(venues[:8])}")
        print(f"  Venue Types: {', '.join(venue_types)}")
        print(f"  Status: {status}")

    # Detailed compliance check for each strategy
    print("\n" + "=" * 80)
    print("STRATEGY DATA REQUIREMENTS CHECK")
    print("=" * 80)

    for strategy_name, reqs in STRATEGY_REQUIREMENTS.items():
        print(f"\n{strategy_name}")
        print("-" * 60)
        print(f"Description: {reqs['description']}")
        print("\nMust-Have Data:")

        all_must_have = True
        for data_req in reqs.get("must_have", []):
            files = data_by_type.get(data_req, [])
            has_data = len(files) > 0 and sum(f.get("records", 0) for f in files) > 100
            status = "YES" if has_data else "NO"
            if not has_data:
                all_must_have = False

            total_recs = sum(f.get("records", 0) for f in files)
            print(f"  - {data_req}: {status} ({total_recs:,} records)")

        print("\nNice-To-Have Data:")
        for data_req in reqs.get("nice_to_have", []):
            files = data_by_type.get(data_req, [])
            has_data = len(files) > 0 and sum(f.get("records", 0) for f in files) > 0
            status = "YES" if has_data else "NO"
            total_recs = sum(f.get("records", 0) for f in files)
            print(f"  - {data_req}: {status} ({total_recs:,} records)")

        # Check venue diversity
        all_venues = []
        for data_req in reqs.get("must_have", []):
            files = data_by_type.get(data_req, [])
            for f in files:
                if f.get("venue") and f.get("venue") != "unknown":
                    all_venues.append(f["venue"])
        unique_venues = list(set(all_venues))

        min_venues = reqs.get("min_venues", 1)
        venue_status = "YES" if len(unique_venues) >= min_venues else "NO"
        print(f"\nVenue Requirement: {min_venues}+ venues")
        print(f"  Venues Found ({len(unique_venues)}): {', '.join(unique_venues[:10])}")
        print(f"  Meets Requirement: {venue_status}")

        overall = "COMPLIANT" if all_must_have and len(unique_venues) >= min_venues else "NEEDS ATTENTION"
        print(f"\nOverall Status: {overall}")

    # Venue coverage analysis
    print("\n" + "=" * 80)
    print("VENUE COVERAGE ANALYSIS")
    print("=" * 80)

    venue_summary = {}
    for r in results:
        venue = r.get("venue", "unknown")
        if venue not in venue_summary:
            venue_summary[venue] = {
                "files": 0,
                "records": 0,
                "data_types": set(),
                "venue_type": r.get("venue_type"),
                "earliest": None,
                "latest": None
            }

        venue_summary[venue]["files"] += 1
        venue_summary[venue]["records"] += r.get("records", 0)
        venue_summary[venue]["data_types"].add(r.get("data_type"))

        start = r.get("start_date")
        end = r.get("end_date")
        if start and start != "N/A":
            if venue_summary[venue]["earliest"] is None or start < venue_summary[venue]["earliest"]:
                venue_summary[venue]["earliest"] = start
        if end and end != "N/A":
            if venue_summary[venue]["latest"] is None or end > venue_summary[venue]["latest"]:
                venue_summary[venue]["latest"] = end

    print(f"\n{'Venue':<20} {'Type':<12} {'Records':>15} {'Date Range':<25} {'Data Types'}")
    print("-" * 100)

    for venue, info in sorted(venue_summary.items(), key=lambda x: x[1]["records"], reverse=True):
        if venue == "unknown":
            continue
        date_range = f"{info['earliest'] or 'N/A'} to {info['latest'] or 'N/A'}"
        data_types = ", ".join(sorted(info["data_types"]))[:30]
        print(f"{venue:<20} {info['venue_type']:<12} {info['records']:>15,} {date_range:<25} {data_types}")

    # Final summary
    print("\n" + "=" * 80)
    print("COMPLIANCE SUMMARY")
    print("=" * 80)

    print(f"\nTotal Parquet Files: {len(parquet_files)}")
    print(f"Total Records: {total_records:,}")
    print(f"Data Types with 2+ Years Coverage: {compliant_types}/{len(data_by_type)}")

    # Check key requirements
    requirements_met = []
    requirements_not_met = []

    # Funding rates check
    funding_files = data_by_type.get("funding_rates", [])
    if len(funding_files) > 0 and sum(f.get("records", 0) for f in funding_files) > 10000:
        requirements_met.append("Funding Rate Data (500k+ records)")
    else:
        requirements_not_met.append("Funding Rate Data")

    # OHLCV check
    ohlcv_files = data_by_type.get("ohlcv", [])
    if len(ohlcv_files) > 0 and sum(f.get("records", 0) for f in ohlcv_files) > 100000:
        requirements_met.append("OHLCV Data (5M+ records)")
    else:
        requirements_not_met.append("OHLCV Data")

    # Options check
    options_files = data_by_type.get("options", [])
    options_recs = sum(f.get("records", 0) for f in options_files)
    if options_recs > 1000:
        requirements_met.append(f"Options Data ({options_recs:,} records)")
    else:
        requirements_not_met.append(f"Options Data (only {options_recs} records - consider alternative strategy)")

    # CEX vs DEX check
    cex_venues = sum(1 for v, info in venue_summary.items() if info.get("venue_type") == "CEX")
    hybrid_venues = sum(1 for v, info in venue_summary.items() if info.get("venue_type") == "Hybrid")
    dex_venues = sum(1 for v, info in venue_summary.items() if info.get("venue_type") == "DEX")

    if cex_venues >= 3 and (hybrid_venues + dex_venues) >= 2:
        requirements_met.append(f"Multi-Venue Coverage ({cex_venues} CEX, {hybrid_venues} Hybrid, {dex_venues} DEX)")
    else:
        requirements_not_met.append("Multi-Venue Coverage")

    # Cross-validation capability
    funding_venues = set(f.get("venue") for f in funding_files if f.get("venue") != "unknown")
    if len(funding_venues) >= 2:
        requirements_met.append(f"Cross-Validation Capability ({len(funding_venues)} funding venues)")
    else:
        requirements_not_met.append("Cross-Validation Capability")

    print("\n REQUIREMENTS MET:")
    for req in requirements_met:
        print(f"  - {req}")

    if requirements_not_met:
        print("\n REQUIREMENTS NEEDING ATTENTION:")
        for req in requirements_not_met:
            print(f"  - {req}")

    # Calculate overall score
    total_reqs = len(requirements_met) + len(requirements_not_met)
    score = len(requirements_met) / total_reqs * 100 if total_reqs > 0 else 0

    print(f"\n{'=' * 80}")
    print(f"OVERALL COMPLIANCE SCORE: {score:.1f}%")
    print(f"{'=' * 80}")

    # Save detailed results
    output = {
        "generated": datetime.now().isoformat(),
        "total_files": len(parquet_files),
        "total_records": total_records,
        "data_by_type": {k: len(v) for k, v in data_by_type.items()},
        "requirements_met": requirements_met,
        "requirements_not_met": requirements_not_met,
        "compliance_score": score,
        "venue_summary": {k: {**v, "data_types": list(v["data_types"])} for k, v in venue_summary.items()}
    }

    output_path = DATA_DIR / "compliance_verification.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    main()
