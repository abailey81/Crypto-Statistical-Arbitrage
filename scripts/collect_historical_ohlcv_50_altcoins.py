"""
Historical OHLCV Data Collection Script - 200+ Altcoins
========================================================

This script collects 2020-present historical hourly OHLCV data for 200+ altcoins
to satisfy Strategy 2 (Altcoin Statistical Arbitrage) requirements.

Project Requirement (Page 9):
- OHLCV data for at least 20 altcoins
- Hourly frequency minimum
- Date range: 2022-2024 minimum (we exceed with 2020-present)

This script uses the centralized SymbolUniverse configuration (config/symbols.yaml)
which provides 200+ tokens (10x the project requirement) across categories:
- L1 Blockchains (25 tokens)
- L2 Solutions (20 tokens)
- DeFi Protocols (40 tokens)
- Infrastructure (25 tokens)
- Gaming/Metaverse (20 tokens)
- AI/Data (20 tokens)
- Major Altcoins (20 tokens)
- Real World Assets (15 tokens)
- Memecoins (15 tokens)
- Emerging (23 tokens)

Author: Tamer Atesyakar
Date: 2026-02-02
"""

import asyncio
import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_collection.cex.binance_collector import BinanceCollector
from data_collection.utils.symbol_universe import SymbolUniverse, get_symbol_universe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'logs' / 'historical_ohlcv_collection.log')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CENTRALIZED SYMBOL UNIVERSE - 200+ Tokens (10x Project Requirement)
# =============================================================================
# Uses config/symbols.yaml via SymbolUniverse utility for centralized management

# Load symbols from centralized configuration
universe = get_symbol_universe()
ALL_SYMBOLS = universe.get_ohlcv_symbols()

# Print statistics
stats = universe.get_statistics()
print(f"=" * 60)
print("SYMBOL UNIVERSE CONFIGURATION")
print(f"=" * 60)
print(f"Total unique symbols: {stats['total_unique_symbols']}")
print(f"Project requirement: {stats['requirement']}")
print(f"Exceeded by: {stats['exceeded_by']}")
print(f"Categories: {stats['total_categories']}")
for cat, count in stats['categories'].items():
    print(f"  - {cat}: {count}")
print(f"=" * 60)


async def collect_historical_ohlcv():
    """
    Collect 2022-2024 historical hourly OHLCV data for all altcoins.
    """
    # Dynamic date range: 2020-01-01 to present
    current_year = datetime.now().year
    current_date = datetime.now().strftime('%Y-%m-%d')
    years_to_collect = list(range(2020, current_year + 1))

    logger.info("=" * 80)
    logger.info("HISTORICAL OHLCV COLLECTION - 50+ ALTCOINS")
    logger.info("=" * 80)
    logger.info(f"Symbols: {len(ALL_SYMBOLS)}")
    logger.info(f"Date range: 2020-01-01 to {current_date}")
    logger.info(f"Years to collect: {years_to_collect}")
    logger.info(f"Timeframe: 1h (hourly)")
    logger.info("=" * 80)

    # Configuration - conservative rate limiting for large historical fetch
    config = {
        'rate_limit': 800,  # Conservative to avoid rate limits
        'timeout': 60,
        'max_retries': 5,
    }

    # Output paths
    output_dir = project_root / 'data' / 'raw' / 'cex' / 'binance'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Backup existing file
    existing_file = output_dir / 'ohlcv_1h.parquet'
    if existing_file.exists():
        backup_file = output_dir / f'ohlcv_1h_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet'
        import shutil
        shutil.copy(existing_file, backup_file)
        logger.info(f"Backed up existing file to: {backup_file}")

    all_data = []
    failed_symbols = []

    # Process in batches to manage memory and rate limits
    batch_size = 10
    total_batches = (len(ALL_SYMBOLS) + batch_size - 1) // batch_size

    async with BinanceCollector(config) as collector:
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(ALL_SYMBOLS))
            batch_symbols = ALL_SYMBOLS[start_idx:end_idx]

            logger.info(f"\n{'='*60}")
            logger.info(f"BATCH {batch_idx + 1}/{total_batches}: {batch_symbols}")
            logger.info(f"{'='*60}")

            try:
                # Collect 2020-present data
                # Split into yearly chunks for reliability
                for year in years_to_collect:
                    start_date = f"{year}-01-01"
                    # Use current date for current year, otherwise Dec 31
                    end_date = current_date if year == current_year else f"{year}-12-31"

                    logger.info(f"Collecting {year} data for batch {batch_idx + 1}...")

                    try:
                        df = await collector.fetch_ohlcv(
                            symbols=batch_symbols,
                            timeframe='1h',
                            start_date=start_date,
                            end_date=end_date,
                            market_type='futures'
                        )

                        if not df.empty:
                            all_data.append(df)
                            logger.info(f"  {year}: Collected {len(df):,} records")
                        else:
                            logger.warning(f"  {year}: No data returned")

                    except Exception as e:
                        logger.error(f"  {year} collection error: {e}")

                    # Small delay between years
                    await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} error: {e}")
                failed_symbols.extend(batch_symbols)

            # Progress update
            stats = collector.get_collection_stats()
            logger.info(f"Progress: {end_idx}/{len(ALL_SYMBOLS)} symbols complete")
            logger.info(f"Stats: {stats}")

            # Delay between batches to respect rate limits
            if batch_idx < total_batches - 1:
                logger.info("Waiting 5 seconds before next batch...")
                await asyncio.sleep(5)

    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # Remove duplicates
        combined_df = combined_df.drop_duplicates(
            subset=['timestamp', 'symbol'],
            keep='last'
        ).sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        logger.info(f"\n{'='*80}")
        logger.info("COLLECTION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total records: {len(combined_df):,}")
        logger.info(f"Unique symbols: {combined_df['symbol'].nunique()}")
        logger.info(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")

        # Verify 2020 coverage
        symbols_with_2020 = []
        for sym in combined_df['symbol'].unique():
            sym_df = combined_df[combined_df['symbol'] == sym]
            min_date = sym_df['timestamp'].min()
            if min_date.year <= 2020:
                symbols_with_2020.append(sym)

        logger.info(f"\nSymbols with 2020 coverage: {len(symbols_with_2020)}")
        logger.info(f"Symbols: {sorted(symbols_with_2020)}")

        # Save to parquet
        output_file = output_dir / 'ohlcv_1h.parquet'
        combined_df.to_parquet(output_file, index=False)
        logger.info(f"\nSaved to: {output_file}")

        # Save summary
        summary = {
            'total_records': len(combined_df),
            'unique_symbols': combined_df['symbol'].nunique(),
            'symbols_with_2020': len(symbols_with_2020),
            'date_range_start': str(combined_df['timestamp'].min()),
            'date_range_end': str(combined_df['timestamp'].max()),
            'symbols_collected': sorted(combined_df['symbol'].unique().tolist()),
            'symbols_with_2020_coverage': sorted(symbols_with_2020),
            'failed_symbols': failed_symbols,
        }

        summary_file = output_dir / 'ohlcv_collection_summary.json'
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved summary to: {summary_file}")

        return combined_df
    else:
        logger.error("No data collected!")
        return pd.DataFrame()


async def verify_existing_and_fill_gaps():
    """
    Verify existing data and only collect what's missing.
    """
    output_file = project_root / 'data' / 'raw' / 'cex' / 'binance' / 'ohlcv_1h.parquet'

    if output_file.exists():
        existing_df = pd.read_parquet(output_file)
        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'], format='ISO8601')

        existing_symbols = set(existing_df['symbol'].unique())
        symbols_with_2022 = set()

        for sym in existing_symbols:
            sym_df = existing_df[existing_df['symbol'] == sym]
            if sym_df['timestamp'].min().year <= 2020:
                symbols_with_2022.add(sym)

        # Find symbols that need collection
        needed_symbols = [s for s in ALL_SYMBOLS if s not in symbols_with_2022]

        logger.info(f"Existing symbols with 2022 data: {len(symbols_with_2022)}")
        logger.info(f"Symbols needing collection: {len(needed_symbols)}")

        if needed_symbols:
            logger.info(f"Will collect: {needed_symbols}")
            return needed_symbols
        else:
            logger.info("All symbols already have 2022 coverage!")
            return []
    else:
        logger.info("No existing file found - collecting all symbols")
        return ALL_SYMBOLS


if __name__ == '__main__':
    # Create logs directory
    (project_root / 'logs').mkdir(exist_ok=True)

    print("=" * 80)
    print("HISTORICAL OHLCV COLLECTION - STRATEGY 2 DATA GAP FIX")
    print("=" * 80)
    print(f"Collecting 2020-present hourly OHLCV data for {len(ALL_SYMBOLS)} altcoins")
    print(f"Symbols: {ALL_SYMBOLS}")
    print("=" * 80)

    # Run collection
    asyncio.run(collect_historical_ohlcv())
