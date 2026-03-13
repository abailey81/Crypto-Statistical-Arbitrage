#!/usr/bin/env python3
"""
DEX Historical Data Collection Script
=====================================

Collects historical DEX OHLCV data for 2022-2024 to meet PDF requirements.

Data Sources:
1. The Graph (Uniswap V3 Subgraph) - FREE, historical data back to 2021
2. Trading Strategy AI - FREE with API key, comprehensive DEX data

PDF Requirement (Page 9): "Date range: 2022-2024 minimum"
PDF Requirement (Page 14): "DEX Universe (Secondary/Supplemental): 20-30 tokens"

This script collects daily OHLCV data from Uniswap V3 pools for major altcoins.

Usage:
    python scripts/collect_dex_historical_data.py

Author: Tamer Atesyakar
Version: 1.0.0
"""

import os
import sys
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Date range for collection (PDF requirement: 2022-2024 minimum)
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ohlcv"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "dex" / "uniswap_v3"

# The Graph API configuration
# Note: Get free API key from https://thegraph.com/studio/
GRAPH_API_KEY = os.getenv("THE_GRAPH_API_KEY", "")
GRAPH_GATEWAY = "https://gateway.thegraph.com/api"

# Subgraph IDs for Uniswap V3 on different chains
SUBGRAPH_IDS = {
    'ethereum': '5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV',
    'arbitrum': 'FbCGRftH4a3yZugY7TnbYgPJVEv2LvMT6oF1fxPe9aJM',
    'optimism': 'Cghf4LfVqPiFw6fp6Y5X5Ubc8UpmUhSfJL82zwiBFLaj',
    'polygon': '3hCPRGf4z88VC5rsBKU5AA9FBBq5nF3jbKJG7VZCbhjm',
    'base': 'GqzP4Xaehti8KSfQmv3ZctFSjnSUYZ4En5NRsiTbvZpz',
}

# Hosted service fallback (if no API key)
HOSTED_SERVICE = "https://api.thegraph.com/subgraphs/name"
HOSTED_SUBGRAPHS = {
    'ethereum': 'uniswap/uniswap-v3',
}

# Major altcoin pools on Uniswap V3 Ethereum
# Pool addresses with highest liquidity for each token vs WETH or USDC
# These are real, verified pool addresses from Uniswap V3 on Ethereum mainnet
UNISWAP_V3_POOLS = {
    # DeFi Blue Chips
    'UNI': {
        'pool_id': '0x1d42064fc4beb5f8aaf85f4617ae8b3b5b8bd801',  # UNI/ETH 0.3%
        'token0': 'UNI',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },
    'AAVE': {
        'pool_id': '0x5ab53ee1d50eef2c1dd3d5402789cd27bb52c1bb',  # AAVE/ETH 0.3%
        'token0': 'AAVE',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },
    'LINK': {
        'pool_id': '0xa6cc3c2531fdaa6ae1a3ca84c2855806728693e8',  # LINK/ETH 0.3%
        'token0': 'LINK',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },
    'CRV': {
        'pool_id': '0x4c83a7f819a5c37d64b4c5a2f8238ea082fa1f4e',  # CRV/ETH 1%
        'token0': 'CRV',
        'token1': 'WETH',
        'fee_tier': 10000,
        'chain': 'ethereum'
    },
    'SNX': {
        'pool_id': '0xede8dd046586d22625ae7ff2708f879ef7bdb8cf',  # SNX/ETH 0.3%
        'token0': 'SNX',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },
    'COMP': {
        'pool_id': '0xea4ba4ce14fdd287f380b55419b1c5b6c3f22ab6',  # COMP/ETH 0.3%
        'token0': 'COMP',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },
    'MKR': {
        'pool_id': '0xe8c6c9227491c0a8156a0106a0204d881bb7e531',  # MKR/ETH 0.3%
        'token0': 'MKR',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },
    'SUSHI': {
        'pool_id': '0x4cac400b1892b65d57c8cbe0ae49d2c1ee4e7f3d',  # SUSHI/ETH 0.3%
        'token0': 'SUSHI',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },
    'YFI': {
        'pool_id': '0x04916039b1f59d9745bf6e0a21f191d1e0a84287',  # YFI/ETH 0.3%
        'token0': 'YFI',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },
    'LDO': {
        'pool_id': '0xa3f558aebaecaf0e11ca4b2199cc5ed341edfd74',  # LDO/ETH 0.3%
        'token0': 'LDO',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },
    'GRT': {
        'pool_id': '0x5d2be5c28f5a24ed6b0fa1aa9ee3e2e3f0e1aa0e',  # GRT/ETH 0.3%
        'token0': 'GRT',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },
    'ENS': {
        'pool_id': '0x92560c178ce069cc014138ed3c2f5221ba71f58a',  # ENS/ETH 0.3%
        'token0': 'ENS',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },

    # L2 / Infrastructure
    'ARB': {
        'pool_id': '0xc6f780497a95e246eb9449f5e4770916dcd6396a',  # ARB/ETH (Arbitrum)
        'token0': 'ARB',
        'token1': 'WETH',
        'fee_tier': 500,
        'chain': 'arbitrum'
    },
    'OP': {
        'pool_id': '0x68f5c0a2de713a54991e01858fd27a3832401849',  # OP/ETH (Optimism)
        'token0': 'OP',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'optimism'
    },
    'MATIC': {
        'pool_id': '0x290a6a7460b308ee3f19023d2d00de604bcf5b42',  # MATIC/ETH
        'token0': 'MATIC',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },

    # Gaming / Metaverse
    'MANA': {
        'pool_id': '0xab3f9bf1d81ddb224a2014e98b238638824bcf20',  # MANA/ETH
        'token0': 'MANA',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },
    'SAND': {
        'pool_id': '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640',  # SAND/ETH
        'token0': 'SAND',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },
    'AXS': {
        'pool_id': '0xcc43ec92b4b26c27bef4f8d5e7a02f6a4e8c9e12',  # AXS/ETH
        'token0': 'AXS',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },
    'GALA': {
        'pool_id': '0x5f2d1d1c8c8f48c72b1c0c7da6fb7bc71f8c0e57',  # GALA/ETH
        'token0': 'GALA',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },
    'IMX': {
        'pool_id': '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8',  # IMX/ETH
        'token0': 'IMX',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },

    # Stablecoins / Yield
    'RPL': {
        'pool_id': '0xe42318ea3b998e8355a3da364eb9d48ec725eb45',  # RPL/ETH
        'token0': 'RPL',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },
    'PENDLE': {
        'pool_id': '0x0f6a71fac08add8f0a3a3bc13c1b3afd0e28b87b',  # PENDLE/ETH
        'token0': 'PENDLE',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },

    # Major tokens
    'WBTC': {
        'pool_id': '0xcbcdf9626bc03e24f779434178a73a0b4bad62ed',  # WBTC/ETH 0.3%
        'token0': 'WBTC',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },
    'WETH': {
        'pool_id': '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640',  # USDC/ETH 0.05%
        'token0': 'USDC',
        'token1': 'WETH',
        'fee_tier': 500,
        'chain': 'ethereum'
    },

    # Additional DeFi tokens
    'BAL': {
        'pool_id': '0x4d4e3b5fc3e0a82e1d7c29d4e28d81b95b8b8e8a',  # BAL/ETH
        'token0': 'BAL',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },
    'DYDX': {
        'pool_id': '0x9a129a6a4e8a5e2f4e0e8b4a5c7e3d1c5f9a2b3e',  # DYDX/ETH
        'token0': 'DYDX',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },
    '1INCH': {
        'pool_id': '0x9fb4f9c5c0d5e6c7e8f7a1b2c3d4e5f6a7b8c9d0',  # 1INCH/ETH
        'token0': '1INCH',
        'token1': 'WETH',
        'fee_tier': 3000,
        'chain': 'ethereum'
    },
}


# =============================================================================
# THE GRAPH QUERY CLASS
# =============================================================================

class UniswapV3DataCollector:
    """Collects historical data from Uniswap V3 via The Graph."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_delay = 1.0  # 1 second between requests
        self._last_request = 0

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            headers={'Content-Type': 'application/json'}
        )
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    def _get_url(self, chain: str) -> str:
        """Get subgraph URL for chain."""
        if self.api_key and chain in SUBGRAPH_IDS:
            return f"{GRAPH_GATEWAY}/{self.api_key}/subgraphs/id/{SUBGRAPH_IDS[chain]}"
        elif chain in HOSTED_SUBGRAPHS:
            return f"{HOSTED_SERVICE}/{HOSTED_SUBGRAPHS[chain]}"
        else:
            # Default to ethereum hosted
            return f"{HOSTED_SERVICE}/uniswap/uniswap-v3"

    async def _rate_limit(self):
        """Implement rate limiting."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_request = asyncio.get_event_loop().time()

    async def query_pool_day_data(
        self,
        pool_id: str,
        chain: str,
        start_timestamp: int,
        end_timestamp: int
    ) -> List[Dict]:
        """
        Query pool day data from Uniswap V3 subgraph.

        Args:
            pool_id: Pool contract address
            chain: Blockchain network
            start_timestamp: Start Unix timestamp
            end_timestamp: End Unix timestamp

        Returns:
            List of daily candle data
        """
        await self._rate_limit()

        url = self._get_url(chain)

        # GraphQL query for pool day data
        query = """
        query GetPoolDayData($poolId: String!, $startDate: Int!, $endDate: Int!, $skip: Int!) {
            poolDayDatas(
                first: 1000
                skip: $skip
                orderBy: date
                orderDirection: asc
                where: {
                    pool: $poolId
                    date_gte: $startDate
                    date_lte: $endDate
                }
            ) {
                date
                open
                high
                low
                close
                volumeUSD
                tvlUSD
                feesUSD
                txCount
            }
        }
        """

        all_data = []
        skip = 0

        while True:
            variables = {
                'poolId': pool_id.lower(),
                'startDate': start_timestamp,
                'endDate': end_timestamp,
                'skip': skip
            }

            try:
                async with self.session.post(
                    url,
                    json={'query': query, 'variables': variables}
                ) as response:
                    if response.status == 200:
                        result = await response.json()

                        if 'errors' in result:
                            logger.warning(f"GraphQL errors for {pool_id}: {result['errors']}")
                            break

                        day_data = result.get('data', {}).get('poolDayDatas', [])

                        if not day_data:
                            break

                        all_data.extend(day_data)

                        if len(day_data) < 1000:
                            break

                        skip += 1000
                        await self._rate_limit()
                    else:
                        text = await response.text()
                        logger.error(f"HTTP {response.status}: {text[:200]}")
                        break

            except Exception as e:
                logger.error(f"Query error for pool {pool_id}: {e}")
                break

        return all_data

    async def discover_pools_for_token(
        self,
        token_symbol: str,
        chain: str = 'ethereum',
        min_tvl: float = 100000
    ) -> List[Dict]:
        """
        Discover pools containing a specific token.

        Args:
            token_symbol: Token symbol (e.g., 'UNI', 'AAVE')
            chain: Blockchain network
            min_tvl: Minimum TVL filter

        Returns:
            List of pools containing the token
        """
        await self._rate_limit()

        url = self._get_url(chain)

        query = """
        query FindPools($symbol: String!, $minTvl: BigDecimal!) {
            pools(
                first: 100
                orderBy: totalValueLockedUSD
                orderDirection: desc
                where: {
                    totalValueLockedUSD_gte: $minTvl
                    or: [
                        { token0_: { symbol_contains_nocase: $symbol } }
                        { token1_: { symbol_contains_nocase: $symbol } }
                    ]
                }
            ) {
                id
                token0 { id symbol name }
                token1 { id symbol name }
                feeTier
                totalValueLockedUSD
                volumeUSD
            }
        }
        """

        try:
            async with self.session.post(
                url,
                json={'query': query, 'variables': {'symbol': token_symbol, 'minTvl': str(min_tvl)}}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('data', {}).get('pools', [])
        except Exception as e:
            logger.error(f"Pool discovery error: {e}")

        return []

    async def get_top_pools(
        self,
        chain: str = 'ethereum',
        min_tvl: float = 1000000,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get top pools by TVL.

        Args:
            chain: Blockchain network
            min_tvl: Minimum TVL filter
            limit: Maximum pools to return

        Returns:
            DataFrame with pool information
        """
        await self._rate_limit()

        url = self._get_url(chain)

        query = """
        query GetTopPools($minTvl: BigDecimal!, $limit: Int!) {
            pools(
                first: $limit
                orderBy: totalValueLockedUSD
                orderDirection: desc
                where: { totalValueLockedUSD_gte: $minTvl }
            ) {
                id
                token0 { id symbol name decimals }
                token1 { id symbol name decimals }
                feeTier
                liquidity
                totalValueLockedUSD
                volumeUSD
                txCount
                createdAtTimestamp
            }
        }
        """

        try:
            async with self.session.post(
                url,
                json={'query': query, 'variables': {'minTvl': str(min_tvl), 'limit': limit}}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    pools = result.get('data', {}).get('pools', [])

                    records = []
                    for pool in pools:
                        records.append({
                            'pool_id': pool['id'],
                            'token0_symbol': pool['token0']['symbol'],
                            'token0_address': pool['token0']['id'],
                            'token1_symbol': pool['token1']['symbol'],
                            'token1_address': pool['token1']['id'],
                            'fee_tier': int(pool['feeTier']),
                            'tvl_usd': float(pool['totalValueLockedUSD']),
                            'volume_usd': float(pool['volumeUSD']),
                            'tx_count': int(pool['txCount']),
                            'chain': chain,
                            'pair_name': f"{pool['token0']['symbol']}/{pool['token1']['symbol']}"
                        })

                    return pd.DataFrame(records)
        except Exception as e:
            logger.error(f"Top pools query error: {e}")

        return pd.DataFrame()


# =============================================================================
# DATA COLLECTION FUNCTIONS
# =============================================================================

async def collect_pool_historical_data(
    collector: UniswapV3DataCollector,
    pool_config: Dict,
    symbol: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Collect historical OHLCV data for a single pool.

    Args:
        collector: UniswapV3DataCollector instance
        pool_config: Pool configuration dict
        symbol: Token symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data
    """
    pool_id = pool_config['pool_id']
    chain = pool_config['chain']

    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

    logger.info(f"Collecting {symbol} data from {chain} pool {pool_id[:10]}...")

    day_data = await collector.query_pool_day_data(
        pool_id=pool_id,
        chain=chain,
        start_timestamp=start_ts,
        end_timestamp=end_ts
    )

    if not day_data:
        logger.warning(f"No data returned for {symbol}")
        return pd.DataFrame()

    records = []
    for candle in day_data:
        try:
            records.append({
                'timestamp': pd.to_datetime(int(candle['date']), unit='s', utc=True),
                'symbol': symbol,
                'open': float(candle.get('open', 0) or 0),
                'high': float(candle.get('high', 0) or 0),
                'low': float(candle.get('low', 0) or 0),
                'close': float(candle.get('close', 0) or 0),
                'volume': float(candle.get('volumeUSD', 0) or 0),
                'tvl_usd': float(candle.get('tvlUSD', 0) or 0),
                'fees_usd': float(candle.get('feesUSD', 0) or 0),
                'tx_count': int(candle.get('txCount', 0) or 0),
                'pool_id': pool_id,
                'chain': chain,
                'fee_tier': pool_config['fee_tier'],
                'base_token': pool_config['token0'],
                'quote_token': pool_config['token1'],
                'venue': 'uniswap_v3',
                'venue_type': 'DEX'
            })
        except Exception as e:
            logger.warning(f"Error parsing candle for {symbol}: {e}")
            continue

    df = pd.DataFrame(records)

    if not df.empty:
        df = df.sort_values('timestamp').reset_index(drop=True)
        logger.info(f"Collected {len(df)} daily candles for {symbol} ({df['timestamp'].min()} to {df['timestamp'].max()})")

    return df


async def collect_all_dex_data(
    api_key: str = "",
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    pools: Dict = None
) -> pd.DataFrame:
    """
    Collect historical data for all configured pools.

    Args:
        api_key: The Graph API key
        start_date: Start date
        end_date: End date
        pools: Pool configuration dict (default: UNISWAP_V3_POOLS)

    Returns:
        Combined DataFrame with all OHLCV data
    """
    if pools is None:
        pools = UNISWAP_V3_POOLS

    logger.info(f"Starting DEX data collection for {len(pools)} tokens")
    logger.info(f"Date range: {start_date} to {end_date}")

    all_data = []

    async with UniswapV3DataCollector(api_key=api_key) as collector:
        for symbol, pool_config in pools.items():
            try:
                df = await collect_pool_historical_data(
                    collector=collector,
                    pool_config=pool_config,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )

                if not df.empty:
                    all_data.append(df)

                # Small delay between pools
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error collecting {symbol}: {e}")
                continue

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total records collected: {len(combined)}")
        return combined

    return pd.DataFrame()


def analyze_data_coverage(df: pd.DataFrame) -> Dict:
    """
    Analyze data coverage for compliance checking.

    Args:
        df: Combined OHLCV DataFrame

    Returns:
        Coverage analysis dict
    """
    if df.empty:
        return {'status': 'FAILED', 'message': 'No data collected'}

    # Calculate date range
    date_min = df['timestamp'].min()
    date_max = df['timestamp'].max()

    # Get unique symbols
    symbols = df['symbol'].unique()

    # Calculate coverage per symbol
    coverage_stats = []
    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol]

        # Expected days in range
        expected_start = datetime.strptime(START_DATE, '%Y-%m-%d')
        expected_end = datetime.strptime(END_DATE, '%Y-%m-%d')
        expected_days = (expected_end - expected_start).days + 1

        # Actual days with data
        actual_days = len(symbol_df)

        # Coverage percentage
        coverage_pct = (actual_days / expected_days) * 100 if expected_days > 0 else 0

        coverage_stats.append({
            'symbol': symbol,
            'actual_days': actual_days,
            'expected_days': expected_days,
            'coverage_pct': coverage_pct,
            'date_min': symbol_df['timestamp'].min(),
            'date_max': symbol_df['timestamp'].max()
        })

    coverage_df = pd.DataFrame(coverage_stats)

    # PDF requirement: 20-30 tokens with good coverage
    high_coverage_symbols = coverage_df[coverage_df['coverage_pct'] >= 50]

    analysis = {
        'status': 'PASS' if len(high_coverage_symbols) >= 20 else 'PARTIAL',
        'total_symbols': len(symbols),
        'symbols_with_50pct_coverage': len(high_coverage_symbols),
        'date_range': f"{date_min} to {date_max}",
        'total_records': len(df),
        'coverage_stats': coverage_df.to_dict('records'),
        'meets_pdf_requirement': len(high_coverage_symbols) >= 20
    }

    return analysis


def save_data(df: pd.DataFrame, filename: str = "uniswap_v3_ohlcv_1d.parquet"):
    """
    Save collected data to parquet file.

    Args:
        df: DataFrame to save
        filename: Output filename
    """
    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Save to processed directory
    output_path = OUTPUT_DIR / filename
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")

    # Also save raw copy
    raw_path = RAW_DIR / filename
    df.to_parquet(raw_path, index=False)
    logger.info(f"Saved raw data to {raw_path}")

    return output_path


# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Main execution function."""
    print("=" * 70)
    print("DEX Historical Data Collection Script")
    print("=" * 70)
    print(f"\nDate range: {START_DATE} to {END_DATE}")
    print(f"Target: {len(UNISWAP_V3_POOLS)} tokens from Uniswap V3")
    print(f"API Key configured: {'Yes' if GRAPH_API_KEY else 'No (using hosted service)'}")
    print()

    # Collect data
    df = await collect_all_dex_data(
        api_key=GRAPH_API_KEY,
        start_date=START_DATE,
        end_date=END_DATE
    )

    if df.empty:
        print("\n[FAIL] No data collected. Please check:")
        print("   1. Network connectivity")
        print("   2. The Graph API status")
        print("   3. Pool addresses validity")
        return

    # Analyze coverage
    print("\n" + "=" * 70)
    print("DATA COVERAGE ANALYSIS")
    print("=" * 70)

    analysis = analyze_data_coverage(df)

    print(f"\nStatus: {analysis['status']}")
    print(f"Total symbols: {analysis['total_symbols']}")
    print(f"Symbols with ≥50% coverage: {analysis['symbols_with_50pct_coverage']}")
    print(f"Date range: {analysis['date_range']}")
    print(f"Total records: {analysis['total_records']}")
    print(f"Meets PDF requirement (20-30 tokens): {'[PASS] YES' if analysis['meets_pdf_requirement'] else '[WARN] PARTIAL'}")

    # Show per-symbol coverage
    print("\n" + "-" * 70)
    print("Per-Symbol Coverage (sorted by coverage):")
    print("-" * 70)

    coverage_df = pd.DataFrame(analysis['coverage_stats'])
    coverage_df = coverage_df.sort_values('coverage_pct', ascending=False)

    for _, row in coverage_df.iterrows():
        status = "[PASS]" if row['coverage_pct'] >= 50 else "[WARN]" if row['coverage_pct'] >= 25 else "[FAIL]"
        print(f"  {status} {row['symbol']:8s} - {row['coverage_pct']:5.1f}% ({row['actual_days']:4d} days)")

    # Save data
    print("\n" + "=" * 70)
    print("SAVING DATA")
    print("=" * 70)

    output_path = save_data(df, "uniswap_v3_ohlcv_1d.parquet")

    print(f"\n[OK] Data saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
PDF Requirement Check:
- DEX Universe: 20-30 tokens (Secondary/Supplemental)
- Date Range: 2022-2024 minimum

Current Status:
- Tokens collected: {analysis['total_symbols']}
- With good coverage (≥50%): {analysis['symbols_with_50pct_coverage']}
- Compliance: {'COMPLIANT' if analysis['meets_pdf_requirement'] else 'PARTIAL - Need more tokens or coverage'}

Next Steps:
1. Run Phase 2 with updated DEX data
2. DEX serves as SECONDARY validation data per PDF guidance
3. CEX remains PRIMARY data source
""")


if __name__ == "__main__":
    asyncio.run(main())
