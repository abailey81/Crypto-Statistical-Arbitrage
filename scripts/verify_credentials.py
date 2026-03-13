#!/usr/bin/env python3
"""
Credential Verification Script (Simplified)
============================================

Quick verification of API credentials for data collection system.

This is a lightweight version of verify_all_credentials.py for rapid checks.
Use verify_all_credentials.py for comprehensive verification.

Features
--------
- Load credentials from .env file or system environment
- Verify required credentials are present
- Test collector imports
- Color-coded console output

Usage
-----
    python scripts/verify_credentials.py
    python scripts/verify_credentials.py --verbose
    python scripts/verify_credentials.py --env-file /path/to/.env

Author: Crypto StatArb System
Version: 2.0.0
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import dotenv
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    print("Note: python-dotenv not installed. Using system environment.")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CredentialConfig:
    """Configuration for a single credential."""
    env_var: str
    description: str
    required: bool = True
    min_length: int = 10


# Credential categories with their environment variables
CREDENTIAL_GROUPS: Dict[str, List[CredentialConfig]] = {
    'CEX Exchanges': [
        CredentialConfig('BINANCE_API_KEY', 'Binance API Key'),
        CredentialConfig('BINANCE_SECRET_KEY', 'Binance Secret Key'),
        CredentialConfig('BYBIT_API_KEY', 'Bybit API Key'),
        CredentialConfig('BYBIT_SECRET_KEY', 'Bybit Secret Key'),
        CredentialConfig('OKX_API_KEY', 'OKX API Key'),
        CredentialConfig('OKX_SECRET_KEY', 'OKX Secret Key'),
        CredentialConfig('OKX_PASSPHRASE', 'OKX Passphrase', min_length=4),
        CredentialConfig('COINBASE_API_KEY', 'Coinbase API Key'),
        CredentialConfig('COINBASE_PRIVATE_KEY', 'Coinbase Private Key', min_length=50),
        CredentialConfig('KRAKEN_API_KEY', 'Kraken API Key'),
        CredentialConfig('KRAKEN_PRIVATE_KEY', 'Kraken Private Key', min_length=50),
    ],
    'Options Venues': [
        CredentialConfig('DERIBIT_CLIENT_ID', 'Deribit Client ID', min_length=8),
        CredentialConfig('DERIBIT_CLIENT_SECRET', 'Deribit Client Secret'),
        CredentialConfig('AEVO_API_KEY', 'AEVO API Key'),
        CredentialConfig('AEVO_API_SECRET', 'AEVO API Secret'),
    ],
    'DEX & Indexers': [
        CredentialConfig('THE_GRAPH_API_KEY', 'The Graph API Key'),
    ],
    'Market Data': [
        CredentialConfig('CRYPTOCOMPARE_API_KEY', 'CryptoCompare API Key'),
        CredentialConfig('COINGECKO_API_KEY', 'CoinGecko API Key', required=False),
        CredentialConfig('MESSARI_API_KEY', 'Messari API Key'),
        CredentialConfig('KAIKO_API_KEY', 'Kaiko API Key', required=False),
    ],
    'On-Chain Analytics': [
        CredentialConfig('GLASSNODE_API_KEY', 'Glassnode API Key'),
        CredentialConfig('NANSEN_API_KEY', 'Nansen API Key', required=False),
        CredentialConfig('ARKHAM_API_KEY', 'Arkham API Key', required=False),
        CredentialConfig('CRYPTOQUANT_API_KEY', 'CryptoQuant API Key'),
        CredentialConfig('SANTIMENT_API_KEY', 'Santiment API Key'),
        CredentialConfig('COINMETRICS_API_KEY', 'Coin Metrics API Key', required=False),
        CredentialConfig('COVALENT_API_KEY', 'Covalent API Key'),
        CredentialConfig('BITQUERY_ACCESS_TOKEN', 'Bitquery Access Token'),
        CredentialConfig('WHALE_ALERT_API_KEY', 'Whale Alert API Key', required=False),
        CredentialConfig('FLIPSIDE_API_KEY', 'Flipside API Key', required=False),
    ],
    'Alternative Data': [
        CredentialConfig('COINALYZE_API_KEY', 'Coinalyze API Key'),
        CredentialConfig('DUNE_API_KEY', 'Dune Analytics API Key'),
        CredentialConfig('LUNARCRUSH_API_KEY', 'LunarCrush API Key'),
    ],
    'Social': [
        CredentialConfig('TWITTER_BEARER_TOKEN', 'Twitter/X Bearer Token', min_length=50),
    ],
}

# Free data sources (no credentials required)
FREE_SOURCES = [
    ('Hyperliquid', 'On-chain perps (1h funding)'),
    ('dYdX V4', 'Cosmos perpetuals (1h funding)'),
    ('GeckoTerminal', 'DEX aggregator (100+ chains)'),
    ('DEXScreener', 'DEX data aggregator'),
    ('1inch', 'DEX aggregator (free tier)'),
    ('0x Protocol', 'DEX aggregator (free tier)'),
    ('GMX', 'Perp DEX on Arbitrum'),
    ('Vertex', 'Orderbook DEX on Arbitrum'),
    ('Jupiter', 'Solana DEX aggregator'),
    ('CowSwap', 'MEV-protected trading'),
    ('Curve Finance', 'Stablecoin DEX'),
    ('SushiSwap', 'Multi-chain DEX'),
    ('DefiLlama', 'TVL data (CC0 license)'),
    ('Lyra Finance', 'Options AMM'),
    ('Dopex', 'Options vaults'),
    ('CCXT', 'Exchange wrapper library'),
]

# Collector modules to test imports
COLLECTOR_MODULES = [
    ('data_collection.cex.binance_collector', 'BinanceCollector'),
    ('data_collection.cex.bybit_collector', 'BybitCollector'),
    ('data_collection.cex.okx_collector', 'OKXCollector'),
    ('data_collection.cex.coinbase_collector', 'CoinbaseCollector'),
    ('data_collection.cex.kraken_collector', 'KrakenCollector'),
    ('data_collection.hybrid.hyperliquid_collector', 'HyperliquidCollector'),
    ('data_collection.hybrid.dydx_collector', 'DYDXCollector'),
    ('data_collection.dex.uniswap_collector', 'UniswapCollector'),
    ('data_collection.dex.geckoterminal_collector', 'GeckoTerminalCollector'),
    ('data_collection.dex.dexscreener_collector', 'DEXScreenerCollector'),
    ('data_collection.dex.gmx_collector', 'GMXCollector'),
    ('data_collection.options.deribit_collector', 'DeribitCollector'),
    ('data_collection.options.aevo_collector', 'AevoCollector'),
    ('data_collection.market_data.cryptocompare_collector', 'CryptoCompareCollector'),
    ('data_collection.market_data.coingecko_collector', 'CoinGeckoCollector'),
    ('data_collection.onchain.glassnode_collector', 'GlassnodeCollector'),
    ('data_collection.onchain.covalent_collector', 'CovalentCollector'),
    ('data_collection.alternative.coinalyze_collector', 'CoinalyzeCollector'),
    ('data_collection.alternative.defillama_collector', 'DefiLlamaCollector'),
    ('data_collection.alternative.dune_analytics_collector', 'DuneAnalyticsCollector'),
]


# =============================================================================
# Helper Functions
# =============================================================================

def mask_credential(value: Optional[str]) -> str:
    """Mask credential for safe display."""
    if not value:
        return 'NOT SET'
    if len(value) <= 8:
        return '*' * len(value)
    return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"


def check_credential(config: CredentialConfig) -> Tuple[bool, str, str]:
    """
    Check if a credential is set and valid.
    
    Returns
    -------
    Tuple[bool, str, str]
        (is_valid, masked_value, message)
    """
    value = os.getenv(config.env_var)
    masked = mask_credential(value)
    
    if not value:
        if config.required:
            return False, masked, 'MISSING (required)'
        else:
            return True, masked, 'NOT SET (optional)'
    
    if len(value) < config.min_length:
        return False, masked, f'TOO SHORT (min {config.min_length})'
    
    return True, masked, 'OK'


def test_import(module_path: str, class_name: str) -> Tuple[bool, Optional[str]]:
    """
    Test that a collector module can be imported.
    
    Returns
    -------
    Tuple[bool, Optional[str]]
        (success, error_message)
    """
    try:
        module = __import__(module_path, fromlist=[class_name])
        _ = getattr(module, class_name)
        return True, None
    except ImportError as e:
        return False, f"ImportError: {e}"
    except AttributeError as e:
        return False, f"AttributeError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# =============================================================================
# Main Verification Functions
# =============================================================================

def verify_credentials(verbose: bool = False) -> Tuple[bool, int, int]:
    """
    Verify all credentials are loaded.
    
    Returns
    -------
    Tuple[bool, int, int]
        (all_required_present, ok_count, missing_count)
    """
    print("=" * 70)
    print("CRYPTO STATISTICAL ARBITRAGE - CREDENTIAL VERIFICATION")
    print("=" * 70)
    print()
    
    all_valid = True
    total_ok = 0
    total_missing = 0
    
    # Check each credential group
    for group_name, credentials in CREDENTIAL_GROUPS.items():
        print(f"{group_name}:")
        print("-" * 70)
        
        for config in credentials:
            is_valid, masked, message = check_credential(config)
            
            status = "+" if is_valid else "x"
            
            if not is_valid and config.required:
                all_valid = False
                total_missing += 1
            else:
                total_ok += 1
            
            print(f"  {status} {config.description:<35} [{config.env_var}]")
            
            if verbose:
                print(f"      Value: {masked}")
                if message != 'OK':
                    print(f"      Status: {message}")
        
        print()
    
    # Show free sources
    print("FREE DATA SOURCES (No API Key Required):")
    print("-" * 70)
    for name, description in FREE_SOURCES:
        print(f"  * {name:<20} - {description}")
    print()
    
    return all_valid, total_ok, total_missing


def test_collector_imports(verbose: bool = False) -> Tuple[bool, int, int]:
    """
    Test that all collectors can be imported.
    
    Returns
    -------
    Tuple[bool, int, int]
        (all_success, success_count, failed_count)
    """
    print("COLLECTOR IMPORT TEST:")
    print("-" * 70)
    
    success_count = 0
    failed_count = 0
    all_success = True
    
    for module_path, class_name in COLLECTOR_MODULES:
        success, error = test_import(module_path, class_name)
        
        if success:
            print(f"  + {class_name}")
            success_count += 1
        else:
            print(f"  x {class_name}")
            if verbose:
                print(f"      Error: {error}")
            failed_count += 1
            all_success = False
    
    print()
    print(f"Import Results: {success_count} success, {failed_count} failed")
    
    return all_success, success_count, failed_count


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Verify API credentials for data collection'
    )
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed output')
    parser.add_argument('--env-file', '-e', type=str,
                       help='Path to .env file')
    parser.add_argument('--skip-imports', action='store_true',
                       help='Skip import testing')
    args = parser.parse_args()
    
    # Load environment
    env_loaded = False
    
    if HAS_DOTENV:
        env_paths = [
            Path(args.env_file) if args.env_file else None,
            PROJECT_ROOT / 'config' / '.env',
            PROJECT_ROOT / '.env',
        ]
        
        for env_path in env_paths:
            if env_path and env_path.exists():
                load_dotenv(env_path)
                print(f"Loaded environment from: {env_path}")
                env_loaded = True
                break
    
    if not env_loaded:
        print("Using system environment variables")
    
    print()
    
    # Verify credentials
    creds_ok, total_ok, total_missing = verify_credentials(args.verbose)
    
    # Test imports
    imports_ok = True
    if not args.skip_imports:
        print()
        imports_ok, _, _ = test_collector_imports(args.verbose)
    
    # Final status
    print()
    print("=" * 70)
    
    if creds_ok and imports_ok:
        print("STATUS: VERIFICATION COMPLETE - SYSTEM READY")
        print("=" * 70)
        return 0
    else:
        print("STATUS: VERIFICATION FAILED - CHECK ISSUES ABOVE")
        if not creds_ok:
            print(f"  - {total_missing} required credentials missing")
        if not imports_ok:
            print(f"  - Some collectors failed to import")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())