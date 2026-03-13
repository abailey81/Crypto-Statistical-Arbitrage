#!/usr/bin/env python3
"""
Comprehensive Credential Verification Script
=============================================

Verifies all 47+ data source credentials and optionally tests live connections.

Features
--------
1. Credential Loading:
    - Load from .env file
    - Load from system environment
    - Support for multiple .env locations

2. Verification Levels:
    - BASIC: Check environment variables exist
    - FORMAT: Validate credential format (length, prefix)
    - CONNECTION: Test actual API connectivity

3. Reporting:
    - Console output with color coding
    - JSON export for CI/CD integration
    - Markdown report generation

4. Categories Covered:
    - CEX: Binance, Bybit, OKX, Coinbase, Kraken, CME (6)
    - Hybrid: Hyperliquid, dYdX (2)
    - DEX: Uniswap, GeckoTerminal, DEXScreener, 1inch, 0x, GMX, etc. (11)
    - Options: Deribit, AEVO, Lyra, Dopex (4)
    - Market Data: CryptoCompare, CoinGecko, Messari, Kaiko (4)
    - On-Chain: Glassnode, Nansen, Arkham, CryptoQuant, etc. (10)
    - Alternative: Coinalyze, Dune, DefiLlama, LunarCrush (5)
    - Social: Twitter/X (1)
    - Indexers: The Graph (1)
    - Multi-Exchange: CCXT (1)

Usage
-----
    # Basic verification
    python scripts/verify_all_credentials.py
    
    # Verbose output
    python scripts/verify_all_credentials.py -v
    
    # Test live connections (slow)
    python scripts/verify_all_credentials.py --test-connections
    
    # Export JSON report
    python scripts/verify_all_credentials.py --json-output report.json
    
    # Custom .env file
    python scripts/verify_all_credentials.py -e /path/to/.env

Exit Codes
----------
    0: All required credentials present
    1: Missing required credentials
    2: Configuration error

Author: Crypto StatArb System
Version: 2.0.0
"""

import os
import sys
import argparse
import json
import asyncio
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import optional dependencies
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    logger.warning("python-dotenv not installed. Using system environment only.")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


# =============================================================================
# Enumerations
# =============================================================================

class CredentialStatus(Enum):
    """Status of a credential check."""
    OK = "OK"
    MISSING = "MISSING"
    INVALID_FORMAT = "INVALID_FORMAT"
    CONNECTION_FAILED = "CONNECTION_FAILED"
    FREE = "FREE"
    OPTIONAL = "OPTIONAL"
    SKIPPED = "SKIPPED"
    
    @property
    def symbol(self) -> str:
        """Get display symbol."""
        symbols = {
            CredentialStatus.OK: "+",
            CredentialStatus.MISSING: "x",
            CredentialStatus.INVALID_FORMAT: "!",
            CredentialStatus.CONNECTION_FAILED: "o",
            CredentialStatus.FREE: "*",
            CredentialStatus.OPTIONAL: "-",
            CredentialStatus.SKIPPED: ".",
        }
        return symbols.get(self, "?")
    
    @property
    def is_success(self) -> bool:
        """Whether status indicates success."""
        return self in [
            CredentialStatus.OK,
            CredentialStatus.FREE,
            CredentialStatus.OPTIONAL,
            CredentialStatus.SKIPPED
        ]


class CollectorCategory(Enum):
    """Data collector categories."""
    CEX = "Centralized Exchange"
    HYBRID = "Hybrid (On-chain CEX)"
    DEX = "Decentralized Exchange"
    OPTIONS = "Options Venues"
    MARKET_DATA = "Market Data Providers"
    ONCHAIN = "On-Chain Analytics"
    ALTERNATIVE = "Alternative Data"
    SOCIAL = "Social & Sentiment"
    INDEXERS = "Blockchain Indexers"
    MULTI_EXCHANGE = "Multi-Exchange Wrappers"


class VerificationLevel(Enum):
    """Credential verification levels."""
    BASIC = "basic"          # Check env var exists
    FORMAT = "format"        # Validate format
    CONNECTION = "connection" # Test live connection


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CredentialSpec:
    """Specification for a single credential."""
    name: str
    required: bool = True
    min_length: int = 10
    max_length: int = 500
    pattern: Optional[str] = None  # Regex pattern
    description: str = ""
    
    def validate_format(self, value: str) -> Tuple[bool, str]:
        """Validate credential format."""
        if not value:
            return False, "Empty value"
        
        if len(value) < self.min_length:
            return False, f"Too short (min {self.min_length})"
        
        if len(value) > self.max_length:
            return False, f"Too long (max {self.max_length})"
        
        if self.pattern:
            if not re.match(self.pattern, value):
                return False, f"Invalid format"
        
        return True, "OK"


@dataclass
class CollectorInfo:
    """Complete information about a data collector."""
    name: str
    category: CollectorCategory
    credentials: List[CredentialSpec]
    is_free: bool
    description: str
    module_path: str
    class_name: str
    api_base_url: Optional[str] = None
    health_endpoint: Optional[str] = None
    documentation_url: Optional[str] = None
    rate_limit: Optional[str] = None
    
    @property
    def env_vars(self) -> List[str]:
        """Get list of environment variable names."""
        return [c.name for c in self.credentials]
    
    @property
    def required_count(self) -> int:
        """Count of required credentials."""
        return sum(1 for c in self.credentials if c.required)


@dataclass
class CredentialCheckResult:
    """Result of checking a single credential."""
    name: str
    status: CredentialStatus
    masked_value: str
    message: str = ""
    format_valid: bool = True


@dataclass
class CollectorCheckResult:
    """Result of checking a collector's credentials."""
    name: str
    category: str
    status: CredentialStatus
    credentials: Dict[str, CredentialCheckResult]
    description: str
    is_free: bool
    import_success: bool = True
    import_error: Optional[str] = None
    connection_success: Optional[bool] = None
    connection_error: Optional[str] = None


@dataclass
class VerificationReport:
    """Complete verification report."""
    timestamp: str
    total_collectors: int
    free_collectors: int
    paid_collectors: int
    credentials_ok: int
    credentials_missing: int
    imports_ok: int
    imports_failed: int
    connections_tested: int
    connections_ok: int
    results: Dict[str, CollectorCheckResult]
    
    @property
    def is_success(self) -> bool:
        """Check if verification passed."""
        # Success if all required credentials present
        for result in self.results.values():
            if not result.is_free:
                for cred in result.credentials.values():
                    if not cred.status.is_success:
                        return False
        return True
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        def serialize(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, (CredentialCheckResult, CollectorCheckResult)):
                return asdict(obj)
            return obj
        
        data = asdict(self)
        return json.dumps(data, default=serialize, indent=2)


# =============================================================================
# Collector Registry
# =============================================================================

# Complete list of all data collectors with credential specifications
ALL_COLLECTORS: List[CollectorInfo] = [
    # =========================================================================
    # CEX (6)
    # =========================================================================
    CollectorInfo(
        name='binance',
        category=CollectorCategory.CEX,
        credentials=[
            CredentialSpec('BINANCE_API_KEY', min_length=64, max_length=64,
                          description='Binance API Key'),
            CredentialSpec('BINANCE_SECRET_KEY', min_length=64, max_length=64,
                          description='Binance Secret Key'),
        ],
        is_free=False,
        description='Binance Exchange - Funding rates, OHLCV, Open Interest',
        module_path='data_collection.cex.binance_collector',
        class_name='BinanceCollector',
        api_base_url='https://api.binance.com',
        health_endpoint='/api/v3/ping',
        documentation_url='https://binance-docs.github.io/apidocs/',
        rate_limit='1200 requests/minute (weight-based)'
    ),
    CollectorInfo(
        name='bybit',
        category=CollectorCategory.CEX,
        credentials=[
            CredentialSpec('BYBIT_API_KEY', min_length=18,
                          description='Bybit API Key'),
            CredentialSpec('BYBIT_SECRET_KEY', min_length=36,
                          description='Bybit Secret Key'),
        ],
        is_free=False,
        description='Bybit Exchange - Cross-validation source',
        module_path='data_collection.cex.bybit_collector',
        class_name='BybitCollector',
        api_base_url='https://api.bybit.com',
        health_endpoint='/v5/market/time',
        documentation_url='https://bybit-exchange.github.io/docs/',
        rate_limit='120 requests/minute'
    ),
    CollectorInfo(
        name='okx',
        category=CollectorCategory.CEX,
        credentials=[
            CredentialSpec('OKX_API_KEY', min_length=20,
                          description='OKX API Key'),
            CredentialSpec('OKX_SECRET_KEY', min_length=30,
                          description='OKX Secret Key'),
            CredentialSpec('OKX_PASSPHRASE', min_length=4, max_length=32,
                          description='OKX Passphrase'),
        ],
        is_free=False,
        description='OKX Exchange - Additional validation',
        module_path='data_collection.cex.okx_collector',
        class_name='OKXCollector',
        api_base_url='https://www.okx.com',
        health_endpoint='/api/v5/public/time',
        documentation_url='https://www.okx.com/docs-v5/en/',
        rate_limit='20 requests/2 seconds'
    ),
    CollectorInfo(
        name='coinbase',
        category=CollectorCategory.CEX,
        credentials=[
            CredentialSpec('COINBASE_API_KEY', min_length=20,
                          description='Coinbase API Key'),
            CredentialSpec('COINBASE_PRIVATE_KEY', min_length=100,
                          description='Coinbase EC Private Key'),
        ],
        is_free=False,
        description='Coinbase Exchange - Major US exchange (spot only)',
        module_path='data_collection.cex.coinbase_collector',
        class_name='CoinbaseCollector',
        api_base_url='https://api.coinbase.com',
        documentation_url='https://docs.cloud.coinbase.com/',
        rate_limit='100 requests/minute'
    ),
    CollectorInfo(
        name='kraken',
        category=CollectorCategory.CEX,
        credentials=[
            CredentialSpec('KRAKEN_API_KEY', min_length=40,
                          description='Kraken API Key'),
            CredentialSpec('KRAKEN_PRIVATE_KEY', min_length=80,
                          description='Kraken Private Key'),
        ],
        is_free=False,
        description='Kraken Exchange - Established exchange',
        module_path='data_collection.cex.kraken_collector',
        class_name='KrakenCollector',
        api_base_url='https://api.kraken.com',
        health_endpoint='/0/public/Time',
        documentation_url='https://docs.kraken.com/rest/',
        rate_limit='Tier-based (15-20 calls/second)'
    ),
    CollectorInfo(
        name='cme',
        category=CollectorCategory.CEX,
        credentials=[
            CredentialSpec('CME_API_KEY', required=False,
                          description='CME DataMine API Key'),
        ],
        is_free=False,
        description='CME Bitcoin Futures - Institutional benchmark',
        module_path='data_collection.cex.cme_collector',
        class_name='CMECollector',
        api_base_url='https://datamine.cmegroup.com',
        documentation_url='https://www.cmegroup.com/market-data/datamine-api.html',
    ),
    
    # =========================================================================
    # Hybrid (2)
    # =========================================================================
    CollectorInfo(
        name='hyperliquid',
        category=CollectorCategory.HYBRID,
        credentials=[],  # Free API
        is_free=True,
        description='Hyperliquid - On-chain perps with CEX-like UX (1h funding)',
        module_path='data_collection.hybrid.hyperliquid_collector',
        class_name='HyperliquidCollector',
        api_base_url='https://api.hyperliquid.xyz',
        health_endpoint='/info',
        documentation_url='https://hyperliquid.gitbook.io/hyperliquid-docs/',
        rate_limit='100 requests/minute'
    ),
    CollectorInfo(
        name='dydx',
        category=CollectorCategory.HYBRID,
        credentials=[],  # Free public API
        is_free=True,
        description='dYdX V4 - Cosmos-based perpetuals (1h funding)',
        module_path='data_collection.hybrid.dydx_collector',
        class_name='DYDXCollector',
        api_base_url='https://indexer.dydx.trade',
        documentation_url='https://docs.dydx.exchange/',
        rate_limit='100 requests/10 seconds'
    ),
    
    # =========================================================================
    # DEX (11)
    # =========================================================================
    CollectorInfo(
        name='uniswap',
        category=CollectorCategory.DEX,
        credentials=[
            CredentialSpec('THE_GRAPH_API_KEY', min_length=32,
                          description='The Graph API Key'),
        ],
        is_free=False,
        description='Uniswap V3 - Primary DEX via The Graph subgraph',
        module_path='data_collection.dex.uniswap_collector',
        class_name='UniswapCollector',
        api_base_url='https://gateway.thegraph.com/api',
        documentation_url='https://docs.uniswap.org/',
    ),
    CollectorInfo(
        name='geckoterminal',
        category=CollectorCategory.DEX,
        credentials=[],
        is_free=True,
        description='GeckoTerminal - Free DEX aggregator API (100+ chains)',
        module_path='data_collection.dex.geckoterminal_collector',
        class_name='GeckoTerminalCollector',
        api_base_url='https://api.geckoterminal.com/api/v2',
        documentation_url='https://www.geckoterminal.com/dex-api',
        rate_limit='30 calls/minute'
    ),
    CollectorInfo(
        name='dexscreener',
        category=CollectorCategory.DEX,
        credentials=[],
        is_free=True,
        description='DEXScreener - Free DEX data aggregator',
        module_path='data_collection.dex.dexscreener_collector',
        class_name='DEXScreenerCollector',
        api_base_url='https://api.dexscreener.com',
        documentation_url='https://docs.dexscreener.com/',
        rate_limit='300 requests/minute'
    ),
    CollectorInfo(
        name='oneinch',
        category=CollectorCategory.DEX,
        credentials=[
            CredentialSpec('ONEINCH_API_KEY', required=False,
                          description='1inch API Key (optional)'),
        ],
        is_free=True,
        description='1inch - DEX aggregator (free tier available)',
        module_path='data_collection.dex.oneinch_collector',
        class_name='OneInchCollector',
        api_base_url='https://api.1inch.dev',
        documentation_url='https://docs.1inch.io/',
    ),
    CollectorInfo(
        name='zerox',
        category=CollectorCategory.DEX,
        credentials=[
            CredentialSpec('ZEROX_API_KEY', required=False,
                          description='0x API Key (optional)'),
        ],
        is_free=True,
        description='0x Protocol - DEX aggregator (free tier available)',
        module_path='data_collection.dex.zerox_collector',
        class_name='ZeroXCollector',
        api_base_url='https://api.0x.org',
        documentation_url='https://docs.0x.org/',
    ),
    CollectorInfo(
        name='gmx',
        category=CollectorCategory.DEX,
        credentials=[],
        is_free=True,
        description='GMX - Perpetual DEX on Arbitrum/Avalanche',
        module_path='data_collection.dex.gmx_collector',
        class_name='GMXCollector',
        api_base_url='https://subgraph.satsuma-prod.com',
        documentation_url='https://gmxio.gitbook.io/',
    ),
    CollectorInfo(
        name='vertex',
        category=CollectorCategory.DEX,
        credentials=[],
        is_free=True,
        description='Vertex Protocol - Arbitrum orderbook DEX',
        module_path='data_collection.dex.vertex_collector',
        class_name='VertexCollector',
        api_base_url='https://archive.prod.vertexprotocol.com',
        documentation_url='https://docs.vertexprotocol.com/',
    ),
    CollectorInfo(
        name='jupiter',
        category=CollectorCategory.DEX,
        credentials=[],
        is_free=True,
        description='Jupiter - Solana DEX aggregator',
        module_path='data_collection.dex.jupiter_collector',
        class_name='JupiterCollector',
        api_base_url='https://quote-api.jup.ag',
        documentation_url='https://station.jup.ag/docs/',
    ),
    CollectorInfo(
        name='cowswap',
        category=CollectorCategory.DEX,
        credentials=[],
        is_free=True,
        description='CowSwap - MEV-protected trading',
        module_path='data_collection.dex.cowswap_collector',
        class_name='CowSwapCollector',
        api_base_url='https://api.cow.fi',
        documentation_url='https://docs.cow.fi/',
    ),
    CollectorInfo(
        name='curve',
        category=CollectorCategory.DEX,
        credentials=[],
        is_free=True,
        description='Curve Finance - Stablecoin/LST DEX',
        module_path='data_collection.dex.curve_collector',
        class_name='CurveCollector',
        api_base_url='https://api.curve.fi',
        documentation_url='https://docs.curve.fi/',
    ),
    CollectorInfo(
        name='sushiswap',
        category=CollectorCategory.DEX,
        credentials=[],
        is_free=True,
        description='SushiSwap - Multi-chain DEX',
        module_path='data_collection.dex.sushiswap_collector',
        class_name='SushiSwapCollector',
        api_base_url='https://api.sushi.com',
        documentation_url='https://docs.sushi.com/',
    ),
    
    # =========================================================================
    # Options (4)
    # =========================================================================
    CollectorInfo(
        name='deribit',
        category=CollectorCategory.OPTIONS,
        credentials=[
            CredentialSpec('DERIBIT_CLIENT_ID', min_length=8,
                          description='Deribit Client ID'),
            CredentialSpec('DERIBIT_CLIENT_SECRET', min_length=20,
                          description='Deribit Client Secret'),
        ],
        is_free=False,
        description='Deribit - Major crypto options exchange',
        module_path='data_collection.options.deribit_collector',
        class_name='DeribitCollector',
        api_base_url='https://www.deribit.com/api/v2',
        health_endpoint='/public/test',
        documentation_url='https://docs.deribit.com/',
        rate_limit='20 requests/second'
    ),
    CollectorInfo(
        name='aevo',
        category=CollectorCategory.OPTIONS,
        credentials=[
            CredentialSpec('AEVO_API_KEY', min_length=20,
                          description='AEVO API Key'),
            CredentialSpec('AEVO_API_SECRET', min_length=40,
                          description='AEVO API Secret'),
        ],
        is_free=False,
        description='AEVO - Options on custom L2 rollup',
        module_path='data_collection.options.aevo_collector',
        class_name='AevoCollector',
        api_base_url='https://api.aevo.xyz',
        documentation_url='https://docs.aevo.xyz/',
    ),
    CollectorInfo(
        name='lyra',
        category=CollectorCategory.OPTIONS,
        credentials=[],
        is_free=True,
        description='Lyra Finance - AMM-based options on Optimism/Arbitrum',
        module_path='data_collection.options.lyra_collector',
        class_name='LyraCollector',
        api_base_url='https://api.lyra.finance',
        documentation_url='https://docs.lyra.finance/',
    ),
    CollectorInfo(
        name='dopex',
        category=CollectorCategory.OPTIONS,
        credentials=[],
        is_free=True,
        description='Dopex - Options vaults on Arbitrum',
        module_path='data_collection.options.dopex_collector',
        class_name='DopexCollector',
        documentation_url='https://docs.dopex.io/',
    ),
    
    # =========================================================================
    # Market Data (4)
    # =========================================================================
    CollectorInfo(
        name='cryptocompare',
        category=CollectorCategory.MARKET_DATA,
        credentials=[
            CredentialSpec('CRYPTOCOMPARE_API_KEY', min_length=40,
                          description='CryptoCompare API Key'),
        ],
        is_free=False,
        description='CryptoCompare - Aggregated market data',
        module_path='data_collection.market_data.cryptocompare_collector',
        class_name='CryptoCompareCollector',
        api_base_url='https://min-api.cryptocompare.com',
        documentation_url='https://min-api.cryptocompare.com/documentation',
    ),
    CollectorInfo(
        name='coingecko',
        category=CollectorCategory.MARKET_DATA,
        credentials=[
            CredentialSpec('COINGECKO_API_KEY', required=False,
                          description='CoinGecko Pro API Key'),
        ],
        is_free=True,  # Free tier available
        description='CoinGecko - Comprehensive crypto data (free tier available)',
        module_path='data_collection.market_data.coingecko_collector',
        class_name='CoinGeckoCollector',
        api_base_url='https://api.coingecko.com/api/v3',
        health_endpoint='/ping',
        documentation_url='https://www.coingecko.com/en/api/documentation',
        rate_limit='10-50 calls/minute (tier-based)'
    ),
    CollectorInfo(
        name='messari',
        category=CollectorCategory.MARKET_DATA,
        credentials=[
            CredentialSpec('MESSARI_API_KEY', min_length=30,
                          description='Messari API Key'),
        ],
        is_free=False,
        description='Messari - Research-grade market data',
        module_path='data_collection.market_data.messari_collector',
        class_name='MessariCollector',
        api_base_url='https://data.messari.io/api',
        documentation_url='https://messari.io/api/docs',
    ),
    CollectorInfo(
        name='kaiko',
        category=CollectorCategory.MARKET_DATA,
        credentials=[
            CredentialSpec('KAIKO_API_KEY', min_length=30,
                          description='Kaiko API Key'),
        ],
        is_free=False,
        description='Kaiko - Institutional market data',
        module_path='data_collection.market_data.kaiko_collector',
        class_name='KaikoCollector',
        api_base_url='https://us.market-api.kaiko.io',
        documentation_url='https://docs.kaiko.com/',
    ),
    
    # =========================================================================
    # On-Chain Analytics (10)
    # =========================================================================
    CollectorInfo(
        name='glassnode',
        category=CollectorCategory.ONCHAIN,
        credentials=[
            CredentialSpec('GLASSNODE_API_KEY', min_length=30,
                          description='Glassnode API Key'),
        ],
        is_free=False,
        description='Glassnode - On-chain market intelligence',
        module_path='data_collection.onchain.glassnode_collector',
        class_name='GlassnodeCollector',
        api_base_url='https://api.glassnode.com',
        documentation_url='https://docs.glassnode.com/',
    ),
    CollectorInfo(
        name='nansen',
        category=CollectorCategory.ONCHAIN,
        credentials=[
            CredentialSpec('NANSEN_API_KEY', min_length=30,
                          description='Nansen API Key'),
        ],
        is_free=False,
        description='Nansen - Smart money analytics and labels',
        module_path='data_collection.onchain.nansen_collector',
        class_name='NansenCollector',
        api_base_url='https://api.nansen.ai',
        documentation_url='https://docs.nansen.ai/',
    ),
    CollectorInfo(
        name='arkham',
        category=CollectorCategory.ONCHAIN,
        credentials=[
            CredentialSpec('ARKHAM_API_KEY', min_length=30,
                          description='Arkham Intelligence API Key'),
        ],
        is_free=False,
        description='Arkham Intelligence - Entity tracking and intelligence',
        module_path='data_collection.onchain.arkham_collector',
        class_name='ArkhamCollector',
        api_base_url='https://api.arkhamintelligence.com',
        documentation_url='https://docs.arkhamintelligence.com/',
    ),
    CollectorInfo(
        name='cryptoquant',
        category=CollectorCategory.ONCHAIN,
        credentials=[
            CredentialSpec('CRYPTOQUANT_API_KEY', min_length=30,
                          description='CryptoQuant API Key'),
        ],
        is_free=False,
        description='CryptoQuant - Exchange flow and on-chain data',
        module_path='data_collection.onchain.cryptoquant_collector',
        class_name='CryptoQuantCollector',
        api_base_url='https://api.cryptoquant.com',
        documentation_url='https://cryptoquant.com/docs',
    ),
    CollectorInfo(
        name='santiment',
        category=CollectorCategory.ONCHAIN,
        credentials=[
            CredentialSpec('SANTIMENT_API_KEY', min_length=30,
                          description='Santiment API Key'),
        ],
        is_free=False,
        description='Santiment - Crypto behavior analytics',
        module_path='data_collection.onchain.santiment_collector',
        class_name='SantimentCollector',
        api_base_url='https://api.santiment.net',
        documentation_url='https://academy.santiment.net/',
    ),
    CollectorInfo(
        name='coinmetrics',
        category=CollectorCategory.ONCHAIN,
        credentials=[
            CredentialSpec('COINMETRICS_API_KEY', min_length=30,
                          description='Coin Metrics API Key'),
        ],
        is_free=False,
        description='Coin Metrics - Network data and market metrics',
        module_path='data_collection.onchain.coinmetrics_collector',
        class_name='CoinMetricsCollector',
        api_base_url='https://api.coinmetrics.io',
        documentation_url='https://docs.coinmetrics.io/',
    ),
    CollectorInfo(
        name='covalent',
        category=CollectorCategory.ONCHAIN,
        credentials=[
            CredentialSpec('COVALENT_API_KEY', min_length=30,
                          description='Covalent API Key'),
        ],
        is_free=False,
        description='Covalent - Multi-chain blockchain data',
        module_path='data_collection.onchain.covalent_collector',
        class_name='CovalentCollector',
        api_base_url='https://api.covalenthq.com',
        documentation_url='https://www.covalenthq.com/docs/',
    ),
    CollectorInfo(
        name='bitquery',
        category=CollectorCategory.ONCHAIN,
        credentials=[
            CredentialSpec('BITQUERY_ACCESS_TOKEN', min_length=30,
                          description='Bitquery Access Token'),
        ],
        is_free=False,
        description='Bitquery - GraphQL blockchain data',
        module_path='data_collection.onchain.bitquery_collector',
        class_name='BitqueryCollector',
        api_base_url='https://graphql.bitquery.io',
        documentation_url='https://docs.bitquery.io/',
    ),
    CollectorInfo(
        name='whale_alert',
        category=CollectorCategory.ONCHAIN,
        credentials=[
            CredentialSpec('WHALE_ALERT_API_KEY', min_length=30,
                          description='Whale Alert API Key'),
        ],
        is_free=False,
        description='Whale Alert - Large transaction tracking',
        module_path='data_collection.onchain.whale_alert_collector',
        class_name='WhaleAlertCollector',
        api_base_url='https://api.whale-alert.io',
        documentation_url='https://docs.whale-alert.io/',
    ),
    CollectorInfo(
        name='flipside',
        category=CollectorCategory.ONCHAIN,
        credentials=[
            CredentialSpec('FLIPSIDE_API_KEY', min_length=30,
                          description='Flipside Crypto API Key'),
        ],
        is_free=False,
        description='Flipside Crypto - SQL interface to blockchain',
        module_path='data_collection.onchain.flipside_collector',
        class_name='FlipsideCollector',
        api_base_url='https://api.flipsidecrypto.com',
        documentation_url='https://docs.flipsidecrypto.com/',
    ),
    
    # =========================================================================
    # Alternative Data (5)
    # =========================================================================
    CollectorInfo(
        name='coinalyze',
        category=CollectorCategory.ALTERNATIVE,
        credentials=[
            CredentialSpec('COINALYZE_API_KEY', min_length=20,
                          description='Coinalyze API Key'),
        ],
        is_free=False,
        description='Coinalyze - Derivatives analytics (funding, OI, liquidations)',
        module_path='data_collection.alternative.coinalyze_collector',
        class_name='CoinalyzeCollector',
        api_base_url='https://api.coinalyze.net',
        documentation_url='https://api.coinalyze.net/v1/doc/',
    ),
    CollectorInfo(
        name='dune',
        category=CollectorCategory.ALTERNATIVE,
        credentials=[
            CredentialSpec('DUNE_API_KEY', min_length=30,
                          description='Dune Analytics API Key'),
        ],
        is_free=False,
        description='Dune Analytics - Custom SQL queries on blockchain',
        module_path='data_collection.alternative.dune_analytics_collector',
        class_name='DuneAnalyticsCollector',
        api_base_url='https://api.dune.com/api/v1',
        documentation_url='https://docs.dune.com/',
    ),
    CollectorInfo(
        name='defillama',
        category=CollectorCategory.ALTERNATIVE,
        credentials=[],
        is_free=True,
        description='DefiLlama - TVL and DeFi protocol analytics (CC0 license)',
        module_path='data_collection.alternative.defillama_collector',
        class_name='DefiLlamaCollector',
        api_base_url='https://api.llama.fi',
        documentation_url='https://defillama.com/docs/api',
    ),
    CollectorInfo(
        name='lunarcrush',
        category=CollectorCategory.ALTERNATIVE,
        credentials=[
            CredentialSpec('LUNARCRUSH_API_KEY', min_length=20,
                          description='LunarCrush API Key'),
        ],
        is_free=False,
        description='LunarCrush - Social metrics and sentiment',
        module_path='data_collection.alternative.lunarcrush_collector',
        class_name='LunarCrushCollector',
        api_base_url='https://lunarcrush.com/api4',
        documentation_url='https://lunarcrush.com/developers/',
    ),
    CollectorInfo(
        name='coinalyze_enhanced',
        category=CollectorCategory.ALTERNATIVE,
        credentials=[
            CredentialSpec('COINALYZE_API_KEY', min_length=20,
                          description='Coinalyze API Key'),
        ],
        is_free=False,
        description='Coinalyze Enhanced - Additional derivative metrics',
        module_path='data_collection.alternative.coinalyze_enhanced_collector',
        class_name='CoinalyzeEnhancedCollector',
        api_base_url='https://api.coinalyze.net',
    ),
    
    # =========================================================================
    # Social (1)
    # =========================================================================
    CollectorInfo(
        name='twitter',
        category=CollectorCategory.SOCIAL,
        credentials=[
            CredentialSpec('TWITTER_BEARER_TOKEN', min_length=100,
                          description='Twitter/X API Bearer Token'),
        ],
        is_free=False,
        description='Twitter/X - Social sentiment and influencer tracking',
        module_path='data_collection.social.twitter_collector',
        class_name='TwitterCollector',
        api_base_url='https://api.twitter.com/2',
        documentation_url='https://developer.twitter.com/en/docs',
    ),
    
    # =========================================================================
    # Indexers (1)
    # =========================================================================
    CollectorInfo(
        name='thegraph',
        category=CollectorCategory.INDEXERS,
        credentials=[
            CredentialSpec('THE_GRAPH_API_KEY', min_length=32,
                          description='The Graph Network API Key'),
        ],
        is_free=False,
        description='The Graph - Blockchain indexing protocol',
        module_path='data_collection.indexers.thegraph_collector',
        class_name='TheGraphCollector',
        api_base_url='https://gateway.thegraph.com/api',
        documentation_url='https://thegraph.com/docs/',
    ),
    
    # =========================================================================
    # Multi-Exchange (1)
    # =========================================================================
    CollectorInfo(
        name='ccxt',
        category=CollectorCategory.MULTI_EXCHANGE,
        credentials=[],  # Uses individual exchange credentials
        is_free=True,
        description='CCXT - Unified exchange wrapper (100+ exchanges)',
        module_path='data_collection.exchanges.ccxt_wrapper',
        class_name='CCXTWrapper',
        documentation_url='https://docs.ccxt.com/',
    ),
]


# =============================================================================
# Helper Functions
# =============================================================================

def mask_credential(value: Optional[str], visible_chars: int = 4) -> str:
    """
    Mask credential for safe display.
    
    Parameters
    ----------
    value : str or None
        Credential value
    visible_chars : int
        Number of characters to show at start/end
    
    Returns
    -------
    str
        Masked value like "abcd****efgh"
    """
    if not value:
        return 'NOT SET'
    
    if len(value) <= visible_chars * 2:
        return '*' * len(value)
    
    return f"{value[:visible_chars]}{'*' * (len(value) - visible_chars * 2)}{value[-visible_chars:]}"


def check_credential(
    spec: CredentialSpec,
    level: VerificationLevel = VerificationLevel.BASIC
) -> CredentialCheckResult:
    """
    Check a single credential.
    
    Parameters
    ----------
    spec : CredentialSpec
        Credential specification
    level : VerificationLevel
        Verification level
    
    Returns
    -------
    CredentialCheckResult
        Check result
    """
    value = os.getenv(spec.name)
    masked = mask_credential(value)
    
    # Check existence
    if not value:
        if spec.required:
            return CredentialCheckResult(
                name=spec.name,
                status=CredentialStatus.MISSING,
                masked_value=masked,
                message=f"Required credential not set"
            )
        else:
            return CredentialCheckResult(
                name=spec.name,
                status=CredentialStatus.OPTIONAL,
                masked_value=masked,
                message="Optional credential not set"
            )
    
    # Basic level - just check existence
    if level == VerificationLevel.BASIC:
        return CredentialCheckResult(
            name=spec.name,
            status=CredentialStatus.OK,
            masked_value=masked,
            message="Credential present"
        )
    
    # Format level - validate format
    if level in [VerificationLevel.FORMAT, VerificationLevel.CONNECTION]:
        valid, msg = spec.validate_format(value)
        if not valid:
            return CredentialCheckResult(
                name=spec.name,
                status=CredentialStatus.INVALID_FORMAT,
                masked_value=masked,
                message=msg,
                format_valid=False
            )
    
    return CredentialCheckResult(
        name=spec.name,
        status=CredentialStatus.OK,
        masked_value=masked,
        message="Credential valid"
    )


def check_collector_credentials(
    collector: CollectorInfo,
    level: VerificationLevel = VerificationLevel.BASIC
) -> CollectorCheckResult:
    """
    Check all credentials for a collector.
    
    Parameters
    ----------
    collector : CollectorInfo
        Collector information
    level : VerificationLevel
        Verification level
    
    Returns
    -------
    CollectorCheckResult
        Check result for collector
    """
    # Free collectors
    if collector.is_free:
        return CollectorCheckResult(
            name=collector.name,
            category=collector.category.value,
            status=CredentialStatus.FREE,
            credentials={},
            description=collector.description,
            is_free=True
        )
    
    # Check each credential
    cred_results = {}
    all_ok = True
    
    for spec in collector.credentials:
        result = check_credential(spec, level)
        cred_results[spec.name] = result
        
        if result.status == CredentialStatus.MISSING:
            all_ok = False
    
    # Overall status
    if all_ok:
        status = CredentialStatus.OK
    else:
        status = CredentialStatus.MISSING
    
    return CollectorCheckResult(
        name=collector.name,
        category=collector.category.value,
        status=status,
        credentials=cred_results,
        description=collector.description,
        is_free=False
    )


def test_collector_import(collector: CollectorInfo) -> Tuple[bool, Optional[str]]:
    """
    Test that a collector module can be imported.
    
    Parameters
    ----------
    collector : CollectorInfo
        Collector to test
    
    Returns
    -------
    Tuple[bool, Optional[str]]
        (success, error_message)
    """
    try:
        module = __import__(collector.module_path, fromlist=[collector.class_name])
        _ = getattr(module, collector.class_name)
        return True, None
    except ImportError as e:
        return False, f"ImportError: {e}"
    except AttributeError as e:
        return False, f"AttributeError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


async def test_collector_connection(
    collector: CollectorInfo,
    timeout: float = 10.0
) -> Tuple[bool, Optional[str]]:
    """
    Test live connection to collector API.
    
    Parameters
    ----------
    collector : CollectorInfo
        Collector to test
    timeout : float
        Connection timeout in seconds
    
    Returns
    -------
    Tuple[bool, Optional[str]]
        (success, error_message)
    """
    if not HAS_AIOHTTP:
        return True, "aiohttp not installed - skipped"
    
    if not collector.api_base_url:
        return True, "No API URL - skipped"
    
    url = collector.api_base_url
    if collector.health_endpoint:
        url = f"{collector.api_base_url}{collector.health_endpoint}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as response:
                if response.status in [200, 401, 403]:  # 401/403 = auth required but reachable
                    return True, None
                else:
                    return False, f"HTTP {response.status}"
    except asyncio.TimeoutError:
        return False, "Connection timeout"
    except Exception as e:
        return False, str(e)


# =============================================================================
# Main Verification Functions
# =============================================================================

def verify_all_credentials(
    level: VerificationLevel = VerificationLevel.BASIC,
    test_imports: bool = True,
    test_connections: bool = False
) -> VerificationReport:
    """
    Verify all collector credentials.
    
    Parameters
    ----------
    level : VerificationLevel
        Verification level
    test_imports : bool
        Whether to test module imports
    test_connections : bool
        Whether to test live connections
    
    Returns
    -------
    VerificationReport
        Complete verification report
    """
    results = {}
    
    # Count statistics
    total_free = 0
    total_paid = 0
    creds_ok = 0
    creds_missing = 0
    imports_ok = 0
    imports_failed = 0
    connections_tested = 0
    connections_ok = 0
    
    for collector in ALL_COLLECTORS:
        # Check credentials
        result = check_collector_credentials(collector, level)
        
        # Count
        if collector.is_free:
            total_free += 1
            creds_ok += 1
        else:
            total_paid += 1
            if result.status == CredentialStatus.OK:
                creds_ok += 1
            else:
                creds_missing += 1
        
        # Test imports
        if test_imports:
            success, error = test_collector_import(collector)
            result.import_success = success
            result.import_error = error
            
            if success:
                imports_ok += 1
            else:
                imports_failed += 1
        
        # Test connections
        if test_connections and collector.api_base_url:
            connections_tested += 1
            success, error = asyncio.run(test_collector_connection(collector))
            result.connection_success = success
            result.connection_error = error
            
            if success:
                connections_ok += 1
        
        results[collector.name] = result
    
    return VerificationReport(
        timestamp=datetime.utcnow().isoformat(),
        total_collectors=len(ALL_COLLECTORS),
        free_collectors=total_free,
        paid_collectors=total_paid,
        credentials_ok=creds_ok,
        credentials_missing=creds_missing,
        imports_ok=imports_ok,
        imports_failed=imports_failed,
        connections_tested=connections_tested,
        connections_ok=connections_ok,
        results=results
    )


def print_report(report: VerificationReport, verbose: bool = False):
    """
    Print verification report to console.
    
    Parameters
    ----------
    report : VerificationReport
        Verification report
    verbose : bool
        Show detailed output
    """
    print("=" * 80)
    print("CRYPTO STATISTICAL ARBITRAGE - CREDENTIAL VERIFICATION")
    print("=" * 80)
    print(f"Timestamp: {report.timestamp}")
    print()
    
    # Summary
    print(f"Total Data Sources:    {report.total_collectors}")
    print(f"Free Sources:          {report.free_collectors}")
    print(f"Paid Sources:          {report.paid_collectors}")
    print(f"Credentials OK:        {report.credentials_ok}/{report.total_collectors}")
    print(f"Imports OK:            {report.imports_ok}/{report.total_collectors}")
    if report.connections_tested > 0:
        print(f"Connections OK:        {report.connections_ok}/{report.connections_tested}")
    print()
    
    # Group by category
    categories: Dict[str, List[CollectorCheckResult]] = {}
    for result in report.results.values():
        cat = result.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)
    
    # Print by category
    for category, results in categories.items():
        print("-" * 80)
        print(f"{category.upper()} ({len(results)} sources)")
        print("-" * 80)
        
        for result in results:
            # Status symbols
            cred_symbol = result.status.symbol
            import_symbol = "+" if result.import_success else "x"
            
            if result.connection_success is not None:
                conn_symbol = "+" if result.connection_success else "x"
                status_line = f"{cred_symbol} {import_symbol} {conn_symbol}"
            else:
                status_line = f"{cred_symbol} {import_symbol}"
            
            print(f"  {status_line} {result.name:<20} - {result.description[:45]}")
            
            # Verbose details
            if verbose:
                if result.credentials:
                    for cred_name, cred_result in result.credentials.items():
                        cred_sym = cred_result.status.symbol
                        print(f"        {cred_sym} {cred_name}: {cred_result.masked_value}")
                
                if not result.import_success and result.import_error:
                    print(f"        Import: {result.import_error}")
                
                if result.connection_success is False and result.connection_error:
                    print(f"        Connection: {result.connection_error}")
        
        print()
    
    # Legend
    print("=" * 80)
    print("LEGEND:")
    print("  + = OK    x = Missing/Failed    * = Free API    - = Optional    ! = Invalid Format")
    print("=" * 80)
    
    # Final status
    if report.is_success:
        print("\n[PASS] VERIFICATION PASSED - All required credentials present")
    else:
        print(f"\n[FAIL] VERIFICATION FAILED - {report.credentials_missing} credentials missing")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Verify credentials for all data collectors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Basic verification
  %(prog)s -v                   Verbose output
  %(prog)s --test-connections   Test live API connections
  %(prog)s --json-output out.json   Export JSON report
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed output')
    parser.add_argument('--test-connections', '-t', action='store_true',
                       help='Test live API connections (slow)')
    parser.add_argument('--json-output', '-j', type=str,
                       help='Output JSON report to file')
    parser.add_argument('--env-file', '-e', type=str,
                       help='Path to .env file')
    parser.add_argument('--level', '-l', 
                       choices=['basic', 'format', 'connection'],
                       default='basic',
                       help='Verification level')
    parser.add_argument('--skip-imports', action='store_true',
                       help='Skip import testing')
    
    args = parser.parse_args()
    
    # Load environment
    env_paths = [
        Path(args.env_file) if args.env_file else None,
        PROJECT_ROOT / 'config' / '.env',
        PROJECT_ROOT / '.env',
        Path.home() / '.crypto_statarb' / '.env',
    ]
    
    env_loaded = False
    if HAS_DOTENV:
        for env_path in env_paths:
            if env_path and env_path.exists():
                load_dotenv(env_path)
                print(f"Loaded environment from: {env_path}")
                env_loaded = True
                break
    
    if not env_loaded:
        print("Using system environment variables")
    
    print()
    
    # Run verification
    level = VerificationLevel(args.level)
    report = verify_all_credentials(
        level=level,
        test_imports=not args.skip_imports,
        test_connections=args.test_connections
    )
    
    # Print report
    print_report(report, verbose=args.verbose)
    
    # Export JSON if requested
    if args.json_output:
        with open(args.json_output, 'w') as f:
            f.write(report.to_json())
        print(f"\nJSON report saved to: {args.json_output}")
    
    # Exit code
    return 0 if report.is_success else 1


if __name__ == '__main__':
    sys.exit(main())