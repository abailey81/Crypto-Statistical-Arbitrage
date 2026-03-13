"""
Collection Manager for Crypto Statistical Arbitrage System.

This module provides unified orchestration for data collection across 47+ venues.
It handles collector instantiation, credential verification, task scheduling,
and concurrent execution with proper rate limiting.

==============================================================================
CRITICAL: IMPORT PATH MAPPING
==============================================================================

This module correctly maps collector names to their hierarchical import paths:
- 'binance' -> 'data_collection.cex.binance_collector'
- 'hyperliquid' -> 'data_collection.hybrid.hyperliquid_collector'
- etc.

The package structure is:
    data_collection/
     cex/ (binance, bybit, okx, coinbase, kraken, cme)
     hybrid/ (hyperliquid, dydx)
     dex/ (uniswap, sushiswap, curve, etc.)
     options/ (deribit, aevo, lyra, dopex)
     onchain/ (glassnode, santiment, etc.)
     market_data/ (coingecko, cryptocompare, messari, kaiko)
     indexers/ (thegraph)
     alternative/ (defillama, coinalyze, etc.)

Version: 3.0.0 (Consolidated)
"""

import asyncio
import importlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import pandas as pd

from .utils.rate_limiter import (
    RateLimiter, AdaptiveRateLimiter, MultiRateLimiter,
    create_venue_limiters, VenueRateLimitConfig
)
from .utils.monitoring import (
    get_monitor, record_error, record_success, update_metrics, is_blacklisted
)
from .utils.batch_optimizer import (
    BatchOptimizer, supports_batch, create_batches, estimate_speedup
)
from .utils.parallel_processor import (
    ParallelSymbolProcessor, ParallelCollectionManager as ParallelManager,
    get_venue_config, BatchResult, SymbolResult
)
from .utils.symbol_universe import SymbolUniverse, get_symbol_universe

logger = logging.getLogger(__name__)

# =============================================================================
# ENUMS
# =============================================================================

class VenueType(Enum):
    """Venue type classification."""
    CEX = "cex"
    HYBRID = "hybrid"
    DEX = "dex"
    OPTIONS = "options"
    ONCHAIN = "onchain"
    MARKET_DATA = "market_data"
    INDEXERS = "indexers"
    ALTERNATIVE = "alternative"

class CollectionStatus(Enum):
    """Collection task status."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    PARTIAL = auto()
    SKIPPED = auto()

class CollectionPriority(Enum):
    """Collection task priority."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

class HealthStatus(Enum):
    """Venue health status."""
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CollectorConfig:
    """Configuration for a single collector."""
    venue: str
    venue_type: VenueType
    module_path: str
    class_name: str
    api_key_env: Optional[str] = None
    api_secret_env: Optional[str] = None
    rate_limit_per_minute: int = 60
    supported_data_types: List[str] = field(default_factory=list)
    requires_auth: bool = True
    enabled: bool = True

    def __post_init__(self):
        if not self.supported_data_types:
            self.supported_data_types = ['funding_rates', 'ohlcv']

# =============================================================================
# COLLECTOR CONFIGURATIONS (47 VENUES) - ALL ENABLED
# =============================================================================

COLLECTOR_CONFIGS: Dict[str, CollectorConfig] = {
    # -------------------------------------------------------------------------
    # CEX (Centralized Exchanges) - 6 venues
    # -------------------------------------------------------------------------
    'binance': CollectorConfig(
        venue='binance',
        venue_type=VenueType.CEX,
        module_path='data_collection.cex.binance_collector',
        class_name='BinanceCollector',
        api_key_env='BINANCE_API_KEY',
        api_secret_env='BINANCE_SECRET_KEY',
        rate_limit_per_minute=1200,
        supported_data_types=['funding_rates', 'ohlcv', 'open_interest', 'trades'],
        requires_auth=False, # Public endpoints available
    ),
    'bybit': CollectorConfig(
        venue='bybit',
        venue_type=VenueType.CEX,
        module_path='data_collection.cex.bybit_collector',
        class_name='BybitCollector',
        api_key_env='BYBIT_API_KEY',
        api_secret_env='BYBIT_SECRET_KEY',
        rate_limit_per_minute=120,
        supported_data_types=['funding_rates', 'ohlcv', 'open_interest'],
        requires_auth=False,
    ),
    'okx': CollectorConfig(
        venue='okx',
        venue_type=VenueType.CEX,
        module_path='data_collection.cex.okx_collector',
        class_name='OKXCollector',
        api_key_env='OKX_API_KEY',
        api_secret_env='OKX_SECRET_KEY',
        rate_limit_per_minute=60,
        supported_data_types=['funding_rates', 'ohlcv', 'open_interest'],
        requires_auth=False,
    ),
    'coinbase': CollectorConfig(
        venue='coinbase',
        venue_type=VenueType.CEX,
        module_path='data_collection.cex.coinbase_collector',
        class_name='CoinbaseCollector',
        api_key_env='COINBASE_API_KEY',
        api_secret_env='COINBASE_SECRET_KEY',
        rate_limit_per_minute=600,
        supported_data_types=['ohlcv', 'trades'], # Spot only - no perpetual futures
        requires_auth=True,
    ),
    'kraken': CollectorConfig(
        venue='kraken',
        venue_type=VenueType.CEX,
        module_path='data_collection.cex.kraken_collector',
        class_name='KrakenCollector',
        api_key_env='KRAKEN_API_KEY',
        api_secret_env='KRAKEN_PRIVATE_KEY',
        rate_limit_per_minute=900,
        supported_data_types=['funding_rates', 'ohlcv'],
        requires_auth=True,
    ),
    'cme': CollectorConfig(
        venue='cme',
        venue_type=VenueType.CEX,
        module_path='data_collection.cex.cme_collector',
        class_name='CMECollector',
        api_key_env=None, # Yahoo Finance is FREE - no API key needed
        rate_limit_per_minute=60,
        supported_data_types=['futures', 'ohlcv', 'funding_rates'],
        requires_auth=False, # Yahoo Finance is publicly accessible
        enabled=True, # ENABLED: Uses FREE Yahoo Finance (BTC=F ticker)
    ),

    # -------------------------------------------------------------------------
    # HYBRID (On-chain settlement, off-chain matching) - 3 venues
    # CRITICAL: These use 1-HOUR funding intervals!
    # -------------------------------------------------------------------------
    'hyperliquid': CollectorConfig(
        venue='hyperliquid',
        venue_type=VenueType.HYBRID,
        module_path='data_collection.hybrid.hyperliquid_collector',
        class_name='HyperliquidCollector',
        api_key_env=None, # Fully public
        rate_limit_per_minute=60,  # Conservative per PDF: "Lower rate limits than Binance"
        supported_data_types=['funding_rates', 'ohlcv', 'open_interest', 'trades'],
        requires_auth=False,
    ),
    'dydx': CollectorConfig(
        venue='dydx',
        venue_type=VenueType.HYBRID,
        module_path='data_collection.hybrid.dydx_collector',
        class_name='DYDXCollector',
        api_key_env=None, # V4 is fully public
        rate_limit_per_minute=100,
        supported_data_types=['funding_rates', 'ohlcv', 'open_interest', 'trades'],
        requires_auth=False,
    ),
    'drift': CollectorConfig(
        venue='drift',
        venue_type=VenueType.HYBRID,
        module_path='data_collection.hybrid.drift_collector',
        class_name='DriftCollector',
        api_key_env=None, # Fully public API
        rate_limit_per_minute=100,
        supported_data_types=['funding_rates', 'open_interest', 'liquidations'],
        requires_auth=False,
    ),

    # -------------------------------------------------------------------------
    # DEX (Decentralized Exchanges) - 11 venues
    # -------------------------------------------------------------------------
    'uniswap': CollectorConfig(
        venue='uniswap',
        venue_type=VenueType.DEX,
        module_path='data_collection.dex.uniswap_collector',
        class_name='UniswapCollector',
        api_key_env='THE_GRAPH_API_KEY',
        rate_limit_per_minute=60,
        supported_data_types=['pool_data', 'swaps', 'liquidity'],
        requires_auth=True,
    ),
    'sushiswap': CollectorConfig(
        venue='sushiswap',
        venue_type=VenueType.DEX,
        module_path='data_collection.dex.sushiswap_v2_collector',
        class_name='SushiSwapV2Collector',
        api_key_env='THE_GRAPH_API_KEY',
        rate_limit_per_minute=60,
        supported_data_types=['pool_data', 'swaps'],
        requires_auth=True,
    ),
    'curve': CollectorConfig(
        venue='curve',
        venue_type=VenueType.DEX,
        module_path='data_collection.dex.curve_collector',
        class_name='CurveCollector',
        api_key_env=None,
        rate_limit_per_minute=60,
        supported_data_types=['pool_data', 'swaps'],
        requires_auth=False,
    ),
    'geckoterminal': CollectorConfig(
        venue='geckoterminal',
        venue_type=VenueType.DEX,
        module_path='data_collection.dex.geckoterminal_collector',
        class_name='GeckoTerminalCollector',
        api_key_env=None,
        rate_limit_per_minute=60,
        supported_data_types=['pool_data', 'ohlcv'],
        requires_auth=False,
    ),
    'dexscreener': CollectorConfig(
        venue='dexscreener',
        venue_type=VenueType.DEX,
        module_path='data_collection.dex.dexscreener_collector',
        class_name='DexScreenerCollector',
        api_key_env=None,
        rate_limit_per_minute=60,
        supported_data_types=['pool_data', 'ohlcv'],
        requires_auth=False,
    ),
    'gmx': CollectorConfig(
        venue='gmx',
        venue_type=VenueType.DEX,
        module_path='data_collection.dex.gmx_collector',
        class_name='GMXCollector',
        api_key_env=None,
        rate_limit_per_minute=60,
        supported_data_types=['funding_rates', 'open_interest', 'positions', 'ohlcv'],
        requires_auth=False,
    ),
    'vertex': CollectorConfig(
        venue='vertex',
        venue_type=VenueType.DEX,
        module_path='data_collection.dex.vertex_collector',
        class_name='VertexCollector',
        api_key_env=None,
        rate_limit_per_minute=60,
        supported_data_types=['funding_rates', 'ohlcv'],
        requires_auth=False,
        enabled=False, # Vertex Protocol shut down Aug 2025
    ),
    'jupiter': CollectorConfig(
        venue='jupiter',
        venue_type=VenueType.DEX,
        module_path='data_collection.dex.jupiter_collector',
        class_name='JupiterCollector',
        api_key_env='JUPITER_API_KEY',
        rate_limit_per_minute=60,
        supported_data_types=['swaps', 'routes'],
        requires_auth=True,
        enabled=False,  # DISABLED: Only swap routes (no OHLCV/funding), lite-api only has quote endpoint
    ),
    'cowswap': CollectorConfig(
        venue='cowswap',
        venue_type=VenueType.DEX,
        module_path='data_collection.dex.cowswap_collector',
        class_name='CowSwapCollector',
        api_key_env=None,
        rate_limit_per_minute=60,
        supported_data_types=['swaps', 'orders'],
        requires_auth=False,
        enabled=False,  # DISABLED: Only swap/order data, no OHLCV/funding for statarb
    ),
    'oneinch': CollectorConfig(
        venue='oneinch',
        venue_type=VenueType.DEX,
        module_path='data_collection.dex.oneinch_collector',
        class_name='OneInchCollector',
        api_key_env='ONEINCH_API_KEY',
        rate_limit_per_minute=60,
        supported_data_types=['swaps', 'routes'],
        requires_auth=True,
        enabled=False,  # DISABLED: Only swap routes, no historical OHLCV/funding data
    ),
    'zerox': CollectorConfig(
        venue='zerox',
        venue_type=VenueType.DEX,
        module_path='data_collection.dex.zerox_collector',
        class_name='ZeroXCollector',
        api_key_env='ZEROX_API_KEY',
        rate_limit_per_minute=60,
        supported_data_types=['swaps', 'routes'],
        requires_auth=True,
    ),

    # -------------------------------------------------------------------------
    # OPTIONS - 4 venues
    # -------------------------------------------------------------------------
    'deribit': CollectorConfig(
        venue='deribit',
        venue_type=VenueType.OPTIONS,
        module_path='data_collection.options.deribit_collector',
        class_name='DeribitCollector',
        api_key_env='DERIBIT_CLIENT_ID',
        api_secret_env='DERIBIT_CLIENT_SECRET',
        rate_limit_per_minute=1200,
        supported_data_types=['funding_rates', 'options', 'ohlcv', 'open_interest', 'dvol'],
        requires_auth=False, # Public API endpoints available
    ),
    'aevo': CollectorConfig(
        venue='aevo',
        venue_type=VenueType.OPTIONS,
        module_path='data_collection.options.aevo_collector',
        class_name='AevoCollector',
        api_key_env='AEVO_API_KEY',
        api_secret_env='AEVO_API_SECRET',
        rate_limit_per_minute=100,
        supported_data_types=['options', 'ohlcv', 'funding_rates'],
        requires_auth=False,
    ),
    'lyra': CollectorConfig(
        venue='lyra',
        venue_type=VenueType.OPTIONS,
        module_path='data_collection.options.lyra_collector',
        class_name='LyraCollector',
        api_key_env=None,
        rate_limit_per_minute=60,
        supported_data_types=['options'],
        requires_auth=False,
        enabled=False, # Deprecated
    ),
    'dopex': CollectorConfig(
        venue='dopex',
        venue_type=VenueType.OPTIONS,
        module_path='data_collection.options.dopex_collector',
        class_name='DopexCollector',
        api_key_env=None,
        rate_limit_per_minute=60,
        supported_data_types=['options'],
        requires_auth=False,
        enabled=False, # Dopex API offline
    ),

    # -------------------------------------------------------------------------
    # ON-CHAIN ANALYTICS - 10 venues
    # -------------------------------------------------------------------------
    'glassnode': CollectorConfig(
        venue='glassnode',
        venue_type=VenueType.ONCHAIN,
        module_path='data_collection.onchain.glassnode_collector',
        class_name='GlassnodeCollector',
        api_key_env='GLASSNODE_API_KEY',
        rate_limit_per_minute=60,
        supported_data_types=['on_chain_metrics', 'exchange_flows'],
        requires_auth=True,
        enabled=False,  # DISABLED: Paid API - no key configured
    ),
    'santiment': CollectorConfig(
        venue='santiment',
        venue_type=VenueType.ONCHAIN,
        module_path='data_collection.onchain.santiment_collector',
        class_name='SantimentCollector',
        api_key_env='SANTIMENT_API_KEY',
        rate_limit_per_minute=60,
        supported_data_types=['on_chain_metrics', 'social', 'ohlcv'],
        requires_auth=True,
        enabled=False,  # DISABLED: Paid - rate limited for 2+ weeks on free plan (429)
    ),
    'cryptoquant': CollectorConfig(
        venue='cryptoquant',
        venue_type=VenueType.ONCHAIN,
        module_path='data_collection.onchain.cryptoquant_collector',
        class_name='CryptoQuantCollector',
        api_key_env='CRYPTOQUANT_API_KEY',
        rate_limit_per_minute=60,
        supported_data_types=['on_chain_metrics', 'exchange_flows'],
        requires_auth=True,
        enabled=False,  # DISABLED: Paid API - no key configured
    ),
    'coinmetrics': CollectorConfig(
        venue='coinmetrics',
        venue_type=VenueType.ONCHAIN,
        module_path='data_collection.onchain.coinmetrics_collector',
        class_name='CoinMetricsCollector',
        api_key_env='COINMETRICS_API_KEY',
        rate_limit_per_minute=60,
        supported_data_types=['on_chain_metrics', 'network_data'],
        requires_auth=True,
        enabled=False,  # DISABLED: Paid API - no key configured
    ),
    'nansen': CollectorConfig(
        venue='nansen',
        venue_type=VenueType.ONCHAIN,
        module_path='data_collection.onchain.nansen_collector',
        class_name='NansenCollector',
        api_key_env='NANSEN_API_KEY',
        rate_limit_per_minute=60,
        supported_data_types=['wallet_analytics', 'smart_money'],
        requires_auth=True,
        enabled=False,  # DISABLED: Paid institutional API - restructured Aug 2025
    ),
    'arkham': CollectorConfig(
        venue='arkham',
        venue_type=VenueType.ONCHAIN,
        module_path='data_collection.onchain.arkham_collector',
        class_name='ArkhamCollector',
        api_key_env='ARKHAM_API_KEY',
        rate_limit_per_minute=60,
        supported_data_types=['entity_tracking', 'wallet_analytics'],
        requires_auth=True,
        enabled=False,  # DISABLED: Paid API - no key configured
    ),
    'flipside': CollectorConfig(
        venue='flipside',
        venue_type=VenueType.ONCHAIN,
        module_path='data_collection.onchain.flipside_collector',
        class_name='FlipsideCollector',
        api_key_env='FLIPSIDE_API_KEY',
        rate_limit_per_minute=60,
        supported_data_types=['on_chain_metrics'],
        requires_auth=True,
        enabled=False,  # DISABLED: Paid API - no key configured
    ),
    'covalent': CollectorConfig(
        venue='covalent',
        venue_type=VenueType.ONCHAIN,
        module_path='data_collection.onchain.covalent_collector',
        class_name='CovalentCollector',
        api_key_env='COVALENT_API_KEY',
        rate_limit_per_minute=60,
        supported_data_types=['wallet_analytics', 'token_balances'],
        requires_auth=True,
    ),
    'bitquery': CollectorConfig(
        venue='bitquery',
        venue_type=VenueType.ONCHAIN,
        module_path='data_collection.onchain.bitquery_collector',
        class_name='BitqueryCollector',
        api_key_env='BITQUERY_API_KEY',
        rate_limit_per_minute=10,
        supported_data_types=['on_chain_metrics', 'dex_trades'],
        requires_auth=True,
        enabled=False,  # DISABLED: Monthly quota exhausted (402), circuit breaker tripped
    ),
    'whale_alert': CollectorConfig(
        venue='whale_alert',
        venue_type=VenueType.ONCHAIN,
        module_path='data_collection.onchain.whale_alert_collector',
        class_name='WhaleAlertCollector',
        api_key_env='WHALE_ALERT_API_KEY',
        rate_limit_per_minute=60,
        supported_data_types=['large_transactions'],
        requires_auth=True,
        enabled=False,  # DISABLED: Paid API - no key configured
    ),

    # -------------------------------------------------------------------------
    # MARKET DATA PROVIDERS - 4 venues
    # -------------------------------------------------------------------------
    'coingecko': CollectorConfig(
        venue='coingecko',
        venue_type=VenueType.MARKET_DATA,
        module_path='data_collection.market_data.coingecko_collector',
        class_name='CoinGeckoCollector',
        api_key_env='COINGECKO_API_KEY',
        rate_limit_per_minute=15,  # Conservative: free tier ~10-30/min
        supported_data_types=['ohlcv', 'market_cap', 'volume'],
        requires_auth=False,
    ),
    'cryptocompare': CollectorConfig(
        venue='cryptocompare',
        venue_type=VenueType.MARKET_DATA,
        module_path='data_collection.market_data.cryptocompare_collector',
        class_name='CryptoCompareCollector',
        api_key_env='CRYPTOCOMPARE_API_KEY',
        rate_limit_per_minute=100,
        supported_data_types=['ohlcv', 'social'],
        requires_auth=True,
    ),
    'messari': CollectorConfig(
        venue='messari',
        venue_type=VenueType.MARKET_DATA,
        module_path='data_collection.market_data.messari_collector',
        class_name='MessariCollector',
        api_key_env='MESSARI_API_KEY',
        rate_limit_per_minute=60,
        supported_data_types=['asset_metrics', 'fundamentals'],
        requires_auth=True,
    ),
    'kaiko': CollectorConfig(
        venue='kaiko',
        venue_type=VenueType.MARKET_DATA,
        module_path='data_collection.market_data.kaiko_collector',
        class_name='KaikoCollector',
        api_key_env='KAIKO_API_KEY',
        rate_limit_per_minute=60,
        supported_data_types=['funding_rates', 'ohlcv', 'trades', 'order_book'],
        requires_auth=True,
        enabled=False,  # DISABLED: Paid API - no key configured
    ),

    # -------------------------------------------------------------------------
    # INDEXERS - 1 venue
    # -------------------------------------------------------------------------
    'thegraph': CollectorConfig(
        venue='thegraph',
        venue_type=VenueType.INDEXERS,
        module_path='data_collection.indexers.thegraph_collector',
        class_name='TheGraphCollector',
        api_key_env='THE_GRAPH_API_KEY',
        rate_limit_per_minute=1000,
        supported_data_types=['subgraph_data'],
        requires_auth=True,
    ),

    # -------------------------------------------------------------------------
    # ALTERNATIVE DATA - 5 venues
    # -------------------------------------------------------------------------
    'defillama': CollectorConfig(
        venue='defillama',
        venue_type=VenueType.ALTERNATIVE,
        module_path='data_collection.alternative.defillama_collector',
        class_name='DefiLlamaCollector',
        api_key_env=None,
        rate_limit_per_minute=60,
        supported_data_types=['tvl', 'yields', 'stablecoins'],
        requires_auth=False,
    ),
    'coinalyze': CollectorConfig(
        venue='coinalyze',
        venue_type=VenueType.ALTERNATIVE,
        module_path='data_collection.alternative.coinalyze_collector',
        class_name='CoinalyzeCollector',
        api_key_env='COINALYZE_API_KEY',
        rate_limit_per_minute=10,  # Very conservative - free tier is heavily limited
        supported_data_types=['funding_rates', 'open_interest', 'liquidations', 'ohlcv'],
        requires_auth=True,
    ),
    'coinalyze_enhanced': CollectorConfig(
        venue='coinalyze_enhanced',
        venue_type=VenueType.ALTERNATIVE,
        module_path='data_collection.alternative.coinalyze_enhanced_collector',
        class_name='CoinalyzeEnhancedCollector',
        api_key_env='COINALYZE_API_KEY',
        rate_limit_per_minute=60,
        supported_data_types=['funding_rates', 'open_interest', 'liquidations', 'ohlcv'],
        requires_auth=True,
        enabled=False, # Use coinalyze instead to avoid duplicates
    ),
    'lunarcrush': CollectorConfig(
        venue='lunarcrush',
        venue_type=VenueType.ALTERNATIVE,
        module_path='data_collection.alternative.lunarcrush_collector',
        class_name='LunarCrushCollector',
        api_key_env='LUNARCRUSH_API_KEY',
        rate_limit_per_minute=60,
        supported_data_types=['social', 'sentiment'],
        requires_auth=True,
        enabled=False,  # DISABLED: 402 Payment Required - no longer has free tier
    ),
    'dune': CollectorConfig(
        venue='dune',
        venue_type=VenueType.ALTERNATIVE,
        module_path='data_collection.alternative.dune_analytics_collector',
        class_name='DuneAnalyticsCollector',
        api_key_env='DUNE_API_KEY',
        rate_limit_per_minute=60,
        supported_data_types=['custom_queries'],
        requires_auth=True,
        enabled=False,  # DISABLED: Needs pre-built SQL queries, not automated for collection
    ),
}

# Venues with public endpoints (no API key required)
FREE_COLLECTORS: List[str] = [
    # CEX public endpoints
    'binance', 'bybit', 'okx', 'cme', # CME via Yahoo Finance (BTC=F) is FREE
    # Fully public hybrid venues
    'hyperliquid', 'dydx', 'drift',
    # DEX with public endpoints
    'geckoterminal', 'dexscreener', 'gmx', 'curve', 'cowswap',
    # Options venues with public endpoints
    'deribit', 'aevo',
    # Market data aggregators
    'defillama', 'coingecko',
]

# Venues that support FUNDING RATES
FUNDING_RATE_VENUES: List[str] = [
    # CEX (8h funding intervals)
    'binance', 'bybit', 'okx', 'kraken',
    # Hybrid (1h funding)
    'hyperliquid', 'dydx', 'drift',
    # Perp DEX
    'gmx',
    # Options with perps
    'deribit', 'aevo',
    # Aggregators
    'coinalyze',
    # 'kaiko',  # DISABLED: Paid API - no key configured
]

# Venues that support OHLCV data
OHLCV_VENUES: List[str] = [
    # CEX
    'binance', 'bybit', 'okx', 'coinbase', 'kraken',
    # Hybrid
    'hyperliquid', 'dydx',
    # DEX
    'gmx', 'geckoterminal', 'dexscreener',
    # Options
    'deribit', 'aevo',
    # Market data
    'coingecko', 'cryptocompare', 'coinalyze', 'santiment',
    # 'kaiko',  # DISABLED: Paid API - no key configured
]

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CollectionTask:
    """A single collection task."""
    venue: str
    data_type: str
    symbols: List[str]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    priority: CollectionPriority = CollectionPriority.NORMAL
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.start_date is None:
            self.start_date = datetime.now(timezone.utc) - timedelta(days=30)
        if self.end_date is None:
            self.end_date = datetime.now(timezone.utc)

@dataclass
class CollectionResult:
    """Result of a collection task."""
    venue: str
    data_type: str
    status: CollectionStatus
    data: Optional[pd.DataFrame] = None
    total_records: int = 0
    error: Optional[str] = None
    duration_seconds: float = 0.0
    symbols_collected: List[str] = field(default_factory=list)
    symbols_failed: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CollectionProgress:
    """Progress tracking for collection."""
    total_venues: int = 0
    completed_venues: int = 0
    failed_venues: int = 0
    total_records: int = 0
    start_time: Optional[datetime] = None
    current_venue: Optional[str] = None
    venue_progress: Dict[str, float] = field(default_factory=dict)

# =============================================================================
# COLLECTOR REGISTRY
# =============================================================================

class CollectorRegistry:
    """Registry for managing collector instances."""

    def __init__(self):
        self._collectors: Dict[str, Any] = {}
        self._import_errors: Dict[str, str] = {}

    def get_collector(self, venue: str, config: Optional[Dict] = None) -> Optional[Any]:
        """Get or create a collector instance."""
        if venue in self._collectors:
            return self._collectors[venue]

        collector_config = COLLECTOR_CONFIGS.get(venue)
        if not collector_config:
            logger.warning(f"Unknown venue: {venue}")
            return None

        if not collector_config.enabled:
            logger.debug(f"Collector {venue} is disabled")
            return None

        # OPTIMIZATION: Early API key check to prevent wasted collector instantiation
        # This prevents 68+ warnings like "Kaiko API key not provided"
        if collector_config.requires_auth and collector_config.api_key_env:
            api_key = os.getenv(collector_config.api_key_env, '')
            if not api_key:
                # Only log once per venue per session (not per call)
                if not hasattr(self, '_missing_key_logged'):
                    self._missing_key_logged = set()
                if venue not in self._missing_key_logged:
                    logger.info(f"[{venue}] Skipping - requires API key ({collector_config.api_key_env}) not set")
                    self._missing_key_logged.add(venue)
                return None

        try:
            module = importlib.import_module(collector_config.module_path)
            collector_class = getattr(module, collector_config.class_name)

            # Create collector with config
            collector = collector_class(config or {})
            self._collectors[venue] = collector
            return collector

        except ImportError as e:
            self._import_errors[venue] = str(e)
            logger.error(f"Failed to import collector for {venue}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to instantiate collector for {venue}: {e}")
            return None

    def get_available_venues(self) -> List[str]:
        """Get list of available venues."""
        available = []
        for venue, config in COLLECTOR_CONFIGS.items():
            if config.enabled:
                available.append(venue)
        return available

    def close_all(self):
        """Close all collector instances."""
        for venue, collector in self._collectors.items():
            try:
                if hasattr(collector, 'close'):
                    asyncio.create_task(collector.close())
            except Exception as e:
                logger.error(f"Error closing collector {venue}: {e}")
        self._collectors.clear()

# =============================================================================
# COLLECTION MANAGER
# =============================================================================

class CollectionManager:
    """
    Unified collection manager for all venues.

    Handles:
    - Collector instantiation and lifecycle
    - Parallel data collection
    - Rate limiting
    - Error handling and retries
    - Progress tracking
    """

    def __init__(
        self,
        max_concurrent_venues: int = 1,
        enable_symbol_parallelism: bool = True,
        max_symbols_per_venue: int = 10
    ):
        self.max_concurrent_venues = max_concurrent_venues
        self.enable_symbol_parallelism = enable_symbol_parallelism
        self.max_symbols_per_venue = max_symbols_per_venue

        self.registry = CollectorRegistry()
        self.progress = CollectionProgress()
        self.rate_limiters = create_venue_limiters()

        self._symbol_processors: Dict[str, ParallelSymbolProcessor] = {}
        self._start_time: Optional[float] = None
        self._symbol_universe = get_symbol_universe()

    def get_default_symbols(self, data_type: str = 'ohlcv') -> List[str]:
        """Get default symbols from centralized SymbolUniverse."""
        if data_type == 'funding_rates':
            return self._symbol_universe.get_funding_rate_symbols()
        elif data_type == 'futures_curve':
            return self._symbol_universe.get_futures_curve_symbols()
        elif data_type == 'options':
            return self._symbol_universe.get_options_symbols()
        else:
            return self._symbol_universe.get_ohlcv_symbols()

    def get_rate_limiter(self, venue: str) -> Optional[RateLimiter]:
        """Get rate limiter for a venue."""
        return self.rate_limiters.get(venue)

    def get_symbol_processor(self, venue: str) -> ParallelSymbolProcessor:
        """Get or create symbol processor for a venue."""
        if venue not in self._symbol_processors:
            rate_limiter = self.get_rate_limiter(venue)
            config = get_venue_config(venue)
            self._symbol_processors[venue] = ParallelSymbolProcessor(
                venue=venue,
                rate_limiter=rate_limiter,
                max_concurrent=config.get('max_concurrent', 5),
                batch_size=config.get('batch_size', 10)
            )
        return self._symbol_processors[venue]

    def is_available(self, collector_name: str) -> bool:
        """Check if a collector is available (enabled and has credentials)."""
        config = COLLECTOR_CONFIGS.get(collector_name)
        if not config:
            return False

        if not config.enabled:
            return False

        # Check API key if required
        if config.requires_auth and config.api_key_env:
            api_key = os.environ.get(config.api_key_env, '')
            if not api_key:
                return False

        return True

    def check_credentials(self) -> Dict[str, Dict]:
        """Check credentials for all collectors."""
        result = {}
        for name, config in COLLECTOR_CONFIGS.items():
            available, reason = self._check_credentials(config)
            result[name] = {
                'available': available and config.enabled,
                'reason': reason if not available else ('Disabled' if not config.enabled else 'Available'),
                'venue_type': config.venue_type.value,
                'rate_limit': config.rate_limit_per_minute,
                'data_types': config.supported_data_types,
            }
        return result

    def _check_credentials(self, config: CollectorConfig) -> Tuple[bool, str]:
        """Check if credentials are available for a collector."""
        if not config.requires_auth:
            return True, "No authentication required"

        if config.api_key_env:
            api_key = os.environ.get(config.api_key_env, '')
            if not api_key:
                return False, f"Missing {config.api_key_env}"

        if config.api_secret_env:
            api_secret = os.environ.get(config.api_secret_env, '')
            if not api_secret:
                return False, f"Missing {config.api_secret_env}"

        return True, "Credentials available"

    def get_collector(self, collector_name: str) -> Optional[Any]:
        """Get a collector instance."""
        config = COLLECTOR_CONFIGS.get(collector_name)
        if not config:
            logger.warning(f"Unknown collector: {collector_name}")
            return None

        if not config.enabled:
            logger.warning(f"Collector {collector_name} is disabled")
            return None

        return self.registry.get_collector(collector_name)

    async def _execute_tasks(self, tasks: List[CollectionTask]) -> List[CollectionResult]:
        """Execute collection tasks with concurrency control."""
        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent_venues)

        async def run_task(task: CollectionTask) -> CollectionResult:
            async with semaphore:
                return await self._collect_single(task)

        # Run tasks concurrently
        task_futures = [run_task(task) for task in tasks]
        results = await asyncio.gather(*task_futures, return_exceptions=True)

        # Convert exceptions to failed results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(CollectionResult(
                    venue=tasks[i].venue,
                    data_type=tasks[i].data_type,
                    status=CollectionStatus.FAILED,
                    error=str(result)
                ))
            else:
                final_results.append(result)

        return final_results

    async def _collect_single(self, task: CollectionTask) -> CollectionResult:
        """Execute a single collection task with timeout protection."""
        start_time = time.time()
        COLLECTION_TIMEOUT = 600 # 10 minutes per collection task

        try:
            collector = self.get_collector(task.venue)
            if not collector:
                return CollectionResult(
                    venue=task.venue,
                    data_type=task.data_type,
                    status=CollectionStatus.SKIPPED,
                    error="Collector not available"
                )

            # Format dates
            start_str = task.start_date.strftime('%Y-%m-%d') if task.start_date else None
            end_str = task.end_date.strftime('%Y-%m-%d') if task.end_date else None

            # Get data based on type with timeout protection
            if task.data_type == 'funding_rates':
                data = await asyncio.wait_for(
                    collector.fetch_funding_rates(
                        symbols=task.symbols,
                        start_date=start_str,
                        end_date=end_str,
                        **task.kwargs
                    ),
                    timeout=COLLECTION_TIMEOUT
                )
            elif task.data_type == 'ohlcv':
                timeframe = task.kwargs.pop('timeframe', '1h') # Remove from kwargs to avoid duplicate
                data = await asyncio.wait_for(
                    collector.fetch_ohlcv(
                        symbols=task.symbols,
                        timeframe=timeframe,
                        start_date=start_str,
                        end_date=end_str,
                        **task.kwargs
                    ),
                    timeout=COLLECTION_TIMEOUT
                )
            else:
                # Generic data type
                if hasattr(collector, f'fetch_{task.data_type}'):
                    fetch_method = getattr(collector, f'fetch_{task.data_type}')
                    data = await asyncio.wait_for(
                        fetch_method(
                            symbols=task.symbols,
                            start_date=start_str,
                            end_date=end_str,
                            **task.kwargs
                        ),
                        timeout=COLLECTION_TIMEOUT
                    )
                else:
                    return CollectionResult(
                        venue=task.venue,
                        data_type=task.data_type,
                        status=CollectionStatus.FAILED,
                        error=f"Data type {task.data_type} not supported"
                    )

            duration = time.time() - start_time

            if data is not None and len(data) > 0:
                return CollectionResult(
                    venue=task.venue,
                    data_type=task.data_type,
                    status=CollectionStatus.COMPLETED,
                    data=data,
                    total_records=len(data),
                    duration_seconds=duration,
                    symbols_collected=task.symbols
                )
            else:
                return CollectionResult(
                    venue=task.venue,
                    data_type=task.data_type,
                    status=CollectionStatus.PARTIAL,
                    data=data,
                    total_records=0,
                    duration_seconds=duration,
                    error="No data returned"
                )

        except asyncio.TimeoutError:
            logger.error(f"TIMEOUT: {task.venue} exceeded 300s timeout")
            return CollectionResult(
                venue=task.venue,
                data_type=task.data_type,
                status=CollectionStatus.FAILED,
                error="Timeout exceeded",
                duration_seconds=time.time() - start_time
            )
        except Exception as e:
            logger.error(f"Collection error for {task.venue}: {e}")
            return CollectionResult(
                venue=task.venue,
                data_type=task.data_type,
                status=CollectionStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time
            )

    async def collect_funding_rates(
        self,
        venues: List[str],
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs
    ) -> List[CollectionResult]:
        """Collect funding rates from multiple venues."""
        if symbols is None or len(symbols) == 0:
            symbols = self.get_default_symbols('funding_rates')
            logger.info(f"Using SymbolUniverse default: {len(symbols)} funding rate symbols")

        tasks = []
        for venue in venues:
            config = COLLECTOR_CONFIGS.get(venue)
            if not config:
                logger.warning(f"Unknown venue: {venue}")
                continue

            if 'funding_rates' not in config.supported_data_types:
                continue

            task = CollectionTask(
                venue=venue,
                data_type='funding_rates',
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                kwargs=kwargs
            )
            tasks.append(task)

        return await self._execute_tasks(tasks)

    async def collect_ohlcv(
        self,
        venues: List[str],
        symbols: Optional[List[str]] = None,
        timeframe: str = '1h',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs
    ) -> List[CollectionResult]:
        """Collect OHLCV data from multiple venues."""
        if symbols is None or len(symbols) == 0:
            symbols = self.get_default_symbols('ohlcv')
            logger.info(f"Using SymbolUniverse default: {len(symbols)} OHLCV symbols")

        tasks = []
        for venue in venues:
            config = COLLECTOR_CONFIGS.get(venue)
            if not config:
                continue

            if 'ohlcv' not in config.supported_data_types:
                continue

            task = CollectionTask(
                venue=venue,
                data_type='ohlcv',
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                kwargs={'timeframe': timeframe, **kwargs}
            )
            tasks.append(task)

        return await self._execute_tasks(tasks)

    async def collect_data_type(
        self,
        data_type: str,
        venues: List[str],
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs
    ) -> List[CollectionResult]:
        """Generic method to collect any data type from multiple venues."""
        if symbols is None or len(symbols) == 0:
            symbols = self.get_default_symbols(data_type)
            logger.info(f"Using SymbolUniverse default: {len(symbols)} symbols for {data_type}")

        tasks = []
        for venue in venues:
            config = COLLECTOR_CONFIGS.get(venue)
            if not config:
                logger.warning(f"Unknown venue: {venue}")
                continue

            if data_type not in config.supported_data_types:
                logger.debug(f"{venue} doesn't support {data_type}")
                continue

            task = CollectionTask(
                venue=venue,
                data_type=data_type,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                kwargs=kwargs
            )
            tasks.append(task)

        return await self._execute_tasks(tasks)

    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            'elapsed_seconds': elapsed,
            'completed_venues': self.progress.completed_venues,
            'failed_venues': self.progress.failed_venues,
            'total_records': self.progress.total_records,
            'records_per_second': self.progress.total_records / elapsed if elapsed > 0 else 0,
            'rate_limiter_stats': self.rate_limiters.get_all_stats() if hasattr(self.rate_limiters, 'get_all_stats') else {},
        }

    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up CollectionManager resources...")

        # Close symbol processors
        if self._symbol_processors:
            logger.info(f"Cleaning up {len(self._symbol_processors)} symbol processors...")
            for venue, processor in self._symbol_processors.items():
                try:
                    if hasattr(processor, 'close'):
                        await processor.close()
                except Exception as e:
                    logger.error(f"Error closing processor {venue}: {e}")
            self._symbol_processors.clear()

        # Close rate limiters
        logger.info("Closing rate limiters...")
        if hasattr(self.rate_limiters, 'close_all'):
            await self.rate_limiters.close_all()

        # Close all collectors
        self.registry.close_all()

        logger.info("CollectionManager cleanup complete")

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'VenueType',
    'CollectionStatus',
    'CollectionPriority',
    'HealthStatus',
    # Config
    'CollectorConfig',
    'COLLECTOR_CONFIGS',
    'FREE_COLLECTORS',
    'FUNDING_RATE_VENUES',
    'OHLCV_VENUES',
    # Data classes
    'CollectionTask',
    'CollectionResult',
    'CollectionProgress',
    # Core classes
    'CollectorRegistry',
    'CollectionManager',
    # Symbol universe
    'SymbolUniverse',
    'get_symbol_universe',
    # Parallel processing
    'ParallelSymbolProcessor',
    'ParallelManager',
]
