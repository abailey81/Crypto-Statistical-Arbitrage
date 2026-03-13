"""
Data Collection Package for Crypto Statistical Arbitrage System.
Version: 3.0.0 (Consolidated)
"""

import logging

logger = logging.getLogger(__name__)

__version__ = '3.0.0'

# Graceful imports
try:
    from .base_collector import (
        BaseCollector, VenueType, DataType, CollectionPhase,
        QualityGrade, CollectionStats, ValidationResult, CollectionMetadata,
    )
except ImportError as e:
    logger.debug(f"base_collector not available: {e}")
    BaseCollector = None

try:
    from .collection_manager import (
        CollectionManager, CollectorRegistry, CollectionTask, CollectionResult,
        CollectionProgress, CollectionStatus, CollectionPriority, HealthStatus,
        COLLECTOR_CONFIGS, FREE_COLLECTORS,
    )
except ImportError as e:
    logger.debug(f"collection_manager not available: {e}")
    CollectionManager = None
    COLLECTOR_CONFIGS = {}
    FREE_COLLECTORS = []

try:
    from .pipeline import (
        DataPipeline, PipelineConfig, PipelineStage, PipelineResult, run_pipeline,
    )
except ImportError as e:
    logger.debug(f"pipeline not available: {e}")
    DataPipeline = None

# =============================================================================
# CENTRALIZED SYMBOL UNIVERSE (200+ altcoins, 10x project requirement)
# =============================================================================
# CRITICAL: All symbol configuration comes from config/symbols.yaml
try:
    from .utils.symbol_universe import (
        SymbolUniverse,
        get_symbol_universe,
        get_all_symbols,
        get_ohlcv_symbols,
        get_funding_symbols,
    )
except ImportError as e:
    logger.debug(f"symbol_universe not available: {e}")
    SymbolUniverse = None
    get_symbol_universe = None

__all__ = [
    # Base collector
    'BaseCollector', 'VenueType', 'DataType', 'CollectionPhase', 'QualityGrade',
    'CollectionStats', 'ValidationResult', 'CollectionMetadata',
    # Collection manager
    'CollectionManager', 'CollectorRegistry', 'CollectionTask', 'CollectionResult',
    'CollectionProgress', 'CollectionStatus', 'CollectionPriority', 'HealthStatus',
    'COLLECTOR_CONFIGS', 'FREE_COLLECTORS',
    # Pipeline
    'DataPipeline', 'PipelineConfig', 'PipelineStage', 'PipelineResult', 'run_pipeline',
    # Symbol Universe (centralized 200+ altcoins, 10x project requirement)
    'SymbolUniverse', 'get_symbol_universe', 'get_all_symbols', 'get_ohlcv_symbols', 'get_funding_symbols',
]
