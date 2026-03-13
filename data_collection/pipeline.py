"""
Unified Data Pipeline for Crypto Statistical Arbitrage System.

This module consolidates data_pipeline.py and run_collection.py into a single,
coherent orchestration layer that provides:

1. DAG-Based Pipeline Processing
   - Collection → Cleaning → Validation → Alignment → Storage
   - Per-venue and cross-venue processing
   - Quality scoring and reporting

2. Integrated Data Cleaning
   - Schema enforcement
   - Deduplication
   - Temporal alignment
   - Outlier treatment
   - Funding rate normalization (CRITICAL for arb strategies)

3. Cross-Venue Operations
   - Timestamp alignment across venues
   - Correlation analysis
   - Spread calculation for arbitrage

==============================================================================
PIPELINE ARCHITECTURE
==============================================================================

Stage 1 - Collection (per venue):
    - Fetch data from venue APIs via collectors
    - Handle rate limiting and retries via NetworkManager
    - Convert to standardized DataFrame format

Stage 2 - Cleaning (per venue):
    - Schema enforcement
    - Deduplication
    - Timestamp alignment
    - Outlier treatment
    - Missing data handling
    - Symbol normalization
    - Rate normalization (funding rates)

Stage 3 - Validation (per venue):
    - Schema validation via DataValidator
    - Business rule validation
    - Statistical validation
    - Temporal consistency validation
    - Generate quality score

Stage 4 - Cross-Venue Alignment:
    - Align timestamps across venues
    - Filter to common symbols
    - Cross-venue consistency check
    - Correlation analysis

Stage 5 - Quality Scoring:
    - Calculate overall quality score
    - Generate quality report
    - Flag issues for review

Stage 6 - Storage:
    - Store cleaned data via DataStorage
    - Store metadata
    - Store quality reports

Stage 7 - Reporting:
    - Generate summary report
    - Log all operations
    - Alert on quality issues

Version: 3.0.0 (Consolidated)
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base_collector import (
    BaseCollector,
    CollectionMetadata,
    DataType,
    QualityGrade,
    ValidationResult,
    VenueType,
)
from .collection_manager import (
    CollectionManager,
    CollectionResult,
    CollectionStatus,
    CollectionTask,
    CollectorRegistry,
    COLLECTOR_CONFIGS,
    FREE_COLLECTORS,
    FUNDING_RATE_VENUES,
    OHLCV_VENUES,
)

# =============================================================================
# CENTRALIZED SYMBOL UNIVERSE (200+ altcoins, 10x project requirement)
# =============================================================================
# CRITICAL: All symbol configuration comes from config/symbols.yaml
from .utils.symbol_universe import (
    SymbolUniverse,
    get_symbol_universe,
    get_funding_symbols,
)

# =============================================================================
# INCREMENTAL CACHING
# =============================================================================
# Cache processed data so we only need to collect missing periods
try:
    from .utils.incremental_cache import (
        IncrementalCacheManager,
        get_cache_manager,
    )
    INCREMENTAL_CACHE_AVAILABLE = True
except ImportError:
    INCREMENTAL_CACHE_AVAILABLE = False

# comprehensive hierarchical cache (v2.0 - industry best practices)
try:
    from .utils.hierarchical_cache import (
        HierarchicalCache,
        BloomFilter,
        TimeLRUCache,
        L2DiskCache,
        get_hierarchical_cache,
    )
    HIERARCHICAL_CACHE_AVAILABLE = True
except ImportError:
    HIERARCHICAL_CACHE_AVAILABLE = False

# Import cross-venue reconciliation for comprehensive alignment
try:
    from .utils.cross_venue_reconciliation import (
        CrossVenueAligner,
        MultiVenueReconciler,
        CEXDEXReconciler,
        VENUE_CATEGORIES,
    )
    CROSS_VENUE_AVAILABLE = True
except ImportError:
    CROSS_VENUE_AVAILABLE = False

# =============================================================================
# GPU ACCELERATION (Optional - falls back to CPU if not available)
# =============================================================================
try:
    from .utils.gpu_accelerator import (
        GPUAccelerator,
        get_accelerator,
        is_gpu_available,
        AcceleratorConfig,
        ProcessingMode,
    )
    GPU_ACCELERATION_AVAILABLE = True
except ImportError:
    GPU_ACCELERATION_AVAILABLE = False
    GPUAccelerator = None
    get_accelerator = None
    is_gpu_available = lambda: False

# Initialize logger after imports but before usage
logger = logging.getLogger(__name__)

# Log cross-venue availability status
if not CROSS_VENUE_AVAILABLE:
    logger.debug("cross_venue_reconciliation not available, using basic alignment")

# =============================================================================
# ENUMS
# =============================================================================

class PipelineStage(Enum):
    """Pipeline processing stages."""
    COLLECTION = auto()
    CLEANING = auto()
    NORMALIZATION = auto()
    VALIDATION = auto()
    ALIGNMENT = auto()
    QUALITY_SCORING = auto()
    STORAGE = auto()
    REPORTING = auto()
    
    @property
    def order(self) -> int:
        """Execution order."""
        return {
            PipelineStage.COLLECTION: 1,
            PipelineStage.CLEANING: 2,
            PipelineStage.NORMALIZATION: 3,
            PipelineStage.VALIDATION: 4,
            PipelineStage.ALIGNMENT: 5,
            PipelineStage.QUALITY_SCORING: 6,
            PipelineStage.STORAGE: 7,
            PipelineStage.REPORTING: 8,
        }[self]

class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"

# =============================================================================
# CONFIGURATION
# =============================================================================

def _get_default_symbols() -> List[str]:
    """
    Get default symbols from centralized SymbolUniverse.

    CRITICAL: This ensures pipeline uses consistent symbol configuration.
    Returns funding rate priority symbols by default (for funding_rates data type).
    """
    return get_funding_symbols()

@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""

    # Data selection
    data_type: str = "funding_rates"
    venues: List[str] = field(default_factory=list)
    # CRITICAL: Symbols from centralized SymbolUniverse (config/symbols.yaml)
    symbols: List[str] = field(default_factory=_get_default_symbols)
    
    # Date range
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Cleaning options
    dedup_strategy: str = "keep_last"
    outlier_action: str = "cap" # 'remove', 'cap', 'flag', 'none'
    outlier_threshold: float = 5.0 # Z-score threshold
    missing_method: str = "forward_fill" # 'drop', 'forward_fill', 'interpolate'
    max_fill_periods: int = 3
    
    # Normalization (CRITICAL for funding rates)
    normalize_funding_rates: bool = True
    target_funding_interval: str = "8h" # Normalize all to 8-hour
    
    # Validation
    min_quality_score: float = 70.0
    validate_schemas: bool = True
    
    # Cross-venue alignment
    align_timestamps: bool = True
    reference_venue: Optional[str] = None # Venue to use as reference
    alignment_tolerance_minutes: int = 5
    
    # Output
    output_dir: str = "data/processed"
    save_intermediate: bool = False
    generate_report: bool = True

    # Execution - Parallel Processing (v2.0)
    max_concurrent: int = 12 # Increased from 5 for better parallelism
    enable_symbol_parallelism: bool = True # Enable parallel symbol processing

    # GPU Acceleration (v2.1)
    use_gpu: bool = True # Use GPU if available (auto-fallback to CPU)
    gpu_memory_fraction: float = 0.8 # Max GPU memory to use
    gpu_batch_size: int = 100_000 # Rows per GPU batch

    def __post_init__(self):
        if not self.venues:
            # Use the centralized venue lists from collection_manager
            # This ensures consistency across the codebase
            if self.data_type == 'funding_rates':
                # Use FUNDING_RATE_VENUES - venues that legitimately support funding rates
                # Note: Spot AMMs (Uniswap, SushiSwap, etc.) do NOT have funding rates by design
                self.venues = FUNDING_RATE_VENUES.copy()
            elif self.data_type == 'ohlcv':
                # Use OHLCV_VENUES - venues that support price data
                self.venues = OHLCV_VENUES.copy()
            elif self.data_type == 'open_interest':
                self.venues = [
                    # CEX
                    'binance', 'bybit', 'okx',
                    # Hybrid
                    'hyperliquid', 'dydx', 'vertex',
                    # Perp DEX and Options
                    'gmx', 'deribit', 'aevo',
                    # Aggregators
                    'coinalyze',
                ]
            else:
                # Fallback to comprehensive perp-capable venues
                self.venues = [
                    'binance', 'bybit', 'okx', 'hyperliquid', 'dydx',
                    'vertex', 'gmx', 'deribit', 'aevo', 'coinalyze',
                ]

            # Log venue selection for transparency
            logger.info(
                f"Auto-selected {len(self.venues)} venues for {self.data_type}: {self.venues}"
            )

        if self.end_date is None:
            self.end_date = datetime.now(timezone.utc)
        if self.start_date is None:
            self.start_date = self.end_date - timedelta(days=30)

# =============================================================================
# RESULT CLASSES
# =============================================================================

@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage: PipelineStage
    venue: str
    success: bool
    data: Optional[pd.DataFrame] = None
    records_in: int = 0
    records_out: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VenueResult:
    """Complete result for a single venue."""
    venue: str
    venue_type: VenueType
    data: Optional[pd.DataFrame] = None
    stage_results: Dict[PipelineStage, StageResult] = field(default_factory=dict)
    quality_score: float = 0.0
    quality_grade: QualityGrade = QualityGrade.POOR
    total_records: int = 0
    total_duration_seconds: float = 0.0
    
    @property
    def success(self) -> bool:
        return all(r.success for r in self.stage_results.values())

@dataclass
class PipelineResult:
    """Complete pipeline result."""
    config: PipelineConfig
    status: PipelineStatus = PipelineStatus.PENDING
    venue_results: Dict[str, VenueResult] = field(default_factory=dict)
    aligned_data: Optional[pd.DataFrame] = None
    cross_venue_metrics: Dict[str, Any] = field(default_factory=dict)
    overall_quality_score: float = 0.0
    total_records: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def success_rate(self) -> float:
        if not self.venue_results:
            return 0.0
        successful = sum(1 for r in self.venue_results.values() if r.success)
        return (successful / len(self.venue_results)) * 100
    
    def get_merged_data(self) -> Optional[pd.DataFrame]:
        """Get merged data from all venues."""
        if self.aligned_data is not None:
            return self.aligned_data
        
        dfs = []
        for venue, result in self.venue_results.items():
            if result.data is not None and not result.data.empty:
                df = result.data.copy()
                if 'venue' not in df.columns:
                    df['venue'] = venue
                dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return None
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "PIPELINE EXECUTION SUMMARY",
            "=" * 60,
            f"Status: {self.status.value.upper()}",
            f"Data Type: {self.config.data_type}",
            f"Venues: {len(self.venue_results)}",
            f"Total Records: {self.total_records:,}",
            f"Duration: {self.duration_seconds:.1f}s",
            f"Success Rate: {self.success_rate:.1f}%",
            f"Quality Score: {self.overall_quality_score:.1f}/100",
            "",
            "Per-Venue Results:",
        ]
        
        for venue, result in self.venue_results.items():
            status = "PASS" if result.success else "FAIL"
            lines.append(
                f" {status} {venue}: {result.total_records:,} records, "
                f"quality={result.quality_score:.1f}"
            )
        
        if self.cross_venue_metrics:
            lines.extend([
                "",
                "Cross-Venue Metrics:",
            ])
            for key, value in self.cross_venue_metrics.items():
                if isinstance(value, float):
                    lines.append(f" {key}: {value:.4f}")
                else:
                    lines.append(f" {key}: {value}")
        
        lines.append("=" * 60)
        return "\n".join(lines)

# =============================================================================
# PIPELINE IMPLEMENTATION
# =============================================================================

class DataPipeline:
    """
    Unified data pipeline for crypto statistical arbitrage.
    
    Orchestrates the complete data processing workflow:
    1. Collection from multiple venues
    2. Cleaning and normalization
    3. Validation and quality scoring
    4. Cross-venue alignment
    5. Storage and reporting
    
    Example
    -------
    >>> config = PipelineConfig(
    ... data_type='funding_rates',
    ... venues=['binance', 'hyperliquid', 'dydx'],
    ... start_date=datetime(2024, 1, 1),
    ... end_date=datetime(2024, 12, 31)
    ... )
    >>> pipeline = DataPipeline(config)
    >>> result = await pipeline.run()
    >>> print(result.summary())
    """
    
    def __init__(self, config: PipelineConfig, use_cache: bool = True):
        self.config = config
        self.use_cache = use_cache
        self.collection_manager = CollectionManager(
            max_concurrent_venues=config.max_concurrent,
            enable_symbol_parallelism=getattr(config, 'enable_symbol_parallelism', True)
        )
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy imports to avoid circular dependencies
        self._cleaner = None
        self._validator = None
        self._quality_checker = None
        self._normalizer = None
        self._storage = None
        self._cache_manager = None

        # GPU Acceleration initialization
        self._gpu_accelerator = None
        self._use_gpu = getattr(config, 'use_gpu', True) and GPU_ACCELERATION_AVAILABLE
        if self._use_gpu:
            try:
                gpu_config = AcceleratorConfig(
                    memory_fraction=getattr(config, 'gpu_memory_fraction', 0.8),
                    chunk_size=getattr(config, 'gpu_batch_size', 100_000),
                ) if AcceleratorConfig else None
                self._gpu_accelerator = get_accelerator(gpu_config) if get_accelerator else None
                if self._gpu_accelerator and self._gpu_accelerator.is_gpu_enabled:
                    logger.info(f"GPU acceleration enabled: {self._gpu_accelerator.gpu_info}")
                else:
                    logger.info("GPU acceleration: falling back to CPU (GPU not detected)")
                    self._use_gpu = False
            except Exception as e:
                logger.warning(f"GPU initialization failed, using CPU: {e}")
                self._use_gpu = False
    
    @property
    def cleaner(self):
        """Lazy load data cleaner."""
        if self._cleaner is None:
            try:
                from .utils.data_cleaner import CleaningPipeline
                self._cleaner = CleaningPipeline()
            except ImportError:
                logger.warning("data_cleaner not available")
        return self._cleaner
    
    @property
    def validator(self):
        """Lazy load data validator."""
        if self._validator is None:
            try:
                from .utils.data_validator import DataValidator
                self._validator = DataValidator()
            except ImportError:
                logger.warning("data_validator not available")
        return self._validator
    
    @property
    def quality_checker(self):
        """Lazy load quality checker."""
        if self._quality_checker is None:
            try:
                from .utils.quality_checks import QualityChecker
                self._quality_checker = QualityChecker()
            except ImportError:
                logger.warning("quality_checks not available")
        return self._quality_checker
    
    @property
    def normalizer(self):
        """Lazy load funding normalizer."""
        if self._normalizer is None:
            try:
                from .utils.funding_normalization import FundingNormalizer
                self._normalizer = FundingNormalizer()
            except ImportError:
                logger.warning("funding_normalization not available")
        return self._normalizer
    
    @property
    def storage(self):
        """Lazy load storage."""
        if self._storage is None:
            try:
                from .utils.storage import ParquetStorage
                self._storage = ParquetStorage(base_path=str(self.output_dir))
            except ImportError:
                logger.warning("storage not available")
        return self._storage

    @property
    def cache_manager(self):
        """
        Lazy load incremental cache manager.

        The cache manager provides:
        - Loading cached processed data
        - Determining gaps in cached data
        - Updating cache after processing
        """
        if self._cache_manager is None and INCREMENTAL_CACHE_AVAILABLE:
            try:
                self._cache_manager = get_cache_manager(str(self.output_dir))
            except Exception as e:
                logger.warning(f"Failed to initialize cache manager: {e}")
        return self._cache_manager

    async def run(self) -> PipelineResult:
        """
        Execute the complete pipeline.

        Returns
        -------
        PipelineResult
            Complete result with all venue data
        """
        result = PipelineResult(
            config=self.config,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now(timezone.utc)
        )

        logger.info(f"Starting pipeline for {len(self.config.venues)} venues")

        try:
            # Stage 1: Collection
            collection_results = await self._stage_collection()

            # Process each venue through remaining stages IN PARALLEL
            # This is CRITICAL for performance - processes all venues concurrently
            async def process_venue_wrapper(venue: str, col_result: CollectionResult):
                """Wrapper to return tuple of (venue, result) for parallel processing."""
                venue_result = await self._process_venue(venue, col_result)
                return venue, venue_result

            # Execute ALL venue processing in parallel using asyncio.gather
            venue_tasks = [
                process_venue_wrapper(venue, col_result)
                for venue, col_result in collection_results.items()
            ]
            venue_results_list = await asyncio.gather(*venue_tasks, return_exceptions=True)

            # Collect results from parallel processing
            for item in venue_results_list:
                if isinstance(item, Exception):
                    logger.error(f"Venue processing error: {item}")
                    continue
                venue, venue_result = item
                result.venue_results[venue] = venue_result
                result.total_records += venue_result.total_records

            # Stage 4: Cross-venue alignment
            if self.config.align_timestamps and len(result.venue_results) > 1:
                result.aligned_data, result.cross_venue_metrics = \
                    self._align_cross_venue(result.venue_results)

            # Calculate overall quality (only include venues with actual data)
            if result.venue_results:
                # Only include venues that collected records in the quality average
                scores = [
                    r.quality_score for r in result.venue_results.values()
                    if r.total_records > 0
                ]
                if scores:
                    result.overall_quality_score = sum(scores) / len(scores)
                else:
                    result.overall_quality_score = 0.0

            # Determine final status
            if all(r.success for r in result.venue_results.values()):
                result.status = PipelineStatus.COMPLETED
            elif any(r.success for r in result.venue_results.values()):
                result.status = PipelineStatus.PARTIAL
            else:
                result.status = PipelineStatus.FAILED

            # Stage 7: Generate report
            if self.config.generate_report:
                self._generate_report(result)

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.errors.append(str(e))
            logger.error(f"Pipeline failed: {e}")
        finally:
            # CRITICAL FIX: Clean up async resources to allow proper exit
            await self.cleanup()

        result.end_time = datetime.now(timezone.utc)
        logger.info(f"Pipeline completed: {result.status.value}")

        return result
    
    async def _stage_collection(self) -> Dict[str, CollectionResult]:
        """Execute collection stage for all venues."""
        logger.info(f"Stage 1: Collecting {self.config.data_type} from {self.config.venues}")

        # Use generic collect_data_type method for ALL data types
        # This supports funding_rates, ohlcv, pool_data, swaps, tvl, on_chain_metrics, etc.
        results = await self.collection_manager.collect_data_type(
            data_type=self.config.data_type,
            venues=self.config.venues,
            symbols=self.config.symbols,
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )

        return {r.venue: r for r in results}
    
    async def _process_venue(
        self,
        venue: str,
        collection_result: CollectionResult
    ) -> VenueResult:
        """Process a single venue through all stages."""
        config = COLLECTOR_CONFIGS.get(venue)
        venue_result = VenueResult(
            venue=venue,
            venue_type=config.venue_type if config else VenueType.CEX
        )
        
        data = collection_result.data
        if data is None or data.empty:
            logger.warning(f"No data collected for {venue}")
            return venue_result
        
        start_time = datetime.now(timezone.utc)
        
        # Stage 2: Cleaning
        cleaned_data, clean_result = self._stage_cleaning(venue, data)
        venue_result.stage_results[PipelineStage.CLEANING] = clean_result
        
        if cleaned_data is None or cleaned_data.empty:
            return venue_result
        
        # Stage 3: Normalization (for funding rates)
        if self.config.data_type == 'funding_rates' and self.config.normalize_funding_rates:
            normalized_data, norm_result = self._stage_normalization(venue, cleaned_data)
            venue_result.stage_results[PipelineStage.NORMALIZATION] = norm_result
            if normalized_data is not None:
                cleaned_data = normalized_data

        # Always add metadata enrichment (venue_type, contract_type for PDF compliance)
        cleaned_data = self._enrich_metadata(venue, cleaned_data)

        # Stage 4: Validation
        validated, val_result = self._stage_validation(venue, cleaned_data)
        venue_result.stage_results[PipelineStage.VALIDATION] = val_result
        venue_result.quality_score = val_result.metrics.get('quality_score', 0.0)
        
        # Determine quality grade
        if venue_result.quality_score >= 90:
            venue_result.quality_grade = QualityGrade.EXCELLENT
        elif venue_result.quality_score >= 80:
            venue_result.quality_grade = QualityGrade.GOOD
        elif venue_result.quality_score >= 70:
            venue_result.quality_grade = QualityGrade.ACCEPTABLE
        else:
            venue_result.quality_grade = QualityGrade.POOR
        
        # Stage 6: Storage
        if validated and venue_result.quality_score >= self.config.min_quality_score:
            storage_result = self._stage_storage(venue, cleaned_data)
            venue_result.stage_results[PipelineStage.STORAGE] = storage_result
        
        venue_result.data = cleaned_data
        venue_result.total_records = len(cleaned_data)
        venue_result.total_duration_seconds = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()
        
        return venue_result
    
    def _stage_cleaning(
        self,
        venue: str,
        data: pd.DataFrame
    ) -> Tuple[Optional[pd.DataFrame], StageResult]:
        """
        Execute comprehensive cleaning stage.

        Uses the DataCleaner module when available for:
        - Schema enforcement
        - Deduplication with configurable strategy
        - Temporal alignment with settlement time snapping
        - detailed outlier detection (LOF, Isolation Forest, ensemble)
        - Missing data handling with cross-venue fill support
        - Symbol normalization
        - OHLCV invariant correction
        """
        start_time = datetime.now(timezone.utc)
        result = StageResult(
            stage=PipelineStage.CLEANING,
            venue=venue,
            success=False,
            records_in=len(data)
        )

        try:
            cleaned = data.copy()

            # Try to use comprehensive DataCleaner
            if self._try_enhanced_cleaning(venue, cleaned, result):
                return result.data, result

            # Fallback to basic cleaning if DataCleaner not available
            logger.debug(f"Using basic cleaning for {venue}")
            cleaned = self._basic_cleaning(venue, cleaned, result)

            result.data = cleaned
            result.records_out = len(cleaned) if cleaned is not None else 0
            result.success = True

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Cleaning failed for {venue}: {e}")

        result.duration_seconds = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()

        return result.data, result

    def _try_enhanced_cleaning(
        self,
        venue: str,
        data: pd.DataFrame,
        result: StageResult
    ) -> bool:
        """
        Try to use comprehensive DataCleaner pipeline.

        Returns True if successful, False to fall back to basic cleaning.
        """
        try:
            from .utils.data_cleaner import (
                CleaningPipeline, CleaningReport, CleaningAction,
                SchemaEnforcementStage, DeduplicationStage, DeduplicationStrategy,
                TemporalAlignmentStage, OutlierTreatmentStage,
                MissingDataStage, SymbolNormalizationStage,
                OHLCVCorrectionStage, DataType, OutlierMethod, OutlierAction
            )

            # Determine data type
            # Map string data types to DataType enum
            data_type_mapping = {
                'funding_rates': DataType.FUNDING_RATES,
                'ohlcv': DataType.OHLCV,
                'open_interest': DataType.OPEN_INTEREST,
                'orderbook': DataType.ORDERBOOK,
                'trades': DataType.TRADES,
                'liquidations': DataType.LIQUIDATIONS,
                'options': DataType.OPTIONS_CHAIN,
                'options_chain': DataType.OPTIONS_CHAIN,
                'pool_data': DataType.DEX_POOLS,
                'dex_pools': DataType.DEX_POOLS,
                'swaps': DataType.DEX_SWAPS,
                'dex_swaps': DataType.DEX_SWAPS,
                'tvl': DataType.TVL,
                'on_chain_metrics': DataType.ONCHAIN_FLOWS,
                'onchain_flows': DataType.ONCHAIN_FLOWS,
                'exchange_flows': DataType.ONCHAIN_FLOWS,
                'social': DataType.SOCIAL_METRICS,
                'sentiment': DataType.SOCIAL_METRICS,
                'social_metrics': DataType.SOCIAL_METRICS,
            }

            # Use mapping or default to FUNDING_RATES for unmapped types
            # Unmapped types (wallet_analytics, asset_metrics, etc.) will use default schema
            data_type = data_type_mapping.get(self.config.data_type, DataType.FUNDING_RATES)

            if self.config.data_type not in data_type_mapping:
                logger.info(f"Data type '{self.config.data_type}' not explicitly mapped, using generic schema (treating as {data_type})")

            # Build cleaning report with required row/column statistics
            report = CleaningReport(
                venue=venue,
                data_type=data_type,
                original_rows=len(data),
                final_rows=len(data), # Will be updated as cleaning progresses
                original_columns=len(data.columns),
                final_columns=len(data.columns) # Will be updated as cleaning progresses
            )

            # Stage 1: Schema enforcement
            schema_stage = SchemaEnforcementStage()
            cleaned = schema_stage.execute(data, report, data_type)

            # Stage 2: Deduplication
            # Convert string strategy to enum
            dedup_strategy_map = {
                'keep_first': DeduplicationStrategy.KEEP_FIRST,
                'keep_last': DeduplicationStrategy.KEEP_LAST,
                'keep_best': DeduplicationStrategy.KEEP_BEST,
                'aggregate_mean': DeduplicationStrategy.AGGREGATE_MEAN,
            }
            dedup_strategy = dedup_strategy_map.get(
                self.config.dedup_strategy,
                DeduplicationStrategy.KEEP_LAST
            )

            dedup_stage = DeduplicationStage()
            cleaned = dedup_stage.execute(
                cleaned, report, data_type,
                strategy=dedup_strategy,
                key_columns=['timestamp', 'symbol'] if 'symbol' in cleaned.columns else ['timestamp']
            )

            # Stage 3: Temporal alignment (snap to settlement times)
            temporal_stage = TemporalAlignmentStage()
            cleaned = temporal_stage.execute(
                cleaned, report, data_type,
                venue=venue,
                snap_to_settlement=True
            )

            # Stage 4: Outlier treatment with detailed ensemble
            outlier_stage = OutlierTreatmentStage()

            # Use detailed ensemble for funding rates, IQR for others
            outlier_method = OutlierMethod.ADVANCED_ENSEMBLE if self.config.data_type == 'funding_rates' else OutlierMethod.IQR

            # Map config outlier action to OutlierAction enum
            action_map = {
                'remove': OutlierAction.REMOVE,
                'cap': OutlierAction.CAP_WINSORIZE,
                'flag': OutlierAction.FLAG_ONLY,
                'none': OutlierAction.FLAG_ONLY,
            }
            outlier_action = action_map.get(self.config.outlier_action, OutlierAction.CAP_WINSORIZE)

            cleaned = outlier_stage.execute(
                cleaned, report, data_type,
                method=outlier_method,
                action=outlier_action,
                threshold=self.config.outlier_threshold
            )

            # Stage 5: Missing data handling
            missing_stage = MissingDataStage()
            cleaned = missing_stage.execute(
                cleaned, report, data_type,
                method=self.config.missing_method,
                max_gap_periods=self.config.max_fill_periods
            )

            # Stage 6: Symbol normalization
            symbol_stage = SymbolNormalizationStage()
            cleaned = symbol_stage.execute(cleaned, report, data_type)

            # Stage 7: OHLCV correction (if applicable)
            if self.config.data_type == 'ohlcv':
                ohlcv_stage = OHLCVCorrectionStage()
                cleaned = ohlcv_stage.execute(cleaned, report, data_type)

            # Update report with final row/column counts
            report.final_rows = len(cleaned)
            report.final_columns = len(cleaned.columns)

            # Extract metrics from report
            result.metrics['stages_completed'] = report.stages_completed
            result.metrics['duplicates_removed'] = report.original_rows - len(cleaned)
            result.metrics['cleaning_actions'] = len(report.log_entries)
            result.metrics['outliers_detected'] = sum(
                1 for entry in report.log_entries
                if entry.action == CleaningAction.OUTLIER_DETECTED or
                   (hasattr(entry, 'action') and 'outlier' in str(entry.action).lower())
            )

            result.data = cleaned
            result.records_out = len(cleaned)
            result.success = True

            logger.info(f"comprehensive cleaning completed for {venue}: {result.records_in} -> {result.records_out} records")
            return True

        except ImportError as e:
            logger.debug(f"DataCleaner not available: {e}")
            return False
        except Exception as e:
            logger.warning(f"comprehensive cleaning failed for {venue}: {e}, falling back to basic")
            return False

    def _basic_cleaning(
        self,
        venue: str,
        data: pd.DataFrame,
        result: StageResult
    ) -> pd.DataFrame:
        """Basic cleaning fallback when DataCleaner not available.

        Uses GPU acceleration when available for:
        - Deduplication
        - Outlier detection and treatment
        - Missing value handling
        """
        # Try GPU-accelerated cleaning first
        if self._use_gpu and self._gpu_accelerator:
            try:
                logger.debug(f"Using GPU-accelerated cleaning for {venue}")
                cleaned = self._gpu_accelerator.clean_ohlcv(data)
                result.metrics['gpu_accelerated'] = True
                return cleaned
            except Exception as e:
                logger.warning(f"GPU cleaning failed for {venue}, falling back to CPU: {e}")

        cleaned = data.copy()

        # 1. Remove complete duplicates
        # Handle DataFrames with unhashable columns (lists, dicts) by converting them
        # to strings for deduplication, then restoring original values
        original_len = len(cleaned)

        # Identify unhashable columns (contain lists or dicts)
        unhashable_cols = []
        for col in cleaned.columns:
            try:
                # Try to hash the first non-null value
                sample = cleaned[col].dropna().head(1)
                if len(sample) > 0:
                    val = sample.iloc[0]
                    if isinstance(val, (list, dict)):
                        unhashable_cols.append(col)
            except (TypeError, ValueError):
                unhashable_cols.append(col)

        if unhashable_cols:
            # Convert unhashable columns to strings for deduplication
            logger.debug(f"Converting unhashable columns for deduplication: {unhashable_cols}")
            original_unhashable = {col: cleaned[col].copy() for col in unhashable_cols}
            for col in unhashable_cols:
                cleaned[col] = cleaned[col].apply(lambda x: str(x) if x is not None else None)
            cleaned = cleaned.drop_duplicates()
            # Restore original values using index alignment
            for col in unhashable_cols:
                cleaned[col] = original_unhashable[col].loc[cleaned.index]
        else:
            cleaned = cleaned.drop_duplicates()

        result.metrics['duplicates_removed'] = original_len - len(cleaned)

        # 2. Handle timestamps
        if 'timestamp' in cleaned.columns:
            try:
                cleaned['timestamp'] = pd.to_datetime(cleaned['timestamp'], utc=True, format='ISO8601')
            except ValueError:
                cleaned['timestamp'] = pd.to_datetime(cleaned['timestamp'], utc=True, format='mixed')
            cleaned = cleaned.sort_values('timestamp')

        # 3. Handle missing values
        if self.config.missing_method == 'drop':
            cleaned = cleaned.dropna()
        elif self.config.missing_method == 'forward_fill':
            cleaned = cleaned.ffill(limit=self.config.max_fill_periods)
        elif self.config.missing_method == 'interpolate':
            numeric_cols = cleaned.select_dtypes(include=['float64', 'int64']).columns
            cleaned[numeric_cols] = cleaned[numeric_cols].interpolate(
                method='linear',
                limit=self.config.max_fill_periods
            )

        # 4. Outlier treatment for funding rates
        if self.config.data_type == 'funding_rates' and 'funding_rate' in cleaned.columns:
            rate_col = cleaned['funding_rate']
            mean = rate_col.mean()
            std = rate_col.std()

            if std > 0:
                z_scores = (rate_col - mean).abs() / std
                outliers = z_scores > self.config.outlier_threshold
                result.metrics['outliers_detected'] = int(outliers.sum())

                if self.config.outlier_action == 'remove':
                    cleaned = cleaned[~outliers]
                elif self.config.outlier_action == 'cap':
                    upper = mean + self.config.outlier_threshold * std
                    lower = mean - self.config.outlier_threshold * std
                    cleaned.loc[cleaned['funding_rate'] > upper, 'funding_rate'] = upper
                    cleaned.loc[cleaned['funding_rate'] < lower, 'funding_rate'] = lower
                elif self.config.outlier_action == 'flag':
                    cleaned['is_outlier'] = outliers

        return cleaned
    
    def _stage_normalization(
        self,
        venue: str,
        data: pd.DataFrame
    ) -> Tuple[Optional[pd.DataFrame], StageResult]:
        """
        Execute funding rate normalization stage.

        CRITICAL: This normalizes hourly funding rates (Hyperliquid, dYdX)
        to 8-hour equivalents for cross-venue comparison.

        Uses GPU acceleration when available for large datasets.
        """
        start_time = datetime.now(timezone.utc)
        result = StageResult(
            stage=PipelineStage.NORMALIZATION,
            venue=venue,
            success=False,
            records_in=len(data)
        )

        try:
            # Use GPU for large datasets if available
            if self._use_gpu and self._gpu_accelerator and len(data) > 10000:
                normalized = self._gpu_accelerator.to_gpu(data)
                result.metrics['gpu_accelerated'] = True
            else:
                normalized = data.copy()
            
            if self.normalizer is not None and 'funding_rate' in normalized.columns:
                # Use the proper normalizer
                normalized = self.normalizer.normalize_to_interval(
                    normalized,
                    venue=venue,
                    target_interval=self.config.target_funding_interval
                )
                result.metrics['normalization_applied'] = True
            else:
                # Manual normalization based on venue intervals
                # Hourly venues: Hyperliquid, dYdX, GMX, Vertex
                hourly_venues = {'hyperliquid', 'dydx', 'gmx', 'vertex'}
                
                if venue.lower() in hourly_venues:
                    if 'funding_rate' in normalized.columns:
                        # Convert 1h rate to 8h: multiply by 8
                        normalized['funding_rate_original'] = normalized['funding_rate']
                        normalized['funding_rate'] = normalized['funding_rate'] * 8
                        normalized['funding_interval'] = '8h'
                        normalized['original_interval'] = '1h'
                        result.metrics['conversion_factor'] = 8
                        result.metrics['original_interval'] = '1h'
                else:
                    # Already 8h (Binance, Bybit, OKX, Deribit)
                    normalized['funding_interval'] = '8h'
                    normalized['original_interval'] = '8h'
                    result.metrics['conversion_factor'] = 1
                    result.metrics['original_interval'] = '8h'
            
            # Ensure 'venue' column is always present after normalization
            if 'venue' not in normalized.columns:
                normalized['venue'] = venue

            # Add venue_type metadata if missing
            if 'venue_type' not in normalized.columns:
                config = COLLECTOR_CONFIGS.get(venue.lower())
                if config:
                    normalized['venue_type'] = config.venue_type.value.upper()
                else:
                    # Default based on known venue categories
                    venue_type_map = {
                        'binance': 'CEX', 'bybit': 'CEX', 'okx': 'CEX',
                        'coinbase': 'CEX', 'kraken': 'CEX', 'cme': 'CEX',
                        'hyperliquid': 'HYBRID', 'dydx': 'HYBRID', 'drift': 'HYBRID',
                        'gmx': 'DEX', 'uniswap': 'DEX', 'curve': 'DEX',
                        'deribit': 'CEX', 'aevo': 'OPTIONS',
                    }
                    normalized['venue_type'] = venue_type_map.get(venue.lower(), 'CEX')

            # Add contract_type metadata if missing (PDF requirement)
            if 'contract_type' not in normalized.columns:
                # Determine contract type based on data and venue
                if self.config.data_type == 'funding_rates':
                    normalized['contract_type'] = 'perpetual'
                elif venue.lower() == 'cme':
                    normalized['contract_type'] = 'futures'
                elif 'funding_rate' in normalized.columns:
                    normalized['contract_type'] = 'perpetual'
                else:
                    # Default to perpetual for perp venues, spot otherwise
                    perp_venues = {'binance', 'bybit', 'okx', 'deribit',
                                   'hyperliquid', 'dydx', 'gmx', 'drift'}
                    if venue.lower() in perp_venues:
                        normalized['contract_type'] = 'perpetual'
                    else:
                        normalized['contract_type'] = 'spot'

            # Convert back to CPU DataFrame if using GPU
            if self._use_gpu and self._gpu_accelerator:
                normalized = self._gpu_accelerator.to_cpu(normalized)

            result.data = normalized
            result.records_out = len(normalized)
            result.success = True

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Normalization failed for {venue}: {e}")
            result.data = data # Return original on failure
        
        result.duration_seconds = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()
        
        return result.data, result

    def _enrich_metadata(self, venue: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich data with PDF-required metadata fields.

        Adds venue_type and contract_type if missing, ensuring Phase 3
        compliance with PDF requirements.
        """
        if data is None or data.empty:
            return data

        enriched = data.copy()

        # Ensure venue column
        if 'venue' not in enriched.columns:
            enriched['venue'] = venue

        # Add venue_type metadata if missing
        if 'venue_type' not in enriched.columns:
            config = COLLECTOR_CONFIGS.get(venue.lower())
            if config:
                enriched['venue_type'] = config.venue_type.value.upper()
            else:
                venue_type_map = {
                    'binance': 'CEX', 'bybit': 'CEX', 'okx': 'CEX',
                    'coinbase': 'CEX', 'kraken': 'CEX', 'cme': 'CEX',
                    'hyperliquid': 'HYBRID', 'dydx': 'HYBRID', 'drift': 'HYBRID',
                    'gmx': 'DEX', 'uniswap': 'DEX', 'curve': 'DEX',
                    'deribit': 'CEX', 'aevo': 'OPTIONS',
                }
                enriched['venue_type'] = venue_type_map.get(venue.lower(), 'CEX')

        # Add contract_type metadata if missing (PDF requirement)
        if 'contract_type' not in enriched.columns:
            if self.config.data_type == 'funding_rates':
                enriched['contract_type'] = 'perpetual'
            elif venue.lower() == 'cme':
                enriched['contract_type'] = 'futures'
            elif 'funding_rate' in enriched.columns:
                enriched['contract_type'] = 'perpetual'
            else:
                perp_venues = {'binance', 'bybit', 'okx', 'deribit',
                               'hyperliquid', 'dydx', 'gmx', 'drift'}
                if venue.lower() in perp_venues:
                    enriched['contract_type'] = 'perpetual'
                else:
                    enriched['contract_type'] = 'spot'

        return enriched

    def _stage_validation(
        self,
        venue: str,
        data: pd.DataFrame
    ) -> Tuple[bool, StageResult]:
        """
        Execute validation stage with venue-aware checks.

        This stage performs comprehensive data quality validation including:
        - Schema completeness validation
        - Null value analysis with configurable thresholds
        - Timestamp monotonicity (with proper sorting first)
        - Venue-specific funding rate range validation (1h vs 8h intervals)
        - Duplicate detection
        - Statistical anomaly detection
        - Cross-venue consistency checks
        """
        start_time = datetime.now(timezone.utc)
        result = StageResult(
            stage=PipelineStage.VALIDATION,
            venue=venue,
            success=False,
            records_in=len(data),
            records_out=len(data)
        )

        try:
            quality_score = 100.0
            issues = []
            metrics = {}

            # =================================================================
            # 1. SCHEMA VALIDATION - Check required columns exist
            # =================================================================
            if self.config.data_type == 'funding_rates':
                required = ['timestamp', 'symbol', 'funding_rate']
                optional = ['venue', 'funding_interval', 'mark_price', 'index_price']
            elif self.config.data_type == 'ohlcv':
                required = ['timestamp', 'symbol', 'open', 'high', 'low', 'close']
                optional = ['volume', 'venue', 'timeframe']
            elif self.config.data_type == 'open_interest':
                required = ['timestamp', 'symbol', 'open_interest']
                optional = ['venue', 'notional_value']
            else:
                required = ['timestamp', 'symbol']
                optional = ['venue']

            missing_cols = [c for c in required if c not in data.columns]
            if missing_cols:
                issues.append(f"Missing required columns: {missing_cols}")
                quality_score -= 20 * len(missing_cols) / len(required)

            present_optional = [c for c in optional if c in data.columns]
            metrics['schema_required'] = len(required) - len(missing_cols)
            metrics['schema_optional'] = len(present_optional)

            # =================================================================
            # 2. NULL VALUE ANALYSIS - With configurable severity
            # =================================================================
            null_report = {}
            for col in required:
                if col in data.columns:
                    null_count = data[col].isnull().sum()
                    null_pct = (null_count / len(data)) * 100 if len(data) > 0 else 0
                    null_report[col] = {'count': int(null_count), 'pct': round(null_pct, 2)}

                    # Tiered penalty: <1% = OK, 1-5% = warning, >5% = penalty
                    if null_pct > 5:
                        issues.append(f"{col}: {null_pct:.1f}% null values (exceeds 5% threshold)")
                        quality_score -= min(15, null_pct * 1.5)
                    elif null_pct > 1:
                        issues.append(f"{col}: {null_pct:.1f}% null values (warning)")
                        quality_score -= null_pct * 0.5

            metrics['null_report'] = null_report

            # =================================================================
            # 3. TIMESTAMP VALIDATION - Sort first, then check
            # =================================================================
            if 'timestamp' in data.columns:
                # Ensure timestamps are datetime
                if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                    try:
                        data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
                    except Exception as e:
                        issues.append(f"Failed to parse timestamps: {e}")
                        quality_score -= 10

                # CRITICAL FIX: Sort by timestamp BEFORE checking monotonicity
                sorted_data = data.sort_values('timestamp')

                # Check for duplicate timestamps (per symbol if present)
                if 'symbol' in data.columns:
                    dup_check = data.groupby('symbol')['timestamp'].apply(
                        lambda x: x.duplicated().sum()
                    )
                    total_dups = dup_check.sum()
                else:
                    total_dups = data['timestamp'].duplicated().sum()

                if total_dups > 0:
                    dup_pct = (total_dups / len(data)) * 100
                    issues.append(f"{total_dups:,} duplicate timestamps ({dup_pct:.2f}%)")
                    quality_score -= min(10, dup_pct * 2)

                metrics['duplicate_timestamps'] = int(total_dups)

                # Check timestamp range coverage
                ts_min = data['timestamp'].min()
                ts_max = data['timestamp'].max()
                expected_range = (self.config.end_date - self.config.start_date).total_seconds() / 3600
                actual_range = (ts_max - ts_min).total_seconds() / 3600 if pd.notna(ts_min) and pd.notna(ts_max) else 0

                coverage_pct = (actual_range / expected_range * 100) if expected_range > 0 else 0
                metrics['timestamp_coverage_pct'] = round(coverage_pct, 1)

                if coverage_pct < 80:
                    issues.append(f"Timestamp coverage: {coverage_pct:.1f}% (below 80% threshold)")
                    quality_score -= min(10, (80 - coverage_pct) / 4)

            # =================================================================
            # 4. FUNDING RATE VALIDATION - Venue-aware ranges
            # =================================================================
            if 'funding_rate' in data.columns:
                rates = data['funding_rate'].dropna()

                if len(rates) > 0:
                    # Determine expected range based on venue funding interval
                    # 1h venues: Hyperliquid, dYdX, GMX, Vertex
                    # 8h venues: Binance, Bybit, OKX, Deribit, etc.
                    hourly_venues = {'hyperliquid', 'dydx', 'gmx', 'vertex'}

                    # Check if already normalized (has funding_interval column)
                    is_normalized = 'funding_interval' in data.columns and data['funding_interval'].iloc[0] == '8h'
                    is_hourly_venue = venue.lower() in hourly_venues and not is_normalized

                    if is_hourly_venue:
                        # 1h funding rates: typical range -1.25% to +1.25% (0.01% to 0.1% per hour)
                        min_rate, max_rate = -0.0125, 0.0125
                        interval_desc = "1h"
                    else:
                        # 8h funding rates: typical range -10% to +10% (but usually -1% to +1%)
                        min_rate, max_rate = -0.10, 0.10
                        interval_desc = "8h"

                    # Count out of range values
                    out_of_range = ((rates < min_rate) | (rates > max_rate)).sum()
                    out_of_range_pct = (out_of_range / len(rates)) * 100

                    metrics['funding_rate_stats'] = {
                        'min': float(rates.min()),
                        'max': float(rates.max()),
                        'mean': float(rates.mean()),
                        'std': float(rates.std()),
                        'median': float(rates.median()),
                        'out_of_range_count': int(out_of_range),
                        'out_of_range_pct': round(out_of_range_pct, 2),
                        'expected_interval': interval_desc,
                    }

                    if out_of_range_pct > 1:
                        issues.append(
                            f"{out_of_range_pct:.2f}% funding rates outside expected "
                            f"{interval_desc} range [{min_rate*100:.2f}%, {max_rate*100:.2f}%]"
                        )
                        quality_score -= min(10, out_of_range_pct * 2)

                    # Check for suspicious patterns (all zeros, all same value)
                    unique_rates = rates.nunique()
                    if unique_rates < 3 and len(rates) > 10:
                        issues.append(f"Suspicious funding rate pattern: only {unique_rates} unique values")
                        quality_score -= 15

                    # Check for extreme outliers (> 6 std from mean)
                    if rates.std() > 0:
                        z_scores = np.abs((rates - rates.mean()) / rates.std())
                        extreme_outliers = (z_scores > 6).sum()
                        if extreme_outliers > 0:
                            issues.append(f"{extreme_outliers} extreme outliers (>6 std)")
                            quality_score -= min(5, extreme_outliers)

            # =================================================================
            # 5. OHLCV VALIDATION - Price consistency checks
            # =================================================================
            if self.config.data_type == 'ohlcv':
                ohlcv_issues = []

                # Check OHLCV invariants: high >= low, high >= open/close, low <= open/close
                if all(c in data.columns for c in ['open', 'high', 'low', 'close']):
                    violations = (
                        (data['high'] < data['low']) |
                        (data['high'] < data['open']) |
                        (data['high'] < data['close']) |
                        (data['low'] > data['open']) |
                        (data['low'] > data['close'])
                    ).sum()

                    if violations > 0:
                        violation_pct = (violations / len(data)) * 100
                        ohlcv_issues.append(f"{violations} OHLCV invariant violations ({violation_pct:.2f}%)")
                        quality_score -= min(10, violation_pct * 2)

                    metrics['ohlcv_violations'] = int(violations)

                    # Check for zero/negative prices
                    zero_prices = (
                        (data['open'] <= 0) | (data['high'] <= 0) |
                        (data['low'] <= 0) | (data['close'] <= 0)
                    ).sum()

                    if zero_prices > 0:
                        ohlcv_issues.append(f"{zero_prices} records with zero/negative prices")
                        quality_score -= min(15, (zero_prices / len(data)) * 100)

                issues.extend(ohlcv_issues)

            # =================================================================
            # 6. RECORD COUNT VALIDATION
            # =================================================================
            if len(data) == 0:
                issues.append("No records in dataset")
                quality_score = 0
            elif len(data) < 10:
                issues.append(f"Very few records: {len(data)}")
                quality_score -= 30
            elif len(data) < 100:
                issues.append(f"Low record count: {len(data)}")
                quality_score -= 10

            metrics['record_count'] = len(data)

            # =================================================================
            # 7. SYMBOL COVERAGE VALIDATION
            # =================================================================
            if 'symbol' in data.columns:
                symbols_in_data = data['symbol'].nunique()
                expected_symbols = len(self.config.symbols) if self.config.symbols else 1
                coverage = (symbols_in_data / expected_symbols) * 100 if expected_symbols > 0 else 0

                metrics['symbol_count'] = symbols_in_data
                metrics['symbol_coverage_pct'] = round(coverage, 1)

                if coverage < 50:
                    issues.append(f"Low symbol coverage: {symbols_in_data}/{expected_symbols} ({coverage:.1f}%)")
                    quality_score -= min(15, (50 - coverage) / 5)

            # =================================================================
            # 8. USE QUALITY CHECKER FOR ADDITIONAL VALIDATION
            # =================================================================
            manual_score = quality_score # Save manual score

            if self.quality_checker is not None:
                try:
                    checker_result = self.quality_checker.check(data, self.config.data_type)
                    # CRITICAL FIX: Combine scores instead of overwriting
                    # Weight: 60% manual validation, 40% quality checker
                    quality_score = (manual_score * 0.6) + (checker_result.overall_score * 0.4)
                    issues.extend(checker_result.issues)
                    metrics['quality_checker_score'] = checker_result.overall_score
                except Exception as e:
                    logger.debug(f"Quality checker not used for {venue}: {e}")
                    metrics['quality_checker_score'] = None

            metrics['manual_score'] = round(manual_score, 1)

            # =================================================================
            # FINALIZE RESULT
            # =================================================================
            result.metrics['quality_score'] = max(0, min(100, quality_score))
            result.metrics['issues'] = issues
            result.metrics.update(metrics)
            # Filter warnings - handle both string and QualityIssue objects
            result.warnings = [
                i for i in issues
                if (isinstance(i, str) and 'warning' in i.lower()) or quality_score >= 70
            ]
            result.success = quality_score >= self.config.min_quality_score

            logger.info(
                f"Validation {venue}: score={quality_score:.1f}, "
                f"issues={len(issues)}, records={len(data)}"
            )

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Validation failed for {venue}: {e}", exc_info=True)

        result.duration_seconds = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()

        return result.success, result
    
    def _stage_storage(self, venue: str, data: pd.DataFrame) -> StageResult:
        """
        Execute storage stage.

        Saves processed data to:
        1. Parquet file (primary storage)
        2. Incremental cache (for smart future collection)

        The cache update is CRITICAL for the incremental caching feature:
        - Future runs will check the cache to see what data already exists
        - Only missing periods will be collected (huge time savings!)
        - Cache is only updated AFTER full pipeline processing (clean/validated)
        """
        start_time = datetime.now(timezone.utc)
        result = StageResult(
            stage=PipelineStage.STORAGE,
            venue=venue,
            success=False,
            records_in=len(data),
            records_out=len(data)
        )

        try:
            # Determine output path
            filename = f"{venue}_{self.config.data_type}.parquet"
            output_path = self.output_dir / venue / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to parquet (primary storage)
            data.to_parquet(output_path, index=False, compression='snappy')

            result.metrics['output_path'] = str(output_path)
            result.metrics['file_size_bytes'] = output_path.stat().st_size
            result.success = True

            logger.info(f"Saved {len(data)} records to {output_path}")

            # =================================================================
            # INCREMENTAL CACHE UPDATE
            # =================================================================
            # Update the incremental cache so future runs can skip this data
            # This is CRITICAL for the "only collect missing data" feature
            if INCREMENTAL_CACHE_AVAILABLE:
                try:
                    cache_manager = get_cache_manager(str(self.output_dir))

                    # Determine timeframe for cache key (only for OHLCV)
                    timeframe = None
                    if self.config.data_type == 'ohlcv':
                        # Try to detect timeframe from data
                        if 'timestamp' in data.columns and len(data) >= 2:
                            timestamps = pd.to_datetime(data['timestamp']).sort_values()
                            if len(timestamps) >= 2:
                                diff = (timestamps.iloc[1] - timestamps.iloc[0]).total_seconds()
                                if diff <= 3600:
                                    timeframe = '1h'
                                elif diff <= 14400:
                                    timeframe = '4h'
                                elif diff <= 86400:
                                    timeframe = '1d'

                    # Determine date range from config or data
                    if self.config.start_date and self.config.end_date:
                        cache_start = self.config.start_date.strftime('%Y-%m-%d') if hasattr(self.config.start_date, 'strftime') else str(self.config.start_date)
                        cache_end = self.config.end_date.strftime('%Y-%m-%d') if hasattr(self.config.end_date, 'strftime') else str(self.config.end_date)
                    elif 'timestamp' in data.columns and len(data) > 0:
                        ts = pd.to_datetime(data['timestamp'])
                        cache_start = ts.min().strftime('%Y-%m-%d')
                        cache_end = ts.max().strftime('%Y-%m-%d')
                    else:
                        cache_start = datetime.now(timezone.utc).strftime('%Y-%m-%d')
                        cache_end = cache_start

                    # Update cache with processed data
                    cache_manager.update_cache(
                        data_type=self.config.data_type,
                        venue=venue,
                        data=data,
                        start_date=cache_start,
                        end_date=cache_end,
                        timeframe=timeframe
                    )

                    result.metrics['cache_updated'] = True
                    logger.debug(f"Updated incremental cache for {venue}/{self.config.data_type}")

                except Exception as cache_error:
                    # Cache update failure shouldn't fail the whole storage operation
                    logger.warning(f"Cache update failed for {venue}: {cache_error}")
                    result.metrics['cache_updated'] = False
                    result.warnings.append(f"Cache update failed: {cache_error}")
            else:
                result.metrics['cache_updated'] = False
                result.metrics['cache_note'] = 'Incremental cache not available'

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Storage failed for {venue}: {e}")

        result.duration_seconds = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()

        return result
    
    def _align_cross_venue(
        self,
        venue_results: Dict[str, VenueResult]
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Align data across venues for cross-venue analysis.

        Uses comprehensive CrossVenueAligner when available for:
        - Multi-frequency timestamp alignment (1h, 4h, 8h intervals)
        - Settlement time snapping per venue
        - Partial overlap support (configurable minimum coverage)
        - CEX-DEX latency compensation

        Returns
        -------
        Tuple[DataFrame, Dict]
            Aligned data and cross-venue metrics
        """
        metrics = {}

        # Collect all DataFrames with valid data
        dfs = {}
        for venue, result in venue_results.items():
            if result.data is not None and not result.data.empty:
                if 'timestamp' in result.data.columns:
                    dfs[venue] = result.data

        if len(dfs) < 1:
            metrics['venues_aligned'] = []
            metrics['total_aligned_records'] = 0
            metrics['alignment_status'] = 'no_data'
            return None, metrics

        if len(dfs) == 1:
            # Single venue - return data with basic metrics (no cross-venue comparison possible)
            venue = list(dfs.keys())[0]
            df = dfs[venue].copy()
            if 'venue' not in df.columns:
                df['venue'] = venue
            metrics['venues_aligned'] = [venue]
            metrics['total_aligned_records'] = len(df)
            metrics['alignment_status'] = 'single_venue'
            metrics['cross_venue_comparison'] = 'not_applicable_single_venue'
            logger.info(f"Single venue data: {venue} with {len(df)} records (cross-venue analysis skipped)")
            return df, metrics

        try:
            # Use comprehensive aligner if available
            if CROSS_VENUE_AVAILABLE:
                return self._enhanced_cross_venue_alignment(dfs, metrics)
            else:
                return self._basic_cross_venue_alignment(dfs, metrics)

        except Exception as e:
            logger.error(f"Cross-venue alignment failed: {e}")
            metrics['alignment_error'] = str(e)
            return None, metrics

    def _enhanced_cross_venue_alignment(
        self,
        dfs: Dict[str, pd.DataFrame],
        metrics: Dict[str, Any]
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        comprehensive cross-venue alignment using CrossVenueAligner.

        Features:
        - Multi-frequency support (1h, 8h funding intervals)
        - Settlement time snapping per venue
        - CEX-DEX reconciliation with latency compensation
        - Correlation matrix with venue-type aware thresholds
        """
        # Determine target interval based on data type
        target_interval = '8h' if self.config.data_type == 'funding_rates' else '1h'

        # Create aligner with partial overlap support
        aligner = CrossVenueAligner(
            target_interval=target_interval,
            min_overlap_pct=0.3, # Accept 30% overlap minimum
            alignment_strategy='snap_to_settlement'
        )

        # Determine value column based on data type
        value_col = 'funding_rate' if self.config.data_type == 'funding_rates' else 'close'

        # Align all venues
        alignment_result = aligner.align(
            venue_dfs=dfs,
            timestamp_col='timestamp',
            value_cols=[value_col] if value_col in dfs[list(dfs.keys())[0]].columns else None
        )

        # Extract alignment quality metrics
        metrics['alignment_quality'] = alignment_result.alignment_quality
        metrics['common_timestamps'] = alignment_result.common_timestamps
        metrics['venue_coverage'] = alignment_result.venue_coverage
        metrics['gap_count'] = len(alignment_result.gaps)

        if alignment_result.metadata.get('time_range'):
            common_start, common_end = alignment_result.metadata['time_range']
            metrics['common_start'] = common_start.isoformat() if hasattr(common_start, 'isoformat') else str(common_start)
            metrics['common_end'] = common_end.isoformat() if hasattr(common_end, 'isoformat') else str(common_end)
            if hasattr(common_end, 'timestamp') and hasattr(common_start, 'timestamp'):
                metrics['common_duration_hours'] = (common_end - common_start).total_seconds() / 3600

        # Run multi-venue reconciliation if we have aligned data
        if not alignment_result.aligned_data.empty and self.config.data_type == 'funding_rates':
            reconciler = MultiVenueReconciler()

            # Prepare venue data for reconciliation
            venue_data = {}
            for venue, df in dfs.items():
                if value_col in df.columns:
                    venue_data[venue] = df

            if len(venue_data) >= 2:
                reconciliation = reconciler.reconcile_all(
                    venue_data=venue_data,
                    symbol='ALL', # Aggregate across symbols
                    timestamp_col='timestamp',
                    value_col=value_col
                )

                # Extract reconciliation metrics
                metrics['reconciliation_quality'] = reconciliation.get('overall_quality', 0)
                metrics['venue_reliability'] = reconciliation.get('venue_reliability', {})
                metrics['pairwise_results_count'] = len(reconciliation.get('pairwise_results', []))

                # Extract correlation matrix
                if reconciliation.get('correlation_matrix') is not None:
                    corr_matrix = reconciliation['correlation_matrix']
                    metrics['cross_venue_correlation'] = corr_matrix.to_dict()

                    # Calculate average off-diagonal correlation
                    corr_values = corr_matrix.values[~np.eye(len(corr_matrix), dtype=bool)]
                    if len(corr_values) > 0:
                        metrics['average_correlation'] = float(np.nanmean(corr_values))

                # Add warnings/errors from reconciliation
                if reconciliation.get('warnings'):
                    metrics['reconciliation_warnings'] = reconciliation['warnings']
                if reconciliation.get('errors'):
                    metrics['reconciliation_errors'] = reconciliation['errors']

        # Build combined aligned DataFrame
        aligned_dfs = []
        for venue, df in dfs.items():
            # Filter to alignment range
            if 'time_range' in alignment_result.metadata:
                common_start, common_end = alignment_result.metadata['time_range']
                filtered = df[
                    (df['timestamp'] >= common_start) &
                    (df['timestamp'] <= common_end)
                ].copy()
            else:
                filtered = df.copy()

            if not filtered.empty:
                if 'venue' not in filtered.columns:
                    filtered['venue'] = venue
                aligned_dfs.append(filtered)

        if not aligned_dfs:
            metrics['total_aligned_records'] = 0
            metrics['venues_aligned'] = []
            return None, metrics

        aligned = pd.concat(aligned_dfs, ignore_index=True)
        metrics['total_aligned_records'] = len(aligned)
        metrics['venues_aligned'] = list(dfs.keys())

        return aligned, metrics

    def _basic_cross_venue_alignment(
        self,
        dfs: Dict[str, pd.DataFrame],
        metrics: Dict[str, Any]
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Basic cross-venue alignment (fallback when comprehensive aligner unavailable).
        """
        # Find timestamp ranges for each venue
        venue_ranges = {}
        for venue, df in dfs.items():
            venue_ranges[venue] = {
                'min': df['timestamp'].min(),
                'max': df['timestamp'].max(),
                'count': len(df)
            }

        # Find common timestamp range across all venues
        min_times = [r['min'] for r in venue_ranges.values()]
        max_times = [r['max'] for r in venue_ranges.values()]

        common_start = max(min_times)
        common_end = min(max_times)

        # Check if there's a valid overlapping range
        if common_start > common_end:
            logger.warning(
                f"No overlapping timestamp range between venues. "
                f"Start ({common_start}) > End ({common_end})"
            )
            metrics['common_start'] = min(min_times).isoformat()
            metrics['common_end'] = max(max_times).isoformat()
            metrics['common_duration_hours'] = 0.0
            metrics['total_aligned_records'] = 0
            metrics['venues_aligned'] = list(dfs.keys())
            metrics['alignment_warning'] = 'No overlapping timestamp range between venues'
            return None, metrics

        metrics['common_start'] = common_start.isoformat()
        metrics['common_end'] = common_end.isoformat()
        metrics['common_duration_hours'] = (common_end - common_start).total_seconds() / 3600

        # Filter to common range
        aligned_dfs = []
        venues_with_data = []
        for venue, df in dfs.items():
            filtered = df[
                (df['timestamp'] >= common_start) &
                (df['timestamp'] <= common_end)
            ].copy()
            if not filtered.empty:
                if 'venue' not in filtered.columns:
                    filtered['venue'] = venue
                aligned_dfs.append(filtered)
                venues_with_data.append(venue)

        if not aligned_dfs:
            metrics['total_aligned_records'] = 0
            metrics['venues_aligned'] = []
            return None, metrics

        aligned = pd.concat(aligned_dfs, ignore_index=True)

        # Calculate cross-venue correlation for funding rates
        if self.config.data_type == 'funding_rates' and 'funding_rate' in aligned.columns:
            pivot = aligned.pivot_table(
                values='funding_rate',
                index=['timestamp', 'symbol'],
                columns='venue',
                aggfunc='mean'
            )

            if len(pivot.columns) >= 2:
                corr = pivot.corr()
                metrics['cross_venue_correlation'] = corr.to_dict()

                corr_values = corr.values[~np.eye(len(corr), dtype=bool)]
                if len(corr_values) > 0:
                    metrics['average_correlation'] = float(np.nanmean(corr_values))

        metrics['total_aligned_records'] = len(aligned)
        metrics['venues_aligned'] = venues_with_data

        return aligned, metrics

    async def cleanup(self) -> None:
        """
        Clean up async resources to ensure proper shutdown.

        CRITICAL: This prevents the event loop from hanging by closing all
        open sessions, cancelling background tasks, and shutting down executors.
        """
        logger.info("Cleaning up pipeline resources...")

        try:
            # Cleanup collection manager
            if hasattr(self.collection_manager, 'cleanup'):
                await self.collection_manager.cleanup()

            # Cleanup all collectors in the registry
            if hasattr(self.collection_manager, 'registry'):
                registry = self.collection_manager.registry
                if hasattr(registry, 'collectors'):
                    for collector in registry.collectors.values():
                        if hasattr(collector, 'cleanup') and asyncio.iscoroutinefunction(collector.cleanup):
                            try:
                                await collector.cleanup()
                            except Exception as e:
                                logger.debug(f"Collector cleanup error: {e}")

            # Cancel any pending symbol processors
            if hasattr(self.collection_manager, '_symbol_processors'):
                for processor in self.collection_manager._symbol_processors.values():
                    if hasattr(processor, 'cleanup'):
                        try:
                            await processor.cleanup()
                        except Exception as e:
                            logger.debug(f"Symbol processor cleanup error: {e}")

            logger.info("Pipeline cleanup complete")

        except Exception as e:
            logger.warning(f"Error during pipeline cleanup: {e}")

    def _generate_report(self, result: PipelineResult) -> None:
        """
        Generate comprehensive pipeline execution report with detailed analysis.

        This report includes:
        - Executive summary with key metrics
        - Per-venue detailed breakdown
        - Data quality analysis
        - Cross-venue comparison
        - Issues and recommendations
        - Compliance checklist for assignment requirements
        """
        report_path = self.output_dir / 'pipeline_report.md'
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

        try:
            lines = []

            # =================================================================
            # HEADER
            # =================================================================
            lines.extend([
                "# Data Pipeline Execution Report",
                "",
                f"**Generated:** {timestamp}",
                f"**Data Type:** {self.config.data_type}",
                f"**Date Range:** {self.config.start_date.strftime('%Y-%m-%d')} to {self.config.end_date.strftime('%Y-%m-%d')}",
                "",
                "---",
                "",
            ])

            # =================================================================
            # EXECUTIVE SUMMARY
            # =================================================================
            lines.extend([
                "## Executive Summary",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Status | **{result.status.value.upper()}** |",
                f"| Total Records | {result.total_records:,} |",
                f"| Venues Processed | {len(result.venue_results)} |",
                f"| Success Rate | {result.success_rate:.1f}% |",
                f"| Overall Quality Score | {result.overall_quality_score:.1f}/100 |",
                f"| Total Duration | {result.duration_seconds:.1f}s |",
                "",
            ])

            # Categorize venues by status
            successful_venues = [v for v, r in result.venue_results.items() if r.success and r.total_records > 0]
            partial_venues = [v for v, r in result.venue_results.items() if r.success and r.total_records == 0]
            failed_venues = [v for v, r in result.venue_results.items() if not r.success]

            lines.extend([
                f"**Successful Venues ({len(successful_venues)}):** {', '.join(successful_venues) if successful_venues else 'None'}",
                "",
                f"**Venues with No Data ({len(partial_venues)}):** {', '.join(partial_venues) if partial_venues else 'None'}",
                "",
                f"**Failed Venues ({len(failed_venues)}):** {', '.join(failed_venues) if failed_venues else 'None'}",
                "",
                "---",
                "",
            ])

            # =================================================================
            # PER-VENUE DETAILED ANALYSIS
            # =================================================================
            lines.extend([
                "## Per-Venue Analysis",
                "",
            ])

            for venue, vr in sorted(result.venue_results.items(), key=lambda x: -x[1].total_records):
                status_icon = "[OK]" if vr.success and vr.total_records > 0 else "[WARN]" if vr.success else "[FAIL]"
                venue_type = vr.venue_type.value if hasattr(vr.venue_type, 'value') else str(vr.venue_type)

                lines.extend([
                    f"### {status_icon} {venue.upper()} ({venue_type})",
                    "",
                    f"| Metric | Value |",
                    f"|--------|-------|",
                    f"| Records Collected | {vr.total_records:,} |",
                    f"| Quality Score | {vr.quality_score:.1f}/100 |",
                    f"| Quality Grade | {vr.quality_grade.name} |",
                    f"| Processing Time | {vr.total_duration_seconds:.1f}s |",
                    "",
                ])

                # Stage-by-stage breakdown
                if vr.stage_results:
                    lines.append("**Pipeline Stages:**")
                    lines.append("")
                    lines.append("| Stage | Status | Records In → Out | Duration |")
                    lines.append("|-------|--------|------------------|----------|")

                    for stage, sr in sorted(vr.stage_results.items(), key=lambda x: x[0].order):
                        stage_status = "PASS" if sr.success else "FAIL"
                        lines.append(
                            f"| {stage.name} | {stage_status} | {sr.records_in:,} -> {sr.records_out:,} | {sr.duration_seconds:.2f}s |"
                        )
                    lines.append("")

                # Validation metrics if available
                val_result = vr.stage_results.get(PipelineStage.VALIDATION)
                if val_result and val_result.metrics:
                    metrics = val_result.metrics

                    # Funding rate stats
                    if 'funding_rate_stats' in metrics:
                        fr_stats = metrics['funding_rate_stats']
                        lines.extend([
                            "**Funding Rate Statistics:**",
                            "",
                            f"- Min: {fr_stats.get('min', 'N/A'):.6f}",
                            f"- Max: {fr_stats.get('max', 'N/A'):.6f}",
                            f"- Mean: {fr_stats.get('mean', 'N/A'):.6f}",
                            f"- Std: {fr_stats.get('std', 'N/A'):.6f}",
                            f"- Out of Range: {fr_stats.get('out_of_range_pct', 0):.2f}%",
                            "",
                        ])

                    # Issues
                    issues = metrics.get('issues', [])
                    if issues:
                        lines.extend([
                            "**Issues Detected:**",
                            "",
                        ])
                        for issue in issues[:5]: # Limit to 5 issues
                            lines.append(f"- WARNING: {issue}")
                        if len(issues) > 5:
                            lines.append(f"- ... and {len(issues) - 5} more issues")
                        lines.append("")

                # Stage errors
                errors = []
                for stage, sr in vr.stage_results.items():
                    errors.extend([f"[{stage.name}] {e}" for e in sr.errors])

                if errors:
                    lines.extend([
                        "**Errors:**",
                        "",
                    ])
                    for error in errors[:3]:
                        lines.append(f"- ERROR: {error}")
                    lines.append("")

                lines.append("---")
                lines.append("")

            # =================================================================
            # CROSS-VENUE ANALYSIS
            # =================================================================
            if result.cross_venue_metrics:
                lines.extend([
                    "## Cross-Venue Analysis",
                    "",
                ])

                cv = result.cross_venue_metrics

                # Alignment info
                lines.extend([
                    "### Timestamp Alignment",
                    "",
                    f"| Metric | Value |",
                    f"|--------|-------|",
                    f"| Venues Aligned | {len(cv.get('venues_aligned', []))} |",
                    f"| Total Aligned Records | {cv.get('total_aligned_records', 0):,} |",
                ])

                if 'common_start' in cv:
                    lines.append(f"| Common Start | {cv['common_start']} |")
                if 'common_end' in cv:
                    lines.append(f"| Common End | {cv['common_end']} |")
                if 'common_duration_hours' in cv:
                    lines.append(f"| Common Duration | {cv['common_duration_hours']:.1f} hours |")
                if 'alignment_quality' in cv:
                    lines.append(f"| Alignment Quality | {cv['alignment_quality']:.1f}% |")

                lines.append("")

                # Correlation matrix
                if 'cross_venue_correlation' in cv:
                    lines.extend([
                        "### Cross-Venue Correlation Matrix",
                        "",
                    ])

                    corr = cv['cross_venue_correlation']
                    if isinstance(corr, dict) and corr:
                        venues_list = list(corr.keys())
                        # Build table header
                        header = "| Venue | " + " | ".join(venues_list) + " |"
                        sep = "|-------|" + "|".join(["------" for _ in venues_list]) + "|"
                        lines.append(header)
                        lines.append(sep)

                        for v1 in venues_list:
                            row_values = []
                            for v2 in venues_list:
                                val = corr.get(v1, {}).get(v2, 0)
                                if isinstance(val, (int, float)):
                                    row_values.append(f"{val:.3f}")
                                else:
                                    row_values.append("N/A")
                            lines.append(f"| {v1} | " + " | ".join(row_values) + " |")
                        lines.append("")

                if 'average_correlation' in cv:
                    avg_corr = cv['average_correlation']
                    corr_quality = "Excellent" if avg_corr > 0.9 else "Good" if avg_corr > 0.7 else "Moderate" if avg_corr > 0.5 else "Low"
                    lines.extend([
                        f"**Average Cross-Venue Correlation:** {avg_corr:.4f} ({corr_quality})",
                        "",
                    ])

                # Warnings
                if cv.get('reconciliation_warnings'):
                    lines.extend([
                        "### Reconciliation Warnings",
                        "",
                    ])
                    for warning in cv['reconciliation_warnings']:
                        lines.append(f"- WARNING: {warning}")
                    lines.append("")

                lines.append("---")
                lines.append("")

            # =================================================================
            # DATA QUALITY SUMMARY
            # =================================================================
            lines.extend([
                "## Data Quality Summary",
                "",
                "| Venue | Records | Quality | Grade | Status |",
                "|-------|---------|---------|-------|--------|",
            ])

            for venue, vr in sorted(result.venue_results.items(), key=lambda x: -x[1].quality_score):
                status = "[OK]" if vr.success and vr.total_records > 0 else "[WARN]" if vr.success else "[FAIL]"
                lines.append(
                    f"| {venue} | {vr.total_records:,} | {vr.quality_score:.1f} | {vr.quality_grade.name} | {status} |"
                )
            lines.append("")

            # Quality score interpretation
            overall = result.overall_quality_score
            if overall >= 90:
                grade_text = "**EXCELLENT** - Data meets all quality standards"
            elif overall >= 80:
                grade_text = "**GOOD** - Data meets most quality standards"
            elif overall >= 70:
                grade_text = "**ACCEPTABLE** - Data meets minimum requirements"
            elif overall >= 50:
                grade_text = "**NEEDS IMPROVEMENT** - Some quality issues detected"
            else:
                grade_text = "**POOR** - Significant quality issues"

            lines.extend([
                f"**Overall Data Quality:** {grade_text}",
                "",
                "---",
                "",
            ])

            # =================================================================
            # RECOMMENDATIONS
            # =================================================================
            recommendations = []

            if failed_venues:
                recommendations.append(
                    f"**Fix Failed Venues:** {', '.join(failed_venues)} failed to collect data. "
                    "Check API credentials and rate limits."
                )

            if result.overall_quality_score < 80:
                recommendations.append(
                    "**Improve Data Quality:** Consider adding more validation rules and "
                    "implementing better outlier detection."
                )

            if len(successful_venues) < 3:
                recommendations.append(
                    "**Expand Data Sources:** Consider adding more venues for better "
                    "cross-venue validation and arbitrage opportunities."
                )

            if result.total_records < 1000:
                recommendations.append(
                    "**Increase Data Coverage:** Current dataset may be too small for "
                    "meaningful statistical analysis."
                )

            if recommendations:
                lines.extend([
                    "## Recommendations",
                    "",
                ])
                for i, rec in enumerate(recommendations, 1):
                    lines.append(f"{i}. {rec}")
                lines.append("")
                lines.append("---")
                lines.append("")

            # =================================================================
            # CONFIGURATION USED
            # =================================================================
            lines.extend([
                "## Configuration Used",
                "",
                "```yaml",
                f"data_type: {self.config.data_type}",
                f"venues: {self.config.venues}",
                f"symbols: {self.config.symbols[:10]}{'...' if len(self.config.symbols) > 10 else ''}",
                f"start_date: {self.config.start_date.isoformat()}",
                f"end_date: {self.config.end_date.isoformat()}",
                f"normalize_funding_rates: {self.config.normalize_funding_rates}",
                f"min_quality_score: {self.config.min_quality_score}",
                f"outlier_action: {self.config.outlier_action}",
                f"outlier_threshold: {self.config.outlier_threshold}",
                "```",
                "",
            ])

            # Write report
            with open(report_path, 'w') as f:
                f.write('\n'.join(lines))

            logger.info(f"Comprehensive report saved to {report_path}")

        except Exception as e:
            logger.error(f"Failed to generate report: {e}", exc_info=True)
    
    def process_dataframes(
        self,
        venue_dataframes: Dict[str, pd.DataFrame]
    ) -> PipelineResult:
        """
        Process existing DataFrames without collection.
        
        Parameters
        ----------
        venue_dataframes : Dict[str, pd.DataFrame]
            Pre-collected data by venue
            
        Returns
        -------
        PipelineResult
            Pipeline result
        """
        result = PipelineResult(
            config=self.config,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now(timezone.utc)
        )
        
        # Create mock collection results
        for venue, df in venue_dataframes.items():
            col_result = CollectionResult(
                venue=venue,
                data_type=self.config.data_type,
                status=CollectionStatus.COMPLETED,
                data=df,
                total_records=len(df) if df is not None and not df.empty else 0
            )
            
            # Process through pipeline
            loop = asyncio.get_event_loop()
            venue_result = loop.run_until_complete(
                self._process_venue(venue, col_result)
            )
            result.venue_results[venue] = venue_result
            result.total_records += venue_result.total_records
        
        # Cross-venue alignment
        if self.config.align_timestamps and len(result.venue_results) > 1:
            result.aligned_data, result.cross_venue_metrics = \
                self._align_cross_venue(result.venue_results)
        
        # Calculate overall quality (only include venues with actual data)
        if result.venue_results:
            scores = [
                r.quality_score for r in result.venue_results.values()
                if r.total_records > 0
            ]
            if scores:
                result.overall_quality_score = sum(scores) / len(scores)
            else:
                result.overall_quality_score = 0.0

        result.status = PipelineStatus.COMPLETED
        result.end_time = datetime.now(timezone.utc)

        return result

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def run_pipeline(
    venue_dataframes: Optional[Dict[str, pd.DataFrame]] = None,
    data_type: str = 'funding_rates',
    venues: Optional[List[str]] = None,
    min_quality_score: float = 70.0,
    output_dir: str = 'data/processed',
    **kwargs
) -> PipelineResult:
    """
    Convenience function to run pipeline.
    
    Parameters
    ----------
    venue_dataframes : Dict[str, pd.DataFrame], optional
        Pre-collected data (skip collection if provided)
    data_type : str
        Data type to process
    venues : List[str], optional
        Venues to collect from
    min_quality_score : float
        Minimum quality score threshold
    output_dir : str
        Output directory
    **kwargs
        Additional PipelineConfig parameters
        
    Returns
    -------
    PipelineResult
        Pipeline execution result
    """
    config = PipelineConfig(
        data_type=data_type,
        venues=venues or [],
        min_quality_score=min_quality_score,
        output_dir=output_dir,
        **kwargs
    )
    
    pipeline = DataPipeline(config)
    
    if venue_dataframes:
        return pipeline.process_dataframes(venue_dataframes)
    else:
        return await pipeline.run()

__all__ = [
    'PipelineStage',
    'PipelineStatus',
    'PipelineConfig',
    'StageResult',
    'VenueResult',
    'PipelineResult',
    'DataPipeline',
    'run_pipeline',
]
