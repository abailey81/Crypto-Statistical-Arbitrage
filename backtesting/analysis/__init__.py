"""
Backtesting Analysis Modules - PDF Section 2.4 Complete Implementation
=======================================================================

Comprehensive analysis modules for backtesting results including all
PDF Section 2.4 requirements from project specification:

Core Analysis:
- Crisis event analysis (11 events per PDF)
- Performance attribution (venue, sector, regime)
- Capacity analysis (CEX $10-30M, DEX $1-5M, Combined $20-50M)
- Grain futures comparison (PDF REQUIRED academic benchmark)

Walk-Forward Optimization:
- 18-month training / 6-month test rolling windows
- Parameter stability tracking
- Regime-adaptive optimization

Venue-Specific Execution:
- CEX execution (0.05% per side, ±2.0 z-score)
- DEX execution (0.50-1.50% + gas + MEV, ±2.5 z-score)
- Mixed/Combined routing optimization

Position Sizing:
- CEX: up to $100,000 per position
- DEX Liquid: $20,000 - $50,000
- DEX Illiquid: $5,000 - $10,000
- Kelly Criterion with volatility targeting

Concentration Limits:
- 40% max sector concentration
- 60% max CEX-only concentration
- 20% max Tier 3 asset allocation

Comprehensive Reporting:
- 5-6 page PDF-compliant reports
- All required metrics and breakdowns

Author: Tamer Atesyakar
Version: 2.0.0
"""

# Core Analysis Modules
from .crisis_analyzer import (
    CrisisAnalyzer,
    CrisisEvent,
    CrisisAnalysisResult,
    CrisisType
)
from .performance_attribution import (
    PerformanceAttributor,
    AttributionResult,
    FactorContribution,
    AttributionFactor
)
from .capacity_analyzer import (
    CapacityAnalyzer,
    VenueCapacity,
    CapacityConstraint
)
from .grain_futures_comparison import (
    GrainFuturesComparison,
    GrainFuturesBenchmark,
    CryptoPairCharacteristics,
    ComparisonResult,
    AssetClass,
    MarketStructure,
    compare_to_grain_futures
)

# Walk-Forward Optimization (PDF: 18m train / 6m test)
from .walk_forward_optimizer import (
    WalkForwardOptimizer,
    WalkForwardWindow,
    WalkForwardConfig,
    WindowResult,
    WalkForwardResult,
    create_walk_forward_optimizer
)

# Venue-Specific Backtesting (PDF: CEX/DEX/Mixed/Combined)
from .venue_specific_backtester import (
    VenueSpecificBacktester,
    VenueCostModel,
    VenueExecutionConfig,
    VenueTradeResult,
    VenueBacktestResult,
    VenueType as BacktestVenueType,
    create_venue_backtester
)

# Full Metrics (All PDF Section 2.4 metrics)
from .advanced_metrics import (
    AdvancedMetricsCalculator,
    PDFCompliantMetrics,
    VenueSpecificMetrics,
    RegimeSpecificMetrics,
    TradeStatistics as TradingMetrics,  # Alias for backwards compatibility
    CostMetrics,
    CapacityMetrics,
    RiskMetrics,
    StatisticalMetrics,
    ExecutionMetrics,
    TimeBasedMetrics,
    RollingMetrics,
    CoreMetrics,
    RiskAdjustedMetrics,
    DrawdownMetrics,
    create_metrics_calculator
)

# Position Sizing (PDF: $100k CEX, $20-50k DEX)
from .position_sizing_engine import (
    PositionSizingEngine,
    PositionSizeConfig,
    PositionSizeResult,
    PortfolioSizeResult,
    VenueType as SizingVenueType,
    LiquidityTier,
    create_position_sizing_engine
)

# Concentration Limits (PDF: 40% sector, 60% CEX, 20% Tier3)
from .concentration_limits import (
    ConcentrationLimitsEnforcer,
    ConcentrationLimitsConfig,
    ConcentrationLimit,
    LimitBreachEvent,
    RebalanceRecommendation,
    LimitType,
    BreachSeverity,
    create_concentration_enforcer
)

# Comprehensive Reporting (5-6 pages per PDF)
from .comprehensive_report import (
    ComprehensiveReportGenerator,
    ReportMetrics,
    ReportSection,
    WalkForwardSummary,
    VenueBreakdown,
    CrisisEventSummary,
    GrainComparisonSummary,
    create_comprehensive_report
)

# Step 4 Complete Orchestrator
from .step4_orchestrator import (
    Step4AdvancedOrchestrator,
    OrchestratorConfig,
    OrchestratorState,
    ComponentStatus,
    ExecutionMode,
    RiskLevel,
    IntegrationMode,
    ComponentResult,
    CrossValidationResult,
    MonteCarloValidation,
    MessageBus,
    CheckpointManager,
    RealTimeMonitor,
    CrossValidationEngine,
    MonteCarloEngine,
    ResultSynthesisEngine,
    DependencyGraph,
    create_step4_orchestrator,
    run_step4_advanced,
)

__all__ = [
    # ============================================================
    # CRISIS ANALYSIS (PDF: 11 events required)
    # ============================================================
    'CrisisAnalyzer',
    'CrisisEvent',
    'CrisisAnalysisResult',
    'CrisisType',

    # ============================================================
    # PERFORMANCE ATTRIBUTION
    # ============================================================
    'PerformanceAttributor',
    'AttributionResult',
    'FactorContribution',
    'AttributionFactor',

    # ============================================================
    # CAPACITY ANALYSIS (PDF: CEX $10-30M, DEX $1-5M, Combined $20-50M)
    # ============================================================
    'CapacityAnalyzer',
    'VenueCapacity',
    'CapacityConstraint',

    # ============================================================
    # GRAIN FUTURES COMPARISON (PDF Section 2.4 REQUIRED)
    # ============================================================
    'GrainFuturesComparison',
    'GrainFuturesBenchmark',
    'CryptoPairCharacteristics',
    'ComparisonResult',
    'AssetClass',
    'MarketStructure',
    'compare_to_grain_futures',

    # ============================================================
    # WALK-FORWARD OPTIMIZATION (PDF: 18m train / 6m test)
    # ============================================================
    'WalkForwardOptimizer',
    'WalkForwardWindow',
    'WalkForwardConfig',
    'WindowResult',
    'WalkForwardResult',
    'create_walk_forward_optimizer',

    # ============================================================
    # VENUE-SPECIFIC BACKTESTING (PDF: CEX/DEX/Mixed/Combined)
    # ============================================================
    'VenueSpecificBacktester',
    'VenueCostModel',
    'VenueExecutionConfig',
    'VenueTradeResult',
    'VenueBacktestResult',
    'BacktestVenueType',
    'create_venue_backtester',

    # ============================================================
    # FULL METRICS (All PDF Section 2.4 metrics - 60+ metrics)
    # ============================================================
    'AdvancedMetricsCalculator',
    'PDFCompliantMetrics',
    'VenueSpecificMetrics',
    'RegimeSpecificMetrics',
    'TradingMetrics',  # Alias for TradeStatistics
    'CostMetrics',
    'CapacityMetrics',
    'RiskMetrics',
    'StatisticalMetrics',
    'ExecutionMetrics',
    'TimeBasedMetrics',
    'RollingMetrics',
    'CoreMetrics',
    'RiskAdjustedMetrics',
    'DrawdownMetrics',
    'create_metrics_calculator',

    # ============================================================
    # POSITION SIZING (PDF: $100k CEX, $20-50k DEX liquid, $5-10k illiquid)
    # ============================================================
    'PositionSizingEngine',
    'PositionSizeConfig',
    'PositionSizeResult',
    'PortfolioSizeResult',
    'SizingVenueType',
    'LiquidityTier',
    'create_position_sizing_engine',

    # ============================================================
    # CONCENTRATION LIMITS (PDF: 40% sector, 60% CEX, 20% Tier3)
    # ============================================================
    'ConcentrationLimitsEnforcer',
    'ConcentrationLimitsConfig',
    'ConcentrationLimit',
    'LimitBreachEvent',
    'RebalanceRecommendation',
    'LimitType',
    'BreachSeverity',
    'create_concentration_enforcer',

    # ============================================================
    # COMPREHENSIVE REPORTING (5-6 pages per PDF)
    # ============================================================
    'ComprehensiveReportGenerator',
    'ReportMetrics',
    'ReportSection',
    'WalkForwardSummary',
    'VenueBreakdown',
    'CrisisEventSummary',
    'GrainComparisonSummary',
    'create_comprehensive_report',

    # ============================================================
    # STEP 4 COMPLETE ORCHESTRATOR
    # ============================================================
    'Step4AdvancedOrchestrator',
    'OrchestratorConfig',
    'OrchestratorState',
    'ComponentStatus',
    'ExecutionMode',
    'RiskLevel',
    'IntegrationMode',
    'ComponentResult',
    'CrossValidationResult',
    'MonteCarloValidation',
    'MessageBus',
    'CheckpointManager',
    'RealTimeMonitor',
    'CrossValidationEngine',
    'MonteCarloEngine',
    'ResultSynthesisEngine',
    'DependencyGraph',
    'create_step4_orchestrator',
    'run_step4_advanced',
]

# Version and compliance information
__version__ = '3.0.0'
__pdf_compliance__ = 'Project Specification'
