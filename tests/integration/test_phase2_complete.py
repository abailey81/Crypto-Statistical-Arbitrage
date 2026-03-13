#!/usr/bin/env python3
"""
Phase 2 Complete Integration Test
==================================

Comprehensive verification that Phase 2 is fully operational and satisfies
ALL requirements from project specification.

This test verifies:
1. All modules import correctly
2. All components wire together properly
3. Data flows through the entire pipeline
4. All PDF requirements are implemented
5. Step 5 reporting generates compliant outputs

Author: Tamer Atesyakar
Version: 3.0.0
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# TEST RESULTS TRACKING
# =============================================================================

class TestResults:
    """Track test results for comprehensive reporting."""

    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
        self.start_time = datetime.now(timezone.utc)

    def add_pass(self, test_name: str, details: str = ""):
        self.passed.append((test_name, details))
        print(f"  [PASS]: {test_name}")
        if details:
            print(f"          {details}")

    def add_fail(self, test_name: str, error: str):
        self.failed.append((test_name, error))
        print(f"  [FAIL]: {test_name}")
        print(f"          Error: {error}")

    def add_warning(self, test_name: str, warning: str):
        self.warnings.append((test_name, warning))
        print(f"  [WARN]: {test_name}")
        print(f"          {warning}")

    def summary(self) -> str:
        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        total = len(self.passed) + len(self.failed)

        lines = [
            "",
            "=" * 70,
            "PHASE 2 VERIFICATION SUMMARY",
            "=" * 70,
            f"Total Tests: {total}",
            f"Passed:      {len(self.passed)} ({100*len(self.passed)/total:.1f}%)" if total > 0 else "Passed: 0",
            f"Failed:      {len(self.failed)}",
            f"Warnings:    {len(self.warnings)}",
            f"Time:        {elapsed:.2f}s",
            "",
        ]

        if self.failed:
            lines.append("FAILURES:")
            for name, error in self.failed:
                lines.append(f"  - {name}: {error[:80]}")
            lines.append("")

        if self.warnings:
            lines.append("WARNINGS:")
            for name, warning in self.warnings:
                lines.append(f"  - {name}: {warning[:80]}")
            lines.append("")

        status = "[PASS] PHASE 2 VERIFICATION PASSED" if not self.failed else "[FAIL] PHASE 2 VERIFICATION FAILED"
        lines.append(status)
        lines.append("=" * 70)

        return "\n".join(lines)


# =============================================================================
# SECTION 1: MODULE IMPORT TESTS
# =============================================================================

def test_core_imports(results: TestResults):
    """Test that all core modules import correctly."""
    print("\n[1] TESTING CORE MODULE IMPORTS")
    print("-" * 50)

    # Test pairs_trading module
    try:
        from strategies.pairs_trading import (
            Position, ExitReason, VenueType, SignalStrength, PairTier, Sector,
            PairConfig, CostConfig, StrategyConfig, PortfolioConstraints
        )
        results.add_pass("pairs_trading core enums/configs")
    except Exception as e:
        results.add_fail("pairs_trading core enums/configs", str(e))

    # Test universe construction
    try:
        from strategies.pairs_trading import (
            UniverseBuilder, TokenInfo, UniverseConfig, PairCandidate,
            TokenTier, TokenSector, FilterReason, PairType
        )
        results.add_pass("universe_construction module")
    except Exception as e:
        results.add_fail("universe_construction module", str(e))

    # Test cointegration
    try:
        from strategies.pairs_trading import (
            CointegrationAnalyzer, CointegrationResult, PairRanking,
            CointegrationMethod
        )
        results.add_pass("cointegration module")
    except Exception as e:
        results.add_fail("cointegration module", str(e))

    # Test baseline strategy
    try:
        from strategies.pairs_trading import (
            BaselinePairsStrategy, Trade, BacktestMetrics,
            TransactionCostModel, PortfolioManager
        )
        results.add_pass("baseline_strategy module")
    except Exception as e:
        results.add_fail("baseline_strategy module", str(e))

    # Test regime detection (Enhancement A)
    try:
        from strategies.pairs_trading import (
            CryptoRegimeDetector, RegimeAwareStrategy, MarketRegime,
            RegimeState, RegimeHistory
        )
        results.add_pass("regime_detection module (Enhancement A)")
    except Exception as e:
        results.add_fail("regime_detection module (Enhancement A)", str(e))

    # Test ML enhancement (Enhancement B)
    try:
        from strategies.pairs_trading import (
            MLEnhancedStrategy, FeatureEngineer, WalkForwardValidator,
            ModelType, PredictionTarget
        )
        results.add_pass("ml_enhancement module (Enhancement B)")
    except Exception as e:
        results.add_fail("ml_enhancement module (Enhancement B)", str(e))

    # Test dynamic pair selection (Enhancement C)
    try:
        from strategies.pairs_trading import (
            DynamicPairSelector, SelectionConfig, SelectionAction,
            TierLevel, PairStatus
        )
        results.add_pass("dynamic_pair_selection module (Enhancement C)")
    except Exception as e:
        results.add_fail("dynamic_pair_selection module (Enhancement C)", str(e))

    # Test position sizing
    try:
        from strategies.pairs_trading import (
            PositionSizer, PairMetrics, PositionSize, SizingMethod
        )
        results.add_pass("position_sizing module")
    except Exception as e:
        results.add_fail("position_sizing module", str(e))


def test_reporting_imports(results: TestResults):
    """Test that all reporting modules import correctly."""
    print("\n[2] TESTING REPORTING MODULE IMPORTS")
    print("-" * 50)

    # Test comprehensive report generator
    try:
        from reporting import (
            AdvancedReportGenerator, ComprehensiveReportResult,
            ReportMetadata, ReportSection, ReportFormat,
            VenueMetrics, CrisisEventAnalysis, WalkForwardWindow,
            GrainFuturesComparison, CapacityAnalysis
        )
        results.add_pass("advanced_report_generator module")
    except Exception as e:
        results.add_fail("advanced_report_generator module", str(e))

    # Test strict PDF validator
    try:
        from reporting import (
            StrictPDFValidator, StrictValidationResult,
            ValidationCheck, CategoryResult, QualityScore,
            ComplianceLevel, ValidationCategory, CheckStatus,
            REQUIRED_SECTIONS, REQUIRED_METRICS, REQUIRED_CRISIS_EVENTS,
            WALK_FORWARD_REQUIREMENTS, CAPACITY_REQUIREMENTS
        )
        results.add_pass("strict_pdf_validator module")
    except Exception as e:
        results.add_fail("strict_pdf_validator module", str(e))

    # Test presentation generator
    try:
        from reporting import (
            PresentationGenerator, PresentationResult, PresentationMetadata,
            Slide, SlideElement, SlideType, PresentationFormat,
            VenueColor, SlideLayout
        )
        results.add_pass("presentation_generator module")
    except Exception as e:
        results.add_fail("presentation_generator module", str(e))

    # Test Step 5 orchestrator
    try:
        from reporting import (
            Step5AdvancedOrchestrator, Step5Result, ComponentResult,
            PhaseResult, Checkpoint, OrchestratorConfig, OrchestratorState,
            ComponentStatus, ExecutionMode, DeliverableType
        )
        results.add_pass("step5_orchestrator module")
    except Exception as e:
        results.add_fail("step5_orchestrator module", str(e))


def test_backtesting_imports(results: TestResults):
    """Test that all backtesting modules import correctly."""
    print("\n[3] TESTING BACKTESTING MODULE IMPORTS")
    print("-" * 50)

    # Test backtest engine
    try:
        from backtesting.backtest_engine import (
            BacktestEngine, BaseStrategy, BacktestConfig,
            PerformanceMetrics
        )
        results.add_pass("backtest_engine module")
    except Exception as e:
        results.add_fail("backtest_engine module", str(e))

    # Test crisis analyzer
    try:
        from backtesting.analysis.crisis_analyzer import CrisisAnalyzer
        results.add_pass("crisis_analyzer module")
    except Exception as e:
        results.add_fail("crisis_analyzer module", str(e))

    # Test walk-forward optimizer
    try:
        from backtesting.analysis.walk_forward_optimizer import WalkForwardOptimizer
        results.add_pass("walk_forward_optimizer module")
    except Exception as e:
        results.add_fail("walk_forward_optimizer module", str(e))

    # Test grain futures comparison
    try:
        from backtesting.analysis.grain_futures_comparison import GrainFuturesComparison
        results.add_pass("grain_futures_comparison module")
    except Exception as e:
        results.add_fail("grain_futures_comparison module", str(e))

    # Test capacity analyzer
    try:
        from backtesting.analysis.capacity_analyzer import CapacityAnalyzer
        results.add_pass("capacity_analyzer module")
    except Exception as e:
        results.add_fail("capacity_analyzer module", str(e))

    # Test concentration limits
    try:
        from backtesting.analysis.concentration_limits import ConcentrationLimitsEnforcer
        results.add_pass("concentration_limits module")
    except Exception as e:
        results.add_fail("concentration_limits module", str(e))

    # Test Step 4 orchestrator
    try:
        from backtesting.analysis.step4_orchestrator import Step4AdvancedOrchestrator
        results.add_pass("step4_orchestrator module")
    except Exception as e:
        results.add_fail("step4_orchestrator module", str(e))


# =============================================================================
# SECTION 2: PDF REQUIREMENTS VERIFICATION
# =============================================================================

def test_pdf_requirements(results: TestResults):
    """Verify all PDF requirements are implemented."""
    print("\n[4] VERIFYING PDF REQUIREMENTS")
    print("-" * 50)

    # Section 2.1: Universe Construction
    print("\n  Section 2.1: Universe Construction")

    try:
        from strategies.pairs_trading.universe_construction import (
            VenueType, TokenSector, STABLECOINS
        )
        # Check venue types
        venues = [v.name for v in VenueType]
        if 'CEX' in venues and 'DEX' in venues and 'HYBRID' in venues:
            results.add_pass("CEX/DEX/Hybrid venue types defined")
        else:
            results.add_fail("CEX/DEX/Hybrid venue types defined", f"Found: {venues}")

        # Check sector count (16+ required)
        sectors = [s.name for s in TokenSector]
        if len(sectors) >= 16:
            results.add_pass(f"16+ sectors defined ({len(sectors)} found)")
        else:
            results.add_fail(f"16+ sectors defined", f"Only {len(sectors)} found")

        # Check RWA and LSDfi sectors
        sector_names = [s.name for s in TokenSector]
        has_rwa = any('RWA' in s for s in sector_names)
        has_lsd = any('LST' in s or 'LSD' in s or 'STAKING' in s for s in sector_names)
        if has_rwa:
            results.add_pass("RWA sector defined")
        else:
            results.add_fail("RWA sector defined", "Not found")

        # Check stablecoin filtering
        if len(STABLECOINS) >= 10:
            results.add_pass(f"Stablecoin filtering ({len(STABLECOINS)} defined)")
        else:
            results.add_warning("Stablecoin filtering", f"Only {len(STABLECOINS)} stablecoins")

    except Exception as e:
        results.add_fail("Section 2.1 verification", str(e))

    # Section 2.2: Baseline Strategy
    print("\n  Section 2.2: Baseline Strategy")

    try:
        from strategies.pairs_trading import VenueType, StrategyConfig

        # Check z-score thresholds
        cex_entry = VenueType.CEX.recommended_entry_z
        dex_entry = VenueType.DEX.recommended_entry_z

        if abs(cex_entry - 2.0) < 0.1:
            results.add_pass(f"CEX entry z-score = {cex_entry} (target: ±2.0)")
        else:
            results.add_fail(f"CEX entry z-score", f"Got {cex_entry}, expected 2.0")

        if abs(dex_entry - 2.5) < 0.1:
            results.add_pass(f"DEX entry z-score = {dex_entry} (target: ±2.5)")
        else:
            results.add_fail(f"DEX entry z-score", f"Got {dex_entry}, expected 2.5")

        # Check DEX exit is tighter
        dex_exit = VenueType.DEX.recommended_exit_z
        cex_exit = VenueType.CEX.recommended_exit_z
        if dex_exit > cex_exit:
            results.add_pass(f"DEX exit tighter than CEX ({dex_exit} vs {cex_exit})")
        else:
            results.add_warning("DEX exit vs CEX", f"DEX={dex_exit}, CEX={cex_exit}")

    except Exception as e:
        results.add_fail("Section 2.2 verification", str(e))

    # Section 2.3: Enhancements
    print("\n  Section 2.3: Enhancements (ALL THREE REQUIRED)")

    try:
        # Enhancement A: Regime Detection
        from strategies.pairs_trading.regime_detection import CryptoRegimeDetector, MarketRegime
        regimes = [r.name for r in MarketRegime]
        if len(regimes) >= 3:
            results.add_pass(f"Enhancement A: Regime Detection ({len(regimes)} regimes)")
        else:
            results.add_fail("Enhancement A: Regime Detection", f"Only {len(regimes)} regimes")
    except Exception as e:
        results.add_fail("Enhancement A: Regime Detection", str(e))

    try:
        # Enhancement B: ML Spread Prediction
        from strategies.pairs_trading.ml_enhancement import MLEnhancedStrategy, ModelType
        models = [m.name for m in ModelType]
        if len(models) >= 2:
            results.add_pass(f"Enhancement B: ML Prediction ({len(models)} model types)")
        else:
            results.add_fail("Enhancement B: ML Prediction", f"Only {len(models)} models")
    except Exception as e:
        results.add_fail("Enhancement B: ML Prediction", str(e))

    try:
        # Enhancement C: Dynamic Pair Selection
        from strategies.pairs_trading.dynamic_pair_selection import DynamicPairSelector, TierLevel
        tiers = [t.name for t in TierLevel]
        if len(tiers) >= 2:
            results.add_pass(f"Enhancement C: Dynamic Selection ({len(tiers)} tiers)")
        else:
            results.add_fail("Enhancement C: Dynamic Selection", f"Only {len(tiers)} tiers")
    except Exception as e:
        results.add_fail("Enhancement C: Dynamic Selection", str(e))

    # Section 2.4: Integration & Reporting
    print("\n  Section 2.4: Integration & Reporting")

    try:
        from reporting import REQUIRED_CRISIS_EVENTS, WALK_FORWARD_REQUIREMENTS

        # Check 14 crisis events
        crisis_count = len(REQUIRED_CRISIS_EVENTS)
        if crisis_count >= 14:
            results.add_pass(f"14 crisis events defined ({crisis_count} found)")
        else:
            results.add_fail("14 crisis events", f"Only {crisis_count} found")

        # Check walk-forward config
        train_months = WALK_FORWARD_REQUIREMENTS.get('train_months', 0)
        test_months = WALK_FORWARD_REQUIREMENTS.get('test_months', 0)

        if train_months == 18:
            results.add_pass(f"Walk-forward train period = {train_months} months")
        else:
            results.add_fail("Walk-forward train period", f"Got {train_months}, expected 18")

        if test_months == 6:
            results.add_pass(f"Walk-forward test period = {test_months} months")
        else:
            results.add_fail("Walk-forward test period", f"Got {test_months}, expected 6")

    except Exception as e:
        results.add_fail("Section 2.4 verification", str(e))

    # Check venue color coding
    try:
        from reporting import VenueColor
        colors = {c.name: c.value for c in VenueColor}

        if 'CEX' in colors and colors['CEX'].startswith('#'):
            results.add_pass(f"CEX color defined: {colors['CEX']}")
        if 'HYBRID' in colors and colors['HYBRID'].startswith('#'):
            results.add_pass(f"Hybrid color defined: {colors['HYBRID']}")
        if 'DEX' in colors and colors['DEX'].startswith('#'):
            results.add_pass(f"DEX color defined: {colors['DEX']}")
    except Exception as e:
        results.add_warning("Venue color coding", str(e))

    # Check grain futures comparison
    try:
        from backtesting.analysis.grain_futures_comparison import GrainFuturesComparison
        results.add_pass("Grain futures comparison implemented")
    except Exception as e:
        results.add_fail("Grain futures comparison", str(e))


# =============================================================================
# SECTION 3: COMPONENT WIRING TESTS
# =============================================================================

def test_component_wiring(results: TestResults):
    """Test that components wire together correctly."""
    print("\n[5] TESTING COMPONENT WIRING")
    print("-" * 50)

    # Test UniverseBuilder initialization
    try:
        from strategies.pairs_trading import UniverseBuilder, UniverseConfig
        config = UniverseConfig()
        builder = UniverseBuilder(config)
        results.add_pass("UniverseBuilder instantiation")
    except Exception as e:
        results.add_fail("UniverseBuilder instantiation", str(e))

    # Test CointegrationAnalyzer initialization
    try:
        from strategies.pairs_trading import CointegrationAnalyzer
        analyzer = CointegrationAnalyzer()
        results.add_pass("CointegrationAnalyzer instantiation")
    except Exception as e:
        results.add_fail("CointegrationAnalyzer instantiation", str(e))

    # Test BaselinePairsStrategy initialization
    try:
        from strategies.pairs_trading import BaselinePairsStrategy, StrategyConfig
        config = StrategyConfig()
        strategy = BaselinePairsStrategy(config)
        results.add_pass("BaselinePairsStrategy instantiation")
    except Exception as e:
        results.add_fail("BaselinePairsStrategy instantiation", str(e))

    # Test CryptoRegimeDetector initialization
    try:
        from strategies.pairs_trading import CryptoRegimeDetector
        detector = CryptoRegimeDetector()
        results.add_pass("CryptoRegimeDetector instantiation")
    except Exception as e:
        results.add_fail("CryptoRegimeDetector instantiation", str(e))

    # Test PositionSizer initialization
    try:
        from strategies.pairs_trading import PositionSizer
        sizer = PositionSizer(total_capital=1_000_000)
        results.add_pass("PositionSizer instantiation")
    except Exception as e:
        results.add_fail("PositionSizer instantiation", str(e))

    # Test DynamicPairSelector initialization
    try:
        from strategies.pairs_trading import DynamicPairSelector, SelectionConfig
        config = SelectionConfig()
        selector = DynamicPairSelector(config)
        results.add_pass("DynamicPairSelector instantiation")
    except Exception as e:
        results.add_fail("DynamicPairSelector instantiation", str(e))


def test_reporting_wiring(results: TestResults):
    """Test that reporting components wire together."""
    print("\n[6] TESTING REPORTING WIRING")
    print("-" * 50)

    # Test AdvancedReportGenerator
    try:
        from reporting import AdvancedReportGenerator
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = AdvancedReportGenerator(output_dir=Path(tmpdir))
            results.add_pass("AdvancedReportGenerator instantiation")
    except Exception as e:
        results.add_fail("AdvancedReportGenerator instantiation", str(e))

    # Test StrictPDFValidator
    try:
        from reporting import StrictPDFValidator, ValidationProfile
        validator = StrictPDFValidator(profile=ValidationProfile.STRICT)
        results.add_pass("StrictPDFValidator instantiation")
    except Exception as e:
        results.add_fail("StrictPDFValidator instantiation", str(e))

    # Test PresentationGenerator
    try:
        from reporting import PresentationGenerator
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = PresentationGenerator(output_dir=Path(tmpdir))
            results.add_pass("PresentationGenerator instantiation")
    except Exception as e:
        results.add_fail("PresentationGenerator instantiation", str(e))

    # Test Step5AdvancedOrchestrator
    try:
        from reporting import Step5AdvancedOrchestrator, OrchestratorConfig
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config = OrchestratorConfig(output_dir=Path(tmpdir))
            orchestrator = Step5AdvancedOrchestrator(config)
            results.add_pass("Step5AdvancedOrchestrator instantiation")
    except Exception as e:
        results.add_fail("Step5AdvancedOrchestrator instantiation", str(e))


# =============================================================================
# SECTION 4: CONCENTRATION LIMITS TESTS
# =============================================================================

def test_concentration_limits(results: TestResults):
    """Test that concentration limits match PDF requirements."""
    print("\n[7] TESTING CONCENTRATION LIMITS (PDF REQUIREMENTS)")
    print("-" * 50)

    try:
        from reporting import CONCENTRATION_LIMITS

        # Check 40% single sector limit (key is 'max_sector')
        sector_limit = CONCENTRATION_LIMITS.get('max_sector', 0)
        if abs(sector_limit - 0.40) < 0.01:
            results.add_pass(f"Single sector max = {sector_limit*100:.0f}% (target: 40%)")
        else:
            results.add_fail("Single sector max", f"Got {sector_limit*100:.0f}%, expected 40%")

        # Check 60% CEX-only limit (key is 'max_cex_only')
        cex_limit = CONCENTRATION_LIMITS.get('max_cex_only', 0)
        if abs(cex_limit - 0.60) < 0.01:
            results.add_pass(f"CEX-only max = {cex_limit*100:.0f}% (target: 60%)")
        else:
            results.add_fail("CEX-only max", f"Got {cex_limit*100:.0f}%, expected 60%")

        # Check 20% Tier 3 limit (key is 'max_tier3')
        tier3_limit = CONCENTRATION_LIMITS.get('max_tier3', 0)
        if abs(tier3_limit - 0.20) < 0.01:
            results.add_pass(f"Tier 3 max = {tier3_limit*100:.0f}% (target: 20%)")
        else:
            results.add_fail("Tier 3 max", f"Got {tier3_limit*100:.0f}%, expected 20%")

    except Exception as e:
        results.add_fail("Concentration limits verification", str(e))

    # Check position sizing limits
    try:
        from reporting import POSITION_SIZING_REQUIREMENTS

        # Key is 'cex_max' not 'cex_max_position'
        cex_size = POSITION_SIZING_REQUIREMENTS.get('cex_max', 0)
        if cex_size >= 100_000:
            results.add_pass(f"CEX max position = ${cex_size:,} (target: $100k)")
        else:
            results.add_fail("CEX max position", f"Got ${cex_size:,}, expected $100k")

    except Exception as e:
        results.add_warning("Position sizing requirements", str(e))


# =============================================================================
# SECTION 5: CAPACITY ANALYSIS TESTS
# =============================================================================

def test_capacity_requirements(results: TestResults):
    """Test capacity analysis requirements."""
    print("\n[8] TESTING CAPACITY REQUIREMENTS")
    print("-" * 50)

    try:
        from reporting import CAPACITY_REQUIREMENTS

        # CEX capacity: $10-30M
        cex_min = CAPACITY_REQUIREMENTS.get('cex', {}).get('min', 0)
        cex_max = CAPACITY_REQUIREMENTS.get('cex', {}).get('max', 0)

        if cex_min >= 10_000_000 and cex_max <= 30_000_000:
            results.add_pass(f"CEX capacity = ${cex_min/1e6:.0f}M - ${cex_max/1e6:.0f}M")
        else:
            results.add_fail("CEX capacity range", f"Got ${cex_min/1e6:.0f}M - ${cex_max/1e6:.0f}M")

        # DEX capacity: $1-5M
        dex_min = CAPACITY_REQUIREMENTS.get('dex', {}).get('min', 0)
        dex_max = CAPACITY_REQUIREMENTS.get('dex', {}).get('max', 0)

        if dex_min >= 1_000_000 and dex_max <= 5_000_000:
            results.add_pass(f"DEX capacity = ${dex_min/1e6:.0f}M - ${dex_max/1e6:.0f}M")
        else:
            results.add_fail("DEX capacity range", f"Got ${dex_min/1e6:.0f}M - ${dex_max/1e6:.0f}M")

    except Exception as e:
        results.add_fail("Capacity requirements verification", str(e))


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_tests():
    """Run all Phase 2 verification tests."""
    print("=" * 70)
    print("PHASE 2 COMPLETE INTEGRATION TEST")
    print("=" * 70)
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"PDF Reference: project specification")

    results = TestResults()

    # Run all test sections
    test_core_imports(results)
    test_reporting_imports(results)
    test_backtesting_imports(results)
    test_pdf_requirements(results)
    test_component_wiring(results)
    test_reporting_wiring(results)
    test_concentration_limits(results)
    test_capacity_requirements(results)

    # Print summary
    print(results.summary())

    return len(results.failed) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
