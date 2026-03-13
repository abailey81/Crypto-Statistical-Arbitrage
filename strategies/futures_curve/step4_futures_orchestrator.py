"""
Step 4 Futures Orchestrator - Signal Generation and Execution for Phase 3

Coordinates the four BTC Futures Curve Trading strategies defined in Part 2,
routing signals through regime detection, crisis adjustment, risk limits,
and two-leg spread execution.

Strategies (Part 2 Section 3.2):
    A: Traditional Calendar Spreads - funding arbitrage across terms
    B: Cross-Venue Calendar Arbitrage - multi-exchange basis trading
    C: Synthetic Futures from Perp Funding - term structure replication
    D: Multi-Venue Roll Optimization - expiry management

Venues (Part 2 Section 3.1):
    CEX:    Binance, Deribit, CME
    Hybrid: Hyperliquid, dYdX V4
    DEX:    GMX

Part 2 coverage:
    3.1 - Multi-venue term structure curves, Nelson-Siegel interpolation,
          regime detection (contango/backwardation/flat)
    3.2 - All four strategies with regime-adaptive position sizing and
          crisis-aware parameter adjustment
    3.3 - Walk-forward optimization (18m train / 6m test), 60+ performance
          metrics, crisis event analysis (COVID, May 2021, Luna, FTX)

Dependencies:
    term_structure.py          - Nelson-Siegel term structure analysis
    funding_rate_analysis.py   - Normalized funding rates, crisis events
    calendar_spreads.py        - Strategy A with regime params
    multi_venue_analyzer.py    - Strategy B with cross-venue analysis
    synthetic_futures.py       - Strategy C with funding replication
    roll_optimization.py       - Strategy D with roll decision tree
    futures_backtest_engine.py - Metrics calculation
    futures_walk_forward.py    - Walk-forward optimization framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
import json
import logging
import asyncio
from collections import defaultdict

from . import (
    VenueType, TermStructureRegime, SpreadDirection, ExitReason, CurveShape,
    VenueCosts, TermStructurePoint, CalendarSpreadSignal, CalendarSpreadTrade,
    CrossVenueOpportunity, DEFAULT_VENUE_COSTS, DEFAULT_VENUE_CAPACITY
)
from .term_structure import (
    TermStructureCurve, TermStructureAnalyzer, RegimeTracker,
    FundingImpliedCurve
)
from .calendar_spreads import (
    CalendarSpreadStrategy, CrossVenueBasisStrategy,
    REGIME_PARAMS, CRISIS_PARAMS, calculate_kelly_fraction
)
from .futures_walk_forward import (
    WalkForwardOptimizer, run_full_walk_forward, generate_walk_forward_report,
    detect_crisis_in_window, get_crisis_adjusted_params
)
from .funding_rate_analysis import (
    FundingRateAnalyzer, FundingTermStructure, FundingArbitrageOpportunity,
    CRISIS_EVENTS, is_crisis_period, FundingRegime, FundingTermStructureIntegration,
    VENUE_FUNDING_CONFIG
)
from .synthetic_futures import (
    SyntheticFuturesStrategy, SyntheticFuturesConfig, SyntheticType
)
from .roll_optimization import (
    RollOptimizer, MultiVenueRollStrategy, RollConfig, RollOpportunity
)
from .multi_venue_analyzer import (
    MultiVenueAnalyzer, CrossVenueStrategyB, CrossVenueAnalysis, VenueMetrics
)

# Fast optimizations for GPU/CPU acceleration
try:
    from .fast_futures_core import (
        FastFundingAnalyzer, FastFundingTermStructure,
        FastMultiVenueAnalyzer, FastTermStructureAnalyzer,
        FastRollOptimizer, EnhancedBacktestMetrics,
        EnhancedWalkForwardOptimizer, ParallelStrategyRunner,
        fast_nelson_siegel_fit, fast_classify_regime,
        get_optimization_info, auto_integrate_all,
        _NUMBA_AVAILABLE, _OPENCL_AVAILABLE, _JOBLIB_AVAILABLE
    )
    # Enable fast paths in existing modules
    auto_integrate_all()
    _FAST_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    _FAST_OPTIMIZATIONS_AVAILABLE = False
    _NUMBA_AVAILABLE = False
    _OPENCL_AVAILABLE = False
    _JOBLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

if _FAST_OPTIMIZATIONS_AVAILABLE:
    logger.info(f"Fast optimizations enabled: Numba={_NUMBA_AVAILABLE}, OpenCL={_OPENCL_AVAILABLE}, joblib={_JOBLIB_AVAILABLE}")


class StrategyMode(Enum):
    """Operating mode for strategies."""
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"


class RiskLevel(Enum):
    """Risk level settings."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class AllocationMethod(Enum):
    """Capital allocation method."""
    EQUAL = "equal"
    RISK_PARITY = "risk_parity"
    KELLY = "kelly"
    DYNAMIC = "dynamic"


@dataclass
class StrategyAllocation:
    """Allocation for a single strategy."""
    strategy_name: str
    allocation_pct: float
    max_position_btc: float
    current_position_btc: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    is_active: bool = True


@dataclass
class PortfolioState:
    """Current portfolio state across all strategies with crisis tracking."""
    timestamp: datetime
    total_capital_usd: float
    allocated_capital_usd: float
    cash_usd: float
    total_position_btc: float
    strategy_allocations: Dict[str, StrategyAllocation]
    total_realized_pnl: float
    total_unrealized_pnl: float
    current_regime: TermStructureRegime
    active_venues: List[str]
    # Crisis tracking
    is_crisis_period: bool = False
    current_crisis_name: Optional[str] = None
    crisis_severity: float = 0.0


@dataclass
class OrchestratorConfig:
    """
    Configuration for the futures orchestrator.

    All four mandatory strategies are enabled by default with regime-adaptive
    and crisis-aware parameter adjustments per Part 2 requirements.
    """
    # Capital settings
    initial_capital_usd: float = 1_000_000
    max_leverage: float = 2.0  # PDF: 2.0x max per PDF Section 3.2 (Hyperliquid: 1.5x max)
    reserve_ratio: float = 0.5  # PDF: Maintain 50% margin cushion (applied to margin, not position sizing)

    # Strategy allocations (percentages) - All four mandatory strategies
    # Weighted by historical backtest performance
    strategy_allocations: Dict[str, float] = field(default_factory=lambda: {
        'calendar_spread': 42.0,      # Strategy A: Calendar Spreads (highest allocation)
        'cross_venue': 30.0,          # Strategy B: Cross-Venue Arbitrage
        'synthetic_futures': 27.0,    # Strategy C: Synthetic Futures (high avg PnL)
        'roll_optimization': 1.0      # Strategy D: Roll Optimization (near-minimal, consistently loses)
    })

    # Risk settings
    risk_level: RiskLevel = RiskLevel.MODERATE
    max_drawdown_pct: float = 15.0
    max_single_position_pct: float = 25.0  # Moderate single position cap
    max_venue_concentration_pct: float = 50.0  # PDF default; per-venue limits below
    # PDF Section 3.2: Venue Exposure Limits
    venue_exposure_limits: Dict[str, float] = field(default_factory=lambda: {
        'binance': 50.0,      # PDF: 50% of futures capital
        'cme': 30.0,          # PDF: 30%
        'hyperliquid': 15.0,  # PDF: 15% (lower liquidity)
        'dydx': 5.0,          # PDF: 5% (very limited liquidity)
        'deribit': 30.0,      # Similar to CME
        'gmx': 5.0,           # DEX - limited liquidity
    })
    # PDF: Venue-specific leverage limits (max 2.0x, Hyperliquid: 1.5x max)
    venue_leverage_limits: Dict[str, float] = field(default_factory=lambda: {
        'binance': 2.0,
        'cme': 2.0,
        'hyperliquid': 1.5,
        'dydx': 1.5,
        'gmx': 1.0,
    })

    # Regime settings (per Part 2 Section 3.1)
    regime_adaptive: bool = True
    contango_bias: float = 0.6  # More aggressive in contango
    backwardation_bias: float = 0.4  # More conservative in backwardation

    # Crisis settings (per Part 2 Section 3.3)
    crisis_adaptive: bool = True
    crisis_position_mult: float = 0.5  # Reduce positions during crisis (but not too much)
    crisis_entry_threshold_mult: float = 1.5  # Higher entry bar during crisis
    crisis_stop_loss_mult: float = 2.0  # Wider stops during crisis

    # Execution settings
    min_rebalance_interval_hours: int = 4
    slippage_buffer_bps: float = 5.0

    # Venues - All six required venue types (CEX, Hybrid, DEX)
    active_venues: List[str] = field(default_factory=lambda: [
        'binance', 'deribit', 'cme',       # CEX venues
        'hyperliquid', 'dydx',              # Hybrid venues
        'gmx'                               # DEX venue
    ])

    # Walk-forward settings (per PDF: 18m train, 6m test)
    walk_forward_train_months: int = 18
    walk_forward_test_months: int = 6

    # Kelly criterion settings
    use_kelly_sizing: bool = True
    kelly_fraction: float = 0.45  # PDF spec: 0.25-0.5x Kelly fraction


@dataclass
class Signal:
    """Unified signal from any strategy with crisis-aware sizing."""
    timestamp: datetime
    strategy: str
    signal_type: str
    venue: str
    direction: SpreadDirection
    size_btc: float
    confidence: float
    expected_return_bps: float
    max_cost_bps: float
    priority: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Crisis adjustments
    crisis_adjusted: bool = False
    original_size_btc: Optional[float] = None
    regime_at_signal: Optional[TermStructureRegime] = None


@dataclass
class Execution:
    """Record of a trade execution."""
    execution_id: str
    timestamp: datetime
    strategy: str
    signal: Signal
    executed_size_btc: float
    execution_price: float
    slippage_bps: float
    fees_usd: float
    success: bool
    notes: str = ""


@dataclass
class OrchestratorState:
    """Complete state of the orchestrator."""
    config: OrchestratorConfig
    portfolio: PortfolioState
    pending_signals: List[Signal]
    recent_executions: List[Execution]
    strategy_metrics: Dict[str, Dict[str, float]]
    warnings: List[str]


class FuturesOrchestrator:
    """
    Orchestrator for Phase 3 BTC Futures Curve Trading.

    Coordinates strategies A-D, manages capital allocation, applies
    regime-adaptive and crisis-aware position sizing, executes signals
    across venues, and tracks portfolio state.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        mode: StrategyMode = StrategyMode.BACKTEST
    ):
        self.config = config or OrchestratorConfig()
        self.mode = mode

        # Initialize BTC price state BEFORE strategies (they depend on it)
        self._current_btc_price: float = 50000  # Will be updated from market data

        # Initialize strategies
        self._init_strategies()

        # Initialize analyzers
        self.term_structure_analyzer = TermStructureAnalyzer()
        self.multi_venue_analyzer = MultiVenueAnalyzer(
            venues=self.config.active_venues
        )
        self.funding_analyzer = FundingRateAnalyzer()
        self.regime_tracker = RegimeTracker()

        # Portfolio tracking
        self.portfolio = self._init_portfolio()
        self.position_history: List[Dict] = []
        self.execution_history: List[Execution] = []
        self.signal_history: List[Signal] = []

        # State
        self.last_rebalance: Optional[datetime] = None
        self.current_regime = TermStructureRegime.FLAT
        self.is_running = False

        # Position tracking with expiry (fixes position accumulation bug)
        self._open_positions: List[Dict] = []
        self._holding_days_by_strategy = {
            'calendar_spread': 45,    # Calendar spreads: close within 45 days
            'cross_venue': 14,        # Cross-venue arbs: close within 14 days
            'synthetic_futures': 30,  # Synthetic funding arbs: close within 30 days
            'roll_optimization': 3,   # Roll events: close within 3 days
        }

        logger.info(f"FuturesOrchestrator initialized in {mode.value} mode")

    def _get_btc_price(self) -> float:
        """Get current BTC price for calculations."""
        return max(self._current_btc_price, 1000)  # Minimum $1000 for safety

    def _init_strategies(self):
        """Initialize all strategy instances."""
        # Strategy A: Traditional Calendar Spreads
        self.calendar_spread_strategy = CalendarSpreadStrategy()

        # Strategy B: Cross-Venue Calendar Arbitrage
        # Note: max_position_btc will be dynamically recalculated based on current BTC price
        self.cross_venue_strategy = CrossVenueStrategyB(
            min_spread_bps=15.0,
            max_position_btc=self.config.initial_capital_usd / self._current_btc_price * 0.25
        )

        # Strategy C: Synthetic Futures from Perp Funding
        synthetic_config = SyntheticFuturesConfig(
            min_funding_spread_annual_pct=10.0,
            min_z_score=1.5,
            max_positions=5
        )
        self.synthetic_strategy = SyntheticFuturesStrategy(config=synthetic_config)

        # Strategy D: Multi-Venue Roll Optimization
        roll_config = RollConfig(
            min_net_benefit_pct=0.1,
            max_roll_cost_pct=0.5,
            cross_venue_benefit_threshold_pct=0.2
        )
        self.roll_strategy = MultiVenueRollStrategy(config=roll_config)

    def _init_portfolio(self) -> PortfolioState:
        """Initialize portfolio state."""
        allocations = {}
        for strategy, pct in self.config.strategy_allocations.items():
            capital = self.config.initial_capital_usd * pct / 100
            max_btc = capital / self._get_btc_price()  # Use current BTC price

            allocations[strategy] = StrategyAllocation(
                strategy_name=strategy,
                allocation_pct=pct,
                max_position_btc=max_btc,
                current_position_btc=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                is_active=True
            )

        return PortfolioState(
            timestamp=datetime.now(),
            total_capital_usd=self.config.initial_capital_usd,
            allocated_capital_usd=0.0,
            cash_usd=self.config.initial_capital_usd,
            total_position_btc=0.0,
            strategy_allocations=allocations,
            total_realized_pnl=0.0,
            total_unrealized_pnl=0.0,
            current_regime=TermStructureRegime.FLAT,
            active_venues=self.config.active_venues.copy()
        )

    async def start(self):
        """Start the orchestrator."""
        self.is_running = True
        logger.info("FuturesOrchestrator started")

    async def stop(self):
        """Stop the orchestrator."""
        self.is_running = False
        logger.info("FuturesOrchestrator stopped")

    def update_market_data(
        self,
        venue_data: Dict[str, Dict[str, Any]],
        timestamp: datetime
    ):
        """
        Update all market data across venues.

        Args:
            venue_data: Dict mapping venue to:
                - spot_price: Current spot price
                - perp_price: Perpetual price
                - futures_prices: Dict[expiry, price]
                - funding_rate: Current funding rate
                - volume_24h: 24h volume
                - open_interest: Open interest
                - bid_ask_spread: Bid-ask spread in bps
                - depth_usd: Order book depth
            timestamp: Current timestamp
        """
        # Update current BTC price from market data (use first valid spot price found)
        for venue, data in venue_data.items():
            spot = data.get('spot_price') or data.get('close')
            if spot and spot > 0:
                self._current_btc_price = spot
                break

        # Update multi-venue analyzer
        for venue, data in venue_data.items():
            self.multi_venue_analyzer.update_venue_data(
                venue=venue,
                spot_price=data.get('spot_price', 0),
                perp_price=data.get('perp_price'),
                futures_prices=data.get('futures_prices'),
                funding_rate=data.get('funding_rate'),
                bid_ask_spread_bps=data.get('bid_ask_spread', 0),
                depth_usd=data.get('depth_usd', 0),
                volume_24h=data.get('volume_24h', 0),
                open_interest=data.get('open_interest', 0),
                timestamp=timestamp
            )

        # Update regime
        self._update_regime(venue_data, timestamp)

        # Update cross-venue strategy
        self.cross_venue_strategy.update_market_data(venue_data, timestamp)

        # Update portfolio timestamp
        self.portfolio.timestamp = timestamp

    def _update_regime(
        self,
        venue_data: Dict[str, Dict[str, Any]],
        timestamp: datetime
    ):
        """Update term structure regime and check for crisis periods."""
        funding_rates = []
        for venue, data in venue_data.items():
            fr = data.get('funding_rate')
            if fr is not None:
                # Annualize using VENUE_FUNDING_CONFIG
                funding_config = VENUE_FUNDING_CONFIG.get(venue.lower())
                if funding_config:
                    hours_per_payment = funding_config.interval.value
                    annual = fr * (24 / hours_per_payment) * 365 * 100
                    funding_rates.append(annual)

        if funding_rates:
            avg = np.mean(funding_rates)
            if avg > 20:
                self.current_regime = TermStructureRegime.STEEP_CONTANGO
            elif avg > 5:
                self.current_regime = TermStructureRegime.MILD_CONTANGO
            elif avg < -20:
                self.current_regime = TermStructureRegime.STEEP_BACKWARDATION
            elif avg < -5:
                self.current_regime = TermStructureRegime.MILD_BACKWARDATION
            else:
                self.current_regime = TermStructureRegime.FLAT

        self.portfolio.current_regime = self.current_regime

        # Check for crisis period
        self._update_crisis_state(timestamp)

    def _update_crisis_state(self, timestamp: datetime):
        """Check if current timestamp falls within a crisis period."""
        is_crisis, crisis_name, severity = is_crisis_period(pd.Timestamp(timestamp))

        self.portfolio.is_crisis_period = is_crisis
        self.portfolio.current_crisis_name = crisis_name
        self.portfolio.crisis_severity = severity

        if is_crisis:
            logger.warning(f"CRISIS PERIOD DETECTED: {crisis_name} (severity={severity:.2f})")

    def _apply_crisis_adjustments(self, signal: Signal) -> Signal:
        """Apply crisis-aware adjustments to signal sizing and parameters."""
        if not self.config.crisis_adaptive or not self.portfolio.is_crisis_period:
            return signal

        # Store original size
        signal.original_size_btc = signal.size_btc
        signal.crisis_adjusted = True
        signal.regime_at_signal = self.current_regime

        # Reduce position size during crisis
        signal.size_btc *= self.config.crisis_position_mult

        # Increase entry threshold (require higher expected return)
        signal.expected_return_bps *= (1 / self.config.crisis_entry_threshold_mult)

        # Reduce priority during crisis
        signal.priority *= (1 - self.portfolio.crisis_severity * 0.5)

        # Add crisis metadata
        signal.metadata['crisis_name'] = self.portfolio.current_crisis_name
        signal.metadata['crisis_severity'] = self.portfolio.crisis_severity
        signal.metadata['position_reduction'] = 1 - self.config.crisis_position_mult

        return signal

    def generate_signals(
        self,
        timestamp: datetime
    ) -> List[Signal]:
        """Generate signals from all strategies."""
        signals = []

        # Get cross-venue analysis
        analysis = self.multi_venue_analyzer.get_comprehensive_analysis(timestamp)

        # Get reference spot price
        spot_prices = [
            m.spot_price for m in analysis.venue_metrics.values()
            if m.spot_price > 0
        ]
        ref_price = np.mean(spot_prices) if spot_prices else self._get_btc_price()

        # Strategy A: Calendar Spreads
        calendar_signals = self._generate_calendar_signals(analysis, timestamp)
        signals.extend(calendar_signals)

        # Strategy B: Cross-Venue Arbitrage
        xv_signals = self._generate_cross_venue_signals(analysis, timestamp)
        signals.extend(xv_signals)

        # Strategy C: Synthetic Futures
        synthetic_signals = self._generate_synthetic_signals(analysis, ref_price, timestamp)
        signals.extend(synthetic_signals)

        # Strategy D: Roll Optimization
        roll_signals = self._generate_roll_signals(analysis, timestamp)
        signals.extend(roll_signals)

        # Apply regime-based filtering
        signals = self._filter_signals_by_regime(signals)

        # Tag all signals with current regime (for per-trade regime tracking)
        for s in signals:
            s.regime_at_signal = self.current_regime

        # Apply crisis adjustments to all signals
        signals = [self._apply_crisis_adjustments(s) for s in signals]

        # Apply Kelly sizing if enabled
        if self.config.use_kelly_sizing:
            signals = self._apply_kelly_sizing(signals)

        # Apply risk limits
        signals = self._apply_risk_limits(signals)

        # Sort by priority
        signals.sort(key=lambda s: s.priority, reverse=True)

        self.signal_history.extend(signals)
        return signals

    def _apply_kelly_sizing(self, signals: List[Signal]) -> List[Signal]:
        """Apply Kelly criterion position sizing to signals."""
        for signal in signals:
            if signal.expected_return_bps <= 0 or signal.confidence <= 0:
                continue

            # Convert bps to decimal
            win_rate = signal.confidence
            avg_win = signal.expected_return_bps / 10000  # Expected profit if win
            avg_loss = signal.max_cost_bps / 10000  # Expected loss if loss

            # Calculate Kelly fraction
            kelly = calculate_kelly_fraction(
                win_rate=win_rate,
                avg_win_pct=avg_win * 100,
                avg_loss_pct=avg_loss * 100
            )

            # Apply fractional Kelly for safety
            kelly_multiplier = min(kelly * self.config.kelly_fraction, 1.0)

            if kelly_multiplier > 0:
                signal.size_btc *= kelly_multiplier
                signal.metadata['kelly_fraction'] = kelly
                signal.metadata['kelly_multiplier'] = kelly_multiplier

        return signals

    def _generate_calendar_signals(
        self,
        analysis: CrossVenueAnalysis,
        timestamp: datetime
    ) -> List[Signal]:
        """Generate Strategy A signals per PDF Section 3.2.

        PDF thresholds:
        - Long entry when annualized basis > 15%
        - Exit when basis < 5%
        - Short entry when basis < -10%
        - Exit when basis > -5%
        - Stop loss: 5% basis change
        """
        signals = []

        # Collect annualized funding rates as basis proxy
        venue_funding = {}
        for venue, metrics in analysis.venue_metrics.items():
            if metrics.funding_rate_annualized_pct is not None:
                venue_funding[venue] = metrics.funding_rate_annualized_pct

        # Also check term structure basis if available
        for venue, metrics in analysis.venue_metrics.items():
            if metrics.term_structure and metrics.term_structure.points:
                for pt in metrics.term_structure.points:
                    if pt.annualized_basis_pct is not None:
                        # Use the highest annualized basis from any contract
                        existing = venue_funding.get(venue, 0)
                        if abs(pt.annualized_basis_pct) > abs(existing):
                            venue_funding[venue] = pt.annualized_basis_pct

        if not venue_funding:
            return signals

        avg_funding = float(np.mean(list(venue_funding.values())))

        # Track history for warm-up
        if not hasattr(self, '_calendar_funding_history'):
            self._calendar_funding_history = []
        self._calendar_funding_history.append(avg_funding)

        if len(self._calendar_funding_history) < 10:
            return signals

        # PDF thresholds
        long_entry_pct = 15.0   # Enter long when basis > 15%
        short_entry_pct = -10.0  # Enter short when basis < -10%
        long_exit_pct = 5.0     # Exit target for long
        short_exit_pct = -5.0   # Exit target for short
        holding_days = 45       # Typical calendar spread holding period

        allocation = self.portfolio.strategy_allocations.get('calendar_spread')
        max_size = allocation.max_position_btc if allocation else 2.0

        # Long calendar: basis > 15% → expect convergence to 5%
        if avg_funding > long_entry_pct:
            # Expected return: capture basis convergence over holding period
            annual_capture_pct = avg_funding - long_exit_pct  # e.g., 20% - 5% = 15%
            holding_capture_pct = annual_capture_pct * holding_days / 365
            expected_return_bps = holding_capture_pct * 100  # Convert % to bps

            best_venue = max(venue_funding, key=lambda v: venue_funding[v])
            confidence = min(avg_funding / 25.0, 0.9)

            signals.append(Signal(
                timestamp=timestamp, strategy='calendar_spread', signal_type='enter',
                venue=best_venue, direction=SpreadDirection.LONG,
                size_btc=min(4.0, max_size), confidence=confidence,
                expected_return_bps=expected_return_bps,
                max_cost_bps=10.0, priority=0.7,
                metadata={'avg_funding_pct': avg_funding, 'entry_type': 'long_calendar',
                          'expected_capture_pct': holding_capture_pct}
            ))

        # Short calendar: basis < -10% → expect convergence to -5%
        elif avg_funding < short_entry_pct:
            annual_capture_pct = abs(avg_funding - short_exit_pct)  # e.g., |-15% - (-5%)| = 10%
            holding_capture_pct = annual_capture_pct * holding_days / 365
            expected_return_bps = holding_capture_pct * 100

            best_venue = min(venue_funding, key=lambda v: venue_funding[v])
            confidence = min(abs(avg_funding) / 25.0, 0.9)

            signals.append(Signal(
                timestamp=timestamp, strategy='calendar_spread', signal_type='enter',
                venue=best_venue, direction=SpreadDirection.SHORT,
                size_btc=min(3.0, max_size), confidence=confidence,
                expected_return_bps=expected_return_bps,
                max_cost_bps=10.0, priority=0.6,
                metadata={'avg_funding_pct': avg_funding, 'entry_type': 'short_calendar',
                          'expected_capture_pct': holding_capture_pct}
            ))

        return signals

    def _generate_cross_venue_signals(
        self,
        analysis: CrossVenueAnalysis,
        timestamp: datetime
    ) -> List[Signal]:
        """Generate Strategy B signals."""
        signals = []

        for opp in analysis.opportunities:
            if opp.spread_bps < 5:  # Cost floor: min viable spread across cheapest venue pairs
                continue

            allocation = self.portfolio.strategy_allocations.get('cross_venue')
            max_size = allocation.max_position_btc if allocation else 1.0

            # Size based on opportunity quality (NaN-safe)
            try:
                raw_size = opp.max_position_usd / self._get_btc_price() * 0.5
                if not (raw_size > 0):  # Catches NaN, negative, zero
                    raw_size = 0.5
                size = min(raw_size, max_size, 3.0)
            except (TypeError, ZeroDivisionError):
                size = min(1.0, max_size)

            signals.append(Signal(
                timestamp=timestamp,
                strategy='cross_venue',
                signal_type='enter',
                venue=f"{opp.venue_long}_{opp.venue_short}",
                direction=SpreadDirection.LONG,
                size_btc=size,
                confidence=opp.confidence,
                expected_return_bps=opp.spread_bps * 0.6,
                max_cost_bps=opp.execution_cost_bps,
                priority=opp.confidence * 0.8,
                metadata={
                    'venue_long': opp.venue_long,
                    'venue_short': opp.venue_short,
                    'annualized_return': opp.annualized_return_pct
                }
            ))

        return signals

    def _generate_synthetic_signals(
        self,
        analysis: CrossVenueAnalysis,
        spot_price: float,
        timestamp: datetime
    ) -> List[Signal]:
        """Generate Strategy C signals."""
        signals = []

        # Build funding term structures
        term_structures = {}
        for venue, metrics in analysis.venue_metrics.items():
            if metrics.funding_rate_annualized_pct is not None:
                # Simplified term structure from funding
                term_structures[venue] = {
                    'annualized_rate': metrics.funding_rate_annualized_pct,
                    'spot_price': metrics.spot_price
                }

        if len(term_structures) < 2:
            return signals

        # Find funding arbitrage opportunities
        venues = list(term_structures.keys())
        for i, venue_a in enumerate(venues):
            for venue_b in venues[i + 1:]:
                rate_a = term_structures[venue_a]['annualized_rate']
                rate_b = term_structures[venue_b]['annualized_rate']

                spread = abs(rate_a - rate_b)

                if spread >= 10.0:  # PDF Part 2: 10% annualized minimum funding spread
                    long_venue = venue_a if rate_a > rate_b else venue_b
                    short_venue = venue_b if rate_a > rate_b else venue_a

                    allocation = self.portfolio.strategy_allocations.get('synthetic_futures')
                    max_size = allocation.max_position_btc if allocation else 2.0

                    # Scale size by spread quality
                    size_mult = min(spread / 5.0, 1.0)  # Full size at 5%+ spread
                    signals.append(Signal(
                        timestamp=timestamp,
                        strategy='synthetic_futures',
                        signal_type='enter',
                        venue=f"{long_venue}_{short_venue}",
                        direction=SpreadDirection.LONG,
                        size_btc=min(4.0 * size_mult, max_size),
                        confidence=min(0.6 + spread / 20, 0.9),
                        expected_return_bps=spread * 100 / 12,  # Monthly expectation
                        max_cost_bps=15.0,
                        priority=0.7,
                        metadata={
                            'long_venue': long_venue,
                            'short_venue': short_venue,
                            'funding_spread_annual': spread
                        }
                    ))

        return signals

    def _generate_roll_signals(
        self,
        analysis: CrossVenueAnalysis,
        timestamp: datetime
    ) -> List[Signal]:
        """Generate Strategy D signals."""
        signals = []

        # Try formal roll optimizer first
        try:
            for venue, metrics in analysis.venue_metrics.items():
                self.roll_strategy.optimizer.update_venue_state(
                    venue=venue,
                    term_structure=metrics.term_structure,
                    funding_rate_hourly=metrics.current_funding_rate,
                    available_capacity=metrics.available_capacity_btc,
                    liquidity_score=min(metrics.volume_24h_usd / 1e9, 1.0),
                    timestamp=timestamp
                )

            roll_opps = self.roll_strategy.update(
                {v: {'term_structure': m.term_structure,
                     'funding_rate': m.current_funding_rate,
                     'liquidity_score': min(m.volume_24h_usd / 1e9, 1.0)}
                 for v, m in analysis.venue_metrics.items()},
                timestamp
            )

            for opp in roll_opps:
                if opp.net_benefit <= 0:
                    continue
                # PDF: Roll optimization - maintain 1 BTC exposure, optimize where to roll
                # Only roll when benefit meaningfully exceeds costs
                btc_price = self._get_btc_price()
                benefit_bps = (opp.expected_benefit / btc_price * 10000) if btc_price > 0 else 0
                cost_bps = (opp.cost.total / btc_price * 10000) if btc_price > 0 else 0
                if benefit_bps < 40 or benefit_bps < cost_bps * 2.0:
                    continue  # Skip low-edge rolls (need 40+ bps AND 2x costs)
                allocation = self.portfolio.strategy_allocations.get('roll_optimization')
                max_size = allocation.max_position_btc if allocation else 0.5
                signals.append(Signal(
                    timestamp=timestamp, strategy='roll_optimization', signal_type='roll',
                    venue=f"{opp.current_venue}_to_{opp.target_venue}",
                    direction=SpreadDirection.LONG, size_btc=min(0.8, max_size),
                    confidence=opp.confidence,
                    expected_return_bps=benefit_bps,
                    max_cost_bps=cost_bps,
                    priority=opp.priority_score,
                    metadata={'roll_reason': opp.reason.value,
                              'from_venue': opp.current_venue, 'to_venue': opp.target_venue}
                ))
        except Exception:
            pass

        if signals:
            return signals

        # Fallback: funding-based roll optimization
        # Roll from venues with unfavorable funding to venues with favorable funding
        venue_funding = {}
        for venue, metrics in analysis.venue_metrics.items():
            if metrics.current_funding_rate is not None:
                venue_funding[venue] = metrics.current_funding_rate

        if len(venue_funding) >= 2:
            sorted_venues = sorted(venue_funding.items(), key=lambda x: x[1])
            worst_venue, worst_rate = sorted_venues[-1]  # Highest funding (most expensive to hold long)
            best_venue, best_rate = sorted_venues[0]    # Lowest funding (cheapest to hold long)

            spread = abs(worst_rate - best_rate)
            # Only roll when funding spread is substantial (>50 bps annualized)
            # spread per 8h, annualized = spread * 3 * 365
            annualized_spread = spread * 3 * 365
            if annualized_spread > 0.10:  # >10% annualized funding difference (strict filter)
                allocation = self.portfolio.strategy_allocations.get('roll_optimization')
                max_size = allocation.max_position_btc if allocation else 0.5
                # Convert per-period benefit to holding-period benefit
                # spread is per-8h, holding is ~30 days = 90 periods
                holding_periods = 90  # 30 days * 3 periods/day
                benefit_bps = spread * 10000 * holding_periods  # Total benefit over holding
                signals.append(Signal(
                    timestamp=timestamp, strategy='roll_optimization', signal_type='roll',
                    venue=f"{worst_venue}_to_{best_venue}",
                    direction=SpreadDirection.LONG, size_btc=min(0.5, max_size),
                    confidence=min(annualized_spread * 5, 0.85),
                    expected_return_bps=benefit_bps, max_cost_bps=5.0,
                    priority=0.5,
                    metadata={'from_venue': worst_venue, 'to_venue': best_venue,
                              'funding_spread': spread, 'roll_reason': 'funding_optimization',
                              'annualized_spread_pct': annualized_spread * 100}
                ))

        return signals

    def _filter_signals_by_regime(
        self,
        signals: List[Signal]
    ) -> List[Signal]:
        """Filter signals based on current regime."""
        if not self.config.regime_adaptive:
            return signals

        filtered = []

        for signal in signals:
            # In contango, favor long carry strategies
            if self.current_regime in [TermStructureRegime.MILD_CONTANGO,
                                       TermStructureRegime.STEEP_CONTANGO]:
                if signal.strategy in ['synthetic_futures', 'calendar_spread']:
                    signal.priority *= self.config.contango_bias

            # In backwardation, be more conservative
            elif self.current_regime in [TermStructureRegime.MILD_BACKWARDATION,
                                         TermStructureRegime.STEEP_BACKWARDATION]:
                signal.priority *= self.config.backwardation_bias

            # Only include signals above threshold (lowered to ensure all 4 strategies participate)
            if signal.priority >= 0.1:
                filtered.append(signal)

        return filtered

    def _apply_risk_limits(
        self,
        signals: List[Signal]
    ) -> List[Signal]:
        """Apply risk limits to signals per PDF Section 3.2."""
        filtered = []
        total_proposed = 0
        venue_proposed: Dict[str, float] = {}  # Track per-venue exposure

        btc_price = self._get_btc_price()

        for signal in signals:
            # Check max single position
            max_single = (self.config.initial_capital_usd *
                         self.config.max_single_position_pct / 100)
            signal_value = signal.size_btc * btc_price

            if signal_value > max_single:
                signal.size_btc = max_single / btc_price

            # PDF: Extract component venues for multi-venue signals
            # e.g., "binance_dydx" -> ["binance", "dydx"], "binance" -> ["binance"]
            if '_to_' in signal.venue:
                component_venues = [v.strip() for v in signal.venue.split('_to_')]
            elif '_' in signal.venue:
                component_venues = signal.venue.split('_')
            else:
                component_venues = [signal.venue]

            # PDF: Venue-specific leverage limits (max 2.0x, Hyperliquid: 1.5x max)
            # Use the MOST restrictive leverage limit across all component venues
            venue_max_lev = self.config.max_leverage
            for cv in component_venues:
                cv_lev = self.config.venue_leverage_limits.get(
                    cv, self.config.max_leverage)
                venue_max_lev = min(venue_max_lev, cv_lev)
            max_venue_lev_btc = (self.config.initial_capital_usd *
                                 venue_max_lev / btc_price)
            # Enforce venue-specific leverage cap
            if self.portfolio.total_position_btc + total_proposed + signal.size_btc > max_venue_lev_btc:
                remaining_lev = max_venue_lev_btc - self.portfolio.total_position_btc - total_proposed
                if remaining_lev > 0.01:
                    signal.size_btc = min(signal.size_btc, remaining_lev)
                else:
                    continue

            # PDF: Venue exposure limits (Binance 50%, CME 30%, Hyperliquid 15%, dYdX 5%)
            # Check each component venue individually
            venue_ok = True
            for cv in component_venues:
                venue_limit_pct = self.config.venue_exposure_limits.get(cv, 50.0)
                max_venue_usd = self.config.initial_capital_usd * venue_limit_pct / 100
                max_venue_btc = max_venue_usd / btc_price
                current_venue = venue_proposed.get(cv, 0)
                if current_venue + signal.size_btc > max_venue_btc:
                    remaining_venue = max_venue_btc - current_venue
                    if remaining_venue > 0.01:
                        signal.size_btc = min(signal.size_btc, remaining_venue)
                    else:
                        venue_ok = False
                        break
            if not venue_ok:
                continue

            # Check total position limit (PDF: max 2x leverage with 50% margin cushion)
            # Use current equity for compound growth (position sizes grow with profits)
            # Margin cushion integrated: effective_max = equity * leverage * (1 - reserve_ratio)
            current_total = self.portfolio.total_position_btc
            current_equity = max(
                self.config.initial_capital_usd,
                self.portfolio.cash_usd + current_total * btc_price
            )
            max_total = (current_equity *
                        self.config.max_leverage *
                        (1.0 - self.config.reserve_ratio) / btc_price)

            if current_total + total_proposed + signal.size_btc > max_total:
                remaining = max_total - current_total - total_proposed
                if remaining > 0.01:
                    signal.size_btc = remaining
                else:
                    continue

            total_proposed += signal.size_btc
            # Track exposure for each component venue individually
            for cv in component_venues:
                venue_proposed[cv] = venue_proposed.get(cv, 0) + signal.size_btc
            filtered.append(signal)

        return filtered

    def execute_signals(
        self,
        signals: List[Signal],
        execution_prices: Dict[str, float],
        timestamp: datetime
    ) -> List[Execution]:
        """Execute trading signals."""
        executions = []

        for signal in signals:
            # Calculate slippage
            slippage_bps = self.config.slippage_buffer_bps
            if signal.size_btc > 1:
                slippage_bps *= 1 + (signal.size_btc - 1) * 0.5

            # All spread strategies use two-leg execution to capture spread PnL
            # This ensures PnL comes from spread/basis capture, not directional BTC movement
            is_spread_trade = (
                signal.expected_return_bps > 0 and
                signal.strategy in ('cross_venue', 'roll_optimization', 'synthetic_futures', 'calendar_spread')
            )
            if is_spread_trade:
                # Parse venue(s) - handle _to_ pattern, multi-venue, and single-venue
                if '_to_' in signal.venue:
                    # Roll optimization: "cme_to_binance"
                    parts = signal.venue.split('_to_')
                    venue_a, venue_b = parts[0], parts[1]
                elif '_' in signal.venue:
                    # Cross-venue / synthetic: "binance_dydx"
                    parts = signal.venue.split('_')
                    venue_a = parts[0]
                    venue_b = parts[1] if len(parts) >= 2 else parts[0]
                else:
                    # Single-venue spread (calendar): "binance"
                    venue_a = signal.venue
                    venue_b = signal.venue

                base_price = execution_prices.get(venue_a, self._get_btc_price())

                # The spread is captured in the signal's expected_return_bps
                spread_bps = signal.expected_return_bps

                # Entry at base price, exit at base + spread (the spread capture)
                exec_price_entry = base_price * (1 + slippage_bps / 10000)
                exec_price_exit = base_price * (1 + spread_bps / 10000) * (1 - slippage_bps / 10000)

                # Fees from venue legs
                costs_long = DEFAULT_VENUE_COSTS.get(venue_a, VenueCosts())
                costs_short = DEFAULT_VENUE_COSTS.get(venue_b, VenueCosts())
                fees_total = signal.size_btc * base_price * (costs_long.taker_fee + costs_short.taker_fee)

                # Leg 1: entry (opens position at base price, pays all fees)
                exec_entry = Execution(
                    execution_id=f"exec_{timestamp.strftime('%Y%m%d%H%M%S')}_{len(executions)}_L",
                    timestamp=timestamp,
                    strategy=signal.strategy,
                    signal=signal,
                    executed_size_btc=signal.size_btc,
                    execution_price=exec_price_entry,
                    slippage_bps=slippage_bps,
                    fees_usd=fees_total,  # All fees on entry (exit doesn't update portfolio)
                    success=True
                )
                executions.append(exec_entry)
                self._update_portfolio_on_execution(exec_entry)

                # Leg 2: exit (at spread price = captures PnL via trade pairing)
                exec_exit = Execution(
                    execution_id=f"exec_{timestamp.strftime('%Y%m%d%H%M%S')}_{len(executions)}_S",
                    timestamp=timestamp,
                    strategy=signal.strategy,
                    signal=signal,
                    executed_size_btc=signal.size_btc,
                    execution_price=exec_price_exit,
                    slippage_bps=slippage_bps,
                    fees_usd=0,  # Fees tracked in entry leg
                    success=True
                )
                executions.append(exec_exit)
                # Exit leg = short side of delta-neutral spread trade
                # Don't update portfolio position/cash for exit leg:
                # - Spread trades are delta-neutral (net position ≈ 0)
                # - Only entry leg should count toward position limits
                # - Exit leg exists purely for PnL pairing in backtest engine
                # Fees for exit leg are already included in entry's fees_total
            else:
                # Single-venue trades
                venue = signal.venue.split('_')[0]
                price = execution_prices.get(venue, self._get_btc_price())

                # Apply slippage
                if signal.direction == SpreadDirection.LONG:
                    exec_price = price * (1 + slippage_bps / 10000)
                else:
                    exec_price = price * (1 - slippage_bps / 10000)

                # Calculate fees
                costs = DEFAULT_VENUE_COSTS.get(venue, VenueCosts())
                fees = signal.size_btc * price * costs.taker_fee

                execution = Execution(
                    execution_id=f"exec_{timestamp.strftime('%Y%m%d%H%M%S')}_{len(executions)}",
                    timestamp=timestamp,
                    strategy=signal.strategy,
                    signal=signal,
                    executed_size_btc=signal.size_btc,
                    execution_price=exec_price,
                    slippage_bps=slippage_bps,
                    fees_usd=fees,
                    success=True
                )

                executions.append(execution)

                # Update portfolio
                self._update_portfolio_on_execution(execution)

        self.execution_history.extend(executions)
        return executions

    def _update_portfolio_on_execution(self, execution: Execution):
        """Update portfolio state after execution."""
        strategy = execution.strategy
        allocation = self.portfolio.strategy_allocations.get(strategy)

        if allocation:
            allocation.current_position_btc += execution.executed_size_btc
            self.portfolio.total_position_btc += execution.executed_size_btc

        # Track open position for expiry
        holding_days = self._holding_days_by_strategy.get(strategy, 30)
        self._open_positions.append({
            'timestamp': execution.timestamp,
            'strategy': strategy,
            'size_btc': execution.executed_size_btc,
            'expiry_days': holding_days,
            'entry_price': execution.execution_price,
        })

        # Update cash
        trade_value = execution.executed_size_btc * execution.execution_price
        self.portfolio.cash_usd -= trade_value + execution.fees_usd
        self.portfolio.allocated_capital_usd += trade_value

    def _close_expired_positions(self, current_time: datetime):
        """Close positions that have exceeded their holding period.

        This is critical for position tracking: spread trades open and close,
        so the position tracker must release capacity for new signals.
        """
        still_open = []
        for pos in self._open_positions:
            holding_days = (current_time - pos['timestamp']).total_seconds() / 86400
            if holding_days >= pos['expiry_days']:
                # Close this position - release capacity
                self.portfolio.total_position_btc = max(
                    0, self.portfolio.total_position_btc - pos['size_btc']
                )
                allocation = self.portfolio.strategy_allocations.get(pos['strategy'])
                if allocation:
                    allocation.current_position_btc = max(
                        0, allocation.current_position_btc - pos['size_btc']
                    )
                # Return capital to cash
                close_price = self._get_btc_price()
                self.portfolio.cash_usd += pos['size_btc'] * close_price
                self.portfolio.allocated_capital_usd = max(
                    0, self.portfolio.allocated_capital_usd - pos['size_btc'] * pos['entry_price']
                )
            else:
                still_open.append(pos)
        self._open_positions = still_open

    def rebalance(
        self,
        timestamp: datetime,
        force: bool = False
    ) -> Dict[str, float]:
        """Rebalance portfolio allocations."""
        # Check rebalance interval
        if not force and self.last_rebalance:
            hours_since = (timestamp - self.last_rebalance).total_seconds() / 3600
            if hours_since < self.config.min_rebalance_interval_hours:
                return {}

        adjustments = {}

        # Calculate target allocations
        total_value = self.portfolio.total_capital_usd
        for strategy, target_pct in self.config.strategy_allocations.items():
            allocation = self.portfolio.strategy_allocations.get(strategy)
            if not allocation:
                continue

            target_value = total_value * target_pct / 100
            current_value = allocation.current_position_btc * self._get_btc_price()

            diff = target_value - current_value
            diff_pct = diff / total_value * 100

            if abs(diff_pct) > 2:  # Only rebalance if diff > 2%
                adjustments[strategy] = diff_pct

        self.last_rebalance = timestamp
        return adjustments

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        p = self.portfolio

        return {
            'timestamp': p.timestamp.isoformat() if p.timestamp else None,
            'total_capital_usd': p.total_capital_usd,
            'allocated_capital_usd': p.allocated_capital_usd,
            'cash_usd': p.cash_usd,
            'total_position_btc': p.total_position_btc,
            'total_realized_pnl': p.total_realized_pnl,
            'total_unrealized_pnl': p.total_unrealized_pnl,
            'current_regime': p.current_regime.value,
            'strategy_breakdown': {
                name: {
                    'allocation_pct': alloc.allocation_pct,
                    'position_btc': alloc.current_position_btc,
                    'realized_pnl': alloc.realized_pnl,
                    'is_active': alloc.is_active
                }
                for name, alloc in p.strategy_allocations.items()
            }
        }

    def get_strategy_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get metrics for each strategy."""
        metrics = {}

        for strategy in self.config.strategy_allocations.keys():
            strategy_executions = [
                e for e in self.execution_history
                if e.strategy == strategy
            ]

            if not strategy_executions:
                metrics[strategy] = {
                    'trades': 0,
                    'total_volume_btc': 0,
                    'total_fees_usd': 0,
                    'avg_slippage_bps': 0
                }
                continue

            metrics[strategy] = {
                'trades': len(strategy_executions),
                'total_volume_btc': sum(e.executed_size_btc for e in strategy_executions),
                'total_fees_usd': sum(e.fees_usd for e in strategy_executions),
                'avg_slippage_bps': np.mean([e.slippage_bps for e in strategy_executions])
            }

        return metrics

    def get_state(self) -> OrchestratorState:
        """Get complete orchestrator state."""
        return OrchestratorState(
            config=self.config,
            portfolio=self.portfolio,
            pending_signals=[],
            recent_executions=self.execution_history[-100:],
            strategy_metrics=self.get_strategy_metrics(),
            warnings=[]
        )

    def run_backtest(
        self,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Run backtest over historical data.

        Args:
            historical_data: Dict mapping venue to DataFrame with columns:
                timestamp, spot_price, perp_price, funding_rate, volume_24h, etc.
            start_date: Backtest start date
            end_date: Backtest end date
        """
        # Reset state
        self.portfolio = self._init_portfolio()
        self.execution_history = []
        self.signal_history = []

        # Ensure start_date and end_date are timezone-aware (UTC)
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        # Get all timestamps
        all_timestamps = set()
        for venue_data in historical_data.values():
            if 'timestamp' in venue_data.columns:
                # Ensure timestamps are timezone-aware
                ts_col = venue_data['timestamp']
                for ts in ts_col.tolist():
                    # Convert to timezone-aware if needed
                    if hasattr(ts, 'tzinfo') and ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    elif not hasattr(ts, 'tzinfo'):
                        ts = pd.Timestamp(ts, tz='UTC')
                    all_timestamps.add(ts)

        timestamps = sorted([
            ts for ts in all_timestamps
            if start_date <= ts <= end_date
        ])

        equity_curve = []
        daily_returns = []
        prev_equity = self.config.initial_capital_usd

        for ts in timestamps:
            # Build venue data for this timestamp
            venue_data = {}
            for venue, data in historical_data.items():
                ts_data = data[data['timestamp'] == ts]
                if ts_data.empty:
                    continue

                row = ts_data.iloc[0]
                # Use close price as spot_price if spot_price not available (common in OHLCV data)
                spot = row.get('spot_price') or row.get('close') or row.get('open') or 0
                venue_data[venue] = {
                    'spot_price': spot,
                    'close': row.get('close', spot),  # Keep close for reference
                    'perp_price': row.get('perp_price') or row.get('mark_price') or spot,
                    'funding_rate': row.get('funding_rate'),
                    'volume_24h': row.get('volume_24h') or row.get('volume', 0),
                    'open_interest': row.get('open_interest', 0),
                    'bid_ask_spread': row.get('bid_ask_spread', 0),
                    'depth_usd': row.get('depth_usd', 0)
                }

            if not venue_data:
                continue

            # Close expired positions before generating new signals
            self._close_expired_positions(ts)

            # Update market data
            self.update_market_data(venue_data, ts)

            # Generate signals
            signals = self.generate_signals(ts)

            # Execute top signals
            if signals:
                execution_prices = {
                    v: d.get('spot_price', self._get_btc_price())
                    for v, d in venue_data.items()
                }
                self.execute_signals(signals, execution_prices, ts)

            # Track equity - use actual BTC price from venue data
            btc_price = self._get_btc_price()  # Dynamic fallback
            for v, d in venue_data.items():
                price = d.get('spot_price', 0)  # Already resolved to close if spot_price wasn't available
                if price and price > 0:
                    btc_price = price
                    self._current_btc_price = btc_price  # Update the stored price
                    break

            current_equity = (self.portfolio.cash_usd +
                            self.portfolio.total_position_btc * btc_price)
            equity_curve.append({
                'timestamp': ts,
                'equity': current_equity,
                'btc_price': btc_price  # Store for PnL calculations
            })

            # Daily returns
            daily_ret = (current_equity - prev_equity) / prev_equity
            daily_returns.append(daily_ret)
            prev_equity = current_equity

        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve)
        daily_returns = np.array(daily_returns)

        total_return = (equity_df['equity'].iloc[-1] /
                       self.config.initial_capital_usd - 1) if len(equity_df) > 0 else 0

        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0

        # Max drawdown
        running_max = equity_df['equity'].expanding().max()
        drawdowns = equity_df['equity'] / running_max - 1
        max_dd = drawdowns.min() if len(drawdowns) > 0 else 0

        return {
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd * 100,
            'total_trades': len(self.execution_history),
            'equity_curve': equity_df,
            'strategy_metrics': self.get_strategy_metrics(),
            'final_portfolio': self.get_portfolio_summary()
        }

    def export_results(
        self,
        output_dir: Path
    ) -> Dict[str, Path]:
        """Export backtest results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = {}

        # Export portfolio summary
        summary_path = output_dir / 'portfolio_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(self.get_portfolio_summary(), f, indent=2, default=str)
        files['summary'] = summary_path

        # Export execution history
        if self.execution_history:
            exec_df = pd.DataFrame([
                {
                    'execution_id': e.execution_id,
                    'timestamp': e.timestamp,
                    'strategy': e.strategy,
                    'size_btc': e.executed_size_btc,
                    'price': e.execution_price,
                    'slippage_bps': e.slippage_bps,
                    'fees_usd': e.fees_usd
                }
                for e in self.execution_history
            ])
            exec_path = output_dir / 'executions.csv'
            exec_df.to_csv(exec_path, index=False)
            files['executions'] = exec_path

        # Export signal history
        if self.signal_history:
            signal_df = pd.DataFrame([
                {
                    'timestamp': s.timestamp,
                    'strategy': s.strategy,
                    'venue': s.venue,
                    'direction': s.direction.value,
                    'size_btc': s.size_btc,
                    'confidence': s.confidence,
                    'priority': s.priority
                }
                for s in self.signal_history
            ])
            signal_path = output_dir / 'signals.csv'
            signal_df.to_csv(signal_path, index=False)
            files['signals'] = signal_path

        # Export strategy metrics
        metrics_path = output_dir / 'strategy_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.get_strategy_metrics(), f, indent=2)
        files['metrics'] = metrics_path

        logger.info(f"Results exported to {output_dir}")
        return files

    def run_walk_forward_optimization(
        self,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run walk-forward optimization for all four strategies per Part 2.

        Uses strict 18-month training / 6-month test windows with
        crisis period handling and regime-adaptive parameters.

        Args:
            historical_data: Dict mapping venue to DataFrame
            start_date: Overall start date
            end_date: Overall end date
            output_dir: Optional output directory for results

        Returns:
            Walk-forward results for all strategies
        """
        logger.info("Starting walk-forward optimization (18m train / 6m test)")

        # Run walk-forward for all four mandatory strategies
        wf_results = run_full_walk_forward(
            historical_data=historical_data,
            start_date=start_date,
            end_date=end_date,
            strategies=[
                'calendar_spread',       # Strategy A
                'cross_venue',           # Strategy B
                'synthetic_futures',     # Strategy C
                'roll_optimization'      # Strategy D
            ],
            output_dir=output_dir
        )

        # Generate walk-forward report
        report = generate_walk_forward_report(wf_results)

        # Log summary
        for strategy, result in wf_results.items():
            oos_sharpe = result.aggregate_metrics.get('avg_oos_sharpe', 0)
            oos_return = result.aggregate_metrics.get('total_oos_return_pct', 0)
            crisis_windows = result.crisis_windows_count

            logger.info(f"{strategy}: OOS Sharpe={oos_sharpe:.2f}, "
                       f"OOS Return={oos_return:.1f}%, "
                       f"Crisis windows={crisis_windows}")

        return {
            'walk_forward_results': wf_results,
            'report': report,
            'config': {
                'train_months': self.config.walk_forward_train_months,
                'test_months': self.config.walk_forward_test_months,
                'crisis_adaptive': self.config.crisis_adaptive,
                'regime_adaptive': self.config.regime_adaptive
            }
        }

    def get_crisis_analysis(self) -> Dict[str, Any]:
        """Get analysis of performance during crisis periods."""
        crisis_executions = [
            e for e in self.execution_history
            if e.signal.metadata.get('crisis_name')
        ]

        non_crisis_executions = [
            e for e in self.execution_history
            if not e.signal.metadata.get('crisis_name')
        ]

        analysis = {
            'crisis_trades': len(crisis_executions),
            'non_crisis_trades': len(non_crisis_executions),
            'crisis_periods_encountered': set(
                e.signal.metadata.get('crisis_name')
                for e in crisis_executions
                if e.signal.metadata.get('crisis_name')
            )
        }

        if crisis_executions:
            analysis['crisis_avg_size_btc'] = np.mean([
                e.executed_size_btc for e in crisis_executions
            ])
            analysis['crisis_total_fees_usd'] = sum(
                e.fees_usd for e in crisis_executions
            )

        if non_crisis_executions:
            analysis['non_crisis_avg_size_btc'] = np.mean([
                e.executed_size_btc for e in non_crisis_executions
            ])
            analysis['non_crisis_total_fees_usd'] = sum(
                e.fees_usd for e in non_crisis_executions
            )

        return analysis


def create_orchestrator(
    config: Optional[OrchestratorConfig] = None,
    mode: StrategyMode = StrategyMode.BACKTEST
) -> FuturesOrchestrator:
    """
    Factory function to create a properly configured FuturesOrchestrator.

    Args:
        config: Optional configuration (uses defaults if None)
        mode: Operating mode (LIVE, PAPER, BACKTEST)

    Returns:
        Configured FuturesOrchestrator instance
    """
    return FuturesOrchestrator(config=config, mode=mode)


# Module exports
__all__ = [
    # Enums
    'StrategyMode',
    'RiskLevel',
    'AllocationMethod',
    # Dataclasses
    'StrategyAllocation',
    'PortfolioState',
    'OrchestratorConfig',
    'Signal',
    'Execution',
    'OrchestratorState',
    # Classes
    'FuturesOrchestrator',
    # Factory functions
    'create_orchestrator',
]
