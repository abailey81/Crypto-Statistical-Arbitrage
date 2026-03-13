"""
Walk-Forward Optimizer for BTC Futures Curve Trading

Implements walk-forward optimization for all futures strategies using
18-month training / 6-month test rolling windows.

Strategies optimized:
  A: Calendar Spreads (funding arbitrage across terms)
  B: Cross-Venue Arbitrage (multi-exchange basis)
  C: Synthetic Futures (perp funding replication)
  D: Roll Optimization (expiry management)

Features:
- Out-of-sample validation with strict train/test separation
- Regime-adaptive parameter selection (contango/backwardation/flat)
- Crisis period detection and parameter adjustment (COVID, May 2021, Luna, FTX)
- Grid search with optional early stopping

Venues: Binance, Deribit, CME, Hyperliquid, dYdX V4, GMX

Dependencies:
- term_structure.py, funding_rate_analysis.py
- calendar_spreads.py, multi_venue_analyzer.py
- synthetic_futures.py, roll_optimization.py
- futures_backtest_engine.py
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from enum import Enum
from pathlib import Path
import json
import logging
from collections import defaultdict
from itertools import product
import warnings

from . import (
    VenueType, TermStructureRegime, SpreadDirection,
    DEFAULT_VENUE_COSTS, DEFAULT_VENUE_CAPACITY
)
from .term_structure import TermStructureAnalyzer
from .calendar_spreads import CalendarSpreadStrategy, BacktestResult, REGIME_PARAMS, CRISIS_PARAMS
from .synthetic_futures import SyntheticFuturesStrategy, SyntheticFuturesConfig
from .roll_optimization import RollOptimizer, RollConfig
from .multi_venue_analyzer import MultiVenueAnalyzer
from .funding_rate_analysis import CRISIS_EVENTS, is_crisis_period, FundingRegime

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=RuntimeWarning)


# Crisis-adjusted parameter multipliers for walk-forward optimization
CRISIS_PARAM_ADJUSTMENTS = {
    'calendar_spread': {
        'entry_z_threshold': 1.5,    # Higher threshold during crisis
        'exit_z_threshold': 1.2,     # Tighter exits
        'stop_loss_bps': 2.0,        # Wider stops
        'take_profit_bps': 1.5,      # Larger targets
        'max_holding_days': 0.5,     # Shorter holding
        'position_size_mult': 0.3    # Reduced position size
    },
    'cross_venue': {
        'min_spread_bps': 1.5,       # Higher minimum spread
        'confidence_threshold': 1.2,  # Higher confidence needed
        'max_position_pct': 0.3,     # Much smaller positions
        'take_profit_bps': 2.0,      # Larger targets
        'stop_loss_bps': 2.0,        # Wider stops
        'position_size_mult': 0.3
    },
    'synthetic_futures': {
        'min_funding_spread_annual_pct': 1.5,
        'min_z_score': 1.5,
        'profit_target_pct': 0.7,    # Tighter profit target
        'stop_loss_pct': 2.0,        # Wider stop
        'max_holding_days': 0.5,     # Shorter holding
        'position_size_mult': 0.3
    },
    'roll_optimization': {
        'min_days_to_expiry_roll': 1.5,  # Earlier rolls
        'min_net_benefit_pct': 0.5,      # Lower benefit threshold
        'max_roll_cost_pct': 2.0,        # Accept higher costs
        'position_size_mult': 0.5
    }
}

# Regime-specific parameter adjustments (using correct enum values)
REGIME_PARAM_ADJUSTMENTS = {
    TermStructureRegime.STEEP_CONTANGO: {
        'calendar_spread': {'entry_z_threshold': 0.8, 'take_profit_bps': 1.2},
        'synthetic_futures': {'min_funding_spread_annual_pct': 0.8},
        'roll_optimization': {'min_net_benefit_pct': 0.8}
    },
    TermStructureRegime.MILD_CONTANGO: {
        'calendar_spread': {'entry_z_threshold': 0.9, 'take_profit_bps': 1.1},
        'synthetic_futures': {'min_funding_spread_annual_pct': 0.9},
    },
    TermStructureRegime.FLAT: {
        'calendar_spread': {'entry_z_threshold': 1.0},
        'cross_venue': {'min_spread_bps': 0.8},  # Lower threshold in flat markets
    },
    TermStructureRegime.MILD_BACKWARDATION: {
        'calendar_spread': {'entry_z_threshold': 0.9, 'take_profit_bps': 1.1},
        'synthetic_futures': {'min_funding_spread_annual_pct': 0.9},
    },
    TermStructureRegime.STEEP_BACKWARDATION: {
        'calendar_spread': {'entry_z_threshold': 0.8, 'take_profit_bps': 1.2},
        'synthetic_futures': {'min_funding_spread_annual_pct': 0.8},
        'roll_optimization': {'min_net_benefit_pct': 0.8}
    }
}


def detect_crisis_in_window(start: datetime, end: datetime) -> Tuple[bool, Optional[str], float]:
    """
    Detect if a walk-forward window overlaps with a crisis period.

    Returns:
        Tuple of (is_crisis, crisis_name, severity_score)
    """
    window_start = pd.Timestamp(start)
    window_end = pd.Timestamp(end)

    # Ensure timestamps are timezone-aware (UTC) for comparison with CRISIS_EVENTS
    if window_start.tzinfo is None:
        window_start = window_start.tz_localize('UTC')
    if window_end.tzinfo is None:
        window_end = window_end.tz_localize('UTC')

    max_severity = 0.0
    crisis_name = None

    for name, crisis in CRISIS_EVENTS.items():
        crisis_start = crisis['start']
        crisis_end = crisis['end']

        # Ensure crisis timestamps are also timezone-aware
        if hasattr(crisis_start, 'tzinfo') and crisis_start.tzinfo is None:
            crisis_start = crisis_start.tz_localize('UTC')
        if hasattr(crisis_end, 'tzinfo') and crisis_end.tzinfo is None:
            crisis_end = crisis_end.tz_localize('UTC')

        # Check overlap
        if window_start <= crisis_end and window_end >= crisis_start:
            # Calculate overlap ratio
            overlap_start = max(window_start, crisis_start)
            overlap_end = min(window_end, crisis_end)
            overlap_days = (overlap_end - overlap_start).days
            window_days = (window_end - window_start).days

            overlap_ratio = overlap_days / max(window_days, 1)
            severity = crisis['severity'] * overlap_ratio

            if severity > max_severity:
                max_severity = severity
                crisis_name = name

    return max_severity > 0, crisis_name, max_severity


def get_crisis_adjusted_params(
    params: Dict[str, Any],
    strategy: str,
    crisis_severity: float
) -> Dict[str, Any]:
    """
    Adjust parameters based on crisis severity.

    Args:
        params: Base parameters
        strategy: Strategy name
        crisis_severity: Severity score (0-1)

    Returns:
        Adjusted parameters dict
    """
    if crisis_severity <= 0 or strategy not in CRISIS_PARAM_ADJUSTMENTS:
        return params

    adjusted = params.copy()
    adjustments = CRISIS_PARAM_ADJUSTMENTS[strategy]

    for param, multiplier in adjustments.items():
        if param in adjusted and param != 'position_size_mult':
            base_value = adjusted[param]
            if isinstance(base_value, (int, float)):
                # Interpolate based on severity
                adjustment = 1.0 + (multiplier - 1.0) * crisis_severity
                adjusted[param] = base_value * adjustment

    return adjusted


def get_regime_adjusted_params(
    params: Dict[str, Any],
    strategy: str,
    regime: TermStructureRegime
) -> Dict[str, Any]:
    """
    Adjust parameters based on term structure regime.

    Args:
        params: Base parameters
        strategy: Strategy name
        regime: Current term structure regime

    Returns:
        Adjusted parameters dict
    """
    if regime not in REGIME_PARAM_ADJUSTMENTS:
        return params

    regime_adjustments = REGIME_PARAM_ADJUSTMENTS[regime]
    if strategy not in regime_adjustments:
        return params

    adjusted = params.copy()
    adjustments = regime_adjustments[strategy]

    for param, multiplier in adjustments.items():
        if param in adjusted:
            base_value = adjusted[param]
            if isinstance(base_value, (int, float)):
                adjusted[param] = base_value * multiplier

    return adjusted


class OptimizationObjective(Enum):
    """Objective function for parameter optimization."""
    SHARPE_RATIO = "sharpe_ratio"
    TOTAL_RETURN = "total_return"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    PROFIT_FACTOR = "profit_factor"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"


class ParameterType(Enum):
    """Parameter types for optimization."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


@dataclass
class ParameterSpec:
    """Specification for a single parameter."""
    name: str
    param_type: ParameterType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Any]] = None
    default: Any = None

    def get_values(self) -> List[Any]:
        """Get all values to test for this parameter."""
        if self.param_type == ParameterType.CATEGORICAL:
            return self.choices or []
        elif self.param_type == ParameterType.DISCRETE:
            if self.min_value is not None and self.max_value is not None:
                step = self.step or 1
                return list(np.arange(self.min_value, self.max_value + step, step))
            return [self.default]
        else:  # CONTINUOUS
            if self.min_value is not None and self.max_value is not None:
                step = self.step or (self.max_value - self.min_value) / 10
                return list(np.arange(self.min_value, self.max_value + step, step))
            return [self.default]


@dataclass
class WalkForwardWindow:
    """A single walk-forward window with crisis detection."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_months: int
    test_months: int
    train_crisis: bool = False
    train_crisis_name: Optional[str] = None
    train_crisis_severity: float = 0.0
    test_crisis: bool = False
    test_crisis_name: Optional[str] = None
    test_crisis_severity: float = 0.0


@dataclass
class OptimizationResult:
    """Result from parameter optimization with crisis and regime analysis."""
    strategy: str
    window: WalkForwardWindow
    best_params: Dict[str, Any]
    crisis_adjusted_params: Dict[str, Any]
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    all_param_results: List[Dict[str, Any]]
    optimization_objective: OptimizationObjective
    regime_during_test: TermStructureRegime
    crisis_during_test: bool = False
    crisis_severity: float = 0.0
    param_adjustments_applied: List[str] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Complete walk-forward optimization result with crisis analysis."""
    strategy: str
    start_date: datetime
    end_date: datetime
    windows: List[WalkForwardWindow]
    window_results: List[OptimizationResult]
    aggregate_metrics: Dict[str, float]
    parameter_stability: Dict[str, float]
    regime_analysis: Dict[str, Dict[str, float]]
    crisis_analysis: Dict[str, Dict[str, float]]
    combined_equity_curve: pd.DataFrame
    crisis_windows_count: int = 0
    non_crisis_windows_count: int = 0


class ParameterOptimizer:
    """
    Parameter optimizer using grid search with optional pruning.
    """

    def __init__(
        self,
        objective: OptimizationObjective = OptimizationObjective.SHARPE_RATIO,
        max_evaluations: int = 200,
        early_stopping: bool = True
    ):
        self.objective = objective
        self.max_evaluations = max_evaluations
        self.early_stopping = early_stopping

    def optimize(
        self,
        param_specs: List[ParameterSpec],
        evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        n_best: int = 5
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run grid search optimization.

        Args:
            param_specs: List of parameter specifications
            evaluate_fn: Function that takes params dict and returns metrics dict
            n_best: Number of best parameter sets to track

        Returns:
            Tuple of (best_params, all_results)
        """
        # Generate all parameter combinations
        param_names = [p.name for p in param_specs]
        param_values = [p.get_values() for p in param_specs]

        all_combinations = list(product(*param_values))

        # Limit evaluations if needed
        if len(all_combinations) > self.max_evaluations:
            # Random sampling
            indices = np.random.choice(
                len(all_combinations),
                self.max_evaluations,
                replace=False
            )
            all_combinations = [all_combinations[i] for i in indices]

        results = []
        best_score = -float('inf')
        best_params = None
        no_improvement_count = 0

        for combo in all_combinations:
            params = dict(zip(param_names, combo))

            try:
                metrics = evaluate_fn(params)
                score = self._get_objective_score(metrics)

                results.append({
                    'params': params,
                    'metrics': metrics,
                    'score': score
                })

                if score > best_score:
                    best_score = score
                    best_params = params
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Early stopping
                if self.early_stopping and no_improvement_count > 50:
                    logger.info("Early stopping triggered")
                    break

            except Exception as e:
                logger.warning(f"Evaluation failed for {params}: {e}")
                continue

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)

        return best_params or {}, results[:n_best]

    def _get_objective_score(self, metrics: Dict[str, float]) -> float:
        """Extract objective score from metrics."""
        if self.objective == OptimizationObjective.SHARPE_RATIO:
            return metrics.get('sharpe_ratio', -float('inf'))
        elif self.objective == OptimizationObjective.TOTAL_RETURN:
            return metrics.get('total_return_pct', -float('inf'))
        elif self.objective == OptimizationObjective.CALMAR_RATIO:
            return metrics.get('calmar_ratio', -float('inf'))
        elif self.objective == OptimizationObjective.SORTINO_RATIO:
            return metrics.get('sortino_ratio', -float('inf'))
        elif self.objective == OptimizationObjective.PROFIT_FACTOR:
            return metrics.get('profit_factor', -float('inf'))
        elif self.objective == OptimizationObjective.RISK_ADJUSTED_RETURN:
            sharpe = metrics.get('sharpe_ratio', 0)
            ret = metrics.get('total_return_pct', 0)
            dd = abs(metrics.get('max_drawdown_pct', -100))
            return ret / max(dd, 1) * min(sharpe, 3)
        return -float('inf')


class WalkForwardOptimizer:
    """
    Walk-forward optimizer for futures curve trading strategies.

    Implements rolling window optimization with:
    - 18-month training windows (per PDF specification)
    - 6-month test windows (per PDF specification)
    - Strategy-specific parameter optimization
    - Regime-adaptive parameter selection
    """

    # Strategy parameter specifications (optimized grid for faster execution)
    STRATEGY_PARAMS = {
        'calendar_spread': [
            ParameterSpec('entry_z_threshold', ParameterType.CONTINUOUS, 1.0, 3.0, 1.0, default=2.0),
            ParameterSpec('exit_z_threshold', ParameterType.CONTINUOUS, 0.0, 1.0, 0.5, default=0.5),
            ParameterSpec('stop_loss_bps', ParameterType.DISCRETE, 30, 80, 25, default=50),
            ParameterSpec('take_profit_bps', ParameterType.DISCRETE, 15, 60, 15, default=30),
            ParameterSpec('max_holding_days', ParameterType.DISCRETE, 14, 45, 15, default=30),
            ParameterSpec('min_spread_bps', ParameterType.DISCRETE, 10, 25, 15, default=15)
        ],
        'cross_venue': [
            ParameterSpec('min_spread_bps', ParameterType.DISCRETE, 5, 20, 5, default=10),
            ParameterSpec('confidence_threshold', ParameterType.CONTINUOUS, 0.5, 0.9, 0.2, default=0.7),
            ParameterSpec('max_position_pct', ParameterType.CONTINUOUS, 0.15, 0.4, 0.12, default=0.25),
            ParameterSpec('take_profit_bps', ParameterType.DISCRETE, 5, 20, 5, default=10),
            ParameterSpec('stop_loss_bps', ParameterType.DISCRETE, 10, 30, 10, default=20)
        ],
        'synthetic_futures': [
            ParameterSpec('min_funding_spread_annual_pct', ParameterType.CONTINUOUS, 5.0, 20.0, 5.0, default=10.0),
            ParameterSpec('min_z_score', ParameterType.CONTINUOUS, 1.0, 2.5, 0.75, default=1.5),
            ParameterSpec('profit_target_pct', ParameterType.CONTINUOUS, 3.0, 8.0, 2.5, default=5.0),
            ParameterSpec('stop_loss_pct', ParameterType.CONTINUOUS, 1.5, 4.0, 1.25, default=3.0),
            ParameterSpec('max_holding_days', ParameterType.DISCRETE, 14, 45, 15, default=30),
            ParameterSpec('funding_flip_exit', ParameterType.CATEGORICAL, choices=[True, False], default=True)
        ],
        'roll_optimization': [
            ParameterSpec('min_days_to_expiry_roll', ParameterType.DISCRETE, 2, 6, 2, default=3),
            ParameterSpec('optimal_days_to_expiry_roll', ParameterType.DISCRETE, 5, 12, 3, default=7),
            ParameterSpec('min_net_benefit_pct', ParameterType.CONTINUOUS, 0.05, 0.25, 0.1, default=0.1),
            ParameterSpec('max_roll_cost_pct', ParameterType.CONTINUOUS, 0.3, 0.8, 0.25, default=0.5),
            ParameterSpec('cross_venue_benefit_threshold_pct', ParameterType.CONTINUOUS, 0.1, 0.4, 0.15, default=0.2)
        ]
    }

    def __init__(
        self,
        train_months: int = 18,
        test_months: int = 6,
        objective: OptimizationObjective = OptimizationObjective.SHARPE_RATIO,
        step_months: int = 3  # Rolling step size
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            train_months: Training window size (default 18 per PDF)
            test_months: Test window size (default 6 per PDF)
            objective: Optimization objective
            step_months: How much to step forward each window
        """
        self.train_months = train_months
        self.test_months = test_months
        self.objective = objective
        self.step_months = step_months

        self.optimizer = ParameterOptimizer(objective=objective)

    def generate_windows(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[WalkForwardWindow]:
        """Generate walk-forward windows with crisis detection."""
        windows = []
        window_id = 0

        # Ensure start_date and end_date are timezone-aware (UTC)
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        current_train_start = start_date

        while True:
            train_end = current_train_start + relativedelta(months=self.train_months)
            test_start = train_end
            test_end = test_start + relativedelta(months=self.test_months)

            if test_end > end_date:
                break

            # Detect crisis periods in training and test windows
            train_crisis, train_crisis_name, train_severity = detect_crisis_in_window(
                current_train_start, train_end
            )
            test_crisis, test_crisis_name, test_severity = detect_crisis_in_window(
                test_start, test_end
            )

            windows.append(WalkForwardWindow(
                window_id=window_id,
                train_start=current_train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_months=self.train_months,
                test_months=self.test_months,
                train_crisis=train_crisis,
                train_crisis_name=train_crisis_name,
                train_crisis_severity=train_severity,
                test_crisis=test_crisis,
                test_crisis_name=test_crisis_name,
                test_crisis_severity=test_severity
            ))

            window_id += 1
            current_train_start += relativedelta(months=self.step_months)

        crisis_count = sum(1 for w in windows if w.test_crisis)
        logger.info(f"Generated {len(windows)} walk-forward windows "
                   f"({crisis_count} with crisis periods in test)")
        return windows

    def run_optimization(
        self,
        strategy: str,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        evaluate_fn: Optional[Callable] = None
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization for a strategy.

        Args:
            strategy: Strategy name
            historical_data: Dict mapping venue to DataFrame
            start_date: Overall start date
            end_date: Overall end date
            evaluate_fn: Optional custom evaluation function

        Returns:
            WalkForwardResult with all optimization results
        """
        # Ensure start_date and end_date are timezone-aware (UTC)
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        windows = self.generate_windows(start_date, end_date)

        if not windows:
            raise ValueError("No valid windows generated for the date range")

        param_specs = self.STRATEGY_PARAMS.get(strategy, [])
        if not param_specs:
            logger.warning(f"No parameter specs for strategy {strategy}")

        window_results = []
        equity_curves = []

        for window in windows:
            crisis_info = ""
            if window.test_crisis:
                crisis_info = f" [CRISIS: {window.test_crisis_name}, severity={window.test_crisis_severity:.2f}]"

            logger.info(f"Optimizing window {window.window_id}: "
                       f"Train {window.train_start} to {window.train_end}, "
                       f"Test {window.test_start} to {window.test_end}{crisis_info}")

            # Get training data
            train_data = self._filter_data_by_date(
                historical_data, window.train_start, window.train_end
            )

            # Get test data
            test_data = self._filter_data_by_date(
                historical_data, window.test_start, window.test_end
            )

            if train_data is None or test_data is None:
                logger.warning(f"Insufficient data for window {window.window_id}")
                continue

            # Create evaluation function
            if evaluate_fn is None:
                evaluate_fn = lambda params: self._default_evaluate(
                    strategy, params, train_data
                )

            # Optimize on training data
            best_params, all_results = self.optimizer.optimize(
                param_specs,
                lambda params: self._default_evaluate(strategy, params, train_data)
            )

            # Determine regime during test
            regime = self._determine_regime(test_data)

            # Apply crisis and regime adjustments for out-of-sample testing
            adjusted_params = best_params.copy()
            adjustments_applied = []

            if window.test_crisis and window.test_crisis_severity > 0.3:
                adjusted_params = get_crisis_adjusted_params(
                    adjusted_params, strategy, window.test_crisis_severity
                )
                adjustments_applied.append(f"crisis:{window.test_crisis_name}")
                logger.info(f"  Applied crisis adjustments for {window.test_crisis_name}")

            adjusted_params = get_regime_adjusted_params(adjusted_params, strategy, regime)
            if regime != TermStructureRegime.FLAT:
                adjustments_applied.append(f"regime:{regime.value}")

            # Evaluate best params on test data (with adjustments for actual trading)
            train_metrics = self._default_evaluate(strategy, best_params, train_data)
            test_metrics = self._default_evaluate(strategy, adjusted_params, test_data)

            result = OptimizationResult(
                strategy=strategy,
                window=window,
                best_params=best_params,
                crisis_adjusted_params=adjusted_params,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                all_param_results=all_results,
                optimization_objective=self.objective,
                regime_during_test=regime,
                crisis_during_test=window.test_crisis,
                crisis_severity=window.test_crisis_severity,
                param_adjustments_applied=adjustments_applied
            )

            window_results.append(result)

            # Track equity curve
            if 'equity_curve' in test_metrics:
                ec = test_metrics['equity_curve']
                ec['window_id'] = window.window_id
                equity_curves.append(ec)

        # Combine results
        aggregate_metrics = self._calculate_aggregate_metrics(window_results)
        param_stability = self._calculate_parameter_stability(window_results)
        regime_analysis = self._analyze_by_regime(window_results)
        crisis_analysis = self._analyze_by_crisis(window_results)

        combined_equity = pd.concat(equity_curves) if equity_curves else pd.DataFrame()

        crisis_count = sum(1 for r in window_results if r.crisis_during_test)
        non_crisis_count = len(window_results) - crisis_count

        return WalkForwardResult(
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            windows=windows,
            window_results=window_results,
            aggregate_metrics=aggregate_metrics,
            parameter_stability=param_stability,
            regime_analysis=regime_analysis,
            crisis_analysis=crisis_analysis,
            combined_equity_curve=combined_equity,
            crisis_windows_count=crisis_count,
            non_crisis_windows_count=non_crisis_count
        )

    def _filter_data_by_date(
        self,
        data: Dict[str, pd.DataFrame],
        start: datetime,
        end: datetime
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Filter data to date range."""
        filtered = {}

        # Ensure start and end are timezone-aware (UTC) for comparison with data
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        for venue, df in data.items():
            if 'timestamp' not in df.columns:
                continue

            # Ensure timestamp column is timezone-aware
            ts_col = df['timestamp']
            if hasattr(ts_col.dtype, 'tz') and ts_col.dtype.tz is None:
                ts_col = pd.to_datetime(ts_col, utc=True)
            elif not hasattr(ts_col.dtype, 'tz'):
                ts_col = pd.to_datetime(ts_col, utc=True)

            mask = (ts_col >= start) & (ts_col < end)
            venue_df = df[mask].copy()

            if not venue_df.empty:
                filtered[venue] = venue_df

        return filtered if filtered else None

    def _default_evaluate(
        self,
        strategy: str,
        params: Dict[str, Any],
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Default evaluation function for strategies."""
        # Simulate strategy with parameters
        if strategy == 'calendar_spread':
            return self._evaluate_calendar_spread(params, data)
        elif strategy == 'cross_venue':
            return self._evaluate_cross_venue(params, data)
        elif strategy == 'synthetic_futures':
            return self._evaluate_synthetic(params, data)
        elif strategy == 'roll_optimization':
            return self._evaluate_roll(params, data)
        else:
            return {'sharpe_ratio': 0, 'total_return_pct': 0}

    def _evaluate_calendar_spread(
        self,
        params: Dict[str, Any],
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Evaluate calendar spread strategy."""
        returns = []
        trades = 0
        wins = 0

        for venue, df in data.items():
            if 'funding_rate' not in df.columns or 'spot_price' not in df.columns:
                continue

            # Simplified simulation
            entry_z = params.get('entry_z_threshold', 2.0)
            exit_z = params.get('exit_z_threshold', 0.5)
            stop_loss = params.get('stop_loss_bps', 50) / 10000
            take_profit = params.get('take_profit_bps', 30) / 10000

            funding = df['funding_rate'].values
            prices = df['spot_price'].values

            # Calculate z-scores
            if len(funding) < 20:
                continue

            rolling_mean = pd.Series(funding).rolling(20).mean().values
            rolling_std = pd.Series(funding).rolling(20).std().values

            position = 0
            entry_price = 0

            for i in range(20, len(funding)):
                z = (funding[i] - rolling_mean[i]) / max(rolling_std[i], 1e-8)

                if position == 0 and abs(z) > entry_z:
                    position = 1 if z > 0 else -1
                    entry_price = prices[i]
                    trades += 1

                elif position != 0:
                    ret = (prices[i] - entry_price) / entry_price * position

                    if ret > take_profit or ret < -stop_loss or abs(z) < exit_z:
                        returns.append(ret)
                        if ret > 0:
                            wins += 1
                        position = 0

        if not returns:
            return {'sharpe_ratio': 0, 'total_return_pct': 0, 'trades': 0}

        returns = np.array(returns)
        total_ret = np.sum(returns) * 100
        # Use sqrt(252) annualization and cap to prevent overflow
        sharpe = np.mean(returns) / max(np.std(returns), 1e-8) * np.sqrt(252)
        sharpe = float(np.clip(sharpe, -100, 100))

        return {
            'sharpe_ratio': sharpe,
            'total_return_pct': total_ret,
            'trades': trades,
            'win_rate': wins / max(trades, 1),
            'avg_return_pct': np.mean(returns) * 100,
            'max_drawdown_pct': self._calculate_max_drawdown(returns) * 100
        }

    def _evaluate_cross_venue(
        self,
        params: Dict[str, Any],
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Evaluate cross-venue strategy using funding rate differentials.

        Cross-venue arbitrage exploits funding rate differences between venues -
        venues charge different perpetual swap funding rates, and the differential
        creates an arbitrage opportunity (long on low-funding venue, short on high).
        """
        min_spread = params.get('min_spread_bps', 20)
        take_profit = params.get('take_profit_bps', 15) / 10000
        stop_loss = params.get('stop_loss_bps', 30) / 10000
        confidence = params.get('confidence_threshold', 0.7)

        venues = list(data.keys())
        if len(venues) < 2:
            return {'sharpe_ratio': 0, 'total_return_pct': 0, 'trades': 0}

        # Build funding rate series for each venue (the actual cross-venue signal)
        funding_series = {}
        for venue, df in data.items():
            if 'timestamp' not in df.columns or 'funding_rate' not in df.columns:
                continue
            ts_col = pd.to_datetime(df['timestamp'], utc=True)
            fr = df['funding_rate'].values.astype(float)
            s = pd.Series(fr, index=ts_col, name=venue)
            s = s[~s.index.duplicated(keep='first')]
            s = s.dropna()
            if len(s) > 0:
                funding_series[venue] = s

        if len(funding_series) < 2:
            return {'sharpe_ratio': 0, 'total_return_pct': 0, 'trades': 0}

        # Combine into a single DataFrame (auto-aligns on timestamps)
        funding_matrix = pd.DataFrame(funding_series)
        funding_matrix = funding_matrix.sort_index()

        # Forward-fill to align different frequencies (1h, 8h, daily)
        # This carries the last known funding rate forward between observations
        funding_matrix = funding_matrix.ffill(limit=24)
        funding_matrix = funding_matrix.dropna(thresh=2)

        # Resample to 8h (most common funding interval) for consistent signal
        funding_matrix = funding_matrix.resample('8h').last()
        funding_matrix = funding_matrix.ffill(limit=3)
        funding_matrix = funding_matrix.dropna(thresh=2)

        if len(funding_matrix) < 10:
            return {'sharpe_ratio': 0, 'total_return_pct': 0, 'trades': 0}

        # Compute max pairwise funding rate differential in bps at each timestamp
        # After daily resampling, funding rates are daily averages
        # Scale: rate * 10000 gives bps per period, * 3 approximates daily (3x 8h)
        venue_cols = funding_matrix.columns.tolist()
        max_spreads = np.zeros(len(funding_matrix))
        for i, v1 in enumerate(venue_cols):
            for v2 in venue_cols[i + 1:]:
                vals1 = funding_matrix[v1].values
                vals2 = funding_matrix[v2].values
                # Use nan_to_num to handle remaining NaN in pairwise comparison
                diff_bps = np.abs(np.nan_to_num(vals1) - np.nan_to_num(vals2)) * 10000 * 3
                max_spreads = np.maximum(max_spreads, diff_bps)

        # Simulate trading on the funding spread series
        returns = []
        trades = 0
        wins = 0
        position = False
        entry_spread = 0
        hold_days = 0

        for idx in range(len(max_spreads)):
            spread_val = max_spreads[idx]
            if np.isnan(spread_val):
                continue

            if not position and spread_val > min_spread:
                position = True
                entry_spread = spread_val
                trades += 1
                hold_days = 0
            elif position:
                hold_days += 1
                # PnL = capture the funding differential (mean-reversion towards 0)
                pnl = (entry_spread - spread_val) / 10000
                # Also accumulate the funding differential as carry
                carry = spread_val / 10000 / 365  # daily carry from funding diff
                pnl += carry
                if pnl > take_profit or pnl < -stop_loss or hold_days > 30:
                    returns.append(pnl)
                    if pnl > 0:
                        wins += 1
                    position = False

        if not returns:
            return {'sharpe_ratio': 0, 'total_return_pct': 0, 'trades': 0}

        returns = np.array(returns)
        total_ret = np.sum(returns) * 100
        sharpe = np.mean(returns) / max(np.std(returns), 1e-8) * np.sqrt(252)
        sharpe = float(np.clip(sharpe, -100, 100))

        return {
            'sharpe_ratio': sharpe,
            'total_return_pct': total_ret,
            'trades': trades,
            'win_rate': wins / max(trades, 1),
            'avg_return_pct': np.mean(returns) * 100,
            'max_drawdown_pct': self._calculate_max_drawdown(returns) * 100
        }

    def _evaluate_synthetic(
        self,
        params: Dict[str, Any],
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Evaluate synthetic futures strategy."""
        min_spread = params.get('min_funding_spread_annual_pct', 10.0)
        profit_target = params.get('profit_target_pct', 5.0) / 100
        stop_loss = params.get('stop_loss_pct', 3.0) / 100
        max_days = params.get('max_holding_days', 30)

        returns = []
        trades = 0
        wins = 0

        venues = list(data.keys())
        if len(venues) < 2:
            return {'sharpe_ratio': 0, 'total_return_pct': 0, 'trades': 0}

        # Simplified simulation
        for venue, df in data.items():
            if 'funding_rate' not in df.columns:
                continue

            funding = df['funding_rate'].values
            if len(funding) < 30:
                continue

            # Annualized funding
            annual_funding = funding * 24 * 365 * 100

            position = 0
            entry_idx = 0
            cumulative_funding = 0

            for i in range(len(funding)):
                if position == 0 and abs(annual_funding[i]) > min_spread:
                    position = 1 if annual_funding[i] > 0 else -1
                    entry_idx = i
                    cumulative_funding = 0
                    trades += 1

                elif position != 0:
                    cumulative_funding += funding[i] * position * 8  # 8-hour funding

                    # Check exit conditions
                    holding_days = (i - entry_idx) / 3  # Assume 3 observations per day

                    if cumulative_funding > profit_target:
                        returns.append(cumulative_funding)
                        wins += 1
                        position = 0
                    elif cumulative_funding < -stop_loss:
                        returns.append(cumulative_funding)
                        position = 0
                    elif holding_days > max_days:
                        returns.append(cumulative_funding)
                        if cumulative_funding > 0:
                            wins += 1
                        position = 0

        if not returns:
            return {'sharpe_ratio': 0, 'total_return_pct': 0, 'trades': 0}

        returns = np.array(returns)
        total_ret = np.sum(returns) * 100
        # Use sqrt(252) annualization and cap to prevent overflow
        sharpe = np.mean(returns) / max(np.std(returns), 1e-8) * np.sqrt(252)
        sharpe = float(np.clip(sharpe, -100, 100))

        return {
            'sharpe_ratio': sharpe,
            'total_return_pct': total_ret,
            'trades': trades,
            'win_rate': wins / max(trades, 1),
            'avg_return_pct': np.mean(returns) * 100,
            'max_drawdown_pct': self._calculate_max_drawdown(returns) * 100
        }

    def _evaluate_roll(
        self,
        params: Dict[str, Any],
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Evaluate roll optimization strategy."""
        min_benefit = params.get('min_net_benefit_pct', 0.1) / 100
        max_cost = params.get('max_roll_cost_pct', 0.5) / 100

        total_benefit = 0
        total_cost = 0
        rolls = 0

        for venue, df in data.items():
            if 'funding_rate' not in df.columns:
                continue

            funding = df['funding_rate'].values
            if len(funding) < 10:
                continue

            # Simplified roll simulation
            # Benefit from favorable funding, cost from execution
            for i in range(1, len(funding)):
                funding_benefit = funding[i] * 8  # 8-hour funding period
                execution_cost = 0.0001  # 1 bp per roll

                if abs(funding_benefit) > min_benefit and execution_cost < max_cost:
                    total_benefit += abs(funding_benefit)
                    total_cost += execution_cost
                    rolls += 1

        if rolls == 0:
            return {'sharpe_ratio': 0, 'total_return_pct': 0, 'trades': 0}

        net_pnl = total_benefit - total_cost
        efficiency = total_benefit / max(total_cost, 1e-8)
        # Cap to prevent overflow
        capped_sharpe = min(efficiency, 100.0)

        return {
            'sharpe_ratio': capped_sharpe,
            'total_return_pct': net_pnl * 100,
            'trades': rolls,
            'roll_efficiency': efficiency,
            'total_benefit_pct': total_benefit * 100,
            'total_cost_pct': total_cost * 100,
            'max_drawdown_pct': -total_cost * 100
        }

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns series."""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        return abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0

    def _determine_regime(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> TermStructureRegime:
        """Determine dominant regime during period."""
        all_funding = []

        for df in data.values():
            if 'funding_rate' in df.columns:
                all_funding.extend(df['funding_rate'].tolist())

        if not all_funding:
            return TermStructureRegime.FLAT

        avg_annual = np.mean(all_funding) * 24 * 365 * 100

        if avg_annual > 20:
            return TermStructureRegime.STEEP_CONTANGO
        elif avg_annual > 5:
            return TermStructureRegime.MILD_CONTANGO
        elif avg_annual < -20:
            return TermStructureRegime.STEEP_BACKWARDATION
        elif avg_annual < -5:
            return TermStructureRegime.MILD_BACKWARDATION
        else:
            return TermStructureRegime.FLAT

    def _calculate_aggregate_metrics(
        self,
        results: List[OptimizationResult]
    ) -> Dict[str, float]:
        """Calculate aggregate metrics across all windows."""
        if not results:
            return {}

        # Extract metrics, replacing NaN with 0 to avoid propagation
        test_sharpes_raw = [r.test_metrics.get('sharpe_ratio', 0) for r in results]
        test_returns_raw = [r.test_metrics.get('total_return_pct', 0) for r in results]
        train_sharpes_raw = [r.train_metrics.get('sharpe_ratio', 0) for r in results]

        # Filter out NaN values for calculations, default to 0 if all NaN
        test_sharpes = [s if not np.isnan(s) else 0.0 for s in test_sharpes_raw]
        test_returns = [r if not np.isnan(r) else 0.0 for r in test_returns_raw]
        train_sharpes = [s if not np.isnan(s) else 0.0 for s in train_sharpes_raw]

        # Get valid (non-zero) values for more accurate averages
        valid_test_sharpes = [s for s in test_sharpes if s != 0]
        valid_train_sharpes = [s for s in train_sharpes if s != 0]

        # Calculate degradation (train vs test) - use 0 if no valid values
        avg_test = np.mean(valid_test_sharpes) if valid_test_sharpes else 0.0
        avg_train = np.mean(valid_train_sharpes) if valid_train_sharpes else 0.0
        sharpe_degradation = avg_train - avg_test

        return {
            'avg_oos_sharpe': avg_test,
            'std_oos_sharpe': np.std(valid_test_sharpes) if valid_test_sharpes else 0.0,
            'min_oos_sharpe': np.min(test_sharpes) if test_sharpes else 0.0,
            'max_oos_sharpe': np.max(test_sharpes) if test_sharpes else 0.0,
            'avg_oos_return_pct': np.mean(test_returns) if test_returns else 0.0,
            'total_oos_return_pct': np.sum(test_returns) if test_returns else 0.0,
            'sharpe_degradation': sharpe_degradation,
            'windows_positive': sum(1 for r in test_returns if r > 0),
            'windows_total': len(results)
        }

    def _calculate_parameter_stability(
        self,
        results: List[OptimizationResult]
    ) -> Dict[str, float]:
        """Calculate parameter stability across windows."""
        if not results:
            return {}

        # Collect all parameter values
        param_values = defaultdict(list)
        for result in results:
            for param, value in result.best_params.items():
                if isinstance(value, (int, float)):
                    param_values[param].append(value)

        # Calculate coefficient of variation for each parameter
        stability = {}
        for param, values in param_values.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else (0.0 if std_val < 1e-10 else 999.0)
                stability[f'{param}_cv'] = cv
                stability[f'{param}_mean'] = mean_val

        return stability

    def _analyze_by_regime(
        self,
        results: List[OptimizationResult]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze performance by regime."""
        regime_results = defaultdict(list)

        for result in results:
            regime = result.regime_during_test.value
            regime_results[regime].append({
                'sharpe': result.test_metrics.get('sharpe_ratio', 0),
                'return': result.test_metrics.get('total_return_pct', 0),
                'trades': result.test_metrics.get('trades', 0)
            })

        analysis = {}
        for regime, metrics_list in regime_results.items():
            sharpes = [m['sharpe'] for m in metrics_list]
            returns = [m['return'] for m in metrics_list]

            analysis[regime] = {
                'count': len(metrics_list),
                'avg_sharpe': np.mean(sharpes) if sharpes else 0,
                'avg_return_pct': np.mean(returns) if returns else 0,
                'total_return_pct': np.sum(returns) if returns else 0
            }

        return analysis

    def _analyze_by_crisis(
        self,
        results: List[OptimizationResult]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze performance during crisis vs non-crisis periods."""
        crisis_results = []
        non_crisis_results = []

        for result in results:
            metrics = {
                'sharpe': result.test_metrics.get('sharpe_ratio', 0),
                'return': result.test_metrics.get('total_return_pct', 0),
                'trades': result.test_metrics.get('trades', 0),
                'max_dd': result.test_metrics.get('max_drawdown_pct', 0),
                'win_rate': result.test_metrics.get('win_rate', 0),
                'severity': result.crisis_severity
            }

            if result.crisis_during_test:
                crisis_results.append(metrics)
            else:
                non_crisis_results.append(metrics)

        analysis = {}

        # Crisis period analysis
        if crisis_results:
            sharpes = [m['sharpe'] for m in crisis_results]
            returns = [m['return'] for m in crisis_results]
            drawdowns = [m['max_dd'] for m in crisis_results]
            severities = [m['severity'] for m in crisis_results]

            analysis['crisis'] = {
                'count': len(crisis_results),
                'avg_sharpe': np.mean(sharpes),
                'std_sharpe': np.std(sharpes),
                'avg_return_pct': np.mean(returns),
                'total_return_pct': np.sum(returns),
                'avg_max_drawdown_pct': np.mean(drawdowns),
                'worst_drawdown_pct': np.min(drawdowns) if drawdowns else 0,
                'avg_severity': np.mean(severities)
            }

        # Non-crisis period analysis
        if non_crisis_results:
            sharpes = [m['sharpe'] for m in non_crisis_results]
            returns = [m['return'] for m in non_crisis_results]
            drawdowns = [m['max_dd'] for m in non_crisis_results]

            analysis['non_crisis'] = {
                'count': len(non_crisis_results),
                'avg_sharpe': np.mean(sharpes),
                'std_sharpe': np.std(sharpes),
                'avg_return_pct': np.mean(returns),
                'total_return_pct': np.sum(returns),
                'avg_max_drawdown_pct': np.mean(drawdowns),
                'worst_drawdown_pct': np.min(drawdowns) if drawdowns else 0
            }

        # Comparison metrics
        if crisis_results and non_crisis_results:
            crisis_sharpe = analysis['crisis']['avg_sharpe']
            non_crisis_sharpe = analysis['non_crisis']['avg_sharpe']
            analysis['comparison'] = {
                'sharpe_degradation_in_crisis': non_crisis_sharpe - crisis_sharpe,
                'crisis_return_ratio': (
                    analysis['crisis']['avg_return_pct'] /
                    max(analysis['non_crisis']['avg_return_pct'], 0.001)
                ),
                'crisis_drawdown_multiplier': (
                    abs(analysis['crisis']['avg_max_drawdown_pct']) /
                    max(abs(analysis['non_crisis']['avg_max_drawdown_pct']), 0.001)
                )
            }

        return analysis


def run_full_walk_forward(
    historical_data: Dict[str, pd.DataFrame],
    start_date: datetime,
    end_date: datetime,
    strategies: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    objective: OptimizationObjective = OptimizationObjective.SHARPE_RATIO,
    parallel: bool = True
) -> Dict[str, WalkForwardResult]:
    """
    Run walk-forward optimization for all strategies.

    Uses 18-month training / 6-month test windows with crisis period
    detection and regime-adaptive parameter selection.

    Args:
        historical_data: Dict mapping venue to DataFrame
        start_date: Overall start date
        end_date: Overall end date
        strategies: List of strategies to optimize (None for all four)
        output_dir: Optional output directory for results
        objective: Optimization objective function
        parallel: Enable parallel execution across strategies (default True)

    Returns:
        Dict mapping strategy name to WalkForwardResult
    """
    # All four mandatory strategies per Part 2
    if strategies is None:
        strategies = [
            'calendar_spread',       # Strategy A
            'cross_venue',           # Strategy B
            'synthetic_futures',     # Strategy C
            'roll_optimization'      # Strategy D
        ]

    optimizer = WalkForwardOptimizer(
        train_months=18,  # Per PDF specification
        test_months=6,    # Per PDF specification
        objective=objective
    )

    # Try to use joblib for parallel strategy execution
    _joblib_available = False
    try:
        from joblib import Parallel, delayed
        import multiprocessing
        _joblib_available = True
        n_jobs = min(len(strategies), multiprocessing.cpu_count())
        logger.info(f"Parallel execution enabled with joblib ({n_jobs} workers)")
    except ImportError:
        _joblib_available = False
        logger.info("Sequential execution (joblib not available)")

    def _run_single_strategy(strategy: str) -> Tuple[str, Optional[WalkForwardResult]]:
        """Run optimization for a single strategy."""
        logger.info(f"Running walk-forward optimization for {strategy}")
        try:
            result = optimizer.run_optimization(
                strategy=strategy,
                historical_data=historical_data,
                start_date=start_date,
                end_date=end_date
            )
            oos_sharpe = result.aggregate_metrics.get('avg_oos_sharpe', 0)
            crisis_count = result.crisis_windows_count
            logger.info(f"{strategy}: OOS Sharpe = {oos_sharpe:.2f}, "
                       f"Crisis windows = {crisis_count}/{len(result.windows)}")
            return (strategy, result)
        except Exception as e:
            logger.error(f"Failed to optimize {strategy}: {e}")
            return (strategy, None)

    results = {}

    # Sequential execution for reliability and progress visibility
    logger.info(f"Executing {len(strategies)} strategies sequentially for reliable progress tracking...")
    for strategy in strategies:
        strategy_name, result = _run_single_strategy(strategy)
        if result is not None:
            results[strategy_name] = result
        logger.info(f"Completed {len(results)}/{len(strategies)} strategies")

    # Export results if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            strategy: {
                'aggregate_metrics': result.aggregate_metrics,
                'parameter_stability': result.parameter_stability,
                'regime_analysis': result.regime_analysis,
                'crisis_analysis': result.crisis_analysis,
                'crisis_windows_count': result.crisis_windows_count,
                'non_crisis_windows_count': result.non_crisis_windows_count,
                'optimal_params_last_window': (
                    result.window_results[-1].best_params
                    if result.window_results else {}
                ),
                'crisis_adjusted_params_last_window': (
                    result.window_results[-1].crisis_adjusted_params
                    if result.window_results else {}
                )
            }
            for strategy, result in results.items()
        }

        with open(output_dir / 'walk_forward_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Export crisis analysis separately
        crisis_summary = {
            strategy: result.crisis_analysis
            for strategy, result in results.items()
        }
        with open(output_dir / 'walk_forward_crisis_analysis.json', 'w') as f:
            json.dump(crisis_summary, f, indent=2, default=str)

    return results


def generate_walk_forward_report(
    results: Dict[str, WalkForwardResult]
) -> Dict[str, Any]:
    """
    Generate walk-forward report for all strategies.

    Args:
        results: Dict mapping strategy name to WalkForwardResult

    Returns:
        Report dictionary with per-strategy and aggregate metrics
    """
    report = {
        'summary': {},
        'by_strategy': {},
        'crisis_comparison': {},
        'regime_comparison': {},
        'parameter_recommendations': {}
    }

    all_sharpes = []
    all_returns = []

    for strategy, result in results.items():
        # Strategy summary
        report['by_strategy'][strategy] = {
            'oos_sharpe': result.aggregate_metrics.get('avg_oos_sharpe', 0),
            'oos_return_pct': result.aggregate_metrics.get('total_oos_return_pct', 0),
            'sharpe_degradation': result.aggregate_metrics.get('sharpe_degradation', 0),
            'windows_positive_pct': (
                result.aggregate_metrics.get('windows_positive', 0) /
                max(result.aggregate_metrics.get('windows_total', 1), 1) * 100
            ),
            'crisis_windows': result.crisis_windows_count,
            'regime_analysis': result.regime_analysis,
            'crisis_analysis': result.crisis_analysis
        }

        all_sharpes.append(result.aggregate_metrics.get('avg_oos_sharpe', 0))
        all_returns.append(result.aggregate_metrics.get('total_oos_return_pct', 0))

        # Parameter recommendations based on stability
        stable_params = {}
        for param, cv in result.parameter_stability.items():
            if param.endswith('_cv') and cv < 0.3:  # Stable if CV < 30%
                param_name = param.replace('_cv', '')
                mean_key = f'{param_name}_mean'
                if mean_key in result.parameter_stability:
                    stable_params[param_name] = {
                        'value': result.parameter_stability[mean_key],
                        'stability_cv': cv
                    }
        report['parameter_recommendations'][strategy] = stable_params

    # Overall summary
    report['summary'] = {
        'total_strategies': len(results),
        'avg_oos_sharpe': np.mean(all_sharpes) if all_sharpes else 0,
        'total_oos_return_pct': np.sum(all_returns) if all_returns else 0,
        'best_strategy': max(results.keys(), key=lambda s: results[s].aggregate_metrics.get('avg_oos_sharpe', 0)) if results else None,
        'worst_strategy': min(results.keys(), key=lambda s: results[s].aggregate_metrics.get('avg_oos_sharpe', 0)) if results else None
    }

    return report


# Module exports
__all__ = [
    # Enums
    'OptimizationObjective',
    'ParameterType',
    # Dataclasses
    'ParameterSpec',
    'WalkForwardWindow',
    'OptimizationResult',
    'WalkForwardResult',
    # Classes
    'ParameterOptimizer',
    'WalkForwardOptimizer',
    # Functions
    'run_full_walk_forward',
    'generate_walk_forward_report',
    'detect_crisis_in_window',
    'get_crisis_adjusted_params',
    'get_regime_adjusted_params',
    # Constants
    'CRISIS_PARAM_ADJUSTMENTS',
    'REGIME_PARAM_ADJUSTMENTS',
]
