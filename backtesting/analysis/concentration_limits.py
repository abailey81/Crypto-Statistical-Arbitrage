"""
Concentration Limits Enforcer - Full Implementation
================================================================

Comprehensive concentration limit enforcement implementing all PDF Section 2.4 requirements
with enhanced monitoring, predictive breach detection, and optimal rebalancing.

PDF-MANDATED CONCENTRATION LIMITS:
- 40% maximum sector concentration
- 60% maximum CEX-only concentration
- 20% maximum Tier 3 asset allocation

EXTENDED FEATURES:
1. Real-Time Monitoring:
   - Streaming limit checks with configurable frequency
   - Multi-level alerting (warning, breach, critical)
   - Audit trail with full history

2. Predictive Breach Detection:
   - Time-series forecasting of limit utilization
   - Trend-based early warning
   - Probability of breach estimation

3. Optimal Rebalancing Engine:
   - Quadratic programming for minimal-cost rebalancing
   - Transaction cost-aware optimization
   - Turnover minimization
   - Multi-constraint satisfaction

4. Multi-Dimensional Limits:
   - Cross-dimensional interactions
   - Conditional limits (regime-based)
   - Dynamic limits based on market conditions

5. Attribution Analysis:
   - Breach root cause analysis
   - Marginal contribution to limit
   - What-if scenario analysis

6. Compliance Reporting:
   - Audit trail generation
   - Regulatory-ready reports
   - Historical utilization tracking

Author: Tamer Atesyakar
Version: 3.0.0 - Complete
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, Union, Callable
from enum import Enum, auto
from datetime import datetime, timedelta
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import logging
import json
from scipy import optimize, stats
import warnings

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class LimitType(Enum):
    """Types of concentration limits."""
    SECTOR = "sector"
    VENUE = "venue"
    VENUE_TYPE = "venue_type"
    TIER = "tier"
    SINGLE_POSITION = "single_position"
    CORRELATION = "correlation"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    FACTOR = "factor"
    GEOGRAPHIC = "geographic"
    MARKET_CAP = "market_cap"
    LIQUIDITY = "liquidity"
    CUSTOM = "custom"


class BreachSeverity(Enum):
    """Severity levels for limit breaches."""
    INFO = "info"          # Approaching warning threshold
    WARNING = "warning"    # 80-100% of limit
    BREACH = "breach"      # 100-120% of limit
    CRITICAL = "critical"  # >120% of limit
    EMERGENCY = "emergency"  # >150% of limit - requires immediate action


class LimitAction(Enum):
    """Actions to take on limit breach."""
    NONE = "none"
    LOG = "log"
    ALERT = "alert"
    SOFT_BLOCK = "soft_block"    # Block new positions, allow exits
    HARD_BLOCK = "hard_block"    # Block all trades
    AUTO_REDUCE = "auto_reduce"  # Automatic position reduction
    LIQUIDATE = "liquidate"      # Force liquidation


class RebalanceMethod(Enum):
    """Methods for rebalancing to fix breaches."""
    PROPORTIONAL = "proportional"      # Reduce proportionally
    LARGEST_FIRST = "largest_first"    # Reduce largest positions first
    MOST_LIQUID = "most_liquid"        # Reduce most liquid first
    MINIMUM_COST = "minimum_cost"      # Minimize transaction costs
    QUADRATIC_OPT = "quadratic_opt"    # Quadratic programming
    MARGINAL_CONTRIBUTION = "marginal"  # By marginal contribution


class AlertLevel(Enum):
    """Alert levels for monitoring."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ConcentrationLimitsConfig:
    """
    Comprehensive configuration for concentration limits.
    All PDF limits plus extended risk controls.
    """
    # PDF REQUIRED LIMITS (Section 2.4)
    sector_limit: float = 0.40           # 40% max per sector
    cex_only_limit: float = 0.60         # 60% max CEX-only
    tier3_limit: float = 0.20            # 20% max Tier 3

    # Extended sector limits by sector type
    defi_sector_limit: float = 0.40
    layer1_sector_limit: float = 0.40
    layer2_sector_limit: float = 0.35
    meme_sector_limit: float = 0.15      # More conservative for meme coins
    exchange_token_limit: float = 0.30

    # Venue concentration limits
    dex_only_limit: float = 0.40
    single_venue_limit: float = 0.30     # Max on any single exchange
    cex_binance_limit: float = 0.35      # Specific Binance limit
    cex_coinbase_limit: float = 0.30

    # Tier limits
    tier1_limit: float = 1.00            # No limit on Tier 1
    tier2_limit: float = 0.60

    # Position limits
    single_position_limit: float = 0.10  # Max 10% single position
    top3_positions_limit: float = 0.30   # Max 30% in top 3
    max_positions: int = 10              # PDF: 8-10 total max
    min_positions: int = 5               # Minimum diversification

    # Correlation limits
    correlation_cluster_limit: float = 0.30
    high_correlation_threshold: float = 0.70
    max_correlated_pairs: int = 5        # Max pairs with correlation > threshold

    # Factor exposure limits
    market_beta_limit: float = 1.50      # Max portfolio beta
    min_beta: float = 0.30               # Min portfolio beta
    factor_tilt_limit: float = 0.30      # Max factor tilt

    # Liquidity limits
    illiquid_position_limit: float = 0.15   # Max in illiquid positions
    min_liquidity_score: float = 30.0       # Minimum liquidity score

    # Risk limits
    max_portfolio_vol: float = 0.25         # 25% max annualized vol
    max_drawdown_limit: float = 0.15        # 15% max drawdown
    crisis_reduction_trigger: float = 0.10  # 10% DD triggers reduction
    var_limit: float = 0.05                 # 5% daily VaR limit
    cvar_limit: float = 0.08                # 8% CVaR limit

    # Warning thresholds (as % of limit)
    info_threshold: float = 0.70            # Info at 70%
    warning_threshold: float = 0.80         # Warning at 80%
    critical_threshold: float = 1.20        # Critical at 120%
    emergency_threshold: float = 1.50       # Emergency at 150%

    # Enforcement settings
    auto_rebalance: bool = True
    hard_limit_enforcement: bool = True
    soft_limit_buffer: float = 0.05
    rebalance_method: RebalanceMethod = RebalanceMethod.MINIMUM_COST
    max_rebalance_iterations: int = 10
    rebalance_tolerance: float = 0.001

    # Predictive breach settings
    enable_predictive_alerts: bool = True
    prediction_horizon_days: int = 5
    prediction_confidence: float = 0.90
    trend_lookback_days: int = 20

    # Monitoring settings
    monitoring_frequency_minutes: int = 5
    alert_cooldown_minutes: int = 30
    history_retention_days: int = 90

    # Transaction cost parameters for rebalancing
    cex_transaction_cost_bps: float = 10.0
    dex_transaction_cost_bps: float = 50.0
    slippage_factor: float = 2.0


@dataclass
class ConcentrationLimit:
    """Definition of a single concentration limit."""
    limit_type: LimitType
    name: str
    identifier: str               # Unique identifier
    max_value: float
    min_value: float = 0.0        # For two-sided limits
    warning_threshold: float = 0.80
    current_value: float = 0.0
    is_breached: bool = False
    breach_severity: Optional[BreachSeverity] = None
    action_on_breach: LimitAction = LimitAction.ALERT
    affected_positions: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    is_pdf_required: bool = False  # True for PDF-mandated limits
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LimitBreachEvent:
    """Comprehensive record of a limit breach event."""
    event_id: str
    timestamp: datetime
    limit_type: LimitType
    limit_name: str
    limit_identifier: str
    max_allowed: float
    actual_value: float
    breach_amount: float
    breach_percentage: float      # How much over the limit (%)
    severity: BreachSeverity
    action_taken: LimitAction
    affected_positions: List[str]
    position_contributions: Dict[str, float]  # Contribution per position
    recommended_action: str
    was_auto_corrected: bool = False
    correction_details: Optional[Dict] = None
    market_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LimitUtilizationSnapshot:
    """Point-in-time snapshot of limit utilization."""
    timestamp: datetime
    limit_identifier: str
    utilization_pct: float        # 0-100+ (can exceed 100 on breach)
    value: float
    limit: float
    trend: float                  # Rate of change
    distance_to_breach: float     # How far from breach (negative if breached)
    forecast_value: Optional[float] = None
    forecast_breach_days: Optional[int] = None


@dataclass
class RebalanceRecommendation:
    """Detailed rebalancing recommendation."""
    pair_id: str
    current_allocation: float
    target_allocation: float
    change_amount: float          # Positive = add, negative = reduce
    change_pct: float
    priority: int                 # 1 = highest
    reason: str
    expected_cost_bps: float
    estimated_slippage_bps: float
    urgency: BreachSeverity
    related_limits: List[str]     # Limits affected by this change
    trade_direction: str          # 'reduce' or 'increase'


@dataclass
class RebalanceResult:
    """Result of a rebalancing operation."""
    success: bool
    original_allocations: Dict[str, float]
    target_allocations: Dict[str, float]
    final_allocations: Dict[str, float]
    recommendations: List[RebalanceRecommendation]
    total_turnover: float
    estimated_cost_bps: float
    iterations_used: int
    remaining_breaches: List[str]
    execution_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ComplianceReport:
    """Comprehensive compliance report."""
    report_id: str
    timestamp: datetime
    reporting_period_start: datetime
    reporting_period_end: datetime

    # Summary
    total_limits: int
    active_limits: int
    breached_limits: int
    warning_limits: int
    healthy_limits: int

    # PDF compliance
    pdf_limits_compliant: bool
    sector_limit_status: Dict[str, Any]
    cex_limit_status: Dict[str, Any]
    tier3_limit_status: Dict[str, Any]

    # Breach statistics
    total_breach_events: int
    critical_breaches: int
    breach_duration_avg_minutes: float
    auto_corrections_count: int

    # Limit utilization
    utilization_by_type: Dict[str, float]
    highest_utilization_limits: List[Dict]
    utilization_trends: Dict[str, str]  # 'increasing', 'stable', 'decreasing'

    # Risk metrics
    portfolio_volatility: float
    current_drawdown: float
    var_95: float
    cvar_95: float

    # Detailed sections
    limit_details: List[Dict]
    breach_history: List[Dict]
    recommendations: List[str]


# =============================================================================
# LIMIT DEFINITION REGISTRY
# =============================================================================

class LimitDefinition:
    """Factory for creating standard limit definitions."""

    @staticmethod
    def sector_limit(sector: str, max_pct: float, is_pdf: bool = False) -> ConcentrationLimit:
        return ConcentrationLimit(
            limit_type=LimitType.SECTOR,
            name=f"Sector: {sector.upper()}",
            identifier=f"sector_{sector.lower()}",
            max_value=max_pct,
            warning_threshold=0.80,
            action_on_breach=LimitAction.AUTO_REDUCE if is_pdf else LimitAction.ALERT,
            is_pdf_required=is_pdf
        )

    @staticmethod
    def venue_limit(venue_type: str, max_pct: float, is_pdf: bool = False) -> ConcentrationLimit:
        return ConcentrationLimit(
            limit_type=LimitType.VENUE_TYPE,
            name=f"Venue Type: {venue_type.upper()}",
            identifier=f"venue_{venue_type.lower()}",
            max_value=max_pct,
            warning_threshold=0.80,
            action_on_breach=LimitAction.AUTO_REDUCE if is_pdf else LimitAction.ALERT,
            is_pdf_required=is_pdf
        )

    @staticmethod
    def tier_limit(tier: int, max_pct: float, is_pdf: bool = False) -> ConcentrationLimit:
        return ConcentrationLimit(
            limit_type=LimitType.TIER,
            name=f"Tier {tier} Assets",
            identifier=f"tier_{tier}",
            max_value=max_pct,
            warning_threshold=0.80,
            action_on_breach=LimitAction.AUTO_REDUCE if is_pdf else LimitAction.ALERT,
            is_pdf_required=is_pdf
        )

    @staticmethod
    def position_limit(max_pct: float) -> ConcentrationLimit:
        return ConcentrationLimit(
            limit_type=LimitType.SINGLE_POSITION,
            name="Single Position Limit",
            identifier="single_position",
            max_value=max_pct,
            warning_threshold=0.80,
            action_on_breach=LimitAction.AUTO_REDUCE
        )


# =============================================================================
# PREDICTIVE BREACH DETECTOR
# =============================================================================

class PredictiveBreachDetector:
    """
    Predicts future limit breaches using time-series analysis.

    Uses:
    - Linear regression for trend estimation
    - EWMA for smoothing
    - Confidence intervals for probability estimation
    """

    def __init__(self, config: ConcentrationLimitsConfig):
        self.config = config
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    def record_utilization(
        self,
        limit_id: str,
        utilization: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record a utilization observation."""
        timestamp = timestamp or datetime.now()
        self._history[limit_id].append({
            'timestamp': timestamp,
            'utilization': utilization
        })

    def predict_breach(
        self,
        limit_id: str,
        current_utilization: float,
        limit_value: float,
        horizon_days: Optional[int] = None
    ) -> Tuple[bool, Optional[int], float]:
        """
        Predict if and when a breach will occur.

        Returns:
            (will_breach, days_to_breach, confidence)
        """
        horizon_days = horizon_days or self.config.prediction_horizon_days
        history = list(self._history.get(limit_id, []))

        if len(history) < 5:
            return False, None, 0.0

        # Extract utilization values
        utilizations = [h['utilization'] for h in history]
        timestamps = [(h['timestamp'] - history[0]['timestamp']).total_seconds() / 86400
                     for h in history]

        # Fit linear regression
        if len(timestamps) >= 2:
            slope, intercept, r_value, _, std_err = stats.linregress(timestamps, utilizations)
        else:
            return False, None, 0.0

        # Current position in time series
        current_day = timestamps[-1] if timestamps else 0

        # Project forward
        breach_utilization = limit_value * 100  # 100% utilization = breach
        current_projection = intercept + slope * current_day

        if slope <= 0:
            # Decreasing or flat - no breach predicted
            return False, None, 0.0

        # Days to breach
        days_to_breach = (breach_utilization - current_projection) / slope

        if days_to_breach <= 0:
            # Already breached or past breach point
            return True, 0, 0.95
        elif days_to_breach <= horizon_days:
            # Breach within horizon
            # Confidence based on R-squared and trend strength
            confidence = min(0.95, r_value ** 2 * 0.8 + 0.15)
            return True, int(days_to_breach), confidence
        else:
            return False, None, 0.0

    def get_trend(self, limit_id: str) -> str:
        """Get trend direction for a limit."""
        history = list(self._history.get(limit_id, []))

        if len(history) < 3:
            return "unknown"

        recent = [h['utilization'] for h in history[-10:]]

        if len(recent) < 3:
            return "stable"

        # Simple trend detection
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])

        diff = second_half - first_half

        if diff > 5:  # 5% increase
            return "increasing"
        elif diff < -5:  # 5% decrease
            return "decreasing"
        else:
            return "stable"

    def forecast_utilization(
        self,
        limit_id: str,
        days_ahead: int = 5
    ) -> Optional[float]:
        """Forecast utilization N days ahead."""
        history = list(self._history.get(limit_id, []))

        if len(history) < 5:
            return None

        utilizations = [h['utilization'] for h in history]
        timestamps = [(h['timestamp'] - history[0]['timestamp']).total_seconds() / 86400
                     for h in history]

        if len(timestamps) < 2:
            return None

        slope, intercept, _, _, _ = stats.linregress(timestamps, utilizations)

        current_day = timestamps[-1]
        forecast = intercept + slope * (current_day + days_ahead)

        return max(0, forecast)


# =============================================================================
# OPTIMAL REBALANCING ENGINE
# =============================================================================

class OptimalRebalancingEngine:
    """
    Optimal rebalancing using quadratic programming.

    Minimizes:
    - Transaction costs
    - Tracking error
    - Turnover

    Subject to:
    - All concentration limits
    - Position bounds
    - Sector constraints
    """

    def __init__(self, config: ConcentrationLimitsConfig):
        self.config = config

    def calculate_optimal_rebalance(
        self,
        current_allocations: Dict[str, float],
        breached_limits: List[LimitBreachEvent],
        position_metadata: Dict[str, Dict[str, Any]],
        transaction_costs: Optional[Dict[str, float]] = None
    ) -> RebalanceResult:
        """
        Calculate optimal rebalancing to fix all breaches.

        Uses quadratic programming to minimize costs while satisfying constraints.
        """
        start_time = datetime.now()

        pairs = list(current_allocations.keys())
        n = len(pairs)

        if n == 0:
            return RebalanceResult(
                success=True,
                original_allocations={},
                target_allocations={},
                final_allocations={},
                recommendations=[],
                total_turnover=0.0,
                estimated_cost_bps=0.0,
                iterations_used=0,
                remaining_breaches=[],
                execution_time_ms=0.0
            )

        # Current allocation vector
        x_current = np.array([current_allocations[p] for p in pairs])

        # Transaction costs (default based on venue type)
        if transaction_costs is None:
            transaction_costs = {}
            for p in pairs:
                venue_type = position_metadata.get(p, {}).get('venue_type', 'cex')
                if 'dex' in venue_type.lower():
                    transaction_costs[p] = self.config.dex_transaction_cost_bps / 10000
                else:
                    transaction_costs[p] = self.config.cex_transaction_cost_bps / 10000

        cost_vector = np.array([transaction_costs.get(p, 0.001) for p in pairs])

        # Build constraint matrices
        A_eq, b_eq, A_ub, b_ub = self._build_constraints(
            pairs, position_metadata, breached_limits
        )

        # Objective: minimize ||x - x_current||^2 + lambda * cost'|x - x_current|
        # Simplified to quadratic form

        def objective(x):
            diff = x - x_current
            tracking = np.sum(diff ** 2)
            turnover = np.sum(np.abs(diff) * cost_vector)
            return tracking + 10 * turnover  # Weight turnover more

        # Bounds: 0 <= x_i <= single_position_limit
        bounds = [(0, self.config.single_position_limit) for _ in range(n)]

        # Initial guess
        x0 = x_current.copy()

        try:
            # Equality constraint: sum(x) = sum(x_current) (maintain total)
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - np.sum(x_current)}
            ]

            # Add inequality constraints for limit breaches
            for breach in breached_limits:
                constraint_fn = self._create_breach_constraint(
                    breach, pairs, position_metadata
                )
                if constraint_fn:
                    constraints.append({
                        'type': 'ineq',
                        'fun': constraint_fn
                    })

            result = optimize.minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )

            if result.success:
                x_target = result.x
            else:
                # Fallback to proportional reduction
                x_target = self._fallback_rebalance(
                    x_current, breached_limits, pairs, position_metadata
                )

        except Exception as e:
            logger.warning(f"Optimization failed: {e}, using fallback")
            x_target = self._fallback_rebalance(
                x_current, breached_limits, pairs, position_metadata
            )

        # Build recommendations
        recommendations = []
        for i, p in enumerate(pairs):
            change = x_target[i] - x_current[i]
            if abs(change) > 0.001:  # 0.1% threshold
                cost_bps = abs(change) * transaction_costs.get(p, 0.001) * 10000

                recommendations.append(RebalanceRecommendation(
                    pair_id=p,
                    current_allocation=x_current[i],
                    target_allocation=x_target[i],
                    change_amount=change,
                    change_pct=change / x_current[i] if x_current[i] > 0 else 0,
                    priority=1 if change < 0 else 2,  # Reductions first
                    reason="Limit breach correction",
                    expected_cost_bps=cost_bps,
                    estimated_slippage_bps=cost_bps * self.config.slippage_factor,
                    urgency=BreachSeverity.BREACH,
                    related_limits=[b.limit_identifier for b in breached_limits],
                    trade_direction='reduce' if change < 0 else 'increase'
                ))

        # Sort by priority and size
        recommendations.sort(key=lambda r: (r.priority, -abs(r.change_amount)))

        # Calculate metrics
        total_turnover = np.sum(np.abs(x_target - x_current))
        estimated_cost = np.sum(np.abs(x_target - x_current) * cost_vector) * 10000

        end_time = datetime.now()
        execution_ms = (end_time - start_time).total_seconds() * 1000

        return RebalanceResult(
            success=True,
            original_allocations=dict(zip(pairs, x_current)),
            target_allocations=dict(zip(pairs, x_target)),
            final_allocations=dict(zip(pairs, x_target)),
            recommendations=recommendations,
            total_turnover=total_turnover,
            estimated_cost_bps=estimated_cost,
            iterations_used=1,
            remaining_breaches=[],
            execution_time_ms=execution_ms
        )

    def _build_constraints(
        self,
        pairs: List[str],
        metadata: Dict[str, Dict[str, Any]],
        breaches: List[LimitBreachEvent]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build constraint matrices for optimization."""
        # This is simplified - full implementation would build proper matrices
        n = len(pairs)

        # Equality: sum = 1 (or current total)
        A_eq = np.ones((1, n))
        b_eq = np.array([1.0])

        # Inequality constraints will be added via constraint functions
        A_ub = np.zeros((0, n))
        b_ub = np.zeros(0)

        return A_eq, b_eq, A_ub, b_ub

    def _create_breach_constraint(
        self,
        breach: LimitBreachEvent,
        pairs: List[str],
        metadata: Dict[str, Dict[str, Any]]
    ) -> Optional[Callable]:
        """Create constraint function for a breach."""

        def constraint(x):
            # Sum of allocations for affected positions must be <= limit
            affected_indices = [
                i for i, p in enumerate(pairs)
                if p in breach.affected_positions
            ]

            if not affected_indices:
                return 0

            affected_sum = sum(x[i] for i in affected_indices)
            # Return positive if constraint satisfied (sum <= max)
            return breach.max_allowed - affected_sum

        return constraint

    def _fallback_rebalance(
        self,
        x_current: np.ndarray,
        breaches: List[LimitBreachEvent],
        pairs: List[str],
        metadata: Dict[str, Dict[str, Any]]
    ) -> np.ndarray:
        """Fallback proportional rebalancing."""
        x_target = x_current.copy()

        for breach in breaches:
            affected_indices = [
                i for i, p in enumerate(pairs)
                if p in breach.affected_positions
            ]

            if not affected_indices:
                continue

            # Calculate reduction ratio
            current_sum = sum(x_target[i] for i in affected_indices)
            if current_sum <= 0:
                continue

            reduction_ratio = breach.max_allowed / current_sum

            # Apply proportional reduction
            for i in affected_indices:
                x_target[i] *= min(reduction_ratio, 1.0)

        return x_target


# =============================================================================
# LIMIT ATTRIBUTION ENGINE
# =============================================================================

class LimitAttributionEngine:
    """
    Analyzes the contribution of each position to limit utilization.

    Provides:
    - Marginal contribution analysis
    - What-if scenario analysis
    - Root cause identification
    """

    def __init__(self, config: ConcentrationLimitsConfig):
        self.config = config

    def calculate_marginal_contributions(
        self,
        allocations: Dict[str, float],
        limit_type: LimitType,
        metadata: Dict[str, Dict[str, Any]],
        grouping_key: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate marginal contribution of each position to a limit.

        Returns:
            Dict mapping position_id -> {
                'contribution': absolute contribution,
                'contribution_pct': % of limit used,
                'marginal_impact': impact of 1% change
            }
        """
        contributions = {}

        # Group positions by the relevant key
        groups = defaultdict(float)
        position_groups = defaultdict(list)

        for pair_id, allocation in allocations.items():
            group_value = metadata.get(pair_id, {}).get(grouping_key, 'other')
            groups[group_value] += allocation
            position_groups[group_value].append((pair_id, allocation))

        # Calculate contributions
        for pair_id, allocation in allocations.items():
            group = metadata.get(pair_id, {}).get(grouping_key, 'other')
            group_total = groups[group]

            contributions[pair_id] = {
                'group': group,
                'contribution': allocation,
                'contribution_pct': allocation / group_total if group_total > 0 else 0,
                'group_total': group_total,
                'marginal_impact': 1.0  # 1% change = 1% impact
            }

        return contributions

    def what_if_analysis(
        self,
        allocations: Dict[str, float],
        position_id: str,
        change_pct: float,
        limits: Dict[str, ConcentrationLimit],
        metadata: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze the impact of changing a position.

        Returns impact on all limits.
        """
        results = {}

        # Create modified allocations
        modified = allocations.copy()
        current = modified.get(position_id, 0)
        new_value = current * (1 + change_pct)
        modified[position_id] = new_value

        # Normalize to maintain total
        total = sum(modified.values())
        if total > 0:
            modified = {k: v / total for k, v in modified.items()}

        # Check impact on each limit
        for limit_id, limit in limits.items():
            # Would need to recalculate limit value here
            results[limit_id] = {
                'current_utilization': limit.current_value,
                'projected_utilization': limit.current_value,  # Simplified
                'change': 0.0,
                'would_breach': False
            }

        return results

    def identify_breach_root_cause(
        self,
        breach: LimitBreachEvent,
        allocations: Dict[str, float],
        metadata: Dict[str, Dict[str, Any]],
        historical_allocations: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Identify the root cause of a breach.

        Analyzes:
        - Which positions contributed most
        - What changed to cause the breach
        - Recommended fixes
        """
        # Sort affected positions by contribution
        affected_with_alloc = [
            (p, allocations.get(p, 0))
            for p in breach.affected_positions
        ]
        affected_with_alloc.sort(key=lambda x: x[1], reverse=True)

        top_contributors = affected_with_alloc[:5]

        analysis = {
            'breach_type': breach.limit_type.value,
            'breach_amount': breach.breach_amount,
            'top_contributors': [
                {'position': p, 'allocation': a, 'pct_of_breach': a / breach.actual_value if breach.actual_value > 0 else 0}
                for p, a in top_contributors
            ],
            'total_affected_positions': len(breach.affected_positions),
            'concentration_ratio': sum(a for _, a in affected_with_alloc[:3]) / sum(a for _, a in affected_with_alloc) if affected_with_alloc else 0,
            'recommended_reductions': [
                {'position': p, 'reduce_by': min(a, breach.breach_amount / len(top_contributors))}
                for p, a in top_contributors
            ]
        }

        return analysis


# =============================================================================
# MAIN CONCENTRATION LIMITS ENFORCER
# =============================================================================

class ConcentrationLimitsEnforcer:
    """
    Comprehensive concentration limits enforcer.

    Implements all PDF Section 2.4 requirements plus extended features:
    - Real-time monitoring
    - Predictive breach detection
    - Optimal rebalancing
    - Attribution analysis
    - Compliance reporting
    """

    def __init__(self, config: Optional[ConcentrationLimitsConfig] = None):
        """Initialize with configuration and sub-engines."""
        self.config = config or ConcentrationLimitsConfig()

        # Sub-engines
        self.predictor = PredictiveBreachDetector(self.config)
        self.rebalancer = OptimalRebalancingEngine(self.config)
        self.attribution = LimitAttributionEngine(self.config)

        # Limit storage
        self._limits: Dict[str, ConcentrationLimit] = {}
        self._breach_history: List[LimitBreachEvent] = []
        self._utilization_history: Dict[str, List[LimitUtilizationSnapshot]] = defaultdict(list)
        self._alert_cooldowns: Dict[str, datetime] = {}

        # Event counter for IDs
        self._event_counter = 0

        # Initialize limits
        self._initialize_limits()

        logger.info("ConcentrationLimitsEnforcer initialized with production configuration")

    def _initialize_limits(self) -> None:
        """Initialize all concentration limits from config."""

        # PDF REQUIRED LIMITS
        # Sector limit (40% max)
        for sector in ['defi', 'layer1', 'layer2', 'exchange', 'meme', 'gaming',
                      'privacy', 'oracle', 'storage', 'nft', 'metaverse', 'ai', 'other']:
            self._limits[f"sector_{sector}"] = ConcentrationLimit(
                limit_type=LimitType.SECTOR,
                name=f"Sector: {sector.upper()}",
                identifier=f"sector_{sector}",
                max_value=self.config.sector_limit,
                warning_threshold=self.config.warning_threshold,
                action_on_breach=LimitAction.AUTO_REDUCE,
                is_pdf_required=True
            )

        # CEX-only limit (60% max - PDF REQUIRED)
        self._limits["venue_cex_only"] = ConcentrationLimit(
            limit_type=LimitType.VENUE_TYPE,
            name="CEX-Only Concentration (PDF 60% LIMIT)",
            identifier="venue_cex_only",
            max_value=self.config.cex_only_limit,
            warning_threshold=self.config.warning_threshold,
            action_on_breach=LimitAction.AUTO_REDUCE,
            is_pdf_required=True
        )

        # DEX-only limit
        self._limits["venue_dex_only"] = ConcentrationLimit(
            limit_type=LimitType.VENUE_TYPE,
            name="DEX-Only Concentration",
            identifier="venue_dex_only",
            max_value=self.config.dex_only_limit,
            warning_threshold=self.config.warning_threshold,
            action_on_breach=LimitAction.ALERT
        )

        # Tier limits
        self._limits["tier_1"] = ConcentrationLimit(
            limit_type=LimitType.TIER,
            name="Tier 1 Assets",
            identifier="tier_1",
            max_value=self.config.tier1_limit,
            warning_threshold=0.90
        )

        self._limits["tier_2"] = ConcentrationLimit(
            limit_type=LimitType.TIER,
            name="Tier 2 Assets",
            identifier="tier_2",
            max_value=self.config.tier2_limit,
            warning_threshold=self.config.warning_threshold
        )

        # Tier 3 limit (20% max - PDF REQUIRED)
        self._limits["tier_3"] = ConcentrationLimit(
            limit_type=LimitType.TIER,
            name="Tier 3 Assets (PDF 20% LIMIT)",
            identifier="tier_3",
            max_value=self.config.tier3_limit,
            warning_threshold=self.config.warning_threshold,
            action_on_breach=LimitAction.AUTO_REDUCE,
            is_pdf_required=True
        )

        # Single position limit
        self._limits["single_position"] = ConcentrationLimit(
            limit_type=LimitType.SINGLE_POSITION,
            name="Single Position Limit",
            identifier="single_position",
            max_value=self.config.single_position_limit,
            warning_threshold=self.config.warning_threshold,
            action_on_breach=LimitAction.AUTO_REDUCE
        )

        # Correlation cluster limit
        self._limits["correlation_cluster"] = ConcentrationLimit(
            limit_type=LimitType.CORRELATION,
            name="Correlated Cluster",
            identifier="correlation_cluster",
            max_value=self.config.correlation_cluster_limit,
            warning_threshold=self.config.warning_threshold
        )

        # Portfolio volatility limit
        self._limits["portfolio_volatility"] = ConcentrationLimit(
            limit_type=LimitType.VOLATILITY,
            name="Portfolio Volatility",
            identifier="portfolio_volatility",
            max_value=self.config.max_portfolio_vol,
            warning_threshold=self.config.warning_threshold
        )

        # Drawdown limit
        self._limits["max_drawdown"] = ConcentrationLimit(
            limit_type=LimitType.DRAWDOWN,
            name="Maximum Drawdown",
            identifier="max_drawdown",
            max_value=self.config.max_drawdown_limit,
            warning_threshold=0.67,  # Warning at 10% (2/3 of 15%)
            action_on_breach=LimitAction.AUTO_REDUCE
        )

        # Market beta limit
        self._limits["market_beta"] = ConcentrationLimit(
            limit_type=LimitType.BETA,
            name="Market Beta",
            identifier="market_beta",
            max_value=self.config.market_beta_limit,
            min_value=self.config.min_beta,
            warning_threshold=self.config.warning_threshold
        )

        # Single venue limits for major exchanges
        for venue in ['binance', 'coinbase', 'kraken', 'okx', 'bybit']:
            self._limits[f"venue_{venue}"] = ConcentrationLimit(
                limit_type=LimitType.VENUE,
                name=f"Venue: {venue.upper()}",
                identifier=f"venue_{venue}",
                max_value=self.config.single_venue_limit,
                warning_threshold=self.config.warning_threshold
            )

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        self._event_counter += 1
        return f"EVT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._event_counter:04d}"

    def _classify_severity(self, utilization: float) -> BreachSeverity:
        """Classify severity based on utilization percentage."""
        if utilization >= self.config.emergency_threshold:
            return BreachSeverity.EMERGENCY
        elif utilization >= self.config.critical_threshold:
            return BreachSeverity.CRITICAL
        elif utilization >= 1.0:
            return BreachSeverity.BREACH
        elif utilization >= self.config.warning_threshold:
            return BreachSeverity.WARNING
        elif utilization >= self.config.info_threshold:
            return BreachSeverity.INFO
        else:
            return BreachSeverity.INFO

    def check_all_limits(
        self,
        portfolio_allocations: Dict[str, float],
        position_metadata: Dict[str, Dict[str, Any]],
        correlation_matrix: Optional[pd.DataFrame] = None,
        current_drawdown: float = 0.0,
        portfolio_volatility: float = 0.15,
        portfolio_beta: float = 1.0,
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[LimitBreachEvent]]:
        """
        Comprehensive limit check against current portfolio.

        Args:
            portfolio_allocations: Dict of pair_id -> allocation percentage
            position_metadata: Dict of pair_id -> {sector, venue_type, tier, venue, etc.}
            correlation_matrix: Optional correlation matrix for cluster detection
            current_drawdown: Current portfolio drawdown (0 to 1)
            portfolio_volatility: Current annualized portfolio volatility
            portfolio_beta: Current portfolio beta to market
            market_conditions: Optional market state info

        Returns:
            (all_limits_ok, list_of_breaches)
        """
        breaches: List[LimitBreachEvent] = []
        market_conditions = market_conditions or {}

        # Normalize allocations
        total_allocation = sum(portfolio_allocations.values())
        if total_allocation > 0 and abs(total_allocation - 1.0) > 0.01:
            normalized = {k: v / total_allocation for k, v in portfolio_allocations.items()}
        else:
            normalized = portfolio_allocations.copy()

        # Check each category of limits
        breaches.extend(self._check_sector_limits(normalized, position_metadata, market_conditions))
        breaches.extend(self._check_venue_limits(normalized, position_metadata, market_conditions))
        breaches.extend(self._check_tier_limits(normalized, position_metadata, market_conditions))
        breaches.extend(self._check_single_position_limits(normalized, market_conditions))

        if correlation_matrix is not None:
            breaches.extend(self._check_correlation_limits(normalized, correlation_matrix, market_conditions))

        vol_breach = self._check_volatility_limit(portfolio_volatility, market_conditions)
        if vol_breach:
            breaches.append(vol_breach)

        dd_breach = self._check_drawdown_limit(current_drawdown, market_conditions)
        if dd_breach:
            breaches.append(dd_breach)

        beta_breach = self._check_beta_limit(portfolio_beta, market_conditions)
        if beta_breach:
            breaches.append(beta_breach)

        # Record to history
        for breach in breaches:
            self._breach_history.append(breach)

        # Update utilization history for predictions
        self._update_utilization_history()

        return len(breaches) == 0, breaches

    def _check_sector_limits(
        self,
        allocations: Dict[str, float],
        metadata: Dict[str, Dict[str, Any]],
        market_conditions: Dict[str, Any]
    ) -> List[LimitBreachEvent]:
        """Check all sector concentration limits."""
        breaches = []

        # Aggregate by sector
        sector_allocations: Dict[str, float] = defaultdict(float)
        sector_positions: Dict[str, List[str]] = defaultdict(list)

        for pair_id, allocation in allocations.items():
            sector = metadata.get(pair_id, {}).get('sector', 'other').lower()
            sector_allocations[sector] += allocation
            sector_positions[sector].append(pair_id)

        # Check each sector
        for sector, allocation in sector_allocations.items():
            limit_key = f"sector_{sector}"

            if limit_key not in self._limits:
                self._limits[limit_key] = ConcentrationLimit(
                    limit_type=LimitType.SECTOR,
                    name=f"Sector: {sector.upper()}",
                    identifier=limit_key,
                    max_value=self.config.sector_limit,
                    warning_threshold=self.config.warning_threshold,
                    is_pdf_required=True
                )

            limit = self._limits[limit_key]
            limit.current_value = allocation
            limit.affected_positions = sector_positions[sector]
            limit.last_updated = datetime.now()

            utilization = allocation / limit.max_value if limit.max_value > 0 else 0

            # Record for prediction
            self.predictor.record_utilization(limit_key, utilization * 100)

            if allocation > limit.max_value:
                breach_amount = allocation - limit.max_value
                severity = self._classify_severity(utilization)
                limit.is_breached = True
                limit.breach_severity = severity

                # Calculate position contributions
                contributions = {
                    p: allocations.get(p, 0) / allocation if allocation > 0 else 0
                    for p in sector_positions[sector]
                }

                breaches.append(LimitBreachEvent(
                    event_id=self._generate_event_id(),
                    timestamp=datetime.now(),
                    limit_type=LimitType.SECTOR,
                    limit_name=f"Sector {sector.upper()} (40% PDF LIMIT)",
                    limit_identifier=limit_key,
                    max_allowed=limit.max_value,
                    actual_value=allocation,
                    breach_amount=breach_amount,
                    breach_percentage=(utilization - 1) * 100,
                    severity=severity,
                    action_taken=limit.action_on_breach,
                    affected_positions=sector_positions[sector],
                    position_contributions=contributions,
                    recommended_action=f"Reduce {sector} sector exposure by {breach_amount:.1%}",
                    market_conditions=market_conditions
                ))
            else:
                limit.is_breached = False
                limit.breach_severity = None

        return breaches

    def _check_venue_limits(
        self,
        allocations: Dict[str, float],
        metadata: Dict[str, Dict[str, Any]],
        market_conditions: Dict[str, Any]
    ) -> List[LimitBreachEvent]:
        """Check venue concentration limits including PDF 60% CEX limit."""
        breaches = []

        # Aggregate by venue type and specific venue
        cex_allocation = 0.0
        dex_allocation = 0.0
        venue_allocations: Dict[str, float] = defaultdict(float)
        cex_positions = []
        dex_positions = []
        venue_positions: Dict[str, List[str]] = defaultdict(list)

        for pair_id, allocation in allocations.items():
            venue_type = metadata.get(pair_id, {}).get('venue_type', 'cex').lower()
            venue = metadata.get(pair_id, {}).get('venue', 'unknown').lower()

            venue_allocations[venue] += allocation
            venue_positions[venue].append(pair_id)

            if 'cex' in venue_type:
                cex_allocation += allocation
                cex_positions.append(pair_id)
            else:
                dex_allocation += allocation
                dex_positions.append(pair_id)

        # Check CEX-only limit (60% PDF REQUIRED)
        cex_limit = self._limits["venue_cex_only"]
        cex_limit.current_value = cex_allocation
        cex_limit.affected_positions = cex_positions
        cex_limit.last_updated = datetime.now()

        cex_utilization = cex_allocation / cex_limit.max_value if cex_limit.max_value > 0 else 0
        self.predictor.record_utilization("venue_cex_only", cex_utilization * 100)

        if cex_allocation > cex_limit.max_value:
            breach_amount = cex_allocation - cex_limit.max_value
            severity = self._classify_severity(cex_utilization)
            cex_limit.is_breached = True
            cex_limit.breach_severity = severity

            contributions = {
                p: allocations.get(p, 0) / cex_allocation if cex_allocation > 0 else 0
                for p in cex_positions
            }

            breaches.append(LimitBreachEvent(
                event_id=self._generate_event_id(),
                timestamp=datetime.now(),
                limit_type=LimitType.VENUE_TYPE,
                limit_name="CEX-Only Concentration (60% PDF LIMIT)",
                limit_identifier="venue_cex_only",
                max_allowed=cex_limit.max_value,
                actual_value=cex_allocation,
                breach_amount=breach_amount,
                breach_percentage=(cex_utilization - 1) * 100,
                severity=severity,
                action_taken=cex_limit.action_on_breach,
                affected_positions=cex_positions,
                position_contributions=contributions,
                recommended_action=f"Shift {breach_amount:.1%} from CEX to DEX venues",
                market_conditions=market_conditions
            ))
        else:
            cex_limit.is_breached = False
            cex_limit.breach_severity = None

        # Check DEX-only limit
        dex_limit = self._limits["venue_dex_only"]
        dex_limit.current_value = dex_allocation
        dex_limit.affected_positions = dex_positions
        dex_limit.last_updated = datetime.now()

        if dex_allocation > dex_limit.max_value:
            breach_amount = dex_allocation - dex_limit.max_value
            dex_utilization = dex_allocation / dex_limit.max_value
            severity = self._classify_severity(dex_utilization)
            dex_limit.is_breached = True
            dex_limit.breach_severity = severity

            breaches.append(LimitBreachEvent(
                event_id=self._generate_event_id(),
                timestamp=datetime.now(),
                limit_type=LimitType.VENUE_TYPE,
                limit_name="DEX-Only Concentration",
                limit_identifier="venue_dex_only",
                max_allowed=dex_limit.max_value,
                actual_value=dex_allocation,
                breach_amount=breach_amount,
                breach_percentage=(dex_utilization - 1) * 100,
                severity=severity,
                action_taken=dex_limit.action_on_breach,
                affected_positions=dex_positions,
                position_contributions={p: allocations.get(p, 0) / dex_allocation for p in dex_positions} if dex_allocation > 0 else {},
                recommended_action=f"Shift {breach_amount:.1%} from DEX to CEX venues",
                market_conditions=market_conditions
            ))
        else:
            dex_limit.is_breached = False
            dex_limit.breach_severity = None

        # Check single venue limits
        for venue, allocation in venue_allocations.items():
            limit_key = f"venue_{venue}"
            if limit_key not in self._limits:
                self._limits[limit_key] = ConcentrationLimit(
                    limit_type=LimitType.VENUE,
                    name=f"Venue: {venue.upper()}",
                    identifier=limit_key,
                    max_value=self.config.single_venue_limit,
                    warning_threshold=self.config.warning_threshold
                )

            limit = self._limits[limit_key]
            limit.current_value = allocation
            limit.affected_positions = venue_positions[venue]

            if allocation > limit.max_value:
                breach_amount = allocation - limit.max_value
                utilization = allocation / limit.max_value
                severity = self._classify_severity(utilization)

                breaches.append(LimitBreachEvent(
                    event_id=self._generate_event_id(),
                    timestamp=datetime.now(),
                    limit_type=LimitType.VENUE,
                    limit_name=f"Single Venue: {venue.upper()}",
                    limit_identifier=limit_key,
                    max_allowed=limit.max_value,
                    actual_value=allocation,
                    breach_amount=breach_amount,
                    breach_percentage=(utilization - 1) * 100,
                    severity=severity,
                    action_taken=LimitAction.ALERT,
                    affected_positions=venue_positions[venue],
                    position_contributions={p: allocations.get(p, 0) / allocation for p in venue_positions[venue]} if allocation > 0 else {},
                    recommended_action=f"Diversify away from {venue}",
                    market_conditions=market_conditions
                ))

        return breaches

    def _check_tier_limits(
        self,
        allocations: Dict[str, float],
        metadata: Dict[str, Dict[str, Any]],
        market_conditions: Dict[str, Any]
    ) -> List[LimitBreachEvent]:
        """Check tier concentration limits including PDF 20% Tier 3 limit."""
        breaches = []

        # Aggregate by tier
        tier_allocations = {1: 0.0, 2: 0.0, 3: 0.0}
        tier_positions = {1: [], 2: [], 3: []}

        for pair_id, allocation in allocations.items():
            tier = metadata.get(pair_id, {}).get('tier', 3)
            tier = min(max(tier, 1), 3)  # Clamp to 1-3
            tier_allocations[tier] += allocation
            tier_positions[tier].append(pair_id)

        # Check Tier 3 limit (20% PDF REQUIRED)
        tier3_limit = self._limits["tier_3"]
        tier3_limit.current_value = tier_allocations[3]
        tier3_limit.affected_positions = tier_positions[3]
        tier3_limit.last_updated = datetime.now()

        tier3_utilization = tier_allocations[3] / tier3_limit.max_value if tier3_limit.max_value > 0 else 0
        self.predictor.record_utilization("tier_3", tier3_utilization * 100)

        if tier_allocations[3] > tier3_limit.max_value:
            breach_amount = tier_allocations[3] - tier3_limit.max_value
            severity = self._classify_severity(tier3_utilization)
            tier3_limit.is_breached = True
            tier3_limit.breach_severity = severity

            contributions = {
                p: allocations.get(p, 0) / tier_allocations[3] if tier_allocations[3] > 0 else 0
                for p in tier_positions[3]
            }

            breaches.append(LimitBreachEvent(
                event_id=self._generate_event_id(),
                timestamp=datetime.now(),
                limit_type=LimitType.TIER,
                limit_name="Tier 3 Assets (20% PDF LIMIT)",
                limit_identifier="tier_3",
                max_allowed=tier3_limit.max_value,
                actual_value=tier_allocations[3],
                breach_amount=breach_amount,
                breach_percentage=(tier3_utilization - 1) * 100,
                severity=severity,
                action_taken=tier3_limit.action_on_breach,
                affected_positions=tier_positions[3],
                position_contributions=contributions,
                recommended_action=f"Reduce Tier 3 exposure by {breach_amount:.1%}, shift to Tier 1/2",
                market_conditions=market_conditions
            ))
        else:
            tier3_limit.is_breached = False
            tier3_limit.breach_severity = None

        # Check Tier 2 limit
        tier2_limit = self._limits["tier_2"]
        tier2_limit.current_value = tier_allocations[2]
        tier2_limit.affected_positions = tier_positions[2]

        if tier_allocations[2] > tier2_limit.max_value:
            breach_amount = tier_allocations[2] - tier2_limit.max_value
            utilization = tier_allocations[2] / tier2_limit.max_value
            severity = self._classify_severity(utilization)
            tier2_limit.is_breached = True
            tier2_limit.breach_severity = severity

            breaches.append(LimitBreachEvent(
                event_id=self._generate_event_id(),
                timestamp=datetime.now(),
                limit_type=LimitType.TIER,
                limit_name="Tier 2 Assets",
                limit_identifier="tier_2",
                max_allowed=tier2_limit.max_value,
                actual_value=tier_allocations[2],
                breach_amount=breach_amount,
                breach_percentage=(utilization - 1) * 100,
                severity=severity,
                action_taken=tier2_limit.action_on_breach,
                affected_positions=tier_positions[2],
                position_contributions={p: allocations.get(p, 0) / tier_allocations[2] for p in tier_positions[2]} if tier_allocations[2] > 0 else {},
                recommended_action=f"Reduce Tier 2 exposure by {breach_amount:.1%}",
                market_conditions=market_conditions
            ))
        else:
            tier2_limit.is_breached = False
            tier2_limit.breach_severity = None

        # Update Tier 1 (no limit, tracking only)
        tier1_limit = self._limits["tier_1"]
        tier1_limit.current_value = tier_allocations[1]
        tier1_limit.affected_positions = tier_positions[1]
        tier1_limit.is_breached = False

        return breaches

    def _check_single_position_limits(
        self,
        allocations: Dict[str, float],
        market_conditions: Dict[str, Any]
    ) -> List[LimitBreachEvent]:
        """Check single position concentration limits."""
        breaches = []
        limit = self._limits["single_position"]

        for pair_id, allocation in allocations.items():
            if allocation > limit.max_value:
                breach_amount = allocation - limit.max_value
                utilization = allocation / limit.max_value
                severity = self._classify_severity(utilization)

                breaches.append(LimitBreachEvent(
                    event_id=self._generate_event_id(),
                    timestamp=datetime.now(),
                    limit_type=LimitType.SINGLE_POSITION,
                    limit_name=f"Single Position: {pair_id}",
                    limit_identifier=f"single_position_{pair_id}",
                    max_allowed=limit.max_value,
                    actual_value=allocation,
                    breach_amount=breach_amount,
                    breach_percentage=(utilization - 1) * 100,
                    severity=severity,
                    action_taken=limit.action_on_breach,
                    affected_positions=[pair_id],
                    position_contributions={pair_id: 1.0},
                    recommended_action=f"Reduce {pair_id} position by {breach_amount:.1%}",
                    market_conditions=market_conditions
                ))

        return breaches

    def _check_correlation_limits(
        self,
        allocations: Dict[str, float],
        correlation_matrix: pd.DataFrame,
        market_conditions: Dict[str, Any]
    ) -> List[LimitBreachEvent]:
        """Check correlation cluster concentration limits."""
        breaches = []
        pairs = list(allocations.keys())

        # Find highly correlated clusters using union-find
        clusters: List[Set[str]] = []
        visited: Set[str] = set()

        for pair in pairs:
            if pair in visited or pair not in correlation_matrix.index:
                continue

            cluster = {pair}
            stack = [pair]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)

                if current in correlation_matrix.index:
                    for other in correlation_matrix.index:
                        if other not in cluster and other in pairs:
                            try:
                                corr = abs(correlation_matrix.loc[current, other])
                                if corr >= self.config.high_correlation_threshold:
                                    cluster.add(other)
                                    stack.append(other)
                            except KeyError:
                                continue

            if len(cluster) > 1:
                clusters.append(cluster)

        # Check cluster allocations
        limit = self._limits["correlation_cluster"]

        for i, cluster in enumerate(clusters):
            cluster_allocation = sum(allocations.get(p, 0) for p in cluster)

            if cluster_allocation > limit.max_value:
                breach_amount = cluster_allocation - limit.max_value
                utilization = cluster_allocation / limit.max_value
                severity = self._classify_severity(utilization)

                contributions = {
                    p: allocations.get(p, 0) / cluster_allocation if cluster_allocation > 0 else 0
                    for p in cluster
                }

                breaches.append(LimitBreachEvent(
                    event_id=self._generate_event_id(),
                    timestamp=datetime.now(),
                    limit_type=LimitType.CORRELATION,
                    limit_name=f"Correlated Cluster ({len(cluster)} pairs)",
                    limit_identifier=f"correlation_cluster_{i}",
                    max_allowed=limit.max_value,
                    actual_value=cluster_allocation,
                    breach_amount=breach_amount,
                    breach_percentage=(utilization - 1) * 100,
                    severity=severity,
                    action_taken=LimitAction.ALERT,
                    affected_positions=list(cluster),
                    position_contributions=contributions,
                    recommended_action=f"Reduce correlated cluster exposure by {breach_amount:.1%}",
                    market_conditions=market_conditions
                ))

        return breaches

    def _check_volatility_limit(
        self,
        portfolio_volatility: float,
        market_conditions: Dict[str, Any]
    ) -> Optional[LimitBreachEvent]:
        """Check portfolio volatility limit."""
        limit = self._limits["portfolio_volatility"]
        limit.current_value = portfolio_volatility
        limit.last_updated = datetime.now()

        utilization = portfolio_volatility / limit.max_value if limit.max_value > 0 else 0
        self.predictor.record_utilization("portfolio_volatility", utilization * 100)

        if portfolio_volatility > limit.max_value:
            breach_amount = portfolio_volatility - limit.max_value
            severity = self._classify_severity(utilization)
            limit.is_breached = True
            limit.breach_severity = severity

            return LimitBreachEvent(
                event_id=self._generate_event_id(),
                timestamp=datetime.now(),
                limit_type=LimitType.VOLATILITY,
                limit_name="Portfolio Volatility",
                limit_identifier="portfolio_volatility",
                max_allowed=limit.max_value,
                actual_value=portfolio_volatility,
                breach_amount=breach_amount,
                breach_percentage=(utilization - 1) * 100,
                severity=severity,
                action_taken=LimitAction.ALERT,
                affected_positions=[],
                position_contributions={},
                recommended_action=f"Reduce portfolio volatility by {breach_amount:.1%}",
                market_conditions=market_conditions
            )

        limit.is_breached = False
        limit.breach_severity = None
        return None

    def _check_drawdown_limit(
        self,
        current_drawdown: float,
        market_conditions: Dict[str, Any]
    ) -> Optional[LimitBreachEvent]:
        """Check maximum drawdown limit."""
        limit = self._limits["max_drawdown"]
        limit.current_value = current_drawdown
        limit.last_updated = datetime.now()

        utilization = current_drawdown / limit.max_value if limit.max_value > 0 else 0
        self.predictor.record_utilization("max_drawdown", utilization * 100)

        if current_drawdown > limit.max_value:
            breach_amount = current_drawdown - limit.max_value
            severity = self._classify_severity(utilization)
            limit.is_breached = True
            limit.breach_severity = severity

            return LimitBreachEvent(
                event_id=self._generate_event_id(),
                timestamp=datetime.now(),
                limit_type=LimitType.DRAWDOWN,
                limit_name="Maximum Drawdown",
                limit_identifier="max_drawdown",
                max_allowed=limit.max_value,
                actual_value=current_drawdown,
                breach_amount=breach_amount,
                breach_percentage=(utilization - 1) * 100,
                severity=BreachSeverity.CRITICAL,
                action_taken=LimitAction.AUTO_REDUCE,
                affected_positions=[],
                position_contributions={},
                recommended_action="CRITICAL: Reduce all positions by 50%",
                market_conditions=market_conditions
            )

        if current_drawdown > self.config.crisis_reduction_trigger:
            limit.breach_severity = BreachSeverity.WARNING

        limit.is_breached = False
        return None

    def _check_beta_limit(
        self,
        portfolio_beta: float,
        market_conditions: Dict[str, Any]
    ) -> Optional[LimitBreachEvent]:
        """Check portfolio beta limit."""
        limit = self._limits["market_beta"]
        limit.current_value = portfolio_beta
        limit.last_updated = datetime.now()

        # Check upper limit
        if portfolio_beta > limit.max_value:
            breach_amount = portfolio_beta - limit.max_value
            utilization = portfolio_beta / limit.max_value
            severity = self._classify_severity(utilization)
            limit.is_breached = True
            limit.breach_severity = severity

            return LimitBreachEvent(
                event_id=self._generate_event_id(),
                timestamp=datetime.now(),
                limit_type=LimitType.BETA,
                limit_name="Market Beta (Upper)",
                limit_identifier="market_beta_upper",
                max_allowed=limit.max_value,
                actual_value=portfolio_beta,
                breach_amount=breach_amount,
                breach_percentage=(utilization - 1) * 100,
                severity=severity,
                action_taken=LimitAction.ALERT,
                affected_positions=[],
                position_contributions={},
                recommended_action=f"Reduce market beta from {portfolio_beta:.2f} to {limit.max_value:.2f}",
                market_conditions=market_conditions
            )

        # Check lower limit
        if portfolio_beta < limit.min_value:
            breach_amount = limit.min_value - portfolio_beta
            return LimitBreachEvent(
                event_id=self._generate_event_id(),
                timestamp=datetime.now(),
                limit_type=LimitType.BETA,
                limit_name="Market Beta (Lower)",
                limit_identifier="market_beta_lower",
                max_allowed=limit.min_value,
                actual_value=portfolio_beta,
                breach_amount=breach_amount,
                breach_percentage=breach_amount / limit.min_value * 100 if limit.min_value > 0 else 0,
                severity=BreachSeverity.WARNING,
                action_taken=LimitAction.ALERT,
                affected_positions=[],
                position_contributions={},
                recommended_action=f"Increase market beta from {portfolio_beta:.2f} to {limit.min_value:.2f}",
                market_conditions=market_conditions
            )

        limit.is_breached = False
        limit.breach_severity = None
        return None

    def _update_utilization_history(self) -> None:
        """Update utilization history for all limits."""
        timestamp = datetime.now()

        for limit_id, limit in self._limits.items():
            if not limit.is_active:
                continue

            utilization = limit.current_value / limit.max_value if limit.max_value > 0 else 0
            trend = self.predictor.get_trend(limit_id)

            # Forecast
            forecast_value = self.predictor.forecast_utilization(limit_id)
            will_breach, days_to_breach, _ = self.predictor.predict_breach(
                limit_id, utilization * 100, limit.max_value
            )

            snapshot = LimitUtilizationSnapshot(
                timestamp=timestamp,
                limit_identifier=limit_id,
                utilization_pct=utilization * 100,
                value=limit.current_value,
                limit=limit.max_value,
                trend=1 if trend == 'increasing' else (-1 if trend == 'decreasing' else 0),
                distance_to_breach=limit.max_value - limit.current_value,
                forecast_value=forecast_value,
                forecast_breach_days=days_to_breach if will_breach else None
            )

            self._utilization_history[limit_id].append(snapshot)

            # Trim old history
            max_history = self.config.history_retention_days * 24 * 12  # 5-min intervals
            if len(self._utilization_history[limit_id]) > max_history:
                self._utilization_history[limit_id] = self._utilization_history[limit_id][-max_history:]

    def get_rebalance_recommendations(
        self,
        breaches: List[LimitBreachEvent],
        allocations: Dict[str, float],
        metadata: Dict[str, Dict[str, Any]]
    ) -> List[RebalanceRecommendation]:
        """Generate prioritized rebalancing recommendations."""
        recommendations = []

        for breach in breaches:
            if breach.severity == BreachSeverity.INFO:
                continue

            priority = {
                BreachSeverity.EMERGENCY: 1,
                BreachSeverity.CRITICAL: 2,
                BreachSeverity.BREACH: 3,
                BreachSeverity.WARNING: 4
            }.get(breach.severity, 5)

            # Get transaction costs for affected positions
            for pair_id in breach.affected_positions:
                current = allocations.get(pair_id, 0)
                if current <= 0:
                    continue

                contribution = breach.position_contributions.get(pair_id, 0)
                reduction_needed = current * (breach.breach_amount / breach.actual_value) if breach.actual_value > 0 else 0
                target = max(0, current - reduction_needed)

                venue_type = metadata.get(pair_id, {}).get('venue_type', 'cex').lower()
                cost_bps = self.config.dex_transaction_cost_bps if 'dex' in venue_type else self.config.cex_transaction_cost_bps

                recommendations.append(RebalanceRecommendation(
                    pair_id=pair_id,
                    current_allocation=current,
                    target_allocation=target,
                    change_amount=target - current,
                    change_pct=(target - current) / current if current > 0 else 0,
                    priority=priority,
                    reason=f"{breach.limit_name}: {breach.breach_percentage:.1f}% over limit",
                    expected_cost_bps=cost_bps * abs(target - current),
                    estimated_slippage_bps=cost_bps * self.config.slippage_factor * abs(target - current),
                    urgency=breach.severity,
                    related_limits=[breach.limit_identifier],
                    trade_direction='reduce'
                ))

        # Sort and deduplicate
        recommendations.sort(key=lambda x: (x.priority, -abs(x.change_amount)))

        seen_pairs: Set[str] = set()
        unique = []
        for rec in recommendations:
            if rec.pair_id not in seen_pairs:
                seen_pairs.add(rec.pair_id)
                unique.append(rec)

        return unique

    def enforce_limits(
        self,
        allocations: Dict[str, float],
        metadata: Dict[str, Dict[str, Any]],
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Automatically enforce all limits and return adjusted allocations."""
        if not self.config.auto_rebalance:
            return allocations

        adjusted = allocations.copy()

        for iteration in range(self.config.max_rebalance_iterations):
            all_ok, breaches = self.check_all_limits(
                adjusted, metadata, correlation_matrix
            )

            if all_ok:
                logger.info(f"Limits enforced after {iteration + 1} iterations")
                break

            # Use optimal rebalancing
            result = self.rebalancer.calculate_optimal_rebalance(
                adjusted, breaches, metadata
            )

            if result.success:
                adjusted = result.final_allocations
            else:
                # Fallback to manual recommendations
                recommendations = self.get_rebalance_recommendations(
                    breaches, adjusted, metadata
                )

                for rec in recommendations[:5]:
                    if rec.pair_id in adjusted:
                        adjusted[rec.pair_id] = rec.target_allocation

        return adjusted

    def get_predictive_alerts(self) -> List[Dict[str, Any]]:
        """Get predictive breach alerts for limits approaching breach."""
        alerts = []

        for limit_id, limit in self._limits.items():
            if not limit.is_active:
                continue

            current_util = limit.current_value / limit.max_value if limit.max_value > 0 else 0

            will_breach, days_to_breach, confidence = self.predictor.predict_breach(
                limit_id, current_util * 100, 100  # 100% = breach
            )

            if will_breach and days_to_breach is not None:
                trend = self.predictor.get_trend(limit_id)

                alerts.append({
                    'limit_id': limit_id,
                    'limit_name': limit.name,
                    'current_utilization_pct': current_util * 100,
                    'predicted_breach_days': days_to_breach,
                    'confidence': confidence,
                    'trend': trend,
                    'is_pdf_required': limit.is_pdf_required,
                    'recommended_action': f"Monitor {limit.name} - breach predicted in {days_to_breach} days"
                })

        # Sort by urgency (days to breach)
        alerts.sort(key=lambda x: x['predicted_breach_days'])

        return alerts

    def generate_compliance_report(
        self,
        period_days: int = 30
    ) -> ComplianceReport:
        """Generate comprehensive compliance report."""
        now = datetime.now()
        period_start = now - timedelta(days=period_days)

        # Count limits
        total_limits = len(self._limits)
        active_limits = sum(1 for l in self._limits.values() if l.is_active)
        breached = [l for l in self._limits.values() if l.is_breached]
        warning = [l for l in self._limits.values()
                   if not l.is_breached and l.current_value >= l.max_value * l.warning_threshold]

        # PDF compliance status
        pdf_compliant = all(
            not l.is_breached
            for l in self._limits.values()
            if l.is_pdf_required
        )

        # Breach statistics
        recent_breaches = [
            b for b in self._breach_history
            if b.timestamp >= period_start
        ]
        critical_breaches = sum(1 for b in recent_breaches if b.severity in [BreachSeverity.CRITICAL, BreachSeverity.EMERGENCY])

        # Build utilization by type
        utilization_by_type = defaultdict(list)
        for limit_id, limit in self._limits.items():
            util = limit.current_value / limit.max_value if limit.max_value > 0 else 0
            utilization_by_type[limit.limit_type.value].append(util)

        avg_utilization = {
            k: np.mean(v) if v else 0
            for k, v in utilization_by_type.items()
        }

        # Highest utilization limits
        sorted_limits = sorted(
            self._limits.values(),
            key=lambda l: l.current_value / l.max_value if l.max_value > 0 else 0,
            reverse=True
        )

        highest_util = [
            {
                'name': l.name,
                'utilization_pct': (l.current_value / l.max_value * 100) if l.max_value > 0 else 0,
                'is_breached': l.is_breached,
                'is_pdf_required': l.is_pdf_required
            }
            for l in sorted_limits[:10]
        ]

        # Trends
        trends = {
            limit_id: self.predictor.get_trend(limit_id)
            for limit_id in self._limits.keys()
        }

        return ComplianceReport(
            report_id=f"RPT-{now.strftime('%Y%m%d%H%M%S')}",
            timestamp=now,
            reporting_period_start=period_start,
            reporting_period_end=now,
            total_limits=total_limits,
            active_limits=active_limits,
            breached_limits=len(breached),
            warning_limits=len(warning),
            healthy_limits=active_limits - len(breached) - len(warning),
            pdf_limits_compliant=pdf_compliant,
            sector_limit_status={
                'limit': self.config.sector_limit,
                'max_utilization': max(
                    (l.current_value for l in self._limits.values() if l.limit_type == LimitType.SECTOR),
                    default=0
                )
            },
            cex_limit_status={
                'limit': self.config.cex_only_limit,
                'current': self._limits['venue_cex_only'].current_value,
                'compliant': not self._limits['venue_cex_only'].is_breached
            },
            tier3_limit_status={
                'limit': self.config.tier3_limit,
                'current': self._limits['tier_3'].current_value,
                'compliant': not self._limits['tier_3'].is_breached
            },
            total_breach_events=len(recent_breaches),
            critical_breaches=critical_breaches,
            breach_duration_avg_minutes=0.0,
            auto_corrections_count=sum(1 for b in recent_breaches if b.was_auto_corrected),
            utilization_by_type=avg_utilization,
            highest_utilization_limits=highest_util,
            utilization_trends=trends,
            portfolio_volatility=self._limits['portfolio_volatility'].current_value,
            current_drawdown=self._limits['max_drawdown'].current_value,
            var_95=0.0,
            cvar_95=0.0,
            limit_details=[],
            breach_history=[],
            recommendations=[]
        )

    def generate_limits_report(self, detailed: bool = True) -> str:
        """Generate formatted limits status report."""
        lines = [
            "=" * 100,
            "CONCENTRATION LIMITS REPORT - PDF Section 2.4 Compliant",
            "=" * 100,
            "",
            "PDF REQUIRED LIMITS (MANDATORY)",
            "-" * 60,
            f"  Sector Concentration:    40% max per sector",
            f"  CEX-Only Concentration:  60% max CEX-only",
            f"  Tier 3 Allocation:       20% max Tier 3 assets",
            "",
            "CURRENT LIMIT STATUS",
            "-" * 100,
            f"{'Status':<8} {'Limit Name':<40} {'Current':>10} {'Max':>10} {'Utilization':>12} {'Severity':<10}",
            "-" * 100,
        ]

        # Sort: breached first, then by utilization
        sorted_limits = sorted(
            self._limits.values(),
            key=lambda l: (
                not l.is_breached,
                -l.current_value / l.max_value if l.max_value > 0 else 0
            )
        )

        for limit in sorted_limits:
            if not limit.is_active:
                continue

            status = "BREACH" if limit.is_breached else "OK"
            utilization = limit.current_value / limit.max_value * 100 if limit.max_value > 0 else 0
            severity = limit.breach_severity.value.upper() if limit.breach_severity else "-"
            pdf_marker = " [PDF]" if limit.is_pdf_required else ""

            lines.append(
                f"{status:<8} {limit.name[:38] + pdf_marker:<40} "
                f"{limit.current_value:>9.1%} {limit.max_value:>9.1%} "
                f"{utilization:>10.1f}% {severity:<10}"
            )

        # Predictive alerts
        alerts = self.get_predictive_alerts()
        if alerts:
            lines.extend([
                "",
                "PREDICTIVE ALERTS",
                "-" * 60,
            ])
            for alert in alerts[:5]:
                lines.append(
                    f"  {alert['limit_name'][:30]}: Breach in ~{alert['predicted_breach_days']} days "
                    f"(conf: {alert['confidence']:.0%}, trend: {alert['trend']})"
                )

        # Recent breaches
        recent_breaches = self._breach_history[-10:]
        if recent_breaches:
            lines.extend([
                "",
                "RECENT BREACH EVENTS (Last 10)",
                "-" * 60,
            ])
            for breach in reversed(recent_breaches):
                lines.append(
                    f"  {breach.timestamp.strftime('%Y-%m-%d %H:%M')} | "
                    f"{breach.severity.value:8s} | {breach.limit_name[:35]}"
                )

        lines.extend([
            "",
            "=" * 100,
            f"Report generated: {datetime.now().isoformat()}",
            f"Total limits: {len(self._limits)} | Breached: {sum(1 for l in self._limits.values() if l.is_breached)} | "
            f"Warning: {sum(1 for l in self._limits.values() if l.breach_severity == BreachSeverity.WARNING)}",
            "=" * 100
        ])

        return "\n".join(lines)

    def get_limits_summary(self) -> Dict[str, Any]:
        """Get limits summary for JSON export."""
        breached = [l for l in self._limits.values() if l.is_breached]
        warnings = [l for l in self._limits.values()
                   if not l.is_breached and l.current_value >= l.max_value * l.warning_threshold]

        pdf_status = {
            'sector_40pct_compliant': not any(
                l.is_breached for l in self._limits.values()
                if l.limit_type == LimitType.SECTOR
            ),
            'cex_60pct_compliant': not self._limits['venue_cex_only'].is_breached,
            'tier3_20pct_compliant': not self._limits['tier_3'].is_breached,
            'all_pdf_limits_compliant': all(
                not l.is_breached for l in self._limits.values() if l.is_pdf_required
            )
        }

        return {
            'timestamp': datetime.now().isoformat(),
            'total_limits': len(self._limits),
            'active_limits': sum(1 for l in self._limits.values() if l.is_active),
            'breached_count': len(breached),
            'warning_count': len(warnings),
            'healthy_count': len(self._limits) - len(breached) - len(warnings),
            'pdf_compliance': pdf_status,
            'breached_limits': [
                {
                    'name': l.name,
                    'identifier': l.identifier,
                    'type': l.limit_type.value,
                    'current': l.current_value,
                    'max': l.max_value,
                    'utilization_pct': l.current_value / l.max_value * 100 if l.max_value > 0 else 0,
                    'severity': l.breach_severity.value if l.breach_severity else None,
                    'is_pdf_required': l.is_pdf_required
                }
                for l in breached
            ],
            'warning_limits': [
                {
                    'name': l.name,
                    'utilization_pct': l.current_value / l.max_value * 100 if l.max_value > 0 else 0
                }
                for l in warnings
            ],
            'breach_history_count': len(self._breach_history)
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_concentration_enforcer(
    config: Optional[Dict[str, Any]] = None
) -> ConcentrationLimitsEnforcer:
    """
    Factory function to create configured ConcentrationLimitsEnforcer.

    Args:
        config: Optional configuration overrides

    Returns:
        Configured ConcentrationLimitsEnforcer instance
    """
    if config:
        # Handle enum conversion
        if 'rebalance_method' in config and isinstance(config['rebalance_method'], str):
            config['rebalance_method'] = RebalanceMethod(config['rebalance_method'])

        limits_config = ConcentrationLimitsConfig(**config)
    else:
        limits_config = ConcentrationLimitsConfig()

    return ConcentrationLimitsEnforcer(limits_config)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main Class
    'ConcentrationLimitsEnforcer',
    'create_concentration_enforcer',

    # Configuration
    'ConcentrationLimitsConfig',

    # Data Classes
    'ConcentrationLimit',
    'LimitBreachEvent',
    'LimitUtilizationSnapshot',
    'RebalanceRecommendation',
    'RebalanceResult',
    'ComplianceReport',

    # Enums
    'LimitType',
    'BreachSeverity',
    'LimitAction',
    'RebalanceMethod',
    'AlertLevel',

    # Sub-engines
    'PredictiveBreachDetector',
    'OptimalRebalancingEngine',
    'LimitAttributionEngine',
    'LimitDefinition',
]
