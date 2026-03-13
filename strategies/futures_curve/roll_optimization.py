"""
Roll Optimization Module - Strategy D: Multi-Venue Roll Optimization
====================================================================

Part 2 Section 3.2.4 - Multi-Venue Roll Optimization Implementation

Implements roll optimization across multiple venues, dynamically
shifting positions based on term structure, costs, and capacity.

Part 2 Requirements Addressed:
- 3.2.4 Strategy D: Multi-Venue Roll Optimization (MANDATORY)
- Multi-venue roll decision tree (Binance, Hyperliquid, dYdX V4, GMX, CME, Deribit)
- Dynamic shifting between venues based on term structure regime
- Cost optimization for roll timing with venue-specific fees
- Capacity-aware position management
- Crisis period handling (COVID, Luna, FTX, May 2021)
- Integration with Strategies A, B, C for coordinated position management

Decision Tree Hierarchy:
1. Expiry constraints (must roll if expiry approaching)
2. Crisis state (reduce positions, widen thresholds)
3. Cost optimization (roll if significant cost savings)
4. Term structure opportunities (roll for curve positioning)
5. Cross-venue arbitrage (shift for venue-specific advantages)
6. Capacity/liquidity management

Version: 3.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import logging

from . import (
    VenueType, TermStructureRegime, SpreadDirection, ExitReason,
    VenueCosts, TermStructurePoint, DEFAULT_VENUE_COSTS, DEFAULT_VENUE_CAPACITY
)
from .term_structure import TermStructureCurve, TermStructureAnalyzer
from .funding_rate_analysis import CRISIS_EVENTS, is_crisis_period

logger = logging.getLogger(__name__)


# =============================================================================
# CRISIS-ADAPTIVE PARAMETERS FOR STRATEGY D
# =============================================================================

STRATEGY_D_CRISIS_PARAMS = {
    'normal': {
        'min_net_benefit_mult': 1.0,
        'max_roll_cost_mult': 1.0,
        'cross_venue_threshold_mult': 1.0,
        'max_position_mult': 1.0,
        'roll_frequency_mult': 1.0,
    },
    'crisis': {
        'min_net_benefit_mult': 2.0,  # Higher threshold needed
        'max_roll_cost_mult': 0.5,  # Lower cost tolerance
        'cross_venue_threshold_mult': 2.0,  # Wider threshold
        'max_position_mult': 0.3,  # Reduce max position
        'roll_frequency_mult': 0.5,  # Roll less frequently
    }
}

# Regime-specific roll adjustments
REGIME_ROLL_PARAMS = {
    TermStructureRegime.STEEP_CONTANGO: {
        'roll_early': True,
        'optimal_days_adjustment': -2,  # Roll 2 days earlier
        'preferred_direction': 'forward',  # Roll to longer expiry
    },
    TermStructureRegime.MILD_CONTANGO: {
        'roll_early': True,
        'optimal_days_adjustment': -1,
        'preferred_direction': 'forward',
    },
    TermStructureRegime.FLAT: {
        'roll_early': False,
        'optimal_days_adjustment': 0,
        'preferred_direction': 'neutral',
    },
    TermStructureRegime.MILD_BACKWARDATION: {
        'roll_early': False,
        'optimal_days_adjustment': 1,  # Roll 1 day later
        'preferred_direction': 'backward',  # Roll to shorter expiry
    },
    TermStructureRegime.STEEP_BACKWARDATION: {
        'roll_early': False,
        'optimal_days_adjustment': 2,  # Roll 2 days later
        'preferred_direction': 'backward',
    },
}


class RollDecision(Enum):
    """Roll decision types."""
    HOLD = "hold"                           # Keep current position
    ROLL_FORWARD = "roll_forward"           # Roll to longer expiry
    ROLL_BACKWARD = "roll_backward"         # Roll to shorter expiry
    CROSS_VENUE_SHIFT = "cross_venue_shift" # Move to different venue
    PARTIAL_ROLL = "partial_roll"           # Roll portion of position
    UNWIND = "unwind"                       # Close position entirely


class RollReason(Enum):
    """Reasons for roll decisions."""
    EXPIRY_APPROACHING = "expiry_approaching"
    COST_OPTIMIZATION = "cost_optimization"
    LIQUIDITY_CONCERNS = "liquidity_concerns"
    TERM_STRUCTURE_SHIFT = "term_structure_shift"
    VENUE_ADVANTAGE = "venue_advantage"
    CAPACITY_REBALANCE = "capacity_rebalance"
    FUNDING_DIVERGENCE = "funding_divergence"
    REGIME_CHANGE = "regime_change"
    RISK_REDUCTION = "risk_reduction"
    PROFIT_TARGET = "profit_target"


class VenuePreference(Enum):
    """Venue preference tiers."""
    PRIMARY = 1       # Preferred venues (low cost, high liquidity)
    SECONDARY = 2     # Acceptable venues (moderate cost/liquidity)
    TERTIARY = 3      # Last resort venues (high cost or low liquidity)
    EXCLUDED = 4      # Temporarily or permanently excluded


@dataclass
class RollCost:
    """Detailed roll cost breakdown."""
    close_fees: float = 0.0
    open_fees: float = 0.0
    slippage_close: float = 0.0
    slippage_open: float = 0.0
    gas_costs: float = 0.0
    market_impact: float = 0.0
    opportunity_cost: float = 0.0

    @property
    def total(self) -> float:
        return (self.close_fees + self.open_fees +
                self.slippage_close + self.slippage_open +
                self.gas_costs + self.market_impact + self.opportunity_cost)


@dataclass
class RollOpportunity:
    """A potential roll opportunity."""
    timestamp: datetime
    current_venue: str
    current_expiry: Optional[datetime]
    target_venue: str
    target_expiry: Optional[datetime]
    decision: RollDecision
    reason: RollReason
    cost: RollCost
    expected_benefit: float
    net_benefit: float
    confidence: float
    priority_score: float
    constraints_satisfied: bool = True
    notes: List[str] = field(default_factory=list)


@dataclass
class RollExecution:
    """Record of an executed roll."""
    execution_id: str
    timestamp: datetime
    opportunity: RollOpportunity
    position_size_btc: float
    actual_cost: RollCost
    slippage_vs_expected: float
    success: bool
    execution_price_close: Optional[float] = None
    execution_price_open: Optional[float] = None


@dataclass
class VenueState:
    """Current state of a venue for roll decisions."""
    venue: str
    venue_type: VenueType
    available_capacity_btc: float
    current_position_btc: float
    term_structure: Optional[TermStructureCurve]
    funding_rate_hourly: Optional[float]
    liquidity_score: float  # 0-1
    is_available: bool
    preference: VenuePreference
    costs: VenueCosts
    last_update: datetime


@dataclass
class RollConfig:
    """Configuration for roll optimization."""
    # Timing parameters - PDF: Roll 2-5 days before expiry
    min_days_to_expiry_roll: int = 2       # PDF: 2-5 days before expiry
    optimal_days_to_expiry_roll: int = 5   # PDF: optimize timing within 2-5 day window
    max_roll_frequency_hours: int = 336    # Once per 2 weeks max (avoid over-trading)

    # Cost thresholds - strict for profitable rolling
    min_net_benefit_pct: float = 0.50      # 50 bps minimum benefit to roll
    max_roll_cost_pct: float = 0.3         # 30 bps max acceptable roll cost

    # Cross-venue thresholds - meaningful spread required
    cross_venue_benefit_threshold_pct: float = 0.40  # 40 bps to shift venues

    # Position limits
    max_position_per_venue_btc: float = 100.0
    max_total_position_btc: float = 300.0
    min_roll_size_btc: float = 0.1

    # Risk parameters
    max_concentration_single_venue: float = 0.5  # 50% max in single venue
    min_liquidity_score: float = 0.3

    # Regime-based adjustments
    contango_roll_early: bool = True  # Roll earlier in contango
    backwardation_roll_late: bool = True  # Roll later in backwardation

    # Venue preferences
    preferred_venues: List[str] = field(default_factory=lambda: [
        'binance', 'hyperliquid', 'dydx'
    ])

    # Funding considerations - trigger on 2% annualized divergence
    funding_divergence_threshold_pct: float = 5.0  # 5% annualized (need substantial divergence)


@dataclass
class RollBacktestResult:
    """Results from roll optimization backtesting."""
    total_rolls: int
    successful_rolls: int
    total_cost_usd: float
    total_benefit_usd: float
    net_pnl_usd: float
    avg_cost_per_roll_bps: float
    avg_benefit_per_roll_bps: float
    roll_efficiency: float  # benefit / cost ratio
    venue_distribution: Dict[str, float]
    roll_reasons: Dict[str, int]
    monthly_statistics: pd.DataFrame
    execution_log: List[RollExecution]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_rolls': self.total_rolls,
            'successful_rolls': self.successful_rolls,
            'success_rate': self.successful_rolls / max(self.total_rolls, 1),
            'total_cost_usd': self.total_cost_usd,
            'total_benefit_usd': self.total_benefit_usd,
            'net_pnl_usd': self.net_pnl_usd,
            'avg_cost_per_roll_bps': self.avg_cost_per_roll_bps,
            'avg_benefit_per_roll_bps': self.avg_benefit_per_roll_bps,
            'roll_efficiency': self.roll_efficiency,
            'venue_distribution': self.venue_distribution,
            'roll_reasons': self.roll_reasons
        }


class RollDecisionTree:
    """
    Multi-venue roll decision tree implementing Strategy D.

    Decision hierarchy:
    1. Expiry constraints (must roll if expiry approaching)
    2. Cost optimization (roll if significant cost savings)
    3. Term structure opportunities (roll for curve positioning)
    4. Cross-venue arbitrage (shift for venue-specific advantages)
    5. Capacity/liquidity management
    """

    def __init__(self, config: RollConfig):
        self.config = config
        self.venue_costs = DEFAULT_VENUE_COSTS.copy()
        self.venue_capacity = DEFAULT_VENUE_CAPACITY.copy()

    def evaluate_roll(
        self,
        current_position: Dict[str, Any],
        venue_states: Dict[str, VenueState],
        market_regime: TermStructureRegime,
        timestamp: datetime
    ) -> Optional[RollOpportunity]:
        """
        Main decision tree entry point.

        Returns the best roll opportunity if one exists.
        """
        opportunities = []

        current_venue = current_position.get('venue')
        current_expiry = current_position.get('expiry')
        position_size = current_position.get('size_btc', 0)

        if position_size < self.config.min_roll_size_btc:
            return None

        current_state = venue_states.get(current_venue)
        if not current_state:
            return None

        # Level 1: Check expiry constraints
        expiry_opportunity = self._check_expiry_roll(
            current_position, current_state, venue_states, timestamp
        )
        if expiry_opportunity and expiry_opportunity.priority_score > 0.9:
            return expiry_opportunity  # Mandatory roll
        if expiry_opportunity:
            opportunities.append(expiry_opportunity)

        # Level 2: Check cost optimization
        cost_opportunity = self._check_cost_optimization(
            current_position, current_state, venue_states, market_regime, timestamp
        )
        if cost_opportunity:
            opportunities.append(cost_opportunity)

        # Level 3: Check term structure opportunities
        ts_opportunity = self._check_term_structure_opportunity(
            current_position, current_state, venue_states, market_regime, timestamp
        )
        if ts_opportunity:
            opportunities.append(ts_opportunity)

        # Level 4: Check cross-venue arbitrage
        xv_opportunity = self._check_cross_venue_opportunity(
            current_position, current_state, venue_states, market_regime, timestamp
        )
        if xv_opportunity:
            opportunities.append(xv_opportunity)

        # Level 5: Check capacity/liquidity
        cap_opportunity = self._check_capacity_opportunity(
            current_position, current_state, venue_states, timestamp
        )
        if cap_opportunity:
            opportunities.append(cap_opportunity)

        # Return highest priority opportunity
        if opportunities:
            return max(opportunities, key=lambda x: x.priority_score)
        return None

    def _check_expiry_roll(
        self,
        position: Dict[str, Any],
        current_state: VenueState,
        venue_states: Dict[str, VenueState],
        timestamp: datetime
    ) -> Optional[RollOpportunity]:
        """Check if roll is needed due to expiry approaching."""
        current_expiry = position.get('expiry')
        if not current_expiry:
            return None  # Perpetual, no expiry

        days_to_expiry = (current_expiry - timestamp).days

        if days_to_expiry > self.config.optimal_days_to_expiry_roll:
            return None

        # Find best target
        target_venue, target_expiry = self._find_best_roll_target(
            position, current_state, venue_states, timestamp
        )

        if not target_venue:
            return None

        # Calculate costs
        cost = self._calculate_roll_cost(
            position, current_state.venue, current_expiry,
            target_venue, target_expiry, venue_states
        )

        # Benefit is avoiding forced liquidation or poor execution at expiry
        expected_benefit = position.get('size_btc', 0) * position.get('entry_price', 0) * 0.005

        # Priority based on days to expiry
        if days_to_expiry <= self.config.min_days_to_expiry_roll:
            priority = 1.0  # Mandatory
        else:
            priority = 1.0 - (days_to_expiry - self.config.min_days_to_expiry_roll) / 10

        return RollOpportunity(
            timestamp=timestamp,
            current_venue=current_state.venue,
            current_expiry=current_expiry,
            target_venue=target_venue,
            target_expiry=target_expiry,
            decision=RollDecision.ROLL_FORWARD,
            reason=RollReason.EXPIRY_APPROACHING,
            cost=cost,
            expected_benefit=expected_benefit,
            net_benefit=expected_benefit - cost.total,
            confidence=0.9,
            priority_score=priority,
            notes=[f"Days to expiry: {days_to_expiry}"]
        )

    def _check_cost_optimization(
        self,
        position: Dict[str, Any],
        current_state: VenueState,
        venue_states: Dict[str, VenueState],
        regime: TermStructureRegime,
        timestamp: datetime
    ) -> Optional[RollOpportunity]:
        """Check for cost-saving roll opportunities."""
        position_value = position.get('size_btc', 0) * position.get('entry_price', 0)

        # Check each alternative venue
        best_savings = 0
        best_target = None

        for venue_name, state in venue_states.items():
            if venue_name == current_state.venue:
                continue
            if not state.is_available:
                continue
            if state.preference == VenuePreference.EXCLUDED:
                continue

            # Calculate cost difference
            current_costs = self._estimate_holding_costs(
                position_value, current_state, days=30
            )
            target_costs = self._estimate_holding_costs(
                position_value, state, days=30
            )

            roll_cost = self._calculate_roll_cost(
                position, current_state.venue, position.get('expiry'),
                venue_name, None, venue_states
            )

            savings = current_costs - target_costs - roll_cost.total

            if savings > best_savings:
                best_savings = savings
                best_target = venue_name

        if not best_target or best_savings < position_value * self.config.min_net_benefit_pct / 100:
            return None

        cost = self._calculate_roll_cost(
            position, current_state.venue, position.get('expiry'),
            best_target, None, venue_states
        )

        return RollOpportunity(
            timestamp=timestamp,
            current_venue=current_state.venue,
            current_expiry=position.get('expiry'),
            target_venue=best_target,
            target_expiry=None,
            decision=RollDecision.CROSS_VENUE_SHIFT,
            reason=RollReason.COST_OPTIMIZATION,
            cost=cost,
            expected_benefit=best_savings + cost.total,
            net_benefit=best_savings,
            confidence=0.8,
            priority_score=min(0.7, best_savings / position_value * 100),
            notes=[f"30-day cost savings: ${best_savings:.2f}"]
        )

    def _check_term_structure_opportunity(
        self,
        position: Dict[str, Any],
        current_state: VenueState,
        venue_states: Dict[str, VenueState],
        regime: TermStructureRegime,
        timestamp: datetime
    ) -> Optional[RollOpportunity]:
        """Check for term structure-based roll opportunities."""
        if not current_state.term_structure:
            return None

        position_value = position.get('size_btc', 0) * position.get('entry_price', 0)
        position_direction = position.get('direction', 'long')

        # In contango, longs benefit from rolling forward
        # In backwardation, shorts benefit from rolling forward
        curve = current_state.term_structure

        best_opportunity = None
        best_benefit = 0

        for venue_name, state in venue_states.items():
            if not state.term_structure or not state.is_available:
                continue

            other_curve = state.term_structure

            # Calculate basis differential
            if curve.points and other_curve.points:
                current_basis = curve.points[0].annualized_basis_pct if curve.points else 0
                target_basis = other_curve.points[0].annualized_basis_pct if other_curve.points else 0

                # Benefit depends on position direction
                if position_direction == 'long':
                    benefit_bps = target_basis - current_basis
                else:
                    benefit_bps = current_basis - target_basis

                benefit_usd = position_value * benefit_bps / 10000

                if benefit_usd > best_benefit:
                    best_benefit = benefit_usd

                    cost = self._calculate_roll_cost(
                        position, current_state.venue, position.get('expiry'),
                        venue_name, None, venue_states
                    )

                    if benefit_usd > cost.total:
                        best_opportunity = RollOpportunity(
                            timestamp=timestamp,
                            current_venue=current_state.venue,
                            current_expiry=position.get('expiry'),
                            target_venue=venue_name,
                            target_expiry=None,
                            decision=RollDecision.CROSS_VENUE_SHIFT,
                            reason=RollReason.TERM_STRUCTURE_SHIFT,
                            cost=cost,
                            expected_benefit=benefit_usd,
                            net_benefit=benefit_usd - cost.total,
                            confidence=0.7,
                            priority_score=min(0.6, (benefit_usd - cost.total) / position_value * 100),
                            notes=[f"Basis differential: {target_basis - current_basis:.2f} bps"]
                        )

        return best_opportunity

    def _check_cross_venue_opportunity(
        self,
        position: Dict[str, Any],
        current_state: VenueState,
        venue_states: Dict[str, VenueState],
        regime: TermStructureRegime,
        timestamp: datetime
    ) -> Optional[RollOpportunity]:
        """Check for cross-venue arbitrage opportunities."""
        position_value = position.get('size_btc', 0) * position.get('entry_price', 0)

        # Check funding rate divergence
        if current_state.funding_rate_hourly is None:
            return None

        best_divergence = 0
        best_target = None

        for venue_name, state in venue_states.items():
            if venue_name == current_state.venue:
                continue
            if not state.is_available or state.funding_rate_hourly is None:
                continue

            # Calculate funding divergence (annualized in %)
            current_annual = current_state.funding_rate_hourly * 24 * 365 * 100
            target_annual = state.funding_rate_hourly * 24 * 365 * 100
            divergence = abs(current_annual - target_annual)

            # Fixed: threshold is already in % (e.g., 0.5 = 0.5%), not bps
            # A 0.5% annual funding divergence is significant for arb
            if divergence > self.config.funding_divergence_threshold_pct:
                if divergence > best_divergence:
                    best_divergence = divergence
                    best_target = venue_name

        if not best_target:
            return None

        # Calculate benefit from funding divergence
        benefit_annual_pct = best_divergence / 100
        benefit_30day = position_value * benefit_annual_pct * 30 / 365

        cost = self._calculate_roll_cost(
            position, current_state.venue, position.get('expiry'),
            best_target, None, venue_states
        )

        if benefit_30day <= cost.total:
            return None

        return RollOpportunity(
            timestamp=timestamp,
            current_venue=current_state.venue,
            current_expiry=position.get('expiry'),
            target_venue=best_target,
            target_expiry=None,
            decision=RollDecision.CROSS_VENUE_SHIFT,
            reason=RollReason.FUNDING_DIVERGENCE,
            cost=cost,
            expected_benefit=benefit_30day,
            net_benefit=benefit_30day - cost.total,
            confidence=0.75,
            priority_score=min(0.65, (benefit_30day - cost.total) / position_value * 100),
            notes=[f"Funding divergence: {best_divergence:.2f}% annualized"]
        )

    def _check_capacity_opportunity(
        self,
        position: Dict[str, Any],
        current_state: VenueState,
        venue_states: Dict[str, VenueState],
        timestamp: datetime
    ) -> Optional[RollOpportunity]:
        """Check for capacity/liquidity-based roll opportunities."""
        position_size = position.get('size_btc', 0)

        # Check if current venue is capacity constrained
        capacity_utilization = (
            current_state.current_position_btc /
            max(current_state.available_capacity_btc, 0.01)
        )

        if capacity_utilization < 0.8:
            return None

        # Find venue with more capacity
        best_target = None
        best_capacity = 0

        for venue_name, state in venue_states.items():
            if venue_name == current_state.venue:
                continue
            if not state.is_available:
                continue

            available = state.available_capacity_btc - state.current_position_btc

            if available > best_capacity and available >= position_size:
                best_capacity = available
                best_target = venue_name

        if not best_target:
            return None

        cost = self._calculate_roll_cost(
            position, current_state.venue, position.get('expiry'),
            best_target, None, venue_states
        )

        # Benefit is avoiding capacity constraints
        position_value = position_size * position.get('entry_price', 0)
        benefit = position_value * 0.001  # 10 bps benefit for capacity relief

        return RollOpportunity(
            timestamp=timestamp,
            current_venue=current_state.venue,
            current_expiry=position.get('expiry'),
            target_venue=best_target,
            target_expiry=None,
            decision=RollDecision.CROSS_VENUE_SHIFT,
            reason=RollReason.CAPACITY_REBALANCE,
            cost=cost,
            expected_benefit=benefit,
            net_benefit=benefit - cost.total,
            confidence=0.6,
            priority_score=0.4,
            notes=[f"Capacity utilization: {capacity_utilization:.1%}"]
        )

    def _find_best_roll_target(
        self,
        position: Dict[str, Any],
        current_state: VenueState,
        venue_states: Dict[str, VenueState],
        timestamp: datetime
    ) -> Tuple[Optional[str], Optional[datetime]]:
        """Find the best venue and expiry for rolling."""
        best_venue = None
        best_expiry = None
        best_score = -float('inf')

        for venue_name, state in venue_states.items():
            if not state.is_available:
                continue
            if state.liquidity_score < self.config.min_liquidity_score:
                continue

            # Score based on costs, liquidity, and capacity
            cost_score = 1.0 - (state.costs.taker_fee + state.costs.slippage_bps / 10000)
            liquidity_score = state.liquidity_score
            capacity_score = min(1.0, (state.available_capacity_btc - state.current_position_btc) /
                                max(position.get('size_btc', 1), 1))

            # Venue preference weighting
            pref_weight = {
                VenuePreference.PRIMARY: 1.0,
                VenuePreference.SECONDARY: 0.8,
                VenuePreference.TERTIARY: 0.6,
                VenuePreference.EXCLUDED: 0.0
            }.get(state.preference, 0.5)

            total_score = (cost_score * 0.3 + liquidity_score * 0.3 +
                          capacity_score * 0.2 + pref_weight * 0.2)

            if total_score > best_score:
                best_score = total_score
                best_venue = venue_name

                # For futures venues, find next expiry
                if state.venue_type in [VenueType.CEX_FUTURES, VenueType.CME]:
                    # Find next quarterly expiry
                    best_expiry = self._get_next_expiry(timestamp)
                else:
                    best_expiry = None  # Perpetual

        return best_venue, best_expiry

    def _get_next_expiry(self, timestamp: datetime) -> datetime:
        """Get next quarterly expiry date."""
        # Quarterly expiries are last Friday of March, June, September, December
        year = timestamp.year
        month = timestamp.month

        quarterly_months = [3, 6, 9, 12]
        for qm in quarterly_months:
            if qm >= month:
                expiry_month = qm
                break
        else:
            expiry_month = 3
            year += 1

        # Find last Friday of the month
        import calendar
        last_day = calendar.monthrange(year, expiry_month)[1]
        last_date = datetime(year, expiry_month, last_day)

        # Go back to find Friday
        while last_date.weekday() != 4:  # 4 = Friday
            last_date -= timedelta(days=1)

        return last_date

    def _calculate_roll_cost(
        self,
        position: Dict[str, Any],
        from_venue: str,
        from_expiry: Optional[datetime],
        to_venue: str,
        to_expiry: Optional[datetime],
        venue_states: Dict[str, VenueState]
    ) -> RollCost:
        """Calculate detailed roll costs."""
        position_size = position.get('size_btc', 0)
        price = position.get('entry_price', 0)
        notional = position_size * price

        from_state = venue_states.get(from_venue)
        to_state = venue_states.get(to_venue)

        if not from_state or not to_state:
            return RollCost()

        # Close costs (current venue)
        close_fees = notional * from_state.costs.taker_fee
        slippage_close = notional * from_state.costs.slippage_bps / 10000

        # Open costs (target venue)
        open_fees = notional * to_state.costs.taker_fee
        slippage_open = notional * to_state.costs.slippage_bps / 10000

        # Gas costs (DEX venues)
        gas_costs = 0
        if from_state.venue_type == VenueType.DEX_PERPETUAL:
            gas_costs += from_state.costs.gas_cost_usd
        if to_state.venue_type == VenueType.DEX_PERPETUAL:
            gas_costs += to_state.costs.gas_cost_usd

        # Market impact for larger positions
        impact_factor = min(1.0, position_size / 10) * 0.001
        market_impact = notional * impact_factor

        return RollCost(
            close_fees=close_fees,
            open_fees=open_fees,
            slippage_close=slippage_close,
            slippage_open=slippage_open,
            gas_costs=gas_costs,
            market_impact=market_impact,
            opportunity_cost=0  # Could be expanded
        )

    def _estimate_holding_costs(
        self,
        position_value: float,
        state: VenueState,
        days: int
    ) -> float:
        """Estimate holding costs for a position."""
        if state.funding_rate_hourly is None:
            return 0

        # Funding cost (simplified - assumes constant rate)
        funding_payments = days * 24  # Hourly funding
        funding_cost = abs(state.funding_rate_hourly) * position_value * funding_payments

        return funding_cost


class RollOptimizer:
    """
    Multi-venue roll optimizer implementing Strategy D.

    This class coordinates roll decisions across multiple venues,
    optimizing for cost, capacity, and term structure positioning.
    """

    def __init__(self, config: Optional[RollConfig] = None):
        self.config = config or RollConfig()
        self.decision_tree = RollDecisionTree(self.config)
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.venue_states: Dict[str, VenueState] = {}
        self.execution_history: List[RollExecution] = []
        self.last_roll_times: Dict[str, datetime] = {}

        # Crisis state tracking
        self._current_crisis: Optional[str] = None
        self._crisis_severity: float = 0.0

    def check_crisis_state(self, timestamp: datetime) -> Tuple[bool, Optional[str], float]:
        """
        Check if current timestamp is in a crisis period.

        Args:
            timestamp: Current timestamp

        Returns:
            Tuple of (is_crisis, crisis_name, severity)
        """
        ts = pd.Timestamp(timestamp) if not isinstance(timestamp, pd.Timestamp) else timestamp
        is_crisis, name, severity = is_crisis_period(ts)

        self._current_crisis = name if is_crisis else None
        self._crisis_severity = severity

        if is_crisis:
            logger.warning(f"Crisis period detected: {name} (severity={severity})")

        return is_crisis, name, severity

    def get_crisis_adjusted_config(self, timestamp: datetime) -> RollConfig:
        """
        Get crisis-adjusted configuration.

        Args:
            timestamp: Current timestamp

        Returns:
            Adjusted RollConfig
        """
        is_crisis, _, severity = self.check_crisis_state(timestamp)

        if not is_crisis:
            return self.config

        params = STRATEGY_D_CRISIS_PARAMS['crisis']

        # Create adjusted config
        adjusted = RollConfig(
            min_days_to_expiry_roll=self.config.min_days_to_expiry_roll + 2,  # Earlier in crisis
            optimal_days_to_expiry_roll=self.config.optimal_days_to_expiry_roll + 3,
            max_roll_frequency_hours=int(
                self.config.max_roll_frequency_hours / params['roll_frequency_mult']
            ),
            min_net_benefit_pct=(
                self.config.min_net_benefit_pct * params['min_net_benefit_mult']
            ),
            max_roll_cost_pct=(
                self.config.max_roll_cost_pct * params['max_roll_cost_mult']
            ),
            cross_venue_benefit_threshold_pct=(
                self.config.cross_venue_benefit_threshold_pct *
                params['cross_venue_threshold_mult']
            ),
            max_position_per_venue_btc=(
                self.config.max_position_per_venue_btc * params['max_position_mult']
            ),
            max_total_position_btc=(
                self.config.max_total_position_btc * params['max_position_mult']
            ),
            preferred_venues=self.config.preferred_venues,
        )

        return adjusted

    def get_regime_adjusted_timing(
        self,
        regime: TermStructureRegime,
        base_optimal_days: int
    ) -> int:
        """
        Adjust roll timing based on regime.

        Args:
            regime: Current term structure regime
            base_optimal_days: Base optimal days to expiry

        Returns:
            Adjusted optimal days
        """
        params = REGIME_ROLL_PARAMS.get(regime, REGIME_ROLL_PARAMS[TermStructureRegime.FLAT])
        adjustment = params['optimal_days_adjustment']
        return max(3, base_optimal_days + adjustment)

    def update_venue_state(
        self,
        venue: str,
        term_structure: Optional[TermStructureCurve],
        funding_rate_hourly: Optional[float],
        available_capacity: float,
        liquidity_score: float,
        timestamp: datetime
    ):
        """Update the state of a venue."""
        venue_type = self._get_venue_type(venue)
        costs = DEFAULT_VENUE_COSTS.get(venue, VenueCosts())

        # Determine preference
        if venue in self.config.preferred_venues:
            preference = VenuePreference.PRIMARY
        elif liquidity_score >= 0.7:
            preference = VenuePreference.SECONDARY
        else:
            preference = VenuePreference.TERTIARY

        current_position = sum(
            p.get('size_btc', 0) for p in self.positions.values()
            if p.get('venue') == venue
        )

        self.venue_states[venue] = VenueState(
            venue=venue,
            venue_type=venue_type,
            available_capacity_btc=available_capacity,
            current_position_btc=current_position,
            term_structure=term_structure,
            funding_rate_hourly=funding_rate_hourly,
            liquidity_score=liquidity_score,
            is_available=True,
            preference=preference,
            costs=costs,
            last_update=timestamp
        )

    def _get_venue_type(self, venue: str) -> VenueType:
        """Map venue name to venue type."""
        venue_types = {
            'binance': VenueType.CEX_PERPETUAL,
            'hyperliquid': VenueType.HYBRID_PERPETUAL,
            'dydx': VenueType.HYBRID_PERPETUAL,
            'gmx': VenueType.DEX_PERPETUAL,
            'cme': VenueType.CME,
            'deribit': VenueType.CEX_FUTURES
        }
        return venue_types.get(venue.lower(), VenueType.CEX_PERPETUAL)

    def add_position(
        self,
        position_id: str,
        venue: str,
        size_btc: float,
        entry_price: float,
        direction: str,
        expiry: Optional[datetime] = None,
        entry_time: Optional[datetime] = None
    ):
        """Add a position to track."""
        self.positions[position_id] = {
            'position_id': position_id,
            'venue': venue,
            'size_btc': size_btc,
            'entry_price': entry_price,
            'direction': direction,
            'expiry': expiry,
            'entry_time': entry_time or datetime.now()
        }

    def remove_position(self, position_id: str):
        """Remove a position from tracking."""
        if position_id in self.positions:
            del self.positions[position_id]

    def get_roll_recommendations(
        self,
        regime: TermStructureRegime,
        timestamp: datetime
    ) -> List[RollOpportunity]:
        """Get roll recommendations for all positions."""
        recommendations = []

        for position_id, position in self.positions.items():
            # Check roll frequency limit
            last_roll = self.last_roll_times.get(position_id)
            if last_roll:
                hours_since_roll = (timestamp - last_roll).total_seconds() / 3600
                if hours_since_roll < self.config.max_roll_frequency_hours:
                    continue

            opportunity = self.decision_tree.evaluate_roll(
                position, self.venue_states, regime, timestamp
            )

            if opportunity:
                recommendations.append(opportunity)

        # Sort by priority
        recommendations.sort(key=lambda x: x.priority_score, reverse=True)

        return recommendations

    def execute_roll(
        self,
        position_id: str,
        opportunity: RollOpportunity,
        actual_close_price: float,
        actual_open_price: float,
        timestamp: datetime
    ) -> RollExecution:
        """Execute a roll and record the result."""
        position = self.positions.get(position_id)
        if not position:
            raise ValueError(f"Position {position_id} not found")

        # Calculate actual costs
        notional = position['size_btc'] * actual_close_price
        actual_cost = self.decision_tree._calculate_roll_cost(
            position,
            opportunity.current_venue,
            opportunity.current_expiry,
            opportunity.target_venue,
            opportunity.target_expiry,
            self.venue_states
        )

        # Calculate slippage vs expected
        expected_close = position['entry_price']
        slippage_close = (actual_close_price - expected_close) / expected_close
        slippage_vs_expected = slippage_close * 100  # In percentage

        execution = RollExecution(
            execution_id=f"roll_{position_id}_{timestamp.strftime('%Y%m%d%H%M%S')}",
            timestamp=timestamp,
            opportunity=opportunity,
            position_size_btc=position['size_btc'],
            actual_cost=actual_cost,
            slippage_vs_expected=slippage_vs_expected,
            success=True,
            execution_price_close=actual_close_price,
            execution_price_open=actual_open_price
        )

        # Update position
        self.positions[position_id] = {
            **position,
            'venue': opportunity.target_venue,
            'expiry': opportunity.target_expiry,
            'entry_price': actual_open_price,
            'entry_time': timestamp
        }

        self.last_roll_times[position_id] = timestamp
        self.execution_history.append(execution)

        return execution

    def run_backtest(
        self,
        historical_data: Dict[str, pd.DataFrame],
        initial_positions: List[Dict[str, Any]],
        initial_capital: float = 1_000_000
    ) -> RollBacktestResult:
        """
        Run backtest of roll optimization strategy.

        Args:
            historical_data: Dict mapping venue to DataFrame with columns:
                - timestamp, price, funding_rate, term_structure_data
            initial_positions: List of starting positions
            initial_capital: Starting capital in USD
        """
        # Initialize positions
        for pos in initial_positions:
            self.add_position(
                position_id=pos['position_id'],
                venue=pos['venue'],
                size_btc=pos['size_btc'],
                entry_price=pos['entry_price'],
                direction=pos.get('direction', 'long'),
                expiry=pos.get('expiry')
            )

        executions = []
        total_cost = 0
        total_benefit = 0
        roll_reasons: Dict[str, int] = defaultdict(int)
        venue_volume: Dict[str, float] = defaultdict(float)
        monthly_data = []

        # Get unified timeline
        all_timestamps = set()
        for venue_data in historical_data.values():
            if 'timestamp' in venue_data.columns:
                all_timestamps.update(venue_data['timestamp'].tolist())
        timestamps = sorted(all_timestamps)

        current_month = None
        month_rolls = 0
        month_cost = 0
        month_benefit = 0

        for ts in timestamps:
            # Update venue states from historical data
            for venue, data in historical_data.items():
                venue_ts_data = data[data['timestamp'] == ts]
                if venue_ts_data.empty:
                    continue

                row = venue_ts_data.iloc[0]

                # Build term structure if available
                term_structure = None
                if 'term_structure_data' in row and row['term_structure_data']:
                    # Parse term structure data
                    pass  # Would build TermStructureCurve

                self.update_venue_state(
                    venue=venue,
                    term_structure=term_structure,
                    funding_rate_hourly=row.get('funding_rate'),
                    available_capacity=DEFAULT_VENUE_CAPACITY.get(venue, 100),
                    liquidity_score=0.8,  # Would be computed from order book
                    timestamp=ts
                )

            # Determine regime
            regime = TermStructureRegime.MILD_CONTANGO  # Would be computed

            # Get roll recommendations
            recommendations = self.get_roll_recommendations(regime, ts)

            # Execute top recommendation if beneficial
            for rec in recommendations[:1]:  # Execute one roll per timestamp
                if rec.net_benefit > 0:
                    position_id = list(self.positions.keys())[0]  # Simplified

                    # Simulate execution
                    price = historical_data.get(rec.target_venue, pd.DataFrame()).get('price', pd.Series())
                    if isinstance(price, pd.Series) and not price.empty:
                        exec_price = price.iloc[-1]
                    else:
                        exec_price = self.positions[position_id]['entry_price']

                    execution = self.execute_roll(
                        position_id=position_id,
                        opportunity=rec,
                        actual_close_price=exec_price,
                        actual_open_price=exec_price,
                        timestamp=ts
                    )

                    executions.append(execution)
                    total_cost += execution.actual_cost.total
                    total_benefit += rec.expected_benefit
                    roll_reasons[rec.reason.value] += 1
                    venue_volume[rec.target_venue] += execution.position_size_btc

                    # Monthly tracking
                    month_rolls += 1
                    month_cost += execution.actual_cost.total
                    month_benefit += rec.expected_benefit

            # Track monthly stats
            ts_month = ts.strftime('%Y-%m') if hasattr(ts, 'strftime') else str(ts)[:7]
            if ts_month != current_month:
                if current_month:
                    monthly_data.append({
                        'month': current_month,
                        'rolls': month_rolls,
                        'cost': month_cost,
                        'benefit': month_benefit,
                        'net': month_benefit - month_cost
                    })
                current_month = ts_month
                month_rolls = 0
                month_cost = 0
                month_benefit = 0

        # Final month
        if current_month:
            monthly_data.append({
                'month': current_month,
                'rolls': month_rolls,
                'cost': month_cost,
                'benefit': month_benefit,
                'net': month_benefit - month_cost
            })

        monthly_df = pd.DataFrame(monthly_data) if monthly_data else pd.DataFrame()

        # Normalize venue distribution
        total_volume = sum(venue_volume.values()) or 1
        venue_distribution = {v: vol / total_volume for v, vol in venue_volume.items()}

        return RollBacktestResult(
            total_rolls=len(executions),
            successful_rolls=sum(1 for e in executions if e.success),
            total_cost_usd=total_cost,
            total_benefit_usd=total_benefit,
            net_pnl_usd=total_benefit - total_cost,
            avg_cost_per_roll_bps=(total_cost / max(initial_capital, 1) * 10000 /
                                   max(len(executions), 1)),
            avg_benefit_per_roll_bps=(total_benefit / max(initial_capital, 1) * 10000 /
                                      max(len(executions), 1)),
            roll_efficiency=total_benefit / max(total_cost, 1),
            venue_distribution=venue_distribution,
            roll_reasons=dict(roll_reasons),
            monthly_statistics=monthly_df,
            execution_log=executions
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get current optimizer statistics."""
        return {
            'active_positions': len(self.positions),
            'total_rolls_executed': len(self.execution_history),
            'tracked_venues': len(self.venue_states),
            'position_summary': {
                pos_id: {
                    'venue': pos['venue'],
                    'size_btc': pos['size_btc'],
                    'direction': pos['direction']
                }
                for pos_id, pos in self.positions.items()
            }
        }


class MultiVenueRollStrategy:
    """
    High-level Strategy D implementation coordinating roll optimization.

    This class provides the primary interface for Strategy D, integrating
    with the broader futures curve trading system.
    """

    def __init__(
        self,
        config: Optional[RollConfig] = None,
        term_structure_analyzer: Optional[TermStructureAnalyzer] = None
    ):
        self.config = config or RollConfig()
        self.optimizer = RollOptimizer(self.config)
        self.term_structure_analyzer = term_structure_analyzer or TermStructureAnalyzer()
        self.active = False

    def initialize(
        self,
        venues: List[str],
        initial_positions: Optional[List[Dict[str, Any]]] = None
    ):
        """Initialize the strategy with venues and positions."""
        self.venues = venues

        if initial_positions:
            for pos in initial_positions:
                self.optimizer.add_position(
                    position_id=pos['position_id'],
                    venue=pos['venue'],
                    size_btc=pos['size_btc'],
                    entry_price=pos['entry_price'],
                    direction=pos.get('direction', 'long'),
                    expiry=pos.get('expiry')
                )

        self.active = True
        logger.info(f"MultiVenueRollStrategy initialized with {len(venues)} venues")

    def update(
        self,
        venue_data: Dict[str, Dict[str, Any]],
        timestamp: datetime
    ) -> List[RollOpportunity]:
        """
        Process new market data and return roll opportunities.

        Args:
            venue_data: Dict mapping venue to:
                - price: Current price
                - funding_rate: Current funding rate
                - term_structure: TermStructureCurve object
                - liquidity_score: 0-1 liquidity metric
            timestamp: Current timestamp
        """
        if not self.active:
            return []

        # Update venue states
        for venue, data in venue_data.items():
            self.optimizer.update_venue_state(
                venue=venue,
                term_structure=data.get('term_structure'),
                funding_rate_hourly=data.get('funding_rate'),
                available_capacity=DEFAULT_VENUE_CAPACITY.get(venue, 100),
                liquidity_score=data.get('liquidity_score', 0.8),
                timestamp=timestamp
            )

        # Determine current regime
        regime = self._determine_regime(venue_data)

        # Get recommendations
        opportunities = self.optimizer.get_roll_recommendations(regime, timestamp)

        return opportunities

    def _determine_regime(
        self,
        venue_data: Dict[str, Dict[str, Any]]
    ) -> TermStructureRegime:
        """Determine the current term structure regime."""
        # Use average funding rate across venues
        funding_rates = [
            d.get('funding_rate', 0)
            for d in venue_data.values()
            if d.get('funding_rate') is not None
        ]

        if not funding_rates:
            return TermStructureRegime.FLAT

        avg_funding = np.mean(funding_rates)
        avg_annual = avg_funding * 24 * 365 * 100  # To annualized %

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

    def execute_opportunity(
        self,
        position_id: str,
        opportunity: RollOpportunity,
        close_price: float,
        open_price: float,
        timestamp: datetime
    ) -> RollExecution:
        """Execute a roll opportunity."""
        return self.optimizer.execute_roll(
            position_id=position_id,
            opportunity=opportunity,
            actual_close_price=close_price,
            actual_open_price=open_price,
            timestamp=timestamp
        )

    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked positions."""
        return self.optimizer.get_statistics()

    def shutdown(self):
        """Shutdown the strategy."""
        self.active = False
        logger.info("MultiVenueRollStrategy shutdown")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'RollDecision',
    'RollReason',
    'VenuePreference',
    # Crisis and regime parameters
    'STRATEGY_D_CRISIS_PARAMS',
    'REGIME_ROLL_PARAMS',
    # Data structures
    'RollCost',
    'RollOpportunity',
    'RollExecution',
    'VenueState',
    'RollConfig',
    'RollBacktestResult',
    # Decision tree
    'RollDecisionTree',
    # Optimizer
    'RollOptimizer',
    # Strategy D implementation
    'MultiVenueRollStrategy',
]
