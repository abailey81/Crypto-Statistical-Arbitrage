"""
Multi-Venue Analyzer - Cross-Venue Term Structure Comparison
============================================================

Part 2 Section 3.2.2 - Strategy B: Cross-Venue Calendar Arbitrage

This module provides cross-venue analysis for BTC futures,
supporting Strategy B (Cross-Venue Calendar Arbitrage) and Strategy D
(Multi-Venue Roll Optimization).

Part 2 Requirements Addressed:
- 3.1.1 Multi-venue term structure comparison (CEX + Hybrid + DEX)
- 3.2.2 Strategy B: Cross-Venue Calendar Arbitrage implementation
- 3.2.4 Strategy D: Multi-Venue Roll Optimization support
- 3.3.3 Crisis event analysis (COVID, Luna, FTX, May 2021)
- Venue cost and capacity optimization
- Regime-aware venue selection with z-score triggers

Venues Supported (per PDF Section 3.1):
- CEX: Binance, CME, Deribit
- Hybrid: Hyperliquid, dYdX V4
- DEX: GMX

Mathematical Framework:
----------------------
Cross-Venue Arbitrage P&L:

    Entry: Long Venue_A, Short Venue_B
    Spread = Basis_A - Basis_B

    P&L = (Entry_Spread - Exit_Spread) × Notional - Costs

    Z_Score = (Spread - μ_historical) / σ_historical
    Entry when |Z_Score| > 2.0 and Net_Spread > Costs

Venue Selection Score:

    Score = α × Funding_Advantage + β × Liquidity_Score - γ × Cost_Score
    Where α, β, γ are regime-dependent weights

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
    VenueType, TermStructureRegime, SpreadDirection, CurveShape,
    VenueCosts, TermStructurePoint, CrossVenueOpportunity,
    DEFAULT_VENUE_COSTS, DEFAULT_VENUE_CAPACITY
)
from .term_structure import (
    TermStructureCurve, TermStructureAnalyzer, RegimeTracker,
    CurveQuality, RegimeTransition
)
from .funding_rate_analysis import (
    FundingRateAnalyzer, FundingTermStructure,
    CrossVenueFundingSpread, VENUE_FUNDING_CONFIG,
    CRISIS_EVENTS, is_crisis_period
)

logger = logging.getLogger(__name__)


# =============================================================================
# CRISIS-ADAPTIVE PARAMETERS FOR STRATEGY B
# =============================================================================

STRATEGY_B_CRISIS_PARAMS = {
    'normal': {
        'min_z_score': 2.0,
        'min_spread_bps': 15.0,
        'max_position_mult': 1.0,
        'stop_loss_mult': 1.0,
    },
    'crisis': {
        'min_z_score': 3.0,  # Higher threshold in crisis
        'min_spread_bps': 30.0,  # Wider spreads needed
        'max_position_mult': 0.3,  # Reduce position size
        'stop_loss_mult': 2.0,  # Wider stops
    }
}

# Regime-specific weights for venue selection
VENUE_SELECTION_WEIGHTS = {
    TermStructureRegime.STEEP_CONTANGO: {
        'funding_weight': 0.4,
        'liquidity_weight': 0.35,
        'cost_weight': 0.25,
    },
    TermStructureRegime.MILD_CONTANGO: {
        'funding_weight': 0.35,
        'liquidity_weight': 0.35,
        'cost_weight': 0.30,
    },
    TermStructureRegime.FLAT: {
        'funding_weight': 0.25,
        'liquidity_weight': 0.40,
        'cost_weight': 0.35,
    },
    TermStructureRegime.MILD_BACKWARDATION: {
        'funding_weight': 0.35,
        'liquidity_weight': 0.35,
        'cost_weight': 0.30,
    },
    TermStructureRegime.STEEP_BACKWARDATION: {
        'funding_weight': 0.4,
        'liquidity_weight': 0.35,
        'cost_weight': 0.25,
    },
}


class VenueHealth(Enum):
    """Venue health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    MAINTENANCE = "maintenance"


class ArbitrageType(Enum):
    """Types of cross-venue arbitrage."""
    BASIS_ARBITRAGE = "basis_arbitrage"          # Spot-futures basis diff
    FUNDING_ARBITRAGE = "funding_arbitrage"      # Funding rate diff
    CALENDAR_ARBITRAGE = "calendar_arbitrage"    # Same expiry, diff venue
    TERM_STRUCTURE_ARBITRAGE = "term_structure"  # Curve shape exploitation
    TRIANGULAR_ARBITRAGE = "triangular"          # Three-way venue arb


@dataclass
class VenueMetrics:
    """Metrics for a single venue."""
    venue: str
    venue_type: VenueType
    timestamp: datetime

    # Price metrics
    spot_price: float
    perp_price: Optional[float] = None
    front_month_price: Optional[float] = None
    back_month_price: Optional[float] = None

    # Basis metrics
    perp_basis_bps: Optional[float] = None
    perp_basis_annualized_pct: Optional[float] = None
    front_month_basis_bps: Optional[float] = None
    back_month_basis_bps: Optional[float] = None

    # Funding metrics
    current_funding_rate: Optional[float] = None
    predicted_funding_rate: Optional[float] = None
    funding_rate_annualized_pct: Optional[float] = None

    # Term structure metrics
    term_structure: Optional[TermStructureCurve] = None
    curve_shape: CurveShape = CurveShape.FLAT
    curve_steepness: float = 0.0

    # Liquidity metrics
    bid_ask_spread_bps: float = 0.0
    depth_usd_5pct: float = 0.0
    volume_24h_usd: float = 0.0
    open_interest_usd: float = 0.0

    # Health metrics
    health: VenueHealth = VenueHealth.HEALTHY
    latency_ms: float = 0.0
    uptime_24h_pct: float = 100.0

    # Cost metrics
    costs: Optional[VenueCosts] = None
    available_capacity_btc: float = 0.0


@dataclass
class CrossVenueSpread:
    """Spread between two venues."""
    venue_a: str
    venue_b: str
    timestamp: datetime
    spread_bps: float
    spread_annualized_pct: float
    spread_type: ArbitrageType
    direction: SpreadDirection
    confidence: float
    expected_profit_bps: float
    costs_bps: float
    net_profit_bps: float
    is_actionable: bool
    notes: List[str] = field(default_factory=list)


@dataclass
class VenueRanking:
    """Ranking of venues for a specific purpose."""
    purpose: str  # e.g., "long_carry", "short_carry", "liquidity"
    timestamp: datetime
    rankings: List[Tuple[str, float]]  # (venue, score) pairs
    explanation: str


@dataclass
class CrossVenueAnalysis:
    """Complete cross-venue analysis result."""
    timestamp: datetime
    regime: TermStructureRegime
    venue_metrics: Dict[str, VenueMetrics]
    spreads: List[CrossVenueSpread]
    opportunities: List[CrossVenueOpportunity]
    rankings: Dict[str, VenueRanking]
    aggregate_metrics: Dict[str, float]
    warnings: List[str]


class MultiVenueAnalyzer:
    """
    Multi-venue analyzer for BTC futures term structure.

    Provides:
    1. Cross-venue spread monitoring
    2. Arbitrage opportunity identification
    3. Venue health and liquidity tracking
    4. Venue selection for different strategies
    5. Cost-adjusted return analysis
    """

    # Venue configurations
    VENUE_CONFIGS = {
        'binance': {
            'type': VenueType.CEX_PERPETUAL,
            'has_perp': True,
            'has_futures': True,
            'min_size_btc': 0.001,
            'max_leverage': 125
        },
        'hyperliquid': {
            'type': VenueType.HYBRID_PERPETUAL,
            'has_perp': True,
            'has_futures': False,
            'min_size_btc': 0.001,
            'max_leverage': 50
        },
        'dydx': {
            'type': VenueType.HYBRID_PERPETUAL,
            'has_perp': True,
            'has_futures': False,
            'min_size_btc': 0.01,
            'max_leverage': 20
        },
        'gmx': {
            'type': VenueType.DEX_PERPETUAL,
            'has_perp': True,
            'has_futures': False,
            'min_size_btc': 0.01,
            'max_leverage': 50
        },
        'deribit': {
            'type': VenueType.CEX_FUTURES,
            'has_perp': True,
            'has_futures': True,
            'min_size_btc': 0.1,
            'max_leverage': 50
        },
        'cme': {
            'type': VenueType.CME,
            'has_perp': False,
            'has_futures': True,
            'min_size_btc': 5.0,
            'max_leverage': 10
        }
    }

    def __init__(
        self,
        venues: Optional[List[str]] = None,
        min_spread_for_opportunity_bps: float = 10.0,
        min_confidence: float = 0.6
    ):
        """
        Initialize multi-venue analyzer.

        Args:
            venues: List of venues to analyze (None for all)
            min_spread_for_opportunity_bps: Minimum spread to flag opportunity
            min_confidence: Minimum confidence for actionable opportunities
        """
        self.venues = venues or list(self.VENUE_CONFIGS.keys())
        self.min_spread_bps = min_spread_for_opportunity_bps
        self.min_confidence = min_confidence

        self.venue_metrics: Dict[str, VenueMetrics] = {}
        self.spread_history: List[CrossVenueSpread] = []
        self.opportunity_history: List[CrossVenueOpportunity] = []

        self.funding_analyzer = FundingRateAnalyzer()
        self.term_structure_analyzer = TermStructureAnalyzer()
        self.regime_tracker = RegimeTracker()

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
        ts = pd.Timestamp(timestamp)
        is_crisis, name, severity = is_crisis_period(ts)

        self._current_crisis = name if is_crisis else None
        self._crisis_severity = severity

        if is_crisis:
            logger.warning(f"Crisis period detected: {name} (severity={severity})")

        return is_crisis, name, severity

    def get_crisis_adjusted_params(
        self,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Get crisis-adjusted parameters for Strategy B.

        Args:
            timestamp: Current timestamp

        Returns:
            Adjusted parameter dictionary
        """
        is_crisis, _, _ = self.check_crisis_state(timestamp)

        if is_crisis:
            return STRATEGY_B_CRISIS_PARAMS['crisis']
        return STRATEGY_B_CRISIS_PARAMS['normal']

    def get_regime_aware_venue_scores(
        self,
        regime: TermStructureRegime
    ) -> Dict[str, float]:
        """
        Calculate venue scores with regime-aware weighting.

        Args:
            regime: Current term structure regime

        Returns:
            Dict of venue -> score
        """
        weights = VENUE_SELECTION_WEIGHTS.get(
            regime, VENUE_SELECTION_WEIGHTS[TermStructureRegime.FLAT]
        )

        scores = {}
        for venue, metrics in self.venue_metrics.items():
            if metrics.health == VenueHealth.UNAVAILABLE:
                scores[venue] = 0.0
                continue

            # Funding component
            funding_score = 0.0
            if metrics.funding_rate_annualized_pct is not None:
                # Normalize to 0-1 scale (assume -100% to +100% range)
                funding_score = (metrics.funding_rate_annualized_pct + 100) / 200
                funding_score = max(0, min(1, funding_score))

            # Liquidity component
            liquidity_score = min(
                metrics.volume_24h_usd / 1e10 +
                metrics.depth_usd_5pct / 1e8,
                1.0
            )

            # Cost component (lower is better, so invert)
            costs = metrics.costs or VenueCosts()
            cost_score = 1.0 - min(
                (costs.taker_fee * 10000 + costs.slippage_bps) / 50,
                1.0
            )

            # Weighted score
            total_score = (
                weights['funding_weight'] * funding_score +
                weights['liquidity_weight'] * liquidity_score +
                weights['cost_weight'] * cost_score
            )

            scores[venue] = total_score

        return scores

    def update_venue_data(
        self,
        venue: str,
        spot_price: float,
        perp_price: Optional[float] = None,
        futures_prices: Optional[Dict[datetime, float]] = None,
        funding_rate: Optional[float] = None,
        bid_ask_spread_bps: float = 0.0,
        depth_usd: float = 0.0,
        volume_24h: float = 0.0,
        open_interest: float = 0.0,
        latency_ms: float = 0.0,
        timestamp: Optional[datetime] = None
    ):
        """Update data for a single venue."""
        timestamp = timestamp or datetime.now()
        venue_config = self.VENUE_CONFIGS.get(venue, {})
        venue_type = venue_config.get('type', VenueType.CEX_PERPETUAL)
        costs = DEFAULT_VENUE_COSTS.get(venue, VenueCosts())
        capacity = DEFAULT_VENUE_CAPACITY.get(venue, 100.0)

        # Calculate basis metrics
        perp_basis_bps = None
        perp_basis_annual = None
        if perp_price:
            perp_basis_bps = (perp_price - spot_price) / spot_price * 10000
            perp_basis_annual = perp_basis_bps * 365 / 100  # Annualized %

        # Calculate funding annualized
        funding_annual = None
        if funding_rate is not None:
            venue_funding_config = VENUE_FUNDING_CONFIG.get(venue)
            if venue_funding_config:
                hours_per_payment = venue_funding_config.interval.value
                funding_annual = funding_rate * (24 / hours_per_payment) * 365 * 100

        # Build term structure if futures data available
        term_structure = None
        curve_shape = CurveShape.FLAT
        steepness = 0.0

        if futures_prices:
            points = []
            for expiry, price in sorted(futures_prices.items()):
                days_to_expiry = (expiry - timestamp).days
                if days_to_expiry > 0:
                    basis_bps = (price - spot_price) / spot_price * 10000
                    annual_basis = basis_bps * 365 / days_to_expiry / 100

                    points.append(TermStructurePoint(
                        expiry=expiry,
                        days_to_expiry=days_to_expiry,
                        price=price,
                        basis_bps=basis_bps,
                        annualized_basis_pct=annual_basis,
                        volume=0,
                        open_interest=0
                    ))

            if points:
                term_structure = TermStructureCurve(
                    timestamp=timestamp,
                    venue=venue,
                    spot_price=spot_price,
                    points=points,
                    quality=CurveQuality.GOOD,
                    regime=TermStructureRegime.MILD_CONTANGO if points[0].basis_bps > 0 else TermStructureRegime.MILD_BACKWARDATION
                )

                curve_shape, steepness = self._classify_curve(points)

        # Determine health
        health = VenueHealth.HEALTHY
        if latency_ms > 1000:
            health = VenueHealth.DEGRADED
        if latency_ms > 5000 or bid_ask_spread_bps > 50:
            health = VenueHealth.UNAVAILABLE

        # Front/back month prices
        front_month_price = None
        back_month_price = None
        front_month_basis = None
        back_month_basis = None

        if futures_prices:
            sorted_expiries = sorted(futures_prices.items())
            if len(sorted_expiries) >= 1:
                front_month_price = sorted_expiries[0][1]
                front_month_basis = (front_month_price - spot_price) / spot_price * 10000
            if len(sorted_expiries) >= 2:
                back_month_price = sorted_expiries[1][1]
                back_month_basis = (back_month_price - spot_price) / spot_price * 10000

        self.venue_metrics[venue] = VenueMetrics(
            venue=venue,
            venue_type=venue_type,
            timestamp=timestamp,
            spot_price=spot_price,
            perp_price=perp_price,
            front_month_price=front_month_price,
            back_month_price=back_month_price,
            perp_basis_bps=perp_basis_bps,
            perp_basis_annualized_pct=perp_basis_annual,
            front_month_basis_bps=front_month_basis,
            back_month_basis_bps=back_month_basis,
            current_funding_rate=funding_rate,
            funding_rate_annualized_pct=funding_annual,
            term_structure=term_structure,
            curve_shape=curve_shape,
            curve_steepness=steepness,
            bid_ask_spread_bps=bid_ask_spread_bps,
            depth_usd_5pct=depth_usd,
            volume_24h_usd=volume_24h,
            open_interest_usd=open_interest,
            health=health,
            latency_ms=latency_ms,
            costs=costs,
            available_capacity_btc=capacity
        )

    def _classify_curve(
        self,
        points: List[TermStructurePoint]
    ) -> Tuple[CurveShape, float]:
        """Classify term structure curve shape."""
        if len(points) < 2:
            return CurveShape.FLAT, 0.0

        bases = [p.annualized_basis_pct for p in points]
        avg_basis = np.mean(bases)
        slope = (bases[-1] - bases[0]) / len(bases) if bases else 0

        if abs(slope) < 0.5:
            shape = CurveShape.FLAT
        elif slope > 2:
            shape = CurveShape.STEEP_CONTANGO
        elif slope > 0:
            shape = CurveShape.CONTANGO
        elif slope < -2:
            shape = CurveShape.STEEP_BACKWARDATION
        else:
            shape = CurveShape.BACKWARDATION

        # Check for humped
        if len(bases) >= 3:
            mid_idx = len(bases) // 2
            if bases[mid_idx] > bases[0] and bases[mid_idx] > bases[-1]:
                shape = CurveShape.HUMPED

        return shape, abs(slope)

    def analyze_cross_venue_spreads(
        self,
        timestamp: Optional[datetime] = None
    ) -> List[CrossVenueSpread]:
        """Analyze spreads between all venue pairs."""
        timestamp = timestamp or datetime.now()
        spreads = []

        venues = list(self.venue_metrics.keys())

        for i, venue_a in enumerate(venues):
            for venue_b in venues[i + 1:]:
                metrics_a = self.venue_metrics[venue_a]
                metrics_b = self.venue_metrics[venue_b]

                # Skip unhealthy venues
                if metrics_a.health == VenueHealth.UNAVAILABLE:
                    continue
                if metrics_b.health == VenueHealth.UNAVAILABLE:
                    continue

                # Perp basis spread
                if metrics_a.perp_basis_bps is not None and metrics_b.perp_basis_bps is not None:
                    spread = self._calculate_perp_spread(
                        venue_a, venue_b, metrics_a, metrics_b, timestamp
                    )
                    if spread:
                        spreads.append(spread)

                # Funding rate spread
                if metrics_a.funding_rate_annualized_pct is not None and \
                   metrics_b.funding_rate_annualized_pct is not None:
                    spread = self._calculate_funding_spread(
                        venue_a, venue_b, metrics_a, metrics_b, timestamp
                    )
                    if spread:
                        spreads.append(spread)

                # Calendar spread (if both have futures)
                if metrics_a.term_structure and metrics_b.term_structure:
                    spread = self._calculate_calendar_spread(
                        venue_a, venue_b, metrics_a, metrics_b, timestamp
                    )
                    if spread:
                        spreads.append(spread)

        self.spread_history.extend(spreads)
        return spreads

    def _calculate_perp_spread(
        self,
        venue_a: str,
        venue_b: str,
        metrics_a: VenueMetrics,
        metrics_b: VenueMetrics,
        timestamp: datetime
    ) -> Optional[CrossVenueSpread]:
        """Calculate perpetual basis spread between venues."""
        spread_bps = metrics_a.perp_basis_bps - metrics_b.perp_basis_bps

        # Costs for the trade
        costs_a = metrics_a.costs or VenueCosts()
        costs_b = metrics_b.costs or VenueCosts()
        total_costs_bps = (
            (costs_a.taker_fee + costs_b.taker_fee) * 10000 +
            costs_a.slippage_bps + costs_b.slippage_bps
        )

        # Direction: positive spread means short A, long B
        if spread_bps > 0:
            direction = SpreadDirection.LONG  # Long near, short far
        else:
            direction = SpreadDirection.SHORT  # Short near, long far

        # Annualized (for perps, this is instantaneous)
        annual_spread = spread_bps * 365 / 100

        # Confidence based on liquidity
        liquidity_score = min(
            metrics_a.volume_24h_usd / 1e9,
            metrics_b.volume_24h_usd / 1e9,
            1.0
        )
        health_score = 1.0 if (
            metrics_a.health == VenueHealth.HEALTHY and
            metrics_b.health == VenueHealth.HEALTHY
        ) else 0.7

        confidence = liquidity_score * health_score

        net_profit = abs(spread_bps) - total_costs_bps
        is_actionable = (
            abs(spread_bps) >= self.min_spread_bps and
            confidence >= self.min_confidence and
            net_profit > 0
        )

        return CrossVenueSpread(
            venue_a=venue_a,
            venue_b=venue_b,
            timestamp=timestamp,
            spread_bps=spread_bps,
            spread_annualized_pct=annual_spread,
            spread_type=ArbitrageType.BASIS_ARBITRAGE,
            direction=direction,
            confidence=confidence,
            expected_profit_bps=abs(spread_bps),
            costs_bps=total_costs_bps,
            net_profit_bps=net_profit,
            is_actionable=is_actionable,
            notes=[
                f"{venue_a} basis: {metrics_a.perp_basis_bps:.2f} bps",
                f"{venue_b} basis: {metrics_b.perp_basis_bps:.2f} bps"
            ]
        )

    def _calculate_funding_spread(
        self,
        venue_a: str,
        venue_b: str,
        metrics_a: VenueMetrics,
        metrics_b: VenueMetrics,
        timestamp: datetime
    ) -> Optional[CrossVenueSpread]:
        """Calculate funding rate spread between venues."""
        spread_annual = (
            metrics_a.funding_rate_annualized_pct -
            metrics_b.funding_rate_annualized_pct
        )
        spread_bps = spread_annual * 100  # Convert to bps

        costs_a = metrics_a.costs or VenueCosts()
        costs_b = metrics_b.costs or VenueCosts()
        total_costs_bps = (
            (costs_a.taker_fee + costs_b.taker_fee) * 10000 +
            costs_a.slippage_bps + costs_b.slippage_bps
        )

        direction = (
            SpreadDirection.LONG
            if spread_bps > 0 else
            SpreadDirection.SHORT
        )

        confidence = 0.8  # Funding is more predictable
        net_profit = abs(spread_bps) - total_costs_bps

        is_actionable = (
            abs(spread_bps) >= self.min_spread_bps * 2 and  # Higher threshold for funding
            confidence >= self.min_confidence and
            net_profit > 0
        )

        return CrossVenueSpread(
            venue_a=venue_a,
            venue_b=venue_b,
            timestamp=timestamp,
            spread_bps=spread_bps,
            spread_annualized_pct=spread_annual,
            spread_type=ArbitrageType.FUNDING_ARBITRAGE,
            direction=direction,
            confidence=confidence,
            expected_profit_bps=abs(spread_bps),
            costs_bps=total_costs_bps,
            net_profit_bps=net_profit,
            is_actionable=is_actionable,
            notes=[
                f"{venue_a} funding: {metrics_a.funding_rate_annualized_pct:.2f}% annual",
                f"{venue_b} funding: {metrics_b.funding_rate_annualized_pct:.2f}% annual"
            ]
        )

    def _calculate_calendar_spread(
        self,
        venue_a: str,
        venue_b: str,
        metrics_a: VenueMetrics,
        metrics_b: VenueMetrics,
        timestamp: datetime
    ) -> Optional[CrossVenueSpread]:
        """Calculate calendar spread between venues with same expiry."""
        ts_a = metrics_a.term_structure
        ts_b = metrics_b.term_structure

        if not ts_a or not ts_b or not ts_a.points or not ts_b.points:
            return None

        # Find matching expiries
        expiries_a = {p.expiry: p for p in ts_a.points}
        expiries_b = {p.expiry: p for p in ts_b.points}

        common_expiries = set(expiries_a.keys()) & set(expiries_b.keys())
        if not common_expiries:
            return None

        # Use nearest common expiry
        nearest = min(common_expiries)
        point_a = expiries_a[nearest]
        point_b = expiries_b[nearest]

        spread_bps = point_a.annualized_basis_pct - point_b.annualized_basis_pct
        spread_bps *= 100  # Convert to bps

        costs_a = metrics_a.costs or VenueCosts()
        costs_b = metrics_b.costs or VenueCosts()
        total_costs_bps = (
            (costs_a.taker_fee + costs_b.taker_fee) * 10000 +
            costs_a.slippage_bps + costs_b.slippage_bps
        )

        direction = (
            SpreadDirection.LONG
            if spread_bps > 0 else
            SpreadDirection.SHORT
        )

        confidence = 0.75
        net_profit = abs(spread_bps) - total_costs_bps

        is_actionable = (
            abs(spread_bps) >= self.min_spread_bps and
            confidence >= self.min_confidence and
            net_profit > 0
        )

        return CrossVenueSpread(
            venue_a=venue_a,
            venue_b=venue_b,
            timestamp=timestamp,
            spread_bps=spread_bps,
            spread_annualized_pct=spread_bps / 100,
            spread_type=ArbitrageType.CALENDAR_ARBITRAGE,
            direction=direction,
            confidence=confidence,
            expected_profit_bps=abs(spread_bps),
            costs_bps=total_costs_bps,
            net_profit_bps=net_profit,
            is_actionable=is_actionable,
            notes=[
                f"Expiry: {nearest.strftime('%Y-%m-%d')}",
                f"{venue_a}: {point_a.annualized_basis_pct:.2f}%",
                f"{venue_b}: {point_b.annualized_basis_pct:.2f}%"
            ]
        )

    def find_arbitrage_opportunities(
        self,
        spreads: Optional[List[CrossVenueSpread]] = None,
        max_opportunities: int = 10
    ) -> List[CrossVenueOpportunity]:
        """Find actionable arbitrage opportunities from spreads."""
        if spreads is None:
            spreads = self.analyze_cross_venue_spreads()

        opportunities = []

        for spread in spreads:
            if not spread.is_actionable:
                continue

            # Get venue metrics for sizing
            metrics_a = self.venue_metrics.get(spread.venue_a)
            metrics_b = self.venue_metrics.get(spread.venue_b)

            if not metrics_a or not metrics_b:
                continue

            # Calculate max position size
            max_size = min(
                metrics_a.available_capacity_btc,
                metrics_b.available_capacity_btc,
                metrics_a.depth_usd_5pct / max(metrics_a.spot_price, 1) * 0.1,
                metrics_b.depth_usd_5pct / max(metrics_b.spot_price, 1) * 0.1
            )

            if max_size < 0.1:  # Minimum viable size
                continue

            # Create opportunity with correct dataclass fields
            max_size_usd = max_size * metrics_a.spot_price
            venue1_basis = metrics_a.perp_basis_annualized_pct or (metrics_a.perp_basis_bps / 100 if metrics_a.perp_basis_bps else 0.0)
            venue2_basis = metrics_b.perp_basis_annualized_pct or (metrics_b.perp_basis_bps / 100 if metrics_b.perp_basis_bps else 0.0)

            opp = CrossVenueOpportunity(
                timestamp=spread.timestamp,
                venue1=spread.venue_a,
                venue2=spread.venue_b,
                venue1_type=metrics_a.venue_type,
                venue2_type=metrics_b.venue_type,
                venue1_basis_pct=venue1_basis,
                venue2_basis_pct=venue2_basis,
                basis_differential=spread.spread_bps / 100,
                z_score=spread.confidence * 2.0,  # Approximate z-score from confidence
                total_cost_bps=spread.costs_bps,
                net_differential_bps=spread.net_profit_bps,
                max_size_usd=max_size_usd,
                recommended_size_usd=max_size_usd * 0.5,  # Conservative sizing
                expected_return_bps=spread.expected_profit_bps,
                annualized_return_pct=abs(spread.spread_annualized_pct)
            )

            opportunities.append(opp)

        # Sort by net profit
        opportunities.sort(
            key=lambda x: x.annualized_return_pct - x.total_cost_bps / 100,
            reverse=True
        )

        self.opportunity_history.extend(opportunities[:max_opportunities])
        return opportunities[:max_opportunities]

    def rank_venues(
        self,
        purpose: str,
        timestamp: Optional[datetime] = None
    ) -> VenueRanking:
        """
        Rank venues for a specific trading purpose.

        Purposes:
        - 'long_carry': Best for earning positive funding/basis
        - 'short_carry': Best for earning negative funding/basis
        - 'liquidity': Best overall liquidity
        - 'cost': Lowest trading costs
        - 'capacity': Largest position capacity
        """
        timestamp = timestamp or datetime.now()
        rankings = []

        for venue, metrics in self.venue_metrics.items():
            if metrics.health == VenueHealth.UNAVAILABLE:
                continue

            score = 0.0

            if purpose == 'long_carry':
                # Higher funding = better for long carry
                if metrics.funding_rate_annualized_pct:
                    score = metrics.funding_rate_annualized_pct
                elif metrics.perp_basis_annualized_pct:
                    score = metrics.perp_basis_annualized_pct

            elif purpose == 'short_carry':
                # Lower (more negative) funding = better for short carry
                if metrics.funding_rate_annualized_pct:
                    score = -metrics.funding_rate_annualized_pct
                elif metrics.perp_basis_annualized_pct:
                    score = -metrics.perp_basis_annualized_pct

            elif purpose == 'liquidity':
                score = (
                    metrics.volume_24h_usd / 1e9 * 0.4 +
                    metrics.open_interest_usd / 1e9 * 0.3 +
                    metrics.depth_usd_5pct / 1e6 * 0.3
                )

            elif purpose == 'cost':
                costs = metrics.costs or VenueCosts()
                # Lower costs = higher score
                score = -((costs.taker_fee * 10000) + costs.slippage_bps + costs.gas_cost_usd)

            elif purpose == 'capacity':
                score = metrics.available_capacity_btc

            else:
                # General score
                score = 0.5

            rankings.append((venue, score))

        rankings.sort(key=lambda x: x[1], reverse=True)

        explanation = f"Venues ranked for {purpose} at {timestamp}"

        return VenueRanking(
            purpose=purpose,
            timestamp=timestamp,
            rankings=rankings,
            explanation=explanation
        )

    def get_comprehensive_analysis(
        self,
        timestamp: Optional[datetime] = None
    ) -> CrossVenueAnalysis:
        """Get complete cross-venue analysis."""
        timestamp = timestamp or datetime.now()

        # Determine overall regime
        funding_rates = [
            m.funding_rate_annualized_pct
            for m in self.venue_metrics.values()
            if m.funding_rate_annualized_pct is not None
        ]

        if funding_rates:
            avg_funding = np.mean(funding_rates)
            if avg_funding > 20:
                regime = TermStructureRegime.STEEP_CONTANGO
            elif avg_funding > 5:
                regime = TermStructureRegime.MILD_CONTANGO
            elif avg_funding < -20:
                regime = TermStructureRegime.STEEP_BACKWARDATION
            elif avg_funding < -5:
                regime = TermStructureRegime.MILD_BACKWARDATION
            else:
                regime = TermStructureRegime.FLAT
        else:
            regime = TermStructureRegime.FLAT

        # Analyze spreads
        spreads = self.analyze_cross_venue_spreads(timestamp)

        # Find opportunities
        opportunities = self.find_arbitrage_opportunities(spreads)

        # Generate rankings
        rankings = {
            purpose: self.rank_venues(purpose, timestamp)
            for purpose in ['long_carry', 'short_carry', 'liquidity', 'cost', 'capacity']
        }

        # Aggregate metrics
        aggregate = self._calculate_aggregate_metrics()

        # Generate warnings
        warnings = self._generate_warnings()

        return CrossVenueAnalysis(
            timestamp=timestamp,
            regime=regime,
            venue_metrics=self.venue_metrics.copy(),
            spreads=spreads,
            opportunities=opportunities,
            rankings=rankings,
            aggregate_metrics=aggregate,
            warnings=warnings
        )

    def _calculate_aggregate_metrics(self) -> Dict[str, float]:
        """Calculate aggregate cross-venue metrics."""
        metrics = self.venue_metrics.values()

        if not metrics:
            return {}

        funding_rates = [
            m.funding_rate_annualized_pct for m in metrics
            if m.funding_rate_annualized_pct is not None
        ]
        basis_rates = [
            m.perp_basis_annualized_pct for m in metrics
            if m.perp_basis_annualized_pct is not None
        ]
        volumes = [m.volume_24h_usd for m in metrics]
        spreads = [m.bid_ask_spread_bps for m in metrics]

        return {
            'avg_funding_annual_pct': np.mean(funding_rates) if funding_rates else 0,
            'funding_dispersion_pct': np.std(funding_rates) if funding_rates else 0,
            'avg_basis_annual_pct': np.mean(basis_rates) if basis_rates else 0,
            'basis_dispersion_pct': np.std(basis_rates) if basis_rates else 0,
            'total_volume_24h_usd': sum(volumes),
            'avg_bid_ask_spread_bps': np.mean(spreads) if spreads else 0,
            'venues_healthy': sum(
                1 for m in metrics if m.health == VenueHealth.HEALTHY
            ),
            'venues_total': len(list(metrics))
        }

    def _generate_warnings(self) -> List[str]:
        """Generate warnings based on current market state."""
        warnings = []

        for venue, metrics in self.venue_metrics.items():
            if metrics.health == VenueHealth.DEGRADED:
                warnings.append(f"{venue}: degraded health (latency: {metrics.latency_ms}ms)")
            if metrics.health == VenueHealth.UNAVAILABLE:
                warnings.append(f"{venue}: unavailable")
            if metrics.bid_ask_spread_bps > 20:
                warnings.append(f"{venue}: wide spread ({metrics.bid_ask_spread_bps:.1f} bps)")

        # Check for unusual funding divergence
        funding_rates = [
            (v, m.funding_rate_annualized_pct)
            for v, m in self.venue_metrics.items()
            if m.funding_rate_annualized_pct is not None
        ]
        if len(funding_rates) >= 2:
            rates = [r[1] for r in funding_rates]
            if max(rates) - min(rates) > 50:  # 50% annualized divergence
                warnings.append(
                    f"High funding divergence: {max(rates):.1f}% to {min(rates):.1f}% annual"
                )

        return warnings

    def get_optimal_execution_venue(
        self,
        trade_size_btc: float,
        direction: str,  # 'long' or 'short'
        priority: str = 'cost'  # 'cost', 'speed', 'liquidity'
    ) -> Optional[str]:
        """
        Determine optimal venue for trade execution.

        Args:
            trade_size_btc: Size of trade
            direction: Trade direction
            priority: Optimization priority
        """
        candidates = []

        for venue, metrics in self.venue_metrics.items():
            if metrics.health == VenueHealth.UNAVAILABLE:
                continue
            if metrics.available_capacity_btc < trade_size_btc:
                continue

            costs = metrics.costs or VenueCosts()
            config = self.VENUE_CONFIGS.get(venue, {})

            if trade_size_btc < config.get('min_size_btc', 0):
                continue

            # Score based on priority
            if priority == 'cost':
                score = -(costs.taker_fee * 10000 + costs.slippage_bps)
            elif priority == 'speed':
                score = -metrics.latency_ms
            elif priority == 'liquidity':
                score = metrics.depth_usd_5pct
            else:
                score = 0

            # Adjust for direction-specific factors
            if direction == 'long' and metrics.funding_rate_annualized_pct:
                # Prefer venues with higher funding for longs (earn funding)
                score += metrics.funding_rate_annualized_pct * 0.1
            elif direction == 'short' and metrics.funding_rate_annualized_pct:
                # Prefer venues with lower funding for shorts
                score -= metrics.funding_rate_annualized_pct * 0.1

            candidates.append((venue, score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def get_spread_statistics(
        self,
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """Get statistics on historical spreads."""
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        recent = [s for s in self.spread_history if s.timestamp >= cutoff]

        if not recent:
            return {}

        by_type = defaultdict(list)
        for s in recent:
            by_type[s.spread_type.value].append(s.spread_bps)

        stats = {}
        for spread_type, spreads in by_type.items():
            stats[spread_type] = {
                'count': len(spreads),
                'mean_bps': np.mean(spreads),
                'std_bps': np.std(spreads),
                'min_bps': min(spreads),
                'max_bps': max(spreads),
                'actionable_pct': sum(1 for s in recent
                                      if s.spread_type.value == spread_type and s.is_actionable) / len(spreads) * 100
            }

        return stats


class CrossVenueStrategyB:
    """
    Strategy B: Cross-Venue Calendar Arbitrage implementation.

    This class coordinates the cross-venue arbitrage strategy using
    the MultiVenueAnalyzer for opportunity identification.
    """

    def __init__(
        self,
        analyzer: Optional[MultiVenueAnalyzer] = None,
        min_spread_bps: float = 15.0,
        max_position_btc: float = 50.0,
        take_profit_bps: float = 10.0,
        stop_loss_bps: float = 20.0
    ):
        self.analyzer = analyzer or MultiVenueAnalyzer()
        self.min_spread_bps = min_spread_bps
        self.max_position_btc = max_position_btc
        self.take_profit_bps = take_profit_bps
        self.stop_loss_bps = stop_loss_bps

        self.positions: Dict[str, Dict[str, Any]] = {}
        self.closed_positions: List[Dict[str, Any]] = []

    def update_market_data(
        self,
        venue_data: Dict[str, Dict[str, Any]],
        timestamp: datetime
    ):
        """Update all venue data."""
        for venue, data in venue_data.items():
            self.analyzer.update_venue_data(
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

    def generate_signals(
        self,
        timestamp: datetime
    ) -> List[CrossVenueOpportunity]:
        """Generate trading signals from cross-venue analysis."""
        analysis = self.analyzer.get_comprehensive_analysis(timestamp)
        return [
            opp for opp in analysis.opportunities
            if opp.spread_bps >= self.min_spread_bps and
            opp.confidence >= 0.7
        ]

    def execute_trade(
        self,
        opportunity: CrossVenueOpportunity,
        size_btc: float,
        timestamp: datetime
    ) -> str:
        """Execute a cross-venue arbitrage trade."""
        size = min(size_btc, self.max_position_btc)

        position_id = f"xv_{timestamp.strftime('%Y%m%d%H%M%S')}"

        self.positions[position_id] = {
            'id': position_id,
            'opportunity': opportunity,
            'size_btc': size,
            'entry_time': timestamp,
            'entry_spread_bps': opportunity.spread_bps,
            'venue_long': opportunity.venue_long,
            'venue_short': opportunity.venue_short,
            'unrealized_pnl': 0,
            'status': 'open'
        }

        return position_id

    def update_positions(
        self,
        current_spreads: Dict[Tuple[str, str], float],
        timestamp: datetime
    ) -> List[str]:
        """Update positions and return IDs of positions to close."""
        to_close = []

        for pos_id, pos in self.positions.items():
            if pos['status'] != 'open':
                continue

            # Get current spread
            key = (pos['venue_long'], pos['venue_short'])
            current = current_spreads.get(key, pos['entry_spread_bps'])

            # Calculate P&L (spread convergence = profit)
            pnl_bps = pos['entry_spread_bps'] - current
            pos['unrealized_pnl'] = pnl_bps

            # Check exit conditions
            if pnl_bps >= self.take_profit_bps:
                to_close.append(pos_id)
            elif pnl_bps <= -self.stop_loss_bps:
                to_close.append(pos_id)

        return to_close

    def close_position(
        self,
        position_id: str,
        exit_spread_bps: float,
        timestamp: datetime
    ):
        """Close a position."""
        pos = self.positions.get(position_id)
        if not pos:
            return

        pos['status'] = 'closed'
        pos['exit_time'] = timestamp
        pos['exit_spread_bps'] = exit_spread_bps
        pos['realized_pnl_bps'] = pos['entry_spread_bps'] - exit_spread_bps

        self.closed_positions.append(pos)
        del self.positions[position_id]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get strategy performance summary."""
        if not self.closed_positions:
            return {'trades': 0, 'total_pnl_bps': 0}

        pnls = [p['realized_pnl_bps'] for p in self.closed_positions]
        wins = [p for p in pnls if p > 0]

        return {
            'trades': len(self.closed_positions),
            'total_pnl_bps': sum(pnls),
            'avg_pnl_bps': np.mean(pnls),
            'win_rate': len(wins) / len(pnls) if pnls else 0,
            'avg_win_bps': np.mean(wins) if wins else 0,
            'avg_loss_bps': np.mean([p for p in pnls if p <= 0]) if any(p <= 0 for p in pnls) else 0,
            'open_positions': len(self.positions)
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'VenueHealth',
    'ArbitrageType',
    # Crisis and regime parameters
    'STRATEGY_B_CRISIS_PARAMS',
    'VENUE_SELECTION_WEIGHTS',
    # Data structures
    'VenueMetrics',
    'CrossVenueSpread',
    'VenueRanking',
    'CrossVenueAnalysis',
    # Analyzer
    'MultiVenueAnalyzer',
    # Strategy B implementation
    'CrossVenueStrategyB',
]
