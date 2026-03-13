"""
Term Structure Analysis Module
==============================

Term structure curve construction, regime tracking,
and cross-venue comparison for BTC futures markets.

Part 2 Requirements Addressed:
- 3.1.1 Multi-venue term structure curve construction
- 3.1.2 Funding rate normalization (hourly vs 8-hour intervals)
- 3.1.3 Synthetic term structure from perpetual funding
- 3.1.4 Cross-venue basis analysis
- 3.1.5 Regime classification and tracking

Mathematical Framework
----------------------
Basis Calculation:

    Basis = (F(T) - S) / S
    Annualized_Basis = Basis × (365 / DTE)

Curve Metrics:

    Slope = dBasis/dDTE (linear regression coefficient)
    Convexity = d²Basis/dDTE² (quadratic term)

Nelson-Siegel Interpolation:

    r(τ) = β₀ + β₁[(1-e^(-τ/λ))/(τ/λ)] + β₂[(1-e^(-τ/λ))/(τ/λ) - e^(-τ/λ)]

    Where:
        β₀ = long-term level
        β₁ = short-term component
        β₂ = medium-term hump
        λ = decay parameter

Regime Classification (per PDF Section 3.1.5):

    STEEP_CONTANGO:      Annualized Basis > 20%
    MILD_CONTANGO:       5% < Annualized Basis ≤ 20%
    FLAT:               -5% ≤ Annualized Basis ≤ 5%
    MILD_BACKWARDATION: -20% ≤ Annualized Basis < -5%
    STEEP_BACKWARDATION: Annualized Basis < -20%

Funding Implied Curve:

    Implied_Price(T) = Spot × (1 + Funding_Annual × T/365)

    Where Funding_Annual = avg_funding × periods_per_year

Cross-Venue Basis Spread:

    Spread(V1, V2, T) = Basis_V1(T) - Basis_V2(T)
    Z_Score = (Spread - μ) / σ
    Signal = ENTRY if |Z_Score| > 2.0 and Net_Spread > Total_Costs

Venues Supported (per PDF):
- CEX: Binance, Deribit, CME
- Hybrid: Hyperliquid, dYdX V4
- DEX: GMX

Version: 3.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

from . import (
    TermStructureRegime, VenueType, CurveShape, InterpolationMethod,
    TermStructurePoint, DEFAULT_VENUE_COSTS
)

logger = logging.getLogger(__name__)


# =============================================================================
# ADDITIONAL ENUMERATIONS
# =============================================================================

class CurveQuality(Enum):
    """Quality assessment of term structure curve."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"
    
    @classmethod
    def from_metrics(
        cls,
        n_points: int,
        min_liquidity: float,
        max_gap_days: int
    ) -> 'CurveQuality':
        """Classify quality from curve metrics."""
        if n_points >= 5 and min_liquidity >= 0.7 and max_gap_days <= 30:
            return cls.EXCELLENT
        elif n_points >= 4 and min_liquidity >= 0.5 and max_gap_days <= 45:
            return cls.GOOD
        elif n_points >= 3 and min_liquidity >= 0.3 and max_gap_days <= 60:
            return cls.ACCEPTABLE
        elif n_points >= 2:
            return cls.POOR
        return cls.UNUSABLE
    
    @property
    def is_tradeable(self) -> bool:
        """True if quality sufficient for trading."""
        return self in [self.EXCELLENT, self.GOOD, self.ACCEPTABLE]
    
    @property
    def confidence_multiplier(self) -> float:
        """Confidence multiplier for signals."""
        multipliers = {
            self.EXCELLENT: 1.0,
            self.GOOD: 0.8,
            self.ACCEPTABLE: 0.6,
            self.POOR: 0.3,
            self.UNUSABLE: 0.0,
        }
        return multipliers.get(self, 0.0)


class RegimeTransition(Enum):
    """Type of regime transition."""
    STEEPENING = "steepening"
    FLATTENING = "flattening"
    INVERSION = "inversion"
    NORMALIZATION = "normalization"
    STABLE = "stable"
    
    @classmethod
    def classify(
        cls,
        from_regime: TermStructureRegime,
        to_regime: TermStructureRegime
    ) -> 'RegimeTransition':
        """Classify transition type."""
        if from_regime == to_regime:
            return cls.STABLE
        
        from_val = {
            TermStructureRegime.STEEP_CONTANGO: 2,
            TermStructureRegime.MILD_CONTANGO: 1,
            TermStructureRegime.FLAT: 0,
            TermStructureRegime.MILD_BACKWARDATION: -1,
            TermStructureRegime.STEEP_BACKWARDATION: -2,
        }.get(from_regime, 0)
        
        to_val = {
            TermStructureRegime.STEEP_CONTANGO: 2,
            TermStructureRegime.MILD_CONTANGO: 1,
            TermStructureRegime.FLAT: 0,
            TermStructureRegime.MILD_BACKWARDATION: -1,
            TermStructureRegime.STEEP_BACKWARDATION: -2,
        }.get(to_regime, 0)
        
        if from_val > 0 and to_val < 0:
            return cls.INVERSION
        elif from_val < 0 and to_val > 0:
            return cls.NORMALIZATION
        elif abs(to_val) > abs(from_val):
            return cls.STEEPENING
        return cls.FLATTENING
    
    @property
    def trading_implication(self) -> str:
        """Trading implication of transition."""
        implications = {
            self.STEEPENING: "Increase calendar spread positions",
            self.FLATTENING: "Reduce calendar spread positions",
            self.INVERSION: "Close longs, consider shorts",
            self.NORMALIZATION: "Close shorts, consider longs",
            self.STABLE: "Maintain current positions",
        }
        return implications.get(self, "No action")


# =============================================================================
# TERM STRUCTURE CURVE DATACLASS
# =============================================================================

@dataclass
class TermStructureCurve:
    """
    Complete BTC futures term structure curve at a point in time.
    
    Contains all contract points with analytics for regime
    classification and trading signals.
    """
    as_of: pd.Timestamp
    spot_price: float
    points: List[TermStructurePoint]
    venue: str
    
    # Calculated metrics (set in __post_init__)
    regime: TermStructureRegime = field(default=TermStructureRegime.FLAT)
    average_basis_pct: float = 0.0
    curve_slope: float = 0.0
    curve_convexity: float = 0.0
    curve_shape: CurveShape = field(default=CurveShape.FLAT)
    quality: CurveQuality = field(default=CurveQuality.POOR)
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if not self.points:
            return
        
        # Sort by DTE
        self.points = sorted(self.points, key=lambda p: p.days_to_expiry)
        
        # Calculate average basis
        basis_values = [p.annualized_basis_pct for p in self.points if p.days_to_expiry > 0]
        if basis_values:
            self.average_basis_pct = np.mean(basis_values)
        
        # Classify regime
        self.regime = TermStructureRegime.from_basis(self.average_basis_pct)
        
        # Calculate slope and convexity
        if len(self.points) >= 2:
            self._calculate_curve_metrics()
        
        # Classify shape
        self._classify_shape()
        
        # Assess quality
        self._assess_quality()
    
    def _calculate_curve_metrics(self):
        """Calculate slope and convexity using regression."""
        dtes = np.array([p.days_to_expiry for p in self.points if p.days_to_expiry > 0])
        basis = np.array([p.annualized_basis_pct for p in self.points if p.days_to_expiry > 0])
        
        if len(dtes) < 2:
            return
        
        # Linear regression for slope
        if len(dtes) >= 2:
            slope, _ = np.polyfit(dtes, basis, 1)
            self.curve_slope = slope
        
        # Quadratic for convexity
        if len(dtes) >= 3:
            coeffs = np.polyfit(dtes, basis, 2)
            self.curve_convexity = coeffs[0] * 2
    
    def _classify_shape(self):
        """Classify curve shape."""
        if len(self.points) < 2:
            self.curve_shape = CurveShape.FLAT
            return
        
        basis_values = [p.annualized_basis_pct for p in self.points]
        
        # Check monotonicity
        increasing = all(b1 <= b2 for b1, b2 in zip(basis_values[:-1], basis_values[1:]))
        decreasing = all(b1 >= b2 for b1, b2 in zip(basis_values[:-1], basis_values[1:]))
        
        if increasing:
            self.curve_shape = CurveShape.NORMAL
        elif decreasing:
            self.curve_shape = CurveShape.INVERTED
        elif self.curve_convexity > 0.01:
            self.curve_shape = CurveShape.HUMPED
        elif abs(self.curve_slope) < 0.01:
            self.curve_shape = CurveShape.FLAT
        else:
            self.curve_shape = CurveShape.KINKED
    
    def _assess_quality(self):
        """Assess curve data quality."""
        n_points = len([p for p in self.points if p.days_to_expiry > 0])
        
        liquidity_scores = [p.liquidity_score for p in self.points]
        min_liquidity = min(liquidity_scores) if liquidity_scores else 0.0
        
        # Calculate max gap between contracts
        dtes = sorted([p.days_to_expiry for p in self.points if p.days_to_expiry > 0])
        if len(dtes) >= 2:
            gaps = [dtes[i+1] - dtes[i] for i in range(len(dtes)-1)]
            max_gap = max(gaps)
        else:
            max_gap = 999
        
        self.quality = CurveQuality.from_metrics(n_points, min_liquidity, max_gap)
    
    # Basic properties
    @property
    def n_points(self) -> int:
        """Number of points on curve."""
        return len(self.points)
    
    @property
    def min_dte(self) -> int:
        """Minimum days to expiry."""
        dtes = [p.days_to_expiry for p in self.points if p.days_to_expiry > 0]
        return min(dtes) if dtes else 0
    
    @property
    def max_dte(self) -> int:
        """Maximum days to expiry."""
        dtes = [p.days_to_expiry for p in self.points]
        return max(dtes) if dtes else 0
    
    @property
    def dte_range(self) -> int:
        """Range of days to expiry covered."""
        return self.max_dte - self.min_dte
    
    # Contract accessors
    @property
    def front_month_point(self) -> Optional[TermStructurePoint]:
        """Get front month contract (excluding perpetual)."""
        dated = [p for p in self.points if p.days_to_expiry > 0]
        return dated[0] if dated else None
    
    @property
    def back_month_point(self) -> Optional[TermStructurePoint]:
        """Get back month contract."""
        dated = [p for p in self.points if p.days_to_expiry > 0]
        return dated[-1] if len(dated) > 1 else None
    
    @property
    def front_month_basis(self) -> float:
        """Front month annualized basis."""
        point = self.front_month_point
        return point.annualized_basis_pct if point else 0.0
    
    @property
    def back_month_basis(self) -> float:
        """Back month annualized basis."""
        point = self.back_month_point
        return point.annualized_basis_pct if point else 0.0
    
    @property
    def calendar_spread(self) -> float:
        """Calendar spread (back - front basis)."""
        return self.back_month_basis - self.front_month_basis
    
    @property
    def term_spread(self) -> float:
        """Term spread in percentage points."""
        return self.calendar_spread
    
    # Liquidity properties
    @property
    def average_liquidity(self) -> float:
        """Average liquidity score across points."""
        scores = [p.liquidity_score for p in self.points]
        return np.mean(scores) if scores else 0.0
    
    @property
    def min_liquidity(self) -> float:
        """Minimum liquidity score."""
        scores = [p.liquidity_score for p in self.points]
        return min(scores) if scores else 0.0
    
    @property
    def total_open_interest(self) -> float:
        """Total open interest across contracts."""
        return sum(p.open_interest or 0 for p in self.points)
    
    @property
    def total_volume(self) -> float:
        """Total 24h volume across contracts."""
        return sum(p.volume_24h or 0 for p in self.points)
    
    # Perpetual properties
    @property
    def has_perpetual(self) -> bool:
        """True if curve includes perpetual contract."""
        return any(p.is_perpetual for p in self.points)
    
    @property
    def perpetual_point(self) -> Optional[TermStructurePoint]:
        """Get perpetual contract point."""
        perps = [p for p in self.points if p.is_perpetual]
        return perps[0] if perps else None
    
    @property
    def perpetual_funding_annual(self) -> float:
        """Annualized perpetual funding rate."""
        perp = self.perpetual_point
        if perp and perp.funding_rate:
            return perp.funding_rate * perp.venue_type.periods_per_year
        return 0.0
    
    @property
    def funding_vs_basis_spread(self) -> float:
        """Spread between funding implied and actual basis."""
        return self.perpetual_funding_annual - self.front_month_basis
    
    # Methods
    def get_point_by_dte(self, target_dte: int, tolerance: int = 5) -> Optional[TermStructurePoint]:
        """Get point closest to target DTE within tolerance."""
        for point in self.points:
            if abs(point.days_to_expiry - target_dte) <= tolerance:
                return point
        return None
    
    def get_point_by_contract(self, contract: str) -> Optional[TermStructurePoint]:
        """Get point by contract name."""
        for point in self.points:
            if point.contract == contract:
                return point
        return None
    
    def interpolate_basis(
        self,
        target_dte: int,
        method: InterpolationMethod = InterpolationMethod.PCHIP
    ) -> float:
        """
        Interpolate basis at target DTE.

        Args:
            target_dte: Target days to expiry
            method: Interpolation method

        Returns:
            Interpolated annualized basis
        """
        dated = [p for p in self.points if p.days_to_expiry > 0]
        if not dated:
            return 0.0

        if len(dated) == 1:
            return dated[0].annualized_basis_pct

        dtes = np.array([p.days_to_expiry for p in dated])
        basis = np.array([p.annualized_basis_pct for p in dated])

        # Clamp target to data range for non-NS methods
        if method != InterpolationMethod.NELSON_SIEGEL:
            target_dte = max(min(target_dte, dtes.max()), dtes.min())

        if method == InterpolationMethod.LINEAR:
            return float(np.interp(target_dte, dtes, basis))

        elif method in [InterpolationMethod.CUBIC, InterpolationMethod.PCHIP]:
            try:
                from scipy.interpolate import PchipInterpolator, CubicSpline
                if method == InterpolationMethod.PCHIP:
                    interp = PchipInterpolator(dtes, basis)
                else:
                    interp = CubicSpline(dtes, basis)
                return float(interp(target_dte))
            except ImportError:
                return float(np.interp(target_dte, dtes, basis))

        elif method == InterpolationMethod.NELSON_SIEGEL:
            return self._nelson_siegel_interpolate(target_dte, dtes, basis)

        return float(np.interp(target_dte, dtes, basis))

    def _nelson_siegel_interpolate(
        self,
        target_dte: int,
        dtes: np.ndarray,
        basis: np.ndarray
    ) -> float:
        """
        Interpolate using Nelson-Siegel model.

        Nelson-Siegel formula:
            r(τ) = β₀ + β₁[(1-e^(-τ/λ))/(τ/λ)] + β₂[(1-e^(-τ/λ))/(τ/λ) - e^(-τ/λ)]

        Where τ is time to maturity, λ is decay factor (typically 1-2 years)

        Args:
            target_dte: Target days to expiry
            dtes: Array of DTEs for known points
            basis: Array of basis values

        Returns:
            Interpolated basis using Nelson-Siegel
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            # Fall back to linear if scipy not available
            return float(np.interp(target_dte, dtes, basis))

        # Convert DTE to years
        tau = dtes / 365.0
        target_tau = target_dte / 365.0

        def ns_factor1(t, lam):
            """First Nelson-Siegel factor."""
            if t == 0:
                return 1.0
            return (1 - np.exp(-t / lam)) / (t / lam)

        def ns_factor2(t, lam):
            """Second Nelson-Siegel factor."""
            if t == 0:
                return 0.0
            return ns_factor1(t, lam) - np.exp(-t / lam)

        def ns_curve(t, beta0, beta1, beta2, lam):
            """Nelson-Siegel curve function."""
            return beta0 + beta1 * ns_factor1(t, lam) + beta2 * ns_factor2(t, lam)

        def objective(params):
            """Minimize sum of squared errors."""
            beta0, beta1, beta2, lam = params
            if lam <= 0.01:
                return 1e10
            predicted = np.array([ns_curve(t, beta0, beta1, beta2, lam) for t in tau])
            return np.sum((predicted - basis) ** 2)

        # Initial parameter guess
        beta0_init = basis[-1] if len(basis) > 0 else 0.0
        beta1_init = basis[0] - beta0_init if len(basis) > 0 else 0.0
        beta2_init = 0.0
        lam_init = 1.0  # 1 year decay

        # Optimize with bounds
        result = minimize(
            objective,
            x0=[beta0_init, beta1_init, beta2_init, lam_init],
            method='L-BFGS-B',
            bounds=[(None, None), (None, None), (None, None), (0.1, 10.0)]
        )

        if result.success:
            beta0, beta1, beta2, lam = result.x
            return float(ns_curve(target_tau, beta0, beta1, beta2, lam))

        # Fall back to linear if optimization fails
        return float(np.interp(target_dte, dtes, basis))

    def get_nelson_siegel_params(self) -> Dict[str, float]:
        """
        Fit Nelson-Siegel model and return parameters.

        Returns:
            Dict with β₀, β₁, β₂, λ parameters and fit quality metrics
        """
        dated = [p for p in self.points if p.days_to_expiry > 0]
        if len(dated) < 3:
            return {'beta0': 0.0, 'beta1': 0.0, 'beta2': 0.0, 'lambda': 1.0, 'r_squared': 0.0}

        dtes = np.array([p.days_to_expiry for p in dated])
        basis = np.array([p.annualized_basis_pct for p in dated])

        try:
            from scipy.optimize import minimize
        except ImportError:
            return {'beta0': basis.mean(), 'beta1': 0.0, 'beta2': 0.0, 'lambda': 1.0, 'r_squared': 0.0}

        tau = dtes / 365.0

        def ns_factor1(t, lam):
            if t == 0:
                return 1.0
            return (1 - np.exp(-t / lam)) / (t / lam)

        def ns_factor2(t, lam):
            if t == 0:
                return 0.0
            return ns_factor1(t, lam) - np.exp(-t / lam)

        def ns_curve(t, beta0, beta1, beta2, lam):
            return beta0 + beta1 * ns_factor1(t, lam) + beta2 * ns_factor2(t, lam)

        def objective(params):
            beta0, beta1, beta2, lam = params
            if lam <= 0.01:
                return 1e10
            predicted = np.array([ns_curve(t, beta0, beta1, beta2, lam) for t in tau])
            return np.sum((predicted - basis) ** 2)

        result = minimize(
            objective,
            x0=[basis[-1], basis[0] - basis[-1], 0.0, 1.0],
            method='L-BFGS-B',
            bounds=[(None, None), (None, None), (None, None), (0.1, 10.0)]
        )

        if result.success:
            beta0, beta1, beta2, lam = result.x

            # Calculate R-squared
            predicted = np.array([ns_curve(t, beta0, beta1, beta2, lam) for t in tau])
            ss_res = np.sum((basis - predicted) ** 2)
            ss_tot = np.sum((basis - basis.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            return {
                'beta0': round(beta0, 4),
                'beta1': round(beta1, 4),
                'beta2': round(beta2, 4),
                'lambda': round(lam, 4),
                'r_squared': round(r_squared, 4),
                'interpretation': {
                    'long_term_level': beta0,
                    'short_term_slope': -beta1,
                    'medium_term_hump': beta2,
                    'decay_speed_years': lam
                }
            }

        return {'beta0': basis.mean(), 'beta1': 0.0, 'beta2': 0.0, 'lambda': 1.0, 'r_squared': 0.0}
    
    def calculate_roll_cost(
        self,
        from_dte: int,
        to_dte: int
    ) -> float:
        """
        Calculate cost of rolling from one expiry to another.
        
        Args:
            from_dte: Current contract DTE
            to_dte: Target contract DTE
            
        Returns:
            Roll cost in basis points
        """
        from_basis = self.interpolate_basis(from_dte)
        to_basis = self.interpolate_basis(to_dte)
        
        return to_basis - from_basis
    
    def find_optimal_roll_dte(
        self,
        current_dte: int,
        min_dte: int = 7,
        max_dte: int = 90
    ) -> Tuple[int, float]:
        """
        Find optimal DTE to roll to based on cost.
        
        Args:
            current_dte: Current contract DTE
            min_dte: Minimum acceptable DTE
            max_dte: Maximum acceptable DTE
            
        Returns:
            Tuple of (optimal_dte, roll_cost)
        """
        candidates = [p for p in self.points 
                     if min_dte <= p.days_to_expiry <= max_dte
                     and p.days_to_expiry > current_dte]
        
        if not candidates:
            return current_dte, 0.0
        
        best_dte = current_dte
        best_cost = float('inf')
        
        for point in candidates:
            cost = abs(self.calculate_roll_cost(current_dte, point.days_to_expiry))
            if cost < best_cost:
                best_cost = cost
                best_dte = point.days_to_expiry
        
        return best_dte, best_cost
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert curve to DataFrame."""
        return pd.DataFrame([p.to_dict() for p in self.points])
    
    def summary(self) -> Dict[str, Any]:
        """Get curve summary."""
        return {
            'as_of': self.as_of,
            'venue': self.venue,
            'spot_price': self.spot_price,
            'n_points': self.n_points,
            'regime': self.regime.value,
            'average_basis_pct': round(self.average_basis_pct, 2),
            'calendar_spread': round(self.calendar_spread, 2),
            'curve_slope': round(self.curve_slope, 4),
            'curve_shape': self.curve_shape.value,
            'quality': self.quality.value,
            'min_dte': self.min_dte,
            'max_dte': self.max_dte,
        }


# =============================================================================
# FUNDING IMPLIED CURVE
# =============================================================================

@dataclass
class FundingImpliedCurve:
    """
    Synthetic term structure implied by perpetual funding rates.
    
    Useful for comparing actual futures curve to funding-implied
    fair values and identifying arbitrage opportunities.
    """
    timestamp: pd.Timestamp
    venue: str
    venue_type: VenueType
    spot_price: float
    avg_funding_rate: float
    recent_funding_rate: float
    funding_volatility: float
    annual_funding_pct: float
    recent_annual_pct: float
    implied_points: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        """Generate implied curve points."""
        if not self.implied_points:
            self._generate_implied_points()
    
    def _generate_implied_points(self):
        """Generate implied price points for standard tenors."""
        tenors = [7, 14, 30, 60, 90, 180]
        
        for dte in tenors:
            # Blend recent and historical funding
            if dte <= 7:
                funding = self.recent_annual_pct
            elif dte >= 30:
                funding = self.annual_funding_pct
            else:
                weight = (dte - 7) / 23
                funding = (1 - weight) * self.recent_annual_pct + weight * self.annual_funding_pct
            
            implied_price = self.spot_price * (1 + funding / 100 * dte / 365)
            implied_basis = (implied_price - self.spot_price) / self.spot_price * 100
            annualized_basis = implied_basis * (365 / dte)
            
            self.implied_points.append({
                'dte': dte,
                'implied_price': implied_price,
                'implied_basis_pct': implied_basis,
                'annualized_basis_pct': annualized_basis,
                'funding_used_pct': funding,
            })
    
    @property
    def funding_premium(self) -> float:
        """Premium of recent funding over historical."""
        return self.recent_annual_pct - self.annual_funding_pct
    
    @property
    def implied_regime(self) -> TermStructureRegime:
        """Implied regime from funding."""
        return TermStructureRegime.from_basis(self.annual_funding_pct)
    
    @property
    def is_elevated(self) -> bool:
        """True if funding is elevated (>15% annualized)."""
        return self.annual_funding_pct > 15.0
    
    @property
    def is_depressed(self) -> bool:
        """True if funding is depressed (<0%)."""
        return self.annual_funding_pct < 0.0
    
    def get_implied_price(self, dte: int) -> float:
        """Get implied price for DTE."""
        for point in self.implied_points:
            if point['dte'] == dte:
                return point['implied_price']
        
        # Interpolate
        funding = self.annual_funding_pct
        return self.spot_price * (1 + funding / 100 * dte / 365)
    
    def get_implied_basis(self, dte: int) -> float:
        """Get implied basis for DTE."""
        implied_price = self.get_implied_price(dte)
        return ((implied_price - self.spot_price) / self.spot_price) * 100
    
    def get_implied_annual_basis(self, dte: int) -> float:
        """Get annualized implied basis."""
        basis = self.get_implied_basis(dte)
        return basis * (365 / max(dte, 1))
    
    def compare_to_actual(
        self,
        actual_curve: TermStructureCurve,
        target_dte: int = 30
    ) -> Dict[str, Any]:
        """
        Compare funding-implied to actual futures curve.
        
        Args:
            actual_curve: Actual term structure curve
            target_dte: DTE for comparison
            
        Returns:
            Comparison analysis dict
        """
        implied_basis = self.get_implied_annual_basis(target_dte)
        actual_basis = actual_curve.interpolate_basis(target_dte)
        
        differential = actual_basis - implied_basis
        
        return {
            'target_dte': target_dte,
            'implied_basis_pct': implied_basis,
            'actual_basis_pct': actual_basis,
            'differential_pct': differential,
            'is_rich': differential > 2.0,
            'is_cheap': differential < -2.0,
            'arbitrage_signal': 'short_futures' if differential > 5.0 else (
                'long_futures' if differential < -5.0 else 'neutral'
            ),
        }


# =============================================================================
# REGIME TRACKER
# =============================================================================

class RegimeTracker:
    """
    Track term structure regime over time.
    
    Detects regime transitions and calculates regime statistics
    for trading signal generation.
    """
    
    def __init__(self, lookback: int = 30, min_persistence: int = 3):
        """
        Initialize regime tracker.
        
        Args:
            lookback: Lookback window for statistics
            min_persistence: Minimum periods for regime confirmation
        """
        self.lookback = lookback
        self.min_persistence = min_persistence
        self.regime_history: List[Tuple[pd.Timestamp, TermStructureRegime]] = []
        self._transition_counts: Dict[Tuple[TermStructureRegime, TermStructureRegime], int] = {}
    
    @property
    def current_regime(self) -> Optional[TermStructureRegime]:
        """Current regime."""
        return self.regime_history[-1][1] if self.regime_history else None
    
    @property
    def previous_regime(self) -> Optional[TermStructureRegime]:
        """Previous regime."""
        if len(self.regime_history) >= 2:
            return self.regime_history[-2][1]
        return None
    
    def update(self, timestamp: pd.Timestamp, regime: TermStructureRegime):
        """Add new regime observation."""
        # Track transition
        if self.current_regime and self.current_regime != regime:
            key = (self.current_regime, regime)
            self._transition_counts[key] = self._transition_counts.get(key, 0) + 1
        
        self.regime_history.append((timestamp, regime))
        
        # Trim history
        if len(self.regime_history) > self.lookback * 10:
            self.regime_history = self.regime_history[-self.lookback * 5:]
    
    def get_regime_duration(self) -> int:
        """Get current regime duration in periods."""
        if not self.regime_history:
            return 0
        
        current = self.current_regime
        duration = 0
        
        for _, regime in reversed(self.regime_history):
            if regime == current:
                duration += 1
            else:
                break
        
        return duration
    
    def is_regime_confirmed(self) -> bool:
        """True if current regime has persisted minimum periods."""
        return self.get_regime_duration() >= self.min_persistence
    
    def get_regime_distribution(self) -> Dict[str, float]:
        """Get distribution of regimes in lookback."""
        if not self.regime_history:
            return {}
        
        recent = self.regime_history[-self.lookback:]
        total = len(recent)
        
        dist: Dict[str, float] = {}
        for _, regime in recent:
            name = regime.value
            dist[name] = dist.get(name, 0) + 1
        
        return {k: v / total for k, v in dist.items()}
    
    def detect_transition(self) -> Optional[RegimeTransition]:
        """Detect if regime transition occurred."""
        if len(self.regime_history) < 2:
            return None
        
        prev = self.previous_regime
        curr = self.current_regime
        
        if prev == curr:
            return None
        
        return RegimeTransition.classify(prev, curr)
    
    def get_transition_type(self) -> Optional[RegimeTransition]:
        """Get type of most recent transition."""
        return self.detect_transition()
    
    def get_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get empirical transition probability matrix."""
        matrix: Dict[str, Dict[str, float]] = {}
        
        for (from_r, to_r), count in self._transition_counts.items():
            from_name = from_r.value
            to_name = to_r.value
            
            if from_name not in matrix:
                matrix[from_name] = {}
            
            matrix[from_name][to_name] = count
        
        # Normalize to probabilities
        for from_name in matrix:
            total = sum(matrix[from_name].values())
            if total > 0:
                matrix[from_name] = {k: v / total for k, v in matrix[from_name].items()}
        
        return matrix
    
    def expected_regime_duration(self, regime: TermStructureRegime) -> float:
        """Expected duration of regime based on history."""
        durations = []
        current_duration = 0
        current_regime = None
        
        for _, r in self.regime_history:
            if r == current_regime:
                current_duration += 1
            else:
                if current_regime == regime and current_duration > 0:
                    durations.append(current_duration)
                current_regime = r
                current_duration = 1
        
        return np.mean(durations) if durations else self.min_persistence
    
    def get_regime_momentum(self) -> float:
        """
        Calculate regime momentum (-1 to 1).
        
        Positive: Moving toward contango
        Negative: Moving toward backwardation
        """
        if len(self.regime_history) < 5:
            return 0.0
        
        recent = self.regime_history[-5:]
        
        values = []
        for _, regime in recent:
            val = {
                TermStructureRegime.STEEP_CONTANGO: 2,
                TermStructureRegime.MILD_CONTANGO: 1,
                TermStructureRegime.FLAT: 0,
                TermStructureRegime.MILD_BACKWARDATION: -1,
                TermStructureRegime.STEEP_BACKWARDATION: -2,
            }.get(regime, 0)
            values.append(val)
        
        if len(values) < 2:
            return 0.0
        
        # Simple trend
        trend = (values[-1] - values[0]) / 4
        return max(-1.0, min(1.0, trend))
    
    def summary(self) -> Dict[str, Any]:
        """Get tracker summary."""
        return {
            'current_regime': self.current_regime.value if self.current_regime else None,
            'regime_duration': self.get_regime_duration(),
            'is_confirmed': self.is_regime_confirmed(),
            'regime_momentum': round(self.get_regime_momentum(), 2),
            'distribution': self.get_regime_distribution(),
            'total_observations': len(self.regime_history),
        }


# =============================================================================
# TERM STRUCTURE ANALYZER
# =============================================================================

class TermStructureAnalyzer:
    """
    Multi-venue term structure construction and analysis.
    
    Builds term structure curves from raw data, tracks regimes,
    and generates cross-venue comparison analytics.
    """
    
    DEFAULT_THRESHOLDS = {
        'steep_contango': 20.0,
        'mild_contango': 5.0,
        'flat': -5.0,
        'mild_backwardation': -20.0,
    }
    
    VENUE_CONFIG = {
        'binance': {'type': VenueType.CEX_FUTURES, 'funding_interval': None, 'min_dte_roll': 7},
        'binance_perp': {'type': VenueType.CEX_PERPETUAL, 'funding_interval': 8, 'min_dte_roll': 0},
        'bybit': {'type': VenueType.CEX_FUTURES, 'funding_interval': None, 'min_dte_roll': 7},
        'okx': {'type': VenueType.CEX_FUTURES, 'funding_interval': None, 'min_dte_roll': 7},
        'cme': {'type': VenueType.CME_FUTURES, 'funding_interval': None, 'min_dte_roll': 5},
        'deribit': {'type': VenueType.CEX_FUTURES, 'funding_interval': None, 'min_dte_roll': 3},
        'hyperliquid': {'type': VenueType.HYBRID_PERPETUAL, 'funding_interval': 1, 'min_dte_roll': 0},
        'dydx': {'type': VenueType.HYBRID_PERPETUAL, 'funding_interval': 1, 'min_dte_roll': 0},
        'gmx': {'type': VenueType.DEX_PERPETUAL, 'funding_interval': 1, 'min_dte_roll': 0},
    }
    
    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize analyzer.
        
        Args:
            thresholds: Custom regime thresholds
        """
        self.thresholds = {**self.DEFAULT_THRESHOLDS}
        if thresholds:
            self.thresholds.update(thresholds)
        
        self._regime_trackers: Dict[str, RegimeTracker] = {}
        self._basis_history: Dict[str, List[float]] = {}
    
    def calculate_basis(
        self,
        futures_price: float,
        spot_price: float,
        days_to_expiry: int
    ) -> Dict[str, float]:
        """
        Calculate basis metrics.
        
        Args:
            futures_price: Futures price
            spot_price: Spot price
            days_to_expiry: Days to expiry
            
        Returns:
            Dict with basis metrics
        """
        if spot_price <= 0:
            return {'basis_absolute': 0, 'basis_pct': 0, 'annualized_pct': 0, 'daily_pct': 0, 'implied_rate': 0}
        
        basis_absolute = futures_price - spot_price
        basis_pct = (basis_absolute / spot_price) * 100
        
        dte = max(days_to_expiry, 1)
        annualized_pct = basis_pct * (365 / dte)
        daily_pct = basis_pct / dte
        implied_rate = annualized_pct / 100
        
        return {
            'basis_absolute': basis_absolute,
            'basis_pct': basis_pct,
            'annualized_pct': annualized_pct,
            'daily_pct': daily_pct,
            'implied_rate': implied_rate,
        }
    
    def classify_regime(self, annualized_basis: float) -> TermStructureRegime:
        """Classify regime from annualized basis."""
        return TermStructureRegime.from_basis(annualized_basis)
    
    def build_curve(
        self,
        data: pd.DataFrame,
        venue: str,
        as_of: pd.Timestamp,
        spot_price: float,
        min_volume: float = 0
    ) -> TermStructureCurve:
        """
        Build term structure curve from data.
        
        Args:
            data: DataFrame with columns: contract, expiry, price, dte, volume, oi
            venue: Venue name
            as_of: Timestamp for curve
            spot_price: Current spot price
            min_volume: Minimum 24h volume filter
            
        Returns:
            TermStructureCurve object
        """
        venue_config = self.VENUE_CONFIG.get(venue.lower(), {
            'type': VenueType.CEX_FUTURES,
            'funding_interval': None,
            'min_dte_roll': 7
        })
        venue_type = venue_config['type']
        
        # Filter data
        df = data.copy()
        if 'volume' in df.columns and min_volume > 0:
            df = df[df['volume'] >= min_volume]
        
        # Build points
        points = []
        for _, row in df.iterrows():
            expiry = pd.Timestamp(row.get('expiry')) if row.get('expiry') else None
            dte = int(row.get('dte', 0))
            
            point = TermStructurePoint(
                timestamp=as_of,
                contract=str(row.get('contract', '')),
                expiry=expiry,
                days_to_expiry=dte,
                futures_price=float(row.get('price', row.get('close', 0))),
                spot_price=spot_price,
                venue=venue,
                venue_type=venue_type,
                open_interest=float(row.get('oi', row.get('open_interest', 0))) if 'oi' in row or 'open_interest' in row else None,
                volume_24h=float(row.get('volume', 0)) if 'volume' in row else None,
                funding_rate=float(row.get('funding_rate', 0)) if 'funding_rate' in row else None,
            )
            points.append(point)
        
        # Create curve
        curve = TermStructureCurve(
            as_of=as_of,
            spot_price=spot_price,
            points=points,
            venue=venue,
        )
        
        # Update regime tracker
        if venue not in self._regime_trackers:
            self._regime_trackers[venue] = RegimeTracker()
        self._regime_trackers[venue].update(as_of, curve.regime)
        
        # Update basis history
        if venue not in self._basis_history:
            self._basis_history[venue] = []
        self._basis_history[venue].append(curve.average_basis_pct)
        if len(self._basis_history[venue]) > 100:
            self._basis_history[venue] = self._basis_history[venue][-100:]
        
        return curve
    
    def build_funding_implied_curve(
        self,
        funding_rates: pd.Series,
        venue: str,
        venue_type: VenueType,
        spot_price: float,
        timestamp: pd.Timestamp
    ) -> FundingImpliedCurve:
        """
        Build funding-implied term structure.
        
        Args:
            funding_rates: Series of funding rates
            venue: Venue name
            venue_type: Venue type
            spot_price: Current spot price
            timestamp: Current timestamp
            
        Returns:
            FundingImpliedCurve object
        """
        periods_per_year = venue_type.periods_per_year
        
        avg_funding = funding_rates.mean()
        recent_funding = funding_rates.tail(7).mean() if len(funding_rates) >= 7 else avg_funding
        funding_vol = funding_rates.std()
        
        annual_funding = avg_funding * periods_per_year * 100
        recent_annual = recent_funding * periods_per_year * 100
        
        return FundingImpliedCurve(
            timestamp=timestamp,
            venue=venue,
            venue_type=venue_type,
            spot_price=spot_price,
            avg_funding_rate=avg_funding,
            recent_funding_rate=recent_funding,
            funding_volatility=funding_vol,
            annual_funding_pct=annual_funding,
            recent_annual_pct=recent_annual,
        )
    
    def compare_curves(
        self,
        curve1: TermStructureCurve,
        curve2: TermStructureCurve,
        target_dte: int = 30
    ) -> Dict[str, Any]:
        """
        Compare two term structure curves.
        
        Args:
            curve1: First curve
            curve2: Second curve
            target_dte: DTE for comparison
            
        Returns:
            Comparison analysis dict
        """
        basis1 = curve1.interpolate_basis(target_dte)
        basis2 = curve2.interpolate_basis(target_dte)
        
        differential = basis1 - basis2
        
        # Z-score from history
        pair_key = f"{curve1.venue}_{curve2.venue}"
        if pair_key in self._basis_history:
            history = self._basis_history[pair_key]
            if len(history) >= 5:
                mean = np.mean(history)
                std = np.std(history)
                z_score = (differential - mean) / std if std > 0 else 0.0
            else:
                z_score = 0.0
        else:
            z_score = 0.0
            self._basis_history[pair_key] = []
        
        self._basis_history[pair_key].append(differential)
        
        # Estimate costs
        costs1 = DEFAULT_VENUE_COSTS.get(curve1.venue.lower())
        costs2 = DEFAULT_VENUE_COSTS.get(curve2.venue.lower())
        
        cost1_bps = costs1.round_trip_taker_bps if costs1 else 20.0
        cost2_bps = costs2.round_trip_taker_bps if costs2 else 20.0
        total_cost_bps = cost1_bps + cost2_bps
        
        net_differential_bps = abs(differential) * 100 - total_cost_bps
        
        # Direction
        if differential > 0:
            direction = f"Short {curve1.venue}, Long {curve2.venue}"
        else:
            direction = f"Long {curve1.venue}, Short {curve2.venue}"
        
        return {
            'venue1': curve1.venue,
            'venue2': curve2.venue,
            'target_dte': target_dte,
            'basis1_pct': basis1,
            'basis2_pct': basis2,
            'differential_pct': differential,
            'differential_bps': differential * 100,
            'z_score': z_score,
            'total_cost_bps': total_cost_bps,
            'net_differential_bps': net_differential_bps,
            'is_profitable': net_differential_bps > 10,
            'direction': direction,
            'arbitrage_signal': 'entry' if net_differential_bps > 20 and abs(z_score) > 2 else 'hold',
        }
    
    def build_historical_curves(
        self,
        data: pd.DataFrame,
        venue: str,
        spot_prices: pd.Series,
        freq: str = 'D'
    ) -> List[TermStructureCurve]:
        """
        Build curves for historical date range.
        
        Args:
            data: DataFrame with timestamp, contract, price columns
            venue: Venue name
            spot_prices: Series of spot prices indexed by timestamp
            freq: Frequency ('D' for daily, 'H' for hourly)
            
        Returns:
            List of TermStructureCurve objects
        """
        curves = []
        
        # Group by date/hour
        data = data.copy()
        data['period'] = data['timestamp'].dt.floor(freq)
        
        for period, group in data.groupby('period'):
            if period not in spot_prices.index:
                continue
            
            spot = spot_prices.loc[period]
            curve = self.build_curve(group, venue, period, spot)
            curves.append(curve)
        
        return curves
    
    def curves_to_dataframe(
        self,
        curves: List[TermStructureCurve]
    ) -> pd.DataFrame:
        """Convert list of curves to summary DataFrame."""
        return pd.DataFrame([c.summary() for c in curves])

    def build_multi_venue_composite(
        self,
        venue_curves: Dict[str, TermStructureCurve],
        timestamp: pd.Timestamp,
        weighting: str = 'liquidity'
    ) -> TermStructureCurve:
        """
        Build composite term structure from multiple venues.

        Per PDF Section 3.1.1: Multi-venue term structure construction
        combining CEX + Hybrid + DEX venues with appropriate weighting.

        Args:
            venue_curves: Dict mapping venue name to TermStructureCurve
            timestamp: As-of timestamp
            weighting: Weighting method ('liquidity', 'equal', 'capacity')

        Returns:
            Composite TermStructureCurve
        """
        if not venue_curves:
            return None

        # Collect all unique DTEs across venues
        all_dtes = set()
        for curve in venue_curves.values():
            for point in curve.points:
                if point.days_to_expiry > 0:
                    all_dtes.add(point.days_to_expiry)

        all_dtes = sorted(all_dtes)
        if not all_dtes:
            return list(venue_curves.values())[0]

        # Calculate weights per venue
        weights = {}
        total_weight = 0.0

        for venue, curve in venue_curves.items():
            if weighting == 'liquidity':
                w = curve.average_liquidity
            elif weighting == 'capacity':
                cap = DEFAULT_VENUE_COSTS.get(venue.lower())
                w = cap.venue_type.capacity_usd / 1e9 if cap else 0.1
            else:  # equal
                w = 1.0

            weights[venue] = w
            total_weight += w

        # Normalize weights
        if total_weight > 0:
            weights = {v: w / total_weight for v, w in weights.items()}

        # Build composite points
        composite_points = []
        avg_spot = np.mean([c.spot_price for c in venue_curves.values()])

        for dte in all_dtes:
            weighted_basis = 0.0
            weighted_price = 0.0
            total_oi = 0.0
            total_vol = 0.0
            contributing_venues = []

            for venue, curve in venue_curves.items():
                w = weights.get(venue, 0)
                basis = curve.interpolate_basis(dte)
                weighted_basis += basis * w

                # Find closest actual point
                closest_point = curve.get_point_by_dte(dte, tolerance=15)
                if closest_point:
                    weighted_price += closest_point.futures_price * w
                    total_oi += closest_point.open_interest or 0
                    total_vol += closest_point.volume_24h or 0
                    contributing_venues.append(venue)
                else:
                    # Estimate price from basis
                    implied_price = avg_spot * (1 + basis / 100 * dte / 365)
                    weighted_price += implied_price * w

            composite_points.append(TermStructurePoint(
                timestamp=timestamp,
                contract=f"COMPOSITE_{dte}D",
                expiry=timestamp + pd.Timedelta(days=dte),
                days_to_expiry=dte,
                futures_price=weighted_price,
                spot_price=avg_spot,
                venue='composite',
                venue_type=VenueType.CEX_FUTURES,
                open_interest=total_oi,
                volume_24h=total_vol,
            ))

        return TermStructureCurve(
            as_of=timestamp,
            spot_price=avg_spot,
            points=composite_points,
            venue='composite',
        )

    def analyze_cross_venue_arbitrage(
        self,
        venue_curves: Dict[str, TermStructureCurve],
        target_dtes: List[int] = None,
        min_spread_bps: float = 10.0
    ) -> List[Dict[str, Any]]:
        """
        Analyze cross-venue arbitrage opportunities.

        Per PDF Section 3.1.4: Cross-venue basis analysis for
        identifying profitable arbitrage opportunities.

        Args:
            venue_curves: Dict mapping venue to curve
            target_dtes: DTEs to analyze (default: [30, 60, 90])
            min_spread_bps: Minimum spread for opportunity

        Returns:
            List of arbitrage opportunity dictionaries
        """
        target_dtes = target_dtes or [30, 60, 90]
        opportunities = []
        venues = list(venue_curves.keys())

        for dte in target_dtes:
            for i, v1 in enumerate(venues):
                for v2 in venues[i + 1:]:
                    curve1 = venue_curves[v1]
                    curve2 = venue_curves[v2]

                    basis1 = curve1.interpolate_basis(dte)
                    basis2 = curve2.interpolate_basis(dte)

                    differential = basis1 - basis2
                    differential_bps = differential * 100

                    # Get costs
                    costs1 = DEFAULT_VENUE_COSTS.get(v1.lower())
                    costs2 = DEFAULT_VENUE_COSTS.get(v2.lower())

                    cost1_bps = costs1.round_trip_taker_bps if costs1 else 20.0
                    cost2_bps = costs2.round_trip_taker_bps if costs2 else 20.0
                    total_cost_bps = cost1_bps + cost2_bps

                    net_spread_bps = abs(differential_bps) - total_cost_bps

                    if net_spread_bps >= min_spread_bps:
                        # Calculate z-score from history
                        pair_key = f"{v1}_{v2}"
                        history = self._basis_history.get(pair_key, [])
                        if len(history) >= 5:
                            z_score = (differential - np.mean(history)) / np.std(history)
                        else:
                            z_score = 0.0
                            self._basis_history[pair_key] = []
                        self._basis_history[pair_key].append(differential)

                        # Direction
                        if differential > 0:
                            direction = f"Short {v1}, Long {v2}"
                            long_venue = v2
                            short_venue = v1
                        else:
                            direction = f"Long {v1}, Short {v2}"
                            long_venue = v1
                            short_venue = v2

                        opportunities.append({
                            'timestamp': curve1.as_of,
                            'dte': dte,
                            'venue_long': long_venue,
                            'venue_short': short_venue,
                            'basis_long_pct': min(basis1, basis2),
                            'basis_short_pct': max(basis1, basis2),
                            'differential_bps': abs(differential_bps),
                            'total_cost_bps': total_cost_bps,
                            'net_spread_bps': net_spread_bps,
                            'z_score': z_score,
                            'direction': direction,
                            'is_significant': abs(z_score) > 2.0,
                            'expected_convergence_days': min(dte // 2, 30),
                            'annualized_return_pct': net_spread_bps / 100 * (365 / max(dte // 2, 7)),
                        })

        # Sort by net spread
        opportunities.sort(key=lambda x: x['net_spread_bps'], reverse=True)
        return opportunities

    def detect_crisis_regime(
        self,
        curves: List[TermStructureCurve],
        lookback: int = 7
    ) -> Dict[str, Any]:
        """
        Detect crisis market conditions based on term structure.

        Per PDF Section 3.3: Crisis event analysis
        (May 2021 crash, Nov 2022 FTX, Luna collapse, etc.)

        Args:
            curves: Recent term structure curves
            lookback: Days to analyze

        Returns:
            Crisis detection analysis
        """
        if len(curves) < lookback:
            return {'is_crisis': False, 'crisis_type': None, 'severity': 0.0}

        recent = curves[-lookback:]

        # Calculate metrics
        basis_values = [c.average_basis_pct for c in recent]
        slopes = [c.curve_slope for c in recent]

        basis_change = basis_values[-1] - basis_values[0]
        basis_volatility = np.std(basis_values)
        avg_basis = np.mean(basis_values)
        slope_change = slopes[-1] - slopes[0]

        # Detect regime transitions
        regimes = [c.regime for c in recent]
        regime_changes = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1])

        # Crisis criteria per PDF crisis events
        crisis_signals = {
            'rapid_basis_collapse': basis_change < -15 and avg_basis > 10,
            'funding_spike': basis_volatility > 10,
            'regime_instability': regime_changes >= 3,
            'curve_inversion': recent[-1].regime.is_backwardation and recent[0].regime.is_contango,
            'extreme_backwardation': avg_basis < -30,
            'extreme_contango': avg_basis > 50,
        }

        is_crisis = any(crisis_signals.values())
        active_signals = [k for k, v in crisis_signals.items() if v]

        # Determine crisis type
        crisis_type = None
        if crisis_signals['rapid_basis_collapse']:
            crisis_type = 'DELEVERAGING'
        elif crisis_signals['curve_inversion']:
            crisis_type = 'MARKET_STRESS'
        elif crisis_signals['extreme_backwardation']:
            crisis_type = 'PANIC_SELLING'
        elif crisis_signals['extreme_contango']:
            crisis_type = 'BUBBLE_FORMATION'
        elif crisis_signals['regime_instability']:
            crisis_type = 'HIGH_VOLATILITY'

        # Calculate severity (0-1)
        severity_scores = [
            min(abs(basis_change) / 20, 1.0),
            min(basis_volatility / 15, 1.0),
            min(regime_changes / 5, 1.0),
        ]
        severity = np.mean(severity_scores) if is_crisis else 0.0

        return {
            'is_crisis': is_crisis,
            'crisis_type': crisis_type,
            'severity': round(severity, 2),
            'active_signals': active_signals,
            'metrics': {
                'basis_change_pct': round(basis_change, 2),
                'basis_volatility_pct': round(basis_volatility, 2),
                'regime_changes': regime_changes,
                'current_regime': recent[-1].regime.value,
            },
            'recommendation': self._get_crisis_recommendation(crisis_type, severity),
        }

    def _get_crisis_recommendation(
        self,
        crisis_type: Optional[str],
        severity: float
    ) -> str:
        """Get trading recommendation during crisis."""
        if crisis_type is None:
            return "Normal market conditions - proceed with standard strategies"

        recommendations = {
            'DELEVERAGING': "Reduce positions 50%, tighten stops, favor short calendar spreads",
            'MARKET_STRESS': "Reduce leverage, close cross-venue positions, wait for stabilization",
            'PANIC_SELLING': "Potential opportunity for long basis trades, scale in carefully",
            'BUBBLE_FORMATION': "Consider short calendar spreads, maintain tight risk limits",
            'HIGH_VOLATILITY': "Reduce position sizes 30%, widen DTE targets, avoid rolls",
        }

        base_rec = recommendations.get(crisis_type, "Exercise caution")

        if severity > 0.7:
            return f"CRITICAL: {base_rec}. Consider full position exit."
        elif severity > 0.4:
            return f"WARNING: {base_rec}"
        return base_rec

    def calculate_term_premium(
        self,
        curve: TermStructureCurve,
        risk_free_rate: float = 0.05
    ) -> Dict[str, float]:
        """
        Calculate term premium components of the curve.

        Decomposes basis into:
        - Expected spot change component
        - Term premium (risk premium for holding futures)
        - Convenience yield

        Args:
            curve: Term structure curve
            risk_free_rate: Annualized risk-free rate

        Returns:
            Term premium decomposition
        """
        if not curve.points:
            return {}

        decomposition = {}

        for point in curve.points:
            if point.days_to_expiry <= 0:
                continue

            dte = point.days_to_expiry
            basis_pct = point.annualized_basis_pct

            # Cost of carry model: F = S * e^((r - y) * T)
            # Where r = risk-free rate, y = convenience yield
            # basis = r - y + term_premium

            # Estimated convenience yield from funding rates
            if point.funding_rate:
                funding_annual = point.funding_rate * point.venue_type.periods_per_year * 100
            else:
                funding_annual = 0.0

            # Term premium = observed basis - (risk-free - funding implied)
            theoretical_basis = (risk_free_rate * 100) - funding_annual
            term_premium = basis_pct - theoretical_basis

            decomposition[f"DTE_{dte}"] = {
                'observed_basis_pct': round(basis_pct, 2),
                'risk_free_component_pct': round(risk_free_rate * 100, 2),
                'funding_implied_pct': round(funding_annual, 2),
                'term_premium_pct': round(term_premium, 2),
                'is_rich': term_premium > 5.0,
                'is_cheap': term_premium < -5.0,
            }

        # Average term premium
        if decomposition:
            avg_premium = np.mean([d['term_premium_pct'] for d in decomposition.values()])
            decomposition['average_term_premium_pct'] = round(avg_premium, 2)

        return decomposition


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'CurveQuality',
    'RegimeTransition',
    'TermStructureCurve',
    'FundingImpliedCurve',
    'RegimeTracker',
    'TermStructureAnalyzer',
]