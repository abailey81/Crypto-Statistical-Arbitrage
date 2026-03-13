"""
Position Sizing Engine - Full Implementation
=========================================================

Comprehensive position sizing system implementing all PDF Section 2.4 requirements
with enhanced quantitative methods for professional-quality trading.

PDF-MANDATED POSITION LIMITS:
- CEX: Up to $100,000 per position
- DEX Liquid: $20,000 - $50,000 per position
- DEX Illiquid: $5,000 - $10,000 per position

EXTENDED SIZING METHODS:
1. Kelly Criterion Suite:
   - Full Kelly, Half Kelly, Quarter Kelly
   - Optimal-f (Ralph Vince method)
   - Secure Kelly (with risk of ruin constraint)
   - Bayesian Kelly (with parameter uncertainty)
   - Correlation-adjusted Kelly

2. Risk-Based Sizing:
   - VaR-constrained sizing
   - CVaR/Expected Shortfall budgeting
   - Tail risk budgeting (Hill estimator)
   - Component VaR allocation
   - Marginal VaR optimization

3. Volatility Models:
   - GARCH(1,1) forecasting
   - EWMA volatility
   - Parkinson/Garman-Klass estimators
   - Regime-conditional volatility

4. Liquidity Analysis:
   - Kyle's Lambda (price impact)
   - Amihud illiquidity ratio
   - Optimal execution sizing
   - Market depth integration

5. Portfolio-Level Methods:
   - Risk Parity allocation
   - Hierarchical Risk Parity (HRP)
   - Maximum Diversification
   - Mean-CVaR optimization

Author: Tamer Atesyakar
Version: 3.0.0 - Complete
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum, auto
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import warnings
from collections import defaultdict
from scipy import stats, optimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class VenueType(Enum):
    """Venue classification for position sizing per PDF."""
    CEX = "cex"
    DEX_LIQUID = "dex_liquid"
    DEX_ILLIQUID = "dex_illiquid"
    HYBRID = "hybrid"


class LiquidityTier(Enum):
    """Liquidity tier classification per PDF Section 2.4."""
    TIER1 = "tier1"  # Top 10 by volume - full position sizing
    TIER2 = "tier2"  # Rank 11-30 - reduced sizing
    TIER3 = "tier3"  # Rank 31+ - minimum sizing (20% limit per PDF)


class KellyVariant(Enum):
    """Kelly Criterion variants for position sizing."""
    FULL = "full"                    # f* = (p*b - q) / b
    HALF = "half"                    # 0.5 * f*
    QUARTER = "quarter"              # 0.25 * f* (conservative default)
    OPTIMAL_F = "optimal_f"          # Ralph Vince optimization
    SECURE = "secure"                # Kelly with risk of ruin constraint
    BAYESIAN = "bayesian"            # Kelly with parameter uncertainty
    CORRELATION_ADJUSTED = "corr_adj"  # Kelly adjusted for correlations


class VolatilityModel(Enum):
    """Volatility estimation models."""
    SIMPLE = "simple"                # Standard deviation
    EWMA = "ewma"                    # Exponentially weighted
    GARCH = "garch"                  # GARCH(1,1)
    PARKINSON = "parkinson"          # High-low range estimator
    GARMAN_KLASS = "garman_klass"    # OHLC estimator
    YANG_ZHANG = "yang_zhang"        # Extended OHLC with overnight gaps
    REGIME_CONDITIONAL = "regime"    # Regime-switching volatility


class RiskMeasure(Enum):
    """Risk measures for position sizing."""
    VOLATILITY = "volatility"
    VAR = "var"                      # Value at Risk
    CVAR = "cvar"                    # Conditional VaR / Expected Shortfall
    MAX_DRAWDOWN = "max_dd"          # Maximum Drawdown
    TAIL_RISK = "tail_risk"          # Hill estimator tail risk


class AllocationMethod(Enum):
    """Portfolio allocation methods."""
    EQUAL_WEIGHT = "equal_weight"
    INVERSE_VOL = "inverse_vol"
    RISK_PARITY = "risk_parity"
    HRP = "hrp"                      # Hierarchical Risk Parity
    MAX_DIVERSIFICATION = "max_div"
    MIN_VARIANCE = "min_var"
    MEAN_CVAR = "mean_cvar"


class SizingRegime(Enum):
    """Market regime for position sizing."""
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PositionSizeConfig:
    """
    PDF-compliant position sizing configuration.

    All position limits from Project Specification.
    """
    # CEX Position Limits (PDF exact values)
    cex_max_position: float = 100_000.0
    cex_min_position: float = 5_000.0
    cex_target_position: float = 50_000.0

    # DEX Liquid Position Limits (PDF exact values)
    dex_liquid_max_position: float = 50_000.0
    dex_liquid_min_position: float = 20_000.0
    dex_liquid_target_position: float = 35_000.0

    # DEX Illiquid Position Limits (PDF exact values)
    dex_illiquid_max_position: float = 10_000.0
    dex_illiquid_min_position: float = 5_000.0
    dex_illiquid_target_position: float = 7_500.0

    # Kelly Criterion parameters
    kelly_variant: KellyVariant = KellyVariant.QUARTER
    kelly_fraction: float = 0.25
    max_kelly_multiplier: float = 2.0
    min_kelly_multiplier: float = 0.1
    kelly_lookback_trades: int = 100
    kelly_min_trades: int = 30

    # Optimal-f parameters
    optimal_f_search_granularity: float = 0.01
    optimal_f_max_value: float = 0.50

    # Secure Kelly parameters
    max_risk_of_ruin: float = 0.01  # 1% max risk of ruin
    ruin_threshold: float = 0.50    # 50% loss = ruin

    # Bayesian Kelly parameters
    kelly_prior_win_rate: float = 0.50
    kelly_prior_strength: int = 20  # Equivalent sample size

    # Volatility targeting
    target_portfolio_vol: float = 0.15  # 15% annual target vol
    vol_lookback_days: int = 60
    vol_model: VolatilityModel = VolatilityModel.EWMA
    ewma_halflife: int = 20
    garch_omega: float = 0.000001
    garch_alpha: float = 0.1
    garch_beta: float = 0.85
    vol_scaling_enabled: bool = True

    # Liquidity-based scaling
    liquidity_scaling_enabled: bool = True
    min_adv_multiple: float = 0.01  # Max 1% of ADV per trade
    max_adv_multiple: float = 0.05  # Absolute max 5% of ADV
    kyle_lambda_estimation: bool = True
    amihud_lookback_days: int = 30

    # Price impact model
    price_impact_model: str = "sqrt"  # sqrt, linear, or almgren_chriss
    price_impact_coefficient: float = 0.1

    # Risk measure constraints
    risk_measure: RiskMeasure = RiskMeasure.CVAR
    var_confidence: float = 0.95
    cvar_confidence: float = 0.95
    max_position_var: float = 0.02  # Max 2% VaR per position
    max_position_cvar: float = 0.03  # Max 3% CVaR per position
    tail_risk_threshold: float = 0.05  # 5% tail probability

    # Drawdown-based adjustments
    drawdown_scaling_enabled: bool = True
    max_acceptable_drawdown: float = 0.20  # 20% max drawdown
    drawdown_reduction_rate: float = 0.9   # 90% reduction at max DD
    min_drawdown_scalar: float = 0.1       # Minimum 10% position

    # Regime-based adjustments
    crisis_position_reduction: float = 0.50   # 50% reduction during crisis
    high_vol_position_reduction: float = 0.30 # 30% reduction in high vol
    low_vol_position_increase: float = 1.20   # 20% increase in low vol
    recovery_position_scalar: float = 0.70    # 70% during recovery

    # Concentration limits (PDF Section 2.4)
    tier1_allocation_max: float = 1.0   # No limit on Tier 1
    tier2_allocation_max: float = 0.60  # 60% max in Tier 2
    tier3_allocation_max: float = 0.20  # 20% max in Tier 3 (PDF REQUIRED)
    max_sector_concentration: float = 0.40  # 40% max per sector
    max_cex_concentration: float = 0.60     # 60% max CEX-only
    max_single_position: float = 0.10       # 10% max single position

    # Correlation adjustments
    correlation_scaling_enabled: bool = True
    high_correlation_threshold: float = 0.70
    correlation_reduction_factor: float = 0.50

    # Monte Carlo validation
    monte_carlo_enabled: bool = True
    monte_carlo_simulations: int = 10000
    monte_carlo_horizon_days: int = 252
    monte_carlo_block_size: int = 20  # For block bootstrap

    # Portfolio allocation method
    allocation_method: AllocationMethod = AllocationMethod.RISK_PARITY


@dataclass
class KellyResult:
    """Result of Kelly Criterion calculation."""
    variant: KellyVariant
    full_kelly: float
    fractional_kelly: float
    optimal_f: Optional[float]
    win_rate: float
    win_loss_ratio: float
    expected_growth: float
    risk_of_ruin: float
    confidence_interval: Tuple[float, float]
    n_trades: int
    is_valid: bool
    warnings: List[str] = field(default_factory=list)


@dataclass
class VolatilityEstimate:
    """Result of volatility estimation."""
    model: VolatilityModel
    current_vol: float
    forecast_vol: float
    vol_of_vol: float
    regime: SizingRegime
    percentile: float  # Where current vol sits historically
    half_life: Optional[float]  # Mean reversion half-life
    is_stationary: bool
    confidence_interval: Tuple[float, float]


@dataclass
class LiquidityAnalysis:
    """Comprehensive liquidity analysis for position sizing."""
    adv_usd: float                    # Average daily volume
    adv_percentile: float             # ADV percentile vs universe
    kyle_lambda: float                # Price impact coefficient
    amihud_ratio: float               # Amihud illiquidity ratio
    bid_ask_spread_bps: float         # Estimated spread
    market_depth_usd: float           # Available depth
    optimal_trade_size: float         # Size minimizing impact
    expected_slippage_bps: float      # Expected execution cost
    max_position_adv_pct: float       # Max position as % ADV
    liquidity_score: float            # 0-100 score
    warnings: List[str] = field(default_factory=list)


@dataclass
class RiskBudget:
    """Risk budget allocation for a position."""
    var_budget: float                 # VaR allocation
    cvar_budget: float                # CVaR allocation
    vol_budget: float                 # Volatility budget
    tail_risk_budget: float           # Tail risk allocation
    marginal_var: float               # Marginal VaR
    component_var: float              # Component VaR
    risk_contribution_pct: float      # % of total portfolio risk


@dataclass
class PositionSizeResult:
    """Comprehensive result of position size calculation."""
    pair_id: str
    venue_type: VenueType
    liquidity_tier: LiquidityTier

    # Base sizing
    base_position_usd: float
    target_position_usd: float
    max_position_usd: float
    min_position_usd: float

    # Kelly analysis
    kelly_result: Optional[KellyResult]
    kelly_multiplier: float

    # Volatility analysis
    volatility_estimate: Optional[VolatilityEstimate]
    volatility_scalar: float

    # Liquidity analysis
    liquidity_analysis: Optional[LiquidityAnalysis]
    liquidity_scalar: float

    # Risk budget
    risk_budget: Optional[RiskBudget]
    risk_scalar: float

    # Regime adjustment
    regime: SizingRegime
    regime_scalar: float

    # Correlation adjustment
    correlation_to_portfolio: float
    correlation_scalar: float

    # Drawdown adjustment
    current_drawdown: float
    drawdown_scalar: float

    # Final position
    final_position_usd: float
    final_position_pct_portfolio: float

    # Constraints status
    adv_pct: float
    slippage_estimate_bps: float
    var_utilization: float
    cvar_utilization: float

    # Monte Carlo validation
    monte_carlo_valid: bool
    monte_carlo_risk_of_ruin: float
    monte_carlo_expected_return: float

    # Metadata
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    warnings: List[str] = field(default_factory=list)
    adjustment_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioSizeResult:
    """Result of portfolio-level position sizing."""
    total_capital: float
    allocated_capital: float
    unallocated_capital: float

    # By venue type
    cex_allocated: float
    dex_liquid_allocated: float
    dex_illiquid_allocated: float
    hybrid_allocated: float

    # By tier
    tier1_allocated: float
    tier2_allocated: float
    tier3_allocated: float

    # Position counts
    total_positions: int
    cex_positions: int
    dex_positions: int

    # Individual positions
    position_sizes: Dict[str, PositionSizeResult] = field(default_factory=dict)

    # Risk metrics
    portfolio_var: float = 0.0
    portfolio_cvar: float = 0.0
    portfolio_volatility: float = 0.0
    expected_return: float = 0.0
    sharpe_ratio: float = 0.0

    # Diversification metrics
    effective_n: float = 0.0  # Effective number of positions
    herfindahl_index: float = 0.0
    diversification_ratio: float = 0.0

    # Constraints status
    tier3_limit_utilized: float = 0.0
    cex_concentration: float = 0.0
    max_sector_concentration: float = 0.0
    max_single_position_pct: float = 0.0

    # Allocation method used
    allocation_method: AllocationMethod = AllocationMethod.RISK_PARITY

    # Optimization stats
    optimization_converged: bool = True
    optimization_iterations: int = 0

    # Timestamp
    calculation_timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# KELLY CRITERION ENGINE
# =============================================================================

class KellyCriterionEngine:
    """
    Comprehensive Kelly Criterion implementation with multiple variants.

    Implements:
    - Full Kelly: Optimal growth rate
    - Half/Quarter Kelly: Conservative fractions
    - Optimal-f: Ralph Vince's geometric mean optimization
    - Secure Kelly: With risk of ruin constraint
    - Bayesian Kelly: With parameter uncertainty
    - Correlation-adjusted Kelly: For correlated positions
    """

    def __init__(self, config: PositionSizeConfig):
        self.config = config

    def calculate_full_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate full Kelly fraction.

        Kelly formula: f* = (p * b - q) / b
        where:
            p = win probability
            b = win/loss ratio (odds)
            q = 1 - p (loss probability)
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0

        p = win_rate
        q = 1 - p
        b = abs(avg_win / avg_loss)

        kelly = (p * b - q) / b

        # Kelly should be positive for profitable systems
        return max(0.0, kelly)

    def calculate_fractional_kelly(
        self,
        full_kelly: float,
        fraction: float = 0.25
    ) -> float:
        """Apply fractional Kelly for conservative sizing."""
        return full_kelly * fraction

    def calculate_optimal_f(
        self,
        returns: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate Optimal-f using Ralph Vince's method.

        Optimal-f maximizes the geometric mean (Terminal Wealth Relative).

        Returns:
            (optimal_f, expected_growth_rate, twr)
        """
        if len(returns) < self.config.kelly_min_trades:
            return 0.25, 0.0, 1.0

        # Find the largest loss
        largest_loss = abs(np.min(returns))
        if largest_loss == 0:
            return 0.25, 0.0, 1.0

        best_f = 0.0
        best_twr = 0.0

        # Search for optimal f
        for f in np.arange(
            self.config.optimal_f_search_granularity,
            self.config.optimal_f_max_value + self.config.optimal_f_search_granularity,
            self.config.optimal_f_search_granularity
        ):
            # Calculate Holding Period Returns
            hpr = 1 + f * (returns / largest_loss)

            # Skip if any HPR <= 0 (ruin)
            if np.any(hpr <= 0):
                continue

            # Terminal Wealth Relative (geometric mean)
            twr = np.prod(hpr) ** (1 / len(returns))

            if twr > best_twr:
                best_twr = twr
                best_f = f

        # Convert f to fraction of capital
        optimal_f_capital = best_f * largest_loss
        expected_growth = best_twr - 1

        return optimal_f_capital, expected_growth, best_twr

    def calculate_secure_kelly(
        self,
        returns: np.ndarray,
        max_risk_of_ruin: float = 0.01
    ) -> Tuple[float, float]:
        """
        Calculate Secure Kelly with risk of ruin constraint.

        Finds the Kelly fraction that keeps risk of ruin below threshold.

        Returns:
            (secure_kelly_fraction, actual_risk_of_ruin)
        """
        if len(returns) < self.config.kelly_min_trades:
            return 0.25, 0.0

        win_rate = np.mean(returns > 0)
        avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
        avg_loss = abs(np.mean(returns[returns < 0])) if np.any(returns < 0) else 0

        full_kelly = self.calculate_full_kelly(win_rate, avg_win, avg_loss)

        # Binary search for secure Kelly
        low, high = 0.0, full_kelly
        secure_f = 0.0

        for _ in range(50):  # Max iterations
            mid = (low + high) / 2
            ror = self._estimate_risk_of_ruin(returns, mid)

            if ror <= max_risk_of_ruin:
                secure_f = mid
                low = mid
            else:
                high = mid

            if high - low < 0.001:
                break

        actual_ror = self._estimate_risk_of_ruin(returns, secure_f)

        return secure_f, actual_ror

    def _estimate_risk_of_ruin(
        self,
        returns: np.ndarray,
        fraction: float,
        n_simulations: int = 1000,
        horizon: int = 252
    ) -> float:
        """Estimate risk of ruin via Monte Carlo."""
        if fraction <= 0:
            return 0.0

        ruin_count = 0
        ruin_threshold = self.config.ruin_threshold

        for _ in range(n_simulations):
            equity = 1.0
            sampled = np.random.choice(returns, size=horizon, replace=True)

            for ret in sampled:
                equity *= (1 + fraction * ret)
                if equity < (1 - ruin_threshold):
                    ruin_count += 1
                    break

        return ruin_count / n_simulations

    def calculate_bayesian_kelly(
        self,
        returns: np.ndarray
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate Bayesian Kelly with parameter uncertainty.

        Uses Beta-Binomial conjugate prior for win rate.
        Returns Kelly fraction with credible interval.
        """
        if len(returns) < self.config.kelly_min_trades:
            return 0.25, (0.1, 0.4)

        wins = np.sum(returns > 0)
        losses = np.sum(returns < 0)
        n = wins + losses

        # Beta posterior parameters (with prior)
        prior_alpha = self.config.kelly_prior_win_rate * self.config.kelly_prior_strength
        prior_beta = (1 - self.config.kelly_prior_win_rate) * self.config.kelly_prior_strength

        posterior_alpha = prior_alpha + wins
        posterior_beta = prior_beta + losses

        # Posterior mean win rate
        posterior_win_rate = posterior_alpha / (posterior_alpha + posterior_beta)

        # Calculate Kelly at posterior mean
        avg_win = np.mean(returns[returns > 0]) if wins > 0 else 0
        avg_loss = abs(np.mean(returns[returns < 0])) if losses > 0 else 0

        kelly_mean = self.calculate_full_kelly(posterior_win_rate, avg_win, avg_loss)

        # Credible interval via sampling
        kelly_samples = []
        for _ in range(1000):
            sampled_win_rate = np.random.beta(posterior_alpha, posterior_beta)
            sampled_kelly = self.calculate_full_kelly(sampled_win_rate, avg_win, avg_loss)
            kelly_samples.append(sampled_kelly)

        ci_low = np.percentile(kelly_samples, 2.5)
        ci_high = np.percentile(kelly_samples, 97.5)

        # Conservative: use lower bound of CI
        conservative_kelly = ci_low * self.config.kelly_fraction

        return conservative_kelly, (ci_low, ci_high)

    def calculate_correlation_adjusted_kelly(
        self,
        returns: np.ndarray,
        portfolio_correlation: float,
        n_positions: int
    ) -> float:
        """
        Calculate correlation-adjusted Kelly for multiple positions.

        Adjusts Kelly for portfolio effect of correlated positions.
        """
        if len(returns) < self.config.kelly_min_trades:
            return 0.25

        win_rate = np.mean(returns > 0)
        avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
        avg_loss = abs(np.mean(returns[returns < 0])) if np.any(returns < 0) else 0

        full_kelly = self.calculate_full_kelly(win_rate, avg_win, avg_loss)

        # Correlation adjustment factor
        # Higher correlation = more conservative sizing
        # Formula: adjusted_kelly = kelly / (1 + (n-1) * rho * kelly)
        if n_positions > 1 and portfolio_correlation > 0:
            adjustment = 1 + (n_positions - 1) * portfolio_correlation * full_kelly
            adjusted_kelly = full_kelly / adjustment
        else:
            adjusted_kelly = full_kelly

        return adjusted_kelly * self.config.kelly_fraction

    def calculate_kelly(
        self,
        returns: np.ndarray,
        variant: Optional[KellyVariant] = None,
        portfolio_correlation: float = 0.0,
        n_positions: int = 1
    ) -> KellyResult:
        """
        Calculate Kelly Criterion using specified variant.

        Args:
            returns: Array of trade returns
            variant: Kelly variant to use (defaults to config)
            portfolio_correlation: Average correlation to portfolio
            n_positions: Number of positions in portfolio

        Returns:
            Comprehensive KellyResult
        """
        variant = variant or self.config.kelly_variant
        warnings = []

        n_trades = len(returns)
        if n_trades < self.config.kelly_min_trades:
            warnings.append(f"Insufficient trades ({n_trades}), using default")
            return KellyResult(
                variant=variant,
                full_kelly=0.25,
                fractional_kelly=0.25 * self.config.kelly_fraction,
                optimal_f=None,
                win_rate=0.5,
                win_loss_ratio=1.0,
                expected_growth=0.0,
                risk_of_ruin=0.0,
                confidence_interval=(0.1, 0.4),
                n_trades=n_trades,
                is_valid=False,
                warnings=warnings
            )

        # Calculate base statistics
        win_rate = np.mean(returns > 0)
        avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
        avg_loss = abs(np.mean(returns[returns < 0])) if np.any(returns < 0) else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        # Full Kelly
        full_kelly = self.calculate_full_kelly(win_rate, avg_win, avg_loss)

        # Calculate based on variant
        optimal_f = None
        expected_growth = 0.0
        risk_of_ruin = 0.0
        confidence_interval = (0.0, 0.0)

        if variant == KellyVariant.FULL:
            fractional = full_kelly

        elif variant == KellyVariant.HALF:
            fractional = full_kelly * 0.5

        elif variant == KellyVariant.QUARTER:
            fractional = full_kelly * 0.25

        elif variant == KellyVariant.OPTIMAL_F:
            optimal_f, expected_growth, _ = self.calculate_optimal_f(returns)
            fractional = optimal_f * self.config.kelly_fraction

        elif variant == KellyVariant.SECURE:
            fractional, risk_of_ruin = self.calculate_secure_kelly(
                returns, self.config.max_risk_of_ruin
            )

        elif variant == KellyVariant.BAYESIAN:
            fractional, confidence_interval = self.calculate_bayesian_kelly(returns)

        elif variant == KellyVariant.CORRELATION_ADJUSTED:
            fractional = self.calculate_correlation_adjusted_kelly(
                returns, portfolio_correlation, n_positions
            )

        else:
            fractional = full_kelly * self.config.kelly_fraction

        # Bound Kelly multiplier
        fractional = np.clip(
            fractional,
            self.config.min_kelly_multiplier,
            self.config.max_kelly_multiplier
        )

        # Calculate expected growth if not already done
        if expected_growth == 0 and fractional > 0:
            expected_growth = win_rate * np.log(1 + fractional * avg_win) + \
                            (1 - win_rate) * np.log(1 - fractional * avg_loss)

        # Estimate risk of ruin if not already done
        if risk_of_ruin == 0:
            risk_of_ruin = self._estimate_risk_of_ruin(returns, fractional)

        # Validation warnings
        if full_kelly < 0:
            warnings.append("Negative Kelly - system is unprofitable")
        if full_kelly > 1:
            warnings.append("Kelly > 100% - high variance system")
        if risk_of_ruin > 0.05:
            warnings.append(f"High risk of ruin: {risk_of_ruin:.1%}")

        return KellyResult(
            variant=variant,
            full_kelly=full_kelly,
            fractional_kelly=fractional,
            optimal_f=optimal_f,
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
            expected_growth=expected_growth,
            risk_of_ruin=risk_of_ruin,
            confidence_interval=confidence_interval,
            n_trades=n_trades,
            is_valid=full_kelly > 0,
            warnings=warnings
        )


# =============================================================================
# VOLATILITY ENGINE
# =============================================================================

class VolatilityEngine:
    """
    Comprehensive volatility estimation with multiple models.

    Implements:
    - Simple historical volatility
    - EWMA (Exponentially Weighted Moving Average)
    - GARCH(1,1) forecasting
    - Parkinson (high-low range) estimator
    - Garman-Klass (OHLC) estimator
    - Yang-Zhang (extended OHLC with overnight gaps)
    - Regime-conditional volatility
    """

    def __init__(self, config: PositionSizeConfig):
        self.config = config

    def calculate_simple_vol(
        self,
        returns: np.ndarray,
        annualize: bool = True
    ) -> float:
        """Calculate simple historical volatility."""
        if len(returns) < 2:
            return 0.30  # Default 30% vol

        vol = np.std(returns, ddof=1)

        if annualize:
            vol *= np.sqrt(252)

        return vol

    def calculate_ewma_vol(
        self,
        returns: np.ndarray,
        halflife: Optional[int] = None,
        annualize: bool = True
    ) -> float:
        """
        Calculate EWMA volatility.

        EWMA: sigma^2_t = lambda * sigma^2_{t-1} + (1-lambda) * r^2_{t-1}
        """
        if len(returns) < 2:
            return 0.30

        halflife = halflife or self.config.ewma_halflife
        decay = np.log(2) / halflife

        # Calculate weights
        n = len(returns)
        weights = np.exp(-decay * np.arange(n)[::-1])
        weights /= weights.sum()

        # Weighted variance
        mean_return = np.average(returns, weights=weights)
        variance = np.average((returns - mean_return) ** 2, weights=weights)

        vol = np.sqrt(variance)

        if annualize:
            vol *= np.sqrt(252)

        return vol

    def calculate_garch_vol(
        self,
        returns: np.ndarray,
        omega: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        annualize: bool = True
    ) -> Tuple[float, float]:
        """
        Calculate GARCH(1,1) volatility and forecast.

        GARCH(1,1): sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}

        Returns:
            (current_vol, forecast_vol)
        """
        if len(returns) < 10:
            return 0.30, 0.30

        omega = omega or self.config.garch_omega
        alpha = alpha or self.config.garch_alpha
        beta = beta or self.config.garch_beta

        # Initialize with sample variance
        variance = np.var(returns)
        variances = [variance]

        # Iterate through returns
        for r in returns:
            variance = omega + alpha * (r ** 2) + beta * variance
            variances.append(variance)

        current_vol = np.sqrt(variances[-1])

        # Long-run variance forecast
        long_run_var = omega / (1 - alpha - beta) if (alpha + beta) < 1 else variance

        # One-step ahead forecast
        forecast_var = omega + alpha * (returns[-1] ** 2) + beta * variances[-1]
        forecast_vol = np.sqrt(forecast_var)

        if annualize:
            current_vol *= np.sqrt(252)
            forecast_vol *= np.sqrt(252)

        return current_vol, forecast_vol

    def calculate_parkinson_vol(
        self,
        high: np.ndarray,
        low: np.ndarray,
        annualize: bool = True
    ) -> float:
        """
        Calculate Parkinson volatility estimator (high-low range).

        More efficient than close-to-close when intraday data available.
        """
        if len(high) < 2:
            return 0.30

        log_hl = np.log(high / low)
        variance = (1 / (4 * np.log(2))) * np.mean(log_hl ** 2)

        vol = np.sqrt(variance)

        if annualize:
            vol *= np.sqrt(252)

        return vol

    def calculate_garman_klass_vol(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        annualize: bool = True
    ) -> float:
        """
        Calculate Garman-Klass volatility estimator (OHLC).

        Most efficient estimator using all OHLC data.
        """
        if len(close) < 2:
            return 0.30

        log_hl = np.log(high / low)
        log_co = np.log(close / open_)

        variance = 0.5 * np.mean(log_hl ** 2) - \
                   (2 * np.log(2) - 1) * np.mean(log_co ** 2)

        # Ensure non-negative
        variance = max(variance, 0)
        vol = np.sqrt(variance)

        if annualize:
            vol *= np.sqrt(252)

        return vol

    def calculate_yang_zhang_vol(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        annualize: bool = True
    ) -> float:
        """
        Calculate Yang-Zhang volatility estimator.

        Handles overnight gaps and is drift-independent.
        """
        if len(close) < 3:
            return 0.30

        n = len(close)

        # Overnight volatility
        log_oc = np.log(open_[1:] / close[:-1])
        overnight_var = np.var(log_oc, ddof=1)

        # Open-to-close volatility
        log_co = np.log(close / open_)
        open_close_var = np.var(log_co, ddof=1)

        # Rogers-Satchell volatility
        log_ho = np.log(high / open_)
        log_lo = np.log(low / open_)
        log_hc = np.log(high / close)
        log_lc = np.log(low / close)

        rs_var = np.mean(log_ho * log_hc + log_lo * log_lc)

        # Yang-Zhang combination
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        variance = overnight_var + k * open_close_var + (1 - k) * rs_var

        # Ensure non-negative
        variance = max(variance, 0)
        vol = np.sqrt(variance)

        if annualize:
            vol *= np.sqrt(252)

        return vol

    def detect_volatility_regime(
        self,
        current_vol: float,
        historical_vols: np.ndarray
    ) -> SizingRegime:
        """Detect volatility regime based on historical distribution."""
        if len(historical_vols) < 20:
            return SizingRegime.NORMAL

        percentile = stats.percentileofscore(historical_vols, current_vol)

        if percentile >= 90:
            return SizingRegime.HIGH_VOLATILITY
        elif percentile >= 75:
            return SizingRegime.CRISIS if current_vol > np.mean(historical_vols) * 2 else SizingRegime.NORMAL
        elif percentile <= 10:
            return SizingRegime.LOW_VOLATILITY
        else:
            return SizingRegime.NORMAL

    def calculate_vol_of_vol(
        self,
        volatilities: np.ndarray
    ) -> float:
        """Calculate volatility of volatility (vol clustering measure)."""
        if len(volatilities) < 10:
            return 0.0

        vol_returns = np.diff(np.log(volatilities + 1e-10))
        return np.std(vol_returns) * np.sqrt(252)

    def estimate_volatility(
        self,
        returns: np.ndarray,
        ohlc_data: Optional[Dict[str, np.ndarray]] = None,
        model: Optional[VolatilityModel] = None
    ) -> VolatilityEstimate:
        """
        Comprehensive volatility estimation.

        Args:
            returns: Array of returns
            ohlc_data: Optional dict with 'open', 'high', 'low', 'close' arrays
            model: Volatility model to use (defaults to config)

        Returns:
            Complete VolatilityEstimate
        """
        model = model or self.config.vol_model

        if len(returns) < 5:
            return VolatilityEstimate(
                model=model,
                current_vol=0.30,
                forecast_vol=0.30,
                vol_of_vol=0.0,
                regime=SizingRegime.NORMAL,
                percentile=50.0,
                half_life=None,
                is_stationary=True,
                confidence_interval=(0.20, 0.40)
            )

        # Calculate based on model
        if model == VolatilityModel.SIMPLE:
            current_vol = self.calculate_simple_vol(returns)
            forecast_vol = current_vol

        elif model == VolatilityModel.EWMA:
            current_vol = self.calculate_ewma_vol(returns)
            forecast_vol = current_vol

        elif model == VolatilityModel.GARCH:
            current_vol, forecast_vol = self.calculate_garch_vol(returns)

        elif model == VolatilityModel.PARKINSON and ohlc_data:
            current_vol = self.calculate_parkinson_vol(
                ohlc_data['high'], ohlc_data['low']
            )
            forecast_vol = current_vol

        elif model == VolatilityModel.GARMAN_KLASS and ohlc_data:
            current_vol = self.calculate_garman_klass_vol(
                ohlc_data['open'], ohlc_data['high'],
                ohlc_data['low'], ohlc_data['close']
            )
            forecast_vol = current_vol

        elif model == VolatilityModel.YANG_ZHANG and ohlc_data:
            current_vol = self.calculate_yang_zhang_vol(
                ohlc_data['open'], ohlc_data['high'],
                ohlc_data['low'], ohlc_data['close']
            )
            forecast_vol = current_vol

        else:
            # Default to EWMA
            current_vol = self.calculate_ewma_vol(returns)
            forecast_vol = current_vol

        # Calculate rolling volatilities for regime detection
        window = min(60, len(returns))
        rolling_vols = []
        for i in range(window, len(returns)):
            rolling_vols.append(self.calculate_simple_vol(returns[i-window:i]))

        rolling_vols = np.array(rolling_vols) if rolling_vols else np.array([current_vol])

        # Detect regime
        regime = self.detect_volatility_regime(current_vol, rolling_vols)

        # Vol of vol
        vol_of_vol = self.calculate_vol_of_vol(rolling_vols)

        # Percentile
        percentile = stats.percentileofscore(rolling_vols, current_vol) if len(rolling_vols) > 0 else 50.0

        # Confidence interval (bootstrap)
        ci_low, ci_high = self._bootstrap_vol_ci(returns)

        # Check stationarity (simplified ADF test)
        is_stationary = self._check_stationarity(returns)

        # Estimate half-life of vol mean reversion
        half_life = self._estimate_vol_halflife(rolling_vols)

        return VolatilityEstimate(
            model=model,
            current_vol=current_vol,
            forecast_vol=forecast_vol,
            vol_of_vol=vol_of_vol,
            regime=regime,
            percentile=percentile,
            half_life=half_life,
            is_stationary=is_stationary,
            confidence_interval=(ci_low, ci_high)
        )

    def _bootstrap_vol_ci(
        self,
        returns: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for volatility."""
        if len(returns) < 20:
            vol = self.calculate_simple_vol(returns)
            return vol * 0.8, vol * 1.2

        bootstrap_vols = []
        n = len(returns)

        for _ in range(n_bootstrap):
            sample = np.random.choice(returns, size=n, replace=True)
            bootstrap_vols.append(self.calculate_simple_vol(sample))

        alpha = (1 - confidence) / 2
        ci_low = np.percentile(bootstrap_vols, alpha * 100)
        ci_high = np.percentile(bootstrap_vols, (1 - alpha) * 100)

        return ci_low, ci_high

    def _check_stationarity(
        self,
        returns: np.ndarray
    ) -> bool:
        """Simplified stationarity check."""
        if len(returns) < 50:
            return True

        # Simple variance ratio test
        half = len(returns) // 2
        var1 = np.var(returns[:half])
        var2 = np.var(returns[half:])

        # If variances are similar, likely stationary
        ratio = max(var1, var2) / (min(var1, var2) + 1e-10)
        return ratio < 2.0

    def _estimate_vol_halflife(
        self,
        volatilities: np.ndarray
    ) -> Optional[float]:
        """Estimate half-life of volatility mean reversion."""
        if len(volatilities) < 30:
            return None

        try:
            # AR(1) regression on log-vol
            log_vols = np.log(volatilities + 1e-10)
            y = log_vols[1:]
            x = log_vols[:-1]

            # Simple regression
            beta = np.cov(x, y)[0, 1] / np.var(x)

            if 0 < beta < 1:
                half_life = -np.log(2) / np.log(beta)
                return half_life
        except Exception:
            pass

        return None


# =============================================================================
# LIQUIDITY ANALYSIS ENGINE
# =============================================================================

class LiquidityAnalysisEngine:
    """
    Comprehensive liquidity analysis for position sizing.

    Implements:
    - Kyle's Lambda (price impact)
    - Amihud illiquidity ratio
    - Bid-ask spread estimation (Roll estimator)
    - Market depth analysis
    - Optimal execution sizing
    """

    def __init__(self, config: PositionSizeConfig):
        self.config = config

    def calculate_kyles_lambda(
        self,
        returns: np.ndarray,
        volumes: np.ndarray
    ) -> float:
        """
        Calculate Kyle's Lambda (price impact coefficient).

        Lambda = Cov(r, sign(V)) / Var(sign(V))
        Higher lambda = higher price impact = lower liquidity
        """
        if len(returns) < 20 or len(volumes) < 20:
            return 0.001  # Default low impact

        # Normalize volumes
        signed_volume = np.sign(returns) * volumes

        # Regression: return = lambda * signed_volume + error
        try:
            covariance = np.cov(returns, signed_volume)[0, 1]
            variance = np.var(signed_volume)

            if variance > 0:
                kyle_lambda = covariance / variance
                return abs(kyle_lambda)
        except Exception:
            pass

        return 0.001

    def calculate_amihud_ratio(
        self,
        returns: np.ndarray,
        volumes_usd: np.ndarray
    ) -> float:
        """
        Calculate Amihud illiquidity ratio.

        ILLIQ = (1/N) * sum(|r_t| / V_t)
        Higher ratio = less liquid
        """
        if len(returns) < 10 or len(volumes_usd) < 10:
            return 0.0

        # Avoid division by zero
        valid_mask = volumes_usd > 0
        if not np.any(valid_mask):
            return 0.0

        illiq = np.mean(np.abs(returns[valid_mask]) / volumes_usd[valid_mask])

        return illiq * 1e6  # Scale for readability

    def estimate_bid_ask_spread(
        self,
        returns: np.ndarray
    ) -> float:
        """
        Estimate bid-ask spread using Roll estimator.

        Spread = 2 * sqrt(-Cov(r_t, r_{t-1})) if negative
        """
        if len(returns) < 10:
            return 10.0  # Default 10 bps

        # Autocovariance at lag 1
        autocov = np.cov(returns[:-1], returns[1:])[0, 1]

        if autocov < 0:
            spread = 2 * np.sqrt(-autocov)
            return spread * 10000  # Convert to bps

        return 10.0  # Default if autocov not negative

    def calculate_market_depth(
        self,
        volumes_usd: np.ndarray,
        percentile: float = 25
    ) -> float:
        """
        Estimate market depth at given percentile.

        Uses lower percentile of daily volumes as proxy for available depth.
        """
        if len(volumes_usd) < 5:
            return 100_000  # Default $100k

        return np.percentile(volumes_usd, percentile)

    def calculate_optimal_trade_size(
        self,
        adv: float,
        kyle_lambda: float,
        max_impact_bps: float = 10
    ) -> float:
        """
        Calculate optimal trade size to limit price impact.

        Uses square-root market impact model.
        """
        if kyle_lambda <= 0:
            return adv * self.config.max_adv_multiple

        # Impact = kyle_lambda * sqrt(size / ADV)
        # Solve for size: size = ADV * (max_impact / kyle_lambda)^2
        max_impact = max_impact_bps / 10000

        optimal_size = adv * (max_impact / kyle_lambda) ** 2

        # Cap at max ADV multiple
        return min(optimal_size, adv * self.config.max_adv_multiple)

    def estimate_slippage(
        self,
        trade_size: float,
        adv: float,
        spread_bps: float,
        kyle_lambda: float
    ) -> float:
        """
        Estimate expected slippage in basis points.

        Total cost = spread / 2 + market impact
        """
        if adv <= 0:
            return 100.0  # High slippage for illiquid

        # Half-spread cost
        half_spread = spread_bps / 2

        # Market impact (square-root model)
        adv_pct = trade_size / adv
        impact = kyle_lambda * np.sqrt(adv_pct) * 10000

        return half_spread + impact

    def calculate_liquidity_score(
        self,
        adv: float,
        amihud: float,
        spread_bps: float
    ) -> float:
        """
        Calculate composite liquidity score (0-100).

        Higher score = more liquid
        """
        # ADV component (0-40 points)
        if adv >= 100_000_000:
            adv_score = 40
        elif adv >= 10_000_000:
            adv_score = 30
        elif adv >= 1_000_000:
            adv_score = 20
        elif adv >= 100_000:
            adv_score = 10
        else:
            adv_score = 5

        # Amihud component (0-30 points)
        if amihud <= 0.01:
            amihud_score = 30
        elif amihud <= 0.1:
            amihud_score = 20
        elif amihud <= 1.0:
            amihud_score = 10
        else:
            amihud_score = 5

        # Spread component (0-30 points)
        if spread_bps <= 5:
            spread_score = 30
        elif spread_bps <= 10:
            spread_score = 25
        elif spread_bps <= 25:
            spread_score = 15
        elif spread_bps <= 50:
            spread_score = 10
        else:
            spread_score = 5

        return adv_score + amihud_score + spread_score

    def analyze_liquidity(
        self,
        returns: np.ndarray,
        volumes_usd: np.ndarray,
        trade_size: float
    ) -> LiquidityAnalysis:
        """
        Comprehensive liquidity analysis.

        Args:
            returns: Array of returns
            volumes_usd: Array of daily volumes in USD
            trade_size: Proposed trade size in USD

        Returns:
            Complete LiquidityAnalysis
        """
        warnings = []

        if len(volumes_usd) < 5:
            return LiquidityAnalysis(
                adv_usd=1_000_000,
                adv_percentile=50.0,
                kyle_lambda=0.001,
                amihud_ratio=0.0,
                bid_ask_spread_bps=10.0,
                market_depth_usd=100_000,
                optimal_trade_size=50_000,
                expected_slippage_bps=15.0,
                max_position_adv_pct=5.0,
                liquidity_score=50.0,
                warnings=["Insufficient data for liquidity analysis"]
            )

        # Calculate metrics
        adv = np.mean(volumes_usd)
        kyle_lambda = self.calculate_kyles_lambda(returns, volumes_usd)
        amihud = self.calculate_amihud_ratio(returns, volumes_usd)
        spread_bps = self.estimate_bid_ask_spread(returns)
        depth = self.calculate_market_depth(volumes_usd)
        optimal_size = self.calculate_optimal_trade_size(adv, kyle_lambda)
        slippage = self.estimate_slippage(trade_size, adv, spread_bps, kyle_lambda)
        score = self.calculate_liquidity_score(adv, amihud, spread_bps)

        # ADV percentage
        adv_pct = (trade_size / adv * 100) if adv > 0 else 100
        max_adv_pct = self.config.max_adv_multiple * 100

        # ADV percentile (where this asset sits vs typical)
        adv_percentile = min(100, adv / 1_000_000 * 10)  # Simplified

        # Warnings
        if adv_pct > max_adv_pct:
            warnings.append(f"Trade size exceeds {max_adv_pct}% of ADV")
        if slippage > 50:
            warnings.append(f"High expected slippage: {slippage:.0f} bps")
        if score < 30:
            warnings.append("Low liquidity score - consider reducing size")
        if trade_size > depth:
            warnings.append("Trade size exceeds estimated market depth")

        return LiquidityAnalysis(
            adv_usd=adv,
            adv_percentile=adv_percentile,
            kyle_lambda=kyle_lambda,
            amihud_ratio=amihud,
            bid_ask_spread_bps=spread_bps,
            market_depth_usd=depth,
            optimal_trade_size=optimal_size,
            expected_slippage_bps=slippage,
            max_position_adv_pct=max_adv_pct,
            liquidity_score=score,
            warnings=warnings
        )


# =============================================================================
# RISK BUDGET ENGINE
# =============================================================================

class RiskBudgetEngine:
    """
    Risk budget allocation for position sizing.

    Implements:
    - VaR-based sizing
    - CVaR/Expected Shortfall constraints
    - Component VaR allocation
    - Marginal VaR optimization
    - Tail risk budgeting
    """

    def __init__(self, config: PositionSizeConfig):
        self.config = config

    def calculate_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
        method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk.

        Methods:
        - historical: Percentile of historical returns
        - parametric: Normal distribution assumption
        - cornish_fisher: Adjusted for skew/kurtosis
        """
        if len(returns) < 10:
            return 0.02  # Default 2%

        if method == "historical":
            var = -np.percentile(returns, (1 - confidence) * 100)

        elif method == "parametric":
            mu = np.mean(returns)
            sigma = np.std(returns)
            z = stats.norm.ppf(1 - confidence)
            var = -(mu + z * sigma)

        elif method == "cornish_fisher":
            mu = np.mean(returns)
            sigma = np.std(returns)
            skew = stats.skew(returns)
            kurt = stats.kurtosis(returns)

            z = stats.norm.ppf(1 - confidence)
            z_cf = z + (z**2 - 1) * skew / 6 + \
                   (z**3 - 3*z) * kurt / 24 - \
                   (2*z**3 - 5*z) * skew**2 / 36

            var = -(mu + z_cf * sigma)

        else:
            var = -np.percentile(returns, (1 - confidence) * 100)

        return max(0, var)

    def calculate_cvar(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).

        CVaR = E[loss | loss > VaR]
        """
        if len(returns) < 10:
            return 0.03  # Default 3%

        var = self.calculate_var(returns, confidence)
        tail_returns = returns[returns < -var]

        if len(tail_returns) > 0:
            cvar = -np.mean(tail_returns)
        else:
            cvar = var * 1.5  # Approximate

        return cvar

    def calculate_tail_risk(
        self,
        returns: np.ndarray,
        threshold: float = 0.05
    ) -> float:
        """
        Calculate tail risk using Hill estimator.

        Estimates the tail index (alpha) of the return distribution.
        """
        if len(returns) < 50:
            return 0.05

        # Use losses (negative returns)
        losses = -returns[returns < 0]

        if len(losses) < 20:
            return 0.05

        # Sort losses
        sorted_losses = np.sort(losses)[::-1]

        # Number of extreme observations
        k = int(len(losses) * threshold)
        k = max(k, 10)

        if k >= len(sorted_losses):
            return 0.05

        # Hill estimator
        threshold_value = sorted_losses[k]
        extreme_losses = sorted_losses[:k]

        if threshold_value <= 0:
            return 0.05

        log_exceedances = np.log(extreme_losses / threshold_value)
        hill_alpha = k / np.sum(log_exceedances)

        # Expected tail loss
        tail_risk = threshold_value * (hill_alpha / (hill_alpha - 1)) if hill_alpha > 1 else threshold_value * 2

        return tail_risk

    def calculate_component_var(
        self,
        weights: np.ndarray,
        returns_matrix: np.ndarray,
        confidence: float = 0.95
    ) -> np.ndarray:
        """
        Calculate component VaR for each position.

        Component VaR = weight * marginal VaR
        Sum of component VaRs = portfolio VaR
        """
        if returns_matrix.shape[0] < 10:
            return weights * 0.02

        # Portfolio returns
        portfolio_returns = returns_matrix @ weights
        portfolio_var = self.calculate_var(portfolio_returns, confidence)
        portfolio_vol = np.std(portfolio_returns)

        # Covariance
        cov_matrix = np.cov(returns_matrix.T)

        # Marginal VaR
        marginal_var = cov_matrix @ weights / portfolio_vol

        # Component VaR
        component_var = weights * marginal_var * stats.norm.ppf(confidence)

        return component_var

    def calculate_marginal_var(
        self,
        weights: np.ndarray,
        returns_matrix: np.ndarray,
        confidence: float = 0.95
    ) -> np.ndarray:
        """
        Calculate marginal VaR for each position.

        Marginal VaR = change in portfolio VaR for small change in weight
        """
        if returns_matrix.shape[0] < 10:
            return np.ones_like(weights) * 0.02

        portfolio_returns = returns_matrix @ weights
        portfolio_vol = np.std(portfolio_returns)

        cov_matrix = np.cov(returns_matrix.T)

        # Marginal contribution to volatility
        marginal_vol = cov_matrix @ weights / portfolio_vol

        # Scale by VaR multiplier
        z = stats.norm.ppf(confidence)
        marginal_var = marginal_vol * z

        return marginal_var

    def allocate_risk_budget(
        self,
        n_positions: int,
        risk_budgets: Optional[np.ndarray] = None,
        returns_matrix: Optional[np.ndarray] = None,
        method: str = "equal"
    ) -> np.ndarray:
        """
        Allocate risk budget across positions.

        Methods:
        - equal: Equal risk contribution
        - inverse_vol: Proportional to inverse volatility
        - custom: Use provided risk_budgets
        """
        if risk_budgets is not None:
            return risk_budgets / np.sum(risk_budgets)

        if method == "equal":
            return np.ones(n_positions) / n_positions

        elif method == "inverse_vol" and returns_matrix is not None:
            vols = np.std(returns_matrix, axis=0)
            inverse_vols = 1 / (vols + 1e-10)
            return inverse_vols / np.sum(inverse_vols)

        return np.ones(n_positions) / n_positions

    def calculate_risk_budget(
        self,
        position_weight: float,
        returns: np.ndarray,
        portfolio_returns: np.ndarray
    ) -> RiskBudget:
        """
        Calculate comprehensive risk budget for a position.
        """
        if len(returns) < 10:
            return RiskBudget(
                var_budget=0.02,
                cvar_budget=0.03,
                vol_budget=0.15,
                tail_risk_budget=0.05,
                marginal_var=0.02,
                component_var=position_weight * 0.02,
                risk_contribution_pct=position_weight * 100
            )

        # Individual metrics
        var = self.calculate_var(returns, self.config.var_confidence)
        cvar = self.calculate_cvar(returns, self.config.cvar_confidence)
        vol = np.std(returns) * np.sqrt(252)
        tail = self.calculate_tail_risk(returns)

        # Portfolio context
        if len(portfolio_returns) > 10:
            port_var = self.calculate_var(portfolio_returns, self.config.var_confidence)
            port_vol = np.std(portfolio_returns) * np.sqrt(252)

            # Marginal contribution
            corr = np.corrcoef(returns, portfolio_returns)[0, 1] if len(returns) == len(portfolio_returns) else 0.5
            marginal = position_weight * vol * corr
            component = position_weight * marginal
            contribution = (component / port_vol * 100) if port_vol > 0 else position_weight * 100
        else:
            marginal = var
            component = position_weight * var
            contribution = position_weight * 100

        return RiskBudget(
            var_budget=var,
            cvar_budget=cvar,
            vol_budget=vol,
            tail_risk_budget=tail,
            marginal_var=marginal,
            component_var=component,
            risk_contribution_pct=contribution
        )


# =============================================================================
# PORTFOLIO ALLOCATION ENGINE
# =============================================================================

class PortfolioAllocationEngine:
    """
    Portfolio-level allocation methods.

    Implements:
    - Equal weight
    - Inverse volatility
    - Risk Parity
    - Hierarchical Risk Parity (HRP)
    - Maximum Diversification
    - Minimum Variance
    - Mean-CVaR optimization
    """

    def __init__(self, config: PositionSizeConfig):
        self.config = config

    def equal_weight(self, n_positions: int) -> np.ndarray:
        """Equal weight allocation."""
        return np.ones(n_positions) / n_positions

    def inverse_volatility(
        self,
        volatilities: np.ndarray
    ) -> np.ndarray:
        """Inverse volatility weighting."""
        inverse_vols = 1 / (volatilities + 1e-10)
        return inverse_vols / np.sum(inverse_vols)

    def risk_parity(
        self,
        cov_matrix: np.ndarray,
        risk_budgets: Optional[np.ndarray] = None,
        max_iterations: int = 1000,
        tolerance: float = 1e-8
    ) -> np.ndarray:
        """
        Risk Parity allocation.

        Each position contributes equally to total portfolio risk.
        """
        n = cov_matrix.shape[0]

        if risk_budgets is None:
            risk_budgets = np.ones(n) / n

        # Initial weights
        weights = np.ones(n) / n

        for _ in range(max_iterations):
            # Portfolio volatility
            port_var = weights @ cov_matrix @ weights
            port_vol = np.sqrt(port_var)

            # Marginal risk contributions
            mrc = (cov_matrix @ weights) / port_vol

            # Risk contributions
            rc = weights * mrc

            # Target risk contributions
            target_rc = risk_budgets * port_vol

            # Update weights
            new_weights = weights * target_rc / (rc + 1e-10)
            new_weights = new_weights / np.sum(new_weights)

            # Check convergence
            if np.max(np.abs(new_weights - weights)) < tolerance:
                break

            weights = new_weights

        return weights

    def hierarchical_risk_parity(
        self,
        returns_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Hierarchical Risk Parity (HRP) allocation.

        Uses hierarchical clustering to improve diversification.
        """
        n = returns_matrix.shape[1]

        if n < 2:
            return np.ones(n) / n

        # Correlation matrix
        corr_matrix = np.corrcoef(returns_matrix.T)

        # Distance matrix
        dist_matrix = np.sqrt((1 - corr_matrix) / 2)
        np.fill_diagonal(dist_matrix, 0)

        # Hierarchical clustering
        try:
            condensed_dist = squareform(dist_matrix)
            link = linkage(condensed_dist, method='ward')

            # Get cluster order
            sorted_idx = self._get_quasi_diag(link)

            # Reorder covariance matrix
            cov_matrix = np.cov(returns_matrix.T)
            sorted_cov = cov_matrix[sorted_idx][:, sorted_idx]

            # Recursive bisection
            weights = self._recursive_bisection(sorted_cov)

            # Reorder weights back
            final_weights = np.zeros(n)
            for i, idx in enumerate(sorted_idx):
                final_weights[idx] = weights[i]

            return final_weights

        except Exception:
            # Fallback to inverse vol
            vols = np.std(returns_matrix, axis=0)
            return self.inverse_volatility(vols)

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """Get quasi-diagonal order from hierarchical clustering."""
        link = link.astype(int)
        sort_idx = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]

        while sort_idx.max() >= num_items:
            sort_idx.index = range(0, sort_idx.shape[0] * 2, 2)
            df0 = sort_idx[sort_idx >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_idx[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_idx = pd.concat([sort_idx, df0])
            sort_idx = sort_idx.sort_index()
            sort_idx.index = range(sort_idx.shape[0])

        return sort_idx.tolist()

    def _recursive_bisection(
        self,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Recursive bisection for HRP."""
        n = cov_matrix.shape[0]
        weights = np.ones(n)
        items = [list(range(n))]

        while len(items) > 0:
            new_items = []
            for subset in items:
                if len(subset) <= 1:
                    continue

                # Split into two clusters
                mid = len(subset) // 2
                left = subset[:mid]
                right = subset[mid:]

                # Variance of each cluster
                var_left = self._cluster_variance(cov_matrix, left)
                var_right = self._cluster_variance(cov_matrix, right)

                # Allocate based on inverse variance
                alpha = 1 - var_left / (var_left + var_right)

                weights[left] *= alpha
                weights[right] *= (1 - alpha)

                if len(left) > 1:
                    new_items.append(left)
                if len(right) > 1:
                    new_items.append(right)

            items = new_items

        return weights / np.sum(weights)

    def _cluster_variance(
        self,
        cov_matrix: np.ndarray,
        cluster_idx: List[int]
    ) -> float:
        """Calculate variance of a cluster."""
        cov_slice = cov_matrix[np.ix_(cluster_idx, cluster_idx)]
        n = len(cluster_idx)
        weights = np.ones(n) / n
        return weights @ cov_slice @ weights

    def maximum_diversification(
        self,
        cov_matrix: np.ndarray,
        volatilities: np.ndarray
    ) -> np.ndarray:
        """
        Maximum Diversification portfolio.

        Maximizes diversification ratio = weighted avg vol / portfolio vol
        """
        n = len(volatilities)

        def neg_div_ratio(w):
            port_vol = np.sqrt(w @ cov_matrix @ w)
            weighted_vol = w @ volatilities
            return -weighted_vol / (port_vol + 1e-10)

        # Constraints: weights sum to 1, all positive
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n)]

        # Initial guess
        x0 = np.ones(n) / n

        try:
            result = optimize.minimize(
                neg_div_ratio,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                return result.x
        except Exception:
            pass

        return self.inverse_volatility(volatilities)

    def minimum_variance(
        self,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Minimum Variance portfolio.

        Minimizes portfolio variance.
        """
        n = cov_matrix.shape[0]

        def portfolio_var(w):
            return w @ cov_matrix @ w

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n)]
        x0 = np.ones(n) / n

        try:
            result = optimize.minimize(
                portfolio_var,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                return result.x
        except Exception:
            pass

        vols = np.sqrt(np.diag(cov_matrix))
        return self.inverse_volatility(vols)

    def mean_cvar_optimization(
        self,
        returns_matrix: np.ndarray,
        expected_returns: np.ndarray,
        confidence: float = 0.95,
        risk_aversion: float = 1.0
    ) -> np.ndarray:
        """
        Mean-CVaR optimization.

        Maximizes expected return - risk_aversion * CVaR
        """
        n = returns_matrix.shape[1]

        def objective(w):
            port_returns = returns_matrix @ w
            expected = w @ expected_returns

            # CVaR
            var = np.percentile(port_returns, (1 - confidence) * 100)
            tail_returns = port_returns[port_returns < var]
            cvar = -np.mean(tail_returns) if len(tail_returns) > 0 else -var

            return -(expected - risk_aversion * cvar)

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n)]
        x0 = np.ones(n) / n

        try:
            result = optimize.minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                return result.x
        except Exception:
            pass

        return np.ones(n) / n

    def allocate(
        self,
        returns_matrix: np.ndarray,
        method: Optional[AllocationMethod] = None,
        expected_returns: Optional[np.ndarray] = None,
        risk_budgets: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, bool]:
        """
        Allocate portfolio using specified method.

        Returns:
            (weights, converged)
        """
        method = method or self.config.allocation_method
        n = returns_matrix.shape[1]

        if n == 0:
            return np.array([]), True

        if n == 1:
            return np.array([1.0]), True

        try:
            cov_matrix = np.cov(returns_matrix.T)
            volatilities = np.std(returns_matrix, axis=0)

            if method == AllocationMethod.EQUAL_WEIGHT:
                return self.equal_weight(n), True

            elif method == AllocationMethod.INVERSE_VOL:
                return self.inverse_volatility(volatilities), True

            elif method == AllocationMethod.RISK_PARITY:
                return self.risk_parity(cov_matrix, risk_budgets), True

            elif method == AllocationMethod.HRP:
                return self.hierarchical_risk_parity(returns_matrix), True

            elif method == AllocationMethod.MAX_DIVERSIFICATION:
                return self.maximum_diversification(cov_matrix, volatilities), True

            elif method == AllocationMethod.MIN_VARIANCE:
                return self.minimum_variance(cov_matrix), True

            elif method == AllocationMethod.MEAN_CVAR:
                if expected_returns is None:
                    expected_returns = np.mean(returns_matrix, axis=0)
                return self.mean_cvar_optimization(returns_matrix, expected_returns), True

            else:
                return self.equal_weight(n), True

        except Exception as e:
            logger.warning(f"Allocation failed: {e}, falling back to equal weight")
            return self.equal_weight(n), False


# =============================================================================
# MAIN POSITION SIZING ENGINE
# =============================================================================

class PositionSizingEngine:
    """
    Comprehensive position sizing engine with comprehensive quantitative methods.

    Implements all PDF Section 2.4 requirements:
    - Venue-specific position limits (CEX $100k, DEX $20-50k liquid, $5-10k illiquid)
    - Kelly Criterion with multiple variants (Full, Half, Quarter, Optimal-f, Secure, Bayesian)
    - Volatility targeting with GARCH/EWMA forecasting
    - Liquidity-aware sizing with Kyle's Lambda and Amihud ratio
    - Risk parity and hierarchical risk parity allocation
    - VaR/CVaR-constrained sizing
    - Regime-based adjustments (crisis, high vol, low vol)
    - Monte Carlo position validation
    - PDF concentration limits (40% sector, 60% CEX, 20% Tier 3)
    """

    def __init__(self, config: Optional[PositionSizeConfig] = None):
        """Initialize with configuration and sub-engines."""
        self.config = config or PositionSizeConfig()

        # Initialize sub-engines
        self.kelly_engine = KellyCriterionEngine(self.config)
        self.volatility_engine = VolatilityEngine(self.config)
        self.liquidity_engine = LiquidityAnalysisEngine(self.config)
        self.risk_budget_engine = RiskBudgetEngine(self.config)
        self.allocation_engine = PortfolioAllocationEngine(self.config)

        # Build lookup tables
        self._venue_limits = self._build_venue_limits()
        self._tier_limits = self._build_tier_limits()

        logger.info("PositionSizingEngine initialized with production configuration")

    def _build_venue_limits(self) -> Dict[VenueType, Dict[str, float]]:
        """Build venue-specific position limits from config."""
        return {
            VenueType.CEX: {
                'max': self.config.cex_max_position,
                'min': self.config.cex_min_position,
                'target': self.config.cex_target_position
            },
            VenueType.DEX_LIQUID: {
                'max': self.config.dex_liquid_max_position,
                'min': self.config.dex_liquid_min_position,
                'target': self.config.dex_liquid_target_position
            },
            VenueType.DEX_ILLIQUID: {
                'max': self.config.dex_illiquid_max_position,
                'min': self.config.dex_illiquid_min_position,
                'target': self.config.dex_illiquid_target_position
            },
            VenueType.HYBRID: {
                'max': (self.config.cex_max_position + self.config.dex_liquid_max_position) / 2,
                'min': min(self.config.cex_min_position, self.config.dex_liquid_min_position),
                'target': (self.config.cex_target_position + self.config.dex_liquid_target_position) / 2
            }
        }

    def _build_tier_limits(self) -> Dict[LiquidityTier, float]:
        """Build tier-specific allocation limits."""
        return {
            LiquidityTier.TIER1: self.config.tier1_allocation_max,
            LiquidityTier.TIER2: self.config.tier2_allocation_max,
            LiquidityTier.TIER3: self.config.tier3_allocation_max
        }

    def classify_venue_type(
        self,
        venue: str,
        daily_volume_usd: float
    ) -> VenueType:
        """Classify venue type based on venue name and liquidity."""
        venue_lower = venue.lower()

        cex_venues = ['binance', 'coinbase', 'kraken', 'okx', 'bybit', 'kucoin',
                      'gate', 'huobi', 'bitfinex', 'gemini', 'ftx', 'bitstamp',
                      'bitget', 'mexc', 'crypto.com', 'phemex']

        dex_venues = ['uniswap', 'sushiswap', 'curve', 'balancer', 'pancakeswap',
                      'dydx', 'gmx', 'perpetual', 'quickswap', 'trader_joe',
                      'raydium', 'orca', 'serum', 'jupiter', 'velodrome']

        is_cex = any(cex in venue_lower for cex in cex_venues)
        is_dex = any(dex in venue_lower for dex in dex_venues)

        if is_cex and is_dex:
            return VenueType.HYBRID
        elif is_cex:
            return VenueType.CEX
        elif is_dex:
            if daily_volume_usd >= 1_000_000:
                return VenueType.DEX_LIQUID
            else:
                return VenueType.DEX_ILLIQUID
        else:
            if daily_volume_usd >= 10_000_000:
                return VenueType.CEX
            elif daily_volume_usd >= 1_000_000:
                return VenueType.DEX_LIQUID
            else:
                return VenueType.DEX_ILLIQUID

    def classify_liquidity_tier(
        self,
        volume_rank: int,
        market_cap_rank: int
    ) -> LiquidityTier:
        """Classify liquidity tier based on rankings."""
        combined_rank = (volume_rank + market_cap_rank) / 2

        if combined_rank <= 10:
            return LiquidityTier.TIER1
        elif combined_rank <= 30:
            return LiquidityTier.TIER2
        else:
            return LiquidityTier.TIER3

    def calculate_regime_scalar(
        self,
        regime: SizingRegime
    ) -> float:
        """Calculate regime-based position scalar."""
        scalars = {
            SizingRegime.NORMAL: 1.0,
            SizingRegime.HIGH_VOLATILITY: 1.0 - self.config.high_vol_position_reduction,
            SizingRegime.LOW_VOLATILITY: self.config.low_vol_position_increase,
            SizingRegime.CRISIS: self.config.crisis_position_reduction,
            SizingRegime.RECOVERY: self.config.recovery_position_scalar,
            SizingRegime.TRENDING: 1.1,
            SizingRegime.MEAN_REVERTING: 1.0
        }
        return scalars.get(regime, 1.0)

    def calculate_drawdown_scalar(
        self,
        current_drawdown: float
    ) -> float:
        """Calculate drawdown-based position reduction."""
        if not self.config.drawdown_scaling_enabled:
            return 1.0

        if current_drawdown <= 0:
            return 1.0

        if current_drawdown >= self.config.max_acceptable_drawdown:
            return self.config.min_drawdown_scalar

        # Linear reduction as drawdown increases
        reduction = (current_drawdown / self.config.max_acceptable_drawdown) * \
                   self.config.drawdown_reduction_rate

        return max(self.config.min_drawdown_scalar, 1.0 - reduction)

    def calculate_correlation_scalar(
        self,
        correlation_to_portfolio: float,
        portfolio_concentration: float = 0.0
    ) -> float:
        """Calculate correlation-based position adjustment."""
        if not self.config.correlation_scaling_enabled:
            return 1.0

        if correlation_to_portfolio >= self.config.high_correlation_threshold:
            scalar = self.config.correlation_reduction_factor
        elif correlation_to_portfolio >= 0.5:
            scalar = 0.75
        elif correlation_to_portfolio >= 0.3:
            scalar = 0.9
        else:
            scalar = 1.0

        # Additional reduction if portfolio concentrated
        if portfolio_concentration > 0.6:
            scalar *= 0.8

        return scalar

    def validate_with_monte_carlo(
        self,
        position_size_pct: float,
        returns: np.ndarray,
        n_simulations: Optional[int] = None,
        horizon: Optional[int] = None
    ) -> Tuple[bool, float, float]:
        """
        Validate position size using Monte Carlo simulation.

        Returns:
            (is_valid, risk_of_ruin, expected_return)
        """
        if not self.config.monte_carlo_enabled or len(returns) < 30:
            return True, 0.0, 0.0

        n_sims = n_simulations or self.config.monte_carlo_simulations
        horizon = horizon or self.config.monte_carlo_horizon_days

        final_values = []
        ruin_count = 0
        ruin_threshold = self.config.ruin_threshold

        for _ in range(n_sims):
            equity = 1.0
            # Block bootstrap
            block_size = self.config.monte_carlo_block_size
            n_blocks = horizon // block_size + 1

            for _ in range(n_blocks):
                start = np.random.randint(0, max(1, len(returns) - block_size))
                block = returns[start:start + block_size]

                for ret in block:
                    equity *= (1 + position_size_pct * ret)
                    if equity < (1 - ruin_threshold):
                        ruin_count += 1
                        break

                if equity < (1 - ruin_threshold):
                    break

            final_values.append(equity)

        risk_of_ruin = ruin_count / n_sims
        expected_return = np.mean(final_values) - 1

        is_valid = risk_of_ruin <= self.config.max_risk_of_ruin

        return is_valid, risk_of_ruin, expected_return

    def calculate_position_size(
        self,
        pair_id: str,
        venue: str,
        total_capital: float,
        pair_data: Dict[str, Any]
    ) -> PositionSizeResult:
        """
        Calculate comprehensive position size for a single pair.

        Args:
            pair_id: Unique pair identifier
            venue: Venue name
            total_capital: Total portfolio capital
            pair_data: Dictionary containing:
                - returns: Array of historical returns
                - daily_volume_usd: Average daily volume
                - volumes_usd: Array of daily volumes
                - volume_rank: Ranking by volume
                - market_cap_rank: Ranking by market cap
                - trade_returns: Array of trade P&L returns
                - portfolio_returns: Array of portfolio returns
                - correlation_to_portfolio: Correlation to existing portfolio
                - current_drawdown: Current portfolio drawdown
                - ohlc_data: Optional OHLC data for volatility estimation

        Returns:
            Comprehensive PositionSizeResult
        """
        warnings = []
        adjustment_breakdown = {}

        # Extract data with defaults
        returns = pair_data.get('returns', np.array([]))
        daily_volume = pair_data.get('daily_volume_usd', 1_000_000)
        volumes_usd = pair_data.get('volumes_usd', np.array([daily_volume] * 60))
        volume_rank = pair_data.get('volume_rank', 50)
        market_cap_rank = pair_data.get('market_cap_rank', 50)
        trade_returns = pair_data.get('trade_returns', np.array([]))
        portfolio_returns = pair_data.get('portfolio_returns', np.array([]))
        correlation_to_portfolio = pair_data.get('correlation_to_portfolio', 0.5)
        current_drawdown = pair_data.get('current_drawdown', 0.0)
        ohlc_data = pair_data.get('ohlc_data')
        n_positions = pair_data.get('n_positions', 10)

        # Classify venue and tier
        venue_type = self.classify_venue_type(venue, daily_volume)
        liquidity_tier = self.classify_liquidity_tier(volume_rank, market_cap_rank)

        # Get venue-specific limits
        limits = self._venue_limits[venue_type]
        base_position = limits['target']
        max_position = limits['max']
        min_position = limits['min']

        # 1. Kelly Criterion Analysis
        kelly_result = None
        kelly_multiplier = 1.0

        if len(trade_returns) >= self.config.kelly_min_trades:
            kelly_result = self.kelly_engine.calculate_kelly(
                trade_returns,
                variant=self.config.kelly_variant,
                portfolio_correlation=correlation_to_portfolio,
                n_positions=n_positions
            )
            kelly_multiplier = kelly_result.fractional_kelly
            warnings.extend(kelly_result.warnings)
        else:
            kelly_multiplier = 1.0
            warnings.append("Insufficient trades for Kelly - using default sizing")

        adjustment_breakdown['kelly'] = kelly_multiplier

        # 2. Volatility Analysis
        volatility_estimate = None
        volatility_scalar = 1.0

        if len(returns) >= 20:
            volatility_estimate = self.volatility_engine.estimate_volatility(
                returns, ohlc_data
            )

            # Scale inversely with volatility
            if volatility_estimate.current_vol > 0:
                vol_ratio = self.config.target_portfolio_vol / volatility_estimate.current_vol
                volatility_scalar = np.clip(vol_ratio, 0.3, 2.0)

        adjustment_breakdown['volatility'] = volatility_scalar

        # 3. Liquidity Analysis
        liquidity_analysis = None
        liquidity_scalar = 1.0

        target_position = base_position * kelly_multiplier * volatility_scalar

        if len(returns) >= 10 and len(volumes_usd) >= 10:
            liquidity_analysis = self.liquidity_engine.analyze_liquidity(
                returns, volumes_usd, target_position
            )

            # Scale based on ADV constraint
            adv_pct = target_position / liquidity_analysis.adv_usd if liquidity_analysis.adv_usd > 0 else 1.0
            max_adv = self.config.max_adv_multiple if venue_type == VenueType.CEX else self.config.min_adv_multiple

            if adv_pct > max_adv:
                liquidity_scalar = max_adv / adv_pct

            warnings.extend(liquidity_analysis.warnings)

        adjustment_breakdown['liquidity'] = liquidity_scalar

        # 4. Risk Budget Analysis
        risk_budget = None
        risk_scalar = 1.0

        position_weight = target_position / total_capital if total_capital > 0 else 0

        if len(returns) >= 10:
            risk_budget = self.risk_budget_engine.calculate_risk_budget(
                position_weight, returns, portfolio_returns
            )

            # Scale based on risk limits
            if risk_budget.var_budget > self.config.max_position_var:
                var_scalar = self.config.max_position_var / risk_budget.var_budget
                risk_scalar = min(risk_scalar, var_scalar)

            if risk_budget.cvar_budget > self.config.max_position_cvar:
                cvar_scalar = self.config.max_position_cvar / risk_budget.cvar_budget
                risk_scalar = min(risk_scalar, cvar_scalar)

        adjustment_breakdown['risk'] = risk_scalar

        # 5. Regime Analysis
        regime = SizingRegime.NORMAL
        if volatility_estimate:
            regime = volatility_estimate.regime

        regime_scalar = self.calculate_regime_scalar(regime)
        adjustment_breakdown['regime'] = regime_scalar

        # 6. Correlation Adjustment
        correlation_scalar = self.calculate_correlation_scalar(
            correlation_to_portfolio
        )
        adjustment_breakdown['correlation'] = correlation_scalar

        # 7. Drawdown Adjustment
        drawdown_scalar = self.calculate_drawdown_scalar(current_drawdown)
        adjustment_breakdown['drawdown'] = drawdown_scalar

        # Calculate final position with all adjustments
        final_position = base_position * kelly_multiplier * volatility_scalar * \
                        liquidity_scalar * risk_scalar * regime_scalar * \
                        correlation_scalar * drawdown_scalar

        # Apply venue-specific bounds
        final_position = np.clip(final_position, min_position, max_position)

        # Apply tier-specific limits
        tier_limit = self._tier_limits[liquidity_tier]
        max_tier_position = total_capital * tier_limit * self.config.max_single_position
        final_position = min(final_position, max_tier_position)

        # Apply max single position limit
        max_single = total_capital * self.config.max_single_position
        final_position = min(final_position, max_single)

        # Monte Carlo validation
        mc_valid = True
        mc_risk_of_ruin = 0.0
        mc_expected_return = 0.0

        if self.config.monte_carlo_enabled and len(returns) >= 30:
            position_pct = final_position / total_capital if total_capital > 0 else 0
            mc_valid, mc_risk_of_ruin, mc_expected_return = self.validate_with_monte_carlo(
                position_pct, returns
            )

            if not mc_valid:
                warnings.append(f"Monte Carlo validation failed: RoR={mc_risk_of_ruin:.2%}")
                # Reduce position to pass validation
                final_position *= 0.7

        # Calculate final metrics
        final_pct = final_position / total_capital if total_capital > 0 else 0
        adv_pct = final_position / daily_volume if daily_volume > 0 else 0
        slippage_bps = liquidity_analysis.expected_slippage_bps if liquidity_analysis else 15.0
        var_util = risk_budget.var_budget / self.config.max_position_var if risk_budget else 0
        cvar_util = risk_budget.cvar_budget / self.config.max_position_cvar if risk_budget else 0

        # Tier warnings
        if liquidity_tier == LiquidityTier.TIER3:
            warnings.append("Tier 3 asset - 20% portfolio limit per PDF")

        return PositionSizeResult(
            pair_id=pair_id,
            venue_type=venue_type,
            liquidity_tier=liquidity_tier,
            base_position_usd=base_position,
            target_position_usd=target_position,
            max_position_usd=max_position,
            min_position_usd=min_position,
            kelly_result=kelly_result,
            kelly_multiplier=kelly_multiplier,
            volatility_estimate=volatility_estimate,
            volatility_scalar=volatility_scalar,
            liquidity_analysis=liquidity_analysis,
            liquidity_scalar=liquidity_scalar,
            risk_budget=risk_budget,
            risk_scalar=risk_scalar,
            regime=regime,
            regime_scalar=regime_scalar,
            correlation_to_portfolio=correlation_to_portfolio,
            correlation_scalar=correlation_scalar,
            current_drawdown=current_drawdown,
            drawdown_scalar=drawdown_scalar,
            final_position_usd=final_position,
            final_position_pct_portfolio=final_pct,
            adv_pct=adv_pct,
            slippage_estimate_bps=slippage_bps,
            var_utilization=var_util,
            cvar_utilization=cvar_util,
            monte_carlo_valid=mc_valid,
            monte_carlo_risk_of_ruin=mc_risk_of_ruin,
            monte_carlo_expected_return=mc_expected_return,
            warnings=warnings,
            adjustment_breakdown=adjustment_breakdown
        )

    def calculate_portfolio_positions(
        self,
        total_capital: float,
        pairs_data: Dict[str, Dict[str, Any]],
        sector_allocations: Optional[Dict[str, str]] = None,
        returns_matrix: Optional[np.ndarray] = None
    ) -> PortfolioSizeResult:
        """
        Calculate position sizes for entire portfolio with concentration limits.

        Enforces PDF constraints:
        - 40% max per sector
        - 60% max CEX-only concentration
        - 20% max Tier 3 allocation

        Args:
            total_capital: Total portfolio capital
            pairs_data: Dictionary of pair_id -> pair data
            sector_allocations: Optional mapping of pair_id -> sector
            returns_matrix: Optional matrix of returns for portfolio optimization
        """
        n_positions = len(pairs_data)

        # Track allocations
        cex_allocated = 0.0
        dex_liquid_allocated = 0.0
        dex_illiquid_allocated = 0.0
        hybrid_allocated = 0.0

        tier1_allocated = 0.0
        tier2_allocated = 0.0
        tier3_allocated = 0.0

        sector_allocated: Dict[str, float] = defaultdict(float)
        position_sizes: Dict[str, PositionSizeResult] = {}

        # Phase 1: Calculate raw position sizes
        raw_positions: Dict[str, PositionSizeResult] = {}

        for pair_id, pair_data in pairs_data.items():
            venue = pair_data.get('venue', 'binance')
            pair_data['n_positions'] = n_positions

            result = self.calculate_position_size(
                pair_id, venue, total_capital, pair_data
            )
            raw_positions[pair_id] = result

        # Phase 2: Portfolio-level allocation if returns available
        allocation_weights = None
        allocation_converged = True

        if returns_matrix is not None and returns_matrix.shape[1] == n_positions:
            try:
                pair_ids = list(pairs_data.keys())
                allocation_weights, allocation_converged = self.allocation_engine.allocate(
                    returns_matrix,
                    method=self.config.allocation_method
                )

                # Adjust positions based on allocation weights
                for i, pair_id in enumerate(pair_ids):
                    if pair_id in raw_positions:
                        # Scale position by relative weight
                        weight_scalar = allocation_weights[i] * n_positions
                        raw_positions[pair_id].final_position_usd *= weight_scalar
                        raw_positions[pair_id].adjustment_breakdown['allocation'] = weight_scalar

            except Exception as e:
                logger.warning(f"Portfolio allocation failed: {e}")
                allocation_converged = False

        # Phase 3: Apply concentration constraints
        sorted_positions = sorted(
            raw_positions.items(),
            key=lambda x: x[1].final_position_usd,
            reverse=True
        )

        for pair_id, result in sorted_positions:
            # Check Tier 3 constraint (20% max per PDF)
            if result.liquidity_tier == LiquidityTier.TIER3:
                remaining_tier3 = (self.config.tier3_allocation_max * total_capital) - tier3_allocated
                if remaining_tier3 <= 0:
                    result.final_position_usd = 0
                    result.warnings.append("Tier 3 limit reached - position excluded")
                else:
                    result.final_position_usd = min(result.final_position_usd, remaining_tier3)

            # Check CEX concentration constraint (60% max per PDF)
            if result.venue_type == VenueType.CEX:
                remaining_cex = (self.config.max_cex_concentration * total_capital) - cex_allocated
                if result.final_position_usd > remaining_cex:
                    result.final_position_usd = max(0, remaining_cex)
                    result.warnings.append("CEX concentration limit applied")

            # Check sector concentration (40% max per PDF)
            if sector_allocations:
                sector = sector_allocations.get(pair_id, 'other')
                current_sector = sector_allocated[sector]
                remaining_sector = (self.config.max_sector_concentration * total_capital) - current_sector
                if result.final_position_usd > remaining_sector:
                    result.final_position_usd = max(0, remaining_sector)
                    result.warnings.append(f"Sector limit for {sector} applied")
                sector_allocated[sector] = current_sector + result.final_position_usd

            # Update allocations
            if result.final_position_usd > 0:
                position_sizes[pair_id] = result
                result.final_position_pct_portfolio = result.final_position_usd / total_capital

                # Track by venue type
                if result.venue_type == VenueType.CEX:
                    cex_allocated += result.final_position_usd
                elif result.venue_type == VenueType.DEX_LIQUID:
                    dex_liquid_allocated += result.final_position_usd
                elif result.venue_type == VenueType.DEX_ILLIQUID:
                    dex_illiquid_allocated += result.final_position_usd
                else:
                    hybrid_allocated += result.final_position_usd

                # Track by tier
                if result.liquidity_tier == LiquidityTier.TIER1:
                    tier1_allocated += result.final_position_usd
                elif result.liquidity_tier == LiquidityTier.TIER2:
                    tier2_allocated += result.final_position_usd
                else:
                    tier3_allocated += result.final_position_usd

        # Calculate portfolio metrics
        total_allocated = cex_allocated + dex_liquid_allocated + dex_illiquid_allocated + hybrid_allocated

        # Diversification metrics
        weights = [p.final_position_pct_portfolio for p in position_sizes.values()]
        if weights:
            herfindahl = sum(w**2 for w in weights)
            effective_n = 1 / herfindahl if herfindahl > 0 else len(weights)
        else:
            herfindahl = 0
            effective_n = 0

        # Portfolio risk metrics
        portfolio_var = 0.0
        portfolio_cvar = 0.0
        portfolio_vol = 0.0

        if returns_matrix is not None and allocation_weights is not None:
            try:
                port_returns = returns_matrix @ allocation_weights
                portfolio_var = self.risk_budget_engine.calculate_var(port_returns)
                portfolio_cvar = self.risk_budget_engine.calculate_cvar(port_returns)
                portfolio_vol = np.std(port_returns) * np.sqrt(252)
            except Exception:
                pass

        # Constraint utilization
        tier3_util = tier3_allocated / total_capital if total_capital > 0 else 0
        cex_conc = cex_allocated / total_capital if total_capital > 0 else 0
        max_sector_conc = max(sector_allocated.values()) / total_capital if sector_allocated and total_capital > 0 else 0
        max_single_pct = max(weights) if weights else 0

        return PortfolioSizeResult(
            total_capital=total_capital,
            allocated_capital=total_allocated,
            unallocated_capital=total_capital - total_allocated,
            cex_allocated=cex_allocated,
            dex_liquid_allocated=dex_liquid_allocated,
            dex_illiquid_allocated=dex_illiquid_allocated,
            hybrid_allocated=hybrid_allocated,
            tier1_allocated=tier1_allocated,
            tier2_allocated=tier2_allocated,
            tier3_allocated=tier3_allocated,
            total_positions=len(position_sizes),
            cex_positions=sum(1 for p in position_sizes.values() if p.venue_type == VenueType.CEX),
            dex_positions=sum(1 for p in position_sizes.values() if p.venue_type in [VenueType.DEX_LIQUID, VenueType.DEX_ILLIQUID]),
            position_sizes=position_sizes,
            portfolio_var=portfolio_var,
            portfolio_cvar=portfolio_cvar,
            portfolio_volatility=portfolio_vol,
            effective_n=effective_n,
            herfindahl_index=herfindahl,
            tier3_limit_utilized=tier3_util,
            cex_concentration=cex_conc,
            max_sector_concentration=max_sector_conc,
            max_single_position_pct=max_single_pct,
            allocation_method=self.config.allocation_method,
            optimization_converged=allocation_converged
        )

    def generate_sizing_report(
        self,
        result: PortfolioSizeResult,
        detailed: bool = True
    ) -> str:
        """Generate comprehensive position sizing report."""
        lines = [
            "=" * 100,
            "POSITION SIZING REPORT - PDF Section 2.4 Compliant",
            "=" * 100,
            "",
            "PORTFOLIO SUMMARY",
            "-" * 50,
            f"Total Capital:         ${result.total_capital:>15,.0f}",
            f"Allocated Capital:     ${result.allocated_capital:>15,.0f} ({result.allocated_capital/result.total_capital:.1%})",
            f"Unallocated:           ${result.unallocated_capital:>15,.0f}",
            f"Total Positions:       {result.total_positions:>15d}",
            f"Allocation Method:     {result.allocation_method.value:>15s}",
            "",
            "VENUE ALLOCATION",
            "-" * 50,
            f"CEX Positions:         {result.cex_positions:>5d}  (${result.cex_allocated:>12,.0f})",
            f"DEX Liquid:            {'N/A':>5s}  (${result.dex_liquid_allocated:>12,.0f})",
            f"DEX Illiquid:          {'N/A':>5s}  (${result.dex_illiquid_allocated:>12,.0f})",
            f"Hybrid:                {'N/A':>5s}  (${result.hybrid_allocated:>12,.0f})",
            "",
            "TIER ALLOCATION",
            "-" * 50,
            f"Tier 1 (Top 10):       ${result.tier1_allocated:>15,.0f}",
            f"Tier 2 (11-30):        ${result.tier2_allocated:>15,.0f}",
            f"Tier 3 (31+):          ${result.tier3_allocated:>15,.0f}",
            "",
            "RISK METRICS",
            "-" * 50,
            f"Portfolio VaR (95%):   {result.portfolio_var:>15.2%}",
            f"Portfolio CVaR (95%):  {result.portfolio_cvar:>15.2%}",
            f"Portfolio Volatility:  {result.portfolio_volatility:>15.2%}",
            f"Effective N:           {result.effective_n:>15.1f}",
            f"Herfindahl Index:      {result.herfindahl_index:>15.4f}",
            "",
            "PDF CONSTRAINT COMPLIANCE",
            "-" * 50,
        ]

        # Check constraints
        tier3_ok = result.tier3_limit_utilized <= 0.20
        cex_ok = result.cex_concentration <= 0.60
        sector_ok = result.max_sector_concentration <= 0.40
        single_ok = result.max_single_position_pct <= 0.10

        lines.extend([
            f"Tier 3 Limit (20% max):      {result.tier3_limit_utilized:>8.1%}  {'PASS' if tier3_ok else 'FAIL'}",
            f"CEX Concentration (60%):     {result.cex_concentration:>8.1%}  {'PASS' if cex_ok else 'FAIL'}",
            f"Max Sector (40% max):        {result.max_sector_concentration:>8.1%}  {'PASS' if sector_ok else 'FAIL'}",
            f"Max Single Position (10%):   {result.max_single_position_pct:>8.1%}  {'PASS' if single_ok else 'FAIL'}",
            "",
            "POSITION LIMITS BY VENUE (PDF VALUES)",
            "-" * 50,
            f"CEX Max Position:      ${self.config.cex_max_position:>12,.0f}",
            f"DEX Liquid Max:        ${self.config.dex_liquid_max_position:>12,.0f}",
            f"DEX Illiquid Max:      ${self.config.dex_illiquid_max_position:>12,.0f}",
        ])

        if detailed and result.position_sizes:
            lines.extend([
                "",
                "TOP POSITIONS BY SIZE",
                "-" * 100,
                f"{'Rank':<5} {'Pair ID':<25} {'Position ($)':<15} {'Pct':>8} {'Venue':<15} {'Tier':<8} {'Kelly':>8}",
                "-" * 100
            ])

            sorted_positions = sorted(
                result.position_sizes.values(),
                key=lambda x: x.final_position_usd,
                reverse=True
            )[:20]

            for i, pos in enumerate(sorted_positions, 1):
                warnings_str = " [!]" if pos.warnings else ""
                lines.append(
                    f"{i:<5d} {pos.pair_id[:25]:<25} ${pos.final_position_usd:>12,.0f} "
                    f"{pos.final_position_pct_portfolio:>7.1%} {pos.venue_type.value:<15} "
                    f"{pos.liquidity_tier.value:<8} {pos.kelly_multiplier:>7.2f}{warnings_str}"
                )

            # Show positions with warnings
            positions_with_warnings = [p for p in result.position_sizes.values() if p.warnings]
            if positions_with_warnings:
                lines.extend([
                    "",
                    "POSITIONS WITH WARNINGS",
                    "-" * 50
                ])
                for pos in positions_with_warnings[:10]:
                    lines.append(f"  {pos.pair_id}: {', '.join(pos.warnings)}")

        lines.extend([
            "",
            "=" * 100,
            f"Report generated: {result.calculation_timestamp.isoformat()}",
            f"Optimization converged: {result.optimization_converged}",
            "=" * 100
        ])

        return "\n".join(lines)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_position_sizing_engine(
    config: Optional[Dict[str, Any]] = None
) -> PositionSizingEngine:
    """
    Factory function to create configured PositionSizingEngine.

    Args:
        config: Optional configuration overrides

    Returns:
        Configured PositionSizingEngine instance
    """
    if config:
        # Handle enum conversions
        if 'kelly_variant' in config and isinstance(config['kelly_variant'], str):
            config['kelly_variant'] = KellyVariant(config['kelly_variant'])
        if 'vol_model' in config and isinstance(config['vol_model'], str):
            config['vol_model'] = VolatilityModel(config['vol_model'])
        if 'risk_measure' in config and isinstance(config['risk_measure'], str):
            config['risk_measure'] = RiskMeasure(config['risk_measure'])
        if 'allocation_method' in config and isinstance(config['allocation_method'], str):
            config['allocation_method'] = AllocationMethod(config['allocation_method'])

        sizing_config = PositionSizeConfig(**config)
    else:
        sizing_config = PositionSizeConfig()

    return PositionSizingEngine(sizing_config)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main Engine
    'PositionSizingEngine',
    'create_position_sizing_engine',

    # Configuration
    'PositionSizeConfig',

    # Results
    'PositionSizeResult',
    'PortfolioSizeResult',
    'KellyResult',
    'VolatilityEstimate',
    'LiquidityAnalysis',
    'RiskBudget',

    # Enums
    'VenueType',
    'LiquidityTier',
    'KellyVariant',
    'VolatilityModel',
    'RiskMeasure',
    'AllocationMethod',
    'SizingRegime',

    # Sub-engines
    'KellyCriterionEngine',
    'VolatilityEngine',
    'LiquidityAnalysisEngine',
    'RiskBudgetEngine',
    'PortfolioAllocationEngine',
]
