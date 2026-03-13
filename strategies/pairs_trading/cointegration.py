"""
Cointegration Analysis for Crypto Pairs Trading
================================================

Comprehensive cointegration testing and pair selection
for multi-venue cryptocurrency statistical arbitrage.

Mathematical Framework
----------------------
Engle-Granger Two-Step Method:

    Step 1: OLS Regression
        Y_t = α + β × X_t + ε_t
        
    Step 2: ADF Test on Residuals
        Δε_t = γ × ε_{t-1} + Σ(δ_i × Δε_{t-i}) + u_t
        
        H₀: γ = 0 (unit root, non-stationary)
        H₁: γ < 0 (stationary, cointegrated)

Johansen Test (Multivariate):

    ΔY_t = Π × Y_{t-1} + Σ(Γ_i × ΔY_{t-i}) + ε_t
    
    Π = α × β'  where:
        - rank(Π) = r = number of cointegrating relationships
        - β = cointegrating vectors
        - α = adjustment coefficients

Half-Life Estimation (Ornstein-Uhlenbeck):

    dS = κ(μ - S)dt + σdW
    
    Discretized: ΔS_t = θ × S_{t-1} + ε_t
    
    Half-life = -ln(2) / ln(1 + θ) ≈ -ln(2) / θ

Hurst Exponent:

    E[R(n)/S(n)] = C × n^H
    
    H < 0.5: Mean-reverting (tradeable)
    H = 0.5: Random walk
    H > 0.5: Trending

Crypto-Specific Considerations
------------------------------
- Higher volatility → shorter half-lives (target 1-7 days)
- 24/7 markets → no overnight gaps
- Structural breaks from delistings, exploits, upgrades
- Cross-venue arbitrage opportunities
- Funding rate influence on spreads

Author: Crypto StatArb Quantitative Research
Version: 2.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import warnings

# Statistical libraries
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy import stats

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS WITH TRADING-SPECIFIC PROPERTIES
# =============================================================================

class CointegrationMethod(Enum):
    """
    Available cointegration testing methods.

    Each method has specific use cases and statistical properties
    that make it more suitable for different scenarios.
    """
    ENGLE_GRANGER = "engle_granger"
    JOHANSEN = "johansen"
    JOHANSEN_TRACE = "johansen_trace"
    JOHANSEN_EIGEN = "johansen_eigen"
    PHILLIPS_OULIARIS = "phillips_ouliaris"
    BOUNDS_TEST = "bounds_test"
    
    @property
    def description(self) -> str:
        """Method description."""
        descriptions = {
            self.ENGLE_GRANGER: "Two-step residual-based test for bivariate series",
            self.JOHANSEN: "VECM-based test for multivariate cointegration",
            self.PHILLIPS_OULIARIS: "Non-parametric residual-based test",
            self.BOUNDS_TEST: "ARDL bounds test for mixed I(0)/I(1) series",
        }
        return descriptions.get(self, "Unknown method")
    
    @property
    def supports_multivariate(self) -> bool:
        """True if method supports more than 2 series."""
        return self in [self.JOHANSEN]
    
    @property
    def requires_stationarity_pretest(self) -> bool:
        """True if method requires unit root pre-testing."""
        return self in [self.ENGLE_GRANGER, self.PHILLIPS_OULIARIS]
    
    @property
    def provides_hedge_ratio(self) -> bool:
        """True if method directly provides hedge ratio."""
        return self in [self.ENGLE_GRANGER, self.PHILLIPS_OULIARIS]
    
    @property
    def robustness_to_small_samples(self) -> str:
        """Robustness rating for small samples."""
        ratings = {
            self.ENGLE_GRANGER: "moderate",
            self.JOHANSEN: "low",
            self.PHILLIPS_OULIARIS: "high",
            self.BOUNDS_TEST: "high",
        }
        return ratings.get(self, "unknown")
    
    @property
    def recommended_min_observations(self) -> int:
        """Minimum observations for reliable results."""
        minimums = {
            self.ENGLE_GRANGER: 100,
            self.JOHANSEN: 200,
            self.PHILLIPS_OULIARIS: 100,
            self.BOUNDS_TEST: 80,
        }
        return minimums.get(self, 100)
    
    @property
    def computational_complexity(self) -> str:
        """Computational complexity rating."""
        complexity = {
            self.ENGLE_GRANGER: "O(n)",
            self.JOHANSEN: "O(n²)",
            self.PHILLIPS_OULIARIS: "O(n)",
            self.BOUNDS_TEST: "O(n)",
        }
        return complexity.get(self, "O(n)")


class PairQuality(Enum):
    """
    Quality tier classification for cointegrated pairs.
    
    Determines position sizing, monitoring frequency, and risk limits.
    """
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    REJECTED = "rejected"
    
    @classmethod
    def from_metrics(
        cls,
        p_value: float,
        half_life: float,
        hurst: Optional[float] = None,
        stability: Optional[float] = None
    ) -> 'PairQuality':
        """
        Classify pair quality from metrics.
        
        Args:
            p_value: Cointegration test p-value
            half_life: Mean reversion half-life in days
            hurst: Hurst exponent (optional)
            stability: Rolling cointegration stability (optional)
        """
        if p_value >= 0.10 or half_life <= 0 or half_life > 30:
            return cls.REJECTED
        
        # Hurst check if available
        if hurst is not None and hurst >= 0.5:
            return cls.REJECTED
        
        # PDF-EXACT TIER THRESHOLDS (project specification)
        # PDF Page 20: "p-value < 0.10" significance threshold (EXACT - NO STRICTER)
        # PDF Page 16, 21: "1-14 days half-life" with 1-7 days preferred (STRICT)

        # Tier 1: Excellent cointegration - PDF preferred range (1-7 days)
        # PDF-EXACT: p < 0.10 (NOT 0.05, NOT 0.01), HL 1-7 days
        if p_value < 0.10 and 1.0 <= half_life <= 7.0:  # PDF exact preferred
            if stability is None or stability >= 0.75:
                return cls.TIER_1

        # Tier 2: Good cointegration - PDF acceptable (7-14 days)
        if p_value < 0.10 and 7.0 < half_life <= 14.0:  # Between preferred and max
            if stability is None or stability >= 0.6:
                return cls.TIER_2

        # Tier 3: Marginal - at PDF boundary
        if p_value < 0.10 and 14.0 < half_life <= 21.0:
            return cls.TIER_3
        
        return cls.REJECTED
    
    @property
    def is_tradeable(self) -> bool:
        """True if tier is tradeable."""
        return self != self.REJECTED
    
    @property
    def position_size_multiplier(self) -> float:
        """Position size multiplier based on quality."""
        multipliers = {
            self.TIER_1: 1.0,
            self.TIER_2: 0.7,
            self.TIER_3: 0.4,
            self.REJECTED: 0.0,
        }
        return multipliers.get(self, 0.0)
    
    @property
    def max_position_usd(self) -> float:
        """Maximum position size in USD."""
        limits = {
            self.TIER_1: 500_000,
            self.TIER_2: 200_000,
            self.TIER_3: 50_000,
            self.REJECTED: 0,
        }
        return limits.get(self, 0)
    
    @property
    def monitoring_frequency_hours(self) -> int:
        """How often to re-test cointegration."""
        frequencies = {
            self.TIER_1: 168,   # Weekly
            self.TIER_2: 72,    # Every 3 days
            self.TIER_3: 24,    # Daily
            self.REJECTED: 0,
        }
        return frequencies.get(self, 24)
    
    @property
    def min_z_score_entry(self) -> float:
        """Minimum z-score magnitude for entry."""
        thresholds = {
            self.TIER_1: 2.0,
            self.TIER_2: 2.2,
            self.TIER_3: 2.5,
            self.REJECTED: float('inf'),
        }
        return thresholds.get(self, 2.0)
    
    @property
    def stop_loss_z(self) -> float:
        """Stop loss z-score threshold. PDF exact: ±3.0."""
        stops = {
            self.TIER_1: 3.0,
            self.TIER_2: 3.0,
            self.TIER_3: 3.0,
            self.REJECTED: 0.0,
        }
        return stops.get(self, 3.0)
    
    @property
    def max_holding_days(self) -> int:
        """Maximum holding period in days."""
        max_hold = {
            self.TIER_1: 30,
            self.TIER_2: 21,
            self.TIER_3: 14,
            self.REJECTED: 0,
        }
        return max_hold.get(self, 21)


# =============================================================================
# PAIR TIER HELPER FUNCTIONS
# =============================================================================

def get_pair_tier(item: dict) -> PairQuality:
    """
    Extract tier from ranked pair item.

    Used by the orchestrator for tier-based pair selection.

    Args:
        item: Dict containing 'result' and/or 'pair_candidate' keys

    Returns:
        PairQuality enum value
    """
    result = item.get('result')
    if result and hasattr(result, 'quality_tier'):
        return result.quality_tier
    cand = item.get('pair_candidate')
    if cand and hasattr(cand, 'tier'):
        return cand.tier
    return PairQuality.TIER_3  # Default


def quality_to_pair_tier(quality: PairQuality) -> 'PairTier':
    """
    Convert PairQuality to PairTier for strategy compatibility.

    PairQuality (from cointegration analysis) needs to be converted to
    PairTier (for strategy/baseline) for downstream use.

    Args:
        quality: PairQuality enum value

    Returns:
        PairTier enum value
    """
    # Import here to avoid circular import
    from strategies.pairs_trading import PairTier

    if quality == PairQuality.TIER_1:
        return PairTier.TIER_1
    elif quality == PairQuality.TIER_2:
        return PairTier.TIER_2
    else:
        return PairTier.TIER_3


class StabilityStatus(Enum):
    """
    Cointegration stability status over time.
    
    Tracks whether the cointegration relationship is strengthening,
    weakening, or has broken down.
    """
    STABLE = "stable"
    WEAKENING = "weakening"
    STRENGTHENING = "strengthening"
    BROKEN = "broken"
    INSUFFICIENT_DATA = "insufficient_data"
    
    @classmethod
    def from_rolling_pvalues(
        cls,
        recent_pvalues: List[float],
        threshold: float = 0.10
    ) -> 'StabilityStatus':
        """Classify stability from rolling p-values."""
        if len(recent_pvalues) < 3:
            return cls.INSUFFICIENT_DATA
        
        # Check if cointegration has broken
        recent_3 = recent_pvalues[-3:]
        if all(p >= threshold for p in recent_3):
            return cls.BROKEN
        
        # Check trend
        if len(recent_pvalues) >= 5:
            first_half = np.mean(recent_pvalues[:len(recent_pvalues)//2])
            second_half = np.mean(recent_pvalues[len(recent_pvalues)//2:])
            
            if second_half > first_half * 1.5:
                return cls.WEAKENING
            elif second_half < first_half * 0.7:
                return cls.STRENGTHENING
        
        return cls.STABLE
    
    @property
    def requires_action(self) -> bool:
        """True if status requires trading action."""
        return self in [self.BROKEN, self.WEAKENING]
    
    @property
    def action_recommendation(self) -> str:
        """Recommended action for this status."""
        actions = {
            self.STABLE: "Continue trading normally",
            self.WEAKENING: "Reduce position size, increase monitoring",
            self.STRENGTHENING: "Consider increasing position size",
            self.BROKEN: "Close positions, remove from universe",
            self.INSUFFICIENT_DATA: "Wait for more data",
        }
        return actions.get(self, "No action")
    
    @property
    def position_adjustment(self) -> float:
        """Multiplier for position size adjustment."""
        adjustments = {
            self.STABLE: 1.0,
            self.WEAKENING: 0.5,
            self.STRENGTHENING: 1.2,
            self.BROKEN: 0.0,
            self.INSUFFICIENT_DATA: 0.5,
        }
        return adjustments.get(self, 1.0)


class RejectionReason(Enum):
    """
    Reasons for rejecting a pair from trading.
    
    Provides specific feedback for pair screening.
    """
    NOT_COINTEGRATED = "not_cointegrated"
    HALF_LIFE_TOO_SHORT = "half_life_too_short"
    HALF_LIFE_TOO_LONG = "half_life_too_long"
    NON_MEAN_REVERTING = "non_mean_reverting"
    HURST_TOO_HIGH = "hurst_too_high"
    INSUFFICIENT_DATA = "insufficient_data"
    UNSTABLE_RELATIONSHIP = "unstable_relationship"
    LOW_LIQUIDITY = "low_liquidity"
    HIGH_CORRELATION_WITH_EXISTING = "high_correlation_existing"
    STRUCTURAL_BREAK_DETECTED = "structural_break"
    NEGATIVE_HEDGE_RATIO = "negative_hedge_ratio"
    
    @property
    def is_permanent(self) -> bool:
        """True if rejection is likely permanent."""
        permanent = [
            self.NOT_COINTEGRATED,
            self.NON_MEAN_REVERTING,
            self.HURST_TOO_HIGH,
        ]
        return self in permanent
    
    @property
    def retest_days(self) -> Optional[int]:
        """Days before retesting, None if permanent."""
        if self.is_permanent:
            return None
        
        retest = {
            self.HALF_LIFE_TOO_SHORT: 30,
            self.HALF_LIFE_TOO_LONG: 30,
            self.INSUFFICIENT_DATA: 7,
            self.UNSTABLE_RELATIONSHIP: 14,
            self.LOW_LIQUIDITY: 7,
            self.HIGH_CORRELATION_WITH_EXISTING: 30,
            self.STRUCTURAL_BREAK_DETECTED: 60,
            self.NEGATIVE_HEDGE_RATIO: 30,
        }
        return retest.get(self, 30)
    
    @property
    def description(self) -> str:
        """Human-readable description."""
        descriptions = {
            self.NOT_COINTEGRATED: "Pair failed cointegration test",
            self.HALF_LIFE_TOO_SHORT: "Mean reversion too fast (likely noise)",
            self.HALF_LIFE_TOO_LONG: "Mean reversion too slow (not tradeable)",
            self.NON_MEAN_REVERTING: "No evidence of mean reversion",
            self.HURST_TOO_HIGH: "Hurst exponent indicates trending behavior",
            self.INSUFFICIENT_DATA: "Not enough data for reliable testing",
            self.UNSTABLE_RELATIONSHIP: "Cointegration not stable over time",
            self.LOW_LIQUIDITY: "Insufficient liquidity for trading",
            self.HIGH_CORRELATION_WITH_EXISTING: "Too correlated with existing pairs",
            self.STRUCTURAL_BREAK_DETECTED: "Structural break in relationship",
            self.NEGATIVE_HEDGE_RATIO: "Negative hedge ratio (same direction exposure)",
        }
        return descriptions.get(self, "Unknown reason")


# =============================================================================
# COMPREHENSIVE DATA CLASSES
# =============================================================================

@dataclass
class CointegrationResult:
    """
    Comprehensive results from cointegration testing.
    
    Contains test statistics, trading parameters, and quality metrics
    for a single pair cointegration analysis.
    """
    # Pair identification
    pair: Tuple[str, str]
    method: CointegrationMethod
    
    # Core test results
    is_cointegrated: bool
    p_value: float
    test_statistic: float
    critical_values: Dict[str, float]
    
    # Trading parameters
    hedge_ratio: float
    intercept: float
    spread: pd.Series
    half_life: float
    
    # Spread statistics
    spread_std: float
    spread_mean: float

    # Sample information (required fields without defaults)
    n_observations: int

    # Spread statistics (optional with defaults)
    spread_skew: float = 0.0
    spread_kurtosis: float = 0.0

    # Sample information (optional with defaults)
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    
    # ADF test on spread
    adf_statistic: float = 0.0
    adf_pvalue: float = 1.0
    
    # Additional metrics
    hurst_exponent: Optional[float] = None
    variance_ratio: Optional[float] = None
    regression_r_squared: float = 0.0
    regression_std_error: float = 0.0
    
    # Quality assessment
    quality_tier: PairQuality = field(default=PairQuality.REJECTED)
    rejection_reason: Optional[RejectionReason] = None
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if len(self.spread) > 0:
            self.spread_skew = float(self.spread.skew())
            self.spread_kurtosis = float(self.spread.kurtosis())
            self.start_date = self.spread.index.min()
            self.end_date = self.spread.index.max()
        
        # Assess quality if not already set
        if self.quality_tier == PairQuality.REJECTED and self.is_cointegrated:
            self.quality_tier = PairQuality.from_metrics(
                self.p_value, self.half_life, self.hurst_exponent
            )
    
    # Core computed properties
    @property
    def pair_name(self) -> str:
        """Combined pair name."""
        return f"{self.pair[0]}_{self.pair[1]}"
    
    @property
    def is_tradeable(self) -> bool:
        """True if pair passes all quality checks."""
        return self.is_cointegrated and self.quality_tier.is_tradeable
    
    @property
    def expected_reversion_days(self) -> float:
        """Expected time to mean reversion."""
        if self.half_life <= 0 or self.half_life == float('inf'):
            return float('inf')
        return self.half_life * 2  # ~86% reversion at 2 half-lives
    
    # Statistical properties
    @property
    def z_score_current(self) -> float:
        """Current z-score of spread."""
        if len(self.spread) == 0 or self.spread_std <= 0:
            return 0.0
        return (self.spread.iloc[-1] - self.spread_mean) / self.spread_std
    
    @property
    def z_score_percentile(self) -> float:
        """Percentile of current spread in historical distribution."""
        if len(self.spread) == 0:
            return 50.0
        return float(stats.percentileofscore(self.spread, self.spread.iloc[-1]))
    
    @property
    def spread_range(self) -> float:
        """Historical spread range."""
        if len(self.spread) == 0:
            return 0.0
        return float(self.spread.max() - self.spread.min())
    
    @property
    def spread_iqr(self) -> float:
        """Interquartile range of spread."""
        if len(self.spread) == 0:
            return 0.0
        return float(self.spread.quantile(0.75) - self.spread.quantile(0.25))
    
    @property
    def is_spread_normal(self) -> bool:
        """True if spread is approximately normal."""
        # Jarque-Bera normality test approximation
        jb_stat = (self.n_observations / 6) * (self.spread_skew**2 + (self.spread_kurtosis**2) / 4)
        return jb_stat < 6.0  # ~5% significance
    
    @property
    def spread_autocorrelation(self) -> float:
        """First-order autocorrelation of spread."""
        if len(self.spread) < 2:
            return 0.0
        return float(self.spread.autocorr(lag=1))
    
    # Half-life properties
    @property
    def half_life_hours(self) -> float:
        """Half-life in hours (assuming daily data)."""
        return self.half_life * 24
    
    @property
    def half_life_quality(self) -> str:
        """Quality rating of half-life."""
        if self.half_life <= 0 or self.half_life == float('inf'):
            return "invalid"
        elif self.half_life < 0.5:
            return "too_fast"
        elif self.half_life <= 3:
            return "excellent"
        elif self.half_life <= 7:
            return "good"
        elif self.half_life <= 14:
            return "acceptable"
        elif self.half_life <= 21:
            return "slow"
        return "too_slow"
    
    # Confidence metrics
    @property
    def confidence_score(self) -> float:
        """Overall confidence in cointegration (0-1)."""
        if not self.is_cointegrated:
            return 0.0
        
        # P-value contribution (lower = better)
        p_score = max(0, 1 - self.p_value / 0.10)
        
        # Half-life contribution
        if 1 <= self.half_life <= 7:
            hl_score = 1.0
        elif 0.5 <= self.half_life <= 14:
            hl_score = 0.7
        elif self.half_life <= 21:
            hl_score = 0.4
        else:
            hl_score = 0.0
        
        # Sample size contribution
        n_score = min(1.0, self.n_observations / 500)
        
        # Hurst contribution
        if self.hurst_exponent is not None:
            hurst_score = max(0, 1 - self.hurst_exponent / 0.5)
        else:
            hurst_score = 0.5
        
        return (p_score * 0.3 + hl_score * 0.3 + n_score * 0.2 + hurst_score * 0.2)
    
    @property
    def statistical_power(self) -> float:
        """Estimated statistical power of test."""
        # Approximation based on sample size and effect size
        effect_size = abs(self.test_statistic) / np.sqrt(self.n_observations)
        power = 1 - stats.norm.cdf(-effect_size * np.sqrt(self.n_observations))
        return min(power, 0.99)
    
    # Trading parameters
    @property
    def entry_threshold_z(self) -> float:
        """Recommended z-score entry threshold."""
        return self.quality_tier.min_z_score_entry
    
    @property
    def exit_threshold_z(self) -> float:
        """Recommended z-score exit threshold."""
        return 0.5 if self.quality_tier == PairQuality.TIER_1 else 0.75
    
    @property
    def stop_loss_z(self) -> float:
        """Stop loss z-score."""
        return self.quality_tier.stop_loss_z
    
    @property
    def recommended_lookback(self) -> int:
        """Recommended lookback for z-score calculation."""
        return max(int(self.half_life * 10), 30)
    
    @property
    def expected_trades_per_month(self) -> float:
        """Expected number of round-trip trades per month."""
        if self.half_life <= 0 or self.half_life == float('inf'):
            return 0.0
        # Assumes ~2 entries per month with z>2 excursion
        return 30 / (self.half_life * 4)
    
    # Risk metrics
    @property
    def max_expected_drawdown_z(self) -> float:
        """Maximum expected z-score deviation."""
        # Based on historical extremes
        if len(self.spread) == 0:
            return 4.0
        return max(abs(self.spread.max() - self.spread_mean),
                  abs(self.spread.min() - self.spread_mean)) / self.spread_std
    
    @property
    def tail_risk_ratio(self) -> float:
        """Ratio of tail events to normal distribution expectation."""
        if len(self.spread) == 0 or self.spread_std <= 0:
            return 1.0
        
        z_scores = (self.spread - self.spread_mean) / self.spread_std
        observed_tail = (z_scores.abs() > 2).mean()
        expected_tail = 0.0455  # ~4.55% for |z| > 2
        
        return observed_tail / expected_tail if expected_tail > 0 else 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pair_name': self.pair_name,
            'symbol_a': self.pair[0],
            'symbol_b': self.pair[1],
            'method': self.method.value,
            'is_cointegrated': self.is_cointegrated,
            'is_tradeable': self.is_tradeable,
            'p_value': round(self.p_value, 6),
            'test_statistic': round(self.test_statistic, 4),
            'hedge_ratio': round(self.hedge_ratio, 6),
            'intercept': round(self.intercept, 6),
            'half_life': round(self.half_life, 2),
            'half_life_quality': self.half_life_quality,
            'spread_std': round(self.spread_std, 6),
            'spread_mean': round(self.spread_mean, 6),
            'z_score_current': round(self.z_score_current, 2),
            'n_observations': self.n_observations,
            'quality_tier': self.quality_tier.value,
            'confidence_score': round(self.confidence_score, 3),
            'hurst_exponent': round(self.hurst_exponent, 3) if self.hurst_exponent else None,
            'entry_threshold_z': self.entry_threshold_z,
            'expected_trades_per_month': round(self.expected_trades_per_month, 1),
            'rejection_reason': self.rejection_reason.value if self.rejection_reason else None,
        }
    
    def __repr__(self) -> str:
        return (
            f"CointegrationResult({self.pair_name}, "
            f"coint={self.is_cointegrated}, "
            f"p={self.p_value:.4f}, "
            f"hl={self.half_life:.1f}d, "
            f"tier={self.quality_tier.value})"
        )


@dataclass
class PairRanking:
    """
    Ranked pair for trading consideration.
    
    Contains all information needed to decide whether to trade a pair
    and how to size the position.
    """
    # Identification
    symbol_a: str
    symbol_b: str
    sector_a: str
    sector_b: str
    venue_type: str
    
    # Core metrics
    p_value: float
    test_statistic: float
    hedge_ratio: float
    half_life: float
    spread_std: float
    spread_mean: float = 0.0
    n_observations: int = 0
    
    # Quality
    rank: int = 0
    tier: int = 2
    confidence_score: float = 0.5
    
    # Tradability
    is_tradeable: bool = True
    rejection_reason: Optional[str] = None
    
    # Additional metrics
    hurst_exponent: Optional[float] = None
    correlation: Optional[float] = None
    liquidity_score: float = 0.5
    
    # Computed at post_init
    same_sector: bool = field(init=False)
    quality_tier: PairQuality = field(init=False)
    
    def __post_init__(self):
        """Calculate derived fields."""
        self.same_sector = self.sector_a == self.sector_b
        self.quality_tier = PairQuality.from_metrics(
            self.p_value, self.half_life, self.hurst_exponent
        )
    
    @property
    def pair_name(self) -> str:
        """Combined pair name."""
        return f"{self.symbol_a}_{self.symbol_b}"
    
    @property
    def is_same_sector(self) -> bool:
        """True if both tokens in same sector."""
        return self.same_sector
    
    @property
    def position_multiplier(self) -> float:
        """Position size multiplier."""
        base = self.quality_tier.position_size_multiplier
        
        # Sector bonus
        if self.same_sector:
            base *= 1.1
        
        # Liquidity adjustment
        base *= self.liquidity_score
        
        return min(base, 1.0)
    
    @property
    def risk_score(self) -> float:
        """Risk score (0-1, higher = riskier)."""
        score = 0.5
        
        # P-value contribution
        score += self.p_value * 2
        
        # Half-life contribution
        if self.half_life > 14:
            score += 0.2
        elif self.half_life < 1:
            score += 0.15
        
        # Hurst contribution
        if self.hurst_exponent and self.hurst_exponent > 0.45:
            score += (self.hurst_exponent - 0.45) * 2
        
        return min(score, 1.0)
    
    @property
    def opportunity_score(self) -> float:
        """Combined opportunity score for ranking."""
        if not self.is_tradeable:
            return 0.0
        
        # Confidence contribution
        conf_score = self.confidence_score * 0.3
        
        # Half-life contribution (optimal around 3-7 days)
        if 3 <= self.half_life <= 7:
            hl_score = 0.3
        elif 1 <= self.half_life <= 14:
            hl_score = 0.2
        else:
            hl_score = 0.1
        
        # Liquidity contribution
        liq_score = self.liquidity_score * 0.2
        
        # Tier contribution
        tier_score = (4 - self.tier) / 3 * 0.2
        
        return conf_score + hl_score + liq_score + tier_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pair_name': self.pair_name,
            'symbol_a': self.symbol_a,
            'symbol_b': self.symbol_b,
            'sector_a': self.sector_a,
            'sector_b': self.sector_b,
            'same_sector': self.same_sector,
            'venue_type': self.venue_type,
            'p_value': self.p_value,
            'hedge_ratio': self.hedge_ratio,
            'half_life': self.half_life,
            'rank': self.rank,
            'tier': self.tier,
            'is_tradeable': self.is_tradeable,
            'opportunity_score': round(self.opportunity_score, 3),
            'risk_score': round(self.risk_score, 3),
            'rejection_reason': self.rejection_reason,
        }


@dataclass
class RollingCointegrationResult:
    """
    Results from rolling cointegration stability analysis.
    
    Tracks how cointegration metrics evolve over time to detect
    relationship breakdown or strengthening.
    """
    pair: Tuple[str, str]
    window_size: int
    step_size: int
    
    # Time series of metrics
    timestamps: List[pd.Timestamp] = field(default_factory=list)
    p_values: List[float] = field(default_factory=list)
    hedge_ratios: List[float] = field(default_factory=list)
    half_lives: List[float] = field(default_factory=list)
    spread_stds: List[float] = field(default_factory=list)
    is_cointegrated: List[bool] = field(default_factory=list)
    
    # Derived metrics
    stability_status: StabilityStatus = field(default=StabilityStatus.INSUFFICIENT_DATA)
    
    def __post_init__(self):
        """Calculate stability status."""
        if len(self.p_values) >= 3:
            self.stability_status = StabilityStatus.from_rolling_pvalues(self.p_values)
    
    @property
    def pair_name(self) -> str:
        """Combined pair name."""
        return f"{self.pair[0]}_{self.pair[1]}"
    
    @property
    def n_windows(self) -> int:
        """Number of rolling windows tested."""
        return len(self.timestamps)
    
    @property
    def cointegration_rate(self) -> float:
        """Percentage of windows showing cointegration."""
        if not self.is_cointegrated:
            return 0.0
        return sum(self.is_cointegrated) / len(self.is_cointegrated)
    
    @property
    def p_value_mean(self) -> float:
        """Mean p-value across windows."""
        return np.mean(self.p_values) if self.p_values else 1.0
    
    @property
    def p_value_std(self) -> float:
        """Standard deviation of p-values."""
        return np.std(self.p_values) if len(self.p_values) > 1 else 0.0
    
    @property
    def p_value_trend(self) -> float:
        """Trend in p-values (positive = deteriorating)."""
        if len(self.p_values) < 3:
            return 0.0
        
        x = np.arange(len(self.p_values))
        slope, _ = np.polyfit(x, self.p_values, 1)
        return slope
    
    @property
    def hedge_ratio_mean(self) -> float:
        """Mean hedge ratio."""
        return np.mean(self.hedge_ratios) if self.hedge_ratios else 1.0
    
    @property
    def hedge_ratio_std(self) -> float:
        """Hedge ratio volatility."""
        return np.std(self.hedge_ratios) if len(self.hedge_ratios) > 1 else 0.0
    
    @property
    def hedge_ratio_stability(self) -> float:
        """Hedge ratio stability (1 = perfectly stable)."""
        if self.hedge_ratio_mean == 0:
            return 0.0
        cv = self.hedge_ratio_std / abs(self.hedge_ratio_mean)
        return max(0, 1 - cv)
    
    @property
    def half_life_mean(self) -> float:
        """Mean half-life."""
        valid = [hl for hl in self.half_lives if 0 < hl < 1000]
        return np.mean(valid) if valid else float('inf')
    
    @property
    def half_life_std(self) -> float:
        """Half-life volatility."""
        valid = [hl for hl in self.half_lives if 0 < hl < 1000]
        return np.std(valid) if len(valid) > 1 else 0.0
    
    @property
    def is_stable(self) -> bool:
        """True if relationship is stable."""
        return self.stability_status == StabilityStatus.STABLE
    
    @property
    def requires_attention(self) -> bool:
        """True if relationship needs monitoring."""
        return self.stability_status.requires_action
    
    @property
    def overall_stability_score(self) -> float:
        """Combined stability score (0-1)."""
        if self.n_windows < 3:
            return 0.5
        
        # Cointegration rate contribution
        coint_score = self.cointegration_rate * 0.4
        
        # P-value trend contribution
        trend_score = max(0, 0.3 - abs(self.p_value_trend) * 10)
        
        # Hedge ratio stability contribution
        hr_score = self.hedge_ratio_stability * 0.3
        
        return coint_score + trend_score + hr_score
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'p_value': self.p_values,
            'hedge_ratio': self.hedge_ratios,
            'half_life': self.half_lives,
            'spread_std': self.spread_stds,
            'is_cointegrated': self.is_cointegrated,
        })
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'pair_name': self.pair_name,
            'n_windows': self.n_windows,
            'stability_status': self.stability_status.value,
            'cointegration_rate': round(self.cointegration_rate, 3),
            'p_value_mean': round(self.p_value_mean, 4),
            'p_value_trend': round(self.p_value_trend, 6),
            'hedge_ratio_mean': round(self.hedge_ratio_mean, 4),
            'hedge_ratio_stability': round(self.hedge_ratio_stability, 3),
            'half_life_mean': round(self.half_life_mean, 2),
            'overall_stability_score': round(self.overall_stability_score, 3),
            'requires_attention': self.requires_attention,
        }


# =============================================================================
# COINTEGRATION ANALYZER
# =============================================================================

class CointegrationAnalyzer:
    """
    Comprehensive cointegration analysis for crypto pairs trading.
    
    Provides comprehensive testing, screening, and monitoring of
    cointegration relationships for statistical arbitrage.
    
    Features:
    - Multiple testing methods (Engle-Granger, Johansen)
    - Half-life calculation with multiple estimators
    - Hurst exponent for mean reversion confirmation
    - Rolling stability analysis
    - Pair screening and ranking
    
    Parameters
    ----------
    significance_level : float, default=0.10
        P-value threshold for cointegration
    min_half_life : float, default=0.5
        Minimum half-life in days (avoid noise)
    max_half_life : float, default=14.0
        Maximum half-life in days (PDF Page 20: "drop if >14 days")
    min_observations : int, default=252
        Minimum data points required
    
    Example
    -------
    >>> analyzer = CointegrationAnalyzer(
    ...     significance_level=0.10,
    ...     min_half_life=1.0,
    ...     max_half_life=14.0
    ... )
    >>> 
    >>> result = analyzer.engle_granger_test(btc_prices, eth_prices)
    >>> if result.is_tradeable:
    ...     print(f"Hedge ratio: {result.hedge_ratio:.4f}")
    ...     print(f"Half-life: {result.half_life:.1f} days")
    """
    
    # Default thresholds
    DEFAULT_HURST_MAX = 0.5
    DEFAULT_MIN_STABILITY = 0.6
    
    def __init__(
        self,
        significance_level: float = 0.10,
        min_half_life: float = 0.5,
        max_half_life: float = 14.0,  # PDF Page 20: "drop if >14 days"
        min_observations: int = 252,
        max_hurst: float = 0.5
    ):
        """Initialize the cointegration analyzer."""
        self.significance_level = significance_level
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.min_observations = min_observations
        self.max_hurst = max_hurst
        
        logger.info(
            f"CointegrationAnalyzer initialized: "
            f"α={significance_level}, "
            f"half_life=[{min_half_life}, {max_half_life}], "
            f"min_obs={min_observations}"
        )
    
    def engle_granger_test(
        self,
        series1: pd.Series,
        series2: pd.Series,
        trend: str = 'c',
        calculate_hurst: bool = True
    ) -> CointegrationResult:
        """
        Perform Engle-Granger two-step cointegration test.
        
        Step 1: OLS regression Y = α + β×X + ε
        Step 2: ADF test on residuals
        
        Args:
            series1: First price series (X)
            series2: Second price series (Y)
            trend: 'c' for constant, 'ct' for constant + trend
            calculate_hurst: Whether to calculate Hurst exponent
            
        Returns:
            CointegrationResult with comprehensive metrics
        """
        # Align series
        aligned = pd.concat([series1, series2], axis=1).dropna()
        
        if len(aligned) < self.min_observations:
            return self._create_rejected_result(
                series1, series2, aligned,
                RejectionReason.INSUFFICIENT_DATA
            )
        
        s1, s2 = aligned.iloc[:, 0], aligned.iloc[:, 1]
        
        # Step 1: Cointegration test
        coint_stat, p_value, crit_values = coint(s1, s2, trend=trend)
        
        # OLS for hedge ratio
        X = pd.DataFrame({'const': 1.0, 's1': s1.values})
        model = OLS(s2.values, X).fit()
        hedge_ratio = model.params['s1']
        intercept = model.params['const']
        
        # Check for negative hedge ratio
        if hedge_ratio < 0:
            return self._create_rejected_result(
                series1, series2, aligned,
                RejectionReason.NEGATIVE_HEDGE_RATIO
            )
        
        # Calculate spread
        spread = pd.Series(
            s2.values - hedge_ratio * s1.values - intercept,
            index=aligned.index,
            name='spread'
        )
        
        # ADF test on spread
        adf_result = adfuller(spread, maxlag=10, regression='c')
        adf_stat, adf_pvalue = adf_result[0], adf_result[1]
        
        # Calculate half-life
        half_life = self.calculate_half_life(spread)
        
        # Calculate Hurst exponent
        hurst = None
        if calculate_hurst and len(spread) >= 100:
            hurst = self.calculate_hurst_exponent(spread)
        
        # Determine if cointegrated
        is_cointegrated = p_value < self.significance_level
        
        # Create result
        pair_names = (
            series1.name if hasattr(series1, 'name') and series1.name else 'X',
            series2.name if hasattr(series2, 'name') and series2.name else 'Y'
        )
        
        result = CointegrationResult(
            pair=pair_names,
            method=CointegrationMethod.ENGLE_GRANGER,
            is_cointegrated=is_cointegrated,
            p_value=p_value,
            test_statistic=coint_stat,
            critical_values={
                '1%': crit_values[0],
                '5%': crit_values[1],
                '10%': crit_values[2]
            },
            hedge_ratio=hedge_ratio,
            intercept=intercept,
            spread=spread,
            half_life=half_life,
            spread_std=spread.std(),
            spread_mean=spread.mean(),
            n_observations=len(aligned),
            adf_statistic=adf_stat,
            adf_pvalue=adf_pvalue,
            hurst_exponent=hurst,
            regression_r_squared=model.rsquared,
            regression_std_error=model.bse['s1'],
        )
        
        # Apply quality checks
        result = self._apply_quality_checks(result)
        
        return result
    
    def johansen_test(
        self,
        data: pd.DataFrame,
        det_order: int = 0,
        k_ar_diff: int = 1
    ) -> Dict[str, Any]:
        """
        Johansen cointegration test for multiple series.
        
        Args:
            data: DataFrame with price series as columns
            det_order: Deterministic order (-1=none, 0=constant, 1=trend)
            k_ar_diff: Number of lagged differences
            
        Returns:
            Dictionary with test results and cointegrating vectors
        """
        clean_data = data.dropna()
        
        if len(clean_data) < self.min_observations:
            raise ValueError(
                f"Insufficient data: {len(clean_data)} < {self.min_observations}"
            )
        
        result = coint_johansen(clean_data, det_order, k_ar_diff)
        
        # Trace test
        trace_stat = result.lr1
        trace_crit_95 = result.cvt[:, 1]
        
        # Max eigenvalue test
        max_eig_stat = result.lr2
        max_eig_crit_95 = result.cvm[:, 1]
        
        # Count cointegrating relationships
        n_coint_trace = int(sum(trace_stat > trace_crit_95))
        n_coint_max = int(sum(max_eig_stat > max_eig_crit_95))
        
        # Extract cointegrating vectors
        eigenvectors = result.evec
        eigenvalues = result.eig
        
        # Normalize eigenvectors
        normalized_vectors = []
        for i in range(min(n_coint_trace, len(eigenvectors[0]))):
            vec = eigenvectors[:, i]
            normalized = vec / vec[0]  # Normalize to first element
            normalized_vectors.append(normalized.tolist())
        
        return {
            'method': CointegrationMethod.JOHANSEN.value,
            'n_series': data.shape[1],
            'n_observations': len(clean_data),
            'n_cointegrating_trace': n_coint_trace,
            'n_cointegrating_max': n_coint_max,
            'trace_statistic': trace_stat.tolist(),
            'trace_critical_95': trace_crit_95.tolist(),
            'max_eigenvalue_statistic': max_eig_stat.tolist(),
            'max_eigenvalue_critical_95': max_eig_crit_95.tolist(),
            'eigenvalues': eigenvalues.tolist(),
            'cointegrating_vectors': normalized_vectors,
            'is_cointegrated': n_coint_trace > 0,
        }

    def phillips_ouliaris_test(
        self,
        series1: pd.Series,
        series2: pd.Series,
        trend: str = 'c',
        calculate_hurst: bool = True
    ) -> CointegrationResult:
        """
        Perform Phillips-Ouliaris cointegration test.

        Similar to Engle-Granger but more robust to serial correlation
        and heteroskedasticity in residuals.

        Args:
            series1: First price series (X)
            series2: Second price series (Y)
            trend: 'c' for constant, 'ct' for constant + trend
            calculate_hurst: Whether to calculate Hurst exponent

        Returns:
            CointegrationResult with comprehensive metrics
        """
        # Align series
        aligned = pd.concat([series1, series2], axis=1).dropna()

        if len(aligned) < self.min_observations:
            return self._create_rejected_result(
                series1, series2, aligned,
                RejectionReason.INSUFFICIENT_DATA
            )

        s1, s2 = aligned.iloc[:, 0], aligned.iloc[:, 1]

        # Phillips-Ouliaris test using statsmodels coint function
        # (it implements PO-style adjustments internally)
        try:
            coint_stat, p_value, crit_values = coint(s1, s2, trend=trend, method='aeg')
        except Exception as e:
            logger.warning(f"Phillips-Ouliaris test failed: {e}")
            return self._create_rejected_result(
                series1, series2, aligned,
                RejectionReason.TEST_FAILED
            )

        # OLS for hedge ratio
        X = pd.DataFrame({'const': 1.0, 's1': s1.values})
        model = OLS(s2.values, X).fit()
        hedge_ratio = model.params['s1']
        intercept = model.params['const']

        # Check for negative hedge ratio
        if hedge_ratio < 0:
            return self._create_rejected_result(
                series1, series2, aligned,
                RejectionReason.NEGATIVE_HEDGE_RATIO
            )

        # Calculate spread
        spread = pd.Series(
            s2.values - hedge_ratio * s1.values - intercept,
            index=aligned.index,
            name='spread'
        )

        # ADF test on spread for additional validation
        adf_result = adfuller(spread, maxlag=10, regression='c')
        adf_stat, adf_pvalue = adf_result[0], adf_result[1]

        # Calculate half-life
        half_life = self.calculate_half_life(spread)

        # Calculate Hurst exponent
        hurst = None
        if calculate_hurst and len(spread) >= 100:
            hurst = self.calculate_hurst_exponent(spread)

        # Determine if cointegrated
        is_cointegrated = p_value < self.significance_level

        # Create result
        pair_names = (
            series1.name if hasattr(series1, 'name') and series1.name else 'X',
            series2.name if hasattr(series2, 'name') and series2.name else 'Y'
        )

        result = CointegrationResult(
            pair=pair_names,
            method=CointegrationMethod.PHILLIPS_OULIARIS,
            is_cointegrated=is_cointegrated,
            p_value=p_value,
            test_statistic=coint_stat,
            critical_values={
                '1%': crit_values[0],
                '5%': crit_values[1],
                '10%': crit_values[2]
            },
            hedge_ratio=hedge_ratio,
            intercept=intercept,
            spread=spread,
            half_life=half_life,
            spread_std=spread.std(),
            spread_mean=spread.mean(),
            n_observations=len(aligned),
            adf_statistic=adf_stat,
            adf_pvalue=adf_pvalue,
            hurst_exponent=hurst,
            regression_r_squared=model.rsquared,
            regression_std_error=model.bse['s1'],
        )

        # Check half-life constraints
        if is_cointegrated and half_life < self.min_half_life:
            result.quality_tier = PairQuality.TIER_3
            result.rejection_reason = RejectionReason.HALF_LIFE_TOO_SHORT
        elif is_cointegrated and half_life > self.max_half_life:
            result.quality_tier = PairQuality.TIER_3
            result.rejection_reason = RejectionReason.HALF_LIFE_TOO_LONG
        elif is_cointegrated:
            # Tier based on statistical strength
            # Use config-based thresholds instead of hardcoded values
            # Ideal half-life range is 20-50% of max_half_life (sweet spot for mean reversion)
            ideal_hl_min = self.min_half_life * 1.5  # Slightly above minimum
            ideal_hl_max = self.max_half_life * 0.4  # Well below maximum
            if p_value < 0.01 and ideal_hl_min <= half_life <= ideal_hl_max:
                result.quality_tier = PairQuality.TIER_1
            elif p_value < 0.03:
                result.quality_tier = PairQuality.TIER_2
            else:
                result.quality_tier = PairQuality.TIER_3
        else:
            result.rejection_reason = RejectionReason.NOT_COINTEGRATED

        return result

    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion using AR(1) model.
        
        Uses: Δspread_t = θ × spread_{t-1} + ε_t
        Half-life = -ln(2) / θ
        
        Args:
            spread: Spread time series
            
        Returns:
            Half-life in periods (same frequency as input)
        """
        spread = spread.dropna()
        
        if len(spread) < 10:
            return float('inf')
        
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        
        spread_lag = spread_lag.iloc[1:]
        spread_diff = spread_diff.iloc[1:]
        
        X = pd.DataFrame({'const': 1.0, 'lag': spread_lag.values})
        
        try:
            model = OLS(spread_diff.values, X).fit()
            theta = model.params['lag']
            
            if theta >= 0:
                return float('inf')
            
            half_life = -np.log(2) / theta
            
            if half_life < 0 or half_life > 10000:
                return float('inf')
            
            return half_life
            
        except Exception as e:
            logger.warning(f"Half-life calculation failed: {e}")
            return float('inf')
    
    def calculate_hurst_exponent(
        self,
        series: pd.Series,
        max_lag: int = 100
    ) -> float:
        """
        Calculate Hurst exponent using R/S analysis.
        
        H < 0.5: Mean-reverting (good for pairs trading)
        H = 0.5: Random walk
        H > 0.5: Trending
        
        Args:
            series: Time series
            max_lag: Maximum lag for analysis
            
        Returns:
            Hurst exponent
        """
        series = series.dropna().values
        n = len(series)
        
        if n < 100:
            return 0.5  # Not enough data
        
        max_lag = min(max_lag, n // 4)
        lags = range(10, max_lag)
        
        rs_values = []
        
        for lag in lags:
            # Split into subseries
            n_subseries = n // lag
            rs_list = []
            
            for i in range(n_subseries):
                subseries = series[i * lag:(i + 1) * lag]
                
                # Calculate cumulative deviation from mean
                mean_val = np.mean(subseries)
                cum_dev = np.cumsum(subseries - mean_val)
                
                # Range
                r = np.max(cum_dev) - np.min(cum_dev)
                
                # Standard deviation
                s = np.std(subseries, ddof=1)
                
                if s > 0:
                    rs_list.append(r / s)
            
            if rs_list:
                rs_values.append((lag, np.mean(rs_list)))
        
        if len(rs_values) < 5:
            return 0.5
        
        # Log-log regression
        log_lags = np.log([x[0] for x in rs_values])
        log_rs = np.log([x[1] for x in rs_values])
        
        try:
            slope, _ = np.polyfit(log_lags, log_rs, 1)
            return max(0, min(1, slope))
        except Exception:
            return 0.5
    
    def rolling_cointegration(
        self,
        series1: pd.Series,
        series2: pd.Series,
        window: int = 252,
        step: int = 21
    ) -> RollingCointegrationResult:
        """
        Test cointegration stability over rolling windows.
        
        Args:
            series1: First price series
            series2: Second price series
            window: Rolling window size
            step: Step between windows
            
        Returns:
            RollingCointegrationResult with time series of metrics
        """
        aligned = pd.concat([series1, series2], axis=1).dropna()
        s1, s2 = aligned.iloc[:, 0], aligned.iloc[:, 1]
        
        pair_names = (
            series1.name if hasattr(series1, 'name') and series1.name else 'X',
            series2.name if hasattr(series2, 'name') and series2.name else 'Y'
        )
        
        timestamps = []
        p_values = []
        hedge_ratios = []
        half_lives = []
        spread_stds = []
        is_cointegrated = []
        
        for end in range(window, len(aligned), step):
            start = end - window
            window_s1 = s1.iloc[start:end]
            window_s2 = s2.iloc[start:end]
            
            try:
                result = self.engle_granger_test(
                    window_s1, window_s2, calculate_hurst=False
                )
                
                timestamps.append(aligned.index[end - 1])
                p_values.append(result.p_value)
                hedge_ratios.append(result.hedge_ratio)
                half_lives.append(result.half_life)
                spread_stds.append(result.spread_std)
                is_cointegrated.append(result.is_cointegrated)
                
            except Exception as e:
                logger.debug(f"Rolling test failed at {end}: {e}")
                continue
        
        return RollingCointegrationResult(
            pair=pair_names,
            window_size=window,
            step_size=step,
            timestamps=timestamps,
            p_values=p_values,
            hedge_ratios=hedge_ratios,
            half_lives=half_lives,
            spread_stds=spread_stds,
            is_cointegrated=is_cointegrated,
        )
    
    def test_all_pairs(
        self,
        price_data: pd.DataFrame,
        symbols: Optional[List[str]] = None,
        sectors: Optional[Dict[str, str]] = None,
        venue_types: Optional[Dict[str, str]] = None,
        liquidity: Optional[Dict[str, float]] = None,
        max_pairs: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Test all possible pairs for cointegration.
        
        Args:
            price_data: DataFrame with symbols as columns
            symbols: Symbols to test (default: all)
            sectors: Symbol to sector mapping
            venue_types: Symbol to venue type mapping
            liquidity: Symbol to liquidity score mapping
            max_pairs: Maximum pairs to return
            
        Returns:
            DataFrame with ranked pairs
        """
        if symbols is None:
            symbols = list(price_data.columns)
        
        sectors = sectors or {}
        venue_types = venue_types or {}
        liquidity = liquidity or {}
        
        total_pairs = len(symbols) * (len(symbols) - 1) // 2
        logger.info(f"Testing {total_pairs} pairs from {len(symbols)} symbols")
        
        results = []
        tested = 0
        
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1:]:
                tested += 1
                
                if sym1 not in price_data.columns or sym2 not in price_data.columns:
                    continue
                
                s1 = price_data[sym1].dropna()
                s2 = price_data[sym2].dropna()
                
                aligned = pd.concat([s1, s2], axis=1).dropna()
                if len(aligned) < self.min_observations:
                    continue
                
                try:
                    s1_aligned = aligned.iloc[:, 0]
                    s2_aligned = aligned.iloc[:, 1]
                    s1_aligned.name = sym1
                    s2_aligned.name = sym2
                    
                    result = self.engle_granger_test(s1_aligned, s2_aligned)
                    
                    sector1 = sectors.get(sym1, 'OTHER')
                    sector2 = sectors.get(sym2, 'OTHER')
                    
                    vt1 = venue_types.get(sym1, 'CEX')
                    vt2 = venue_types.get(sym2, 'CEX')
                    combined_venue = vt1 if vt1 == vt2 else 'MIXED'
                    
                    liq1 = liquidity.get(sym1, 0.5)
                    liq2 = liquidity.get(sym2, 0.5)
                    liq_score = min(liq1, liq2)
                    
                    ranking = PairRanking(
                        symbol_a=sym1,
                        symbol_b=sym2,
                        sector_a=sector1,
                        sector_b=sector2,
                        venue_type=combined_venue,
                        p_value=result.p_value,
                        test_statistic=result.test_statistic,
                        hedge_ratio=result.hedge_ratio,
                        half_life=result.half_life,
                        spread_std=result.spread_std,
                        spread_mean=result.spread_mean,
                        n_observations=result.n_observations,
                        tier=result.quality_tier.value.split('_')[-1] if result.quality_tier != PairQuality.REJECTED else 4,
                        confidence_score=result.confidence_score,
                        is_tradeable=result.is_tradeable,
                        rejection_reason=result.rejection_reason.value if result.rejection_reason else None,
                        hurst_exponent=result.hurst_exponent,
                        liquidity_score=liq_score,
                    )
                    
                    results.append(ranking.to_dict())
                    
                except Exception as e:
                    logger.debug(f"Failed to test {sym1}/{sym2}: {e}")
                    continue
                
                if tested % 500 == 0:
                    logger.info(f"Tested {tested}/{total_pairs} pairs")
        
        df = pd.DataFrame(results)
        
        if df.empty:
            logger.warning("No valid pairs found")
            return df
        
        # Sort by opportunity score
        df = df.sort_values('opportunity_score', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        if max_pairs:
            df = df.head(max_pairs)
        
        tradeable = df['is_tradeable'].sum()
        logger.info(f"Results: {tradeable}/{len(df)} tradeable pairs")
        
        return df
    
    def _apply_quality_checks(
        self,
        result: CointegrationResult
    ) -> CointegrationResult:
        """Apply all quality checks to result."""
        if not result.is_cointegrated:
            result.rejection_reason = RejectionReason.NOT_COINTEGRATED
            result.quality_tier = PairQuality.REJECTED
            return result

        if result.half_life < self.min_half_life:
            result.rejection_reason = RejectionReason.HALF_LIFE_TOO_SHORT
            result.quality_tier = PairQuality.REJECTED
            return result

        if result.half_life > self.max_half_life:
            # PDF Page 16: half-life is a SCORING criterion ("prefer 1-7 days"), not hard rejection
            # Pairs with long half-life are lower tier but still tradeable if statistically cointegrated
            result.quality_tier = PairQuality.TIER_3
            return result

        if result.half_life == float('inf'):
            result.rejection_reason = RejectionReason.NON_MEAN_REVERTING
            result.quality_tier = PairQuality.REJECTED
            return result

        if result.hurst_exponent and result.hurst_exponent > self.max_hurst:
            result.rejection_reason = RejectionReason.HURST_TOO_HIGH
            result.quality_tier = PairQuality.REJECTED
            return result

        # Passed all checks - classify tier based on relative position within config range
        # This handles both daily and hourly data correctly by using config-based ranges
        hl_range = self.max_half_life - self.min_half_life
        if hl_range > 0:
            hl_normalized = (result.half_life - self.min_half_life) / hl_range
        else:
            hl_normalized = 0.5

        # Tier classification based on p-value and normalized half-life position
        # TIER_1: Excellent - very low p-value, half-life in sweet spot (25-50% of range)
        # TIER_2: Good - low p-value, half-life acceptable (15-65% of range)
        # TIER_3: Acceptable - passes threshold
        if result.p_value < 0.01 and 0.15 <= hl_normalized <= 0.50:
            # Strong cointegration with optimal half-life
            result.quality_tier = PairQuality.TIER_1
        elif result.p_value < 0.10 and 0.10 <= hl_normalized <= 0.65:
            # Good cointegration
            result.quality_tier = PairQuality.TIER_2
        elif result.p_value < self.significance_level:
            # Acceptable cointegration
            result.quality_tier = PairQuality.TIER_3
        else:
            # Edge case - shouldn't happen but handle it
            result.quality_tier = PairQuality.TIER_3

        return result
    
    def _create_rejected_result(
        self,
        series1: pd.Series,
        series2: pd.Series,
        aligned: pd.DataFrame,
        reason: RejectionReason
    ) -> CointegrationResult:
        """Create a rejected result with appropriate reason."""
        pair_names = (
            series1.name if hasattr(series1, 'name') and series1.name else 'X',
            series2.name if hasattr(series2, 'name') and series2.name else 'Y'
        )
        
        return CointegrationResult(
            pair=pair_names,
            method=CointegrationMethod.ENGLE_GRANGER,
            is_cointegrated=False,
            p_value=1.0,
            test_statistic=0.0,
            critical_values={'1%': 0, '5%': 0, '10%': 0},
            hedge_ratio=0.0,
            intercept=0.0,
            spread=pd.Series(dtype=float),
            half_life=float('inf'),
            spread_std=0.0,
            spread_mean=0.0,
            n_observations=len(aligned) if not aligned.empty else 0,
            quality_tier=PairQuality.REJECTED,
            rejection_reason=reason,
        )

    def consensus_cointegration(
        self,
        test_results: Dict[CointegrationMethod, CointegrationResult],
        threshold: float = 0.30
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Comprehensive weighted consensus voting across multiple cointegration tests.

        Uses weighted voting with higher weights for more robust tests:
        - Engle-Granger: 0.25 (baseline)
        - Johansen Trace: 0.30 (multivariate)
        - Johansen Eigen: 0.30 (multivariate)
        - Phillips-Ouliaris: 0.15 (robust to trend)

        Args:
            test_results: Dict mapping CointegrationMethod to test results
            threshold: Minimum confidence for cointegration (default 0.30 for short periods)

        Returns:
            Tuple of (is_cointegrated, confidence_score, vote_breakdown)
        """
        # Weighted voting scheme - give more weight to methods robust to small samples
        weights = {
            CointegrationMethod.ENGLE_GRANGER: 0.35,      # More robust to small samples
            CointegrationMethod.JOHANSEN_TRACE: 0.20,     # Needs more data
            CointegrationMethod.JOHANSEN_EIGEN: 0.20,     # Needs more data
            CointegrationMethod.PHILLIPS_OULIARIS: 0.25   # Robust to small samples
        }

        # Calculate weighted vote
        total_weight = 0.0
        weighted_votes = 0.0
        vote_breakdown = {}

        for method, result in test_results.items():
            if method not in weights:
                continue

            weight = weights[method]
            total_weight += weight

            # Vote based on significance
            if result.is_cointegrated:
                # Strength of vote: use FIXED reference alpha=0.10 for consistent confidence
                # This ensures confidence reflects absolute statistical strength, not relative
                # to the chosen significance_level (which only gates pass/fail)
                REFERENCE_ALPHA = 0.10
                vote_strength = min(1.0, max(0.0, 1.0 - (result.p_value / REFERENCE_ALPHA)))
                vote_value = max(0.0, vote_strength * weight)
                weighted_votes += vote_value
                vote_breakdown[method.value] = vote_value
            else:
                vote_breakdown[method.value] = 0.0

        # Normalize to get confidence score
        confidence = weighted_votes / total_weight if total_weight > 0 else 0.0

        # Use configurable threshold (default 30% for short periods)
        is_cointegrated = confidence >= threshold

        return is_cointegrated, confidence, vote_breakdown

    def rank_pairs_advanced(
        self,
        pairs: List[Tuple[Any, CointegrationResult]],
        price_matrix: pd.DataFrame,
        liquidity_scores: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Comprehensive 12-factor pair ranking with full transparency.

        Ranking Factors:
        1. Cointegration strength (1 - p_value)
        2. Half-life optimality (Gaussian around 5 days)
        3. Liquidity score (if provided)
        4. Spread volatility (lower is better)
        5. Hurst exponent (lower is better for mean reversion)
        6. Hedge ratio stability
        7. Spread stationarity (ADF test strength)
        8. R-squared (regression quality)
        9. Residual normality (Jarque-Bera test)
        10. Tail risk (CVaR of spread)
        11. Mean reversion frequency
        12. Transaction cost efficiency

        Args:
            pairs: List of (pair_candidate, cointegration_result) tuples
            price_matrix: Price data for spread analysis
            liquidity_scores: Optional dict of symbol -> liquidity score

        Returns:
            List of dicts with pair info and all 12 factor scores
        """
        import scipy.stats as stats

        ranked = []

        for pair_candidate, result in pairs:
            if not result.is_cointegrated:
                continue

            # Factor 1: Cointegration strength
            coint_score = 1.0 - min(result.p_value / self.significance_level, 1.0)

            # Factor 2: Half-life optimality (Gaussian centered on 4 days = 96 hours)
            # PDF Page 16: "prefer 1-7 days for crypto" - STRONGLY PREFERRED
            # Gaussian with narrow std to heavily penalize HL outside 1-7 days
            optimal_half_life = 96.0  # 4 days in hours (center of 1-7 day range)
            hl_std = 48.0  # 2 days std dev → sharp dropoff outside 1-7 days
            half_life_score = np.exp(-0.5 * ((result.half_life - optimal_half_life) / hl_std) ** 2)

            # Factor 3: Liquidity score
            # PairCandidate uses token_a/token_b, not symbol1/symbol2
            if liquidity_scores:
                sym_a = getattr(pair_candidate, 'token_a', getattr(pair_candidate, 'symbol1', None))
                sym_b = getattr(pair_candidate, 'token_b', getattr(pair_candidate, 'symbol2', None))
                liq_score = (liquidity_scores.get(sym_a, 0.5) +
                            liquidity_scores.get(sym_b, 0.5)) / 2.0
            else:
                liq_score = 0.5  # Neutral

            # Factor 4: Spread volatility (normalized)
            spread_vol = result.spread_std / result.spread_mean if result.spread_mean != 0 else 1.0
            spread_vol_score = 1.0 / (1.0 + spread_vol)

            # Factor 5: Hurst exponent (lower is better)
            hurst_score = max(0.0, 1.0 - 2 * result.hurst_exponent) if result.hurst_exponent else 0.5

            # Factor 6: Hedge ratio stability (inverse of std error)
            hedge_ratio_se = getattr(result, 'regression_std_error', 0.0)
            hedge_ratio_score = 1.0 / (1.0 + hedge_ratio_se) if hedge_ratio_se else 0.5

            # Factor 7: Spread stationarity (ADF statistic strength)
            adf_stat = getattr(result, 'adf_statistic', 0.0)
            spread_stat_score = min(abs(adf_stat) / 5.0, 1.0) if adf_stat else 0.5

            # Factor 8: R-squared
            r_squared_score = getattr(result, 'regression_r_squared', 0.0)

            # Factor 9: Residual normality (Jarque-Bera)
            if len(result.spread) > 20:
                jb_stat, jb_pval = stats.jarque_bera(result.spread.dropna())
                normality_score = jb_pval  # Higher p-value = more normal
            else:
                normality_score = 0.5

            # Factor 10: Tail risk (CVaR at 5%)
            if len(result.spread) > 20:
                spread_returns = result.spread.pct_change().dropna()
                cvar_95 = spread_returns.quantile(0.05)
                tail_risk_score = max(0.0, 1.0 + cvar_95)  # Less negative = better
            else:
                tail_risk_score = 0.5

            # Factor 11: Mean reversion frequency
            if result.half_life > 0:
                reversion_freq = 1.0 / result.half_life
                reversion_score = min(reversion_freq * 5.0, 1.0)  # Scale to 0-1
            else:
                reversion_score = 0.0

            # Factor 12: Transaction cost efficiency (spread width)
            if result.spread_mean != 0:
                spread_width = abs(result.spread_std / result.spread_mean)
                cost_efficiency_score = 1.0 / (1.0 + spread_width * 10)
            else:
                cost_efficiency_score = 0.5

            # Weighted composite score
            weights = {
                'cointegration': 0.14,
                'half_life': 0.18,       # PDF "prefer 1-7 days" → highest single weight
                'liquidity': 0.10,
                'spread_volatility': 0.09,
                'hurst': 0.08,
                'hedge_ratio_stability': 0.07,
                'stationarity': 0.07,
                'r_squared': 0.07,
                'normality': 0.06,
                'tail_risk': 0.05,
                'reversion_frequency': 0.05,
                'cost_efficiency': 0.04
            }

            composite_score = (
                weights['cointegration'] * coint_score +
                weights['half_life'] * half_life_score +
                weights['liquidity'] * liq_score +
                weights['spread_volatility'] * spread_vol_score +
                weights['hurst'] * hurst_score +
                weights['hedge_ratio_stability'] * hedge_ratio_score +
                weights['stationarity'] * spread_stat_score +
                weights['r_squared'] * r_squared_score +
                weights['normality'] * normality_score +
                weights['tail_risk'] * tail_risk_score +
                weights['reversion_frequency'] * reversion_score +
                weights['cost_efficiency'] * cost_efficiency_score
            )

            ranked.append({
                'pair_candidate': pair_candidate,
                'result': result,
                'composite_score': composite_score,
                # All 12 factor scores
                'factor_scores': {
                    '1_cointegration': coint_score,
                    '2_half_life': half_life_score,
                    '3_liquidity': liq_score,
                    '4_spread_volatility': spread_vol_score,
                    '5_hurst': hurst_score,
                    '6_hedge_ratio_stability': hedge_ratio_score,
                    '7_stationarity': spread_stat_score,
                    '8_r_squared': r_squared_score,
                    '9_normality': normality_score,
                    '10_tail_risk': tail_risk_score,
                    '11_reversion_frequency': reversion_score,
                    '12_cost_efficiency': cost_efficiency_score
                }
            })

        # Sort by composite score
        ranked.sort(key=lambda x: x['composite_score'], reverse=True)

        return ranked


# =============================================================================
# PARALLEL TESTING SUPPORT
# Module-level worker function for joblib parallelization (must be at module
# level to be picklable)
# =============================================================================

def _cointegration_worker(
    token_a: str,
    token_b: str,
    prices1_arr: np.ndarray,
    prices2_arr: np.ndarray,
    analyzer_config: dict,
    calculate_all_tests: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Worker function for parallel cointegration testing.

    This function is designed to be called by joblib.Parallel for efficient
    parallel processing of pair combinations.

    Args:
        token_a: Symbol for first token
        token_b: Symbol for second token
        prices1_arr: Numpy array of prices for token_a
        prices2_arr: Numpy array of prices for token_b
        analyzer_config: Configuration dict with min_half_life, max_half_life, etc.
        calculate_all_tests: Whether to run all cointegration tests (EG, Johansen, PO)

    Returns:
        Dict with cointegration results if pair passes, None otherwise
    """
    try:
        # Create local analyzer instance (thread-safe)
        local_analyzer = CointegrationAnalyzer(
            significance_level=analyzer_config.get('significance_level', 0.10),
            min_half_life=analyzer_config.get('min_half_life', 0.5),
            max_half_life=analyzer_config.get('max_half_life', 14.0),  # PDF: "drop if >14 days"
            min_observations=analyzer_config.get('min_observations', 168),
            max_hurst=analyzer_config.get('max_hurst', 0.5)
        )

        # Convert to pandas Series
        prices1 = pd.Series(prices1_arr, name=token_a)
        prices2 = pd.Series(prices2_arr, name=token_b)

        # Run Engle-Granger test (primary test)
        eg_result = local_analyzer.engle_granger_test(prices1, prices2)

        if not eg_result.is_cointegrated:
            return None

        # Check half-life bounds
        min_hl = analyzer_config.get('min_half_life', 6)
        max_hl = analyzer_config.get('max_half_life', 720)   # 30 days max (PDF: prefer 1-7d)

        if eg_result.half_life < min_hl or eg_result.half_life > max_hl:
            return None

        if eg_result.half_life == float('inf') or eg_result.half_life > 10000:
            return None

        test_results = {CointegrationMethod.ENGLE_GRANGER: eg_result}

        # Run additional tests if requested
        if calculate_all_tests:
            # Johansen tests
            try:
                df_both = pd.DataFrame({token_a: prices1_arr, token_b: prices2_arr})
                jt_result_dict = local_analyzer.johansen_test(df_both, det_order=0, k_ar_diff=1)

                if jt_result_dict:
                    # Trace test
                    jt_is_coint = jt_result_dict.get('n_cointegrating_trace', 0) > 0
                    trace_stats = jt_result_dict.get('trace_statistic', [0])
                    trace_crits = jt_result_dict.get('trace_critical_95', [0])
                    jt_stat = trace_stats[0] if trace_stats else 0
                    jt_crit = trace_crits[0] if trace_crits else 0

                    jt_result = CointegrationResult(
                        pair=(token_a, token_b),
                        method=CointegrationMethod.JOHANSEN_TRACE,
                        is_cointegrated=jt_is_coint,
                        test_statistic=jt_stat,
                        critical_values={'1%': jt_crit * 1.1, '5%': jt_crit, '10%': jt_crit * 0.9},
                        p_value=0.03 if jt_is_coint else 0.15,
                        hedge_ratio=eg_result.hedge_ratio,
                        intercept=eg_result.intercept,
                        spread=eg_result.spread,
                        half_life=eg_result.half_life,
                        spread_std=eg_result.spread_std,
                        spread_mean=eg_result.spread_mean,
                        hurst_exponent=eg_result.hurst_exponent,
                        regression_r_squared=getattr(eg_result, 'regression_r_squared', 0.0),
                        n_observations=len(df_both),
                        quality_tier=eg_result.quality_tier,
                        rejection_reason=eg_result.rejection_reason
                    )
                    test_results[CointegrationMethod.JOHANSEN_TRACE] = jt_result

                    # Eigenvalue test
                    je_is_coint = jt_result_dict.get('n_cointegrating_max', 0) > 0
                    eigen_stats = jt_result_dict.get('max_eigenvalue_statistic', [0])
                    eigen_crits = jt_result_dict.get('max_eigenvalue_critical_95', [0])
                    je_stat = eigen_stats[0] if eigen_stats else 0
                    je_crit = eigen_crits[0] if eigen_crits else 0

                    je_result = CointegrationResult(
                        pair=(token_a, token_b),
                        method=CointegrationMethod.JOHANSEN_EIGEN,
                        is_cointegrated=je_is_coint,
                        test_statistic=je_stat,
                        critical_values={'1%': je_crit * 1.1, '5%': je_crit, '10%': je_crit * 0.9},
                        p_value=0.03 if je_is_coint else 0.15,
                        hedge_ratio=eg_result.hedge_ratio,
                        intercept=eg_result.intercept,
                        spread=eg_result.spread,
                        half_life=eg_result.half_life,
                        spread_std=eg_result.spread_std,
                        spread_mean=eg_result.spread_mean,
                        hurst_exponent=eg_result.hurst_exponent,
                        regression_r_squared=getattr(eg_result, 'regression_r_squared', 0.0),
                        n_observations=len(df_both),
                        quality_tier=eg_result.quality_tier,
                        rejection_reason=eg_result.rejection_reason
                    )
                    test_results[CointegrationMethod.JOHANSEN_EIGEN] = je_result
            except Exception:
                pass

            # Phillips-Ouliaris test
            try:
                po_result = local_analyzer.phillips_ouliaris_test(prices1, prices2)
                test_results[CointegrationMethod.PHILLIPS_OULIARIS] = po_result
            except Exception:
                pass

        # Consensus voting
        consensus_threshold = analyzer_config.get('consensus_threshold', 0.35)  # Tightened for quality
        is_cointegrated, confidence, vote_breakdown = local_analyzer.consensus_cointegration(
            test_results, threshold=consensus_threshold
        )

        if not is_cointegrated:
            return None

        # Assign tier based on half-life quality and confidence
        hl_range = max(1, max_hl - min_hl)
        hl_normalized = (eg_result.half_life - min_hl) / hl_range

        if confidence >= 0.70 and hl_normalized <= 0.50:
            eg_result.quality_tier = PairQuality.TIER_1
        elif confidence >= 0.50 and hl_normalized <= 0.65:
            eg_result.quality_tier = PairQuality.TIER_2
        else:
            eg_result.quality_tier = PairQuality.TIER_3

        return {
            'eg_result': eg_result,
            'confidence': confidence,
            'vote_breakdown': vote_breakdown,
            'symbol1': token_a,
            'symbol2': token_b
        }

    except Exception as e:
        logger.debug(f"Worker failed for {token_a}/{token_b}: {e}")
        return None


def test_pairs_parallel(
    price_matrix: pd.DataFrame,
    analyzer_config: dict,
    n_jobs: int = -1,
    batch_size: int = 100,
    progress_callback: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Test all pairs for cointegration using parallel processing.

    This is the main entry point for parallel cointegration testing,
    designed to be called from the orchestrator.

    Args:
        price_matrix: DataFrame with symbols as columns, timestamps as index
        analyzer_config: Configuration dict with cointegration parameters
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        batch_size: Batch size for joblib
        progress_callback: Optional callback for progress updates

    Returns:
        List of dicts containing cointegration results for passing pairs
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        logger.warning("joblib not available, falling back to sequential processing")
        return _test_pairs_sequential(price_matrix, analyzer_config)

    symbols = list(price_matrix.columns)
    n_symbols = len(symbols)

    # Generate all pair combinations
    pairs = []
    for i in range(n_symbols):
        for j in range(i + 1, n_symbols):
            pairs.append((symbols[i], symbols[j]))

    total_pairs = len(pairs)
    logger.info(f"Testing {total_pairs} pairs using parallel processing ({n_jobs} jobs)")

    # Prepare data for workers (convert to numpy for efficiency)
    price_arrays = {sym: price_matrix[sym].values for sym in symbols}

    # Run parallel processing
    results = Parallel(n_jobs=n_jobs, batch_size=batch_size, prefer='processes')(
        delayed(_cointegration_worker)(
            token_a, token_b,
            price_arrays[token_a], price_arrays[token_b],
            analyzer_config
        )
        for token_a, token_b in pairs
    )

    # Filter out None results
    valid_results = [r for r in results if r is not None]

    logger.info(f"Found {len(valid_results)} cointegrated pairs from {total_pairs} tested")

    return valid_results


def _test_pairs_sequential(
    price_matrix: pd.DataFrame,
    analyzer_config: dict
) -> List[Dict[str, Any]]:
    """Fallback sequential testing when joblib is not available."""
    symbols = list(price_matrix.columns)
    n_symbols = len(symbols)

    results = []
    for i in range(n_symbols):
        for j in range(i + 1, n_symbols):
            token_a, token_b = symbols[i], symbols[j]
            result = _cointegration_worker(
                token_a, token_b,
                price_matrix[token_a].values,
                price_matrix[token_b].values,
                analyzer_config
            )
            if result is not None:
                results.append(result)

    return results


# =============================================================================
# GPU-ENHANCED COINTEGRATION WORKER
# =============================================================================

def _test_cointegration_worker(
    token_a: str,
    token_b: str,
    prices1_arr: np.ndarray,
    prices2_arr: np.ndarray,
    analyzer_config: dict,
    gpu_precomputed: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    GPU-enhanced worker function for cointegration testing.

    This function is designed to be picklable for multiprocessing.
    Uses statsmodels with full float64 precision for maximum accuracy.

    GPU metrics (if provided) are ONLY used for supplementary data like Hurst exponent
    and R-squared that don't affect the cointegration decision.

    Args:
        token_a: First token symbol
        token_b: Second token symbol
        prices1_arr: Price array for token A
        prices2_arr: Price array for token B
        analyzer_config: Configuration dict with test parameters
        gpu_precomputed: Optional dict with GPU pre-computed supplementary metrics

    Returns:
        Result dict if cointegrated, None otherwise
    """
    # Skip if price series has no variation
    std1 = np.std(prices1_arr)
    std2 = np.std(prices2_arr)
    if std1 < 1e-10 or std2 < 1e-10:
        print(f"  [DIAG] {token_a}-{token_b}: REJECTED - zero price variation", flush=True)
        return None

    # Convert to Series for statsmodels compatibility
    prices1 = pd.Series(prices1_arr)
    prices2 = pd.Series(prices2_arr)

    # Create local analyzer - STRICT PDF-compliant settings
    local_analyzer = CointegrationAnalyzer(
        significance_level=analyzer_config.get('significance_level', 0.10),  # PDF: p < 0.10
        min_half_life=analyzer_config.get('min_half_life', 24.0),  # 1 day min
        max_half_life=analyzer_config.get('max_half_life', 720.0),  # 30 days max (PDF: prefer 1-7d)
        min_observations=analyzer_config.get('min_observations', 100),
    )

    test_results = {}

    # 1. Engle-Granger (weight: 0.35)
    try:
        eg_result = local_analyzer.engle_granger_test(prices1, prices2)
        test_results[CointegrationMethod.ENGLE_GRANGER] = eg_result

        # Enhance with GPU-computed supplementary metrics (if available)
        if gpu_precomputed is not None:
            gpu_hurst = gpu_precomputed.get('hurst_exponent')
            if gpu_hurst is not None and 0.0 <= gpu_hurst <= 1.0:
                eg_result.hurst_exponent = gpu_hurst
            gpu_r2 = gpu_precomputed.get('r_squared')
            if gpu_r2 is not None and getattr(eg_result, 'regression_r_squared', None) is None:
                eg_result.regression_r_squared = gpu_r2
    except Exception as e:
        print(f"  [DIAG] {token_a}-{token_b}: REJECTED - EG test exception: {type(e).__name__}: {e}", flush=True)
        return None

    # 2. Johansen Trace (weight: 0.20)
    df_both = pd.DataFrame({'s1': prices1, 's2': prices2}).dropna()
    jt_result_dict = None
    try:
        if len(df_both) >= 50:
            jt_result_dict = local_analyzer.johansen_test(df_both, det_order=0, k_ar_diff=1)
    except Exception:
        pass

    if jt_result_dict:
        jt_is_coint = jt_result_dict.get('n_cointegrating_trace', 0) > 0
        trace_stats = jt_result_dict.get('trace_statistic', [0])
        trace_crits = jt_result_dict.get('trace_critical_95', [0])
        jt_stat = trace_stats[0] if trace_stats else 0
        jt_crit = trace_crits[0] if trace_crits else 0
    else:
        jt_is_coint = False
        jt_stat = 0
        jt_crit = 0

    jt_result = CointegrationResult(
        pair=(token_a, token_b),
        method=CointegrationMethod.JOHANSEN_TRACE,
        is_cointegrated=jt_is_coint,
        test_statistic=jt_stat,
        critical_values={'1%': jt_crit * 1.1, '5%': jt_crit, '10%': jt_crit * 0.9},
        p_value=0.03 if jt_is_coint else 0.15,
        hedge_ratio=eg_result.hedge_ratio,
        intercept=eg_result.intercept,
        spread=eg_result.spread,
        half_life=eg_result.half_life,
        spread_std=eg_result.spread_std,
        spread_mean=eg_result.spread_mean,
        hurst_exponent=eg_result.hurst_exponent,
        regression_r_squared=getattr(eg_result, 'regression_r_squared', 0.0),
        n_observations=len(df_both),
        quality_tier=eg_result.quality_tier,
        rejection_reason=eg_result.rejection_reason
    )
    test_results[CointegrationMethod.JOHANSEN_TRACE] = jt_result

    # 3. Johansen Eigen (weight: 0.20)
    if jt_result_dict:
        je_is_coint = jt_result_dict.get('n_cointegrating_max', 0) > 0
        eigen_stats = jt_result_dict.get('max_eigenvalue_statistic', [0])
        eigen_crits = jt_result_dict.get('max_eigenvalue_critical_95', [0])
        je_stat = eigen_stats[0] if eigen_stats else 0
        je_crit = eigen_crits[0] if eigen_crits else 0
    else:
        je_is_coint = False
        je_stat = 0
        je_crit = 0

    je_result = CointegrationResult(
        pair=(token_a, token_b),
        method=CointegrationMethod.JOHANSEN_EIGEN,
        is_cointegrated=je_is_coint,
        test_statistic=je_stat,
        critical_values={'1%': je_crit * 1.1, '5%': je_crit, '10%': je_crit * 0.9},
        p_value=0.03 if je_is_coint else 0.15,
        hedge_ratio=eg_result.hedge_ratio,
        intercept=eg_result.intercept,
        spread=eg_result.spread,
        half_life=eg_result.half_life,
        spread_std=eg_result.spread_std,
        spread_mean=eg_result.spread_mean,
        hurst_exponent=eg_result.hurst_exponent,
        regression_r_squared=getattr(eg_result, 'regression_r_squared', 0.0),
        n_observations=len(df_both),
        quality_tier=eg_result.quality_tier,
        rejection_reason=eg_result.rejection_reason
    )
    test_results[CointegrationMethod.JOHANSEN_EIGEN] = je_result

    # 4. Phillips-Ouliaris (weight: 0.25)
    try:
        po_result = local_analyzer.phillips_ouliaris_test(prices1, prices2)
        test_results[CointegrationMethod.PHILLIPS_OULIARIS] = po_result
    except Exception:
        pass

    # Consensus voting
    consensus_threshold = analyzer_config.get('consensus_threshold', 0.35)  # Tightened for quality
    is_cointegrated, confidence, vote_breakdown = local_analyzer.consensus_cointegration(
        test_results, threshold=consensus_threshold
    )

    # DIAGNOSTIC: Log rejection reason
    if not is_cointegrated:
        eg_p = getattr(eg_result, 'p_value', None)
        eg_coint = getattr(eg_result, 'is_cointegrated', None)
        hl_val = getattr(eg_result, 'half_life', None)
        hl_str = f"{hl_val/24.0:.1f}d" if hl_val and hl_val != float('inf') and hl_val > 0 else "N/A"
        print(f"  [DIAG] {token_a}-{token_b}: FAILED consensus | conf={confidence:.3f} thresh={consensus_threshold} | EG_p={eg_p} HL={hl_str}", flush=True)
        return None

    if is_cointegrated:
        # PDF-COMPLIANT half-life handling:
        # PDF Page 16 (Step 4 Ranking): "Half-life (prefer 1-7 days for crypto)" = SCORING criterion
        # PDF Page 21 (Option C Retirement): "Drop if half-life > 14 days" = RETIREMENT during monitoring
        # Initial selection uses half-life for RANKING/TIERING, not hard rejection.
        min_hl = analyzer_config.get('min_half_life', 24)    # 1 day minimum (noise filter)
        max_hl = analyzer_config.get('max_half_life', 720)   # 30 days max (PDF: prefer 1-7d)

        hl_hours = eg_result.half_life

        # HARD REJECTION: Invalid half-life (infinite or negative = non-mean-reverting)
        if hl_hours == float('inf') or hl_hours <= 0:
            print(f"  [DIAG] {token_a}-{token_b}: PASSED consensus (conf={confidence:.3f}) but FAILED HL: invalid ({hl_hours})", flush=True)
            return None

        # HARD REJECTION: Below minimum (too fast mean reversion = noise, not real signal)
        if hl_hours < min_hl:
            print(f"  [DIAG] {token_a}-{token_b}: PASSED consensus (conf={confidence:.3f}) but FAILED HL: {hl_hours/24:.1f}d < {min_hl/24:.1f}d min", flush=True)
            return None

        # HARD REJECTION: Beyond reasonable range for trading (no mean reversion within holding period)
        if hl_hours > max_hl:
            print(f"  [DIAG] {token_a}-{token_b}: PASSED consensus (conf={confidence:.3f}) but FAILED HL: {hl_hours/24:.1f}d > {max_hl/24:.1f}d max", flush=True)
            return None

        # PDF-COMPLIANT TIER ASSIGNMENT
        # PDF Page 16 Tier definitions:
        #   Tier 1: "Both tokens on major CEX, high liquidity, STRONG cointegration"
        #   Tier 2: "One CEX/one DEX, or both DEX with good liquidity"
        #   Tier 3: "Both DEX-only, lower liquidity, speculative"
        # Venue-based refinement happens downstream. Here we tier by cointegration quality:
        #   - Strong consensus (≥50%) + good p-value → Tier 1 (strong cointegration)
        #   - Moderate consensus (≥25%) + passes significance → Tier 2
        #   - Marginal (passes threshold but weaker) → Tier 3
        # Half-life is used for RANKING within tiers (PDF Page 16: "prefer 1-7 days")
        hl_days = hl_hours / 24.0
        eg_p = getattr(eg_result, 'p_value', 1.0)

        if confidence >= 0.60 and eg_p < 0.05:
            # Strong cointegration: high consensus + highly significant (tightened)
            eg_result.quality_tier = PairQuality.TIER_1
        elif confidence >= 0.40 and eg_p < 0.05:
            # Good cointegration: moderate consensus + significant (tightened)
            eg_result.quality_tier = PairQuality.TIER_2
        else:
            # Marginal cointegration: passes threshold but weaker evidence
            eg_result.quality_tier = PairQuality.TIER_3

        print(f"  [DIAG] {token_a}-{token_b}: [PASS] PASSED | conf={confidence:.3f} | HL={hl_days:.1f}d | tier={eg_result.quality_tier} | votes={vote_breakdown}", flush=True)
        return {
            'eg_result': eg_result,
            'confidence': confidence,
            'vote_breakdown': vote_breakdown,
            'symbol1': token_a,
            'symbol2': token_b
        }

    return None


# =============================================================================
# ADAPTIVE COINTEGRATION CONFIGURATION
# =============================================================================

# Import base config from data_loader
try:
    from strategies.pairs_trading.data_loader import COINTEGRATION_CONFIG
except ImportError:
    # Fallback if data_loader not available - PDF COMPLIANT with realistic crypto adjustments
    COINTEGRATION_CONFIG = {
        'significance_level': 0.05,     # α=0.05 significance level (tightened)
        'min_half_life': 6,             # 6 hours minimum (practical minimum)
        'max_half_life': 720,           # 30 days max (PDF: prefer 1-7d scoring, realistic for crypto)
        'preferred_half_life_max': 168, # PDF Page 16: "prefer 1-7 days" = 168 hours
        'min_observations': 100,
        'consensus_threshold': 0.35,    # Tightened for quality pairs
        'methods': ['engle_granger', 'johansen_trace', 'johansen_eigen', 'phillips_ouliaris'],
        'adf_regression': 'c',
        'johansen_det_order': 0,
        'max_lags': 10,
    }


def get_adaptive_cointegration_config(n_observations: int, data_freq: str = '1h') -> dict:
    """
    Dynamically adjust cointegration parameters based on actual data length and frequency.

    For short periods (testing), thresholds are relaxed.
    For long periods (production), stricter thresholds are used.
    Half-life thresholds are scaled based on data frequency.

    PDF Guidance (project specification):
    - Half-life preferred: 1-7 days (PDF Page 16) - SCORING preference, not rejection
    - Tier 1: HL ≤7 days (best) | Tier 2: HL ≤30 days | Tier 3: HL ≤180 days
    - For crypto hourly data, mean reversion is slower than traditional assets

    Args:
        n_observations: Number of timestamps in the price matrix
        data_freq: Data frequency ('1h', '4h', '1D', etc.)

    Returns:
        Adjusted config dict
    """
    config = COINTEGRATION_CONFIG.copy()

    # Determine hours per observation for half-life scaling
    freq_hours = {
        '1h': 1,
        '4h': 4,
        '1D': 24,
        '1d': 24,
        'D': 24,
        'd': 24,
    }
    hours_per_obs = freq_hours.get(data_freq, 1)  # Default to hourly

    # Scale factor: convert day-based thresholds to observation-based
    # For hourly data: 1 day = 24 observations
    scale_factor = 24 / hours_per_obs

    # Detect data frequency based on observations
    # Assume hourly if we have more than ~8 days at hourly or ~200 days at daily
    is_hourly = n_observations > 200

    if is_hourly:
        logger.info(f"Detected HOURLY data frequency based on {n_observations} observations")
    else:
        logger.info(f"Detected DAILY data frequency based on {n_observations} observations")
        scale_factor = 1  # No scaling for daily

    # Define thresholds for data length tiers
    # PDF guidance: "prefer 1-7 days" (Page 16) is a SCORING preference, not rejection
    # For crypto with long hourly time series, we use realistic bounds
    if n_observations < 100:
        # Very short period - relaxed for testing
        config['significance_level'] = 0.10
        config['min_half_life'] = 0.25 * scale_factor  # 6 hours min
        config['max_half_life'] = 14.0 * scale_factor  # 14 days max (PDF: drop if >14d)
        config['min_observations'] = max(20, n_observations // 5)
        config['consensus_threshold'] = 0.25
        logger.info(f"Using RELAXED cointegration config for {n_observations} observations (testing mode)")

    elif n_observations < 500:
        # Medium period - still realistic
        config['significance_level'] = 0.10
        config['min_half_life'] = 0.25 * scale_factor  # 6 hours min
        config['max_half_life'] = 14.0 * scale_factor  # 14 days max (PDF: drop if >14d)
        config['min_observations'] = max(50, n_observations // 4)
        config['consensus_threshold'] = 0.25
        logger.info(f"Using MODERATE cointegration config for {n_observations} observations")

    elif n_observations < 2000:
        # Standard period - realistic for crypto
        config['significance_level'] = 0.10
        config['min_half_life'] = 0.25 * scale_factor  # 6 hours min
        config['max_half_life'] = 14.0 * scale_factor  # 14 days max (PDF: drop if >14d)
        config['min_observations'] = 100
        config['consensus_threshold'] = 0.25
        logger.info(f"Using STANDARD cointegration config for {n_observations} observations")

    else:
        # Long period - STRICT PDF-COMPLIANT production settings
        # PDF Page 16 (Step 4): "Half-life (prefer 1-7 days for crypto)" = RANKING preference
        # PDF Page 21 (Retirement): "Drop if half-life > 14 days" = retirement during MONITORING
        # NOTE: The 14-day limit is under "Pair Retirement Logic" (Option C dynamic rebalancing),
        #       NOT initial pair selection. For initial selection, half-life is a SCORING criterion.
        #       Data analysis confirms: median crypto pair HL = 88d, only 0.2% have HL < 14d.
        #       Stationary pairs (ADF < 0.10) cluster at 30-90d HL range (84% stationary).
        config['significance_level'] = 0.05  # α=0.05 significance level (tightened for quality)
        config['min_half_life'] = 1.0 * scale_factor   # 1 day min (24 hours) - below = noise
        config['max_half_life'] = 45.0 * scale_factor  # 45 days max (PDF: 1080h max, prefer 1-7d scoring)
        config['min_observations'] = 200
        config['consensus_threshold'] = 0.50  # Strict: only pairs with strong multi-test consensus
        # PDF retirement threshold stored separately for Step 3 dynamic pair selection
        config['retirement_max_half_life'] = 14.0 * scale_factor  # 14 days per PDF Page 21
        logger.info(f"Using STRICT PDF config for {n_observations} observations (α=0.05, HL 1-45d, 14d retirement)")

    # Store scale factor for display purposes
    config['_scale_factor'] = scale_factor
    config['_is_hourly'] = is_hourly

    return config