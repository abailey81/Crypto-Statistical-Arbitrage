"""
Regime Detection for Crypto Pairs Trading
==========================================

Comprehensive Hidden Markov Model (HMM) based market regime detection
with DeFi-specific features for adaptive strategy adjustment.

Mathematical Framework
----------------------
Hidden Markov Model:

    P(O|λ) = Σ P(O|Q,λ) × P(Q|λ)
    
    Where:
        O = Observation sequence (features)
        Q = Hidden state sequence (regimes)
        λ = Model parameters (A, B, π)
        A = State transition matrix
        B = Emission probabilities
        π = Initial state distribution

Forward Algorithm:

    α_t(i) = P(O_1,...,O_t, q_t=i|λ)
    α_t(j) = [Σ α_{t-1}(i) × a_{ij}] × b_j(O_t)

Viterbi Algorithm (Most Likely State Sequence):

    δ_t(j) = max_i [δ_{t-1}(i) × a_{ij}] × b_j(O_t)
    q*_T = argmax_i [δ_T(i)]

Baum-Welch (Parameter Estimation):

    ξ_t(i,j) = P(q_t=i, q_{t+1}=j|O,λ)
    γ_t(i) = P(q_t=i|O,λ)

Regime Classification
---------------------
LOW_VOL (Risk-On):
    - BTC volatility < 40th percentile
    - Funding rates moderate
    - Full position sizes, all tiers allowed
    
MEDIUM_VOL (Normal):
    - BTC volatility 40th-80th percentile
    - Standard trading parameters
    - Tier 1 and 2 allowed
    
HIGH_VOL (Crisis):
    - BTC volatility > 80th percentile
    - Elevated funding rates
    - Defensive mode, Tier 1 only

TRENDING (Directional):
    - Strong momentum signal
    - Reduced mean-reversion confidence
    - Tighter stops, faster exits

Feature Engineering
-------------------
Traditional:
    - BTC/ETH returns (1d, 5d, 20d)
    - Realized volatility (rolling 20d)
    - Funding rate aggregate
    - Volume patterns

DeFi-Specific:
    - DEX TVL changes
    - Gas prices (activity indicator)
    - Stablecoin flows
    - Liquidation events
    - Protocol revenue

Author: Crypto StatArb Quantitative Research
Version: 2.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS WITH TRADING-SPECIFIC PROPERTIES
# =============================================================================

class MarketRegime(Enum):
    """
    Market regime classifications for pairs trading.
    
    Each regime has specific trading implications that affect
    position sizing, tier access, and risk parameters.
    """
    LOW_VOL = "low_vol"
    MEDIUM_VOL = "medium_vol"
    HIGH_VOL = "high_vol"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    CRISIS = "crisis"
    UNKNOWN = "unknown"
    
    @property
    def description(self) -> str:
        """Regime description."""
        descriptions = {
            self.LOW_VOL: "Low volatility risk-on environment",
            self.MEDIUM_VOL: "Normal market conditions",
            self.HIGH_VOL: "Elevated volatility, cautious trading",
            self.TRENDING_UP: "Strong upward momentum",
            self.TRENDING_DOWN: "Strong downward momentum",
            self.CRISIS: "Market crisis, defensive mode",
            self.UNKNOWN: "Insufficient data for classification",
        }
        return descriptions.get(self, "Unknown regime")
    
    @property
    def is_favorable_for_mean_reversion(self) -> bool:
        """True if regime is favorable for mean reversion."""
        return self in [self.LOW_VOL, self.MEDIUM_VOL]
    
    @property
    def is_trending(self) -> bool:
        """True if market is trending."""
        return self in [self.TRENDING_UP, self.TRENDING_DOWN]
    
    @property
    def is_crisis(self) -> bool:
        """True if market is in crisis."""
        return self in [self.HIGH_VOL, self.CRISIS]
    
    @property
    def position_multiplier(self) -> float:
        """Position size multiplier."""
        multipliers = {
            self.LOW_VOL: 1.0,
            self.MEDIUM_VOL: 0.8,
            self.HIGH_VOL: 0.4,
            self.TRENDING_UP: 0.6,
            self.TRENDING_DOWN: 0.5,
            self.CRISIS: 0.2,
            self.UNKNOWN: 0.5,
        }
        return multipliers.get(self, 0.5)
    
    @property
    def allowed_tiers(self) -> List[int]:
        """Allowed pair tiers in this regime."""
        tiers = {
            self.LOW_VOL: [1, 2, 3],
            self.MEDIUM_VOL: [1, 2],
            self.HIGH_VOL: [1],
            self.TRENDING_UP: [1, 2],
            self.TRENDING_DOWN: [1],
            self.CRISIS: [1],
            self.UNKNOWN: [1],
        }
        return tiers.get(self, [1])
    
    @property
    def max_dex_positions(self) -> int:
        """Maximum DEX positions allowed."""
        limits = {
            self.LOW_VOL: 5,
            self.MEDIUM_VOL: 3,
            self.HIGH_VOL: 1,
            self.TRENDING_UP: 2,
            self.TRENDING_DOWN: 1,
            self.CRISIS: 0,
            self.UNKNOWN: 1,
        }
        return limits.get(self, 1)
    
    @property
    def entry_z_threshold(self) -> float:
        """Z-score threshold for entry."""
        thresholds = {
            self.LOW_VOL: 1.8,
            self.MEDIUM_VOL: 2.0,
            self.HIGH_VOL: 2.5,
            self.TRENDING_UP: 2.2,
            self.TRENDING_DOWN: 2.5,
            self.CRISIS: 3.0,
            self.UNKNOWN: 2.5,
        }
        return thresholds.get(self, 2.0)
    
    @property
    def exit_z_threshold(self) -> float:
        """Z-score threshold for exit."""
        thresholds = {
            self.LOW_VOL: 0.0,
            self.MEDIUM_VOL: 0.25,
            self.HIGH_VOL: 0.5,
            self.TRENDING_UP: 0.25,
            self.TRENDING_DOWN: 0.5,
            self.CRISIS: 0.75,
            self.UNKNOWN: 0.5,
        }
        return thresholds.get(self, 0.25)
    
    @property
    def stop_z_threshold(self) -> float:
        """Stop loss z-score threshold."""
        thresholds = {
            self.LOW_VOL: 4.0,
            self.MEDIUM_VOL: 3.5,
            self.HIGH_VOL: 2.5,
            self.TRENDING_UP: 3.0,
            self.TRENDING_DOWN: 2.5,
            self.CRISIS: 2.0,
            self.UNKNOWN: 3.0,
        }
        return thresholds.get(self, 3.0)
    
    @property
    def max_holding_days(self) -> int:
        """Maximum holding period in days."""
        days = {
            self.LOW_VOL: 30,
            self.MEDIUM_VOL: 21,
            self.HIGH_VOL: 10,
            self.TRENDING_UP: 14,
            self.TRENDING_DOWN: 7,
            self.CRISIS: 5,
            self.UNKNOWN: 14,
        }
        return days.get(self, 14)
    
    @property
    def rebalance_frequency_hours(self) -> int:
        """How often to check for rebalancing."""
        hours = {
            self.LOW_VOL: 24,
            self.MEDIUM_VOL: 12,
            self.HIGH_VOL: 4,
            self.TRENDING_UP: 8,
            self.TRENDING_DOWN: 4,
            self.CRISIS: 1,
            self.UNKNOWN: 8,
        }
        return hours.get(self, 8)
    
    @property
    def recommended_leverage(self) -> float:
        """Recommended maximum leverage."""
        leverage = {
            self.LOW_VOL: 3.0,
            self.MEDIUM_VOL: 2.0,
            self.HIGH_VOL: 1.0,
            self.TRENDING_UP: 1.5,
            self.TRENDING_DOWN: 1.0,
            self.CRISIS: 1.0,
            self.UNKNOWN: 1.0,
        }
        return leverage.get(self, 1.0)
    
    @property
    def color_code(self) -> str:
        """Color code for visualization."""
        colors = {
            self.LOW_VOL: "#00FF00",      # Green
            self.MEDIUM_VOL: "#FFFF00",   # Yellow
            self.HIGH_VOL: "#FF8000",     # Orange
            self.TRENDING_UP: "#00FFFF",  # Cyan
            self.TRENDING_DOWN: "#FF00FF", # Magenta
            self.CRISIS: "#FF0000",       # Red
            self.UNKNOWN: "#808080",      # Gray
        }
        return colors.get(self, "#808080")


class RegimeTransition(Enum):
    """
    Types of regime transitions.
    
    Classifies how the market moved from one regime to another.
    """
    STABLE = "stable"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    CRISIS_ENTRY = "crisis_entry"
    CRISIS_EXIT = "crisis_exit"
    TREND_START = "trend_start"
    TREND_END = "trend_end"
    REVERSAL = "reversal"
    
    @classmethod
    def classify(
        cls,
        from_regime: MarketRegime,
        to_regime: MarketRegime
    ) -> 'RegimeTransition':
        """Classify transition between regimes."""
        if from_regime == to_regime:
            return cls.STABLE
        
        # Crisis transitions
        if to_regime in [MarketRegime.CRISIS, MarketRegime.HIGH_VOL]:
            if from_regime not in [MarketRegime.CRISIS, MarketRegime.HIGH_VOL]:
                return cls.CRISIS_ENTRY
        
        if from_regime in [MarketRegime.CRISIS, MarketRegime.HIGH_VOL]:
            if to_regime not in [MarketRegime.CRISIS, MarketRegime.HIGH_VOL]:
                return cls.CRISIS_EXIT
        
        # Trend transitions
        if to_regime.is_trending and not from_regime.is_trending:
            return cls.TREND_START
        
        if from_regime.is_trending and not to_regime.is_trending:
            return cls.TREND_END
        
        # Reversal
        if (from_regime == MarketRegime.TRENDING_UP and 
            to_regime == MarketRegime.TRENDING_DOWN):
            return cls.REVERSAL
        if (from_regime == MarketRegime.TRENDING_DOWN and 
            to_regime == MarketRegime.TRENDING_UP):
            return cls.REVERSAL
        
        # Risk on/off
        risk_order = {
            MarketRegime.LOW_VOL: 1,
            MarketRegime.MEDIUM_VOL: 2,
            MarketRegime.TRENDING_UP: 2,
            MarketRegime.TRENDING_DOWN: 3,
            MarketRegime.HIGH_VOL: 4,
            MarketRegime.CRISIS: 5,
            MarketRegime.UNKNOWN: 3,
        }
        
        from_risk = risk_order.get(from_regime, 3)
        to_risk = risk_order.get(to_regime, 3)
        
        if to_risk < from_risk:
            return cls.RISK_ON
        return cls.RISK_OFF
    
    @property
    def trading_implication(self) -> str:
        """Trading implication of transition."""
        implications = {
            self.STABLE: "Maintain current positions",
            self.RISK_ON: "Consider increasing exposure",
            self.RISK_OFF: "Reduce exposure, tighten stops",
            self.CRISIS_ENTRY: "Emergency risk reduction",
            self.CRISIS_EXIT: "Gradually rebuild positions",
            self.TREND_START: "Reduce mean-reversion trades",
            self.TREND_END: "Resume normal trading",
            self.REVERSAL: "Close directional bias, reassess",
        }
        return implications.get(self, "No action")
    
    @property
    def urgency(self) -> str:
        """Urgency level of required action."""
        urgency = {
            self.STABLE: "none",
            self.RISK_ON: "low",
            self.RISK_OFF: "medium",
            self.CRISIS_ENTRY: "immediate",
            self.CRISIS_EXIT: "low",
            self.TREND_START: "medium",
            self.TREND_END: "low",
            self.REVERSAL: "high",
        }
        return urgency.get(self, "low")
    
    @property
    def position_action(self) -> str:
        """Position action to take."""
        actions = {
            self.STABLE: "hold",
            self.RISK_ON: "scale_up",
            self.RISK_OFF: "scale_down",
            self.CRISIS_ENTRY: "close_risky",
            self.CRISIS_EXIT: "scale_up",
            self.TREND_START: "reduce",
            self.TREND_END: "resume",
            self.REVERSAL: "flatten",
        }
        return actions.get(self, "hold")


class FeatureCategory(Enum):
    """
    Categories of features used in regime detection.
    
    Organizes features by type for better feature engineering
    and importance analysis.
    """
    RETURNS = "returns"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    FUNDING = "funding"
    VOLUME = "volume"
    CORRELATION = "correlation"
    DEFI = "defi"
    ONCHAIN = "onchain"
    SENTIMENT = "sentiment"
    SEASONALITY = "seasonality"
    
    @property
    def is_required(self) -> bool:
        """True if feature category is required."""
        return self in [self.RETURNS, self.VOLATILITY]
    
    @property
    def typical_feature_count(self) -> int:
        """Typical number of features in this category."""
        counts = {
            self.RETURNS: 6,
            self.VOLATILITY: 8,
            self.MOMENTUM: 4,
            self.FUNDING: 6,
            self.VOLUME: 4,
            self.CORRELATION: 3,
            self.DEFI: 8,
            self.ONCHAIN: 6,
            self.SENTIMENT: 4,
            self.SEASONALITY: 6,
        }
        return counts.get(self, 4)
    
    @property
    def regime_predictive_power(self) -> str:
        """Relative predictive power for regime detection."""
        power = {
            self.RETURNS: "high",
            self.VOLATILITY: "very_high",
            self.MOMENTUM: "high",
            self.FUNDING: "medium",
            self.VOLUME: "medium",
            self.CORRELATION: "medium",
            self.DEFI: "medium",
            self.ONCHAIN: "low",
            self.SENTIMENT: "low",
            self.SEASONALITY: "low",
        }
        return power.get(self, "medium")


class DetectorType(Enum):
    """
    Types of regime detectors available.
    
    Different approaches for regime detection with varying
    complexity and data requirements.
    """
    HMM = "hmm"
    GMM = "gmm"
    RULE_BASED = "rule_based"
    ROLLING_PERCENTILE = "rolling_percentile"
    CHANGE_POINT = "change_point"
    ENSEMBLE = "ensemble"
    
    @property
    def description(self) -> str:
        """Detector description."""
        descriptions = {
            self.HMM: "Hidden Markov Model with Gaussian emissions",
            self.GMM: "Gaussian Mixture Model clustering",
            self.RULE_BASED: "Simple volatility percentile rules",
            self.ROLLING_PERCENTILE: "Rolling percentile classification",
            self.CHANGE_POINT: "Bayesian change point detection",
            self.ENSEMBLE: "Ensemble of multiple detectors",
        }
        return descriptions.get(self, "Unknown detector")
    
    @property
    def requires_training(self) -> bool:
        """True if detector requires training."""
        return self in [self.HMM, self.GMM, self.CHANGE_POINT, self.ENSEMBLE]
    
    @property
    def min_observations(self) -> int:
        """Minimum observations for reliable results."""
        minimums = {
            self.HMM: 500,
            self.GMM: 200,
            self.RULE_BASED: 60,
            self.ROLLING_PERCENTILE: 100,
            self.CHANGE_POINT: 300,
            self.ENSEMBLE: 500,
        }
        return minimums.get(self, 100)
    
    @property
    def computational_complexity(self) -> str:
        """Computational complexity."""
        complexity = {
            self.HMM: "O(T × N²)",
            self.GMM: "O(T × N × K)",
            self.RULE_BASED: "O(T)",
            self.ROLLING_PERCENTILE: "O(T × W)",
            self.CHANGE_POINT: "O(T²)",
            self.ENSEMBLE: "O(T × N²)",
        }
        return complexity.get(self, "O(T)")
    
    @property
    def handles_non_stationarity(self) -> bool:
        """True if detector handles non-stationary data well."""
        return self in [self.HMM, self.CHANGE_POINT, self.ROLLING_PERCENTILE]


# =============================================================================
# CONFIGURATION DATA CLASSES
# =============================================================================

@dataclass
class RegimeConfig:
    """
    Configuration for regime-specific trading parameters.
    
    Defines all trading parameters that change based on market regime.
    """
    regime: MarketRegime
    position_multiplier: float
    allowed_tiers: List[int]
    max_dex_positions: int
    entry_z_threshold: float
    exit_z_threshold: float
    stop_z_threshold: float
    max_holding_days: int
    max_leverage: float = 1.0
    rebalance_hours: int = 8
    
    @classmethod
    def from_regime(cls, regime: MarketRegime) -> 'RegimeConfig':
        """Create config from regime enum."""
        return cls(
            regime=regime,
            position_multiplier=regime.position_multiplier,
            allowed_tiers=regime.allowed_tiers,
            max_dex_positions=regime.max_dex_positions,
            entry_z_threshold=regime.entry_z_threshold,
            exit_z_threshold=regime.exit_z_threshold,
            stop_z_threshold=regime.stop_z_threshold,
            max_holding_days=regime.max_holding_days,
            max_leverage=regime.recommended_leverage,
            rebalance_hours=regime.rebalance_frequency_hours,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'regime': self.regime.value,
            'position_multiplier': self.position_multiplier,
            'allowed_tiers': self.allowed_tiers,
            'max_dex_positions': self.max_dex_positions,
            'entry_z_threshold': self.entry_z_threshold,
            'exit_z_threshold': self.exit_z_threshold,
            'stop_z_threshold': self.stop_z_threshold,
            'max_holding_days': self.max_holding_days,
            'max_leverage': self.max_leverage,
        }


@dataclass
class FeatureSpec:
    """
    Specification for a single feature.
    
    Contains all information needed to compute and validate a feature.
    """
    name: str
    category: FeatureCategory
    lookback: int
    description: str = ""
    required: bool = False
    normalize: bool = True
    winsorize_pct: float = 0.01
    
    @property
    def full_name(self) -> str:
        """Full feature name with lookback."""
        return f"{self.name}_{self.lookback}" if self.lookback > 0 else self.name


# =============================================================================
# REGIME STATE DATA CLASS
# =============================================================================

@dataclass
class RegimeState:
    """
    Comprehensive current regime state with probabilities and diagnostics.
    
    Contains all information about the current market regime and
    metadata for monitoring and debugging.
    """
    # Core state
    current_regime: MarketRegime
    regime_probabilities: Dict[MarketRegime, float]
    confidence: float
    timestamp: pd.Timestamp
    
    # Transition info
    previous_regime: Optional[MarketRegime] = None
    transition_type: Optional[RegimeTransition] = None
    regime_duration_periods: int = 0
    
    # Feature diagnostics
    features_used: Dict[str, float] = field(default_factory=dict)
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Model diagnostics
    log_likelihood: Optional[float] = None
    n_observations_used: int = 0
    detector_type: DetectorType = DetectorType.HMM
    
    def __post_init__(self):
        """Calculate derived fields."""
        if self.previous_regime and self.transition_type is None:
            self.transition_type = RegimeTransition.classify(
                self.previous_regime, self.current_regime
            )
    
    # Probability properties
    @property
    def max_probability(self) -> float:
        """Maximum regime probability."""
        if not self.regime_probabilities:
            return 0.0
        return max(self.regime_probabilities.values())
    
    @property
    def entropy(self) -> float:
        """Entropy of regime distribution (uncertainty measure)."""
        if not self.regime_probabilities:
            return 0.0
        
        probs = [p for p in self.regime_probabilities.values() if p > 0]
        if not probs:
            return 0.0
        
        return -sum(p * np.log(p) for p in probs)
    
    @property
    def is_confident(self) -> bool:
        """True if confidence exceeds threshold."""
        return self.confidence >= 0.6
    
    @property
    def is_uncertain(self) -> bool:
        """True if state is uncertain."""
        return self.confidence < 0.4
    
    @property
    def second_most_likely_regime(self) -> Optional[MarketRegime]:
        """Second most likely regime."""
        if len(self.regime_probabilities) < 2:
            return None
        
        sorted_regimes = sorted(
            self.regime_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_regimes[1][0]
    
    @property
    def probability_gap(self) -> float:
        """Gap between top 2 regime probabilities."""
        if len(self.regime_probabilities) < 2:
            return 1.0
        
        probs = sorted(self.regime_probabilities.values(), reverse=True)
        return probs[0] - probs[1]
    
    # Transition properties
    @property
    def just_transitioned(self) -> bool:
        """True if regime just changed."""
        return self.regime_duration_periods <= 1
    
    @property
    def is_stable(self) -> bool:
        """True if regime has been stable."""
        return self.regime_duration_periods >= 5
    
    @property
    def transition_urgency(self) -> str:
        """Urgency of transition response."""
        if self.transition_type:
            return self.transition_type.urgency
        return "none"
    
    # Config access
    @property
    def config(self) -> RegimeConfig:
        """Get config for current regime."""
        return RegimeConfig.from_regime(self.current_regime)
    
    @property
    def position_multiplier(self) -> float:
        """Position multiplier from current regime."""
        return self.current_regime.position_multiplier
    
    @property
    def entry_z(self) -> float:
        """Entry z-score threshold."""
        return self.current_regime.entry_z_threshold
    
    @property
    def exit_z(self) -> float:
        """Exit z-score threshold."""
        return self.current_regime.exit_z_threshold
    
    @property
    def stop_z(self) -> float:
        """Stop loss z-score."""
        return self.current_regime.stop_z_threshold
    
    # Diagnostic properties
    @property
    def top_features(self) -> List[Tuple[str, float]]:
        """Top contributing features."""
        if not self.feature_contributions:
            return []
        
        return sorted(
            self.feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
    
    @property
    def anomaly_score(self) -> float:
        """Anomaly score based on log-likelihood."""
        if self.log_likelihood is None:
            return 0.0
        
        # Lower log-likelihood = more anomalous
        # Normalize to 0-1 scale (approximate)
        return 1.0 / (1.0 + np.exp(self.log_likelihood / 100))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'current_regime': self.current_regime.value,
            'confidence': round(self.confidence, 3),
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'previous_regime': self.previous_regime.value if self.previous_regime else None,
            'transition_type': self.transition_type.value if self.transition_type else None,
            'regime_duration_periods': self.regime_duration_periods,
            'is_confident': self.is_confident,
            'is_stable': self.is_stable,
            'entropy': round(self.entropy, 4),
            'position_multiplier': self.position_multiplier,
            'entry_z': self.entry_z,
            'top_features': self.top_features,
        }
    
    def __repr__(self) -> str:
        return (
            f"RegimeState({self.current_regime.value}, "
            f"conf={self.confidence:.2f}, "
            f"dur={self.regime_duration_periods})"
        )


@dataclass
class RegimeHistory:
    """
    Historical record of regime states.
    
    Tracks regime evolution over time for analysis and persistence.
    """
    states: List[RegimeState] = field(default_factory=list)
    
    @property
    def n_observations(self) -> int:
        """Number of observations."""
        return len(self.states)
    
    @property
    def current_state(self) -> Optional[RegimeState]:
        """Most recent state."""
        return self.states[-1] if self.states else None
    
    @property
    def current_regime(self) -> MarketRegime:
        """Current regime."""
        if self.current_state:
            return self.current_state.current_regime
        return MarketRegime.UNKNOWN
    
    @property
    def regime_distribution(self) -> Dict[MarketRegime, float]:
        """Distribution of regimes over history."""
        if not self.states:
            return {}
        
        counts = {}
        for state in self.states:
            regime = state.current_regime
            counts[regime] = counts.get(regime, 0) + 1
        
        total = len(self.states)
        return {r: c / total for r, c in counts.items()}
    
    @property
    def n_transitions(self) -> int:
        """Number of regime transitions."""
        if len(self.states) < 2:
            return 0
        
        transitions = 0
        for i in range(1, len(self.states)):
            if self.states[i].current_regime != self.states[i-1].current_regime:
                transitions += 1
        
        return transitions
    
    @property
    def transition_rate(self) -> float:
        """Transition rate (transitions per period)."""
        if len(self.states) < 2:
            return 0.0
        return self.n_transitions / (len(self.states) - 1)
    
    @property
    def average_regime_duration(self) -> float:
        """Average duration of regime persistence."""
        if len(self.states) < 2 or self.n_transitions == 0:
            return len(self.states)
        return len(self.states) / (self.n_transitions + 1)
    
    @property
    def transition_matrix(self) -> pd.DataFrame:
        """Empirical transition probability matrix."""
        regimes = list(MarketRegime)
        n_regimes = len(regimes)
        
        counts = np.zeros((n_regimes, n_regimes))
        
        for i in range(1, len(self.states)):
            from_idx = regimes.index(self.states[i-1].current_regime)
            to_idx = regimes.index(self.states[i].current_regime)
            counts[from_idx, to_idx] += 1
        
        # Normalize rows
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        probs = counts / row_sums
        
        return pd.DataFrame(
            probs,
            index=[r.value for r in regimes],
            columns=[r.value for r in regimes]
        )
    
    def add_state(self, state: RegimeState):
        """Add new state to history."""
        if self.states:
            state.previous_regime = self.states[-1].current_regime
            
            if state.current_regime == self.states[-1].current_regime:
                state.regime_duration_periods = self.states[-1].regime_duration_periods + 1
            else:
                state.regime_duration_periods = 1
                state.transition_type = RegimeTransition.classify(
                    state.previous_regime, state.current_regime
                )
        
        self.states.append(state)
    
    def get_recent_states(self, n: int = 10) -> List[RegimeState]:
        """Get n most recent states."""
        return self.states[-n:]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        records = []
        for state in self.states:
            records.append({
                'timestamp': state.timestamp,
                'regime': state.current_regime.value,
                'confidence': state.confidence,
                'duration': state.regime_duration_periods,
                'transition': state.transition_type.value if state.transition_type else None,
                'position_mult': state.position_multiplier,
            })
        
        return pd.DataFrame(records)
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'n_observations': self.n_observations,
            'current_regime': self.current_regime.value,
            'regime_distribution': {
                r.value: round(p, 3)
                for r, p in self.regime_distribution.items()
            },
            'n_transitions': self.n_transitions,
            'transition_rate': round(self.transition_rate, 4),
            'average_regime_duration': round(self.average_regime_duration, 1),
        }


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class RegimeFeatureEngineer:
    """
    Feature engineering for regime detection.
    
    Computes traditional and DeFi-specific features for HMM input.
    """
    
    # Default lookback windows
    DEFAULT_RETURNS_LOOKBACK = 20
    DEFAULT_VOL_LOOKBACK = 20
    DEFAULT_FUNDING_LOOKBACK = 24  # 8 days at 8-hour intervals
    
    def __init__(
        self,
        returns_lookback: int = 20,
        volatility_lookback: int = 20,
        funding_lookback: int = 24,
        momentum_lookbacks: List[int] = None
    ):
        """Initialize feature engineer."""
        self.returns_lookback = returns_lookback
        self.volatility_lookback = volatility_lookback
        self.funding_lookback = funding_lookback
        self.momentum_lookbacks = momentum_lookbacks or [5, 10, 20]
        
        self.feature_specs: List[FeatureSpec] = []
        self.feature_names: List[str] = []
    
    def compute_returns_features(
        self,
        prices: pd.Series,
        prefix: str = "btc"
    ) -> pd.DataFrame:
        """Compute return-based features."""
        features = pd.DataFrame(index=prices.index)
        returns = prices.pct_change()
        
        features[f'{prefix}_return_1d'] = returns
        features[f'{prefix}_return_ma'] = returns.rolling(self.returns_lookback).mean()
        
        for lookback in self.momentum_lookbacks:
            features[f'{prefix}_momentum_{lookback}d'] = prices.pct_change(lookback)
        
        # Trend indicator
        features[f'{prefix}_trend'] = np.sign(
            prices.rolling(20).mean() - prices.rolling(50).mean()
        )
        
        return features
    
    def compute_volatility_features(
        self,
        prices: pd.Series,
        prefix: str = "btc"
    ) -> pd.DataFrame:
        """Compute volatility-based features."""
        features = pd.DataFrame(index=prices.index)
        returns = prices.pct_change()
        
        # Realized volatility (annualized)
        vol = returns.rolling(self.volatility_lookback).std() * np.sqrt(365)
        features[f'{prefix}_vol'] = vol
        
        # Volatility change
        features[f'{prefix}_vol_change'] = vol.pct_change()
        
        # Normalized volatility
        features[f'{prefix}_vol_zscore'] = (
            vol - vol.rolling(60).mean()
        ) / (vol.rolling(60).std() + 1e-10)
        
        # Volatility percentile
        features[f'{prefix}_vol_pctl'] = vol.rolling(252).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10)
            if len(x) > 0 else 0.5,
            raw=False
        )
        
        # High-low range volatility
        features[f'{prefix}_range'] = (
            prices.rolling(5).max() - prices.rolling(5).min()
        ) / prices
        
        return features
    
    def compute_funding_features(
        self,
        funding_rates: pd.DataFrame,
        prefix: str = "funding"
    ) -> pd.DataFrame:
        """Compute funding rate features."""
        features = pd.DataFrame()
        
        if funding_rates.empty:
            return features
        
        # Aggregate funding across venues/symbols
        if 'symbol' in funding_rates.columns:
            funding_rates = funding_rates.set_index(
                funding_rates.get('timestamp', funding_rates.index)
            )
            agg_funding = funding_rates.groupby(funding_rates.index)['funding_rate'].mean()
        else:
            agg_funding = funding_rates['funding_rate']
        
        features[f'{prefix}_rate'] = agg_funding
        features[f'{prefix}_annual'] = agg_funding * 1095  # Annualized
        features[f'{prefix}_ma'] = agg_funding.rolling(self.funding_lookback).mean()
        features[f'{prefix}_vol'] = agg_funding.rolling(self.funding_lookback).std()
        features[f'{prefix}_zscore'] = (
            agg_funding - agg_funding.rolling(72).mean()
        ) / (agg_funding.rolling(72).std() + 1e-10)
        features[f'{prefix}_extreme'] = (
            features[f'{prefix}_zscore'].abs() > 2
        ).astype(float)
        
        return features
    
    def compute_defi_features(
        self,
        tvl_data: Optional[pd.Series] = None,
        gas_prices: Optional[pd.Series] = None,
        liquidations: Optional[pd.Series] = None,
        stablecoin_mcap: Optional[pd.Series] = None,
        total_crypto_mcap: Optional[pd.Series] = None,
        dex_volume: Optional[pd.Series] = None,
        cex_volume: Optional[pd.Series] = None,
        bridge_volume: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Compute DeFi-specific features for regime detection.

        PDF Requirement (Section 2.3 Option A):
        - TVL changes
        - Stablecoin dominance
        - Gas prices (activity indicator)
        - DEX/CEX volume ratio
        - Bridge volumes
        - Liquidation events

        Args:
            tvl_data: Total Value Locked in DeFi protocols
            gas_prices: Ethereum gas prices (gwei)
            liquidations: DeFi liquidation volumes (USD)
            stablecoin_mcap: Total stablecoin market cap
            total_crypto_mcap: Total crypto market cap
            dex_volume: DEX trading volume
            cex_volume: CEX trading volume
            bridge_volume: Cross-chain bridge transfer volume

        Returns:
            DataFrame with DeFi-specific features
        """
        features = pd.DataFrame()

        # =====================================================================
        # TVL FEATURES - Tracks DeFi ecosystem health
        # =====================================================================
        if tvl_data is not None and not tvl_data.empty:
            features['tvl_change'] = tvl_data.pct_change()
            features['tvl_change_5d'] = tvl_data.pct_change(5)
            features['tvl_change_20d'] = tvl_data.pct_change(20)
            features['tvl_zscore'] = (
                tvl_data - tvl_data.rolling(30).mean()
            ) / (tvl_data.rolling(30).std() + 1e-10)
            # TVL momentum - positive indicates growth
            features['tvl_momentum'] = tvl_data.rolling(5).mean() / tvl_data.rolling(20).mean() - 1
            # TVL drawdown from rolling max
            features['tvl_drawdown'] = tvl_data / tvl_data.rolling(30).max() - 1

        # =====================================================================
        # GAS PRICE FEATURES - Network activity indicator
        # =====================================================================
        if gas_prices is not None and not gas_prices.empty:
            features['gas_log'] = np.log1p(gas_prices)
            features['gas_zscore'] = (
                gas_prices - gas_prices.rolling(30).mean()
            ) / (gas_prices.rolling(30).std() + 1e-10)
            features['gas_spike'] = (
                gas_prices > gas_prices.rolling(30).quantile(0.9)
            ).astype(float)
            # Gas percentile (0-1) - high gas = high activity/congestion
            features['gas_percentile'] = gas_prices.rolling(60).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10)
                if len(x) > 0 else 0.5,
                raw=False
            )
            # Gas trend - rising gas often precedes volatility
            features['gas_trend'] = gas_prices.rolling(5).mean() / gas_prices.rolling(20).mean() - 1

        # =====================================================================
        # LIQUIDATION FEATURES - Stress indicator
        # =====================================================================
        if liquidations is not None and not liquidations.empty:
            features['liq_log'] = np.log1p(liquidations)
            features['liq_zscore'] = (
                liquidations - liquidations.rolling(30).mean()
            ) / (liquidations.rolling(30).std() + 1e-10)
            # Liquidation spike detection (crisis indicator)
            features['liq_spike'] = (
                liquidations > liquidations.rolling(30).quantile(0.95)
            ).astype(float)
            # Cumulative liquidations over rolling window
            features['liq_cumulative_5d'] = liquidations.rolling(5).sum()
            features['liq_acceleration'] = liquidations.pct_change(3)

        # =====================================================================
        # STABLECOIN DOMINANCE - Risk appetite indicator (PDF requirement)
        # =====================================================================
        if stablecoin_mcap is not None and total_crypto_mcap is not None:
            if not stablecoin_mcap.empty and not total_crypto_mcap.empty:
                # Stablecoin dominance: high = risk-off, low = risk-on
                stable_dominance = stablecoin_mcap / (total_crypto_mcap + 1e-10)
                features['stable_dominance'] = stable_dominance
                features['stable_dominance_change'] = stable_dominance.pct_change(5)
                features['stable_dominance_zscore'] = (
                    stable_dominance - stable_dominance.rolling(30).mean()
                ) / (stable_dominance.rolling(30).std() + 1e-10)
                # Rising stablecoin dominance often signals risk-off
                features['stable_dominance_trend'] = (
                    stable_dominance.rolling(5).mean() / stable_dominance.rolling(20).mean() - 1
                )

        # =====================================================================
        # DEX/CEX VOLUME RATIO - Decentralization/retail activity (PDF requirement)
        # =====================================================================
        if dex_volume is not None and cex_volume is not None:
            if not dex_volume.empty and not cex_volume.empty:
                # DEX/CEX ratio: high = more decentralized activity
                dex_cex_ratio = dex_volume / (cex_volume + 1e-10)
                features['dex_cex_ratio'] = dex_cex_ratio
                features['dex_cex_ratio_log'] = np.log1p(dex_cex_ratio)
                features['dex_cex_ratio_change'] = dex_cex_ratio.pct_change(5)
                features['dex_cex_ratio_zscore'] = (
                    dex_cex_ratio - dex_cex_ratio.rolling(30).mean()
                ) / (dex_cex_ratio.rolling(30).std() + 1e-10)
                # DEX volume share
                total_volume = dex_volume + cex_volume
                features['dex_share'] = dex_volume / (total_volume + 1e-10)

        # =====================================================================
        # BRIDGE VOLUME FEATURES - Cross-chain activity (PDF requirement)
        # =====================================================================
        if bridge_volume is not None and not bridge_volume.empty:
            features['bridge_log'] = np.log1p(bridge_volume)
            features['bridge_zscore'] = (
                bridge_volume - bridge_volume.rolling(30).mean()
            ) / (bridge_volume.rolling(30).std() + 1e-10)
            features['bridge_change_5d'] = bridge_volume.pct_change(5)
            # Bridge activity spike (often precedes volatility or opportunities)
            features['bridge_spike'] = (
                bridge_volume > bridge_volume.rolling(30).quantile(0.9)
            ).astype(float)
            # Bridge momentum
            features['bridge_momentum'] = (
                bridge_volume.rolling(5).mean() / bridge_volume.rolling(20).mean() - 1
            )

        return features
    
    def prepare_features(
        self,
        btc_prices: pd.Series,
        eth_prices: Optional[pd.Series] = None,
        funding_rates: Optional[pd.DataFrame] = None,
        tvl_data: Optional[pd.Series] = None,
        gas_prices: Optional[pd.Series] = None,
        liquidations: Optional[pd.Series] = None,
        stablecoin_mcap: Optional[pd.Series] = None,
        total_crypto_mcap: Optional[pd.Series] = None,
        dex_volume: Optional[pd.Series] = None,
        cex_volume: Optional[pd.Series] = None,
        bridge_volume: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Prepare complete feature matrix for regime detection.

        PDF Requirement (Section 2.3 Option A):
        Comprehensive feature set including traditional market indicators
        and DeFi-specific features for crypto regime detection.

        Args:
            btc_prices: Bitcoin price series (required)
            eth_prices: Ethereum price series (optional)
            funding_rates: Perpetual funding rates (optional)
            tvl_data: Total Value Locked in DeFi (optional)
            gas_prices: Ethereum gas prices in gwei (optional)
            liquidations: DeFi liquidation volumes (optional)
            stablecoin_mcap: Stablecoin market cap for dominance (optional)
            total_crypto_mcap: Total crypto market cap (optional)
            dex_volume: DEX trading volume (optional)
            cex_volume: CEX trading volume (optional)
            bridge_volume: Cross-chain bridge volume (optional)

        Returns:
            DataFrame with all computed features, NaN rows dropped
        """
        all_features = []

        # =====================================================================
        # REQUIRED: BTC features (core market reference)
        # =====================================================================
        all_features.append(self.compute_returns_features(btc_prices, 'btc'))
        all_features.append(self.compute_volatility_features(btc_prices, 'btc'))

        # =====================================================================
        # OPTIONAL: ETH features (altcoin reference)
        # =====================================================================
        if eth_prices is not None and not eth_prices.empty:
            all_features.append(self.compute_returns_features(eth_prices, 'eth'))
            all_features.append(self.compute_volatility_features(eth_prices, 'eth'))

            # ETH/BTC ratio features - important for alt season detection
            eth_btc = eth_prices / btc_prices
            ratio_df = pd.DataFrame(index=btc_prices.index)
            ratio_df['eth_btc_ratio'] = eth_btc
            ratio_df['eth_btc_change'] = eth_btc.pct_change(5)
            ratio_df['eth_btc_zscore'] = (
                eth_btc - eth_btc.rolling(30).mean()
            ) / (eth_btc.rolling(30).std() + 1e-10)
            # ETH outperformance indicator
            ratio_df['eth_outperforming'] = (eth_btc.pct_change(5) > 0).astype(float)
            all_features.append(ratio_df)

        # =====================================================================
        # OPTIONAL: Funding rate features (derivatives sentiment)
        # =====================================================================
        if funding_rates is not None and not funding_rates.empty:
            all_features.append(self.compute_funding_features(funding_rates))

        # =====================================================================
        # OPTIONAL: DeFi-specific features (PDF Section 2.3 Option A)
        # =====================================================================
        defi_feats = self.compute_defi_features(
            tvl_data=tvl_data,
            gas_prices=gas_prices,
            liquidations=liquidations,
            stablecoin_mcap=stablecoin_mcap,
            total_crypto_mcap=total_crypto_mcap,
            dex_volume=dex_volume,
            cex_volume=cex_volume,
            bridge_volume=bridge_volume
        )
        if not defi_feats.empty:
            all_features.append(defi_feats)

        # =====================================================================
        # COMBINE AND CLEAN
        # =====================================================================
        features = pd.concat(all_features, axis=1)

        # Handle duplicate columns (can happen with merges)
        features = features.loc[:, ~features.columns.duplicated()]

        # Forward fill small gaps then drop remaining NaN
        features = features.ffill(limit=3)
        features = features.dropna()

        self.feature_names = list(features.columns)
        logger.info(f"Prepared {len(features)} obs with {len(self.feature_names)} features")
        logger.info(f"Feature categories: Returns, Volatility, Funding, DeFi")

        return features


# =============================================================================
# CRYPTO REGIME DETECTOR
# =============================================================================

class CryptoRegimeDetector:
    """
    Comprehensive regime detection using HMM.
    
    Supports multiple detection methods with comprehensive
    diagnostics and model persistence.
    
    Parameters
    ----------
    n_regimes : int, default=4
        Number of hidden states (regimes)
    detector_type : DetectorType, default=HMM
        Type of detector to use
    n_iter : int, default=100
        EM iterations for training
    random_state : int, default=42
        Random seed
    
    Example
    -------
    >>> detector = CryptoRegimeDetector(n_regimes=4)
    >>> detector.fit(features)
    >>> state = detector.get_current_state(features)
    >>> print(f"Regime: {state.current_regime}, Conf: {state.confidence:.2f}")
    """
    
    def __init__(
        self,
        n_regimes: int = 4,
        detector_type: DetectorType = DetectorType.HMM,
        n_iter: int = 100,
        random_state: int = 42,
        covariance_type: str = 'full'
    ):
        """Initialize detector."""
        self.n_regimes = n_regimes
        self.detector_type = detector_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.covariance_type = covariance_type
        
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.regime_labels: Dict[int, MarketRegime] = {}
        self.is_fitted: bool = False
        
        self.history = RegimeHistory()
        
        # Import model library
        self._use_hmm = detector_type == DetectorType.HMM
        try:
            if self._use_hmm:
                from hmmlearn import hmm
                self._hmm_module = hmm
            else:
                from sklearn.mixture import GaussianMixture
                self._gmm_module = GaussianMixture
        except ImportError:
            logger.warning("hmmlearn not available, using GMM")
            self._use_hmm = False
            from sklearn.mixture import GaussianMixture
            self._gmm_module = GaussianMixture
    
    def _create_model(self):
        """Create underlying model."""
        if self._use_hmm:
            return self._hmm_module.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
        else:
            return self._gmm_module(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                n_init=10,
                random_state=self.random_state,
            )
    
    def _scale_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale features."""
        from sklearn.preprocessing import StandardScaler
        
        if fit:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X)
        
        if self.scaler is None:
            raise ValueError("Scaler not fitted")
        return self.scaler.transform(X)
    
    def fit(self, features: pd.DataFrame) -> 'CryptoRegimeDetector':
        """
        Fit regime detection model.

        Args:
            features: Feature matrix

        Returns:
            Self (fitted detector)
        """
        self.feature_names = list(features.columns)

        # Clean data: handle NaN/inf values before scaling
        features_clean = features.copy()

        # Replace inf with NaN, then fill with column median
        features_clean = features_clean.replace([np.inf, -np.inf], np.nan)

        # For each column, fill NaN with median (robust to outliers)
        for col in features_clean.columns:
            if features_clean[col].isna().any():
                median_val = features_clean[col].median()
                if np.isnan(median_val) or np.isinf(median_val):
                    median_val = 0.0  # Fallback to 0 if all values are NaN
                features_clean[col] = features_clean[col].fillna(median_val)

        # Clip extreme values to prevent overflow (99.9th percentile)
        for col in features_clean.columns:
            q_low = features_clean[col].quantile(0.001)
            q_high = features_clean[col].quantile(0.999)
            if not np.isnan(q_low) and not np.isnan(q_high):
                features_clean[col] = features_clean[col].clip(lower=q_low, upper=q_high)

        X = features_clean.values

        # Final safety check - any remaining NaN/inf to zero
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_scaled = self._scale_features(X, fit=True)
        
        self.model = self._create_model()
        self.model.fit(X_scaled)
        
        self._label_regimes(features)
        
        self.is_fitted = True
        logger.info(f"Fitted {self.detector_type.value} with {self.n_regimes} regimes")
        
        return self
    
    def _label_regimes(self, features: pd.DataFrame):
        """Label regimes by volatility level."""
        vol_cols = [c for c in features.columns if 'vol' in c.lower() and 'change' not in c.lower()]
        
        if not vol_cols:
            # Default labeling
            regime_order = list(range(self.n_regimes))
        else:
            vol_col = vol_cols[0]
            vol_idx = self.feature_names.index(vol_col)
            
            means = self.model.means_
            unscaled_means = self.scaler.inverse_transform(means)
            
            regime_vols = [(i, unscaled_means[i, vol_idx]) for i in range(self.n_regimes)]
            regime_vols.sort(key=lambda x: x[1])
            regime_order = [r[0] for r in regime_vols]
        
        # Map to MarketRegime
        labels = [
            MarketRegime.LOW_VOL,
            MarketRegime.MEDIUM_VOL,
            MarketRegime.HIGH_VOL,
            MarketRegime.CRISIS,
        ][:self.n_regimes]
        
        self.regime_labels = {}
        for i, regime_idx in enumerate(regime_order):
            if i < len(labels):
                self.regime_labels[regime_idx] = labels[i]
            else:
                self.regime_labels[regime_idx] = MarketRegime.UNKNOWN
        
        logger.info(f"Regime labels: {self.regime_labels}")
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict regime for each observation."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        # Clean data: handle NaN/inf values before scaling (same as fit)
        features_clean = features[self.feature_names].copy()
        features_clean = features_clean.replace([np.inf, -np.inf], np.nan)

        for col in features_clean.columns:
            if features_clean[col].isna().any():
                median_val = features_clean[col].median()
                if np.isnan(median_val) or np.isinf(median_val):
                    median_val = 0.0
                features_clean[col] = features_clean[col].fillna(median_val)

        X = features_clean.values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_scaled = self._scale_features(X)
        
        raw_preds = self.model.predict(X_scaled)
        
        labeled = pd.Series(
            [self.regime_labels.get(p, MarketRegime.UNKNOWN) for p in raw_preds],
            index=features.index,
            name='regime'
        )
        
        return labeled
    
    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Get regime probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X = features[self.feature_names].values
        X_scaled = self._scale_features(X)
        
        probs = self.model.predict_proba(X_scaled)
        
        columns = [
            f'prob_{self.regime_labels.get(i, MarketRegime.UNKNOWN).value}'
            for i in range(self.n_regimes)
        ]
        
        return pd.DataFrame(probs, index=features.index, columns=columns)
    
    def get_current_state(
        self,
        features: pd.DataFrame,
        lookback: int = 5
    ) -> RegimeState:
        """
        Get current regime state with full diagnostics.
        
        Args:
            features: Feature matrix (most recent at end)
            lookback: Observations for confidence calculation
            
        Returns:
            RegimeState with probabilities and diagnostics
        """
        if len(features) < lookback:
            lookback = len(features)
        
        recent = features.iloc[-lookback:]
        
        regimes = self.predict(recent)
        probs = self.predict_proba(recent)
        
        current = regimes.iloc[-1]
        current_probs_raw = probs.iloc[-1].to_dict()
        
        # Map to MarketRegime
        current_probs = {}
        for i in range(self.n_regimes):
            regime = self.regime_labels.get(i, MarketRegime.UNKNOWN)
            prob_key = f'prob_{regime.value}'
            current_probs[regime] = current_probs_raw.get(prob_key, 0.0)
        
        confidence = max(current_probs.values())
        
        # Transition probability
        transitions = (regimes != regimes.shift(1)).sum() / max(lookback - 1, 1)
        
        # Feature values
        feature_values = recent.iloc[-1].to_dict()
        
        # Log-likelihood
        X = recent[self.feature_names].values
        X_scaled = self._scale_features(X)
        log_likelihood = self.model.score(X_scaled) if hasattr(self.model, 'score') else None
        
        state = RegimeState(
            current_regime=current,
            regime_probabilities=current_probs,
            confidence=confidence,
            timestamp=features.index[-1] if hasattr(features.index[-1], 'isoformat') else pd.Timestamp.now(),
            features_used=feature_values,
            log_likelihood=log_likelihood,
            n_observations_used=lookback,
            detector_type=self.detector_type,
        )
        
        # Add to history
        self.history.add_state(state)
        
        return state
    
    def save(self, path: str):
        """Save fitted model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'regime_labels': self.regime_labels,
            'n_regimes': self.n_regimes,
            'detector_type': self.detector_type,
            'use_hmm': self._use_hmm,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved detector to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'CryptoRegimeDetector':
        """Load fitted model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        detector = cls(
            n_regimes=data['n_regimes'],
            detector_type=data.get('detector_type', DetectorType.HMM)
        )
        detector.model = data['model']
        detector.scaler = data['scaler']
        detector.feature_names = data['feature_names']
        detector.regime_labels = data['regime_labels']
        detector._use_hmm = data.get('use_hmm', True)
        detector.is_fitted = True
        
        return detector


# =============================================================================
# REGIME-AWARE STRATEGY
# =============================================================================

class RegimeAwareStrategy:
    """
    Pairs trading strategy with regime-based parameter adjustment.
    
    Dynamically adjusts all trading parameters based on current
    market regime detected by the HMM.
    """
    
    def __init__(
        self,
        regime_detector: CryptoRegimeDetector,
        custom_configs: Optional[Dict[MarketRegime, RegimeConfig]] = None
    ):
        """Initialize strategy."""
        self.detector = regime_detector
        self.configs = custom_configs or {
            r: RegimeConfig.from_regime(r) for r in MarketRegime
        }
        self._current_state: Optional[RegimeState] = None
    
    def update_regime(self, features: pd.DataFrame) -> RegimeState:
        """Update regime from latest features."""
        self._current_state = self.detector.get_current_state(features)
        return self._current_state
    
    @property
    def current_state(self) -> Optional[RegimeState]:
        """Current regime state."""
        return self._current_state
    
    @property
    def current_regime(self) -> MarketRegime:
        """Current regime."""
        if self._current_state:
            return self._current_state.current_regime
        return MarketRegime.UNKNOWN
    
    @property
    def current_config(self) -> RegimeConfig:
        """Config for current regime."""
        return self.configs.get(
            self.current_regime,
            RegimeConfig.from_regime(MarketRegime.UNKNOWN)
        )
    
    def should_trade(
        self,
        tier: int,
        is_dex: bool,
        current_dex_count: int
    ) -> Tuple[bool, str]:
        """Check if trade allowed in current regime."""
        config = self.current_config
        
        if tier not in config.allowed_tiers:
            return False, f"Tier {tier} not allowed in {self.current_regime.value}"
        
        if is_dex and current_dex_count >= config.max_dex_positions:
            return False, f"Max DEX positions ({config.max_dex_positions}) reached"
        
        return True, f"Trade allowed in {self.current_regime.value}"
    
    def get_position_size(self, base_size: float) -> float:
        """Get regime-adjusted position size."""
        return base_size * self.current_config.position_multiplier
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get current trading thresholds."""
        config = self.current_config
        return {
            'entry_z': config.entry_z_threshold,
            'exit_z': config.exit_z_threshold,
            'stop_z': config.stop_z_threshold,
            'max_holding_days': config.max_holding_days,
            'max_leverage': config.max_leverage,
        }
    
    def get_history_summary(self) -> Dict[str, Any]:
        """Get regime history summary."""
        return self.detector.history.summary()


# =============================================================================
# SIMPLE RULE-BASED DETECTOR
# =============================================================================

def detect_regime_simple(
    btc_prices: pd.Series,
    lookback: int = 20,
    vol_high_pctl: float = 0.80,
    vol_low_pctl: float = 0.40,
    momentum_threshold: float = 0.15
) -> MarketRegime:
    """
    Simple rule-based regime detection.
    
    Args:
        btc_prices: BTC price series
        lookback: Volatility lookback
        vol_high_pctl: High volatility percentile
        vol_low_pctl: Low volatility percentile
        momentum_threshold: Trend threshold
        
    Returns:
        Detected regime
    """
    if len(btc_prices) < lookback * 3:
        return MarketRegime.UNKNOWN
    
    returns = btc_prices.pct_change().dropna()
    
    # Volatility
    recent_vol = returns.iloc[-lookback:].std() * np.sqrt(365)
    rolling_vol = returns.rolling(lookback).std() * np.sqrt(365)
    vol_high = rolling_vol.quantile(vol_high_pctl)
    vol_low = rolling_vol.quantile(vol_low_pctl)
    
    # Momentum
    momentum = btc_prices.iloc[-1] / btc_prices.iloc[-lookback] - 1
    
    # Classify
    if recent_vol > vol_high * 1.5:
        return MarketRegime.CRISIS
    
    if recent_vol > vol_high:
        return MarketRegime.HIGH_VOL
    
    if abs(momentum) > momentum_threshold:
        return MarketRegime.TRENDING_UP if momentum > 0 else MarketRegime.TRENDING_DOWN
    
    if recent_vol < vol_low:
        return MarketRegime.LOW_VOL
    
    return MarketRegime.MEDIUM_VOL