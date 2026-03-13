"""
Signal Combiner Module for Pairs Trading Strategy
==================================================

Combines baseline signals with enhancement layers (regime detection, ML predictions,
dynamic pair selection) into unified, strength-weighted trading signals.

PDF Requirement (Section 2.3):
Integration of regime detection, ML predictions, and dynamic pair selection
into a unified signal enhancement system.

Mathematical Framework
----------------------
Signal Strength Adjustment:

    S_adjusted = S_baseline × R_multiplier × ML_boost

    Where:
        S_baseline = Original signal from z-score mean reversion
        R_multiplier = Regime-specific position multiplier (0.2-1.0)
        ML_boost = ML agreement/disagreement adjustment (0.6-1.25)

Regime Multipliers (from PDF Section 2.3 Option A):
    LOW_VOL (0):      1.0   - Full position size
    MEDIUM_VOL (1):   0.85  - Slightly reduced
    HIGH_VOL (2):     0.6   - Significantly reduced
    TRENDING_UP (3):  0.9   - Reduced (trends break mean reversion)
    TRENDING_DOWN (4):0.9   - Reduced
    CRISIS (5):       0.2   - Minimal exposure

ML Agreement Logic:
    - High confidence agreement (>70%): 1.25× boost
    - Medium confidence agreement (50-70%): 1.1× boost
    - High confidence disagreement (>70%): 0.6× penalty

Enhancement Quality Metric:
    Q = 0.4 × R_multiplier + 0.4 × ML_confidence + 0.2 × 1(S > 0.5)

Author: Crypto StatArb Quantitative Research
Version: 2.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SignalCombinerConfig:
    """Configuration for signal combination."""

    # Regime multipliers
    regime_multipliers: Dict[int, float] = None

    # ML boost/penalty factors
    ml_high_conf_boost: float = 1.25
    ml_med_conf_boost: float = 1.1
    ml_disagreement_penalty: float = 0.6

    # Confidence thresholds
    high_confidence_threshold: float = 0.7
    med_confidence_threshold: float = 0.5

    # Default ML confidence when not available
    default_ml_confidence: float = 0.5

    def __post_init__(self):
        """Set default regime multipliers if not provided."""
        if self.regime_multipliers is None:
            # MarketRegime: LOW_VOL=0, MEDIUM_VOL=1, HIGH_VOL=2,
            # TRENDING_UP=3, TRENDING_DOWN=4, CRISIS=5
            self.regime_multipliers = {
                0: 1.0,    # LOW_VOL: Full size
                1: 0.85,   # MEDIUM_VOL: Slightly reduced
                2: 0.6,    # HIGH_VOL: Significantly reduced
                3: 0.9,    # TRENDING_UP: Slightly reduced (trends break mean reversion)
                4: 0.9,    # TRENDING_DOWN: Slightly reduced
                5: 0.2,    # CRISIS: Minimal exposure
            }


# =============================================================================
# SIGNAL COMBINER CLASS
# =============================================================================

class SignalCombiner:
    """
    Combines baseline signals with enhancement layers.

    Implements PDF Section 2.3 requirements for integrated signal enhancement.
    """

    def __init__(self, config: SignalCombinerConfig = None):
        """
        Initialize signal combiner.

        Args:
            config: Configuration object. Uses defaults if not provided.
        """
        self.config = config or SignalCombinerConfig()
        logger.info("SignalCombiner initialized with config: %s", self.config)

    def combine_enhancements(
        self,
        signals: pd.DataFrame,
        regime_states: pd.Series,
        ml_predictions: pd.DataFrame,
        dynamic_pairs: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Combine baseline signals with all three enhancement signals.

        PDF Requirement (Section 2.3):
        Integration of regime detection, ML predictions, and dynamic pair selection
        into a unified signal enhancement system.

        Args:
            signals: Baseline signals from Step 2
            regime_states: Regime states from HMM (Enhancement A)
            ml_predictions: ML predictions (Enhancement B)
            dynamic_pairs: Dynamic pair selection results (Enhancement C)

        Returns:
            Enhanced signals DataFrame with all enhancements integrated
        """
        if signals is None or len(signals) == 0:
            logger.warning("Empty signals provided to combine_enhancements")
            return pd.DataFrame()

        enhanced_signals = signals.copy()

        # Apply each enhancement layer
        enhanced_signals = self._apply_regime_enhancement(enhanced_signals, regime_states)
        enhanced_signals = self._apply_ml_enhancement(enhanced_signals, ml_predictions)
        enhanced_signals = self._calculate_signal_strength(enhanced_signals)
        enhanced_signals = self._apply_dynamic_pair_filter(enhanced_signals, dynamic_pairs)
        enhanced_signals = self._calculate_final_signal(enhanced_signals)

        logger.info(
            "Combined enhancements: %d signals processed, avg strength: %.3f",
            len(enhanced_signals),
            enhanced_signals.get('signal_strength', pd.Series([0.5])).mean()
        )

        return enhanced_signals

    def _apply_regime_enhancement(
        self,
        signals: pd.DataFrame,
        regime_states: pd.Series
    ) -> pd.DataFrame:
        """
        Apply Enhancement A: Regime-based signal adjustment.

        Args:
            signals: Signals DataFrame
            regime_states: Regime states from HMM

        Returns:
            Signals with regime enhancement applied
        """
        if isinstance(regime_states, pd.Series) and len(regime_states) > 0:
            # Handle index dtype mismatch: signals may have int index with
            # 'timestamp' column while regime_states has datetime index
            if (signals.index.dtype != regime_states.index.dtype and
                    'timestamp' in signals.columns):
                ts_col = pd.to_datetime(signals['timestamp'], utc=True)
                # Use merge_asof for duplicate-safe nearest-time lookup
                sig_ts = pd.DataFrame({'_ts': ts_col, '_idx': range(len(ts_col))}).sort_values('_ts')
                reg_df = regime_states.to_frame('regime').sort_index()
                reg_df.index = pd.to_datetime(reg_df.index, utc=True)
                merged = pd.merge_asof(sig_ts, reg_df, left_on='_ts', right_index=True, direction='backward')
                merged = merged.sort_values('_idx')
                signals['regime'] = merged['regime'].values
            else:
                signals['regime'] = regime_states.reindex(
                    signals.index, method='ffill'
                )

            # Apply regime multipliers
            signals['regime_multiplier'] = signals['regime'].apply(
                lambda r: self.config.regime_multipliers.get(
                    r.value if hasattr(r, 'value') else int(r) if pd.notna(r) else 1,
                    0.7  # Default for unknown regimes
                )
            )
        else:
            signals['regime'] = 0
            signals['regime_multiplier'] = 1.0

        return signals

    def _apply_ml_enhancement(
        self,
        signals: pd.DataFrame,
        ml_predictions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply Enhancement B: ML prediction integration.

        Args:
            signals: Signals DataFrame
            ml_predictions: ML predictions DataFrame

        Returns:
            Signals with ML enhancement applied
        """
        if isinstance(ml_predictions, pd.DataFrame) and len(ml_predictions) > 0:
            # Handle index dtype mismatch: use timestamp column if available
            use_ts = (signals.index.dtype != ml_predictions.index.dtype and
                      'timestamp' in signals.columns)

            if use_ts:
                # Use merge_asof for duplicate-safe nearest-time lookup
                ts_col = pd.to_datetime(signals['timestamp'], utc=True)
                sig_ts = pd.DataFrame({'_ts': ts_col, '_idx': range(len(ts_col))}).sort_values('_ts')

                # Prepare ML predictions with datetime index
                ml_df = ml_predictions.copy()
                ml_df.index = pd.to_datetime(ml_df.index, utc=True)
                ml_df = ml_df.sort_index()

                merged = pd.merge_asof(sig_ts, ml_df, left_on='_ts', right_index=True, direction='nearest')
                merged = merged.sort_values('_idx')

                signals['ml_prediction'] = merged['prediction'].values if 'prediction' in merged.columns else 0
                signals['ml_confidence'] = merged['confidence'].values if 'confidence' in merged.columns else self.config.default_ml_confidence
                if 'kalman_hedge_ratio' in merged.columns:
                    signals['kalman_hedge_ratio'] = merged['kalman_hedge_ratio'].values
            else:
                # Add ML predictions
                if 'prediction' in ml_predictions.columns:
                    signals['ml_prediction'] = ml_predictions['prediction'].reindex(
                        signals.index, fill_value=0
                    )
                else:
                    signals['ml_prediction'] = 0

                # Add ML confidence
                if 'confidence' in ml_predictions.columns:
                    signals['ml_confidence'] = ml_predictions['confidence'].reindex(
                        signals.index, fill_value=self.config.default_ml_confidence
                    )
                else:
                    signals['ml_confidence'] = self.config.default_ml_confidence

                # Add Kalman hedge ratio if available
                if 'kalman_hedge_ratio' in ml_predictions.columns:
                    signals['kalman_hedge_ratio'] = ml_predictions['kalman_hedge_ratio'].reindex(
                        signals.index, method='ffill'
                    )
        else:
            signals['ml_prediction'] = 0
            signals['ml_confidence'] = self.config.default_ml_confidence

        return signals

    def _calculate_signal_strength(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate combined signal strength incorporating all enhancements.

        Args:
            signals: Signals DataFrame with regime and ML enhancements

        Returns:
            Signals with calculated signal_strength
        """
        if 'signal_strength' not in signals.columns:
            return signals

        original_strength = signals['signal_strength'].copy()

        # Start with regime-adjusted strength
        signals['signal_strength'] = original_strength * signals['regime_multiplier']

        # ML agreement boost/penalty
        if 'signal' in signals.columns:
            # Calculate ML agreement score
            signal_direction = np.sign(signals['signal'])
            ml_direction = np.sign(signals['ml_prediction'])

            # Agreement mask: both agree on direction
            ml_agreement = (signal_direction == ml_direction) & (signal_direction != 0)
            ml_disagreement = (signal_direction != ml_direction) & (signal_direction != 0) & (ml_direction != 0)

            # Boost when ML agrees with high confidence
            high_conf_agreement = ml_agreement & (
                signals['ml_confidence'] > self.config.high_confidence_threshold
            )
            signals.loc[high_conf_agreement, 'signal_strength'] *= self.config.ml_high_conf_boost

            # Moderate boost for medium confidence agreement
            med_conf_agreement = ml_agreement & (
                signals['ml_confidence'] > self.config.med_confidence_threshold
            ) & (
                signals['ml_confidence'] <= self.config.high_confidence_threshold
            )
            signals.loc[med_conf_agreement, 'signal_strength'] *= self.config.ml_med_conf_boost

            # Penalty when ML disagrees with high confidence
            high_conf_disagreement = ml_disagreement & (
                signals['ml_confidence'] > self.config.high_confidence_threshold
            )
            signals.loc[high_conf_disagreement, 'signal_strength'] *= self.config.ml_disagreement_penalty

        # Clip to valid range [0, 1]
        signals['signal_strength'] = signals['signal_strength'].clip(0, 1)

        # Add enhancement quality metric
        signals['enhancement_quality'] = (
            signals['regime_multiplier'] * 0.4 +
            signals['ml_confidence'] * 0.4 +
            (signals['signal_strength'] > 0.5).astype(float) * 0.2
        )

        return signals

    def _apply_dynamic_pair_filter(
        self,
        signals: pd.DataFrame,
        dynamic_pairs: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Apply Enhancement C: Dynamic pair selection filtering.

        Args:
            signals: Signals DataFrame
            dynamic_pairs: Dynamic pair selection results

        Returns:
            Signals with dynamic pair filter applied
        """
        if dynamic_pairs is not None and len(dynamic_pairs) > 0:
            # Add pair tier information if available
            if 'pair' in signals.columns:
                # This would filter signals based on dynamically selected pairs
                # For now, just add a flag
                signals['dynamic_pair_selected'] = True
            signals['dynamic_selection_active'] = True
        else:
            signals['dynamic_selection_active'] = False

        return signals

    def _calculate_final_signal(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate final adjusted signal incorporating all enhancements.

        Args:
            signals: Signals DataFrame with all enhancements

        Returns:
            Signals with adjusted_signal column
        """
        if 'signal' in signals.columns and 'signal_strength' in signals.columns:
            # Create final adjusted signal incorporating all enhancements
            signals['adjusted_signal'] = (
                signals['signal'] * signals['signal_strength']
            )
        elif 'signal' in signals.columns:
            signals['adjusted_signal'] = signals['signal']

        return signals


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTION
# =============================================================================

def combine_enhancements(
    signals: pd.DataFrame,
    regime_states: pd.Series,
    ml_predictions: pd.DataFrame,
    dynamic_pairs: Optional[pd.DataFrame] = None,
    config: SignalCombinerConfig = None
) -> pd.DataFrame:
    """
    Module-level convenience function for combining enhancements.

    This is the primary entry point for the orchestrator to call.

    PDF Requirement (Section 2.3):
    Integration of regime detection, ML predictions, and dynamic pair selection
    into a unified signal enhancement system.

    Args:
        signals: Baseline signals from Step 2
        regime_states: Regime states from HMM (Enhancement A)
        ml_predictions: ML predictions (Enhancement B)
        dynamic_pairs: Dynamic pair selection results (Enhancement C)
        config: Optional configuration object

    Returns:
        Enhanced signals DataFrame with all enhancements integrated

    Example:
        >>> from strategies.pairs_trading.signal_combiner import combine_enhancements
        >>> enhanced = combine_enhancements(
        ...     signals=baseline_signals,
        ...     regime_states=hmm_states,
        ...     ml_predictions=ml_preds,
        ...     dynamic_pairs=selected_pairs
        ... )
    """
    combiner = SignalCombiner(config=config)
    return combiner.combine_enhancements(
        signals=signals,
        regime_states=regime_states,
        ml_predictions=ml_predictions,
        dynamic_pairs=dynamic_pairs
    )
