"""
Adaptive Threshold Calculator for Pairs Trading
===============================================

Implements volatility-adjusted and regime-aware entry/exit thresholds.

Static thresholds (e.g., always ±2.0) perform poorly because:
- In high volatility: Too tight → many false signals
- In low volatility: Too wide → miss opportunities
- Regime changes: Need different behavior in trending vs mean-reverting markets

Mathematical Framework
----------------------

Volatility Adjustment:
    adjusted_threshold = base_threshold × volatility_multiplier

Where volatility_multiplier is calculated from recent spread volatility:
    vol_mult = min(max(current_vol / target_vol, 0.7), 1.5)

Regime Adjustment:
    final_threshold = adjusted_threshold × regime_factor

Where regime_factor depends on market state:
    - High volatility regime: 1.2-1.5 (widen thresholds)
    - Mean-reverting regime: 0.8-0.9 (tighten thresholds)
    - Trending regime: 1.3-1.6 (widen to avoid false entries)

Author: Crypto StatArb Quantitative Research
Version: 2.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    MEAN_REVERTING = "mean_reverting"
    TRENDING = "trending"
    NORMAL = "normal"

    @property
    def threshold_multiplier(self) -> float:
        """Multiplier to apply to base thresholds."""
        multipliers = {
            self.HIGH_VOLATILITY: 1.3,   # Widen thresholds
            self.LOW_VOLATILITY: 0.85,   # Tighten thresholds
            self.MEAN_REVERTING: 0.9,    # Slightly tighten
            self.TRENDING: 1.4,          # Significantly widen
            self.NORMAL: 1.0             # No adjustment
        }
        return multipliers.get(self, 1.0)


@dataclass
class ThresholdConfig:
    """Configuration for adaptive threshold calculation."""

    # Base thresholds (static minimums)
    base_entry_cex: float = 2.0
    base_entry_dex: float = 2.5
    base_exit_cex: float = 0.0
    base_exit_dex: float = 1.0
    base_stop: float = 3.0

    # Volatility adjustment parameters
    target_volatility: float = 0.15      # Target annual volatility
    min_vol_multiplier: float = 0.7      # Don't go below 70% of base
    max_vol_multiplier: float = 1.5      # Don't exceed 150% of base
    vol_lookback: int = 24               # Hours for volatility calculation

    # Regime detection parameters
    regime_lookback: int = 168           # Hours (7 days)
    high_vol_threshold: float = 0.25     # Annual vol > 25% = high vol
    low_vol_threshold: float = 0.10      # Annual vol < 10% = low vol
    trend_threshold: float = 0.6         # Hurst > 0.6 = trending

    # Smoothing
    smoothing_alpha: float = 0.3         # EMA smoothing factor


@dataclass
class AdaptiveThresholds:
    """Calculated adaptive thresholds for a pair."""

    entry_threshold: float
    exit_threshold: float
    stop_threshold: float
    volatility_multiplier: float
    regime_multiplier: float
    current_regime: MarketRegime
    spread_volatility: float

    def summary(self) -> dict:
        """Return summary as dict."""
        return {
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold,
            'stop_threshold': self.stop_threshold,
            'volatility_multiplier': self.volatility_multiplier,
            'regime_multiplier': self.regime_multiplier,
            'current_regime': self.current_regime.value,
            'spread_volatility': self.spread_volatility
        }


class AdaptiveThresholdCalculator:
    """
    Calculate volatility-adjusted and regime-aware thresholds.

    Automatically adjusts entry/exit/stop thresholds based on:
    1. Recent spread volatility
    2. Current market regime
    3. Venue type (CEX vs DEX)
    """

    def __init__(self, config: Optional[ThresholdConfig] = None):
        """
        Initialize threshold calculator.

        Args:
            config: Configuration parameters (uses defaults if None)
        """
        self.config = config or ThresholdConfig()

        # State
        self._smoothed_vol: Dict[str, float] = {}
        self._regime_history: Dict[str, list] = {}

    def calculate(
        self,
        pair_name: str,
        spread: pd.Series,
        venue: str = 'CEX',
        zscore: Optional[pd.Series] = None
    ) -> AdaptiveThresholds:
        """
        Calculate adaptive thresholds for a pair.

        Args:
            pair_name: Identifier for the pair (for state tracking)
            spread: Spread time series
            venue: 'CEX' or 'DEX'
            zscore: Optional z-score series for regime detection

        Returns:
            AdaptiveThresholds with calculated values
        """
        # 1. Calculate spread volatility
        spread_vol = self._calculate_spread_volatility(spread)

        # 2. Calculate volatility multiplier
        vol_mult = self._calculate_volatility_multiplier(
            pair_name, spread_vol
        )

        # 3. Detect market regime
        regime = self._detect_regime(
            pair_name, spread, zscore
        )

        # 4. Get regime multiplier
        regime_mult = regime.threshold_multiplier

        # 5. Calculate base thresholds for venue
        if venue.upper() == 'DEX':
            base_entry = self.config.base_entry_dex
            base_exit = self.config.base_exit_dex
        else:
            base_entry = self.config.base_entry_cex
            base_exit = self.config.base_exit_cex

        base_stop = self.config.base_stop

        # 6. Apply multipliers
        entry = base_entry * vol_mult * regime_mult
        exit_threshold = base_exit * vol_mult * regime_mult
        stop = base_stop * vol_mult * regime_mult

        # 7. Enforce sanity bounds
        entry = max(1.5, min(entry, 4.0))      # Entry between 1.5-4.0
        stop = max(2.5, min(stop, 5.0))        # Stop between 2.5-5.0

        return AdaptiveThresholds(
            entry_threshold=entry,
            exit_threshold=exit_threshold,
            stop_threshold=stop,
            volatility_multiplier=vol_mult,
            regime_multiplier=regime_mult,
            current_regime=regime,
            spread_volatility=spread_vol
        )

    def _calculate_spread_volatility(self, spread: pd.Series) -> float:
        """Calculate realized spread volatility (annualized)."""
        if len(spread) < self.config.vol_lookback:
            lookback = len(spread)
        else:
            lookback = self.config.vol_lookback

        recent_spread = spread.iloc[-lookback:]

        # Calculate hourly returns
        returns = recent_spread.pct_change().dropna()

        if len(returns) < 2:
            return 0.15  # Default

        # Annualize (assuming hourly data)
        hourly_vol = returns.std()
        annual_vol = hourly_vol * np.sqrt(24 * 365)

        return annual_vol

    def _calculate_volatility_multiplier(
        self,
        pair_name: str,
        current_vol: float
    ) -> float:
        """
        Calculate volatility multiplier with smoothing.

        High volatility → increase thresholds (multiplier > 1)
        Low volatility → decrease thresholds (multiplier < 1)
        """
        # Smoothing using EMA
        if pair_name in self._smoothed_vol:
            smoothed = (
                self.config.smoothing_alpha * current_vol +
                (1 - self.config.smoothing_alpha) * self._smoothed_vol[pair_name]
            )
        else:
            smoothed = current_vol

        self._smoothed_vol[pair_name] = smoothed

        # Calculate multiplier
        mult = smoothed / self.config.target_volatility

        # Bound the multiplier
        mult = max(self.config.min_vol_multiplier, mult)
        mult = min(self.config.max_vol_multiplier, mult)

        return mult

    def _detect_regime(
        self,
        pair_name: str,
        spread: pd.Series,
        zscore: Optional[pd.Series]
    ) -> MarketRegime:
        """
        Detect current market regime.

        Uses multiple indicators:
        1. Volatility level
        2. Hurst exponent (trending vs mean-reverting)
        3. Z-score behavior
        """
        lookback = min(self.config.regime_lookback, len(spread))
        recent = spread.iloc[-lookback:]

        # 1. Volatility-based classification
        vol = self._calculate_spread_volatility(recent)

        if vol > self.config.high_vol_threshold:
            return MarketRegime.HIGH_VOLATILITY
        elif vol < self.config.low_vol_threshold:
            return MarketRegime.LOW_VOLATILITY

        # 2. Hurst exponent for trending vs mean-reverting
        hurst = self._calculate_hurst(recent)

        if hurst > self.config.trend_threshold:
            return MarketRegime.TRENDING
        elif hurst < 0.45:  # Mean-reverting
            return MarketRegime.MEAN_REVERTING

        return MarketRegime.NORMAL

    def _calculate_hurst(self, series: pd.Series) -> float:
        """
        Calculate Hurst exponent using R/S analysis.

        H < 0.5: Mean-reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        if len(series) < 20:
            return 0.5  # Assume random walk

        lags = range(2, min(len(series) // 2, 20))
        tau = []
        rs_values = []

        for lag in lags:
            # Split into sub-periods
            data = series.values
            n_chunks = len(data) // lag

            if n_chunks < 2:
                continue

            rs_chunk = []
            for i in range(n_chunks):
                chunk = data[i*lag:(i+1)*lag]

                if len(chunk) < 2:
                    continue

                # Calculate R/S for this chunk
                mean = np.mean(chunk)
                deviations = chunk - mean
                cumsum = np.cumsum(deviations)

                R = np.max(cumsum) - np.min(cumsum)  # Range
                S = np.std(chunk, ddof=1)             # Std dev

                if S > 0:
                    rs_chunk.append(R / S)

            if rs_chunk:
                rs_values.append(np.mean(rs_chunk))
                tau.append(lag)

        if len(tau) < 2:
            return 0.5

        # Hurst = slope of log(R/S) vs log(tau)
        log_tau = np.log(tau)
        log_rs = np.log(rs_values)

        # Linear regression
        hurst = np.polyfit(log_tau, log_rs, 1)[0]

        # Bound to reasonable range
        return max(0.0, min(1.0, hurst))

    def get_regime_statistics(self, pair_name: str) -> dict:
        """Get regime detection statistics for a pair."""
        if pair_name not in self._regime_history:
            return {}

        history = self._regime_history[pair_name]
        if not history:
            return {}

        from collections import Counter
        regime_counts = Counter([r.value for r in history[-100:]])  # Last 100

        return {
            'recent_regimes': dict(regime_counts),
            'current_smoothed_vol': self._smoothed_vol.get(pair_name, 0.0)
        }
