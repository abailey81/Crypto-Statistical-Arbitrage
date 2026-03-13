"""
Kalman Filter for Dynamic Hedge Ratio Estimation
=================================================

Implements Kalman filtering for time-varying hedge ratio estimation in pairs trading.
This allows hedge ratios to adapt to changing market conditions rather than remaining static.

Mathematical Framework
----------------------

State-Space Model:
    Observation equation: y_t = α_t + β_t * x_t + v_t,  v_t ~ N(0, R)
    State evolution:      α_t = α_(t-1) + w_α,  w_α ~ N(0, Q_α)
                          β_t = β_(t-1) + w_β,  w_β ~ N(0, Q_β)

Where:
    y_t = log(Price_A)  - dependent variable
    x_t = log(Price_B)  - independent variable
    α_t = intercept (time-varying)
    β_t = hedge ratio (time-varying)
    R   = observation noise covariance
    Q   = state transition noise covariance

The Kalman filter recursively estimates α_t and β_t by:
1. Prediction: Forecast next state based on current estimate
2. Update: Refine forecast using new observation

Advantages over OLS:
- Adapts to regime changes in real-time
- Smooths out noise in hedge ratio estimates
- Provides uncertainty quantification

Author: Crypto StatArb Quantitative Research
Version: 2.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class KalmanHedgeResult:
    """Results from Kalman filter hedge ratio estimation."""

    hedge_ratios: pd.Series      # Time-varying β_t
    intercepts: pd.Series         # Time-varying α_t
    spreads: pd.Series            # Calculated spread: y - β*x - α
    covariances: np.ndarray       # State covariance matrices (T x 2 x 2)
    prediction_errors: pd.Series  # Innovation sequence
    kalman_gain: np.ndarray       # Kalman gains over time (T x 2)
    log_likelihood: float         # Log-likelihood of model

    def summary(self) -> dict:
        """Return summary statistics."""
        return {
            'mean_hedge_ratio': float(self.hedge_ratios.mean()),
            'std_hedge_ratio': float(self.hedge_ratios.std()),
            'mean_intercept': float(self.intercepts.mean()),
            'std_intercept': float(self.intercepts.std()),
            'mean_abs_innovation': float(self.prediction_errors.abs().mean()),
            'log_likelihood': float(self.log_likelihood),
            'n_observations': len(self.hedge_ratios)
        }


class KalmanHedgeRatio:
    """
    Kalman Filter for time-varying hedge ratio estimation.

    Implements a 2-state Kalman filter where the states are:
    - State 1: Intercept (α)
    - State 2: Hedge ratio (β)

    Both states evolve as random walks with Gaussian noise.
    """

    def __init__(
        self,
        delta: float = 0.0001,
        obs_noise: float = 0.001,
        initial_hedge: Optional[float] = None,
        initial_intercept: float = 0.0
    ):
        """
        Initialize Kalman filter.

        Args:
            delta: State transition noise variance (Q matrix diagonal)
                  Higher values = hedge ratio can change more rapidly
                  Lower values = hedge ratio changes slowly
            obs_noise: Observation noise variance (R)
                      Variance of residuals
            initial_hedge: Initial hedge ratio estimate (default: OLS estimate)
            initial_intercept: Initial intercept estimate
        """
        self.delta = delta
        self.obs_noise = obs_noise
        self.initial_hedge = initial_hedge
        self.initial_intercept = initial_intercept

    def fit(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        use_log: bool = True
    ) -> KalmanHedgeResult:
        """
        Fit Kalman filter to estimate time-varying hedge ratio.

        Args:
            price_a: Price series for asset A (dependent variable)
            price_b: Price series for asset B (independent variable)
            use_log: Whether to use log prices (recommended for prices)

        Returns:
            KalmanHedgeResult with time-varying estimates
        """
        # Align series
        aligned = pd.DataFrame({
            'A': price_a,
            'B': price_b
        }).dropna()

        if len(aligned) < 10:
            raise ValueError(f"Insufficient data after alignment: {len(aligned)} rows")

        # Convert to log prices if requested
        if use_log:
            y = np.log(aligned['A'].values)
            x = np.log(aligned['B'].values)
        else:
            y = aligned['A'].values
            x = aligned['B'].values

        n = len(y)

        # Initialize state estimate
        if self.initial_hedge is None:
            # Use OLS as initial estimate
            initial_beta = np.cov(y, x)[0, 1] / np.var(x)
        else:
            initial_beta = self.initial_hedge

        initial_alpha = self.initial_intercept

        # State vector: [α, β]
        state = np.array([initial_alpha, initial_beta])

        # State covariance matrix P
        P = np.eye(2) * 1.0  # Initial uncertainty

        # Process noise covariance Q (random walk)
        Q = np.eye(2) * self.delta

        # Observation noise R
        R = self.obs_noise

        # Storage
        hedge_ratios = np.zeros(n)
        intercepts = np.zeros(n)
        spreads = np.zeros(n)
        covariances = np.zeros((n, 2, 2))
        prediction_errors = np.zeros(n)
        kalman_gains = np.zeros((n, 2))
        log_likelihood = 0.0

        # Kalman filter iteration
        for t in range(n):
            # ===== PREDICTION STEP =====
            # State prediction: state_pred = state (random walk)
            state_pred = state.copy()

            # Covariance prediction: P_pred = P + Q
            P_pred = P + Q

            # ===== UPDATE STEP =====
            # Observation at time t
            y_t = y[t]
            x_t = x[t]

            # Predicted observation: y_pred = α + β * x
            y_pred = state_pred[0] + state_pred[1] * x_t

            # Innovation (prediction error)
            innovation = y_t - y_pred
            prediction_errors[t] = innovation

            # Observation matrix H = [1, x_t] (derivative of observation w.r.t. state)
            H = np.array([1.0, x_t])

            # Innovation covariance: S = H * P_pred * H' + R
            S = H @ P_pred @ H.T + R

            # Kalman gain: K = P_pred * H' / S
            K = (P_pred @ H) / S
            kalman_gains[t] = K

            # State update: state = state_pred + K * innovation
            state = state_pred + K * innovation

            # Covariance update: P = (I - K * H) * P_pred
            I_KH = np.eye(2) - np.outer(K, H)
            P = I_KH @ P_pred

            # Store results
            intercepts[t] = state[0]
            hedge_ratios[t] = state[1]
            spreads[t] = y_t - state[0] - state[1] * x_t
            covariances[t] = P.copy()

            # Update log-likelihood (for model comparison)
            log_likelihood += -0.5 * (np.log(2 * np.pi * S) + (innovation ** 2) / S)

        return KalmanHedgeResult(
            hedge_ratios=pd.Series(hedge_ratios, index=aligned.index, name='hedge_ratio'),
            intercepts=pd.Series(intercepts, index=aligned.index, name='intercept'),
            spreads=pd.Series(spreads, index=aligned.index, name='spread'),
            covariances=covariances,
            prediction_errors=pd.Series(prediction_errors, index=aligned.index, name='innovation'),
            kalman_gain=kalman_gains,
            log_likelihood=log_likelihood
        )

    def smooth_zscore(
        self,
        zscore: pd.Series,
        smoothing_factor: float = 0.1
    ) -> pd.Series:
        """
        Apply Kalman smoothing to z-score series.

        Uses a simple 1D Kalman filter to smooth noisy z-scores.

        Args:
            zscore: Raw z-score series
            smoothing_factor: Process noise (0.01 = very smooth, 1.0 = follows data closely)

        Returns:
            Smoothed z-score series
        """
        z = zscore.dropna().values
        n = len(z)

        if n < 2:
            return zscore

        # Initialize
        state = z[0]  # Initial state = first z-score
        P = 1.0       # Initial variance
        Q = smoothing_factor  # Process noise
        R = 0.5       # Observation noise (assumed)

        smoothed = np.zeros(n)

        for t in range(n):
            # Prediction
            state_pred = state
            P_pred = P + Q

            # Update
            K = P_pred / (P_pred + R)
            state = state_pred + K * (z[t] - state_pred)
            P = (1 - K) * P_pred

            smoothed[t] = state

        # Create series with non-NaN values smoothed
        smoothed_series = pd.Series(smoothed, index=zscore.dropna().index, name='smoothed_zscore')
        # Reindex to original index to preserve length (NaN values stay NaN)
        return smoothed_series.reindex(zscore.index)


def compare_hedge_ratio_methods(
    price_a: pd.Series,
    price_b: pd.Series
) -> dict:
    """
    Compare Kalman filter hedge ratio to OLS and rolling OLS.

    Useful for diagnostics and understanding how much the hedge ratio varies.

    Returns:
        dict with 'kalman', 'ols', and 'rolling_ols' hedge ratios
    """
    # Kalman
    kalman = KalmanHedgeRatio()
    kalman_result = kalman.fit(price_a, price_b)

    # OLS (static)
    aligned = pd.DataFrame({'A': price_a, 'B': price_b}).dropna()
    log_a = np.log(aligned['A'])
    log_b = np.log(aligned['B'])
    ols_beta = np.cov(log_a, log_b)[0, 1] / np.var(log_b)

    # Rolling OLS (30-day window)
    window = min(30, len(aligned) // 3)
    rolling_beta = []
    for i in range(window, len(aligned)):
        window_a = log_a.iloc[i-window:i]
        window_b = log_b.iloc[i-window:i]
        beta = np.cov(window_a, window_b)[0, 1] / np.var(window_b)
        rolling_beta.append(beta)

    return {
        'kalman': kalman_result.hedge_ratios,
        'ols': ols_beta,
        'rolling_ols': pd.Series(
            [ols_beta] * window + rolling_beta,
            index=aligned.index
        ),
        'kalman_std': float(kalman_result.hedge_ratios.std()),
        'rolling_std': float(pd.Series(rolling_beta).std())
    }
