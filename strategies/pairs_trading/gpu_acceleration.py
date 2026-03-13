"""
GPU Acceleration Module for Crypto Statistical Arbitrage
=========================================================

High-performance GPU-accelerated computations for pairs trading analysis.
Supports AMD GPUs via PyOpenCL (Intel Macs) with automatic fallback to
enhanced Numba JIT compilation for CPU-only systems.

Acceleration Methods (in order of preference):
1. PyOpenCL - AMD/Intel GPU acceleration (OpenCL 1.2+)
2. Numba CUDA - NVIDIA GPU acceleration (if available)
3. Numba JIT - Multi-threaded CPU with SIMD
4. NumPy/SciPy - Intel MKL-optimized fallback

Key Optimizations:
- Batch processing of cointegration tests (100-1000x pairs in parallel)
- Vectorized correlation matrix computation
- GPU-accelerated OLS regression for hedge ratios
- Parallel ADF testing for spread stationarity
- Optimized memory layout for cache efficiency

Author: Tamer Atesyakar
Version: 1.0.0
Date: February 2026
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
import time
import multiprocessing

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# ACCELERATION BACKEND DETECTION
# =============================================================================

class AccelerationBackend(Enum):
    """Available acceleration backends."""
    PYOPENCL = "pyopencl"      # AMD/Intel GPU via OpenCL
    NUMBA_CUDA = "numba_cuda"  # NVIDIA GPU via CUDA
    NUMBA_CPU = "numba_cpu"    # Multi-threaded CPU with JIT
    NUMPY_MKL = "numpy_mkl"    # Intel MKL-optimized NumPy
    NUMPY_BASIC = "numpy_basic"  # Basic NumPy fallback


# Detect available backends
_PYOPENCL_AVAILABLE = False
_PYOPENCL_DEVICE = None
_PYOPENCL_CONTEXT = None
_PYOPENCL_QUEUE = None

try:
    import pyopencl as cl
    import pyopencl.array as cl_array

    # Try to get AMD or Intel GPU
    platforms = cl.get_platforms()
    for platform in platforms:
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        if devices:
            _PYOPENCL_DEVICE = devices[0]
            _PYOPENCL_CONTEXT = cl.Context([_PYOPENCL_DEVICE])
            _PYOPENCL_QUEUE = cl.CommandQueue(_PYOPENCL_CONTEXT)
            _PYOPENCL_AVAILABLE = True
            logger.info(f"[GPU] PyOpenCL initialized with: {_PYOPENCL_DEVICE.name}")
            break

    if not _PYOPENCL_AVAILABLE:
        # Try CPU as fallback
        for platform in platforms:
            devices = platform.get_devices(device_type=cl.device_type.CPU)
            if devices:
                _PYOPENCL_DEVICE = devices[0]
                _PYOPENCL_CONTEXT = cl.Context([_PYOPENCL_DEVICE])
                _PYOPENCL_QUEUE = cl.CommandQueue(_PYOPENCL_CONTEXT)
                _PYOPENCL_AVAILABLE = True
                logger.info(f"[OpenCL] Using CPU device: {_PYOPENCL_DEVICE.name}")
                break

except ImportError:
    logger.info("[GPU] PyOpenCL not available - install with: pip install pyopencl")
except Exception as e:
    logger.warning(f"[GPU] PyOpenCL initialization failed: {e}")

# Numba detection
_NUMBA_AVAILABLE = False
_NUMBA_CUDA_AVAILABLE = False

try:
    from numba import jit, prange, set_num_threads
    import numba
    _NUMBA_AVAILABLE = True

    # Set thread count for parallel operations
    n_threads = multiprocessing.cpu_count()
    set_num_threads(n_threads)
    logger.info(f"[CPU] Numba JIT enabled with {n_threads} threads")

    # Check for CUDA
    try:
        from numba import cuda
        if cuda.is_available():
            _NUMBA_CUDA_AVAILABLE = True
            logger.info(f"[GPU] Numba CUDA available: {cuda.get_current_device().name}")
    except Exception:
        pass

except ImportError:
    logger.info("[CPU] Numba not available - install with: pip install numba")

# Intel MKL detection (suppress stdout from np.show_config)
_MKL_AVAILABLE = False
try:
    import numpy as np
    import io
    import sys
    # Check if NumPy is linked to MKL without printing to stdout
    np_config = np.__config__
    if hasattr(np_config, 'get_info'):
        # NumPy >= 1.24 has get_info() which returns dict
        try:
            info = np_config.get_info()
            if isinstance(info, dict):
                config_str = str(info).lower()
                if 'mkl' in config_str:
                    _MKL_AVAILABLE = True
                    logger.info("[CPU] Intel MKL detected for NumPy")
        except Exception:
            pass
    elif hasattr(np_config, 'show'):
        # Fallback: capture stdout to avoid spam
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            np_config.show()
            config_str = sys.stdout.getvalue().lower()
            if 'mkl' in config_str:
                _MKL_AVAILABLE = True
                logger.info("[CPU] Intel MKL detected for NumPy")
        finally:
            sys.stdout = old_stdout
except Exception:
    pass


def get_best_backend() -> AccelerationBackend:
    """Get the best available acceleration backend."""
    if _PYOPENCL_AVAILABLE:
        return AccelerationBackend.PYOPENCL
    if _NUMBA_CUDA_AVAILABLE:
        return AccelerationBackend.NUMBA_CUDA
    if _NUMBA_AVAILABLE:
        return AccelerationBackend.NUMBA_CPU
    if _MKL_AVAILABLE:
        return AccelerationBackend.NUMPY_MKL
    return AccelerationBackend.NUMPY_BASIC


def get_acceleration_info() -> Dict[str, Any]:
    """Get detailed information about available acceleration."""
    info = {
        'best_backend': get_best_backend().value,
        'pyopencl_available': _PYOPENCL_AVAILABLE,
        'pyopencl_device': _PYOPENCL_DEVICE.name if _PYOPENCL_DEVICE else None,
        'numba_available': _NUMBA_AVAILABLE,
        'numba_cuda_available': _NUMBA_CUDA_AVAILABLE,
        'mkl_available': _MKL_AVAILABLE,
        'cpu_cores': multiprocessing.cpu_count(),
    }

    if _PYOPENCL_AVAILABLE and _PYOPENCL_DEVICE:
        info['gpu_compute_units'] = _PYOPENCL_DEVICE.max_compute_units
        info['gpu_global_mem_mb'] = _PYOPENCL_DEVICE.global_mem_size // (1024 * 1024)
        info['gpu_max_work_group'] = _PYOPENCL_DEVICE.max_work_group_size

    return info


# =============================================================================
# OPENCL KERNELS FOR GPU ACCELERATION
# =============================================================================

if _PYOPENCL_AVAILABLE:
    # OpenCL kernel source code for GPU computations
    _OPENCL_KERNELS = """
    // Correlation matrix kernel - computes full correlation matrix in parallel
    __kernel void correlation_matrix(
        __global const float* prices,    // [n_obs x n_assets] price matrix
        __global float* correlations,    // [n_assets x n_assets] output
        const int n_obs,
        const int n_assets
    ) {
        int i = get_global_id(0);
        int j = get_global_id(1);

        if (i >= n_assets || j >= n_assets) return;

        // Compute correlation between asset i and j
        float sum_i = 0.0f, sum_j = 0.0f;
        float sum_ii = 0.0f, sum_jj = 0.0f, sum_ij = 0.0f;

        for (int t = 0; t < n_obs; t++) {
            float xi = prices[t * n_assets + i];
            float xj = prices[t * n_assets + j];
            sum_i += xi;
            sum_j += xj;
            sum_ii += xi * xi;
            sum_jj += xj * xj;
            sum_ij += xi * xj;
        }

        float n = (float)n_obs;
        float mean_i = sum_i / n;
        float mean_j = sum_j / n;
        float var_i = sum_ii / n - mean_i * mean_i;
        float var_j = sum_jj / n - mean_j * mean_j;
        float cov_ij = sum_ij / n - mean_i * mean_j;

        float std_i = sqrt(var_i);
        float std_j = sqrt(var_j);

        float corr = (std_i > 1e-10f && std_j > 1e-10f) ?
                     cov_ij / (std_i * std_j) : 0.0f;

        correlations[i * n_assets + j] = corr;
    }

    // Batch OLS regression kernel - computes hedge ratios for multiple pairs
    __kernel void batch_ols_regression(
        __global const float* price_a,   // [n_pairs x n_obs] prices of asset A
        __global const float* price_b,   // [n_pairs x n_obs] prices of asset B
        __global float* alpha,           // [n_pairs] intercepts
        __global float* beta,            // [n_pairs] slopes (hedge ratios)
        __global float* residuals,       // [n_pairs x n_obs] residuals
        const int n_pairs,
        const int n_obs
    ) {
        int pair_idx = get_global_id(0);
        if (pair_idx >= n_pairs) return;

        // Compute OLS: y = alpha + beta * x
        // beta = Cov(x,y) / Var(x)
        // alpha = mean(y) - beta * mean(x)

        float sum_x = 0.0f, sum_y = 0.0f;
        float sum_xx = 0.0f, sum_xy = 0.0f;

        int offset = pair_idx * n_obs;

        for (int t = 0; t < n_obs; t++) {
            float x = price_b[offset + t];
            float y = price_a[offset + t];
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        float n = (float)n_obs;
        float mean_x = sum_x / n;
        float mean_y = sum_y / n;
        float var_x = sum_xx / n - mean_x * mean_x;
        float cov_xy = sum_xy / n - mean_x * mean_y;

        float b = (var_x > 1e-10f) ? cov_xy / var_x : 0.0f;
        float a = mean_y - b * mean_x;

        alpha[pair_idx] = a;
        beta[pair_idx] = b;

        // Compute residuals
        for (int t = 0; t < n_obs; t++) {
            float x = price_b[offset + t];
            float y = price_a[offset + t];
            residuals[offset + t] = y - a - b * x;
        }
    }

    // Rolling statistics kernel - computes mean and std for z-scores
    __kernel void rolling_statistics(
        __global const float* spread,    // [n_obs] spread series
        __global float* rolling_mean,    // [n_obs] output rolling mean
        __global float* rolling_std,     // [n_obs] output rolling std
        const int n_obs,
        const int lookback
    ) {
        int t = get_global_id(0);
        if (t < lookback || t >= n_obs) return;

        float sum = 0.0f;
        float sum_sq = 0.0f;

        for (int i = t - lookback; i < t; i++) {
            float val = spread[i];
            sum += val;
            sum_sq += val * val;
        }

        float mean = sum / (float)lookback;
        float var = sum_sq / (float)lookback - mean * mean;
        float std = sqrt(max(var, 1e-10f));

        rolling_mean[t] = mean;
        rolling_std[t] = std;
    }

    // ADF test statistic kernel - for cointegration testing
    __kernel void adf_test_batch(
        __global const float* residuals,  // [n_pairs x n_obs] residuals
        __global float* adf_stats,        // [n_pairs] ADF statistics
        const int n_pairs,
        const int n_obs
    ) {
        int pair_idx = get_global_id(0);
        if (pair_idx >= n_pairs) return;

        int offset = pair_idx * n_obs;

        // Compute first difference and lag
        float sum_xy = 0.0f;
        float sum_xx = 0.0f;
        float sum_yy = 0.0f;
        int n = n_obs - 1;

        for (int t = 1; t < n_obs; t++) {
            float y = residuals[offset + t] - residuals[offset + t - 1];  // diff
            float x = residuals[offset + t - 1];  // lag
            sum_xy += x * y;
            sum_xx += x * x;
            sum_yy += y * y;
        }

        // OLS coefficient
        float beta = (sum_xx > 1e-10f) ? sum_xy / sum_xx : 0.0f;

        // Compute SSE for standard error
        float sse = 0.0f;
        for (int t = 1; t < n_obs; t++) {
            float y = residuals[offset + t] - residuals[offset + t - 1];
            float x = residuals[offset + t - 1];
            float resid = y - beta * x;
            sse += resid * resid;
        }

        float mse = sse / (float)(n - 1);
        float se_beta = sqrt(mse / max(sum_xx, 1e-10f));

        adf_stats[pair_idx] = (se_beta > 1e-10f) ? beta / se_beta : 0.0f;
    }

    // Half-life calculation kernel
    __kernel void half_life_batch(
        __global const float* residuals,  // [n_pairs x n_obs] residuals
        __global float* half_lives,       // [n_pairs] half-lives
        const int n_pairs,
        const int n_obs
    ) {
        int pair_idx = get_global_id(0);
        if (pair_idx >= n_pairs) return;

        int offset = pair_idx * n_obs;

        // AR(1) regression: delta = theta * lag
        float sum_xy = 0.0f;
        float sum_xx = 0.0f;

        for (int t = 1; t < n_obs; t++) {
            float y = residuals[offset + t] - residuals[offset + t - 1];  // diff
            float x = residuals[offset + t - 1];  // lag
            sum_xy += x * y;
            sum_xx += x * x;
        }

        float theta = (sum_xx > 1e-10f) ? sum_xy / sum_xx : 0.0f;

        // Half-life = -ln(2) / ln(1 + theta)
        float half_life;
        if (theta >= 0.0f) {
            half_life = 100.0f;  // No mean reversion
        } else {
            half_life = -log(2.0f) / log(1.0f + theta);
            half_life = clamp(half_life, 1.0f, 1000.0f);
        }

        half_lives[pair_idx] = half_life;
    }
    """

    # Compile kernels
    try:
        _OPENCL_PROGRAM = cl.Program(_PYOPENCL_CONTEXT, _OPENCL_KERNELS).build()
        logger.info("[GPU] OpenCL kernels compiled successfully")
    except Exception as e:
        logger.error(f"[GPU] OpenCL kernel compilation failed: {e}")
        _PYOPENCL_AVAILABLE = False


# =============================================================================
# NUMBA JIT-COMPILED FUNCTIONS (CPU FALLBACK)
# =============================================================================

if _NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def _numba_correlation_matrix(prices: np.ndarray) -> np.ndarray:
        """JIT-compiled parallel correlation matrix calculation."""
        n_assets = prices.shape[1]
        n_obs = prices.shape[0]
        corr = np.empty((n_assets, n_assets), dtype=np.float64)

        # Compute means and stds
        means = np.empty(n_assets, dtype=np.float64)
        stds = np.empty(n_assets, dtype=np.float64)
        for i in prange(n_assets):
            means[i] = np.mean(prices[:, i])
            stds[i] = np.std(prices[:, i])

        # Compute correlations in parallel
        for i in prange(n_assets):
            for j in range(i, n_assets):
                if stds[i] > 1e-10 and stds[j] > 1e-10:
                    cov = np.mean((prices[:, i] - means[i]) * (prices[:, j] - means[j]))
                    corr[i, j] = cov / (stds[i] * stds[j])
                    corr[j, i] = corr[i, j]
                else:
                    corr[i, j] = 0.0
                    corr[j, i] = 0.0
        return corr

    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def _numba_batch_ols(prices_a: np.ndarray, prices_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        JIT-compiled batch OLS regression for multiple pairs.

        Args:
            prices_a: [n_pairs x n_obs] prices of asset A
            prices_b: [n_pairs x n_obs] prices of asset B

        Returns:
            alphas: [n_pairs] intercepts
            betas: [n_pairs] slopes (hedge ratios)
            residuals: [n_pairs x n_obs] residuals
        """
        n_pairs = prices_a.shape[0]
        n_obs = prices_a.shape[1]

        alphas = np.empty(n_pairs, dtype=np.float64)
        betas = np.empty(n_pairs, dtype=np.float64)
        residuals = np.empty((n_pairs, n_obs), dtype=np.float64)

        for pair_idx in prange(n_pairs):
            sum_x = 0.0
            sum_y = 0.0
            sum_xx = 0.0
            sum_xy = 0.0

            for t in range(n_obs):
                x = prices_b[pair_idx, t]
                y = prices_a[pair_idx, t]
                sum_x += x
                sum_y += y
                sum_xx += x * x
                sum_xy += x * y

            n = float(n_obs)
            mean_x = sum_x / n
            mean_y = sum_y / n
            var_x = sum_xx / n - mean_x * mean_x
            cov_xy = sum_xy / n - mean_x * mean_y

            beta = cov_xy / var_x if var_x > 1e-10 else 0.0
            alpha = mean_y - beta * mean_x

            alphas[pair_idx] = alpha
            betas[pair_idx] = beta

            # Compute residuals
            for t in range(n_obs):
                residuals[pair_idx, t] = prices_a[pair_idx, t] - alpha - beta * prices_b[pair_idx, t]

        return alphas, betas, residuals

    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def _numba_batch_adf(residuals: np.ndarray) -> np.ndarray:
        """
        JIT-compiled batch ADF test for multiple pairs.

        Args:
            residuals: [n_pairs x n_obs] residuals from OLS

        Returns:
            adf_stats: [n_pairs] ADF test statistics
        """
        n_pairs = residuals.shape[0]
        n_obs = residuals.shape[1]
        adf_stats = np.empty(n_pairs, dtype=np.float64)

        for pair_idx in prange(n_pairs):
            sum_xy = 0.0
            sum_xx = 0.0

            for t in range(1, n_obs):
                y = residuals[pair_idx, t] - residuals[pair_idx, t - 1]  # diff
                x = residuals[pair_idx, t - 1]  # lag
                sum_xy += x * y
                sum_xx += x * x

            beta = sum_xy / sum_xx if sum_xx > 1e-10 else 0.0

            # Compute SSE
            sse = 0.0
            for t in range(1, n_obs):
                y = residuals[pair_idx, t] - residuals[pair_idx, t - 1]
                x = residuals[pair_idx, t - 1]
                resid = y - beta * x
                sse += resid * resid

            n = n_obs - 1
            mse = sse / (n - 1) if n > 1 else 1.0
            se_beta = np.sqrt(mse / sum_xx) if sum_xx > 0 else 1.0

            adf_stats[pair_idx] = beta / se_beta if se_beta > 1e-10 else 0.0

        return adf_stats

    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def _numba_batch_half_life(residuals: np.ndarray) -> np.ndarray:
        """
        JIT-compiled batch half-life calculation.

        Args:
            residuals: [n_pairs x n_obs] residuals

        Returns:
            half_lives: [n_pairs] half-lives in periods
        """
        n_pairs = residuals.shape[0]
        n_obs = residuals.shape[1]
        half_lives = np.empty(n_pairs, dtype=np.float64)

        for pair_idx in prange(n_pairs):
            sum_xy = 0.0
            sum_xx = 0.0

            for t in range(1, n_obs):
                y = residuals[pair_idx, t] - residuals[pair_idx, t - 1]
                x = residuals[pair_idx, t - 1]
                sum_xy += x * y
                sum_xx += x * x

            theta = sum_xy / sum_xx if sum_xx > 1e-10 else 0.0

            if theta >= 0.0:
                half_lives[pair_idx] = 100.0  # No mean reversion
            else:
                half_life = -np.log(2.0) / np.log(1.0 + theta)
                half_lives[pair_idx] = max(1.0, min(half_life, 1000.0))

        return half_lives

    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def _numba_rolling_zscore(spread: np.ndarray, lookback: int) -> np.ndarray:
        """
        JIT-compiled rolling z-score calculation.

        Args:
            spread: [n_obs] spread series
            lookback: rolling window size

        Returns:
            z_scores: [n_obs] z-scores (NaN for first lookback observations)
        """
        n_obs = len(spread)
        z_scores = np.empty(n_obs, dtype=np.float64)
        z_scores[:lookback] = np.nan

        for t in prange(lookback, n_obs):
            window = spread[t - lookback:t]
            mean = np.mean(window)
            std = np.std(window)
            if std > 1e-10:
                z_scores[t] = (spread[t] - mean) / std
            else:
                z_scores[t] = 0.0

        return z_scores

    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def _numba_batch_hurst(residuals: np.ndarray, max_lag: int = 20) -> np.ndarray:
        """
        JIT-compiled batch Hurst exponent calculation using R/S method.

        The Hurst exponent indicates:
        - H < 0.5: Mean-reverting (tradeable for pairs)
        - H = 0.5: Random walk
        - H > 0.5: Trending

        Args:
            residuals: [n_pairs x n_obs] spread residuals
            max_lag: Maximum lag for R/S calculation

        Returns:
            hurst: [n_pairs] Hurst exponents
        """
        n_pairs = residuals.shape[0]
        n_obs = residuals.shape[1]
        hurst = np.empty(n_pairs, dtype=np.float64)

        for pair_idx in prange(n_pairs):
            # Use simplified variance ratio method for speed
            # Hurst ≈ 0.5 + 0.5 * log(Var(k)/Var(1)) / log(k)
            spread = residuals[pair_idx, :]

            # Calculate returns
            returns = np.empty(n_obs - 1, dtype=np.float64)
            for i in range(n_obs - 1):
                returns[i] = spread[i + 1] - spread[i]

            # Variance at lag 1
            var_1 = np.var(returns)
            if var_1 < 1e-10:
                hurst[pair_idx] = 0.5
                continue

            # Calculate variance ratio for multiple lags
            sum_log_ratio = 0.0
            sum_log_k = 0.0
            count = 0

            for k in range(2, min(max_lag + 1, n_obs // 4)):
                # Aggregate returns at lag k
                n_agg = (n_obs - 1) // k
                if n_agg < 3:
                    continue

                var_k_sum = 0.0
                for i in range(n_agg):
                    agg_return = 0.0
                    for j in range(k):
                        if i * k + j < n_obs - 1:
                            agg_return += returns[i * k + j]
                    var_k_sum += agg_return * agg_return

                var_k = var_k_sum / n_agg - (np.sum(returns[:n_agg * k]) / n_agg) ** 2
                if var_k > 1e-10 and var_1 > 1e-10:
                    # Hurst relationship: Var(k) ∝ k^(2H)
                    log_ratio = np.log(var_k / var_1)
                    log_k = np.log(float(k))
                    sum_log_ratio += log_ratio
                    sum_log_k += log_k
                    count += 1

            if count > 0:
                # H = log(Var(k)/Var(1)) / (2 * log(k))
                h = 0.5 * sum_log_ratio / sum_log_k
                hurst[pair_idx] = max(0.0, min(1.0, h))
            else:
                hurst[pair_idx] = 0.5

        return hurst

    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def _numba_batch_spread_stats(residuals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        JIT-compiled batch spread statistics calculation.

        Args:
            residuals: [n_pairs x n_obs] spread residuals

        Returns:
            spread_mean: [n_pairs] mean of spread
            spread_std: [n_pairs] standard deviation of spread
            spread_zscore_current: [n_pairs] current z-score (last observation)
        """
        n_pairs = residuals.shape[0]
        n_obs = residuals.shape[1]

        spread_mean = np.empty(n_pairs, dtype=np.float64)
        spread_std = np.empty(n_pairs, dtype=np.float64)
        spread_zscore = np.empty(n_pairs, dtype=np.float64)

        for pair_idx in prange(n_pairs):
            mean_val = np.mean(residuals[pair_idx, :])
            std_val = np.std(residuals[pair_idx, :])

            spread_mean[pair_idx] = mean_val
            spread_std[pair_idx] = std_val

            if std_val > 1e-10:
                spread_zscore[pair_idx] = (residuals[pair_idx, -1] - mean_val) / std_val
            else:
                spread_zscore[pair_idx] = 0.0

        return spread_mean, spread_std, spread_zscore

    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def _numba_batch_r_squared(prices_a: np.ndarray, prices_b: np.ndarray,
                                alphas: np.ndarray, betas: np.ndarray) -> np.ndarray:
        """
        JIT-compiled batch R-squared calculation for regression quality.

        Args:
            prices_a: [n_pairs x n_obs] prices of asset A (dependent)
            prices_b: [n_pairs x n_obs] prices of asset B (independent)
            alphas: [n_pairs] intercepts from OLS
            betas: [n_pairs] slopes from OLS

        Returns:
            r_squared: [n_pairs] R² values
        """
        n_pairs = prices_a.shape[0]
        n_obs = prices_a.shape[1]
        r_squared = np.empty(n_pairs, dtype=np.float64)

        for pair_idx in prange(n_pairs):
            # Total sum of squares
            mean_y = np.mean(prices_a[pair_idx, :])
            ss_tot = 0.0
            ss_res = 0.0

            alpha = alphas[pair_idx]
            beta = betas[pair_idx]

            for t in range(n_obs):
                y = prices_a[pair_idx, t]
                x = prices_b[pair_idx, t]
                y_pred = alpha + beta * x

                ss_tot += (y - mean_y) ** 2
                ss_res += (y - y_pred) ** 2

            if ss_tot > 1e-10:
                r_squared[pair_idx] = 1.0 - (ss_res / ss_tot)
            else:
                r_squared[pair_idx] = 0.0

        return r_squared


# =============================================================================
# GPU-ACCELERATED FUNCTIONS (MAIN API)
# =============================================================================

@dataclass
class AcceleratedResult:
    """Result container with timing information."""
    data: Any
    compute_time_ms: float
    backend_used: AccelerationBackend

    def __repr__(self):
        return f"AcceleratedResult(backend={self.backend_used.value}, time={self.compute_time_ms:.2f}ms)"


class GPUAccelerator:
    """
    GPU-accelerated computation engine for pairs trading.

    Automatically selects the best available backend (PyOpenCL GPU,
    Numba JIT, or NumPy) and provides unified API for accelerated
    computations.
    """

    def __init__(self, prefer_gpu: bool = True):
        """
        Initialize the GPU accelerator.

        Args:
            prefer_gpu: If True, prefer GPU backends when available
        """
        self.prefer_gpu = prefer_gpu
        self.backend = get_best_backend()
        self.info = get_acceleration_info()

        logger.info(f"[Accelerator] Initialized with backend: {self.backend.value}")

    def correlation_matrix(self, prices: np.ndarray) -> AcceleratedResult:
        """
        Compute correlation matrix using best available backend.

        Args:
            prices: [n_obs x n_assets] price matrix

        Returns:
            AcceleratedResult with [n_assets x n_assets] correlation matrix
        """
        start = time.perf_counter()

        # Ensure float32 for GPU
        prices = np.ascontiguousarray(prices, dtype=np.float32)
        n_obs, n_assets = prices.shape

        if self.backend == AccelerationBackend.PYOPENCL and _PYOPENCL_AVAILABLE:
            # GPU acceleration via OpenCL
            prices_gpu = cl_array.to_device(_PYOPENCL_QUEUE, prices)
            corr_gpu = cl_array.empty(_PYOPENCL_QUEUE, (n_assets, n_assets), dtype=np.float32)

            _OPENCL_PROGRAM.correlation_matrix(
                _PYOPENCL_QUEUE,
                (n_assets, n_assets),  # global size
                None,  # local size (auto)
                prices_gpu.data,
                corr_gpu.data,
                np.int32(n_obs),
                np.int32(n_assets)
            )

            result = corr_gpu.get()

        elif _NUMBA_AVAILABLE:
            # CPU acceleration via Numba JIT
            result = _numba_correlation_matrix(prices.astype(np.float64))

        else:
            # NumPy fallback
            result = np.corrcoef(prices.T)

        elapsed_ms = (time.perf_counter() - start) * 1000
        return AcceleratedResult(result, elapsed_ms, self.backend)

    def batch_cointegration_test(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray
    ) -> AcceleratedResult:
        """
        Batch cointegration test for multiple pairs.

        This is the main acceleration target - testing O(n²) pairs
        in parallel on GPU vs sequential on CPU.

        Args:
            prices_a: [n_pairs x n_obs] log prices of asset A
            prices_b: [n_pairs x n_obs] log prices of asset B

        Returns:
            AcceleratedResult with dict containing:
                - alphas: [n_pairs] intercepts
                - betas: [n_pairs] hedge ratios
                - residuals: [n_pairs x n_obs] spread residuals
                - adf_stats: [n_pairs] ADF test statistics
                - half_lives: [n_pairs] mean reversion half-lives
        """
        start = time.perf_counter()

        n_pairs, n_obs = prices_a.shape

        if self.backend == AccelerationBackend.PYOPENCL and _PYOPENCL_AVAILABLE:
            # GPU-accelerated batch processing
            prices_a_f32 = np.ascontiguousarray(prices_a, dtype=np.float32)
            prices_b_f32 = np.ascontiguousarray(prices_b, dtype=np.float32)

            # Transfer to GPU
            pa_gpu = cl_array.to_device(_PYOPENCL_QUEUE, prices_a_f32)
            pb_gpu = cl_array.to_device(_PYOPENCL_QUEUE, prices_b_f32)

            # Allocate outputs
            alpha_gpu = cl_array.empty(_PYOPENCL_QUEUE, n_pairs, dtype=np.float32)
            beta_gpu = cl_array.empty(_PYOPENCL_QUEUE, n_pairs, dtype=np.float32)
            residuals_gpu = cl_array.empty(_PYOPENCL_QUEUE, (n_pairs, n_obs), dtype=np.float32)
            adf_gpu = cl_array.empty(_PYOPENCL_QUEUE, n_pairs, dtype=np.float32)
            hl_gpu = cl_array.empty(_PYOPENCL_QUEUE, n_pairs, dtype=np.float32)

            # Run OLS kernel
            _OPENCL_PROGRAM.batch_ols_regression(
                _PYOPENCL_QUEUE,
                (n_pairs,),
                None,
                pa_gpu.data,
                pb_gpu.data,
                alpha_gpu.data,
                beta_gpu.data,
                residuals_gpu.data,
                np.int32(n_pairs),
                np.int32(n_obs)
            )

            # Run ADF kernel
            _OPENCL_PROGRAM.adf_test_batch(
                _PYOPENCL_QUEUE,
                (n_pairs,),
                None,
                residuals_gpu.data,
                adf_gpu.data,
                np.int32(n_pairs),
                np.int32(n_obs)
            )

            # Run half-life kernel
            _OPENCL_PROGRAM.half_life_batch(
                _PYOPENCL_QUEUE,
                (n_pairs,),
                None,
                residuals_gpu.data,
                hl_gpu.data,
                np.int32(n_pairs),
                np.int32(n_obs)
            )

            # Get results from GPU
            alphas = alpha_gpu.get()
            betas = beta_gpu.get()
            residuals = residuals_gpu.get()
            adf_stats = adf_gpu.get()
            half_lives = hl_gpu.get()

            # Calculate additional metrics using Numba (not available as OpenCL kernels)
            # Convert to float64 for Numba compatibility
            residuals_f64 = residuals.astype(np.float64)
            if _NUMBA_AVAILABLE:
                hurst_exponents = _numba_batch_hurst(residuals_f64)
                spread_mean, spread_std, spread_zscore = _numba_batch_spread_stats(residuals_f64)
                prices_a_f64 = prices_a.astype(np.float64)
                prices_b_f64 = prices_b.astype(np.float64)
                alphas_f64 = alphas.astype(np.float64)
                betas_f64 = betas.astype(np.float64)
                r_squared = _numba_batch_r_squared(prices_a_f64, prices_b_f64, alphas_f64, betas_f64)
            else:
                hurst_exponents = np.full(n_pairs, 0.5)
                spread_mean = np.array([np.mean(residuals[i]) for i in range(n_pairs)])
                spread_std = np.array([np.std(residuals[i]) for i in range(n_pairs)])
                spread_zscore = np.zeros(n_pairs)
                r_squared = np.zeros(n_pairs)

            result = {
                'alphas': alphas,
                'betas': betas,
                'residuals': residuals,
                'adf_stats': adf_stats,
                'half_lives': half_lives,
                'hurst_exponents': hurst_exponents,
                'spread_mean': spread_mean,
                'spread_std': spread_std,
                'spread_zscore': spread_zscore,
                'r_squared': r_squared,
            }

        elif _NUMBA_AVAILABLE:
            # CPU batch processing with Numba JIT
            prices_a_f64 = np.ascontiguousarray(prices_a, dtype=np.float64)
            prices_b_f64 = np.ascontiguousarray(prices_b, dtype=np.float64)

            alphas, betas, residuals = _numba_batch_ols(prices_a_f64, prices_b_f64)
            adf_stats = _numba_batch_adf(residuals)
            half_lives = _numba_batch_half_life(residuals)

            # Additional metrics for PDF compliance
            hurst_exponents = _numba_batch_hurst(residuals)
            spread_mean, spread_std, spread_zscore = _numba_batch_spread_stats(residuals)
            r_squared = _numba_batch_r_squared(prices_a_f64, prices_b_f64, alphas, betas)

            result = {
                'alphas': alphas,
                'betas': betas,
                'residuals': residuals,
                'adf_stats': adf_stats,
                'half_lives': half_lives,
                'hurst_exponents': hurst_exponents,
                'spread_mean': spread_mean,
                'spread_std': spread_std,
                'spread_zscore': spread_zscore,
                'r_squared': r_squared,
            }

        else:
            # NumPy fallback (sequential)
            alphas = np.empty(n_pairs)
            betas = np.empty(n_pairs)
            residuals = np.empty((n_pairs, n_obs))
            adf_stats = np.empty(n_pairs)
            half_lives = np.empty(n_pairs)

            for i in range(n_pairs):
                # OLS
                x = prices_b[i]
                y = prices_a[i]
                beta = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) > 1e-10 else 0.0
                alpha = np.mean(y) - beta * np.mean(x)

                alphas[i] = alpha
                betas[i] = beta
                residuals[i] = y - alpha - beta * x

                # Simple ADF approximation
                diff = np.diff(residuals[i])
                lag = residuals[i, :-1]
                if np.var(lag) > 1e-10:
                    adf_beta = np.cov(lag, diff)[0, 1] / np.var(lag)
                    adf_stats[i] = adf_beta / (np.std(diff) / np.sqrt(np.var(lag)))
                else:
                    adf_stats[i] = 0.0

                # Half-life
                if np.var(lag) > 1e-10:
                    theta = np.cov(lag, diff)[0, 1] / np.var(lag)
                    if theta < 0:
                        half_lives[i] = max(1.0, min(-np.log(2) / np.log(1 + theta), 1000.0))
                    else:
                        half_lives[i] = 100.0
                else:
                    half_lives[i] = 100.0

            # Additional metrics for NumPy fallback
            hurst_exponents = np.full(n_pairs, 0.5)  # Default to random walk
            spread_mean = np.array([np.mean(residuals[i]) for i in range(n_pairs)])
            spread_std = np.array([np.std(residuals[i]) for i in range(n_pairs)])
            spread_zscore = np.array([
                (residuals[i, -1] - spread_mean[i]) / spread_std[i]
                if spread_std[i] > 1e-10 else 0.0
                for i in range(n_pairs)
            ])
            r_squared = np.zeros(n_pairs)
            for i in range(n_pairs):
                y = prices_a[i]
                y_pred = alphas[i] + betas[i] * prices_b[i]
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                ss_res = np.sum((y - y_pred) ** 2)
                r_squared[i] = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

            result = {
                'alphas': alphas,
                'betas': betas,
                'residuals': residuals,
                'adf_stats': adf_stats,
                'half_lives': half_lives,
                'hurst_exponents': hurst_exponents,
                'spread_mean': spread_mean,
                'spread_std': spread_std,
                'spread_zscore': spread_zscore,
                'r_squared': r_squared,
            }

        elapsed_ms = (time.perf_counter() - start) * 1000
        return AcceleratedResult(result, elapsed_ms, self.backend)

    def rolling_zscore(self, spread: np.ndarray, lookback: int = 20) -> AcceleratedResult:
        """
        Calculate rolling z-score for spread series.

        Args:
            spread: [n_obs] spread series
            lookback: rolling window size

        Returns:
            AcceleratedResult with [n_obs] z-scores
        """
        start = time.perf_counter()

        spread = np.ascontiguousarray(spread, dtype=np.float64)

        if _NUMBA_AVAILABLE:
            result = _numba_rolling_zscore(spread, lookback)
        else:
            # NumPy fallback with pandas
            spread_series = pd.Series(spread)
            rolling_mean = spread_series.rolling(lookback, min_periods=lookback).mean()
            rolling_std = spread_series.rolling(lookback, min_periods=lookback).std()
            result = ((spread_series - rolling_mean) / rolling_std).values

        elapsed_ms = (time.perf_counter() - start) * 1000
        return AcceleratedResult(result, elapsed_ms, self.backend)

    def batch_rolling_zscore(
        self,
        spreads: np.ndarray,
        lookback: int = 20
    ) -> AcceleratedResult:
        """
        Batch rolling z-score calculation for multiple spreads.

        Args:
            spreads: [n_pairs x n_obs] spread series
            lookback: rolling window size

        Returns:
            AcceleratedResult with [n_pairs x n_obs] z-scores
        """
        start = time.perf_counter()

        n_pairs, n_obs = spreads.shape
        z_scores = np.empty_like(spreads)

        if _NUMBA_AVAILABLE:
            for i in range(n_pairs):
                z_scores[i] = _numba_rolling_zscore(
                    np.ascontiguousarray(spreads[i], dtype=np.float64),
                    lookback
                )
        else:
            for i in range(n_pairs):
                spread_series = pd.Series(spreads[i])
                rolling_mean = spread_series.rolling(lookback, min_periods=lookback).mean()
                rolling_std = spread_series.rolling(lookback, min_periods=lookback).std()
                z_scores[i] = ((spread_series - rolling_mean) / rolling_std).values

        elapsed_ms = (time.perf_counter() - start) * 1000
        return AcceleratedResult(z_scores, elapsed_ms, self.backend)


# =============================================================================
# BATCH COINTEGRATION TESTING (HIGH-LEVEL API)
# =============================================================================

@dataclass
class BatchCointegrationResult:
    """Result of batch cointegration testing."""
    pair_indices: List[Tuple[int, int]]
    alphas: np.ndarray
    betas: np.ndarray
    adf_stats: np.ndarray
    half_lives: np.ndarray
    p_values: np.ndarray
    is_cointegrated: np.ndarray
    compute_time_ms: float
    backend: AccelerationBackend

    def get_cointegrated_pairs(self, p_threshold: float = 0.05) -> List[Tuple[int, int]]:
        """Get list of cointegrated pairs."""
        return [
            self.pair_indices[i]
            for i in range(len(self.pair_indices))
            if self.is_cointegrated[i]
        ]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame({
            'asset_a_idx': [p[0] for p in self.pair_indices],
            'asset_b_idx': [p[1] for p in self.pair_indices],
            'alpha': self.alphas,
            'beta': self.betas,
            'adf_stat': self.adf_stats,
            'p_value': self.p_values,
            'half_life': self.half_lives,
            'is_cointegrated': self.is_cointegrated
        })


def batch_test_cointegration(
    price_matrix: pd.DataFrame,
    pair_indices: Optional[List[Tuple[int, int]]] = None,
    p_threshold: float = 0.05,
    min_half_life: float = 1.0,
    max_half_life: float = 45.0,
    use_log_prices: bool = True
) -> BatchCointegrationResult:
    """
    Test cointegration for multiple pairs in batch using GPU acceleration.

    This is the main entry point for accelerated cointegration testing.
    Tests all pairs or specified pairs in parallel.

    Args:
        price_matrix: DataFrame with price columns for each asset
        pair_indices: List of (i, j) tuples specifying pairs to test.
                     If None, tests all unique pairs.
        p_threshold: P-value threshold for cointegration
        min_half_life: Minimum acceptable half-life
        max_half_life: Maximum acceptable half-life
        use_log_prices: If True, use log prices for cointegration test

    Returns:
        BatchCointegrationResult with all test results
    """
    accelerator = GPUAccelerator()

    # Get price matrix as numpy array
    prices = price_matrix.values
    n_obs, n_assets = prices.shape
    asset_names = price_matrix.columns.tolist()

    # Generate pair indices if not provided
    if pair_indices is None:
        pair_indices = [
            (i, j) for i in range(n_assets)
            for j in range(i + 1, n_assets)
        ]

    n_pairs = len(pair_indices)
    logger.info(f"[Batch] Testing {n_pairs} pairs with {accelerator.backend.value}")

    # Prepare price arrays for batch processing
    if use_log_prices:
        prices = np.log(np.maximum(prices, 1e-10))

    prices_a = np.empty((n_pairs, n_obs), dtype=np.float64)
    prices_b = np.empty((n_pairs, n_obs), dtype=np.float64)

    for idx, (i, j) in enumerate(pair_indices):
        prices_a[idx] = prices[:, i]
        prices_b[idx] = prices[:, j]

    # Run batch cointegration test
    result = accelerator.batch_cointegration_test(prices_a, prices_b)

    # Convert ADF stats to p-values (approximate using critical values)
    # MacKinnon critical values for n=100: 1%=-3.51, 5%=-2.89, 10%=-2.58
    adf_stats = result.data['adf_stats']
    p_values = np.where(
        adf_stats < -3.51, 0.01,
        np.where(
            adf_stats < -2.89, 0.05,
            np.where(adf_stats < -2.58, 0.10, 0.50)
        )
    )

    # Determine cointegration status
    half_lives = result.data['half_lives']
    is_cointegrated = (
        (p_values <= p_threshold) &
        (half_lives >= min_half_life) &
        (half_lives <= max_half_life) &
        (result.data['betas'] > 0)  # Positive hedge ratio
    )

    return BatchCointegrationResult(
        pair_indices=pair_indices,
        alphas=result.data['alphas'],
        betas=result.data['betas'],
        adf_stats=adf_stats,
        half_lives=half_lives,
        p_values=p_values,
        is_cointegrated=is_cointegrated,
        compute_time_ms=result.compute_time_ms,
        backend=result.backend_used
    )


# =============================================================================
# STANDALONE NUMBA JIT FUNCTIONS (for module-level import)
# These are simpler versions that can be used directly without GPUAccelerator
# =============================================================================

if _NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def fast_correlation_matrix(prices: np.ndarray) -> np.ndarray:
        """JIT-compiled parallel correlation matrix calculation."""
        n_assets = prices.shape[1]
        n_obs = prices.shape[0]
        corr = np.empty((n_assets, n_assets), dtype=np.float64)

        # Compute means and stds
        means = np.empty(n_assets, dtype=np.float64)
        stds = np.empty(n_assets, dtype=np.float64)
        for i in prange(n_assets):
            means[i] = np.mean(prices[:, i])
            stds[i] = np.std(prices[:, i])

        # Compute correlations in parallel
        for i in prange(n_assets):
            for j in range(i, n_assets):
                if stds[i] > 1e-10 and stds[j] > 1e-10:
                    cov = np.mean((prices[:, i] - means[i]) * (prices[:, j] - means[j]))
                    corr[i, j] = cov / (stds[i] * stds[j])
                    corr[j, i] = corr[i, j]
                else:
                    corr[i, j] = 0.0
                    corr[j, i] = 0.0
        return corr

    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def fast_ols_residuals(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """JIT-compiled OLS residuals for cointegration testing."""
        n = len(y)
        # Add constant (intercept)
        x_with_const = np.empty((n, 2), dtype=np.float64)
        x_with_const[:, 0] = 1.0
        x_with_const[:, 1] = x

        # OLS: beta = (X'X)^-1 X'y
        xtx = np.zeros((2, 2), dtype=np.float64)
        xty = np.zeros(2, dtype=np.float64)

        for i in prange(n):
            for j in range(2):
                xty[j] += x_with_const[i, j] * y[i]
                for k in range(2):
                    xtx[j, k] += x_with_const[i, j] * x_with_const[i, k]

        # 2x2 matrix inverse
        det = xtx[0, 0] * xtx[1, 1] - xtx[0, 1] * xtx[1, 0]
        if abs(det) < 1e-10:
            return y  # Return y if singular

        inv = np.empty((2, 2), dtype=np.float64)
        inv[0, 0] = xtx[1, 1] / det
        inv[1, 1] = xtx[0, 0] / det
        inv[0, 1] = -xtx[0, 1] / det
        inv[1, 0] = -xtx[1, 0] / det

        beta = np.zeros(2, dtype=np.float64)
        for j in range(2):
            for k in range(2):
                beta[j] += inv[j, k] * xty[k]

        # Compute residuals
        residuals = np.empty(n, dtype=np.float64)
        for i in prange(n):
            residuals[i] = y[i] - beta[0] - beta[1] * x[i]

        return residuals

    @jit(nopython=True, cache=True, fastmath=True)
    def fast_adf_statistic(residuals: np.ndarray, max_lags: int = 12) -> float:
        """JIT-compiled ADF test statistic for cointegration."""
        n = len(residuals)
        if n < max_lags + 2:
            return 0.0

        # First difference
        diff = np.empty(n - 1, dtype=np.float64)
        for i in range(n - 1):
            diff[i] = residuals[i + 1] - residuals[i]

        # Lag of residuals
        lag_resid = residuals[:-1]

        # Simple ADF without lags for speed
        y = diff[1:]
        x = lag_resid[:-1]

        n_reg = len(y)

        # OLS for ADF
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_xx = 0.0

        for i in range(n_reg):
            sum_x += x[i]
            sum_y += y[i]
            sum_xy += x[i] * y[i]
            sum_xx += x[i] * x[i]

        mean_x = sum_x / n_reg
        mean_y = sum_y / n_reg

        denom = sum_xx - n_reg * mean_x * mean_x
        if abs(denom) < 1e-10:
            return 0.0

        beta = (sum_xy - n_reg * mean_x * mean_y) / denom

        # Compute standard error
        sse = 0.0
        for i in range(n_reg):
            resid = y[i] - mean_y - beta * (x[i] - mean_x)
            sse += resid * resid

        mse = sse / (n_reg - 2) if n_reg > 2 else 1.0
        se_beta = np.sqrt(mse / denom) if denom > 0 else 1.0

        if se_beta < 1e-10:
            return 0.0

        return beta / se_beta

    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def fast_half_life(residuals: np.ndarray) -> float:
        """JIT-compiled half-life calculation."""
        n = len(residuals)
        if n < 3:
            return 100.0

        # Lag and difference
        y = np.empty(n - 1, dtype=np.float64)
        x = np.empty(n - 1, dtype=np.float64)
        for i in prange(n - 1):
            y[i] = residuals[i + 1] - residuals[i]
            x[i] = residuals[i]

        # Simple OLS for AR(1)
        n_reg = len(y)
        sum_xy = 0.0
        sum_xx = 0.0

        for i in range(n_reg):
            sum_xy += x[i] * y[i]
            sum_xx += x[i] * x[i]

        if sum_xx < 1e-10:
            return 100.0

        theta = sum_xy / sum_xx

        if theta >= 0:
            return 100.0  # No mean reversion

        half_life = -np.log(2) / np.log(1 + theta)
        return max(1.0, min(half_life, 1000.0))

else:
    # Fallback non-numba versions
    def fast_correlation_matrix(prices):
        """Fallback correlation matrix using numpy."""
        return np.corrcoef(prices.T)

    def fast_ols_residuals(y, x):
        """Fallback OLS residuals using scipy."""
        from scipy import stats
        slope, intercept, _, _, _ = stats.linregress(x, y)
        return y - intercept - slope * x

    def fast_adf_statistic(residuals, max_lags=12):
        """Fallback ADF - returns 0 (will use statsmodels)."""
        return 0.0

    def fast_half_life(residuals):
        """Fallback half-life - returns default."""
        return 100.0


# =============================================================================
# PERFORMANCE BENCHMARK
# =============================================================================

def benchmark_acceleration(n_assets: int = 50, n_obs: int = 500) -> Dict[str, Any]:
    """
    Benchmark GPU acceleration performance.

    Args:
        n_assets: Number of assets for correlation matrix test
        n_obs: Number of observations

    Returns:
        Dict with benchmark results
    """
    print(f"\n{'='*60}")
    print("GPU ACCELERATION BENCHMARK")
    print(f"{'='*60}")
    print(f"Configuration: {n_assets} assets, {n_obs} observations")
    print(f"Backend: {get_best_backend().value}")
    print()

    # Generate test data
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(n_obs, n_assets) * 0.02 + 0.0001, axis=0)
    prices = np.exp(prices) * 100  # Convert to price levels

    accelerator = GPUAccelerator()
    results = {}

    # Test 1: Correlation Matrix
    print("Test 1: Correlation Matrix")
    result = accelerator.correlation_matrix(prices)
    print(f"  Time: {result.compute_time_ms:.2f} ms")
    print(f"  Shape: {result.data.shape}")
    results['correlation_matrix'] = result.compute_time_ms

    # Test 2: Batch Cointegration (all pairs)
    n_pairs = n_assets * (n_assets - 1) // 2
    print(f"\nTest 2: Batch Cointegration ({n_pairs} pairs)")

    # Prepare batch data
    pair_indices = [(i, j) for i in range(n_assets) for j in range(i + 1, n_assets)]
    log_prices = np.log(np.maximum(prices, 1e-10))

    prices_a = np.empty((n_pairs, n_obs))
    prices_b = np.empty((n_pairs, n_obs))
    for idx, (i, j) in enumerate(pair_indices):
        prices_a[idx] = log_prices[:, i]
        prices_b[idx] = log_prices[:, j]

    result = accelerator.batch_cointegration_test(prices_a, prices_b)
    print(f"  Time: {result.compute_time_ms:.2f} ms")
    print(f"  Pairs/second: {n_pairs / (result.compute_time_ms / 1000):.0f}")
    results['batch_cointegration'] = result.compute_time_ms
    results['pairs_per_second'] = n_pairs / (result.compute_time_ms / 1000)

    # Test 3: Rolling Z-Score (single series)
    print("\nTest 3: Rolling Z-Score")
    spread = np.random.randn(n_obs)
    result = accelerator.rolling_zscore(spread, lookback=20)
    print(f"  Time: {result.compute_time_ms:.2f} ms")
    results['rolling_zscore'] = result.compute_time_ms

    print(f"\n{'='*60}")
    print("ACCELERATION INFO")
    print(f"{'='*60}")
    info = get_acceleration_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    return results


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    'GPUAccelerator',
    'AcceleratedResult',
    'BatchCointegrationResult',
    'AccelerationBackend',

    # Functions
    'batch_test_cointegration',
    'get_best_backend',
    'get_acceleration_info',
    'benchmark_acceleration',

    # Standalone Numba JIT functions (for direct import)
    'fast_correlation_matrix',
    'fast_ols_residuals',
    'fast_adf_statistic',
    'fast_half_life',

    # Backend detection flags
    '_PYOPENCL_AVAILABLE',
    '_NUMBA_AVAILABLE',
    '_NUMBA_CUDA_AVAILABLE',
    '_MKL_AVAILABLE',
]


if __name__ == "__main__":
    # Run benchmark when executed directly
    benchmark_acceleration()
