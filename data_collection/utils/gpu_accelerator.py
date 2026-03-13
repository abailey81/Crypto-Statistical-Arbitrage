"""
GPU Acceleration Module for Phase 1 Data Processing
====================================================

This module provides GPU-accelerated data processing using NVIDIA RAPIDS (cuDF/cuPy)
when available, with automatic fallback to CPU-based pandas/numpy operations.

Supported Operations:
1. DataFrame Operations - cuDF for large DataFrame manipulations
2. Numerical Computations - cuPy for array operations
3. Statistical Analysis - GPU-accelerated rolling windows, correlations
4. Parallel Processing - Multi-GPU support for large datasets

Requirements (Optional - falls back to CPU if not available):
- NVIDIA GPU with CUDA support
- cuDF (pip install cudf-cu11 or cudf-cu12)
- cuPy (pip install cupy-cuda11x or cupy-cuda12x)
- rmm (RAPIDS Memory Manager)

Usage:
    from data_collection.utils.gpu_accelerator import GPUAccelerator, get_accelerator

    # Get accelerator (auto-detects GPU availability)
    accel = get_accelerator()

    # Process data (uses GPU if available, else CPU)
    df_processed = accel.clean_ohlcv(df)
    df_outliers = accel.detect_outliers(df, columns=['close', 'volume'])

Version: 1.0.0
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum, auto
import warnings

import pandas as pd
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

# =============================================================================
# GPU AVAILABILITY DETECTION
# =============================================================================

# Try to import RAPIDS cuDF
try:
    import cudf
    import cupy as cp
    from cudf import DataFrame as cuDataFrame
    CUDF_AVAILABLE = True
    logger.info("RAPIDS cuDF detected - GPU acceleration available")
except ImportError:
    CUDF_AVAILABLE = False
    cudf = None
    cp = None
    cuDataFrame = None
    logger.debug("cuDF not available - using CPU processing")

# Try to import dask for distributed processing
try:
    import dask.dataframe as dd
    from dask.distributed import Client, LocalCluster
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    dd = None

# Try to import GPU dask
try:
    import dask_cudf
    DASK_CUDF_AVAILABLE = True
except ImportError:
    DASK_CUDF_AVAILABLE = False
    dask_cudf = None

# Check for GPU availability
def _check_gpu_available() -> Tuple[bool, Optional[str]]:
    """Check if GPU is available and return info."""
    if not CUDF_AVAILABLE:
        return False, "cuDF not installed"

    try:
        # Try to get GPU info
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            return True, gpu_info
        return False, "nvidia-smi failed"
    except Exception as e:
        return False, f"GPU check failed: {e}"

GPU_AVAILABLE, GPU_INFO = _check_gpu_available()

# =============================================================================
# ENUMS
# =============================================================================

class ProcessingMode(Enum):
    """Processing mode selection."""
    AUTO = auto() # Auto-detect best mode
    GPU = auto() # Force GPU (fails if not available)
    CPU = auto() # Force CPU
    HYBRID = auto() # Use GPU for compute, CPU for I/O

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AcceleratorConfig:
    """Configuration for GPU accelerator."""
    mode: ProcessingMode = ProcessingMode.AUTO
    chunk_size: int = 100_000 # Rows per chunk for batch processing
    memory_fraction: float = 0.8 # Max GPU memory to use
    enable_rmm_pool: bool = True # Use RAPIDS Memory Manager pooling
    n_workers: int = 1 # Number of Dask workers for distributed
    threads_per_worker: int = 4
    enable_spilling: bool = True # Spill to host memory if GPU full

    def __post_init__(self):
        if self.mode == ProcessingMode.GPU and not GPU_AVAILABLE:
            logger.warning("GPU mode requested but GPU not available, falling back to CPU")
            self.mode = ProcessingMode.CPU

@dataclass
class ProcessingStats:
    """Statistics from accelerated processing."""
    mode_used: str
    rows_processed: int
    processing_time_ms: float
    gpu_memory_used_mb: float = 0.0
    speedup_factor: float = 1.0 # Compared to estimated CPU time

# =============================================================================
# GPU ACCELERATOR CLASS
# =============================================================================

class GPUAccelerator:
    """
    GPU-accelerated data processing for Phase 1 pipeline.

    Provides transparent acceleration using RAPIDS cuDF when available,
    with automatic fallback to pandas for compatibility.
    """

    def __init__(self, config: Optional[AcceleratorConfig] = None):
        """Initialize GPU accelerator."""
        self.config = config or AcceleratorConfig()
        self._use_gpu = False
        self._rmm_initialized = False

        # Determine if we should use GPU
        if self.config.mode == ProcessingMode.AUTO:
            self._use_gpu = GPU_AVAILABLE
        elif self.config.mode == ProcessingMode.GPU:
            if not GPU_AVAILABLE:
                raise RuntimeError("GPU mode requested but GPU not available")
            self._use_gpu = True
        elif self.config.mode == ProcessingMode.HYBRID:
            self._use_gpu = GPU_AVAILABLE
        else:
            self._use_gpu = False

        # Initialize RMM pool allocator for better GPU memory management
        if self._use_gpu and self.config.enable_rmm_pool:
            self._init_rmm_pool()

        logger.info(f"GPUAccelerator initialized: mode={self.config.mode.name}, "
                   f"use_gpu={self._use_gpu}")

    def _init_rmm_pool(self):
        """Initialize RAPIDS Memory Manager pool allocator."""
        if self._rmm_initialized:
            return

        try:
            import rmm

            # Get GPU memory info
            if cp is not None:
                device = cp.cuda.Device(0)
                total_memory = device.mem_info[1]
                pool_size = int(total_memory * self.config.memory_fraction)

                # Initialize pool allocator
                rmm.reinitialize(
                    pool_allocator=True,
                    initial_pool_size=pool_size,
                    maximum_pool_size=pool_size,
                    managed_memory=False
                )
                self._rmm_initialized = True
                logger.info(f"RMM pool initialized: {pool_size / 1e9:.1f} GB")
        except Exception as e:
            logger.warning(f"RMM pool initialization failed: {e}")

    @property
    def is_gpu_enabled(self) -> bool:
        """Check if GPU processing is enabled."""
        return self._use_gpu

    @property
    def gpu_info(self) -> Optional[str]:
        """Get GPU information."""
        return GPU_INFO if self._use_gpu else None

    # =========================================================================
    # DATAFRAME CONVERSION
    # =========================================================================

    def to_gpu(self, df: pd.DataFrame) -> Union[pd.DataFrame, 'cuDataFrame']:
        """Convert pandas DataFrame to GPU DataFrame if possible."""
        if not self._use_gpu or cudf is None:
            return df

        try:
            return cudf.DataFrame.from_pandas(df)
        except Exception as e:
            logger.warning(f"GPU conversion failed, using CPU: {e}")
            return df

    def to_cpu(self, df: Union[pd.DataFrame, 'cuDataFrame']) -> pd.DataFrame:
        """Convert GPU DataFrame back to pandas if needed."""
        if not self._use_gpu or cudf is None:
            return df

        if isinstance(df, cudf.DataFrame):
            return df.to_pandas()
        return df

    # =========================================================================
    # CLEANING OPERATIONS (GPU-Accelerated)
    # =========================================================================

    def clean_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        GPU-accelerated OHLCV cleaning.

        Operations:
        - Deduplication
        - Type coercion
        - OHLC consistency fixes
        - Null handling
        """
        import time
        start_time = time.time()

        if self._use_gpu and cudf is not None:
            try:
                # Convert to GPU
                gdf = cudf.DataFrame.from_pandas(df)

                # Deduplication (GPU)
                if 'timestamp' in gdf.columns:
                    gdf = gdf.drop_duplicates(subset=['timestamp'], keep='last')

                # Sort by timestamp (GPU)
                gdf = gdf.sort_values('timestamp')

                # OHLC consistency (GPU)
                if all(c in gdf.columns for c in ['open', 'high', 'low', 'close']):
                    # Fix high - must be >= all others
                    gdf['high'] = gdf[['open', 'high', 'low', 'close']].max(axis=1)
                    # Fix low - must be <= all others
                    gdf['low'] = gdf[['open', 'high', 'low', 'close']].min(axis=1)

                # Null handling (GPU ffill)
                for col in ['open', 'high', 'low', 'close']:
                    if col in gdf.columns and gdf[col].isnull().any():
                        gdf[col] = gdf[col].ffill().bfill()

                # Convert back to CPU
                result = gdf.to_pandas()

                elapsed = (time.time() - start_time) * 1000
                logger.info(f"GPU clean_ohlcv: {len(df)} rows in {elapsed:.1f}ms")

                return result

            except Exception as e:
                logger.warning(f"GPU cleaning failed, falling back to CPU: {e}")

        # CPU fallback
        return self._clean_ohlcv_cpu(df)

    def _clean_ohlcv_cpu(self, df: pd.DataFrame) -> pd.DataFrame:
        """CPU-based OHLCV cleaning."""
        # Deduplication
        if 'timestamp' in df.columns:
            df = df.drop_duplicates(subset=['timestamp'], keep='last')

        # Sort
        df = df.sort_values('timestamp').reset_index(drop=True)

        # OHLC consistency
        if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
            df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
            df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

        # Null handling
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()

        return df

    # =========================================================================
    # OUTLIER DETECTION (GPU-Accelerated)
    # =========================================================================

    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        GPU-accelerated outlier detection.

        Methods:
        - 'iqr': Interquartile range method
        - 'zscore': Z-score method
        - 'mad': Median absolute deviation

        Returns:
            DataFrame with outliers flagged, dict of outlier counts per column
        """
        outlier_counts = {}

        if self._use_gpu and cudf is not None and cp is not None:
            try:
                gdf = cudf.DataFrame.from_pandas(df)

                for col in columns:
                    if col not in gdf.columns:
                        continue

                    values = gdf[col].values

                    if method == 'iqr':
                        q1 = float(gdf[col].quantile(0.25))
                        q3 = float(gdf[col].quantile(0.75))
                        iqr = q3 - q1
                        lower = q1 - threshold * iqr
                        upper = q3 + threshold * iqr
                        is_outlier = (gdf[col] < lower) | (gdf[col] > upper)

                    elif method == 'zscore':
                        mean = float(gdf[col].mean())
                        std = float(gdf[col].std())
                        if std > 0:
                            z_scores = (gdf[col] - mean) / std
                            is_outlier = z_scores.abs() > threshold
                        else:
                            is_outlier = cudf.Series([False] * len(gdf))

                    elif method == 'mad':
                        median = float(gdf[col].median())
                        mad = float((gdf[col] - median).abs().median())
                        if mad > 0:
                            modified_z = 0.6745 * (gdf[col] - median) / mad
                            is_outlier = modified_z.abs() > threshold
                        else:
                            is_outlier = cudf.Series([False] * len(gdf))

                    outlier_counts[col] = int(is_outlier.sum())
                    gdf[f'{col}_is_outlier'] = is_outlier

                return gdf.to_pandas(), outlier_counts

            except Exception as e:
                logger.warning(f"GPU outlier detection failed: {e}")

        # CPU fallback
        return self._detect_outliers_cpu(df, columns, method, threshold)

    def _detect_outliers_cpu(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str,
        threshold: float
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """CPU-based outlier detection."""
        outlier_counts = {}
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            if method == 'iqr':
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                is_outlier = (df[col] < lower) | (df[col] > upper)

            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    z_scores = (df[col] - mean) / std
                    is_outlier = z_scores.abs() > threshold
                else:
                    is_outlier = pd.Series([False] * len(df))

            elif method == 'mad':
                median = df[col].median()
                mad = (df[col] - median).abs().median()
                if mad > 0:
                    modified_z = 0.6745 * (df[col] - median) / mad
                    is_outlier = modified_z.abs() > threshold
                else:
                    is_outlier = pd.Series([False] * len(df))

            outlier_counts[col] = int(is_outlier.sum())
            df[f'{col}_is_outlier'] = is_outlier

        return df, outlier_counts

    # =========================================================================
    # STATISTICAL OPERATIONS (GPU-Accelerated)
    # =========================================================================

    def rolling_statistics(
        self,
        df: pd.DataFrame,
        column: str,
        window: int,
        operations: List[str] = ['mean', 'std']
    ) -> pd.DataFrame:
        """
        GPU-accelerated rolling window statistics.

        Operations: 'mean', 'std', 'min', 'max', 'sum', 'median'
        """
        if self._use_gpu and cudf is not None:
            try:
                gdf = cudf.DataFrame.from_pandas(df[[column]])

                for op in operations:
                    if op == 'mean':
                        gdf[f'{column}_rolling_mean'] = gdf[column].rolling(window).mean()
                    elif op == 'std':
                        gdf[f'{column}_rolling_std'] = gdf[column].rolling(window).std()
                    elif op == 'min':
                        gdf[f'{column}_rolling_min'] = gdf[column].rolling(window).min()
                    elif op == 'max':
                        gdf[f'{column}_rolling_max'] = gdf[column].rolling(window).max()
                    elif op == 'sum':
                        gdf[f'{column}_rolling_sum'] = gdf[column].rolling(window).sum()

                result = gdf.to_pandas()
                # Merge back with original
                for col in result.columns:
                    if col != column:
                        df[col] = result[col].values
                return df

            except Exception as e:
                logger.warning(f"GPU rolling stats failed: {e}")

        # CPU fallback
        for op in operations:
            rolling = df[column].rolling(window)
            if hasattr(rolling, op):
                df[f'{column}_rolling_{op}'] = getattr(rolling, op)()

        return df

    def compute_correlations(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """GPU-accelerated correlation matrix computation."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if self._use_gpu and cudf is not None:
            try:
                gdf = cudf.DataFrame.from_pandas(df[columns])
                corr_matrix = gdf.corr()
                return corr_matrix.to_pandas()
            except Exception as e:
                logger.warning(f"GPU correlation failed: {e}")

        # CPU fallback
        return df[columns].corr()

    # =========================================================================
    # BATCH PROCESSING (GPU-Accelerated)
    # =========================================================================

    def process_in_batches(
        self,
        df: pd.DataFrame,
        process_func: callable,
        chunk_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Process large DataFrame in GPU-memory-friendly batches.

        Useful for datasets that don't fit entirely in GPU memory.
        """
        chunk_size = chunk_size or self.config.chunk_size
        n_chunks = (len(df) + chunk_size - 1) // chunk_size

        results = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(df))
            chunk = df.iloc[start:end]

            # Process chunk (will use GPU if available)
            processed = process_func(chunk)
            results.append(processed)

            logger.debug(f"Processed batch {i+1}/{n_chunks}")

        return pd.concat(results, ignore_index=True)

    # =========================================================================
    # CROSS-VENUE OPERATIONS (GPU-Accelerated)
    # =========================================================================

    def merge_venue_data(
        self,
        venue_dfs: Dict[str, pd.DataFrame],
        on: str = 'timestamp',
        how: str = 'outer'
    ) -> pd.DataFrame:
        """GPU-accelerated multi-venue data merge."""
        if not venue_dfs:
            return pd.DataFrame()

        if self._use_gpu and cudf is not None:
            try:
                # Convert all to GPU
                gdfs = {
                    name: cudf.DataFrame.from_pandas(df)
                    for name, df in venue_dfs.items()
                }

                # Merge sequentially
                result = None
                for name, gdf in gdfs.items():
                    # Add venue suffix
                    gdf = gdf.rename(columns={
                        c: f'{c}_{name}' for c in gdf.columns if c != on
                    })

                    if result is None:
                        result = gdf
                    else:
                        result = result.merge(gdf, on=on, how=how)

                return result.to_pandas()

            except Exception as e:
                logger.warning(f"GPU merge failed: {e}")

        # CPU fallback
        result = None
        for name, df in venue_dfs.items():
            df = df.rename(columns={
                c: f'{c}_{name}' for c in df.columns if c != on
            })

            if result is None:
                result = df
            else:
                result = result.merge(df, on=on, how=how)

        return result if result is not None else pd.DataFrame()

    # =========================================================================
    # WASH TRADING DETECTION (GPU-Accelerated)
    # =========================================================================

    def detect_volume_anomalies(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        volume_col: str = 'volume',
        window: int = 20
    ) -> Dict[str, Any]:
        """GPU-accelerated volume anomaly detection for wash trading."""
        result = {
            'volume_price_divergence': 0.0,
            'volume_consistency': 0.0,
            'suspicious_periods': 0
        }

        if self._use_gpu and cudf is not None:
            try:
                gdf = cudf.DataFrame.from_pandas(df[[price_col, volume_col]])

                # Price change
                gdf['price_change'] = gdf[price_col].pct_change().abs()

                # Volume normalized
                gdf['volume_mean'] = gdf[volume_col].rolling(window).mean()
                gdf['volume_norm'] = gdf[volume_col] / gdf['volume_mean']

                # VPD: High volume with low price change
                gdf['is_suspicious'] = (gdf['volume_norm'] > 2) & (gdf['price_change'] < 0.001)

                result['suspicious_periods'] = int(gdf['is_suspicious'].sum())
                result['volume_price_divergence'] = (
                    result['suspicious_periods'] / len(gdf) * 100
                )

                # Volume consistency (CV)
                vol_mean = float(gdf[volume_col].mean())
                vol_std = float(gdf[volume_col].std())
                result['volume_consistency'] = vol_std / vol_mean if vol_mean > 0 else 0

                return result

            except Exception as e:
                logger.warning(f"GPU wash detection failed: {e}")

        # CPU fallback
        df_temp = df.copy()
        df_temp['price_change'] = df_temp[price_col].pct_change().abs()
        df_temp['volume_mean'] = df_temp[volume_col].rolling(window).mean()
        df_temp['volume_norm'] = df_temp[volume_col] / df_temp['volume_mean']
        df_temp['is_suspicious'] = (df_temp['volume_norm'] > 2) & (df_temp['price_change'] < 0.001)

        result['suspicious_periods'] = int(df_temp['is_suspicious'].sum())
        result['volume_price_divergence'] = result['suspicious_periods'] / len(df) * 100
        result['volume_consistency'] = (
            df[volume_col].std() / df[volume_col].mean()
            if df[volume_col].mean() > 0 else 0
        )

        return result

# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_accelerator_instance: Optional[GPUAccelerator] = None

def get_accelerator(config: Optional[AcceleratorConfig] = None) -> GPUAccelerator:
    """Get or create the global GPU accelerator instance."""
    global _accelerator_instance

    if _accelerator_instance is None or config is not None:
        _accelerator_instance = GPUAccelerator(config)

    return _accelerator_instance

def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return GPU_AVAILABLE

def get_gpu_info() -> Optional[str]:
    """Get GPU information string."""
    return GPU_INFO if GPU_AVAILABLE else None

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def accelerate_dataframe(df: pd.DataFrame) -> Union[pd.DataFrame, 'cuDataFrame']:
    """Convert DataFrame to GPU if available."""
    accel = get_accelerator()
    return accel.to_gpu(df)

def decelerate_dataframe(df: Union[pd.DataFrame, 'cuDataFrame']) -> pd.DataFrame:
    """Convert GPU DataFrame back to pandas."""
    accel = get_accelerator()
    return accel.to_cpu(df)
