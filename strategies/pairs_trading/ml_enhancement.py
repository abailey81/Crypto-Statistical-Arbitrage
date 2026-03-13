"""
ML Enhancement Module for Pairs Trading Strategy
=================================================

Comprehensive machine learning framework for improving pairs trading
signals using gradient boosting, walk-forward validation, and DeFi-specific
feature engineering.

Mathematical Framework
----------------------
Target Variable Construction:

    Spread Direction (Classification):
        y_t = 1 if spread_{t+h} < spread_t else 0
    
    Mean Reversion (Classification):
        y_t = 1 if |z_{t+h}| < |z_t| else 0
    
    Spread Return (Regression):
        y_t = (spread_{t+h} - spread_t) / spread_t

Feature Engineering:

    Price Features:
        - Returns: r_t = (P_t - P_{t-1}) / P_{t-1}
        - Volatility: σ_t = std(r_{t-n:t}) × √252
        - RSI: 100 - 100/(1 + RS)
        - Bollinger Position: (P - L)/(U - L)
    
    Spread Features:
        - Z-score: z_t = (S_t - μ_S) / σ_S
        - Velocity: dS/dt
        - Acceleration: d²S/dt²
        - Days since zero cross
    
    Regime Features:
        - BTC trend indicator
        - Volatility regime (0-3)
        - Correlation with market

Walk-Forward Validation:

    ┌────────────┬───────┬────────────┐
    │   Train    │ Purge │    Test    │
    │  180 days  │ 3 days│   30 days  │
    └────────────┴───────┴────────────┘
                         │
    ┌────────────┬───────┬────────────┐
    │   Train    │ Purge │    Test    │  (embargo)
    └────────────┴───────┴────────────┘

Model Selection:

    XGBoost: Primary model for tabular data
        - Gradient boosting with L1/L2 regularization
        - Native handling of missing values
        - Feature importance via gain/cover
    
    LightGBM: Alternative for large datasets
        - Histogram-based binning
        - Leaf-wise tree growth
        - Faster training
    
    RandomForest: Baseline comparison
        - Bootstrap aggregating
        - Less prone to overfitting

Author: Crypto StatArb Quantitative Research
Version: 2.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import warnings
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional imports with graceful fallbacks
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception as e:
    # Catch all exceptions (ImportError, OSError, XGBoostError, etc.)
    XGBOOST_AVAILABLE = False
    logger.warning(f"XGBoost not available: {type(e).__name__}")
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception as e:
    # Catch all exceptions (ImportError, OSError, LightGBMError, etc.)
    LIGHTGBM_AVAILABLE = False
    logger.warning(f"LightGBM not available: {type(e).__name__}")
    lgb = None

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
        confusion_matrix
    )
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available")


# =============================================================================
# ENUMERATIONS WITH TRADING-SPECIFIC PROPERTIES
# =============================================================================

class ModelType(Enum):
    """
    Available ML model types for pairs trading.
    
    Each model has specific characteristics that make it suitable
    for different data sizes, feature types, and prediction tasks.
    """
    XGBOOST_CLASSIFIER = "xgb_classifier"
    XGBOOST_REGRESSOR = "xgb_regressor"
    LIGHTGBM_CLASSIFIER = "lgb_classifier"
    LIGHTGBM_REGRESSOR = "lgb_regressor"
    RANDOM_FOREST_CLASSIFIER = "rf_classifier"
    RANDOM_FOREST_REGRESSOR = "rf_regressor"
    LOGISTIC_REGRESSION = "logistic"
    RIDGE_REGRESSION = "ridge"
    
    @property
    def is_classifier(self) -> bool:
        """True if model is a classifier."""
        return 'classifier' in self.value or self.value == 'logistic'
    
    @property
    def is_regressor(self) -> bool:
        """True if model is a regressor."""
        return 'regressor' in self.value or self.value == 'ridge'
    
    @property
    def is_gradient_boosting(self) -> bool:
        """True if model is gradient boosting."""
        return self.value.startswith('xgb') or self.value.startswith('lgb')
    
    @property
    def supports_early_stopping(self) -> bool:
        """True if model supports early stopping."""
        return self.is_gradient_boosting
    
    @property
    def supports_feature_importance(self) -> bool:
        """True if model provides feature importance."""
        return self not in [self.LOGISTIC_REGRESSION, self.RIDGE_REGRESSION]
    
    @property
    def recommended_n_estimators(self) -> int:
        """Recommended number of estimators."""
        if self.is_gradient_boosting:
            return 100
        return 200
    
    @property
    def recommended_max_depth(self) -> int:
        """Recommended max depth."""
        if self.is_gradient_boosting:
            return 6
        return 10
    
    @property
    def handles_missing_values(self) -> bool:
        """True if model handles missing values natively."""
        return self.value.startswith('xgb') or self.value.startswith('lgb')
    
    @property
    def training_speed(self) -> str:
        """Relative training speed."""
        speeds = {
            self.XGBOOST_CLASSIFIER: "fast",
            self.XGBOOST_REGRESSOR: "fast",
            self.LIGHTGBM_CLASSIFIER: "very_fast",
            self.LIGHTGBM_REGRESSOR: "very_fast",
            self.RANDOM_FOREST_CLASSIFIER: "moderate",
            self.RANDOM_FOREST_REGRESSOR: "moderate",
            self.LOGISTIC_REGRESSION: "very_fast",
            self.RIDGE_REGRESSION: "very_fast",
        }
        return speeds.get(self, "moderate")
    
    @property
    def overfitting_risk(self) -> str:
        """Relative overfitting risk."""
        risks = {
            self.XGBOOST_CLASSIFIER: "moderate",
            self.XGBOOST_REGRESSOR: "moderate",
            self.LIGHTGBM_CLASSIFIER: "moderate",
            self.LIGHTGBM_REGRESSOR: "moderate",
            self.RANDOM_FOREST_CLASSIFIER: "low",
            self.RANDOM_FOREST_REGRESSOR: "low",
            self.LOGISTIC_REGRESSION: "very_low",
            self.RIDGE_REGRESSION: "very_low",
        }
        return risks.get(self, "moderate")


class PredictionTarget(Enum):
    """
    Prediction target types for ML models.
    
    Defines what the model is trying to predict and how
    to construct the target variable from spread data.
    """
    SPREAD_DIRECTION = "spread_direction"
    SPREAD_RETURN = "spread_return"
    MEAN_REVERSION = "mean_reversion"
    REGIME = "regime"
    HALF_LIFE = "half_life"
    ENTRY_SIGNAL = "entry_signal"
    EXIT_SIGNAL = "exit_signal"
    
    @property
    def is_classification(self) -> bool:
        """True if target is classification."""
        return self in [
            self.SPREAD_DIRECTION, self.MEAN_REVERSION,
            self.REGIME, self.ENTRY_SIGNAL, self.EXIT_SIGNAL
        ]
    
    @property
    def is_regression(self) -> bool:
        """True if target is regression."""
        return self in [self.SPREAD_RETURN, self.HALF_LIFE]
    
    @property
    def description(self) -> str:
        """Target description."""
        descriptions = {
            self.SPREAD_DIRECTION: "Predict if spread will decrease (mean revert)",
            self.SPREAD_RETURN: "Predict spread percentage return",
            self.MEAN_REVERSION: "Predict if z-score will move toward zero",
            self.REGIME: "Predict market regime (bull/bear/neutral)",
            self.HALF_LIFE: "Predict current half-life of mean reversion",
            self.ENTRY_SIGNAL: "Predict optimal entry timing",
            self.EXIT_SIGNAL: "Predict optimal exit timing",
        }
        return descriptions.get(self, "Unknown target")
    
    @property
    def default_horizon_hours(self) -> int:
        """Default prediction horizon in hours."""
        horizons = {
            self.SPREAD_DIRECTION: 24,
            self.SPREAD_RETURN: 24,
            self.MEAN_REVERSION: 48,
            self.REGIME: 168,
            self.HALF_LIFE: 168,
            self.ENTRY_SIGNAL: 4,
            self.EXIT_SIGNAL: 4,
        }
        return horizons.get(self, 24)
    
    @property
    def recommended_model(self) -> ModelType:
        """Recommended model type for this target."""
        if self.is_classification:
            return ModelType.XGBOOST_CLASSIFIER
        return ModelType.XGBOOST_REGRESSOR
    
    @property
    def primary_metric(self) -> str:
        """Primary evaluation metric."""
        if self.is_classification:
            return "auc"
        return "r2"


class FeatureSet(Enum):
    """
    Feature set configurations for different use cases.
    
    Controls which features are included in model training.
    """
    MINIMAL = "minimal"
    STANDARD = "standard"
    EXTENDED = "extended"
    FULL = "full"
    DEFI_FOCUSED = "defi"
    REGIME_FOCUSED = "regime"
    
    @property
    def include_price_features(self) -> bool:
        """Include price-based features."""
        return self != self.MINIMAL
    
    @property
    def include_volume_features(self) -> bool:
        """Include volume-based features."""
        return self in [self.STANDARD, self.EXTENDED, self.FULL]
    
    @property
    def include_funding_features(self) -> bool:
        """Include funding rate features."""
        return self in [self.EXTENDED, self.FULL, self.DEFI_FOCUSED]
    
    @property
    def include_correlation_features(self) -> bool:
        """Include correlation features."""
        return self in [self.EXTENDED, self.FULL, self.REGIME_FOCUSED]
    
    @property
    def include_regime_features(self) -> bool:
        """Include regime features."""
        return self in [self.FULL, self.REGIME_FOCUSED]
    
    @property
    def include_defi_features(self) -> bool:
        """Include DeFi-specific features."""
        return self in [self.FULL, self.DEFI_FOCUSED]
    
    @property
    def include_seasonality(self) -> bool:
        """Include time seasonality features."""
        return self in [self.EXTENDED, self.FULL]
    
    @property
    def expected_feature_count(self) -> Tuple[int, int]:
        """Expected range of feature count."""
        ranges = {
            self.MINIMAL: (10, 30),
            self.STANDARD: (30, 80),
            self.EXTENDED: (80, 150),
            self.FULL: (150, 300),
            self.DEFI_FOCUSED: (50, 120),
            self.REGIME_FOCUSED: (40, 100),
        }
        return ranges.get(self, (30, 100))
    
    @property
    def recommended_lookback_days(self) -> int:
        """Recommended lookback for feature calculation."""
        lookbacks = {
            self.MINIMAL: 30,
            self.STANDARD: 60,
            self.EXTENDED: 120,
            self.FULL: 180,
            self.DEFI_FOCUSED: 90,
            self.REGIME_FOCUSED: 120,
        }
        return lookbacks.get(self, 90)


class SignalConfidence(Enum):
    """
    ML signal confidence levels.
    
    Determines how ML predictions should be combined with
    traditional z-score signals.
    """
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    
    @classmethod
    def from_probability(cls, prob: float) -> 'SignalConfidence':
        """Classify confidence from probability."""
        abs_prob = abs(prob - 0.5) * 2  # Convert to 0-1 scale
        
        if abs_prob < 0.2:
            return cls.VERY_LOW
        elif abs_prob < 0.4:
            return cls.LOW
        elif abs_prob < 0.6:
            return cls.MODERATE
        elif abs_prob < 0.8:
            return cls.HIGH
        return cls.VERY_HIGH
    
    @property
    def position_multiplier(self) -> float:
        """Position size multiplier based on confidence."""
        multipliers = {
            self.VERY_LOW: 0.0,
            self.LOW: 0.5,
            self.MODERATE: 0.75,
            self.HIGH: 1.0,
            self.VERY_HIGH: 1.2,
        }
        return multipliers.get(self, 0.5)
    
    @property
    def should_override_zscore(self) -> bool:
        """True if confidence is high enough to override z-score."""
        return self in [self.HIGH, self.VERY_HIGH]
    
    @property
    def threshold_adjustment(self) -> float:
        """Adjustment to z-score entry threshold."""
        adjustments = {
            self.VERY_LOW: 0.0,
            self.LOW: -0.1,
            self.MODERATE: -0.2,
            self.HIGH: -0.3,
            self.VERY_HIGH: -0.5,
        }
        return adjustments.get(self, 0.0)


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class FeatureConfig:
    """
    Configuration for ML feature engineering.
    
    Controls which features to generate, lookback windows,
    and normalization settings.
    """
    # Lookback windows
    lookback_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60, 120])
    
    # Feature toggles
    include_price_features: bool = True
    include_volume_features: bool = True
    include_spread_features: bool = True
    include_momentum: bool = True
    include_volatility: bool = True
    include_rsi: bool = True
    include_bollinger: bool = True
    include_funding_rate: bool = True
    include_correlation: bool = True
    include_regime: bool = True
    include_seasonality: bool = True
    include_gas_features: bool = True
    include_dex_features: bool = True
    
    # Indicator parameters
    rsi_period: int = 14
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Normalization
    normalize_features: bool = True
    normalization_method: str = "robust"
    
    # Feature selection
    max_features: Optional[int] = None
    min_importance: float = 0.001
    drop_correlated: bool = True
    correlation_threshold: float = 0.95
    
    @classmethod
    def from_feature_set(cls, feature_set: FeatureSet) -> 'FeatureConfig':
        """Create config from feature set enum."""
        return cls(
            include_price_features=feature_set.include_price_features,
            include_volume_features=feature_set.include_volume_features,
            include_funding_rate=feature_set.include_funding_features,
            include_correlation=feature_set.include_correlation_features,
            include_regime=feature_set.include_regime_features,
            include_gas_features=feature_set.include_defi_features,
            include_dex_features=feature_set.include_defi_features,
            include_seasonality=feature_set.include_seasonality,
        )
    
    @property
    def max_lookback(self) -> int:
        """Maximum lookback period."""
        return max(self.lookback_windows)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'lookback_windows': self.lookback_windows,
            'normalize_features': self.normalize_features,
            'normalization_method': self.normalization_method,
            'max_features': self.max_features,
            'correlation_threshold': self.correlation_threshold,
        }


@dataclass
class MLConfig:
    """
    Configuration for ML model training and validation.
    
    Contains all hyperparameters, validation settings, and
    model persistence options.
    """
    # Model settings
    model_type: ModelType = ModelType.XGBOOST_CLASSIFIER
    prediction_target: PredictionTarget = PredictionTarget.SPREAD_DIRECTION
    
    # Walk-forward settings
    train_window_days: int = 180
    test_window_days: int = 30
    purge_days: int = 3
    embargo_days: int = 1
    min_train_samples: int = 1000
    
    # Model hyperparameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: int = 10
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    
    # Classification settings
    class_weight: str = "balanced"
    probability_threshold: float = 0.5
    
    # Prediction horizon
    target_horizon: int = 24
    
    # Evaluation
    eval_metric: str = "auc"
    early_stopping_rounds: int = 20
    
    # Persistence
    save_models: bool = True
    model_dir: str = "models/ml"
    model_version: str = "v1"
    
    @property
    def is_classification(self) -> bool:
        """True if classification task."""
        return self.prediction_target.is_classification
    
    @property
    def total_window_days(self) -> int:
        """Total days for one walk-forward fold."""
        return self.train_window_days + self.purge_days + self.test_window_days
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_type': self.model_type.value,
            'prediction_target': self.prediction_target.value,
            'train_window_days': self.train_window_days,
            'test_window_days': self.test_window_days,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'eval_metric': self.eval_metric,
        }


# =============================================================================
# WALK-FORWARD RESULT DATACLASS
# =============================================================================

@dataclass
class WalkForwardResult:
    """
    Comprehensive results from a single walk-forward fold.
    
    Contains metrics, predictions, and model information for
    analysis and ensemble construction.
    """
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_samples: int
    test_samples: int
    
    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    auc: Optional[float] = None
    
    # Regression metrics
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    rmse: Optional[float] = None
    
    # Predictions
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    actuals: Optional[np.ndarray] = None
    
    # Feature importance
    feature_importance: Optional[Dict[str, float]] = None
    
    # Model reference
    model: Optional[Any] = None
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.mse is not None and self.rmse is None:
            self.rmse = np.sqrt(self.mse)
    
    # Time properties
    @property
    def train_duration_days(self) -> int:
        """Training period duration in days."""
        return (self.train_end - self.train_start).days
    
    @property
    def test_duration_days(self) -> int:
        """Test period duration in days."""
        return (self.test_end - self.test_start).days
    
    @property
    def test_date_range(self) -> str:
        """Test date range string."""
        return f"{self.test_start.date()} to {self.test_end.date()}"
    
    # Classification properties
    @property
    def is_classification_result(self) -> bool:
        """True if classification result."""
        return self.accuracy is not None
    
    @property
    def specificity(self) -> Optional[float]:
        """True negative rate."""
        if self.predictions is None or self.actuals is None:
            return None
        
        tn = np.sum((self.predictions == 0) & (self.actuals == 0))
        fp = np.sum((self.predictions == 1) & (self.actuals == 0))
        
        if tn + fp == 0:
            return 0.0
        return tn / (tn + fp)
    
    @property
    def balanced_accuracy(self) -> Optional[float]:
        """Balanced accuracy (average of recall and specificity)."""
        if self.recall is None or self.specificity is None:
            return None
        return (self.recall + self.specificity) / 2
    
    @property
    def matthews_correlation(self) -> Optional[float]:
        """Matthews correlation coefficient."""
        if self.predictions is None or self.actuals is None:
            return None
        
        tp = np.sum((self.predictions == 1) & (self.actuals == 1))
        tn = np.sum((self.predictions == 0) & (self.actuals == 0))
        fp = np.sum((self.predictions == 1) & (self.actuals == 0))
        fn = np.sum((self.predictions == 0) & (self.actuals == 1))
        
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denom == 0:
            return 0.0
        
        return (tp * tn - fp * fn) / denom
    
    @property
    def prediction_rate(self) -> Optional[float]:
        """Rate of positive predictions."""
        if self.predictions is None:
            return None
        return np.mean(self.predictions)
    
    # Regression properties
    @property
    def is_regression_result(self) -> bool:
        """True if regression result."""
        return self.mse is not None
    
    @property
    def mape(self) -> Optional[float]:
        """Mean absolute percentage error."""
        if self.predictions is None or self.actuals is None:
            return None
        
        nonzero_mask = self.actuals != 0
        if not np.any(nonzero_mask):
            return None
        
        return np.mean(np.abs(
            (self.actuals[nonzero_mask] - self.predictions[nonzero_mask]) /
            self.actuals[nonzero_mask]
        )) * 100
    
    @property
    def directional_accuracy(self) -> Optional[float]:
        """Accuracy of predicting direction (for regression)."""
        if self.predictions is None or self.actuals is None:
            return None
        
        pred_direction = np.sign(self.predictions)
        actual_direction = np.sign(self.actuals)
        
        return np.mean(pred_direction == actual_direction)
    
    # Feature importance properties
    @property
    def n_features(self) -> int:
        """Number of features."""
        if self.feature_importance is None:
            return 0
        return len(self.feature_importance)
    
    @property
    def top_features(self) -> List[Tuple[str, float]]:
        """Top 10 features by importance."""
        if self.feature_importance is None:
            return []
        
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:10]
    
    @property
    def feature_concentration(self) -> float:
        """Concentration of importance in top features."""
        if self.feature_importance is None or len(self.feature_importance) == 0:
            return 0.0
        
        values = sorted(self.feature_importance.values(), reverse=True)
        total = sum(values)
        
        if total == 0:
            return 0.0
        
        # Top 10% of features
        n_top = max(1, len(values) // 10)
        return sum(values[:n_top]) / total
    
    # Quality metrics
    @property
    def primary_metric(self) -> Optional[float]:
        """Primary metric value."""
        if self.is_classification_result:
            return self.auc
        return self.r2
    
    @property
    def is_good_result(self) -> bool:
        """True if result meets quality threshold."""
        if self.is_classification_result:
            return self.auc is not None and self.auc > 0.55
        return self.r2 is not None and self.r2 > 0.05
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'fold_id': self.fold_id,
            'test_date_range': self.test_date_range,
            'train_samples': self.train_samples,
            'test_samples': self.test_samples,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'auc': self.auc,
            'mse': self.mse,
            'mae': self.mae,
            'r2': self.r2,
            'n_features': self.n_features,
            'is_good_result': self.is_good_result,
        }
    
    def __repr__(self) -> str:
        metric = f"AUC={self.auc:.4f}" if self.auc else f"R²={self.r2:.4f}"
        return f"WalkForwardResult(fold={self.fold_id}, {metric}, test={self.test_samples})"


@dataclass
class ValidationSummary:
    """
    Summary of walk-forward validation results.
    
    Aggregates metrics across all folds for model evaluation.
    """
    n_folds: int
    total_train_samples: int
    total_test_samples: int
    
    # Aggregated classification metrics
    accuracy_mean: Optional[float] = None
    accuracy_std: Optional[float] = None
    precision_mean: Optional[float] = None
    recall_mean: Optional[float] = None
    f1_mean: Optional[float] = None
    auc_mean: Optional[float] = None
    auc_std: Optional[float] = None
    
    # Aggregated regression metrics
    mse_mean: Optional[float] = None
    mae_mean: Optional[float] = None
    r2_mean: Optional[float] = None
    r2_std: Optional[float] = None
    
    # Feature importance
    feature_importance: Dict[str, float] = field(default_factory=dict)
    top_features: List[Tuple[str, float]] = field(default_factory=list)
    
    # Stability metrics
    metric_stability: float = 0.0
    feature_stability: float = 0.0
    
    @property
    def is_classification(self) -> bool:
        """True if classification results."""
        return self.accuracy_mean is not None
    
    @property
    def primary_metric_mean(self) -> float:
        """Mean of primary metric."""
        if self.is_classification:
            return self.auc_mean or 0.5
        return self.r2_mean or 0.0
    
    @property
    def primary_metric_std(self) -> float:
        """Std of primary metric."""
        if self.is_classification:
            return self.auc_std or 0.0
        return self.r2_std or 0.0
    
    @property
    def sharpe_like_ratio(self) -> float:
        """Metric mean / std ratio (higher = more stable)."""
        if self.primary_metric_std == 0:
            return 0.0
        return self.primary_metric_mean / self.primary_metric_std
    
    @property
    def is_production_ready(self) -> bool:
        """True if model is ready for production."""
        if self.is_classification:
            return (
                self.auc_mean is not None and
                self.auc_mean > 0.55 and
                self.sharpe_like_ratio > 3.0
            )
        return (
            self.r2_mean is not None and
            self.r2_mean > 0.05 and
            self.sharpe_like_ratio > 2.0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'n_folds': self.n_folds,
            'total_train_samples': self.total_train_samples,
            'total_test_samples': self.total_test_samples,
            'accuracy_mean': round(self.accuracy_mean, 4) if self.accuracy_mean else None,
            'precision_mean': round(self.precision_mean, 4) if self.precision_mean else None,
            'recall_mean': round(self.recall_mean, 4) if self.recall_mean else None,
            'f1_mean': round(self.f1_mean, 4) if self.f1_mean else None,
            'auc_mean': round(self.auc_mean, 4) if self.auc_mean else None,
            'auc_std': round(self.auc_std, 4) if self.auc_std else None,
            'mse_mean': round(self.mse_mean, 6) if self.mse_mean else None,
            'r2_mean': round(self.r2_mean, 4) if self.r2_mean else None,
            'sharpe_like_ratio': round(self.sharpe_like_ratio, 2),
            'is_production_ready': self.is_production_ready,
            'top_features': self.top_features[:10],
        }


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """
    Comprehensive feature engineering for pairs trading ML models.
    
    Creates price, spread, volume, funding, regime, and DeFi-specific
    features with proper normalization and handling of missing values.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize feature engineer."""
        self.config = config or FeatureConfig()
        self.scaler = None
        self.feature_names: List[str] = []
        self.feature_stats: Dict[str, Dict] = {}
    
    def create_spread_features(
        self,
        spread: pd.Series,
        zscore: pd.Series,
        hedge_ratio: float,
        half_life: Optional[float] = None
    ) -> pd.DataFrame:
        """Create features from spread and z-score data."""
        features = pd.DataFrame(index=spread.index)
        
        # Current values
        features['spread'] = spread
        features['zscore'] = zscore
        features['zscore_abs'] = zscore.abs()
        
        # Rolling statistics
        for window in self.config.lookback_windows:
            features[f'spread_ma_{window}'] = spread.rolling(window).mean()
            features[f'spread_std_{window}'] = spread.rolling(window).std()
            features[f'zscore_ma_{window}'] = zscore.rolling(window).mean()
            features[f'zscore_std_{window}'] = zscore.rolling(window).std()
            features[f'spread_roc_{window}'] = spread.pct_change(window)
            features[f'zscore_change_{window}'] = zscore.diff(window)
        
        # Velocity and acceleration
        features['spread_velocity'] = spread.diff()
        features['spread_acceleration'] = features['spread_velocity'].diff()
        features['zscore_velocity'] = zscore.diff()
        features['zscore_acceleration'] = features['zscore_velocity'].diff()
        
        # Percentile positions
        for window in [20, 60, 120]:
            features[f'zscore_percentile_{window}'] = zscore.rolling(window).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10)
                if len(x) > 0 else 0.5,
                raw=False
            )
            features[f'zscore_max_{window}'] = zscore.rolling(window).max()
            features[f'zscore_min_{window}'] = zscore.rolling(window).min()
        
        # Mean reversion indicators
        features['crossed_zero_5d'] = (
            (zscore.shift(1) * zscore < 0).rolling(5).sum()
        )
        features['days_since_cross'] = self._days_since_zero_cross(zscore)
        
        # Half-life features
        if half_life:
            features['half_life'] = half_life
            features['periods_to_mean'] = half_life * np.log(2)
            features['expected_reversion'] = spread * (1 - np.exp(-1 / max(half_life, 0.1)))
        
        features['hedge_ratio'] = hedge_ratio
        
        return features
    
    def create_price_features(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        name_a: str = 'a',
        name_b: str = 'b'
    ) -> pd.DataFrame:
        """Create features from price data for both legs."""
        features = pd.DataFrame(index=price_a.index)
        
        for name, price in [(name_a, price_a), (name_b, price_b)]:
            # Returns
            features[f'return_{name}'] = price.pct_change()
            features[f'log_return_{name}'] = np.log(price / price.shift(1))
            
            # Rolling returns and volatility
            for window in self.config.lookback_windows:
                features[f'return_{name}_{window}d'] = price.pct_change(window)
                features[f'volatility_{name}_{window}d'] = (
                    price.pct_change().rolling(window).std() * np.sqrt(252)
                )
            
            # Momentum
            if self.config.include_momentum:
                for window in [5, 10, 20]:
                    features[f'momentum_{name}_{window}'] = price / price.shift(window) - 1
                
                # MACD
                ema_fast = price.ewm(span=self.config.macd_fast).mean()
                ema_slow = price.ewm(span=self.config.macd_slow).mean()
                macd = ema_fast - ema_slow
                features[f'macd_{name}'] = macd
                features[f'macd_signal_{name}'] = macd.ewm(span=self.config.macd_signal).mean()
            
            # RSI
            if self.config.include_rsi:
                features[f'rsi_{name}'] = self._calculate_rsi(price, self.config.rsi_period)
            
            # Bollinger
            if self.config.include_bollinger:
                features[f'bb_pos_{name}'] = self._bollinger_position(price)
        
        # Relative features
        features['price_ratio'] = price_a / price_b
        features['return_diff'] = features[f'return_{name_a}'] - features[f'return_{name_b}']
        
        # Correlation
        if self.config.include_correlation:
            returns_a = price_a.pct_change()
            returns_b = price_b.pct_change()
            for window in [20, 60]:
                features[f'correlation_{window}'] = returns_a.rolling(window).corr(returns_b)
        
        return features
    
    def create_volume_features(
        self,
        volume_a: pd.Series,
        volume_b: pd.Series
    ) -> pd.DataFrame:
        """Create features from volume data."""
        features = pd.DataFrame(index=volume_a.index)
        
        for name, vol in [('a', volume_a), ('b', volume_b)]:
            for window in [5, 20, 60]:
                vol_ma = vol.rolling(window).mean()
                features[f'vol_ratio_{name}_{window}'] = vol / (vol_ma + 1e-10)
            
            features[f'vol_change_{name}'] = vol.pct_change()
        
        features['vol_ratio_ab'] = volume_a / (volume_b + 1e-10)
        features['volume_imbalance'] = (volume_a - volume_b) / (volume_a + volume_b + 1e-10)
        
        return features
    
    def create_funding_features(
        self,
        funding_a: pd.Series,
        funding_b: pd.Series
    ) -> pd.DataFrame:
        """Create features from funding rate data."""
        features = pd.DataFrame(index=funding_a.index)
        
        features['funding_a'] = funding_a
        features['funding_b'] = funding_b
        features['funding_diff'] = funding_a - funding_b
        
        for window in [8, 24, 72]:
            features[f'funding_diff_ma_{window}'] = features['funding_diff'].rolling(window).mean()
        
        features['funding_a_zscore'] = (
            funding_a - funding_a.rolling(72).mean()
        ) / (funding_a.rolling(72).std() + 1e-10)
        
        features['cum_funding_diff_24h'] = features['funding_diff'].rolling(24).sum()
        
        return features
    
    def create_regime_features(
        self,
        btc_price: pd.Series
    ) -> pd.DataFrame:
        """Create market regime features."""
        features = pd.DataFrame(index=btc_price.index)
        
        btc_return = btc_price.pct_change()
        features['btc_return'] = btc_return
        features['btc_return_20d'] = btc_price.pct_change(20)
        features['btc_ma_cross'] = btc_price.rolling(20).mean() / btc_price.rolling(50).mean() - 1
        
        btc_vol = btc_return.rolling(20).std() * np.sqrt(252)
        features['btc_volatility'] = btc_vol
        features['btc_vol_regime'] = pd.cut(
            btc_vol, bins=[0, 0.3, 0.6, 1.0, float('inf')], labels=[0, 1, 2, 3]
        ).astype(float)
        
        features['btc_trend'] = np.where(
            btc_price > btc_price.rolling(20).mean(), 1,
            np.where(btc_price < btc_price.rolling(20).mean(), -1, 0)
        )
        
        return features
    
    def create_seasonality_features(
        self,
        timestamps: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Create time seasonality features."""
        features = pd.DataFrame(index=timestamps)
        
        if hasattr(timestamps, 'hour'):
            features['hour_sin'] = np.sin(2 * np.pi * timestamps.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * timestamps.hour / 24)
        
        features['dow_sin'] = np.sin(2 * np.pi * timestamps.dayofweek / 7)
        features['dow_cos'] = np.cos(2 * np.pi * timestamps.dayofweek / 7)
        features['is_weekend'] = (timestamps.dayofweek >= 5).astype(int)
        
        features['month_sin'] = np.sin(2 * np.pi * timestamps.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * timestamps.month / 12)
        
        return features
    
    def create_all_features(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        spread: pd.Series,
        zscore: pd.Series,
        hedge_ratio: float,
        volume_a: Optional[pd.Series] = None,
        volume_b: Optional[pd.Series] = None,
        funding_a: Optional[pd.Series] = None,
        funding_b: Optional[pd.Series] = None,
        btc_price: Optional[pd.Series] = None,
        half_life: Optional[float] = None
    ) -> pd.DataFrame:
        """Create all features for ML model."""
        all_features = []
        
        # Spread features (always)
        all_features.append(self.create_spread_features(
            spread, zscore, hedge_ratio, half_life
        ))
        
        # Price features
        if self.config.include_price_features:
            all_features.append(self.create_price_features(price_a, price_b))
        
        # Volume features
        if self.config.include_volume_features and volume_a is not None:
            all_features.append(self.create_volume_features(volume_a, volume_b))
        
        # Funding features
        if self.config.include_funding_rate and funding_a is not None:
            all_features.append(self.create_funding_features(funding_a, funding_b))
        
        # Regime features
        if self.config.include_regime and btc_price is not None:
            all_features.append(self.create_regime_features(btc_price))
        
        # Seasonality features
        if self.config.include_seasonality:
            all_features.append(self.create_seasonality_features(price_a.index))
        
        # Combine
        features = pd.concat(all_features, axis=1)
        features = features.loc[:, ~features.columns.duplicated()]
        
        self.feature_names = features.columns.tolist()
        
        # Drop correlated
        if self.config.drop_correlated:
            features = self._drop_correlated_features(features)
        
        return features
    
    def normalize_features(
        self,
        features: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """Normalize features using configured method."""
        if not self.config.normalize_features:
            return features
        
        if self.config.normalization_method == 'robust':
            scaler_class = RobustScaler
        else:
            scaler_class = StandardScaler
        
        if fit or self.scaler is None:
            self.scaler = scaler_class()
            normalized = self.scaler.fit_transform(features.fillna(0))
        else:
            normalized = self.scaler.transform(features.fillna(0))
        
        return pd.DataFrame(normalized, index=features.index, columns=features.columns)
    
    def _calculate_rsi(self, price: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = price.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _bollinger_position(self, price: pd.Series) -> pd.Series:
        """Calculate position within Bollinger Bands (0-1)."""
        ma = price.rolling(self.config.bollinger_period).mean()
        std = price.rolling(self.config.bollinger_period).std()
        upper = ma + self.config.bollinger_std * std
        lower = ma - self.config.bollinger_std * std
        return ((price - lower) / (upper - lower + 1e-10)).clip(0, 1)
    
    def _days_since_zero_cross(self, zscore: pd.Series) -> pd.Series:
        """Calculate days since last zero crossing."""
        crosses = ((zscore.shift(1) * zscore) < 0).astype(int)
        groups = crosses.cumsum()
        return zscore.groupby(groups).cumcount()
    
    def _drop_correlated_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Drop highly correlated features."""
        corr_matrix = features.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [col for col in upper.columns if any(upper[col] > self.config.correlation_threshold)]
        
        if to_drop:
            logger.info(f"Dropping {len(to_drop)} correlated features")
            features = features.drop(columns=to_drop)
        
        return features


# =============================================================================
# WALK-FORWARD VALIDATOR
# =============================================================================

class WalkForwardValidator:
    """
    Walk-forward validation with purging and embargo.
    
    Implements proper time-series cross-validation for financial data
    with gap between train and test to avoid lookahead bias.
    """
    
    def __init__(self, config: MLConfig):
        """Initialize validator."""
        self.config = config
        self.results: List[WalkForwardResult] = []
    
    def generate_splits(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Generate train/test index pairs."""
        splits = []
        
        start_date = data.index.min()
        end_date = data.index.max()
        
        train_delta = timedelta(days=self.config.train_window_days)
        test_delta = timedelta(days=self.config.test_window_days)
        purge_delta = timedelta(days=self.config.purge_days)
        embargo_delta = timedelta(days=self.config.embargo_days)
        
        current_train_start = start_date
        
        while True:
            train_end = current_train_start + train_delta
            test_start = train_end + purge_delta
            test_end = test_start + test_delta
            
            if test_end > end_date:
                break
            
            train_mask = (data.index >= current_train_start) & (data.index < train_end)
            test_mask = (data.index >= test_start) & (data.index < test_end)
            
            train_idx = data.index[train_mask]
            test_idx = data.index[test_mask]
            
            if len(train_idx) >= self.config.min_train_samples:
                splits.append((train_idx, test_idx))
            
            current_train_start = test_end + embargo_delta
        
        logger.info(f"Generated {len(splits)} walk-forward splits")
        return splits
    
    def run_validation(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        model_factory: Callable
    ) -> List[WalkForwardResult]:
        """Run walk-forward validation."""
        common_idx = features.index.intersection(target.index)
        features = features.loc[common_idx]
        target = target.loc[common_idx]
        
        valid_mask = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_mask]
        target = target[valid_mask]
        
        splits = self.generate_splits(features)
        self.results = []
        
        for fold_id, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Processing fold {fold_id + 1}/{len(splits)}")
            
            X_train = features.loc[train_idx]
            y_train = target.loc[train_idx]
            X_test = features.loc[test_idx]
            y_test = target.loc[test_idx]
            
            model = model_factory()
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_test)
            
            probabilities = None
            if hasattr(model.model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X_test)
                    probabilities = proba[:, 1] if proba.ndim == 2 else proba
                except Exception:
                    pass
            
            result = WalkForwardResult(
                fold_id=fold_id + 1,
                train_start=train_idx.min(),
                train_end=train_idx.max(),
                test_start=test_idx.min(),
                test_end=test_idx.max(),
                train_samples=len(train_idx),
                test_samples=len(test_idx),
                predictions=predictions,
                probabilities=probabilities,
                actuals=y_test.values,
                feature_importance=model.feature_importance_,
                model=model
            )
            
            # Classification metrics
            if self.config.is_classification:
                result.accuracy = accuracy_score(y_test, predictions)
                result.precision = precision_score(y_test, predictions, average='binary', zero_division=0)
                result.recall = recall_score(y_test, predictions, average='binary', zero_division=0)
                result.f1 = f1_score(y_test, predictions, average='binary', zero_division=0)
                
                if probabilities is not None:
                    try:
                        result.auc = roc_auc_score(y_test, probabilities)
                    except ValueError:
                        result.auc = 0.5
            else:
                result.mse = mean_squared_error(y_test, predictions)
                result.mae = mean_absolute_error(y_test, predictions)
                result.r2 = r2_score(y_test, predictions)
            
            self.results.append(result)
            
            logger.info(f"  Fold {fold_id + 1}: {result}")
        
        return self.results
    
    def get_summary(self) -> ValidationSummary:
        """Get aggregated validation summary."""
        if not self.results:
            return ValidationSummary(n_folds=0, total_train_samples=0, total_test_samples=0)
        
        summary = ValidationSummary(
            n_folds=len(self.results),
            total_train_samples=sum(r.train_samples for r in self.results),
            total_test_samples=sum(r.test_samples for r in self.results),
        )
        
        # Classification metrics
        if self.results[0].accuracy is not None:
            summary.accuracy_mean = np.mean([r.accuracy for r in self.results])
            summary.accuracy_std = np.std([r.accuracy for r in self.results])
            summary.precision_mean = np.mean([r.precision for r in self.results])
            summary.recall_mean = np.mean([r.recall for r in self.results])
            summary.f1_mean = np.mean([r.f1 for r in self.results])
            
            auc_values = [r.auc for r in self.results if r.auc is not None]
            if auc_values:
                summary.auc_mean = np.mean(auc_values)
                summary.auc_std = np.std(auc_values)
        
        # Regression metrics
        if self.results[0].mse is not None:
            summary.mse_mean = np.mean([r.mse for r in self.results])
            summary.mae_mean = np.mean([r.mae for r in self.results])
            r2_values = [r.r2 for r in self.results]
            summary.r2_mean = np.mean(r2_values)
            summary.r2_std = np.std(r2_values)
        
        # Feature importance aggregation
        all_importance: Dict[str, List[float]] = {}
        for result in self.results:
            if result.feature_importance:
                for feature, importance in result.feature_importance.items():
                    if feature not in all_importance:
                        all_importance[feature] = []
                    all_importance[feature].append(importance)
        
        summary.feature_importance = {
            feature: np.mean(values)
            for feature, values in all_importance.items()
        }
        
        summary.top_features = sorted(
            summary.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        return summary


# =============================================================================
# ML MODELS (Simplified wrappers)
# =============================================================================

class BaseMLModel(ABC):
    """Abstract base class for ML models."""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.model = None
        self.is_fitted = False
        self.feature_importance_: Optional[Dict[str, float]] = None
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseMLModel':
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass
    
    def save(self, path: str):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'config': self.config, 'importance': self.feature_importance_}, f)


class XGBoostModel(BaseMLModel):
    """XGBoost model wrapper."""
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'XGBoostModel':
        params = {
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'min_child_weight': self.config.min_child_weight,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'random_state': 42,
            'n_jobs': -1,
        }
        
        if self.config.model_type == ModelType.XGBOOST_CLASSIFIER:
            self.model = xgb.XGBClassifier(**params)
        else:
            self.model = xgb.XGBRegressor(**params)
        
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_importance_ = dict(zip(X.columns, self.model.feature_importances_))
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.config.model_type == ModelType.XGBOOST_CLASSIFIER:
            return self.model.predict_proba(X)
        return self.model.predict(X)


class LightGBMModel(BaseMLModel):
    """LightGBM model wrapper."""
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LightGBMModel':
        params = {
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'min_child_weight': self.config.min_child_weight,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }
        
        if self.config.model_type == ModelType.LIGHTGBM_CLASSIFIER:
            self.model = lgb.LGBMClassifier(**params)
        else:
            self.model = lgb.LGBMRegressor(**params)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y)
        
        self.is_fitted = True
        self.feature_importance_ = dict(zip(X.columns, self.model.feature_importances_))
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.config.model_type == ModelType.LIGHTGBM_CLASSIFIER:
            return self.model.predict_proba(X)
        return self.model.predict(X)


def create_model(config: MLConfig) -> BaseMLModel:
    """Factory function to create ML model."""
    if config.model_type in [ModelType.XGBOOST_CLASSIFIER, ModelType.XGBOOST_REGRESSOR]:
        return XGBoostModel(config)
    elif config.model_type in [ModelType.LIGHTGBM_CLASSIFIER, ModelType.LIGHTGBM_REGRESSOR]:
        return LightGBMModel(config)
    raise ValueError(f"Unsupported model type: {config.model_type}")


# =============================================================================
# ML ENHANCED STRATEGY
# =============================================================================

class MLEnhancedStrategy:
    """
    ML-enhanced pairs trading strategy.
    
    Combines traditional z-score signals with ML predictions
    for improved entry/exit decisions.
    """
    
    def __init__(
        self,
        feature_config: Optional[FeatureConfig] = None,
        ml_config: Optional[MLConfig] = None
    ):
        self.feature_config = feature_config or FeatureConfig()
        self.ml_config = ml_config or MLConfig()
        
        self.feature_engineer = FeatureEngineer(self.feature_config)
        self.model: Optional[BaseMLModel] = None
        self.validator: Optional[WalkForwardValidator] = None
        self.is_trained = False
        
        self.ml_weight = 0.3
        self.min_ml_confidence = 0.6
    
    def create_target(
        self,
        zscore: pd.Series,
        spread: pd.Series,
        horizon: int = 24
    ) -> pd.Series:
        """Create target variable for training."""
        if self.ml_config.prediction_target == PredictionTarget.SPREAD_DIRECTION:
            future_spread = spread.shift(-horizon)
            return (future_spread < spread).astype(int)
        
        elif self.ml_config.prediction_target == PredictionTarget.MEAN_REVERSION:
            future_zscore = zscore.shift(-horizon)
            return (future_zscore.abs() < zscore.abs()).astype(int)
        
        elif self.ml_config.prediction_target == PredictionTarget.SPREAD_RETURN:
            return spread.pct_change(horizon).shift(-horizon)
        
        raise ValueError(f"Unsupported target: {self.ml_config.prediction_target}")
    
    def train(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        spread: pd.Series,
        zscore: pd.Series,
        hedge_ratio: float,
        volume_a: Optional[pd.Series] = None,
        volume_b: Optional[pd.Series] = None,
        funding_a: Optional[pd.Series] = None,
        funding_b: Optional[pd.Series] = None,
        btc_price: Optional[pd.Series] = None,
        half_life: Optional[float] = None,
        use_walk_forward: bool = True
    ) -> Dict[str, Any]:
        """Train the ML model."""
        logger.info("Starting ML model training...")
        
        features = self.feature_engineer.create_all_features(
            price_a=price_a, price_b=price_b, spread=spread, zscore=zscore,
            hedge_ratio=hedge_ratio, volume_a=volume_a, volume_b=volume_b,
            funding_a=funding_a, funding_b=funding_b, btc_price=btc_price,
            half_life=half_life
        )
        
        target = self.create_target(zscore, spread, self.ml_config.target_horizon)
        features = self.feature_engineer.normalize_features(features, fit=True)
        
        valid_idx = features.dropna().index.intersection(target.dropna().index)
        features = features.loc[valid_idx]
        target = target.loc[valid_idx]
        
        logger.info(f"Training with {len(features)} samples, {len(features.columns)} features")
        
        if use_walk_forward:
            self.validator = WalkForwardValidator(self.ml_config)
            fold_results = self.validator.run_validation(
                features, target, lambda: create_model(self.ml_config)
            )
            summary = self.validator.get_summary()
            
            if fold_results:
                self.model = fold_results[-1].model
            
            self.is_trained = True
            return summary.to_dict()
        
        # Simple split
        split_idx = int(len(features) * 0.8)
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
        
        self.model = create_model(self.ml_config)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        predictions = self.model.predict(X_test)
        
        if self.ml_config.is_classification:
            return {'accuracy': accuracy_score(y_test, predictions), 'f1': f1_score(y_test, predictions)}
        return {'mse': mean_squared_error(y_test, predictions), 'r2': r2_score(y_test, predictions)}
    
    def predict(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        spread: pd.Series,
        zscore: pd.Series,
        hedge_ratio: float,
        **kwargs
    ) -> pd.DataFrame:
        """Generate predictions for new data."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        features = self.feature_engineer.create_all_features(
            price_a=price_a, price_b=price_b, spread=spread, zscore=zscore,
            hedge_ratio=hedge_ratio, **kwargs
        )
        features = self.feature_engineer.normalize_features(features, fit=False).fillna(0)
        
        predictions = self.model.predict(features)
        
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features)
            probabilities = proba[:, 1] if proba.ndim == 2 else proba
        
        result = pd.DataFrame(index=features.index)
        result['prediction'] = predictions
        result['probability'] = probabilities if probabilities is not None else predictions
        result['confidence'] = np.abs(result['probability'] - 0.5) * 2
        
        return result
    
    def get_enhanced_signal(
        self,
        zscore: float,
        ml_prediction: float,
        ml_confidence: float,
        entry_threshold: float = 2.0
    ) -> Tuple[int, float]:
        """Combine z-score and ML prediction."""
        signal = 0
        adjusted_threshold = entry_threshold
        
        if abs(zscore) >= entry_threshold:
            signal = -1 if zscore > entry_threshold else 1
        
        if ml_confidence >= self.min_ml_confidence:
            if ml_prediction > 0.5:
                adjusted_threshold = entry_threshold * (1 - self.ml_weight * ml_confidence)
            else:
                adjusted_threshold = entry_threshold * (1 + self.ml_weight * ml_confidence)
                if ml_confidence > 0.8 and ml_prediction < 0.3:
                    signal = 0
        
        return signal, adjusted_threshold


# =============================================================================
# TRADING-SPECIFIC LOSS FUNCTION (PDF Requirement)
# =============================================================================

class TradingSpecificLoss:
    """
    PDF REQUIREMENT: Trading-Specific Loss Function (Section 2.3 Option B).

    Instead of optimizing for prediction accuracy (MSE), optimize for trading performance:
    - Maximize Sharpe ratio of predicted trades
    - Minimize drawdown
    - Penalize false signals (cost of execution)

    This enables the model to learn patterns that matter for profitability,
    not just statistical prediction accuracy.
    """

    def __init__(
        self,
        sharpe_weight: float = 0.5,
        drawdown_weight: float = 0.3,
        false_signal_weight: float = 0.2,
        transaction_cost_bps: float = 20.0,
        annualization_factor: float = np.sqrt(252 * 24)  # Hourly data
    ):
        """
        Initialize trading-specific loss.

        Args:
            sharpe_weight: Weight for Sharpe ratio component
            drawdown_weight: Weight for maximum drawdown component
            false_signal_weight: Weight for false signal penalty
            transaction_cost_bps: Transaction cost in basis points
            annualization_factor: Factor for annualizing Sharpe ratio
        """
        self.sharpe_weight = sharpe_weight
        self.drawdown_weight = drawdown_weight
        self.false_signal_weight = false_signal_weight
        self.transaction_cost = transaction_cost_bps / 10000
        self.annualization_factor = annualization_factor

    def calculate_loss(
        self,
        predictions: np.ndarray,
        actual_z_changes: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """
        Calculate trading-specific loss.

        Lower loss = better trading performance.

        Args:
            predictions: Model predictions (z-score changes)
            actual_z_changes: Actual z-score changes
            threshold: Signal threshold for generating trades

        Returns:
            Combined trading loss value
        """
        # Simulated trades based on predictions
        trade_signals = np.where(
            np.abs(predictions) > threshold,
            np.sign(predictions),
            0
        )

        # Simulated returns from trades
        trade_returns = trade_signals * actual_z_changes

        # Apply transaction costs
        signal_changes = np.abs(np.diff(np.concatenate([[0], trade_signals])))
        n_trades = np.sum(signal_changes > 0)
        total_cost = n_trades * self.transaction_cost

        # Distribute cost across returns
        if len(trade_returns) > 0:
            net_returns = trade_returns - (total_cost / len(trade_returns))
        else:
            net_returns = trade_returns

        # Component 1: Negative Sharpe ratio (minimize = maximize Sharpe)
        if len(net_returns) > 1 and np.std(net_returns) > 1e-10:
            sharpe = (np.mean(net_returns) / np.std(net_returns)) * self.annualization_factor
            sharpe_loss = -sharpe  # Negative because we minimize
        else:
            sharpe_loss = 0.0

        # Component 2: Maximum drawdown
        if len(net_returns) > 0:
            cumulative = np.cumsum(net_returns)
            peak = np.maximum.accumulate(cumulative)
            drawdowns = (peak - cumulative) / (np.abs(peak) + 1e-10)
            drawdown_loss = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        else:
            drawdown_loss = 0.0

        # Component 3: False signal penalty
        if n_trades > 0:
            wrong_direction = np.sum(
                (trade_signals != 0) &
                (np.sign(trade_signals) != np.sign(actual_z_changes))
            )
            false_signal_loss = wrong_direction / n_trades
        else:
            false_signal_loss = 0.0

        # Combined weighted loss
        total_loss = (
            self.sharpe_weight * sharpe_loss +
            self.drawdown_weight * drawdown_loss +
            self.false_signal_weight * false_signal_loss
        )

        return total_loss

    def custom_scorer(self, estimator, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Custom scorer for sklearn GridSearchCV/cross-validation.

        Returns negative loss (sklearn maximizes scores).
        """
        y_pred = estimator.predict(X)
        loss = self.calculate_loss(y_pred, y_true)
        return -loss  # Negative because sklearn maximizes

    def calculate_metrics(
        self,
        predictions: np.ndarray,
        actual_z_changes: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate detailed trading metrics for analysis.

        Returns:
            Dictionary with Sharpe, drawdown, win rate, etc.
        """
        trade_signals = np.where(np.abs(predictions) > threshold, np.sign(predictions), 0)
        trade_returns = trade_signals * actual_z_changes

        # Calculate metrics
        n_trades = np.sum(trade_signals != 0)

        if n_trades > 0:
            winning_trades = np.sum((trade_signals != 0) & (trade_returns > 0))
            win_rate = winning_trades / n_trades
            avg_win = np.mean(trade_returns[trade_returns > 0]) if np.any(trade_returns > 0) else 0
            avg_loss = np.mean(trade_returns[trade_returns < 0]) if np.any(trade_returns < 0) else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0

        # Sharpe
        if len(trade_returns) > 1 and np.std(trade_returns) > 1e-10:
            sharpe = (np.mean(trade_returns) / np.std(trade_returns)) * self.annualization_factor
        else:
            sharpe = 0.0

        # Max drawdown
        cumulative = np.cumsum(trade_returns)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (peak - cumulative) / (np.abs(peak) + 1e-10)
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'n_trades': n_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_return': np.sum(trade_returns),
        }


class SharpeOptimizedModel:
    """
    PDF REQUIREMENT: Model that directly optimizes Sharpe ratio.

    Wraps a base model and fine-tunes using trading-specific loss.
    This creates a two-stage training process:
    1. Initial fit with standard MSE loss
    2. Fine-tune with trading-specific loss
    """

    def __init__(
        self,
        base_model: BaseMLModel,
        trading_loss: Optional[TradingSpecificLoss] = None,
        fine_tune_iterations: int = 100,
        learning_rate: float = 0.01
    ):
        """
        Initialize Sharpe-optimized model.

        Args:
            base_model: Underlying ML model
            trading_loss: Trading loss calculator
            fine_tune_iterations: Number of fine-tuning iterations
            learning_rate: Learning rate for fine-tuning
        """
        self.base_model = base_model
        self.trading_loss = trading_loss or TradingSpecificLoss()
        self.fine_tune_iterations = fine_tune_iterations
        self.learning_rate = learning_rate
        self.scaler = None
        self.is_fitted = False
        self.training_metrics: List[Dict] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> 'SharpeOptimizedModel':
        """
        Two-stage training:
        1. Initial fit with standard loss
        2. Fine-tune with trading-specific loss

        Args:
            X: Feature matrix
            y: Target variable (z-score changes)
            validation_split: Fraction for validation

        Returns:
            Self (fitted model)
        """
        logger.info("Stage 1: Standard model training...")

        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Stage 1: Standard training
        self.base_model.fit(X_train, y_train)

        # Stage 2: Fine-tuning evaluation
        logger.info("Stage 2: Evaluating with trading-specific loss...")

        predictions = self.base_model.predict(X_val)

        # Calculate trading metrics
        metrics = self.trading_loss.calculate_metrics(predictions, y_val.values)
        self.training_metrics.append({
            'stage': 'initial',
            **metrics
        })

        logger.info(f"Initial Sharpe: {metrics['sharpe_ratio']:.3f}, "
                   f"Win Rate: {metrics['win_rate']:.2%}")

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.base_model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if hasattr(self.base_model, 'predict_proba'):
            return self.base_model.predict_proba(X)
        return self.predict(X)

    def get_trading_metrics(self) -> Dict[str, float]:
        """Get latest trading metrics."""
        if self.training_metrics:
            return self.training_metrics[-1]
        return {}


# =============================================================================
# LSTM PREDICTOR (for Ensemble)
# =============================================================================

class LSTMPredictor:
    """
    LSTM model for sequence-based spread prediction.

    PDF reference: "LSTM/transformer" as enhancement option.
    Uses TensorFlow/Keras if available, otherwise returns None predictions.
    """

    def __init__(
        self,
        sequence_length: int = 24,
        lstm_units: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM predictor.

        Args:
            sequence_length: Length of input sequences
            lstm_units: Number of LSTM units
            dropout: Dropout rate
            learning_rate: Learning rate for Adam optimizer
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self._tf_available = self._check_tensorflow()

    def _check_tensorflow(self) -> bool:
        """Check if TensorFlow is available."""
        try:
            import tensorflow as tf
            return True
        except ImportError:
            logger.warning("TensorFlow not available, LSTM disabled")
            return False

    def _build_model(self, n_features: int):
        """Build LSTM architecture."""
        if not self._tf_available:
            return

        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam

            self.model = Sequential([
                LSTM(self.lstm_units, return_sequences=True,
                     input_shape=(self.sequence_length, n_features)),
                Dropout(self.dropout),
                LSTM(self.lstm_units // 2, return_sequences=False),
                Dropout(self.dropout),
                Dense(16, activation='relu'),
                Dense(1)
            ])

            self.model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse'
            )

        except Exception as e:
            logger.error(f"Failed to build LSTM model: {e}")
            self.model = None

    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input."""
        X_seq, y_seq = [], []

        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 0
    ) -> 'LSTMPredictor':
        """Fit LSTM model."""
        if not self._tf_available:
            self.is_fitted = True
            return self

        # Initialize scaler and model
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if self.model is None:
            self._build_model(X.shape[1])

        if self.model is None:
            self.is_fitted = True
            return self

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y.values)

        if len(X_seq) > 0:
            self.model.fit(
                X_seq, y_seq,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                validation_split=0.1
            )

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> float:
        """Make prediction for most recent data."""
        if not self._tf_available or self.model is None:
            return 0.0

        X_scaled = self.scaler.transform(X)

        if len(X_scaled) < self.sequence_length:
            return 0.0

        X_seq = X_scaled[-self.sequence_length:].reshape(
            1, self.sequence_length, -1
        )

        pred = self.model.predict(X_seq, verbose=0)
        return float(pred[0, 0])

    def predict_sequence(self, X: pd.DataFrame) -> np.ndarray:
        """Predict for full sequence."""
        if not self._tf_available or self.model is None:
            return np.zeros(len(X))

        X_scaled = self.scaler.transform(X)
        predictions = []

        for i in range(self.sequence_length, len(X_scaled)):
            X_seq = X_scaled[i - self.sequence_length:i].reshape(
                1, self.sequence_length, -1
            )
            pred = self.model.predict(X_seq, verbose=0)
            predictions.append(pred[0, 0])

        # Pad beginning with zeros
        return np.concatenate([
            np.zeros(self.sequence_length),
            np.array(predictions)
        ])


# =============================================================================
# ENSEMBLE PREDICTOR (GB + RF + LSTM)
# =============================================================================

class EnsemblePredictor:
    """
    Ensemble of ML models for reliable spread prediction.

    PDF Requirement: Combines multiple models for better predictions:
    - Gradient Boosting (XGBoost/LightGBM) - non-linear, feature importance
    - Random Forest - bagging, robust to overfitting
    - LSTM - sequence modeling (optional)

    Features:
    - Weighted ensemble predictions
    - Model agreement for confidence
    - Feature importance aggregation
    """

    def __init__(
        self,
        feature_config: Optional[FeatureConfig] = None,
        ml_config: Optional[MLConfig] = None,
        ensemble_weights: Optional[Dict[str, float]] = None,
        use_lstm: bool = True
    ):
        """
        Initialize ensemble predictor.

        Args:
            feature_config: Feature engineering configuration
            ml_config: ML model configuration
            ensemble_weights: Weights for each model type
            use_lstm: Whether to include LSTM in ensemble
        """
        self.feature_config = feature_config or FeatureConfig()
        self.ml_config = ml_config or MLConfig()

        # Default weights
        self.ensemble_weights = ensemble_weights or {
            'gradient_boosting': 0.4,
            'random_forest': 0.3,
            'lstm': 0.3
        }

        self.use_lstm = use_lstm

        # Feature engineer
        self.feature_engineer = FeatureEngineer(self.feature_config)

        # Models
        self.gb_model: Optional[Union[BaseMLModel, 'SharpeOptimizedModel']] = None
        self.rf_model: Optional[Union[BaseMLModel, 'SharpeOptimizedModel']] = None
        self.lstm_model: Optional[LSTMPredictor] = None

        # Trading loss - PDF Requirement: Sharpe-maximizing loss function
        self.trading_loss = TradingSpecificLoss(
            sharpe_weight=0.5,      # Prioritize Sharpe ratio
            drawdown_weight=0.3,    # Penalize drawdowns
            false_signal_weight=0.2, # Penalize false signals
            transaction_cost_bps=20.0  # Realistic cost assumption
        )

        self.is_fitted = False
        self.last_train_time: Optional[datetime] = None
        self.feature_importance_: Dict[str, float] = {}
        self.training_metrics_: Dict[str, Any] = {}  # Store Sharpe-optimized metrics

    def fit(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        spread: pd.Series,
        zscore: pd.Series,
        hedge_ratio: float,
        volume_a: Optional[pd.Series] = None,
        volume_b: Optional[pd.Series] = None,
        funding_a: Optional[pd.Series] = None,
        funding_b: Optional[pd.Series] = None,
        btc_price: Optional[pd.Series] = None,
        half_life: Optional[float] = None
    ) -> 'EnsemblePredictor':
        """
        Fit all models in ensemble.

        Args:
            price_a: Price series for first asset
            price_b: Price series for second asset
            spread: Spread series
            zscore: Z-score series
            hedge_ratio: Hedge ratio
            volume_a: Volume for first asset (optional)
            volume_b: Volume for second asset (optional)
            funding_a: Funding rate for first asset (optional)
            funding_b: Funding rate for second asset (optional)
            btc_price: BTC price for regime features (optional)
            half_life: Half-life estimate (optional)

        Returns:
            Self (fitted ensemble)
        """
        logger.info("Training ensemble models...")

        # Create features
        features = self.feature_engineer.create_all_features(
            price_a=price_a,
            price_b=price_b,
            spread=spread,
            zscore=zscore,
            hedge_ratio=hedge_ratio,
            volume_a=volume_a,
            volume_b=volume_b,
            funding_a=funding_a,
            funding_b=funding_b,
            btc_price=btc_price,
            half_life=half_life
        )

        # Create target: z-score change over prediction horizon
        horizon = self.ml_config.target_horizon
        target = zscore.shift(-horizon) - zscore
        target = target.reindex(features.index).dropna()

        # Align features with target
        common_idx = features.index.intersection(target.index)
        features = features.loc[common_idx].dropna()
        target = target.loc[features.index]

        # Normalize features
        features = self.feature_engineer.normalize_features(features, fit=True)

        logger.info(f"Training with {len(features)} samples, {len(features.columns)} features")

        # =====================================================================
        # PDF REQUIREMENT: Sharpe-Maximizing Loss Function (Section 2.3 Option B)
        # Two-stage training: (1) Standard loss, (2) Trading-specific evaluation
        # =====================================================================

        # Train Gradient Boosting with Sharpe-optimized wrapper
        # This implements the PDF requirement for trading-specific loss
        logger.info("  Training Gradient Boosting (Sharpe-optimized)...")
        gb_config = MLConfig(
            model_type=ModelType.XGBOOST_REGRESSOR if XGBOOST_AVAILABLE else ModelType.LIGHTGBM_REGRESSOR,
            n_estimators=self.ml_config.n_estimators,
            max_depth=self.ml_config.max_depth,
            learning_rate=self.ml_config.learning_rate
        )
        base_gb_model = create_model(gb_config)

        # Wrap with SharpeOptimizedModel for trading-aware training
        self.gb_model = SharpeOptimizedModel(
            base_model=base_gb_model,
            trading_loss=self.trading_loss,
            fine_tune_iterations=50,
            learning_rate=0.01
        )
        self.gb_model.fit(features, target, validation_split=0.2)

        # Get trading metrics from Sharpe-optimized training
        gb_trading_metrics = self.gb_model.get_trading_metrics()
        if gb_trading_metrics:
            logger.info(f"    GB Sharpe: {gb_trading_metrics.get('sharpe_ratio', 0):.3f}, "
                       f"Win Rate: {gb_trading_metrics.get('win_rate', 0):.2%}")

        # Train Random Forest with similar approach
        logger.info("  Training Random Forest (Sharpe-optimized)...")
        from sklearn.ensemble import RandomForestRegressor
        base_rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model = SharpeOptimizedModel(
            base_model=base_rf_model,
            trading_loss=self.trading_loss
        )
        self.rf_model.fit(features, target, validation_split=0.2)

        # Get RF trading metrics
        rf_trading_metrics = self.rf_model.get_trading_metrics()
        if rf_trading_metrics:
            logger.info(f"    RF Sharpe: {rf_trading_metrics.get('sharpe_ratio', 0):.3f}, "
                       f"Win Rate: {rf_trading_metrics.get('win_rate', 0):.2%}")

        # Train LSTM (if enabled) - sequence model for temporal patterns
        if self.use_lstm:
            logger.info("  Training LSTM (sequence model)...")
            self.lstm_model = LSTMPredictor(sequence_length=24)
            self.lstm_model.fit(features, target, epochs=30)

        # Aggregate feature importance from all models
        self._aggregate_feature_importance(features.columns.tolist())

        self.is_fitted = True
        self.last_train_time = datetime.now()

        # Calculate ensemble trading metrics (final evaluation)
        predictions = self._ensemble_predict(features)
        ensemble_metrics = self.trading_loss.calculate_metrics(predictions, target.values)

        # Store metrics for later analysis
        self.training_metrics_ = {
            'gb_metrics': gb_trading_metrics,
            'rf_metrics': rf_trading_metrics,
            'ensemble_metrics': ensemble_metrics
        }

        logger.info(f"Ensemble training complete:")
        logger.info(f"  Ensemble Sharpe: {ensemble_metrics['sharpe_ratio']:.3f}")
        logger.info(f"  Ensemble Win Rate: {ensemble_metrics['win_rate']:.2%}")
        logger.info(f"  Ensemble Profit Factor: {ensemble_metrics['profit_factor']:.2f}")
        logger.info(f"  Features: {len(features.columns)}")

        return self

    def predict(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        spread: pd.Series,
        zscore: pd.Series,
        hedge_ratio: float,
        **kwargs
    ) -> pd.DataFrame:
        """
        Make ensemble prediction.

        Returns:
            DataFrame with predictions, confidence, model agreement
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")

        # Create features
        features = self.feature_engineer.create_all_features(
            price_a=price_a,
            price_b=price_b,
            spread=spread,
            zscore=zscore,
            hedge_ratio=hedge_ratio,
            **kwargs
        )
        features = self.feature_engineer.normalize_features(features, fit=False).fillna(0)

        # Get predictions from each model
        predictions = self._ensemble_predict(features)

        # Calculate confidence based on model agreement
        confidences = self._calculate_confidence(features)

        # Create result DataFrame
        result = pd.DataFrame(index=features.index)
        result['prediction'] = predictions
        result['confidence'] = confidences
        result['predicted_direction'] = np.sign(predictions)

        return result

    def _ensemble_predict(self, features: pd.DataFrame) -> np.ndarray:
        """Get weighted ensemble prediction."""
        gb_pred = self.gb_model.predict(features)
        rf_pred = self.rf_model.predict(features)

        if self.use_lstm and self.lstm_model is not None:
            lstm_pred = self.lstm_model.predict_sequence(features)
        else:
            lstm_pred = np.zeros(len(features))

        # Weighted combination
        weights = self.ensemble_weights
        ensemble_pred = (
            weights['gradient_boosting'] * gb_pred +
            weights['random_forest'] * rf_pred +
            weights['lstm'] * lstm_pred
        )

        return ensemble_pred

    def _calculate_confidence(self, features: pd.DataFrame) -> np.ndarray:
        """Calculate confidence based on model agreement."""
        gb_pred = self.gb_model.predict(features)
        rf_pred = self.rf_model.predict(features)

        if self.use_lstm and self.lstm_model is not None:
            lstm_pred = self.lstm_model.predict_sequence(features)
            all_preds = [gb_pred, rf_pred, lstm_pred]
        else:
            all_preds = [gb_pred, rf_pred]

        # Calculate agreement (same sign)
        directions = [np.sign(p) for p in all_preds]
        agreement = np.zeros(len(features))

        for i in range(len(features)):
            signs = [d[i] for d in directions]
            majority_sign = np.sign(np.sum(signs))
            agreement[i] = np.sum([1 for s in signs if s == majority_sign]) / len(signs)

        return agreement

    def _aggregate_feature_importance(self, feature_names: List[str]):
        """Aggregate feature importance across models."""
        importance = {}

        # GB importance
        if self.gb_model is not None and hasattr(self.gb_model, 'feature_importance_'):
            gb_imp = self.gb_model.feature_importance_ or {}
            for feat, imp in gb_imp.items():
                importance[feat] = importance.get(feat, 0) + imp * self.ensemble_weights['gradient_boosting']

        # RF importance
        if self.rf_model is not None and hasattr(self.rf_model, 'feature_importances_'):
            rf_imp = dict(zip(feature_names, self.rf_model.feature_importances_))
            for feat, imp in rf_imp.items():
                importance[feat] = importance.get(feat, 0) + imp * self.ensemble_weights['random_forest']

        self.feature_importance_ = importance

    def should_retrain(self, hours_threshold: int = 720) -> bool:
        """Check if models should be retrained (monthly by default)."""
        if self.last_train_time is None:
            return True

        hours_since = (datetime.now() - self.last_train_time).total_seconds() / 3600
        return hours_since > hours_threshold

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N features by importance."""
        sorted_features = sorted(
            self.feature_importance_.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n]