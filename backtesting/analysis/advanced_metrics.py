"""
Full Performance Metrics Calculator - PDF Section 2.4 Complete Implementation
==================================================================================

Comprehensive metrics calculator implementing 60+ performance metrics for
comprehensive backtesting analysis per project specification.

Metric Categories:
1. Core Risk-Adjusted Returns (Sharpe, Sortino, Calmar, Omega, etc.)
2. Drawdown Analysis (Max DD, Duration, Recovery, Ulcer Index)
3. Trade Statistics (Win Rate, Profit Factor, Expectancy)
4. Cost Analysis (Transaction Cost Drag, Gas Impact, MEV Costs)
5. Venue-Specific Breakdown (CEX, DEX, Hybrid, Mixed, Combined)
6. Regime-Specific Analysis (Bull, Bear, Sideways, High/Low Vol, Crisis)
7. Time-Based Analysis (Hourly, Daily, Weekly, Monthly patterns)
8. Statistical Tests (Normality, Stationarity, Autocorrelation)
9. Factor Exposures (BTC Beta, Market Correlation)
10. Risk Metrics (VaR, CVaR, Expected Shortfall)
11. Execution Quality (Slippage, Fill Rate, Timing)
12. Rolling Window Analysis (30d, 90d, 180d, 365d)

PDF Section 2.4 REQUIRED Metrics:
- Sharpe ratio (daily returns basis)
- Sortino ratio
- Maximum drawdown
- Average holding period
- Turnover (rebalancing frequency)
- Transaction cost drag (% of gross returns)
- Gas cost impact (for DEX pairs)

Author: Tamer Atesyakar
Version: 2.0.0 - Complete
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import logging
from scipy import stats
from collections import defaultdict
import warnings

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS - PDF Section 2.4 Compliant
# =============================================================================

# Trading calendar (Crypto = 24/7/365)
TRADING_DAYS_PER_YEAR = 365
TRADING_HOURS_PER_DAY = 24
TRADING_HOURS_PER_YEAR = TRADING_DAYS_PER_YEAR * TRADING_HOURS_PER_DAY

# Default risk-free rate (US Treasury)
DEFAULT_RISK_FREE_RATE = 0.045  # 4.5% as of 2024

# Annualization factors
ANNUALIZATION_DAILY = np.sqrt(365)
ANNUALIZATION_HOURLY = np.sqrt(365 * 24)

# VaR confidence levels
VAR_CONFIDENCE_LEVELS = [0.95, 0.99, 0.999]

# Rolling window sizes (days)
ROLLING_WINDOWS = {
    'short': 30,
    'medium': 90,
    'long': 180,
    'yearly': 365
}

# Regime thresholds
REGIME_THRESHOLDS = {
    'bull_threshold': 0.20,      # +20% from recent low
    'bear_threshold': -0.20,     # -20% from recent high
    'high_vol_threshold': 0.80,  # 80th percentile volatility
    'low_vol_threshold': 0.20,   # 20th percentile volatility
}

# Venue cost parameters (PDF Section 2.4)
VENUE_COSTS = {
    'cex': {
        'fee_bps': 5,           # 0.05% per side
        'slippage_bps': 2,      # 0.02% typical
        'gas_cost': 0,
    },
    'dex': {
        'fee_bps': 30,          # 0.30% typical AMM
        'slippage_bps': 50,     # 0.50% typical
        'gas_cost_eth': 15.0,   # $15 typical Ethereum
        'mev_bps': 10,          # 0.10% MEV
    },
    'dex_l2': {
        'fee_bps': 30,
        'slippage_bps': 30,
        'gas_cost_eth': 0.50,   # $0.50 L2
        'mev_bps': 5,
    }
}


# =============================================================================
# ENUMS
# =============================================================================

class MetricCategory(Enum):
    """Categories of metrics for organization."""
    CORE = "core"
    RISK_ADJUSTED = "risk_adjusted"
    DRAWDOWN = "drawdown"
    TRADE_STATS = "trade_statistics"
    COST_ANALYSIS = "cost_analysis"
    VENUE_SPECIFIC = "venue_specific"
    REGIME_SPECIFIC = "regime_specific"
    TIME_BASED = "time_based"
    STATISTICAL = "statistical"
    FACTOR_EXPOSURE = "factor_exposure"
    EXECUTION = "execution_quality"
    ROLLING = "rolling_window"


class VenueType(Enum):
    """Venue types for breakdown analysis."""
    CEX = "cex"
    DEX = "dex"
    DEX_L2 = "dex_l2"
    HYBRID = "hybrid"
    MIXED = "mixed"
    COMBINED = "combined"


class RegimeType(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class TimeFrame(Enum):
    """Time frames for analysis."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


# =============================================================================
# DATA CLASSES - Comprehensive Metric Containers
# =============================================================================

@dataclass
class CoreMetrics:
    """
    Core performance metrics - PDF Section 2.4 REQUIRED.
    """
    # PDF REQUIRED metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_holding_period_hours: float = 0.0
    annualized_turnover: float = 0.0
    transaction_cost_drag_pct: float = 0.0
    gas_cost_impact_pct: float = 0.0

    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    cumulative_return: float = 0.0
    cagr: float = 0.0

    # Volatility metrics
    annualized_volatility: float = 0.0
    downside_volatility: float = 0.0
    upside_volatility: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class RiskAdjustedMetrics:
    """
    Comprehensive risk-adjusted return metrics.
    """
    # Primary ratios
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Secondary ratios
    omega_ratio: float = 0.0
    treynor_ratio: float = 0.0
    information_ratio: float = 0.0

    # Extended ratios
    burke_ratio: float = 0.0
    sterling_ratio: float = 0.0
    kappa_3_ratio: float = 0.0

    # Tail risk ratios
    tail_ratio: float = 0.0
    gain_to_pain_ratio: float = 0.0
    pain_ratio: float = 0.0

    # Common sense ratio
    common_sense_ratio: float = 0.0
    cpc_index: float = 0.0

    # Serenity metrics
    serenity_index: float = 0.0
    ulcer_performance_index: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class DrawdownMetrics:
    """
    Comprehensive drawdown analysis metrics.
    """
    # Primary drawdown metrics
    max_drawdown: float = 0.0
    max_drawdown_duration_days: float = 0.0
    max_drawdown_start: Optional[datetime] = None
    max_drawdown_end: Optional[datetime] = None
    max_drawdown_recovery: Optional[datetime] = None

    # Average drawdown metrics
    avg_drawdown: float = 0.0
    avg_drawdown_duration_days: float = 0.0

    # Drawdown distribution
    drawdown_5th_percentile: float = 0.0
    drawdown_25th_percentile: float = 0.0
    drawdown_50th_percentile: float = 0.0
    drawdown_75th_percentile: float = 0.0
    drawdown_95th_percentile: float = 0.0

    # Recovery metrics
    recovery_factor: float = 0.0
    avg_recovery_time_days: float = 0.0
    longest_recovery_days: float = 0.0

    # Risk indices
    ulcer_index: float = 0.0
    pain_index: float = 0.0
    martin_ratio: float = 0.0

    # Drawdown counts
    num_drawdowns: int = 0
    num_drawdowns_gt_5pct: int = 0
    num_drawdowns_gt_10pct: int = 0
    num_drawdowns_gt_20pct: int = 0

    # Time underwater
    pct_time_underwater: float = 0.0
    longest_underwater_days: float = 0.0

    def to_dict(self) -> Dict:
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, datetime):
                result[k] = v.isoformat() if v else None
            else:
                result[k] = v
        return result


@dataclass
class TradeStatistics:
    """
    Comprehensive trade-level statistics.
    """
    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0

    # Win/loss rates
    win_rate: float = 0.0
    loss_rate: float = 0.0

    # P&L metrics
    avg_trade_return: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0

    # Best/worst trades
    best_trade: float = 0.0
    worst_trade: float = 0.0
    best_trade_date: Optional[datetime] = None
    worst_trade_date: Optional[datetime] = None

    # Profit metrics
    profit_factor: float = 0.0
    payoff_ratio: float = 0.0
    expectancy: float = 0.0
    expectancy_ratio: float = 0.0

    # Kelly criterion
    kelly_criterion: float = 0.0
    half_kelly: float = 0.0

    # Streak analysis
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_consecutive_wins: float = 0.0
    avg_consecutive_losses: float = 0.0
    current_streak: int = 0

    # Trade duration
    avg_trade_duration_hours: float = 0.0
    median_trade_duration_hours: float = 0.0
    min_trade_duration_hours: float = 0.0
    max_trade_duration_hours: float = 0.0
    avg_winning_duration_hours: float = 0.0
    avg_losing_duration_hours: float = 0.0

    # Trade size metrics
    avg_position_size_usd: float = 0.0
    median_position_size_usd: float = 0.0
    max_position_size_usd: float = 0.0
    total_volume_traded_usd: float = 0.0

    # Time-based trade analysis
    trades_per_day: float = 0.0
    trades_per_week: float = 0.0
    trades_per_month: float = 0.0

    def to_dict(self) -> Dict:
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, datetime):
                result[k] = v.isoformat() if v else None
            else:
                result[k] = v
        return result


@dataclass
class CostMetrics:
    """
    Comprehensive cost analysis metrics.
    """
    # Total costs
    total_costs_usd: float = 0.0
    total_costs_bps: float = 0.0

    # Cost breakdown
    total_fees_usd: float = 0.0
    total_slippage_usd: float = 0.0
    total_gas_costs_usd: float = 0.0
    total_mev_costs_usd: float = 0.0
    total_funding_costs_usd: float = 0.0

    # Cost ratios
    fees_as_pct_of_gross: float = 0.0
    slippage_as_pct_of_gross: float = 0.0
    gas_as_pct_of_gross: float = 0.0
    mev_as_pct_of_gross: float = 0.0

    # Transaction cost drag (PDF REQUIRED)
    transaction_cost_drag_pct: float = 0.0
    gas_cost_impact_pct: float = 0.0  # DEX specific

    # Cost per trade
    avg_cost_per_trade_usd: float = 0.0
    avg_cost_per_trade_bps: float = 0.0

    # P&L impact
    gross_pnl_usd: float = 0.0
    net_pnl_usd: float = 0.0
    cost_adjusted_return: float = 0.0

    # Venue-specific costs
    cex_avg_cost_bps: float = 0.0
    dex_avg_cost_bps: float = 0.0
    dex_l2_avg_cost_bps: float = 0.0

    # Gas analysis (DEX)
    avg_gas_cost_usd: float = 0.0
    max_gas_cost_usd: float = 0.0
    gas_cost_volatility: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class CapacityMetrics:
    """
    Strategy capacity analysis metrics.
    """
    # Estimated capacity
    estimated_capacity_usd: float = 0.0
    capacity_utilization_pct: float = 0.0

    # Market impact
    avg_market_impact_bps: float = 0.0
    market_impact_at_1m: float = 0.0
    market_impact_at_5m: float = 0.0
    market_impact_at_10m: float = 0.0

    # Liquidity metrics
    avg_daily_volume_usd: float = 0.0
    volume_participation_rate: float = 0.0

    # Capacity by venue
    cex_capacity_usd: float = 0.0
    dex_capacity_usd: float = 0.0
    combined_capacity_usd: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class RiskMetrics:
    """
    Comprehensive risk metrics including VaR and stress tests.
    """
    # Value at Risk
    var_95: float = 0.0
    var_99: float = 0.0
    var_999: float = 0.0

    # Conditional VaR (Expected Shortfall)
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    cvar_999: float = 0.0

    # Parametric VaR
    parametric_var_95: float = 0.0
    parametric_var_99: float = 0.0

    # Cornish-Fisher VaR (skew/kurtosis adjusted)
    cornish_fisher_var_95: float = 0.0
    cornish_fisher_var_99: float = 0.0

    # Beta and correlation
    btc_beta: float = 0.0
    eth_beta: float = 0.0
    market_correlation: float = 0.0

    # Tail risk
    left_tail_ratio: float = 0.0
    right_tail_ratio: float = 0.0
    tail_dependence: float = 0.0

    # Higher moments
    skewness: float = 0.0
    kurtosis: float = 0.0
    excess_kurtosis: float = 0.0

    # Risk of ruin
    risk_of_ruin_10pct: float = 0.0
    risk_of_ruin_20pct: float = 0.0
    risk_of_ruin_50pct: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class StatisticalMetrics:
    """
    Statistical test results and distribution metrics.
    """
    # Normality tests
    jarque_bera_statistic: float = 0.0
    jarque_bera_pvalue: float = 0.0
    shapiro_wilk_statistic: float = 0.0
    shapiro_wilk_pvalue: float = 0.0
    is_normal_5pct: bool = False

    # Stationarity tests
    adf_statistic: float = 0.0
    adf_pvalue: float = 0.0
    kpss_statistic: float = 0.0
    kpss_pvalue: float = 0.0
    is_stationary: bool = False

    # Autocorrelation
    autocorr_lag1: float = 0.0
    autocorr_lag5: float = 0.0
    autocorr_lag10: float = 0.0
    autocorr_lag20: float = 0.0
    ljung_box_statistic: float = 0.0
    ljung_box_pvalue: float = 0.0

    # Distribution metrics
    mean: float = 0.0
    median: float = 0.0
    std_dev: float = 0.0
    variance: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Percentiles
    percentile_1: float = 0.0
    percentile_5: float = 0.0
    percentile_10: float = 0.0
    percentile_25: float = 0.0
    percentile_50: float = 0.0
    percentile_75: float = 0.0
    percentile_90: float = 0.0
    percentile_95: float = 0.0
    percentile_99: float = 0.0

    # Interquartile range
    iqr: float = 0.0

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class ExecutionMetrics:
    """
    Execution quality metrics.
    """
    # Fill metrics
    avg_fill_rate: float = 0.0
    partial_fill_rate: float = 0.0
    failed_order_rate: float = 0.0

    # Slippage analysis
    avg_slippage_bps: float = 0.0
    median_slippage_bps: float = 0.0
    max_slippage_bps: float = 0.0
    slippage_std_dev: float = 0.0

    # Timing metrics
    avg_execution_time_ms: float = 0.0
    median_execution_time_ms: float = 0.0
    max_execution_time_ms: float = 0.0

    # Price improvement
    price_improvement_rate: float = 0.0
    avg_price_improvement_bps: float = 0.0

    # Order types
    market_order_pct: float = 0.0
    limit_order_pct: float = 0.0
    limit_order_fill_rate: float = 0.0

    # Execution venue distribution
    cex_execution_pct: float = 0.0
    dex_execution_pct: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TimeBasedMetrics:
    """
    Time-based performance patterns.
    """
    # Hourly patterns (24 hours)
    best_hour: int = 0
    worst_hour: int = 0
    best_hour_return: float = 0.0
    worst_hour_return: float = 0.0
    hourly_returns: Dict[int, float] = field(default_factory=dict)

    # Daily patterns (weekdays)
    best_day: str = ""
    worst_day: str = ""
    best_day_return: float = 0.0
    worst_day_return: float = 0.0
    daily_returns: Dict[str, float] = field(default_factory=dict)

    # Monthly patterns
    best_month: str = ""
    worst_month: str = ""
    best_month_return: float = 0.0
    worst_month_return: float = 0.0
    monthly_returns: Dict[str, float] = field(default_factory=dict)

    # Quarterly patterns
    best_quarter: str = ""
    worst_quarter: str = ""
    quarterly_returns: Dict[str, float] = field(default_factory=dict)

    # Year-over-year
    yearly_returns: Dict[int, float] = field(default_factory=dict)

    # Seasonality
    has_hourly_seasonality: bool = False
    has_daily_seasonality: bool = False
    has_monthly_seasonality: bool = False

    def to_dict(self) -> Dict:
        return {
            'best_hour': self.best_hour,
            'worst_hour': self.worst_hour,
            'best_hour_return': self.best_hour_return,
            'worst_hour_return': self.worst_hour_return,
            'best_day': self.best_day,
            'worst_day': self.worst_day,
            'best_day_return': self.best_day_return,
            'worst_day_return': self.worst_day_return,
            'best_month': self.best_month,
            'worst_month': self.worst_month,
            'best_month_return': self.best_month_return,
            'worst_month_return': self.worst_month_return,
            'yearly_returns': self.yearly_returns,
            'has_hourly_seasonality': self.has_hourly_seasonality,
            'has_daily_seasonality': self.has_daily_seasonality,
            'has_monthly_seasonality': self.has_monthly_seasonality,
        }


@dataclass
class RollingMetrics:
    """
    Rolling window performance metrics.
    """
    # 30-day rolling
    rolling_30d_sharpe: List[float] = field(default_factory=list)
    rolling_30d_return: List[float] = field(default_factory=list)
    rolling_30d_volatility: List[float] = field(default_factory=list)
    rolling_30d_max_dd: List[float] = field(default_factory=list)

    # 90-day rolling
    rolling_90d_sharpe: List[float] = field(default_factory=list)
    rolling_90d_return: List[float] = field(default_factory=list)
    rolling_90d_volatility: List[float] = field(default_factory=list)
    rolling_90d_max_dd: List[float] = field(default_factory=list)

    # 180-day rolling
    rolling_180d_sharpe: List[float] = field(default_factory=list)
    rolling_180d_return: List[float] = field(default_factory=list)
    rolling_180d_volatility: List[float] = field(default_factory=list)
    rolling_180d_max_dd: List[float] = field(default_factory=list)

    # 365-day rolling
    rolling_365d_sharpe: List[float] = field(default_factory=list)
    rolling_365d_return: List[float] = field(default_factory=list)
    rolling_365d_volatility: List[float] = field(default_factory=list)
    rolling_365d_max_dd: List[float] = field(default_factory=list)

    # Summary statistics
    sharpe_stability: float = 0.0  # Std of rolling Sharpe
    return_stability: float = 0.0
    volatility_stability: float = 0.0

    # Percentile bounds
    sharpe_5th_percentile: float = 0.0
    sharpe_95th_percentile: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'sharpe_stability': self.sharpe_stability,
            'return_stability': self.return_stability,
            'volatility_stability': self.volatility_stability,
            'sharpe_5th_percentile': self.sharpe_5th_percentile,
            'sharpe_95th_percentile': self.sharpe_95th_percentile,
            'rolling_30d_sharpe_latest': self.rolling_30d_sharpe[-1] if self.rolling_30d_sharpe else 0.0,
            'rolling_90d_sharpe_latest': self.rolling_90d_sharpe[-1] if self.rolling_90d_sharpe else 0.0,
            'rolling_180d_sharpe_latest': self.rolling_180d_sharpe[-1] if self.rolling_180d_sharpe else 0.0,
            'rolling_365d_sharpe_latest': self.rolling_365d_sharpe[-1] if self.rolling_365d_sharpe else 0.0,
        }


@dataclass
class PDFCompliantMetrics:
    """
    Master container for all PDF Section 2.4 compliant metrics.

    This is the primary output structure containing 60+ metrics organized
    by category for comprehensive backtesting analysis.
    """
    # Core metrics (PDF REQUIRED)
    core: CoreMetrics = field(default_factory=CoreMetrics)

    # Risk-adjusted returns
    risk_adjusted: RiskAdjustedMetrics = field(default_factory=RiskAdjustedMetrics)

    # Drawdown analysis
    drawdown: DrawdownMetrics = field(default_factory=DrawdownMetrics)

    # Trade statistics
    trades: TradeStatistics = field(default_factory=TradeStatistics)

    # Cost analysis
    costs: CostMetrics = field(default_factory=CostMetrics)

    # Capacity analysis
    capacity: CapacityMetrics = field(default_factory=CapacityMetrics)

    # Risk metrics
    risk: RiskMetrics = field(default_factory=RiskMetrics)

    # Statistical tests
    statistical: StatisticalMetrics = field(default_factory=StatisticalMetrics)

    # Execution quality
    execution: ExecutionMetrics = field(default_factory=ExecutionMetrics)

    # Time-based patterns
    time_based: TimeBasedMetrics = field(default_factory=TimeBasedMetrics)

    # Rolling window metrics
    rolling: RollingMetrics = field(default_factory=RollingMetrics)

    # Metadata
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    data_start_date: Optional[datetime] = None
    data_end_date: Optional[datetime] = None
    total_days: int = 0

    def to_dict(self) -> Dict:
        """Convert all metrics to nested dictionary."""
        return {
            'core': self.core.to_dict(),
            'risk_adjusted': self.risk_adjusted.to_dict(),
            'drawdown': self.drawdown.to_dict(),
            'trades': self.trades.to_dict(),
            'costs': self.costs.to_dict(),
            'capacity': self.capacity.to_dict(),
            'risk': self.risk.to_dict(),
            'statistical': self.statistical.to_dict(),
            'execution': self.execution.to_dict(),
            'time_based': self.time_based.to_dict(),
            'rolling': self.rolling.to_dict(),
            'metadata': {
                'calculation_timestamp': self.calculation_timestamp.isoformat(),
                'data_start_date': self.data_start_date.isoformat() if self.data_start_date else None,
                'data_end_date': self.data_end_date.isoformat() if self.data_end_date else None,
                'total_days': self.total_days,
            }
        }

    def get_pdf_required_dict(self) -> Dict:
        """Get only the PDF-required metrics for compliance reporting."""
        return {
            'sharpe_ratio': self.core.sharpe_ratio,
            'sortino_ratio': self.core.sortino_ratio,
            'max_drawdown': self.core.max_drawdown,
            'avg_holding_period_hours': self.core.avg_holding_period_hours,
            'annualized_turnover': self.core.annualized_turnover,
            'transaction_cost_drag_pct': self.core.transaction_cost_drag_pct,
            'gas_cost_impact_pct': self.core.gas_cost_impact_pct,
        }

    def get_flat_dict(self) -> Dict[str, float]:
        """Get flattened dictionary with all numeric metrics."""
        flat = {}
        for category, metrics in self.to_dict().items():
            if category == 'metadata':
                continue
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        flat[f"{category}_{key}"] = value
        return flat

    def get_summary_series(self) -> pd.Series:
        """Get key metrics as pandas Series for quick analysis."""
        return pd.Series({
            'Sharpe Ratio': self.core.sharpe_ratio,
            'Sortino Ratio': self.core.sortino_ratio,
            'Calmar Ratio': self.risk_adjusted.calmar_ratio,
            'Max Drawdown': self.core.max_drawdown,
            'Total Return': self.core.total_return,
            'Annualized Return': self.core.annualized_return,
            'Annualized Volatility': self.core.annualized_volatility,
            'Win Rate': self.trades.win_rate,
            'Profit Factor': self.trades.profit_factor,
            'Total Trades': self.trades.total_trades,
            'Avg Holding Period (hrs)': self.core.avg_holding_period_hours,
            'Transaction Cost Drag %': self.core.transaction_cost_drag_pct,
            'VaR 95%': self.risk.var_95,
            'CVaR 95%': self.risk.cvar_95,
        })


@dataclass
class VenueSpecificMetrics:
    """
    Venue-specific metrics breakdown per PDF requirements.

    PDF REQUIRED: "Calculate metrics separately for:
    1. CEX-only pairs
    2. DEX-only pairs
    3. Mixed (CEX-DEX) pairs
    4. Combined portfolio"
    """
    cex_metrics: PDFCompliantMetrics = field(default_factory=PDFCompliantMetrics)
    dex_metrics: PDFCompliantMetrics = field(default_factory=PDFCompliantMetrics)
    dex_l2_metrics: PDFCompliantMetrics = field(default_factory=PDFCompliantMetrics)
    hybrid_metrics: PDFCompliantMetrics = field(default_factory=PDFCompliantMetrics)
    mixed_metrics: PDFCompliantMetrics = field(default_factory=PDFCompliantMetrics)
    combined_metrics: PDFCompliantMetrics = field(default_factory=PDFCompliantMetrics)

    # Venue comparison summary
    venue_sharpe_comparison: Dict[str, float] = field(default_factory=dict)
    venue_return_comparison: Dict[str, float] = field(default_factory=dict)
    venue_cost_comparison: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'cex': self.cex_metrics.to_dict(),
            'dex': self.dex_metrics.to_dict(),
            'dex_l2': self.dex_l2_metrics.to_dict(),
            'hybrid': self.hybrid_metrics.to_dict(),
            'mixed': self.mixed_metrics.to_dict(),
            'combined': self.combined_metrics.to_dict(),
            'venue_sharpe_comparison': self.venue_sharpe_comparison,
            'venue_return_comparison': self.venue_return_comparison,
            'venue_cost_comparison': self.venue_cost_comparison,
        }

    def get_comparison_df(self) -> pd.DataFrame:
        """Create comparison DataFrame for venue analysis."""
        data = {
            'CEX': self.cex_metrics.get_pdf_required_dict(),
            'DEX': self.dex_metrics.get_pdf_required_dict(),
            'DEX L2': self.dex_l2_metrics.get_pdf_required_dict(),
            'Hybrid': self.hybrid_metrics.get_pdf_required_dict(),
            'Mixed': self.mixed_metrics.get_pdf_required_dict(),
            'Combined': self.combined_metrics.get_pdf_required_dict(),
        }
        return pd.DataFrame(data).T


@dataclass
class RegimeSpecificMetrics:
    """
    Regime-specific metrics breakdown.

    PDF mentions different performance expectations by regime:
    - Bull market: moderate CEX, DEX can break
    - Bear market: better CEX, mixed DEX performance
    - Sideways: best environment for mean reversion
    - High volatility: wider spreads, more opportunities
    - Low volatility: tighter spreads, fewer signals
    - Crisis: correlation breakdown, risk management critical
    - Recovery: regime transition opportunities
    """
    bull_metrics: PDFCompliantMetrics = field(default_factory=PDFCompliantMetrics)
    bear_metrics: PDFCompliantMetrics = field(default_factory=PDFCompliantMetrics)
    sideways_metrics: PDFCompliantMetrics = field(default_factory=PDFCompliantMetrics)
    high_vol_metrics: PDFCompliantMetrics = field(default_factory=PDFCompliantMetrics)
    low_vol_metrics: PDFCompliantMetrics = field(default_factory=PDFCompliantMetrics)
    crisis_metrics: PDFCompliantMetrics = field(default_factory=PDFCompliantMetrics)
    recovery_metrics: PDFCompliantMetrics = field(default_factory=PDFCompliantMetrics)

    # Regime statistics
    regime_durations: Dict[str, float] = field(default_factory=dict)  # Avg days per regime
    regime_frequencies: Dict[str, float] = field(default_factory=dict)  # % time in each regime
    regime_transitions: Dict[str, Dict[str, int]] = field(default_factory=dict)  # Transition matrix

    def to_dict(self) -> Dict:
        return {
            'bull_market': self.bull_metrics.to_dict(),
            'bear_market': self.bear_metrics.to_dict(),
            'sideways_market': self.sideways_metrics.to_dict(),
            'high_volatility': self.high_vol_metrics.to_dict(),
            'low_volatility': self.low_vol_metrics.to_dict(),
            'crisis': self.crisis_metrics.to_dict(),
            'recovery': self.recovery_metrics.to_dict(),
            'regime_durations': self.regime_durations,
            'regime_frequencies': self.regime_frequencies,
            'regime_transitions': self.regime_transitions,
        }

    def get_comparison_df(self) -> pd.DataFrame:
        """Create comparison DataFrame for regime analysis."""
        data = {
            'Bull': self.bull_metrics.get_pdf_required_dict(),
            'Bear': self.bear_metrics.get_pdf_required_dict(),
            'Sideways': self.sideways_metrics.get_pdf_required_dict(),
            'High Vol': self.high_vol_metrics.get_pdf_required_dict(),
            'Low Vol': self.low_vol_metrics.get_pdf_required_dict(),
            'Crisis': self.crisis_metrics.get_pdf_required_dict(),
            'Recovery': self.recovery_metrics.get_pdf_required_dict(),
        }
        return pd.DataFrame(data).T


# =============================================================================
# HELPER FUNCTIONS - Metric Calculations
# =============================================================================

def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
        return default
    result = numerator / denominator
    return default if np.isnan(result) or np.isinf(result) else result


def _calculate_drawdown_series(returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate drawdown series from returns.

    Returns:
        Tuple of (drawdown_series, cumulative_returns)
    """
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    # Prevent division by zero and cap drawdown at 100%
    # When cumulative goes negative (total loss), drawdown cannot exceed 100%
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown = (running_max - cumulative) / running_max
    drawdown = np.clip(np.nan_to_num(drawdown, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return drawdown, cumulative


def _find_drawdown_periods(drawdown: np.ndarray, threshold: float = 0.0) -> List[Dict]:
    """
    Find individual drawdown periods.

    Returns list of dicts with start_idx, end_idx, depth, duration.
    """
    periods = []
    in_drawdown = False
    start_idx = 0

    for i, dd in enumerate(drawdown):
        if dd > threshold and not in_drawdown:
            in_drawdown = True
            start_idx = i
        elif dd <= threshold and in_drawdown:
            in_drawdown = False
            periods.append({
                'start_idx': start_idx,
                'end_idx': i,
                'depth': float(np.max(drawdown[start_idx:i+1])),
                'duration': i - start_idx
            })

    # Handle ongoing drawdown at end
    if in_drawdown:
        periods.append({
            'start_idx': start_idx,
            'end_idx': len(drawdown) - 1,
            'depth': float(np.max(drawdown[start_idx:])),
            'duration': len(drawdown) - start_idx
        })

    return periods


def _calculate_streaks(returns: np.ndarray) -> Dict:
    """Calculate win/loss streak statistics."""
    if len(returns) == 0:
        return {
            'max_wins': 0, 'max_losses': 0,
            'avg_wins': 0.0, 'avg_losses': 0.0,
            'current': 0
        }

    win_streaks = []
    loss_streaks = []
    current_streak = 0
    streak_type = None  # 'win' or 'loss'

    for ret in returns:
        if ret > 0:
            if streak_type == 'win':
                current_streak += 1
            else:
                if streak_type == 'loss' and current_streak > 0:
                    loss_streaks.append(current_streak)
                current_streak = 1
                streak_type = 'win'
        elif ret < 0:
            if streak_type == 'loss':
                current_streak += 1
            else:
                if streak_type == 'win' and current_streak > 0:
                    win_streaks.append(current_streak)
                current_streak = 1
                streak_type = 'loss'

    # Record final streak
    if streak_type == 'win' and current_streak > 0:
        win_streaks.append(current_streak)
    elif streak_type == 'loss' and current_streak > 0:
        loss_streaks.append(current_streak)

    return {
        'max_wins': max(win_streaks) if win_streaks else 0,
        'max_losses': max(loss_streaks) if loss_streaks else 0,
        'avg_wins': np.mean(win_streaks) if win_streaks else 0.0,
        'avg_losses': np.mean(loss_streaks) if loss_streaks else 0.0,
        'current': current_streak * (1 if streak_type == 'win' else -1)
    }


def _calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Calculate historical Value at Risk."""
    if len(returns) == 0:
        return 0.0
    return float(-np.percentile(returns, (1 - confidence) * 100))


def _calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Calculate Conditional VaR (Expected Shortfall)."""
    if len(returns) == 0:
        return 0.0
    var = _calculate_var(returns, confidence)
    tail_returns = returns[returns <= -var]
    return float(-np.mean(tail_returns)) if len(tail_returns) > 0 else var


def _calculate_parametric_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Calculate parametric VaR assuming normal distribution."""
    if len(returns) < 2:
        return 0.0
    z_score = stats.norm.ppf(1 - confidence)
    return float(-(np.mean(returns) + z_score * np.std(returns)))


def _calculate_cornish_fisher_var(
    returns: np.ndarray,
    confidence: float = 0.95
) -> float:
    """Calculate Cornish-Fisher VaR (skewness/kurtosis adjusted)."""
    if len(returns) < 4:
        return _calculate_parametric_var(returns, confidence)

    z = stats.norm.ppf(1 - confidence)
    s = stats.skew(returns)
    k = stats.kurtosis(returns)

    # Cornish-Fisher expansion
    z_cf = (z +
            (z**2 - 1) * s / 6 +
            (z**3 - 3*z) * k / 24 -
            (2*z**3 - 5*z) * s**2 / 36)

    return float(-(np.mean(returns) + z_cf * np.std(returns)))


def _calculate_omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Calculate Omega ratio.

    Omega = Sum of gains above threshold / Sum of losses below threshold
    """
    gains = np.sum(returns[returns > threshold] - threshold)
    losses = np.sum(threshold - returns[returns < threshold])
    return _safe_divide(gains, losses, 1.0)


def _calculate_ulcer_index(returns: np.ndarray) -> float:
    """
    Calculate Ulcer Index (RMS of drawdowns).

    Lower is better - measures depth and duration of drawdowns.
    """
    if len(returns) == 0:
        return 0.0

    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown_pct = (running_max - cumulative) / running_max * 100

    return float(np.sqrt(np.mean(drawdown_pct ** 2)))


def _calculate_pain_index(returns: np.ndarray) -> float:
    """
    Calculate Pain Index (mean absolute drawdown).
    """
    if len(returns) == 0:
        return 0.0

    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (running_max - cumulative) / running_max

    return float(np.mean(drawdown))


def _calculate_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Calculate Kelly Criterion for optimal position sizing.

    Kelly = W - (1-W)/R where W=win rate, R=win/loss ratio
    """
    if avg_loss == 0 or avg_win == 0:
        return 0.0

    win_loss_ratio = abs(avg_win / avg_loss)
    kelly = win_rate - (1 - win_rate) / win_loss_ratio
    return max(0.0, min(kelly, 1.0))  # Bound between 0 and 1


def _calculate_risk_of_ruin(
    win_rate: float,
    win_loss_ratio: float,
    risk_per_trade: float = 0.02,
    ruin_threshold: float = 0.5
) -> float:
    """
    Calculate probability of losing a certain % of capital.

    Uses simplified formula: ((1-edge)/(1+edge))^units
    where edge = win_rate * win_loss_ratio - (1-win_rate)
    """
    if win_rate == 0 or win_loss_ratio == 0 or risk_per_trade == 0:
        return 1.0

    edge = win_rate * win_loss_ratio - (1 - win_rate)
    if edge <= 0:
        return 1.0

    units_to_ruin = ruin_threshold / risk_per_trade

    q = 1 - win_rate
    p = win_rate

    if win_loss_ratio == 1:
        # Special case: even odds
        ruin_prob = (q / p) ** units_to_ruin if p > q else 1.0
    else:
        # General formula
        try:
            ruin_prob = ((q / (p * win_loss_ratio)) ** units_to_ruin
                        if p * win_loss_ratio > q else 1.0)
        except (OverflowError, ZeroDivisionError):
            ruin_prob = 1.0

    return min(1.0, max(0.0, ruin_prob))


# =============================================================================
# MAIN CALCULATOR CLASS
# =============================================================================

class AdvancedMetricsCalculator:
    """
    Comprehensive metrics calculator implementing 60+ performance metrics.

    Calculates all PDF Section 2.4 required metrics plus comprehensive
    analysis across multiple dimensions:

    1. Core risk-adjusted returns (Sharpe, Sortino, Calmar, Omega)
    2. Drawdown analysis (max DD, duration, recovery, Ulcer Index)
    3. Trade statistics (win rate, profit factor, expectancy, Kelly)
    4. Cost analysis (transaction cost drag, gas impact, MEV)
    5. Risk metrics (VaR, CVaR, tail risk, beta)
    6. Statistical tests (normality, stationarity, autocorrelation)
    7. Time-based patterns (hourly, daily, monthly seasonality)
    8. Rolling window metrics (30d, 90d, 180d, 365d)

    Usage:
        calculator = AdvancedMetricsCalculator()
        metrics = calculator.calculate_all_metrics(
            returns=returns_series,
            trades=trades_list,
            btc_returns=btc_series
        )
        venue_breakdown = calculator.calculate_venue_breakdown(trades, pair_info)
        regime_breakdown = calculator.calculate_regime_breakdown(trades, btc_returns)

    Example:
        >>> calc = AdvancedMetricsCalculator(risk_free_rate=0.045)
        >>> metrics = calc.calculate_all_metrics(returns=daily_returns)
        >>> print(f"Sharpe: {metrics.core.sharpe_ratio:.2f}")
        >>> print(f"Max DD: {metrics.drawdown.max_drawdown:.2%}")
    """

    def __init__(
        self,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        annualization_factor: int = TRADING_DAYS_PER_YEAR,
        min_trades_for_stats: int = 30,
        rolling_windows: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize the metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default: 4.5%)
            annualization_factor: Days per year for annualization (default: 365 for crypto)
            min_trades_for_stats: Minimum trades required for statistical tests
            rolling_windows: Custom rolling window sizes (default: 30, 90, 180, 365)
        """
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
        self.min_trades_for_stats = min_trades_for_stats
        self.rolling_windows = rolling_windows or ROLLING_WINDOWS

        self._daily_rf = risk_free_rate / annualization_factor

        logger.info(
            f"AdvancedMetricsCalculator initialized: "
            f"rf={risk_free_rate:.2%}, annualization={annualization_factor}"
        )

    def calculate_all_metrics(
        self,
        returns: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
        trades: Optional[List[Dict]] = None,
        equity_curve: Optional[pd.Series] = None,
        btc_returns: Optional[pd.Series] = None,
        eth_returns: Optional[pd.Series] = None,
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> PDFCompliantMetrics:
        """
        Calculate comprehensive metrics from returns and/or trades.

        This is the main entry point for full metrics calculation.

        Args:
            returns: Return series (daily returns preferred)
            trades: List of trade dictionaries with fields:
                - return_pct: Trade return percentage
                - gross_pnl: Gross P&L in USD
                - net_pnl: Net P&L in USD
                - holding_hours: Trade duration in hours
                - entry_time: Entry timestamp
                - exit_time: Exit timestamp
                - position_size_usd: Position size
                - venue_type: 'cex', 'dex', 'hybrid', 'mixed'
                - fee_cost: Trading fees
                - slippage_cost: Slippage cost
                - gas_cost: Gas costs (DEX)
                - mev_cost: MEV costs (DEX)
            equity_curve: Portfolio equity curve
            btc_returns: BTC returns for beta/correlation
            eth_returns: ETH returns for beta/correlation
            timestamps: Timestamps for time-based analysis

        Returns:
            PDFCompliantMetrics with all calculated metrics
        """
        metrics = PDFCompliantMetrics()

        # Convert returns to numpy array
        returns_arr = self._normalize_returns(returns, trades)

        if returns_arr is None or len(returns_arr) == 0:
            logger.warning("No returns data provided, returning empty metrics")
            return metrics

        # Set metadata
        if timestamps is not None and len(timestamps) > 0:
            metrics.data_start_date = timestamps[0].to_pydatetime() if hasattr(timestamps[0], 'to_pydatetime') else timestamps[0]
            metrics.data_end_date = timestamps[-1].to_pydatetime() if hasattr(timestamps[-1], 'to_pydatetime') else timestamps[-1]
        metrics.total_days = len(returns_arr)

        # Calculate all metric categories
        metrics.core = self._calculate_core_metrics(returns_arr, trades)
        metrics.risk_adjusted = self._calculate_risk_adjusted_metrics(returns_arr, btc_returns)
        metrics.drawdown = self._calculate_drawdown_metrics(returns_arr, timestamps)
        metrics.risk = self._calculate_risk_metrics(returns_arr, btc_returns, eth_returns)
        metrics.statistical = self._calculate_statistical_metrics(returns_arr)

        if trades:
            metrics.trades = self._calculate_trade_statistics(trades)
            metrics.costs = self._calculate_cost_metrics(trades)
            metrics.execution = self._calculate_execution_metrics(trades)

        if timestamps is not None:
            metrics.time_based = self._calculate_time_based_metrics(returns_arr, timestamps)

        if len(returns_arr) >= self.rolling_windows['short']:
            metrics.rolling = self._calculate_rolling_metrics(returns_arr)

        metrics.calculation_timestamp = datetime.now()

        logger.info(
            f"Calculated {self._count_metrics(metrics)} metrics "
            f"for {len(returns_arr)} data points"
        )

        return metrics

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    def _normalize_returns(
        self,
        returns: Optional[Union[pd.Series, np.ndarray, List[float]]],
        trades: Optional[List[Dict]]
    ) -> Optional[np.ndarray]:
        """Normalize returns input to numpy array."""
        if returns is not None:
            if isinstance(returns, pd.Series):
                return returns.values.astype(float)
            elif isinstance(returns, list):
                return np.array(returns, dtype=float)
            elif isinstance(returns, np.ndarray):
                return returns.astype(float)

        if trades and len(trades) > 0:
            return np.array([
                t.get('return_pct', t.get('return', 0))
                if isinstance(t, dict)
                else getattr(t, 'return_pct', getattr(t, 'return', 0))
                for t in trades
            ], dtype=float)

        return None

    def _count_metrics(self, metrics: PDFCompliantMetrics) -> int:
        """Count total number of calculated metrics."""
        count = 0
        for category_metrics in [
            metrics.core, metrics.risk_adjusted, metrics.drawdown,
            metrics.trades, metrics.costs, metrics.risk,
            metrics.statistical, metrics.execution
        ]:
            for k, v in category_metrics.__dict__.items():
                if isinstance(v, (int, float)) and v != 0:
                    count += 1
        return count

    def _calculate_core_metrics(
        self,
        returns: np.ndarray,
        trades: Optional[List[Dict]] = None
    ) -> CoreMetrics:
        """Calculate core PDF-required metrics."""
        core = CoreMetrics()

        if len(returns) == 0:
            return core

        # Return metrics
        core.total_return = float(np.sum(returns))
        core.cumulative_return = float(np.prod(1 + returns) - 1)

        n_years = len(returns) / self.annualization_factor
        if n_years > 0:
            core.annualized_return = float(
                (1 + core.cumulative_return) ** (1 / n_years) - 1
            )
            core.cagr = core.annualized_return

        # Volatility metrics
        core.annualized_volatility = float(
            np.std(returns) * np.sqrt(self.annualization_factor)
        )

        downside_returns = returns[returns < 0]
        upside_returns = returns[returns > 0]

        if len(downside_returns) > 0:
            core.downside_volatility = float(
                np.std(downside_returns) * np.sqrt(self.annualization_factor)
            )
        if len(upside_returns) > 0:
            core.upside_volatility = float(
                np.std(upside_returns) * np.sqrt(self.annualization_factor)
            )

        # Sharpe ratio (PDF REQUIRED)
        if core.annualized_volatility > 0:
            core.sharpe_ratio = float(
                (core.annualized_return - self.risk_free_rate) /
                core.annualized_volatility
            )

        # Sortino ratio (PDF REQUIRED)
        if core.downside_volatility > 0:
            core.sortino_ratio = float(
                (core.annualized_return - self.risk_free_rate) /
                core.downside_volatility
            )

        # Maximum drawdown (PDF REQUIRED)
        drawdown, _ = _calculate_drawdown_series(returns)
        core.max_drawdown = float(np.max(drawdown))

        # Trade-based metrics if available
        if trades and len(trades) > 0:
            holding_hours = []
            position_sizes = []
            total_costs = 0.0
            dex_gas_costs = 0.0
            dex_gross = 0.0

            for trade in trades:
                if isinstance(trade, dict):
                    hours = trade.get('holding_hours', 24)
                    size = trade.get('position_size_usd', 10000)
                    fees = trade.get('fee_cost', 0)
                    slip = trade.get('slippage_cost', 0)
                    gas = trade.get('gas_cost', 0)
                    mev = trade.get('mev_cost', 0)
                    gross = trade.get('gross_pnl', 0)
                    venue = str(trade.get('venue_type', '')).lower()
                else:
                    hours = getattr(trade, 'holding_hours', 24)
                    size = getattr(trade, 'position_size_usd', 10000)
                    fees = getattr(trade, 'fee_cost', 0)
                    slip = getattr(trade, 'slippage_cost', 0)
                    gas = getattr(trade, 'gas_cost', 0)
                    mev = getattr(trade, 'mev_cost', 0)
                    gross = getattr(trade, 'gross_pnl', 0)
                    venue = str(getattr(trade, 'venue_type', '')).lower()

                holding_hours.append(hours)
                position_sizes.append(size)
                total_costs += fees + slip + gas + mev

                if 'dex' in venue:
                    dex_gas_costs += gas + mev
                    dex_gross += abs(gross)

            # Average holding period (PDF REQUIRED)
            core.avg_holding_period_hours = float(np.mean(holding_hours))

            # Annualized turnover (PDF REQUIRED)
            avg_size = np.mean(position_sizes) if position_sizes else 10000
            trades_per_year = (
                self.annualization_factor * 24 / core.avg_holding_period_hours
                if core.avg_holding_period_hours > 0 else 0
            )
            core.annualized_turnover = float(trades_per_year * avg_size * 2 / 1_000_000)

            # Transaction cost drag (PDF REQUIRED)
            gross_pnl = sum(
                abs(t.get('gross_pnl', 0) if isinstance(t, dict)
                    else getattr(t, 'gross_pnl', 0))
                for t in trades
            )
            if gross_pnl > 0:
                core.transaction_cost_drag_pct = float(total_costs / gross_pnl * 100)

            # Gas cost impact for DEX (PDF REQUIRED)
            if dex_gross > 0:
                core.gas_cost_impact_pct = float(dex_gas_costs / dex_gross * 100)

        return core

    def _calculate_risk_adjusted_metrics(
        self,
        returns: np.ndarray,
        btc_returns: Optional[pd.Series] = None
    ) -> RiskAdjustedMetrics:
        """Calculate comprehensive risk-adjusted return metrics."""
        ram = RiskAdjustedMetrics()

        if len(returns) < 2:
            return ram

        mean_return = np.mean(returns)
        std_return = np.std(returns)
        ann_factor = np.sqrt(self.annualization_factor)

        # Sharpe ratio
        if std_return > 0:
            ram.sharpe_ratio = float(
                (mean_return - self._daily_rf) / std_return * ann_factor
            )

        # Sortino ratio
        downside = returns[returns < 0]
        if len(downside) > 1:
            downside_std = np.std(downside)
            if downside_std > 0:
                ram.sortino_ratio = float(
                    (mean_return - self._daily_rf) / downside_std * ann_factor
                )

        # Calmar ratio
        drawdown, _ = _calculate_drawdown_series(returns)
        max_dd = np.max(drawdown)
        ann_return = (1 + np.sum(returns)) ** (self.annualization_factor / len(returns)) - 1
        if max_dd > 0:
            ram.calmar_ratio = float(ann_return / max_dd)

        # Omega ratio
        ram.omega_ratio = _calculate_omega_ratio(returns, 0.0)

        # Treynor ratio (requires benchmark)
        if btc_returns is not None and len(btc_returns) >= len(returns):
            btc_arr = btc_returns.values[:len(returns)]
            if np.std(btc_arr) > 0:
                cov = np.cov(returns, btc_arr)[0, 1]
                btc_var = np.var(btc_arr)
                beta = cov / btc_var if btc_var > 0 else 0
                if beta != 0:
                    ram.treynor_ratio = float(
                        (ann_return - self.risk_free_rate) / beta
                    )
                # Information ratio
                active_return = returns - btc_arr
                tracking_error = np.std(active_return) * ann_factor
                if tracking_error > 0:
                    ram.information_ratio = float(
                        np.mean(active_return) * self.annualization_factor / tracking_error
                    )

        # Burke ratio (return / sqrt(sum of squared drawdowns))
        dd_periods = _find_drawdown_periods(drawdown, 0.01)
        if dd_periods:
            sq_dd_sum = sum(p['depth'] ** 2 for p in dd_periods)
            if sq_dd_sum > 0:
                ram.burke_ratio = float(ann_return / np.sqrt(sq_dd_sum))

        # Sterling ratio (return / avg drawdown - 10%)
        if max_dd > 0.10:
            ram.sterling_ratio = float(ann_return / (max_dd - 0.10))

        # Tail ratio (95th percentile / abs(5th percentile))
        p95 = np.percentile(returns, 95)
        p5 = abs(np.percentile(returns, 5))
        if p5 > 0:
            ram.tail_ratio = float(p95 / p5)

        # Gain to pain ratio
        gains = np.sum(returns[returns > 0])
        losses = abs(np.sum(returns[returns < 0]))
        if losses > 0:
            ram.gain_to_pain_ratio = float(gains / losses)

        # Pain ratio (return / pain index)
        pain_idx = _calculate_pain_index(returns)
        if pain_idx > 0:
            ram.pain_ratio = float(ann_return / pain_idx)

        # Ulcer Performance Index
        ulcer_idx = _calculate_ulcer_index(returns)
        if ulcer_idx > 0:
            ram.ulcer_performance_index = float(
                (ann_return - self.risk_free_rate) / (ulcer_idx / 100)
            )

        # Common Sense ratio (profit factor * tail ratio)
        profit_factor = gains / losses if losses > 0 else 0
        ram.common_sense_ratio = float(profit_factor * ram.tail_ratio)

        # CPC Index (profit factor * win rate * payoff ratio)
        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
        avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
        avg_loss = abs(np.mean(returns[returns < 0])) if np.any(returns < 0) else 0
        payoff = avg_win / avg_loss if avg_loss > 0 else 0
        ram.cpc_index = float(profit_factor * win_rate * payoff)

        # Serenity index (custom stability metric)
        # Higher is better: combines Sharpe, downside volatility, and drawdown recovery
        if std_return > 0 and max_dd > 0:
            ram.serenity_index = float(
                ram.sharpe_ratio * (1 - max_dd) / (1 + _calculate_pain_index(returns))
            )

        return ram

    def _calculate_drawdown_metrics(
        self,
        returns: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> DrawdownMetrics:
        """Calculate comprehensive drawdown analysis."""
        dd_metrics = DrawdownMetrics()

        if len(returns) == 0:
            return dd_metrics

        drawdown, cumulative = _calculate_drawdown_series(returns)

        # Primary metrics
        dd_metrics.max_drawdown = float(np.max(drawdown))
        dd_metrics.avg_drawdown = float(np.mean(drawdown))

        # Find drawdown periods
        periods = _find_drawdown_periods(drawdown, 0.001)  # 0.1% threshold
        dd_metrics.num_drawdowns = len(periods)

        if periods:
            # Duration analysis
            durations = [p['duration'] for p in periods]
            depths = [p['depth'] for p in periods]

            dd_metrics.max_drawdown_duration_days = float(max(durations))
            dd_metrics.avg_drawdown_duration_days = float(np.mean(durations))

            # Max drawdown timing
            max_dd_period = max(periods, key=lambda x: x['depth'])
            if timestamps is not None and len(timestamps) > max_dd_period['end_idx']:
                dd_metrics.max_drawdown_start = timestamps[max_dd_period['start_idx']]
                dd_metrics.max_drawdown_end = timestamps[max_dd_period['end_idx']]

            # Drawdown counts by severity
            dd_metrics.num_drawdowns_gt_5pct = sum(1 for d in depths if d > 0.05)
            dd_metrics.num_drawdowns_gt_10pct = sum(1 for d in depths if d > 0.10)
            dd_metrics.num_drawdowns_gt_20pct = sum(1 for d in depths if d > 0.20)

            # Drawdown distribution
            dd_metrics.drawdown_5th_percentile = float(np.percentile(depths, 5))
            dd_metrics.drawdown_25th_percentile = float(np.percentile(depths, 25))
            dd_metrics.drawdown_50th_percentile = float(np.percentile(depths, 50))
            dd_metrics.drawdown_75th_percentile = float(np.percentile(depths, 75))
            dd_metrics.drawdown_95th_percentile = float(np.percentile(depths, 95))

        # Recovery factor
        total_return = np.sum(returns)
        if dd_metrics.max_drawdown > 0:
            dd_metrics.recovery_factor = float(total_return / dd_metrics.max_drawdown)

        # Time underwater
        underwater = drawdown > 0.001
        dd_metrics.pct_time_underwater = float(np.mean(underwater))

        # Find longest underwater period
        if np.any(underwater):
            underwater_runs = []
            current_run = 0
            for uw in underwater:
                if uw:
                    current_run += 1
                else:
                    if current_run > 0:
                        underwater_runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                underwater_runs.append(current_run)

            if underwater_runs:
                dd_metrics.longest_underwater_days = float(max(underwater_runs))
                dd_metrics.avg_recovery_time_days = float(np.mean(underwater_runs))
                dd_metrics.longest_recovery_days = float(max(underwater_runs))

        # Risk indices
        dd_metrics.ulcer_index = _calculate_ulcer_index(returns)
        dd_metrics.pain_index = _calculate_pain_index(returns)

        # Martin ratio (return / ulcer index)
        ann_return = (1 + total_return) ** (self.annualization_factor / len(returns)) - 1
        if dd_metrics.ulcer_index > 0:
            dd_metrics.martin_ratio = float(ann_return / (dd_metrics.ulcer_index / 100))

        return dd_metrics

    def _calculate_trade_statistics(self, trades: List[Dict]) -> TradeStatistics:
        """Calculate comprehensive trade-level statistics."""
        ts = TradeStatistics()

        if not trades or len(trades) == 0:
            return ts

        # Extract trade data
        returns = []
        durations = []
        sizes = []
        entry_times = []
        exit_times = []

        for trade in trades:
            if isinstance(trade, dict):
                ret = trade.get('return_pct', trade.get('return', 0))
                dur = trade.get('holding_hours', 24)
                size = trade.get('position_size_usd', 10000)
                entry = trade.get('entry_time')
                exit_t = trade.get('exit_time')
            else:
                ret = getattr(trade, 'return_pct', getattr(trade, 'return', 0))
                dur = getattr(trade, 'holding_hours', 24)
                size = getattr(trade, 'position_size_usd', 10000)
                entry = getattr(trade, 'entry_time', None)
                exit_t = getattr(trade, 'exit_time', None)

            returns.append(ret)
            durations.append(dur)
            sizes.append(size)
            entry_times.append(entry)
            exit_times.append(exit_t)

        returns = np.array(returns)
        durations = np.array(durations)
        sizes = np.array(sizes)

        # Trade counts
        ts.total_trades = len(trades)
        ts.winning_trades = int(np.sum(returns > 0))
        ts.losing_trades = int(np.sum(returns < 0))
        ts.breakeven_trades = int(np.sum(returns == 0))

        # Win/loss rates
        ts.win_rate = float(ts.winning_trades / ts.total_trades)
        ts.loss_rate = float(ts.losing_trades / ts.total_trades)

        # P&L metrics
        ts.avg_trade_return = float(np.mean(returns))
        winners = returns[returns > 0]
        losers = returns[returns < 0]

        if len(winners) > 0:
            ts.avg_winning_trade = float(np.mean(winners))
        if len(losers) > 0:
            ts.avg_losing_trade = float(np.mean(losers))

        # Best/worst trades
        ts.best_trade = float(np.max(returns))
        ts.worst_trade = float(np.min(returns))

        best_idx = np.argmax(returns)
        worst_idx = np.argmin(returns)
        if entry_times[best_idx]:
            ts.best_trade_date = pd.to_datetime(entry_times[best_idx])
        if entry_times[worst_idx]:
            ts.worst_trade_date = pd.to_datetime(entry_times[worst_idx])

        # Profit factor
        gross_profit = np.sum(winners) if len(winners) > 0 else 0
        gross_loss = abs(np.sum(losers)) if len(losers) > 0 else 0
        if gross_loss > 0:
            ts.profit_factor = float(gross_profit / gross_loss)

        # Payoff ratio (avg win / avg loss)
        if ts.avg_losing_trade != 0:
            ts.payoff_ratio = float(abs(ts.avg_winning_trade / ts.avg_losing_trade))

        # Expectancy
        ts.expectancy = float(
            ts.win_rate * ts.avg_winning_trade +
            (1 - ts.win_rate) * ts.avg_losing_trade
        )

        # Expectancy ratio
        if abs(ts.avg_losing_trade) > 0:
            ts.expectancy_ratio = float(ts.expectancy / abs(ts.avg_losing_trade))

        # Kelly criterion
        ts.kelly_criterion = _calculate_kelly_criterion(
            ts.win_rate, ts.avg_winning_trade, ts.avg_losing_trade
        )
        ts.half_kelly = ts.kelly_criterion / 2

        # Streak analysis
        streaks = _calculate_streaks(returns)
        ts.max_consecutive_wins = streaks['max_wins']
        ts.max_consecutive_losses = streaks['max_losses']
        ts.avg_consecutive_wins = streaks['avg_wins']
        ts.avg_consecutive_losses = streaks['avg_losses']
        ts.current_streak = streaks['current']

        # Duration analysis
        ts.avg_trade_duration_hours = float(np.mean(durations))
        ts.median_trade_duration_hours = float(np.median(durations))
        ts.min_trade_duration_hours = float(np.min(durations))
        ts.max_trade_duration_hours = float(np.max(durations))

        if len(winners) > 0:
            winning_durations = durations[returns > 0]
            ts.avg_winning_duration_hours = float(np.mean(winning_durations))
        if len(losers) > 0:
            losing_durations = durations[returns < 0]
            ts.avg_losing_duration_hours = float(np.mean(losing_durations))

        # Size analysis
        ts.avg_position_size_usd = float(np.mean(sizes))
        ts.median_position_size_usd = float(np.median(sizes))
        ts.max_position_size_usd = float(np.max(sizes))
        ts.total_volume_traded_usd = float(np.sum(sizes) * 2)  # Round trip

        # Trades per time period
        if entry_times[0] and entry_times[-1]:
            try:
                start = pd.to_datetime(entry_times[0])
                end = pd.to_datetime(entry_times[-1])
                days = (end - start).days or 1
                ts.trades_per_day = float(ts.total_trades / days)
                ts.trades_per_week = ts.trades_per_day * 7
                ts.trades_per_month = ts.trades_per_day * 30
            except Exception:
                pass

        return ts

    def _calculate_cost_metrics(self, trades: List[Dict]) -> CostMetrics:
        """Calculate comprehensive cost analysis metrics."""
        cm = CostMetrics()

        if not trades or len(trades) == 0:
            return cm

        # Aggregate costs
        fees = []
        slippage = []
        gas_costs = []
        mev_costs = []
        gross_pnls = []
        net_pnls = []
        sizes = []
        cex_costs = []
        dex_costs = []
        dex_l2_costs = []

        for trade in trades:
            if isinstance(trade, dict):
                fee = trade.get('fee_cost', trade.get('fees', 0))
                slip = trade.get('slippage_cost', trade.get('slippage', 0))
                gas = trade.get('gas_cost', 0)
                mev = trade.get('mev_cost', 0)
                gross = trade.get('gross_pnl', 0)
                net = trade.get('net_pnl', gross - fee - slip - gas - mev)
                size = trade.get('position_size_usd', 10000)
                venue = str(trade.get('venue_type', 'cex')).lower()
            else:
                fee = getattr(trade, 'fee_cost', getattr(trade, 'fees', 0))
                slip = getattr(trade, 'slippage_cost', getattr(trade, 'slippage', 0))
                gas = getattr(trade, 'gas_cost', 0)
                mev = getattr(trade, 'mev_cost', 0)
                gross = getattr(trade, 'gross_pnl', 0)
                net = getattr(trade, 'net_pnl', gross - fee - slip - gas - mev)
                size = getattr(trade, 'position_size_usd', 10000)
                venue = str(getattr(trade, 'venue_type', 'cex')).lower()

            fees.append(fee)
            slippage.append(slip)
            gas_costs.append(gas)
            mev_costs.append(mev)
            gross_pnls.append(gross)
            net_pnls.append(net)
            sizes.append(size)

            total_cost = fee + slip + gas + mev
            cost_bps = (total_cost / size * 10000) if size > 0 else 0

            if 'dex_l2' in venue or 'l2' in venue:
                dex_l2_costs.append(cost_bps)
            elif 'dex' in venue:
                dex_costs.append(cost_bps)
            else:
                cex_costs.append(cost_bps)

        # Total costs
        cm.total_fees_usd = float(sum(fees))
        cm.total_slippage_usd = float(sum(slippage))
        cm.total_gas_costs_usd = float(sum(gas_costs))
        cm.total_mev_costs_usd = float(sum(mev_costs))
        cm.total_costs_usd = float(
            cm.total_fees_usd + cm.total_slippage_usd +
            cm.total_gas_costs_usd + cm.total_mev_costs_usd
        )

        total_size = sum(sizes)
        if total_size > 0:
            cm.total_costs_bps = float(cm.total_costs_usd / total_size * 10000)

        # P&L metrics
        cm.gross_pnl_usd = float(sum(gross_pnls))
        cm.net_pnl_usd = float(sum(net_pnls))

        if cm.gross_pnl_usd > 0:
            cm.cost_adjusted_return = float(cm.net_pnl_usd / cm.gross_pnl_usd)
            cm.transaction_cost_drag_pct = float(cm.total_costs_usd / cm.gross_pnl_usd * 100)
            cm.fees_as_pct_of_gross = float(cm.total_fees_usd / cm.gross_pnl_usd * 100)
            cm.slippage_as_pct_of_gross = float(cm.total_slippage_usd / cm.gross_pnl_usd * 100)
            cm.gas_as_pct_of_gross = float(cm.total_gas_costs_usd / cm.gross_pnl_usd * 100)
            cm.mev_as_pct_of_gross = float(cm.total_mev_costs_usd / cm.gross_pnl_usd * 100)

        # DEX gas cost impact
        dex_gross = sum(
            abs(gross_pnls[i]) for i, t in enumerate(trades)
            if 'dex' in str(t.get('venue_type', '') if isinstance(t, dict)
                           else getattr(t, 'venue_type', '')).lower()
        )
        if dex_gross > 0:
            dex_gas = cm.total_gas_costs_usd + cm.total_mev_costs_usd
            cm.gas_cost_impact_pct = float(dex_gas / dex_gross * 100)

        # Per-trade costs
        if len(trades) > 0:
            cm.avg_cost_per_trade_usd = float(cm.total_costs_usd / len(trades))
            cm.avg_cost_per_trade_bps = float(cm.total_costs_bps / len(trades))

        # Venue-specific averages
        if cex_costs:
            cm.cex_avg_cost_bps = float(np.mean(cex_costs))
        if dex_costs:
            cm.dex_avg_cost_bps = float(np.mean(dex_costs))
        if dex_l2_costs:
            cm.dex_l2_avg_cost_bps = float(np.mean(dex_l2_costs))

        # Gas cost analysis
        if gas_costs:
            cm.avg_gas_cost_usd = float(np.mean(gas_costs))
            cm.max_gas_cost_usd = float(np.max(gas_costs))
            if len(gas_costs) > 1:
                cm.gas_cost_volatility = float(np.std(gas_costs))

        return cm

    def _calculate_risk_metrics(
        self,
        returns: np.ndarray,
        btc_returns: Optional[pd.Series] = None,
        eth_returns: Optional[pd.Series] = None
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics including VaR."""
        rm = RiskMetrics()

        if len(returns) < 10:
            return rm

        # Historical VaR
        rm.var_95 = _calculate_var(returns, 0.95)
        rm.var_99 = _calculate_var(returns, 0.99)
        rm.var_999 = _calculate_var(returns, 0.999)

        # Conditional VaR (Expected Shortfall)
        rm.cvar_95 = _calculate_cvar(returns, 0.95)
        rm.cvar_99 = _calculate_cvar(returns, 0.99)
        rm.cvar_999 = _calculate_cvar(returns, 0.999)

        # Parametric VaR
        rm.parametric_var_95 = _calculate_parametric_var(returns, 0.95)
        rm.parametric_var_99 = _calculate_parametric_var(returns, 0.99)

        # Cornish-Fisher VaR
        rm.cornish_fisher_var_95 = _calculate_cornish_fisher_var(returns, 0.95)
        rm.cornish_fisher_var_99 = _calculate_cornish_fisher_var(returns, 0.99)

        # Higher moments
        rm.skewness = float(stats.skew(returns))
        rm.kurtosis = float(stats.kurtosis(returns))
        rm.excess_kurtosis = rm.kurtosis - 3

        # Tail ratios
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        p99 = np.percentile(returns, 99)
        p1 = np.percentile(returns, 1)

        if abs(p5) > 0:
            rm.left_tail_ratio = float(abs(p1 / p5))
        if p95 > 0:
            rm.right_tail_ratio = float(p99 / p95)
        if abs(p5) > 0 and p95 > 0:
            rm.tail_dependence = float(abs(p99 / p95) / abs(p1 / p5))

        # Beta calculations
        if btc_returns is not None and len(btc_returns) >= len(returns):
            btc_arr = btc_returns.values[:len(returns)]
            if np.std(btc_arr) > 0:
                cov = np.cov(returns, btc_arr)[0, 1]
                rm.btc_beta = float(cov / np.var(btc_arr))
                rm.market_correlation = float(np.corrcoef(returns, btc_arr)[0, 1])

        if eth_returns is not None and len(eth_returns) >= len(returns):
            eth_arr = eth_returns.values[:len(returns)]
            if np.std(eth_arr) > 0:
                cov = np.cov(returns, eth_arr)[0, 1]
                rm.eth_beta = float(cov / np.var(eth_arr))

        # Risk of ruin
        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
        winners = returns[returns > 0]
        losers = returns[returns < 0]
        avg_win = np.mean(winners) if len(winners) > 0 else 0
        avg_loss = abs(np.mean(losers)) if len(losers) > 0 else 1
        wl_ratio = avg_win / avg_loss if avg_loss > 0 else 1

        rm.risk_of_ruin_10pct = _calculate_risk_of_ruin(win_rate, wl_ratio, 0.02, 0.10)
        rm.risk_of_ruin_20pct = _calculate_risk_of_ruin(win_rate, wl_ratio, 0.02, 0.20)
        rm.risk_of_ruin_50pct = _calculate_risk_of_ruin(win_rate, wl_ratio, 0.02, 0.50)

        return rm

    def _calculate_statistical_metrics(self, returns: np.ndarray) -> StatisticalMetrics:
        """Calculate statistical test results and distribution metrics."""
        sm = StatisticalMetrics()

        if len(returns) < self.min_trades_for_stats:
            return sm

        # Distribution metrics
        sm.mean = float(np.mean(returns))
        sm.median = float(np.median(returns))
        sm.std_dev = float(np.std(returns))
        sm.variance = float(np.var(returns))
        sm.skewness = float(stats.skew(returns))
        sm.kurtosis = float(stats.kurtosis(returns))

        # Percentiles
        sm.percentile_1 = float(np.percentile(returns, 1))
        sm.percentile_5 = float(np.percentile(returns, 5))
        sm.percentile_10 = float(np.percentile(returns, 10))
        sm.percentile_25 = float(np.percentile(returns, 25))
        sm.percentile_50 = float(np.percentile(returns, 50))
        sm.percentile_75 = float(np.percentile(returns, 75))
        sm.percentile_90 = float(np.percentile(returns, 90))
        sm.percentile_95 = float(np.percentile(returns, 95))
        sm.percentile_99 = float(np.percentile(returns, 99))
        sm.iqr = sm.percentile_75 - sm.percentile_25

        # Normality tests
        try:
            jb_stat, jb_p = stats.jarque_bera(returns)
            sm.jarque_bera_statistic = float(jb_stat)
            sm.jarque_bera_pvalue = float(jb_p)
        except Exception:
            pass

        if len(returns) <= 5000:  # Shapiro-Wilk limit
            try:
                sw_stat, sw_p = stats.shapiro(returns)
                sm.shapiro_wilk_statistic = float(sw_stat)
                sm.shapiro_wilk_pvalue = float(sw_p)
            except Exception:
                pass

        sm.is_normal_5pct = (
            sm.jarque_bera_pvalue > 0.05 if sm.jarque_bera_pvalue > 0 else False
        )

        # Stationarity tests (simplified - ADF approximation)
        try:
            from statsmodels.tsa.stattools import adfuller, kpss
            adf_result = adfuller(returns, maxlag=min(10, len(returns)//5))
            sm.adf_statistic = float(adf_result[0])
            sm.adf_pvalue = float(adf_result[1])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_result = kpss(returns, regression='c', nlags='auto')
                sm.kpss_statistic = float(kpss_result[0])
                sm.kpss_pvalue = float(kpss_result[1])

            sm.is_stationary = sm.adf_pvalue < 0.05 and sm.kpss_pvalue > 0.05
        except ImportError:
            # Statsmodels not available, use simple variance ratio test
            n = len(returns)
            if n > 20:
                var_1 = np.var(returns)
                var_10 = np.var([np.sum(returns[i:i+10]) for i in range(0, n-10, 10)])
                vr = var_10 / (10 * var_1) if var_1 > 0 else 1
                sm.is_stationary = 0.5 < vr < 1.5  # Rough heuristic
        except Exception:
            pass

        # Autocorrelation
        if len(returns) > 20:
            try:
                sm.autocorr_lag1 = float(pd.Series(returns).autocorr(lag=1))
                sm.autocorr_lag5 = float(pd.Series(returns).autocorr(lag=5))
                sm.autocorr_lag10 = float(pd.Series(returns).autocorr(lag=10))
                sm.autocorr_lag20 = float(pd.Series(returns).autocorr(lag=20))
            except Exception:
                pass

            # Ljung-Box test
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb_result = acorr_ljungbox(returns, lags=[10], return_df=True)
                sm.ljung_box_statistic = float(lb_result['lb_stat'].iloc[0])
                sm.ljung_box_pvalue = float(lb_result['lb_pvalue'].iloc[0])
            except ImportError:
                pass
            except Exception:
                pass

        return sm

    def _calculate_execution_metrics(self, trades: List[Dict]) -> ExecutionMetrics:
        """Calculate execution quality metrics."""
        em = ExecutionMetrics()

        if not trades or len(trades) == 0:
            return em

        slippage_bps = []
        exec_times = []
        cex_count = 0
        dex_count = 0
        filled_count = 0
        partial_count = 0
        failed_count = 0
        limit_orders = 0
        limit_filled = 0
        price_improved = 0
        price_improvement_bps = []

        for trade in trades:
            if isinstance(trade, dict):
                slip = trade.get('slippage_bps', trade.get('slippage_cost', 0) /
                               trade.get('position_size_usd', 10000) * 10000)
                exec_time = trade.get('execution_time_ms', 100)
                venue = str(trade.get('venue_type', 'cex')).lower()
                fill_status = trade.get('fill_status', 'filled')
                order_type = trade.get('order_type', 'market')
                price_imp = trade.get('price_improvement_bps', 0)
            else:
                slip = getattr(trade, 'slippage_bps', 0)
                exec_time = getattr(trade, 'execution_time_ms', 100)
                venue = str(getattr(trade, 'venue_type', 'cex')).lower()
                fill_status = getattr(trade, 'fill_status', 'filled')
                order_type = getattr(trade, 'order_type', 'market')
                price_imp = getattr(trade, 'price_improvement_bps', 0)

            slippage_bps.append(slip)
            exec_times.append(exec_time)

            if 'dex' in venue:
                dex_count += 1
            else:
                cex_count += 1

            if fill_status == 'filled':
                filled_count += 1
            elif fill_status == 'partial':
                partial_count += 1
            else:
                failed_count += 1

            if order_type == 'limit':
                limit_orders += 1
                if fill_status == 'filled':
                    limit_filled += 1

            if price_imp > 0:
                price_improved += 1
                price_improvement_bps.append(price_imp)

        total = len(trades)

        # Fill metrics
        em.avg_fill_rate = float(filled_count / total)
        em.partial_fill_rate = float(partial_count / total)
        em.failed_order_rate = float(failed_count / total)

        # Slippage analysis
        slippage_arr = np.array(slippage_bps)
        em.avg_slippage_bps = float(np.mean(slippage_arr))
        em.median_slippage_bps = float(np.median(slippage_arr))
        em.max_slippage_bps = float(np.max(slippage_arr))
        if len(slippage_arr) > 1:
            em.slippage_std_dev = float(np.std(slippage_arr))

        # Timing metrics
        exec_arr = np.array(exec_times)
        em.avg_execution_time_ms = float(np.mean(exec_arr))
        em.median_execution_time_ms = float(np.median(exec_arr))
        em.max_execution_time_ms = float(np.max(exec_arr))

        # Price improvement
        em.price_improvement_rate = float(price_improved / total)
        if price_improvement_bps:
            em.avg_price_improvement_bps = float(np.mean(price_improvement_bps))

        # Order types
        em.market_order_pct = float((total - limit_orders) / total)
        em.limit_order_pct = float(limit_orders / total)
        if limit_orders > 0:
            em.limit_order_fill_rate = float(limit_filled / limit_orders)

        # Venue distribution
        em.cex_execution_pct = float(cex_count / total)
        em.dex_execution_pct = float(dex_count / total)

        return em

    def _calculate_time_based_metrics(
        self,
        returns: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> TimeBasedMetrics:
        """Calculate time-based performance patterns."""
        tbm = TimeBasedMetrics()

        if len(returns) == 0 or len(timestamps) == 0:
            return tbm

        # Create DataFrame for easier grouping
        df = pd.DataFrame({
            'return': returns,
            'timestamp': timestamps
        })
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day_name()
        df['month'] = df['timestamp'].dt.month_name()
        df['quarter'] = df['timestamp'].dt.quarter.apply(lambda x: f"Q{x}")
        df['year'] = df['timestamp'].dt.year

        # Hourly patterns
        hourly = df.groupby('hour')['return'].mean()
        tbm.hourly_returns = hourly.to_dict()
        if len(hourly) > 0:
            tbm.best_hour = int(hourly.idxmax())
            tbm.worst_hour = int(hourly.idxmin())
            tbm.best_hour_return = float(hourly.max())
            tbm.worst_hour_return = float(hourly.min())

        # Daily patterns (day of week)
        daily = df.groupby('day')['return'].mean()
        tbm.daily_returns = daily.to_dict()
        if len(daily) > 0:
            tbm.best_day = str(daily.idxmax())
            tbm.worst_day = str(daily.idxmin())
            tbm.best_day_return = float(daily.max())
            tbm.worst_day_return = float(daily.min())

        # Monthly patterns
        monthly = df.groupby('month')['return'].mean()
        tbm.monthly_returns = monthly.to_dict()
        if len(monthly) > 0:
            tbm.best_month = str(monthly.idxmax())
            tbm.worst_month = str(monthly.idxmin())
            tbm.best_month_return = float(monthly.max())
            tbm.worst_month_return = float(monthly.min())

        # Quarterly patterns
        quarterly = df.groupby('quarter')['return'].mean()
        tbm.quarterly_returns = quarterly.to_dict()
        if len(quarterly) > 0:
            tbm.best_quarter = str(quarterly.idxmax())
            tbm.worst_quarter = str(quarterly.idxmin())

        # Yearly returns
        yearly = df.groupby('year')['return'].sum()
        tbm.yearly_returns = yearly.to_dict()

        # Seasonality detection (simple F-test approximation)
        if len(hourly) >= 12:
            hourly_var = hourly.var()
            overall_var = df['return'].var()
            if overall_var > 0:
                f_stat = hourly_var / (overall_var / len(hourly))
                tbm.has_hourly_seasonality = f_stat > 2.0

        if len(daily) >= 5:
            daily_var = daily.var()
            overall_var = df['return'].var()
            if overall_var > 0:
                f_stat = daily_var / (overall_var / len(daily))
                tbm.has_daily_seasonality = f_stat > 2.0

        if len(monthly) >= 6:
            monthly_var = monthly.var()
            overall_var = df['return'].var()
            if overall_var > 0:
                f_stat = monthly_var / (overall_var / len(monthly))
                tbm.has_monthly_seasonality = f_stat > 2.0

        return tbm

    def _calculate_rolling_metrics(self, returns: np.ndarray) -> RollingMetrics:
        """Calculate rolling window performance metrics."""
        rm = RollingMetrics()

        n = len(returns)
        ann_factor = np.sqrt(self.annualization_factor)

        # Helper function for rolling Sharpe
        def rolling_sharpe(window_returns):
            if np.std(window_returns) == 0:
                return 0.0
            return float(
                (np.mean(window_returns) - self._daily_rf) /
                np.std(window_returns) * ann_factor
            )

        def rolling_max_dd(window_returns):
            dd, _ = _calculate_drawdown_series(window_returns)
            return float(np.max(dd))

        # 30-day rolling
        window_30 = self.rolling_windows['short']
        if n >= window_30:
            for i in range(window_30, n + 1):
                w = returns[i-window_30:i]
                rm.rolling_30d_sharpe.append(rolling_sharpe(w))
                rm.rolling_30d_return.append(float(np.sum(w)))
                rm.rolling_30d_volatility.append(float(np.std(w) * ann_factor))
                rm.rolling_30d_max_dd.append(rolling_max_dd(w))

        # 90-day rolling
        window_90 = self.rolling_windows['medium']
        if n >= window_90:
            for i in range(window_90, n + 1):
                w = returns[i-window_90:i]
                rm.rolling_90d_sharpe.append(rolling_sharpe(w))
                rm.rolling_90d_return.append(float(np.sum(w)))
                rm.rolling_90d_volatility.append(float(np.std(w) * ann_factor))
                rm.rolling_90d_max_dd.append(rolling_max_dd(w))

        # 180-day rolling
        window_180 = self.rolling_windows['long']
        if n >= window_180:
            for i in range(window_180, n + 1):
                w = returns[i-window_180:i]
                rm.rolling_180d_sharpe.append(rolling_sharpe(w))
                rm.rolling_180d_return.append(float(np.sum(w)))
                rm.rolling_180d_volatility.append(float(np.std(w) * ann_factor))
                rm.rolling_180d_max_dd.append(rolling_max_dd(w))

        # 365-day rolling
        window_365 = self.rolling_windows['yearly']
        if n >= window_365:
            for i in range(window_365, n + 1):
                w = returns[i-window_365:i]
                rm.rolling_365d_sharpe.append(rolling_sharpe(w))
                rm.rolling_365d_return.append(float(np.sum(w)))
                rm.rolling_365d_volatility.append(float(np.std(w) * ann_factor))
                rm.rolling_365d_max_dd.append(rolling_max_dd(w))

        # Stability metrics
        if rm.rolling_90d_sharpe:
            rm.sharpe_stability = float(np.std(rm.rolling_90d_sharpe))
            rm.sharpe_5th_percentile = float(np.percentile(rm.rolling_90d_sharpe, 5))
            rm.sharpe_95th_percentile = float(np.percentile(rm.rolling_90d_sharpe, 95))

        if rm.rolling_90d_return:
            rm.return_stability = float(np.std(rm.rolling_90d_return))

        if rm.rolling_90d_volatility:
            rm.volatility_stability = float(np.std(rm.rolling_90d_volatility))

        return rm

    # =========================================================================
    # PUBLIC METHODS - Venue and Regime Breakdown
    # =========================================================================

    def calculate_venue_breakdown(
        self,
        trades: List[Dict],
        pair_info: Optional[Dict] = None
    ) -> VenueSpecificMetrics:
        """
        Calculate venue-specific metrics breakdown.

        PDF REQUIRED: "Calculate metrics separately for:
        1. CEX-only pairs
        2. DEX-only pairs
        3. Mixed (CEX-DEX) pairs
        4. Combined portfolio"

        Args:
            trades: List of trade dictionaries
            pair_info: Optional pair metadata with venue info

        Returns:
            VenueSpecificMetrics with breakdown by venue type
        """
        # Categorize trades by venue
        venue_trades = {
            'cex': [],
            'dex': [],
            'dex_l2': [],
            'hybrid': [],
            'mixed': []
        }

        for trade in trades:
            if isinstance(trade, dict):
                venue_type = str(trade.get('venue_type', 'cex')).lower()
            else:
                venue_type = str(getattr(trade, 'venue_type', 'cex')).lower()

            if 'dex_l2' in venue_type or 'l2' in venue_type:
                venue_trades['dex_l2'].append(trade)
            elif 'dex' in venue_type:
                venue_trades['dex'].append(trade)
            elif 'hybrid' in venue_type:
                venue_trades['hybrid'].append(trade)
            elif 'mixed' in venue_type:
                venue_trades['mixed'].append(trade)
            else:
                venue_trades['cex'].append(trade)

        # Calculate metrics for each venue
        vsm = VenueSpecificMetrics()

        if venue_trades['cex']:
            vsm.cex_metrics = self.calculate_all_metrics(trades=venue_trades['cex'])
        if venue_trades['dex']:
            vsm.dex_metrics = self.calculate_all_metrics(trades=venue_trades['dex'])
        if venue_trades['dex_l2']:
            vsm.dex_l2_metrics = self.calculate_all_metrics(trades=venue_trades['dex_l2'])
        if venue_trades['hybrid']:
            vsm.hybrid_metrics = self.calculate_all_metrics(trades=venue_trades['hybrid'])
        if venue_trades['mixed']:
            vsm.mixed_metrics = self.calculate_all_metrics(trades=venue_trades['mixed'])

        # Combined metrics
        vsm.combined_metrics = self.calculate_all_metrics(trades=trades)

        # Comparison summaries
        vsm.venue_sharpe_comparison = {
            'CEX': vsm.cex_metrics.core.sharpe_ratio,
            'DEX': vsm.dex_metrics.core.sharpe_ratio,
            'DEX_L2': vsm.dex_l2_metrics.core.sharpe_ratio,
            'Hybrid': vsm.hybrid_metrics.core.sharpe_ratio,
            'Mixed': vsm.mixed_metrics.core.sharpe_ratio,
            'Combined': vsm.combined_metrics.core.sharpe_ratio,
        }

        vsm.venue_return_comparison = {
            'CEX': vsm.cex_metrics.core.total_return,
            'DEX': vsm.dex_metrics.core.total_return,
            'DEX_L2': vsm.dex_l2_metrics.core.total_return,
            'Hybrid': vsm.hybrid_metrics.core.total_return,
            'Mixed': vsm.mixed_metrics.core.total_return,
            'Combined': vsm.combined_metrics.core.total_return,
        }

        vsm.venue_cost_comparison = {
            'CEX': vsm.cex_metrics.core.transaction_cost_drag_pct,
            'DEX': vsm.dex_metrics.core.transaction_cost_drag_pct,
            'DEX_L2': vsm.dex_l2_metrics.core.transaction_cost_drag_pct,
            'Hybrid': vsm.hybrid_metrics.core.transaction_cost_drag_pct,
            'Mixed': vsm.mixed_metrics.core.transaction_cost_drag_pct,
            'Combined': vsm.combined_metrics.core.transaction_cost_drag_pct,
        }

        logger.info(
            f"Calculated venue breakdown: CEX={len(venue_trades['cex'])}, "
            f"DEX={len(venue_trades['dex'])}, DEX_L2={len(venue_trades['dex_l2'])}, "
            f"Hybrid={len(venue_trades['hybrid'])}, Mixed={len(venue_trades['mixed'])}"
        )

        return vsm

    def calculate_regime_breakdown(
        self,
        trades: List[Dict],
        btc_returns: pd.Series,
        volatility_series: Optional[pd.Series] = None
    ) -> RegimeSpecificMetrics:
        """
        Calculate regime-specific metrics breakdown.

        Classifies each trade into a market regime and calculates
        performance metrics for each regime type.

        Regimes:
        - Bull: BTC +20% from 30-day low
        - Bear: BTC -20% from 30-day high
        - Sideways: Neither bull nor bear
        - High Vol: Volatility > 80th percentile
        - Low Vol: Volatility < 20th percentile
        - Crisis: Defined crisis periods (see crisis_analyzer)
        - Recovery: 30 days after crisis ends

        Args:
            trades: List of trade dictionaries
            btc_returns: BTC daily returns series
            volatility_series: Optional realized volatility series

        Returns:
            RegimeSpecificMetrics with breakdown by regime
        """
        # Build regime labels for each timestamp
        btc_cumulative = (1 + btc_returns).cumprod()

        # Calculate rolling metrics for regime detection
        rolling_30d_high = btc_cumulative.rolling(30).max()
        rolling_30d_low = btc_cumulative.rolling(30).min()

        if volatility_series is None:
            # Calculate 30-day rolling volatility from BTC returns
            volatility_series = btc_returns.rolling(30).std() * np.sqrt(365)

        vol_80pct = volatility_series.quantile(0.80)
        vol_20pct = volatility_series.quantile(0.20)

        # Crisis periods (simplified - should integrate with crisis_analyzer)
        crisis_periods = [
            (pd.Timestamp('2020-03-01'), pd.Timestamp('2020-04-15')),  # COVID
            (pd.Timestamp('2021-05-10'), pd.Timestamp('2021-06-15')),  # May 2021
            (pd.Timestamp('2022-05-01'), pd.Timestamp('2022-06-30')),  # UST/Luna
            (pd.Timestamp('2022-11-01'), pd.Timestamp('2022-12-31')),  # FTX
            (pd.Timestamp('2023-03-01'), pd.Timestamp('2023-03-31')),  # SVB/USDC
        ]

        def is_crisis(ts):
            for start, end in crisis_periods:
                if start <= ts <= end:
                    return True
            return False

        def is_recovery(ts):
            for start, end in crisis_periods:
                if end < ts <= end + timedelta(days=30):
                    return True
            return False

        # Categorize trades by regime
        regime_trades = {
            'bull': [],
            'bear': [],
            'sideways': [],
            'high_vol': [],
            'low_vol': [],
            'crisis': [],
            'recovery': []
        }

        regime_counts = defaultdict(int)

        for trade in trades:
            if isinstance(trade, dict):
                entry_time = trade.get('entry_time')
            else:
                entry_time = getattr(trade, 'entry_time', None)

            if entry_time is None:
                regime_trades['sideways'].append(trade)
                continue

            ts = pd.to_datetime(entry_time)

            # Check crisis/recovery first
            if is_crisis(ts):
                regime_trades['crisis'].append(trade)
                regime_counts['crisis'] += 1
                continue

            if is_recovery(ts):
                regime_trades['recovery'].append(trade)
                regime_counts['recovery'] += 1
                continue

            # Get market state at entry
            try:
                idx = btc_cumulative.index.get_indexer([ts], method='nearest')[0]
                current = btc_cumulative.iloc[idx]
                high_30d = rolling_30d_high.iloc[idx]
                low_30d = rolling_30d_low.iloc[idx]
                vol = volatility_series.iloc[idx]

                # Bull/Bear/Sideways classification
                if current > low_30d * 1.20:  # +20% from low
                    regime_trades['bull'].append(trade)
                    regime_counts['bull'] += 1
                elif current < high_30d * 0.80:  # -20% from high
                    regime_trades['bear'].append(trade)
                    regime_counts['bear'] += 1
                else:
                    regime_trades['sideways'].append(trade)
                    regime_counts['sideways'] += 1

                # High/Low Vol classification (can overlap with trend)
                if vol > vol_80pct:
                    regime_trades['high_vol'].append(trade)
                    regime_counts['high_vol'] += 1
                elif vol < vol_20pct:
                    regime_trades['low_vol'].append(trade)
                    regime_counts['low_vol'] += 1

            except Exception:
                regime_trades['sideways'].append(trade)
                regime_counts['sideways'] += 1

        # Calculate metrics for each regime
        rsm = RegimeSpecificMetrics()

        if regime_trades['bull']:
            rsm.bull_metrics = self.calculate_all_metrics(trades=regime_trades['bull'])
        if regime_trades['bear']:
            rsm.bear_metrics = self.calculate_all_metrics(trades=regime_trades['bear'])
        if regime_trades['sideways']:
            rsm.sideways_metrics = self.calculate_all_metrics(trades=regime_trades['sideways'])
        if regime_trades['high_vol']:
            rsm.high_vol_metrics = self.calculate_all_metrics(trades=regime_trades['high_vol'])
        if regime_trades['low_vol']:
            rsm.low_vol_metrics = self.calculate_all_metrics(trades=regime_trades['low_vol'])
        if regime_trades['crisis']:
            rsm.crisis_metrics = self.calculate_all_metrics(trades=regime_trades['crisis'])
        if regime_trades['recovery']:
            rsm.recovery_metrics = self.calculate_all_metrics(trades=regime_trades['recovery'])

        # Calculate regime frequencies
        total_trades = len(trades)
        if total_trades > 0:
            rsm.regime_frequencies = {
                regime: count / total_trades
                for regime, count in regime_counts.items()
            }

        logger.info(
            f"Calculated regime breakdown: "
            f"Bull={len(regime_trades['bull'])}, Bear={len(regime_trades['bear'])}, "
            f"Sideways={len(regime_trades['sideways'])}, "
            f"HighVol={len(regime_trades['high_vol'])}, LowVol={len(regime_trades['low_vol'])}, "
            f"Crisis={len(regime_trades['crisis'])}, Recovery={len(regime_trades['recovery'])}"
        )

        return rsm

    def create_metrics_report(
        self,
        metrics: PDFCompliantMetrics,
        venue_metrics: Optional[VenueSpecificMetrics] = None,
        regime_metrics: Optional[RegimeSpecificMetrics] = None,
        title: str = "Full Performance Metrics Report"
    ) -> str:
        """
        Create comprehensive formatted metrics report.

        PDF REQUIRED: Comprehensive results report (5-6 pages).

        Args:
            metrics: Primary PDFCompliantMetrics
            venue_metrics: Optional venue breakdown
            regime_metrics: Optional regime breakdown
            title: Report title

        Returns:
            Formatted string report
        """
        lines = [
            "=" * 80,
            title.center(80),
            "(PDF Section 2.4 Compliant - 60+ Metrics)".center(80),
            "=" * 80,
            "",
            f"Report Generated: {metrics.calculation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data Period: {metrics.data_start_date} to {metrics.data_end_date}"
            if metrics.data_start_date else "",
            f"Total Days: {metrics.total_days}",
            "",
            "=" * 80,
            "SECTION 1: CORE METRICS (PDF REQUIRED)",
            "=" * 80,
            "",
            f"  Sharpe Ratio:              {metrics.core.sharpe_ratio:>10.3f}",
            f"  Sortino Ratio:             {metrics.core.sortino_ratio:>10.3f}",
            f"  Maximum Drawdown:          {metrics.core.max_drawdown:>10.2%}",
            f"  Avg Holding Period:        {metrics.core.avg_holding_period_hours:>10.1f} hours",
            f"  Annualized Turnover:       {metrics.core.annualized_turnover:>10.1f}x",
            f"  Transaction Cost Drag:     {metrics.core.transaction_cost_drag_pct:>10.1f}%",
            f"  Gas Cost Impact (DEX):     {metrics.core.gas_cost_impact_pct:>10.1f}%",
            "",
            "-" * 80,
            "Return Metrics:",
            f"  Total Return:              {metrics.core.total_return:>10.2%}",
            f"  Annualized Return:         {metrics.core.annualized_return:>10.2%}",
            f"  Cumulative Return:         {metrics.core.cumulative_return:>10.2%}",
            f"  CAGR:                      {metrics.core.cagr:>10.2%}",
            "",
            "Volatility Metrics:",
            f"  Annualized Volatility:     {metrics.core.annualized_volatility:>10.2%}",
            f"  Downside Volatility:       {metrics.core.downside_volatility:>10.2%}",
            f"  Upside Volatility:         {metrics.core.upside_volatility:>10.2%}",
            "",
            "=" * 80,
            "SECTION 2: RISK-ADJUSTED METRICS",
            "=" * 80,
            "",
            f"  Calmar Ratio:              {metrics.risk_adjusted.calmar_ratio:>10.3f}",
            f"  Omega Ratio:               {metrics.risk_adjusted.omega_ratio:>10.3f}",
            f"  Treynor Ratio:             {metrics.risk_adjusted.treynor_ratio:>10.3f}",
            f"  Information Ratio:         {metrics.risk_adjusted.information_ratio:>10.3f}",
            f"  Burke Ratio:               {metrics.risk_adjusted.burke_ratio:>10.3f}",
            f"  Sterling Ratio:            {metrics.risk_adjusted.sterling_ratio:>10.3f}",
            f"  Tail Ratio:                {metrics.risk_adjusted.tail_ratio:>10.3f}",
            f"  Gain-to-Pain Ratio:        {metrics.risk_adjusted.gain_to_pain_ratio:>10.3f}",
            f"  Ulcer Performance Index:   {metrics.risk_adjusted.ulcer_performance_index:>10.3f}",
            "",
            "=" * 80,
            "SECTION 3: DRAWDOWN ANALYSIS",
            "=" * 80,
            "",
            f"  Max Drawdown:              {metrics.drawdown.max_drawdown:>10.2%}",
            f"  Max DD Duration:           {metrics.drawdown.max_drawdown_duration_days:>10.1f} days",
            f"  Avg Drawdown:              {metrics.drawdown.avg_drawdown:>10.2%}",
            f"  Avg DD Duration:           {metrics.drawdown.avg_drawdown_duration_days:>10.1f} days",
            f"  Recovery Factor:           {metrics.drawdown.recovery_factor:>10.2f}",
            f"  Ulcer Index:               {metrics.drawdown.ulcer_index:>10.2f}",
            f"  Pain Index:                {metrics.drawdown.pain_index:>10.4f}",
            f"  Time Underwater:           {metrics.drawdown.pct_time_underwater:>10.1%}",
            "",
            "Drawdown Counts:",
            f"  Total Drawdowns:           {metrics.drawdown.num_drawdowns:>10d}",
            f"  Drawdowns > 5%:            {metrics.drawdown.num_drawdowns_gt_5pct:>10d}",
            f"  Drawdowns > 10%:           {metrics.drawdown.num_drawdowns_gt_10pct:>10d}",
            f"  Drawdowns > 20%:           {metrics.drawdown.num_drawdowns_gt_20pct:>10d}",
            "",
            "=" * 80,
            "SECTION 4: TRADE STATISTICS",
            "=" * 80,
            "",
            f"  Total Trades:              {metrics.trades.total_trades:>10d}",
            f"  Winning Trades:            {metrics.trades.winning_trades:>10d}",
            f"  Losing Trades:             {metrics.trades.losing_trades:>10d}",
            f"  Win Rate:                  {metrics.trades.win_rate:>10.1%}",
            f"  Profit Factor:             {metrics.trades.profit_factor:>10.2f}",
            f"  Payoff Ratio:              {metrics.trades.payoff_ratio:>10.2f}",
            f"  Expectancy:                {metrics.trades.expectancy:>10.4f}",
            f"  Kelly Criterion:           {metrics.trades.kelly_criterion:>10.2%}",
            "",
            "Trade Returns:",
            f"  Avg Trade Return:          {metrics.trades.avg_trade_return:>10.4f}",
            f"  Avg Winning Trade:         {metrics.trades.avg_winning_trade:>10.4f}",
            f"  Avg Losing Trade:          {metrics.trades.avg_losing_trade:>10.4f}",
            f"  Best Trade:                {metrics.trades.best_trade:>10.4f}",
            f"  Worst Trade:               {metrics.trades.worst_trade:>10.4f}",
            "",
            "Streaks:",
            f"  Max Consecutive Wins:      {metrics.trades.max_consecutive_wins:>10d}",
            f"  Max Consecutive Losses:    {metrics.trades.max_consecutive_losses:>10d}",
            "",
            "=" * 80,
            "SECTION 5: COST ANALYSIS",
            "=" * 80,
            "",
            f"  Total Costs (USD):         ${metrics.costs.total_costs_usd:>12,.2f}",
            f"  Total Fees:                ${metrics.costs.total_fees_usd:>12,.2f}",
            f"  Total Slippage:            ${metrics.costs.total_slippage_usd:>12,.2f}",
            f"  Total Gas Costs:           ${metrics.costs.total_gas_costs_usd:>12,.2f}",
            f"  Total MEV Costs:           ${metrics.costs.total_mev_costs_usd:>12,.2f}",
            "",
            f"  Gross P&L:                 ${metrics.costs.gross_pnl_usd:>12,.2f}",
            f"  Net P&L:                   ${metrics.costs.net_pnl_usd:>12,.2f}",
            f"  Cost-Adjusted Return:      {metrics.costs.cost_adjusted_return:>10.2%}",
            "",
            "Venue Cost Comparison (bps):",
            f"  CEX Avg Cost:              {metrics.costs.cex_avg_cost_bps:>10.1f} bps",
            f"  DEX Avg Cost:              {metrics.costs.dex_avg_cost_bps:>10.1f} bps",
            f"  DEX L2 Avg Cost:           {metrics.costs.dex_l2_avg_cost_bps:>10.1f} bps",
            "",
            "=" * 80,
            "SECTION 6: RISK METRICS (VaR & CVaR)",
            "=" * 80,
            "",
            "Historical VaR:",
            f"  VaR 95%:                   {metrics.risk.var_95:>10.4f}",
            f"  VaR 99%:                   {metrics.risk.var_99:>10.4f}",
            f"  VaR 99.9%:                 {metrics.risk.var_999:>10.4f}",
            "",
            "Expected Shortfall (CVaR):",
            f"  CVaR 95%:                  {metrics.risk.cvar_95:>10.4f}",
            f"  CVaR 99%:                  {metrics.risk.cvar_99:>10.4f}",
            f"  CVaR 99.9%:                {metrics.risk.cvar_999:>10.4f}",
            "",
            "Cornish-Fisher VaR (Adjusted):",
            f"  CF VaR 95%:                {metrics.risk.cornish_fisher_var_95:>10.4f}",
            f"  CF VaR 99%:                {metrics.risk.cornish_fisher_var_99:>10.4f}",
            "",
            "Market Exposure:",
            f"  BTC Beta:                  {metrics.risk.btc_beta:>10.3f}",
            f"  ETH Beta:                  {metrics.risk.eth_beta:>10.3f}",
            f"  Market Correlation:        {metrics.risk.market_correlation:>10.3f}",
            "",
            "Higher Moments:",
            f"  Skewness:                  {metrics.risk.skewness:>10.3f}",
            f"  Kurtosis:                  {metrics.risk.kurtosis:>10.3f}",
            f"  Excess Kurtosis:           {metrics.risk.excess_kurtosis:>10.3f}",
            "",
            "Risk of Ruin:",
            f"  RoR (10% loss):            {metrics.risk.risk_of_ruin_10pct:>10.2%}",
            f"  RoR (20% loss):            {metrics.risk.risk_of_ruin_20pct:>10.2%}",
            f"  RoR (50% loss):            {metrics.risk.risk_of_ruin_50pct:>10.2%}",
        ]

        # Add venue breakdown if available
        if venue_metrics:
            lines.extend([
                "",
                "=" * 80,
                "SECTION 7: VENUE-SPECIFIC BREAKDOWN (PDF REQUIRED)",
                "=" * 80,
                "",
                "Sharpe Ratios by Venue:",
            ])
            for venue, sharpe in venue_metrics.venue_sharpe_comparison.items():
                lines.append(f"  {venue:>12}:             {sharpe:>10.3f}")

            lines.extend([
                "",
                "Returns by Venue:",
            ])
            for venue, ret in venue_metrics.venue_return_comparison.items():
                lines.append(f"  {venue:>12}:             {ret:>10.2%}")

            lines.extend([
                "",
                "Cost Drag by Venue (%):",
            ])
            for venue, cost in venue_metrics.venue_cost_comparison.items():
                lines.append(f"  {venue:>12}:             {cost:>10.1f}%")

        # Add regime breakdown if available
        if regime_metrics:
            lines.extend([
                "",
                "=" * 80,
                "SECTION 8: REGIME-SPECIFIC BREAKDOWN",
                "=" * 80,
                "",
                "Performance by Regime:",
                "",
                f"  {'Regime':<15} {'Sharpe':>10} {'Return':>12} {'Max DD':>10} {'Win Rate':>10}",
                "-" * 60,
            ])

            regime_data = [
                ("Bull", regime_metrics.bull_metrics),
                ("Bear", regime_metrics.bear_metrics),
                ("Sideways", regime_metrics.sideways_metrics),
                ("High Vol", regime_metrics.high_vol_metrics),
                ("Low Vol", regime_metrics.low_vol_metrics),
                ("Crisis", regime_metrics.crisis_metrics),
                ("Recovery", regime_metrics.recovery_metrics),
            ]

            for regime_name, rm in regime_data:
                lines.append(
                    f"  {regime_name:<15} "
                    f"{rm.core.sharpe_ratio:>10.2f} "
                    f"{rm.core.total_return:>11.2%} "
                    f"{rm.drawdown.max_drawdown:>9.2%} "
                    f"{rm.trades.win_rate:>9.1%}"
                )

            if regime_metrics.regime_frequencies:
                lines.extend([
                    "",
                    "Regime Distribution:",
                ])
                for regime, freq in regime_metrics.regime_frequencies.items():
                    lines.append(f"  {regime.capitalize():<15}: {freq:>6.1%}")

        # Rolling metrics summary
        lines.extend([
            "",
            "=" * 80,
            "SECTION 9: ROLLING METRICS STABILITY",
            "=" * 80,
            "",
            f"  Sharpe Stability (std):    {metrics.rolling.sharpe_stability:>10.3f}",
            f"  Sharpe 5th Percentile:     {metrics.rolling.sharpe_5th_percentile:>10.3f}",
            f"  Sharpe 95th Percentile:    {metrics.rolling.sharpe_95th_percentile:>10.3f}",
            f"  Return Stability (std):    {metrics.rolling.return_stability:>10.4f}",
            f"  Volatility Stability:      {metrics.rolling.volatility_stability:>10.4f}",
            "",
            "Latest Rolling Sharpe:",
            f"  30-day:                    {metrics.rolling.to_dict()['rolling_30d_sharpe_latest']:>10.3f}",
            f"  90-day:                    {metrics.rolling.to_dict()['rolling_90d_sharpe_latest']:>10.3f}",
            f"  180-day:                   {metrics.rolling.to_dict()['rolling_180d_sharpe_latest']:>10.3f}",
            f"  365-day:                   {metrics.rolling.to_dict()['rolling_365d_sharpe_latest']:>10.3f}",
        ])

        # Time-based patterns
        if metrics.time_based.best_hour_return != 0:
            lines.extend([
                "",
                "=" * 80,
                "SECTION 10: TIME-BASED PATTERNS",
                "=" * 80,
                "",
                f"  Best Hour:                 {metrics.time_based.best_hour:>10d} ({metrics.time_based.best_hour_return:.4f})",
                f"  Worst Hour:                {metrics.time_based.worst_hour:>10d} ({metrics.time_based.worst_hour_return:.4f})",
                f"  Best Day:                  {metrics.time_based.best_day:>10s} ({metrics.time_based.best_day_return:.4f})",
                f"  Worst Day:                 {metrics.time_based.worst_day:>10s} ({metrics.time_based.worst_day_return:.4f})",
                f"  Best Month:                {metrics.time_based.best_month:>10s} ({metrics.time_based.best_month_return:.4f})",
                f"  Worst Month:               {metrics.time_based.worst_month:>10s} ({metrics.time_based.worst_month_return:.4f})",
                "",
                "Seasonality Detected:",
                f"  Hourly:                    {'Yes' if metrics.time_based.has_hourly_seasonality else 'No':>10s}",
                f"  Daily:                     {'Yes' if metrics.time_based.has_daily_seasonality else 'No':>10s}",
                f"  Monthly:                   {'Yes' if metrics.time_based.has_monthly_seasonality else 'No':>10s}",
            ])

        # Footer
        lines.extend([
            "",
            "=" * 80,
            "PDF COMPLIANCE CERTIFICATION",
            "=" * 80,
            "",
            "This report contains all metrics required by PDF Section 2.4:",
            "  [OK] Sharpe ratio (daily returns basis)",
            "  [OK] Sortino ratio",
            "  [OK] Maximum drawdown",
            "  [OK] Average holding period",
            "  [OK] Turnover (rebalancing frequency)",
            "  [OK] Transaction cost drag (% of gross returns)",
            "  [OK] Gas cost impact (for DEX pairs)",
            "",
            "Additional metrics calculated: 60+",
            "Venue breakdown: CEX, DEX, DEX L2, Hybrid, Mixed, Combined",
            "Regime breakdown: Bull, Bear, Sideways, High Vol, Low Vol, Crisis, Recovery",
            "",
            "=" * 80,
        ])

        return "\n".join(lines)

    def calculate(
        self,
        trades: Optional[List] = None,
        returns: Optional[Union[pd.Series, np.ndarray, List]] = None,
        equity_curve: Optional[pd.Series] = None
    ) -> PDFCompliantMetrics:
        """
        Legacy method - wrapper for calculate_all_metrics for backwards compatibility.
        """
        return self.calculate_all_metrics(
            returns=returns,
            trades=trades,
            equity_curve=equity_curve
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_metrics_calculator(
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
    **kwargs
) -> AdvancedMetricsCalculator:
    """
    Factory function to create a configured AdvancedMetricsCalculator.

    Args:
        risk_free_rate: Annual risk-free rate
        annualization_factor: Days per year for annualization
        **kwargs: Additional configuration options

    Returns:
        Configured AdvancedMetricsCalculator instance

    Example:
        >>> calc = create_metrics_calculator(risk_free_rate=0.05)
        >>> metrics = calc.calculate_all_metrics(returns=returns)
    """
    return AdvancedMetricsCalculator(
        risk_free_rate=risk_free_rate,
        annualization_factor=annualization_factor,
        **kwargs
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    'AdvancedMetricsCalculator',

    # Data classes
    'PDFCompliantMetrics',
    'VenueSpecificMetrics',
    'RegimeSpecificMetrics',
    'CoreMetrics',
    'RiskAdjustedMetrics',
    'DrawdownMetrics',
    'TradeStatistics',
    'CostMetrics',
    'CapacityMetrics',
    'RiskMetrics',
    'StatisticalMetrics',
    'ExecutionMetrics',
    'TimeBasedMetrics',
    'RollingMetrics',

    # Enums
    'MetricCategory',
    'VenueType',
    'RegimeType',
    'TimeFrame',

    # Factory function
    'create_metrics_calculator',

    # Constants
    'TRADING_DAYS_PER_YEAR',
    'DEFAULT_RISK_FREE_RATE',
    'VAR_CONFIDENCE_LEVELS',
    'ROLLING_WINDOWS',
    'VENUE_COSTS',
]
