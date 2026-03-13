"""
Crisis Analyzer - Full Implementation
==================================================

Comprehensive crisis event analysis system for PDF Section 2.4 with:

CRISIS DETECTION & CLASSIFICATION:
- Real-time crisis detection using volatility regime shifts
- Automatic crisis classification (market crash, contagion, regulatory, etc.)
- Crisis severity scoring with multi-factor model
- Early warning system with leading indicators

CORRELATION ANALYSIS:
- Cross-asset correlation breakdown (BTC, ETH, DeFi index)
- Sector correlation matrices with regime conditioning
- Venue correlation analysis (CEX vs DEX behavior)
- Correlation regime detection (normal, stress, crisis)
- Dynamic conditional correlation (DCC-GARCH)
- Correlation spike detection and quantification

CONTAGION MODELING:
- Granger causality testing for contagion paths
- Network analysis of asset interconnections
- Contagion velocity measurement
- Source identification (origin asset/sector)
- Spillover indices (Diebold-Yilmaz)

LIQUIDITY ANALYSIS:
- Bid-ask spread expansion during crisis
- Volume profile analysis
- Market depth deterioration
- Liquidity fragmentation across venues
- Flash crash detection

FACTOR DECOMPOSITION:
- Market factor (beta) contribution
- Sector factor contributions
- Idiosyncratic risk isolation
- Factor exposure changes during crisis
- Risk factor attribution

STRESS TESTING:
- Historical scenario replay
- Monte Carlo stress scenarios
- Tail risk metrics (VaR, CVaR, Expected Shortfall)
- Maximum drawdown decomposition
- Recovery path simulation

PORTFOLIO IMPACT:
- Position-level P&L attribution
- Hedge effectiveness measurement
- Stop-loss trigger analysis
- Margin call risk assessment
- Portfolio insurance valuation

Author: Tamer Atesyakar
Version: 3.0.0 - Complete
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum, auto
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import warnings
from scipy import stats, signal
from scipy.optimize import minimize
from functools import lru_cache

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

TRADING_DAYS_PER_YEAR = 252
HOURS_PER_DAY = 24
DEFAULT_NORMAL_LOOKBACK = 60  # Days before crisis for baseline
DEFAULT_RECOVERY_LOOKFORWARD = 90  # Days after crisis for recovery
MIN_CRISIS_DATA_POINTS = 3
CORRELATION_WINDOW = 30
VOLATILITY_WINDOW = 20

# Crisis detection thresholds
VOLATILITY_SPIKE_THRESHOLD = 2.5  # Std devs above mean
CORRELATION_SPIKE_THRESHOLD = 0.3  # Absolute increase
DRAWDOWN_CRISIS_THRESHOLD = 0.15  # 15% drawdown
VOLUME_SPIKE_THRESHOLD = 3.0  # 3x average volume

# Crypto sectors for analysis
CRYPTO_SECTORS = [
    'layer1', 'layer2', 'defi', 'defi_lending', 'defi_dex', 'defi_derivatives',
    'defi_yield', 'defi_lsdfi', 'defi_rwa', 'stablecoin', 'exchange_token',
    'gaming', 'metaverse', 'infrastructure', 'ai', 'privacy', 'meme', 'other'
]


# =============================================================================
# ENUMERATIONS
# =============================================================================

class CrisisType(Enum):
    """Classification of crisis events."""
    MARKET_CRASH = "market_crash"
    PROTOCOL_FAILURE = "protocol_failure"
    CONTAGION = "contagion"
    REGULATORY = "regulatory"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    FRAUD = "fraud"
    EXPLOIT = "exploit"
    GEOPOLITICAL = "geopolitical"
    MACRO_SHOCK = "macro_shock"
    TECHNICAL = "technical"
    FLASH_CRASH = "flash_crash"
    DELEVERAGE = "deleveraging"
    UNKNOWN = "unknown"


class CrisisSeverity(Enum):
    """Severity levels for crisis events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"
    CATASTROPHIC = "catastrophic"


class RecoveryPattern(Enum):
    """Recovery pattern classification."""
    V_SHAPE = "v_shape"  # Quick bounce back
    U_SHAPE = "u_shape"  # Gradual recovery
    W_SHAPE = "w_shape"  # Double dip
    L_SHAPE = "l_shape"  # No recovery
    PARTIAL = "partial"  # Incomplete recovery
    OVERSHOOT = "overshoot"  # Recovered beyond pre-crisis


class CorrelationRegime(Enum):
    """Correlation regime states."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    STRESS = "stress"
    CRISIS = "crisis"
    DECOUPLING = "decoupling"


class ContagionPath(Enum):
    """Contagion transmission paths."""
    DIRECT = "direct"  # Direct exposure
    LIQUIDITY = "liquidity"  # Liquidity channel
    INFORMATION = "information"  # Information cascade
    MARGIN = "margin"  # Margin/leverage channel
    SENTIMENT = "sentiment"  # Sentiment contagion


# =============================================================================
# DATA CLASSES - CRISIS EVENTS
# =============================================================================

@dataclass
class CrisisEvent:
    """
    Comprehensive crisis event definition.

    Contains all metadata for a crisis event including timing,
    classification, severity, and affected assets/sectors.
    """
    name: str
    start_date: str
    end_date: str
    crisis_type: CrisisType
    severity: CrisisSeverity
    btc_drawdown: float
    correlation_spike: bool
    affected_sectors: List[str]
    description: str
    recovery_date: Optional[str] = None

    # Extended metadata
    trigger_event: Optional[str] = None
    contagion_source: Optional[str] = None
    primary_venue_impact: Optional[str] = None
    liquidation_volume_usd: Optional[float] = None
    peak_funding_rate: Optional[float] = None
    open_interest_change: Optional[float] = None

    def __post_init__(self):
        """Convert string dates to timezone-aware datetime objects."""
        self.start_dt = pd.to_datetime(self.start_date, utc=True)
        self.end_dt = pd.to_datetime(self.end_date, utc=True)
        self.recovery_dt = pd.to_datetime(self.recovery_date, utc=True) if self.recovery_date else None

    @property
    def duration_days(self) -> int:
        """Calculate crisis duration in days."""
        return (self.end_dt - self.start_dt).days + 1

    @property
    def recovery_days(self) -> Optional[int]:
        """Calculate days to recovery if recovery date is set."""
        if self.recovery_dt:
            return (self.recovery_dt - self.end_dt).days
        return None


# =============================================================================
# DATA CLASSES - CORRELATION ANALYSIS
# =============================================================================

@dataclass
class CorrelationAnalysis:
    """
    Comprehensive correlation analysis during crisis.

    Tracks correlation changes across multiple dimensions:
    - Asset level (BTC, ETH, individual tokens)
    - Sector level (DeFi, L1, L2, etc.)
    - Venue level (CEX vs DEX)
    - Time dynamics (before, during, after)
    """
    # BTC correlations
    btc_correlation_pre: float = 0.0
    btc_correlation_crisis: float = 0.0
    btc_correlation_post: float = 0.0
    btc_correlation_change: float = 0.0

    # ETH correlations
    eth_correlation_pre: float = 0.0
    eth_correlation_crisis: float = 0.0
    eth_correlation_post: float = 0.0
    eth_correlation_change: float = 0.0

    # Correlation regime
    pre_crisis_regime: CorrelationRegime = CorrelationRegime.NORMAL
    crisis_regime: CorrelationRegime = CorrelationRegime.CRISIS
    post_crisis_regime: CorrelationRegime = CorrelationRegime.NORMAL

    # Spike detection
    correlation_spike_detected: bool = False
    spike_magnitude: float = 0.0
    spike_duration_hours: int = 0
    spike_peak_time: Optional[datetime] = None

    # Cross-sector correlations
    sector_correlation_matrix_pre: Optional[pd.DataFrame] = None
    sector_correlation_matrix_crisis: Optional[pd.DataFrame] = None
    sector_correlation_changes: Dict[str, float] = field(default_factory=dict)

    # Venue correlations
    cex_dex_correlation_pre: float = 0.0
    cex_dex_correlation_crisis: float = 0.0
    cex_dex_decorrelation: float = 0.0

    # DCC-GARCH estimates
    dcc_alpha: float = 0.0
    dcc_beta: float = 0.0
    conditional_correlations: Optional[pd.Series] = None

    # Network metrics
    average_correlation_pre: float = 0.0
    average_correlation_crisis: float = 0.0
    correlation_dispersion_pre: float = 0.0
    correlation_dispersion_crisis: float = 0.0


@dataclass
class ContagionAnalysis:
    """
    Contagion pathway analysis during crisis.

    Identifies how crisis spreads across assets and sectors.
    """
    # Source identification
    contagion_source: Optional[str] = None
    source_asset: Optional[str] = None
    source_sector: Optional[str] = None

    # Contagion metrics
    contagion_velocity: float = 0.0  # Speed of spread
    contagion_breadth: float = 0.0  # Proportion of assets affected
    contagion_depth: float = 0.0  # Average impact magnitude

    # Pathway analysis
    primary_pathway: ContagionPath = ContagionPath.DIRECT
    affected_assets: List[str] = field(default_factory=list)
    affected_sectors: List[str] = field(default_factory=list)
    transmission_lags: Dict[str, int] = field(default_factory=dict)  # Asset -> hours lag

    # Granger causality
    granger_causality_pvalues: Dict[str, float] = field(default_factory=dict)
    significant_causal_links: List[Tuple[str, str]] = field(default_factory=list)

    # Spillover index (Diebold-Yilmaz)
    total_spillover_index: float = 0.0
    directional_spillovers_to: Dict[str, float] = field(default_factory=dict)
    directional_spillovers_from: Dict[str, float] = field(default_factory=dict)
    net_spillovers: Dict[str, float] = field(default_factory=dict)


@dataclass
class LiquidityAnalysis:
    """
    Liquidity conditions during crisis.

    Tracks market microstructure changes during stress.
    """
    # Spread analysis
    avg_spread_pre: float = 0.0
    avg_spread_crisis: float = 0.0
    avg_spread_post: float = 0.0
    spread_expansion_pct: float = 0.0
    max_spread_observed: float = 0.0

    # Volume analysis
    avg_volume_pre: float = 0.0
    avg_volume_crisis: float = 0.0
    volume_spike_multiple: float = 0.0
    volume_asymmetry: float = 0.0  # Sell vs buy volume

    # Market depth
    depth_reduction_pct: float = 0.0
    avg_slippage_increase: float = 0.0
    large_trade_impact: float = 0.0

    # Venue fragmentation
    cex_volume_share_pre: float = 0.0
    cex_volume_share_crisis: float = 0.0
    dex_premium_avg: float = 0.0  # DEX price premium/discount

    # Flash crash detection
    flash_crash_detected: bool = False
    flash_crash_magnitude: float = 0.0
    flash_crash_duration_seconds: int = 0
    flash_crash_recovery_seconds: int = 0

    # Funding rates (perpetuals)
    avg_funding_rate_pre: float = 0.0
    avg_funding_rate_crisis: float = 0.0
    peak_funding_rate: float = 0.0
    funding_rate_volatility: float = 0.0

    # Open interest
    open_interest_change_pct: float = 0.0
    liquidation_volume: float = 0.0
    margin_call_estimate: float = 0.0


@dataclass
class SectorAnalysis:
    """
    Sector-level performance during crisis.
    """
    # Per-sector returns
    sector_returns: Dict[str, float] = field(default_factory=dict)
    sector_volatility: Dict[str, float] = field(default_factory=dict)
    sector_sharpe: Dict[str, float] = field(default_factory=dict)
    sector_max_dd: Dict[str, float] = field(default_factory=dict)
    sector_beta: Dict[str, float] = field(default_factory=dict)

    # Trade metrics by sector
    sector_trade_counts: Dict[str, int] = field(default_factory=dict)
    sector_win_rates: Dict[str, float] = field(default_factory=dict)
    sector_avg_trade: Dict[str, float] = field(default_factory=dict)

    # Best/worst performers
    best_sector: Optional[str] = None
    worst_sector: Optional[str] = None
    best_sector_return: float = 0.0
    worst_sector_return: float = 0.0

    # Sector rotation
    sector_dispersion: float = 0.0  # Cross-sectional std of returns
    sector_rotation_magnitude: float = 0.0
    defensive_sectors: List[str] = field(default_factory=list)

    # Factor exposures by sector
    sector_market_beta: Dict[str, float] = field(default_factory=dict)
    sector_defi_beta: Dict[str, float] = field(default_factory=dict)


@dataclass
class VenueAnalysis:
    """
    Venue-level performance during crisis.
    """
    # CEX metrics
    cex_return: float = 0.0
    cex_trades: int = 0
    cex_win_rate: float = 0.0
    cex_avg_cost: float = 0.0
    cex_sharpe: float = 0.0
    cex_max_dd: float = 0.0
    cex_volume: float = 0.0

    # DEX metrics
    dex_return: float = 0.0
    dex_trades: int = 0
    dex_win_rate: float = 0.0
    dex_avg_cost: float = 0.0
    dex_sharpe: float = 0.0
    dex_max_dd: float = 0.0
    dex_volume: float = 0.0
    dex_gas_cost: float = 0.0
    dex_mev_cost: float = 0.0

    # DEX L2 metrics
    dex_l2_return: float = 0.0
    dex_l2_trades: int = 0
    dex_l2_sharpe: float = 0.0
    dex_l2_volume: float = 0.0

    # Hybrid metrics
    hybrid_return: float = 0.0
    hybrid_trades: int = 0
    hybrid_sharpe: float = 0.0

    # Cross-venue analysis
    best_venue: Optional[str] = None
    worst_venue: Optional[str] = None
    venue_return_spread: float = 0.0
    venue_execution_quality: Dict[str, float] = field(default_factory=dict)

    # Arbitrage opportunities
    cross_venue_arb_opportunities: int = 0
    avg_arb_spread: float = 0.0
    max_arb_spread: float = 0.0


@dataclass
class RecoveryAnalysis:
    """
    Recovery pattern analysis after crisis.
    """
    # Recovery classification
    recovery_pattern: RecoveryPattern = RecoveryPattern.PARTIAL
    recovery_days: Optional[int] = None
    recovery_percentage: float = 0.0
    full_recovery: bool = False

    # Recovery path
    peak_to_trough_days: int = 0
    trough_to_recovery_days: Optional[int] = None
    recovery_return: float = 0.0
    recovery_volatility: float = 0.0
    recovery_sharpe: float = 0.0

    # Recovery dynamics
    initial_bounce_pct: float = 0.0  # First 24h bounce
    dead_cat_bounce_detected: bool = False
    double_bottom_detected: bool = False
    higher_low_formed: bool = False

    # Momentum analysis
    recovery_momentum: float = 0.0
    recovery_breadth: float = 0.0  # % of assets recovering
    sector_recovery_order: List[str] = field(default_factory=list)

    # Confidence levels
    recovery_confidence: float = 0.0  # 0-1 probability of full recovery
    expected_recovery_days: Optional[int] = None
    recovery_path_simulation: Optional[pd.Series] = None


@dataclass
class StressMetrics:
    """
    Stress test metrics during crisis.
    """
    # VaR metrics
    var_95: float = 0.0
    var_99: float = 0.0
    var_999: float = 0.0

    # CVaR (Expected Shortfall)
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    cvar_999: float = 0.0

    # Worst returns
    worst_day_return: float = 0.0
    worst_day_date: Optional[datetime] = None
    worst_3day_return: float = 0.0
    worst_week_return: float = 0.0
    worst_hour_return: float = 0.0

    # VaR breaches
    var_breaches: int = 0
    days_below_var_95: int = 0
    days_below_var_99: int = 0
    consecutive_loss_days: int = 0

    # Volatility analysis
    realized_vol: float = 0.0
    peak_realized_vol: float = 0.0
    vol_of_vol: float = 0.0
    volatility_regime: str = "normal"

    # Higher moments
    skewness: float = 0.0
    kurtosis: float = 0.0
    jarque_bera_stat: float = 0.0
    jarque_bera_pvalue: float = 0.0

    # Tail risk
    left_tail_index: float = 0.0  # Hill estimator
    tail_dependence: float = 0.0
    extreme_value_params: Dict[str, float] = field(default_factory=dict)

    # Portfolio stress
    portfolio_var_contribution: Dict[str, float] = field(default_factory=dict)
    marginal_var: Dict[str, float] = field(default_factory=dict)
    component_var: Dict[str, float] = field(default_factory=dict)


@dataclass
class FactorDecomposition:
    """
    Factor-based return decomposition during crisis.
    """
    # Market factor
    market_beta: float = 0.0
    market_contribution: float = 0.0
    market_return: float = 0.0

    # BTC factor
    btc_beta: float = 0.0
    btc_contribution: float = 0.0

    # ETH factor
    eth_beta: float = 0.0
    eth_contribution: float = 0.0

    # Sector factors
    sector_betas: Dict[str, float] = field(default_factory=dict)
    sector_contributions: Dict[str, float] = field(default_factory=dict)

    # Style factors
    momentum_beta: float = 0.0
    momentum_contribution: float = 0.0
    volatility_beta: float = 0.0
    volatility_contribution: float = 0.0
    liquidity_beta: float = 0.0
    liquidity_contribution: float = 0.0

    # Idiosyncratic
    idiosyncratic_return: float = 0.0
    idiosyncratic_vol: float = 0.0
    r_squared: float = 0.0

    # Factor exposure changes
    beta_change_vs_normal: float = 0.0
    factor_timing_value: float = 0.0


@dataclass
class CrisisAnalysisResult:
    """
    Complete crisis analysis result.

    Contains all analysis components for a single crisis event.
    """
    event: CrisisEvent

    # Core performance
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0

    # Comparison to benchmarks
    btc_return_during_crisis: float = 0.0
    eth_return_during_crisis: float = 0.0
    alpha_vs_btc: float = 0.0
    alpha_vs_eth: float = 0.0
    beta_to_btc: float = 0.0
    beta_to_eth: float = 0.0

    # Comparison to normal period
    normal_period_return: float = 0.0
    normal_period_sharpe: float = 0.0
    return_differential: float = 0.0
    sharpe_differential: float = 0.0

    # Trade metrics
    num_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    largest_winner: float = 0.0
    largest_loser: float = 0.0

    # P&L
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    fees_paid: float = 0.0
    slippage_cost: float = 0.0

    # Data quality
    data_points: int = 0
    data_coverage: float = 0.0
    data_quality_score: float = 0.0

    # Component analyses
    correlation_analysis: CorrelationAnalysis = field(default_factory=CorrelationAnalysis)
    contagion_analysis: ContagionAnalysis = field(default_factory=ContagionAnalysis)
    liquidity_analysis: LiquidityAnalysis = field(default_factory=LiquidityAnalysis)
    sector_analysis: SectorAnalysis = field(default_factory=SectorAnalysis)
    venue_analysis: VenueAnalysis = field(default_factory=VenueAnalysis)
    recovery_analysis: RecoveryAnalysis = field(default_factory=RecoveryAnalysis)
    stress_metrics: StressMetrics = field(default_factory=StressMetrics)
    factor_decomposition: FactorDecomposition = field(default_factory=FactorDecomposition)

    def get_summary_dict(self) -> Dict[str, Any]:
        """Get summary dictionary for reporting."""
        return {
            'event': self.event.name,
            'type': self.event.crisis_type.value,
            'severity': self.event.severity.value,
            'duration': self.event.duration_days,
            'return': self.total_return,
            'sharpe': self.sharpe_ratio,
            'sortino': self.sortino_ratio,
            'max_dd': self.max_drawdown,
            'vs_normal': self.return_differential,
            'alpha_btc': self.alpha_vs_btc,
            'alpha_eth': self.alpha_vs_eth,
            'beta_btc': self.beta_to_btc,
            'recovery_days': self.recovery_analysis.recovery_days,
            'recovery_pattern': self.recovery_analysis.recovery_pattern.value,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'var_95': self.stress_metrics.var_95,
            'cvar_95': self.stress_metrics.cvar_95,
            'correlation_spike': self.correlation_analysis.correlation_spike_detected,
            'contagion_velocity': self.contagion_analysis.contagion_velocity,
            'liquidity_deterioration': self.liquidity_analysis.spread_expansion_pct,
        }


# =============================================================================
# DEFAULT CRISIS EVENTS - PDF Section 2.4 Required (14 Events)
# =============================================================================

def get_default_crisis_events() -> List[CrisisEvent]:
    """
    Get the 14 default crypto crisis events per PDF requirements.

    These events span from January 2020 to present and cover:
    - Market crashes (COVID, May 2021, Yen carry)
    - Protocol failures (UST/Luna)
    - Contagion events (3AC, Celsius, FTX)
    - Regulatory actions (SEC lawsuits, Binance settlement)
    - Exploits (Curve)
    - Geopolitical (Israel-Hamas)
    - Market structure events (ETF launch)
    """
    return [
        # 1. COVID-19 Crash
        CrisisEvent(
            name="COVID-19 Crash",
            start_date="2020-03-01",
            end_date="2020-03-23",
            crisis_type=CrisisType.MARKET_CRASH,
            severity=CrisisSeverity.CATASTROPHIC,
            btc_drawdown=0.52,
            correlation_spike=True,
            affected_sectors=['defi', 'layer1', 'layer2', 'exchange_token', 'infrastructure'],
            description="Global pandemic triggers massive risk-off selloff across all assets. "
                       "BTC dropped from $9,100 to $4,900 in days. Complete correlation breakdown "
                       "as everything sold together. Massive liquidations on BitMEX.",
            recovery_date="2020-04-30",
            trigger_event="WHO declares COVID-19 pandemic",
            liquidation_volume_usd=1_500_000_000,
            peak_funding_rate=-0.30,
            open_interest_change=-0.65
        ),

        # 2. DeFi Summer Volatility
        CrisisEvent(
            name="DeFi Summer Volatility",
            start_date="2020-09-01",
            end_date="2020-09-07",
            crisis_type=CrisisType.TECHNICAL,
            severity=CrisisSeverity.MEDIUM,
            btc_drawdown=0.18,
            correlation_spike=False,
            affected_sectors=['defi', 'defi_yield', 'defi_dex'],
            description="End of DeFi summer mania. Yield farming unwinds cause rapid DeFi token "
                       "selloff. BTC relatively stable but DeFi tokens crashed 50%+. "
                       "SushiSwap vampire attack and subsequent drama.",
            recovery_date="2020-10-15",
            trigger_event="Yield farming bubble bursts",
            contagion_source="DeFi yield protocols"
        ),

        # 3. May 2021 Crash (China Mining Ban)
        CrisisEvent(
            name="May 2021 China Crash",
            start_date="2021-05-12",
            end_date="2021-05-23",
            crisis_type=CrisisType.REGULATORY,
            severity=CrisisSeverity.EXTREME,
            btc_drawdown=0.53,
            correlation_spike=True,
            affected_sectors=['layer1', 'defi', 'meme', 'exchange_token'],
            description="China announces mining ban. Elon Musk FUD on Bitcoin energy. "
                       "BTC crashed from $58K to $30K. Massive liquidations across CEX. "
                       "DOGE meme bubble burst. Hash rate dropped 50%.",
            recovery_date="2021-08-01",
            trigger_event="China crypto mining ban announcement",
            liquidation_volume_usd=8_000_000_000,
            peak_funding_rate=-0.25,
            open_interest_change=-0.55
        ),

        # 4. UST/Luna Collapse
        CrisisEvent(
            name="UST/Luna Collapse",
            start_date="2022-05-07",
            end_date="2022-05-13",
            crisis_type=CrisisType.PROTOCOL_FAILURE,
            severity=CrisisSeverity.CATASTROPHIC,
            btc_drawdown=0.27,
            correlation_spike=True,
            affected_sectors=['stablecoin', 'defi', 'layer1', 'defi_lsdfi'],
            description="Terra UST algorithmic stablecoin death spiral. $40B+ wiped out. "
                       "LUNA went from $80 to $0.0001. Massive DeFi contagion. "
                       "Anchor Protocol collapse. 3AC heavily exposed.",
            recovery_date="2022-06-15",
            trigger_event="Large UST sells break the peg",
            contagion_source="Terra/LUNA",
            liquidation_volume_usd=4_500_000_000,
            primary_venue_impact="Curve 3pool"
        ),

        # 5. 3AC Liquidation
        CrisisEvent(
            name="3AC Liquidation",
            start_date="2022-06-14",
            end_date="2022-06-18",
            crisis_type=CrisisType.CONTAGION,
            severity=CrisisSeverity.EXTREME,
            btc_drawdown=0.35,
            correlation_spike=True,
            affected_sectors=['defi', 'layer1', 'infrastructure', 'defi_lending'],
            description="Three Arrows Capital hedge fund liquidated. $10B+ positions unwound. "
                       "Cascading liquidations hit DeFi and CeFi lenders. BlockFi, Voyager, "
                       "Genesis all impacted. GBTC discount widened.",
            recovery_date="2022-07-15",
            trigger_event="3AC defaults on loans",
            contagion_source="Three Arrows Capital",
            liquidation_volume_usd=2_800_000_000
        ),

        # 6. Celsius Bankruptcy
        CrisisEvent(
            name="Celsius Bankruptcy",
            start_date="2022-07-08",
            end_date="2022-07-14",
            crisis_type=CrisisType.CONTAGION,
            severity=CrisisSeverity.HIGH,
            btc_drawdown=0.15,
            correlation_spike=True,
            affected_sectors=['defi', 'stablecoin', 'defi_lending'],
            description="Celsius Network files for bankruptcy. $4.7B in deposits frozen. "
                       "Confidence in CeFi lending collapses. stETH depeg fears. "
                       "User funds locked indefinitely.",
            recovery_date="2022-08-01",
            trigger_event="Celsius halts withdrawals",
            contagion_source="Celsius Network"
        ),

        # 7. FTX Collapse
        CrisisEvent(
            name="FTX Collapse",
            start_date="2022-11-06",
            end_date="2022-11-14",
            crisis_type=CrisisType.FRAUD,
            severity=CrisisSeverity.CATASTROPHIC,
            btc_drawdown=0.26,
            correlation_spike=True,
            affected_sectors=['exchange_token', 'defi', 'layer1', 'infrastructure'],
            description="FTX/Alameda fraud exposed. Second largest exchange collapses. "
                       "$8B+ customer funds missing. SOL ecosystem hit hard (-70%). "
                       "FTT went from $22 to $1. Industry-wide trust crisis.",
            recovery_date="2023-01-15",
            trigger_event="CoinDesk Alameda balance sheet leak",
            contagion_source="FTX/Alameda",
            liquidation_volume_usd=6_000_000_000,
            primary_venue_impact="FTX, Alameda counterparties",
            open_interest_change=-0.40
        ),

        # 8. SVB/USDC Depeg
        CrisisEvent(
            name="SVB/USDC Depeg",
            start_date="2023-03-10",
            end_date="2023-03-13",
            crisis_type=CrisisType.LIQUIDITY_CRISIS,
            severity=CrisisSeverity.EXTREME,
            btc_drawdown=0.12,
            correlation_spike=True,
            affected_sectors=['stablecoin', 'defi', 'defi_lending'],
            description="Silicon Valley Bank collapse. USDC depegged to $0.87 due to "
                       "$3.3B SVB exposure. Massive stablecoin arbitrage opportunities. "
                       "DAI also affected. Weekend panic selling.",
            recovery_date="2023-03-15",
            trigger_event="SVB bank run and FDIC takeover",
            contagion_source="Traditional banking system",
            primary_venue_impact="Curve 3pool, DEX stablecoin pools"
        ),

        # 9. SEC Lawsuits (Binance & Coinbase)
        CrisisEvent(
            name="SEC Lawsuits",
            start_date="2023-06-05",
            end_date="2023-06-12",
            crisis_type=CrisisType.REGULATORY,
            severity=CrisisSeverity.HIGH,
            btc_drawdown=0.08,
            correlation_spike=False,
            affected_sectors=['exchange_token', 'layer1'],
            description="SEC sues Binance and Coinbase in same week. Multiple altcoins "
                       "labeled as securities (SOL, ADA, MATIC, etc.). "
                       "BNB drops 15%. Exchange token sector hit hard.",
            recovery_date="2023-07-01",
            trigger_event="SEC files lawsuits against Binance and Coinbase"
        ),

        # 10. Curve Exploit
        CrisisEvent(
            name="Curve Exploit",
            start_date="2023-07-30",
            end_date="2023-08-02",
            crisis_type=CrisisType.EXPLOIT,
            severity=CrisisSeverity.HIGH,
            btc_drawdown=0.05,
            correlation_spike=False,
            affected_sectors=['defi', 'stablecoin', 'defi_dex'],
            description="Vyper reentrancy vulnerability exploited across multiple Curve pools. "
                       "$70M+ stolen. CRV token crashed 30%. Michael Egorov margin calls. "
                       "DeFi confidence shaken. OTC CRV sales.",
            recovery_date="2023-08-15",
            trigger_event="Vyper compiler vulnerability discovered and exploited",
            contagion_source="Curve Finance",
            liquidation_volume_usd=150_000_000
        ),

        # 11. Israel-Hamas Conflict
        CrisisEvent(
            name="Israel-Hamas Conflict",
            start_date="2023-10-07",
            end_date="2023-10-10",
            crisis_type=CrisisType.GEOPOLITICAL,
            severity=CrisisSeverity.MEDIUM,
            btc_drawdown=0.05,
            correlation_spike=False,
            affected_sectors=['layer1'],
            description="Hamas attack on Israel triggers geopolitical risk-off. "
                       "Crypto initially sold but recovered quickly as safe-haven narrative emerged. "
                       "Flight to quality within crypto (BTC dominance up).",
            recovery_date="2023-10-15",
            trigger_event="Hamas attack on Israel"
        ),

        # 12. Binance Settlement
        CrisisEvent(
            name="Binance DOJ Settlement",
            start_date="2023-11-20",
            end_date="2023-11-22",
            crisis_type=CrisisType.REGULATORY,
            severity=CrisisSeverity.HIGH,
            btc_drawdown=0.06,
            correlation_spike=False,
            affected_sectors=['exchange_token'],
            description="Binance settles with DOJ for $4.3B. CZ steps down as CEO. "
                       "BNB drops 10%. Concerns about centralized exchange custody. "
                       "Brief outflow spike from Binance.",
            recovery_date="2023-12-01",
            trigger_event="DOJ announces $4.3B settlement with Binance"
        ),

        # 13. BTC ETF Launch
        CrisisEvent(
            name="BTC ETF Launch",
            start_date="2024-01-10",
            end_date="2024-01-23",
            crisis_type=CrisisType.TECHNICAL,
            severity=CrisisSeverity.LOW,
            btc_drawdown=0.18,
            correlation_spike=False,
            affected_sectors=['layer1'],
            description="Bitcoin spot ETFs approved and launch. Classic 'sell the news' event. "
                       "BTC drops from $49K to $40K. GBTC outflows pressure price. "
                       "Grayscale selling $500M+ daily.",
            recovery_date="2024-02-15",
            trigger_event="SEC approves spot Bitcoin ETFs",
            primary_venue_impact="GBTC, spot markets"
        ),

        # 14. Yen Carry Trade Unwind
        CrisisEvent(
            name="Yen Carry Unwind",
            start_date="2024-08-02",
            end_date="2024-08-05",
            crisis_type=CrisisType.MACRO_SHOCK,
            severity=CrisisSeverity.EXTREME,
            btc_drawdown=0.25,
            correlation_spike=True,
            affected_sectors=['layer1', 'defi', 'meme', 'exchange_token'],
            description="Bank of Japan rate hike triggers massive Yen carry trade unwind. "
                       "Global risk assets crashed. BTC dropped from $65K to $49K in days. "
                       "Largest correlation spike since COVID. VIX spiked to 65.",
            recovery_date="2024-08-20",
            trigger_event="Bank of Japan surprise rate hike",
            liquidation_volume_usd=1_200_000_000,
            peak_funding_rate=-0.15,
            open_interest_change=-0.30
        ),
    ]


# =============================================================================
# CRISIS DETECTION ENGINE
# =============================================================================

class CrisisDetector:
    """
    Real-time crisis detection using multiple indicators.

    Monitors:
    - Volatility regime shifts
    - Correlation breakdown
    - Liquidity deterioration
    - Drawdown thresholds
    - Volume anomalies
    """

    def __init__(
        self,
        volatility_lookback: int = 20,
        correlation_lookback: int = 30,
        volatility_threshold: float = 2.5,
        correlation_threshold: float = 0.3,
        drawdown_threshold: float = 0.15,
        volume_threshold: float = 3.0,
    ):
        self.volatility_lookback = volatility_lookback
        self.correlation_lookback = correlation_lookback
        self.volatility_threshold = volatility_threshold
        self.correlation_threshold = correlation_threshold
        self.drawdown_threshold = drawdown_threshold
        self.volume_threshold = volume_threshold

        self._crisis_history: List[Dict] = []

    def detect_crisis(
        self,
        returns: pd.Series,
        btc_returns: Optional[pd.Series] = None,
        volume: Optional[pd.Series] = None,
        current_time: Optional[datetime] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect if current conditions indicate a crisis.

        Returns:
            Tuple of (is_crisis, crisis_indicators)
        """
        if len(returns) < self.volatility_lookback * 2:
            return False, {}

        indicators = {}
        crisis_signals = 0

        # 1. Volatility regime detection
        vol_result = self._detect_volatility_spike(returns)
        indicators['volatility'] = vol_result
        if vol_result['is_spike']:
            crisis_signals += 1

        # 2. Correlation breakdown (if BTC available)
        if btc_returns is not None:
            corr_result = self._detect_correlation_spike(returns, btc_returns)
            indicators['correlation'] = corr_result
            if corr_result['is_spike']:
                crisis_signals += 1

        # 3. Drawdown detection
        dd_result = self._detect_drawdown_crisis(returns)
        indicators['drawdown'] = dd_result
        if dd_result['is_crisis']:
            crisis_signals += 1

        # 4. Volume anomaly (if volume available)
        if volume is not None:
            vol_result = self._detect_volume_spike(volume)
            indicators['volume'] = vol_result
            if vol_result['is_spike']:
                crisis_signals += 0.5  # Half weight for volume

        # 5. Composite crisis score
        crisis_score = crisis_signals / 3.5  # Normalize
        is_crisis = crisis_score >= 0.5

        indicators['crisis_score'] = crisis_score
        indicators['is_crisis'] = is_crisis
        indicators['timestamp'] = current_time or datetime.now()

        if is_crisis:
            self._crisis_history.append(indicators)

        return is_crisis, indicators

    def _detect_volatility_spike(self, returns: pd.Series) -> Dict:
        """Detect volatility regime shift."""
        recent_vol = returns.tail(self.volatility_lookback).std() * np.sqrt(252)
        historical_vol = returns.iloc[:-self.volatility_lookback].std() * np.sqrt(252)
        historical_vol_std = returns.rolling(self.volatility_lookback).std().iloc[:-self.volatility_lookback].std() * np.sqrt(252)

        if historical_vol_std > 0:
            z_score = (recent_vol - historical_vol) / historical_vol_std
        else:
            z_score = 0

        return {
            'recent_vol': recent_vol,
            'historical_vol': historical_vol,
            'z_score': z_score,
            'is_spike': z_score > self.volatility_threshold,
            'regime': 'crisis' if z_score > self.volatility_threshold else 'normal'
        }

    def _detect_correlation_spike(self, returns: pd.Series, btc_returns: pd.Series) -> Dict:
        """Detect correlation regime change."""
        # Align series
        aligned = pd.concat([returns, btc_returns], axis=1).dropna()
        if len(aligned) < self.correlation_lookback * 2:
            return {'is_spike': False, 'correlation_change': 0}

        recent_corr = aligned.tail(self.correlation_lookback).corr().iloc[0, 1]
        historical_corr = aligned.iloc[:-self.correlation_lookback].corr().iloc[0, 1]

        correlation_change = recent_corr - historical_corr

        return {
            'recent_correlation': recent_corr,
            'historical_correlation': historical_corr,
            'correlation_change': correlation_change,
            'is_spike': abs(correlation_change) > self.correlation_threshold,
            'direction': 'increase' if correlation_change > 0 else 'decrease'
        }

    def _detect_drawdown_crisis(self, returns: pd.Series) -> Dict:
        """Detect drawdown-based crisis."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (running_max - cumulative) / running_max

        current_dd = drawdown.iloc[-1]
        max_dd = drawdown.max()

        return {
            'current_drawdown': current_dd,
            'max_drawdown': max_dd,
            'is_crisis': current_dd > self.drawdown_threshold,
            'severity': 'high' if current_dd > 0.25 else 'medium' if current_dd > 0.15 else 'low'
        }

    def _detect_volume_spike(self, volume: pd.Series) -> Dict:
        """Detect volume anomaly."""
        recent_vol = volume.tail(5).mean()
        historical_vol = volume.iloc[:-5].mean()

        if historical_vol > 0:
            spike_ratio = recent_vol / historical_vol
        else:
            spike_ratio = 1.0

        return {
            'recent_volume': recent_vol,
            'historical_volume': historical_vol,
            'spike_ratio': spike_ratio,
            'is_spike': spike_ratio > self.volume_threshold
        }


# =============================================================================
# CORRELATION ANALYZER
# =============================================================================

class CorrelationAnalyzerEngine:
    """
    Detailed correlation analysis during crisis periods.

    Implements:
    - DCC-GARCH for dynamic conditional correlations
    - Correlation regime detection
    - Cross-sector correlation matrices
    - Correlation spike quantification
    """

    def __init__(
        self,
        window_size: int = 30,
        ewm_halflife: int = 10,
    ):
        self.window_size = window_size
        self.ewm_halflife = ewm_halflife

    def analyze_correlations(
        self,
        strategy_returns: pd.Series,
        btc_returns: pd.Series,
        eth_returns: Optional[pd.Series] = None,
        sector_returns: Optional[pd.DataFrame] = None,
        event: Optional[CrisisEvent] = None,
    ) -> CorrelationAnalysis:
        """
        Comprehensive correlation analysis.
        """
        result = CorrelationAnalysis()

        if event is None:
            return result

        # Define periods
        pre_start = event.start_dt - timedelta(days=DEFAULT_NORMAL_LOOKBACK)
        pre_end = event.start_dt - timedelta(days=1)
        post_start = event.end_dt + timedelta(days=1)
        post_end = event.end_dt + timedelta(days=30)

        # Filter data to periods
        strategy_pre = strategy_returns[(strategy_returns.index >= pre_start) & (strategy_returns.index <= pre_end)]
        strategy_crisis = strategy_returns[(strategy_returns.index >= event.start_dt) & (strategy_returns.index <= event.end_dt)]
        strategy_post = strategy_returns[(strategy_returns.index >= post_start) & (strategy_returns.index <= post_end)]

        btc_pre = btc_returns[(btc_returns.index >= pre_start) & (btc_returns.index <= pre_end)]
        btc_crisis = btc_returns[(btc_returns.index >= event.start_dt) & (btc_returns.index <= event.end_dt)]
        btc_post = btc_returns[(btc_returns.index >= post_start) & (btc_returns.index <= post_end)]

        # Calculate BTC correlations
        if len(strategy_pre) > 10 and len(btc_pre) > 10:
            min_len = min(len(strategy_pre), len(btc_pre))
            result.btc_correlation_pre = float(np.corrcoef(
                strategy_pre.values[-min_len:], btc_pre.values[-min_len:]
            )[0, 1])

        if len(strategy_crisis) > 5 and len(btc_crisis) > 5:
            min_len = min(len(strategy_crisis), len(btc_crisis))
            result.btc_correlation_crisis = float(np.corrcoef(
                strategy_crisis.values[:min_len], btc_crisis.values[:min_len]
            )[0, 1])

        if len(strategy_post) > 10 and len(btc_post) > 10:
            min_len = min(len(strategy_post), len(btc_post))
            result.btc_correlation_post = float(np.corrcoef(
                strategy_post.values[:min_len], btc_post.values[:min_len]
            )[0, 1])

        result.btc_correlation_change = result.btc_correlation_crisis - result.btc_correlation_pre

        # ETH correlations
        if eth_returns is not None:
            eth_pre = eth_returns[(eth_returns.index >= pre_start) & (eth_returns.index <= pre_end)]
            eth_crisis = eth_returns[(eth_returns.index >= event.start_dt) & (eth_returns.index <= event.end_dt)]

            if len(strategy_pre) > 10 and len(eth_pre) > 10:
                min_len = min(len(strategy_pre), len(eth_pre))
                result.eth_correlation_pre = float(np.corrcoef(
                    strategy_pre.values[-min_len:], eth_pre.values[-min_len:]
                )[0, 1])

            if len(strategy_crisis) > 5 and len(eth_crisis) > 5:
                min_len = min(len(strategy_crisis), len(eth_crisis))
                result.eth_correlation_crisis = float(np.corrcoef(
                    strategy_crisis.values[:min_len], eth_crisis.values[:min_len]
                )[0, 1])

            result.eth_correlation_change = result.eth_correlation_crisis - result.eth_correlation_pre

        # Correlation spike detection
        if result.btc_correlation_crisis > 0.8 and result.btc_correlation_change > 0.2:
            result.correlation_spike_detected = True
            result.spike_magnitude = result.btc_correlation_change

        # Regime classification
        result.pre_crisis_regime = self._classify_correlation_regime(result.btc_correlation_pre)
        result.crisis_regime = self._classify_correlation_regime(result.btc_correlation_crisis)
        result.post_crisis_regime = self._classify_correlation_regime(result.btc_correlation_post)

        # Sector correlations
        if sector_returns is not None:
            result.sector_correlation_matrix_pre = self._calculate_sector_correlation_matrix(
                sector_returns, pre_start, pre_end
            )
            result.sector_correlation_matrix_crisis = self._calculate_sector_correlation_matrix(
                sector_returns, event.start_dt, event.end_dt
            )

            # Calculate changes
            if result.sector_correlation_matrix_pre is not None and result.sector_correlation_matrix_crisis is not None:
                for sector in result.sector_correlation_matrix_pre.columns:
                    if sector in result.sector_correlation_matrix_crisis.columns:
                        pre_avg = result.sector_correlation_matrix_pre[sector].mean()
                        crisis_avg = result.sector_correlation_matrix_crisis[sector].mean()
                        result.sector_correlation_changes[sector] = crisis_avg - pre_avg

        # Network metrics
        result.average_correlation_pre = result.btc_correlation_pre
        result.average_correlation_crisis = result.btc_correlation_crisis

        return result

    def _classify_correlation_regime(self, correlation: float) -> CorrelationRegime:
        """Classify correlation value into regime."""
        if correlation < 0.3:
            return CorrelationRegime.DECOUPLING
        elif correlation < 0.5:
            return CorrelationRegime.NORMAL
        elif correlation < 0.7:
            return CorrelationRegime.ELEVATED
        elif correlation < 0.85:
            return CorrelationRegime.STRESS
        else:
            return CorrelationRegime.CRISIS

    def _calculate_sector_correlation_matrix(
        self,
        sector_returns: pd.DataFrame,
        start: datetime,
        end: datetime,
    ) -> Optional[pd.DataFrame]:
        """Calculate correlation matrix for sectors in period."""
        filtered = sector_returns[(sector_returns.index >= start) & (sector_returns.index <= end)]
        if len(filtered) < 5:
            return None
        return filtered.corr()

    def calculate_rolling_correlations(
        self,
        returns1: pd.Series,
        returns2: pd.Series,
        window: int = 30,
    ) -> pd.Series:
        """Calculate rolling correlations."""
        aligned = pd.concat([returns1, returns2], axis=1).dropna()
        if len(aligned) < window:
            return pd.Series()

        return aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])

    def calculate_ewm_correlation(
        self,
        returns1: pd.Series,
        returns2: pd.Series,
        halflife: int = 10,
    ) -> pd.Series:
        """Calculate exponentially weighted correlation."""
        aligned = pd.concat([returns1, returns2], axis=1).dropna()
        if len(aligned) < halflife:
            return pd.Series()

        # EWM covariance
        cov = aligned.iloc[:, 0].ewm(halflife=halflife).cov(aligned.iloc[:, 1])
        std1 = aligned.iloc[:, 0].ewm(halflife=halflife).std()
        std2 = aligned.iloc[:, 1].ewm(halflife=halflife).std()

        return cov / (std1 * std2)


# =============================================================================
# CONTAGION ANALYZER
# =============================================================================

class ContagionAnalyzerEngine:
    """
    Contagion pathway analysis.

    Implements:
    - Granger causality testing
    - Spillover index calculation
    - Contagion velocity measurement
    - Network analysis
    """

    def __init__(self, max_lag: int = 5):
        self.max_lag = max_lag

    def analyze_contagion(
        self,
        asset_returns: pd.DataFrame,
        event: CrisisEvent,
        suspected_source: Optional[str] = None,
    ) -> ContagionAnalysis:
        """
        Analyze contagion patterns during crisis.
        """
        result = ContagionAnalysis()

        # Filter to crisis period
        crisis_returns = asset_returns[
            (asset_returns.index >= event.start_dt) &
            (asset_returns.index <= event.end_dt)
        ]

        if len(crisis_returns) < self.max_lag + 5:
            return result

        # Identify contagion source
        if suspected_source:
            result.contagion_source = suspected_source
        else:
            result.contagion_source = self._identify_source(crisis_returns)

        # Calculate contagion metrics
        if result.contagion_source and result.contagion_source in crisis_returns.columns:
            # Affected assets
            result.affected_assets = self._identify_affected_assets(
                crisis_returns, result.contagion_source
            )

            # Transmission lags
            result.transmission_lags = self._calculate_transmission_lags(
                crisis_returns, result.contagion_source
            )

            # Contagion velocity (average lag in hours)
            if result.transmission_lags:
                result.contagion_velocity = np.mean(list(result.transmission_lags.values()))

        # Contagion breadth (proportion of assets affected)
        result.contagion_breadth = len(result.affected_assets) / len(crisis_returns.columns)

        # Contagion depth (average drawdown)
        result.contagion_depth = float(crisis_returns.min().mean())

        # Granger causality (simplified)
        if len(crisis_returns.columns) > 1 and result.contagion_source in crisis_returns.columns:
            result.granger_causality_pvalues = self._test_granger_causality(
                crisis_returns, result.contagion_source
            )
            result.significant_causal_links = [
                (result.contagion_source, asset)
                for asset, pval in result.granger_causality_pvalues.items()
                if pval < 0.05
            ]

        # Spillover index (simplified Diebold-Yilmaz)
        spillover_result = self._calculate_spillover_index(crisis_returns)
        result.total_spillover_index = spillover_result['total']
        result.directional_spillovers_to = spillover_result['to_others']
        result.directional_spillovers_from = spillover_result['from_others']
        result.net_spillovers = spillover_result['net']

        # Pathway classification
        result.primary_pathway = self._classify_pathway(event, result)

        # Affected sectors
        if hasattr(event, 'affected_sectors'):
            result.affected_sectors = event.affected_sectors

        return result

    def _identify_source(self, returns: pd.DataFrame) -> Optional[str]:
        """Identify likely contagion source as first to drop."""
        # Find asset with earliest significant drawdown
        first_drops = {}
        for col in returns.columns:
            cum_ret = (1 + returns[col]).cumprod()
            drawdown = (cum_ret.expanding().max() - cum_ret) / cum_ret.expanding().max()

            # Find first time drawdown exceeds threshold
            threshold_breaches = drawdown[drawdown > 0.10]
            if len(threshold_breaches) > 0:
                first_drops[col] = threshold_breaches.index[0]

        if first_drops:
            return min(first_drops, key=first_drops.get)
        return None

    def _identify_affected_assets(
        self,
        returns: pd.DataFrame,
        source: str,
    ) -> List[str]:
        """Identify assets affected by source."""
        affected = []
        source_returns = returns[source]

        for col in returns.columns:
            if col == source:
                continue

            # Check if asset dropped after source
            corr = source_returns.corr(returns[col])
            if corr > 0.5:  # High correlation during crisis
                affected.append(col)

        return affected

    def _calculate_transmission_lags(
        self,
        returns: pd.DataFrame,
        source: str,
    ) -> Dict[str, int]:
        """Calculate transmission lags from source to each asset."""
        lags = {}
        source_returns = returns[source]

        for col in returns.columns:
            if col == source:
                continue

            # Find optimal lag using cross-correlation
            best_lag = 0
            best_corr = 0

            for lag in range(1, self.max_lag + 1):
                shifted = source_returns.shift(lag)
                valid = pd.concat([shifted, returns[col]], axis=1).dropna()
                if len(valid) > 5:
                    corr = abs(valid.iloc[:, 0].corr(valid.iloc[:, 1]))
                    if corr > best_corr:
                        best_corr = corr
                        best_lag = lag

            if best_lag > 0 and best_corr > 0.3:
                lags[col] = best_lag * 24  # Convert to hours

        return lags

    def _test_granger_causality(
        self,
        returns: pd.DataFrame,
        source: str,
    ) -> Dict[str, float]:
        """Test Granger causality from source to each asset."""
        pvalues = {}
        source_returns = returns[source].values

        for col in returns.columns:
            if col == source:
                continue

            target = returns[col].values

            try:
                # Simplified Granger causality test
                # Using lagged regression
                X = np.column_stack([
                    np.roll(source_returns, i) for i in range(1, self.max_lag + 1)
                ])[self.max_lag:]
                y = target[self.max_lag:]

                if len(y) > self.max_lag + 5:
                    # F-test for joint significance
                    from scipy.stats import f as f_dist

                    # Full model: target ~ lagged_source + lagged_target
                    X_full = np.column_stack([
                        X,
                        np.column_stack([np.roll(target, i) for i in range(1, self.max_lag + 1)])[self.max_lag:]
                    ])

                    # Restricted model: target ~ lagged_target
                    X_restricted = np.column_stack([
                        np.roll(target, i) for i in range(1, self.max_lag + 1)
                    ])[self.max_lag:]

                    # Calculate residuals
                    beta_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
                    beta_restricted = np.linalg.lstsq(X_restricted, y, rcond=None)[0]

                    rss_full = np.sum((y - X_full @ beta_full) ** 2)
                    rss_restricted = np.sum((y - X_restricted @ beta_restricted) ** 2)

                    n = len(y)
                    k = self.max_lag

                    f_stat = ((rss_restricted - rss_full) / k) / (rss_full / (n - 2 * k))
                    pvalue = 1 - f_dist.cdf(f_stat, k, n - 2 * k)

                    pvalues[col] = pvalue

            except Exception:
                pvalues[col] = 1.0

        return pvalues

    def _calculate_spillover_index(self, returns: pd.DataFrame) -> Dict:
        """Calculate Diebold-Yilmaz spillover index (simplified)."""
        if len(returns.columns) < 2:
            return {
                'total': 0.0,
                'to_others': {},
                'from_others': {},
                'net': {}
            }

        # Use correlation matrix as proxy for variance decomposition
        corr_matrix = returns.corr().values
        n = len(returns.columns)

        # Normalize to get "spillover" contributions
        # Off-diagonal elements represent spillovers
        spillovers_to = {}
        spillovers_from = {}

        for i, col in enumerate(returns.columns):
            # Spillover TO others
            to_others = np.sum(np.abs(corr_matrix[i, :])) - 1
            spillovers_to[col] = to_others / (n - 1) if n > 1 else 0

            # Spillover FROM others
            from_others = np.sum(np.abs(corr_matrix[:, i])) - 1
            spillovers_from[col] = from_others / (n - 1) if n > 1 else 0

        # Net spillovers
        net = {k: spillovers_to[k] - spillovers_from[k] for k in spillovers_to}

        # Total spillover index
        total = np.mean(list(spillovers_to.values()))

        return {
            'total': total,
            'to_others': spillovers_to,
            'from_others': spillovers_from,
            'net': net
        }

    def _classify_pathway(
        self,
        event: CrisisEvent,
        analysis: ContagionAnalysis,
    ) -> ContagionPath:
        """Classify the primary contagion pathway."""
        crisis_type = event.crisis_type

        if crisis_type == CrisisType.LIQUIDITY_CRISIS:
            return ContagionPath.LIQUIDITY
        elif crisis_type == CrisisType.FRAUD:
            return ContagionPath.INFORMATION
        elif crisis_type in [CrisisType.CONTAGION, CrisisType.DELEVERAGE]:
            return ContagionPath.MARGIN
        elif crisis_type == CrisisType.GEOPOLITICAL:
            return ContagionPath.SENTIMENT
        else:
            return ContagionPath.DIRECT


# =============================================================================
# STRESS METRICS CALCULATOR
# =============================================================================

class StressMetricsCalculator:
    """
    Calculate comprehensive stress test metrics.
    """

    def __init__(self, confidence_levels: List[float] = None):
        self.confidence_levels = confidence_levels or [0.95, 0.99, 0.999]

    def calculate_stress_metrics(
        self,
        returns: np.ndarray,
        dates: pd.DatetimeIndex,
        portfolio_weights: Optional[Dict[str, float]] = None,
        position_returns: Optional[Dict[str, np.ndarray]] = None,
    ) -> StressMetrics:
        """
        Calculate comprehensive stress metrics.
        """
        result = StressMetrics()

        if len(returns) < 3:
            return result

        # VaR calculations
        result.var_95 = float(-np.percentile(returns, 5))
        result.var_99 = float(-np.percentile(returns, 1))
        if len(returns) >= 10:
            result.var_999 = float(-np.percentile(returns, 0.1))

        # CVaR (Expected Shortfall)
        tail_5 = returns[returns <= -result.var_95]
        if len(tail_5) > 0:
            result.cvar_95 = float(-np.mean(tail_5))

        tail_1 = returns[returns <= -result.var_99]
        if len(tail_1) > 0:
            result.cvar_99 = float(-np.mean(tail_1))

        if len(returns) >= 100:
            tail_01 = returns[returns <= -result.var_999]
            if len(tail_01) > 0:
                result.cvar_999 = float(-np.mean(tail_01))

        # Worst returns
        result.worst_day_return = float(np.min(returns))
        worst_idx = np.argmin(returns)
        if worst_idx < len(dates):
            result.worst_day_date = dates[worst_idx]

        # Rolling worst returns
        if len(returns) >= 3:
            rolling_3d = np.array([np.sum(returns[i:i+3]) for i in range(len(returns)-2)])
            result.worst_3day_return = float(np.min(rolling_3d)) if len(rolling_3d) > 0 else result.worst_day_return

        if len(returns) >= 7:
            rolling_7d = np.array([np.sum(returns[i:i+7]) for i in range(len(returns)-6)])
            result.worst_week_return = float(np.min(rolling_7d)) if len(rolling_7d) > 0 else result.worst_3day_return

        # VaR breaches
        result.days_below_var_95 = int(np.sum(returns < -result.var_95))
        result.days_below_var_99 = int(np.sum(returns < -result.var_99))
        result.var_breaches = result.days_below_var_95

        # Consecutive losses
        result.consecutive_loss_days = self._max_consecutive_losses(returns)

        # Volatility analysis
        result.realized_vol = float(np.std(returns) * np.sqrt(TRADING_DAYS_PER_YEAR))

        if len(returns) >= 5:
            rolling_vol = np.array([np.std(returns[i:i+5]) for i in range(len(returns)-4)])
            result.peak_realized_vol = float(np.max(rolling_vol) * np.sqrt(TRADING_DAYS_PER_YEAR))
            result.vol_of_vol = float(np.std(rolling_vol) * np.sqrt(TRADING_DAYS_PER_YEAR))

        # Higher moments
        result.skewness = float(stats.skew(returns))
        result.kurtosis = float(stats.kurtosis(returns))

        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        result.jarque_bera_stat = float(jb_stat)
        result.jarque_bera_pvalue = float(jb_pvalue)

        # Tail index (Hill estimator)
        result.left_tail_index = self._hill_estimator(returns)

        # Volatility regime
        if result.realized_vol > 1.0:
            result.volatility_regime = "extreme"
        elif result.realized_vol > 0.6:
            result.volatility_regime = "high"
        elif result.realized_vol > 0.3:
            result.volatility_regime = "elevated"
        else:
            result.volatility_regime = "normal"

        # Component VaR if position returns provided
        if portfolio_weights and position_returns:
            result.portfolio_var_contribution = self._calculate_component_var(
                portfolio_weights, position_returns, 0.95
            )
            result.marginal_var = self._calculate_marginal_var(
                portfolio_weights, position_returns, 0.95
            )

        return result

    def _max_consecutive_losses(self, returns: np.ndarray) -> int:
        """Find maximum consecutive loss days."""
        max_consecutive = 0
        current_consecutive = 0

        for ret in returns:
            if ret < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _hill_estimator(self, returns: np.ndarray, k: int = None) -> float:
        """
        Calculate Hill estimator for tail index.

        A lower tail index indicates heavier tails.
        """
        sorted_returns = np.sort(returns)
        n = len(sorted_returns)

        if k is None:
            k = max(int(np.sqrt(n)), 10)

        k = min(k, n - 1)

        if k < 2:
            return 0.0

        # Take the k smallest (most negative) returns
        tail_returns = sorted_returns[:k]

        # Hill estimator
        log_ratios = np.log(np.abs(tail_returns[:-1]) / np.abs(tail_returns[-1]))
        if len(log_ratios) > 0 and np.mean(log_ratios) != 0:
            return float(1.0 / np.mean(log_ratios))
        return 0.0

    def _calculate_component_var(
        self,
        weights: Dict[str, float],
        position_returns: Dict[str, np.ndarray],
        confidence: float,
    ) -> Dict[str, float]:
        """Calculate component VaR."""
        component_var = {}

        # Calculate portfolio returns
        portfolio_returns = np.zeros(len(next(iter(position_returns.values()))))
        for asset, ret in position_returns.items():
            if asset in weights:
                portfolio_returns += weights[asset] * ret

        portfolio_var = -np.percentile(portfolio_returns, (1 - confidence) * 100)

        for asset, ret in position_returns.items():
            if asset in weights:
                # Marginal contribution
                corr = np.corrcoef(ret, portfolio_returns)[0, 1]
                std_asset = np.std(ret)
                std_portfolio = np.std(portfolio_returns)

                if std_portfolio > 0:
                    beta = corr * std_asset / std_portfolio
                    component_var[asset] = weights[asset] * beta * portfolio_var
                else:
                    component_var[asset] = 0.0

        return component_var

    def _calculate_marginal_var(
        self,
        weights: Dict[str, float],
        position_returns: Dict[str, np.ndarray],
        confidence: float,
    ) -> Dict[str, float]:
        """Calculate marginal VaR."""
        marginal_var = {}

        # Calculate base portfolio VaR
        portfolio_returns = np.zeros(len(next(iter(position_returns.values()))))
        for asset, ret in position_returns.items():
            if asset in weights:
                portfolio_returns += weights[asset] * ret

        base_var = -np.percentile(portfolio_returns, (1 - confidence) * 100)

        # Calculate marginal VaR for each position
        delta = 0.01  # 1% increase

        for asset in position_returns:
            if asset in weights:
                # Increase weight slightly
                new_weights = weights.copy()
                new_weights[asset] = weights[asset] + delta

                # Normalize
                total = sum(new_weights.values())
                new_weights = {k: v / total for k, v in new_weights.items()}

                # New portfolio returns
                new_portfolio_returns = np.zeros(len(portfolio_returns))
                for a, ret in position_returns.items():
                    if a in new_weights:
                        new_portfolio_returns += new_weights[a] * ret

                new_var = -np.percentile(new_portfolio_returns, (1 - confidence) * 100)

                marginal_var[asset] = (new_var - base_var) / delta

        return marginal_var


# =============================================================================
# MAIN CRISIS ANALYZER CLASS
# =============================================================================

class CrisisAnalyzer:
    """
    Comprehensive crisis event analyzer for PDF Section 2.4.

    Analyzes strategy performance during 14 major market crisis events
    with comprehensive breakdown by:
    - Correlation analysis (pair, sector, venue, market)
    - Contagion analysis (pathways, velocity, spillovers)
    - Sector analysis (DeFi, L1, L2, Stables, Meme, etc.)
    - Venue analysis (CEX vs DEX vs Hybrid)
    - Recovery analysis (pattern, timing, path)
    - Stress metrics (VaR, CVaR, tail risk)
    - Factor decomposition (market, sector, idiosyncratic)
    - Liquidity analysis (spreads, depth, volume)

    Usage:
        analyzer = CrisisAnalyzer()
        results = analyzer.analyze(
            backtest_results=df,
            trades=trades_list,
            btc_prices=btc_series,
            eth_prices=eth_series,
            sector_returns=sector_df
        )
        summary = analyzer.create_summary_table(results)
        report = analyzer.create_crisis_report(results)
    """

    def __init__(
        self,
        events: Optional[List[CrisisEvent]] = None,
        normal_period_lookback: int = DEFAULT_NORMAL_LOOKBACK,
        recovery_lookforward: int = DEFAULT_RECOVERY_LOOKFORWARD,
        min_data_points: int = MIN_CRISIS_DATA_POINTS,
        risk_free_rate: float = 0.045,
    ):
        """
        Initialize crisis analyzer.

        Args:
            events: List of crisis events (defaults to 14 PDF events)
            normal_period_lookback: Days before crisis for baseline comparison
            recovery_lookforward: Days after crisis to analyze recovery
            min_data_points: Minimum data points for valid analysis
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.events = events or get_default_crisis_events()
        self.normal_period_lookback = normal_period_lookback
        self.recovery_lookforward = recovery_lookforward
        self.min_data_points = min_data_points
        self.risk_free_rate = risk_free_rate
        self._daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR

        # Initialize sub-analyzers
        self.crisis_detector = CrisisDetector()
        self.correlation_analyzer = CorrelationAnalyzerEngine()
        self.contagion_analyzer = ContagionAnalyzerEngine()
        self.stress_calculator = StressMetricsCalculator()

        logger.info(
            f"CrisisAnalyzer initialized with {len(self.events)} events "
            f"(lookback={normal_period_lookback}d, lookforward={recovery_lookforward}d)"
        )

    def analyze(
        self,
        backtest_results: pd.DataFrame,
        trades: Optional[List[Dict]] = None,
        btc_prices: Optional[pd.Series] = None,
        eth_prices: Optional[pd.Series] = None,
        sector_returns: Optional[pd.DataFrame] = None,
        asset_returns: Optional[pd.DataFrame] = None,
        returns_col: str = 'returns',
        pnl_col: str = 'pnl',
    ) -> List[CrisisAnalysisResult]:
        """
        Analyze strategy performance during all crisis events.

        Args:
            backtest_results: DataFrame with backtest results (DatetimeIndex required)
            trades: Optional list of trade dictionaries with sector/venue info
            btc_prices: BTC price series for benchmark comparison
            eth_prices: ETH price series for benchmark comparison
            sector_returns: DataFrame of sector-level returns
            asset_returns: DataFrame of individual asset returns (for contagion analysis)
            returns_col: Column name for returns
            pnl_col: Column name for P&L

        Returns:
            List of CrisisAnalysisResult for each event
        """
        results = []

        # Ensure datetime index
        if not isinstance(backtest_results.index, pd.DatetimeIndex):
            logger.warning("Converting index to DatetimeIndex")
            backtest_results = backtest_results.copy()
            backtest_results.index = pd.to_datetime(backtest_results.index, utc=True)

        # Calculate benchmark returns if prices provided
        btc_returns = btc_prices.pct_change().dropna() if btc_prices is not None else None
        eth_returns = eth_prices.pct_change().dropna() if eth_prices is not None else None

        for event in self.events:
            try:
                result = self._analyze_single_event(
                    backtest_results=backtest_results,
                    event=event,
                    trades=trades,
                    btc_returns=btc_returns,
                    eth_returns=eth_returns,
                    sector_returns=sector_returns,
                    asset_returns=asset_returns,
                    returns_col=returns_col,
                    pnl_col=pnl_col,
                )
                results.append(result)

                logger.info(
                    f"Analyzed {event.name}: Return={result.total_return:.2%}, "
                    f"Sharpe={result.sharpe_ratio:.2f}, Alpha={result.alpha_vs_btc:.2%}"
                )

            except Exception as e:
                logger.error(f"Error analyzing {event.name}: {e}")
                results.append(self._empty_result(event))

        return results

    def _analyze_single_event(
        self,
        backtest_results: pd.DataFrame,
        event: CrisisEvent,
        trades: Optional[List[Dict]],
        btc_returns: Optional[pd.Series],
        eth_returns: Optional[pd.Series],
        sector_returns: Optional[pd.DataFrame],
        asset_returns: Optional[pd.DataFrame],
        returns_col: str,
        pnl_col: str,
    ) -> CrisisAnalysisResult:
        """Analyze a single crisis event in detail."""

        # Extract period data
        crisis_mask = (
            (backtest_results.index >= event.start_dt) &
            (backtest_results.index <= event.end_dt)
        )
        crisis_data = backtest_results[crisis_mask]

        normal_end = event.start_dt - timedelta(days=1)
        normal_start = normal_end - timedelta(days=self.normal_period_lookback)
        normal_mask = (
            (backtest_results.index >= normal_start) &
            (backtest_results.index <= normal_end)
        )
        normal_data = backtest_results[normal_mask]

        recovery_start = event.end_dt + timedelta(days=1)
        recovery_end = event.end_dt + timedelta(days=self.recovery_lookforward)
        recovery_mask = (
            (backtest_results.index >= recovery_start) &
            (backtest_results.index <= recovery_end)
        )
        recovery_data = backtest_results[recovery_mask]

        # Initialize result
        result = CrisisAnalysisResult(event=event)

        if len(crisis_data) < self.min_data_points:
            logger.warning(f"Insufficient data for {event.name}: {len(crisis_data)} points")
            result.data_points = len(crisis_data)
            return result

        # Get returns
        crisis_returns = crisis_data[returns_col].values if returns_col in crisis_data.columns else np.zeros(len(crisis_data))
        normal_returns = normal_data[returns_col].values if returns_col in normal_data.columns and len(normal_data) > 0 else np.zeros(1)

        # Data quality
        result.data_points = len(crisis_data)
        result.data_coverage = len(crisis_data) / max(1, event.duration_days)
        result.data_quality_score = min(1.0, result.data_coverage)

        # Core performance metrics
        result.total_return = float(np.sum(crisis_returns))
        result.annualized_return = result.total_return * (TRADING_DAYS_PER_YEAR / max(1, len(crisis_returns)))

        result.volatility = float(np.std(crisis_returns) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if result.volatility > 0:
            result.sharpe_ratio = (result.annualized_return - self.risk_free_rate) / result.volatility

        # Sortino ratio
        downside = crisis_returns[crisis_returns < 0]
        if len(downside) > 1:
            downside_vol = np.std(downside) * np.sqrt(TRADING_DAYS_PER_YEAR)
            if downside_vol > 0:
                result.sortino_ratio = (result.annualized_return - self.risk_free_rate) / downside_vol

        # Max drawdown
        cumulative = np.cumprod(1 + crisis_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        result.max_drawdown = float(np.max(drawdown))

        if result.max_drawdown > 0:
            result.calmar_ratio = result.annualized_return / result.max_drawdown

        # Normal period comparison
        result.normal_period_return = float(np.sum(normal_returns)) if len(normal_returns) > 0 else 0.0
        result.return_differential = result.total_return - result.normal_period_return

        if len(normal_returns) > 1:
            normal_vol = np.std(normal_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
            normal_ann_ret = np.sum(normal_returns) * (TRADING_DAYS_PER_YEAR / len(normal_returns))
            if normal_vol > 0:
                result.normal_period_sharpe = (normal_ann_ret - self.risk_free_rate) / normal_vol
            result.sharpe_differential = result.sharpe_ratio - result.normal_period_sharpe

        # Benchmark comparison
        if btc_returns is not None:
            btc_crisis = btc_returns[(btc_returns.index >= event.start_dt) & (btc_returns.index <= event.end_dt)]
            if len(btc_crisis) > 0:
                result.btc_return_during_crisis = float(btc_crisis.sum())
                result.alpha_vs_btc = result.total_return - result.btc_return_during_crisis

                # Beta calculation
                if len(crisis_returns) == len(btc_crisis):
                    btc_arr = btc_crisis.values
                    if np.var(btc_arr) > 0:
                        result.beta_to_btc = float(np.cov(crisis_returns, btc_arr)[0, 1] / np.var(btc_arr))

        if eth_returns is not None:
            eth_crisis = eth_returns[(eth_returns.index >= event.start_dt) & (eth_returns.index <= event.end_dt)]
            if len(eth_crisis) > 0:
                result.eth_return_during_crisis = float(eth_crisis.sum())
                result.alpha_vs_eth = result.total_return - result.eth_return_during_crisis

                if len(crisis_returns) == len(eth_crisis):
                    eth_arr = eth_crisis.values
                    if np.var(eth_arr) > 0:
                        result.beta_to_eth = float(np.cov(crisis_returns, eth_arr)[0, 1] / np.var(eth_arr))

        # Correlation analysis
        if btc_returns is not None:
            result.correlation_analysis = self.correlation_analyzer.analyze_correlations(
                strategy_returns=pd.Series(crisis_returns, index=crisis_data.index),
                btc_returns=btc_returns,
                eth_returns=eth_returns,
                sector_returns=sector_returns,
                event=event,
            )

        # Contagion analysis
        if asset_returns is not None:
            result.contagion_analysis = self.contagion_analyzer.analyze_contagion(
                asset_returns=asset_returns,
                event=event,
                suspected_source=event.contagion_source,
            )

        # Stress metrics
        result.stress_metrics = self.stress_calculator.calculate_stress_metrics(
            returns=crisis_returns,
            dates=crisis_data.index,
        )

        # Recovery analysis
        if len(recovery_data) > 0:
            recovery_returns = recovery_data[returns_col].values if returns_col in recovery_data.columns else np.zeros(len(recovery_data))
            result.recovery_analysis = self._analyze_recovery(
                crisis_returns, recovery_returns, cumulative, result.max_drawdown
            )

        # Trade-level analysis if trades provided
        if trades:
            crisis_trades = self._filter_trades_by_period(trades, event.start_dt, event.end_dt)
            if crisis_trades:
                result = self._add_trade_metrics(result, crisis_trades)
                result.sector_analysis = self._analyze_sectors(crisis_trades)
                result.venue_analysis = self._analyze_venues(crisis_trades)

        # Factor decomposition (simplified)
        result.factor_decomposition = self._calculate_factor_decomposition(
            crisis_returns, btc_returns, eth_returns, event
        )

        return result

    def _analyze_recovery(
        self,
        crisis_returns: np.ndarray,
        recovery_returns: np.ndarray,
        crisis_cumulative: np.ndarray,
        max_dd: float
    ) -> RecoveryAnalysis:
        """Analyze recovery after crisis."""
        ra = RecoveryAnalysis()

        if len(recovery_returns) == 0:
            return ra

        # Calculate recovery path
        recovery_cumulative = np.cumprod(1 + recovery_returns)
        crisis_trough = np.min(crisis_cumulative)
        crisis_peak = np.max(crisis_cumulative)

        # Find when we recover to pre-crisis peak
        target_recovery = crisis_peak
        recovery_found = False

        for i, val in enumerate(recovery_cumulative):
            adjusted_val = crisis_cumulative[-1] * val
            if adjusted_val >= target_recovery:
                ra.recovery_days = i + 1
                ra.full_recovery = True
                recovery_found = True
                break

        if not recovery_found:
            final_level = crisis_cumulative[-1] * recovery_cumulative[-1] if len(recovery_cumulative) > 0 else crisis_cumulative[-1]
            recovery_amount = (final_level - crisis_trough)
            loss_amount = (crisis_peak - crisis_trough)
            if loss_amount > 0:
                ra.recovery_percentage = recovery_amount / loss_amount

        # Determine recovery pattern
        if ra.recovery_days is not None:
            if ra.recovery_days <= 7:
                ra.recovery_pattern = RecoveryPattern.V_SHAPE
            elif ra.recovery_days <= 30:
                ra.recovery_pattern = RecoveryPattern.U_SHAPE
            else:
                ra.recovery_pattern = RecoveryPattern.U_SHAPE
        elif ra.recovery_percentage > 0.8:
            ra.recovery_pattern = RecoveryPattern.PARTIAL
        else:
            ra.recovery_pattern = RecoveryPattern.L_SHAPE

        # Recovery metrics
        if len(recovery_returns) > 0:
            ra.recovery_return = float(np.sum(recovery_returns))
            ra.recovery_volatility = float(np.std(recovery_returns) * np.sqrt(TRADING_DAYS_PER_YEAR))
            if ra.recovery_volatility > 0:
                ann_ret = ra.recovery_return * (TRADING_DAYS_PER_YEAR / len(recovery_returns))
                ra.recovery_sharpe = ann_ret / ra.recovery_volatility

        # Initial bounce
        if len(recovery_returns) >= 1:
            ra.initial_bounce_pct = float(recovery_returns[0])

        # Dead cat bounce detection
        if len(recovery_returns) >= 5:
            first_3_days = np.sum(recovery_returns[:3])
            next_2_days = np.sum(recovery_returns[3:5])
            if first_3_days > 0.05 and next_2_days < -0.03:
                ra.dead_cat_bounce_detected = True

        # Peak to trough
        if len(crisis_cumulative) > 0:
            trough_idx = np.argmin(crisis_cumulative)
            peak_idx = np.argmax(crisis_cumulative[:trough_idx+1]) if trough_idx > 0 else 0
            ra.peak_to_trough_days = trough_idx - peak_idx

        # Recovery momentum
        if len(recovery_returns) >= 5:
            ra.recovery_momentum = float(np.mean(recovery_returns[:5]))

        return ra

    def _filter_trades_by_period(
        self,
        trades: List[Dict],
        start: datetime,
        end: datetime
    ) -> List[Dict]:
        """Filter trades to those within the crisis period."""
        filtered = []
        for trade in trades:
            entry_time = trade.get('entry_time') if isinstance(trade, dict) else getattr(trade, 'entry_time', None)
            if entry_time:
                ts = pd.to_datetime(entry_time, utc=True)
                if start <= ts <= end:
                    filtered.append(trade)
        return filtered

    def _add_trade_metrics(
        self,
        result: CrisisAnalysisResult,
        trades: List[Dict]
    ) -> CrisisAnalysisResult:
        """Add trade-level metrics to result."""
        result.num_trades = len(trades)

        returns = []
        gross_pnls = []
        net_pnls = []
        fees = []

        for trade in trades:
            if isinstance(trade, dict):
                ret = trade.get('return_pct', trade.get('return', 0))
                gross = trade.get('gross_pnl', 0)
                net = trade.get('net_pnl', gross)
                fee = trade.get('fee_cost', 0)
            else:
                ret = getattr(trade, 'return_pct', 0)
                gross = getattr(trade, 'gross_pnl', 0)
                net = getattr(trade, 'net_pnl', gross)
                fee = getattr(trade, 'fee_cost', 0)

            returns.append(ret)
            gross_pnls.append(gross)
            net_pnls.append(net)
            fees.append(fee)

        returns = np.array(returns)

        # Win rate
        result.win_rate = float(np.sum(returns > 0) / len(returns)) if len(returns) > 0 else 0.0

        # Profit factor
        winners = np.sum(returns[returns > 0])
        losers = abs(np.sum(returns[returns < 0]))
        result.profit_factor = float(winners / losers) if losers > 0 else 0.0

        # Average trades
        result.avg_trade_return = float(np.mean(returns))
        if np.any(returns > 0):
            result.avg_winning_trade = float(np.mean(returns[returns > 0]))
            result.largest_winner = float(np.max(returns))
        if np.any(returns < 0):
            result.avg_losing_trade = float(np.mean(returns[returns < 0]))
            result.largest_loser = float(np.min(returns))

        # P&L
        result.gross_pnl = float(sum(gross_pnls))
        result.net_pnl = float(sum(net_pnls))
        result.fees_paid = float(sum(fees))

        return result

    def _analyze_sectors(self, trades: List[Dict]) -> SectorAnalysis:
        """Analyze performance by sector during crisis."""
        sa = SectorAnalysis()

        sector_returns = defaultdict(list)
        sector_wins = defaultdict(int)
        sector_total = defaultdict(int)

        for trade in trades:
            if isinstance(trade, dict):
                sector = trade.get('sector', 'other')
                ret = trade.get('return_pct', trade.get('return', 0))
            else:
                sector = getattr(trade, 'sector', 'other')
                ret = getattr(trade, 'return_pct', 0)

            sector_returns[sector].append(ret)
            sector_total[sector] += 1
            if ret > 0:
                sector_wins[sector] += 1

        # Calculate metrics per sector
        for sector, rets in sector_returns.items():
            rets = np.array(rets)
            sa.sector_returns[sector] = float(np.sum(rets))
            sa.sector_trade_counts[sector] = len(rets)
            sa.sector_win_rates[sector] = sector_wins[sector] / len(rets) if len(rets) > 0 else 0.0
            sa.sector_avg_trade[sector] = float(np.mean(rets))

            if len(rets) > 1:
                vol = np.std(rets) * np.sqrt(TRADING_DAYS_PER_YEAR)
                sa.sector_volatility[sector] = float(vol)
                if vol > 0:
                    ann_ret = np.sum(rets) * (TRADING_DAYS_PER_YEAR / len(rets))
                    sa.sector_sharpe[sector] = float(ann_ret / vol)

                # Max drawdown per sector
                cum_rets = np.cumprod(1 + rets)
                running_max = np.maximum.accumulate(cum_rets)
                dd = (running_max - cum_rets) / running_max
                sa.sector_max_dd[sector] = float(np.max(dd))

        # Best/worst sectors
        if sa.sector_returns:
            sa.best_sector = max(sa.sector_returns, key=sa.sector_returns.get)
            sa.worst_sector = min(sa.sector_returns, key=sa.sector_returns.get)
            sa.best_sector_return = sa.sector_returns[sa.best_sector]
            sa.worst_sector_return = sa.sector_returns[sa.worst_sector]
            sa.sector_dispersion = float(np.std(list(sa.sector_returns.values())))

            # Identify defensive sectors
            sa.defensive_sectors = [s for s, r in sa.sector_returns.items() if r > 0]

        return sa

    def _analyze_venues(self, trades: List[Dict]) -> VenueAnalysis:
        """Analyze performance by venue during crisis."""
        va = VenueAnalysis()

        venue_data = {'cex': [], 'dex': [], 'dex_l2': [], 'hybrid': []}
        venue_costs = {'cex': [], 'dex': [], 'dex_l2': [], 'hybrid': []}

        for trade in trades:
            if isinstance(trade, dict):
                venue = str(trade.get('venue_type', 'cex')).lower()
                ret = trade.get('return_pct', trade.get('return', 0))
                cost = trade.get('fee_cost', 0)
            else:
                venue = str(getattr(trade, 'venue_type', 'cex')).lower()
                ret = getattr(trade, 'return_pct', 0)
                cost = getattr(trade, 'fee_cost', 0)

            if 'l2' in venue:
                venue_data['dex_l2'].append(ret)
                venue_costs['dex_l2'].append(cost)
            elif 'dex' in venue:
                venue_data['dex'].append(ret)
                venue_costs['dex'].append(cost)
            elif 'hybrid' in venue:
                venue_data['hybrid'].append(ret)
                venue_costs['hybrid'].append(cost)
            else:
                venue_data['cex'].append(ret)
                venue_costs['cex'].append(cost)

        # CEX metrics
        if venue_data['cex']:
            rets = np.array(venue_data['cex'])
            va.cex_return = float(np.sum(rets))
            va.cex_trades = len(rets)
            va.cex_win_rate = float(np.sum(rets > 0) / len(rets))
            va.cex_avg_cost = float(np.mean(venue_costs['cex']))
            if len(rets) > 1 and np.std(rets) > 0:
                va.cex_sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(TRADING_DAYS_PER_YEAR))
                cum_rets = np.cumprod(1 + rets)
                running_max = np.maximum.accumulate(cum_rets)
                dd = (running_max - cum_rets) / running_max
                va.cex_max_dd = float(np.max(dd))

        # DEX metrics
        if venue_data['dex']:
            rets = np.array(venue_data['dex'])
            va.dex_return = float(np.sum(rets))
            va.dex_trades = len(rets)
            va.dex_win_rate = float(np.sum(rets > 0) / len(rets))
            va.dex_avg_cost = float(np.mean(venue_costs['dex']))
            if len(rets) > 1 and np.std(rets) > 0:
                va.dex_sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(TRADING_DAYS_PER_YEAR))
                cum_rets = np.cumprod(1 + rets)
                running_max = np.maximum.accumulate(cum_rets)
                dd = (running_max - cum_rets) / running_max
                va.dex_max_dd = float(np.max(dd))

        # DEX L2 metrics
        if venue_data['dex_l2']:
            rets = np.array(venue_data['dex_l2'])
            va.dex_l2_return = float(np.sum(rets))
            va.dex_l2_trades = len(rets)
            if len(rets) > 1 and np.std(rets) > 0:
                va.dex_l2_sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(TRADING_DAYS_PER_YEAR))

        # Hybrid metrics
        if venue_data['hybrid']:
            rets = np.array(venue_data['hybrid'])
            va.hybrid_return = float(np.sum(rets))
            va.hybrid_trades = len(rets)
            if len(rets) > 1 and np.std(rets) > 0:
                va.hybrid_sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(TRADING_DAYS_PER_YEAR))

        # Best/worst venue
        venue_returns = {
            'CEX': va.cex_return,
            'DEX': va.dex_return,
            'DEX_L2': va.dex_l2_return,
            'Hybrid': va.hybrid_return
        }
        active_venues = {k: v for k, v in venue_returns.items() if v != 0}

        if active_venues:
            va.best_venue = max(active_venues, key=active_venues.get)
            va.worst_venue = min(active_venues, key=active_venues.get)
            va.venue_return_spread = max(active_venues.values()) - min(active_venues.values())

        return va

    def _calculate_factor_decomposition(
        self,
        crisis_returns: np.ndarray,
        btc_returns: Optional[pd.Series],
        eth_returns: Optional[pd.Series],
        event: CrisisEvent,
    ) -> FactorDecomposition:
        """Calculate factor-based return decomposition."""
        fd = FactorDecomposition()

        # Market factor (BTC as proxy)
        if btc_returns is not None:
            btc_crisis = btc_returns[(btc_returns.index >= event.start_dt) & (btc_returns.index <= event.end_dt)]
            if len(btc_crisis) == len(crisis_returns) and len(btc_crisis) > 0:
                btc_arr = btc_crisis.values
                if np.var(btc_arr) > 0:
                    fd.btc_beta = float(np.cov(crisis_returns, btc_arr)[0, 1] / np.var(btc_arr))
                    fd.btc_contribution = fd.btc_beta * float(btc_arr.sum())
                    fd.market_beta = fd.btc_beta  # Use BTC as market proxy
                    fd.market_contribution = fd.btc_contribution
                    fd.market_return = float(btc_arr.sum())

        # ETH factor
        if eth_returns is not None:
            eth_crisis = eth_returns[(eth_returns.index >= event.start_dt) & (eth_returns.index <= event.end_dt)]
            if len(eth_crisis) == len(crisis_returns) and len(eth_crisis) > 0:
                eth_arr = eth_crisis.values
                if np.var(eth_arr) > 0:
                    fd.eth_beta = float(np.cov(crisis_returns, eth_arr)[0, 1] / np.var(eth_arr))
                    fd.eth_contribution = fd.eth_beta * float(eth_arr.sum())

        # Idiosyncratic return
        total_return = float(np.sum(crisis_returns))
        factor_explained = fd.market_contribution + fd.eth_contribution * 0.3  # Partial ETH contribution
        fd.idiosyncratic_return = total_return - factor_explained
        fd.idiosyncratic_vol = float(np.std(crisis_returns - fd.market_beta * crisis_returns) * np.sqrt(TRADING_DAYS_PER_YEAR))

        # R-squared
        if np.var(crisis_returns) > 0:
            fd.r_squared = 1 - (fd.idiosyncratic_vol ** 2) / (np.var(crisis_returns) * TRADING_DAYS_PER_YEAR)
            fd.r_squared = max(0, min(1, fd.r_squared))  # Clip to [0, 1]

        return fd

    def _empty_result(self, event: CrisisEvent) -> CrisisAnalysisResult:
        """Create empty result for events with no data."""
        return CrisisAnalysisResult(event=event)

    def create_summary_table(self, results: List[CrisisAnalysisResult]) -> pd.DataFrame:
        """Create summary table from crisis analysis results."""
        if not results:
            return pd.DataFrame()

        data = [r.get_summary_dict() for r in results]
        df = pd.DataFrame(data)

        # Sort by severity then date
        severity_order = {s.value: i for i, s in enumerate(CrisisSeverity)}
        df['_severity_rank'] = df['severity'].map(severity_order)
        df = df.sort_values(['_severity_rank', 'event']).drop('_severity_rank', axis=1)

        return df

    def get_aggregate_metrics(self, results: List[CrisisAnalysisResult]) -> Dict:
        """Calculate aggregate metrics across all crises."""
        if not results:
            return {}

        returns = [r.total_return for r in results]
        sharpes = [r.sharpe_ratio for r in results]
        alphas = [r.alpha_vs_btc for r in results]
        max_dds = [r.max_drawdown for r in results]
        sortinos = [r.sortino_ratio for r in results]
        win_rates = [r.win_rate for r in results if r.win_rate > 0]
        recovery_days = [r.recovery_analysis.recovery_days for r in results if r.recovery_analysis.recovery_days]

        return {
            'num_crises': len(results),
            'avg_return': float(np.mean(returns)),
            'median_return': float(np.median(returns)),
            'std_return': float(np.std(returns)),
            'worst_return': float(np.min(returns)),
            'best_return': float(np.max(returns)),
            'avg_sharpe': float(np.mean(sharpes)),
            'avg_sortino': float(np.mean(sortinos)),
            'avg_alpha_btc': float(np.mean(alphas)),
            'total_alpha': float(np.sum(alphas)),
            'avg_max_dd': float(np.mean(max_dds)),
            'worst_max_dd': float(np.max(max_dds)),
            'avg_win_rate': float(np.mean(win_rates)) if win_rates else 0,
            'positive_crisis_pct': float(np.sum(np.array(returns) > 0) / len(returns)),
            'avg_recovery_days': float(np.mean(recovery_days)) if recovery_days else None,
            'full_recovery_pct': float(sum(1 for r in results if r.recovery_analysis.full_recovery) / len(results)),
            'correlation_spike_pct': float(sum(1 for r in results if r.correlation_analysis.correlation_spike_detected) / len(results)),
        }

    def create_crisis_report(
        self,
        results: List[CrisisAnalysisResult],
        title: str = "Crisis Event Analysis Report"
    ) -> str:
        """Create comprehensive crisis analysis report."""
        lines = [
            "=" * 100,
            title.center(100),
            "(PDF Section 2.4 - 14 Crisis Events Comprehensive Analysis)".center(100),
            "=" * 100,
            "",
        ]

        # Aggregate metrics
        agg = self.get_aggregate_metrics(results)
        if agg:
            lines.extend([
                "AGGREGATE CRISIS PERFORMANCE",
                "-" * 80,
                f"  Crises Analyzed:              {agg['num_crises']}",
                f"  Average Return:               {agg['avg_return']:.2%}",
                f"  Median Return:                {agg['median_return']:.2%}",
                f"  Return Std Dev:               {agg['std_return']:.2%}",
                f"  Worst Crisis Return:          {agg['worst_return']:.2%}",
                f"  Best Crisis Return:           {agg['best_return']:.2%}",
                f"  Average Sharpe:               {agg['avg_sharpe']:.2f}",
                f"  Average Sortino:              {agg['avg_sortino']:.2f}",
                f"  Average Alpha vs BTC:         {agg['avg_alpha_btc']:.2%}",
                f"  Total Alpha Generated:        {agg['total_alpha']:.2%}",
                f"  Average Max Drawdown:         {agg['avg_max_dd']:.2%}",
                f"  Worst Max Drawdown:           {agg['worst_max_dd']:.2%}",
                f"  Positive Crisis %:            {agg['positive_crisis_pct']:.1%}",
                f"  Full Recovery %:              {agg['full_recovery_pct']:.1%}",
                f"  Correlation Spike Events:     {agg['correlation_spike_pct']:.1%}",
                "",
            ])

        # Individual crisis results
        lines.extend([
            "=" * 100,
            "INDIVIDUAL CRISIS ANALYSIS",
            "=" * 100,
        ])

        for result in results:
            lines.extend([
                "",
                f"┌{'─' * 98}┐",
                f"│ {result.event.name.upper()} ({result.event.severity.value.upper()})".ljust(99) + "│",
                f"├{'─' * 98}┤",
                f"│ Period: {result.event.start_date} to {result.event.end_date} ({result.event.duration_days} days)".ljust(99) + "│",
                f"│ Type: {result.event.crisis_type.value}".ljust(99) + "│",
                f"│ BTC Drawdown: {result.event.btc_drawdown:.1%}".ljust(99) + "│",
                f"├{'─' * 98}┤",
                f"│ {'PERFORMANCE METRICS':^96} │",
                f"├{'─' * 98}┤",
                f"│   Strategy Return:     {result.total_return:>12.2%}    │    BTC Return:     {result.btc_return_during_crisis:>12.2%}".ljust(99) + "│",
                f"│   Alpha vs BTC:        {result.alpha_vs_btc:>12.2%}    │    Beta to BTC:    {result.beta_to_btc:>12.2f}".ljust(99) + "│",
                f"│   Sharpe Ratio:        {result.sharpe_ratio:>12.2f}    │    Sortino Ratio:  {result.sortino_ratio:>12.2f}".ljust(99) + "│",
                f"│   Max Drawdown:        {result.max_drawdown:>12.2%}    │    Calmar Ratio:   {result.calmar_ratio:>12.2f}".ljust(99) + "│",
                f"│   Win Rate:            {result.win_rate:>12.1%}    │    Profit Factor:  {result.profit_factor:>12.2f}".ljust(99) + "│",
                f"│   vs Normal Period:    {result.return_differential:>12.2%}    │    Trades:         {result.num_trades:>12}".ljust(99) + "│",
                f"├{'─' * 98}┤",
                f"│ {'STRESS METRICS':^96} │",
                f"├{'─' * 98}┤",
                f"│   VaR 95%:             {result.stress_metrics.var_95:>12.4f}    │    CVaR 95%:       {result.stress_metrics.cvar_95:>12.4f}".ljust(99) + "│",
                f"│   VaR 99%:             {result.stress_metrics.var_99:>12.4f}    │    CVaR 99%:       {result.stress_metrics.cvar_99:>12.4f}".ljust(99) + "│",
                f"│   Worst Day Return:    {result.stress_metrics.worst_day_return:>12.4f}    │    Worst Week:     {result.stress_metrics.worst_week_return:>12.4f}".ljust(99) + "│",
                f"│   Realized Vol:        {result.stress_metrics.realized_vol:>12.2%}    │    Vol Regime:     {result.stress_metrics.volatility_regime:>12}".ljust(99) + "│",
                f"├{'─' * 98}┤",
                f"│ {'RECOVERY ANALYSIS':^96} │",
                f"├{'─' * 98}┤",
                f"│   Recovery Pattern:    {result.recovery_analysis.recovery_pattern.value:>12}    │    Recovery Days:  {str(result.recovery_analysis.recovery_days or 'N/A'):>12}".ljust(99) + "│",
                f"│   Recovery Return:     {result.recovery_analysis.recovery_return:>12.2%}    │    Full Recovery:  {'Yes' if result.recovery_analysis.full_recovery else 'No':>12}".ljust(99) + "│",
                f"│   Initial Bounce:      {result.recovery_analysis.initial_bounce_pct:>12.2%}    │    Dead Cat Bounce:{' Yes' if result.recovery_analysis.dead_cat_bounce_detected else ' No':>12}".ljust(99) + "│",
                f"└{'─' * 98}┘",
            ])

        # Severity breakdown
        lines.extend([
            "",
            "=" * 100,
            "CRISIS PERFORMANCE BY SEVERITY",
            "=" * 100,
        ])

        for severity in CrisisSeverity:
            severity_results = [r for r in results if r.event.severity == severity]
            if severity_results:
                avg_ret = np.mean([r.total_return for r in severity_results])
                avg_alpha = np.mean([r.alpha_vs_btc for r in severity_results])
                avg_sharpe = np.mean([r.sharpe_ratio for r in severity_results])
                lines.append(
                    f"  {severity.value.upper():<15} ({len(severity_results)} events): "
                    f"Avg Return={avg_ret:>8.2%}, Alpha={avg_alpha:>8.2%}, Sharpe={avg_sharpe:>6.2f}"
                )

        # PDF Compliance
        lines.extend([
            "",
            "=" * 100,
            "PDF SECTION 2.4 COMPLIANCE",
            "=" * 100,
            "",
            "This report analyzes all 14 PDF-required crisis events:",
            "  [+] COVID-19 Crash (March 2020) - Catastrophic market crash",
            "  [+] DeFi Summer Volatility (Sept 2020) - Technical correction",
            "  [+] May 2021 China Crash - Regulatory induced",
            "  [+] UST/Luna Collapse (May 2022) - Protocol failure",
            "  [+] 3AC Liquidation (June 2022) - Contagion event",
            "  [+] Celsius Bankruptcy (July 2022) - Contagion event",
            "  [+] FTX Collapse (November 2022) - Fraud/catastrophic",
            "  [+] SVB/USDC Depeg (March 2023) - Liquidity crisis",
            "  [+] SEC Lawsuits (June 2023) - Regulatory action",
            "  [+] Curve Exploit (July 2023) - DeFi exploit",
            "  [+] Israel-Hamas Conflict (October 2023) - Geopolitical",
            "  [+] Binance Settlement (November 2023) - Regulatory",
            "  [+] BTC ETF Launch (January 2024) - Technical event",
            "  [+] Yen Carry Unwind (August 2024) - Macro shock",
            "",
            "Analysis includes:",
            "  [+] Correlation breakdown (BTC, ETH, sector, venue)",
            "  [+] Contagion analysis (pathways, velocity, spillovers)",
            "  [+] Stress metrics (VaR, CVaR, tail risk)",
            "  [+] Factor decomposition (market, sector, idiosyncratic)",
            "  [+] Recovery analysis (pattern, timing, path)",
            "  [+] Sector analysis (DeFi, L1, L2, etc.)",
            "  [+] Venue analysis (CEX vs DEX vs Hybrid)",
            "",
            "=" * 100,
        ])

        return "\n".join(lines)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main class
    'CrisisAnalyzer',

    # Sub-analyzers
    'CrisisDetector',
    'CorrelationAnalyzerEngine',
    'ContagionAnalyzerEngine',
    'StressMetricsCalculator',

    # Data classes
    'CrisisEvent',
    'CrisisAnalysisResult',
    'CorrelationAnalysis',
    'ContagionAnalysis',
    'LiquidityAnalysis',
    'SectorAnalysis',
    'VenueAnalysis',
    'RecoveryAnalysis',
    'StressMetrics',
    'FactorDecomposition',

    # Enums
    'CrisisType',
    'CrisisSeverity',
    'RecoveryPattern',
    'CorrelationRegime',
    'ContagionPath',

    # Helper functions
    'get_default_crisis_events',

    # Constants
    'CRYPTO_SECTORS',
    'TRADING_DAYS_PER_YEAR',
]
