"""
Portfolio Construction Package
==============================

Professional-quality multi-venue portfolio optimization with comprehensive
risk management for crypto statistical arbitrage strategies.

Architecture
------------
This package implements a hierarchical portfolio construction framework:

    Level 1: Strategy Selection
        - Universe filtering based on quality scores
        - Survivorship bias adjustment
        - Capacity constraints
    
    Level 2: Weight Optimization
        - Multiple optimization methods (HRP, Risk Parity, MVO)
        - Constraint satisfaction (venue, sector, position limits)
        - Transaction cost awareness
    
    Level 3: Risk Management
        - Real-time risk monitoring
        - VaR/CVaR limits
        - Stress testing and scenario analysis
        - Correlation breakdown detection

Modules
-------
correlation_analysis : Strategy and venue-level correlation analytics
    - Pearson, Spearman, Kendall correlations
    - Time-varying correlations (rolling, EWMA, DCC-GARCH)
    - Crisis correlation analysis
    - Hierarchical clustering

optimization : Portfolio weight optimization
    - Equal-weighted (baseline)
    - Mean-Variance Optimization (MVO)
    - Risk Parity (inverse volatility, ERC)
    - Hierarchical Risk Parity (HRP) - recommended
    - Black-Litterman (with views)
    - Maximum Diversification

risk_analysis : Comprehensive risk analytics
    - Value at Risk (parametric, historical, Monte Carlo)
    - Conditional VaR / Expected Shortfall
    - Drawdown analysis
    - Factor exposure analysis

stress_testing : Scenario and stress analysis
    - Historical scenarios (FTX, Luna, etc.)
    - Hypothetical scenarios
    - Reverse stress testing
    - Correlation breakdown analysis

allocation : Venue-tiered allocation with constraints
    - CEX/DEX/Hybrid allocation
    - Sector exposure limits
    - Position concentration limits
    - Liquidity-adjusted sizing

Key Constraints
---------------
Venue Constraints:
    - Max CEX allocation: 70% (counterparty risk)
    - Max DEX allocation: 30% (smart contract risk)
    - Max Hybrid allocation: 40%
    - Max single venue: 25%

Strategy Constraints:
    - Max per strategy: 25%
    - Min per strategy: 1% (if included)
    - Max correlated strategies: 40% combined

Sector Constraints:
    - Max sector exposure: 40%
    - Funding rate arbitrage: max 50%
    - Basis trading: max 40%
    - Options strategies: max 30%

Risk Constraints:
    - Portfolio VaR (95%, 1-day): max 3%
    - Maximum drawdown limit: 20%
    - Leverage limit: 2x gross

Performance Targets
-------------------
Return Metrics:
    - Target annual return: 20-40%
    - Target Sharpe ratio: 1.5-2.5+
    - Target Sortino ratio: 2.0+

Risk Metrics:
    - Correlation to BTC: <0.3
    - Correlation to ETH: <0.3
    - Maximum drawdown: <20%
    - Recovery period: <60 days

Scalability:
    - Target AUM: $5-50M
    - Liquidity buffer: 20% of position in 24h

Stress Test Scenarios
---------------------
1. Exchange Failure (FTX-like):
    - 100% loss on affected venue
    - Correlation spike to 0.8
    - 50% liquidity reduction

2. DEX Exploit:
    - 50-100% loss on affected pool
    - Gas spike 10x
    - Bridge failure

3. Liquidity Crisis:
    - 10x slippage increase
    - Funding rate spike 10x
    - Order book depth -80%

4. Correlation Breakdown:
    - All correlations → 0.95
    - Diversification benefit → 0
    - Vol spike 3x

5. Black Swan (March 2020):
    - BTC -50% in 24h
    - Liquidation cascade
    - Exchange outages

Example Usage
-------------
Basic Portfolio Construction:

    >>> from portfolio import (
    ...     CorrelationAnalyzer,
    ...     HierarchicalRiskParity,
    ...     RiskAnalyzer,
    ...     StressTestRunner,
    ...     VenueConstraints
    ... )
    >>> 
    >>> # Analyze correlations
    >>> corr_analyzer = CorrelationAnalyzer(returns_df, metadata_df)
    >>> corr_result = corr_analyzer.calculate_correlation()
    >>> rolling_corr = corr_analyzer.rolling_correlation('Strat1', 'Strat2')
    >>> 
    >>> # Optimize portfolio with HRP
    >>> optimizer = HierarchicalRiskParity(returns_df)
    >>> raw_weights = optimizer.optimize()
    >>> 
    >>> # Apply venue constraints
    >>> constraints = VenueConstraints(max_cex=0.70, max_dex=0.30)
    >>> weights = constraints.apply(raw_weights, metadata_df)
    >>> 
    >>> # Risk analysis
    >>> risk = RiskAnalyzer(returns_df, weights)
    >>> var_95 = risk.calculate_var(confidence=0.95)
    >>> cvar_95 = risk.calculate_cvar(confidence=0.95)
    >>> 
    >>> # Stress testing
    >>> stress = StressTestRunner(weights, returns_df, metadata_df)
    >>> results = stress.run_all_scenarios()
    >>> stress.generate_report()

Extended Usage with Views:

    >>> from portfolio import BlackLitterman
    >>> 
    >>> # Define views: Strategy A will outperform by 5% annually
    >>> views = {
    ...     'absolute': [('StrategyA', 0.05)],
    ...     'relative': [('StrategyB', 'StrategyC', 0.02)]  # B > C by 2%
    ... }
    >>> 
    >>> bl = BlackLitterman(returns_df, market_weights, views)
    >>> bl_weights = bl.optimize(tau=0.05, view_confidence=0.5)

Risk Monitoring:

    >>> from portfolio import RiskMonitor
    >>> 
    >>> monitor = RiskMonitor(
    ...     weights=weights,
    ...     returns=returns_df,
    ...     var_limit=0.03,
    ...     max_drawdown=0.20
    ... )
    >>> 
    >>> # Check current risk status
    >>> status = monitor.check_limits()
    >>> if status.breach_detected:
    ...     print(f"Risk breach: {status.breached_limits}")

Dependencies
------------
Required:
    - numpy >= 1.21.0
    - pandas >= 1.3.0
    - scipy >= 1.7.0

Optional (for extended features):
    - cvxpy >= 1.2.0 (convex optimization)
    - arch >= 5.0.0 (GARCH models)
    - scikit-learn >= 1.0.0 (clustering)

References
----------
1. López de Prado, M. (2016). Building Diversified Portfolios that 
   Outperform Out-of-Sample. Journal of Portfolio Management.
   
2. Meucci, A. (2005). Risk and Asset Allocation. Springer.

3. Black, F., & Litterman, R. (1992). Global Portfolio Optimization.
   Financial Analysts Journal.

Author: Crypto StatArb Quantitative Research
Version: 2.0.0
Date: January 2025
"""

from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
import warnings

# Version info
__version__ = '2.0.0'
__author__ = 'Crypto StatArb Quantitative Research'


# =============================================================================
# Enumerations
# =============================================================================

class OptimizationMethod(Enum):
    """Portfolio optimization methods."""
    EQUAL_WEIGHT = 'equal_weight'
    INVERSE_VOLATILITY = 'inverse_volatility'
    RISK_PARITY = 'risk_parity'
    MIN_VARIANCE = 'min_variance'
    MAX_SHARPE = 'max_sharpe'
    MAX_DIVERSIFICATION = 'max_diversification'
    HRP = 'hierarchical_risk_parity'
    BLACK_LITTERMAN = 'black_litterman'
    
    @property
    def requires_expected_returns(self) -> bool:
        """Whether method requires return estimates."""
        return self in [
            OptimizationMethod.MAX_SHARPE,
            OptimizationMethod.BLACK_LITTERMAN
        ]
    
    @property
    def is_risk_based(self) -> bool:
        """Whether method is purely risk-based."""
        return self in [
            OptimizationMethod.INVERSE_VOLATILITY,
            OptimizationMethod.RISK_PARITY,
            OptimizationMethod.MIN_VARIANCE,
            OptimizationMethod.HRP
        ]


class RiskMeasure(Enum):
    """Risk measurement methods."""
    VOLATILITY = 'volatility'
    VAR_PARAMETRIC = 'var_parametric'
    VAR_HISTORICAL = 'var_historical'
    VAR_MONTE_CARLO = 'var_monte_carlo'
    CVAR = 'conditional_var'
    MAX_DRAWDOWN = 'max_drawdown'
    DOWNSIDE_DEVIATION = 'downside_deviation'
    
    @property
    def is_tail_risk(self) -> bool:
        """Whether measure captures tail risk."""
        return self in [
            RiskMeasure.VAR_PARAMETRIC,
            RiskMeasure.VAR_HISTORICAL,
            RiskMeasure.VAR_MONTE_CARLO,
            RiskMeasure.CVAR,
            RiskMeasure.MAX_DRAWDOWN
        ]


class VenueType(Enum):
    """Trading venue types."""
    CEX = 'CEX'
    DEX = 'DEX'
    HYBRID = 'HYBRID'
    
    @property
    def max_allocation(self) -> float:
        """Default maximum allocation for venue type."""
        limits = {
            VenueType.CEX: 0.70,
            VenueType.DEX: 0.30,
            VenueType.HYBRID: 0.40
        }
        return limits[self]
    
    @property
    def risk_premium(self) -> float:
        """Risk premium for venue type (for risk budgeting)."""
        premiums = {
            VenueType.CEX: 1.0,      # Baseline
            VenueType.DEX: 1.5,      # Smart contract risk
            VenueType.HYBRID: 1.2    # Mixed risk profile
        }
        return premiums[self]


class StrategySector(Enum):
    """Strategy sector classifications."""
    FUNDING_RATE = 'funding_rate'
    BASIS_TRADING = 'basis_trading'
    PAIRS_TRADING = 'pairs_trading'
    OPTIONS_VOL = 'options_volatility'
    MARKET_MAKING = 'market_making'
    LIQUIDATION = 'liquidation'
    MEV = 'mev'
    
    @property
    def max_allocation(self) -> float:
        """Default maximum sector allocation."""
        limits = {
            StrategySector.FUNDING_RATE: 0.50,
            StrategySector.BASIS_TRADING: 0.40,
            StrategySector.PAIRS_TRADING: 0.35,
            StrategySector.OPTIONS_VOL: 0.30,
            StrategySector.MARKET_MAKING: 0.25,
            StrategySector.LIQUIDATION: 0.20,
            StrategySector.MEV: 0.15
        }
        return limits[self]


class StressScenario(Enum):
    """Predefined stress test scenarios."""
    EXCHANGE_FAILURE = 'exchange_failure'
    DEX_EXPLOIT = 'dex_exploit'
    LIQUIDITY_CRISIS = 'liquidity_crisis'
    CORRELATION_BREAKDOWN = 'correlation_breakdown'
    BLACK_SWAN = 'black_swan'
    GAS_SPIKE = 'gas_spike'
    STABLECOIN_DEPEG = 'stablecoin_depeg'
    REGULATORY_SHOCK = 'regulatory_shock'


class ConstraintType(Enum):
    """Types of portfolio constraints."""
    VENUE = 'venue'
    SECTOR = 'sector'
    POSITION = 'position'
    CORRELATION = 'correlation'
    RISK = 'risk'
    TURNOVER = 'turnover'
    LIQUIDITY = 'liquidity'


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PortfolioConstraints:
    """
    Comprehensive portfolio constraints configuration.
    
    Attributes
    ----------
    max_cex : float
        Maximum CEX allocation (default: 0.70)
    max_dex : float
        Maximum DEX allocation (default: 0.30)
    max_hybrid : float
        Maximum hybrid venue allocation (default: 0.40)
    max_single_venue : float
        Maximum allocation to single venue (default: 0.25)
    max_strategy : float
        Maximum allocation to single strategy (default: 0.25)
    min_strategy : float
        Minimum allocation if strategy included (default: 0.01)
    max_sector : float
        Maximum sector allocation (default: 0.40)
    max_correlated : float
        Maximum combined allocation to highly correlated strategies
    correlation_threshold : float
        Threshold for "highly correlated" (default: 0.7)
    var_limit : float
        Maximum 1-day 95% VaR (default: 0.03)
    max_drawdown : float
        Maximum drawdown limit (default: 0.20)
    max_leverage : float
        Maximum gross leverage (default: 2.0)
    max_turnover : float
        Maximum daily turnover (default: 0.20)
    """
    # Venue constraints
    max_cex: float = 0.70
    max_dex: float = 0.30
    max_hybrid: float = 0.40
    max_single_venue: float = 0.25
    
    # Strategy constraints
    max_strategy: float = 0.25
    min_strategy: float = 0.01
    
    # Sector constraints
    max_sector: float = 0.40
    sector_limits: Dict[str, float] = field(default_factory=dict)
    
    # Correlation constraints
    max_correlated: float = 0.40
    correlation_threshold: float = 0.70
    
    # Risk constraints
    var_limit: float = 0.03
    max_drawdown: float = 0.20
    max_leverage: float = 1.0  # PDF: 1.0x leverage only

    # Turnover constraints
    max_turnover: float = 0.20
    
    def __post_init__(self):
        """Set default sector limits if not provided."""
        if not self.sector_limits:
            self.sector_limits = {
                sector.value: sector.max_allocation 
                for sector in StrategySector
            }
    
    def validate(self) -> List[str]:
        """Validate constraint consistency."""
        errors = []
        
        if self.max_cex + self.max_dex < 1.0:
            pass  # OK - hybrid fills gap
        
        if self.max_strategy < self.min_strategy:
            errors.append("max_strategy must be >= min_strategy")
        
        if self.var_limit <= 0 or self.var_limit > 0.10:
            errors.append("var_limit should be between 0 and 10%")
        
        if self.max_drawdown <= 0 or self.max_drawdown > 0.50:
            errors.append("max_drawdown should be between 0 and 50%")
        
        return errors


@dataclass
class RiskLimits:
    """
    Real-time risk monitoring limits.
    
    Attributes
    ----------
    var_95_1d : float
        1-day 95% VaR limit
    var_99_1d : float
        1-day 99% VaR limit
    cvar_95_1d : float
        1-day 95% CVaR limit
    max_drawdown : float
        Maximum drawdown limit
    max_daily_loss : float
        Maximum single-day loss limit
    max_position_risk : float
        Maximum risk contribution from single position
    correlation_limit : float
        Trigger for correlation breakdown alert
    """
    var_95_1d: float = 0.03
    var_99_1d: float = 0.05
    cvar_95_1d: float = 0.045
    max_drawdown: float = 0.20
    max_daily_loss: float = 0.05
    max_position_risk: float = 0.30
    correlation_limit: float = 0.80


@dataclass
class StrategyMetadata:
    """
    Metadata for a single strategy.
    
    Attributes
    ----------
    name : str
        Strategy identifier
    venue : str
        Trading venue
    venue_type : VenueType
        Type of venue (CEX/DEX/Hybrid)
    sector : StrategySector
        Strategy sector
    tier : int
        Quality tier (1=best, 5=worst)
    capacity_usd : float
        Maximum strategy capacity in USD
    min_holding_period : int
        Minimum holding period in hours
    expected_sharpe : float
        Historical/expected Sharpe ratio
    correlation_to_btc : float
        Correlation to BTC
    """
    name: str
    venue: str
    venue_type: VenueType
    sector: StrategySector
    tier: int = 3
    capacity_usd: float = 1_000_000
    min_holding_period: int = 8
    expected_sharpe: float = 1.5
    correlation_to_btc: float = 0.2


@dataclass
class OptimizationResult:
    """
    Result of portfolio optimization.
    
    Attributes
    ----------
    weights : Dict[str, float]
        Optimal portfolio weights
    method : OptimizationMethod
        Optimization method used
    expected_return : float
        Expected portfolio return (annualized)
    expected_volatility : float
        Expected portfolio volatility (annualized)
    sharpe_ratio : float
        Expected Sharpe ratio
    diversification_ratio : float
        Portfolio diversification ratio
    effective_n : float
        Effective number of strategies
    constraint_violations : List[str]
        Any constraint violations
    solver_status : str
        Optimization solver status
    """
    weights: Dict[str, float]
    method: OptimizationMethod
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    diversification_ratio: float = 1.0
    effective_n: float = 1.0
    constraint_violations: List[str] = field(default_factory=list)
    solver_status: str = 'optimal'
    
    @property
    def is_feasible(self) -> bool:
        """Check if solution is feasible."""
        return (
            self.solver_status == 'optimal' and 
            len(self.constraint_violations) == 0
        )
    
    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Optimization Result ({self.method.value})\n"
            f"{'=' * 40}\n"
            f"Expected Return:     {self.expected_return:>8.2%}\n"
            f"Expected Volatility: {self.expected_volatility:>8.2%}\n"
            f"Sharpe Ratio:        {self.sharpe_ratio:>8.2f}\n"
            f"Diversification:     {self.diversification_ratio:>8.2f}\n"
            f"Effective N:         {self.effective_n:>8.1f}\n"
            f"Strategies:          {len(self.weights):>8d}\n"
            f"Status:              {self.solver_status}"
        )


@dataclass
class RiskReport:
    """
    Comprehensive risk analysis report.
    
    Attributes
    ----------
    var_95 : float
        95% Value at Risk (1-day)
    var_99 : float
        99% Value at Risk (1-day)
    cvar_95 : float
        95% Conditional VaR (1-day)
    volatility : float
        Annualized volatility
    max_drawdown : float
        Maximum historical drawdown
    current_drawdown : float
        Current drawdown from peak
    correlation_to_btc : float
        Portfolio correlation to BTC
    beta_to_btc : float
        Portfolio beta to BTC
    risk_contributions : Dict[str, float]
        Risk contribution by strategy
    factor_exposures : Dict[str, float]
        Factor exposure estimates
    """
    var_95: float
    var_99: float
    cvar_95: float
    volatility: float
    max_drawdown: float
    current_drawdown: float
    correlation_to_btc: float
    beta_to_btc: float
    risk_contributions: Dict[str, float] = field(default_factory=dict)
    factor_exposures: Dict[str, float] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Risk Report\n"
            f"{'=' * 40}\n"
            f"VaR (95%, 1d):       {self.var_95:>8.2%}\n"
            f"VaR (99%, 1d):       {self.var_99:>8.2%}\n"
            f"CVaR (95%, 1d):      {self.cvar_95:>8.2%}\n"
            f"Volatility (ann):    {self.volatility:>8.2%}\n"
            f"Max Drawdown:        {self.max_drawdown:>8.2%}\n"
            f"Current Drawdown:    {self.current_drawdown:>8.2%}\n"
            f"Correlation to BTC:  {self.correlation_to_btc:>8.2f}\n"
            f"Beta to BTC:         {self.beta_to_btc:>8.2f}"
        )


@dataclass
class StressTestResult:
    """
    Result of a single stress test scenario.
    
    Attributes
    ----------
    scenario : StressScenario
        Scenario tested
    portfolio_loss : float
        Estimated portfolio loss
    var_breach : bool
        Whether VaR limit breached
    drawdown_breach : bool
        Whether drawdown limit breached
    worst_strategy : str
        Strategy with largest loss
    worst_strategy_loss : float
        Loss of worst strategy
    recovery_estimate_days : int
        Estimated recovery time
    """
    scenario: StressScenario
    portfolio_loss: float
    var_breach: bool
    drawdown_breach: bool
    worst_strategy: str
    worst_strategy_loss: float
    recovery_estimate_days: int
    strategy_losses: Dict[str, float] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Stress Test: {self.scenario.value}\n"
            f"{'=' * 40}\n"
            f"Portfolio Loss:      {self.portfolio_loss:>8.2%}\n"
            f"VaR Breach:          {'YES' if self.var_breach else 'NO':>8}\n"
            f"Drawdown Breach:     {'YES' if self.drawdown_breach else 'NO':>8}\n"
            f"Worst Strategy:      {self.worst_strategy:>8}\n"
            f"Worst Loss:          {self.worst_strategy_loss:>8.2%}\n"
            f"Recovery (days):     {self.recovery_estimate_days:>8}"
        )


# =============================================================================
# Module Exports
# =============================================================================

# Core classes (to be implemented in submodules)
__all__ = [
    # Enums
    'OptimizationMethod',
    'RiskMeasure',
    'VenueType',
    'StrategySector',
    'StressScenario',
    'ConstraintType',
    
    # Data classes
    'PortfolioConstraints',
    'RiskLimits',
    'StrategyMetadata',
    'OptimizationResult',
    'RiskReport',
    'StressTestResult',
    
    # Core modules (from submodules)
    'CorrelationAnalyzer',
    'PortfolioOptimizer',
    'HierarchicalRiskParity',
    'RiskParity',
    'BlackLitterman',
    'RiskAnalyzer',
    'StressTestRunner',
    'VenueConstraints',
    'ConstraintManager',
    'RiskMonitor',
]

# Lazy imports for optional heavy dependencies
def __getattr__(name):
    """Lazy import for heavy modules."""
    if name == 'CorrelationAnalyzer':
        from .correlation_analysis import CorrelationAnalyzer
        return CorrelationAnalyzer
    elif name == 'PortfolioOptimizer':
        from .optimization import PortfolioOptimizer
        return PortfolioOptimizer
    elif name == 'HierarchicalRiskParity':
        from .optimization import HierarchicalRiskParity
        return HierarchicalRiskParity
    elif name == 'RiskParity':
        from .optimization import RiskParity
        return RiskParity
    elif name == 'BlackLitterman':
        from .optimization import BlackLitterman
        return BlackLitterman
    elif name == 'RiskAnalyzer':
        from .risk_analysis import RiskAnalyzer
        return RiskAnalyzer
    elif name == 'StressTestRunner':
        from .stress_testing import StressTestRunner
        return StressTestRunner
    elif name == 'VenueConstraints':
        from .allocation import VenueConstraints
        return VenueConstraints
    elif name == 'ConstraintManager':
        from .allocation import ConstraintManager
        return ConstraintManager
    elif name == 'RiskMonitor':
        from .risk_analysis import RiskMonitor
        return RiskMonitor
    
    raise AttributeError(f"module 'portfolio' has no attribute '{name}'")