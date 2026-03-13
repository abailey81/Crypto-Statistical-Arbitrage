"""
Correlation Analysis Module
===========================

Professional-quality correlation analysis for multi-venue crypto portfolio.

Features
--------
1. Static Correlation Analysis:
    - Pearson, Spearman, Kendall correlations
    - Statistical significance testing
    - Confidence intervals
    
2. Time-Varying Correlations:
    - Rolling window correlations
    - Exponentially weighted (EWMA) correlations
    - DCC-GARCH correlations (requires arch package)
    
3. Crisis Correlation Analysis:
    - Pre/during/post crisis comparison
    - Correlation breakdown detection
    - Regime-dependent correlations
    
4. Correlation Clustering:
    - Hierarchical clustering
    - Optimal leaf ordering
    - Dendrogram visualization support

5. Cross-Venue Analysis:
    - Venue-level correlations
    - Sector-level correlations
    - Strategy tier correlations

Theory
------
Correlation breakdown during crises is a well-documented phenomenon in
financial markets. During stress periods, correlations tend to increase
toward 1.0, reducing diversification benefits precisely when they are
most needed. This module provides tools to:

1. Detect correlation regimes
2. Estimate crisis correlations
3. Stress test diversification assumptions
4. Identify truly uncorrelated strategies

References
----------
1. Engle, R. (2002). Dynamic Conditional Correlation. Journal of Business
   & Economic Statistics.
   
2. López de Prado, M. (2016). Building Diversified Portfolios that
   Outperform Out-of-Sample. Journal of Portfolio Management.

Author: Crypto StatArb Quantitative Research
Version: 2.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class CorrelationType(Enum):
    """Types of correlation measures."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    
    @property
    def scipy_func(self):
        """Get corresponding scipy function."""
        funcs = {
            CorrelationType.PEARSON: stats.pearsonr,
            CorrelationType.SPEARMAN: stats.spearmanr,
            CorrelationType.KENDALL: stats.kendalltau,
        }
        return funcs[self]


class CorrelationRegime(Enum):
    """Correlation regime classifications."""
    LOW = "low"           # Avg correlation < 0.3
    NORMAL = "normal"     # Avg correlation 0.3 - 0.6
    HIGH = "high"         # Avg correlation 0.6 - 0.8
    CRISIS = "crisis"     # Avg correlation > 0.8
    
    @classmethod
    def from_value(cls, avg_corr: float) -> 'CorrelationRegime':
        """Classify regime from average correlation."""
        if avg_corr < 0.3:
            return cls.LOW
        elif avg_corr < 0.6:
            return cls.NORMAL
        elif avg_corr < 0.8:
            return cls.HIGH
        else:
            return cls.CRISIS


class ClusterMethod(Enum):
    """Hierarchical clustering methods."""
    WARD = "ward"
    COMPLETE = "complete"
    AVERAGE = "average"
    SINGLE = "single"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CorrelationResult:
    """
    Container for correlation analysis results.
    
    Attributes
    ----------
    correlation_matrix : pd.DataFrame
        N x N correlation matrix
    p_values : pd.DataFrame
        Statistical significance p-values
    method : str
        Correlation method used
    period : str
        Analysis period description
    n_observations : int
        Number of observations used
    significant_pairs : List[Tuple]
        Pairs with statistically significant correlation
    regime : CorrelationRegime
        Current correlation regime
    avg_correlation : float
        Average off-diagonal correlation
    """
    correlation_matrix: pd.DataFrame
    p_values: pd.DataFrame
    method: str
    period: str
    n_observations: int
    significant_pairs: List[Tuple[str, str, float]]
    regime: CorrelationRegime = CorrelationRegime.NORMAL
    avg_correlation: float = 0.0
    
    def __post_init__(self):
        """Calculate derived attributes."""
        if self.avg_correlation == 0.0:
            upper = self.correlation_matrix.values[
                np.triu_indices_from(self.correlation_matrix.values, k=1)
            ]
            self.avg_correlation = np.nanmean(upper)
        
        if self.regime == CorrelationRegime.NORMAL:
            self.regime = CorrelationRegime.from_value(self.avg_correlation)
    
    def get_correlation(self, strategy1: str, strategy2: str) -> float:
        """Get correlation between two strategies."""
        return self.correlation_matrix.loc[strategy1, strategy2]
    
    def get_p_value(self, strategy1: str, strategy2: str) -> float:
        """Get p-value for correlation between two strategies."""
        return self.p_values.loc[strategy1, strategy2]
    
    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Correlation Analysis ({self.method})\n"
            f"{'=' * 50}\n"
            f"Period:                {self.period}\n"
            f"Observations:          {self.n_observations}\n"
            f"Strategies:            {len(self.correlation_matrix)}\n"
            f"Average Correlation:   {self.avg_correlation:.4f}\n"
            f"Regime:                {self.regime.value}\n"
            f"Significant Pairs:     {len(self.significant_pairs)}"
        )


@dataclass
class RollingCorrelationResult:
    """
    Container for rolling correlation analysis.
    
    Attributes
    ----------
    correlations : pd.Series
        Time series of correlations
    strategy1 : str
        First strategy name
    strategy2 : str
        Second strategy name
    window : int
        Rolling window size
    statistics : Dict
        Summary statistics
    """
    correlations: pd.Series
    strategy1: str
    strategy2: str
    window: int
    statistics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate statistics."""
        if not self.statistics:
            clean = self.correlations.dropna()
            self.statistics = {
                'mean': clean.mean(),
                'std': clean.std(),
                'min': clean.min(),
                'max': clean.max(),
                'range': clean.max() - clean.min(),
                'stability': 1 - clean.std(),  # Higher = more stable
                'current': clean.iloc[-1] if len(clean) > 0 else np.nan,
            }


@dataclass 
class CrisisCorrelationResult:
    """
    Container for crisis correlation analysis.
    
    Attributes
    ----------
    crisis_name : str
        Name of crisis period
    pre_crisis_corr : float
        Average correlation before crisis
    crisis_corr : float
        Average correlation during crisis
    post_crisis_corr : float
        Average correlation after crisis
    correlation_change : float
        Change in correlation (crisis - pre)
    max_correlation_increase : float
        Maximum correlation increase for any pair
    worst_pair : Tuple[str, str]
        Pair with largest correlation increase
    diversification_loss : float
        Estimated diversification benefit loss
    """
    crisis_name: str
    pre_crisis_corr: float
    crisis_corr: float
    post_crisis_corr: float = np.nan
    correlation_change: float = 0.0
    max_correlation_increase: float = 0.0
    worst_pair: Tuple[str, str] = ('', '')
    diversification_loss: float = 0.0
    pre_crisis_matrix: Optional[pd.DataFrame] = None
    crisis_matrix: Optional[pd.DataFrame] = None
    
    def __post_init__(self):
        """Calculate derived attributes."""
        self.correlation_change = self.crisis_corr - self.pre_crisis_corr
        
        # Estimate diversification loss
        # Diversification ratio ~ 1 / sqrt(avg_correlation)
        if self.pre_crisis_corr > 0 and self.crisis_corr > 0:
            pre_div = 1 / np.sqrt(self.pre_crisis_corr)
            crisis_div = 1 / np.sqrt(self.crisis_corr)
            self.diversification_loss = (pre_div - crisis_div) / pre_div


@dataclass
class ClusteringResult:
    """
    Container for correlation clustering results.
    
    Attributes
    ----------
    linkage : np.ndarray
        Scipy linkage matrix
    ordered_strategies : List[str]
        Strategies in clustered order
    distance_matrix : pd.DataFrame
        Distance matrix used for clustering
    n_clusters : int
        Optimal number of clusters
    cluster_assignments : Dict[str, int]
        Strategy to cluster mapping
    """
    linkage: np.ndarray
    ordered_strategies: List[str]
    distance_matrix: pd.DataFrame
    n_clusters: int = 0
    cluster_assignments: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# Main Correlation Analyzer Class
# =============================================================================

class CorrelationAnalyzer:
    """
    Comprehensive correlation analysis for strategy returns.
    
    Supports multiple correlation methods, time-varying analysis,
    crisis detection, and hierarchical clustering.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Strategy returns with datetime index and strategy columns
    metadata : pd.DataFrame, optional
        Strategy metadata (venue, tier, sector, etc.)
    risk_free_rate : float, optional
        Annual risk-free rate for excess returns (default: 0.0)
    
    Attributes
    ----------
    returns : pd.DataFrame
        Strategy returns
    metadata : pd.DataFrame
        Strategy metadata
    n_strategies : int
        Number of strategies
    n_observations : int
        Number of return observations
    
    Example
    -------
    >>> analyzer = CorrelationAnalyzer(returns_df, metadata_df)
    >>> 
    >>> # Static correlation
    >>> result = analyzer.calculate_correlation()
    >>> print(result.summary())
    >>> 
    >>> # Rolling correlation
    >>> rolling = analyzer.rolling_correlation('Strat1', 'Strat2', window=60)
    >>> 
    >>> # Crisis analysis
    >>> crisis = analyzer.crisis_correlation_analysis({
    ...     'FTX': ('2022-11-01', '2022-11-30')
    ... })
    >>> 
    >>> # Clustering
    >>> clusters = analyzer.correlation_clustering()
    """
    
    # Default crisis periods for crypto markets
    DEFAULT_CRISIS_PERIODS = {
        'COVID_Crash': ('2020-03-01', '2020-03-31'),
        'May_2021_Crash': ('2021-05-01', '2021-05-31'),
        'UST_Luna': ('2022-05-01', '2022-05-31'),
        'FTX_Collapse': ('2022-11-01', '2022-11-30'),
        'Bank_Crisis_USDC': ('2023-03-01', '2023-03-31'),
        'SEC_Lawsuits': ('2023-06-01', '2023-06-30'),
    }
    
    def __init__(
        self,
        returns: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.0
    ):
        """
        Initialize correlation analyzer.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns for each strategy (columns=strategies)
        metadata : pd.DataFrame, optional
            Strategy metadata with columns: strategy, venue, venue_type, tier, sector
        risk_free_rate : float
            Annual risk-free rate (default: 0.0)
        """
        # Validate inputs
        if returns.empty:
            raise ValueError("Returns DataFrame cannot be empty")
        
        self.returns = returns.copy()
        self.metadata = metadata.copy() if metadata is not None else None
        self.risk_free_rate = risk_free_rate
        
        # Ensure datetime index
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            self.returns.index = pd.to_datetime(self.returns.index)
        
        # Sort by date
        self.returns = self.returns.sort_index()
        
        # Store dimensions
        self.n_strategies = len(self.returns.columns)
        self.n_observations = len(self.returns)
        
        # Cache for expensive computations
        self._correlation_cache: Dict[str, CorrelationResult] = {}
        
        logger.info(
            f"Initialized CorrelationAnalyzer: "
            f"{self.n_strategies} strategies, {self.n_observations} observations"
        )
    
    # =========================================================================
    # Static Correlation Analysis
    # =========================================================================
    
    def calculate_correlation(
        self,
        method: CorrelationType = CorrelationType.PEARSON,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_periods: int = 30,
        use_cache: bool = True
    ) -> CorrelationResult:
        """
        Calculate correlation matrix for all strategies.
        
        Parameters
        ----------
        method : CorrelationType
            Correlation method (pearson, spearman, kendall)
        start_date : str, optional
            Start date for analysis (YYYY-MM-DD)
        end_date : str, optional
            End date for analysis (YYYY-MM-DD)
        min_periods : int
            Minimum observations required
        use_cache : bool
            Whether to use cached results
        
        Returns
        -------
        CorrelationResult
            Correlation matrix and statistics
        
        Raises
        ------
        ValueError
            If insufficient data for analysis
        """
        # Check cache
        cache_key = f"{method.value}_{start_date}_{end_date}"
        if use_cache and cache_key in self._correlation_cache:
            return self._correlation_cache[cache_key]
        
        # Filter date range
        data = self._filter_date_range(start_date, end_date)
        
        if len(data) < min_periods:
            raise ValueError(
                f"Insufficient data: {len(data)} < {min_periods} required"
            )
        
        # Calculate correlation matrix
        corr_matrix = data.corr(method=method.value)
        
        # Calculate p-values
        p_values = self._calculate_p_values(data, method)
        
        # Find significant pairs
        significant = self._find_significant_pairs(corr_matrix, p_values)
        
        # Calculate average correlation
        upper = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        avg_corr = np.nanmean(upper)
        
        # Determine period string
        period = f"{data.index.min().date()} to {data.index.max().date()}"
        
        result = CorrelationResult(
            correlation_matrix=corr_matrix,
            p_values=p_values,
            method=method.value,
            period=period,
            n_observations=len(data),
            significant_pairs=significant,
            avg_correlation=avg_corr,
            regime=CorrelationRegime.from_value(avg_corr)
        )
        
        # Cache result
        if use_cache:
            self._correlation_cache[cache_key] = result
        
        return result
    
    def correlation_confidence_interval(
        self,
        strategy1: str,
        strategy2: str,
        confidence: float = 0.95,
        method: CorrelationType = CorrelationType.PEARSON
    ) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for correlation using Fisher z-transform.
        
        Parameters
        ----------
        strategy1 : str
            First strategy name
        strategy2 : str
            Second strategy name
        confidence : float
            Confidence level (default: 0.95)
        method : CorrelationType
            Correlation method
        
        Returns
        -------
        Tuple[float, float, float]
            (correlation, lower_bound, upper_bound)
        """
        self._validate_strategies([strategy1, strategy2])
        
        data = self.returns[[strategy1, strategy2]].dropna()
        n = len(data)
        
        if n < 4:
            return np.nan, np.nan, np.nan
        
        # Calculate correlation
        r = data[strategy1].corr(data[strategy2], method=method.value)
        
        # Fisher z-transform
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        
        # Z-critical value
        z_crit = stats.norm.ppf((1 + confidence) / 2)
        
        # Confidence interval in z-space
        z_lower = z - z_crit * se
        z_upper = z + z_crit * se
        
        # Transform back to correlation space
        r_lower = np.tanh(z_lower)
        r_upper = np.tanh(z_upper)
        
        return r, r_lower, r_upper
    
    # =========================================================================
    # Time-Varying Correlation Analysis
    # =========================================================================
    
    def rolling_correlation(
        self,
        strategy1: str,
        strategy2: str,
        window: int = 60,
        min_periods: Optional[int] = None,
        method: CorrelationType = CorrelationType.PEARSON
    ) -> RollingCorrelationResult:
        """
        Calculate rolling correlation between two strategies.
        
        Parameters
        ----------
        strategy1 : str
            First strategy name
        strategy2 : str
            Second strategy name
        window : int
            Rolling window size (days)
        min_periods : int, optional
            Minimum periods for calculation (default: window//2)
        method : CorrelationType
            Correlation method
        
        Returns
        -------
        RollingCorrelationResult
            Rolling correlation time series with statistics
        """
        self._validate_strategies([strategy1, strategy2])
        
        if min_periods is None:
            min_periods = max(window // 2, 10)
        
        # Calculate rolling correlation
        rolling = self.returns[strategy1].rolling(
            window=window,
            min_periods=min_periods
        ).corr(self.returns[strategy2])
        
        rolling.name = f"corr_{strategy1}_{strategy2}"
        
        return RollingCorrelationResult(
            correlations=rolling,
            strategy1=strategy1,
            strategy2=strategy2,
            window=window
        )
    
    def ewma_correlation(
        self,
        strategy1: str,
        strategy2: str,
        halflife: int = 30,
        min_periods: int = 20
    ) -> pd.Series:
        """
        Calculate exponentially weighted moving average correlation.
        
        EWMA gives more weight to recent observations, useful for
        detecting correlation regime changes.
        
        Parameters
        ----------
        strategy1 : str
            First strategy name
        strategy2 : str
            Second strategy name
        halflife : int
            Halflife for exponential weighting (days)
        min_periods : int
            Minimum periods for calculation
        
        Returns
        -------
        pd.Series
            EWMA correlation time series
        """
        self._validate_strategies([strategy1, strategy2])
        
        # Calculate EWMA covariance and variances
        ret1 = self.returns[strategy1]
        ret2 = self.returns[strategy2]
        
        # Demeaned returns (using EWMA of mean)
        mean1 = ret1.ewm(halflife=halflife, min_periods=min_periods).mean()
        mean2 = ret2.ewm(halflife=halflife, min_periods=min_periods).mean()
        
        demean1 = ret1 - mean1
        demean2 = ret2 - mean2
        
        # EWMA covariance
        cov = (demean1 * demean2).ewm(halflife=halflife, min_periods=min_periods).mean()
        
        # EWMA variances
        var1 = (demean1 ** 2).ewm(halflife=halflife, min_periods=min_periods).mean()
        var2 = (demean2 ** 2).ewm(halflife=halflife, min_periods=min_periods).mean()
        
        # Correlation
        corr = cov / np.sqrt(var1 * var2)
        corr.name = f"ewma_corr_{strategy1}_{strategy2}"
        
        return corr
    
    def dcc_correlation(
        self,
        strategy1: str,
        strategy2: str,
        p: int = 1,
        q: int = 1
    ) -> pd.Series:
        """
        Calculate DCC-GARCH dynamic conditional correlation.
        
        DCC (Dynamic Conditional Correlation) model captures time-varying
        correlations more accurately than rolling windows during volatility
        clustering periods.
        
        Requires: arch package (pip install arch)
        
        Parameters
        ----------
        strategy1 : str
            First strategy name
        strategy2 : str
            Second strategy name
        p : int
            GARCH lag order
        q : int
            ARCH lag order
        
        Returns
        -------
        pd.Series
            DCC correlation time series
        """
        self._validate_strategies([strategy1, strategy2])
        
        try:
            from arch import arch_model
            from arch.univariate import GARCH
        except ImportError:
            logger.warning(
                "arch package not installed. Using EWMA correlation instead. "
                "Install with: pip install arch"
            )
            return self.ewma_correlation(strategy1, strategy2)
        
        data = self.returns[[strategy1, strategy2]].dropna() * 100  # Scale for numerical stability
        
        # Fit univariate GARCH models
        garch_params = {}
        std_residuals = {}
        
        for col in [strategy1, strategy2]:
            model = arch_model(data[col], vol='Garch', p=p, q=q, mean='Zero')
            res = model.fit(disp='off')
            garch_params[col] = res
            std_residuals[col] = res.std_resid
        
        # DCC estimation
        z1 = std_residuals[strategy1]
        z2 = std_residuals[strategy2]
        
        # Unconditional correlation
        rho_bar = np.corrcoef(z1, z2)[0, 1]
        
        # DCC parameters (simplified estimation)
        alpha = 0.05  # Weight on recent observations
        beta = 0.90   # Persistence
        
        # Initialize Q
        Q = np.array([[1, rho_bar], [rho_bar, 1]])
        Q_bar = Q.copy()
        
        correlations = []
        
        for t in range(len(z1)):
            if t == 0:
                correlations.append(rho_bar)
            else:
                # Update Q
                z_t = np.array([z1.iloc[t-1], z2.iloc[t-1]])
                Q = (1 - alpha - beta) * Q_bar + alpha * np.outer(z_t, z_t) + beta * Q
                
                # Standardize to correlation
                D = np.sqrt(np.diag(Q))
                R = Q / np.outer(D, D)
                correlations.append(R[0, 1])
        
        result = pd.Series(correlations, index=data.index, name=f"dcc_corr_{strategy1}_{strategy2}")
        return result
    
    def all_pairwise_rolling_correlations(
        self,
        window: int = 60
    ) -> pd.DataFrame:
        """
        Calculate rolling correlations for all strategy pairs.
        
        Parameters
        ----------
        window : int
            Rolling window size
        
        Returns
        -------
        pd.DataFrame
            DataFrame with datetime index and pair columns
        """
        strategies = self.returns.columns.tolist()
        n = len(strategies)
        
        results = {}
        
        for i in range(n):
            for j in range(i + 1, n):
                s1, s2 = strategies[i], strategies[j]
                pair_name = f"{s1}_{s2}"
                
                roll_result = self.rolling_correlation(s1, s2, window=window)
                results[pair_name] = roll_result.correlations
        
        return pd.DataFrame(results)
    
    def average_rolling_correlation(
        self,
        window: int = 60
    ) -> pd.Series:
        """
        Calculate average correlation across all pairs over time.
        
        Useful for detecting correlation regime changes.
        
        Parameters
        ----------
        window : int
            Rolling window size
        
        Returns
        -------
        pd.Series
            Average correlation time series
        """
        all_rolling = self.all_pairwise_rolling_correlations(window=window)
        avg_corr = all_rolling.mean(axis=1)
        avg_corr.name = 'avg_correlation'
        return avg_corr
    
    # =========================================================================
    # Cross-Venue and Sector Correlation
    # =========================================================================
    
    def venue_correlation(self) -> pd.DataFrame:
        """
        Calculate average correlation by venue type.
        
        Aggregates strategies by venue type (CEX, DEX, Hybrid) and
        computes correlation between venue-level returns.
        
        Returns
        -------
        pd.DataFrame
            Correlation matrix by venue type
        
        Raises
        ------
        ValueError
            If metadata not provided
        """
        if self.metadata is None:
            raise ValueError("Metadata required for venue correlation")
        
        # Build venue map
        venue_map = self._build_venue_map()
        
        venue_returns = {}
        for venue_type in ['CEX', 'DEX', 'HYBRID']:
            strategies = [
                s for s in self.returns.columns 
                if venue_map.get(s, '').upper() == venue_type
            ]
            if strategies:
                venue_returns[venue_type] = self.returns[strategies].mean(axis=1)
        
        if not venue_returns:
            raise ValueError("No strategies mapped to venues")
        
        venue_df = pd.DataFrame(venue_returns)
        return venue_df.corr()
    
    def sector_correlation(self) -> pd.DataFrame:
        """
        Calculate correlation between strategy sectors.
        
        Returns
        -------
        pd.DataFrame
            Correlation matrix by sector
        
        Raises
        ------
        ValueError
            If metadata not provided or no sectors mapped
        """
        if self.metadata is None:
            raise ValueError("Metadata required for sector correlation")
        
        # Build sector map
        sector_map = self._build_sector_map()
        
        sector_returns = {}
        sectors = set(sector_map.values())
        
        for sector in sectors:
            if pd.isna(sector):
                continue
            strategies = [
                s for s in self.returns.columns 
                if sector_map.get(s) == sector
            ]
            if strategies:
                sector_returns[sector] = self.returns[strategies].mean(axis=1)
        
        if not sector_returns:
            raise ValueError("No strategies mapped to sectors")
        
        sector_df = pd.DataFrame(sector_returns)
        return sector_df.corr()
    
    def tier_correlation(self) -> pd.DataFrame:
        """
        Calculate correlation between strategy tiers.
        
        Returns
        -------
        pd.DataFrame
            Correlation matrix by tier
        """
        if self.metadata is None:
            raise ValueError("Metadata required for tier correlation")
        
        tier_map = self._build_tier_map()
        
        tier_returns = {}
        tiers = sorted(set(tier_map.values()))
        
        for tier in tiers:
            strategies = [
                s for s in self.returns.columns 
                if tier_map.get(s) == tier
            ]
            if strategies:
                tier_returns[f"Tier_{tier}"] = self.returns[strategies].mean(axis=1)
        
        if not tier_returns:
            raise ValueError("No strategies mapped to tiers")
        
        tier_df = pd.DataFrame(tier_returns)
        return tier_df.corr()
    
    # =========================================================================
    # Crisis Correlation Analysis
    # =========================================================================
    
    def crisis_correlation_analysis(
        self,
        crisis_periods: Optional[Dict[str, Tuple[str, str]]] = None,
        comparison_window: int = 90
    ) -> Dict[str, CrisisCorrelationResult]:
        """
        Analyze correlation changes during crisis periods.
        
        Compares correlations before, during, and after crisis events.
        Useful for understanding diversification breakdown during stress.
        
        Parameters
        ----------
        crisis_periods : Dict[str, Tuple[str, str]], optional
            Dictionary of crisis name to (start_date, end_date)
            Uses default crypto crises if not provided
        comparison_window : int
            Days before crisis for comparison
        
        Returns
        -------
        Dict[str, CrisisCorrelationResult]
            Crisis analysis results for each period
        """
        if crisis_periods is None:
            crisis_periods = self.DEFAULT_CRISIS_PERIODS
        
        results = {}
        
        for crisis_name, (start, end) in crisis_periods.items():
            crisis_start = pd.to_datetime(start)
            crisis_end = pd.to_datetime(end)
            
            # Check data availability
            if crisis_start < self.returns.index.min():
                logger.warning(f"No data for {crisis_name}: starts before data")
                continue
            
            if crisis_end > self.returns.index.max():
                logger.warning(f"No data for {crisis_name}: ends after data")
                continue
            
            # Pre-crisis period
            pre_start = crisis_start - pd.Timedelta(days=comparison_window)
            pre_end = crisis_start - pd.Timedelta(days=1)
            
            # Post-crisis period
            post_start = crisis_end + pd.Timedelta(days=1)
            post_end = crisis_end + pd.Timedelta(days=comparison_window)
            
            try:
                # Calculate correlations for each period
                pre_result = self.calculate_correlation(
                    start_date=str(pre_start.date()),
                    end_date=str(pre_end.date()),
                    min_periods=20,
                    use_cache=False
                )
                
                crisis_result = self.calculate_correlation(
                    start_date=start,
                    end_date=end,
                    min_periods=5,
                    use_cache=False
                )
                
                # Calculate change matrix
                change_matrix = (
                    crisis_result.correlation_matrix - 
                    pre_result.correlation_matrix
                )
                
                # Find worst pair
                upper_idx = np.triu_indices_from(change_matrix.values, k=1)
                changes = change_matrix.values[upper_idx]
                max_idx = np.argmax(changes)
                
                pairs = [
                    (change_matrix.columns[i], change_matrix.columns[j])
                    for i, j in zip(*upper_idx)
                ]
                worst_pair = pairs[max_idx] if pairs else ('', '')
                
                # Post-crisis correlation (if data available)
                post_corr = np.nan
                if post_end <= self.returns.index.max():
                    try:
                        post_result = self.calculate_correlation(
                            start_date=str(post_start.date()),
                            end_date=str(post_end.date()),
                            min_periods=20,
                            use_cache=False
                        )
                        post_corr = post_result.avg_correlation
                    except Exception:
                        pass
                
                results[crisis_name] = CrisisCorrelationResult(
                    crisis_name=crisis_name,
                    pre_crisis_corr=pre_result.avg_correlation,
                    crisis_corr=crisis_result.avg_correlation,
                    post_crisis_corr=post_corr,
                    max_correlation_increase=changes.max() if len(changes) > 0 else 0,
                    worst_pair=worst_pair,
                    pre_crisis_matrix=pre_result.correlation_matrix,
                    crisis_matrix=crisis_result.correlation_matrix
                )
                
                logger.info(
                    f"Crisis {crisis_name}: correlation "
                    f"{pre_result.avg_correlation:.3f} → {crisis_result.avg_correlation:.3f}"
                )
                
            except Exception as e:
                logger.warning(f"Could not analyze {crisis_name}: {e}")
        
        return results
    
    def detect_correlation_breakdown(
        self,
        threshold: float = 0.2,
        window: int = 30
    ) -> pd.DataFrame:
        """
        Detect periods where correlations spike significantly.
        
        Parameters
        ----------
        threshold : float
            Minimum correlation increase to flag (default: 0.2)
        window : int
            Rolling window for analysis
        
        Returns
        -------
        pd.DataFrame
            Periods with correlation breakdown detected
        """
        avg_corr = self.average_rolling_correlation(window=window)
        
        # Calculate correlation change
        corr_change = avg_corr.diff(periods=window // 2)
        
        # Find breakdown periods
        breakdowns = corr_change[corr_change > threshold]
        
        results = []
        for date, change in breakdowns.items():
            results.append({
                'date': date,
                'correlation_change': change,
                'correlation_level': avg_corr.loc[date],
                'regime': CorrelationRegime.from_value(avg_corr.loc[date]).value
            })
        
        return pd.DataFrame(results)
    
    # =========================================================================
    # Correlation Clustering
    # =========================================================================
    
    def correlation_clustering(
        self,
        method: ClusterMethod = ClusterMethod.WARD,
        metric: str = 'euclidean',
        n_clusters: Optional[int] = None
    ) -> ClusteringResult:
        """
        Perform hierarchical clustering based on correlations.
        
        Converts correlation matrix to distance matrix and performs
        agglomerative clustering. Useful for identifying groups of
        correlated strategies.
        
        Parameters
        ----------
        method : ClusterMethod
            Linkage method ('ward', 'complete', 'average', 'single')
        metric : str
            Distance metric
        n_clusters : int, optional
            Number of clusters to form (auto-detected if None)
        
        Returns
        -------
        ClusteringResult
            Clustering results with linkage matrix and assignments
        """
        corr = self.returns.corr()
        
        # Convert correlation to distance (1 - |corr|)
        distance = 1 - np.abs(corr.values)
        np.fill_diagonal(distance, 0)
        
        # Ensure symmetry
        distance = (distance + distance.T) / 2
        
        # Store distance matrix
        dist_df = pd.DataFrame(
            distance, 
            index=corr.index, 
            columns=corr.columns
        )
        
        # Convert to condensed form for scipy
        condensed = squareform(distance)
        
        # Perform clustering
        linkage = hierarchy.linkage(condensed, method=method.value)
        
        # Get optimal leaf ordering
        try:
            order = hierarchy.leaves_list(
                hierarchy.optimal_leaf_ordering(linkage, condensed)
            )
        except Exception:
            order = hierarchy.leaves_list(linkage)
        
        ordered_names = [self.returns.columns[i] for i in order]
        
        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            n_clusters = self._optimal_n_clusters(linkage)
        
        # Get cluster assignments
        clusters = hierarchy.fcluster(linkage, n_clusters, criterion='maxclust')
        assignments = {
            self.returns.columns[i]: int(clusters[i])
            for i in range(len(clusters))
        }
        
        return ClusteringResult(
            linkage=linkage,
            ordered_strategies=ordered_names,
            distance_matrix=dist_df,
            n_clusters=n_clusters,
            cluster_assignments=assignments
        )
    
    def get_cluster_correlations(
        self,
        clustering: ClusteringResult
    ) -> Dict[int, pd.DataFrame]:
        """
        Get correlation matrices within each cluster.
        
        Parameters
        ----------
        clustering : ClusteringResult
            Clustering result from correlation_clustering()
        
        Returns
        -------
        Dict[int, pd.DataFrame]
            Correlation matrix for each cluster
        """
        cluster_corrs = {}
        
        for cluster_id in range(1, clustering.n_clusters + 1):
            strategies = [
                s for s, c in clustering.cluster_assignments.items()
                if c == cluster_id
            ]
            
            if len(strategies) > 1:
                cluster_corrs[cluster_id] = self.returns[strategies].corr()
            elif len(strategies) == 1:
                cluster_corrs[cluster_id] = pd.DataFrame(
                    [[1.0]], 
                    index=strategies, 
                    columns=strategies
                )
        
        return cluster_corrs
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_highly_correlated_pairs(
        self,
        threshold: float = 0.7,
        method: CorrelationType = CorrelationType.PEARSON
    ) -> pd.DataFrame:
        """
        Find pairs with correlation above threshold.
        
        Parameters
        ----------
        threshold : float
            Minimum absolute correlation
        method : CorrelationType
            Correlation method
        
        Returns
        -------
        pd.DataFrame
            Highly correlated pairs with details
        """
        result = self.calculate_correlation(method=method)
        corr = result.correlation_matrix
        p_vals = result.p_values
        
        pairs = []
        for i, s1 in enumerate(corr.columns):
            for j, s2 in enumerate(corr.columns):
                if i < j:  # Upper triangle only
                    c = corr.loc[s1, s2]
                    if abs(c) >= threshold:
                        pairs.append({
                            'strategy1': s1,
                            'strategy2': s2,
                            'correlation': c,
                            'abs_correlation': abs(c),
                            'p_value': p_vals.loc[s1, s2]
                        })
        
        if not pairs:
            return pd.DataFrame(columns=[
                'strategy1', 'strategy2', 'correlation', 
                'abs_correlation', 'p_value'
            ])
        
        df = pd.DataFrame(pairs)
        return df.sort_values('abs_correlation', ascending=False)
    
    def correlation_stability(
        self,
        window: int = 60,
        min_observations: int = 30
    ) -> pd.DataFrame:
        """
        Analyze stability of correlations over time.
        
        Parameters
        ----------
        window : int
            Rolling window for correlation calculation
        min_observations : int
            Minimum observations for stability calculation
        
        Returns
        -------
        pd.DataFrame
            Stability metrics for each strategy pair
        """
        results = []
        strategies = self.returns.columns.tolist()
        n = len(strategies)
        
        for i in range(n):
            for j in range(i + 1, n):
                s1, s2 = strategies[i], strategies[j]
                
                roll_result = self.rolling_correlation(s1, s2, window=window)
                rolling = roll_result.correlations.dropna()
                
                if len(rolling) < min_observations:
                    continue
                
                results.append({
                    'strategy1': s1,
                    'strategy2': s2,
                    'mean_correlation': rolling.mean(),
                    'std_correlation': rolling.std(),
                    'min_correlation': rolling.min(),
                    'max_correlation': rolling.max(),
                    'range': rolling.max() - rolling.min(),
                    'stability_score': 1 - rolling.std(),
                    'n_observations': len(rolling)
                })
        
        if not results:
            return pd.DataFrame()
        
        return pd.DataFrame(results).sort_values('stability_score', ascending=False)
    
    def clear_cache(self):
        """Clear correlation cache."""
        self._correlation_cache.clear()
        logger.info("Correlation cache cleared")
    
    # =========================================================================
    # Private Helper Methods
    # =========================================================================
    
    def _filter_date_range(
        self,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Filter returns to date range."""
        data = self.returns.copy()
        
        if start_date is not None:
            data = data[data.index >= start_date]
        if end_date is not None:
            data = data[data.index <= end_date]
        
        return data
    
    def _validate_strategies(self, strategies: List[str]):
        """Validate strategy names exist."""
        missing = [s for s in strategies if s not in self.returns.columns]
        if missing:
            raise ValueError(f"Strategies not found: {missing}")
    
    def _calculate_p_values(
        self,
        data: pd.DataFrame,
        method: CorrelationType
    ) -> pd.DataFrame:
        """Calculate p-values for correlation matrix."""
        n = len(data.columns)
        p_matrix = pd.DataFrame(
            np.zeros((n, n)),
            index=data.columns,
            columns=data.columns
        )
        
        corr_func = method.scipy_func
        
        for i, col1 in enumerate(data.columns):
            for j, col2 in enumerate(data.columns):
                if i < j:
                    x = data[col1].dropna()
                    y = data[col2].dropna()
                    
                    # Align
                    common = x.index.intersection(y.index)
                    x = x[common]
                    y = y[common]
                    
                    if len(x) < 3:
                        p_matrix.loc[col1, col2] = np.nan
                        p_matrix.loc[col2, col1] = np.nan
                        continue
                    
                    _, p = corr_func(x, y)
                    
                    p_matrix.loc[col1, col2] = p
                    p_matrix.loc[col2, col1] = p
        
        return p_matrix
    
    def _find_significant_pairs(
        self,
        corr_matrix: pd.DataFrame,
        p_values: pd.DataFrame,
        alpha: float = 0.05
    ) -> List[Tuple[str, str, float]]:
        """Find statistically significant correlation pairs."""
        significant = []
        
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:
                    p = p_values.loc[col1, col2]
                    if pd.notna(p) and p < alpha:
                        corr = corr_matrix.loc[col1, col2]
                        significant.append((col1, col2, corr))
        
        return sorted(significant, key=lambda x: abs(x[2]), reverse=True)
    
    def _build_venue_map(self) -> Dict[str, str]:
        """Build strategy to venue type mapping."""
        if self.metadata is None:
            return {}
        
        if 'venue_type' in self.metadata.columns:
            return self.metadata.set_index('strategy')['venue_type'].to_dict()
        elif 'venue' in self.metadata.columns:
            # Try to infer venue type from venue name
            return self.metadata.set_index('strategy')['venue'].to_dict()
        
        return {}
    
    def _build_sector_map(self) -> Dict[str, str]:
        """Build strategy to sector mapping."""
        if self.metadata is None:
            return {}
        
        if 'sector' in self.metadata.columns:
            return self.metadata.set_index('strategy')['sector'].to_dict()
        
        return {}
    
    def _build_tier_map(self) -> Dict[str, int]:
        """Build strategy to tier mapping."""
        if self.metadata is None:
            return {}
        
        if 'tier' in self.metadata.columns:
            return self.metadata.set_index('strategy')['tier'].to_dict()
        
        return {}
    
    def _optimal_n_clusters(
        self,
        linkage: np.ndarray,
        max_clusters: int = 10
    ) -> int:
        """Determine optimal number of clusters using elbow method."""
        from scipy.cluster.hierarchy import fcluster
        
        # Calculate within-cluster variance for different cluster numbers
        variances = []
        
        for n in range(2, min(max_clusters + 1, self.n_strategies)):
            clusters = fcluster(linkage, n, criterion='maxclust')
            
            # Calculate total within-cluster variance
            total_var = 0
            for c in range(1, n + 1):
                mask = clusters == c
                if mask.sum() > 1:
                    cluster_returns = self.returns.iloc[:, mask]
                    total_var += cluster_returns.var().sum()
            
            variances.append(total_var)
        
        if len(variances) < 2:
            return 3  # Default
        
        # Find elbow point (maximum curvature)
        variances = np.array(variances)
        
        # Simple elbow detection: largest decrease
        decreases = np.diff(variances)
        optimal_n = np.argmin(decreases) + 2  # +2 because we start at 2 clusters
        
        return max(2, min(optimal_n, max_clusters))


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_correlation_check(
    returns: pd.DataFrame,
    threshold: float = 0.7
) -> pd.DataFrame:
    """
    Quick utility to check for highly correlated strategies.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Strategy returns
    threshold : float
        Correlation threshold
    
    Returns
    -------
    pd.DataFrame
        Highly correlated pairs
    """
    analyzer = CorrelationAnalyzer(returns)
    return analyzer.get_highly_correlated_pairs(threshold=threshold)


def correlation_heatmap_data(
    returns: pd.DataFrame,
    method: CorrelationType = CorrelationType.PEARSON
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare data for correlation heatmap visualization.
    
    Returns correlation matrix with hierarchically clustered ordering.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Strategy returns
    method : CorrelationType
        Correlation method
    
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Reordered correlation matrix and strategy order
    """
    analyzer = CorrelationAnalyzer(returns)
    clustering = analyzer.correlation_clustering()
    result = analyzer.calculate_correlation(method=method)
    
    ordered_corr = result.correlation_matrix.loc[
        clustering.ordered_strategies,
        clustering.ordered_strategies
    ]
    
    return ordered_corr, clustering.ordered_strategies