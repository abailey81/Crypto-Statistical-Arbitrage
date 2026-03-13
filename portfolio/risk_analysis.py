"""
Risk Analysis Module
=====================

Comprehensive risk analytics for multi-venue crypto portfolios.

Provides Value at Risk (VaR), Conditional VaR, drawdown analysis,
factor exposure, and real-time risk monitoring with venue-specific
risk adjustments.

Risk Measures Implemented:
    - Parametric VaR (Gaussian)
    - Historical VaR (empirical quantile)
    - Monte Carlo VaR (simulation-based)
    - Conditional VaR / Expected Shortfall
    - Maximum Drawdown analysis
    - Venue-specific risk decomposition

Risk Limits (per PDF):
    - Portfolio VaR (95%, 1-day): max 3%
    - Maximum drawdown: 20%
    - Leverage limit: 1.0x (pairs), 2.0x (futures)
    - BTC correlation: < 0.3

Author: Tamer Atesyakar
Version: 2.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class VaRResult:
    """Value at Risk calculation result."""
    confidence: float
    horizon_days: int
    method: str
    var_pct: float         # VaR as percentage
    var_usd: float = 0.0   # VaR in USD (if portfolio value provided)
    cvar_pct: float = 0.0  # Conditional VaR / Expected Shortfall
    cvar_usd: float = 0.0

    def summary(self) -> str:
        return (
            f"VaR ({self.confidence:.0%}, {self.horizon_days}d, {self.method})\n"
            f"  VaR:  {self.var_pct:.2%}"
            f"{f'  (${self.var_usd:,.0f})' if self.var_usd else ''}\n"
            f"  CVaR: {self.cvar_pct:.2%}"
            f"{f'  (${self.cvar_usd:,.0f})' if self.cvar_usd else ''}"
        )


@dataclass
class DrawdownResult:
    """Drawdown analysis result."""
    max_drawdown: float
    max_dd_start: Optional[pd.Timestamp] = None
    max_dd_end: Optional[pd.Timestamp] = None
    max_dd_recovery: Optional[pd.Timestamp] = None
    recovery_days: int = 0
    current_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    drawdown_series: Optional[pd.Series] = None
    top_drawdowns: List[Dict] = field(default_factory=list)


@dataclass
class RiskDecomposition:
    """Risk contribution by strategy and venue."""
    total_volatility: float
    strategy_contributions: Dict[str, float] = field(default_factory=dict)
    venue_contributions: Dict[str, float] = field(default_factory=dict)
    sector_contributions: Dict[str, float] = field(default_factory=dict)
    marginal_risk: Dict[str, float] = field(default_factory=dict)


@dataclass
class RiskLimitCheck:
    """Result of checking risk limits."""
    var_95_compliant: bool
    max_dd_compliant: bool
    leverage_compliant: bool
    correlation_compliant: bool
    breaches: List[str] = field(default_factory=list)

    @property
    def all_compliant(self) -> bool:
        return (
            self.var_95_compliant
            and self.max_dd_compliant
            and self.leverage_compliant
            and self.correlation_compliant
        )


# ---------------------------------------------------------------------------
# Risk Analyzer
# ---------------------------------------------------------------------------

class RiskAnalyzer:
    """
    Comprehensive risk analysis for crypto portfolios.

    Parameters
    ----------
    returns : pd.DataFrame
        Portfolio or strategy returns.
    weights : Dict[str, float], optional
        Portfolio weights. Equal weight if not provided.
    portfolio_value : float, optional
        Portfolio notional for USD-denominated risk.
    btc_returns : pd.Series, optional
        BTC returns for correlation analysis.

    Example
    -------
        >>> analyzer = RiskAnalyzer(returns_df, weights, portfolio_value=1_000_000)
        >>> var = analyzer.calculate_var(confidence=0.95)
        >>> dd = analyzer.analyze_drawdowns()
        >>> decomp = analyzer.decompose_risk()
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
        portfolio_value: float = 0.0,
        btc_returns: Optional[pd.Series] = None,
    ):
        self.returns = returns
        self.portfolio_value = portfolio_value
        self.btc_returns = btc_returns

        if weights is None:
            n = len(returns.columns)
            self.weights = {s: 1.0 / n for s in returns.columns}
        else:
            self.weights = weights

        # Portfolio returns
        w = np.array([self.weights.get(s, 0) for s in returns.columns])
        self.portfolio_returns = (returns * w).sum(axis=1)

    # -----------------------------------------------------------------------
    # Value at Risk
    # -----------------------------------------------------------------------

    def calculate_var(
        self,
        confidence: float = 0.95,
        horizon_days: int = 1,
        method: str = 'historical',
    ) -> VaRResult:
        """
        Calculate Value at Risk.

        Parameters
        ----------
        confidence : float
            Confidence level (0.95 or 0.99).
        horizon_days : int
            Risk horizon in days.
        method : str
            'parametric', 'historical', or 'monte_carlo'.
        """
        if method == 'parametric':
            var_pct, cvar_pct = self._parametric_var(confidence, horizon_days)
        elif method == 'historical':
            var_pct, cvar_pct = self._historical_var(confidence, horizon_days)
        elif method == 'monte_carlo':
            var_pct, cvar_pct = self._monte_carlo_var(confidence, horizon_days)
        else:
            raise ValueError(f"Unknown VaR method: {method}")

        return VaRResult(
            confidence=confidence,
            horizon_days=horizon_days,
            method=method,
            var_pct=var_pct,
            var_usd=abs(var_pct) * self.portfolio_value,
            cvar_pct=cvar_pct,
            cvar_usd=abs(cvar_pct) * self.portfolio_value,
        )

    def _parametric_var(
        self, confidence: float, horizon: int
    ) -> Tuple[float, float]:
        """Gaussian parametric VaR."""
        mu = self.portfolio_returns.mean() * horizon
        sigma = self.portfolio_returns.std() * np.sqrt(horizon)
        z = stats.norm.ppf(1 - confidence)
        var = -(mu + z * sigma)

        # CVaR (Expected Shortfall)
        pdf_z = stats.norm.pdf(z)
        cvar = -(mu - sigma * pdf_z / (1 - confidence))

        return float(var), float(cvar)

    def _historical_var(
        self, confidence: float, horizon: int
    ) -> Tuple[float, float]:
        """Historical simulation VaR."""
        if horizon > 1:
            rolling = self.portfolio_returns.rolling(horizon).sum().dropna()
        else:
            rolling = self.portfolio_returns

        var = -np.percentile(rolling, (1 - confidence) * 100)
        tail = rolling[rolling <= -var]
        cvar = -tail.mean() if len(tail) > 0 else var

        return float(var), float(cvar)

    def _monte_carlo_var(
        self,
        confidence: float,
        horizon: int,
        n_sims: int = 10_000,
    ) -> Tuple[float, float]:
        """Monte Carlo simulation VaR."""
        mu = self.portfolio_returns.mean()
        sigma = self.portfolio_returns.std()

        # Simulate paths
        sims = np.random.normal(
            mu * horizon, sigma * np.sqrt(horizon), n_sims
        )

        var = -np.percentile(sims, (1 - confidence) * 100)
        tail = sims[sims <= -var]
        cvar = -tail.mean() if len(tail) > 0 else var

        return float(var), float(cvar)

    # -----------------------------------------------------------------------
    # Drawdown Analysis
    # -----------------------------------------------------------------------

    def analyze_drawdowns(self, top_n: int = 5) -> DrawdownResult:
        """
        Comprehensive drawdown analysis.

        Parameters
        ----------
        top_n : int
            Number of top drawdowns to report.
        """
        cum_returns = (1 + self.portfolio_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns - running_max) / running_max

        max_dd = drawdowns.min()
        max_dd_idx = drawdowns.idxmin()

        # Find start and recovery
        dd_end = max_dd_idx
        dd_start = cum_returns[:dd_end].idxmax() if dd_end is not None else None

        recovery = None
        recovery_days = 0
        if dd_end is not None:
            post_dd = cum_returns[dd_end:]
            peak = running_max[dd_end]
            recovered = post_dd[post_dd >= peak]
            if len(recovered) > 0:
                recovery = recovered.index[0]
                recovery_days = (recovery - dd_end).days

        # Top N drawdowns
        top_dds = []
        dd_copy = drawdowns.copy()
        for _ in range(min(top_n, len(dd_copy))):
            if dd_copy.min() >= 0:
                break
            idx = dd_copy.idxmin()
            top_dds.append({
                'drawdown': float(dd_copy[idx]),
                'date': str(idx),
            })
            # Mask nearby values
            mask_start = max(0, dd_copy.index.get_loc(idx) - 20)
            mask_end = min(len(dd_copy), dd_copy.index.get_loc(idx) + 20)
            dd_copy.iloc[mask_start:mask_end] = 0

        return DrawdownResult(
            max_drawdown=float(max_dd),
            max_dd_start=dd_start,
            max_dd_end=dd_end,
            max_dd_recovery=recovery,
            recovery_days=recovery_days,
            current_drawdown=float(drawdowns.iloc[-1]) if len(drawdowns) > 0 else 0,
            avg_drawdown=float(drawdowns.mean()),
            drawdown_series=drawdowns,
            top_drawdowns=top_dds,
        )

    # -----------------------------------------------------------------------
    # Risk Decomposition
    # -----------------------------------------------------------------------

    def decompose_risk(
        self,
        metadata: Optional[pd.DataFrame] = None,
    ) -> RiskDecomposition:
        """
        Decompose portfolio risk by strategy, venue, and sector.

        Parameters
        ----------
        metadata : pd.DataFrame, optional
            Must have columns: venue_type, sector.
        """
        w = np.array([self.weights.get(s, 0) for s in self.returns.columns])
        cov = self.returns.cov().values

        port_var = w @ cov @ w
        port_vol = np.sqrt(port_var) * np.sqrt(252) if port_var > 0 else 0

        # Marginal risk contributions
        marginal = cov @ w
        risk_contrib = w * marginal
        if port_var > 0:
            risk_contrib_pct = risk_contrib / port_var
        else:
            risk_contrib_pct = np.zeros_like(w)

        strategy_contributions = {
            s: float(risk_contrib_pct[i])
            for i, s in enumerate(self.returns.columns)
        }

        marginal_risk = {
            s: float(marginal[i] * np.sqrt(252))
            for i, s in enumerate(self.returns.columns)
        }

        # Venue and sector aggregation
        venue_contrib = {}
        sector_contrib = {}
        if metadata is not None:
            for venue in metadata.get('venue_type', pd.Series()).unique():
                mask = metadata['venue_type'] == venue
                venue_contrib[str(venue)] = sum(
                    strategy_contributions.get(s, 0)
                    for s in metadata[mask].index
                    if s in strategy_contributions
                )
            for sector in metadata.get('sector', pd.Series()).unique():
                mask = metadata['sector'] == sector
                sector_contrib[str(sector)] = sum(
                    strategy_contributions.get(s, 0)
                    for s in metadata[mask].index
                    if s in strategy_contributions
                )

        return RiskDecomposition(
            total_volatility=port_vol,
            strategy_contributions=strategy_contributions,
            venue_contributions=venue_contrib,
            sector_contributions=sector_contrib,
            marginal_risk=marginal_risk,
        )

    # -----------------------------------------------------------------------
    # Correlation Analysis
    # -----------------------------------------------------------------------

    def btc_correlation(self) -> float:
        """Portfolio correlation to BTC. Target: < 0.3."""
        if self.btc_returns is None:
            return 0.0

        aligned = pd.concat([
            self.portfolio_returns, self.btc_returns
        ], axis=1).dropna()

        if len(aligned) < 30:
            return 0.0

        return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))

    def btc_beta(self) -> float:
        """Portfolio beta to BTC."""
        if self.btc_returns is None:
            return 0.0

        aligned = pd.concat([
            self.portfolio_returns, self.btc_returns
        ], axis=1).dropna()

        if len(aligned) < 30:
            return 0.0

        cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
        btc_var = cov[1, 1]
        return float(cov[0, 1] / btc_var) if btc_var > 0 else 0.0

    # -----------------------------------------------------------------------
    # Risk Limit Checks
    # -----------------------------------------------------------------------

    def check_limits(
        self,
        var_limit: float = 0.03,
        dd_limit: float = 0.20,
        leverage_limit: float = 1.0,
        corr_limit: float = 0.30,
    ) -> RiskLimitCheck:
        """
        Check all risk limits against current portfolio state.

        Parameters
        ----------
        var_limit : float
            Maximum 1-day 95% VaR (default: 3%).
        dd_limit : float
            Maximum drawdown (default: 20%).
        leverage_limit : float
            Maximum leverage (default: 1.0x for pairs).
        corr_limit : float
            Maximum BTC correlation (default: 0.3).
        """
        var = self.calculate_var(confidence=0.95, horizon_days=1, method='historical')
        dd = self.analyze_drawdowns()
        corr = self.btc_correlation()
        leverage = sum(abs(w) for w in self.weights.values())

        breaches = []
        var_ok = var.var_pct <= var_limit
        if not var_ok:
            breaches.append(f"VaR {var.var_pct:.2%} exceeds limit {var_limit:.2%}")

        dd_ok = abs(dd.max_drawdown) <= dd_limit
        if not dd_ok:
            breaches.append(f"Max DD {dd.max_drawdown:.2%} exceeds limit {dd_limit:.2%}")

        lev_ok = leverage <= leverage_limit * 1.05  # 5% tolerance
        if not lev_ok:
            breaches.append(f"Leverage {leverage:.2f}x exceeds limit {leverage_limit:.1f}x")

        corr_ok = abs(corr) <= corr_limit
        if not corr_ok:
            breaches.append(f"BTC correlation {corr:.2f} exceeds limit {corr_limit:.2f}")

        return RiskLimitCheck(
            var_95_compliant=var_ok,
            max_dd_compliant=dd_ok,
            leverage_compliant=lev_ok,
            correlation_compliant=corr_ok,
            breaches=breaches,
        )


# ---------------------------------------------------------------------------
# Risk Monitor (real-time)
# ---------------------------------------------------------------------------

class RiskMonitor:
    """
    Real-time risk monitoring for live portfolio.

    Tracks risk metrics and triggers alerts when limits are approached
    or breached.

    Parameters
    ----------
    weights : Dict[str, float]
        Current portfolio weights.
    returns : pd.DataFrame
        Historical returns for risk estimation.
    var_limit : float
        Maximum VaR (95%, 1-day).
    max_drawdown : float
        Maximum drawdown limit.
    """

    def __init__(
        self,
        weights: Dict[str, float],
        returns: pd.DataFrame,
        var_limit: float = 0.03,
        max_drawdown: float = 0.20,
    ):
        self.analyzer = RiskAnalyzer(returns, weights)
        self.var_limit = var_limit
        self.max_drawdown = max_drawdown

    def check_limits(self) -> RiskLimitCheck:
        """Run full risk limit check."""
        return self.analyzer.check_limits(
            var_limit=self.var_limit,
            dd_limit=self.max_drawdown,
        )

    def current_var(self) -> float:
        """Current 1-day 95% VaR."""
        result = self.analyzer.calculate_var(0.95, 1, 'historical')
        return result.var_pct

    def current_drawdown(self) -> float:
        """Current drawdown from peak."""
        dd = self.analyzer.analyze_drawdowns()
        return dd.current_drawdown
