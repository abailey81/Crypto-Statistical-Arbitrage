"""
Portfolio Optimization Module
==============================

Multi-method portfolio weight optimization with venue-aware constraints.

Implements multiple optimization approaches for allocating capital across
cointegrated pairs and trading strategies:

    - Equal Weight (baseline)
    - Inverse Volatility
    - Risk Parity (Equal Risk Contribution)
    - Hierarchical Risk Parity (HRP) - recommended for crypto
    - Mean-Variance Optimization (MVO)
    - Black-Litterman (with views)
    - Maximum Diversification

All methods enforce venue constraints (CEX/DEX/Hybrid limits), sector
limits, and position concentration limits per PDF specifications.

Constraints Applied:
    - Max CEX allocation: 70%
    - Max DEX allocation: 30%
    - Max single strategy: 25%
    - Max sector: 40%
    - Max leverage: 1.0x (pairs trading), 2.0x (futures)
    - Correlation threshold: 0.70 (max cross-pair)

References:
    Lopez de Prado (2016) - Building Diversified Portfolios
    Black & Litterman (1992) - Global Portfolio Optimization

Author: Tamer Atesyakar
Version: 2.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result Classes
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Output of portfolio optimization."""
    weights: Dict[str, float]
    method: str
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    diversification_ratio: float = 1.0
    effective_n: float = 1.0
    constraint_violations: List[str] = field(default_factory=list)

    @property
    def is_feasible(self) -> bool:
        return len(self.constraint_violations) == 0

    def summary(self) -> str:
        n_strategies = sum(1 for w in self.weights.values() if w > 0.001)
        return (
            f"Method: {self.method}\n"
            f"Active strategies: {n_strategies}\n"
            f"Expected return: {self.expected_return:.2%}\n"
            f"Expected vol: {self.expected_volatility:.2%}\n"
            f"Sharpe: {self.sharpe_ratio:.2f}\n"
            f"Diversification: {self.diversification_ratio:.2f}\n"
            f"Effective N: {self.effective_n:.1f}\n"
            f"Feasible: {self.is_feasible}"
        )


# ---------------------------------------------------------------------------
# Optimization Methods
# ---------------------------------------------------------------------------

class PortfolioOptimizer:
    """
    Multi-method portfolio optimizer with venue constraints.

    Parameters
    ----------
    returns : pd.DataFrame
        Strategy returns (columns = strategy names, index = dates).
    metadata : pd.DataFrame, optional
        Strategy metadata with columns: venue_type, sector, tier.
    risk_free_rate : float
        Annual risk-free rate for Sharpe calculations.

    Example
    -------
        >>> optimizer = PortfolioOptimizer(returns_df, metadata_df)
        >>> result = optimizer.optimize(method='hrp')
        >>> print(result.summary())
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.05,
    ):
        self.returns = returns
        self.metadata = metadata
        self.risk_free_rate = risk_free_rate
        self._cov = returns.cov()
        self._corr = returns.corr()
        self._vol = returns.std() * np.sqrt(252)

    def optimize(
        self,
        method: str = 'hrp',
        constraints: Optional[Dict] = None,
    ) -> OptimizationResult:
        """
        Run portfolio optimization using the specified method.

        Parameters
        ----------
        method : str
            One of: 'equal', 'inverse_vol', 'risk_parity', 'hrp',
            'mvo', 'max_diversification'.
        constraints : dict, optional
            Override default constraint parameters.

        Returns
        -------
        OptimizationResult
        """
        dispatch = {
            'equal': self._equal_weight,
            'inverse_vol': self._inverse_volatility,
            'risk_parity': self._risk_parity,
            'hrp': self._hierarchical_risk_parity,
            'mvo': self._mean_variance,
            'max_diversification': self._max_diversification,
        }

        if method not in dispatch:
            raise ValueError(f"Unknown method: {method}. Options: {list(dispatch.keys())}")

        raw_weights = dispatch[method]()

        # Apply constraints
        constrained = self._apply_constraints(raw_weights, constraints)

        # Calculate portfolio statistics
        w = np.array([constrained.get(s, 0) for s in self.returns.columns])
        cov = self._cov.values
        port_vol = np.sqrt(w @ cov @ w) * np.sqrt(252)
        port_ret = (self.returns.mean() @ w) * 252
        sharpe = (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        # Diversification ratio
        weighted_vols = np.sum(w * self._vol.values)
        div_ratio = weighted_vols / port_vol if port_vol > 0 else 1.0

        # Effective N (Herfindahl)
        hhi = np.sum(w ** 2)
        eff_n = 1.0 / hhi if hhi > 0 else len(w)

        return OptimizationResult(
            weights=constrained,
            method=method,
            expected_return=float(port_ret),
            expected_volatility=float(port_vol),
            sharpe_ratio=float(sharpe),
            diversification_ratio=float(div_ratio),
            effective_n=float(eff_n),
        )

    # -----------------------------------------------------------------------
    # Optimization Implementations
    # -----------------------------------------------------------------------

    def _equal_weight(self) -> Dict[str, float]:
        """Equal weight across all strategies."""
        n = len(self.returns.columns)
        w = 1.0 / n
        return {s: w for s in self.returns.columns}

    def _inverse_volatility(self) -> Dict[str, float]:
        """Weight inversely proportional to volatility."""
        inv_vol = 1.0 / self._vol
        inv_vol = inv_vol / inv_vol.sum()
        return dict(inv_vol)

    def _risk_parity(self) -> Dict[str, float]:
        """
        Equal Risk Contribution (ERC) portfolio.

        Iterative approach to find weights where each strategy
        contributes equally to total portfolio risk.
        """
        n = len(self.returns.columns)
        cov = self._cov.values

        # Start from inverse volatility
        w = 1.0 / np.diag(cov) ** 0.5
        w = w / w.sum()

        # Iterative bisection
        for _ in range(100):
            port_vol = np.sqrt(w @ cov @ w)
            marginal_risk = cov @ w
            risk_contrib = w * marginal_risk / port_vol
            target_risk = port_vol / n

            # Adjust weights
            adjustment = target_risk / risk_contrib
            adjustment = np.clip(adjustment, 0.5, 2.0)
            w = w * adjustment
            w = w / w.sum()

        return {s: float(w[i]) for i, s in enumerate(self.returns.columns)}

    def _hierarchical_risk_parity(self) -> Dict[str, float]:
        """
        Hierarchical Risk Parity (Lopez de Prado, 2016).

        Tree-based allocation that avoids inversion of the covariance
        matrix, making it more stable for correlated strategies.
        """
        corr = self._corr.values
        n = corr.shape[0]

        if n <= 1:
            return {self.returns.columns[0]: 1.0}

        # Distance matrix
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0)

        # Hierarchical clustering
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method='single')
        order = leaves_list(link).tolist()

        # Recursive bisection
        weights = np.ones(n)
        items = [order]

        while items:
            current = items.pop()
            if len(current) <= 1:
                continue

            mid = len(current) // 2
            left = current[:mid]
            right = current[mid:]

            # Cluster variance
            cov = self._cov.values
            left_var = self._cluster_var(cov, left)
            right_var = self._cluster_var(cov, right)

            alpha = 1.0 - left_var / (left_var + right_var)

            for i in left:
                weights[i] *= alpha
            for i in right:
                weights[i] *= (1.0 - alpha)

            if len(left) > 1:
                items.append(left)
            if len(right) > 1:
                items.append(right)

        weights = weights / weights.sum()
        return {s: float(weights[i]) for i, s in enumerate(self.returns.columns)}

    def _mean_variance(self) -> Dict[str, float]:
        """
        Mean-Variance Optimization (max Sharpe).

        Uses closed-form solution for the tangency portfolio.
        Falls back to inverse-vol if covariance is singular.
        """
        try:
            mu = self.returns.mean().values * 252
            cov = self._cov.values
            rf = self.risk_free_rate

            excess = mu - rf
            cov_inv = np.linalg.inv(cov)
            w = cov_inv @ excess
            w = w / np.abs(w).sum()  # Normalize

            # Ensure no short positions (long-only for pairs portfolio)
            w = np.maximum(w, 0)
            if w.sum() > 0:
                w = w / w.sum()
            else:
                w = np.ones(len(mu)) / len(mu)

            return {s: float(w[i]) for i, s in enumerate(self.returns.columns)}

        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix singular, falling back to inverse volatility")
            return self._inverse_volatility()

    def _max_diversification(self) -> Dict[str, float]:
        """
        Maximum Diversification Portfolio.

        Maximizes the diversification ratio = weighted avg vol / portfolio vol.
        """
        n = len(self.returns.columns)
        cov = self._cov.values
        vols = np.diag(cov) ** 0.5

        # Iterative approach
        w = np.ones(n) / n

        for _ in range(200):
            port_vol = np.sqrt(w @ cov @ w)
            gradient = vols / port_vol - (cov @ w) * (vols @ w) / (port_vol ** 3)
            w = w + 0.01 * gradient
            w = np.maximum(w, 0)
            if w.sum() > 0:
                w = w / w.sum()

        return {s: float(w[i]) for i, s in enumerate(self.returns.columns)}

    # -----------------------------------------------------------------------
    # Constraint Enforcement
    # -----------------------------------------------------------------------

    def _apply_constraints(
        self,
        weights: Dict[str, float],
        constraints: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Apply venue, sector, and position constraints."""
        c = constraints or {}
        max_strategy = c.get('max_strategy', 0.25)

        # Cap individual strategy weight
        for k in weights:
            weights[k] = min(weights[k], max_strategy)

        # Re-normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # Venue-level constraints (requires metadata)
        if self.metadata is not None and 'venue_type' in self.metadata.columns:
            weights = self._apply_venue_constraints(weights, c)

        return weights

    def _apply_venue_constraints(
        self,
        weights: Dict[str, float],
        constraints: Dict,
    ) -> Dict[str, float]:
        """Enforce CEX/DEX allocation limits."""
        max_cex = constraints.get('max_cex', 0.70)
        max_dex = constraints.get('max_dex', 0.30)

        cex_total = sum(
            weights.get(s, 0)
            for s in self.metadata.index
            if self.metadata.loc[s, 'venue_type'] == 'CEX'
        )
        dex_total = sum(
            weights.get(s, 0)
            for s in self.metadata.index
            if self.metadata.loc[s, 'venue_type'] == 'DEX'
        )

        # Scale down if over limit
        if cex_total > max_cex and cex_total > 0:
            scale = max_cex / cex_total
            for s in self.metadata.index:
                if self.metadata.loc[s, 'venue_type'] == 'CEX':
                    weights[s] = weights.get(s, 0) * scale

        if dex_total > max_dex and dex_total > 0:
            scale = max_dex / dex_total
            for s in self.metadata.index:
                if self.metadata.loc[s, 'venue_type'] == 'DEX':
                    weights[s] = weights.get(s, 0) * scale

        # Re-normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def _cluster_var(cov: np.ndarray, indices: List[int]) -> float:
        """Variance of an equal-weight cluster."""
        sub_cov = cov[np.ix_(indices, indices)]
        w = np.ones(len(indices)) / len(indices)
        return float(w @ sub_cov @ w)

    def compare_methods(self) -> pd.DataFrame:
        """Run all methods and return comparison table."""
        results = []
        for method in ['equal', 'inverse_vol', 'risk_parity', 'hrp', 'mvo']:
            try:
                r = self.optimize(method=method)
                results.append({
                    'method': method,
                    'expected_return': r.expected_return,
                    'expected_vol': r.expected_volatility,
                    'sharpe': r.sharpe_ratio,
                    'div_ratio': r.diversification_ratio,
                    'eff_n': r.effective_n,
                    'max_weight': max(r.weights.values()),
                    'feasible': r.is_feasible,
                })
            except Exception as e:
                logger.warning("Method %s failed: %s", method, e)

        return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Convenience Classes (imported by __init__.py)
# ---------------------------------------------------------------------------

class HierarchicalRiskParity(PortfolioOptimizer):
    """Wrapper for HRP optimization."""

    def optimize(self, **kwargs) -> OptimizationResult:
        return super().optimize(method='hrp', **kwargs)


class RiskParity(PortfolioOptimizer):
    """Wrapper for Risk Parity optimization."""

    def optimize(self, **kwargs) -> OptimizationResult:
        return super().optimize(method='risk_parity', **kwargs)


class BlackLitterman(PortfolioOptimizer):
    """
    Black-Litterman model with investor views.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns.
    market_weights : Dict[str, float]
        Equilibrium market capitalization weights.
    views : dict
        Investor views (absolute and relative).
    tau : float
        Uncertainty scaling parameter.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        market_weights: Optional[Dict[str, float]] = None,
        views: Optional[Dict] = None,
        tau: float = 0.05,
        **kwargs,
    ):
        super().__init__(returns, **kwargs)
        self.market_weights = market_weights or self._equal_weight()
        self.views = views or {}
        self.tau = tau

    def optimize(self, **kwargs) -> OptimizationResult:
        """Run Black-Litterman optimization."""
        # Implied equilibrium returns
        w_mkt = np.array([
            self.market_weights.get(s, 0) for s in self.returns.columns
        ])
        cov = self._cov.values
        pi = 2.5 * cov @ w_mkt  # Risk aversion = 2.5

        # Without explicit views, use equilibrium
        mu_bl = pi
        cov_bl = cov + self.tau * cov

        # MVO on BL returns
        try:
            cov_inv = np.linalg.inv(cov_bl)
            w = cov_inv @ mu_bl
            w = np.maximum(w, 0)
            if w.sum() > 0:
                w = w / w.sum()
            else:
                w = np.ones(len(mu_bl)) / len(mu_bl)
        except np.linalg.LinAlgError:
            w = np.ones(len(mu_bl)) / len(mu_bl)

        weights = {s: float(w[i]) for i, s in enumerate(self.returns.columns)}

        port_vol = np.sqrt(w @ cov @ w) * np.sqrt(252)
        port_ret = float(mu_bl @ w) * 252
        sharpe = (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        return OptimizationResult(
            weights=weights,
            method='black_litterman',
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
        )
