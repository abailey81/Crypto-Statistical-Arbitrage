"""
Performance Attribution Module
==============================

Decomposes strategy returns into contributing factors:
- Venue attribution (CEX vs DEX vs Hybrid)
- Sector attribution (L1, DeFi, Meme, etc.)
- Regime attribution (Bull, Bear, Sideways)
- Enhancement attribution (ML, Regime Detection, Dynamic Selection)
- Alpha vs Beta decomposition

Provides detailed factor-based performance analysis for
understanding sources of returns and risk.

Author: Tamer Atesyakar
Version: 2.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AttributionFactor(Enum):
    """Types of attribution factors."""
    VENUE = "venue"
    SECTOR = "sector"
    TIER = "tier"
    REGIME = "regime"
    ENHANCEMENT = "enhancement"
    PAIR = "pair"
    TIMING = "timing"


@dataclass
class FactorContribution:
    """Contribution of a single factor to performance."""
    factor_type: AttributionFactor
    factor_name: str
    total_return: float
    contribution_pct: float  # % of total strategy returns
    sharpe_ratio: float
    volatility: float
    num_observations: int
    weight: float = 0.0  # Average weight in portfolio

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'factor_type': self.factor_type.value,
            'factor_name': self.factor_name,
            'total_return': self.total_return,
            'contribution_pct': self.contribution_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'volatility': self.volatility,
            'num_observations': self.num_observations,
            'weight': self.weight
        }


@dataclass
class AttributionResult:
    """Complete attribution analysis results."""
    total_return: float
    total_volatility: float
    total_sharpe: float

    # Factor contributions
    venue_attribution: List[FactorContribution] = field(default_factory=list)
    sector_attribution: List[FactorContribution] = field(default_factory=list)
    tier_attribution: List[FactorContribution] = field(default_factory=list)
    regime_attribution: List[FactorContribution] = field(default_factory=list)
    enhancement_attribution: List[FactorContribution] = field(default_factory=list)

    # Decomposition
    explained_return: float = 0.0
    unexplained_return: float = 0.0
    r_squared: float = 0.0

    def get_summary_dict(self) -> Dict:
        """Get summary dictionary."""
        return {
            'total_return': self.total_return,
            'total_volatility': self.total_volatility,
            'total_sharpe': self.total_sharpe,
            'explained_return': self.explained_return,
            'unexplained_return': self.unexplained_return,
            'r_squared': self.r_squared,
            'num_venue_factors': len(self.venue_attribution),
            'num_sector_factors': len(self.sector_attribution),
            'num_regime_factors': len(self.regime_attribution),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all factor contributions to DataFrame."""
        all_contributions = (
            self.venue_attribution +
            self.sector_attribution +
            self.tier_attribution +
            self.regime_attribution +
            self.enhancement_attribution
        )

        if not all_contributions:
            return pd.DataFrame()

        data = [contrib.to_dict() for contrib in all_contributions]
        return pd.DataFrame(data)


class PerformanceAttributor:
    """
    Comprehensive performance attribution analyzer.

    Decomposes strategy returns into multiple factors to understand
    sources of performance. Uses Brinson attribution methodology
    adapted for crypto pairs trading.

    Key Features:
    - Multi-factor attribution (venue, sector, regime, etc.)
    - Factor interaction effects
    - Time-varying attribution
    - Risk-adjusted metrics

    Usage:
        attributor = PerformanceAttributor()
        result = attributor.attribute(
            backtest_results=results,
            enhanced_signals=signals,
            regime_states=regimes
        )
        df = result.to_dataframe()
    """

    def __init__(
        self,
        min_observations: int = 10,
        annualization_factor: int = 252
    ):
        """
        Initialize performance attributor.

        Args:
            min_observations: Minimum data points for valid attribution
            annualization_factor: Days per year for annualization
        """
        self.min_observations = min_observations
        self.annualization_factor = annualization_factor

        logger.info(f"PerformanceAttributor initialized")

    def attribute(
        self,
        backtest_results: pd.DataFrame,
        enhanced_signals: Optional[pd.DataFrame] = None,
        regime_states: Optional[pd.Series] = None,
        returns_col: str = 'returns',
        venue_col: str = 'venue_type',
        sector_col: str = 'sector',
        tier_col: str = 'tier'
    ) -> AttributionResult:
        """
        Perform comprehensive performance attribution.

        Args:
            backtest_results: DataFrame with backtest results
            enhanced_signals: Optional enhanced signals for enhancement attribution
            regime_states: Optional regime states for regime attribution
            returns_col: Column name for returns
            venue_col: Column name for venue type
            sector_col: Column name for sector
            tier_col: Column name for tier

        Returns:
            AttributionResult with factor contributions
        """
        # Calculate total metrics
        if returns_col not in backtest_results.columns:
            logger.warning(f"Column {returns_col} not found, using default returns")
            returns = pd.Series([0.0], index=backtest_results.index)
        else:
            returns = backtest_results[returns_col]

        total_return = returns.sum() if len(returns) > 0 else 0.0
        total_volatility = returns.std() * np.sqrt(self.annualization_factor) if len(returns) > 1 else 0.0
        total_sharpe = (total_return * self.annualization_factor / total_volatility) if total_volatility > 0 else 0.0

        # Venue attribution
        venue_attribution = self._attribute_by_factor(
            backtest_results,
            returns_col,
            venue_col,
            AttributionFactor.VENUE,
            total_return
        )

        # Sector attribution
        sector_attribution = self._attribute_by_factor(
            backtest_results,
            returns_col,
            sector_col,
            AttributionFactor.SECTOR,
            total_return
        )

        # Tier attribution
        tier_attribution = self._attribute_by_factor(
            backtest_results,
            returns_col,
            tier_col,
            AttributionFactor.TIER,
            total_return
        )

        # Regime attribution
        regime_attribution = []
        if regime_states is not None and len(regime_states) > 0:
            regime_attribution = self._attribute_by_regime(
                backtest_results,
                returns_col,
                regime_states,
                total_return
            )

        # Enhancement attribution
        enhancement_attribution = []
        if enhanced_signals is not None and len(enhanced_signals) > 0:
            enhancement_attribution = self._attribute_enhancements(
                backtest_results,
                enhanced_signals,
                returns_col,
                total_return
            )

        # Calculate explained vs unexplained
        explained_return = sum(c.total_return for c in venue_attribution + sector_attribution)
        unexplained_return = total_return - explained_return
        r_squared = 1.0 - (unexplained_return / total_return) if total_return != 0 else 0.0

        return AttributionResult(
            total_return=total_return,
            total_volatility=total_volatility,
            total_sharpe=total_sharpe,
            venue_attribution=venue_attribution,
            sector_attribution=sector_attribution,
            tier_attribution=tier_attribution,
            regime_attribution=regime_attribution,
            enhancement_attribution=enhancement_attribution,
            explained_return=explained_return,
            unexplained_return=unexplained_return,
            r_squared=r_squared
        )

    def _attribute_by_factor(
        self,
        data: pd.DataFrame,
        returns_col: str,
        factor_col: str,
        factor_type: AttributionFactor,
        total_return: float
    ) -> List[FactorContribution]:
        """Attribute returns by a single factor (venue, sector, tier)."""
        contributions = []

        if factor_col not in data.columns:
            logger.warning(f"Column {factor_col} not found, skipping {factor_type.value} attribution")
            return contributions

        # Group by factor and calculate metrics
        grouped = data.groupby(factor_col)[returns_col]

        for factor_name, group_returns in grouped:
            if len(group_returns) < self.min_observations:
                continue

            factor_return = group_returns.sum()
            contribution_pct = (factor_return / total_return * 100) if total_return != 0 else 0.0

            factor_vol = group_returns.std() * np.sqrt(self.annualization_factor)
            factor_sharpe = (factor_return * self.annualization_factor / factor_vol) if factor_vol > 0 else 0.0

            contributions.append(FactorContribution(
                factor_type=factor_type,
                factor_name=str(factor_name),
                total_return=factor_return,
                contribution_pct=contribution_pct,
                sharpe_ratio=factor_sharpe,
                volatility=factor_vol,
                num_observations=len(group_returns),
                weight=len(group_returns) / len(data)
            ))

        return contributions

    def _attribute_by_regime(
        self,
        data: pd.DataFrame,
        returns_col: str,
        regime_states: pd.Series,
        total_return: float
    ) -> List[FactorContribution]:
        """Attribute returns by market regime."""
        contributions = []

        # Align regime states with data
        aligned_regimes = regime_states.reindex(data.index, method='ffill')

        if aligned_regimes.isna().all():
            logger.warning("No valid regime states after alignment")
            return contributions

        # Group by regime
        regime_groups = {}
        for idx, regime in aligned_regimes.items():
            if pd.isna(regime):
                continue
            if regime not in regime_groups:
                regime_groups[regime] = []
            if idx in data.index and returns_col in data.columns:
                regime_groups[regime].append(data.loc[idx, returns_col])

        # Calculate metrics for each regime
        for regime_name, regime_returns in regime_groups.items():
            if len(regime_returns) < self.min_observations:
                continue

            regime_returns_series = pd.Series(regime_returns)
            regime_return = regime_returns_series.sum()
            contribution_pct = (regime_return / total_return * 100) if total_return != 0 else 0.0

            regime_vol = regime_returns_series.std() * np.sqrt(self.annualization_factor)
            regime_sharpe = (regime_return * self.annualization_factor / regime_vol) if regime_vol > 0 else 0.0

            contributions.append(FactorContribution(
                factor_type=AttributionFactor.REGIME,
                factor_name=str(regime_name),
                total_return=regime_return,
                contribution_pct=contribution_pct,
                sharpe_ratio=regime_sharpe,
                volatility=regime_vol,
                num_observations=len(regime_returns),
                weight=len(regime_returns) / len(data)
            ))

        return contributions

    def _attribute_enhancements(
        self,
        backtest_results: pd.DataFrame,
        enhanced_signals: pd.DataFrame,
        returns_col: str,
        total_return: float
    ) -> List[FactorContribution]:
        """Attribute returns to enhancements (ML, regime, dynamic selection)."""
        contributions = []

        # Check for enhancement indicators in signals
        enhancement_cols = {
            'ml_prediction': 'ML Enhancement',
            'regime': 'Regime Detection',
            'dynamic_pair': 'Dynamic Pair Selection'
        }

        for col, name in enhancement_cols.items():
            if col not in enhanced_signals.columns:
                continue

            # Simple attribution: compare periods with/without enhancement
            enhanced_mask = enhanced_signals[col] != 0

            if enhanced_mask.sum() < self.min_observations:
                continue

            # Align with backtest results
            aligned_mask = enhanced_mask.reindex(backtest_results.index, fill_value=False)

            enhanced_returns = backtest_results.loc[aligned_mask, returns_col]
            baseline_returns = backtest_results.loc[~aligned_mask, returns_col]

            if len(enhanced_returns) > 0 and len(baseline_returns) > 0:
                enhanced_return = enhanced_returns.sum()
                baseline_return = baseline_returns.sum()
                enhancement_effect = enhanced_return - baseline_return * (len(enhanced_returns) / len(baseline_returns))

                contribution_pct = (enhancement_effect / total_return * 100) if total_return != 0 else 0.0

                enhanced_vol = enhanced_returns.std() * np.sqrt(self.annualization_factor)
                enhanced_sharpe = (enhanced_return * self.annualization_factor / enhanced_vol) if enhanced_vol > 0 else 0.0

                contributions.append(FactorContribution(
                    factor_type=AttributionFactor.ENHANCEMENT,
                    factor_name=name,
                    total_return=enhancement_effect,
                    contribution_pct=contribution_pct,
                    sharpe_ratio=enhanced_sharpe,
                    volatility=enhanced_vol,
                    num_observations=len(enhanced_returns),
                    weight=len(enhanced_returns) / len(backtest_results)
                ))

        return contributions

    def create_waterfall_data(
        self,
        result: AttributionResult
    ) -> pd.DataFrame:
        """
        Create waterfall chart data for visualization.

        Args:
            result: Attribution result

        Returns:
            DataFrame suitable for waterfall chart
        """
        waterfall_data = []

        # Start with total
        waterfall_data.append({
            'factor': 'Total Return',
            'value': result.total_return,
            'cumulative': result.total_return
        })

        cumulative = 0.0

        # Add venue contributions
        for contrib in result.venue_attribution:
            cumulative += contrib.total_return
            waterfall_data.append({
                'factor': f"Venue: {contrib.factor_name}",
                'value': contrib.total_return,
                'cumulative': cumulative
            })

        # Add sector contributions
        for contrib in result.sector_attribution:
            cumulative += contrib.total_return
            waterfall_data.append({
                'factor': f"Sector: {contrib.factor_name}",
                'value': contrib.total_return,
                'cumulative': cumulative
            })

        # Add unexplained
        waterfall_data.append({
            'factor': 'Unexplained',
            'value': result.unexplained_return,
            'cumulative': result.total_return
        })

        return pd.DataFrame(waterfall_data)
