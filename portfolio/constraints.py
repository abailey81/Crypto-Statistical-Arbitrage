"""
Portfolio Constraints Module
============================

Applies portfolio-level constraints to positions including:
- Sector allocation limits
- Venue type limits (CEX, DEX, Hybrid)
- Tier allocation limits
- Gross/net exposure limits
- Position concentration limits

Ensures strategy adheres to risk management rules.

Author: Tamer Atesyakar
Version: 2.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConstraintViolation:
    """Record of a constraint violation."""
    constraint_type: str
    current_value: float
    limit_value: float
    adjustment_made: float
    affected_positions: int


class PortfolioConstraintEnforcer:
    """
    Enforces portfolio-level constraints on positions.

    Applies constraints in order of priority:
    1. Gross exposure limits
    2. Sector allocation limits
    3. Venue type limits
    4. Tier allocation limits
    5. Position concentration limits

    Usage:
        enforcer = PortfolioConstraintEnforcer(
            max_sector_allocation=0.40,
            max_cex_allocation=0.60,
            max_tier3_allocation=0.20
        )
        adjusted_positions = enforcer.apply_constraints(
            positions=positions,
            universe_snapshot=universe
        )
    """

    def __init__(
        self,
        # Gross exposure
        max_gross_exposure: float = 2.0,
        max_net_exposure: float = 0.30,

        # Sector limits
        max_sector_allocation: float = 0.40,
        max_sector_count: int = 5,

        # Venue limits
        max_cex_allocation: float = 0.60,
        max_dex_allocation: float = 0.30,
        max_hybrid_allocation: float = 0.30,

        # Tier limits
        max_tier1_allocation: float = 0.70,
        max_tier2_allocation: float = 0.25,
        max_tier3_allocation: float = 0.20,

        # Position limits (PDF: 8-10 total max)
        max_positions: int = 10,
        max_position_pct: float = 0.10,  # Max 10% in single position

        # Minimum position size
        min_position_usd: float = 1000.0,

        # Correlation filter
        max_pair_correlation: float = 0.70,  # Max correlation between pairs in portfolio
    ):
        """
        Initialize constraint enforcer.

        Args:
            max_gross_exposure: Maximum gross exposure (sum of long + short)
            max_net_exposure: Maximum net exposure (long - short)
            max_sector_allocation: Maximum allocation to single sector
            max_sector_count: Maximum number of sectors
            max_cex_allocation: Maximum allocation to CEX pairs
            max_dex_allocation: Maximum allocation to DEX pairs
            max_hybrid_allocation: Maximum allocation to Hybrid pairs
            max_tier1_allocation: Maximum allocation to Tier 1 pairs
            max_tier2_allocation: Maximum allocation to Tier 2 pairs
            max_tier3_allocation: Maximum allocation to Tier 3 pairs
            max_positions: Maximum number of positions
            max_position_pct: Maximum single position size as % of portfolio
            min_position_usd: Minimum position size in USD
            max_pair_correlation: Maximum correlation between pairs in portfolio
        """
        self.max_gross_exposure = max_gross_exposure
        self.max_net_exposure = max_net_exposure
        self.max_sector_allocation = max_sector_allocation
        self.max_sector_count = max_sector_count
        self.max_cex_allocation = max_cex_allocation
        self.max_dex_allocation = max_dex_allocation
        self.max_hybrid_allocation = max_hybrid_allocation
        self.max_tier1_allocation = max_tier1_allocation
        self.max_tier2_allocation = max_tier2_allocation
        self.max_tier3_allocation = max_tier3_allocation
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct
        self.min_position_usd = min_position_usd
        self.max_pair_correlation = max_pair_correlation

        self.violations: List[ConstraintViolation] = []

        logger.info(f"PortfolioConstraintEnforcer initialized")

    def apply_constraints(
        self,
        positions: pd.DataFrame,
        universe_snapshot: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Apply all portfolio constraints to positions.

        Args:
            positions: DataFrame with position details (notional_usd, sector, venue_type, tier)
            universe_snapshot: Optional universe snapshot for additional metadata

        Returns:
            DataFrame with adjusted positions
        """
        if len(positions) == 0:
            logger.warning("No positions to constrain")
            return positions

        # Reset violations
        self.violations = []

        # Make a copy to avoid modifying original
        adjusted = positions.copy()

        # Ensure required columns exist
        required_cols = ['notional_usd']
        missing_cols = [col for col in required_cols if col not in adjusted.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return adjusted

        # 1. Remove positions below minimum size
        initial_count = len(adjusted)
        adjusted = adjusted[adjusted['notional_usd'] >= self.min_position_usd]
        if len(adjusted) < initial_count:
            logger.info(f"Removed {initial_count - len(adjusted)} positions below ${self.min_position_usd:,.0f}")

        # 2. Limit total number of positions
        if len(adjusted) > self.max_positions:
            # Keep largest positions
            adjusted = adjusted.nlargest(self.max_positions, 'notional_usd')
            logger.info(f"Limited to {self.max_positions} largest positions")

        # 3. Apply gross exposure limit
        adjusted = self._apply_gross_exposure_limit(adjusted)

        # 4. Apply sector allocation limits
        if 'sector' in adjusted.columns:
            adjusted = self._apply_sector_limits(adjusted)

        # 5. Apply venue type limits
        if 'venue_type' in adjusted.columns:
            adjusted = self._apply_venue_limits(adjusted)

        # 6. Apply tier limits
        if 'tier' in adjusted.columns:
            adjusted = self._apply_tier_limits(adjusted)

        # 7. Apply single position concentration limit
        adjusted = self._apply_position_concentration_limit(adjusted)

        # 8. Apply correlation filter (remove highly correlated pairs)
        if 'pair_name' in adjusted.columns or 'symbol_a' in adjusted.columns:
            adjusted = self._apply_correlation_filter(adjusted, universe_snapshot)

        logger.info(f"Constraints applied: {len(positions)} -> {len(adjusted)} positions")
        logger.info(f"Violations recorded: {len(self.violations)}")

        return adjusted

    def _apply_gross_exposure_limit(self, positions: pd.DataFrame) -> pd.DataFrame:
        """Apply gross exposure limit."""
        total_exposure = positions['notional_usd'].sum()

        if total_exposure == 0:
            return positions

        # Assuming $1M capital base (would be parameterized in production)
        capital_base = 1_000_000
        current_gross = total_exposure / capital_base

        if current_gross > self.max_gross_exposure:
            # Scale down all positions proportionally
            scale_factor = self.max_gross_exposure / current_gross
            positions = positions.copy()
            positions['notional_usd'] *= scale_factor

            self.violations.append(ConstraintViolation(
                constraint_type='gross_exposure',
                current_value=current_gross,
                limit_value=self.max_gross_exposure,
                adjustment_made=1.0 - scale_factor,
                affected_positions=len(positions)
            ))

            logger.info(f"Scaled down gross exposure from {current_gross:.2f}x to {self.max_gross_exposure:.2f}x")

        return positions

    def _apply_sector_limits(self, positions: pd.DataFrame) -> pd.DataFrame:
        """Apply sector allocation limits."""
        total_notional = positions['notional_usd'].sum()

        if total_notional == 0:
            return positions

        # Calculate sector exposures
        sector_exposure = positions.groupby('sector')['notional_usd'].sum()

        # Check for violations
        for sector, exposure in sector_exposure.items():
            sector_pct = exposure / total_notional

            if sector_pct > self.max_sector_allocation:
                # Scale down positions in this sector
                scale_factor = self.max_sector_allocation / sector_pct
                mask = positions['sector'] == sector

                original_total = positions.loc[mask, 'notional_usd'].sum()
                positions.loc[mask, 'notional_usd'] *= scale_factor
                new_total = positions.loc[mask, 'notional_usd'].sum()

                self.violations.append(ConstraintViolation(
                    constraint_type=f'sector_{sector}',
                    current_value=sector_pct,
                    limit_value=self.max_sector_allocation,
                    adjustment_made=original_total - new_total,
                    affected_positions=mask.sum()
                ))

                logger.info(f"Scaled down {sector} from {sector_pct:.1%} to {self.max_sector_allocation:.1%}")

        return positions

    def _apply_venue_limits(self, positions: pd.DataFrame) -> pd.DataFrame:
        """Apply venue type allocation limits."""
        total_notional = positions['notional_usd'].sum()

        if total_notional == 0:
            return positions

        # Calculate venue exposures
        venue_limits = {
            'CEX': self.max_cex_allocation,
            'DEX': self.max_dex_allocation,
            'HYBRID': self.max_hybrid_allocation
        }

        for venue_type, max_allocation in venue_limits.items():
            mask = positions['venue_type'] == venue_type
            venue_exposure = positions.loc[mask, 'notional_usd'].sum()
            venue_pct = venue_exposure / total_notional

            if venue_pct > max_allocation:
                # Scale down positions in this venue
                scale_factor = max_allocation / venue_pct

                original_total = positions.loc[mask, 'notional_usd'].sum()
                positions.loc[mask, 'notional_usd'] *= scale_factor
                new_total = positions.loc[mask, 'notional_usd'].sum()

                self.violations.append(ConstraintViolation(
                    constraint_type=f'venue_{venue_type}',
                    current_value=venue_pct,
                    limit_value=max_allocation,
                    adjustment_made=original_total - new_total,
                    affected_positions=mask.sum()
                ))

                logger.info(f"Scaled down {venue_type} from {venue_pct:.1%} to {max_allocation:.1%}")

        return positions

    def _apply_tier_limits(self, positions: pd.DataFrame) -> pd.DataFrame:
        """Apply tier allocation limits."""
        total_notional = positions['notional_usd'].sum()

        if total_notional == 0:
            return positions

        # Calculate tier exposures
        tier_limits = {
            'TIER_1': self.max_tier1_allocation,
            'TIER_2': self.max_tier2_allocation,
            'TIER_3': self.max_tier3_allocation
        }

        for tier, max_allocation in tier_limits.items():
            mask = positions['tier'] == tier
            tier_exposure = positions.loc[mask, 'notional_usd'].sum()
            tier_pct = tier_exposure / total_notional

            if tier_pct > max_allocation:
                # Scale down positions in this tier
                scale_factor = max_allocation / tier_pct

                original_total = positions.loc[mask, 'notional_usd'].sum()
                positions.loc[mask, 'notional_usd'] *= scale_factor
                new_total = positions.loc[mask, 'notional_usd'].sum()

                self.violations.append(ConstraintViolation(
                    constraint_type=f'tier_{tier}',
                    current_value=tier_pct,
                    limit_value=max_allocation,
                    adjustment_made=original_total - new_total,
                    affected_positions=mask.sum()
                ))

                logger.info(f"Scaled down {tier} from {tier_pct:.1%} to {max_allocation:.1%}")

        return positions

    def _apply_position_concentration_limit(self, positions: pd.DataFrame) -> pd.DataFrame:
        """Apply single position concentration limit."""
        total_notional = positions['notional_usd'].sum()

        if total_notional == 0:
            return positions

        # Cap individual positions
        max_position_size = total_notional * self.max_position_pct
        oversized_mask = positions['notional_usd'] > max_position_size

        if oversized_mask.any():
            positions.loc[oversized_mask, 'notional_usd'] = max_position_size

            self.violations.append(ConstraintViolation(
                constraint_type='position_concentration',
                current_value=positions['notional_usd'].max() / total_notional,
                limit_value=self.max_position_pct,
                adjustment_made=0.0,  # Can't easily quantify
                affected_positions=oversized_mask.sum()
            ))

            logger.info(f"Capped {oversized_mask.sum()} positions to {self.max_position_pct:.1%}")

        return positions

    def _apply_correlation_filter(
        self,
        positions: pd.DataFrame,
        universe_snapshot: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Remove pairs with correlation above threshold.

        For pairs with correlation > max_pair_correlation, keep the one
        with better characteristics (higher Sharpe, lower costs, etc.)
        """
        if len(positions) < 2:
            return positions

        # Try to get price data from universe_snapshot for correlation calculation
        # If not available, skip correlation filtering
        if universe_snapshot is None:
            logger.warning("No universe_snapshot provided, skipping correlation filter")
            return positions

        # Extract pair names
        if 'pair_name' in positions.columns:
            pair_names = positions['pair_name'].unique()
        elif 'symbol_a' in positions.columns and 'symbol_b' in positions.columns:
            pair_names = positions.apply(
                lambda row: f"{row['symbol_a']}/{row['symbol_b']}", axis=1
            ).unique()
        else:
            logger.warning("Cannot identify pairs for correlation filtering")
            return positions

        if len(pair_names) < 2:
            return positions

        # Track pairs to remove
        pairs_to_remove = set()

        # Simple heuristic: check if pairs share common symbols
        # Pairs sharing both symbols would have correlation ~1.0
        # Pairs sharing one symbol would have high correlation
        for i, pair1 in enumerate(pair_names):
            if pair1 in pairs_to_remove:
                continue

            symbols1 = set(pair1.split('/'))

            for pair2 in pair_names[i+1:]:
                if pair2 in pairs_to_remove:
                    continue

                symbols2 = set(pair2.split('/'))
                shared = symbols1.intersection(symbols2)

                # If pairs share symbols, they're likely highly correlated
                if len(shared) >= 1:
                    # Remove the pair with smaller position size
                    if 'pair_name' in positions.columns:
                        size1 = positions[positions['pair_name'] == pair1]['notional_usd'].sum()
                        size2 = positions[positions['pair_name'] == pair2]['notional_usd'].sum()
                    else:
                        # Fallback: remove the second one
                        size1 = 1.0
                        size2 = 0.5

                    if size1 >= size2:
                        pairs_to_remove.add(pair2)
                        logger.info(f"Correlation filter: removing {pair2} (correlated with {pair1})")
                    else:
                        pairs_to_remove.add(pair1)
                        logger.info(f"Correlation filter: removing {pair1} (correlated with {pair2})")

                    self.violations.append(ConstraintViolation(
                        constraint_type='pair_correlation',
                        current_value=1.0,  # Estimated
                        limit_value=self.max_pair_correlation,
                        adjustment_made=1.0,  # Removed 1 pair
                        affected_positions=1
                    ))

        # Remove flagged pairs
        if pairs_to_remove:
            initial_count = len(positions)
            if 'pair_name' in positions.columns:
                positions = positions[~positions['pair_name'].isin(pairs_to_remove)]
            else:
                # Remove by reconstructed pair name
                positions = positions[
                    ~positions.apply(
                        lambda row: f"{row['symbol_a']}/{row['symbol_b']}" in pairs_to_remove,
                        axis=1
                    )
                ]
            logger.info(f"Removed {initial_count - len(positions)} positions due to correlation filter")

        return positions

    def get_violations_summary(self) -> pd.DataFrame:
        """Get summary of all constraint violations."""
        if not self.violations:
            return pd.DataFrame()

        data = []
        for v in self.violations:
            data.append({
                'constraint': v.constraint_type,
                'current': v.current_value,
                'limit': v.limit_value,
                'adjustment': v.adjustment_made,
                'affected_positions': v.affected_positions
            })

        return pd.DataFrame(data)
