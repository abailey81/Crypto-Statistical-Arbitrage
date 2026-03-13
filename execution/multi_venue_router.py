"""
Multi-Venue Router
===================

Unified order routing across CEX, DEX, and hybrid venues.

Routes orders to the optimal venue based on cost, liquidity, and
execution quality. Handles venue-tiered allocation per PDF requirements.

Venue Tiers (PDF Section 2.1):
    - Tier 1: Both tokens on major CEX (e.g. AAVE-COMP on Binance)
    - Tier 2: Mixed CEX/DEX (e.g. GMX on CEX, GNS on DEX)
    - Tier 3: Both DEX-only (smaller protocols, emerging L2s)

Position Limits:
    - CEX pairs: up to $100k per pair
    - DEX pairs (liquid): $20-50k
    - DEX pairs (illiquid): $5-10k

Author: Tamer Atesyakar
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class VenueTier(Enum):
    TIER_1 = 1   # Both CEX
    TIER_2 = 2   # Mixed CEX/DEX
    TIER_3 = 3   # Both DEX


class ExecutionVenue(Enum):
    CEX = 'CEX'
    DEX = 'DEX'
    HYBRID = 'HYBRID'


@dataclass
class VenueCapacity:
    """Capacity constraints for a venue."""
    venue: ExecutionVenue
    max_position_usd: float
    max_daily_volume_pct: float  # Max % of venue daily volume
    current_utilization: float = 0.0

    @property
    def remaining_capacity_usd(self) -> float:
        return max(0, self.max_position_usd - self.current_utilization)


@dataclass
class RoutingDecision:
    """Result of venue routing for a pair trade."""
    pair_id: str
    long_token: str
    short_token: str
    tier: VenueTier
    long_venue: ExecutionVenue
    short_venue: ExecutionVenue
    position_size_usd: float
    estimated_cost_bps: float
    requires_bridge: bool = False
    routing_reason: str = ''


# ---------------------------------------------------------------------------
# Tier-specific Position Limits
# ---------------------------------------------------------------------------

TIER_POSITION_LIMITS: Dict[VenueTier, Dict[str, float]] = {
    VenueTier.TIER_1: {
        'max_position_usd': 100_000,
        'max_simultaneous': 8,
        'concentration_limit': 0.60,  # Max 60% in CEX-only
    },
    VenueTier.TIER_2: {
        'max_position_usd': 50_000,
        'max_simultaneous': 3,
        'concentration_limit': 0.40,
    },
    VenueTier.TIER_3: {
        'max_position_usd': 10_000,
        'max_simultaneous': 2,
        'concentration_limit': 0.20,  # Max 20% in Tier 3
    },
}


# ---------------------------------------------------------------------------
# Multi-Venue Router
# ---------------------------------------------------------------------------

class MultiVenueRouter:
    """
    Unified order router across CEX, DEX, and hybrid venues.

    Determines optimal execution venue for each leg of a pairs trade
    based on token availability, liquidity, costs, and tier constraints.

    Parameters
    ----------
    cex_tokens : set
        Tokens available on centralized exchanges.
    dex_tokens : set
        Tokens available on decentralized exchanges.
    max_cex_allocation : float
        Maximum portfolio allocation to CEX pairs (PDF: 60%).
    max_tier3_allocation : float
        Maximum allocation to Tier 3 DEX-only pairs (PDF: 20%).

    Example
    -------
        >>> router = MultiVenueRouter(
        ...     cex_tokens={'UNI', 'SUSHI', 'AAVE', 'COMP'},
        ...     dex_tokens={'UNI', 'SUSHI', 'RDNT', 'PENDLE'}
        ... )
        >>> decision = router.route_pair('UNI', 'SUSHI', 50_000)
        >>> print(f"Tier: {decision.tier.name}, Cost: {decision.estimated_cost_bps:.1f} bps")
    """

    def __init__(
        self,
        cex_tokens: Optional[set] = None,
        dex_tokens: Optional[set] = None,
        max_cex_allocation: float = 0.60,
        max_tier3_allocation: float = 0.20,
        max_sector_allocation: float = 0.40,
    ):
        self.cex_tokens = cex_tokens or set()
        self.dex_tokens = dex_tokens or set()
        self.max_cex_allocation = max_cex_allocation
        self.max_tier3_allocation = max_tier3_allocation
        self.max_sector_allocation = max_sector_allocation
        self._active_positions: List[RoutingDecision] = []

    def classify_tier(self, token_a: str, token_b: str) -> VenueTier:
        """
        Classify a pair into venue tier based on token availability.

        Tier 1: Both on CEX
        Tier 2: One CEX + one DEX, or both available on both
        Tier 3: Both DEX-only
        """
        a_on_cex = token_a in self.cex_tokens
        b_on_cex = token_b in self.cex_tokens
        a_on_dex = token_a in self.dex_tokens
        b_on_dex = token_b in self.dex_tokens

        if a_on_cex and b_on_cex:
            return VenueTier.TIER_1
        elif (a_on_cex or b_on_cex) and (a_on_dex or b_on_dex):
            return VenueTier.TIER_2
        elif a_on_dex and b_on_dex:
            return VenueTier.TIER_3
        else:
            logger.warning(
                "Pair %s-%s not available on any tracked venue", token_a, token_b
            )
            return VenueTier.TIER_3

    def route_pair(
        self,
        long_token: str,
        short_token: str,
        requested_size_usd: float,
    ) -> RoutingDecision:
        """
        Determine routing for a pairs trade.

        Returns venue assignment, position sizing, and cost estimates.
        """
        tier = self.classify_tier(long_token, short_token)
        limits = TIER_POSITION_LIMITS[tier]

        # Cap position to tier limit
        size_usd = min(requested_size_usd, limits['max_position_usd'])

        # Determine execution venue for each leg
        if tier == VenueTier.TIER_1:
            long_venue = ExecutionVenue.CEX
            short_venue = ExecutionVenue.CEX
            cost_bps = 20.0  # 0.20% total (4 legs CEX)
            requires_bridge = False
            reason = 'Both tokens on CEX; standard CEX execution'

        elif tier == VenueTier.TIER_2:
            long_on_cex = long_token in self.cex_tokens
            short_on_cex = short_token in self.cex_tokens

            long_venue = ExecutionVenue.CEX if long_on_cex else ExecutionVenue.DEX
            short_venue = ExecutionVenue.CEX if short_on_cex else ExecutionVenue.DEX
            cost_bps = 85.0  # Mixed: ~0.20% CEX leg + ~0.65% DEX leg
            requires_bridge = True
            reason = 'Mixed venue pair; bridge may be required'

        else:  # TIER_3
            long_venue = ExecutionVenue.DEX
            short_venue = ExecutionVenue.DEX
            cost_bps = 130.0  # ~1.30% total (both DEX)
            requires_bridge = False
            reason = 'Both tokens DEX-only; on-chain execution'

        decision = RoutingDecision(
            pair_id=f"{long_token}_{short_token}",
            long_token=long_token,
            short_token=short_token,
            tier=tier,
            long_venue=long_venue,
            short_venue=short_venue,
            position_size_usd=size_usd,
            estimated_cost_bps=cost_bps,
            requires_bridge=requires_bridge,
            routing_reason=reason,
        )

        return decision

    def check_concentration_limits(self) -> Dict[str, bool]:
        """
        Check if current positions comply with concentration limits.

        Returns dict of limit name -> whether compliant.
        """
        if not self._active_positions:
            return {'cex_limit': True, 'tier3_limit': True, 'sector_limit': True}

        total_usd = sum(p.position_size_usd for p in self._active_positions)
        if total_usd == 0:
            return {'cex_limit': True, 'tier3_limit': True, 'sector_limit': True}

        cex_usd = sum(
            p.position_size_usd for p in self._active_positions
            if p.tier == VenueTier.TIER_1
        )
        tier3_usd = sum(
            p.position_size_usd for p in self._active_positions
            if p.tier == VenueTier.TIER_3
        )

        return {
            'cex_limit': (cex_usd / total_usd) <= self.max_cex_allocation,
            'tier3_limit': (tier3_usd / total_usd) <= self.max_tier3_allocation,
            'sector_limit': True,  # Requires sector metadata
        }

    def get_capacity_summary(self) -> Dict[str, int]:
        """Summary of capacity utilization by tier."""
        counts = {tier: 0 for tier in VenueTier}
        for pos in self._active_positions:
            counts[pos.tier] += 1

        return {
            'tier1_active': counts[VenueTier.TIER_1],
            'tier1_max': int(TIER_POSITION_LIMITS[VenueTier.TIER_1]['max_simultaneous']),
            'tier2_active': counts[VenueTier.TIER_2],
            'tier2_max': int(TIER_POSITION_LIMITS[VenueTier.TIER_2]['max_simultaneous']),
            'tier3_active': counts[VenueTier.TIER_3],
            'tier3_max': int(TIER_POSITION_LIMITS[VenueTier.TIER_3]['max_simultaneous']),
            'total_active': len(self._active_positions),
            'total_max': 10,
        }
