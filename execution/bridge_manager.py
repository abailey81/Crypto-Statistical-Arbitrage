"""
Bridge Manager
===============

Cross-chain asset transfer management for multi-venue trading strategies.

Handles bridging assets between Ethereum L1, Arbitrum, Optimism, Polygon,
and other L2s to enable cross-venue pairs trading and funding arbitrage.

Key Considerations:
    - Bridge latency varies: L1->L2 (~10 min), L2->L1 (~7 days for optimistic)
    - Costs: Native bridge (cheap, slow) vs third-party (faster, ~0.05-0.20%)
    - Capital efficiency: Assets locked during transit reduce available capital
    - Failure modes: Bridge exploits, congestion delays, stuck transactions

Supported Bridges:
    - Arbitrum native bridge (canonical, slow L2->L1)
    - Optimism native bridge (canonical)
    - Across Protocol (fast bridge, ~0.05% fee)
    - Stargate (LayerZero, multi-chain)
    - Hop Protocol (L2-to-L2 direct)

Author: Tamer Atesyakar
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class BridgeProtocol(Enum):
    NATIVE = 'native'
    ACROSS = 'across'
    STARGATE = 'stargate'
    HOP = 'hop'
    CELER = 'celer'


class BridgeStatus(Enum):
    PENDING = 'pending'
    CONFIRMING = 'confirming'
    COMPLETED = 'completed'
    FAILED = 'failed'
    STUCK = 'stuck'


class ChainID(Enum):
    ETHEREUM = 1
    ARBITRUM = 42161
    OPTIMISM = 10
    POLYGON = 137
    BASE = 8453


@dataclass
class BridgeRoute:
    """Specification for a cross-chain transfer route."""
    source_chain: ChainID
    dest_chain: ChainID
    protocol: BridgeProtocol
    estimated_time_minutes: float
    fee_bps: float
    min_amount_usd: float
    max_amount_usd: float


@dataclass
class BridgeTransfer:
    """Record of a bridge transfer."""
    transfer_id: str
    source_chain: ChainID
    dest_chain: ChainID
    protocol: BridgeProtocol
    token: str
    amount_usd: float
    fee_usd: float
    status: BridgeStatus
    initiated_at: Optional[pd.Timestamp] = None
    completed_at: Optional[pd.Timestamp] = None
    source_tx_hash: Optional[str] = None
    dest_tx_hash: Optional[str] = None

    @property
    def elapsed_minutes(self) -> float:
        if self.initiated_at and self.completed_at:
            return (self.completed_at - self.initiated_at).total_seconds() / 60
        return 0.0


# ---------------------------------------------------------------------------
# Route Database
# ---------------------------------------------------------------------------

BRIDGE_ROUTES: List[BridgeRoute] = [
    # Ethereum <-> Arbitrum
    BridgeRoute(ChainID.ETHEREUM, ChainID.ARBITRUM, BridgeProtocol.NATIVE,
                estimated_time_minutes=10, fee_bps=0, min_amount_usd=100, max_amount_usd=10_000_000),
    BridgeRoute(ChainID.ETHEREUM, ChainID.ARBITRUM, BridgeProtocol.ACROSS,
                estimated_time_minutes=2, fee_bps=5, min_amount_usd=100, max_amount_usd=1_000_000),
    BridgeRoute(ChainID.ARBITRUM, ChainID.ETHEREUM, BridgeProtocol.NATIVE,
                estimated_time_minutes=10080, fee_bps=0, min_amount_usd=100, max_amount_usd=10_000_000),
    BridgeRoute(ChainID.ARBITRUM, ChainID.ETHEREUM, BridgeProtocol.ACROSS,
                estimated_time_minutes=2, fee_bps=8, min_amount_usd=100, max_amount_usd=1_000_000),

    # Ethereum <-> Optimism
    BridgeRoute(ChainID.ETHEREUM, ChainID.OPTIMISM, BridgeProtocol.NATIVE,
                estimated_time_minutes=15, fee_bps=0, min_amount_usd=100, max_amount_usd=10_000_000),
    BridgeRoute(ChainID.ETHEREUM, ChainID.OPTIMISM, BridgeProtocol.HOP,
                estimated_time_minutes=5, fee_bps=6, min_amount_usd=50, max_amount_usd=500_000),

    # L2 <-> L2 (fast bridges only)
    BridgeRoute(ChainID.ARBITRUM, ChainID.OPTIMISM, BridgeProtocol.HOP,
                estimated_time_minutes=5, fee_bps=8, min_amount_usd=100, max_amount_usd=500_000),
    BridgeRoute(ChainID.ARBITRUM, ChainID.OPTIMISM, BridgeProtocol.STARGATE,
                estimated_time_minutes=3, fee_bps=6, min_amount_usd=100, max_amount_usd=1_000_000),
]


# ---------------------------------------------------------------------------
# Bridge Manager
# ---------------------------------------------------------------------------

class BridgeManager:
    """
    Cross-chain asset transfer manager.

    Selects optimal bridge routes based on cost, speed, and amount
    constraints. Tracks in-flight transfers and manages capital
    allocation across chains.

    Parameters
    ----------
    prefer_speed : bool
        If True, prefer faster (paid) bridges over native (free, slow).
    max_in_flight_usd : float
        Maximum capital in transit at any time.

    Example
    -------
        >>> manager = BridgeManager(prefer_speed=True)
        >>> route = manager.find_best_route(
        ...     ChainID.ETHEREUM, ChainID.ARBITRUM, amount_usd=50_000
        ... )
        >>> transfer = manager.initiate_transfer(route, 'USDC', 50_000)
    """

    def __init__(
        self,
        prefer_speed: bool = True,
        max_in_flight_usd: float = 200_000,
    ):
        self.prefer_speed = prefer_speed
        self.max_in_flight_usd = max_in_flight_usd
        self._transfers: List[BridgeTransfer] = []
        self._transfer_counter = 0

    def find_best_route(
        self,
        source: ChainID,
        dest: ChainID,
        amount_usd: float,
    ) -> Optional[BridgeRoute]:
        """
        Find the best bridge route for a transfer.

        Selection criteria:
        - Must support the transfer amount
        - If prefer_speed: minimize time (accepting higher fees)
        - Otherwise: minimize fees (accepting longer times)
        """
        candidates = [
            r for r in BRIDGE_ROUTES
            if r.source_chain == source
            and r.dest_chain == dest
            and r.min_amount_usd <= amount_usd <= r.max_amount_usd
        ]

        if not candidates:
            logger.warning(
                "No bridge route found: %s -> %s for $%.0f",
                source.name, dest.name, amount_usd
            )
            return None

        if self.prefer_speed:
            return min(candidates, key=lambda r: r.estimated_time_minutes)
        return min(candidates, key=lambda r: r.fee_bps)

    def initiate_transfer(
        self,
        route: BridgeRoute,
        token: str,
        amount_usd: float,
    ) -> BridgeTransfer:
        """Initiate a bridge transfer using the specified route."""
        self._transfer_counter += 1
        fee_usd = amount_usd * (route.fee_bps / 10_000)

        transfer = BridgeTransfer(
            transfer_id=f"BRIDGE-{self._transfer_counter:06d}",
            source_chain=route.source_chain,
            dest_chain=route.dest_chain,
            protocol=route.protocol,
            token=token,
            amount_usd=amount_usd,
            fee_usd=fee_usd,
            status=BridgeStatus.PENDING,
            initiated_at=pd.Timestamp.now(tz='UTC'),
        )

        self._transfers.append(transfer)
        logger.info(
            "Bridge transfer initiated: %s %s $%.0f via %s (%s -> %s)",
            transfer.transfer_id, token, amount_usd,
            route.protocol.value, route.source_chain.name, route.dest_chain.name
        )

        return transfer

    def get_in_flight_total(self) -> float:
        """Total USD value currently in transit."""
        return sum(
            t.amount_usd for t in self._transfers
            if t.status in (BridgeStatus.PENDING, BridgeStatus.CONFIRMING)
        )

    def get_chain_balances(self) -> Dict[str, float]:
        """Estimated capital available on each chain (after in-flight)."""
        # Placeholder: in production, query on-chain balances
        return {chain.name: 0.0 for chain in ChainID}

    def estimate_transfer_cost(
        self, source: ChainID, dest: ChainID, amount_usd: float
    ) -> Dict[str, float]:
        """Estimate bridge transfer cost and time."""
        route = self.find_best_route(source, dest, amount_usd)
        if route is None:
            return {'fee_usd': float('inf'), 'time_minutes': float('inf')}

        return {
            'fee_usd': amount_usd * (route.fee_bps / 10_000),
            'fee_bps': route.fee_bps,
            'time_minutes': route.estimated_time_minutes,
            'protocol': route.protocol.value,
        }
