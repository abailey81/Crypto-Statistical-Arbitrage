"""
Exit Manager for Pairs Trading
==============================

Implements multi-stage exit logic including:
1. Partial exits (scale out as spread converges)
2. Trailing stops (lock in profits)
3. Time-based exits (max holding period)
4. Regime-based exits (exit if regime changes)

Why This Matters
----------------

All-or-nothing exits are suboptimal because:
- Miss profit from overshoot mean reversions
- Give back all gains if spread doesn't fully close
- Higher drawdowns from sudden reversals

Partial Exits solve this by:
- Locking in profits at multiple levels
- Reducing position risk as profit accumulates
- Allowing remaining position to capture overshoots

Trailing Stops solve this by:
- Protecting unrealized gains
- Letting winners run while limiting losers
- Dynamic adjustment based on volatility

Author: Crypto StatArb Quantitative Research
Version: 2.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Reason for position exit."""
    MEAN_REVERSION = "mean_reversion"      # Z-score crossed target
    PARTIAL_EXIT = "partial_exit"          # Scaled out at profit target
    TRAILING_STOP = "trailing_stop"        # Trailing stop hit
    STOP_LOSS = "stop_loss"                # Hard stop loss hit
    MAX_HOLD = "max_hold"                  # Maximum holding period
    REGIME_CHANGE = "regime_change"        # Market regime changed
    EMERGENCY = "emergency"                # Emergency liquidation


@dataclass
class ExitConfig:
    """Configuration for exit logic."""

    # Partial exit targets (z-score levels)
    partial_exit_levels: List[float] = field(default_factory=lambda: [-1.0, -0.5, 0.0])
    partial_exit_sizes: List[float] = field(default_factory=lambda: [0.33, 0.33, 0.34])

    # Trailing stop parameters
    enable_trailing: bool = True
    trailing_activation_pct: float = 0.02  # Activate after 2% profit
    trailing_distance_pct: float = 0.50    # Trail 50% behind high water mark

    # Hard stops
    stop_loss_z: float = 3.5
    max_hold_hours: int = 168  # 7 days

    # Regime-based
    exit_on_regime_change: bool = True


@dataclass
class PositionState:
    """State tracking for an open position."""

    pair_name: str
    direction: str              # 'long_spread' or 'short_spread'
    entry_time: datetime
    entry_z: float
    entry_prices: Dict[str, float]
    initial_size: float
    current_size: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    high_water_mark: float = 0.0
    trailing_stop_level: float = 0.0
    partial_exits_done: int = 0

    def remaining_fraction(self) -> float:
        """Fraction of position still open."""
        if self.initial_size == 0:
            return 0.0
        return self.current_size / self.initial_size


@dataclass
class ExitDecision:
    """Decision about whether and how to exit."""

    should_exit: bool
    exit_fraction: float  # 0.0 to 1.0 (fraction of current position to exit)
    exit_reason: ExitReason
    details: dict = field(default_factory=dict)


class ExitManager:
    """
    Manage multi-stage exit logic for pairs trading.

    Handles:
    - Partial exits at multiple profit targets
    - Trailing stops that lock in gains
    - Hard stops and max holding periods
    - Regime-based exits
    """

    def __init__(self, config: Optional[ExitConfig] = None):
        """
        Initialize exit manager.

        Args:
            config: Exit configuration (uses defaults if None)
        """
        self.config = config or ExitConfig()

        # Validate config
        if len(self.config.partial_exit_levels) != len(self.config.partial_exit_sizes):
            raise ValueError("partial_exit_levels and partial_exit_sizes must have same length")

        if not np.isclose(sum(self.config.partial_exit_sizes), 1.0):
            raise ValueError(f"partial_exit_sizes must sum to 1.0, got {sum(self.config.partial_exit_sizes)}")

    def check_exit(
        self,
        position: PositionState,
        current_z: float,
        current_prices: Dict[str, float],
        current_time: datetime,
        current_regime: Optional[str] = None,
        entry_regime: Optional[str] = None
    ) -> ExitDecision:
        """
        Check if position should be exited and by how much.

        Args:
            position: Current position state
            current_z: Current z-score
            current_prices: Current prices for both legs
            current_time: Current timestamp
            current_regime: Current market regime
            entry_regime: Regime when position was entered

        Returns:
            ExitDecision with exit instructions
        """
        # 1. Check hard stop loss
        if abs(current_z) > self.config.stop_loss_z:
            if self._should_stop(position, current_z):
                return ExitDecision(
                    should_exit=True,
                    exit_fraction=1.0,
                    exit_reason=ExitReason.STOP_LOSS,
                    details={'z_score': current_z, 'stop_level': self.config.stop_loss_z}
                )

        # 2. Check max holding period
        holding_hours = (current_time - position.entry_time).total_seconds() / 3600
        if holding_hours > self.config.max_hold_hours:
            return ExitDecision(
                should_exit=True,
                exit_fraction=1.0,
                exit_reason=ExitReason.MAX_HOLD,
                details={'holding_hours': holding_hours, 'max_hours': self.config.max_hold_hours}
            )

        # 3. Check regime change
        if (self.config.exit_on_regime_change and
            current_regime is not None and
            entry_regime is not None and
            current_regime != entry_regime):
            return ExitDecision(
                should_exit=True,
                exit_fraction=0.5,  # Exit half on regime change
                exit_reason=ExitReason.REGIME_CHANGE,
                details={'entry_regime': entry_regime, 'current_regime': current_regime}
            )

        # 4. Update unrealized P&L
        position.unrealized_pnl = self._calculate_unrealized_pnl(
            position, current_prices
        )

        # 5. Update high water mark and trailing stop
        if position.unrealized_pnl > position.high_water_mark:
            position.high_water_mark = position.unrealized_pnl

        # 6. Check trailing stop
        if self.config.enable_trailing:
            trailing_exit = self._check_trailing_stop(position)
            if trailing_exit.should_exit:
                return trailing_exit

        # 7. Check partial exits
        partial_exit = self._check_partial_exits(position, current_z)
        if partial_exit.should_exit:
            return partial_exit

        # 8. Check mean reversion exit (full close)
        mean_exit = self._check_mean_reversion_exit(position, current_z)
        if mean_exit.should_exit:
            return mean_exit

        # No exit
        return ExitDecision(
            should_exit=False,
            exit_fraction=0.0,
            exit_reason=ExitReason.MEAN_REVERSION  # Placeholder
        )

    def _should_stop(self, position: PositionState, current_z: float) -> bool:
        """Check if stop loss should be triggered."""
        if position.direction == 'long_spread':
            # Long spread: z-score should go up
            # Stop if z-score goes too far negative
            return current_z < -self.config.stop_loss_z
        else:
            # Short spread: z-score should go down
            # Stop if z-score goes too far positive
            return current_z > self.config.stop_loss_z

    def _check_trailing_stop(self, position: PositionState) -> ExitDecision:
        """Check if trailing stop should be triggered."""
        # Only activate trailing stop if profit exceeds threshold
        profit_pct = position.unrealized_pnl / (position.current_size + 1e-10)

        if profit_pct < self.config.trailing_activation_pct:
            return ExitDecision(False, 0.0, ExitReason.TRAILING_STOP)

        # Calculate trailing stop level
        trailing_stop = position.high_water_mark * (1 - self.config.trailing_distance_pct)

        # Check if current P&L fell below trailing stop
        if position.unrealized_pnl < trailing_stop:
            return ExitDecision(
                should_exit=True,
                exit_fraction=1.0,
                exit_reason=ExitReason.TRAILING_STOP,
                details={
                    'high_water_mark': position.high_water_mark,
                    'trailing_stop': trailing_stop,
                    'current_pnl': position.unrealized_pnl
                }
            )

        return ExitDecision(False, 0.0, ExitReason.TRAILING_STOP)

    def _check_partial_exits(self, position: PositionState, current_z: float) -> ExitDecision:
        """Check if any partial exit targets have been hit."""
        # Check which partial exit level we should be at
        target_level = position.partial_exits_done

        if target_level >= len(self.config.partial_exit_levels):
            return ExitDecision(False, 0.0, ExitReason.PARTIAL_EXIT)

        # Get the z-score target for this level
        z_target = self.config.partial_exit_levels[target_level]
        exit_size = self.config.partial_exit_sizes[target_level]

        # Check if target hit
        if position.direction == 'long_spread':
            # Long spread: exit as z-score rises (converges)
            if current_z >= z_target:
                return ExitDecision(
                    should_exit=True,
                    exit_fraction=exit_size,
                    exit_reason=ExitReason.PARTIAL_EXIT,
                    details={
                        'level': target_level,
                        'z_target': z_target,
                        'current_z': current_z,
                        'exit_fraction': exit_size
                    }
                )
        else:
            # Short spread: exit as z-score falls (converges)
            if current_z <= -z_target:
                return ExitDecision(
                    should_exit=True,
                    exit_fraction=exit_size,
                    exit_reason=ExitReason.PARTIAL_EXIT,
                    details={
                        'level': target_level,
                        'z_target': -z_target,
                        'current_z': current_z,
                        'exit_fraction': exit_size
                    }
                )

        return ExitDecision(False, 0.0, ExitReason.PARTIAL_EXIT)

    def _check_mean_reversion_exit(self, position: PositionState, current_z: float) -> ExitDecision:
        """Check for full mean reversion exit."""
        # Exit remaining position when z-score crosses zero
        if position.direction == 'long_spread':
            if current_z >= 0.0:
                return ExitDecision(
                    should_exit=True,
                    exit_fraction=1.0,
                    exit_reason=ExitReason.MEAN_REVERSION,
                    details={'current_z': current_z}
                )
        else:
            if current_z <= 0.0:
                return ExitDecision(
                    should_exit=True,
                    exit_fraction=1.0,
                    exit_reason=ExitReason.MEAN_REVERSION,
                    details={'current_z': current_z}
                )

        return ExitDecision(False, 0.0, ExitReason.MEAN_REVERSION)

    def _calculate_unrealized_pnl(
        self,
        position: PositionState,
        current_prices: Dict[str, float]
    ) -> float:
        """Calculate unrealized P&L for position."""
        # This is a simplified calculation
        # Real implementation would use hedge ratio and position details
        return 0.0  # Placeholder

    def execute_exit(
        self,
        position: PositionState,
        exit_decision: ExitDecision
    ) -> Tuple[float, float]:
        """
        Execute an exit and update position state.

        Args:
            position: Position to exit from
            exit_decision: Exit decision with instructions

        Returns:
            (exit_size, realized_pnl) tuple
        """
        # Calculate exit size
        exit_size = position.current_size * exit_decision.exit_fraction

        # Calculate realized P&L (simplified)
        realized_pnl = position.unrealized_pnl * exit_decision.exit_fraction

        # Update position
        position.current_size -= exit_size
        position.realized_pnl += realized_pnl

        # If partial exit, increment counter
        if exit_decision.exit_reason == ExitReason.PARTIAL_EXIT:
            position.partial_exits_done += 1

        logger.info(
            f"Executed {exit_decision.exit_reason.value} exit: "
            f"{exit_decision.exit_fraction*100:.1f}% of position, "
            f"Realized P&L: ${realized_pnl:,.2f}"
        )

        return exit_size, realized_pnl
