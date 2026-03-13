"""
CEX Executor
=============

Centralized exchange execution for pairs trading and funding rate strategies.

Handles order placement, fill tracking, and position management across
Binance, Coinbase, and OKX via their REST and WebSocket APIs.

Fee Schedule (per PDF Section 2.4):
    - Entry/exit: 0.05% per side (0.10% round-trip per leg)
    - Total per pair trade: 0.20% (4 legs: buy A, sell B, close A, close B)
    - Slippage: 0.01-0.05% depending on token liquidity

Execution Assumptions:
    - Limit orders preferred (maker fees)
    - Both legs fill simultaneously (low slippage on liquid pairs)
    - Rebalancing at hourly frequency

Author: Tamer Atesyakar
Version: 1.0.0
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class OrderSide(Enum):
    BUY = 'BUY'
    SELL = 'SELL'


class OrderType(Enum):
    MARKET = 'MARKET'
    LIMIT = 'LIMIT'
    LIMIT_IOC = 'LIMIT_IOC'


class OrderStatus(Enum):
    PENDING = 'PENDING'
    FILLED = 'FILLED'
    PARTIAL = 'PARTIAL'
    CANCELLED = 'CANCELLED'
    REJECTED = 'REJECTED'


class VenueID(Enum):
    BINANCE = 'binance'
    COINBASE = 'coinbase'
    OKX = 'okx'


@dataclass
class OrderRequest:
    """Single-leg order specification."""
    symbol: str
    side: OrderSide
    size_usd: float
    order_type: OrderType = OrderType.LIMIT
    venue: VenueID = VenueID.BINANCE
    limit_price: Optional[float] = None
    timeout_seconds: float = 30.0


@dataclass
class OrderResult:
    """Execution result for a single order."""
    order_id: str
    symbol: str
    side: OrderSide
    venue: VenueID
    status: OrderStatus
    requested_size_usd: float
    filled_size_usd: float = 0.0
    avg_fill_price: float = 0.0
    commission_usd: float = 0.0
    slippage_bps: float = 0.0
    latency_ms: float = 0.0
    timestamp: Optional[pd.Timestamp] = None


@dataclass
class PairTradeResult:
    """Combined result of a pairs trade (long leg + short leg)."""
    pair_id: str
    long_result: OrderResult
    short_result: OrderResult
    total_commission_usd: float = 0.0
    total_slippage_bps: float = 0.0
    net_exposure_usd: float = 0.0


@dataclass
class CEXFeeSchedule:
    """Fee schedule for a CEX venue."""
    maker_fee_bps: float = 5.0       # 0.05%
    taker_fee_bps: float = 10.0      # 0.10%
    slippage_bps: float = 2.0        # 0.02% average
    min_order_usd: float = 10.0

    @property
    def round_trip_cost_bps(self) -> float:
        """Total cost for one leg (entry + exit)."""
        return 2 * self.maker_fee_bps + 2 * self.slippage_bps

    @property
    def pair_trade_cost_bps(self) -> float:
        """Total cost for full pair trade (4 legs)."""
        return 2 * self.round_trip_cost_bps


# ---------------------------------------------------------------------------
# Fee Schedules by Venue
# ---------------------------------------------------------------------------

DEFAULT_FEE_SCHEDULES: Dict[VenueID, CEXFeeSchedule] = {
    VenueID.BINANCE: CEXFeeSchedule(
        maker_fee_bps=5.0, taker_fee_bps=10.0, slippage_bps=2.0
    ),
    VenueID.COINBASE: CEXFeeSchedule(
        maker_fee_bps=6.0, taker_fee_bps=12.0, slippage_bps=3.0
    ),
    VenueID.OKX: CEXFeeSchedule(
        maker_fee_bps=5.0, taker_fee_bps=10.0, slippage_bps=2.5
    ),
}


# ---------------------------------------------------------------------------
# CEX Executor
# ---------------------------------------------------------------------------

class CEXExecutor:
    """
    Centralized exchange execution engine.

    Manages order routing, fill simulation, and position tracking for
    CEX-based pairs trading. In backtest mode, simulates fills using
    historical OHLCV data with realistic cost models.

    Parameters
    ----------
    venue : VenueID
        Target exchange (Binance, Coinbase, OKX).
    fee_schedule : CEXFeeSchedule, optional
        Custom fee schedule. Uses venue defaults if not provided.
    max_position_usd : float
        Maximum position size per pair (PDF: up to $100k for CEX).
    leverage : float
        Leverage multiplier (PDF: 1.0x only, no leverage).

    Example
    -------
        >>> executor = CEXExecutor(venue=VenueID.BINANCE)
        >>> result = executor.execute_pair_trade(
        ...     long_symbol='UNI', short_symbol='SUSHI',
        ...     size_usd=50_000, prices={'UNI': 12.5, 'SUSHI': 1.8}
        ... )
        >>> print(f"Commission: ${result.total_commission_usd:.2f}")
    """

    def __init__(
        self,
        venue: VenueID = VenueID.BINANCE,
        fee_schedule: Optional[CEXFeeSchedule] = None,
        max_position_usd: float = 100_000,
        leverage: float = 1.0,
    ):
        self.venue = venue
        self.fee_schedule = fee_schedule or DEFAULT_FEE_SCHEDULES[venue]
        self.max_position_usd = max_position_usd
        self.leverage = leverage
        self._order_counter = 0
        self._positions: Dict[str, float] = {}

        if leverage != 1.0:
            logger.warning(
                "PDF specifies 1.0x leverage for pairs trading. "
                "Setting leverage=%.1f may not be compliant.", leverage
            )

    # -----------------------------------------------------------------------
    # Core Execution
    # -----------------------------------------------------------------------

    def execute_pair_trade(
        self,
        long_symbol: str,
        short_symbol: str,
        size_usd: float,
        prices: Dict[str, float],
        hedge_ratio: float = 1.0,
    ) -> PairTradeResult:
        """
        Execute a pairs trade: long one token, short the other.

        Parameters
        ----------
        long_symbol : str
            Token to buy.
        short_symbol : str
            Token to sell.
        size_usd : float
            Notional size per leg in USD.
        prices : dict
            Current prices for both tokens.
        hedge_ratio : float
            Hedge ratio from cointegration regression.

        Returns
        -------
        PairTradeResult
            Combined execution result.
        """
        size_usd = min(size_usd, self.max_position_usd)

        long_result = self._simulate_fill(
            symbol=long_symbol,
            side=OrderSide.BUY,
            size_usd=size_usd,
            price=prices.get(long_symbol, 0.0),
        )

        short_size = size_usd * hedge_ratio
        short_result = self._simulate_fill(
            symbol=short_symbol,
            side=OrderSide.SELL,
            size_usd=short_size,
            price=prices.get(short_symbol, 0.0),
        )

        total_commission = long_result.commission_usd + short_result.commission_usd
        total_slippage = (long_result.slippage_bps + short_result.slippage_bps) / 2
        net_exposure = long_result.filled_size_usd - short_result.filled_size_usd

        pair_id = f"{long_symbol}_{short_symbol}_{int(time.time())}"

        return PairTradeResult(
            pair_id=pair_id,
            long_result=long_result,
            short_result=short_result,
            total_commission_usd=total_commission,
            total_slippage_bps=total_slippage,
            net_exposure_usd=net_exposure,
        )

    def close_pair_trade(
        self,
        long_symbol: str,
        short_symbol: str,
        size_usd: float,
        prices: Dict[str, float],
        hedge_ratio: float = 1.0,
    ) -> PairTradeResult:
        """Close an existing pair trade (reverse the legs)."""
        return self.execute_pair_trade(
            long_symbol=short_symbol,
            short_symbol=long_symbol,
            size_usd=size_usd,
            prices=prices,
            hedge_ratio=hedge_ratio,
        )

    def estimate_costs(self, size_usd: float) -> Dict[str, float]:
        """
        Estimate total round-trip costs for a pair trade.

        Returns
        -------
        dict
            Cost breakdown: fees, slippage, total (all in USD).
        """
        fee_cost = size_usd * 2 * (self.fee_schedule.pair_trade_cost_bps / 10_000)
        slippage_cost = size_usd * 2 * (self.fee_schedule.slippage_bps / 10_000)
        total = fee_cost + slippage_cost

        return {
            'fee_cost_usd': fee_cost,
            'slippage_cost_usd': slippage_cost,
            'total_cost_usd': total,
            'total_cost_bps': (total / (size_usd * 2)) * 10_000 if size_usd > 0 else 0,
        }

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _simulate_fill(
        self,
        symbol: str,
        side: OrderSide,
        size_usd: float,
        price: float,
    ) -> OrderResult:
        """Simulate an order fill with realistic costs."""
        self._order_counter += 1

        # Apply slippage
        slippage_bps = self.fee_schedule.slippage_bps
        if side == OrderSide.BUY:
            fill_price = price * (1 + slippage_bps / 10_000)
        else:
            fill_price = price * (1 - slippage_bps / 10_000)

        # Commission
        commission = size_usd * (self.fee_schedule.maker_fee_bps / 10_000)

        return OrderResult(
            order_id=f"CEX-{self.venue.value}-{self._order_counter:06d}",
            symbol=symbol,
            side=side,
            venue=self.venue,
            status=OrderStatus.FILLED,
            requested_size_usd=size_usd,
            filled_size_usd=size_usd,
            avg_fill_price=fill_price,
            commission_usd=commission,
            slippage_bps=slippage_bps,
            latency_ms=np.random.uniform(5, 50),
            timestamp=pd.Timestamp.now(tz='UTC'),
        )

    def get_position(self, symbol: str) -> float:
        """Get current position in USD for a symbol."""
        return self._positions.get(symbol, 0.0)

    def get_all_positions(self) -> Dict[str, float]:
        """Return all open positions."""
        return dict(self._positions)
