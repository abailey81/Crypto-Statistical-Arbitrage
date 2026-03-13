"""
Transaction Cost Modeling for Multi-Venue Crypto Futures Trading

Venue-specific cost models capturing the full cost structure of executing
futures basis trades across centralized exchanges (Binance, CME), hybrid
order books (Hyperliquid, dYdX), decentralized perpetuals (GMX), and
options-centric venues (Deribit).

Each model decomposes costs into: exchange fees (maker/taker), market impact
(slippage), gas/settlement costs, funding rate drag, and roll costs. The
TransactionCostAnalyzer provides cross-venue comparison and optimal routing.
The CostSimulator stress-tests cost assumptions via Monte Carlo methods.

Fee schedules reflect publicly documented rates as of Q4 2024.
"""

from __future__ import annotations

import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# =============================================================================
# ENUMERATIONS
# =============================================================================

class OrderSide(Enum):
    """Trade direction."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Execution method affecting fee tier selection."""
    MARKET = "market"
    LIMIT = "limit"


class VenueType(Enum):
    """Venue infrastructure classification."""
    CEX = "CEX"
    HYBRID = "hybrid"
    DEX = "DEX"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class CostBreakdown:
    """
    Itemized cost breakdown for a single trade execution.

    All monetary values are in USD. Rate fields are expressed as
    decimals (e.g., 0.0004 = 4 basis points).
    """
    exchange_fee_usd: float
    exchange_fee_rate: float
    slippage_usd: float
    slippage_rate: float
    gas_cost_usd: float
    funding_cost_usd: float
    roll_cost_usd: float
    total_cost_usd: float
    total_cost_bps: float
    venue: str
    notional_usd: float

    def __repr__(self) -> str:
        return (
            f"CostBreakdown(venue={self.venue}, "
            f"total={self.total_cost_usd:.2f} USD, "
            f"{self.total_cost_bps:.1f} bps)"
        )


@dataclass(frozen=True)
class VenueComparison:
    """Cost comparison result across multiple venues for the same trade."""
    trade_notional_usd: float
    trade_size_btc: float
    breakdowns: Dict[str, CostBreakdown]
    cheapest_venue: str
    most_expensive_venue: str
    cost_spread_bps: float


@dataclass
class TradeSpec:
    """Specification of a trade for cost estimation."""
    notional_usd: float
    size_btc: float
    btc_price: float
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    holding_period_hours: float = 24.0
    volatility_daily: float = 0.02
    order_book_depth_usd: float = 1_000_000.0
    funding_rate_8h: float = 0.0001
    days_to_roll: float = 0.0
    is_roll_trade: bool = False


@dataclass
class SimulationResult:
    """Monte Carlo simulation output for cost variability analysis."""
    venue: str
    n_simulations: int
    mean_cost_bps: float
    median_cost_bps: float
    std_cost_bps: float
    p5_cost_bps: float
    p95_cost_bps: float
    max_cost_bps: float
    cost_samples_bps: np.ndarray


# =============================================================================
# BASE COST MODEL
# =============================================================================

class VenueCostModel(ABC):
    """
    Base class for venue-specific transaction cost models.

    Subclasses must implement fee calculation, slippage estimation,
    gas cost computation, funding rate modeling, and roll cost logic.
    """

    def __init__(
        self,
        venue_name: str,
        venue_type: VenueType,
        maker_fee: float,
        taker_fee: float,
        base_slippage_bps: float = 1.0,
    ):
        self.venue_name = venue_name
        self.venue_type = venue_type
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.base_slippage_bps = base_slippage_bps

    def estimate_total_cost(self, trade: TradeSpec) -> CostBreakdown:
        """Compute full cost breakdown for a given trade specification."""
        fee_rate = self._get_fee_rate(trade.order_type)
        fee_usd = trade.notional_usd * fee_rate

        slippage_rate = self._estimate_slippage(
            trade.notional_usd,
            trade.order_book_depth_usd,
            trade.volatility_daily,
        )
        slippage_usd = trade.notional_usd * slippage_rate

        gas_usd = self._estimate_gas_cost(trade)
        funding_usd = self._estimate_funding_cost(trade)
        roll_usd = self._estimate_roll_cost(trade)

        total_usd = fee_usd + slippage_usd + gas_usd + funding_usd + roll_usd
        total_bps = (total_usd / trade.notional_usd) * 10_000 if trade.notional_usd > 0 else 0.0

        return CostBreakdown(
            exchange_fee_usd=fee_usd,
            exchange_fee_rate=fee_rate,
            slippage_usd=slippage_usd,
            slippage_rate=slippage_rate,
            gas_cost_usd=gas_usd,
            funding_cost_usd=funding_usd,
            roll_cost_usd=roll_usd,
            total_cost_usd=total_usd,
            total_cost_bps=total_bps,
            venue=self.venue_name,
            notional_usd=trade.notional_usd,
        )

    def _get_fee_rate(self, order_type: OrderType) -> float:
        """Select maker or taker fee based on order type."""
        if order_type == OrderType.LIMIT:
            return self.maker_fee
        return self.taker_fee

    @abstractmethod
    def _estimate_slippage(
        self,
        notional_usd: float,
        order_book_depth_usd: float,
        volatility_daily: float,
    ) -> float:
        """
        Estimate market impact as a fraction of notional.

        Slippage depends on order size relative to available liquidity
        and current volatility regime. Returns a decimal rate.
        """

    @abstractmethod
    def _estimate_gas_cost(self, trade: TradeSpec) -> float:
        """Estimate gas or settlement cost in USD. Zero for off-chain venues."""

    @abstractmethod
    def _estimate_funding_cost(self, trade: TradeSpec) -> float:
        """
        Estimate funding rate cost over the holding period.

        Perpetual contracts accrue funding every 8 hours.
        Quarterly futures have zero funding but carry basis risk.
        """

    @abstractmethod
    def _estimate_roll_cost(self, trade: TradeSpec) -> float:
        """
        Estimate the cost of rolling a futures position to the next expiry.

        Includes the bid-ask spread of the calendar spread and any
        basis convergence cost during the roll window.
        """

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"venue={self.venue_name}, "
            f"maker={self.maker_fee*10000:.1f}bps, "
            f"taker={self.taker_fee*10000:.1f}bps)"
        )


# =============================================================================
# BINANCE FUTURES COST MODEL
# =============================================================================

class BinanceFuturesCostModel(VenueCostModel):
    """
    Binance USDT-margined futures cost model.

    Fee schedule: VIP 0 tier (0.01% maker, 0.04% taker).
    Funding rate: applied every 8 hours, typically 0.01% per interval
    in contango. Slippage is low for BTC due to deep order books
    ($50M+ within 0.1% of mid on BTCUSDT perpetual).
    Roll cost: ~0.02% for quarterly-to-quarterly calendar spreads.
    """

    # Typical BTC/USDT perpetual order book depth within 0.1% of mid-price
    TYPICAL_DEPTH_USD: float = 50_000_000.0
    # Default funding rate per 8h interval (long-run average in contango)
    DEFAULT_FUNDING_RATE_8H: float = 0.0001
    # Calendar spread roll cost as fraction of notional
    ROLL_COST_RATE: float = 0.0002
    # Minimum tick impact for any market order
    MIN_SLIPPAGE_BPS: float = 0.1

    def __init__(self):
        super().__init__(
            venue_name="binance",
            venue_type=VenueType.CEX,
            maker_fee=0.0001,
            taker_fee=0.0004,
            base_slippage_bps=1.0,
        )

    def _estimate_slippage(
        self,
        notional_usd: float,
        order_book_depth_usd: float,
        volatility_daily: float,
    ) -> float:
        """
        Binance slippage model: square-root impact with volatility scaling.

        For liquid BTC pairs, impact follows sqrt(participation_rate).
        Volatility amplifies slippage linearly as market makers widen quotes.
        """
        depth = max(order_book_depth_usd, self.TYPICAL_DEPTH_USD)
        participation_rate = notional_usd / depth
        # Square-root market impact model (Almgren-Chriss)
        base_impact = math.sqrt(participation_rate) * self.base_slippage_bps / 10_000
        # Volatility scaling: slippage increases with realized vol
        vol_multiplier = 1.0 + max(0.0, (volatility_daily - 0.02)) * 10.0
        slippage = base_impact * vol_multiplier
        return max(slippage, self.MIN_SLIPPAGE_BPS / 10_000)

    def _estimate_gas_cost(self, trade: TradeSpec) -> float:
        """Binance is off-chain. No gas costs."""
        return 0.0

    def _estimate_funding_cost(self, trade: TradeSpec) -> float:
        """
        Funding accrues every 8 hours. Cost depends on position direction
        and the prevailing funding rate.

        Positive funding rate: longs pay shorts.
        Negative funding rate: shorts pay longs.
        """
        intervals = trade.holding_period_hours / 8.0
        rate = trade.funding_rate_8h if trade.funding_rate_8h != 0.0 else self.DEFAULT_FUNDING_RATE_8H
        # For basis trades, long futures pays funding in contango
        if trade.side == OrderSide.BUY:
            return trade.notional_usd * rate * intervals
        # Short futures receives funding in contango
        return -trade.notional_usd * rate * intervals

    def _estimate_roll_cost(self, trade: TradeSpec) -> float:
        """
        Roll cost for quarterly futures.
        Includes calendar spread bid-ask and basis convergence.
        """
        if not trade.is_roll_trade:
            return 0.0
        return trade.notional_usd * self.ROLL_COST_RATE


# =============================================================================
# CME FUTURES COST MODEL
# =============================================================================

class CMEFuturesCostModel(VenueCostModel):
    """
    CME Bitcoin futures (BTC) cost model.

    CME charges per-contract fees rather than ad-valorem rates.
    Standard contract: 5 BTC. Micro contract (MBT): 0.1 BTC.
    Exchange fees: $6 per standard contract ($10 all-in with clearing).
    Slippage is minimal for institutional-size orders due to deep
    institutional liquidity. No funding rate (quarterly settlement).
    """

    # CME fee per standard contract (5 BTC), inclusive of exchange + clearing
    FEE_PER_CONTRACT_USD: float = 10.0
    # Contract size in BTC
    CONTRACT_SIZE_BTC: float = 5.0
    # Micro contract fee
    MICRO_FEE_PER_CONTRACT_USD: float = 2.50
    MICRO_CONTRACT_SIZE_BTC: float = 0.1
    # CME institutional depth (top-of-book typically $20M+)
    TYPICAL_DEPTH_USD: float = 100_000_000.0
    # Roll cost is lower on CME due to active calendar spread market
    ROLL_COST_RATE: float = 0.00015

    def __init__(self, use_micro: bool = False):
        # CME fees are per-contract; effective rate depends on BTC price
        super().__init__(
            venue_name="cme",
            venue_type=VenueType.CEX,
            maker_fee=0.0,  # Per-contract, not ad-valorem
            taker_fee=0.0,
            base_slippage_bps=0.5,
        )
        self.use_micro = use_micro

    def _get_fee_rate(self, order_type: OrderType) -> float:
        """CME uses per-contract fees; this returns 0. Fees computed in estimate_total_cost."""
        return 0.0

    def estimate_total_cost(self, trade: TradeSpec) -> CostBreakdown:
        """Override to handle CME per-contract fee structure."""
        if self.use_micro:
            n_contracts = trade.size_btc / self.MICRO_CONTRACT_SIZE_BTC
            fee_usd = n_contracts * self.MICRO_FEE_PER_CONTRACT_USD
        else:
            n_contracts = trade.size_btc / self.CONTRACT_SIZE_BTC
            fee_usd = n_contracts * self.FEE_PER_CONTRACT_USD

        fee_rate = fee_usd / trade.notional_usd if trade.notional_usd > 0 else 0.0

        slippage_rate = self._estimate_slippage(
            trade.notional_usd,
            trade.order_book_depth_usd,
            trade.volatility_daily,
        )
        slippage_usd = trade.notional_usd * slippage_rate
        gas_usd = self._estimate_gas_cost(trade)
        funding_usd = self._estimate_funding_cost(trade)
        roll_usd = self._estimate_roll_cost(trade)

        total_usd = fee_usd + slippage_usd + gas_usd + funding_usd + roll_usd
        total_bps = (total_usd / trade.notional_usd) * 10_000 if trade.notional_usd > 0 else 0.0

        return CostBreakdown(
            exchange_fee_usd=fee_usd,
            exchange_fee_rate=fee_rate,
            slippage_usd=slippage_usd,
            slippage_rate=slippage_rate,
            gas_cost_usd=gas_usd,
            funding_cost_usd=funding_usd,
            roll_cost_usd=roll_usd,
            total_cost_usd=total_usd,
            total_cost_bps=total_bps,
            venue=self.venue_name,
            notional_usd=trade.notional_usd,
        )

    def _estimate_slippage(
        self,
        notional_usd: float,
        order_book_depth_usd: float,
        volatility_daily: float,
    ) -> float:
        """
        CME slippage is minimal for institutional-size orders.
        The BTC contract has narrow bid-ask spreads during US trading hours.
        Off-hours liquidity is thinner, modeled by a small vol adjustment.
        """
        depth = max(order_book_depth_usd, self.TYPICAL_DEPTH_USD)
        participation_rate = notional_usd / depth
        base_impact = math.sqrt(participation_rate) * self.base_slippage_bps / 10_000
        vol_multiplier = 1.0 + max(0.0, (volatility_daily - 0.02)) * 5.0
        return base_impact * vol_multiplier

    def _estimate_gas_cost(self, trade: TradeSpec) -> float:
        """CME is a traditional exchange. No gas costs."""
        return 0.0

    def _estimate_funding_cost(self, trade: TradeSpec) -> float:
        """CME quarterly futures have no funding rate. Cost is zero."""
        return 0.0

    def _estimate_roll_cost(self, trade: TradeSpec) -> float:
        """
        CME calendar spread roll cost.
        CME has an active spread market, so roll costs are competitive.
        """
        if not trade.is_roll_trade:
            return 0.0
        return trade.notional_usd * self.ROLL_COST_RATE


# =============================================================================
# HYPERLIQUID COST MODEL
# =============================================================================

class HyperliquidCostModel(VenueCostModel):
    """
    Hyperliquid L1 perpetual futures cost model.

    Maker rebate: 0.00% (free limit orders). Taker fee: 0.025%.
    On-chain settlement on Hyperliquid L1 with ~$0.50 gas per trade.
    Order book-based matching (not AMM), but thinner liquidity than
    Binance leads to higher slippage at size.

    Max leverage constraint: 1.5x (per risk mandate).
    """

    GAS_COST_USD: float = 0.50
    TYPICAL_DEPTH_USD: float = 10_000_000.0
    DEFAULT_FUNDING_RATE_8H: float = 0.0001

    def __init__(self):
        super().__init__(
            venue_name="hyperliquid",
            venue_type=VenueType.HYBRID,
            maker_fee=0.0000,
            taker_fee=0.00025,
            base_slippage_bps=2.0,
        )

    def _estimate_slippage(
        self,
        notional_usd: float,
        order_book_depth_usd: float,
        volatility_daily: float,
    ) -> float:
        """
        Hyperliquid has thinner liquidity than Binance.
        Slippage grows faster with order size.
        """
        depth = min(order_book_depth_usd, self.TYPICAL_DEPTH_USD)
        participation_rate = notional_usd / depth
        # Higher base impact due to thinner books
        base_impact = math.sqrt(participation_rate) * self.base_slippage_bps / 10_000
        vol_multiplier = 1.0 + max(0.0, (volatility_daily - 0.02)) * 15.0
        return max(base_impact * vol_multiplier, 0.5 / 10_000)

    def _estimate_gas_cost(self, trade: TradeSpec) -> float:
        """Fixed gas cost on Hyperliquid L1."""
        return self.GAS_COST_USD

    def _estimate_funding_cost(self, trade: TradeSpec) -> float:
        """
        Hyperliquid perpetuals accrue funding every 8 hours,
        identical mechanism to centralized perpetual exchanges.
        """
        intervals = trade.holding_period_hours / 8.0
        rate = trade.funding_rate_8h if trade.funding_rate_8h != 0.0 else self.DEFAULT_FUNDING_RATE_8H
        if trade.side == OrderSide.BUY:
            return trade.notional_usd * rate * intervals
        return -trade.notional_usd * rate * intervals

    def _estimate_roll_cost(self, trade: TradeSpec) -> float:
        """Hyperliquid offers only perpetuals. No roll cost."""
        return 0.0


# =============================================================================
# DYDX V4 COST MODEL
# =============================================================================

class DYDXCostModel(VenueCostModel):
    """
    dYdX V4 (Cosmos appchain) perpetual futures cost model.

    Maker: 0.00%. Taker: 0.05%. dYdX V4 runs its own Cosmos appchain,
    so gas costs are minimal (~$0.01 per transaction). Liquidity is
    lower than Binance but growing. Venue allocation capped at 5%.
    """

    GAS_COST_USD: float = 0.01
    TYPICAL_DEPTH_USD: float = 5_000_000.0
    DEFAULT_FUNDING_RATE_8H: float = 0.0001

    def __init__(self):
        super().__init__(
            venue_name="dydx",
            venue_type=VenueType.HYBRID,
            maker_fee=0.0000,
            taker_fee=0.0005,
            base_slippage_bps=3.0,
        )

    def _estimate_slippage(
        self,
        notional_usd: float,
        order_book_depth_usd: float,
        volatility_daily: float,
    ) -> float:
        """
        dYdX V4 has limited but improving liquidity.
        Higher base slippage than CEX venues, especially for larger orders.
        """
        depth = min(order_book_depth_usd, self.TYPICAL_DEPTH_USD)
        participation_rate = notional_usd / depth
        base_impact = math.sqrt(participation_rate) * self.base_slippage_bps / 10_000
        vol_multiplier = 1.0 + max(0.0, (volatility_daily - 0.02)) * 20.0
        return max(base_impact * vol_multiplier, 1.0 / 10_000)

    def _estimate_gas_cost(self, trade: TradeSpec) -> float:
        """Cosmos chain gas is negligible."""
        return self.GAS_COST_USD

    def _estimate_funding_cost(self, trade: TradeSpec) -> float:
        """dYdX perpetual funding, 8-hour intervals."""
        intervals = trade.holding_period_hours / 8.0
        rate = trade.funding_rate_8h if trade.funding_rate_8h != 0.0 else self.DEFAULT_FUNDING_RATE_8H
        if trade.side == OrderSide.BUY:
            return trade.notional_usd * rate * intervals
        return -trade.notional_usd * rate * intervals

    def _estimate_roll_cost(self, trade: TradeSpec) -> float:
        """dYdX offers only perpetuals. No roll cost."""
        return 0.0


# =============================================================================
# DERIBIT COST MODEL
# =============================================================================

class DeribitCostModel(VenueCostModel):
    """
    Deribit futures and options cost model.

    Maker: 0.01%. Taker: 0.05%. Deribit is the dominant venue for
    BTC/ETH options and also offers futures. Options trades incur
    additional Greeks-related costs (delta hedging slippage, gamma
    scalping costs, vega exposure from vol surface movement).

    Deribit futures have quarterly expiries and an active basis market.
    """

    TYPICAL_DEPTH_USD: float = 15_000_000.0
    ROLL_COST_RATE: float = 0.00018
    # Approximate delta-hedging cost for options strategies (per rebalance)
    DELTA_HEDGE_COST_BPS: float = 2.0
    # Vega cost: PnL impact of 1-vol-point move, normalized to bps
    VEGA_EXPOSURE_BPS_PER_VOL_POINT: float = 0.5

    def __init__(self):
        super().__init__(
            venue_name="deribit",
            venue_type=VenueType.CEX,
            maker_fee=0.0001,
            taker_fee=0.0005,
            base_slippage_bps=1.5,
        )

    def _estimate_slippage(
        self,
        notional_usd: float,
        order_book_depth_usd: float,
        volatility_daily: float,
    ) -> float:
        """
        Deribit futures slippage. Liquidity is concentrated around
        front-month and quarterly expiries. Back-month contracts
        are thinner.
        """
        depth = min(order_book_depth_usd, self.TYPICAL_DEPTH_USD)
        participation_rate = notional_usd / depth
        base_impact = math.sqrt(participation_rate) * self.base_slippage_bps / 10_000
        vol_multiplier = 1.0 + max(0.0, (volatility_daily - 0.02)) * 12.0
        return max(base_impact * vol_multiplier, 0.5 / 10_000)

    def _estimate_gas_cost(self, trade: TradeSpec) -> float:
        """Deribit is off-chain. No gas costs."""
        return 0.0

    def _estimate_funding_cost(self, trade: TradeSpec) -> float:
        """
        Deribit perpetuals have 8-hour funding. Quarterly futures do not.
        This model assumes quarterly futures by default (no funding).
        """
        return 0.0

    def _estimate_roll_cost(self, trade: TradeSpec) -> float:
        """
        Deribit quarterly futures roll cost.
        Active calendar spread market keeps roll costs moderate.
        """
        if not trade.is_roll_trade:
            return 0.0
        return trade.notional_usd * self.ROLL_COST_RATE

    def estimate_options_greeks_cost(
        self,
        notional_usd: float,
        delta_hedge_frequency: int = 4,
        implied_vol_change: float = 1.0,
    ) -> float:
        """
        Estimate the cost of Greeks management for options positions.

        Args:
            notional_usd: Options notional exposure.
            delta_hedge_frequency: Number of delta hedges per day.
            implied_vol_change: Expected IV change in vol points per day.

        Returns:
            Estimated daily Greeks management cost in USD.
        """
        delta_cost = notional_usd * (self.DELTA_HEDGE_COST_BPS / 10_000) * delta_hedge_frequency
        vega_cost = notional_usd * (self.VEGA_EXPOSURE_BPS_PER_VOL_POINT / 10_000) * implied_vol_change
        return delta_cost + vega_cost


# =============================================================================
# GMX COST MODEL
# =============================================================================

class GMXCostModel(VenueCostModel):
    """
    GMX V2 perpetual cost model.

    GMX uses oracle-based pricing (Chainlink + secondary oracles),
    so there is no traditional slippage from order book impact.
    Instead, traders face:
      - 0.1% open/close fee (applied to position notional)
      - Borrow fee proportional to utilization and position duration
      - Price lag from oracle update latency (1-3 seconds)
      - Execution fee (gas on Arbitrum, ~$0.50)

    GMX V2 introduced isolated markets and dynamic fees based on
    open interest imbalance (long/short skew).
    """

    OPEN_CLOSE_FEE: float = 0.001  # 0.1%
    # Borrow fee: annualized rate that scales with pool utilization
    BASE_BORROW_RATE_ANNUAL: float = 0.05  # 5% annualized at 50% utilization
    GAS_COST_USD: float = 0.50  # Arbitrum execution
    # Oracle price lag creates implicit cost (est. 1-2 bps in normal conditions)
    ORACLE_LAG_BPS: float = 1.5
    TYPICAL_DEPTH_USD: float = 20_000_000.0  # Pool-based, not order book

    def __init__(self):
        super().__init__(
            venue_name="gmx",
            venue_type=VenueType.DEX,
            maker_fee=self.OPEN_CLOSE_FEE,
            taker_fee=self.OPEN_CLOSE_FEE,
            base_slippage_bps=0.0,  # No order book slippage (oracle pricing)
        )

    def _get_fee_rate(self, order_type: OrderType) -> float:
        """GMX charges the same fee regardless of order type."""
        return self.OPEN_CLOSE_FEE

    def _estimate_slippage(
        self,
        notional_usd: float,
        order_book_depth_usd: float,
        volatility_daily: float,
    ) -> float:
        """
        GMX has no order book slippage (oracle pricing), but oracle lag
        creates an implicit cost. During volatile periods, the lag cost
        increases as price moves between submission and execution.
        """
        base_lag = self.ORACLE_LAG_BPS / 10_000
        # During high vol, oracle lag becomes more costly
        vol_multiplier = 1.0 + max(0.0, (volatility_daily - 0.02)) * 25.0
        return base_lag * vol_multiplier

    def _estimate_gas_cost(self, trade: TradeSpec) -> float:
        """Arbitrum execution gas cost."""
        return self.GAS_COST_USD

    def _estimate_funding_cost(self, trade: TradeSpec) -> float:
        """
        GMX borrow fee for leveraged positions.
        Accrues hourly based on pool utilization rate.
        Approximated as a fraction of the annualized borrow rate.
        """
        hours = trade.holding_period_hours
        hourly_rate = self.BASE_BORROW_RATE_ANNUAL / (365.0 * 24.0)
        return trade.notional_usd * hourly_rate * hours

    def _estimate_roll_cost(self, trade: TradeSpec) -> float:
        """GMX perpetuals do not expire. No roll cost."""
        return 0.0


# =============================================================================
# VENUE REGISTRY
# =============================================================================

_VENUE_MODELS: Dict[str, VenueCostModel] = {}


def _build_venue_registry() -> Dict[str, VenueCostModel]:
    """Instantiate all venue cost models and return a name-keyed dict."""
    if _VENUE_MODELS:
        return _VENUE_MODELS

    models = [
        BinanceFuturesCostModel(),
        CMEFuturesCostModel(),
        CMEFuturesCostModel(use_micro=True),
        HyperliquidCostModel(),
        DYDXCostModel(),
        DeribitCostModel(),
        GMXCostModel(),
    ]
    for m in models:
        name = m.venue_name
        if isinstance(m, CMEFuturesCostModel) and m.use_micro:
            name = "cme_micro"
        _VENUE_MODELS[name] = m
    return _VENUE_MODELS


def get_venue_model(venue_name: str) -> VenueCostModel:
    """Retrieve a venue cost model by name."""
    registry = _build_venue_registry()
    if venue_name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown venue '{venue_name}'. Available: {available}")
    return registry[venue_name]


def list_venues() -> List[str]:
    """Return sorted list of available venue names."""
    return sorted(_build_venue_registry().keys())


# =============================================================================
# TRANSACTION COST ANALYZER
# =============================================================================

class TransactionCostAnalyzer:
    """
    Cross-venue transaction cost analysis.

    Provides cost comparison, optimal venue routing, cost attribution,
    and return impact estimation for multi-venue futures basis trades.
    """

    def __init__(self, venues: Optional[List[str]] = None):
        """
        Args:
            venues: List of venue names to include. Defaults to all.
        """
        registry = _build_venue_registry()
        if venues is not None:
            self.models = {v: registry[v] for v in venues if v in registry}
        else:
            self.models = dict(registry)

    def calculate_cost(self, venue: str, trade: TradeSpec) -> CostBreakdown:
        """Calculate total cost for a single venue."""
        model = self.models.get(venue)
        if model is None:
            raise ValueError(f"Venue '{venue}' not configured in analyzer.")
        return model.estimate_total_cost(trade)

    def compare_venues(self, trade: TradeSpec) -> VenueComparison:
        """
        Compare costs across all configured venues for the same trade.

        Returns a VenueComparison with per-venue breakdowns, the cheapest
        and most expensive venues, and the cost spread in basis points.
        """
        breakdowns: Dict[str, CostBreakdown] = {}
        for venue_name, model in self.models.items():
            breakdowns[venue_name] = model.estimate_total_cost(trade)

        sorted_by_cost = sorted(breakdowns.items(), key=lambda x: x[1].total_cost_bps)
        cheapest = sorted_by_cost[0][0]
        most_expensive = sorted_by_cost[-1][0]
        spread = sorted_by_cost[-1][1].total_cost_bps - sorted_by_cost[0][1].total_cost_bps

        return VenueComparison(
            trade_notional_usd=trade.notional_usd,
            trade_size_btc=trade.size_btc,
            breakdowns=breakdowns,
            cheapest_venue=cheapest,
            most_expensive_venue=most_expensive,
            cost_spread_bps=spread,
        )

    def optimal_venue(self, trade: TradeSpec) -> Tuple[str, CostBreakdown]:
        """Return the venue with the lowest total cost for the given trade."""
        comparison = self.compare_venues(trade)
        best = comparison.cheapest_venue
        return best, comparison.breakdowns[best]

    def cost_attribution(self, trade: TradeSpec) -> Dict[str, Dict[str, float]]:
        """
        Break down cost components as a percentage of total cost per venue.

        Returns a dict of venue -> {component_name: percentage_of_total}.
        Useful for identifying the dominant cost driver at each venue.
        """
        result: Dict[str, Dict[str, float]] = {}
        for venue_name, model in self.models.items():
            bd = model.estimate_total_cost(trade)
            total = bd.total_cost_usd
            if total <= 0:
                result[venue_name] = {
                    "exchange_fee": 0.0, "slippage": 0.0,
                    "gas": 0.0, "funding": 0.0, "roll": 0.0,
                }
                continue
            result[venue_name] = {
                "exchange_fee": bd.exchange_fee_usd / total * 100.0,
                "slippage": bd.slippage_usd / total * 100.0,
                "gas": bd.gas_cost_usd / total * 100.0,
                "funding": max(0.0, bd.funding_cost_usd) / total * 100.0,
                "roll": bd.roll_cost_usd / total * 100.0,
            }
        return result

    def cost_impact_on_returns(
        self,
        venue: str,
        trade: TradeSpec,
        gross_return_bps: float,
    ) -> Dict[str, float]:
        """
        Estimate the impact of transaction costs on strategy gross returns.

        Args:
            venue: Target venue.
            trade: Trade specification.
            gross_return_bps: Expected gross return per trade in bps.

        Returns:
            Dict with gross return, cost, net return, cost-to-return ratio.
        """
        bd = self.calculate_cost(venue, trade)
        net_return = gross_return_bps - bd.total_cost_bps
        ratio = bd.total_cost_bps / gross_return_bps if gross_return_bps > 0 else float("inf")
        return {
            "gross_return_bps": gross_return_bps,
            "total_cost_bps": bd.total_cost_bps,
            "net_return_bps": net_return,
            "cost_to_return_ratio": ratio,
            "is_profitable": net_return > 0,
        }

    def round_trip_cost(
        self,
        venue: str,
        trade: TradeSpec,
    ) -> CostBreakdown:
        """
        Estimate total round-trip cost (entry + exit).

        Doubles the exchange fee and slippage components. Gas is charged
        twice (once per leg). Funding and roll are single-charge.
        """
        model = self.models.get(venue)
        if model is None:
            raise ValueError(f"Venue '{venue}' not configured.")

        entry = model.estimate_total_cost(trade)

        rt_fee_usd = entry.exchange_fee_usd * 2.0
        rt_slip_usd = entry.slippage_usd * 2.0
        rt_gas_usd = entry.gas_cost_usd * 2.0
        rt_fund_usd = entry.funding_cost_usd  # Accrues over hold, not per leg
        rt_roll_usd = entry.roll_cost_usd

        rt_total_usd = rt_fee_usd + rt_slip_usd + rt_gas_usd + rt_fund_usd + rt_roll_usd
        rt_total_bps = (rt_total_usd / trade.notional_usd) * 10_000 if trade.notional_usd > 0 else 0.0

        return CostBreakdown(
            exchange_fee_usd=rt_fee_usd,
            exchange_fee_rate=entry.exchange_fee_rate * 2.0,
            slippage_usd=rt_slip_usd,
            slippage_rate=entry.slippage_rate * 2.0,
            gas_cost_usd=rt_gas_usd,
            funding_cost_usd=rt_fund_usd,
            roll_cost_usd=rt_roll_usd,
            total_cost_usd=rt_total_usd,
            total_cost_bps=rt_total_bps,
            venue=venue,
            notional_usd=trade.notional_usd,
        )


# =============================================================================
# COST SIMULATOR (MONTE CARLO)
# =============================================================================

class CostSimulator:
    """
    Monte Carlo simulation of transaction cost variability.

    Perturbs slippage (via volatility), funding rates, and gas costs
    to produce a distribution of realized costs. Useful for stress
    testing cost assumptions and estimating expected cost savings
    from optimal venue routing.
    """

    def __init__(
        self,
        analyzer: Optional[TransactionCostAnalyzer] = None,
        seed: int = 42,
    ):
        self.analyzer = analyzer or TransactionCostAnalyzer()
        self.rng = np.random.default_rng(seed)

    def simulate_cost_distribution(
        self,
        venue: str,
        base_trade: TradeSpec,
        n_simulations: int = 10_000,
        vol_range: Tuple[float, float] = (0.005, 0.08),
        funding_range: Tuple[float, float] = (-0.0003, 0.0005),
        gas_multiplier_range: Tuple[float, float] = (0.5, 3.0),
    ) -> SimulationResult:
        """
        Simulate cost variability for a single venue.

        Draws random volatility, funding rate, and gas multiplier from
        uniform distributions and computes costs under each scenario.

        Args:
            venue: Venue name.
            base_trade: Base trade specification (volatility and funding
                        will be overridden per simulation).
            n_simulations: Number of Monte Carlo draws.
            vol_range: (min, max) daily volatility.
            funding_range: (min, max) 8-hour funding rate.
            gas_multiplier_range: (min, max) multiplier on base gas cost.

        Returns:
            SimulationResult with cost distribution statistics.
        """
        model = self.analyzer.models.get(venue)
        if model is None:
            raise ValueError(f"Venue '{venue}' not available for simulation.")

        vols = self.rng.uniform(vol_range[0], vol_range[1], n_simulations)
        fundings = self.rng.uniform(funding_range[0], funding_range[1], n_simulations)
        gas_mults = self.rng.uniform(gas_multiplier_range[0], gas_multiplier_range[1], n_simulations)

        costs_bps = np.empty(n_simulations)

        for i in range(n_simulations):
            trade = TradeSpec(
                notional_usd=base_trade.notional_usd,
                size_btc=base_trade.size_btc,
                btc_price=base_trade.btc_price,
                side=base_trade.side,
                order_type=base_trade.order_type,
                holding_period_hours=base_trade.holding_period_hours,
                volatility_daily=float(vols[i]),
                order_book_depth_usd=base_trade.order_book_depth_usd,
                funding_rate_8h=float(fundings[i]),
                days_to_roll=base_trade.days_to_roll,
                is_roll_trade=base_trade.is_roll_trade,
            )
            bd = model.estimate_total_cost(trade)
            costs_bps[i] = bd.total_cost_bps

        return SimulationResult(
            venue=venue,
            n_simulations=n_simulations,
            mean_cost_bps=float(np.mean(costs_bps)),
            median_cost_bps=float(np.median(costs_bps)),
            std_cost_bps=float(np.std(costs_bps)),
            p5_cost_bps=float(np.percentile(costs_bps, 5)),
            p95_cost_bps=float(np.percentile(costs_bps, 95)),
            max_cost_bps=float(np.max(costs_bps)),
            cost_samples_bps=costs_bps,
        )

    def stress_test_high_volatility(
        self,
        base_trade: TradeSpec,
        stress_vol: float = 0.10,
        n_simulations: int = 5_000,
    ) -> Dict[str, SimulationResult]:
        """
        Stress test all venues under a high-volatility regime.

        Sets the minimum volatility to stress_vol and widens the
        funding rate distribution to capture crisis conditions
        (e.g., LUNA collapse, FTX bankruptcy).

        Args:
            base_trade: Base trade specification.
            stress_vol: Floor volatility for stress scenario (daily).
            n_simulations: Number of draws per venue.

        Returns:
            Dict of venue_name -> SimulationResult.
        """
        results: Dict[str, SimulationResult] = {}
        for venue_name in self.analyzer.models:
            results[venue_name] = self.simulate_cost_distribution(
                venue=venue_name,
                base_trade=base_trade,
                n_simulations=n_simulations,
                vol_range=(stress_vol, stress_vol * 2.0),
                funding_range=(-0.001, 0.003),
                gas_multiplier_range=(1.0, 10.0),
            )
        return results

    def routing_savings_estimate(
        self,
        base_trade: TradeSpec,
        n_simulations: int = 10_000,
        venues: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Estimate cost savings from optimal venue routing vs. fixed venue.

        For each simulation, computes costs at all venues and selects
        the cheapest. Compares to the average cost of using each venue
        exclusively.

        Args:
            base_trade: Base trade specification.
            n_simulations: Number of draws.
            venues: Subset of venues to include (defaults to all).

        Returns:
            Dict with 'optimal_mean_bps', per-venue mean bps, and
            per-venue savings in bps from switching to optimal routing.
        """
        target_venues = venues or list(self.analyzer.models.keys())
        models = {v: self.analyzer.models[v] for v in target_venues if v in self.analyzer.models}

        vols = self.rng.uniform(0.005, 0.08, n_simulations)
        fundings = self.rng.uniform(-0.0003, 0.0005, n_simulations)

        venue_costs = {v: np.empty(n_simulations) for v in models}
        optimal_costs = np.empty(n_simulations)

        for i in range(n_simulations):
            trade = TradeSpec(
                notional_usd=base_trade.notional_usd,
                size_btc=base_trade.size_btc,
                btc_price=base_trade.btc_price,
                side=base_trade.side,
                order_type=base_trade.order_type,
                holding_period_hours=base_trade.holding_period_hours,
                volatility_daily=float(vols[i]),
                order_book_depth_usd=base_trade.order_book_depth_usd,
                funding_rate_8h=float(fundings[i]),
                days_to_roll=base_trade.days_to_roll,
                is_roll_trade=base_trade.is_roll_trade,
            )
            min_cost = float("inf")
            for v, model in models.items():
                bd = model.estimate_total_cost(trade)
                venue_costs[v][i] = bd.total_cost_bps
                if bd.total_cost_bps < min_cost:
                    min_cost = bd.total_cost_bps
            optimal_costs[i] = min_cost

        result: Dict[str, float] = {
            "optimal_mean_bps": float(np.mean(optimal_costs)),
        }
        for v in models:
            v_mean = float(np.mean(venue_costs[v]))
            result[f"{v}_mean_bps"] = v_mean
            result[f"{v}_savings_bps"] = v_mean - result["optimal_mean_bps"]

        return result
