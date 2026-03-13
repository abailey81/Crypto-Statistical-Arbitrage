"""
Single Venue Funding Rate Strategy
Implements funding rate arbitrage on a single exchange by going long spot and short perpetual.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class PositionSide(Enum):
    """Position side enumeration."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class FundingPosition:
    """Represents a funding rate arbitrage position."""
    symbol: str
    venue: str
    spot_size: float
    perp_size: float
    entry_time: pd.Timestamp
    entry_funding_rate: float
    entry_spot_price: float
    entry_perp_price: float

    @property
    def is_active(self) -> bool:
        return self.spot_size != 0 or self.perp_size != 0


class SingleVenueFundingStrategy:
    """
    Single venue funding rate arbitrage strategy.

    Strategy: Long spot + Short perpetual to capture positive funding rates.

    Entry conditions:
    - Annualized funding rate > threshold (e.g., 10%)
    - Sufficient liquidity in both spot and perp markets

    Exit conditions:
    - Funding rate drops below exit threshold
    - Position held for maximum duration
    - Stop-loss triggered
    """

    def __init__(
        self,
        venue: str = "binance",
        entry_threshold: float = 0.10,  # 10% annualized
        exit_threshold: float = 0.02,   # 2% annualized
        max_position_size: float = 100000,  # USD
        max_hold_days: int = 30,
        stop_loss_pct: float = 0.05,
        transaction_cost: float = 0.001  # 0.1% per side
    ):
        self.venue = venue
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_position_size = max_position_size
        self.max_hold_days = max_hold_days
        self.stop_loss_pct = stop_loss_pct
        self.transaction_cost = transaction_cost

        self.positions: Dict[str, FundingPosition] = {}
        self.trade_history: List[Dict] = []

    def annualize_funding_rate(
        self,
        funding_rate: float,
        funding_interval_hours: int = 8
    ) -> float:
        """Convert per-period funding rate to annualized rate."""
        periods_per_year = (365 * 24) / funding_interval_hours
        return funding_rate * periods_per_year

    def calculate_entry_signal(
        self,
        funding_rate: float,
        funding_interval_hours: int = 8,
        spot_volume_24h: float = 0,
        perp_volume_24h: float = 0,
        min_volume: float = 1_000_000
    ) -> Tuple[bool, str]:
        """
        Calculate entry signal for funding rate arbitrage.

        Returns:
            Tuple of (should_enter, reason)
        """
        annualized_rate = self.annualize_funding_rate(funding_rate, funding_interval_hours)

        # Check funding rate threshold
        if annualized_rate < self.entry_threshold:
            return False, f"Funding rate {annualized_rate:.2%} below threshold {self.entry_threshold:.2%}"

        # Check liquidity
        if spot_volume_24h < min_volume:
            return False, f"Spot volume ${spot_volume_24h:,.0f} below minimum ${min_volume:,.0f}"

        if perp_volume_24h < min_volume:
            return False, f"Perp volume ${perp_volume_24h:,.0f} below minimum ${min_volume:,.0f}"

        return True, f"Entry signal: {annualized_rate:.2%} annualized funding"

    def calculate_exit_signal(
        self,
        position: FundingPosition,
        current_funding_rate: float,
        current_spot_price: float,
        current_perp_price: float,
        current_time: pd.Timestamp,
        funding_interval_hours: int = 8
    ) -> Tuple[bool, str]:
        """
        Calculate exit signal for existing position.

        Returns:
            Tuple of (should_exit, reason)
        """
        annualized_rate = self.annualize_funding_rate(current_funding_rate, funding_interval_hours)

        # Check funding rate threshold
        if annualized_rate < self.exit_threshold:
            return True, f"Funding rate {annualized_rate:.2%} below exit threshold {self.exit_threshold:.2%}"

        # Check maximum hold duration
        days_held = (current_time - position.entry_time).days
        if days_held >= self.max_hold_days:
            return True, f"Maximum hold duration {self.max_hold_days} days reached"

        # Check stop loss
        spot_pnl_pct = (current_spot_price - position.entry_spot_price) / position.entry_spot_price
        perp_pnl_pct = (position.entry_perp_price - current_perp_price) / position.entry_perp_price
        total_pnl_pct = spot_pnl_pct + perp_pnl_pct

        if total_pnl_pct < -self.stop_loss_pct:
            return True, f"Stop loss triggered: {total_pnl_pct:.2%} loss"

        return False, "Position still valid"

    def calculate_position_pnl(
        self,
        position: FundingPosition,
        current_spot_price: float,
        current_perp_price: float,
        cumulative_funding_received: float
    ) -> Dict[str, float]:
        """Calculate P&L breakdown for a position."""
        # Price P&L
        spot_pnl = position.spot_size * (current_spot_price - position.entry_spot_price)
        perp_pnl = -position.perp_size * (current_perp_price - position.entry_perp_price)

        # Funding P&L
        funding_pnl = cumulative_funding_received

        # Transaction costs (entry only, exit not yet incurred)
        entry_costs = (
            abs(position.spot_size * position.entry_spot_price) * self.transaction_cost +
            abs(position.perp_size * position.entry_perp_price) * self.transaction_cost
        )

        return {
            "spot_pnl": spot_pnl,
            "perp_pnl": perp_pnl,
            "price_pnl": spot_pnl + perp_pnl,
            "funding_pnl": funding_pnl,
            "transaction_costs": entry_costs,
            "total_pnl": spot_pnl + perp_pnl + funding_pnl - entry_costs
        }

    def open_position(
        self,
        symbol: str,
        spot_price: float,
        perp_price: float,
        funding_rate: float,
        position_size_usd: float,
        timestamp: pd.Timestamp
    ) -> FundingPosition:
        """Open a new funding rate arbitrage position."""
        # Calculate position sizes
        spot_size = position_size_usd / spot_price
        perp_size = position_size_usd / perp_price

        position = FundingPosition(
            symbol=symbol,
            venue=self.venue,
            spot_size=spot_size,
            perp_size=perp_size,
            entry_time=timestamp,
            entry_funding_rate=funding_rate,
            entry_spot_price=spot_price,
            entry_perp_price=perp_price
        )

        self.positions[symbol] = position
        return position

    def close_position(
        self,
        symbol: str,
        spot_price: float,
        perp_price: float,
        cumulative_funding: float,
        timestamp: pd.Timestamp,
        reason: str
    ) -> Dict:
        """Close an existing position and record the trade."""
        position = self.positions.pop(symbol, None)
        if position is None:
            raise ValueError(f"No position found for {symbol}")

        pnl = self.calculate_position_pnl(
            position, spot_price, perp_price, cumulative_funding
        )

        # Add exit transaction costs
        exit_costs = (
            abs(position.spot_size * spot_price) * self.transaction_cost +
            abs(position.perp_size * perp_price) * self.transaction_cost
        )
        pnl["exit_costs"] = exit_costs
        pnl["total_pnl"] -= exit_costs

        trade_record = {
            "symbol": symbol,
            "venue": self.venue,
            "entry_time": position.entry_time,
            "exit_time": timestamp,
            "hold_days": (timestamp - position.entry_time).days,
            "entry_spot_price": position.entry_spot_price,
            "exit_spot_price": spot_price,
            "entry_perp_price": position.entry_perp_price,
            "exit_perp_price": perp_price,
            "position_size_usd": position.spot_size * position.entry_spot_price,
            "exit_reason": reason,
            **pnl
        }

        self.trade_history.append(trade_record)
        return trade_record

    def backtest(
        self,
        funding_data: pd.DataFrame,
        price_data: pd.DataFrame,
        symbols: List[str],
        initial_capital: float = 1_000_000
    ) -> pd.DataFrame:
        """
        Run backtest on historical data.

        Args:
            funding_data: DataFrame with columns [timestamp, symbol, funding_rate, venue]
            price_data: DataFrame with columns [timestamp, symbol, spot_price, perp_price, venue]
            symbols: List of symbols to trade
            initial_capital: Starting capital in USD

        Returns:
            DataFrame with daily portfolio values and metrics
        """
        # Implementation would iterate through data and apply strategy logic
        # This is a placeholder for the full backtest implementation
        raise NotImplementedError("Full backtest implementation pending")

    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics from trade history."""
        if not self.trade_history:
            return {}

        df = pd.DataFrame(self.trade_history)

        returns = df["total_pnl"] / df["position_size_usd"]

        return {
            "total_trades": len(df),
            "win_rate": (df["total_pnl"] > 0).mean(),
            "avg_return": returns.mean(),
            "total_return": returns.sum(),
            "avg_hold_days": df["hold_days"].mean(),
            "sharpe_ratio": returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 else 0,
            "max_drawdown": (returns.cumsum() - returns.cumsum().cummax()).min(),
            "avg_funding_pnl": df["funding_pnl"].mean(),
            "avg_price_pnl": df["price_pnl"].mean(),
        }
