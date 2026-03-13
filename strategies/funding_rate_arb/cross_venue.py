"""
Cross-Venue Funding Rate Arbitrage Strategy
Implements funding rate arbitrage across multiple venues (CEX, Hybrid, DEX).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class VenueType(Enum):
    """Venue type classification."""
    CEX = "cex"
    HYBRID = "hybrid"
    DEX = "dex"


@dataclass
class VenueConfig:
    """Configuration for a trading venue."""
    name: str
    venue_type: VenueType
    maker_fee: float
    taker_fee: float
    funding_interval_hours: int
    gas_cost_usd: float = 0  # For on-chain venues
    min_position_size: float = 100
    max_position_size: float = 1_000_000


# Default venue configurations
VENUE_CONFIGS = {
    "binance": VenueConfig(
        name="binance",
        venue_type=VenueType.CEX,
        maker_fee=0.0002,
        taker_fee=0.0004,
        funding_interval_hours=8
    ),
    "bybit": VenueConfig(
        name="bybit",
        venue_type=VenueType.CEX,
        maker_fee=0.0001,
        taker_fee=0.0006,
        funding_interval_hours=8
    ),
    "okx": VenueConfig(
        name="okx",
        venue_type=VenueType.CEX,
        maker_fee=0.0002,
        taker_fee=0.0005,
        funding_interval_hours=8
    ),
    "hyperliquid": VenueConfig(
        name="hyperliquid",
        venue_type=VenueType.HYBRID,
        maker_fee=0.0000,
        taker_fee=0.00025,
        funding_interval_hours=1,
        gas_cost_usd=0.50
    ),
    "dydx": VenueConfig(
        name="dydx",
        venue_type=VenueType.HYBRID,
        maker_fee=0.0000,
        taker_fee=0.0005,
        funding_interval_hours=1,
        gas_cost_usd=0.10
    ),
    "gmx": VenueConfig(
        name="gmx",
        venue_type=VenueType.DEX,
        maker_fee=0.001,
        taker_fee=0.001,
        funding_interval_hours=1,
        gas_cost_usd=2.00
    ),
}


@dataclass
class CrossVenuePosition:
    """Represents a cross-venue arbitrage position."""
    symbol: str
    long_venue: str
    short_venue: str
    long_size: float
    short_size: float
    entry_time: pd.Timestamp
    entry_long_price: float
    entry_short_price: float
    entry_long_funding: float
    entry_short_funding: float


class CrossVenueFundingArbitrage:
    """
    Cross-venue funding rate arbitrage strategy.

    Strategy: Exploit funding rate differentials between venues.
    - Long perpetual on venue with lower (or negative) funding
    - Short perpetual on venue with higher (positive) funding
    - Collect the funding rate differential

    Key considerations:
    - CEX-to-CEX: Traditional, high competition
    - CEX-to-Hybrid: Interesting opportunities, moderate complexity
    - Hybrid-to-Hybrid: Both on-chain, different mechanisms
    """

    def __init__(
        self,
        venues: List[str] = None,
        min_spread_annualized: float = 0.05,  # 5% minimum spread to enter
        exit_spread_annualized: float = 0.01,  # 1% to exit
        max_position_size: float = 100000,
        stop_loss_pct: float = 0.05
    ):
        self.venues = venues or list(VENUE_CONFIGS.keys())
        self.venue_configs = {v: VENUE_CONFIGS[v] for v in self.venues if v in VENUE_CONFIGS}
        self.min_spread_annualized = min_spread_annualized
        self.exit_spread_annualized = exit_spread_annualized
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct

        self.positions: Dict[str, CrossVenuePosition] = {}
        self.trade_history: List[Dict] = []

    def normalize_funding_rate(
        self,
        funding_rate: float,
        source_interval_hours: int,
        target_interval_hours: int = 8
    ) -> float:
        """Normalize funding rate to a common interval (default 8 hours)."""
        return funding_rate * (target_interval_hours / source_interval_hours)

    def annualize_funding_rate(
        self,
        funding_rate: float,
        interval_hours: int = 8
    ) -> float:
        """Convert per-period funding rate to annualized rate."""
        periods_per_year = (365 * 24) / interval_hours
        return funding_rate * periods_per_year

    def calculate_funding_spread(
        self,
        funding_rates: Dict[str, float],
        intervals: Dict[str, int]
    ) -> Dict[str, Dict]:
        """
        Calculate funding rate spreads between all venue pairs.

        Returns dict of {pair_key: {spread, long_venue, short_venue, annualized_spread}}
        """
        spreads = {}
        venues = list(funding_rates.keys())

        for i, venue_a in enumerate(venues):
            for venue_b in venues[i+1:]:
                # Normalize to 8-hour interval
                rate_a = self.normalize_funding_rate(
                    funding_rates[venue_a],
                    intervals.get(venue_a, 8)
                )
                rate_b = self.normalize_funding_rate(
                    funding_rates[venue_b],
                    intervals.get(venue_b, 8)
                )

                # Spread = higher rate - lower rate (we short the higher, long the lower)
                spread = abs(rate_a - rate_b)
                if rate_a > rate_b:
                    long_venue, short_venue = venue_b, venue_a
                else:
                    long_venue, short_venue = venue_a, venue_b

                pair_key = f"{venue_a}_{venue_b}"
                spreads[pair_key] = {
                    "spread": spread,
                    "annualized_spread": self.annualize_funding_rate(spread),
                    "long_venue": long_venue,
                    "short_venue": short_venue,
                    "long_funding": funding_rates[long_venue],
                    "short_funding": funding_rates[short_venue],
                }

        return spreads

    def calculate_transaction_costs(
        self,
        long_venue: str,
        short_venue: str,
        position_size_usd: float,
        is_entry: bool = True
    ) -> float:
        """Calculate total transaction costs for entering/exiting a position."""
        long_config = self.venue_configs.get(long_venue)
        short_config = self.venue_configs.get(short_venue)

        if not long_config or not short_config:
            raise ValueError(f"Unknown venue: {long_venue} or {short_venue}")

        # Use taker fees for simplicity (conservative estimate)
        long_fee = position_size_usd * long_config.taker_fee
        short_fee = position_size_usd * short_config.taker_fee

        # Add gas costs for on-chain venues
        gas_costs = long_config.gas_cost_usd + short_config.gas_cost_usd

        return long_fee + short_fee + gas_costs

    def calculate_entry_signal(
        self,
        symbol: str,
        funding_rates: Dict[str, float],
        intervals: Dict[str, int],
        prices: Dict[str, float]
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Calculate entry signal for cross-venue arbitrage.

        Returns:
            Tuple of (should_enter, opportunity_details)
        """
        spreads = self.calculate_funding_spread(funding_rates, intervals)

        # Find best opportunity
        best_spread = None
        best_details = None

        for pair_key, details in spreads.items():
            if details["annualized_spread"] >= self.min_spread_annualized:
                if best_spread is None or details["annualized_spread"] > best_spread:
                    best_spread = details["annualized_spread"]
                    best_details = details

        if best_details:
            # Calculate expected profit after costs
            tx_costs = self.calculate_transaction_costs(
                best_details["long_venue"],
                best_details["short_venue"],
                self.max_position_size
            )
            tx_cost_pct = tx_costs / self.max_position_size

            # Need spread to exceed transaction costs
            if best_spread / 365 * 30 > tx_cost_pct * 2:  # Expect 30 day hold, need 2x costs
                return True, {
                    "symbol": symbol,
                    **best_details,
                    "transaction_costs": tx_costs,
                    "tx_cost_pct": tx_cost_pct
                }

        return False, None

    def calculate_exit_signal(
        self,
        position: CrossVenuePosition,
        current_funding_rates: Dict[str, float],
        intervals: Dict[str, int],
        current_prices: Dict[str, float],
        current_time: pd.Timestamp
    ) -> Tuple[bool, str]:
        """Calculate exit signal for existing position."""
        long_rate = current_funding_rates.get(position.long_venue, 0)
        short_rate = current_funding_rates.get(position.short_venue, 0)

        # Normalize and calculate spread
        long_rate_norm = self.normalize_funding_rate(
            long_rate,
            intervals.get(position.long_venue, 8)
        )
        short_rate_norm = self.normalize_funding_rate(
            short_rate,
            intervals.get(position.short_venue, 8)
        )

        current_spread = short_rate_norm - long_rate_norm  # We're short the higher rate
        annualized_spread = self.annualize_funding_rate(current_spread)

        # Check if spread has compressed
        if annualized_spread < self.exit_spread_annualized:
            return True, f"Spread compressed to {annualized_spread:.2%}"

        # Check if spread has inverted (we're losing on funding)
        if current_spread < 0:
            return True, f"Spread inverted: {annualized_spread:.2%}"

        # Check position P&L
        long_price = current_prices.get(position.long_venue, position.entry_long_price)
        short_price = current_prices.get(position.short_venue, position.entry_short_price)

        long_pnl_pct = (long_price - position.entry_long_price) / position.entry_long_price
        short_pnl_pct = (position.entry_short_price - short_price) / position.entry_short_price
        total_pnl_pct = long_pnl_pct + short_pnl_pct

        if total_pnl_pct < -self.stop_loss_pct:
            return True, f"Stop loss triggered: {total_pnl_pct:.2%}"

        return False, "Position valid"

    def open_position(
        self,
        symbol: str,
        opportunity: Dict,
        timestamp: pd.Timestamp,
        prices: Dict[str, float]
    ) -> CrossVenuePosition:
        """Open a cross-venue arbitrage position."""
        long_venue = opportunity["long_venue"]
        short_venue = opportunity["short_venue"]

        long_price = prices.get(long_venue, prices.get(list(prices.keys())[0]))
        short_price = prices.get(short_venue, prices.get(list(prices.keys())[0]))

        position_size = self.max_position_size
        long_size = position_size / long_price
        short_size = position_size / short_price

        position = CrossVenuePosition(
            symbol=symbol,
            long_venue=long_venue,
            short_venue=short_venue,
            long_size=long_size,
            short_size=short_size,
            entry_time=timestamp,
            entry_long_price=long_price,
            entry_short_price=short_price,
            entry_long_funding=opportunity["long_funding"],
            entry_short_funding=opportunity["short_funding"]
        )

        self.positions[symbol] = position
        return position

    def analyze_historical_opportunities(
        self,
        funding_data: pd.DataFrame,
        price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze historical cross-venue funding rate opportunities.

        Args:
            funding_data: DataFrame with [timestamp, symbol, venue, funding_rate]
            price_data: DataFrame with [timestamp, symbol, venue, price]

        Returns:
            DataFrame with opportunity analysis
        """
        # Pivot funding data to have venues as columns
        funding_pivot = funding_data.pivot_table(
            index=["timestamp", "symbol"],
            columns="venue",
            values="funding_rate"
        ).reset_index()

        opportunities = []

        for _, row in funding_pivot.iterrows():
            timestamp = row["timestamp"]
            symbol = row["symbol"]

            # Get funding rates for available venues
            rates = {}
            intervals = {}
            for venue in self.venues:
                if venue in row.index and pd.notna(row[venue]):
                    rates[venue] = row[venue]
                    intervals[venue] = self.venue_configs[venue].funding_interval_hours

            if len(rates) >= 2:
                spreads = self.calculate_funding_spread(rates, intervals)

                for pair_key, details in spreads.items():
                    opportunities.append({
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "pair": pair_key,
                        "long_venue": details["long_venue"],
                        "short_venue": details["short_venue"],
                        "spread_8h": details["spread"],
                        "spread_annualized": details["annualized_spread"],
                        "is_tradeable": details["annualized_spread"] >= self.min_spread_annualized
                    })

        return pd.DataFrame(opportunities)

    def get_venue_comparison(self) -> pd.DataFrame:
        """Get comparison of venue characteristics."""
        data = []
        for name, config in self.venue_configs.items():
            data.append({
                "venue": name,
                "type": config.venue_type.value,
                "maker_fee": config.maker_fee,
                "taker_fee": config.taker_fee,
                "funding_interval_hours": config.funding_interval_hours,
                "gas_cost_usd": config.gas_cost_usd,
                "min_position": config.min_position_size,
                "max_position": config.max_position_size
            })
        return pd.DataFrame(data)
