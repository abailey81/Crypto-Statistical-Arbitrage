"""
Strategy C: Synthetic Futures from Perp Funding
================================================

Part 2 Section 3.2.3 - Synthetic Futures Implementation

Create synthetic futures positions using multiple perpetual contracts
to replicate futures exposure at lower implied carry cost.

Part 2 Requirements Addressed:
- 3.1.3 Synthetic term structure from perpetual funding rates
- 3.2.3 Strategy C: Synthetic Futures from Perp Funding (MANDATORY)
- Integration with Strategy A (calendar spreads) for combined signals
- Cross-venue funding arbitrage with z-score triggers
- Crisis period handling (COVID, Luna, FTX, May 2021)

Mathematical Framework
----------------------
Synthetic Futures Position:

    Instead of buying expensive futures (high contango basis):

    1. Long Perpetual A (lower funding venue)
    2. Short Perpetual B (higher funding venue)

    Net BTC Exposure: 0 (delta neutral)
    Net Carry: Funding_B - Funding_A - Costs

Implied Carry Cost Comparison:

    Futures Carry Cost = Basis_Annual / 100

    Perp Carry Cost = Funding_Rate_Annual / 100

    Savings = Futures_Carry - Perp_Carry - Transaction_Costs

Position Replication:

    To replicate long 1 BTC futures exposure via perps:

    If Funding_A < Funding_B:
        Long 1 BTC on Venue A (pay less funding)

    Effective Futures Price:
        F_synthetic = Spot × (1 + Funding_A_annual × T/365)

Cross-Venue Synthetic:

    Long Perp A (low funding) + Short Perp B (high funding)

    Net Funding = Funding_B - Funding_A (positive = profit)

    BTC Exposure: Long A - Short B = 0 (if equal size)

Risk-Adjusted Sizing:

    Size = Base × min(Funding_Spread / Target_Spread, 1.0)
         × Liquidity_Factor × Confidence

Version: 2.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import uuid
import logging

from . import (
    TermStructureRegime, VenueType, SpreadDirection, ExitReason,
    VenueCosts, DEFAULT_VENUE_COSTS, DEFAULT_VENUE_CAPACITY
)
from .funding_rate_analysis import (
    FundingTermStructure, FundingRateAnalyzer, FundingRegime,
    VENUE_FUNDING_CONFIG, CrossVenueFundingSpread,
    FundingTermStructureIntegration, CRISIS_EVENTS, is_crisis_period
)

logger = logging.getLogger(__name__)


# =============================================================================
# CRISIS-ADAPTIVE PARAMETERS FOR STRATEGY C
# =============================================================================

STRATEGY_C_CRISIS_PARAMS = {
    'normal': {
        'min_funding_spread_mult': 1.0,
        'position_size_mult': 1.0,
        'max_holding_mult': 1.0,
        'stop_loss_mult': 1.0,
    },
    'crisis': {
        'min_funding_spread_mult': 2.0,  # Higher threshold
        'position_size_mult': 0.25,  # Reduce size significantly
        'max_holding_mult': 0.5,  # Exit faster
        'stop_loss_mult': 0.5,  # Tighter stops
    }
}

# Regime-adaptive parameters
FUNDING_REGIME_PARAMS = {
    FundingRegime.EXTREMELY_POSITIVE: {
        'preferred_direction': 'short_perp',  # Receive funding
        'entry_threshold_mult': 0.8,  # Lower threshold (stronger signal)
        'confidence_boost': 0.1,
    },
    FundingRegime.HIGHLY_POSITIVE: {
        'preferred_direction': 'short_perp',
        'entry_threshold_mult': 0.9,
        'confidence_boost': 0.05,
    },
    FundingRegime.MODERATELY_POSITIVE: {
        'preferred_direction': 'neutral',
        'entry_threshold_mult': 1.0,
        'confidence_boost': 0.0,
    },
    FundingRegime.NEUTRAL: {
        'preferred_direction': 'neutral',
        'entry_threshold_mult': 1.2,  # Higher threshold
        'confidence_boost': -0.05,
    },
    FundingRegime.MODERATELY_NEGATIVE: {
        'preferred_direction': 'neutral',
        'entry_threshold_mult': 1.0,
        'confidence_boost': 0.0,
    },
    FundingRegime.HIGHLY_NEGATIVE: {
        'preferred_direction': 'long_perp',  # Receive funding
        'entry_threshold_mult': 0.9,
        'confidence_boost': 0.05,
    },
    FundingRegime.EXTREMELY_NEGATIVE: {
        'preferred_direction': 'long_perp',
        'entry_threshold_mult': 0.8,
        'confidence_boost': 0.1,
    },
}


# =============================================================================
# SYNTHETIC FUTURES ENUMERATIONS
# =============================================================================

class SyntheticType(Enum):
    """Type of synthetic futures position."""
    SINGLE_VENUE_LONG = "single_venue_long"      # Long perp on one venue
    SINGLE_VENUE_SHORT = "single_venue_short"    # Short perp on one venue
    CROSS_VENUE_SPREAD = "cross_venue_spread"    # Long one, short another
    FUNDING_HARVEST = "funding_harvest"          # Pure funding collection
    CARRY_REPLICATION = "carry_replication"      # Replicate futures carry

    @property
    def is_delta_neutral(self) -> bool:
        return self in [self.CROSS_VENUE_SPREAD, self.FUNDING_HARVEST]

    @property
    def requires_multiple_venues(self) -> bool:
        return self in [self.CROSS_VENUE_SPREAD, self.FUNDING_HARVEST]

    @property
    def max_leverage(self) -> float:
        max_lev = {
            self.SINGLE_VENUE_LONG: 3.0,
            self.SINGLE_VENUE_SHORT: 3.0,
            self.CROSS_VENUE_SPREAD: 2.0,
            self.FUNDING_HARVEST: 2.0,
            self.CARRY_REPLICATION: 2.0,
        }
        return max_lev.get(self, 2.0)


class SyntheticSignalType(Enum):
    """Signal type for synthetic positions."""
    ENTRY = "entry"
    EXIT_PROFIT = "exit_profit"
    EXIT_STOP = "exit_stop"
    EXIT_FUNDING_FLIP = "exit_funding_flip"
    EXIT_MAX_HOLD = "exit_max_hold"
    EXIT_REBALANCE = "exit_rebalance"
    HOLD = "hold"
    SKIP = "skip"

    @property
    def is_entry(self) -> bool:
        return self == self.ENTRY

    @property
    def is_exit(self) -> bool:
        return self in [
            self.EXIT_PROFIT, self.EXIT_STOP, self.EXIT_FUNDING_FLIP,
            self.EXIT_MAX_HOLD, self.EXIT_REBALANCE
        ]

    @property
    def exit_reason(self) -> Optional[ExitReason]:
        mapping = {
            self.EXIT_PROFIT: ExitReason.PROFIT_TARGET,
            self.EXIT_STOP: ExitReason.STOP_LOSS,
            self.EXIT_MAX_HOLD: ExitReason.MAX_HOLD,
            self.EXIT_FUNDING_FLIP: ExitReason.REGIME_CHANGE,
            self.EXIT_REBALANCE: ExitReason.REGIME_CHANGE,
        }
        return mapping.get(self)


# =============================================================================
# SYNTHETIC FUTURES DATA STRUCTURES
# =============================================================================

@dataclass
class SyntheticFuturesConfig:
    """Configuration for synthetic futures strategy."""
    # Entry thresholds
    min_funding_spread_annual_pct: float = 10.0  # Min spread to enter
    min_z_score: float = 1.5
    min_confidence: float = 0.5

    # Position sizing
    max_position_pct: float = 0.25
    max_positions: int = 5
    min_position_usd: float = 10_000
    max_position_usd: float = 500_000
    max_venue_exposure_pct: float = 0.40

    # Exit thresholds
    profit_target_pct: float = 5.0
    stop_loss_pct: float = 3.0
    max_holding_days: int = 30
    funding_flip_exit: bool = True

    # Risk management
    max_leverage: float = 2.0  # PDF: 2.0x max per PDF Section 3.2 (Hyperliquid: 1.5x max)
    max_drawdown_pct: float = 10.0
    correlation_threshold: float = 0.8

    # Venues
    allowed_venues: List[str] = field(default_factory=lambda: [
        'binance', 'bybit', 'okx', 'hyperliquid', 'dydx', 'gmx'
    ])
    preferred_long_venues: List[str] = field(default_factory=lambda: [
        'hyperliquid', 'dydx'  # Lower funding typically
    ])
    preferred_short_venues: List[str] = field(default_factory=lambda: [
        'binance', 'bybit'  # Higher funding typically
    ])


@dataclass
class SyntheticSignal:
    """Signal for synthetic futures position."""
    timestamp: pd.Timestamp
    signal_type: SyntheticSignalType
    synthetic_type: SyntheticType
    venue_long: Optional[str]
    venue_short: Optional[str]
    funding_long_annual_pct: float
    funding_short_annual_pct: float
    gross_spread_annual_pct: float
    net_spread_annual_pct: float
    z_score: float
    signal_strength: float
    confidence: float
    recommended_size_usd: float
    expected_daily_pnl_bps: float
    reason: str

    @property
    def is_actionable(self) -> bool:
        return (
            self.signal_type.is_entry and
            self.signal_strength >= 0.3 and
            self.confidence >= 0.5
        )

    @property
    def direction(self) -> SpreadDirection:
        if self.synthetic_type == SyntheticType.SINGLE_VENUE_LONG:
            return SpreadDirection.LONG
        elif self.synthetic_type == SyntheticType.SINGLE_VENUE_SHORT:
            return SpreadDirection.SHORT
        return SpreadDirection.FLAT  # Delta neutral

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'signal_type': self.signal_type.value,
            'synthetic_type': self.synthetic_type.value,
            'venue_long': self.venue_long,
            'venue_short': self.venue_short,
            'gross_spread_annual_pct': round(self.gross_spread_annual_pct, 2),
            'net_spread_annual_pct': round(self.net_spread_annual_pct, 2),
            'z_score': round(self.z_score, 2),
            'signal_strength': round(self.signal_strength, 2),
            'recommended_size_usd': self.recommended_size_usd,
            'reason': self.reason,
        }


@dataclass
class SyntheticPosition:
    """Active synthetic futures position."""
    position_id: str
    entry_time: pd.Timestamp
    synthetic_type: SyntheticType
    venue_long: Optional[str]
    venue_short: Optional[str]
    venue_long_type: Optional[VenueType]
    venue_short_type: Optional[VenueType]

    # Entry metrics
    entry_funding_long: float
    entry_funding_short: float
    entry_spread_annual_pct: float
    entry_spot_price: float

    # Position sizing
    size_usd: float
    leverage: float
    margin_used: float

    # Exit tracking
    exit_time: Optional[pd.Timestamp] = None
    exit_reason: Optional[ExitReason] = None
    exit_funding_long: Optional[float] = None
    exit_funding_short: Optional[float] = None
    exit_spread_annual_pct: Optional[float] = None
    exit_spot_price: Optional[float] = None

    # P&L tracking
    cumulative_funding_pnl: float = 0.0
    cumulative_basis_pnl: float = 0.0
    transaction_costs: float = 0.0
    gas_costs: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

    # History
    funding_history: List[Dict] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    @property
    def is_closed(self) -> bool:
        return self.exit_time is not None

    @property
    def holding_period_days(self) -> float:
        if self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds() / 86400
        return 0.0

    @property
    def gross_pnl(self) -> float:
        return self.cumulative_funding_pnl + self.cumulative_basis_pnl

    @property
    def total_costs(self) -> float:
        return self.transaction_costs + self.gas_costs

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.total_costs

    @property
    def return_pct(self) -> float:
        if self.size_usd <= 0:
            return 0.0
        return (self.net_pnl / self.size_usd) * 100

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0

    @property
    def annualized_return_pct(self) -> float:
        if self.holding_period_days <= 0:
            return 0.0
        return self.return_pct * (365 / self.holding_period_days)

    @property
    def efficiency(self) -> float:
        """P&L / Maximum favorable excursion."""
        if self.max_favorable_excursion <= 0:
            return 0.0
        return self.net_pnl / self.max_favorable_excursion

    def update_mtm(
        self,
        timestamp: pd.Timestamp,
        funding_long: float,
        funding_short: float,
        spot_price: float
    ):
        """Update mark-to-market with new funding data."""
        # Calculate funding P&L for this period
        # Assume 8-hour periods for funding
        period_funding = (funding_short - funding_long) * self.size_usd

        self.cumulative_funding_pnl += period_funding

        # Track basis P&L (spot price changes)
        if self.synthetic_type not in [SyntheticType.CROSS_VENUE_SPREAD, SyntheticType.FUNDING_HARVEST]:
            # Directional position has spot exposure
            price_change_pct = (spot_price - self.entry_spot_price) / self.entry_spot_price
            if self.synthetic_type == SyntheticType.SINGLE_VENUE_LONG:
                self.cumulative_basis_pnl = price_change_pct * self.size_usd
            elif self.synthetic_type == SyntheticType.SINGLE_VENUE_SHORT:
                self.cumulative_basis_pnl = -price_change_pct * self.size_usd

        # Track excursions
        current_pnl = self.gross_pnl
        self.max_favorable_excursion = max(self.max_favorable_excursion, current_pnl)
        self.max_adverse_excursion = min(self.max_adverse_excursion, current_pnl)

        # Record history
        self.funding_history.append({
            'timestamp': timestamp,
            'funding_long': funding_long,
            'funding_short': funding_short,
            'period_funding_pnl': period_funding,
            'cumulative_funding_pnl': self.cumulative_funding_pnl,
            'spot_price': spot_price,
            'total_pnl': self.gross_pnl,
        })

    def close(
        self,
        exit_time: pd.Timestamp,
        exit_reason: ExitReason,
        funding_long: float,
        funding_short: float,
        spot_price: float,
        exit_costs: float
    ):
        """Close the position."""
        self.exit_time = exit_time
        self.exit_reason = exit_reason
        self.exit_funding_long = funding_long
        self.exit_funding_short = funding_short
        self.exit_spread_annual_pct = funding_short - funding_long
        self.exit_spot_price = spot_price
        self.transaction_costs += exit_costs

    def to_dict(self) -> Dict[str, Any]:
        return {
            'position_id': self.position_id,
            'synthetic_type': self.synthetic_type.value,
            'venue_long': self.venue_long,
            'venue_short': self.venue_short,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'size_usd': self.size_usd,
            'entry_spread_pct': round(self.entry_spread_annual_pct, 2),
            'cumulative_funding_pnl': round(self.cumulative_funding_pnl, 2),
            'net_pnl': round(self.net_pnl, 2),
            'return_pct': round(self.return_pct, 2),
            'holding_days': round(self.holding_period_days, 1),
            'exit_reason': self.exit_reason.value if self.exit_reason else None,
        }


@dataclass
class SyntheticBacktestResult:
    """Results from synthetic futures backtest."""
    positions: List[SyntheticPosition]
    signals: List[SyntheticSignal]
    equity_curve: pd.DataFrame
    initial_capital: float
    final_capital: float

    # Metrics
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0

    # Funding statistics
    total_funding_collected: float = 0.0
    total_costs: float = 0.0
    avg_holding_days: float = 0.0

    # Venue breakdown
    venue_stats: Dict[str, Dict] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate metrics after initialization."""
        self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculate all performance metrics."""
        closed = [p for p in self.positions if p.is_closed]

        if not closed:
            return

        self.total_trades = len(closed)
        self.winning_trades = len([p for p in closed if p.is_winner])
        self.losing_trades = self.total_trades - self.winning_trades

        self.win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0

        winners = [p.net_pnl for p in closed if p.is_winner]
        losers = [p.net_pnl for p in closed if not p.is_winner]

        self.avg_win = np.mean(winners) if winners else 0.0
        self.avg_loss = np.mean(losers) if losers else 0.0

        gross_wins = sum(winners)
        gross_losses = abs(sum(losers))
        self.profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

        self.expectancy = (self.win_rate / 100) * self.avg_win + (1 - self.win_rate / 100) * self.avg_loss

        self.total_funding_collected = sum(p.cumulative_funding_pnl for p in closed)
        self.total_costs = sum(p.total_costs for p in closed)
        self.avg_holding_days = np.mean([p.holding_period_days for p in closed])

        # Calculate equity curve metrics
        if not self.equity_curve.empty and 'equity' in self.equity_curve.columns:
            returns = self.equity_curve['equity'].pct_change().dropna()

            if len(returns) > 5 and returns.std() > 0:
                self.sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365 * 3)

                downside = returns[returns < 0]
                if len(downside) > 0 and downside.std() > 0:
                    self.sortino_ratio = returns.mean() / downside.std() * np.sqrt(365 * 3)

            peak = self.equity_curve['equity'].expanding().max()
            drawdown = (self.equity_curve['equity'] - peak) / peak * 100
            self.max_drawdown_pct = abs(drawdown.min())

            if self.max_drawdown_pct > 0:
                ann_return = self.total_return_pct * (365 / max(len(self.equity_curve), 1))
                self.calmar_ratio = ann_return / self.max_drawdown_pct

        # Venue breakdown
        for p in closed:
            for v in [p.venue_long, p.venue_short]:
                if v and v not in self.venue_stats:
                    self.venue_stats[v] = {
                        'trades': 0,
                        'volume': 0.0,
                        'pnl': 0.0,
                        'funding_collected': 0.0,
                    }
                if v:
                    self.venue_stats[v]['trades'] += 1
                    self.venue_stats[v]['volume'] += p.size_usd
                    self.venue_stats[v]['pnl'] += p.net_pnl / 2  # Split between venues
                    self.venue_stats[v]['funding_collected'] += p.cumulative_funding_pnl / 2

    def summary(self) -> Dict[str, Any]:
        return {
            'total_pnl': round(self.total_pnl, 2),
            'total_return_pct': round(self.total_return_pct, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'sortino_ratio': round(self.sortino_ratio, 2),
            'max_drawdown_pct': round(self.max_drawdown_pct, 2),
            'calmar_ratio': round(self.calmar_ratio, 2),
            'total_trades': self.total_trades,
            'win_rate': round(self.win_rate, 1),
            'profit_factor': round(self.profit_factor, 2),
            'expectancy': round(self.expectancy, 2),
            'total_funding_collected': round(self.total_funding_collected, 2),
            'total_costs': round(self.total_costs, 2),
            'avg_holding_days': round(self.avg_holding_days, 1),
        }

    def positions_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([p.to_dict() for p in self.positions])

    def signals_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([s.to_dict() for s in self.signals])


# =============================================================================
# SYNTHETIC FUTURES STRATEGY
# =============================================================================

class SyntheticFuturesStrategy:
    """
    Strategy C: Synthetic Futures from Perp Funding.

    Creates synthetic futures exposure using perpetual contracts
    to achieve lower implied carry costs than actual futures.

    Key Features:
    - Multi-venue funding comparison
    - Cross-venue spread trading
    - Dynamic position sizing
    - Funding flip detection
    - Risk management

    Usage:
        config = SyntheticFuturesConfig(min_funding_spread_annual_pct=10.0)
        strategy = SyntheticFuturesStrategy(config)
        result = strategy.run_backtest(funding_data, spot_prices)
    """

    def __init__(
        self,
        config: Optional[SyntheticFuturesConfig] = None,
        funding_analyzer: Optional[FundingRateAnalyzer] = None
    ):
        """
        Initialize strategy.

        Args:
            config: Strategy configuration
            funding_analyzer: Funding rate analyzer instance
        """
        self.config = config or SyntheticFuturesConfig()
        self.analyzer = funding_analyzer or FundingRateAnalyzer(
            venues=self.config.allowed_venues,
            min_spread_bps=self.config.min_funding_spread_annual_pct * 100,
            min_z_score=self.config.min_z_score,
        )

        # State
        self._open_positions: Dict[str, SyntheticPosition] = {}
        self._spread_history: Dict[Tuple[str, str], List[float]] = {}

        # Crisis state tracking
        self._current_crisis: Optional[str] = None
        self._crisis_severity: float = 0.0

        # Integration with term structure for Strategy A synergy
        self._term_structure_integration: Optional[FundingTermStructureIntegration] = None

        logger.info(f"SyntheticFuturesStrategy initialized with {len(self.config.allowed_venues)} venues")

    def integrate_term_structure(self, integration: FundingTermStructureIntegration) -> None:
        """
        Integrate with term structure for Strategy A/C synergy.

        Args:
            integration: FundingTermStructureIntegration instance
        """
        self._term_structure_integration = integration
        logger.info("Integrated with FundingTermStructureIntegration")

    def check_crisis_state(self, timestamp: pd.Timestamp) -> Tuple[bool, Optional[str], float]:
        """
        Check if current timestamp is in a crisis period.

        Args:
            timestamp: Current timestamp

        Returns:
            Tuple of (is_crisis, crisis_name, severity)
        """
        is_crisis, name, severity = is_crisis_period(timestamp)

        self._current_crisis = name if is_crisis else None
        self._crisis_severity = severity

        if is_crisis:
            logger.warning(f"Crisis period detected: {name} (severity={severity})")

        return is_crisis, name, severity

    def get_crisis_adjusted_config(
        self,
        timestamp: pd.Timestamp
    ) -> SyntheticFuturesConfig:
        """
        Get crisis-adjusted configuration.

        Args:
            timestamp: Current timestamp

        Returns:
            Adjusted configuration
        """
        is_crisis, _, severity = self.check_crisis_state(timestamp)

        if not is_crisis:
            return self.config

        # Create adjusted config
        params = STRATEGY_C_CRISIS_PARAMS['crisis']

        adjusted = SyntheticFuturesConfig(
            min_funding_spread_annual_pct=(
                self.config.min_funding_spread_annual_pct *
                params['min_funding_spread_mult']
            ),
            min_z_score=self.config.min_z_score * 1.5,  # Higher z-score in crisis
            min_confidence=min(self.config.min_confidence * 1.2, 0.8),
            max_position_pct=self.config.max_position_pct * params['position_size_mult'],
            max_positions=max(1, self.config.max_positions // 2),
            max_holding_days=int(
                self.config.max_holding_days * params['max_holding_mult']
            ),
            stop_loss_pct=self.config.stop_loss_pct * params['stop_loss_mult'],
            profit_target_pct=self.config.profit_target_pct * 0.75,
            allowed_venues=self.config.allowed_venues,
        )

        return adjusted

    def get_regime_adjusted_threshold(
        self,
        base_threshold: float,
        funding_regime: FundingRegime
    ) -> float:
        """
        Adjust entry threshold based on funding regime.

        Args:
            base_threshold: Base threshold value
            funding_regime: Current funding regime

        Returns:
            Adjusted threshold
        """
        params = FUNDING_REGIME_PARAMS.get(
            funding_regime,
            FUNDING_REGIME_PARAMS[FundingRegime.NEUTRAL]
        )
        return base_threshold * params['entry_threshold_mult']

    def compare_synthetic_vs_actual_futures(
        self,
        funding_ts: FundingTermStructure,
        actual_futures: Dict[int, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Compare synthetic funding-implied prices to actual futures.

        This supports the integration between Strategy A and C.

        Args:
            funding_ts: Funding-implied term structure
            actual_futures: Actual futures prices by DTE

        Returns:
            Comparison analysis with trade recommendations
        """
        if not self._term_structure_integration:
            self._term_structure_integration = FundingTermStructureIntegration(
                self.analyzer
            )

        return self._term_structure_integration.compare_synthetic_vs_actual(
            funding_ts, actual_futures
        )

    def generate_signal(
        self,
        timestamp: pd.Timestamp,
        term_structures: Dict[str, FundingTermStructure],
        spot_price: float,
        current_capital: float,
        open_position_ids: Optional[List[str]] = None
    ) -> List[SyntheticSignal]:
        """
        Generate trading signals from current funding data.

        Args:
            timestamp: Current timestamp
            term_structures: Funding term structures by venue
            spot_price: Current spot price
            current_capital: Available capital
            open_position_ids: IDs of open positions

        Returns:
            List of SyntheticSignal objects
        """
        signals = []

        # Check for exits first
        for pos_id, pos in self._open_positions.items():
            exit_signal = self._check_exit_conditions(
                timestamp, pos, term_structures, spot_price
            )
            if exit_signal:
                signals.append(exit_signal)

        # Find entry opportunities
        if len(self._open_positions) < self.config.max_positions:
            entry_signals = self._find_entry_opportunities(
                timestamp, term_structures, spot_price, current_capital
            )
            signals.extend(entry_signals)

        return signals

    def _check_exit_conditions(
        self,
        timestamp: pd.Timestamp,
        position: SyntheticPosition,
        term_structures: Dict[str, FundingTermStructure],
        spot_price: float
    ) -> Optional[SyntheticSignal]:
        """Check if position should be exited."""
        # Get current funding rates
        funding_long = 0.0
        funding_short = 0.0

        if position.venue_long and position.venue_long in term_structures:
            funding_long = term_structures[position.venue_long].annual_funding_pct

        if position.venue_short and position.venue_short in term_structures:
            funding_short = term_structures[position.venue_short].annual_funding_pct

        current_spread = funding_short - funding_long

        # Update position MTM
        position.update_mtm(timestamp, funding_long / 100, funding_short / 100, spot_price)

        # Check holding period
        holding_days = (timestamp - position.entry_time).total_seconds() / 86400
        if holding_days >= self.config.max_holding_days:
            return self._create_exit_signal(
                timestamp, position, SyntheticSignalType.EXIT_MAX_HOLD,
                funding_long, funding_short, current_spread,
                f"Max holding period ({self.config.max_holding_days} days) reached"
            )

        # Check profit target
        if position.return_pct >= self.config.profit_target_pct:
            return self._create_exit_signal(
                timestamp, position, SyntheticSignalType.EXIT_PROFIT,
                funding_long, funding_short, current_spread,
                f"Profit target ({self.config.profit_target_pct}%) reached"
            )

        # Check stop loss
        if position.return_pct <= -self.config.stop_loss_pct:
            return self._create_exit_signal(
                timestamp, position, SyntheticSignalType.EXIT_STOP,
                funding_long, funding_short, current_spread,
                f"Stop loss ({self.config.stop_loss_pct}%) triggered"
            )

        # Check funding flip
        if self.config.funding_flip_exit:
            # Exit if spread flipped sign
            if position.entry_spread_annual_pct > 0 and current_spread < 0:
                return self._create_exit_signal(
                    timestamp, position, SyntheticSignalType.EXIT_FUNDING_FLIP,
                    funding_long, funding_short, current_spread,
                    "Funding spread flipped negative"
                )
            elif position.entry_spread_annual_pct < 0 and current_spread > 0:
                return self._create_exit_signal(
                    timestamp, position, SyntheticSignalType.EXIT_FUNDING_FLIP,
                    funding_long, funding_short, current_spread,
                    "Funding spread flipped positive"
                )

        return None

    def _create_exit_signal(
        self,
        timestamp: pd.Timestamp,
        position: SyntheticPosition,
        signal_type: SyntheticSignalType,
        funding_long: float,
        funding_short: float,
        current_spread: float,
        reason: str
    ) -> SyntheticSignal:
        """Create exit signal for position."""
        return SyntheticSignal(
            timestamp=timestamp,
            signal_type=signal_type,
            synthetic_type=position.synthetic_type,
            venue_long=position.venue_long,
            venue_short=position.venue_short,
            funding_long_annual_pct=funding_long,
            funding_short_annual_pct=funding_short,
            gross_spread_annual_pct=current_spread,
            net_spread_annual_pct=current_spread,  # Costs already paid
            z_score=0.0,
            signal_strength=1.0,
            confidence=1.0,
            recommended_size_usd=position.size_usd,
            expected_daily_pnl_bps=0.0,
            reason=reason,
        )

    def _find_entry_opportunities(
        self,
        timestamp: pd.Timestamp,
        term_structures: Dict[str, FundingTermStructure],
        spot_price: float,
        current_capital: float
    ) -> List[SyntheticSignal]:
        """Find entry opportunities across venue pairs."""
        signals = []

        venues = list(term_structures.keys())

        # Check all venue pairs
        for i, v1 in enumerate(venues):
            if v1 not in self.config.allowed_venues:
                continue

            for v2 in venues[i+1:]:
                if v2 not in self.config.allowed_venues:
                    continue

                # Skip if already have this pair
                pair_key = tuple(sorted([v1, v2]))
                if any(
                    tuple(sorted([p.venue_long, p.venue_short])) == pair_key
                    for p in self._open_positions.values()
                    if p.venue_long and p.venue_short
                ):
                    continue

                ts1 = term_structures[v1]
                ts2 = term_structures[v2]

                # Calculate spread
                spread = self.analyzer.calculate_cross_venue_spread(ts1, ts2)

                if not spread.is_profitable:
                    continue

                if spread.net_spread_bps < self.config.min_funding_spread_annual_pct * 100:
                    continue

                if abs(spread.z_score) < self.config.min_z_score:
                    continue

                if spread.confidence < self.config.min_confidence:
                    continue

                # Calculate position size
                max_size = current_capital * self.config.max_position_pct

                # Check venue capacity
                cap1 = DEFAULT_VENUE_CAPACITY.get(v1.lower(), 100_000_000)
                cap2 = DEFAULT_VENUE_CAPACITY.get(v2.lower(), 100_000_000)
                venue_max = min(cap1, cap2) * 0.01

                size = min(max_size, venue_max, self.config.max_position_usd)
                size = max(size, self.config.min_position_usd)

                # Adjust by signal strength
                strength = min(abs(spread.z_score) / 3, 1.0) * spread.confidence
                size *= strength

                if size < self.config.min_position_usd:
                    continue

                signals.append(SyntheticSignal(
                    timestamp=timestamp,
                    signal_type=SyntheticSignalType.ENTRY,
                    synthetic_type=SyntheticType.CROSS_VENUE_SPREAD,
                    venue_long=spread.venue_long,
                    venue_short=spread.venue_short,
                    funding_long_annual_pct=spread.funding_long,
                    funding_short_annual_pct=spread.funding_short,
                    gross_spread_annual_pct=spread.spread_annual_pct,
                    net_spread_annual_pct=spread.net_spread_bps / 100,
                    z_score=spread.z_score,
                    signal_strength=strength,
                    confidence=spread.confidence,
                    recommended_size_usd=size,
                    expected_daily_pnl_bps=spread.expected_daily_return_bps,
                    reason=f"Funding spread {spread.spread_annual_pct:.1f}% (z={spread.z_score:.1f})",
                ))

        # Sort by expected daily P&L
        signals.sort(key=lambda s: s.expected_daily_pnl_bps, reverse=True)

        return signals

    def open_position(
        self,
        signal: SyntheticSignal,
        spot_price: float
    ) -> SyntheticPosition:
        """
        Open new synthetic position from signal.

        Args:
            signal: Entry signal
            spot_price: Current spot price

        Returns:
            SyntheticPosition object
        """
        # Get venue types
        venue_long_type = None
        venue_short_type = None

        if signal.venue_long:
            config = VENUE_FUNDING_CONFIG.get(signal.venue_long.lower())
            if config:
                venue_long_type = config.venue_type

        if signal.venue_short:
            config = VENUE_FUNDING_CONFIG.get(signal.venue_short.lower())
            if config:
                venue_short_type = config.venue_type

        # Calculate costs
        costs_long = DEFAULT_VENUE_COSTS.get(signal.venue_long.lower() if signal.venue_long else '')
        costs_short = DEFAULT_VENUE_COSTS.get(signal.venue_short.lower() if signal.venue_short else '')

        entry_costs = 0.0
        gas_costs = 0.0

        if costs_long:
            entry_costs += signal.recommended_size_usd * costs_long.taker_fee_bps / 10000
            gas_costs += costs_long.gas_cost_usd

        if costs_short:
            entry_costs += signal.recommended_size_usd * costs_short.taker_fee_bps / 10000
            gas_costs += costs_short.gas_cost_usd

        # Create position
        leverage = min(self.config.max_leverage, signal.synthetic_type.max_leverage)
        margin = signal.recommended_size_usd / leverage

        position = SyntheticPosition(
            position_id=str(uuid.uuid4())[:8],
            entry_time=signal.timestamp,
            synthetic_type=signal.synthetic_type,
            venue_long=signal.venue_long,
            venue_short=signal.venue_short,
            venue_long_type=venue_long_type,
            venue_short_type=venue_short_type,
            entry_funding_long=signal.funding_long_annual_pct / 100,
            entry_funding_short=signal.funding_short_annual_pct / 100,
            entry_spread_annual_pct=signal.gross_spread_annual_pct,
            entry_spot_price=spot_price,
            size_usd=signal.recommended_size_usd,
            leverage=leverage,
            margin_used=margin,
            transaction_costs=entry_costs,
            gas_costs=gas_costs,
        )

        self._open_positions[position.position_id] = position
        return position

    def close_position(
        self,
        position_id: str,
        signal: SyntheticSignal,
        spot_price: float
    ) -> Optional[SyntheticPosition]:
        """
        Close existing position.

        Args:
            position_id: Position ID to close
            signal: Exit signal
            spot_price: Current spot price

        Returns:
            Closed position or None if not found
        """
        if position_id not in self._open_positions:
            return None

        position = self._open_positions[position_id]

        # Calculate exit costs
        costs_long = DEFAULT_VENUE_COSTS.get(position.venue_long.lower() if position.venue_long else '')
        costs_short = DEFAULT_VENUE_COSTS.get(position.venue_short.lower() if position.venue_short else '')

        exit_costs = 0.0
        if costs_long:
            exit_costs += position.size_usd * costs_long.taker_fee_bps / 10000
        if costs_short:
            exit_costs += position.size_usd * costs_short.taker_fee_bps / 10000

        position.close(
            exit_time=signal.timestamp,
            exit_reason=signal.signal_type.exit_reason or ExitReason.PROFIT_TARGET,
            funding_long=signal.funding_long_annual_pct / 100,
            funding_short=signal.funding_short_annual_pct / 100,
            spot_price=spot_price,
            exit_costs=exit_costs,
        )

        del self._open_positions[position_id]
        return position

    def run_backtest(
        self,
        funding_data: Dict[str, pd.DataFrame],
        spot_prices: pd.Series,
        initial_capital: float = 1_000_000
    ) -> SyntheticBacktestResult:
        """
        Run backtest on historical data.

        Args:
            funding_data: Dict of venue -> DataFrame with funding rates
            spot_prices: Series of spot prices
            initial_capital: Starting capital

        Returns:
            SyntheticBacktestResult
        """
        # Reset state
        self._open_positions = {}
        self._spread_history = {}

        capital = initial_capital
        positions: List[SyntheticPosition] = []
        signals: List[SyntheticSignal] = []
        equity_records = []

        # Get common timestamps
        all_timestamps = set()
        for df in funding_data.values():
            if 'timestamp' in df.columns:
                all_timestamps.update(df['timestamp'].tolist())

        all_timestamps = sorted(all_timestamps)

        for ts in all_timestamps:
            ts = pd.Timestamp(ts)
            # Ensure timestamp is timezone-aware (UTC)
            if ts.tzinfo is None:
                ts = ts.tz_localize('UTC')

            # Get spot price
            if ts not in spot_prices.index:
                continue
            spot = spot_prices.loc[ts]

            # Build term structures
            term_structures = {}
            for venue, df in funding_data.items():
                venue_df = df[df['timestamp'] <= ts].tail(30 * 24)
                if not venue_df.empty:
                    term_structures[venue] = self.analyzer.build_funding_term_structure(
                        venue_df, venue, spot, ts
                    )

            if not term_structures:
                continue

            # Generate signals
            ts_signals = self.generate_signal(
                ts, term_structures, spot, capital,
                list(self._open_positions.keys())
            )
            signals.extend(ts_signals)

            # Process signals
            for signal in ts_signals:
                if signal.signal_type.is_exit:
                    # Find position to close
                    for pos_id, pos in list(self._open_positions.items()):
                        if pos.venue_long == signal.venue_long and pos.venue_short == signal.venue_short:
                            closed_pos = self.close_position(pos_id, signal, spot)
                            if closed_pos:
                                capital += closed_pos.net_pnl
                                positions.append(closed_pos)
                            break

                elif signal.signal_type.is_entry and signal.is_actionable:
                    if len(self._open_positions) < self.config.max_positions:
                        # Check capital
                        if signal.recommended_size_usd <= capital * self.config.max_position_pct:
                            pos = self.open_position(signal, spot)
                            capital -= pos.transaction_costs + pos.gas_costs

            # Calculate unrealized P&L
            unrealized = sum(p.net_pnl for p in self._open_positions.values())

            equity_records.append({
                'timestamp': ts,
                'capital': capital,
                'unrealized': unrealized,
                'equity': capital + unrealized,
                'open_positions': len(self._open_positions),
            })

        # Close remaining positions
        for pos_id, pos in list(self._open_positions.items()):
            # Create synthetic exit signal
            exit_signal = SyntheticSignal(
                timestamp=all_timestamps[-1] if all_timestamps else pd.Timestamp.now(),
                signal_type=SyntheticSignalType.EXIT_MAX_HOLD,
                synthetic_type=pos.synthetic_type,
                venue_long=pos.venue_long,
                venue_short=pos.venue_short,
                funding_long_annual_pct=pos.entry_funding_long * 100,
                funding_short_annual_pct=pos.entry_funding_short * 100,
                gross_spread_annual_pct=pos.entry_spread_annual_pct,
                net_spread_annual_pct=pos.entry_spread_annual_pct,
                z_score=0.0,
                signal_strength=1.0,
                confidence=1.0,
                recommended_size_usd=pos.size_usd,
                expected_daily_pnl_bps=0.0,
                reason="End of backtest",
            )

            closed_pos = self.close_position(pos_id, exit_signal, spot_prices.iloc[-1])
            if closed_pos:
                capital += closed_pos.net_pnl
                positions.append(closed_pos)

        # Build equity curve
        equity_df = pd.DataFrame(equity_records)
        if not equity_df.empty:
            equity_df.set_index('timestamp', inplace=True)

        # Create result
        total_pnl = capital - initial_capital
        total_return_pct = (total_pnl / initial_capital) * 100

        result = SyntheticBacktestResult(
            positions=positions,
            signals=signals,
            equity_curve=equity_df,
            initial_capital=initial_capital,
            final_capital=capital,
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
        )

        return result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'SyntheticType',
    'SyntheticSignalType',
    # Configuration
    'SyntheticFuturesConfig',
    # Crisis and regime parameters
    'STRATEGY_C_CRISIS_PARAMS',
    'FUNDING_REGIME_PARAMS',
    # Data structures
    'SyntheticSignal',
    'SyntheticPosition',
    'SyntheticBacktestResult',
    # Strategy C implementation
    'SyntheticFuturesStrategy',
]
