"""
BTC Calendar Spread Strategies (Strategy A)
==========================================

Part 2 Section 3.2.1 - Calendar Spread Implementation

Calendar spread and cross-venue basis arbitrage strategies
for BTC futures markets.

Part 2 Requirements Addressed:
- 3.2.1 Strategy A: Traditional Calendar Spreads
- 3.1.4 Regime classification integration
- 3.3.3 Crisis event analysis (COVID, Luna, FTX, May 2021)
- Multi-venue support (Binance, Deribit, CME, Hyperliquid, dYdX, GMX)
- Kelly criterion position sizing
- Regime-adaptive parameter adjustment

Mathematical Framework
----------------------
Calendar Spread P&L:

    For LONG calendar (long near, short far):
        P&L = -ΔSpread × Notional
        
    Where:
        Spread = Far_Basis - Near_Basis
        ΔSpread = Exit_Spread - Entry_Spread
    
    Profit when spread NARROWS (contango decreases)

    For SHORT calendar (short near, long far):
        P&L = +ΔSpread × Notional
        
    Profit when spread WIDENS (contango increases)

Signal Generation:

    Entry Signal Strength:
        s = min(1.0, |current_basis - threshold| / volatility)
    
    Position Size:
        size = base_size × signal_strength × regime_multiplier × liquidity_factor

Cross-Venue Arbitrage:

    Differential = Basis_V1 - Basis_V2
    Z-Score = (Differential - μ) / σ
    
    Entry: |Z-Score| > 2.0 and Net_Differential > Costs
    Exit: |Z-Score| < 0.5 or Max_Hold exceeded

Risk Management:

    Stop Loss: |Basis_Change| > threshold (typically 5%)
    Expiry Exit: Near_DTE < min_dte (typically 7 days)
    Max Hold: holding_days > max_days (typically 90 days)

Performance Metrics:

    Sharpe = (Avg_Return - Rf) / Std_Return × √(252/avg_holding)
    Profit_Factor = Gross_Wins / Gross_Losses
    Efficiency = Net_P&L / Max_Favorable_Excursion

Version: 2.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import uuid
import logging

from . import (
    TermStructureRegime, VenueType, SpreadDirection, ExitReason,
    CurveShape, VenueCosts, TermStructurePoint, CalendarSpreadSignal,
    CalendarSpreadTrade, CrossVenueOpportunity,
    DEFAULT_VENUE_COSTS, DEFAULT_VENUE_CAPACITY,
    DEFAULT_CALENDAR_PARAMS, DEFAULT_CROSS_VENUE_PARAMS
)
from .term_structure import (
    TermStructureCurve, TermStructureAnalyzer, RegimeTracker
)
from .funding_rate_analysis import (
    CRISIS_EVENTS, is_crisis_period, FundingRateAnalyzer,
    FundingTermStructure, FundingRegime
)

logger = logging.getLogger(__name__)


# =============================================================================
# REGIME-ADAPTIVE PARAMETERS
# =============================================================================

REGIME_PARAMS = {
    TermStructureRegime.STEEP_CONTANGO: {
        'long_entry_multiplier': 1.2,  # More aggressive entries
        'short_entry_multiplier': 0.5,  # Conservative shorts
        'position_size_mult': 1.0,
        'expected_reversion': True,
        'max_holding_days': 45,
    },
    TermStructureRegime.MILD_CONTANGO: {
        'long_entry_multiplier': 1.0,
        'short_entry_multiplier': 0.8,
        'position_size_mult': 0.8,
        'expected_reversion': True,
        'max_holding_days': 60,
    },
    TermStructureRegime.FLAT: {
        'long_entry_multiplier': 0.7,
        'short_entry_multiplier': 0.7,
        'position_size_mult': 0.5,
        'expected_reversion': False,
        'max_holding_days': 30,
    },
    TermStructureRegime.MILD_BACKWARDATION: {
        'long_entry_multiplier': 0.8,
        'short_entry_multiplier': 1.0,
        'position_size_mult': 0.8,
        'expected_reversion': True,
        'max_holding_days': 60,
    },
    TermStructureRegime.STEEP_BACKWARDATION: {
        'long_entry_multiplier': 0.5,
        'short_entry_multiplier': 1.2,
        'position_size_mult': 1.0,
        'expected_reversion': True,
        'max_holding_days': 45,
    },
}

CRISIS_PARAMS = {
    'position_size_mult': 0.3,  # Reduce size in crises
    'entry_threshold_mult': 1.5,  # Wider thresholds
    'stop_loss_mult': 2.0,  # Wider stops
    'max_holding_days': 14,  # Shorter holds
}


# =============================================================================
# KELLY CRITERION POSITION SIZING
# =============================================================================

def calculate_kelly_fraction(
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    kelly_fraction: float = 0.5  # Half-Kelly for safety
) -> float:
    """
    Calculate optimal Kelly fraction for position sizing.

    The Kelly Criterion determines the optimal fraction of capital to risk
    on each trade to maximize long-term growth.

    Formula:
        f* = (p * b - q) / b
        where:
            f* = Kelly fraction
            p = probability of winning
            q = probability of losing (1 - p)
            b = ratio of avg win to avg loss

    Args:
        win_rate: Probability of winning trade (0-1)
        avg_win_pct: Average win as percentage
        avg_loss_pct: Average loss as percentage (absolute value)
        kelly_fraction: Fraction of full Kelly to use (0.5 = half-Kelly)

    Returns:
        Optimal position size fraction (0-1)
    """
    if win_rate <= 0 or win_rate >= 1:
        return 0.0

    if avg_loss_pct <= 0:
        return 0.0

    # Calculate odds ratio (b)
    b = avg_win_pct / avg_loss_pct

    # Probability of loss
    q = 1 - win_rate

    # Kelly formula
    kelly = (win_rate * b - q) / b

    # Apply fractional Kelly and clamp to valid range
    return max(0.0, min(kelly * kelly_fraction, 1.0))


def get_regime_adjusted_params(
    base_params: Dict[str, Any],
    regime: TermStructureRegime
) -> Dict[str, Any]:
    """
    Adjust parameters based on term structure regime.

    Args:
        base_params: Base parameter dictionary
        regime: Current term structure regime

    Returns:
        Adjusted parameters dictionary
    """
    adjusted = base_params.copy()
    regime_config = REGIME_PARAMS.get(regime, {})

    for key, mult in regime_config.items():
        if key in adjusted and isinstance(mult, (int, float)):
            adjusted[key] = adjusted[key] * mult

    return adjusted


def get_crisis_adjusted_params(
    base_params: Dict[str, Any],
    crisis_severity: float
) -> Dict[str, Any]:
    """
    Adjust parameters for crisis periods.

    Args:
        base_params: Base parameter dictionary
        crisis_severity: Severity of crisis (0-1)

    Returns:
        Adjusted parameters dictionary
    """
    if crisis_severity <= 0:
        return base_params

    adjusted = base_params.copy()

    # Apply crisis multipliers interpolated by severity
    for key, mult in CRISIS_PARAMS.items():
        if key in adjusted and isinstance(mult, (int, float)):
            # Interpolate between base and crisis value
            adjustment = 1.0 + (mult - 1.0) * crisis_severity
            adjusted[key] = adjusted[key] * adjustment

    return adjusted


# =============================================================================
# ADDITIONAL ENUMERATIONS
# =============================================================================

class SignalType(Enum):
    """Signal type for strategy actions."""
    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT_PROFIT = "exit_profit"
    EXIT_STOP = "exit_stop"
    EXIT_TIME = "exit_time"
    EXIT_EXPIRY = "exit_expiry"
    HOLD = "hold"
    NO_ACTION = "no_action"
    
    @property
    def is_entry(self) -> bool:
        """True if signal is an entry."""
        return self in [self.ENTRY_LONG, self.ENTRY_SHORT]
    
    @property
    def is_exit(self) -> bool:
        """True if signal is an exit."""
        return self in [
            self.EXIT_PROFIT, self.EXIT_STOP, 
            self.EXIT_TIME, self.EXIT_EXPIRY
        ]
    
    @property
    def direction(self) -> SpreadDirection:
        """Get spread direction for entry signals."""
        if self == self.ENTRY_LONG:
            return SpreadDirection.LONG
        elif self == self.ENTRY_SHORT:
            return SpreadDirection.SHORT
        return SpreadDirection.FLAT


class PositionStatus(Enum):
    """Current position status."""
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"
    PENDING_EXIT = "pending_exit"
    
    @property
    def is_active(self) -> bool:
        """True if position is active."""
        return self in [self.LONG, self.SHORT, self.PENDING_EXIT]


# =============================================================================
# BACKTEST RESULT DATACLASS
# =============================================================================

@dataclass
class BacktestResult:
    """
    Backtest results with performance analytics.

    Includes risk-adjusted returns, drawdown analysis, and
    trade-level statistics.
    """
    trades: List[CalendarSpreadTrade]
    equity_curve: pd.DataFrame
    signals: List[CalendarSpreadSignal]
    initial_capital: float
    final_capital: float
    
    # Strategy metadata
    strategy_name: str = "CalendarSpread"
    venue: str = "binance"
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Basic metrics
    @property
    def total_trades(self) -> int:
        """Total number of completed trades."""
        return len([t for t in self.trades if t.is_closed])
    
    @property
    def open_trades(self) -> int:
        """Number of open trades."""
        return len([t for t in self.trades if t.is_open])
    
    @property
    def winning_trades(self) -> int:
        """Number of winning trades."""
        return len([t for t in self.trades if t.is_closed and t.is_winner])
    
    @property
    def losing_trades(self) -> int:
        """Number of losing trades."""
        return len([t for t in self.trades if t.is_closed and t.is_loser])
    
    # P&L metrics
    @property
    def total_pnl(self) -> float:
        """Total net P&L."""
        return sum(t.net_pnl for t in self.trades if t.is_closed)
    
    @property
    def gross_pnl(self) -> float:
        """Total gross P&L (before costs)."""
        return sum(t.gross_pnl for t in self.trades if t.is_closed)
    
    @property
    def total_costs(self) -> float:
        """Total transaction costs."""
        return sum(t.total_costs for t in self.trades if t.is_closed)
    
    @property
    def total_return_pct(self) -> float:
        """Total return as percentage of initial capital."""
        if self.initial_capital <= 0:
            return 0.0
        return (self.total_pnl / self.initial_capital) * 100
    
    @property
    def annualized_return_pct(self) -> float:
        """Annualized return percentage."""
        if self.start_date is None or self.end_date is None:
            return 0.0
        
        days = (self.end_date - self.start_date).days
        if days <= 0:
            return 0.0
        
        total_return = self.total_pnl / self.initial_capital
        return ((1 + total_return) ** (365 / days) - 1) * 100
    
    # Win/Loss metrics
    @property
    def win_rate(self) -> float:
        """Percentage of winning trades."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    @property
    def avg_win(self) -> float:
        """Average winning trade P&L."""
        winners = [t.net_pnl for t in self.trades if t.is_closed and t.is_winner]
        return np.mean(winners) if winners else 0.0
    
    @property
    def avg_loss(self) -> float:
        """Average losing trade P&L (negative)."""
        losers = [t.net_pnl for t in self.trades if t.is_closed and t.is_loser]
        return np.mean(losers) if losers else 0.0
    
    @property
    def avg_pnl(self) -> float:
        """Average P&L per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades
    
    @property
    def profit_factor(self) -> float:
        """Gross wins / Gross losses."""
        gross_wins = sum(t.gross_pnl for t in self.trades if t.is_closed and t.gross_pnl > 0)
        gross_losses = abs(sum(t.gross_pnl for t in self.trades if t.is_closed and t.gross_pnl < 0))
        
        if gross_losses <= 0:
            return float('inf') if gross_wins > 0 else 0.0
        return gross_wins / gross_losses
    
    @property
    def payoff_ratio(self) -> float:
        """Average win / Average loss (absolute)."""
        if self.avg_loss == 0:
            return float('inf') if self.avg_win > 0 else 0.0
        return abs(self.avg_win / self.avg_loss)
    
    @property
    def expectancy(self) -> float:
        """Expected value per trade."""
        return (self.win_rate / 100) * self.avg_win + (1 - self.win_rate / 100) * self.avg_loss
    
    # Risk metrics
    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio (assuming 0 risk-free rate)."""
        if self.equity_curve.empty:
            return 0.0
        
        returns = self.equity_curve['equity'].pct_change().dropna()
        if returns.empty or returns.std() == 0:
            return 0.0
        
        # Annualize based on average holding period
        avg_holding = self.avg_holding_days
        periods_per_year = 365 / max(avg_holding, 1)
        
        return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
    
    @property
    def sortino_ratio(self) -> float:
        """Annualized Sortino ratio (downside deviation)."""
        if self.equity_curve.empty:
            return 0.0
        
        returns = self.equity_curve['equity'].pct_change().dropna()
        if returns.empty:
            return 0.0
        
        downside = returns[returns < 0]
        if downside.empty or downside.std() == 0:
            return float('inf') if returns.mean() > 0 else 0.0
        
        avg_holding = self.avg_holding_days
        periods_per_year = 365 / max(avg_holding, 1)
        
        return (returns.mean() / downside.std()) * np.sqrt(periods_per_year)
    
    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as percentage."""
        if self.equity_curve.empty:
            return 0.0
        
        equity = self.equity_curve['equity']
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak * 100
        
        return abs(drawdown.min())
    
    @property
    def max_drawdown_duration(self) -> int:
        """Maximum drawdown duration in days."""
        if self.equity_curve.empty:
            return 0
        
        equity = self.equity_curve['equity']
        peak = equity.expanding().max()
        
        in_drawdown = equity < peak
        
        if not in_drawdown.any():
            return 0
        
        # Find longest consecutive drawdown
        max_duration = 0
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    @property
    def calmar_ratio(self) -> float:
        """Annualized return / Max drawdown."""
        if self.max_drawdown == 0:
            return float('inf') if self.annualized_return_pct > 0 else 0.0
        return self.annualized_return_pct / self.max_drawdown
    
    # Holding period metrics
    @property
    def avg_holding_days(self) -> float:
        """Average holding period in days."""
        closed = [t for t in self.trades if t.is_closed]
        if not closed:
            return 0.0
        return np.mean([t.holding_period_days for t in closed])
    
    @property
    def max_holding_days(self) -> float:
        """Maximum holding period in days."""
        closed = [t for t in self.trades if t.is_closed]
        if not closed:
            return 0.0
        return max(t.holding_period_days for t in closed)
    
    @property
    def min_holding_days(self) -> float:
        """Minimum holding period in days."""
        closed = [t for t in self.trades if t.is_closed]
        if not closed:
            return 0.0
        return min(t.holding_period_days for t in closed)
    
    # Exit analysis
    @property
    def exit_reason_distribution(self) -> Dict[str, int]:
        """Distribution of exit reasons."""
        dist: Dict[str, int] = {}
        for t in self.trades:
            if t.is_closed and t.exit_reason:
                reason = t.exit_reason.value
                dist[reason] = dist.get(reason, 0) + 1
        return dist
    
    @property
    def profitable_exit_rate(self) -> float:
        """Percentage of exits that were profitable."""
        closed = [t for t in self.trades if t.is_closed and t.exit_reason]
        if not closed:
            return 0.0
        
        profitable = len([t for t in closed if t.exit_reason.is_profitable_exit])
        return (profitable / len(closed)) * 100
    
    # Efficiency metrics
    @property
    def avg_efficiency(self) -> float:
        """Average trade efficiency (P&L / MFE)."""
        closed = [t for t in self.trades if t.is_closed and t.max_favorable_excursion > 0]
        if not closed:
            return 0.0
        return np.mean([t.efficiency for t in closed])
    
    @property
    def avg_mae_pct(self) -> float:
        """Average maximum adverse excursion."""
        closed = [t for t in self.trades if t.is_closed]
        if not closed:
            return 0.0
        return np.mean([t.mae_pct for t in closed])
    
    @property
    def avg_mfe_pct(self) -> float:
        """Average maximum favorable excursion."""
        closed = [t for t in self.trades if t.is_closed]
        if not closed:
            return 0.0
        return np.mean([t.mfe_pct for t in closed])
    
    # Cost analysis
    @property
    def cost_per_trade(self) -> float:
        """Average cost per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.total_costs / self.total_trades
    
    @property
    def cost_drag_pct(self) -> float:
        """Costs as percentage of gross P&L."""
        if self.gross_pnl == 0:
            return 0.0
        return (self.total_costs / abs(self.gross_pnl)) * 100
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'strategy': self.strategy_name,
            'venue': self.venue,
            'period': f"{self.start_date} to {self.end_date}",
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_pnl': round(self.total_pnl, 2),
            'total_return_pct': round(self.total_return_pct, 2),
            'annualized_return_pct': round(self.annualized_return_pct, 2),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate, 1),
            'avg_win': round(self.avg_win, 2),
            'avg_loss': round(self.avg_loss, 2),
            'profit_factor': round(self.profit_factor, 2),
            'payoff_ratio': round(self.payoff_ratio, 2),
            'expectancy': round(self.expectancy, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'sortino_ratio': round(self.sortino_ratio, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'calmar_ratio': round(self.calmar_ratio, 2),
            'avg_holding_days': round(self.avg_holding_days, 1),
            'avg_efficiency': round(self.avg_efficiency, 2),
            'total_costs': round(self.total_costs, 2),
            'cost_drag_pct': round(self.cost_drag_pct, 1),
        }
    
    def trades_to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        return pd.DataFrame([t.to_dict() for t in self.trades])
    
    def signals_to_dataframe(self) -> pd.DataFrame:
        """Convert signals to DataFrame."""
        return pd.DataFrame([s.to_dict() for s in self.signals])


# =============================================================================
# CALENDAR SPREAD STRATEGY
# =============================================================================

class CalendarSpreadStrategy:
    """
    BTC Calendar Spread Trading Strategy.
    
    Exploits mean reversion in the term structure by taking
    positions in calendar spreads when basis is extended.
    
    Key Features:
    - Regime-aware signal generation
    - Dynamic position sizing
    - Multi-venue support
    - Cost modeling
    - Risk management
    
    Usage:
        strategy = CalendarSpreadStrategy(
            params={'long_entry_basis_pct': 15.0},
            venues=['binance', 'deribit']
        )
        results = strategy.run_backtest(curves, capital=1_000_000)
        print(results.summary())
    """
    
    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        venues: Optional[List[str]] = None,
        venue_costs: Optional[Dict[str, VenueCosts]] = None
    ):
        """
        Initialize strategy.
        
        Args:
            params: Strategy parameters (see DEFAULT_CALENDAR_PARAMS)
            venues: List of venues to trade
            venue_costs: Custom venue cost configurations
        """
        self.params = {**DEFAULT_CALENDAR_PARAMS}
        if params:
            self.params.update(params)
        
        self.venues = venues or ['binance']
        self.venue_costs = venue_costs or DEFAULT_VENUE_COSTS
        
        # State tracking
        self._open_positions: Dict[str, CalendarSpreadTrade] = {}
        self._regime_tracker = RegimeTracker()
        
        # Funding analyzer integration for Strategy C synergy
        self._funding_analyzer: Optional[FundingRateAnalyzer] = None

        # Historical performance tracking for Kelly
        self._trade_results: List[float] = []  # Win/loss returns

        logger.info(f"CalendarSpreadStrategy initialized: venues={self.venues}")

    def integrate_funding_analyzer(self, analyzer: FundingRateAnalyzer) -> None:
        """
        Integrate with funding rate analyzer for Strategy C synergy.

        Args:
            analyzer: FundingRateAnalyzer instance
        """
        self._funding_analyzer = analyzer
        logger.info("Integrated with FundingRateAnalyzer")

    def get_regime_adjusted_params(
        self,
        regime: TermStructureRegime,
        timestamp: pd.Timestamp
    ) -> Dict[str, Any]:
        """
        Get parameters adjusted for current regime and crisis periods.

        Args:
            regime: Current term structure regime
            timestamp: Current timestamp

        Returns:
            Adjusted parameters dictionary
        """
        # Start with base params
        adjusted = dict(self.params)

        # Apply regime adjustments
        regime_adj = REGIME_PARAMS.get(regime, REGIME_PARAMS[TermStructureRegime.FLAT])

        adjusted['long_entry_basis_pct'] *= regime_adj['long_entry_multiplier']
        adjusted['short_entry_basis_pct'] *= regime_adj['short_entry_multiplier']
        adjusted['max_holding_days'] = min(
            adjusted.get('max_holding_days', 90),
            regime_adj['max_holding_days']
        )

        # Check for crisis period
        is_crisis, crisis_name, severity = is_crisis_period(timestamp)
        if is_crisis:
            logger.warning(f"Crisis period detected: {crisis_name} (severity={severity})")
            adjusted['max_position_pct'] *= CRISIS_PARAMS['position_size_mult']
            adjusted['long_entry_basis_pct'] *= CRISIS_PARAMS['entry_threshold_mult']
            adjusted['short_entry_basis_pct'] *= CRISIS_PARAMS['entry_threshold_mult']
            adjusted['stop_loss_basis_change_pct'] *= CRISIS_PARAMS['stop_loss_mult']
            adjusted['max_holding_days'] = min(
                adjusted['max_holding_days'],
                CRISIS_PARAMS['max_holding_days']
            )
            adjusted['_is_crisis'] = True
            adjusted['_crisis_name'] = crisis_name
            adjusted['_crisis_severity'] = severity
        else:
            adjusted['_is_crisis'] = False

        return adjusted

    def calculate_kelly_fraction(
        self,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> float:
        """
        Calculate Kelly criterion fraction for position sizing.

        Kelly Formula: f* = (p * b - q) / b
        Where:
            p = probability of win
            q = probability of loss (1 - p)
            b = win/loss ratio

        Returns fractional Kelly (typically 1/4 to 1/2 full Kelly)
        for more conservative sizing.

        Returns:
            Kelly fraction (0 to 0.25 typically)
        """
        # Use historical data if available
        if self._trade_results and len(self._trade_results) >= 10:
            wins = [r for r in self._trade_results if r > 0]
            losses = [r for r in self._trade_results if r < 0]

            if wins and losses:
                win_rate = len(wins) / len(self._trade_results)
                avg_win = np.mean(wins)
                avg_loss = abs(np.mean(losses))

        # Use provided or default values
        if win_rate is None:
            win_rate = 0.55  # Conservative default
        if avg_win is None:
            avg_win = 0.05  # 5% average win
        if avg_loss is None:
            avg_loss = 0.03  # 3% average loss

        # Calculate Kelly
        if avg_loss <= 0:
            return 0.0

        b = avg_win / avg_loss  # Win/loss ratio
        q = 1 - win_rate

        kelly_full = (win_rate * b - q) / b

        # Use fractional Kelly (1/4)
        kelly_fraction = kelly_full * 0.25

        # Bound to reasonable range
        return max(0.0, min(kelly_fraction, 0.25))

    def classify_regime(self, annualized_basis: float) -> TermStructureRegime:
        """Classify current regime."""
        return TermStructureRegime.from_basis(annualized_basis)
    
    def calculate_signal_strength(
        self,
        current_basis: float,
        threshold: float,
        volatility: float,
        regime: TermStructureRegime
    ) -> float:
        """
        Calculate signal strength (0-1).
        
        Based on:
        - Distance from threshold
        - Historical volatility
        - Regime confidence
        """
        if volatility <= 0:
            volatility = 1.0
        
        distance = abs(current_basis - threshold)
        raw_strength = min(distance / volatility, 1.0)
        
        # Regime adjustment
        regime_mult = regime.signal_strength_multiplier
        
        return raw_strength * regime_mult
    
    def generate_signal(
        self,
        curve: TermStructureCurve,
        current_position: Optional[CalendarSpreadTrade] = None,
        basis_history: Optional[List[float]] = None
    ) -> CalendarSpreadSignal:
        """
        Generate trading signal from current curve.
        
        Args:
            curve: Current term structure curve
            current_position: Existing position (if any)
            basis_history: Historical basis values for volatility
            
        Returns:
            CalendarSpreadSignal object
        """
        # Check curve quality
        if not curve.quality.is_tradeable:
            return CalendarSpreadSignal(
                timestamp=curve.as_of,
                signal_type='hold',
                direction=SpreadDirection.FLAT,
                near_contract='',
                far_contract='',
                near_price=0.0,
                far_price=0.0,
                spot_price=curve.spot_price,
                near_basis_pct=0.0,
                far_basis_pct=0.0,
                spread_basis_pct=0.0,
                annualized_spread=0.0,
                regime=curve.regime,
                venue=curve.venue,
                signal_strength=0.0,
                reason='Poor curve quality',
            )
        
        # Get contract points
        near_point = curve.front_month_point
        far_point = curve.back_month_point
        
        if not near_point or not far_point:
            return self._create_no_signal(curve, 'Insufficient contract points')
        
        # Check DTE requirements
        min_dte = self.params.get('min_days_to_expiry', 7)
        min_dte_diff = self.params.get('min_dte_difference', 30)
        
        if near_point.days_to_expiry < min_dte:
            return self._create_no_signal(curve, f'Near contract DTE < {min_dte}')
        
        if far_point.days_to_expiry - near_point.days_to_expiry < min_dte_diff:
            return self._create_no_signal(curve, f'DTE difference < {min_dte_diff}')
        
        # Calculate metrics
        near_basis = near_point.annualized_basis_pct
        far_basis = far_point.annualized_basis_pct
        spread = far_basis - near_basis
        avg_basis = (near_basis + far_basis) / 2
        
        # Calculate volatility
        if basis_history and len(basis_history) >= 5:
            basis_vol = np.std(basis_history)
        else:
            basis_vol = 5.0  # Default volatility
        
        # Update regime tracker
        self._regime_tracker.update(curve.as_of, curve.regime)
        
        # Check if we have a position to exit
        if current_position and current_position.is_open:
            return self._check_exit_signal(
                curve, current_position, near_point, far_point,
                spread, avg_basis
            )
        
        # Check entry conditions
        long_threshold = self.params.get('long_entry_basis_pct', 15.0)
        short_threshold = self.params.get('short_entry_basis_pct', -10.0)
        min_strength = self.params.get('min_signal_strength', 0.3)
        
        # Long calendar entry (contango)
        if avg_basis > long_threshold and curve.regime.is_contango:
            strength = self.calculate_signal_strength(
                avg_basis, long_threshold, basis_vol, curve.regime
            )
            
            if strength >= min_strength:
                expected_pnl = self._estimate_expected_pnl(
                    avg_basis, long_threshold, SpreadDirection.LONG, curve.venue
                )
                
                return CalendarSpreadSignal(
                    timestamp=curve.as_of,
                    signal_type='entry',
                    direction=SpreadDirection.LONG,
                    near_contract=near_point.contract,
                    far_contract=far_point.contract,
                    near_price=near_point.futures_price,
                    far_price=far_point.futures_price,
                    spot_price=curve.spot_price,
                    near_basis_pct=near_basis,
                    far_basis_pct=far_basis,
                    spread_basis_pct=spread,
                    annualized_spread=spread * (365 / (far_point.days_to_expiry - near_point.days_to_expiry)),
                    regime=curve.regime,
                    venue=curve.venue,
                    signal_strength=strength,
                    reason=f'Basis {avg_basis:.1f}% > {long_threshold}%',
                    near_dte=near_point.days_to_expiry,
                    far_dte=far_point.days_to_expiry,
                    expected_pnl_bps=expected_pnl,
                    expected_holding_days=curve.regime.expected_half_life_days,
                    confidence=curve.quality.confidence_multiplier,
                    near_liquidity=near_point.liquidity_score,
                    far_liquidity=far_point.liquidity_score,
                )
        
        # Short calendar entry (backwardation)
        if avg_basis < short_threshold and curve.regime.is_backwardation:
            strength = self.calculate_signal_strength(
                avg_basis, short_threshold, basis_vol, curve.regime
            )
            
            if strength >= min_strength:
                expected_pnl = self._estimate_expected_pnl(
                    avg_basis, short_threshold, SpreadDirection.SHORT, curve.venue
                )
                
                return CalendarSpreadSignal(
                    timestamp=curve.as_of,
                    signal_type='entry',
                    direction=SpreadDirection.SHORT,
                    near_contract=near_point.contract,
                    far_contract=far_point.contract,
                    near_price=near_point.futures_price,
                    far_price=far_point.futures_price,
                    spot_price=curve.spot_price,
                    near_basis_pct=near_basis,
                    far_basis_pct=far_basis,
                    spread_basis_pct=spread,
                    annualized_spread=spread * (365 / (far_point.days_to_expiry - near_point.days_to_expiry)),
                    regime=curve.regime,
                    venue=curve.venue,
                    signal_strength=strength,
                    reason=f'Basis {avg_basis:.1f}% < {short_threshold}%',
                    near_dte=near_point.days_to_expiry,
                    far_dte=far_point.days_to_expiry,
                    expected_pnl_bps=expected_pnl,
                    expected_holding_days=curve.regime.expected_half_life_days,
                    confidence=curve.quality.confidence_multiplier,
                    near_liquidity=near_point.liquidity_score,
                    far_liquidity=far_point.liquidity_score,
                )
        
        return self._create_no_signal(curve, 'No entry conditions met')
    
    def _check_exit_signal(
        self,
        curve: TermStructureCurve,
        position: CalendarSpreadTrade,
        near_point: TermStructurePoint,
        far_point: TermStructurePoint,
        current_spread: float,
        current_basis: float
    ) -> CalendarSpreadSignal:
        """Check exit conditions for existing position."""
        
        # Calculate holding period
        holding_days = (curve.as_of - position.entry_time).days
        
        # Check expiry approach
        min_dte = self.params.get('min_days_to_expiry', 7)
        if near_point.days_to_expiry < min_dte:
            return self._create_exit_signal(
                curve, near_point, far_point, position,
                'exit', ExitReason.EXPIRY_APPROACH,
                f'Near DTE {near_point.days_to_expiry} < {min_dte}'
            )
        
        # Check max holding
        max_hold = self.params.get('max_holding_days', 90)
        if holding_days > max_hold:
            return self._create_exit_signal(
                curve, near_point, far_point, position,
                'exit', ExitReason.MAX_HOLD,
                f'Holding {holding_days} days > {max_hold}'
            )
        
        # Check stop loss
        stop_loss = self.params.get('stop_loss_basis_change_pct', 5.0)
        basis_change = current_basis - position.entry_basis_pct
        
        if position.direction == SpreadDirection.LONG:
            # Long loses if basis increases further
            if basis_change > stop_loss:
                return self._create_exit_signal(
                    curve, near_point, far_point, position,
                    'exit', ExitReason.STOP_LOSS,
                    f'Basis change {basis_change:.1f}% > stop {stop_loss}%'
                )
        else:
            # Short loses if basis decreases further
            if basis_change < -stop_loss:
                return self._create_exit_signal(
                    curve, near_point, far_point, position,
                    'exit', ExitReason.STOP_LOSS,
                    f'Basis change {basis_change:.1f}% < stop {-stop_loss}%'
                )
        
        # Check profit target
        long_exit = self.params.get('long_exit_basis_pct', 5.0)
        short_exit = self.params.get('short_exit_basis_pct', -3.0)
        
        if position.direction == SpreadDirection.LONG:
            if current_basis < long_exit:
                return self._create_exit_signal(
                    curve, near_point, far_point, position,
                    'exit', ExitReason.PROFIT_TARGET,
                    f'Basis {current_basis:.1f}% < target {long_exit}%'
                )
        else:
            if current_basis > short_exit:
                return self._create_exit_signal(
                    curve, near_point, far_point, position,
                    'exit', ExitReason.PROFIT_TARGET,
                    f'Basis {current_basis:.1f}% > target {short_exit}%'
                )
        
        # Hold
        return self._create_no_signal(curve, 'Holding position')
    
    def _create_no_signal(
        self,
        curve: TermStructureCurve,
        reason: str
    ) -> CalendarSpreadSignal:
        """Create no-action signal."""
        return CalendarSpreadSignal(
            timestamp=curve.as_of,
            signal_type='hold',
            direction=SpreadDirection.FLAT,
            near_contract='',
            far_contract='',
            near_price=0.0,
            far_price=0.0,
            spot_price=curve.spot_price,
            near_basis_pct=curve.front_month_basis,
            far_basis_pct=curve.back_month_basis,
            spread_basis_pct=curve.calendar_spread,
            annualized_spread=0.0,
            regime=curve.regime,
            venue=curve.venue,
            signal_strength=0.0,
            reason=reason,
        )
    
    def _create_exit_signal(
        self,
        curve: TermStructureCurve,
        near_point: TermStructurePoint,
        far_point: TermStructurePoint,
        position: CalendarSpreadTrade,
        signal_type: str,
        exit_reason: ExitReason,
        reason: str
    ) -> CalendarSpreadSignal:
        """Create exit signal."""
        spread = far_point.annualized_basis_pct - near_point.annualized_basis_pct
        
        return CalendarSpreadSignal(
            timestamp=curve.as_of,
            signal_type=signal_type,
            direction=position.direction,
            near_contract=near_point.contract,
            far_contract=far_point.contract,
            near_price=near_point.futures_price,
            far_price=far_point.futures_price,
            spot_price=curve.spot_price,
            near_basis_pct=near_point.annualized_basis_pct,
            far_basis_pct=far_point.annualized_basis_pct,
            spread_basis_pct=spread,
            annualized_spread=0.0,
            regime=curve.regime,
            venue=curve.venue,
            signal_strength=1.0,
            reason=reason,
            near_dte=near_point.days_to_expiry,
            far_dte=far_point.days_to_expiry,
        )
    
    def _estimate_expected_pnl(
        self,
        current_basis: float,
        target_basis: float,
        direction: SpreadDirection,
        venue: str
    ) -> float:
        """Estimate expected P&L in basis points."""
        basis_change = abs(current_basis - target_basis)
        
        # Get costs
        costs = self.venue_costs.get(venue.lower())
        if costs:
            cost_bps = costs.round_trip_taker_bps
        else:
            cost_bps = 20.0  # Default
        
        expected_move_bps = basis_change * 100
        net_expected = expected_move_bps - cost_bps
        
        return max(net_expected, 0)
    
    def calculate_position_size(
        self,
        signal: CalendarSpreadSignal,
        capital: float,
        current_positions: int = 0,
        use_kelly: bool = True
    ) -> float:
        """
        Calculate position size for signal using Kelly criterion.

        Per PDF Section 3.2, position sizing incorporates:
        - Kelly criterion for optimal bet sizing
        - Regime-adaptive adjustments
        - Crisis period reductions
        - Liquidity constraints

        Args:
            signal: Trading signal
            capital: Available capital
            current_positions: Number of open positions
            use_kelly: Whether to use Kelly criterion

        Returns:
            Position size in USD
        """
        if not signal.is_actionable:
            return 0.0

        # Get regime-adjusted params
        adjusted_params = self.get_regime_adjusted_params(
            signal.regime, signal.timestamp
        )

        max_position_pct = adjusted_params.get('max_position_pct', 0.25)
        max_positions = adjusted_params.get('max_open_positions', 5)

        if current_positions >= max_positions:
            return 0.0

        # Base size from Kelly or fixed percentage
        if use_kelly and self._trade_results:
            kelly_frac = self.calculate_kelly_fraction()
            base_pct = min(kelly_frac, max_position_pct)
        else:
            base_pct = max_position_pct

        base_size = capital * base_pct

        # Adjustments
        strength_adj = signal.signal_strength
        liquidity_adj = signal.position_size_multiplier

        # Regime adjustment
        regime_mult = REGIME_PARAMS.get(
            signal.regime, REGIME_PARAMS[TermStructureRegime.FLAT]
        )['position_size_mult']

        # Crisis adjustment
        if adjusted_params.get('_is_crisis', False):
            crisis_mult = CRISIS_PARAMS['position_size_mult']
            logger.info(
                f"Crisis position reduction: {adjusted_params.get('_crisis_name')}"
            )
        else:
            crisis_mult = 1.0

        # Venue capacity limit (1% of venue capacity)
        venue_capacity = DEFAULT_VENUE_CAPACITY.get(signal.venue.lower(), 100_000_000)
        max_venue_size = venue_capacity * 0.01

        # Calculate final size
        size = base_size * strength_adj * liquidity_adj * regime_mult * crisis_mult

        # Apply limits
        min_size = self.params.get('min_position_usd', 10_000)
        max_size = min(
            adjusted_params.get('max_position_usd', 1_000_000),
            max_venue_size
        )

        return max(min_size, min(size, max_size))
    
    def open_trade(
        self,
        signal: CalendarSpreadSignal,
        position_size: float,
        leverage: float = 2.0
    ) -> CalendarSpreadTrade:
        """
        Create new trade from signal.
        
        Args:
            signal: Entry signal
            position_size: Position size in USD
            leverage: Leverage to use
            
        Returns:
            CalendarSpreadTrade object
        """
        max_leverage = self.params.get('max_leverage', 2.0)  # PDF: 2.0x max per PDF Section 3.2 (Hyperliquid: 1.5x max)
        leverage = min(leverage, max_leverage)
        
        margin_used = position_size / leverage
        
        # Get venue type
        venue_lower = signal.venue.lower()
        venue_config = self.venue_costs.get(venue_lower)
        venue_type = venue_config.venue_type if venue_config else VenueType.CEX_FUTURES
        
        trade = CalendarSpreadTrade(
            trade_id=str(uuid.uuid4())[:8],
            entry_time=signal.timestamp,
            direction=signal.direction,
            near_contract=signal.near_contract,
            far_contract=signal.far_contract,
            venue=signal.venue,
            venue_type=venue_type,
            entry_near_price=signal.near_price,
            entry_far_price=signal.far_price,
            entry_spot_price=signal.spot_price,
            entry_basis_pct=(signal.near_basis_pct + signal.far_basis_pct) / 2,
            entry_spread_pct=signal.spread_basis_pct,
            entry_near_dte=signal.near_dte,
            entry_far_dte=signal.far_dte,
            notional_size=position_size,
            leverage=leverage,
            margin_used=margin_used,
            entry_signal=signal,
        )
        
        return trade
    
    def close_trade(
        self,
        trade: CalendarSpreadTrade,
        signal: CalendarSpreadSignal,
        exit_reason: ExitReason
    ) -> CalendarSpreadTrade:
        """
        Close trade with exit signal.
        
        Args:
            trade: Trade to close
            signal: Exit signal
            exit_reason: Reason for exit
            
        Returns:
            Updated trade object
        """
        trade.exit_time = signal.timestamp
        trade.exit_near_price = signal.near_price
        trade.exit_far_price = signal.far_price
        trade.exit_spot_price = signal.spot_price
        trade.exit_basis_pct = (signal.near_basis_pct + signal.far_basis_pct) / 2
        trade.exit_spread_pct = signal.spread_basis_pct
        trade.exit_near_dte = signal.near_dte
        trade.exit_far_dte = signal.far_dte
        trade.exit_reason = exit_reason
        
        # Calculate costs
        costs = self.venue_costs.get(trade.venue.lower())
        if costs:
            cost_info = costs.calculate_cost(
                trade.notional_size,
                is_maker=False,
                n_legs=4  # 2 legs in, 2 legs out
            )
            trade.transaction_costs = cost_info['pct_cost_usd']
            trade.gas_costs = cost_info['gas_cost_usd']
        else:
            trade.transaction_costs = trade.notional_size * 0.002  # Default 0.2%
        
        # Calculate P&L
        trade.calculate_pnl()

        # Track result for Kelly criterion
        if trade.notional_size > 0:
            return_pct = trade.net_pnl / trade.notional_size
            self._trade_results.append(return_pct)
            # Keep last 100 trades
            if len(self._trade_results) > 100:
                self._trade_results = self._trade_results[-100:]

        return trade

    def get_funding_adjusted_signal(
        self,
        curve: TermStructureCurve,
        funding_ts: Optional[FundingTermStructure] = None,
        current_position: Optional[CalendarSpreadTrade] = None,
        basis_history: Optional[List[float]] = None
    ) -> CalendarSpreadSignal:
        """
        Generate signal incorporating funding rate analysis.

        This provides Strategy A/C integration per PDF requirements.

        Args:
            curve: Current term structure curve
            funding_ts: Funding-implied term structure (if available)
            current_position: Existing position
            basis_history: Historical basis values

        Returns:
            CalendarSpreadSignal with funding-adjusted confidence
        """
        # Get base signal
        signal = self.generate_signal(curve, current_position, basis_history)

        if not funding_ts or not signal.is_entry:
            return signal

        # Compare actual vs funding-implied basis
        implied_basis = funding_ts.get_implied_basis(signal.near_dte or 30)
        actual_basis = signal.near_basis_pct

        differential = actual_basis - implied_basis

        # Adjust confidence based on funding alignment
        # If actual > implied, futures rich vs funding (support short signal)
        # If actual < implied, futures cheap vs funding (support long signal)
        if signal.direction == SpreadDirection.LONG:
            # Long benefits from actual < implied (cheap futures)
            if differential < -2:
                signal.confidence = min(signal.confidence * 1.2, 1.0)
                signal.reason += f' (Funding supports: diff={differential:.1f}%)'
            elif differential > 2:
                signal.confidence *= 0.8
                signal.reason += f' (Funding contra: diff={differential:.1f}%)'
        elif signal.direction == SpreadDirection.SHORT:
            # Short benefits from actual > implied (rich futures)
            if differential > 2:
                signal.confidence = min(signal.confidence * 1.2, 1.0)
                signal.reason += f' (Funding supports: diff={differential:.1f}%)'
            elif differential < -2:
                signal.confidence *= 0.8
                signal.reason += f' (Funding contra: diff={differential:.1f}%)'

        return signal
    
    def run_backtest(
        self,
        curves: List[TermStructureCurve],
        initial_capital: float = 1_000_000
    ) -> BacktestResult:
        """
        Run backtest on historical curves.
        
        Args:
            curves: List of TermStructureCurve objects
            initial_capital: Starting capital
            
        Returns:
            BacktestResult object
        """
        if not curves:
            return BacktestResult(
                trades=[],
                equity_curve=pd.DataFrame(),
                signals=[],
                initial_capital=initial_capital,
                final_capital=initial_capital,
            )
        
        # Sort by timestamp
        curves = sorted(curves, key=lambda c: c.as_of)
        
        # Initialize state
        capital = initial_capital
        trades: List[CalendarSpreadTrade] = []
        signals: List[CalendarSpreadSignal] = []
        equity_records = []
        open_trade: Optional[CalendarSpreadTrade] = None
        basis_history: List[float] = []
        
        for curve in curves:
            # Track basis history
            basis_history.append(curve.average_basis_pct)
            if len(basis_history) > 30:
                basis_history = basis_history[-30:]
            
            # Generate signal
            signal = self.generate_signal(curve, open_trade, basis_history)
            signals.append(signal)
            
            # Process signal
            if signal.is_exit and open_trade:
                # Determine exit reason
                if 'profit' in signal.reason.lower() or 'target' in signal.reason.lower():
                    exit_reason = ExitReason.PROFIT_TARGET
                elif 'stop' in signal.reason.lower():
                    exit_reason = ExitReason.STOP_LOSS
                elif 'expir' in signal.reason.lower() or 'dte' in signal.reason.lower():
                    exit_reason = ExitReason.EXPIRY_APPROACH
                elif 'hold' in signal.reason.lower() or 'max' in signal.reason.lower():
                    exit_reason = ExitReason.MAX_HOLD
                else:
                    exit_reason = ExitReason.MANUAL
                
                open_trade = self.close_trade(open_trade, signal, exit_reason)
                trades.append(open_trade)
                capital += open_trade.net_pnl
                open_trade = None
            
            elif signal.is_entry and not open_trade:
                # Calculate position size
                n_positions = len([t for t in trades if t.is_open])
                size = self.calculate_position_size(signal, capital, n_positions)
                
                if size > 0:
                    open_trade = self.open_trade(signal, size)
            
            elif open_trade:
                # Update MTM
                near_point = curve.get_point_by_contract(open_trade.near_contract)
                far_point = curve.get_point_by_contract(open_trade.far_contract)
                
                if near_point and far_point:
                    open_trade.update_mtm(
                        curve.as_of,
                        near_point.futures_price,
                        far_point.futures_price,
                        curve.spot_price,
                        near_point.days_to_expiry,
                        far_point.days_to_expiry
                    )
            
            # Track equity
            unrealized = 0.0
            if open_trade and open_trade.mtm_history:
                unrealized = open_trade.mtm_history[-1]['unrealized_pnl']
            
            equity_records.append({
                'timestamp': curve.as_of,
                'capital': capital,
                'unrealized': unrealized,
                'equity': capital + unrealized,
                'regime': curve.regime.value,
            })
        
        # Close any remaining open trade at end
        if open_trade and curves:
            last_curve = curves[-1]
            final_signal = self.generate_signal(last_curve, open_trade, basis_history)
            open_trade = self.close_trade(
                open_trade, final_signal, ExitReason.END_OF_DATA
            )
            trades.append(open_trade)
            capital += open_trade.net_pnl
        
        equity_df = pd.DataFrame(equity_records)
        if not equity_df.empty:
            equity_df.set_index('timestamp', inplace=True)
        
        return BacktestResult(
            trades=trades,
            equity_curve=equity_df,
            signals=signals,
            initial_capital=initial_capital,
            final_capital=capital,
            strategy_name='CalendarSpread',
            venue=self.venues[0] if self.venues else 'unknown',
            start_date=curves[0].as_of if curves else None,
            end_date=curves[-1].as_of if curves else None,
            params=self.params,
        )


# =============================================================================
# CROSS-VENUE BASIS STRATEGY
# =============================================================================

class CrossVenueBasisStrategy:
    """
    Cross-Venue Basis Arbitrage Strategy.
    
    Exploits pricing discrepancies between BTC futures across
    different trading venues through simultaneous long/short positions.
    
    Key Features:
    - Z-score based signal generation
    - Multi-venue comparison
    - Cost-aware opportunity filtering
    - Convergence time estimation
    
    Usage:
        strategy = CrossVenueBasisStrategy(
            min_differential_bps=20,
            venue_pairs=[('cme', 'binance'), ('binance', 'hyperliquid')]
        )
        opportunities = strategy.find_opportunities(curves_dict)
    """
    
    def __init__(
        self,
        min_differential_bps: float = 20.0,
        min_z_score: float = 2.0,
        max_position_usd: float = 500_000,
        max_convergence_days: int = 30,
        venue_pairs: Optional[List[Tuple[str, str]]] = None,
        venue_costs: Optional[Dict[str, VenueCosts]] = None,
        lookback: int = 30
    ):
        """
        Initialize strategy.
        
        Args:
            min_differential_bps: Minimum net differential to trade
            min_z_score: Minimum z-score for entry
            max_position_usd: Maximum position size
            max_convergence_days: Maximum expected convergence time
            venue_pairs: Venue pairs to compare
            venue_costs: Custom venue costs
            lookback: Lookback for z-score calculation
        """
        self.min_differential_bps = min_differential_bps
        self.min_z_score = min_z_score
        self.max_position_usd = max_position_usd
        self.max_convergence_days = max_convergence_days
        self.venue_pairs = venue_pairs or DEFAULT_CROSS_VENUE_PARAMS['venue_pairs']
        self.venue_costs = venue_costs or DEFAULT_VENUE_COSTS
        self.lookback = lookback
        
        # History tracking
        self._differential_history: Dict[Tuple[str, str], List[float]] = {}
        
        logger.info(f"CrossVenueBasisStrategy initialized: pairs={len(self.venue_pairs)}")
    
    def calculate_cross_venue_differential(
        self,
        curve1: TermStructureCurve,
        curve2: TermStructureCurve,
        target_dte: int = 30
    ) -> Dict[str, Any]:
        """
        Calculate basis differential between two venues.
        
        Args:
            curve1: First venue curve
            curve2: Second venue curve
            target_dte: DTE for comparison
            
        Returns:
            Dict with differential analysis
        """
        basis1 = curve1.interpolate_basis(target_dte)
        basis2 = curve2.interpolate_basis(target_dte)
        
        differential = basis1 - basis2
        differential_bps = differential * 100
        
        # Get costs
        costs1 = self.venue_costs.get(curve1.venue.lower())
        costs2 = self.venue_costs.get(curve2.venue.lower())
        
        venue1_cost_bps = costs1.round_trip_taker_bps if costs1 else 20.0
        venue2_cost_bps = costs2.round_trip_taker_bps if costs2 else 20.0
        total_cost_bps = venue1_cost_bps + venue2_cost_bps
        
        net_differential_bps = abs(differential_bps) - total_cost_bps
        
        # Update history for z-score
        pair_key = (curve1.venue, curve2.venue)
        if pair_key not in self._differential_history:
            self._differential_history[pair_key] = []
        
        self._differential_history[pair_key].append(differential)
        if len(self._differential_history[pair_key]) > self.lookback:
            self._differential_history[pair_key] = \
                self._differential_history[pair_key][-self.lookback:]
        
        # Calculate z-score
        history = self._differential_history[pair_key]
        if len(history) >= 5:
            mean = np.mean(history)
            std = np.std(history)
            z_score = (differential - mean) / std if std > 0 else 0.0
            vol = std
        else:
            z_score = 0.0
            vol = 0.0
        
        # Determine direction
        if differential > 0:
            direction = f"Short {curve1.venue}, Long {curve2.venue}"
        else:
            direction = f"Long {curve1.venue}, Short {curve2.venue}"
        
        # Check profitability
        is_profitable = net_differential_bps > self.min_differential_bps
        
        return {
            'timestamp': curve1.as_of,
            'venue1': curve1.venue,
            'venue2': curve2.venue,
            'basis1_pct': basis1,
            'basis2_pct': basis2,
            'differential_pct': differential,
            'differential_bps': differential_bps,
            'z_score': z_score,
            'venue1_cost_bps': venue1_cost_bps,
            'venue2_cost_bps': venue2_cost_bps,
            'total_cost_bps': total_cost_bps,
            'net_differential_bps': net_differential_bps,
            'direction': direction,
            'is_profitable': is_profitable,
            'is_significant': abs(z_score) > self.min_z_score,
            'historical_vol': vol,
        }
    
    def find_opportunities(
        self,
        curves: Dict[str, TermStructureCurve],
        target_dte: int = 30
    ) -> List[CrossVenueOpportunity]:
        """
        Find all cross-venue arbitrage opportunities.
        
        Args:
            curves: Dict mapping venue name to curve
            target_dte: DTE for comparison
            
        Returns:
            List of CrossVenueOpportunity objects
        """
        opportunities = []
        
        for venue1, venue2 in self.venue_pairs:
            if venue1 not in curves or venue2 not in curves:
                continue
            
            curve1 = curves[venue1]
            curve2 = curves[venue2]
            
            analysis = self.calculate_cross_venue_differential(
                curve1, curve2, target_dte
            )
            
            if not analysis['is_profitable']:
                continue
            
            # Get venue types
            costs1 = self.venue_costs.get(venue1.lower())
            costs2 = self.venue_costs.get(venue2.lower())
            
            venue1_type = costs1.venue_type if costs1 else VenueType.CEX_FUTURES
            venue2_type = costs2.venue_type if costs2 else VenueType.CEX_FUTURES
            
            # Estimate convergence time
            if analysis['historical_vol'] > 0:
                expected_days = min(
                    int(abs(analysis['differential_pct']) / analysis['historical_vol'] * 5),
                    self.max_convergence_days
                )
            else:
                expected_days = 14
            
            # Calculate expected return
            expected_return_bps = analysis['net_differential_bps'] * 0.5  # Conservative
            annualized = expected_return_bps * (365 / expected_days) / 100
            
            # Position sizing
            max_size = min(
                self.max_position_usd,
                DEFAULT_VENUE_CAPACITY.get(venue1.lower(), 100_000_000) * 0.005,
                DEFAULT_VENUE_CAPACITY.get(venue2.lower(), 100_000_000) * 0.005,
            )
            
            recommended_size = max_size * min(abs(analysis['z_score']) / 3, 1.0)
            
            opportunity = CrossVenueOpportunity(
                timestamp=analysis['timestamp'],
                venue1=venue1,
                venue2=venue2,
                venue1_type=venue1_type,
                venue2_type=venue2_type,
                venue1_basis_pct=analysis['basis1_pct'],
                venue2_basis_pct=analysis['basis2_pct'],
                basis_differential=analysis['differential_pct'],
                z_score=analysis['z_score'],
                venue1_round_trip_bps=analysis['venue1_cost_bps'],
                venue2_round_trip_bps=analysis['venue2_cost_bps'],
                total_cost_bps=analysis['total_cost_bps'],
                net_differential_bps=analysis['net_differential_bps'],
                max_size_usd=max_size,
                recommended_size_usd=recommended_size,
                expected_convergence_days=expected_days,
                expected_return_bps=expected_return_bps,
                annualized_return_pct=annualized,
                historical_vol_spread=analysis['historical_vol'],
            )
            
            opportunities.append(opportunity)
        
        return opportunities
    
    def rank_opportunities(
        self,
        opportunities: List[CrossVenueOpportunity]
    ) -> List[CrossVenueOpportunity]:
        """
        Rank opportunities by attractiveness.
        
        Args:
            opportunities: List of opportunities
            
        Returns:
            Sorted list (best first)
        """
        return sorted(
            opportunities,
            key=lambda o: o.opportunity_score,
            reverse=True
        )
    
    def filter_actionable(
        self,
        opportunities: List[CrossVenueOpportunity]
    ) -> List[CrossVenueOpportunity]:
        """
        Filter to only actionable opportunities.
        
        Args:
            opportunities: List of opportunities
            
        Returns:
            Filtered list
        """
        return [
            o for o in opportunities
            if o.is_profitable and o.is_significant
        ]
    
    def summary(
        self,
        opportunities: List[CrossVenueOpportunity]
    ) -> pd.DataFrame:
        """
        Create summary DataFrame of opportunities.
        
        Args:
            opportunities: List of opportunities
            
        Returns:
            Summary DataFrame
        """
        if not opportunities:
            return pd.DataFrame()
        
        return pd.DataFrame([o.to_dict() for o in opportunities])


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'SignalType',
    'PositionStatus',
    # Regime parameters
    'REGIME_PARAMS',
    'CRISIS_PARAMS',
    # Kelly criterion and parameter functions
    'calculate_kelly_fraction',
    'get_regime_adjusted_params',
    'get_crisis_adjusted_params',
    # Backtest result
    'BacktestResult',
    # Strategy A: Calendar Spreads
    'CalendarSpreadStrategy',
    # Cross-venue basis (related to Strategy B)
    'CrossVenueBasisStrategy',
]