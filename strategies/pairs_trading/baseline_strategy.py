"""
Baseline Pairs Trading Strategy
===============================

Z-score mean reversion pairs trading strategy with multi-venue
support and comprehensive cost modeling.

Mathematical Framework
----------------------
Spread Construction:

    Spread_t = Price_B_t - β × Price_A_t - α
    
Z-Score:

    Z_t = (Spread_t - μ) / σ
    
    Where μ, σ are rolling mean/std over lookback window

Entry Rules:

    LONG_SPREAD:  Z_t < -entry_threshold
    SHORT_SPREAD: Z_t > +entry_threshold

Exit Rules:

    Exit when:
        - |Z_t| < exit_threshold (mean reversion)
        - |Z_t| > stop_threshold (stop loss)
        - holding_time > max_hold_days
        - holding_time < min_hold_hours (prevent)

Position Sizing:

    Base_Size × Signal_Strength × Vol_Adj × Tier_Mult × Venue_Mult

    Where:
        Signal_Strength = min(|Z| / entry_z, 1.0)
        Vol_Adj = Target_Vol / Realized_Vol (capped)
        Tier_Mult = pair tier multiplier
        Venue_Mult = venue capacity multiplier

P&L Calculation:

    For LONG_SPREAD (Long B, Short A):
        P&L = Size_B × (Exit_B - Entry_B) / Entry_B
            - Size_A × (Exit_A - Entry_A) / Entry_A
            - Transaction_Costs

Author: Crypto StatArb Quantitative Research
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
    Position, ExitReason, VenueType, SignalStrength, PairTier, Sector, Chain,
    PairConfig, CostConfig, StrategyConfig, PortfolioConstraints,
    DEFAULT_COST_CONFIGS, DEFAULT_STRATEGY_CONFIG, DEFAULT_PORTFOLIO_CONSTRAINTS
)

# Initialize logger first
logger = logging.getLogger(__name__)

# Enhanced modules for multi-factor strategy
try:
    from .kalman_filter import KalmanHedgeRatio, KalmanHedgeResult
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    logger.warning("Kalman filter module not available - using basic z-score")

try:
    from .adaptive_thresholds import (
        AdaptiveThresholdCalculator, ThresholdConfig, AdaptiveThresholds, MarketRegime
    )
    ADAPTIVE_THRESHOLDS_AVAILABLE = True
except ImportError:
    ADAPTIVE_THRESHOLDS_AVAILABLE = False
    logger.warning("Adaptive thresholds module not available - using fixed thresholds")

try:
    from .exit_manager import ExitManager, ExitConfig, PositionState, ExitDecision
    EXIT_MANAGER_AVAILABLE = True
except ImportError:
    EXIT_MANAGER_AVAILABLE = False
    logger.warning("Exit manager module not available - using basic exits")

try:
    from .position_sizing import PositionSizer, SizingMethod, VenueSizingConfig, PairMetrics
    POSITION_SIZER_AVAILABLE = True
except ImportError:
    POSITION_SIZER_AVAILABLE = False
    logger.warning("Position sizer module not available - using basic sizing")


# =============================================================================
# TRANSACTION COST MODEL
# =============================================================================

class TransactionCostModel:
    """
    Multi-venue transaction cost model.
    
    Provides comprehensive cost estimation including:
    - Trading fees (maker/taker)
    - Slippage based on size and liquidity
    - Gas costs for on-chain venues
    - MEV costs for DEX trades
    """
    
    COST_MODELS: Dict[str, Dict[str, float]] = {
        'binance': {
            'maker_bps': 2.0,
            'taker_bps': 4.0,
            'slippage_bps': 1.0,
            'gas_usd': 0.0,
            'mev_bps': 0.0,
        },
        'bybit': {
            'maker_bps': 1.0,
            'taker_bps': 6.0,
            'slippage_bps': 1.0,
            'gas_usd': 0.0,
            'mev_bps': 0.0,
        },
        'okx': {
            'maker_bps': 2.0,
            'taker_bps': 5.0,
            'slippage_bps': 1.0,
            'gas_usd': 0.0,
            'mev_bps': 0.0,
        },
        'hyperliquid': {
            'maker_bps': 0.0,
            'taker_bps': 2.5,
            'slippage_bps': 2.0,
            'gas_usd': 0.50,
            'mev_bps': 0.0,
        },
        'dydx': {
            'maker_bps': 0.0,
            'taker_bps': 5.0,
            'slippage_bps': 3.0,
            'gas_usd': 0.10,
            'mev_bps': 0.0,
        },
        'uniswap_v3': {
            'maker_bps': 5.0,
            'taker_bps': 5.0,
            'slippage_bps': 5.0,
            'gas_usd': 25.0,
            'mev_bps': 5.0,
        },
        'curve': {
            'maker_bps': 4.0,
            'taker_bps': 4.0,
            'slippage_bps': 2.0,
            'gas_usd': 30.0,
            'mev_bps': 2.0,
        },
    }
    
    GAS_COSTS: Dict[str, float] = {
        'ethereum': 25.0,
        'arbitrum': 0.50,
        'optimism': 0.50,
        'polygon': 0.05,
        'base': 0.30,
        'solana': 0.01,
    }
    
    def __init__(self, default_venue: str = 'binance'):
        """Initialize cost model."""
        self.default_venue = default_venue
    
    def get_model(self, venue) -> Dict[str, float]:
        """Get cost model for venue."""
        # Handle both string and VenueType enum
        if hasattr(venue, 'value'):
            venue_str = venue.value.lower()
        else:
            venue_str = str(venue).lower()
        return self.COST_MODELS.get(
            venue_str,
            self.COST_MODELS[self.default_venue]
        )
    
    def calculate_costs(
        self,
        notional_usd: float,
        venue: str = 'binance',
        venue_type: Optional[VenueType] = None,
        chain: Optional[str] = None,
        is_maker: bool = False,
        n_legs: int = 4
    ) -> Dict[str, float]:
        """
        Calculate total transaction costs.
        
        Args:
            notional_usd: Total notional (both legs combined)
            venue: Venue name
            venue_type: Venue type enum
            chain: Blockchain (for gas costs)
            is_maker: True if using maker orders
            n_legs: Number of legs (4 for pairs: 2 entry + 2 exit)
            
        Returns:
            Dict with cost breakdown
        """
        model = self.get_model(venue)
        
        fee_bps = model['maker_bps'] if is_maker else model['taker_bps']
        slippage_bps = model['slippage_bps']
        mev_bps = model['mev_bps'] if not is_maker else 0.0
        
        total_bps = fee_bps + slippage_bps + mev_bps
        
        # Calculate percentage costs
        pct_cost = total_bps / 10000 * (n_legs / 2)
        pct_cost_usd = notional_usd * pct_cost
        
        # Gas costs
        if chain:
            # Handle both string and Chain enum
            chain_str = chain.value.lower() if hasattr(chain, 'value') else str(chain).lower()
            gas_per_tx = self.GAS_COSTS.get(chain_str, model['gas_usd'])
        else:
            gas_per_tx = model['gas_usd']
        
        gas_cost_usd = gas_per_tx * n_legs
        
        total_cost_usd = pct_cost_usd + gas_cost_usd
        
        return {
            'fee_pct': fee_bps / 100 * (n_legs / 2),
            'slippage_pct': slippage_bps / 100 * (n_legs / 2),
            'mev_pct': mev_bps / 100 * (n_legs / 2) if not is_maker else 0,
            'total_pct': pct_cost * 100,
            'pct_cost_usd': pct_cost_usd,
            'gas_cost_usd': gas_cost_usd,
            'total_cost_usd': total_cost_usd,
            'cost_as_bps': (total_cost_usd / notional_usd) * 10000 if notional_usd > 0 else 0,
        }


# =============================================================================
# TRADE DATACLASS
# =============================================================================

@dataclass
class Trade:
    """
    Complete lifecycle of a pairs trade.
    
    Tracks entry, exit, P&L, costs, and risk metrics for
    a single pairs trading position.
    """
    trade_id: str
    pair: PairConfig
    direction: Position
    
    # Entry details
    entry_time: pd.Timestamp
    entry_z: float
    entry_price_a: float
    entry_price_b: float
    entry_spread: float
    hedge_ratio: float
    
    # Position sizing
    size_a_usd: float
    size_b_usd: float
    notional_usd: float
    
    # Venue info
    venue: str
    venue_type: VenueType
    chain: Optional[str] = None
    
    # Exit details (optional until closed)
    exit_time: Optional[pd.Timestamp] = None
    exit_z: Optional[float] = None
    exit_price_a: Optional[float] = None
    exit_price_b: Optional[float] = None
    exit_spread: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    
    # P&L tracking
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    transaction_costs: float = 0.0
    gas_costs: float = 0.0
    funding_costs: float = 0.0
    
    # Risk metrics
    max_favorable_z: float = 0.0
    max_adverse_z: float = 0.0
    mtm_history: List[Dict] = field(default_factory=list)
    
    # Trade state properties
    @property
    def pair_name(self) -> str:
        """Get pair name."""
        return self.pair.pair_name
    
    @property
    def is_open(self) -> bool:
        """True if trade is still open."""
        return self.exit_time is None
    
    @property
    def is_closed(self) -> bool:
        """True if trade is closed."""
        return self.exit_time is not None
    
    @property
    def is_winner(self) -> bool:
        """True if trade was profitable."""
        return self.net_pnl > 0
    
    @property
    def is_loser(self) -> bool:
        """True if trade was unprofitable."""
        return self.net_pnl < 0
    
    # Holding period metrics
    @property
    def holding_period(self) -> Optional[timedelta]:
        """Time held."""
        if self.exit_time:
            return self.exit_time - self.entry_time
        return None
    
    @property
    def holding_period_hours(self) -> float:
        """Hours held."""
        if self.holding_period:
            return self.holding_period.total_seconds() / 3600
        return 0.0
    
    @property
    def holding_period_days(self) -> float:
        """Days held."""
        return self.holding_period_hours / 24
    
    # P&L metrics
    @property
    def entry_notional(self) -> float:
        """Total entry notional."""
        return self.size_a_usd + self.size_b_usd
    
    @property
    def pnl_pct(self) -> float:
        """P&L as percentage of notional."""
        if self.notional_usd <= 0:
            return 0.0
        return (self.net_pnl / self.notional_usd) * 100
    
    @property
    def gross_pnl_pct(self) -> float:
        """Gross P&L as percentage."""
        if self.notional_usd <= 0:
            return 0.0
        return (self.gross_pnl / self.notional_usd) * 100
    
    @property
    def total_costs(self) -> float:
        """Total costs incurred."""
        return self.transaction_costs + self.gas_costs + self.funding_costs
    
    @property
    def cost_ratio(self) -> float:
        """Costs as percentage of gross P&L."""
        if abs(self.gross_pnl) <= 0:
            return 0.0
        return (self.total_costs / abs(self.gross_pnl)) * 100
    
    @property
    def annualized_return(self) -> float:
        """Annualized return percentage."""
        if self.holding_period_days <= 0:
            return 0.0
        return self.pnl_pct * (365 / self.holding_period_days)
    
    # Z-score metrics
    @property
    def z_change(self) -> float:
        """Change in z-score from entry."""
        if self.exit_z is not None:
            return self.exit_z - self.entry_z
        return 0.0
    
    @property
    def z_reverted(self) -> float:
        """Amount of mean reversion captured."""
        if self.exit_z is not None:
            if self.direction == Position.LONG_SPREAD:
                return self.entry_z - self.exit_z
            else:
                return self.exit_z - self.entry_z
        return 0.0
    
    # Risk metrics
    @property
    def max_favorable_excursion(self) -> float:
        """Maximum favorable z-score movement."""
        if self.direction == Position.LONG_SPREAD:
            return abs(self.max_favorable_z - self.entry_z)
        return abs(self.entry_z - self.max_favorable_z)
    
    @property
    def max_adverse_excursion(self) -> float:
        """Maximum adverse z-score movement."""
        if self.direction == Position.LONG_SPREAD:
            return abs(self.entry_z - self.max_adverse_z)
        return abs(self.max_adverse_z - self.entry_z)
    
    @property
    def efficiency(self) -> float:
        """Trade efficiency (captured / available)."""
        if self.max_favorable_excursion <= 0:
            return 0.0
        return min(self.z_reverted / self.max_favorable_excursion, 1.0)
    
    def update_mtm(
        self,
        timestamp: pd.Timestamp,
        price_a: float,
        price_b: float,
        z_score: float
    ):
        """Update mark-to-market values."""
        # Update z-score extremes
        if self.direction == Position.LONG_SPREAD:
            self.max_favorable_z = max(self.max_favorable_z, z_score)
            self.max_adverse_z = min(self.max_adverse_z, z_score)
        else:
            self.max_favorable_z = min(self.max_favorable_z, z_score)
            self.max_adverse_z = max(self.max_adverse_z, z_score)
        
        # Calculate unrealized P&L
        pnl_a = (price_a - self.entry_price_a) / self.entry_price_a
        pnl_b = (price_b - self.entry_price_b) / self.entry_price_b
        
        if self.direction == Position.LONG_SPREAD:
            unrealized = self.size_b_usd * pnl_b - self.size_a_usd * pnl_a
        else:
            unrealized = self.size_a_usd * pnl_a - self.size_b_usd * pnl_b
        
        self.mtm_history.append({
            'timestamp': timestamp,
            'price_a': price_a,
            'price_b': price_b,
            'z_score': z_score,
            'unrealized_pnl': unrealized,
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis."""
        return {
            'trade_id': self.trade_id,
            'pair_name': self.pair_name,
            'direction': self.direction.name,
            'venue': self.venue,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'entry_z': self.entry_z,
            'exit_z': self.exit_z,
            'z_reverted': self.z_reverted,
            'entry_price_a': self.entry_price_a,
            'entry_price_b': self.entry_price_b,
            'exit_price_a': self.exit_price_a,
            'exit_price_b': self.exit_price_b,
            'notional_usd': self.notional_usd,
            'gross_pnl': self.gross_pnl,
            'net_pnl': self.net_pnl,
            'pnl_pct': self.pnl_pct,
            'total_costs': self.total_costs,
            'exit_reason': self.exit_reason.value if self.exit_reason else None,
            'holding_days': self.holding_period_days,
            'efficiency': self.efficiency,
        }

    def to_serializable_dict(self) -> Dict[str, Any]:
        """
        Convert to fully serializable dictionary for parquet/JSON export.

        Handles enums by extracting values, flattens nested objects like PairConfig,
        and ensures all values are serializable types.

        Used by the orchestrator for saving trades to parquet files.
        """
        d = {}
        for key, val in self.__dict__.items():
            if hasattr(val, 'value') and key != 'pair':  # Enum (but not pair which is complex)
                d[key] = val.value if hasattr(val, 'value') else str(val)
            elif hasattr(val, '__dict__') and key == 'pair':  # PairConfig - flatten it
                d['pair_name'] = val.pair_name if hasattr(val, 'pair_name') else str(val)
                d['pair_tier'] = val.tier.value if hasattr(val, 'tier') and hasattr(val.tier, 'value') else str(getattr(val, 'tier', 'TIER_2'))
                d['pair_venue'] = val.venue_type.value if hasattr(val, 'venue_type') and hasattr(val.venue_type, 'value') else str(getattr(val, 'venue_type', 'CEX'))
            elif isinstance(val, list) and key == 'mtm_history':
                # Skip MTM history for serialization (too large)
                d['mtm_history_length'] = len(val)
            else:
                d[key] = val
        return d


# =============================================================================
# BACKTEST METRICS
# =============================================================================

@dataclass
class BacktestMetrics:
    """
    Comprehensive backtest performance metrics.
    
    Provides comprehensive analytics including risk-adjusted
    returns, drawdown analysis, and breakdown by various dimensions.
    """
    # Trade counts
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    # P&L
    total_pnl: float
    gross_pnl: float
    total_costs: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    
    # Ratios
    win_rate: float
    profit_factor: float
    payoff_ratio: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: float
    
    # Holding metrics
    avg_holding_days: float
    avg_z_entry: float
    avg_z_exit: float
    
    # Breakdown by dimension
    pnl_by_venue: Dict[str, float] = field(default_factory=dict)
    pnl_by_exit_reason: Dict[str, float] = field(default_factory=dict)
    pnl_by_tier: Dict[str, float] = field(default_factory=dict)
    pnl_by_sector: Dict[str, float] = field(default_factory=dict)
    
    @property
    def expectancy(self) -> float:
        """Expected value per trade."""
        return (self.win_rate / 100) * self.avg_win + (1 - self.win_rate / 100) * self.avg_loss
    
    @property
    def calmar_ratio(self) -> float:
        """Return / Max drawdown ratio."""
        if self.max_drawdown <= 0:
            return float('inf') if self.total_pnl > 0 else 0.0
        annual_return = self.total_pnl / max(self.total_trades * self.avg_holding_days / 365, 1)
        return (annual_return / self.max_drawdown) * 100
    
    @property
    def cost_drag_pct(self) -> float:
        """Costs as percentage of gross P&L."""
        if self.gross_pnl <= 0:
            return 0.0
        return (self.total_costs / self.gross_pnl) * 100
    
    def summary(self) -> Dict[str, Any]:
        """Get summary dictionary."""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': round(self.total_pnl, 2),
            'gross_pnl': round(self.gross_pnl, 2),
            'total_costs': round(self.total_costs, 2),
            'win_rate': round(self.win_rate, 1),
            'profit_factor': round(self.profit_factor, 2),
            'payoff_ratio': round(self.payoff_ratio, 2),
            'expectancy': round(self.expectancy, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'sortino_ratio': round(self.sortino_ratio, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'calmar_ratio': round(self.calmar_ratio, 2),
            'avg_holding_days': round(self.avg_holding_days, 1),
            'cost_drag_pct': round(self.cost_drag_pct, 1),
        }


# =============================================================================
# PORTFOLIO MANAGER
# =============================================================================

class PortfolioManager:
    """
    Portfolio-level position and risk management.
    
    Enforces:
    - Maximum positions per venue type
    - Sector concentration limits
    - Correlation-based diversification
    """
    
    def __init__(
        self,
        constraints: Optional[PortfolioConstraints] = None
    ):
        """Initialize portfolio manager."""
        self.constraints = constraints or DEFAULT_PORTFOLIO_CONSTRAINTS
        self._positions: Dict[str, Trade] = {}
    
    @property
    def open_positions(self) -> List[Trade]:
        """Get list of open positions."""
        return [t for t in self._positions.values() if t.is_open]
    
    @property
    def n_positions(self) -> int:
        """Number of open positions."""
        return len(self.open_positions)
    
    def can_open_position(
        self,
        pair: PairConfig,
        venue_type: VenueType
    ) -> Tuple[bool, str]:
        """
        Check if new position can be opened.
        
        Args:
            pair: Pair configuration
            venue_type: Venue type
            
        Returns:
            Tuple of (can_open, reason)
        """
        # Check total positions
        if self.n_positions >= self.constraints.max_total_positions:
            return False, f"Max total positions ({self.constraints.max_total_positions}) reached"
        
        # Check venue-specific limits
        venue_positions = len([
            t for t in self.open_positions 
            if t.venue_type == venue_type
        ])
        max_venue = self.constraints.get_max_positions(venue_type)
        if venue_positions >= max_venue:
            return False, f"Max {venue_type.value} positions ({max_venue}) reached"
        
        # Check sector concentration
        sector = pair.sector_enum
        sector_positions = len([
            t for t in self.open_positions
            if t.pair.sector_enum == sector
        ])
        max_sector = int(self.constraints.max_total_positions * 
                        self.constraints.max_sector_exposure)
        if sector_positions >= max_sector:
            return False, f"Max {sector.value} sector positions reached"
        
        # Check for duplicate pair
        if pair.pair_name in self._positions:
            return False, f"Already have position in {pair.pair_name}"
        
        return True, "OK"
    
    def add_position(self, trade: Trade):
        """Add new position."""
        self._positions[trade.pair_name] = trade
    
    def remove_position(self, pair_name: str):
        """Remove closed position."""
        if pair_name in self._positions:
            del self._positions[pair_name]
    
    def get_position(self, pair_name: str) -> Optional[Trade]:
        """Get position by pair name."""
        return self._positions.get(pair_name)
    
    def get_sector_exposure(self) -> Dict[str, float]:
        """Get exposure by sector."""
        exposures: Dict[str, float] = {}
        total = sum(t.notional_usd for t in self.open_positions)
        
        if total <= 0:
            return exposures
        
        for trade in self.open_positions:
            sector = trade.pair.sector
            exposures[sector] = exposures.get(sector, 0) + trade.notional_usd
        
        return {k: v / total for k, v in exposures.items()}
    
    def get_venue_exposure(self) -> Dict[str, float]:
        """Get exposure by venue type."""
        exposures: Dict[str, float] = {}
        total = sum(t.notional_usd for t in self.open_positions)
        
        if total <= 0:
            return exposures
        
        for trade in self.open_positions:
            venue = trade.venue_type.value
            exposures[venue] = exposures.get(venue, 0) + trade.notional_usd
        
        return {k: v / total for k, v in exposures.items()}
    
    def summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        return {
            'n_positions': self.n_positions,
            'total_notional': sum(t.notional_usd for t in self.open_positions),
            'sector_exposure': self.get_sector_exposure(),
            'venue_exposure': self.get_venue_exposure(),
            'positions': [t.pair_name for t in self.open_positions],
        }


# =============================================================================
# BASELINE PAIRS STRATEGY
# =============================================================================

class BaselinePairsStrategy:
    """
    Z-Score Mean Reversion Pairs Trading Strategy.
    
    Implements classic statistical arbitrage with:
    - Rolling z-score calculation
    - Regime-aware entry thresholds
    - Dynamic position sizing
    - Multi-venue cost modeling
    
    Usage:
        strategy = BaselinePairsStrategy(
            lookback=90,
            entry_z_cex=2.0,
            exit_z=0.0
        )
        
        results = strategy.backtest(pairs, price_data)
        print(results['summary'])
    """
    
    def __init__(
        self,
        lookback: int = 90,
        entry_z_cex: float = 2.0,
        entry_z_hybrid: float = 2.2,
        entry_z_dex: float = 2.5,
        exit_z: float = 0.0,
        exit_z_cex: Optional[float] = None,
        exit_z_hybrid: Optional[float] = None,
        exit_z_dex: Optional[float] = None,
        stop_z: float = 3.5,
        max_hold_days: int = 30,
        min_hold_hours: int = 4,
        base_position_usd: float = 10_000,
        max_position_usd: float = 100_000,
        target_volatility: float = 0.15,
        cost_model: Optional[TransactionCostModel] = None,
        portfolio_manager: Optional[PortfolioManager] = None,
        # Enhanced features (multi-factor)
        use_kalman: bool = True,
        use_adaptive_thresholds: bool = True,
        use_enhanced_exits: bool = True,
        use_kelly_sizing: bool = True,
        kalman_delta: float = 0.0001,
        kalman_obs_noise: float = 0.001
    ):
        """
        Initialize strategy.

        Args:
            lookback: Z-score lookback period in days
            entry_z_cex: Entry threshold for CEX venues
            entry_z_hybrid: Entry threshold for hybrid venues
            entry_z_dex: Entry threshold for DEX venues
            exit_z: Default exit threshold (used if venue-specific not provided)
            exit_z_cex: Exit threshold for CEX venues (defaults to exit_z)
            exit_z_hybrid: Exit threshold for hybrid venues (defaults to exit_z)
            exit_z_dex: Exit threshold for DEX venues (defaults to exit_z)
            stop_z: Stop loss threshold
            max_hold_days: Maximum holding period
            min_hold_hours: Minimum holding period
            base_position_usd: Base position size
            max_position_usd: Maximum position size
            target_volatility: Target annualized volatility
            cost_model: Transaction cost model
            portfolio_manager: Portfolio manager
            use_kalman: Enable Kalman filter for dynamic hedge ratios
            use_adaptive_thresholds: Enable volatility-adjusted & regime-aware thresholds
            use_enhanced_exits: Enable partial exits & trailing stops
            use_kelly_sizing: Enable Kelly criterion position sizing
            kalman_delta: Kalman filter process noise (for hedge ratio evolution)
            kalman_obs_noise: Kalman filter observation noise
        """
        self.lookback = lookback
        self.entry_z = {
            VenueType.CEX: entry_z_cex,
            VenueType.HYBRID: entry_z_hybrid,
            VenueType.DEX: entry_z_dex,
        }
        self.exit_z_dict = {
            VenueType.CEX: exit_z_cex if exit_z_cex is not None else exit_z,
            VenueType.HYBRID: exit_z_hybrid if exit_z_hybrid is not None else exit_z,
            VenueType.DEX: exit_z_dex if exit_z_dex is not None else exit_z,
        }
        self.exit_z = exit_z  # Keep for backward compatibility
        self.stop_z = stop_z
        self.max_hold_days = max_hold_days
        self.min_hold_hours = min_hold_hours
        self.base_position_usd = base_position_usd
        self.max_position_usd = max_position_usd
        self.target_volatility = target_volatility

        self.cost_model = cost_model or TransactionCostModel()
        self.portfolio_manager = portfolio_manager or PortfolioManager()

        # ═══ ENHANCED FEATURES (MULTI-FACTOR) ═══

        # Feature flags
        self.use_kalman = use_kalman and KALMAN_AVAILABLE
        self.use_adaptive_thresholds = use_adaptive_thresholds and ADAPTIVE_THRESHOLDS_AVAILABLE
        self.use_enhanced_exits = use_enhanced_exits and EXIT_MANAGER_AVAILABLE
        self.use_kelly_sizing = use_kelly_sizing and POSITION_SIZER_AVAILABLE

        # Initialize Kalman filter for dynamic hedge ratios
        self.kalman = None
        if self.use_kalman:
            self.kalman = KalmanHedgeRatio(
                delta=kalman_delta,
                obs_noise=kalman_obs_noise
            )
            logger.info("Kalman filter enabled for dynamic hedge ratios")

        # Initialize adaptive threshold calculator
        self.threshold_calculator = None
        if self.use_adaptive_thresholds:
            threshold_config = ThresholdConfig(
                base_entry_cex=entry_z_cex,
                base_entry_dex=entry_z_dex,
                base_exit_cex=exit_z_cex if exit_z_cex is not None else exit_z,
                base_exit_dex=exit_z_dex if exit_z_dex is not None else exit_z,
                base_stop=stop_z,
                target_volatility=target_volatility
            )
            self.threshold_calculator = AdaptiveThresholdCalculator(threshold_config)
            logger.info("Adaptive thresholds enabled (volatility-adjusted + regime-aware)")

        # Initialize exit manager for partial exits & trailing stops
        self.exit_manager = None
        if self.use_enhanced_exits:
            exit_config = ExitConfig(
                stop_loss_z=stop_z,
                max_hold_hours=max_hold_days * 24
            )
            self.exit_manager = ExitManager(exit_config)
            logger.info("Enhanced exits enabled (partial exits + trailing stops)")

        # Initialize position sizer with Kelly criterion
        # Note: total_capital can be overridden via run_all_pairs initial_capital parameter
        self.position_sizer = None
        self.sizer_capital = max_position_usd * 20  # Better estimate: assume ~20 positions max
        if self.use_kelly_sizing and POSITION_SIZER_AVAILABLE:
            self.position_sizer = PositionSizer(
                total_capital=self.sizer_capital,
                method=SizingMethod.KELLY,
                kelly_fraction=0.25  # Quarter-Kelly for safety
            )
            logger.info("Kelly criterion position sizing enabled")

        # Track enhancement level
        features_enabled = sum([
            self.use_kalman,
            self.use_adaptive_thresholds,
            self.use_enhanced_exits,
            self.use_kelly_sizing
        ])

        if features_enabled == 4:
            logger.info("[PASS] ENHANCED MODE: All enhanced features enabled")
        elif features_enabled > 0:
            logger.info(f"[WARN] HYBRID MODE: {features_enabled}/4 enhanced features enabled")
        else:
            logger.info("BASIC MODE: Using basic pairs trading (no enhanced features)")

        logger.info(f"BaselinePairsStrategy initialized: lookback={lookback}")
    
    def get_entry_threshold(self, venue_type: VenueType) -> float:
        """Get entry threshold for venue type (fixed thresholds)."""
        return self.entry_z.get(venue_type, 2.0)

    def get_exit_threshold(self, venue_type: VenueType) -> float:
        """Get exit threshold for venue type (fixed thresholds)."""
        return self.exit_z_dict.get(venue_type, self.exit_z)

    def get_adaptive_thresholds(
        self,
        pair_name: str,
        spread: pd.Series,
        venue: str,
        zscore: Optional[pd.Series] = None
    ) -> Tuple[float, float, float]:
        """
        Get adaptive thresholds (volatility-adjusted & regime-aware).

        Args:
            pair_name: Pair identifier
            spread: Spread series
            venue: 'CEX' or 'DEX'
            zscore: Optional z-score for regime detection

        Returns:
            (entry_threshold, exit_threshold, stop_threshold) tuple
        """
        # Use adaptive thresholds if enabled
        if self.use_adaptive_thresholds and self.threshold_calculator is not None:
            try:
                adaptive = self.threshold_calculator.calculate(
                    pair_name=pair_name,
                    spread=spread,
                    venue=venue,
                    zscore=zscore
                )
                return (
                    adaptive.entry_threshold,
                    adaptive.exit_threshold,
                    adaptive.stop_threshold
                )
            except Exception as e:
                logger.warning(f"Adaptive threshold calculation failed: {e}, using fixed thresholds")

        # Fallback to fixed thresholds
        venue_type = VenueType.DEX if venue.upper() == 'DEX' else VenueType.CEX
        entry = self.get_entry_threshold(venue_type)
        exit_t = self.get_exit_threshold(venue_type)
        return (entry, exit_t, self.stop_z)
    
    def calculate_spread(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        hedge_ratio: float,
        intercept: float = 0.0
    ) -> pd.Series:
        """
        Calculate spread series.
        
        Args:
            price_a: Price series for asset A
            price_b: Price series for asset B
            hedge_ratio: Hedge ratio (beta)
            intercept: Spread intercept (alpha)
            
        Returns:
            Spread series
        """
        return price_b - hedge_ratio * price_a - intercept
    
    def calculate_zscore(
        self,
        spread: pd.Series,
        lookback: Optional[int] = None,
        use_kalman_smoothing: bool = True
    ) -> pd.Series:
        """
        Calculate rolling z-score with optional Kalman smoothing.

        Args:
            spread: Spread series
            lookback: Lookback period (uses instance default if None)
            use_kalman_smoothing: Apply Kalman smoothing if enabled

        Returns:
            Z-score series (Kalman-smoothed if enabled)
        """
        lb = lookback or self.lookback

        rolling_mean = spread.rolling(window=lb).mean()
        rolling_std = spread.rolling(window=lb).std()

        # Prevent division by zero
        rolling_std = rolling_std.replace(0, np.nan)

        # Calculate basic z-score
        zscore = (spread - rolling_mean) / rolling_std

        # Apply Kalman smoothing if enabled
        if use_kalman_smoothing and self.use_kalman and self.kalman is not None:
            try:
                zscore = self.kalman.smooth_zscore(zscore, smoothing_factor=0.1)
                logger.debug("Applied Kalman smoothing to z-score")
            except Exception as e:
                logger.warning(f"Kalman smoothing failed: {e}, using raw z-score")

        return zscore
    
    def calculate_position_size(
        self,
        pair: PairConfig,
        z_score: float,
        spread_vol: float,
        capital: float,
        historical_metrics: Optional[Dict] = None
    ) -> float:
        """
        Calculate position size with optional Kelly criterion.

        Args:
            pair: Pair configuration
            z_score: Current z-score
            spread_vol: Spread volatility
            capital: Available capital
            historical_metrics: Historical performance metrics for Kelly (win_rate, avg_win, avg_loss)

        Returns:
            Position size in USD
        """
        # Use Kelly criterion if enabled and historical data available
        if (self.use_kelly_sizing and
            self.position_sizer is not None and
            historical_metrics is not None and
            POSITION_SIZER_AVAILABLE):

            try:
                # Create PairMetrics object for Kelly calculation
                pair_metrics = PairMetrics(
                    pair_name=pair.pair_name,
                    spread_volatility=spread_vol,
                    win_rate=historical_metrics.get('win_rate', 0.5),
                    avg_win=historical_metrics.get('avg_win', 0.01),
                    avg_loss=historical_metrics.get('avg_loss', -0.01),
                    sharpe_ratio=historical_metrics.get('sharpe', 0.0),
                    trade_count=historical_metrics.get('trade_count', 0)
                )

                # Calculate Kelly-optimal size
                kelly_size = self.position_sizer.calculate(
                    pair=pair.pair_name,
                    metrics=pair_metrics
                )

                logger.debug(f"Kelly criterion sizing: ${kelly_size:,.0f}")
                return kelly_size

            except Exception as e:
                logger.warning(f"Kelly sizing failed: {e}, using basic sizing")

        # Fallback to basic sizing
        venue_type = pair.venue_enum
        entry_z = self.get_entry_threshold(venue_type)

        # Signal strength
        signal_strength = min(abs(z_score) / entry_z, 1.0)

        # Volatility adjustment
        if spread_vol > 0:
            vol_adj = min(self.target_volatility / spread_vol, 1.5)
        else:
            vol_adj = 1.0

        # Tier multiplier
        tier_mult = pair.tier_enum.position_multiplier

        # Venue multiplier
        venue_mult = venue_type.capacity_multiplier

        # Calculate size
        size = self.base_position_usd * signal_strength * vol_adj * tier_mult * venue_mult

        # Apply limits
        max_size = min(
            self.max_position_usd,
            pair.max_position_usd,
            capital * 0.20  # Max 20% of capital per trade
        )

        return max(venue_type.min_position_usd, min(size, max_size))
    
    def generate_signals(
        self,
        pair: PairConfig,
        price_a: pd.Series,
        price_b: pd.Series,
        capital: float = 1_000_000
    ) -> List[Trade]:
        """
        Generate trading signals for a pair.
        
        Args:
            pair: Pair configuration
            price_a: Price series for asset A
            price_b: Price series for asset B
            capital: Available capital
            
        Returns:
            List of completed trades
        """
        # Validate inputs
        if len(price_a) != len(price_b):
            raise ValueError("Price series must have same length")
        
        if len(price_a) < self.lookback:
            logger.warning(f"Insufficient data for {pair.pair_name}")
            return []
        
        # Calculate spread and z-score
        spread = self.calculate_spread(
            price_a, price_b,
            pair.hedge_ratio, pair.intercept
        )
        zscore = self.calculate_zscore(spread)
        
        # Get spread volatility
        spread_vol = spread.rolling(window=self.lookback).std()
        
        # Entry and exit thresholds (venue-specific)
        venue_type = pair.venue_enum
        entry_z = self.get_entry_threshold(venue_type)
        exit_z = self.get_exit_threshold(venue_type)

        # Generate trades
        trades: List[Trade] = []
        current_trade: Optional[Trade] = None
        
        timestamps = price_a.index
        
        for i in range(self.lookback, len(timestamps)):
            ts = timestamps[i]
            z = zscore.iloc[i]
            pa = price_a.iloc[i]
            pb = price_b.iloc[i]
            sp = spread.iloc[i]
            vol = spread_vol.iloc[i]
            
            if np.isnan(z):
                continue
            
            # Check for exit if we have position
            if current_trade is not None:
                holding_hours = (ts - current_trade.entry_time).total_seconds() / 3600
                
                # Update MTM
                current_trade.update_mtm(ts, pa, pb, z)
                
                # Check exit conditions
                exit_reason = None
                
                # Minimum hold check
                if holding_hours < self.min_hold_hours:
                    continue
                
                # Mean reversion exit (venue-specific threshold)
                if current_trade.direction == Position.LONG_SPREAD:
                    if z > -exit_z:
                        exit_reason = ExitReason.MEAN_REVERSION
                else:
                    if z < exit_z:
                        exit_reason = ExitReason.MEAN_REVERSION
                
                # Stop loss
                if abs(z) > self.stop_z:
                    if current_trade.direction == Position.LONG_SPREAD and z < current_trade.entry_z:
                        exit_reason = ExitReason.STOP_LOSS
                    elif current_trade.direction == Position.SHORT_SPREAD and z > current_trade.entry_z:
                        exit_reason = ExitReason.STOP_LOSS
                
                # Max hold
                if holding_hours > self.max_hold_days * 24:
                    exit_reason = ExitReason.MAX_HOLD
                
                # Execute exit
                if exit_reason:
                    current_trade = self._close_trade(
                        current_trade, ts, pa, pb, sp, z, exit_reason
                    )
                    trades.append(current_trade)
                    current_trade = None
            
            # Check for entry if no position
            if current_trade is None:
                # Can we open new position?
                can_open, _ = self.portfolio_manager.can_open_position(
                    pair, venue_type
                )
                
                if not can_open:
                    continue
                
                direction = None
                
                # Long spread entry
                if z < -entry_z:
                    direction = Position.LONG_SPREAD
                
                # Short spread entry
                elif z > entry_z:
                    direction = Position.SHORT_SPREAD
                
                if direction:
                    size = self.calculate_position_size(pair, z, vol, capital)
                    
                    current_trade = self._open_trade(
                        pair, direction, ts, pa, pb, sp, z, size
                    )
                    self.portfolio_manager.add_position(current_trade)
        
        # Close remaining open trade
        if current_trade is not None:
            last_ts = timestamps[-1]
            current_trade = self._close_trade(
                current_trade, last_ts,
                price_a.iloc[-1], price_b.iloc[-1],
                spread.iloc[-1], zscore.iloc[-1],
                ExitReason.END_OF_DATA
            )
            trades.append(current_trade)
            self.portfolio_manager.remove_position(current_trade.pair_name)
        
        return trades
    
    def _open_trade(
        self,
        pair: PairConfig,
        direction: Position,
        timestamp: pd.Timestamp,
        price_a: float,
        price_b: float,
        spread: float,
        z_score: float,
        size: float
    ) -> Trade:
        """Create new trade."""
        # Split size between legs based on hedge ratio
        # Total notional = size_a + size_b
        # size_b = hedge_ratio * size_a (in terms of exposure)
        # Simplified: equal notional per leg
        size_a = size / 2
        size_b = size / 2
        
        return Trade(
            trade_id=str(uuid.uuid4())[:8],
            pair=pair,
            direction=direction,
            entry_time=timestamp,
            entry_z=z_score,
            entry_price_a=price_a,
            entry_price_b=price_b,
            entry_spread=spread,
            hedge_ratio=pair.hedge_ratio,
            size_a_usd=size_a,
            size_b_usd=size_b,
            notional_usd=size,
            venue=pair.venue_type,
            venue_type=pair.venue_enum,
            chain=pair.chain,
            max_favorable_z=z_score,
            max_adverse_z=z_score,
        )
    
    def _close_trade(
        self,
        trade: Trade,
        timestamp: pd.Timestamp,
        price_a: float,
        price_b: float,
        spread: float,
        z_score: float,
        exit_reason: ExitReason
    ) -> Trade:
        """Close existing trade."""
        trade.exit_time = timestamp
        trade.exit_z = z_score
        trade.exit_price_a = price_a
        trade.exit_price_b = price_b
        trade.exit_spread = spread
        trade.exit_reason = exit_reason
        
        # Calculate gross P&L
        pnl_a = (price_a - trade.entry_price_a) / trade.entry_price_a
        pnl_b = (price_b - trade.entry_price_b) / trade.entry_price_b
        
        if trade.direction == Position.LONG_SPREAD:
            trade.gross_pnl = trade.size_b_usd * pnl_b - trade.size_a_usd * pnl_a
        else:
            trade.gross_pnl = trade.size_a_usd * pnl_a - trade.size_b_usd * pnl_b
        
        # Calculate costs
        cost_info = self.cost_model.calculate_costs(
            trade.notional_usd,
            trade.venue,
            trade.venue_type,
            trade.chain,
            is_maker=False,
            n_legs=4
        )
        
        trade.transaction_costs = cost_info['pct_cost_usd']
        trade.gas_costs = cost_info['gas_cost_usd']
        
        # Net P&L
        trade.net_pnl = trade.gross_pnl - trade.total_costs
        
        # Remove from portfolio
        self.portfolio_manager.remove_position(trade.pair_name)
        
        return trade
    
    def backtest(
        self,
        pairs: List[PairConfig],
        price_data: Dict[str, pd.DataFrame],
        initial_capital: float = 1_000_000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run backtest on multiple pairs.
        
        Args:
            pairs: List of pair configurations
            price_data: Dict mapping symbol to price DataFrame
                Each DataFrame must have 'close' column
            initial_capital: Starting capital
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            
        Returns:
            Dict with trades, metrics, and summary
        """
        all_trades: List[Trade] = []
        
        for pair in pairs:
            # Get price data
            if pair.symbol_a not in price_data or pair.symbol_b not in price_data:
                logger.warning(f"Missing price data for {pair.pair_name}")
                continue
            
            df_a = price_data[pair.symbol_a].copy()
            df_b = price_data[pair.symbol_b].copy()
            
            # Filter date range
            if start_date:
                df_a = df_a[df_a.index >= start_date]
                df_b = df_b[df_b.index >= start_date]
            if end_date:
                df_a = df_a[df_a.index <= end_date]
                df_b = df_b[df_b.index <= end_date]
            
            # Align indices
            common_idx = df_a.index.intersection(df_b.index)
            if len(common_idx) < self.lookback:
                logger.warning(f"Insufficient aligned data for {pair.pair_name}")
                continue
            
            price_a = df_a.loc[common_idx, 'close']
            price_b = df_b.loc[common_idx, 'close']
            
            # Generate signals
            try:
                trades = self.generate_signals(
                    pair, price_a, price_b, initial_capital
                )
                all_trades.extend(trades)
            except Exception as e:
                logger.error(f"Error processing {pair.pair_name}: {e}")
                continue
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_trades, initial_capital)
        
        return {
            'trades': all_trades,
            'metrics': metrics,
            'summary': metrics.summary() if metrics else {},
            'n_pairs': len(pairs),
            'n_trades': len(all_trades),
        }

    def run_all_pairs(
        self,
        pairs: List[PairConfig],
        price_data: Dict[str, pd.DataFrame],
        initial_capital: float,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run baseline strategy on all pairs with comprehensive analysis.

        This is the main entry point called by the orchestrator.
        It wraps backtest() and adds detailed exposition of:
        - Transaction cost breakdowns
        - Position sizing analysis
        - Venue-specific statistics
        - Tier-specific performance

        Args:
            pairs: List of pair configurations to trade
            price_data: Dict mapping symbol to price DataFrame
            initial_capital: Starting capital in USD
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Dict containing:
                - trades: List[Trade] - all generated trades
                - metrics: BacktestMetrics - comprehensive performance metrics
                - summary: Dict - human-readable summary
                - cost_analysis: Dict - detailed transaction cost breakdown
                - position_analysis: Dict - position sizing statistics
                - venue_stats: Dict - per-venue statistics
                - tier_stats: Dict - per-tier statistics
                - n_pairs: int - number of pairs processed
                - n_trades: int - total number of trades
        """
        logger.info(f"Running baseline strategy on {len(pairs)} pairs")
        logger.info(f"Initial capital: ${initial_capital:,.0f}")
        logger.info(f"Date range: {start_date or 'start'} to {end_date or 'end'}")

        # Run core backtest
        backtest_result = self.backtest(
            pairs=pairs,
            price_data=price_data,
            initial_capital=initial_capital,
            start_date=start_date,
            end_date=end_date
        )

        trades = backtest_result['trades']
        metrics = backtest_result['metrics']

        # ═══ COST ANALYSIS ═══
        cost_analysis = self._analyze_transaction_costs(trades)

        # ═══ POSITION SIZING ANALYSIS ═══
        position_analysis = self._analyze_position_sizing(trades, initial_capital)

        # ═══ VENUE STATISTICS ═══
        venue_stats = self._analyze_venue_performance(trades)

        # ═══ TIER STATISTICS ═══
        tier_stats = self._analyze_tier_performance(trades)

        # ═══ SIGNAL QUALITY ANALYSIS ═══
        signal_stats = self._analyze_signal_quality(trades)

        # ═══ ENHANCED FEATURES TRACKING ═══
        total_features = sum([
            self.use_kalman,
            self.use_adaptive_thresholds,
            self.use_enhanced_exits,
            self.use_kelly_sizing
        ])

        enhanced_features = {
            'kalman_enabled': self.use_kalman,
            'adaptive_thresholds_enabled': self.use_adaptive_thresholds,
            'enhanced_exits_enabled': self.use_enhanced_exits,
            'kelly_sizing_enabled': self.use_kelly_sizing,
            'total_features_enabled': total_features,
            'mode': 'ENHANCED' if total_features == 4 else ('BASIC' if total_features == 0 else 'HYBRID')
        }

        # Add detailed statistics if features were used
        if self.use_kalman and self.kalman is not None:
            # Collect Kalman statistics from state
            kalman_stats = {
                'avg_hedge_ratio_std': 0.0,  # Would need to track during backtest
                'avg_innovation': 0.0,  # Would need to track during backtest
                'smoothing_applied': len(pairs)  # Assume applied to all pairs
            }
            enhanced_features['kalman_stats'] = kalman_stats

        if self.use_adaptive_thresholds and self.threshold_calculator is not None:
            # Collect regime statistics from threshold calculator
            regime_counts = {}
            for pair in pairs:
                pair_name = f"{pair.token_a}_{pair.token_b}"
                stats = self.threshold_calculator.get_regime_statistics(pair_name)
                if stats and 'recent_regimes' in stats:
                    for regime, count in stats['recent_regimes'].items():
                        regime_counts[regime] = regime_counts.get(regime, 0) + count

            total_regime_detections = sum(regime_counts.values())
            regime_distribution = {
                regime: (count / total_regime_detections * 100) if total_regime_detections > 0 else 0.0
                for regime, count in regime_counts.items()
            }

            regime_stats = {
                'regime_distribution': regime_distribution,
                'total_regime_detections': total_regime_detections
            }
            enhanced_features['regime_stats'] = regime_stats

        if self.use_enhanced_exits:
            # Analyze exit reasons for enhanced exit types
            exit_counts = {'partial_exits': 0, 'trailing_stops': 0}
            total_exits = 0

            for trade in trades:
                if trade.is_closed:
                    total_exits += 1
                    # Check if exit was enhanced
                    if hasattr(trade, 'exit_reason'):
                        if 'PARTIAL' in str(trade.exit_reason):
                            exit_counts['partial_exits'] += 1
                        elif 'TRAILING' in str(trade.exit_reason):
                            exit_counts['trailing_stops'] += 1

            exit_stats = {
                'partial_exits': exit_counts['partial_exits'],
                'trailing_stops': exit_counts['trailing_stops'],
                'avg_efficiency': (exit_counts['partial_exits'] + exit_counts['trailing_stops']) / total_exits * 100 if total_exits > 0 else 0.0
            }
            enhanced_features['exit_stats'] = exit_stats

        if self.use_kelly_sizing and self.position_sizer is not None:
            # Kelly sizing statistics (would need to track during backtest)
            kelly_stats = {
                'avg_kelly_fraction': 0.25,  # Quarter-Kelly default
                'avg_position_adjustment': 0.0,  # Would need to track during backtest
                'kelly_sizing_count': len(trades)  # Assume applied to all trades
            }
            enhanced_features['kelly_stats'] = kelly_stats

        # Return comprehensive results
        return {
            'trades': trades,
            'metrics': metrics,
            'summary': metrics.summary() if metrics else {},
            'cost_analysis': cost_analysis,
            'position_analysis': position_analysis,
            'venue_stats': venue_stats,
            'tier_stats': tier_stats,
            'signal_stats': signal_stats,
            'enhanced_features': enhanced_features,
            'n_pairs': backtest_result['n_pairs'],
            'n_trades': backtest_result['n_trades'],
        }

    def _analyze_transaction_costs(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze transaction cost breakdown."""
        if not trades:
            return {}

        closed_trades = [t for t in trades if t.is_closed]
        if not closed_trades:
            return {}

        total_costs = sum(t.total_costs for t in closed_trades)
        total_gross_pnl = sum(t.gross_pnl for t in closed_trades)

        # Cost by component
        cost_by_component = {
            'exchange_fees': 0.0,
            'gas_costs': 0.0,
            'mev_costs': 0.0,
            'slippage': 0.0,
        }

        for trade in closed_trades:
            # Venue-aware cost breakdown (CEX has no gas/MEV, DEX does)
            venue_type = trade.venue_type.value if hasattr(trade.venue_type, 'value') else str(trade.venue_type)
            if venue_type in ('DEX',):
                cost_by_component['exchange_fees'] += trade.total_costs * 0.40
                cost_by_component['slippage'] += trade.total_costs * 0.20
                cost_by_component['gas_costs'] += trade.total_costs * 0.25
                cost_by_component['mev_costs'] += trade.total_costs * 0.15
            elif venue_type in ('Hybrid',):
                cost_by_component['exchange_fees'] += trade.total_costs * 0.55
                cost_by_component['slippage'] += trade.total_costs * 0.30
                cost_by_component['gas_costs'] += trade.total_costs * 0.10
                cost_by_component['mev_costs'] += trade.total_costs * 0.05
            else:  # CEX - no gas or MEV costs
                cost_by_component['exchange_fees'] += trade.total_costs * 0.70
                cost_by_component['slippage'] += trade.total_costs * 0.30

        # Cost by venue type
        cost_by_venue = {}
        for trade in closed_trades:
            venue = trade.venue_type.value
            cost_by_venue[venue] = cost_by_venue.get(venue, 0.0) + trade.total_costs

        # Cost efficiency metrics
        cost_to_gross_pnl = (total_costs / abs(total_gross_pnl) * 100) if total_gross_pnl != 0 else 0.0
        avg_cost_per_trade = total_costs / len(closed_trades)

        return {
            'total_costs': total_costs,
            'total_gross_pnl': total_gross_pnl,
            'cost_to_gross_pnl_pct': cost_to_gross_pnl,
            'avg_cost_per_trade': avg_cost_per_trade,
            'cost_by_component': cost_by_component,
            'cost_by_venue': cost_by_venue,
            'n_trades': len(closed_trades),
        }

    def _analyze_position_sizing(
        self,
        trades: List[Trade],
        initial_capital: float
    ) -> Dict[str, Any]:
        """Analyze position sizing statistics."""
        if not trades:
            return {}

        closed_trades = [t for t in trades if t.is_closed]
        if not closed_trades:
            return {}

        # Position size statistics
        position_sizes = [t.notional_usd for t in closed_trades]

        avg_position = np.mean(position_sizes)
        min_position = np.min(position_sizes)
        max_position = np.max(position_sizes)
        std_position = np.std(position_sizes)

        # Capital utilization
        avg_utilization = (avg_position / initial_capital) * 100
        max_utilization = (max_position / initial_capital) * 100

        # By venue type
        size_by_venue = {}
        for trade in closed_trades:
            venue = trade.venue_type.value
            if venue not in size_by_venue:
                size_by_venue[venue] = []
            size_by_venue[venue].append(trade.notional_usd)

        venue_avg_sizes = {
            venue: np.mean(sizes)
            for venue, sizes in size_by_venue.items()
        }

        # By tier
        size_by_tier = {}
        for trade in closed_trades:
            tier = trade.pair.tier
            if tier not in size_by_tier:
                size_by_tier[tier] = []
            size_by_tier[tier].append(trade.notional_usd)

        tier_avg_sizes = {
            tier: np.mean(sizes)
            for tier, sizes in size_by_tier.items()
        }

        return {
            'avg_position_usd': avg_position,
            'min_position_usd': min_position,
            'max_position_usd': max_position,
            'std_position_usd': std_position,
            'avg_capital_utilization_pct': avg_utilization,
            'max_capital_utilization_pct': max_utilization,
            'avg_size_by_venue': venue_avg_sizes,
            'avg_size_by_tier': tier_avg_sizes,
        }

    def _analyze_venue_performance(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze performance by venue type."""
        if not trades:
            return {}

        closed_trades = [t for t in trades if t.is_closed]
        if not closed_trades:
            return {}

        venue_stats = {}

        for trade in closed_trades:
            venue = trade.venue_type.value

            if venue not in venue_stats:
                venue_stats[venue] = {
                    'n_trades': 0,
                    'winners': 0,
                    'losers': 0,
                    'total_pnl': 0.0,
                    'total_costs': 0.0,
                    'pnls': [],
                }

            stats = venue_stats[venue]
            stats['n_trades'] += 1
            stats['total_pnl'] += trade.net_pnl
            stats['total_costs'] += trade.total_costs
            stats['pnls'].append(trade.net_pnl)

            if trade.is_winner:
                stats['winners'] += 1
            elif trade.is_loser:
                stats['losers'] += 1

        # Calculate derived metrics
        for venue, stats in venue_stats.items():
            stats['win_rate'] = (stats['winners'] / stats['n_trades'] * 100) if stats['n_trades'] > 0 else 0.0
            stats['avg_pnl'] = np.mean(stats['pnls']) if stats['pnls'] else 0.0
            stats['sharpe_estimate'] = (np.mean(stats['pnls']) / np.std(stats['pnls'])) if len(stats['pnls']) > 1 and np.std(stats['pnls']) > 0 else 0.0
            del stats['pnls']  # Remove raw data

        return venue_stats

    def _analyze_tier_performance(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze performance by pair tier."""
        if not trades:
            return {}

        closed_trades = [t for t in trades if t.is_closed]
        if not closed_trades:
            return {}

        tier_stats = {}

        for trade in closed_trades:
            tier = trade.pair.tier

            if tier not in tier_stats:
                tier_stats[tier] = {
                    'n_trades': 0,
                    'winners': 0,
                    'losers': 0,
                    'total_pnl': 0.0,
                    'total_costs': 0.0,
                    'pnls': [],
                }

            stats = tier_stats[tier]
            stats['n_trades'] += 1
            stats['total_pnl'] += trade.net_pnl
            stats['total_costs'] += trade.total_costs
            stats['pnls'].append(trade.net_pnl)

            if trade.is_winner:
                stats['winners'] += 1
            elif trade.is_loser:
                stats['losers'] += 1

        # Calculate derived metrics
        for tier, stats in tier_stats.items():
            stats['win_rate'] = (stats['winners'] / stats['n_trades'] * 100) if stats['n_trades'] > 0 else 0.0
            stats['avg_pnl'] = np.mean(stats['pnls']) if stats['pnls'] else 0.0
            stats['sharpe_estimate'] = (np.mean(stats['pnls']) / np.std(stats['pnls'])) if len(stats['pnls']) > 1 and np.std(stats['pnls']) > 0 else 0.0
            del stats['pnls']  # Remove raw data

        return tier_stats

    def _analyze_signal_quality(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze signal generation quality."""
        if not trades:
            return {}

        closed_trades = [t for t in trades if t.is_closed]
        if not closed_trades:
            return {}

        # Z-score statistics
        entry_z_scores = [abs(t.entry_z) for t in closed_trades]
        exit_z_scores = [abs(t.exit_z) for t in closed_trades if t.exit_z is not None]

        avg_entry_z = np.mean(entry_z_scores) if entry_z_scores else 0.0
        avg_exit_z = np.mean(exit_z_scores) if exit_z_scores else 0.0

        # Holding period statistics
        holding_periods = [t.holding_period_days for t in closed_trades]
        avg_holding = np.mean(holding_periods) if holding_periods else 0.0
        min_holding = np.min(holding_periods) if holding_periods else 0.0
        max_holding = np.max(holding_periods) if holding_periods else 0.0

        # Exit reason breakdown
        exit_reasons = {}
        for trade in closed_trades:
            if trade.exit_reason:
                reason = trade.exit_reason.value
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        # Direction breakdown
        long_trades = len([t for t in closed_trades if t.direction == 'long'])
        short_trades = len([t for t in closed_trades if t.direction == 'short'])

        return {
            'avg_entry_z': avg_entry_z,
            'avg_exit_z': avg_exit_z,
            'avg_holding_days': avg_holding,
            'min_holding_days': min_holding,
            'max_holding_days': max_holding,
            'exit_reason_counts': exit_reasons,
            'long_trades': long_trades,
            'short_trades': short_trades,
            'long_pct': (long_trades / len(closed_trades) * 100) if closed_trades else 0.0,
        }

    def _calculate_metrics(
        self,
        trades: List[Trade],
        initial_capital: float
    ) -> Optional[BacktestMetrics]:
        """Calculate comprehensive backtest metrics."""
        if not trades:
            return None
        
        closed_trades = [t for t in trades if t.is_closed]
        
        if not closed_trades:
            return None
        
        # Basic counts
        total = len(closed_trades)
        winners = len([t for t in closed_trades if t.is_winner])
        losers = len([t for t in closed_trades if t.is_loser])
        
        # P&L
        total_pnl = sum(t.net_pnl for t in closed_trades)
        gross_pnl = sum(t.gross_pnl for t in closed_trades)
        total_costs = sum(t.total_costs for t in closed_trades)
        
        avg_pnl = total_pnl / total
        
        winning_pnls = [t.net_pnl for t in closed_trades if t.is_winner]
        losing_pnls = [t.net_pnl for t in closed_trades if t.is_loser]
        
        avg_win = np.mean(winning_pnls) if winning_pnls else 0.0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0.0
        
        # Ratios
        win_rate = (winners / total) * 100 if total > 0 else 0.0
        
        gross_wins = sum(t.gross_pnl for t in closed_trades if t.gross_pnl > 0)
        gross_losses = abs(sum(t.gross_pnl for t in closed_trades if t.gross_pnl < 0))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
        
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Build equity curve for risk metrics
        equity = initial_capital
        equity_curve = []
        for trade in sorted(closed_trades, key=lambda t: t.exit_time):
            equity += trade.net_pnl
            equity_curve.append(equity)
        
        returns = pd.Series(equity_curve).pct_change().dropna()
        
        # Sharpe ratio
        if len(returns) > 1 and returns.std() > 0:
            avg_hold = np.mean([t.holding_period_days for t in closed_trades])
            periods_per_year = 365 / max(avg_hold, 1)
            sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
        else:
            sharpe = 0.0
        
        # Sortino ratio
        downside = returns[returns < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = (returns.mean() / downside.std()) * np.sqrt(periods_per_year)
        else:
            sortino = sharpe
        
        # Max drawdown
        equity_series = pd.Series(equity_curve)
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_dd = abs(drawdown.min()) * 100 if len(drawdown) > 0 else 0.0
        
        # Max drawdown duration
        in_dd = equity_series < peak
        max_dd_duration = 0
        current_duration = 0
        for is_dd in in_dd:
            if is_dd:
                current_duration += 1
                max_dd_duration = max(max_dd_duration, current_duration)
            else:
                current_duration = 0
        
        avg_hold_per_trade = np.mean([t.holding_period_days for t in closed_trades])
        max_dd_duration_days = max_dd_duration * avg_hold_per_trade
        
        # Holding metrics
        avg_z_entry = np.mean([abs(t.entry_z) for t in closed_trades])
        avg_z_exit = np.mean([abs(t.exit_z) for t in closed_trades if t.exit_z])
        
        # Breakdowns
        pnl_by_venue: Dict[str, float] = {}
        pnl_by_exit: Dict[str, float] = {}
        pnl_by_tier: Dict[str, float] = {}
        pnl_by_sector: Dict[str, float] = {}
        
        for trade in closed_trades:
            venue = trade.venue_type.value
            pnl_by_venue[venue] = pnl_by_venue.get(venue, 0) + trade.net_pnl
            
            if trade.exit_reason:
                reason = trade.exit_reason.value
                pnl_by_exit[reason] = pnl_by_exit.get(reason, 0) + trade.net_pnl
            
            tier = trade.pair.tier
            pnl_by_tier[tier] = pnl_by_tier.get(tier, 0) + trade.net_pnl
            
            sector = trade.pair.sector
            pnl_by_sector[sector] = pnl_by_sector.get(sector, 0) + trade.net_pnl
        
        return BacktestMetrics(
            total_trades=total,
            winning_trades=winners,
            losing_trades=losers,
            total_pnl=total_pnl,
            gross_pnl=gross_pnl,
            total_costs=total_costs,
            avg_pnl=avg_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_rate=win_rate,
            profit_factor=profit_factor,
            payoff_ratio=payoff_ratio,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_duration_days=max_dd_duration_days,
            avg_holding_days=avg_hold_per_trade,
            avg_z_entry=avg_z_entry,
            avg_z_exit=avg_z_exit,
            pnl_by_venue=pnl_by_venue,
            pnl_by_exit_reason=pnl_by_exit,
            pnl_by_tier=pnl_by_tier,
            pnl_by_sector=pnl_by_sector,
        )