"""
Comprehensive Backtesting Engine for Crypto Statistical Arbitrage

Comprehensive backtesting framework supporting multi-venue execution
(CEX, DEX, hybrid) with realistic transaction cost modeling, walk-forward
optimization, and comprehensive performance attribution.

Features:
    - Walk-forward optimization with configurable train/test windows
    - Multi-venue execution simulation (CEX, hybrid, DEX)
    - Realistic transaction cost modeling (fees, slippage, gas, MEV)
    - Risk management (position limits, drawdown stops, VaR limits)
    - Crisis period analysis (UST/LUNA, FTX, etc.)
    - Comprehensive performance metrics and attribution
    - Benchmark comparison and alpha/beta decomposition
    - Regime-aware performance analysis

Author: Tamer Atesyakar
Version: 2.0.0
Date: January 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
import warnings

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class VenueType(Enum):
    """Trading venue classification."""
    CEX = "CEX"
    HYBRID = "hybrid"
    DEX = "DEX"


class OrderType(Enum):
    """Order execution type."""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    VWAP = "vwap"


class OrderSide(Enum):
    """Order direction."""
    BUY = "buy"
    SELL = "sell"


class SignalType(Enum):
    """Strategy signal types."""
    LONG_ENTRY = "long_entry"
    LONG_EXIT = "long_exit"
    SHORT_ENTRY = "short_entry"
    SHORT_EXIT = "short_exit"
    HOLD = "hold"
    REDUCE = "reduce"
    INCREASE = "increase"


class RegimeType(Enum):
    """Market regime classification."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"


class ExitReason(Enum):
    """Trade exit reasons for attribution."""
    SIGNAL = "signal"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    TIME_STOP = "time_stop"
    DRAWDOWN_STOP = "drawdown_stop"
    END_OF_PERIOD = "end_of_period"
    MARGIN_CALL = "margin_call"
    RISK_LIMIT = "risk_limit"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Order:
    """
    Represents a single order.
    
    Attributes:
        timestamp: Order creation time
        symbol: Trading symbol
        side: Buy or sell
        quantity: Order quantity in base currency
        price: Limit price or reference price for market orders
        venue: Exchange/venue name
        venue_type: Type of venue (CEX, hybrid, DEX)
        order_type: Execution type
        order_id: Unique order identifier
        strategy: Strategy name that generated the order
        metadata: Additional order metadata
    """
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    venue: str
    venue_type: VenueType
    order_type: OrderType = OrderType.MARKET
    order_id: str = ""
    strategy: str = ""
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.order_id:
            self.order_id = f"{self.timestamp.strftime('%Y%m%d%H%M%S')}_{self.symbol}_{self.side.value}"
    
    @property
    def notional(self) -> float:
        """Order notional value."""
        return self.price * self.quantity


@dataclass
class Fill:
    """
    Represents an order fill with detailed cost breakdown.
    
    Attributes:
        order: Original order
        fill_price: Actual fill price (including slippage)
        fill_quantity: Filled quantity
        slippage: Slippage cost in USD
        fees: Exchange fees in USD
        gas_cost: Gas cost for on-chain transactions
        mev_cost: MEV extraction cost estimate
        total_cost: Sum of all costs
        fill_timestamp: Time of fill
    """
    order: Order
    fill_price: float
    fill_quantity: float
    slippage: float
    fees: float
    gas_cost: float
    mev_cost: float
    total_cost: float
    fill_timestamp: pd.Timestamp
    
    @property
    def notional(self) -> float:
        """Notional value of the fill."""
        return self.fill_price * self.fill_quantity
    
    @property
    def cost_bps(self) -> float:
        """Total cost in basis points."""
        return (self.total_cost / self.notional) * 10000 if self.notional > 0 else 0
    
    @property
    def effective_price(self) -> float:
        """Effective price including all costs."""
        if self.order.side == OrderSide.BUY:
            return self.fill_price + (self.total_cost / self.fill_quantity)
        else:
            return self.fill_price - (self.total_cost / self.fill_quantity)


@dataclass
class Position:
    """
    Represents an open position.
    
    Attributes:
        symbol: Trading symbol
        quantity: Position quantity (negative for short)
        avg_price: Average entry price
        venue: Trading venue
        venue_type: Venue classification
        entry_time: Position entry time
        strategy: Strategy name
        unrealized_pnl: Current unrealized P&L
        realized_pnl: Accumulated realized P&L
        margin_used: Margin requirement
        stop_loss: Stop loss price (optional)
        take_profit: Take profit price (optional)
    """
    symbol: str
    quantity: float
    avg_price: float
    venue: str
    venue_type: VenueType
    entry_time: pd.Timestamp
    strategy: str = ""
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    margin_used: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    highest_price: float = 0.0  # For trailing stops
    lowest_price: float = float('inf')
    
    @property
    def notional(self) -> float:
        """Position notional value."""
        return abs(self.quantity * self.avg_price)
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0
    
    @property
    def side(self) -> str:
        """Position side as string."""
        return "long" if self.is_long else "short"
    
    def update_unrealized_pnl(self, current_price: float) -> None:
        """Update unrealized P&L based on current price."""
        if self.is_long:
            self.unrealized_pnl = (current_price - self.avg_price) * self.quantity
        else:
            self.unrealized_pnl = (self.avg_price - current_price) * abs(self.quantity)
        
        # Track high/low for trailing stops
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)
    
    def check_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss is triggered."""
        if self.stop_loss is None:
            return False
        if self.is_long:
            return current_price <= self.stop_loss
        else:
            return current_price >= self.stop_loss
    
    def check_take_profit(self, current_price: float) -> bool:
        """Check if take profit is triggered."""
        if self.take_profit is None:
            return False
        if self.is_long:
            return current_price >= self.take_profit
        else:
            return current_price <= self.take_profit


@dataclass
class TradeRecord:
    """
    Complete record of a round-trip trade.
    
    Contains all information needed for performance attribution
    and trade analysis.
    """
    trade_id: str
    symbol: str
    strategy: str
    venue: str
    venue_type: VenueType
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # 'long' or 'short'
    gross_pnl: float
    fees: float
    slippage: float
    gas_cost: float
    mev_cost: float
    net_pnl: float
    holding_period_hours: float
    exit_reason: str
    regime: str = ""
    entry_signal_strength: float = 0.0
    exit_signal_strength: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    @property
    def return_pct(self) -> float:
        """Return percentage on entry notional."""
        entry_notional = abs(self.entry_price * self.quantity)
        return (self.net_pnl / entry_notional * 100) if entry_notional > 0 else 0
    
    @property
    def gross_return_pct(self) -> float:
        """Gross return percentage (before costs)."""
        entry_notional = abs(self.entry_price * self.quantity)
        return (self.gross_pnl / entry_notional * 100) if entry_notional > 0 else 0
    
    @property
    def cost_pct(self) -> float:
        """Total cost as percentage of notional."""
        entry_notional = abs(self.entry_price * self.quantity)
        total_cost = self.fees + self.slippage + self.gas_cost + self.mev_cost
        return (total_cost / entry_notional * 100) if entry_notional > 0 else 0
    
    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.net_pnl > 0
    
    @property
    def r_multiple(self) -> float:
        """R-multiple (if stop loss was used)."""
        if self.max_adverse_excursion == 0:
            return 0.0
        return self.net_pnl / abs(self.max_adverse_excursion)


@dataclass
class BacktestConfig:
    """
    Comprehensive backtesting configuration.
    
    Controls all aspects of the backtest including capital,
    risk management, walk-forward settings, and crisis periods.
    """
    # Capital and periods
    initial_capital: float = 1_000_000
    base_currency: str = "USD"
    
    # Date ranges
    train_start: str = "2022-01-01"
    train_end: str = "2023-06-30"
    test_start: str = "2023-07-01"
    test_end: str = "2025-01-31"
    
    # Execution settings
    slippage_model: str = "volume_impact"  # 'percentage', 'volume_impact', 'fixed'
    execution_delay_ms: int = 100  # Simulated execution delay
    partial_fills: bool = False  # Allow partial fills
    
    # Position sizing
    position_sizing_method: str = "fixed_pct"  # 'fixed_pct', 'volatility_target', 'kelly'
    base_position_pct: float = 0.05  # 5% base position
    volatility_target: float = 0.15  # 15% annualized vol target
    
    # Risk limits
    max_position_pct: float = 0.10  # Max 10% of capital per position
    max_portfolio_leverage: float = 1.0  # PDF: 1.0x leverage only
    max_drawdown_stop: float = 0.20  # Stop trading at 20% drawdown
    daily_var_limit: float = 0.05  # 5% daily VaR limit
    max_correlation: float = 0.70  # Max correlation between positions
    max_positions: int = 10  # PDF: "Total: 8-10 pairs max"
    
    # Stop loss / take profit
    use_stop_loss: bool = True
    stop_loss_pct: float = 0.05  # 5% stop loss
    use_take_profit: bool = False
    take_profit_pct: float = 0.10  # 10% take profit
    use_trailing_stop: bool = False
    trailing_stop_pct: float = 0.03  # 3% trailing stop
    
    # Margin/leverage
    use_margin: bool = True
    max_leverage: float = 1.0  # PDF: 1.0x leverage only
    margin_call_threshold: float = 0.30  # Margin call at 30% equity
    maintenance_margin: float = 0.05  # 5% maintenance margin
    
    # Walk-forward settings (PDF Section 2.4: 18m train / 6m test)
    walk_forward_enabled: bool = True
    walk_forward_periods: int = 4  # For 3 years of data with 18m/6m rolling windows
    train_window_days: int = 548  # 18 months training (PDF requirement)
    test_window_days: int = 183   # 6 months testing (PDF requirement)
    reoptimize_frequency: str = "quarterly"  # 'monthly', 'quarterly', 'none'
    optimization_metric: str = "sharpe_ratio"  # Metric to optimize
    
    # Benchmarks
    benchmarks: List[str] = field(default_factory=lambda: ["BTC", "ETH"])
    risk_free_rate: float = 0.04  # 4% annualized
    
    # Crisis periods for analysis
    crisis_periods: Dict[str, Tuple[str, str]] = field(default_factory=lambda: {
        'luna_collapse': ('2022-05-07', '2022-05-15'),
        'three_ac_liquidation': ('2022-06-13', '2022-06-18'),
        'ftx_bankruptcy': ('2022-11-06', '2022-11-15'),
        'usdc_depeg': ('2023-03-10', '2023-03-15'),
        'sec_lawsuits': ('2023-06-05', '2023-06-12'),
        'btc_etf_approval': ('2024-01-08', '2024-01-15'),
    })
    
    # Reporting
    generate_tearsheet: bool = True
    save_trades: bool = True
    save_equity_curve: bool = True
    verbose: bool = True


# =============================================================================
# TRANSACTION COST MODEL
# =============================================================================

class TransactionCostModel:
    """
    Comprehensive transaction cost model for multi-venue execution.
    
    Models all components of trading costs:
    - Exchange fees (maker/taker)
    - Slippage (volume-based market impact or percentage)
    - Gas costs (for DEX/hybrid venues)
    - MEV costs (sandwich attacks, front-running, JIT liquidity)
    
    Fee schedules based on empirical data from 2022-2024.
    """
    
    # Fee schedules by venue (in decimal, e.g., 0.0004 = 0.04% = 4 bps)
    FEE_SCHEDULES = {
        # CEX venues - tiered by volume, using mid-tier estimates
        'binance': {'maker': 0.0002, 'taker': 0.0004},
        'bybit': {'maker': 0.0001, 'taker': 0.0006},
        'okx': {'maker': 0.0002, 'taker': 0.0005},
        'coinbase': {'maker': 0.004, 'taker': 0.006},  # Higher retail fees
        'kraken': {'maker': 0.0016, 'taker': 0.0026},
        'deribit': {'maker': 0.0002, 'taker': 0.0005},
        'cme': {'maker': 0.0001, 'taker': 0.0002},  # Institutional
        
        # Hybrid venues - typically lower or zero maker fees
        'hyperliquid': {'maker': 0.0000, 'taker': 0.00025},
        'dydx_v4': {'maker': 0.0000, 'taker': 0.0005},
        'vertex': {'maker': 0.0000, 'taker': 0.0002},
        'gmx': {'maker': 0.0000, 'taker': 0.001},  # Position fee
        
        # DEX venues - pool-based fees
        'uniswap_v3_001': {'fee': 0.0001},  # 0.01% concentrated stable
        'uniswap_v3_005': {'fee': 0.0005},  # 0.05% stable pairs
        'uniswap_v3_030': {'fee': 0.003},   # 0.30% standard pairs
        'uniswap_v3_100': {'fee': 0.01},    # 1.00% exotic pairs
        'uniswap_v3': {'fee': 0.003},       # Default 0.30%
        'curve': {'fee': 0.0004},           # 0.04% typical stables
        'sushiswap': {'fee': 0.003},        # 0.30%
        'balancer': {'fee': 0.002},         # Variable, ~0.2% typical
        'pancakeswap': {'fee': 0.0025},     # 0.25%
    }
    
    # Gas costs by chain (USD per transaction, based on 2024 averages)
    GAS_COSTS = {
        'ethereum': 15.0,     # Highly variable, conservative estimate
        'arbitrum': 0.30,
        'optimism': 0.30,
        'polygon': 0.02,
        'base': 0.20,
        'avalanche': 0.15,
        'solana': 0.005,
        'cosmos': 0.02,       # For dYdX V4
        'bsc': 0.10,
        'fantom': 0.01,
        'gnosis': 0.001,
        'hyperliquid_l1': 0.0,  # Gas abstracted
    }
    
    # Base slippage by venue type (percentage of notional)
    BASE_SLIPPAGE = {
        VenueType.CEX: 0.0001,     # 1 bp base for liquid markets
        VenueType.HYBRID: 0.0002,  # 2 bps
        VenueType.DEX: 0.0015,     # 15 bps base
    }
    
    # MEV cost estimates by venue type (percentage of notional)
    MEV_COSTS = {
        VenueType.CEX: 0.0,        # No MEV on CEX
        VenueType.HYBRID: 0.0001,  # 1 bp (some sequencer risk)
        VenueType.DEX: 0.0008,     # 8 bps average (sandwich, frontrun)
    }
    
    # Market impact coefficients (for volume impact model)
    IMPACT_COEFFICIENTS = {
        VenueType.CEX: 0.05,       # 5% of sqrt(participation)
        VenueType.HYBRID: 0.10,    # 10%
        VenueType.DEX: 0.20,       # 20%
    }
    
    def __init__(self, custom_fees: Optional[Dict] = None):
        """
        Initialize transaction cost model.
        
        Args:
            custom_fees: Optional custom fee schedule to override defaults
        """
        self.fees = {**self.FEE_SCHEDULES}
        if custom_fees:
            self.fees.update(custom_fees)
    
    def calculate_fees(
        self,
        notional: float,
        venue: str,
        order_type: OrderType = OrderType.MARKET
    ) -> float:
        """
        Calculate exchange fees.
        
        Args:
            notional: Trade notional value in USD
            venue: Exchange/venue name
            order_type: Order type (affects maker/taker)
            
        Returns:
            Fee amount in USD
        """
        venue_key = venue.lower().replace(' ', '_').replace('-', '_')
        
        if venue_key not in self.fees:
            logger.warning(f"Unknown venue '{venue}', using conservative default fees")
            venue_fees = {'maker': 0.001, 'taker': 0.001, 'fee': 0.003}
        else:
            venue_fees = self.fees[venue_key]
        
        # DEX uses single fee structure (AMM pool fee)
        if 'fee' in venue_fees:
            return notional * venue_fees['fee']
        
        # CEX/hybrid uses maker/taker
        if order_type == OrderType.LIMIT:
            return notional * venue_fees['maker']
        else:
            return notional * venue_fees['taker']
    
    def calculate_slippage(
        self,
        notional: float,
        venue_type: VenueType,
        daily_volume: Optional[float] = None,
        order_book_depth: Optional[float] = None,
        volatility: Optional[float] = None,
        urgency: float = 1.0
    ) -> float:
        """
        Calculate expected slippage using volume impact model.
        
        Uses the square-root market impact model:
        slippage = base + coefficient * sqrt(participation_rate) * volatility_factor * urgency
        
        Args:
            notional: Trade notional value
            venue_type: Type of venue
            daily_volume: Daily trading volume (optional)
            order_book_depth: Order book depth at touch (optional)
            volatility: Current volatility (optional, annualized)
            urgency: Order urgency factor (1.0 = normal, >1 = urgent)
            
        Returns:
            Slippage amount in USD
        """
        base_slip = self.BASE_SLIPPAGE[venue_type]
        
        # Volume impact component
        volume_impact = 0.0
        if daily_volume and daily_volume > 0:
            participation_rate = notional / daily_volume
            coefficient = self.IMPACT_COEFFICIENTS[venue_type]
            volume_impact = coefficient * np.sqrt(participation_rate)
        
        # Order book depth adjustment
        depth_adjustment = 1.0
        if order_book_depth and order_book_depth > 0:
            # Higher depth = lower slippage
            depth_ratio = notional / order_book_depth
            if depth_ratio > 1:
                depth_adjustment = np.sqrt(depth_ratio)
        
        # Volatility adjustment (higher vol = more slippage)
        vol_factor = 1.0
        if volatility:
            # Normalize to ~20% annualized vol baseline
            vol_factor = max(0.5, min(3.0, volatility / 0.20))
        
        slippage_pct = (base_slip + volume_impact) * vol_factor * depth_adjustment * urgency
        
        # Cap slippage at reasonable levels
        max_slippage = {
            VenueType.CEX: 0.005,    # 50 bps max
            VenueType.HYBRID: 0.01,  # 100 bps max
            VenueType.DEX: 0.03,     # 300 bps max
        }
        slippage_pct = min(slippage_pct, max_slippage[venue_type])
        
        return notional * slippage_pct
    
    def calculate_gas_cost(
        self,
        venue: str,
        chain: str,
        num_transactions: int = 1,
        is_complex: bool = False
    ) -> float:
        """
        Calculate gas costs for on-chain transactions.
        
        Args:
            venue: Exchange/venue name
            chain: Blockchain network
            num_transactions: Number of transactions
            is_complex: Whether transaction is complex (swap vs transfer)
            
        Returns:
            Gas cost in USD
        """
        # CEX has no gas costs
        cex_venues = {'binance', 'bybit', 'okx', 'coinbase', 'kraken', 'deribit', 'cme'}
        if venue.lower() in cex_venues:
            return 0.0
        
        # Get base gas cost for chain
        chain_key = chain.lower().replace(' ', '_').replace('-', '_')
        base_gas = self.GAS_COSTS.get(chain_key, 1.0)
        
        # Complex transactions (multi-hop swaps) cost more
        complexity_multiplier = 2.5 if is_complex else 1.0
        
        return base_gas * num_transactions * complexity_multiplier
    
    def calculate_mev_cost(
        self,
        notional: float,
        venue_type: VenueType,
        is_large_order: bool = False,
        use_private_mempool: bool = False
    ) -> float:
        """
        Estimate MEV extraction costs.
        
        MEV (Maximal Extractable Value) includes:
        - Sandwich attacks
        - Front-running
        - Just-in-time (JIT) liquidity
        
        Args:
            notional: Trade notional value
            venue_type: Type of venue
            is_large_order: Whether order is large (attracts more MEV)
            use_private_mempool: Whether using private transaction submission
            
        Returns:
            Estimated MEV cost in USD
        """
        mev_rate = self.MEV_COSTS[venue_type]
        
        # Large orders attract more MEV
        if is_large_order and venue_type == VenueType.DEX:
            mev_rate *= 1.5
        
        # Private mempool reduces MEV risk
        if use_private_mempool:
            mev_rate *= 0.3
        
        return notional * mev_rate
    
    def total_cost(
        self,
        notional: float,
        venue: str,
        venue_type: VenueType,
        chain: str = "arbitrum",
        order_type: OrderType = OrderType.MARKET,
        daily_volume: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate total transaction costs.
        
        Args:
            notional: Trade notional value
            venue: Exchange/venue name
            venue_type: Type of venue
            chain: Blockchain network (for gas)
            order_type: Order type
            daily_volume: Daily volume for slippage calculation
            volatility: Current volatility
            
        Returns:
            Dictionary with detailed cost breakdown
        """
        fees = self.calculate_fees(notional, venue, order_type)
        slippage = self.calculate_slippage(
            notional, venue_type, daily_volume, volatility=volatility
        )
        gas = self.calculate_gas_cost(venue, chain)
        mev = self.calculate_mev_cost(
            notional, venue_type, is_large_order=(notional > 100000)
        )
        
        total = fees + slippage + gas + mev
        
        return {
            'fees': fees,
            'slippage': slippage,
            'gas': gas,
            'mev': mev,
            'total': total,
            'total_bps': (total / notional * 10000) if notional > 0 else 0
        }


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

class PerformanceMetrics:
    """
    Comprehensive performance metrics calculation.
    
    Includes:
    - Risk-adjusted returns (Sharpe, Sortino, Calmar, Omega)
    - Drawdown analysis
    - Trade statistics
    - Risk metrics (VaR, CVaR, tail ratio)
    - Benchmark comparison (alpha, beta, information ratio)
    - Regime-specific performance
    """
    
    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Args:
            returns: Daily returns series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            
        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / periods_per_year
        return (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year)
    
    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 252,
        target_return: float = 0.0
    ) -> float:
        """
        Calculate Sortino ratio using downside deviation.
        
        Args:
            returns: Daily returns series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            target_return: Target return for downside calculation
            
        Returns:
            Annualized Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return np.inf if excess_returns.mean() > 0 else 0.0
        
        # Downside deviation
        downside_squared = np.minimum(returns - target_return, 0) ** 2
        downside_std = np.sqrt(downside_squared.mean())
        
        if downside_std == 0:
            return np.inf if excess_returns.mean() > 0 else 0.0
        
        return (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)
    
    @staticmethod
    def max_drawdown(
        equity_curve: pd.Series
    ) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate maximum drawdown with peak and trough dates.
        
        Args:
            equity_curve: Equity curve series
            
        Returns:
            Tuple of (max_drawdown_pct, peak_date, trough_date)
        """
        if len(equity_curve) < 2:
            return 0.0, equity_curve.index[0], equity_curve.index[0]
        
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        
        max_dd = drawdown.min()
        trough_idx = drawdown.idxmin()
        peak_idx = equity_curve[:trough_idx].idxmax()
        
        return abs(max_dd), peak_idx, trough_idx
    
    @staticmethod
    def drawdown_series(equity_curve: pd.Series) -> pd.Series:
        """
        Calculate drawdown series.
        
        Args:
            equity_curve: Equity curve series
            
        Returns:
            Drawdown series (negative values represent drawdowns)
        """
        peak = equity_curve.expanding().max()
        return (equity_curve - peak) / peak
    
    @staticmethod
    def drawdown_duration(equity_curve: pd.Series) -> Tuple[int, int, int]:
        """
        Calculate drawdown duration statistics.
        
        Args:
            equity_curve: Equity curve series
            
        Returns:
            Tuple of (max_duration_days, avg_duration_days, current_duration_days)
        """
        drawdown = PerformanceMetrics.drawdown_series(equity_curve)
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        
        # Calculate durations
        durations = []
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        # Handle ongoing drawdown
        if current_duration > 0:
            durations.append(current_duration)
        
        if not durations:
            return 0, 0, 0
        
        return max(durations), int(np.mean(durations)), current_duration
    
    @staticmethod
    def calmar_ratio(
        returns: pd.Series,
        equity_curve: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).
        
        Args:
            returns: Daily returns series
            equity_curve: Equity curve series
            periods_per_year: Trading periods per year
            
        Returns:
            Calmar ratio
        """
        annual_return = returns.mean() * periods_per_year
        max_dd, _, _ = PerformanceMetrics.max_drawdown(equity_curve)
        
        if max_dd == 0:
            return np.inf if annual_return > 0 else 0.0
        
        return annual_return / max_dd
    
    @staticmethod
    def omega_ratio(
        returns: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """
        Calculate Omega ratio.
        
        Args:
            returns: Daily returns series
            threshold: Return threshold
            
        Returns:
            Omega ratio
        """
        returns_above = returns[returns > threshold] - threshold
        returns_below = threshold - returns[returns <= threshold]
        
        if returns_below.sum() == 0:
            return np.inf if returns_above.sum() > 0 else 0.0
        
        return returns_above.sum() / returns_below.sum()
    
    @staticmethod
    def value_at_risk(
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Daily returns series
            confidence: Confidence level (e.g., 0.95 for 95%)
            method: 'historical' or 'parametric'
            
        Returns:
            VaR (negative value indicating loss)
        """
        if len(returns) < 10:
            return 0.0
        
        if method == 'parametric':
            from scipy import stats
            z_score = stats.norm.ppf(1 - confidence)
            return returns.mean() + z_score * returns.std()
        else:
            return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def expected_shortfall(
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR / CVaR).
        
        Args:
            returns: Daily returns series
            confidence: Confidence level
            
        Returns:
            Expected Shortfall (average loss beyond VaR)
        """
        if len(returns) < 10:
            return 0.0
        
        var = PerformanceMetrics.value_at_risk(returns, confidence)
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) == 0:
            return var
        
        return tail_returns.mean()
    
    @staticmethod
    def tail_ratio(returns: pd.Series) -> float:
        """
        Calculate tail ratio (95th percentile / |5th percentile|).
        
        Higher values indicate positive skew (larger gains than losses).
        
        Args:
            returns: Daily returns series
            
        Returns:
            Tail ratio
        """
        if len(returns) < 20:
            return 0.0
        
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        
        if p5 == 0:
            return np.inf if p95 > 0 else 0.0
        
        return p95 / abs(p5)
    
    @staticmethod
    def win_rate(trades: List[TradeRecord]) -> float:
        """Calculate percentage of profitable trades."""
        if not trades:
            return 0.0
        winners = sum(1 for t in trades if t.net_pnl > 0)
        return winners / len(trades) * 100
    
    @staticmethod
    def profit_factor(trades: List[TradeRecord]) -> float:
        """Calculate gross profit / gross loss."""
        gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        gross_loss = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
        
        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    @staticmethod
    def average_trade(trades: List[TradeRecord]) -> Dict[str, float]:
        """Calculate average trade statistics."""
        if not trades:
            return {
                'avg_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_holding_hours': 0.0,
                'avg_return_pct': 0.0,
                'median_pnl': 0.0
            }
        
        winners = [t for t in trades if t.net_pnl > 0]
        losers = [t for t in trades if t.net_pnl < 0]
        pnls = [t.net_pnl for t in trades]
        
        return {
            'avg_pnl': np.mean(pnls),
            'avg_win': np.mean([t.net_pnl for t in winners]) if winners else 0.0,
            'avg_loss': np.mean([t.net_pnl for t in losers]) if losers else 0.0,
            'avg_holding_hours': np.mean([t.holding_period_hours for t in trades]),
            'avg_return_pct': np.mean([t.return_pct for t in trades]),
            'median_pnl': np.median(pnls)
        }
    
    @staticmethod
    def trade_statistics(trades: List[TradeRecord]) -> Dict[str, Any]:
        """Calculate comprehensive trade statistics."""
        if not trades:
            return {}
        
        winners = [t for t in trades if t.net_pnl > 0]
        losers = [t for t in trades if t.net_pnl < 0]
        pnls = [t.net_pnl for t in trades]
        
        # Consecutive wins/losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            if trade.net_pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        # Payoff ratio
        avg_win = np.mean([t.net_pnl for t in winners]) if winners else 0.0
        avg_loss = abs(np.mean([t.net_pnl for t in losers])) if losers else 1.0
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf
        
        # Expectancy
        win_rate = len(winners) / len(trades)
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate_pct': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': -abs(np.mean([t.net_pnl for t in losers])) if losers else 0.0,
            'largest_win': max(pnls),
            'largest_loss': min(pnls),
            'payoff_ratio': payoff_ratio,
            'profit_factor': PerformanceMetrics.profit_factor(trades),
            'expectancy': expectancy,
            'expectancy_pct': np.mean([t.return_pct for t in trades]),
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'avg_holding_hours': np.mean([t.holding_period_hours for t in trades]),
            'total_fees': sum(t.fees for t in trades),
            'total_slippage': sum(t.slippage for t in trades),
            'total_gas': sum(t.gas_cost for t in trades),
            'total_mev': sum(t.mev_cost for t in trades),
            'total_costs': sum(t.fees + t.slippage + t.gas_cost + t.mev_cost for t in trades),
        }
    
    @staticmethod
    def monthly_returns(equity_curve: pd.Series) -> pd.Series:
        """Calculate monthly returns."""
        return equity_curve.resample('M').last().pct_change().dropna()
    
    @staticmethod
    def rolling_sharpe(
        returns: pd.Series,
        window: int = 252,
        risk_free_rate: float = 0.04
    ) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        excess = returns - risk_free_rate / 252
        rolling_mean = excess.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        return (rolling_mean / rolling_std) * np.sqrt(252)
    
    @staticmethod
    def calculate_alpha_beta(
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.04
    ) -> Dict[str, float]:
        """
        Calculate alpha and beta vs benchmark.
        
        Args:
            strategy_returns: Strategy daily returns
            benchmark_returns: Benchmark daily returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary with alpha, beta, and related metrics
        """
        # Align indices
        aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 30:
            return {'alpha': 0.0, 'beta': 0.0, 'r_squared': 0.0}
        
        strat = aligned.iloc[:, 0]
        bench = aligned.iloc[:, 1]
        
        # Calculate beta
        covariance = strat.cov(bench)
        benchmark_variance = bench.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Calculate alpha (Jensen's alpha)
        rf_daily = risk_free_rate / 252
        alpha = (strat.mean() - rf_daily) - beta * (bench.mean() - rf_daily)
        alpha_annualized = alpha * 252
        
        # R-squared
        correlation = strat.corr(bench)
        r_squared = correlation ** 2
        
        # Tracking error
        active_returns = strat - bench
        tracking_error = active_returns.std() * np.sqrt(252)
        
        # Information ratio
        info_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        return {
            'alpha': alpha_annualized,
            'beta': beta,
            'r_squared': r_squared,
            'correlation': correlation,
            'tracking_error': tracking_error,
            'information_ratio': info_ratio
        }
    
    @staticmethod
    def calculate_all(
        returns: pd.Series,
        equity_curve: pd.Series,
        trades: List[TradeRecord],
        risk_free_rate: float = 0.04,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Calculate all performance metrics.
        
        Args:
            returns: Daily returns series
            equity_curve: Equity curve series
            trades: List of trade records
            risk_free_rate: Annual risk-free rate
            benchmark_returns: Optional benchmark returns for comparison
            
        Returns:
            Comprehensive dictionary with all metrics
        """
        if len(returns) == 0 or len(equity_curve) == 0:
            return {}
        
        max_dd, peak_date, trough_date = PerformanceMetrics.max_drawdown(equity_curve)
        dd_max, dd_avg, dd_current = PerformanceMetrics.drawdown_duration(equity_curve)
        avg_trade = PerformanceMetrics.average_trade(trades)
        trade_stats = PerformanceMetrics.trade_statistics(trades)
        
        # Time underwater
        drawdown_series = PerformanceMetrics.drawdown_series(equity_curve)
        underwater_periods = (drawdown_series < 0).sum()
        total_periods = len(drawdown_series)
        
        metrics = {
            # Return metrics
            'total_return_pct': (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100,
            'annualized_return_pct': returns.mean() * 252 * 100,
            'volatility_pct': returns.std() * np.sqrt(252) * 100,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            
            # Risk-adjusted returns
            'sharpe_ratio': PerformanceMetrics.sharpe_ratio(returns, risk_free_rate),
            'sortino_ratio': PerformanceMetrics.sortino_ratio(returns, risk_free_rate),
            'calmar_ratio': PerformanceMetrics.calmar_ratio(returns, equity_curve),
            'omega_ratio': PerformanceMetrics.omega_ratio(returns),
            'tail_ratio': PerformanceMetrics.tail_ratio(returns),
            
            # Drawdown metrics
            'max_drawdown_pct': max_dd * 100,
            'max_dd_peak_date': peak_date,
            'max_dd_trough_date': trough_date,
            'max_dd_duration_days': dd_max,
            'avg_dd_duration_days': dd_avg,
            'current_dd_duration_days': dd_current,
            'avg_drawdown_pct': abs(drawdown_series.mean()) * 100,
            'time_underwater_pct': (underwater_periods / total_periods) * 100 if total_periods > 0 else 0,
            
            # Risk metrics
            'var_95_pct': PerformanceMetrics.value_at_risk(returns, 0.95) * 100,
            'var_99_pct': PerformanceMetrics.value_at_risk(returns, 0.99) * 100,
            'cvar_95_pct': PerformanceMetrics.expected_shortfall(returns, 0.95) * 100,
            
            # Trade metrics
            'total_trades': len(trades),
            'win_rate_pct': PerformanceMetrics.win_rate(trades),
            'profit_factor': PerformanceMetrics.profit_factor(trades),
            'avg_trade_pnl': avg_trade['avg_pnl'],
            'avg_win': avg_trade['avg_win'],
            'avg_loss': avg_trade['avg_loss'],
            'median_trade_pnl': avg_trade['median_pnl'],
            'avg_holding_hours': avg_trade['avg_holding_hours'],
            
            # Detailed trade stats
            'trade_statistics': trade_stats,
        }
        
        # Benchmark comparison
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            benchmark_metrics = PerformanceMetrics.calculate_alpha_beta(
                returns, benchmark_returns, risk_free_rate
            )
            metrics.update({
                'alpha': benchmark_metrics['alpha'],
                'beta': benchmark_metrics['beta'],
                'r_squared': benchmark_metrics['r_squared'],
                'correlation_to_benchmark': benchmark_metrics['correlation'],
                'information_ratio': benchmark_metrics['information_ratio'],
                'tracking_error_pct': benchmark_metrics['tracking_error'] * 100,
            })
        
        return metrics


# =============================================================================
# STRATEGY BASE CLASS
# =============================================================================

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    Subclasses must implement:
    - generate_signals(): Generate trading signals from data
    - get_parameters(): Return strategy parameters
    - set_parameters(): Set strategy parameters
    
    Optional:
    - fit(): Optimize parameters on training data
    - calculate_position_size(): Custom position sizing
    """
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        self.name = name
        self.config = config or {}
        self.positions: Dict[str, Position] = {}
        self.signals: pd.DataFrame = pd.DataFrame()
        self.is_fitted = False
        self._parameters: Dict[str, Any] = {}
    
    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate trading signals from input data.
        
        Args:
            data: Input data (OHLCV, funding rates, etc.)
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with 'signal' column containing SignalType values
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return current strategy parameters."""
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set strategy parameters."""
        pass
    
    def fit(
        self,
        train_data: pd.DataFrame,
        metric: str = 'sharpe_ratio',
        **kwargs
    ) -> 'BaseStrategy':
        """
        Fit/optimize strategy on training data.
        
        Args:
            train_data: Training data
            metric: Optimization metric
            **kwargs: Additional arguments
            
        Returns:
            Self for chaining
        """
        self.is_fitted = True
        return self
    
    def calculate_position_size(
        self,
        signal: SignalType,
        current_price: float,
        capital: float,
        max_position_pct: float = 0.10,
        volatility: Optional[float] = None,
        signal_strength: float = 1.0
    ) -> float:
        """
        Calculate position size based on signal and risk parameters.
        
        Args:
            signal: Trading signal
            current_price: Current asset price
            capital: Available capital
            max_position_pct: Maximum position as % of capital
            volatility: Optional volatility for vol-targeting
            signal_strength: Signal strength multiplier (0 to 1)
            
        Returns:
            Position size in base currency units
        """
        if signal in [SignalType.HOLD, SignalType.LONG_EXIT, SignalType.SHORT_EXIT]:
            return 0.0
        
        max_notional = capital * max_position_pct * signal_strength
        
        # Volatility targeting (optional)
        if volatility and volatility > 0:
            target_vol = 0.15  # 15% target volatility
            vol_scalar = target_vol / (volatility * np.sqrt(252))
            vol_scalar = np.clip(vol_scalar, 0.5, 2.0)
            max_notional *= vol_scalar
        
        position_size = max_notional / current_price
        
        return position_size


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """
    Comprehensive backtesting engine with walk-forward optimization.
    
    Features:
    - Walk-forward optimization with configurable windows
    - Multi-venue execution simulation (CEX, hybrid, DEX)
    - Realistic transaction costs (fees, slippage, gas, MEV)
    - Risk management (position limits, drawdown stops, VaR)
    - Crisis period analysis
    - Performance attribution by venue, strategy, regime
    - Benchmark comparison
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtest engine.
        
        Args:
            config: Backtest configuration (uses defaults if None)
        """
        self.config = config or BacktestConfig()
        self.cost_model = TransactionCostModel()
        
        # State
        self.capital = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[TradeRecord] = []
        self.fills: List[Fill] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.daily_returns: List[Tuple[pd.Timestamp, float]] = []
        
        # Tracking
        self.trade_counter = 0
        self.current_drawdown = 0.0
        self.peak_equity = self.capital
        self.is_stopped = False
        self.stop_reason = ""
        
        # Walk-forward results
        self.walk_forward_results: List[Dict] = []
        
        # Performance tracking
        self.daily_equity: Dict[pd.Timestamp, float] = {}
        self.position_history: List[Dict] = []
        
        logger.info(
            f"Initialized BacktestEngine with ${self.capital:,.0f} capital, "
            f"max position {self.config.max_position_pct:.0%}"
        )
    
    def reset(self) -> None:
        """Reset engine state for new backtest."""
        self.capital = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.fills = []
        self.equity_curve = []
        self.daily_returns = []
        self.trade_counter = 0
        self.current_drawdown = 0.0
        self.peak_equity = self.capital
        self.is_stopped = False
        self.stop_reason = ""
        self.walk_forward_results = []
        self.daily_equity = {}
        self.position_history = []
    
    def _get_total_equity(self, prices: Dict[str, float]) -> float:
        """
        Calculate total equity including unrealized P&L.
        
        Args:
            prices: Current prices by symbol
            
        Returns:
            Total equity value
        """
        equity = self.capital
        
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_unrealized_pnl(prices[symbol])
                equity += position.unrealized_pnl
        
        return equity
    
    def _check_risk_limits(self, equity: float) -> Tuple[bool, str]:
        """
        Check if risk limits are breached.
        
        Args:
            equity: Current equity
            
        Returns:
            Tuple of (should_continue, stop_reason)
        """
        # Update peak and drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
        
        # Check max drawdown stop
        if self.current_drawdown >= self.config.max_drawdown_stop:
            return False, f"Max drawdown stop: {self.current_drawdown:.1%}"
        
        # Check margin call
        if self.config.use_margin:
            total_margin = sum(pos.margin_used for pos in self.positions.values())
            margin_ratio = total_margin / equity if equity > 0 else 0
            if margin_ratio > (1 - self.config.margin_call_threshold):
                return False, f"Margin call: {margin_ratio:.1%}"
        
        return True, ""
    
    def _calculate_gross_exposure(self) -> float:
        """Calculate total gross exposure."""
        return sum(pos.notional for pos in self.positions.values())
    
    def _can_open_position(
        self,
        notional: float,
        equity: float
    ) -> Tuple[bool, str]:
        """
        Check if new position can be opened within limits.
        
        Args:
            notional: Proposed position notional
            equity: Current equity
            
        Returns:
            Tuple of (can_open, reason_if_not)
        """
        # Check max positions
        if len(self.positions) >= self.config.max_positions:
            return False, f"Max positions ({self.config.max_positions}) reached"
        
        # Check position size limit
        if notional > equity * self.config.max_position_pct:
            return False, f"Position too large: {notional/equity:.1%} > {self.config.max_position_pct:.1%}"
        
        # Check leverage limit
        current_exposure = self._calculate_gross_exposure()
        if (current_exposure + notional) > equity * self.config.max_portfolio_leverage:
            return False, f"Leverage limit exceeded"
        
        return True, ""
    
    def execute_order(
        self,
        order: Order,
        market_data: Dict[str, Any],
        timestamp: pd.Timestamp
    ) -> Optional[Fill]:
        """
        Execute an order with realistic costs.
        
        Args:
            order: Order to execute
            market_data: Current market data (price, volume, volatility, etc.)
            timestamp: Execution timestamp
            
        Returns:
            Fill object if executed, None otherwise
        """
        # Get market price
        price = market_data.get('price', order.price)
        daily_volume = market_data.get('volume')
        volatility = market_data.get('volatility')
        chain = market_data.get('chain', 'arbitrum')
        
        notional = price * order.quantity
        
        # Calculate costs
        costs = self.cost_model.total_cost(
            notional=notional,
            venue=order.venue,
            venue_type=order.venue_type,
            chain=chain,
            order_type=order.order_type,
            daily_volume=daily_volume,
            volatility=volatility
        )
        
        # Apply slippage to fill price
        slippage_pct = costs['slippage'] / notional if notional > 0 else 0
        if order.side == OrderSide.BUY:
            fill_price = price * (1 + slippage_pct)
        else:
            fill_price = price * (1 - slippage_pct)
        
        fill = Fill(
            order=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            slippage=costs['slippage'],
            fees=costs['fees'],
            gas_cost=costs['gas'],
            mev_cost=costs['mev'],
            total_cost=costs['total'],
            fill_timestamp=timestamp
        )
        
        self.fills.append(fill)
        
        return fill
    
    def open_position(
        self,
        fill: Fill,
        strategy: str = "",
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None
    ) -> Position:
        """
        Open a new position from a fill.
        
        Args:
            fill: Executed fill
            strategy: Strategy name
            stop_loss_pct: Stop loss percentage (optional)
            take_profit_pct: Take profit percentage (optional)
            
        Returns:
            New Position object
        """
        symbol = fill.order.symbol
        
        # Deduct costs from capital
        self.capital -= fill.total_cost
        
        # Determine position quantity (negative for short)
        quantity = fill.fill_quantity
        if fill.order.side == OrderSide.SELL:
            quantity = -quantity
        
        # Calculate stop loss and take profit levels
        stop_loss = None
        take_profit = None
        
        if stop_loss_pct or (self.config.use_stop_loss and self.config.stop_loss_pct):
            sl_pct = stop_loss_pct or self.config.stop_loss_pct
            if quantity > 0:  # Long
                stop_loss = fill.fill_price * (1 - sl_pct)
            else:  # Short
                stop_loss = fill.fill_price * (1 + sl_pct)
        
        if take_profit_pct or (self.config.use_take_profit and self.config.take_profit_pct):
            tp_pct = take_profit_pct or self.config.take_profit_pct
            if quantity > 0:  # Long
                take_profit = fill.fill_price * (1 + tp_pct)
            else:  # Short
                take_profit = fill.fill_price * (1 - tp_pct)
        
        # Calculate margin
        margin_used = 0.0
        if self.config.use_margin:
            margin_used = fill.notional / self.config.max_leverage
        
        position = Position(
            symbol=symbol,
            quantity=quantity,
            avg_price=fill.fill_price,
            venue=fill.order.venue,
            venue_type=fill.order.venue_type,
            entry_time=fill.fill_timestamp,
            strategy=strategy or fill.order.strategy,
            margin_used=margin_used,
            stop_loss=stop_loss,
            take_profit=take_profit,
            highest_price=fill.fill_price,
            lowest_price=fill.fill_price
        )
        
        self.positions[symbol] = position
        
        if self.config.verbose:
            logger.debug(
                f"Opened {'long' if quantity > 0 else 'short'} position: "
                f"{symbol} @ ${fill.fill_price:.2f}, qty={abs(quantity):.4f}, "
                f"SL={stop_loss}, TP={take_profit}"
            )
        
        return position
    
    def close_position(
        self,
        symbol: str,
        fill: Fill,
        exit_reason: str = "signal"
    ) -> TradeRecord:
        """
        Close an existing position.
        
        Args:
            symbol: Symbol to close
            fill: Exit fill
            exit_reason: Reason for closing (signal, stop_loss, etc.)
            
        Returns:
            TradeRecord for the round-trip trade
        """
        if symbol not in self.positions:
            raise ValueError(f"No position found for {symbol}")
        
        position = self.positions[symbol]
        
        # Calculate P&L
        if position.is_long:
            gross_pnl = (fill.fill_price - position.avg_price) * position.quantity
        else:
            gross_pnl = (position.avg_price - fill.fill_price) * abs(position.quantity)
        
        net_pnl = gross_pnl - fill.total_cost
        
        # Update capital
        self.capital += position.notional + net_pnl
        
        # Calculate excursions
        if position.is_long:
            max_favorable = (position.highest_price - position.avg_price) * position.quantity
            max_adverse = (position.avg_price - position.lowest_price) * position.quantity
        else:
            max_favorable = (position.avg_price - position.lowest_price) * abs(position.quantity)
            max_adverse = (position.highest_price - position.avg_price) * abs(position.quantity)
        
        # Create trade record
        self.trade_counter += 1
        holding_hours = (fill.fill_timestamp - position.entry_time).total_seconds() / 3600
        
        trade = TradeRecord(
            trade_id=f"T{self.trade_counter:06d}",
            symbol=symbol,
            strategy=position.strategy,
            venue=position.venue,
            venue_type=position.venue_type,
            entry_time=position.entry_time,
            exit_time=fill.fill_timestamp,
            entry_price=position.avg_price,
            exit_price=fill.fill_price,
            quantity=abs(position.quantity),
            side='long' if position.is_long else 'short',
            gross_pnl=gross_pnl,
            fees=fill.fees,
            slippage=fill.slippage,
            gas_cost=fill.gas_cost,
            mev_cost=fill.mev_cost,
            net_pnl=net_pnl,
            holding_period_hours=holding_hours,
            exit_reason=exit_reason,
            max_favorable_excursion=max_favorable,
            max_adverse_excursion=max_adverse
        )
        
        self.trades.append(trade)
        del self.positions[symbol]
        
        if self.config.verbose:
            logger.debug(
                f"Closed {trade.side} position: {symbol}, "
                f"P&L=${net_pnl:+,.2f} ({trade.return_pct:+.2f}%), "
                f"held {holding_hours:.1f}h, reason={exit_reason}"
            )
        
        return trade
    
    def _check_stop_conditions(
        self,
        symbol: str,
        position: Position,
        current_price: float,
        timestamp: pd.Timestamp
    ) -> Optional[Tuple[str, float]]:
        """
        Check if any stop conditions are met.
        
        Args:
            symbol: Trading symbol
            position: Current position
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            Tuple of (exit_reason, exit_price) if stop triggered, else None
        """
        # Check stop loss
        if position.check_stop_loss(current_price):
            return ExitReason.STOP_LOSS.value, current_price
        
        # Check take profit
        if position.check_take_profit(current_price):
            return ExitReason.TAKE_PROFIT.value, current_price
        
        # Check trailing stop
        if self.config.use_trailing_stop:
            trailing_pct = self.config.trailing_stop_pct
            if position.is_long:
                trailing_stop = position.highest_price * (1 - trailing_pct)
                if current_price <= trailing_stop:
                    return ExitReason.TRAILING_STOP.value, current_price
            else:
                trailing_stop = position.lowest_price * (1 + trailing_pct)
                if current_price >= trailing_stop:
                    return ExitReason.TRAILING_STOP.value, current_price
        
        return None
    
    def run_backtest(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        price_col: str = 'close',
        signal_col: str = 'signal',
        symbol_col: Optional[str] = 'symbol',
        volume_col: Optional[str] = 'volume',
        venue: str = 'binance',
        venue_type: VenueType = VenueType.CEX,
        chain: str = 'arbitrum',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Run a backtest for a strategy.
        
        Args:
            strategy: Strategy instance
            data: OHLCV data with signals
            price_col: Price column name
            signal_col: Signal column name
            symbol_col: Symbol column name (if multi-asset)
            volume_col: Volume column name
            venue: Trading venue
            venue_type: Venue type
            chain: Blockchain (for DEX gas costs)
            start_date: Start date filter
            end_date: End date filter
            benchmark_data: Optional benchmark data for comparison
            
        Returns:
            Backtest results dictionary
        """
        self.reset()
        
        # Filter data by date range
        if start_date:
            data = data[data.index >= pd.Timestamp(start_date)]
        if end_date:
            data = data[data.index <= pd.Timestamp(end_date)]
        
        if len(data) < 10:
            logger.warning("Insufficient data for backtest")
            return {}
        
        # Generate signals if not present
        if signal_col not in data.columns:
            data = strategy.generate_signals(data.copy())
        
        if signal_col not in data.columns:
            logger.error(f"Signal column '{signal_col}' not found after generate_signals()")
            return {}
        
        # Track previous equity for daily returns
        prev_equity = self.capital
        prev_date = None
        
        # Main backtest loop
        for timestamp, row in data.iterrows():
            # Check if stopped
            if self.is_stopped:
                break
            
            price = row[price_col]
            signal = row.get(signal_col, SignalType.HOLD)
            volume = row.get(volume_col) if volume_col else None
            symbol = row.get(symbol_col, 'ASSET') if symbol_col else 'ASSET'
            
            # Convert signal string to enum if needed
            if isinstance(signal, str):
                try:
                    signal = SignalType[signal.upper()]
                except KeyError:
                    signal = SignalType.HOLD
            
            # Market data for execution
            market_data = {
                'price': price,
                'volume': volume,
                'chain': chain if venue_type != VenueType.CEX else None,
            }
            
            # Calculate current equity
            equity = self._get_total_equity({symbol: price})
            
            # Check risk limits
            should_continue, stop_reason = self._check_risk_limits(equity)
            if not should_continue:
                self.is_stopped = True
                self.stop_reason = stop_reason
                logger.warning(f"Backtest stopped: {stop_reason}")
                
                # Close all positions
                for sym in list(self.positions.keys()):
                    pos = self.positions[sym]
                    exit_order = Order(
                        timestamp=timestamp,
                        symbol=sym,
                        side=OrderSide.SELL if pos.is_long else OrderSide.BUY,
                        quantity=abs(pos.quantity),
                        price=price,
                        venue=venue,
                        venue_type=venue_type,
                        strategy=strategy.name
                    )
                    fill = self.execute_order(exit_order, market_data, timestamp)
                    if fill:
                        self.close_position(sym, fill, exit_reason=ExitReason.DRAWDOWN_STOP.value)
                break
            
            # Check stop conditions for existing positions
            if symbol in self.positions:
                position = self.positions[symbol]
                position.update_unrealized_pnl(price)
                
                stop_result = self._check_stop_conditions(symbol, position, price, timestamp)
                if stop_result:
                    exit_reason, exit_price = stop_result
                    exit_order = Order(
                        timestamp=timestamp,
                        symbol=symbol,
                        side=OrderSide.SELL if position.is_long else OrderSide.BUY,
                        quantity=abs(position.quantity),
                        price=exit_price,
                        venue=venue,
                        venue_type=venue_type,
                        strategy=strategy.name
                    )
                    fill = self.execute_order(exit_order, market_data, timestamp)
                    if fill:
                        self.close_position(symbol, fill, exit_reason=exit_reason)
            
            # Process strategy signals
            has_position = symbol in self.positions
            
            if signal == SignalType.LONG_ENTRY and not has_position:
                size = strategy.calculate_position_size(
                    signal, price, equity, self.config.max_position_pct
                )
                
                can_open, reason = self._can_open_position(size * price, equity)
                if size > 0 and can_open:
                    order = Order(
                        timestamp=timestamp,
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=size,
                        price=price,
                        venue=venue,
                        venue_type=venue_type,
                        strategy=strategy.name
                    )
                    fill = self.execute_order(order, market_data, timestamp)
                    if fill:
                        self.open_position(fill, strategy.name)
            
            elif signal == SignalType.SHORT_ENTRY and not has_position:
                size = strategy.calculate_position_size(
                    signal, price, equity, self.config.max_position_pct
                )
                
                can_open, reason = self._can_open_position(size * price, equity)
                if size > 0 and can_open:
                    order = Order(
                        timestamp=timestamp,
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=size,
                        price=price,
                        venue=venue,
                        venue_type=venue_type,
                        strategy=strategy.name
                    )
                    fill = self.execute_order(order, market_data, timestamp)
                    if fill:
                        self.open_position(fill, strategy.name)
            
            elif signal in [SignalType.LONG_EXIT, SignalType.SHORT_EXIT] and has_position:
                position = self.positions[symbol]
                order = Order(
                    timestamp=timestamp,
                    symbol=symbol,
                    side=OrderSide.SELL if position.is_long else OrderSide.BUY,
                    quantity=abs(position.quantity),
                    price=price,
                    venue=venue,
                    venue_type=venue_type,
                    strategy=strategy.name
                )
                fill = self.execute_order(order, market_data, timestamp)
                if fill:
                    self.close_position(symbol, fill, exit_reason=ExitReason.SIGNAL.value)
            
            # Record equity
            equity = self._get_total_equity({symbol: price})
            self.equity_curve.append((timestamp, equity))
            
            # Calculate daily return (only once per day)
            current_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
            if prev_date is not None and current_date != prev_date and prev_equity > 0:
                daily_return = (equity - prev_equity) / prev_equity
                self.daily_returns.append((timestamp, daily_return))
            
            prev_equity = equity
            prev_date = current_date
        
        # Close any remaining positions at end
        for sym in list(self.positions.keys()):
            position = self.positions[sym]
            last_price = data[price_col].iloc[-1]
            market_data = {'price': last_price, 'chain': chain}
            
            exit_order = Order(
                timestamp=data.index[-1],
                symbol=sym,
                side=OrderSide.SELL if position.is_long else OrderSide.BUY,
                quantity=abs(position.quantity),
                price=last_price,
                venue=venue,
                venue_type=venue_type,
                strategy=strategy.name
            )
            fill = self.execute_order(exit_order, market_data, data.index[-1])
            if fill:
                self.close_position(sym, fill, exit_reason=ExitReason.END_OF_PERIOD.value)
        
        # Get benchmark returns if provided
        benchmark_returns = None
        if benchmark_data is not None and price_col in benchmark_data.columns:
            benchmark_returns = benchmark_data[price_col].pct_change().dropna()
        
        # Generate results
        return self._generate_results(strategy.name, benchmark_returns)
    
    def run_walk_forward(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        optimize_func: Callable[[BaseStrategy, pd.DataFrame], Dict],
        price_col: str = 'close',
        venue: str = 'binance',
        venue_type: VenueType = VenueType.CEX,
        chain: str = 'arbitrum'
    ) -> Dict[str, Any]:
        """
        Run walk-forward optimization.
        
        Divides data into train/test periods, optimizes on training,
        then evaluates on out-of-sample test data.
        
        Args:
            strategy: Strategy instance
            data: Full dataset
            optimize_func: Function(strategy, train_data) -> optimal_params
            price_col: Price column name
            venue: Trading venue
            venue_type: Venue type
            chain: Blockchain for gas costs
            
        Returns:
            Walk-forward results with period breakdown
        """
        self.reset()
        
        train_days = self.config.train_window_days
        test_days = self.config.test_window_days
        n_periods = self.config.walk_forward_periods
        
        # Calculate period boundaries
        total_days = len(data)
        period_length = train_days + test_days
        
        if total_days < period_length:
            logger.warning("Insufficient data for walk-forward optimization")
            return {}
        
        all_trades: List[TradeRecord] = []
        all_equity: List[Tuple[pd.Timestamp, float]] = []
        period_results: List[Dict] = []
        
        cumulative_capital = self.config.initial_capital
        
        for period in range(n_periods):
            # Calculate indices
            start_idx = period * test_days
            train_end_idx = start_idx + train_days
            test_end_idx = train_end_idx + test_days
            
            if test_end_idx > total_days:
                logger.info(f"Stopping at period {period + 1}: insufficient data")
                break
            
            train_data = data.iloc[start_idx:train_end_idx].copy()
            test_data = data.iloc[train_end_idx:test_end_idx].copy()
            
            logger.info(
                f"Walk-forward period {period + 1}/{n_periods}: "
                f"Train {train_data.index[0].date()} to {train_data.index[-1].date()}, "
                f"Test {test_data.index[0].date()} to {test_data.index[-1].date()}"
            )
            
            # Optimize on training data
            try:
                optimal_params = optimize_func(strategy, train_data)
                strategy.set_parameters(optimal_params)
            except Exception as e:
                logger.error(f"Optimization failed for period {period + 1}: {e}")
                continue
            
            # Generate signals for test data
            test_signals = strategy.generate_signals(test_data)
            
            # Run backtest on test data
            self.config.initial_capital = cumulative_capital
            period_result = self.run_backtest(
                strategy=strategy,
                data=test_signals,
                price_col=price_col,
                venue=venue,
                venue_type=venue_type,
                chain=chain
            )
            
            if period_result:
                cumulative_capital = period_result.get('final_capital', cumulative_capital)
                
                period_result['period'] = period + 1
                period_result['train_start'] = train_data.index[0]
                period_result['train_end'] = train_data.index[-1]
                period_result['test_start'] = test_data.index[0]
                period_result['test_end'] = test_data.index[-1]
                period_result['optimal_params'] = optimal_params
                
                period_results.append(period_result)
                all_trades.extend(self.trades)
                all_equity.extend(self.equity_curve)
        
        self.walk_forward_results = period_results
        self.trades = all_trades
        self.equity_curve = all_equity
        
        return self._generate_walk_forward_summary()
    
    def _generate_results(
        self,
        strategy_name: str,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive backtest results."""
        if not self.equity_curve:
            return {}
        
        # Convert to series
        equity_series = pd.Series(
            [e[1] for e in self.equity_curve],
            index=[e[0] for e in self.equity_curve]
        )
        
        returns_series = pd.Series(
            [r[1] for r in self.daily_returns],
            index=[r[0] for r in self.daily_returns]
        ) if self.daily_returns else equity_series.pct_change().dropna()
        
        # Calculate all metrics
        metrics = PerformanceMetrics.calculate_all(
            returns=returns_series,
            equity_curve=equity_series,
            trades=self.trades,
            risk_free_rate=self.config.risk_free_rate,
            benchmark_returns=benchmark_returns
        )
        
        # Add metadata
        metrics['strategy'] = strategy_name
        metrics['initial_capital'] = self.config.initial_capital
        metrics['final_capital'] = equity_series.iloc[-1] if len(equity_series) > 0 else self.capital
        metrics['start_date'] = equity_series.index[0] if len(equity_series) > 0 else None
        metrics['end_date'] = equity_series.index[-1] if len(equity_series) > 0 else None
        metrics['trading_days'] = len(equity_series)
        metrics['stopped_early'] = self.is_stopped
        metrics['stop_reason'] = self.stop_reason
        
        # Cost breakdown
        total_fees = sum(t.fees for t in self.trades)
        total_slippage = sum(t.slippage for t in self.trades)
        total_gas = sum(t.gas_cost for t in self.trades)
        total_mev = sum(t.mev_cost for t in self.trades)
        
        metrics['total_costs'] = {
            'fees': total_fees,
            'slippage': total_slippage,
            'gas': total_gas,
            'mev': total_mev,
            'total': total_fees + total_slippage + total_gas + total_mev,
            'as_pct_of_pnl': 0.0
        }
        
        gross_pnl = sum(t.gross_pnl for t in self.trades)
        if gross_pnl != 0:
            metrics['total_costs']['as_pct_of_pnl'] = metrics['total_costs']['total'] / abs(gross_pnl) * 100
        
        return metrics
    
    def _generate_walk_forward_summary(self) -> Dict[str, Any]:
        """Generate walk-forward optimization summary."""
        if not self.walk_forward_results:
            return {}
        
        # Aggregate across periods
        sharpe_ratios = [r.get('sharpe_ratio', 0) for r in self.walk_forward_results]
        returns = [r.get('total_return_pct', 0) for r in self.walk_forward_results]
        win_rates = [r.get('win_rate_pct', 0) for r in self.walk_forward_results]
        max_dds = [r.get('max_drawdown_pct', 0) for r in self.walk_forward_results]
        
        # Calculate consistency metrics
        positive_periods = sum(1 for r in returns if r > 0)
        
        return {
            'n_periods': len(self.walk_forward_results),
            'positive_periods': positive_periods,
            'consistency_pct': positive_periods / len(self.walk_forward_results) * 100,
            
            # Sharpe statistics
            'avg_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'min_sharpe': np.min(sharpe_ratios),
            'max_sharpe': np.max(sharpe_ratios),
            
            # Return statistics
            'avg_return_pct': np.mean(returns),
            'total_return_pct': np.sum(returns),
            'std_return_pct': np.std(returns),
            
            # Other metrics
            'avg_win_rate': np.mean(win_rates),
            'avg_max_dd_pct': np.mean(max_dds),
            'worst_dd_pct': np.max(max_dds),
            
            # Detailed period results
            'period_results': self.walk_forward_results,
            'total_trades': len(self.trades),
            'final_equity': self.equity_curve[-1][1] if self.equity_curve else self.capital
        }
    
    def analyze_crisis_periods(
        self,
        data: pd.DataFrame,
        price_col: str = 'close'
    ) -> Dict[str, Dict]:
        """
        Analyze strategy performance during crisis periods.
        
        Args:
            data: Price data
            price_col: Price column name
            
        Returns:
            Performance metrics by crisis period
        """
        crisis_analysis = {}
        
        for crisis_name, (start, end) in self.config.crisis_periods.items():
            # Filter trades during crisis
            crisis_trades = [
                t for t in self.trades
                if (start <= str(t.entry_time.date()) <= end or 
                    start <= str(t.exit_time.date()) <= end)
            ]
            
            # Filter equity during crisis
            crisis_equity = [
                (ts, eq) for ts, eq in self.equity_curve
                if start <= str(ts.date()) <= end
            ]
            
            if not crisis_equity:
                continue
            
            equity_series = pd.Series(
                [e[1] for e in crisis_equity],
                index=[e[0] for e in crisis_equity]
            )
            
            crisis_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100
            max_dd, _, _ = PerformanceMetrics.max_drawdown(equity_series)
            
            crisis_analysis[crisis_name] = {
                'period': f"{start} to {end}",
                'return_pct': crisis_return,
                'max_drawdown_pct': max_dd * 100,
                'n_trades': len(crisis_trades),
                'avg_trade_pnl': np.mean([t.net_pnl for t in crisis_trades]) if crisis_trades else 0,
                'win_rate_pct': PerformanceMetrics.win_rate(crisis_trades),
                'total_pnl': sum(t.net_pnl for t in crisis_trades),
            }
        
        return crisis_analysis
    
    def generate_report(self, include_trades: bool = False) -> str:
        """
        Generate a text report of backtest results.
        
        Args:
            include_trades: Whether to include trade list
            
        Returns:
            Formatted report string
        """
        results = self._generate_results("Strategy")
        
        if not results:
            return "No backtest results available."
        
        lines = []
        lines.append("=" * 70)
        lines.append("BACKTEST RESULTS REPORT")
        lines.append("=" * 70)
        lines.append("")
        
        # Summary
        lines.append("PERFORMANCE SUMMARY")
        lines.append("-" * 50)
        lines.append(f"Strategy:             {results.get('strategy', 'N/A')}")
        lines.append(f"Period:               {results.get('start_date')} to {results.get('end_date')}")
        lines.append(f"Trading Days:         {results.get('trading_days', 0):,}")
        lines.append("")
        lines.append(f"Initial Capital:      ${results.get('initial_capital', 0):,.2f}")
        lines.append(f"Final Capital:        ${results.get('final_capital', 0):,.2f}")
        lines.append(f"Total Return:         {results.get('total_return_pct', 0):+.2f}%")
        lines.append(f"Annualized Return:    {results.get('annualized_return_pct', 0):+.2f}%")
        lines.append(f"Volatility:           {results.get('volatility_pct', 0):.2f}%")
        lines.append("")
        
        # Risk-adjusted metrics
        lines.append("RISK-ADJUSTED METRICS")
        lines.append("-" * 50)
        lines.append(f"Sharpe Ratio:         {results.get('sharpe_ratio', 0):.3f}")
        lines.append(f"Sortino Ratio:        {results.get('sortino_ratio', 0):.3f}")
        lines.append(f"Calmar Ratio:         {results.get('calmar_ratio', 0):.3f}")
        lines.append(f"Omega Ratio:          {results.get('omega_ratio', 0):.3f}")
        lines.append("")
        
        # Drawdown
        lines.append("DRAWDOWN ANALYSIS")
        lines.append("-" * 50)
        lines.append(f"Max Drawdown:         {results.get('max_drawdown_pct', 0):.2f}%")
        lines.append(f"Max DD Duration:      {results.get('max_dd_duration_days', 0)} days")
        lines.append(f"Time Underwater:      {results.get('time_underwater_pct', 0):.1f}%")
        lines.append("")
        
        # Risk metrics
        lines.append("RISK METRICS")
        lines.append("-" * 50)
        lines.append(f"VaR (95%):            {results.get('var_95_pct', 0):.2f}%")
        lines.append(f"CVaR (95%):           {results.get('cvar_95_pct', 0):.2f}%")
        lines.append(f"Skewness:             {results.get('skewness', 0):.3f}")
        lines.append(f"Kurtosis:             {results.get('kurtosis', 0):.3f}")
        lines.append("")
        
        # Trade statistics
        lines.append("TRADE STATISTICS")
        lines.append("-" * 50)
        lines.append(f"Total Trades:         {results.get('total_trades', 0)}")
        lines.append(f"Win Rate:             {results.get('win_rate_pct', 0):.1f}%")
        lines.append(f"Profit Factor:        {results.get('profit_factor', 0):.2f}")
        lines.append(f"Avg Trade P&L:        ${results.get('avg_trade_pnl', 0):,.2f}")
        lines.append(f"Avg Win:              ${results.get('avg_win', 0):,.2f}")
        lines.append(f"Avg Loss:             ${results.get('avg_loss', 0):,.2f}")
        lines.append(f"Avg Holding (hrs):    {results.get('avg_holding_hours', 0):.1f}")
        lines.append("")
        
        # Transaction costs
        costs = results.get('total_costs', {})
        lines.append("TRANSACTION COSTS")
        lines.append("-" * 50)
        lines.append(f"Total Fees:           ${costs.get('fees', 0):,.2f}")
        lines.append(f"Total Slippage:       ${costs.get('slippage', 0):,.2f}")
        lines.append(f"Total Gas:            ${costs.get('gas', 0):,.2f}")
        lines.append(f"Total MEV:            ${costs.get('mev', 0):,.2f}")
        lines.append(f"Total Costs:          ${costs.get('total', 0):,.2f}")
        lines.append(f"Costs as % of P&L:    {costs.get('as_pct_of_pnl', 0):.1f}%")
        lines.append("")
        
        # Benchmark comparison
        if 'alpha' in results:
            lines.append("BENCHMARK COMPARISON")
            lines.append("-" * 50)
            lines.append(f"Alpha (ann.):         {results.get('alpha', 0)*100:.2f}%")
            lines.append(f"Beta:                 {results.get('beta', 0):.3f}")
            lines.append(f"Correlation:          {results.get('correlation_to_benchmark', 0):.3f}")
            lines.append(f"Information Ratio:    {results.get('information_ratio', 0):.3f}")
            lines.append("")
        
        lines.append("=" * 70)

        return "\n".join(lines)

    def run(
        self,
        signals: pd.DataFrame,
        price_data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        venue: str = 'binance',
        venue_type: VenueType = VenueType.CEX,
        chain: str = 'arbitrum'
    ) -> pd.DataFrame:
        """
        Simplified run method for orchestrator integration.

        Runs backtest directly on provided signals without strategy object.
        This method provides compatibility with the phase2run.py orchestrator.

        Args:
            signals: DataFrame with trading signals (must have 'signal' column)
            price_data: DataFrame with price data
            start_date: Start date for backtest
            end_date: End date for backtest
            venue: Trading venue name
            venue_type: Type of venue (CEX, HYBRID, DEX)
            chain: Blockchain for gas costs

        Returns:
            DataFrame with backtest results and metrics
        """
        self.reset()

        # Merge signals with price data
        if len(signals) == 0:
            logger.warning("No signals provided for backtest")
            return pd.DataFrame()

        # Filter by date range if provided
        data = signals.copy()
        if start_date is not None:
            data = data[data.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            data = data[data.index <= pd.Timestamp(end_date)]

        if len(data) < 10:
            logger.warning("Insufficient data for backtest")
            return pd.DataFrame()

        # Get price column
        price_col = 'close' if 'close' in data.columns else 'price'
        if price_col not in data.columns and 'price_a' in data.columns:
            price_col = 'price_a'

        # Default signal column
        signal_col = 'signal' if 'signal' in data.columns else 'entry_signal'
        if signal_col not in data.columns:
            # If no signals, create neutral signals
            data['signal'] = 0
            signal_col = 'signal'

        # Track P&L and returns
        prev_equity = self.capital
        prev_date = None
        results_list = []

        for timestamp, row in data.iterrows():
            # Get price
            if price_col in row:
                price = row[price_col]
            else:
                # Try to get from price_data
                if timestamp in price_data.index:
                    price = price_data.loc[timestamp].get('close', 100.0)
                else:
                    price = 100.0  # Fallback

            if pd.isna(price) or price <= 0:
                continue

            # Get signal
            signal_val = row.get(signal_col, 0)
            if pd.isna(signal_val):
                signal_val = 0

            # Calculate hypothetical return based on signal
            # Positive signal = long exposure, negative = short
            position_return = 0.0
            if len(results_list) > 0:
                prev_price = results_list[-1].get('price', price)
                if prev_price > 0:
                    price_return = (price - prev_price) / prev_price
                    position_return = signal_val * price_return

            # Apply transaction costs when signal changes
            transaction_cost = 0.0
            if len(results_list) > 0:
                prev_signal = results_list[-1].get('signal', 0)
                if prev_signal != signal_val:
                    notional = self.capital * abs(signal_val - prev_signal) * self.config.base_position_pct
                    costs = self.cost_model.total_cost(
                        notional=notional,
                        venue=venue,
                        venue_type=venue_type,
                        chain=chain
                    )
                    transaction_cost = costs['total']

            # Update capital
            gross_return = prev_equity * position_return
            net_return = gross_return - transaction_cost
            current_equity = prev_equity + net_return

            # Track daily returns
            daily_return = net_return / prev_equity if prev_equity > 0 else 0.0

            # Update tracking
            self.equity_curve.append((timestamp, current_equity))
            self.daily_returns.append((timestamp, daily_return))

            # Check risk limits
            can_continue, stop_reason = self._check_risk_limits(current_equity)
            if not can_continue:
                self.is_stopped = True
                self.stop_reason = stop_reason
                logger.warning(f"Backtest stopped: {stop_reason}")
                break

            # Store result
            results_list.append({
                'timestamp': timestamp,
                'price': price,
                'signal': signal_val,
                'gross_return': gross_return,
                'transaction_cost': transaction_cost,
                'net_return': net_return,
                'equity': current_equity,
                'returns': daily_return,
                'pnl': net_return,
                'cumulative_return': (current_equity / self.config.initial_capital - 1)
            })

            prev_equity = current_equity

        # Convert to DataFrame
        if not results_list:
            return pd.DataFrame()

        results_df = pd.DataFrame(results_list)
        results_df.set_index('timestamp', inplace=True)

        # Add aggregate metrics
        if len(results_df) > 0:
            equity_series = pd.Series(
                results_df['equity'].values,
                index=results_df.index
            )
            returns_series = pd.Series(
                results_df['returns'].values,
                index=results_df.index
            )

            # Calculate metrics
            results_df.attrs['total_return'] = (equity_series.iloc[-1] / self.config.initial_capital - 1)
            results_df.attrs['sharpe_ratio'] = PerformanceMetrics.sharpe_ratio(returns_series)
            results_df.attrs['sortino_ratio'] = PerformanceMetrics.sortino_ratio(returns_series)
            results_df.attrs['max_drawdown'], _, _ = PerformanceMetrics.max_drawdown(equity_series)
            results_df.attrs['volatility'] = returns_series.std() * np.sqrt(252)
            results_df.attrs['final_capital'] = equity_series.iloc[-1]

        logger.info(f"Backtest completed: {len(results_df)} periods, "
                    f"final equity ${prev_equity:,.2f}")

        return results_df

    def get_equity_curve(self) -> pd.Series:
        """Return equity curve as pandas Series."""
        if not self.equity_curve:
            return pd.Series(dtype=float)
        return pd.Series(
            [e[1] for e in self.equity_curve],
            index=[e[0] for e in self.equity_curve],
            name='equity'
        )
    
    def get_trades_df(self) -> pd.DataFrame:
        """Return trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'trade_id': t.trade_id,
                'symbol': t.symbol,
                'strategy': t.strategy,
                'venue': t.venue,
                'venue_type': t.venue_type.value,
                'side': t.side,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'quantity': t.quantity,
                'gross_pnl': t.gross_pnl,
                'net_pnl': t.net_pnl,
                'return_pct': t.return_pct,
                'fees': t.fees,
                'slippage': t.slippage,
                'gas_cost': t.gas_cost,
                'mev_cost': t.mev_cost,
                'holding_hours': t.holding_period_hours,
                'exit_reason': t.exit_reason,
            }
            for t in self.trades
        ])


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compare_venues(
    strategy: BaseStrategy,
    data: pd.DataFrame,
    venues: Dict[str, Tuple[VenueType, str]],
    config: Optional[BacktestConfig] = None
) -> pd.DataFrame:
    """
    Compare strategy performance across different venues.
    
    Args:
        strategy: Strategy to test
        data: Input data
        venues: Dict mapping venue_name to (VenueType, chain)
        config: Backtest configuration
        
    Returns:
        DataFrame comparing venue performance
    """
    results = []
    
    for venue_name, (venue_type, chain) in venues.items():
        engine = BacktestEngine(config)
        result = engine.run_backtest(
            strategy=strategy,
            data=data.copy(),
            venue=venue_name,
            venue_type=venue_type,
            chain=chain
        )
        
        if result:
            results.append({
                'venue': venue_name,
                'venue_type': venue_type.value,
                'chain': chain,
                'sharpe_ratio': result.get('sharpe_ratio', 0),
                'total_return_pct': result.get('total_return_pct', 0),
                'max_drawdown_pct': result.get('max_drawdown_pct', 0),
                'win_rate_pct': result.get('win_rate_pct', 0),
                'total_trades': result.get('total_trades', 0),
                'total_costs': result.get('total_costs', {}).get('total', 0),
                'final_capital': result.get('final_capital', 0),
            })
    
    return pd.DataFrame(results)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'VenueType',
    'OrderType',
    'OrderSide',
    'SignalType',
    'RegimeType',
    'ExitReason',
    
    # Data classes
    'Order',
    'Fill',
    'Position',
    'TradeRecord',
    'BacktestConfig',
    
    # Core classes
    'TransactionCostModel',
    'PerformanceMetrics',
    'BaseStrategy',
    'BacktestEngine',
    
    # Utilities
    'compare_venues',
]