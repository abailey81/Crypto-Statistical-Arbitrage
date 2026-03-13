"""
Position Sizing for Crypto Pairs Trading
=========================================

Comprehensive position sizing framework for multi-venue
cryptocurrency statistical arbitrage.

Mathematical Framework
----------------------
Volatility-Weighted Sizing:

    Size = (Target_Vol × Capital) / Spread_Vol
    
    Where:
        Target_Vol = Target portfolio volatility (e.g., 15% annualized)
        Spread_Vol = Realized spread volatility
        
    Ensures equal volatility contribution from each pair.

Kelly Criterion:

    f* = (p × b - q) / b
    
    Where:
        f* = Optimal fraction of capital
        p = Win probability
        q = 1 - p (loss probability)
        b = Win/loss ratio (average win / average loss)
    
    Fractional Kelly (typically 25-50%) reduces variance.

Risk Parity:

    w_i = (1/σ_i) / Σ(1/σ_j)
    
    Equal risk contribution from each position.

Venue Adjustments:

    Size_adjusted = Base_Size × Venue_Mult × Liquidity_Mult × Tier_Mult
    
    Where:
        Venue_Mult: CEX=1.0, Hybrid=0.7, DEX=0.3
        Liquidity_Mult: min(1.0, Volume / Threshold)
        Tier_Mult: Tier1=1.0, Tier2=0.7, Tier3=0.4

Constraints:

    1. Max Position: min(Capital × 0.20, Venue_Max, Liquidity_Max)
    2. Venue Allocation: CEX ≤ 70%, Hybrid ≤ 30%, DEX ≤ 30%
    3. Sector Concentration: ≤ 40% per sector
    4. Correlation: New pairs correlation < 0.7 with existing

Author: Crypto StatArb Quantitative Research
Version: 2.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS WITH TRADING-SPECIFIC PROPERTIES
# =============================================================================

class VenueType(Enum):
    """
    Trading venue classification for position sizing.
    
    Each venue type has different liquidity characteristics,
    fee structures, and execution risks.
    """
    CEX = "CEX"
    HYBRID = "HYBRID"
    DEX = "DEX"
    
    @property
    def position_multiplier(self) -> float:
        """Base position size multiplier."""
        multipliers = {
            self.CEX: 1.0,
            self.HYBRID: 0.7,
            self.DEX: 0.3,
        }
        return multipliers.get(self, 0.5)
    
    @property
    def max_allocation(self) -> float:
        """Maximum portfolio allocation to this venue type."""
        allocations = {
            self.CEX: 0.70,
            self.HYBRID: 0.30,
            self.DEX: 0.30,
        }
        return allocations.get(self, 0.30)
    
    @property
    def max_position_usd(self) -> float:
        """Maximum single position size."""
        limits = {
            self.CEX: 200_000,
            self.HYBRID: 100_000,
            self.DEX: 50_000,
        }
        return limits.get(self, 50_000)
    
    @property
    def min_position_usd(self) -> float:
        """Minimum position size (due to fixed costs)."""
        minimums = {
            self.CEX: 1_000,
            self.HYBRID: 2_000,
            self.DEX: 5_000,
        }
        return minimums.get(self, 1_000)
    
    @property
    def typical_slippage_bps(self) -> float:
        """Typical slippage in basis points."""
        slippage = {
            self.CEX: 2.0,
            self.HYBRID: 5.0,
            self.DEX: 15.0,
        }
        return slippage.get(self, 5.0)
    
    @property
    def has_gas_costs(self) -> bool:
        """True if venue has gas costs."""
        return self in [self.HYBRID, self.DEX]
    
    @property
    def typical_gas_usd(self) -> float:
        """Typical gas cost per transaction."""
        gas = {
            self.CEX: 0.0,
            self.HYBRID: 0.50,
            self.DEX: 25.0,
        }
        return gas.get(self, 0.0)
    
    @property
    def execution_risk(self) -> str:
        """Execution risk level."""
        risks = {
            self.CEX: "low",
            self.HYBRID: "moderate",
            self.DEX: "high",
        }
        return risks.get(self, "moderate")
    
    @property
    def max_positions(self) -> int:
        """Maximum concurrent positions."""
        limits = {
            self.CEX: 10,
            self.HYBRID: 5,
            self.DEX: 3,
        }
        return limits.get(self, 5)
    
    @property
    def liquidity_threshold_usd(self) -> float:
        """Liquidity threshold for full position size."""
        thresholds = {
            self.CEX: 1_000_000,
            self.HYBRID: 500_000,
            self.DEX: 100_000,
        }
        return thresholds.get(self, 500_000)


class SizingMethod(Enum):
    """
    Position sizing methodology.
    
    Each method has different risk characteristics and
    suitability for different market conditions.
    """
    EQUAL_DOLLAR = "equal_dollar"
    VOLATILITY_WEIGHTED = "volatility_weighted"
    KELLY = "kelly"
    RISK_PARITY = "risk_parity"
    VENUE_ADJUSTED = "venue_adjusted"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    
    @property
    def description(self) -> str:
        """Method description."""
        descriptions = {
            self.EQUAL_DOLLAR: "Equal dollar allocation per pair",
            self.VOLATILITY_WEIGHTED: "Size inversely to spread volatility",
            self.KELLY: "Kelly criterion (fractional) based on win rate",
            self.RISK_PARITY: "Equal risk contribution from each pair",
            self.VENUE_ADJUSTED: "Adjusted for venue liquidity and type",
            self.CONFIDENCE_WEIGHTED: "Weighted by cointegration confidence",
        }
        return descriptions.get(self, "Unknown method")
    
    @property
    def requires_historical_data(self) -> bool:
        """True if method needs historical trade data."""
        return self == self.KELLY
    
    @property
    def requires_volatility(self) -> bool:
        """True if method needs volatility estimate."""
        return self in [self.VOLATILITY_WEIGHTED, self.RISK_PARITY]
    
    @property
    def risk_profile(self) -> str:
        """Risk profile of sizing method."""
        profiles = {
            self.EQUAL_DOLLAR: "neutral",
            self.VOLATILITY_WEIGHTED: "conservative",
            self.KELLY: "aggressive",
            self.RISK_PARITY: "conservative",
            self.VENUE_ADJUSTED: "adaptive",
            self.CONFIDENCE_WEIGHTED: "moderate",
        }
        return profiles.get(self, "neutral")
    
    @property
    def recommended_for(self) -> str:
        """Recommended use case."""
        cases = {
            self.EQUAL_DOLLAR: "Simple strategies, backtesting",
            self.VOLATILITY_WEIGHTED: "Multiple pairs with varying volatility",
            self.KELLY: "Strategies with stable win rates",
            self.RISK_PARITY: "Diversified portfolios",
            self.VENUE_ADJUSTED: "Multi-venue strategies",
            self.CONFIDENCE_WEIGHTED: "Varying quality pairs",
        }
        return cases.get(self, "General use")


class ConstraintType(Enum):
    """
    Types of position constraints applied.
    
    Tracking which constraints are binding helps optimize
    the portfolio and identify bottlenecks.
    """
    MAX_POSITION = "max_position"
    MIN_POSITION = "min_position"
    MAX_WEIGHT = "max_pair_weight"
    VENUE_LIMIT = "venue_allocation_limit"
    SECTOR_LIMIT = "sector_concentration_limit"
    LIQUIDITY_LIMIT = "liquidity_limit"
    TIER_SCALING = "tier_based_scaling"
    CORRELATION_LIMIT = "correlation_limit"
    CAPITAL_LIMIT = "capital_available"
    
    @property
    def is_hard_constraint(self) -> bool:
        """True if constraint cannot be relaxed."""
        hard = [
            self.MAX_POSITION, self.MIN_POSITION,
            self.VENUE_LIMIT, self.CAPITAL_LIMIT
        ]
        return self in hard
    
    @property
    def priority(self) -> int:
        """Constraint priority (lower = checked first)."""
        priorities = {
            self.CAPITAL_LIMIT: 1,
            self.MAX_POSITION: 2,
            self.VENUE_LIMIT: 3,
            self.SECTOR_LIMIT: 4,
            self.CORRELATION_LIMIT: 5,
            self.LIQUIDITY_LIMIT: 6,
            self.TIER_SCALING: 7,
            self.MAX_WEIGHT: 8,
            self.MIN_POSITION: 9,
        }
        return priorities.get(self, 10)
    
    @property
    def description(self) -> str:
        """Constraint description."""
        descriptions = {
            self.MAX_POSITION: "Maximum position size reached",
            self.MIN_POSITION: "Below minimum position size",
            self.MAX_WEIGHT: "Maximum portfolio weight reached",
            self.VENUE_LIMIT: "Venue allocation limit reached",
            self.SECTOR_LIMIT: "Sector concentration limit reached",
            self.LIQUIDITY_LIMIT: "Liquidity insufficient for size",
            self.TIER_SCALING: "Position scaled by pair tier",
            self.CORRELATION_LIMIT: "Too correlated with existing",
            self.CAPITAL_LIMIT: "Insufficient capital",
        }
        return descriptions.get(self, "Unknown constraint")


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class VenueSizingConfig:
    """
    Configuration for venue-specific position sizing.
    
    Contains all parameters needed to determine maximum
    position sizes for a specific venue type.
    """
    venue_type: VenueType
    max_position_usd: float
    min_position_usd: float
    liquidity_scaling: bool = True
    max_volume_pct: float = 0.05
    max_tvl_pct: float = 0.10
    slippage_limit_bps: float = 50.0
    
    @property
    def has_tvl_constraint(self) -> bool:
        """True if TVL constraint applies."""
        return self.venue_type == VenueType.DEX
    
    def calculate_liquidity_limit(
        self,
        daily_volume: float,
        tvl: Optional[float] = None
    ) -> float:
        """Calculate maximum position from liquidity."""
        volume_limit = daily_volume * self.max_volume_pct
        
        if self.has_tvl_constraint and tvl:
            tvl_limit = tvl * self.max_tvl_pct
            return min(volume_limit, tvl_limit)
        
        return volume_limit
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'venue_type': self.venue_type.value,
            'max_position_usd': self.max_position_usd,
            'min_position_usd': self.min_position_usd,
            'liquidity_scaling': self.liquidity_scaling,
            'max_volume_pct': self.max_volume_pct,
        }


# Default venue configurations
DEFAULT_VENUE_CONFIGS: Dict[VenueType, VenueSizingConfig] = {
    VenueType.CEX: VenueSizingConfig(
        venue_type=VenueType.CEX,
        max_position_usd=200_000,
        min_position_usd=1_000,
        liquidity_scaling=True,
        max_volume_pct=0.05,
    ),
    VenueType.HYBRID: VenueSizingConfig(
        venue_type=VenueType.HYBRID,
        max_position_usd=100_000,
        min_position_usd=2_000,
        liquidity_scaling=True,
        max_volume_pct=0.03,
    ),
    VenueType.DEX: VenueSizingConfig(
        venue_type=VenueType.DEX,
        max_position_usd=50_000,
        min_position_usd=5_000,
        liquidity_scaling=True,
        max_volume_pct=0.02,
        max_tvl_pct=0.10,
    ),
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PairMetrics:
    """
    Comprehensive metrics for position sizing a trading pair.
    
    Contains all information needed to calculate optimal
    position size for a specific pair.
    """
    # Identification
    symbol_a: str
    symbol_b: str
    venue_type: VenueType
    tier: int = 1
    
    # Volatility metrics
    spread_volatility: float = 0.0
    spread_volatility_annualized: float = 0.0
    price_a_volatility: float = 0.0
    price_b_volatility: float = 0.0
    
    # Cointegration metrics
    half_life: float = 7.0
    hedge_ratio: float = 1.0
    cointegration_pvalue: float = 0.05
    confidence_score: float = 0.5
    
    # Liquidity metrics
    liquidity_a: float = 0.0
    liquidity_b: float = 0.0
    avg_daily_volume_a: float = 0.0
    avg_daily_volume_b: float = 0.0
    tvl_a: Optional[float] = None
    tvl_b: Optional[float] = None
    
    # Historical performance (for Kelly)
    win_rate: Optional[float] = None
    avg_win_pct: Optional[float] = None
    avg_loss_pct: Optional[float] = None
    n_trades: int = 0
    
    # Correlation metrics
    correlation_btc: Optional[float] = None
    correlation_with_portfolio: float = 0.0
    
    # Sector
    sector: str = "OTHER"
    
    @property
    def pair_name(self) -> str:
        """Combined pair name."""
        return f"{self.symbol_a}_{self.symbol_b}"
    
    @property
    def min_liquidity(self) -> float:
        """Minimum liquidity of the two legs."""
        return min(self.liquidity_a, self.liquidity_b)
    
    @property
    def liquidity_score(self) -> float:
        """Liquidity score (0-1)."""
        threshold = self.venue_type.liquidity_threshold_usd
        return min(1.0, self.min_liquidity / threshold)
    
    @property
    def tier_multiplier(self) -> float:
        """Position multiplier based on tier."""
        multipliers = {1: 1.0, 2: 0.7, 3: 0.4}
        return multipliers.get(self.tier, 0.4)
    
    @property
    def quality_score(self) -> float:
        """Overall quality score for sizing."""
        score = 0.0
        
        # Cointegration contribution
        if self.cointegration_pvalue < 0.01:
            score += 0.3
        elif self.cointegration_pvalue < 0.05:
            score += 0.2
        
        # Half-life contribution
        if 1 <= self.half_life <= 7:
            score += 0.3
        elif self.half_life <= 14:
            score += 0.2
        
        # Liquidity contribution
        score += self.liquidity_score * 0.2
        
        # Confidence contribution
        score += self.confidence_score * 0.2
        
        return min(1.0, score)
    
    @property
    def has_kelly_data(self) -> bool:
        """True if historical data available for Kelly."""
        return (
            self.win_rate is not None and
            self.avg_win_pct is not None and
            self.avg_loss_pct is not None and
            self.n_trades >= 20
        )
    
    @property
    def kelly_fraction(self) -> float:
        """Optimal Kelly fraction (if data available)."""
        if not self.has_kelly_data:
            return 0.0
        
        p = self.win_rate
        q = 1 - p
        
        if self.avg_loss_pct == 0:
            return 0.0
        
        b = abs(self.avg_win_pct / self.avg_loss_pct)
        kelly = (p * b - q) / b
        
        return max(0, kelly)
    
    @property
    def is_high_correlation_risk(self) -> bool:
        """True if correlation with portfolio is high."""
        return self.correlation_with_portfolio > 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pair_name': self.pair_name,
            'symbol_a': self.symbol_a,
            'symbol_b': self.symbol_b,
            'venue_type': self.venue_type.value,
            'tier': self.tier,
            'spread_volatility': self.spread_volatility,
            'half_life': self.half_life,
            'hedge_ratio': self.hedge_ratio,
            'liquidity_score': round(self.liquidity_score, 3),
            'quality_score': round(self.quality_score, 3),
            'tier_multiplier': self.tier_multiplier,
            'has_kelly_data': self.has_kelly_data,
            'sector': self.sector,
        }


@dataclass
class PositionSize:
    """
    Calculated position size for a trading pair.
    
    Contains the final size, breakdown by leg, and
    tracking of which constraints were applied.
    """
    # Identification
    pair: Tuple[str, str]
    venue_type: VenueType
    method: SizingMethod
    
    # Size details
    notional_usd: float
    size_a_usd: float
    size_b_usd: float
    size_a_units: float = 0.0
    size_b_units: float = 0.0
    
    # Portfolio weight
    weight: float = 0.0
    
    # Scaling factors applied
    scaling_factors: Dict[str, float] = field(default_factory=dict)
    
    # Constraints applied
    constraints_applied: List[ConstraintType] = field(default_factory=list)
    
    # Risk metrics
    expected_volatility_contribution: float = 0.0
    margin_required: float = 0.0
    
    @property
    def pair_name(self) -> str:
        """Combined pair name."""
        return f"{self.pair[0]}_{self.pair[1]}"
    
    @property
    def is_tradeable(self) -> bool:
        """True if position size is tradeable."""
        return self.notional_usd >= self.venue_type.min_position_usd
    
    @property
    def was_constrained(self) -> bool:
        """True if any constraints were applied."""
        return len(self.constraints_applied) > 0
    
    @property
    def primary_constraint(self) -> Optional[ConstraintType]:
        """Most impactful constraint (first applied)."""
        if not self.constraints_applied:
            return None
        return min(self.constraints_applied, key=lambda c: c.priority)
    
    @property
    def n_constraints(self) -> int:
        """Number of constraints applied."""
        return len(self.constraints_applied)
    
    @property
    def total_scaling(self) -> float:
        """Total scaling factor applied."""
        if not self.scaling_factors:
            return 1.0
        
        total = 1.0
        for factor in self.scaling_factors.values():
            total *= factor
        return total
    
    @property
    def risk_weight(self) -> float:
        """Risk-adjusted weight."""
        return self.weight * self.expected_volatility_contribution
    
    @property
    def leverage_ratio(self) -> float:
        """Implied leverage ratio."""
        if self.margin_required <= 0:
            return 1.0
        return self.notional_usd / self.margin_required
    
    @property
    def leg_a_weight(self) -> float:
        """Weight of leg A in position."""
        if self.notional_usd <= 0:
            return 0.5
        return self.size_a_usd / self.notional_usd
    
    @property
    def leg_b_weight(self) -> float:
        """Weight of leg B in position."""
        return 1 - self.leg_a_weight
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pair_name': self.pair_name,
            'venue_type': self.venue_type.value,
            'method': self.method.value,
            'notional_usd': round(self.notional_usd, 2),
            'size_a_usd': round(self.size_a_usd, 2),
            'size_b_usd': round(self.size_b_usd, 2),
            'weight': round(self.weight, 4),
            'is_tradeable': self.is_tradeable,
            'was_constrained': self.was_constrained,
            'primary_constraint': self.primary_constraint.value if self.primary_constraint else None,
            'n_constraints': self.n_constraints,
            'total_scaling': round(self.total_scaling, 3),
        }
    
    def __repr__(self) -> str:
        return (
            f"PositionSize({self.pair_name}, "
            f"${self.notional_usd:,.0f}, "
            f"w={self.weight:.2%})"
        )


@dataclass
class PortfolioState:
    """
    Current state of the portfolio for constraint checking.
    
    Tracks allocations, exposures, and open positions.
    """
    total_capital: float
    cash_available: float
    
    # Current allocations
    venue_allocations: Dict[VenueType, float] = field(default_factory=dict)
    sector_allocations: Dict[str, float] = field(default_factory=dict)
    
    # Position counts
    position_count: int = 0
    venue_position_counts: Dict[VenueType, int] = field(default_factory=dict)
    
    # Open positions
    open_positions: Dict[str, PositionSize] = field(default_factory=dict)
    
    # Correlation matrix
    position_correlations: Optional[pd.DataFrame] = None
    
    @property
    def total_allocated(self) -> float:
        """Total capital currently allocated."""
        return sum(p.notional_usd for p in self.open_positions.values())
    
    @property
    def allocation_pct(self) -> float:
        """Percentage of capital allocated."""
        if self.total_capital <= 0:
            return 0.0
        return self.total_allocated / self.total_capital
    
    @property
    def cash_pct(self) -> float:
        """Percentage of capital in cash."""
        return 1 - self.allocation_pct
    
    def get_venue_allocation(self, venue_type: VenueType) -> float:
        """Get current allocation to venue type."""
        return self.venue_allocations.get(venue_type, 0.0)
    
    def get_sector_allocation(self, sector: str) -> float:
        """Get current allocation to sector."""
        return self.sector_allocations.get(sector, 0.0)
    
    def get_venue_position_count(self, venue_type: VenueType) -> int:
        """Get number of positions at venue."""
        return self.venue_position_counts.get(venue_type, 0)
    
    def can_add_position(
        self,
        venue_type: VenueType,
        size_usd: float,
        sector: str
    ) -> Tuple[bool, Optional[ConstraintType]]:
        """Check if new position can be added."""
        # Capital check
        if size_usd > self.cash_available:
            return False, ConstraintType.CAPITAL_LIMIT
        
        # Venue position count
        current_count = self.get_venue_position_count(venue_type)
        if current_count >= venue_type.max_positions:
            return False, ConstraintType.VENUE_LIMIT
        
        # Venue allocation
        new_venue_alloc = self.get_venue_allocation(venue_type) + size_usd / self.total_capital
        if new_venue_alloc > venue_type.max_allocation:
            return False, ConstraintType.VENUE_LIMIT
        
        # Sector allocation
        new_sector_alloc = self.get_sector_allocation(sector) + size_usd / self.total_capital
        if new_sector_alloc > 0.40:  # 40% sector limit
            return False, ConstraintType.SECTOR_LIMIT
        
        return True, None
    
    def add_position(self, position: PositionSize, sector: str):
        """Add position to portfolio state."""
        self.open_positions[position.pair_name] = position
        self.position_count += 1
        
        # Update venue counts and allocations
        self.venue_position_counts[position.venue_type] = (
            self.venue_position_counts.get(position.venue_type, 0) + 1
        )
        self.venue_allocations[position.venue_type] = (
            self.venue_allocations.get(position.venue_type, 0) +
            position.weight
        )
        
        # Update sector allocations
        self.sector_allocations[sector] = (
            self.sector_allocations.get(sector, 0) +
            position.weight
        )
        
        # Update cash
        self.cash_available -= position.notional_usd
    
    def remove_position(self, pair_name: str) -> Optional[PositionSize]:
        """Remove position from portfolio state."""
        if pair_name not in self.open_positions:
            return None
        
        position = self.open_positions.pop(pair_name)
        self.position_count -= 1
        
        # Update venue counts
        self.venue_position_counts[position.venue_type] -= 1
        self.venue_allocations[position.venue_type] -= position.weight
        
        # Update cash
        self.cash_available += position.notional_usd
        
        return position
    
    def summary(self) -> Dict[str, Any]:
        """Get portfolio state summary."""
        return {
            'total_capital': self.total_capital,
            'cash_available': round(self.cash_available, 2),
            'allocation_pct': round(self.allocation_pct * 100, 1),
            'position_count': self.position_count,
            'venue_allocations': {
                v.value: round(a * 100, 1)
                for v, a in self.venue_allocations.items()
            },
            'sector_allocations': {
                s: round(a * 100, 1)
                for s, a in self.sector_allocations.items()
            },
        }


# =============================================================================
# POSITION SIZER
# =============================================================================

class PositionSizer:
    """
    Comprehensive position sizing for pairs trading.
    
    Supports multiple sizing methods with comprehensive
    constraint handling and venue-specific adjustments.
    
    Parameters
    ----------
    total_capital : float
        Total capital for trading
    method : SizingMethod
        Primary sizing method
    target_volatility : float
        Target annualized portfolio volatility
    max_pair_weight : float
        Maximum weight for any single pair
    kelly_fraction : float
        Fraction of Kelly to use (0.25 = quarter Kelly)
    
    Example
    -------
    >>> sizer = PositionSizer(
    ...     total_capital=1_000_000,
    ...     method=SizingMethod.VOLATILITY_WEIGHTED,
    ...     target_volatility=0.15
    ... )
    >>> 
    >>> size = sizer.size_pair(metrics, price_a=100, price_b=50)
    >>> print(f"Position: ${size.notional_usd:,.0f}")
    """
    
    def __init__(
        self,
        total_capital: float,
        method: SizingMethod = SizingMethod.VOLATILITY_WEIGHTED,
        target_volatility: float = 0.15,
        max_pair_weight: float = 0.20,
        max_sector_weight: float = 0.40,
        kelly_fraction: float = 0.25,
        min_cash_reserve: float = 0.20,
        venue_configs: Optional[Dict[VenueType, VenueSizingConfig]] = None
    ):
        """Initialize position sizer."""
        self.total_capital = total_capital
        self.method = method
        self.target_volatility = target_volatility
        self.max_pair_weight = max_pair_weight
        self.max_sector_weight = max_sector_weight
        self.kelly_fraction = kelly_fraction
        self.min_cash_reserve = min_cash_reserve
        
        self.venue_configs = venue_configs or DEFAULT_VENUE_CONFIGS
        
        # Initialize portfolio state
        self.portfolio_state = PortfolioState(
            total_capital=total_capital,
            cash_available=total_capital * (1 - min_cash_reserve)
        )
        
        logger.info(
            f"PositionSizer initialized: capital=${total_capital:,.0f}, "
            f"method={method.value}, target_vol={target_volatility:.0%}"
        )
    
    def size_pair(
        self,
        metrics: PairMetrics,
        price_a: float,
        price_b: float
    ) -> PositionSize:
        """
        Calculate position size for a single pair.
        
        Args:
            metrics: Pair metrics for sizing
            price_a: Current price of token A
            price_b: Current price of token B
            
        Returns:
            PositionSize with complete breakdown
        """
        # Calculate base notional
        if self.method == SizingMethod.EQUAL_DOLLAR:
            notional = self._equal_dollar_size(metrics)
        elif self.method == SizingMethod.VOLATILITY_WEIGHTED:
            notional = self._volatility_weighted_size(metrics)
        elif self.method == SizingMethod.KELLY:
            notional = self._kelly_size(metrics)
        elif self.method == SizingMethod.RISK_PARITY:
            notional = self._risk_parity_size(metrics)
        elif self.method == SizingMethod.VENUE_ADJUSTED:
            notional = self._venue_adjusted_size(metrics)
        elif self.method == SizingMethod.CONFIDENCE_WEIGHTED:
            notional = self._confidence_weighted_size(metrics)
        else:
            notional = self._equal_dollar_size(metrics)
        
        # Apply constraints
        notional, constraints, scaling_factors = self._apply_constraints(
            notional, metrics
        )
        
        # Calculate leg sizes
        hr = metrics.hedge_ratio
        size_b_usd = notional / (1 + hr)
        size_a_usd = notional * hr / (1 + hr)
        
        # Convert to units
        size_a_units = size_a_usd / price_a if price_a > 0 else 0
        size_b_units = size_b_usd / price_b if price_b > 0 else 0
        
        # Calculate weight
        weight = notional / self.total_capital
        
        # Expected volatility contribution
        vol_contribution = metrics.spread_volatility * weight
        
        return PositionSize(
            pair=(metrics.symbol_a, metrics.symbol_b),
            venue_type=metrics.venue_type,
            method=self.method,
            notional_usd=notional,
            size_a_usd=size_a_usd,
            size_b_usd=size_b_usd,
            size_a_units=size_a_units,
            size_b_units=size_b_units,
            weight=weight,
            scaling_factors=scaling_factors,
            constraints_applied=constraints,
            expected_volatility_contribution=vol_contribution,
        )
    
    def calculate_sizes(
        self,
        pair_metrics_list: List[PairMetrics],
        prices: Dict[str, float]
    ) -> List[PositionSize]:
        """
        Calculate sizes for multiple pairs with portfolio constraints.
        
        Args:
            pair_metrics_list: List of pair metrics
            prices: Current prices by symbol
            
        Returns:
            List of position sizes
        """
        sizes = []
        
        # Sort by quality (higher quality first)
        sorted_metrics = sorted(
            pair_metrics_list,
            key=lambda m: m.quality_score,
            reverse=True
        )
        
        for metrics in sorted_metrics:
            price_a = prices.get(metrics.symbol_a, 0)
            price_b = prices.get(metrics.symbol_b, 0)
            
            if price_a <= 0 or price_b <= 0:
                logger.warning(f"Missing prices for {metrics.pair_name}")
                continue
            
            # Check if position can be added
            can_add, constraint = self.portfolio_state.can_add_position(
                metrics.venue_type,
                self.venue_configs[metrics.venue_type].max_position_usd,
                metrics.sector
            )
            
            if not can_add:
                logger.debug(f"Cannot add {metrics.pair_name}: {constraint}")
                continue
            
            # Calculate size
            size = self.size_pair(metrics, price_a, price_b)
            
            if size.is_tradeable:
                self.portfolio_state.add_position(size, metrics.sector)
                sizes.append(size)
        
        return sizes
    
    def _equal_dollar_size(self, metrics: PairMetrics) -> float:
        """Equal dollar allocation."""
        return self.total_capital * self.max_pair_weight
    
    def _volatility_weighted_size(self, metrics: PairMetrics) -> float:
        """Volatility-weighted sizing."""
        if metrics.spread_volatility <= 0:
            return 0.0
        
        # Target equal volatility contribution
        n_pairs = max(5, self.portfolio_state.position_count + 1)
        target_pair_vol = self.target_volatility / np.sqrt(n_pairs)
        
        notional = (target_pair_vol * self.total_capital) / metrics.spread_volatility
        
        return notional
    
    def _kelly_size(self, metrics: PairMetrics) -> float:
        """Kelly criterion sizing."""
        if not metrics.has_kelly_data:
            return self._volatility_weighted_size(metrics)
        
        kelly_pct = metrics.kelly_fraction * self.kelly_fraction
        kelly_pct = min(kelly_pct, self.max_pair_weight)
        kelly_pct = max(kelly_pct, 0)
        
        return kelly_pct * self.total_capital
    
    def _risk_parity_size(self, metrics: PairMetrics) -> float:
        """Risk parity sizing."""
        return self._volatility_weighted_size(metrics)
    
    def _venue_adjusted_size(self, metrics: PairMetrics) -> float:
        """Venue-adjusted sizing."""
        base_size = self._volatility_weighted_size(metrics)
        
        venue_config = self.venue_configs.get(
            metrics.venue_type,
            DEFAULT_VENUE_CONFIGS[VenueType.CEX]
        )
        
        # Liquidity adjustment
        if venue_config.liquidity_scaling:
            liquidity_limit = venue_config.calculate_liquidity_limit(
                metrics.min_liquidity,
                min(metrics.tvl_a or float('inf'), metrics.tvl_b or float('inf'))
            )
            base_size = min(base_size, liquidity_limit)
        
        # Apply venue limits
        base_size = min(base_size, venue_config.max_position_usd)
        
        return base_size
    
    def _confidence_weighted_size(self, metrics: PairMetrics) -> float:
        """Confidence-weighted sizing."""
        base_size = self._volatility_weighted_size(metrics)
        return base_size * metrics.confidence_score
    
    def _apply_constraints(
        self,
        notional: float,
        metrics: PairMetrics
    ) -> Tuple[float, List[ConstraintType], Dict[str, float]]:
        """Apply all position constraints."""
        constraints = []
        scaling_factors = {}
        
        venue_config = self.venue_configs.get(
            metrics.venue_type,
            DEFAULT_VENUE_CONFIGS[VenueType.CEX]
        )
        
        # Max position
        if notional > venue_config.max_position_usd:
            scaling_factors['max_position'] = venue_config.max_position_usd / notional
            notional = venue_config.max_position_usd
            constraints.append(ConstraintType.MAX_POSITION)
        
        # Max weight
        max_by_weight = self.total_capital * self.max_pair_weight
        if notional > max_by_weight:
            scaling_factors['max_weight'] = max_by_weight / notional
            notional = max_by_weight
            constraints.append(ConstraintType.MAX_WEIGHT)
        
        # Cash available
        if notional > self.portfolio_state.cash_available:
            scaling_factors['capital'] = self.portfolio_state.cash_available / notional
            notional = self.portfolio_state.cash_available
            constraints.append(ConstraintType.CAPITAL_LIMIT)
        
        # Tier scaling
        if metrics.tier > 1:
            tier_mult = metrics.tier_multiplier
            scaling_factors['tier'] = tier_mult
            notional *= tier_mult
            constraints.append(ConstraintType.TIER_SCALING)
        
        # Liquidity scaling
        if metrics.liquidity_score < 1.0:
            scaling_factors['liquidity'] = metrics.liquidity_score
            notional *= metrics.liquidity_score
            constraints.append(ConstraintType.LIQUIDITY_LIMIT)
        
        # Minimum position check
        if notional < venue_config.min_position_usd:
            if notional > 0:
                constraints.append(ConstraintType.MIN_POSITION)
                notional = 0
        
        return notional, constraints, scaling_factors
    
    def get_summary(self) -> Dict[str, Any]:
        """Get sizer summary."""
        return {
            'total_capital': self.total_capital,
            'method': self.method.value,
            'target_volatility': self.target_volatility,
            'max_pair_weight': self.max_pair_weight,
            'kelly_fraction': self.kelly_fraction,
            'portfolio_state': self.portfolio_state.summary(),
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_spread_volatility(
    prices_a: pd.Series,
    prices_b: pd.Series,
    hedge_ratio: float,
    window: int = 20
) -> float:
    """
    Calculate annualized spread volatility.
    
    Args:
        prices_a, prices_b: Price series
        hedge_ratio: Hedge ratio
        window: Lookback window
        
    Returns:
        Annualized spread volatility
    """
    spread = prices_b - hedge_ratio * prices_a
    spread_returns = spread.pct_change().dropna()
    
    if len(spread_returns) < window:
        return np.nan
    
    daily_vol = spread_returns.rolling(window=window).std().iloc[-1]
    return daily_vol * np.sqrt(365)


def create_pair_metrics(
    symbol_a: str,
    symbol_b: str,
    prices_a: pd.Series,
    prices_b: pd.Series,
    hedge_ratio: float,
    half_life: float,
    venue_type: VenueType,
    liquidity_a: float,
    liquidity_b: float,
    tier: int = 1,
    cointegration_pvalue: float = 0.05,
    confidence_score: float = 0.5,
    sector: str = "OTHER",
    trade_history: Optional[pd.DataFrame] = None
) -> PairMetrics:
    """
    Create PairMetrics from price data and parameters.
    
    Args:
        symbol_a, symbol_b: Token symbols
        prices_a, prices_b: Price series
        hedge_ratio: Cointegration hedge ratio
        half_life: Mean reversion half-life
        venue_type: Trading venue type
        liquidity_a, liquidity_b: Liquidity metrics
        tier: Pair tier
        cointegration_pvalue: Cointegration p-value
        confidence_score: Cointegration confidence
        sector: Sector classification
        trade_history: Historical trades for Kelly
        
    Returns:
        Complete PairMetrics object
    """
    spread_vol = calculate_spread_volatility(prices_a, prices_b, hedge_ratio)
    
    # Kelly inputs from trade history
    win_rate = None
    avg_win = None
    avg_loss = None
    n_trades = 0
    
    if trade_history is not None and len(trade_history) > 0:
        n_trades = len(trade_history)
        wins = trade_history[trade_history['pnl_pct'] > 0]['pnl_pct']
        losses = trade_history[trade_history['pnl_pct'] <= 0]['pnl_pct']
        
        if n_trades > 0:
            win_rate = len(wins) / n_trades
        if len(wins) > 0:
            avg_win = wins.mean()
        if len(losses) > 0:
            avg_loss = losses.mean()
    
    return PairMetrics(
        symbol_a=symbol_a,
        symbol_b=symbol_b,
        venue_type=venue_type,
        tier=tier,
        spread_volatility=spread_vol if not np.isnan(spread_vol) else 0.0,
        spread_volatility_annualized=spread_vol if not np.isnan(spread_vol) else 0.0,
        half_life=half_life,
        hedge_ratio=hedge_ratio,
        cointegration_pvalue=cointegration_pvalue,
        confidence_score=confidence_score,
        liquidity_a=liquidity_a,
        liquidity_b=liquidity_b,
        win_rate=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        n_trades=n_trades,
        sector=sector,
    )