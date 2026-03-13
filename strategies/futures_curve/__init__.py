"""
BTC Futures Curve Trading Package - Part 2

Multi-venue futures curve analysis and calendar spread strategies
across CEX, hybrid, and DEX venues per Part 2 Section 3 requirements.

Strategies (Section 3.2):
    A: Traditional Calendar Spreads (calendar_spreads.py)
    B: Cross-Venue Calendar Arbitrage (multi_venue_analyzer.py)
    C: Synthetic Futures from Perp Funding (synthetic_futures.py)
    D: Multi-Venue Roll Optimization (roll_optimization.py)

Venues (Section 3.1):
    CEX: Binance, Deribit, CME
    Hybrid: Hyperliquid, dYdX V4
    DEX: GMX

Analysis (Section 3.3):
    Walk-forward optimization (18-month train / 6-month test windows)
    60+ performance metrics
    Crisis event analysis (COVID, May 2021, LUNA, FTX, 3AC)
    Regime-conditional performance
    Cross-venue funding rate normalization
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

__version__ = '2.0.0'

logger = logging.getLogger(__name__)


class TermStructureRegime(Enum):
    """Market regime classification based on annualized basis."""
    STEEP_CONTANGO = "steep_contango"
    MILD_CONTANGO = "mild_contango"
    FLAT = "flat"
    MILD_BACKWARDATION = "mild_backwardation"
    STEEP_BACKWARDATION = "steep_backwardation"
    
    @classmethod
    def from_basis(cls, annualized_basis_pct: float) -> 'TermStructureRegime':
        if annualized_basis_pct > 20.0:
            return cls.STEEP_CONTANGO
        elif annualized_basis_pct > 5.0:
            return cls.MILD_CONTANGO
        elif annualized_basis_pct > -5.0:
            return cls.FLAT
        elif annualized_basis_pct > -20.0:
            return cls.MILD_BACKWARDATION
        return cls.STEEP_BACKWARDATION
    
    @property
    def is_contango(self) -> bool:
        return self in [self.STEEP_CONTANGO, self.MILD_CONTANGO]
    
    @property
    def is_backwardation(self) -> bool:
        return self in [self.STEEP_BACKWARDATION, self.MILD_BACKWARDATION]
    
    @property
    def is_tradeable(self) -> bool:
        return self in [self.STEEP_CONTANGO, self.STEEP_BACKWARDATION]
    
    @property
    def calendar_spread_direction(self) -> int:
        directions = {
            self.STEEP_CONTANGO: 1, self.MILD_CONTANGO: 0, self.FLAT: 0,
            self.MILD_BACKWARDATION: 0, self.STEEP_BACKWARDATION: -1,
        }
        return directions.get(self, 0)
    
    @property
    def expected_half_life_days(self) -> int:
        half_lives = {
            self.STEEP_CONTANGO: 30, self.MILD_CONTANGO: 60, self.FLAT: 120,
            self.MILD_BACKWARDATION: 45, self.STEEP_BACKWARDATION: 25,
        }
        return half_lives.get(self, 60)
    
    @property
    def signal_strength_multiplier(self) -> float:
        multipliers = {
            self.STEEP_CONTANGO: 1.0, self.MILD_CONTANGO: 0.5, self.FLAT: 0.0,
            self.MILD_BACKWARDATION: 0.5, self.STEEP_BACKWARDATION: 1.0,
        }
        return multipliers.get(self, 0.0)


class VenueType(Enum):
    """Classification of trading venue types for futures."""
    CEX_FUTURES = "cex_futures"
    CEX_PERPETUAL = "cex_perpetual"
    CME_FUTURES = "cme_futures"
    CME = "cme"  # Alias for CME futures
    HYBRID_PERPETUAL = "hybrid_perpetual"
    DEX_PERPETUAL = "dex_perpetual"
    
    @property
    def funding_interval_hours(self) -> Optional[int]:
        intervals = {
            self.CEX_FUTURES: None, self.CEX_PERPETUAL: 8, self.CME_FUTURES: None,
            self.HYBRID_PERPETUAL: 1, self.DEX_PERPETUAL: 1,
        }
        return intervals.get(self)
    
    @property
    def periods_per_year(self) -> int:
        periods = {
            self.CEX_FUTURES: 0, self.CEX_PERPETUAL: 1095, self.CME_FUTURES: 0,
            self.HYBRID_PERPETUAL: 8760, self.DEX_PERPETUAL: 8760,
        }
        return periods.get(self, 0)
    
    @property
    def has_funding(self) -> bool:
        return self.funding_interval_hours is not None
    
    @property
    def is_on_chain(self) -> bool:
        return self in [self.HYBRID_PERPETUAL, self.DEX_PERPETUAL]
    
    @property
    def typical_taker_fee_bps(self) -> float:
        fees = {
            self.CEX_FUTURES: 4.0, self.CEX_PERPETUAL: 4.0, self.CME_FUTURES: 1.0,
            self.HYBRID_PERPETUAL: 2.5, self.DEX_PERPETUAL: 7.0,
        }
        return fees.get(self, 4.0)
    
    @property
    def capacity_usd(self) -> float:
        capacity = {
            self.CEX_FUTURES: 500_000_000, self.CEX_PERPETUAL: 1_000_000_000,
            self.CME_FUTURES: 2_000_000_000, self.HYBRID_PERPETUAL: 100_000_000,
            self.DEX_PERPETUAL: 50_000_000,
        }
        return capacity.get(self, 100_000_000)


class SpreadDirection(Enum):
    """Calendar spread direction."""
    LONG = 1
    SHORT = -1
    FLAT = 0
    
    @property
    def description(self) -> str:
        descriptions = {
            self.LONG: "Long Calendar (long near, short far)",
            self.SHORT: "Short Calendar (short near, long far)",
            self.FLAT: "No Position",
        }
        return descriptions.get(self, "Unknown")
    
    @property
    def near_leg_direction(self) -> int:
        return self.value
    
    @property
    def far_leg_direction(self) -> int:
        return -self.value


class ExitReason(Enum):
    """Reason for closing a calendar spread position."""
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    EXPIRY_APPROACH = "expiry_approach"
    MAX_HOLD = "max_hold_period"
    REGIME_CHANGE = "regime_change"
    END_OF_DATA = "end_of_data"
    
    @property
    def is_profitable_exit(self) -> bool:
        return self in [self.PROFIT_TARGET]
    
    @property
    def is_risk_exit(self) -> bool:
        return self in [self.STOP_LOSS]


class CurveShape(Enum):
    """Term structure curve shape classification."""
    NORMAL = "normal"
    INVERTED = "inverted"
    HUMPED = "humped"
    FLAT = "flat"
    KINKED = "kinked"
    
    @property
    def is_monotonic(self) -> bool:
        return self in [self.NORMAL, self.INVERTED]


class InterpolationMethod(Enum):
    """Method for interpolating term structure curve."""
    LINEAR = "linear"
    CUBIC = "cubic"
    PCHIP = "pchip"
    NELSON_SIEGEL = "nelson_siegel"
    
    @property
    def preserves_monotonicity(self) -> bool:
        return self in [self.LINEAR, self.PCHIP]


@dataclass
class VenueCosts:
    """Comprehensive transaction cost model for a trading venue."""
    venue: str = ""
    venue_type: VenueType = VenueType.CEX_PERPETUAL
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 4.0
    slippage_bps: float = 1.0
    gas_cost_usd: float = 0.0
    roll_cost_bps: float = 0.0
    per_contract_fee: Optional[float] = None
    contract_size_btc: Optional[float] = None

    # Alternative field names for compatibility
    maker_fee: float = 0.0002  # As decimal
    taker_fee: float = 0.0004  # As decimal

    @property
    def round_trip_taker_bps(self) -> float:
        return (self.taker_fee_bps + self.slippage_bps) * 2

    def calculate_cost(self, notional_usd: float, is_maker: bool = False, n_legs: int = 2) -> Dict[str, float]:
        fee_bps = self.maker_fee_bps if is_maker else self.taker_fee_bps
        pct_cost = (fee_bps + self.slippage_bps) / 10000 * n_legs
        return {
            'pct_cost_usd': notional_usd * pct_cost,
            'gas_cost_usd': self.gas_cost_usd * n_legs,
            'total_cost_usd': notional_usd * pct_cost + self.gas_cost_usd * n_legs,
        }


@dataclass
class TermStructurePoint:
    """Single point on the BTC futures term structure curve."""
    timestamp: pd.Timestamp
    contract: str
    expiry: Optional[pd.Timestamp]
    days_to_expiry: int
    futures_price: float
    spot_price: float
    venue: str
    venue_type: VenueType
    open_interest: Optional[float] = None
    volume_24h: Optional[float] = None
    funding_rate: Optional[float] = None
    
    @property
    def basis_pct(self) -> float:
        if self.spot_price <= 0:
            return 0.0
        return ((self.futures_price - self.spot_price) / self.spot_price) * 100
    
    @property
    def annualized_basis_pct(self) -> float:
        dte = max(self.days_to_expiry, 1)
        if self.is_perpetual:
            dte = 30
        return self.basis_pct * (365 / dte)
    
    @property
    def is_perpetual(self) -> bool:
        return self.expiry is None or self.days_to_expiry <= 0
    
    @property
    def liquidity_score(self) -> float:
        score = 0.5
        if self.volume_24h:
            score = min(self.volume_24h / 100_000_000, 1.0) * 0.5 + 0.5
        return round(score, 3)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp, 'contract': self.contract,
            'days_to_expiry': self.days_to_expiry, 'basis_pct': self.basis_pct,
            'annualized_basis_pct': self.annualized_basis_pct, 'venue': self.venue,
        }


@dataclass
class CalendarSpreadSignal:
    """Signal for calendar spread entry or exit."""
    timestamp: pd.Timestamp
    signal_type: str
    direction: SpreadDirection
    near_contract: str
    far_contract: str
    near_price: float
    far_price: float
    spot_price: float
    near_basis_pct: float
    far_basis_pct: float
    spread_basis_pct: float
    annualized_spread: float
    regime: TermStructureRegime
    venue: str
    signal_strength: float
    reason: str
    near_dte: int = 30
    far_dte: int = 90
    expected_pnl_bps: float = 0.0
    confidence: float = 0.5
    
    @property
    def is_entry(self) -> bool:
        return self.signal_type.lower() == 'entry'
    
    @property
    def is_exit(self) -> bool:
        return self.signal_type.lower() == 'exit'
    
    @property
    def is_actionable(self) -> bool:
        return self.signal_strength >= 0.3 and self.confidence >= 0.4
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp, 'signal_type': self.signal_type,
            'direction': self.direction.name, 'regime': self.regime.value,
            'signal_strength': self.signal_strength, 'expected_pnl_bps': self.expected_pnl_bps,
        }


@dataclass
class CalendarSpreadTrade:
    """Complete lifecycle of a calendar spread trade."""
    trade_id: str
    entry_time: pd.Timestamp
    direction: SpreadDirection
    near_contract: str
    far_contract: str
    venue: str
    venue_type: VenueType
    entry_near_price: float
    entry_far_price: float
    entry_spot_price: float
    entry_basis_pct: float
    entry_spread_pct: float
    entry_near_dte: int
    entry_far_dte: int
    notional_size: float
    leverage: float
    margin_used: float
    exit_time: Optional[pd.Timestamp] = None
    exit_near_price: Optional[float] = None
    exit_far_price: Optional[float] = None
    exit_spread_pct: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    transaction_costs: float = 0.0
    gas_costs: float = 0.0
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0
    mtm_history: List[Dict] = field(default_factory=list)
    
    @property
    def is_open(self) -> bool:
        return self.exit_time is None
    
    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0
    
    @property
    def holding_period_days(self) -> float:
        if self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds() / 86400
        return 0.0
    
    @property
    def return_pct(self) -> float:
        if self.notional_size <= 0:
            return 0.0
        return (self.net_pnl / self.notional_size) * 100
    
    @property
    def total_costs(self) -> float:
        return self.transaction_costs + self.gas_costs
    
    def calculate_pnl(self) -> float:
        if self.exit_spread_pct is None:
            return 0.0
        spread_change = self.exit_spread_pct - self.entry_spread_pct
        if self.direction == SpreadDirection.LONG:
            self.gross_pnl = -spread_change * self.notional_size / 100
        else:
            self.gross_pnl = spread_change * self.notional_size / 100
        self.net_pnl = self.gross_pnl - self.total_costs
        return self.net_pnl
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trade_id': self.trade_id, 'direction': self.direction.name,
            'venue': self.venue, 'entry_time': self.entry_time, 'exit_time': self.exit_time,
            'net_pnl': self.net_pnl, 'return_pct': self.return_pct, 'holding_days': self.holding_period_days,
        }


@dataclass
class CrossVenueOpportunity:
    """Cross-venue basis arbitrage opportunity."""
    timestamp: pd.Timestamp
    venue1: str
    venue2: str
    venue1_type: VenueType
    venue2_type: VenueType
    venue1_basis_pct: float
    venue2_basis_pct: float
    basis_differential: float
    z_score: float
    total_cost_bps: float
    net_differential_bps: float
    max_size_usd: float
    recommended_size_usd: float
    expected_return_bps: float
    annualized_return_pct: float

    @property
    def is_profitable(self) -> bool:
        return self.net_differential_bps > 10.0

    @property
    def direction(self) -> str:
        if self.basis_differential > 0:
            return f"Short {self.venue1}, Long {self.venue2}"
        return f"Long {self.venue1}, Short {self.venue2}"

    @property
    def opportunity_score(self) -> float:
        diff_score = min(self.net_differential_bps / 50, 1.0)
        z_score = min(abs(self.z_score) / 3, 1.0)
        return (diff_score * 0.5 + z_score * 0.5)

    # Backward compatibility properties
    @property
    def venue_long(self) -> str:
        return self.venue2 if self.basis_differential > 0 else self.venue1

    @property
    def venue_short(self) -> str:
        return self.venue1 if self.basis_differential > 0 else self.venue2

    @property
    def spread_bps(self) -> float:
        return abs(self.basis_differential * 100)

    @property
    def confidence(self) -> float:
        return min(self.z_score / 2.0, 1.0)

    @property
    def max_position_usd(self) -> float:
        return self.max_size_usd

    @property
    def execution_cost_bps(self) -> float:
        return self.total_cost_bps


DEFAULT_VENUE_COSTS: Dict[str, VenueCosts] = {
    'binance': VenueCosts('binance', VenueType.CEX_FUTURES, 1.0, 4.0, 1.0, maker_fee=0.0001, taker_fee=0.0004),
    'binance_perp': VenueCosts('binance_perp', VenueType.CEX_PERPETUAL, 1.0, 4.0, 1.0, maker_fee=0.0001, taker_fee=0.0004),
    'cme': VenueCosts('cme', VenueType.CME_FUTURES, 0.5, 1.0, 0.5, per_contract_fee=6.0, contract_size_btc=5.0, maker_fee=0.00005, taker_fee=0.0001),
    'deribit': VenueCosts('deribit', VenueType.CEX_FUTURES, 1.0, 5.0, 1.0, maker_fee=0.0001, taker_fee=0.0005),
    'hyperliquid': VenueCosts('hyperliquid', VenueType.HYBRID_PERPETUAL, 0.0, 2.5, 2.0, gas_cost_usd=0.50, maker_fee=0.0, taker_fee=0.00025),
    'dydx': VenueCosts('dydx', VenueType.HYBRID_PERPETUAL, 0.0, 5.0, 3.0, gas_cost_usd=0.10, maker_fee=0.0, taker_fee=0.0005),
    'gmx': VenueCosts('gmx', VenueType.DEX_PERPETUAL, 0.0, 7.0, 5.0, gas_cost_usd=2.0, maker_fee=0.0, taker_fee=0.0007),
}

DEFAULT_VENUE_CAPACITY: Dict[str, float] = {
    'binance': 75_000_000, 'binance_perp': 100_000_000,  # PDF: $50-100M (very liquid)
    'cme': 300_000_000,                                     # PDF: $100-500M
    'deribit': 50_000_000,                                  # Similar to Binance futures
    'hyperliquid': 7_500_000,                               # PDF: $5-10M (growing but limited)
    'dydx': 3_500_000,                                      # PDF: $2-5M (very limited)
    'gmx': 2_000_000,                                       # DEX - very limited liquidity
}

DEFAULT_CALENDAR_PARAMS: Dict[str, Any] = {
    'long_entry_basis_pct': 15.0, 'short_entry_basis_pct': -10.0,  # PDF Part 2: 15%/-10% entry thresholds
    'long_exit_basis_pct': 5.0, 'short_exit_basis_pct': -5.0,    # PDF Part 2: 5% exit thresholds
    'stop_loss_basis_change_pct': 5.0, 'max_leverage': 2.0,  # PDF: 2.0x max per PDF Section 3.2 (Hyperliquid: 1.5x max)
    'min_days_to_expiry': 7, 'max_holding_days': 90,  # PDF Part 2: roll 3-7 days before expiry
    'min_signal_strength': 0.15,  # Lower threshold for signal generation
    'max_position_pct': 25.0,    # Larger position sizing
}

DEFAULT_CROSS_VENUE_PARAMS: Dict[str, Any] = {
    'min_basis_differential_bps': 15.0,  # PDF Part 2: 15 bps minimum spread threshold
    'min_z_score': 2.0,  # PDF Part 2: 2.0 entry Z-score threshold
    'exit_z_score': 0.5,  # PDF Part 2: 0.5 exit Z-score threshold
    'max_holding_days': 45,  # Longer holding allowed
    'min_liquidity_score': 0.3,  # Accept lower liquidity venues
    'max_single_venue_exposure_pct': 50.0,  # PDF: Binance 50%, others lower
}

__all__ = [
    # Core types and enums
    'TermStructureRegime', 'VenueType', 'SpreadDirection', 'ExitReason',
    'CurveShape', 'InterpolationMethod', 'VenueCosts', 'TermStructurePoint',
    'CalendarSpreadSignal', 'CalendarSpreadTrade', 'CrossVenueOpportunity',
    'DEFAULT_VENUE_COSTS', 'DEFAULT_VENUE_CAPACITY', 'DEFAULT_CALENDAR_PARAMS',
    'DEFAULT_CROSS_VENUE_PARAMS',

    # Term structure analysis
    'TermStructureAnalyzer', 'TermStructureCurve', 'RegimeTracker', 'FundingImpliedCurve',

    # Strategy A & B: Calendar spreads
    'CalendarSpreadStrategy', 'CrossVenueBasisStrategy', 'BacktestResult', 'calculate_kelly_fraction',

    # Strategy C: Synthetic futures from funding
    'SyntheticFuturesStrategy', 'SyntheticFuturesConfig',

    # Strategy D: Roll optimization
    'RollOptimizer', 'MultiVenueRollStrategy', 'RollConfig', 'RollOpportunity',

    # Funding rate analysis
    'FundingRateAnalyzer', 'FundingTermStructure', 'VENUE_FUNDING_CONFIG',
    'CRISIS_EVENTS', 'is_crisis_period', 'FundingRegime',
    'FundingTermStructureIntegration',

    # Multi-venue analysis
    'MultiVenueAnalyzer', 'CrossVenueStrategyB', 'CrossVenueAnalysis',

    # Orchestrators
    'FuturesOrchestrator', 'OrchestratorConfig',
    'Step5FuturesOrchestrator', 'ReportFormat',

    # Walk-forward optimization
    'WalkForwardOptimizer', 'WalkForwardResult', 'run_full_walk_forward',
    'generate_walk_forward_report', 'detect_crisis_in_window',
    'CRISIS_PARAM_ADJUSTMENTS', 'REGIME_PARAM_ADJUSTMENTS',

    # Backtesting
    'FuturesBacktestEngine', 'BacktestMetrics',

    # Main runner
    'Phase3Runner', 'Phase3Config', 'Phase3Results', 'run_phase3',

    # Computation acceleration
    'fast_nelson_siegel_fit', 'fast_nelson_siegel_interpolate',
    'FastFundingAnalyzer', 'FastFundingTermStructure',
    'FastMultiVenueAnalyzer', 'FastTermStructureAnalyzer',
    'FastRollOptimizer', 'FastBacktestMetrics', 'EnhancedBacktestMetrics',
    'ParallelWalkForwardOptimizer', 'EnhancedWalkForwardOptimizer',
    'ParallelStrategyRunner',
    'TTLCache', 'DiskBackedCache', 'cached_with_ttl', 'clear_all_caches',
    'fast_classify_regime', 'batch_classify_regimes',
    'get_optimization_info', 'benchmark_phase3', 'benchmark_phase3_comprehensive',
    'auto_integrate_all',

    # Progress tracking
    'Phase3ProgressTracker', 'get_phase3_tracker',
    'ProgressStatus', 'ProgressPhase', 'ProgressMetrics', 'ProgressTask',
]

def __getattr__(name: str):
    # Term structure module
    if name in ['TermStructureAnalyzer', 'TermStructureCurve', 'RegimeTracker', 'FundingImpliedCurve']:
        from .term_structure import TermStructureAnalyzer, TermStructureCurve, RegimeTracker, FundingImpliedCurve
        return locals()[name]

    # Calendar spreads module
    if name in ['CalendarSpreadStrategy', 'CrossVenueBasisStrategy', 'BacktestResult', 'calculate_kelly_fraction']:
        from .calendar_spreads import CalendarSpreadStrategy, CrossVenueBasisStrategy, BacktestResult, calculate_kelly_fraction
        return locals()[name]

    # Synthetic futures module (Strategy C)
    if name in ['SyntheticFuturesStrategy', 'SyntheticFuturesConfig']:
        from .synthetic_futures import SyntheticFuturesStrategy, SyntheticFuturesConfig
        return locals()[name]

    # Roll optimization module (Strategy D)
    if name in ['RollOptimizer', 'MultiVenueRollStrategy', 'RollConfig', 'RollOpportunity']:
        from .roll_optimization import RollOptimizer, MultiVenueRollStrategy, RollConfig, RollOpportunity
        return locals()[name]

    # Funding rate analysis module
    if name in ['FundingRateAnalyzer', 'FundingTermStructure', 'VENUE_FUNDING_CONFIG',
                'CRISIS_EVENTS', 'is_crisis_period', 'FundingRegime',
                'FundingTermStructureIntegration']:
        from .funding_rate_analysis import (
            FundingRateAnalyzer, FundingTermStructure, VENUE_FUNDING_CONFIG,
            CRISIS_EVENTS, is_crisis_period, FundingRegime,
            FundingTermStructureIntegration
        )
        return locals()[name]

    # Multi-venue analyzer module
    if name in ['MultiVenueAnalyzer', 'CrossVenueStrategyB', 'CrossVenueAnalysis']:
        from .multi_venue_analyzer import MultiVenueAnalyzer, CrossVenueStrategyB, CrossVenueAnalysis
        return locals()[name]

    # Step 4 orchestrator
    if name in ['FuturesOrchestrator', 'OrchestratorConfig']:
        from .step4_futures_orchestrator import FuturesOrchestrator, OrchestratorConfig
        return locals()[name]

    # Walk-forward optimization
    if name in ['WalkForwardOptimizer', 'WalkForwardResult', 'run_full_walk_forward',
                'generate_walk_forward_report', 'detect_crisis_in_window',
                'CRISIS_PARAM_ADJUSTMENTS', 'REGIME_PARAM_ADJUSTMENTS']:
        from .futures_walk_forward import (
            WalkForwardOptimizer, WalkForwardResult, run_full_walk_forward,
            generate_walk_forward_report, detect_crisis_in_window,
            CRISIS_PARAM_ADJUSTMENTS, REGIME_PARAM_ADJUSTMENTS
        )
        return locals()[name]

    # Backtest engine
    if name in ['FuturesBacktestEngine', 'BacktestMetrics']:
        from .futures_backtest_engine import FuturesBacktestEngine, BacktestMetrics
        return locals()[name]

    # Step 5 reporting orchestrator
    if name in ['Step5FuturesOrchestrator', 'ReportFormat']:
        from .step5_futures_orchestrator import Step5FuturesOrchestrator, ReportFormat
        return locals()[name]

    # Phase 3 runner
    if name in ['Phase3Runner', 'Phase3Config', 'Phase3Results', 'run_phase3']:
        from .phase3run import Phase3Runner, Phase3Config, Phase3Results, run_phase3
        return locals()[name]

    # Fast optimizations (GPU/CPU acceleration)
    if name in ['fast_nelson_siegel_fit', 'fast_nelson_siegel_interpolate',
                'FastFundingAnalyzer', 'FastFundingTermStructure',
                'FastMultiVenueAnalyzer', 'FastTermStructureAnalyzer',
                'FastRollOptimizer', 'FastBacktestMetrics', 'EnhancedBacktestMetrics',
                'ParallelWalkForwardOptimizer', 'EnhancedWalkForwardOptimizer',
                'ParallelStrategyRunner',
                'TTLCache', 'DiskBackedCache', 'cached_with_ttl', 'clear_all_caches',
                'fast_classify_regime', 'batch_classify_regimes',
                'get_optimization_info', 'benchmark_phase3', 'benchmark_phase3_comprehensive',
                'auto_integrate_all']:
        from .fast_futures_core import (
            fast_nelson_siegel_fit, fast_nelson_siegel_interpolate,
            FastFundingAnalyzer, FastFundingTermStructure,
            FastMultiVenueAnalyzer, FastTermStructureAnalyzer,
            FastRollOptimizer, FastBacktestMetrics, EnhancedBacktestMetrics,
            ParallelWalkForwardOptimizer, EnhancedWalkForwardOptimizer,
            ParallelStrategyRunner,
            TTLCache, DiskBackedCache, cached_with_ttl, clear_all_caches,
            fast_classify_regime, batch_classify_regimes,
            get_optimization_info, benchmark_phase3, benchmark_phase3_comprehensive,
            auto_integrate_all
        )
        return locals()[name]

    # Progress tracking
    if name in ['Phase3ProgressTracker', 'get_phase3_tracker',
                'ProgressStatus', 'ProgressPhase', 'ProgressMetrics', 'ProgressTask']:
        from ..progress_tracker import (
            Phase3ProgressTracker, get_phase3_tracker,
            ProgressStatus, ProgressPhase, ProgressMetrics, ProgressTask
        )
        return locals()[name]

    raise AttributeError(f"module 'futures_curve' has no attribute '{name}'")