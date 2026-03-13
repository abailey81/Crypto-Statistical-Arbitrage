"""
Funding Rate Term Structure Analysis
====================================

Funding rate analysis for constructing synthetic term structures
from perpetual funding rates across multiple venues.

Part 2 Requirements Addressed:
- 3.1.2 Funding rate normalization (hourly vs 8-hour intervals)
- 3.1.3 Synthetic term structure from perpetual funding rates
- 3.2.3 Strategy C: Synthetic Futures from Perp Funding integration
- Cross-venue funding arbitrage detection

Mathematical Framework
----------------------
Funding Rate Annualization:

    For hourly venues (Hyperliquid, dYdX V4, GMX):
        Annual_Rate = hourly_rate × 8760
        8h_Equivalent = hourly_rate × 8

    For 8-hour venues (Binance, Bybit, OKX):
        Annual_Rate = 8h_rate × 1095

    For Deribit (8-hour perpetual funding):
        Annual_Rate = 8h_rate × 1095

Synthetic Term Structure:

    Implied_Price(T) = Spot × (1 + Funding_Annual × T/365)

    Where T is days to target maturity

Funding Premium Analysis:

    Premium = Venue1_Annual - Venue2_Annual
    Z_Score = (Premium - μ) / σ

    Arbitrage when |Z_Score| > 2.0 and costs covered

Cross-Venue Funding Spread:

    Spread(V1, V2) = Funding_V1 - Funding_V2
    Carry_Return = Spread × Position_Days / 365

Venue-Specific Normalization (per PDF venues):
- CEX: Binance (8h), Deribit (8h)
- Hybrid: Hyperliquid (1h), dYdX V4 (1h)
- DEX: GMX (1h continuous)
- Note: CME has no perpetuals (dated futures only)

Version: 3.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from collections import deque
from scipy import stats
from scipy.optimize import minimize

from . import (
    TermStructureRegime, VenueType, VenueCosts,
    DEFAULT_VENUE_COSTS, DEFAULT_VENUE_CAPACITY
)

logger = logging.getLogger(__name__)


# =============================================================================
# CRISIS EVENT DEFINITIONS (per PDF Section 3.3.3)
# =============================================================================

CRISIS_EVENTS = {
    'covid_crash': {
        'start': pd.Timestamp('2020-03-12', tz='UTC'),
        'end': pd.Timestamp('2020-03-20', tz='UTC'),
        'severity': 1.0,
        'name': 'COVID-19 Market Crash',
        'expected_funding_behavior': 'extremely_negative',  # Panic liquidations
    },
    'may_2021_crash': {
        'start': pd.Timestamp('2021-05-19', tz='UTC'),
        'end': pd.Timestamp('2021-05-25', tz='UTC'),
        'severity': 0.8,
        'name': 'May 2021 BTC Crash',
        'expected_funding_behavior': 'highly_negative',
    },
    'luna_collapse': {
        'start': pd.Timestamp('2022-05-09', tz='UTC'),
        'end': pd.Timestamp('2022-05-15', tz='UTC'),
        'severity': 0.9,
        'name': 'Terra/LUNA Collapse',
        'expected_funding_behavior': 'extremely_negative',
    },
    'ftx_collapse': {
        'start': pd.Timestamp('2022-11-08', tz='UTC'),
        'end': pd.Timestamp('2022-11-15', tz='UTC'),
        'severity': 0.95,
        'name': 'FTX Exchange Collapse',
        'expected_funding_behavior': 'extremely_negative',
    },
    '3ac_liquidation': {
        'start': pd.Timestamp('2022-06-13', tz='UTC'),
        'end': pd.Timestamp('2022-06-20', tz='UTC'),
        'severity': 0.7,
        'name': '3AC Liquidation Event',
        'expected_funding_behavior': 'highly_negative',
    },
}


def is_crisis_period(timestamp: pd.Timestamp) -> Tuple[bool, Optional[str], float]:
    """
    Check if timestamp falls within a known crisis period.

    Returns:
        Tuple of (is_crisis, crisis_name, severity)
    """
    # Ensure timestamp is timezone-aware (UTC) for comparison
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize('UTC')
    for crisis_id, crisis in CRISIS_EVENTS.items():
        if crisis['start'] <= timestamp <= crisis['end']:
            return True, crisis['name'], crisis['severity']
    return False, None, 0.0


# =============================================================================
# FUNDING RATE ENUMERATIONS
# =============================================================================

class FundingInterval(Enum):
    """Funding rate payment interval by venue type."""
    HOURLY = 1
    FOUR_HOURLY = 4
    EIGHT_HOURLY = 8

    @property
    def periods_per_day(self) -> int:
        return 24 // self.value

    @property
    def periods_per_year(self) -> int:
        return self.periods_per_day * 365

    def annualize(self, rate: float) -> float:
        """Convert single-period rate to annualized percentage."""
        return rate * self.periods_per_year * 100

    def to_8h_equivalent(self, rate: float) -> float:
        """Convert to 8-hour equivalent for cross-venue comparison."""
        if self == FundingInterval.HOURLY:
            return rate * 8
        elif self == FundingInterval.FOUR_HOURLY:
            return rate * 2
        return rate


class FundingRegime(Enum):
    """Funding rate regime classification."""
    EXTREMELY_POSITIVE = "extremely_positive"  # >50% annualized
    HIGHLY_POSITIVE = "highly_positive"        # 20-50%
    MODERATELY_POSITIVE = "moderately_positive"  # 5-20%
    NEUTRAL = "neutral"                         # -5% to 5%
    MODERATELY_NEGATIVE = "moderately_negative"  # -20% to -5%
    HIGHLY_NEGATIVE = "highly_negative"         # -50% to -20%
    EXTREMELY_NEGATIVE = "extremely_negative"   # <-50%

    @classmethod
    def from_annual_rate(cls, annual_pct: float) -> 'FundingRegime':
        """Classify regime from annualized funding rate."""
        if annual_pct > 50:
            return cls.EXTREMELY_POSITIVE
        elif annual_pct > 20:
            return cls.HIGHLY_POSITIVE
        elif annual_pct > 5:
            return cls.MODERATELY_POSITIVE
        elif annual_pct > -5:
            return cls.NEUTRAL
        elif annual_pct > -20:
            return cls.MODERATELY_NEGATIVE
        elif annual_pct > -50:
            return cls.HIGHLY_NEGATIVE
        return cls.EXTREMELY_NEGATIVE

    @property
    def is_positive(self) -> bool:
        return self in [
            self.EXTREMELY_POSITIVE,
            self.HIGHLY_POSITIVE,
            self.MODERATELY_POSITIVE
        ]

    @property
    def is_negative(self) -> bool:
        return self in [
            self.EXTREMELY_NEGATIVE,
            self.HIGHLY_NEGATIVE,
            self.MODERATELY_NEGATIVE
        ]

    @property
    def is_extreme(self) -> bool:
        return self in [self.EXTREMELY_POSITIVE, self.EXTREMELY_NEGATIVE]

    @property
    def position_bias(self) -> int:
        """Suggested position bias: +1 receive funding, -1 pay funding, 0 neutral."""
        biases = {
            self.EXTREMELY_POSITIVE: -1,  # Short perp (receive funding)
            self.HIGHLY_POSITIVE: -1,
            self.MODERATELY_POSITIVE: 0,
            self.NEUTRAL: 0,
            self.MODERATELY_NEGATIVE: 0,
            self.HIGHLY_NEGATIVE: 1,      # Long perp (receive funding)
            self.EXTREMELY_NEGATIVE: 1,
        }
        return biases.get(self, 0)

    @property
    def expected_reversion_days(self) -> int:
        """Expected days for funding to revert to neutral."""
        days = {
            self.EXTREMELY_POSITIVE: 7,
            self.HIGHLY_POSITIVE: 14,
            self.MODERATELY_POSITIVE: 30,
            self.NEUTRAL: 60,
            self.MODERATELY_NEGATIVE: 30,
            self.HIGHLY_NEGATIVE: 14,
            self.EXTREMELY_NEGATIVE: 7,
        }
        return days.get(self, 30)


class FundingArbitrageType(Enum):
    """Type of funding rate arbitrage opportunity."""
    CASH_AND_CARRY = "cash_and_carry"           # Long spot, short perp
    REVERSE_CASH_AND_CARRY = "reverse_cash_carry"  # Short spot, long perp
    CROSS_VENUE_LONG = "cross_venue_long"       # Long low funding, short high
    CROSS_VENUE_SHORT = "cross_venue_short"     # Reverse
    FUNDING_RATE_HARVEST = "funding_harvest"    # Pure funding collection

    @property
    def requires_spot(self) -> bool:
        return self in [self.CASH_AND_CARRY, self.REVERSE_CASH_AND_CARRY]

    @property
    def is_delta_neutral(self) -> bool:
        return True  # All are delta neutral by design


# =============================================================================
# FUNDING RATE DATA STRUCTURES
# =============================================================================

@dataclass
class VenueFundingConfig:
    """Configuration for a venue's funding rate mechanics."""
    venue: str
    venue_type: VenueType
    interval: FundingInterval
    max_rate_per_period: float = 0.01  # 1% cap per period
    min_rate_per_period: float = -0.01
    has_insurance_fund: bool = True
    settlement_currency: str = "USDT"

    @property
    def annualization_factor(self) -> int:
        return self.interval.periods_per_year

    def normalize_rate(self, rate: float) -> float:
        """Normalize to 8-hour equivalent."""
        return self.interval.to_8h_equivalent(rate)

    def annualize_rate(self, rate: float) -> float:
        """Convert single period rate to annual percentage."""
        return self.interval.annualize(rate)


# Default venue funding configurations (all venues per PDF Section 3)
VENUE_FUNDING_CONFIG: Dict[str, VenueFundingConfig] = {
    # ==========================================================================
    # CEX VENUES
    # ==========================================================================
    'binance': VenueFundingConfig(
        venue='binance',
        venue_type=VenueType.CEX_PERPETUAL,
        interval=FundingInterval.EIGHT_HOURLY,
        max_rate_per_period=0.03,
        min_rate_per_period=-0.03,
        settlement_currency='USDT',
    ),
    'bybit': VenueFundingConfig(
        venue='bybit',
        venue_type=VenueType.CEX_PERPETUAL,
        interval=FundingInterval.EIGHT_HOURLY,
        max_rate_per_period=0.01,
        min_rate_per_period=-0.01,
        settlement_currency='USDT',
    ),
    'okx': VenueFundingConfig(
        venue='okx',
        venue_type=VenueType.CEX_PERPETUAL,
        interval=FundingInterval.EIGHT_HOURLY,
        max_rate_per_period=0.015,
        min_rate_per_period=-0.015,
        settlement_currency='USDT',
    ),
    'deribit': VenueFundingConfig(
        venue='deribit',
        venue_type=VenueType.CEX_PERPETUAL,  # Deribit also has perpetual
        interval=FundingInterval.EIGHT_HOURLY,
        max_rate_per_period=0.01,
        min_rate_per_period=-0.01,
        settlement_currency='USD',  # Settled in USD-margined
    ),

    # ==========================================================================
    # HYBRID VENUES (On-chain settlement, off-chain matching)
    # ==========================================================================
    'hyperliquid': VenueFundingConfig(
        venue='hyperliquid',
        venue_type=VenueType.HYBRID_PERPETUAL,
        interval=FundingInterval.HOURLY,
        max_rate_per_period=0.00125,  # 1.25% per hour max
        min_rate_per_period=-0.00125,
        settlement_currency='USDC',
    ),
    'dydx': VenueFundingConfig(
        venue='dydx',
        venue_type=VenueType.HYBRID_PERPETUAL,
        interval=FundingInterval.HOURLY,
        max_rate_per_period=0.001,  # 0.1% per hour
        min_rate_per_period=-0.001,
        settlement_currency='USDC',
    ),
    'vertex': VenueFundingConfig(
        venue='vertex',
        venue_type=VenueType.HYBRID_PERPETUAL,
        interval=FundingInterval.HOURLY,
        max_rate_per_period=0.001,
        min_rate_per_period=-0.001,
        settlement_currency='USDC',
    ),

    # ==========================================================================
    # DEX VENUES (Fully on-chain)
    # ==========================================================================
    'gmx': VenueFundingConfig(
        venue='gmx',
        venue_type=VenueType.DEX_PERPETUAL,
        interval=FundingInterval.HOURLY,
        max_rate_per_period=0.002,  # Higher volatility on DEX
        min_rate_per_period=-0.002,
        has_insurance_fund=False,  # Uses GLP pool
        settlement_currency='USDC',
    ),
}

# =============================================================================
# VENUE NORMALIZATION MATRIX
# =============================================================================

# For converting funding rates between venues with different intervals
FUNDING_NORMALIZATION_MATRIX = {
    # (from_interval, to_interval): multiplier
    (FundingInterval.HOURLY, FundingInterval.EIGHT_HOURLY): 8.0,
    (FundingInterval.HOURLY, FundingInterval.FOUR_HOURLY): 4.0,
    (FundingInterval.FOUR_HOURLY, FundingInterval.EIGHT_HOURLY): 2.0,
    (FundingInterval.FOUR_HOURLY, FundingInterval.HOURLY): 0.25,
    (FundingInterval.EIGHT_HOURLY, FundingInterval.HOURLY): 0.125,
    (FundingInterval.EIGHT_HOURLY, FundingInterval.FOUR_HOURLY): 0.5,
}


def normalize_funding_rate_cross_venue(
    rate: float,
    from_venue: str,
    to_venue: str
) -> float:
    """
    Normalize funding rate from one venue's interval to another.

    This ensures accurate cross-venue comparison by standardizing
    the funding rate period.

    Args:
        rate: Raw funding rate from source venue
        from_venue: Source venue name
        to_venue: Target venue name for normalization

    Returns:
        Normalized funding rate in target venue's interval
    """
    from_config = VENUE_FUNDING_CONFIG.get(from_venue.lower())
    to_config = VENUE_FUNDING_CONFIG.get(to_venue.lower())

    if not from_config or not to_config:
        return rate

    if from_config.interval == to_config.interval:
        return rate

    key = (from_config.interval, to_config.interval)
    multiplier = FUNDING_NORMALIZATION_MATRIX.get(key, 1.0)

    return rate * multiplier


def get_annualized_funding(rate: float, venue: str) -> float:
    """
    Get annualized funding rate percentage for any venue.

    Args:
        rate: Single-period funding rate
        venue: Venue name

    Returns:
        Annualized funding rate as percentage
    """
    config = VENUE_FUNDING_CONFIG.get(venue.lower())
    if not config:
        # Default to 8-hour assumption
        return rate * 1095 * 100

    return config.annualize_rate(rate)


def get_8h_equivalent_funding(rate: float, venue: str) -> float:
    """
    Convert any venue's funding rate to 8-hour equivalent.

    This is the standard for cross-venue comparison.

    Args:
        rate: Single-period funding rate
        venue: Venue name

    Returns:
        8-hour equivalent funding rate
    """
    config = VENUE_FUNDING_CONFIG.get(venue.lower())
    if not config:
        return rate

    return config.normalize_rate(rate)


@dataclass
class FundingRateSnapshot:
    """Single funding rate observation."""
    timestamp: pd.Timestamp
    venue: str
    symbol: str
    funding_rate: float  # Raw rate for single period
    funding_rate_8h: float  # Normalized to 8h
    funding_rate_annual_pct: float  # Annualized percentage
    mark_price: float
    index_price: float
    next_funding_time: Optional[pd.Timestamp] = None
    open_interest: Optional[float] = None
    predicted_rate: Optional[float] = None

    @property
    def basis_pct(self) -> float:
        """Mark to index basis."""
        if self.index_price <= 0:
            return 0.0
        return ((self.mark_price - self.index_price) / self.index_price) * 100

    @property
    def regime(self) -> FundingRegime:
        return FundingRegime.from_annual_rate(self.funding_rate_annual_pct)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'venue': self.venue,
            'symbol': self.symbol,
            'funding_rate': self.funding_rate,
            'funding_rate_8h': self.funding_rate_8h,
            'funding_rate_annual_pct': self.funding_rate_annual_pct,
            'mark_price': self.mark_price,
            'basis_pct': self.basis_pct,
            'regime': self.regime.value,
        }


@dataclass
class FundingImpliedPoint:
    """Single point on funding-implied term structure."""
    days_to_expiry: int
    implied_price: float
    implied_basis_pct: float
    annualized_basis_pct: float
    funding_rate_used: float
    confidence: float = 1.0


@dataclass
class FundingTermStructure:
    """Complete funding-implied term structure for a venue."""
    timestamp: pd.Timestamp
    venue: str
    venue_type: VenueType
    spot_price: float
    current_funding_rate: float
    avg_funding_rate_7d: float
    avg_funding_rate_30d: float
    funding_volatility: float
    annual_funding_pct: float
    implied_points: List[FundingImpliedPoint] = field(default_factory=list)
    regime: FundingRegime = field(default=FundingRegime.NEUTRAL)

    def __post_init__(self):
        """Generate implied curve and classify regime."""
        self.regime = FundingRegime.from_annual_rate(self.annual_funding_pct)
        if not self.implied_points:
            self._generate_implied_curve()

    def _generate_implied_curve(self):
        """Generate implied prices for standard tenors."""
        tenors = [1, 3, 7, 14, 30, 60, 90, 180, 365]

        for dte in tenors:
            # Blend funding rates based on tenor
            if dte <= 7:
                # Use recent funding for short tenors
                funding = self.current_funding_rate
                confidence = 0.9
            elif dte <= 30:
                # Blend recent and 7d average
                weight = (dte - 7) / 23
                funding = (1 - weight) * self.current_funding_rate + weight * self.avg_funding_rate_7d
                confidence = 0.7
            else:
                # Use 30d average for longer tenors
                funding = self.avg_funding_rate_30d
                confidence = 0.5 - (dte - 30) / 700  # Decreasing confidence

            # Calculate implied price
            annual_rate = funding * VENUE_FUNDING_CONFIG.get(
                self.venue.lower(),
                VENUE_FUNDING_CONFIG['binance']
            ).annualization_factor

            implied_price = self.spot_price * (1 + annual_rate * dte / 365)
            implied_basis = ((implied_price - self.spot_price) / self.spot_price) * 100
            annualized_basis = implied_basis * (365 / dte)

            self.implied_points.append(FundingImpliedPoint(
                days_to_expiry=dte,
                implied_price=implied_price,
                implied_basis_pct=implied_basis,
                annualized_basis_pct=annualized_basis,
                funding_rate_used=funding,
                confidence=max(0.2, confidence),
            ))

    def get_implied_price(self, dte: int) -> float:
        """Get implied price for specific DTE."""
        # Find bracketing points
        for i, point in enumerate(self.implied_points):
            if point.days_to_expiry == dte:
                return point.implied_price
            if point.days_to_expiry > dte:
                if i == 0:
                    return point.implied_price
                prev = self.implied_points[i-1]
                # Linear interpolation
                weight = (dte - prev.days_to_expiry) / (point.days_to_expiry - prev.days_to_expiry)
                return prev.implied_price + weight * (point.implied_price - prev.implied_price)

        # Extrapolate for longer tenors
        if self.implied_points:
            last = self.implied_points[-1]
            annual_rate = self.avg_funding_rate_30d * VENUE_FUNDING_CONFIG.get(
                self.venue.lower(), VENUE_FUNDING_CONFIG['binance']
            ).annualization_factor
            return self.spot_price * (1 + annual_rate * dte / 365)

        return self.spot_price

    def get_implied_basis(self, dte: int) -> float:
        """Get implied annualized basis for specific DTE."""
        implied = self.get_implied_price(dte)
        basis = ((implied - self.spot_price) / self.spot_price) * 100
        return basis * (365 / max(dte, 1))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame([
            {
                'dte': p.days_to_expiry,
                'implied_price': p.implied_price,
                'implied_basis_pct': p.implied_basis_pct,
                'annualized_basis_pct': p.annualized_basis_pct,
                'funding_rate_used': p.funding_rate_used,
                'confidence': p.confidence,
            }
            for p in self.implied_points
        ])

    def summary(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'venue': self.venue,
            'spot_price': self.spot_price,
            'current_funding_pct': self.current_funding_rate * 100,
            'annual_funding_pct': self.annual_funding_pct,
            'regime': self.regime.value,
            'funding_volatility': self.funding_volatility,
            'n_implied_points': len(self.implied_points),
        }


@dataclass
class CrossVenueFundingSpread:
    """Funding rate spread between two venues."""
    timestamp: pd.Timestamp
    venue_long: str  # Venue where we're long (pay funding if positive)
    venue_short: str  # Venue where we're short (receive funding if positive)
    venue_long_type: VenueType
    venue_short_type: VenueType
    funding_long: float  # Annual %
    funding_short: float  # Annual %
    spread_annual_pct: float  # funding_short - funding_long (positive = profitable)
    z_score: float
    total_cost_bps: float
    net_spread_bps: float
    recommended_direction: str
    expected_daily_return_bps: float
    confidence: float

    @property
    def is_profitable(self) -> bool:
        return self.net_spread_bps > 5.0

    @property
    def is_significant(self) -> bool:
        return abs(self.z_score) > 1.5

    @property
    def attractiveness_score(self) -> float:
        """Score 0-1 based on spread magnitude and significance."""
        spread_score = min(abs(self.net_spread_bps) / 50, 1.0)
        z_score_contrib = min(abs(self.z_score) / 3, 1.0)
        return (spread_score * 0.6 + z_score_contrib * 0.4) * self.confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'venue_long': self.venue_long,
            'venue_short': self.venue_short,
            'spread_annual_pct': self.spread_annual_pct,
            'z_score': self.z_score,
            'net_spread_bps': self.net_spread_bps,
            'is_profitable': self.is_profitable,
            'attractiveness_score': self.attractiveness_score,
        }


@dataclass
class FundingArbitrageOpportunity:
    """Complete funding rate arbitrage opportunity."""
    timestamp: pd.Timestamp
    arb_type: FundingArbitrageType
    venue_long: str
    venue_short: str
    symbol: str
    spot_price: float
    funding_long_annual_pct: float
    funding_short_annual_pct: float
    gross_spread_annual_pct: float
    venue_long_costs_bps: float
    venue_short_costs_bps: float
    total_costs_bps: float
    net_spread_annual_pct: float
    expected_daily_pnl_bps: float
    max_position_usd: float
    recommended_position_usd: float
    expected_holding_days: int
    confidence: float
    risk_score: float  # 0-1, higher = more risk

    @property
    def is_actionable(self) -> bool:
        return (
            self.net_spread_annual_pct > 5.0 and
            self.confidence > 0.5 and
            self.risk_score < 0.7
        )

    @property
    def expected_total_return_pct(self) -> float:
        """Expected return over holding period."""
        return self.net_spread_annual_pct * self.expected_holding_days / 365

    @property
    def sharpe_estimate(self) -> float:
        """Rough Sharpe ratio estimate."""
        if self.risk_score <= 0:
            return 0.0
        # Assume daily vol = risk_score * 2%
        daily_vol = self.risk_score * 0.02
        daily_return = self.expected_daily_pnl_bps / 10000
        if daily_vol <= 0:
            return 0.0
        return (daily_return / daily_vol) * np.sqrt(365)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'arb_type': self.arb_type.value,
            'venue_long': self.venue_long,
            'venue_short': self.venue_short,
            'gross_spread_annual_pct': round(self.gross_spread_annual_pct, 2),
            'net_spread_annual_pct': round(self.net_spread_annual_pct, 2),
            'expected_daily_pnl_bps': round(self.expected_daily_pnl_bps, 2),
            'recommended_position_usd': self.recommended_position_usd,
            'is_actionable': self.is_actionable,
            'confidence': self.confidence,
        }


# =============================================================================
# FUNDING RATE ANALYZER
# =============================================================================

class FundingRateAnalyzer:
    """
    Multi-venue funding rate analysis and term structure construction.

    Provides:
    - Funding rate normalization across venues
    - Synthetic term structure from funding
    - Cross-venue spread analysis
    - Arbitrage opportunity detection
    - Historical funding regime tracking
    """

    def __init__(
        self,
        venues: Optional[List[str]] = None,
        lookback_days: int = 30,
        min_spread_bps: float = 10.0,
        min_z_score: float = 1.5,
    ):
        """
        Initialize analyzer.

        Args:
            venues: List of venues to analyze
            lookback_days: Lookback for statistics
            min_spread_bps: Minimum spread for opportunities
            min_z_score: Minimum z-score for significance
        """
        self.venues = venues or list(VENUE_FUNDING_CONFIG.keys())
        self.lookback_days = lookback_days
        self.min_spread_bps = min_spread_bps
        self.min_z_score = min_z_score

        # History tracking
        self._funding_history: Dict[str, deque] = {
            v: deque(maxlen=lookback_days * 24) for v in self.venues
        }
        self._spread_history: Dict[Tuple[str, str], deque] = {}
        self._regime_history: Dict[str, List[Tuple[pd.Timestamp, FundingRegime]]] = {}

        logger.info(f"FundingRateAnalyzer initialized: venues={self.venues}")

    def normalize_funding_rate(
        self,
        rate: float,
        venue: str,
        to_interval: FundingInterval = FundingInterval.EIGHT_HOURLY
    ) -> float:
        """
        Normalize funding rate to target interval.

        Args:
            rate: Raw funding rate
            venue: Source venue
            to_interval: Target interval

        Returns:
            Normalized rate
        """
        config = VENUE_FUNDING_CONFIG.get(venue.lower())
        if not config:
            logger.warning(f"Unknown venue {venue}, using default normalization")
            return rate

        # Convert to hourly first
        hourly = rate / config.interval.value

        # Then to target interval
        return hourly * to_interval.value

    def annualize_funding_rate(self, rate: float, venue: str) -> float:
        """
        Annualize funding rate from single period.

        Args:
            rate: Raw funding rate for single period
            venue: Venue name

        Returns:
            Annualized percentage
        """
        config = VENUE_FUNDING_CONFIG.get(venue.lower())
        if not config:
            # Default to 8-hour assumption
            return rate * 1095 * 100

        return config.annualize_rate(rate)

    def process_funding_snapshot(
        self,
        snapshot: FundingRateSnapshot
    ) -> FundingRateSnapshot:
        """
        Process and normalize a funding rate snapshot.

        Args:
            snapshot: Raw snapshot

        Returns:
            Processed snapshot with normalized rates
        """
        venue = snapshot.venue.lower()
        config = VENUE_FUNDING_CONFIG.get(venue)

        if config:
            snapshot.funding_rate_8h = config.normalize_rate(snapshot.funding_rate)
            snapshot.funding_rate_annual_pct = config.annualize_rate(snapshot.funding_rate)
        else:
            # Default assumptions
            snapshot.funding_rate_8h = snapshot.funding_rate
            snapshot.funding_rate_annual_pct = snapshot.funding_rate * 1095 * 100

        # Update history
        if venue in self._funding_history:
            self._funding_history[venue].append(snapshot)

        # Track regime
        regime = snapshot.regime
        if venue not in self._regime_history:
            self._regime_history[venue] = []
        self._regime_history[venue].append((snapshot.timestamp, regime))

        return snapshot

    def build_funding_term_structure(
        self,
        funding_history: pd.DataFrame,
        venue: str,
        spot_price: float,
        timestamp: pd.Timestamp
    ) -> FundingTermStructure:
        """
        Build funding-implied term structure from historical data.

        Args:
            funding_history: DataFrame with funding_rate column
            venue: Venue name
            spot_price: Current spot price
            timestamp: As-of timestamp

        Returns:
            FundingTermStructure object
        """
        if funding_history.empty:
            return FundingTermStructure(
                timestamp=timestamp,
                venue=venue,
                venue_type=VENUE_FUNDING_CONFIG.get(
                    venue.lower(), VENUE_FUNDING_CONFIG['binance']
                ).venue_type,
                spot_price=spot_price,
                current_funding_rate=0.0,
                avg_funding_rate_7d=0.0,
                avg_funding_rate_30d=0.0,
                funding_volatility=0.0,
                annual_funding_pct=0.0,
            )

        config = VENUE_FUNDING_CONFIG.get(venue.lower(), VENUE_FUNDING_CONFIG['binance'])

        # Get rates
        rates = funding_history['funding_rate'].dropna()

        current_rate = rates.iloc[-1] if len(rates) > 0 else 0.0

        # Calculate averages (assuming data is sorted by time)
        n_7d = min(len(rates), config.interval.periods_per_day * 7)
        n_30d = min(len(rates), config.interval.periods_per_day * 30)

        avg_7d = rates.tail(n_7d).mean() if n_7d > 0 else current_rate
        avg_30d = rates.tail(n_30d).mean() if n_30d > 0 else current_rate

        volatility = rates.tail(n_30d).std() if n_30d > 5 else 0.0

        annual_pct = config.annualize_rate(avg_30d)

        return FundingTermStructure(
            timestamp=timestamp,
            venue=venue,
            venue_type=config.venue_type,
            spot_price=spot_price,
            current_funding_rate=current_rate,
            avg_funding_rate_7d=avg_7d,
            avg_funding_rate_30d=avg_30d,
            funding_volatility=volatility,
            annual_funding_pct=annual_pct,
        )

    def calculate_cross_venue_spread(
        self,
        ts1: FundingTermStructure,
        ts2: FundingTermStructure,
    ) -> CrossVenueFundingSpread:
        """
        Calculate funding spread between two venues.

        Args:
            ts1: First venue term structure
            ts2: Second venue term structure

        Returns:
            CrossVenueFundingSpread object
        """
        # Determine which venue to be long/short
        # We want to receive funding on net
        spread = ts2.annual_funding_pct - ts1.annual_funding_pct

        if spread > 0:
            # ts2 has higher funding, short ts2 (receive), long ts1 (pay less)
            venue_long = ts1.venue
            venue_short = ts2.venue
            funding_long = ts1.annual_funding_pct
            funding_short = ts2.annual_funding_pct
        else:
            # ts1 has higher funding
            venue_long = ts2.venue
            venue_short = ts1.venue
            funding_long = ts2.annual_funding_pct
            funding_short = ts1.annual_funding_pct
            spread = -spread

        # Get costs
        costs1 = DEFAULT_VENUE_COSTS.get(ts1.venue.lower())
        costs2 = DEFAULT_VENUE_COSTS.get(ts2.venue.lower())

        cost1_bps = costs1.round_trip_taker_bps if costs1 else 20.0
        cost2_bps = costs2.round_trip_taker_bps if costs2 else 20.0
        total_cost_bps = cost1_bps + cost2_bps

        spread_bps = spread * 100  # Convert to bps
        net_spread_bps = spread_bps - total_cost_bps

        # Calculate z-score from history
        pair_key = tuple(sorted([ts1.venue, ts2.venue]))
        if pair_key not in self._spread_history:
            self._spread_history[pair_key] = deque(maxlen=self.lookback_days * 24)

        self._spread_history[pair_key].append(spread)

        history = list(self._spread_history[pair_key])
        if len(history) >= 5:
            mean = np.mean(history)
            std = np.std(history)
            z_score = (spread - mean) / std if std > 0 else 0.0
        else:
            z_score = 0.0

        # Calculate confidence based on data quality
        confidence = min(len(history) / (self.lookback_days * 24), 1.0)
        vol_sum = ts1.funding_volatility + ts2.funding_volatility
        if vol_sum < 0.02:  # Guard against division and ensure valid range
            confidence *= (1 - abs(vol_sum) / 0.02)
        else:
            confidence *= 0.0  # Very high volatility = no confidence
        confidence = max(0.3, min(confidence, 1.0))

        # Expected daily return
        expected_daily_return_bps = net_spread_bps / 365

        return CrossVenueFundingSpread(
            timestamp=ts1.timestamp,
            venue_long=venue_long,
            venue_short=venue_short,
            venue_long_type=VENUE_FUNDING_CONFIG.get(
                venue_long.lower(), VENUE_FUNDING_CONFIG['binance']
            ).venue_type,
            venue_short_type=VENUE_FUNDING_CONFIG.get(
                venue_short.lower(), VENUE_FUNDING_CONFIG['binance']
            ).venue_type,
            funding_long=funding_long,
            funding_short=funding_short,
            spread_annual_pct=spread,
            z_score=z_score,
            total_cost_bps=total_cost_bps,
            net_spread_bps=net_spread_bps,
            recommended_direction=f"Long {venue_long}, Short {venue_short}",
            expected_daily_return_bps=expected_daily_return_bps,
            confidence=confidence,
        )

    def find_arbitrage_opportunities(
        self,
        term_structures: Dict[str, FundingTermStructure],
        spot_price: float,
        max_position_usd: float = 1_000_000,
    ) -> List[FundingArbitrageOpportunity]:
        """
        Find all funding rate arbitrage opportunities.

        Args:
            term_structures: Dict of venue -> FundingTermStructure
            spot_price: Current spot price
            max_position_usd: Maximum position size

        Returns:
            List of opportunities sorted by attractiveness
        """
        opportunities = []

        venues = list(term_structures.keys())

        # Check all venue pairs
        for i, v1 in enumerate(venues):
            for v2 in venues[i+1:]:
                ts1 = term_structures[v1]
                ts2 = term_structures[v2]

                spread = self.calculate_cross_venue_spread(ts1, ts2)

                if not spread.is_profitable:
                    continue

                # Determine arb type
                if ts1.venue_type.is_on_chain or ts2.venue_type.is_on_chain:
                    arb_type = FundingArbitrageType.CROSS_VENUE_LONG
                else:
                    arb_type = FundingArbitrageType.FUNDING_RATE_HARVEST

                # Get venue costs
                costs1 = DEFAULT_VENUE_COSTS.get(v1.lower())
                costs2 = DEFAULT_VENUE_COSTS.get(v2.lower())

                cost1_bps = costs1.round_trip_taker_bps if costs1 else 20.0
                cost2_bps = costs2.round_trip_taker_bps if costs2 else 20.0

                # Calculate capacity
                cap1 = DEFAULT_VENUE_CAPACITY.get(v1.lower(), 100_000_000)
                cap2 = DEFAULT_VENUE_CAPACITY.get(v2.lower(), 100_000_000)
                venue_max = min(cap1, cap2) * 0.01  # 1% of venue capacity

                actual_max = min(max_position_usd, venue_max)

                # Calculate risk score
                risk_score = 0.3  # Base risk
                if ts1.venue_type.is_on_chain or ts2.venue_type.is_on_chain:
                    risk_score += 0.2  # On-chain risk
                if spread.z_score > 3:
                    risk_score -= 0.1  # Strong signal reduces risk perception
                risk_score += ts1.funding_volatility * 10  # Volatility adds risk
                risk_score = max(0.1, min(risk_score, 0.9))

                # Expected holding based on regime
                regime1 = ts1.regime
                regime2 = ts2.regime
                expected_days = min(
                    regime1.expected_reversion_days,
                    regime2.expected_reversion_days
                )

                # Recommended position based on z-score and costs
                position_mult = min(abs(spread.z_score) / 3, 1.0)
                recommended = actual_max * position_mult * spread.confidence

                opportunity = FundingArbitrageOpportunity(
                    timestamp=spread.timestamp,
                    arb_type=arb_type,
                    venue_long=spread.venue_long,
                    venue_short=spread.venue_short,
                    symbol='BTC',
                    spot_price=spot_price,
                    funding_long_annual_pct=spread.funding_long,
                    funding_short_annual_pct=spread.funding_short,
                    gross_spread_annual_pct=spread.spread_annual_pct,
                    venue_long_costs_bps=cost1_bps if spread.venue_long == v1 else cost2_bps,
                    venue_short_costs_bps=cost2_bps if spread.venue_short == v2 else cost1_bps,
                    total_costs_bps=spread.total_cost_bps,
                    net_spread_annual_pct=spread.net_spread_bps / 100,
                    expected_daily_pnl_bps=spread.expected_daily_return_bps,
                    max_position_usd=actual_max,
                    recommended_position_usd=recommended,
                    expected_holding_days=expected_days,
                    confidence=spread.confidence,
                    risk_score=risk_score,
                )

                opportunities.append(opportunity)

        # Sort by attractiveness
        opportunities.sort(
            key=lambda o: o.expected_daily_pnl_bps * o.confidence / (o.risk_score + 0.1),
            reverse=True
        )

        return opportunities

    def compare_to_futures_curve(
        self,
        funding_ts: FundingTermStructure,
        actual_basis_by_dte: Dict[int, float],
    ) -> Dict[str, Any]:
        """
        Compare funding-implied to actual futures curve.

        Args:
            funding_ts: Funding-implied term structure
            actual_basis_by_dte: Actual basis by DTE

        Returns:
            Comparison analysis
        """
        comparisons = []

        for dte, actual_basis in actual_basis_by_dte.items():
            implied_basis = funding_ts.get_implied_basis(dte)
            differential = actual_basis - implied_basis

            comparisons.append({
                'dte': dte,
                'actual_basis_pct': actual_basis,
                'implied_basis_pct': implied_basis,
                'differential_pct': differential,
                'is_rich': differential > 2.0,
                'is_cheap': differential < -2.0,
            })

        df = pd.DataFrame(comparisons)

        avg_differential = df['differential_pct'].mean() if not df.empty else 0.0

        return {
            'venue': funding_ts.venue,
            'timestamp': funding_ts.timestamp,
            'comparisons': comparisons,
            'avg_differential_pct': avg_differential,
            'futures_rich': avg_differential > 1.0,
            'futures_cheap': avg_differential < -1.0,
            'recommendation': 'short_futures' if avg_differential > 3.0 else (
                'long_futures' if avg_differential < -3.0 else 'neutral'
            ),
        }

    def get_regime_statistics(self, venue: str) -> Dict[str, Any]:
        """Get regime statistics for a venue."""
        if venue not in self._regime_history:
            return {'venue': venue, 'no_data': True}

        history = self._regime_history[venue]
        if not history:
            return {'venue': venue, 'no_data': True}

        regimes = [r for _, r in history]

        distribution = {}
        for r in FundingRegime:
            count = sum(1 for x in regimes if x == r)
            distribution[r.value] = count / len(regimes) if regimes else 0.0

        current = history[-1][1] if history else FundingRegime.NEUTRAL

        # Duration of current regime
        duration = 0
        for _, r in reversed(history):
            if r == current:
                duration += 1
            else:
                break

        return {
            'venue': venue,
            'current_regime': current.value,
            'regime_duration': duration,
            'distribution': distribution,
            'total_observations': len(history),
        }

    def summary(self) -> Dict[str, Any]:
        """Get analyzer summary."""
        return {
            'venues': self.venues,
            'lookback_days': self.lookback_days,
            'min_spread_bps': self.min_spread_bps,
            'min_z_score': self.min_z_score,
            'venue_stats': {
                v: self.get_regime_statistics(v) for v in self.venues
            },
        }

    def predict_funding_rate_ewma(
        self,
        venue: str,
        forecast_periods: int = 24,
        alpha: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Predict future funding rates using EWMA model.

        Args:
            venue: Venue name
            forecast_periods: Number of periods to forecast
            alpha: EWMA smoothing parameter

        Returns:
            Dictionary with predictions and confidence intervals
        """
        history = list(self._funding_history.get(venue.lower(), []))
        if len(history) < 10:
            return {'error': 'Insufficient history', 'venue': venue}

        rates = [s.funding_rate_8h for s in history]

        # Calculate EWMA
        ewma = [rates[0]]
        for r in rates[1:]:
            ewma.append(alpha * r + (1 - alpha) * ewma[-1])

        current_ewma = ewma[-1]

        # Calculate volatility for confidence intervals
        residuals = [r - e for r, e in zip(rates[-len(ewma):], ewma)]
        vol = np.std(residuals) if residuals else 0.01

        # Generate forecasts (assume mean reversion)
        long_term_mean = np.mean(rates[-min(len(rates), 720):])  # Last 30 days
        reversion_speed = 0.02  # Daily reversion rate

        forecasts = []
        current = current_ewma
        for i in range(forecast_periods):
            # Mean reverting forecast
            forecast = current + reversion_speed * (long_term_mean - current)
            ci_lower = forecast - 1.96 * vol * np.sqrt(i + 1)
            ci_upper = forecast + 1.96 * vol * np.sqrt(i + 1)

            forecasts.append({
                'period': i + 1,
                'forecast': forecast,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
            })
            current = forecast

        return {
            'venue': venue,
            'current_ewma': current_ewma,
            'long_term_mean': long_term_mean,
            'volatility': vol,
            'forecasts': forecasts,
            'annualized_forecast': forecasts[-1]['forecast'] * 1095 * 100 if forecasts else 0,
        }

    def calculate_optimal_hedge_ratio(
        self,
        venue_long: str,
        venue_short: str,
        lookback_periods: int = 168,  # 7 days of hourly data
    ) -> Dict[str, float]:
        """
        Calculate optimal hedge ratio for cross-venue funding arbitrage.

        Uses OLS regression to find minimum variance hedge ratio.

        Args:
            venue_long: Venue for long position
            venue_short: Venue for short position
            lookback_periods: Lookback for calculation

        Returns:
            Dictionary with hedge ratio and statistics
        """
        history_long = list(self._funding_history.get(venue_long.lower(), []))
        history_short = list(self._funding_history.get(venue_short.lower(), []))

        if len(history_long) < 20 or len(history_short) < 20:
            return {
                'hedge_ratio': 1.0,
                'r_squared': 0.0,
                'error': 'Insufficient data',
            }

        # Align by timestamp
        rates_long = {s.timestamp: s.funding_rate_8h for s in history_long}
        rates_short = {s.timestamp: s.funding_rate_8h for s in history_short}

        common_ts = sorted(set(rates_long.keys()) & set(rates_short.keys()))[-lookback_periods:]

        if len(common_ts) < 20:
            return {
                'hedge_ratio': 1.0,
                'r_squared': 0.0,
                'error': 'Insufficient overlapping data',
            }

        y = np.array([rates_short[ts] for ts in common_ts])
        x = np.array([rates_long[ts] for ts in common_ts])

        # OLS regression for hedge ratio
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            hedge_ratio = 1.0
            r_squared = 0.0
        else:
            hedge_ratio = numerator / denominator
            ss_res = np.sum((y - (hedge_ratio * x + (y_mean - hedge_ratio * x_mean))) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Correlation for additional insight
        correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0

        return {
            'hedge_ratio': hedge_ratio,
            'r_squared': r_squared,
            'correlation': correlation,
            'venue_long': venue_long,
            'venue_short': venue_short,
            'data_points': len(common_ts),
        }

    def detect_funding_regime_shift(
        self,
        venue: str,
        sensitivity: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Detect recent regime shifts in funding rates.

        Uses CUSUM-like algorithm to detect changes.

        Args:
            venue: Venue name
            sensitivity: Sensitivity threshold for detection

        Returns:
            Dictionary with regime shift analysis
        """
        history = list(self._funding_history.get(venue.lower(), []))
        if len(history) < 48:  # Need at least 2 days
            return {'venue': venue, 'no_shift': True, 'reason': 'Insufficient data'}

        rates = [s.funding_rate_8h for s in history]

        # Split into reference and test periods
        split_idx = len(rates) // 2
        ref_mean = np.mean(rates[:split_idx])
        ref_std = np.std(rates[:split_idx]) if np.std(rates[:split_idx]) > 0 else 0.001

        # CUSUM calculation
        cusum_pos = 0.0
        cusum_neg = 0.0
        shift_detected = False
        shift_idx = None

        for i, r in enumerate(rates[split_idx:], split_idx):
            z = (r - ref_mean) / ref_std
            cusum_pos = max(0, cusum_pos + z - 0.5)
            cusum_neg = min(0, cusum_neg + z + 0.5)

            if cusum_pos > sensitivity or cusum_neg < -sensitivity:
                shift_detected = True
                shift_idx = i
                break

        if shift_detected and shift_idx:
            new_mean = np.mean(rates[shift_idx:])
            shift_magnitude = (new_mean - ref_mean) / ref_std
            shift_direction = 'upward' if new_mean > ref_mean else 'downward'

            return {
                'venue': venue,
                'shift_detected': True,
                'shift_index': shift_idx,
                'shift_timestamp': history[shift_idx].timestamp if shift_idx < len(history) else None,
                'reference_mean': ref_mean,
                'new_mean': new_mean,
                'shift_magnitude_std': shift_magnitude,
                'shift_direction': shift_direction,
                'current_regime': FundingRegime.from_annual_rate(new_mean * 1095 * 100).value,
            }

        return {
            'venue': venue,
            'shift_detected': False,
            'current_mean': np.mean(rates[-48:]) if len(rates) >= 48 else np.mean(rates),
            'current_regime': history[-1].regime.value if history else 'unknown',
        }

    def calculate_funding_carry_curve(
        self,
        venue: str,
        spot_price: float,
        max_days: int = 365,
    ) -> pd.DataFrame:
        """
        Calculate implied carry curve from funding rates.

        This creates a synthetic term structure by compounding
        expected funding rates forward.

        Args:
            venue: Venue name
            spot_price: Current spot price
            max_days: Maximum days forward

        Returns:
            DataFrame with implied prices by DTE
        """
        prediction = self.predict_funding_rate_ewma(venue, forecast_periods=max_days * 3)

        if 'error' in prediction:
            # Fallback to last known rate
            history = list(self._funding_history.get(venue.lower(), []))
            if not history:
                return pd.DataFrame()
            last_rate = history[-1].funding_rate_8h
            forecasts = [{'forecast': last_rate} for _ in range(max_days * 3)]
        else:
            forecasts = prediction.get('forecasts', [])

        config = VENUE_FUNDING_CONFIG.get(venue.lower(), VENUE_FUNDING_CONFIG['binance'])
        periods_per_day = config.interval.periods_per_day

        results = []
        cumulative_funding = 0.0

        for dte in range(1, max_days + 1):
            # Sum funding for this day
            start_idx = (dte - 1) * periods_per_day
            end_idx = dte * periods_per_day

            daily_funding = sum(
                f['forecast'] for f in forecasts[start_idx:end_idx]
                if f
            ) if len(forecasts) > end_idx else (
                forecasts[-1]['forecast'] * periods_per_day if forecasts else 0
            )

            cumulative_funding += daily_funding

            implied_price = spot_price * (1 + cumulative_funding)
            implied_basis_pct = (implied_price - spot_price) / spot_price * 100
            annualized_basis = implied_basis_pct * (365 / dte)

            results.append({
                'dte': dte,
                'implied_price': implied_price,
                'cumulative_funding_rate': cumulative_funding,
                'implied_basis_pct': implied_basis_pct,
                'annualized_basis_pct': annualized_basis,
            })

        return pd.DataFrame(results)

    def analyze_crisis_funding_behavior(
        self,
        funding_history: pd.DataFrame,
        venue: str,
    ) -> Dict[str, Any]:
        """
        Analyze funding rate behavior during known crisis periods.

        Per PDF Section 3.3.3, this examines:
        - COVID crash (March 2020)
        - May 2021 crash
        - Luna collapse
        - FTX collapse

        Args:
            funding_history: DataFrame with timestamp and funding_rate columns
            venue: Venue name

        Returns:
            Dictionary with crisis period analysis
        """
        if funding_history.empty or 'timestamp' not in funding_history.columns:
            return {'venue': venue, 'error': 'Invalid data'}

        config = VENUE_FUNDING_CONFIG.get(venue.lower(), VENUE_FUNDING_CONFIG['binance'])
        crisis_analysis = {}

        for crisis_id, crisis in CRISIS_EVENTS.items():
            mask = (
                (funding_history['timestamp'] >= crisis['start']) &
                (funding_history['timestamp'] <= crisis['end'])
            )
            crisis_data = funding_history[mask]

            if crisis_data.empty:
                crisis_analysis[crisis_id] = {
                    'name': crisis['name'],
                    'has_data': False,
                }
                continue

            rates = crisis_data['funding_rate'].values
            annual_rates = config.annualize_rate(rates)

            # Calculate statistics
            min_rate = np.min(annual_rates)
            max_rate = np.max(annual_rates)
            mean_rate = np.mean(annual_rates)
            vol = np.std(annual_rates)

            # Determine actual vs expected behavior
            actual_regime = FundingRegime.from_annual_rate(mean_rate)
            expected = crisis['expected_funding_behavior']
            matched_expectation = actual_regime.value == expected

            crisis_analysis[crisis_id] = {
                'name': crisis['name'],
                'has_data': True,
                'data_points': len(crisis_data),
                'severity': crisis['severity'],
                'min_annual_pct': round(min_rate, 2),
                'max_annual_pct': round(max_rate, 2),
                'mean_annual_pct': round(mean_rate, 2),
                'volatility': round(vol, 4),
                'actual_regime': actual_regime.value,
                'expected_regime': expected,
                'matched_expectation': matched_expectation,
            }

        # Overall crisis summary
        valid_crises = [c for c in crisis_analysis.values() if c.get('has_data')]
        avg_crisis_funding = np.mean([c['mean_annual_pct'] for c in valid_crises]) if valid_crises else 0

        return {
            'venue': venue,
            'crisis_periods': crisis_analysis,
            'avg_crisis_funding_pct': round(avg_crisis_funding, 2),
            'crises_analyzed': len(valid_crises),
            'prediction_accuracy': sum(
                1 for c in valid_crises if c.get('matched_expectation')
            ) / len(valid_crises) if valid_crises else 0,
        }


# =============================================================================
# INTEGRATION WITH TERM_STRUCTURE.PY
# =============================================================================

class FundingTermStructureIntegration:
    """
    Integration layer between funding_rate_analysis and term_structure modules.

    This class provides the tight wiring required per PDF Section 3.1 to ensure
    synthetic funding-implied curves can be directly compared and combined with
    actual futures term structures.
    """

    def __init__(self, funding_analyzer: FundingRateAnalyzer):
        """
        Initialize integration.

        Args:
            funding_analyzer: FundingRateAnalyzer instance
        """
        self.analyzer = funding_analyzer
        logger.info("FundingTermStructureIntegration initialized")

    def build_synthetic_curve_points(
        self,
        funding_ts: FundingTermStructure,
        target_dtes: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Build synthetic curve points matching actual futures tenor points.

        Args:
            funding_ts: Funding term structure
            target_dtes: List of DTEs to generate points for

        Returns:
            List of curve point dictionaries compatible with TermStructureCurve
        """
        points = []
        for dte in target_dtes:
            implied_price = funding_ts.get_implied_price(dte)
            implied_basis = ((implied_price - funding_ts.spot_price) / funding_ts.spot_price) * 100
            annualized = implied_basis * (365 / max(dte, 1))

            points.append({
                'dte': dte,
                'expiry': funding_ts.timestamp + pd.Timedelta(days=dte),
                'price': implied_price,
                'basis_pct': implied_basis,
                'annualized_basis_pct': annualized,
                'source': 'funding_implied',
                'venue': funding_ts.venue,
                'confidence': self._calculate_point_confidence(funding_ts, dte),
            })

        return points

    def _calculate_point_confidence(
        self,
        funding_ts: FundingTermStructure,
        dte: int,
    ) -> float:
        """Calculate confidence score for a funding-implied point."""
        # Base confidence from funding volatility (lower vol = higher confidence)
        vol_penalty = min(funding_ts.funding_volatility * 50, 0.3)

        # Tenor penalty (longer = less confident)
        tenor_penalty = min(dte / 365 * 0.4, 0.4)

        # Regime penalty (extreme regimes are less stable)
        regime_penalty = 0.1 if funding_ts.regime.is_extreme else 0.0

        confidence = 1.0 - vol_penalty - tenor_penalty - regime_penalty
        return max(0.2, min(confidence, 0.95))

    def compare_synthetic_vs_actual(
        self,
        funding_ts: FundingTermStructure,
        actual_futures: Dict[int, Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Compare funding-implied synthetic curve to actual futures.

        This identifies mispricings between perpetual funding expectations
        and actual dated futures prices.

        Args:
            funding_ts: Funding-implied term structure
            actual_futures: Dict of DTE -> {'price': float, 'basis_pct': float}

        Returns:
            Comparison analysis
        """
        comparisons = []
        total_mispricing = 0.0

        for dte, actual in actual_futures.items():
            synthetic_price = funding_ts.get_implied_price(dte)
            synthetic_basis = funding_ts.get_implied_basis(dte)
            actual_basis = actual.get('basis_pct', 0.0)

            mispricing_pct = actual_basis - synthetic_basis
            total_mispricing += abs(mispricing_pct)

            # Determine trade recommendation
            if mispricing_pct > 3.0:  # Futures rich vs funding
                recommendation = 'short_futures_long_perp'
                signal_strength = min(mispricing_pct / 10, 1.0)
            elif mispricing_pct < -3.0:  # Futures cheap vs funding
                recommendation = 'long_futures_short_perp'
                signal_strength = min(abs(mispricing_pct) / 10, 1.0)
            else:
                recommendation = 'neutral'
                signal_strength = 0.0

            comparisons.append({
                'dte': dte,
                'actual_price': actual.get('price', 0),
                'synthetic_price': synthetic_price,
                'price_diff': actual.get('price', 0) - synthetic_price,
                'actual_basis_pct': actual_basis,
                'synthetic_basis_pct': synthetic_basis,
                'mispricing_pct': mispricing_pct,
                'recommendation': recommendation,
                'signal_strength': signal_strength,
            })

        # Overall analysis
        avg_mispricing = np.mean([c['mispricing_pct'] for c in comparisons]) if comparisons else 0
        futures_bias = 'rich' if avg_mispricing > 1.5 else ('cheap' if avg_mispricing < -1.5 else 'fair')

        # Best opportunity
        if comparisons:
            best_opp = max(comparisons, key=lambda c: abs(c['mispricing_pct']))
        else:
            best_opp = None

        return {
            'venue': funding_ts.venue,
            'timestamp': funding_ts.timestamp,
            'comparisons': comparisons,
            'avg_mispricing_pct': round(avg_mispricing, 2),
            'total_absolute_mispricing': round(total_mispricing, 2),
            'futures_bias': futures_bias,
            'funding_regime': funding_ts.regime.value,
            'best_opportunity': best_opp,
            'n_actionable_signals': sum(
                1 for c in comparisons if c['signal_strength'] > 0.3
            ),
        }

    def calculate_basis_convergence_expected(
        self,
        funding_ts: FundingTermStructure,
        actual_basis_pct: float,
        dte: int,
    ) -> Dict[str, Any]:
        """
        Calculate expected basis convergence path.

        Args:
            funding_ts: Funding term structure
            actual_basis_pct: Current actual futures basis
            dte: Days to expiry

        Returns:
            Expected convergence analysis
        """
        synthetic_basis = funding_ts.get_implied_basis(dte)
        current_gap = actual_basis_pct - synthetic_basis

        # Estimate daily convergence
        # Basis should converge to spot at expiry
        daily_convergence_actual = actual_basis_pct / max(dte, 1)
        daily_convergence_synthetic = synthetic_basis / max(dte, 1)

        # Expected P&L from convergence
        # If actual > synthetic, actual will converge faster
        convergence_rate = daily_convergence_actual - daily_convergence_synthetic

        paths = []
        cumulative_convergence = 0.0
        for d in range(1, dte + 1):
            expected_actual = actual_basis_pct * (1 - d / dte)
            expected_synthetic = synthetic_basis * (1 - d / dte)
            gap = expected_actual - expected_synthetic

            paths.append({
                'days_forward': d,
                'expected_actual_basis': expected_actual,
                'expected_synthetic_basis': expected_synthetic,
                'expected_gap': gap,
            })

        return {
            'venue': funding_ts.venue,
            'dte': dte,
            'initial_gap_pct': current_gap,
            'daily_convergence_differential': convergence_rate,
            'total_expected_convergence_pnl_pct': current_gap,
            'paths': paths,
            'favorable_trade': 'long_futures' if current_gap < -2 else (
                'short_futures' if current_gap > 2 else 'neutral'
            ),
        }

    def get_cross_venue_synthetic_spread(
        self,
        funding_structures: Dict[str, FundingTermStructure],
        dte: int,
    ) -> Dict[str, Any]:
        """
        Calculate synthetic spread across venues for a target DTE.

        Args:
            funding_structures: Dict of venue -> FundingTermStructure
            dte: Target days to expiry

        Returns:
            Cross-venue spread analysis
        """
        venue_points = {}
        for venue, ts in funding_structures.items():
            implied_basis = ts.get_implied_basis(dte)
            venue_points[venue] = {
                'implied_basis_pct': implied_basis,
                'funding_regime': ts.regime.value,
                'funding_annual_pct': ts.annual_funding_pct,
            }

        if len(venue_points) < 2:
            return {'error': 'Need at least 2 venues', 'dte': dte}

        # Find best spread
        venues = list(venue_points.keys())
        best_spread = 0.0
        best_pair = None

        for i, v1 in enumerate(venues):
            for v2 in venues[i+1:]:
                spread = venue_points[v1]['implied_basis_pct'] - venue_points[v2]['implied_basis_pct']
                if abs(spread) > abs(best_spread):
                    best_spread = spread
                    best_pair = (v1, v2) if spread > 0 else (v2, v1)

        # Costs adjustment
        if best_pair:
            costs1 = DEFAULT_VENUE_COSTS.get(best_pair[0].lower())
            costs2 = DEFAULT_VENUE_COSTS.get(best_pair[1].lower())
            total_cost_bps = (
                (costs1.round_trip_taker_bps if costs1 else 20) +
                (costs2.round_trip_taker_bps if costs2 else 20)
            )
            net_spread_bps = abs(best_spread) * 100 - total_cost_bps
        else:
            net_spread_bps = 0.0
            total_cost_bps = 0.0

        return {
            'dte': dte,
            'venue_points': venue_points,
            'best_pair': best_pair,
            'gross_spread_pct': abs(best_spread) if best_spread else 0,
            'total_cost_bps': total_cost_bps,
            'net_spread_bps': net_spread_bps,
            'is_actionable': net_spread_bps > 10,
            'recommendation': f"Long {best_pair[0]}, Short {best_pair[1]}" if best_pair and net_spread_bps > 10 else 'No action',
        }


# =============================================================================
# FUNDING RATE BACKTEST ENGINE
# =============================================================================

@dataclass
class FundingBacktestConfig:
    """Configuration for funding rate backtest."""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float = 1_000_000
    max_position_pct: float = 0.25
    max_positions: int = 3
    min_spread_bps: float = 15.0
    min_z_score: float = 1.5
    max_holding_days: int = 30
    stop_loss_pct: float = 2.0
    rebalance_frequency: str = 'daily'
    include_gas_costs: bool = True
    leverage: float = 2.0  # PDF: 2.0x max per PDF Section 3.2 (Hyperliquid: 1.5x max)


@dataclass
class FundingBacktestResult:
    """Results from funding rate backtest."""
    config: FundingBacktestConfig
    trades: List[Dict[str, Any]]
    equity_curve: pd.DataFrame
    total_pnl: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    avg_holding_days: float
    total_funding_collected: float
    total_costs: float
    venue_breakdown: Dict[str, Dict[str, float]]

    def summary(self) -> Dict[str, Any]:
        return {
            'total_pnl': round(self.total_pnl, 2),
            'total_return_pct': round(self.total_return_pct, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'max_drawdown_pct': round(self.max_drawdown_pct, 2),
            'win_rate': round(self.win_rate, 1),
            'profit_factor': round(self.profit_factor, 2),
            'total_trades': len(self.trades),
            'avg_holding_days': round(self.avg_holding_days, 1),
            'funding_collected': round(self.total_funding_collected, 2),
            'total_costs': round(self.total_costs, 2),
        }


class FundingRateBacktester:
    """
    Backtest funding rate arbitrage strategies.

    Simulates:
    - Cross-venue funding arbitrage
    - Cash-and-carry with funding
    - Funding rate harvesting
    """

    def __init__(self, config: FundingBacktestConfig):
        """Initialize backtester."""
        self.config = config
        self.analyzer = FundingRateAnalyzer()

        logger.info(f"FundingRateBacktester initialized: {config.start_date} to {config.end_date}")

    def run_backtest(
        self,
        funding_data: Dict[str, pd.DataFrame],
        spot_prices: pd.Series,
    ) -> FundingBacktestResult:
        """
        Run funding rate arbitrage backtest.

        Args:
            funding_data: Dict of venue -> DataFrame with funding rates
            spot_prices: Series of spot prices

        Returns:
            FundingBacktestResult
        """
        capital = self.config.initial_capital
        trades: List[Dict[str, Any]] = []
        equity_records = []
        open_positions: Dict[str, Dict] = {}

        total_funding = 0.0
        total_costs = 0.0
        venue_stats: Dict[str, Dict[str, float]] = {}

        # Get common timestamps
        all_timestamps = set()
        for df in funding_data.values():
            if 'timestamp' in df.columns:
                all_timestamps.update(df['timestamp'].tolist())

        # Ensure config dates are timezone-aware for comparison
        start_date = self.config.start_date
        end_date = self.config.end_date
        if hasattr(start_date, 'tzinfo') and start_date.tzinfo is None:
            start_date = pd.Timestamp(start_date).tz_localize('UTC')
        if hasattr(end_date, 'tzinfo') and end_date.tzinfo is None:
            end_date = pd.Timestamp(end_date).tz_localize('UTC')

        all_timestamps = sorted([
            t for t in all_timestamps
            if start_date <= pd.Timestamp(t) <= end_date
        ])

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
                venue_df = df[df['timestamp'] <= ts].tail(30 * 24)  # Last 30 days
                if not venue_df.empty:
                    term_structures[venue] = self.analyzer.build_funding_term_structure(
                        venue_df, venue, spot, ts
                    )

            # Update open positions (collect/pay funding)
            for pos_id, pos in list(open_positions.items()):
                # Calculate funding for this period
                venue_long = pos['venue_long']
                venue_short = pos['venue_short']

                if venue_long in term_structures and venue_short in term_structures:
                    ts_long = term_structures[venue_long]
                    ts_short = term_structures[venue_short]

                    # Funding paid/received
                    funding_paid = ts_long.current_funding_rate * pos['size_usd']
                    funding_received = ts_short.current_funding_rate * pos['size_usd']

                    net_funding = funding_received - funding_paid
                    pos['cumulative_funding'] += net_funding
                    total_funding += net_funding

                    pos['holding_periods'] += 1

                    # Check exit conditions
                    holding_days = pos['holding_periods'] / 3  # Assuming 8h periods
                    pnl_pct = pos['cumulative_funding'] / pos['size_usd'] * 100

                    should_exit = (
                        holding_days >= self.config.max_holding_days or
                        pnl_pct < -self.config.stop_loss_pct
                    )

                    if should_exit:
                        # Close position
                        costs = pos['entry_costs'] * 2  # Exit costs
                        total_costs += costs

                        net_pnl = pos['cumulative_funding'] - costs
                        capital += net_pnl

                        trades.append({
                            'entry_time': pos['entry_time'],
                            'exit_time': ts,
                            'venue_long': venue_long,
                            'venue_short': venue_short,
                            'size_usd': pos['size_usd'],
                            'gross_pnl': pos['cumulative_funding'],
                            'costs': pos['entry_costs'] + costs,
                            'net_pnl': net_pnl,
                            'holding_days': holding_days,
                        })

                        del open_positions[pos_id]

            # Find new opportunities
            if len(open_positions) < self.config.max_positions:
                opportunities = self.analyzer.find_arbitrage_opportunities(
                    term_structures, spot,
                    capital * self.config.max_position_pct
                )

                for opp in opportunities:
                    if not opp.is_actionable:
                        continue
                    if len(open_positions) >= self.config.max_positions:
                        break

                    # Check if we already have this pair
                    pair_key = tuple(sorted([opp.venue_long, opp.venue_short]))
                    if any(
                        tuple(sorted([p['venue_long'], p['venue_short']])) == pair_key
                        for p in open_positions.values()
                    ):
                        continue

                    # Open position
                    size = min(
                        opp.recommended_position_usd,
                        capital * self.config.max_position_pct
                    )

                    entry_costs = size * opp.total_costs_bps / 10000
                    total_costs += entry_costs

                    pos_id = f"{ts}_{opp.venue_long}_{opp.venue_short}"
                    open_positions[pos_id] = {
                        'entry_time': ts,
                        'venue_long': opp.venue_long,
                        'venue_short': opp.venue_short,
                        'size_usd': size,
                        'entry_costs': entry_costs,
                        'cumulative_funding': 0.0,
                        'holding_periods': 0,
                    }

                    # Track venue stats
                    for v in [opp.venue_long, opp.venue_short]:
                        if v not in venue_stats:
                            venue_stats[v] = {'trades': 0, 'volume': 0.0}
                        venue_stats[v]['trades'] += 1
                        venue_stats[v]['volume'] += size

            # Track equity
            unrealized = sum(p['cumulative_funding'] for p in open_positions.values())
            equity_records.append({
                'timestamp': ts,
                'capital': capital,
                'unrealized': unrealized,
                'equity': capital + unrealized,
                'open_positions': len(open_positions),
            })

        # Close remaining positions at end
        for pos_id, pos in open_positions.items():
            costs = pos['entry_costs']
            net_pnl = pos['cumulative_funding'] - costs
            capital += net_pnl

            trades.append({
                'entry_time': pos['entry_time'],
                'exit_time': self.config.end_date,
                'venue_long': pos['venue_long'],
                'venue_short': pos['venue_short'],
                'size_usd': pos['size_usd'],
                'gross_pnl': pos['cumulative_funding'],
                'costs': pos['entry_costs'] * 2,
                'net_pnl': net_pnl,
                'holding_days': pos['holding_periods'] / 3,
            })

        # Build equity curve
        equity_df = pd.DataFrame(equity_records)
        if not equity_df.empty:
            equity_df.set_index('timestamp', inplace=True)

        # Calculate metrics
        total_pnl = capital - self.config.initial_capital
        total_return_pct = (total_pnl / self.config.initial_capital) * 100

        if not equity_df.empty:
            returns = equity_df['equity'].pct_change().dropna()
            sharpe = (returns.mean() / returns.std() * np.sqrt(365 * 3)) if returns.std() > 0 else 0.0

            peak = equity_df['equity'].expanding().max()
            drawdown = (equity_df['equity'] - peak) / peak * 100
            max_dd = abs(drawdown.min())
        else:
            sharpe = 0.0
            max_dd = 0.0

        winners = [t for t in trades if t['net_pnl'] > 0]
        win_rate = (len(winners) / len(trades) * 100) if trades else 0.0

        gross_wins = sum(t['net_pnl'] for t in trades if t['net_pnl'] > 0)
        gross_losses = abs(sum(t['net_pnl'] for t in trades if t['net_pnl'] < 0))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

        avg_holding = np.mean([t['holding_days'] for t in trades]) if trades else 0.0

        return FundingBacktestResult(
            config=self.config,
            trades=trades,
            equity_curve=equity_df,
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_holding_days=avg_holding,
            total_funding_collected=total_funding,
            total_costs=total_costs,
            venue_breakdown=venue_stats,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'FundingInterval',
    'FundingRegime',
    'FundingArbitrageType',
    # Configs
    'VenueFundingConfig',
    'VENUE_FUNDING_CONFIG',
    'FUNDING_NORMALIZATION_MATRIX',
    # Crisis definitions
    'CRISIS_EVENTS',
    'is_crisis_period',
    # Data structures
    'FundingRateSnapshot',
    'FundingImpliedPoint',
    'FundingTermStructure',
    'CrossVenueFundingSpread',
    'FundingArbitrageOpportunity',
    # Analyzers
    'FundingRateAnalyzer',
    # Integration with term_structure.py
    'FundingTermStructureIntegration',
    # Backtesting
    'FundingBacktestConfig',
    'FundingBacktestResult',
    'FundingRateBacktester',
    # Utility functions
    'normalize_funding_rate_cross_venue',
    'get_annualized_funding',
    'get_8h_equivalent_funding',
]
