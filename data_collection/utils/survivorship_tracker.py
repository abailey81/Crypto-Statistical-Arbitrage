"""
Survivorship Bias Tracker for Crypto Statistical Arbitrage.

This module tracks and adjusts for survivorship bias in crypto data, which is
critical for accurate backtesting and strategy evaluation.

Survivorship Bias in Crypto:
- Failed/delisted tokens are excluded from current datasets
- Creates upward bias in historical performance metrics
- Research shows:
  - 0.93% annualized bias for value-weighted portfolios
  - 62.19% annualized bias for equal-weighted portfolios (crypto-specific)

Key Features:
- Track delisted tokens with last trading dates
- Monitor exchange listing/delisting events
- Track symbol renames and migrations
- Calculate bias adjustment factors
- Provide survivorship-adjusted returns

References:
- Elendner et al. (2018): "Cross-section of Crypto-Asset Returns"
- Liu et al. (2019): "Risks and Returns of Cryptocurrency"

Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class DelistingReason(Enum):
    """Reason for token delisting."""
    LOW_VOLUME = auto() # Insufficient trading volume
    REGULATORY = auto() # Regulatory concerns
    SECURITY_ISSUE = auto() # Security vulnerability/hack
    PROJECT_FAILED = auto() # Project abandoned or failed
    MERGED = auto() # Token merged/migrated to new token
    REBRANDED = auto() # Token rebranded with new symbol
    UPGRADED = auto() # Token upgraded to new version
    UNKNOWN = auto() # Unknown reason

class BiasType(Enum):
    """Type of survivorship bias calculation."""
    VALUE_WEIGHTED = 'value_weighted' # Weight by market cap
    EQUAL_WEIGHTED = 'equal_weighted' # Equal weight all tokens
    LIQUIDITY_WEIGHTED = 'liquidity_weighted' # Weight by trading volume

# Documented survivorship bias estimates from academic research
ANNUAL_BIAS_ESTIMATES = {
    BiasType.VALUE_WEIGHTED: 0.0093, # 0.93% per year (Liu et al.)
    BiasType.EQUAL_WEIGHTED: 0.6219, # 62.19% per year (extreme for small caps)
    BiasType.LIQUIDITY_WEIGHTED: 0.0250, # ~2.5% estimated
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DelistedToken:
    """Record of a delisted token."""
    symbol: str
    name: Optional[str] = None
    delisting_date: Optional[datetime] = None
    first_seen_date: Optional[datetime] = None
    last_trading_date: Optional[datetime] = None
    reason: DelistingReason = DelistingReason.UNKNOWN
    exchanges: List[str] = field(default_factory=list)
    successor_symbol: Optional[str] = None # For migrations/mergers
    peak_market_cap_usd: Optional[float] = None
    final_price_usd: Optional[float] = None
    total_return_pct: Optional[float] = None # Lifetime return
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'delisting_date': self.delisting_date.isoformat() if self.delisting_date else None,
            'first_seen_date': self.first_seen_date.isoformat() if self.first_seen_date else None,
            'last_trading_date': self.last_trading_date.isoformat() if self.last_trading_date else None,
            'reason': self.reason.name,
            'exchanges': self.exchanges,
            'successor_symbol': self.successor_symbol,
            'peak_market_cap_usd': self.peak_market_cap_usd,
            'final_price_usd': self.final_price_usd,
            'total_return_pct': self.total_return_pct,
        }

@dataclass
class ExchangeEvent:
    """Record of an exchange listing/delisting event."""
    symbol: str
    exchange: str
    event_type: str # 'listing' or 'delisting'
    event_date: datetime
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SymbolRename:
    """Record of a symbol rename/migration."""
    old_symbol: str
    new_symbol: str
    effective_date: datetime
    exchange: Optional[str] = None # None means all exchanges
    migration_type: str = 'rename' # 'rename', 'migration', 'upgrade'
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SurvivorshipAdjustment:
    """Result of survivorship bias adjustment calculation."""
    start_date: datetime
    end_date: datetime
    bias_type: BiasType
    raw_return: float
    adjustment_factor: float
    adjusted_return: float
    delisted_tokens_count: int
    total_tokens_count: int
    attrition_rate: float
    confidence: float # Confidence in the adjustment (0-1)
    details: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# SURVIVORSHIP TRACKER
# =============================================================================

class SurvivorshipBiasTracker:
    """
    Track and adjust for survivorship bias in cryptocurrency data.

    Survivorship bias is one of the most significant issues in crypto backtesting:
    - Datasets typically only contain currently-trading tokens
    - Failed/delisted tokens are excluded from historical analysis
    - This creates artificial inflation of historical returns

    Example Impact:
    - Equal-weighted portfolio: 62.19% annual bias
    - This means backtested returns are ~62% higher than reality

    Usage:
        tracker = SurvivorshipBiasTracker()
        tracker.add_delisted_token(DelistedToken(symbol='LUNA', ...))
        adjustment = tracker.calculate_bias_adjustment(
            portfolio_weights='value',
            date_range=(start, end)
        )
        adjusted_return = raw_return * adjustment.adjustment_factor
    """

    def __init__(self):
        """Initialize tracker."""
        self.delisted_tokens: Dict[str, DelistedToken] = {}
        self.exchange_events: List[ExchangeEvent] = []
        self.symbol_renames: List[SymbolRename] = []
        self._universe_snapshots: Dict[datetime, Set[str]] = {}

    def add_delisted_token(self, token: DelistedToken) -> None:
        """Add a delisted token record."""
        self.delisted_tokens[token.symbol] = token
        logger.debug(f"Added delisted token: {token.symbol}")

    def add_exchange_event(self, event: ExchangeEvent) -> None:
        """Add an exchange listing/delisting event."""
        self.exchange_events.append(event)
        logger.debug(f"Added {event.event_type} event: {event.symbol} on {event.exchange}")

    def add_symbol_rename(self, rename: SymbolRename) -> None:
        """Add a symbol rename record."""
        self.symbol_renames.append(rename)
        logger.debug(f"Added rename: {rename.old_symbol} -> {rename.new_symbol}")

    def record_universe_snapshot(
        self,
        snapshot_date: datetime,
        symbols: Set[str]
    ) -> None:
        """
        Record the active trading universe at a point in time.

        This is crucial for:
        - Tracking universe composition over time
        - Detecting when tokens enter/exit
        - Accurate bias calculation
        """
        self._universe_snapshots[snapshot_date] = symbols.copy()

    def get_delisted_in_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[DelistedToken]:
        """Get tokens delisted within a date range."""
        delisted = []
        for token in self.delisted_tokens.values():
            if token.delisting_date:
                if start_date <= token.delisting_date <= end_date:
                    delisted.append(token)
            elif token.last_trading_date:
                if start_date <= token.last_trading_date <= end_date:
                    delisted.append(token)
        return delisted

    def get_attrition_rate(
        self,
        start_date: datetime,
        end_date: datetime,
        universe_size: Optional[int] = None
    ) -> float:
        """
        Calculate the attrition rate (percentage of tokens delisted) over a period.

        Parameters:
            start_date: Period start
            end_date: Period end
            universe_size: Total tokens at start (if not using snapshots)

        Returns:
            Attrition rate as a decimal (0.05 = 5%)
        """
        delisted = self.get_delisted_in_range(start_date, end_date)
        delisted_count = len(delisted)

        # Try to get universe size from snapshots
        if universe_size is None:
            closest_snapshot = self._get_closest_snapshot(start_date)
            if closest_snapshot:
                universe_size = len(closest_snapshot)
            else:
                # Estimate based on typical crypto universe
                universe_size = 100 # Conservative estimate

        if universe_size == 0:
            return 0.0

        return delisted_count / universe_size

    def _get_closest_snapshot(self, target_date: datetime) -> Optional[Set[str]]:
        """Get the closest universe snapshot to a date."""
        if not self._universe_snapshots:
            return None

        closest_date = min(
            self._universe_snapshots.keys(),
            key=lambda d: abs((d - target_date).total_seconds())
        )
        return self._universe_snapshots[closest_date]

    def calculate_bias_adjustment(
        self,
        date_range: Tuple[datetime, datetime],
        portfolio_weights: BiasType = BiasType.VALUE_WEIGHTED,
        raw_return: Optional[float] = None,
        universe_size: Optional[int] = None,
    ) -> SurvivorshipAdjustment:
        """
        Calculate survivorship bias adjustment factor.

        The adjustment factor can be used to deflate backtested returns:
            adjusted_return = raw_return / adjustment_factor

        Parameters:
            date_range: (start_date, end_date) tuple
            portfolio_weights: Type of portfolio weighting
            raw_return: Optional raw return to adjust
            universe_size: Optional universe size at start

        Returns:
            SurvivorshipAdjustment with adjustment factor and details
        """
        start_date, end_date = date_range
        years = (end_date - start_date).days / 365.25

        # Get annual bias rate based on portfolio type
        annual_bias = ANNUAL_BIAS_ESTIMATES.get(portfolio_weights, 0.01)

        # Calculate cumulative bias over the period
        # Bias compounds over time: (1 + annual_bias)^years - 1
        cumulative_bias = (1 + annual_bias) ** years - 1

        # Adjustment factor: multiply raw return by this to get "true" return
        # A factor < 1 means the raw return is inflated
        adjustment_factor = 1 / (1 + cumulative_bias)

        # Get attrition data
        delisted = self.get_delisted_in_range(start_date, end_date)
        attrition_rate = self.get_attrition_rate(start_date, end_date, universe_size)

        # Estimate confidence based on data quality
        confidence = self._calculate_confidence(len(delisted), years, universe_size)

        # Calculate adjusted return if raw return provided
        adjusted_return = raw_return * adjustment_factor if raw_return else 0.0

        return SurvivorshipAdjustment(
            start_date=start_date,
            end_date=end_date,
            bias_type=portfolio_weights,
            raw_return=raw_return or 0.0,
            adjustment_factor=adjustment_factor,
            adjusted_return=adjusted_return,
            delisted_tokens_count=len(delisted),
            total_tokens_count=universe_size or 100,
            attrition_rate=attrition_rate,
            confidence=confidence,
            details={
                'annual_bias_rate': annual_bias,
                'cumulative_bias': cumulative_bias,
                'period_years': years,
                'delisted_symbols': [t.symbol for t in delisted],
            }
        )

    def _calculate_confidence(
        self,
        delisted_count: int,
        years: float,
        universe_size: Optional[int]
    ) -> float:
        """
        Calculate confidence in the adjustment based on data quality.

        Factors:
        - More delisting data = higher confidence
        - Longer time period = higher confidence
        - Larger universe = higher confidence
        """
        # Base confidence
        confidence = 0.5

        # Adjust for delisting data availability
        if delisted_count > 50:
            confidence += 0.2
        elif delisted_count > 20:
            confidence += 0.1
        elif delisted_count == 0:
            confidence -= 0.2 # No data, using estimates

        # Adjust for time period
        if years >= 2:
            confidence += 0.1
        elif years < 0.5:
            confidence -= 0.1

        # Adjust for universe size
        if universe_size and universe_size > 50:
            confidence += 0.1

        return max(0.1, min(1.0, confidence))

    def resolve_symbol(
        self,
        symbol: str,
        as_of_date: Optional[datetime] = None
    ) -> str:
        """
        Resolve a symbol to its current name, accounting for renames.

        Parameters:
            symbol: Original symbol
            as_of_date: Optional date context

        Returns:
            Current symbol (or original if no rename found)
        """
        current_symbol = symbol

        for rename in sorted(self.symbol_renames, key=lambda r: r.effective_date):
            if as_of_date and rename.effective_date > as_of_date:
                break
            if rename.old_symbol == current_symbol:
                current_symbol = rename.new_symbol

        return current_symbol

    def get_historical_symbol(
        self,
        current_symbol: str,
        as_of_date: datetime
    ) -> str:
        """
        Get the historical symbol name at a specific point in time.

        Useful for matching historical data where symbols may have changed.
        """
        # Reverse lookup through renames
        historical_symbol = current_symbol

        for rename in sorted(self.symbol_renames, key=lambda r: r.effective_date, reverse=True):
            if rename.effective_date > as_of_date:
                continue
            if rename.new_symbol == historical_symbol:
                historical_symbol = rename.old_symbol

        return historical_symbol

    def is_delisted(
        self,
        symbol: str,
        as_of_date: Optional[datetime] = None
    ) -> bool:
        """Check if a symbol is/was delisted."""
        token = self.delisted_tokens.get(symbol)
        if not token:
            return False

        if as_of_date is None:
            return True

        if token.delisting_date:
            return token.delisting_date <= as_of_date
        if token.last_trading_date:
            return token.last_trading_date <= as_of_date

        return True

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        reasons_count = {}
        for token in self.delisted_tokens.values():
            reason = token.reason.name
            reasons_count[reason] = reasons_count.get(reason, 0) + 1

        return {
            'total_delisted_tokens': len(self.delisted_tokens),
            'total_exchange_events': len(self.exchange_events),
            'total_symbol_renames': len(self.symbol_renames),
            'universe_snapshots': len(self._universe_snapshots),
            'delisting_reasons': reasons_count,
            'annual_bias_estimates': {
                k.value: v for k, v in ANNUAL_BIAS_ESTIMATES.items()
            },
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert delisted tokens to DataFrame."""
        if not self.delisted_tokens:
            return pd.DataFrame()

        records = [t.to_dict() for t in self.delisted_tokens.values()]
        return pd.DataFrame(records)

# =============================================================================
# KNOWN DELISTED TOKENS DATABASE
# =============================================================================

def get_known_delistings() -> List[DelistedToken]:
    """
    Get a curated list of known major delistings.

    This is a starting point - should be supplemented with
    exchange-specific delisting announcements.
    """
    return [
        DelistedToken(
            symbol='LUNA',
            name='Terra Luna Classic',
            delisting_date=datetime(2022, 5, 13, tzinfo=timezone.utc),
            reason=DelistingReason.PROJECT_FAILED,
            exchanges=['binance', 'okx', 'kraken', 'coinbase'],
            successor_symbol='LUNC',
            total_return_pct=-99.99,
            metadata={'event': 'Terra collapse', 'ust_depeg': True}
        ),
        DelistedToken(
            symbol='UST',
            name='TerraUSD',
            delisting_date=datetime(2022, 5, 13, tzinfo=timezone.utc),
            reason=DelistingReason.PROJECT_FAILED,
            exchanges=['binance', 'okx', 'kraken', 'coinbase'],
            successor_symbol='USTC',
            total_return_pct=-99.99,
            metadata={'event': 'UST depeg'}
        ),
        DelistedToken(
            symbol='FTT',
            name='FTX Token',
            delisting_date=datetime(2022, 11, 11, tzinfo=timezone.utc),
            reason=DelistingReason.PROJECT_FAILED,
            exchanges=['binance', 'okx', 'kraken'],
            total_return_pct=-95.0,
            metadata={'event': 'FTX collapse'}
        ),
        DelistedToken(
            symbol='CEL',
            name='Celsius',
            delisting_date=datetime(2022, 7, 13, tzinfo=timezone.utc),
            reason=DelistingReason.PROJECT_FAILED,
            exchanges=['binance', 'okx'],
            total_return_pct=-99.0,
            metadata={'event': 'Celsius bankruptcy'}
        ),
        DelistedToken(
            symbol='SRM',
            name='Serum',
            delisting_date=datetime(2022, 12, 1, tzinfo=timezone.utc),
            reason=DelistingReason.PROJECT_FAILED,
            exchanges=['binance', 'okx'],
            total_return_pct=-98.0,
            metadata={'event': 'FTX-related, Serum discontinued'}
        ),
    ]

def create_tracker_with_known_delistings() -> SurvivorshipBiasTracker:
    """Create a tracker pre-populated with known major delistings."""
    tracker = SurvivorshipBiasTracker()
    for token in get_known_delistings():
        tracker.add_delisted_token(token)
    return tracker

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'DelistingReason',
    'BiasType',
    'DelistedToken',
    'ExchangeEvent',
    'SymbolRename',
    'SurvivorshipAdjustment',
    'SurvivorshipBiasTracker',
    'ANNUAL_BIAS_ESTIMATES',
    'get_known_delistings',
    'create_tracker_with_known_delistings',
]
