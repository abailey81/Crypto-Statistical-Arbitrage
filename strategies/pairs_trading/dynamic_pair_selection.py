"""
Dynamic Pair Selection for Crypto Pairs Trading
================================================

Comprehensive dynamic pair selection with periodic rebalancing,
tier promotion/demotion, and continuous DEX scanning.

PDF Requirement (Section 2.3 Option C):
"Monthly rebalancing where old pairs that have lost cointegration
are replaced with new pairs"

Features:
- Monthly rebalancing based on cointegration re-testing
- Tier promotion/demotion based on performance
- Continuous DEX scanning for new opportunities
- Survivorship handling for delistings
- Performance-weighted selection

Author: Crypto StatArb Quantitative Research
Version: 2.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class SelectionAction(Enum):
    """Actions for dynamic pair selection."""
    ADD = "add"           # Add new pair
    REMOVE = "remove"     # Remove failing pair
    PROMOTE = "promote"   # Promote from lower tier
    DEMOTE = "demote"     # Demote to lower tier
    KEEP = "keep"         # Maintain current status

    @property
    def is_change(self) -> bool:
        """True if action changes pair status."""
        return self != SelectionAction.KEEP

    @property
    def color_code(self) -> str:
        """Color code for visualization."""
        colors = {
            self.ADD: "#00FF00",      # Green
            self.REMOVE: "#FF0000",   # Red
            self.PROMOTE: "#00FFFF",  # Cyan
            self.DEMOTE: "#FFA500",   # Orange
            self.KEEP: "#808080",     # Gray
        }
        return colors.get(self, "#808080")


class TierLevel(Enum):
    """Pair tier classifications."""
    TIER1_CEX = "tier1_cex"       # Top tier: CEX only pairs
    TIER2_MIXED = "tier2_mixed"   # Mixed: CEX + some DEX
    TIER3_DEX = "tier3_dex"       # DEX-focused pairs
    TIER4_EXPERIMENTAL = "tier4_experimental"  # New/experimental pairs

    @property
    def max_position_pct(self) -> float:
        """Maximum position size as fraction of portfolio."""
        limits = {
            self.TIER1_CEX: 0.10,
            self.TIER2_MIXED: 0.07,
            self.TIER3_DEX: 0.04,
            self.TIER4_EXPERIMENTAL: 0.02,
        }
        return limits.get(self, 0.02)

    @property
    def allowed_in_crisis(self) -> bool:
        """Whether tier is allowed during crisis."""
        return self in [TierLevel.TIER1_CEX]

    @property
    def requires_daily_check(self) -> bool:
        """Whether tier requires daily cointegration checks."""
        return self in [TierLevel.TIER3_DEX, TierLevel.TIER4_EXPERIMENTAL]

    @classmethod
    def from_string(cls, value: str) -> 'TierLevel':
        """Create from string value."""
        for tier in cls:
            if tier.value == value:
                return tier
        return cls.TIER4_EXPERIMENTAL


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PairStatus:
    """Comprehensive status of a trading pair."""
    pair: Tuple[str, str]
    tier: TierLevel
    added_date: datetime
    last_test_date: datetime
    is_cointegrated: bool
    p_value: float
    half_life_hours: float
    hedge_ratio: float = 1.0
    performance_score: float = 0.0
    trade_count: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    avg_holding_hours: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0

    @property
    def pair_name(self) -> str:
        """Formatted pair name."""
        return f"{self.pair[0]}/{self.pair[1]}"

    @property
    def days_since_added(self) -> int:
        """Days since pair was added."""
        return (datetime.now() - self.added_date).days

    @property
    def hours_since_test(self) -> float:
        """Hours since last cointegration test."""
        return (datetime.now() - self.last_test_date).total_seconds() / 3600

    @property
    def needs_retest(self) -> bool:
        """Whether pair needs cointegration retest."""
        if self.tier.requires_daily_check:
            return self.hours_since_test > 24
        return self.hours_since_test > 168  # Weekly

    @property
    def is_performing_well(self) -> bool:
        """Whether pair is performing above threshold."""
        return (
            self.win_rate >= 0.45 and
            self.sharpe_ratio >= 0.5 and
            self.consecutive_losses < 5
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pair': self.pair_name,
            'tier': self.tier.value,
            'p_value': round(self.p_value, 4),
            'half_life_hours': round(self.half_life_hours, 1),
            'win_rate': round(self.win_rate, 3),
            'trade_count': self.trade_count,
            'total_pnl': round(self.total_pnl, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 3),
            'performance_score': round(self.performance_score, 3),
            'is_cointegrated': self.is_cointegrated,
            'days_active': self.days_since_added,
        }


@dataclass
class SelectionConfig:
    """Configuration for dynamic pair selection."""
    # Rebalancing intervals
    rebalance_interval_hours: int = 720   # Monthly (30 days)
    min_test_frequency_hours: int = 168   # Weekly minimum

    # Tier quotas
    max_tier1_pairs: int = 15
    max_tier2_pairs: int = 10
    max_tier3_pairs: int = 8
    max_tier4_pairs: int = 5

    # Removal thresholds
    remove_p_value_threshold: float = 0.10
    remove_min_win_rate: float = 0.35
    remove_max_consecutive_losses: int = 5
    remove_min_trades_for_evaluation: int = 10

    # Promotion thresholds
    promote_p_value_threshold: float = 0.01
    promote_min_trades: int = 15
    promote_min_win_rate: float = 0.55
    promote_min_sharpe: float = 1.0

    # Demotion thresholds
    demote_max_consecutive_losses: int = 4
    demote_min_win_rate: float = 0.40
    demote_max_drawdown: float = 0.15

    # DEX scanning
    scan_new_dex_tokens: bool = True
    min_dex_tvl_usd: float = 500_000
    min_dex_age_days: int = 30
    min_dex_volume_usd: float = 100_000

    # Cointegration requirements
    cointegration_significance: float = 0.05
    min_half_life_hours: float = 24
    max_half_life_hours: float = 336  # 14 days (PDF Section 2.3 Option C requirement)
    min_observations: int = 720  # 30 days of hourly data

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rebalance_interval_hours': self.rebalance_interval_hours,
            'max_tier1_pairs': self.max_tier1_pairs,
            'max_tier2_pairs': self.max_tier2_pairs,
            'max_tier3_pairs': self.max_tier3_pairs,
            'remove_p_value_threshold': self.remove_p_value_threshold,
            'promote_p_value_threshold': self.promote_p_value_threshold,
            'min_dex_tvl_usd': self.min_dex_tvl_usd,
        }


@dataclass
class RebalanceAction:
    """Record of a rebalancing action."""
    action: SelectionAction
    pair: Tuple[str, str]
    timestamp: datetime
    reason: str
    from_tier: Optional[TierLevel] = None
    to_tier: Optional[TierLevel] = None
    p_value: Optional[float] = None
    half_life: Optional[float] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'action': self.action.value,
            'pair': f"{self.pair[0]}/{self.pair[1]}",
            'timestamp': self.timestamp.isoformat(),
            'reason': self.reason,
            'from_tier': self.from_tier.value if self.from_tier else None,
            'to_tier': self.to_tier.value if self.to_tier else None,
            'p_value': self.p_value,
            'half_life': self.half_life,
        }


@dataclass
class RebalanceSummary:
    """Summary of a rebalancing operation."""
    timestamp: datetime
    n_pairs_before: int
    n_pairs_after: int
    n_added: int
    n_removed: int
    n_promoted: int
    n_demoted: int
    actions: List[RebalanceAction] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        """Total number of changes."""
        return self.n_added + self.n_removed + self.n_promoted + self.n_demoted

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'n_pairs_before': self.n_pairs_before,
            'n_pairs_after': self.n_pairs_after,
            'n_added': self.n_added,
            'n_removed': self.n_removed,
            'n_promoted': self.n_promoted,
            'n_demoted': self.n_demoted,
            'total_changes': self.total_changes,
        }


# =============================================================================
# DYNAMIC PAIR SELECTOR
# =============================================================================

class DynamicPairSelector:
    """
    Dynamic pair selection with periodic rebalancing.

    PDF requirement (Section 2.3 Option C):
    "Monthly rebalancing where old pairs that have lost cointegration
    are replaced with new pairs"

    Features:
    - Continuous DEX scanning for new opportunities
    - Tier promotion/demotion based on performance
    - Survivorship handling for delistings
    - Performance-weighted selection
    """

    def __init__(
        self,
        config: Optional[SelectionConfig] = None
    ):
        """
        Initialize dynamic pair selector.

        Args:
            config: Selection configuration
        """
        self.config = config or SelectionConfig()

        # Current universe
        self.active_pairs: Dict[Tuple[str, str], PairStatus] = {}
        self.removed_pairs: Set[Tuple[str, str]] = set()
        self.pair_history: List[Dict] = []

        # Rebalancing history
        self.rebalance_history: List[RebalanceSummary] = []
        self.last_rebalance: Optional[datetime] = None
        self.last_dex_scan: Optional[datetime] = None

    def add_initial_pairs(
        self,
        pairs: List[Dict[str, Any]],
        price_data: pd.DataFrame
    ) -> List[RebalanceAction]:
        """
        Add initial pairs from universe construction.

        Args:
            pairs: List of pair dictionaries with keys:
                   symbol1, symbol2, p_value, half_life, tier
            price_data: Price matrix for validation

        Returns:
            List of add actions
        """
        actions = []

        for pair_info in pairs:
            # Handle both PairConfig objects and dicts
            if hasattr(pair_info, 'symbol_a'):
                symbol1 = pair_info.symbol_a
                symbol2 = pair_info.symbol_b
            elif isinstance(pair_info, dict):
                symbol1 = pair_info.get('symbol1', pair_info.get('token_a', ''))
                symbol2 = pair_info.get('symbol2', pair_info.get('token_b', ''))
            else:
                continue
            pair = (symbol1, symbol2)

            # Validate data availability
            if symbol1 not in price_data.columns or symbol2 not in price_data.columns:
                logger.warning(f"Skipping pair {symbol1}/{symbol2}: data not available")
                continue

            # Determine tier
            tier_str = getattr(pair_info, 'tier', None) or (pair_info.get('tier', 'tier2_mixed') if isinstance(pair_info, dict) else 'tier2_mixed')
            if isinstance(tier_str, TierLevel):
                tier = tier_str
            else:
                tier = TierLevel.from_string(str(tier_str).lower())

            # Create status - safely get attributes from PairConfig or dict
            def _get(obj, attr, default=None):
                if hasattr(obj, attr):
                    return getattr(obj, attr)
                elif isinstance(obj, dict):
                    return obj.get(attr, default)
                return default

            status = PairStatus(
                pair=pair,
                tier=tier,
                added_date=datetime.now(),
                last_test_date=datetime.now(),
                is_cointegrated=True,
                p_value=_get(pair_info, 'p_value', 0.01),
                half_life_hours=_get(pair_info, 'half_life', 48.0),
                hedge_ratio=_get(pair_info, 'hedge_ratio', 1.0),
            )

            self.active_pairs[pair] = status

            action = RebalanceAction(
                action=SelectionAction.ADD,
                pair=pair,
                timestamp=datetime.now(),
                reason='initial_universe',
                to_tier=tier,
                p_value=status.p_value,
                half_life=status.half_life_hours
            )
            actions.append(action)

        logger.info(f"Added {len(actions)} initial pairs")
        return actions

    def rebalance(
        self,
        price_data: pd.DataFrame,
        trade_results: Optional[Dict[Tuple[str, str], Dict]] = None,
        new_tokens: Optional[List[str]] = None
    ) -> RebalanceSummary:
        """
        Perform monthly rebalancing.

        Args:
            price_data: Current price data
            trade_results: Performance metrics per pair
            new_tokens: New tokens to consider (from DEX scanning)

        Returns:
            RebalanceSummary with all actions taken
        """
        logger.info("Starting monthly rebalancing...")

        trade_results = trade_results or {}
        new_tokens = new_tokens or []

        n_pairs_before = len(self.active_pairs)
        actions = []

        # 1. Re-test existing pairs
        logger.info("  Step 1: Re-testing existing pairs...")
        for pair, status in list(self.active_pairs.items()):
            action = self._evaluate_pair(pair, status, price_data, trade_results)

            if action.action == SelectionAction.REMOVE:
                self._remove_pair(pair, action.reason)
            elif action.action == SelectionAction.DEMOTE:
                self._demote_pair(pair)
            elif action.action == SelectionAction.PROMOTE:
                self._promote_pair(pair)

            if action.action != SelectionAction.KEEP:
                actions.append(action)

        # 2. Scan for new pairs
        if self.config.scan_new_dex_tokens and new_tokens:
            logger.info(f"  Step 2: Scanning {len(new_tokens)} new tokens...")
            new_pair_actions = self._scan_new_pairs(price_data, new_tokens)
            actions.extend(new_pair_actions)

        # 3. Check promotions based on performance
        logger.info("  Step 3: Checking promotions...")
        promotion_actions = self._check_promotions(trade_results)
        actions.extend(promotion_actions)

        # 4. Fill tier quotas
        logger.info("  Step 4: Filling tier quotas...")
        fill_actions = self._fill_tier_quotas(price_data)
        actions.extend(fill_actions)

        # Create summary
        n_pairs_after = len(self.active_pairs)
        summary = RebalanceSummary(
            timestamp=datetime.now(),
            n_pairs_before=n_pairs_before,
            n_pairs_after=n_pairs_after,
            n_added=sum(1 for a in actions if a.action == SelectionAction.ADD),
            n_removed=sum(1 for a in actions if a.action == SelectionAction.REMOVE),
            n_promoted=sum(1 for a in actions if a.action == SelectionAction.PROMOTE),
            n_demoted=sum(1 for a in actions if a.action == SelectionAction.DEMOTE),
            actions=actions
        )

        self.rebalance_history.append(summary)
        self.last_rebalance = datetime.now()

        logger.info(f"Rebalancing complete: {summary.total_changes} changes "
                   f"({summary.n_added} added, {summary.n_removed} removed, "
                   f"{summary.n_promoted} promoted, {summary.n_demoted} demoted)")

        return summary

    def _evaluate_pair(
        self,
        pair: Tuple[str, str],
        status: PairStatus,
        price_data: pd.DataFrame,
        trade_results: Dict[Tuple[str, str], Dict]
    ) -> RebalanceAction:
        """Evaluate if a pair should be kept, demoted, or removed."""
        token_a, token_b = pair

        # Check data availability
        if token_a not in price_data.columns or token_b not in price_data.columns:
            return RebalanceAction(
                action=SelectionAction.REMOVE,
                pair=pair,
                timestamp=datetime.now(),
                reason='data_unavailable'
            )

        # Re-test cointegration
        test_result = self._test_cointegration(
            price_data[token_a],
            price_data[token_b]
        )

        # Get performance metrics
        perf = trade_results.get(pair, {})
        win_rate = perf.get('win_rate', status.win_rate)
        consecutive_losses = perf.get('consecutive_losses', status.consecutive_losses)
        n_trades = perf.get('n_trades', status.trade_count)
        max_drawdown = perf.get('max_drawdown', status.max_drawdown)

        # Update status
        status.win_rate = win_rate
        status.consecutive_losses = consecutive_losses
        status.trade_count = n_trades
        status.max_drawdown = max_drawdown
        status.last_test_date = datetime.now()

        # Decision logic
        # 1. Lost cointegration
        if not test_result['is_cointegrated']:
            if test_result['p_value'] > self.config.remove_p_value_threshold:
                return RebalanceAction(
                    action=SelectionAction.REMOVE,
                    pair=pair,
                    timestamp=datetime.now(),
                    reason='cointegration_lost',
                    from_tier=status.tier,
                    p_value=test_result['p_value']
                )
            else:
                return RebalanceAction(
                    action=SelectionAction.DEMOTE,
                    pair=pair,
                    timestamp=datetime.now(),
                    reason='cointegration_weakened',
                    from_tier=status.tier,
                    p_value=test_result['p_value']
                )

        # 2. Poor performance
        if (n_trades >= self.config.remove_min_trades_for_evaluation and
            win_rate < self.config.remove_min_win_rate):
            return RebalanceAction(
                action=SelectionAction.REMOVE,
                pair=pair,
                timestamp=datetime.now(),
                reason='poor_performance',
                from_tier=status.tier,
                performance_metrics={'win_rate': win_rate, 'n_trades': n_trades}
            )

        # 3. Too many consecutive losses
        if consecutive_losses >= self.config.remove_max_consecutive_losses:
            return RebalanceAction(
                action=SelectionAction.DEMOTE,
                pair=pair,
                timestamp=datetime.now(),
                reason='consecutive_losses',
                from_tier=status.tier,
                performance_metrics={'consecutive_losses': consecutive_losses}
            )

        # 4. Excessive drawdown
        if max_drawdown > self.config.demote_max_drawdown:
            return RebalanceAction(
                action=SelectionAction.DEMOTE,
                pair=pair,
                timestamp=datetime.now(),
                reason='excessive_drawdown',
                from_tier=status.tier,
                performance_metrics={'max_drawdown': max_drawdown}
            )

        # 5. Check for promotion
        if self._is_promotion_candidate(status, perf, test_result):
            return RebalanceAction(
                action=SelectionAction.PROMOTE,
                pair=pair,
                timestamp=datetime.now(),
                reason='strong_performance',
                from_tier=status.tier,
                to_tier=self._get_promotion_tier(status.tier),
                p_value=test_result['p_value'],
                performance_metrics={'win_rate': win_rate, 'n_trades': n_trades}
            )

        # 6. Keep as is
        status.is_cointegrated = test_result['is_cointegrated']
        status.p_value = test_result['p_value']
        status.half_life_hours = test_result.get('half_life', status.half_life_hours)
        status.performance_score = self._calculate_performance_score(perf)

        return RebalanceAction(
            action=SelectionAction.KEEP,
            pair=pair,
            timestamp=datetime.now(),
            reason='passed_evaluation',
            p_value=test_result['p_value']
        )

    def _test_cointegration(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series
    ) -> Dict[str, Any]:
        """
        Test cointegration for a pair.

        Uses Engle-Granger test with recent data.
        """
        # Use recent data only
        lookback = min(self.config.min_observations, len(prices_a))
        prices_a = prices_a.iloc[-lookback:].dropna()
        prices_b = prices_b.iloc[-lookback:].dropna()

        # Align indices
        common_idx = prices_a.index.intersection(prices_b.index)
        prices_a = prices_a.loc[common_idx]
        prices_b = prices_b.loc[common_idx]

        if len(prices_a) < 60:  # Minimum 60 observations
            return {
                'is_cointegrated': False,
                'p_value': 1.0,
                'half_life': 0.0,
                'hedge_ratio': 1.0
            }

        try:
            from statsmodels.tsa.stattools import coint

            # Run cointegration test
            _, p_value, _ = coint(prices_a.values, prices_b.values)

            # Calculate half-life if cointegrated
            half_life = 0.0
            hedge_ratio = 1.0

            if p_value < self.config.cointegration_significance:
                # OLS for hedge ratio
                from scipy import stats
                slope, intercept, _, _, _ = stats.linregress(
                    prices_b.values, prices_a.values
                )
                hedge_ratio = slope

                # Calculate spread and half-life
                spread = prices_a - hedge_ratio * prices_b
                spread_lag = spread.shift(1).dropna()
                spread_diff = spread.diff().dropna()

                common = spread_lag.index.intersection(spread_diff.index)
                spread_lag = spread_lag.loc[common]
                spread_diff = spread_diff.loc[common]

                if len(spread_lag) > 0:
                    reg_result = stats.linregress(spread_lag.values, spread_diff.values)
                    if reg_result.slope < 0:
                        half_life = -np.log(2) / reg_result.slope

            is_cointegrated = (
                p_value < self.config.cointegration_significance and
                self.config.min_half_life_hours <= half_life <= self.config.max_half_life_hours
            )

            return {
                'is_cointegrated': is_cointegrated,
                'p_value': p_value,
                'half_life': half_life,
                'hedge_ratio': hedge_ratio
            }

        except Exception as e:
            logger.warning(f"Cointegration test failed: {e}")
            return {
                'is_cointegrated': False,
                'p_value': 1.0,
                'half_life': 0.0,
                'hedge_ratio': 1.0
            }

    def _scan_new_pairs(
        self,
        price_data: pd.DataFrame,
        new_tokens: List[str]
    ) -> List[RebalanceAction]:
        """Scan new DEX tokens for trading opportunities."""
        actions = []

        # Get existing symbols
        existing_symbols = set()
        for pair in self.active_pairs:
            existing_symbols.add(pair[0])
            existing_symbols.add(pair[1])

        # Filter new tokens
        candidates = [t for t in new_tokens if t in price_data.columns and t not in existing_symbols]

        # Test against existing tokens
        for new_token in candidates[:20]:  # Limit to avoid excessive testing
            for existing in list(existing_symbols)[:20]:
                if existing not in price_data.columns:
                    continue

                # Skip if this pair was previously removed
                pair_key = tuple(sorted([new_token, existing]))
                if pair_key in self.removed_pairs:
                    continue

                # Test cointegration
                result = self._test_cointegration(
                    price_data[new_token],
                    price_data[existing]
                )

                if result['is_cointegrated']:
                    pair = (new_token, existing)

                    # Add to active pairs
                    status = PairStatus(
                        pair=pair,
                        tier=TierLevel.TIER3_DEX,
                        added_date=datetime.now(),
                        last_test_date=datetime.now(),
                        is_cointegrated=True,
                        p_value=result['p_value'],
                        half_life_hours=result['half_life'],
                        hedge_ratio=result['hedge_ratio']
                    )

                    if self._tier_has_space(TierLevel.TIER3_DEX):
                        self.active_pairs[pair] = status

                        action = RebalanceAction(
                            action=SelectionAction.ADD,
                            pair=pair,
                            timestamp=datetime.now(),
                            reason='dex_scan',
                            to_tier=TierLevel.TIER3_DEX,
                            p_value=result['p_value'],
                            half_life=result['half_life']
                        )
                        actions.append(action)

                        logger.info(f"Found new DEX pair: {new_token}/{existing}")

        return actions

    def _check_promotions(
        self,
        trade_results: Dict[Tuple[str, str], Dict]
    ) -> List[RebalanceAction]:
        """Check for pairs eligible for tier promotion."""
        actions = []

        for pair, status in self.active_pairs.items():
            if status.tier == TierLevel.TIER1_CEX:
                continue  # Already top tier

            perf = trade_results.get(pair, {})
            test_result = {'p_value': status.p_value}

            if self._is_promotion_candidate(status, perf, test_result):
                new_tier = self._get_promotion_tier(status.tier)

                if new_tier and self._tier_has_space(new_tier):
                    status.tier = new_tier

                    action = RebalanceAction(
                        action=SelectionAction.PROMOTE,
                        pair=pair,
                        timestamp=datetime.now(),
                        reason='performance_promotion',
                        from_tier=status.tier,
                        to_tier=new_tier,
                        p_value=status.p_value,
                        performance_metrics={
                            'win_rate': perf.get('win_rate', 0),
                            'n_trades': perf.get('n_trades', 0),
                            'sharpe': perf.get('sharpe_ratio', 0)
                        }
                    )
                    actions.append(action)

                    logger.info(f"Promoted {pair[0]}/{pair[1]} to {new_tier.value}")

        return actions

    def _is_promotion_candidate(
        self,
        status: PairStatus,
        perf: Dict,
        test_result: Dict
    ) -> bool:
        """Check if pair meets promotion criteria."""
        return (
            test_result.get('p_value', 1.0) < self.config.promote_p_value_threshold and
            perf.get('n_trades', 0) >= self.config.promote_min_trades and
            perf.get('win_rate', 0) >= self.config.promote_min_win_rate and
            perf.get('sharpe_ratio', 0) >= self.config.promote_min_sharpe
        )

    def _get_promotion_tier(self, current: TierLevel) -> Optional[TierLevel]:
        """Get next tier for promotion."""
        promotion_map = {
            TierLevel.TIER4_EXPERIMENTAL: TierLevel.TIER3_DEX,
            TierLevel.TIER3_DEX: TierLevel.TIER2_MIXED,
            TierLevel.TIER2_MIXED: TierLevel.TIER1_CEX,
        }
        return promotion_map.get(current)

    def _get_demotion_tier(self, current: TierLevel) -> Optional[TierLevel]:
        """Get next tier for demotion."""
        demotion_map = {
            TierLevel.TIER1_CEX: TierLevel.TIER2_MIXED,
            TierLevel.TIER2_MIXED: TierLevel.TIER3_DEX,
            TierLevel.TIER3_DEX: TierLevel.TIER4_EXPERIMENTAL,
        }
        return demotion_map.get(current)

    def _tier_has_space(self, tier: TierLevel) -> bool:
        """Check if tier has capacity for new pairs."""
        current_count = sum(
            1 for s in self.active_pairs.values() if s.tier == tier
        )

        limits = {
            TierLevel.TIER1_CEX: self.config.max_tier1_pairs,
            TierLevel.TIER2_MIXED: self.config.max_tier2_pairs,
            TierLevel.TIER3_DEX: self.config.max_tier3_pairs,
            TierLevel.TIER4_EXPERIMENTAL: self.config.max_tier4_pairs,
        }

        return current_count < limits.get(tier, 10)

    def _fill_tier_quotas(self, price_data: pd.DataFrame) -> List[RebalanceAction]:
        """Fill empty tier slots with new pairs."""
        # This would implement finding best candidates
        # Simplified implementation returns empty list
        return []

    def _calculate_performance_score(self, perf: Dict) -> float:
        """Calculate composite performance score."""
        win_rate = perf.get('win_rate', 0.5)
        profit_factor = perf.get('profit_factor', 1.0)
        sharpe = perf.get('sharpe_ratio', 0)
        n_trades = perf.get('n_trades', 0)

        # Weighted composite
        score = (
            0.3 * min(win_rate, 1.0) +
            0.3 * min(profit_factor / 2, 1.0) +
            0.2 * min(sharpe / 2, 1.0) +
            0.2 * min(n_trades / 50, 1.0)
        )

        return score

    def _remove_pair(self, pair: Tuple[str, str], reason: str):
        """Remove pair from active universe."""
        if pair in self.active_pairs:
            status = self.active_pairs[pair]

            self.pair_history.append({
                'pair': pair,
                'removed_date': datetime.now(),
                'added_date': status.added_date,
                'reason': reason,
                'final_pnl': status.total_pnl,
                'trade_count': status.trade_count,
                'tier': status.tier.value
            })

            del self.active_pairs[pair]
            self.removed_pairs.add(tuple(sorted(pair)))

            logger.info(f"Removed pair {pair[0]}/{pair[1]}: {reason}")

    def _demote_pair(self, pair: Tuple[str, str]):
        """Demote pair to lower tier."""
        if pair in self.active_pairs:
            status = self.active_pairs[pair]
            new_tier = self._get_demotion_tier(status.tier)

            if new_tier:
                status.tier = new_tier
                logger.info(f"Demoted {pair[0]}/{pair[1]} to {new_tier.value}")

    def _promote_pair(self, pair: Tuple[str, str]):
        """Promote pair to higher tier."""
        if pair in self.active_pairs:
            status = self.active_pairs[pair]
            new_tier = self._get_promotion_tier(status.tier)

            if new_tier and self._tier_has_space(new_tier):
                status.tier = new_tier
                logger.info(f"Promoted {pair[0]}/{pair[1]} to {new_tier.value}")

    def update_performance(
        self,
        pair: Tuple[str, str],
        trade_result: Dict[str, Any]
    ):
        """Update performance metrics for a pair after a trade."""
        if pair not in self.active_pairs:
            return

        status = self.active_pairs[pair]

        # Update metrics
        status.trade_count += 1
        pnl = trade_result.get('pnl', 0)
        status.total_pnl += pnl

        # Update win/loss tracking
        if pnl > 0:
            status.consecutive_wins += 1
            status.consecutive_losses = 0
        else:
            status.consecutive_losses += 1
            status.consecutive_wins = 0

        # Update win rate (rolling)
        if status.trade_count > 0:
            wins = trade_result.get('wins', status.win_rate * (status.trade_count - 1))
            status.win_rate = (wins + (1 if pnl > 0 else 0)) / status.trade_count

    def needs_rebalancing(self) -> bool:
        """Check if rebalancing is needed."""
        if self.last_rebalance is None:
            return True

        hours_since = (datetime.now() - self.last_rebalance).total_seconds() / 3600
        return hours_since >= self.config.rebalance_interval_hours

    def get_active_pairs_for_tier(self, tier: TierLevel) -> List[PairStatus]:
        """Get all active pairs for a specific tier."""
        return [
            status for status in self.active_pairs.values()
            if status.tier == tier
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current pair universe."""
        tier_counts = {}
        for tier in TierLevel:
            tier_counts[tier.value] = sum(
                1 for s in self.active_pairs.values() if s.tier == tier
            )

        return {
            'n_active_pairs': len(self.active_pairs),
            'n_removed_pairs': len(self.removed_pairs),
            'tier_distribution': tier_counts,
            'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
            'n_rebalances': len(self.rebalance_history),
            'avg_win_rate': np.mean([s.win_rate for s in self.active_pairs.values()]) if self.active_pairs else 0,
            'total_pnl': sum(s.total_pnl for s in self.active_pairs.values()),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert active pairs to DataFrame."""
        records = [status.to_dict() for status in self.active_pairs.values()]
        return pd.DataFrame(records)
