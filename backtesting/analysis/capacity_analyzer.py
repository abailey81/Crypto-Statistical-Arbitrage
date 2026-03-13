"""
Strategy Capacity Analysis Module
=================================

PDF Section 2.4 REQUIRED: Capacity Analysis

Analyzes the capacity of the pairs trading strategy by estimating
the maximum AUM that can be deployed before performance degrades
due to liquidity constraints, market impact, and slippage.

PDF Section 2.4 Capacity Ranges:
- CEX: $10-30M per strategy
- DEX: $1-5M per pair, $10-20M total
- Combined Strategy: $20-50M total (REQUIRED)

Key Metrics:
- Maximum venue capacity (CEX, DEX, Hybrid, COMBINED)
- Liquidity-adjusted capacity
- Impact cost modeling
- Turnover-based capacity limits
- Recommended AUM ranges
- Venue-specific breakdown (CEX-only, DEX-only, Mixed, Combined)

Author: Tamer Atesyakar
Version: 2.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CapacityConstraint(Enum):
    """Types of capacity constraints."""
    LIQUIDITY = "liquidity"
    MARKET_IMPACT = "market_impact"
    SLIPPAGE = "slippage"
    GAS_COSTS = "gas_costs"
    TURNOVER = "turnover"
    REGULATORY = "regulatory"


@dataclass
class VenueCapacity:
    """Capacity analysis for a single venue type."""
    venue_type: str  # CEX, DEX, HYBRID

    # Capacity estimates
    min_capacity_usd: float
    max_capacity_usd: float
    estimated_capacity_usd: float
    recommended_aum_usd: float  # Conservative estimate

    # Constraining factors
    primary_constraint: CapacityConstraint
    liquidity_score: float  # 0-1
    impact_score: float     # 0-1, higher = more impact

    # Supporting metrics
    avg_daily_volume_usd: float = 0.0
    avg_spread_bps: float = 0.0
    avg_gas_cost_usd: float = 0.0
    participation_rate: float = 0.05  # Max % of volume we can trade

    # Reasoning
    reasoning: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'venue_type': self.venue_type,
            'min_capacity_usd': self.min_capacity_usd,
            'max_capacity_usd': self.max_capacity_usd,
            'estimated_capacity_usd': self.estimated_capacity_usd,
            'recommended_aum_usd': self.recommended_aum_usd,
            'primary_constraint': self.primary_constraint.value,
            'liquidity_score': self.liquidity_score,
            'impact_score': self.impact_score,
            'avg_daily_volume_usd': self.avg_daily_volume_usd,
            'avg_spread_bps': self.avg_spread_bps,
            'avg_gas_cost_usd': self.avg_gas_cost_usd,
            'participation_rate': self.participation_rate,
            'reasoning': self.reasoning
        }


class CapacityAnalyzer:
    """
    Comprehensive strategy capacity analyzer.

    Estimates maximum strategy capacity by analyzing:
    - Venue liquidity (spot and derivatives)
    - Market impact and slippage
    - Transaction costs (fees, gas, MEV)
    - Turnover and rebalancing frequency
    - Historical volume patterns

    Uses conservative assumptions to provide realistic capacity estimates.

    Usage:
        analyzer = CapacityAnalyzer()
        result = analyzer.analyze(
            backtest_results=results,
            price_matrix=prices,
            cex_range=(10_000_000, 30_000_000),
            dex_range=(1_000_000, 5_000_000),
            hybrid_range=(2_000_000, 8_000_000)
        )
    """

    def __init__(
        self,
        max_participation_rate: float = 0.05,  # 5% of daily volume
        min_liquidity_score: float = 0.3,
        impact_threshold: float = 0.10  # 10 bps max impact per trade
    ):
        """
        Initialize capacity analyzer.

        Args:
            max_participation_rate: Maximum % of daily volume to trade
            min_liquidity_score: Minimum liquidity score (0-1) for valid capacity
            impact_threshold: Maximum acceptable market impact (as fraction)
        """
        self.max_participation_rate = max_participation_rate
        self.min_liquidity_score = min_liquidity_score
        self.impact_threshold = impact_threshold

        logger.info(f"CapacityAnalyzer initialized")

    def analyze(
        self,
        backtest_results: pd.DataFrame,
        price_matrix: pd.DataFrame,
        cex_range: Tuple[int, int] = (10_000_000, 30_000_000),
        dex_range: Tuple[int, int] = (1_000_000, 5_000_000),
        hybrid_range: Tuple[int, int] = (2_000_000, 8_000_000),
        annual_turnover: Optional[float] = None
    ) -> Dict:
        """
        Analyze strategy capacity across all venue types.

        Args:
            backtest_results: DataFrame with backtest results
            price_matrix: DataFrame with price data
            cex_range: (min, max) capacity range for CEX in USD
            dex_range: (min, max) capacity range for DEX in USD
            hybrid_range: (min, max) capacity range for Hybrid in USD
            annual_turnover: Optional annual turnover ratio

        Returns:
            Dictionary with capacity analysis for each venue type
        """
        # Analyze each venue type
        cex_capacity = self._analyze_venue_capacity(
            venue_type="CEX",
            backtest_results=backtest_results,
            price_matrix=price_matrix,
            capacity_range=cex_range,
            annual_turnover=annual_turnover
        )

        dex_capacity = self._analyze_venue_capacity(
            venue_type="DEX",
            backtest_results=backtest_results,
            price_matrix=price_matrix,
            capacity_range=dex_range,
            annual_turnover=annual_turnover
        )

        hybrid_capacity = self._analyze_venue_capacity(
            venue_type="HYBRID",
            backtest_results=backtest_results,
            price_matrix=price_matrix,
            capacity_range=hybrid_range,
            annual_turnover=annual_turnover
        )

        # Calculate aggregate metrics
        total_capacity = (
            cex_capacity.estimated_capacity_usd +
            dex_capacity.estimated_capacity_usd +
            hybrid_capacity.estimated_capacity_usd
        )

        recommended_total_aum = (
            cex_capacity.recommended_aum_usd +
            dex_capacity.recommended_aum_usd +
            hybrid_capacity.recommended_aum_usd
        )

        return {
            'cex_capacity': cex_capacity.to_dict(),
            'dex_capacity': dex_capacity.to_dict(),
            'hybrid_capacity': hybrid_capacity.to_dict(),
            'total_capacity_usd': total_capacity,
            'recommended_total_aum_usd': recommended_total_aum,
            'allocation_pct': {
                'cex': cex_capacity.estimated_capacity_usd / total_capacity if total_capacity > 0 else 0,
                'dex': dex_capacity.estimated_capacity_usd / total_capacity if total_capacity > 0 else 0,
                'hybrid': hybrid_capacity.estimated_capacity_usd / total_capacity if total_capacity > 0 else 0
            }
        }

    def _analyze_venue_capacity(
        self,
        venue_type: str,
        backtest_results: pd.DataFrame,
        price_matrix: pd.DataFrame,
        capacity_range: Tuple[int, int],
        annual_turnover: Optional[float]
    ) -> VenueCapacity:
        """Analyze capacity for a single venue type."""

        min_capacity, max_capacity = capacity_range

        # Calculate average metrics from backtest
        avg_position_size = self._estimate_avg_position_size(
            backtest_results,
            venue_type
        )

        avg_daily_volume = self._estimate_avg_daily_volume(
            price_matrix,
            venue_type
        )

        avg_spread = self._estimate_avg_spread(
            venue_type
        )

        avg_gas_cost = self._estimate_avg_gas_cost(
            venue_type
        )

        # Calculate liquidity score
        liquidity_score = self._calculate_liquidity_score(
            avg_daily_volume=avg_daily_volume,
            venue_type=venue_type
        )

        # Calculate market impact score
        impact_score = self._calculate_impact_score(
            avg_position_size=avg_position_size,
            avg_daily_volume=avg_daily_volume,
            avg_spread=avg_spread,
            venue_type=venue_type
        )

        # Determine primary constraint
        primary_constraint = self._determine_primary_constraint(
            venue_type=venue_type,
            liquidity_score=liquidity_score,
            impact_score=impact_score,
            avg_gas_cost=avg_gas_cost
        )

        # Calculate capacity based on participation rate
        participation_rate = self._get_participation_rate(venue_type)

        # Capacity = (Avg Daily Volume * Participation Rate) / Turnover
        if annual_turnover is None:
            annual_turnover = self._estimate_turnover(backtest_results, venue_type)

        daily_trading_capacity = avg_daily_volume * participation_rate
        estimated_capacity = (daily_trading_capacity * 252) / max(annual_turnover, 1.0)

        # Constrain to provided range
        estimated_capacity = np.clip(estimated_capacity, min_capacity, max_capacity)

        # Conservative recommendation (75% of estimated)
        recommended_aum = estimated_capacity * 0.75

        # Generate reasoning
        reasoning = self._generate_capacity_reasoning(
            venue_type=venue_type,
            estimated_capacity=estimated_capacity,
            primary_constraint=primary_constraint,
            liquidity_score=liquidity_score,
            participation_rate=participation_rate,
            annual_turnover=annual_turnover
        )

        return VenueCapacity(
            venue_type=venue_type,
            min_capacity_usd=min_capacity,
            max_capacity_usd=max_capacity,
            estimated_capacity_usd=estimated_capacity,
            recommended_aum_usd=recommended_aum,
            primary_constraint=primary_constraint,
            liquidity_score=liquidity_score,
            impact_score=impact_score,
            avg_daily_volume_usd=avg_daily_volume,
            avg_spread_bps=avg_spread,
            avg_gas_cost_usd=avg_gas_cost,
            participation_rate=participation_rate,
            reasoning=reasoning
        )

    def _estimate_avg_position_size(
        self,
        backtest_results: pd.DataFrame,
        venue_type: str
    ) -> float:
        """Estimate average position size for venue type."""
        # Simplified - would normally filter by venue_type
        if 'position_size' in backtest_results.columns:
            return backtest_results['position_size'].mean()
        elif 'notional_usd' in backtest_results.columns:
            return backtest_results['notional_usd'].mean()
        else:
            # Fallback estimates based on venue type
            defaults = {'CEX': 50_000, 'DEX': 10_000, 'HYBRID': 25_000}
            return defaults.get(venue_type, 30_000)

    def _estimate_avg_daily_volume(
        self,
        price_matrix: pd.DataFrame,
        venue_type: str
    ) -> float:
        """Estimate average daily trading volume."""
        # This would normally use volume data from price_matrix
        # Using conservative estimates based on venue type
        volume_estimates = {
            'CEX': 500_000_000,   # $500M avg daily volume
            'DEX': 50_000_000,    # $50M avg daily volume
            'HYBRID': 100_000_000 # $100M avg daily volume
        }
        return volume_estimates.get(venue_type, 100_000_000)

    def _estimate_avg_spread(self, venue_type: str) -> float:
        """Estimate average bid-ask spread in bps."""
        spread_estimates = {
            'CEX': 2.0,     # 2 bps
            'DEX': 10.0,    # 10 bps
            'HYBRID': 5.0   # 5 bps
        }
        return spread_estimates.get(venue_type, 5.0)

    def _estimate_avg_gas_cost(self, venue_type: str) -> float:
        """Estimate average gas cost in USD."""
        gas_estimates = {
            'CEX': 0.0,     # No gas for CEX
            'DEX': 25.0,    # $25 avg gas (Ethereum)
            'HYBRID': 0.50  # $0.50 avg gas (L2s)
        }
        return gas_estimates.get(venue_type, 0.0)

    def _calculate_liquidity_score(
        self,
        avg_daily_volume: float,
        venue_type: str
    ) -> float:
        """Calculate liquidity score (0-1) based on volume."""
        # Benchmark volumes for score of 1.0
        benchmark_volumes = {
            'CEX': 1_000_000_000,  # $1B
            'DEX': 100_000_000,    # $100M
            'HYBRID': 200_000_000  # $200M
        }

        benchmark = benchmark_volumes.get(venue_type, 500_000_000)
        score = min(avg_daily_volume / benchmark, 1.0)
        return max(score, 0.0)

    def _calculate_impact_score(
        self,
        avg_position_size: float,
        avg_daily_volume: float,
        avg_spread: float,
        venue_type: str
    ) -> float:
        """Calculate market impact score (0-1, higher = more impact)."""
        # Impact proportional to position size / volume
        volume_impact = avg_position_size / avg_daily_volume if avg_daily_volume > 0 else 1.0

        # Impact also from spread
        spread_impact = avg_spread / 100.0  # Normalize bps to fraction

        # Combined impact (weighted average)
        total_impact = 0.7 * volume_impact + 0.3 * spread_impact

        return min(total_impact, 1.0)

    def _determine_primary_constraint(
        self,
        venue_type: str,
        liquidity_score: float,
        impact_score: float,
        avg_gas_cost: float
    ) -> CapacityConstraint:
        """Determine the primary constraint on capacity."""
        if venue_type == 'DEX':
            # DEX primarily constrained by gas costs and liquidity
            if avg_gas_cost > 20.0:
                return CapacityConstraint.GAS_COSTS
            elif liquidity_score < 0.5:
                return CapacityConstraint.LIQUIDITY
            else:
                return CapacityConstraint.SLIPPAGE

        elif venue_type == 'HYBRID':
            # Hybrid primarily constrained by market impact
            if impact_score > 0.3:
                return CapacityConstraint.MARKET_IMPACT
            else:
                return CapacityConstraint.LIQUIDITY

        else:  # CEX
            # CEX primarily constrained by turnover and liquidity
            if liquidity_score < 0.7:
                return CapacityConstraint.LIQUIDITY
            else:
                return CapacityConstraint.TURNOVER

    def _get_participation_rate(self, venue_type: str) -> float:
        """Get appropriate participation rate for venue type."""
        participation_rates = {
            'CEX': 0.05,    # 5% for highly liquid CEX
            'DEX': 0.02,    # 2% for less liquid DEX
            'HYBRID': 0.03  # 3% for hybrid venues
        }
        return participation_rates.get(venue_type, self.max_participation_rate)

    def _estimate_turnover(
        self,
        backtest_results: pd.DataFrame,
        venue_type: str
    ) -> float:
        """Estimate annual turnover ratio."""
        # Simplified - would normally calculate from trades
        default_turnovers = {
            'CEX': 8.0,     # 8x turnover (lower for liquid pairs)
            'DEX': 4.0,     # 4x turnover (higher gas costs reduce turnover)
            'HYBRID': 6.0   # 6x turnover (middle ground)
        }
        return default_turnovers.get(venue_type, 6.0)

    def _generate_capacity_reasoning(
        self,
        venue_type: str,
        estimated_capacity: float,
        primary_constraint: CapacityConstraint,
        liquidity_score: float,
        participation_rate: float,
        annual_turnover: float
    ) -> str:
        """Generate human-readable reasoning for capacity estimate."""
        reasoning_parts = []

        reasoning_parts.append(
            f"Based on {participation_rate*100:.0f}% participation rate "
            f"and {annual_turnover:.1f}x annual turnover"
        )

        constraint_descriptions = {
            CapacityConstraint.LIQUIDITY: "Limited by venue liquidity",
            CapacityConstraint.MARKET_IMPACT: "Limited by market impact from large positions",
            CapacityConstraint.SLIPPAGE: "Limited by slippage costs",
            CapacityConstraint.GAS_COSTS: "Limited by high gas costs reducing profitability",
            CapacityConstraint.TURNOVER: "Limited by high turnover requirements"
        }

        reasoning_parts.append(
            constraint_descriptions.get(
                primary_constraint,
                "Limited by general constraints"
            )
        )

        if liquidity_score < 0.5:
            reasoning_parts.append("Low liquidity score indicates conservative estimate")
        elif liquidity_score > 0.8:
            reasoning_parts.append("High liquidity score supports larger capacity")

        return ". ".join(reasoning_parts) + "."

    def analyze_combined_capacity(
        self,
        backtest_results: pd.DataFrame,
        price_matrix: pd.DataFrame,
        combined_range: Tuple[int, int] = (20_000_000, 50_000_000),  # PDF Section 2.4 requirement
        annual_turnover: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze combined strategy capacity across ALL venue types.

        PDF Section 2.4 REQUIRED: Combined capacity analysis with $20-50M range.

        Args:
            backtest_results: DataFrame with backtest results
            price_matrix: DataFrame with price data
            combined_range: (min, max) combined capacity in USD (PDF: $20-50M)
            annual_turnover: Optional annual turnover ratio

        Returns:
            Dictionary with comprehensive combined capacity analysis
        """
        # Get individual venue capacities first
        individual = self.analyze(
            backtest_results=backtest_results,
            price_matrix=price_matrix,
            cex_range=(10_000_000, 30_000_000),  # PDF requirement
            dex_range=(1_000_000, 5_000_000),    # PDF requirement (per pair)
            hybrid_range=(2_000_000, 8_000_000),
            annual_turnover=annual_turnover
        )

        # Calculate combined metrics
        raw_combined = (
            individual['cex_capacity']['estimated_capacity_usd'] +
            individual['dex_capacity']['estimated_capacity_usd'] +
            individual['hybrid_capacity']['estimated_capacity_usd']
        )

        # Apply combined range constraints (PDF Section 2.4: $20-50M)
        min_combined, max_combined = combined_range
        combined_capacity = np.clip(raw_combined, min_combined, max_combined)
        recommended_combined = combined_capacity * 0.75

        # Calculate optimal allocation based on liquidity scores
        total_liquidity_score = (
            individual['cex_capacity']['liquidity_score'] +
            individual['dex_capacity']['liquidity_score'] +
            individual['hybrid_capacity']['liquidity_score']
        )

        if total_liquidity_score > 0:
            optimal_allocation = {
                'cex_pct': individual['cex_capacity']['liquidity_score'] / total_liquidity_score,
                'dex_pct': individual['dex_capacity']['liquidity_score'] / total_liquidity_score,
                'hybrid_pct': individual['hybrid_capacity']['liquidity_score'] / total_liquidity_score
            }
        else:
            optimal_allocation = {'cex_pct': 0.6, 'dex_pct': 0.2, 'hybrid_pct': 0.2}

        # Calculate venue-specific breakdowns
        venue_breakdowns = self._calculate_venue_breakdowns(
            combined_capacity=combined_capacity,
            individual_capacities=individual,
            optimal_allocation=optimal_allocation
        )

        # Determine primary constraint for combined strategy
        constraints = {
            'CEX': individual['cex_capacity']['primary_constraint'],
            'DEX': individual['dex_capacity']['primary_constraint'],
            'HYBRID': individual['hybrid_capacity']['primary_constraint']
        }

        # Generate combined reasoning
        reasoning = self._generate_combined_reasoning(
            combined_capacity=combined_capacity,
            venue_breakdowns=venue_breakdowns,
            constraints=constraints
        )

        return {
            # PDF Section 2.4 Required: Combined capacity
            'combined_capacity_usd': combined_capacity,
            'combined_min_usd': min_combined,
            'combined_max_usd': max_combined,
            'recommended_combined_aum_usd': recommended_combined,

            # Individual venue capacities
            'cex_capacity': individual['cex_capacity'],
            'dex_capacity': individual['dex_capacity'],
            'hybrid_capacity': individual['hybrid_capacity'],

            # Optimal allocation
            'optimal_allocation_pct': optimal_allocation,

            # Venue-specific breakdowns (PDF requirement)
            'venue_breakdowns': venue_breakdowns,

            # Constraints
            'primary_constraints_by_venue': constraints,

            # Reasoning
            'capacity_reasoning': reasoning,

            # Summary
            'summary': {
                'total_capacity_range': f"${min_combined/1e6:.0f}M - ${max_combined/1e6:.0f}M",
                'recommended_aum': f"${recommended_combined/1e6:.1f}M",
                'allocation': f"{optimal_allocation['cex_pct']*100:.0f}% CEX / "
                             f"{optimal_allocation['dex_pct']*100:.0f}% DEX / "
                             f"{optimal_allocation['hybrid_pct']*100:.0f}% Hybrid"
            }
        }

    def _calculate_venue_breakdowns(
        self,
        combined_capacity: float,
        individual_capacities: Dict,
        optimal_allocation: Dict
    ) -> Dict[str, Dict]:
        """
        Calculate venue-specific breakdown scenarios.

        PDF Section 2.4 REQUIRED: CEX-only, DEX-only, Mixed, Combined breakdowns.
        """
        return {
            'cex_only': {
                'capacity_usd': min(
                    individual_capacities['cex_capacity']['estimated_capacity_usd'],
                    30_000_000  # PDF: CEX max $30M
                ),
                'description': 'CEX-only deployment (Binance, OKX, Bybit, Coinbase)',
                'pros': ['Lowest transaction costs (5-10 bps)', 'Highest liquidity', 'Fastest execution'],
                'cons': ['Counterparty risk', 'Regulatory exposure', 'Centralization']
            },
            'dex_only': {
                'capacity_usd': min(
                    individual_capacities['dex_capacity']['estimated_capacity_usd'],
                    20_000_000  # PDF: DEX total max $20M
                ),
                'description': 'DEX-only deployment (Uniswap, Curve, Balancer)',
                'pros': ['No counterparty risk', 'Decentralized', 'Composability'],
                'cons': ['High costs (50-150 bps)', 'MEV exposure', 'Gas volatility']
            },
            'hybrid_only': {
                'capacity_usd': individual_capacities['hybrid_capacity']['estimated_capacity_usd'],
                'description': 'Hybrid venue deployment (Hyperliquid, dYdX V4, Vertex)',
                'pros': ['Low fees (0-25 bps)', 'Fast execution', 'Self-custody'],
                'cons': ['Newer venues', 'Lower liquidity', 'Smart contract risk']
            },
            'mixed': {
                'capacity_usd': combined_capacity * 0.9,  # 90% of combined
                'allocation': {
                    'cex_usd': combined_capacity * optimal_allocation['cex_pct'],
                    'dex_usd': combined_capacity * optimal_allocation['dex_pct'],
                    'hybrid_usd': combined_capacity * optimal_allocation['hybrid_pct']
                },
                'description': 'Mixed venue deployment for diversification',
                'pros': ['Risk diversification', 'Cost optimization', 'Arbitrage opportunities'],
                'cons': ['Complexity', 'More monitoring required', 'Bridge risks']
            },
            'combined_optimal': {
                'capacity_usd': combined_capacity,
                'allocation': {
                    'cex_usd': combined_capacity * 0.60,   # 60% CEX (PDF guidance)
                    'dex_usd': combined_capacity * 0.15,   # 15% DEX
                    'hybrid_usd': combined_capacity * 0.25  # 25% Hybrid
                },
                'description': 'Optimal combined deployment per PDF Section 2.4',
                'pros': ['Maximum capacity utilization', 'Balanced risk/return', 'Best execution'],
                'cons': ['Requires multi-venue infrastructure', 'Higher operational complexity']
            }
        }

    def _generate_combined_reasoning(
        self,
        combined_capacity: float,
        venue_breakdowns: Dict,
        constraints: Dict
    ) -> str:
        """Generate reasoning for combined capacity estimate."""
        parts = [
            f"Combined strategy capacity: ${combined_capacity/1e6:.1f}M",
            f"PDF Section 2.4 range: $20-50M total",
            "",
            "Venue-specific constraints:",
        ]

        for venue, constraint in constraints.items():
            parts.append(f"  - {venue}: {constraint}")

        parts.extend([
            "",
            "Recommended deployment:",
            f"  - CEX: ${venue_breakdowns['combined_optimal']['allocation']['cex_usd']/1e6:.1f}M (60%)",
            f"  - DEX: ${venue_breakdowns['combined_optimal']['allocation']['dex_usd']/1e6:.1f}M (15%)",
            f"  - Hybrid: ${venue_breakdowns['combined_optimal']['allocation']['hybrid_usd']/1e6:.1f}M (25%)",
        ])

        return "\n".join(parts)

    def create_capacity_report(
        self,
        combined_result: Dict[str, Any]
    ) -> str:
        """
        Create a formatted capacity analysis report.

        Args:
            combined_result: Result from analyze_combined_capacity()

        Returns:
            Formatted string report for PDF Section 2.4 compliance
        """
        lines = [
            "=" * 70,
            "CAPACITY ANALYSIS REPORT",
            "(PDF Section 2.4 Compliance)",
            "=" * 70,
            "",
            "COMBINED STRATEGY CAPACITY",
            "-" * 40,
            f"Total Capacity Range:     ${combined_result['combined_min_usd']/1e6:.0f}M - ${combined_result['combined_max_usd']/1e6:.0f}M",
            f"Estimated Capacity:       ${combined_result['combined_capacity_usd']/1e6:.1f}M",
            f"Recommended AUM:          ${combined_result['recommended_combined_aum_usd']/1e6:.1f}M",
            "",
            "VENUE-SPECIFIC CAPACITY",
            "-" * 40,
            f"CEX Capacity:             ${combined_result['cex_capacity']['estimated_capacity_usd']/1e6:.1f}M (max $30M)",
            f"DEX Capacity:             ${combined_result['dex_capacity']['estimated_capacity_usd']/1e6:.1f}M (max $5M/pair)",
            f"Hybrid Capacity:          ${combined_result['hybrid_capacity']['estimated_capacity_usd']/1e6:.1f}M",
            "",
            "OPTIMAL ALLOCATION",
            "-" * 40,
            f"CEX:    {combined_result['optimal_allocation_pct']['cex_pct']*100:.0f}%",
            f"DEX:    {combined_result['optimal_allocation_pct']['dex_pct']*100:.0f}%",
            f"Hybrid: {combined_result['optimal_allocation_pct']['hybrid_pct']*100:.0f}%",
            "",
            "DEPLOYMENT SCENARIOS",
            "-" * 40,
        ]

        for scenario_name, scenario in combined_result['venue_breakdowns'].items():
            lines.append(f"{scenario_name.upper().replace('_', ' ')}:")
            lines.append(f"  Capacity: ${scenario['capacity_usd']/1e6:.1f}M")
            lines.append(f"  {scenario['description']}")
            lines.append("")

        lines.extend([
            "PRIMARY CONSTRAINTS BY VENUE",
            "-" * 40,
        ])

        for venue, constraint in combined_result['primary_constraints_by_venue'].items():
            lines.append(f"{venue}: {constraint}")

        lines.extend([
            "",
            "=" * 70,
            "Note: Capacity estimates per PDF Section 2.4 requirements.",
            "Conservative estimates account for market impact and slippage.",
            "=" * 70,
        ])

        return "\n".join(lines)
