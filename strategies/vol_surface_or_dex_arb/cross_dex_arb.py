"""
Cross-DEX Arbitrage Strategies
Implements strategies to exploit price differences across decentralized exchanges.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Chain(Enum):
    """Blockchain network enumeration."""
    ETHEREUM = "ethereum"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    POLYGON = "polygon"
    BASE = "base"
    SOLANA = "solana"


class DEXType(Enum):
    """DEX type enumeration."""
    AMM = "amm"  # Uniswap V2 style
    CONCENTRATED = "concentrated"  # Uniswap V3 style
    CURVE = "curve"  # Curve style stable swaps
    ORDER_BOOK = "order_book"  # Hybrid DEX


@dataclass
class DEXPool:
    """Represents a DEX liquidity pool."""
    dex_name: str
    chain: Chain
    dex_type: DEXType
    token0: str
    token1: str
    reserve0: float
    reserve1: float
    fee_tier: float  # e.g., 0.003 for 0.3%
    tvl_usd: float
    volume_24h: float
    price: float  # token1 per token0


@dataclass
class ArbOpportunity:
    """Represents an arbitrage opportunity."""
    buy_dex: str
    buy_chain: Chain
    sell_dex: str
    sell_chain: Chain
    token_pair: str
    buy_price: float
    sell_price: float
    spread_pct: float
    expected_profit_pct: float
    gas_cost_usd: float
    min_size_usd: float
    max_size_usd: float  # Limited by liquidity


# Gas costs by chain (approximate, in USD)
GAS_COSTS = {
    Chain.ETHEREUM: 15.0,
    Chain.ARBITRUM: 0.50,
    Chain.OPTIMISM: 0.30,
    Chain.POLYGON: 0.05,
    Chain.BASE: 0.20,
    Chain.SOLANA: 0.01,
}

# Bridge costs and times
BRIDGE_COSTS = {
    (Chain.ETHEREUM, Chain.ARBITRUM): {"cost_usd": 5.0, "time_minutes": 10},
    (Chain.ETHEREUM, Chain.OPTIMISM): {"cost_usd": 5.0, "time_minutes": 10},
    (Chain.ETHEREUM, Chain.POLYGON): {"cost_usd": 10.0, "time_minutes": 30},
    (Chain.ARBITRUM, Chain.OPTIMISM): {"cost_usd": 2.0, "time_minutes": 5},
}


class CrossDEXArbitrage:
    """
    Cross-DEX arbitrage strategy.

    Exploits price discrepancies across:
    1. Same-chain DEXs (Uniswap vs Sushiswap on same chain)
    2. Cross-chain DEXs (Uniswap Arbitrum vs Uniswap Optimism)
    3. CEX-DEX arbitrage (Binance vs Uniswap)

    Key considerations:
    - Gas costs vary significantly by chain
    - MEV risk (sandwich attacks)
    - Bridge costs and delays for cross-chain
    - Slippage based on liquidity depth
    """

    def __init__(
        self,
        min_spread_pct: float = 0.005,  # 0.5% minimum spread
        max_slippage_pct: float = 0.01,  # 1% max slippage
        min_liquidity_usd: float = 100_000,
        use_flashbots: bool = True,  # MEV protection
        max_trade_size_usd: float = 50_000
    ):
        self.min_spread_pct = min_spread_pct
        self.max_slippage_pct = max_slippage_pct
        self.min_liquidity_usd = min_liquidity_usd
        self.use_flashbots = use_flashbots
        self.max_trade_size_usd = max_trade_size_usd
        self.opportunities: List[ArbOpportunity] = []
        self.trade_history: List[Dict] = []

    def calculate_slippage(
        self,
        pool: DEXPool,
        trade_size_usd: float,
        is_buy: bool
    ) -> float:
        """
        Calculate expected slippage for a trade.

        Uses constant product formula for AMMs.
        """
        if pool.dex_type == DEXType.AMM:
            # x * y = k
            # For buying token0 with token1:
            # (x - dx) * (y + dy) = k
            # Price impact = dy / y (approximately)
            trade_size_in_reserve = trade_size_usd / pool.tvl_usd * 2
            slippage = trade_size_in_reserve  # Simplified approximation

        elif pool.dex_type == DEXType.CONCENTRATED:
            # Uniswap V3 has better capital efficiency
            # Slippage is lower for same TVL
            trade_size_in_reserve = trade_size_usd / pool.tvl_usd
            slippage = trade_size_in_reserve * 0.5  # Better efficiency

        elif pool.dex_type == DEXType.CURVE:
            # Curve has very low slippage for stable pairs
            slippage = trade_size_usd / pool.tvl_usd * 0.1

        else:
            slippage = trade_size_usd / pool.tvl_usd

        return min(slippage, 0.1)  # Cap at 10%

    def calculate_gas_cost(
        self,
        chain: Chain,
        is_complex_swap: bool = False
    ) -> float:
        """Calculate gas cost for a swap on given chain."""
        base_cost = GAS_COSTS.get(chain, 10.0)
        if is_complex_swap:
            base_cost *= 1.5  # Complex routing costs more
        return base_cost

    def calculate_bridge_cost(
        self,
        from_chain: Chain,
        to_chain: Chain
    ) -> Tuple[float, int]:
        """
        Calculate bridge cost and time.

        Returns (cost_usd, time_minutes)
        """
        key = (from_chain, to_chain)
        reverse_key = (to_chain, from_chain)

        if key in BRIDGE_COSTS:
            return BRIDGE_COSTS[key]["cost_usd"], BRIDGE_COSTS[key]["time_minutes"]
        elif reverse_key in BRIDGE_COSTS:
            return BRIDGE_COSTS[reverse_key]["cost_usd"], BRIDGE_COSTS[reverse_key]["time_minutes"]
        else:
            return 20.0, 60  # Default high cost for unknown routes

    def find_same_chain_opportunities(
        self,
        pools: List[DEXPool]
    ) -> List[ArbOpportunity]:
        """Find arbitrage opportunities on the same chain."""
        opportunities = []

        # Group pools by chain and token pair
        pools_by_chain = {}
        for pool in pools:
            if pool.tvl_usd < self.min_liquidity_usd:
                continue

            key = (pool.chain, f"{pool.token0}/{pool.token1}")
            if key not in pools_by_chain:
                pools_by_chain[key] = []
            pools_by_chain[key].append(pool)

        # Compare prices within each group
        for (chain, pair), chain_pools in pools_by_chain.items():
            if len(chain_pools) < 2:
                continue

            # Sort by price
            sorted_pools = sorted(chain_pools, key=lambda p: p.price)
            cheapest = sorted_pools[0]
            most_expensive = sorted_pools[-1]

            spread_pct = (most_expensive.price - cheapest.price) / cheapest.price

            if spread_pct < self.min_spread_pct:
                continue

            # Calculate costs
            gas_cost = self.calculate_gas_cost(chain) * 2  # Buy + sell

            # Calculate max size based on liquidity
            max_size = min(
                cheapest.tvl_usd * 0.1,  # 10% of smaller pool
                most_expensive.tvl_usd * 0.1,
                self.max_trade_size_usd
            )

            # Calculate expected slippage
            slippage_buy = self.calculate_slippage(cheapest, max_size, True)
            slippage_sell = self.calculate_slippage(most_expensive, max_size, False)
            total_slippage = slippage_buy + slippage_sell

            # Net profit after costs
            fee_cost = cheapest.fee_tier + most_expensive.fee_tier
            gas_cost_pct = gas_cost / max_size if max_size > 0 else 1

            expected_profit_pct = spread_pct - total_slippage - fee_cost - gas_cost_pct

            if expected_profit_pct > 0:
                opportunities.append(ArbOpportunity(
                    buy_dex=cheapest.dex_name,
                    buy_chain=chain,
                    sell_dex=most_expensive.dex_name,
                    sell_chain=chain,
                    token_pair=pair,
                    buy_price=cheapest.price,
                    sell_price=most_expensive.price,
                    spread_pct=spread_pct,
                    expected_profit_pct=expected_profit_pct,
                    gas_cost_usd=gas_cost,
                    min_size_usd=gas_cost / expected_profit_pct if expected_profit_pct > 0 else float('inf'),
                    max_size_usd=max_size
                ))

        return opportunities

    def find_cross_chain_opportunities(
        self,
        pools: List[DEXPool]
    ) -> List[ArbOpportunity]:
        """Find arbitrage opportunities across chains."""
        opportunities = []

        # Group pools by token pair (across all chains)
        pools_by_pair = {}
        for pool in pools:
            if pool.tvl_usd < self.min_liquidity_usd:
                continue

            pair = f"{pool.token0}/{pool.token1}"
            if pair not in pools_by_pair:
                pools_by_pair[pair] = []
            pools_by_pair[pair].append(pool)

        # Compare prices across chains
        for pair, pair_pools in pools_by_pair.items():
            if len(pair_pools) < 2:
                continue

            # Sort by price
            sorted_pools = sorted(pair_pools, key=lambda p: p.price)
            cheapest = sorted_pools[0]
            most_expensive = sorted_pools[-1]

            # Skip if same chain (handled by same_chain function)
            if cheapest.chain == most_expensive.chain:
                continue

            spread_pct = (most_expensive.price - cheapest.price) / cheapest.price

            if spread_pct < self.min_spread_pct * 2:  # Higher threshold for cross-chain
                continue

            # Calculate costs including bridging
            gas_buy = self.calculate_gas_cost(cheapest.chain)
            gas_sell = self.calculate_gas_cost(most_expensive.chain)
            bridge_cost, bridge_time = self.calculate_bridge_cost(
                cheapest.chain, most_expensive.chain
            )

            total_gas = gas_buy + gas_sell + bridge_cost

            # Max size limited by both pools
            max_size = min(
                cheapest.tvl_usd * 0.05,  # 5% for cross-chain (more conservative)
                most_expensive.tvl_usd * 0.05,
                self.max_trade_size_usd
            )

            # Slippage
            slippage_buy = self.calculate_slippage(cheapest, max_size, True)
            slippage_sell = self.calculate_slippage(most_expensive, max_size, False)
            total_slippage = slippage_buy + slippage_sell

            # Price risk during bridge time
            # Assume 0.5% price risk per 10 minutes for volatile assets
            price_risk = (bridge_time / 10) * 0.005

            # Net profit
            fee_cost = cheapest.fee_tier + most_expensive.fee_tier
            gas_cost_pct = total_gas / max_size if max_size > 0 else 1

            expected_profit_pct = spread_pct - total_slippage - fee_cost - gas_cost_pct - price_risk

            if expected_profit_pct > 0:
                opportunities.append(ArbOpportunity(
                    buy_dex=cheapest.dex_name,
                    buy_chain=cheapest.chain,
                    sell_dex=most_expensive.dex_name,
                    sell_chain=most_expensive.chain,
                    token_pair=pair,
                    buy_price=cheapest.price,
                    sell_price=most_expensive.price,
                    spread_pct=spread_pct,
                    expected_profit_pct=expected_profit_pct,
                    gas_cost_usd=total_gas,
                    min_size_usd=total_gas / expected_profit_pct if expected_profit_pct > 0 else float('inf'),
                    max_size_usd=max_size
                ))

        return opportunities

    def scan_for_opportunities(
        self,
        pools: List[DEXPool]
    ) -> Dict[str, List[ArbOpportunity]]:
        """
        Comprehensive scan for all arbitrage opportunities.

        Returns dict with opportunities by type.
        """
        same_chain = self.find_same_chain_opportunities(pools)
        cross_chain = self.find_cross_chain_opportunities(pools)

        # Sort by expected profit
        same_chain.sort(key=lambda x: x.expected_profit_pct, reverse=True)
        cross_chain.sort(key=lambda x: x.expected_profit_pct, reverse=True)

        return {
            "same_chain": same_chain,
            "cross_chain": cross_chain,
            "total_opportunities": len(same_chain) + len(cross_chain),
            "best_same_chain": same_chain[0] if same_chain else None,
            "best_cross_chain": cross_chain[0] if cross_chain else None
        }

    def estimate_daily_returns(
        self,
        opportunities: List[ArbOpportunity],
        trades_per_day: int = 10,
        capital: float = 100_000
    ) -> Dict[str, float]:
        """
        Estimate daily returns from arbitrage opportunities.

        Assumes we can execute a subset of opportunities per day.
        """
        if not opportunities:
            return {"daily_return": 0, "annual_return": 0, "sharpe_estimate": 0}

        # Take top opportunities
        top_opps = sorted(opportunities, key=lambda x: x.expected_profit_pct, reverse=True)[:trades_per_day]

        total_profit = 0
        total_traded = 0

        for opp in top_opps:
            trade_size = min(opp.max_size_usd, capital / trades_per_day)
            profit = trade_size * opp.expected_profit_pct
            total_profit += profit
            total_traded += trade_size

        daily_return = total_profit / capital if capital > 0 else 0
        annual_return = daily_return * 365

        # Rough Sharpe estimate (assuming 50% win rate variance)
        daily_vol = daily_return * 0.5
        sharpe = (daily_return * 365) / (daily_vol * np.sqrt(365)) if daily_vol > 0 else 0

        return {
            "daily_return": daily_return,
            "annual_return": annual_return,
            "sharpe_estimate": sharpe,
            "avg_trade_size": total_traded / len(top_opps) if top_opps else 0,
            "num_opportunities": len(top_opps)
        }
