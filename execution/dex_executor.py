"""
DEX Executor
=============

Decentralized exchange execution with MEV protection and gas optimization.

Handles on-chain trade execution across Uniswap V3, Curve, and DEX
aggregators (1inch, 0x, CowSwap) with realistic cost modeling.

Fee Schedule (per PDF Section 2.4):
    - Swap fee: 0.05-1.00% depending on pool tier
    - Slippage: 0.10-0.50% for less liquid tokens
    - MEV tax: ~0.05-0.10% (sandwich attacks, front-running)
    - Gas costs: $0.50-50 per swap depending on chain
    - Total DEX cost: 0.50-1.50% per trade all-in

Chain-Specific Gas Costs:
    - Ethereum mainnet: $10-50 per swap (often prohibitive)
    - Arbitrum: $0.50-2 per swap (feasible)
    - Optimism: similar to Arbitrum
    - Polygon: $0.05-0.20 per swap
    - Solana: $0.01-0.05 per swap

Author: Tamer Atesyakar
Version: 1.0.0
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class Chain(Enum):
    ETHEREUM = 'ethereum'
    ARBITRUM = 'arbitrum'
    OPTIMISM = 'optimism'
    POLYGON = 'polygon'
    SOLANA = 'solana'
    BASE = 'base'


class DEXProtocol(Enum):
    UNISWAP_V3 = 'uniswap_v3'
    UNISWAP_V2 = 'uniswap_v2'
    CURVE = 'curve'
    SUSHISWAP = 'sushiswap'
    ONEINCH = '1inch'
    COWSWAP = 'cowswap'
    ZEROX = '0x'


class MEVProtection(Enum):
    NONE = 'none'
    PRIVATE_RPC = 'private_rpc'
    COWSWAP_BATCH = 'cowswap_batch'
    FLASHBOTS = 'flashbots'


@dataclass
class GasCostModel:
    """Gas cost parameters by chain."""
    chain: Chain
    avg_gas_usd: float
    max_gas_usd: float
    min_gas_usd: float


# Per-chain gas cost models
GAS_COSTS: Dict[Chain, GasCostModel] = {
    Chain.ETHEREUM: GasCostModel(Chain.ETHEREUM, avg_gas_usd=25.0, max_gas_usd=50.0, min_gas_usd=10.0),
    Chain.ARBITRUM: GasCostModel(Chain.ARBITRUM, avg_gas_usd=1.0, max_gas_usd=2.0, min_gas_usd=0.50),
    Chain.OPTIMISM: GasCostModel(Chain.OPTIMISM, avg_gas_usd=1.0, max_gas_usd=2.0, min_gas_usd=0.50),
    Chain.POLYGON: GasCostModel(Chain.POLYGON, avg_gas_usd=0.10, max_gas_usd=0.20, min_gas_usd=0.05),
    Chain.SOLANA: GasCostModel(Chain.SOLANA, avg_gas_usd=0.03, max_gas_usd=0.05, min_gas_usd=0.01),
    Chain.BASE: GasCostModel(Chain.BASE, avg_gas_usd=0.80, max_gas_usd=1.50, min_gas_usd=0.30),
}


@dataclass
class DEXFeeSchedule:
    """DEX fee parameters for a specific pool."""
    swap_fee_bps: float = 30.0      # 0.30% (Uniswap V3 standard tier)
    slippage_bps: float = 25.0      # 0.25%
    mev_tax_bps: float = 7.5        # ~0.075%
    gas_cost_usd: float = 1.0       # Chain-dependent

    @property
    def total_cost_bps(self) -> float:
        """Total non-gas cost in basis points."""
        return self.swap_fee_bps + self.slippage_bps + self.mev_tax_bps

    def total_cost_usd(self, trade_size_usd: float) -> float:
        """Total cost including gas for a given trade size."""
        bps_cost = trade_size_usd * (self.total_cost_bps / 10_000)
        return bps_cost + self.gas_cost_usd


@dataclass
class DEXTradeResult:
    """Result of a single DEX swap."""
    tx_hash: str
    protocol: DEXProtocol
    chain: Chain
    token_in: str
    token_out: str
    amount_in_usd: float
    amount_out_usd: float
    swap_fee_usd: float
    gas_cost_usd: float
    slippage_bps: float
    mev_cost_usd: float
    effective_price: float
    timestamp: Optional[pd.Timestamp] = None

    @property
    def total_cost_usd(self) -> float:
        return self.swap_fee_usd + self.gas_cost_usd + self.mev_cost_usd


@dataclass
class DEXPairTradeResult:
    """Combined result of a DEX pairs trade."""
    pair_id: str
    long_result: DEXTradeResult
    short_result: DEXTradeResult
    total_cost_usd: float = 0.0
    total_cost_bps: float = 0.0


# ---------------------------------------------------------------------------
# DEX Executor
# ---------------------------------------------------------------------------

class DEXExecutor:
    """
    Decentralized exchange execution engine.

    Handles on-chain trade routing with MEV protection, gas optimization,
    and realistic cost modeling for DEX-based pairs trading.

    Parameters
    ----------
    chain : Chain
        Target blockchain (Arbitrum recommended for cost efficiency).
    protocol : DEXProtocol
        Primary DEX protocol for execution.
    mev_protection : MEVProtection
        MEV protection strategy.
    max_position_usd : float
        Maximum position size (PDF: $5k-$50k for DEX).
    min_trade_usd : float
        Minimum trade size to justify gas costs (PDF: $5,000).

    Example
    -------
        >>> executor = DEXExecutor(chain=Chain.ARBITRUM)
        >>> result = executor.execute_swap(
        ...     token_in='USDC', token_out='UNI',
        ...     amount_usd=10_000, price=12.5
        ... )
    """

    def __init__(
        self,
        chain: Chain = Chain.ARBITRUM,
        protocol: DEXProtocol = DEXProtocol.UNISWAP_V3,
        mev_protection: MEVProtection = MEVProtection.PRIVATE_RPC,
        max_position_usd: float = 50_000,
        min_trade_usd: float = 5_000,
    ):
        self.chain = chain
        self.protocol = protocol
        self.mev_protection = mev_protection
        self.max_position_usd = max_position_usd
        self.min_trade_usd = min_trade_usd
        self.gas_model = GAS_COSTS[chain]
        self._trade_counter = 0

    def execute_swap(
        self,
        token_in: str,
        token_out: str,
        amount_usd: float,
        price: float,
        pool_fee_tier: int = 3000,
    ) -> DEXTradeResult:
        """
        Execute a single DEX swap.

        Parameters
        ----------
        token_in : str
            Input token symbol.
        token_out : str
            Output token symbol.
        amount_usd : float
            Trade size in USD.
        price : float
            Reference price of output token.
        pool_fee_tier : int
            Uniswap V3 fee tier in hundredths of bps (500, 3000, 10000).

        Returns
        -------
        DEXTradeResult
        """
        self._trade_counter += 1

        if amount_usd < self.min_trade_usd:
            logger.warning(
                "Trade size $%.0f below minimum $%.0f for gas justification",
                amount_usd, self.min_trade_usd
            )

        amount_usd = min(amount_usd, self.max_position_usd)

        # Fee calculation
        swap_fee_bps = pool_fee_tier / 100  # e.g. 3000 -> 30 bps
        swap_fee_usd = amount_usd * (swap_fee_bps / 10_000)

        # Slippage (higher for DEX)
        slippage_bps = np.random.uniform(10, 50)
        slippage_usd = amount_usd * (slippage_bps / 10_000)

        # MEV cost (reduced with protection)
        mev_multiplier = {
            MEVProtection.NONE: 1.0,
            MEVProtection.PRIVATE_RPC: 0.3,
            MEVProtection.COWSWAP_BATCH: 0.1,
            MEVProtection.FLASHBOTS: 0.2,
        }[self.mev_protection]
        mev_cost_usd = amount_usd * (7.5 / 10_000) * mev_multiplier

        # Gas cost
        gas_cost_usd = self.gas_model.avg_gas_usd

        # Effective amount received
        amount_out_usd = amount_usd - swap_fee_usd - slippage_usd - mev_cost_usd
        effective_price = price * (amount_out_usd / amount_usd)

        return DEXTradeResult(
            tx_hash=f"0x{self._trade_counter:064x}",
            protocol=self.protocol,
            chain=self.chain,
            token_in=token_in,
            token_out=token_out,
            amount_in_usd=amount_usd,
            amount_out_usd=amount_out_usd,
            swap_fee_usd=swap_fee_usd,
            gas_cost_usd=gas_cost_usd,
            slippage_bps=slippage_bps,
            mev_cost_usd=mev_cost_usd,
            effective_price=effective_price,
            timestamp=pd.Timestamp.now(tz='UTC'),
        )

    def execute_pair_trade(
        self,
        long_token: str,
        short_token: str,
        size_usd: float,
        prices: Dict[str, float],
        hedge_ratio: float = 1.0,
    ) -> DEXPairTradeResult:
        """
        Execute a DEX-based pairs trade.

        Routes through USDC as intermediate (Token A -> USDC -> Token B).
        """
        long_result = self.execute_swap(
            token_in='USDC',
            token_out=long_token,
            amount_usd=size_usd,
            price=prices.get(long_token, 0.0),
        )

        short_size = size_usd * hedge_ratio
        short_result = self.execute_swap(
            token_in=short_token,
            token_out='USDC',
            amount_usd=short_size,
            price=prices.get(short_token, 0.0),
        )

        total_cost = long_result.total_cost_usd + short_result.total_cost_usd
        notional = size_usd + short_size
        total_bps = (total_cost / notional * 10_000) if notional > 0 else 0

        pair_id = f"DEX-{long_token}_{short_token}_{int(time.time())}"

        return DEXPairTradeResult(
            pair_id=pair_id,
            long_result=long_result,
            short_result=short_result,
            total_cost_usd=total_cost,
            total_cost_bps=total_bps,
        )

    def estimate_costs(
        self, size_usd: float, pool_fee_tier: int = 3000
    ) -> Dict[str, float]:
        """
        Estimate total costs for a DEX pair trade.

        Returns
        -------
        dict
            Breakdown: swap_fee, gas, slippage, mev, total.
        """
        swap_fee_bps = pool_fee_tier / 100
        swap_fee = size_usd * 2 * (swap_fee_bps / 10_000)
        gas = self.gas_model.avg_gas_usd * 4  # 4 swaps for pair trade
        slippage = size_usd * 2 * (25 / 10_000)
        mev = size_usd * 2 * (5 / 10_000)
        total = swap_fee + gas + slippage + mev

        return {
            'swap_fee_usd': swap_fee,
            'gas_cost_usd': gas,
            'slippage_usd': slippage,
            'mev_cost_usd': mev,
            'total_cost_usd': total,
            'total_cost_bps': (total / (size_usd * 2)) * 10_000 if size_usd > 0 else 0,
            'chain': self.chain.value,
            'min_profitable_spread_bps': (total / (size_usd * 2)) * 10_000 * 2 if size_usd > 0 else 0,
        }

    def is_trade_viable(self, size_usd: float, expected_profit_bps: float) -> bool:
        """Check if a trade is worth executing given expected profit."""
        costs = self.estimate_costs(size_usd)
        return expected_profit_bps > costs['total_cost_bps']
