"""
Universe Construction for Crypto Statistical Arbitrage
=======================================================

Comprehensive module for constructing and managing token universes
across CEX (Centralized Exchange), DEX (Decentralized Exchange), and 
hybrid venues for pairs trading strategies.

Mathematical Framework
----------------------
Universe Quality Score:

    Q_universe = Σ w_i × q_i
    
    Where:
        q_liquidity = log(volume) / log(max_volume)
        q_depth = min(1, depth_2pct / target_depth)
        q_stability = 1 - (zero_volume_days / total_days)
        q_spread = 1 - (avg_spread / max_spread)

Pair Selection Scoring:

    S_pair = α × correlation + β × sector_match + γ × liquidity_score
           + δ × tier_score - ε × execution_cost
    
    Where α + β + γ + δ - ε = 1

Wash Trading Detection (DEX):

    P(wash) = sigmoid(volume/TVL - threshold)
    
    If volume/TVL > 10: High wash trading probability

Survivorship Bias Adjustment:

    R_adjusted = R_raw - Σ (R_delisted × weight_delisted)

Key Features
------------
1. Multi-Venue Universe Construction:
   - CEX: Binance, Bybit, OKX, Coinbase, Kraken
   - Hybrid: Hyperliquid, dYdX V4, Vertex
   - DEX: Uniswap V3, Curve, SushiSwap, Balancer

2. Token Filtering Pipeline:
   - Liquidity thresholds (volume, TVL, depth)
   - Stablecoin/Wrapped/Leveraged exclusion
   - Wash trading detection
   - Survivorship bias tracking

3. Sector Classification:
   - L1/L2 Blockchains
   - DeFi (Lending, DEX, Derivatives, Yield)
   - Gaming/Metaverse
   - Infrastructure (Oracles, Storage)
   - Meme, AI, Privacy, RWA

4. Pair Candidate Generation:
   - Same-sector preference
   - Cross-venue opportunities
   - Tiered allocation
   - Cost-adjusted scoring

Author: Crypto StatArb Quantitative Research
Version: 2.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import json
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS WITH TRADING-SPECIFIC PROPERTIES
# =============================================================================

class VenueType(Enum):
    """
    Classification of trading venue types with execution characteristics.
    
    Each venue type has specific properties affecting position sizing,
    execution costs, and risk management parameters.
    """
    CEX = "CEX"
    HYBRID = "HYBRID"
    DEX = "DEX"
    
    @property
    def description(self) -> str:
        """Venue type description."""
        descriptions = {
            self.CEX: "Centralized Exchange - High liquidity, standard execution",
            self.HYBRID: "Hybrid Venue - Decentralized orderbook, funding rates",
            self.DEX: "Decentralized Exchange - AMM-based, variable liquidity",
        }
        return descriptions.get(self, "Unknown venue type")
    
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
    def max_allocation_pct(self) -> float:
        """Maximum portfolio allocation percentage."""
        allocations = {
            self.CEX: 0.70,
            self.HYBRID: 0.30,
            self.DEX: 0.30,
        }
        return allocations.get(self, 0.20)
    
    @property
    def max_position_usd(self) -> float:
        """Maximum single position in USD."""
        limits = {
            self.CEX: 500_000,
            self.HYBRID: 200_000,
            self.DEX: 100_000,
        }
        return limits.get(self, 50_000)
    
    @property
    def min_position_usd(self) -> float:
        """Minimum position size in USD."""
        limits = {
            self.CEX: 1_000,
            self.HYBRID: 2_000,
            self.DEX: 5_000,
        }
        return limits.get(self, 1_000)
    
    @property
    def typical_slippage_bps(self) -> float:
        """Typical slippage in basis points."""
        slippage = {
            self.CEX: 2.0,
            self.HYBRID: 5.0,
            self.DEX: 15.0,
        }
        return slippage.get(self, 10.0)
    
    @property
    def typical_fee_bps(self) -> float:
        """Typical trading fee in basis points (one-way)."""
        fees = {
            self.CEX: 5.0,      # 0.05% taker
            self.HYBRID: 3.0,   # 0.03% typical
            self.DEX: 30.0,     # 0.30% swap fee
        }
        return fees.get(self, 10.0)
    
    @property
    def round_trip_cost_bps(self) -> float:
        """Round trip cost in basis points (4 legs for pairs trade)."""
        costs = {
            self.CEX: 20.0,     # 4 × 5bps
            self.HYBRID: 12.0,  # 4 × 3bps
            self.DEX: 150.0,    # 4 × 30bps + gas
        }
        return costs.get(self, 50.0)
    
    @property
    def has_gas_costs(self) -> bool:
        """True if venue has gas costs."""
        return self in [self.DEX, self.HYBRID]
    
    @property
    def typical_gas_usd(self) -> float:
        """Typical gas cost per transaction in USD."""
        gas = {
            self.CEX: 0.0,
            self.HYBRID: 0.50,  # L2 gas
            self.DEX: 25.0,     # Varies by chain
        }
        return gas.get(self, 0.0)
    
    @property
    def has_funding_rate(self) -> bool:
        """True if venue has perpetual funding rates."""
        return self in [self.CEX, self.HYBRID]
    
    @property
    def execution_risk(self) -> str:
        """Execution risk level."""
        risk = {
            self.CEX: "low",
            self.HYBRID: "moderate",
            self.DEX: "high",
        }
        return risk.get(self, "moderate")
    
    @property
    def max_positions(self) -> int:
        """Maximum concurrent positions."""
        limits = {
            self.CEX: 15,
            self.HYBRID: 8,
            self.DEX: 5,
        }
        return limits.get(self, 5)
    
    @property
    def recommended_entry_z(self) -> float:
        """Recommended z-score entry threshold."""
        thresholds = {
            self.CEX: 2.0,
            self.HYBRID: 2.2,
            self.DEX: 2.5,
        }
        return thresholds.get(self, 2.0)
    
    @property
    def recommended_exit_z(self) -> float:
        """Recommended z-score exit threshold."""
        thresholds = {
            self.CEX: 0.0,
            self.HYBRID: 0.25,
            self.DEX: 1.0,
        }
        return thresholds.get(self, 0.25)
    
    @property
    def liquidity_threshold_usd(self) -> float:
        """Minimum liquidity threshold in USD."""
        thresholds = {
            self.CEX: 10_000_000,
            self.HYBRID: 5_000_000,
            self.DEX: 500_000,
        }
        return thresholds.get(self, 1_000_000)


class TokenTier(Enum):
    """
    Token tier classification based on liquidity and venue availability.
    
    Tier 1: High liquidity, available on major CEX (70% allocation)
    Tier 2: Medium liquidity, mixed CEX/DEX (25% allocation)
    Tier 3: Lower liquidity, DEX-only, speculative (5% allocation)
    """
    TIER_1 = 1
    TIER_2 = 2
    TIER_3 = 3
    
    @classmethod
    def from_metrics(
        cls,
        volume_usd: float,
        mcap_usd: float,
        tvl_usd: float = 0.0,
        venue_type: VenueType = VenueType.CEX
    ) -> 'TokenTier':
        """
        Classify tier from liquidity metrics.
        
        Args:
            volume_usd: 24h volume in USD
            mcap_usd: Market cap in USD
            tvl_usd: TVL in USD (for DEX tokens)
            venue_type: Primary venue type
        """
        if venue_type == VenueType.CEX:
            if volume_usd >= 50_000_000 and mcap_usd >= 500_000_000:
                return cls.TIER_1
            elif volume_usd >= 5_000_000 and mcap_usd >= 50_000_000:
                return cls.TIER_2
            return cls.TIER_3
        elif venue_type == VenueType.HYBRID:
            if volume_usd >= 20_000_000:
                return cls.TIER_1
            elif volume_usd >= 2_000_000:
                return cls.TIER_2
            return cls.TIER_3
        else:  # DEX
            if tvl_usd >= 10_000_000 and volume_usd >= 5_000_000:
                return cls.TIER_1
            elif tvl_usd >= 1_000_000 and volume_usd >= 500_000:
                return cls.TIER_2
            return cls.TIER_3
    
    @property
    def description(self) -> str:
        """Tier description."""
        descriptions = {
            self.TIER_1: "High liquidity, major CEX availability",
            self.TIER_2: "Medium liquidity, mixed venue availability",
            self.TIER_3: "Lower liquidity, speculative",
        }
        return descriptions.get(self, "Unknown tier")
    
    @property
    def allocation_pct(self) -> float:
        """Target allocation percentage."""
        allocations = {
            self.TIER_1: 0.70,
            self.TIER_2: 0.25,
            self.TIER_3: 0.05,
        }
        return allocations.get(self, 0.05)
    
    @property
    def position_multiplier(self) -> float:
        """Position size multiplier."""
        multipliers = {
            self.TIER_1: 1.0,
            self.TIER_2: 0.7,
            self.TIER_3: 0.4,
        }
        return multipliers.get(self, 0.4)
    
    @property
    def max_position_usd(self) -> float:
        """Maximum position size in USD."""
        limits = {
            self.TIER_1: 500_000,
            self.TIER_2: 200_000,
            self.TIER_3: 50_000,
        }
        return limits.get(self, 50_000)
    
    @property
    def min_volume_usd(self) -> float:
        """Minimum required daily volume."""
        volumes = {
            self.TIER_1: 50_000_000,
            self.TIER_2: 5_000_000,
            self.TIER_3: 100_000,
        }
        return volumes.get(self, 100_000)
    
    @property
    def risk_weight(self) -> float:
        """Risk weight for portfolio construction."""
        weights = {
            self.TIER_1: 1.0,
            self.TIER_2: 1.5,
            self.TIER_3: 2.5,
        }
        return weights.get(self, 2.5)
    
    @property
    def monitoring_frequency_hours(self) -> int:
        """How often to monitor positions."""
        hours = {
            self.TIER_1: 24,
            self.TIER_2: 8,
            self.TIER_3: 4,
        }
        return hours.get(self, 4)
    
    @property
    def max_holding_days(self) -> int:
        """Maximum holding period."""
        days = {
            self.TIER_1: 30,
            self.TIER_2: 21,
            self.TIER_3: 14,
        }
        return days.get(self, 14)


class TokenSector(Enum):
    """
    Token sector classification for pairs selection.
    
    Pairs within the same sector tend to have higher correlation,
    making them better candidates for statistical arbitrage.
    """
    # Layer 1 blockchains
    L1 = "L1"
    L1_EVM = "L1_EVM"
    L1_NON_EVM = "L1_NON_EVM"
    
    # Layer 2 scaling solutions
    L2 = "L2"
    L2_OPTIMISTIC = "L2_Optimistic"
    L2_ZK = "L2_ZK"
    
    # Decentralized Finance
    DEFI = "DeFi"
    DEFI_LENDING = "DeFi_Lending"
    DEFI_DEX = "DeFi_DEX"
    DEFI_DERIVATIVES = "DeFi_Derivatives"
    DEFI_YIELD = "DeFi_Yield"
    DEFI_STABLECOIN = "DeFi_Stablecoin"
    
    # Liquid Staking
    LST = "Liquid_Staking"
    
    # Meme tokens
    MEME = "Meme"
    MEME_DOG = "Meme_Dog"
    MEME_CAT = "Meme_Cat"
    
    # AI/ML tokens
    AI = "AI"
    AI_COMPUTE = "AI_Compute"
    AI_DATA = "AI_Data"
    
    # Gaming/Metaverse
    GAMING = "Gaming"
    GAMING_INFRA = "Gaming_Infra"
    METAVERSE = "Metaverse"
    
    # Infrastructure
    INFRA = "Infrastructure"
    ORACLE = "Oracle"
    STORAGE = "Storage"
    INDEXING = "Indexing"
    INTEROP = "Interoperability"
    
    # Exchange tokens
    CEX_TOKEN = "CEX_Token"
    
    # Privacy
    PRIVACY = "Privacy"
    
    # Real World Assets
    RWA = "RWA"
    RWA_COMMODITY = "RWA_Commodity"
    RWA_SECURITY = "RWA_Security"
    
    # Other
    OTHER = "Other"
    
    @classmethod
    def get_parent_sector(cls, sector: 'TokenSector') -> Optional['TokenSector']:
        """Get parent sector for sub-sectors."""
        parent_map = {
            cls.L1_EVM: cls.L1,
            cls.L1_NON_EVM: cls.L1,
            cls.L2_OPTIMISTIC: cls.L2,
            cls.L2_ZK: cls.L2,
            cls.DEFI_LENDING: cls.DEFI,
            cls.DEFI_DEX: cls.DEFI,
            cls.DEFI_DERIVATIVES: cls.DEFI,
            cls.DEFI_YIELD: cls.DEFI,
            cls.DEFI_STABLECOIN: cls.DEFI,
            cls.MEME_DOG: cls.MEME,
            cls.MEME_CAT: cls.MEME,
            cls.AI_COMPUTE: cls.AI,
            cls.AI_DATA: cls.AI,
            cls.GAMING_INFRA: cls.GAMING,
            cls.METAVERSE: cls.GAMING,
            cls.ORACLE: cls.INFRA,
            cls.STORAGE: cls.INFRA,
            cls.INDEXING: cls.INFRA,
            cls.INTEROP: cls.INFRA,
            cls.RWA_COMMODITY: cls.RWA,
            cls.RWA_SECURITY: cls.RWA,
        }
        return parent_map.get(sector)
    
    @classmethod
    def are_related(cls, sector_a: 'TokenSector', sector_b: 'TokenSector') -> bool:
        """Check if two sectors are related (same or parent-child)."""
        if sector_a == sector_b:
            return True
        
        parent_a = cls.get_parent_sector(sector_a)
        parent_b = cls.get_parent_sector(sector_b)
        
        # Same parent
        if parent_a and parent_b and parent_a == parent_b:
            return True
        
        # One is parent of other
        if parent_a == sector_b or parent_b == sector_a:
            return True
        
        return False
    
    @property
    def description(self) -> str:
        """Sector description."""
        descriptions = {
            self.L1: "Layer 1 blockchain protocols",
            self.L2: "Layer 2 scaling solutions",
            self.DEFI: "Decentralized finance protocols",
            self.DEFI_LENDING: "DeFi lending and borrowing",
            self.DEFI_DEX: "Decentralized exchanges",
            self.DEFI_DERIVATIVES: "DeFi derivatives protocols",
            self.LST: "Liquid staking tokens",
            self.MEME: "Meme tokens",
            self.AI: "AI and machine learning tokens",
            self.GAMING: "Gaming and metaverse tokens",
            self.INFRA: "Infrastructure tokens",
            self.ORACLE: "Oracle networks",
            self.STORAGE: "Decentralized storage",
            self.CEX_TOKEN: "Centralized exchange tokens",
            self.PRIVACY: "Privacy-focused tokens",
            self.RWA: "Real world asset tokens",
            self.OTHER: "Other/uncategorized",
        }
        return descriptions.get(self, "Unknown sector")
    
    @property
    def correlation_expectation(self) -> str:
        """Expected correlation within sector."""
        high_corr = [self.L1, self.L2, self.DEFI_DEX, self.MEME, self.AI, self.GAMING]
        medium_corr = [self.DEFI, self.DEFI_LENDING, self.INFRA, self.ORACLE]
        
        if self in high_corr:
            return "high"
        elif self in medium_corr:
            return "medium"
        return "low"
    
    @property
    def volatility_profile(self) -> str:
        """Expected volatility profile."""
        very_high = [self.MEME, self.MEME_DOG, self.MEME_CAT, self.AI]
        high = [self.GAMING, self.L2, self.DEFI_DERIVATIVES]
        moderate = [self.L1, self.DEFI, self.DEFI_DEX]
        
        if self in very_high:
            return "very_high"
        elif self in high:
            return "high"
        elif self in moderate:
            return "moderate"
        return "low"
    
    @property
    def recommended_half_life_range(self) -> Tuple[float, float]:
        """Recommended half-life range in days."""
        ranges = {
            self.L1: (3.0, 21.0),
            self.L2: (2.0, 14.0),
            self.DEFI: (2.0, 14.0),
            self.MEME: (0.5, 7.0),
            self.AI: (1.0, 10.0),
            self.GAMING: (1.0, 10.0),
            self.LST: (3.0, 21.0),
            self.ORACLE: (2.0, 14.0),
        }
        return ranges.get(self, (1.0, 14.0))
    
    @property
    def pair_quality_bonus(self) -> float:
        """Bonus multiplier for same-sector pairs."""
        bonuses = {
            self.L1: 1.2,
            self.L2: 1.15,
            self.DEFI_DEX: 1.25,
            self.DEFI_LENDING: 1.2,
            self.MEME: 1.1,
            self.AI: 1.15,
            self.GAMING: 1.1,
        }
        return bonuses.get(self, 1.0)


class Chain(Enum):
    """
    Supported blockchain networks for DEX trading.
    
    Each chain has specific gas costs and execution characteristics.
    """
    ETHEREUM = "ethereum"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    POLYGON = "polygon"
    BASE = "base"
    AVALANCHE = "avalanche"
    BSC = "bsc"
    SOLANA = "solana"
    FANTOM = "fantom"
    GNOSIS = "gnosis"
    BLAST = "blast"
    SCROLL = "scroll"
    ZKSYNC = "zksync"
    LINEA = "linea"
    MANTLE = "mantle"
    
    @property
    def is_evm(self) -> bool:
        """True if chain is EVM-compatible."""
        return self != self.SOLANA
    
    @property
    def is_l2(self) -> bool:
        """True if chain is an L2."""
        l2_chains = [
            self.ARBITRUM, self.OPTIMISM, self.BASE, self.BLAST,
            self.SCROLL, self.ZKSYNC, self.LINEA, self.MANTLE
        ]
        return self in l2_chains
    
    @property
    def typical_gas_usd(self) -> float:
        """Typical gas cost in USD for a swap."""
        gas_costs = {
            self.ETHEREUM: 25.0,
            self.ARBITRUM: 0.50,
            self.OPTIMISM: 0.30,
            self.POLYGON: 0.05,
            self.BASE: 0.10,
            self.AVALANCHE: 0.50,
            self.BSC: 0.20,
            self.SOLANA: 0.01,
            self.FANTOM: 0.05,
            self.GNOSIS: 0.01,
            self.BLAST: 0.20,
            self.SCROLL: 0.30,
            self.ZKSYNC: 0.25,
            self.LINEA: 0.20,
            self.MANTLE: 0.10,
        }
        return gas_costs.get(self, 1.0)
    
    @property
    def block_time_seconds(self) -> float:
        """Typical block time in seconds."""
        block_times = {
            self.ETHEREUM: 12.0,
            self.ARBITRUM: 0.25,
            self.OPTIMISM: 2.0,
            self.POLYGON: 2.0,
            self.BASE: 2.0,
            self.AVALANCHE: 2.0,
            self.BSC: 3.0,
            self.SOLANA: 0.4,
            self.FANTOM: 1.0,
            self.BLAST: 2.0,
        }
        return block_times.get(self, 2.0)
    
    @property
    def finality_seconds(self) -> float:
        """Time to finality in seconds."""
        finality = {
            self.ETHEREUM: 780.0,  # 13 minutes
            self.ARBITRUM: 900.0,  # 15 minutes (L1 finality)
            self.OPTIMISM: 900.0,
            self.POLYGON: 180.0,   # 3 minutes
            self.BASE: 900.0,
            self.AVALANCHE: 2.0,   # Near instant
            self.BSC: 45.0,
            self.SOLANA: 0.4,      # Near instant
        }
        return finality.get(self, 60.0)
    
    @property
    def reliability_score(self) -> float:
        """Network reliability score (0-1)."""
        scores = {
            self.ETHEREUM: 0.99,
            self.ARBITRUM: 0.95,
            self.OPTIMISM: 0.95,
            self.POLYGON: 0.90,
            self.BASE: 0.95,
            self.AVALANCHE: 0.92,
            self.BSC: 0.88,
            self.SOLANA: 0.85,
        }
        return scores.get(self, 0.80)
    
    @property
    def position_multiplier(self) -> float:
        """Position size multiplier based on chain risk."""
        multipliers = {
            self.ETHEREUM: 1.0,
            self.ARBITRUM: 0.9,
            self.OPTIMISM: 0.9,
            self.BASE: 0.85,
            self.POLYGON: 0.8,
            self.AVALANCHE: 0.8,
            self.BSC: 0.7,
            self.SOLANA: 0.75,
        }
        return multipliers.get(self, 0.6)


class FilterReason(Enum):
    """
    Reasons for filtering out tokens from universe.
    
    Used for tracking and debugging universe construction.
    """
    PASSED = "passed"
    STABLECOIN = "stablecoin"
    WRAPPED = "wrapped"
    LEVERAGED = "leveraged"
    LOW_VOLUME = "low_volume"
    LOW_MCAP = "low_mcap"
    LOW_TVL = "low_tvl"
    LOW_TRADES = "low_trades"
    LOW_PRICE = "low_price"
    HIGH_SPREAD = "high_spread"
    NO_VENUE = "no_venue"
    WASH_TRADING = "wash_trading"
    INSUFFICIENT_HISTORY = "insufficient_history"
    HIGH_ZERO_VOLUME = "high_zero_volume"
    NO_FUNDING_RATE = "no_funding_rate"
    TEST_TOKEN = "test_token"
    DELISTED = "delisted"
    BLACKLISTED = "blacklisted"
    
    @property
    def is_permanent(self) -> bool:
        """True if filter is permanent."""
        permanent = [
            self.STABLECOIN, self.WRAPPED, self.LEVERAGED,
            self.TEST_TOKEN, self.BLACKLISTED
        ]
        return self in permanent
    
    @property
    def retry_days(self) -> Optional[int]:
        """Days to wait before retrying if not permanent."""
        if self.is_permanent:
            return None
        
        retry_map = {
            self.LOW_VOLUME: 7,
            self.LOW_MCAP: 14,
            self.LOW_TVL: 7,
            self.LOW_TRADES: 3,
            self.HIGH_SPREAD: 3,
            self.NO_VENUE: 30,
            self.WASH_TRADING: 14,
            self.INSUFFICIENT_HISTORY: 30,
            self.HIGH_ZERO_VOLUME: 7,
            self.NO_FUNDING_RATE: 30,
            self.DELISTED: None,
        }
        return retry_map.get(self, 7)
    
    @property
    def description(self) -> str:
        """Filter reason description."""
        descriptions = {
            self.PASSED: "Passed all filters",
            self.STABLECOIN: "Token is a stablecoin",
            self.WRAPPED: "Token is a wrapped version",
            self.LEVERAGED: "Token is a leveraged product",
            self.LOW_VOLUME: "Volume below threshold",
            self.LOW_MCAP: "Market cap below threshold",
            self.LOW_TVL: "TVL below threshold",
            self.LOW_TRADES: "Trade count below threshold",
            self.LOW_PRICE: "Price below threshold",
            self.HIGH_SPREAD: "Spread above threshold",
            self.NO_VENUE: "Not available on configured venues",
            self.WASH_TRADING: "High wash trading probability",
            self.INSUFFICIENT_HISTORY: "Not enough trading history",
            self.HIGH_ZERO_VOLUME: "Too many zero-volume days",
            self.NO_FUNDING_RATE: "Missing funding rate data",
            self.TEST_TOKEN: "Test or demo token",
            self.DELISTED: "Token has been delisted",
            self.BLACKLISTED: "Token is blacklisted",
        }
        return descriptions.get(self, "Unknown reason")


class PairType(Enum):
    """
    Classification of pair types for trading.
    
    Different pair types have different risk and execution characteristics.
    """
    CEX_CEX = "CEX_CEX"
    CEX_HYBRID = "CEX_HYBRID"
    HYBRID_HYBRID = "HYBRID_HYBRID"
    CEX_DEX = "CEX_DEX"
    HYBRID_DEX = "HYBRID_DEX"
    DEX_DEX = "DEX_DEX"
    CROSS_CHAIN = "CROSS_CHAIN"
    
    @classmethod
    def from_venues(
        cls,
        venue_a: VenueType,
        venue_b: VenueType,
        chain_a: Optional[Chain] = None,
        chain_b: Optional[Chain] = None
    ) -> 'PairType':
        """Classify pair type from venue types."""
        # Check cross-chain
        if chain_a and chain_b and chain_a != chain_b:
            return cls.CROSS_CHAIN
        
        venues = sorted([venue_a.value, venue_b.value])
        
        if venues == ['CEX', 'CEX']:
            return cls.CEX_CEX
        elif venues == ['CEX', 'HYBRID']:
            return cls.CEX_HYBRID
        elif venues == ['HYBRID', 'HYBRID']:
            return cls.HYBRID_HYBRID
        elif venues == ['CEX', 'DEX']:
            return cls.CEX_DEX
        elif venues == ['DEX', 'HYBRID']:
            return cls.HYBRID_DEX
        else:
            return cls.DEX_DEX
    
    @property
    def description(self) -> str:
        """Pair type description."""
        descriptions = {
            self.CEX_CEX: "Both tokens on CEX - highest liquidity",
            self.CEX_HYBRID: "CEX + Hybrid venue pair",
            self.HYBRID_HYBRID: "Both on hybrid venues",
            self.CEX_DEX: "CEX + DEX cross-venue pair",
            self.HYBRID_DEX: "Hybrid + DEX pair",
            self.DEX_DEX: "Both DEX-only - highest risk",
            self.CROSS_CHAIN: "Cross-chain pair - complex execution",
        }
        return descriptions.get(self, "Unknown pair type")
    
    @property
    def recommended_tier(self) -> TokenTier:
        """Recommended tier for this pair type."""
        tier_map = {
            self.CEX_CEX: TokenTier.TIER_1,
            self.CEX_HYBRID: TokenTier.TIER_1,
            self.HYBRID_HYBRID: TokenTier.TIER_2,
            self.CEX_DEX: TokenTier.TIER_2,
            self.HYBRID_DEX: TokenTier.TIER_2,
            self.DEX_DEX: TokenTier.TIER_3,
            self.CROSS_CHAIN: TokenTier.TIER_3,
        }
        return tier_map.get(self, TokenTier.TIER_3)
    
    @property
    def execution_complexity(self) -> str:
        """Execution complexity level."""
        complexity = {
            self.CEX_CEX: "simple",
            self.CEX_HYBRID: "moderate",
            self.HYBRID_HYBRID: "moderate",
            self.CEX_DEX: "complex",
            self.HYBRID_DEX: "complex",
            self.DEX_DEX: "complex",
            self.CROSS_CHAIN: "very_complex",
        }
        return complexity.get(self, "complex")
    
    @property
    def position_multiplier(self) -> float:
        """Position size multiplier."""
        multipliers = {
            self.CEX_CEX: 1.0,
            self.CEX_HYBRID: 0.85,
            self.HYBRID_HYBRID: 0.7,
            self.CEX_DEX: 0.6,
            self.HYBRID_DEX: 0.5,
            self.DEX_DEX: 0.3,
            self.CROSS_CHAIN: 0.25,
        }
        return multipliers.get(self, 0.3)
    
    @property
    def estimated_round_trip_cost_bps(self) -> float:
        """Estimated round trip cost in basis points."""
        costs = {
            self.CEX_CEX: 20.0,
            self.CEX_HYBRID: 16.0,
            self.HYBRID_HYBRID: 12.0,
            self.CEX_DEX: 85.0,
            self.HYBRID_DEX: 80.0,
            self.DEX_DEX: 150.0,
            self.CROSS_CHAIN: 200.0,
        }
        return costs.get(self, 100.0)
    
    @property
    def min_half_life_hours(self) -> float:
        """Minimum half-life to be profitable."""
        # Based on breakeven analysis
        half_lives = {
            self.CEX_CEX: 12.0,
            self.CEX_HYBRID: 18.0,
            self.HYBRID_HYBRID: 24.0,
            self.CEX_DEX: 48.0,
            self.HYBRID_DEX: 48.0,
            self.DEX_DEX: 72.0,
            self.CROSS_CHAIN: 96.0,
        }
        return half_lives.get(self, 48.0)


# =============================================================================
# CONSTANTS AND MAPPINGS
# =============================================================================

# Token to sector mappings (comprehensive curated list)
TOKEN_SECTOR_MAP: Dict[str, TokenSector] = {
    # Layer 1 - EVM Compatible
    'ETH': TokenSector.L1_EVM, 'BNB': TokenSector.L1_EVM,
    'AVAX': TokenSector.L1_EVM, 'FTM': TokenSector.L1_EVM,
    'CELO': TokenSector.L1_EVM, 'KAVA': TokenSector.L1_EVM,
    'METIS': TokenSector.L1_EVM, 'CANTO': TokenSector.L1_EVM,
    
    # Layer 1 - Non-EVM
    'BTC': TokenSector.L1_NON_EVM, 'SOL': TokenSector.L1_NON_EVM,
    'ADA': TokenSector.L1_NON_EVM, 'DOT': TokenSector.L1_NON_EVM,
    'ATOM': TokenSector.L1_NON_EVM, 'NEAR': TokenSector.L1_NON_EVM,
    'APT': TokenSector.L1_NON_EVM, 'SUI': TokenSector.L1_NON_EVM,
    'SEI': TokenSector.L1_NON_EVM, 'INJ': TokenSector.L1_NON_EVM,
    'TIA': TokenSector.L1_NON_EVM, 'ALGO': TokenSector.L1_NON_EVM,
    'XLM': TokenSector.L1_NON_EVM, 'XRP': TokenSector.L1_NON_EVM,
    'TRX': TokenSector.L1_NON_EVM, 'TON': TokenSector.L1_NON_EVM,
    'HBAR': TokenSector.L1_NON_EVM, 'ICP': TokenSector.L1_NON_EVM,
    'EGLD': TokenSector.L1_NON_EVM, 'XTZ': TokenSector.L1_NON_EVM,
    'FLOW': TokenSector.L1_NON_EVM, 'MINA': TokenSector.L1_NON_EVM,
    'KAS': TokenSector.L1_NON_EVM, 'STX': TokenSector.L1_NON_EVM,
    
    # Layer 2 - Optimistic
    'ARB': TokenSector.L2_OPTIMISTIC, 'OP': TokenSector.L2_OPTIMISTIC,
    'METIS': TokenSector.L2_OPTIMISTIC, 'BOBA': TokenSector.L2_OPTIMISTIC,
    'BLAST': TokenSector.L2_OPTIMISTIC, 'MODE': TokenSector.L2_OPTIMISTIC,
    
    # Layer 2 - ZK
    'MATIC': TokenSector.L2_ZK, 'IMX': TokenSector.L2_ZK,
    'STRK': TokenSector.L2_ZK, 'MANTA': TokenSector.L2_ZK,
    'ZK': TokenSector.L2_ZK, 'SCROLL': TokenSector.L2_ZK,
    'TAIKO': TokenSector.L2_ZK, 'LINEA': TokenSector.L2_ZK,
    
    # DeFi - Lending
    'AAVE': TokenSector.DEFI_LENDING, 'COMP': TokenSector.DEFI_LENDING,
    'MKR': TokenSector.DEFI_LENDING, 'MORPHO': TokenSector.DEFI_LENDING,
    'SPARK': TokenSector.DEFI_LENDING, 'VENUS': TokenSector.DEFI_LENDING,
    'RADIANT': TokenSector.DEFI_LENDING, 'BENQI': TokenSector.DEFI_LENDING,
    
    # DeFi - DEX
    'UNI': TokenSector.DEFI_DEX, 'SUSHI': TokenSector.DEFI_DEX,
    'CRV': TokenSector.DEFI_DEX, 'BAL': TokenSector.DEFI_DEX,
    'CAKE': TokenSector.DEFI_DEX, 'JOE': TokenSector.DEFI_DEX,
    'VELO': TokenSector.DEFI_DEX, 'AERO': TokenSector.DEFI_DEX,
    '1INCH': TokenSector.DEFI_DEX, 'RUNE': TokenSector.DEFI_DEX,
    'OSMO': TokenSector.DEFI_DEX, 'RAY': TokenSector.DEFI_DEX,
    
    # DeFi - Derivatives
    'DYDX': TokenSector.DEFI_DERIVATIVES, 'GMX': TokenSector.DEFI_DERIVATIVES,
    'GNS': TokenSector.DEFI_DERIVATIVES, 'PERP': TokenSector.DEFI_DERIVATIVES,
    'SNX': TokenSector.DEFI_DERIVATIVES, 'KWENTA': TokenSector.DEFI_DERIVATIVES,
    'LYRA': TokenSector.DEFI_DERIVATIVES, 'HEGIC': TokenSector.DEFI_DERIVATIVES,
    'AEVO': TokenSector.DEFI_DERIVATIVES, 'VERTEX': TokenSector.DEFI_DERIVATIVES,
    
    # DeFi - Yield
    'CVX': TokenSector.DEFI_YIELD, 'YFI': TokenSector.DEFI_YIELD,
    'PENDLE': TokenSector.DEFI_YIELD, 'BIFI': TokenSector.DEFI_YIELD,
    'SPELL': TokenSector.DEFI_YIELD, 'ALCX': TokenSector.DEFI_YIELD,
    'FXS': TokenSector.DEFI_YIELD, 'ENA': TokenSector.DEFI_YIELD,
    
    # Liquid Staking
    'LDO': TokenSector.LST, 'RPL': TokenSector.LST,
    'CBETH': TokenSector.LST, 'RETH': TokenSector.LST,
    'STETH': TokenSector.LST, 'SFRXETH': TokenSector.LST,
    'ANKR': TokenSector.LST, 'SD': TokenSector.LST,
    
    # Meme - Dog
    'DOGE': TokenSector.MEME_DOG, 'SHIB': TokenSector.MEME_DOG,
    'WIF': TokenSector.MEME_DOG, 'BONK': TokenSector.MEME_DOG,
    'FLOKI': TokenSector.MEME_DOG, 'ELON': TokenSector.MEME_DOG,
    'NEIRO': TokenSector.MEME_DOG, 'POPCAT': TokenSector.MEME_DOG,
    
    # Meme - Cat & Other
    'PEPE': TokenSector.MEME, 'TURBO': TokenSector.MEME,
    'MOG': TokenSector.MEME, 'BRETT': TokenSector.MEME,
    'MEME': TokenSector.MEME, 'WOJAK': TokenSector.MEME,
    'ANDY': TokenSector.MEME, 'TOSHI': TokenSector.MEME,
    
    # AI - Compute
    'RNDR': TokenSector.AI_COMPUTE, 'AKT': TokenSector.AI_COMPUTE,
    'TAO': TokenSector.AI_COMPUTE, 'THETA': TokenSector.AI_COMPUTE,
    'GLM': TokenSector.AI_COMPUTE, 'NOS': TokenSector.AI_COMPUTE,
    
    # AI - Data/Agents
    'FET': TokenSector.AI_DATA, 'AGIX': TokenSector.AI_DATA,
    'OCEAN': TokenSector.AI_DATA, 'WLD': TokenSector.AI_DATA,
    'ARKM': TokenSector.AI_DATA, 'PRIME': TokenSector.AI_DATA,
    'AI16Z': TokenSector.AI_DATA, 'VIRTUAL': TokenSector.AI_DATA,
    
    # Gaming - Infrastructure
    'IMX': TokenSector.GAMING_INFRA, 'RONIN': TokenSector.GAMING_INFRA,
    'BEAM': TokenSector.GAMING_INFRA, 'PRIME': TokenSector.GAMING_INFRA,
    'PORTAL': TokenSector.GAMING_INFRA, 'XAI': TokenSector.GAMING_INFRA,
    
    # Gaming/Metaverse
    'AXS': TokenSector.GAMING, 'GALA': TokenSector.GAMING,
    'ILV': TokenSector.GAMING, 'PIXEL': TokenSector.GAMING,
    'YGG': TokenSector.GAMING, 'MAGIC': TokenSector.GAMING,
    'SAND': TokenSector.METAVERSE, 'MANA': TokenSector.METAVERSE,
    'APE': TokenSector.METAVERSE, 'ENJ': TokenSector.METAVERSE,
    'SUPER': TokenSector.METAVERSE, 'HIGH': TokenSector.METAVERSE,
    
    # Oracle
    'LINK': TokenSector.ORACLE, 'BAND': TokenSector.ORACLE,
    'API3': TokenSector.ORACLE, 'PYTH': TokenSector.ORACLE,
    'DIA': TokenSector.ORACLE, 'UMA': TokenSector.ORACLE,
    'TRB': TokenSector.ORACLE,
    
    # Storage
    'FIL': TokenSector.STORAGE, 'AR': TokenSector.STORAGE,
    'STORJ': TokenSector.STORAGE, 'SC': TokenSector.STORAGE,
    'IOTX': TokenSector.STORAGE, 'BLZ': TokenSector.STORAGE,
    
    # Indexing/Data
    'GRT': TokenSector.INDEXING, 'POKT': TokenSector.INDEXING,
    'COVAL': TokenSector.INDEXING,
    
    # Interoperability
    'ATOM': TokenSector.INTEROP, 'DOT': TokenSector.INTEROP,
    'WORMHOLE': TokenSector.INTEROP, 'AXL': TokenSector.INTEROP,
    'LZ': TokenSector.INTEROP, 'ZRO': TokenSector.INTEROP,
    
    # CEX Tokens
    'BNB': TokenSector.CEX_TOKEN, 'OKB': TokenSector.CEX_TOKEN,
    'CRO': TokenSector.CEX_TOKEN, 'KCS': TokenSector.CEX_TOKEN,
    'LEO': TokenSector.CEX_TOKEN, 'GT': TokenSector.CEX_TOKEN,
    'MX': TokenSector.CEX_TOKEN, 'HT': TokenSector.CEX_TOKEN,
    
    # Privacy
    'XMR': TokenSector.PRIVACY, 'ZEC': TokenSector.PRIVACY,
    'DASH': TokenSector.PRIVACY, 'SCRT': TokenSector.PRIVACY,
    'ROSE': TokenSector.PRIVACY, 'ZEN': TokenSector.PRIVACY,
    
    # RWA - Commodities
    'PAXG': TokenSector.RWA_COMMODITY, 'XAUT': TokenSector.RWA_COMMODITY,
    
    # RWA - Securities
    'ONDO': TokenSector.RWA_SECURITY, 'MPL': TokenSector.RWA_SECURITY,
    'CFG': TokenSector.RWA_SECURITY, 'RIO': TokenSector.RWA_SECURITY,
}

# Stablecoin symbols to exclude
STABLECOINS: Set[str] = {
    'USDT', 'USDC', 'BUSD', 'DAI', 'FRAX', 'TUSD', 'USDP', 'GUSD',
    'LUSD', 'USDD', 'USDE', 'FDUSD', 'PYUSD', 'CUSD', 'SUSD',
    'MIM', 'HUSD', 'UST', 'USTC', 'OUSD', 'DOLA', 'CRVUSD', 'GHO',
    'EURC', 'EURS', 'EURT', 'AGEUR', 'ALUSD', 'MKUSD', 'HAY',
    'USDJ', 'RSV', 'PAX', 'ZUSD', 'MUSD', 'AAVE', 'USDN',
    'USDK', 'USDX', 'TRIBE', 'IRON', 'ESD', 'BAC', 'DSD',
    'FLEXUSD', 'XSGD', 'BIDR', 'IDRT', 'GYEN', 'BRLA',
}

# Wrapped token patterns
WRAPPED_PATTERNS: List[str] = [
    'WBTC', 'WETH', 'WAVAX', 'WBNB', 'WMATIC', 'WSOL', 'WFTM',
    'WONE', 'WCELO', 'WXDAI', 'WROSE', 'WKCS', 'WHBAR',
]

WRAPPED_PREFIXES: List[str] = ['W', 'WRAPPED', 'BRIDGED']

# Liquid staking derivatives patterns
LST_PATTERNS: List[str] = [
    'STETH', 'RETH', 'CBETH', 'SFRXETH', 'ANKRBNB', 'STKBNB',
    'STSTX', 'MSOL', 'BSOL', 'JITOSOL', 'STAKESOL',
]

# Leveraged token patterns
LEVERAGED_PATTERNS: List[str] = [
    'UP', 'DOWN', 'BULL', 'BEAR', '2X', '3X', '4X', '5X',
    '-L', '-S', 'LONG', 'SHORT', '2L', '3L', '2S', '3S',
    'BTCUP', 'BTCDOWN', 'ETHUP', 'ETHDOWN',
]

# Blacklisted tokens (known scams, hacks, defunct)
# Must match survivorship_tracker.py known delistings + additional scams
BLACKLISTED_TOKENS: Set[str] = {
    # Major failures tracked in survivorship_tracker.py
    'LUNA', 'UST', 'FTT', 'LUNC', 'USTC', 'CEL', 'SRM',
    # Additional defunct/scam tokens
    'VOYAGER', 'SQUID', 'IRON', 'TITAN', 'SAFEMOON', 'BUNNY',
    # Deprecated/problematic tokens
    'EOS', 'BITCONNECT', 'HEX',
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TokenInfo:
    """
    Comprehensive information about a single token with computed properties.
    """
    symbol: str
    name: str = ""
    sector: TokenSector = TokenSector.OTHER
    primary_venue: VenueType = VenueType.CEX
    available_venues: List[str] = field(default_factory=list)
    chains: List[Chain] = field(default_factory=list)
    avg_daily_volume_usd: float = 0.0
    market_cap_usd: float = 0.0
    tvl_usd: float = 0.0
    fdv_usd: float = 0.0
    circulating_supply: float = 0.0
    total_supply: float = 0.0
    is_stablecoin: bool = False
    is_wrapped: bool = False
    is_leveraged: bool = False
    listing_date: Optional[datetime] = None
    delisting_date: Optional[datetime] = None
    tier: TokenTier = TokenTier.TIER_1
    coingecko_id: Optional[str] = None
    contract_addresses: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Filter tracking
    filter_result: FilterReason = FilterReason.PASSED
    filter_timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Calculate derived fields."""
        # Auto-classify tier if not set
        if self.avg_daily_volume_usd > 0 or self.market_cap_usd > 0:
            self.tier = TokenTier.from_metrics(
                self.avg_daily_volume_usd,
                self.market_cap_usd,
                self.tvl_usd,
                self.primary_venue
            )
    
    # Status properties
    def is_active(self, as_of_date: Optional[datetime] = None) -> bool:
        """Check if token was active on a given date."""
        as_of_date = as_of_date or datetime.utcnow()
        if self.listing_date and as_of_date < self.listing_date:
            return False
        if self.delisting_date and as_of_date > self.delisting_date:
            return False
        return True
    
    @property
    def is_tradeable(self) -> bool:
        """Check if token is suitable for trading."""
        return (
            not self.is_stablecoin and
            not self.is_wrapped and
            not self.is_leveraged and
            self.delisting_date is None and
            self.filter_result == FilterReason.PASSED
        )
    
    @property
    def is_delisted(self) -> bool:
        """True if token has been delisted."""
        return self.delisting_date is not None
    
    # Liquidity properties
    @property
    def liquidity_score(self) -> float:
        """Liquidity score (0-1)."""
        volume_score = min(1.0, np.log1p(self.avg_daily_volume_usd) / np.log1p(100_000_000))
        mcap_score = min(1.0, np.log1p(self.market_cap_usd) / np.log1p(10_000_000_000))
        tvl_score = min(1.0, np.log1p(self.tvl_usd) / np.log1p(100_000_000)) if self.tvl_usd > 0 else 0.5
        
        if self.primary_venue == VenueType.CEX:
            return 0.5 * volume_score + 0.4 * mcap_score + 0.1 * tvl_score
        elif self.primary_venue == VenueType.DEX:
            return 0.3 * volume_score + 0.2 * mcap_score + 0.5 * tvl_score
        else:
            return 0.4 * volume_score + 0.3 * mcap_score + 0.3 * tvl_score
    
    @property
    def volume_mcap_ratio(self) -> float:
        """Volume to market cap ratio (turnover)."""
        if self.market_cap_usd <= 0:
            return 0.0
        return self.avg_daily_volume_usd / self.market_cap_usd
    
    @property
    def volume_tvl_ratio(self) -> float:
        """Volume to TVL ratio (wash trading indicator)."""
        if self.tvl_usd <= 0:
            return 0.0
        return self.avg_daily_volume_usd / self.tvl_usd
    
    @property
    def wash_trading_probability(self) -> float:
        """Estimated wash trading probability."""
        ratio = self.volume_tvl_ratio
        if ratio <= 2.0:
            return 0.1
        elif ratio <= 5.0:
            return 0.3
        elif ratio <= 10.0:
            return 0.6
        return 0.9
    
    # Trading properties
    @property
    def position_multiplier(self) -> float:
        """Combined position multiplier."""
        return (
            self.tier.position_multiplier *
            self.primary_venue.position_multiplier *
            (0.5 if self.wash_trading_probability > 0.5 else 1.0)
        )
    
    @property
    def max_position_usd(self) -> float:
        """Maximum position size."""
        return min(
            self.tier.max_position_usd,
            self.primary_venue.max_position_usd,
            self.avg_daily_volume_usd * 0.05  # Max 5% of daily volume
        )
    
    @property
    def recommended_entry_z(self) -> float:
        """Recommended z-score entry threshold."""
        base_z = self.primary_venue.recommended_entry_z
        # Adjust for tier
        tier_adjustment = (self.tier.value - 1) * 0.2
        return base_z + tier_adjustment
    
    @property
    def estimated_slippage_bps(self) -> float:
        """Estimated slippage in basis points."""
        base_slippage = self.primary_venue.typical_slippage_bps
        
        # Adjust for liquidity
        if self.liquidity_score < 0.3:
            return base_slippage * 2.0
        elif self.liquidity_score < 0.5:
            return base_slippage * 1.5
        return base_slippage
    
    @property
    def estimated_round_trip_cost_bps(self) -> float:
        """Estimated round trip cost for pairs trade."""
        fee_cost = self.primary_venue.round_trip_cost_bps
        slippage_cost = self.estimated_slippage_bps * 4  # 4 legs
        return fee_cost + slippage_cost
    
    # Venue properties
    @property
    def venue_count(self) -> int:
        """Number of available venues."""
        return len(self.available_venues)
    
    @property
    def chain_count(self) -> int:
        """Number of available chains."""
        return len(self.chains)
    
    @property
    def primary_chain(self) -> Optional[Chain]:
        """Primary chain (first in list)."""
        return self.chains[0] if self.chains else None
    
    @property
    def has_multiple_chains(self) -> bool:
        """True if available on multiple chains."""
        return len(self.chains) > 1
    
    # Quality properties
    @property
    def quality_score(self) -> float:
        """Overall quality score (0-1)."""
        liquidity = self.liquidity_score
        wash_penalty = 1.0 - self.wash_trading_probability * 0.5
        venue_bonus = min(1.0, self.venue_count / 5) * 0.2
        tier_score = (4 - self.tier.value) / 3
        
        return 0.4 * liquidity + 0.3 * wash_penalty + 0.1 * venue_bonus + 0.2 * tier_score
    
    @property
    def token_id(self) -> str:
        """Unique token identifier."""
        return hashlib.md5(
            f"{self.symbol}:{self.primary_venue.value}:{self.coingecko_id or ''}".encode()
        ).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'sector': self.sector.value,
            'primary_venue': self.primary_venue.value,
            'available_venues': self.available_venues,
            'chains': [c.value for c in self.chains],
            'avg_daily_volume_usd': self.avg_daily_volume_usd,
            'market_cap_usd': self.market_cap_usd,
            'tvl_usd': self.tvl_usd,
            'tier': self.tier.value,
            'is_tradeable': self.is_tradeable,
            'liquidity_score': round(self.liquidity_score, 3),
            'quality_score': round(self.quality_score, 3),
            'position_multiplier': round(self.position_multiplier, 2),
            'max_position_usd': round(self.max_position_usd, 0),
            'estimated_round_trip_cost_bps': round(self.estimated_round_trip_cost_bps, 1),
            'filter_result': self.filter_result.value,
        }
    
    def __repr__(self) -> str:
        return (
            f"TokenInfo({self.symbol}, tier={self.tier.value}, "
            f"venue={self.primary_venue.value}, liq={self.liquidity_score:.2f})"
        )


@dataclass
class UniverseConfig:
    """
    Configuration for universe construction with comprehensive thresholds.
    """
    # CEX filters
    cex_min_daily_volume_usd: float = 10_000_000
    cex_min_market_cap_usd: float = 100_000_000
    cex_min_price_usd: float = 0.0001
    cex_max_spread_pct: float = 0.5
    
    # DEX filters
    dex_min_tvl_usd: float = 500_000
    dex_min_daily_volume_usd: float = 50_000
    dex_min_daily_trades: int = 100
    dex_max_volume_tvl_ratio: float = 10.0
    dex_min_liquidity_depth: float = 10_000
    
    # Hybrid venue filters
    hybrid_min_open_interest_usd: float = 1_000_000
    hybrid_min_daily_volume_usd: float = 5_000_000
    
    # General filters
    min_trading_days: int = 30
    max_zero_volume_pct: float = 0.10
    require_funding_rate: bool = True
    
    # Tier thresholds
    tier1_min_volume: float = 50_000_000
    tier1_min_mcap: float = 500_000_000
    tier2_min_volume: float = 5_000_000
    tier2_min_mcap: float = 50_000_000
    
    # Chain configuration
    included_chains: List[Chain] = field(default_factory=lambda: [
        Chain.ETHEREUM, Chain.ARBITRUM, Chain.OPTIMISM,
        Chain.BASE, Chain.POLYGON, Chain.AVALANCHE
    ])
    
    # Venue configuration
    cex_venues: List[str] = field(default_factory=lambda: [
        'binance', 'bybit', 'okx', 'coinbase', 'kraken'
    ])
    
    hybrid_venues: List[str] = field(default_factory=lambda: [
        'hyperliquid', 'dydx_v4', 'vertex'
    ])
    
    dex_venues: List[str] = field(default_factory=lambda: [
        'uniswap_v3', 'sushiswap', 'curve', 'balancer'
    ])
    
    # Pair generation
    max_pairs_per_token: int = 10
    require_same_sector: bool = False
    require_common_venue: bool = True
    min_combined_volume: float = 20_000_000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cex_min_daily_volume_usd': self.cex_min_daily_volume_usd,
            'cex_min_market_cap_usd': self.cex_min_market_cap_usd,
            'dex_min_tvl_usd': self.dex_min_tvl_usd,
            'dex_min_daily_volume_usd': self.dex_min_daily_volume_usd,
            'min_trading_days': self.min_trading_days,
            'tier1_min_volume': self.tier1_min_volume,
            'tier1_min_mcap': self.tier1_min_mcap,
            'included_chains': [c.value for c in self.included_chains],
            'cex_venues': self.cex_venues,
            'hybrid_venues': self.hybrid_venues,
            'dex_venues': self.dex_venues,
        }


@dataclass
class PairCandidate:
    """
    A candidate trading pair with comprehensive scoring and metrics.
    """
    token_a: str
    token_b: str
    venue_type: VenueType
    pair_type: PairType = PairType.CEX_CEX
    available_venues: List[str] = field(default_factory=list)
    chains: List[Chain] = field(default_factory=list)
    
    # Metrics
    combined_volume: float = 0.0
    min_liquidity: float = 0.0
    sector_a: TokenSector = TokenSector.OTHER
    sector_b: TokenSector = TokenSector.OTHER
    tier: TokenTier = TokenTier.TIER_1
    
    # Scoring
    correlation: Optional[float] = None
    spread_volatility: Optional[float] = None
    estimated_half_life: Optional[float] = None
    
    # Costs
    estimated_round_trip_bps: float = 0.0
    estimated_gas_usd: float = 0.0
    
    # Ranking
    rank: int = 0
    score: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived fields."""
        self.sector_match = self.is_same_sector
        if self.estimated_round_trip_bps == 0.0:
            self.estimated_round_trip_bps = self.pair_type.estimated_round_trip_cost_bps
    
    @property
    def pair_id(self) -> str:
        """Unique pair identifier."""
        tokens = sorted([self.token_a, self.token_b])
        return f"{tokens[0]}_{tokens[1]}"
    
    @property
    def pair_name(self) -> str:
        """Human-readable pair name."""
        return f"{self.token_a}/{self.token_b}"
    
    @property
    def is_same_sector(self) -> bool:
        """True if tokens are in same sector."""
        return TokenSector.are_related(self.sector_a, self.sector_b)
    
    @property
    def is_cross_venue(self) -> bool:
        """True if pair spans different venue types."""
        return self.pair_type in [
            PairType.CEX_DEX, PairType.CEX_HYBRID,
            PairType.HYBRID_DEX, PairType.CROSS_CHAIN
        ]
    
    @property
    def is_cross_chain(self) -> bool:
        """True if pair spans different chains."""
        return self.pair_type == PairType.CROSS_CHAIN or len(set(self.chains)) > 1
    
    @property
    def execution_complexity(self) -> str:
        """Execution complexity level."""
        return self.pair_type.execution_complexity
    
    @property
    def position_multiplier(self) -> float:
        """Position size multiplier."""
        return (
            self.pair_type.position_multiplier *
            self.tier.position_multiplier
        )
    
    @property
    def max_position_usd(self) -> float:
        """Maximum position size."""
        return min(
            self.tier.max_position_usd,
            self.min_liquidity * 0.05,  # 5% of min liquidity
            self.combined_volume * 0.025  # 2.5% of combined volume
        )
    
    @property
    def sector_bonus(self) -> float:
        """Bonus for same-sector pairs."""
        if self.is_same_sector:
            return self.sector_a.pair_quality_bonus
        return 1.0
    
    @property
    def breakeven_half_life_hours(self) -> float:
        """Minimum half-life to be profitable."""
        return self.pair_type.min_half_life_hours
    
    @property
    def is_profitable_half_life(self) -> bool:
        """True if estimated half-life exceeds breakeven."""
        if self.estimated_half_life is None:
            return True  # Unknown, assume possible
        return self.estimated_half_life * 24 >= self.breakeven_half_life_hours
    
    @property
    def opportunity_score(self) -> float:
        """Overall opportunity score (0-1)."""
        # Volume component (30%)
        volume_score = min(1.0, np.log1p(self.combined_volume) / np.log1p(100_000_000))
        
        # Sector component (20%)
        sector_score = 1.0 if self.is_same_sector else 0.5
        
        # Tier component (20%)
        tier_score = (4 - self.tier.value) / 3
        
        # Cost component (15%)
        cost_score = 1.0 - min(1.0, self.estimated_round_trip_bps / 200)
        
        # Complexity component (15%)
        complexity_map = {'simple': 1.0, 'moderate': 0.7, 'complex': 0.4, 'very_complex': 0.2}
        complexity_score = complexity_map.get(self.execution_complexity, 0.5)
        
        return (
            0.30 * volume_score +
            0.20 * sector_score +
            0.20 * tier_score +
            0.15 * cost_score +
            0.15 * complexity_score
        )
    
    def calculate_score(
        self,
        volume_weight: float = 0.3,
        sector_weight: float = 0.2,
        tier_weight: float = 0.2,
        cost_weight: float = 0.15,
        complexity_weight: float = 0.15
    ) -> float:
        """Calculate custom-weighted score."""
        volume_score = min(1.0, np.log1p(self.combined_volume) / np.log1p(100_000_000))
        sector_score = 1.0 if self.is_same_sector else 0.5
        tier_score = (4 - self.tier.value) / 3
        cost_score = 1.0 - min(1.0, self.estimated_round_trip_bps / 200)
        complexity_map = {'simple': 1.0, 'moderate': 0.7, 'complex': 0.4, 'very_complex': 0.2}
        complexity_score = complexity_map.get(self.execution_complexity, 0.5)
        
        self.score = (
            volume_weight * volume_score +
            sector_weight * sector_score +
            tier_weight * tier_score +
            cost_weight * cost_score +
            complexity_weight * complexity_score
        )
        return self.score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pair_id': self.pair_id,
            'token_a': self.token_a,
            'token_b': self.token_b,
            'venue_type': self.venue_type.value,
            'pair_type': self.pair_type.value,
            'tier': self.tier.value,
            'combined_volume': self.combined_volume,
            'sector_a': self.sector_a.value,
            'sector_b': self.sector_b.value,
            'is_same_sector': self.is_same_sector,
            'estimated_round_trip_bps': round(self.estimated_round_trip_bps, 1),
            'position_multiplier': round(self.position_multiplier, 2),
            'max_position_usd': round(self.max_position_usd, 0),
            'opportunity_score': round(self.opportunity_score, 3),
            'execution_complexity': self.execution_complexity,
            'rank': self.rank,
        }
    
    def __repr__(self) -> str:
        return (
            f"PairCandidate({self.pair_name}, type={self.pair_type.value}, "
            f"tier={self.tier.value}, score={self.opportunity_score:.3f})"
        )


@dataclass
class UniverseSnapshot:
    """
    Snapshot of universe state for tracking and analysis.
    """
    timestamp: datetime
    total_tokens: int
    tradeable_tokens: int
    total_pairs: int
    
    # By venue
    cex_tokens: int = 0
    hybrid_tokens: int = 0
    dex_tokens: int = 0
    
    # By tier
    tier1_tokens: int = 0
    tier2_tokens: int = 0
    tier3_tokens: int = 0
    
    # Metrics
    total_volume_usd: float = 0.0
    total_mcap_usd: float = 0.0
    total_tvl_usd: float = 0.0
    
    # Filter stats
    filter_stats: Dict[str, int] = field(default_factory=dict)
    
    # Changes
    newly_listed: List[str] = field(default_factory=list)
    delisted: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_tokens': self.total_tokens,
            'tradeable_tokens': self.tradeable_tokens,
            'total_pairs': self.total_pairs,
            'by_venue': {
                'cex': self.cex_tokens,
                'hybrid': self.hybrid_tokens,
                'dex': self.dex_tokens,
            },
            'by_tier': {
                'tier1': self.tier1_tokens,
                'tier2': self.tier2_tokens,
                'tier3': self.tier3_tokens,
            },
            'total_volume_usd': self.total_volume_usd,
            'total_mcap_usd': self.total_mcap_usd,
            'total_tvl_usd': self.total_tvl_usd,
            'newly_listed': len(self.newly_listed),
            'delisted': len(self.delisted),
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_token_sector(symbol: str) -> TokenSector:
    """Get sector classification for a token."""
    symbol_upper = symbol.upper()
    
    if symbol_upper in TOKEN_SECTOR_MAP:
        return TOKEN_SECTOR_MAP[symbol_upper]
    
    # Pattern-based inference
    patterns = [
        (['SWAP', 'DEX', 'AMM'], TokenSector.DEFI_DEX),
        (['LEND', 'BORROW', 'CREDIT'], TokenSector.DEFI_LENDING),
        (['INU', 'DOGE', 'SHIB'], TokenSector.MEME_DOG),
        (['CAT', 'MEOW', 'KITTY'], TokenSector.MEME_CAT),
        (['PEPE', 'FROG', 'WOJAK'], TokenSector.MEME),
        (['AI', 'GPT', 'NEURAL', 'ML'], TokenSector.AI),
        (['GAME', 'PLAY', 'QUEST'], TokenSector.GAMING),
        (['CHAIN', 'NET', 'PROTOCOL'], TokenSector.INFRA),
    ]
    
    for keywords, sector in patterns:
        if any(kw in symbol_upper for kw in keywords):
            return sector
    
    return TokenSector.OTHER


def is_stablecoin(symbol: str, name: str = "") -> bool:
    """Check if token is a stablecoin."""
    symbol_upper = symbol.upper()
    name_upper = name.upper() if name else ""
    
    if symbol_upper in STABLECOINS:
        return True
    
    stable_keywords = ['USD', 'DOLLAR', 'STABLE', 'PEG', 'FIAT']
    if any(kw in name_upper for kw in stable_keywords):
        return True
    
    return False


def is_wrapped_token(symbol: str, name: str = "") -> bool:
    """Check if token is a wrapped version."""
    symbol_upper = symbol.upper()
    name_upper = name.upper() if name else ""
    
    if symbol_upper in WRAPPED_PATTERNS:
        return True
    
    if symbol_upper in LST_PATTERNS:
        return True
    
    if 'WRAPPED' in name_upper or 'BRIDGED' in name_upper:
        return True
    
    # Check prefix patterns
    for prefix in WRAPPED_PREFIXES:
        if symbol_upper.startswith(prefix) and len(symbol_upper) > len(prefix) + 2:
            base = symbol_upper[len(prefix):]
            if base in ['BTC', 'ETH', 'AVAX', 'BNB', 'MATIC', 'SOL', 'FTM']:
                return True
    
    return False


def is_leveraged_token(symbol: str) -> bool:
    """Check if token is a leveraged product."""
    symbol_upper = symbol.upper()
    
    for pattern in LEVERAGED_PATTERNS:
        if pattern in symbol_upper:
            return True
    
    return False


def is_blacklisted(symbol: str) -> bool:
    """Check if token is blacklisted."""
    return symbol.upper() in BLACKLISTED_TOKENS


def filter_token(
    symbol: str,
    name: str = "",
    volume: float = 0.0,
    mcap: float = 0.0,
    tvl: float = 0.0,
    config: Optional[UniverseConfig] = None,
    venue_type: VenueType = VenueType.CEX
) -> Tuple[bool, FilterReason]:
    """
    Apply all filters to a token.
    
    Returns:
        Tuple of (passed, reason)
    """
    config = config or UniverseConfig()
    
    # Blacklist check
    if is_blacklisted(symbol):
        return False, FilterReason.BLACKLISTED
    
    # Type checks
    if is_stablecoin(symbol, name):
        return False, FilterReason.STABLECOIN
    
    if is_wrapped_token(symbol, name):
        return False, FilterReason.WRAPPED
    
    if is_leveraged_token(symbol):
        return False, FilterReason.LEVERAGED
    
    # Test token check
    if any(kw in symbol.upper() for kw in ['TEST', 'DEMO', 'SAMPLE', 'FAKE']):
        return False, FilterReason.TEST_TOKEN
    
    # Venue-specific filters
    if venue_type == VenueType.CEX:
        if volume < config.cex_min_daily_volume_usd:
            return False, FilterReason.LOW_VOLUME
        if mcap < config.cex_min_market_cap_usd:
            return False, FilterReason.LOW_MCAP
    
    elif venue_type == VenueType.DEX:
        if tvl < config.dex_min_tvl_usd:
            return False, FilterReason.LOW_TVL
        if volume < config.dex_min_daily_volume_usd:
            return False, FilterReason.LOW_VOLUME
        
        # Wash trading check
        if tvl > 0 and volume / tvl > config.dex_max_volume_tvl_ratio:
            return False, FilterReason.WASH_TRADING
    
    elif venue_type == VenueType.HYBRID:
        if volume < config.hybrid_min_daily_volume_usd:
            return False, FilterReason.LOW_VOLUME
    
    return True, FilterReason.PASSED


# =============================================================================
# UNIVERSE BUILDER CLASS
# =============================================================================

class UniverseBuilder:
    """
    Comprehensive universe construction for pairs trading.
    
    Handles:
    - Multi-venue token universe construction (CEX, DEX, Hybrid)
    - Comprehensive filtering pipeline
    - Survivorship bias tracking
    - Pair candidate generation with scoring
    - Universe persistence and updates
    
    Example
    -------
    >>> builder = UniverseBuilder(config)
    >>> builder.build_cex_universe(cex_data)
    >>> builder.build_dex_universe(dex_data)
    >>> builder.combine_universes()
    >>> pairs = builder.generate_pair_candidates()
    >>> snapshot = builder.create_snapshot()
    """
    
    def __init__(self, config: Optional[UniverseConfig] = None):
        """Initialize universe builder."""
        self.config = config or UniverseConfig()
        
        # Universes
        self.cex_universe: Dict[str, TokenInfo] = {}
        self.dex_universe: Dict[str, TokenInfo] = {}
        self.hybrid_universe: Dict[str, TokenInfo] = {}
        self.combined_universe: Dict[str, TokenInfo] = {}
        
        # Tracking
        self.delisted_tokens: Dict[str, datetime] = {}
        self.newly_listed_tokens: Dict[str, datetime] = {}
        self.filter_stats: Dict[FilterReason, int] = defaultdict(int)
        
        # Pairs
        self.pair_candidates: List[PairCandidate] = []
        
        # History
        self.snapshots: List[UniverseSnapshot] = []
        
        # Build stats
        self.build_stats: Dict[str, Any] = {}
        self.last_build_time: Optional[datetime] = None
        
        logger.info("UniverseBuilder initialized")
    
    def _apply_filters(
        self,
        symbol: str,
        name: str,
        volume: float,
        mcap: float,
        tvl: float,
        venue_type: VenueType
    ) -> Tuple[bool, FilterReason]:
        """Apply all filters and track statistics."""
        passed, reason = filter_token(
            symbol, name, volume, mcap, tvl,
            self.config, venue_type
        )
        self.filter_stats[reason] += 1
        return passed, reason
    
    def build_cex_universe(
        self,
        market_data: pd.DataFrame,
        volume_history: Optional[pd.DataFrame] = None
    ) -> Dict[str, TokenInfo]:
        """
        Build CEX token universe from market data.
        
        Args:
            market_data: DataFrame with columns:
                - symbol, name, price, volume_24h, market_cap, exchanges
            volume_history: Optional historical volume data
        """
        logger.info("Building CEX universe...")
        
        universe = {}
        
        for _, row in market_data.iterrows():
            symbol = str(row.get('symbol', '')).upper()
            name = str(row.get('name', ''))
            
            if not symbol:
                continue
            
            volume = float(row.get('volume_24h', 0) or 0)
            mcap = float(row.get('market_cap', 0) or 0)
            price = float(row.get('price', 0) or 0)
            
            # Apply filters
            passed, reason = self._apply_filters(
                symbol, name, volume, mcap, 0.0, VenueType.CEX
            )
            
            # Get exchanges
            exchanges = row.get('exchanges', [])
            if isinstance(exchanges, str):
                exchanges = [e.strip() for e in exchanges.split(',')]
            
            available_venues = [
                e for e in exchanges
                if any(v.lower() in e.lower() for v in self.config.cex_venues)
            ]
            
            if passed and not available_venues:
                passed = False
                reason = FilterReason.NO_VENUE
                self.filter_stats[reason] += 1
            
            # Create token info
            token_info = TokenInfo(
                symbol=symbol,
                name=name,
                sector=get_token_sector(symbol),
                primary_venue=VenueType.CEX,
                available_venues=available_venues,
                avg_daily_volume_usd=volume,
                market_cap_usd=mcap,
                coingecko_id=row.get('coingecko_id'),
                filter_result=reason,
                filter_timestamp=datetime.utcnow(),
            )
            
            if passed:
                universe[symbol] = token_info
        
        logger.info(f"CEX universe built: {len(universe)} tokens")
        
        self.cex_universe = universe
        self.build_stats['cex'] = {
            'total': len(universe),
            'build_time': datetime.utcnow().isoformat()
        }
        
        return universe
    
    def build_dex_universe(
        self,
        pool_data: pd.DataFrame,
        chain_filter: Optional[Chain] = None,
        wash_detector: Optional[Any] = None,
        mev_analyzer: Optional[Any] = None
    ) -> Tuple[Dict[str, TokenInfo], Dict[str, Any]]:
        """
        Build DEX token universe from pool data with wash trading detection.

        Args:
            pool_data: DataFrame with columns:
                - symbol, name, tvl, volume_24h, tx_count, chain, dex
            chain_filter: Optional chain to filter by
            wash_detector: Optional WashTradingDetector instance
            mev_analyzer: Optional MEVAnalyzer instance

        Returns:
            Tuple of (universe_dict, detection_stats_dict)
        """
        logger.info("Building DEX universe with comprehensive filtering...")

        # Aggregate by token
        token_metrics = defaultdict(lambda: {
            'tvl': 0.0, 'volume': 0.0, 'tx_count': 0,
            'chains': set(), 'dexes': set(), 'name': ''
        })

        for _, row in pool_data.iterrows():
            symbol = str(row.get('symbol', '')).upper()
            if not symbol:
                continue

            pool_chain = row.get('chain', '')

            # Chain filter
            if chain_filter and pool_chain.lower() != chain_filter.value:
                continue

            if pool_chain.lower() not in [c.value for c in self.config.included_chains]:
                continue

            token_metrics[symbol]['tvl'] += float(row.get('tvl', 0) or 0)
            token_metrics[symbol]['volume'] += float(row.get('volume_24h', 0) or 0)
            token_metrics[symbol]['tx_count'] += int(row.get('tx_count', 0) or 0)
            token_metrics[symbol]['chains'].add(pool_chain)
            token_metrics[symbol]['dexes'].add(row.get('dex', 'unknown'))
            token_metrics[symbol]['name'] = row.get('name', '')

        universe = {}
        wash_trading_stats = {
            'total_analyzed': 0,
            'high_risk_count': 0,
            'medium_risk_count': 0,
            'low_risk_count': 0,
            'flagged_tokens': [],
            'avg_risk_score': 0.0
        }

        risk_scores = []

        for symbol, metrics in token_metrics.items():
            passed, reason = self._apply_filters(
                symbol, metrics['name'],
                metrics['volume'], 0.0, metrics['tvl'],
                VenueType.DEX
            )

            # Trade count filter
            if passed and metrics['tx_count'] < self.config.dex_min_daily_trades:
                passed = False
                reason = FilterReason.LOW_TRADES
                self.filter_stats[reason] += 1

            # Wash trading detection (if detector provided and symbol passed filters)
            wash_risk_level = 'UNKNOWN'
            wash_risk_score = 0.0
            if passed and wash_detector is not None:
                try:
                    # Get symbol data from pool_data for this specific symbol
                    symbol_data = pool_data[pool_data['symbol'].str.upper() == symbol].copy()

                    if not symbol_data.empty and len(symbol_data) >= 10:
                        # Analyze for wash trading
                        wash_result = wash_detector.analyze(
                            data=symbol_data,
                            venue='DEX_AGGREGATED',
                            price_col='close' if 'close' in symbol_data.columns else 'price',
                            volume_col='volume_24h',
                            timestamp_col='timestamp' if 'timestamp' in symbol_data.columns else symbol_data.index
                        )

                        wash_risk_level = wash_result.risk_level
                        wash_risk_score = wash_result.risk_score
                        risk_scores.append(wash_risk_score)
                        wash_trading_stats['total_analyzed'] += 1

                        # Categorize risk
                        if wash_risk_score >= 70:
                            wash_trading_stats['high_risk_count'] += 1
                            wash_trading_stats['flagged_tokens'].append({
                                'symbol': symbol,
                                'risk_score': wash_risk_score,
                                'indicators': list(wash_result.indicators.keys())
                            })
                        elif wash_risk_score >= 40:
                            wash_trading_stats['medium_risk_count'] += 1
                        else:
                            wash_trading_stats['low_risk_count'] += 1

                        # Filter out high-risk tokens
                        if wash_risk_score >= 80:  # Very high risk threshold
                            passed = False
                            reason = FilterReason.WASH_TRADING
                            self.filter_stats[FilterReason.WASH_TRADING] = self.filter_stats.get(FilterReason.WASH_TRADING, 0) + 1
                            logger.info(f"Filtered {symbol} for wash trading risk: {wash_risk_score:.1f}")
                except Exception as e:
                    logger.warning(f"Wash trading analysis failed for {symbol}: {e}")

            # Map chains
            chains = []
            for c in metrics['chains']:
                try:
                    chains.append(Chain(c.lower()))
                except ValueError:
                    pass

            token_info = TokenInfo(
                symbol=symbol,
                name=metrics['name'],
                sector=get_token_sector(symbol),
                primary_venue=VenueType.DEX,
                available_venues=list(metrics['dexes']),
                chains=chains,
                avg_daily_volume_usd=metrics['volume'],
                tvl_usd=metrics['tvl'],
                filter_result=reason,
                filter_timestamp=datetime.utcnow(),
            )

            if passed:
                universe[symbol] = token_info

        # Calculate average risk score
        if risk_scores:
            wash_trading_stats['avg_risk_score'] = sum(risk_scores) / len(risk_scores)
            wash_trading_stats['high_risk_pct'] = wash_trading_stats['high_risk_count'] / wash_trading_stats['total_analyzed'] if wash_trading_stats['total_analyzed'] > 0 else 0.0
            wash_trading_stats['medium_risk_pct'] = wash_trading_stats['medium_risk_count'] / wash_trading_stats['total_analyzed'] if wash_trading_stats['total_analyzed'] > 0 else 0.0
            wash_trading_stats['low_risk_pct'] = wash_trading_stats['low_risk_count'] / wash_trading_stats['total_analyzed'] if wash_trading_stats['total_analyzed'] > 0 else 0.0

        logger.info(f"DEX universe built: {len(universe)} tokens")
        if wash_detector:
            logger.info(f"Wash trading: {wash_trading_stats['high_risk_count']} high risk, {wash_trading_stats['medium_risk_count']} medium risk")

        self.dex_universe = universe
        self.build_stats['dex'] = {
            'total': len(universe),
            'build_time': datetime.utcnow().isoformat(),
            'wash_trading_stats': wash_trading_stats
        }

        return universe, wash_trading_stats
    
    def build_hybrid_universe(
        self,
        perp_data: pd.DataFrame
    ) -> Dict[str, TokenInfo]:
        """Build hybrid venue universe (Hyperliquid, dYdX, Vertex)."""
        logger.info("Building hybrid venue universe...")
        
        token_metrics = defaultdict(lambda: {
            'open_interest': 0.0, 'volume': 0.0,
            'venues': set(), 'has_funding': False
        })
        
        for _, row in perp_data.iterrows():
            symbol = str(row.get('symbol', '')).upper()
            if not symbol:
                continue
            
            venue = row.get('venue', '')
            if venue.lower() not in [v.lower() for v in self.config.hybrid_venues]:
                continue
            
            token_metrics[symbol]['open_interest'] += float(row.get('open_interest', 0) or 0)
            token_metrics[symbol]['volume'] += float(row.get('volume_24h', 0) or 0)
            token_metrics[symbol]['venues'].add(venue)
            
            if row.get('funding_rate') is not None:
                token_metrics[symbol]['has_funding'] = True
        
        universe = {}
        
        for symbol, metrics in token_metrics.items():
            passed, reason = self._apply_filters(
                symbol, '', metrics['volume'], 0.0, 0.0, VenueType.HYBRID
            )
            
            # OI filter
            if passed and metrics['open_interest'] < self.config.hybrid_min_open_interest_usd:
                passed = False
                reason = FilterReason.LOW_VOLUME
                self.filter_stats[reason] += 1
            
            # Funding filter
            if passed and self.config.require_funding_rate and not metrics['has_funding']:
                passed = False
                reason = FilterReason.NO_FUNDING_RATE
                self.filter_stats[reason] += 1
            
            token_info = TokenInfo(
                symbol=symbol,
                sector=get_token_sector(symbol),
                primary_venue=VenueType.HYBRID,
                available_venues=list(metrics['venues']),
                avg_daily_volume_usd=metrics['volume'],
                filter_result=reason,
                filter_timestamp=datetime.utcnow(),
                metadata={'open_interest': metrics['open_interest']},
            )
            
            if passed:
                universe[symbol] = token_info
        
        logger.info(f"Hybrid universe built: {len(universe)} tokens")
        
        self.hybrid_universe = universe
        self.build_stats['hybrid'] = {
            'total': len(universe),
            'build_time': datetime.utcnow().isoformat()
        }
        
        return universe
    
    def combine_universes(self) -> Dict[str, TokenInfo]:
        """
        Combine CEX, DEX, and hybrid universes.
        
        Priority: CEX > Hybrid > DEX
        """
        combined = {}
        overlap_count = 0
        
        # Add CEX tokens (highest priority)
        for symbol, info in self.cex_universe.items():
            combined[symbol] = TokenInfo(
                symbol=info.symbol,
                name=info.name,
                sector=info.sector,
                primary_venue=VenueType.CEX,
                available_venues=info.available_venues.copy(),
                chains=info.chains.copy(),
                avg_daily_volume_usd=info.avg_daily_volume_usd,
                market_cap_usd=info.market_cap_usd,
                tvl_usd=info.tvl_usd,
                coingecko_id=info.coingecko_id,
            )
        
        # Add hybrid venues
        for symbol, info in self.hybrid_universe.items():
            if symbol in combined:
                combined[symbol].available_venues.extend(info.available_venues)
                combined[symbol].available_venues = list(set(combined[symbol].available_venues))
                overlap_count += 1
            else:
                combined[symbol] = TokenInfo(
                    symbol=info.symbol,
                    name=info.name,
                    sector=info.sector,
                    primary_venue=VenueType.HYBRID,
                    available_venues=info.available_venues.copy(),
                    avg_daily_volume_usd=info.avg_daily_volume_usd,
                    metadata=info.metadata.copy(),
                )
        
        # Add DEX tokens
        for symbol, info in self.dex_universe.items():
            if symbol in combined:
                combined[symbol].available_venues.extend(info.available_venues)
                combined[symbol].available_venues = list(set(combined[symbol].available_venues))
                combined[symbol].chains.extend(info.chains)
                combined[symbol].chains = list(set(combined[symbol].chains))
                combined[symbol].tvl_usd = max(combined[symbol].tvl_usd, info.tvl_usd)
                overlap_count += 1
            else:
                combined[symbol] = TokenInfo(
                    symbol=info.symbol,
                    name=info.name,
                    sector=info.sector,
                    primary_venue=VenueType.DEX,
                    available_venues=info.available_venues.copy(),
                    chains=info.chains.copy(),
                    avg_daily_volume_usd=info.avg_daily_volume_usd,
                    tvl_usd=info.tvl_usd,
                )
        
        logger.info(f"Combined universe: {len(combined)} tokens, {overlap_count} overlaps merged")
        
        self.combined_universe = combined
        self.last_build_time = datetime.utcnow()
        
        return combined
    
    def track_delistings(
        self,
        previous_universe: Dict[str, TokenInfo]
    ) -> Tuple[Set[str], Set[str]]:
        """Track token listings and delistings for survivorship bias."""
        previous = set(previous_universe.keys())
        current = set(self.combined_universe.keys())
        
        delisted = previous - current
        newly_listed = current - previous
        
        now = datetime.utcnow()
        
        for symbol in delisted:
            self.delisted_tokens[symbol] = now
            if symbol in previous_universe:
                previous_universe[symbol].delisting_date = now
        
        for symbol in newly_listed:
            self.newly_listed_tokens[symbol] = now
            if symbol in self.combined_universe:
                self.combined_universe[symbol].listing_date = now
        
        logger.info(f"Tracked: {len(delisted)} delisted, {len(newly_listed)} newly listed")
        
        return delisted, newly_listed
    
    def generate_pair_candidates(
        self,
        same_sector_only: bool = False,
        min_tier: TokenTier = TokenTier.TIER_3,
        require_common_venue: bool = True,
        max_pairs: int = 1000
    ) -> List[PairCandidate]:
        """
        Generate candidate trading pairs from universe.
        
        Args:
            same_sector_only: Only generate pairs within same sector
            min_tier: Minimum tier for both tokens
            require_common_venue: Require shared venue
            max_pairs: Maximum pairs to return
        """
        universe = self.combined_universe
        
        # Filter eligible tokens
        eligible = {
            symbol: info
            for symbol, info in universe.items()
            if info.tier.value <= min_tier.value and info.is_tradeable
        }
        
        logger.info(f"Generating pairs from {len(eligible)} eligible tokens...")
        
        candidates = []
        symbols = list(eligible.keys())
        
        for i, symbol_a in enumerate(symbols):
            info_a = eligible[symbol_a]
            
            for symbol_b in symbols[i+1:]:
                info_b = eligible[symbol_b]
                
                # Sector check
                sector_match = TokenSector.are_related(info_a.sector, info_b.sector)
                if same_sector_only and not sector_match:
                    continue
                
                # Common venue check
                common_venues = set(info_a.available_venues) & set(info_b.available_venues)
                if require_common_venue and not common_venues:
                    continue
                
                # Common chains
                common_chains = list(set(info_a.chains) & set(info_b.chains))
                
                # Determine pair type
                pair_type = PairType.from_venues(
                    info_a.primary_venue,
                    info_b.primary_venue,
                    info_a.primary_chain,
                    info_b.primary_chain
                )
                
                # Primary venue type
                if info_a.primary_venue == info_b.primary_venue:
                    venue_type = info_a.primary_venue
                elif VenueType.CEX in [info_a.primary_venue, info_b.primary_venue]:
                    venue_type = VenueType.CEX
                else:
                    venue_type = VenueType.DEX
                
                # Metrics
                combined_volume = info_a.avg_daily_volume_usd + info_b.avg_daily_volume_usd
                min_liquidity = min(info_a.avg_daily_volume_usd, info_b.avg_daily_volume_usd)
                min_tier_val = max(info_a.tier.value, info_b.tier.value)
                
                # Round trip cost
                rt_cost = max(
                    info_a.estimated_round_trip_cost_bps,
                    info_b.estimated_round_trip_cost_bps
                )
                
                # Gas cost
                gas_cost = 0.0
                if common_chains:
                    gas_cost = min(c.typical_gas_usd for c in common_chains) * 4
                
                candidate = PairCandidate(
                    token_a=symbol_a,
                    token_b=symbol_b,
                    venue_type=venue_type,
                    pair_type=pair_type,
                    available_venues=list(common_venues),
                    chains=common_chains,
                    combined_volume=combined_volume,
                    min_liquidity=min_liquidity,
                    sector_a=info_a.sector,
                    sector_b=info_b.sector,
                    tier=TokenTier(min_tier_val),
                    estimated_round_trip_bps=rt_cost,
                    estimated_gas_usd=gas_cost,
                )
                
                # Calculate score
                candidate.calculate_score()
                
                candidates.append(candidate)
        
        # Sort by score and rank
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        for i, c in enumerate(candidates):
            c.rank = i + 1
        
        # Limit results
        candidates = candidates[:max_pairs]
        
        logger.info(f"Generated {len(candidates)} pair candidates")
        
        self.pair_candidates = candidates
        return candidates
    
    def create_snapshot(self) -> UniverseSnapshot:
        """Create snapshot of current universe state."""
        universe = self.combined_universe
        
        # Count by venue
        cex_count = sum(1 for t in universe.values() if t.primary_venue == VenueType.CEX)
        hybrid_count = sum(1 for t in universe.values() if t.primary_venue == VenueType.HYBRID)
        dex_count = sum(1 for t in universe.values() if t.primary_venue == VenueType.DEX)
        
        # Count by tier
        tier1_count = sum(1 for t in universe.values() if t.tier == TokenTier.TIER_1)
        tier2_count = sum(1 for t in universe.values() if t.tier == TokenTier.TIER_2)
        tier3_count = sum(1 for t in universe.values() if t.tier == TokenTier.TIER_3)
        
        # Totals
        total_volume = sum(t.avg_daily_volume_usd for t in universe.values())
        total_mcap = sum(t.market_cap_usd for t in universe.values())
        total_tvl = sum(t.tvl_usd for t in universe.values())
        
        snapshot = UniverseSnapshot(
            timestamp=datetime.utcnow(),
            total_tokens=len(universe),
            tradeable_tokens=sum(1 for t in universe.values() if t.is_tradeable),
            total_pairs=len(self.pair_candidates),
            cex_tokens=cex_count,
            hybrid_tokens=hybrid_count,
            dex_tokens=dex_count,
            tier1_tokens=tier1_count,
            tier2_tokens=tier2_count,
            tier3_tokens=tier3_count,
            total_volume_usd=total_volume,
            total_mcap_usd=total_mcap,
            total_tvl_usd=total_tvl,
            filter_stats={k.value: v for k, v in self.filter_stats.items()},
            newly_listed=list(self.newly_listed_tokens.keys()),
            delisted=list(self.delisted_tokens.keys()),
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def export_universe(
        self,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Export universe to DataFrame."""
        records = [info.to_dict() for info in self.combined_universe.values()]
        df = pd.DataFrame(records)
        df = df.sort_values('avg_daily_volume_usd', ascending=False).reset_index(drop=True)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Universe exported to {output_path}")
        
        return df
    
    def export_pairs(
        self,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Export pair candidates to DataFrame."""
        records = [pair.to_dict() for pair in self.pair_candidates]
        df = pd.DataFrame(records)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Pairs exported to {output_path}")
        
        return df
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive universe summary."""
        universe = self.combined_universe
        
        if not universe:
            return {'status': 'empty'}
        
        return {
            'total_tokens': len(universe),
            'tradeable_tokens': sum(1 for t in universe.values() if t.is_tradeable),
            'total_pairs': len(self.pair_candidates),
            'by_venue': {
                'cex': len(self.cex_universe),
                'hybrid': len(self.hybrid_universe),
                'dex': len(self.dex_universe),
            },
            'by_tier': {
                'tier1': sum(1 for t in universe.values() if t.tier == TokenTier.TIER_1),
                'tier2': sum(1 for t in universe.values() if t.tier == TokenTier.TIER_2),
                'tier3': sum(1 for t in universe.values() if t.tier == TokenTier.TIER_3),
            },
            'filter_stats': {k.value: v for k, v in self.filter_stats.items()},
            'last_build': self.last_build_time.isoformat() if self.last_build_time else None,
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'VenueType',
    'TokenTier',
    'TokenSector',
    'Chain',
    'FilterReason',
    'PairType',
    
    # Data classes
    'TokenInfo',
    'UniverseConfig',
    'PairCandidate',
    'UniverseSnapshot',
    
    # Main class
    'UniverseBuilder',
    
    # Functions
    'get_token_sector',
    'is_stablecoin',
    'is_wrapped_token',
    'is_leveraged_token',
    'is_blacklisted',
    'filter_token',
    
    # Constants
    'TOKEN_SECTOR_MAP',
    'STABLECOINS',
    'BLACKLISTED_TOKENS',
]