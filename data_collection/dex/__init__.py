"""
DEX (Decentralized Exchange) Collectors Package.

Provides data collectors for decentralized exchanges and aggregators:
- Uniswap V3: Leading Ethereum DEX
- SushiSwap: Multi-chain DEX
- Curve: Stablecoin-optimized DEX
- GeckoTerminal: Multi-chain DEX aggregator
- DEXScreener: Real-time DEX analytics
- GMX: Decentralized perpetuals (Arbitrum)
- Vertex: Hybrid DEX (DEPRECATED)
- Jupiter: Solana DEX aggregator
- CowSwap: MEV-protected DEX
- 1inch: Multi-chain aggregator
- 0x: DEX aggregation protocol

Note: GMX uses 1-HOUR funding intervals (same as Hyperliquid/dYdX)
"""

from .uniswap_collector import UniswapCollector
from .sushiswap_v2_collector import SushiSwapV2Collector
from .curve_collector import CurveCollector
from .geckoterminal_collector import GeckoTerminalCollector
from .dexscreener_collector import DexScreenerCollector
from .gmx_collector import GMXCollector
from .vertex_collector import VertexCollector
from .jupiter_collector import JupiterCollector
from .cowswap_collector import CowSwapCollector
from .oneinch_collector import OneInchCollector
from .zerox_collector import ZeroXCollector

__all__ = [
    'UniswapCollector',
    'SushiSwapV2Collector',
    'CurveCollector',
    'GeckoTerminalCollector',
    'DexScreenerCollector',
    'GMXCollector',
    'VertexCollector',
    'JupiterCollector',
    'CowSwapCollector',
    'OneInchCollector',
    'ZeroXCollector',
]
