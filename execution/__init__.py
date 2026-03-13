"""
Execution Module
================

Multi-venue trade execution framework for crypto statistical arbitrage.

Provides venue-specific execution logic for CEX (Binance, Coinbase, OKX),
DEX (Uniswap V3, Curve, 1inch), and cross-chain bridge operations.

Modules
-------
cex_executor : Centralized exchange execution via REST/WebSocket APIs
dex_executor : On-chain DEX execution with MEV protection and gas optimization
bridge_manager : Cross-chain asset transfers for multi-venue strategies
multi_venue_router : Unified order routing across CEX, DEX, and hybrid venues

Author: Tamer Atesyakar
Version: 1.0.0
"""

from typing import Dict, Optional

__version__ = '1.0.0'
__all__ = [
    'CEXExecutor',
    'DEXExecutor',
    'BridgeManager',
    'MultiVenueRouter',
]


def __getattr__(name):
    """Lazy import for execution modules."""
    if name == 'CEXExecutor':
        from .cex_executor import CEXExecutor
        return CEXExecutor
    elif name == 'DEXExecutor':
        from .dex_executor import DEXExecutor
        return DEXExecutor
    elif name == 'BridgeManager':
        from .bridge_manager import BridgeManager
        return BridgeManager
    elif name == 'MultiVenueRouter':
        from .multi_venue_router import MultiVenueRouter
        return MultiVenueRouter
    raise AttributeError(f"module 'execution' has no attribute '{name}'")
