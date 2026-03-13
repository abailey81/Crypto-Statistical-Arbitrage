"""
Hybrid Venue Collectors Package.

Provides data collectors for hybrid venues (on-chain settlement, off-chain matching):
- Hyperliquid: High-performance perpetuals with on-chain settlement
- dYdX V4: Cosmos-based decentralized perpetuals

CRITICAL: These venues use 1-HOUR funding intervals!
Must normalize to 8-hour for cross-venue comparison.

Conversion: rate_8h = rate_1h × 8
"""

from .hyperliquid_collector import HyperliquidCollector
from .dydx_collector import DYDXCollector

__all__ = [
    'HyperliquidCollector',
    'DYDXCollector',
]
