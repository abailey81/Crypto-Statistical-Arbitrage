"""
Options Venue Collectors Package.

Provides data collectors for crypto options exchanges:
- Deribit: Leading crypto options exchange (BTC/ETH)
- AEVO: On-chain options exchange
- Lyra: Decentralized options (DEPRECATED)
- Dopex: Single-sided options vaults

Data types supported:
- Options chains (strikes, expiries)
- Implied volatility surfaces
- Greeks (delta, gamma, theta, vega)
- Open interest by strike
- Funding rates (for perpetuals)
"""

from .deribit_collector import DeribitCollector
from .aevo_collector import AevoCollector
from .lyra_collector import LyraCollector
from .dopex_collector import DopexCollector

__all__ = [
    'DeribitCollector',
    'AevoCollector',
    'LyraCollector',
    'DopexCollector',
]
