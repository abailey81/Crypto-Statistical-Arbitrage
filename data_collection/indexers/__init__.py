"""
Blockchain Indexers Collectors Package.

Provides data collectors for blockchain indexing services:
- The Graph: Decentralized indexing protocol

Used for querying DeFi subgraphs (Uniswap, Aave, Compound, etc.)
"""

from .thegraph_collector import TheGraphCollector

__all__ = [
    'TheGraphCollector',
]
