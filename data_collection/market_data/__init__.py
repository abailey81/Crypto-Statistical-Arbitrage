"""
Market Data Providers Collectors Package.

Provides data collectors for market data aggregators:
- CoinGecko: Free market data API
- CryptoCompare: Historical and real-time data
- Messari: Research-grade asset metrics
- Kaiko: Enterprise market data

Data types supported:
- OHLCV data
- Market capitalization
- Volume analysis
- Social metrics
- Asset fundamentals
"""

from .coingecko_collector import CoinGeckoCollector
from .cryptocompare_collector import CryptoCompareCollector
from .messari_collector import MessariCollector
from .kaiko_collector import KaikoCollector

__all__ = [
    'CoinGeckoCollector',
    'CryptoCompareCollector',
    'MessariCollector',
    'KaikoCollector',
]
