"""
CEX (Centralized Exchange) Collectors Package.

Provides data collectors for major centralized exchanges:
- Binance: World's largest crypto exchange
- Bybit: Major derivatives exchange
- OKX: Multi-asset exchange
- Coinbase: US-regulated exchange
- Kraken: European exchange
- CME: Chicago Mercantile Exchange (BTC/ETH futures)

All collectors inherit from BaseCollector and support:
- Funding rates (perpetual swaps)
- OHLCV data
- Open interest
"""

from .binance_collector import BinanceCollector
from .bybit_collector import BybitCollector
from .okx_collector import OKXCollector
from .coinbase_collector import CoinbaseCollector
from .kraken_collector import KrakenCollector
from .cme_collector import CMECollector

__all__ = [
    'BinanceCollector',
    'BybitCollector',
    'OKXCollector',
    'CoinbaseCollector',
    'KrakenCollector',
    'CMECollector',
]
