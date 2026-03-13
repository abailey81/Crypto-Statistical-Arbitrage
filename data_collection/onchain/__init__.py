"""
On-Chain Analytics Collectors Package.

Provides data collectors for blockchain analytics platforms:
- Glassnode: Bitcoin/Ethereum on-chain metrics
- Santiment: Social and on-chain analytics
- CryptoQuant: Exchange flows and miner data
- CoinMetrics: Network data and metrics
- Nansen: Wallet labeling and smart money tracking
- Arkham: Entity tracking and intelligence
- Flipside: On-chain analytics
- Covalent: Multi-chain API
- Bitquery: GraphQL blockchain data
- Whale Alert: Large transaction tracking

Data types supported:
- Exchange inflows/outflows
- Wallet analytics
- Network metrics (hash rate, active addresses)
- Large transaction alerts
- Smart money movements
"""

from .glassnode_collector import GlassnodeCollector
from .santiment_collector import SantimentCollector
from .cryptoquant_collector import CryptoQuantCollector
from .coinmetrics_collector import CoinMetricsCollector
from .nansen_collector import NansenCollector
from .arkham_collector import ArkhamCollector
from .flipside_collector import FlipsideCollector
from .covalent_collector import CovalentCollector
from .bitquery_collector import BitqueryCollector
from .whale_alert_collector import WhaleAlertCollector

__all__ = [
    'GlassnodeCollector',
    'SantimentCollector',
    'CryptoQuantCollector',
    'CoinMetricsCollector',
    'NansenCollector',
    'ArkhamCollector',
    'FlipsideCollector',
    'CovalentCollector',
    'BitqueryCollector',
    'WhaleAlertCollector',
]
