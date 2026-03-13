"""
Alternative Data Collectors Package.

Provides data collectors for alternative data sources:
- DefiLlama: DeFi TVL and yields
- Coinalyze: Aggregated funding rates and liquidations
- Coinalyze Enhanced: Extended Coinalyze features
- LunarCrush: Social intelligence
- Dune Analytics: Custom SQL queries

Data types supported:
- Total Value Locked (TVL)
- Aggregated funding rates
- Liquidation data
- Social sentiment
- Custom analytics
"""

from .defillama_collector import DefiLlamaCollector
from .coinalyze_collector import CoinalyzeCollector
from .coinalyze_enhanced_collector import CoinalyzeEnhancedCollector
from .lunarcrush_collector import LunarCrushCollector
from .dune_analytics_collector import DuneAnalyticsCollector

__all__ = [
    'DefiLlamaCollector',
    'CoinalyzeCollector',
    'CoinalyzeEnhancedCollector',
    'LunarCrushCollector',
    'DuneAnalyticsCollector',
]
