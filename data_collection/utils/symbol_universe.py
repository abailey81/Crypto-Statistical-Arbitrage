"""
Symbol Universe Configuration Loader
=====================================

Centralizes symbol configuration for all data collectors.
Loads symbols from config/symbols.yaml and provides helper methods.

Project Requirement: 20+ altcoins for Strategy 2
Implemented: 200+ altcoins (10x requirement)

Usage:
    from data_collection.utils.symbol_universe import SymbolUniverse

    universe = SymbolUniverse()

    # Get all OHLCV symbols (200+)
    symbols = universe.get_ohlcv_symbols()

    # Get funding rate symbols (30+)
    symbols = universe.get_funding_rate_symbols()

    # Get symbols by category
    defi_symbols = universe.get_category_symbols('defi_protocols')

    # Get symbols by tier
    tier1_symbols = universe.get_tier_symbols(1)
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
import yaml

logger = logging.getLogger(__name__)

class SymbolUniverse:
    """
    Centralized symbol configuration manager.

    Loads symbol definitions from config/symbols.yaml and provides
    methods for different strategies and data collection needs.
    """

    # Singleton pattern for configuration caching
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._load_config()

    def _load_config(self):
        """Load symbols configuration from YAML file."""
        # Find config file relative to project root
        config_paths = [
            Path(__file__).parent.parent.parent / 'config' / 'symbols.yaml',
            Path.cwd() / 'config' / 'symbols.yaml',
            Path(os.environ.get('PROJECT_ROOT', '.')) / 'config' / 'symbols.yaml',
        ]

        config_path = None
        for path in config_paths:
            if path.exists():
                config_path = path
                break

        if config_path is None:
            logger.warning("symbols.yaml not found, using default configuration")
            self._config = self._get_default_config()
            return

        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Loaded symbol universe from {config_path}")
        except Exception as e:
            logger.error(f"Error loading symbols.yaml: {e}, using defaults")
            self._config = self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Return minimal default configuration."""
        return {
            'core': {'symbols': ['BTC', 'ETH']},
            'major_altcoins': {
                'symbols': [
                    'BNB', 'XRP', 'DOGE', 'SOL', 'ADA', 'AVAX', 'DOT',
                    'LINK', 'ATOM', 'LTC', 'BCH', 'ETC', 'TRX', 'XLM',
                    'NEAR', 'UNI', 'AAVE', 'ARB', 'OP', 'INJ'
                ]
            },
            'l1_blockchains': {'symbols': []},
            'l2_solutions': {'symbols': []},
            'defi_protocols': {'symbols': []},
            'infrastructure': {'symbols': []},
            'ai_data': {'symbols': []},
            'gaming_metaverse': {'symbols': []},
            'real_world_assets': {'symbols': []},
            'memecoins': {'symbols': []},
            'emerging': {'symbols': []},
            'category_metadata': {},
        }

    # -------------------------------------------------------------------------
    # CATEGORY-BASED SYMBOL ACCESS
    # -------------------------------------------------------------------------

    def get_category_symbols(self, category: str) -> List[str]:
        """
        Get symbols for a specific category.

        Args:
            category: Category name (e.g., 'l1_blockchains', 'defi_protocols')

        Returns:
            List of symbol strings
        """
        if category not in self._config:
            logger.warning(f"Unknown category: {category}")
            return []

        cat_config = self._config[category]
        if isinstance(cat_config, dict) and 'symbols' in cat_config:
            return cat_config['symbols']
        return []

    def get_all_categories(self) -> List[str]:
        """Get list of all available categories."""
        excluded = {'metadata', 'category_metadata', 'delisted_tokens',
                   'exchange_coverage', 'funding_rate_symbols', 'ohlcv_symbols',
                   'futures_curve_symbols', 'options_symbols', 'quality_thresholds',
                   'cointegration_groups'}
        return [k for k in self._config.keys()
                if k not in excluded and isinstance(self._config.get(k), dict)
                and 'symbols' in self._config.get(k, {})]

    # -------------------------------------------------------------------------
    # STRATEGY-SPECIFIC SYMBOL LISTS
    # -------------------------------------------------------------------------

    def get_all_symbols(self, dedupe: bool = True) -> List[str]:
        """
        Get ALL symbols from all categories (200+).

        Args:
            dedupe: Remove duplicates (default True)

        Returns:
            List of all symbol strings
        """
        all_symbols = []
        for category in self.get_all_categories():
            all_symbols.extend(self.get_category_symbols(category))

        if dedupe:
            # Preserve order while deduping
            seen = set()
            result = []
            for sym in all_symbols:
                if sym not in seen:
                    seen.add(sym)
                    result.append(sym)
            return result

        return all_symbols

    def get_ohlcv_symbols(self) -> List[str]:
        """
        Get symbols for OHLCV collection (Strategy 2: Altcoin Stat Arb).

        Project Requirement: 20+ altcoins
        Returns: 200+ symbols (10x requirement)
        """
        return self.get_all_symbols(dedupe=True)

    def get_funding_rate_symbols(self) -> List[str]:
        """
        Get symbols for funding rate collection (Strategy 1).

        Returns priority symbols most likely to have perpetual futures.
        """
        # Use predefined priority list if available
        funding_config = self._config.get('funding_rate_symbols', {})
        priority = funding_config.get('priority_symbols', [])

        if priority:
            return priority

        # Fallback: core + major altcoins + top L1s
        symbols = []
        symbols.extend(self.get_category_symbols('core'))
        symbols.extend(self.get_category_symbols('major_altcoins'))
        symbols.extend(self.get_category_symbols('l1_blockchains')[:15])
        symbols.extend(self.get_category_symbols('defi_protocols')[:10])

        # Dedupe
        seen = set()
        return [s for s in symbols if not (s in seen or seen.add(s))]

    def get_futures_curve_symbols(self) -> List[str]:
        """
        Get symbols for futures curve trading (Strategy 3).

        Only BTC/ETH/SOL have quarterly futures on major venues.
        """
        config = self._config.get('futures_curve_symbols', {})
        return config.get('symbols', ['BTC', 'ETH', 'SOL'])

    def get_options_symbols(self) -> List[str]:
        """
        Get symbols for options trading (Strategy 4).

        Only BTC/ETH have liquid options on Deribit.
        """
        config = self._config.get('options_symbols', {})
        return config.get('symbols', ['BTC', 'ETH', 'SOL'])

    # -------------------------------------------------------------------------
    # TIER-BASED FILTERING
    # -------------------------------------------------------------------------

    def get_tier_symbols(self, tier: int) -> List[str]:
        """
        Get symbols by risk/liquidity tier.

        Args:
            tier: 1 (highest liquidity), 2 (medium), 3 (lower)

        Returns:
            List of symbols in that tier
        """
        metadata = self._config.get('category_metadata', {})
        tier_symbols = []

        for category in self.get_all_categories():
            cat_meta = metadata.get(category, {})
            if cat_meta.get('tier') == tier:
                tier_symbols.extend(self.get_category_symbols(category))

        # Dedupe
        seen = set()
        return [s for s in tier_symbols if not (s in seen or seen.add(s))]

    def get_high_liquidity_symbols(self, min_volume_usd: float = 50_000_000) -> List[str]:
        """
        Get symbols meeting minimum volume threshold.

        Args:
            min_volume_usd: Minimum 24h volume in USD

        Returns:
            List of high-liquidity symbols
        """
        metadata = self._config.get('category_metadata', {})
        high_liq = []

        for category in self.get_all_categories():
            cat_meta = metadata.get(category, {})
            cat_min_vol = cat_meta.get('min_volume', 0)
            if cat_min_vol >= min_volume_usd:
                high_liq.extend(self.get_category_symbols(category))

        # Dedupe
        seen = set()
        return [s for s in high_liq if not (s in seen or seen.add(s))]

    # -------------------------------------------------------------------------
    # SURVIVORSHIP BIAS HANDLING
    # -------------------------------------------------------------------------

    def get_delisted_tokens(self) -> List[str]:
        """Get list of known delisted/defunct tokens."""
        config = self._config.get('delisted_tokens', {})
        return config.get('tokens', [])

    def is_delisted(self, symbol: str) -> bool:
        """Check if a symbol is known to be delisted."""
        return symbol.upper() in [t.upper() for t in self.get_delisted_tokens()]

    def get_active_symbols(self) -> List[str]:
        """Get all symbols excluding known delisted ones."""
        delisted = set(t.upper() for t in self.get_delisted_tokens())
        return [s for s in self.get_all_symbols() if s.upper() not in delisted]

    # -------------------------------------------------------------------------
    # EXCHANGE-SPECIFIC HELPERS
    # -------------------------------------------------------------------------

    def format_for_binance(self, symbols: List[str], quote: str = 'USDT') -> List[str]:
        """Format symbols for Binance API (e.g., BTC -> BTCUSDT)."""
        return [f"{s.upper()}{quote}" for s in symbols]

    def format_for_bybit(self, symbols: List[str]) -> List[str]:
        """Format symbols for Bybit API (e.g., BTC -> BTCUSDT)."""
        return [f"{s.upper()}USDT" for s in symbols]

    def format_for_hyperliquid(self, symbols: List[str]) -> List[str]:
        """Format symbols for Hyperliquid (uses base symbol only)."""
        return [s.upper() for s in symbols]

    def format_for_dydx(self, symbols: List[str]) -> List[str]:
        """Format symbols for dYdX V4 (e.g., BTC -> BTC-USD)."""
        return [f"{s.upper()}-USD" for s in symbols]

    # -------------------------------------------------------------------------
    # STATISTICS AND METADATA
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict:
        """Get statistics about the symbol universe."""
        all_symbols = self.get_all_symbols()
        categories = self.get_all_categories()

        stats = {
            'total_unique_symbols': len(all_symbols),
            'total_categories': len(categories),
            'requirement': 20,
            'exceeded_by': f"{len(all_symbols) / 20:.1f}x",
            'categories': {},
        }

        for cat in categories:
            stats['categories'][cat] = len(self.get_category_symbols(cat))

        return stats

    def print_summary(self):
        """Print summary of symbol universe configuration."""
        stats = self.get_statistics()

        print("=" * 60)
        print("SYMBOL UNIVERSE SUMMARY")
        print("=" * 60)
        print(f"Total Unique Symbols: {stats['total_unique_symbols']}")
        print(f"Project Requirement: {stats['requirement']}")
        print(f"Exceeded By: {stats['exceeded_by']}")
        print("-" * 60)
        print("Symbols by Category:")
        for cat, count in stats['categories'].items():
            print(f" {cat}: {count}")
        print("=" * 60)

# Convenience function for quick access
def get_symbol_universe() -> SymbolUniverse:
    """Get the singleton SymbolUniverse instance."""
    return SymbolUniverse()

# Module-level exports for direct imports
def get_all_symbols() -> List[str]:
    """Quick access to all symbols."""
    return SymbolUniverse().get_all_symbols()

def get_ohlcv_symbols() -> List[str]:
    """Quick access to OHLCV symbols."""
    return SymbolUniverse().get_ohlcv_symbols()

def get_funding_symbols() -> List[str]:
    """Quick access to funding rate symbols."""
    return SymbolUniverse().get_funding_rate_symbols()

if __name__ == '__main__':
    # Test the module
    universe = SymbolUniverse()
    universe.print_summary()

    print("\nOHLCV Symbols (first 20):")
    print(universe.get_ohlcv_symbols()[:20])

    print("\nFunding Rate Symbols:")
    print(universe.get_funding_rate_symbols())

    print("\nFormatted for Binance (first 10):")
    print(universe.format_for_binance(universe.get_all_symbols()[:10]))
