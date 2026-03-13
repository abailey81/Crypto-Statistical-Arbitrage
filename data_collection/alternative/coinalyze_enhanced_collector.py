"""
Enhanced Coinalyze Collector - Liquidation Heatmaps and detailed Derivatives Analytics.

validated collector for detailed liquidation analysis:
- Liquidation heatmaps (price levels with concentrated liquidations)
- Open interest by price level distribution
- Long/short ratios with sentiment analysis
- Aggregated funding rates with predictions
- Liquidation cascade prediction
- Stop-hunt zone identification
- Position concentration risk metrics

API: https://coinalyze.net/
Rate Limits: 40 calls/minute with API key

Version: 2.0.0
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..base_collector import BaseCollector
from ..utils.rate_limiter import get_shared_rate_limiter
from ..utils.retry_handler import RetryHandler

logger = logging.getLogger(__name__)

class LiquidationZoneType(Enum):
    """Classification of liquidation zones."""
    STOP_HUNT = 'stop_hunt' # Price level targeted by whales
    SUPPORT = 'support' # Heavy long liquidations below
    RESISTANCE = 'resistance' # Heavy short liquidations above
    NEUTRAL = 'neutral' # Balanced liquidations
    CASCADE_RISK = 'cascade_risk' # Risk of cascading liquidations

@dataclass
class LiquidationZone:
    """A price zone with concentrated liquidations."""
    price_level: float
    price_range_low: float
    price_range_high: float
    long_liquidations_usd: float
    short_liquidations_usd: float
    total_liquidations_usd: float
    concentration_pct: float
    zone_type: LiquidationZoneType
    risk_score: float # 0-100
    distance_from_current_pct: float

@dataclass
class RiskMetrics:
    """Risk metrics based on liquidation data."""
    symbol: str
    current_price: float
    total_liq_at_risk_usd: float
    long_liq_at_risk_usd: float
    short_liq_at_risk_usd: float
    nearest_liq_zone_distance_pct: float
    liquidation_imbalance: float # >0 = more longs at risk, <0 = more shorts
    cascade_risk_score: float # 0-100
    suggested_stop_distance_pct: float
    timestamp: datetime

class CoinalyzeEnhancedCollector(BaseCollector):
    """
    Enhanced Coinalyze collector with liquidation heatmaps.
    
    Features:
    - Liquidation heatmaps: price levels where liquidations concentrate
    - Open interest distribution by price
    - Long/short ratio analysis with sentiment
    - Multi-exchange aggregation
    - Predicted funding rates
    - Historical liquidation data
    - Stop-hunt zone identification
    - Cascade risk assessment
    - Position sizing recommendations
    
    Use Cases:
    - Identify stop-hunt zones to avoid
    - Find support/resistance from liquidation clusters
    - Assess cascade liquidation risk
    - Optimize stop-loss placement
    - Detect market manipulation patterns
    """
    
    VENUE = 'coinalyze'
    VENUE_TYPE = 'analytics'
    BASE_URL = 'https://api.coinalyze.net/v1'
    
    # Supported exchanges for liquidation data
    SUPPORTED_EXCHANGES = [
        'binance', 'okx', 'bybit', 'bitget', 'deribit',
        'bitmex', 'huobi', 'kraken', 'phemex', 'coinex'
    ]
    
    # Major symbols with good liquidation data
    MAJOR_SYMBOLS = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
        'AVAXUSDT', 'LINKUSDT', 'MATICUSDT', 'ARBUSDT', 'OPUSDT',
        'BNBUSDT', 'ADAUSDT', 'DOTUSDT', 'ATOMUSDT', 'LTCUSDT'
    ]
    
    # Heatmap intervals
    HEATMAP_INTERVALS = ['1h', '4h', '12h', '1d', '3d', '7d']
    
    # Risk thresholds
    CASCADE_RISK_THRESHOLD = 70
    STOP_HUNT_CONCENTRATION_THRESHOLD = 5.0 # % of total liquidations
    
    def __init__(self, config: Dict):
        """
        Initialize enhanced Coinalyze collector.
        
        Args:
            config: Configuration with:
                - coinalyze_api_key: API key
                - rate_limit: Requests per minute
        """
        super().__init__(config)
        self.api_key = config.get('coinalyze_api_key', config.get('api_key', ''))
        self.session = None
        
        # Rate limiting
        rate_limit = config.get('rate_limit', 15)
        self.rate_limiter = get_shared_rate_limiter('coinalyze_enhanced', rate=rate_limit, per=60.0, burst=config.get('burst', 10))
        
        # Retry handler
        self.retry_handler = RetryHandler(
            max_retries=3,
            base_delay=2.0,
            max_delay=30.0
        )
        
        # Collection stats
        self.collection_stats = {
            'records_collected': 0,
            'api_calls': 0,
            'errors': 0
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            headers = {}
            if self.api_key:
                headers['api_key'] = self.api_key
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self.session
    
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Make API request with error handling."""
        session = await self._get_session()
        acquire_result = await self.rate_limiter.acquire(timeout=120.0)
        if hasattr(acquire_result, 'acquired') and not acquire_result.acquired:
            logger.debug(f"Coinalyze rate limiter timeout for {endpoint}")
            return None
        self.collection_stats['api_calls'] += 1

        url = f"{self.BASE_URL}/{endpoint}"
        
        async def _request():
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 429:
                    logger.warning("Coinalyze rate limit hit")
                    raise aiohttp.ClientError("Rate limit")
                else:
                    text = await resp.text()
                    logger.error(f"Coinalyze error {resp.status}: {text[:200]}")
                    return None
        
        try:
            return await self.retry_handler.execute(_request)
        except Exception as e:
            logger.error(f"Coinalyze request error: {e}")
            self.collection_stats['errors'] += 1
            return None
    
    # =========================================================================
    # LIQUIDATION HEATMAP
    # =========================================================================
    
    async def fetch_liquidation_heatmap(
        self,
        symbol: str = 'BTCUSDT',
        interval: str = '4h',
        exchange: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch liquidation heatmap data.
        
        Shows price levels where liquidations are concentrated,
        useful for identifying support/resistance and stop-hunt levels.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Time interval for aggregation
            exchange: Specific exchange or None for aggregated
            
        Returns:
            DataFrame with liquidation levels
        """
        params = {
            'symbols': symbol,
            'interval': interval
        }
        if exchange:
            params['exchange'] = exchange
        
        data = await self._make_request('liquidation-heatmap', params)
        
        if not data or 'data' not in data:
            return pd.DataFrame()
        
        records = []
        for entry in data.get('data', []):
            long_liq = float(entry.get('longLiquidations', 0) or 0)
            short_liq = float(entry.get('shortLiquidations', 0) or 0)
            total_liq = float(entry.get('totalLiquidations', 0) or 0)
            
            if total_liq == 0:
                total_liq = long_liq + short_liq
            
            records.append({
                'timestamp': datetime.utcnow(),
                'symbol': symbol,
                'price_level': float(entry.get('price', 0)),
                'long_liquidations_usd': long_liq,
                'short_liquidations_usd': short_liq,
                'total_liquidations_usd': total_liq,
                'long_positions_at_risk': float(entry.get('longPositions', 0) or 0),
                'short_positions_at_risk': float(entry.get('shortPositions', 0) or 0),
                'interval': interval,
                'exchange': exchange or 'aggregated'
            })
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            # Calculate concentration metrics
            total_liq = df['total_liquidations_usd'].sum()
            if total_liq > 0:
                df['concentration_pct'] = df['total_liquidations_usd'] / total_liq * 100
            else:
                df['concentration_pct'] = 0
            
            # Calculate imbalance (positive = more longs at risk)
            df['liquidation_imbalance'] = (
                (df['long_liquidations_usd'] - df['short_liquidations_usd']) / 
                df['total_liquidations_usd'].replace(0, 1)
            )
            
            self.collection_stats['records_collected'] += len(df)
        
        return df
    
    async def fetch_liquidation_levels(
        self,
        symbol: str = 'BTCUSDT',
        current_price: Optional[float] = None,
        range_pct: float = 10.0
    ) -> Dict[str, Any]:
        """
        Analyze liquidation levels around current price.
        
        Returns key levels where liquidations would trigger.
        
        Args:
            symbol: Trading pair
            current_price: Current market price (fetched if None)
            range_pct: Percentage range to analyze (e.g., 10 = +/-10%)
            
        Returns:
            Dictionary with liquidation level analysis
        """
        heatmap = await self.fetch_liquidation_heatmap(symbol, '4h')
        
        if heatmap.empty:
            return {}
        
        if current_price is None:
            current_price = heatmap['price_level'].median()
        
        # Filter to range
        lower = current_price * (1 - range_pct / 100)
        upper = current_price * (1 + range_pct / 100)
        
        in_range = heatmap[
            (heatmap['price_level'] >= lower) &
            (heatmap['price_level'] <= upper)
        ].copy()
        
        if in_range.empty:
            return {
                'symbol': symbol,
                'current_price': current_price,
                'analysis_range': {'lower': lower, 'upper': upper},
                'warning': 'No liquidation data in range'
            }
        
        # Calculate distance from current price
        in_range['distance_pct'] = (
            (in_range['price_level'] - current_price) / current_price * 100
        )
        
        # Separate above and below
        above_price = in_range[in_range['price_level'] > current_price]
        below_price = in_range[in_range['price_level'] < current_price]
        
        # Find top liquidation levels
        top_above = above_price.nlargest(5, 'total_liquidations_usd') if not above_price.empty else pd.DataFrame()
        top_below = below_price.nlargest(5, 'total_liquidations_usd') if not below_price.empty else pd.DataFrame()
        
        # Calculate totals
        total_liq_above = above_price['total_liquidations_usd'].sum() if not above_price.empty else 0
        total_liq_below = below_price['total_liquidations_usd'].sum() if not below_price.empty else 0
        
        long_liq_above = above_price['long_liquidations_usd'].sum() if not above_price.empty else 0
        short_liq_above = above_price['short_liquidations_usd'].sum() if not above_price.empty else 0
        long_liq_below = below_price['long_liquidations_usd'].sum() if not below_price.empty else 0
        short_liq_below = below_price['short_liquidations_usd'].sum() if not below_price.empty else 0
        
        # Determine bias
        if total_liq_above > 0 and total_liq_below > 0:
            above_ratio = long_liq_above / (long_liq_above + short_liq_above) if (long_liq_above + short_liq_above) > 0 else 0.5
            below_ratio = long_liq_below / (long_liq_below + short_liq_below) if (long_liq_below + short_liq_below) > 0 else 0.5
            liq_bias = 'long_heavy' if above_ratio > 0.6 else ('short_heavy' if above_ratio < 0.4 else 'balanced')
        else:
            liq_bias = 'unknown'
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'analysis_range': {'lower': lower, 'upper': upper, 'range_pct': range_pct},
            'total_liq_above_usd': total_liq_above,
            'total_liq_below_usd': total_liq_below,
            'long_liq_above_usd': long_liq_above,
            'short_liq_above_usd': short_liq_above,
            'long_liq_below_usd': long_liq_below,
            'short_liq_below_usd': short_liq_below,
            'liquidation_bias': liq_bias,
            'key_levels_above': top_above[['price_level', 'total_liquidations_usd', 'concentration_pct', 'liquidation_imbalance', 'distance_pct']].to_dict('records') if not top_above.empty else [],
            'key_levels_below': top_below[['price_level', 'total_liquidations_usd', 'concentration_pct', 'liquidation_imbalance', 'distance_pct']].to_dict('records') if not top_below.empty else [],
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def identify_stop_hunt_zones(
        self,
        heatmap: pd.DataFrame,
        current_price: float,
        threshold_pct: float = 5.0
    ) -> List[LiquidationZone]:
        """
        Identify potential stop-hunt zones from liquidation heatmap.
        
        Stop-hunt zones are price levels with high liquidation concentration
        that may attract market manipulation.
        
        Args:
            heatmap: Liquidation heatmap DataFrame
            current_price: Current market price
            threshold_pct: Minimum concentration percentage to flag
            
        Returns:
            List of LiquidationZone objects
        """
        if heatmap.empty:
            return []
        
        zones = []
        total_liq = heatmap['total_liquidations_usd'].sum()
        
        for _, row in heatmap.iterrows():
            if total_liq == 0:
                continue
            
            concentration = row['total_liquidations_usd'] / total_liq * 100
            
            if concentration < threshold_pct:
                continue
            
            price = row['price_level']
            distance_pct = (price - current_price) / current_price * 100
            
            # Classify zone type
            imbalance = row.get('liquidation_imbalance', 0)
            
            if concentration >= self.STOP_HUNT_CONCENTRATION_THRESHOLD * 2:
                zone_type = LiquidationZoneType.STOP_HUNT
            elif price < current_price and imbalance > 0.3:
                zone_type = LiquidationZoneType.SUPPORT
            elif price > current_price and imbalance < -0.3:
                zone_type = LiquidationZoneType.RESISTANCE
            elif concentration >= self.STOP_HUNT_CONCENTRATION_THRESHOLD:
                zone_type = LiquidationZoneType.CASCADE_RISK
            else:
                zone_type = LiquidationZoneType.NEUTRAL
            
            # Risk score (0-100)
            risk_score = min(100, concentration * 10 + abs(imbalance) * 20)
            
            zones.append(LiquidationZone(
                price_level=price,
                price_range_low=price * 0.995,
                price_range_high=price * 1.005,
                long_liquidations_usd=row['long_liquidations_usd'],
                short_liquidations_usd=row['short_liquidations_usd'],
                total_liquidations_usd=row['total_liquidations_usd'],
                concentration_pct=concentration,
                zone_type=zone_type,
                risk_score=risk_score,
                distance_from_current_pct=distance_pct
            ))
        
        # Sort by risk score
        zones.sort(key=lambda x: x.risk_score, reverse=True)
        
        return zones
    
    async def calculate_risk_metrics(
        self,
        symbol: str = 'BTCUSDT',
        current_price: Optional[float] = None
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics based on liquidation data.
        
        Args:
            symbol: Trading pair
            current_price: Current market price
            
        Returns:
            RiskMetrics dataclass
        """
        heatmap = await self.fetch_liquidation_heatmap(symbol, '4h')
        
        if heatmap.empty:
            return RiskMetrics(
                symbol=symbol,
                current_price=current_price or 0,
                total_liq_at_risk_usd=0,
                long_liq_at_risk_usd=0,
                short_liq_at_risk_usd=0,
                nearest_liq_zone_distance_pct=100,
                liquidation_imbalance=0,
                cascade_risk_score=0,
                suggested_stop_distance_pct=5.0,
                timestamp=datetime.utcnow()
            )
        
        if current_price is None:
            current_price = heatmap['price_level'].median()
        
        # Liquidations within 5% of current price
        range_pct = 5.0
        lower = current_price * (1 - range_pct / 100)
        upper = current_price * (1 + range_pct / 100)
        
        nearby = heatmap[
            (heatmap['price_level'] >= lower) &
            (heatmap['price_level'] <= upper)
        ]
        
        total_at_risk = nearby['total_liquidations_usd'].sum() if not nearby.empty else 0
        long_at_risk = nearby['long_liquidations_usd'].sum() if not nearby.empty else 0
        short_at_risk = nearby['short_liquidations_usd'].sum() if not nearby.empty else 0
        
        # Find nearest high-concentration zone
        stop_hunt_zones = self.identify_stop_hunt_zones(heatmap, current_price)
        nearest_distance = min([abs(z.distance_from_current_pct) for z in stop_hunt_zones]) if stop_hunt_zones else 100
        
        # Calculate imbalance
        imbalance = (long_at_risk - short_at_risk) / max(total_at_risk, 1)
        
        # Cascade risk score
        cascade_risk = 0
        if stop_hunt_zones:
            # High concentration near current price = high cascade risk
            near_zones = [z for z in stop_hunt_zones if abs(z.distance_from_current_pct) < 3]
            if near_zones:
                cascade_risk = min(100, sum(z.risk_score for z in near_zones) / len(near_zones) * 1.5)
        
        # Suggested stop distance (beyond major liquidation zones)
        if stop_hunt_zones:
            # Find zones below current price
            below_zones = [z for z in stop_hunt_zones if z.distance_from_current_pct < 0]
            if below_zones:
                # Suggest stop beyond the nearest dangerous zone
                nearest_below = min(below_zones, key=lambda z: abs(z.distance_from_current_pct))
                suggested_stop = abs(nearest_below.distance_from_current_pct) + 1.0
            else:
                suggested_stop = 5.0
        else:
            suggested_stop = 5.0
        
        return RiskMetrics(
            symbol=symbol,
            current_price=current_price,
            total_liq_at_risk_usd=total_at_risk,
            long_liq_at_risk_usd=long_at_risk,
            short_liq_at_risk_usd=short_at_risk,
            nearest_liq_zone_distance_pct=nearest_distance,
            liquidation_imbalance=imbalance,
            cascade_risk_score=cascade_risk,
            suggested_stop_distance_pct=suggested_stop,
            timestamp=datetime.utcnow()
        )
    
    # =========================================================================
    # FUNDING AND L/S RATIOS
    # =========================================================================
    
    async def fetch_aggregated_funding(
        self,
        symbols: List[str],
        include_predicted: bool = True
    ) -> pd.DataFrame:
        """
        Fetch aggregated funding rates across all exchanges.
        
        Args:
            symbols: List of symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
            include_predicted: Include predicted next funding
            
        Returns:
            DataFrame with funding rate data
        """
        params = {
            'symbols': ','.join(symbols)
        }
        
        data = await self._make_request('funding-rate', params)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        entries = data if isinstance(data, list) else data.get('data', [])
        
        for entry in entries:
            rate = float(entry.get('fundingRate', 0) or 0)
            
            record = {
                'timestamp': datetime.utcnow(),
                'symbol': entry.get('symbol'),
                'funding_rate': rate,
                'funding_rate_annualized': rate * 3 * 365 * 100,
                'exchange': entry.get('exchange', 'aggregated'),
                'next_funding_time': pd.to_datetime(
                    entry.get('nextFundingTime'), unit='ms', utc=True
                ) if entry.get('nextFundingTime') else None
            }
            
            if include_predicted and 'predictedRate' in entry:
                predicted = float(entry.get('predictedRate', 0) or 0)
                record['predicted_funding'] = predicted
                record['predicted_annualized'] = predicted * 3 * 365 * 100
                record['funding_change'] = predicted - rate
            
            records.append(record)
        
        df = pd.DataFrame(records)
        self.collection_stats['records_collected'] += len(df)
        return df
    
    async def fetch_open_interest_distribution(
        self,
        symbol: str = 'BTCUSDT',
        exchange: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch open interest distribution by price level.
        
        Shows where positions are concentrated.
        
        Args:
            symbol: Trading pair
            exchange: Specific exchange or aggregated
            
        Returns:
            DataFrame with OI distribution
        """
        params = {'symbols': symbol}
        if exchange:
            params['exchange'] = exchange
        
        data = await self._make_request('open-interest-distribution', params)
        
        if not data or 'data' not in data:
            return pd.DataFrame()
        
        records = []
        for entry in data.get('data', []):
            long_oi = float(entry.get('longOI', 0) or 0)
            short_oi = float(entry.get('shortOI', 0) or 0)
            total_oi = float(entry.get('totalOI', 0) or 0)
            
            if total_oi == 0:
                total_oi = long_oi + short_oi
            
            records.append({
                'timestamp': datetime.utcnow(),
                'symbol': symbol,
                'price_level': float(entry.get('price', 0)),
                'long_oi_usd': long_oi,
                'short_oi_usd': short_oi,
                'total_oi_usd': total_oi,
                'exchange': exchange or 'aggregated'
            })
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            total_oi = df['total_oi_usd'].sum()
            if total_oi > 0:
                df['oi_concentration_pct'] = df['total_oi_usd'] / total_oi * 100
            else:
                df['oi_concentration_pct'] = 0
            
            df['long_short_ratio'] = df['long_oi_usd'] / df['short_oi_usd'].replace(0, 1)
            
            self.collection_stats['records_collected'] += len(df)
        
        return df
    
    async def fetch_long_short_ratio(
        self,
        symbols: List[str],
        interval: str = '1h'
    ) -> pd.DataFrame:
        """
        Fetch long/short ratio history.
        
        Args:
            symbols: List of trading pairs
            interval: Time interval
            
        Returns:
            DataFrame with L/S ratio data
        """
        params = {
            'symbols': ','.join(symbols),
            'interval': interval
        }
        
        data = await self._make_request('long-short-ratio', params)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        entries = data if isinstance(data, list) else data.get('data', [])
        
        for entry in entries:
            long_ratio = float(entry.get('longRatio', 50) or 50)
            short_ratio = float(entry.get('shortRatio', 50) or 50)
            
            # Determine sentiment
            if long_ratio >= 60:
                sentiment = 'strongly_bullish'
            elif long_ratio >= 55:
                sentiment = 'bullish'
            elif long_ratio <= 40:
                sentiment = 'strongly_bearish'
            elif long_ratio <= 45:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'
            
            records.append({
                'timestamp': pd.to_datetime(entry.get('timestamp'), unit='ms', utc=True) if entry.get('timestamp') else datetime.utcnow(),
                'symbol': entry.get('symbol'),
                'long_ratio': long_ratio,
                'short_ratio': short_ratio,
                'ls_ratio': long_ratio / max(short_ratio, 1),
                'sentiment': sentiment,
                'exchange': entry.get('exchange', 'aggregated')
            })
        
        df = pd.DataFrame(records)
        self.collection_stats['records_collected'] += len(df)
        return df
    
    # =========================================================================
    # HISTORICAL LIQUIDATIONS
    # =========================================================================
    
    async def fetch_historical_liquidations(
        self,
        symbol: str = 'BTCUSDT',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        exchange: Optional[str] = None,
        interval: str = '1h'
    ) -> pd.DataFrame:
        """
        Fetch historical liquidation events.
        
        Args:
            symbol: Trading pair
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            exchange: Specific exchange or aggregated
            interval: Aggregation interval
            
        Returns:
            DataFrame with historical liquidations
        """
        params = {'symbols': symbol, 'interval': interval}
        
        if start_date:
            params['from'] = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        if end_date:
            params['to'] = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        if exchange:
            params['exchange'] = exchange
        
        data = await self._make_request('liquidation-history', params)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        entries = data if isinstance(data, list) else data.get('data', [])
        
        for entry in entries:
            records.append({
                'timestamp': pd.to_datetime(entry.get('timestamp'), unit='ms', utc=True) if entry.get('timestamp') else None,
                'symbol': entry.get('symbol', symbol),
                'side': entry.get('side'),
                'quantity': float(entry.get('quantity', 0) or 0),
                'price': float(entry.get('price', 0) or 0),
                'value_usd': float(entry.get('value', 0) or 0),
                'exchange': entry.get('exchange', exchange or 'aggregated')
            })
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
            self.collection_stats['records_collected'] += len(df)
        
        return df
    
    async def fetch_liquidation_summary(
        self,
        symbols: List[str],
        interval: str = '24h'
    ) -> pd.DataFrame:
        """
        Fetch liquidation summary statistics.
        
        Args:
            symbols: List of trading pairs
            interval: Time period ('1h', '4h', '24h')
            
        Returns:
            DataFrame with liquidation summary
        """
        params = {
            'symbols': ','.join(symbols),
            'interval': interval
        }
        
        data = await self._make_request('liquidation-summary', params)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        entries = data if isinstance(data, list) else data.get('data', [])
        
        for entry in entries:
            long_liq = float(entry.get('longLiquidations', 0) or 0)
            short_liq = float(entry.get('shortLiquidations', 0) or 0)
            total_liq = long_liq + short_liq
            
            records.append({
                'timestamp': datetime.utcnow(),
                'symbol': entry.get('symbol'),
                'interval': interval,
                'long_liquidations_usd': long_liq,
                'short_liquidations_usd': short_liq,
                'total_liquidations_usd': total_liq,
                'liq_ratio': long_liq / max(short_liq, 1),
                'dominant_side': 'longs' if long_liq > short_liq else 'shorts',
                'exchange': entry.get('exchange', 'aggregated')
            })
        
        df = pd.DataFrame(records)
        self.collection_stats['records_collected'] += len(df)
        return df
    
    # =========================================================================
    # COMPREHENSIVE ANALYSIS
    # =========================================================================
    
    async def fetch_comprehensive_liquidation_data(
        self,
        symbol: str = 'BTCUSDT',
        current_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Fetch comprehensive liquidation analysis for a symbol.
        
        Returns complete picture of liquidation landscape.
        
        Args:
            symbol: Trading pair
            current_price: Current market price
            
        Returns:
            Dictionary with all liquidation analysis
        """
        results = {}
        
        logger.info(f"Fetching comprehensive liquidation data for {symbol}")
        
        # Liquidation heatmap
        results['heatmap'] = await self.fetch_liquidation_heatmap(symbol, '4h')
        
        # Key liquidation levels
        results['levels'] = await self.fetch_liquidation_levels(symbol, current_price)
        
        # Stop-hunt zones
        if not results['heatmap'].empty:
            cp = current_price or results['heatmap']['price_level'].median()
            results['stop_hunt_zones'] = [
                {
                    'price_level': z.price_level,
                    'total_liquidations': z.total_liquidations_usd,
                    'concentration_pct': z.concentration_pct,
                    'zone_type': z.zone_type.value,
                    'risk_score': z.risk_score,
                    'distance_pct': z.distance_from_current_pct
                }
                for z in self.identify_stop_hunt_zones(results['heatmap'], cp)[:10]
            ]
        else:
            results['stop_hunt_zones'] = []
        
        # Risk metrics
        results['risk_metrics'] = await self.calculate_risk_metrics(symbol, current_price)
        
        # OI distribution
        results['oi_distribution'] = await self.fetch_open_interest_distribution(symbol)
        
        # L/S ratio
        results['ls_ratio'] = await self.fetch_long_short_ratio([symbol])
        
        # 24h liquidation summary
        results['summary_24h'] = await self.fetch_liquidation_summary([symbol], '24h')
        
        # Aggregated funding
        results['funding'] = await self.fetch_aggregated_funding([symbol])
        
        return results
    
    # =========================================================================
    # REQUIRED ABSTRACT METHODS
    # =========================================================================
    
    async def fetch_funding_rates(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch funding rates using aggregated endpoint."""
        return await self.fetch_aggregated_funding(symbols)
    
    async def fetch_ohlcv(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Coinalyze Enhanced doesn't provide direct OHLCV. Use base Coinalyze."""
        logger.info("Use CoinalyzeCollector for OHLCV data")
        return pd.DataFrame()
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return self.collection_stats
    
    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
        logger.info(f"CoinalyzeEnhanced collector closed. Stats: {self.collection_stats}")

# =============================================================================
# Testing
# =============================================================================

async def test_coinalyze_enhanced():
    """Test enhanced Coinalyze collector."""
    config = {
        'rate_limit': 20,
        'coinalyze_api_key': ''
    }
    
    collector = CoinalyzeEnhancedCollector(config)
    
    try:
        print("Testing Enhanced Coinalyze Collector")
        print("=" * 60)
        
        # Test liquidation summary
        print("\n1. Testing liquidation summary...")
        summary = await collector.fetch_liquidation_summary(['BTCUSDT', 'ETHUSDT'], '24h')
        if not summary.empty:
            print(f" Found {len(summary)} records")
            for _, row in summary.iterrows():
                print(f" {row['symbol']}: ${row['total_liquidations_usd']:,.0f} total ({row['dominant_side']})")
        
        # Test L/S ratio
        print("\n2. Testing long/short ratio...")
        ls_ratio = await collector.fetch_long_short_ratio(['BTCUSDT'])
        if not ls_ratio.empty:
            print(f" Found {len(ls_ratio)} records")
            print(f" Sentiment: {ls_ratio.iloc[0]['sentiment']}")
        
        # Test aggregated funding
        print("\n3. Testing aggregated funding...")
        funding = await collector.fetch_aggregated_funding(['BTCUSDT', 'ETHUSDT'])
        if not funding.empty:
            print(f" Found {len(funding)} records")
            for _, row in funding.iterrows():
                print(f" {row['symbol']}: {row['funding_rate_annualized']:.2f}% ann.")
        
        # Test liquidation heatmap
        print("\n4. Testing liquidation heatmap...")
        heatmap = await collector.fetch_liquidation_heatmap('BTCUSDT', '4h')
        if not heatmap.empty:
            print(f" Found {len(heatmap)} price levels")
            top = heatmap.nlargest(3, 'total_liquidations_usd')
            for _, row in top.iterrows():
                print(f" ${row['price_level']:,.0f}: ${row['total_liquidations_usd']:,.0f} ({row['concentration_pct']:.1f}%)")
        
        # Test risk metrics
        print("\n5. Testing risk metrics...")
        risk = await collector.calculate_risk_metrics('BTCUSDT')
        print(f" Current price: ${risk.current_price:,.0f}")
        print(f" Total at risk: ${risk.total_liq_at_risk_usd:,.0f}")
        print(f" Cascade risk score: {risk.cascade_risk_score:.0f}")
        print(f" Suggested stop: {risk.suggested_stop_distance_pct:.1f}%")
        
        print("\n" + "=" * 60)
        print(f"Collection stats: {collector.get_collection_stats()}")
        
    finally:
        await collector.close()

if __name__ == '__main__':
    asyncio.run(test_coinalyze_enhanced())