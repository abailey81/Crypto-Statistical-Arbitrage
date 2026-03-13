"""
Santiment On-Chain and Social Metrics Collector

validated collector for on-chain metrics, social sentiment, and
development activity data from Santiment. Comprehensive alternative data
for market analysis and sentiment tracking.

===============================================================================
SANTIMENT OVERVIEW
===============================================================================

Santiment provides unique alternative data combining on-chain metrics with
social and development signals. Known for sentiment analysis, trending topics,
and developer activity tracking.

Key Differentiators:
    - Social volume and sentiment analysis
    - Development activity tracking (GitHub)
    - Holder distribution analysis
    - Network growth metrics
    - Trending words and topics
    - Whale transaction alerts

===============================================================================
API SPECIFICATIONS
===============================================================================

Base URL: https://api.santiment.net/graphql

Authentication:
    - API Key in Authorization header
    - GraphQL query interface

Rate Limits by Tier:
    ============ ============== ================ ===============
    Tier Requests/min Historical Data Features
    ============ ============== ================ ===============
    Free Limited 30 days Basic metrics
    Pro 100 2 years Full metrics
    Enterprise Unlimited Full history All features
    ============ ============== ================ ===============

===============================================================================
METRIC CATEGORIES
===============================================================================

On-Chain Metrics:
    - daily_active_addresses: User engagement
    - transaction_volume: Network usage
    - exchange_inflow/outflow: Exchange flows
    - circulation, velocity: Money supply dynamics

Social Metrics:
    - social_volume_total: Total social mentions
    - social_dominance: Share of crypto social
    - sentiment_balance: Positive vs negative
    - trending_words: Current hot topics

Development Metrics:
    - dev_activity: GitHub commits/activity
    - dev_activity_contributors_count: Active developers
    - github_activity: Raw GitHub metrics

Network Metrics:
    - nvt: Network Value to Transactions
    - mvrv_ratio: Market Value to Realized Value
    - network_growth: New address creation
    - realized_value: Realized capitalization

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

Sentiment Signals:
    - Social volume spikes for volatility prediction
    - Sentiment divergence from price
    - Trending topics for emerging narratives

Development Signals:
    - Active development as quality indicator
    - Developer count trends
    - Commit frequency analysis

Flow Analysis:
    - Exchange flow divergence
    - Velocity changes
    - Circulation patterns

Version: 2.0.0
"""

import os
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any

import aiohttp
import pandas as pd
import numpy as np

# =============================================================================
# ENUMS
# =============================================================================

class MetricType(Enum):
    """Santiment metric types."""
    ONCHAIN = 'onchain'
    SOCIAL = 'social'
    DEVELOPMENT = 'development'
    NETWORK = 'network'
    FINANCIAL = 'financial'

class SocialMetric(Enum):
    """Social metrics available."""
    SOCIAL_VOLUME_TOTAL = 'socialVolumeTotal'
    SOCIAL_DOMINANCE = 'socialDominance'
    SENTIMENT_BALANCE = 'sentimentBalance'
    SENTIMENT_VOLUME_CONSUMED = 'sentimentVolumeConsumed'
    SENTIMENT_POSITIVE = 'sentimentPositive'
    SENTIMENT_NEGATIVE = 'sentimentNegative'

class OnChainMetric(Enum):
    """On-chain metrics available."""
    DAILY_ACTIVE_ADDRESSES = 'dailyActiveAddresses'
    TRANSACTION_VOLUME = 'transactionVolume'
    EXCHANGE_INFLOW = 'exchangeInflow'
    EXCHANGE_OUTFLOW = 'exchangeOutflow'
    EXCHANGE_BALANCE = 'exchangeBalance'
    AGE_CONSUMED = 'ageConsumed'
    CIRCULATION = 'circulation'
    VELOCITY = 'velocity'

class DevelopmentMetric(Enum):
    """Development metrics available."""
    DEV_ACTIVITY = 'devActivity'
    DEV_ACTIVITY_CONTRIBUTORS = 'devActivityContributorsCount'
    GITHUB_ACTIVITY = 'githubActivity'

class NetworkMetric(Enum):
    """Network/valuation metrics available."""
    NVT = 'nvtRatio'
    MVRV_RATIO = 'mvrvRatio'
    NETWORK_GROWTH = 'networkGrowth'
    REALIZED_VALUE = 'realizedValue'

class FinancialMetric(Enum):
    """Financial metrics available."""
    PRICE_USD = 'price_usd'
    VOLUME_USD = 'volume_usd'
    MARKETCAP_USD = 'marketcap_usd'

class SentimentLevel(Enum):
    """Sentiment level classification."""
    EXTREME_FEAR = 'extreme_fear'
    FEAR = 'fear'
    NEUTRAL = 'neutral'
    GREED = 'greed'
    EXTREME_GREED = 'extreme_greed'

class SocialActivity(Enum):
    """Social activity level."""
    VERY_LOW = 'very_low'
    LOW = 'low'
    NORMAL = 'normal'
    HIGH = 'high'
    VIRAL = 'viral'

class DevelopmentHealth(Enum):
    """Development health assessment."""
    INACTIVE = 'inactive'
    LOW = 'low'
    MODERATE = 'moderate'
    ACTIVE = 'active'
    VERY_ACTIVE = 'very_active'

class FlowDirection(Enum):
    """Exchange flow direction."""
    STRONG_INFLOW = 'strong_inflow'
    INFLOW = 'inflow'
    NEUTRAL = 'neutral'
    OUTFLOW = 'outflow'
    STRONG_OUTFLOW = 'strong_outflow'

class Interval(Enum):
    """Data intervals."""
    HOURLY = '1h'
    DAILY = '1d'
    WEEKLY = '1w'

# =============================================================================
# ASSET SLUG MAPPING
# =============================================================================

ASSET_SLUGS = {
    # Major cryptocurrencies
    'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana', 'AVAX': 'avalanche',
    'MATIC': 'polygon', 'ARB': 'arbitrum', 'OP': 'optimism', 'LINK': 'chainlink',
    'DOGE': 'dogecoin', 'XRP': 'ripple', 'ADA': 'cardano', 'DOT': 'polkadot',

    # DeFi tokens
    'UNI': 'uniswap', 'AAVE': 'aave', 'MKR': 'maker', 'CRV': 'curve-dao-token',
    'SNX': 'synthetix', 'COMP': 'compound', 'LDO': 'lido-dao', 'GMX': 'gmx',
    'DYDX': 'dydx', 'SUSHI': 'sushi', 'BAL': 'balancer', 'YFI': 'yearn-finance',
    'FXS': 'frax-share',

    # Layer 1/2
    'ATOM': 'cosmos', 'NEAR': 'near-protocol', 'FTM': 'fantom',
    'APT': 'aptos', 'SUI': 'sui', 'SEI': 'sei-network',

    # Other tokens
    'APE': 'apecoin', 'BLUR': 'blur', 'WLD': 'worldcoin',
    'INJ': 'injective-protocol', 'RPL': 'rocket-pool',
}

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class SocialData:
    """Social metrics data with sentiment analysis."""
    timestamp: datetime
    slug: str
    social_volume: float = 0.0
    social_dominance: float = 0.0
    sentiment_balance: float = 0.0
    sentiment_positive: float = 0.0
    sentiment_negative: float = 0.0
    
    @property
    def sentiment_level(self) -> SentimentLevel:
        """Classify sentiment level."""
        if self.sentiment_balance < -0.3:
            return SentimentLevel.EXTREME_FEAR
        elif self.sentiment_balance < -0.1:
            return SentimentLevel.FEAR
        elif self.sentiment_balance <= 0.1:
            return SentimentLevel.NEUTRAL
        elif self.sentiment_balance <= 0.3:
            return SentimentLevel.GREED
        return SentimentLevel.EXTREME_GREED
    
    @property
    def activity_level(self) -> SocialActivity:
        """Classify social activity level."""
        if self.social_volume < 10:
            return SocialActivity.VERY_LOW
        elif self.social_volume < 100:
            return SocialActivity.LOW
        elif self.social_volume < 1000:
            return SocialActivity.NORMAL
        elif self.social_volume < 10000:
            return SocialActivity.HIGH
        return SocialActivity.VIRAL
    
    @property
    def is_bullish(self) -> bool:
        """Check if sentiment is bullish."""
        return self.sentiment_balance > 0.1
    
    @property
    def is_bearish(self) -> bool:
        """Check if sentiment is bearish."""
        return self.sentiment_balance < -0.1
    
    @property
    def is_viral(self) -> bool:
        """Check if topic is viral."""
        return self.social_volume >= 10000 or self.social_dominance >= 10
    
    @property
    def sentiment_ratio(self) -> float:
        """Ratio of positive to negative sentiment."""
        if self.sentiment_negative > 0:
            return self.sentiment_positive / self.sentiment_negative
        return float('inf') if self.sentiment_positive > 0 else 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'slug': self.slug, 'social_volume': self.social_volume,
            'social_dominance': self.social_dominance, 'sentiment_balance': self.sentiment_balance,
            'sentiment_level': self.sentiment_level.value, 'activity_level': self.activity_level.value,
            'is_bullish': self.is_bullish, 'is_bearish': self.is_bearish, 'is_viral': self.is_viral,
        }

@dataclass
class DevelopmentData:
    """Development activity data with health assessment."""
    timestamp: datetime
    slug: str
    dev_activity: float = 0.0
    contributors_count: int = 0
    github_activity: float = 0.0
    
    @property
    def health(self) -> DevelopmentHealth:
        """Assess development health."""
        if self.dev_activity < 1:
            return DevelopmentHealth.INACTIVE
        elif self.dev_activity < 10:
            return DevelopmentHealth.LOW
        elif self.dev_activity < 50:
            return DevelopmentHealth.MODERATE
        elif self.dev_activity < 100:
            return DevelopmentHealth.ACTIVE
        return DevelopmentHealth.VERY_ACTIVE
    
    @property
    def is_actively_developed(self) -> bool:
        """Check if actively developed."""
        return self.dev_activity >= 10 and self.contributors_count >= 5
    
    @property
    def contributor_productivity(self) -> float:
        """Activity per contributor."""
        if self.contributors_count > 0:
            return self.dev_activity / self.contributors_count
        return 0.0
    
    @property
    def quality_score(self) -> float:
        """Development quality score (0-100)."""
        activity_score = min(50, self.dev_activity / 2)
        contributor_score = min(30, self.contributors_count * 3)
        productivity_score = min(20, self.contributor_productivity * 5)
        return activity_score + contributor_score + productivity_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'slug': self.slug, 'dev_activity': self.dev_activity,
            'contributors_count': self.contributors_count, 'health': self.health.value,
            'is_actively_developed': self.is_actively_developed, 'quality_score': self.quality_score,
        }

@dataclass
class OnChainData:
    """On-chain metrics data with flow analysis."""
    timestamp: datetime
    slug: str
    active_addresses: int = 0
    transaction_volume: float = 0.0
    exchange_inflow: float = 0.0
    exchange_outflow: float = 0.0
    circulation: float = 0.0
    velocity: float = 0.0
    
    @property
    def net_exchange_flow(self) -> float:
        """Net exchange flow."""
        return self.exchange_inflow - self.exchange_outflow
    
    @property
    def flow_direction(self) -> FlowDirection:
        """Classify flow direction."""
        net = self.net_exchange_flow
        if self.exchange_inflow + self.exchange_outflow == 0:
            return FlowDirection.NEUTRAL
        
        ratio = net / (self.exchange_inflow + self.exchange_outflow)
        
        if ratio > 0.3:
            return FlowDirection.STRONG_INFLOW
        elif ratio > 0.1:
            return FlowDirection.INFLOW
        elif ratio < -0.3:
            return FlowDirection.STRONG_OUTFLOW
        elif ratio < -0.1:
            return FlowDirection.OUTFLOW
        return FlowDirection.NEUTRAL
    
    @property
    def is_accumulation(self) -> bool:
        """Check if accumulation (outflows > inflows)."""
        return self.net_exchange_flow < 0
    
    @property
    def is_distribution(self) -> bool:
        """Check if distribution (inflows > outflows)."""
        return self.net_exchange_flow > 0
    
    @property
    def network_usage(self) -> str:
        """Classify network usage."""
        if self.active_addresses > 1_000_000:
            return 'very_high'
        elif self.active_addresses > 500_000:
            return 'high'
        elif self.active_addresses > 100_000:
            return 'moderate'
        elif self.active_addresses > 10_000:
            return 'low'
        return 'minimal'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'slug': self.slug, 'active_addresses': self.active_addresses,
            'transaction_volume': self.transaction_volume, 'net_exchange_flow': self.net_exchange_flow,
            'flow_direction': self.flow_direction.value, 'is_accumulation': self.is_accumulation,
            'network_usage': self.network_usage,
        }

@dataclass
class NetworkData:
    """Network valuation metrics."""
    timestamp: datetime
    slug: str
    nvt: float = 0.0
    mvrv_ratio: float = 0.0
    network_growth: float = 0.0
    realized_value: float = 0.0
    
    @property
    def nvt_signal(self) -> str:
        """NVT valuation signal."""
        if self.nvt < 20:
            return 'undervalued'
        elif self.nvt < 50:
            return 'fair'
        elif self.nvt < 100:
            return 'overvalued'
        return 'extreme_overvalued'
    
    @property
    def mvrv_signal(self) -> str:
        """MVRV valuation signal."""
        if self.mvrv_ratio < 0.8:
            return 'deep_value'
        elif self.mvrv_ratio < 1.0:
            return 'undervalued'
        elif self.mvrv_ratio < 2.5:
            return 'fair'
        elif self.mvrv_ratio < 4.0:
            return 'overvalued'
        return 'extreme_overvalued'
    
    @property
    def is_growing(self) -> bool:
        """Check if network is growing."""
        return self.network_growth > 0
    
    @property
    def growth_rate(self) -> str:
        """Classify growth rate."""
        if self.network_growth > 5:
            return 'rapid'
        elif self.network_growth > 2:
            return 'strong'
        elif self.network_growth > 0:
            return 'moderate'
        elif self.network_growth > -2:
            return 'stagnant'
        return 'declining'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'slug': self.slug, 'nvt': self.nvt, 'mvrv_ratio': self.mvrv_ratio,
            'nvt_signal': self.nvt_signal, 'mvrv_signal': self.mvrv_signal,
            'is_growing': self.is_growing, 'growth_rate': self.growth_rate,
        }

@dataclass
class TrendingWord:
    """Trending word data."""
    timestamp: datetime
    rank: int
    word: str
    score: float
    
    @property
    def is_top_trend(self) -> bool:
        """Check if top trending."""
        return self.rank <= 3
    
    @property
    def trend_strength(self) -> str:
        """Classify trend strength."""
        if self.score > 1000:
            return 'viral'
        elif self.score > 500:
            return 'strong'
        elif self.score > 100:
            return 'moderate'
        return 'weak'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'rank': self.rank, 'word': self.word, 'score': self.score,
            'is_top_trend': self.is_top_trend, 'trend_strength': self.trend_strength,
        }

# =============================================================================
# COLLECTOR CLASS
# =============================================================================

class SantimentCollector:
    """
    Santiment on-chain and social metrics collector.

    Features:
    - On-chain metrics: Active addresses, transaction volume, exchange flows
    - Social metrics: Social volume, sentiment, trending words
    - Development: GitHub activity, contributors
    - Network: NVT, MVRV, network growth

    RATE LIMITS (per Santiment docs):
    - Free tier: ~10 requests/min, 1000/month (VERY restrictive!)
    - Pro tier: 100 requests/min
    - Each GraphQL query counts as 1 API call
    - 429 response means rate limit exceeded

    IMPORTANT: Free tier is almost unusable for multi-symbol collection.
    This collector will disable itself after hitting rate limits to avoid
    wasting time with endless retries.
    """

    VENUE = 'santiment'
    VENUE_TYPE = 'alternative'
    BASE_URL = 'https://api.santiment.net/graphql'

    SLUGS = ASSET_SLUGS

    # Rate limiting constants
    MAX_CONCURRENT_REQUESTS = 5 # Allow parallel requests
    REQUEST_DELAY_SECONDS = 2.0 # Conservative delay between requests
    RATE_LIMIT_BACKOFF = 65 # Seconds to wait after 429

    # Auto-disable thresholds - IMMEDIATELY skip if rate limited
    # If we hit even 1 rate limit, likely at monthly quota - disable immediately
    MAX_CONSECUTIVE_RATE_LIMITS = 1 # Disable after first 429 (monthly quota likely hit)
    MAX_TOTAL_RATE_LIMITS = 2 # Disable after 2 total 429s in session

    def __init__(self, config: Dict):
        """Initialize Santiment collector."""
        self.api_key = config.get('api_key') or config.get('santiment_api_key') or os.getenv('SANTIMENT_API_KEY', '')
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.collection_stats = {'requests': 0, 'records': 0, 'errors': 0, 'rate_limits': 0}

        # CRITICAL: Semaphore to limit concurrent requests (now set to 1 for sequential)
        self._request_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)
        self._last_request_time = 0.0
        self._rate_limited_until = 0.0

        # Auto-disable mechanism - skip immediately if rate limited
        self._disabled = False
        self._disable_reason = ""
        self._consecutive_rate_limits = 0
        self._logged_disabled_warning = False # Only log DISABLED message once
    
    async def __aenter__(self) -> 'SantimentCollector':
        """Async context manager entry."""
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            headers = {'Content-Type': 'application/json', 'Authorization': f'Apikey {self.api_key}'}
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self.session
    
    def _get_slug(self, symbol: str) -> str:
        """Convert symbol to Santiment slug."""
        return self.SLUGS.get(symbol.upper(), symbol.lower())
    
    def is_disabled(self) -> bool:
        """Check if collector is disabled due to rate limiting."""
        return self._disabled

    def _check_should_disable(self) -> bool:
        """Check if we should disable due to rate limits and set flag if so."""
        if self._disabled:
            return True

        # Check if we hit rate limits - likely monthly quota exhausted
        if self._consecutive_rate_limits >= self.MAX_CONSECUTIVE_RATE_LIMITS:
            self._disabled = True
            self._disable_reason = "Rate limit hit - likely monthly quota exhausted (1000 calls/month free tier)"
            if not self._logged_disabled_warning:
                self._logged_disabled_warning = True
                self.logger.warning(f"Santiment DISABLED: {self._disable_reason}")
            return True

        # Check total rate limits in session
        if self.collection_stats['rate_limits'] >= self.MAX_TOTAL_RATE_LIMITS:
            self._disabled = True
            self._disable_reason = "Multiple rate limits hit - skipping to save time"
            if not self._logged_disabled_warning:
                self._logged_disabled_warning = True
                self.logger.warning(f"Santiment DISABLED: {self._disable_reason}")
            return True

        return False

    async def _execute_query(self, query: str) -> Dict:
        """
        Execute GraphQL query against Santiment API.

        CRITICAL: Auto-disables on first rate limit hit (likely monthly quota exhausted).
        """
        import time

        # CRITICAL: Check if disabled - skip immediately
        if self._check_should_disable():
            return {}

        # Use semaphore to limit concurrent requests
        async with self._request_semaphore:
            # Small delay between requests
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            if time_since_last < self.REQUEST_DELAY_SECONDS:
                await asyncio.sleep(self.REQUEST_DELAY_SECONDS - time_since_last)

            self._last_request_time = time.time()
            session = await self._get_session()
            self.collection_stats['requests'] += 1

            try:
                async with session.post(self.BASE_URL, json={'query': query}) as response:
                    if response.status == 200:
                        # Success - reset consecutive rate limit counter
                        self._consecutive_rate_limits = 0
                        result = await response.json()
                        if 'errors' in result:
                            error_msg = str(result['errors'])[:100]
                            self.logger.debug(f"GraphQL errors: {error_msg}")
                            self.collection_stats['errors'] += 1
                            return {}
                        return result.get('data', {})
                    elif response.status == 429:
                        # Rate limit hit - IMMEDIATELY disable (likely monthly quota exhausted)
                        self._consecutive_rate_limits += 1
                        self.collection_stats['rate_limits'] += 1
                        self._disabled = True
                        self._disable_reason = "Rate limit hit (429) - monthly quota likely exhausted"
                        # Only log once to avoid spam from parallel requests
                        if not self._logged_disabled_warning:
                            self._logged_disabled_warning = True
                            self.logger.warning(
                                f"Santiment DISABLED: {self._disable_reason}. "
                                f"Free tier = 1000 calls/month. Skipping remaining requests."
                            )
                        return {}
                    elif response.status == 401:
                        # Invalid/missing API key - disable immediately
                        self._disabled = True
                        self._disable_reason = "API key invalid or missing"
                        self.logger.error(f"Santiment DISABLED: {self._disable_reason}")
                        self.collection_stats['errors'] += 1
                        return {}
                    else:
                        self.logger.debug(f"Santiment API error: HTTP {response.status}")
                        self.collection_stats['errors'] += 1
                        return {}
            except asyncio.TimeoutError:
                self.logger.debug("Santiment API timeout")
                self.collection_stats['errors'] += 1
                return {}
            except Exception as e:
                self.logger.debug(f"Santiment request error: {e}")
                self.collection_stats['errors'] += 1
                return {}
    
    async def fetch_metric(
        self, slug: str, metric: str, start_date: str,
        end_date: str, interval: str = '1d'
    ) -> pd.DataFrame:
        """Fetch a specific metric for an asset."""
        query = f'''
        {{
            getMetric(metric: "{metric}") {{
                timeseriesData(
                    slug: "{slug}"
                    from: "{start_date}T00:00:00Z"
                    to: "{end_date}T23:59:59Z"
                    interval: "{interval}"
                ) {{
                    datetime
                    value
                }}
            }}
        }}
        '''
        
        data = await self._execute_query(query)
        timeseries = data.get('getMetric', {}).get('timeseriesData', [])
        
        if not timeseries:
            return pd.DataFrame()
        
        records = [{
            'timestamp': pd.to_datetime(p['datetime']),
            'slug': slug, 'metric': metric,
            'value': float(p['value']) if p['value'] is not None else None,
            'venue': self.VENUE,
        } for p in timeseries]
        
        self.collection_stats['records'] += len(records)
        return pd.DataFrame(records)
    
    async def fetch_social_metrics(
        self, slug: str, start_date: str, end_date: str, interval: str = '1d'
    ) -> pd.DataFrame:
        """Fetch social metrics for an asset."""
        if self._check_should_disable():
            return pd.DataFrame()

        metrics = ['socialVolumeTotal', 'socialDominance', 'sentimentBalance']
        all_data = []

        for metric in metrics:
            if self._check_should_disable():
                break
            df = await self.fetch_metric(slug, metric, start_date, end_date, interval)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        result = all_data[0]
        for df in all_data[1:]:
            metric_name = df['metric'].iloc[0]
            result = result.merge(
                df[['timestamp', 'value']].rename(columns={'value': metric_name}),
                on='timestamp', how='outer'
            )
        
        # Add sentiment classification
        if 'sentimentBalance' in result.columns:
            result['sentiment_level'] = result['sentimentBalance'].apply(
                lambda v: SentimentLevel.EXTREME_FEAR.value if v < -0.3 else
                          SentimentLevel.FEAR.value if v < -0.1 else
                          SentimentLevel.NEUTRAL.value if v <= 0.1 else
                          SentimentLevel.GREED.value if v <= 0.3 else
                          SentimentLevel.EXTREME_GREED.value)
            result['is_bullish'] = result['sentimentBalance'] > 0.1
        
        return result.sort_values('timestamp')
    
    async def fetch_development_metrics(
        self, slug: str, start_date: str, end_date: str, interval: str = '1d'
    ) -> pd.DataFrame:
        """Fetch development activity metrics."""
        if self._check_should_disable():
            return pd.DataFrame()

        metrics = ['devActivity', 'devActivityContributorsCount', 'githubActivity']
        all_data = []

        for metric in metrics:
            if self._check_should_disable():
                break
            df = await self.fetch_metric(slug, metric, start_date, end_date, interval)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        result = all_data[0]
        for df in all_data[1:]:
            metric_name = df['metric'].iloc[0]
            result = result.merge(
                df[['timestamp', 'value']].rename(columns={'value': metric_name}),
                on='timestamp', how='outer'
            )
        
        # Add health classification
        if 'devActivity' in result.columns:
            result['health'] = result['devActivity'].apply(
                lambda v: DevelopmentHealth.INACTIVE.value if v < 1 else
                          DevelopmentHealth.LOW.value if v < 10 else
                          DevelopmentHealth.MODERATE.value if v < 50 else
                          DevelopmentHealth.ACTIVE.value if v < 100 else
                          DevelopmentHealth.VERY_ACTIVE.value)
        
        return result.sort_values('timestamp')
    
    async def fetch_onchain_metrics(
        self, slug: str, start_date: str, end_date: str, interval: str = '1d'
    ) -> pd.DataFrame:
        """Fetch on-chain metrics."""
        if self._check_should_disable():
            return pd.DataFrame()

        metrics = ['dailyActiveAddresses', 'transactionVolume', 'exchangeInflow',
                   'exchangeOutflow', 'circulation', 'velocity']
        all_data = []

        for metric in metrics:
            if self._check_should_disable():
                break
            df = await self.fetch_metric(slug, metric, start_date, end_date, interval)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        result = all_data[0]
        for df in all_data[1:]:
            metric_name = df['metric'].iloc[0]
            result = result.merge(
                df[['timestamp', 'value']].rename(columns={'value': metric_name}),
                on='timestamp', how='outer'
            )
        
        # Add flow analysis
        if 'exchangeInflow' in result.columns and 'exchangeOutflow' in result.columns:
            result['net_exchange_flow'] = result['exchangeInflow'] - result['exchangeOutflow']
            result['is_accumulation'] = result['net_exchange_flow'] < 0
        
        return result.sort_values('timestamp')
    
    async def fetch_network_metrics(
        self, slug: str, start_date: str, end_date: str, interval: str = '1d'
    ) -> pd.DataFrame:
        """Fetch network valuation metrics."""
        if self._check_should_disable():
            return pd.DataFrame()

        metrics = ['nvtRatio', 'mvrvRatio', 'networkGrowth', 'realizedValue']
        all_data = []

        for metric in metrics:
            if self._check_should_disable():
                break
            df = await self.fetch_metric(slug, metric, start_date, end_date, interval)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        result = all_data[0]
        for df in all_data[1:]:
            metric_name = df['metric'].iloc[0]
            result = result.merge(
                df[['timestamp', 'value']].rename(columns={'value': metric_name}),
                on='timestamp', how='outer'
            )
        
        # Add valuation signals
        if 'mvrvRatio' in result.columns:
            result['mvrv_signal'] = result['mvrvRatio'].apply(
                lambda v: 'deep_value' if v < 0.8 else 'undervalued' if v < 1.0 else
                          'fair' if v < 2.5 else 'overvalued' if v < 4.0 else 'extreme_overvalued')
        
        return result.sort_values('timestamp')
    
    async def fetch_exchange_flows(
        self, slug: str, start_date: str, end_date: str, interval: str = '1d'
    ) -> pd.DataFrame:
        """Fetch exchange flow data."""
        if self._check_should_disable():
            return pd.DataFrame()

        metrics = ['exchangeInflow', 'exchangeOutflow', 'exchangeBalance']
        all_data = []

        for metric in metrics:
            if self._check_should_disable():
                break
            df = await self.fetch_metric(slug, metric, start_date, end_date, interval)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        result = all_data[0]
        for df in all_data[1:]:
            metric_name = df['metric'].iloc[0]
            result = result.merge(
                df[['timestamp', 'value']].rename(columns={'value': metric_name}),
                on='timestamp', how='outer'
            )
        
        # Add flow classifications
        if 'exchangeInflow' in result.columns and 'exchangeOutflow' in result.columns:
            result['net_flow'] = result['exchangeInflow'] - result['exchangeOutflow']
            result['flow_direction'] = result.apply(
                lambda r: FlowDirection.STRONG_INFLOW.value if r['net_flow'] > r['exchangeInflow'] * 0.3 else
                          FlowDirection.INFLOW.value if r['net_flow'] > 0 else
                          FlowDirection.STRONG_OUTFLOW.value if r['net_flow'] < -r['exchangeOutflow'] * 0.3 else
                          FlowDirection.OUTFLOW.value if r['net_flow'] < 0 else FlowDirection.NEUTRAL.value,
                axis=1
            )
        
        return result.sort_values('timestamp')
    
    async def fetch_trending_words(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None, size: int = 10
    ) -> pd.DataFrame:
        """Fetch trending words in crypto social media."""
        if not start_date:
            start_date = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')
        
        query = f'''
        {{
            getTrendingWords(
                from: "{start_date}T00:00:00Z"
                to: "{end_date}T23:59:59Z"
                size: {size}
            ) {{
                datetime
                topWords {{
                    word
                    score
                }}
            }}
        }}
        '''
        
        data = await self._execute_query(query)
        trending = data.get('getTrendingWords', [])
        
        if not trending:
            return pd.DataFrame()
        
        records = []
        for period in trending:
            dt = period['datetime']
            for i, word_data in enumerate(period.get('topWords', [])):
                records.append({
                    'timestamp': pd.to_datetime(dt), 'rank': i + 1,
                    'word': word_data['word'],
                    'score': float(word_data['score']) if word_data['score'] else 0,
                    'is_top_trend': i < 3,
                    'trend_strength': 'viral' if float(word_data['score'] or 0) > 1000 else
                                     'strong' if float(word_data['score'] or 0) > 500 else
                                     'moderate' if float(word_data['score'] or 0) > 100 else 'weak',
                    'venue': self.VENUE,
                })
        
        self.collection_stats['records'] += len(records)
        return pd.DataFrame(records)
    
    async def fetch_comprehensive_metrics(
        self, symbol: str, start_date: str, end_date: str, interval: str = '1d'
    ) -> pd.DataFrame:
        """Fetch comprehensive metrics for an asset."""
        slug = self._get_slug(symbol)
        
        all_metrics = [
            'price_usd', 'volume_usd', 'marketcap_usd',
            'dailyActiveAddresses', 'transactionVolume',
            'exchangeInflow', 'exchangeOutflow',
            'socialVolumeTotal', 'sentimentBalance',
            'devActivity', 'nvtRatio', 'mvrvRatio'
        ]
        
        all_data = []
        for metric in all_metrics:
            df = await self.fetch_metric(slug, metric, start_date, end_date, interval)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        result = all_data[0]
        for df in all_data[1:]:
            metric_name = df['metric'].iloc[0]
            result = result.merge(
                df[['timestamp', 'value']].rename(columns={'value': metric_name}),
                on='timestamp', how='outer'
            )
        
        result['symbol'] = symbol
        return result.sort_values('timestamp')
    
    async def _fetch_single_sentiment_signal(
        self, symbol: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch sentiment signal for a single symbol."""
        try:
            slug = self._get_slug(symbol)
            df = await self.fetch_social_metrics(slug, start_date, end_date)
            if not df.empty:
                df['symbol'] = symbol
                return df
        except Exception as e:
            self.logger.error(f"Error fetching sentiment for {symbol}: {e}")
        return None

    async def fetch_sentiment_signals(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch sentiment signals for multiple assets."""
        # CRITICAL: Check if disabled - skip immediately
        if self._check_should_disable():
            return pd.DataFrame()

        # Parallel processing - will auto-disable on first rate limit
        tasks = [
            self._fetch_single_sentiment_signal(symbol, start_date, end_date)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    async def fetch_funding_rates(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Santiment doesn't provide funding rates."""
        return pd.DataFrame()
    
    async def _fetch_single_ohlcv(
        self, symbol: str, timeframe: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV for a single symbol."""
        try:
            slug = self._get_slug(symbol)
            df = await self.fetch_metric(slug, 'price_usd', start_date, end_date, timeframe)
            if not df.empty:
                df['symbol'] = symbol
                df = df.rename(columns={'value': 'close'})
                return df
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV for {symbol}: {e}")
        return None

    async def fetch_ohlcv(
        self, symbols: List[str], timeframe: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch price data from Santiment."""
        # CRITICAL: Check if disabled - skip immediately
        if self._check_should_disable():
            return pd.DataFrame()

        # Parallel processing - will auto-disable on first rate limit
        tasks = [
            self._fetch_single_ohlcv(symbol, timeframe, start_date, end_date)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    async def _collect_single_onchain_metric(
        self, symbol: str, start_str: str, end_str: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """Collect on-chain metrics for a single symbol."""
        try:
            slug = self._get_slug(symbol)
            df = await self.fetch_onchain_metrics(
                slug=slug,
                start_date=start_str,
                end_date=end_str,
                interval=interval
            )
            if not df.empty:
                df['symbol'] = symbol.upper()
                df['venue'] = self.VENUE
                df['venue_type'] = self.VENUE_TYPE
                return df
        except Exception as e:
            self.logger.error(f"Error collecting on-chain for {symbol}: {e}")
        return None

    async def collect_on_chain_metrics(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect on-chain metrics - wraps fetch_onchain_metrics().

        Standardized method name for collection manager compatibility.
        Skips immediately if rate limited (likely monthly quota exhausted).
        """
        # CRITICAL: Check if disabled BEFORE doing any work - skip immediately
        if self._check_should_disable():
            self.logger.info(f"Santiment skipped: {self._disable_reason}")
            return pd.DataFrame()

        try:
            # Convert dates to string format if needed
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            interval = kwargs.get('interval', '1d')

            # Parallel processing - will auto-disable on first rate limit
            tasks = [
                self._collect_single_onchain_metric(symbol, start_str, end_str, interval)
                for symbol in symbols
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_records = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]
            if all_records:
                return pd.concat(all_records, ignore_index=True)
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Santiment collect_on_chain_metrics error: {e}")
            return pd.DataFrame()

    async def _collect_single_social_metric(
        self, symbol: str, start_str: str, end_str: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """Collect social metrics for a single symbol."""
        try:
            slug = self._get_slug(symbol)
            df = await self.fetch_social_metrics(
                slug=slug,
                start_date=start_str,
                end_date=end_str,
                interval=interval
            )
            if not df.empty:
                df['symbol'] = symbol.upper()
                df['venue'] = self.VENUE
                df['venue_type'] = self.VENUE_TYPE
                return df
        except Exception as e:
            self.logger.error(f"Error collecting social for {symbol}: {e}")
        return None

    async def collect_social(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect social metrics - wraps fetch_social_metrics().

        Standardized method name for collection manager compatibility.
        Skips immediately if rate limited (likely monthly quota exhausted).
        """
        # CRITICAL: Check if disabled BEFORE doing any work - skip immediately
        if self._check_should_disable():
            self.logger.info(f"Santiment skipped: {self._disable_reason}")
            return pd.DataFrame()

        try:
            # Convert dates to string format if needed
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            interval = kwargs.get('interval', '1d')

            # Parallel processing - will auto-disable on first rate limit
            tasks = [
                self._collect_single_social_metric(symbol, start_str, end_str, interval)
                for symbol in symbols
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_records = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]
            if all_records:
                return pd.concat(all_records, ignore_index=True)
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Santiment collect_social error: {e}")
            return pd.DataFrame()

    async def collect_funding_rates(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect funding rates - Santiment doesn't provide these.

        Standardized method name for collection manager compatibility.
        Returns empty DataFrame immediately (no API calls needed).
        """
        # Santiment doesn't provide funding rates - return immediately
        return pd.DataFrame()

    async def collect_ohlcv(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect OHLCV data - wraps fetch_ohlcv().

        Standardized method name for collection manager compatibility.
        """
        # CRITICAL: Check if disabled before doing any work
        if self._check_should_disable():
            self.logger.info(f"Santiment skipped (disabled): {self._disable_reason}")
            return pd.DataFrame()

        try:
            timeframe = kwargs.get('timeframe', '1h')

            # Convert dates to string format if needed
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            return await self.fetch_ohlcv(
                symbols=symbols,
                timeframe=timeframe,
                start_date=start_str,
                end_date=end_str
            )
        except Exception as e:
            self.logger.error(f"Santiment collect_ohlcv error: {e}")
            return pd.DataFrame()

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            **self.collection_stats,
            'venue': self.VENUE,
            'disabled': self._disabled,
            'disable_reason': self._disable_reason,
        }
    
    @staticmethod
    def get_supported_symbols() -> List[str]:
        """Get list of supported symbols."""
        return list(ASSET_SLUGS.keys())

    def reset_rate_limit_state(self) -> None:
        """
        Reset rate limit state to allow retrying.
        Use this to manually reset after waiting for quota to refresh.
        """
        self._disabled = False
        self._disable_reason = ""
        self._consecutive_rate_limits = 0
        self._logged_disabled_warning = False
        self.logger.info("Santiment rate limit state reset - ready to retry")
    
    async def close(self) -> None:
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

async def test_santiment():
    """Test Santiment collector."""
    config = {'api_key': ''}
    collector = SantimentCollector(config)
    try:
        print(f"Supported symbols: {collector.get_supported_symbols()[:10]}")
        
        # Test dataclasses
        social = SocialData(
            timestamp=datetime.utcnow(), slug='bitcoin',
            social_volume=5000, sentiment_balance=0.25
        )
        print(f"Sentiment level: {social.sentiment_level.value}, is_bullish: {social.is_bullish}")
        
        dev = DevelopmentData(
            timestamp=datetime.utcnow(), slug='ethereum',
            dev_activity=75, contributors_count=50
        )
        print(f"Dev health: {dev.health.value}, quality_score: {dev.quality_score:.1f}")
    finally:
        await collector.close()

if __name__ == '__main__':
    asyncio.run(test_santiment())