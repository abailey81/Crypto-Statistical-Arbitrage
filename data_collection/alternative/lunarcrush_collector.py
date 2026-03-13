"""
LunarCrush API Collector for Social and Alternative Data.

validated collector providing social intelligence and alternative data:
- Social volume and engagement metrics across platforms
- Galaxy Score (proprietary social ranking)
- AltRank (alternative ranking metric)
- Influencer tracking and impact analysis
- Social sentiment analysis with momentum
- Market correlations with social data
- Social momentum indicators for trading signals

API Documentation: https://lunarcrush.com/developers/api/endpoints
Rate Limits: Vary by plan (free tier: 100 requests/day, Pro: 1000/day)
Registration: https://lunarcrush.com

Version: 2.0.0
"""

import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import logging
import json

from ..base_collector import BaseCollector
from ..utils.rate_limiter import get_shared_rate_limiter
from ..utils.retry_handler import RetryHandler

logger = logging.getLogger(__name__)

class SocialPlatform(Enum):
    """Supported social platforms."""
    TWITTER = 'twitter'
    REDDIT = 'reddit'
    YOUTUBE = 'youtube'
    TIKTOK = 'tiktok'
    DISCORD = 'discord'
    TELEGRAM = 'telegram'
    MEDIUM = 'medium'
    NEWS = 'news'

class SentimentSignal(Enum):
    """Sentiment-based trading signals."""
    STRONG_BULLISH = 'strong_bullish'
    BULLISH = 'bullish'
    NEUTRAL = 'neutral'
    BEARISH = 'bearish'
    STRONG_BEARISH = 'strong_bearish'

@dataclass
class SocialMomentum:
    """Social momentum indicators."""
    symbol: str
    galaxy_score: float
    galaxy_score_change_24h: float
    social_volume: float
    social_volume_change_24h: float
    sentiment: float
    sentiment_change_24h: float
    engagement_rate: float
    momentum_score: float
    signal: SentimentSignal
    timestamp: datetime

class LunarCrushCollector(BaseCollector):
    """
    LunarCrush data collector for social intelligence.
    
    Features:
    - Social volume and engagement across platforms
    - Galaxy Score (proprietary composite metric 0-100)
    - AltRank (alternative ranking based on social + market data)
    - Influencer analysis and impact tracking
    - Social sentiment scoring (-5 to +5)
    - News and content metrics
    - Social momentum indicators
    - Correlation analysis with price
    
    Use Cases:
    - Sentiment-based trading signals
    - Social momentum detection
    - Influencer impact analysis
    - Alternative data for alpha generation
    - Contrarian indicators
    
    Galaxy Score Components:
    - Social engagement (likes, shares, comments)
    - Social volume (total posts/mentions)
    - Price correlation with social activity
    - Market cap / volume weighting
    - Spam filtering
    """
    
    VENUE = 'lunarcrush'
    VENUE_TYPE = 'alternative'
    BASE_URL = 'https://lunarcrush.com/api4/public'
    
    # Available metrics and descriptions
    METRICS = {
        'galaxy_score': 'Proprietary composite social score (0-100)',
        'alt_rank': 'Alternative ranking based on social activity',
        'social_volume': 'Total social posts mentioning asset',
        'social_engagement': 'Total engagement (likes, shares, comments)',
        'social_contributors': 'Unique accounts posting about asset',
        'social_dominance': 'Percentage of total crypto social volume',
        'market_dominance': 'Market cap dominance percentage',
        'sentiment': 'Average sentiment score (-5 to 5)',
        'spam': 'Spam score (0-100)',
        'news': 'News articles count',
        'url_shares': 'URL shares count',
        'bullish_sentiment': 'Percentage of bullish posts',
        'bearish_sentiment': 'Percentage of bearish posts',
    }
    
    # Supported social platforms
    PLATFORMS = [p.value for p in SocialPlatform]
    
    # Major coins for baseline comparison
    MAJOR_COINS = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE', 'AVAX', 'DOT', 'MATIC']
    
    # Galaxy score thresholds
    GALAXY_THRESHOLDS = {
        'exceptional': 80,
        'strong': 60,
        'moderate': 40,
        'weak': 20,
    }
    
    def __init__(self, config: Dict):
        """
        Initialize LunarCrush collector.

        Args:
            config: Configuration with:
                - api_key: LunarCrush API key
                - rate_limit: Requests per minute (default 10 for free tier)
                - cache_ttl: Cache TTL in seconds (default 300)
        """
        super().__init__(config)

        # CRITICAL: Set supported data types for dynamic routing
        self.supported_data_types = ['social', 'sentiment']
        self.venue = 'lunarcrush'

        # Import VenueType from base_collector
        from ..base_collector import VenueType
        self.venue_type = VenueType.ALTERNATIVE
        self.requires_auth = True # Requires LunarCrush API key

        self.api_key = config.get('api_key') or config.get('lunarcrush_api_key') or os.getenv('LUNARCRUSH_API_KEY', '')
        self.session = None

        # Rate limiting (conservative for free tier)
        rate_limit = config.get('rate_limit', 5)
        self.rate_limiter = get_shared_rate_limiter('lunarcrush', rate=rate_limit, per=60.0, burst=config.get('burst', 3))

        if self.api_key:
            logger.info(f"Initialized LunarCrush collector with data types: {self.supported_data_types}")
        else:
            logger.warning("LunarCrush API key not provided")

        # Retry handler
        self.retry_handler = RetryHandler(
            max_retries=3,
            base_delay=2.0,
            max_delay=30.0
        )
        
        # Cache for expensive queries
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl = config.get('cache_ttl', 300)
        
        # Collection stats
        self.collection_stats = {
            'records_collected': 0,
            'api_calls': 0,
            'cache_hits': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
        
        if not self.api_key:
            logger.warning("No LunarCrush API key provided - limited functionality")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Accept': 'application/json',
                'User-Agent': 'CryptoStatArb/2.0'
            }
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self.session
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self._cache:
            timestamp, value = self._cache[key]
            if (datetime.utcnow() - timestamp).total_seconds() < self._cache_ttl:
                self.collection_stats['cache_hits'] += 1
                return value
            del self._cache[key]
        return None
    
    def _set_cached(self, key: str, value: Any):
        """Set value in cache."""
        self._cache[key] = (datetime.utcnow(), value)
    
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """
        Make API request with rate limiting, caching, and error handling.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            use_cache: Whether to use caching
            
        Returns:
            Response data or None on error
        """
        # Check cache
        cache_key = f"{endpoint}_{json.dumps(params or {}, sort_keys=True)}"
        if use_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached
        
        await self.rate_limiter.acquire()
        session = await self._get_session()
        
        url = f"{self.BASE_URL}/{endpoint}"
        self.collection_stats['api_calls'] += 1
        
        async def _request():
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    # Handle both wrapped and unwrapped responses
                    if isinstance(data, dict) and 'data' in data:
                        return data['data']
                    return data
                elif response.status == 401:
                    logger.warning("LunarCrush authentication failed - API key may be invalid or expired")
                    return None
                elif response.status == 429:
                    logger.warning("LunarCrush rate limit hit - backing off")
                    raise aiohttp.ClientError("Rate limit exceeded")
                elif response.status == 404:
                    logger.debug(f"LunarCrush endpoint not found: {endpoint}")
                    return None
                elif response.status == 403:
                    logger.warning("LunarCrush access forbidden - endpoint may require higher tier")
                    return None
                else:
                    text = await response.text()
                    logger.debug(f"LunarCrush API error {response.status}: {text[:200]}")
                    raise aiohttp.ClientError(f"API error {response.status}")
        
        try:
            result = await self.retry_handler.execute(_request)
            if result is not None and use_cache:
                self._set_cached(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"LunarCrush request failed: {endpoint} - {e}")
            self.collection_stats['errors'] += 1
            return None
    
    # =========================================================================
    # Asset Data Methods
    # =========================================================================
    
    async def get_coins(
        self,
        sort: str = 'galaxy_score',
        limit: int = 100,
        desc: bool = True,
        min_volume: float = 0,
        min_market_cap: float = 0
    ) -> pd.DataFrame:
        """
        Get list of tracked coins with social metrics.
        
        Args:
            sort: Sort field (galaxy_score, alt_rank, market_cap, social_volume)
            limit: Number of coins to return (max 500)
            desc: Sort descending if True
            min_volume: Minimum 24h volume filter
            min_market_cap: Minimum market cap filter
            
        Returns:
            DataFrame with coin data and social metrics
        """
        params = {
            'sort': sort,
            'limit': min(limit, 500),
            'desc': str(desc).lower()
        }
        
        data = await self._make_request('coins/list', params)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        for coin in data:
            volume = float(coin.get('volume_24h', 0) or 0)
            mcap = float(coin.get('market_cap', 0) or 0)
            
            # Apply filters
            if volume < min_volume or mcap < min_market_cap:
                continue
            
            # Calculate derived metrics
            galaxy_score = float(coin.get('galaxy_score', 0) or 0)
            sentiment = float(coin.get('average_sentiment', 0) or 0)
            social_volume = float(coin.get('social_volume', 0) or 0)
            
            # Sentiment classification
            if sentiment >= 3:
                sentiment_class = 'very_bullish'
            elif sentiment >= 1:
                sentiment_class = 'bullish'
            elif sentiment >= -1:
                sentiment_class = 'neutral'
            elif sentiment >= -3:
                sentiment_class = 'bearish'
            else:
                sentiment_class = 'very_bearish'
            
            # Galaxy score tier
            if galaxy_score >= self.GALAXY_THRESHOLDS['exceptional']:
                galaxy_tier = 'exceptional'
            elif galaxy_score >= self.GALAXY_THRESHOLDS['strong']:
                galaxy_tier = 'strong'
            elif galaxy_score >= self.GALAXY_THRESHOLDS['moderate']:
                galaxy_tier = 'moderate'
            else:
                galaxy_tier = 'weak'
            
            records.append({
                'symbol': coin.get('symbol', '').upper(),
                'name': coin.get('name'),
                'price': float(coin.get('price', 0) or 0),
                'market_cap': mcap,
                'volume_24h': volume,
                'price_change_24h': float(coin.get('percent_change_24h', 0) or 0),
                'price_change_7d': float(coin.get('percent_change_7d', 0) or 0),
                'galaxy_score': galaxy_score,
                'galaxy_tier': galaxy_tier,
                'alt_rank': int(coin.get('alt_rank', 0) or 0),
                'social_volume': social_volume,
                'social_volume_24h_change': float(coin.get('social_volume_change_24h', 0) or 0),
                'social_engagement': float(coin.get('social_engagement', 0) or 0),
                'social_contributors': int(coin.get('social_contributors', 0) or 0),
                'social_dominance': float(coin.get('social_dominance', 0) or 0),
                'sentiment': sentiment,
                'sentiment_class': sentiment_class,
                'bullish_pct': float(coin.get('bullish_sentiment', 0) or 0),
                'bearish_pct': float(coin.get('bearish_sentiment', 0) or 0),
                'news_count': int(coin.get('news', 0) or 0),
                'url_shares': int(coin.get('url_shares', 0) or 0),
                'spam_score': float(coin.get('spam', 0) or 0),
                'timestamp': datetime.utcnow(),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE
            })
        
        df = pd.DataFrame(records)
        self.collection_stats['records_collected'] += len(df)
        return df
    
    async def get_coin_details(self, symbol: str) -> Optional[Dict]:
        """
        Get detailed data for a specific coin.
        
        Args:
            symbol: Coin symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Dictionary with detailed coin data
        """
        data = await self._make_request(f'coins/{symbol.lower()}')
        return data
    
    async def get_coin_time_series(
        self,
        symbol: str,
        interval: str = '1d',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days_back: int = 30
    ) -> pd.DataFrame:
        """
        Get historical time series data for a coin.
        
        Args:
            symbol: Coin symbol
            interval: Time interval ('1h', '1d', '1w')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            days_back: Days of history if no dates provided
            
        Returns:
            DataFrame with historical social metrics
        """
        params = {'interval': interval}
        
        if start_date:
            params['start'] = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        else:
            params['start'] = int((datetime.utcnow() - timedelta(days=days_back)).timestamp())
        
        if end_date:
            params['end'] = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        else:
            params['end'] = int(datetime.utcnow().timestamp())
        
        data = await self._make_request(f'coins/{symbol.lower()}/time-series', params, use_cache=False)

        # Handle different API response formats (v3 vs v4)
        # v3: data = { 'timeSeries': [...] }
        # v4: data = { 'data': [...] } or just [...]
        time_series_data = None
        if data:
            if isinstance(data, list):
                time_series_data = data
            elif 'timeSeries' in data:
                time_series_data = data['timeSeries']
            elif 'data' in data:
                time_series_data = data['data']
            elif 'time_series' in data:
                time_series_data = data['time_series']

        if not time_series_data:
            logger.debug(f"LunarCrush: No time series data for {symbol}. Response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            return pd.DataFrame()

        records = []
        for point in time_series_data:
            galaxy_score = float(point.get('galaxy_score', 0) or 0)
            sentiment = float(point.get('average_sentiment', 0) or 0)
            
            records.append({
                'timestamp': pd.to_datetime(point.get('time'), unit='s', utc=True),
                'symbol': symbol.upper(),
                'open': float(point.get('open', 0) or 0),
                'high': float(point.get('high', 0) or 0),
                'low': float(point.get('low', 0) or 0),
                'close': float(point.get('close', 0) or 0),
                'volume': float(point.get('volume', 0) or 0),
                'market_cap': float(point.get('market_cap', 0) or 0),
                'galaxy_score': galaxy_score,
                'alt_rank': int(point.get('alt_rank', 0) or 0),
                'social_volume': float(point.get('social_volume', 0) or 0),
                'social_engagement': float(point.get('social_engagement', 0) or 0),
                'social_contributors': int(point.get('social_contributors', 0) or 0),
                'social_dominance': float(point.get('social_dominance', 0) or 0),
                'sentiment': sentiment,
                'news_count': int(point.get('news', 0) or 0),
                'url_shares': int(point.get('url_shares', 0) or 0),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE
            })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
            self.collection_stats['records_collected'] += len(df)
        return df
    
    # =========================================================================
    # Social Metrics Methods
    # =========================================================================
    
    async def get_social_mentions(
        self,
        symbol: str,
        hours: int = 24,
        platforms: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get recent social mentions for a coin.
        
        Args:
            symbol: Coin symbol
            hours: Hours of history to fetch
            platforms: Filter by platforms (twitter, reddit, etc.)
            
        Returns:
            DataFrame with social mention data
        """
        params = {'hours': hours}
        data = await self._make_request(f'coins/{symbol.lower()}/mentions', params, use_cache=False)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        for mention in data:
            platform = mention.get('type', '').lower()
            
            # Filter by platform if specified
            if platforms and platform not in [p.lower() for p in platforms]:
                continue
            
            sentiment_val = float(mention.get('sentiment', 0) or 0)
            
            records.append({
                'timestamp': pd.to_datetime(mention.get('time'), unit='s', utc=True),
                'symbol': symbol.upper(),
                'platform': platform,
                'title': mention.get('title', ''),
                'body': (mention.get('body', '') or '')[:500],
                'sentiment': sentiment_val,
                'sentiment_label': 'bullish' if sentiment_val > 0 else ('bearish' if sentiment_val < 0 else 'neutral'),
                'engagement': int(mention.get('interactions_total', 0) or 0),
                'likes': int(mention.get('likes', 0) or 0),
                'comments': int(mention.get('comments', 0) or 0),
                'shares': int(mention.get('shares', 0) or 0),
                'followers': int(mention.get('account_followers', 0) or 0),
                'url': mention.get('url', ''),
                'is_influencer': int(mention.get('account_followers', 0) or 0) > 10000,
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE
            })
        
        df = pd.DataFrame(records)
        self.collection_stats['records_collected'] += len(df)
        return df
    
    async def get_influencers(
        self,
        symbol: Optional[str] = None,
        limit: int = 50,
        min_followers: int = 0
    ) -> pd.DataFrame:
        """
        Get top influencers, optionally filtered by coin.
        
        Args:
            symbol: Optional coin symbol to filter by
            limit: Number of influencers to return
            min_followers: Minimum follower count filter
            
        Returns:
            DataFrame with influencer data
        """
        endpoint = f'coins/{symbol.lower()}/influencers' if symbol else 'influencers'
        params = {'limit': limit}
        
        data = await self._make_request(endpoint, params)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        for influencer in data:
            followers = int(influencer.get('followers', 0) or 0)
            
            if followers < min_followers:
                continue
            
            engagement = float(influencer.get('engagement', 0) or 0)
            posts = int(influencer.get('posts', 0) or 0)
            
            # Calculate engagement rate
            engagement_rate = (engagement / followers * 100) if followers > 0 else 0
            
            # Influence tier
            if followers >= 1000000:
                tier = 'mega'
            elif followers >= 100000:
                tier = 'macro'
            elif followers >= 10000:
                tier = 'micro'
            else:
                tier = 'nano'
            
            records.append({
                'username': influencer.get('twitter_screen_name') or influencer.get('display_name'),
                'display_name': influencer.get('display_name', ''),
                'platform': influencer.get('network', 'twitter'),
                'followers': followers,
                'engagement_total': engagement,
                'engagement_rate': engagement_rate,
                'influence_score': float(influencer.get('influence_score', 0) or 0),
                'posts_count': posts,
                'avg_engagement_per_post': engagement / posts if posts > 0 else 0,
                'average_sentiment': float(influencer.get('average_sentiment', 0) or 0),
                'tier': tier,
                'symbol': symbol.upper() if symbol else None,
                'timestamp': datetime.utcnow(),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE
            })
        
        df = pd.DataFrame(records)
        self.collection_stats['records_collected'] += len(df)
        return df
    
    async def get_trending(
        self,
        limit: int = 20,
        include_metrics: bool = True
    ) -> pd.DataFrame:
        """
        Get trending coins by social activity.
        
        Args:
            limit: Number of trending coins
            include_metrics: Include detailed metrics
            
        Returns:
            DataFrame with trending coin data
        """
        params = {'limit': limit}
        data = await self._make_request('coins/trending', params)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        for i, coin in enumerate(data, 1):
            galaxy_score = float(coin.get('galaxy_score', 0) or 0)
            galaxy_change = float(coin.get('galaxy_score_change', 0) or 0)
            social_volume = float(coin.get('social_volume', 0) or 0)
            social_change = float(coin.get('social_volume_change_24h', 0) or 0)
            price_change = float(coin.get('percent_change_24h', 0) or 0)
            
            # Calculate trend strength
            trend_strength = 0
            if galaxy_change > 10:
                trend_strength += 2
            elif galaxy_change > 5:
                trend_strength += 1
            if social_change > 50:
                trend_strength += 2
            elif social_change > 20:
                trend_strength += 1
            
            records.append({
                'rank': i,
                'symbol': coin.get('symbol', '').upper(),
                'name': coin.get('name'),
                'price': float(coin.get('price', 0) or 0),
                'price_change_24h': price_change,
                'market_cap': float(coin.get('market_cap', 0) or 0),
                'volume_24h': float(coin.get('volume_24h', 0) or 0),
                'galaxy_score': galaxy_score,
                'galaxy_score_change': galaxy_change,
                'social_volume': social_volume,
                'social_volume_change': social_change,
                'sentiment': float(coin.get('average_sentiment', 0) or 0),
                'trend_strength': trend_strength,
                'price_social_correlation': 1 if (price_change > 0 and social_change > 0) or (price_change < 0 and social_change < 0) else -1 if price_change * social_change < 0 else 0,
                'timestamp': datetime.utcnow(),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE
            })
        
        df = pd.DataFrame(records)
        self.collection_stats['records_collected'] += len(df)
        return df
    
    async def get_market_pairs(
        self,
        symbol: str,
        limit: int = 20
    ) -> pd.DataFrame:
        """
        Get trading pairs and exchanges for a coin.
        
        Args:
            symbol: Coin symbol
            limit: Number of pairs to return
            
        Returns:
            DataFrame with market pair data
        """
        params = {'limit': limit}
        data = await self._make_request(f'coins/{symbol.lower()}/markets', params)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        for pair in data:
            volume = float(pair.get('volume_24h', 0) or 0)
            
            records.append({
                'symbol': symbol.upper(),
                'pair': pair.get('pair'),
                'exchange': pair.get('market'),
                'price': float(pair.get('price', 0) or 0),
                'volume_24h': volume,
                'spread': float(pair.get('spread', 0) or 0),
                'trust_score': float(pair.get('trust_score', 0) or 0),
                'volume_share': volume,
                'timestamp': datetime.utcnow(),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE
            })
        
        # Calculate volume share
        if records:
            total_volume = sum(r['volume_24h'] for r in records)
            if total_volume > 0:
                for r in records:
                    r['volume_share'] = r['volume_24h'] / total_volume * 100
        
        df = pd.DataFrame(records)
        self.collection_stats['records_collected'] += len(df)
        return df
    
    # =========================================================================
    # Global Market Methods
    # =========================================================================
    
    async def get_global_metrics(self) -> Dict:
        """
        Get global crypto market metrics.
        
        Returns:
            Dictionary with global market data
        """
        data = await self._make_request('market')
        
        if not data:
            return {}
        
        return {
            'total_market_cap': float(data.get('market_cap', 0) or 0),
            'total_volume_24h': float(data.get('volume_24h', 0) or 0),
            'btc_dominance': float(data.get('btc_dominance', 0) or 0),
            'eth_dominance': float(data.get('eth_dominance', 0) or 0),
            'total_social_volume': int(data.get('social_volume', 0) or 0),
            'total_social_engagement': int(data.get('social_engagement', 0) or 0),
            'average_sentiment': float(data.get('average_sentiment', 0) or 0),
            'fear_greed_index': int(data.get('fear_greed', 0) or 0),
            'active_coins': int(data.get('active_coins', 0) or 0),
            'timestamp': datetime.utcnow(),
            'venue': self.VENUE,
            'venue_type': self.VENUE_TYPE
        }
    
    async def get_categories(self) -> pd.DataFrame:
        """
        Get coin categories with aggregated metrics.
        
        Returns:
            DataFrame with category data
        """
        data = await self._make_request('categories')
        
        if not data:
            return pd.DataFrame()
        
        records = []
        for category in data:
            records.append({
                'category': category.get('name'),
                'coin_count': int(category.get('num_coins', 0) or 0),
                'market_cap': float(category.get('market_cap', 0) or 0),
                'volume_24h': float(category.get('volume_24h', 0) or 0),
                'market_cap_change_24h': float(category.get('market_cap_change_24h', 0) or 0),
                'social_volume': int(category.get('social_volume', 0) or 0),
                'average_galaxy_score': float(category.get('average_galaxy_score', 0) or 0),
                'timestamp': datetime.utcnow(),
                'venue': self.VENUE,
                'venue_type': self.VENUE_TYPE
            })
        
        df = pd.DataFrame(records)
        self.collection_stats['records_collected'] += len(df)
        return df
    
    # =========================================================================
    # High-Level Data Fetching Methods
    # =========================================================================
    
    async def fetch_social_metrics(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch comprehensive social metrics for multiple symbols.
        
        Args:
            symbols: List of coin symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Time interval
            
        Returns:
            DataFrame with social metrics time series
        """
        # PARALLELIZED: Fetch social metrics for all symbols concurrently
        symbols_with_data = 0
        symbols_without_data = 0

        async def _fetch_single_social_metric(symbol: str) -> Optional[pd.DataFrame]:
            nonlocal symbols_with_data, symbols_without_data

            try:
                df = await self.get_coin_time_series(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date
                )

                if not df.empty:
                    # Add momentum indicators
                    df = self._add_momentum_indicators(df)
                    symbols_with_data += 1
                    return df
                else:
                    symbols_without_data += 1

            except Exception as e:
                logger.debug(f"LunarCrush error fetching {symbol}: {e}")
                self.collection_stats['errors'] += 1

            return None

        tasks = [_fetch_single_social_metric(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_data = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

        # Log summary once (not per-symbol spam)
        if symbols_with_data > 0 or symbols_without_data > 0:
            logger.info(
                f"LunarCrush: {symbols_with_data}/{len(symbols)} symbols returned data"
                + (f" ({symbols_without_data} empty)" if symbols_without_data > 0 else "")
            )

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result = result.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
            return result

        return pd.DataFrame()
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators to time series data."""
        if df.empty or len(df) < 2:
            return df
        
        # Galaxy score momentum
        df['galaxy_score_ma7'] = df['galaxy_score'].rolling(window=7, min_periods=1).mean()
        df['galaxy_score_momentum'] = df['galaxy_score'] - df['galaxy_score_ma7']
        
        # Social volume momentum
        df['social_volume_ma7'] = df['social_volume'].rolling(window=7, min_periods=1).mean()
        df['social_volume_change'] = df['social_volume'].pct_change() * 100
        
        # Sentiment momentum
        df['sentiment_ma7'] = df['sentiment'].rolling(window=7, min_periods=1).mean()
        df['sentiment_change'] = df['sentiment'].diff()
        
        # Engagement rate (if price available)
        if 'close' in df.columns and 'volume' in df.columns:
            df['price_change'] = df['close'].pct_change() * 100
            
            # Price-social correlation (rolling)
            df['price_social_corr'] = df['price_change'].rolling(window=7, min_periods=3).corr(df['social_volume_change'])
        
        return df
    
    async def fetch_sentiment_signals(
        self,
        symbols: List[str]
    ) -> pd.DataFrame:
        """
        Fetch current sentiment signals for trading.
        
        Args:
            symbols: List of coin symbols
            
        Returns:
            DataFrame with sentiment-based trading signals
        """
        # PARALLELIZED: Generate signals for all symbols concurrently
        async def _generate_single_trading_signal(symbol: str) -> Optional[Dict]:
            try:
                details = await self.get_coin_details(symbol)

                if not details:
                    return None

                # Extract metrics
                galaxy_score = float(details.get('galaxy_score', 0) or 0)
                galaxy_change = float(details.get('galaxy_score_change_24h', 0) or 0)
                sentiment = float(details.get('average_sentiment', 0) or 0)
                social_volume = int(details.get('social_volume', 0) or 0)
                social_volume_change = float(details.get('social_volume_change_24h', 0) or 0)
                bullish_pct = float(details.get('bullish_sentiment', 0) or 0)
                bearish_pct = float(details.get('bearish_sentiment', 0) or 0)

                # Calculate signal components
                bullish_signals = 0
                bearish_signals = 0

                # Galaxy score signal
                if galaxy_score > 70:
                    bullish_signals += 1
                elif galaxy_score < 30:
                    bearish_signals += 1

                # Galaxy momentum signal
                if galaxy_change > 5:
                    bullish_signals += 1
                elif galaxy_change < -5:
                    bearish_signals += 1
                
                # Sentiment signal
                if sentiment > 2:
                    bullish_signals += 1
                elif sentiment < -2:
                    bearish_signals += 1

                # Social volume momentum
                if social_volume_change > 50:
                    bullish_signals += 1
                elif social_volume_change < -30:
                    bearish_signals += 1

                # Bullish/bearish ratio
                if bullish_pct > 60:
                    bullish_signals += 1
                elif bearish_pct > 60:
                    bearish_signals += 1

                # Determine overall signal
                net_signal = bullish_signals - bearish_signals
                if net_signal >= 3:
                    signal = SentimentSignal.STRONG_BULLISH
                elif net_signal >= 1:
                    signal = SentimentSignal.BULLISH
                elif net_signal <= -3:
                    signal = SentimentSignal.STRONG_BEARISH
                elif net_signal <= -1:
                    signal = SentimentSignal.BEARISH
                else:
                    signal = SentimentSignal.NEUTRAL

                # Confidence score (0-100)
                confidence = min(100, (abs(net_signal) / 5) * 100)

                return {
                    'timestamp': datetime.utcnow(),
                    'symbol': symbol.upper(),
                    'galaxy_score': galaxy_score,
                    'galaxy_score_change_24h': galaxy_change,
                    'sentiment': sentiment,
                    'social_volume': social_volume,
                    'social_volume_change_24h': social_volume_change,
                    'bullish_pct': bullish_pct,
                    'bearish_pct': bearish_pct,
                    'bullish_signals': bullish_signals,
                    'bearish_signals': bearish_signals,
                    'net_signal': net_signal,
                    'signal': signal.value,
                    'confidence': confidence,
                    'price': float(details.get('price', 0) or 0),
                    'price_change_24h': float(details.get('percent_change_24h', 0) or 0),
                    'venue': self.VENUE,
                    'venue_type': self.VENUE_TYPE
                }

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                self.collection_stats['errors'] += 1

            return None

        tasks = [_generate_single_trading_signal(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_data = [r for r in results if isinstance(r, dict)]

        df = pd.DataFrame(all_data)
        self.collection_stats['records_collected'] += len(df)
        return df
    
    async def fetch_social_momentum(
        self,
        symbols: List[str]
    ) -> List[SocialMomentum]:
        """
        Calculate social momentum indicators for symbols.
        
        Args:
            symbols: List of coin symbols
            
        Returns:
            List of SocialMomentum dataclass instances
        """
        results = []
        signals_df = await self.fetch_sentiment_signals(symbols)
        
        if signals_df.empty:
            return results
        
        for _, row in signals_df.iterrows():
            # Calculate composite momentum score
            galaxy_momentum = row['galaxy_score_change_24h'] / 10 # Normalize
            social_momentum = min(1, row['social_volume_change_24h'] / 100) # Cap at 100%
            sentiment_normalized = row['sentiment'] / 5 # Normalize -5 to 5 -> -1 to 1
            
            momentum_score = (
                galaxy_momentum * 0.4 +
                social_momentum * 0.35 +
                sentiment_normalized * 0.25
            ) * 100
            
            # Determine signal
            if momentum_score >= 50:
                signal = SentimentSignal.STRONG_BULLISH
            elif momentum_score >= 20:
                signal = SentimentSignal.BULLISH
            elif momentum_score <= -50:
                signal = SentimentSignal.STRONG_BEARISH
            elif momentum_score <= -20:
                signal = SentimentSignal.BEARISH
            else:
                signal = SentimentSignal.NEUTRAL
            
            results.append(SocialMomentum(
                symbol=row['symbol'],
                galaxy_score=row['galaxy_score'],
                galaxy_score_change_24h=row['galaxy_score_change_24h'],
                social_volume=row['social_volume'],
                social_volume_change_24h=row['social_volume_change_24h'],
                sentiment=row['sentiment'],
                sentiment_change_24h=0, # Would need historical data
                engagement_rate=0, # Would need additional call
                momentum_score=momentum_score,
                signal=signal,
                timestamp=datetime.utcnow()
            ))
        
        return results
    
    async def fetch_trending_opportunities(
        self,
        min_galaxy_score: int = 60,
        min_social_volume_change: float = 20
    ) -> pd.DataFrame:
        """
        Fetch trending coins with potential trading opportunities.
        
        Args:
            min_galaxy_score: Minimum galaxy score filter
            min_social_volume_change: Minimum social volume change filter
            
        Returns:
            DataFrame with trending opportunities
        """
        trending = await self.get_trending(limit=50)
        
        if trending.empty:
            return pd.DataFrame()
        
        # Filter for opportunities
        opportunities = trending[
            (trending['galaxy_score'] >= min_galaxy_score) &
            (trending['social_volume_change'].fillna(0) >= min_social_volume_change)
        ].copy()
        
        if opportunities.empty:
            return pd.DataFrame()
        
        # Calculate opportunity score
        opportunities['opportunity_score'] = (
            opportunities['galaxy_score'] * 0.4 +
            opportunities['social_volume_change'].fillna(0) * 0.3 +
            (opportunities['sentiment'].fillna(0) + 5) * 4 * 0.2 +
            opportunities['trend_strength'] * 5 * 0.1
        )
        
        # Rank opportunities
        opportunities = opportunities.sort_values('opportunity_score', ascending=False)
        opportunities['opportunity_rank'] = range(1, len(opportunities) + 1)
        
        return opportunities
    
    async def fetch_batch_coins(
        self,
        symbols: List[str],
        include_time_series: bool = False,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """
        Fetch data for multiple coins in batch.
        
        Args:
            symbols: List of coin symbols
            include_time_series: Include historical time series
            days_back: Days of history for time series
            
        Returns:
            Dictionary with current data and optional time series
        """
        results = {
            'current': [],
            'time_series': {} if include_time_series else None
        }

        # PARALLELIZED: Fetch batch data for all symbols concurrently
        async def _fetch_single_batch_data(symbol: str) -> Dict:
            data = {'details': None, 'time_series': None}
            try:
                # Get current details
                details = await self.get_coin_details(symbol)
                if details:
                    data['details'] = {
                        'symbol': symbol.upper(),
                        **details
                    }

                # Get time series if requested
                if include_time_series:
                    ts = await self.get_coin_time_series(
                        symbol=symbol,
                        interval='1d',
                        days_back=days_back
                    )
                    if not ts.empty:
                        data['time_series'] = (symbol.upper(), ts)

            except Exception as e:
                logger.error(f"Error fetching batch data for {symbol}: {e}")

            return data

        tasks = [_fetch_single_batch_data(symbol) for symbol in symbols]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for batch_data in batch_results:
            if isinstance(batch_data, dict):
                if batch_data.get('details'):
                    results['current'].append(batch_data['details'])
                if batch_data.get('time_series'):
                    symbol_key, ts_data = batch_data['time_series']
                    results['time_series'][symbol_key] = ts_data

        results['current'] = pd.DataFrame(results['current'])
        return results
    
    # =========================================================================
    # Required Base Methods
    # =========================================================================
    
    async def fetch_funding_rates(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """LunarCrush doesn't provide funding rates - return empty."""
        logger.info("LunarCrush doesn't provide funding rates. Use Coinalyze or exchange collectors.")
        return pd.DataFrame()
    
    async def fetch_ohlcv(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data with social metrics overlay.
        
        Args:
            symbols: List of coin symbols
            timeframe: Time interval
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with OHLCV and social data
        """
        return await self.fetch_social_metrics(symbols, start_date, end_date, interval=timeframe)
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        return {
            **self.collection_stats,
            'cache_size': len(self._cache),
        }
    
    # =========================================================================
    # Standardized Collection Methods (for dynamic routing)
    # =========================================================================

    async def collect_social(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect social metrics for symbols (standardized interface).

        Wraps fetch_social_metrics() to match collection_manager expectations.

        NOTE: LunarCrush API v4 requires authentication. Without a valid API key,
        this method returns empty DataFrame immediately to avoid wasting time.
        """
        # CRITICAL: Skip if no API key - LunarCrush v4 requires authentication
        if not self.api_key:
            logger.info("LunarCrush: No API key configured - skipping social collection")
            return pd.DataFrame()

        try:
            interval = kwargs.get('interval', '1d')

            # Convert dates to strings if needed
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)

            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)

            logger.info(f"LunarCrush: Collecting social metrics for {len(symbols)} symbols")

            # Use existing fetch_social_metrics method
            df = await self.fetch_social_metrics(
                symbols=symbols,
                start_date=start_str,
                end_date=end_str,
                interval=interval
            )

            if not df.empty:
                logger.info(f"LunarCrush: Collected social data for {len(df)} records")
                return df

            logger.debug(f"LunarCrush: No social data found for symbols")
            return pd.DataFrame()

        except Exception as e:
            logger.debug(f"LunarCrush collect_social error: {e}")
            return pd.DataFrame()

    async def _fetch_alternative_me_fng(self, limit: int = 30) -> pd.DataFrame:
        """
        Fetch Fear & Greed Index from Alternative.me (FREE, no API key required).

        This is a fallback when LunarCrush API is not available.
        API docs: https://alternative.me/crypto/api/

        Args:
            limit: Number of historical data points (0 = all)

        Returns:
            DataFrame with Fear & Greed sentiment data
        """
        url = 'https://api.alternative.me/fng/'
        params = {'limit': limit, 'format': 'json'}

        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"Alternative.me API error: {resp.status}")
                    return pd.DataFrame()

                data = await resp.json()
                records = []

                for item in data.get('data', []):
                    # Convert timestamp to datetime
                    timestamp = pd.to_datetime(int(item['timestamp']), unit='s', utc=True)
                    value = int(item['value'])
                    classification = item['value_classification']

                    # Map to sentiment scores
                    records.append({
                        'timestamp': timestamp,
                        'symbol': 'BTC', # F&G is BTC-focused
                        'fear_greed_value': value,
                        'fear_greed_classification': classification,
                        'sentiment_score': value / 100.0, # Normalize to 0-1
                        'is_fear': value < 50,
                        'is_greed': value > 50,
                        'venue': 'alternative.me',
                        'venue_type': 'ALTERNATIVE',
                        'data_source': 'fear_greed_index'
                    })

                if records:
                    df = pd.DataFrame(records)
                    logger.info(f"Alternative.me: Fetched {len(df)} Fear & Greed records")
                    return df

                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Alternative.me fetch error: {e}")
            return pd.DataFrame()

    async def collect_sentiment(
        self,
        symbols: List[str],
        start_date: Any,
        end_date: Any,
        **kwargs
    ) -> pd.DataFrame:
        """
        Collect sentiment data for symbols (standardized interface).

        Falls back to Alternative.me Fear & Greed Index if LunarCrush unavailable.

        NOTE: LunarCrush API v4 requires paid subscription. This method
        immediately falls back to free Alternative.me API if no key is configured.
        """
        try:
            # If no API key, go straight to fallback (skip LunarCrush)
            if not self.api_key:
                logger.info("LunarCrush: No API key - using Alternative.me Fear & Greed fallback")
                df_fng = await self._fetch_alternative_me_fng(limit=30)
                if not df_fng.empty:
                    return df_fng
                return pd.DataFrame()

            logger.info(f"LunarCrush: Collecting sentiment for {len(symbols)} symbols")

            # Get social metrics (includes sentiment)
            df = await self.collect_social(symbols, start_date, end_date, **kwargs)

            if not df.empty:
                # Extract sentiment-related columns
                sentiment_cols = [
                    'symbol', 'timestamp', 'venue', 'venue_type',
                    'average_sentiment', 'bullish_sentiment', 'bearish_sentiment',
                    'sentiment_absolute', 'sentiment_relative'
                ]

                # Keep only sentiment columns that exist
                available_cols = [col for col in sentiment_cols if col in df.columns]

                if available_cols:
                    df_sentiment = df[available_cols].copy()
                    logger.info(f"LunarCrush: Collected sentiment for {len(df_sentiment)} records")
                    return df_sentiment

            # Fallback to Alternative.me Fear & Greed Index (completely free)
            logger.info("LunarCrush unavailable, using Alternative.me Fear & Greed fallback")
            df_fng = await self._fetch_alternative_me_fng(limit=30)
            if not df_fng.empty:
                return df_fng

            logger.debug("No sentiment data found")
            return pd.DataFrame()

        except Exception as e:
            logger.debug(f"LunarCrush collect_sentiment error: {e}")
            return pd.DataFrame()

    async def close(self):
        """Close aiohttp session and cleanup."""
        if self.session and not self.session.closed:
            await self.session.close()
        self._cache.clear()
        logger.info(f"LunarCrush collector closed. Stats: {self.get_collection_stats()}")

# =============================================================================
# Standalone Testing
# =============================================================================

async def test_collector():
    """Test LunarCrush collector functionality."""
    import os
    
    config = {
        'api_key': os.getenv('LUNARCRUSH_API_KEY', ''),
        'rate_limit': 10,
    }
    
    collector = LunarCrushCollector(config)
    
    try:
        print("Testing LunarCrush Collector")
        print("=" * 60)
        
        # Test get coins list
        print("\n1. Testing get_coins (top 10 by galaxy score)...")
        coins = await collector.get_coins(limit=10, min_market_cap=100_000_000)
        if not coins.empty:
            print(f" Found {len(coins)} coins")
            print(f" Top: {coins.iloc[0]['symbol']} - Galaxy: {coins.iloc[0]['galaxy_score']}, Tier: {coins.iloc[0]['galaxy_tier']}")
        
        # Test trending
        print("\n2. Testing get_trending...")
        trending = await collector.get_trending(limit=5)
        if not trending.empty:
            print(f" Found {len(trending)} trending coins")
            for _, row in trending.head(3).iterrows():
                print(f" #{row['rank']}: {row['symbol']} - Galaxy: {row['galaxy_score']}, Strength: {row['trend_strength']}")
        
        # Test coin time series
        print("\n3. Testing get_coin_time_series (BTC, 7 days)...")
        ts = await collector.get_coin_time_series(symbol='BTC', interval='1d', days_back=7)
        if not ts.empty:
            print(f" Found {len(ts)} data points")
            print(f" Latest galaxy score: {ts.iloc[-1]['galaxy_score']}")
        
        # Test sentiment signals
        print("\n4. Testing fetch_sentiment_signals...")
        signals = await collector.fetch_sentiment_signals(['BTC', 'ETH', 'SOL'])
        if not signals.empty:
            print(f" Generated signals for {len(signals)} coins")
            for _, row in signals.iterrows():
                print(f" {row['symbol']}: {row['signal']} (confidence: {row['confidence']:.0f}%)")
        
        # Test social momentum
        print("\n5. Testing fetch_social_momentum...")
        momentum = await collector.fetch_social_momentum(['BTC', 'ETH'])
        for m in momentum:
            print(f" {m.symbol}: momentum={m.momentum_score:.1f}, signal={m.signal.value}")
        
        # Test trending opportunities
        print("\n6. Testing fetch_trending_opportunities...")
        opps = await collector.fetch_trending_opportunities(min_galaxy_score=50, min_social_volume_change=10)
        if not opps.empty:
            print(f" Found {len(opps)} opportunities")
            for _, row in opps.head(3).iterrows():
                print(f" {row['symbol']}: score={row['opportunity_score']:.1f}")
        
        print("\n" + "=" * 60)
        print(f"Collection stats: {collector.get_collection_stats()}")
        print("LunarCrush collector tests completed!")
        
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await collector.close()

if __name__ == '__main__':
    asyncio.run(test_collector())