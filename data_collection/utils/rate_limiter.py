"""
Rate Limiter Module for Crypto Statistical Arbitrage Systems.

This module provides professional-quality rate limiting infrastructure
specifically designed for multi-venue crypto data collection with
trading-optimized configurations and adaptive throttling.

==============================================================================
RATE LIMITING FUNDAMENTALS
==============================================================================

Token Bucket Algorithm:
    The token bucket algorithm allows controlled bursting while maintaining
    long-term rate limits. Tokens accumulate at a fixed rate and are consumed
    per request.

    tokens(t) = min(max_tokens, tokens(t-1) + (t - last_refill) × rate)

    Key Parameters:
        - rate: Tokens added per second (requests/second capacity)
        - max_tokens: Bucket capacity (burst allowance)
        - tokens: Current available tokens

Leaky Bucket vs Token Bucket:
+------------------+------------------+------------------+
| Property | Token Bucket | Leaky Bucket |
+------------------+------------------+------------------+
| Burst handling | Allows bursts | Smooths output |
| Implementation | Counter-based | Queue-based |
| Best for | API rate limits | Traffic shaping |
| Our use case | Preferred | Not used |
+------------------+------------------+------------------+

==============================================================================
VENUE RATE LIMITS REFERENCE
==============================================================================

CEX Rate Limits:
+------------------+------------------+------------------+------------------+
| Venue | Requests/min | Weight System | Burst Allowed |
+------------------+------------------+------------------+------------------+
| Binance Spot | 1200 | Yes (1200/min) | Yes |
| Binance Futures | 2400 | Yes (2400/min) | Yes |
| Bybit | 120 | No | Limited |
| OKX | 60 | Yes (varies) | Yes |
| Coinbase | 10/sec | No | No |
| Kraken | 15/sec | Yes (decay) | Yes |
| Deribit | 20/sec | No | Yes |
+------------------+------------------+------------------+------------------+

Hybrid/DEX Rate Limits:
+------------------+------------------+------------------+------------------+
| Venue | Requests/min | Special Notes | Burst Allowed |
+------------------+------------------+------------------+------------------+
| Hyperliquid | 100 | WebSocket pref | Limited |
| dYdX v4 | 100 | Cosmos-based | Yes |
| GMX | Via RPC | Chain limits | No |
| Vertex | 60 | Per endpoint | Yes |
+------------------+------------------+------------------+------------------+

Data Provider Rate Limits:
+------------------+------------------+------------------+------------------+
| Provider | Requests/min | Tier System | Cost/Overage |
+------------------+------------------+------------------+------------------+
| The Graph | 1000 (free) | Yes | Query fees |
| CoinGecko | 10-50 | Yes | Paid tiers |
| DefiLlama | 60 | No | Free |
| Glassnode | 10 | Yes | Paid only |
| Kaiko | Custom | Enterprise | Contract |
+------------------+------------------+------------------+------------------+

==============================================================================
ADAPTIVE RATE LIMITING
==============================================================================

Response Header Parsing:
    Most APIs return rate limit info in headers:
    - X-RateLimit-Limit: Maximum requests allowed
    - X-RateLimit-Remaining: Requests remaining in window
    - X-RateLimit-Reset: Unix timestamp when limit resets
    - Retry-After: Seconds to wait (on 429 response)

Adaptive Strategy:
    1. Parse response headers after each request
    2. Adjust internal rate based on remaining quota
    3. Pre-emptively throttle before hitting limits
    4. Respect Retry-After headers on 429 responses

    adaptive_rate = remaining / (reset_time - current_time) × safety_factor

    where safety_factor ∈ [0.7, 0.9] prevents edge-case violations

==============================================================================
STATISTICAL ARBITRAGE IMPLICATIONS
==============================================================================

1. DATA FRESHNESS VS RATE LIMITS
   - Higher rate limits → fresher data → better signals
   - Rate limit violations → data gaps → signal degradation
   - Optimal: Stay at 80-90% of limit for safety margin

2. CROSS-VENUE SYNCHRONIZATION
   - Different venues have different limits
   - Prioritize rate allocation to signal-critical data
   - Funding rates: High priority (8h intervals)
   - OHLCV: Medium priority (can interpolate)
   - Orderbook: Low priority (snapshot-based)

3. BURST ALLOCATION FOR EVENTS
   - Reserve burst capacity for market events
   - Liquidation cascades require rapid data
   - Schedule batch operations during low-activity periods

4. COST OPTIMIZATION
   - Paid tiers have higher limits
   - ROI calculation: data_value / api_cost
   - Hybrid approach: free tier + burst paid access

==============================================================================
USAGE EXAMPLES
==============================================================================

Basic Rate Limiting:
    >>> limiter = RateLimiter(rate=100, per=60.0, name='hyperliquid')
    >>> await limiter.acquire()
    >>> response = await fetch_funding_rates()

Adaptive Rate Limiting:
    >>> limiter = AdaptiveRateLimiter(
    ... base_rate=1200, per=60.0, name='binance',
    ... safety_factor=0.85
    ... )
    >>> await limiter.acquire()
    >>> response = await api_call()
    >>> limiter.update_from_headers(response.headers)

Multi-Venue Management:
    >>> manager = RateLimitManager()
    >>> manager.add_venue('binance', VenueRateLimitConfig.BINANCE)
    >>> manager.add_venue('hyperliquid', VenueRateLimitConfig.HYPERLIQUID)
    >>>
    >>> async with manager.acquire('binance') as token:
    ... response = await binance_api.fetch()

Priority-Based Allocation:
    >>> limiter = PriorityRateLimiter(rate=100, per=60.0)
    >>> # High priority for funding rates
    >>> await limiter.acquire(priority=RequestPriority.CRITICAL)
    >>> # Low priority for historical data
    >>> await limiter.acquire(priority=RequestPriority.LOW)

Version: 2.0.0
"""

import asyncio
import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# ENUMS
# =============================================================================

class VenueTier(Enum):
    """
    Venue classification by rate limit characteristics.

    Determines base configuration and throttling strategies
    appropriate for each venue type.
    """

    TIER_1_HIGH_VOLUME = "tier_1_high_volume"
    TIER_2_STANDARD = "tier_2_standard"
    TIER_3_RESTRICTED = "tier_3_restricted"
    TIER_4_PREMIUM = "tier_4_premium"
    TIER_5_FREE = "tier_5_free"

    @property
    def base_requests_per_minute(self) -> int:
        """Typical requests per minute for this tier."""
        rates = {
            VenueTier.TIER_1_HIGH_VOLUME: 400,
            VenueTier.TIER_2_STANDARD: 60,
            VenueTier.TIER_3_RESTRICTED: 15,
            VenueTier.TIER_4_PREMIUM: 200,
            VenueTier.TIER_5_FREE: 5,
        }
        return rates.get(self, 60)

    @property
    def burst_multiplier(self) -> float:
        """Burst capacity as multiplier of base rate."""
        multipliers = {
            VenueTier.TIER_1_HIGH_VOLUME: 1.2,
            VenueTier.TIER_2_STANDARD: 1.0,
            VenueTier.TIER_3_RESTRICTED: 1.0,
            VenueTier.TIER_4_PREMIUM: 1.2,
            VenueTier.TIER_5_FREE: 1.0,
        }
        return multipliers.get(self, 1.0)

    @property
    def safety_factor(self) -> float:
        """Recommended safety margin (fraction of limit to use)."""
        factors = {
            VenueTier.TIER_1_HIGH_VOLUME: 0.50,
            VenueTier.TIER_2_STANDARD: 0.45,
            VenueTier.TIER_3_RESTRICTED: 0.40,
            VenueTier.TIER_4_PREMIUM: 0.55,
            VenueTier.TIER_5_FREE: 0.35,
        }
        return factors.get(self, 0.80)

    @property
    def retry_base_delay_seconds(self) -> float:
        """Base delay for retries after rate limit hit."""
        delays = {
            VenueTier.TIER_1_HIGH_VOLUME: 1.0,
            VenueTier.TIER_2_STANDARD: 2.0,
            VenueTier.TIER_3_RESTRICTED: 5.0,
            VenueTier.TIER_4_PREMIUM: 0.5,
            VenueTier.TIER_5_FREE: 10.0,
        }
        return delays.get(self, 2.0)

    @property
    def supports_adaptive(self) -> bool:
        """Whether venue provides rate limit headers."""
        return self in {
            VenueTier.TIER_1_HIGH_VOLUME,
            VenueTier.TIER_2_STANDARD,
            VenueTier.TIER_4_PREMIUM,
        }

class RateLimitStrategy(Enum):
    """
    Rate limiting strategy selection.

    Different strategies optimize for different use cases
    in the data collection pipeline.
    """

    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    BURST_THEN_STEADY = "burst"
    PRIORITY_BASED = "priority"

    @property
    def safety_factor(self) -> float:
        """Safety factor for this strategy."""
        factors = {
            RateLimitStrategy.FIXED: 0.85,
            RateLimitStrategy.ADAPTIVE: 0.90,
            RateLimitStrategy.AGGRESSIVE: 0.95,
            RateLimitStrategy.CONSERVATIVE: 0.70,
            RateLimitStrategy.BURST_THEN_STEADY: 0.85,
            RateLimitStrategy.PRIORITY_BASED: 0.85,
        }
        return factors.get(self, 0.85)

    @property
    def allows_burst(self) -> bool:
        """Whether strategy allows burst requests."""
        return self in {
            RateLimitStrategy.AGGRESSIVE,
            RateLimitStrategy.BURST_THEN_STEADY,
            RateLimitStrategy.ADAPTIVE,
        }

    @property
    def description(self) -> str:
        """Strategy description."""
        descriptions = {
            RateLimitStrategy.FIXED: "Fixed rate with no adaptation",
            RateLimitStrategy.ADAPTIVE: "Adapts to API response headers",
            RateLimitStrategy.AGGRESSIVE: "Maximizes throughput, higher risk",
            RateLimitStrategy.CONSERVATIVE: "Minimizes rate limit risk",
            RateLimitStrategy.BURST_THEN_STEADY: "Initial burst then steady rate",
            RateLimitStrategy.PRIORITY_BASED: "Allocates based on request priority",
        }
        return descriptions.get(self, "Unknown strategy")

class ThrottleStatus(Enum):
    """
    Current throttling status of a rate limiter.

    Used for monitoring and alerting on rate limit conditions.
    """

    NORMAL = "normal"
    ELEVATED = "elevated"
    WARNING = "warning"
    CRITICAL = "critical"
    THROTTLED = "throttled"
    BLOCKED = "blocked"
    RECOVERING = "recovering"

    @property
    def is_healthy(self) -> bool:
        """Whether status indicates healthy operation."""
        return self in {ThrottleStatus.NORMAL, ThrottleStatus.ELEVATED}

    @property
    def allows_requests(self) -> bool:
        """Whether requests should proceed."""
        return self not in {ThrottleStatus.BLOCKED}

    @property
    def utilization_range(self) -> Tuple[float, float]:
        """Utilization percentage range for this status."""
        ranges = {
            ThrottleStatus.NORMAL: (0.0, 0.70),
            ThrottleStatus.ELEVATED: (0.70, 0.90),
            ThrottleStatus.WARNING: (0.90, 0.95),
            ThrottleStatus.CRITICAL: (0.95, 1.0),
            ThrottleStatus.THROTTLED: (0.95, 1.0),
            ThrottleStatus.BLOCKED: (1.0, float('inf')),
            ThrottleStatus.RECOVERING: (0.50, 0.90),
        }
        return ranges.get(self, (0.0, 1.0))

    @classmethod
    def from_utilization(cls, utilization: float) -> 'ThrottleStatus':
        """Determine status from utilization percentage."""
        if utilization >= 1.0:
            return cls.BLOCKED
        elif utilization >= 0.95:
            return cls.CRITICAL
        elif utilization >= 0.90:
            return cls.WARNING
        elif utilization >= 0.70:
            return cls.ELEVATED
        return cls.NORMAL

    @property
    def alert_level(self) -> str:
        """Alert level for monitoring."""
        levels = {
            ThrottleStatus.NORMAL: "info",
            ThrottleStatus.ELEVATED: "info",
            ThrottleStatus.WARNING: "warning",
            ThrottleStatus.CRITICAL: "error",
            ThrottleStatus.THROTTLED: "error",
            ThrottleStatus.BLOCKED: "critical",
            ThrottleStatus.RECOVERING: "warning",
        }
        return levels.get(self, "info")

class RequestPriority(Enum):
    """
    Request priority levels for priority-based rate limiting.

    Higher priority requests are allocated rate limit capacity first.
    """

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"

    @property
    def weight(self) -> float:
        """Weight for rate allocation."""
        weights = {
            RequestPriority.CRITICAL: 10.0,
            RequestPriority.HIGH: 5.0,
            RequestPriority.NORMAL: 2.0,
            RequestPriority.LOW: 1.0,
            RequestPriority.BACKGROUND: 0.5,
        }
        return weights.get(self, 1.0)

    @property
    def max_wait_seconds(self) -> float:
        """Maximum wait time before timeout."""
        waits = {
            RequestPriority.CRITICAL: 30.0,
            RequestPriority.HIGH: 60.0,
            RequestPriority.NORMAL: 120.0,
            RequestPriority.LOW: 180.0,
            RequestPriority.BACKGROUND: 300.0,
        }
        return waits.get(self, 30.0)

    @property
    def can_be_dropped(self) -> bool:
        """Whether request can be dropped under pressure."""
        return self in {RequestPriority.LOW, RequestPriority.BACKGROUND}

    @property
    def reserved_capacity_pct(self) -> float:
        """Percentage of capacity reserved for this priority."""
        reserved = {
            RequestPriority.CRITICAL: 20.0,
            RequestPriority.HIGH: 30.0,
            RequestPriority.NORMAL: 30.0,
            RequestPriority.LOW: 15.0,
            RequestPriority.BACKGROUND: 5.0,
        }
        return reserved.get(self, 10.0)

class WindowType(Enum):
    """
    Rate limit window types.
    """

    FIXED = "fixed"
    SLIDING = "sliding"
    TOKEN_BUCKET = "token"
    LEAKY_BUCKET = "leaky"

    @property
    def allows_burst(self) -> bool:
        """Whether window type allows request bursting."""
        return self in {WindowType.TOKEN_BUCKET, WindowType.SLIDING}

    @property
    def description(self) -> str:
        """Window type description."""
        descriptions = {
            WindowType.FIXED: "Fixed time windows, resets at intervals",
            WindowType.SLIDING: "Rolling window, continuous tracking",
            WindowType.TOKEN_BUCKET: "Token bucket with burst allowance",
            WindowType.LEAKY_BUCKET: "Leaky bucket, smooths traffic",
        }
        return descriptions.get(self, "Unknown")

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class RateLimiterStats:
    """
    Comprehensive statistics for rate limiter performance monitoring.
    """

    requests_made: int = 0
    requests_delayed: int = 0
    requests_dropped: int = 0
    total_wait_time_seconds: float = 0.0
    rate_limit_hits: int = 0
    burst_requests: int = 0
    window_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    requests_by_priority: Dict[str, int] = field(default_factory=dict)

    @property
    def total_requests(self) -> int:
        """Total requests (successful + dropped)."""
        return self.requests_made + self.requests_dropped

    @property
    def success_rate(self) -> float:
        """Percentage of requests that succeeded."""
        if self.total_requests == 0:
            return 100.0
        return (self.requests_made / self.total_requests) * 100

    @property
    def delay_rate(self) -> float:
        """Percentage of requests that were delayed."""
        if self.requests_made == 0:
            return 0.0
        return (self.requests_delayed / self.requests_made) * 100

    @property
    def average_wait_time(self) -> float:
        """Average wait time per delayed request."""
        if self.requests_delayed == 0:
            return 0.0
        return self.total_wait_time_seconds / self.requests_delayed

    @property
    def burst_rate(self) -> float:
        """Percentage of requests that used burst capacity."""
        if self.requests_made == 0:
            return 0.0
        return (self.burst_requests / self.requests_made) * 100

    @property
    def rate_limit_hit_rate(self) -> float:
        """Rate limit hits per 1000 requests."""
        if self.requests_made == 0:
            return 0.0
        return (self.rate_limit_hits / self.requests_made) * 1000

    @property
    def window_duration_seconds(self) -> float:
        """Duration of statistics window."""
        return (datetime.now(timezone.utc) - self.window_start).total_seconds()

    @property
    def requests_per_minute(self) -> float:
        """Average requests per minute in window."""
        duration_minutes = self.window_duration_seconds / 60
        if duration_minutes == 0:
            return 0.0
        return self.requests_made / duration_minutes

    @property
    def health_score(self) -> float:
        """Overall health score (0-100)."""
        score = 100.0
        score -= min(20, self.delay_rate * 0.5)
        score -= min(30, self.rate_limit_hit_rate * 3)
        drop_rate = (self.requests_dropped / max(self.total_requests, 1)) * 100
        score -= min(30, drop_rate * 3)
        return max(0, score)

    def reset(self) -> None:
        """Reset all statistics."""
        self.requests_made = 0
        self.requests_delayed = 0
        self.requests_dropped = 0
        self.total_wait_time_seconds = 0.0
        self.rate_limit_hits = 0
        self.burst_requests = 0
        self.window_start = datetime.now(timezone.utc)
        self.requests_by_priority = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'requests_made': self.requests_made,
            'requests_delayed': self.requests_delayed,
            'requests_dropped': self.requests_dropped,
            'total_wait_time_seconds': round(self.total_wait_time_seconds, 3),
            'rate_limit_hits': self.rate_limit_hits,
            'burst_requests': self.burst_requests,
            'success_rate': round(self.success_rate, 2),
            'delay_rate': round(self.delay_rate, 2),
            'average_wait_time': round(self.average_wait_time, 3),
            'burst_rate': round(self.burst_rate, 2),
            'rate_limit_hit_rate': round(self.rate_limit_hit_rate, 2),
            'requests_per_minute': round(self.requests_per_minute, 2),
            'health_score': round(self.health_score, 2),
            'window_duration_seconds': round(self.window_duration_seconds, 2),
            'requests_by_priority': self.requests_by_priority,
        }

@dataclass
class RateLimitState:
    """
    Current state of rate limiting for monitoring.
    """

    name: str
    status: ThrottleStatus
    tokens_available: float
    tokens_max: float
    utilization: float
    requests_in_window: int
    window_limit: int
    reset_at: Optional[datetime]
    last_request_at: Optional[datetime]
    strategy: RateLimitStrategy

    @property
    def tokens_used(self) -> float:
        """Tokens consumed."""
        return self.tokens_max - self.tokens_available

    @property
    def is_healthy(self) -> bool:
        """Whether rate limiter is healthy."""
        return self.status.is_healthy

    @property
    def time_until_reset(self) -> Optional[float]:
        """Seconds until rate limit resets."""
        if self.reset_at is None:
            return None
        delta = self.reset_at - datetime.now(timezone.utc)
        return max(0, delta.total_seconds())

    @property
    def headroom_pct(self) -> float:
        """Percentage of capacity remaining."""
        return (1.0 - self.utilization) * 100

    @property
    def can_burst(self) -> bool:
        """Whether burst capacity is available."""
        return self.utilization < 0.7 and self.strategy.allows_burst

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'status': self.status.value,
            'tokens_available': round(self.tokens_available, 2),
            'tokens_max': self.tokens_max,
            'tokens_used': round(self.tokens_used, 2),
            'utilization': round(self.utilization, 4),
            'requests_in_window': self.requests_in_window,
            'window_limit': self.window_limit,
            'reset_at': self.reset_at.isoformat() if self.reset_at else None,
            'time_until_reset': self.time_until_reset,
            'last_request_at': self.last_request_at.isoformat() if self.last_request_at else None,
            'is_healthy': self.is_healthy,
            'headroom_pct': round(self.headroom_pct, 2),
            'can_burst': self.can_burst,
            'strategy': self.strategy.value,
        }

@dataclass
class VenueRateLimitConfig:
    """
    Venue-specific rate limit configuration.
    """

    venue: str
    tier: VenueTier
    requests_per_minute: int
    burst_size: Optional[int] = None
    window_type: WindowType = WindowType.TOKEN_BUCKET
    strategy: RateLimitStrategy = RateLimitStrategy.ADAPTIVE
    weight_system: bool = False
    weight_per_request: int = 1
    max_weight_per_minute: Optional[int] = None
    limit_header: str = "X-RateLimit-Limit"
    remaining_header: str = "X-RateLimit-Remaining"
    reset_header: str = "X-RateLimit-Reset"

    @property
    def requests_per_second(self) -> float:
        """Requests per second."""
        return self.requests_per_minute / 60

    @property
    def effective_burst_size(self) -> int:
        """Effective burst size."""
        if self.burst_size:
            return self.burst_size
        return int(self.requests_per_minute * self.tier.burst_multiplier / 60)

    @property
    def safe_requests_per_minute(self) -> int:
        """Safe requests per minute with safety factor."""
        return int(self.requests_per_minute * self.tier.safety_factor)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'venue': self.venue,
            'tier': self.tier.value,
            'requests_per_minute': self.requests_per_minute,
            'requests_per_second': round(self.requests_per_second, 2),
            'burst_size': self.effective_burst_size,
            'window_type': self.window_type.value,
            'strategy': self.strategy.value,
            'weight_system': self.weight_system,
            'safe_requests_per_minute': self.safe_requests_per_minute,
        }

    @classmethod
    def binance(cls) -> 'VenueRateLimitConfig':
        """Binance configuration - conservative to prevent IP bans."""
        return cls(
            venue='binance',
            tier=VenueTier.TIER_1_HIGH_VOLUME,
            requests_per_minute=400,
            burst_size=20,
            weight_system=True,
            weight_per_request=1,
            max_weight_per_minute=400,
            limit_header='X-MBX-USED-WEIGHT-1M',
        )

    @classmethod
    def binance_futures(cls) -> 'VenueRateLimitConfig':
        """Binance Futures configuration - conservative."""
        return cls(
            venue='binance_futures',
            tier=VenueTier.TIER_1_HIGH_VOLUME,
            requests_per_minute=600,
            burst_size=30,
            weight_system=True,
            weight_per_request=1,
            max_weight_per_minute=600,
        )

    @classmethod
    def bybit(cls) -> 'VenueRateLimitConfig':
        """Bybit configuration."""
        return cls(
            venue='bybit',
            tier=VenueTier.TIER_2_STANDARD,
            requests_per_minute=60,
            burst_size=10,
            strategy=RateLimitStrategy.CONSERVATIVE,
        )

    @classmethod
    def okx(cls) -> 'VenueRateLimitConfig':
        """OKX configuration."""
        return cls(
            venue='okx',
            tier=VenueTier.TIER_2_STANDARD,
            requests_per_minute=30,
            burst_size=5,
            weight_system=True,
        )

    @classmethod
    def hyperliquid(cls) -> 'VenueRateLimitConfig':
        """Hyperliquid configuration."""
        return cls(
            venue='hyperliquid',
            tier=VenueTier.TIER_3_RESTRICTED,
            requests_per_minute=30,
            burst_size=5,
            strategy=RateLimitStrategy.CONSERVATIVE,
        )

    @classmethod
    def dydx_v4(cls) -> 'VenueRateLimitConfig':
        """dYdX v4 configuration."""
        return cls(
            venue='dydx_v4',
            tier=VenueTier.TIER_2_STANDARD,
            requests_per_minute=50,
            burst_size=5,
        )

    @classmethod
    def deribit(cls) -> 'VenueRateLimitConfig':
        """Deribit configuration."""
        return cls(
            venue='deribit',
            tier=VenueTier.TIER_1_HIGH_VOLUME,
            requests_per_minute=400,
            burst_size=20,
        )

    @classmethod
    def coinbase(cls) -> 'VenueRateLimitConfig':
        """Coinbase configuration."""
        return cls(
            venue='coinbase',
            tier=VenueTier.TIER_2_STANDARD,
            requests_per_minute=200,
            burst_size=10,
            strategy=RateLimitStrategy.FIXED,
        )

    @classmethod
    def thegraph(cls) -> 'VenueRateLimitConfig':
        """The Graph configuration."""
        return cls(
            venue='thegraph',
            tier=VenueTier.TIER_5_FREE,
            requests_per_minute=60,
            burst_size=10,
            strategy=RateLimitStrategy.BURST_THEN_STEADY,
        )

    @classmethod
    def coingecko_free(cls) -> 'VenueRateLimitConfig':
        """CoinGecko free tier configuration."""
        return cls(
            venue='coingecko',
            tier=VenueTier.TIER_5_FREE,
            requests_per_minute=5,
            burst_size=2,
            strategy=RateLimitStrategy.CONSERVATIVE,
        )

    @classmethod
    def defillama(cls) -> 'VenueRateLimitConfig':
        """DefiLlama configuration."""
        return cls(
            venue='defillama',
            tier=VenueTier.TIER_5_FREE,
            requests_per_minute=30,
            burst_size=5,
        )

@dataclass
class AcquireResult:
    """
    Result of acquiring rate limit token.
    """

    acquired: bool
    wait_time_seconds: float
    priority: RequestPriority
    was_burst: bool
    tokens_remaining: float
    status_before: ThrottleStatus
    status_after: ThrottleStatus
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def was_delayed(self) -> bool:
        """Whether request was delayed."""
        return self.wait_time_seconds > 0

    @property
    def status_changed(self) -> bool:
        """Whether status changed during acquire."""
        return self.status_before != self.status_after

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'acquired': self.acquired,
            'wait_time_seconds': round(self.wait_time_seconds, 4),
            'priority': self.priority.value,
            'was_burst': self.was_burst,
            'was_delayed': self.was_delayed,
            'tokens_remaining': round(self.tokens_remaining, 2),
            'status_before': self.status_before.value,
            'status_after': self.status_after.value,
            'status_changed': self.status_changed,
            'timestamp': self.timestamp.isoformat(),
        }

# =============================================================================
# RATE LIMITER CLASSES
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter with comprehensive monitoring.

    Implements token bucket algorithm with support for configurable rate,
    burst size, priority-based requests, and comprehensive statistics.
    """

    def __init__(
        self,
        rate: int,
        per: float = 60.0,
        burst: Optional[int] = None,
        name: Optional[str] = None,
        strategy: RateLimitStrategy = RateLimitStrategy.FIXED,
    ):
        self.rate = rate
        self.per = per
        self.name = name or "rate_limiter"
        self.strategy = strategy

        self.tokens_per_second = rate / per
        self.max_tokens = burst if burst is not None else max(1, int(rate / 60))
        self.tokens = float(self.max_tokens)
        self.last_refill = time.monotonic()

        self._requests_in_window: deque = deque()
        self._window_size = per
        self._lock = asyncio.Lock()

        self.stats = RateLimiterStats()
        self._last_request_at: Optional[datetime] = None
        self._rate_limit_reset_at: Optional[datetime] = None

        logger.debug(
            f"RateLimiter '{self.name}' initialized: "
            f"{rate} req/{per}s, burst={self.max_tokens}, strategy={strategy.value}"
        )

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.tokens_per_second
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        self.last_refill = now

    def _clean_request_window(self) -> None:
        """Remove old requests from tracking window."""
        cutoff = time.monotonic() - self._window_size
        while self._requests_in_window and self._requests_in_window[0] < cutoff:
            self._requests_in_window.popleft()

    @property
    def utilization(self) -> float:
        """Current utilization (0.0 to 1.0)."""
        self._refill()
        return 1.0 - (self.tokens / self.max_tokens)

    @property
    def status(self) -> ThrottleStatus:
        """Current throttle status."""
        return ThrottleStatus.from_utilization(self.utilization)

    def get_state(self) -> RateLimitState:
        """Get current rate limit state."""
        self._clean_request_window()
        self._refill()

        return RateLimitState(
            name=self.name,
            status=self.status,
            tokens_available=self.tokens,
            tokens_max=self.max_tokens,
            utilization=self.utilization,
            requests_in_window=len(self._requests_in_window),
            window_limit=self.rate,
            reset_at=self._rate_limit_reset_at,
            last_request_at=self._last_request_at,
            strategy=self.strategy,
        )

    async def acquire(
        self,
        tokens: int = 1,
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: Optional[float] = None,
    ) -> AcquireResult:
        """Acquire rate limit tokens."""
        if timeout is None:
            timeout = priority.max_wait_seconds

        status_before = self.status
        wait_time = 0.0
        was_burst = False

        async with self._lock:
            self._refill()

            if self.tokens < tokens:
                tokens_needed = tokens - self.tokens
                required_wait = tokens_needed / self.tokens_per_second

                if required_wait > timeout:
                    self.stats.requests_dropped += 1
                    return AcquireResult(
                        acquired=False,
                        wait_time_seconds=0,
                        priority=priority,
                        was_burst=False,
                        tokens_remaining=self.tokens,
                        status_before=status_before,
                        status_after=self.status,
                    )

                self.stats.requests_delayed += 1
                self.stats.total_wait_time_seconds += required_wait

                self._lock.release()
                try:
                    await asyncio.sleep(required_wait)
                    wait_time = required_wait
                finally:
                    await self._lock.acquire()

                self._refill()

            if self.tokens > self.max_tokens * 0.5:
                was_burst = True
                self.stats.burst_requests += 1

            self.tokens -= tokens
            self._requests_in_window.append(time.monotonic())
            self._last_request_at = datetime.now(timezone.utc)

            self.stats.requests_made += 1
            priority_key = priority.value
            self.stats.requests_by_priority[priority_key] = \
                self.stats.requests_by_priority.get(priority_key, 0) + 1

        return AcquireResult(
            acquired=True,
            wait_time_seconds=wait_time,
            priority=priority,
            was_burst=was_burst,
            tokens_remaining=self.tokens,
            status_before=status_before,
            status_after=self.status,
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass

    def available(self) -> int:
        """Get number of immediately available tokens."""
        self._refill()
        return int(self.tokens)

    def reset(self) -> None:
        """Reset rate limiter to full capacity."""
        self.tokens = float(self.max_tokens)
        self.last_refill = time.monotonic()
        self._requests_in_window.clear()
        logger.debug(f"RateLimiter '{self.name}' reset")

    def __repr__(self) -> str:
        return (
            f"RateLimiter(name={self.name}, rate={self.rate}/{self.per}s, "
            f"tokens={self.tokens:.1f}/{self.max_tokens}, status={self.status.value})"
        )

class AdaptiveRateLimiter(RateLimiter):
    """
    Rate limiter that adapts based on API response headers.
    """

    def __init__(
        self,
        base_rate: int,
        per: float = 60.0,
        safety_factor: float = 0.85,
        config: Optional[VenueRateLimitConfig] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            rate=base_rate,
            per=per,
            name=name or (config.venue if config else "adaptive"),
            strategy=RateLimitStrategy.ADAPTIVE,
        )

        self.base_rate = base_rate
        self.safety_factor = safety_factor
        self.config = config

        self._current_limit: Optional[int] = None
        self._current_remaining: Optional[int] = None
        self._reset_timestamp: Optional[float] = None

        self._limit_header = config.limit_header if config else "X-RateLimit-Limit"
        self._remaining_header = config.remaining_header if config else "X-RateLimit-Remaining"
        self._reset_header = config.reset_header if config else "X-RateLimit-Reset"

    def update_from_headers(self, headers: Dict[str, str]) -> None:
        """Update rate limits from API response headers."""
        try:
            if self._limit_header in headers:
                self._current_limit = int(headers[self._limit_header])

            if self._remaining_header in headers:
                self._current_remaining = int(headers[self._remaining_header])

            if self._reset_header in headers:
                reset_val = headers[self._reset_header]
                if len(reset_val) > 10:
                    self._reset_timestamp = float(reset_val)
                else:
                    self._reset_timestamp = time.time() + float(reset_val)

                self._rate_limit_reset_at = datetime.fromtimestamp(
                    self._reset_timestamp, tz=timezone.utc
                )

            if self._current_remaining is not None and self._reset_timestamp is not None:
                self._adapt_rate()

        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse rate limit headers: {e}")

    def _adapt_rate(self) -> None:
        """Adapt rate based on current state."""
        if self._current_remaining is None or self._reset_timestamp is None:
            return

        time_until_reset = max(1, self._reset_timestamp - time.time())
        safe_remaining = int(self._current_remaining * self.safety_factor)
        new_rate = safe_remaining / time_until_reset

        min_rate = self.base_rate / self.per * 0.1
        max_rate = self.base_rate / self.per * 1.5

        self.tokens_per_second = max(min_rate, min(max_rate, new_rate))

        logger.debug(
            f"AdaptiveRateLimiter '{self.name}' adapted: "
            f"remaining={self._current_remaining}, rate={self.tokens_per_second:.2f}/s"
        )

    def handle_rate_limit_response(self, retry_after: Optional[float] = None) -> None:
        """Handle 429 rate limit response."""
        self.stats.rate_limit_hits += 1

        if retry_after:
            self._rate_limit_reset_at = datetime.now(timezone.utc) + timedelta(seconds=retry_after)
            self._reset_timestamp = time.time() + retry_after

        self.tokens_per_second *= 0.5
        self.tokens = 0

        logger.warning(
            f"AdaptiveRateLimiter '{self.name}' hit rate limit, "
            f"retry_after={retry_after}s"
        )

class MultiRateLimiter:
    """
    Manager for multiple rate limiters across venues.
    """

    def __init__(self):
        self._limiters: Dict[str, RateLimiter] = {}

    def add(
        self,
        name: str,
        rate: int,
        per: float = 60.0,
        burst: Optional[int] = None,
        strategy: RateLimitStrategy = RateLimitStrategy.FIXED,
    ) -> RateLimiter:
        """Add a rate limiter."""
        limiter = RateLimiter(
            rate=rate,
            per=per,
            burst=burst,
            name=name,
            strategy=strategy,
        )
        self._limiters[name] = limiter
        return limiter

    def add_from_config(self, config: VenueRateLimitConfig) -> RateLimiter:
        """Add rate limiter from venue configuration."""
        if config.tier.supports_adaptive:
            limiter = AdaptiveRateLimiter(
                base_rate=config.requests_per_minute,
                per=60.0,
                safety_factor=config.tier.safety_factor,
                config=config,
                name=config.venue,
            )
        else:
            limiter = RateLimiter(
                rate=config.requests_per_minute,
                per=60.0,
                burst=config.effective_burst_size,
                name=config.venue,
                strategy=config.strategy,
            )

        self._limiters[config.venue] = limiter
        return limiter

    def get(self, name: str) -> Optional[RateLimiter]:
        """Get rate limiter by name."""
        return self._limiters.get(name)

    async def acquire(
        self,
        name: str,
        tokens: int = 1,
        priority: RequestPriority = RequestPriority.NORMAL,
    ) -> AcquireResult:
        """Acquire tokens from a specific limiter."""
        limiter = self._limiters.get(name)
        if limiter is None:
            raise KeyError(f"Rate limiter '{name}' not found")
        return await limiter.acquire(tokens, priority)

    def get_all_states(self) -> Dict[str, RateLimitState]:
        """Get states of all limiters."""
        return {name: lim.get_state() for name, lim in self._limiters.items()}

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all limiters."""
        return {name: lim.stats.to_dict() for name, lim in self._limiters.items()}

    def reset_all(self) -> None:
        """Reset all rate limiters."""
        for limiter in self._limiters.values():
            limiter.reset()

    def reset_stats(self) -> None:
        """Reset stats for all limiters."""
        for limiter in self._limiters.values():
            limiter.stats.reset()

    @property
    def overall_health(self) -> float:
        """Overall health score across all limiters."""
        if not self._limiters:
            return 100.0
        scores = [lim.stats.health_score for lim in self._limiters.values()]
        return sum(scores) / len(scores)

    def __repr__(self) -> str:
        return f"MultiRateLimiter(venues={list(self._limiters.keys())})"

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_venue_limiters() -> MultiRateLimiter:
    """
    Create pre-configured rate limiters for all 44+ supported venues.
    """
    limiters = MultiRateLimiter()

    # CEX venues (with pre-configured classes)
    limiters.add_from_config(VenueRateLimitConfig.binance())
    limiters.add_from_config(VenueRateLimitConfig.binance_futures())
    limiters.add_from_config(VenueRateLimitConfig.bybit())
    limiters.add_from_config(VenueRateLimitConfig.okx())
    limiters.add_from_config(VenueRateLimitConfig.coinbase())
    limiters.add_from_config(VenueRateLimitConfig.deribit())
    limiters.add_from_config(VenueRateLimitConfig.hyperliquid())
    limiters.add_from_config(VenueRateLimitConfig.dydx_v4())
    limiters.add_from_config(VenueRateLimitConfig.thegraph())
    limiters.add_from_config(VenueRateLimitConfig.defillama())
    limiters.add_from_config(VenueRateLimitConfig.coingecko_free())

    # CEX venues (manual configuration)
    limiters.add('kraken', rate=900, per=60.0, burst=30)
    limiters.add('cme', rate=60, per=60.0, burst=5)

    # Hybrid venues
    limiters.add('drift', rate=100, per=60.0, burst=10, strategy=RateLimitStrategy.CONSERVATIVE)

    # DEX venues
    limiters.add('geckoterminal', rate=30, per=60.0, burst=5)
    limiters.add('dexscreener', rate=60, per=60.0, burst=10)
    limiters.add('uniswap', rate=60, per=60.0, burst=10)
    limiters.add('sushiswap', rate=60, per=60.0, burst=10)
    limiters.add('curve', rate=60, per=60.0, burst=10)
    limiters.add('gmx', rate=60, per=60.0, burst=10)
    limiters.add('vertex', rate=60, per=60.0, burst=10)
    limiters.add('jupiter', rate=60, per=60.0, burst=10)
    limiters.add('cowswap', rate=60, per=60.0, burst=10)
    limiters.add('oneinch', rate=60, per=60.0, burst=10)
    limiters.add('zerox', rate=60, per=60.0, burst=10)

    # Options venues
    limiters.add('aevo', rate=100, per=60.0, burst=10)
    limiters.add('lyra', rate=60, per=60.0, burst=10)
    limiters.add('dopex', rate=60, per=60.0, burst=10)

    # On-chain analytics venues
    limiters.add('glassnode', rate=60, per=60.0, burst=5, strategy=RateLimitStrategy.CONSERVATIVE)
    limiters.add('santiment', rate=60, per=60.0, burst=5, strategy=RateLimitStrategy.CONSERVATIVE)
    limiters.add('cryptoquant', rate=60, per=60.0, burst=5, strategy=RateLimitStrategy.CONSERVATIVE)
    limiters.add('coinmetrics', rate=60, per=60.0, burst=5, strategy=RateLimitStrategy.CONSERVATIVE)
    limiters.add('nansen', rate=60, per=60.0, burst=5, strategy=RateLimitStrategy.CONSERVATIVE)
    limiters.add('arkham', rate=60, per=60.0, burst=5, strategy=RateLimitStrategy.CONSERVATIVE)
    limiters.add('flipside', rate=60, per=60.0, burst=5, strategy=RateLimitStrategy.CONSERVATIVE)
    limiters.add('covalent', rate=60, per=60.0, burst=10)
    limiters.add('bitquery', rate=10, per=60.0, burst=2, strategy=RateLimitStrategy.CONSERVATIVE)
    limiters.add('whale_alert', rate=60, per=60.0, burst=5)

    # Market data providers
    limiters.add('cryptocompare', rate=100, per=60.0, burst=10)
    limiters.add('messari', rate=60, per=60.0, burst=10)
    limiters.add('kaiko', rate=60, per=60.0, burst=10)

    # Alternative data
    limiters.add('coinalyze', rate=20, per=60.0, burst=3, strategy=RateLimitStrategy.CONSERVATIVE)
    limiters.add('coinalyze_enhanced', rate=20, per=60.0, burst=3, strategy=RateLimitStrategy.CONSERVATIVE)
    limiters.add('lunarcrush', rate=60, per=60.0, burst=10)
    limiters.add('dune', rate=60, per=60.0, burst=10)
    limiters.add('coinglass', rate=30, per=60.0, burst=5)

    return limiters

async def rate_limited_request(
    limiter: RateLimiter,
    coro: Callable,
    *args,
    priority: RequestPriority = RequestPriority.NORMAL,
    **kwargs,
) -> Any:
    """Execute coroutine with rate limiting."""
    result = await limiter.acquire(priority=priority)
    if not result.acquired:
        raise RuntimeError(f"Failed to acquire rate limit token for {limiter.name}")
    return await coro(*args, **kwargs)

# =============================================================================
# GLOBAL RATE LIMITER REGISTRY (SINGLETON)
# =============================================================================
# This registry ensures rate limiters are created ONCE and shared across all
# collector instances, preventing the massive overhead of re-initialization.
# Before: 62,524 rate limiter initializations
# After: ~47 rate limiter initializations (one per venue)

class RateLimiterRegistry:
    """
    Global singleton registry for shared rate limiters.

    This solves the critical bottleneck where rate limiters were being
    re-initialized 62,524 times instead of being shared across collectors.

    Usage:
        >>> limiter = get_shared_rate_limiter('binance', rate=1200, per=60.0)
        >>> # Returns existing limiter or creates new one
    """

    _instance: Optional['RateLimiterRegistry'] = None
    _lock: asyncio.Lock = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._limiters: Dict[str, RateLimiter] = {}
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._limiters: Dict[str, RateLimiter] = {}
            self._initialized = True
            logger.info("RateLimiterRegistry initialized (singleton)")

    def get(
        self,
        name: str,
        rate: int = 60,
        per: float = 60.0,
        burst: Optional[int] = None,
        strategy: RateLimitStrategy = RateLimitStrategy.FIXED,
    ) -> RateLimiter:
        """
        Get or create a shared rate limiter.

        If a rate limiter with this name already exists, returns the existing one.
        Otherwise creates a new one with the specified parameters.

        Parameters
        ----------
        name : str
            Unique name for the rate limiter (typically venue name)
        rate : int
            Requests allowed per time period
        per : float
            Time period in seconds
        burst : int, optional
            Burst capacity
        strategy : RateLimitStrategy
            Rate limiting strategy

        Returns
        -------
        RateLimiter
            Shared rate limiter instance
        """
        if name not in self._limiters:
            self._limiters[name] = RateLimiter(
                rate=rate,
                per=per,
                burst=burst,
                name=name,
                strategy=strategy,
            )
            # Log only on first creation (not DEBUG level to track actual creations)
            logger.info(f"RateLimiterRegistry: Created shared limiter '{name}'")
        return self._limiters[name]

    def get_adaptive(
        self,
        name: str,
        base_rate: int = 60,
        per: float = 60.0,
        safety_factor: float = 0.85,
        config: Optional[VenueRateLimitConfig] = None,
    ) -> AdaptiveRateLimiter:
        """Get or create a shared adaptive rate limiter."""
        if name not in self._limiters:
            self._limiters[name] = AdaptiveRateLimiter(
                base_rate=base_rate,
                per=per,
                safety_factor=safety_factor,
                config=config,
                name=name,
            )
            logger.info(f"RateLimiterRegistry: Created shared adaptive limiter '{name}'")
        return self._limiters[name]

    def get_all(self) -> Dict[str, RateLimiter]:
        """Get all registered rate limiters."""
        return self._limiters.copy()

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all rate limiters."""
        return {name: lim.stats.to_dict() for name, lim in self._limiters.items()}

    def reset_all(self) -> None:
        """Reset all rate limiters to full capacity."""
        for limiter in self._limiters.values():
            limiter.reset()
        logger.info(f"RateLimiterRegistry: Reset {len(self._limiters)} limiters")

    @property
    def count(self) -> int:
        """Number of registered rate limiters."""
        return len(self._limiters)

# Global registry instance
_rate_limiter_registry: Optional[RateLimiterRegistry] = None

def get_rate_limiter_registry() -> RateLimiterRegistry:
    """Get the global rate limiter registry singleton."""
    global _rate_limiter_registry
    if _rate_limiter_registry is None:
        _rate_limiter_registry = RateLimiterRegistry()
    return _rate_limiter_registry

def get_shared_rate_limiter(
    name: str,
    rate: int = 60,
    per: float = 60.0,
    burst: Optional[int] = None,
    strategy: RateLimitStrategy = RateLimitStrategy.FIXED,
) -> RateLimiter:
    """
    Get a shared rate limiter from the global registry.

    This is the preferred way for collectors to get rate limiters,
    ensuring they share the same instance across all collector instances.

    Example:
        >>> # In collector __init__:
        >>> self.rate_limiter = get_shared_rate_limiter(
        ... 'binance', rate=1200, per=60.0, burst=100
        ... )
    """
    return get_rate_limiter_registry().get(name, rate, per, burst, strategy)

__all__ = [
    # Enums
    'VenueTier',
    'RateLimitStrategy',
    'ThrottleStatus',
    'RequestPriority',
    'WindowType',
    # Dataclasses
    'RateLimiterStats',
    'RateLimitState',
    'VenueRateLimitConfig',
    'AcquireResult',
    # Classes
    'RateLimiter',
    'AdaptiveRateLimiter',
    'MultiRateLimiter',
    'RateLimiterRegistry',
    # Functions
    'create_venue_limiters',
    'rate_limited_request',
    'get_rate_limiter_registry',
    'get_shared_rate_limiter',
]
