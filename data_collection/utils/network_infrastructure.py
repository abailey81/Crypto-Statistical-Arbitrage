"""
Unified Network Infrastructure for Crypto Statistical Arbitrage Systems.

Consolidates rate limiting, retry handling, and circuit breaker into single module.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar('T')

class VenueTier(Enum):
    TIER_1_HIGH_VOLUME = auto()
    TIER_2_STANDARD = auto()
    TIER_3_RESTRICTED = auto()
    TIER_4_PREMIUM = auto()
    TIER_5_FREE = auto()

class RateLimitStrategy(Enum):
    FIXED = auto()
    ADAPTIVE = auto()
    AGGRESSIVE = auto()
    CONSERVATIVE = auto()
    BURST_THEN_STEADY = auto()
    PRIORITY_BASED = auto()

class ThrottleStatus(Enum):
    NORMAL = auto()
    ELEVATED = auto()
    WARNING = auto()
    CRITICAL = auto()
    THROTTLED = auto()
    BLOCKED = auto()
    RECOVERING = auto()

class RequestPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

class RetryStrategy(Enum):
    CONSTANT = auto()
    LINEAR = auto()
    EXPONENTIAL = auto()
    FIBONACCI = auto()
    DECORRELATED = auto()
    
    def calculate_delay(self, attempt: int, base_delay: float, max_delay: float, last_delay: float = 0.0) -> float:
        if self == RetryStrategy.CONSTANT:
            delay = base_delay
        elif self == RetryStrategy.LINEAR:
            delay = base_delay * (attempt + 1)
        elif self == RetryStrategy.EXPONENTIAL:
            delay = base_delay * (2 ** attempt)
        elif self == RetryStrategy.FIBONACCI:
            a, b = 0, 1
            for _ in range(attempt + 2):
                a, b = b, a + b
            delay = base_delay * a
        elif self == RetryStrategy.DECORRELATED:
            delay = random.uniform(base_delay, max(base_delay, 3 * last_delay))
        else:
            delay = base_delay
        return min(delay, max_delay)

class FailureCategory(Enum):
    TRANSIENT = auto()
    RATE_LIMIT = auto()
    SERVER_ERROR = auto()
    CLIENT_ERROR = auto()
    AUTHENTICATION = auto()
    NETWORK = auto()
    TIMEOUT = auto()
    UNKNOWN = auto()
    
    @property
    def should_retry(self) -> bool:
        return self in {FailureCategory.TRANSIENT, FailureCategory.RATE_LIMIT, FailureCategory.SERVER_ERROR, FailureCategory.NETWORK, FailureCategory.TIMEOUT, FailureCategory.UNKNOWN}
    
    @property
    def base_delay_modifier(self) -> float:
        return {FailureCategory.TRANSIENT: 1.0, FailureCategory.RATE_LIMIT: 2.0, FailureCategory.SERVER_ERROR: 1.5, FailureCategory.CLIENT_ERROR: 1.0, FailureCategory.AUTHENTICATION: 1.0, FailureCategory.NETWORK: 1.0, FailureCategory.TIMEOUT: 1.5, FailureCategory.UNKNOWN: 1.0}[self]

class CircuitState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()

class RecoveryAction(Enum):
    IMMEDIATE = auto()
    GRADUAL = auto()
    MANUAL = auto()

@dataclass
class RateLimitConfig:
    venue: str
    requests_per_minute: int = 60
    burst_capacity: Optional[int] = None
    strategy: RateLimitStrategy = RateLimitStrategy.ADAPTIVE
    safety_factor: float = 0.85
    
    def __post_init__(self):
        if self.burst_capacity is None:
            self.burst_capacity = int(self.requests_per_minute / 6)
    
    @property
    def tokens_per_second(self) -> float:
        return self.requests_per_minute / 60.0

@dataclass
class CircuitBreakerConfig:
    venue: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    success_threshold: int = 2
    failure_window_seconds: float = 60.0
    recovery_action: RecoveryAction = RecoveryAction.GRADUAL

@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter_factor: float = 0.25
    retry_status_codes: Set[int] = field(default_factory=lambda: {408, 429, 500, 502, 503, 504})
    no_retry_status_codes: Set[int] = field(default_factory=lambda: {400, 401, 403, 404, 405, 422})

@dataclass
class AcquireResult:
    acquired: bool
    wait_time: float
    tokens_remaining: float
    throttle_status: ThrottleStatus
    message: str = ""

@dataclass
class ExecutionResult:
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    attempts: int = 1
    total_delay: float = 0.0
    rate_limit_wait: float = 0.0
    failure_category: Optional[FailureCategory] = None

class RateLimiter:
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.venue = config.venue
        self._tokens = float(config.burst_capacity)
        self._max_tokens = float(config.burst_capacity)
        self._tokens_per_second = config.tokens_per_second
        self._last_update = time.monotonic()
        self._adaptive_rate: Optional[float] = None
        self._lock = asyncio.Lock()
    
    def _refill_tokens(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now
        rate = self._adaptive_rate or self._tokens_per_second
        rate *= self.config.safety_factor
        self._tokens = min(self._max_tokens, self._tokens + elapsed * rate)
    
    def _get_throttle_status(self) -> ThrottleStatus:
        ratio = self._tokens / self._max_tokens
        if ratio >= 0.9: return ThrottleStatus.NORMAL
        elif ratio >= 0.7: return ThrottleStatus.ELEVATED
        elif ratio >= 0.5: return ThrottleStatus.WARNING
        elif ratio >= 0.2: return ThrottleStatus.CRITICAL
        else: return ThrottleStatus.THROTTLED
    
    async def acquire(self, tokens: int = 1, priority: RequestPriority = RequestPriority.NORMAL, timeout: Optional[float] = None) -> AcquireResult:
        async with self._lock:
            self._refill_tokens()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return AcquireResult(acquired=True, wait_time=0.0, tokens_remaining=self._tokens, throttle_status=self._get_throttle_status())
            rate = self._adaptive_rate or self._tokens_per_second
            rate *= self.config.safety_factor
            wait_time = (tokens - self._tokens) / rate
            if timeout is not None and wait_time > timeout:
                return AcquireResult(acquired=False, wait_time=wait_time, tokens_remaining=self._tokens, throttle_status=ThrottleStatus.THROTTLED, message=f"Wait time exceeds timeout")
        await asyncio.sleep(wait_time)
        async with self._lock:
            self._refill_tokens()
            self._tokens = max(0, self._tokens - tokens)
            return AcquireResult(acquired=True, wait_time=wait_time, tokens_remaining=self._tokens, throttle_status=self._get_throttle_status())
    
    def update_from_headers(self, headers: Dict[str, str]) -> None:
        if self.config.strategy != RateLimitStrategy.ADAPTIVE: return
        try:
            remaining, limit = None, None
            for key, value in headers.items():
                key_lower = key.lower()
                if 'ratelimit-remaining' in key_lower: remaining = int(value)
                elif 'ratelimit-limit' in key_lower: limit = int(value)
                elif key_lower == 'retry-after':
                    self._tokens = 0
                    self._adaptive_rate = self._tokens_per_second * 0.5
                    return
            if remaining is not None and limit is not None:
                ratio = remaining / limit
                if ratio < 0.1: self._adaptive_rate = self._tokens_per_second * 0.3
                elif ratio < 0.3: self._adaptive_rate = self._tokens_per_second * 0.5
                elif ratio < 0.5: self._adaptive_rate = self._tokens_per_second * 0.7
                elif ratio > 0.8: self._adaptive_rate = min(self._tokens_per_second * 1.1, limit / 60.0)
                else: self._adaptive_rate = self._tokens_per_second
        except: pass
    
    def reset(self) -> None:
        self._tokens = self._max_tokens
        self._adaptive_rate = None

class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.venue = config.venue
        self._state = CircuitState.CLOSED
        self._failures: List[datetime] = []
        self._half_open_calls = 0
        self._half_open_successes = 0
        self._last_failure_time: Optional[datetime] = None
    
    @property
    def state(self) -> CircuitState:
        return self._state
    
    def _clean_old_failures(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.config.failure_window_seconds)
        self._failures = [f for f in self._failures if f > cutoff]
    
    def _transition_to(self, new_state: CircuitState) -> None:
        old_state = self._state
        self._state = new_state
        logger.info(f"[{self.venue}] Circuit breaker: {old_state.name} -> {new_state.name}")
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._half_open_successes = 0
        elif new_state == CircuitState.CLOSED:
            self._failures.clear()
    
    def can_proceed(self) -> bool:
        if self._state == CircuitState.CLOSED: return True
        if self._state == CircuitState.OPEN:
            if self._last_failure_time:
                elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
                if elapsed >= self.config.recovery_timeout:
                    self._transition_to(CircuitState.HALF_OPEN)
                    return True
            return False
        return self._half_open_calls < self.config.half_open_max_calls
    
    def record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            self._half_open_successes += 1
            if self._half_open_successes >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
        elif self._state == CircuitState.CLOSED and self._failures:
            self._failures.pop(0)
    
    def record_failure(self, exception: Optional[Exception] = None) -> None:
        now = datetime.now(timezone.utc)
        self._failures.append(now)
        self._last_failure_time = now
        self._clean_old_failures()
        if self._state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.CLOSED and len(self._failures) >= self.config.failure_threshold:
            self._transition_to(CircuitState.OPEN)
    
    def reset(self) -> None:
        self._state = CircuitState.CLOSED
        self._failures.clear()
        self._half_open_calls = 0
        self._half_open_successes = 0
        self._last_failure_time = None

class RetryHandler:
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def categorize_failure(self, exception: Exception) -> FailureCategory:
        if isinstance(exception, asyncio.TimeoutError): return FailureCategory.TIMEOUT
        if isinstance(exception, (ConnectionError, ConnectionResetError)): return FailureCategory.NETWORK
        if hasattr(exception, 'status'):
            status = exception.status
            if status == 429: return FailureCategory.RATE_LIMIT
            elif status == 401: return FailureCategory.AUTHENTICATION
            elif status in self.config.no_retry_status_codes: return FailureCategory.CLIENT_ERROR
            elif status in self.config.retry_status_codes: return FailureCategory.SERVER_ERROR
        return FailureCategory.UNKNOWN
    
    def should_retry(self, exception: Exception, attempt: int, priority: RequestPriority = RequestPriority.NORMAL) -> Tuple[bool, FailureCategory]:
        category = self.categorize_failure(exception)
        if not category.should_retry: return False, category
        if attempt >= self.config.max_retries: return False, category
        return True, category
    
    def calculate_delay(self, attempt: int, category: FailureCategory, last_delay: float = 0.0) -> float:
        base_delay = self.config.base_delay * category.base_delay_modifier
        delay = self.config.strategy.calculate_delay(attempt, base_delay, self.config.max_delay, last_delay)
        jitter_range = delay * self.config.jitter_factor
        return max(0.1, delay + random.uniform(-jitter_range, jitter_range))
    
    async def execute(self, func: Callable[..., Coroutine[Any, Any, T]], *args, circuit_breaker: Optional[CircuitBreaker] = None, priority: RequestPriority = RequestPriority.NORMAL, **kwargs) -> ExecutionResult:
        last_delay = 0.0
        total_delay = 0.0
        last_exception: Optional[Exception] = None
        last_category: Optional[FailureCategory] = None
        for attempt in range(self.config.max_retries + 1):
            if circuit_breaker and not circuit_breaker.can_proceed():
                return ExecutionResult(success=False, error=RuntimeError(f"Circuit breaker open"), attempts=attempt, total_delay=total_delay, failure_category=FailureCategory.TRANSIENT)
            try:
                result = await func(*args, **kwargs)
                if circuit_breaker: circuit_breaker.record_success()
                return ExecutionResult(success=True, result=result, attempts=attempt + 1, total_delay=total_delay)
            except Exception as e:
                last_exception = e
                if circuit_breaker: circuit_breaker.record_failure(e)
                should_retry, category = self.should_retry(e, attempt, priority)
                last_category = category
                if not should_retry: break
                delay = self.calculate_delay(attempt, category, last_delay)
                last_delay = delay
                total_delay += delay
                logger.warning(f"Retry {attempt + 1}/{self.config.max_retries}: {category.name} - waiting {delay:.2f}s")
                await asyncio.sleep(delay)
        return ExecutionResult(success=False, error=last_exception, attempts=self.config.max_retries + 1, total_delay=total_delay, failure_category=last_category)

class NetworkManager:
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        self.retry_config = retry_config or RetryConfig()
        self.retry_handler = RetryHandler(self.retry_config)
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def register_venue(self, venue: str, requests_per_minute: int = 60, burst_capacity: Optional[int] = None, strategy: RateLimitStrategy = RateLimitStrategy.ADAPTIVE, failure_threshold: int = 5, recovery_timeout: float = 30.0) -> None:
        rate_config = RateLimitConfig(venue=venue, requests_per_minute=requests_per_minute, burst_capacity=burst_capacity, strategy=strategy)
        self.rate_limiters[venue] = RateLimiter(rate_config)
        cb_config = CircuitBreakerConfig(venue=venue, failure_threshold=failure_threshold, recovery_timeout=recovery_timeout)
        self.circuit_breakers[venue] = CircuitBreaker(cb_config)
    
    async def execute(self, venue: str, func: Callable[..., Coroutine[Any, Any, T]], *args, priority: RequestPriority = RequestPriority.NORMAL, timeout: Optional[float] = None, **kwargs) -> ExecutionResult:
        rate_limit_wait = 0.0
        if venue in self.rate_limiters:
            result = await self.rate_limiters[venue].acquire(priority=priority, timeout=timeout)
            if not result.acquired:
                return ExecutionResult(success=False, error=RuntimeError(f"Rate limit timeout for {venue}"), rate_limit_wait=result.wait_time, failure_category=FailureCategory.RATE_LIMIT)
            rate_limit_wait = result.wait_time
        circuit_breaker = self.circuit_breakers.get(venue)
        exec_result = await self.retry_handler.execute(func, *args, circuit_breaker=circuit_breaker, priority=priority, **kwargs)
        exec_result.rate_limit_wait = rate_limit_wait
        return exec_result
    
    def update_from_headers(self, venue: str, headers: Dict[str, str]) -> None:
        if venue in self.rate_limiters: self.rate_limiters[venue].update_from_headers(headers)
    
    def get_venue_health(self, venue: str) -> Dict[str, Any]:
        result = {'venue': venue, 'available': True}
        if venue in self.rate_limiters:
            limiter = self.rate_limiters[venue]
            result['rate_limiter'] = {'tokens': limiter._tokens, 'max_tokens': limiter._max_tokens}
        if venue in self.circuit_breakers:
            breaker = self.circuit_breakers[venue]
            result['circuit_breaker'] = {'state': breaker.state.name, 'failures': len(breaker._failures)}
            if breaker.state == CircuitState.OPEN: result['available'] = False
        return result
    
    def reset_venue(self, venue: str) -> None:
        if venue in self.rate_limiters: self.rate_limiters[venue].reset()
        if venue in self.circuit_breakers: self.circuit_breakers[venue].reset()
    
    def reset_all(self) -> None:
        for limiter in self.rate_limiters.values(): limiter.reset()
        for breaker in self.circuit_breakers.values(): breaker.reset()

def create_network_manager() -> NetworkManager:
    manager = NetworkManager()
    # CEX
    manager.register_venue('binance', requests_per_minute=1200)
    manager.register_venue('bybit', requests_per_minute=120)
    manager.register_venue('okx', requests_per_minute=60)
    manager.register_venue('coinbase', requests_per_minute=600)
    manager.register_venue('kraken', requests_per_minute=900)
    manager.register_venue('cme', requests_per_minute=60)
    # Hybrid
    manager.register_venue('hyperliquid', requests_per_minute=100)
    manager.register_venue('dydx', requests_per_minute=100)
    # Options
    manager.register_venue('deribit', requests_per_minute=1200)
    manager.register_venue('aevo', requests_per_minute=60)
    manager.register_venue('lyra', requests_per_minute=60)
    manager.register_venue('dopex', requests_per_minute=60)
    # DEX
    for venue in ['uniswap', 'sushiswap', 'curve', 'geckoterminal', 'dexscreener', 'gmx', 'vertex', 'jupiter', 'cowswap', 'oneinch', 'zerox']:
        manager.register_venue(venue, requests_per_minute=60)
    # Indexers
    manager.register_venue('thegraph', requests_per_minute=1000)
    # Market data
    manager.register_venue('coingecko', requests_per_minute=30)
    manager.register_venue('cryptocompare', requests_per_minute=100)
    manager.register_venue('messari', requests_per_minute=60)
    manager.register_venue('kaiko', requests_per_minute=60)
    # On-chain
    for venue in ['glassnode', 'santiment', 'cryptoquant', 'coinmetrics', 'nansen', 'arkham', 'flipside', 'covalent', 'bitquery', 'whale_alert']:
        manager.register_venue(venue, requests_per_minute=60)
    # Alternative
    for venue in ['defillama', 'coinalyze', 'lunarcrush', 'dune']:
        manager.register_venue(venue, requests_per_minute=60)
    return manager

def with_retry(max_retries: int = 3, base_delay: float = 1.0, strategy: RetryStrategy = RetryStrategy.EXPONENTIAL, circuit_breaker: Optional[CircuitBreaker] = None):
    config = RetryConfig(max_retries=max_retries, base_delay=base_delay, strategy=strategy)
    handler = RetryHandler(config)
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await handler.execute(func, *args, circuit_breaker=circuit_breaker, **kwargs)
            if result.success: return result.result
            raise result.error
        return wrapper
    return decorator

def with_rate_limit(venue: str, requests_per_minute: int = 60, manager: Optional[NetworkManager] = None):
    _manager = manager or NetworkManager()
    if venue not in _manager.rate_limiters:
        _manager.register_venue(venue, requests_per_minute=requests_per_minute)
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            await _manager.rate_limiters[venue].acquire()
            return await func(*args, **kwargs)
        return wrapper
    return decorator

__all__ = [
    'VenueTier', 'RateLimitStrategy', 'ThrottleStatus', 'RequestPriority',
    'RetryStrategy', 'FailureCategory', 'CircuitState', 'RecoveryAction',
    'RateLimitConfig', 'CircuitBreakerConfig', 'RetryConfig',
    'AcquireResult', 'ExecutionResult',
    'RateLimiter', 'CircuitBreaker', 'RetryHandler', 'NetworkManager',
    'create_network_manager', 'with_retry', 'with_rate_limit',
]
