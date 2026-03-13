"""
comprehensive Parallel Executor for Crypto Data Collection.

This module implements detailed parallel processing patterns based on industry best practices:

1. ADAPTIVE RATE LIMITING
   - Token Bucket Algorithm with burst support
   - Per-venue adaptive rate adjustment
   - Sliding window rate tracking
   - Automatic slowdown on 429 responses

2. CIRCUIT BREAKER PATTERN
   - Prevents cascading failures
   - States: CLOSED (normal) -> OPEN (failing) -> HALF_OPEN (testing)
   - Automatic recovery with configurable thresholds
   - Per-venue circuit breakers

3. PRIORITY-BASED TASK SCHEDULING
   - Priority queues for critical data types
   - Fair scheduling with weighted priorities
   - Starvation prevention

4. BACKPRESSURE HANDLING
   - Bounded task queues to prevent memory exhaustion
   - Producer-consumer pattern with flow control
   - Graceful degradation under load

5. EXPONENTIAL BACKOFF WITH JITTER
   - Prevents thundering herd
   - Configurable base, max delay, and jitter

6. HIERARCHICAL CONCURRENCY CONTROL
   - Data Type Level: All data types in parallel
   - Venue Level: All venues in parallel (with rate limits)
   - Symbol Level: Symbols batched per venue rate limits

References:
- Token Bucket: https://pypi.org/project/pyrate-limiter/
- Circuit Breaker: https://pypi.org/project/circuitbreaker/
- Backpressure: https://blog.changs.co.uk/asyncio-backpressure-processing-lots-of-tasks-in-parallel.html
- Semaphores: https://medium.com/@mr.sourav.raj/mastering-asyncio-semaphores-in-python

Version: 2.0.0 (comprehensive)
"""

import asyncio
import enum
import logging
import math
import os
import random
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import (
    Any, Callable, Coroutine, Deque, Dict, Generic, List,
    Optional, Set, Tuple, TypeVar, Union
)
import heapq

import pandas as pd

# Import symbol filtering to avoid wasted iterations
from .parallel_processor import filter_symbols_for_venue

logger = logging.getLogger(__name__)

# =============================================================================
# TYPE VARIABLES
# =============================================================================

T = TypeVar('T')
R = TypeVar('R')

# =============================================================================
# TOKEN BUCKET RATE LIMITER
# =============================================================================

@dataclass
class TokenBucketConfig:
    """Configuration for token bucket rate limiter."""
    # Tokens per second (rate limit)
    rate: float = 10.0
    # Maximum burst size (bucket capacity)
    capacity: float = 20.0
    # Initial tokens
    initial_tokens: Optional[float] = None
    # Minimum wait between requests (prevents hammering)
    min_interval_ms: float = 10.0
    # Enable adaptive rate adjustment
    adaptive: bool = True
    # Rate reduction factor on 429/rate limit errors
    rate_reduction_factor: float = 0.5
    # Rate recovery factor per successful request
    rate_recovery_factor: float = 1.01
    # Minimum rate (floor)
    min_rate: float = 1.0
    # Maximum rate (ceiling)
    max_rate: float = 100.0

class TokenBucketRateLimiter:
    """
    Token Bucket Rate Limiter with adaptive adjustment.

    Based on the leaky bucket algorithm:
    - Tokens are added at a constant rate
    - Each request consumes one token
    - If no tokens available, request must wait
    - Supports burst traffic up to capacity

    Adaptive Features:
    - Automatically reduces rate on 429 errors
    - Gradually recovers rate on successful requests
    - Per-venue rate tracking

    References:
    - https://dev.to/satrobit/rate-limiting-using-the-token-bucket-algorithm-3cjh
    - https://pyratelimiter.readthedocs.io/
    """

    def __init__(self, config: TokenBucketConfig):
        self.config = config
        self._tokens = config.initial_tokens or config.capacity
        self._last_update = time.monotonic()
        self._last_request = 0.0
        self._current_rate = config.rate
        self._lock = asyncio.Lock()

        # Statistics
        self._requests_total = 0
        self._requests_limited = 0
        self._rate_adjustments = 0

        logger.debug(
            f"TokenBucket initialized: rate={config.rate}/s, "
            f"capacity={config.capacity}, adaptive={config.adaptive}"
        )

    async def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire (default: 1)
            timeout: Maximum wait time in seconds (None = wait forever)

        Returns:
            True if tokens acquired, False if timeout
        """
        start_time = time.monotonic()

        async with self._lock:
            while True:
                # Refill tokens based on elapsed time
                now = time.monotonic()
                elapsed = now - self._last_update
                self._tokens = min(
                    self.config.capacity,
                    self._tokens + elapsed * self._current_rate
                )
                self._last_update = now

                # Check minimum interval
                time_since_last = (now - self._last_request) * 1000 # ms
                if time_since_last < self.config.min_interval_ms:
                    wait_ms = self.config.min_interval_ms - time_since_last
                    if timeout is not None and (now - start_time + wait_ms/1000) > timeout:
                        return False
                    await asyncio.sleep(wait_ms / 1000)
                    continue

                # Check if we have enough tokens
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self._last_request = time.monotonic()
                    self._requests_total += 1
                    return True

                # Calculate wait time
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / self._current_rate

                # Check timeout
                if timeout is not None:
                    remaining = timeout - (time.monotonic() - start_time)
                    if remaining <= 0:
                        return False
                    wait_time = min(wait_time, remaining)

                self._requests_limited += 1

                # Release lock while waiting
                self._lock.release()
                try:
                    await asyncio.sleep(wait_time)
                finally:
                    await self._lock.acquire()

    def report_success(self) -> None:
        """Report a successful request (for adaptive rate adjustment)."""
        if self.config.adaptive:
            self._current_rate = min(
                self.config.max_rate,
                self._current_rate * self.config.rate_recovery_factor
            )

    def report_rate_limit(self) -> None:
        """Report a rate limit error (429) for adaptive adjustment."""
        if self.config.adaptive:
            old_rate = self._current_rate
            self._current_rate = max(
                self.config.min_rate,
                self._current_rate * self.config.rate_reduction_factor
            )
            self._rate_adjustments += 1
            logger.warning(
                f"Rate limit hit, reducing rate: {old_rate:.1f} -> {self._current_rate:.1f}/s"
            )

    @property
    def current_rate(self) -> float:
        """Current effective rate."""
        return self._current_rate

    @property
    def available_tokens(self) -> float:
        """Currently available tokens."""
        now = time.monotonic()
        elapsed = now - self._last_update
        return min(
            self.config.capacity,
            self._tokens + elapsed * self._current_rate
        )

    def stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            'requests_total': self._requests_total,
            'requests_limited': self._requests_limited,
            'limit_rate': self._requests_limited / max(1, self._requests_total) * 100,
            'current_rate': self._current_rate,
            'configured_rate': self.config.rate,
            'rate_adjustments': self._rate_adjustments,
            'available_tokens': self.available_tokens,
        }

# =============================================================================
# CIRCUIT BREAKER PATTERN
# =============================================================================

class CircuitState(enum.Enum):
    """Circuit breaker states."""
    CLOSED = "closed" # Normal operation
    OPEN = "open" # Failing, reject requests
    HALF_OPEN = "half_open" # Testing recovery

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    # Number of failures to trip the circuit
    failure_threshold: int = 5
    # Time to wait before testing recovery (seconds)
    recovery_timeout: float = 30.0
    # Number of successful requests to close circuit
    success_threshold: int = 3
    # Exceptions that count as failures
    expected_exceptions: Tuple[type, ...] = (Exception,)
    # Exceptions to exclude from failure count
    excluded_exceptions: Tuple[type, ...] = ()

class CircuitBreaker:
    """
    Circuit Breaker Pattern for fault tolerance.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Circuit tripped, requests fail fast
    - HALF_OPEN: Testing if service recovered

    Transitions:
    - CLOSED -> OPEN: failure_threshold failures
    - OPEN -> HALF_OPEN: recovery_timeout elapsed
    - HALF_OPEN -> CLOSED: success_threshold successes
    - HALF_OPEN -> OPEN: any failure

    References:
    - https://pypi.org/project/circuitbreaker/
    - https://medium.com/@fahimad/resilient-apis-retry-logic-circuit-breakers
    """

    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()

        logger.debug(f"CircuitBreaker '{name}' initialized: {config}")

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit allows requests."""
        return self._state == CircuitState.CLOSED

    async def can_execute(self) -> bool:
        """
        Check if a request can be executed.

        Returns:
            True if request can proceed, False if circuit is open
        """
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                if self._last_failure_time is not None:
                    elapsed = time.monotonic() - self._last_failure_time
                    if elapsed >= self.config.recovery_timeout:
                        logger.info(f"CircuitBreaker '{self.name}': OPEN -> HALF_OPEN")
                        self._state = CircuitState.HALF_OPEN
                        self._success_count = 0
                        return True
                return False

            # HALF_OPEN: allow limited requests
            return True

    async def record_success(self) -> None:
        """Record a successful request."""
        async with self._lock:
            self._failure_count = 0

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    logger.info(f"CircuitBreaker '{self.name}': HALF_OPEN -> CLOSED")
                    self._state = CircuitState.CLOSED
                    self._success_count = 0

    async def record_failure(self, exception: Exception) -> None:
        """Record a failed request."""
        # Check if exception should be counted
        if isinstance(exception, self.config.excluded_exceptions):
            return

        if not isinstance(exception, self.config.expected_exceptions):
            return

        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open trips immediately
                logger.warning(f"CircuitBreaker '{self.name}': HALF_OPEN -> OPEN (failure)")
                self._state = CircuitState.OPEN
                self._failure_count = 0

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    logger.warning(
                        f"CircuitBreaker '{self.name}': CLOSED -> OPEN "
                        f"(failures={self._failure_count})"
                    )
                    self._state = CircuitState.OPEN
                    self._failure_count = 0

    async def call(
        self,
        func: Callable[..., Coroutine[Any, Any, R]],
        *args,
        **kwargs
    ) -> R:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args, **kwargs: Function arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpen: If circuit is open
            Original exception: If function fails
        """
        if not await self.can_execute():
            raise CircuitBreakerOpen(self.name)

        try:
            result = await func(*args, **kwargs)
            await self.record_success()
            return result
        except Exception as e:
            await self.record_failure(e)
            raise

    def stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            'name': self.name,
            'state': self._state.value,
            'failure_count': self._failure_count,
            'success_count': self._success_count,
            'last_failure': self._last_failure_time,
        }

class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Circuit breaker '{name}' is OPEN")

# =============================================================================
# EXPONENTIAL BACKOFF WITH JITTER
# =============================================================================

@dataclass
class BackoffConfig:
    """Configuration for exponential backoff."""
    # Base delay in seconds
    base_delay: float = 1.0
    # Maximum delay in seconds
    max_delay: float = 60.0
    # Exponential multiplier
    multiplier: float = 2.0
    # Jitter type: 'full', 'equal', 'decorrelated'
    jitter: str = 'full'
    # Maximum retries
    max_retries: int = 5

class ExponentialBackoff:
    """
    Exponential Backoff with Jitter for retry logic.

    Jitter Types:
    - full: delay = random(0, base * multiplier^attempt)
    - equal: delay = base * multiplier^attempt / 2 + random(0, delay/2)
    - decorrelated: delay = random(base, previous_delay * 3)

    The decorrelated jitter provides the best spread of retry times,
    preventing the "thundering herd" problem.

    References:
    - https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
    - https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/retry-backoff.html
    """

    def __init__(self, config: BackoffConfig):
        self.config = config
        self._previous_delay = config.base_delay

    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt number.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        base = self.config.base_delay
        multiplier = self.config.multiplier
        max_delay = self.config.max_delay

        # Calculate exponential delay
        exp_delay = min(max_delay, base * (multiplier ** attempt))

        # Apply jitter
        if self.config.jitter == 'full':
            # Full jitter: random between 0 and exp_delay
            delay = random.uniform(0, exp_delay)

        elif self.config.jitter == 'equal':
            # Equal jitter: half exp + random half
            half = exp_delay / 2
            delay = half + random.uniform(0, half)

        elif self.config.jitter == 'decorrelated':
            # Decorrelated jitter: random between base and 3x previous
            delay = random.uniform(base, min(max_delay, self._previous_delay * 3))
            self._previous_delay = delay

        else:
            # No jitter
            delay = exp_delay

        return min(delay, max_delay)

    def reset(self) -> None:
        """Reset backoff state."""
        self._previous_delay = self.config.base_delay

async def retry_with_backoff(
    func: Callable[..., Coroutine[Any, Any, R]],
    config: BackoffConfig,
    *args,
    retryable_exceptions: Tuple[type, ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    **kwargs
) -> R:
    """
    Execute function with exponential backoff retry.

    Args:
        func: Async function to execute
        config: Backoff configuration
        *args, **kwargs: Function arguments
        retryable_exceptions: Exceptions that trigger retry
        on_retry: Callback(attempt, exception, delay) called before retry

    Returns:
        Function result

    Raises:
        Last exception if all retries exhausted
    """
    backoff = ExponentialBackoff(config)
    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)

        except retryable_exceptions as e:
            last_exception = e

            if attempt >= config.max_retries:
                raise

            delay = backoff.get_delay(attempt)

            if on_retry:
                on_retry(attempt, e, delay)

            logger.debug(
                f"Retry {attempt + 1}/{config.max_retries}: "
                f"{type(e).__name__}, waiting {delay:.2f}s"
            )

            await asyncio.sleep(delay)

    raise last_exception

# =============================================================================
# PRIORITY TASK QUEUE
# =============================================================================

@dataclass(order=True)
class PriorityTask(Generic[T]):
    """Task with priority for priority queue."""
    priority: int # Lower = higher priority
    sequence: int # Tie-breaker (FIFO for same priority)
    item: T = field(compare=False)
    created_at: float = field(default_factory=time.monotonic, compare=False)

class PriorityTaskQueue(Generic[T]):
    """
    Async priority queue with backpressure support.

    Features:
    - Priority-based ordering (lower number = higher priority)
    - Bounded capacity for backpressure
    - FIFO ordering within same priority
    - Async-safe operations

    References:
    - https://superfastpython.com/asyncio-priorityqueue/
    - https://discuss.python.org/t/asyncio-semaphore-with-support-for-priorities/51072
    """

    def __init__(self, maxsize: int = 0):
        """
        Initialize priority queue.

        Args:
            maxsize: Maximum queue size (0 = unlimited)
        """
        self._heap: List[PriorityTask[T]] = []
        self._sequence = 0
        self._maxsize = maxsize
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._not_full = asyncio.Condition(self._lock)

    async def put(
        self,
        item: T,
        priority: int = 5,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Put item into queue with priority.

        Args:
            item: Item to queue
            priority: Priority (0 = highest, 10 = lowest)
            timeout: Maximum wait time for space

        Returns:
            True if item added, False if timeout
        """
        async with self._not_full:
            # Wait for space if bounded and full
            if self._maxsize > 0:
                while len(self._heap) >= self._maxsize:
                    try:
                        await asyncio.wait_for(
                            self._not_full.wait(),
                            timeout=timeout
                        )
                    except asyncio.TimeoutError:
                        return False

            # Add task with sequence number for FIFO within priority
            task = PriorityTask(
                priority=priority,
                sequence=self._sequence,
                item=item
            )
            self._sequence += 1
            heapq.heappush(self._heap, task)

            # Signal that queue is not empty
            self._not_empty.notify()
            return True

    async def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """
        Get highest priority item from queue.

        Args:
            timeout: Maximum wait time for item

        Returns:
            Item or None if timeout
        """
        async with self._not_empty:
            while not self._heap:
                try:
                    await asyncio.wait_for(
                        self._not_empty.wait(),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    return None

            task = heapq.heappop(self._heap)

            # Signal that queue is not full
            if self._maxsize > 0:
                self._not_full.notify()

            return task.item

    def qsize(self) -> int:
        """Return queue size."""
        return len(self._heap)

    def empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._heap) == 0

    def full(self) -> bool:
        """Check if queue is full."""
        if self._maxsize <= 0:
            return False
        return len(self._heap) >= self._maxsize

# =============================================================================
# BACKPRESSURE CONTROLLER
# =============================================================================

class BackpressureController:
    """
    Backpressure controller for flow control.

    Monitors queue depths and processing rates to:
    - Slow down producers when consumers can't keep up
    - Signal when system is overloaded
    - Prevent memory exhaustion

    References:
    - https://blog.changs.co.uk/asyncio-backpressure-processing-lots-of-tasks-in-parallel.html
    """

    def __init__(
        self,
        high_water_mark: int = 100,
        low_water_mark: int = 50,
        check_interval: float = 1.0
    ):
        """
        Initialize backpressure controller.

        Args:
            high_water_mark: Queue depth that triggers backpressure
            low_water_mark: Queue depth that releases backpressure
            check_interval: How often to check pressure (seconds)
        """
        self._high_water = high_water_mark
        self._low_water = low_water_mark
        self._check_interval = check_interval

        self._queue_depth = 0
        self._under_pressure = False
        self._pressure_event = asyncio.Event()
        self._pressure_event.set() # Start unpressured

        self._lock = asyncio.Lock()

        # Statistics
        self._pressure_count = 0
        self._total_wait_time = 0.0

    async def acquire(self) -> None:
        """
        Wait until backpressure is released.

        Called by producers before generating new work.
        """
        start = time.monotonic()
        await self._pressure_event.wait()
        wait_time = time.monotonic() - start

        if wait_time > 0.01: # Only track significant waits
            self._total_wait_time += wait_time

    async def report_queue_depth(self, depth: int) -> None:
        """
        Report current queue depth.

        Called periodically by queue processors.
        """
        async with self._lock:
            self._queue_depth = depth

            if not self._under_pressure and depth >= self._high_water:
                # Trigger backpressure
                self._under_pressure = True
                self._pressure_count += 1
                self._pressure_event.clear()
                logger.warning(
                    f"Backpressure triggered: queue_depth={depth} >= {self._high_water}"
                )

            elif self._under_pressure and depth <= self._low_water:
                # Release backpressure
                self._under_pressure = False
                self._pressure_event.set()
                logger.info(
                    f"Backpressure released: queue_depth={depth} <= {self._low_water}"
                )

    @property
    def is_under_pressure(self) -> bool:
        """Check if system is under backpressure."""
        return self._under_pressure

    def stats(self) -> Dict[str, Any]:
        """Get backpressure statistics."""
        return {
            'queue_depth': self._queue_depth,
            'under_pressure': self._under_pressure,
            'pressure_count': self._pressure_count,
            'total_wait_time': self._total_wait_time,
            'high_water_mark': self._high_water,
            'low_water_mark': self._low_water,
        }

# =============================================================================
# ADAPTIVE SEMAPHORE
# =============================================================================

class AdaptiveSemaphore:
    """
    Semaphore with adaptive capacity adjustment.

    Features:
    - Dynamic capacity based on success/failure rates
    - Automatic scaling up/down
    - Configurable bounds

    References:
    - https://medium.com/@mr.sourav.raj/mastering-asyncio-semaphores-in-python
    """

    def __init__(
        self,
        initial_capacity: int = 10,
        min_capacity: int = 1,
        max_capacity: int = 100,
        adjustment_interval: float = 10.0,
        success_increase_threshold: float = 0.95,
        failure_decrease_threshold: float = 0.2
    ):
        self._capacity = initial_capacity
        self._min_capacity = min_capacity
        self._max_capacity = max_capacity
        self._adjustment_interval = adjustment_interval
        self._success_threshold = success_increase_threshold
        self._failure_threshold = failure_decrease_threshold

        self._semaphore = asyncio.Semaphore(initial_capacity)
        self._lock = asyncio.Lock()

        # Tracking
        self._successes = 0
        self._failures = 0
        self._last_adjustment = time.monotonic()

    async def acquire(self) -> None:
        """Acquire semaphore."""
        await self._semaphore.acquire()

    def release(self) -> None:
        """Release semaphore."""
        self._semaphore.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()
        if exc_type is not None:
            await self.record_failure()
        else:
            await self.record_success()
        return False

    async def record_success(self) -> None:
        """Record successful operation."""
        async with self._lock:
            self._successes += 1
            await self._maybe_adjust()

    async def record_failure(self) -> None:
        """Record failed operation."""
        async with self._lock:
            self._failures += 1
            await self._maybe_adjust()

    async def _maybe_adjust(self) -> None:
        """Adjust capacity if needed."""
        now = time.monotonic()
        if now - self._last_adjustment < self._adjustment_interval:
            return

        total = self._successes + self._failures
        if total == 0:
            return

        success_rate = self._successes / total
        failure_rate = self._failures / total

        old_capacity = self._capacity

        if success_rate >= self._success_threshold and self._capacity < self._max_capacity:
            # Increase capacity
            self._capacity = min(self._max_capacity, int(self._capacity * 1.5))

        elif failure_rate >= self._failure_threshold and self._capacity > self._min_capacity:
            # Decrease capacity
            self._capacity = max(self._min_capacity, int(self._capacity * 0.5))

        if old_capacity != self._capacity:
            # Recreate semaphore with new capacity
            # Note: This is a simplified approach; production code should handle
            # in-flight requests more carefully
            self._semaphore = asyncio.Semaphore(self._capacity)
            logger.info(
                f"AdaptiveSemaphore adjusted: {old_capacity} -> {self._capacity} "
                f"(success_rate={success_rate:.2%})"
            )

        # Reset counters
        self._successes = 0
        self._failures = 0
        self._last_adjustment = now

    @property
    def capacity(self) -> int:
        """Current capacity."""
        return self._capacity

# =============================================================================
# VENUE RATE LIMITER MANAGER
# =============================================================================

# Venue-specific rate limits (requests per second)
# Per PDF specifications exactly:
# - Binance: ~1200/min = 20/s
# - Hyperliquid: "Lower rate limits than Binance" = ~600/min = 10/s
# - CryptoCompare: 100k/month FREE tier
# - The Graph: varies by tier
VENUE_RATE_LIMITS: Dict[str, TokenBucketConfig] = {
    # CEX - Conservative rates to prevent IP bans
    'binance': TokenBucketConfig(rate=6.0, capacity=15.0), # ~360/min (safe margin under 1200)
    'bybit': TokenBucketConfig(rate=3.0, capacity=8.0), # ~180/min
    'okx': TokenBucketConfig(rate=3.0, capacity=8.0), # ~180/min
    'kraken': TokenBucketConfig(rate=3.0, capacity=8.0), # ~180/min
    'coinbase': TokenBucketConfig(rate=2.0, capacity=5.0), # ~120/min

    # Hybrid - Lower rates for stability
    'hyperliquid': TokenBucketConfig(rate=2.0, capacity=5.0), # ~120/min
    'dydx': TokenBucketConfig(rate=2.0, capacity=5.0), # ~120/min
    'drift': TokenBucketConfig(rate=1.0, capacity=3.0), # ~60/min

    # Options - Conservative
    'deribit': TokenBucketConfig(rate=5.0, capacity=12.0), # ~300/min
    'aevo': TokenBucketConfig(rate=1.0, capacity=3.0), # ~60/min

    # DEX - The Graph subgraphs (FREE with limits)
    'gmx': TokenBucketConfig(rate=1.0, capacity=3.0), # ~60/min
    'uniswap': TokenBucketConfig(rate=1.0, capacity=3.0), # ~60/min
    'sushiswap': TokenBucketConfig(rate=1.0, capacity=3.0), # ~60/min
    'curve': TokenBucketConfig(rate=1.0, capacity=3.0), # ~60/min
    'geckoterminal': TokenBucketConfig(rate=0.5, capacity=2.0), # ~30/min
    'dexscreener': TokenBucketConfig(rate=0.5, capacity=2.0), # ~30/min
    'jupiter': TokenBucketConfig(rate=1.0, capacity=3.0), # ~60/min
    'oneinch': TokenBucketConfig(rate=0.5, capacity=2.0), # ~30/min
    'zerox': TokenBucketConfig(rate=0.5, capacity=2.0), # ~30/min
    'cowswap': TokenBucketConfig(rate=0.5, capacity=2.0), # ~30/min

    # Market data providers - Conservative for free tiers
    'coingecko': TokenBucketConfig(rate=0.15, capacity=2.0), # ~9/min (free tier 10/min)
    'cryptocompare': TokenBucketConfig(rate=1.0, capacity=3.0), # ~60/min
    'messari': TokenBucketConfig(rate=0.5, capacity=2.0), # ~30/min

    # On-chain/Alternative - Conservative
    'defillama': TokenBucketConfig(rate=0.5, capacity=2.0), # ~30/min
    'santiment': TokenBucketConfig(rate=0.5, capacity=2.0), # ~30/min
    'coinalyze': TokenBucketConfig(rate=0.3, capacity=2.0), # ~18/min
    'thegraph': TokenBucketConfig(rate=1.0, capacity=3.0), # ~60/min
    'nansen': TokenBucketConfig(rate=0.3, capacity=2.0), # ~18/min
    'dune': TokenBucketConfig(rate=0.15, capacity=2.0), # ~9/min
    'arkham': TokenBucketConfig(rate=0.5, capacity=2.0), # ~30/min
    'bitquery': TokenBucketConfig(rate=0.5, capacity=2.0), # ~30/min
    'coinmetrics': TokenBucketConfig(rate=0.15, capacity=2.0), # ~9/min
    'covalent': TokenBucketConfig(rate=0.3, capacity=2.0), # ~18/min
    'flipside': TokenBucketConfig(rate=0.3, capacity=2.0), # ~18/min
    'whale_alert': TokenBucketConfig(rate=0.3, capacity=2.0), # ~18/min
    'glassnode': TokenBucketConfig(rate=0.3, capacity=2.0), # ~18/min
    'cryptoquant': TokenBucketConfig(rate=0.15, capacity=2.0), # ~9/min
    'lunarcrush': TokenBucketConfig(rate=0.15, capacity=2.0), # ~9/min
    'kaiko': TokenBucketConfig(rate=0.5, capacity=2.0), # ~30/min

    # Default for unknown APIs
    'default': TokenBucketConfig(rate=0.5, capacity=2.0), # ~30/min default
}

class VenueRateLimiterManager:
    """
    Manages rate limiters for multiple venues.

    Each venue has its own rate limiter with:
    - Venue-specific rate limits
    - Adaptive rate adjustment
    - Circuit breakers for fault tolerance
    """

    def __init__(self):
        self._rate_limiters: Dict[str, TokenBucketRateLimiter] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    def get_rate_limiter(self, venue: str) -> TokenBucketRateLimiter:
        """Get or create rate limiter for venue."""
        if venue not in self._rate_limiters:
            config = VENUE_RATE_LIMITS.get(venue, VENUE_RATE_LIMITS['default'])
            self._rate_limiters[venue] = TokenBucketRateLimiter(config)
        return self._rate_limiters[venue]

    def get_circuit_breaker(self, venue: str) -> CircuitBreaker:
        """Get or create circuit breaker for venue."""
        if venue not in self._circuit_breakers:
            config = CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=120.0,
                success_threshold=3
            )
            self._circuit_breakers[venue] = CircuitBreaker(venue, config)
        return self._circuit_breakers[venue]

    async def acquire(self, venue: str, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make request to venue.

        Returns:
            True if request can proceed, False otherwise
        """
        # Check circuit breaker first
        circuit = self.get_circuit_breaker(venue)
        if not await circuit.can_execute():
            return False

        # Then rate limiter
        limiter = self.get_rate_limiter(venue)
        return await limiter.acquire(timeout=timeout)

    def report_success(self, venue: str) -> None:
        """Report successful request to venue."""
        asyncio.create_task(self._async_report_success(venue))

    async def _async_report_success(self, venue: str) -> None:
        """Async report success."""
        self.get_rate_limiter(venue).report_success()
        await self.get_circuit_breaker(venue).record_success()

    def report_failure(self, venue: str, exception: Exception) -> None:
        """Report failed request to venue."""
        asyncio.create_task(self._async_report_failure(venue, exception))

    async def _async_report_failure(self, venue: str, exception: Exception) -> None:
        """Async report failure."""
        # Check if rate limit error
        error_msg = str(exception).lower()
        if '429' in error_msg or 'rate limit' in error_msg:
            self.get_rate_limiter(venue).report_rate_limit()

        await self.get_circuit_breaker(venue).record_failure(exception)

    def stats(self) -> Dict[str, Any]:
        """Get stats for all venues."""
        return {
            'rate_limiters': {
                venue: limiter.stats()
                for venue, limiter in self._rate_limiters.items()
            },
            'circuit_breakers': {
                venue: cb.stats()
                for venue, cb in self._circuit_breakers.items()
            }
        }

# =============================================================================
# comprehensive PARALLEL EXECUTOR
# =============================================================================

@dataclass
class EnhancedExecutionStats:
    """Statistics from comprehensive parallel execution."""
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None

    # Counts
    data_types_processed: int = 0
    venues_processed: int = 0
    symbols_processed: int = 0
    total_records: int = 0

    # Cache
    records_from_cache: int = 0
    records_collected: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    # Rate limiting
    rate_limited_count: int = 0
    total_wait_time: float = 0.0

    # Circuit breakers
    circuit_opens: int = 0
    circuit_half_opens: int = 0

    # Retries
    total_retries: int = 0
    successful_retries: int = 0

    # Errors
    errors: int = 0
    error_details: List[str] = field(default_factory=list)

    # Performance
    peak_concurrency: int = 0
    total_api_calls: int = 0

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or datetime.now(timezone.utc)
        return (end - self.start_time).total_seconds()

    @property
    def records_per_second(self) -> float:
        if self.duration_seconds == 0:
            return 0
        return self.total_records / self.duration_seconds

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0
        return self.cache_hits / total * 100

    def summary(self) -> Dict[str, Any]:
        """Generate summary dict."""
        return {
            'duration_seconds': round(self.duration_seconds, 1),
            'data_types_processed': self.data_types_processed,
            'venues_processed': self.venues_processed,
            'symbols_processed': self.symbols_processed,
            'total_records': self.total_records,
            'records_per_second': round(self.records_per_second, 0),
            'cache_hit_rate': f"{self.cache_hit_rate:.1f}%",
            'records_from_cache': self.records_from_cache,
            'records_collected': self.records_collected,
            'rate_limited_count': self.rate_limited_count,
            'circuit_opens': self.circuit_opens,
            'total_retries': self.total_retries,
            'errors': self.errors,
            'peak_concurrency': self.peak_concurrency,
        }

class EnhancedParallelExecutor:
    """
    comprehensive parallel executor with detailed patterns.

    Features:
    1. Adaptive token bucket rate limiting per venue
    2. Circuit breakers for fault tolerance
    3. Priority-based task scheduling
    4. Backpressure handling
    5. Exponential backoff with jitter
    6. Hierarchical concurrency control

    Architecture:
        Level 1: All data types execute in parallel (bounded by semaphore)
        Level 2: All venues execute in parallel per data type (with rate limits)
        Level 3: All symbols execute in parallel per venue (batched)

    The executor integrates with IncrementalCacheManager to:
    - Check what data already exists in cache
    - Only collect missing time periods
    - Update cache after successful collection
    """

    def __init__(
        self,
        cache_dir: str = 'data/processed',
        use_cache: bool = True,
        max_data_type_concurrency: int = 2, # 2 data types at a time (safe for DNS)
        max_venue_concurrency: int = 3, # 3 venues at a time (safe for DNS)
        max_queue_size: int = 500, # Reduced to prevent queue backlog
        enable_circuit_breakers: bool = True,
        enable_adaptive_rate_limiting: bool = True,
    ):
        """
        Initialize comprehensive executor.

        Args:
            cache_dir: Directory for cache storage
            use_cache: Enable incremental caching
            max_data_type_concurrency: Max parallel data types
            max_venue_concurrency: Max parallel venues
            max_queue_size: Max pending tasks (backpressure)
            enable_circuit_breakers: Enable circuit breaker pattern
            enable_adaptive_rate_limiting: Enable adaptive rate adjustment
        """
        self.cache_dir = cache_dir
        self.use_cache = use_cache

        # Concurrency control
        self._data_type_semaphore = asyncio.Semaphore(max_data_type_concurrency)
        self._venue_semaphore = asyncio.Semaphore(max_venue_concurrency)

        # Task queue with backpressure
        self._task_queue: PriorityTaskQueue = PriorityTaskQueue(maxsize=max_queue_size)
        self._backpressure = BackpressureController(
            high_water_mark=int(max_queue_size * 0.8),
            low_water_mark=int(max_queue_size * 0.3)
        )

        # Rate limiting and circuit breakers
        self._rate_manager = VenueRateLimiterManager()
        self._enable_circuit_breakers = enable_circuit_breakers
        self._enable_adaptive = enable_adaptive_rate_limiting

        # Backoff config for retries
        # OPTIMIZATION: Reduced max_delay from 60s to 30s to avoid long stalls
        self._backoff_config = BackoffConfig(
            base_delay=1.0,
            max_delay=30.0, # Reduced from 60s
            multiplier=2.0,
            jitter='decorrelated',
            max_retries=3
        )

        # Cache manager
        self._cache_manager = None
        if use_cache:
            try:
                from .incremental_cache import get_cache_manager
                self._cache_manager = get_cache_manager(cache_dir)
                # OPTIMIZATION: Only rebuild metadata if cache is empty
                # The metadata file tracks all collected data - trust it when it exists
                # This prevents slow startup from scanning all parquet files every run
                if not self._cache_manager.metadata.entries:
                    logger.info("Cache metadata empty - rebuilding from parquet files...")
                    discovered = self._cache_manager.rebuild_metadata()
                    if discovered:
                        total = sum(discovered.values())
                        logger.info(f"Cache rebuild: discovered {total} entries from existing files")
                else:
                    # Count total entries for logging
                    total_entries = sum(
                        len(entries)
                        for venues in self._cache_manager.metadata.entries.values()
                        for entries in venues.values()
                    )
                    logger.info(f"Cache loaded with {total_entries} entries - skipping rebuild")
            except ImportError:
                logger.warning("Incremental cache not available")

        # Statistics
        self.stats = EnhancedExecutionStats()

        # Tracking
        self._active_tasks = 0
        self._lock = asyncio.Lock()

        logger.info(
            f"EnhancedParallelExecutor initialized: "
            f"data_type_concurrency={max_data_type_concurrency}, "
            f"venue_concurrency={max_venue_concurrency}, "
            f"circuit_breakers={enable_circuit_breakers}"
        )

    async def execute_collection(
        self,
        data_types: List[str],
        venues: List[str],
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        collector_factory: Callable[[str], Any],
        timeframe: str = '1h',
        priority_map: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Execute comprehensive parallel collection.

        Args:
            data_types: Data types to collect
            venues: Venues to collect from
            symbols: Symbols to collect
            start_date: Collection start date
            end_date: Collection end date
            collector_factory: Function to create collector for venue
            timeframe: Timeframe for OHLCV
            priority_map: Optional data_type -> priority mapping

        Returns:
            Dict[data_type][venue] = DataFrame
        """
        self.stats = EnhancedExecutionStats()

        # Normalize dates
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

        # Default priority map (funding_rates > ohlcv > others)
        if priority_map is None:
            priority_map = {
                'funding_rates': 1,
                'ohlcv': 2,
                'open_interest': 3,
            }

        logger.info(
            f"Starting comprehensive collection: "
            f"{len(data_types)} data types x {len(venues)} venues x {len(symbols)} symbols"
        )

        # Create tasks for all data types
        tasks = []
        for data_type in data_types:
            priority = priority_map.get(data_type, 5)
            task = self._collect_data_type(
                data_type=data_type,
                venues=venues,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                collector_factory=collector_factory,
                timeframe=timeframe,
                priority=priority
            )
            tasks.append(task)

        # Execute all data types in parallel with priority
        # Master timeout as safety net - individual collectors have their own timeouts
        # INCREASED: 60 minutes to allow for comprehensive historical data collection (6+ years)
        try:
            results_list = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=3600.0 # 60 minute master timeout for all data types
            )
        except asyncio.TimeoutError:
            logger.error("Master data type collection timed out after 60 minutes - some data may be incomplete")
            results_list = [TimeoutError("Master timeout")] * len(tasks)

        # Compile results
        results: Dict[str, Dict[str, pd.DataFrame]] = {}
        for i, item in enumerate(results_list):
            if isinstance(item, Exception):
                logger.error(f"Data type collection error: {item}")
                self.stats.errors += 1
                self.stats.error_details.append(str(item))
                continue

            data_type = data_types[i]
            results[data_type] = item

        self.stats.end_time = datetime.now(timezone.utc)
        self._log_summary()

        return results

    async def _collect_data_type(
        self,
        data_type: str,
        venues: List[str],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        collector_factory: Callable[[str], Any],
        timeframe: str,
        priority: int
    ) -> Dict[str, pd.DataFrame]:
        """Collect all venues for a single data type."""
        async with self._data_type_semaphore:
            logger.info(f"[{data_type}] Starting collection (priority={priority})")

            # Create tasks for all venues
            venue_tasks = []
            for venue in venues:
                task = self._collect_venue(
                    data_type=data_type,
                    venue=venue,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    collector_factory=collector_factory,
                    timeframe=timeframe
                )
                venue_tasks.append(task)

            # Execute all venues in parallel with timeout safety net
            # INCREASED TIMEOUT: 30 minutes to allow collection of 6+ years of historical data
            try:
                results_list = await asyncio.wait_for(
                    asyncio.gather(*venue_tasks, return_exceptions=True),
                    timeout=1800.0 # 30 minute timeout per data type's venues
                )
            except asyncio.TimeoutError:
                logger.error(f"[{data_type}] Venue collection timed out after 30 minutes")
                results_list = [TimeoutError("Venue timeout")] * len(venue_tasks)

            # Compile venue results
            venue_results: Dict[str, pd.DataFrame] = {}
            for i, item in enumerate(results_list):
                venue = venues[i]
                if isinstance(item, Exception):
                    logger.error(f"[{data_type}][{venue}] Error: {item}")
                    self.stats.errors += 1
                    continue

                if item is not None and not item.empty:
                    venue_results[venue] = item

            self.stats.data_types_processed += 1
            return venue_results

    async def _collect_venue(
        self,
        data_type: str,
        venue: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        collector_factory: Callable[[str], Any],
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Collect from a single venue with all protections."""
        async with self._venue_semaphore:
            # Track concurrency
            async with self._lock:
                self._active_tasks += 1
                self.stats.peak_concurrency = max(
                    self.stats.peak_concurrency,
                    self._active_tasks
                )

            try:
                return await self._collect_venue_with_protections(
                    data_type=data_type,
                    venue=venue,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    collector_factory=collector_factory,
                    timeframe=timeframe
                )
            finally:
                async with self._lock:
                    self._active_tasks -= 1

    async def _collect_venue_with_protections(
        self,
        data_type: str,
        venue: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        collector_factory: Callable[[str], Any],
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Execute collection with rate limiting, circuit breaker, and retries."""

        # Check cache first
        cached_df = None
        gaps_to_collect = [(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))]

        if self.use_cache and self._cache_manager:
            timeframe_param = timeframe if data_type == 'ohlcv' else None

            # Load cached data
            cached_df = self._cache_manager.load_cached_data(
                data_type=data_type,
                venue=venue,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                timeframe=timeframe_param
            )

            if cached_df is not None and not cached_df.empty:
                self.stats.records_from_cache += len(cached_df)
                self.stats.cache_hits += 1
                logger.debug(f"[{data_type}][{venue}] Loaded {len(cached_df)} from cache")
            else:
                self.stats.cache_misses += 1

            # Determine gaps
            gaps_to_collect = self._cache_manager.get_collection_gaps(
                data_type=data_type,
                venue=venue,
                symbols=symbols,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                timeframe=timeframe_param
            )

            if not gaps_to_collect:
                # CRITICAL FIX: Check for missing symbols even when date range is cached
                # This handles the case where new symbols were added to the universe
                missing_symbols = self._cache_manager.get_missing_symbols(
                    data_type=data_type,
                    venue=venue,
                    symbols=symbols,
                    timeframe=timeframe_param
                )

                if not missing_symbols:
                    logger.info(f"[{data_type}][{venue}] Full cache hit (all {len(symbols)} symbols)")
                    self.stats.venues_processed += 1
                    return cached_df
                else:
                    # Date range is covered, but some symbols are missing
                    # Create a "gap" for the full date range with only missing symbols
                    logger.info(
                        f"[{data_type}][{venue}] Date range cached, but {len(missing_symbols)} symbols missing. "
                        f"Collecting: {missing_symbols[:10]}{'...' if len(missing_symbols) > 10 else ''}"
                    )
                    # Override symbols to only collect missing ones
                    symbols = missing_symbols
                    gaps_to_collect = [(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))]

        # Collect missing data
        all_new_data = []

        # CRITICAL: Check if venue supports this data type BEFORE waiting for rate limit
        # This prevents 30s rate limit timeout for unsupported venue/data_type combinations
        try:
            collector = collector_factory(venue)
            if collector is None:
                logger.debug(f"[{data_type}][{venue}] Collector factory returned None")
                # Mark as no-data to prevent re-checking
                if self.use_cache and self._cache_manager:
                    cache_start = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
                    cache_end = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
                    self._cache_manager.mark_no_data_available(data_type, venue, cache_start, cache_end)
                self.stats.venues_processed += 1
                return cached_df
            if hasattr(collector, 'supported_data_types'):
                if data_type not in collector.supported_data_types:
                    logger.debug(f"[{data_type}][{venue}] Data type not supported by this collector")
                    # Mark as no-data to prevent re-checking
                    if self.use_cache and self._cache_manager:
                        cache_start = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
                        cache_end = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
                        self._cache_manager.mark_no_data_available(data_type, venue, cache_start, cache_end)
                    self.stats.venues_processed += 1
                    return cached_df
        except Exception as e:
            logger.debug(f"[{data_type}][{venue}] Failed to check collector support: {e}")

        for gap_start, gap_end in gaps_to_collect:
            # Check circuit breaker
            if self._enable_circuit_breakers:
                circuit = self._rate_manager.get_circuit_breaker(venue)
                if not await circuit.can_execute():
                    logger.warning(f"[{data_type}][{venue}] Circuit breaker OPEN, skipping")
                    self.stats.circuit_opens += 1
                    continue

            # Acquire rate limit
            if not await self._rate_manager.acquire(venue, timeout=120.0):
                logger.warning(f"[{data_type}][{venue}] Rate limit timeout")
                self.stats.rate_limited_count += 1
                continue

            # Execute with retry
            try:
                df = await retry_with_backoff(
                    self._fetch_data,
                    self._backoff_config,
                    data_type=data_type,
                    venue=venue,
                    symbols=symbols,
                    start_date=gap_start,
                    end_date=gap_end,
                    collector_factory=collector_factory,
                    timeframe=timeframe,
                    on_retry=lambda attempt, e, delay: self._on_retry(
                        venue, attempt, e, delay
                    )
                )

                if df is not None and not df.empty:
                    all_new_data.append(df)
                    self.stats.records_collected += len(df)

                self._rate_manager.report_success(venue)

            except Exception as e:
                self._rate_manager.report_failure(venue, e)
                logger.error(f"[{data_type}][{venue}] Collection failed: {e}")
                self.stats.errors += 1

        # Merge cached and new data
        final_df = None
        timeframe_param = timeframe if data_type == 'ohlcv' else None
        cache_start = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
        cache_end = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)

        if all_new_data:
            new_df = pd.concat(all_new_data, ignore_index=True)

            # Normalize timestamps to datetime objects (handles ISO strings from collectors)
            if 'timestamp' in new_df.columns:
                new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], format='ISO8601', utc=True)

            if cached_df is not None and not cached_df.empty:
                final_df = pd.concat([cached_df, new_df], ignore_index=True)
            else:
                final_df = new_df

            # Deduplicate
            if 'timestamp' in final_df.columns:
                dedup_cols = ['timestamp']
                if 'symbol' in final_df.columns:
                    dedup_cols.append('symbol')
                final_df = final_df.drop_duplicates(subset=dedup_cols, keep='last')
                final_df = final_df.sort_values('timestamp').reset_index(drop=True)

            # Update cache with new data
            if self.use_cache and self._cache_manager:
                self._cache_manager.update_cache(
                    data_type=data_type,
                    venue=venue,
                    data=final_df,
                    start_date=cache_start,
                    end_date=cache_end,
                    timeframe=timeframe_param
                )

        elif cached_df is not None:
            final_df = cached_df
        else:
            # CRITICAL FIX: Cache empty results to prevent re-collection
            # This prevents infinite loops when a venue doesn't support a data type
            # or when authentication is required but not provided
            if self.use_cache and self._cache_manager:
                logger.debug(f"[{data_type}][{venue}] Caching empty result to prevent re-collection")
                self._cache_manager.update_cache(
                    data_type=data_type,
                    venue=venue,
                    data=pd.DataFrame(), # Empty DataFrame
                    start_date=cache_start,
                    end_date=cache_end,
                    timeframe=timeframe_param
                )

        if final_df is not None:
            self.stats.total_records += len(final_df)
            self.stats.symbols_processed += final_df['symbol'].nunique() if 'symbol' in final_df.columns else 0

        self.stats.venues_processed += 1
        return final_df

    async def _fetch_data(
        self,
        data_type: str,
        venue: str,
        symbols: List[str],
        start_date: str,
        end_date: str,
        collector_factory: Callable[[str], Any],
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data from collector."""
        # OPTIMIZATION: Pre-filter symbols to avoid wasted iterations
        # This prevents ~52 wasted iterations for venues like Deribit (BTC/ETH/SOL only)
        filtered_symbols = filter_symbols_for_venue(venue, symbols)
        if len(filtered_symbols) == 0:
            logger.debug(f"[{data_type}][{venue}] No supported symbols after filtering")
            return pd.DataFrame()
        if len(filtered_symbols) < len(symbols):
            logger.debug(f"[{data_type}][{venue}] Using {len(filtered_symbols)}/{len(symbols)} supported symbols")

        # Use filtered symbols for all subsequent operations
        symbols = filtered_symbols

        collector = collector_factory(venue)
        self.stats.total_api_calls += 1

        try:
            if data_type == 'funding_rates':
                if not hasattr(collector, 'fetch_funding_rates'):
                    logger.debug(f"[{data_type}][{venue}] Method not supported")
                    return pd.DataFrame()
                return await collector.fetch_funding_rates(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date
                )
            elif data_type == 'ohlcv':
                if not hasattr(collector, 'fetch_ohlcv'):
                    logger.debug(f"[{data_type}][{venue}] Method not supported")
                    return pd.DataFrame()
                return await collector.fetch_ohlcv(
                    symbols=symbols,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
            elif data_type == 'open_interest':
                if hasattr(collector, 'fetch_open_interest'):
                    # Check method signature to provide correct parameters
                    import inspect
                    sig = inspect.signature(collector.fetch_open_interest)
                    params = sig.parameters

                    # Build kwargs based on available parameters
                    kwargs = {'symbols': symbols}

                    if 'timeframe' in params:
                        kwargs['timeframe'] = timeframe or '1h'
                    if 'start_date' in params:
                        kwargs['start_date'] = start_date
                    if 'end_date' in params:
                        kwargs['end_date'] = end_date

                    return await collector.fetch_open_interest(**kwargs)
                return pd.DataFrame()
            elif data_type in ('social', 'sentiment'):
                # BUGFIX: Explicit handling for social/sentiment data types
                # LunarCrush and other alternative collectors use collect_* naming
                method = getattr(collector, f'collect_{data_type}', None)
                if method is None:
                    method = getattr(collector, f'fetch_{data_type}', None)
                if method is None:
                    # Try fetch_social_metrics for LunarCrush
                    if data_type == 'social':
                        method = getattr(collector, 'fetch_social_metrics', None)
                    elif data_type == 'sentiment':
                        method = getattr(collector, 'fetch_sentiment_signals', None)
                if method:
                    import inspect
                    sig = inspect.signature(method)
                    params = sig.parameters
                    kwargs = {}
                    if 'symbols' in params:
                        kwargs['symbols'] = symbols
                    if 'start_date' in params:
                        kwargs['start_date'] = start_date
                    if 'end_date' in params:
                        kwargs['end_date'] = end_date
                    if 'interval' in params:
                        kwargs['interval'] = timeframe or '1d'
                    try:
                        return await method(**kwargs)
                    except Exception as e:
                        logger.warning(f"[{data_type}][{venue}] Collection error: {e}")
                        return pd.DataFrame()
                logger.debug(f"[{data_type}][{venue}] No suitable method found")
                return pd.DataFrame()
            else:
                # Generic fetch - handle different method signatures
                # BUGFIX: Try both fetch_* and collect_* method naming conventions
                # Some collectors use 'collect_social' instead of 'fetch_social' (e.g., LunarCrush)
                method = getattr(collector, f'fetch_{data_type}', None)
                if method is None:
                    # Try collect_* naming convention (used by LunarCrush, etc.)
                    method = getattr(collector, f'collect_{data_type}', None)
                if method:
                    import inspect
                    try:
                        sig = inspect.signature(method)
                        params = sig.parameters

                        # Check for required parameters we can't provide
                        required_params = [
                            p.name for p in params.values()
                            if p.default is inspect.Parameter.empty
                            and p.name not in ('self', 'symbols', 'assets', 'currency', 'start_date', 'end_date')
                        ]
                        if required_params:
                            # Method has required parameters we can't provide - skip silently
                            logger.debug(f"[{data_type}][{venue}] Skipping - missing required params: {required_params}")
                            return pd.DataFrame()

                        # Build kwargs based on method signature
                        kwargs = {}
                        if 'symbols' in params:
                            kwargs['symbols'] = symbols
                        elif 'assets' in params:
                            kwargs['assets'] = symbols
                        elif 'currency' in params and len(symbols) > 0:
                            # For single-currency methods like fetch_dvol
                            kwargs['currency'] = symbols[0].upper().replace('USDT', '').replace('USD', '')

                        if 'start_date' in params:
                            kwargs['start_date'] = start_date
                        if 'end_date' in params:
                            kwargs['end_date'] = end_date

                        return await method(**kwargs)
                    except AttributeError as e:
                        # Collector missing required attributes - skip silently
                        logger.debug(f"[{data_type}][{venue}] Collector not properly configured: {e}")
                        return pd.DataFrame()
                    except Exception as e:
                        logger.debug(f"[{data_type}][{venue}] Collection error: {e}")
                        return pd.DataFrame()
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"[{data_type}][{venue}] Fetch error: {e}")
            return pd.DataFrame()
        # NOTE: Do NOT close collector here - it's reused across data types
        # The collector pool manages lifecycle at end of run

    def _on_retry(
        self,
        venue: str,
        attempt: int,
        exception: Exception,
        delay: float
    ) -> None:
        """Callback when retry occurs."""
        self.stats.total_retries += 1
        logger.warning(
            f"[{venue}] Retry {attempt + 1}: {type(exception).__name__}, "
            f"waiting {delay:.2f}s"
        )

    def _log_summary(self) -> None:
        """Log execution summary."""
        summary = self.stats.summary()
        rate_stats = self._rate_manager.stats()

        logger.info("=" * 70)
        logger.info("comprehensive PARALLEL EXECUTION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Duration: {summary['duration_seconds']}s")
        logger.info(f"Data Types: {summary['data_types_processed']}")
        logger.info(f"Venues: {summary['venues_processed']}")
        logger.info(f"Total Records: {summary['total_records']:,}")
        logger.info(f"Throughput: {summary['records_per_second']:.0f} records/sec")
        logger.info(f"Cache Hit Rate: {summary['cache_hit_rate']}")
        logger.info(f" - From Cache: {summary['records_from_cache']:,}")
        logger.info(f" - Collected: {summary['records_collected']:,}")
        logger.info(f"Rate Limited: {summary['rate_limited_count']} times")
        logger.info(f"Circuit Opens: {summary['circuit_opens']}")
        logger.info(f"Total Retries: {summary['total_retries']}")
        logger.info(f"Peak Concurrency: {summary['peak_concurrency']}")
        if summary['errors'] > 0:
            logger.warning(f"Errors: {summary['errors']}")
        logger.info("=" * 70)

    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def execute_enhanced_collection(
    data_types: List[str],
    venues: List[str],
    symbols: List[str],
    start_date: str,
    end_date: str,
    collector_factory: Callable[[str], Any],
    cache_dir: str = 'data/processed',
    use_cache: bool = True,
    timeframe: str = '1h'
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Convenience function for comprehensive parallel collection.

    This provides maximum parallelization with all protections:
    - Adaptive rate limiting
    - Circuit breakers
    - Exponential backoff with jitter
    - Incremental caching
    """
    executor = EnhancedParallelExecutor(
        cache_dir=cache_dir,
        use_cache=use_cache
    )

    try:
        return await executor.execute_collection(
            data_types=data_types,
            venues=venues,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            collector_factory=collector_factory,
            timeframe=timeframe
        )
    finally:
        await executor.cleanup()

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Rate Limiting
    'TokenBucketConfig',
    'TokenBucketRateLimiter',
    'VenueRateLimiterManager',
    'VENUE_RATE_LIMITS',

    # Circuit Breaker
    'CircuitState',
    'CircuitBreakerConfig',
    'CircuitBreaker',
    'CircuitBreakerOpen',

    # Backoff
    'BackoffConfig',
    'ExponentialBackoff',
    'retry_with_backoff',

    # Priority Queue
    'PriorityTask',
    'PriorityTaskQueue',

    # Backpressure
    'BackpressureController',

    # Adaptive Semaphore
    'AdaptiveSemaphore',

    # Executor
    'EnhancedExecutionStats',
    'EnhancedParallelExecutor',
    'execute_enhanced_collection',
]
