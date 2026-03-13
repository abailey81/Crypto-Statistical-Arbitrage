"""
Retry Handler Module for Crypto Statistical Arbitrage Systems.

This module provides professional-quality retry handling with exponential backoff,
circuit breaker patterns, and adaptive strategies specifically designed for
multi-venue crypto data collection reliability.

==============================================================================
RETRY STRATEGY FUNDAMENTALS
==============================================================================

Exponential Backoff with Jitter:
    delay = min(base_delay × (exponential_base ^ attempt) + jitter, max_delay)

    where:
        jitter = random(0, jitter_range × delay)

    This prevents thundering herd problems when multiple collectors
    retry simultaneously after a transient failure.

Backoff Strategy Comparison:
+------------------+------------------+------------------+------------------+
| Strategy | Formula | Best For | Risk |
+------------------+------------------+------------------+------------------+
| Linear | base × attempt | Gentle backoff | Slow recovery |
| Exponential | base × 2^attempt | Rate limits | Long waits |
| Fibonacci | fib(attempt) | Balanced | Moderate waits |
| Constant | base | Known issues | Hammering |
| Decorrelated | random(base,last)| AWS-style | Unpredictable |
+------------------+------------------+------------------+------------------+

==============================================================================
CIRCUIT BREAKER PATTERN
==============================================================================

The circuit breaker pattern prevents cascading failures by stopping
requests to failing services:

States:
    CLOSED: Normal operation, requests flow through
    OPEN: Failures exceeded threshold, requests fail immediately
    HALF_OPEN: Testing if service recovered

State Transitions:
    CLOSED → OPEN: failure_count >= failure_threshold
    OPEN → HALF_OPEN: recovery_timeout elapsed
    HALF_OPEN → CLOSED: success in half-open state
    HALF_OPEN → OPEN: failure in half-open state

Configuration Guidelines:
+------------------+------------------+------------------+------------------+
| Venue Type | Failure Thresh | Recovery Time | Half-Open Reqs |
+------------------+------------------+------------------+------------------+
| CEX (Binance) | 5 | 30 seconds | 3 |
| Hybrid (HL) | 3 | 60 seconds | 2 |
| DEX (TheGraph) | 3 | 120 seconds | 1 |
| Data Provider | 5 | 60 seconds | 2 |
+------------------+------------------+------------------+------------------+

==============================================================================
FAILURE CLASSIFICATION
==============================================================================

Failure types determine retry behavior:

Transient (Should Retry):
    - Network timeouts
    - Connection refused (temporary)
    - 5xx server errors
    - Rate limit (429) with Retry-After

Semi-Transient (Limited Retry):
    - DNS resolution failures
    - SSL certificate issues
    - Service unavailable

Permanent (No Retry):
    - Authentication failures (401)
    - Authorization failures (403)
    - Not found (404)
    - Bad request (400)
    - Validation errors

==============================================================================
STATISTICAL ARBITRAGE IMPLICATIONS
==============================================================================

1. DATA CONTINUITY
   - Retry failures risk data gaps
   - Circuit breaker prevents cascade failures
   - Prioritize retries for time-sensitive data (funding rates)

2. CROSS-VENUE SYNCHRONIZATION
   - Venue A failure shouldn't impact venue B
   - Isolate circuit breakers per venue
   - Fallback to alternative data sources

3. SIGNAL INTEGRITY
   - Stale data from excessive retries degrades signals
   - Timeout thresholds should match signal freshness requirements
   - Funding rates: 5-15 minute retry budget
   - OHLCV: 1-5 minute retry budget

4. COST MANAGEMENT
   - Retries consume API quota
   - Failed requests may still count against limits
   - Circuit breaker reduces wasted requests

==============================================================================
USAGE EXAMPLES
==============================================================================

Basic Retry:
    >>> handler = RetryHandler(max_retries=3, base_delay=1.0)
    >>> result = await handler.execute(fetch_funding_rates, symbol='BTC')

With Circuit Breaker:
    >>> circuit = CircuitBreaker(
    ... name='binance',
    ... failure_threshold=5,
    ... recovery_timeout=30.0
    ... )
    >>> handler = RetryHandler(max_retries=3, circuit_breaker=circuit)
    >>> result = await handler.execute(api_call)

Decorator Pattern:
    >>> @retry(max_retries=3, base_delay=1.0, circuit_breaker=circuit)
    >>> async def fetch_data(symbol: str):
    ... return await api.get(f'/data/{symbol}')

Adaptive Retry:
    >>> handler = AdaptiveRetryHandler(
    ... base_retries=3,
    ... base_delay=1.0,
    ... adapt_to_failure_type=True
    ... )

Version: 2.0.0
"""

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')

# =============================================================================
# SAFE CONVERSION UTILITIES
# =============================================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float, handling empty strings, None, and invalid values.

    This is critical for API responses where numeric fields may be:
    - Empty strings ''
    - None/null
    - Missing keys
    - Invalid numeric strings

    Args:
        value: The value to convert (can be str, int, float, None, etc.)
        default: Default value if conversion fails (default: 0.0)

    Returns:
        The float value or default if conversion fails

    Examples:
        >>> safe_float('123.45')
        123.45
        >>> safe_float('')
        0.0
        >>> safe_float(None)
        0.0
        >>> safe_float('invalid', -1.0)
        -1.0
    """
    if value is None or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to int, handling empty strings, None, and invalid values.

    Args:
        value: The value to convert
        default: Default value if conversion fails (default: 0)

    Returns:
        The int value or default if conversion fails
    """
    if value is None or value == '':
        return default
    try:
        return int(float(value)) # Handle float strings like '123.45'
    except (ValueError, TypeError):
        return default

# =============================================================================
# ENUMS
# =============================================================================

class RetryStrategy(Enum):
    """
    Retry delay calculation strategies.

    Different strategies optimize for different failure patterns.
    """

    CONSTANT = "constant" # Fixed delay
    LINEAR = "linear" # Linear increase
    EXPONENTIAL = "exponential" # Exponential backoff
    FIBONACCI = "fibonacci" # Fibonacci sequence
    DECORRELATED = "decorrelated" # AWS-style decorrelated jitter

    @property
    def description(self) -> str:
        """Strategy description."""
        descriptions = {
            RetryStrategy.CONSTANT: "Fixed delay between retries",
            RetryStrategy.LINEAR: "Linearly increasing delay",
            RetryStrategy.EXPONENTIAL: "Exponentially increasing delay (recommended)",
            RetryStrategy.FIBONACCI: "Fibonacci sequence delays",
            RetryStrategy.DECORRELATED: "Decorrelated jitter (AWS-style)",
        }
        return descriptions.get(self, "Unknown")

    @property
    def best_for(self) -> str:
        """Best use case for this strategy."""
        use_cases = {
            RetryStrategy.CONSTANT: "Known transient issues with fixed recovery time",
            RetryStrategy.LINEAR: "Gentle backoff for light load",
            RetryStrategy.EXPONENTIAL: "Rate limits and unknown recovery time",
            RetryStrategy.FIBONACCI: "Balanced approach between linear and exponential",
            RetryStrategy.DECORRELATED: "Avoiding thundering herd in distributed systems",
        }
        return use_cases.get(self, "General use")

    def calculate_delay(
        self,
        attempt: int,
        base_delay: float,
        max_delay: float,
        last_delay: float = 0,
    ) -> float:
        """Calculate delay for given attempt."""
        if self == RetryStrategy.CONSTANT:
            delay = base_delay
        elif self == RetryStrategy.LINEAR:
            delay = base_delay * (attempt + 1)
        elif self == RetryStrategy.EXPONENTIAL:
            delay = base_delay * (2 ** attempt)
        elif self == RetryStrategy.FIBONACCI:
            delay = base_delay * self._fibonacci(attempt + 2)
        elif self == RetryStrategy.DECORRELATED:
            delay = random.uniform(base_delay, max(base_delay, last_delay * 3))
        else:
            delay = base_delay * (2 ** attempt)

        return min(delay, max_delay)

    @staticmethod
    def _fibonacci(n: int) -> int:
        """Calculate fibonacci number."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

class FailureCategory(Enum):
    """
    Failure type classification for retry decisions.

    Determines whether and how to retry failed requests.
    """

    TRANSIENT = "transient" # Should retry
    RATE_LIMITED = "rate_limited" # Retry with backoff
    SERVER_ERROR = "server_error" # Retry with caution
    CLIENT_ERROR = "client_error" # Usually no retry
    NETWORK_ERROR = "network_error" # Should retry
    TIMEOUT = "timeout" # Should retry
    AUTHENTICATION = "authentication" # No retry
    AUTHORIZATION = "authorization" # No retry
    NOT_FOUND = "not_found" # No retry
    VALIDATION = "validation" # No retry
    UNKNOWN = "unknown" # Limited retry

    @property
    def should_retry(self) -> bool:
        """Whether this failure type should be retried."""
        return self in {
            FailureCategory.TRANSIENT,
            FailureCategory.RATE_LIMITED,
            FailureCategory.SERVER_ERROR,
            FailureCategory.NETWORK_ERROR,
            FailureCategory.TIMEOUT,
            FailureCategory.UNKNOWN,
        }

    @property
    def max_retries_modifier(self) -> float:
        """Modifier for max retries (0.0 to 2.0)."""
        modifiers = {
            FailureCategory.TRANSIENT: 1.0,
            FailureCategory.RATE_LIMITED: 1.5,
            FailureCategory.SERVER_ERROR: 1.0,
            FailureCategory.CLIENT_ERROR: 0.0,
            FailureCategory.NETWORK_ERROR: 1.5,
            FailureCategory.TIMEOUT: 1.0,
            FailureCategory.AUTHENTICATION: 0.0,
            FailureCategory.AUTHORIZATION: 0.0,
            FailureCategory.NOT_FOUND: 0.0,
            FailureCategory.VALIDATION: 0.0,
            FailureCategory.UNKNOWN: 0.5,
        }
        return modifiers.get(self, 0.5)

    @property
    def base_delay_modifier(self) -> float:
        """Modifier for base delay (0.5 to 3.0)."""
        modifiers = {
            FailureCategory.TRANSIENT: 1.0,
            FailureCategory.RATE_LIMITED: 2.0,
            FailureCategory.SERVER_ERROR: 1.5,
            FailureCategory.NETWORK_ERROR: 1.0,
            FailureCategory.TIMEOUT: 0.5,
            FailureCategory.UNKNOWN: 1.0,
        }
        return modifiers.get(self, 1.0)

    @classmethod
    def from_status_code(cls, status_code: int) -> 'FailureCategory':
        """Classify from HTTP status code."""
        if status_code == 429:
            return cls.RATE_LIMITED
        elif status_code == 401:
            return cls.AUTHENTICATION
        elif status_code == 403:
            return cls.AUTHORIZATION
        elif status_code == 404:
            return cls.NOT_FOUND
        elif status_code == 400:
            return cls.VALIDATION
        elif 400 <= status_code < 500:
            return cls.CLIENT_ERROR
        elif 500 <= status_code < 600:
            return cls.SERVER_ERROR
        return cls.UNKNOWN

    @classmethod
    def from_exception(cls, exception: Exception) -> 'FailureCategory':
        """Classify from exception type."""
        exc_type = type(exception).__name__

        # Network errors
        if exc_type in ('ConnectionError', 'ConnectionRefusedError', 'ConnectionResetError'):
            return cls.NETWORK_ERROR

        # Timeout errors
        if exc_type in ('TimeoutError', 'asyncio.TimeoutError', 'ReadTimeout', 'ConnectTimeout'):
            return cls.TIMEOUT

        # Check for status code attribute
        if hasattr(exception, 'status'):
            return cls.from_status_code(exception.status)
        if hasattr(exception, 'status_code'):
            return cls.from_status_code(exception.status_code)
        if hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
            return cls.from_status_code(exception.response.status_code)

        return cls.UNKNOWN

class CircuitState(Enum):
    """
    Circuit breaker state.
    """

    CLOSED = "closed" # Normal operation
    OPEN = "open" # Failing, reject requests
    HALF_OPEN = "half_open" # Testing recovery

    @property
    def allows_requests(self) -> bool:
        """Whether state allows requests."""
        return self in {CircuitState.CLOSED, CircuitState.HALF_OPEN}

    @property
    def is_healthy(self) -> bool:
        """Whether state indicates healthy service."""
        return self == CircuitState.CLOSED

    @property
    def description(self) -> str:
        """State description."""
        descriptions = {
            CircuitState.CLOSED: "Normal operation, requests allowed",
            CircuitState.OPEN: "Service failing, requests rejected",
            CircuitState.HALF_OPEN: "Testing if service recovered",
        }
        return descriptions.get(self, "Unknown")

class RecoveryAction(Enum):
    """
    Recommended action after failure.
    """

    RETRY_IMMEDIATELY = "retry_immediately"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    WAIT_FOR_RESET = "wait_for_reset"
    USE_FALLBACK = "use_fallback"
    FAIL_PERMANENTLY = "fail_permanently"
    CIRCUIT_OPEN = "circuit_open"

    @property
    def description(self) -> str:
        """Action description."""
        descriptions = {
            RecoveryAction.RETRY_IMMEDIATELY: "Retry without delay",
            RecoveryAction.RETRY_WITH_BACKOFF: "Retry with exponential backoff",
            RecoveryAction.WAIT_FOR_RESET: "Wait for rate limit reset",
            RecoveryAction.USE_FALLBACK: "Use fallback data source",
            RecoveryAction.FAIL_PERMANENTLY: "No retry, fail request",
            RecoveryAction.CIRCUIT_OPEN: "Circuit breaker open, reject",
        }
        return descriptions.get(self, "Unknown")

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class RetryStats:
    """
    Comprehensive statistics for retry operations.
    """

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_retries: int = 0
    total_delay_time_seconds: float = 0.0
    failures_by_category: Dict[str, int] = field(default_factory=dict)
    window_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def success_rate(self) -> float:
        """Success rate percentage."""
        if self.total_calls == 0:
            return 100.0
        return (self.successful_calls / self.total_calls) * 100

    @property
    def failure_rate(self) -> float:
        """Failure rate percentage."""
        return 100.0 - self.success_rate

    @property
    def average_retries_per_failure(self) -> float:
        """Average retries per failed call."""
        if self.failed_calls == 0:
            return 0.0
        return self.total_retries / self.failed_calls

    @property
    def average_delay_per_retry(self) -> float:
        """Average delay per retry."""
        if self.total_retries == 0:
            return 0.0
        return self.total_delay_time_seconds / self.total_retries

    @property
    def most_common_failure(self) -> Optional[str]:
        """Most common failure category."""
        if not self.failures_by_category:
            return None
        return max(self.failures_by_category, key=self.failures_by_category.get)

    @property
    def health_score(self) -> float:
        """Overall health score (0-100)."""
        score = 100.0
        score -= min(40, self.failure_rate * 0.8)
        score -= min(30, self.average_retries_per_failure * 5)
        return max(0, score)

    def reset(self) -> None:
        """Reset statistics."""
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_retries = 0
        self.total_delay_time_seconds = 0.0
        self.failures_by_category = {}
        self.window_start = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'total_retries': self.total_retries,
            'total_delay_time_seconds': round(self.total_delay_time_seconds, 3),
            'success_rate': round(self.success_rate, 2),
            'failure_rate': round(self.failure_rate, 2),
            'average_retries_per_failure': round(self.average_retries_per_failure, 2),
            'average_delay_per_retry': round(self.average_delay_per_retry, 3),
            'most_common_failure': self.most_common_failure,
            'failures_by_category': self.failures_by_category,
            'health_score': round(self.health_score, 2),
        }

@dataclass
class CircuitBreakerState:
    """
    Current state of a circuit breaker.
    """

    name: str
    state: CircuitState
    failure_count: int
    success_count: int
    failure_threshold: int
    last_failure_at: Optional[datetime]
    last_success_at: Optional[datetime]
    opened_at: Optional[datetime]
    recovery_timeout_seconds: float

    @property
    def is_healthy(self) -> bool:
        """Whether circuit is healthy."""
        return self.state.is_healthy

    @property
    def allows_requests(self) -> bool:
        """Whether requests are allowed."""
        return self.state.allows_requests

    @property
    def failure_rate(self) -> float:
        """Current failure rate."""
        total = self.failure_count + self.success_count
        if total == 0:
            return 0.0
        return (self.failure_count / total) * 100

    @property
    def time_until_half_open(self) -> Optional[float]:
        """Seconds until circuit transitions to half-open."""
        if self.state != CircuitState.OPEN or self.opened_at is None:
            return None
        elapsed = (datetime.now(timezone.utc) - self.opened_at).total_seconds()
        remaining = self.recovery_timeout_seconds - elapsed
        return max(0, remaining)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'state': self.state.value,
            'is_healthy': self.is_healthy,
            'allows_requests': self.allows_requests,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'failure_threshold': self.failure_threshold,
            'failure_rate': round(self.failure_rate, 2),
            'last_failure_at': self.last_failure_at.isoformat() if self.last_failure_at else None,
            'last_success_at': self.last_success_at.isoformat() if self.last_success_at else None,
            'opened_at': self.opened_at.isoformat() if self.opened_at else None,
            'time_until_half_open': self.time_until_half_open,
            'recovery_timeout_seconds': self.recovery_timeout_seconds,
        }

@dataclass
class RetryResult:
    """
    Result of a retry operation.
    """

    success: bool
    value: Any
    attempts: int
    total_delay_seconds: float
    failure_category: Optional[FailureCategory]
    last_exception: Optional[Exception]
    recovery_action: RecoveryAction
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def was_retried(self) -> bool:
        """Whether any retries were attempted."""
        return self.attempts > 1

    @property
    def error_message(self) -> Optional[str]:
        """Error message from last exception."""
        if self.last_exception:
            return str(self.last_exception)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'attempts': self.attempts,
            'total_delay_seconds': round(self.total_delay_seconds, 3),
            'was_retried': self.was_retried,
            'failure_category': self.failure_category.value if self.failure_category else None,
            'error_message': self.error_message,
            'recovery_action': self.recovery_action.value,
            'timestamp': self.timestamp.isoformat(),
        }

# =============================================================================
# EXCEPTIONS
# =============================================================================

class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(
        self,
        message: str,
        last_exception: Optional[Exception] = None,
        attempts: int = 0,
        failure_category: Optional[FailureCategory] = None,
    ):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts
        self.failure_category = failure_category

class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, name: str, time_until_reset: Optional[float] = None):
        super().__init__(f"Circuit breaker '{name}' is open")
        self.name = name
        self.time_until_reset = time_until_reset

# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker for preventing cascading failures.

    Implements the circuit breaker pattern to stop requests
    to failing services and allow recovery.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        recovery_timeout: float = 30.0,
        half_open_max_requests: int = 3,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_requests = half_open_max_requests

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_requests = 0
        self._last_failure_at: Optional[datetime] = None
        self._last_success_at: Optional[datetime] = None
        self._opened_at: Optional[datetime] = None
        self._lock = asyncio.Lock()

        logger.debug(f"CircuitBreaker '{name}' initialized")

    @property
    def state(self) -> CircuitState:
        """Current state (may trigger transition)."""
        self._check_state_transition()
        return self._state

    def _check_state_transition(self) -> None:
        """Check and perform state transitions."""
        if self._state == CircuitState.OPEN and self._opened_at:
            elapsed = (datetime.now(timezone.utc) - self._opened_at).total_seconds()
            if elapsed >= self.recovery_timeout:
                self._transition_to_half_open()

    def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        self._state = CircuitState.HALF_OPEN
        self._half_open_requests = 0
        logger.info(f"CircuitBreaker '{self.name}' transitioned to HALF_OPEN")

    def _transition_to_open(self) -> None:
        """Transition to open state."""
        self._state = CircuitState.OPEN
        self._opened_at = datetime.now(timezone.utc)
        logger.warning(f"CircuitBreaker '{self.name}' transitioned to OPEN")

    def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at = None
        logger.info(f"CircuitBreaker '{self.name}' transitioned to CLOSED")

    async def acquire(self) -> bool:
        """
        Attempt to acquire permission to make request.

        Returns True if allowed, False if circuit is open.
        """
        async with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.OPEN:
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_requests >= self.half_open_max_requests:
                    return False
                self._half_open_requests += 1

            return True

    async def record_success(self) -> None:
        """Record a successful request."""
        async with self._lock:
            self._success_count += 1
            self._last_success_at = datetime.now(timezone.utc)

            if self._state == CircuitState.HALF_OPEN:
                if self._success_count >= self.success_threshold:
                    self._transition_to_closed()

    async def record_failure(self, exception: Optional[Exception] = None) -> None:
        """Record a failed request."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_at = datetime.now(timezone.utc)

            if self._state == CircuitState.HALF_OPEN:
                self._transition_to_open()
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._transition_to_open()

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return CircuitBreakerState(
            name=self.name,
            state=self.state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            failure_threshold=self.failure_threshold,
            last_failure_at=self._last_failure_at,
            last_success_at=self._last_success_at,
            opened_at=self._opened_at,
            recovery_timeout_seconds=self.recovery_timeout,
        )

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._transition_to_closed()
        logger.info(f"CircuitBreaker '{self.name}' reset")

# =============================================================================
# RETRY HANDLER
# =============================================================================

class RetryHandler:
    """
    Comprehensive retry handler with circuit breaker support.

    Provides configurable retry logic with exponential backoff,
    jitter, failure classification, and circuit breaker integration.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        jitter: bool = True,
        jitter_range: float = 0.5,
        circuit_breaker: Optional[CircuitBreaker] = None,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
        name: Optional[str] = None,
        on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.strategy = strategy
        self.jitter = jitter
        self.jitter_range = jitter_range
        self.circuit_breaker = circuit_breaker
        self.name = name or "retry_handler"
        self.on_retry = on_retry

        self.retryable_exceptions = retryable_exceptions or (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
        )

        self.stats = RetryStats()

        logger.debug(
            f"RetryHandler '{self.name}' initialized: "
            f"max_retries={max_retries}, strategy={strategy.value}"
        )

    def _calculate_delay(self, attempt: int, last_delay: float = 0) -> float:
        """Calculate delay for given attempt."""
        delay = self.strategy.calculate_delay(
            attempt=attempt,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            last_delay=last_delay,
        )

        if self.jitter:
            jitter = random.uniform(0, self.jitter_range * delay)
            delay += jitter

        return min(delay, self.max_delay)

    def _should_retry(
        self,
        exception: Exception,
        attempt: int,
    ) -> Tuple[bool, FailureCategory]:
        """Determine if exception should be retried."""
        category = FailureCategory.from_exception(exception)

        if not category.should_retry:
            return False, category

        if isinstance(exception, self.retryable_exceptions):
            return True, category

        if hasattr(exception, 'status'):
            status = exception.status
            if status == 429 or status >= 500:
                return True, category

        return category.should_retry, category

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute function with retry logic.

        Returns the result or raises RetryExhausted/CircuitBreakerOpen.
        """
        self.stats.total_calls += 1
        last_exception: Optional[Exception] = None
        last_category: Optional[FailureCategory] = None
        total_delay = 0.0
        last_delay = 0.0

        # Check circuit breaker
        if self.circuit_breaker:
            if not await self.circuit_breaker.acquire():
                state = self.circuit_breaker.get_state()
                raise CircuitBreakerOpen(
                    self.circuit_breaker.name,
                    state.time_until_half_open,
                )

        for attempt in range(self.max_retries + 1):
            try:
                result = await func(*args, **kwargs)

                self.stats.successful_calls += 1

                if self.circuit_breaker:
                    await self.circuit_breaker.record_success()

                if attempt > 0:
                    logger.info(
                        f"RetryHandler '{self.name}': succeeded on attempt {attempt + 1}"
                    )

                return result

            except Exception as e:
                last_exception = e

                should_retry, category = self._should_retry(e, attempt)
                last_category = category

                # Update failure stats
                category_key = category.value
                self.stats.failures_by_category[category_key] = \
                    self.stats.failures_by_category.get(category_key, 0) + 1

                if self.circuit_breaker:
                    await self.circuit_breaker.record_failure(e)

                if not should_retry:
                    logger.warning(
                        f"RetryHandler '{self.name}': non-retryable error "
                        f"({category.value}): {e}"
                    )
                    self.stats.failed_calls += 1
                    raise

                if attempt >= self.max_retries:
                    logger.error(
                        f"RetryHandler '{self.name}': exhausted {self.max_retries + 1} attempts"
                    )
                    self.stats.failed_calls += 1
                    raise RetryExhausted(
                        f"All {self.max_retries + 1} attempts failed",
                        last_exception=last_exception,
                        attempts=attempt + 1,
                        failure_category=last_category,
                    )

                # Calculate delay
                delay = self._calculate_delay(attempt, last_delay)
                delay *= category.base_delay_modifier
                delay = min(delay, self.max_delay)

                total_delay += delay
                last_delay = delay
                self.stats.total_retries += 1
                self.stats.total_delay_time_seconds += delay

                logger.warning(
                    f"RetryHandler '{self.name}': attempt {attempt + 1} failed "
                    f"({category.value}): {e}. Retrying in {delay:.2f}s..."
                )

                if self.on_retry:
                    try:
                        self.on_retry(attempt, e, delay)
                    except Exception as callback_error:
                        logger.error(f"Retry callback error: {callback_error}")

                await asyncio.sleep(delay)

        self.stats.failed_calls += 1
        raise RetryExhausted(
            "Retry loop exited unexpectedly",
            last_exception=last_exception,
            attempts=self.max_retries + 1,
            failure_category=last_category,
        )

    def reset_stats(self) -> None:
        """Reset retry statistics."""
        self.stats.reset()

    def __repr__(self) -> str:
        return (
            f"RetryHandler(name={self.name}, max_retries={self.max_retries}, "
            f"strategy={self.strategy.value})"
        )

# =============================================================================
# DECORATOR
# =============================================================================

def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    jitter: bool = True,
    circuit_breaker: Optional[CircuitBreaker] = None,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
) -> Callable:
    """
    Decorator for adding retry logic to async functions.
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        handler = RetryHandler(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            strategy=strategy,
            jitter=jitter,
            circuit_breaker=circuit_breaker,
            retryable_exceptions=retryable_exceptions,
            name=func.__name__,
        )

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await handler.execute(func, *args, **kwargs)

        wrapper.retry_handler = handler
        return wrapper

    return decorator

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def with_timeout_and_retry(
    func: Callable[..., Awaitable[T]],
    timeout: float,
    max_retries: int = 3,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute async function with both timeout and retry."""
    handler = RetryHandler(max_retries=max_retries)

    async def wrapped():
        return await asyncio.wait_for(
            func(*args, **kwargs),
            timeout=timeout,
        )

    return await handler.execute(wrapped)

def create_venue_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """Create circuit breakers for all venues."""
    return {
        'binance': CircuitBreaker('binance', failure_threshold=5, recovery_timeout=30.0),
        'binance_futures': CircuitBreaker('binance_futures', failure_threshold=5, recovery_timeout=30.0),
        'bybit': CircuitBreaker('bybit', failure_threshold=5, recovery_timeout=45.0),
        'okx': CircuitBreaker('okx', failure_threshold=5, recovery_timeout=45.0),
        'hyperliquid': CircuitBreaker('hyperliquid', failure_threshold=3, recovery_timeout=60.0),
        'dydx_v4': CircuitBreaker('dydx_v4', failure_threshold=3, recovery_timeout=60.0),
        'deribit': CircuitBreaker('deribit', failure_threshold=5, recovery_timeout=30.0),
        'thegraph': CircuitBreaker('thegraph', failure_threshold=3, recovery_timeout=120.0),
        'defillama': CircuitBreaker('defillama', failure_threshold=5, recovery_timeout=60.0),
    }

__all__ = [
    # Enums
    'RetryStrategy',
    'FailureCategory',
    'CircuitState',
    'RecoveryAction',
    # Dataclasses
    'RetryStats',
    'CircuitBreakerState',
    'RetryResult',
    # Exceptions
    'RetryExhausted',
    'CircuitBreakerOpen',
    # Classes
    'CircuitBreaker',
    'RetryHandler',
    # Decorator
    'retry',
    # Functions
    'with_timeout_and_retry',
    'create_venue_circuit_breakers',
]
