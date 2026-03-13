"""
Utility modules for Crypto Statistical Arbitrage System.

This package provides core infrastructure and processing utilities:

Core Infrastructure:
- network_infrastructure: Unified rate limiting and retry handling
- storage: Tiered parquet storage with metadata

Data Processing:
- data_cleaner: professional-quality data cleaning
- data_validator: Schema and quality validation
- quality_checks: Comprehensive quality assessment
- funding_normalization: CRITICAL - 1h/8h funding rate conversion
- funding_processor: Funding rate analytics

External Integrations:
- ccxt_wrapper: CCXT library wrapper for multi-exchange support
"""

import logging

logger = logging.getLogger(__name__)

# Network Infrastructure (always available)
from .network_infrastructure import (
    VenueTier,
    RateLimitStrategy,
    ThrottleStatus,
    RequestPriority,
    RetryStrategy,
    FailureCategory,
    CircuitState,
    RecoveryAction,
    RateLimitConfig,
    CircuitBreakerConfig,
    RetryConfig,
    RateLimiter,
    CircuitBreaker,
    RetryHandler,
    NetworkManager,
    create_network_manager,
    with_retry,
    with_rate_limit,
)

__all__ = [
    # Network Infrastructure
    'VenueTier',
    'RateLimitStrategy',
    'ThrottleStatus',
    'RequestPriority',
    'RetryStrategy',
    'FailureCategory',
    'CircuitState',
    'RecoveryAction',
    'RateLimitConfig',
    'CircuitBreakerConfig',
    'RetryConfig',
    'RateLimiter',
    'CircuitBreaker',
    'RetryHandler',
    'NetworkManager',
    'create_network_manager',
    'with_retry',
    'with_rate_limit',
]

# Graceful imports for modules that may have complex dependencies
try:
    from .storage import DataStorage, StorageTier, StorageConfig
    __all__.extend(['DataStorage', 'StorageTier', 'StorageConfig'])
except ImportError as e:
    logger.debug(f"storage module not available: {e}")

try:
    from .data_cleaner import CleaningPipeline, CleaningResult
    __all__.extend(['CleaningPipeline', 'CleaningResult'])
except ImportError as e:
    logger.debug(f"data_cleaner module not available: {e}")

try:
    from .data_validator import DataValidator, ValidationResult
    __all__.extend(['DataValidator', 'ValidationResult'])
except ImportError as e:
    logger.debug(f"data_validator module not available: {e}")

try:
    from .quality_checks import QualityChecker, QualityResult
    __all__.extend(['QualityChecker', 'QualityResult'])
except ImportError as e:
    logger.debug(f"quality_checks module not available: {e}")

try:
    from .funding_normalization import (
        FundingNormalizer,
        NormalizationInterval,
        AlignmentStrategy,
    )
    __all__.extend(['FundingNormalizer', 'NormalizationInterval', 'AlignmentStrategy'])
except ImportError as e:
    logger.debug(f"funding_normalization module not available: {e}")

try:
    from .funding_processor import FundingProcessor, FundingAnalysis
    __all__.extend(['FundingProcessor', 'FundingAnalysis'])
except ImportError as e:
    logger.debug(f"funding_processor module not available: {e}")
