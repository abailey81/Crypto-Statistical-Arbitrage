"""
Integrated Error Monitoring and Performance Tracking for Phase 1.

This module provides comprehensive monitoring capabilities that are automatically
enabled when running Phase 1 data collection. No separate scripts needed.

Features:
- Automatic error categorization and tracking
- Performance metrics per venue/data type
- Resource usage monitoring
- Real-time progress tracking
- Automatic report generation

Version: 1.0.0
"""

import asyncio
import logging
import time
import traceback
import json
import psutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)

# =============================================================================
# ERROR TRACKING
# =============================================================================

class ErrorCategory(Enum):
    """Error categories for automatic classification."""
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    DATA_QUALITY = "data_quality"
    PARSE_ERROR = "parse_error"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RESOURCE = "resource"
    UNKNOWN = "unknown"

class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class ErrorRecord:
    """Detailed error record."""
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    venue: str
    data_type: str
    symbol: Optional[str]
    error_message: str
    stack_trace: Optional[str] = None
    retry_count: int = 0
    resolved: bool = False

@dataclass
class VenueMetrics:
    """Performance metrics per venue."""
    venue: str
    data_type: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_records: int = 0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    rate_limit_hits: int = 0
    retry_count: int = 0
    total_duration_sec: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

class IntegratedMonitor:
    """
    Integrated monitoring system for Phase 1.

    Automatically tracks errors, performance, and resource usage
    without requiring separate monitoring scripts.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern - only one monitor instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize monitor (only once)."""
        if self._initialized:
            return

        self.errors: List[ErrorRecord] = []
        self.metrics: Dict[Tuple[str, str], VenueMetrics] = {}
        self.consecutive_errors: Dict[str, int] = defaultdict(int)
        self.venue_blacklist: set = set()

        # Configuration
        self.max_errors_per_venue = 100
        self.max_consecutive_errors = 10
        self.blacklist_enabled = True

        # Resource monitoring
        try:
            self.process = psutil.Process()
            self.initial_cpu = self.process.cpu_percent()
            self.initial_memory = self.process.memory_info().rss
        except:
            self.process = None

        # Progress tracking
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.start_time = None
        self.end_time = None

        self._initialized = True
        logger.info(" Integrated monitoring system initialized")

    def categorize_error(self, error_message: str) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Automatically categorize error based on message."""
        error_lower = error_message.lower()

        # Network errors
        if any(kw in error_lower for kw in ['timeout', 'connection', 'unreachable', 'connreset']):
            return ErrorCategory.NETWORK, ErrorSeverity.MEDIUM

        # Rate limiting
        if any(kw in error_lower for kw in ['429', 'rate limit', 'too many requests', 'throttle']):
            return ErrorCategory.RATE_LIMIT, ErrorSeverity.LOW

        # Authentication
        if any(kw in error_lower for kw in ['401', '403', 'unauthorized', 'forbidden', 'api key']):
            return ErrorCategory.AUTHENTICATION, ErrorSeverity.CRITICAL

        # API errors
        if any(kw in error_lower for kw in ['400', '404', '500', '502', '503', '504']):
            severity = ErrorSeverity.LOW if '404' in error_lower else ErrorSeverity.MEDIUM
            return ErrorCategory.API_ERROR, severity

        # Data quality
        if any(kw in error_lower for kw in ['no data', 'empty', 'invalid data', 'missing']):
            return ErrorCategory.DATA_QUALITY, ErrorSeverity.LOW

        # Parse errors
        if any(kw in error_lower for kw in ['json', 'parse', 'decode', 'unmarshal']):
            return ErrorCategory.PARSE_ERROR, ErrorSeverity.MEDIUM

        # Resource errors
        if any(kw in error_lower for kw in ['memory', 'disk', 'space', 'resource']):
            return ErrorCategory.RESOURCE, ErrorSeverity.HIGH

        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM

    def record_error(
        self,
        venue: str,
        data_type: str,
        error_message: str,
        symbol: Optional[str] = None,
        exc_info: Optional[Exception] = None
    ) -> None:
        """Record an error with automatic categorization."""
        category, severity = self.categorize_error(error_message)

        stack_trace = None
        if exc_info:
            stack_trace = ''.join(traceback.format_exception(
                type(exc_info), exc_info, exc_info.__traceback__
            ))

        error = ErrorRecord(
            timestamp=datetime.now(timezone.utc),
            category=category,
            severity=severity,
            venue=venue,
            data_type=data_type,
            symbol=symbol,
            error_message=error_message[:500], # Limit length
            stack_trace=stack_trace
        )

        self.errors.append(error)

        # Track consecutive errors
        key = f"{venue}:{data_type}"
        self.consecutive_errors[key] += 1

        # Check for blacklisting
        if self.blacklist_enabled:
            venue_errors = sum(1 for e in self.errors if e.venue == venue and not e.resolved)
            if venue_errors >= self.max_errors_per_venue:
                if venue not in self.venue_blacklist:
                    self.venue_blacklist.add(venue)
                    logger.warning(f"WARNING: Blacklisting {venue} due to {venue_errors} errors")

        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.error(f"CRITICAL [{venue}:{data_type}]: {error_message[:200]}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH [{venue}:{data_type}]: {error_message[:200]}")
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM [{venue}:{data_type}]: {error_message[:200]}")

    def record_success(self, venue: str, data_type: str) -> None:
        """Record successful operation."""
        key = f"{venue}:{data_type}"
        self.consecutive_errors[key] = 0

    def update_metrics(
        self,
        venue: str,
        data_type: str,
        records_collected: int,
        latency_ms: float,
        success: bool,
        rate_limited: bool = False
    ) -> None:
        """Update performance metrics."""
        key = (venue, data_type)

        if key not in self.metrics:
            self.metrics[key] = VenueMetrics(
                venue=venue,
                data_type=data_type,
                start_time=datetime.now(timezone.utc)
            )

        metrics = self.metrics[key]
        metrics.total_requests += 1

        if success:
            metrics.successful_requests += 1
            metrics.total_records += records_collected
            self.record_success(venue, data_type)
        else:
            metrics.failed_requests += 1
            metrics.retry_count += 1

        if rate_limited:
            metrics.rate_limit_hits += 1

        # Update latency
        if latency_ms > 0 and success:
            n = metrics.successful_requests
            metrics.avg_latency_ms = (
                (metrics.avg_latency_ms * (n - 1) + latency_ms) / n
                if n > 1 else latency_ms
            )
            metrics.min_latency_ms = min(metrics.min_latency_ms, latency_ms)
            metrics.max_latency_ms = max(metrics.max_latency_ms, latency_ms)

    def is_blacklisted(self, venue: str) -> bool:
        """Check if venue is blacklisted."""
        return venue in self.venue_blacklist

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary."""
        if not self.start_time:
            return {"status": "not_started"}

        duration = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        return {
            "status": "completed" if self.end_time else "running",
            "total_tasks": self.total_tasks,
            "completed": self.completed_tasks,
            "failed": self.failed_tasks,
            "success_rate": f"{(self.completed_tasks/self.total_tasks*100):.1f}%" if self.total_tasks > 0 else "0%",
            "duration_seconds": duration,
            "duration_formatted": f"{int(duration//60)}m {int(duration%60)}s"
        }

    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        if not self.errors:
            return {"total": 0, "by_category": {}, "by_severity": {}}

        by_category = defaultdict(int)
        by_severity = defaultdict(int)
        by_venue = defaultdict(int)

        for error in self.errors:
            by_category[error.category.value] += 1
            by_severity[error.severity.value] += 1
            by_venue[error.venue] += 1

        return {
            "total": len(self.errors),
            "by_category": dict(by_category),
            "by_severity": dict(by_severity),
            "by_venue": dict(sorted(by_venue.items(), key=lambda x: x[1], reverse=True)[:10]),
            "blacklisted_venues": list(self.venue_blacklist)
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics:
            return {}

        summary = {}
        for (venue, data_type), metrics in self.metrics.items():
            key = f"{venue}:{data_type}"

            success_rate = (
                metrics.successful_requests / metrics.total_requests * 100
                if metrics.total_requests > 0 else 0
            )

            throughput = (
                metrics.total_records / metrics.total_duration_sec
                if metrics.total_duration_sec > 0 else 0
            )

            summary[key] = {
                "requests": metrics.total_requests,
                "success_rate": f"{success_rate:.1f}%",
                "records": metrics.total_records,
                "avg_latency_ms": f"{metrics.avg_latency_ms:.1f}",
                "rate_limit_hits": metrics.rate_limit_hits,
                "throughput_rps": f"{throughput:.1f}"
            }

        return summary

    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        if not self.process:
            return {}

        try:
            mem_info = self.process.memory_info()
            return {
                "cpu_percent": self.process.cpu_percent(),
                "memory_mb": mem_info.rss / 1024 / 1024,
                "memory_percent": self.process.memory_percent(),
                "num_threads": self.process.num_threads(),
                "open_files": len(self.process.open_files()) if hasattr(self.process, 'open_files') else 0
            }
        except:
            return {}

    def save_reports(self, output_dir: Path) -> None:
        """Save comprehensive reports."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Comprehensive monitoring report
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "progress": self.get_progress_summary(),
            "errors": self.get_error_summary(),
            "performance": self.get_performance_summary(),
            "resources": self.get_resource_usage()
        }

        report_file = output_dir / "monitoring_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"BATCH: Monitoring report saved to {report_file}")

    def print_summary(self) -> None:
        """Print summary to console."""
        print("\n" + "="*80)
        print("PHASE 1 MONITORING SUMMARY")
        print("="*80)

        progress = self.get_progress_summary()
        print(f"\nBATCH: Progress:")
        print(f" Status: {progress.get('status', 'unknown')}")
        print(f" Tasks: {progress.get('completed', 0)}/{progress.get('total_tasks', 0)}")
        print(f" Success Rate: {progress.get('success_rate', '0%')}")
        print(f" Duration: {progress.get('duration_formatted', '0m 0s')}")

        errors = self.get_error_summary()
        print(f"\nERRORS:")
        print(f" Total: {errors.get('total', 0)}")
        if errors.get('by_severity'):
            print(f" By Severity: {errors['by_severity']}")
        if errors.get('blacklisted_venues'):
            print(f" Blacklisted: {errors['blacklisted_venues']}")

        perf = self.get_performance_summary()
        if perf:
            print(f"\nPERFORMANCE - Top Results:")
            sorted_perf = sorted(
                perf.items(),
                key=lambda x: int(x[1].get('records', 0)),
                reverse=True
            )[:5]
            for venue_dt, metrics in sorted_perf:
                print(f" {venue_dt}: {metrics.get('records', 0)} records, "
                      f"{metrics.get('success_rate', '0%')} success")

        resources = self.get_resource_usage()
        if resources:
            print(f"\nRESOURCES:")
            print(f" CPU: {resources.get('cpu_percent', 0):.1f}%")
            print(f" Memory: {resources.get('memory_mb', 0):.1f} MB "
                  f"({resources.get('memory_percent', 0):.1f}%)")
            print(f" Threads: {resources.get('num_threads', 0)}")

        print("\n" + "="*80 + "\n")

# =============================================================================
# GLOBAL MONITOR INSTANCE
# =============================================================================

# Singleton instance - automatically available across all modules
_monitor = IntegratedMonitor()

def get_monitor() -> IntegratedMonitor:
    """Get the global monitor instance."""
    return _monitor

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def record_error(venue: str, data_type: str, error_msg: str,
                symbol: Optional[str] = None, exc: Optional[Exception] = None) -> None:
    """Convenience function to record an error."""
    _monitor.record_error(venue, data_type, error_msg, symbol, exc)

def record_success(venue: str, data_type: str) -> None:
    """Convenience function to record success."""
    _monitor.record_success(venue, data_type)

def update_metrics(venue: str, data_type: str, records: int,
                  latency_ms: float, success: bool, rate_limited: bool = False) -> None:
    """Convenience function to update metrics."""
    _monitor.update_metrics(venue, data_type, records, latency_ms, success, rate_limited)

def is_blacklisted(venue: str) -> bool:
    """Check if venue is blacklisted."""
    return _monitor.is_blacklisted(venue)

__all__ = [
    'IntegratedMonitor',
    'ErrorCategory',
    'ErrorSeverity',
    'get_monitor',
    'record_error',
    'record_success',
    'update_metrics',
    'is_blacklisted'
]
