"""
Progress Tracking System

Multi-level progress tracking with visual output for all pipeline phases.
Supports colored console progress bars, ETA estimation, speed metrics,
memory monitoring, tqdm/rich integration, and thread-safe parallel updates.

Phases supported:
    Phase 1: Pairs Trading
    Phase 3: Futures Curve Trading
"""

from __future__ import annotations

import os
import sys
import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
import json

logger = logging.getLogger(__name__)

# Check for tqdm availability
try:
    from tqdm import tqdm
    from tqdm.auto import tqdm as tqdm_auto
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False
    tqdm = None

# Check for rich availability (optional enhanced display)
try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
        TaskProgressColumn, TransferSpeedColumn
    )
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.layout import Layout
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False
    Console = None


# =============================================================================
# PROGRESS STATUS TYPES
# =============================================================================

class ProgressStatus(Enum):
    """Status of a progress task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CACHED = "cached"


class ProgressPhase(Enum):
    """Major phases for crypto statarb."""
    # Phase 1: Pairs Trading
    PHASE1_DATA_LOAD = "Phase 1: Data Loading"
    PHASE1_COINTEGRATION = "Phase 1: Cointegration Analysis"
    PHASE1_UNIVERSE = "Phase 1: Universe Construction"
    PHASE1_BACKTEST = "Phase 1: Backtesting"
    PHASE1_ENHANCEMENT = "Phase 1: ML Enhancement"
    PHASE1_WALK_FORWARD = "Phase 1: Walk-Forward Optimization"

    # Phase 3: Futures Curve
    PHASE3_DATA_LOAD = "Phase 3: Data Loading"
    PHASE3_TERM_STRUCTURE = "Phase 3: Term Structure Analysis"
    PHASE3_STRATEGY_A = "Phase 3: Strategy A (Calendar Spreads)"
    PHASE3_STRATEGY_B = "Phase 3: Strategy B (Cross-Venue Arb)"
    PHASE3_STRATEGY_C = "Phase 3: Strategy C (Synthetic Futures)"
    PHASE3_STRATEGY_D = "Phase 3: Strategy D (Roll Optimization)"
    PHASE3_BACKTEST = "Phase 3: Backtesting"
    PHASE3_WALK_FORWARD = "Phase 3: Walk-Forward Optimization"
    PHASE3_METRICS = "Phase 3: Performance Metrics"


# =============================================================================
# PROGRESS METRICS
# =============================================================================

@dataclass
class ProgressMetrics:
    """Real-time metrics for a progress task."""
    items_total: int = 0
    items_completed: int = 0
    items_failed: int = 0
    items_skipped: int = 0
    items_cached: int = 0

    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    last_update: Optional[float] = None

    # Speed metrics
    items_per_second: float = 0.0
    recent_items_per_second: float = 0.0  # Last 10 items

    # Memory metrics (MB)
    memory_start: float = 0.0
    memory_current: float = 0.0
    memory_peak: float = 0.0

    # Custom metrics
    custom: Dict[str, Any] = field(default_factory=dict)

    @property
    def progress_pct(self) -> float:
        """Get completion percentage."""
        if self.items_total == 0:
            return 0.0
        return (self.items_completed / self.items_total) * 100.0

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.started_at is None:
            return 0.0
        end = self.completed_at or time.time()
        return end - self.started_at

    @property
    def eta_seconds(self) -> float:
        """Estimate remaining time in seconds."""
        if self.items_per_second <= 0 or self.items_total == 0:
            return 0.0
        remaining = self.items_total - self.items_completed
        return remaining / self.items_per_second

    @property
    def cache_hit_rate(self) -> float:
        """
        Get cache hit rate.

        items_cached is a subset of items_completed (cached items count as completed).
        cache_hit_rate = items_cached / items_completed * 100
        """
        if self.items_completed == 0:
            return 0.0
        return (self.items_cached / self.items_completed) * 100.0

    def update_speed(self) -> None:
        """Update speed calculations."""
        if self.started_at is None or self.elapsed_seconds == 0:
            return
        self.items_per_second = self.items_completed / self.elapsed_seconds


@dataclass
class ProgressTask:
    """A single progress task."""
    task_id: str
    name: str
    phase: Optional[ProgressPhase] = None
    status: ProgressStatus = ProgressStatus.PENDING
    metrics: ProgressMetrics = field(default_factory=ProgressMetrics)
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    description: str = ""
    error_message: Optional[str] = None


# =============================================================================
# ASCII PROGRESS BAR
# =============================================================================

def ascii_progress_bar(current: int, total: int, width: int = 40,
                       fill: str = '█', empty: str = '░',
                       prefix: str = '', suffix: str = '') -> str:
    """
    Create an ASCII progress bar string.

    Args:
        current: Current progress value
        total: Total value
        width: Width of the bar in characters
        fill: Character for filled portion
        empty: Character for empty portion
        prefix: Text before the bar
        suffix: Text after the bar

    Returns:
        Formatted progress bar string
    """
    if total == 0:
        pct = 0.0
    else:
        pct = min(1.0, current / total)

    filled_width = int(width * pct)
    bar = fill * filled_width + empty * (width - filled_width)
    pct_str = f"{pct*100:5.1f}%"

    return f"{prefix}|{bar}| {pct_str} ({current}/{total}) {suffix}"


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m}m {s}s"


def format_number(n: float) -> str:
    """Format large numbers with K/M suffixes."""
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif abs(n) >= 1_000:
        return f"{n/1_000:.2f}K"
    else:
        return f"{n:.2f}"


# =============================================================================
# CONSOLE PROGRESS DISPLAY
# =============================================================================

class ConsoleProgressDisplay:
    """
    Console-based progress display with colors and formatting.

    Features:
    - Multi-line progress bars
    - Real-time metrics
    - Color coding by status
    - Nested task support
    """

    # ANSI color codes
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'magenta': '\033[95m',
        'gray': '\033[90m',
    }

    STATUS_COLORS = {
        ProgressStatus.PENDING: 'gray',
        ProgressStatus.RUNNING: 'blue',
        ProgressStatus.COMPLETED: 'green',
        ProgressStatus.FAILED: 'red',
        ProgressStatus.SKIPPED: 'yellow',
        ProgressStatus.CACHED: 'cyan',
    }

    def __init__(self, use_colors: bool = True, bar_width: int = 40):
        """Initialize console display."""
        self.use_colors = use_colors and sys.stdout.isatty()
        self.bar_width = bar_width
        self._lock = threading.RLock()

    def color(self, text: str, color_name: str) -> str:
        """Apply color to text."""
        if not self.use_colors or color_name not in self.COLORS:
            return text
        return f"{self.COLORS[color_name]}{text}{self.COLORS['reset']}"

    def status_icon(self, status: ProgressStatus) -> str:
        """Get status icon."""
        icons = {
            ProgressStatus.PENDING: '-',
            ProgressStatus.RUNNING: '>',
            ProgressStatus.COMPLETED: '+',
            ProgressStatus.FAILED: 'x',
            ProgressStatus.SKIPPED: 'o',
            ProgressStatus.CACHED: '~',
        }
        icon = icons.get(status, '?')
        color = self.STATUS_COLORS.get(status, 'reset')
        return self.color(icon, color)

    def format_task_line(self, task: ProgressTask, indent: int = 0) -> str:
        """Format a single task line."""
        prefix = '  ' * indent
        icon = self.status_icon(task.status)
        name = task.name

        m = task.metrics
        progress = ascii_progress_bar(
            m.items_completed, m.items_total,
            width=self.bar_width,
            prefix='',
            suffix=''
        )

        # Additional info
        info_parts = []
        if m.items_cached > 0:
            info_parts.append(self.color(f"cached:{m.items_cached}", 'cyan'))
        if m.items_failed > 0:
            info_parts.append(self.color(f"failed:{m.items_failed}", 'red'))
        if m.elapsed_seconds > 0:
            info_parts.append(f"elapsed:{format_time(m.elapsed_seconds)}")
        if m.eta_seconds > 0 and task.status == ProgressStatus.RUNNING:
            info_parts.append(f"ETA:{format_time(m.eta_seconds)}")
        if m.items_per_second > 0:
            info_parts.append(f"{m.items_per_second:.1f}/s")

        info = ' '.join(info_parts)

        # Build line
        status_color = self.STATUS_COLORS.get(task.status, 'reset')
        name_colored = self.color(name, status_color) if task.status == ProgressStatus.RUNNING else name

        return f"{prefix}{icon} {name_colored}\n{prefix}   {progress}\n{prefix}   {info}"

    def print_task(self, task: ProgressTask, indent: int = 0) -> None:
        """Print task progress to console."""
        with self._lock:
            line = self.format_task_line(task, indent)
            print(line)

    def print_summary(self, tasks: List[ProgressTask]) -> None:
        """Print summary of all tasks."""
        with self._lock:
            print("\n" + "=" * 60)
            print(self.color("PROGRESS SUMMARY", 'bold'))
            print("=" * 60)

            total_items = sum(t.metrics.items_total for t in tasks)
            completed_items = sum(t.metrics.items_completed for t in tasks)
            cached_items = sum(t.metrics.items_cached for t in tasks)
            failed_items = sum(t.metrics.items_failed for t in tasks)

            print(f"Total Tasks: {len(tasks)}")
            print(f"Total Items: {format_number(total_items)}")
            print(f"Completed:   {format_number(completed_items)} ({completed_items/total_items*100:.1f}%)" if total_items > 0 else "")
            print(f"From Cache:  {format_number(cached_items)} ({cached_items/total_items*100:.1f}%)" if total_items > 0 else "")
            if failed_items > 0:
                print(self.color(f"Failed:      {format_number(failed_items)}", 'red'))

            total_elapsed = sum(t.metrics.elapsed_seconds for t in tasks)
            print(f"Total Time:  {format_time(total_elapsed)}")
            print("=" * 60 + "\n")


# =============================================================================
# PROGRESS TRACKER
# =============================================================================

class AdvancedProgressTracker:
    """
    Multi-level progress tracker with real-time speed, ETA, and memory metrics.
    Thread-safe for use in parallel processing pipelines.
    """

    def __init__(self, use_tqdm: bool = True, use_rich: bool = False,
                 update_interval: float = 0.5, log_level: int = logging.INFO):
        """
        Initialize progress tracker.

        Args:
            use_tqdm: Use tqdm progress bars if available
            use_rich: Use rich console if available (takes precedence)
            update_interval: Minimum seconds between display updates
            log_level: Logging level for progress messages
        """
        self.use_tqdm = use_tqdm and _TQDM_AVAILABLE
        self.use_rich = use_rich and _RICH_AVAILABLE
        self.update_interval = update_interval
        self.log_level = log_level

        self._tasks: Dict[str, ProgressTask] = {}
        self._task_order: List[str] = []
        self._lock = threading.RLock()
        self._display = ConsoleProgressDisplay()
        self._callbacks: List[Callable] = []
        self._last_update = 0.0

        # tqdm bars
        self._tqdm_bars: Dict[str, Any] = {}

        # Rich progress
        self._rich_console = Console() if self.use_rich else None
        self._rich_progress = None

        # Speed calculation
        self._recent_timestamps: Dict[str, List[float]] = {}

    def create_task(self, task_id: str, name: str, total: int,
                   phase: ProgressPhase = None, parent_id: str = None,
                   description: str = "") -> ProgressTask:
        """
        Create a new progress task.

        Args:
            task_id: Unique identifier for the task
            name: Display name
            total: Total number of items to process
            phase: Optional phase classification
            parent_id: Optional parent task ID for nesting
            description: Optional description

        Returns:
            Created ProgressTask
        """
        with self._lock:
            task = ProgressTask(
                task_id=task_id,
                name=name,
                phase=phase,
                status=ProgressStatus.PENDING,
                parent_id=parent_id,
                description=description,
            )
            task.metrics.items_total = total

            self._tasks[task_id] = task
            self._task_order.append(task_id)

            # Link to parent
            if parent_id and parent_id in self._tasks:
                self._tasks[parent_id].children.append(task_id)

            # Initialize tqdm bar
            if self.use_tqdm:
                self._tqdm_bars[task_id] = tqdm(
                    total=total,
                    desc=name,
                    unit='items',
                    leave=True,
                    dynamic_ncols=True,
                )

            return task

    def start_task(self, task_id: str) -> None:
        """Mark task as started."""
        with self._lock:
            if task_id not in self._tasks:
                return

            task = self._tasks[task_id]
            task.status = ProgressStatus.RUNNING
            task.metrics.started_at = time.time()
            task.metrics.memory_start = self._get_memory_mb()

            self._recent_timestamps[task_id] = []

            self._notify_update(task)

    def update_task(self, task_id: str, completed: int = None,
                   increment: int = None, cached: int = 0,
                   failed: int = 0, custom: Dict = None) -> None:
        """
        Update task progress.

        Args:
            task_id: Task to update
            completed: Set absolute completed count
            increment: Increment completed count by this amount
            cached: Number of items loaded from cache
            failed: Number of failed items
            custom: Custom metrics to update
        """
        with self._lock:
            if task_id not in self._tasks:
                return

            task = self._tasks[task_id]
            m = task.metrics

            # Update completed count
            prev_completed = m.items_completed
            if completed is not None:
                m.items_completed = completed
            elif increment is not None:
                m.items_completed += increment

            m.items_cached += cached
            m.items_failed += failed

            if custom:
                m.custom.update(custom)

            m.last_update = time.time()

            # Track recent completions for speed calculation
            if m.items_completed > prev_completed:
                self._recent_timestamps[task_id].append(time.time())
                # Keep only last 20 timestamps
                self._recent_timestamps[task_id] = self._recent_timestamps[task_id][-20:]

            # Update speed metrics
            m.update_speed()
            if len(self._recent_timestamps[task_id]) >= 2:
                ts = self._recent_timestamps[task_id]
                time_span = ts[-1] - ts[0]
                if time_span > 0:
                    m.recent_items_per_second = len(ts) / time_span

            # Update memory
            m.memory_current = self._get_memory_mb()
            m.memory_peak = max(m.memory_peak, m.memory_current)

            # Update tqdm
            if self.use_tqdm and task_id in self._tqdm_bars:
                bar = self._tqdm_bars[task_id]
                bar.n = m.items_completed
                bar.set_postfix({
                    'cached': m.items_cached,
                    'failed': m.items_failed,
                    'speed': f'{m.items_per_second:.1f}/s'
                })
                bar.refresh()

            # Throttled display update
            if time.time() - self._last_update > self.update_interval:
                self._notify_update(task)
                self._last_update = time.time()

    def complete_task(self, task_id: str, success: bool = True,
                     error_message: str = None) -> None:
        """Mark task as completed or failed."""
        with self._lock:
            if task_id not in self._tasks:
                return

            task = self._tasks[task_id]
            task.metrics.completed_at = time.time()

            if success:
                task.status = ProgressStatus.COMPLETED
            else:
                task.status = ProgressStatus.FAILED
                task.error_message = error_message

            # Close tqdm bar
            if self.use_tqdm and task_id in self._tqdm_bars:
                bar = self._tqdm_bars[task_id]
                bar.n = task.metrics.items_completed
                bar.close()
                del self._tqdm_bars[task_id]

            self._notify_update(task)

    def skip_task(self, task_id: str, reason: str = "cached") -> None:
        """Mark task as skipped (e.g., loaded from cache)."""
        with self._lock:
            if task_id not in self._tasks:
                return

            task = self._tasks[task_id]
            task.status = ProgressStatus.SKIPPED if reason != "cached" else ProgressStatus.CACHED
            task.metrics.completed_at = time.time()

            # Close tqdm bar
            if self.use_tqdm and task_id in self._tqdm_bars:
                self._tqdm_bars[task_id].close()
                del self._tqdm_bars[task_id]

            self._notify_update(task)

    def get_task(self, task_id: str) -> Optional[ProgressTask]:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[ProgressTask]:
        """Get all tasks in order."""
        return [self._tasks[tid] for tid in self._task_order if tid in self._tasks]

    @contextmanager
    def track(self, task_id: str, name: str, total: int,
             phase: ProgressPhase = None):
        """
        Context manager for tracking a task.

        Usage:
            with tracker.track('my_task', 'Processing items', 100) as update:
                for item in items:
                    process(item)
                    update(1)
        """
        self.create_task(task_id, name, total, phase)
        self.start_task(task_id)

        def update_fn(increment: int = 1, cached: int = 0, failed: int = 0):
            self.update_task(task_id, increment=increment, cached=cached, failed=failed)

        try:
            yield update_fn
            self.complete_task(task_id, success=True)
        except Exception as e:
            self.complete_task(task_id, success=False, error_message=str(e))
            raise

    def add_callback(self, callback: Callable[[ProgressTask], None]) -> None:
        """Add callback for progress updates."""
        self._callbacks.append(callback)

    def _notify_update(self, task: ProgressTask) -> None:
        """Notify callbacks of update."""
        for cb in self._callbacks:
            try:
                cb(task)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0

    def print_summary(self) -> None:
        """Print summary of all tasks."""
        self._display.print_summary(self.get_all_tasks())

    def get_summary_dict(self) -> Dict[str, Any]:
        """Get summary as dictionary."""
        tasks = self.get_all_tasks()
        return {
            'total_tasks': len(tasks),
            'completed': sum(1 for t in tasks if t.status == ProgressStatus.COMPLETED),
            'failed': sum(1 for t in tasks if t.status == ProgressStatus.FAILED),
            'cached': sum(1 for t in tasks if t.status == ProgressStatus.CACHED),
            'total_items': sum(t.metrics.items_total for t in tasks),
            'completed_items': sum(t.metrics.items_completed for t in tasks),
            'cached_items': sum(t.metrics.items_cached for t in tasks),
            'total_elapsed': sum(t.metrics.elapsed_seconds for t in tasks),
            'tasks': [
                {
                    'id': t.task_id,
                    'name': t.name,
                    'status': t.status.value,
                    'progress_pct': t.metrics.progress_pct,
                    'elapsed': t.metrics.elapsed_seconds,
                    'cached': t.metrics.items_cached,
                }
                for t in tasks
            ]
        }


# =============================================================================
# PHASE-SPECIFIC PROGRESS TRACKERS
# =============================================================================

class Phase1ProgressTracker(AdvancedProgressTracker):
    """
    Progress tracker specialized for Phase 1 Pairs Trading.

    Tracks:
    - Cointegration testing
    - Universe construction
    - Backtesting
    - ML enhancement
    - Walk-forward optimization
    """

    def track_cointegration(self, n_pairs: int) -> str:
        """Create cointegration testing task."""
        task_id = f"coint_{int(time.time())}"
        self.create_task(
            task_id=task_id,
            name="Cointegration Testing",
            total=n_pairs,
            phase=ProgressPhase.PHASE1_COINTEGRATION,
            description=f"Testing {format_number(n_pairs)} pairs for cointegration"
        )
        return task_id

    def track_universe(self, n_symbols: int) -> str:
        """Create universe construction task."""
        task_id = f"universe_{int(time.time())}"
        self.create_task(
            task_id=task_id,
            name="Universe Construction",
            total=n_symbols,
            phase=ProgressPhase.PHASE1_UNIVERSE,
            description=f"Processing {n_symbols} symbols"
        )
        return task_id

    def track_backtest(self, n_windows: int) -> str:
        """Create backtest task."""
        task_id = f"backtest_{int(time.time())}"
        self.create_task(
            task_id=task_id,
            name="Backtesting",
            total=n_windows,
            phase=ProgressPhase.PHASE1_BACKTEST,
            description=f"Running {n_windows} backtest windows"
        )
        return task_id

    def track_walk_forward(self, n_windows: int) -> str:
        """Create walk-forward optimization task."""
        task_id = f"wf_{int(time.time())}"
        self.create_task(
            task_id=task_id,
            name="Walk-Forward Optimization",
            total=n_windows,
            phase=ProgressPhase.PHASE1_WALK_FORWARD,
            description=f"Optimizing {n_windows} walk-forward windows"
        )
        return task_id


class Phase3ProgressTracker(AdvancedProgressTracker):
    """
    Progress tracker specialized for Phase 3 Futures Curve Trading.

    Tracks:
    - Term structure analysis
    - Strategy execution (A, B, C, D)
    - Walk-forward optimization
    - Performance metrics calculation
    """

    def track_term_structure(self, n_timestamps: int) -> str:
        """Create term structure analysis task."""
        task_id = f"ts_{int(time.time())}"
        self.create_task(
            task_id=task_id,
            name="Term Structure Analysis",
            total=n_timestamps,
            phase=ProgressPhase.PHASE3_TERM_STRUCTURE,
            description=f"Analyzing {format_number(n_timestamps)} timestamps"
        )
        return task_id

    def track_strategy(self, strategy: str, n_days: int) -> str:
        """Create strategy execution task."""
        phases = {
            'A': ProgressPhase.PHASE3_STRATEGY_A,
            'B': ProgressPhase.PHASE3_STRATEGY_B,
            'C': ProgressPhase.PHASE3_STRATEGY_C,
            'D': ProgressPhase.PHASE3_STRATEGY_D,
        }
        task_id = f"strategy_{strategy}_{int(time.time())}"
        self.create_task(
            task_id=task_id,
            name=f"Strategy {strategy}",
            total=n_days,
            phase=phases.get(strategy, ProgressPhase.PHASE3_BACKTEST),
            description=f"Executing Strategy {strategy} for {n_days} days"
        )
        return task_id

    def track_walk_forward(self, n_windows: int) -> str:
        """Create walk-forward optimization task."""
        task_id = f"wf_{int(time.time())}"
        self.create_task(
            task_id=task_id,
            name="Walk-Forward Optimization",
            total=n_windows,
            phase=ProgressPhase.PHASE3_WALK_FORWARD,
            description=f"Optimizing {n_windows} walk-forward windows (18m train / 6m test)"
        )
        return task_id

    def track_metrics(self, n_metrics: int) -> str:
        """Create metrics calculation task."""
        task_id = f"metrics_{int(time.time())}"
        self.create_task(
            task_id=task_id,
            name="Performance Metrics",
            total=n_metrics,
            phase=ProgressPhase.PHASE3_METRICS,
            description=f"Calculating {n_metrics} performance metrics"
        )
        return task_id

    def track_crisis_analysis(self, n_events: int) -> str:
        """Create crisis analysis task."""
        task_id = f"crisis_{int(time.time())}"
        self.create_task(
            task_id=task_id,
            name="Crisis Event Analysis",
            total=n_events,
            phase=ProgressPhase.PHASE3_BACKTEST,
            description=f"Analyzing {n_events} crisis events (COVID, Luna, FTX, etc.)"
        )
        return task_id


# =============================================================================
# MULTI-PHASE PROGRESS TRACKER
# =============================================================================

class MultiPhaseProgressTracker:
    """
    Combined progress tracker for multiple phases.

    Provides unified tracking across Phase 1, 2, and 3.
    """

    def __init__(self):
        """Initialize multi-phase tracker."""
        self.phase1 = Phase1ProgressTracker()
        self.phase3 = Phase3ProgressTracker()
        self._current_phase = None

    def set_current_phase(self, phase: int) -> None:
        """Set current active phase."""
        self._current_phase = phase

    def get_tracker(self, phase: int = None) -> AdvancedProgressTracker:
        """Get tracker for specified phase."""
        phase = phase or self._current_phase
        if phase == 1:
            return self.phase1
        elif phase == 3:
            return self.phase3
        else:
            return AdvancedProgressTracker()

    def print_all_summaries(self) -> None:
        """Print summaries for all phases."""
        print("\n" + "=" * 70)
        print("PHASE 1 SUMMARY")
        self.phase1.print_summary()

        print("\n" + "=" * 70)
        print("PHASE 3 SUMMARY")
        self.phase3.print_summary()


# =============================================================================
# GLOBAL PROGRESS TRACKER INSTANCE
# =============================================================================

_GLOBAL_TRACKER: Optional[MultiPhaseProgressTracker] = None

def get_progress_tracker() -> MultiPhaseProgressTracker:
    """Get global progress tracker instance."""
    global _GLOBAL_TRACKER
    if _GLOBAL_TRACKER is None:
        _GLOBAL_TRACKER = MultiPhaseProgressTracker()
    return _GLOBAL_TRACKER

def get_phase1_tracker() -> Phase1ProgressTracker:
    """Get Phase 1 progress tracker."""
    return get_progress_tracker().phase1

def get_phase3_tracker() -> Phase3ProgressTracker:
    """Get Phase 3 progress tracker."""
    return get_progress_tracker().phase3


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Status types
    'ProgressStatus',
    'ProgressPhase',

    # Data classes
    'ProgressMetrics',
    'ProgressTask',

    # Trackers
    'AdvancedProgressTracker',
    'Phase1ProgressTracker',
    'Phase3ProgressTracker',
    'MultiPhaseProgressTracker',

    # Utilities
    'ascii_progress_bar',
    'format_time',
    'format_number',
    'ConsoleProgressDisplay',

    # Global access
    'get_progress_tracker',
    'get_phase1_tracker',
    'get_phase3_tracker',
]
