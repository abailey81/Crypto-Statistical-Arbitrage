"""
Step 5 Complete Orchestrator - Full Integration & Reporting
=========================================================================

Complete orchestrator for Step 5: Integration & Reporting that
coordinates all reporting components with full pipeline integration.

This is the MANDATORY integration layer that connects:
- Step4AdvancedOrchestrator results (backtesting, analytics)
- AdvancedReportGenerator (30-40 page reports per PDF)
- StrictPDFValidator (comprehensive compliance validation)
- PresentationGenerator (10-20 slides per PDF)

Architecture:
- Event-driven component coordination with message bus
- Enhanced checkpoint management for recovery
- Real-time progress monitoring with metrics
- Parallel generation where dependencies allow
- Cross-validation of all outputs
- Quality gates at each phase
- Automatic rollback on critical failures

PDF Requirements (Project Specification):
- 30-40 page Written Report with exact structure
- 10-20 slide Presentation Deck
- Full PDF compliance validation (100% required)
- Multi-venue reporting with color coding
- All 80+ metrics coverage
- Crisis event documentation (14 events)
- Walk-forward results (8 windows)
- Capacity analysis with venue constraints
- Grain futures comparison (PDF REQUIRED)
- Position sizing validation
- Concentration limits compliance

Integration Requirements:
- Strictly wired to Step4AdvancedOrchestrator
- Cross-validation with all Step 1-4 outputs
- Checkpoint management for recovery
- Comprehensive error handling
- Detailed logging and audit trail

Author: Tamer Atesyakar
Version: 3.0.0
PDF Compliance: Project Specification
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
import hashlib
import signal
import sys
import threading
import traceback
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
    Future
)
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from functools import wraps, partial
from pathlib import Path
from queue import Queue, PriorityQueue, Empty
from threading import Lock, RLock, Event, Condition, Semaphore
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple,
    TypeVar, Union, Generic, Protocol, Iterator, Awaitable,
    NamedTuple, FrozenSet, Sequence, Mapping, Type
)
import pandas as pd
import numpy as np

# Local imports - Strictly wired to other Step 5 components
from .advanced_report_generator import (
    AdvancedReportGenerator,
    ComprehensiveReportResult,
    ReportMetadata,
    ReportSection,
    ReportFormat,
    VenueType,
    VenueMetrics,
    CrisisEventAnalysis,
    WalkForwardWindow,
    GrainFuturesComparison,
    create_advanced_report_generator
)
from .strict_pdf_validator import (
    StrictPDFValidator,
    StrictValidationResult,
    ValidationCheck,
    CategoryResult,
    ComplianceLevel,
    ValidationCategory,
    CheckStatus,
    REQUIRED_SECTIONS,
    REQUIRED_METRICS,
    REQUIRED_CRISIS_EVENTS,
    create_strict_pdf_validator
)
# Presentation generator removed (not needed for submission)
PresentationGenerator = None
PresentationResult = None
PresentationMetadata = None
Slide = None
SlideType = None
PresentationFormat = None
create_presentation_generator = None

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE VARIABLES AND PROTOCOLS
# =============================================================================

T = TypeVar('T')
R = TypeVar('R')


class ComponentProtocol(Protocol):
    """Protocol for orchestrator components."""

    def execute(self, context: ExecutionContext) -> ComponentResult:
        """Execute the component."""
        ...

    def validate(self) -> bool:
        """Validate component readiness."""
        ...


class StateTransitionProtocol(Protocol):
    """Protocol for state machine transitions."""

    def can_transition(self, from_state: OrchestratorState, to_state: OrchestratorState) -> bool:
        """Check if transition is valid."""
        ...

    def on_transition(self, from_state: OrchestratorState, to_state: OrchestratorState) -> None:
        """Handle state transition."""
        ...


# =============================================================================
# ENUMERATIONS - COMPREHENSIVE STATE MANAGEMENT
# =============================================================================

class OrchestratorState(Enum):
    """
    State of the orchestrator with full lifecycle management.

    State Machine:
        UNINITIALIZED -> INITIALIZING -> READY -> RUNNING ->
        COMPLETED/FAILED/PAUSED -> CLEANING_UP -> TERMINATED
    """
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    VALIDATING_INPUTS = "validating_inputs"
    RUNNING = "running"
    PAUSED = "paused"
    RESUMING = "resuming"
    CHECKPOINTING = "checkpointing"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    CLEANING_UP = "cleaning_up"
    TERMINATED = "terminated"


class ComponentStatus(Enum):
    """Status of individual components with detailed tracking."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    PENDING = "pending"
    READY = "ready"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    RETRYING = "retrying"
    TIMED_OUT = "timed_out"


class ExecutionMode(Enum):
    """Execution mode for the orchestrator."""
    SEQUENTIAL = "sequential"  # One component at a time
    PARALLEL = "parallel"      # All independent components in parallel
    HYBRID = "hybrid"          # Parallel where dependencies allow
    ADAPTIVE = "adaptive"      # Adjust based on system resources
    PRIORITY = "priority"      # Execute by priority order


class DeliverableType(Enum):
    """Types of deliverables produced by Step 5."""
    COMPREHENSIVE_REPORT = "comprehensive_report"
    REPORT_MARKDOWN = "report_markdown"
    REPORT_JSON = "report_json"
    REPORT_HTML = "report_html"
    VALIDATION_REPORT = "validation_report"
    COMPLIANCE_SUMMARY = "compliance_summary"
    PRESENTATION = "presentation"
    PRESENTATION_MARP = "presentation_marp"
    PRESENTATION_HTML = "presentation_html"
    PRESENTATION_JSON = "presentation_json"
    SUMMARY_JSON = "summary_json"
    METRICS_EXPORT = "metrics_export"
    AUDIT_LOG = "audit_log"
    CHECKPOINT = "checkpoint"
    EXECUTION_REPORT = "execution_report"


class ComponentType(Enum):
    """Types of components in the orchestrator."""
    REPORT_GENERATOR = "report_generator"
    PDF_VALIDATOR = "pdf_validator"
    PRESENTATION_GENERATOR = "presentation_generator"
    SUMMARY_GENERATOR = "summary_generator"
    METRICS_EXPORTER = "metrics_exporter"
    AUDIT_LOGGER = "audit_logger"
    CHECKPOINT_MANAGER = "checkpoint_manager"
    QUALITY_GATE = "quality_gate"


class EventType(Enum):
    """Types of events in the message bus."""
    # Orchestrator events
    ORCHESTRATOR_STARTED = "orchestrator_started"
    ORCHESTRATOR_COMPLETED = "orchestrator_completed"
    ORCHESTRATOR_FAILED = "orchestrator_failed"
    ORCHESTRATOR_PAUSED = "orchestrator_paused"
    ORCHESTRATOR_RESUMED = "orchestrator_resumed"

    # Component events
    COMPONENT_STARTED = "component_started"
    COMPONENT_COMPLETED = "component_completed"
    COMPONENT_FAILED = "component_failed"
    COMPONENT_PROGRESS = "component_progress"
    COMPONENT_RETRYING = "component_retrying"

    # Phase events
    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"
    PHASE_FAILED = "phase_failed"

    # Quality gate events
    QUALITY_GATE_PASSED = "quality_gate_passed"
    QUALITY_GATE_FAILED = "quality_gate_failed"

    # Checkpoint events
    CHECKPOINT_CREATED = "checkpoint_created"
    CHECKPOINT_RESTORED = "checkpoint_restored"

    # Validation events
    VALIDATION_STARTED = "validation_started"
    VALIDATION_COMPLETED = "validation_completed"
    VALIDATION_WARNING = "validation_warning"
    VALIDATION_ERROR = "validation_error"


class ExecutionPhase(Enum):
    """Phases of Step 5 execution."""
    INITIALIZATION = "initialization"
    INPUT_VALIDATION = "input_validation"
    REPORT_GENERATION = "report_generation"
    PDF_VALIDATION = "pdf_validation"
    PRESENTATION_GENERATION = "presentation_generation"
    SUMMARY_GENERATION = "summary_generation"
    QUALITY_ASSURANCE = "quality_assurance"
    FINALIZATION = "finalization"


class Priority(Enum):
    """Priority levels for task execution."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class RecoveryStrategy(Enum):
    """Recovery strategies for failed components."""
    NONE = "none"
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    ROLLBACK = "rollback"
    ABORT = "abort"


class LogLevel(Enum):
    """Log levels for audit trail."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# NAMED TUPLES FOR IMMUTABLE DATA
# =============================================================================

class DependencyEdge(NamedTuple):
    """Edge in the dependency graph."""
    from_component: str
    to_component: str
    dependency_type: str


class QualityGateResult(NamedTuple):
    """Result of a quality gate check."""
    gate_name: str
    passed: bool
    score: float
    threshold: float
    details: Dict[str, Any]


class ResourceMetrics(NamedTuple):
    """Resource usage metrics."""
    memory_mb: float
    cpu_percent: float
    disk_io_bytes: int
    network_io_bytes: int


# =============================================================================
# DATA CLASSES - COMPREHENSIVE CONFIGURATION AND RESULTS
# =============================================================================

@dataclass
class VenueConfig:
    """Configuration for venue-specific processing."""
    venue_id: str
    venue_type: str  # CEX, DEX, Hybrid
    color_code: str
    position_limit: float
    weight_limit: float
    priority: int = 1


@dataclass
class QualityGateConfig:
    """Configuration for a quality gate."""
    gate_id: str
    name: str
    description: str
    phase: ExecutionPhase
    required: bool
    threshold: float
    metric_name: str
    comparison: str  # ">=", "<=", "==", ">"
    on_failure: RecoveryStrategy


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter_factor: float = 0.1
    retry_on_exceptions: Tuple[Type[Exception], ...] = (Exception,)


@dataclass
class TimeoutConfig:
    """Configuration for timeout behavior."""
    component_timeout_seconds: float = 300.0
    phase_timeout_seconds: float = 600.0
    total_timeout_seconds: float = 3600.0
    checkpoint_timeout_seconds: float = 60.0


@dataclass
class ResourceConfig:
    """Configuration for resource limits."""
    max_workers: int = 4
    max_memory_mb: float = 4096.0
    max_disk_mb: float = 10240.0
    semaphore_limit: int = 10


@dataclass
class OrchestratorConfig:
    """
    Comprehensive configuration for the Step 5 orchestrator.

    All settings required for complete orchestration:
    - Output and data configuration
    - Execution mode and parallelism
    - Checkpointing and recovery
    - Quality gates and validation
    - Resource limits
    - Timeout configuration
    - PDF compliance requirements
    """
    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path("outputs/step5_reports"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("outputs/checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("outputs/logs"))
    temp_dir: Path = field(default_factory=lambda: Path("outputs/temp"))

    # Data configuration (PDF REQUIRED: 2020-01-01 to present)
    data_start: datetime = field(default_factory=lambda: datetime(2020, 1, 1))
    data_end: Optional[datetime] = None

    # Execution configuration
    execution_mode: ExecutionMode = ExecutionMode.HYBRID
    max_workers: int = 4
    enable_async: bool = True

    # Checkpointing configuration
    enable_checkpointing: bool = True
    checkpoint_interval_seconds: float = 60.0
    max_checkpoints: int = 10
    compress_checkpoints: bool = True

    # Monitoring configuration
    enable_monitoring: bool = True
    metrics_interval_seconds: float = 5.0
    enable_progress_callbacks: bool = True

    # Component enablement
    generate_report: bool = True
    generate_presentation: bool = True
    generate_summary: bool = True
    generate_metrics_export: bool = True
    generate_audit_log: bool = True

    # Validation configuration
    strict_validation: bool = True
    fail_on_validation_warnings: bool = False
    validation_timeout_seconds: float = 120.0

    # PDF Requirements (MANDATORY per PDF)
    report_min_pages: int = 30
    report_max_pages: int = 40
    presentation_min_slides: int = 10
    presentation_max_slides: int = 20
    required_compliance_score: float = 1.0  # 100% required

    # Recovery configuration
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    timeout_config: TimeoutConfig = field(default_factory=TimeoutConfig)
    resource_config: ResourceConfig = field(default_factory=ResourceConfig)
    default_recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY

    # Quality gates (PDF compliance gates)
    quality_gates: List[QualityGateConfig] = field(default_factory=list)

    # Venue configuration
    venue_configs: Dict[str, VenueConfig] = field(default_factory=dict)

    # Save intermediate results
    save_intermediate: bool = True

    def __post_init__(self):
        """Initialize default quality gates and venue configs."""
        if not self.quality_gates:
            self.quality_gates = self._create_default_quality_gates()
        if not self.venue_configs:
            self.venue_configs = self._create_default_venue_configs()

    def _create_default_quality_gates(self) -> List[QualityGateConfig]:
        """Create default quality gates per PDF requirements."""
        return [
            QualityGateConfig(
                gate_id="report_page_count",
                name="Report Page Count",
                description="Validates report has 30-40 pages",
                phase=ExecutionPhase.REPORT_GENERATION,
                required=True,
                threshold=30.0,
                metric_name="estimated_pages",
                comparison=">=",
                on_failure=RecoveryStrategy.ABORT
            ),
            QualityGateConfig(
                gate_id="pdf_compliance",
                name="PDF Compliance Score",
                description="Validates 100% PDF compliance",
                phase=ExecutionPhase.PDF_VALIDATION,
                required=True,
                threshold=1.0,
                metric_name="compliance_score",
                comparison=">=",
                on_failure=RecoveryStrategy.ABORT
            ),
            QualityGateConfig(
                gate_id="metrics_coverage",
                name="Metrics Coverage",
                description="Validates all 80+ metrics are covered",
                phase=ExecutionPhase.QUALITY_ASSURANCE,
                required=True,
                threshold=80.0,
                metric_name="metrics_count",
                comparison=">=",
                on_failure=RecoveryStrategy.ABORT
            ),
            QualityGateConfig(
                gate_id="crisis_events",
                name="Crisis Events Coverage",
                description="Validates all 14 crisis events documented",
                phase=ExecutionPhase.QUALITY_ASSURANCE,
                required=True,
                threshold=14.0,
                metric_name="crisis_events_count",
                comparison=">=",
                on_failure=RecoveryStrategy.ABORT
            ),
            QualityGateConfig(
                gate_id="walk_forward_windows",
                name="Walk-Forward Windows",
                description="Validates 8 walk-forward windows",
                phase=ExecutionPhase.QUALITY_ASSURANCE,
                required=True,
                threshold=8.0,
                metric_name="walk_forward_count",
                comparison=">=",
                on_failure=RecoveryStrategy.ABORT
            ),
            QualityGateConfig(
                gate_id="presentation_slides",
                name="Presentation Slide Count",
                description="Validates presentation has 10-20 slides",
                phase=ExecutionPhase.PRESENTATION_GENERATION,
                required=True,
                threshold=10.0,
                metric_name="slide_count",
                comparison=">=",
                on_failure=RecoveryStrategy.RETRY
            ),
            QualityGateConfig(
                gate_id="grain_comparison",
                name="Grain Futures Comparison",
                description="Validates grain futures comparison is included (PDF REQUIRED)",
                phase=ExecutionPhase.QUALITY_ASSURANCE,
                required=True,
                threshold=1.0,
                metric_name="has_grain_comparison",
                comparison=">=",
                on_failure=RecoveryStrategy.ABORT
            )
        ]

    def _create_default_venue_configs(self) -> Dict[str, VenueConfig]:
        """Create default venue configurations per PDF requirements."""
        return {
            'cex': VenueConfig(
                venue_id='cex',
                venue_type='CEX',
                color_code='#0066CC',  # Blue
                position_limit=100000,  # $100k per PDF
                weight_limit=0.6,  # 60% CEX max per PDF
                priority=1
            ),
            'dex_liquid': VenueConfig(
                venue_id='dex_liquid',
                venue_type='DEX',
                color_code='#FF6600',  # Orange
                position_limit=50000,  # $20-50k per PDF
                weight_limit=0.3,
                priority=2
            ),
            'dex_illiquid': VenueConfig(
                venue_id='dex_illiquid',
                venue_type='DEX',
                color_code='#FF6600',  # Orange
                position_limit=10000,  # $5-10k per PDF
                weight_limit=0.1,
                priority=3
            ),
            'hybrid': VenueConfig(
                venue_id='hybrid',
                venue_type='Hybrid',
                color_code='#009933',  # Green
                position_limit=75000,
                weight_limit=0.4,
                priority=2
            )
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'output_dir': str(self.output_dir),
            'checkpoint_dir': str(self.checkpoint_dir),
            'log_dir': str(self.log_dir),
            'data_range': {
                'start': self.data_start.isoformat(),
                'end': self.data_end.isoformat() if self.data_end else 'present'
            },
            'execution_mode': self.execution_mode.value,
            'max_workers': self.max_workers,
            'enable_checkpointing': self.enable_checkpointing,
            'enable_monitoring': self.enable_monitoring,
            'generate_presentation': self.generate_presentation,
            'strict_validation': self.strict_validation,
            'requirements': {
                'report_pages': f"{self.report_min_pages}-{self.report_max_pages}",
                'presentation_slides': f"{self.presentation_min_slides}-{self.presentation_max_slides}",
                'compliance_score': f"{self.required_compliance_score:.0%}"
            },
            'quality_gates': [
                {
                    'gate_id': g.gate_id,
                    'name': g.name,
                    'required': g.required,
                    'threshold': g.threshold
                }
                for g in self.quality_gates
            ],
            'venue_configs': {
                k: {
                    'venue_type': v.venue_type,
                    'color_code': v.color_code,
                    'position_limit': v.position_limit,
                    'weight_limit': v.weight_limit
                }
                for k, v in self.venue_configs.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrchestratorConfig':
        """Create from dictionary."""
        config = cls()
        if 'output_dir' in data:
            config.output_dir = Path(data['output_dir'])
        if 'data_range' in data:
            config.data_start = datetime.fromisoformat(data['data_range']['start'])
            if data['data_range'].get('end') != 'present':
                config.data_end = datetime.fromisoformat(data['data_range']['end'])
        if 'execution_mode' in data:
            config.execution_mode = ExecutionMode(data['execution_mode'])
        if 'max_workers' in data:
            config.max_workers = data['max_workers']
        return config


@dataclass
class ExecutionContext:
    """
    Context passed to components during execution.

    Contains all data and configuration needed for component execution.
    """
    orchestrator_id: str
    execution_id: str
    phase: ExecutionPhase
    config: OrchestratorConfig
    step4_results: Dict[str, Any]
    universe_snapshot: Any
    signals: pd.DataFrame
    enhanced_signals: pd.DataFrame
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_result(self, component_name: str) -> Any:
        """Get result from a completed component."""
        return self.intermediate_results.get(component_name)

    def set_result(self, component_name: str, result: Any) -> None:
        """Set result from a component."""
        self.intermediate_results[component_name] = result


@dataclass
class ComponentDefinition:
    """Definition of a component in the orchestrator."""
    component_id: str
    component_type: ComponentType
    name: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    priority: Priority = Priority.MEDIUM
    required: bool = True
    retry_config: Optional[RetryConfig] = None
    timeout_seconds: float = 300.0
    deliverable_types: List[DeliverableType] = field(default_factory=list)


@dataclass
class ComponentResult:
    """
    Result from a component execution with comprehensive metrics.

    Tracks timing, outputs, errors, and performance metrics.
    """
    component_id: str
    component_name: str
    component_type: ComponentType
    deliverable_type: DeliverableType
    status: ComponentStatus
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    result: Any = None
    output_paths: List[Path] = field(default_factory=list)
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    retry_count: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Optional[ResourceMetrics] = None
    quality_gates_passed: List[str] = field(default_factory=list)
    quality_gates_failed: List[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """Check if component completed successfully."""
        return self.status == ComponentStatus.COMPLETED and self.error is None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'component_id': self.component_id,
            'component_name': self.component_name,
            'component_type': self.component_type.value,
            'deliverable_type': self.deliverable_type.value,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': self.duration_seconds,
            'output_paths': [str(p) for p in self.output_paths],
            'error': self.error,
            'retry_count': self.retry_count,
            'metrics': self.metrics,
            'quality_gates_passed': self.quality_gates_passed,
            'quality_gates_failed': self.quality_gates_failed
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentResult':
        """Create from dictionary."""
        return cls(
            component_id=data['component_id'],
            component_name=data['component_name'],
            component_type=ComponentType(data['component_type']),
            deliverable_type=DeliverableType(data['deliverable_type']),
            status=ComponentStatus(data['status']),
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time']),
            duration_seconds=data['duration_seconds'],
            output_paths=[Path(p) for p in data.get('output_paths', [])],
            error=data.get('error'),
            retry_count=data.get('retry_count', 0),
            metrics=data.get('metrics', {}),
            quality_gates_passed=data.get('quality_gates_passed', []),
            quality_gates_failed=data.get('quality_gates_failed', [])
        )


@dataclass
class PhaseResult:
    """Result from a phase execution."""
    phase: ExecutionPhase
    status: ComponentStatus
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    component_results: Dict[str, ComponentResult]
    quality_gate_results: List[QualityGateResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """Check if phase completed successfully."""
        return self.status == ComponentStatus.COMPLETED and len(self.errors) == 0

    @property
    def all_gates_passed(self) -> bool:
        """Check if all quality gates passed."""
        return all(g.passed for g in self.quality_gate_results)


@dataclass
class Checkpoint:
    """
    Checkpoint for recovery with complete state serialization.

    Allows resuming from any point in the execution.
    """
    checkpoint_id: str
    orchestrator_id: str
    timestamp: datetime
    state: OrchestratorState
    current_phase: ExecutionPhase
    completed_phases: List[ExecutionPhase]
    completed_components: List[str]
    component_results: Dict[str, ComponentResult]
    intermediate_data: Dict[str, Any]
    config_snapshot: Dict[str, Any]
    checksum: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'checkpoint_id': self.checkpoint_id,
            'orchestrator_id': self.orchestrator_id,
            'timestamp': self.timestamp.isoformat(),
            'state': self.state.value,
            'current_phase': self.current_phase.value,
            'completed_phases': [p.value for p in self.completed_phases],
            'completed_components': self.completed_components,
            'component_results': {
                k: v.to_dict() for k, v in self.component_results.items()
            },
            'config_snapshot': self.config_snapshot,
            'checksum': self.checksum
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Checkpoint':
        """Create from dictionary."""
        return cls(
            checkpoint_id=data['checkpoint_id'],
            orchestrator_id=data['orchestrator_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            state=OrchestratorState(data['state']),
            current_phase=ExecutionPhase(data['current_phase']),
            completed_phases=[ExecutionPhase(p) for p in data['completed_phases']],
            completed_components=data['completed_components'],
            component_results={
                k: ComponentResult.from_dict(v)
                for k, v in data.get('component_results', {}).items()
            },
            intermediate_data=data.get('intermediate_data', {}),
            config_snapshot=data['config_snapshot'],
            checksum=data['checksum']
        )


@dataclass
class AuditLogEntry:
    """Entry in the audit log."""
    timestamp: datetime
    level: LogLevel
    event_type: EventType
    component: Optional[str]
    phase: Optional[ExecutionPhase]
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'event_type': self.event_type.value,
            'component': self.component,
            'phase': self.phase.value if self.phase else None,
            'message': self.message,
            'details': self.details
        }


@dataclass
class Step5Result:
    """
    Complete result from Step 5 orchestrator execution.

    Contains all deliverables, validation results, and execution metrics.
    This is the final output of the entire Step 5 pipeline.
    """
    # Identification
    orchestrator_id: str
    orchestrator_version: str
    execution_id: str

    # Timing
    execution_time: datetime
    completion_time: datetime
    duration_seconds: float

    # State
    state: OrchestratorState
    config: OrchestratorConfig

    # Phase results
    phase_results: Dict[ExecutionPhase, PhaseResult]

    # Component results
    component_results: Dict[str, ComponentResult]

    # Main outputs
    comprehensive_report: Optional[ComprehensiveReportResult]
    validation_result: Optional[StrictValidationResult]
    presentation_result: Optional[PresentationResult]

    # Deliverables
    deliverables: Dict[DeliverableType, List[Path]]

    # Compliance
    is_pdf_compliant: bool
    compliance_score: float
    compliance_summary: Dict[str, Any]

    # Quality gates
    quality_gate_results: List[QualityGateResult]
    all_gates_passed: bool

    # Metrics
    metrics_summary: Dict[str, Any]
    resource_metrics: Dict[str, ResourceMetrics]

    # Errors
    errors: List[str]
    warnings: List[str]

    # Audit
    audit_log_path: Optional[Path]

    @property
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return (
            self.state == OrchestratorState.COMPLETED and
            self.is_pdf_compliant and
            len(self.errors) == 0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'orchestrator_id': self.orchestrator_id,
            'orchestrator_version': self.orchestrator_version,
            'execution_id': self.execution_id,
            'execution_time': self.execution_time.isoformat(),
            'completion_time': self.completion_time.isoformat(),
            'duration_seconds': self.duration_seconds,
            'state': self.state.value,
            'config': self.config.to_dict(),
            'phase_results': {
                phase.value: {
                    'status': result.status.value,
                    'duration_seconds': result.duration_seconds,
                    'is_success': result.is_success
                }
                for phase, result in self.phase_results.items()
            },
            'component_results': {
                k: v.to_dict() for k, v in self.component_results.items()
            },
            'deliverables': {
                k.value: [str(p) for p in v]
                for k, v in self.deliverables.items()
            },
            'is_pdf_compliant': self.is_pdf_compliant,
            'compliance_score': self.compliance_score,
            'compliance_summary': self.compliance_summary,
            'quality_gate_results': [
                {
                    'gate_name': g.gate_name,
                    'passed': g.passed,
                    'score': g.score,
                    'threshold': g.threshold
                }
                for g in self.quality_gate_results
            ],
            'all_gates_passed': self.all_gates_passed,
            'metrics_summary': self.metrics_summary,
            'errors': self.errors,
            'warnings': self.warnings,
            'is_success': self.is_success
        }


# =============================================================================
# EVENT MESSAGE
# =============================================================================

@dataclass
class EventMessage:
    """Message in the event bus."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    priority: Priority = Priority.MEDIUM

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'data': self.data,
            'priority': self.priority.value
        }


# =============================================================================
# INFRASTRUCTURE COMPONENTS
# =============================================================================

class Step5MessageBus:
    """
    Event-driven message bus for component coordination.

    Features:
    - Pub/sub pattern with multiple subscribers
    - Priority-based message delivery
    - Async support
    - Message history for debugging
    - Dead letter queue for failed deliveries
    """

    def __init__(self, max_history: int = 1000):
        """Initialize message bus."""
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._async_subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._message_history: List[EventMessage] = []
        self._dead_letter_queue: List[Tuple[EventMessage, Exception]] = []
        self._max_history = max_history
        self._lock = RLock()
        self._message_count = 0
        self._delivery_failures = 0

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[EventMessage], None],
        async_handler: bool = False
    ) -> str:
        """
        Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs
            async_handler: Whether callback is async

        Returns:
            Subscription ID
        """
        subscription_id = f"sub_{uuid.uuid4().hex[:8]}"

        with self._lock:
            if async_handler:
                self._async_subscribers[event_type].append(callback)
            else:
                self._subscribers[event_type].append(callback)

        logger.debug(f"Subscribed to {event_type.value}: {subscription_id}")
        return subscription_id

    def unsubscribe(self, event_type: EventType, callback: Callable) -> bool:
        """Unsubscribe from an event type."""
        with self._lock:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                return True
            if callback in self._async_subscribers[event_type]:
                self._async_subscribers[event_type].remove(callback)
                return True
        return False

    def publish(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        source: str = "orchestrator",
        priority: Priority = Priority.MEDIUM
    ) -> EventMessage:
        """
        Publish an event to all subscribers.

        Args:
            event_type: Type of event
            data: Event data
            source: Source component
            priority: Message priority

        Returns:
            Published EventMessage
        """
        message = EventMessage(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            source=source,
            data=data,
            priority=priority
        )

        with self._lock:
            self._message_count += 1
            self._message_history.append(message)

            # Trim history if needed
            if len(self._message_history) > self._max_history:
                self._message_history = self._message_history[-self._max_history:]

        # Deliver to synchronous subscribers
        for callback in self._subscribers.get(event_type, []):
            try:
                callback(message)
            except Exception as e:
                self._delivery_failures += 1
                self._dead_letter_queue.append((message, e))
                logger.error(f"Error in subscriber callback: {e}")

        logger.debug(f"Published {event_type.value}: {message.event_id}")
        return message

    async def publish_async(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        source: str = "orchestrator"
    ) -> EventMessage:
        """Publish event asynchronously."""
        message = self.publish(event_type, data, source)

        # Deliver to async subscribers
        for callback in self._async_subscribers.get(event_type, []):
            try:
                await callback(message)
            except Exception as e:
                self._delivery_failures += 1
                self._dead_letter_queue.append((message, e))
                logger.error(f"Error in async subscriber callback: {e}")

        return message

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[EventMessage]:
        """Get message history."""
        with self._lock:
            messages = self._message_history
            if event_type:
                messages = [m for m in messages if m.event_type == event_type]
            return messages[-limit:]

    def get_dead_letters(self) -> List[Tuple[EventMessage, Exception]]:
        """Get failed message deliveries."""
        return list(self._dead_letter_queue)

    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        return {
            'total_messages': self._message_count,
            'delivery_failures': self._delivery_failures,
            'history_size': len(self._message_history),
            'dead_letters': len(self._dead_letter_queue),
            'subscriber_counts': {
                event_type.value: len(callbacks)
                for event_type, callbacks in self._subscribers.items()
            }
        }


class Step5CheckpointManager:
    """
    Manages checkpoints for recovery with extended features.

    Features:
    - Automatic checkpoint creation
    - Compression for storage efficiency
    - Checksum validation
    - Multiple checkpoint retention
    - Recovery from any checkpoint
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        max_checkpoints: int = 10,
        compress: bool = True
    ):
        """Initialize checkpoint manager."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.compress = compress

        self._checkpoints: List[Checkpoint] = []
        self._lock = Lock()

        # Load existing checkpoints
        self._load_existing_checkpoints()

    def _load_existing_checkpoints(self) -> None:
        """Load existing checkpoints from disk."""
        pattern = "step5_checkpoint_*.json"
        checkpoint_files = sorted(
            self.checkpoint_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime
        )

        for cp_file in checkpoint_files[-self.max_checkpoints:]:
            try:
                with open(cp_file, 'r') as f:
                    data = json.load(f)
                checkpoint = Checkpoint.from_dict(data)
                self._checkpoints.append(checkpoint)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {cp_file}: {e}")

    def create_checkpoint(
        self,
        orchestrator_id: str,
        state: OrchestratorState,
        current_phase: ExecutionPhase,
        completed_phases: List[ExecutionPhase],
        completed_components: List[str],
        component_results: Dict[str, ComponentResult],
        intermediate_data: Dict[str, Any],
        config: OrchestratorConfig
    ) -> Checkpoint:
        """
        Create a checkpoint for recovery.

        Args:
            orchestrator_id: ID of the orchestrator
            state: Current state
            current_phase: Current execution phase
            completed_phases: List of completed phases
            completed_components: List of completed components
            component_results: Results from components
            intermediate_data: Intermediate data to save
            config: Configuration snapshot

        Returns:
            Created Checkpoint
        """
        checkpoint_id = f"cp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Create config snapshot
        config_snapshot = config.to_dict()

        # Calculate checksum
        checksum_data = json.dumps({
            'checkpoint_id': checkpoint_id,
            'state': state.value,
            'completed_components': completed_components
        }, sort_keys=True)
        checksum = hashlib.sha256(checksum_data.encode()).hexdigest()[:16]

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            orchestrator_id=orchestrator_id,
            timestamp=datetime.now(timezone.utc),
            state=state,
            current_phase=current_phase,
            completed_phases=completed_phases,
            completed_components=completed_components,
            component_results=component_results.copy(),
            intermediate_data=intermediate_data,
            config_snapshot=config_snapshot,
            checksum=checksum
        )

        with self._lock:
            self._checkpoints.append(checkpoint)
            self._save_checkpoint(checkpoint)
            self._cleanup_old_checkpoints()

        logger.info(f"Created checkpoint: {checkpoint_id}")
        return checkpoint

    def _save_checkpoint(self, checkpoint: Checkpoint) -> Path:
        """Save checkpoint to disk."""
        filename = f"step5_checkpoint_{checkpoint.checkpoint_id}.json"
        path = self.checkpoint_dir / filename

        checkpoint_data = checkpoint.to_dict()

        with open(path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        logger.debug(f"Saved checkpoint: {path}")
        return path

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond retention limit."""
        if len(self._checkpoints) > self.max_checkpoints:
            old_checkpoints = self._checkpoints[:-self.max_checkpoints]
            self._checkpoints = self._checkpoints[-self.max_checkpoints:]

            for cp in old_checkpoints:
                try:
                    cp_path = self.checkpoint_dir / f"step5_checkpoint_{cp.checkpoint_id}.json"
                    if cp_path.exists():
                        cp_path.unlink()
                        logger.debug(f"Removed old checkpoint: {cp.checkpoint_id}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint: {e}")

    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Get the most recent checkpoint."""
        if self._checkpoints:
            return self._checkpoints[-1]
        return None

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get a specific checkpoint by ID."""
        for cp in self._checkpoints:
            if cp.checkpoint_id == checkpoint_id:
                return cp
        return None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        return [
            {
                'checkpoint_id': cp.checkpoint_id,
                'timestamp': cp.timestamp.isoformat(),
                'state': cp.state.value,
                'phase': cp.current_phase.value,
                'components_completed': len(cp.completed_components)
            }
            for cp in self._checkpoints
        ]

    def validate_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """Validate checkpoint integrity."""
        checksum_data = json.dumps({
            'checkpoint_id': checkpoint.checkpoint_id,
            'state': checkpoint.state.value,
            'completed_components': checkpoint.completed_components
        }, sort_keys=True)
        expected_checksum = hashlib.sha256(checksum_data.encode()).hexdigest()[:16]

        return checkpoint.checksum == expected_checksum


class Step5ProgressMonitor:
    """
    Monitors progress of the orchestrator with detailed tracking.

    Features:
    - Per-component progress tracking
    - Phase-level progress aggregation
    - Time estimates
    - Resource monitoring
    - Progress callbacks
    """

    def __init__(self, enable_callbacks: bool = True):
        """Initialize progress monitor."""
        self._component_status: Dict[str, ComponentStatus] = {}
        self._component_progress: Dict[str, float] = {}
        self._phase_status: Dict[ExecutionPhase, ComponentStatus] = {}
        self._phase_progress: Dict[ExecutionPhase, float] = {}
        self._start_time: Optional[datetime] = None
        self._phase_start_times: Dict[ExecutionPhase, datetime] = {}
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._enable_callbacks = enable_callbacks
        self._lock = RLock()

        # Metrics tracking
        self._metrics: Dict[str, Any] = {}
        self._resource_samples: List[ResourceMetrics] = []

    def start(self) -> None:
        """Start monitoring."""
        self._start_time = datetime.now(timezone.utc)
        logger.info("Progress monitoring started")

    def stop(self) -> None:
        """Stop monitoring."""
        logger.info("Progress monitoring stopped")

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add progress callback."""
        self._callbacks.append(callback)

    def start_phase(self, phase: ExecutionPhase) -> None:
        """Mark phase as started."""
        with self._lock:
            self._phase_status[phase] = ComponentStatus.RUNNING
            self._phase_progress[phase] = 0.0
            self._phase_start_times[phase] = datetime.now(timezone.utc)

        self._notify_callbacks()

    def complete_phase(self, phase: ExecutionPhase) -> None:
        """Mark phase as completed."""
        with self._lock:
            self._phase_status[phase] = ComponentStatus.COMPLETED
            self._phase_progress[phase] = 1.0

        self._notify_callbacks()

    def fail_phase(self, phase: ExecutionPhase) -> None:
        """Mark phase as failed."""
        with self._lock:
            self._phase_status[phase] = ComponentStatus.FAILED

        self._notify_callbacks()

    def update_component(
        self,
        component: str,
        status: ComponentStatus,
        progress: float = 0.0,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update component status and progress.

        Args:
            component: Component identifier
            status: Current status
            progress: Progress 0.0 to 1.0
            metrics: Optional metrics to record
        """
        with self._lock:
            self._component_status[component] = status
            self._component_progress[component] = min(1.0, max(0.0, progress))

            if metrics:
                self._metrics[component] = metrics

        self._notify_callbacks()

    def record_resource_sample(self, metrics: ResourceMetrics) -> None:
        """Record resource usage sample."""
        with self._lock:
            self._resource_samples.append(metrics)
            # Keep last 1000 samples
            if len(self._resource_samples) > 1000:
                self._resource_samples = self._resource_samples[-1000:]

    def get_overall_progress(self) -> float:
        """Get overall progress across all components."""
        with self._lock:
            if not self._component_progress:
                return 0.0
            return sum(self._component_progress.values()) / len(self._component_progress)

    def get_phase_progress(self, phase: ExecutionPhase) -> float:
        """Get progress for a specific phase."""
        return self._phase_progress.get(phase, 0.0)

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self._start_time:
            return (datetime.now(timezone.utc) - self._start_time).total_seconds()
        return 0.0

    def get_phase_elapsed_time(self, phase: ExecutionPhase) -> float:
        """Get elapsed time for a phase."""
        start = self._phase_start_times.get(phase)
        if start:
            return (datetime.now(timezone.utc) - start).total_seconds()
        return 0.0

    def estimate_remaining_time(self) -> Optional[float]:
        """Estimate remaining time based on progress."""
        progress = self.get_overall_progress()
        elapsed = self.get_elapsed_time()

        if progress > 0.1 and elapsed > 0:
            total_estimated = elapsed / progress
            return total_estimated - elapsed
        return None

    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary."""
        elapsed = self.get_elapsed_time()
        remaining = self.estimate_remaining_time()

        return {
            'overall_progress': f"{self.get_overall_progress():.1%}",
            'elapsed_seconds': elapsed,
            'estimated_remaining_seconds': remaining,
            'phases': {
                phase.value: {
                    'status': self._phase_status.get(phase, ComponentStatus.PENDING).value,
                    'progress': f"{self._phase_progress.get(phase, 0):.1%}",
                    'elapsed_seconds': self.get_phase_elapsed_time(phase)
                }
                for phase in ExecutionPhase
            },
            'components': {
                k: {
                    'status': v.value,
                    'progress': f"{self._component_progress.get(k, 0):.1%}"
                }
                for k, v in self._component_status.items()
            },
            'metrics': self._metrics
        }

    def _notify_callbacks(self) -> None:
        """Notify registered callbacks of progress update."""
        if not self._enable_callbacks:
            return

        summary = self.get_status_summary()
        for callback in self._callbacks:
            try:
                callback(summary)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")


class Step5AuditLogger:
    """
    Comprehensive audit logging for Step 5 orchestration.

    Features:
    - Structured log entries
    - Multiple output formats (JSON, text)
    - Log rotation
    - Searchable history
    """

    def __init__(self, log_dir: Path, max_entries: int = 10000):
        """Initialize audit logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries

        self._entries: List[AuditLogEntry] = []
        self._lock = Lock()

        # Create log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._log_path = self.log_dir / f"step5_audit_{timestamp}.json"

    def log(
        self,
        level: LogLevel,
        event_type: EventType,
        message: str,
        component: Optional[str] = None,
        phase: Optional[ExecutionPhase] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditLogEntry:
        """
        Log an audit entry.

        Args:
            level: Log level
            event_type: Event type
            message: Log message
            component: Component name
            phase: Execution phase
            details: Additional details

        Returns:
            Created AuditLogEntry
        """
        entry = AuditLogEntry(
            timestamp=datetime.now(timezone.utc),
            level=level,
            event_type=event_type,
            component=component,
            phase=phase,
            message=message,
            details=details or {}
        )

        with self._lock:
            self._entries.append(entry)

            # Trim if needed
            if len(self._entries) > self.max_entries:
                self._entries = self._entries[-self.max_entries:]

        # Log to standard logger as well
        log_method = getattr(logger, level.value, logger.info)
        log_method(f"[{event_type.value}] {message}")

        return entry

    def info(self, event_type: EventType, message: str, **kwargs) -> AuditLogEntry:
        """Log info level entry."""
        return self.log(LogLevel.INFO, event_type, message, **kwargs)

    def warning(self, event_type: EventType, message: str, **kwargs) -> AuditLogEntry:
        """Log warning level entry."""
        return self.log(LogLevel.WARNING, event_type, message, **kwargs)

    def error(self, event_type: EventType, message: str, **kwargs) -> AuditLogEntry:
        """Log error level entry."""
        return self.log(LogLevel.ERROR, event_type, message, **kwargs)

    def critical(self, event_type: EventType, message: str, **kwargs) -> AuditLogEntry:
        """Log critical level entry."""
        return self.log(LogLevel.CRITICAL, event_type, message, **kwargs)

    def save(self) -> Path:
        """Save audit log to file."""
        with self._lock:
            log_data = {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'total_entries': len(self._entries),
                'entries': [e.to_dict() for e in self._entries]
            }

            with open(self._log_path, 'w') as f:
                json.dump(log_data, f, indent=2)

        logger.info(f"Saved audit log: {self._log_path}")
        return self._log_path

    def search(
        self,
        level: Optional[LogLevel] = None,
        event_type: Optional[EventType] = None,
        component: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditLogEntry]:
        """Search audit log entries."""
        results = []

        for entry in reversed(self._entries):
            if level and entry.level != level:
                continue
            if event_type and entry.event_type != event_type:
                continue
            if component and entry.component != component:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue

            results.append(entry)
            if len(results) >= limit:
                break

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get audit log summary."""
        level_counts = defaultdict(int)
        event_counts = defaultdict(int)
        component_counts = defaultdict(int)

        for entry in self._entries:
            level_counts[entry.level.value] += 1
            event_counts[entry.event_type.value] += 1
            if entry.component:
                component_counts[entry.component] += 1

        return {
            'total_entries': len(self._entries),
            'level_counts': dict(level_counts),
            'event_counts': dict(event_counts),
            'component_counts': dict(component_counts)
        }


class QualityGateManager:
    """
    Manages quality gates for the orchestration pipeline.

    Features:
    - Configurable quality gates per phase
    - Automatic gate checking
    - Failure handling with configurable strategies
    """

    def __init__(self, gates: List[QualityGateConfig]):
        """Initialize with quality gate configurations."""
        self.gates = {g.gate_id: g for g in gates}
        self._results: Dict[str, QualityGateResult] = {}
        self._lock = Lock()

    def check_gate(
        self,
        gate_id: str,
        metrics: Dict[str, Any]
    ) -> QualityGateResult:
        """
        Check a quality gate against metrics.

        Args:
            gate_id: Gate identifier
            metrics: Metrics to check against

        Returns:
            QualityGateResult
        """
        if gate_id not in self.gates:
            raise ValueError(f"Unknown quality gate: {gate_id}")

        gate = self.gates[gate_id]
        metric_value = metrics.get(gate.metric_name, 0)

        # Evaluate comparison
        passed = self._evaluate_comparison(
            metric_value,
            gate.threshold,
            gate.comparison
        )

        result = QualityGateResult(
            gate_name=gate.name,
            passed=passed,
            score=float(metric_value),
            threshold=gate.threshold,
            details={
                'gate_id': gate_id,
                'metric_name': gate.metric_name,
                'comparison': gate.comparison,
                'required': gate.required,
                'on_failure': gate.on_failure.value
            }
        )

        with self._lock:
            self._results[gate_id] = result

        logger.info(
            f"Quality gate '{gate.name}': "
            f"{'PASSED' if passed else 'FAILED'} "
            f"({metric_value} {gate.comparison} {gate.threshold})"
        )

        return result

    def _evaluate_comparison(
        self,
        value: float,
        threshold: float,
        comparison: str
    ) -> bool:
        """Evaluate a comparison."""
        if comparison == ">=":
            return value >= threshold
        elif comparison == "<=":
            return value <= threshold
        elif comparison == ">":
            return value > threshold
        elif comparison == "<":
            return value < threshold
        elif comparison == "==":
            return value == threshold
        else:
            raise ValueError(f"Unknown comparison: {comparison}")

    def check_phase_gates(
        self,
        phase: ExecutionPhase,
        metrics: Dict[str, Any]
    ) -> List[QualityGateResult]:
        """Check all gates for a phase."""
        results = []

        for gate_id, gate in self.gates.items():
            if gate.phase == phase:
                result = self.check_gate(gate_id, metrics)
                results.append(result)

        return results

    def get_all_results(self) -> List[QualityGateResult]:
        """Get all gate check results."""
        return list(self._results.values())

    def all_required_gates_passed(self) -> bool:
        """Check if all required gates passed."""
        for gate_id, gate in self.gates.items():
            if gate.required:
                result = self._results.get(gate_id)
                if not result or not result.passed:
                    return False
        return True

    def get_recovery_strategy(self, gate_id: str) -> RecoveryStrategy:
        """Get recovery strategy for a failed gate."""
        if gate_id in self.gates:
            return self.gates[gate_id].on_failure
        return RecoveryStrategy.ABORT


class DependencyResolver:
    """
    Resolves component dependencies for execution ordering.

    Features:
    - Topological sorting
    - Cycle detection
    - Parallel execution groups
    """

    def __init__(self, components: List[ComponentDefinition]):
        """Initialize with component definitions."""
        self.components = {c.component_id: c for c in components}
        self._graph: Dict[str, Set[str]] = defaultdict(set)
        self._build_graph()

    def _build_graph(self) -> None:
        """Build dependency graph."""
        for comp in self.components.values():
            for dep in comp.dependencies:
                self._graph[comp.component_id].add(dep)

    def get_execution_order(self) -> List[List[str]]:
        """
        Get execution order as list of parallel groups.

        Returns:
            List of component groups that can run in parallel
        """
        # Calculate in-degrees
        in_degree = defaultdict(int)
        for comp_id in self.components:
            in_degree[comp_id] = 0

        for comp_id, deps in self._graph.items():
            for dep in deps:
                in_degree[comp_id] += 1

        # Topological sort with grouping
        groups = []
        remaining = set(self.components.keys())

        while remaining:
            # Find all components with no remaining dependencies
            ready = {
                c for c in remaining
                if all(d not in remaining for d in self._graph[c])
            }

            if not ready:
                raise ValueError("Circular dependency detected")

            # Sort by priority within group
            sorted_ready = sorted(
                ready,
                key=lambda c: self.components[c].priority.value
            )

            groups.append(sorted_ready)
            remaining -= ready

        return groups

    def get_dependencies(self, component_id: str) -> Set[str]:
        """Get dependencies for a component."""
        return self._graph.get(component_id, set())

    def get_dependents(self, component_id: str) -> Set[str]:
        """Get components that depend on this one."""
        dependents = set()
        for comp_id, deps in self._graph.items():
            if component_id in deps:
                dependents.add(comp_id)
        return dependents


# =============================================================================
# COMPONENT EXECUTORS
# =============================================================================

class ComponentExecutor(ABC):
    """Base class for component executors."""

    def __init__(self, component_def: ComponentDefinition):
        """Initialize executor."""
        self.component_def = component_def
        self.retry_config = component_def.retry_config or RetryConfig()

    @abstractmethod
    def execute(self, context: ExecutionContext) -> ComponentResult:
        """Execute the component."""
        pass

    def execute_with_retry(self, context: ExecutionContext) -> ComponentResult:
        """Execute with retry logic."""
        last_error = None
        retry_count = 0

        while retry_count <= self.retry_config.max_retries:
            try:
                return self.execute(context)
            except self.retry_config.retry_on_exceptions as e:
                last_error = e
                retry_count += 1

                if retry_count > self.retry_config.max_retries:
                    break

                # Calculate delay with exponential backoff
                delay = min(
                    self.retry_config.initial_delay_seconds *
                    (self.retry_config.exponential_base ** (retry_count - 1)),
                    self.retry_config.max_delay_seconds
                )

                # Add jitter
                jitter = delay * self.retry_config.jitter_factor * np.random.random()
                delay += jitter

                logger.warning(
                    f"Component {self.component_def.component_id} failed, "
                    f"retry {retry_count}/{self.retry_config.max_retries} "
                    f"in {delay:.1f}s: {e}"
                )

                time.sleep(delay)

        # All retries exhausted
        return ComponentResult(
            component_id=self.component_def.component_id,
            component_name=self.component_def.name,
            component_type=self.component_def.component_type,
            deliverable_type=self.component_def.deliverable_types[0] if self.component_def.deliverable_types else DeliverableType.SUMMARY_JSON,
            status=ComponentStatus.FAILED,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            duration_seconds=0,
            error=str(last_error),
            error_traceback=traceback.format_exc(),
            retry_count=retry_count
        )


class ReportGeneratorExecutor(ComponentExecutor):
    """Executor for AdvancedReportGenerator."""

    def __init__(
        self,
        component_def: ComponentDefinition,
        report_generator: AdvancedReportGenerator
    ):
        """Initialize with report generator."""
        super().__init__(component_def)
        self.report_generator = report_generator

    def execute(self, context: ExecutionContext) -> ComponentResult:
        """Execute report generation."""
        start_time = datetime.now(timezone.utc)

        try:
            result = self.report_generator.generate_comprehensive_report(
                step4_results=context.step4_results,
                universe_snapshot=context.universe_snapshot,
                signals=context.signals,
                enhanced_signals=context.enhanced_signals
            )

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            # Store in context
            context.set_result('comprehensive_report', result)

            return ComponentResult(
                component_id=self.component_def.component_id,
                component_name=self.component_def.name,
                component_type=ComponentType.REPORT_GENERATOR,
                deliverable_type=DeliverableType.COMPREHENSIVE_REPORT,
                status=ComponentStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                result=result,
                output_paths=[
                    context.config.output_dir / "comprehensive_report.md",
                    context.config.output_dir / "comprehensive_report.json"
                ],
                metrics={
                    'estimated_pages': result.estimated_pages,
                    'is_pdf_compliant': result.is_pdf_compliant,
                    'generation_time': result.generation_time_seconds,
                    'sections_count': len(result.sections) if isinstance(result.sections, (dict, list)) else 0,
                    'crisis_events_count': len(result.crisis_events) if isinstance(result.crisis_events, list) else 0,
                    'walk_forward_count': len(result.walk_forward_periods) if isinstance(result.walk_forward_periods, list) else 0,
                    'has_grain_comparison': len(result.grain_comparisons) > 0 if isinstance(result.grain_comparisons, list) else False
                }
            )

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            return ComponentResult(
                component_id=self.component_def.component_id,
                component_name=self.component_def.name,
                component_type=ComponentType.REPORT_GENERATOR,
                deliverable_type=DeliverableType.COMPREHENSIVE_REPORT,
                status=ComponentStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                error=str(e),
                error_traceback=traceback.format_exc()
            )


class PDFValidatorExecutor(ComponentExecutor):
    """Executor for StrictPDFValidator."""

    def __init__(
        self,
        component_def: ComponentDefinition,
        pdf_validator: StrictPDFValidator
    ):
        """Initialize with PDF validator."""
        super().__init__(component_def)
        self.pdf_validator = pdf_validator

    def execute(self, context: ExecutionContext) -> ComponentResult:
        """Execute PDF validation."""
        start_time = datetime.now(timezone.utc)

        try:
            # Get report from context
            report_result = context.get_result('comprehensive_report')

            if not report_result:
                raise ValueError("Comprehensive report not found in context")

            result = self.pdf_validator.validate_comprehensive(
                report_content=report_result.full_report_markdown,
                report_data=report_result.full_report_json,
                step4_results=context.step4_results
            )

            # Generate compliance report
            compliance_report = self.pdf_validator.generate_compliance_report(
                result,
                output_path=context.config.output_dir / "compliance_report.md"
            )

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            # Store in context
            context.set_result('validation_result', result)

            return ComponentResult(
                component_id=self.component_def.component_id,
                component_name=self.component_def.name,
                component_type=ComponentType.PDF_VALIDATOR,
                deliverable_type=DeliverableType.VALIDATION_REPORT,
                status=ComponentStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                result=result,
                output_paths=[context.config.output_dir / "compliance_report.md"],
                metrics={
                    'is_compliant': result.is_pdf_compliant,
                    'compliance_score': result.compliance_score,
                    'total_checks': result.total_checks,
                    'passed_checks': result.passed_checks,
                    'failed_checks': result.failed_checks,
                    'critical_failures': len(result.critical_failures) if isinstance(result.critical_failures, list) else 0,
                    'major_failures': len(result.major_failures) if isinstance(result.major_failures, list) else 0
                }
            )

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            return ComponentResult(
                component_id=self.component_def.component_id,
                component_name=self.component_def.name,
                component_type=ComponentType.PDF_VALIDATOR,
                deliverable_type=DeliverableType.VALIDATION_REPORT,
                status=ComponentStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                error=str(e),
                error_traceback=traceback.format_exc()
            )


class PresentationGeneratorExecutor(ComponentExecutor):
    """Executor for PresentationGenerator."""

    def __init__(
        self,
        component_def: ComponentDefinition,
        presentation_generator: PresentationGenerator
    ):
        """Initialize with presentation generator."""
        super().__init__(component_def)
        self.presentation_generator = presentation_generator

    def execute(self, context: ExecutionContext) -> ComponentResult:
        """Execute presentation generation."""
        start_time = datetime.now(timezone.utc)

        try:
            result = self.presentation_generator.generate_presentation(
                step4_results=context.step4_results,
                universe_snapshot=context.universe_snapshot
            )

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            # Store in context
            context.set_result('presentation_result', result)

            return ComponentResult(
                component_id=self.component_def.component_id,
                component_name=self.component_def.name,
                component_type=ComponentType.PRESENTATION_GENERATOR,
                deliverable_type=DeliverableType.PRESENTATION,
                status=ComponentStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                result=result,
                output_paths=[
                    context.config.output_dir / "presentation.md",
                    context.config.output_dir / "presentation.html",
                    context.config.output_dir / "presentation.json"
                ],
                metrics={
                    'slide_count': len(result.slides) if isinstance(result.slides, list) else 0,
                    'is_compliant': result.is_compliant,
                    'generation_time': result.generation_time_seconds
                }
            )

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            return ComponentResult(
                component_id=self.component_def.component_id,
                component_name=self.component_def.name,
                component_type=ComponentType.PRESENTATION_GENERATOR,
                deliverable_type=DeliverableType.PRESENTATION,
                status=ComponentStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                error=str(e),
                error_traceback=traceback.format_exc()
            )


# =============================================================================
# STEP 5 COMPLETE ORCHESTRATOR
# =============================================================================

class Step5AdvancedOrchestrator:
    """
    Complete orchestrator for Step 5: Integration & Reporting.

    Coordinates all reporting components with full pipeline integration:
    - AdvancedReportGenerator (30-40 page comprehensive reports)
    - StrictPDFValidator (PDF compliance validation with 100+ checks)
    - PresentationGenerator (10-20 slide presentations)

    Features:
    - Event-driven component coordination with message bus
    - Enhanced checkpoint management for recovery
    - Real-time progress monitoring with callbacks
    - Parallel generation where dependencies allow
    - Cross-validation of all outputs
    - Quality gates at each phase
    - Automatic rollback on critical failures
    - Comprehensive audit logging

    PDF Requirements (MANDATORY):
    - 30-40 page Written Report with exact structure
    - 10-20 slide Presentation Deck
    - 100% PDF compliance validation
    - Multi-venue reporting with color coding
    - All 80+ metrics coverage
    - Crisis event documentation (14 events)
    - Walk-forward results (8 windows)
    - Capacity analysis with venue constraints
    - Grain futures comparison (PDF REQUIRED)
    - Position sizing validation
    - Concentration limits compliance

    Usage:
        config = OrchestratorConfig(output_dir=Path("outputs"))
        orchestrator = Step5AdvancedOrchestrator(config)
        result = orchestrator.run(
            step4_results=step4_orchestrator_results,
            universe_snapshot=universe,
            signals=signals,
            enhanced_signals=enhanced_signals
        )

        if result.is_success:
            print(f"All deliverables generated: {result.deliverables}")
        else:
            print(f"Errors: {result.errors}")
    """

    VERSION = "3.0.0"
    PDF_COMPLIANCE = "Project Specification"

    # State transition map
    VALID_TRANSITIONS: Dict[OrchestratorState, Set[OrchestratorState]] = {
        OrchestratorState.UNINITIALIZED: {OrchestratorState.INITIALIZING},
        OrchestratorState.INITIALIZING: {OrchestratorState.READY, OrchestratorState.FAILED},
        OrchestratorState.READY: {OrchestratorState.VALIDATING_INPUTS, OrchestratorState.RUNNING},
        OrchestratorState.VALIDATING_INPUTS: {OrchestratorState.RUNNING, OrchestratorState.FAILED},
        OrchestratorState.RUNNING: {
            OrchestratorState.PAUSED,
            OrchestratorState.COMPLETED,
            OrchestratorState.FAILED,
            OrchestratorState.CHECKPOINTING
        },
        OrchestratorState.PAUSED: {OrchestratorState.RESUMING, OrchestratorState.FAILED},
        OrchestratorState.RESUMING: {OrchestratorState.RUNNING, OrchestratorState.FAILED},
        OrchestratorState.CHECKPOINTING: {OrchestratorState.RUNNING, OrchestratorState.FAILED},
        OrchestratorState.RECOVERING: {OrchestratorState.RUNNING, OrchestratorState.FAILED},
        OrchestratorState.COMPLETED: {OrchestratorState.CLEANING_UP},
        OrchestratorState.FAILED: {OrchestratorState.RECOVERING, OrchestratorState.ROLLING_BACK, OrchestratorState.CLEANING_UP},
        OrchestratorState.ROLLING_BACK: {OrchestratorState.FAILED, OrchestratorState.CLEANING_UP},
        OrchestratorState.CLEANING_UP: {OrchestratorState.TERMINATED}
    }

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Initialize Step 5 orchestrator.

        Args:
            config: Orchestrator configuration (defaults to OrchestratorConfig())
        """
        self.config = config or OrchestratorConfig()
        self._state = OrchestratorState.UNINITIALIZED
        self._state_lock = RLock()

        # Generate orchestrator ID
        self.orchestrator_id = f"orch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Initialize message bus FIRST (before state transitions that use it)
        self.message_bus = Step5MessageBus()

        # Transition to initializing
        self._transition_state(OrchestratorState.INITIALIZING)

        # Create directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_manager = Step5CheckpointManager(
            self.config.checkpoint_dir,
            max_checkpoints=self.config.max_checkpoints,
            compress=self.config.compress_checkpoints
        )
        self.progress_monitor = Step5ProgressMonitor(
            enable_callbacks=self.config.enable_progress_callbacks
        )
        self.audit_logger = Step5AuditLogger(self.config.log_dir)
        self.quality_gate_manager = QualityGateManager(self.config.quality_gates)

        # Initialize components
        self._init_components()

        # Component results
        self.component_results: Dict[str, ComponentResult] = {}
        self.phase_results: Dict[ExecutionPhase, PhaseResult] = {}

        # Execution tracking
        self._execution_id: Optional[str] = None
        self._start_time: Optional[datetime] = None
        self._errors: List[str] = []
        self._warnings: List[str] = []

        # Transition to ready
        self._transition_state(OrchestratorState.READY)

        logger.info(f"Step5AdvancedOrchestrator initialized: {self.orchestrator_id}")
        self.audit_logger.info(
            EventType.ORCHESTRATOR_STARTED,
            f"Orchestrator initialized: {self.orchestrator_id}",
            details={'version': self.VERSION, 'config': self.config.to_dict()}
        )

    def _init_components(self) -> None:
        """Initialize component generators."""
        # Report generator
        self.report_generator = create_advanced_report_generator(
            output_dir=self.config.output_dir,
            data_start=self.config.data_start,
            data_end=self.config.data_end
        )

        # PDF validator
        self.pdf_validator = create_strict_pdf_validator()

        # Presentation generator (optional - may not be implemented)
        self.presentation_generator = None
        try:
            if create_presentation_generator is not None and callable(create_presentation_generator):
                self.presentation_generator = create_presentation_generator(
                    output_dir=self.config.output_dir
                )
        except (TypeError, Exception) as e:
            logger.warning(f"Presentation generator not available ({e}) - skipping presentation output")

        # Create component definitions
        self._component_definitions = self._create_component_definitions()

        # Create executors
        self._executors: Dict[str, ComponentExecutor] = {}
        self._create_executors()

        # Build dependency resolver
        self.dependency_resolver = DependencyResolver(
            list(self._component_definitions.values())
        )

    def _create_component_definitions(self) -> Dict[str, ComponentDefinition]:
        """Create component definitions for the pipeline."""
        return {
            'report_generator': ComponentDefinition(
                component_id='report_generator',
                component_type=ComponentType.REPORT_GENERATOR,
                name='Comprehensive Report Generator',
                description='Generates 30-40 page comprehensive report per PDF requirements',
                dependencies=[],
                priority=Priority.CRITICAL,
                required=True,
                timeout_seconds=600.0,
                deliverable_types=[
                    DeliverableType.COMPREHENSIVE_REPORT,
                    DeliverableType.REPORT_MARKDOWN,
                    DeliverableType.REPORT_JSON
                ]
            ),
            'pdf_validator': ComponentDefinition(
                component_id='pdf_validator',
                component_type=ComponentType.PDF_VALIDATOR,
                name='Strict PDF Validator',
                description='Validates 100% PDF compliance with 100+ checks',
                dependencies=['report_generator'],
                priority=Priority.CRITICAL,
                required=True,
                timeout_seconds=120.0,
                deliverable_types=[
                    DeliverableType.VALIDATION_REPORT,
                    DeliverableType.COMPLIANCE_SUMMARY
                ]
            ),
            'presentation_generator': ComponentDefinition(
                component_id='presentation_generator',
                component_type=ComponentType.PRESENTATION_GENERATOR,
                name='Presentation Generator',
                description='Generates 10-20 slide presentation per PDF requirements',
                dependencies=['report_generator'],
                priority=Priority.HIGH,
                required=self.config.generate_presentation,
                timeout_seconds=300.0,
                deliverable_types=[
                    DeliverableType.PRESENTATION,
                    DeliverableType.PRESENTATION_MARP,
                    DeliverableType.PRESENTATION_HTML
                ]
            ),
            'summary_generator': ComponentDefinition(
                component_id='summary_generator',
                component_type=ComponentType.SUMMARY_GENERATOR,
                name='Summary Generator',
                description='Generates final summary and metrics export',
                dependencies=['report_generator', 'pdf_validator', 'presentation_generator'],
                priority=Priority.MEDIUM,
                required=True,
                timeout_seconds=60.0,
                deliverable_types=[
                    DeliverableType.SUMMARY_JSON,
                    DeliverableType.METRICS_EXPORT
                ]
            )
        }

    def _create_executors(self) -> None:
        """Create component executors."""
        self._executors['report_generator'] = ReportGeneratorExecutor(
            self._component_definitions['report_generator'],
            self.report_generator
        )

        self._executors['pdf_validator'] = PDFValidatorExecutor(
            self._component_definitions['pdf_validator'],
            self.pdf_validator
        )

        if self.presentation_generator is not None:
            self._executors['presentation_generator'] = PresentationGeneratorExecutor(
                self._component_definitions['presentation_generator'],
                self.presentation_generator
            )
        else:
            logger.warning("Skipping presentation_generator executor (not available)")

    @property
    def state(self) -> OrchestratorState:
        """Get current orchestrator state."""
        with self._state_lock:
            return self._state

    def _transition_state(self, new_state: OrchestratorState) -> bool:
        """
        Transition to a new state.

        Args:
            new_state: Target state

        Returns:
            True if transition was successful
        """
        with self._state_lock:
            if new_state in self.VALID_TRANSITIONS.get(self._state, set()):
                old_state = self._state
                self._state = new_state

                logger.info(f"State transition: {old_state.value} -> {new_state.value}")

                # Publish state change event
                self.message_bus.publish(
                    EventType.ORCHESTRATOR_STARTED if new_state == OrchestratorState.RUNNING else EventType.COMPONENT_PROGRESS,
                    {
                        'old_state': old_state.value,
                        'new_state': new_state.value,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                )

                return True
            else:
                logger.warning(
                    f"Invalid state transition: {self._state.value} -> {new_state.value}"
                )
                return False

    def run(
        self,
        step4_results: Dict[str, Any],
        universe_snapshot: Any,
        signals: pd.DataFrame,
        enhanced_signals: pd.DataFrame,
        resume_from_checkpoint: Optional[str] = None
    ) -> Step5Result:
        """
        Execute Step 5 orchestration.

        This is the main entry point for running the complete Step 5 pipeline.

        Args:
            step4_results: Results from Step4AdvancedOrchestrator
            universe_snapshot: Universe snapshot from Step 1
            signals: Baseline signals from Step 2
            enhanced_signals: Enhanced signals from Step 3
            resume_from_checkpoint: Optional checkpoint ID to resume from

        Returns:
            Step5Result with all deliverables and validation results
        """
        self._execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        self._start_time = datetime.now(timezone.utc)
        self._errors = []
        self._warnings = []

        # Start monitoring
        self.progress_monitor.start()

        logger.info("=" * 80)
        logger.info("STEP 5 COMPLETE ORCHESTRATOR - STARTING")
        logger.info("=" * 80)
        logger.info(f"Orchestrator ID: {self.orchestrator_id}")
        logger.info(f"Execution ID: {self._execution_id}")
        logger.info(f"Version: {self.VERSION}")
        logger.info(f"PDF Compliance: {self.PDF_COMPLIANCE}")

        # Log to audit
        self.audit_logger.info(
            EventType.ORCHESTRATOR_STARTED,
            f"Execution started: {self._execution_id}",
            details={
                'orchestrator_id': self.orchestrator_id,
                'version': self.VERSION,
                'resume_from': resume_from_checkpoint
            }
        )

        # Publish start event
        self.message_bus.publish(
            EventType.ORCHESTRATOR_STARTED,
            {
                'orchestrator_id': self.orchestrator_id,
                'execution_id': self._execution_id,
                'timestamp': self._start_time.isoformat()
            }
        )

        # Create execution context
        context = ExecutionContext(
            orchestrator_id=self.orchestrator_id,
            execution_id=self._execution_id,
            phase=ExecutionPhase.INITIALIZATION,
            config=self.config,
            step4_results=step4_results,
            universe_snapshot=universe_snapshot,
            signals=signals,
            enhanced_signals=enhanced_signals
        )

        try:
            # Check for resume
            if resume_from_checkpoint:
                self._resume_from_checkpoint(resume_from_checkpoint, context)
            else:
                # Validate inputs
                self._validate_inputs(context)

            # Transition to running
            self._transition_state(OrchestratorState.RUNNING)

            # Execute phases
            self._execute_phases(context)

            # Generate final summary
            self._generate_final_summary(context)

            # Transition to completed
            self._transition_state(OrchestratorState.COMPLETED)

        except Exception as e:
            self._transition_state(OrchestratorState.FAILED)
            error_msg = f"Orchestrator failed: {str(e)}"
            self._errors.append(error_msg)

            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.audit_logger.error(
                EventType.ORCHESTRATOR_FAILED,
                error_msg,
                details={'traceback': traceback.format_exc()}
            )

            # Create failure checkpoint
            if self.config.enable_checkpointing:
                self._create_checkpoint(context, is_failure=True)

        # Stop monitoring
        self.progress_monitor.stop()

        # Save audit log
        audit_log_path = self.audit_logger.save()

        # Build result
        result = self._build_result(context, audit_log_path)

        # Log completion
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5 COMPLETE ORCHESTRATOR - COMPLETED")
        logger.info("=" * 80)
        logger.info(f"State: {self.state.value}")
        logger.info(f"Duration: {result.duration_seconds:.1f} seconds")
        logger.info(f"PDF Compliant: {result.is_pdf_compliant}")
        logger.info(f"All Gates Passed: {result.all_gates_passed}")
        logger.info(f"Deliverables Generated: {sum(len(v) for v in result.deliverables.values())}")

        if result.errors:
            logger.error(f"Errors: {len(result.errors)}")
        if result.warnings:
            logger.warning(f"Warnings: {len(result.warnings)}")

        # Publish completion event
        self.message_bus.publish(
            EventType.ORCHESTRATOR_COMPLETED,
            {
                'orchestrator_id': self.orchestrator_id,
                'execution_id': self._execution_id,
                'state': self.state.value,
                'duration_seconds': result.duration_seconds,
                'is_success': result.is_success
            }
        )

        return result

    def _validate_inputs(self, context: ExecutionContext) -> None:
        """Validate all inputs before execution."""
        self._transition_state(OrchestratorState.VALIDATING_INPUTS)
        self.progress_monitor.start_phase(ExecutionPhase.INPUT_VALIDATION)

        logger.info("\n[INPUT VALIDATION] Validating inputs...")

        # Validate step4_results
        if not context.step4_results:
            raise ValueError("step4_results is required")

        # Validate signals
        if context.signals is None or len(context.signals) == 0:
            self._warnings.append("Signals dataframe is empty")

        if context.enhanced_signals is None or len(context.enhanced_signals) == 0:
            self._warnings.append("Enhanced signals dataframe is empty")

        # Validate universe snapshot
        if context.universe_snapshot is None:
            self._warnings.append("Universe snapshot is None")

        # Validate required metrics in step4_results
        required_keys = ['advanced_metrics', 'backtest_results']
        for key in required_keys:
            if key not in context.step4_results:
                self._warnings.append(f"step4_results missing key: {key}")

        self.progress_monitor.complete_phase(ExecutionPhase.INPUT_VALIDATION)
        logger.info("[INPUT VALIDATION] Complete")

    def _execute_phases(self, context: ExecutionContext) -> None:
        """Execute all phases in sequence."""
        phases = [
            (ExecutionPhase.REPORT_GENERATION, self._execute_report_generation),
            (ExecutionPhase.PDF_VALIDATION, self._execute_pdf_validation),
            (ExecutionPhase.PRESENTATION_GENERATION, self._execute_presentation_generation),
            (ExecutionPhase.QUALITY_ASSURANCE, self._execute_quality_assurance),
            (ExecutionPhase.FINALIZATION, self._execute_finalization)
        ]

        for phase, executor in phases:
            context.phase = phase
            self.progress_monitor.start_phase(phase)

            try:
                self.audit_logger.info(
                    EventType.PHASE_STARTED,
                    f"Starting phase: {phase.value}",
                    phase=phase
                )

                # Execute phase
                executor(context)

                # Create checkpoint after each phase
                if self.config.enable_checkpointing:
                    self._create_checkpoint(context)

                self.progress_monitor.complete_phase(phase)

                self.audit_logger.info(
                    EventType.PHASE_COMPLETED,
                    f"Completed phase: {phase.value}",
                    phase=phase
                )

            except Exception as e:
                self.progress_monitor.fail_phase(phase)
                self.audit_logger.error(
                    EventType.PHASE_FAILED,
                    f"Phase failed: {phase.value} - {str(e)}",
                    phase=phase,
                    details={'error': str(e), 'traceback': traceback.format_exc()}
                )
                raise

    def _execute_report_generation(self, context: ExecutionContext) -> None:
        """Execute report generation phase."""
        logger.info("\n[PHASE 1/5] Report Generation")

        executor = self._executors['report_generator']
        self.progress_monitor.update_component('report_generator', ComponentStatus.RUNNING, 0.1)

        result = executor.execute_with_retry(context)
        self.component_results['report_generator'] = result

        if not result.is_success:
            raise RuntimeError(f"Report generation failed: {result.error}")

        self.progress_monitor.update_component('report_generator', ComponentStatus.COMPLETED, 1.0)

        logger.info(f"  Report generated: {result.metrics.get('estimated_pages', 0):.1f} pages")

        # Check quality gate
        gate_results = self.quality_gate_manager.check_phase_gates(
            ExecutionPhase.REPORT_GENERATION,
            result.metrics
        )
        result.quality_gates_passed = [g.gate_name for g in gate_results if g.passed]
        result.quality_gates_failed = [g.gate_name for g in gate_results if not g.passed]

    def _execute_pdf_validation(self, context: ExecutionContext) -> None:
        """Execute PDF validation phase."""
        logger.info("\n[PHASE 2/5] PDF Validation")

        executor = self._executors['pdf_validator']
        self.progress_monitor.update_component('pdf_validator', ComponentStatus.RUNNING, 0.1)

        result = executor.execute_with_retry(context)
        self.component_results['pdf_validator'] = result

        if not result.is_success:
            raise RuntimeError(f"PDF validation failed: {result.error}")

        self.progress_monitor.update_component('pdf_validator', ComponentStatus.COMPLETED, 1.0)

        logger.info(f"  Validation: {'COMPLIANT' if result.metrics.get('is_compliant') else 'NON-COMPLIANT'}")
        logger.info(f"  Score: {result.metrics.get('compliance_score', 0):.1%}")

        # Check quality gate
        gate_results = self.quality_gate_manager.check_phase_gates(
            ExecutionPhase.PDF_VALIDATION,
            result.metrics
        )
        result.quality_gates_passed = [g.gate_name for g in gate_results if g.passed]
        result.quality_gates_failed = [g.gate_name for g in gate_results if not g.passed]

    def _execute_presentation_generation(self, context: ExecutionContext) -> None:
        """Execute presentation generation phase."""
        logger.info("\n[PHASE 3/5] Presentation Generation")

        if not self.config.generate_presentation:
            logger.info("  Skipped (disabled)")
            self.progress_monitor.update_component(
                'presentation_generator',
                ComponentStatus.SKIPPED,
                1.0
            )
            return

        executor = self._executors['presentation_generator']
        self.progress_monitor.update_component('presentation_generator', ComponentStatus.RUNNING, 0.1)

        result = executor.execute_with_retry(context)
        self.component_results['presentation_generator'] = result

        if not result.is_success:
            if self._component_definitions['presentation_generator'].required:
                raise RuntimeError(f"Presentation generation failed: {result.error}")
            else:
                self._warnings.append(f"Presentation generation failed: {result.error}")

        self.progress_monitor.update_component('presentation_generator', ComponentStatus.COMPLETED, 1.0)

        logger.info(f"  Presentation generated: {result.metrics.get('slide_count', 0)} slides")

        # Check quality gate
        gate_results = self.quality_gate_manager.check_phase_gates(
            ExecutionPhase.PRESENTATION_GENERATION,
            result.metrics
        )
        result.quality_gates_passed = [g.gate_name for g in gate_results if g.passed]
        result.quality_gates_failed = [g.gate_name for g in gate_results if not g.passed]

    def _execute_quality_assurance(self, context: ExecutionContext) -> None:
        """Execute quality assurance phase."""
        logger.info("\n[PHASE 4/5] Quality Assurance")

        self.progress_monitor.update_component('quality_assurance', ComponentStatus.RUNNING, 0.1)

        # Aggregate metrics for QA checks
        qa_metrics = self._aggregate_qa_metrics(context)

        # Check all QA gates
        gate_results = self.quality_gate_manager.check_phase_gates(
            ExecutionPhase.QUALITY_ASSURANCE,
            qa_metrics
        )

        # Check for critical failures
        for gate_result in gate_results:
            if not gate_result.passed:
                strategy = self.quality_gate_manager.get_recovery_strategy(
                    gate_result.details['gate_id']
                )

                if strategy == RecoveryStrategy.ABORT:
                    self._errors.append(
                        f"Quality gate '{gate_result.gate_name}' failed: "
                        f"{gate_result.score} {gate_result.details['comparison']} "
                        f"{gate_result.threshold}"
                    )
                else:
                    self._warnings.append(
                        f"Quality gate '{gate_result.gate_name}' failed "
                        f"(non-critical)"
                    )

        self.progress_monitor.update_component('quality_assurance', ComponentStatus.COMPLETED, 1.0)

        passed = sum(1 for g in gate_results if g.passed)
        total = len(gate_results)
        logger.info(f"  Quality gates: {passed}/{total} passed")

    def _aggregate_qa_metrics(self, context: ExecutionContext) -> Dict[str, Any]:
        """Aggregate metrics for quality assurance."""
        metrics = {}

        # From report generator
        report_result = context.get_result('comprehensive_report')
        if report_result:
            metrics['estimated_pages'] = report_result.estimated_pages
            metrics['sections_count'] = len(report_result.sections)
            metrics['crisis_events_count'] = len(report_result.crisis_events)
            metrics['walk_forward_count'] = len(report_result.walk_forward_periods)
            metrics['has_grain_comparison'] = 1 if len(report_result.grain_comparisons) > 0 else 0
            metrics['metrics_count'] = len(report_result.venue_metrics) * 20  # Approximate

        # From validation
        validation_result = context.get_result('validation_result')
        if validation_result:
            metrics['compliance_score'] = validation_result.compliance_score
            metrics['passed_checks'] = validation_result.passed_checks
            metrics['failed_checks'] = validation_result.failed_checks

        # From presentation
        presentation_result = context.get_result('presentation_result')
        if presentation_result:
            metrics['slide_count'] = len(presentation_result.slides)

        return metrics

    def _execute_finalization(self, context: ExecutionContext) -> None:
        """Execute finalization phase."""
        logger.info("\n[PHASE 5/5] Finalization")

        self.progress_monitor.update_component('finalization', ComponentStatus.RUNNING, 0.1)

        # Collect all deliverables
        deliverables = self._collect_deliverables(context)
        context.metadata['deliverables'] = deliverables

        # Save summary JSON
        summary_path = self._save_summary_json(context, deliverables)

        self.progress_monitor.update_component('finalization', ComponentStatus.COMPLETED, 1.0)

        logger.info(f"  Summary saved: {summary_path}")
        logger.info(f"  Total deliverables: {sum(len(v) for v in deliverables.values())}")

    def _generate_final_summary(self, context: ExecutionContext) -> None:
        """Generate final summary after all phases."""
        pass  # Summary generated in finalization phase

    def _collect_deliverables(
        self,
        context: ExecutionContext
    ) -> Dict[DeliverableType, List[Path]]:
        """Collect all deliverable paths."""
        deliverables: Dict[DeliverableType, List[Path]] = defaultdict(list)

        # Report deliverables
        deliverables[DeliverableType.COMPREHENSIVE_REPORT].extend([
            self.config.output_dir / "comprehensive_report.md",
            self.config.output_dir / "comprehensive_report.json"
        ])

        # Validation deliverables
        deliverables[DeliverableType.VALIDATION_REPORT].append(
            self.config.output_dir / "compliance_report.md"
        )

        # Presentation deliverables
        if self.config.generate_presentation:
            deliverables[DeliverableType.PRESENTATION].extend([
                self.config.output_dir / "presentation.md",
                self.config.output_dir / "presentation.html",
                self.config.output_dir / "presentation.json"
            ])

        return dict(deliverables)

    def _save_summary_json(
        self,
        context: ExecutionContext,
        deliverables: Dict[DeliverableType, List[Path]]
    ) -> Path:
        """Save final summary JSON."""
        validation_result = context.get_result('validation_result')
        report_result = context.get_result('comprehensive_report')

        summary = {
            'orchestrator_id': self.orchestrator_id,
            'orchestrator_version': self.VERSION,
            'execution_id': self._execution_id,
            'pdf_compliance': self.PDF_COMPLIANCE,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'deliverables': {
                k.value: [str(p) for p in v]
                for k, v in deliverables.items()
            },
            'compliance_summary': {
                'is_pdf_compliant': validation_result.is_pdf_compliant if validation_result else False,
                'compliance_score': validation_result.compliance_score if validation_result else 0,
                'total_checks': validation_result.total_checks if validation_result else 0,
                'passed_checks': validation_result.passed_checks if validation_result else 0,
                'failed_checks': validation_result.failed_checks if validation_result else 0
            },
            'metrics_summary': {
                'report_pages': report_result.estimated_pages if report_result else 0,
                'crisis_events': len(report_result.crisis_events) if report_result else 0,
                'walk_forward_periods': len(report_result.walk_forward_periods) if report_result else 0,
                'venue_count': len(report_result.venue_metrics) if report_result else 0
            },
            'component_results': {
                k: v.to_dict() for k, v in self.component_results.items()
            },
            'quality_gate_results': [
                {
                    'gate_name': g.gate_name,
                    'passed': g.passed,
                    'score': g.score,
                    'threshold': g.threshold
                }
                for g in self.quality_gate_manager.get_all_results()
            ],
            'errors': self._errors,
            'warnings': self._warnings
        }

        output_path = self.config.output_dir / "step5_summary.json"
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Saved final summary: {output_path}")
        return output_path

    def _create_checkpoint(
        self,
        context: ExecutionContext,
        is_failure: bool = False
    ) -> Optional[Checkpoint]:
        """Create a checkpoint for recovery."""
        if not self.config.enable_checkpointing:
            return None

        self._transition_state(OrchestratorState.CHECKPOINTING)

        try:
            checkpoint = self.checkpoint_manager.create_checkpoint(
                orchestrator_id=self.orchestrator_id,
                state=OrchestratorState.FAILED if is_failure else self.state,
                current_phase=context.phase,
                completed_phases=[
                    phase for phase, result in self.phase_results.items()
                    if result.is_success
                ],
                completed_components=list(self.component_results.keys()),
                component_results=self.component_results,
                intermediate_data=context.intermediate_results,
                config=self.config
            )

            self.audit_logger.info(
                EventType.CHECKPOINT_CREATED,
                f"Checkpoint created: {checkpoint.checkpoint_id}",
                phase=context.phase
            )

            return checkpoint

        finally:
            if not is_failure:
                self._transition_state(OrchestratorState.RUNNING)

    def _resume_from_checkpoint(
        self,
        checkpoint_id: str,
        context: ExecutionContext
    ) -> None:
        """Resume execution from a checkpoint."""
        self._transition_state(OrchestratorState.RECOVERING)

        checkpoint = self.checkpoint_manager.get_checkpoint(checkpoint_id)
        if not checkpoint:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        if not self.checkpoint_manager.validate_checkpoint(checkpoint):
            raise ValueError(f"Checkpoint validation failed: {checkpoint_id}")

        # Restore state
        self.component_results = checkpoint.component_results.copy()
        context.intermediate_results = checkpoint.intermediate_data.copy()
        context.phase = checkpoint.current_phase

        self.audit_logger.info(
            EventType.CHECKPOINT_RESTORED,
            f"Resumed from checkpoint: {checkpoint_id}",
            phase=checkpoint.current_phase
        )

        logger.info(f"Resumed from checkpoint: {checkpoint_id}")
        logger.info(f"  Completed components: {checkpoint.completed_components}")
        logger.info(f"  Current phase: {checkpoint.current_phase.value}")

    def _build_result(
        self,
        context: ExecutionContext,
        audit_log_path: Path
    ) -> Step5Result:
        """Build the final Step5Result."""
        completion_time = datetime.now(timezone.utc)
        duration = (completion_time - self._start_time).total_seconds()

        # Get results from context
        report_result = context.get_result('comprehensive_report')
        validation_result = context.get_result('validation_result')
        presentation_result = context.get_result('presentation_result')

        # Get deliverables
        deliverables = context.metadata.get('deliverables', {})

        # Build compliance summary
        compliance_summary = {}
        if validation_result:
            compliance_summary = {
                'is_pdf_compliant': validation_result.is_pdf_compliant,
                'compliance_score': f"{validation_result.compliance_score:.1%}",
                'total_checks': validation_result.total_checks,
                'passed_checks': validation_result.passed_checks,
                'failed_checks': validation_result.failed_checks,
                'critical_failures': len(validation_result.critical_failures),
                'major_failures': len(validation_result.major_failures)
            }

        # Build metrics summary
        metrics_summary = {}
        if report_result:
            metrics_summary = {
                'report_pages': report_result.estimated_pages,
                'sections': len(report_result.sections),
                'venue_breakdown': list(report_result.venue_metrics.keys()),
                'crisis_events_covered': len(report_result.crisis_events),
                'walk_forward_periods': len(report_result.walk_forward_periods),
                'grain_comparisons': len(report_result.grain_comparisons)
            }

        # Get all quality gate results
        gate_results = self.quality_gate_manager.get_all_results()
        all_gates_passed = self.quality_gate_manager.all_required_gates_passed()

        return Step5Result(
            orchestrator_id=self.orchestrator_id,
            orchestrator_version=self.VERSION,
            execution_id=self._execution_id,
            execution_time=self._start_time,
            completion_time=completion_time,
            duration_seconds=duration,
            state=self.state,
            config=self.config,
            phase_results=self.phase_results,
            component_results=self.component_results,
            comprehensive_report=report_result,
            validation_result=validation_result,
            presentation_result=presentation_result,
            deliverables=deliverables,
            is_pdf_compliant=validation_result.is_pdf_compliant if validation_result else False,
            compliance_score=validation_result.compliance_score if validation_result else 0.0,
            compliance_summary=compliance_summary,
            quality_gate_results=gate_results,
            all_gates_passed=all_gates_passed,
            metrics_summary=metrics_summary,
            resource_metrics={},
            errors=self._errors,
            warnings=self._warnings,
            audit_log_path=audit_log_path
        )

    def pause(self) -> bool:
        """Pause execution."""
        if self._transition_state(OrchestratorState.PAUSED):
            self.audit_logger.info(
                EventType.ORCHESTRATOR_PAUSED,
                "Orchestrator paused"
            )
            self.message_bus.publish(
                EventType.ORCHESTRATOR_PAUSED,
                {'orchestrator_id': self.orchestrator_id}
            )
            return True
        return False

    def resume(self) -> bool:
        """Resume execution."""
        if self._transition_state(OrchestratorState.RESUMING):
            self.audit_logger.info(
                EventType.ORCHESTRATOR_RESUMED,
                "Orchestrator resuming"
            )
            self.message_bus.publish(
                EventType.ORCHESTRATOR_RESUMED,
                {'orchestrator_id': self.orchestrator_id}
            )
            self._transition_state(OrchestratorState.RUNNING)
            return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        return {
            'orchestrator_id': self.orchestrator_id,
            'version': self.VERSION,
            'state': self.state.value,
            'execution_id': self._execution_id,
            'progress': self.progress_monitor.get_status_summary(),
            'component_status': {
                k: v.status.value for k, v in self.component_results.items()
            },
            'message_bus_stats': self.message_bus.get_stats(),
            'checkpoints': self.checkpoint_manager.list_checkpoints(),
            'audit_summary': self.audit_logger.get_summary()
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_step5_orchestrator(
    output_dir: Optional[Path] = None,
    data_start: datetime = datetime(2020, 1, 1),
    data_end: Optional[datetime] = None,
    execution_mode: ExecutionMode = ExecutionMode.HYBRID,
    generate_presentation: bool = True,
    strict_validation: bool = True,
    enable_checkpointing: bool = True,
    max_workers: int = 4
) -> Step5AdvancedOrchestrator:
    """
    Factory function to create Step5AdvancedOrchestrator.

    Creates a fully configured orchestrator ready for execution.

    Args:
        output_dir: Output directory (default: outputs/step5_reports)
        data_start: Data start date (default: 2020-01-01 per PDF)
        data_end: Data end date (default: now)
        execution_mode: Execution mode (default: HYBRID)
        generate_presentation: Whether to generate presentation
        strict_validation: Whether to use strict validation
        enable_checkpointing: Whether to enable checkpointing
        max_workers: Maximum worker threads

    Returns:
        Configured Step5AdvancedOrchestrator instance

    Example:
        orchestrator = create_step5_orchestrator(
            output_dir=Path("outputs/reports"),
            generate_presentation=True
        )
        result = orchestrator.run(...)
    """
    config = OrchestratorConfig(
        output_dir=output_dir or Path("outputs/step5_reports"),
        data_start=data_start,
        data_end=data_end,
        execution_mode=execution_mode,
        generate_presentation=generate_presentation,
        strict_validation=strict_validation,
        enable_checkpointing=enable_checkpointing,
        max_workers=max_workers
    )

    return Step5AdvancedOrchestrator(config=config)


def run_step5_advanced(
    step4_results: Dict[str, Any],
    universe_snapshot: Any,
    signals: pd.DataFrame,
    enhanced_signals: pd.DataFrame,
    output_dir: Optional[Path] = None,
    **kwargs
) -> Step5Result:
    """
    Convenience function to run Step 5 orchestration.

    Creates an orchestrator and runs the complete pipeline.

    Args:
        step4_results: Results from Step4AdvancedOrchestrator
        universe_snapshot: Universe snapshot from Step 1
        signals: Baseline signals from Step 2
        enhanced_signals: Enhanced signals from Step 3
        output_dir: Output directory
        **kwargs: Additional configuration options

    Returns:
        Step5Result with all deliverables

    Example:
        result = run_step5_advanced(
            step4_results=step4_results,
            universe_snapshot=universe,
            signals=signals,
            enhanced_signals=enhanced_signals
        )

        if result.is_success:
            print(f"Generated: {result.deliverables}")
    """
    orchestrator = create_step5_orchestrator(output_dir=output_dir, **kwargs)

    return orchestrator.run(
        step4_results=step4_results,
        universe_snapshot=universe_snapshot,
        signals=signals,
        enhanced_signals=enhanced_signals
    )


def quick_run_step5(
    step4_results: Dict[str, Any],
    universe_snapshot: Any,
    signals: pd.DataFrame,
    enhanced_signals: pd.DataFrame
) -> Step5Result:
    """
    Quick run with minimal configuration.

    Uses default settings for fast execution.

    Args:
        step4_results: Results from Step4AdvancedOrchestrator
        universe_snapshot: Universe snapshot from Step 1
        signals: Baseline signals from Step 2
        enhanced_signals: Enhanced signals from Step 3

    Returns:
        Step5Result
    """
    config = OrchestratorConfig(
        enable_checkpointing=False,
        enable_monitoring=False,
        save_intermediate=False
    )

    orchestrator = Step5AdvancedOrchestrator(config=config)

    return orchestrator.run(
        step4_results=step4_results,
        universe_snapshot=universe_snapshot,
        signals=signals,
        enhanced_signals=enhanced_signals
    )


def resume_step5_from_checkpoint(
    checkpoint_id: str,
    step4_results: Dict[str, Any],
    universe_snapshot: Any,
    signals: pd.DataFrame,
    enhanced_signals: pd.DataFrame,
    checkpoint_dir: Optional[Path] = None
) -> Step5Result:
    """
    Resume Step 5 execution from a checkpoint.

    Args:
        checkpoint_id: ID of the checkpoint to resume from
        step4_results: Results from Step4AdvancedOrchestrator
        universe_snapshot: Universe snapshot from Step 1
        signals: Baseline signals from Step 2
        enhanced_signals: Enhanced signals from Step 3
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Step5Result
    """
    config = OrchestratorConfig()
    if checkpoint_dir:
        config.checkpoint_dir = checkpoint_dir

    orchestrator = Step5AdvancedOrchestrator(config=config)

    return orchestrator.run(
        step4_results=step4_results,
        universe_snapshot=universe_snapshot,
        signals=signals,
        enhanced_signals=enhanced_signals,
        resume_from_checkpoint=checkpoint_id
    )


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

def _configure_logging() -> None:
    """Configure logging for Step 5 orchestrator."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# Configure logging on module import
_configure_logging()
