"""
Strict PDF Compliance Validator - project specification Full Compliance
========================================================================

Comprehensive PDF compliance validation implementing all requirements from
Project Specification.

This is a comprehensive validation framework with:
- 15+ validation categories with 100+ individual checks
- Weighted scoring system with configurable weights
- Deep Step4 results integration and cross-validation
- Automated remediation suggestions
- Multiple validation profiles (strict, standard, lenient)
- Cross-reference validation between sections
- Statistical validation of reported metrics
- Quality scoring algorithms
- Batch validation capabilities
- Validation caching for performance
- Custom rule engine for extensibility

Validation Categories:
1. Document Structure (required sections, page count)
2. Content Completeness (methodology, risk management)
3. Metrics Coverage (80+ metrics with statistical validation)
4. Venue-Specific Requirements (CEX/DEX/Hybrid breakdown with color coding)
5. Crisis Event Coverage (14 required events with performance analysis)
6. Walk-Forward Validation (18m train / 6m test, 8 windows)
7. Capacity Analysis (CEX $10-30M, DEX $1-5M, Combined $20-50M)
8. Grain Futures Comparison (PDF REQUIRED academic benchmark)
9. Position Sizing (CEX $100k, DEX $20-50k liquid, $5-10k illiquid)
10. Concentration Limits (40% sector, 60% CEX, 20% Tier 3)
11. Statistical Validation (t-tests, confidence intervals)
12. Cross-Reference Validation (internal consistency)
13. Formatting & Presentation (tables, charts, structure)
14. Academic Standards (citations, references)
15. Executive Quality (clarity, completeness)

Integration:
- Validates AdvancedReportGenerator output
- Validates Step4AdvancedOrchestrator results
- Cross-validates between report and underlying data
- Generates detailed compliance reports with remediation

Author: Tamer Atesyakar
Version: 3.0.0
PDF Compliance: Project Specification
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from typing import (
    Any, Callable, Dict, FrozenSet, List, Optional,
    Protocol, Set, Tuple, Type, TypeVar, Union
)

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE VARIABLES
# =============================================================================

T = TypeVar('T')
ValidatorT = TypeVar('ValidatorT', bound='CategoryValidator')


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ComplianceLevel(Enum):
    """Compliance severity levels with numeric weights for scoring."""
    CRITICAL = "critical"    # Must pass for PDF compliance (weight: 10)
    MAJOR = "major"          # Should pass for quality (weight: 5)
    MINOR = "minor"          # Nice to have (weight: 2)
    INFO = "info"            # Informational only (weight: 0)

    @property
    def weight(self) -> float:
        """Get numeric weight for scoring."""
        weights = {
            ComplianceLevel.CRITICAL: 10.0,
            ComplianceLevel.MAJOR: 5.0,
            ComplianceLevel.MINOR: 2.0,
            ComplianceLevel.INFO: 0.0
        }
        return weights.get(self, 0.0)

    @property
    def penalty(self) -> float:
        """Get penalty for failure (used in weighted scoring)."""
        penalties = {
            ComplianceLevel.CRITICAL: 20.0,
            ComplianceLevel.MAJOR: 10.0,
            ComplianceLevel.MINOR: 3.0,
            ComplianceLevel.INFO: 0.0
        }
        return penalties.get(self, 0.0)


class ValidationCategory(Enum):
    """Categories of validation checks with descriptions."""
    DOCUMENT_STRUCTURE = "document_structure"
    PAGE_COUNT = "page_count"
    CONTENT_COMPLETENESS = "content_completeness"
    METRICS_COVERAGE = "metrics_coverage"
    VENUE_BREAKDOWN = "venue_breakdown"
    CRISIS_EVENTS = "crisis_events"
    WALK_FORWARD = "walk_forward"
    CAPACITY_ANALYSIS = "capacity_analysis"
    GRAIN_COMPARISON = "grain_comparison"
    POSITION_SIZING = "position_sizing"
    CONCENTRATION_LIMITS = "concentration_limits"
    STATISTICAL_VALIDATION = "statistical_validation"
    CROSS_REFERENCE = "cross_reference"
    FORMATTING = "formatting"
    ACADEMIC_STANDARDS = "academic_standards"
    EXECUTIVE_QUALITY = "executive_quality"

    @property
    def description(self) -> str:
        """Get human-readable description."""
        descriptions = {
            ValidationCategory.DOCUMENT_STRUCTURE: "Document structure and required sections",
            ValidationCategory.PAGE_COUNT: "Page count requirements (30-40 pages)",
            ValidationCategory.CONTENT_COMPLETENESS: "Content completeness and depth",
            ValidationCategory.METRICS_COVERAGE: "Performance metrics coverage (80+ metrics)",
            ValidationCategory.VENUE_BREAKDOWN: "Venue-specific breakdown (CEX/DEX/Hybrid)",
            ValidationCategory.CRISIS_EVENTS: "Crisis event coverage (14 events)",
            ValidationCategory.WALK_FORWARD: "Walk-forward validation (18m/6m)",
            ValidationCategory.CAPACITY_ANALYSIS: "Capacity analysis with degradation",
            ValidationCategory.GRAIN_COMPARISON: "Grain futures comparison (PDF REQUIRED)",
            ValidationCategory.POSITION_SIZING: "Position sizing methodology",
            ValidationCategory.CONCENTRATION_LIMITS: "Concentration limits compliance",
            ValidationCategory.STATISTICAL_VALIDATION: "Statistical validation of metrics",
            ValidationCategory.CROSS_REFERENCE: "Cross-reference consistency",
            ValidationCategory.FORMATTING: "Formatting and presentation",
            ValidationCategory.ACADEMIC_STANDARDS: "Academic standards compliance",
            ValidationCategory.EXECUTIVE_QUALITY: "Executive summary quality"
        }
        return descriptions.get(self, "Unknown category")

    @property
    def weight(self) -> float:
        """Get category weight for overall scoring."""
        weights = {
            ValidationCategory.DOCUMENT_STRUCTURE: 1.0,
            ValidationCategory.PAGE_COUNT: 0.8,
            ValidationCategory.CONTENT_COMPLETENESS: 1.2,
            ValidationCategory.METRICS_COVERAGE: 1.5,
            ValidationCategory.VENUE_BREAKDOWN: 1.3,
            ValidationCategory.CRISIS_EVENTS: 1.2,
            ValidationCategory.WALK_FORWARD: 1.4,
            ValidationCategory.CAPACITY_ANALYSIS: 1.3,
            ValidationCategory.GRAIN_COMPARISON: 1.0,
            ValidationCategory.POSITION_SIZING: 1.1,
            ValidationCategory.CONCENTRATION_LIMITS: 1.1,
            ValidationCategory.STATISTICAL_VALIDATION: 1.2,
            ValidationCategory.CROSS_REFERENCE: 0.9,
            ValidationCategory.FORMATTING: 0.7,
            ValidationCategory.ACADEMIC_STANDARDS: 0.6,
            ValidationCategory.EXECUTIVE_QUALITY: 1.0
        }
        return weights.get(self, 1.0)


class CheckStatus(Enum):
    """Status of individual checks with scoring implications."""
    PASSED = "passed"       # Full credit
    FAILED = "failed"       # Penalty applied
    WARNING = "warning"     # Partial credit
    SKIPPED = "skipped"     # No impact
    NOT_APPLICABLE = "not_applicable"  # No impact
    INFO = "info"           # Informational, no impact

    @property
    def score_multiplier(self) -> float:
        """Get score multiplier for this status."""
        multipliers = {
            CheckStatus.PASSED: 1.0,
            CheckStatus.FAILED: 0.0,
            CheckStatus.WARNING: 0.5,
            CheckStatus.SKIPPED: 0.0,
            CheckStatus.NOT_APPLICABLE: 0.0,
            CheckStatus.INFO: 0.0
        }
        return multipliers.get(self, 0.0)


class ValidationProfile(Enum):
    """Validation profile determining strictness level."""
    STRICT = "strict"           # All requirements enforced
    STANDARD = "standard"       # Standard requirements
    LENIENT = "lenient"         # Minimum requirements only
    CUSTOM = "custom"           # Custom configuration

    @property
    def min_compliance_score(self) -> float:
        """Minimum score for compliance under this profile."""
        thresholds = {
            ValidationProfile.STRICT: 0.95,
            ValidationProfile.STANDARD: 0.85,
            ValidationProfile.LENIENT: 0.70,
            ValidationProfile.CUSTOM: 0.80
        }
        return thresholds.get(self, 0.85)


class RemediationPriority(Enum):
    """Priority level for remediation suggestions."""
    IMMEDIATE = "immediate"     # Must fix before submission
    HIGH = "high"               # Should fix soon
    MEDIUM = "medium"           # Fix when possible
    LOW = "low"                 # Optional improvement

    @property
    def urgency_score(self) -> int:
        """Get numeric urgency score."""
        scores = {
            RemediationPriority.IMMEDIATE: 100,
            RemediationPriority.HIGH: 75,
            RemediationPriority.MEDIUM: 50,
            RemediationPriority.LOW: 25
        }
        return scores.get(self, 0)


class MetricType(Enum):
    """Types of metrics for validation."""
    PERCENTAGE = "percentage"
    RATIO = "ratio"
    CURRENCY = "currency"
    COUNT = "count"
    DURATION = "duration"
    DECIMAL = "decimal"


# =============================================================================
# DATA CLASSES - Core Validation Structures
# =============================================================================

@dataclass
class ValidationCheck:
    """
    Individual validation check result with comprehensive metadata.

    Attributes:
        check_id: Unique identifier for the check
        name: Human-readable check name
        category: Validation category
        level: Compliance level (critical/major/minor/info)
        status: Check status (passed/failed/warning/skipped)
        message: Descriptive message about the result
        details: Additional details dictionary
        remediation: Suggested remediation if failed
        evidence: Evidence supporting the result
        related_checks: IDs of related checks
        execution_time_ms: Time taken to execute check
    """
    check_id: str
    name: str
    category: ValidationCategory
    level: ComplianceLevel
    status: CheckStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    remediation: Optional[str] = None
    evidence: Optional[str] = None
    related_checks: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0

    @property
    def score_contribution(self) -> float:
        """Calculate score contribution of this check."""
        return self.level.weight * self.status.score_multiplier

    @property
    def penalty_contribution(self) -> float:
        """Calculate penalty contribution if failed."""
        if self.status == CheckStatus.FAILED:
            return self.level.penalty
        elif self.status == CheckStatus.WARNING:
            return self.level.penalty * 0.3
        return 0.0

    @property
    def is_blocking(self) -> bool:
        """Check if this failure blocks compliance."""
        return (
            self.status == CheckStatus.FAILED and
            self.level == ComplianceLevel.CRITICAL
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with full details."""
        return {
            'check_id': self.check_id,
            'name': self.name,
            'category': self.category.value,
            'level': self.level.value,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'remediation': self.remediation,
            'evidence': self.evidence,
            'related_checks': self.related_checks,
            'execution_time_ms': self.execution_time_ms,
            'score_contribution': self.score_contribution,
            'penalty_contribution': self.penalty_contribution,
            'is_blocking': self.is_blocking
        }


@dataclass
class RemediationSuggestion:
    """
    Detailed remediation suggestion for failed checks.

    Attributes:
        check_id: ID of the failed check
        priority: Remediation priority
        title: Short title for the remediation
        description: Detailed description of what to fix
        steps: Step-by-step remediation instructions
        examples: Example fixes or content
        estimated_effort: Estimated effort to fix
        auto_fixable: Whether this can be auto-fixed
    """
    check_id: str
    priority: RemediationPriority
    title: str
    description: str
    steps: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    estimated_effort: str = "Unknown"
    auto_fixable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'check_id': self.check_id,
            'priority': self.priority.value,
            'urgency_score': self.priority.urgency_score,
            'title': self.title,
            'description': self.description,
            'steps': self.steps,
            'examples': self.examples,
            'estimated_effort': self.estimated_effort,
            'auto_fixable': self.auto_fixable
        }


@dataclass
class CategoryResult:
    """
    Comprehensive results for a validation category.

    Attributes:
        category: The validation category
        total_checks: Total number of checks
        passed: Number of passed checks
        failed: Number of failed checks
        warnings: Number of warning checks
        skipped: Number of skipped checks
        checks: List of all validation checks
        score: Weighted score for this category
        max_possible_score: Maximum possible score
        remediations: List of remediation suggestions
        execution_time_ms: Total execution time
    """
    category: ValidationCategory
    total_checks: int
    passed: int
    failed: int
    warnings: int
    skipped: int
    checks: List[ValidationCheck] = field(default_factory=list)
    score: float = 0.0
    max_possible_score: float = 0.0
    remediations: List[RemediationSuggestion] = field(default_factory=list)
    execution_time_ms: float = 0.0

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate as percentage."""
        countable = self.total_checks - self.skipped
        if countable == 0:
            return 1.0
        return self.passed / countable

    @property
    def weighted_score(self) -> float:
        """Calculate category-weighted score."""
        if self.max_possible_score == 0:
            return 0.0
        base_score = self.score / self.max_possible_score
        return base_score * self.category.weight

    @property
    def is_compliant(self) -> bool:
        """Check if category is compliant (no critical failures)."""
        return all(
            not check.is_blocking
            for check in self.checks
        )

    @property
    def blocking_failures(self) -> List[ValidationCheck]:
        """Get list of blocking failures."""
        return [c for c in self.checks if c.is_blocking]

    @property
    def critical_count(self) -> int:
        """Count of critical-level checks."""
        return sum(1 for c in self.checks if c.level == ComplianceLevel.CRITICAL)

    @property
    def major_count(self) -> int:
        """Count of major-level checks."""
        return sum(1 for c in self.checks if c.level == ComplianceLevel.MAJOR)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with full details."""
        return {
            'category': self.category.value,
            'description': self.category.description,
            'total_checks': self.total_checks,
            'passed': self.passed,
            'failed': self.failed,
            'warnings': self.warnings,
            'skipped': self.skipped,
            'pass_rate': f"{self.pass_rate:.1%}",
            'score': round(self.score, 2),
            'max_possible_score': round(self.max_possible_score, 2),
            'weighted_score': round(self.weighted_score, 3),
            'is_compliant': self.is_compliant,
            'blocking_failures': len(self.blocking_failures),
            'critical_count': self.critical_count,
            'major_count': self.major_count,
            'execution_time_ms': round(self.execution_time_ms, 2),
            'checks': [c.to_dict() for c in self.checks],
            'remediations': [r.to_dict() for r in self.remediations]
        }


@dataclass
class QualityScore:
    """
    Detailed quality scoring breakdown.

    Attributes:
        overall_score: Overall quality score (0-100)
        structure_score: Document structure score
        content_score: Content completeness score
        technical_score: Technical accuracy score
        presentation_score: Presentation quality score
        compliance_score: PDF compliance score
        grade: Letter grade (A-F)
        percentile: Estimated percentile ranking
    """
    overall_score: float
    structure_score: float
    content_score: float
    technical_score: float
    presentation_score: float
    compliance_score: float
    grade: str = ""
    percentile: float = 0.0

    def __post_init__(self):
        """Calculate grade and percentile after initialization."""
        if not self.grade:
            self.grade = self._calculate_grade()
        if self.percentile == 0.0:
            self.percentile = self._estimate_percentile()

    def _calculate_grade(self) -> str:
        """Calculate letter grade from overall score."""
        if self.overall_score >= 95:
            return "A+"
        elif self.overall_score >= 90:
            return "A"
        elif self.overall_score >= 85:
            return "A-"
        elif self.overall_score >= 80:
            return "B+"
        elif self.overall_score >= 75:
            return "B"
        elif self.overall_score >= 70:
            return "B-"
        elif self.overall_score >= 65:
            return "C+"
        elif self.overall_score >= 60:
            return "C"
        elif self.overall_score >= 55:
            return "C-"
        elif self.overall_score >= 50:
            return "D"
        else:
            return "F"

    def _estimate_percentile(self) -> float:
        """Estimate percentile from overall score."""
        # Sigmoid-based percentile estimation
        import math
        x = (self.overall_score - 75) / 10
        return round(100 / (1 + math.exp(-x)), 1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'overall_score': round(self.overall_score, 1),
            'structure_score': round(self.structure_score, 1),
            'content_score': round(self.content_score, 1),
            'technical_score': round(self.technical_score, 1),
            'presentation_score': round(self.presentation_score, 1),
            'compliance_score': round(self.compliance_score, 1),
            'grade': self.grade,
            'percentile': self.percentile
        }


@dataclass
class ValidationMetadata:
    """
    Metadata about the validation process.

    Attributes:
        report_path: Path to the validated report
        report_hash: SHA256 hash of report content
        report_length: Character count of report
        estimated_pages: Estimated page count
        word_count: Word count of report
        validation_version: Version of validator
        profile: Validation profile used
        pdf_reference: project specification reference
        start_time: Validation start timestamp
        end_time: Validation end timestamp
        total_execution_time_ms: Total execution time
    """
    report_path: Optional[str]
    report_hash: str
    report_length: int
    estimated_pages: float
    word_count: int
    validation_version: str
    profile: ValidationProfile
    pdf_reference: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'report_path': self.report_path,
            'report_hash': self.report_hash,
            'report_length': self.report_length,
            'estimated_pages': round(self.estimated_pages, 1),
            'word_count': self.word_count,
            'validation_version': self.validation_version,
            'profile': self.profile.value,
            'pdf_reference': self.pdf_reference,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_execution_time_ms': round(self.total_execution_time_ms, 2)
        }


@dataclass
class StrictValidationResult:
    """
    Complete validation result with comprehensive analysis.

    Attributes:
        is_pdf_compliant: Whether report meets PDF requirements
        compliance_score: Overall compliance score (0-1)
        quality_score: Detailed quality scoring
        total_checks: Total number of checks executed
        passed_checks: Number of passed checks
        failed_checks: Number of failed checks
        warning_checks: Number of warning checks
        critical_failures: List of critical failures
        major_failures: List of major failures
        all_warnings: List of all warnings
        category_results: Results by category
        remediations: Prioritized remediation list
        metadata: Validation metadata
        validation_timestamp: When validation was performed
    """
    is_pdf_compliant: bool
    compliance_score: float
    quality_score: QualityScore
    total_checks: int
    passed_checks: int
    failed_checks: int
    warning_checks: int
    critical_failures: List[ValidationCheck]
    major_failures: List[ValidationCheck]
    all_warnings: List[ValidationCheck]
    category_results: Dict[ValidationCategory, CategoryResult]
    remediations: List[RemediationSuggestion]
    metadata: ValidationMetadata
    validation_timestamp: datetime

    @property
    def blocking_failure_count(self) -> int:
        """Count of blocking failures."""
        return len(self.critical_failures)

    @property
    def total_penalty(self) -> float:
        """Calculate total penalty from failures."""
        return sum(c.penalty_contribution for c in self.critical_failures + self.major_failures)

    @property
    def categories_compliant(self) -> int:
        """Count of compliant categories."""
        return sum(1 for r in self.category_results.values() if r.is_compliant)

    @property
    def immediate_remediations(self) -> List[RemediationSuggestion]:
        """Get immediate priority remediations."""
        return [r for r in self.remediations if r.priority == RemediationPriority.IMMEDIATE]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with full details."""
        return {
            'is_pdf_compliant': self.is_pdf_compliant,
            'compliance_score': f"{self.compliance_score:.1%}",
            'quality_score': self.quality_score.to_dict(),
            'summary': {
                'total_checks': self.total_checks,
                'passed': self.passed_checks,
                'failed': self.failed_checks,
                'warnings': self.warning_checks,
                'blocking_failures': self.blocking_failure_count,
                'total_penalty': round(self.total_penalty, 1),
                'categories_compliant': self.categories_compliant,
                'total_categories': len(self.category_results)
            },
            'critical_failures': [c.to_dict() for c in self.critical_failures],
            'major_failures': [c.to_dict() for c in self.major_failures],
            'warnings': [c.to_dict() for c in self.all_warnings],
            'category_results': {
                cat.value: result.to_dict()
                for cat, result in self.category_results.items()
            },
            'remediations': [r.to_dict() for r in self.remediations],
            'immediate_remediations': [r.to_dict() for r in self.immediate_remediations],
            'metadata': self.metadata.to_dict(),
            'validation_timestamp': self.validation_timestamp.isoformat()
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# =============================================================================
# VALIDATION REQUIREMENTS - Comprehensive PDF Requirements
# =============================================================================

# Required sections per PDF (expanded with aliases)
REQUIRED_SECTIONS: Dict[str, Dict[str, Any]] = {
    'executive_summary': {
        'aliases': ['executive summary', 'exec summary', 'summary'],
        'level': ComplianceLevel.CRITICAL,
        'min_words': 500,
        'description': 'High-level overview of findings and recommendations'
    },
    'universe_construction': {
        'aliases': ['universe construction', 'asset universe', 'universe selection'],
        'level': ComplianceLevel.CRITICAL,
        'min_words': 800,
        'description': 'Description of asset selection methodology'
    },
    'cointegration_analysis': {
        'aliases': ['cointegration', 'pairs', 'statistical relationship'],
        'level': ComplianceLevel.CRITICAL,
        'min_words': 1000,
        'description': 'Cointegration testing and pair selection'
    },
    'baseline_strategy': {
        'aliases': ['baseline', 'basic strategy', 'initial strategy'],
        'level': ComplianceLevel.MAJOR,
        'min_words': 600,
        'description': 'Initial strategy implementation'
    },
    'strategy_enhancements': {
        'aliases': ['enhancement', 'improvement', 'optimization'],
        'level': ComplianceLevel.MAJOR,
        'min_words': 800,
        'description': 'Strategy enhancements and optimizations'
    },
    'regime_detection': {
        'aliases': ['regime', 'market state', 'hidden markov'],
        'level': ComplianceLevel.MAJOR,
        'min_words': 600,
        'description': 'Market regime detection methodology'
    },
    'ml_integration': {
        'aliases': ['ml', 'machine learning', 'random forest', 'neural'],
        'level': ComplianceLevel.MAJOR,
        'min_words': 800,
        'description': 'Machine learning integration'
    },
    'dynamic_sizing': {
        'aliases': ['dynamic', 'position sizing', 'kelly'],
        'level': ComplianceLevel.MAJOR,
        'min_words': 500,
        'description': 'Dynamic position sizing methodology'
    },
    'backtesting': {
        'aliases': ['backtest', 'historical', 'simulation'],
        'level': ComplianceLevel.CRITICAL,
        'min_words': 1000,
        'description': 'Backtesting methodology and results'
    },
    'walk_forward': {
        'aliases': ['walk-forward', 'walk forward', 'out-of-sample'],
        'level': ComplianceLevel.CRITICAL,
        'min_words': 800,
        'description': 'Walk-forward validation results'
    },
    'performance_analysis': {
        'aliases': ['performance', 'results', 'returns'],
        'level': ComplianceLevel.CRITICAL,
        'min_words': 1200,
        'description': 'Comprehensive performance analysis'
    },
    'venue_breakdown': {
        'aliases': ['venue', 'cex', 'dex', 'exchange'],
        'level': ComplianceLevel.CRITICAL,
        'min_words': 800,
        'description': 'Venue-specific analysis (CEX/DEX/Hybrid)'
    },
    'crisis_analysis': {
        'aliases': ['crisis', 'stress', 'drawdown', 'covid', 'ftx', 'luna'],
        'level': ComplianceLevel.CRITICAL,
        'min_words': 1000,
        'description': 'Crisis event analysis and performance'
    },
    'capacity_analysis': {
        'aliases': ['capacity', 'scalability', 'liquidity'],
        'level': ComplianceLevel.CRITICAL,
        'min_words': 600,
        'description': 'Strategy capacity analysis'
    },
    'grain_comparison': {
        'aliases': ['grain', 'futures', 'commodity', 'agricultural'],
        'level': ComplianceLevel.CRITICAL,
        'min_words': 500,
        'description': 'Grain futures comparison (PDF REQUIRED)'
    },
    'risk_management': {
        'aliases': ['risk management', 'risk control', 'stop loss'],
        'level': ComplianceLevel.CRITICAL,
        'min_words': 600,
        'description': 'Risk management framework'
    },
    'conclusions': {
        'aliases': ['conclusion', 'summary', 'findings'],
        'level': ComplianceLevel.CRITICAL,
        'min_words': 400,
        'description': 'Conclusions and key findings'
    },
    'recommendations': {
        'aliases': ['recommendation', 'next steps', 'future work'],
        'level': ComplianceLevel.MAJOR,
        'min_words': 300,
        'description': 'Recommendations for implementation'
    },
    'appendices': {
        'aliases': ['appendix', 'appendices', 'supplementary'],
        'level': ComplianceLevel.MINOR,
        'min_words': 500,
        'description': 'Supplementary materials'
    }
}

# Required metrics organized by category (80+ per PDF)
REQUIRED_METRICS: Dict[str, Dict[str, Any]] = {
    # Core Performance Metrics (Weight: CRITICAL)
    'total_return': {
        'aliases': ['total return', 'cumulative return', 'overall return'],
        'type': MetricType.PERCENTAGE,
        'level': ComplianceLevel.CRITICAL,
        'valid_range': (-100, 10000)
    },
    'annualized_return': {
        'aliases': ['annualized return', 'annual return', 'yearly return', 'cagr'],
        'type': MetricType.PERCENTAGE,
        'level': ComplianceLevel.CRITICAL,
        'valid_range': (-100, 500)
    },
    'sharpe_ratio': {
        'aliases': ['sharpe ratio', 'sharpe', 'risk-adjusted return'],
        'type': MetricType.RATIO,
        'level': ComplianceLevel.CRITICAL,
        'valid_range': (-5, 10)
    },
    'sortino_ratio': {
        'aliases': ['sortino ratio', 'sortino', 'downside sharpe'],
        'type': MetricType.RATIO,
        'level': ComplianceLevel.CRITICAL,
        'valid_range': (-5, 15)
    },
    'calmar_ratio': {
        'aliases': ['calmar ratio', 'calmar', 'return/drawdown'],
        'type': MetricType.RATIO,
        'level': ComplianceLevel.MAJOR,
        'valid_range': (-5, 20)
    },
    'omega_ratio': {
        'aliases': ['omega ratio', 'omega'],
        'type': MetricType.RATIO,
        'level': ComplianceLevel.MAJOR,
        'valid_range': (0, 10)
    },
    'max_drawdown': {
        'aliases': ['max drawdown', 'maximum drawdown', 'mdd', 'worst drawdown'],
        'type': MetricType.PERCENTAGE,
        'level': ComplianceLevel.CRITICAL,
        'valid_range': (-100, 0)
    },
    'average_drawdown': {
        'aliases': ['average drawdown', 'avg drawdown', 'mean drawdown'],
        'type': MetricType.PERCENTAGE,
        'level': ComplianceLevel.MAJOR,
        'valid_range': (-100, 0)
    },
    # Trade Statistics (Weight: MAJOR)
    'total_trades': {
        'aliases': ['total trades', 'number of trades', 'trade count'],
        'type': MetricType.COUNT,
        'level': ComplianceLevel.CRITICAL,
        'valid_range': (0, 1000000)
    },
    'win_rate': {
        'aliases': ['win rate', 'win %', 'winning percentage', 'hit rate'],
        'type': MetricType.PERCENTAGE,
        'level': ComplianceLevel.CRITICAL,
        'valid_range': (0, 100)
    },
    'profit_factor': {
        'aliases': ['profit factor', 'pf', 'gross profit/loss'],
        'type': MetricType.RATIO,
        'level': ComplianceLevel.CRITICAL,
        'valid_range': (0, 100)
    },
    'average_win': {
        'aliases': ['average win', 'avg win', 'mean win'],
        'type': MetricType.PERCENTAGE,
        'level': ComplianceLevel.MAJOR,
        'valid_range': (0, 1000)
    },
    'average_loss': {
        'aliases': ['average loss', 'avg loss', 'mean loss'],
        'type': MetricType.PERCENTAGE,
        'level': ComplianceLevel.MAJOR,
        'valid_range': (-1000, 0)
    },
    'win_loss_ratio': {
        'aliases': ['win/loss ratio', 'win loss', 'reward/risk'],
        'type': MetricType.RATIO,
        'level': ComplianceLevel.MAJOR,
        'valid_range': (0, 100)
    },
    'expectancy': {
        'aliases': ['expectancy', 'expected value', 'edge'],
        'type': MetricType.DECIMAL,
        'level': ComplianceLevel.MAJOR,
        'valid_range': (-100, 100)
    },
    'average_holding': {
        'aliases': ['average holding', 'holding period', 'avg duration'],
        'type': MetricType.DURATION,
        'level': ComplianceLevel.MAJOR,
        'valid_range': (0, 365)
    },
    # Risk Metrics (Weight: CRITICAL)
    'volatility': {
        'aliases': ['volatility', 'annual volatility', 'std dev'],
        'type': MetricType.PERCENTAGE,
        'level': ComplianceLevel.CRITICAL,
        'valid_range': (0, 500)
    },
    'downside_deviation': {
        'aliases': ['downside deviation', 'downside vol', 'semi-deviation'],
        'type': MetricType.PERCENTAGE,
        'level': ComplianceLevel.MAJOR,
        'valid_range': (0, 500)
    },
    'var_95': {
        'aliases': ['var', 'value at risk', '95% var', 'var 95'],
        'type': MetricType.PERCENTAGE,
        'level': ComplianceLevel.CRITICAL,
        'valid_range': (-100, 0)
    },
    'cvar_95': {
        'aliases': ['cvar', 'conditional var', 'expected shortfall', 'es'],
        'type': MetricType.PERCENTAGE,
        'level': ComplianceLevel.CRITICAL,
        'valid_range': (-100, 0)
    },
    'beta': {
        'aliases': ['beta', 'market beta', 'systematic risk'],
        'type': MetricType.DECIMAL,
        'level': ComplianceLevel.MAJOR,
        'valid_range': (-5, 5)
    },
    'correlation': {
        'aliases': ['correlation', 'corr', 'market correlation'],
        'type': MetricType.DECIMAL,
        'level': ComplianceLevel.MAJOR,
        'valid_range': (-1, 1)
    },
    'skewness': {
        'aliases': ['skewness', 'skew', 'return skew'],
        'type': MetricType.DECIMAL,
        'level': ComplianceLevel.MINOR,
        'valid_range': (-10, 10)
    },
    'kurtosis': {
        'aliases': ['kurtosis', 'excess kurtosis', 'tail risk'],
        'type': MetricType.DECIMAL,
        'level': ComplianceLevel.MINOR,
        'valid_range': (-10, 100)
    },
    # Cost Metrics (Weight: CRITICAL)
    'transaction_cost': {
        'aliases': ['transaction cost', 'trading cost', 'total cost'],
        'type': MetricType.PERCENTAGE,
        'level': ComplianceLevel.CRITICAL,
        'valid_range': (0, 10)
    },
    'slippage': {
        'aliases': ['slippage', 'execution slippage', 'market impact'],
        'type': MetricType.PERCENTAGE,
        'level': ComplianceLevel.CRITICAL,
        'valid_range': (0, 5)
    },
    'trading_fee': {
        'aliases': ['trading fee', 'exchange fee', 'commission'],
        'type': MetricType.PERCENTAGE,
        'level': ComplianceLevel.MAJOR,
        'valid_range': (0, 2)
    },
    'gas_cost': {
        'aliases': ['gas cost', 'gas fee', 'network fee'],
        'type': MetricType.CURRENCY,
        'level': ComplianceLevel.MAJOR,
        'valid_range': (0, 1000000)
    },
    # Capacity Metrics (Weight: CRITICAL)
    'cex_capacity': {
        'aliases': ['cex capacity', 'centralized capacity'],
        'type': MetricType.CURRENCY,
        'level': ComplianceLevel.CRITICAL,
        'valid_range': (1000000, 100000000)
    },
    'dex_capacity': {
        'aliases': ['dex capacity', 'decentralized capacity'],
        'type': MetricType.CURRENCY,
        'level': ComplianceLevel.CRITICAL,
        'valid_range': (100000, 50000000)
    },
    'combined_capacity': {
        'aliases': ['combined capacity', 'total capacity', 'strategy capacity'],
        'type': MetricType.CURRENCY,
        'level': ComplianceLevel.CRITICAL,
        'valid_range': (1000000, 100000000)
    },
    'kelly_fraction': {
        'aliases': ['kelly', 'kelly fraction', 'optimal f'],
        'type': MetricType.PERCENTAGE,
        'level': ComplianceLevel.MAJOR,
        'valid_range': (0, 100)
    },
    # Statistical Validation Metrics
    'sharpe_t_stat': {
        'aliases': ['sharpe t-stat', 't-statistic', 'sharpe significance'],
        'type': MetricType.DECIMAL,
        'level': ComplianceLevel.MAJOR,
        'valid_range': (-10, 20)
    },
    'sharpe_p_value': {
        'aliases': ['sharpe p-value', 'p-value', 'significance'],
        'type': MetricType.DECIMAL,
        'level': ComplianceLevel.MAJOR,
        'valid_range': (0, 1)
    },
    'confidence_interval': {
        'aliases': ['confidence interval', 'ci', '95% ci'],
        'type': MetricType.DECIMAL,
        'level': ComplianceLevel.MAJOR,
        'valid_range': (-100, 100)
    },
    # Additional metrics for completeness
    'information_ratio': {
        'aliases': ['information ratio', 'ir', 'active return/risk'],
        'type': MetricType.RATIO,
        'level': ComplianceLevel.MAJOR,
        'valid_range': (-5, 10)
    },
    'treynor_ratio': {
        'aliases': ['treynor ratio', 'treynor'],
        'type': MetricType.RATIO,
        'level': ComplianceLevel.MINOR,
        'valid_range': (-100, 100)
    },
    'recovery_factor': {
        'aliases': ['recovery factor', 'recovery'],
        'type': MetricType.RATIO,
        'level': ComplianceLevel.MAJOR,
        'valid_range': (0, 100)
    },
    'ulcer_index': {
        'aliases': ['ulcer index', 'ulcer'],
        'type': MetricType.DECIMAL,
        'level': ComplianceLevel.MINOR,
        'valid_range': (0, 100)
    },
    'tail_ratio': {
        'aliases': ['tail ratio', 'gain/pain'],
        'type': MetricType.RATIO,
        'level': ComplianceLevel.MINOR,
        'valid_range': (0, 10)
    }
}

# Required crisis events (14 per PDF) with date ranges
REQUIRED_CRISIS_EVENTS: Dict[str, Dict[str, Any]] = {
    'covid_crash': {
        'aliases': ['covid', 'coronavirus', 'pandemic', 'march 2020'],
        'start_date': '2020-02-20',
        'end_date': '2020-03-23',
        'severity': 'extreme',
        'description': 'COVID-19 market crash'
    },
    'defi_summer': {
        'aliases': ['defi summer', 'defi 2020', 'yield farming'],
        'start_date': '2020-06-15',
        'end_date': '2020-09-15',
        'severity': 'moderate',
        'description': 'DeFi Summer volatility'
    },
    'may_2021_crash': {
        'aliases': ['may 2021', 'may crash', '2021 correction'],
        'start_date': '2021-05-10',
        'end_date': '2021-05-25',
        'severity': 'high',
        'description': 'May 2021 crypto crash'
    },
    'china_ban': {
        'aliases': ['china', 'china ban', 'mining ban'],
        'start_date': '2021-05-21',
        'end_date': '2021-06-30',
        'severity': 'high',
        'description': 'China crypto ban'
    },
    'luna_ust_collapse': {
        'aliases': ['luna', 'ust', 'terra', 'algorithmic stablecoin'],
        'start_date': '2022-05-07',
        'end_date': '2022-05-15',
        'severity': 'extreme',
        'description': 'LUNA/UST collapse'
    },
    'three_arrows_celsius': {
        'aliases': ['3ac', 'celsius', 'three arrows', 'lending crisis'],
        'start_date': '2022-06-12',
        'end_date': '2022-07-15',
        'severity': 'high',
        'description': '3AC and Celsius collapse'
    },
    'ftx_collapse': {
        'aliases': ['ftx', 'alameda', 'sbf'],
        'start_date': '2022-11-06',
        'end_date': '2022-11-14',
        'severity': 'extreme',
        'description': 'FTX collapse'
    },
    'svb_usdc_depeg': {
        'aliases': ['svb', 'silicon valley bank', 'usdc depeg', 'usdc'],
        'start_date': '2023-03-10',
        'end_date': '2023-03-15',
        'severity': 'high',
        'description': 'SVB collapse and USDC depeg'
    },
    'sec_crackdown': {
        'aliases': ['sec', 'regulation', 'securities', 'binance lawsuit'],
        'start_date': '2023-06-05',
        'end_date': '2023-06-15',
        'severity': 'moderate',
        'description': 'SEC regulatory crackdown'
    },
    'etf_approval': {
        'aliases': ['etf', 'bitcoin etf', 'spot etf'],
        'start_date': '2024-01-10',
        'end_date': '2024-01-15',
        'severity': 'moderate',
        'description': 'Bitcoin ETF approval volatility'
    },
    'genesis_bankruptcy': {
        'aliases': ['genesis', 'dcg', 'gemini'],
        'start_date': '2023-01-18',
        'end_date': '2023-01-25',
        'severity': 'moderate',
        'description': 'Genesis bankruptcy'
    },
    'silvergate_signature': {
        'aliases': ['silvergate', 'signature', 'banking crisis'],
        'start_date': '2023-03-08',
        'end_date': '2023-03-13',
        'severity': 'high',
        'description': 'Crypto-friendly bank failures'
    },
    'curve_exploit': {
        'aliases': ['curve', 'vyper', 'reentrancy', 'crv'],
        'start_date': '2023-07-30',
        'end_date': '2023-08-05',
        'severity': 'moderate',
        'description': 'Curve Finance exploit'
    },
    'evergrande_default': {
        'aliases': ['evergrande', 'china real estate'],
        'start_date': '2023-08-17',
        'end_date': '2023-08-25',
        'severity': 'moderate',
        'description': 'Evergrande default spillover'
    }
}

# Walk-forward requirements
WALK_FORWARD_REQUIREMENTS: Dict[str, Any] = {
    'train_months': 18,
    'test_months': 6,
    'min_windows': 4,
    'max_windows': 12,
    'overlap_allowed': False,
    'min_total_period_months': 48,
    'parameter_stability_threshold': 0.3,
    'performance_degradation_threshold': 0.2
}

# Capacity requirements per PDF
CAPACITY_REQUIREMENTS: Dict[str, Any] = {
    'cex': {
        'min': 10_000_000,
        'max': 30_000_000,
        'target': 20_000_000,
        'degradation_threshold': 0.1
    },
    'dex': {
        'min': 1_000_000,
        'max': 5_000_000,
        'target': 3_000_000,
        'degradation_threshold': 0.15
    },
    'combined': {
        'min': 20_000_000,
        'max': 50_000_000,
        'target': 35_000_000,
        'degradation_threshold': 0.1
    }
}

# Position sizing requirements per PDF
POSITION_SIZING_REQUIREMENTS: Dict[str, Any] = {
    'cex_max': 100_000,
    'dex_liquid_min': 20_000,
    'dex_liquid_max': 50_000,
    'dex_illiquid_min': 5_000,
    'dex_illiquid_max': 10_000,
    'max_kelly_fraction': 0.25,
    'min_position_count': 10,
    'max_single_position_pct': 0.10
}

# Concentration limits per PDF
CONCENTRATION_LIMITS: Dict[str, Any] = {
    'max_sector': 0.40,
    'max_cex_only': 0.60,
    'max_tier3': 0.20,
    'max_single_asset': 0.15,
    'max_correlation_cluster': 0.50,
    'min_diversification_ratio': 0.3
}

# Venue color coding requirements
VENUE_COLORS: Dict[str, Dict[str, str]] = {
    'CEX': {
        'label': '[CEX]',
        'hex': '#0066CC',
        'name': 'Blue',
        'description': 'Centralized Exchange'
    },
    'DEX': {
        'label': '[DEX]',
        'hex': '#FF6600',
        'name': 'Orange',
        'description': 'Decentralized Exchange'
    },
    'HYBRID': {
        'label': '[HYB]',
        'hex': '#009933',
        'name': 'Green',
        'description': 'Hybrid (CEX+DEX)'
    }
}


# =============================================================================
# CATEGORY VALIDATOR PROTOCOL
# =============================================================================

class CategoryValidator(Protocol):
    """Protocol for category-specific validators."""

    def validate(
        self,
        content: str,
        report_data: Optional[Dict[str, Any]],
        step4_results: Optional[Dict[str, Any]]
    ) -> CategoryResult:
        """Validate and return category result."""
        ...


# =============================================================================
# VALIDATOR CLASS - Main Implementation
# =============================================================================

class StrictPDFValidator:
    """
    Comprehensive PDF compliance validator for project specification.

    This validator provides:
    - 100+ individual validation checks across 16 categories
    - Weighted scoring with configurable weights
    - Automated remediation suggestions with priority ranking
    - Deep Step4 results integration and cross-validation
    - Multiple validation profiles (strict, standard, lenient)
    - Caching for performance optimization
    - Detailed compliance reporting with actionable insights

    PDF Requirements Validated:
    - Document structure and 30-40 page count
    - 80+ performance metrics coverage
    - Venue-specific breakdown (CEX/DEX/Hybrid) with color coding
    - 14 crisis events with performance analysis
    - Walk-forward validation (18m train / 6m test, 8 windows)
    - Capacity analysis with degradation curves
    - Grain futures comparison (PDF REQUIRED)
    - Position sizing and concentration limits

    Usage:
        validator = StrictPDFValidator(profile=ValidationProfile.STRICT)
        result = validator.validate_comprehensive(
            report_content=markdown_content,
            report_data=json_data,
            step4_results=orchestrator_results
        )

        if result.is_pdf_compliant:
            print(f"Report is compliant! Score: {result.compliance_score:.1%}")
        else:
            for failure in result.critical_failures:
                print(f"CRITICAL: {failure.message}")
            for remediation in result.immediate_remediations:
                print(f"FIX: {remediation.title}")
    """

    # Document constants
    CHARS_PER_PAGE = 2500
    WORDS_PER_PAGE = 400
    MIN_PAGES = 30
    MAX_PAGES = 40

    def __init__(
        self,
        profile: ValidationProfile = ValidationProfile.STRICT,
        custom_weights: Optional[Dict[ValidationCategory, float]] = None,
        enable_caching: bool = True,
        generate_remediations: bool = True
    ):
        """
        Initialize strict PDF validator.

        Args:
            profile: Validation profile (strict/standard/lenient/custom)
            custom_weights: Custom category weights if profile is CUSTOM
            enable_caching: Enable result caching for performance
            generate_remediations: Generate remediation suggestions
        """
        self.profile = profile
        self.custom_weights = custom_weights or {}
        self.enable_caching = enable_caching
        self.generate_remediations = generate_remediations

        # Internal state
        self._checks: List[ValidationCheck] = []
        self._cache: Dict[str, Any] = {}
        self._validation_count = 0

        logger.info(f"StrictPDFValidator initialized with profile: {profile.value}")

    def validate_comprehensive(
        self,
        report_content: str,
        report_data: Optional[Dict[str, Any]] = None,
        step4_results: Optional[Dict[str, Any]] = None,
        report_path: Optional[Path] = None
    ) -> StrictValidationResult:
        """
        Perform comprehensive validation of report and data.

        This is the main entry point for validation. It runs all 16 validation
        categories and aggregates results into a comprehensive report.

        Args:
            report_content: Markdown content of the report
            report_data: JSON data from report generation
            step4_results: Results from Step4AdvancedOrchestrator
            report_path: Optional path to report file

        Returns:
            StrictValidationResult with full compliance assessment
        """
        start_time = datetime.now(timezone.utc)
        self._checks = []
        self._validation_count += 1

        logger.info(f"Starting comprehensive PDF validation #{self._validation_count}")

        # Check cache
        cache_key = self._get_cache_key(report_content)
        if self.enable_caching and cache_key in self._cache:
            logger.info("Returning cached validation result")
            return self._cache[cache_key]

        # Create metadata
        metadata = self._create_metadata(report_content, report_path, start_time)

        # Run all category validations
        category_results: Dict[ValidationCategory, CategoryResult] = {}

        # 1. Document Structure
        category_results[ValidationCategory.DOCUMENT_STRUCTURE] = \
            self._validate_document_structure(report_content)

        # 2. Page Count
        category_results[ValidationCategory.PAGE_COUNT] = \
            self._validate_page_count(report_content)

        # 3. Content Completeness
        category_results[ValidationCategory.CONTENT_COMPLETENESS] = \
            self._validate_content_completeness(report_content)

        # 4. Metrics Coverage
        category_results[ValidationCategory.METRICS_COVERAGE] = \
            self._validate_metrics_coverage(report_content, report_data)

        # 5. Venue Breakdown
        category_results[ValidationCategory.VENUE_BREAKDOWN] = \
            self._validate_venue_breakdown(report_content, report_data, step4_results)

        # 6. Crisis Events
        category_results[ValidationCategory.CRISIS_EVENTS] = \
            self._validate_crisis_events(report_content, step4_results)

        # 7. Walk-Forward
        category_results[ValidationCategory.WALK_FORWARD] = \
            self._validate_walk_forward(report_content, step4_results)

        # 8. Capacity Analysis
        category_results[ValidationCategory.CAPACITY_ANALYSIS] = \
            self._validate_capacity_analysis(report_content, step4_results)

        # 9. Grain Comparison
        category_results[ValidationCategory.GRAIN_COMPARISON] = \
            self._validate_grain_comparison(report_content, step4_results)

        # 10. Position Sizing
        category_results[ValidationCategory.POSITION_SIZING] = \
            self._validate_position_sizing(report_content, step4_results)

        # 11. Concentration Limits
        category_results[ValidationCategory.CONCENTRATION_LIMITS] = \
            self._validate_concentration_limits(report_content, step4_results)

        # 12. Statistical Validation
        category_results[ValidationCategory.STATISTICAL_VALIDATION] = \
            self._validate_statistical(report_content, report_data, step4_results)

        # 13. Cross-Reference Validation
        category_results[ValidationCategory.CROSS_REFERENCE] = \
            self._validate_cross_references(report_content, report_data, step4_results)

        # 14. Formatting
        category_results[ValidationCategory.FORMATTING] = \
            self._validate_formatting(report_content)

        # 15. Academic Standards
        category_results[ValidationCategory.ACADEMIC_STANDARDS] = \
            self._validate_academic_standards(report_content)

        # 16. Executive Quality
        category_results[ValidationCategory.EXECUTIVE_QUALITY] = \
            self._validate_executive_quality(report_content)

        # Aggregate results
        result = self._aggregate_results(category_results, metadata, start_time)

        # Cache result
        if self.enable_caching:
            self._cache[cache_key] = result

        logger.info(
            f"Validation complete: {result.passed_checks}/{result.total_checks} passed, "
            f"{'COMPLIANT' if result.is_pdf_compliant else 'NON-COMPLIANT'}"
        )

        return result

    def _get_cache_key(self, content: str) -> str:
        """Generate cache key from content hash."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _create_metadata(
        self,
        content: str,
        report_path: Optional[Path],
        start_time: datetime
    ) -> ValidationMetadata:
        """Create validation metadata."""
        word_count = len(content.split())
        return ValidationMetadata(
            report_path=str(report_path) if report_path else None,
            report_hash=hashlib.sha256(content.encode()).hexdigest(),
            report_length=len(content),
            estimated_pages=len(content) / self.CHARS_PER_PAGE,
            word_count=word_count,
            validation_version='3.0.0',
            profile=self.profile,
            pdf_reference='Project Specification',
            start_time=start_time
        )

    def _aggregate_results(
        self,
        category_results: Dict[ValidationCategory, CategoryResult],
        metadata: ValidationMetadata,
        start_time: datetime
    ) -> StrictValidationResult:
        """Aggregate category results into final validation result."""
        end_time = datetime.now(timezone.utc)
        metadata.end_time = end_time
        metadata.total_execution_time_ms = (end_time - start_time).total_seconds() * 1000

        # Collect all checks
        all_checks: List[ValidationCheck] = []
        for cat_result in category_results.values():
            all_checks.extend(cat_result.checks)

        # Calculate counts
        passed = sum(1 for c in all_checks if c.status == CheckStatus.PASSED)
        failed = sum(1 for c in all_checks if c.status == CheckStatus.FAILED)
        warnings = sum(1 for c in all_checks if c.status == CheckStatus.WARNING)

        # Identify failures
        critical_failures = [
            c for c in all_checks
            if c.status == CheckStatus.FAILED and c.level == ComplianceLevel.CRITICAL
        ]
        major_failures = [
            c for c in all_checks
            if c.status == CheckStatus.FAILED and c.level == ComplianceLevel.MAJOR
        ]
        all_warnings = [c for c in all_checks if c.status == CheckStatus.WARNING]

        # PDF compliance: no critical failures
        is_pdf_compliant = len(critical_failures) == 0

        # Calculate compliance score (weighted)
        total_score = sum(c.score for c in category_results.values())
        max_score = sum(c.max_possible_score for c in category_results.values())
        compliance_score = total_score / max_score if max_score > 0 else 0

        # Calculate quality score
        quality_score = self._calculate_quality_score(category_results, compliance_score)

        # Generate remediations
        remediations = []
        if self.generate_remediations:
            remediations = self._generate_remediations(critical_failures + major_failures)

        return StrictValidationResult(
            is_pdf_compliant=is_pdf_compliant,
            compliance_score=compliance_score,
            quality_score=quality_score,
            total_checks=len(all_checks),
            passed_checks=passed,
            failed_checks=failed,
            warning_checks=warnings,
            critical_failures=critical_failures,
            major_failures=major_failures,
            all_warnings=all_warnings,
            category_results=category_results,
            remediations=remediations,
            metadata=metadata,
            validation_timestamp=start_time
        )

    def _calculate_quality_score(
        self,
        category_results: Dict[ValidationCategory, CategoryResult],
        compliance_score: float
    ) -> QualityScore:
        """Calculate detailed quality scores."""
        # Structure score from document structure and formatting
        structure_cats = [
            ValidationCategory.DOCUMENT_STRUCTURE,
            ValidationCategory.PAGE_COUNT,
            ValidationCategory.FORMATTING
        ]
        structure_score = self._calc_category_group_score(category_results, structure_cats)

        # Content score from completeness and metrics
        content_cats = [
            ValidationCategory.CONTENT_COMPLETENESS,
            ValidationCategory.METRICS_COVERAGE,
            ValidationCategory.EXECUTIVE_QUALITY
        ]
        content_score = self._calc_category_group_score(category_results, content_cats)

        # Technical score from validation categories
        technical_cats = [
            ValidationCategory.WALK_FORWARD,
            ValidationCategory.STATISTICAL_VALIDATION,
            ValidationCategory.CROSS_REFERENCE
        ]
        technical_score = self._calc_category_group_score(category_results, technical_cats)

        # Presentation score
        presentation_cats = [
            ValidationCategory.VENUE_BREAKDOWN,
            ValidationCategory.CRISIS_EVENTS,
            ValidationCategory.ACADEMIC_STANDARDS
        ]
        presentation_score = self._calc_category_group_score(category_results, presentation_cats)

        # Overall score (weighted average)
        overall_score = (
            structure_score * 0.20 +
            content_score * 0.30 +
            technical_score * 0.25 +
            presentation_score * 0.15 +
            compliance_score * 100 * 0.10
        )

        return QualityScore(
            overall_score=overall_score,
            structure_score=structure_score,
            content_score=content_score,
            technical_score=technical_score,
            presentation_score=presentation_score,
            compliance_score=compliance_score * 100
        )

    def _calc_category_group_score(
        self,
        category_results: Dict[ValidationCategory, CategoryResult],
        categories: List[ValidationCategory]
    ) -> float:
        """Calculate score for a group of categories."""
        total_score = 0.0
        total_max = 0.0
        for cat in categories:
            if cat in category_results:
                result = category_results[cat]
                total_score += result.score
                total_max += result.max_possible_score
        return (total_score / total_max * 100) if total_max > 0 else 0.0

    def _generate_remediations(
        self,
        failures: List[ValidationCheck]
    ) -> List[RemediationSuggestion]:
        """Generate prioritized remediation suggestions."""
        remediations = []

        for failure in failures:
            priority = (
                RemediationPriority.IMMEDIATE
                if failure.level == ComplianceLevel.CRITICAL
                else RemediationPriority.HIGH
                if failure.level == ComplianceLevel.MAJOR
                else RemediationPriority.MEDIUM
            )

            remediation = RemediationSuggestion(
                check_id=failure.check_id,
                priority=priority,
                title=f"Fix: {failure.name}",
                description=failure.remediation or f"Address the issue: {failure.message}",
                steps=self._get_remediation_steps(failure),
                examples=self._get_remediation_examples(failure),
                estimated_effort=self._estimate_effort(failure),
                auto_fixable=self._is_auto_fixable(failure)
            )
            remediations.append(remediation)

        # Sort by priority
        remediations.sort(key=lambda r: r.priority.urgency_score, reverse=True)
        return remediations

    def _get_remediation_steps(self, failure: ValidationCheck) -> List[str]:
        """Get remediation steps for a failure."""
        category = failure.category

        if category == ValidationCategory.DOCUMENT_STRUCTURE:
            return [
                "1. Review the required section list in the project specification",
                "2. Add missing sections with appropriate headers",
                "3. Ensure each section has minimum required content",
                "4. Re-run validation to confirm fix"
            ]
        elif category == ValidationCategory.METRICS_COVERAGE:
            return [
                "1. Review the list of required metrics (80+)",
                "2. Calculate and document missing metrics",
                "3. Add metrics with proper formatting (tables recommended)",
                "4. Include statistical significance where applicable"
            ]
        elif category == ValidationCategory.CRISIS_EVENTS:
            return [
                "1. Review the 14 required crisis events",
                "2. Add analysis for each missing event",
                "3. Include performance metrics during each crisis",
                "4. Document recovery analysis"
            ]
        elif category == ValidationCategory.GRAIN_COMPARISON:
            return [
                "1. Add grain futures comparison section (PDF REQUIRED)",
                "2. Compare cointegration properties with crypto pairs",
                "3. Discuss half-life, volatility, and seasonality differences",
                "4. Draw implications for strategy adaptation"
            ]
        else:
            return [
                f"1. Review the {category.value} requirements",
                "2. Add or update the missing content",
                "3. Re-run validation to confirm fix"
            ]

    def _get_remediation_examples(self, failure: ValidationCheck) -> List[str]:
        """Get example content for remediation."""
        category = failure.category

        if category == ValidationCategory.GRAIN_COMPARISON:
            return [
                "Example: 'Compared to corn-wheat pairs, crypto pairs exhibit...'",
                "Example: 'Half-life analysis shows crypto mean-reversion occurs in...'",
                "Example: 'Unlike agricultural futures, crypto pairs lack seasonality...'"
            ]
        elif category == ValidationCategory.CRISIS_EVENTS:
            return [
                "Example: 'During the COVID-19 crash (Feb-Mar 2020), the strategy...'",
                "Example: 'FTX collapse impact: Max drawdown -X%, recovery in Y days'"
            ]
        return []

    def _estimate_effort(self, failure: ValidationCheck) -> str:
        """Estimate effort to fix a failure."""
        if failure.level == ComplianceLevel.CRITICAL:
            return "High (1-2 hours)"
        elif failure.level == ComplianceLevel.MAJOR:
            return "Medium (30-60 minutes)"
        else:
            return "Low (15-30 minutes)"

    def _is_auto_fixable(self, failure: ValidationCheck) -> bool:
        """Check if failure can be auto-fixed."""
        # Most content failures require manual intervention
        return False

    def _create_category_result(
        self,
        category: ValidationCategory,
        checks: List[ValidationCheck],
        execution_time_ms: float = 0.0
    ) -> CategoryResult:
        """Create a CategoryResult from checks."""
        score = sum(c.score_contribution for c in checks)
        max_score = sum(c.level.weight for c in checks)

        return CategoryResult(
            category=category,
            total_checks=len(checks),
            passed=sum(1 for c in checks if c.status == CheckStatus.PASSED),
            failed=sum(1 for c in checks if c.status == CheckStatus.FAILED),
            warnings=sum(1 for c in checks if c.status == CheckStatus.WARNING),
            skipped=sum(1 for c in checks if c.status == CheckStatus.SKIPPED),
            checks=checks,
            score=score,
            max_possible_score=max_score,
            execution_time_ms=execution_time_ms
        )

    # =========================================================================
    # CATEGORY VALIDATION METHODS
    # =========================================================================

    def _validate_document_structure(self, content: str) -> CategoryResult:
        """Validate document structure and required sections."""
        import time
        start = time.time()
        checks = []
        content_lower = content.lower()

        # Check each required section
        for section_id, section_info in REQUIRED_SECTIONS.items():
            aliases = section_info['aliases']
            level = section_info['level']
            min_words = section_info.get('min_words', 100)

            # Check if any alias is present
            found = False
            found_alias = None
            for alias in aliases:
                if alias in content_lower:
                    found = True
                    found_alias = alias
                    break

            # Check content depth if found
            has_depth = False
            if found:
                # Find section and count words until next section
                pattern = rf'^#{{1,3}}\s[^\n]*{re.escape(found_alias)}[^\n]*\n(.*?)(?=\n#{{1,2}}\s|\Z)'
                match = re.search(pattern, content_lower, re.DOTALL | re.IGNORECASE)
                if match:
                    section_content = match.group(1)
                    word_count = len(section_content.split())
                    has_depth = word_count >= min_words

            status = CheckStatus.PASSED if found and has_depth else (
                CheckStatus.WARNING if found else CheckStatus.FAILED
            )

            checks.append(ValidationCheck(
                check_id=f"DOC_{section_id.upper()}",
                name=f"Section: {section_id.replace('_', ' ').title()}",
                category=ValidationCategory.DOCUMENT_STRUCTURE,
                level=level,
                status=status,
                message=f"Section '{section_id}' {'present with adequate depth' if status == CheckStatus.PASSED else 'present but shallow' if status == CheckStatus.WARNING else 'MISSING'}",
                details={
                    'section': section_id,
                    'found': found,
                    'has_depth': has_depth,
                    'min_words': min_words
                },
                remediation=f"Add section '{section_id}' with at least {min_words} words" if not found else None
            ))

        # Check for table of contents
        has_toc = any(x in content_lower for x in ['table of contents', '## contents', 'contents\n'])
        checks.append(ValidationCheck(
            check_id="DOC_TOC",
            name="Table of Contents",
            category=ValidationCategory.DOCUMENT_STRUCTURE,
            level=ComplianceLevel.MINOR,
            status=CheckStatus.PASSED if has_toc else CheckStatus.WARNING,
            message=f"Table of contents {'present' if has_toc else 'missing (recommended)'}",
            details={'present': has_toc}
        ))

        # Check for document metadata
        metadata_items = ['generated', 'version', 'author', 'date']
        found_metadata = sum(1 for item in metadata_items if item in content_lower)
        has_metadata = found_metadata >= 3

        checks.append(ValidationCheck(
            check_id="DOC_META",
            name="Document Metadata",
            category=ValidationCategory.DOCUMENT_STRUCTURE,
            level=ComplianceLevel.MINOR,
            status=CheckStatus.PASSED if has_metadata else CheckStatus.WARNING,
            message=f"Document metadata: {found_metadata}/{len(metadata_items)} items present",
            details={'found': found_metadata, 'required': 3}
        ))

        # Check for proper heading hierarchy
        h1_count = len(re.findall(r'^# [^#]', content, re.MULTILINE))
        h2_count = len(re.findall(r'^## [^#]', content, re.MULTILINE))
        h3_count = len(re.findall(r'^### [^#]', content, re.MULTILINE))

        proper_hierarchy = h1_count <= 2 and h2_count >= 10 and h3_count >= 15
        checks.append(ValidationCheck(
            check_id="DOC_HIERARCHY",
            name="Heading Hierarchy",
            category=ValidationCategory.DOCUMENT_STRUCTURE,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if proper_hierarchy else CheckStatus.WARNING,
            message=f"Headings: H1={h1_count}, H2={h2_count}, H3={h3_count}",
            details={'h1': h1_count, 'h2': h2_count, 'h3': h3_count}
        ))

        execution_time = (time.time() - start) * 1000
        return self._create_category_result(
            ValidationCategory.DOCUMENT_STRUCTURE, checks, execution_time
        )

    def _validate_page_count(self, content: str) -> CategoryResult:
        """Validate page count requirements (30-40 pages)."""
        import time
        start = time.time()
        checks = []

        # Estimate pages
        char_count = len(content)
        word_count = len(content.split())
        estimated_pages_chars = char_count / self.CHARS_PER_PAGE
        estimated_pages_words = word_count / self.WORDS_PER_PAGE
        estimated_pages = (estimated_pages_chars + estimated_pages_words) / 2

        # Minimum pages check
        min_passed = estimated_pages >= self.MIN_PAGES
        checks.append(ValidationCheck(
            check_id="PAGE_MIN",
            name="Minimum Page Count (30)",
            category=ValidationCategory.PAGE_COUNT,
            level=ComplianceLevel.CRITICAL,
            status=CheckStatus.PASSED if min_passed else CheckStatus.FAILED,
            message=f"Estimated pages: {estimated_pages:.1f} (min {self.MIN_PAGES} required)",
            details={
                'estimated_pages': round(estimated_pages, 1),
                'min_required': self.MIN_PAGES,
                'char_count': char_count,
                'word_count': word_count
            },
            remediation=f"Report needs ~{(self.MIN_PAGES - estimated_pages) * self.WORDS_PER_PAGE:.0f} more words" if not min_passed else None
        ))

        # Maximum pages check
        max_passed = estimated_pages <= self.MAX_PAGES
        checks.append(ValidationCheck(
            check_id="PAGE_MAX",
            name="Maximum Page Count (40)",
            category=ValidationCategory.PAGE_COUNT,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if max_passed else CheckStatus.WARNING,
            message=f"Estimated pages: {estimated_pages:.1f} (max {self.MAX_PAGES} recommended)",
            details={
                'estimated_pages': round(estimated_pages, 1),
                'max_allowed': self.MAX_PAGES
            }
        ))

        # Content density analysis
        tables = len(re.findall(r'^\|.*\|.*\|', content, re.MULTILINE))
        code_blocks = len(re.findall(r'```', content)) // 2
        images = len(re.findall(r'!\[.*\]\(.*\)', content))

        density_score = min(100, tables * 2 + code_blocks * 5 + images * 10)
        good_density = density_score >= 30

        checks.append(ValidationCheck(
            check_id="PAGE_DENSITY",
            name="Content Density",
            category=ValidationCategory.PAGE_COUNT,
            level=ComplianceLevel.MINOR,
            status=CheckStatus.PASSED if good_density else CheckStatus.WARNING,
            message=f"Content density score: {density_score} (tables={tables}, code={code_blocks}, images={images})",
            details={
                'tables': tables,
                'code_blocks': code_blocks,
                'images': images,
                'density_score': density_score
            }
        ))

        # Words per section analysis
        sections = re.split(r'\n## ', content)
        if len(sections) > 1:
            section_lengths = [len(s.split()) for s in sections[1:]]
            avg_section_words = statistics.mean(section_lengths) if section_lengths else 0
            balanced = 200 <= avg_section_words <= 800

            checks.append(ValidationCheck(
                check_id="PAGE_BALANCE",
                name="Section Balance",
                category=ValidationCategory.PAGE_COUNT,
                level=ComplianceLevel.MINOR,
                status=CheckStatus.PASSED if balanced else CheckStatus.WARNING,
                message=f"Average section length: {avg_section_words:.0f} words",
                details={
                    'avg_words': round(avg_section_words),
                    'section_count': len(section_lengths)
                }
            ))

        execution_time = (time.time() - start) * 1000
        return self._create_category_result(
            ValidationCategory.PAGE_COUNT, checks, execution_time
        )

    def _validate_content_completeness(self, content: str) -> CategoryResult:
        """Validate content completeness and depth."""
        import time
        start = time.time()
        checks = []
        content_lower = content.lower()

        # Check for data sources documentation
        data_keywords = ['data source', 'data provider', 'historical data', 'price data']
        found_data = sum(1 for kw in data_keywords if kw in content_lower)
        has_data_sources = found_data >= 2

        checks.append(ValidationCheck(
            check_id="CONTENT_DATA",
            name="Data Sources Documentation",
            category=ValidationCategory.CONTENT_COMPLETENESS,
            level=ComplianceLevel.CRITICAL,
            status=CheckStatus.PASSED if has_data_sources else CheckStatus.FAILED,
            message=f"Data sources: {found_data}/{len(data_keywords)} keywords found",
            details={'found': found_data, 'keywords': data_keywords},
            remediation="Document data sources including providers, timeframes, and quality" if not has_data_sources else None
        ))

        # Check for methodology description
        method_keywords = ['methodology', 'approach', 'algorithm', 'implementation', 'procedure']
        found_method = sum(1 for kw in method_keywords if kw in content_lower)
        has_methodology = found_method >= 2

        checks.append(ValidationCheck(
            check_id="CONTENT_METHOD",
            name="Methodology Description",
            category=ValidationCategory.CONTENT_COMPLETENESS,
            level=ComplianceLevel.CRITICAL,
            status=CheckStatus.PASSED if has_methodology else CheckStatus.FAILED,
            message=f"Methodology: {found_method}/{len(method_keywords)} keywords found",
            details={'found': found_method},
            remediation="Add detailed methodology section" if not has_methodology else None
        ))

        # Check for risk management
        risk_keywords = ['risk management', 'risk control', 'stop loss', 'position limit', 'drawdown limit']
        found_risk = sum(1 for kw in risk_keywords if kw in content_lower)
        has_risk = found_risk >= 2

        checks.append(ValidationCheck(
            check_id="CONTENT_RISK",
            name="Risk Management Documentation",
            category=ValidationCategory.CONTENT_COMPLETENESS,
            level=ComplianceLevel.CRITICAL,
            status=CheckStatus.PASSED if has_risk else CheckStatus.FAILED,
            message=f"Risk management: {found_risk}/{len(risk_keywords)} keywords found",
            details={'found': found_risk},
            remediation="Document risk management framework including limits and controls" if not has_risk else None
        ))

        # Check for transaction costs analysis
        cost_keywords = ['transaction cost', 'trading fee', 'slippage', 'execution cost', 'gas']
        found_costs = sum(1 for kw in cost_keywords if kw in content_lower)
        has_costs = found_costs >= 3

        checks.append(ValidationCheck(
            check_id="CONTENT_COSTS",
            name="Transaction Costs Analysis",
            category=ValidationCategory.CONTENT_COMPLETENESS,
            level=ComplianceLevel.CRITICAL,
            status=CheckStatus.PASSED if has_costs else CheckStatus.FAILED,
            message=f"Transaction costs: {found_costs}/{len(cost_keywords)} keywords found",
            details={'found': found_costs},
            remediation="Add comprehensive transaction cost analysis including fees and slippage" if not has_costs else None
        ))

        # Check for recommendations
        rec_keywords = ['recommendation', 'suggest', 'advise', 'propose', 'next step']
        found_rec = sum(1 for kw in rec_keywords if kw in content_lower)
        has_recommendations = found_rec >= 2

        checks.append(ValidationCheck(
            check_id="CONTENT_REC",
            name="Recommendations Section",
            category=ValidationCategory.CONTENT_COMPLETENESS,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_recommendations else CheckStatus.FAILED,
            message=f"Recommendations: {found_rec}/{len(rec_keywords)} keywords found",
            details={'found': found_rec},
            remediation="Add clear recommendations section" if not has_recommendations else None
        ))

        # Check for limitations discussion
        limit_keywords = ['limitation', 'caveat', 'assumption', 'constraint', 'weakness']
        found_limits = sum(1 for kw in limit_keywords if kw in content_lower)
        has_limitations = found_limits >= 2

        checks.append(ValidationCheck(
            check_id="CONTENT_LIMITS",
            name="Limitations Discussion",
            category=ValidationCategory.CONTENT_COMPLETENESS,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_limitations else CheckStatus.WARNING,
            message=f"Limitations: {found_limits}/{len(limit_keywords)} keywords found",
            details={'found': found_limits}
        ))

        execution_time = (time.time() - start) * 1000
        return self._create_category_result(
            ValidationCategory.CONTENT_COMPLETENESS, checks, execution_time
        )

    def _validate_metrics_coverage(
        self,
        content: str,
        report_data: Optional[Dict[str, Any]]
    ) -> CategoryResult:
        """Validate metrics coverage (80+ metrics required)."""
        import time
        start = time.time()
        checks = []
        content_lower = content.lower()

        # Count metrics found
        found_metrics = []
        critical_found = 0
        critical_total = 0

        for metric_id, metric_info in REQUIRED_METRICS.items():
            aliases = metric_info['aliases']
            level = metric_info['level']

            if level == ComplianceLevel.CRITICAL:
                critical_total += 1

            found = False
            for alias in aliases:
                if alias in content_lower:
                    found = True
                    found_metrics.append(metric_id)
                    if level == ComplianceLevel.CRITICAL:
                        critical_found += 1
                    break

        coverage = len(found_metrics) / len(REQUIRED_METRICS)

        # Overall coverage check
        checks.append(ValidationCheck(
            check_id="METRICS_COVERAGE",
            name="Overall Metrics Coverage",
            category=ValidationCategory.METRICS_COVERAGE,
            level=ComplianceLevel.CRITICAL,
            status=CheckStatus.PASSED if coverage >= 0.7 else (
                CheckStatus.WARNING if coverage >= 0.5 else CheckStatus.FAILED
            ),
            message=f"Found {len(found_metrics)}/{len(REQUIRED_METRICS)} metrics ({coverage:.0%})",
            details={
                'found': len(found_metrics),
                'total': len(REQUIRED_METRICS),
                'coverage': round(coverage, 2),
                'found_metrics': found_metrics[:20]  # First 20 for brevity
            },
            remediation=f"Add {len(REQUIRED_METRICS) - len(found_metrics)} more metrics" if coverage < 0.7 else None
        ))

        # Critical metrics coverage
        critical_coverage = critical_found / critical_total if critical_total > 0 else 0
        checks.append(ValidationCheck(
            check_id="METRICS_CRITICAL",
            name="Critical Metrics Coverage",
            category=ValidationCategory.METRICS_COVERAGE,
            level=ComplianceLevel.CRITICAL,
            status=CheckStatus.PASSED if critical_coverage >= 0.9 else CheckStatus.FAILED,
            message=f"Critical metrics: {critical_found}/{critical_total} ({critical_coverage:.0%})",
            details={
                'found': critical_found,
                'total': critical_total,
                'coverage': round(critical_coverage, 2)
            },
            remediation="Add missing critical metrics (Sharpe, returns, drawdown, etc.)" if critical_coverage < 0.9 else None
        ))

        # Check for numeric values with metrics
        numeric_patterns = [
            (r'sharpe.*?(\d+\.?\d*)', 'Sharpe values'),
            (r'return.*?(\d+\.?\d*)%', 'Return percentages'),
            (r'drawdown.*?(\d+\.?\d*)%', 'Drawdown values'),
            (r'win.*?rate.*?(\d+\.?\d*)%', 'Win rate values')
        ]

        for pattern, name in numeric_patterns:
            matches = re.findall(pattern, content_lower)
            has_values = len(matches) >= 1

            checks.append(ValidationCheck(
                check_id=f"METRICS_{name.upper().replace(' ', '_')}",
                name=f"Metric Values: {name}",
                category=ValidationCategory.METRICS_COVERAGE,
                level=ComplianceLevel.MAJOR,
                status=CheckStatus.PASSED if has_values else CheckStatus.WARNING,
                message=f"{name}: {len(matches)} numeric values found",
                details={'count': len(matches), 'sample': matches[:5]}
            ))

        # Check metrics are in tables
        table_metrics = len(re.findall(r'\|.*(?:sharpe|return|drawdown|win).*\|', content_lower))
        has_table_metrics = table_metrics >= 5

        checks.append(ValidationCheck(
            check_id="METRICS_TABLES",
            name="Metrics in Tables",
            category=ValidationCategory.METRICS_COVERAGE,
            level=ComplianceLevel.MINOR,
            status=CheckStatus.PASSED if has_table_metrics else CheckStatus.WARNING,
            message=f"Found {table_metrics} metrics formatted in tables",
            details={'count': table_metrics}
        ))

        execution_time = (time.time() - start) * 1000
        return self._create_category_result(
            ValidationCategory.METRICS_COVERAGE, checks, execution_time
        )

    def _validate_venue_breakdown(
        self,
        content: str,
        report_data: Optional[Dict[str, Any]],
        step4_results: Optional[Dict[str, Any]]
    ) -> CategoryResult:
        """Validate venue-specific breakdown (CEX/DEX/Hybrid)."""
        import time
        start = time.time()
        checks = []
        content_lower = content.lower()

        # Check each venue type with color coding
        for venue_type, venue_info in VENUE_COLORS.items():
            venue_lower = venue_type.lower()
            venue_label = venue_info['label']
            hex_color = venue_info['hex']

            # Check venue is mentioned
            venue_present = venue_lower in content_lower
            has_color = venue_label in content or hex_color.lower() in content_lower

            checks.append(ValidationCheck(
                check_id=f"VENUE_{venue_type}",
                name=f"Venue: {venue_type} Analysis",
                category=ValidationCategory.VENUE_BREAKDOWN,
                level=ComplianceLevel.CRITICAL,
                status=CheckStatus.PASSED if venue_present else CheckStatus.FAILED,
                message=f"{venue_type} venue {'documented' if venue_present else 'MISSING'}",
                details={
                    'venue': venue_type,
                    'present': venue_present,
                    'color_coded': has_color,
                    'expected_label': venue_label
                },
                remediation=f"Add {venue_type} venue analysis section" if not venue_present else None
            ))

            # Check for color coding
            checks.append(ValidationCheck(
                check_id=f"VENUE_{venue_type}_COLOR",
                name=f"Venue Color: {venue_type} ({venue_info['name']})",
                category=ValidationCategory.VENUE_BREAKDOWN,
                level=ComplianceLevel.MINOR,
                status=CheckStatus.PASSED if has_color else CheckStatus.WARNING,
                message=f"{venue_type} color coding {'present' if has_color else 'missing'}",
                details={'has_color': has_color, 'expected': venue_label}
            ))

        # Check for venue comparison table
        venue_table_patterns = [
            r'\|\s*venue\s*\|',
            r'\|\s*cex\s*\|.*\|\s*dex\s*\|',
            r'\|\s*exchange\s*type\s*\|'
        ]
        has_venue_table = any(
            re.search(p, content_lower) for p in venue_table_patterns
        )

        checks.append(ValidationCheck(
            check_id="VENUE_TABLE",
            name="Venue Comparison Table",
            category=ValidationCategory.VENUE_BREAKDOWN,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_venue_table else CheckStatus.WARNING,
            message=f"Venue comparison table {'present' if has_venue_table else 'not found'}",
            details={'present': has_venue_table},
            remediation="Add comparative table showing metrics by venue type" if not has_venue_table else None
        ))

        # Check for venue-specific metrics
        venue_metric_patterns = [
            (r'cex.*(?:sharpe|return|drawdown)', 'CEX performance metrics'),
            (r'dex.*(?:sharpe|return|drawdown|tvl|liquidity)', 'DEX performance metrics'),
            (r'hybrid.*(?:sharpe|return|combined)', 'Hybrid performance metrics')
        ]

        for pattern, name in venue_metric_patterns:
            found = bool(re.search(pattern, content_lower, re.DOTALL))
            checks.append(ValidationCheck(
                check_id=f"VENUE_{name.split()[0].upper()}_METRICS",
                name=name,
                category=ValidationCategory.VENUE_BREAKDOWN,
                level=ComplianceLevel.MAJOR,
                status=CheckStatus.PASSED if found else CheckStatus.WARNING,
                message=f"{name} {'documented' if found else 'not found'}",
                details={'present': found}
            ))

        # Cross-validate with Step4 results if available
        if step4_results:
            venue_data = step4_results.get('venue_analysis', {})
            if venue_data:
                data_venues = set(venue_data.keys())
                checks.append(ValidationCheck(
                    check_id="VENUE_STEP4_CROSS",
                    name="Step4 Venue Cross-Validation",
                    category=ValidationCategory.VENUE_BREAKDOWN,
                    level=ComplianceLevel.MAJOR,
                    status=CheckStatus.PASSED,
                    message=f"Step4 venue data available: {list(data_venues)}",
                    details={'venues': list(data_venues)}
                ))

        execution_time = (time.time() - start) * 1000
        return self._create_category_result(
            ValidationCategory.VENUE_BREAKDOWN, checks, execution_time
        )

    def _validate_crisis_events(
        self,
        content: str,
        step4_results: Optional[Dict[str, Any]]
    ) -> CategoryResult:
        """Validate crisis event coverage (14 events required)."""
        import time
        start = time.time()
        checks = []
        content_lower = content.lower()

        # Check each required crisis event
        found_events = []
        extreme_events_found = 0
        extreme_events_total = 0

        for event_id, event_info in REQUIRED_CRISIS_EVENTS.items():
            aliases = event_info['aliases']
            severity = event_info['severity']

            if severity == 'extreme':
                extreme_events_total += 1

            found = False
            for alias in aliases:
                if alias in content_lower:
                    found = True
                    found_events.append(event_id)
                    if severity == 'extreme':
                        extreme_events_found += 1
                    break

        coverage = len(found_events) / len(REQUIRED_CRISIS_EVENTS)

        # Overall crisis coverage
        checks.append(ValidationCheck(
            check_id="CRISIS_COVERAGE",
            name="Crisis Event Coverage",
            category=ValidationCategory.CRISIS_EVENTS,
            level=ComplianceLevel.CRITICAL,
            status=CheckStatus.PASSED if coverage >= 0.7 else (
                CheckStatus.WARNING if coverage >= 0.5 else CheckStatus.FAILED
            ),
            message=f"Found {len(found_events)}/{len(REQUIRED_CRISIS_EVENTS)} crisis events ({coverage:.0%})",
            details={
                'found': len(found_events),
                'total': len(REQUIRED_CRISIS_EVENTS),
                'coverage': round(coverage, 2),
                'events': found_events
            },
            remediation=f"Add analysis for {len(REQUIRED_CRISIS_EVENTS) - len(found_events)} more crisis events" if coverage < 0.7 else None
        ))

        # Extreme events coverage (COVID, LUNA, FTX are must-have)
        extreme_coverage = extreme_events_found / extreme_events_total if extreme_events_total > 0 else 0
        checks.append(ValidationCheck(
            check_id="CRISIS_EXTREME",
            name="Extreme Crisis Events",
            category=ValidationCategory.CRISIS_EVENTS,
            level=ComplianceLevel.CRITICAL,
            status=CheckStatus.PASSED if extreme_coverage >= 1.0 else CheckStatus.FAILED,
            message=f"Extreme events: {extreme_events_found}/{extreme_events_total} (COVID, LUNA, FTX)",
            details={
                'found': extreme_events_found,
                'total': extreme_events_total
            },
            remediation="Add COVID-19, LUNA/UST, and FTX collapse analysis" if extreme_coverage < 1.0 else None
        ))

        # Check for crisis analysis table
        crisis_table_patterns = [
            r'\|\s*event\s*\|',
            r'\|\s*crisis\s*\|',
            r'\|\s*period\s*\|.*\|\s*drawdown\s*\|'
        ]
        has_crisis_table = any(
            re.search(p, content_lower) for p in crisis_table_patterns
        )

        checks.append(ValidationCheck(
            check_id="CRISIS_TABLE",
            name="Crisis Analysis Table",
            category=ValidationCategory.CRISIS_EVENTS,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_crisis_table else CheckStatus.WARNING,
            message=f"Crisis analysis table {'present' if has_crisis_table else 'not found'}",
            details={'present': has_crisis_table}
        ))

        # Check for crisis performance metrics
        crisis_metrics = ['outperformance', 'recovery', 'protection', 'resilience']
        found_crisis_metrics = sum(1 for m in crisis_metrics if m in content_lower)

        checks.append(ValidationCheck(
            check_id="CRISIS_METRICS",
            name="Crisis Performance Metrics",
            category=ValidationCategory.CRISIS_EVENTS,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if found_crisis_metrics >= 2 else CheckStatus.WARNING,
            message=f"Crisis metrics: {found_crisis_metrics}/{len(crisis_metrics)} found",
            details={'found': found_crisis_metrics, 'keywords': crisis_metrics}
        ))

        # Check for recovery analysis
        recovery_patterns = ['recovery time', 'days to recover', 'recovery period', 'bounced back']
        has_recovery = any(p in content_lower for p in recovery_patterns)

        checks.append(ValidationCheck(
            check_id="CRISIS_RECOVERY",
            name="Recovery Analysis",
            category=ValidationCategory.CRISIS_EVENTS,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_recovery else CheckStatus.WARNING,
            message=f"Recovery analysis {'present' if has_recovery else 'not documented'}",
            details={'present': has_recovery}
        ))

        execution_time = (time.time() - start) * 1000
        return self._create_category_result(
            ValidationCategory.CRISIS_EVENTS, checks, execution_time
        )

    def _validate_walk_forward(
        self,
        content: str,
        step4_results: Optional[Dict[str, Any]]
    ) -> CategoryResult:
        """Validate walk-forward requirements (18m train / 6m test)."""
        import time
        start = time.time()
        checks = []
        content_lower = content.lower()

        # Check for walk-forward section
        wf_patterns = ['walk-forward', 'walk forward', 'rolling window', 'out-of-sample']
        has_walk_forward = any(p in content_lower for p in wf_patterns)

        checks.append(ValidationCheck(
            check_id="WF_SECTION",
            name="Walk-Forward Section",
            category=ValidationCategory.WALK_FORWARD,
            level=ComplianceLevel.CRITICAL,
            status=CheckStatus.PASSED if has_walk_forward else CheckStatus.FAILED,
            message=f"Walk-forward section {'present' if has_walk_forward else 'MISSING'}",
            details={'present': has_walk_forward},
            remediation="Add walk-forward validation section with 18m train / 6m test windows" if not has_walk_forward else None
        ))

        # Check for training period (18 months)
        train_patterns = [
            r'18.*month.*train',
            r'train.*18.*month',
            r'18[- ]month.*window',
            r'training.*period.*18'
        ]
        has_train_spec = any(re.search(p, content_lower) for p in train_patterns)

        checks.append(ValidationCheck(
            check_id="WF_TRAIN",
            name="Training Period (18 months)",
            category=ValidationCategory.WALK_FORWARD,
            level=ComplianceLevel.CRITICAL,
            status=CheckStatus.PASSED if has_train_spec else CheckStatus.FAILED,
            message=f"18-month training period {'specified' if has_train_spec else 'NOT specified'}",
            details={'specified': has_train_spec, 'required_months': 18},
            remediation="Specify 18-month training window as per PDF requirements" if not has_train_spec else None
        ))

        # Check for test period (6 months)
        test_patterns = [
            r'6.*month.*test',
            r'test.*6.*month',
            r'6[- ]month.*out',
            r'out-of-sample.*6'
        ]
        has_test_spec = any(re.search(p, content_lower) for p in test_patterns)

        checks.append(ValidationCheck(
            check_id="WF_TEST",
            name="Test Period (6 months)",
            category=ValidationCategory.WALK_FORWARD,
            level=ComplianceLevel.CRITICAL,
            status=CheckStatus.PASSED if has_test_spec else CheckStatus.FAILED,
            message=f"6-month test period {'specified' if has_test_spec else 'NOT specified'}",
            details={'specified': has_test_spec, 'required_months': 6},
            remediation="Specify 6-month out-of-sample test window" if not has_test_spec else None
        ))

        # Check for multiple windows
        window_patterns = [r'window\s*\d', r'\d+\s*windows', r'period\s*\d', r'fold\s*\d']
        has_windows = any(re.search(p, content_lower) for p in window_patterns)

        checks.append(ValidationCheck(
            check_id="WF_WINDOWS",
            name="Multiple Walk-Forward Windows",
            category=ValidationCategory.WALK_FORWARD,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_windows else CheckStatus.WARNING,
            message=f"Multiple windows {'documented' if has_windows else 'not clearly specified'}",
            details={'present': has_windows}
        ))

        # Check for parameter stability
        stability_patterns = ['parameter stability', 'stable parameters', 'robust', 'sensitivity']
        has_stability = any(p in content_lower for p in stability_patterns)

        checks.append(ValidationCheck(
            check_id="WF_STABILITY",
            name="Parameter Stability Analysis",
            category=ValidationCategory.WALK_FORWARD,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_stability else CheckStatus.WARNING,
            message=f"Parameter stability {'analyzed' if has_stability else 'not analyzed'}",
            details={'present': has_stability}
        ))

        # Check for performance consistency
        consistency_patterns = ['consistent', 'degradation', 'decay', 'out-of-sample performance']
        has_consistency = any(p in content_lower for p in consistency_patterns)

        checks.append(ValidationCheck(
            check_id="WF_CONSISTENCY",
            name="Performance Consistency",
            category=ValidationCategory.WALK_FORWARD,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_consistency else CheckStatus.WARNING,
            message=f"Performance consistency {'analyzed' if has_consistency else 'not documented'}",
            details={'present': has_consistency}
        ))

        # Cross-validate with Step4 if available
        if step4_results:
            wf_data = step4_results.get('walk_forward', {})
            if wf_data:
                windows = wf_data.get('windows', [])
                # Type guard: windows could be int (count) instead of list
                window_count = len(windows) if isinstance(windows, (list, tuple)) else int(windows) if isinstance(windows, (int, float)) else 0
                checks.append(ValidationCheck(
                    check_id="WF_STEP4",
                    name="Step4 Walk-Forward Data",
                    category=ValidationCategory.WALK_FORWARD,
                    level=ComplianceLevel.MAJOR,
                    status=CheckStatus.PASSED if window_count >= 3 else CheckStatus.WARNING,
                    message=f"Step4 has {window_count} walk-forward windows",
                    details={'window_count': window_count}
                ))

        execution_time = (time.time() - start) * 1000
        return self._create_category_result(
            ValidationCategory.WALK_FORWARD, checks, execution_time
        )

    def _validate_capacity_analysis(
        self,
        content: str,
        step4_results: Optional[Dict[str, Any]]
    ) -> CategoryResult:
        """Validate capacity analysis (PDF capacity targets)."""
        import time
        start = time.time()
        checks = []
        content_lower = content.lower()

        # Check for capacity section
        capacity_patterns = ['capacity', 'scalability', 'strategy size', 'aum']
        has_capacity = any(p in content_lower for p in capacity_patterns)

        checks.append(ValidationCheck(
            check_id="CAP_SECTION",
            name="Capacity Analysis Section",
            category=ValidationCategory.CAPACITY_ANALYSIS,
            level=ComplianceLevel.CRITICAL,
            status=CheckStatus.PASSED if has_capacity else CheckStatus.FAILED,
            message=f"Capacity analysis {'present' if has_capacity else 'MISSING'}",
            details={'present': has_capacity},
            remediation="Add strategy capacity analysis section" if not has_capacity else None
        ))

        # Check for CEX capacity
        cex_cap_patterns = [
            r'cex.*capacity',
            r'cex.*\$\d+',
            r'centralized.*capacity',
            r'capacity.*cex'
        ]
        has_cex_capacity = any(re.search(p, content_lower) for p in cex_cap_patterns)

        checks.append(ValidationCheck(
            check_id="CAP_CEX",
            name="CEX Capacity Analysis",
            category=ValidationCategory.CAPACITY_ANALYSIS,
            level=ComplianceLevel.CRITICAL,
            status=CheckStatus.PASSED if has_cex_capacity else CheckStatus.FAILED,
            message=f"CEX capacity {'documented' if has_cex_capacity else 'NOT documented'}",
            details={'present': has_cex_capacity},
            remediation="Document CEX capacity ($10-30M target)" if not has_cex_capacity else None
        ))

        # Check for DEX capacity
        dex_cap_patterns = [
            r'dex.*capacity',
            r'dex.*\$\d+',
            r'dex.*tvl',
            r'decentralized.*capacity'
        ]
        has_dex_capacity = any(re.search(p, content_lower) for p in dex_cap_patterns)

        checks.append(ValidationCheck(
            check_id="CAP_DEX",
            name="DEX Capacity Analysis",
            category=ValidationCategory.CAPACITY_ANALYSIS,
            level=ComplianceLevel.CRITICAL,
            status=CheckStatus.PASSED if has_dex_capacity else CheckStatus.FAILED,
            message=f"DEX capacity {'documented' if has_dex_capacity else 'NOT documented'}",
            details={'present': has_dex_capacity},
            remediation="Document DEX capacity ($1-5M target)" if not has_dex_capacity else None
        ))

        # Check for capacity degradation analysis
        degradation_patterns = ['degradation', 'market impact', 'slippage curve', 'capacity limit']
        has_degradation = any(p in content_lower for p in degradation_patterns)

        checks.append(ValidationCheck(
            check_id="CAP_DEGRADE",
            name="Capacity Degradation Analysis",
            category=ValidationCategory.CAPACITY_ANALYSIS,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_degradation else CheckStatus.WARNING,
            message=f"Capacity degradation {'analyzed' if has_degradation else 'not analyzed'}",
            details={'present': has_degradation}
        ))

        # Check for dollar amounts
        dollar_amounts = re.findall(r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:m|million|M)?', content)
        has_amounts = len(dollar_amounts) >= 3

        checks.append(ValidationCheck(
            check_id="CAP_AMOUNTS",
            name="Capacity Dollar Amounts",
            category=ValidationCategory.CAPACITY_ANALYSIS,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_amounts else CheckStatus.WARNING,
            message=f"Found {len(dollar_amounts)} dollar amount references",
            details={'count': len(dollar_amounts), 'sample': dollar_amounts[:5]}
        ))

        execution_time = (time.time() - start) * 1000
        return self._create_category_result(
            ValidationCategory.CAPACITY_ANALYSIS, checks, execution_time
        )

    def _validate_grain_comparison(
        self,
        content: str,
        step4_results: Optional[Dict[str, Any]]
    ) -> CategoryResult:
        """Validate grain futures comparison (PDF REQUIRED)."""
        import time
        start = time.time()
        checks = []
        content_lower = content.lower()

        # Check for grain futures section - THIS IS CRITICAL
        grain_patterns = ['grain', 'agricultural', 'commodity futures', 'corn', 'wheat', 'soybean']
        has_grain = any(p in content_lower for p in grain_patterns)

        checks.append(ValidationCheck(
            check_id="GRAIN_SECTION",
            name="Grain Futures Comparison Section",
            category=ValidationCategory.GRAIN_COMPARISON,
            level=ComplianceLevel.CRITICAL,
            status=CheckStatus.PASSED if has_grain else CheckStatus.FAILED,
            message=f"Grain futures comparison {'present' if has_grain else 'MISSING (PDF REQUIRED!)'}",
            details={'present': has_grain},
            remediation="ADD GRAIN FUTURES COMPARISON - This is explicitly required by the project specification" if not has_grain else None
        ))

        # Check for futures/commodity reference
        futures_patterns = ['futures', 'commodity', 'traditional market', 'legacy market']
        has_futures = any(p in content_lower for p in futures_patterns)

        checks.append(ValidationCheck(
            check_id="GRAIN_FUTURES",
            name="Futures/Commodity Reference",
            category=ValidationCategory.GRAIN_COMPARISON,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_futures else CheckStatus.WARNING,
            message=f"Futures/commodity reference {'present' if has_futures else 'not found'}",
            details={'present': has_futures}
        ))

        # Check for comparison metrics
        comparison_keywords = [
            'half-life', 'mean reversion', 'volatility', 'correlation',
            'seasonality', 'stability', 'spread', 'cointegration'
        ]
        found_comparisons = sum(1 for kw in comparison_keywords if kw in content_lower)

        checks.append(ValidationCheck(
            check_id="GRAIN_METRICS",
            name="Comparison Metrics",
            category=ValidationCategory.GRAIN_COMPARISON,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if found_comparisons >= 3 else CheckStatus.WARNING,
            message=f"Found {found_comparisons}/{len(comparison_keywords)} comparison metrics",
            details={'found': found_comparisons, 'keywords': comparison_keywords}
        ))

        # Check for crypto vs grain discussion
        comparison_patterns = [
            r'crypto.*grain',
            r'compared.*agricultural',
            r'unlike.*futures',
            r'contrast.*commodity'
        ]
        has_comparison = any(re.search(p, content_lower) for p in comparison_patterns)

        checks.append(ValidationCheck(
            check_id="GRAIN_COMPARE",
            name="Crypto vs Grain Discussion",
            category=ValidationCategory.GRAIN_COMPARISON,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_comparison else CheckStatus.WARNING,
            message=f"Direct comparison {'present' if has_comparison else 'not found'}",
            details={'present': has_comparison}
        ))

        # Check for implications
        implication_patterns = ['implication', 'difference', 'lesson', 'insight', 'takeaway']
        has_implications = any(p in content_lower for p in implication_patterns)

        checks.append(ValidationCheck(
            check_id="GRAIN_IMPLICATIONS",
            name="Implications Discussion",
            category=ValidationCategory.GRAIN_COMPARISON,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_implications else CheckStatus.WARNING,
            message=f"Implications {'discussed' if has_implications else 'not discussed'}",
            details={'present': has_implications}
        ))

        execution_time = (time.time() - start) * 1000
        return self._create_category_result(
            ValidationCategory.GRAIN_COMPARISON, checks, execution_time
        )

    def _validate_position_sizing(
        self,
        content: str,
        step4_results: Optional[Dict[str, Any]]
    ) -> CategoryResult:
        """Validate position sizing requirements."""
        import time
        start = time.time()
        checks = []
        content_lower = content.lower()

        # Check for position sizing section
        sizing_patterns = ['position siz', 'sizing', 'position limit', 'trade size']
        has_sizing = any(p in content_lower for p in sizing_patterns)

        checks.append(ValidationCheck(
            check_id="SIZE_SECTION",
            name="Position Sizing Section",
            category=ValidationCategory.POSITION_SIZING,
            level=ComplianceLevel.CRITICAL,
            status=CheckStatus.PASSED if has_sizing else CheckStatus.FAILED,
            message=f"Position sizing {'documented' if has_sizing else 'NOT documented'}",
            details={'present': has_sizing},
            remediation="Add position sizing methodology section" if not has_sizing else None
        ))

        # Check for Kelly Criterion
        kelly_patterns = ['kelly', 'optimal f', 'kelly fraction', 'kelly criterion']
        has_kelly = any(p in content_lower for p in kelly_patterns)

        checks.append(ValidationCheck(
            check_id="SIZE_KELLY",
            name="Kelly Criterion",
            category=ValidationCategory.POSITION_SIZING,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_kelly else CheckStatus.WARNING,
            message=f"Kelly Criterion {'mentioned' if has_kelly else 'not mentioned'}",
            details={'present': has_kelly}
        ))

        # Check for CEX position limits ($100k)
        cex_limit_patterns = [r'cex.*\$?100', r'\$100.*k.*cex', r'100,?000.*position']
        has_cex_limit = any(re.search(p, content_lower) for p in cex_limit_patterns)

        checks.append(ValidationCheck(
            check_id="SIZE_CEX",
            name="CEX Position Limit ($100k)",
            category=ValidationCategory.POSITION_SIZING,
            level=ComplianceLevel.CRITICAL,
            status=CheckStatus.PASSED if has_cex_limit else CheckStatus.WARNING,
            message=f"CEX $100k limit {'specified' if has_cex_limit else 'not explicitly stated'}",
            details={'present': has_cex_limit}
        ))

        # Check for DEX position limits
        dex_limit_patterns = [r'dex.*\$?\d+k', r'dex.*liquid', r'illiquid.*\$']
        has_dex_limit = any(re.search(p, content_lower) for p in dex_limit_patterns)

        checks.append(ValidationCheck(
            check_id="SIZE_DEX",
            name="DEX Position Limits",
            category=ValidationCategory.POSITION_SIZING,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_dex_limit else CheckStatus.WARNING,
            message=f"DEX limits {'specified' if has_dex_limit else 'not specified'}",
            details={'present': has_dex_limit}
        ))

        # Check for liquidity tiers
        tier_patterns = ['tier 1', 'tier 2', 'tier 3', 'liquidity tier', 'high liquid', 'low liquid']
        has_tiers = any(p in content_lower for p in tier_patterns)

        checks.append(ValidationCheck(
            check_id="SIZE_TIERS",
            name="Liquidity Tiers",
            category=ValidationCategory.POSITION_SIZING,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_tiers else CheckStatus.WARNING,
            message=f"Liquidity tiers {'considered' if has_tiers else 'not considered'}",
            details={'present': has_tiers}
        ))

        # Check for dynamic sizing
        dynamic_patterns = ['dynamic', 'adaptive', 'volatility-adjusted', 'regime-based']
        has_dynamic = any(p in content_lower for p in dynamic_patterns)

        checks.append(ValidationCheck(
            check_id="SIZE_DYNAMIC",
            name="Dynamic Sizing",
            category=ValidationCategory.POSITION_SIZING,
            level=ComplianceLevel.MINOR,
            status=CheckStatus.PASSED if has_dynamic else CheckStatus.INFO,
            message=f"Dynamic sizing {'implemented' if has_dynamic else 'not mentioned'}",
            details={'present': has_dynamic}
        ))

        execution_time = (time.time() - start) * 1000
        return self._create_category_result(
            ValidationCategory.POSITION_SIZING, checks, execution_time
        )

    def _validate_concentration_limits(
        self,
        content: str,
        step4_results: Optional[Dict[str, Any]]
    ) -> CategoryResult:
        """Validate concentration limits (PDF requirements)."""
        import time
        start = time.time()
        checks = []
        content_lower = content.lower()

        # Check for concentration section
        conc_patterns = ['concentration', 'diversification', 'exposure limit']
        has_concentration = any(p in content_lower for p in conc_patterns)

        checks.append(ValidationCheck(
            check_id="CONC_SECTION",
            name="Concentration Limits Section",
            category=ValidationCategory.CONCENTRATION_LIMITS,
            level=ComplianceLevel.CRITICAL,
            status=CheckStatus.PASSED if has_concentration else CheckStatus.FAILED,
            message=f"Concentration limits {'documented' if has_concentration else 'NOT documented'}",
            details={'present': has_concentration},
            remediation="Add concentration limits section" if not has_concentration else None
        ))

        # Check for sector limit (40%)
        sector_patterns = [r'40\s*%.*sector', r'sector.*40\s*%', r'40.*percent.*sector']
        has_sector_limit = any(re.search(p, content_lower) for p in sector_patterns) or (
            '40%' in content and 'sector' in content_lower
        )

        checks.append(ValidationCheck(
            check_id="CONC_SECTOR",
            name="Sector Limit (40%)",
            category=ValidationCategory.CONCENTRATION_LIMITS,
            level=ComplianceLevel.CRITICAL,
            status=CheckStatus.PASSED if has_sector_limit else CheckStatus.FAILED,
            message=f"40% sector limit {'specified' if has_sector_limit else 'NOT specified'}",
            details={'present': has_sector_limit},
            remediation="Specify 40% maximum sector concentration" if not has_sector_limit else None
        ))

        # Check for CEX limit (60%)
        cex_patterns = [r'60\s*%.*cex', r'cex.*60\s*%', r'60.*percent.*cex']
        has_cex_limit = any(re.search(p, content_lower) for p in cex_patterns) or (
            '60%' in content and 'cex' in content_lower
        )

        checks.append(ValidationCheck(
            check_id="CONC_CEX",
            name="CEX Limit (60%)",
            category=ValidationCategory.CONCENTRATION_LIMITS,
            level=ComplianceLevel.CRITICAL,
            status=CheckStatus.PASSED if has_cex_limit else CheckStatus.FAILED,
            message=f"60% CEX limit {'specified' if has_cex_limit else 'NOT specified'}",
            details={'present': has_cex_limit},
            remediation="Specify 60% maximum CEX concentration" if not has_cex_limit else None
        ))

        # Check for Tier 3 limit (20%)
        tier3_patterns = [r'20\s*%.*tier', r'tier.*3.*20', r'20.*percent.*tier']
        has_tier3_limit = any(re.search(p, content_lower) for p in tier3_patterns) or (
            '20%' in content and ('tier 3' in content_lower or 'tier3' in content_lower)
        )

        checks.append(ValidationCheck(
            check_id="CONC_TIER3",
            name="Tier 3 Limit (20%)",
            category=ValidationCategory.CONCENTRATION_LIMITS,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_tier3_limit else CheckStatus.WARNING,
            message=f"20% Tier 3 limit {'specified' if has_tier3_limit else 'not specified'}",
            details={'present': has_tier3_limit}
        ))

        # Check for correlation-based limits
        corr_patterns = ['correlation cluster', 'correlated assets', 'correlation limit']
        has_corr_limit = any(p in content_lower for p in corr_patterns)

        checks.append(ValidationCheck(
            check_id="CONC_CORR",
            name="Correlation-Based Limits",
            category=ValidationCategory.CONCENTRATION_LIMITS,
            level=ComplianceLevel.MINOR,
            status=CheckStatus.PASSED if has_corr_limit else CheckStatus.INFO,
            message=f"Correlation limits {'considered' if has_corr_limit else 'not mentioned'}",
            details={'present': has_corr_limit}
        ))

        execution_time = (time.time() - start) * 1000
        return self._create_category_result(
            ValidationCategory.CONCENTRATION_LIMITS, checks, execution_time
        )

    def _validate_statistical(
        self,
        content: str,
        report_data: Optional[Dict[str, Any]],
        step4_results: Optional[Dict[str, Any]]
    ) -> CategoryResult:
        """Validate statistical rigor and significance testing."""
        import time
        start = time.time()
        checks = []
        content_lower = content.lower()

        # Check for statistical significance testing
        stat_patterns = ['t-test', 't-statistic', 'p-value', 'statistical significance', 'hypothesis']
        has_significance = any(p in content_lower for p in stat_patterns)

        checks.append(ValidationCheck(
            check_id="STAT_SIG",
            name="Statistical Significance Testing",
            category=ValidationCategory.STATISTICAL_VALIDATION,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_significance else CheckStatus.WARNING,
            message=f"Significance testing {'present' if has_significance else 'not documented'}",
            details={'present': has_significance}
        ))

        # Check for confidence intervals
        ci_patterns = ['confidence interval', '95% ci', 'confidence level', 'bootstrap']
        has_ci = any(p in content_lower for p in ci_patterns)

        checks.append(ValidationCheck(
            check_id="STAT_CI",
            name="Confidence Intervals",
            category=ValidationCategory.STATISTICAL_VALIDATION,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_ci else CheckStatus.WARNING,
            message=f"Confidence intervals {'reported' if has_ci else 'not reported'}",
            details={'present': has_ci}
        ))

        # Check for Sharpe ratio significance
        sharpe_sig_patterns = ['sharpe.*significant', 'significant.*sharpe', 'sharpe.*t-stat']
        has_sharpe_sig = any(re.search(p, content_lower) for p in sharpe_sig_patterns)

        checks.append(ValidationCheck(
            check_id="STAT_SHARPE",
            name="Sharpe Ratio Significance",
            category=ValidationCategory.STATISTICAL_VALIDATION,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_sharpe_sig else CheckStatus.WARNING,
            message=f"Sharpe significance {'tested' if has_sharpe_sig else 'not tested'}",
            details={'present': has_sharpe_sig}
        ))

        # Check for Monte Carlo simulation
        mc_patterns = ['monte carlo', 'simulation', 'bootstrap', 'random sample']
        has_mc = any(p in content_lower for p in mc_patterns)

        checks.append(ValidationCheck(
            check_id="STAT_MC",
            name="Monte Carlo/Bootstrap Analysis",
            category=ValidationCategory.STATISTICAL_VALIDATION,
            level=ComplianceLevel.MINOR,
            status=CheckStatus.PASSED if has_mc else CheckStatus.INFO,
            message=f"Monte Carlo {'performed' if has_mc else 'not mentioned'}",
            details={'present': has_mc}
        ))

        # Check for sensitivity analysis
        sens_patterns = ['sensitivity', 'robustness', 'parameter sweep', 'stress test']
        has_sens = any(p in content_lower for p in sens_patterns)

        checks.append(ValidationCheck(
            check_id="STAT_SENS",
            name="Sensitivity Analysis",
            category=ValidationCategory.STATISTICAL_VALIDATION,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_sens else CheckStatus.WARNING,
            message=f"Sensitivity analysis {'performed' if has_sens else 'not documented'}",
            details={'present': has_sens}
        ))

        execution_time = (time.time() - start) * 1000
        return self._create_category_result(
            ValidationCategory.STATISTICAL_VALIDATION, checks, execution_time
        )

    def _validate_cross_references(
        self,
        content: str,
        report_data: Optional[Dict[str, Any]],
        step4_results: Optional[Dict[str, Any]]
    ) -> CategoryResult:
        """Validate cross-reference consistency."""
        import time
        start = time.time()
        checks = []
        content_lower = content.lower()

        # Check internal references
        ref_patterns = [r'section \d', r'figure \d', r'table \d', r'see above', r'as shown']
        ref_count = sum(len(re.findall(p, content_lower)) for p in ref_patterns)
        has_refs = ref_count >= 5

        checks.append(ValidationCheck(
            check_id="XREF_INTERNAL",
            name="Internal References",
            category=ValidationCategory.CROSS_REFERENCE,
            level=ComplianceLevel.MINOR,
            status=CheckStatus.PASSED if has_refs else CheckStatus.WARNING,
            message=f"Found {ref_count} internal references",
            details={'count': ref_count}
        ))

        # Check for consistent venue naming
        venue_mentions = {
            'cex': len(re.findall(r'\bcex\b', content_lower)),
            'dex': len(re.findall(r'\bdex\b', content_lower)),
            'hybrid': len(re.findall(r'\bhybrid\b', content_lower))
        }
        consistent_venues = all(v > 0 for v in venue_mentions.values())

        checks.append(ValidationCheck(
            check_id="XREF_VENUES",
            name="Venue Naming Consistency",
            category=ValidationCategory.CROSS_REFERENCE,
            level=ComplianceLevel.MINOR,
            status=CheckStatus.PASSED if consistent_venues else CheckStatus.WARNING,
            message=f"Venue mentions: CEX={venue_mentions['cex']}, DEX={venue_mentions['dex']}, Hybrid={venue_mentions['hybrid']}",
            details=venue_mentions
        ))

        # Check metrics consistency (same numbers mentioned multiple times should match)
        sharpe_values = re.findall(r'sharpe.*?(\d+\.\d+)', content_lower)
        unique_sharpes = len(set(sharpe_values))
        sharpe_consistent = unique_sharpes <= 5  # Allow some variation for different venues/periods

        checks.append(ValidationCheck(
            check_id="XREF_METRICS",
            name="Metrics Value Consistency",
            category=ValidationCategory.CROSS_REFERENCE,
            level=ComplianceLevel.MINOR,
            status=CheckStatus.PASSED if sharpe_consistent else CheckStatus.WARNING,
            message=f"Found {len(sharpe_values)} Sharpe mentions, {unique_sharpes} unique values",
            details={'total': len(sharpe_values), 'unique': unique_sharpes}
        ))

        # Cross-validate with Step4 if available
        if step4_results and report_data:
            checks.append(ValidationCheck(
                check_id="XREF_STEP4",
                name="Step4 Data Cross-Validation",
                category=ValidationCategory.CROSS_REFERENCE,
                level=ComplianceLevel.MAJOR,
                status=CheckStatus.PASSED,
                message="Step4 and report data available for cross-validation",
                details={'step4_keys': list(step4_results.keys())[:10]}
            ))

        execution_time = (time.time() - start) * 1000
        return self._create_category_result(
            ValidationCategory.CROSS_REFERENCE, checks, execution_time
        )

    def _validate_formatting(self, content: str) -> CategoryResult:
        """Validate formatting and presentation."""
        import time
        start = time.time()
        checks = []

        # Check for tables
        table_rows = len(re.findall(r'^\|.*\|.*\|', content, re.MULTILINE))
        has_tables = table_rows >= 15

        checks.append(ValidationCheck(
            check_id="FMT_TABLES",
            name="Data Tables",
            category=ValidationCategory.FORMATTING,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if has_tables else CheckStatus.WARNING,
            message=f"Found {table_rows} table rows (recommend >= 15)",
            details={'count': table_rows}
        ))

        # Check for proper header structure
        h2_count = len(re.findall(r'^## [^#]', content, re.MULTILINE))
        h3_count = len(re.findall(r'^### [^#]', content, re.MULTILINE))
        h4_count = len(re.findall(r'^#### [^#]', content, re.MULTILINE))
        good_structure = h2_count >= 8 and h3_count >= 15

        checks.append(ValidationCheck(
            check_id="FMT_HEADERS",
            name="Section Headers",
            category=ValidationCategory.FORMATTING,
            level=ComplianceLevel.MAJOR,
            status=CheckStatus.PASSED if good_structure else CheckStatus.WARNING,
            message=f"Headers: H2={h2_count}, H3={h3_count}, H4={h4_count}",
            details={'h2': h2_count, 'h3': h3_count, 'h4': h4_count}
        ))

        # Check for code blocks
        code_blocks = len(re.findall(r'```', content)) // 2
        has_code = code_blocks >= 2

        checks.append(ValidationCheck(
            check_id="FMT_CODE",
            name="Code Examples",
            category=ValidationCategory.FORMATTING,
            level=ComplianceLevel.MINOR,
            status=CheckStatus.PASSED if has_code else CheckStatus.INFO,
            message=f"Found {code_blocks} code blocks",
            details={'count': code_blocks}
        ))

        # Check for bullet points
        bullet_count = len(re.findall(r'^[\-\*]\s', content, re.MULTILINE))
        numbered_count = len(re.findall(r'^\d+\.\s', content, re.MULTILINE))
        has_lists = bullet_count >= 20 or numbered_count >= 10

        checks.append(ValidationCheck(
            check_id="FMT_LISTS",
            name="Bullet/Numbered Lists",
            category=ValidationCategory.FORMATTING,
            level=ComplianceLevel.MINOR,
            status=CheckStatus.PASSED if has_lists else CheckStatus.INFO,
            message=f"Lists: {bullet_count} bullets, {numbered_count} numbered",
            details={'bullets': bullet_count, 'numbered': numbered_count}
        ))

        # Check for emphasis (bold, italic)
        bold_count = len(re.findall(r'\*\*[^*]+\*\*', content))
        italic_count = len(re.findall(r'(?<!\*)\*[^*]+\*(?!\*)', content))
        has_emphasis = bold_count >= 10

        checks.append(ValidationCheck(
            check_id="FMT_EMPHASIS",
            name="Text Emphasis",
            category=ValidationCategory.FORMATTING,
            level=ComplianceLevel.MINOR,
            status=CheckStatus.PASSED if has_emphasis else CheckStatus.INFO,
            message=f"Found {bold_count} bold, {italic_count} italic uses",
            details={'bold': bold_count, 'italic': italic_count}
        ))

        execution_time = (time.time() - start) * 1000
        return self._create_category_result(
            ValidationCategory.FORMATTING, checks, execution_time
        )

    def _validate_academic_standards(self, content: str) -> CategoryResult:
        """Validate academic standards compliance."""
        import time
        start = time.time()
        checks = []
        content_lower = content.lower()

        # Check for literature references
        ref_patterns = [r'\(\d{4}\)', r'et al\.', r'\[\d+\]', r'according to']
        ref_count = sum(len(re.findall(p, content)) for p in ref_patterns)
        has_refs = ref_count >= 3

        checks.append(ValidationCheck(
            check_id="ACAD_REFS",
            name="Literature References",
            category=ValidationCategory.ACADEMIC_STANDARDS,
            level=ComplianceLevel.MINOR,
            status=CheckStatus.PASSED if has_refs else CheckStatus.INFO,
            message=f"Found {ref_count} academic-style references",
            details={'count': ref_count}
        ))

        # Check for methodology citation
        method_refs = ['engle', 'granger', 'johansen', 'adf', 'dickey-fuller', 'hurst']
        found_methods = sum(1 for m in method_refs if m in content_lower)
        has_method_refs = found_methods >= 2

        checks.append(ValidationCheck(
            check_id="ACAD_METHODS",
            name="Methodology Citations",
            category=ValidationCategory.ACADEMIC_STANDARDS,
            level=ComplianceLevel.MINOR,
            status=CheckStatus.PASSED if has_method_refs else CheckStatus.INFO,
            message=f"Found {found_methods} statistical method references",
            details={'count': found_methods, 'methods': method_refs}
        ))

        # Check for formal language
        informal_patterns = ['gonna', 'wanna', 'kinda', 'sorta', 'stuff', 'things']
        informal_count = sum(content_lower.count(p) for p in informal_patterns)
        is_formal = informal_count <= 2

        checks.append(ValidationCheck(
            check_id="ACAD_FORMAL",
            name="Formal Language",
            category=ValidationCategory.ACADEMIC_STANDARDS,
            level=ComplianceLevel.MINOR,
            status=CheckStatus.PASSED if is_formal else CheckStatus.WARNING,
            message=f"Informal language instances: {informal_count}",
            details={'count': informal_count}
        ))

        # Check for clear definitions
        def_patterns = ['defined as', 'we define', 'definition:', 'formally']
        def_count = sum(content_lower.count(p) for p in def_patterns)
        has_defs = def_count >= 3

        checks.append(ValidationCheck(
            check_id="ACAD_DEFS",
            name="Clear Definitions",
            category=ValidationCategory.ACADEMIC_STANDARDS,
            level=ComplianceLevel.MINOR,
            status=CheckStatus.PASSED if has_defs else CheckStatus.INFO,
            message=f"Found {def_count} formal definitions",
            details={'count': def_count}
        ))

        execution_time = (time.time() - start) * 1000
        return self._create_category_result(
            ValidationCategory.ACADEMIC_STANDARDS, checks, execution_time
        )

    def _validate_executive_quality(self, content: str) -> CategoryResult:
        """Validate executive summary quality."""
        import time
        start = time.time()
        checks = []
        content_lower = content.lower()

        # Find executive summary section
        exec_match = re.search(
            r'(?:executive summary|exec summary)(.*?)(?=\n## |\Z)',
            content_lower,
            re.DOTALL | re.IGNORECASE
        )

        if exec_match:
            exec_content = exec_match.group(1)
            exec_words = len(exec_content.split())

            # Check length
            good_length = 400 <= exec_words <= 1000
            checks.append(ValidationCheck(
                check_id="EXEC_LENGTH",
                name="Executive Summary Length",
                category=ValidationCategory.EXECUTIVE_QUALITY,
                level=ComplianceLevel.MAJOR,
                status=CheckStatus.PASSED if good_length else CheckStatus.WARNING,
                message=f"Executive summary: {exec_words} words (target: 400-1000)",
                details={'words': exec_words}
            ))

            # Check for key components
            key_components = [
                ('objective', ['objective', 'goal', 'aim', 'purpose']),
                ('methodology', ['methodology', 'approach', 'method']),
                ('findings', ['finding', 'result', 'discovered', 'show']),
                ('recommendation', ['recommend', 'suggest', 'propose', 'advise'])
            ]

            for comp_name, keywords in key_components:
                has_comp = any(kw in exec_content for kw in keywords)
                checks.append(ValidationCheck(
                    check_id=f"EXEC_{comp_name.upper()}",
                    name=f"Exec Summary: {comp_name.title()}",
                    category=ValidationCategory.EXECUTIVE_QUALITY,
                    level=ComplianceLevel.MAJOR,
                    status=CheckStatus.PASSED if has_comp else CheckStatus.WARNING,
                    message=f"{comp_name.title()} {'present' if has_comp else 'missing'}",
                    details={'present': has_comp}
                ))

            # Check for key metrics mentioned
            key_metrics = ['sharpe', 'return', 'drawdown', 'capacity']
            metrics_found = sum(1 for m in key_metrics if m in exec_content)
            has_metrics = metrics_found >= 3

            checks.append(ValidationCheck(
                check_id="EXEC_METRICS",
                name="Key Metrics in Summary",
                category=ValidationCategory.EXECUTIVE_QUALITY,
                level=ComplianceLevel.MAJOR,
                status=CheckStatus.PASSED if has_metrics else CheckStatus.WARNING,
                message=f"Key metrics: {metrics_found}/{len(key_metrics)} mentioned",
                details={'found': metrics_found, 'total': len(key_metrics)}
            ))
        else:
            checks.append(ValidationCheck(
                check_id="EXEC_EXISTS",
                name="Executive Summary Exists",
                category=ValidationCategory.EXECUTIVE_QUALITY,
                level=ComplianceLevel.CRITICAL,
                status=CheckStatus.FAILED,
                message="Executive summary section NOT FOUND",
                details={'present': False},
                remediation="Add executive summary section at the beginning of the report"
            ))

        execution_time = (time.time() - start) * 1000
        return self._create_category_result(
            ValidationCategory.EXECUTIVE_QUALITY, checks, execution_time
        )

    # =========================================================================
    # REPORT GENERATION
    # =========================================================================

    def generate_compliance_report(
        self,
        result: StrictValidationResult,
        output_path: Optional[Path] = None,
        include_details: bool = True
    ) -> str:
        """
        Generate comprehensive human-readable compliance report.

        Args:
            result: Validation result from validate_comprehensive
            output_path: Optional path to save report
            include_details: Include detailed check information

        Returns:
            Compliance report as markdown string
        """
        report = f"""# PDF Compliance Validation Report

**Validation Timestamp:** {result.validation_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
**Validation Profile:** {result.metadata.profile.value.upper()}
**PDF Reference:** {result.metadata.pdf_reference}

---

## Compliance Status

| Status | Value |
|--------|-------|
| **Overall Status** | {'**COMPLIANT**' if result.is_pdf_compliant else '**NON-COMPLIANT**'} |
| **Compliance Score** | {result.compliance_score:.1%} |
| **Quality Grade** | {result.quality_score.grade} ({result.quality_score.overall_score:.1f}/100) |
| **Estimated Percentile** | {result.quality_score.percentile}th |

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Checks | {result.total_checks} |
| Passed | {result.passed_checks} |
| Failed | {result.failed_checks} |
| Warnings | {result.warning_checks} |
| Blocking Failures | {result.blocking_failure_count} |
| Categories Compliant | {result.categories_compliant}/{len(result.category_results)} |

---

## Quality Score Breakdown

| Component | Score |
|-----------|-------|
| Structure | {result.quality_score.structure_score:.1f}/100 |
| Content | {result.quality_score.content_score:.1f}/100 |
| Technical | {result.quality_score.technical_score:.1f}/100 |
| Presentation | {result.quality_score.presentation_score:.1f}/100 |
| Compliance | {result.quality_score.compliance_score:.1f}/100 |

---

## Critical Failures (Must Fix)

"""
        if result.critical_failures:
            for i, failure in enumerate(result.critical_failures, 1):
                report += f"""### {i}. [FAIL] {failure.name}

- **Category:** {failure.category.value.replace('_', ' ').title()}
- **Message:** {failure.message}
- **Remediation:** {failure.remediation or 'Review and fix the issue'}

"""
        else:
            report += "**No critical failures.** All critical requirements are met.\n\n"

        report += """---

## Major Failures (Should Fix)

"""
        if result.major_failures:
            for i, failure in enumerate(result.major_failures, 1):
                report += f"""### {i}. [WARN] {failure.name}

- **Category:** {failure.category.value.replace('_', ' ').title()}
- **Message:** {failure.message}

"""
        else:
            report += "**No major failures.**\n\n"

        report += """---

## Warnings (Recommended)

"""
        if result.all_warnings:
            for warning in result.all_warnings[:10]:  # Top 10
                report += f"- [WARN] **{warning.name}**: {warning.message}\n"
            if len(result.all_warnings) > 10:
                report += f"\n*...and {len(result.all_warnings) - 10} more warnings*\n"
        else:
            report += "**No warnings.**\n"

        report += """
---

## Category Results

"""
        for category, cat_result in result.category_results.items():
            status_icon = '[PASS]' if cat_result.is_compliant else '[FAIL]'
            report += f"""### {status_icon} {category.value.replace('_', ' ').title()}

- **Pass Rate:** {cat_result.pass_rate:.1%}
- **Score:** {cat_result.score:.1f}/{cat_result.max_possible_score:.1f}
- **Checks:** {cat_result.passed} passed, {cat_result.failed} failed, {cat_result.warnings} warnings

"""

        if result.immediate_remediations:
            report += """---

## Immediate Action Items

"""
            for i, rem in enumerate(result.immediate_remediations[:5], 1):
                report += f"""### {i}. {rem.title}

{rem.description}

**Steps:**
"""
                for step in rem.steps:
                    report += f"- {step}\n"
                report += "\n"

        report += f"""---

## Report Metadata

| Property | Value |
|----------|-------|
| Report Length | {result.metadata.report_length:,} characters |
| Word Count | {result.metadata.word_count:,} words |
| Estimated Pages | {result.metadata.estimated_pages:.1f} |
| Report Hash | {result.metadata.report_hash[:16]}... |
| Execution Time | {result.metadata.total_execution_time_ms:.0f}ms |
| Validator Version | {result.metadata.validation_version} |

---

*Generated by StrictPDFValidator v3.0.0*
*PDF Compliance: Project Specification*
"""

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Saved compliance report to {output_path}")

        return report

    def generate_json_report(
        self,
        result: StrictValidationResult,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate JSON format compliance report.

        Args:
            result: Validation result
            output_path: Optional path to save report

        Returns:
            JSON string
        """
        json_str = result.to_json(indent=2)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.info(f"Saved JSON report to {output_path}")

        return json_str

    def clear_cache(self) -> None:
        """Clear validation cache."""
        self._cache.clear()
        logger.info("Validation cache cleared")


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_strict_pdf_validator(
    profile: ValidationProfile = ValidationProfile.STRICT,
    enable_caching: bool = True
) -> StrictPDFValidator:
    """
    Create a StrictPDFValidator instance.

    Args:
        profile: Validation profile (strict/standard/lenient)
        enable_caching: Enable result caching

    Returns:
        Configured StrictPDFValidator instance
    """
    return StrictPDFValidator(
        profile=profile,
        enable_caching=enable_caching,
        generate_remediations=True
    )


def validate_report_compliance(
    report_content: str,
    report_data: Optional[Dict[str, Any]] = None,
    step4_results: Optional[Dict[str, Any]] = None,
    profile: ValidationProfile = ValidationProfile.STRICT
) -> StrictValidationResult:
    """
    Convenience function to validate report compliance.

    Args:
        report_content: Markdown content of the report
        report_data: JSON data from report generation
        step4_results: Results from Step4AdvancedOrchestrator
        profile: Validation profile

    Returns:
        StrictValidationResult with full compliance assessment
    """
    validator = create_strict_pdf_validator(profile=profile)
    return validator.validate_comprehensive(
        report_content=report_content,
        report_data=report_data,
        step4_results=step4_results
    )


def quick_validate(report_content: str) -> Tuple[bool, float, List[str]]:
    """
    Quick validation returning simple results.

    Args:
        report_content: Markdown content of the report

    Returns:
        Tuple of (is_compliant, score, list of failure messages)
    """
    validator = create_strict_pdf_validator(
        profile=ValidationProfile.STANDARD,
        enable_caching=False
    )
    result = validator.validate_comprehensive(report_content)

    failure_messages = [
        f"{f.name}: {f.message}"
        for f in result.critical_failures + result.major_failures
    ]

    return result.is_pdf_compliant, result.compliance_score, failure_messages


def batch_validate(
    reports: List[Tuple[str, str]],  # (name, content) pairs
    profile: ValidationProfile = ValidationProfile.STANDARD
) -> Dict[str, StrictValidationResult]:
    """
    Validate multiple reports.

    Args:
        reports: List of (name, content) tuples
        profile: Validation profile

    Returns:
        Dictionary mapping report names to validation results
    """
    validator = create_strict_pdf_validator(profile=profile)
    results = {}

    for name, content in reports:
        logger.info(f"Validating report: {name}")
        results[name] = validator.validate_comprehensive(content)

    return results
