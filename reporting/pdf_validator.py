"""
PDF Compliance Validator
========================

Validates that generated reports meet PDF project requirements including:
- Required sections present
- Minimum/maximum page count (30-40 pages)
- Content completeness
- Format compliance

Note: This validates markdown reports that can be converted to PDF.
Actual PDF generation would use tools like pandoc, weasyprint, or reportlab.

Author: Tamer Atesyakar
Version: 2.0.0
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ComplianceCheck(Enum):
    """Types of compliance checks."""
    SECTIONS = "required_sections"
    PAGE_COUNT = "page_count"
    CONTENT_LENGTH = "content_length"
    FORMATTING = "formatting"
    COMPLETENESS = "completeness"


@dataclass
class ValidationResult:
    """Result of PDF compliance validation."""
    is_compliant: bool
    total_checks: int
    passed_checks: int
    failed_checks: List[str]
    warnings: List[str]
    metadata: Dict

    @property
    def compliance_score(self) -> float:
        """Calculate compliance score (0-1)."""
        return self.passed_checks / self.total_checks if self.total_checks > 0 else 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'is_compliant': self.is_compliant,
            'compliance_score': self.compliance_score,
            'total_checks': self.total_checks,
            'passed_checks': self.passed_checks,
            'failed_checks': self.failed_checks,
            'warnings': self.warnings,
            'metadata': self.metadata
        }


class PDFValidator:
    """
    PDF compliance validator for project requirements.

    Validates that generated markdown reports meet the requirements
    for the final 30-40 page PDF deliverable.

    Required Sections (from project):
    1. Universe Construction
    2. Cointegration Analysis
    3. Baseline Strategy
    4. Enhancements (all three)
    5. Backtesting Methodology
    6. Performance Results
    7. Crisis Analysis
    8. Capacity Analysis
    9. Grain Futures Comparison
    10. Conclusions

    Usage:
        validator = PDFValidator()
        result = validator.validate(
            markdown_path=Path("comprehensive_report.md"),
            required_sections=[...],
            min_pages=30,
            max_pages=40
        )
    """

    # Estimate 500 words per page, 5 characters per word
    CHARS_PER_PAGE = 2500

    REQUIRED_SECTIONS_DEFAULT = [
        'executive summary',
        'universe construction',
        'cointegration',
        'baseline strategy',
        'enhancements',
        'backtesting',
        'crisis analysis',
        'capacity analysis',
        'conclusions'
    ]

    def __init__(self):
        """Initialize PDF validator."""
        logger.info("PDFValidator initialized")

    def validate(
        self,
        markdown_path: Path,
        required_sections: Optional[List[str]] = None,
        min_pages: int = 30,
        max_pages: int = 40
    ) -> ValidationResult:
        """
        Validate markdown report for PDF compliance.

        Args:
            markdown_path: Path to markdown report file
            required_sections: List of required section names
            min_pages: Minimum page count
            max_pages: Maximum page count

        Returns:
            ValidationResult with compliance assessment
        """
        if required_sections is None:
            required_sections = self.REQUIRED_SECTIONS_DEFAULT

        failed_checks = []
        warnings = []
        metadata = {}

        # Check 1: File exists
        if not markdown_path.exists():
            failed_checks.append(f"File not found: {markdown_path}")
            return ValidationResult(
                is_compliant=False,
                total_checks=5,
                passed_checks=0,
                failed_checks=failed_checks,
                warnings=warnings,
                metadata=metadata
            )

        # Read content
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()

        content_lower = content.lower()

        # Check 2: Required sections
        missing_sections = []
        for section in required_sections:
            if section.lower() not in content_lower:
                missing_sections.append(section)

        if missing_sections:
            failed_checks.append(f"Missing required sections: {', '.join(missing_sections)}")

        metadata['found_sections'] = len(required_sections) - len(missing_sections)
        metadata['missing_sections'] = missing_sections

        # Check 3: Page count (estimated)
        estimated_pages = len(content) / self.CHARS_PER_PAGE

        metadata['estimated_pages'] = round(estimated_pages, 1)
        metadata['min_pages'] = min_pages
        metadata['max_pages'] = max_pages

        if estimated_pages < min_pages:
            warnings.append(f"Report may be too short: ~{estimated_pages:.0f} pages (minimum {min_pages})")
        elif estimated_pages > max_pages:
            warnings.append(f"Report may be too long: ~{estimated_pages:.0f} pages (maximum {max_pages})")

        # Check 4: Content length
        min_chars = min_pages * self.CHARS_PER_PAGE
        if len(content) < min_chars:
            failed_checks.append(f"Content too short: {len(content)} chars (minimum {min_chars})")

        metadata['content_length'] = len(content)

        # Check 5: Formatting (basic markdown checks)
        has_headers = '##' in content or '#' in content
        has_lists = '-' in content or '*' in content
        has_code_blocks = '```' in content

        if not has_headers:
            warnings.append("No markdown headers found")
        if not has_lists:
            warnings.append("No markdown lists found")

        metadata['has_headers'] = has_headers
        metadata['has_lists'] = has_lists
        metadata['has_code_blocks'] = has_code_blocks

        # Calculate results
        total_checks = 5
        passed_checks = total_checks - len(failed_checks)

        is_compliant = (
            len(failed_checks) == 0 and
            estimated_pages >= min_pages and
            estimated_pages <= max_pages
        )

        logger.info(f"Validation complete: {passed_checks}/{total_checks} checks passed")

        return ValidationResult(
            is_compliant=is_compliant,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warnings=warnings,
            metadata=metadata
        )

    def generate_compliance_report(
        self,
        validation_result: ValidationResult,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate human-readable compliance report.

        Args:
            validation_result: Validation result to report on
            output_path: Optional path to save report

        Returns:
            Compliance report as string
        """
        report = f"""# PDF Compliance Validation Report

**Compliance Status:** {'PASS' if validation_result.is_compliant else 'FAIL'}
**Compliance Score:** {validation_result.compliance_score:.1%}

## Summary

- Total Checks: {validation_result.total_checks}
- Passed: {validation_result.passed_checks}
- Failed: {len(validation_result.failed_checks)}
- Warnings: {len(validation_result.warnings)}

## Metadata

- Estimated Pages: {validation_result.metadata.get('estimated_pages', 0)}
- Content Length: {validation_result.metadata.get('content_length', 0):,} characters
- Sections Found: {validation_result.metadata.get('found_sections', 0)}

## Failed Checks

"""

        if validation_result.failed_checks:
            for check in validation_result.failed_checks:
                report += f"- [FAIL] {check}\n"
        else:
            report += "- None - All checks passed!\n"

        report += "\n## Warnings\n\n"

        if validation_result.warnings:
            for warning in validation_result.warnings:
                report += f"- WARNING: {warning}\n"
        else:
            report += "- None\n"

        if validation_result.metadata.get('missing_sections'):
            report += "\n## Missing Sections\n\n"
            for section in validation_result.metadata['missing_sections']:
                report += f"- {section}\n"

        report += "\n---\n*Generated by PDFValidator v2.0.0*\n"

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Saved compliance report to {output_path}")

        return report
