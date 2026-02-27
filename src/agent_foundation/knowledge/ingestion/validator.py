"""Knowledge Validation using regex-based and LLM-based checks."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from agent_foundation.knowledge.ingestion.prompts.validation import VALIDATION_PROMPT
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.models.results import ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for knowledge validation."""

    enabled: bool = True
    checks_enabled: Set[str] = field(
        default_factory=lambda: {
            "correctness",
            "authenticity",
            "consistency",
            "completeness",
            "staleness",
            "security",
            "privacy",
            "policy_compliance",
        }
    )
    security_patterns: List[str] = field(
        default_factory=lambda: [
            r"(?i)(api[_-]?key|secret|password|token|credential)\s*[:=]",
            r"(?i)bearer\s+[a-zA-Z0-9\-._~+/]+=*",
        ]
    )
    privacy_patterns: List[str] = field(
        default_factory=lambda: [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        ]
    )


class KnowledgeValidator:
    """LLM-based knowledge validation.

    Performs fast regex checks for security/privacy patterns and
    LLM-based semantic checks for configurable quality categories.
    """

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        config: Optional[ValidationConfig] = None,
    ):
        self.llm_fn = llm_fn
        self.config = config or ValidationConfig()

    def validate(self, piece: KnowledgePiece) -> ValidationResult:
        """Validate a knowledge piece.

        Runs fast regex checks first (security, privacy), then LLM-based
        semantic checks for remaining enabled categories. Returns a
        ValidationResult with confidence as the ratio of passed checks.
        """
        if not self.config.enabled:
            return ValidationResult(
                is_valid=True,
                confidence=1.0,
                checks_passed=list(self.config.checks_enabled),
            )

        issues: List[str] = []
        suggestions: List[str] = []
        checks_passed: List[str] = []
        checks_failed: List[str] = []

        # Fast regex-based checks
        if "security" in self.config.checks_enabled:
            if self._check_patterns(piece.content, self.config.security_patterns):
                checks_failed.append("security")
                issues.append("Content may contain credentials or secrets")
            else:
                checks_passed.append("security")

        if "privacy" in self.config.checks_enabled:
            if self._check_patterns(piece.content, self.config.privacy_patterns):
                checks_failed.append("privacy")
                issues.append("Content may contain PII")
            else:
                checks_passed.append("privacy")

        # LLM-based checks
        llm_checks = self.config.checks_enabled - {"security", "privacy"}
        if llm_checks:
            llm_result = self._llm_validate(piece, llm_checks)
            checks_passed.extend(llm_result.get("passed", []))
            checks_failed.extend(llm_result.get("failed", []))
            issues.extend(llm_result.get("issues", []))
            suggestions.extend(llm_result.get("suggestions", []))

        is_valid = len(checks_failed) == 0
        total_checks = len(self.config.checks_enabled)
        confidence = len(checks_passed) / total_checks if total_checks > 0 else 1.0

        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            issues=issues,
            suggestions=suggestions,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )

    def _check_patterns(self, content: str, patterns: List[str]) -> bool:
        """Check content against a list of regex patterns."""
        return any(re.search(pattern, content) for pattern in patterns)

    def _llm_validate(
        self, piece: KnowledgePiece, checks: Set[str]
    ) -> Dict[str, List[str]]:
        """Run LLM-based validation checks.

        If no llm_fn is provided or the call fails, treats all LLM checks
        as passed and logs a warning (Requirement 14.4).
        """
        if self.llm_fn is None:
            logger.warning("No LLM function provided; treating LLM checks as passed")
            return {
                "passed": list(checks),
                "failed": [],
                "issues": [],
                "suggestions": [],
            }

        prompt = VALIDATION_PROMPT.format(
            content=piece.content[:1000],
            domain=piece.domain,
            source=piece.source or "unknown",
            created_at=piece.created_at or "unknown",
            checks_to_perform=", ".join(checks),
        )

        try:
            response = self.llm_fn(prompt)
            return json.loads(response)
        except Exception as e:
            logger.warning("LLM validation failed: %s", e)
            return {
                "passed": list(checks),
                "failed": [],
                "issues": [],
                "suggestions": [],
            }
