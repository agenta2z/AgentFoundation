"""Unit tests for KnowledgeValidator and ValidationConfig."""

import json
from typing import List

import pytest

from agent_foundation.knowledge.ingestion.validator import (
    KnowledgeValidator,
    ValidationConfig,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_piece(content: str, **kwargs) -> KnowledgePiece:
    """Create a KnowledgePiece with the given content."""
    return KnowledgePiece(content=content, **kwargs)


def _make_llm_fn(passed: List[str] = None, failed: List[str] = None,
                 issues: List[str] = None, suggestions: List[str] = None):
    """Return a fake LLM function that returns a canned JSON response."""
    result = {
        "passed": passed or [],
        "failed": failed or [],
        "issues": issues or [],
        "suggestions": suggestions or [],
    }
    return lambda _prompt: json.dumps(result)


# ---------------------------------------------------------------------------
# ValidationConfig tests
# ---------------------------------------------------------------------------

class TestValidationConfig:
    def test_defaults(self):
        cfg = ValidationConfig()
        assert cfg.enabled is True
        assert "security" in cfg.checks_enabled
        assert "privacy" in cfg.checks_enabled
        assert "correctness" in cfg.checks_enabled
        assert len(cfg.security_patterns) > 0
        assert len(cfg.privacy_patterns) > 0

    def test_custom_checks(self):
        cfg = ValidationConfig(checks_enabled={"security"})
        assert cfg.checks_enabled == {"security"}

    def test_custom_patterns(self):
        cfg = ValidationConfig(
            security_patterns=[r"SECRET"],
            privacy_patterns=[r"SSN"],
        )
        assert cfg.security_patterns == [r"SECRET"]
        assert cfg.privacy_patterns == [r"SSN"]


# ---------------------------------------------------------------------------
# Security pattern detection
# ---------------------------------------------------------------------------

class TestSecurityPatterns:
    def test_api_key_detected(self):
        piece = _make_piece("config: api_key=abc123secret")
        validator = KnowledgeValidator(config=ValidationConfig(checks_enabled={"security"}))
        result = validator.validate(piece)
        assert not result.is_valid
        assert "security" in result.checks_failed

    def test_password_detected(self):
        piece = _make_piece("password=hunter2")
        validator = KnowledgeValidator(config=ValidationConfig(checks_enabled={"security"}))
        result = validator.validate(piece)
        assert not result.is_valid
        assert "security" in result.checks_failed

    def test_bearer_token_detected(self):
        piece = _make_piece("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9")
        validator = KnowledgeValidator(config=ValidationConfig(checks_enabled={"security"}))
        result = validator.validate(piece)
        assert not result.is_valid
        assert "security" in result.checks_failed

    def test_secret_key_detected(self):
        piece = _make_piece("aws secret=AKIAIOSFODNN7EXAMPLE")
        validator = KnowledgeValidator(config=ValidationConfig(checks_enabled={"security"}))
        result = validator.validate(piece)
        assert not result.is_valid
        assert "security" in result.checks_failed

    def test_clean_content_passes_security(self):
        piece = _make_piece("This is a normal document about Python programming.")
        validator = KnowledgeValidator(config=ValidationConfig(checks_enabled={"security"}))
        result = validator.validate(piece)
        assert result.is_valid
        assert "security" in result.checks_passed


# ---------------------------------------------------------------------------
# Privacy pattern detection
# ---------------------------------------------------------------------------

class TestPrivacyPatterns:
    def test_email_detected(self):
        piece = _make_piece("Contact us at user@example.com for details.")
        validator = KnowledgeValidator(config=ValidationConfig(checks_enabled={"privacy"}))
        result = validator.validate(piece)
        assert not result.is_valid
        assert "privacy" in result.checks_failed

    def test_phone_number_detected(self):
        piece = _make_piece("Call 555-123-4567 for support.")
        validator = KnowledgeValidator(config=ValidationConfig(checks_enabled={"privacy"}))
        result = validator.validate(piece)
        assert not result.is_valid
        assert "privacy" in result.checks_failed

    def test_phone_number_no_dashes_detected(self):
        piece = _make_piece("Phone: 5551234567")
        validator = KnowledgeValidator(config=ValidationConfig(checks_enabled={"privacy"}))
        result = validator.validate(piece)
        assert not result.is_valid
        assert "privacy" in result.checks_failed

    def test_clean_content_passes_privacy(self):
        piece = _make_piece("This document has no personal information.")
        validator = KnowledgeValidator(config=ValidationConfig(checks_enabled={"privacy"}))
        result = validator.validate(piece)
        assert result.is_valid
        assert "privacy" in result.checks_passed


# ---------------------------------------------------------------------------
# LLM-based validation
# ---------------------------------------------------------------------------

class TestLLMValidation:
    def test_llm_checks_passed(self):
        llm_fn = _make_llm_fn(passed=["correctness", "authenticity"])
        validator = KnowledgeValidator(
            llm_fn=llm_fn,
            config=ValidationConfig(checks_enabled={"correctness", "authenticity"}),
        )
        piece = _make_piece("Valid knowledge content.")
        result = validator.validate(piece)
        assert result.is_valid
        assert "correctness" in result.checks_passed
        assert "authenticity" in result.checks_passed

    def test_llm_checks_failed(self):
        llm_fn = _make_llm_fn(
            passed=[],
            failed=["correctness"],
            issues=["Content is factually incorrect"],
            suggestions=["Verify with authoritative source"],
        )
        validator = KnowledgeValidator(
            llm_fn=llm_fn,
            config=ValidationConfig(checks_enabled={"correctness"}),
        )
        piece = _make_piece("Incorrect knowledge.")
        result = validator.validate(piece)
        assert not result.is_valid
        assert "correctness" in result.checks_failed
        assert "Content is factually incorrect" in result.issues
        assert "Verify with authoritative source" in result.suggestions

    def test_llm_failure_treats_checks_as_passed(self):
        """Requirement 14.4: LLM failure => all LLM checks treated as passed."""
        def failing_llm(_prompt):
            raise RuntimeError("LLM service unavailable")

        validator = KnowledgeValidator(
            llm_fn=failing_llm,
            config=ValidationConfig(checks_enabled={"correctness", "staleness"}),
        )
        piece = _make_piece("Some content.")
        result = validator.validate(piece)
        assert result.is_valid
        assert "correctness" in result.checks_passed
        assert "staleness" in result.checks_passed

    def test_llm_returns_invalid_json_treats_as_passed(self):
        """LLM returning non-JSON should be treated as failure => checks passed."""
        validator = KnowledgeValidator(
            llm_fn=lambda _: "not json at all",
            config=ValidationConfig(checks_enabled={"completeness"}),
        )
        piece = _make_piece("Some content.")
        result = validator.validate(piece)
        assert result.is_valid
        assert "completeness" in result.checks_passed

    def test_no_llm_fn_treats_llm_checks_as_passed(self):
        """When no llm_fn is provided, LLM checks should pass."""
        validator = KnowledgeValidator(
            llm_fn=None,
            config=ValidationConfig(checks_enabled={"correctness", "security"}),
        )
        piece = _make_piece("Clean content with no secrets.")
        result = validator.validate(piece)
        assert result.is_valid
        assert "correctness" in result.checks_passed
        assert "security" in result.checks_passed


# ---------------------------------------------------------------------------
# Combined regex + LLM checks
# ---------------------------------------------------------------------------

class TestCombinedValidation:
    def test_security_fail_with_llm_pass(self):
        """Security regex fails but LLM checks pass => overall invalid."""
        llm_fn = _make_llm_fn(passed=["correctness"])
        validator = KnowledgeValidator(
            llm_fn=llm_fn,
            config=ValidationConfig(checks_enabled={"security", "correctness"}),
        )
        piece = _make_piece("api_key=secret123 but otherwise good content")
        result = validator.validate(piece)
        assert not result.is_valid
        assert "security" in result.checks_failed
        assert "correctness" in result.checks_passed

    def test_all_checks_pass(self):
        """All regex and LLM checks pass => valid with full confidence."""
        llm_fn = _make_llm_fn(
            passed=["correctness", "authenticity", "consistency",
                     "completeness", "staleness", "policy_compliance"]
        )
        validator = KnowledgeValidator(llm_fn=llm_fn)
        piece = _make_piece("Clean, valid knowledge content.")
        result = validator.validate(piece)
        assert result.is_valid
        assert result.confidence == 1.0
        assert len(result.checks_failed) == 0

    def test_confidence_ratio(self):
        """Confidence = passed / total checks."""
        llm_fn = _make_llm_fn(passed=["correctness"], failed=["staleness"])
        validator = KnowledgeValidator(
            llm_fn=llm_fn,
            config=ValidationConfig(
                checks_enabled={"security", "correctness", "staleness"}
            ),
        )
        piece = _make_piece("Normal content, no secrets.")
        result = validator.validate(piece)
        # security passes (regex), correctness passes (LLM), staleness fails (LLM)
        # 2 passed / 3 total
        assert result.confidence == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# Disabled validation
# ---------------------------------------------------------------------------

class TestDisabledValidation:
    def test_disabled_returns_valid(self):
        validator = KnowledgeValidator(
            config=ValidationConfig(enabled=False),
        )
        piece = _make_piece("api_key=secret password=hunter2")
        result = validator.validate(piece)
        assert result.is_valid
        assert result.confidence == 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_content(self):
        piece = _make_piece("")
        validator = KnowledgeValidator(
            config=ValidationConfig(checks_enabled={"security", "privacy"}),
        )
        result = validator.validate(piece)
        assert result.is_valid
        assert "security" in result.checks_passed
        assert "privacy" in result.checks_passed

    def test_only_llm_checks_no_regex(self):
        """When only LLM checks are enabled, no regex checks run."""
        llm_fn = _make_llm_fn(passed=["correctness"])
        validator = KnowledgeValidator(
            llm_fn=llm_fn,
            config=ValidationConfig(checks_enabled={"correctness"}),
        )
        piece = _make_piece("api_key=secret but only LLM checks enabled")
        result = validator.validate(piece)
        assert result.is_valid
        assert "correctness" in result.checks_passed
        assert "security" not in result.checks_passed
        assert "security" not in result.checks_failed

    def test_empty_checks_enabled(self):
        """No checks enabled => valid with confidence 1.0."""
        validator = KnowledgeValidator(
            config=ValidationConfig(checks_enabled=set()),
        )
        piece = _make_piece("api_key=secret")
        result = validator.validate(piece)
        assert result.is_valid
        assert result.confidence == 1.0
        assert len(result.checks_passed) == 0
        assert len(result.checks_failed) == 0
