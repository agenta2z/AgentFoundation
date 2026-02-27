"""
Property-based tests for the knowledge validator module.

Feature: knowledge-module-migration
- Property 22: Validator security pattern detection

**Validates: Requirements 14.1**
"""
import sys
from pathlib import Path

# Path resolution for imports
_current_file = Path(__file__).resolve()
_current_path = _current_file.parent
while _current_path.name != "test" and _current_path.parent != _current_path:
    _current_path = _current_path.parent
_src_dir = _current_path.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import pytest
from hypothesis import given, settings, assume, strategies as st

from agent_foundation.knowledge.ingestion.validator import (
    KnowledgeValidator,
    ValidationConfig,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece


# ── Strategies ────────────────────────────────────────────────────────────────

# Alphabet that avoids accidentally forming security/privacy patterns
SAFE_ALPHABET = "abcdfghjklmnquvwxyz ABCDFGHJKLMNQUVWXYZ\n"

# Security pattern fragments to inject into content
SECURITY_FRAGMENTS = st.sampled_from([
    "api_key=SECRET123",
    "api-key=mysecret",
    "secret=hunter2",
    "password=p@ssw0rd",
    "token=abc123xyz",
    "credential:hidden",
    "API_KEY:val",
    "Bearer eyJhbGciOiJIUzI1NiJ9",
    "bearer abc123",
])


@st.composite
def content_with_security_pattern(draw):
    """Generate content that contains at least one security pattern."""
    prefix = draw(st.text(alphabet=SAFE_ALPHABET, min_size=0, max_size=50))
    fragment = draw(SECURITY_FRAGMENTS)
    suffix = draw(st.text(alphabet=SAFE_ALPHABET, min_size=0, max_size=50))
    return prefix + fragment + suffix


@st.composite
def clean_content(draw):
    """Generate content that does NOT match any default security or privacy patterns.

    Uses a restricted alphabet and avoids words/structures that could
    accidentally trigger the regex patterns.
    """
    text = draw(st.text(alphabet=SAFE_ALPHABET, min_size=1, max_size=200))
    # Ensure no accidental pattern matches by checking against defaults
    config = ValidationConfig()
    import re
    for pattern in config.security_patterns + config.privacy_patterns:
        assume(not re.search(pattern, text))
    return text


def _make_piece(content: str) -> KnowledgePiece:
    """Create a minimal KnowledgePiece for validation testing."""
    return KnowledgePiece(content=content, entity_id="test-entity")


# ── Property 22: Validator security pattern detection ─────────────────────────


class TestValidatorSecurityPatternDetection:
    """Property 22: Validator security pattern detection.

    *For any* content string containing a substring matching a security
    pattern (e.g., "api_key=...", "Bearer ..."), the validator should
    include "security" in checks_failed. *For any* content string not
    matching any security or privacy pattern, those checks should be in
    checks_passed.

    **Validates: Requirements 14.1**
    """

    @given(content=content_with_security_pattern())
    @settings(max_examples=100)
    def test_security_pattern_detected_in_checks_failed(self, content: str):
        """Content with a security pattern must have 'security' in checks_failed."""
        validator = KnowledgeValidator(llm_fn=None, config=ValidationConfig())
        piece = _make_piece(content)

        result = validator.validate(piece)

        assert "security" in result.checks_failed, (
            f"Expected 'security' in checks_failed for content containing "
            f"a security pattern, got checks_failed={result.checks_failed}"
        )
        assert "security" not in result.checks_passed

    @given(content=clean_content())
    @settings(max_examples=100)
    def test_clean_content_passes_security_and_privacy(self, content: str):
        """Content without any security or privacy patterns must pass both checks."""
        validator = KnowledgeValidator(llm_fn=None, config=ValidationConfig())
        piece = _make_piece(content)

        result = validator.validate(piece)

        assert "security" in result.checks_passed, (
            f"Expected 'security' in checks_passed for clean content, "
            f"got checks_passed={result.checks_passed}"
        )
        assert "privacy" in result.checks_passed, (
            f"Expected 'privacy' in checks_passed for clean content, "
            f"got checks_passed={result.checks_passed}"
        )
        assert "security" not in result.checks_failed
        assert "privacy" not in result.checks_failed
