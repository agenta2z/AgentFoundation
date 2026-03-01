"""
Property-based tests for SpaceClassifier.

Feature: knowledge-space-restructuring
- Property 9: SpaceClassifier Additive Rules
- Property 10: SpaceClassifier Developmental Exclusivity

**Validates: Requirements 6.1, 6.2, 6.3**
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

_rpu_src = Path(__file__).resolve().parents[4] / "RichPythonUtils" / "src"
if _rpu_src.exists() and str(_rpu_src) not in sys.path:
    sys.path.insert(0, str(_rpu_src))

from hypothesis import given, settings, strategies as st

from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.ingestion.space_classifier import (
    SpaceClassifier,
    ClassificationResult,
)


# ── Strategies ───────────────────────────────────────────────────────────────

_non_empty_text = st.text(min_size=1).filter(lambda s: s.strip())

# Entity IDs that start with "user:" (triggers personal rule)
_user_entity_id = st.text(min_size=1, max_size=30).map(lambda s: f"user:{s}")

# Entity IDs that do NOT start with "user:" (should get "main")
_non_user_entity_id = st.one_of(
    st.none(),
    st.text(min_size=1, max_size=30).filter(
        lambda s: not s.startswith("user:")
    ),
)

# Validation statuses that are NOT "failed" (non-developmental)
_non_failed_validation = st.sampled_from(["not_validated", "pending", "passed"])

# info_type values that are NOT "user_profile"
_non_profile_info_type = st.sampled_from(["context", "instructions", "skills"])


# Feature: knowledge-space-restructuring, Property 9: SpaceClassifier Additive Rules


class TestSpaceClassifierAdditiveRules:
    """Property 9: SpaceClassifier Additive Rules.

    For any KnowledgePiece with entity_id starting with "user:" and
    validation_status != "failed", the SpaceClassifier SHALL include "personal"
    in the returned auto_spaces. For any KnowledgePiece without a user entity
    prefix and validation_status != "failed", the SpaceClassifier SHALL include
    "main" in the returned auto_spaces.

    **Validates: Requirements 6.1, 6.2**
    """

    @given(
        content=_non_empty_text,
        entity_id=_user_entity_id,
        validation_status=_non_failed_validation,
        info_type=_non_profile_info_type,
    )
    @settings(max_examples=100)
    def test_user_entity_gets_personal_space(
        self, content, entity_id, validation_status, info_type
    ):
        """Pieces with user: entity_id prefix get "personal" in auto_spaces.

        **Validates: Requirements 6.1**
        """
        piece = KnowledgePiece(
            content=content,
            entity_id=entity_id,
            validation_status=validation_status,
            info_type=info_type,
        )
        classifier = SpaceClassifier()
        result = classifier.classify_piece(piece)

        assert "personal" in result.auto_spaces
        # Main should also be present (additive)
        assert "main" in result.auto_spaces

    @given(
        content=_non_empty_text,
        entity_id=_non_user_entity_id,
        validation_status=_non_failed_validation,
        info_type=_non_profile_info_type,
    )
    @settings(max_examples=100)
    def test_non_user_entity_gets_main_space(
        self, content, entity_id, validation_status, info_type
    ):
        """Pieces without user: entity_id prefix get "main" in auto_spaces.

        **Validates: Requirements 6.2**
        """
        piece = KnowledgePiece(
            content=content,
            entity_id=entity_id,
            validation_status=validation_status,
            info_type=info_type,
        )
        classifier = SpaceClassifier()
        result = classifier.classify_piece(piece)

        assert "main" in result.auto_spaces
        # Personal should NOT be present (no user: prefix, no user_profile info_type)
        assert "personal" not in result.auto_spaces

    @given(
        content=_non_empty_text,
        entity_id=_non_user_entity_id,
        validation_status=_non_failed_validation,
    )
    @settings(max_examples=100)
    def test_user_profile_info_type_gets_personal_space(
        self, content, entity_id, validation_status
    ):
        """Pieces with info_type="user_profile" get "personal" in auto_spaces.

        **Validates: Requirements 6.1**
        """
        piece = KnowledgePiece(
            content=content,
            entity_id=entity_id,
            validation_status=validation_status,
            info_type="user_profile",
        )
        classifier = SpaceClassifier()
        result = classifier.classify_piece(piece)

        assert "personal" in result.auto_spaces
        assert "main" in result.auto_spaces


# Feature: knowledge-space-restructuring, Property 10: SpaceClassifier Developmental Exclusivity


class TestSpaceClassifierDevelopmentalExclusivity:
    """Property 10: SpaceClassifier Developmental Exclusivity.

    For any KnowledgePiece with validation_status == "failed", the
    SpaceClassifier SHALL return exactly ["developmental"], regardless of
    entity_id or other fields.

    **Validates: Requirements 6.3**
    """

    @given(
        content=_non_empty_text,
        entity_id=st.one_of(_user_entity_id, _non_user_entity_id),
        info_type=st.sampled_from(["context", "instructions", "user_profile", "skills"]),
        knowledge_type=st.sampled_from(list(KnowledgeType)),
    )
    @settings(max_examples=100)
    def test_failed_validation_returns_only_developmental(
        self, content, entity_id, info_type, knowledge_type
    ):
        """Pieces with validation_status="failed" get exactly ["developmental"].

        **Validates: Requirements 6.3**
        """
        piece = KnowledgePiece(
            content=content,
            entity_id=entity_id,
            validation_status="failed",
            info_type=info_type,
            knowledge_type=knowledge_type,
        )
        classifier = SpaceClassifier()
        result = classifier.classify_piece(piece)

        assert result.auto_spaces == ["developmental"]
        # No suggestions should be present for exclusive auto rules
        assert result.suggested_spaces == []
