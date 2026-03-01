"""
Property-based tests for DocumentIngester ingestion pipeline.

Feature: knowledge-module-migration
- Property 30: Failed validation moves piece to developmental space

Feature: knowledge-space-restructuring
- Space classifier integration in ingestion pipeline

**Validates: Requirements 21.4, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6**
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Path resolution for imports
_current_file = Path(__file__).resolve()
_current_path = _current_file.parent
while _current_path.name != "test" and _current_path.parent != _current_path:
    _current_path = _current_path.parent
_src_dir = _current_path.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from hypothesis import given, settings, strategies as st

from agent_foundation.knowledge.ingestion.document_ingester import (
    DocumentIngester,
    IngesterConfig,
)
from agent_foundation.knowledge.ingestion.validator import (
    KnowledgeValidator,
    ValidationResult,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)

# ── Strategies ───────────────────────────────────────────────────────────────

_non_empty_text = st.text(min_size=1).filter(lambda s: s.strip())

_identifier_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=50,
)


@st.composite
def piece_dict_strategy(draw):
    """Generate a random piece dict suitable for _apply_enhancements input."""
    content = draw(_non_empty_text)
    piece_id = draw(_identifier_text)
    knowledge_type = draw(st.sampled_from([kt.value for kt in KnowledgeType]))
    info_type = draw(st.sampled_from(["user_profile", "instructions", "context"]))
    domain = draw(st.sampled_from(["general", "model_optimization", "data_engineering", "testing"]))
    space = draw(st.sampled_from(["main", "personal"]))

    return {
        "piece_id": piece_id,
        "content": content,
        "knowledge_type": knowledge_type,
        "info_type": info_type,
        "domain": domain,
        "space": space,
    }


# ── Property 30: Failed validation moves piece to developmental space ────────


class TestFailedValidationMovesDevelopmental:
    """Property 30: Failed validation moves piece to developmental space.

    For any piece that fails validation during ingestion, the piece's
    ``space`` field should be set to "developmental" rather than being
    discarded.

    **Validates: Requirements 21.4**
    """

    @given(piece_dict=piece_dict_strategy())
    @settings(max_examples=100)
    def test_failed_validation_sets_developmental_space(self, piece_dict):
        """A piece that fails validation is moved to developmental space, not discarded.

        **Validates: Requirements 21.4**
        """
        mock_validator = MagicMock(spec=KnowledgeValidator)
        mock_validator.validate.return_value = ValidationResult(
            is_valid=False,
            confidence=0.0,
            issues=["validation failed"],
            checks_failed=["correctness"],
        )

        ingester = DocumentIngester(
            inferencer=lambda p: "",
            validator=mock_validator,
        )

        data = {"pieces": [piece_dict]}
        result_data, counts, _ = ingester._apply_enhancements(data)

        # Piece must NOT be discarded
        assert len(result_data["pieces"]) == 1, (
            "Failed-validation piece was discarded instead of kept"
        )

        output_piece = result_data["pieces"][0]

        # Space must be "developmental"
        assert output_piece["space"] == "developmental", (
            f"Expected space='developmental', got space='{output_piece['space']}'"
        )

        # Spaces must be ["developmental"] (Req 7.4)
        assert output_piece["spaces"] == ["developmental"], (
            f"Expected spaces=['developmental'], got spaces='{output_piece['spaces']}'"
        )

        # Validation status must be "failed"
        assert output_piece["validation_status"] == "failed", (
            f"Expected validation_status='failed', got '{output_piece['validation_status']}'"
        )

        # Enhancement counts should reflect the failure
        assert counts["failed_validation"] == 1


from agent_foundation.knowledge.ingestion.space_classifier import (
    ClassificationResult,
    SpaceClassifier,
    SpaceRule,
)


# ── Strategies for space classifier integration tests ────────────────────────

_valid_spaces = st.sampled_from(["main", "personal", "developmental"])

_user_entity_id = st.text(min_size=1, max_size=20).map(lambda s: f"user:{s}")
_non_user_entity_id = st.one_of(
    st.text(min_size=1, max_size=20).map(lambda s: f"service:{s}"),
    st.none(),
)


@st.composite
def piece_dict_with_entity_strategy(draw):
    """Generate a piece dict with a specific entity_id pattern."""
    content = draw(_non_empty_text)
    piece_id = draw(_identifier_text)
    knowledge_type = draw(st.sampled_from([kt.value for kt in KnowledgeType]))
    info_type = draw(st.sampled_from(["user_profile", "instructions", "context"]))
    domain = draw(st.sampled_from(["general", "model_optimization", "data_engineering"]))
    entity_id = draw(st.one_of(_user_entity_id, _non_user_entity_id))

    d = {
        "piece_id": piece_id,
        "content": content,
        "knowledge_type": knowledge_type,
        "info_type": info_type,
        "domain": domain,
    }
    if entity_id is not None:
        d["entity_id"] = entity_id
    return d


@st.composite
def user_spaces_strategy(draw):
    """Generate a non-empty list of valid user-specified spaces."""
    spaces = draw(st.lists(_valid_spaces, min_size=1, max_size=3))
    # Deduplicate preserving order
    return list(dict.fromkeys(spaces))


# ── Space Classifier Integration Property Tests ──────────────────────────────


class TestSpaceClassifierIntegration:
    """Property tests for SpaceClassifier integration in DocumentIngester.

    **Validates: Requirements 7.1, 7.4, 7.5, 7.6**
    """

    @given(piece_dict=piece_dict_with_entity_strategy())
    @settings(max_examples=100)
    def test_classifier_invoked_on_valid_pieces(self, piece_dict):
        """SpaceClassifier is invoked on each piece and auto_spaces are applied.

        **Validates: Requirements 7.1**
        """
        ingester = DocumentIngester(inferencer=lambda p: "")

        data = {"pieces": [piece_dict]}
        result_data, counts, _ = ingester._apply_enhancements(data)

        assert len(result_data["pieces"]) == 1
        output_piece = result_data["pieces"][0]

        # Piece must have spaces set by the classifier
        assert "spaces" in output_piece
        assert isinstance(output_piece["spaces"], list)
        assert len(output_piece["spaces"]) > 0

        # space must equal spaces[0]
        assert output_piece["space"] == output_piece["spaces"][0]

        # Verify classifier logic: user: entity_id → personal in spaces
        entity_id = piece_dict.get("entity_id")
        if entity_id and entity_id.startswith("user:"):
            assert "personal" in output_piece["spaces"], (
                f"Expected 'personal' in spaces for entity_id='{entity_id}', "
                f"got {output_piece['spaces']}"
            )
        elif piece_dict.get("info_type") == "user_profile":
            assert "personal" in output_piece["spaces"], (
                f"Expected 'personal' in spaces for info_type='user_profile', "
                f"got {output_piece['spaces']}"
            )

    @given(piece_dict=piece_dict_strategy(), user_spaces=user_spaces_strategy())
    @settings(max_examples=100)
    def test_user_specified_spaces_override_classifier(self, piece_dict, user_spaces):
        """When user provides spaces, the classifier is skipped.

        **Validates: Requirements 7.5**
        """
        ingester = DocumentIngester(inferencer=lambda p: "")

        data = {"pieces": [piece_dict]}
        result_data, counts, _ = ingester._apply_enhancements(
            data, user_spaces=user_spaces
        )

        assert len(result_data["pieces"]) == 1
        output_piece = result_data["pieces"][0]

        # Spaces must match user-specified spaces exactly
        assert output_piece["spaces"] == user_spaces, (
            f"Expected spaces={user_spaces}, got {output_piece['spaces']}"
        )
        assert output_piece["space"] == user_spaces[0]

    @given(piece_dict=piece_dict_strategy())
    @settings(max_examples=100)
    def test_failed_validation_overrides_user_spaces(self, piece_dict):
        """Failed validation sets spaces=["developmental"] even with user-specified spaces.

        **Validates: Requirements 7.4**
        """
        mock_validator = MagicMock(spec=KnowledgeValidator)
        mock_validator.validate.return_value = ValidationResult(
            is_valid=False,
            confidence=0.0,
            issues=["validation failed"],
            checks_failed=["correctness"],
        )

        ingester = DocumentIngester(
            inferencer=lambda p: "",
            validator=mock_validator,
        )

        data = {"pieces": [piece_dict]}
        result_data, counts, _ = ingester._apply_enhancements(
            data, user_spaces=["personal", "main"]
        )

        assert len(result_data["pieces"]) == 1
        output_piece = result_data["pieces"][0]

        # Failed validation must override user spaces
        assert output_piece["space"] == "developmental"
        assert output_piece["spaces"] == ["developmental"]

    @given(piece_dict=piece_dict_with_entity_strategy())
    @settings(max_examples=100)
    def test_suggestion_mode_rules_stored_on_piece(self, piece_dict):
        """Suggestion-mode rules store pending suggestions on the piece.

        **Validates: Requirements 7.6**
        """
        # Create a classifier with a suggestion-mode rule
        suggestion_rule = SpaceRule(
            name="test_suggestion",
            space="personal",
            condition=lambda _: True,  # Always triggers
            priority=10,
            mode="suggestion",
        )
        classifier = SpaceClassifier(rules=[
            suggestion_rule,
            SpaceRule(
                name="main_default",
                space="main",
                condition=lambda _: True,
                priority=0,
                mode="auto",
            ),
        ])

        ingester = DocumentIngester(
            inferencer=lambda p: "",
            space_classifier=classifier,
        )

        data = {"pieces": [piece_dict]}
        result_data, counts, _ = ingester._apply_enhancements(data)

        assert len(result_data["pieces"]) == 1
        output_piece = result_data["pieces"][0]

        # Auto spaces should be applied
        assert "main" in output_piece["spaces"]

        # Suggestions should be stored
        assert output_piece.get("pending_space_suggestions") == ["personal"]
        assert output_piece.get("space_suggestion_status") == "pending"
        assert len(output_piece.get("space_suggestion_reasons", [])) == 1
