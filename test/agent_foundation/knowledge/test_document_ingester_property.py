"""
Property-based tests for DocumentIngester ingestion pipeline.

Feature: knowledge-module-migration
- Property 30: Failed validation moves piece to developmental space

**Validates: Requirements 21.4**
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

        # Validation status must be "failed"
        assert output_piece["validation_status"] == "failed", (
            f"Expected validation_status='failed', got '{output_piece['validation_status']}'"
        )

        # Enhancement counts should reflect the failure
        assert counts["failed_validation"] == 1
