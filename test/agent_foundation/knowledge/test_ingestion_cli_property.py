"""
Property-based tests for KnowledgeIngestionCLI schema validation.

Property 4: Schema validation rejects invalid structures.
For any JSON dictionary that is missing one or more of the required top-level
sections (metadata, pieces, graph), or where pieces entries are missing required
fields (piece_id, content, knowledge_type, info_type), the _parse_and_validate
method of KnowledgeIngestionCLI should raise a ValueError.

# Feature: knowledge-agent-integration, Property 4: Schema validation rejects invalid structures

**Validates: Requirements 8.2**
"""
import json
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
_spu_src = Path(__file__).resolve().parents[3] / "SciencePythonUtils" / "src"
if _spu_src.exists() and str(_spu_src) not in sys.path:
    sys.path.insert(0, str(_spu_src))

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from science_modeling_tools.knowledge.ingestion_cli import KnowledgeIngestionCLI


# ── Hypothesis strategies ────────────────────────────────────────────────────


@st.composite
def missing_section_dicts(draw):
    """Generate dicts missing at least one required top-level section."""
    sections = ["metadata", "pieces", "graph"]
    # Choose which sections to include (at least one must be missing)
    included = draw(
        st.lists(st.sampled_from(sections), min_size=0, max_size=2, unique=True)
    )
    assume(set(included) != set(sections))  # At least one missing

    result = {}
    if "metadata" in included:
        result["metadata"] = {}
    if "pieces" in included:
        result["pieces"] = []
    if "graph" in included:
        result["graph"] = {"nodes": [], "edges": []}
    return result


@st.composite
def missing_field_pieces(draw):
    """Generate a valid structure but with pieces missing required fields."""
    required_fields = ["piece_id", "content", "knowledge_type", "info_type"]
    # Choose which fields to include (at least one must be missing)
    included = draw(
        st.lists(
            st.sampled_from(required_fields), min_size=0, max_size=3, unique=True
        )
    )
    assume(set(included) != set(required_fields))  # At least one missing

    piece = {}
    if "piece_id" in included:
        piece["piece_id"] = draw(st.text(min_size=1, max_size=20))
    if "content" in included:
        piece["content"] = draw(st.text(min_size=1, max_size=50))
    if "knowledge_type" in included:
        piece["knowledge_type"] = draw(
            st.sampled_from(["fact", "instruction", "procedure"])
        )
    if "info_type" in included:
        piece["info_type"] = draw(
            st.sampled_from(["user_profile", "instructions", "context"])
        )

    return {
        "metadata": {},
        "pieces": [piece],
        "graph": {"nodes": [], "edges": []},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Property 4: Schema validation rejects invalid structures
# ══════════════════════════════════════════════════════════════════════════════


class TestSchemaValidationProperty:
    """Property 4: Schema validation rejects invalid structures.

    For any JSON dictionary that is missing one or more of the required
    top-level sections (metadata, pieces, graph), or where pieces entries
    are missing required fields (piece_id, content, knowledge_type, info_type),
    the _parse_and_validate method of KnowledgeIngestionCLI should raise
    a ValueError.

    **Validates: Requirements 8.2**
    """

    @given(data=missing_section_dicts())
    @settings(max_examples=100)
    def test_missing_sections_rejected(self, data):
        """For any dict missing required top-level sections, _parse_and_validate raises ValueError.

        **Validates: Requirements 8.2**
        """
        cli = KnowledgeIngestionCLI(inferencer=lambda p: "")
        with pytest.raises(ValueError):
            cli._parse_and_validate(json.dumps(data))

    @given(data=missing_field_pieces())
    @settings(max_examples=100)
    def test_missing_piece_fields_rejected(self, data):
        """For any piece missing required fields, _parse_and_validate raises ValueError.

        **Validates: Requirements 8.2**
        """
        cli = KnowledgeIngestionCLI(inferencer=lambda p: "")
        with pytest.raises(ValueError):
            cli._parse_and_validate(json.dumps(data))
