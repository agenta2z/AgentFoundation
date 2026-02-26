"""
Unit tests for KnowledgeFormatter and RetrievalResult.

Tests cover:
- Empty results return empty string
- Metadata formatting with sorted keys
- Knowledge pieces formatting grouped by type
- Graph context formatting with sorted entries
- Deterministic output regardless of input order
- include_tags and include_scores options
- Global metadata formatting
- Combined sections with section_delimiter

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
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

from science_modeling_tools.knowledge.formatter import (
    KnowledgeFormatter,
    RetrievalResult,
)
from science_modeling_tools.knowledge.models.entity_metadata import EntityMetadata
from science_modeling_tools.knowledge.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)


@pytest.fixture
def formatter():
    """Default formatter with tags enabled, scores disabled."""
    return KnowledgeFormatter()


@pytest.fixture
def sample_metadata():
    """Sample entity metadata."""
    return EntityMetadata(
        entity_id="user:xinli",
        entity_type="user",
        properties={"name": "Xinli", "location": "Seattle", "zip": "98121"},
        created_at="2025-01-15T10:30:00+00:00",
        updated_at="2025-01-15T10:30:00+00:00",
    )


@pytest.fixture
def sample_pieces():
    """Sample knowledge pieces with scores."""
    piece1 = KnowledgePiece(
        content="Prefers organic/pasture-raised eggs",
        piece_id="piece-001",
        knowledge_type=KnowledgeType.Preference,
        tags=["grocery", "eggs", "organic"],
        created_at="2025-01-15T10:30:00+00:00",
        updated_at="2025-01-15T10:30:00+00:00",
    )
    piece2 = KnowledgePiece(
        content="When checking prices, compare at least 2 stores before reporting",
        piece_id="piece-002",
        knowledge_type=KnowledgeType.Instruction,
        tags=["grocery", "workflow"],
        created_at="2025-01-15T10:30:00+00:00",
        updated_at="2025-01-15T10:30:00+00:00",
    )
    return [(piece1, 0.9), (piece2, 0.8)]


@pytest.fixture
def sample_graph_context():
    """Sample graph context entries."""
    return [
        {
            "relation_type": "SHOPS_AT",
            "target_node_id": "store:costco",
            "target_label": "Costco",
            "piece": KnowledgePiece(
                content="Costco Executive membership - 2% cashback",
                piece_id="gp-001",
                created_at="2025-01-15T10:30:00+00:00",
                updated_at="2025-01-15T10:30:00+00:00",
            ),
            "depth": 1,
        },
        {
            "relation_type": "SHOPS_AT",
            "target_node_id": "store:qfc",
            "target_label": "",
            "piece": None,
            "depth": 1,
        },
        {
            "relation_type": "SHOPS_AT",
            "target_node_id": "store:whole_foods",
            "target_label": "",
            "piece": None,
            "depth": 1,
        },
    ]


class TestRetrievalResult:
    """Tests for RetrievalResult data class."""

    def test_default_empty(self):
        """Empty RetrievalResult has None metadata and empty lists."""
        result = RetrievalResult()
        assert result.metadata is None
        assert result.global_metadata is None
        assert result.pieces == []
        assert result.graph_context == []

    def test_with_all_fields(self, sample_metadata, sample_pieces, sample_graph_context):
        """RetrievalResult can hold all three layers of data."""
        result = RetrievalResult(
            metadata=sample_metadata,
            pieces=sample_pieces,
            graph_context=sample_graph_context,
        )
        assert result.metadata is sample_metadata
        assert len(result.pieces) == 2
        assert len(result.graph_context) == 3


class TestFormatterEmptyResults:
    """Tests for empty result handling."""

    def test_empty_result_returns_empty_string(self, formatter):
        """Empty RetrievalResult produces empty string."""
        result = RetrievalResult()
        assert formatter.format(result) == ""

    def test_metadata_with_empty_properties(self, formatter):
        """Metadata with no properties is treated as empty."""
        metadata = EntityMetadata(
            entity_id="user:test",
            entity_type="user",
            properties={},
        )
        result = RetrievalResult(metadata=metadata)
        assert formatter.format(result) == ""

    def test_none_metadata_no_pieces_no_graph(self, formatter):
        """All None/empty fields produce empty string."""
        result = RetrievalResult(
            metadata=None,
            global_metadata=None,
            pieces=[],
            graph_context=[],
        )
        assert formatter.format(result) == ""


class TestFormatterMetadata:
    """Tests for metadata formatting."""

    def test_metadata_sorted_keys(self, formatter, sample_metadata):
        """Metadata keys are sorted alphabetically."""
        result = RetrievalResult(metadata=sample_metadata)
        output = formatter.format(result)
        assert output == "[Metadata]\nlocation: Seattle\nname: Xinli\nzip: 98121"

    def test_metadata_single_key(self, formatter):
        """Single key metadata formats correctly."""
        metadata = EntityMetadata(
            entity_id="user:test",
            entity_type="user",
            properties={"name": "Test"},
        )
        result = RetrievalResult(metadata=metadata)
        output = formatter.format(result)
        assert output == "[Metadata]\nname: Test"

    def test_global_metadata_label(self, formatter):
        """Global metadata uses 'Global Metadata' label."""
        global_meta = EntityMetadata(
            entity_id="global",
            entity_type="global",
            properties={"version": "1.0"},
        )
        result = RetrievalResult(global_metadata=global_meta)
        output = formatter.format(result)
        assert output == "[Global Metadata]\nversion: 1.0"

    def test_both_metadata_sections(self, formatter, sample_metadata):
        """Both entity and global metadata appear as separate sections."""
        global_meta = EntityMetadata(
            entity_id="global",
            entity_type="global",
            properties={"app": "grocery_checker"},
        )
        result = RetrievalResult(
            metadata=sample_metadata,
            global_metadata=global_meta,
        )
        output = formatter.format(result)
        assert "[Metadata]" in output
        assert "[Global Metadata]" in output
        assert "name: Xinli" in output
        assert "app: grocery_checker" in output


class TestFormatterPieces:
    """Tests for knowledge piece formatting."""

    def test_pieces_grouped_by_type(self, formatter, sample_pieces):
        """Pieces are grouped by KnowledgeType."""
        result = RetrievalResult(pieces=sample_pieces)
        output = formatter.format(result)
        # instruction comes before preference alphabetically
        instruction_pos = output.index("[instruction]")
        preference_pos = output.index("[preference]")
        assert instruction_pos < preference_pos

    def test_pieces_with_tags(self, formatter):
        """Tags are shown when include_tags is True."""
        piece = KnowledgePiece(
            content="Test content",
            piece_id="p1",
            knowledge_type=KnowledgeType.Fact,
            tags=["alpha", "beta"],
            created_at="2025-01-15T10:30:00+00:00",
            updated_at="2025-01-15T10:30:00+00:00",
        )
        result = RetrievalResult(pieces=[(piece, 0.9)])
        output = formatter.format(result)
        assert "Tags: alpha, beta" in output

    def test_pieces_without_tags(self):
        """Tags are hidden when include_tags is False."""
        formatter = KnowledgeFormatter(include_tags=False)
        piece = KnowledgePiece(
            content="Test content",
            piece_id="p1",
            knowledge_type=KnowledgeType.Fact,
            tags=["alpha", "beta"],
            created_at="2025-01-15T10:30:00+00:00",
            updated_at="2025-01-15T10:30:00+00:00",
        )
        result = RetrievalResult(pieces=[(piece, 0.9)])
        output = formatter.format(result)
        assert "Tags:" not in output

    def test_pieces_with_scores(self):
        """Scores are shown when include_scores is True."""
        formatter = KnowledgeFormatter(include_scores=True)
        piece = KnowledgePiece(
            content="Test content",
            piece_id="p1",
            knowledge_type=KnowledgeType.Fact,
            tags=[],
            created_at="2025-01-15T10:30:00+00:00",
            updated_at="2025-01-15T10:30:00+00:00",
        )
        result = RetrievalResult(pieces=[(piece, 0.85)])
        output = formatter.format(result)
        assert "(score: 0.85)" in output

    def test_pieces_without_scores(self, formatter):
        """Scores are hidden by default."""
        piece = KnowledgePiece(
            content="Test content",
            piece_id="p1",
            knowledge_type=KnowledgeType.Fact,
            tags=[],
            created_at="2025-01-15T10:30:00+00:00",
            updated_at="2025-01-15T10:30:00+00:00",
        )
        result = RetrievalResult(pieces=[(piece, 0.85)])
        output = formatter.format(result)
        assert "score:" not in output

    def test_pieces_sorted_by_score_then_id(self, formatter):
        """Pieces with same type are sorted by score desc, then piece_id asc."""
        piece_a = KnowledgePiece(
            content="A content",
            piece_id="aaa",
            knowledge_type=KnowledgeType.Fact,
            tags=[],
            created_at="2025-01-15T10:30:00+00:00",
            updated_at="2025-01-15T10:30:00+00:00",
        )
        piece_b = KnowledgePiece(
            content="B content",
            piece_id="bbb",
            knowledge_type=KnowledgeType.Fact,
            tags=[],
            created_at="2025-01-15T10:30:00+00:00",
            updated_at="2025-01-15T10:30:00+00:00",
        )
        # Same score, so sorted by piece_id ascending
        result = RetrievalResult(pieces=[(piece_b, 0.5), (piece_a, 0.5)])
        output = formatter.format(result)
        a_pos = output.index("A content")
        b_pos = output.index("B content")
        assert a_pos < b_pos

    def test_pieces_higher_score_first(self, formatter):
        """Higher scored pieces appear before lower scored ones."""
        piece_low = KnowledgePiece(
            content="Low score",
            piece_id="p1",
            knowledge_type=KnowledgeType.Fact,
            tags=[],
            created_at="2025-01-15T10:30:00+00:00",
            updated_at="2025-01-15T10:30:00+00:00",
        )
        piece_high = KnowledgePiece(
            content="High score",
            piece_id="p2",
            knowledge_type=KnowledgeType.Fact,
            tags=[],
            created_at="2025-01-15T10:30:00+00:00",
            updated_at="2025-01-15T10:30:00+00:00",
        )
        result = RetrievalResult(pieces=[(piece_low, 0.3), (piece_high, 0.9)])
        output = formatter.format(result)
        high_pos = output.index("High score")
        low_pos = output.index("Low score")
        assert high_pos < low_pos

    def test_piece_with_empty_tags_no_tag_line(self, formatter):
        """Piece with no tags doesn't show Tags line even when include_tags is True."""
        piece = KnowledgePiece(
            content="No tags here",
            piece_id="p1",
            knowledge_type=KnowledgeType.Fact,
            tags=[],
            created_at="2025-01-15T10:30:00+00:00",
            updated_at="2025-01-15T10:30:00+00:00",
        )
        result = RetrievalResult(pieces=[(piece, 0.9)])
        output = formatter.format(result)
        assert "Tags:" not in output

    def test_tags_sorted_alphabetically(self, formatter):
        """Tags within a piece are sorted alphabetically."""
        piece = KnowledgePiece(
            content="Test",
            piece_id="p1",
            knowledge_type=KnowledgeType.Fact,
            tags=["zebra", "apple", "mango"],
            created_at="2025-01-15T10:30:00+00:00",
            updated_at="2025-01-15T10:30:00+00:00",
        )
        result = RetrievalResult(pieces=[(piece, 0.9)])
        output = formatter.format(result)
        assert "Tags: apple, mango, zebra" in output

    def test_item_delimiter_between_pieces(self, formatter):
        """Items within Knowledge section are separated by item_delimiter."""
        piece1 = KnowledgePiece(
            content="First",
            piece_id="p1",
            knowledge_type=KnowledgeType.Fact,
            tags=[],
            created_at="2025-01-15T10:30:00+00:00",
            updated_at="2025-01-15T10:30:00+00:00",
        )
        piece2 = KnowledgePiece(
            content="Second",
            piece_id="p2",
            knowledge_type=KnowledgeType.Fact,
            tags=[],
            created_at="2025-01-15T10:30:00+00:00",
            updated_at="2025-01-15T10:30:00+00:00",
        )
        result = RetrievalResult(pieces=[(piece1, 0.9), (piece2, 0.8)])
        output = formatter.format(result)
        assert "\n---\n" in output


class TestFormatterGraphContext:
    """Tests for graph context formatting."""

    def test_graph_context_sorted(self, formatter, sample_graph_context):
        """Graph context entries are sorted by relation_type then target_node_id."""
        result = RetrievalResult(graph_context=sample_graph_context)
        output = formatter.format(result)
        lines = output.strip().split("\n")
        # Header + 3 entries
        assert lines[0] == "[Relationships]"
        assert "Store:costco" in lines[1]
        assert "Store:qfc" in lines[2]
        assert "Store:whole_foods" in lines[3]

    def test_graph_context_with_piece(self, formatter):
        """Graph entry with linked piece shows piece content in parentheses."""
        ctx = [
            {
                "relation_type": "SHOPS_AT",
                "target_node_id": "store:costco",
                "target_label": "Costco",
                "piece": KnowledgePiece(
                    content="Executive membership",
                    piece_id="gp1",
                    created_at="2025-01-15T10:30:00+00:00",
                    updated_at="2025-01-15T10:30:00+00:00",
                ),
                "depth": 1,
            }
        ]
        result = RetrievalResult(graph_context=ctx)
        output = formatter.format(result)
        assert "SHOPS_AT → Store:costco (Executive membership)" in output

    def test_graph_context_without_piece_with_label(self, formatter):
        """Graph entry without piece but with label shows label in parentheses."""
        ctx = [
            {
                "relation_type": "SHOPS_AT",
                "target_node_id": "store:qfc",
                "target_label": "QFC Grocery",
                "piece": None,
                "depth": 1,
            }
        ]
        result = RetrievalResult(graph_context=ctx)
        output = formatter.format(result)
        assert "SHOPS_AT → Store:qfc (QFC Grocery)" in output

    def test_graph_context_without_piece_or_label(self, formatter):
        """Graph entry without piece or label shows just the relation and target."""
        ctx = [
            {
                "relation_type": "SHOPS_AT",
                "target_node_id": "store:qfc",
                "target_label": "",
                "piece": None,
                "depth": 1,
            }
        ]
        result = RetrievalResult(graph_context=ctx)
        output = formatter.format(result)
        assert "SHOPS_AT → Store:qfc" in output
        assert "(" not in output.split("\n")[1]  # No parentheses

    def test_graph_context_different_relation_types(self, formatter):
        """Different relation types are sorted alphabetically."""
        ctx = [
            {
                "relation_type": "SHOPS_AT",
                "target_node_id": "store:costco",
                "target_label": "",
                "piece": None,
                "depth": 1,
            },
            {
                "relation_type": "LIVES_IN",
                "target_node_id": "city:seattle",
                "target_label": "",
                "piece": None,
                "depth": 1,
            },
        ]
        result = RetrievalResult(graph_context=ctx)
        output = formatter.format(result)
        lines = output.strip().split("\n")
        assert "LIVES_IN" in lines[1]
        assert "SHOPS_AT" in lines[2]


class TestFormatterCombinedSections:
    """Tests for combined output with multiple sections."""

    def test_full_output_matches_example(self, formatter, sample_metadata, sample_pieces, sample_graph_context):
        """Full output matches the design document example format."""
        result = RetrievalResult(
            metadata=sample_metadata,
            pieces=sample_pieces,
            graph_context=sample_graph_context,
        )
        output = formatter.format(result)

        # Check all sections present
        assert "[Metadata]" in output
        assert "[Knowledge]" in output
        assert "[Relationships]" in output

        # Check metadata sorted
        assert "location: Seattle" in output
        assert "name: Xinli" in output
        assert "zip: 98121" in output

        # Check pieces present with types
        assert "[preference]" in output
        assert "[instruction]" in output

        # Check relationships present
        assert "SHOPS_AT → Store:costco" in output

    def test_section_delimiter_used(self):
        """Custom section delimiter is used between sections."""
        formatter = KnowledgeFormatter(section_delimiter="\n===\n")
        metadata = EntityMetadata(
            entity_id="user:test",
            entity_type="user",
            properties={"name": "Test"},
        )
        piece = KnowledgePiece(
            content="Test piece",
            piece_id="p1",
            knowledge_type=KnowledgeType.Fact,
            tags=[],
            created_at="2025-01-15T10:30:00+00:00",
            updated_at="2025-01-15T10:30:00+00:00",
        )
        result = RetrievalResult(metadata=metadata, pieces=[(piece, 0.9)])
        output = formatter.format(result)
        assert "\n===\n" in output

    def test_custom_item_delimiter(self):
        """Custom item delimiter is used between pieces."""
        formatter = KnowledgeFormatter(item_delimiter="\n***\n")
        piece1 = KnowledgePiece(
            content="First",
            piece_id="p1",
            knowledge_type=KnowledgeType.Fact,
            tags=[],
            created_at="2025-01-15T10:30:00+00:00",
            updated_at="2025-01-15T10:30:00+00:00",
        )
        piece2 = KnowledgePiece(
            content="Second",
            piece_id="p2",
            knowledge_type=KnowledgeType.Fact,
            tags=[],
            created_at="2025-01-15T10:30:00+00:00",
            updated_at="2025-01-15T10:30:00+00:00",
        )
        result = RetrievalResult(pieces=[(piece1, 0.9), (piece2, 0.8)])
        output = formatter.format(result)
        assert "\n***\n" in output


class TestFormatterDeterminism:
    """Tests for deterministic output."""

    def test_same_input_same_output(self, formatter, sample_metadata, sample_pieces, sample_graph_context):
        """Formatting the same result multiple times produces identical output."""
        result = RetrievalResult(
            metadata=sample_metadata,
            pieces=sample_pieces,
            graph_context=sample_graph_context,
        )
        output1 = formatter.format(result)
        output2 = formatter.format(result)
        output3 = formatter.format(result)
        assert output1 == output2 == output3

    def test_different_piece_order_same_output(self, formatter):
        """Pieces in different order produce the same output."""
        piece_a = KnowledgePiece(
            content="Alpha",
            piece_id="p-alpha",
            knowledge_type=KnowledgeType.Fact,
            tags=["tag1"],
            created_at="2025-01-15T10:30:00+00:00",
            updated_at="2025-01-15T10:30:00+00:00",
        )
        piece_b = KnowledgePiece(
            content="Beta",
            piece_id="p-beta",
            knowledge_type=KnowledgeType.Fact,
            tags=["tag2"],
            created_at="2025-01-15T10:30:00+00:00",
            updated_at="2025-01-15T10:30:00+00:00",
        )
        result1 = RetrievalResult(pieces=[(piece_a, 0.9), (piece_b, 0.8)])
        result2 = RetrievalResult(pieces=[(piece_b, 0.8), (piece_a, 0.9)])
        assert formatter.format(result1) == formatter.format(result2)

    def test_different_graph_context_order_same_output(self, formatter):
        """Graph context in different order produces the same output."""
        ctx1 = {
            "relation_type": "SHOPS_AT",
            "target_node_id": "store:costco",
            "target_label": "",
            "piece": None,
            "depth": 1,
        }
        ctx2 = {
            "relation_type": "LIVES_IN",
            "target_node_id": "city:seattle",
            "target_label": "",
            "piece": None,
            "depth": 1,
        }
        result1 = RetrievalResult(graph_context=[ctx1, ctx2])
        result2 = RetrievalResult(graph_context=[ctx2, ctx1])
        assert formatter.format(result1) == formatter.format(result2)
