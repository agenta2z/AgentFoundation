"""
Property-based tests for KnowledgeFormatter.

# Feature: agent-knowledge-base, Property 16: Formatter output determinism
# Feature: agent-knowledge-base, Property 17: Formatter output structure
# **Validates: Requirements 10.1, 10.2, 10.3, 10.5**
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

import random
import copy

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from science_modeling_tools.knowledge.formatter import (
    KnowledgeFormatter,
    RetrievalResult,
)
from science_modeling_tools.knowledge.models.entity_metadata import EntityMetadata
from science_modeling_tools.knowledge.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)

# Import strategies from conftest
_test_dir = Path(__file__).resolve().parent
if str(_test_dir) not in sys.path:
    sys.path.insert(0, str(_test_dir))
from conftest import knowledge_piece_strategy, entity_metadata_strategy


# ── Strategies ───────────────────────────────────────────────────────────────

_score_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

_identifier_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=50,
)


@st.composite
def scored_piece_strategy(draw):
    """Generate a (KnowledgePiece, score) tuple."""
    piece = draw(knowledge_piece_strategy())
    score = draw(_score_strategy)
    return (piece, score)


@st.composite
def graph_context_entry_strategy(draw):
    """Generate a graph context dict matching the expected format."""
    relation_type = draw(_identifier_text)
    # target_node_id uses type:name format
    node_type = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=1,
        max_size=15,
    ))
    node_name = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=1,
        max_size=15,
    ))
    target_node_id = f"{node_type}:{node_name}"
    target_label = draw(st.text(max_size=30))
    # Optionally include a linked piece
    include_piece = draw(st.booleans())
    piece = draw(knowledge_piece_strategy()) if include_piece else None
    depth = draw(st.integers(min_value=1, max_value=5))

    return {
        "relation_type": relation_type,
        "target_node_id": target_node_id,
        "target_label": target_label,
        "piece": piece,
        "depth": depth,
    }


@st.composite
def retrieval_result_strategy(draw):
    """Generate a random RetrievalResult with optional data in all layers."""
    include_metadata = draw(st.booleans())
    include_global_metadata = draw(st.booleans())
    metadata = draw(entity_metadata_strategy()) if include_metadata else None
    global_metadata = draw(entity_metadata_strategy()) if include_global_metadata else None
    pieces = draw(st.lists(scored_piece_strategy(), max_size=8))
    graph_context = draw(st.lists(graph_context_entry_strategy(), max_size=5))

    return RetrievalResult(
        metadata=metadata,
        global_metadata=global_metadata,
        pieces=pieces,
        graph_context=graph_context,
    )


@st.composite
def mixed_type_scored_pieces_strategy(draw):
    """Generate a list of scored pieces with at least 2 different KnowledgeType values.

    This ensures we can test grouping behavior across types.
    """
    # Pick at least 2 distinct knowledge types
    all_types = list(KnowledgeType)
    types = draw(
        st.lists(
            st.sampled_from(all_types),
            min_size=2,
            max_size=len(all_types),
            unique=True,
        )
    )

    pieces = []
    for kt in types:
        # Generate at least 1 piece per type
        num_pieces = draw(st.integers(min_value=1, max_value=3))
        for _ in range(num_pieces):
            piece = draw(knowledge_piece_strategy())
            # Override the knowledge_type to ensure mixed types
            piece = KnowledgePiece(
                content=piece.content,
                piece_id=piece.piece_id,
                knowledge_type=kt,
                tags=piece.tags,
                entity_id=piece.entity_id,
                source=piece.source,
                embedding_text=piece.embedding_text,
                created_at=piece.created_at,
                updated_at=piece.updated_at,
            )
            score = draw(_score_strategy)
            pieces.append((piece, score))

    return pieces


# ── Property 16: Formatter output determinism ────────────────────────────────


class TestFormatterOutputDeterminism:
    """Property 16: Formatter output determinism.

    For any RetrievalResult, formatting it multiple times SHALL produce the
    same output string. Additionally, for any set of (KnowledgePiece, score)
    tuples, formatting them in any permutation SHALL produce the same output
    string.

    **Validates: Requirements 10.5**
    """

    # Feature: agent-knowledge-base, Property 16: Formatter output determinism

    @given(result=retrieval_result_strategy())
    @settings(max_examples=200)
    def test_formatting_same_result_multiple_times_is_idempotent(self, result):
        """Formatting the same RetrievalResult multiple times produces identical output.

        **Validates: Requirements 10.5**
        """
        formatter = KnowledgeFormatter()
        output1 = formatter.format(result)
        output2 = formatter.format(result)
        output3 = formatter.format(result)
        assert output1 == output2, "First and second format calls differ"
        assert output2 == output3, "Second and third format calls differ"

    @given(pieces=st.lists(scored_piece_strategy(), min_size=2, max_size=10))
    @settings(max_examples=200)
    def test_piece_permutation_produces_same_output(self, pieces):
        """Formatting pieces in any permutation produces the same output string.

        Pieces with unique piece_ids (the realistic case — IDs are auto-generated
        UUIDs) should produce identical output regardless of input order.

        **Validates: Requirements 10.5**
        """
        # Ensure unique piece_ids so the sort key (-score, piece_id) is unambiguous
        seen_ids = set()
        unique_pieces = []
        for piece, score in pieces:
            if piece.piece_id not in seen_ids:
                seen_ids.add(piece.piece_id)
                unique_pieces.append((piece, score))
        assume(len(unique_pieces) >= 2)
        pieces = unique_pieces

        formatter = KnowledgeFormatter()

        # Original order
        result_original = RetrievalResult(pieces=list(pieces))
        output_original = formatter.format(result_original)

        # Reversed order
        result_reversed = RetrievalResult(pieces=list(reversed(pieces)))
        output_reversed = formatter.format(result_reversed)

        assert output_original == output_reversed, (
            "Reversed piece order produced different output"
        )

        # Random shuffle
        shuffled = list(pieces)
        random.shuffle(shuffled)
        result_shuffled = RetrievalResult(pieces=shuffled)
        output_shuffled = formatter.format(result_shuffled)

        assert output_original == output_shuffled, (
            "Shuffled piece order produced different output"
        )

    @given(graph_context=st.lists(graph_context_entry_strategy(), min_size=2, max_size=8))
    @settings(max_examples=200)
    def test_graph_context_permutation_produces_same_output(self, graph_context):
        """Formatting graph_context in any permutation produces the same output string.

        Graph context entries with unique (relation_type, target_node_id) sort keys
        should produce identical output regardless of input order.

        **Validates: Requirements 10.5**
        """
        # Ensure unique sort keys so ordering is unambiguous
        seen_keys = set()
        unique_ctx = []
        for ctx in graph_context:
            key = (ctx.get("relation_type", ""), ctx.get("target_node_id", ""))
            if key not in seen_keys:
                seen_keys.add(key)
                unique_ctx.append(ctx)
        assume(len(unique_ctx) >= 2)
        graph_context = unique_ctx

        formatter = KnowledgeFormatter()

        # Original order
        result_original = RetrievalResult(graph_context=list(graph_context))
        output_original = formatter.format(result_original)

        # Reversed order
        result_reversed = RetrievalResult(graph_context=list(reversed(graph_context)))
        output_reversed = formatter.format(result_reversed)

        assert output_original == output_reversed, (
            "Reversed graph_context order produced different output"
        )

        # Random shuffle
        shuffled = list(graph_context)
        random.shuffle(shuffled)
        result_shuffled = RetrievalResult(graph_context=shuffled)
        output_shuffled = formatter.format(result_shuffled)

        assert output_original == output_shuffled, (
            "Shuffled graph_context order produced different output"
        )


# ── Property 17: Formatter output structure ──────────────────────────────────


class TestFormatterOutputStructure:
    """Property 17: Formatter output structure.

    For any non-empty RetrievalResult with pieces of mixed KnowledgeType values,
    the formatted output SHALL contain each piece's content, pieces SHALL be
    grouped by type, and groups SHALL be separated by the configured delimiter.

    **Validates: Requirements 10.1, 10.2, 10.3**
    """

    # Feature: agent-knowledge-base, Property 17: Formatter output structure

    @given(pieces=mixed_type_scored_pieces_strategy())
    @settings(max_examples=200)
    def test_each_piece_content_appears_in_output(self, pieces):
        """Every piece's content appears in the formatted output.

        **Validates: Requirements 10.2**
        """
        formatter = KnowledgeFormatter()
        result = RetrievalResult(pieces=pieces)
        output = formatter.format(result)

        for piece, _score in pieces:
            assert piece.content in output, (
                f"Piece content {piece.content!r} not found in output"
            )

    @given(pieces=mixed_type_scored_pieces_strategy())
    @settings(max_examples=200)
    def test_pieces_grouped_by_type(self, pieces):
        """Pieces are grouped by KnowledgeType — all pieces of the same type
        appear contiguously in the output.

        **Validates: Requirements 10.1**
        """
        formatter = KnowledgeFormatter()
        result = RetrievalResult(pieces=pieces)
        output = formatter.format(result)

        # Extract the knowledge section (after [Knowledge] header)
        assert "[Knowledge]" in output
        # Split on first occurrence only to avoid issues with content containing "[Knowledge]"
        knowledge_section = output.split("[Knowledge]", 1)[1]

        # For each KnowledgeType present, find the positions of all pieces of that type
        types_present = set(piece.knowledge_type for piece, _ in pieces)

        for kt in types_present:
            type_tag = f"[{kt.value}]"
            # Find all positions of this type tag in the knowledge section
            positions = []
            start = 0
            while True:
                idx = knowledge_section.find(type_tag, start)
                if idx == -1:
                    break
                positions.append(idx)
                start = idx + 1

            if len(positions) < 2:
                continue

            # All occurrences of this type tag should be contiguous —
            # no other type tag should appear between the first and last occurrence
            first_pos = positions[0]
            last_pos = positions[-1]
            between = knowledge_section[first_pos:last_pos]

            for other_kt in types_present:
                if other_kt == kt:
                    continue
                other_tag = f"[{other_kt.value}]"
                assert other_tag not in between, (
                    f"Type [{kt.value}] pieces are not contiguous: "
                    f"found [{other_kt.value}] between first and last [{kt.value}]"
                )

    @given(pieces=mixed_type_scored_pieces_strategy())
    @settings(max_examples=200)
    def test_groups_separated_by_delimiter(self, pieces):
        """Groups of different KnowledgeType are separated by the configured
        item_delimiter.

        **Validates: Requirements 10.3**
        """
        item_delimiter = "\n---\n"
        formatter = KnowledgeFormatter(item_delimiter=item_delimiter)
        result = RetrievalResult(pieces=pieces)
        output = formatter.format(result)

        # If there are multiple pieces, the delimiter should appear between them
        if len(pieces) > 1:
            assert item_delimiter in output, (
                "Item delimiter not found in output with multiple pieces"
            )

    @given(pieces=mixed_type_scored_pieces_strategy())
    @settings(max_examples=200)
    def test_type_groups_sorted_alphabetically(self, pieces):
        """KnowledgeType groups appear in alphabetical order by type value.

        **Validates: Requirements 10.1**
        """
        formatter = KnowledgeFormatter()
        result = RetrievalResult(pieces=pieces)
        output = formatter.format(result)

        # Extract the knowledge section (split on first occurrence only)
        knowledge_section = output.split("[Knowledge]", 1)[1]

        # Find the first occurrence position of each type tag in the knowledge section.
        # We look for the type tag at the start of a line (after newline) to avoid
        # matching content that happens to contain the tag string.
        types_present = sorted(
            set(piece.knowledge_type for piece, _ in pieces),
            key=lambda kt: kt.value,
        )

        first_positions = []
        for kt in types_present:
            type_tag = f"[{kt.value}]"
            # Search for the tag preceded by a newline (formatter puts each piece on a new line)
            search_tag = f"\n{type_tag}"
            pos = knowledge_section.find(search_tag)
            assert pos != -1, f"Type tag {type_tag} not found in knowledge section"
            first_positions.append(pos)

        # Positions should be in ascending order (alphabetical by type value)
        assert first_positions == sorted(first_positions), (
            f"Type groups not in alphabetical order. "
            f"Types: {[kt.value for kt in types_present]}, "
            f"Positions: {first_positions}"
        )
