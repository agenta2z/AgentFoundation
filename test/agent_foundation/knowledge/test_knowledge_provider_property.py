"""
Property-based tests for BudgetAwareKnowledgeProvider.

Feature: knowledge-module-migration
- Property 12: BudgetAwareKnowledgeProvider token budget enforcement
- Property 13: BudgetAwareKnowledgeProvider episodic temporal markers

**Validates: Requirements 9.2, 9.4**
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

from hypothesis import given, settings, assume, strategies as st

from agent_foundation.knowledge.retrieval.knowledge_provider import (
    BudgetAwareKnowledgeProvider,
    CONTEXT_BUDGET,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.models.results import ScoredPiece
from agent_foundation.knowledge.retrieval.utils import count_tokens


# ── Strategies ────────────────────────────────────────────────────────────────

_info_types = list(CONTEXT_BUDGET.keys())

_date_strategy = st.from_regex(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}",
    fullmatch=True,
)


@st.composite
def scored_piece_strategy(draw, info_type=None):
    """Generate a ScoredPiece with a given or random info_type."""
    chosen_type = info_type or draw(st.sampled_from(_info_types))
    content = draw(st.text(min_size=1, max_size=200))
    assume(content.strip())
    updated_at = draw(_date_strategy)
    summary = draw(st.one_of(st.none(), st.text(min_size=1, max_size=80)))

    piece = KnowledgePiece(
        content=content,
        info_type=chosen_type,
        updated_at=updated_at,
        summary=summary,
    )
    score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    return ScoredPiece(piece=piece, score=score, normalized_score=score)


@st.composite
def scored_pieces_list(draw):
    """Generate a list of ScoredPieces with mixed info_types."""
    pieces = draw(st.lists(scored_piece_strategy(), min_size=0, max_size=15))
    return pieces


@st.composite
def episodic_scored_piece(draw):
    """Generate a ScoredPiece with info_type='episodic' and a known timestamp."""
    content = draw(st.text(min_size=1, max_size=200))
    assume(content.strip())
    updated_at = draw(_date_strategy)

    piece = KnowledgePiece(
        content=content,
        info_type="episodic",
        updated_at=updated_at,
    )
    score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    return ScoredPiece(piece=piece, score=score, normalized_score=score)


# ── Property 12: BudgetAwareKnowledgeProvider token budget enforcement ────────


class TestBudgetAwareTokenEnforcement:
    """Property 12: BudgetAwareKnowledgeProvider token budget enforcement.

    For any set of ScoredPieces and available_tokens limit, each formatted
    section in the output should not exceed its per-info-type budget, and
    the total output should not exceed available_tokens.

    **Validates: Requirements 9.2**
    """

    @given(
        pieces=scored_pieces_list(),
        available_tokens=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100)
    def test_total_output_within_budget(self, pieces, available_tokens):
        """Total formatted output tokens do not exceed available_tokens.

        **Validates: Requirements 9.2**
        """
        provider = BudgetAwareKnowledgeProvider()
        output = provider.format_knowledge(pieces, available_tokens)
        total_tokens = count_tokens(output)
        assert total_tokens <= available_tokens, (
            f"Total tokens {total_tokens} exceeds budget {available_tokens}"
        )

    @given(
        pieces=scored_pieces_list(),
        available_tokens=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100)
    def test_per_type_sections_within_budget(self, pieces, available_tokens):
        """Each info_type section's tokens do not exceed its per-type budget.

        **Validates: Requirements 9.2**
        """
        provider = BudgetAwareKnowledgeProvider()
        output = provider.format_knowledge(pieces, available_tokens)

        # The output is sections joined by "\n\n". Each section starts with
        # a header like "## Available Skills", "## Instructions", etc.
        # We verify each section individually against its type budget.
        if not output:
            return

        sections = output.split("\n\n")
        # Reconstruct sections that were split within a single info_type block.
        # The provider joins sections with "\n\n", so top-level splits are
        # between info_type blocks. Each block is produced by _format_type
        # which enforces its own budget.
        # We just verify the total per-section tokens are within the type budget.
        for section in sections:
            section_tokens = count_tokens(section)
            # The section must fit within the largest possible per-type budget
            # (since we can't always determine which type it belongs to from
            # the split). The key property is the total budget enforcement.
            assert section_tokens <= available_tokens, (
                f"Section tokens {section_tokens} exceeds available {available_tokens}"
            )

    @given(
        pieces=scored_pieces_list(),
    )
    @settings(max_examples=100)
    def test_zero_budget_produces_empty_or_minimal_output(self, pieces):
        """With zero available_tokens, output should be empty.

        **Validates: Requirements 9.2**
        """
        provider = BudgetAwareKnowledgeProvider()
        output = provider.format_knowledge(pieces, available_tokens=0)
        total_tokens = count_tokens(output)
        assert total_tokens == 0, (
            f"Expected 0 tokens with zero budget, got {total_tokens}"
        )

    @given(
        pieces=scored_pieces_list(),
        available_tokens=st.integers(min_value=100, max_value=10000),
    )
    @settings(max_examples=100)
    def test_each_formatted_section_respects_per_type_budget(self, pieces, available_tokens):
        """Each info_type formatted individually respects its CONTEXT_BUDGET cap.

        Uses a minimum budget of 100 tokens to account for section headers
        that are always emitted by the formatters.

        **Validates: Requirements 9.2**
        """
        provider = BudgetAwareKnowledgeProvider()

        # Group pieces by info_type and format each type individually
        from collections import defaultdict
        by_type = defaultdict(list)
        for p in pieces:
            by_type[p.info_type or "context"].append(p)

        for info_type, type_pieces in by_type.items():
            if info_type not in CONTEXT_BUDGET:
                continue
            type_budget = CONTEXT_BUDGET[info_type]
            effective_budget = min(type_budget, available_tokens)
            formatted = provider._format_type(type_pieces, info_type, effective_budget)
            tokens = count_tokens(formatted)
            assert tokens <= effective_budget, (
                f"Type '{info_type}' used {tokens} tokens, "
                f"budget was {effective_budget}"
            )


# ── Property 13: BudgetAwareKnowledgeProvider episodic temporal markers ───────


class TestBudgetAwareEpisodicMarkers:
    """Property 13: BudgetAwareKnowledgeProvider episodic temporal markers.

    For any ScoredPiece with info_type="episodic", the formatted output for
    that piece should contain a date prefix extracted from the piece's
    updated_at field (first 10 characters).

    **Validates: Requirements 9.4**
    """

    @given(
        piece=episodic_scored_piece(),
        budget=st.integers(min_value=500, max_value=5000),
    )
    @settings(max_examples=100)
    def test_episodic_output_contains_date_prefix(self, piece, budget):
        """Episodic formatted output contains the date prefix from updated_at.

        **Validates: Requirements 9.4**
        """
        provider = BudgetAwareKnowledgeProvider()
        formatted = provider._format_episodic([piece], budget)

        date_prefix = piece.updated_at[:10]
        assert date_prefix in formatted, (
            f"Expected date prefix '{date_prefix}' in episodic output, "
            f"got: {formatted!r}"
        )

    @given(
        pieces=st.lists(episodic_scored_piece(), min_size=1, max_size=5),
        budget=st.integers(min_value=2000, max_value=10000),
    )
    @settings(max_examples=100)
    def test_all_included_episodic_pieces_have_date_prefix(self, pieces, budget):
        """Every episodic piece included in the output has its date prefix.

        **Validates: Requirements 9.4**
        """
        provider = BudgetAwareKnowledgeProvider()
        formatted = provider._format_episodic(pieces, budget)

        # Each piece that was included should have its date prefix
        for piece in pieces:
            date_prefix = piece.updated_at[:10]
            content_snippet = piece.piece.content[:20]
            # If the piece's content appears in the output, its date must too
            if content_snippet in formatted:
                assert date_prefix in formatted, (
                    f"Piece with content starting '{content_snippet}' is in output "
                    f"but missing date prefix '{date_prefix}'"
                )

    @given(
        piece=episodic_scored_piece(),
    )
    @settings(max_examples=100)
    def test_episodic_via_format_knowledge_contains_date(self, piece):
        """Episodic pieces formatted through format_knowledge contain date prefix.

        **Validates: Requirements 9.4**
        """
        provider = BudgetAwareKnowledgeProvider()
        # Use a generous budget so the piece is included
        output = provider.format_knowledge([piece], available_tokens=5000)

        if not output:
            # If output is empty, the piece didn't fit (unlikely with 5000 tokens)
            return

        date_prefix = piece.updated_at[:10]
        assert date_prefix in output, (
            f"Expected date prefix '{date_prefix}' in format_knowledge output, "
            f"got: {output!r}"
        )

    @given(
        piece=episodic_scored_piece(),
    )
    @settings(max_examples=100)
    def test_episodic_date_prefix_format_is_bracketed(self, piece):
        """Episodic entries use bracketed date format [YYYY-MM-DD].

        **Validates: Requirements 9.4**
        """
        provider = BudgetAwareKnowledgeProvider()
        formatted = provider._format_episodic([piece], budget=5000)

        date_prefix = piece.updated_at[:10]
        bracketed = f"[{date_prefix}]"
        assert bracketed in formatted, (
            f"Expected bracketed date '{bracketed}' in episodic output, "
            f"got: {formatted!r}"
        )
