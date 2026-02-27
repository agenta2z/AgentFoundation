"""
Property-based tests for the three-tier deduplicator module.

Feature: knowledge-module-migration
- Property 17: Deduplicator threshold decision boundaries
- Property 18: Deduplicator Tier 1 exact hash match

**Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5**
"""
import sys
from pathlib import Path
from typing import List, Optional, Tuple

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

from agent_foundation.knowledge.ingestion.deduplicator import (
    DedupConfig,
    ThreeTierDeduplicator,
)
from agent_foundation.knowledge.retrieval.models.enums import DedupAction
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.models.results import DedupResult
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore


# ── Test Helpers ──────────────────────────────────────────────────────────────


class InMemoryPieceStore(KnowledgePieceStore):
    """Minimal in-memory store for property testing with controllable search scores."""

    def __init__(self, pieces: Optional[List[KnowledgePiece]] = None, score: float = 0.5):
        self._pieces: dict[str, KnowledgePiece] = {}
        self._score = score
        for p in (pieces or []):
            self._pieces[p.piece_id] = p

    def add(self, piece: KnowledgePiece) -> str:
        self._pieces[piece.piece_id] = piece
        return piece.piece_id

    def get_by_id(self, piece_id: str) -> Optional[KnowledgePiece]:
        return self._pieces.get(piece_id)

    def update(self, piece: KnowledgePiece) -> bool:
        if piece.piece_id in self._pieces:
            self._pieces[piece.piece_id] = piece
            return True
        return False

    def remove(self, piece_id: str) -> bool:
        return self._pieces.pop(piece_id, None) is not None

    def search(
        self, query, entity_id=None, knowledge_type=None, tags=None, top_k=5
    ) -> List[Tuple[KnowledgePiece, float]]:
        """Return all pieces with the configured score."""
        results = []
        for p in self._pieces.values():
            if entity_id is not None and p.entity_id != entity_id:
                continue
            results.append((p, self._score))
        return results[:top_k]

    def list_all(self, entity_id=None, knowledge_type=None) -> List[KnowledgePiece]:
        results = []
        for p in self._pieces.values():
            if entity_id is not None and p.entity_id != entity_id:
                continue
            if knowledge_type is not None and p.knowledge_type != knowledge_type:
                continue
            results.append(p)
        return results


def _dummy_embedding_fn(text: str) -> List[float]:
    """Dummy embedding function for testing."""
    return [0.1, 0.2, 0.3]


# ── Strategies ────────────────────────────────────────────────────────────────

# Content strings for generating pieces
_content_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=5,
    max_size=200,
)

# Thresholds: auto_dedup > llm_judge, both in (0, 1)
_threshold_pair_strategy = st.tuples(
    st.floats(min_value=0.01, max_value=0.99),
    st.floats(min_value=0.01, max_value=0.99),
).filter(lambda t: t[0] > t[1]).map(
    lambda t: (t[0], t[1])  # (auto_dedup_threshold, llm_judge_threshold)
)

# Score above auto_dedup_threshold
_score_above_auto_dedup = st.floats(min_value=0.981, max_value=1.0, allow_nan=False, allow_infinity=False)

# Score below llm_judge_threshold
_score_below_llm_judge = st.floats(min_value=0.0, max_value=0.849, allow_nan=False, allow_infinity=False)

# Score between thresholds (borderline)
_score_borderline = st.floats(min_value=0.851, max_value=0.979, allow_nan=False, allow_infinity=False)


# Feature: knowledge-module-migration, Property 17: Deduplicator threshold decision boundaries


class TestDeduplicatorThresholdDecisionBoundaries:
    """Property 17: Deduplicator threshold decision boundaries.

    For any piece and similarity score from Tier 2: if score > auto_dedup_threshold,
    the action should be NO_OP; if score < llm_judge_threshold, the action should be
    ADD; if score is between the thresholds, Tier 3 should be invoked (or ADD if
    Tier 3 is disabled).

    **Validates: Requirements 12.2, 12.3, 12.4, 12.5**
    """

    @given(
        content=_content_strategy,
        auto_dedup_threshold=st.floats(min_value=0.90, max_value=0.99, allow_nan=False, allow_infinity=False),
        llm_judge_threshold=st.floats(min_value=0.50, max_value=0.89, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_score_above_auto_dedup_returns_no_op(
        self, content: str, auto_dedup_threshold: float, llm_judge_threshold: float
    ):
        """When Tier 2 score > auto_dedup_threshold, action should be NO_OP.

        **Validates: Requirements 12.4**
        """
        assume(auto_dedup_threshold > llm_judge_threshold)
        assume(len(content.strip()) > 0)

        # Use a score strictly above the auto_dedup_threshold
        score = min(auto_dedup_threshold + 0.005, 1.0)

        existing_piece = KnowledgePiece(content="existing content")
        store = InMemoryPieceStore(pieces=[existing_piece], score=score)

        config = DedupConfig(
            auto_dedup_threshold=auto_dedup_threshold,
            llm_judge_threshold=llm_judge_threshold,
            enable_tier1=False,  # Disable Tier 1 to isolate Tier 2
            enable_tier2=True,
            enable_tier3=False,
        )

        deduplicator = ThreeTierDeduplicator(
            piece_store=store,
            embedding_fn=_dummy_embedding_fn,
            config=config,
        )

        new_piece = KnowledgePiece(content=content)
        result = deduplicator.deduplicate(new_piece)

        assert result.action == DedupAction.NO_OP
        assert result.existing_piece_id == existing_piece.piece_id

    @given(
        content=_content_strategy,
        auto_dedup_threshold=st.floats(min_value=0.90, max_value=0.99, allow_nan=False, allow_infinity=False),
        llm_judge_threshold=st.floats(min_value=0.50, max_value=0.89, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_score_below_llm_judge_returns_add(
        self, content: str, auto_dedup_threshold: float, llm_judge_threshold: float
    ):
        """When Tier 2 score < llm_judge_threshold, action should be ADD.

        **Validates: Requirements 12.6**
        """
        assume(auto_dedup_threshold > llm_judge_threshold)
        assume(len(content.strip()) > 0)

        # Use a score strictly below the llm_judge_threshold
        score = max(llm_judge_threshold - 0.005, 0.0)

        existing_piece = KnowledgePiece(content="existing content")
        store = InMemoryPieceStore(pieces=[existing_piece], score=score)

        config = DedupConfig(
            auto_dedup_threshold=auto_dedup_threshold,
            llm_judge_threshold=llm_judge_threshold,
            enable_tier1=False,
            enable_tier2=True,
            enable_tier3=False,
        )

        deduplicator = ThreeTierDeduplicator(
            piece_store=store,
            embedding_fn=_dummy_embedding_fn,
            config=config,
        )

        new_piece = KnowledgePiece(content=content)
        result = deduplicator.deduplicate(new_piece)

        assert result.action == DedupAction.ADD

    @given(
        content=_content_strategy,
        auto_dedup_threshold=st.floats(min_value=0.90, max_value=0.99, allow_nan=False, allow_infinity=False),
        llm_judge_threshold=st.floats(min_value=0.50, max_value=0.89, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_borderline_score_with_tier3_disabled_returns_add(
        self, content: str, auto_dedup_threshold: float, llm_judge_threshold: float
    ):
        """When score is between thresholds and Tier 3 is disabled, action should be ADD.

        **Validates: Requirements 12.5**
        """
        assume(auto_dedup_threshold > llm_judge_threshold)
        assume(len(content.strip()) > 0)
        assume(auto_dedup_threshold - llm_judge_threshold > 0.02)

        # Score between the two thresholds
        score = (auto_dedup_threshold + llm_judge_threshold) / 2.0

        existing_piece = KnowledgePiece(content="existing content")
        store = InMemoryPieceStore(pieces=[existing_piece], score=score)

        config = DedupConfig(
            auto_dedup_threshold=auto_dedup_threshold,
            llm_judge_threshold=llm_judge_threshold,
            enable_tier1=False,
            enable_tier2=True,
            enable_tier3=False,  # Tier 3 disabled
        )

        deduplicator = ThreeTierDeduplicator(
            piece_store=store,
            embedding_fn=_dummy_embedding_fn,
            config=config,
        )

        new_piece = KnowledgePiece(content=content)
        result = deduplicator.deduplicate(new_piece)

        assert result.action == DedupAction.ADD
        assert "Tier 3 disabled" in result.reason

    @given(
        content=_content_strategy,
        auto_dedup_threshold=st.floats(min_value=0.90, max_value=0.99, allow_nan=False, allow_infinity=False),
        llm_judge_threshold=st.floats(min_value=0.50, max_value=0.89, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_borderline_score_with_tier3_enabled_invokes_llm(
        self, content: str, auto_dedup_threshold: float, llm_judge_threshold: float
    ):
        """When score is between thresholds and Tier 3 is enabled, Tier 3 should be invoked.

        We verify this by providing no llm_fn, which causes Tier 3 to default to ADD
        with a specific reason about no LLM function.

        **Validates: Requirements 12.5**
        """
        assume(auto_dedup_threshold > llm_judge_threshold)
        assume(len(content.strip()) > 0)
        assume(auto_dedup_threshold - llm_judge_threshold > 0.02)

        score = (auto_dedup_threshold + llm_judge_threshold) / 2.0

        existing_piece = KnowledgePiece(content="existing content")
        store = InMemoryPieceStore(pieces=[existing_piece], score=score)

        config = DedupConfig(
            auto_dedup_threshold=auto_dedup_threshold,
            llm_judge_threshold=llm_judge_threshold,
            enable_tier1=False,
            enable_tier2=True,
            enable_tier3=True,  # Tier 3 enabled
        )

        deduplicator = ThreeTierDeduplicator(
            piece_store=store,
            embedding_fn=_dummy_embedding_fn,
            llm_fn=None,  # No LLM → Tier 3 defaults to ADD with "No LLM function" reason
            config=config,
        )

        new_piece = KnowledgePiece(content=content)
        result = deduplicator.deduplicate(new_piece)

        # Tier 3 was invoked (evidenced by the "No LLM function" fallback)
        assert result.action == DedupAction.ADD
        assert "No LLM function" in result.reason


# Feature: knowledge-module-migration, Property 18: Deduplicator Tier 1 exact hash match


class TestDeduplicatorTier1ExactHashMatch:
    """Property 18: Deduplicator Tier 1 exact hash match.

    For any piece whose content_hash matches an existing piece in the store,
    Tier 1 should return DedupAction.NO_OP with the existing piece's ID.

    **Validates: Requirements 12.1**
    """

    @given(content=_content_strategy)
    @settings(max_examples=100)
    def test_matching_hash_returns_no_op(self, content: str):
        """When a piece's content_hash matches an existing piece, Tier 1 returns NO_OP.

        **Validates: Requirements 12.1**
        """
        assume(len(content.strip()) > 0)

        # Create an existing piece with the same content (same hash)
        existing_piece = KnowledgePiece(content=content)
        store = InMemoryPieceStore(pieces=[existing_piece])

        config = DedupConfig(
            enable_tier1=True,
            enable_tier2=False,
            enable_tier3=False,
        )

        deduplicator = ThreeTierDeduplicator(
            piece_store=store,
            embedding_fn=_dummy_embedding_fn,
            config=config,
        )

        # New piece with the same content → same content_hash
        new_piece = KnowledgePiece(content=content)

        result = deduplicator.deduplicate(new_piece)

        assert result.action == DedupAction.NO_OP
        assert result.existing_piece_id == existing_piece.piece_id

    @given(
        content1=_content_strategy,
        content2=_content_strategy,
    )
    @settings(max_examples=100)
    def test_whitespace_only_difference_matches_hash(self, content1: str, content2: str):
        """Two pieces with content differing only in whitespace should have the same hash.

        **Validates: Requirements 12.1**
        """
        assume(len(content1.strip()) > 0)

        # Create a whitespace-varied version of content1
        ws_content = "  " + content1 + "  \n\t"

        existing_piece = KnowledgePiece(content=content1)
        store = InMemoryPieceStore(pieces=[existing_piece])

        config = DedupConfig(
            enable_tier1=True,
            enable_tier2=False,
            enable_tier3=False,
        )

        deduplicator = ThreeTierDeduplicator(
            piece_store=store,
            embedding_fn=_dummy_embedding_fn,
            config=config,
        )

        # New piece with whitespace-varied content → same normalized hash
        new_piece = KnowledgePiece(content=ws_content)

        result = deduplicator.deduplicate(new_piece)

        assert result.action == DedupAction.NO_OP
        assert result.existing_piece_id == existing_piece.piece_id

    @given(
        content1=_content_strategy,
        content2=_content_strategy,
    )
    @settings(max_examples=100)
    def test_different_content_no_hash_match(self, content1: str, content2: str):
        """Two pieces with genuinely different content should not match on hash.

        **Validates: Requirements 12.1**
        """
        assume(len(content1.strip()) > 0)
        assume(len(content2.strip()) > 0)
        # Ensure the normalized content is actually different
        assume(content1.strip() != content2.strip())
        assume(" ".join(content1.split()) != " ".join(content2.split()))

        existing_piece = KnowledgePiece(content=content1)
        store = InMemoryPieceStore(pieces=[existing_piece])

        config = DedupConfig(
            enable_tier1=True,
            enable_tier2=False,
            enable_tier3=False,
        )

        deduplicator = ThreeTierDeduplicator(
            piece_store=store,
            embedding_fn=_dummy_embedding_fn,
            config=config,
        )

        new_piece = KnowledgePiece(content=content2)
        result = deduplicator.deduplicate(new_piece)

        # Different content → no hash match → ADD (since Tier 2 is disabled)
        assert result.action == DedupAction.ADD
