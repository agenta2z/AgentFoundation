"""
Property-based tests for the merge strategy module.

Feature: knowledge-module-migration
- Property 19: Merge strategy default mapping
- Property 20: Suggestion-on-ingest sets pending fields
- Property 21: Post-ingestion strategy defers processing

**Validates: Requirements 13.2, 13.4, 13.5**
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

from agent_foundation.knowledge.ingestion.merge_strategy import (
    MergeStrategyConfig,
    MergeStrategyManager,
)
from agent_foundation.knowledge.retrieval.models.enums import (
    MergeAction,
    MergeStrategy,
    MergeType,
    SuggestionStatus,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.models.results import MergeCandidate
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore


# ── Test Helpers ──────────────────────────────────────────────────────────────


class InMemoryPieceStore(KnowledgePieceStore):
    """Minimal in-memory store for property testing."""

    def __init__(self, pieces: Optional[List[KnowledgePiece]] = None):
        self._pieces: dict[str, KnowledgePiece] = {}
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
        return []

    def list_all(self, entity_id=None, knowledge_type=None) -> List[KnowledgePiece]:
        return list(self._pieces.values())


# ── Strategies ────────────────────────────────────────────────────────────────

# All KnowledgeType values
_knowledge_type_strategy = st.sampled_from(list(KnowledgeType))

# Content strings for generating pieces
_content_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=5,
    max_size=200,
)

# Similarity scores for merge candidates
_similarity_strategy = st.floats(
    min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Merge type for candidates
_merge_type_strategy = st.sampled_from(list(MergeType))

# Reason strings for candidates
_reason_strategy = st.text(min_size=1, max_size=50)


def _make_candidate(
    piece_id: str = "candidate-1",
    similarity: float = 0.9,
    merge_type: MergeType = MergeType.OVERLAPPING,
    reason: str = "Similar content",
) -> MergeCandidate:
    return MergeCandidate(
        piece_id=piece_id,
        similarity=similarity,
        merge_type=merge_type,
        reason=reason,
    )


# ── Default strategy mapping from MergeStrategyConfig ─────────────────────────

DEFAULT_STRATEGY_MAP = {
    KnowledgeType.Procedure: MergeStrategy.MANUAL_ONLY,
    KnowledgeType.Instruction: MergeStrategy.SUGGESTION_ON_INGEST,
    KnowledgeType.Fact: MergeStrategy.AUTO_MERGE_ON_INGEST,
    KnowledgeType.Preference: MergeStrategy.MANUAL_ONLY,
    KnowledgeType.Episodic: MergeStrategy.POST_INGESTION_AUTO,
    KnowledgeType.Note: MergeStrategy.AUTO_MERGE_ON_INGEST,
    KnowledgeType.Example: MergeStrategy.SUGGESTION_ON_INGEST,
}


# Feature: knowledge-module-migration, Property 19: Merge strategy default mapping


class TestMergeStrategyDefaultMapping:
    """Property 19: Merge strategy default mapping.

    For any KnowledgeType, the get_strategy() method should return the configured
    default strategy for that type when the piece has no override.

    **Validates: Requirements 13.2**
    """

    @given(
        knowledge_type=_knowledge_type_strategy,
        content=_content_strategy,
    )
    @settings(max_examples=100)
    def test_default_strategy_matches_config(
        self, knowledge_type: KnowledgeType, content: str
    ):
        """get_strategy() returns the default strategy for the piece's KnowledgeType.

        **Validates: Requirements 13.2**
        """
        assume(len(content.strip()) > 0)

        store = InMemoryPieceStore()
        manager = MergeStrategyManager(piece_store=store)

        piece = KnowledgePiece(
            content=content,
            knowledge_type=knowledge_type,
        )

        result = manager.get_strategy(piece)
        expected = DEFAULT_STRATEGY_MAP[knowledge_type]

        assert result == expected

    @given(
        knowledge_type=_knowledge_type_strategy,
        content=_content_strategy,
        override_strategy=st.sampled_from(list(MergeStrategy)),
    )
    @settings(max_examples=100)
    def test_override_ignored_when_allow_override_false(
        self,
        knowledge_type: KnowledgeType,
        content: str,
        override_strategy: MergeStrategy,
    ):
        """When allow_override is False, piece-level overrides are ignored.

        **Validates: Requirements 13.2**
        """
        assume(len(content.strip()) > 0)

        config = MergeStrategyConfig(allow_override=False)
        store = InMemoryPieceStore()
        manager = MergeStrategyManager(piece_store=store, config=config)

        piece = KnowledgePiece(
            content=content,
            knowledge_type=knowledge_type,
            merge_strategy=override_strategy.value,
        )

        result = manager.get_strategy(piece)
        expected = DEFAULT_STRATEGY_MAP[knowledge_type]

        assert result == expected


# Feature: knowledge-module-migration, Property 20: Suggestion-on-ingest sets pending fields


class TestSuggestionOnIngestSetsPendingFields:
    """Property 20: Suggestion-on-ingest sets pending fields.

    For any piece with suggestion-on-ingest strategy and at least one merge candidate,
    after apply_strategy(), the piece's pending_merge_suggestion should be set to the
    top candidate's piece_id, and suggestion_status should be "pending".

    **Validates: Requirements 13.4**
    """

    @given(
        content=_content_strategy,
        candidate_id=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=3,
            max_size=30,
        ),
        similarity=_similarity_strategy,
        merge_type=_merge_type_strategy,
        reason=_reason_strategy,
    )
    @settings(max_examples=100)
    def test_suggestion_sets_pending_fields(
        self,
        content: str,
        candidate_id: str,
        similarity: float,
        merge_type: MergeType,
        reason: str,
    ):
        """apply_strategy() with suggestion-on-ingest sets pending_merge_suggestion and suggestion_status.

        **Validates: Requirements 13.4**
        """
        assume(len(content.strip()) > 0)
        assume(len(candidate_id.strip()) > 0)

        store = InMemoryPieceStore()
        # Configure so that the piece's type maps to SUGGESTION_ON_INGEST
        config = MergeStrategyConfig(
            default_by_type={KnowledgeType.Fact: MergeStrategy.SUGGESTION_ON_INGEST}
        )
        manager = MergeStrategyManager(piece_store=store, config=config)

        piece = KnowledgePiece(
            content=content,
            knowledge_type=KnowledgeType.Fact,
        )

        candidate = MergeCandidate(
            piece_id=candidate_id,
            similarity=similarity,
            merge_type=merge_type,
            reason=reason,
        )

        result = manager.apply_strategy(piece, [candidate])

        assert result.action == MergeAction.PENDING_REVIEW
        assert piece.pending_merge_suggestion == candidate_id
        assert piece.suggestion_status == SuggestionStatus.PENDING.value
        assert piece.merge_suggestion_reason == reason


# Feature: knowledge-module-migration, Property 21: Post-ingestion strategy defers processing


class TestPostIngestionStrategyDefersProcessing:
    """Property 21: Post-ingestion strategy defers processing.

    For any piece with a post-ingestion strategy, after apply_strategy(), the piece's
    merge_processed should be False and the returned MergeResult action should be DEFERRED.

    **Validates: Requirements 13.5**
    """

    @given(
        content=_content_strategy,
        post_strategy=st.sampled_from([
            MergeStrategy.POST_INGESTION_AUTO,
            MergeStrategy.POST_INGESTION_SUGGESTION,
        ]),
        candidate_id=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=3,
            max_size=30,
        ),
        similarity=_similarity_strategy,
    )
    @settings(max_examples=100)
    def test_post_ingestion_defers_and_marks_unprocessed(
        self,
        content: str,
        post_strategy: MergeStrategy,
        candidate_id: str,
        similarity: float,
    ):
        """apply_strategy() with post-ingestion strategy returns DEFERRED and sets merge_processed=False.

        **Validates: Requirements 13.5**
        """
        assume(len(content.strip()) > 0)

        store = InMemoryPieceStore()
        # Configure the piece's type to use the given post-ingestion strategy
        config = MergeStrategyConfig(
            default_by_type={KnowledgeType.Fact: post_strategy}
        )
        manager = MergeStrategyManager(piece_store=store, config=config)

        piece = KnowledgePiece(
            content=content,
            knowledge_type=KnowledgeType.Fact,
        )

        candidate = _make_candidate(
            piece_id=candidate_id,
            similarity=similarity,
        )

        result = manager.apply_strategy(piece, [candidate])

        assert result.action == MergeAction.DEFERRED
        assert piece.merge_processed is False
