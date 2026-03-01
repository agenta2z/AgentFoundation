"""Unit tests for PostIngestionMergeJob."""

from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

from agent_foundation.knowledge.ingestion.post_ingestion_merge_job import (
    PostIngestionMergeJob,
)
from agent_foundation.knowledge.retrieval.models.enums import (
    MergeStrategy,
    SuggestionStatus,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.models.results import MergeJobResult
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore


class InMemoryPieceStore(KnowledgePieceStore):
    """Minimal in-memory store for testing."""

    def __init__(self) -> None:
        self._pieces: Dict[str, KnowledgePiece] = {}

    def add(self, piece: KnowledgePiece) -> str:
        if piece.piece_id in self._pieces:
            raise ValueError(f"Duplicate piece_id: {piece.piece_id}")
        self._pieces[piece.piece_id] = piece
        return piece.piece_id

    def get_by_id(self, piece_id: str) -> Optional[KnowledgePiece]:
        return self._pieces.get(piece_id)

    def update(self, piece: KnowledgePiece) -> bool:
        if piece.piece_id not in self._pieces:
            return False
        self._pieces[piece.piece_id] = piece
        return True

    def remove(self, piece_id: str) -> bool:
        if piece_id not in self._pieces:
            return False
        del self._pieces[piece_id]
        return True

    def search(self, query, entity_id=None, knowledge_type=None, tags=None, top_k=5, spaces=None):
        return []

    def list_all(self, entity_id=None, knowledge_type=None, spaces=None) -> List[KnowledgePiece]:
        return [
            p for p in self._pieces.values() if p.entity_id == entity_id
        ]


def _make_piece(
    content: str = "test content",
    merge_strategy: Optional[str] = None,
    merge_processed: bool = False,
    space: str = "main",
    spaces: Optional[List[str]] = None,
    piece_id: Optional[str] = None,
    entity_id: Optional[str] = None,
) -> KnowledgePiece:
    kwargs = dict(
        content=content,
        piece_id=piece_id or None,
        merge_strategy=merge_strategy,
        merge_processed=merge_processed,
        space=space,
        entity_id=entity_id,
    )
    if spaces is not None:
        kwargs["spaces"] = spaces
    return KnowledgePiece(**kwargs)


class TestFindDeferredPieces:
    """Tests for _find_deferred_pieces filtering logic."""

    def test_finds_post_ingestion_auto_pieces(self):
        store = InMemoryPieceStore()
        piece = _make_piece(
            merge_strategy=MergeStrategy.POST_INGESTION_AUTO.value,
            merge_processed=False,
        )
        store.add(piece)

        job = PostIngestionMergeJob(store, detect_candidates_fn=lambda p: [])
        deferred = job._find_deferred_pieces("main")

        assert len(deferred) == 1
        assert deferred[0].piece_id == piece.piece_id

    def test_finds_post_ingestion_suggestion_pieces(self):
        store = InMemoryPieceStore()
        piece = _make_piece(
            merge_strategy=MergeStrategy.POST_INGESTION_SUGGESTION.value,
            merge_processed=False,
        )
        store.add(piece)

        job = PostIngestionMergeJob(store, detect_candidates_fn=lambda p: [])
        deferred = job._find_deferred_pieces("main")

        assert len(deferred) == 1

    def test_skips_already_processed_pieces(self):
        store = InMemoryPieceStore()
        piece = _make_piece(
            merge_strategy=MergeStrategy.POST_INGESTION_AUTO.value,
            merge_processed=True,
        )
        store.add(piece)

        job = PostIngestionMergeJob(store, detect_candidates_fn=lambda p: [])
        deferred = job._find_deferred_pieces("main")

        assert len(deferred) == 0

    def test_skips_non_post_ingestion_strategies(self):
        store = InMemoryPieceStore()
        piece = _make_piece(
            merge_strategy=MergeStrategy.AUTO_MERGE_ON_INGEST.value,
            merge_processed=False,
        )
        store.add(piece)

        job = PostIngestionMergeJob(store, detect_candidates_fn=lambda p: [])
        deferred = job._find_deferred_pieces("main")

        assert len(deferred) == 0

    def test_filters_by_space(self):
        store = InMemoryPieceStore()
        piece_main = _make_piece(
            merge_strategy=MergeStrategy.POST_INGESTION_AUTO.value,
            merge_processed=False,
            space="main",
        )
        piece_personal = _make_piece(
            merge_strategy=MergeStrategy.POST_INGESTION_AUTO.value,
            merge_processed=False,
            space="personal",
        )
        store.add(piece_main)
        store.add(piece_personal)

        job = PostIngestionMergeJob(store, detect_candidates_fn=lambda p: [])
        deferred = job._find_deferred_pieces("main")

        assert len(deferred) == 1
        assert deferred[0].piece_id == piece_main.piece_id

    def test_finds_multi_space_piece_by_any_space(self):
        """A piece in ['personal', 'main'] should be found when filtering by 'personal'."""
        store = InMemoryPieceStore()
        piece = _make_piece(
            merge_strategy=MergeStrategy.POST_INGESTION_AUTO.value,
            merge_processed=False,
            spaces=["personal", "main"],
        )
        store.add(piece)

        job = PostIngestionMergeJob(store, detect_candidates_fn=lambda p: [])

        # Should be found when filtering by either space
        assert len(job._find_deferred_pieces("personal")) == 1
        assert len(job._find_deferred_pieces("main")) == 1
        # Should NOT be found for a space it doesn't belong to
        assert len(job._find_deferred_pieces("developmental")) == 0

    def test_filters_by_spaces_list(self):
        """The spaces list parameter filters by multiple spaces at once."""
        store = InMemoryPieceStore()
        piece_main = _make_piece(
            merge_strategy=MergeStrategy.POST_INGESTION_AUTO.value,
            merge_processed=False,
            space="main",
        )
        piece_personal = _make_piece(
            merge_strategy=MergeStrategy.POST_INGESTION_AUTO.value,
            merge_processed=False,
            space="personal",
        )
        piece_dev = _make_piece(
            merge_strategy=MergeStrategy.POST_INGESTION_AUTO.value,
            merge_processed=False,
            space="developmental",
        )
        store.add(piece_main)
        store.add(piece_personal)
        store.add(piece_dev)

        job = PostIngestionMergeJob(store, detect_candidates_fn=lambda p: [])
        deferred = job._find_deferred_pieces(spaces=["main", "personal"])

        assert len(deferred) == 2
        ids = {p.piece_id for p in deferred}
        assert piece_main.piece_id in ids
        assert piece_personal.piece_id in ids

    def test_spaces_list_overrides_single_space(self):
        """When both space and spaces are provided, spaces takes precedence."""
        store = InMemoryPieceStore()
        piece = _make_piece(
            merge_strategy=MergeStrategy.POST_INGESTION_AUTO.value,
            merge_processed=False,
            space="personal",
        )
        store.add(piece)

        job = PostIngestionMergeJob(store, detect_candidates_fn=lambda p: [])
        # space="main" would exclude this piece, but spaces=["personal"] should find it
        deferred = job._find_deferred_pieces(space="main", spaces=["personal"])

        assert len(deferred) == 1
        assert deferred[0].piece_id == piece.piece_id

    def test_run_with_spaces_list(self):
        """The run() method accepts a spaces list parameter."""
        store = InMemoryPieceStore()
        piece_main = _make_piece(
            merge_strategy=MergeStrategy.POST_INGESTION_AUTO.value,
            merge_processed=False,
            space="main",
            piece_id="p-main",
        )
        piece_personal = _make_piece(
            merge_strategy=MergeStrategy.POST_INGESTION_AUTO.value,
            merge_processed=False,
            space="personal",
            piece_id="p-personal",
        )
        store.add(piece_main)
        store.add(piece_personal)

        job = PostIngestionMergeJob(store, detect_candidates_fn=lambda p: [])
        result = job.run(spaces=["main", "personal"])

        assert result.processed == 2


class TestRunAutoMerge:
    """Tests for POST_INGESTION_AUTO merge execution."""

    def test_auto_merge_with_candidate(self):
        store = InMemoryPieceStore()
        existing = _make_piece(content="existing content", piece_id="existing-1")
        existing.merge_processed = True
        store.add(existing)

        new_piece = _make_piece(
            content="new content",
            merge_strategy=MergeStrategy.POST_INGESTION_AUTO.value,
            merge_processed=False,
            piece_id="new-1",
        )
        store.add(new_piece)

        merged_piece = _make_piece(content="merged content", piece_id="merged-1")

        def detect_fn(p):
            if p.piece_id == "new-1":
                return [("existing-1", 0.9, "similar")]
            return []

        def merge_fn(p1, p2):
            return merged_piece

        job = PostIngestionMergeJob(store, detect_fn, merge_fn)
        result = job.run()

        assert result.merged == 1
        assert result.processed == 1
        assert len(result.errors) == 0
        # Existing piece should be deactivated
        assert store.get_by_id("existing-1").is_active is False
        # New piece should be removed
        assert store.get_by_id("new-1") is None
        # Merged piece should be added
        assert store.get_by_id("merged-1") is not None

    def test_auto_merge_no_candidates(self):
        store = InMemoryPieceStore()
        piece = _make_piece(
            merge_strategy=MergeStrategy.POST_INGESTION_AUTO.value,
            merge_processed=False,
            piece_id="p1",
        )
        store.add(piece)

        job = PostIngestionMergeJob(store, detect_candidates_fn=lambda p: [])
        result = job.run()

        assert result.merged == 0
        assert result.processed == 1
        # Piece should still be marked as processed
        assert store.get_by_id("p1").merge_processed is True

    def test_auto_merge_no_merge_fn(self):
        store = InMemoryPieceStore()
        existing = _make_piece(content="existing", piece_id="existing-1")
        existing.merge_processed = True
        store.add(existing)

        piece = _make_piece(
            merge_strategy=MergeStrategy.POST_INGESTION_AUTO.value,
            merge_processed=False,
            piece_id="p1",
        )
        store.add(piece)

        def detect_fn(p):
            return [("existing-1", 0.9, "similar")]

        job = PostIngestionMergeJob(store, detect_fn, merge_fn=None)
        result = job.run()

        # No merge should happen without merge_fn
        assert result.merged == 0
        assert result.processed == 1


class TestRunSuggestion:
    """Tests for POST_INGESTION_SUGGESTION strategy."""

    def test_suggestion_with_candidate(self):
        store = InMemoryPieceStore()
        piece = _make_piece(
            merge_strategy=MergeStrategy.POST_INGESTION_SUGGESTION.value,
            merge_processed=False,
            piece_id="p1",
        )
        store.add(piece)

        def detect_fn(p):
            return [("candidate-1", 0.85, "overlapping content")]

        job = PostIngestionMergeJob(store, detect_fn)
        result = job.run()

        assert result.suggestions_created == 1
        assert result.processed == 1

        updated = store.get_by_id("p1")
        assert updated.pending_merge_suggestion == "candidate-1"
        assert updated.merge_suggestion_reason == "overlapping content"
        assert updated.suggestion_status == SuggestionStatus.PENDING.value
        assert updated.merge_processed is True

    def test_suggestion_no_candidates(self):
        store = InMemoryPieceStore()
        piece = _make_piece(
            merge_strategy=MergeStrategy.POST_INGESTION_SUGGESTION.value,
            merge_processed=False,
            piece_id="p1",
        )
        store.add(piece)

        job = PostIngestionMergeJob(store, detect_candidates_fn=lambda p: [])
        result = job.run()

        assert result.suggestions_created == 0
        assert result.processed == 1
        assert store.get_by_id("p1").merge_processed is True


class TestRunErrorHandling:
    """Tests for error handling during job execution."""

    def test_error_in_detect_candidates(self):
        store = InMemoryPieceStore()
        piece = _make_piece(
            merge_strategy=MergeStrategy.POST_INGESTION_AUTO.value,
            merge_processed=False,
            piece_id="p1",
        )
        store.add(piece)

        def detect_fn(p):
            raise RuntimeError("detection failed")

        job = PostIngestionMergeJob(store, detect_fn)
        result = job.run()

        assert len(result.errors) == 1
        assert "p1" in result.errors[0]
        assert result.processed == 0  # len(deferred) - len(errors) = 1 - 1

    def test_multiple_pieces_partial_failure(self):
        store = InMemoryPieceStore()
        good_piece = _make_piece(
            merge_strategy=MergeStrategy.POST_INGESTION_AUTO.value,
            merge_processed=False,
            piece_id="good",
        )
        bad_piece = _make_piece(
            merge_strategy=MergeStrategy.POST_INGESTION_AUTO.value,
            merge_processed=False,
            piece_id="bad",
        )
        store.add(good_piece)
        store.add(bad_piece)

        def detect_fn(p):
            if p.piece_id == "bad":
                raise RuntimeError("boom")
            return []

        job = PostIngestionMergeJob(store, detect_fn)
        result = job.run()

        assert len(result.errors) == 1
        assert result.processed == 1  # 2 deferred - 1 error


class TestMergeJobResult:
    """Tests for MergeJobResult structure."""

    def test_result_has_duration(self):
        store = InMemoryPieceStore()
        job = PostIngestionMergeJob(store, detect_candidates_fn=lambda p: [])
        result = job.run()

        assert result.duration_seconds >= 0.0

    def test_empty_run_returns_zero_counts(self):
        store = InMemoryPieceStore()
        job = PostIngestionMergeJob(store, detect_candidates_fn=lambda p: [])
        result = job.run()

        assert result.processed == 0
        assert result.merged == 0
        assert result.suggestions_created == 0
        assert result.errors == []
