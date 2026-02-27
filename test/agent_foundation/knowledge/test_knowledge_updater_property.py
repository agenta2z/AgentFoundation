"""
Property-based tests for the knowledge updater module.

Feature: knowledge-module-migration
- Property 25: Update with preserve_history creates supersedes chain
- Property 26: Update content_hash recomputation

**Validates: Requirements 16.5, 16.7**
"""
import hashlib
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

from agent_foundation.knowledge.ingestion.knowledge_updater import (
    KnowledgeUpdater,
    UpdateConfig,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
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
        return [(p, 0.9) for p in list(self._pieces.values())[:top_k]]

    def list_all(self, entity_id=None, knowledge_type=None) -> List[KnowledgePiece]:
        return list(self._pieces.values())


# ── Strategies ────────────────────────────────────────────────────────────────

# Content strings — non-empty, printable text
_content_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=300,
)

# Version numbers for existing pieces
_version_strategy = st.integers(min_value=1, max_value=100)


# Feature: knowledge-module-migration, Property 25: Update with preserve_history creates supersedes chain


class TestUpdatePreserveHistorySupersedes:
    """Property 25: Update with preserve_history creates supersedes chain.

    For any update operation with preserve_history=True, the new piece should
    have `supersedes` equal to the old piece's piece_id, `version` equal to
    old_version + 1, and the old piece should have `is_active=False`.

    **Validates: Requirements 16.5**
    """

    @given(
        original_content=_content_strategy,
        new_content=_content_strategy,
        old_version=_version_strategy,
    )
    @settings(max_examples=100)
    def test_supersedes_chain_created(
        self, original_content: str, new_content: str, old_version: int
    ):
        """When preserve_history=True, the new piece supersedes the old piece,
        has version = old_version + 1, and the old piece is deactivated.

        **Validates: Requirements 16.5**
        """
        assume(len(original_content.strip()) > 0)
        assume(len(new_content.strip()) > 0)

        old_piece = KnowledgePiece(
            content=original_content,
            version=old_version,
            tags=["test"],
        )
        old_piece_id = old_piece.piece_id

        store = InMemoryPieceStore(pieces=[old_piece])
        config = UpdateConfig(preserve_history=True)
        updater = KnowledgeUpdater(piece_store=store, config=config)

        result = updater.update_by_id(old_piece_id, new_content)

        assert result.success is True

        # New piece should have supersedes link and incremented version
        new_piece = store.get_by_id(result.piece_id)
        assert new_piece is not None
        assert new_piece.supersedes == old_piece_id
        assert new_piece.version == old_version + 1

        # Old piece should be deactivated
        old_piece_after = store.get_by_id(old_piece_id)
        assert old_piece_after is not None
        assert old_piece_after.is_active is False

    @given(
        original_content=_content_strategy,
        new_content=_content_strategy,
        old_version=_version_strategy,
        domain=st.sampled_from(["general", "testing", "debugging", "infrastructure"]),
    )
    @settings(max_examples=100)
    def test_new_piece_is_distinct_from_old(
        self, original_content: str, new_content: str, old_version: int, domain: str
    ):
        """The new piece should be a different piece (different piece_id) from
        the old one, with the new content.

        **Validates: Requirements 16.5**
        """
        assume(len(original_content.strip()) > 0)
        assume(len(new_content.strip()) > 0)

        old_piece = KnowledgePiece(
            content=original_content,
            version=old_version,
            domain=domain,
            tags=["test"],
        )
        old_piece_id = old_piece.piece_id

        store = InMemoryPieceStore(pieces=[old_piece])
        config = UpdateConfig(preserve_history=True)
        updater = KnowledgeUpdater(piece_store=store, config=config)

        result = updater.update_by_id(old_piece_id, new_content)

        assert result.success is True
        assert result.piece_id != old_piece_id

        new_piece = store.get_by_id(result.piece_id)
        assert new_piece.content == new_content

    @given(
        original_content=_content_strategy,
        new_content=_content_strategy,
        old_version=_version_strategy,
    )
    @settings(max_examples=100)
    def test_operation_result_versions(
        self, original_content: str, new_content: str, old_version: int
    ):
        """The OperationResult should report old_version and new_version correctly.

        **Validates: Requirements 16.5**
        """
        assume(len(original_content.strip()) > 0)
        assume(len(new_content.strip()) > 0)

        old_piece = KnowledgePiece(
            content=original_content,
            version=old_version,
            tags=["test"],
        )

        store = InMemoryPieceStore(pieces=[old_piece])
        config = UpdateConfig(preserve_history=True)
        updater = KnowledgeUpdater(piece_store=store, config=config)

        result = updater.update_by_id(old_piece.piece_id, new_content)

        assert result.success is True
        assert result.old_version == old_version
        assert result.new_version == old_version + 1


# Feature: knowledge-module-migration, Property 26: Update content_hash recomputation


class TestUpdateContentHashRecomputation:
    """Property 26: Update content_hash recomputation.

    For any updated piece (whether in-place or new piece), the content_hash
    should match the SHA256 of the whitespace-normalized new content.

    **Validates: Requirements 16.7**
    """

    @staticmethod
    def _expected_content_hash(content: str) -> str:
        """Compute the expected content hash matching KnowledgePiece._compute_content_hash."""
        normalized = " ".join(content.split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    @given(
        original_content=_content_strategy,
        new_content=_content_strategy,
    )
    @settings(max_examples=100)
    def test_in_place_update_recomputes_hash(
        self, original_content: str, new_content: str
    ):
        """When preserve_history=False (in-place update), the content_hash of
        the updated piece should match SHA256 of whitespace-normalized new content.

        **Validates: Requirements 16.7**
        """
        assume(len(original_content.strip()) > 0)
        assume(len(new_content.strip()) > 0)

        piece = KnowledgePiece(content=original_content, tags=["test"])
        store = InMemoryPieceStore(pieces=[piece])
        config = UpdateConfig(preserve_history=False)
        updater = KnowledgeUpdater(piece_store=store, config=config)

        result = updater.update_by_id(piece.piece_id, new_content)

        assert result.success is True
        updated_piece = store.get_by_id(result.piece_id)
        expected_hash = self._expected_content_hash(new_content)
        assert updated_piece.content_hash == expected_hash

    @given(
        original_content=_content_strategy,
        new_content=_content_strategy,
    )
    @settings(max_examples=100)
    def test_new_piece_with_history_has_correct_hash(
        self, original_content: str, new_content: str
    ):
        """When preserve_history=True, the new piece's content_hash should match
        SHA256 of whitespace-normalized new content (auto-computed on creation).

        **Validates: Requirements 16.7**
        """
        assume(len(original_content.strip()) > 0)
        assume(len(new_content.strip()) > 0)

        piece = KnowledgePiece(content=original_content, tags=["test"])
        store = InMemoryPieceStore(pieces=[piece])
        config = UpdateConfig(preserve_history=True)
        updater = KnowledgeUpdater(piece_store=store, config=config)

        result = updater.update_by_id(piece.piece_id, new_content)

        assert result.success is True
        new_piece = store.get_by_id(result.piece_id)
        expected_hash = self._expected_content_hash(new_content)
        assert new_piece.content_hash == expected_hash

    @given(
        words=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N")),
                min_size=1,
                max_size=20,
            ),
            min_size=1,
            max_size=10,
        ),
        extra_spaces=st.lists(
            st.integers(min_value=1, max_value=5),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=100)
    def test_whitespace_variants_produce_same_hash(
        self, words: List[str], extra_spaces: List[int]
    ):
        """Content that differs only in whitespace should produce the same
        content_hash after update.

        **Validates: Requirements 16.7**
        """
        # Build two whitespace variants from the same words
        assume(any(len(w.strip()) > 0 for w in words))
        clean_words = [w for w in words if len(w.strip()) > 0]
        assume(len(clean_words) > 0)

        content_a = " ".join(clean_words)
        # Build content_b with variable spacing between words
        parts = []
        for i, w in enumerate(clean_words):
            parts.append(w)
            if i < len(clean_words) - 1:
                spaces = extra_spaces[i % len(extra_spaces)]
                parts.append(" " * (spaces + 1))
        content_b = "".join(parts)

        piece = KnowledgePiece(content="original", tags=["test"])
        store = InMemoryPieceStore(pieces=[piece])
        config = UpdateConfig(preserve_history=False)
        updater = KnowledgeUpdater(piece_store=store, config=config)

        # Update with content_a
        updater.update_by_id(piece.piece_id, content_a)
        hash_a = store.get_by_id(piece.piece_id).content_hash

        # Reset and update with content_b
        piece2 = KnowledgePiece(content="original", tags=["test"])
        store2 = InMemoryPieceStore(pieces=[piece2])
        updater2 = KnowledgeUpdater(piece_store=store2, config=config)

        updater2.update_by_id(piece2.piece_id, content_b)
        hash_b = store2.get_by_id(piece2.piece_id).content_hash

        assert hash_a == hash_b
