"""
Tests for KnowledgePieceStore.find_by_content_hash default implementation.

Validates Requirement 12.1: The KnowledgePieceStore ABC SHALL include a
find_by_content_hash(content_hash, entity_id) method with a default
linear-scan implementation that searches for a piece matching the given
content hash.
"""
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Path setup
_src_dir = Path(__file__).resolve().parents[3] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import pytest

from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.stores.pieces.base import (
    KnowledgePieceStore,
)


class InMemoryPieceStore(KnowledgePieceStore):
    """Minimal concrete implementation for testing the default find_by_content_hash."""

    def __init__(self):
        self._pieces: List[KnowledgePiece] = []

    def add(self, piece: KnowledgePiece) -> str:
        self._pieces.append(piece)
        return piece.piece_id

    def get_by_id(self, piece_id: str) -> Optional[KnowledgePiece]:
        for p in self._pieces:
            if p.piece_id == piece_id:
                return p
        return None

    def update(self, piece: KnowledgePiece) -> bool:
        for i, p in enumerate(self._pieces):
            if p.piece_id == piece.piece_id:
                self._pieces[i] = piece
                return True
        return False

    def remove(self, piece_id: str) -> bool:
        for i, p in enumerate(self._pieces):
            if p.piece_id == piece_id:
                self._pieces.pop(i)
                return True
        return False

    def search(
        self,
        query: str,
        entity_id: str = None,
        knowledge_type: KnowledgeType = None,
        tags: List[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[KnowledgePiece, float]]:
        return [(p, 1.0) for p in self._filter(entity_id, knowledge_type)][:top_k]

    def list_all(
        self,
        entity_id: str = None,
        knowledge_type: KnowledgeType = None,
    ) -> List[KnowledgePiece]:
        return self._filter(entity_id, knowledge_type)

    def _filter(self, entity_id, knowledge_type):
        results = [p for p in self._pieces if p.entity_id == entity_id]
        if knowledge_type:
            results = [p for p in results if p.knowledge_type == knowledge_type]
        return results


class TestFindByContentHash:
    """Unit tests for the default linear-scan find_by_content_hash."""

    def test_returns_none_on_empty_store(self):
        store = InMemoryPieceStore()
        assert store.find_by_content_hash("abc123") is None

    def test_finds_global_piece_by_hash(self):
        store = InMemoryPieceStore()
        piece = KnowledgePiece(content="hello world", entity_id=None)
        store.add(piece)
        found = store.find_by_content_hash(piece.content_hash)
        assert found is not None
        assert found.piece_id == piece.piece_id

    def test_finds_entity_scoped_piece(self):
        store = InMemoryPieceStore()
        piece = KnowledgePiece(content="entity content", entity_id="user-1")
        store.add(piece)
        # Not found without entity_id (global scan only)
        assert store.find_by_content_hash(piece.content_hash) is None
        # Found when entity_id is provided
        found = store.find_by_content_hash(piece.content_hash, entity_id="user-1")
        assert found is not None
        assert found.piece_id == piece.piece_id

    def test_returns_none_for_nonexistent_hash(self):
        store = InMemoryPieceStore()
        store.add(KnowledgePiece(content="some content", entity_id=None))
        assert store.find_by_content_hash("nonexistent_hash") is None

    def test_global_piece_found_before_entity_piece(self):
        store = InMemoryPieceStore()
        # Same content in global and entity scope
        global_piece = KnowledgePiece(content="shared content", entity_id=None)
        entity_piece = KnowledgePiece(content="shared content", entity_id="user-1")
        store.add(global_piece)
        store.add(entity_piece)
        # Should find the global one first
        found = store.find_by_content_hash(global_piece.content_hash, entity_id="user-1")
        assert found.piece_id == global_piece.piece_id

    def test_entity_id_none_only_scans_global(self):
        store = InMemoryPieceStore()
        piece = KnowledgePiece(content="entity only", entity_id="user-1")
        store.add(piece)
        # entity_id=None means only scan global pieces
        assert store.find_by_content_hash(piece.content_hash, entity_id=None) is None

    def test_uses_getattr_safely_for_missing_content_hash(self):
        """Pieces without content_hash attribute should not cause errors."""
        store = InMemoryPieceStore()
        piece = KnowledgePiece(content="test", entity_id=None)
        # Manually clear content_hash to simulate old pieces
        object.__setattr__(piece, "content_hash", None)
        store.add(piece)
        assert store.find_by_content_hash("some_hash") is None
