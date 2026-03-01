"""Unit tests for knowledge pack / skill space assignment (Tasks 15.1–15.5)."""

import json
import os
import tempfile
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from agent_foundation.knowledge.packs.models import (
    KnowledgePack,
    PackInstallResult,
    PackSource,
    PackStatus,
)
from agent_foundation.knowledge.packs.pack_manager import KnowledgePackManager
from agent_foundation.knowledge.packs.local_pack_loader import LocalPackLoader
from agent_foundation.knowledge.ingestion.skill_synthesizer import (
    SkillSynthesizer,
    SkillSynthesisConfig,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore
from agent_foundation.knowledge.models.entity_metadata import EntityMetadata


# ---------------------------------------------------------------------------
# Minimal in-memory stores for testing
# ---------------------------------------------------------------------------

class InMemoryPieceStore(KnowledgePieceStore):
    """Minimal in-memory piece store for pack tests."""

    def __init__(self):
        self._pieces: dict[str, KnowledgePiece] = {}

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
        self, query, entity_id=None, knowledge_type=None, tags=None, top_k=5,
        spaces=None,
    ) -> List[Tuple[KnowledgePiece, float]]:
        return [(p, 0.9) for p in list(self._pieces.values())[:top_k]]

    def list_all(self, entity_id=None, knowledge_type=None, spaces=None) -> List[KnowledgePiece]:
        return list(self._pieces.values())


class InMemoryMetadataStore:
    """Minimal in-memory metadata store for pack tests."""

    def __init__(self):
        self._store: dict[str, EntityMetadata] = {}

    def save_metadata(self, metadata: EntityMetadata):
        self._store[metadata.entity_id] = metadata

    def get_metadata(self, entity_id: str) -> Optional[EntityMetadata]:
        return self._store.get(entity_id)

    def delete_metadata(self, entity_id: str):
        self._store.pop(entity_id, None)

    def list_entities(self) -> List[str]:
        return list(self._store.keys())


class InMemoryGraphStore:
    """Minimal in-memory graph store for pack tests."""

    def __init__(self):
        self._nodes: dict = {}
        self._edges: list = []

    def add_node(self, node):
        self._nodes[node.node_id] = node

    def get_node(self, node_id: str):
        return self._nodes.get(node_id)

    def add_relation(self, edge):
        self._edges.append(edge)

    def remove_node(self, node_id: str):
        self._nodes.pop(node_id, None)
        self._edges = [e for e in self._edges if e.source_id != node_id and e.target_id != node_id]

    def get_relations(self, entity_id: str, relation_type=None, direction=None):
        results = [e for e in self._edges if e.source_id == entity_id or e.target_id == entity_id]
        if relation_type:
            results = [e for e in results if e.edge_type == relation_type]
        if direction == "outgoing":
            results = [e for e in results if e.source_id == entity_id]
        elif direction == "incoming":
            results = [e for e in results if e.target_id == entity_id]
        return results


def _make_piece(content="test content", **kwargs) -> KnowledgePiece:
    return KnowledgePiece(content=content, **kwargs)


def _make_manager():
    piece_store = InMemoryPieceStore()
    metadata_store = InMemoryMetadataStore()
    graph_store = InMemoryGraphStore()
    # KnowledgePackManager expects a KnowledgeBase with store attributes
    kb = MagicMock()
    kb.piece_store = piece_store
    kb.metadata_store = metadata_store
    kb.graph_store = graph_store
    manager = KnowledgePackManager(kb=kb)
    return manager, piece_store, metadata_store


# ===========================================================================
# Task 15.1: KnowledgePack model spaces field
# ===========================================================================

class TestKnowledgePackSpacesField:
    """Tests for the spaces field on KnowledgePack."""

    def test_default_spaces_is_none(self):
        pack = KnowledgePack(pack_id="p1", name="test")
        assert pack.spaces is None

    def test_spaces_set_on_construction(self):
        pack = KnowledgePack(pack_id="p1", name="test", spaces=["personal", "main"])
        assert pack.spaces == ["personal", "main"]

    def test_to_dict_includes_spaces_when_set(self):
        pack = KnowledgePack(pack_id="p1", name="test", spaces=["personal"])
        d = pack.to_dict()
        assert d["spaces"] == ["personal"]

    def test_to_dict_spaces_none_when_not_set(self):
        pack = KnowledgePack(pack_id="p1", name="test")
        d = pack.to_dict()
        assert d["spaces"] is None

    def test_from_dict_reads_spaces(self):
        data = {"pack_id": "p1", "name": "test", "spaces": ["developmental"]}
        pack = KnowledgePack.from_dict(data)
        assert pack.spaces == ["developmental"]

    def test_from_dict_defaults_spaces_to_none(self):
        data = {"pack_id": "p1", "name": "test"}
        pack = KnowledgePack.from_dict(data)
        assert pack.spaces is None

    def test_round_trip_with_spaces(self):
        original = KnowledgePack(pack_id="p1", name="test", spaces=["personal", "main"])
        restored = KnowledgePack.from_dict(original.to_dict())
        assert restored.spaces == original.spaces

    def test_round_trip_without_spaces(self):
        original = KnowledgePack(pack_id="p1", name="test")
        restored = KnowledgePack.from_dict(original.to_dict())
        assert restored.spaces is None


# ===========================================================================
# Task 15.2: KnowledgePackManager install/update space stamping
# ===========================================================================

class TestPackManagerSpaceStamping:
    """Tests for space stamping in install() and update()."""

    def test_install_stamps_spaces_on_pieces(self):
        manager, piece_store, _ = _make_manager()
        pack = KnowledgePack(
            pack_id="pack:test:1", name="test", spaces=["personal"]
        )
        pieces = [_make_piece("content A"), _make_piece("content B")]

        result = manager.install(pack, pieces)
        assert result.success

        for pid in pack.piece_ids:
            stored = piece_store.get_by_id(pid)
            assert stored.spaces == ["personal"]
            assert stored.space == "personal"

    def test_install_none_spaces_leaves_pieces_unchanged(self):
        manager, piece_store, _ = _make_manager()
        pack = KnowledgePack(pack_id="pack:test:2", name="test")  # spaces=None
        pieces = [_make_piece("content")]

        result = manager.install(pack, pieces)
        assert result.success

        stored = piece_store.get_by_id(pack.piece_ids[0])
        assert stored.spaces == ["main"]  # default from KnowledgePiece

    def test_install_multi_spaces(self):
        manager, piece_store, _ = _make_manager()
        pack = KnowledgePack(
            pack_id="pack:test:3", name="test", spaces=["personal", "main"]
        )
        pieces = [_make_piece("content")]

        result = manager.install(pack, pieces)
        assert result.success

        stored = piece_store.get_by_id(pack.piece_ids[0])
        assert stored.spaces == ["personal", "main"]
        assert stored.space == "personal"

    def test_update_stamps_spaces_on_new_pieces(self):
        manager, piece_store, _ = _make_manager()

        # First install without spaces
        pack_v1 = KnowledgePack(pack_id="pack:test:4", name="test", version="1.0")
        pieces_v1 = [_make_piece("old content")]
        manager.install(pack_v1, pieces_v1)

        # Update with spaces
        pack_v2 = KnowledgePack(
            pack_id="pack:test:4", name="test", version="2.0",
            spaces=["developmental"],
        )
        new_pieces = [_make_piece("new content")]
        result = manager.update(pack_v2, new_pieces)
        assert result.success

        for pid in pack_v2.piece_ids:
            stored = piece_store.get_by_id(pid)
            assert stored.spaces == ["developmental"]
            assert stored.space == "developmental"

    def test_update_none_spaces_leaves_new_pieces_unchanged(self):
        manager, piece_store, _ = _make_manager()

        pack_v1 = KnowledgePack(pack_id="pack:test:5", name="test", version="1.0")
        pieces_v1 = [_make_piece("old")]
        manager.install(pack_v1, pieces_v1)

        pack_v2 = KnowledgePack(pack_id="pack:test:5", name="test", version="2.0")
        new_pieces = [_make_piece("new")]
        result = manager.update(pack_v2, new_pieces)
        assert result.success

        stored = piece_store.get_by_id(pack_v2.piece_ids[0])
        assert stored.spaces == ["main"]


# ===========================================================================
# Task 15.3: ClawhubPackAdapter spaces parameter
# ===========================================================================

class TestClawhubAdapterSpaces:
    """Tests for spaces parameter on ClawhubPackAdapter import/update."""

    def test_import_skill_passes_spaces_to_pack(self):
        """Verify import_skill constructs KnowledgePack with spaces."""
        from agent_foundation.knowledge.packs.clawhub_adapter import ClawhubPackAdapter

        manager, piece_store, _ = _make_manager()
        client = MagicMock()
        client.base_url = "https://clawhub.example.com"
        client.get_skill.return_value = {
            "skill": {"displayName": "Test Skill", "summary": "A test"},
            "latestVersion": {"version": "1.0.0"},
        }
        client.get_version.return_value = {
            "version": {"version": "1.0.0", "files": [{"path": "SKILL.md", "size": 100}]},
        }
        client.get_file.return_value = "---\nname: Test\ndescription: A test\n---\nBody content"

        adapter = ClawhubPackAdapter(client=client, pack_manager=manager)
        result = adapter.import_skill("test-skill", spaces=["personal"])

        assert result.success
        # Verify pieces got stamped with spaces
        for pid in result.details.get("piece_ids", []):
            stored = piece_store.get_by_id(pid)
            if stored:
                assert stored.spaces == ["personal"]

    def test_import_skill_no_spaces_default(self):
        """Verify import_skill without spaces leaves pieces at default."""
        from agent_foundation.knowledge.packs.clawhub_adapter import ClawhubPackAdapter

        manager, piece_store, _ = _make_manager()
        client = MagicMock()
        client.base_url = "https://clawhub.example.com"
        client.get_skill.return_value = {
            "skill": {"displayName": "Test Skill", "summary": "A test"},
            "latestVersion": {"version": "1.0.0"},
        }
        client.get_version.return_value = {
            "version": {"version": "1.0.0", "files": [{"path": "SKILL.md", "size": 100}]},
        }
        client.get_file.return_value = "---\nname: Test\ndescription: A test\n---\nBody content"

        adapter = ClawhubPackAdapter(client=client, pack_manager=manager)
        result = adapter.import_skill("test-skill-2")

        assert result.success
        # Pieces should have default spaces
        for p in piece_store._pieces.values():
            assert p.spaces == ["main"]


# ===========================================================================
# Task 15.4: LocalPackLoader spaces parameter
# ===========================================================================

class TestLocalPackLoaderSpaces:
    """Tests for spaces parameter on LocalPackLoader."""

    def test_load_with_explicit_spaces(self):
        manager, piece_store, _ = _make_manager()
        loader = LocalPackLoader(pack_manager=manager)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal SKILL.md
            skill_path = os.path.join(tmpdir, "SKILL.md")
            with open(skill_path, "w") as f:
                f.write("---\nname: Test\n---\nSkill body")

            result = loader.load_from_directory(tmpdir, spaces=["personal", "main"])
            assert result.success

            for p in piece_store._pieces.values():
                assert p.spaces == ["personal", "main"]

    def test_load_with_manifest_spaces(self):
        manager, piece_store, _ = _make_manager()
        loader = LocalPackLoader(pack_manager=manager)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create pack.json with spaces
            manifest = {"name": "test-pack", "spaces": ["developmental"]}
            with open(os.path.join(tmpdir, "pack.json"), "w") as f:
                json.dump(manifest, f)

            skill_path = os.path.join(tmpdir, "SKILL.md")
            with open(skill_path, "w") as f:
                f.write("---\nname: Test\n---\nSkill body")

            result = loader.load_from_directory(tmpdir)
            assert result.success

            for p in piece_store._pieces.values():
                assert p.spaces == ["developmental"]

    def test_explicit_spaces_override_manifest(self):
        manager, piece_store, _ = _make_manager()
        loader = LocalPackLoader(pack_manager=manager)

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = {"name": "test-pack2", "spaces": ["developmental"]}
            with open(os.path.join(tmpdir, "pack.json"), "w") as f:
                json.dump(manifest, f)

            skill_path = os.path.join(tmpdir, "SKILL.md")
            with open(skill_path, "w") as f:
                f.write("---\nname: Test\n---\nSkill body")

            result = loader.load_from_directory(tmpdir, spaces=["personal"])
            assert result.success

            for p in piece_store._pieces.values():
                assert p.spaces == ["personal"]

    def test_load_no_spaces_defaults_to_main(self):
        manager, piece_store, _ = _make_manager()
        loader = LocalPackLoader(pack_manager=manager)

        with tempfile.TemporaryDirectory() as tmpdir:
            skill_path = os.path.join(tmpdir, "SKILL.md")
            with open(skill_path, "w") as f:
                f.write("---\nname: Test\n---\nSkill body")

            result = loader.load_from_directory(tmpdir)
            assert result.success

            for p in piece_store._pieces.values():
                assert p.spaces == ["main"]


# ===========================================================================
# Task 15.5: SkillSynthesizer._create_skill_piece() space inheritance
# ===========================================================================

class TestSkillSynthesizerSpaceInheritance:
    """Tests for space inheritance in _create_skill_piece."""

    def test_inherits_spaces_from_first_source_piece(self):
        store = InMemoryPieceStore()
        synthesizer = SkillSynthesizer(piece_store=store)

        source_pieces = [
            _make_piece("piece 1", spaces=["personal", "main"]),
            _make_piece("piece 2", spaces=["developmental"]),
        ]

        skill_dict = {
            "name": "Test Skill",
            "description": "A synthesized skill",
            "steps": [{"step": 1, "description": "Do something"}],
        }

        result = synthesizer._create_skill_piece(skill_dict, source_pieces)
        assert result.spaces == ["personal", "main"]

    def test_defaults_to_main_when_no_source_pieces(self):
        store = InMemoryPieceStore()
        synthesizer = SkillSynthesizer(piece_store=store)

        skill_dict = {
            "name": "Test Skill",
            "description": "A synthesized skill",
            "steps": [],
        }

        result = synthesizer._create_skill_piece(skill_dict, [])
        assert result.spaces == ["main"]

    def test_inherits_single_space(self):
        store = InMemoryPieceStore()
        synthesizer = SkillSynthesizer(piece_store=store)

        source_pieces = [
            _make_piece("piece 1", spaces=["developmental"]),
        ]

        skill_dict = {
            "name": "Dev Skill",
            "description": "A dev skill",
            "steps": [],
        }

        result = synthesizer._create_skill_piece(skill_dict, source_pieces)
        assert result.spaces == ["developmental"]
        assert result.space == "developmental"
