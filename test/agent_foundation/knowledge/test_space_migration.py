"""
Unit tests for SpaceMigrationUtility and MigrationReport.

Tests the migration utility with mock stores to verify:
- All entity scopes are iterated (not just global)
- SpaceClassifier auto_spaces are applied, suggestions ignored
- Pieces, metadata, and graph nodes are updated when spaces change
- Errors are handled per-item (logged and continued)
- MigrationReport counts are accurate
- Migration is idempotent (second run reports zero updates)

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 13.1, 13.2
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Path setup
_current_file = Path(__file__).resolve()
_current_path = _current_file.parent
while _current_path.name != "test" and _current_path.parent != _current_path:
    _current_path = _current_path.parent
_src_dir = _current_path.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
_rpu_src = Path(__file__).resolve().parents[4] / "RichPythonUtils" / "src"
if _rpu_src.exists() and str(_rpu_src) not in sys.path:
    sys.path.insert(0, str(_rpu_src))

from agent_foundation.knowledge.ingestion.space_classifier import (
    ClassificationResult,
    SpaceClassifier,
)
from agent_foundation.knowledge.ingestion.space_migration import (
    MigrationReport,
    SpaceMigrationUtility,
)
from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from rich_python_utils.service_utils.graph_service.graph_node import (
    GraphEdge,
    GraphNode,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_piece(piece_id="p1", entity_id=None, spaces=None, validation_status="passed"):
    """Create a KnowledgePiece with sensible defaults for testing."""
    return KnowledgePiece(
        content="test content",
        piece_id=piece_id,
        knowledge_type=KnowledgeType.Fact,
        info_type="context",
        entity_id=entity_id,
        spaces=spaces or ["main"],
        validation_status=validation_status,
    )


def _make_metadata(entity_id="user:alice", entity_type="user", spaces=None):
    """Create an EntityMetadata with sensible defaults for testing."""
    return EntityMetadata(
        entity_id=entity_id,
        entity_type=entity_type,
        spaces=spaces or ["main"],
    )


def _make_kb_mock(pieces_by_scope=None, metadata_list=None, entity_ids=None,
                  relations_by_entity=None, nodes_by_id=None):
    """Create a mock KnowledgeBase with configurable stores.

    Args:
        pieces_by_scope: Dict mapping entity_id (or None) to list of KnowledgePiece.
        metadata_list: List of EntityMetadata objects.
        entity_ids: List of entity_id strings returned by list_entities().
        relations_by_entity: Dict mapping entity_id to list of GraphEdge.
        nodes_by_id: Dict mapping node_id to GraphNode.
    """
    pieces_by_scope = pieces_by_scope or {}
    metadata_list = metadata_list or []
    entity_ids = entity_ids or []
    relations_by_entity = relations_by_entity or {}
    nodes_by_id = nodes_by_id or {}

    kb = MagicMock()

    # Metadata store
    kb.metadata_store.list_entities.return_value = entity_ids
    metadata_by_id = {m.entity_id: m for m in metadata_list}
    kb.metadata_store.get_metadata.side_effect = lambda eid: metadata_by_id.get(eid)

    # Piece store
    def list_all_side_effect(entity_id=None, **kwargs):
        return list(pieces_by_scope.get(entity_id, []))

    kb.piece_store.list_all.side_effect = list_all_side_effect
    kb.piece_store.update.return_value = True

    # Graph store
    def get_relations_side_effect(node_id, direction="both"):
        return list(relations_by_entity.get(node_id, []))

    kb.graph_store.get_relations.side_effect = get_relations_side_effect
    kb.graph_store.get_node.side_effect = lambda nid: nodes_by_id.get(nid)

    return kb


# ── MigrationReport tests ───────────────────────────────────────────────────


class TestMigrationReport:
    def test_default_values(self):
        report = MigrationReport()
        assert report.pieces_updated == 0
        assert report.metadata_updated == 0
        assert report.graph_nodes_updated == 0
        assert report.graph_edges_updated == 0
        assert report.space_counts == {}
        assert report.errors == []
        assert report.total_scanned == 0

    def test_mutable_defaults_are_independent(self):
        r1 = MigrationReport()
        r2 = MigrationReport()
        r1.errors.append("err")
        r1.space_counts["main"] = 5
        assert r2.errors == []
        assert r2.space_counts == {}


# ── SpaceMigrationUtility tests ─────────────────────────────────────────────


class TestSpaceMigrationUtility:
    def test_empty_store_produces_empty_report(self):
        """Migration on an empty store should scan nothing and update nothing."""
        kb = _make_kb_mock()
        classifier = SpaceClassifier()
        utility = SpaceMigrationUtility(kb, classifier)

        report = utility.migrate()

        assert report.pieces_updated == 0
        assert report.metadata_updated == 0
        assert report.graph_nodes_updated == 0
        assert report.total_scanned == 0
        assert report.errors == []

    def test_iterates_all_entity_scopes_and_global(self):
        """Migration must call list_all for each entity AND for global (None)."""
        p_global = _make_piece("g1", entity_id=None)
        p_user = _make_piece("u1", entity_id="user:alice")
        p_service = _make_piece("s1", entity_id="service:grocery")

        kb = _make_kb_mock(
            pieces_by_scope={
                None: [p_global],
                "user:alice": [p_user],
                "service:grocery": [p_service],
            },
            entity_ids=["user:alice", "service:grocery"],
        )
        classifier = SpaceClassifier()
        utility = SpaceMigrationUtility(kb, classifier)

        report = utility.migrate()

        # All 3 pieces should be scanned
        assert report.total_scanned >= 3

        # list_all should have been called for each entity + global
        calls = kb.piece_store.list_all.call_args_list
        called_entity_ids = [c.kwargs.get("entity_id", c.args[0] if c.args else None) for c in calls]
        assert None in called_entity_ids  # global
        assert "user:alice" in called_entity_ids
        assert "service:grocery" in called_entity_ids

    def test_user_pieces_get_personal_space(self):
        """Pieces with user: entity_id should be classified as personal+main."""
        piece = _make_piece("u1", entity_id="user:alice", spaces=["main"])

        kb = _make_kb_mock(
            pieces_by_scope={"user:alice": [piece]},
            entity_ids=["user:alice"],
        )
        classifier = SpaceClassifier()
        utility = SpaceMigrationUtility(kb, classifier)

        report = utility.migrate()

        assert report.pieces_updated == 1
        # The piece should now have personal in its spaces
        assert "personal" in piece.spaces

    def test_failed_validation_gets_developmental(self):
        """Pieces with validation_status=failed should get developmental space."""
        piece = _make_piece("f1", spaces=["main"], validation_status="failed")

        kb = _make_kb_mock(
            pieces_by_scope={None: [piece]},
            entity_ids=[],
        )
        classifier = SpaceClassifier()
        utility = SpaceMigrationUtility(kb, classifier)

        report = utility.migrate()

        assert report.pieces_updated == 1
        assert piece.spaces == ["developmental"]

    def test_no_change_means_no_update(self):
        """Pieces already in the correct space should not be updated."""
        piece = _make_piece("m1", spaces=["main"])

        kb = _make_kb_mock(
            pieces_by_scope={None: [piece]},
            entity_ids=[],
        )
        classifier = SpaceClassifier()
        utility = SpaceMigrationUtility(kb, classifier)

        report = utility.migrate()

        assert report.pieces_updated == 0
        kb.piece_store.update.assert_not_called()

    def test_idempotent_second_run_zero_updates(self):
        """Running migration twice should produce zero updates on the second run."""
        piece = _make_piece("u1", entity_id="user:alice", spaces=["main"])

        kb = _make_kb_mock(
            pieces_by_scope={"user:alice": [piece]},
            entity_ids=["user:alice"],
        )
        classifier = SpaceClassifier()
        utility = SpaceMigrationUtility(kb, classifier)

        # First run: should update
        report1 = utility.migrate()
        assert report1.pieces_updated == 1

        # Second run: piece already has correct spaces
        report2 = utility.migrate()
        assert report2.pieces_updated == 0

    def test_metadata_migration(self):
        """Metadata for user entities should get personal space."""
        meta = _make_metadata("user:alice", "user", spaces=["main"])

        kb = _make_kb_mock(
            metadata_list=[meta],
            entity_ids=["user:alice"],
        )
        classifier = SpaceClassifier()
        utility = SpaceMigrationUtility(kb, classifier)

        report = utility.migrate()

        assert report.metadata_updated == 1
        assert "personal" in meta.spaces
        kb.metadata_store.save_metadata.assert_called_once()

    def test_metadata_no_change_no_update(self):
        """Metadata already in correct space should not be saved."""
        meta = _make_metadata("service:grocery", "app", spaces=["main"])

        kb = _make_kb_mock(
            metadata_list=[meta],
            entity_ids=["service:grocery"],
        )
        classifier = SpaceClassifier()
        utility = SpaceMigrationUtility(kb, classifier)

        report = utility.migrate()

        assert report.metadata_updated == 0
        kb.metadata_store.save_metadata.assert_not_called()

    def test_graph_node_migration(self):
        """Graph nodes discovered via relations should have spaces updated."""
        node_user = GraphNode(
            node_id="user:alice", node_type="user", properties={}
        )
        node_service = GraphNode(
            node_id="service:grocery", node_type="service", properties={}
        )
        edge = GraphEdge(
            source_id="user:alice",
            target_id="service:grocery",
            edge_type="USES",
        )

        kb = _make_kb_mock(
            entity_ids=["user:alice"],
            relations_by_entity={"user:alice": [edge]},
            nodes_by_id={
                "user:alice": node_user,
                "service:grocery": node_service,
            },
        )
        classifier = SpaceClassifier()
        utility = SpaceMigrationUtility(kb, classifier)

        report = utility.migrate()

        # user:alice node should get personal space
        assert report.graph_nodes_updated >= 1
        assert "personal" in node_user.properties.get("spaces", [])

    def test_error_handling_continues_processing(self):
        """Errors on individual items should be logged but not stop migration."""
        good_piece = _make_piece("g1", spaces=["main"])
        bad_piece = _make_piece("b1", spaces=["main"])

        kb = _make_kb_mock(
            pieces_by_scope={None: [good_piece, bad_piece]},
            entity_ids=[],
        )

        # Make update fail for the bad piece
        def update_side_effect(piece):
            if piece.piece_id == "b1":
                raise RuntimeError("DB write failed")
            return True

        # Use a classifier that forces an update on both pieces
        classifier = SpaceClassifier()

        # Override classify_piece to force a change
        original_classify = classifier.classify_piece

        def force_change_classify(piece):
            return ClassificationResult(auto_spaces=["developmental"])

        classifier.classify_piece = force_change_classify
        kb.piece_store.update.side_effect = update_side_effect

        utility = SpaceMigrationUtility(kb, classifier)
        report = utility.migrate()

        # Both pieces scanned, one error
        assert report.total_scanned == 2
        assert len(report.errors) == 1
        assert "b1" in report.errors[0]
        # The good piece should still have been updated
        assert report.pieces_updated == 1

    def test_space_counts_tracked(self):
        """Migration report should track how many items are in each space."""
        pieces = [
            _make_piece("p1", entity_id="user:alice", spaces=["main"]),
            _make_piece("p2", entity_id=None, spaces=["main"]),
            _make_piece("p3", entity_id=None, spaces=["main"], validation_status="failed"),
        ]

        kb = _make_kb_mock(
            pieces_by_scope={
                "user:alice": [pieces[0]],
                None: [pieces[1], pieces[2]],
            },
            entity_ids=["user:alice"],
        )
        classifier = SpaceClassifier()
        utility = SpaceMigrationUtility(kb, classifier)

        report = utility.migrate()

        # user:alice piece → personal + main, p2 → main, p3 → developmental
        assert "main" in report.space_counts
        assert "developmental" in report.space_counts

    def test_suggestion_mode_rules_ignored(self):
        """Only auto_spaces should be applied; suggested_spaces should be ignored."""
        piece = _make_piece("p1", spaces=["main"])

        kb = _make_kb_mock(
            pieces_by_scope={None: [piece]},
            entity_ids=[],
        )

        # Create a classifier that returns suggestions
        classifier = SpaceClassifier()
        original_classify = classifier.classify_piece

        def classify_with_suggestions(p):
            return ClassificationResult(
                auto_spaces=["main"],
                suggested_spaces=["personal"],
                suggestion_reasons=["Test suggestion"],
            )

        classifier.classify_piece = classify_with_suggestions
        utility = SpaceMigrationUtility(kb, classifier)

        report = utility.migrate()

        # No update because auto_spaces == current spaces
        assert report.pieces_updated == 0
        # piece should NOT have pending_space_suggestions set
        assert piece.pending_space_suggestions is None

    def test_orphaned_scope_discovery_via_retrieval_service(self):
        """Orphaned scopes should be discovered via retrieval_service.namespaces()."""
        orphan_piece = _make_piece("o1", entity_id="orphan:scope", spaces=["main"])

        kb = _make_kb_mock(
            pieces_by_scope={
                "orphan:scope": [orphan_piece],
            },
            entity_ids=[],  # metadata doesn't know about this scope
        )

        # Simulate a retrieval service with namespaces()
        mock_retrieval_service = MagicMock()
        mock_retrieval_service.namespaces.return_value = ["orphan:scope"]
        kb.piece_store.retrieval_service = mock_retrieval_service

        classifier = SpaceClassifier()
        utility = SpaceMigrationUtility(kb, classifier)

        report = utility.migrate()

        # The orphaned piece should have been scanned
        assert report.total_scanned >= 1

    def test_graph_edge_spaces_updated(self):
        """Graph edges should get spaces based on their endpoint nodes."""
        node_user = GraphNode(
            node_id="user:alice", node_type="user", properties={"spaces": ["personal"]}
        )
        node_service = GraphNode(
            node_id="service:grocery", node_type="service", properties={"spaces": ["main"]}
        )
        edge = GraphEdge(
            source_id="user:alice",
            target_id="service:grocery",
            edge_type="USES",
            properties={},
        )

        kb = _make_kb_mock(
            entity_ids=["user:alice"],
            relations_by_entity={"user:alice": [edge]},
            nodes_by_id={
                "user:alice": node_user,
                "service:grocery": node_service,
            },
        )
        classifier = SpaceClassifier()
        utility = SpaceMigrationUtility(kb, classifier)

        report = utility.migrate()

        assert report.graph_edges_updated >= 1
        edge_spaces = edge.properties.get("spaces", [])
        assert "personal" in edge_spaces or "main" in edge_spaces

    def test_list_entities_failure_handled(self):
        """If list_entities() fails, migration should still process global pieces."""
        global_piece = _make_piece("g1", spaces=["main"])

        kb = _make_kb_mock(
            pieces_by_scope={None: [global_piece]},
        )
        kb.metadata_store.list_entities.side_effect = RuntimeError("DB down")

        classifier = SpaceClassifier()
        utility = SpaceMigrationUtility(kb, classifier)

        report = utility.migrate()

        assert len(report.errors) >= 1
        assert "list entities" in report.errors[0].lower() or "DB down" in report.errors[0]
        # Global pieces should still be processed
        assert report.total_scanned >= 1
