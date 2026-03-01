"""
Property-based tests for SpaceMigrationUtility.

Feature: knowledge-space-restructuring
- Property 11: Migration Idempotence

For any knowledge store, running the migration utility twice SHALL produce
the same spaces assignments on all pieces, metadata, and graph nodes. The
second run SHALL report zero updates.

**Validates: Requirements 8.6**
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

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

from hypothesis import given, settings, HealthCheck, strategies as st

from agent_foundation.knowledge.ingestion.space_classifier import SpaceClassifier
from agent_foundation.knowledge.ingestion.space_migration import (
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


# ── Strategies ───────────────────────────────────────────────────────────────

_valid_space = st.sampled_from(["main", "personal", "developmental"])

# Entity IDs that exercise different classifier branches
_entity_id_strategy = st.one_of(
    st.none(),
    st.just("user:").flatmap(lambda prefix: st.text(min_size=1, max_size=20).filter(
        lambda s: s.strip()
    ).map(lambda s: prefix + s)),
    st.just("service:").flatmap(lambda prefix: st.text(min_size=1, max_size=20).filter(
        lambda s: s.strip()
    ).map(lambda s: prefix + s)),
)

_validation_status = st.sampled_from(["passed", "not_validated", "failed", "pending"])

_info_type = st.sampled_from(["user_profile", "instructions", "context"])


@st.composite
def _piece_strategy(draw):
    """Generate a KnowledgePiece with random entity_id and spaces."""
    piece_id = draw(st.uuids().map(str))
    entity_id = draw(_entity_id_strategy)
    spaces = draw(st.lists(_valid_space, min_size=1, max_size=3).map(
        lambda xs: list(dict.fromkeys(xs))
    ))
    validation_status = draw(_validation_status)
    info_type = draw(_info_type)

    content = draw(st.text(min_size=1, max_size=50).filter(lambda s: s.strip()))
    knowledge_type = draw(st.sampled_from(list(KnowledgeType)))
    tags = draw(st.lists(st.text(min_size=1, max_size=10).filter(lambda s: s.strip()), max_size=3))
    source = draw(st.one_of(st.none(), st.text(min_size=1, max_size=20).filter(lambda s: s.strip())))
    domain = draw(st.sampled_from(["general", "health", "finance", "tech"]))
    secondary_domains = draw(st.lists(st.sampled_from(["general", "health", "finance", "tech"]), max_size=2))
    is_active = draw(st.booleans())
    version = draw(st.integers(min_value=1, max_value=10))
    summary = draw(st.one_of(st.none(), st.text(min_size=1, max_size=30).filter(lambda s: s.strip())))

    return KnowledgePiece(
        content=content,
        piece_id=piece_id,
        knowledge_type=knowledge_type,
        info_type=info_type,
        tags=tags,
        entity_id=entity_id,
        source=source,
        domain=domain,
        secondary_domains=secondary_domains,
        spaces=spaces,
        validation_status=validation_status,
        is_active=is_active,
        version=version,
        summary=summary,
    )


@st.composite
def _metadata_strategy(draw):
    """Generate an EntityMetadata with random entity_id and spaces."""
    entity_id = draw(st.one_of(
        st.just("user:").flatmap(lambda p: st.text(min_size=1, max_size=10).filter(
            lambda s: s.strip()
        ).map(lambda s: p + s)),
        st.just("service:").flatmap(lambda p: st.text(min_size=1, max_size=10).filter(
            lambda s: s.strip()
        ).map(lambda s: p + s)),
    ))
    entity_type = draw(st.sampled_from(["user", "app", "tool"]))
    spaces = draw(st.lists(_valid_space, min_size=1, max_size=3).map(
        lambda xs: list(dict.fromkeys(xs))
    ))
    return EntityMetadata(
        entity_id=entity_id,
        entity_type=entity_type,
        spaces=spaces,
    )


@st.composite
def _graph_node_strategy(draw):
    """Generate a GraphNode with random node_id and spaces property."""
    node_id = draw(st.one_of(
        st.just("user:").flatmap(lambda p: st.text(min_size=1, max_size=10).filter(
            lambda s: s.strip()
        ).map(lambda s: p + s)),
        st.just("service:").flatmap(lambda p: st.text(min_size=1, max_size=10).filter(
            lambda s: s.strip()
        ).map(lambda s: p + s)),
    ))
    node_type = draw(st.sampled_from(["user", "service"]))
    spaces = draw(st.lists(_valid_space, min_size=1, max_size=3).map(
        lambda xs: list(dict.fromkeys(xs))
    ))
    return GraphNode(
        node_id=node_id,
        node_type=node_type,
        properties={"spaces": spaces},
    )


def _build_kb_mock(pieces_by_scope, metadata_list, entity_ids, nodes_by_id,
                   relations_by_entity):
    """Build a mock KnowledgeBase with the given data.

    The mock stores are backed by the actual mutable objects so that
    migrate() can update them in-place and a second run sees the changes.
    """
    kb = MagicMock()

    # Metadata store
    kb.metadata_store.list_entities.return_value = entity_ids
    metadata_by_id = {m.entity_id: m for m in metadata_list}
    kb.metadata_store.get_metadata.side_effect = lambda eid: metadata_by_id.get(eid)

    # Piece store — returns the live list so in-place mutations are visible
    def list_all_side_effect(entity_id=None, **kwargs):
        return list(pieces_by_scope.get(entity_id, []))

    kb.piece_store.list_all.side_effect = list_all_side_effect
    kb.piece_store.update.return_value = True

    # Graph store
    def get_relations_side_effect(node_id, direction="both"):
        return list(relations_by_entity.get(node_id, []))

    kb.graph_store.get_relations.side_effect = get_relations_side_effect
    kb.graph_store.get_node.side_effect = lambda nid: nodes_by_id.get(nid)

    # No retrieval_service (avoid orphan scope discovery path)
    del kb.piece_store.retrieval_service
    # No _table attribute (avoid LanceDB orphan scope discovery path)
    del kb.piece_store._table

    return kb


# ── Property 11: Migration Idempotence ───────────────────────────────────────


class TestMigrationIdempotence:
    """Property 11: Migration Idempotence.

    For any knowledge store, running the migration utility twice SHALL produce
    the same spaces assignments on all pieces, metadata, and graph nodes. The
    second run SHALL report zero updates.

    **Validates: Requirements 8.6**
    """

    @given(
        pieces=st.lists(_piece_strategy(), min_size=1, max_size=8),
    )
    @settings(max_examples=100)
    def test_second_migration_reports_zero_piece_updates(self, pieces):
        """Running migrate() twice: second run has pieces_updated == 0.

        **Validates: Requirements 8.6**
        """
        # Group pieces by entity_id scope
        pieces_by_scope = {}
        entity_ids = set()
        for p in pieces:
            scope = p.entity_id
            pieces_by_scope.setdefault(scope, []).append(p)
            if scope is not None:
                entity_ids.add(scope)

        kb = _build_kb_mock(
            pieces_by_scope=pieces_by_scope,
            metadata_list=[],
            entity_ids=list(entity_ids),
            nodes_by_id={},
            relations_by_entity={},
        )
        classifier = SpaceClassifier()
        utility = SpaceMigrationUtility(kb, classifier)

        # First migration — applies changes
        utility.migrate()

        # Reset update call tracking for the second run
        kb.piece_store.update.reset_mock()

        # Second migration — should find no changes
        report2 = utility.migrate()

        assert report2.pieces_updated == 0
        kb.piece_store.update.assert_not_called()

    @given(
        metadata_list=st.lists(_metadata_strategy(), min_size=1, max_size=5),
    )
    @settings(max_examples=100)
    def test_second_migration_reports_zero_metadata_updates(self, metadata_list):
        """Running migrate() twice: second run has metadata_updated == 0.

        **Validates: Requirements 8.6**
        """
        # Deduplicate by entity_id (migration looks up by entity_id)
        seen = set()
        unique_metadata = []
        for m in metadata_list:
            if m.entity_id not in seen:
                seen.add(m.entity_id)
                unique_metadata.append(m)

        entity_ids = [m.entity_id for m in unique_metadata]

        kb = _build_kb_mock(
            pieces_by_scope={},
            metadata_list=unique_metadata,
            entity_ids=entity_ids,
            nodes_by_id={},
            relations_by_entity={},
        )
        classifier = SpaceClassifier()
        utility = SpaceMigrationUtility(kb, classifier)

        # First migration
        utility.migrate()

        # Reset tracking
        kb.metadata_store.save_metadata.reset_mock()

        # Second migration
        report2 = utility.migrate()

        assert report2.metadata_updated == 0
        kb.metadata_store.save_metadata.assert_not_called()

    @given(
        nodes=st.lists(_graph_node_strategy(), min_size=1, max_size=5),
    )
    @settings(max_examples=100)
    def test_second_migration_reports_zero_graph_updates(self, nodes):
        """Running migrate() twice: second run has graph_nodes_updated == 0.

        **Validates: Requirements 8.6**
        """
        # Deduplicate nodes by node_id
        seen = set()
        unique_nodes = []
        for n in nodes:
            if n.node_id not in seen:
                seen.add(n.node_id)
                unique_nodes.append(n)

        nodes_by_id = {n.node_id: n for n in unique_nodes}
        entity_ids = [n.node_id for n in unique_nodes]

        # Create edges so migration discovers the nodes via get_relations
        relations_by_entity = {}
        for n in unique_nodes:
            # Self-referencing edge to ensure the node is discovered
            edge = GraphEdge(
                source_id=n.node_id,
                target_id=n.node_id,
                edge_type="SELF",
            )
            relations_by_entity[n.node_id] = [edge]

        kb = _build_kb_mock(
            pieces_by_scope={},
            metadata_list=[],
            entity_ids=entity_ids,
            nodes_by_id=nodes_by_id,
            relations_by_entity=relations_by_entity,
        )
        classifier = SpaceClassifier()
        utility = SpaceMigrationUtility(kb, classifier)

        # First migration
        utility.migrate()

        # Reset tracking
        kb.graph_store.add_node.reset_mock()

        # Second migration
        report2 = utility.migrate()

        assert report2.graph_nodes_updated == 0
        kb.graph_store.add_node.assert_not_called()

    @given(
        pieces=st.lists(_piece_strategy(), min_size=1, max_size=5),
        metadata_list=st.lists(_metadata_strategy(), min_size=0, max_size=3),
        nodes=st.lists(_graph_node_strategy(), min_size=0, max_size=3),
    )
    @settings(max_examples=100)
    def test_full_store_second_migration_all_zero(self, pieces, metadata_list, nodes):
        """Full store with pieces, metadata, and graph: second run reports all zeros.

        **Validates: Requirements 8.6**
        """
        # Group pieces by scope
        pieces_by_scope = {}
        piece_entity_ids = set()
        for p in pieces:
            scope = p.entity_id
            pieces_by_scope.setdefault(scope, []).append(p)
            if scope is not None:
                piece_entity_ids.add(scope)

        # Deduplicate metadata
        seen_meta = set()
        unique_metadata = []
        for m in metadata_list:
            if m.entity_id not in seen_meta:
                seen_meta.add(m.entity_id)
                unique_metadata.append(m)

        # Deduplicate nodes
        seen_nodes = set()
        unique_nodes = []
        for n in nodes:
            if n.node_id not in seen_nodes:
                seen_nodes.add(n.node_id)
                unique_nodes.append(n)

        nodes_by_id = {n.node_id: n for n in unique_nodes}

        # Build edges for node discovery
        relations_by_entity = {}
        for n in unique_nodes:
            edge = GraphEdge(
                source_id=n.node_id,
                target_id=n.node_id,
                edge_type="SELF",
            )
            relations_by_entity[n.node_id] = [edge]

        # Combine all entity_ids
        all_entity_ids = list(
            piece_entity_ids
            | {m.entity_id for m in unique_metadata}
            | {n.node_id for n in unique_nodes}
        )

        kb = _build_kb_mock(
            pieces_by_scope=pieces_by_scope,
            metadata_list=unique_metadata,
            entity_ids=all_entity_ids,
            nodes_by_id=nodes_by_id,
            relations_by_entity=relations_by_entity,
        )
        classifier = SpaceClassifier()
        utility = SpaceMigrationUtility(kb, classifier)

        # First migration
        utility.migrate()

        # Reset all tracking
        kb.piece_store.update.reset_mock()
        kb.metadata_store.save_metadata.reset_mock()
        kb.graph_store.add_node.reset_mock()
        kb.graph_store.add_relation.reset_mock()

        # Second migration — everything should be zero
        report2 = utility.migrate()

        assert report2.pieces_updated == 0
        assert report2.metadata_updated == 0
        assert report2.graph_nodes_updated == 0
        assert report2.graph_edges_updated == 0


# ── Helpers for Property 12 ──────────────────────────────────────────────────

# Fields that migration is allowed to change
_SPACE_FIELDS = frozenset({
    "space",
    "spaces",
    "pending_space_suggestions",
    "space_suggestion_reasons",
    "space_suggestion_status",
})


def _snapshot_non_space_fields(piece: KnowledgePiece) -> dict:
    """Capture all non-space fields of a KnowledgePiece as a dict snapshot."""
    d = piece.to_dict()
    return {k: v for k, v in d.items() if k not in _SPACE_FIELDS}


# ── Property 12: Migration Preserves Non-Space Fields ────────────────────────


class TestMigrationPreservesNonSpaceFields:
    """Property 12: Migration Preserves Non-Space Fields.

    For any KnowledgePiece in the store, after migration, all fields except
    space-related fields SHALL remain identical to their pre-migration values.
    Specifically, content, piece_id, tags, domain, entity_id, knowledge_type,
    content_hash, is_active, version, info_type, source, and other non-space
    fields SHALL be unchanged.

    **Validates: Requirements 8.2**
    """

    @given(
        pieces=st.lists(_piece_strategy(), min_size=1, max_size=8),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_migration_preserves_non_space_fields(self, pieces):
        """After migration, all non-space fields on every piece are unchanged.

        **Validates: Requirements 8.2**
        """
        # Snapshot non-space fields before migration
        snapshots = {p.piece_id: _snapshot_non_space_fields(p) for p in pieces}

        # Group pieces by entity_id scope
        pieces_by_scope = {}
        entity_ids = set()
        for p in pieces:
            scope = p.entity_id
            pieces_by_scope.setdefault(scope, []).append(p)
            if scope is not None:
                entity_ids.add(scope)

        kb = _build_kb_mock(
            pieces_by_scope=pieces_by_scope,
            metadata_list=[],
            entity_ids=list(entity_ids),
            nodes_by_id={},
            relations_by_entity={},
        )
        classifier = SpaceClassifier()
        utility = SpaceMigrationUtility(kb, classifier)

        # Run migration
        utility.migrate()

        # Verify non-space fields are unchanged
        for piece in pieces:
            after = _snapshot_non_space_fields(piece)
            before = snapshots[piece.piece_id]
            assert after == before, (
                f"Non-space fields changed for piece '{piece.piece_id}': "
                f"diff keys = {set(k for k in before if before[k] != after.get(k))}"
            )

    @given(
        pieces=st.lists(_piece_strategy(), min_size=1, max_size=5),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_migration_preserves_individual_fields(self, pieces):
        """Verify specific critical fields are preserved individually.

        **Validates: Requirements 8.2**
        """
        # Snapshot individual fields before migration
        field_snapshots = {}
        for p in pieces:
            field_snapshots[p.piece_id] = {
                "content": p.content,
                "piece_id": p.piece_id,
                "knowledge_type": p.knowledge_type,
                "info_type": p.info_type,
                "tags": list(p.tags),
                "entity_id": p.entity_id,
                "source": p.source,
                "domain": p.domain,
                "secondary_domains": list(p.secondary_domains),
                "content_hash": p.content_hash,
                "is_active": p.is_active,
                "version": p.version,
                "summary": p.summary,
                "validation_status": p.validation_status,
                "validation_issues": list(p.validation_issues),
                "merge_strategy": p.merge_strategy,
                "merge_processed": p.merge_processed,
            }

        # Group pieces by entity_id scope
        pieces_by_scope = {}
        entity_ids = set()
        for p in pieces:
            scope = p.entity_id
            pieces_by_scope.setdefault(scope, []).append(p)
            if scope is not None:
                entity_ids.add(scope)

        kb = _build_kb_mock(
            pieces_by_scope=pieces_by_scope,
            metadata_list=[],
            entity_ids=list(entity_ids),
            nodes_by_id={},
            relations_by_entity={},
        )
        classifier = SpaceClassifier()
        utility = SpaceMigrationUtility(kb, classifier)

        # Run migration
        utility.migrate()

        # Verify each field individually
        for piece in pieces:
            snap = field_snapshots[piece.piece_id]
            assert piece.content == snap["content"]
            assert piece.piece_id == snap["piece_id"]
            assert piece.knowledge_type == snap["knowledge_type"]
            assert piece.info_type == snap["info_type"]
            assert piece.tags == snap["tags"]
            assert piece.entity_id == snap["entity_id"]
            assert piece.source == snap["source"]
            assert piece.domain == snap["domain"]
            assert piece.secondary_domains == snap["secondary_domains"]
            assert piece.content_hash == snap["content_hash"]
            assert piece.is_active == snap["is_active"]
            assert piece.version == snap["version"]
            assert piece.summary == snap["summary"]
            assert piece.validation_status == snap["validation_status"]
            assert piece.validation_issues == snap["validation_issues"]
            assert piece.merge_strategy == snap["merge_strategy"]
            assert piece.merge_processed == snap["merge_processed"]
