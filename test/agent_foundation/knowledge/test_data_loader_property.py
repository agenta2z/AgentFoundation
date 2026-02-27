"""
Property-based test for KnowledgeDataLoader round-trip.

Generates random metadata, knowledge pieces, and graph data, serializes
to a temp JSON file, loads via KnowledgeDataLoader, and verifies
round-trip equivalence across all three stores.

# Feature: knowledge-agent-integration
# Property 1: Knowledge data file round-trip

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 9.1, 9.2, 9.3, 9.4, 9.5**
"""
import json
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

_spu_src = Path(__file__).resolve().parents[3] / "SciencePythonUtils" / "src"
if _spu_src.exists() and str(_spu_src) not in sys.path:
    sys.path.insert(0, str(_spu_src))

import pytest
from hypothesis import given, settings, assume, strategies as st

from agent_foundation.knowledge.retrieval.data_loader import KnowledgeDataLoader
from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgeType
from agent_foundation.knowledge.retrieval.stores.metadata.keyvalue_adapter import (
    KeyValueMetadataStore,
)
from agent_foundation.knowledge.retrieval.stores.pieces.retrieval_adapter import (
    RetrievalKnowledgePieceStore,
)
from agent_foundation.knowledge.retrieval.stores.graph.graph_adapter import (
    GraphServiceEntityGraphStore,
)
from rich_python_utils.service_utils.keyvalue_service.memory_keyvalue_service import (
    MemoryKeyValueService,
)
from rich_python_utils.service_utils.retrieval_service.memory_retrieval_service import (
    MemoryRetrievalService,
)
from rich_python_utils.service_utils.graph_service.memory_graph_service import (
    MemoryGraphService,
)


# ── Helper: create a fresh KnowledgeBase with in-memory stores ───────────────


def _make_kb():
    """Create a KnowledgeBase with in-memory stores for testing."""
    return KnowledgeBase(
        metadata_store=KeyValueMetadataStore(MemoryKeyValueService()),
        piece_store=RetrievalKnowledgePieceStore(MemoryRetrievalService()),
        graph_store=GraphServiceEntityGraphStore(MemoryGraphService()),
        active_entity_id="user:test",
    )


# ── Strategies ───────────────────────────────────────────────────────────────

# Safe alphanumeric identifiers (no sensitive patterns, no whitespace-only)
_safe_id = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=1,
    max_size=20,
)

# Entity ID in "type:name" format — required by KeyValueMetadataStore
# which uses parse_entity_type(entity_id) to extract the namespace
_entity_type_prefix = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz",
    min_size=1,
    max_size=10,
)
_entity_name_suffix = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=1,
    max_size=10,
)
_entity_id_strategy = st.tuples(_entity_type_prefix, _entity_name_suffix).map(
    lambda t: f"{t[0]}:{t[1]}"
)

# Safe alphanumeric identifiers for piece_ids, node_ids, etc.
_safe_id = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=1,
    max_size=20,
)

# Safe content: alphanumeric text that won't trigger sensitive content filters
_safe_content = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ",
    min_size=1,
    max_size=80,
).filter(lambda s: s.strip())

# Entity type strategy
_entity_type = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz",
    min_size=1,
    max_size=15,
)

# JSON-serializable property values (simple types only for reliable round-trip)
_simple_value = st.one_of(
    st.text(max_size=30),
    st.integers(min_value=-1000, max_value=1000),
    st.booleans(),
)

# Properties dict with string keys and simple values
_properties_strategy = st.dictionaries(
    st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz",
        min_size=1,
        max_size=15,
    ),
    _simple_value,
    max_size=5,
)

# Valid knowledge types
_knowledge_type_strategy = st.sampled_from(list(KnowledgeType))

# Valid info types
_info_type_strategy = st.sampled_from(["user_profile", "instructions", "context"])

# Tags: simple lowercase strings
_tag_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=1,
    max_size=15,
)


@st.composite
def metadata_entries_strategy(draw):
    """Generate a dict of metadata entries for the data file.

    Entity IDs follow the "type:name" format required by KeyValueMetadataStore.
    The entity_type in the entry matches the prefix of the entity_id so that
    save_metadata and get_metadata use the same namespace.

    Returns:
        Dict mapping entity_id -> {"entity_type": ..., "properties": {...}}
    """
    n = draw(st.integers(min_value=0, max_value=3))
    entries = {}
    used_ids = set()
    for _ in range(n):
        entity_id = draw(_entity_id_strategy.filter(lambda x: x not in used_ids))
        used_ids.add(entity_id)
        # entity_type must match the prefix of entity_id for round-trip
        entity_type = entity_id.split(":")[0]
        properties = draw(_properties_strategy)
        entries[entity_id] = {
            "entity_type": entity_type,
            "properties": properties,
        }
    return entries


@st.composite
def pieces_entries_strategy(draw):
    """Generate a list of knowledge piece dicts for the data file.

    Uses safe content to avoid triggering sensitive content filters.

    Returns:
        List of piece dicts with unique piece_ids.
    """
    n = draw(st.integers(min_value=0, max_value=4))
    pieces = []
    used_ids = set()
    for _ in range(n):
        piece_id = draw(_safe_id.filter(lambda x: x not in used_ids))
        used_ids.add(piece_id)
        content = draw(_safe_content)
        knowledge_type = draw(_knowledge_type_strategy)
        info_type = draw(_info_type_strategy)
        tags = draw(st.lists(_tag_strategy, max_size=3))
        entity_id = draw(st.one_of(st.none(), _safe_id))
        embedding_text = draw(st.one_of(st.none(), _safe_content))

        pieces.append({
            "piece_id": piece_id,
            "content": content,
            "knowledge_type": knowledge_type.value,
            "info_type": info_type,
            "tags": tags,
            "entity_id": entity_id,
            "embedding_text": embedding_text,
        })
    return pieces


@st.composite
def graph_entries_strategy(draw):
    """Generate graph nodes and edges for the data file.

    Edges reference only generated node_ids to avoid ValueError from
    MemoryGraphService.add_edge requiring existing nodes.

    Returns:
        Dict with "nodes" and "edges" lists.
    """
    n_nodes = draw(st.integers(min_value=0, max_value=4))
    nodes = []
    node_ids = []
    for _ in range(n_nodes):
        node_id = draw(_safe_id)
        # Ensure unique node_ids
        if node_id in node_ids:
            continue
        node_ids.append(node_id)
        node_type = draw(_entity_type)
        label = draw(st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ",
            max_size=30,
        ))
        properties = draw(_properties_strategy)
        nodes.append({
            "node_id": node_id,
            "node_type": node_type,
            "label": label,
            "properties": properties,
        })

    # Generate edges only between existing nodes, unique by (source, target, type)
    edges = []
    used_edge_keys = set()
    if len(node_ids) >= 2:
        n_edges = draw(st.integers(min_value=0, max_value=min(3, len(node_ids))))
        for _ in range(n_edges):
            source_id = draw(st.sampled_from(node_ids))
            target_id = draw(st.sampled_from(node_ids).filter(lambda x: x != source_id))
            edge_type = draw(_safe_id)
            edge_key = (source_id, target_id, edge_type)
            if edge_key in used_edge_keys:
                continue
            used_edge_keys.add(edge_key)
            properties = draw(_properties_strategy)
            edges.append({
                "source_id": source_id,
                "target_id": target_id,
                "edge_type": edge_type,
                "properties": properties,
            })

    return {"nodes": nodes, "edges": edges}


@st.composite
def knowledge_data_file_strategy(draw):
    """Generate a complete knowledge data file structure.

    Returns:
        Dict with "metadata", "pieces", and "graph" sections.
    """
    metadata = draw(metadata_entries_strategy())
    pieces = draw(pieces_entries_strategy())
    graph = draw(graph_entries_strategy())
    return {
        "metadata": metadata,
        "pieces": pieces,
        "graph": graph,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Property 1: Knowledge data file round-trip
# ══════════════════════════════════════════════════════════════════════════════


class TestKnowledgeDataFileRoundTrip:
    """Property 1: Knowledge data file round-trip.

    For any valid knowledge data structure (containing arbitrary metadata
    entries, knowledge pieces with valid KnowledgeType and info_type values,
    and graph nodes/edges), serializing it to a JSON file and then loading
    it via KnowledgeDataLoader into a KnowledgeBase should produce stores
    where:
    (a) each metadata entity_id returns an EntityMetadata with equivalent properties
    (b) each piece_id is retrievable with equivalent content, knowledge_type, and info_type
    (c) each graph node and edge is present in the graph store

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 9.1, 9.2, 9.3, 9.4, 9.5**
    """

    @given(data_file=knowledge_data_file_strategy())
    @settings(max_examples=100)
    def test_round_trip_equivalence(self, data_file, tmp_path_factory):
        """Serialize to JSON, load via KnowledgeDataLoader, verify equivalence.

        **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 9.1, 9.2, 9.3, 9.4, 9.5**
        """
        tmp_path = tmp_path_factory.mktemp("roundtrip")

        # Serialize to temp JSON file
        file_path = str(tmp_path / "knowledge_data.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_file, f)

        # Load into a fresh KnowledgeBase
        kb = _make_kb()
        counts = KnowledgeDataLoader.load(kb, file_path)

        # ── (a) Verify metadata round-trip ───────────────────────────────
        metadata_section = data_file["metadata"]
        assert counts["metadata"] == len(metadata_section)

        for entity_id, entry in metadata_section.items():
            loaded_meta = kb.metadata_store.get_metadata(entity_id)
            assert loaded_meta is not None, (
                f"Metadata for entity_id '{entity_id}' not found after load"
            )
            assert loaded_meta.entity_id == entity_id
            assert loaded_meta.entity_type == entry.get("entity_type", "unknown")

            # Verify each property key-value pair
            expected_props = entry.get("properties", {})
            for key, value in expected_props.items():
                assert key in loaded_meta.properties, (
                    f"Property '{key}' missing for entity '{entity_id}'"
                )
                assert loaded_meta.properties[key] == value, (
                    f"Property '{key}' mismatch for entity '{entity_id}': "
                    f"expected {value!r}, got {loaded_meta.properties[key]!r}"
                )

        # ── (b) Verify pieces round-trip ─────────────────────────────────
        pieces_section = data_file["pieces"]
        # Count may be <= len(pieces_section) if some were skipped
        # (sensitive content, etc.), but with safe content it should match
        assert counts["pieces"] == len(pieces_section), (
            f"Expected {len(pieces_section)} pieces loaded, got {counts['pieces']}"
        )

        for piece_entry in pieces_section:
            piece_id = piece_entry["piece_id"]
            loaded_piece = kb.piece_store.get_by_id(piece_id)
            assert loaded_piece is not None, (
                f"Piece '{piece_id}' not found after load"
            )
            assert loaded_piece.content == piece_entry["content"], (
                f"Content mismatch for piece '{piece_id}'"
            )
            assert loaded_piece.knowledge_type.value == piece_entry["knowledge_type"], (
                f"knowledge_type mismatch for piece '{piece_id}': "
                f"expected {piece_entry['knowledge_type']!r}, "
                f"got {loaded_piece.knowledge_type.value!r}"
            )
            assert loaded_piece.info_type == piece_entry["info_type"], (
                f"info_type mismatch for piece '{piece_id}': "
                f"expected {piece_entry['info_type']!r}, "
                f"got {loaded_piece.info_type!r}"
            )

        # ── (c) Verify graph nodes round-trip ────────────────────────────
        graph_section = data_file["graph"]
        nodes = graph_section.get("nodes", [])
        # Deduplicate nodes by node_id (strategy may generate duplicates)
        unique_nodes = {}
        for node_data in nodes:
            unique_nodes[node_data["node_id"]] = node_data
        assert counts["graph_nodes"] == len(unique_nodes), (
            f"Expected {len(unique_nodes)} graph nodes loaded, "
            f"got {counts['graph_nodes']}"
        )

        for node_id, node_data in unique_nodes.items():
            loaded_node = kb.graph_store.get_node(node_id)
            assert loaded_node is not None, (
                f"Graph node '{node_id}' not found after load"
            )
            assert loaded_node.node_type == node_data["node_type"], (
                f"node_type mismatch for node '{node_id}'"
            )
            assert loaded_node.label == node_data.get("label", ""), (
                f"label mismatch for node '{node_id}'"
            )
            # Verify properties
            expected_props = node_data.get("properties", {})
            for key, value in expected_props.items():
                assert key in loaded_node.properties, (
                    f"Property '{key}' missing for node '{node_id}'"
                )
                assert loaded_node.properties[key] == value, (
                    f"Property '{key}' mismatch for node '{node_id}'"
                )

        # ── (c) Verify graph edges round-trip ────────────────────────────
        edges = graph_section.get("edges", [])
        assert counts["graph_edges"] == len(edges), (
            f"Expected {len(edges)} graph edges loaded, "
            f"got {counts['graph_edges']}"
        )

        for edge_data in edges:
            source_id = edge_data["source_id"]
            # Get all outgoing edges from source
            loaded_edges = kb.graph_store.get_relations(
                source_id, direction="outgoing"
            )
            # Find the matching edge
            matching = [
                e for e in loaded_edges
                if e.target_id == edge_data["target_id"]
                and e.edge_type == edge_data["edge_type"]
            ]
            assert len(matching) >= 1, (
                f"Graph edge {source_id} --[{edge_data['edge_type']}]--> "
                f"{edge_data['target_id']} not found after load"
            )
            # Verify edge properties
            loaded_edge = matching[0]
            expected_props = edge_data.get("properties", {})
            for key, value in expected_props.items():
                assert key in loaded_edge.properties, (
                    f"Property '{key}' missing for edge "
                    f"{source_id} -> {edge_data['target_id']}"
                )
                assert loaded_edge.properties[key] == value, (
                    f"Property '{key}' mismatch for edge "
                    f"{source_id} -> {edge_data['target_id']}"
                )
