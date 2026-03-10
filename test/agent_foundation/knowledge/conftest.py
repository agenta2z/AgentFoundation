"""
Hypothesis strategies for knowledge module data models.

Provides reusable strategies for generating random instances of:
- KnowledgePiece
- EntityMetadata
- GraphNode
- GraphEdge

Used by property-based tests across the knowledge module.
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Path resolution for imports
_current_file = Path(__file__).resolve()
_current_path = _current_file.parent
while _current_path.name != "test" and _current_path.parent != _current_path:
    _current_path = _current_path.parent
_src_dir = _current_path.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Also add RichPythonUtils src to path (provides rich_python_utils.service_utils)
_rpu_src = Path(__file__).resolve().parents[4] / "RichPythonUtils" / "src"
if _rpu_src.exists() and str(_rpu_src) not in sys.path:
    sys.path.insert(0, str(_rpu_src))

from hypothesis import strategies as st

from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgeType,
    KnowledgePiece,
)
from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata
from rich_python_utils.service_utils.graph_service.graph_node import (
    GraphNode,
    GraphEdge,
)
from rich_python_utils.service_utils.retrieval_service.document import Document
from rich_python_utils.service_utils.retrieval_service.retrieval_service_base import (
    RetrievalServiceBase,
)
from agent_foundation.knowledge.retrieval.stores.graph.base import EntityGraphStore


# ── Shared helper strategies ─────────────────────────────────────────────────

# Non-empty text strategy (content must be non-empty after stripping)
_non_empty_text = st.text(min_size=1).filter(lambda s: s.strip())

# Simple text for identifiers and labels
_identifier_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=50,
)

# Tag strategy: non-empty strings (will be normalized by KnowledgePiece)
_tag_strategy = st.text(min_size=1, max_size=30)

# ISO 8601 timestamp strategy
_timestamp_strategy = st.from_regex(
    r"20[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]\+00:00",
    fullmatch=True,
)

# JSON-serializable values for properties dicts
_json_leaf = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    st.text(max_size=50),
)

_json_value = st.recursive(
    _json_leaf,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(min_size=1, max_size=20), children, max_size=5),
    ),
    max_leaves=10,
)

# Properties dict strategy (JSON-serializable values)
_properties_strategy = st.dictionaries(
    st.text(min_size=1, max_size=20),
    _json_value,
    max_size=5,
)


# ── KnowledgePiece strategy ─────────────────────────────────────────────────


@st.composite
def knowledge_piece_strategy(draw, include_new_fields=False):
    """Generate a random KnowledgePiece instance.

    Args:
        include_new_fields: If True, also populate all new fields (domain,
            content_hash, embedding, space, merge strategy, validation,
            versioning, summary). If False, only original 10 fields are set.

    - content: non-empty text (required, must have non-whitespace chars)
    - piece_id: explicit UUID-like string to ensure round-trip stability
    - knowledge_type: random KnowledgeType enum value
    - tags: list of text strings (will be normalized by __attrs_post_init__)
    - entity_id: optional string
    - source: optional string
    - embedding_text: optional string
    - created_at/updated_at: explicit ISO 8601 timestamps for round-trip stability
    """
    content = draw(_non_empty_text)
    piece_id = draw(_identifier_text)
    knowledge_type = draw(st.sampled_from(list(KnowledgeType)))
    info_type = draw(st.sampled_from(["user_profile", "instructions", "context"]))
    tags = draw(st.lists(_tag_strategy, max_size=5))
    entity_id = draw(st.one_of(st.none(), _identifier_text))
    source = draw(st.one_of(st.none(), st.text(max_size=50)))
    embedding_text = draw(st.one_of(st.none(), st.text(max_size=100)))
    created_at = draw(_timestamp_strategy)
    updated_at = draw(_timestamp_strategy)

    kwargs = dict(
        content=content,
        piece_id=piece_id,
        knowledge_type=knowledge_type,
        info_type=info_type,
        tags=tags,
        entity_id=entity_id,
        source=source,
        embedding_text=embedding_text,
        created_at=created_at,
        updated_at=updated_at,
    )

    if include_new_fields:
        _domain_strategy = st.sampled_from(["general", "model_optimization", "data_engineering", "testing", "debugging"])
        _space_strategy = st.sampled_from(["main", "personal", "developmental"])
        _merge_strategy_values = st.sampled_from([None, "auto-merge-on-ingest", "suggestion-on-ingest", "post-ingestion-auto", "manual-only"])
        _suggestion_status_values = st.sampled_from([None, "pending", "approved", "rejected", "expired"])
        _validation_status_values = st.sampled_from(["not_validated", "pending", "passed", "failed"])
        _embedding_strategy = st.one_of(
            st.none(),
            st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-10.0, max_value=10.0), min_size=3, max_size=8),
        )

        kwargs.update(
            domain=draw(_domain_strategy),
            secondary_domains=draw(st.lists(_domain_strategy, max_size=3)),
            custom_tags=draw(st.lists(st.text(min_size=1, max_size=20), max_size=3)),
            embedding=draw(_embedding_strategy),
            space=draw(_space_strategy),
            merge_strategy=draw(_merge_strategy_values),
            merge_processed=draw(st.booleans()),
            pending_merge_suggestion=draw(st.one_of(st.none(), _identifier_text)),
            merge_suggestion_reason=draw(st.one_of(st.none(), st.text(max_size=50))),
            suggestion_status=draw(_suggestion_status_values),
            validation_status=draw(_validation_status_values),
            validation_issues=draw(st.lists(st.text(min_size=1, max_size=30), max_size=3)),
            supersedes=draw(st.one_of(st.none(), _identifier_text)),
            is_active=draw(st.booleans()),
            version=draw(st.integers(min_value=1, max_value=100)),
            summary=draw(st.one_of(st.none(), st.text(max_size=100))),
        )
        # content_hash is auto-computed, so we don't set it explicitly

    return KnowledgePiece(**kwargs)



# ── EntityMetadata strategy ──────────────────────────────────────────────────


@st.composite
def entity_metadata_strategy(draw, include_spaces=False):
    """Generate a random EntityMetadata instance.

    - entity_id: required non-empty string
    - entity_type: required non-empty string
    - properties: dict with JSON-serializable values
    - created_at/updated_at: explicit ISO 8601 timestamps for round-trip stability
    - spaces: optional list of space strings (when include_spaces=True)
    """
    entity_id = draw(_identifier_text)
    entity_type = draw(_identifier_text)
    properties = draw(_properties_strategy)
    created_at = draw(_timestamp_strategy)
    updated_at = draw(_timestamp_strategy)

    kwargs = dict(
        entity_id=entity_id,
        entity_type=entity_type,
        properties=properties,
        created_at=created_at,
        updated_at=updated_at,
    )

    if include_spaces:
        _space_strategy = st.sampled_from(["main", "personal", "developmental"])
        spaces = draw(st.lists(_space_strategy, min_size=1, max_size=3).map(
            lambda xs: list(dict.fromkeys(xs))  # deduplicate, preserve order
        ))
        kwargs["spaces"] = spaces

    return EntityMetadata(**kwargs)


# ── GraphNode strategy ───────────────────────────────────────────────────────


@st.composite
def graph_node_strategy(draw):
    """Generate a random GraphNode instance.

    - node_id: required non-empty string
    - node_type: required non-empty string
    - label: optional string (defaults to "")
    - properties: dict with JSON-serializable values
    """
    node_id = draw(_identifier_text)
    node_type = draw(_identifier_text)
    label = draw(st.text(max_size=50))
    properties = draw(_properties_strategy)

    return GraphNode(
        node_id=node_id,
        node_type=node_type,
        label=label,
        properties=properties,
    )


# ── GraphEdge strategy ───────────────────────────────────────────────────────


@st.composite
def graph_edge_strategy(draw):
    """Generate a random GraphEdge instance.

    - source_id: required non-empty string
    - target_id: required non-empty string
    - edge_type: required non-empty string
    - properties: dict with JSON-serializable values
    """
    source_id = draw(_identifier_text)
    target_id = draw(_identifier_text)
    edge_type = draw(_identifier_text)
    properties = draw(_properties_strategy)

    return GraphEdge(
        source_id=source_id,
        target_id=target_id,
        edge_type=edge_type,
        properties=properties,
    )


# ── Backward-compatible aliases ──────────────────────────────────────────────
# These aliases allow existing tests that reference the old strategy names
# to continue working during the transition period.
entity_node_strategy = graph_node_strategy
entity_relation_strategy = graph_edge_strategy


# ── Shared in-memory test helpers ────────────────────────────────────────────


class InMemoryEntityGraphStore(EntityGraphStore):
    """Simple in-memory EntityGraphStore for testing."""

    def __init__(self):
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: List[GraphEdge] = []

    def add_node(self, node: GraphNode, **kwargs) -> None:
        self._nodes[node.node_id] = node

    def get_node(self, node_id: str, **kwargs) -> Optional[GraphNode]:
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str, **kwargs) -> bool:
        if node_id in self._nodes:
            del self._nodes[node_id]
            self._edges = [
                e for e in self._edges
                if e.source_id != node_id and e.target_id != node_id
            ]
            return True
        return False

    def add_relation(self, relation: GraphEdge, **kwargs) -> None:
        self._edges.append(relation)

    def get_relations(self, node_id: str, relation_type=None, direction="outgoing", **kwargs) -> List[GraphEdge]:
        results = []
        for e in self._edges:
            if direction in ("outgoing", "both") and e.source_id == node_id:
                if relation_type is None or e.edge_type == relation_type:
                    results.append(e)
            if direction in ("incoming", "both") and e.target_id == node_id:
                if relation_type is None or e.edge_type == relation_type:
                    results.append(e)
        return results

    def remove_relation(self, source_id: str, target_id: str, relation_type: str, **kwargs) -> bool:
        for i, e in enumerate(self._edges):
            if e.source_id == source_id and e.target_id == target_id and e.edge_type == relation_type:
                self._edges.pop(i)
                return True
        return False

    def get_neighbors(self, node_id: str, relation_type=None, depth=1, **kwargs) -> List[Tuple[GraphNode, int]]:
        results = []
        visited = {node_id}
        current_level = {node_id}
        for d in range(1, depth + 1):
            next_level = set()
            for nid in current_level:
                for e in self._edges:
                    if e.source_id == nid:
                        if relation_type is None or e.edge_type == relation_type:
                            if e.target_id not in visited:
                                next_level.add(e.target_id)
            for nid in next_level:
                visited.add(nid)
                node = self._nodes.get(nid)
                if node:
                    results.append((node, d))
            current_level = next_level
        return results

    def list_nodes(self, node_type=None, include_inactive=False, **kwargs) -> List[GraphNode]:
        nodes = list(self._nodes.values())
        if not include_inactive:
            nodes = [n for n in nodes if n.is_active]
        if node_type:
            nodes = [n for n in nodes if n.node_type == node_type]
        return nodes

    def close(self):
        pass


class InMemoryRetrievalService(RetrievalServiceBase):
    """Simple in-memory RetrievalServiceBase for testing."""

    def __init__(self):
        self._docs: Dict[str, Dict[str, Document]] = {}
        self._closed = False

    def _ns(self, namespace):
        return namespace or "_default"

    def add(self, doc: Document, namespace: Optional[str] = None) -> str:
        ns = self._ns(namespace)
        if ns not in self._docs:
            self._docs[ns] = {}
        if doc.doc_id in self._docs[ns]:
            raise ValueError(f"Document {doc.doc_id} already exists in namespace {ns}")
        self._docs[ns][doc.doc_id] = doc
        return doc.doc_id

    def get_by_id(self, doc_id: str, namespace: Optional[str] = None) -> Optional[Document]:
        ns = self._ns(namespace)
        return self._docs.get(ns, {}).get(doc_id)

    def update(self, doc: Document, namespace: Optional[str] = None) -> bool:
        ns = self._ns(namespace)
        if ns in self._docs and doc.doc_id in self._docs[ns]:
            self._docs[ns][doc.doc_id] = doc
            return True
        return False

    def remove(self, doc_id: str, namespace: Optional[str] = None) -> bool:
        ns = self._ns(namespace)
        if ns in self._docs and doc_id in self._docs[ns]:
            del self._docs[ns][doc_id]
            return True
        return False

    def search(self, query: str, filters=None, namespace=None, top_k=5) -> List[Tuple[Document, float]]:
        ns = self._ns(namespace)
        docs = list(self._docs.get(ns, {}).values())
        if filters:
            filtered = []
            for doc in docs:
                match = True
                for k, v in filters.items():
                    if k not in doc.metadata or doc.metadata[k] != v:
                        match = False
                        break
                if match:
                    filtered.append(doc)
            docs = filtered
        # Simple keyword matching: score 0.8 if query appears in content, else 0.3
        results = []
        for doc in docs:
            score = 0.8 if query.lower() in doc.content.lower() else 0.3
            results.append((doc, score))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def list_all(self, filters=None, namespace=None) -> List[Document]:
        ns = self._ns(namespace)
        return list(self._docs.get(ns, {}).values())

    def size(self, namespace=None) -> int:
        ns = self._ns(namespace)
        return len(self._docs.get(ns, {}))

    def clear(self, namespace=None) -> int:
        ns = self._ns(namespace)
        count = len(self._docs.get(ns, {}))
        self._docs[ns] = {}
        return count

    def namespaces(self) -> List[str]:
        return list(self._docs.keys())

    def get_stats(self, namespace=None) -> Dict[str, Any]:
        return {}

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self) -> str:
        return "InMemoryRetrievalService()"
