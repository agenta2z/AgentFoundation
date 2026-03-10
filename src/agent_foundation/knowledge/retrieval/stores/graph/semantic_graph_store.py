"""
SemanticGraphStore — EntityGraphStore wrapper with sidecar semantic search.

Wraps an existing EntityGraphStore and keeps a RetrievalServiceBase sidecar
index synchronized on every node mutation. Supports three search modes:
native (delegate to wrapped store), sidecar (search retrieval index), or
both (run both and merge via Reciprocal Rank Fusion).

All delegated methods pass ``**kwargs`` through to the wrapped store to
support extended parameters such as ``operation_id`` and ``include_inactive``
from ``GraphServiceEntityGraphStore``.

Requirements: 3.1, 3.2, 3.3, 4.1–4.5, 5.1–5.5, 6.1–6.3, 8.1–8.3, 9.1–9.3
"""

import logging
from typing import Callable, List, Optional, Tuple

from attr import attrib, attrs

from rich_python_utils.service_utils.graph_service.graph_node import GraphEdge, GraphNode
from rich_python_utils.service_utils.retrieval_service.document import Document
from rich_python_utils.service_utils.retrieval_service.retrieval_service_base import (
    RetrievalServiceBase,
)

from agent_foundation.knowledge.retrieval.stores.graph.base import EntityGraphStore
from agent_foundation.knowledge.retrieval.stores.graph.node_text_builder import (
    NodeTextBuilder,
    default_node_text_builder,
)
from agent_foundation.knowledge.retrieval.stores.graph.search_mode import SearchMode

logger = logging.getLogger(__name__)


@attrs
class SemanticGraphStore(EntityGraphStore):
    """EntityGraphStore wrapper that adds semantic search via a sidecar index.

    Delegates all graph operations to the wrapped ``graph_store`` and keeps a
    ``RetrievalServiceBase`` sidecar index synchronized on node mutations.

    Note: ``node.properties`` must contain only JSON-serializable values for
    sidecar indexing to work correctly.

    Attributes:
        graph_store: The underlying EntityGraphStore to delegate to.
        retrieval_service: Optional sidecar retrieval service for semantic
            search. Required when search_mode is SIDECAR or BOTH; may be
            None for NATIVE mode.
        search_mode: Controls which search backends are used.
        node_text_builder: Callable that converts a GraphNode to searchable
            text for the sidecar index.
        rrf_k: Reciprocal Rank Fusion parameter (default 60).
        index_namespace: Namespace used in the sidecar retrieval service.
    """

    graph_store: EntityGraphStore = attrib()
    retrieval_service: Optional[RetrievalServiceBase] = attrib(default=None)
    search_mode: SearchMode = attrib(default=SearchMode.SIDECAR)
    node_text_builder: Callable[[GraphNode], str] = attrib(default=default_node_text_builder)
    rrf_k: int = attrib(default=60)
    index_namespace: str = attrib(default="_graph_nodes")

    def __attrs_post_init__(self):
        if self.search_mode in (SearchMode.SIDECAR, SearchMode.BOTH):
            if self.retrieval_service is None:
                raise ValueError(
                    f"search_mode={self.search_mode.value} requires a retrieval_service "
                    f"but None was provided"
                )
        if self.search_mode in (SearchMode.NATIVE, SearchMode.BOTH):
            if not self.graph_store.supports_semantic_search:
                raise ValueError(
                    f"search_mode={self.search_mode.value} requires a graph store "
                    f"that supports semantic search, but {type(self.graph_store).__name__} "
                    f"does not"
                )

    @property
    def supports_semantic_search(self) -> bool:
        """Always True — SemanticGraphStore exists to provide semantic search."""
        return True

    # ── Node ↔ Document conversion ──────────────────────────────────────

    def _node_to_doc(self, node: GraphNode) -> Document:
        """Convert a GraphNode to a Document for sidecar indexing.

        Note: ``node.properties`` must contain only JSON-serializable values.
        The ``history`` field is NOT stored in metadata to avoid excessive
        index size.

        Args:
            node: The GraphNode to convert.

        Returns:
            A Document with doc_id matching node.node_id.
        """
        text = self.node_text_builder(node)
        embedding_text = node.properties.get("embedding_text")
        return Document(
            doc_id=node.node_id,
            content=text,
            metadata={
                "node_type": node.node_type,
                "label": node.label,
                "is_active": node.is_active,
                "properties": node.properties,
            },
            embedding_text=embedding_text,
        )

    def _doc_to_node(self, doc: Document) -> GraphNode:
        """Convert a Document back to a lightweight GraphNode.

        WARNING: This reconstruction is lossy — the ``history`` field will be
        empty. Prefer fetching the real node via
        ``graph_store.get_node(doc.doc_id)`` when full fidelity is needed.
        This method is a fallback for when the node has been deleted from the
        graph store between indexing and search.

        Args:
            doc: The Document to convert.

        Returns:
            A GraphNode with empty history.
        """
        return GraphNode(
            node_id=doc.doc_id,
            node_type=doc.metadata["node_type"],
            label=doc.metadata.get("label", ""),
            properties=doc.metadata.get("properties", {}),
            is_active=doc.metadata.get("is_active", True),
        )

    # ── CRUD delegation with sidecar sync ──────────────────────────────

    def add_node(self, node: GraphNode, **kwargs) -> None:
        """Add or update a node, syncing the sidecar index.

        Delegates to the wrapped store with ``**kwargs``, then indexes the
        node in the sidecar retrieval service. Uses try-add / catch-ValueError
        / update pattern because ``RetrievalServiceBase.add()`` raises on
        duplicate ``doc_id``.

        If ``retrieval_service`` is None (NATIVE mode), sidecar sync is skipped.

        Args:
            node: The GraphNode to add or update.
            **kwargs: Passed through to the wrapped store (e.g. ``operation_id``).
        """
        self.graph_store.add_node(node, **kwargs)
        if self.retrieval_service is None:
            return
        try:
            doc = self._node_to_doc(node)
            try:
                self.retrieval_service.add(doc, namespace=self.index_namespace)
            except ValueError:
                self.retrieval_service.update(doc, namespace=self.index_namespace)
        except Exception:
            logger.warning(f"Failed to index node {node.node_id} in sidecar", exc_info=True)


    def remove_node(self, node_id: str, **kwargs) -> bool:
        """Remove a node, syncing the sidecar index.

        Delegates to the wrapped store with ``**kwargs``, then removes the
        corresponding document from the sidecar retrieval service.

        If ``retrieval_service`` is None (NATIVE mode), sidecar sync is skipped.

        Args:
            node_id: The unique identifier of the node to remove.
            **kwargs: Passed through to the wrapped store (e.g. ``operation_id``).

        Returns:
            True if the node existed and was removed, False if not found.
        """
        result = self.graph_store.remove_node(node_id, **kwargs)
        if self.retrieval_service is None:
            return result
        try:
            self.retrieval_service.remove(node_id, namespace=self.index_namespace)
        except Exception:
            logger.warning(f"Failed to remove node {node_id} from sidecar", exc_info=True)
        return result

    # ── Pure delegation with **kwargs ────────────────────────────────────

    def get_node(self, node_id: str, **kwargs) -> Optional[GraphNode]:
        """Get a node by ID (pure delegation).

        Args:
            node_id: The unique identifier of the node.
            **kwargs: Passed through to the wrapped store (e.g. ``include_inactive``).

        Returns:
            The GraphNode if found, or None.
        """
        return self.graph_store.get_node(node_id, **kwargs)

    def add_relation(self, relation: GraphEdge, **kwargs) -> None:
        """Add an edge (pure delegation, no sidecar sync).

        Args:
            relation: The GraphEdge to add.
            **kwargs: Passed through to the wrapped store (e.g. ``operation_id``).
        """
        self.graph_store.add_relation(relation, **kwargs)

    def get_relations(self, node_id: str, relation_type=None, direction="outgoing", **kwargs) -> List[GraphEdge]:
        """Get edges for a node (pure delegation).

        Args:
            node_id: The node whose edges to retrieve.
            relation_type: Optional edge type filter.
            direction: Direction filter ("outgoing", "incoming", or "both").
            **kwargs: Passed through to the wrapped store (e.g. ``include_inactive``).

        Returns:
            A list of GraphEdge objects matching the filter criteria.
        """
        return self.graph_store.get_relations(node_id, relation_type=relation_type, direction=direction, **kwargs)

    def remove_relation(self, source_id: str, target_id: str, relation_type: str, **kwargs) -> bool:
        """Remove a specific edge (pure delegation).

        Args:
            source_id: The source node ID.
            target_id: The target node ID.
            relation_type: The edge type to remove.
            **kwargs: Passed through to the wrapped store (e.g. ``operation_id``).

        Returns:
            True if the edge existed and was removed, False if not found.
        """
        return self.graph_store.remove_relation(source_id, target_id, relation_type, **kwargs)

    def get_neighbors(self, node_id: str, relation_type=None, depth=1, **kwargs) -> List[Tuple[GraphNode, int]]:
        """Get neighboring nodes via traversal (pure delegation).

        Args:
            node_id: The starting node for traversal.
            relation_type: Optional edge type filter.
            depth: Maximum traversal depth.
            **kwargs: Passed through to the wrapped store.

        Returns:
            A list of (GraphNode, depth) tuples.
        """
        return self.graph_store.get_neighbors(node_id, relation_type=relation_type, depth=depth, **kwargs)

    def list_nodes(self, node_type=None, include_inactive=False, **kwargs) -> List[GraphNode]:
        """List all nodes (pure delegation).

        Args:
            node_type: Optional node type filter.
            include_inactive: If True, include soft-deleted nodes.
            **kwargs: Passed through to the wrapped store.

        Returns:
            A list of GraphNode objects matching the filter criteria.
        """
        return self.graph_store.list_nodes(node_type=node_type, include_inactive=include_inactive, **kwargs)

    # ── Search ─────────────────────────────────────────────────────────

    def search_nodes(
        self,
        query: str,
        top_k: int = 5,
        node_type: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> List[Tuple[GraphNode, float]]:
        """Search nodes by semantic query with configurable search mode.

        Routes the query to the appropriate backend(s) based on ``search_mode``:
        - SIDECAR: search the retrieval service index, fetch full nodes from
          the graph store for fidelity, fall back to ``_doc_to_node`` if the
          node has been deleted.
        - NATIVE: delegate to the wrapped store's ``search_nodes``.
        - BOTH: run both and merge via ``_rrf_merge``.

        Args:
            query: The search query string. Empty/whitespace returns [].
            top_k: Maximum number of results to return.
            node_type: Optional filter to return only nodes of this type.
            namespace: Reserved for future use.

        Returns:
            A list of (GraphNode, float) tuples ordered by descending score.
        """
        if not query or not query.strip():
            return []

        sidecar_results: List[Tuple[GraphNode, float]] = []
        native_results: List[Tuple[GraphNode, float]] = []

        if self.search_mode in (SearchMode.SIDECAR, SearchMode.BOTH):
            filters = {"node_type": node_type} if node_type else None
            doc_results = self.retrieval_service.search(
                query, filters=filters, namespace=self.index_namespace, top_k=top_k
            )
            for doc, score in doc_results:
                # Prefer full-fidelity node from graph store
                full_node = self.graph_store.get_node(doc.doc_id)
                if full_node is not None:
                    sidecar_results.append((full_node, score))
                else:
                    # Fallback: lossy reconstruction (no history)
                    sidecar_results.append((self._doc_to_node(doc), score))

        if self.search_mode in (SearchMode.NATIVE, SearchMode.BOTH):
            native_results = self.graph_store.search_nodes(
                query, top_k=top_k, node_type=node_type
            )

        if self.search_mode == SearchMode.BOTH:
            return self._rrf_merge(sidecar_results, native_results, top_k)
        elif self.search_mode == SearchMode.NATIVE:
            return native_results
        else:
            return sidecar_results


    # ── RRF merge ──────────────────────────────────────────────────────

    def _rrf_merge(
        self,
        list_a: List[Tuple[GraphNode, float]],
        list_b: List[Tuple[GraphNode, float]],
        top_k: int,
    ) -> List[Tuple[GraphNode, float]]:
        """Merge two ranked lists using Reciprocal Rank Fusion.

        Score = sum of 1/(rrf_k + rank + 1) across lists for each node.
        Results are ordered by descending fused score and truncated to top_k.
        """
        if not list_a and not list_b:
            return []

        scores: dict[str, float] = {}
        node_map: dict[str, GraphNode] = {}
        for rank, (node, _) in enumerate(list_a):
            rrf = 1.0 / (self.rrf_k + rank + 1)
            scores[node.node_id] = scores.get(node.node_id, 0) + rrf
            node_map[node.node_id] = node
        for rank, (node, _) in enumerate(list_b):
            rrf = 1.0 / (self.rrf_k + rank + 1)
            scores[node.node_id] = scores.get(node.node_id, 0) + rrf
            node_map[node.node_id] = node
        sorted_ids = sorted(scores, key=lambda nid: -scores[nid])
        return [(node_map[nid], scores[nid]) for nid in sorted_ids[:top_k]]


    # ── Reindex ────────────────────────────────────────────────────────

    def reindex(self) -> int:
        """Rebuild the sidecar index from all active nodes in the graph store.

        Raises ValueError if retrieval_service is None (NATIVE mode).

        Returns:
            Number of nodes indexed.
        """
        if self.retrieval_service is None:
            raise ValueError("Cannot reindex: no retrieval_service configured (search_mode=native)")
        self.retrieval_service.clear(namespace=self.index_namespace)
        nodes = self.graph_store.list_nodes(include_inactive=False)
        count = 0
        for node in nodes:
            try:
                doc = self._node_to_doc(node)
                self.retrieval_service.add(doc, namespace=self.index_namespace)
                count += 1
            except Exception:
                logger.warning(f"Failed to index node {node.node_id} during reindex", exc_info=True)
        return count


    # ── Lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the wrapped graph store and sidecar retrieval service."""
        self.graph_store.close()
        if self.retrieval_service is not None:
            self.retrieval_service.close()
