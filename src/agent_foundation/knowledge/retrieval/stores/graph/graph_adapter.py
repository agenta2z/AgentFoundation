"""
GraphServiceEntityGraphStore — EntityGraphStore adapter backed by GraphServiceBase.

Implements the EntityGraphStore ABC by delegating all operations to a
general-purpose GraphServiceBase instance. Since the EntityGraphStore ABC
uses GraphNode/GraphEdge directly, this adapter is a thin delegation layer
with no model conversion needed.

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5
"""
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from attr import attrs, attrib

from rich_python_utils.service_utils.data_operation_record import (
    DataOperationRecord,
    generate_operation_id,
)
from rich_python_utils.service_utils.graph_service.graph_service_base import (
    GraphServiceBase,
)
from rich_python_utils.service_utils.graph_service.graph_node import (
    GraphEdge,
    GraphNode,
)
from agent_foundation.knowledge.retrieval.stores.graph.base import EntityGraphStore


@attrs
class GraphServiceEntityGraphStore(EntityGraphStore):
    """EntityGraphStore backed by any GraphServiceBase.

    Since EntityGraphStore ABC now uses GraphNode/GraphEdge directly,
    this adapter is a thin delegation layer with no model conversion.

    Attributes:
        graph_service: The underlying graph service instance.
    """

    graph_service: GraphServiceBase = attrib()

    def add_node(
        self,
        node: GraphNode,
        operation_id: Optional[str] = None,
    ) -> None:
        """Add or update a node in the graph with history tracking.

        If the node already exists (upsert), appends an UPDATE record.
        If new, appends an ADD record.

        Args:
            node: The GraphNode to add or update.
            operation_id: Optional shared operation ID for batch grouping.
        """
        now = datetime.now(timezone.utc).isoformat()
        op_id = operation_id or generate_operation_id("GraphStore", "add_node")
        existing = self.graph_service.get_node(node.node_id)
        if existing:
            node.history = existing.history + node.history
            node.history.append(DataOperationRecord(
                operation="update",
                timestamp=now,
                operation_id=op_id,
                source="GraphServiceEntityGraphStore.add_node",
                properties_before=dict(existing.properties),
                properties_after=dict(node.properties),
            ))
        else:
            node.history.append(DataOperationRecord(
                operation="add",
                timestamp=now,
                operation_id=op_id,
                source="GraphServiceEntityGraphStore.add_node",
            ))
        self.graph_service.add_node(node)

    def get_node(
        self,
        node_id: str,
        include_inactive: bool = False,
    ) -> Optional[GraphNode]:
        """Get a node by its ID, filtering out soft-deleted nodes.

        Args:
            node_id: The unique identifier of the node.
            include_inactive: If True, return soft-deleted nodes too.

        Returns:
            The GraphNode if found (and active), or None.
        """
        node = self.graph_service.get_node(node_id)
        if node is None:
            return None
        if not include_inactive and not node.is_active:
            return None
        return node

    def remove_node(
        self,
        node_id: str,
        operation_id: Optional[str] = None,
    ) -> bool:
        """Soft-delete a node and cascade to connected edges.

        Sets is_active=False on the node and all connected edges,
        appending DELETE history records with a shared operation_id.

        Args:
            node_id: The unique identifier of the node to remove.
            operation_id: Optional shared operation ID for batch grouping.

        Returns:
            True if the node existed and was soft-deleted, False if not
            found or already inactive.
        """
        node = self.graph_service.get_node(node_id)
        if node is None or not node.is_active:
            return False

        now = datetime.now(timezone.utc).isoformat()
        op_id = operation_id or generate_operation_id("GraphStore", "remove_node")

        # Soft-delete the node
        node.is_active = False
        node.history.append(DataOperationRecord(
            operation="delete",
            timestamp=now,
            operation_id=op_id,
            source="GraphServiceEntityGraphStore.remove_node",
            details={"delete_mode": "soft"},
        ))
        self.graph_service.add_node(node)  # upsert with updated state

        # Cascade: soft-delete all connected edges
        edges = self.graph_service.get_edges(node_id, direction="both")
        for edge in edges:
            if not edge.is_active:
                continue
            edge.is_active = False
            edge.history.append(DataOperationRecord(
                operation="delete",
                timestamp=now,
                operation_id=op_id,
                source="GraphServiceEntityGraphStore.remove_node",
                details={"delete_mode": "soft", "cascade_from": node_id},
            ))
            # Re-save edge — strategy varies by backend but add_edge handles it
            self.graph_service.remove_edge(edge.source_id, edge.target_id, edge.edge_type)
            self.graph_service.add_edge(edge)

        return True

    def add_relation(
        self,
        relation: GraphEdge,
        operation_id: Optional[str] = None,
    ) -> None:
        """Add an edge between two existing nodes with history tracking.

        Args:
            relation: The GraphEdge to add.
            operation_id: Optional shared operation ID for batch grouping.

        Raises:
            ValueError: If either the source or target node does not exist.
        """
        now = datetime.now(timezone.utc).isoformat()
        op_id = operation_id or generate_operation_id("GraphStore", "add_relation")
        relation.history.append(DataOperationRecord(
            operation="add",
            timestamp=now,
            operation_id=op_id,
            source="GraphServiceEntityGraphStore.add_relation",
        ))
        self.graph_service.add_edge(relation)

    def get_relations(
        self,
        node_id: str,
        relation_type: str = None,
        direction: str = "outgoing",
        include_inactive: bool = False,
    ) -> List[GraphEdge]:
        """Get edges for a node, filtering out soft-deleted edges.

        Args:
            node_id: The node whose edges to retrieve.
            relation_type: If specified, only return edges of this type.
            direction: Direction filter ("outgoing", "incoming", or "both").
            include_inactive: If True, include soft-deleted edges.

        Returns:
            A list of active GraphEdge objects matching the filter criteria.
        """
        edges = self.graph_service.get_edges(
            node_id, edge_type=relation_type, direction=direction
        )
        if include_inactive:
            return edges
        return [e for e in edges if e.is_active]

    def remove_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        operation_id: Optional[str] = None,
    ) -> bool:
        """Soft-delete a specific edge between two nodes.

        Fetches the edge, sets is_active=False, appends a DELETE history
        record, and re-saves via remove+add (to handle backends like Neo4j
        that use CREATE instead of MERGE for edges).

        Args:
            source_id: The source node ID of the edge.
            target_id: The target node ID of the edge.
            relation_type: The type of the edge to remove.
            operation_id: Optional shared operation ID for batch grouping.

        Returns:
            True if the edge existed and was soft-deleted, False if not
            found or already inactive.
        """
        # Find the specific edge
        edges = self.graph_service.get_edges(source_id, edge_type=relation_type, direction="outgoing")
        target_edge = None
        for e in edges:
            if e.target_id == target_id and e.edge_type == relation_type:
                target_edge = e
                break
        if target_edge is None or not target_edge.is_active:
            return False

        now = datetime.now(timezone.utc).isoformat()
        op_id = operation_id or generate_operation_id("GraphStore", "remove_relation")
        target_edge.is_active = False
        target_edge.history.append(DataOperationRecord(
            operation="delete",
            timestamp=now,
            operation_id=op_id,
            source="GraphServiceEntityGraphStore.remove_relation",
            details={"delete_mode": "soft"},
        ))
        # Delete and re-create to update stored state
        self.graph_service.remove_edge(source_id, target_id, relation_type)
        self.graph_service.add_edge(target_edge)
        return True

    def get_neighbors(
        self,
        node_id: str,
        relation_type: str = None,
        depth: int = 1,
    ) -> List[Tuple[GraphNode, int]]:
        """Get neighboring nodes up to a given depth via graph traversal.

        Filters out inactive nodes and inactive edges from the results.

        Args:
            node_id: The starting node for traversal.
            relation_type: If specified, only follow edges of this type.
            depth: Maximum traversal depth.

        Returns:
            A list of (GraphNode, depth) tuples where depth indicates how
            many hops from the source node. Only active nodes returned.
        """
        neighbors = self.graph_service.get_neighbors(
            node_id, edge_type=relation_type, depth=depth
        )
        return [(n, d) for n, d in neighbors if n.is_active]

    def list_nodes(
        self,
        node_type: Optional[str] = None,
        include_inactive: bool = False,
    ) -> List[GraphNode]:
        """List all nodes, optionally filtered by type and active status.

        Delegates to graph_service.list_nodes() and applies is_active filter.

        Args:
            node_type: Optional node type filter. If None, returns all types.
            include_inactive: If True, include soft-deleted nodes.

        Returns:
            A list of GraphNode objects matching the filter criteria.
        """
        nodes = self.graph_service.list_nodes(node_type=node_type)
        if include_inactive:
            return nodes
        return [n for n in nodes if n.is_active]

    @property
    def supports_semantic_search(self) -> bool:
        """Delegate to the underlying graph service's ``supports_search``.

        Uses ``getattr`` with a ``False`` fallback for cross-package
        compatibility — the adapter (AgentFoundation) and the graph service
        (RichPythonUtils) may be at different versions.
        """
        return getattr(self.graph_service, "supports_search", False)

    def search_nodes(
        self,
        query: str,
        top_k: int = 5,
        node_type: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> List[Tuple[GraphNode, float]]:
        """Delegate search to the underlying graph service.

        Filters results to only include active nodes, consistent with
        how ``get_node()``, ``list_nodes()``, and ``get_neighbors()``
        filter inactive nodes.

        Args:
            query: The search query string.
            top_k: Maximum number of results.
            node_type: Optional node type filter.
            namespace: Optional namespace to scope the search.

        Returns:
            List of ``(GraphNode, score)`` tuples for active nodes only.
        """
        results = self.graph_service.search_nodes(
            query, top_k=top_k, node_type=node_type, namespace=namespace
        )
        return [(node, score) for node, score in results if node.is_active]

    def close(self):
        """Close the underlying graph service."""
        self.graph_service.close()


