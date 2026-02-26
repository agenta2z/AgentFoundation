"""
EntityGraphStore abstract base class.

Defines the abstract interface for entity graph storage backends. All graph
store implementations must implement this interface.

The EntityGraphStore manages a directed graph of entities (nodes) and their
typed relationships (edges). It supports CRUD operations on nodes and
edges, as well as graph traversal via neighbor queries.

Uses GraphNode and GraphEdge from SciencePythonUtils graph_service directly,
eliminating the need for domain-specific EntityNode/EntityRelation models.

Requirements: 2.1
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from rich_python_utils.service_utils.graph_service.graph_node import (
    GraphEdge,
    GraphNode,
)


class EntityGraphStore(ABC):
    """Abstract base class for entity graph storage backends.

    Provides the interface for CRUD operations on graph nodes and edges,
    plus traversal queries. Uses GraphNode and GraphEdge from SciencePythonUtils
    directly.

    All implementations must support:
    - Adding and retrieving nodes (GraphNode)
    - Removing nodes (with cascade deletion of associated edges)
    - Adding and retrieving edges between existing nodes (GraphEdge)
    - Removing specific edges
    - Traversing neighbors up to a given depth

    The ``close()`` method is a concrete no-op by default. Subclasses that hold
    external connections (e.g., Neo4j) should override it to release resources.
    """

    @abstractmethod
    def add_node(self, node: GraphNode) -> None:
        """Add or update a node in the graph.

        If a node with the same node_id already exists, it is updated
        with the new node's data.

        Args:
            node: The GraphNode to add or update.
        """
        ...

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by its ID.

        Args:
            node_id: The unique identifier of the node.

        Returns:
            The GraphNode if found, or None if not found.
        """
        ...

    @abstractmethod
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its associated edges.

        When a node is removed, all edges where this node is either
        the source or the target are also removed (cascade deletion).

        Args:
            node_id: The unique identifier of the node to remove.

        Returns:
            True if the node existed and was removed, False if not found.
        """
        ...

    @abstractmethod
    def add_relation(self, relation: GraphEdge) -> None:
        """Add an edge between two existing nodes.

        Both the source and target nodes must already exist in the graph.

        Args:
            relation: The GraphEdge to add.

        Raises:
            ValueError: If either the source or target node does not exist.
        """
        ...

    @abstractmethod
    def get_relations(
        self,
        node_id: str,
        relation_type: str = None,
        direction: str = "outgoing",
    ) -> List[GraphEdge]:
        """Get edges for a node, optionally filtered by type and direction.

        Args:
            node_id: The node whose edges to retrieve.
            relation_type: If specified, only return edges of this type.
                           If None, return all edges.
            direction: Direction filter. One of:
                       - "outgoing": edges where node_id is the source
                       - "incoming": edges where node_id is the target
                       - "both": edges in either direction

        Returns:
            A list of GraphEdge objects matching the filter criteria.
        """
        ...

    @abstractmethod
    def remove_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
    ) -> bool:
        """Remove a specific edge between two nodes.

        Args:
            source_id: The source node ID of the edge.
            target_id: The target node ID of the edge.
            relation_type: The type of the edge to remove.

        Returns:
            True if the edge existed and was removed, False if not found.
        """
        ...

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        relation_type: str = None,
        depth: int = 1,
    ) -> List[Tuple[GraphNode, int]]:
        """Get neighboring nodes up to a given depth via graph traversal.

        Performs a breadth-first traversal from the given node, following
        outgoing edges. Returns all reachable nodes within the specified
        depth, along with their distance from the source.

        Args:
            node_id: The starting node for traversal.
            relation_type: If specified, only follow edges of this type.
                           If None, follow all edge types.
            depth: Maximum traversal depth (1 = direct neighbors only,
                   2 = neighbors and their neighbors, etc.).

        Returns:
            A list of (GraphNode, depth) tuples where depth indicates how
            many hops from the source node (1 = direct neighbor,
            2 = neighbor's neighbor, etc.). The source node itself is not
            included in the results.
        """
        ...

    def close(self):
        """Close any underlying connections.

        Default no-op for file-based and in-memory stores. Override for
        stores with external connections (e.g., Neo4j) to release resources.
        """
        pass
