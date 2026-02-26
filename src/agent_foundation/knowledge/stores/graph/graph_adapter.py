"""
GraphServiceEntityGraphStore — EntityGraphStore adapter backed by GraphServiceBase.

Implements the EntityGraphStore ABC by delegating all operations to a
general-purpose GraphServiceBase instance. Since the EntityGraphStore ABC
uses GraphNode/GraphEdge directly, this adapter is a thin delegation layer
with no model conversion needed.

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5
"""
from typing import List, Optional, Tuple

from attr import attrs, attrib

from rich_python_utils.service_utils.graph_service.graph_service_base import (
    GraphServiceBase,
)
from rich_python_utils.service_utils.graph_service.graph_node import (
    GraphEdge,
    GraphNode,
)
from science_modeling_tools.knowledge.stores.graph.base import EntityGraphStore


@attrs
class GraphServiceEntityGraphStore(EntityGraphStore):
    """EntityGraphStore backed by any GraphServiceBase.

    Since EntityGraphStore ABC now uses GraphNode/GraphEdge directly,
    this adapter is a thin delegation layer with no model conversion.

    Attributes:
        graph_service: The underlying graph service instance.
    """

    graph_service: GraphServiceBase = attrib()

    def add_node(self, node: GraphNode) -> None:
        """Add or update a node in the graph.

        Delegates directly to the graph service's add_node method.

        Args:
            node: The GraphNode to add or update.
        """
        self.graph_service.add_node(node)

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by its ID.

        Delegates directly to the graph service's get_node method.

        Args:
            node_id: The unique identifier of the node.

        Returns:
            The GraphNode if found, or None if not found.
        """
        return self.graph_service.get_node(node_id)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its associated edges.

        Delegates to the graph service's remove_node method, which
        cascade-deletes all edges involving this node.

        Args:
            node_id: The unique identifier of the node to remove.

        Returns:
            True if the node existed and was removed, False if not found.
        """
        return self.graph_service.remove_node(node_id)

    def add_relation(self, relation: GraphEdge) -> None:
        """Add an edge between two existing nodes.

        Delegates directly to the graph service's add_edge method.

        Args:
            relation: The GraphEdge to add.

        Raises:
            ValueError: If either the source or target node does not exist.
        """
        self.graph_service.add_edge(relation)

    def get_relations(
        self,
        node_id: str,
        relation_type: str = None,
        direction: str = "outgoing",
    ) -> List[GraphEdge]:
        """Get edges for a node, optionally filtered by type and direction.

        Delegates directly to the graph service's get_edges method.

        Args:
            node_id: The node whose edges to retrieve.
            relation_type: If specified, only return edges of this type.
            direction: Direction filter ("outgoing", "incoming", or "both").

        Returns:
            A list of GraphEdge objects matching the filter criteria.
        """
        return self.graph_service.get_edges(
            node_id, edge_type=relation_type, direction=direction
        )

    def remove_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
    ) -> bool:
        """Remove a specific edge between two nodes.

        Delegates directly to the graph service's remove_edge method.

        Args:
            source_id: The source node ID of the edge.
            target_id: The target node ID of the edge.
            relation_type: The type of the edge to remove.

        Returns:
            True if the edge existed and was removed, False if not found.
        """
        return self.graph_service.remove_edge(source_id, target_id, relation_type)

    def get_neighbors(
        self,
        node_id: str,
        relation_type: str = None,
        depth: int = 1,
    ) -> List[Tuple[GraphNode, int]]:
        """Get neighboring nodes up to a given depth via graph traversal.

        Delegates directly to the graph service's get_neighbors method.

        Args:
            node_id: The starting node for traversal.
            relation_type: If specified, only follow edges of this type.
            depth: Maximum traversal depth.

        Returns:
            A list of (GraphNode, depth) tuples where depth indicates how
            many hops from the source node.
        """
        return self.graph_service.get_neighbors(
            node_id, edge_type=relation_type, depth=depth
        )

    def close(self):
        """Close the underlying graph service."""
        self.graph_service.close()
