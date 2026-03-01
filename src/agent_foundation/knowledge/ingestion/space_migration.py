"""
SpaceMigrationUtility: Reclassify existing knowledge into appropriate spaces.

Scans all knowledge pieces, metadata, and graph nodes across all entity scopes,
applies the SpaceClassifier, and updates the spaces field where changed. Only
auto_spaces from the ClassificationResult are applied; suggestion-mode rules
are ignored (existing knowledge is treated as already reviewed).

The migration is idempotent: running it twice produces the same result, with
the second run reporting zero updates.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 13.1, 13.2
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set

from agent_foundation.knowledge.ingestion.space_classifier import (
    SpaceClassifier,
)
from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


@dataclass
class MigrationReport:
    """Summary report produced by the migration utility.

    Attributes:
        pieces_updated: Number of knowledge pieces whose spaces were changed.
        metadata_updated: Number of metadata entries whose spaces were changed.
        graph_nodes_updated: Number of graph nodes whose spaces property was changed.
        graph_edges_updated: Number of graph edges whose spaces property was changed.
        space_counts: Mapping of space name to count of items classified into it.
        errors: List of error messages for items that failed processing.
        total_scanned: Total number of items scanned across all types.
    """

    pieces_updated: int = 0
    metadata_updated: int = 0
    graph_nodes_updated: int = 0
    graph_edges_updated: int = 0
    space_counts: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    total_scanned: int = 0


class SpaceMigrationUtility:
    """Utility to reclassify existing knowledge into appropriate spaces.

    Iterates all entity scopes (via metadata_store.list_entities()), processes
    pieces per entity plus global pieces, updates metadata spaces, and discovers
    graph nodes via relation traversal. Only auto_spaces from the classifier
    are applied; suggestion-mode rules are ignored.

    Args:
        kb: The KnowledgeBase providing access to all stores.
        classifier: The SpaceClassifier to determine spaces.
    """

    def __init__(self, kb: KnowledgeBase, classifier: SpaceClassifier):
        self._kb = kb
        self._classifier = classifier

    def migrate(self) -> MigrationReport:
        """Scan all knowledge and reclassify into spaces.

        Steps:
        1. Iterate all entity_ids from metadata_store.list_entities()
        2. For each entity, list_all pieces and classify
        3. List global pieces (entity_id=None) and classify
        4. Attempt to discover orphaned entity scopes not in metadata
        5. Classify and update metadata entries
        6. Discover and update graph nodes via get_relations()
        7. Produce a summary report

        Returns:
            MigrationReport with counts and errors.
        """
        report = MigrationReport()

        # Collect known entity_ids from metadata store
        try:
            known_entity_ids = self._kb.metadata_store.list_entities()
        except Exception as exc:
            msg = f"Failed to list entities from metadata store: {exc}"
            logger.error(msg)
            report.errors.append(msg)
            known_entity_ids = []

        processed_scopes: Set[str] = set()

        # Process pieces for each known entity
        for entity_id in known_entity_ids:
            self._migrate_pieces_for_scope(entity_id, report)
            processed_scopes.add(entity_id)

        # Process global pieces (entity_id=None)
        self._migrate_pieces_for_scope(None, report)

        # Attempt to discover orphaned entity scopes
        self._discover_and_migrate_orphaned_scopes(processed_scopes, report)

        # Migrate metadata
        self._migrate_metadata(known_entity_ids, report)

        # Migrate graph nodes
        self._migrate_graph_nodes(known_entity_ids, report)

        return report

    def _migrate_pieces_for_scope(
        self, entity_id, report: MigrationReport
    ) -> None:
        """Classify and update pieces for a single entity scope.

        Args:
            entity_id: The entity scope, or None for global pieces.
            report: The MigrationReport to update.
        """
        scope_label = entity_id or "(global)"
        try:
            pieces = self._kb.piece_store.list_all(entity_id=entity_id)
        except Exception as exc:
            msg = f"Failed to list pieces for scope '{scope_label}': {exc}"
            logger.error(msg)
            report.errors.append(msg)
            return

        for piece in pieces:
            report.total_scanned += 1
            try:
                result = self._classifier.classify_piece(piece)
                new_spaces = result.auto_spaces

                # Track space counts
                for sp in new_spaces:
                    report.space_counts[sp] = report.space_counts.get(sp, 0) + 1

                # Update only if spaces changed
                if sorted(new_spaces) != sorted(piece.spaces):
                    piece.spaces = new_spaces
                    piece.space = new_spaces[0] if new_spaces else "main"
                    self._kb.piece_store.update(piece)
                    report.pieces_updated += 1
            except Exception as exc:
                msg = f"Error migrating piece '{piece.piece_id}' in scope '{scope_label}': {exc}"
                logger.warning(msg)
                report.errors.append(msg)

    def _discover_and_migrate_orphaned_scopes(
        self, processed_scopes: Set[str], report: MigrationReport
    ) -> None:
        """Discover entity scopes not known to the metadata store.

        For file-based stores, uses retrieval_service.namespaces().
        For LanceDB, uses a distinct query on entity_id.

        Args:
            processed_scopes: Set of entity_ids already processed.
            report: The MigrationReport to update.
        """
        orphaned_scopes: Set[str] = set()

        # Try file-based store discovery via retrieval_service.namespaces()
        piece_store = self._kb.piece_store
        retrieval_service = getattr(piece_store, "retrieval_service", None)
        if retrieval_service is not None and hasattr(retrieval_service, "namespaces"):
            try:
                namespaces = retrieval_service.namespaces()
                for ns in namespaces:
                    if ns and ns not in processed_scopes:
                        orphaned_scopes.add(ns)
            except Exception as exc:
                msg = f"Failed to discover namespaces from retrieval service: {exc}"
                logger.warning(msg)
                report.errors.append(msg)

        # Try LanceDB discovery via distinct entity_id query
        table = getattr(piece_store, "_table", None)
        if table is not None:
            try:
                # Fetch a large sample and extract distinct entity_ids
                records = table.search().limit(100000).to_list()
                for record in records:
                    eid = record.get("entity_id")
                    if eid and eid != "__global__" and eid not in processed_scopes:
                        orphaned_scopes.add(eid)
            except Exception as exc:
                msg = f"Failed to discover entity scopes from LanceDB: {exc}"
                logger.warning(msg)
                report.errors.append(msg)

        # Process orphaned scopes
        for entity_id in orphaned_scopes:
            self._migrate_pieces_for_scope(entity_id, report)

    def _migrate_metadata(
        self, entity_ids: List[str], report: MigrationReport
    ) -> None:
        """Classify and update metadata entries.

        Args:
            entity_ids: List of entity_ids to process.
            report: The MigrationReport to update.
        """
        for entity_id in entity_ids:
            try:
                metadata = self._kb.metadata_store.get_metadata(entity_id)
                if metadata is None:
                    continue

                report.total_scanned += 1
                result = self._classifier.classify_metadata(metadata)
                new_spaces = result.auto_spaces

                # Track space counts
                for sp in new_spaces:
                    report.space_counts[sp] = report.space_counts.get(sp, 0) + 1

                # Update only if spaces changed
                if sorted(new_spaces) != sorted(metadata.spaces):
                    metadata.spaces = new_spaces
                    self._kb.metadata_store.save_metadata(metadata)
                    report.metadata_updated += 1
            except Exception as exc:
                msg = f"Error migrating metadata for entity '{entity_id}': {exc}"
                logger.warning(msg)
                report.errors.append(msg)

    def _migrate_graph_nodes(
        self, entity_ids: List[str], report: MigrationReport
    ) -> None:
        """Discover and update graph nodes via relation traversal.

        Since EntityGraphStore has no list_nodes() method, we discover nodes
        by calling get_relations() for each known entity and collecting
        unique node IDs from both source and target of each edge.

        Args:
            entity_ids: List of entity_ids to traverse from.
            report: The MigrationReport to update.
        """
        discovered_node_ids: Set[str] = set()
        discovered_edges: list = []

        # Discover nodes and edges via relations from known entities
        for entity_id in entity_ids:
            try:
                relations = self._kb.graph_store.get_relations(
                    entity_id, direction="both"
                )
                for edge in relations:
                    discovered_node_ids.add(edge.source_id)
                    discovered_node_ids.add(edge.target_id)
                    discovered_edges.append(edge)
            except Exception as exc:
                msg = f"Error getting relations for entity '{entity_id}': {exc}"
                logger.warning(msg)
                report.errors.append(msg)

        # Update graph nodes
        processed_nodes: Set[str] = set()
        for node_id in discovered_node_ids:
            if node_id in processed_nodes:
                continue
            processed_nodes.add(node_id)

            try:
                node = self._kb.graph_store.get_node(node_id)
                if node is None:
                    continue

                report.total_scanned += 1
                result = self._classifier.classify_graph_node(node)
                new_spaces = result.auto_spaces

                # Track space counts
                for sp in new_spaces:
                    report.space_counts[sp] = report.space_counts.get(sp, 0) + 1

                current_spaces = node.properties.get("spaces", ["main"])
                if sorted(new_spaces) != sorted(current_spaces):
                    node.properties["spaces"] = new_spaces
                    self._kb.graph_store.add_node(node)
                    report.graph_nodes_updated += 1
            except Exception as exc:
                msg = f"Error migrating graph node '{node_id}': {exc}"
                logger.warning(msg)
                report.errors.append(msg)

        # Update graph edges
        processed_edge_keys: Set[tuple] = set()
        for edge in discovered_edges:
            edge_key = (edge.source_id, edge.target_id, edge.edge_type)
            if edge_key in processed_edge_keys:
                continue
            processed_edge_keys.add(edge_key)

            try:
                report.total_scanned += 1

                # Determine edge spaces based on connected node spaces
                source_node = self._kb.graph_store.get_node(edge.source_id)
                target_node = self._kb.graph_store.get_node(edge.target_id)

                source_spaces = set(
                    source_node.properties.get("spaces", ["main"])
                    if source_node
                    else ["main"]
                )
                target_spaces = set(
                    target_node.properties.get("spaces", ["main"])
                    if target_node
                    else ["main"]
                )
                # Edge belongs to the union of its endpoint spaces
                new_spaces = sorted(source_spaces | target_spaces)

                # Track space counts
                for sp in new_spaces:
                    report.space_counts[sp] = report.space_counts.get(sp, 0) + 1

                current_spaces = edge.properties.get("spaces", ["main"])
                if sorted(new_spaces) != sorted(current_spaces):
                    edge.properties["spaces"] = new_spaces
                    self._kb.graph_store.add_relation(edge)
                    report.graph_edges_updated += 1
            except Exception as exc:
                msg = (
                    f"Error migrating graph edge "
                    f"'{edge.source_id}' -> '{edge.target_id}' ({edge.edge_type}): {exc}"
                )
                logger.warning(msg)
                report.errors.append(msg)
