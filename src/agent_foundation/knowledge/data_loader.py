"""
KnowledgeDataLoader — loads knowledge data from a JSON file into a KnowledgeBase.

Reads a structured JSON file with three sections (metadata, pieces, graph) and
populates the corresponding stores in a KnowledgeBase instance.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 9.1, 9.2, 9.3, 9.4
"""
import json
import logging
from typing import Any, Dict, List

from science_modeling_tools.knowledge.knowledge_base import KnowledgeBase
from science_modeling_tools.knowledge.models.entity_metadata import EntityMetadata
from science_modeling_tools.knowledge.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from rich_python_utils.service_utils.graph_service.graph_node import (
    GraphNode,
    GraphEdge,
)

logger = logging.getLogger(__name__)

# Required top-level sections in the knowledge data file
REQUIRED_SECTIONS = {"metadata", "pieces", "graph"}


class KnowledgeDataLoader:
    """Loads knowledge data from a JSON file into a KnowledgeBase."""

    @staticmethod
    def load(kb: KnowledgeBase, file_path: str) -> Dict[str, int]:
        """Load knowledge data from a JSON file into the given KnowledgeBase.

        Args:
            kb: The KnowledgeBase to populate.
            file_path: Path to the JSON knowledge data file.

        Returns:
            Dict with counts: {"metadata": N, "pieces": N, "graph_nodes": N, "graph_edges": N}

        Raises:
            FileNotFoundError: If file_path does not exist.
            ValueError: If the file is malformed or missing required sections.
        """
        # Read and parse JSON
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Knowledge data file not found: {file_path}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Knowledge data file is not valid JSON ({file_path}): {e}"
            )

        # Validate required sections
        if not isinstance(data, dict):
            raise ValueError(
                f"Knowledge data file must contain a JSON object ({file_path})"
            )
        missing = REQUIRED_SECTIONS - set(data.keys())
        if missing:
            raise ValueError(
                f"Knowledge data file is missing required sections: "
                f"{', '.join(sorted(missing))} ({file_path})"
            )

        # Load each section
        metadata_count = KnowledgeDataLoader._load_metadata(
            kb, data["metadata"]
        )
        pieces_count = KnowledgeDataLoader._load_pieces(kb, data["pieces"])
        node_count, edge_count = KnowledgeDataLoader._load_graph(
            kb, data["graph"]
        )

        return {
            "metadata": metadata_count,
            "pieces": pieces_count,
            "graph_nodes": node_count,
            "graph_edges": edge_count,
        }

    @staticmethod
    def _load_metadata(
        kb: KnowledgeBase, metadata_section: Dict[str, Any]
    ) -> int:
        """Load metadata entries. Each key is an entity_id mapped to properties.

        Expected format:
            {
                "<entity_id>": {
                    "entity_type": "<type>",
                    "properties": { "<key>": "<value>" }
                }
            }

        Args:
            kb: The KnowledgeBase whose metadata_store to populate.
            metadata_section: The metadata section from the data file.

        Returns:
            Count of metadata entries loaded.
        """
        count = 0
        for entity_id, entry in metadata_section.items():
            try:
                metadata = EntityMetadata(
                    entity_id=entity_id,
                    entity_type=entry.get("entity_type", "unknown"),
                    properties=entry.get("properties", {}),
                )
                kb.metadata_store.save_metadata(metadata)
                count += 1
            except Exception as e:
                logger.warning(
                    "Skipping metadata entry '%s': %s", entity_id, e
                )
        return count

    @staticmethod
    def _load_pieces(kb: KnowledgeBase, pieces_section: List[Any]) -> int:
        """Load knowledge pieces. Each entry is a KnowledgePiece dict.

        Validates knowledge_type against KnowledgeType enum — skips and logs
        a warning if invalid. Delegates sensitive content validation to
        kb.add_piece() which raises ValueError for sensitive content.

        Args:
            kb: The KnowledgeBase to add pieces to.
            pieces_section: The pieces section (list) from the data file.

        Returns:
            Count of pieces successfully loaded.
        """
        count = 0
        for i, entry in enumerate(pieces_section):
            piece_id = entry.get("piece_id", f"piece-{i}")
            try:
                # Validate knowledge_type is a valid enum value
                knowledge_type_str = entry.get("knowledge_type", "fact")
                try:
                    KnowledgeType(knowledge_type_str)
                except ValueError:
                    logger.warning(
                        "Skipping piece '%s': invalid knowledge_type '%s'",
                        piece_id,
                        knowledge_type_str,
                    )
                    continue

                # Create KnowledgePiece via from_dict
                piece = KnowledgePiece.from_dict(entry)

                # Add to KB — this handles sensitive content validation
                kb.add_piece(piece)
                count += 1
            except ValueError as e:
                logger.warning("Skipping piece '%s': %s", piece_id, e)
            except Exception as e:
                logger.warning("Skipping piece '%s': %s", piece_id, e)
        return count

    @staticmethod
    def _load_graph(
        kb: KnowledgeBase, graph_section: Dict[str, Any]
    ) -> tuple:
        """Load graph nodes and edges.

        Expected format:
            {
                "nodes": [ { "node_id": ..., "node_type": ..., ... } ],
                "edges": [ { "source_id": ..., "target_id": ..., "edge_type": ..., ... } ]
            }

        Args:
            kb: The KnowledgeBase whose graph_store to populate.
            graph_section: The graph section from the data file.

        Returns:
            Tuple of (node_count, edge_count).
        """
        node_count = 0
        edge_count = 0

        # Load nodes
        nodes = graph_section.get("nodes", [])
        for i, node_data in enumerate(nodes):
            try:
                node = GraphNode.from_dict(node_data)
                kb.graph_store.add_node(node)
                node_count += 1
            except Exception as e:
                logger.warning("Skipping graph node %d: %s", i, e)

        # Load edges
        edges = graph_section.get("edges", [])
        for i, edge_data in enumerate(edges):
            try:
                edge = GraphEdge.from_dict(edge_data)
                kb.graph_store.add_relation(edge)
                edge_count += 1
            except Exception as e:
                logger.warning("Skipping graph edge %d: %s", i, e)

        return node_count, edge_count
