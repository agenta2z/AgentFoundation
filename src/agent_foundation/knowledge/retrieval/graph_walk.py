"""
Unified graph walk module — seed finding and graph traversal.

Consolidates the duplicated BFS logic from ``_extract_graph_knowledge`` (L3b)
and ``_extract_search_graph_knowledge`` (L3a) into two phases:

1. **Seed finding** — ``find_search_seeds()`` and ``find_identity_seeds()``
   discover starting nodes via semantic search or identity lookup.
2. **Graph walk** — ``graph_walk()`` takes seed nodes and produces graph
   context entries with depth-decayed scoring.
3. **Merge** — ``merge_graph_contexts()`` deduplicates entries from both paths.

Requirements: 1.1, 1.2, 1.3, 1.4, 2.1–2.4, 3.1–3.4
"""
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from rich_python_utils.service_utils.graph_service.graph_node import GraphNode

if TYPE_CHECKING:
    from agent_foundation.knowledge.retrieval.stores.graph.base import EntityGraphStore
    from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeedNode:
    """A graph node identified as a starting point for traversal.

    Attributes:
        node: The underlying GraphNode object.
        score: Relevance score (0.0–1.0). Search seeds use the search
               relevance score; identity seeds always use 1.0.
        source: Discovery method tag — ``"search"`` or ``"identity"``.
    """

    node: GraphNode
    score: float
    source: str  # "search" | "identity"


def _node_passes_space_filter(node: GraphNode, spaces: Optional[List[str]]) -> bool:
    """Check if a node's spaces intersect with the requested spaces (OR semantics).

    Nodes without a ``spaces`` property default to ``["main"]``.

    Args:
        node: The graph node to check.
        spaces: Requested space filter. ``None`` means no filtering.

    Returns:
        True if the node should be kept.
    """
    if not spaces:
        return True
    node_spaces = set(node.properties.get("spaces", ["main"]))
    return bool(node_spaces & set(spaces))


def find_search_seeds(
    graph_store: "EntityGraphStore",
    query: str,
    top_k: int = 5,
    spaces: Optional[List[str]] = None,
) -> List[SeedNode]:
    """Discover seed nodes via semantic search on the graph store.

    Args:
        graph_store: The graph store (must support semantic search).
        query: The search query string.
        top_k: Maximum number of seed nodes to return.
        spaces: Optional space filter (OR semantics).

    Returns:
        List of SeedNode with ``source="search"``. Empty if the store
        doesn't support semantic search or query is empty/whitespace.
    """
    if not query or not query.strip():
        return []
    if not graph_store.supports_semantic_search:
        return []

    try:
        search_results: List[Tuple[GraphNode, float]] = graph_store.search_nodes(
            query, top_k
        )
    except Exception:
        logger.warning(
            "graph_store.search_nodes() failed for query=%r; returning empty seeds",
            query,
            exc_info=True,
        )
        return []

    seeds: List[SeedNode] = []
    for node, score in search_results:
        if _node_passes_space_filter(node, spaces):
            seeds.append(SeedNode(node=node, score=score, source="search"))
    return seeds


def find_identity_seeds(
    graph_store: "EntityGraphStore",
    entity_id: Optional[str],
    spaces: Optional[List[str]] = None,
) -> List[SeedNode]:
    """Discover a seed node by direct entity_id lookup.

    The identity node is always accepted regardless of its own spaces
    property — it represents the active user/entity and must not be
    filtered out.  Space filtering is applied to *neighbors* during
    ``graph_walk()`` instead.

    Args:
        graph_store: The graph store.
        entity_id: The entity ID to look up.
        spaces: Optional space filter (OR semantics).  Stored but NOT
                applied to the identity node itself.

    Returns:
        List containing a single SeedNode with ``score=1.0`` and
        ``source="identity"``, or empty list if not found.
    """
    if not entity_id:
        return []

    node = graph_store.get_node(entity_id)
    if node is None:
        return []

    # NOTE: intentionally no space filtering on the identity node.
    # The identity node is the user's own entity — always a valid seed.
    # Space filtering is applied to neighbors during graph_walk().

    return [SeedNode(node=node, score=1.0, source="identity")]


def _should_skip_piece(
    piece_id: str,
    already_retrieved_piece_ids: Optional[Dict[str, str]],
    ignore_already_retrieved: Union[bool, Tuple[str, ...], List[str]],
) -> bool:
    """Check if a graph-linked piece should be skipped (already retrieved).

    Args:
        piece_id: The piece_id from the graph edge properties.
        already_retrieved_piece_ids: Dict mapping piece_id to info_type
            for pieces already found by Layer 2 search, or None.
        ignore_already_retrieved: Controls dedup behavior:
            - False: no dedup
            - True: skip all pieces in already_retrieved_piece_ids
            - Tuple/List of str: skip only pieces whose info_type is in this list

    Returns:
        True if the piece should be skipped.
    """
    if not already_retrieved_piece_ids or piece_id not in already_retrieved_piece_ids:
        return False
    if ignore_already_retrieved is True:
        return True
    if isinstance(ignore_already_retrieved, (list, tuple)):
        piece_info_type = already_retrieved_piece_ids[piece_id]
        return piece_info_type in ignore_already_retrieved
    return False


def graph_walk(
    graph_store: "EntityGraphStore",
    piece_store: "KnowledgePieceStore",
    seeds: List[SeedNode],
    traversal_depth: int = 1,
    already_retrieved_piece_ids: Optional[Dict[str, str]] = None,
    spaces: Optional[List[str]] = None,
    ignore_already_retrieved: Union[bool, Tuple[str, ...], List[str]] = False,
) -> List[Dict[str, Any]]:
    """Perform a unified BFS walk from seed nodes, producing graph context entries.

    For each seed:
    1. Emit a depth-0 entry with relation_type derived from seed.source
       ("SEARCH_HIT" for search, "IDENTITY" for identity).
    2. Call ``graph_store.get_neighbors(seed.node.node_id, depth=traversal_depth)``.
    3. Filter neighbors by spaces (OR semantics) if provided.
    4. For each neighbor at depth D, compute score = ``seed.score × 1/(D+1)``.
    5. For depth-1 neighbors, look up the edge relation and extract linked pieces.
       ``get_relations(seed.node_id)`` is called ONCE per seed and cached.

    Args:
        graph_store: The graph store for neighbor traversal and edge lookup.
        piece_store: The piece store for looking up linked pieces.
        seeds: List of SeedNode objects to walk from.
        traversal_depth: Maximum BFS depth.
        already_retrieved_piece_ids: Piece IDs already found by L2 (for dedup).
        spaces: Optional space filter (OR semantics).
        ignore_already_retrieved: Controls piece dedup behavior.

    Returns:
        List of graph context entry dicts. Entries from different seeds
        are NOT merged (preserves per-seed provenance).

    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8
    """
    graph_context: List[Dict[str, Any]] = []

    for seed in seeds:
        # Depth-0 entry for the seed itself
        relation_type_0 = "SEARCH_HIT" if seed.source == "search" else "IDENTITY"
        depth_0_entry: Dict[str, Any] = {
            "relation_type": relation_type_0,
            "target_node_id": seed.node.node_id,
            "target_label": seed.node.label,
            "piece": None,
            "depth": 0,
            "score": seed.score,
        }
        graph_context.append(depth_0_entry)

        # Walk neighbors from the seed node
        try:
            neighbors = graph_store.get_neighbors(
                seed.node.node_id, depth=traversal_depth
            )
        except Exception:
            logger.warning(
                "graph_store.get_neighbors() failed for seed %s; skipping walk",
                seed.node.node_id,
                exc_info=True,
            )
            continue

        # Filter neighbors by spaces
        if spaces:
            neighbors = [
                (n, d)
                for n, d in neighbors
                if _node_passes_space_filter(n, spaces)
            ]

        # Cache relations for depth-1 lookups (called ONCE per seed)
        relations_cache: Optional[list] = None

        for neighbor, depth in neighbors:
            depth_factor = 1.0 / (depth + 1)
            combined_score = seed.score * depth_factor

            # For depth-1 neighbors, look up ALL edges and emit one entry per edge
            if depth == 1:
                # Lazy-load relations cache once per seed
                if relations_cache is None:
                    try:
                        relations_cache = graph_store.get_relations(
                            seed.node.node_id, direction="outgoing"
                        )
                    except Exception:
                        logger.warning(
                            "graph_store.get_relations() failed for seed %s; "
                            "using RELATED and no piece",
                            seed.node.node_id,
                            exc_info=True,
                        )
                        relations_cache = []

                # Collect ALL matching edges for this neighbor
                matching_rels = [
                    rel for rel in relations_cache
                    if rel.target_id == neighbor.node_id
                ]

                if matching_rels:
                    for rel in matching_rels:
                        rel_entry: Dict[str, Any] = {
                            "relation_type": rel.edge_type,
                            "target_node_id": neighbor.node_id,
                            "target_label": neighbor.label,
                            "piece": None,
                            "depth": depth,
                            "score": combined_score,
                        }
                        piece_id = rel.properties.get("piece_id")
                        if piece_id and not _should_skip_piece(
                            piece_id,
                            already_retrieved_piece_ids,
                            ignore_already_retrieved,
                        ):
                            try:
                                piece = piece_store.get_by_id(piece_id)
                            except Exception:
                                logger.warning(
                                    "piece_store.get_by_id(%s) failed; setting piece=None",
                                    piece_id,
                                    exc_info=True,
                                )
                                piece = None
                            if piece:
                                rel_entry["piece"] = piece
                        graph_context.append(rel_entry)
                    continue  # skip the default RELATED append below

            # Default entry: non-depth-1, or depth-1 with no matching relations
            neighbor_entry: Dict[str, Any] = {
                "relation_type": "RELATED",
                "target_node_id": neighbor.node_id,
                "target_label": neighbor.label,
                "piece": None,
                "depth": depth,
                "score": combined_score,
            }
            graph_context.append(neighbor_entry)

    return graph_context


def merge_graph_contexts(
    search_context: List[Dict[str, Any]],
    identity_context: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge graph context entries from search and identity paths.

    Deduplicates by ``(target_node_id, relation_type)``. When duplicates exist:

    - Keep the entry with the higher score.
    - If scores are equal, keep the entry with shorter depth.

    The same node CAN appear with different ``relation_type`` values.

    This is extracted from ``KnowledgeBase._merge_graph_contexts`` with
    identical behavior.

    Args:
        search_context: Graph context from search-based seeds.
        identity_context: Graph context from identity-based seeds.

    Returns:
        Merged, deduplicated list of graph context entries.
    """
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for entry in search_context + identity_context:
        key = (entry["target_node_id"], entry.get("relation_type", "RELATED"))
        if key not in merged:
            merged[key] = entry
        else:
            existing = merged[key]
            new_score = entry.get("score", 0)
            existing_score = existing.get("score", 0)
            if new_score > existing_score or (
                new_score == existing_score and entry["depth"] < existing["depth"]
            ):
                merged[key] = entry
    return list(merged.values())
