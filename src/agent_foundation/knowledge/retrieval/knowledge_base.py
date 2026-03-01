"""
KnowledgeBase — main orchestrator for the Agent Knowledge Base.

The KnowledgeBase coordinates retrieval across three layers:
1. Metadata Layer — structured key-value pairs for an entity
2. Knowledge Pieces Layer — unstructured text chunks retrieved via search
3. Entity Graph Layer — typed relationships and linked knowledge

It implements ``__call__`` so it can be directly assigned to
``Agent.user_profile`` or ``Agent.context``. The existing agent code calls
these attributes if they are callable, passing ``user_input`` as the argument.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8,
              5.1, 5.2, 5.3, 5.4, 5.6, 6.1, 6.2, 6.3, 6.4, 6.5,
              8.1, 8.2, 8.3, 8.4, 8.5, 9.1, 9.2, 9.4
"""
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from attr import attrs, attrib

from agent_foundation.knowledge.retrieval.formatter import (
    KnowledgeFormatter,
    RetrievalResult,
)
from agent_foundation.knowledge.retrieval.hybrid_search import (
    HybridRetriever,
    HybridSearchConfig,
)
from agent_foundation.knowledge.retrieval.mmr_reranking import (
    MMRConfig,
    apply_mmr_reranking,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.models.results import ScoredPiece
from agent_foundation.knowledge.retrieval.stores.graph.base import EntityGraphStore
from agent_foundation.knowledge.retrieval.stores.metadata.base import MetadataStore
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore
from agent_foundation.knowledge.retrieval.temporal_decay import (
    TemporalDecayConfig,
    apply_temporal_decay,
)

logger = logging.getLogger(__name__)


@attrs
class KnowledgeBase:
    """Main orchestrator for the Agent Knowledge Base.

    Coordinates retrieval across metadata, knowledge pieces, and entity graph
    layers. Implements ``__call__`` for seamless Agent integration.

    Attributes:
        metadata_store: Backend for structured entity metadata.
        piece_store: Backend for unstructured knowledge pieces.
        graph_store: Backend for entity graph traversal.
        active_entity_id: Default entity for retrieval (e.g., "user:xinli").
        default_top_k: Default number of results to return.
        include_metadata: Whether to include metadata in retrieval.
        include_pieces: Whether to include knowledge pieces in retrieval.
        include_graph: Whether to include graph context in retrieval.
        graph_traversal_depth: How many hops to traverse in the graph.
        sensitive_patterns: Regex patterns for detecting sensitive content.
        formatter: Formatter for converting RetrievalResult to string.
    """

    metadata_store: MetadataStore = attrib()
    piece_store: KnowledgePieceStore = attrib()
    graph_store: EntityGraphStore = attrib()

    # Active entity context
    active_entity_id: Optional[str] = attrib(default=None)

    # Retrieval config
    default_top_k: int = attrib(default=5)
    include_metadata: bool = attrib(default=True)
    include_pieces: bool = attrib(default=True)
    include_graph: bool = attrib(default=True)
    graph_traversal_depth: int = attrib(default=1)
    graph_retrieval_ignore_pieces_already_retrieved: Union[bool, Tuple[str, ...], List[str]] = attrib(default=False)

    # Security
    sensitive_patterns: List[str] = attrib(
        factory=lambda: [
            r"(?i)(api[_-]?key|secret|password|token|credential)\s*[:=]",
            r"(?i)bearer\s+[a-zA-Z0-9\-._~+/]+=*",
        ]
    )

    # Formatting
    formatter: KnowledgeFormatter = attrib(default=None)

    def __attrs_post_init__(self):
        if self.formatter is None:
            self.formatter = KnowledgeFormatter()
        # Optional enhanced retrieval components (set via setters)
        self._hybrid_retriever: Optional[HybridRetriever] = None
        self._temporal_decay_config: Optional[TemporalDecayConfig] = None
        self._mmr_config: Optional[MMRConfig] = None

    # ── Setter methods for optional retrieval enhancements ───────────────

    def set_hybrid_retriever(self, retriever: HybridRetriever) -> None:
        """Configure a HybridRetriever for enhanced retrieval.

        When set, the retrieve() method uses the enhanced path
        (hybrid search → temporal decay → MMR) instead of the
        standard fallback logic.

        Args:
            retriever: A configured HybridRetriever instance.
        """
        self._hybrid_retriever = retriever

    def set_temporal_decay(self, config: TemporalDecayConfig) -> None:
        """Configure temporal decay scoring.

        Args:
            config: A TemporalDecayConfig instance.
        """
        self._temporal_decay_config = config

    def set_mmr_config(self, config: MMRConfig) -> None:
        """Configure MMR diversity re-ranking.

        Args:
            config: An MMRConfig instance.
        """
        self._mmr_config = config

    # ── Callable interface ───────────────────────────────────────────────

    def __call__(self, query: str, **kwargs) -> str:
        """Callable interface for Agent integration.

        Retrieves from all enabled layers and returns a formatted string
        suitable for prompt injection.

        Args:
            query: The user query string.
            **kwargs: Passed through to ``retrieve()``. Supports ``spaces``
                      keyword argument for space-filtered retrieval.

        Returns:
            A formatted string of retrieved knowledge, or empty string
            if nothing is found.
        """
        spaces = kwargs.pop("spaces", None)
        results = self.retrieve(query, spaces=spaces, **kwargs)
        return self.formatter.format(results)

    # ── Retrieval ────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        entity_id: str = None,
        top_k: int = None,
        include_global: bool = True,
        domain: Optional[str] = None,
        secondary_domains: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        min_results: int = 1,
        spaces: Optional[List[str]] = None,
        **kwargs,
    ) -> RetrievalResult:
        """Orchestrate retrieval across all layers.

        1. Get metadata for the entity (always, regardless of query)
        2. If query is non-empty: search knowledge pieces (entity-scoped + global)
           - When a HybridRetriever is configured, uses the enhanced path:
             hybrid search → space filter → domain filter → temporal decay → MMR → truncate
           - Otherwise, uses the standard path with optional domain-aware
             fallback via ``_retrieve_pieces_with_fallback()``
        3. Traverse entity graph for related knowledge (always, entity-based)
        4. Merge and return

        Empty/whitespace queries skip piece search but still return metadata
        and graph context.

        Args:
            query: The search query string.
            entity_id: Override for active_entity_id.
            top_k: Override for default_top_k.
            include_global: Whether to merge global results.
            domain: Optional primary domain filter for retrieval.
            secondary_domains: Optional list of secondary domains to expand to.
            tags: Optional list of tags to filter by.
            min_results: Minimum number of results before fallback (default 1).
            spaces: Optional list of space strings to filter by (OR semantics).
                    When None, no space filtering is applied (all spaces returned).
            **kwargs: Reserved for future use.

        Returns:
            A RetrievalResult containing metadata, pieces, and graph context.
        """
        entity_id = entity_id or self.active_entity_id
        top_k = top_k if top_k is not None else self.default_top_k

        result = RetrievalResult()

        # Layer 1: Metadata
        if self.include_metadata and entity_id:
            result.metadata = self.metadata_store.get_metadata(entity_id)
            if include_global:
                result.global_metadata = self.metadata_store.get_metadata("global")
            # Filter metadata by spaces intersection (OR semantics)
            if spaces:
                if result.metadata:
                    meta_spaces = set(getattr(result.metadata, "spaces", ["main"]))
                    if not meta_spaces & set(spaces):
                        result.metadata = None
                if result.global_metadata:
                    global_meta_spaces = set(getattr(result.global_metadata, "spaces", ["main"]))
                    if not global_meta_spaces & set(spaces):
                        result.global_metadata = None

        # Layer 2: Knowledge pieces (skip for empty/whitespace queries)
        if query and query.strip():
            if self.include_pieces:
                if self._hybrid_retriever is not None:
                    # Enhanced retrieval path: hybrid → space filter → domain filter → temporal decay → MMR
                    scored = self._hybrid_retriever.search(
                        query=query,
                        top_k=top_k * 3,  # fetch more for post-processing
                        entity_id=entity_id,
                        tags=tags,
                    )

                    # Space post-filter on hybrid results (before domain/temporal/MMR)
                    if spaces:
                        scored = [sp for sp in scored if set(sp.piece.spaces) & set(spaces)]

                    # Domain filter: keep pieces matching domain or secondary_domains
                    if domain:
                        all_domains = {domain}
                        if secondary_domains:
                            all_domains.update(secondary_domains)
                        filtered = [
                            sp for sp in scored
                            if getattr(sp.piece, "domain", "general") in all_domains
                        ]
                        # Fall back to unfiltered if domain filter yields too few
                        if len(filtered) >= min_results:
                            scored = filtered

                    # Temporal decay
                    if self._temporal_decay_config is not None:
                        scored = apply_temporal_decay(scored, self._temporal_decay_config)

                    # MMR diversity re-ranking
                    if self._mmr_config is not None:
                        scored = apply_mmr_reranking(scored, self._mmr_config, top_k=top_k)
                    else:
                        scored = scored[:top_k]

                    # Convert ScoredPiece list to (KnowledgePiece, float) tuples
                    pieces = [(sp.piece, sp.score) for sp in scored]

                    if include_global:
                        global_scored = self._hybrid_retriever.search(
                            query=query,
                            top_k=top_k,
                            entity_id=None,
                            tags=tags,
                        )
                        # Apply space post-filter to global pieces before merging
                        if spaces:
                            global_scored = [sp for sp in global_scored if set(sp.piece.spaces) & set(spaces)]
                        global_pieces = [(sp.piece, sp.score) for sp in global_scored]
                        pieces = self._merge_scored_pieces(pieces, global_pieces, top_k)

                    result.pieces = pieces
                elif domain or tags:
                    # Domain-aware fallback path
                    pieces = self._retrieve_pieces_with_fallback(
                        query=query,
                        entity_id=entity_id,
                        top_k=top_k,
                        domain=domain,
                        secondary_domains=secondary_domains,
                        tags=tags,
                        min_results=min_results,
                        spaces=spaces,
                    )
                    if include_global:
                        global_pieces = self._search_with_space_strategy(
                            query, entity_id=None, top_k=top_k, spaces=spaces
                        )
                        pieces = self._merge_scored_pieces(pieces, global_pieces, top_k)
                    result.pieces = pieces
                else:
                    # Standard retrieval path (no domain/tag filters)
                    pieces = self._search_with_space_strategy(
                        query, entity_id=entity_id, top_k=top_k, spaces=spaces
                    )
                    if include_global:
                        global_pieces = self._search_with_space_strategy(
                            query, entity_id=None, top_k=top_k, spaces=spaces
                        )
                        pieces = self._merge_scored_pieces(pieces, global_pieces, top_k)
                    result.pieces = pieces

        # Build set of already-retrieved piece IDs for graph dedup
        already_retrieved = None
        if self.graph_retrieval_ignore_pieces_already_retrieved and result.pieces:
            already_retrieved = {
                piece.piece_id: piece.info_type
                for piece, score in result.pieces
            }

        # Layer 3: Entity graph
        if self.include_graph and entity_id:
            neighbors = self.graph_store.get_neighbors(
                entity_id, depth=self.graph_traversal_depth
            )
            # Filter graph neighbors by spaces intersection (OR semantics)
            if spaces:
                neighbors = [
                    (n, d) for n, d in neighbors
                    if set(n.properties.get("spaces", ["main"])) & set(spaces)
                ]
            result.graph_context = self._extract_graph_knowledge(
                entity_id, neighbors,
                already_retrieved_piece_ids=already_retrieved,
            )

        return result

    # ── Domain-aware fallback retrieval ─────────────────────────────────

    def _retrieve_pieces_with_fallback(
        self,
        query: str,
        entity_id: Optional[str],
        top_k: int,
        domain: Optional[str] = None,
        secondary_domains: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        min_results: int = 1,
        spaces: Optional[List[str]] = None,
    ) -> List[Tuple[KnowledgePiece, float]]:
        """Retrieve pieces with a 4-tier fallback strategy.

        Tier 1: Domain + tags filter
        Tier 2: Expand to secondary_domains
        Tier 3: Tags only (no domain filter)
        Tier 4: Pure semantic search (no filters)

        Each tier is tried only if the previous tier returned fewer
        results than ``min_results``.

        Args:
            query: The search query string.
            entity_id: Entity scope for the search.
            top_k: Maximum number of results.
            domain: Primary domain filter.
            secondary_domains: Secondary domains to expand to in Tier 2.
            tags: Tag filters.
            min_results: Minimum acceptable result count before fallback.
            spaces: Optional list of space strings to filter by (OR semantics).

        Returns:
            A list of (KnowledgePiece, score) tuples.
        """
        # Tier 1: Domain + tags
        pieces = self._search_with_space_strategy(
            query, entity_id=entity_id, tags=tags, top_k=top_k, spaces=spaces
        )
        if domain:
            pieces = [
                (p, s) for p, s in pieces
                if getattr(p, "domain", "general") == domain
            ]
        if len(pieces) >= min_results:
            return pieces[:top_k]

        # Tier 2: Expand to secondary_domains
        if secondary_domains:
            all_domains = {domain} if domain else set()
            all_domains.update(secondary_domains)
            pieces = self._search_with_space_strategy(
                query, entity_id=entity_id, tags=tags, top_k=top_k, spaces=spaces
            )
            pieces = [
                (p, s) for p, s in pieces
                if getattr(p, "domain", "general") in all_domains
            ]
            if len(pieces) >= min_results:
                return pieces[:top_k]

        # Tier 3: Tags only (no domain filter)
        if tags:
            pieces = self._search_with_space_strategy(
                query, entity_id=entity_id, tags=tags, top_k=top_k, spaces=spaces
            )
            if len(pieces) >= min_results:
                return pieces[:top_k]

        # Tier 4: Pure semantic search (no filters)
        pieces = self._search_with_space_strategy(
            query, entity_id=entity_id, top_k=top_k, spaces=spaces
        )
        return pieces[:top_k]

    # ── Adaptive space-aware search ──────────────────────────────────────

    def _search_with_space_strategy(
        self,
        query: str,
        entity_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        top_k: int = 5,
        spaces: Optional[List[str]] = None,
    ) -> List[Tuple[KnowledgePiece, float]]:
        """Search pieces using the adaptive space filtering strategy.

        If the piece store natively supports space filtering
        (``supports_space_filter is True``), passes ``spaces`` directly
        and trusts the pre-filtered results.

        If the store does not support native filtering, applies progressive
        over-fetch (5× initial, 20× retry if insufficient) and post-filters
        results by spaces intersection.

        When ``spaces`` is None, delegates directly to the store with no
        space filtering.

        Args:
            query: The search query string.
            entity_id: Entity scope for the search.
            tags: Optional tag filters.
            top_k: Maximum number of results.
            spaces: Optional list of space strings to filter by.

        Returns:
            A list of (KnowledgePiece, score) tuples.
        """
        if not spaces:
            # No space filtering — direct pass-through
            return self.piece_store.search(
                query, entity_id=entity_id, tags=tags, top_k=top_k
            )

        if self.piece_store.supports_space_filter:
            # Store handles filtering natively (e.g., LanceDB)
            return self.piece_store.search(
                query, entity_id=entity_id, tags=tags, top_k=top_k, spaces=spaces
            )

        # Non-native store: over-fetch and post-filter
        initial_top_k = top_k * 5
        pieces = self.piece_store.search(
            query, entity_id=entity_id, tags=tags, top_k=initial_top_k
        )
        pieces = [(p, s) for p, s in pieces if set(p.spaces) & set(spaces)]

        if len(pieces) < top_k:
            # Retry with larger fetch
            pieces = self.piece_store.search(
                query, entity_id=entity_id, tags=tags, top_k=top_k * 20
            )
            pieces = [(p, s) for p, s in pieces if set(p.spaces) & set(spaces)]

        return pieces[:top_k]

    # ── Merge logic ──────────────────────────────────────────────────────

    def _merge_scored_pieces(
        self,
        a: List[Tuple[KnowledgePiece, float]],
        b: List[Tuple[KnowledgePiece, float]],
        top_k: int,
    ) -> List[Tuple[KnowledgePiece, float]]:
        """Merge two scored lists, deduplicate by piece_id, take top_k.

        Deduplication: If the same piece_id appears in both lists, keep the
        entry with the higher score.

        Sorting: By score descending, then by piece_id ascending for stability.

        Args:
            a: First scored list (e.g., entity-scoped results).
            b: Second scored list (e.g., global results).
            top_k: Maximum number of results to return.

        Returns:
            Merged, deduplicated, sorted, and truncated list.
        """
        merged: Dict[str, Tuple[KnowledgePiece, float]] = {}
        for piece, score in a + b:
            if piece.piece_id not in merged or score > merged[piece.piece_id][1]:
                merged[piece.piece_id] = (piece, score)
        sorted_pieces = sorted(
            merged.values(), key=lambda x: (-x[1], x[0].piece_id)
        )
        return sorted_pieces[:top_k]

    # ── Graph knowledge extraction ───────────────────────────────────────

    def _extract_graph_knowledge(
        self,
        entity_id: str,
        neighbors: List[Tuple[Any, int]],
        already_retrieved_piece_ids: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract knowledge from graph neighbors — relations and linked pieces.

        For each (neighbor, depth):
        1. Get the relation connecting to this neighbor (depth-1 only)
        2. If relation has 'piece_id' in properties, look up the piece
           (unless already retrieved by Layer 2, controlled by
           graph_retrieval_ignore_pieces_already_retrieved)
        3. Score graph-derived pieces by traversal depth: depth-N → 1.0/N
        4. Return list of dicts with relation_type, target info, piece, depth

        Args:
            entity_id: The source entity ID.
            neighbors: List of (GraphNode, depth) tuples from get_neighbors.
            already_retrieved_piece_ids: Dict mapping piece_id to info_type
                for pieces already found by Layer 2 search. Used for dedup
                when graph_retrieval_ignore_pieces_already_retrieved is set.

        Returns:
            List of graph context dictionaries.
        """
        graph_context = []
        for neighbor, depth in neighbors:
            entry: Dict[str, Any] = {
                "relation_type": "RELATED",
                "target_node_id": neighbor.node_id,
                "target_label": neighbor.label,
                "piece": None,
                "depth": depth,
            }

            # For depth-1 neighbors, get the actual relation
            if depth == 1:
                relations = self.graph_store.get_relations(
                    entity_id, direction="outgoing"
                )
                for rel in relations:
                    if rel.target_id == neighbor.node_id:
                        entry["relation_type"] = rel.edge_type
                        # Check for linked piece
                        piece_id = rel.properties.get("piece_id")
                        if piece_id and not self._should_skip_graph_piece(
                            piece_id, already_retrieved_piece_ids
                        ):
                            piece = self.piece_store.get_by_id(piece_id)
                            if piece:
                                entry["piece"] = piece
                        break

            graph_context.append(entry)

        return graph_context

    def _should_skip_graph_piece(
        self,
        piece_id: str,
        already_retrieved_piece_ids: Optional[Dict[str, str]],
    ) -> bool:
        """Check if a graph-linked piece should be skipped (already retrieved).

        Args:
            piece_id: The piece_id from the graph edge properties.
            already_retrieved_piece_ids: Dict mapping piece_id to info_type
                for pieces already found by Layer 2 search, or None.

        Returns:
            True if the piece should be skipped (not attached to graph entry).
        """
        if not already_retrieved_piece_ids or piece_id not in already_retrieved_piece_ids:
            return False
        if self.graph_retrieval_ignore_pieces_already_retrieved is True:
            return True
        if isinstance(self.graph_retrieval_ignore_pieces_already_retrieved, (list, tuple)):
            piece_info_type = already_retrieved_piece_ids[piece_id]
            return piece_info_type in self.graph_retrieval_ignore_pieces_already_retrieved
        return False

    # ── CRUD operations ──────────────────────────────────────────────────

    def add_piece(self, piece: KnowledgePiece) -> str:
        """Add a knowledge piece after validation.

        Validates that content is non-empty and does not contain sensitive
        patterns, then delegates to the piece store.

        Args:
            piece: The KnowledgePiece to add.

        Returns:
            The piece_id of the added piece.

        Raises:
            ValueError: If content is empty or contains sensitive patterns.
        """
        self._validate_content(piece.content)
        return self.piece_store.add(piece)

    def update_piece(self, piece: KnowledgePiece) -> bool:
        """Update a knowledge piece after validation.

        Validates content, updates the ``updated_at`` timestamp, then
        delegates to the piece store.

        Args:
            piece: The KnowledgePiece with updated fields.

        Returns:
            True if the piece was found and updated, False if not found.

        Raises:
            ValueError: If content is empty or contains sensitive patterns.
        """
        self._validate_content(piece.content)
        piece.updated_at = datetime.now(timezone.utc).isoformat()
        return self.piece_store.update(piece)

    def remove_piece(self, piece_id: str) -> bool:
        """Remove a knowledge piece by ID.

        Args:
            piece_id: The unique identifier of the piece to remove.

        Returns:
            True if the piece existed and was removed, False if not found.
        """
        return self.piece_store.remove(piece_id)

    # ── Bulk loading ─────────────────────────────────────────────────────

    def bulk_load(self, file_path: str) -> int:
        """Load KnowledgePiece items from a JSON file.

        Reads a JSON file containing a list of KnowledgePiece dictionaries.
        Each item is validated and added to the store. Items that fail
        validation are logged and skipped.

        Args:
            file_path: Path to a JSON file containing a list of piece dicts.

        Returns:
            Count of successfully loaded items.

        Raises:
            FileNotFoundError: If file_path does not exist.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        count = 0
        for i, item_dict in enumerate(items):
            try:
                piece = KnowledgePiece.from_dict(item_dict)
                self.add_piece(piece)
                count += 1
            except Exception as e:
                logger.warning("Skipping item %d: %s", i, e)

        return count

    # ── Validation ───────────────────────────────────────────────────────

    def _validate_content(self, content: str):
        """Validate content is non-empty and contains no secrets.

        Args:
            content: The content string to validate.

        Raises:
            ValueError: If content is empty/whitespace or contains
                        sensitive patterns.
        """
        if not content or not content.strip():
            raise ValueError("Content must be a non-empty string")
        if self._contains_sensitive_content(content):
            raise ValueError(
                "Content contains potentially sensitive information "
                "(API keys, passwords, tokens)"
            )

    def _contains_sensitive_content(self, content: str) -> bool:
        """Check content against sensitive_patterns using regex.

        This is heuristic/best-effort. Users can configure
        ``sensitive_patterns`` to reduce false positives.

        Args:
            content: The content string to check.

        Returns:
            True if any sensitive pattern matches, False otherwise.
        """
        for pattern in self.sensitive_patterns:
            if re.search(pattern, content):
                return True
        return False

    # ── Lifecycle ────────────────────────────────────────────────────────

    def close(self):
        """Close all underlying store connections.

        Delegates to metadata_store.close(), piece_store.close(), and
        graph_store.close(). Safe to call multiple times.
        """
        self.metadata_store.close()
        self.piece_store.close()
        self.graph_store.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit — closes all stores."""
        self.close()
        return False
