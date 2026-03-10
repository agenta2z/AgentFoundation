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
              5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3, 6.4, 6.5,
              8.1, 8.2, 8.3, 9.1, 9.2
"""
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from attr import attrs, attrib

from rich_python_utils.service_utils.data_operation_record import (
    DataOperationRecord,
    generate_operation_id,
)
from agent_foundation.knowledge.retrieval.models.kb_metadata import (
    KnowledgeBaseMetadata,
)
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
from agent_foundation.knowledge.retrieval.graph_walk import (
    find_search_seeds,
    find_identity_seeds,
    graph_walk,
    merge_graph_contexts,
)
from agent_foundation.knowledge.retrieval.utils import parse_entity_type

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

    # Store path for KB-level metadata persistence
    store_path: Optional[str] = attrib(default=None)



    def __attrs_post_init__(self):
        if self.formatter is None:
            self.formatter = KnowledgeFormatter()
        # Optional enhanced retrieval components (set via setters)
        self._hybrid_retriever: Optional[HybridRetriever] = None
        self._temporal_decay_config: Optional[TemporalDecayConfig] = None
        self._mmr_config: Optional[MMRConfig] = None
        # KB-level metadata
        self._kb_metadata: Optional[KnowledgeBaseMetadata] = None
        if self.store_path:
            self._load_kb_metadata()

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

        Delegates to ``RetrievalPipeline`` with ``FlatStringPostProcessor``
        for flat string output.  Preserves the existing signature.

        Imports are deferred (inside method body) to avoid circular import:
        ``knowledge_base.py`` → ``retrieval_pipeline.py`` → ``knowledge_base.py``.

        Args:
            query: The user query string.
            **kwargs: Passed through to ``retrieve()``. Supports ``spaces``
                      keyword argument for space-filtered retrieval.

        Returns:
            A formatted string of retrieved knowledge, or empty string
            if nothing is found.

        Requirements: 8.1, 8.2, 8.3
        """
        # Deferred imports to avoid circular dependency
        from agent_foundation.knowledge.retrieval.retrieval_pipeline import RetrievalPipeline
        from agent_foundation.knowledge.retrieval.post_processors import FlatStringPostProcessor

        pipeline = RetrievalPipeline(
            kb=self,
            post_processor=FlatStringPostProcessor(formatter=self.formatter),
        )
        spaces = kwargs.get("spaces", None)
        # Remove spaces from kwargs to avoid passing it twice to pipeline.execute
        kwargs.pop("spaces", None)
        return pipeline.execute(query, spaces=spaces, **kwargs)


    # ── Layer Methods ────────────────────────────────────────────────────

    def retrieve_metadata(
        self,
        entity_id: Optional[str] = None,
        include_global: bool = True,
        spaces: Optional[List[str]] = None,
    ) -> Tuple[Optional["EntityMetadata"], Optional["EntityMetadata"]]:
        """Layer 1: Retrieve entity and global metadata.

        Returns ``(None, None)`` if ``self.include_metadata`` is False or
        entity_id is not provided.

        Args:
            entity_id: Override for active_entity_id. Resolved internally
                       as ``entity_id or self.active_entity_id``.
            include_global: Whether to include global metadata.
            spaces: Optional space filter (OR semantics).

        Returns:
            Tuple of (entity_metadata, global_metadata). Either may be None.

        Requirements: 13.1, 13.6
        """
        entity_id = entity_id or self.active_entity_id
        metadata = None
        global_metadata = None

        if not self.include_metadata or not entity_id:
            return (None, None)

        metadata = self.metadata_store.get_metadata(entity_id)
        if include_global:
            global_metadata = self.metadata_store.get_metadata("global")

        # Filter metadata by spaces intersection (OR semantics)
        if spaces:
            if metadata:
                meta_spaces = set(getattr(metadata, "spaces", ["main"]))
                if not meta_spaces & set(spaces):
                    metadata = None
            if global_metadata:
                global_meta_spaces = set(getattr(global_metadata, "spaces", ["main"]))
                if not global_meta_spaces & set(spaces):
                    global_metadata = None

        return (metadata, global_metadata)

    def retrieve_pieces(
        self,
        query: str,
        entity_id: Optional[str] = None,
        top_k: Optional[int] = None,
        include_global: bool = True,
        domain: Optional[str] = None,
        secondary_domains: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        min_results: int = 1,
        spaces: Optional[List[str]] = None,
    ) -> List[Tuple[KnowledgePiece, float]]:
        """Layer 2: Search knowledge pieces (entity-scoped + optional global).

        Handles all three sub-paths: hybrid retriever, domain-aware fallback,
        and standard. Returns empty list if ``self.include_pieces`` is False
        or query is empty/whitespace.

        Args:
            query: The search query string.
            entity_id: Override for active_entity_id. Resolved internally
                       as ``entity_id or self.active_entity_id``.
            top_k: Override for default_top_k. Resolved internally
                   as ``top_k if top_k is not None else self.default_top_k``.
            include_global: Whether to merge global results.
            domain: Optional primary domain filter.
            secondary_domains: Optional secondary domain list.
            tags: Optional tag filter.
            min_results: Minimum results before fallback.
            spaces: Optional space filter (OR semantics).

        Returns:
            List of (KnowledgePiece, score) tuples.

        Requirements: 13.2, 13.6
        """
        entity_id = entity_id or self.active_entity_id
        top_k = top_k if top_k is not None else self.default_top_k

        if not query or not query.strip() or not self.include_pieces:
            return []

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

            return pieces

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
            return pieces

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
            return pieces

    def retrieve_search_graph(
        self,
        query: str,
        top_k: Optional[int] = None,
        spaces: Optional[List[str]] = None,
        already_retrieved_piece_ids: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Layer 3a: Query-driven graph search via unified graph walk.

        Calls ``find_search_seeds()`` + ``graph_walk()`` internally.
        Returns empty list if ``self.include_graph`` is False.

        Args:
            query: The search query string.
            top_k: Maximum seed nodes for graph search. Resolved internally
                   as ``top_k if top_k is not None else self.default_top_k``.
            spaces: Optional space filter (OR semantics).
            already_retrieved_piece_ids: Piece IDs from L2 for dedup.

        Returns:
            List of graph context entry dicts from search-based seeds.

        Requirements: 13.3, 13.6
        """
        top_k = top_k if top_k is not None else self.default_top_k

        if not self.include_graph:
            return []

        search_seeds = find_search_seeds(
            self.graph_store, query, top_k, spaces
        )
        if not search_seeds:
            return []

        return graph_walk(
            self.graph_store,
            self.piece_store,
            search_seeds,
            self.graph_traversal_depth,
            already_retrieved_piece_ids,
            spaces,
            self.graph_retrieval_ignore_pieces_already_retrieved,
        )

    def retrieve_identity_graph(
        self,
        entity_id: Optional[str] = None,
        spaces: Optional[List[str]] = None,
        already_retrieved_piece_ids: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Layer 3b: Identity-based graph traversal via unified graph walk.

        Calls ``find_identity_seeds()`` + ``graph_walk()`` internally.
        Returns empty list if ``self.include_graph`` is False.

        Args:
            entity_id: Override for active_entity_id. Resolved internally
                       as ``entity_id or self.active_entity_id``.
            spaces: Optional space filter (OR semantics).
            already_retrieved_piece_ids: Piece IDs from L2 for dedup.

        Returns:
            List of graph context entry dicts from identity-based seeds.

        Requirements: 13.4, 13.6
        """
        entity_id = entity_id or self.active_entity_id

        if not self.include_graph:
            return []

        identity_seeds = find_identity_seeds(
            self.graph_store, entity_id, spaces
        )
        if not identity_seeds:
            return []

        return graph_walk(
            self.graph_store,
            self.piece_store,
            identity_seeds,
            self.graph_traversal_depth,
            already_retrieved_piece_ids,
            spaces,
            self.graph_retrieval_ignore_pieces_already_retrieved,
        )


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
        """Thin orchestrator calling the four layer methods.

        Signature and return type unchanged — full backward compatibility.

        1. L1: retrieve_metadata() — entity and global metadata
        2. L2: retrieve_pieces() — knowledge piece search (if query non-empty)
        3. L3a: retrieve_search_graph() — query-driven graph search
        4. L3b: retrieve_identity_graph() — identity-based graph traversal
        5. Merge L3a + L3b graph contexts

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

        Requirements: 13.5
        """
        entity_id = entity_id or self.active_entity_id
        top_k = top_k if top_k is not None else self.default_top_k

        result = RetrievalResult()

        # L1: Metadata
        result.metadata, result.global_metadata = self.retrieve_metadata(
            entity_id, include_global, spaces
        )

        # L2: Knowledge pieces (skip for empty/whitespace queries)
        if query and query.strip() and self.include_pieces:
            result.pieces = self.retrieve_pieces(
                query, entity_id, top_k, include_global,
                domain, secondary_domains, tags, min_results, spaces,
            )

        # Build dedup set from L2 pieces for graph walk
        already_retrieved_piece_ids = None
        if self.graph_retrieval_ignore_pieces_already_retrieved and result.pieces:
            already_retrieved_piece_ids = {
                p.piece_id: p.info_type for p, _ in result.pieces
            }

        # L3a: Query-driven graph search
        search_ctx = self.retrieve_search_graph(
            query, top_k, spaces, already_retrieved_piece_ids
        )

        # L3b: Identity-based graph traversal
        identity_ctx = self.retrieve_identity_graph(
            entity_id, spaces, already_retrieved_piece_ids
        )

        # Merge graph contexts from both paths
        if search_ctx or identity_ctx:
            result.graph_context = merge_graph_contexts(search_ctx, identity_ctx)

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







    # ── CRUD operations ──────────────────────────────────────────────────

    def add_piece(
        self,
        piece: KnowledgePiece,
        operation_id: Optional[str] = None,
    ) -> str:
        """Add a knowledge piece after validation.

        Validates that content is non-empty and does not contain sensitive
        patterns, appends an ADD history record, then delegates to the
        piece store.

        Args:
            piece: The KnowledgePiece to add.
            operation_id: Optional shared operation ID for batch grouping.

        Returns:
            The piece_id of the added piece.

        Raises:
            ValueError: If content is empty or contains sensitive patterns.
        """
        self._validate_content(piece.content)
        now = datetime.now(timezone.utc).isoformat()
        op_id = operation_id or generate_operation_id("KnowledgeBase", "add_piece")
        piece.history.append(DataOperationRecord(
            operation="add",
            timestamp=now,
            operation_id=op_id,
            source="KnowledgeBase.add_piece",
        ))
        result = self.piece_store.add(piece)
        self._log_kb_operation(op_id, f"Added piece {piece.piece_id}", "KnowledgeBase.add_piece", 1)
        return result

    def update_piece(
        self,
        piece: KnowledgePiece,
        operation_id: Optional[str] = None,
    ) -> bool:
        """Update a knowledge piece after validation.

        Fetches the existing piece first to capture content_before for
        history tracking. Validates content, appends an UPDATE record,
        updates the ``updated_at`` timestamp, then delegates to the store.

        Args:
            piece: The KnowledgePiece with updated fields.
            operation_id: Optional shared operation ID for batch grouping.

        Returns:
            True if the piece was found and updated, False if not found.

        Raises:
            ValueError: If content is empty or contains sensitive patterns.
        """
        self._validate_content(piece.content)

        # Fetch existing to capture content_before
        existing = self.piece_store.get_by_id(piece.piece_id)
        if existing is None:
            return False

        now = datetime.now(timezone.utc).isoformat()
        op_id = operation_id or generate_operation_id("KnowledgeBase", "update_piece")

        # Build fields_changed for non-content field diffs
        fields_changed = {}
        for field_name in ("spaces", "tags", "domain", "info_type", "knowledge_type"):
            old_val = getattr(existing, field_name, None)
            new_val = getattr(piece, field_name, None)
            # Normalize KnowledgeType to string for comparison
            if hasattr(old_val, "value"):
                old_val = old_val.value
            if hasattr(new_val, "value"):
                new_val = new_val.value
            if old_val != new_val:
                fields_changed[field_name] = {"before": old_val, "after": new_val}

        piece.history.append(DataOperationRecord(
            operation="update",
            timestamp=now,
            operation_id=op_id,
            source="KnowledgeBase.update_piece",
            content_before=existing.content if existing.content != piece.content else None,
            content_after=piece.content if existing.content != piece.content else None,
            fields_changed=fields_changed or None,
        ))
        piece.updated_at = now
        result = self.piece_store.update(piece)
        if result:
            self._log_kb_operation(op_id, f"Updated piece {piece.piece_id}", "KnowledgeBase.update_piece", 1)
        return result

    def remove_piece(
        self,
        piece_id: str,
        operation_id: Optional[str] = None,
        hard: bool = False,
    ) -> bool:
        """Remove a knowledge piece by ID.

        By default performs a soft delete (sets is_active=False and appends
        a DELETE history record). Pass ``hard=True`` for permanent removal.

        Args:
            piece_id: The unique identifier of the piece to remove.
            operation_id: Optional shared operation ID for batch grouping.
            hard: If True, permanently removes the piece from the store.

        Returns:
            True if the piece existed and was removed/deactivated,
            False if not found.
        """
        if hard:
            result = self.piece_store.remove(piece_id)
            if result:
                op_id = operation_id or generate_operation_id("KnowledgeBase", "hard_delete")
                self._log_kb_operation(op_id, f"Hard-deleted piece {piece_id}", "KnowledgeBase.remove_piece", 1)
            return result

        # Soft delete
        piece = self.piece_store.get_by_id(piece_id)
        if piece is None:
            return False
        if not piece.is_active:
            return False  # Already soft-deleted

        now = datetime.now(timezone.utc).isoformat()
        op_id = operation_id or generate_operation_id("KnowledgeBase", "delete_piece")
        piece.is_active = False
        piece.history.append(DataOperationRecord(
            operation="delete",
            timestamp=now,
            operation_id=op_id,
            source="KnowledgeBase.remove_piece",
            details={"delete_mode": "soft"},
        ))
        piece.updated_at = now
        result = self.piece_store.update(piece)
        if result:
            self._log_kb_operation(op_id, f"Soft-deleted piece {piece_id}", "KnowledgeBase.remove_piece", 1)
        return result

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

    # ── KB-level metadata ──────────────────────────────────────────────

    def _load_kb_metadata(self) -> None:
        """Load or create KB metadata from store_path."""
        meta_path = os.path.join(self.store_path, "_kb_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self._kb_metadata = KnowledgeBaseMetadata.from_dict(json.load(f))
            except Exception as exc:
                logger.warning("Failed to load KB metadata: %s", exc)
                self._kb_metadata = KnowledgeBaseMetadata(kb_id=self.store_path)
        else:
            self._kb_metadata = KnowledgeBaseMetadata(kb_id=self.store_path)

    def _save_kb_metadata(self) -> None:
        """Persist KB metadata to store_path."""
        if self._kb_metadata is None or self.store_path is None:
            return
        meta_path = os.path.join(self.store_path, "_kb_metadata.json")
        try:
            os.makedirs(self.store_path, exist_ok=True)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self._kb_metadata.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Failed to save KB metadata: %s", exc)

    def _log_kb_operation(
        self,
        operation_id: str,
        description: str,
        source: str,
        entity_count: int = 1,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an operation to KB metadata and persist."""
        if self._kb_metadata is None:
            return
        self._kb_metadata.log_operation(
            operation_id=operation_id,
            description=description,
            source=source,
            entity_count=entity_count,
            details=details,
        )
        self._save_kb_metadata()

    def get_kb_metadata(self) -> Optional[KnowledgeBaseMetadata]:
        """Return the KB-level metadata object."""
        return self._kb_metadata

    def get_operations_since(self, timestamp: str) -> list:
        """Return KB operation entries after the given timestamp."""
        if self._kb_metadata is None:
            return []
        return [
            op for op in self._kb_metadata.operations
            if op.timestamp > timestamp
        ]

    def get_operation_by_id(self, operation_id: str):
        """Look up a KB operation entry by its operation_id."""
        if self._kb_metadata is None:
            return None
        for op in self._kb_metadata.operations:
            if op.operation_id == operation_id:
                return op
        return None

    # ── Rollback ─────────────────────────────────────────────────────────

    def rollback_to(self, timestamp: str) -> Dict[str, Any]:
        """Restore all knowledge to its state at the given ISO 8601 timestamp.

        Scans all pieces, metadata, and graph entities across all namespaces,
        finds history records after the target timestamp, and applies undo
        operations in strict reverse chronological order.

        Args:
            timestamp: ISO 8601 UTC timestamp to roll back to.

        Returns:
            Dict with counts of entities rolled back per type and
            operation_ids affected.
        """
        op_id = generate_operation_id("KnowledgeBase", "rollback_to")
        result = {
            "pieces": 0,
            "metadata": 0,
            "graph_nodes": 0,
            "graph_edges": 0,
            "operation_ids": set(),
        }

        # ── Pieces rollback ──────────────────────────────────────────
        namespaces = list(self.piece_store.retrieval_service.namespaces()) if hasattr(self.piece_store, 'retrieval_service') else []
        namespaces.append(None)  # default namespace

        for ns in namespaces:
            pieces = self.piece_store.list_all(entity_id=ns)
            for piece in pieces:
                records_after = [
                    r for r in piece.history
                    if r.timestamp > timestamp
                ]
                if not records_after:
                    continue

                records_after.sort(key=lambda r: r.timestamp, reverse=True)

                changed = False
                for record in records_after:
                    if record.operation_id:
                        result["operation_ids"].add(record.operation_id)

                    if record.operation == "update" and record.content_before is not None:
                        piece.content = record.content_before
                        piece.content_hash = piece._compute_content_hash()
                        changed = True
                    elif record.operation == "delete":
                        piece.is_active = True
                        changed = True
                    elif record.operation == "add":
                        self.piece_store.remove(piece.piece_id)
                        result["pieces"] += 1
                        changed = False
                        break
                    elif record.operation == "restore":
                        piece.is_active = False
                        changed = True

                if changed:
                    piece.history = [r for r in piece.history if r.timestamp <= timestamp]
                    piece.updated_at = datetime.now(timezone.utc).isoformat()
                    self.piece_store.update(piece)
                    result["pieces"] += 1

        # ── Metadata rollback ────────────────────────────────────────
        if hasattr(self.metadata_store, 'kv_service'):
            all_entity_ids = self.metadata_store.list_entities(include_inactive=True)
            for eid in all_entity_ids:
                meta = self.metadata_store.get_metadata(eid, include_inactive=True)
                if meta is None:
                    continue
                records_after = [
                    r for r in meta.history
                    if r.timestamp > timestamp
                ]
                if not records_after:
                    continue

                records_after.sort(key=lambda r: r.timestamp, reverse=True)

                changed = False
                for record in records_after:
                    if record.operation_id:
                        result["operation_ids"].add(record.operation_id)

                    if record.operation == "update" and record.properties_before is not None:
                        meta.properties = dict(record.properties_before)
                        changed = True
                    elif record.operation == "delete":
                        meta.is_active = True
                        changed = True
                    elif record.operation == "add":
                        # Hard-remove metadata added after target
                        entity_type = parse_entity_type(eid)
                        self.metadata_store.kv_service.delete(eid, namespace=entity_type)
                        result["metadata"] += 1
                        changed = False
                        break
                    elif record.operation == "restore":
                        meta.is_active = False
                        changed = True

                if changed:
                    meta.history = [r for r in meta.history if r.timestamp <= timestamp]
                    meta.updated_at = datetime.now(timezone.utc).isoformat()
                    # Write directly to KV service to bypass adapter history tracking
                    entity_type = parse_entity_type(eid)
                    self.metadata_store.kv_service.put(
                        eid, meta.to_dict(), namespace=entity_type,
                    )
                    result["metadata"] += 1

        # ── Graph rollback (nodes then edges) ────────────────────────
        if hasattr(self.graph_store, 'graph_service'):
            gs = self.graph_store.graph_service

            # Phase 1: Rollback graph nodes
            all_nodes = gs.list_nodes()
            for node in all_nodes:
                records_after = [
                    r for r in node.history
                    if r.timestamp > timestamp
                ]
                if not records_after:
                    continue

                records_after.sort(key=lambda r: r.timestamp, reverse=True)

                changed = False
                for record in records_after:
                    if record.operation_id:
                        result["operation_ids"].add(record.operation_id)

                    if record.operation == "update" and record.properties_before is not None:
                        node.properties = dict(record.properties_before)
                        changed = True
                    elif record.operation == "delete":
                        node.is_active = True
                        changed = True
                    elif record.operation == "add":
                        # Hard-remove node added after target (cascades edges)
                        gs.remove_node(node.node_id)
                        result["graph_nodes"] += 1
                        changed = False
                        break
                    elif record.operation == "restore":
                        node.is_active = False
                        changed = True

                if changed:
                    node.history = [r for r in node.history if r.timestamp <= timestamp]
                    gs.add_node(node)
                    result["graph_nodes"] += 1

            # Phase 2: Rollback graph edges
            processed_edges = set()
            remaining_nodes = gs.list_nodes()
            for node in remaining_nodes:
                edges = gs.get_edges(node.node_id, direction="outgoing")
                for edge in edges:
                    edge_key = (edge.source_id, edge.target_id, edge.edge_type)
                    if edge_key in processed_edges:
                        continue
                    processed_edges.add(edge_key)

                    records_after = [
                        r for r in edge.history
                        if r.timestamp > timestamp
                    ]
                    if not records_after:
                        continue

                    records_after.sort(key=lambda r: r.timestamp, reverse=True)

                    changed = False
                    for record in records_after:
                        if record.operation_id:
                            result["operation_ids"].add(record.operation_id)

                        if record.operation == "update" and record.properties_before is not None:
                            edge.properties = dict(record.properties_before)
                            changed = True
                        elif record.operation == "delete":
                            edge.is_active = True
                            changed = True
                        elif record.operation == "add":
                            gs.remove_edge(edge.source_id, edge.target_id, edge.edge_type)
                            result["graph_edges"] += 1
                            changed = False
                            break
                        elif record.operation == "restore":
                            edge.is_active = False
                            changed = True

                    if changed:
                        edge.history = [r for r in edge.history if r.timestamp <= timestamp]
                        gs.remove_edge(edge.source_id, edge.target_id, edge.edge_type)
                        gs.add_edge(edge)
                        result["graph_edges"] += 1

        total = result["pieces"] + result["metadata"] + result["graph_nodes"] + result["graph_edges"]
        result["operation_ids"] = list(result["operation_ids"])
        self._log_kb_operation(
            op_id,
            f"Rolled back to {timestamp}",
            "KnowledgeBase.rollback_to",
            total,
        )
        return result

    def rollback_operation(self, operation_id: str) -> Dict[str, Any]:
        """Undo all changes from a specific batch operation.

        Finds all entities (pieces, metadata, graph nodes/edges) with history
        records matching the operation_id and reverses those records in strict
        reverse chronological order.

        Args:
            operation_id: The operation ID to roll back.

        Returns:
            Dict with counts of entities rolled back per type.
        """
        rb_op_id = generate_operation_id("KnowledgeBase", "rollback_op")
        result = {"pieces": 0, "metadata": 0, "graph_nodes": 0, "graph_edges": 0}

        # ── Pieces ────────────────────────────────────────────────────
        namespaces = list(self.piece_store.retrieval_service.namespaces()) if hasattr(self.piece_store, 'retrieval_service') else []
        namespaces.append(None)

        for ns in namespaces:
            pieces = self.piece_store.list_all(entity_id=ns)
            for piece in pieces:
                matching = [
                    r for r in piece.history
                    if r.operation_id == operation_id
                ]
                if not matching:
                    continue

                matching.sort(key=lambda r: r.timestamp, reverse=True)

                changed = False
                for record in matching:
                    if record.operation == "update" and record.content_before is not None:
                        piece.content = record.content_before
                        piece.content_hash = piece._compute_content_hash()
                        changed = True
                    elif record.operation == "delete":
                        piece.is_active = True
                        changed = True
                    elif record.operation == "add":
                        self.piece_store.remove(piece.piece_id)
                        result["pieces"] += 1
                        changed = False
                        break
                    elif record.operation == "restore":
                        piece.is_active = False
                        changed = True

                if changed:
                    piece.history = [
                        r for r in piece.history
                        if r.operation_id != operation_id
                    ]
                    piece.updated_at = datetime.now(timezone.utc).isoformat()
                    self.piece_store.update(piece)
                    result["pieces"] += 1

        # ── Metadata ──────────────────────────────────────────────────
        if hasattr(self.metadata_store, 'kv_service'):
            all_entity_ids = self.metadata_store.list_entities(include_inactive=True)
            for eid in all_entity_ids:
                meta = self.metadata_store.get_metadata(eid, include_inactive=True)
                if meta is None:
                    continue
                matching = [
                    r for r in meta.history
                    if r.operation_id == operation_id
                ]
                if not matching:
                    continue

                matching.sort(key=lambda r: r.timestamp, reverse=True)

                changed = False
                for record in matching:
                    if record.operation == "update" and record.properties_before is not None:
                        meta.properties = dict(record.properties_before)
                        changed = True
                    elif record.operation == "delete":
                        meta.is_active = True
                        changed = True
                    elif record.operation == "add":
                        entity_type = parse_entity_type(eid)
                        self.metadata_store.kv_service.delete(eid, namespace=entity_type)
                        result["metadata"] += 1
                        changed = False
                        break
                    elif record.operation == "restore":
                        meta.is_active = False
                        changed = True

                if changed:
                    meta.history = [
                        r for r in meta.history
                        if r.operation_id != operation_id
                    ]
                    meta.updated_at = datetime.now(timezone.utc).isoformat()
                    entity_type = parse_entity_type(eid)
                    self.metadata_store.kv_service.put(
                        eid, meta.to_dict(), namespace=entity_type,
                    )
                    result["metadata"] += 1

        # ── Graph (nodes then edges) ─────────────────────────────────
        if hasattr(self.graph_store, 'graph_service'):
            gs = self.graph_store.graph_service

            # Phase 1: nodes
            all_nodes = gs.list_nodes()
            for node in all_nodes:
                matching = [
                    r for r in node.history
                    if r.operation_id == operation_id
                ]
                if not matching:
                    continue

                matching.sort(key=lambda r: r.timestamp, reverse=True)

                changed = False
                for record in matching:
                    if record.operation == "update" and record.properties_before is not None:
                        node.properties = dict(record.properties_before)
                        changed = True
                    elif record.operation == "delete":
                        node.is_active = True
                        changed = True
                    elif record.operation == "add":
                        gs.remove_node(node.node_id)
                        result["graph_nodes"] += 1
                        changed = False
                        break
                    elif record.operation == "restore":
                        node.is_active = False
                        changed = True

                if changed:
                    node.history = [
                        r for r in node.history
                        if r.operation_id != operation_id
                    ]
                    gs.add_node(node)
                    result["graph_nodes"] += 1

            # Phase 2: edges
            processed_edges = set()
            remaining_nodes = gs.list_nodes()
            for node in remaining_nodes:
                edges = gs.get_edges(node.node_id, direction="outgoing")
                for edge in edges:
                    edge_key = (edge.source_id, edge.target_id, edge.edge_type)
                    if edge_key in processed_edges:
                        continue
                    processed_edges.add(edge_key)

                    matching = [
                        r for r in edge.history
                        if r.operation_id == operation_id
                    ]
                    if not matching:
                        continue

                    matching.sort(key=lambda r: r.timestamp, reverse=True)

                    changed = False
                    for record in matching:
                        if record.operation == "update" and record.properties_before is not None:
                            edge.properties = dict(record.properties_before)
                            changed = True
                        elif record.operation == "delete":
                            edge.is_active = True
                            changed = True
                        elif record.operation == "add":
                            gs.remove_edge(edge.source_id, edge.target_id, edge.edge_type)
                            result["graph_edges"] += 1
                            changed = False
                            break
                        elif record.operation == "restore":
                            edge.is_active = False
                            changed = True

                    if changed:
                        edge.history = [
                            r for r in edge.history
                            if r.operation_id != operation_id
                        ]
                        gs.remove_edge(edge.source_id, edge.target_id, edge.edge_type)
                        gs.add_edge(edge)
                        result["graph_edges"] += 1

        total = result["pieces"] + result["metadata"] + result["graph_nodes"] + result["graph_edges"]
        self._log_kb_operation(
            rb_op_id,
            f"Rolled back operation {operation_id}",
            "KnowledgeBase.rollback_operation",
            total,
        )
        return result

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
