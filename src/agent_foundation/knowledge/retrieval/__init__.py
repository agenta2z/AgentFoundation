"""Retrieval sub-package for the knowledge module.

Provides core retrieval components including the KnowledgeBase orchestrator,
composable RetrievalPipeline with pluggable post-processors, data models,
store ABCs and adapters, hybrid search, MMR re-ranking, temporal decay,
budget-aware knowledge provider, formatter, data loader, utilities, and
ingestion CLI.
"""

# ── Data Loading ─────────────────────────────────────────────────────────
from .data_loader import KnowledgeDataLoader

# ── Formatter ────────────────────────────────────────────────────────────
from .formatter import KnowledgeFormatter, RetrievalResult

# ── Hybrid Search ────────────────────────────────────────────────────────
from .hybrid_search import HybridRetriever, HybridSearchConfig

# ── Ingestion CLI (legacy) ──────────────────────────────────────────────
from .ingestion_cli import KnowledgeIngestionCLI

# ── Orchestrator ─────────────────────────────────────────────────────────
from .knowledge_base import KnowledgeBase

# ── Knowledge Consolidator ──────────────────────────────────────────────
from .knowledge_consolidator import KnowledgeConsolidator

# ── Budget-Aware Provider ────────────────────────────────────────────────
from .knowledge_provider import BudgetAwareKnowledgeProvider

# ── MMR Re-ranking ───────────────────────────────────────────────────────
from .mmr_reranking import apply_mmr_reranking, MMRConfig
from .models.entity_metadata import EntityMetadata
from .models.enums import (
    ConsolidationMode,
    DedupAction,
    DeleteMode,
    MergeAction,
    MergeStrategy,
    MergeType,
    Space,
    SuggestionStatus,
    UpdateAction,
    ValidationStatus,
)

# ── Data Models ──────────────────────────────────────────────────────────
from .models.knowledge_piece import KnowledgePiece, KnowledgeType
from .models.results import (
    DedupResult,
    MergeCandidate,
    MergeJobResult,
    MergeResult,
    OperationResult,
    ScoredPiece,
    ValidationResult,
)
from .post_processors import (
    AggregatingPostProcessor,
    BudgetAwarePostProcessor,
    FlatStringPostProcessor,
    GroupedDictPostProcessor,
)

# ── Provider ─────────────────────────────────────────────────────────────
from .provider import InfoType
# ── Retrieval Pipeline ──────────────────────────────────────────────────

# ── Query Decomposition & Agentic Models ─────────────────────────────────
from .retrieval_pipeline import (
    AgenticRetrievalResult,
    create_domain_decomposer,
    create_llm_decomposer,
    PostProcessor,
    QueryExpander,
    RetrievalPipeline,
    SubQuery,
)
from .stores.graph.base import EntityGraphStore
from .stores.graph.graph_adapter import GraphServiceEntityGraphStore

# ── Store ABCs ───────────────────────────────────────────────────────────
from .stores.metadata.base import MetadataStore

# ── Adapter-Based Store Implementations ──────────────────────────────────
from .stores.metadata.keyvalue_adapter import KeyValueMetadataStore
from .stores.pieces.base import KnowledgePieceStore
from .stores.pieces.lancedb_store import LanceDBKnowledgePieceStore
from .stores.pieces.retrieval_adapter import RetrievalKnowledgePieceStore

# ── Temporal Decay ───────────────────────────────────────────────────────
from .temporal_decay import apply_temporal_decay, TemporalDecayConfig

# ── Utilities ────────────────────────────────────────────────────────────
from .utils import (
    cosine_similarity,
    count_tokens,
    parse_entity_type,
    sanitize_id,
    unsanitize_id,
)

__all__ = [
    # Data models
    "KnowledgePiece",
    "KnowledgeType",
    "EntityMetadata",
    # Enums
    "Space",
    "MergeStrategy",
    "MergeAction",
    "DedupAction",
    "MergeType",
    "ValidationStatus",
    "SuggestionStatus",
    "UpdateAction",
    "DeleteMode",
    "ConsolidationMode",
    # Result types
    "DedupResult",
    "MergeCandidate",
    "MergeResult",
    "ValidationResult",
    "ScoredPiece",
    "MergeJobResult",
    "OperationResult",
    # Store ABCs
    "MetadataStore",
    "KnowledgePieceStore",
    "EntityGraphStore",
    # Adapter-based stores
    "KeyValueMetadataStore",
    "RetrievalKnowledgePieceStore",
    "GraphServiceEntityGraphStore",
    "LanceDBKnowledgePieceStore",
    # Orchestrator
    "KnowledgeBase",
    # Data Loading
    "KnowledgeDataLoader",
    # Provider
    "InfoType",
    "BudgetAwareKnowledgeProvider",
    # Knowledge Consolidator
    "KnowledgeConsolidator",
    # Hybrid Search
    "HybridSearchConfig",
    "HybridRetriever",
    # MMR Re-ranking
    "MMRConfig",
    "apply_mmr_reranking",
    # Temporal Decay
    "TemporalDecayConfig",
    "apply_temporal_decay",
    # Query Decomposition & Agentic Models
    "SubQuery",
    "AgenticRetrievalResult",
    "create_domain_decomposer",
    "create_llm_decomposer",
    # Retrieval Pipeline
    "RetrievalPipeline",
    "QueryExpander",
    "PostProcessor",
    "FlatStringPostProcessor",
    "GroupedDictPostProcessor",
    "AggregatingPostProcessor",
    "BudgetAwarePostProcessor",
    # Ingestion CLI (legacy)
    "KnowledgeIngestionCLI",
    # Formatter
    "KnowledgeFormatter",
    "RetrievalResult",
    # Utilities
    "sanitize_id",
    "unsanitize_id",
    "parse_entity_type",
    "cosine_similarity",
    "count_tokens",
]
