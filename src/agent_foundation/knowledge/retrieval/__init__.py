"""Retrieval sub-package for the knowledge module.

Provides core retrieval components including the KnowledgeBase orchestrator,
data models, store ABCs and adapters, hybrid search, MMR re-ranking,
temporal decay, agentic multi-query retrieval, budget-aware knowledge
provider, formatter, data loader, utilities, and ingestion CLI.
"""

# ── Data Models ──────────────────────────────────────────────────────────
from .models.knowledge_piece import KnowledgePiece, KnowledgeType
from .models.entity_metadata import EntityMetadata
from .models.enums import (
    Space,
    MergeStrategy,
    MergeAction,
    DedupAction,
    MergeType,
    ValidationStatus,
    SuggestionStatus,
    UpdateAction,
    DeleteMode,
)
from .models.results import (
    DedupResult,
    MergeCandidate,
    MergeResult,
    ValidationResult,
    ScoredPiece,
    MergeJobResult,
    OperationResult,
)

# ── Store ABCs ───────────────────────────────────────────────────────────
from .stores.metadata.base import MetadataStore
from .stores.pieces.base import KnowledgePieceStore
from .stores.graph.base import EntityGraphStore

# ── Adapter-Based Store Implementations ──────────────────────────────────
from .stores.metadata.keyvalue_adapter import KeyValueMetadataStore
from .stores.pieces.retrieval_adapter import RetrievalKnowledgePieceStore
from .stores.graph.graph_adapter import GraphServiceEntityGraphStore
from .stores.pieces.lancedb_store import LanceDBKnowledgePieceStore

# ── Orchestrator ─────────────────────────────────────────────────────────
from .knowledge_base import KnowledgeBase

# ── Data Loading ─────────────────────────────────────────────────────────
from .data_loader import KnowledgeDataLoader

# ── Provider ─────────────────────────────────────────────────────────────
from .provider import KnowledgeProvider, InfoType

# ── Budget-Aware Provider ────────────────────────────────────────────────
from .knowledge_provider import BudgetAwareKnowledgeProvider

# ── Hybrid Search ────────────────────────────────────────────────────────
from .hybrid_search import HybridSearchConfig, HybridRetriever

# ── MMR Re-ranking ───────────────────────────────────────────────────────
from .mmr_reranking import MMRConfig, apply_mmr_reranking

# ── Temporal Decay ───────────────────────────────────────────────────────
from .temporal_decay import TemporalDecayConfig, apply_temporal_decay

# ── Agentic Retriever ───────────────────────────────────────────────────
from .agentic_retriever import (
    SubQuery,
    AgenticRetrievalResult,
    AgenticRetriever,
    create_domain_decomposer,
    create_llm_decomposer,
)

# ── Ingestion CLI (legacy) ──────────────────────────────────────────────
from .ingestion_cli import KnowledgeIngestionCLI

# ── Formatter ────────────────────────────────────────────────────────────
from .formatter import KnowledgeFormatter, RetrievalResult

# ── Utilities ────────────────────────────────────────────────────────────
from .utils import (
    sanitize_id,
    unsanitize_id,
    parse_entity_type,
    cosine_similarity,
    count_tokens,
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
    "KnowledgeProvider",
    "InfoType",
    "BudgetAwareKnowledgeProvider",
    # Hybrid Search
    "HybridSearchConfig",
    "HybridRetriever",
    # MMR Re-ranking
    "MMRConfig",
    "apply_mmr_reranking",
    # Temporal Decay
    "TemporalDecayConfig",
    "apply_temporal_decay",
    # Agentic Retriever
    "SubQuery",
    "AgenticRetrievalResult",
    "AgenticRetriever",
    "create_domain_decomposer",
    "create_llm_decomposer",
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
