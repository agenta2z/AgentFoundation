"""
Agent Knowledge Base — a comprehensive knowledge retrieval and ingestion framework.

This module provides a retrieval-oriented knowledge framework organized into
three complementary layers, plus a full ingestion pipeline for processing
documents into structured knowledge.

Retrieval Layers
----------------

1. **Metadata Layer** — Structured key-value pairs for entities (name, location,
   preferences). Retrieved by key lookup or full iteration.

2. **Knowledge Pieces Layer** — Unstructured text chunks (facts, procedures,
   notes, instructions). Retrieved via keyword/BM25 search or semantic similarity.

3. **Entity Graph Layer** — Entities with typed relationships linking metadata
   and knowledge pieces together. Enables relationship-aware retrieval.

Advanced Retrieval
------------------

- **Hybrid Search** — Combines vector similarity and keyword search via RRF.
- **MMR Re-ranking** — Maximal Marginal Relevance for diversity.
- **Temporal Decay** — Exponential decay scoring for freshness.
- **Retrieval Pipeline** — Composable pipeline with pluggable query expansion and post-processing.
- **Budget-Aware Provider** — Per-info-type token budget enforcement.

Ingestion Pipeline
------------------

- **Markdown Chunker** — Header-aware document splitting.
- **Three-Tier Deduplication** — Hash, embedding, and LLM-based dedup.
- **Merge Strategies** — Configurable duplicate handling.
- **Validation** — Security, privacy, and semantic quality checks.
- **Skill Synthesis** — Automatic procedural skill generation.
- **Knowledge Lifecycle** — Update and delete with versioning.

Public API
----------

Data Models:
    KnowledgePiece, KnowledgeType, EntityMetadata, GraphNode, GraphEdge

Enums:
    Space, MergeStrategy, MergeAction, DedupAction, MergeType,
    ValidationStatus, SuggestionStatus, UpdateAction, DeleteMode,
    ConsolidationMode

Result Types:
    DedupResult, MergeCandidate, MergeResult, ValidationResult,
    ScoredPiece, MergeJobResult, OperationResult

Store ABCs:
    MetadataStore, KnowledgePieceStore, EntityGraphStore

Adapter-Based Stores:
    KeyValueMetadataStore, RetrievalKnowledgePieceStore,
    GraphServiceEntityGraphStore, LanceDBKnowledgePieceStore

Orchestrator:
    KnowledgeBase

Data Loading:
    KnowledgeDataLoader

Provider:
    InfoType, BudgetAwareKnowledgeProvider

Retrieval:
    HybridSearchConfig, HybridRetriever,
    MMRConfig, apply_mmr_reranking,
    TemporalDecayConfig, apply_temporal_decay,
    SubQuery, AgenticRetrievalResult,
    create_domain_decomposer, create_llm_decomposer,
    RetrievalPipeline, QueryExpander, PostProcessor,
    FlatStringPostProcessor, GroupedDictPostProcessor,
    AggregatingPostProcessor, BudgetAwarePostProcessor

Ingestion CLI:
    KnowledgeIngestionCLI

Formatter:
    KnowledgeFormatter, RetrievalResult

Taxonomy:
    DOMAIN_TAXONOMY, get_all_domains, get_domain_tags,
    validate_domain, validate_tags, format_taxonomy_for_prompt

Chunking:
    DocumentChunk, ChunkerConfig, MarkdownChunker,
    chunk_markdown_file, estimate_tokens

Deduplication:
    DedupConfig, ThreeTierDeduplicator

Merge:
    MergeStrategyConfig, MergeStrategyManager

Validation:
    ValidationConfig, KnowledgeValidator

Skill Synthesis:
    SkillSynthesisConfig, SkillSynthesisResult, SkillSynthesizer

Knowledge Lifecycle:
    UpdateConfig, KnowledgeUpdater,
    DeleteConfig, ConfirmationRequiredError, KnowledgeDeleter

Knowledge Packs:
    KnowledgePack, PackStatus, PackSource, PackInstallResult, PackManagerConfig,
    KnowledgePackManager, ClawhubClient, ClawhubPackAdapter, parse_skill_md,
    LocalPackLoader

Pipeline:
    DocumentIngester, PostIngestionMergeJob, IngestionDebugSession

Space Classification & Migration:
    SpaceClassifier, SpaceRule, ClassificationResult,
    SpaceMigrationUtility, MigrationReport

Utilities:
    sanitize_id, unsanitize_id, parse_entity_type, cosine_similarity, count_tokens

Requirements: All
"""

# ── Chunking ─────────────────────────────────────────────────────────────
from agent_foundation.knowledge.ingestion.chunker import (
    chunk_markdown_file,
    ChunkerConfig,
    DocumentChunk,
    estimate_tokens,
    MarkdownChunker,
)
from agent_foundation.knowledge.ingestion.debug_session import IngestionDebugSession

# ── Deduplication ────────────────────────────────────────────────────────
from agent_foundation.knowledge.ingestion.deduplicator import (
    DedupConfig,
    ThreeTierDeduplicator,
)

# ── Pipeline Orchestration ───────────────────────────────────────────────
from agent_foundation.knowledge.ingestion.document_ingester import DocumentIngester
from agent_foundation.knowledge.ingestion.knowledge_deleter import (
    ConfirmationRequiredError,
    DeleteConfig,
    KnowledgeDeleter,
)

# ── Knowledge Lifecycle ──────────────────────────────────────────────────
from agent_foundation.knowledge.ingestion.knowledge_updater import (
    KnowledgeUpdater,
    UpdateConfig,
)

# ── Merge Strategy ───────────────────────────────────────────────────────
from agent_foundation.knowledge.ingestion.merge_strategy import (
    MergeStrategyConfig,
    MergeStrategyManager,
)
from agent_foundation.knowledge.ingestion.post_ingestion_merge_job import (
    PostIngestionMergeJob,
)

# ── Skill Synthesis ──────────────────────────────────────────────────────
from agent_foundation.knowledge.ingestion.skill_synthesizer import (
    SkillSynthesisConfig,
    SkillSynthesisResult,
    SkillSynthesizer,
)

# ── Space Classification & Migration ─────────────────────────────────────
from agent_foundation.knowledge.ingestion.space_classifier import (
    ClassificationResult,
    SpaceClassifier,
    SpaceRule,
)
from agent_foundation.knowledge.ingestion.space_migration import (
    MigrationReport,
    SpaceMigrationUtility,
)

# ── Taxonomy ─────────────────────────────────────────────────────────────
from agent_foundation.knowledge.ingestion.taxonomy import (
    DOMAIN_TAXONOMY,
    format_taxonomy_for_prompt,
    get_all_domains,
    get_domain_tags,
    validate_domain,
    validate_tags,
)

# ── Validation ───────────────────────────────────────────────────────────
from agent_foundation.knowledge.ingestion.validator import (
    KnowledgeValidator,
    ValidationConfig,
)

# ── Knowledge Packs ─────────────────────────────────────────────────
from agent_foundation.knowledge.packs import (
    ClawhubClient,
    ClawhubPackAdapter,
    KnowledgePack,
    KnowledgePackManager,
    LocalPackLoader,
    PackInstallResult,
    PackManagerConfig,
    PackSource,
    PackStatus,
    parse_skill_md,
)

# ── Data Loading ─────────────────────────────────────────────────────────
from agent_foundation.knowledge.retrieval.data_loader import KnowledgeDataLoader

# ── Formatter ────────────────────────────────────────────────────────────
from agent_foundation.knowledge.retrieval.formatter import (
    KnowledgeFormatter,
    RetrievalResult,
)

# ── Hybrid Search ────────────────────────────────────────────────────────
from agent_foundation.knowledge.retrieval.hybrid_search import (
    HybridRetriever,
    HybridSearchConfig,
)

# ── Ingestion CLI (legacy) ──────────────────────────────────────────────
from agent_foundation.knowledge.retrieval.ingestion_cli import KnowledgeIngestionCLI

# ── Orchestrator ─────────────────────────────────────────────────────────
from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase

# ── Budget-Aware Provider ────────────────────────────────────────────────
from agent_foundation.knowledge.retrieval.knowledge_provider import (
    BudgetAwareKnowledgeProvider,
)

# ── MMR Re-ranking ───────────────────────────────────────────────────────
from agent_foundation.knowledge.retrieval.mmr_reranking import (
    apply_mmr_reranking,
    MMRConfig,
)
from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata
# ── Consolidation Mode ───────────────────────────────────────────────────

# ── Enums ────────────────────────────────────────────────────────────────
from agent_foundation.knowledge.retrieval.models.enums import (
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
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)

# ── Result Types ─────────────────────────────────────────────────────────
from agent_foundation.knowledge.retrieval.models.results import (
    DedupResult,
    MergeCandidate,
    MergeJobResult,
    MergeResult,
    OperationResult,
    ScoredPiece,
    ValidationResult,
)
from agent_foundation.knowledge.retrieval.post_processors import (
    AggregatingPostProcessor,
    BudgetAwarePostProcessor,
    FlatStringPostProcessor,
    GroupedDictPostProcessor,
)

# ── Provider ─────────────────────────────────────────────────────────────
from agent_foundation.knowledge.retrieval.provider import InfoType
# ── Retrieval Pipeline ──────────────────────────────────────────────────

# ── Query Decomposition & Agentic Models ─────────────────────────────────
from agent_foundation.knowledge.retrieval.retrieval_pipeline import (
    AgenticRetrievalResult,
    create_domain_decomposer,
    create_llm_decomposer,
    PostProcessor,
    QueryExpander,
    RetrievalPipeline,
    SubQuery,
)
from agent_foundation.knowledge.retrieval.stores.graph.base import EntityGraphStore
from agent_foundation.knowledge.retrieval.stores.graph.graph_adapter import (
    GraphServiceEntityGraphStore,
)

# ── Store ABCs ───────────────────────────────────────────────────────────
from agent_foundation.knowledge.retrieval.stores.metadata.base import MetadataStore

# ── Adapter-Based Store Implementations ──────────────────────────────────
from agent_foundation.knowledge.retrieval.stores.metadata.keyvalue_adapter import (
    KeyValueMetadataStore,
)
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore
from agent_foundation.knowledge.retrieval.stores.pieces.lancedb_store import (
    LanceDBKnowledgePieceStore,
)
from agent_foundation.knowledge.retrieval.stores.pieces.retrieval_adapter import (
    RetrievalKnowledgePieceStore,
)

# ── Temporal Decay ───────────────────────────────────────────────────────
from agent_foundation.knowledge.retrieval.temporal_decay import (
    apply_temporal_decay,
    TemporalDecayConfig,
)

# ── Utilities ────────────────────────────────────────────────────────────
from agent_foundation.knowledge.retrieval.utils import (
    cosine_similarity,
    count_tokens,
    parse_entity_type,
    sanitize_id,
    unsanitize_id,
)
from rich_python_utils.service_utils.graph_service.graph_node import (
    GraphEdge,
    GraphNode,
)

__all__ = [
    # Data models
    "KnowledgePiece",
    "KnowledgeType",
    "EntityMetadata",
    "GraphNode",
    "GraphEdge",
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
    # Taxonomy
    "DOMAIN_TAXONOMY",
    "get_all_domains",
    "get_domain_tags",
    "validate_domain",
    "validate_tags",
    "format_taxonomy_for_prompt",
    # Chunking
    "DocumentChunk",
    "ChunkerConfig",
    "MarkdownChunker",
    "chunk_markdown_file",
    "estimate_tokens",
    # Deduplication
    "DedupConfig",
    "ThreeTierDeduplicator",
    # Merge Strategy
    "MergeStrategyConfig",
    "MergeStrategyManager",
    # Validation
    "ValidationConfig",
    "KnowledgeValidator",
    # Skill Synthesis
    "SkillSynthesisConfig",
    "SkillSynthesisResult",
    "SkillSynthesizer",
    # Knowledge Lifecycle
    "UpdateConfig",
    "KnowledgeUpdater",
    "DeleteConfig",
    "ConfirmationRequiredError",
    "KnowledgeDeleter",
    # Knowledge Packs
    "KnowledgePack",
    "PackStatus",
    "PackSource",
    "PackInstallResult",
    "PackManagerConfig",
    "KnowledgePackManager",
    "ClawhubClient",
    "ClawhubPackAdapter",
    "parse_skill_md",
    "LocalPackLoader",
    # Pipeline Orchestration
    "DocumentIngester",
    "PostIngestionMergeJob",
    "IngestionDebugSession",
    # Space Classification & Migration
    "SpaceClassifier",
    "SpaceRule",
    "ClassificationResult",
    "SpaceMigrationUtility",
    "MigrationReport",
    # Utilities
    "sanitize_id",
    "unsanitize_id",
    "parse_entity_type",
    "cosine_similarity",
    "count_tokens",
]
