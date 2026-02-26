"""
Agent Knowledge Base — a three-layer knowledge retrieval framework for agents.

This module provides a retrieval-oriented knowledge framework organized into
three complementary layers:

1. **Metadata Layer** — Structured key-value pairs for entities (name, location,
   preferences). Retrieved by key lookup or full iteration.

2. **Knowledge Pieces Layer** — Unstructured text chunks (facts, procedures,
   notes, instructions). Retrieved via keyword/BM25 search or semantic similarity.

3. **Entity Graph Layer** — Entities with typed relationships linking metadata
   and knowledge pieces together. Enables relationship-aware retrieval.

The ``KnowledgeBase`` orchestrator coordinates retrieval across all three layers
and implements ``__call__`` for seamless integration with ``Agent.user_profile``
and ``Agent.context``.

Public API
----------

Data Models:
    KnowledgePiece, KnowledgeType, EntityMetadata, GraphNode, GraphEdge

Store ABCs:
    MetadataStore, KnowledgePieceStore, EntityGraphStore

Adapter-Based Stores:
    KeyValueMetadataStore, RetrievalKnowledgePieceStore, GraphServiceEntityGraphStore

Orchestrator:
    KnowledgeBase

Data Loading:
    KnowledgeDataLoader

Provider:
    KnowledgeProvider, InfoType

Ingestion:
    KnowledgeIngestionCLI

Formatter:
    KnowledgeFormatter, RetrievalResult

Utilities:
    sanitize_id, unsanitize_id, parse_entity_type

Requirements: 6.1, 6.2, 6.3
"""

# ── Data Models ──────────────────────────────────────────────────────────
from science_modeling_tools.knowledge.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from science_modeling_tools.knowledge.models.entity_metadata import EntityMetadata
from rich_python_utils.service_utils.graph_service.graph_node import (
    GraphNode,
    GraphEdge,
)

# ── Store ABCs ───────────────────────────────────────────────────────────
from science_modeling_tools.knowledge.stores.metadata.base import MetadataStore
from science_modeling_tools.knowledge.stores.pieces.base import KnowledgePieceStore
from science_modeling_tools.knowledge.stores.graph.base import EntityGraphStore

# ── Adapter-Based Store Implementations ──────────────────────────────────
from science_modeling_tools.knowledge.stores.metadata.keyvalue_adapter import (
    KeyValueMetadataStore,
)
from science_modeling_tools.knowledge.stores.pieces.retrieval_adapter import (
    RetrievalKnowledgePieceStore,
)
from science_modeling_tools.knowledge.stores.graph.graph_adapter import (
    GraphServiceEntityGraphStore,
)

# ── Orchestrator ─────────────────────────────────────────────────────────
from science_modeling_tools.knowledge.knowledge_base import KnowledgeBase

# ── Data Loading ─────────────────────────────────────────────────────────
from science_modeling_tools.knowledge.data_loader import KnowledgeDataLoader

# ── Provider ─────────────────────────────────────────────────────────────
from science_modeling_tools.knowledge.provider import KnowledgeProvider, InfoType

# ── Ingestion ────────────────────────────────────────────────────────────
from science_modeling_tools.knowledge.ingestion_cli import KnowledgeIngestionCLI

# ── Formatter ────────────────────────────────────────────────────────────
from science_modeling_tools.knowledge.formatter import (
    KnowledgeFormatter,
    RetrievalResult,
)

# ── Utilities ────────────────────────────────────────────────────────────
from science_modeling_tools.knowledge.utils import (
    sanitize_id,
    unsanitize_id,
    parse_entity_type,
)

__all__ = [
    # Data models
    "KnowledgePiece",
    "KnowledgeType",
    "EntityMetadata",
    "GraphNode",
    "GraphEdge",
    # Store ABCs
    "MetadataStore",
    "KnowledgePieceStore",
    "EntityGraphStore",
    # Adapter-based stores
    "KeyValueMetadataStore",
    "RetrievalKnowledgePieceStore",
    "GraphServiceEntityGraphStore",
    # Orchestrator
    "KnowledgeBase",
    # Data Loading
    "KnowledgeDataLoader",
    # Provider
    "KnowledgeProvider",
    "InfoType",
    # Ingestion
    "KnowledgeIngestionCLI",
    # Formatter
    "KnowledgeFormatter",
    "RetrievalResult",
    # Utilities
    "sanitize_id",
    "unsanitize_id",
    "parse_entity_type",
]
