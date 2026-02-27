"""Ingestion sub-package for the knowledge module.

Provides document ingestion pipeline components including chunking,
deduplication, merge strategies, validation, skill synthesis,
knowledge lifecycle management, and supporting infrastructure.
"""

from .taxonomy import (
    DOMAIN_TAXONOMY,
    get_all_domains,
    get_domain_tags,
    validate_domain,
    validate_tags,
    format_taxonomy_for_prompt,
)
from .chunker import (
    DocumentChunk,
    ChunkerConfig,
    MarkdownChunker,
    chunk_markdown_file,
    estimate_tokens,
)
from .deduplicator import DedupConfig, ThreeTierDeduplicator
from .merge_strategy import MergeStrategyConfig, MergeStrategyManager
from .validator import ValidationConfig, KnowledgeValidator
from .skill_synthesizer import (
    SkillSynthesisConfig,
    SkillSynthesisResult,
    SkillSynthesizer,
)
from .knowledge_updater import UpdateConfig, KnowledgeUpdater
from .knowledge_deleter import DeleteConfig, ConfirmationRequiredError, KnowledgeDeleter
from .document_ingester import (
    DocumentIngester,
    IngestionResult,
    IngesterConfig,
    ingest_markdown_files,
    ingest_directory,
)
from .post_ingestion_merge_job import PostIngestionMergeJob
from .debug_session import (
    IngestionDebugSession,
    get_knowledge_base_dir,
    get_ingestion_runtime_dir,
    list_all_ingestion_sessions,
)

__all__ = [
    # Taxonomy
    "DOMAIN_TAXONOMY",
    "get_all_domains",
    "get_domain_tags",
    "validate_domain",
    "validate_tags",
    "format_taxonomy_for_prompt",
    # Chunker
    "DocumentChunk",
    "ChunkerConfig",
    "MarkdownChunker",
    "chunk_markdown_file",
    "estimate_tokens",
    # Deduplicator
    "DedupConfig",
    "ThreeTierDeduplicator",
    # Merge Strategy
    "MergeStrategyConfig",
    "MergeStrategyManager",
    # Validator
    "ValidationConfig",
    "KnowledgeValidator",
    # Skill Synthesizer
    "SkillSynthesisConfig",
    "SkillSynthesisResult",
    "SkillSynthesizer",
    # Knowledge Updater
    "UpdateConfig",
    "KnowledgeUpdater",
    # Knowledge Deleter
    "DeleteConfig",
    "ConfirmationRequiredError",
    "KnowledgeDeleter",
    # Document Ingester
    "DocumentIngester",
    "IngestionResult",
    "IngesterConfig",
    "ingest_markdown_files",
    "ingest_directory",
    # Post-Ingestion Merge Job
    "PostIngestionMergeJob",
    # Debug Session
    "IngestionDebugSession",
    "get_knowledge_base_dir",
    "get_ingestion_runtime_dir",
    "list_all_ingestion_sessions",
]
