"""Ingestion sub-package for the knowledge module.

Provides document ingestion pipeline components including chunking,
deduplication, merge strategies, validation, skill synthesis,
knowledge lifecycle management, and supporting infrastructure.
"""

from .chunker import (
    chunk_markdown_file,
    ChunkerConfig,
    DocumentChunk,
    estimate_tokens,
    MarkdownChunker,
)
from .debug_session import (
    get_ingestion_runtime_dir,
    get_knowledge_base_dir,
    IngestionDebugSession,
    list_all_ingestion_sessions,
)
from .deduplicator import DedupConfig, ThreeTierDeduplicator
from .document_ingester import (
    DocumentIngester,
    ingest_directory,
    ingest_markdown_files,
    IngesterConfig,
    IngestionResult,
)
from .knowledge_deleter import ConfirmationRequiredError, DeleteConfig, KnowledgeDeleter
from .knowledge_updater import KnowledgeUpdater, UpdateConfig
from .merge_strategy import MergeStrategyConfig, MergeStrategyManager
from .post_ingestion_merge_job import PostIngestionMergeJob
from .skill_synthesizer import (
    SkillSynthesisConfig,
    SkillSynthesisResult,
    SkillSynthesizer,
)
from .space_classifier import ClassificationResult, SpaceClassifier, SpaceRule
from .space_migration import MigrationReport, SpaceMigrationUtility
from .taxonomy import (
    DOMAIN_TAXONOMY,
    format_taxonomy_for_prompt,
    get_all_domains,
    get_domain_tags,
    validate_domain,
    validate_tags,
)
from .validator import KnowledgeValidator, ValidationConfig

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
    # Space Classifier
    "SpaceClassifier",
    "SpaceRule",
    "ClassificationResult",
    # Space Migration
    "SpaceMigrationUtility",
    "MigrationReport",
]
