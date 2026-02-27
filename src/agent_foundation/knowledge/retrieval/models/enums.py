"""
Enumerations for the knowledge system.

Provides type-safe enums for spaces, merge strategies, validation status,
and other categorical fields in the knowledge system.
"""

from enum import StrEnum


class Space(StrEnum):
    """Knowledge space for scoping and lifecycle management."""

    MAIN = "main"
    PERSONAL = "personal"
    DEVELOPMENTAL = "developmental"


class MergeStrategy(StrEnum):
    """Strategy for handling potential duplicates during ingestion."""

    AUTO_MERGE_ON_INGEST = "auto-merge-on-ingest"
    SUGGESTION_ON_INGEST = "suggestion-on-ingest"
    POST_INGESTION_AUTO = "post-ingestion-auto"
    POST_INGESTION_SUGGESTION = "post-ingestion-suggestion"
    MANUAL_ONLY = "manual-only"


class MergeAction(StrEnum):
    """Result of applying a merge strategy."""

    MERGED = "merged"
    PENDING_REVIEW = "pending_review"
    NO_CANDIDATES = "no_candidates"
    DEFERRED = "deferred"
    NO_AUTO_MERGE = "no_auto_merge"
    ERROR = "error"


class DedupAction(StrEnum):
    """Decision from three-tier deduplication."""

    ADD = "add"
    UPDATE = "update"
    MERGE = "merge"
    NO_OP = "no_op"


class MergeType(StrEnum):
    """Type of merge relationship between two pieces."""

    DUPLICATE = "duplicate"
    SUPERSET = "superset"
    SUBSET = "subset"
    OVERLAPPING = "overlapping"
    UPDATE = "update"
    UNRELATED = "unrelated"


class ValidationStatus(StrEnum):
    """Status of knowledge validation."""

    NOT_VALIDATED = "not_validated"
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


class SuggestionStatus(StrEnum):
    """Status of a merge suggestion."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class UpdateAction(StrEnum):
    """Action from update intent analysis."""

    REPLACE = "replace"
    MERGE = "merge"
    NO_CHANGE = "no_change"


class DeleteMode(StrEnum):
    """Delete mode."""

    SOFT = "soft"
    HARD = "hard"
