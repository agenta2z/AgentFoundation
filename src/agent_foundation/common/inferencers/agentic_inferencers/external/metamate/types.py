# pyre-strict
"""
Type Definitions for Metamate Integration.

This module contains all type definitions used across the Metamate
integration, including research types, knowledge types, event types,
and common types.

IMPORTANT: Tool names are canonical from MetamateAgentEngineTypes.php
Verify at https://www.internalfb.com/metamate/agent/tools before implementation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# =============================================================================
# Common Types
# =============================================================================


class Permission(Enum):
    """Permissions for Metamate tool access."""

    READ_WIKI = "read_wiki"
    READ_WORKPLACE = "read_workplace"
    READ_DIFF = "read_diff"
    READ_TASK = "read_task"
    READ_SEV = "read_sev"
    EXECUTE_RESEARCH = "execute_research"
    EXECUTE_DEBUG = "execute_debug"


@dataclass
class MetamateConfig:
    """
    Configuration for Metamate integration.

    Attributes:
        connection_timeout_seconds: Timeout for establishing connection.
        tool_timeout_seconds: Default timeout for tool execution.
        max_retries: Maximum number of retry attempts for failed operations.
        retry_delay_seconds: Delay between retry attempts.
        enable_caching: Whether to cache repeated tool calls.
        cache_ttl_seconds: Time-to-live for cached results.
        fallback_enabled: Whether to enable fallback to alternative services.
        max_concurrent_searches: Maximum concurrent searches per request.
    """

    connection_timeout_seconds: float = 30.0
    tool_timeout_seconds: float = 300.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    fallback_enabled: bool = True
    max_concurrent_searches: int = 5


@dataclass
class Citation:
    """
    Citation for a source referenced in research results.

    Attributes:
        source_id: Unique identifier for the source.
        source_type: Type of source (wiki, workplace, diff, task, sev, etc.).
        title: Title of the source document.
        url: URL to access the source.
        snippet: Relevant excerpt from the source.
        relevance_score: Score indicating relevance to the query (0.0 to 1.0).
        author: Author of the source if available.
        created_at: Timestamp when the source was created.
        metadata: Additional source-specific metadata.
    """

    source_id: str
    source_type: str
    title: str
    url: str
    snippet: str
    relevance_score: float = 0.0
    author: str | None = None
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Finding:
    """
    A single finding from research synthesis.

    Attributes:
        finding_id: Unique identifier for the finding.
        content: The main content/text of the finding.
        confidence_score: Confidence in the finding (0.0 to 1.0).
        citations: Sources supporting this finding.
        category: Category/type of finding.
        metadata: Additional finding metadata.
    """

    finding_id: str
    content: str
    confidence_score: float
    citations: list[Citation]
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResponse:
    """
    Response from a Metamate tool invocation.

    Attributes:
        success: Whether the tool execution succeeded.
        tool_name: Name of the tool that was executed.
        data: The response data from the tool.
        error_message: Error message if execution failed.
        execution_time_ms: Execution time in milliseconds.
        metadata: Additional response metadata.
    """

    success: bool
    tool_name: str
    data: Any = None
    error_message: str | None = None
    execution_time_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Research Types
# =============================================================================


class QueryType(Enum):
    """Type of research query."""

    EXTERNAL = "external"  # External academic/web research
    INTERNAL = "internal"  # Internal Meta docs research
    HYBRID = "hybrid"  # Both external and internal


class ResearchEventType(Enum):
    """Types of events emitted during research execution."""

    # Planning phase
    PLAN_CREATED = "plan_created"
    PLAN_UPDATED = "plan_updated"

    # Gathering phase
    GATHERING_STARTED = "gathering_started"
    SOURCE_FOUND = "source_found"
    GATHERING_COMPLETE = "gathering_complete"

    # Synthesis phase
    SYNTHESIS_STARTED = "synthesis_started"
    FINDING_GENERATED = "finding_generated"
    SYNTHESIS_COMPLETE = "synthesis_complete"

    # Report phase
    REPORT_STARTED = "report_started"
    REPORT_READY = "report_ready"

    # Error/fallback
    ERROR = "error"
    FALLBACK_ACTIVATED = "fallback_activated"

    # Progress updates
    PROGRESS_UPDATE = "progress_update"


@dataclass
class ResearchQuery:
    """
    A single research query.

    Attributes:
        query_text: The research question or topic.
        query_type: Type of research (external, internal, hybrid).
        max_sources: Maximum number of sources to retrieve.
        filters: Optional filters for the search.
        priority: Priority of this query (higher = more important).
    """

    query_text: str
    query_type: QueryType = QueryType.HYBRID
    max_sources: int = 10
    filters: dict[str, Any] = field(default_factory=dict)
    priority: int = 1


@dataclass
class ResearchPlan:
    """
    A plan for executing research across multiple queries.

    Attributes:
        plan_id: Unique identifier for the plan.
        queries: List of research queries to execute.
        total_estimated_time_seconds: Estimated time to complete.
        created_at: Timestamp when the plan was created.
        metadata: Additional plan metadata.
    """

    plan_id: str
    queries: list[ResearchQuery]
    total_estimated_time_seconds: int = 0
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchReport:
    """
    Final research report with all findings.

    Attributes:
        report_id: Unique identifier for the report.
        title: Title of the report.
        summary: Executive summary of findings.
        findings: List of findings from the research.
        all_citations: All citations used in the report.
        total_sources_searched: Number of sources searched.
        execution_time_ms: Total execution time in milliseconds.
        metadata: Additional report metadata.
    """

    report_id: str
    title: str
    summary: str
    findings: list[Finding]
    all_citations: list[Citation]
    total_sources_searched: int = 0
    execution_time_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchResults:
    """
    Complete research results including report and execution stats.

    Attributes:
        report: The final research report.
        queries_executed: Number of queries executed.
        sources_found: Total sources found.
        findings_generated: Number of findings generated.
        fallback_used: Whether fallback was activated.
        errors: List of errors encountered.
    """

    report: ResearchReport
    queries_executed: int = 0
    sources_found: int = 0
    findings_generated: int = 0
    fallback_used: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class ResearchEvent:
    """
    Event emitted during research execution.

    Attributes:
        type: Type of the event.
        timestamp: Unix timestamp of the event.
        message: Human-readable message.
        plan: Research plan (for PLAN_CREATED event).
        source: Source citation (for SOURCE_FOUND event).
        finding: Finding (for FINDING_GENERATED event).
        report: Research report (for REPORT_READY event).
        progress: Progress percentage (0.0 to 1.0).
        error: Error details if applicable.
        data: Additional event-specific data.
    """

    type: ResearchEventType
    timestamp: float = 0.0
    message: str = ""
    plan: ResearchPlan | None = None
    source: Citation | None = None
    finding: Finding | None = None
    report: ResearchReport | None = None
    progress: float | None = None
    error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Knowledge Types
# =============================================================================


class KnowledgeSource(Enum):
    """Sources of internal knowledge."""

    WIKI = "wiki"  # InternWiki pages, Static Docs
    WORKPLACE = "workplace"  # Workplace posts, Q&A
    DIFF = "diff"  # Phabricator diffs
    TASK = "task"  # Phabricator tasks
    SEV = "sev"  # SEV database


@dataclass
class KnowledgeQuery:
    """
    Query for internal knowledge search.

    Attributes:
        query_text: The search query text.
        sources: Specific sources to search (None = all sources).
        max_results_per_source: Maximum results per source.
        filters: Source-specific filters.
        include_snippets: Whether to include text snippets.
    """

    query_text: str
    sources: set[KnowledgeSource] | None = None
    max_results_per_source: int = 10
    filters: dict[str, Any] = field(default_factory=dict)
    include_snippets: bool = True


@dataclass
class KnowledgeResult:
    """
    A single result from knowledge search.

    Attributes:
        result_id: Unique identifier for the result.
        source: Source of the result.
        title: Title of the document/item.
        url: URL to access the result.
        snippet: Relevant text snippet.
        relevance_score: Relevance score (0.0 to 1.0).
        author: Author if available.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        metadata: Source-specific metadata.
    """

    result_id: str
    source: KnowledgeSource
    title: str
    url: str
    snippet: str
    relevance_score: float = 0.0
    author: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedKnowledgeResults:
    """
    Aggregated results from multiple knowledge sources.

    Attributes:
        results: List of knowledge results, ranked by relevance.
        sources_searched: Sources that were successfully searched.
        total_found: Total number of results found.
        query: Original query that was executed.
        execution_time_ms: Total execution time in milliseconds.
        errors: List of errors from failed sources.
    """

    results: list[KnowledgeResult]
    sources_searched: list[KnowledgeSource]
    total_found: int = 0
    query: KnowledgeQuery | None = None
    execution_time_ms: int = 0
    errors: list[str] | None = None


# =============================================================================
# Debug Types
# =============================================================================


@dataclass
class DebugContext:
    """
    Context for debugging assistance.

    Attributes:
        error_message: The error message to investigate.
        error_pattern: Pattern extracted from the error.
        file_path: File path where error occurred.
        line_number: Line number of the error.
        stack_trace: Full stack trace if available.
        metadata: Additional context metadata.
    """

    error_message: str
    error_pattern: str = ""
    file_path: str | None = None
    line_number: int | None = None
    stack_trace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DebugHint:
    """
    A debugging hint from the debug assistant.

    Attributes:
        hint_id: Unique identifier for the hint.
        hint_type: Type of hint (fix, investigation, workaround).
        content: The hint content.
        confidence: Confidence in the hint (0.0 to 1.0).
        source: Source of the hint (similar issue, diff, sev, etc.).
        citations: Supporting citations.
    """

    hint_id: str
    hint_type: str
    content: str
    confidence: float
    source: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class RelatedDiff:
    """
    A diff related to a debugging context.

    Attributes:
        diff_id: Phabricator diff ID.
        title: Diff title.
        url: URL to the diff.
        relevance_score: Relevance to the error.
        author: Diff author.
        files_changed: Files modified in the diff.
    """

    diff_id: str
    title: str
    url: str
    relevance_score: float
    author: str | None = None
    files_changed: list[str] = field(default_factory=list)


@dataclass
class DebugAssistantResults:
    """
    Results from the debug assistant.

    Attributes:
        context: Original debug context.
        hints: List of debugging hints.
        related_diffs: Related diffs that might have caused the issue.
        similar_issues: Similar issues found in tasks/SEVs.
        execution_time_ms: Total execution time.
    """

    context: DebugContext
    hints: list[DebugHint]
    related_diffs: list[RelatedDiff] = field(default_factory=list)
    similar_issues: list[KnowledgeResult] = field(default_factory=list)
    execution_time_ms: int = 0


# =============================================================================
# Q&A Types
# =============================================================================


@dataclass
class QAQuery:
    """
    Query for Platform Q&A.

    Attributes:
        question: The question to answer.
        context: Additional context for the question.
        max_citations: Maximum citations to include.
    """

    question: str
    context: str = ""
    max_citations: int = 5


@dataclass
class QAResponse:
    """
    Response from Platform Q&A.

    Attributes:
        answer: The answer text.
        citations: Supporting citations.
        confidence: Confidence in the answer.
        fallback_used: Whether fallback was used.
        execution_time_ms: Execution time.
    """

    answer: str
    citations: list[Citation]
    confidence: float = 0.0
    fallback_used: bool = False
    execution_time_ms: int = 0
