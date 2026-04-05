# pyre-strict
"""MetaMate Integration — inferencers, adapters, clients, and types.

This package provides:
- MetamateSDKInferencer: Async, polling-based SDK using MetamateGraphQLClient
- MetamateCliInferencer: Subprocess-based CLI execution via query_metamate
- Adapters: Deep research, knowledge discovery, platform Q&A, debug assistance
- Clients: Production, mock, and fallback MetaMate clients
"""

from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.metamate_cli_inferencer import (  # noqa: F401
    MetamateCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.metamate_sdk_inferencer import (  # noqa: F401
    MetamateSDKInferencer,
)

# Adapters
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.adapters import (  # noqa: F401
    DebugAssistantAdapter,
    DeepResearchAdapter,
    KnowledgeDiscoveryAdapter,
    PlatformQAAdapter,
)

# Clients
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.clients import (  # noqa: F401
    FallbackClient,
    MetamateClient,
    MetamateClientInterface,
    MockMetamateClient,
)

# Exceptions
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.exceptions import (  # noqa: F401
    KnowledgeSearchError,
    MetamateConnectionError,
    MetamateError,
    MetamateSecurityError,
    MetamateTimeoutError,
    MetamateToolError,
    MetamateUnavailableError,
    MetamateValidationError,
    ResearchError,
)

# Types
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.types import (  # noqa: F401
    AggregatedKnowledgeResults,
    Citation,
    Finding,
    KnowledgeQuery,
    KnowledgeResult,
    KnowledgeSource,
    MetamateConfig,
    QueryType,
    ResearchEvent,
    ResearchEventType,
    ResearchPlan,
    ResearchQuery,
    ResearchResults,
    ToolResponse,
)

__all__ = [
    # Inferencers
    "MetamateSDKInferencer",
    "MetamateCliInferencer",
    # Adapters
    "DeepResearchAdapter",
    "KnowledgeDiscoveryAdapter",
    "PlatformQAAdapter",
    "DebugAssistantAdapter",
    # Clients
    "MetamateClient",
    "MetamateClientInterface",
    "MockMetamateClient",
    "FallbackClient",
    # Exceptions
    "MetamateError",
    "MetamateConnectionError",
    "MetamateTimeoutError",
    "MetamateToolError",
    "MetamateSecurityError",
    "MetamateUnavailableError",
    "MetamateValidationError",
    "ResearchError",
    "KnowledgeSearchError",
    # Types
    "ResearchQuery",
    "ResearchPlan",
    "ResearchResults",
    "ResearchEvent",
    "ResearchEventType",
    "QueryType",
    "Finding",
    "Citation",
    "KnowledgeSource",
    "KnowledgeQuery",
    "KnowledgeResult",
    "AggregatedKnowledgeResults",
    "MetamateConfig",
    "ToolResponse",
]
