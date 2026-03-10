"""
Backward-compatibility shim for ``agentic_retriever`` imports.

.. deprecated::
    The ``AgenticRetriever`` class has been removed.  Use
    ``RetrievalPipeline`` with ``AggregatingPostProcessor`` instead.

    ``SubQuery``, ``AgenticRetrievalResult``, ``create_domain_decomposer``,
    ``create_llm_decomposer``, ``QueryExpander``, and ``_LegacyQueryExpander``
    are now defined in ``retrieval_pipeline.py``.  This module re-exports
    them so that existing ``from agentic_retriever import …`` statements
    continue to work.
"""

# Re-export everything from the canonical location
from agent_foundation.knowledge.retrieval.retrieval_pipeline import (  # noqa: F401
    AgenticRetrievalResult,
    QueryExpander,
    SubQuery,
    create_domain_decomposer,
    create_llm_decomposer,
)

__all__ = [
    "SubQuery",
    "AgenticRetrievalResult",
    "QueryExpander",
    "create_domain_decomposer",
    "create_llm_decomposer",
]
