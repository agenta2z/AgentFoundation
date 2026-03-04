"""LLM-based knowledge consolidation after retrieval.

Provides the KnowledgeConsolidator class that deduplicates retrieved knowledge
and detects conflicts (e.g., contradictory instructions) before the knowledge
is injected into the agent prompt.
"""

import logging
from typing import Callable, Dict

from agent_foundation.knowledge.retrieval.models.enums import ConsolidationMode
from agent_foundation.knowledge.prompt_templates import render_prompt
from agent_foundation.knowledge.retrieval.utils import count_tokens

logger = logging.getLogger(__name__)


class KnowledgeConsolidator:
    """Consolidates formatted knowledge by deduplication and conflict detection.

    Takes Dict[str, str] from KnowledgeProvider, combines content values
    (excluding user_profile), sends to LLM for dedup/conflict analysis,
    and ADDS a ``consolidated_knowledge`` key to the dict.  Original keys
    are preserved for backward compatibility with templates that have not
    been updated to use ``consolidated_knowledge``.

    Attributes:
        llm_fn: Callable[[str], str] for LLM inference.
        mode: ConsolidationMode controlling when consolidation runs.
        short_knowledge_threshold: Token count (via ``count_tokens``) below
            which consolidation is skipped in DISABLED_FOR_SHORT_KNOWLEDGE mode.
    """

    CONSOLIDATED_KEY = "consolidated_knowledge"

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        mode: ConsolidationMode = ConsolidationMode.DISABLED,
        short_knowledge_threshold: int = 200,
    ):
        self.llm_fn = llm_fn
        self.mode = mode
        self.short_knowledge_threshold = short_knowledge_threshold

    def consolidate(
        self, query: str, formatted_groups: Dict[str, str]
    ) -> Dict[str, str]:
        """Consolidate formatted knowledge for the given query.

        Returns a **new** dict with all original keys preserved plus a
        ``consolidated_knowledge`` key when consolidation runs.
        Returns the input dict unchanged when consolidation is skipped.

        Skip conditions:
        - DISABLED mode
        - No content knowledge (all values empty after excluding user_profile)
        - DISABLED_FOR_SHORT_KNOWLEDGE and total tokens < threshold
        - LLM failure (graceful fallback)
        """
        if self.mode == ConsolidationMode.DISABLED:
            return formatted_groups

        # Combine content knowledge values (exclude user_profile to avoid
        # duplication — user_profile is rendered separately in <UserProfile>
        # by the HBS templates)
        content_values = [
            v for k, v in formatted_groups.items()
            if k != "user_profile" and v and v.strip()
        ]
        all_knowledge = "\n\n".join(content_values)

        if not all_knowledge.strip():
            return formatted_groups

        # Check threshold (token-based, consistent with BudgetAwareKnowledgeProvider)
        if (
            self.mode == ConsolidationMode.DISABLED_FOR_SHORT_KNOWLEDGE
            and count_tokens(all_knowledge) < self.short_knowledge_threshold
        ):
            logger.debug(
                "Skipping consolidation: %d tokens < threshold %d",
                count_tokens(all_knowledge),
                self.short_knowledge_threshold,
            )
            return formatted_groups

        # Run LLM consolidation
        prompt = render_prompt(
            "retrieval/KnowledgeConsolidation",
            query=query,
            retrieved_knowledge=all_knowledge,
        )

        try:
            result = self.llm_fn(prompt)
            consolidated = result.strip()
            if not consolidated:
                logger.debug("LLM returned empty consolidation")
                return formatted_groups
        except Exception:
            logger.warning("Knowledge consolidation failed", exc_info=True)
            return formatted_groups

        # Add consolidated key alongside original keys
        output = dict(formatted_groups)
        output[self.CONSOLIDATED_KEY] = consolidated
        return output
