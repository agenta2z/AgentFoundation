"""Prompts module for knowledge ingestion.

This module contains prompt templates for LLM-based knowledge structuring,
deduplication, merge, validation, and skill synthesis.
"""

from agent_foundation.knowledge.ingestion.prompts.structuring_prompt import (
    get_structuring_prompt,
    STRUCTURING_PROMPT_TEMPLATE,
)
from agent_foundation.knowledge.ingestion.prompts.dedup_llm_judge import (
    DEDUP_LLM_JUDGE_PROMPT,
    DEDUP_JUDGE_CONFIG,
)
from agent_foundation.knowledge.ingestion.prompts.merge_candidate import (
    MERGE_CANDIDATE_PROMPT,
    MERGE_CANDIDATE_CONFIG,
)
from agent_foundation.knowledge.ingestion.prompts.merge_execution import (
    MERGE_EXECUTION_PROMPT,
    MERGE_EXECUTION_CONFIG,
)
from agent_foundation.knowledge.ingestion.prompts.validation import (
    VALIDATION_PROMPT,
    VALIDATION_CONFIG,
)
from agent_foundation.knowledge.ingestion.prompts.skill_synthesis import (
    SKILL_SYNTHESIS_PROMPT,
    SKILL_SYNTHESIS_CONFIG,
)
from agent_foundation.knowledge.ingestion.prompts.update_prompt import (
    UPDATE_INTENT_PROMPT,
    UPDATE_PROMPT_CONFIG,
)

__all__ = [
    # Structuring
    "get_structuring_prompt",
    "STRUCTURING_PROMPT_TEMPLATE",
    # Dedup
    "DEDUP_LLM_JUDGE_PROMPT",
    "DEDUP_JUDGE_CONFIG",
    # Merge
    "MERGE_CANDIDATE_PROMPT",
    "MERGE_CANDIDATE_CONFIG",
    "MERGE_EXECUTION_PROMPT",
    "MERGE_EXECUTION_CONFIG",
    # Validation
    "VALIDATION_PROMPT",
    "VALIDATION_CONFIG",
    # Skill Synthesis
    "SKILL_SYNTHESIS_PROMPT",
    "SKILL_SYNTHESIS_CONFIG",
    # Update
    "UPDATE_INTENT_PROMPT",
    "UPDATE_PROMPT_CONFIG",
]
