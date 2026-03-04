"""Advisory LLM configuration for knowledge prompts.

These configs are NOT consumed by TemplateManager or render_prompt().
They document recommended temperature/max_tokens for callers that
construct their own inferencer wrappers.
"""

PROMPT_CONFIGS = {
    "ingestion/Structuring": {"temperature": 0.0, "max_tokens": 2000},
    "ingestion/Classification": {"temperature": 0.0, "max_tokens": 500},
    "quality/DedupJudge": {"temperature": 0.0, "max_tokens": 200},
    "quality/MergeCandidate": {"temperature": 0.0, "max_tokens": 500},
    "quality/MergeExecution": {"temperature": 0.3, "max_tokens": 1000},
    "quality/UpdateIntent": {"temperature": 0.0, "max_tokens": 500},
    "quality/UpdateContentGeneration": {"temperature": 0.2, "max_tokens": 16384},
    "quality/Validation": {"temperature": 0.0, "max_tokens": 500},
    "quality/SkillSynthesis": {"temperature": 0.3, "max_tokens": 800},
    "retrieval/QueryDecomposition": {"temperature": 0.0, "max_tokens": 500},
    "retrieval/KnowledgeConsolidation": {"temperature": 0.0, "max_tokens": 4096},
}

# Special non-LLM config for skill synthesis
SKILL_SYNTHESIS_THRESHOLDS = {
    "min_pieces_for_skill": 3,
    "min_avg_similarity": 0.75,
}
