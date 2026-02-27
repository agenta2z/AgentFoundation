"""Skill synthesis prompt for detecting and creating skills from knowledge clusters."""

SKILL_SYNTHESIS_PROMPT = """You are a skill synthesizer.

A cluster of related knowledge pieces has been detected. Determine if they collectively
describe a coherent technique or skill that should be synthesized.

## Related Pieces (Cluster)
{pieces_formatted}

## Cluster Statistics
- Number of pieces: {num_pieces}
- Average similarity: {avg_similarity:.3f}
- Common tags: {common_tags}
- Domains represented: {domains}

## Skill Detection Criteria
A cluster is "skill-worthy" if:
1. Pieces describe steps or components of a coherent technique
2. Combined, they would form actionable instructions
3. At least 3 pieces contribute distinct steps/aspects
4. The technique has a clear name/purpose

## Your Analysis

If is_skill_worthy is false, set synthesized_skill to null:
{{
  "is_skill_worthy": false,
  "confidence": 0.3,
  "reasoning": "Pieces are related but don't form a coherent technique with actionable steps",
  "synthesized_skill": null
}}

If is_skill_worthy is true, populate synthesized_skill:
{{
  "is_skill_worthy": true,
  "confidence": 0.85,
  "reasoning": "Pieces describe a coherent 3-step technique for optimizing attention",
  "synthesized_skill": {{
    "name": "skill-name-in-kebab-case",
    "description": "One-line description of what this skill does",
    "steps": [
      {{"step": 1, "description": "First step", "source_piece_ids": ["piece_id_1"]}},
      {{"step": 2, "description": "Second step", "source_piece_ids": ["piece_id_2", "piece_id_3"]}}
    ],
    "prerequisites": ["any required knowledge or setup"],
    "source_piece_ids": ["all", "contributing", "piece_ids"]
  }}
}}

Respond based on actual analysis, not these examples.
"""

SKILL_SYNTHESIS_CONFIG = {
    "min_pieces_for_skill": 3,
    "min_avg_similarity": 0.75,
    "temperature": 0.3,
    "max_tokens": 800,
}
