"""LLM Judge prompt for three-tier deduplication."""

DEDUP_LLM_JUDGE_PROMPT = """You are a knowledge deduplication judge.

Two pieces have been flagged as potentially duplicate (embedding similarity: {similarity:.3f}).
Determine the relationship and recommended action.

## Existing Piece (Already in KB)
Content: {existing_content}
Domain: {existing_domain}
Tags: {existing_tags}
Created: {existing_created_at}

## New Piece (Being Ingested)
Content: {new_content}
Domain: {new_domain}
Tags: {new_tags}

## Decision Criteria

**ADD** — Keep both pieces if:
- They cover genuinely different subtopics despite surface similarity
- Each provides unique value that would be lost if merged

**UPDATE** — Replace existing with new if:
- New piece corrects errors in existing
- New piece is explicitly marked as an update
- New piece has more recent, accurate information

**MERGE** — Combine into single piece if:
- Pieces contain complementary information about the same topic
- Neither is strictly better; both have valuable details

**NO_OP** — Discard new piece if:
- Exact or near-exact duplicate
- New piece is less specific than existing
- New piece adds no value

## Handling Contradictions
If pieces contradict (e.g., A says X, B says NOT X):
- If new piece has explicit update/correction language → UPDATE
- If unclear which is correct → ADD (keep both, flag for review)

## Your Judgment
Return JSON:
{{
  "action": "ADD|UPDATE|MERGE|NO_OP",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why this action was chosen",
  "contradiction_detected": true|false,
  "contradiction_note": "If contradiction detected, explain. Otherwise null."
}}
"""

DEDUP_JUDGE_CONFIG = {
    "temperature": 0.0,
    "max_tokens": 200,
}
