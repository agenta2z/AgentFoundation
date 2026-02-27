"""Merge candidate detection prompt."""

MERGE_CANDIDATE_PROMPT = """You are analyzing potential merge candidates for a knowledge piece.

## New Piece
Content: {new_content}
Domain: {new_domain}
Tags: {new_tags}

## Candidate Pieces
{candidates_formatted}

## Analysis Required
For each candidate, determine:
1. merge_type: duplicate|superset|subset|overlapping|update|unrelated
2. similarity: 0.0-1.0 (semantic similarity)
3. reason: Why this merge type was assigned

Return JSON:
{{
  "candidates": [
    {{
      "piece_id": "...",
      "merge_type": "duplicate|superset|subset|overlapping|update|unrelated",
      "similarity": 0.0-1.0,
      "reason": "..."
    }}
  ]
}}
"""

MERGE_CANDIDATE_CONFIG = {
    "temperature": 0.0,
    "max_tokens": 500,
}
