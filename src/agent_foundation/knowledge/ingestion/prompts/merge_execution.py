"""Merge execution prompt for combining two knowledge pieces."""

MERGE_EXECUTION_PROMPT = """You are a knowledge merger. Combine two related pieces into one coherent piece.

## Piece A (Existing)
Content: {piece_a_content}
Domain: {piece_a_domain}
Tags: {piece_a_tags}

## Piece B (New)
Content: {piece_b_content}
Domain: {piece_b_domain}
Tags: {piece_b_tags}

## Merge Guidelines
1. Preserve ALL unique information from both pieces
2. Eliminate redundancy without losing detail
3. Maintain consistent tone and style
4. Use the more specific domain if they differ
5. Union the tags from both pieces
6. If pieces contradict, include BOTH viewpoints with clear attribution

## Your Merged Result
Return JSON:
{{
  "merged_content": "The combined content preserving all unique information",
  "merged_domain": "The most appropriate domain",
  "merged_tags": ["union", "of", "all", "tags"],
  "merge_notes": "Brief explanation of how content was combined"
}}
"""

MERGE_EXECUTION_CONFIG = {
    "temperature": 0.3,
    "max_tokens": 1000,
}
