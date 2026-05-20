"""Update prompt for LLM-based knowledge updates.

Design Decision (v1.1): The LLM determines the ACTION only, not the content.
Content merging is done programmatically to avoid data loss from truncation.
"""

UPDATE_INTENT_PROMPT = """You are a knowledge update assistant.

The user wants to update existing knowledge. Analyze the request and determine the best action.

## Existing Knowledge (Current)
ID: {existing_id}
Content (first 2000 chars): {existing_content}
Full length: {existing_length} characters
Domain: {existing_domain}
Tags: {existing_tags}
Last Updated: {existing_updated_at}

## Update Request (New)
Content (first 2000 chars): {new_content}
Full length: {new_length} characters
{update_instruction}

## Update Modes

**REPLACE** — Completely replace existing content with new content when:
- New content is a corrected/updated version of the same topic
- Existing content is outdated or incorrect
- User explicitly says "replace" or "overwrite"

**MERGE** — Combine existing and new content when:
- New content adds information to the same topic
- Both have valuable details that should be preserved
- User says "add to", "append", or "merge"

**NO_CHANGE** — Do not update when:
- New content doesn't relate to existing
- New content is older/less accurate than existing
- Update would cause information loss

## Your Analysis
Return JSON:
{{
    "action": "replace|merge|no_change",
    "confidence": 0.0-1.0,
    "reasoning": "Why this action was chosen",
    "merge_strategy": "prepend|append|interleave" (only for MERGE action),
    "updated_domain": "domain_name" or null to keep existing,
    "updated_tags": ["list", "of", "tags"] or null to keep existing,
    "clear_tags": true if tags should be cleared (empty list intended),
    "changes_summary": "Brief summary of what will change"
}}

IMPORTANT: Do NOT return updated_content. The actual content merging is done
programmatically to preserve full content without truncation.
"""

UPDATE_PROMPT_CONFIG = {
    "temperature": 0.0,
    "max_tokens": 500,
}
