"""
Structuring prompt for LLM-based knowledge classification.

This module provides the prompt template used to guide LLMs in extracting
and classifying knowledge pieces from unstructured text. The prompt includes
the domain taxonomy and guides the LLM to assign:
1. Retrieval classification: domain, secondary_domains, tags, custom_tags
2. Injection classification: info_type, knowledge_type

The prompt produces structured JSON output compatible with KnowledgeDataLoader,
including metadata, pieces, and graph sections.
"""

from agent_foundation.knowledge.ingestion.taxonomy import format_taxonomy_for_prompt


# Full structuring prompt that produces KnowledgeDataLoader-compatible output
STRUCTURING_PROMPT_TEMPLATE = """You are a knowledge structuring assistant for a co-science ML framework.
Extract and organize knowledge into structured JSON with three sections: metadata, pieces, and graph.

## Domain Taxonomy (for classification)

{domain_taxonomy}

## Classification Guide

### Retrieval Classification (for FINDING this knowledge later)

**Primary Domain** — Select the most specific domain that fits the content.

**Secondary Domains** — 0-2 additional domains if the content spans multiple areas.

**Tags** — Select 1-5 specific topics from the domain's tag list.
If a relevant topic is not in the taxonomy, add it to "custom_tags" instead.

### Injection Classification (for ORGANIZING in agent prompts)

**Info Type** — Where should this go in the agent's prompt?
- "skills": Multi-step systematic techniques the agent can perform
- "instructions": Individual rules/directives the agent should follow
- "context": Background knowledge the agent should reason over
- "user_profile": Information about the user (use sparingly, usually set manually)
- "supplementary": Supporting material for deeper understanding

**Knowledge Type** — What kind of content is this?
- "fact": A factual statement
- "instruction": A single rule or directive
- "procedure": A multi-step process or technique
- "preference": A preference or opinion
- "note": General observation
- "episodic": Record of a past event
- "example": Worked example or case study

## Output Schema

Return a JSON object with three sections: metadata, pieces, and graph.

```json
{{
  "metadata": {{
    "<entity_id>": {{
      "entity_type": "<type>",
      "properties": {{ "<key>": "<value>" }}
    }}
  }},
  "pieces": [
    {{
      "piece_id": "<unique-kebab-case-id>",
      "content": "<the knowledge content>",
      "domain": "<primary domain from taxonomy>",
      "secondary_domains": ["<optional 0-2 domains>"],
      "tags": ["<from domain's tag list>"],
      "custom_tags": ["<any relevant tags NOT in taxonomy>"],
      "info_type": "<skills|instructions|context|user_profile|supplementary>",
      "knowledge_type": "<fact|instruction|procedure|preference|note|episodic|example>",
      "entity_id": "<owner entity_id or null for global>",
      "embedding_text": "<search-optimized keywords and synonyms>"
    }}
  ],
  "graph": {{
    "nodes": [
      {{ "node_id": "<type:name>", "node_type": "<type>", "label": "<display name>", "properties": {{}} }}
    ],
    "edges": [
      {{ "source_id": "<node_id>", "target_id": "<node_id>", "edge_type": "<RELATIONSHIP_TYPE>", "properties": {{}} }}
    ]
  }}
}}
```

## Guidelines

1. **Be Specific**: Choose the most specific domain, not "general" unless nothing else matches.
2. **Cross-Domain**: Use secondary_domains when knowledge genuinely spans multiple areas.
3. **Tags**: Prefer tags from the taxonomy; use custom_tags only for concepts not listed.
4. **Procedures vs Instructions**: A "procedure" has multiple steps; an "instruction" is a single rule.
5. **Skills**: Only use info_type="skills" for actionable multi-step techniques the agent can perform.
6. **Chunking**: Split long content into logically coherent pieces (1-3 paragraphs each).
7. **IDs**: Use descriptive kebab-case IDs like "flash-attention-optimization-steps".
8. **Embedding Text**: Include search keywords, synonyms, and related terms for better retrieval.
9. **Graph**: Create nodes for key entities/concepts and edges for their relationships.
10. **Metadata**: Use for structured properties of entities (e.g., user profile, tool configurations).
11. **No Secrets**: Do NOT include passwords, API keys, tokens, or credentials.

## Context (if provided)

{context}

## User Input

{user_input}

Respond with ONLY the JSON, no explanation.
"""


# Simpler prompt for pieces-only output (when full schema not needed)
PIECES_ONLY_PROMPT_TEMPLATE = """You are a knowledge structuring assistant for a co-science framework.
Extract and organize knowledge into structured JSON.

## Step 1: Retrieval Classification (for FINDING this knowledge later)

**Primary Domain** — What is this primarily about?
{domain_taxonomy}

**Secondary Domains** — Does this also relate to other domains? (optional, 0-2)

**Tags** — Select 1-5 specific topics from the domain's tag list.
If a relevant topic is not in the list, add it to "custom_tags" instead.

## Step 2: Injection Classification (for ORGANIZING in agent prompts)

**Info Type** — Where should this go in the agent's prompt?
- "skills": Multi-step systematic techniques the agent can perform
- "instructions": Individual rules/directives the agent should follow
- "context": Background knowledge the agent should reason over
- "user_profile": Information about the user (use sparingly, usually set manually)
- "supplementary": Supporting material for deeper understanding

**Knowledge Type** — What kind of content is this?
- "fact": A factual statement
- "instruction": A single rule or directive
- "procedure": A multi-step process or technique
- "preference": A preference or opinion
- "note": General observation
- "episodic": Record of a past event
- "example": Worked example or case study

## Output Schema

Return a JSON object with a "pieces" array. Each piece should have:

```json
{{
  "pieces": [
    {{
      "piece_id": "<unique-kebab-case-id>",
      "content": "<the knowledge>",
      "domain": "<primary domain from taxonomy>",
      "secondary_domains": ["<optional>", "<0-2 domains>"],
      "tags": ["<from domain's tag list>"],
      "custom_tags": ["<any relevant tags NOT in taxonomy>"],
      "info_type": "<skills|instructions|context|user_profile|supplementary>",
      "knowledge_type": "<fact|instruction|procedure|preference|note|episodic|example>"
    }}
  ]
}}
```

## Guidelines

1. **Be Specific**: Choose the most specific domain that fits, not "general" unless nothing else matches.
2. **Cross-Domain**: Use secondary_domains when knowledge genuinely spans multiple areas.
3. **Tags**: Prefer tags from the taxonomy; use custom_tags only for concepts not listed.
4. **Procedures vs Instructions**: A "procedure" has multiple steps; an "instruction" is a single rule.
5. **Skills**: Only use info_type="skills" for actionable multi-step techniques.
6. **Chunking**: Split long content into logically coherent pieces (1-3 paragraphs each).
7. **IDs**: Use descriptive kebab-case IDs like "flash-attention-optimization-steps".

## User Input

{user_input}
"""


def get_structuring_prompt(
    user_input: str,
    context: str = "",
    full_schema: bool = True,
) -> str:
    """Generate the full structuring prompt with taxonomy and user input.

    Args:
        user_input: The raw text to be structured into knowledge pieces.
        context: Optional context string (e.g., header hierarchy from chunker).
        full_schema: If True, use full schema with metadata/graph sections.
            If False, use pieces-only schema.

    Returns:
        The complete prompt string ready for LLM inference.
    """
    domain_taxonomy = format_taxonomy_for_prompt()

    if full_schema:
        return STRUCTURING_PROMPT_TEMPLATE.format(
            domain_taxonomy=domain_taxonomy,
            context=context if context else "(No additional context)",
            user_input=user_input,
        )
    else:
        return PIECES_ONLY_PROMPT_TEMPLATE.format(
            domain_taxonomy=domain_taxonomy,
            user_input=user_input,
        )


# Prompt for re-classifying existing pieces that lack domain information
CLASSIFICATION_PROMPT_TEMPLATE = """You are a knowledge classification assistant.
Given a knowledge piece, determine its domain classification.

## Domain Taxonomy
{domain_taxonomy}

## Current Piece
Content: {content}
Existing tags: {tags}
Existing info_type: {info_type}
Existing knowledge_type: {knowledge_type}

## Task
Classify this piece into the taxonomy. Return JSON:

```json
{{
  "domain": "<primary domain>",
  "secondary_domains": ["<optional>"],
  "tags": ["<from domain's tag list>"],
  "custom_tags": ["<any relevant tags NOT in taxonomy>"]
}}
```

Only output the JSON, no explanation.
"""


def get_classification_prompt(
    content: str,
    tags: list[str],
    info_type: str,
    knowledge_type: str,
) -> str:
    """Generate a prompt for classifying an existing knowledge piece.

    Used for migrating legacy pieces that don't have domain classification.

    Args:
        content: The piece content.
        tags: Existing tags (may be empty).
        info_type: Existing info_type value.
        knowledge_type: Existing knowledge_type value.

    Returns:
        The complete prompt string ready for LLM inference.
    """
    domain_taxonomy = format_taxonomy_for_prompt()
    return CLASSIFICATION_PROMPT_TEMPLATE.format(
        domain_taxonomy=domain_taxonomy,
        content=content,
        tags=", ".join(tags) if tags else "(none)",
        info_type=info_type,
        knowledge_type=knowledge_type,
    )
