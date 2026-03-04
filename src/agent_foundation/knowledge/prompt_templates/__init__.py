"""Knowledge prompt templates backed by TemplateManager.

All LLM prompts for knowledge ingestion, quality management, and retrieval
are stored as .hbs (Handlebars) template files in this package's subdirectories
and rendered at runtime via TemplateManager from RichPythonUtils.

Public API:
    KNOWLEDGE_TEMPLATE_MANAGER  - The TemplateManager instance.
    render_prompt(key, **kw)    - Render a template by key with keyword args.
    get_structuring_prompt(...)  - Helper that injects taxonomy + chooses variant.
    get_classification_prompt(...)  - Helper that injects taxonomy for classification.
"""

from pathlib import Path
from typing import List, Optional

from rich_python_utils.string_utils.formatting.template_manager import TemplateManager
from rich_python_utils.string_utils.formatting.handlebars_format import (
    format_template as handlebars_template_format,
)

_TEMPLATES_DIR = str(Path(__file__).resolve().parent)

KNOWLEDGE_TEMPLATE_MANAGER = TemplateManager(
    templates=_TEMPLATES_DIR,
    template_formatter=handlebars_template_format,
    active_template_type=None,
)


def render_prompt(template_key: str, template_version: str = None, **kwargs) -> str:
    """Render a knowledge prompt template by key.

    Args:
        template_key: Slash-separated key, e.g. "quality/DedupJudge".
        template_version: Optional version suffix, e.g. "PiecesOnly", "Cli".
        **kwargs: Template variables passed to handlebars_template_format.

    Returns:
        The rendered prompt string.
    """
    mgr = KNOWLEDGE_TEMPLATE_MANAGER
    if template_version:
        mgr = mgr.switch(template_version=template_version)
    return mgr(template_key, **kwargs)


def get_structuring_prompt(
    user_input: str,
    context: str = "",
    full_schema: bool = True,
) -> str:
    """Generate the full structuring prompt with taxonomy and user input.

    Loads domain taxonomy dynamically via format_taxonomy_for_prompt() and
    selects the appropriate template variant (full schema vs pieces-only)
    via the TemplateManager version mechanism.
    """
    from agent_foundation.knowledge.ingestion.taxonomy import format_taxonomy_for_prompt

    domain_taxonomy = format_taxonomy_for_prompt()
    version = None if full_schema else "PiecesOnly"
    return render_prompt(
        "ingestion/Structuring",
        template_version=version,
        domain_taxonomy=domain_taxonomy,
        context=context if context else "(No additional context)",
        user_input=user_input,
    )


def get_classification_prompt(
    content: str,
    tags: List[str],
    info_type: str,
    knowledge_type: str,
) -> str:
    """Generate a prompt for classifying an existing knowledge piece."""
    from agent_foundation.knowledge.ingestion.taxonomy import format_taxonomy_for_prompt

    domain_taxonomy = format_taxonomy_for_prompt()
    return render_prompt(
        "ingestion/Classification",
        domain_taxonomy=domain_taxonomy,
        content=content,
        tags=", ".join(tags) if tags else "(none)",
        info_type=info_type,
        knowledge_type=knowledge_type,
    )


__all__ = [
    "KNOWLEDGE_TEMPLATE_MANAGER",
    "render_prompt",
    "get_structuring_prompt",
    "get_classification_prompt",
]
