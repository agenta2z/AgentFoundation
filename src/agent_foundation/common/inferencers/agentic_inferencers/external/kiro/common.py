# pyre-strict

"""Common utilities for Kiro CLI inferencer.

Kiro CLI uses dot-separated version numbers in model tags
(e.g. ``claude-opus-4.6``), matching the Devmate convention.
Claude Code CLI uses dash-separated versions (e.g. ``claude-opus-4-6``),
and the Anthropic API uses full date-qualified identifiers
(e.g. ``claude-opus-4-6-20260204``).

This module provides ``resolve_model_tag`` to normalize any model tag
string into the dot-separated format expected by Kiro CLI.

Available Kiro CLI models (as of v1.26):
    auto, claude-opus-4.6, claude-sonnet-4.6, claude-opus-4.5,
    claude-sonnet-4.5, claude-sonnet-4, claude-haiku-4.5,
    deepseek-3.2, minimax-m2.5, minimax-m2.1, glm-5, qwen3-coder-next
"""

import re

# Explicit mapping for model tags that cannot be derived by regex.
# Covers legacy 3.x naming and short aliases from other systems.
_KNOWN_ALIASES: dict[str, str] = {
    # Legacy 3.x Anthropic API names
    "claude-3-opus-20240229": "claude-3-opus",
    "claude-3-haiku-20240307": "claude-3-haiku",
    "claude-3-5-sonnet-20241022": "claude-3.5-sonnet",
    "claude-3-7-sonnet-20250219": "claude-3.7-sonnet",
    # Claude Code CLI short aliases (dash-separated → dot-separated)
    "claude-3-5-sonnet": "claude-3.5-sonnet",
    "claude-3-7-sonnet": "claude-3.7-sonnet",
    "claude-3-5-haiku": "claude-3.5-haiku",
    # Devmate plugboard names (already dot-separated, pass through)
    "claude3.5-sonnet": "claude-3.5-sonnet",
    "claude3.7-sonnet": "claude-3.7-sonnet",
    "claude3-haiku": "claude-3-haiku",
    "claude4-sonnet": "claude-sonnet-4",
    # Common short aliases
    "sonnet": "claude-sonnet-4.6",
    "opus": "claude-opus-4.6",
    "haiku": "claude-haiku-4.5",
}

# Pattern: trailing -YYYYMMDD (8-digit date suffix)
_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")

# Pattern: digit-digit NOT followed by 3+ more digits (version separator)
# Converts dash-separated versions to dot-separated: 4-6 → 4.6
_DASH_VERSION_RE = re.compile(r"(\d)-(\d)(?!\d{3,})")


def resolve_model_tag(model_tag: str) -> str:
    """Normalize a model tag for Kiro CLI.

    Kiro CLI expects dot-separated version numbers (e.g. ``claude-opus-4.6``).
    This function handles input from any format used in the framework:

    1. **Anthropic API / ClaudeModels** (full date-qualified)::

        claude-opus-4-6-20260204   → claude-opus-4.6
        claude-sonnet-4-5-20250929 → claude-sonnet-4.5

    2. **Claude Code CLI / dash-separated versions**::

        claude-opus-4-6            → claude-opus-4.6
        claude-sonnet-4-5          → claude-sonnet-4.5

    3. **Short aliases**::

        sonnet                     → claude-sonnet-4.6
        opus                       → claude-opus-4.6
        haiku                      → claude-haiku-4.5

    4. **Already-correct Kiro CLI format** (no-op)::

        claude-opus-4.6            → claude-opus-4.6
        auto                       → auto
        deepseek-3.2               → deepseek-3.2

    Args:
        model_tag: Model tag string in any format.

    Returns:
        Model tag normalized for Kiro CLI (dot-separated versions,
        no date suffix).
    """
    # 1. Check explicit alias table first
    if model_tag in _KNOWN_ALIASES:
        return _KNOWN_ALIASES[model_tag]

    # 2. Strip trailing date suffix (-YYYYMMDD)
    result = _DATE_SUFFIX_RE.sub("", model_tag)

    # 3. Check alias table again after stripping date
    if result != model_tag and result in _KNOWN_ALIASES:
        return _KNOWN_ALIASES[result]

    # 4. Convert dash-separated version digits to dots
    result = _DASH_VERSION_RE.sub(r"\1.\2", result)

    return result
