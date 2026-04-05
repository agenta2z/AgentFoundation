# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

"""Common utilities for Claude Code inferencers.

Claude Code CLI expects dash-separated version numbers in model tags
(e.g. ``claude-opus-4-6``), while other systems like Devmate use
dot-separated versions (e.g. ``claude-opus-4.6``) and the Anthropic
API uses full date-qualified identifiers (e.g. ``claude-opus-4-6-20260204``
from ``ClaudeModels``).

This module provides ``resolve_model_tag`` to normalize any model tag
string into the dash-separated format expected by Claude Code.
"""

import re

# Explicit mapping for Anthropic API / ClaudeModels values whose short
# alias cannot be derived by simple date-stripping + regex conversion.
# Covers legacy 3.x naming where family and version are ordered differently.
_KNOWN_ALIASES: dict[str, str] = {
    # Legacy 3.x (API: claude-{major}-{family}-{date})
    "claude-3-opus-20240229": "claude-3-opus",
    "claude-3-haiku-20240307": "claude-3-haiku",
    # Legacy 3.5 / 3.7 (API: claude-{major}-{minor}-{family}-{date})
    "claude-3-5-sonnet-20241022": "claude-3-5-sonnet",
    "claude-3-7-sonnet-20250219": "claude-3-7-sonnet",
    # Devmate plugboard names (dot → dash)
    "claude3.5-sonnet": "claude-3-5-sonnet",
    "claude3.7-sonnet": "claude-3-7-sonnet",
    "claude3.5-haiku": "claude-3-5-haiku",
    "claude3-haiku": "claude-3-haiku",
    "claude4-sonnet": "claude-sonnet-4",
}

# Pattern: trailing -YYYYMMDD (8-digit date suffix)
_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")

# Pattern: digit.digit NOT followed by 3+ more digits (version separator)
_DOT_VERSION_RE = re.compile(r"(\d)\.(\d)(?!\d{3,})")


def resolve_model_tag(model_tag: str) -> str:
    """Normalize a model tag for Claude Code CLI / SDK.

    Handles three input formats:

    1. **Anthropic API / ClaudeModels** (full date-qualified)::

        claude-opus-4-6-20260204   → claude-opus-4-6
        claude-sonnet-4-5-20250929 → claude-sonnet-4-5
        claude-3-5-sonnet-20241022 → claude-3-5-sonnet

    2. **Devmate / dot-separated versions**::

        claude-opus-4.6            → claude-opus-4-6
        claude-sonnet-4.5          → claude-sonnet-4-5
        claude3.5-sonnet           → claude-3-5-sonnet

    3. **Already-correct Claude Code format** (no-op)::

        claude-opus-4-6            → claude-opus-4-6
        sonnet                     → sonnet

    Args:
        model_tag: Model tag string in any format.

    Returns:
        Model tag normalized for Claude Code CLI (dash-separated versions,
        no date suffix).
    """
    # 1. Check explicit alias table first
    if model_tag in _KNOWN_ALIASES:
        return _KNOWN_ALIASES[model_tag]

    # 2. Strip trailing date suffix (-YYYYMMDD)
    result = _DATE_SUFFIX_RE.sub("", model_tag)

    # 3. Convert dot-separated version digits to dashes
    result = _DOT_VERSION_RE.sub(r"\1-\2", result)

    return result
