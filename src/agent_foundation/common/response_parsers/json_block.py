# pyre-strict

"""Extract a labeled ```json <label> ... ``` fenced block and decode it.

Promoted from ``common/inferencers/flow_parsers.py`` so it can be shared by
``common/response_parsers`` consumers (e.g. ``AgenticOutput.json``) without a
dependency on the flow-parser module; ``flow_parsers`` re-binds the old private
names to these functions for its existing call sites.

:func:`find_json_block_text` returns the raw fenced body so a caller can apply
its own (e.g. hardened) JSON decode; :func:`extract_json_block` is the
behavior-preserving plain decode (dict or ``None``) the flow parsers rely on.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Matches ```json <label> ... ``` (label may be followed by extra text on
# the fence line; we capture the body between the fences).
JSON_FENCE_TEMPLATE = r"```json\s+{label}\b[^\n]*\n([\s\S]*?)\n\s*```"


def find_json_block_text(s: str, label: str) -> str | None:
    """Return the raw text inside a labeled ```json <label> fence, or ``None``."""
    pattern = re.compile(JSON_FENCE_TEMPLATE.format(label=re.escape(label)))
    m = pattern.search(s)
    return m.group(1) if m else None


def extract_json_block(s: str, label: str) -> dict[str, Any] | None:
    """Find ```json <label> ... ``` and parse the body as JSON.

    Returns the decoded dict, or ``None`` if the block is absent or invalid.
    """
    raw = find_json_block_text(s, label)
    if raw is None:
        return None
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None
