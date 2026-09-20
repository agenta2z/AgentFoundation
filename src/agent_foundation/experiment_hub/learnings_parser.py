# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Parser for tag-discriminated JSON code fences in markdown docs.

Generalizes the proposal_summary fence pattern so it can be reused for
``` ```json learnings_actions ``` ``` and any future tag-keyed fences.

Used to split a markdown doc into (markdown_body_without_fence,
parsed_actions_dict).

Ported faithfully from RankEvolve's ``agentic_foundation.common.ui.learnings_parser``;
provides its own ``split_fence`` (no dependency on the framework split_fence).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)


def make_fence_re(tag: str) -> re.Pattern[str]:
    """Build a regex matching a ``` ```json {tag} ``` ``` fenced block.

    The regex captures the JSON body. Multiline (re.DOTALL) so the body
    can span any number of lines. The tag is regex-escaped to allow
    discriminator strings with special characters (though in practice
    they are simple identifiers).
    """
    return re.compile(
        r"```json\s+" + re.escape(tag) + r"\s*\n(.*?)\n```",
        re.DOTALL,
    )


def split_fence(md: str, tag: str) -> tuple[str, dict[str, Any] | None]:
    """Split markdown into (body_without_fence, parsed_json_dict).

    If the fence is missing OR the JSON is invalid, returns
    (original_md, None). The caller decides whether absence is fatal.

    The returned body has the entire ``` ```json {tag} ... ``` ``` block
    removed (including surrounding blank lines), so the body can be
    rendered without the trailing JSON block leaking into the markdown.
    """
    pat = make_fence_re(tag)
    m = pat.search(md)
    if not m:
        return md, None
    try:
        actions = json.loads(m.group(1))
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Fence parse failed for tag=%s: %s", tag, e)
        return md, None
    # Cut the fence (and any trailing blank lines that immediately follow).
    start = m.start()
    end = m.end()
    while end < len(md) and md[end] in (" ", "\t", "\n"):
        end += 1
    body = (md[:start].rstrip() + "\n").lstrip("\n") if start > 0 else ""
    if end < len(md):
        # If anything follows the fence, append it after a blank line
        body = body + "\n" + md[end:]
    return body, actions


def extract_fence_json(md: str, tag: str) -> dict[str, Any] | None:
    """Convenience: just return the parsed JSON dict (no body split)."""
    _, actions = split_fence(md, tag)
    return actions
