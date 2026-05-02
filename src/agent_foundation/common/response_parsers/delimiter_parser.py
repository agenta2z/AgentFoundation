
# pyre-strict

"""Simple delimiter-based response parser.

Extracts content between configurable delimiter tags (e.g. ``<Response>``).
Designed for use with DualInferencer's ``response_parser`` parameter.
"""

import re
from typing import Optional


_PROMPT_ECHO_MARKERS = ("**IMPORTANT", "wrap your response")


def extract_delimited(
    raw: str,
    open_tag: str = "<Response>",
    close_tag: str = "</Response>",
) -> Optional[str]:
    """Extract content between delimiter tags.

    Searches for content enclosed by ``open_tag`` and ``close_tag``.
    Iterates matches in reverse order ("most-recent-wins" semantic) and
    returns the first match whose content is NOT a prompt-template echo.

    Three return modes:

    1. **No matches at all** -- returns the raw string unchanged (passthrough).
    2. **One or more clean matches** -- returns the most-recent clean match's
       content (stripped).
    3. **Matches exist but ALL are prompt-template echoes** -- returns ``None``.

    Args:
        raw: The full raw output string from an inferencer.
        open_tag: Opening delimiter tag. Defaults to ``<Response>``.
        close_tag: Closing delimiter tag. Defaults to ``</Response>``.

    Returns:
        The extracted content (stripped of leading/trailing whitespace), the
        raw string (passthrough -- no matches), or ``None`` (all matches were
        prompt-echo -- hard failure).
    """
    pattern = re.escape(open_tag) + r"([\s\S]*?)" + re.escape(close_tag)
    matches = list(re.finditer(pattern, raw))
    if not matches:
        return raw
    for match in reversed(matches):
        content = match.group(1).strip()
        if all(marker in content for marker in _PROMPT_ECHO_MARKERS):
            continue
        return content
    return None
