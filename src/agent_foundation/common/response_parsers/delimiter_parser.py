# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

"""Simple delimiter-based response parser.

Extracts content between configurable delimiter tags (e.g. ``<Response>``).
Designed for use with DualInferencer's ``response_parser`` parameter.
"""

import re


def extract_delimited(
    raw: str,
    open_tag: str = "<Response>",
    close_tag: str = "</Response>",
) -> str:
    """Extract content between delimiter tags.

    Searches for content enclosed by ``open_tag`` and ``close_tag``.
    If multiple occurrences exist, returns the content from the **last**
    match (to handle cases where the agent outputs multiple attempts).

    If no tags are found, returns the full raw string unchanged (passthrough).

    Args:
        raw: The full raw output string from an inferencer.
        open_tag: Opening delimiter tag. Defaults to ``<Response>``.
        close_tag: Closing delimiter tag. Defaults to ``</Response>``.

    Returns:
        The extracted content (stripped of leading/trailing whitespace),
        or the original string if no delimiters are found.
    """
    pattern = re.escape(open_tag) + r"([\s\S]*?)" + re.escape(close_tag)
    matches = list(re.finditer(pattern, raw))
    if matches:
        return matches[-1].group(1).strip()
    return raw
