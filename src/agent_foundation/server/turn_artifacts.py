"""Shared turn artifact writer — single source of truth for turn folder format.

Both SessionStore.save_turn_data (OpenStartup) and SOPSession.save_turn_data
delegate to this to avoid format drift.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


def save_turn_artifacts(
    turn_dir: Path,
    *,
    prompt_data: dict[str, Any] | None = None,
    user_input: str = "",
    response: str = "",
    source: Literal["human", "synthetic", "system"] = "human",
    phase_id: str = "",
    instance_id: str = "",
    caller_id: str = "",
    **extra: Any,
) -> None:
    """Write standard turn artifacts to a turn directory.

    Creates the directory if it doesn't exist. Writes:
    - metadata.json (source, phase_id, instance_id, caller_id, extras)
    - user_input.txt (the user's input or synthetic response)
    - inference_response.txt (the LLM's response)
    - template_feed.json (if prompt_data provided)
    - rendered_prompt.txt (if prompt_data has 'rendered_prompt')
    """
    turn_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source": source,
        "phase_id": phase_id,
        "instance_id": instance_id,
        "caller_id": caller_id,
        **extra,
    }
    (turn_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    if user_input:
        (turn_dir / "user_input.txt").write_text(user_input, encoding="utf-8")

    if response:
        (turn_dir / "inference_response.txt").write_text(response, encoding="utf-8")

    if prompt_data:
        rendered = prompt_data.get("rendered_prompt", "")
        if rendered:
            (turn_dir / "rendered_prompt.txt").write_text(
                str(rendered), encoding="utf-8"
            )

        feed = prompt_data.get("template_feed")
        if feed:
            try:
                (turn_dir / "template_feed.json").write_text(
                    json.dumps(feed, indent=2, default=str), encoding="utf-8"
                )
            except (TypeError, ValueError):
                (turn_dir / "template_feed.json").write_text(
                    str(feed), encoding="utf-8"
                )
