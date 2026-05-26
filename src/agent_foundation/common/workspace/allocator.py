"""Unified workspace allocation for tool runs.

Two-path architecture:
  - Path A (standalone CLI):     <repo>/_runtime/tasks/<tool>/<tool>_<TS>_<uuid8>/
  - Path B (server-affiliated):  <base_dir>/<tool>_<TS>_<uuid8>/
                                 where base_dir is typically <session_root>/tasks/

Helper takes a base_dir argument, NOT a session_context dict. The
dispatcher/executor layer is responsible for translating session_context
fields into the right base_dir value before calling the helper.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


_FALLBACK_HOME_DIR = Path.home() / ".agent_foundation" / "_runtime"


def find_runtime_root() -> Path:
    """Locate the canonical _runtime/ directory using a 4-tier fallback chain.

    Resolution order (first match wins):
      1. $AGENT_FOUNDATION_RUNTIME_DIR or $OPENTEAM_RUNTIME_DIR env var
         (CI/prod override).
      2. Walk up from this file to find a ``src/`` ancestor ->
         <src_parent>/_runtime (dev tree).
      3. Walk up from CWD to find ``src/`` ancestor or any directory
         containing BOTH ``src/`` and ``pyproject.toml`` ->
         <root>/_runtime (cwd-launched).
      4. Fallback: ~/.agent_foundation/_runtime (pip-installed package).
    """
    env_dir = os.environ.get("AGENT_FOUNDATION_RUNTIME_DIR") or os.environ.get(
        "OPENTEAM_RUNTIME_DIR"
    )
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    for ancestor in Path(__file__).resolve().parents:
        if ancestor.name == "src":
            return ancestor.parent / "_runtime"

    cwd = Path.cwd().resolve()
    for ancestor in [cwd, *cwd.parents]:
        if ancestor.name == "src":
            return ancestor.parent / "_runtime"
        if (ancestor / "src").is_dir() and (ancestor / "pyproject.toml").is_file():
            return ancestor / "_runtime"

    return _FALLBACK_HOME_DIR


def make_workspace_dirname(tool_name: str) -> str:
    """Generate ``<tool>_<YYYYMMDD>_<HHMMSS>_<uuid8>``.

    Pure function: produces a name without touching the filesystem.
    Raises ValueError if tool_name is empty or not a valid Python identifier.
    """
    if not tool_name or not tool_name.isidentifier():
        raise ValueError(
            f"tool_name must be a non-empty Python identifier, got: {tool_name!r}"
        )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{tool_name}_{ts}_{short}"


def allocate_tool_workspace(
    tool_name: str,
    base_dir: Optional[Path] = None,
) -> Path:
    """Allocate a fresh workspace directory for a tool run.

    Args:
        tool_name: must be a valid Python identifier.
        base_dir: if provided (absolute path), workspace is created at
            <base_dir>/<tool>_<TS>_<uuid8>/. If None, workspace is created
            at <find_runtime_root()>/tasks/<tool>/<tool>_<TS>_<uuid8>/.

    Returns:
        Path to newly-created workspace directory.

    Raises:
        ValueError: invalid tool_name or non-absolute base_dir.
        FileExistsError: 3 consecutive UUID8 collisions.
    """
    if not tool_name or not tool_name.isidentifier():
        raise ValueError(
            f"tool_name must be a non-empty Python identifier, got: {tool_name!r}"
        )

    if base_dir is not None:
        if not base_dir.is_absolute():
            raise ValueError(f"base_dir must be absolute, got: {base_dir!r}")
        parent = base_dir
    else:
        parent = find_runtime_root() / "tasks" / tool_name

    parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(3):
        dirname = make_workspace_dirname(tool_name)
        ws = parent / dirname
        try:
            ws.mkdir(exist_ok=False)
            return ws
        except FileExistsError:
            if attempt == 2:
                raise
            continue
    raise AssertionError("unreachable")
