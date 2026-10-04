# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""Workspace cache-directory helpers + the Experiment-Hub user cache dir.

Reimplemented locally (no ``rankevolve`` import). Two concerns:

* ``get_cache_dir(workspace)`` — the inferencer cache directory INSIDE a task
  workspace (``<workspace>/_runtime/inferencer_cache/``). This is the directory
  the ``WorkspaceStreamTailer`` watches for ``stream_*.txt`` files; bridges and
  the submission runner write there. Mirrors the original
  ``rankevolve.src.common.workspace.layout.get_cache_dir``.

* ``get_hub_cache_dir()`` — the process-wide on-disk cache for the Experiment
  Hub itself (``~/.cache/agent_foundation/experiment_hub/``), created on first
  use. Replaces RankEvolve's ``get_cache_dir()`` user-cache helper.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Relative subdirectory names (workspace layout) ───────────────────────────
RUNTIME_DIR: str = "_runtime"
CACHE_DIR: str = os.path.join(RUNTIME_DIR, "inferencer_cache")
OUTPUTS_DIR: str = "outputs"
RESULTS_DIR: str = "results"
LOGS_DIR: str = "logs"
ANALYSIS_DIR: str = "analysis"
REQUEST_FILE: str = "request.txt"


def get_cache_dir(workspace: str | Path) -> Path:
    """Return the inferencer cache directory for a task workspace."""
    return Path(workspace) / CACHE_DIR


def get_outputs_dir(workspace: str | Path) -> Path:
    """Return the outputs directory for a workspace."""
    return Path(workspace) / OUTPUTS_DIR


def get_results_dir(workspace: str | Path) -> Path:
    """Return the results directory for a workspace."""
    return Path(workspace) / RESULTS_DIR


def get_hub_cache_dir() -> Path:
    """Return (and create) the Experiment-Hub process-wide cache directory.

    Lives under the XDG cache root (``$XDG_CACHE_HOME`` if set, else
    ``~/.cache``) at ``agent_foundation/experiment_hub/``. Created on first
    use so callers can write into it without their own ``mkdir`` dance.
    """
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    cache_dir = Path(base) / "agent_foundation" / "experiment_hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
