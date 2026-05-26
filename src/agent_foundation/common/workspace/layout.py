"""Workspace directory layout constants and helpers.

A workspace is the top-level directory created per task/run.  Inside it
lives a fixed directory tree used by inferencer bridges and tool executors.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


RUNTIME_DIR = "_runtime"
CACHE_DIR: str = os.path.join(RUNTIME_DIR, "inferencer_cache")
OUTPUTS_DIR = "outputs"
ARTIFACTS_DIR = "artifacts"
CHECKPOINTS_DIR = "checkpoints"
CHILDREN_DIR = "children"
FINAL_DELIVERABLES_DIR = "final_deliverables"
RESULTS_DIR = "results"
LOGS_DIR = "logs"
ANALYSIS_DIR = "analysis"
REQUEST_FILE = "request.txt"
PROMPT_TEMPLATES_DIR = "prompt_templates"


def get_cache_dir(workspace: str | Path) -> Path:
    """Return the inferencer cache directory for a workspace."""
    return Path(workspace) / CACHE_DIR


def get_outputs_dir(workspace: str | Path) -> Path:
    """Return the outputs directory for a workspace."""
    return Path(workspace) / OUTPUTS_DIR


def get_results_dir(workspace: str | Path) -> Path:
    """Return the results directory for a workspace."""
    return Path(workspace) / RESULTS_DIR


def list_output_files(workspace: str | Path) -> list[dict[str, Any]]:
    """Return metadata about output files in the workspace."""
    outputs = Path(workspace) / OUTPUTS_DIR
    if not outputs.is_dir():
        return []
    result = []
    for f in sorted(outputs.iterdir()):
        if f.is_file():
            result.append(
                {
                    "name": f.name,
                    "size": f.stat().st_size,
                    "modified": f.stat().st_mtime,
                }
            )
    return result


def list_result_files(workspace: str | Path) -> list[dict[str, Any]]:
    """Return metadata about result files in the workspace."""
    results = Path(workspace) / RESULTS_DIR
    if not results.is_dir():
        return []
    out = []
    for f in sorted(results.iterdir()):
        if f.is_file():
            out.append(
                {
                    "name": f.name,
                    "size": f.stat().st_size,
                    "modified": f.stat().st_mtime,
                }
            )
    return out


def get_request_text(workspace: str | Path) -> str:
    """Read the request.txt from a workspace, or return empty string."""
    req = Path(workspace) / REQUEST_FILE
    if req.is_file():
        return req.read_text(encoding="utf-8", errors="replace")
    return ""


def validate_workspace_subpath(
    workspace: Path,
    requested_path: Path,
) -> Path:
    """Resolve *requested_path* and verify it lives inside *workspace*.

    Raises:
        ValueError: If the resolved path escapes the workspace root.
    """
    resolved_ws = workspace.resolve()
    resolved_req = (workspace / requested_path).resolve()
    if not resolved_req.is_relative_to(resolved_ws):
        raise ValueError(f"Path traversal blocked: {requested_path} escapes workspace")
    return resolved_req
