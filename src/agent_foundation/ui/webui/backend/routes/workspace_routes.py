
# pyre-strict
"""Workspace file access REST API for the WebUI.

Provides endpoints to browse workspace directory trees, read individual files,
and list output/result artifacts produced by task runs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from agent_foundation.common.workspace.path_completion import (
    complete_path,
    PathContainmentError,
    PrefixNotADirectory,
)
from rankevolve.src.common.workspace.layout import (  # TODO: migrate from rankevolve.src.common.workspace.layout
    ANALYSIS_DIR,
    get_request_text,
    list_output_files,
    list_result_files,
    LOGS_DIR,
    OUTPUTS_DIR,
    RESULTS_DIR,
    validate_workspace_subpath,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/tree")
async def workspace_tree(
    workspace: str = Query(..., description="Absolute path to workspace directory"),
) -> dict[str, Any]:
    """Return organized directory listing of a workspace."""
    ws = Path(workspace)
    if not ws.is_dir():
        raise HTTPException(status_code=404, detail="Workspace not found")

    return {
        "workspace": str(ws),
        "request": get_request_text(ws),
        "outputs": list_output_files(ws),
        "results": list_result_files(ws),
        "logs": _list_dir_files(ws / LOGS_DIR),
        "analysis": _list_dir_files(ws / ANALYSIS_DIR),
    }


@router.get("/file")
async def workspace_file(
    workspace: str = Query(..., description="Absolute path to workspace directory"),
    path: str = Query(..., description="Relative path within workspace"),
) -> dict[str, Any]:
    """Return contents of a single file within the workspace."""
    ws = Path(workspace)
    if not ws.is_dir():
        raise HTTPException(status_code=404, detail="Workspace not found")

    try:
        resolved = validate_workspace_subpath(ws, Path(path))
    except ValueError:
        raise HTTPException(status_code=403, detail="Path traversal blocked")

    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    suffix = resolved.suffix.lower()
    if suffix == ".json":
        try:
            content = json.loads(resolved.read_text(encoding="utf-8"))
            return {"name": resolved.name, "type": "json", "content": content}
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Cannot read file: {e}")

    return {"name": resolved.name, "type": "text", "content": text}


@router.get("/outputs")
async def workspace_outputs(
    workspace: str = Query(..., description="Absolute path to workspace directory"),
) -> dict[str, Any]:
    """Return list of output files with metadata."""
    ws = Path(workspace)
    if not ws.is_dir():
        raise HTTPException(status_code=404, detail="Workspace not found")

    return {"workspace": str(ws), "outputs": list_output_files(ws)}


@router.get("/results")
async def workspace_results(
    workspace: str = Query(..., description="Absolute path to workspace directory"),
) -> dict[str, Any]:
    """Return consensus results summary."""
    ws = Path(workspace)
    if not ws.is_dir():
        raise HTTPException(status_code=404, detail="Workspace not found")

    return {"workspace": str(ws), "results": list_result_files(ws)}


def _list_dir_files(directory: Path) -> list[dict[str, Any]]:
    """List files in a directory with basic metadata."""
    if not directory.is_dir():
        return []
    result = []
    for f in sorted(directory.iterdir()):
        if f.is_file():
            result.append(
                {
                    "name": f.name,
                    "size": f.stat().st_size,
                    "modified": f.stat().st_mtime,
                }
            )
    return result


@router.get("/path-complete")
async def path_complete(
    prefix: str = Query(..., description="Base directory path"),
    partial: str = Query("", description="User's partial path input"),
    dirs_only: bool = Query(True, description="Only return directories"),
    limit: int = Query(50, description="Max suggestions"),
) -> dict[str, Any]:
    """List subdirectories/files for path autocomplete.

    Joins prefix + partial to find the deepest valid directory, then lists
    its children that match the remaining partial name fragment. Delegates to
    the shared :func:`complete_path` helper, which hardens containment with
    ``Path.resolve().relative_to(...)``.
    """
    try:
        return complete_path(prefix, partial, dirs_only=dirs_only, limit=limit)
    except PrefixNotADirectory:
        raise HTTPException(status_code=404, detail=f"Prefix directory not found: {prefix}")
    except PathContainmentError:
        raise HTTPException(status_code=403, detail="Path traversal blocked")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")
