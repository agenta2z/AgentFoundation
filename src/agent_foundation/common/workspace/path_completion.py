# pyre-strict
"""Pure path-completion helper for workspace path autocomplete.

Factored out of the WebUI ``/path-complete`` route so the same listing logic
can be reused by multiple backends (AgentFoundation WebUI, OpenStartup server).

The single public entry point :func:`complete_path` is filesystem-only and
framework-agnostic: it raises plain exceptions that callers map to HTTP
responses. Containment is enforced with ``Path.resolve().relative_to(...)``
(catching ``ValueError``) rather than a string ``startswith`` check, which is
spoofable by sibling prefixes (e.g. ``/tmp/root2`` vs ``/tmp/root``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class PrefixNotADirectory(ValueError):
    """Raised when the ``prefix`` base directory does not exist / is not a dir."""


class PathContainmentError(ValueError):
    """Raised when the resolved search directory escapes the prefix root."""


def _is_contained(child: Path, root: Path) -> bool:
    """Return True iff ``child`` resolves to a path inside ``root``.

    Uses ``relative_to`` on resolved paths. ``relative_to`` raises
    ``ValueError`` when ``child`` is not under ``root``; we treat that as
    out-of-root. This rejects both ``../`` traversal and sibling-prefix
    spoofing (``/tmp/root2`` is not under ``/tmp/root``).
    """
    try:
        child.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def complete_path(
    prefix: str,
    partial: str = "",
    dirs_only: bool = False,
    limit: int = 200,
) -> dict[str, Any]:
    """List subdirectories/files for path autocomplete.

    Joins ``prefix`` + ``partial`` to find the deepest valid directory, then
    lists its children that match the remaining partial name fragment.

    Args:
        prefix: Base directory path. Must be an existing directory.
        partial: User's partial path input, relative to ``prefix``.
        dirs_only: When True, only directories are returned.
        limit: Maximum number of suggestions (capped at 200).

    Returns:
        ``{"suggestions": [{"name", "path", "is_dir"}], "prefix", "partial"}``.
        Suggestion ``path`` values are relative to ``prefix``.

    Raises:
        PrefixNotADirectory: ``prefix`` is missing or not a directory.
        PathContainmentError: the resolved search directory escapes ``prefix``.
        ValueError: the path is otherwise invalid (e.g. OS error on resolve).
    """
    limit = max(0, min(limit, 200))

    base = Path(prefix)
    if not base.is_dir():
        raise PrefixNotADirectory(f"Prefix directory not found: {prefix}")

    # Split partial into a directory part + name fragment.
    partial_path = Path(partial) if partial else Path(".")
    full_path = base / partial_path

    if full_path.is_dir():
        search_dir = full_path
        fragment = ""
    else:
        search_dir = full_path.parent
        fragment = full_path.name

    # Validate search_dir is contained under prefix (reject traversal and
    # sibling-prefix spoofing). resolve() can raise OSError on some platforms.
    try:
        base_resolved = base.resolve()
        if not _is_contained(search_dir, base):
            raise PathContainmentError("Path traversal blocked")
        # Use the resolved search_dir for listing so child.relative_to(base_resolved)
        # is well-defined even when partial contained "." or symlinks.
        search_dir = search_dir.resolve()
    except OSError as exc:
        raise ValueError("Invalid path") from exc

    if not search_dir.is_dir():
        return {"suggestions": [], "prefix": prefix, "partial": partial}

    suggestions: list[dict[str, Any]] = []
    try:
        for child in sorted(search_dir.iterdir()):
            if child.name.startswith("."):
                continue
            if dirs_only and not child.is_dir():
                continue
            if fragment and not child.name.lower().startswith(fragment.lower()):
                continue

            rel = child.relative_to(base_resolved)
            is_dir = child.is_dir()
            display_name = child.name + ("/" if is_dir else "")

            suggestions.append(
                {
                    "name": display_name,
                    "path": str(rel) + ("/" if is_dir else ""),
                    "is_dir": is_dir,
                }
            )

            if len(suggestions) >= limit:
                break
    except PermissionError:
        pass

    return {"suggestions": suggestions, "prefix": prefix, "partial": partial}
