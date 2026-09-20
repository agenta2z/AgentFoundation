# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""Reference-script library loader for the Experiment Hub setup wizard.

Two helpers used by the hub controller's ``setup_submission_script(...)``:

- ``_resolve_library_template(template_id)`` resolves a manifest entry to its
  constituent script paths (returns ``[]`` if the manifest does not exist yet
  — the library degrades gracefully).
- ``_inline_script(path)`` reads a script and returns a fenced markdown block
  ready for inlining into the PTI prompt. Truncates if the file exceeds a
  byte budget (PTI prompts have a context budget).

There is intentionally no caching here — manifests and scripts are tiny and
the setup wizard is invoked rarely (once per hub). Re-reads on every call
keep iteration simple and let the user edit templates without restarting.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)


# Default location of the bundled reference script library — used as the
# filesystem fallback when ``importlib.resources`` doesn't return a real
# directory (the common case under direct ``python -m`` execution but NOT
# under buck2's link-tree packaging — see ``_resolve_templates_dir``).
_FILESYSTEM_TEMPLATES_DIR: Path = (
    Path(__file__).resolve().parent / "resources" / "submission_templates"
)


def _resolve_templates_dir() -> Path | None:
    """Return the on-disk path to ``resources/submission_templates/``.

    Tries ``importlib.resources`` first (works under buck2 link-tree
    packaging), falls back to the source-tree filesystem path for
    direct-Python runs. Returns ``None`` if neither path exists yet
    (template library not yet vendored — the loader degrades to empty
    results gracefully).
    """
    try:
        from importlib import resources

        pkg = resources.files(
            "agent_foundation.experiment_hub.resources.submission_templates"
        )
        # Some Loader paths return a MultiplexedPath that doesn't pass
        # is_dir() via os.path; coerce to str and Path defensively.
        candidate = Path(str(pkg))
        if candidate.is_dir():
            return candidate
    except Exception:
        pass
    if _FILESYSTEM_TEMPLATES_DIR.is_dir():
        return _FILESYSTEM_TEMPLATES_DIR
    return None


# Byte cap when inlining a reference script into the PTI prompt. 64 KiB is a
# reasonable default — large enough to fit a typical submit_job.py with room
# for context, small enough to leave headroom for additional refs and the
# rest of the prompt. Truncation always preserves the FILE HEAD because the
# top of a script (imports, entry point, config) is the most informative
# part for the agent.
_DEFAULT_INLINE_BUDGET_BYTES: int = 64 * 1024


def _load_manifest(templates_dir: Path | None = None) -> dict[str, dict[str, object]]:
    """Read manifest.yaml and return ``{template_id -> entry_dict}``.

    Returns an empty dict if the manifest is absent or unreadable. Logs a
    warning on parse failure rather than raising — a bad manifest must not
    crash setup-wizard requests for users who don't pick a library template.
    """
    base = templates_dir or _resolve_templates_dir()
    if base is None:
        return {}
    manifest_path = base / "manifest.yaml"
    if not manifest_path.is_file():
        return {}
    try:
        # pyyaml is available to the hub's deps.
        import yaml  # @manual

        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to parse %s: %s", manifest_path, e)
        return {}
    if not isinstance(raw, dict):
        return {}
    templates = raw.get("templates", [])
    if not isinstance(templates, list):
        return {}
    out: dict[str, dict[str, object]] = {}
    for entry in templates:
        if not isinstance(entry, dict):
            continue
        tid = entry.get("id")
        if isinstance(tid, str) and tid:
            out[tid] = entry
    return out


def list_library_templates(
    templates_dir: Path | None = None,
) -> list[dict[str, object]]:
    """Return the manifest entries as a list, sorted by id.

    Used by the submission-templates service. For each template, expand the
    ``files`` list into a small preview structure so the frontend can show
    the file names without separately fetching each file.
    """
    base = templates_dir or _resolve_templates_dir()
    out: list[dict[str, object]] = []
    for tid, entry in sorted(_load_manifest(base).items()):
        files_raw = entry.get("files", [])
        files: list[dict[str, str]] = []
        if isinstance(files_raw, list):
            for fname in files_raw:
                if isinstance(fname, str) and fname:
                    files.append({"name": fname})
        out.append(
            {
                "id": tid,
                "label": str(entry.get("label", tid)),
                "description": str(entry.get("description", "")),
                "files": files,
            }
        )
    return out


def _resolve_library_template(
    template_id: str,
    templates_dir: Path | None = None,
) -> list[Path]:
    """Resolve a template ID from manifest.yaml to its constituent script paths.

    Returns absolute paths to the snapshot files bundled in
    ``resources/submission_templates/<id>/``. Returns an empty list when:
      - the manifest is missing (library not yet shipped)
      - the template ID is not in the manifest
      - the template's ``files`` list is missing or empty
      - any listed file is missing on disk

    This function never raises — failure modes are logged and degrade to ``[]``
    so the setup wizard still succeeds (the user just gets a smaller prompt).
    """
    if not template_id:
        return []
    base = templates_dir or _resolve_templates_dir()
    if base is None:
        return []
    manifest = _load_manifest(base)
    entry = manifest.get(template_id)
    if entry is None:
        logger.info(
            "Library template %s not found in manifest %s",
            template_id,
            base / "manifest.yaml",
        )
        return []
    files_raw = entry.get("files", [])
    if not isinstance(files_raw, list):
        return []
    template_dir = base / template_id
    out: list[Path] = []
    for fname in files_raw:
        if not isinstance(fname, str) or not fname:
            continue
        path = template_dir / fname
        if not path.is_file():
            logger.warning(
                "Library template %s lists missing file %s", template_id, path
            )
            continue
        out.append(path)
    return out


def _inline_script(
    path: Path,
    max_bytes: int = _DEFAULT_INLINE_BUDGET_BYTES,
) -> str:
    """Read a script file and return a fenced markdown block ready for inlining.

    Truncates with a clear marker if the file exceeds ``max_bytes`` (PTI prompts
    have a context budget). Falls back to ``<unreadable: <path>>`` on read
    error rather than failing the whole setup — one bad reference shouldn't
    break script generation.
    """
    if not isinstance(path, Path):
        path = Path(path)
    try:
        # Read up to max_bytes + 1 to detect truncation without reading the
        # entire file when it's much larger than the budget.
        with path.open("rb") as fh:
            raw = fh.read(max_bytes + 1)
    except Exception as e:
        logger.warning("Failed to read reference script %s: %s", path, e)
        return f"<unreadable: {path}>"
    truncated = len(raw) > max_bytes
    body = raw[:max_bytes].decode("utf-8", errors="replace")
    suffix = path.suffix.lstrip(".") or "text"
    # Common extensions we'd see for reference materials. ``json``/``yaml``
    # for launch configs, ``py`` for the actual scripts, ``md``/``txt`` for
    # notes. Anything else falls through as a fenced code block without
    # syntax highlighting — still readable.
    fence_lang = {
        "py": "python",
        "json": "json",
        "yaml": "yaml",
        "yml": "yaml",
        "md": "markdown",
        "sh": "bash",
        "bash": "bash",
        "txt": "",
    }.get(suffix, "")
    header = f"### Reference: `{path}`"
    if truncated:
        header += f" (truncated to {max_bytes} bytes)"
    return f"{header}\n\n```{fence_lang}\n{body}\n```\n"
