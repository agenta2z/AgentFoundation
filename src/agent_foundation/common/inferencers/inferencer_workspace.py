"""InferencerWorkspace — unified directory layout for flow inferencer file I/O.

Provides deterministic path resolution, directory creation, child workspace
composition, artifact scanning, and marker file management.

Standard layout::

    root/
    ├── outputs/       # final deliverable output
    ├── artifacts/     # step-completion markers, intermediate files
    ├── checkpoints/   # Workflow checkpoint JSON files
    ├── logs/          # per-round prompt/response logs
    ├── children/      # child inferencer workspaces (created on demand)
    ├── analysis/      # PTI analysis files (optional, PTI-specific)
    ├── results/       # PTI results (optional, PTI-specific)
    └── _runtime/      # app-layer cache/temp (optional)

NOT related to Debuggable's ``write_json`` parts system (independent).
NOT related to ``experiment_management/workspace.py`` (different domain).
"""

import glob
import json
import os
from datetime import datetime, timezone
from typing import List, Optional

from agent_foundation.common.workspace.layout import (
    ARTIFACTS_DIR,
    CHECKPOINTS_DIR,
    CHILDREN_DIR,
    LOGS_DIR,
    OUTPUTS_DIR,
)
from attr import attrib, attrs

DEFAULT_OUTPUT_FILENAME = "output.md"


def indexed_child_name(prefix: str, index: int) -> str:
    """Consistent zero-padded naming for indexed workspace children.

    All indexed workspace children (workers, flows, rounds, panelists) use this
    single format: ``{prefix}_{index:02d}`` (e.g. ``worker_00``, ``flow_01``,
    ``round_01``, ``panelist_01``). Centralizing the format here prevents
    naming inconsistencies across the orchestrator hierarchy.
    """
    return f"{prefix}_{index:02d}"


@attrs
class InferencerWorkspace:
    """Unified directory layout manager for flow inferencer workspaces.

    Provides path resolution, directory creation, child workspace composition,
    artifact scanning, and marker file management.  Does NOT own file I/O for
    outputs/artifacts — inferencers handle that themselves.

    Serialization: store ``workspace.root`` as a plain string attribute on
    ``@attrs`` classes (including ``@artifact_type``-decorated classes like PTI).
    Reconstruct the workspace object from the root path string in
    ``__attrs_post_init__``.
    """

    root: str = attrib(default="")

    # -- Standard directory properties --

    @property
    def outputs_dir(self) -> str:
        """Final output directory."""
        return os.path.join(self.root, OUTPUTS_DIR)

    @property
    def artifacts_dir(self) -> str:
        """Intermediate files, step-completion markers."""
        return os.path.join(self.root, ARTIFACTS_DIR)

    @property
    def checkpoints_dir(self) -> str:
        """Workflow checkpoint JSON files."""
        return os.path.join(self.root, CHECKPOINTS_DIR)

    @property
    def logs_dir(self) -> str:
        """Per-round prompt/response logs."""
        return os.path.join(self.root, LOGS_DIR)

    @property
    def children_dir(self) -> str:
        """Child inferencer workspace roots."""
        return os.path.join(self.root, CHILDREN_DIR)

    def deliverable_path(self, relative: str) -> str:
        """Resolve a deliverable path. Part 2 (two-axis model): deliverables live
        directly in ``outputs/`` (``final_deliverables/`` is RETIRED), so this is
        simply ``outputs/<relative>`` (== :meth:`output_path`).
        """
        return self.output_path(relative)

    @property
    def has_deliverables(self) -> bool:
        """True iff ``outputs/`` exists on disk AND is non-empty. Part 2: deliverables
        live directly in ``outputs/`` (``final_deliverables/`` retired). The non-empty
        check prevents spurious 'I have deliverables' signals from a directory that was
        created by ensure_dirs() but never written into.
        """
        d = self.outputs_dir
        return bool(d and os.path.isdir(d) and os.listdir(d))

    def deliverable_paths(self) -> List[str]:
        """Return all deliverable file paths (recursively) under ``outputs/``.

        Paths are relative to ``outputs/``. Returns empty list when the directory
        doesn't exist.
        """
        d = self.outputs_dir
        if not (d and os.path.isdir(d)):
            return []
        result = []
        for root_dir, _dirs, files in os.walk(d):
            for f in files:
                abs_path = os.path.join(root_dir, f)
                rel_path = os.path.relpath(abs_path, d)
                result.append(rel_path)
        return sorted(result)

    def surface_outputs_from(
        self,
        source_workspace: "InferencerWorkspace",
        *,
        namespace: "Optional[str]" = None,
        skip_existing: bool = True,
    ) -> List[str]:
        """Copy a source workspace's deliverables into this workspace's outputs_dir.

        v1.7 Phase 0 PRIMITIVE: low-level file-copy used by the boundary helpers
        in `deliverable_boundary.py`. Per-file copy preserves provenance.

        Args:
            source_workspace: The child workspace whose deliverables to copy.
            namespace: Optional subdirectory under self.outputs_dir to
                copy files into (e.g., "workers/worker_0" or "planner").
                If None, files copy directly to self.outputs_dir/.
            skip_existing: If True, never overwrite files that already exist.

        Returns:
            List of relative paths copied (relative to self.outputs_dir).

        Returns empty list (no-op) if source has no deliverables.
        """
        import shutil

        if not source_workspace.has_deliverables:
            return []

        # Compute destination root
        dst_root = self.outputs_dir
        if namespace:
            # Validate each path component
            for component in namespace.split("/"):
                if component:
                    self._validate_child_name(component)
            dst_root = os.path.join(dst_root, namespace)

        os.makedirs(dst_root, exist_ok=True)
        copied = []
        src_root = source_workspace.outputs_dir

        for root_dir, _dirs, files in os.walk(src_root):
            for f in files:
                src_path = os.path.join(root_dir, f)
                rel_path = os.path.relpath(src_path, src_root)
                dst_path = os.path.join(dst_root, rel_path)

                if skip_existing and os.path.exists(dst_path):
                    continue

                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                # Track relative to self.outputs_dir (include namespace)
                copied.append(os.path.relpath(dst_path, self.outputs_dir))

        return copied

    # -- Directory creation --

    def ensure_dirs(self, *extra_subdirs: str) -> None:
        """Create 4 core directories + optional extras.

        Core: ``outputs/``, ``artifacts/``, ``checkpoints/``, ``logs/``.
        Does NOT create ``children/`` (created on demand by :meth:`child`).
        Does NOT create ``analysis/``, ``results/``, ``_runtime/`` unless
        passed as *extra_subdirs*.

        Example::

            ws.ensure_dirs("analysis", "results", "_runtime")
        """
        for d in (
            self.outputs_dir,
            self.artifacts_dir,
            self.checkpoints_dir,
            self.logs_dir,
        ):
            os.makedirs(d, exist_ok=True)
        for sub in extra_subdirs:
            os.makedirs(os.path.join(self.root, sub), exist_ok=True)

    # -- Path resolution --

    def output_path(self, relative: str) -> str:
        """Resolve ``<root>/outputs/<relative>``."""
        return os.path.join(self.outputs_dir, relative)

    def artifact_path(self, relative: str) -> str:
        """Resolve ``<root>/artifacts/<relative>``."""
        return os.path.join(self.artifacts_dir, relative)

    def checkpoint_path(self, relative: str) -> str:
        """Resolve ``<root>/checkpoints/<relative>``."""
        return os.path.join(self.checkpoints_dir, relative)

    def log_path(self, relative: str) -> str:
        """Resolve ``<root>/logs/<relative>``."""
        return os.path.join(self.logs_dir, relative)

    def analysis_path(self, relative: str) -> str:
        """PTI-specific: ``<root>/analysis/<relative>``."""
        return os.path.join(self.root, "analysis", relative)

    def results_path(self, relative: str) -> str:
        """PTI-specific: ``<root>/results/<relative>``."""
        return os.path.join(self.root, "results", relative)

    def subdir(self, name: str) -> str:
        """Return ``<root>/<name>/`` without creating it."""
        return os.path.join(self.root, name)

    # -- Child workspace management --

    @staticmethod
    def _validate_child_name(name: str) -> None:
        """Reject names that could escape the workspace hierarchy."""
        if (
            not name
            or name == "."
            or ".." in name
            or "/" in name
            or "\\" in name
            or os.sep in name
        ):
            raise ValueError(
                f"Invalid child workspace name: {name!r}. "
                f"Must not contain path separators or '..'."
            )

    def child(self, name: str) -> "InferencerWorkspace":
        """Create a child workspace rooted at ``children/<name>/``.

        Does NOT create directories on disk — call
        :meth:`ensure_dirs` on the returned workspace when ready.
        """
        self._validate_child_name(name)
        return InferencerWorkspace(
            root=os.path.join(self.children_dir, name),
        )

    def child_output(self, child_name: str, output_relative: str) -> str:
        """Resolve a child's output path without creating the child workspace.

        Returns ``<root>/children/<child_name>/outputs/<output_relative>``.
        Useful when the parent just needs to read a child's known output.
        """
        self._validate_child_name(child_name)
        return os.path.join(self.children_dir, child_name, OUTPUTS_DIR, output_relative)

    # -- Artifact scanning --

    def glob_outputs(self, pattern: str) -> List[str]:
        """Glob within ``outputs/``.  Returns sorted list."""
        return sorted(glob.glob(os.path.join(self.outputs_dir, pattern)))

    def glob_artifacts(self, pattern: str) -> List[str]:
        """Glob within ``artifacts/``.  Returns sorted list."""
        return sorted(glob.glob(os.path.join(self.artifacts_dir, pattern)))

    # -- Marker files --

    def write_marker(self, name: str, metadata: Optional[dict] = None) -> None:
        """Write ``artifacts/.<name>_completed`` with timestamp.

        Args:
            name: Phase name (e.g. ``"plan"``).
            metadata: Optional custom payload.  Defaults to a dict with
                ``completed_at`` (ISO-8601 UTC) and ``step``.
        """
        marker_path = self.artifact_path(f".{name}_completed")
        os.makedirs(os.path.dirname(marker_path), exist_ok=True)
        data = metadata or {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "step": name,
        }
        with open(marker_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def has_marker(self, name: str) -> bool:
        """Check ``artifacts/`` first, then ``outputs/`` (legacy fallback)."""
        if os.path.isfile(self.artifact_path(f".{name}_completed")):
            return True
        # Legacy: markers used to be in outputs/
        return os.path.isfile(self.output_path(f".{name}_completed"))

    def clear_marker(self, name: str) -> None:
        """Remove marker from both new and legacy locations."""
        for path in (
            self.artifact_path(f".{name}_completed"),
            self.output_path(f".{name}_completed"),
        ):
            if os.path.isfile(path):
                os.remove(path)


# =============================================================================
# Module-level path resolution helper for orchestrators that need to pass a
# child inferencer's output path downstream (e.g., BTA aggregator, PTI executor,
# MFDual peer flows). Part 2 (final_deliverables/ retired): resolution is a
# single tier — outputs/<filename> — see resolve_canonical_output_path() below.
# =============================================================================


def resolve_canonical_output_path(
    workspace: Optional["InferencerWorkspace"],
    *,
    filename: str = DEFAULT_OUTPUT_FILENAME,
    deliverables_fallback: str = "first_match",
) -> Optional[str]:
    """Returns the ABSOLUTE on-disk path to an inferencer's canonical
    output file, or ``None``.

    Part 2 (``final_deliverables/`` retired): deliverables now live directly in
    ``outputs/``, so resolution is a **single tier** — ``outputs/<filename>`` —
    for BOTH orchestrators AND leaf CLI inferencers (RovoDevCli, ClaudeCodeCli).
    There is no longer a separate deliverable subfolder to prefer or fall back
    within, so the former three-tier scheme collapses to one lookup.

    Resolution:
      * Try ``outputs/<filename>`` directly. If it exists on disk, return its
        absolute path.
      * Otherwise return ``None`` (no usable file exists).

    Parameters
    ----------
    workspace : InferencerWorkspace or None
        The inferencer's ``_workspace``. ``None`` returns ``None``.
    filename : str
        Preferred filename (default ``"output.md"``).
    deliverables_fallback : {"first_match", "alphabetical_scan", "none"}
        RETAINED for call-site compatibility ONLY — no longer consulted. It
        used to select fallback behavior within the (now-retired)
        ``final_deliverables/`` tier; with deliverables living directly in
        ``outputs/`` there is no separate subfolder to scan, so this parameter
        has no effect.

    Returns
    -------
    Optional[str]
        ABSOLUTE filesystem path (CWD-independent; safe for resume; safe
        for shell ``cp``), or ``None`` if no usable output file exists.

    Notes
    -----
    * **Returns absolute paths via ``os.path.abspath`` (NOT ``os.path.realpath``)**:
      Symlinks are PRESERVED, not resolved. This matches downstream usage —
      templates do ``cp '{{ prior_output_path }}' '{{ output_path }}'`` and
      callers expect to operate on the alias the orchestrator captured, not
      its symlink target.
    * Returns ``None`` (not ``""``) so callers can branch cleanly. Plan
      contract: orchestrators inject ``None`` (not empty string) into
      ``template_extra_feed`` for "no deliverable".
    * **TOCTOU caveat**: ``os.path.isfile`` checks happen at call time. A
      file deleted between this call and consumption returns a stale path
      reference. Long-running consumers should re-validate before use.
    * **Filename contract**: Caller MUST pass a non-empty, non-absolute
      filename (e.g. ``"output.md"``). No validation here.
    * Does NOT escape for shell — templates MUST single-quote: ``'{{ p }}'``
    * Never raises — all exceptions caught and treated as "not found".
    """
    if workspace is None:
        return None

    # Part 2 (final_deliverables/ retired): deliverables live directly in
    # ``outputs/``, so resolution is a single tier — ``outputs/<filename>`` — for
    # BOTH orchestrators and leaf CLI inferencers. The ``deliverables_fallback``
    # parameter is retained for call-site compatibility but no longer consulted
    # (there is no separate deliverable subfolder to fall back within).
    try:
        out_path = (
            workspace.output_path(filename)
            if hasattr(workspace, "output_path")
            else None
        )
    except Exception:
        out_path = None
    if out_path and os.path.isfile(out_path):
        return os.path.abspath(out_path)

    # Nothing on disk.
    return None
