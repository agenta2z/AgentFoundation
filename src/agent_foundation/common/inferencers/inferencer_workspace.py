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
from typing import List, Optional, Union

from attr import attrib, attrs

DEFAULT_OUTPUT_FILENAME = "output.md"


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
    # Controls whether a final_deliverables/ directory is created and used.
    # False (default)   → no deliverables directory
    # True              → uses the conventional name "final_deliverables/"
    # str               → uses that string as the subdirectory name
    use_final_deliverables_folder: "Union[bool, str]" = attrib(default=False)

    # -- Standard directory properties --

    @property
    def outputs_dir(self) -> str:
        """Final output directory."""
        return os.path.join(self.root, "outputs")

    @property
    def artifacts_dir(self) -> str:
        """Intermediate files, step-completion markers."""
        return os.path.join(self.root, "artifacts")

    @property
    def checkpoints_dir(self) -> str:
        """Workflow checkpoint JSON files."""
        return os.path.join(self.root, "checkpoints")

    @property
    def logs_dir(self) -> str:
        """Per-round prompt/response logs."""
        return os.path.join(self.root, "logs")

    @property
    def children_dir(self) -> str:
        """Child inferencer workspace roots."""
        return os.path.join(self.root, "children")

    @property
    def deliverables_dir(self) -> "Optional[str]":
        """Final deliverables directory under outputs/, or None if use_final_deliverables_folder is False.

        When ``use_final_deliverables_folder=True`` uses ``outputs/final_deliverables/``
        by convention. When a string, uses ``outputs/<name>/``.
        """
        if not self.use_final_deliverables_folder:
            return None
        name = (
            self.use_final_deliverables_folder
            if isinstance(self.use_final_deliverables_folder, str)
            else "final_deliverables"
        )
        return os.path.join(self.root, "outputs", name)

    def deliverable_path(self, relative: str) -> "Optional[str]":
        """Resolve a path relative to deliverables_dir, or None if not configured."""
        d = self.deliverables_dir
        if d is None:
            return None
        return os.path.join(d, relative)

    @property
    def has_deliverables(self) -> bool:
        """True iff deliverables_dir exists on disk AND is non-empty.

        v1.7 Phase 1: the non-empty check prevents spurious 'I have
        deliverables' signals from a directory that was created by
        ensure_dirs() but never written into.
        """
        d = self.deliverables_dir
        return bool(d and os.path.isdir(d) and os.listdir(d))

    def deliverable_paths(self) -> List[str]:
        """Return all file paths currently in deliverables_dir, recursively.

        Paths are relative to deliverables_dir. Returns empty list when
        deliverables are not configured or the directory doesn't exist.
        """
        d = self.deliverables_dir
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
        """Copy a source workspace's deliverables into this workspace's deliverables_dir.

        v1.7 Phase 0 PRIMITIVE: low-level file-copy used by the boundary helpers
        in `deliverable_boundary.py`. Per-file copy preserves provenance.

        Args:
            source_workspace: The child workspace whose deliverables to copy.
            namespace: Optional subdirectory under self.deliverables_dir to
                copy files into (e.g., "workers/worker_0" or "planner").
                If None, files copy directly to self.deliverables_dir/.
            skip_existing: If True, never overwrite files that already exist.

        Returns:
            List of relative paths copied (relative to self.deliverables_dir).

        Returns empty list (no-op) if either workspace lacks deliverables_dir,
        or if source has no deliverables.
        """
        import shutil

        if self.deliverables_dir is None:
            return []
        if source_workspace.deliverables_dir is None:
            return []
        if not source_workspace.has_deliverables:
            return []

        # Compute destination root
        dst_root = self.deliverables_dir
        if namespace:
            # Validate each path component
            for component in namespace.split("/"):
                if component:
                    self._validate_child_name(component)
            dst_root = os.path.join(dst_root, namespace)

        os.makedirs(dst_root, exist_ok=True)
        copied = []
        src_root = source_workspace.deliverables_dir

        for root_dir, _dirs, files in os.walk(src_root):
            _dirs[:] = [d for d in _dirs if d != "final_deliverables"]  # v4.8: prevent nesting
            for f in files:
                src_path = os.path.join(root_dir, f)
                rel_path = os.path.relpath(src_path, src_root)
                dst_path = os.path.join(dst_root, rel_path)

                if skip_existing and os.path.exists(dst_path):
                    continue

                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                # Track relative to self.deliverables_dir (include namespace)
                copied.append(os.path.relpath(dst_path, self.deliverables_dir))

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
        for d in (self.outputs_dir, self.artifacts_dir,
                  self.checkpoints_dir, self.logs_dir):
            os.makedirs(d, exist_ok=True)
        # v1.7 Phase 0 BUG FIX: also create deliverables_dir when configured.
        # Previously deliverables_dir was returned by the property but never
        # created on disk, causing has_deliverables() and downstream copy
        # operations to silently fail with "directory not found".
        if self.deliverables_dir is not None:
            os.makedirs(self.deliverables_dir, exist_ok=True)
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

        v1.7 Phase 0 BUG FIX: propagates ``use_final_deliverables_folder`` to
        the child workspace. Previously this was lost on every child(), so
        nested orchestrators silently ran without deliverable directories
        even though the root requested them.
        """
        self._validate_child_name(name)
        return InferencerWorkspace(
            root=os.path.join(self.children_dir, name),
            use_final_deliverables_folder=self.use_final_deliverables_folder,
        )

    def child_output(self, child_name: str, output_relative: str) -> str:
        """Resolve a child's output path without creating the child workspace.

        Returns ``<root>/children/<child_name>/outputs/<output_relative>``.
        Useful when the parent just needs to read a child's known output.
        """
        self._validate_child_name(child_name)
        return os.path.join(
            self.children_dir, child_name, "outputs", output_relative
        )

    # -- Artifact scanning --

    def glob_outputs(self, pattern: str) -> List[str]:
        """Glob within ``outputs/``.  Returns sorted list."""
        return sorted(glob.glob(os.path.join(self.outputs_dir, pattern)))

    def glob_artifacts(self, pattern: str) -> List[str]:
        """Glob within ``artifacts/``.  Returns sorted list."""
        return sorted(glob.glob(os.path.join(self.artifacts_dir, pattern)))

    # -- Marker files --

    def write_marker(
        self, name: str, metadata: Optional[dict] = None
    ) -> None:
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
# MFDual peer flows). Provides canonical 3-tier resolution with explicit
# fallback semantics — see resolve_canonical_output_path() below.
# =============================================================================


def resolve_canonical_output_path(
    workspace: Optional["InferencerWorkspace"],
    *,
    filename: str = DEFAULT_OUTPUT_FILENAME,
    deliverables_fallback: str = "first_match",
) -> Optional[str]:
    """Returns the ABSOLUTE on-disk path to an inferencer's canonical
    output file, or ``None``.

    Implements the **THREE-tier resolution** used by ``DualInferencer``'s
    ``_resolve_prior_proposer_output_path`` (the reference implementation),
    so it works for BOTH orchestrators (which have ``final_deliverables/``)
    AND leaf CLI inferencers (which write only to ``outputs/output.md``).

    **Critical**: Without Tier 2, this helper returns ``None`` for the most
    common production case (RovoDevCli, ClaudeCodeCli) which write to
    ``outputs/output.md`` but typically don't promote to
    ``final_deliverables/``. Tier 2 ensures leaf inferencers also resolve.

    Tiers (in order):
      Tier 1 — Deliverable file (preferred for orchestrators):
        If ``workspace.has_deliverables``, try ``final_deliverables/<filename>``.
        On miss, apply ``deliverables_fallback`` policy.
      Tier 2 — Outputs file (canonical for leaf inferencers):
        Try ``outputs/<filename>`` directly. This catches CLI inferencers
        and any case where the deliverable hasn't been promoted yet.
      Tier 3 — None: no usable file exists.

    Parameters
    ----------
    workspace : InferencerWorkspace or None
        The inferencer's ``_workspace``. ``None`` returns ``None``.
    filename : str
        Preferred filename (default ``"output.md"``).
    deliverables_fallback : {"first_match", "alphabetical_scan", "none"}
        Behavior WITHIN Tier 1 when the preferred filename is missing:
          * ``"first_match"`` (DEFAULT, MFDual semantics):
            return ``deliverable_paths()[0]``
          * ``"alphabetical_scan"`` (DualInferencer semantics):
            return first non-dotfile deliverable in sorted order
            (filters ``.self_promoted`` etc.)
          * ``"none"``: skip Tier 1 fallback; proceed to Tier 2 immediately

        (Note: this controls fallback WITHIN Tier 1 only. Tier 2 always runs
        when Tier 1 produces no result.)

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

    # === Tier 1: deliverable file ===
    if getattr(workspace, "has_deliverables", False):
        deliverables_dir = getattr(workspace, "deliverables_dir", None)
        if deliverables_dir:
            candidate = os.path.join(str(deliverables_dir), filename)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

            if deliverables_fallback != "none":
                try:
                    deliverable_paths_fn = getattr(
                        workspace, "deliverable_paths", None
                    )
                    deliverable_paths = (
                        deliverable_paths_fn()
                        if callable(deliverable_paths_fn)
                        else []
                    )
                except Exception:
                    deliverable_paths = []

                if deliverable_paths:
                    chosen: Optional[str] = None
                    if deliverables_fallback == "alphabetical_scan":
                        non_dotfiles = sorted(
                            p
                            for p in deliverable_paths
                            if not os.path.basename(p).startswith(".")
                        )
                        if non_dotfiles:
                            chosen = non_dotfiles[0]
                    else:  # "first_match" (DEFAULT)
                        chosen = deliverable_paths[0]

                    if chosen:
                        # deliverable_paths() may return basenames or full paths
                        if not os.path.isabs(chosen):
                            try:
                                chosen = workspace.deliverable_path(chosen)
                            except Exception:
                                chosen = None
                        if chosen and os.path.isfile(chosen):
                            return os.path.abspath(chosen)

    # === Tier 2: outputs/<filename> (CRITICAL for leaf CLI inferencers) ===
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

    # === Tier 3: nothing on disk ===
    return None
