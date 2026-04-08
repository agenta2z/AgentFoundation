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

from attr import attrib, attrs


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

    root: str = attrib()

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
        return InferencerWorkspace(root=os.path.join(self.children_dir, name))

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
        with open(marker_path, "w") as f:
            json.dump(data, f, indent=2)

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
