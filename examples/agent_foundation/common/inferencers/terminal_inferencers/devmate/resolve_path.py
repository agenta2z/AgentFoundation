"""
Path resolution utility for ScienceModelingTools and related packages.
"""

import sys
from pathlib import Path


def _find_project_root(
    start_path: Path, marker_name: str = "ScienceModelingTools"
) -> Path:
    """Find the project root by searching upward from the start path."""
    current = start_path.resolve()

    for _ in range(20):
        if current.name == marker_name:
            return current
        if current.parent.name == marker_name:
            return current.parent
        if current.parent == current:
            break
        current = current.parent

    raise RuntimeError(
        f"Could not find project root '{marker_name}' starting from {start_path}."
    )


def _get_project_root() -> Path:
    """Get the root directory of ScienceModelingTools project."""
    current_file = Path(__file__).resolve()
    return _find_project_root(current_file, "ScienceModelingTools")


def _get_myprojects_root() -> Path:
    """Get the MyProjects root directory."""
    return _get_project_root().parent


def setup_paths():
    """Add source paths for all related packages to sys.path."""
    myprojects_root = _get_myprojects_root()
    project_root = _get_project_root()

    paths_to_add = [
        project_root / "src",
        project_root / "tools",
        myprojects_root / "SciencePythonUtils" / "src",
        myprojects_root / "WebAgent" / "src",
    ]

    for path in paths_to_add:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


setup_paths()
