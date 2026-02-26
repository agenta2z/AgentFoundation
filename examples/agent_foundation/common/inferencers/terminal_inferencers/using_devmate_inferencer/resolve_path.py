"""
Path resolution utility for ScienceModelingTools and related packages.

This module automatically adds the source paths for ScienceModelingTools,
SciencePythonUtils, and WebAgent to sys.path, enabling imports from these
packages without requiring them to be installed.

This file can be copied to any subdirectory within ScienceModelingTools
and will automatically find the project root.

Usage:
    # At the top of your script, before any package imports:
    import resolve_path  # noqa: F401

    # Now you can import from the packages:
    from agent_foundation.apis.metagen import generate_text
    from rich_python_utils.common_utils import get_
    from webaxon.core import Agent
"""

import sys
from pathlib import Path


def _find_project_root(
    start_path: Path, marker_name: str = "ScienceModelingTools"
) -> Path:
    """
    Find the project root by searching upward from the start path.

    Searches upward until it finds a directory with the marker name,
    or a directory containing setup.py with the project name.

    Args:
        start_path: Path to start searching from.
        marker_name: Name of the project directory to find.

    Returns:
        Path to the project root.

    Raises:
        RuntimeError: If project root cannot be found.
    """
    current = start_path.resolve()

    # Search upward until we find the project root
    for _ in range(20):  # Limit search depth to avoid infinite loops
        # Check if current directory is named after the project
        if current.name == marker_name:
            return current

        # Check if parent directory is named after the project
        if current.parent.name == marker_name:
            return current.parent

        # Check if we've reached the filesystem root
        if current.parent == current:
            break

        current = current.parent

    raise RuntimeError(
        f"Could not find project root '{marker_name}' starting from {start_path}. "
        f"Make sure this file is within the {marker_name} project directory."
    )


def _get_project_root() -> Path:
    """
    Get the root directory of ScienceModelingTools project.

    Automatically searches upward from this file's location.

    Returns:
        Path to the ScienceModelingTools project root.
    """
    current_file = Path(__file__).resolve()
    return _find_project_root(current_file, "ScienceModelingTools")


def _get_myprojects_root() -> Path:
    """
    Get the MyProjects root directory (parent of all sibling projects).

    Returns:
        Path to the MyProjects directory containing all projects.
    """
    project_root = _get_project_root()
    # ScienceModelingTools is inside MyProjects
    return project_root.parent


def setup_paths():
    """
    Add source paths for all related packages to sys.path.

    Adds paths for:
    - ScienceModelingTools/src
    - ScienceModelingTools/tools
    - SciencePythonUtils/src
    - WebAgent/src
    """
    myprojects_root = _get_myprojects_root()
    project_root = _get_project_root()

    # Define paths to add
    paths_to_add = [
        # ScienceModelingTools source
        project_root / "src",
        # ScienceModelingTools tools (for tools.ui.chatbot_demo, etc.)
        project_root / "tools",
        # SciencePythonUtils source
        myprojects_root / "SciencePythonUtils" / "src",
        # WebAgent source
        myprojects_root / "WebAgent" / "src",
    ]

    # Add paths that exist and aren't already in sys.path
    for path in paths_to_add:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


# Automatically set up paths when this module is imported
setup_paths()


# For debugging: print the paths that were added
if __name__ == "__main__":
    print("Path resolution for ScienceModelingTools")
    print("=" * 60)
    print(f"This file: {Path(__file__).resolve()}")
    print(f"Project root: {_get_project_root()}")
    print(f"MyProjects root: {_get_myprojects_root()}")
    print("\nPaths added to sys.path:")
    for i, p in enumerate(sys.path[:10]):
        print(f"  [{i}] {p}")
    if len(sys.path) > 10:
        print(f"  ... and {len(sys.path) - 10} more")
