"""
Path resolution helper for ActionGraph examples.

This module sets up the Python path to allow imports from the agent_foundation
and rich_python_utils packages. Import this module at the top of example files
before importing any project modules.

Usage:
    import resolve_path  # Must be first import
    from agent_foundation.automation.schema.action_graph import ActionGraph
"""
import sys
from pathlib import Path

# Configuration
PROJECT_ROOT_NAME = 'ScienceModelingTools'  # The project root folder name

# Get absolute path to this file
current_file = Path(__file__).resolve()

# Navigate up to find the project root by name
current_path = current_file.parent
while current_path.name != PROJECT_ROOT_NAME and current_path.parent != current_path:
    current_path = current_path.parent

if current_path.name != PROJECT_ROOT_NAME:
    raise RuntimeError(f"Could not find '{PROJECT_ROOT_NAME}' folder in path hierarchy")

project_root = current_path

# Add src directory to path for agent_foundation imports
src_dir = project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Add sibling science packages if they exist
projects_root = project_root.parent
rich_python_utils_src = projects_root / "SciencePythonUtils" / "src"

if rich_python_utils_src.exists() and str(rich_python_utils_src) not in sys.path:
    sys.path.insert(0, str(rich_python_utils_src))

# Verify the setup worked
_modeling_tools_module_path = src_dir / "agent_foundation"
if not _modeling_tools_module_path.exists():
    raise RuntimeError(f"agent_foundation module not found at {_modeling_tools_module_path}")
