"""
Path resolution helper for automation schema tests.

This module sets up the Python path to allow imports from the agent_foundation package.
Import this module at the top of test files before importing agent_foundation modules.

Usage:
    import resolve_path  # Must be first import
    from agent_foundation.automation.schema.action_metadata import ActionTypeMetadata
"""
import sys
from pathlib import Path

# Configuration
PIVOT_FOLDER_NAME = 'test'  # The folder name we're inside of

# Get absolute path to this file
current_file = Path(__file__).resolve()

# Navigate up to find the pivot folder (test directory)
current_path = current_file.parent
while current_path.name != PIVOT_FOLDER_NAME and current_path.parent != current_path:
    current_path = current_path.parent

if current_path.name != PIVOT_FOLDER_NAME:
    raise RuntimeError(f"Could not find '{PIVOT_FOLDER_NAME}' folder in path hierarchy")

# ScienceModelingTools root is parent of test/ directory
agent_foundation_root = current_path.parent

# Add src directory to path for agent_foundation imports
src_dir = agent_foundation_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Add SciencePythonUtils if it exists (for tests that need it)
projects_root = agent_foundation_root.parent
rich_python_utils_src = projects_root / "SciencePythonUtils" / "src"

if rich_python_utils_src.exists() and str(rich_python_utils_src) not in sys.path:
    sys.path.insert(0, str(rich_python_utils_src))

# Verify the setup worked
_module_path = src_dir / "agent_foundation"
if not _module_path.exists():
    raise RuntimeError(f"agent_foundation module not found at {_module_path}")
