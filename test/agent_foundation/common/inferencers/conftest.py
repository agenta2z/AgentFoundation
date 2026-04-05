"""Pytest configuration for inferencer tests.

Adds src/ and sibling project paths to sys.path so that
`agent_foundation` and `rich_python_utils` imports resolve correctly.
"""
import sys
from pathlib import Path

# Walk up to find the 'test' directory, then derive src/ from its parent
_current_path = Path(__file__).resolve().parent
while _current_path.name != "test" and _current_path.parent != _current_path:
    _current_path = _current_path.parent
_src_dir = _current_path.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Also add RichPythonUtils src to path (provides rich_python_utils.*)
_workspace_root = _current_path.parent.parent  # CoreProjects/
_rpu_src = _workspace_root / "RichPythonUtils" / "src"
if _rpu_src.exists() and str(_rpu_src) not in sys.path:
    sys.path.insert(0, str(_rpu_src))
