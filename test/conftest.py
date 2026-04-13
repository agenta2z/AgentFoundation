"""Pytest configuration for AgentFoundation tests."""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent

# Add AgentFoundation src
sys.path.insert(0, str(_ROOT / "src"))

# Add RichPythonUtils src (dependency)
_RICH_SRC = _ROOT.parent / "RichPythonUtils" / "src"
if _RICH_SRC.exists():
    sys.path.insert(0, str(_RICH_SRC))
