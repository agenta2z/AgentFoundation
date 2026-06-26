"""Compatibility shim: ``agent_foundation.common.ui`` -> ``agent_foundation.ui``.

Production conversational handlers and several tests import
``agent_foundation.common.ui.<submodule>`` (``input_modes``, ``interactive_base``,
``terminal_interactive``, ``queue_interactive``, ...), but the package currently
lives at ``agent_foundation.ui``. This shim aliases the real package and its
submodules under the ``common.ui`` path so both import styles resolve to the
*same* module objects (no duplication).
"""

import sys as _sys

import agent_foundation.ui as _ui

# Re-export top-level public names.
from agent_foundation.ui import *  # noqa: F401,F403

# Mirror the real package's path/attributes so submodule discovery works.
__path__ = _ui.__path__  # type: ignore[attr-defined]

# Eagerly alias the known submodules so ``from agent_foundation.common.ui.X
# import Y`` resolves to ``agent_foundation.ui.X`` (the same loaded module).
for _sub in (
    "input_modes",
    "interactive_base",
    "terminal_interactive",
    "queue_interactive",
    "interactive_checkpoint",
    "graph_interactive_adapter",
    "graph_reporter_factory",
):
    try:
        _m = __import__(f"agent_foundation.ui.{_sub}", fromlist=[_sub])
        _sys.modules[f"agent_foundation.common.ui.{_sub}"] = _m
    except Exception:  # pragma: no cover - optional submodules
        pass
