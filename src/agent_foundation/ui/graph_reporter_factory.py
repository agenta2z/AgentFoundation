"""Factory: pick the right graph reporter; precedence WS > Stdio > None.

Called from each tool executor's existing graph_reporter attach block.
No tool_cli.run_cli patch — factory is the single attach point.
"""
from __future__ import annotations
import logging
from typing import Any

_logger = logging.getLogger(__name__)


def make_graph_reporter(session_context: dict, task_id: str = "") -> Any:
    """Returns a graph_reporter or None."""
    interactive = session_context.get("interactive")
    if interactive is not None and task_id:
        try:
            from agent_foundation.ui.graph_interactive_adapter import WebSocketGraphReporter
            r = WebSocketGraphReporter(interactive, task_id)
            _logger.info("[graph_reporter_factory] WebSocketGraphReporter (task_id=%s)", task_id)
            return r
        except Exception as exc:
            _logger.warning("[graph_reporter_factory] WS attach failed: %s", exc)
    try:
        from agent_foundation.ui.stdio_graph_reporter import StdioGraphReporter
        r = StdioGraphReporter.from_env(task_id=task_id)
        if r is not None:
            _logger.info("[graph_reporter_factory] StdioGraphReporter (task_id=%s)", task_id)
            return r
    except ImportError as exc:
        _logger.debug("[graph_reporter_factory] StdioGraphReporter unavailable: %s", exc)
    except Exception as exc:
        _logger.warning("[graph_reporter_factory] Stdio attach failed: %s", exc)
    return None
