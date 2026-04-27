"""MockToolExecutor — deterministic tool dispatcher with execution tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolExecutionResult:
    """Compatible with conversational tool execution result shape."""
    result: str
    context_updates: dict = field(default_factory=dict)


class MockToolExecutor:
    """Deterministic tool dispatcher.

    Supports built-in tools (read_file, write_file, list_dir, deploy_service)
    and a `custom_handlers` dict for caller-provided behavior.

    self.executions records every (name, arguments) call for assertion.
    self.fail_on lets you inject failures by tool name.
    """

    def __init__(
        self,
        custom_handlers: Optional[Dict[str, Callable]] = None,
        fail_on: Optional[Dict[str, BaseException]] = None,
    ):
        self.executions: List[tuple] = []
        self.custom_handlers = custom_handlers or {}
        self.fail_on = fail_on or {}

    async def __call__(self, tool_name: str, arguments: dict) -> Any:
        self.executions.append((tool_name, dict(arguments)))

        if tool_name in self.fail_on:
            raise self.fail_on[tool_name]

        if tool_name in self.custom_handlers:
            handler = self.custom_handlers[tool_name]
            return handler(arguments)

        # Built-in deterministic tools
        if tool_name == "read_file":
            return f"<contents of {arguments.get('path', '?')}>"
        if tool_name == "write_file":
            return f"Wrote to {arguments.get('path', '?')}"
        if tool_name == "list_dir":
            return "file1.txt\nfile2.txt\nfile3.py"
        if tool_name == "deploy_service":
            region = arguments.get("region", "us-west-2")
            replicas = arguments.get("replicas", 1)
            return ToolExecutionResult(
                result=f"Deployed to {region} with {replicas} replicas",
                context_updates={
                    "deployment_status": "in_progress",
                    "deployment_id": "dep-12345",
                },
            )

        raise ValueError(f"Unknown tool: {tool_name}")
