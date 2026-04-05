# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Callable protocols for ConversationalInferencer pluggability.

These protocols define the interfaces that server-layer components implement
and framework-layer ConversationalInferencer consumes, keeping the dependency
direction clean (framework never imports server).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolExecutionResult:
    """Return type from tool executor."""

    result: str  # tool output text
    context_updates: dict[str, Any] = field(
        default_factory=dict
    )  # updates to apply to prior_context


@runtime_checkable
class ToolExecutorCallable(Protocol):
    async def __call__(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ToolExecutionResult: ...


@runtime_checkable
class ContextCompressorCallable(Protocol):
    async def __call__(self, context: str, max_length: int) -> str: ...


@runtime_checkable
class PromptRenderer(Protocol):
    def render(self, variables: dict[str, Any]) -> str: ...

    @property
    def template_source(self) -> str: ...
