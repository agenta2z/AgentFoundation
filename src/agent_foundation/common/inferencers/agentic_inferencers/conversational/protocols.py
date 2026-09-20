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
class HubAwareToolExecutor(ToolExecutorCallable, Protocol):
    """Capability Protocol for executors that can create batched task hubs.

    A host application that orchestrates experiments (e.g. RankEvolve's
    ``SessionToolExecutor``) structurally satisfies this. Framework code
    narrows ``tool_executor`` via ``isinstance(executor, HubAwareToolExecutor)``
    before calling ``create_experiment_hub``; AF apps without a Hub simply do
    not satisfy the Protocol and the caller degrades gracefully.

    Mock gotcha: a bare ``Mock()`` will lie about satisfying this Protocol
    (it autocreates any attribute). Tests MUST use
    ``MagicMock(spec=ToolExecutorCallable)`` so the ``create_experiment_hub``
    attribute is genuinely absent for the negative case.
    """

    async def create_experiment_hub(
        self,
        selected_details: list[dict[str, Any]],
        proposals_data: dict[str, Any],
        custom_queries: list[str] | None = None,
        group_by: str = "batch",
        auto_implement: bool = True,
    ) -> str: ...

    async def open_experiment_hub(
        self,
        proposals_data: dict[str, Any],
        selected_proposal_ids: list[str] | None = None,
        pre_select_top_n: int = 5,
        initial_view: str = "selection",
    ) -> str:
        """Open (or idempotently focus) the hub as a shell seeded with proposals.

        The "skip detailed selection, go straight to the hub" path. Returns the
        hub/dashboard id.
        """
        ...


@runtime_checkable
class DashboardAwareToolExecutor(ToolExecutorCallable, Protocol):
    """Generic capability Protocol for executors that can open a Dashboard subtab.

    The generic ``--<dashboard>`` convention maps a Conversation tool's
    ``tool.metadata.open_dashboard == "<dashboard_id>"`` onto a call of
    ``open_dashboard(dashboard_id, seed)``. ``HubAwareToolExecutor`` is the
    experiment-hub-specific specialization; a host that implements this generic
    method supports *any* Dashboard tool. Framework code narrows via
    ``isinstance(executor, DashboardAwareToolExecutor)`` and degrades gracefully
    when unsatisfied.

    Mock gotcha (same as HubAwareToolExecutor): a bare ``Mock()`` lies about
    satisfying this Protocol; tests MUST use ``MagicMock(spec=ToolExecutorCallable)``
    so ``open_dashboard`` is genuinely absent for the negative case.
    """

    async def open_dashboard(
        self,
        dashboard_id: str,
        seed: dict[str, Any] | None = None,
    ) -> str: ...


@runtime_checkable
class ContextCompressorCallable(Protocol):
    async def __call__(self, context: str, max_length: int) -> str: ...


@runtime_checkable
class PromptRenderer(Protocol):
    def render(self, variables: dict[str, Any]) -> str: ...

    @property
    def template_source(self) -> str: ...

    def set_variable(self, name: str, value: Any) -> None: ...
