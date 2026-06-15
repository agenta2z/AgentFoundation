"""Commit 3 (D3): HubAwareToolExecutor capability-Protocol recognition."""
from unittest.mock import MagicMock

from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
    HubAwareToolExecutor,
    ToolExecutionResult,
    ToolExecutorCallable,
)


class _HubExecutor:
    async def __call__(self, tool_name, arguments):
        return ToolExecutionResult(result="")

    async def create_experiment_hub(
        self, selected_details, proposals_data, custom_queries=None, group_by="batch"
    ):
        return "multi_task_1"


class _PlainExecutor:
    async def __call__(self, tool_name, arguments):
        return ToolExecutionResult(result="")


def test_hub_aware_executor_is_recognised():
    assert isinstance(_HubExecutor(), HubAwareToolExecutor)


def test_plain_executor_is_not_hub_aware():
    assert not isinstance(_PlainExecutor(), HubAwareToolExecutor)


def test_spec_restricted_mock_is_not_hub_aware():
    """The documented test discipline: spec-restrict so the hub method is absent."""
    m = MagicMock(spec=ToolExecutorCallable)
    assert not isinstance(m, HubAwareToolExecutor)
