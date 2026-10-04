"""Builders shared by the native-orchestrator tests."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
    ToolExecutionResult,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native import (
    InMemoryRecordStore,
    NativeConversationalInferencer,
)
from agent_foundation.resources.tools.models import ParameterDef, ToolDefinition
from agent_foundation.resources.tools.registry import load_all_tools
from fakes import FakeBackendFactory, RecordingInteractive

FIXTURE_SOPS = str(Path(__file__).parent / "fixtures" / "sops")


def shared_dir() -> str:
    """A fresh session directory, for inferencers that continue one session."""
    return tempfile.mkdtemp(prefix="af_native_test_")


async def wait_for(predicate, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.005)


class RecordingExecutor:
    """tool_executor double: records calls, returns canned results."""

    def __init__(self, results: Optional[dict[str, str]] = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.results = results or {}

    async def __call__(self, name: str, arguments: dict) -> ToolExecutionResult:
        self.calls.append((name, dict(arguments or {})))
        return ToolExecutionResult(result=self.results.get(name, f"{name} done"))


def tool_registry(*, async_brief: bool = False) -> dict[str, ToolDefinition]:
    tools = {k: v for k, v in load_all_tools().items() if v.tool_type == "Conversation"}
    tools["write_brief"] = ToolDefinition(
        name="write_brief",
        description="Write a research brief.",
        tool_type="Action",
        parameters=[
            ParameterDef(name="topic", type="string", required=True, positional=True),
            ParameterDef(name="--depth", type="string", choices=["quick", "deep"]),
        ],
        asynchronous=async_brief,
    )
    return tools


def make_native(
    scripts: list,
    *,
    answers: Optional[list] = None,
    session_yolo: bool = False,
    async_brief: bool = False,
    record_store: Any = None,
    runtime_manager: Any = None,
    conversation_key: str = "conv-test",
    session_dir: Optional[str] = None,
    factory: Optional[FakeBackendFactory] = None,
    executor: Any = None,
    backend: Any = None,
    **kwargs: Any,
) -> tuple[
    NativeConversationalInferencer,
    FakeBackendFactory,
    RecordingInteractive,
    RecordingExecutor,
]:
    if factory is None:
        factory = FakeBackendFactory(scripts)
    else:
        factory.scripts.extend(scripts)
    interactive = RecordingInteractive(answers)
    executor = executor if executor is not None else RecordingExecutor()
    session_dir = session_dir or tempfile.mkdtemp(prefix="af_native_test_")
    native = NativeConversationalInferencer(
        backend=backend
        if backend is not None
        else {"kind": "claude_sdk", "cwd": session_dir},
        backend_factory=factory,
        tool_registry=tool_registry(async_brief=async_brief),
        tool_executor=executor,
        interactive=interactive,
        prior_context={"native_session_dir": session_dir},
        extra_sop_dirs=[FIXTURE_SOPS],
        allowed_sops=["mini_research"],
        session_yolo=session_yolo,
        record_store=record_store or InMemoryRecordStore(),
        runtime_manager=runtime_manager,
        conversation_key=conversation_key,
        **kwargs,
    )
    return native, factory, interactive, executor
