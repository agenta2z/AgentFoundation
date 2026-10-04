"""Stream counters live in the invocation (plan v8 §13, P10; B28 token/usage family).

A streaming leaf counts tokens, tool uses and usage while its stream runs and its
``_ainfer`` reports them (``SDKInferencerResponse``). The counters are a
``StreamStats`` component of the invocation: ``_ainfer`` starts each attempt at zero
(``_reset_stream_stats``) and the transport adds to the same object
(``_stream_stats``), so overlapping calls on one leaf each report their own counts.
The SDK leaves (rovochat, metamate SDK, devmate SDK, codex SDK, claude_code SDK)
no longer declare per-instance counters.
"""

from __future__ import annotations

import asyncio

import attr
import pytest
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_sdk_inferencer import (
    ClaudeCodeSdkInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.codex.codex_sdk_inferencer import (
    CodexSdkInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_sdk_inferencer import (
    DevmateSDKInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.metamate_sdk_inferencer import (
    MetamateSDKInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.rovochat_inferencer import (
    RovoChatInferencer,
)
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import RunContext
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from attr import attrs


@attrs
class _Counting(StreamingInferencerBase):
    """Streams its input one character at a time, slower for "slow" inputs; reports
    the tokens its own stream counted."""

    def _infer(self, inference_input, inference_config=None, **kwargs):
        raise NotImplementedError

    async def _ainfer_streaming(self, prompt, **kwargs):
        stats = self._stream_stats()
        for char in str(prompt):
            await asyncio.sleep(0.02 if "slow" in str(prompt) else 0)
            stats.tokens += 1
            yield char

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        stats = self._reset_stream_stats()
        await super()._ainfer(inference_input, inference_config, **kwargs)
        return stats.tokens


def _host(root):
    return RunContext.root(workspace=InferencerWorkspace(root=str(root)))


def test_overlapping_calls_on_one_leaf_each_report_their_own_counts(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(_Counting, "_HOST_PURE_CERTIFIED", True, raising=False)
    leaf = _Counting()

    async def main():
        return await asyncio.gather(
            leaf.ainfer("slow-one", run_context=_host(tmp_path / "a")),
            leaf.ainfer("b", run_context=_host(tmp_path / "b")),
        )

    assert asyncio.run(main()) == [len("slow-one"), 1]


def test_each_attempt_starts_from_zero():
    leaf = _Counting()
    assert asyncio.run(leaf.ainfer("abc")) == 3
    assert asyncio.run(leaf.ainfer("de")) == 2


@pytest.mark.parametrize(
    "cls",
    (
        RovoChatInferencer,
        MetamateSDKInferencer,
        DevmateSDKInferencer,
        CodexSdkInferencer,
        ClaudeCodeSdkInferencer,
    ),
)
def test_the_sdk_leaves_declare_no_per_instance_counters(cls):
    names = {field.name for field in attr.fields(cls)}
    assert not names & {"_last_token_count", "_last_tool_use_count", "_last_usage"}
