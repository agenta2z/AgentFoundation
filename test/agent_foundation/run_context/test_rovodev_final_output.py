"""rovodev's clean output is per invocation and published as its final output
(plan v8 §5.3, §13, P10; B23).

rovodev streams noisy TUI stdout; its clean output comes from the call's
``--output-file`` (legacy) or from the trailing JSON of its filtered stdout
(non-legacy). Each attempt starts from empty output sources, so a call whose file
is read late (the cache path disabled) never returns the previous call's output,
and overlapping host calls each return and publish their own. A configured
``output_file`` is no longer clobbered at the end of the stream (inventory §35).
The outcome's ``final_output`` is what Conversational reads (``_final_output_at``);
the ``get_final_output()`` getter only in true no-ctx.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev import (
    rovodev_cli_inferencer as rovodev_module,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer import (
    RovoDevCliInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    NodeOutcomeState,
    read_outcome,
    RunContext,
)
from attr import attrs


async def _lines(*lines):
    for line in lines:
        yield line


def _answer(prompt):
    return f"clean answer to {prompt}"


def _base_pipeline(cache):
    """The base streaming pipeline: the transport writes the clean answer to the
    ``--output-file`` (legacy) or ends its stdout with it as JSON (non-legacy).
    With ``cache`` the base finally reads the file (``_get_clean_output_for_cache``);
    without it nothing does before rovodev's own fallback. "slow" prompts linger
    first."""

    async def pipeline(self, inference_input, inference_config=None, **kwargs):
        prompt = str(inference_input)
        await asyncio.sleep(0.05 if "slow" in prompt else 0)
        stdout = ["Working in /tmp\n", "[MCP] noise\n"]
        if self.enable_legacy:
            path = Path(kwargs.get("output_file") or self.output_file)
            path.write_text(_answer(prompt), encoding="utf-8")
        else:
            stdout.append('{"response": "%s"}\n' % _answer(prompt))
        try:
            async for chunk in self._yield_filter(_lines(*stdout)):
                yield chunk
        finally:
            if cache:
                self._get_clean_output_for_cache()

    return pipeline


def _use_base_pipeline(monkeypatch, cache):
    monkeypatch.setattr(
        RovoDevCliInferencer.__mro__[1],
        "_ainfer_streaming_pipeline",
        _base_pipeline(cache),
        raising=False,
    )


@pytest.fixture
def rovodev(tmp_path, monkeypatch):
    _use_base_pipeline(monkeypatch, cache=False)
    monkeypatch.setattr(rovodev_module, "find_latest_session_id", lambda **kw: None)
    work = tmp_path / "work"
    work.mkdir()
    return RovoDevCliInferencer(acli_path="/usr/bin/acli", target_path=str(work))


def _host(root):
    return RunContext.root(workspace=InferencerWorkspace(root=str(root)))


def test_a_late_read_output_file_never_returns_the_previous_calls_output(rovodev):
    first = asyncio.run(rovodev.ainfer("one"))
    second = asyncio.run(rovodev.ainfer("two"))
    assert (first.output, second.output) == (_answer("one"), _answer("two"))
    assert rovodev.get_final_output() == _answer("two")


def test_a_configured_output_file_stays_the_final_output(
    rovodev, tmp_path, monkeypatch
):
    _use_base_pipeline(monkeypatch, cache=True)
    rovodev.output_file = str(tmp_path / "answer.md")
    result = asyncio.run(rovodev.ainfer("q"))
    assert result.output == _answer("q")
    assert "[MCP] noise" in result.raw_output
    assert rovodev.get_final_output() == _answer("q")


def test_a_sync_call_records_no_final_output(rovodev, monkeypatch):
    asyncio.run(rovodev.ainfer("one"))

    def construct_command(inference_input, **kwargs):
        Path(kwargs["output_file"]).write_text(_answer("two"), encoding="utf-8")
        return "true"

    monkeypatch.setattr(rovodev, "construct_command", construct_command)
    assert rovodev.infer("two").output == _answer("two")
    assert rovodev.get_final_output() is None


def test_overlapping_host_calls_each_return_and_publish_their_own_output(
    rovodev, tmp_path
):
    slow_ctx, fast_ctx = _host(tmp_path / "a"), _host(tmp_path / "b")

    async def main():
        return await asyncio.gather(
            rovodev.ainfer("slow", run_context=slow_ctx),
            rovodev.ainfer("fast", run_context=fast_ctx),
        )

    slow, fast = asyncio.run(main())
    assert (slow.output, fast.output) == (_answer("slow"), _answer("fast"))
    assert read_outcome(slow_ctx).final_output == _answer("slow")
    assert read_outcome(fast_ctx).final_output == _answer("fast")
    assert rovodev._last_clean_output is None


def test_a_non_legacy_stream_publishes_its_trailing_json_response(rovodev, tmp_path):
    rovodev.enable_legacy = False
    ctx = _host(tmp_path / "a")

    async def drain():
        return [c async for c in rovodev.ainfer_streaming("q", run_context=ctx)]

    chunks = asyncio.run(drain())
    assert "[MCP] noise\n" in chunks
    assert read_outcome(ctx).final_output == _answer("q")
    assert rovodev._last_raw_stdout == ""


@attrs(slots=False)
class _NoisyLeaf(InferencerBase):
    """Streams differ from its final output: it publishes "published output",
    while its getter holds "getter output" (what a bare caller would see)."""

    streams_differ_from_final_output = True

    def _infer(self, inference_input, inference_config=None, **kwargs):
        raise NotImplementedError

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return "noisy stream"

    def get_final_output(self):
        return "getter output"

    def _outcome_for(self, frame):
        return NodeOutcomeState(final_output="published output")


def _assistant_messages(conversational):
    return [
        m["content"] for m in conversational.get_messages() if m["role"] == "assistant"
    ]


def test_conversational_reads_the_published_final_output(tmp_path):
    conversational = ConversationalInferencer(base_inferencer=_NoisyLeaf())
    asyncio.run(
        conversational.run_agentic_loop("task", run_context=_host(tmp_path / "a"))
    )
    assert _assistant_messages(conversational) == ["published output"]


def test_conversational_over_rovodev_never_reads_a_stale_getter(rovodev, tmp_path):
    rovodev._last_clean_output = "stale output of an earlier bare call"
    conversational = ConversationalInferencer(base_inferencer=rovodev)
    asyncio.run(
        conversational.run_agentic_loop("task", run_context=_host(tmp_path / "a"))
    )
    (message,) = _assistant_messages(conversational)
    assert message.startswith("clean answer to ")


def test_only_true_no_ctx_falls_back_to_the_getter(tmp_path):
    leaf = _NoisyLeaf()
    assert InferencerBase._final_output_at(leaf, None) == "getter output"
    assert InferencerBase._final_output_at(leaf, _host(tmp_path)) is None
