"""A terminal call's transport result lives in the invocation (plan v8 §13, P10 c1; B28).

The streaming transports publish a ``TerminalStreamResult`` (stdout, stderr, return
code) and ``_ainfer`` reads it back from the same invocation, so overlapping calls on
one leaf each parse their own result. ``_last_streaming_output`` / ``_stderr`` /
``_return_code`` are the key's compat fields: a bare call still leaves them for
``get_streaming_result``; a host call never writes them.
"""

from __future__ import annotations

import asyncio

import pytest
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    NoInvocationError,
    read_result,
    RunContext,
)
from agent_foundation.common.inferencers.terminal_inferencers.terminal_session_inferencer_base import (
    TerminalSessionInferencerBase,
)
from attr import attrs


@attrs
class _Shell(TerminalSessionInferencerBase):
    """Runs its input as a shell command; parses stdout and the return code."""

    def construct_command(self, inference_input, **kwargs):
        return str(inference_input)

    def parse_output(self, stdout, stderr, return_code):
        return {"output": stdout.strip(), "return_code": return_code}

    def _build_session_args(self, session_id, is_resume):
        return ""


def _host(root):
    return RunContext.root(workspace=InferencerWorkspace(root=str(root)))


def _compat(inf):
    return (
        inf._last_streaming_output,
        inf._last_streaming_stderr,
        inf._last_streaming_return_code,
    )


def test_overlapping_host_calls_on_one_leaf_each_parse_their_own_result(
    tmp_path, monkeypatch
):
    """Call A's process finishes after call B's; each returns its own output and
    exit code, and the leaf's compat fields stay untouched."""
    monkeypatch.setattr(_Shell, "_HOST_PURE_CERTIFIED", True, raising=False)
    leaf = _Shell()

    async def main():
        return await asyncio.gather(
            leaf.ainfer(
                "sleep 0.3; echo from-a; exit 3", run_context=_host(tmp_path / "a")
            ),
            leaf.ainfer("echo from-b", run_context=_host(tmp_path / "b")),
        )

    first, second = asyncio.run(main())
    assert (first.output, first.return_code) == ("from-a", 3)
    assert (second.output, second.return_code) == ("from-b", 0)
    assert _compat(leaf) == ("", "", 0)


def test_a_bare_call_leaves_the_compat_fields_for_the_getter():
    leaf = _Shell()
    result = asyncio.run(leaf.ainfer("echo bare; exit 2"))
    assert (result.output, result.return_code) == ("bare", 2)
    assert _compat(leaf) == ("bare\n", "", 2)


def test_the_result_is_read_only_inside_an_invocation():
    with pytest.raises(NoInvocationError):
        read_result(_Shell(), _Shell._TERMINAL_RESULT)
