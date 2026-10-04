"""Real end-to-end driver for the native conversational orchestrator.

Drives a multi-turn conversation through the REAL Claude Code backend and a
lightweight research SOP (clarification -> write_brief action -> confirmation),
verifying the native path: L1 persona/catalog in the system prompt, AF tools
over in-process MCP, deferred widgets, SOP phase progression, and resume.

Run (not a buck test; spawns the real `claude` binary):

    source /tmp/af_env.sh
    PYTHONPATH="$PWD/src:$PWD/../RichPythonUtils/src:$AFL" \
        python3 scripts/native_spikes/e2e_research_sop.py --scenario all --model sonnet
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Any

from _spike_common import find_transcript, read_entries, tool_uses, user_prompts
from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
    ToolExecutionResult,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native import (
    InMemoryRecordStore,
    NativeConversationalInferencer,
    NativeRuntimeManager,
)
from agent_foundation.resources.tools.models import ParameterDef, ToolDefinition
from agent_foundation.resources.tools.registry import load_all_tools

_SOP_DIR = str(Path(__file__).parent / "sops")
_CLAUDE_KINDS = frozenset({"claude_sdk", "claude_cli"})
_WRITES = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})
_EMPLOYEE = {
    "name": "Ada",
    "role": "Research Assistant",
    "mindset": "Be rigorous, cite evidence, keep the user informed.",
}


def _tools() -> dict[str, ToolDefinition]:
    tools = {k: v for k, v in load_all_tools().items() if v.tool_type == "Conversation"}
    tools["write_brief"] = ToolDefinition(
        name="write_brief",
        description="Write a short research brief on a topic.",
        tool_type="Action",
        parameters=[
            ParameterDef(name="topic", type="string", required=True, positional=True)
        ],
    )
    return tools


async def _executor(name: str, arguments: dict) -> ToolExecutionResult:
    if name == "write_brief":
        topic = arguments.get("topic", "the topic")
        return ToolExecutionResult(
            result=f"BRIEF on {topic}:\n1. Background\n2. Key questions\n3. Next steps"
        )
    return ToolExecutionResult(result=f"{name} done")


class PrintingInteractive:
    """Prints streamed text and widgets; answers widgets from a scripted queue."""

    def __init__(self, answers: list) -> None:
        self.answers = list(answers)
        self.widget_count = 0

    async def stream_token_batches(
        self, tokens, session_id, send_stream_end=False, turn_number=0
    ):
        chunks = []
        async for chunk, _meta in tokens:
            chunks.append(chunk)
            print(chunk, end="", flush=True)
        print()
        return "".join(chunks)

    async def asend_response(
        self, text, flag=None, input_mode=None, prompt_data=None, **_kw
    ):
        self.widget_count += 1
        meta = getattr(input_mode, "metadata", {}) or {}
        kind = "compound" if meta.get("compound") else getattr(input_mode, "mode", "?")
        print(f"\n  [WIDGET #{self.widget_count} {kind}] {text!r}")

    async def aget_input(self):
        answer = self.answers.pop(0) if self.answers else "ok"
        print(f"  [USER ANSWERS] {answer!r}")
        return answer

    def set_round_context(self, ctx):
        pass

    async def send_turn_boundary(self, session_id, turn_number=0, cache_folder=""):
        pass

    def persist_pending_widget(self, tools, action_tools, blob):
        pass


def _make(
    model: str,
    interactive,
    record_store,
    runtime,
    session_dir,
    backend_kind="claude_sdk",
    environment="inherit",
) -> NativeConversationalInferencer:
    return NativeConversationalInferencer(
        backend={
            "kind": backend_kind,
            "cwd": session_dir,
            "model": model,
            "l2_envelope_allowed": backend_kind != "claude_sdk",
            "environment": environment,
        },
        tool_registry=_tools(),
        tool_executor=_executor,
        interactive=interactive,
        prior_context={"employee": _EMPLOYEE, "native_session_dir": session_dir},
        extra_sop_dirs=[_SOP_DIR],
        allowed_sops=["native_research"],
        record_store=record_store,
        runtime_manager=runtime,
        conversation_key="e2e",
        soft_max_iterations=20,
    )


async def _turn(native, text, **kw) -> Any:
    print(f"\n{'=' * 70}\nUSER: {text}\n{'-' * 70}")
    result = await native.run_agentic_loop(text, **kw)
    print(
        f"\n{'-' * 70}\n[result] iterations={result.iterations_used} "
        f"has_widget={result.has_conversation_tool} meta={result.native_meta}"
    )
    return result


async def scenario_chat(model: str, backend_kind: str = "claude_sdk") -> None:
    print("\n########## SCENARIO: chat + catalog ##########")
    runtime = NativeRuntimeManager()
    native = _make(
        model,
        PrintingInteractive([]),
        InMemoryRecordStore(),
        runtime,
        tempfile.mkdtemp(),
        backend_kind,
    )
    async with native:
        r1 = await _turn(native, "Hi, who are you?", turn_number=1)
        assert "Ada" in r1.text or "Research" in r1.text, "persona (L1) not reflected"
        r2 = await _turn(native, "What SOPs can you run?", turn_number=2)
        assert "research" in r2.text.lower(), "SOP catalog (L1) not reflected"
    print("\n[scenario_chat PASS]")


async def scenario_sop(model: str, backend_kind: str = "claude_sdk") -> None:
    print("\n########## SCENARIO: full research SOP ##########")
    runtime = NativeRuntimeManager()
    interactive = PrintingInteractive(["quantum error correction", "yes, looks good"])
    native = _make(
        model,
        interactive,
        InMemoryRecordStore(),
        runtime,
        tempfile.mkdtemp(),
        backend_kind,
    )
    async with native:
        await _turn(native, "Start the native_research SOP for me.", turn_number=1)
        for i in range(2, 7):
            if native.sop_state and set(native.sop_state.completed_phase_ids()) >= {
                "0",
                "1",
            }:
                break
            await _turn(native, "continue", turn_number=i, origin="user")
        state = native.sop_state
        done = set(state.completed_phase_ids()) if state else set()
        print(
            f"\n[completed phases] {sorted(done)}; widgets shown={interactive.widget_count}"
        )
        assert interactive.widget_count >= 1, "expected at least one widget"
        assert "0" in done, "Phase 0 (topic) did not complete"
    print("\n[scenario_sop PASS]")


async def scenario_resume(model: str, backend_kind: str = "claude_sdk") -> None:
    """A rebuilt inferencer continues the same vendor session. Asserted on
    transport evidence: the record keeps the vendor session id and generation
    across the rebuild, and (Claude) that session's transcript holds turn 1
    before turn 2. The recall check cannot be satisfied by the agent's memory
    files: the codeword is new each run, Claude runs hermetic (no user
    CLAUDE.md, settings or hooks) and must not have written any file."""
    print("\n########## SCENARIO: resume after rebuild ##########")
    store = InMemoryRecordStore()
    runtime = NativeRuntimeManager()
    session_dir = tempfile.mkdtemp()
    claude = backend_kind in _CLAUDE_KINDS
    environment = "hermetic" if claude else "inherit"
    codeword = f"PERIWINKLE-{uuid.uuid4().hex[:4].upper()}"
    first = (
        f"Remember the codeword {codeword}. Keep it in this conversation only: "
        "do not save it to memory or any file. Just acknowledge briefly."
    )
    ask = "What was the codeword? Reply with just the codeword."
    try:
        native = _make(
            model,
            PrintingInteractive([]),
            store,
            runtime,
            session_dir,
            backend_kind,
            environment=environment,
        )
        await _turn(native, first, turn_number=1)
        await native.aclose()
        before = store.load("e2e")
        rebuilt = _make(
            model,
            PrintingInteractive([]),
            store,
            runtime,
            session_dir,
            backend_kind,
            environment=environment,
        )
        r = await _turn(rebuilt, ask, turn_number=2)
        await rebuilt.aclose()
        after = store.load("e2e")
    finally:
        await runtime.aclose_all()
    assert before is not None and before.vendor_session_id, "turn 1 recorded no session"
    assert after is not None and (after.vendor_session_id, after.generation) == (
        before.vendor_session_id,
        before.generation,
    ), "the rebuilt inferencer did not continue the same vendor session"
    print(f"[resume] same vendor session {before.vendor_session_id[:8]} across rebuild")
    if claude:
        entries = read_entries(find_transcript(before.vendor_session_id))
        prompts = user_prompts(entries)
        at = [
            next((i for i, p in enumerate(prompts) if text in p), -1)
            for text in (first, ask)
        ]
        print(f"[resume] transcript: {len(prompts)} user prompts, turns at {at}")
        assert 0 <= at[0] < at[1], "the session transcript lacks turn 1 before turn 2"
        writes = [u.get("name") for u in tool_uses(entries) if u.get("name") in _WRITES]
        assert not writes, f"the agent wrote files ({writes}); recall is not evidence"
    assert codeword in r.text.upper(), "resume did not preserve vendor memory"
    print("\n[scenario_resume PASS]")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scenario", default="all", choices=["all", "chat", "sop", "resume"]
    )
    ap.add_argument("--model", default="sonnet")
    ap.add_argument(
        "--backend",
        default="claude_sdk",
        choices=["claude_sdk", "claude_cli", "devmate_dm", "codex_cli"],
    )
    args = ap.parse_args()
    scenarios = {"chat": scenario_chat, "sop": scenario_sop, "resume": scenario_resume}
    to_run = (
        scenarios
        if args.scenario == "all"
        else {args.scenario: scenarios[args.scenario]}
    )
    for name, fn in to_run.items():
        try:
            await fn(args.model, args.backend)
        except AssertionError as exc:
            print(f"\n[{name} FAIL] {exc}")
            raise


if __name__ == "__main__":
    asyncio.run(main())
