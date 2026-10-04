"""S18 (Codex) — two widget calls from one model response.

``raw``: ``codex exec --json`` (the ``codex_cli`` argv shape) against a spike
HTTP MCP server whose question tools answer like the AF bridge
(``AF_END_TURN — question queued …``) and record every ``tools/call`` with
its MCP ``_meta``.

  same   the prompt asks for both questions together: both calls arrive with
         the same ``_meta.itemId`` (one model output item)
  later  each result says the answer arrived, so the model asks the second
         question in a later response: the calls carry different
         ``_meta.itemId``s, while the ``--json`` stream between them looks
         exactly like ``same`` (no agent_message, no response boundary)

``native``: the real ``NativeConversationalInferencer`` on ``codex_cli`` runs
an SOP whose first phase asks ``clarification`` + ``single_choice`` (like
``model_optimization`` Phase 0a): one compound widget with both questions is
shown, and its answer completes the phase.

Recorded (``[INFO]``): the ``--json`` event / MCP call timeline of each run.

    source /tmp/af_env.sh
    PYTHONPATH="$PWD/src:$PWD/../RichPythonUtils/src:$AFL" \\
        python3 scripts/native_spikes/s18_codex_compound_widget.py [--parts raw,native]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from _spike_common import (
    Checks,
    HttpMcpServer,
    run_claude as run_jsonl,
    section,
    SpikeTool,
)

TOKEN_ENV = "AF_MCP_TOKEN"  # session/codex_cli.py _TOKEN_ENV
QUEUED = (
    "AF_END_TURN — question queued for the user. It is shown when you end your "
    "turn; the answer arrives as the next message. End your turn now and make no "
    "further tool calls."
)
ANSWERED = "The user answered: /tmp/toy. Continue with the next step now."
L1 = (
    "You gather inputs with the question tools of MCP server 'af'. Questions are "
    "shown to the user only after you end your turn. When a step needs several "
    "inputs, ask them all together in the same single response, then end your turn."
)
PROMPTS = {
    "same": (
        "Start setup. You need two inputs at once: the target path (clarification tool, "
        "prompt 'Which path?') and the discovery mode (single_choice tool, prompt 'How to "
        "find artifacts?', choices auto_discover, manual). Ask both together in one response."
    ),
    "later": (
        "Call the clarification tool with prompt 'Which path?'. Only after you have seen "
        "its result, call the single_choice tool with prompt 'How to find artifacts?' and "
        "choices auto_discover, manual. Strictly one after the other, never together."
    ),
}
_SOP_DIR = str(Path(__file__).parent / "sops")


# ----------------------------------------------------------------------------
# raw: codex exec + spike MCP server
# ----------------------------------------------------------------------------


def _question_tools(calls: list[dict[str, Any]], result: str) -> list[SpikeTool]:
    from mcp.server.lowlevel.server import request_ctx

    def make(name: str, properties: dict[str, Any]) -> SpikeTool:
        async def _ask(args: dict[str, Any]) -> str:
            meta = request_ctx.get().meta
            extra = dict(meta.model_extra or {}) if meta is not None else {}
            calls.append(
                {
                    "tool": name,
                    "t": time.time(),
                    "itemId": extra.get("itemId"),
                    "callId": extra.get("callId"),
                }
            )
            return result

        return SpikeTool(
            name=name,
            description=f"Ask the user a question ({name} widget).",
            properties=properties,
            required=tuple(properties),
            handler=_ask,
        )

    return [
        make("clarification", {"prompt": {"type": "string"}}),
        make(
            "single_choice",
            {
                "prompt": {"type": "string"},
                "choices": {"type": "array", "items": {"type": "string"}},
            },
        ),
    ]


def _argv(server: HttpMcpServer, prompt: str) -> list[str]:
    return [
        shutil.which("codex") or "codex",
        "exec",
        "--json",
        "--skip-git-repo-check",
        "-c",
        f"mcp_servers.af.url={json.dumps(server.url)}",
        "-c",
        f"mcp_servers.af.bearer_token_env_var={json.dumps(TOKEN_ENV)}",
        "-c",
        "mcp_servers.af.required=true",
        "-c",
        f"developer_instructions={json.dumps(L1)}",
        "--dangerously-bypass-approvals-and-sandbox",
        prompt,
    ]


def _timeline(c: Checks, k: str, events: list[dict], calls: list[dict]) -> list[str]:
    rows = []
    for e in events:
        item = e.get("item") or {}
        label = e.get("type", "")
        if item:
            label += f" {item.get('id')} {item.get('type')}"
            if item.get("type") == "mcp_tool_call":
                label += f" {item.get('tool')}"
        rows.append((e["_recv"], label))
    for call in calls:
        rows.append(
            (
                call["t"],
                f"MCP tools/call {call['tool']} itemId=…{str(call['itemId'])[-8:]}",
            )
        )
    rows.sort()
    t0 = rows[0][0] if rows else 0.0
    for t, label in rows:
        c.info(f"{k} t+{t - t0:6.2f}s", label)
    return [label for _, label in rows]


async def raw_part(c: Checks) -> None:
    for scenario, result in (("same", QUEUED), ("later", ANSWERED)):
        k = f"S18[raw/{scenario}]"
        section(k)
        calls: list[dict[str, Any]] = []
        server = await HttpMcpServer(_question_tools(calls, result)).start()
        try:
            run = await run_jsonl(
                _argv(server, PROMPTS[scenario]),
                cwd=tempfile.mkdtemp(prefix="s18_"),
                env={TOKEN_ENV: server.token},
                timeout=300,
            )
        finally:
            await server.stop()
        labels = _timeline(c, k, run.events, calls)
        tools = sorted(x["tool"] for x in calls)
        c.check(
            f"{k} both questions asked",
            tools == ["clarification", "single_choice"],
            tools,
        )
        items = [x["itemId"] for x in calls]
        c.check(f"{k} every call carries _meta.itemId", all(items), items)
        same = len(set(items)) == 1
        c.check(
            f"{k} itemIds {'equal' if scenario == 'same' else 'differ'}",
            same == (scenario == "same"),
            [str(i)[-8:] for i in items],
        )
        mcp = [i for i, label in enumerate(labels) if label.startswith("MCP")]
        if len(mcp) >= 2:
            between = labels[mcp[0] + 1 : mcp[1]]
            c.info(
                f"{k} --json events between the two calls",
                [x for x in between if not x.startswith("item.started")],
            )


# ----------------------------------------------------------------------------
# native: the real inferencer on codex_cli
# ----------------------------------------------------------------------------


class RecordingInteractive:
    def __init__(self) -> None:
        self.widgets: list[dict[str, Any]] = []

    async def stream_token_batches(
        self, tokens, session_id, send_stream_end=False, turn_number=0
    ):
        return "".join([chunk async for chunk, _meta in tokens])

    async def asend_response(
        self, text, flag=None, input_mode=None, prompt_data=None, **_kw
    ):
        meta = getattr(input_mode, "metadata", {}) or {}
        types = (
            [t.get("tool_type") for t in meta.get("tools", [])]
            if meta.get("compound")
            else [getattr(input_mode, "mode", "?")]
        )
        self.widgets.append(
            {"compound": bool(meta.get("compound")), "types": types, "meta": meta}
        )

    async def aget_input(self):
        last = self.widgets[-1]
        if last["compound"]:
            outs = [t.get("output_var") for t in last["meta"].get("tools", [])]
            answers = {
                "workflow_target_path": "/tmp/toy",
                "workflow_modeling_artifacts_mode": "auto_discover",
            }
            return {"values": {o: answers.get(o, "ok") for o in outs}}
        return "/tmp/toy"

    def set_round_context(self, ctx):
        pass

    async def send_turn_boundary(self, session_id, turn_number=0, cache_folder=""):
        pass

    def persist_pending_widget(self, tools, action_tools, blob):
        pass


async def native_part(c: Checks, model: str) -> None:
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

    k = "S18[native/codex_cli]"
    section(k)
    tools = {n: t for n, t in load_all_tools().items() if t.tool_type == "Conversation"}
    tools["write_brief"] = ToolDefinition(
        name="write_brief",
        description="Write a short brief on a topic.",
        tool_type="Action",
        parameters=[
            ParameterDef(name="topic", type="string", required=True, positional=True)
        ],
    )

    async def _executor(name: str, arguments: dict) -> ToolExecutionResult:
        return ToolExecutionResult(result=f"BRIEF on {arguments.get('topic')}: done")

    interactive = RecordingInteractive()
    runtime = NativeRuntimeManager()
    work = tempfile.mkdtemp(prefix="s18_native_")
    native = NativeConversationalInferencer(
        backend={
            "kind": "codex_cli",
            "cwd": work,
            "model": model,
            "l2_envelope_allowed": True,
        },
        tool_registry=tools,
        tool_executor=_executor,
        interactive=interactive,
        prior_context={
            "employee": {"name": "Ada", "role": "Assistant"},
            "native_session_dir": work,
        },
        extra_sop_dirs=[_SOP_DIR],
        allowed_sops=["native_compound"],
        record_store=InMemoryRecordStore(),
        runtime_manager=runtime,
        conversation_key="s18",
        soft_max_iterations=20,
    )
    try:
        async with native:
            result = await native.run_agentic_loop(
                "Start the native_compound SOP for the codebase at /tmp/toy.",
                turn_number=1,
            )
            state = native.sop_state
            done = sorted(state.completed_phase_ids()) if state else []
    finally:
        await runtime.aclose_all()
    for i, w in enumerate(interactive.widgets):
        c.info(f"{k} widget #{i + 1}", f"compound={w['compound']} {w['types']}")
    c.info(
        f"{k} result", f"iterations={result.iterations_used} text={result.text[:160]!r}"
    )
    first = interactive.widgets[0] if interactive.widgets else {}
    c.check(
        f"{k} first widget is ONE compound widget with clarification + single_choice",
        bool(first.get("compound"))
        and sorted(first.get("types", [])) == ["clarification", "single_choice"],
        first.get("types"),
    )
    c.check(f"{k} Phase 0a completed by that answer", "0a" in done, done)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", default="raw,native")
    parser.add_argument("--model", default="")
    args = parser.parse_args()
    c = Checks("S18")
    parts = args.parts.split(",")
    if "raw" in parts:
        await raw_part(c)
    if "native" in parts:
        await native_part(c, args.model)
    return c.exit_code()


if __name__ == "__main__":
    os.chdir(tempfile.mkdtemp())
    sys.exit(asyncio.run(main()))
