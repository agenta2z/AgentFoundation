"""G3 — the same three-turn conversation through classic CI and native, with
per-turn tokens and latency, and a check that native never re-sends history.

Script (one fresh conversation per system, each in its own temp cwd):

  1. chat: remember a codeword, say who you are
  2. "Start the native_research SOP for me." — the SOP's Phase 0
     ``clarification`` widget is answered by a scripted interactive, Phase 1
     runs the ``write_brief`` action, Phase 2's ``confirmation`` is answered
  3. recall the codeword

Systems (model haiku — Sonnet 5.5 refuses the classic rendered prompt):

  * ``classic``    ``ConversationalInferencer`` over ``ClaudeCodeCliInferencer``,
                   streaming path (an interactive), a ``turn_N`` RunContext child
                   per turn (OpenStartup's shape)
  * ``native_cli`` ``NativeConversationalInferencer``, ``claude_cli`` backend
  * ``native_sdk`` ``NativeConversationalInferencer``, ``claude_sdk`` backend

Tokens come from the Claude transcripts of each run's project dir (one record
per API response: input, cache write, cache read, output), bucketed into turns
by timestamp; latency is wall time per ``run_agentic_loop`` call. Asserted for
every system: each turn completes, the SOP widget is shown and answered and
Phase 0 completes (also when the agent exits the finished SOP), turn 3 recalls
the codeword, tokens are recorded. Asserted for native: one vendor session,
each user turn's text sent once and verbatim, and no user message carries an
earlier turn's text.

``--repeat N`` runs each system N times and reports how often turn 2 asked the
SOP question through the question tool (model judgment varies run to run);
``--templates DIR`` renders the native lanes from another prompt-templates root
(an A/B copy of ``resources/prompt_templates``).

    source /tmp/af_env.sh
    PYTHONPATH="$PWD/src:$PWD/../RichPythonUtils/src:$AFL" \\
        python3 scripts/native_spikes/g3_native_vs_classic.py \\
        [--systems classic,native_cli,native_sdk] [--repeat N] [--templates DIR]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from _spike_common import (
    api_usage,
    attachments,
    Checks,
    entry_text,
    hook_contexts,
    read_entries,
    section,
    snapshot_texts,
    transcripts,
    user_prompts,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
    ToolExecutionResult,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.template_manager_renderer import (
    TemplateManagerPromptRenderer,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native import (
    InMemoryRecordStore,
    NativeConversationalInferencer,
    NativeRuntimeManager,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import RunContext
from agent_foundation.resources.tools._ci_host import build_ci_from_config
from agent_foundation.resources.tools.models import ParameterDef, ToolDefinition
from agent_foundation.resources.tools.registry import load_all_tools
from rich_python_utils.string_utils.formatting.template_manager.template_manager import (
    TemplateManager,
)

CODEWORD = "HERON-913"
TURNS = (
    f"Hi! Please remember the codeword {CODEWORD} for later. Who are you? Answer in one sentence.",
    "Start the native_research SOP for me.",
    "What codeword did I give you at the start? Reply with just the codeword.",
)
WIDGET_ANSWERS = ("quantum error correction", "yes, looks good")
EMPLOYEE = {
    "name": "Ada",
    "role": "Research Assistant",
    "mindset": "Be rigorous, cite evidence, keep the user informed.",
}
_NATIVE_KINDS = {"native_cli": "claude_cli", "native_sdk": "claude_sdk"}
_HERE = Path(__file__).resolve().parent
_SOP_DIR = str(_HERE / "sops")
_SOP_NAME = "native_research"
_CONFIG = (
    _HERE.parents[1]
    / "src/agent_foundation/resources/configs/conversational/default.yaml"
)


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


class ScriptedInteractive:
    """Collects streamed text; answers widgets from a script."""

    def __init__(self, answers: tuple[str, ...]) -> None:
        self.answers = list(answers)
        self.widgets: list[str] = []

    async def stream_token_batches(
        self, tokens, session_id, send_stream_end=False, turn_number=0
    ) -> str:
        return "".join([chunk async for chunk, _meta in tokens])

    async def asend_response(
        self, text, flag=None, input_mode=None, prompt_data=None, **_kw
    ) -> None:
        if input_mode is not None:
            self.widgets.append(str(getattr(input_mode, "mode", "?")))

    async def aget_input(self) -> str:
        return self.answers.pop(0) if self.answers else "ok"

    def set_round_context(self, ctx: Any) -> None:
        pass

    async def send_turn_boundary(
        self, session_id, turn_number=0, cache_folder=""
    ) -> None:
        pass

    def persist_pending_widget(self, tools, action_tools, blob) -> None:
        pass


@dataclass
class TurnRecord:
    n: int
    started: float
    ended: float
    text: str = ""
    rounds: int = 0
    widgets: int = 0
    error: str = ""


@dataclass
class SystemRun:
    name: str
    work: str
    turns: list[TurnRecord] = field(default_factory=list)
    phases_done: set[str] = field(default_factory=set)

    def entries(self) -> list[list[dict[str, Any]]]:
        return [read_entries(path) for path in transcripts(self.work)]


def _phases_done(inferencer: Any, sop_name: str) -> set[str]:
    """Completed phases of ``sop_name``, wherever it ended up: still active
    (completed in place), or paused/exited (the agent may call ``exit_sop``
    once the SOP is done)."""
    states = [getattr(inferencer, "sop_state", None)]
    states += getattr(inferencer, "suspended_sops", None) or []
    for state in states:
        if state is not None and state.sop_name == sop_name:
            return set(state.completed_phase_ids())
    return set()


def _epoch(stamp: str) -> float:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


def _native_renderer(templates: str) -> Any:
    """A prompt renderer over another prompt-templates root (``--templates``:
    an A/B copy of ``resources/prompt_templates`` with edited native lanes)."""
    return TemplateManagerPromptRenderer(
        template_manager=TemplateManager(
            templates=templates,
            active_template_root_space="conversation_native",
            active_template_type="main",
        ),
        template_key="session",
    )


def _build(
    system: str,
    model: str,
    work: str,
    interactive: ScriptedInteractive,
    templates: str = "",
) -> tuple[Any, Any]:
    if system == "classic":
        ci = build_ci_from_config(
            _CONFIG,
            base_inferencer=ClaudeCodeCliInferencer(target_path=work, model_name=model),
            tool_registry=_tools(),
            tool_executor=_executor,
            extra_sop_dirs=[_SOP_DIR],
            allowed_sops=[_SOP_NAME],
        )
        ci.set_prior_context({"employee": EMPLOYEE})
        return ci, None
    runtime = NativeRuntimeManager()
    native = NativeConversationalInferencer(
        backend={"kind": _NATIVE_KINDS[system], "cwd": work, "model": model},
        tool_registry=_tools(),
        tool_executor=_executor,
        interactive=interactive,
        prompt_renderer=_native_renderer(templates) if templates else None,
        prior_context={"employee": EMPLOYEE},
        extra_sop_dirs=[_SOP_DIR],
        allowed_sops=[_SOP_NAME],
        record_store=InMemoryRecordStore(),
        runtime_manager=runtime,
        conversation_key="g3",
        native_session_dir=tempfile.mkdtemp(prefix=f"g3_{system}_private_"),
        soft_max_iterations=20,
    )
    return native, runtime


async def run_system(system: str, model: str, templates: str = "") -> SystemRun:
    section(f"G3 {system}")
    work = tempfile.mkdtemp(prefix=f"g3_{system}_")
    print(f"  cwd (Claude transcripts are keyed by it): {work}", flush=True)
    interactive = ScriptedInteractive(WIDGET_ANSWERS)
    inferencer, runtime = _build(system, model, work, interactive, templates)
    root = (
        RunContext.root(workspace=InferencerWorkspace(root=work))
        if system == "classic"
        else None
    )
    run = SystemRun(system, work)
    try:
        for n, text in enumerate(TURNS, start=1):
            before = len(interactive.widgets)
            record = TurnRecord(n, time.time(), 0.0)
            try:
                kwargs: dict[str, Any] = {"interactive": interactive, "turn_number": n}
                if root is not None:
                    kwargs["run_context"] = root.child(f"turn_{n}")
                result = await inferencer.run_agentic_loop(text, **kwargs)
                record.text, record.rounds = result.text or "", result.iterations_used
            except Exception as exc:  # recorded and asserted below
                record.error = f"{type(exc).__name__}: {exc}"
            record.ended = time.time()
            record.widgets = len(interactive.widgets) - before
            run.turns.append(record)
            print(
                f"  turn {n}: {record.ended - record.started:5.1f}s rounds={record.rounds} "
                f"widgets={record.widgets} {record.error or record.text[:90]!r}",
                flush=True,
            )
        run.phases_done = _phases_done(inferencer, _SOP_NAME)
    finally:
        if runtime is not None:
            await inferencer.aclose()
            await runtime.aclose_all()
    return run


def _per_turn(run: SystemRun) -> list[dict[str, Any]]:
    """Each API response belongs to the last turn that started before it."""
    usage = [u for entries in run.entries() for u in api_usage(entries)]
    buckets: dict[int, list[dict[str, Any]]] = {t.n: [] for t in run.turns}
    for u in usage:
        stamp = _epoch(u["timestamp"])
        owner = [t.n for t in run.turns if t.started - 0.5 <= stamp]
        if owner:
            buckets[owner[-1]].append(u)
    rows = []
    for t in run.turns:
        mine = buckets[t.n]
        row = {
            key: sum(u[key] for u in mine)
            for key in ("input", "cache_write", "cache_read", "output")
        }
        row.update(
            system=run.name,
            turn=t.n,
            wall_s=t.ended - t.started,
            calls=len(mine),
            prompt=row["input"] + row["cache_write"] + row["cache_read"],
            # Input-token equivalent at API list-price ratios (5-minute cache
            # write 1.25x, cache read 0.1x).
            weighted=round(
                row["input"] + 1.25 * row["cache_write"] + 0.1 * row["cache_read"]
            ),
        )
        rows.append(row)
    return rows


_COLUMNS = (
    "wall_s",
    "calls",
    "input",
    "cache_write",
    "cache_read",
    "prompt",
    "weighted",
    "output",
)


def _table(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'system':<11} {'turn':>4} {'wall s':>7} {'API calls':>9} {'input':>7} {'cache wr':>9} "
        f"{'cache rd':>9} {'prompt':>8} {'in-equiv':>9} {'output':>7}"
    )
    print("\n" + header + "\n" + "-" * len(header))
    totals: dict[str, dict[str, float]] = {}
    for r in rows:
        _row(r["system"], str(r["turn"]), r)
        tot = totals.setdefault(r["system"], {})
        for key in _COLUMNS:
            tot[key] = tot.get(key, 0) + r[key]
    print("-" * len(header))
    for name, tot in totals.items():
        _row(name, "all", tot)
    print(
        "prompt = input + cache write + cache read; in-equiv = input + 1.25 x cache write + 0.1 x cache read"
    )


def _row(system: str, turn: str, r: dict[str, Any]) -> None:
    print(
        f"{system:<11} {turn:>4} {r['wall_s']:>7.1f} {int(r['calls']):>9} {int(r['input']):>7} "
        f"{int(r['cache_write']):>9} {int(r['cache_read']):>9} {int(r['prompt']):>8} "
        f"{int(r['weighted']):>9} {int(r['output']):>7}"
    )


def _check_common(run: SystemRun, rows: list[dict[str, Any]], c: Checks) -> None:
    k = f"G3[{run.name}]"
    for t in run.turns:
        c.check(
            f"{k} turn {t.n} completed",
            not t.error and bool(t.text.strip()),
            t.error or t.text[:80],
        )
    sop = run.turns[1] if len(run.turns) > 1 else None
    c.check(
        f"{k} SOP widget shown and answered in turn 2",
        sop is not None and sop.widgets >= 1,
        sop.widgets if sop else None,
    )
    c.check(
        f"{k} SOP Phase 0 completed", "0" in run.phases_done, sorted(run.phases_done)
    )
    recall = run.turns[2].text if len(run.turns) > 2 else ""
    c.check(f"{k} turn 3 recalls the codeword", CODEWORD in recall, recall[:80])
    c.check(
        f"{k} tokens recorded for every turn",
        all(r["calls"] and r["prompt"] for r in rows),
        [r["calls"] for r in rows],
    )


def _check_native(run: SystemRun, c: Checks) -> None:
    k = f"G3[{run.name}]"
    sessions = run.entries()
    c.check(
        f"{k} one vendor session for the conversation",
        len(sessions) == 1,
        len(sessions),
    )
    prompts = [p for entries in sessions for p in user_prompts(entries)]
    for n, text in enumerate(TURNS, start=1):
        c.check(
            f"{k} turn {n}'s text sent once, verbatim",
            prompts.count(text) == 1,
            f"{prompts.count(text)} exact match(es)",
        )
    resent = [
        (i, j)
        for j, later in enumerate(prompts)
        for i, earlier in enumerate(prompts[:j])
        if len(earlier.strip()) >= 20 and earlier in later
    ]
    c.check(
        f"{k} no user message carries an earlier user message", not resent, resent[:3]
    )
    c.check(
        f"{k} the codeword travels in exactly one user message",
        sum(CODEWORD in p for p in prompts) == 1,
        sum(CODEWORD in p for p in prompts),
    )
    entries = [e for es in sessions for e in es]
    c.info(
        f"{k} L2 hook blocks (chars each)",
        [len(entry_text(e)) for e in hook_contexts(entries)],
    )
    snapshots = snapshot_texts(entries)
    recorded = [
        e["attachment"].get("systemPrompt")
        for e in attachments(entries, "prompt_snapshot")
    ]
    if snapshots and isinstance(recorded[0], list) and recorded[0]:
        c.info(
            f"{k} recorded system prompt chars (preset + L1) / L1 append chars",
            f"{len(snapshots[0])} / {len(recorded[0][-1])}",
        )
    c.info(f"{k} user messages", [p[:60] for p in prompts])


def _check_classic(run: SystemRun, c: Checks) -> None:
    sessions = run.entries()
    prompts = [p for entries in sessions for p in user_prompts(entries)]
    carrying = sum(CODEWORD in p for p in prompts)
    c.info(
        f"G3[{run.name}] vendor sessions / user messages / carrying turn 1's codeword",
        f"{len(sessions)} / {len(prompts)} / {carrying}; prompt chars {[len(p) for p in prompts]}",
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--systems", default="classic,native_cli,native_sdk")
    parser.add_argument(
        "--templates",
        default="",
        help="prompt-templates root for the native systems (an A/B copy of "
        "resources/prompt_templates); default: the packaged templates",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run the conversation this many times per system and report how "
        "often the SOP widget was asked through the question tool",
    )
    args = parser.parse_args()
    c = Checks("G3")
    all_rows: list[dict[str, Any]] = []
    widget_runs: dict[str, list[bool]] = {}
    for _ in range(max(1, args.repeat)):
        for system in (s.strip() for s in args.systems.split(",")):
            try:
                run = await run_system(system, args.model, args.templates)
            except (
                Exception
            ) as exc:  # e.g. the system cannot be built; the rest still run
                c.check(f"G3[{system}] runs", False, f"{type(exc).__name__}: {exc}")
                continue
            rows = _per_turn(run)
            all_rows += rows
            _check_common(run, rows, c)
            if run.name == "classic":
                _check_classic(run, c)
            else:
                _check_native(run, c)
            asked = len(run.turns) > 1 and run.turns[1].widgets >= 1
            widget_runs.setdefault(system, []).append(asked)
    _table(all_rows)
    for system, asked in widget_runs.items():
        print(f"SOP widget asked in turn 2 [{system}]: {sum(asked)}/{len(asked)}")
    return c.exit_code()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
