"""S17 — does classic CI re-send history into a continuing Claude session?

CI renders the whole conversation into every round's prompt, so each round
must reach a fresh vendor session; a continued one carries the transcript twice
(the vendor's history plus CI's rendered copy). Measured on real Claude Code:

  A. CLI, non-streaming (no ``interactive``), programmatic host (no run context)
  B. CLI, non-streaming, OpenStartup-style host (a ``turn_N`` child per turn)
  C. CLI, streaming (an ``interactive``), OpenStartup-style host
  D. SDK, streaming, OpenStartup-style host

Turn 1 needs an AF tool (two CI rounds); turn 2 is a plain question. Each
scenario runs in its own directory, so every Claude transcript of that project
is one of its vendor sessions: a session with more than one user message was
continued, and its later messages repeat what it already holds.

Sonnet 5.5 refuses the classic rendered prompt ("safeguards flagged this
message"), so the measurement uses haiku.

    source /tmp/af_env.sh
    PYTHONPATH="$PWD/src:$PWD/../RichPythonUtils/src:$AFL" \\
        python3 scripts/native_spikes/s17_ci_nonstreaming_duplication.py --model haiku
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
    ToolExecutionResult,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_sdk_inferencer import (
    ClaudeCodeSdkInferencer,
)
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import RunContext
from agent_foundation.resources.tools._ci_host import build_ci_from_config
from agent_foundation.resources.tools.models import ParameterDef, ToolDefinition

CODEWORD = "HERON-913"
TURNS = (
    f"Remember the codeword {CODEWORD}. Then use the lookup_record tool with "
    "key 'A-7' and tell me the record's status.",
    "What was the codeword I gave you? Reply with just the codeword.",
)
_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "src/agent_foundation/resources/configs/conversational/default.yaml"
)


def _tools() -> dict[str, ToolDefinition]:
    return {
        "lookup_record": ToolDefinition(
            name="lookup_record",
            description="Look up a record in the archive by key.",
            tool_type="Action",
            parameters=[ParameterDef(name="key", type="string", required=True)],
        )
    }


async def _executor(name: str, arguments: dict) -> ToolExecutionResult:
    return ToolExecutionResult(result=f"record {arguments.get('key')}: status=green")


class _Collector:
    """The one interactive method CI's streaming path needs."""

    async def stream_token_batches(
        self, tokens, session_id, send_stream_end=False, turn_number=0
    ) -> str:
        return "".join([chunk async for chunk, _meta in tokens])


def _transcripts(work: str) -> list[str]:
    config = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")
    project = re.sub(r"[^a-zA-Z0-9]", "-", work)
    return sorted(
        glob.glob(os.path.join(config, "projects", project, "*.jsonl")),
        key=os.path.getmtime,
    )


def _user_texts(path: str) -> list[str]:
    texts = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("type") != "user":
                continue
            content = entry.get("message", {}).get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                texts.extend(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
    return texts


async def _scenario(
    key: str, title: str, base_kind: str, model: str, *, stream: bool, host_root: bool
) -> None:
    work = tempfile.mkdtemp(prefix=f"s17_{key}_")
    if base_kind == "sdk":
        base: Any = ClaudeCodeSdkInferencer(target_path=work, model_id=model)
    else:
        base = ClaudeCodeCliInferencer(target_path=work, model_name=model)
    ci = build_ci_from_config(
        _CONFIG, base_inferencer=base, tool_registry=_tools(), tool_executor=_executor
    )
    root = (
        RunContext.root(workspace=InferencerWorkspace(root=work)) if host_root else None
    )
    print(f"\n== {key}. {title}")
    rounds = 0
    try:
        for n, text in enumerate(TURNS, start=1):
            result = await ci.run_agentic_loop(
                text,
                run_context=root.child(f"turn_{n}") if root else None,
                interactive=_Collector() if stream else None,
                turn_number=n,
            )
            rounds += result.iterations_used
            print(
                f"  turn {n}: {result.iterations_used} round(s), {result.text[:90]!r}"
            )
    finally:
        if base_kind == "sdk":
            await base.adisconnect()
    sessions = _transcripts(work)
    continued = 0
    for path in sessions:
        texts = _user_texts(path)
        continued += len(texts) > 1
        print(
            f"  session {Path(path).stem[:8]}: {len(texts)} user message(s), "
            f"{sum(CODEWORD in t for t in texts)} carry turn 1's text, "
            f"{sum(map(len, texts))} chars"
        )
    print(
        f"  => {rounds} CI rounds, {len(sessions)} vendor sessions, "
        f"{continued} continued with re-sent history"
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--only", default="ABCD")
    args = parser.parse_args()
    scenarios = (
        ("A", "CLI, non-streaming, programmatic host", "cli", False, False),
        ("B", "CLI, non-streaming, per-turn host", "cli", False, True),
        ("C", "CLI, streaming, per-turn host", "cli", True, True),
        ("D", "SDK, streaming, per-turn host", "sdk", True, True),
    )
    for key, title, base_kind, stream, host_root in scenarios:
        if key in args.only:
            await _scenario(
                key, title, base_kind, args.model, stream=stream, host_root=host_root
            )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
