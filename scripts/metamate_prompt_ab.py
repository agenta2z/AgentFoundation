"""Replay fan-out worker prompts against MetaMate under prompt variants.

MetaMate runs a whole agent turn inside one web request, which the server cuts at
120 s, and a turn also stops at its per-request memory budget. This driver sends
the same worker prompts (``--questions``: a JSON list of ``{"id", "prompt"}``,
e.g. the shards a fan-out gate sent) once per variant, concurrently per question
so every variant meets the same server conditions, and records how each request
ended:

* ``A`` — the prompt as the fan-out sent it;
* ``B`` — plus a tool-call budget (``tool_call_budget_directive``, ``--budget``
  tool calls), then answer from what was verified;
* ``C`` — plus rounds: a small tool budget per turn, each turn ends with
  ``STATUS: CONTINUE`` or ``STATUS: DONE``, and the driver replies ``continue`` in
  the same conversation (``--max-rounds``), so every turn gets its own request
  time and memory budget;
* ``D`` — variant B plus an answer-length cap (``--answer-words``): writing the
  answer counts against the same 120 s, and a full task's ~40 KB answer alone
  takes most of it;
* ``E`` — variant B plus a structural cap (``answer_format_directive``,
  ``--answer-findings`` findings of a few bullets each), for when a word count
  is ignored.

Every request uses the fan-out workers' MetaMate settings, a fixed code scope
(``--scope-paths``, as ``reproduce_flow03 --scope-paths``) and one attempt; each
variant's workspace keeps the leaf's session log, whose
``MetamateConversationSummary`` records hold the server side of every failure.
Results go to ``OUT/results.jsonl`` (one line per question and variant) as they
finish.

Usage::

    buck2 run //_tony_dev/CoreProjects/AgentFoundation/scripts:metamate_prompt_ab -- \\
        --questions prompts.json --out /abs/out/dir [--arms A,B,C] [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
from typing import Any

import agent_foundation.common.configs.registered_targets  # noqa: F401 — registers inferencer aliases
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.code_scope_judge import (
    CodeSearchScope,
    Corpus,
    Repo,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.common import (
    answer_format_directive,
    tool_call_budget_directive,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.metamate_sdk_inferencer import (
    MetamateSDKInferencer,
)
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from rich_python_utils.common_utils.function_helper import FallbackMode

log: logging.Logger = logging.getLogger("metamate_prompt_ab")

ROUNDS_SUFFIX = """

## Working in rounds
MetaMate cuts off any single turn after about two minutes, and anything unfinished
in that turn is lost. So work in rounds:
- In each round, use at most 5 tool calls (searches and file reads), then stop using
  tools and write up what you found so far.
- End every round with exactly one final line: `STATUS: CONTINUE` if important work
  remains (say first what you will do next), or `STATUS: DONE` when your answer is
  complete.
- When I reply `continue`, do the next round, building on your earlier findings;
  do not repeat earlier work.
- In your DONE round, give the complete answer, consolidating every round."""

ANSWER_LENGTH_SUFFIX = """
Keep the answer under {words} words, since writing it counts against the same two \
minutes: lead with the findings, use compact bullets, and cite file paths instead \
of quoting code."""

CONTINUE_REPLY = "continue"
FINAL_REPLY = (
    "This is the last round: make no more tool calls. Write the complete answer, "
    "consolidating every round, and end with `STATUS: DONE`."
)
ARMS = ("A", "B", "C", "D", "E")
MEMORY_MARKERS = ("memory budget", "had to stop early", "memory limit", "out of memory")
# MetaMate cites as ``{{#metamate_citation DOCUMENT <url>}}``; plain ``file.py:12``
# anchors count too.
CITATION_RE = re.compile(
    r"metamate_citation\s+\w+\s+(\S+?)\}\}|([\w./-]+\.(?:py|cu|cuh|cpp|h|rst|md):\d+)"
)
STATUS_RE = re.compile(r"STATUS:\s*(CONTINUE|DONE)", re.IGNORECASE)


class _FixedScopeJudge:
    """Always the given scope, as ``reproduce_flow03 --scope-paths``."""

    def __init__(self, scope: CodeSearchScope) -> None:
        self.scope = scope

    @property
    def resume_identity(self) -> dict[str, Any]:
        return {"paths": list(self.scope.paths)}

    async def __call__(self, task: str) -> Any:
        return self.scope


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="metamate_prompt_ab",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--questions", required=True, help='JSON list of {"id", "prompt"}.')
    p.add_argument("--out", required=True, help="Output directory (created).")
    p.add_argument("--arms", default="A,B,C", help="Comma-separated variants.")
    p.add_argument("--limit", type=int, default=0, help="First N questions only.")
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument(
        "--budget", type=int, default=6, help="Variants B, D and E's tool calls."
    )
    p.add_argument(
        "--answer-words", type=int, default=2000, help="Variant D's answer cap."
    )
    p.add_argument(
        "--answer-findings", type=int, default=8, help="Variant E's answer cap."
    )
    p.add_argument(
        "--scope-paths", nargs="+", default=["fbcode/generative_recommenders"]
    )
    p.add_argument("--total-timeout", type=int, default=1800)
    p.add_argument("--idle-timeout", type=int, default=600)
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = p.parse_args()
    unknown = set(args.arms.split(",")) - set(ARMS)
    if unknown:
        p.error(f"unknown arms {sorted(unknown)}; choose from {ARMS}")
    return args


def _inferencer(args: argparse.Namespace, workspace: str, arm: str) -> Any:
    """The fan-out workers' MetaMate settings (``reproduce_flow03``), minus the
    leaf's default tool-call budget: each variant carries its own budget text.
    Variant C drives its own rounds, so MetaMate's clarification auto-continue is
    off there."""
    scope = CodeSearchScope(
        repo=Repo.FBSOURCE,
        corpora=(Corpus.FBCODE,),
        paths=tuple(args.scope_paths),
        rationale="fixed by --scope-paths",
    )
    return MetamateSDKInferencer(
        surface="VS_CODE",
        mode="AUTO",
        stream_type="SKYWALKER_PER_REQUEST",
        agent_name=None,
        cat_token=None,
        auto_continue=arm != "C",
        stream_total_timeout_seconds=args.total_timeout,
        idle_timeout_seconds=args.idle_timeout,
        max_retry=1,
        fallback_mode=FallbackMode.NEVER,
        code_scope_judge=_FixedScopeJudge(scope),
        tool_call_budget=None,
        debug_mode=True,
        workspace=InferencerWorkspace(root=workspace),
    )


def _classify(text: str, failed: bool) -> str:
    if failed and not text:
        return "killed"
    lower = text.lower()
    if any(m in lower for m in MEMORY_MARKERS) or len(text) < 1000:
        return "memory_giveup"
    if len(text) >= 2000:
        return "substantive"
    return "partial"


async def _call(inf: Any, prompt: str, **kwargs: Any) -> dict[str, Any]:
    started = time.monotonic()
    try:
        text = await inf.ainfer(prompt, prepared_input=True, **kwargs)
        return {
            "text": text if isinstance(text, str) else str(text),
            "elapsed_s": round(time.monotonic() - started, 1),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — every server outcome is a data point
        return {
            "text": "",
            "elapsed_s": round(time.monotonic() - started, 1),
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
        }


async def _run_arm(
    args: argparse.Namespace, question: dict[str, Any], arm: str
) -> dict[str, Any]:
    workspace = os.path.join(args.out, question["id"], arm)
    inf = _inferencer(args, workspace, arm)
    suffix = {
        "A": "",
        "B": tool_call_budget_directive(args.budget),
        "C": ROUNDS_SUFFIX,
        "D": tool_call_budget_directive(args.budget)
        + ANSWER_LENGTH_SUFFIX.format(words=args.answer_words),
        "E": tool_call_budget_directive(args.budget)
        + answer_format_directive(args.answer_findings),
    }[arm]
    rounds = []
    first = await _call(inf, question["prompt"] + suffix, new_session=True)
    rounds.append(first)
    accumulated = first["text"]
    final_round_text = first["text"]
    while arm == "C" and not first["error"] and len(rounds) < args.max_rounds:
        statuses = STATUS_RE.findall(final_round_text)
        if not statuses or statuses[-1].upper() != "CONTINUE":
            break
        reply = FINAL_REPLY if len(rounds) == args.max_rounds - 1 else CONTINUE_REPLY
        nxt = await _call(inf, reply, session_id=inf.active_session_id)
        rounds.append(nxt)
        if nxt["error"]:
            break
        text = nxt["text"]
        final_round_text = (
            text[len(accumulated) :] if text.startswith(accumulated) else text
        )
        accumulated = text
    failed = any(r["error"] for r in rounds)
    statuses = STATUS_RE.findall(final_round_text)
    return {
        "id": question["id"],
        "arm": arm,
        "outcome": _classify(final_round_text if arm == "C" else accumulated, failed),
        "rounds": [
            {
                "elapsed_s": r["elapsed_s"],
                "error": r["error"],
                "chars": len(r["text"]),
            }
            for r in rounds
        ],
        "killed_rounds": sum(1 for r in rounds if r["error"]),
        "final_status": statuses[-1].upper() if statuses else None,
        "final_chars": len(accumulated),
        "final_round_chars": len(final_round_text),
        "citations": len({a or b for a, b in CITATION_RE.findall(accumulated)}),
        "memory_marker": any(m in accumulated.lower() for m in MEMORY_MARKERS),
    }


async def _run_all(args: argparse.Namespace) -> None:
    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[: args.limit]
    arms = args.arms.split(",")
    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "results.jsonl")
    for i, question in enumerate(questions):
        log.info("question %d/%d %s", i + 1, len(questions), question["id"])
        results = await asyncio.gather(*(_run_arm(args, question, arm) for arm in arms))
        with open(results_path, "a", encoding="utf-8") as out:
            for result in results:
                out.write(json.dumps(result) + "\n")
                log.info(
                    "  %s %s rounds=%s chars=%d citations=%d",
                    result["arm"],
                    result["outcome"],
                    [(r["elapsed_s"], bool(r["error"])) for r in result["rounds"]],
                    result["final_chars"],
                    result["citations"],
                )


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_run_all(args))
    return 0
