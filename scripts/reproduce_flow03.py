"""REPRODUCTION harness (NOT a fix) for the flow_03 Metamate OOM, with a fan-out arm.

Re-drives flow_03's *exact* rendered InferenceInput (the 12,780 B prompt from
run ``research_propose_20260903_200405_046e8623``) through a fresh
``MetamateSDKInferencer`` per run, and classifies the server outcome:

- ``HARD_OOM_500``          — engine_start_v2 raised an HTTP 500 from interngraph
                              (HHVM killed the request at its per-turn memory cap).
- ``MEMORY_GUARD_GIVEUP``   — the turn completed, but with the server's graceful
                              memory-headroom give-up message ("... MB of the
                              513 MB memory budget for a single turn ...").
- ``COMPLETED_DELIVERABLE`` — the turn completed with real research output
                              (OOM NOT reproduced on this turn).
- ``OTHER_EXCEPTION``       — some other failure (auth, timeout, dependency).
- ``NO_OUTPUT``             — (fan-out workers only) the worker left no output.

The prompt goes through the public ``ainfer`` with ``prepared_input=True``: it is
sent verbatim (never re-rendered) and the code-scope judge still applies. Each
Metamate request gets ``--attempts`` attempts (default 1) and no recovery
fallback, so every server outcome is observed instead of being retried away.

``--fanout`` gives the same inferencer the ``bta_inferencer`` template a
research_propose flow leaf gets from ``_params.metamate_fan_out``: a breakdown
splits the prompt into self-contained shards, each shard runs as its own fresh
Metamate request, and an aggregator merges the results. Every worker's outcome is
classified as well. The template is built by the research_propose tool's own
config loader (config, env prefix, template master version, config overrides),
so ``RESEARCH_PROPOSE__*`` env overrides apply as in production. Re-running with
the same ``--workspace-root`` resumes each interrupted fan-out from its saved
breakdown and worker checkpoints.

Usage::

    buck2 run //_tony_dev/CoreProjects/AgentFoundation/scripts:reproduce_flow03 -- \
        --prompt-file /abs/path/to/InferenceInput/20260903_201926_aab7ece0.txt \
        --runs 1 [--fanout] [--no-judge | --scope-paths PATH ...] \
        [--workspace-root DIR]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, NamedTuple

import agent_foundation.common.configs.registered_targets  # noqa: F401 — registers inferencer aliases
import agent_foundation.resources.tools.task as task_tool
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.code_scope_judge import (
    CodeSearchScope,
    Corpus,
    judge_code_scope,
    Repo,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.metamate_sdk_inferencer import (
    MetamateSDKInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import BTA_INFERENCER_SLOT
from agent_foundation.common.inferencers.inferencer_workspace import (
    DEFAULT_OUTPUT_FILENAME,
    indexed_child_name,
    InferencerWorkspace,
    resolve_canonical_output_path,
)
from agent_foundation.common.inferencers.run_context.resume_identity import (
    qualified_name,
)
from agent_foundation.resources.tools.registry import load_tool
from rich_python_utils.common_utils.function_helper import FallbackMode
from rich_python_utils.config_utils import instantiate, load_config

# Server-side memory-guard give-up markers (from TMetamateGPTEngine's
# genMemoryHeadroomExitMessage format string).
_GUARD_MARKERS: tuple[str, ...] = (
    "memory budget for a single turn",
    "had to stop early",
    "MB of the",
    "start a new conversation to reset",
)

# Hard-OOM markers: HHVM kills the request at the cap → bare HTTP 500 from
# interngraph, surfaced here as a raised exception.
_HARD_OOM_MARKERS: tuple[str, ...] = (
    "500 Server Error",
    "Internal Server Error",
    "interngraph",
)

_OOM_VERDICTS: frozenset[str] = frozenset({"HARD_OOM_500", "MEMORY_GUARD_GIVEUP"})

log: logging.Logger = logging.getLogger("reproduce_flow03")


class _RunResult(NamedTuple):
    verdict: str
    elapsed: float
    chars: int
    workers: dict[str, str]


def _classify(text: str, exc: BaseException | None) -> str:
    if exc is not None:
        r = repr(exc)
        if any(m in r for m in _HARD_OOM_MARKERS):
            return "HARD_OOM_500"
        return f"OTHER_EXCEPTION:{type(exc).__name__}"
    if any(m in text for m in _GUARD_MARKERS):
        return "MEMORY_GUARD_GIVEUP"
    return "COMPLETED_DELIVERABLE"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="reproduce_flow03",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--prompt-file",
        required=True,
        help="Path to flow_03's rendered InferenceInput (the 12,780 B prompt).",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=1,
        help="How many fresh-conversation turns to fire (default 1).",
    )
    p.add_argument(
        "--total-timeout",
        type=int,
        default=1800,
        help="Wall-clock cap for one Metamate streaming poll loop, in seconds.",
    )
    p.add_argument("--idle-timeout", type=int, default=600)
    p.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Attempts per Metamate request, with no recovery fallback (default 1).",
    )
    p.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable the code-scope judge (code_scope_judge=None). This is the "
        "A/B control arm: it reproduces the pre-judge behaviour, where the task "
        "reaches MetaMate with no scope directive and no read discipline.",
    )
    p.add_argument(
        "--scope-paths",
        nargs="+",
        metavar="PATH",
        help="Skip the judge and scope every Metamate request to these paths, "
        "with the repo and corpus the judge picks for this prompt (fbsource / "
        "fbcode); e.g. fbcode/generative_recommenders. The A/B arm for search "
        "width: only the directive's 'Prefer paths under' line changes.",
    )
    p.add_argument(
        "--fanout",
        action="store_true",
        help="Fan the prompt out through research_propose's Metamate "
        "bta_inferencer template (breakdown -> one Metamate request per shard -> "
        "aggregate) instead of sending it as one request.",
    )
    p.add_argument(
        "--workspace-root",
        help="Directory for the run workspaces (default: a new temp dir). "
        "Re-running --fanout with the same root resumes each run_<i> from its "
        "saved fan-out checkpoints: completed workers are loaded, not re-run.",
    )
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = p.parse_args()
    if args.no_judge and args.scope_paths:
        p.error("--no-judge and --scope-paths are exclusive")
    return args


def _fan_out_template(workspace_root: str) -> Any:
    """Return the ``bta_inferencer`` template of a research_propose flow leaf
    enabled with ``_params.metamate_fan_out``, loaded the way the tool loads it."""
    derived_from = load_tool("research_propose").derived_from
    if derived_from is None:
        raise ValueError("research_propose's tool.json declares no derived_from")
    defaults = derived_from["defaults"]
    config_path = Path(task_tool.__file__).parent / "configs" / defaults["config"]
    cfg = load_config(
        f"{config_path}.yaml",
        overrides={
            "_params.workspace_root": workspace_root,
            "_template_master_version": defaults["template_master_version"],
            "_params.flow_bta_inferencers": ["${_params.metamate_fan_out}"],
        },
        env_prefix=defaults["env_prefix"],
        config_defaults=dict(defaults["config_overrides"]),
    )
    planner = instantiate(cfg)
    flows = planner.base_inferencer.worker_inferencers().base_inferencer
    # `_repeat_` pads the one-element list to every flow, so flow 0 carries the
    # template regardless of which flow index runs Metamate.
    return flows.flow_configs[0]["initial_inferencer"].bta_inferencer


class _RecordingJudge:
    """The default code-scope judge, recording every scope it picks (one per
    Metamate request) without paying for a second judge call."""

    # Recording changes no scope the judge picks, so the resume of a run checks
    # the default judge's name.
    resume_identity: str = qualified_name(judge_code_scope)

    def __init__(self, scopes: list[Any]) -> None:
        self.scopes = scopes

    async def __call__(self, task: str) -> Any:
        scope = await judge_code_scope(task)
        self.scopes.append(scope)
        return scope


class _FixedScopeJudge:
    """A judge that always picks ``scope`` (``--scope-paths``) without a judge
    call, recording each pick like ``_RecordingJudge``."""

    def __init__(self, scope: CodeSearchScope, scopes: list[Any]) -> None:
        self.scope = scope
        self.scopes = scopes

    @property
    def resume_identity(self) -> dict[str, Any]:
        return {
            "repo": self.scope.repo.value,
            "corpora": [c.value for c in self.scope.corpora],
            "paths": list(self.scope.paths),
        }

    async def __call__(self, task: str) -> Any:
        self.scopes.append(self.scope)
        return self.scope


def _make_judge(args: argparse.Namespace, scopes: list[Any]) -> Any:
    if args.no_judge:
        return None
    if args.scope_paths:
        scope = CodeSearchScope(
            repo=Repo.FBSOURCE,
            corpora=(Corpus.FBCODE,),
            paths=tuple(args.scope_paths),
            rationale="fixed by --scope-paths",
        )
        return _FixedScopeJudge(scope, scopes)
    return _RecordingJudge(scopes)


def _build_inferencer(
    args: argparse.Namespace, workspace_root: str, judge: Any, template: Any
) -> Any:
    """Faithful flow_03 Metamate config on a fresh conversation; fan-out workers
    are fresh copies of it, so they share every setting here."""
    return MetamateSDKInferencer(
        surface="VS_CODE",
        mode="AUTO",
        stream_type="SKYWALKER_PER_REQUEST",
        agent_name=None,
        cat_token=None,
        auto_continue=True,
        stream_total_timeout_seconds=args.total_timeout,
        idle_timeout_seconds=args.idle_timeout,
        max_retry=args.attempts,
        fallback_mode=FallbackMode.NEVER,
        code_scope_judge=judge,
        debug_mode=True,
        workspace=InferencerWorkspace(root=workspace_root),
        output_path=DEFAULT_OUTPUT_FILENAME,
        bta_inferencer=template,
    )


def _worker_verdicts(workspace_root: str) -> dict[str, str]:
    """Classify each fan-out worker's canonical output, keyed by worker child name."""
    bta_ws = InferencerWorkspace(root=workspace_root).child(BTA_INFERENCER_SLOT)
    children = (
        set(os.listdir(bta_ws.children_dir))
        if os.path.isdir(bta_ws.children_dir)
        else set()
    )
    names = [indexed_child_name("worker", i) for i in range(len(children))]
    verdicts: dict[str, str] = {}
    for name in (n for n in names if n in children):
        path = resolve_canonical_output_path(
            bta_ws.child(name),
            filename=DEFAULT_OUTPUT_FILENAME,
            deliverables_fallback="none",
        )
        verdicts[name] = (
            _classify(Path(path).read_text(), None) if path else "NO_OUTPUT"
        )
    return verdicts


def _describe_scope(scope: Any) -> str:
    repo = getattr(getattr(scope, "repo", None), "value", "n/a")
    return f"{repo}:{list(getattr(scope, 'paths', ()) or [])}"


def _log_exception(exc: BaseException) -> None:
    log.info("  exception=%r", exc)
    # For requests.HTTPError (the interngraph 500), dump status + body:
    # the body is the only client-visible hint at WHY the server 500'd
    # (memory kill vs auth crash vs other).
    resp = getattr(exc, "response", None)
    if resp is None:
        return
    try:
        log.info("  http_status=%s", getattr(resp, "status_code", "?"))
        body = getattr(resp, "text", "") or ""
        log.info("  http_body[:2500]=\n%s", body[:2500])
    except Exception as _e:  # noqa: BLE001
        log.info("  (could not read response body: %r)", _e)


def _log_run(
    run_idx: int,
    result: _RunResult,
    text: str,
    exc: BaseException | None,
    judge_note: str,
) -> None:
    log.info("=" * 72)
    log.info("RUN %d VERDICT: %s", run_idx, result.verdict)
    log.info("  %s", judge_note)
    log.info("  elapsed=%.1fs  chars=%d", result.elapsed, result.chars)
    for name, verdict in result.workers.items():
        log.info("  worker %s: %s", name, verdict)
    if exc is not None:
        _log_exception(exc)
    head = text[:2000]
    if head:
        log.info("  --- text head (first 2000 chars) ---\n%s", head)
    if len(text) > 4000:
        log.info("  --- text tail (last 2000 chars) ---\n%s", text[-2000:])
    log.info("=" * 72)


async def _one_run(
    args: argparse.Namespace, prompt: str, template: Any, workspace_root: str
) -> tuple[_RunResult, str, BaseException | None, str]:
    scopes: list[Any] = []
    judge = _make_judge(args, scopes)
    inf = _build_inferencer(args, workspace_root, judge, template)

    # Surface a missing msl dependency early with a clear error.
    await inf.preflight()

    exc: BaseException | None = None
    t0 = time.monotonic()
    text = ""
    try:
        result = await inf.ainfer(prompt, prepared_input=True)
        text = result if isinstance(result, str) else str(result)
    except BaseException as e:  # noqa: BLE001 - probe wants every failure mode
        exc = e
    run = _RunResult(
        verdict=_classify(text, exc),
        elapsed=time.monotonic() - t0,
        chars=len(text),
        workers=_worker_verdicts(workspace_root) if template is not None else {},
    )
    if args.no_judge:
        judge_note = "judge=off"
    else:
        kind = "fixed" if args.scope_paths else "on"
        judge_note = f"judge={kind}  scopes={[_describe_scope(s) for s in scopes]}"
    return run, text, exc, judge_note


def _summarize(results: list[_RunResult], fanout: bool) -> int:
    log.info("############ SUMMARY ############")
    for i, r in enumerate(results):
        log.info(
            "run %d: %-22s elapsed=%.1fs chars=%d", i, r.verdict, r.elapsed, r.chars
        )
    oom = [r for r in results if r.verdict in _OOM_VERDICTS]
    log.info(
        "OOM_REPRODUCED=%s  (%d/%d turns hit a server memory limit)",
        bool(oom),
        len(oom),
        len(results),
    )
    if fanout:
        workers = [v for r in results for v in r.workers.values()]
        log.info(
            "WORKER_OOM=%d/%d  (fan-out worker requests that hit a server memory "
            "limit; NO_OUTPUT=%d)",
            sum(v in _OOM_VERDICTS for v in workers),
            len(workers),
            workers.count("NO_OUTPUT"),
        )
    return 1 if oom else 0


async def _run_all(args: argparse.Namespace, prompt: str, root: str) -> int:
    template = _fan_out_template(os.path.join(root, "planner")) if args.fanout else None
    results: list[_RunResult] = []
    for i in range(args.runs):
        log.info("---- starting run %d/%d (fresh conversation) ----", i, args.runs - 1)
        run, text, exc, judge_note = await _one_run(
            args, prompt, template, os.path.join(root, f"run_{i}")
        )
        _log_run(i, run, text, exc, judge_note)
        results.append(run)
    return _summarize(results, args.fanout)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    prompt = Path(args.prompt_file).expanduser().read_text()
    log.info(
        "Loaded prompt: %d bytes (%d chars) from %s",
        len(prompt.encode("utf-8")),
        len(prompt),
        args.prompt_file,
    )
    if args.workspace_root:
        root = os.path.abspath(os.path.expanduser(args.workspace_root))
        os.makedirs(root, exist_ok=True)
    else:
        root = tempfile.mkdtemp(prefix="reproduce_flow03_")
    log.info("Workspaces under %s (fanout=%s)", root, args.fanout)
    return asyncio.run(_run_all(args, prompt, root))


if __name__ == "__main__":
    sys.exit(main())
