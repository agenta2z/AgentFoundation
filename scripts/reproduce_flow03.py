"""Throwaway REPRODUCTION harness (NOT a fix) for the flow_03 Metamate OOM.

Re-drives flow_03's *exact* rendered InferenceInput (the 12,780 B prompt from
run ``research_propose_20260903_200405_046e8623``) through
``MetamateSDKInferencer``'s raw single-turn primitive ``_ainfer_streaming`` on a
FRESH conversation, and classifies the server outcome:

- ``HARD_OOM_500``          — engine_start_v2 raised an HTTP 500 from interngraph
                              (HHVM killed the request at its per-turn memory cap).
- ``MEMORY_GUARD_GIVEUP``   — the turn completed, but with the server's graceful
                              memory-headroom give-up message ("... MB of the
                              513 MB memory budget for a single turn ...").
- ``COMPLETED_DELIVERABLE`` — the turn completed with real research output
                              (OOM NOT reproduced on this turn).
- ``OTHER_EXCEPTION``       — some other failure (auth, timeout, dependency).

Why the raw primitive: ``_ainfer_streaming`` issues ONE ``engine_start_v2`` +
poll turn with flow_03's faithful params (surface=VS_CODE, mode=AUTO,
stream_type=SKYWALKER_PER_REQUEST, agent_name=None → DEFAULT agent). It
deliberately bypasses ``async_execute_with_retry`` / guardrail / fallback — that
machinery is client-side and was the ~52-min churn tail in the reference run,
not the OOM itself. This isolates the server's behavior on a clean turn.

Usage::

    buck2 run //_tony_dev/CoreProjects/AgentFoundation/scripts:reproduce_flow03 -- \
        --prompt-file /abs/path/to/InferenceInput/20260903_201926_aab7ece0.txt \
        --runs 1
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any

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


def _classify(text: str, exc: BaseException | None) -> str:
    if exc is not None:
        r = repr(exc)
        if any(m in r for m in _HARD_OOM_MARKERS):
            return "HARD_OOM_500"
        return f"OTHER_EXCEPTION:{type(exc).__name__}"
    if any(m in text for m in _GUARD_MARKERS):
        return "MEMORY_GUARD_GIVEUP"
    return "COMPLETED_DELIVERABLE"


def main() -> int:
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
    p.add_argument("--total-timeout", type=int, default=1800)
    p.add_argument("--idle-timeout", type=int, default=600)
    p.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable the code-scope judge (code_scope_judge=None). This is the "
        "A/B control arm: it reproduces the pre-judge behaviour, where the task "
        "reaches MetaMate with no scope directive and no read discipline.",
    )
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("reproduce_flow03")

    prompt = Path(args.prompt_file).expanduser().read_text()
    log.info(
        "Loaded prompt: %d bytes (%d chars) from %s",
        len(prompt.encode("utf-8")),
        len(prompt),
        args.prompt_file,
    )

    from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.metamate_sdk_inferencer import (
        MetamateSDKInferencer,
    )

    async def _one_run(run_idx: int) -> tuple[str, float, int]:
        # Faithful flow_03 config; fresh MetamateSDKInferencer (→ fresh
        # conversation, conversation_uuid=None) per run.
        inf = MetamateSDKInferencer(
            surface="VS_CODE",
            mode="AUTO",
            stream_type="SKYWALKER_PER_REQUEST",
            agent_name=None,
            cat_token=None,
            auto_continue=True,
            total_timeout_seconds=args.total_timeout,
            idle_timeout_seconds=args.idle_timeout,
        )

        # Record the scope the judge actually picked, without paying for a second
        # judge call, by wrapping the field the seam already invokes.
        used_scope: dict[str, Any] = {}
        if args.no_judge:
            inf.code_scope_judge = None
        else:
            _judge = inf.code_scope_judge

            async def _recording_judge(task: str) -> Any:
                scope = await _judge(task)
                used_scope["scope"] = scope
                return scope

            inf.code_scope_judge = _recording_judge

        # Surface a missing msl dependency early with a clear error.
        await inf.preflight()

        exc: BaseException | None = None
        t0 = time.monotonic()
        text = ""
        try:
            # ``_ainfer`` (NOT ``_ainfer_streaming``): the scope-judge seam lives in
            # ``_ainfer``, so driving the streaming primitive directly would bypass
            # the judge entirely and measure nothing. ``_ainfer`` delegates down to
            # the same streaming primitive, so the turn is otherwise identical.
            result = await inf._ainfer(prompt)
            text = result if isinstance(result, str) else str(result)
        except BaseException as e:  # noqa: BLE001 - probe wants every failure mode
            exc = e
        dt = time.monotonic() - t0
        verdict = _classify(text, exc)

        scope = used_scope.get("scope")
        log.info("=" * 72)
        log.info("RUN %d VERDICT: %s", run_idx, verdict)
        log.info(
            "  judge=%s  scope=%s  paths=%s",
            "off" if args.no_judge else "on",
            getattr(getattr(scope, "repo", None), "value", "n/a"),
            list(getattr(scope, "paths", ()) or []),
        )
        log.info("  elapsed=%.1fs  chars=%d", dt, len(text))
        if exc is not None:
            log.info("  exception=%r", exc)
            # For requests.HTTPError (the interngraph 500), dump status + body:
            # the body is the only client-visible hint at WHY the server 500'd
            # (memory kill vs auth crash vs other).
            resp = getattr(exc, "response", None)
            if resp is not None:
                try:
                    log.info("  http_status=%s", getattr(resp, "status_code", "?"))
                    body = getattr(resp, "text", "") or ""
                    log.info("  http_body[:2500]=\n%s", body[:2500])
                except Exception as _e:  # noqa: BLE001
                    log.info("  (could not read response body: %r)", _e)
        head = text[:2000]
        if head:
            log.info("  --- text head (first 2000 chars) ---\n%s", head)
        if len(text) > 4000:
            log.info("  --- text tail (last 2000 chars) ---\n%s", text[-2000:])
        log.info("=" * 72)
        return verdict, dt, len(text)

    async def _run_all() -> int:
        results: list[tuple[str, float, int]] = []
        for i in range(args.runs):
            log.info(
                "---- starting run %d/%d (fresh conversation) ----", i, args.runs - 1
            )
            results.append(await _one_run(i))
        log.info("############ SUMMARY ############")
        for i, (v, dt, n) in enumerate(results):
            log.info("run %d: %-22s elapsed=%.1fs chars=%d", i, v, dt, n)
        verdicts = [v for (v, _, _) in results]
        oom = [v for v in verdicts if v in ("HARD_OOM_500", "MEMORY_GUARD_GIVEUP")]
        log.info(
            "OOM_REPRODUCED=%s  (%d/%d turns hit a server memory limit)",
            bool(oom),
            len(oom),
            len(verdicts),
        )
        return 1 if oom else 0

    return asyncio.run(_run_all())


if __name__ == "__main__":
    sys.exit(main())
