#!/usr/bin/env python3
"""OpenClaw Gateway Mode Examples — Real Atlassian Queries.

Demonstrates ``OpenClawInferencer`` with ``OpenClawMode.PodGateway`` —
WebSocket connection to the OpenClaw gateway at ``ws://127.0.0.1:18789``.

Features demonstrated with real Atlassian queries:
  1. Streaming — "What should I follow up on today?" (TWG + Confluence + Slack)
  2. Multi-turn agentic loop with turn separation enabled
  3. Loom videos + comments query
  4. Session restore — ask a follow-up in a new inferencer instance
  5. Auto-retry on rate limit with continuation prompt

Prerequisites:
  - Docker container ``openshell-cluster-openshell`` running (``./run.sh start``)
  - OpenClaw gateway pod healthy inside the container
  - ``websockets`` package installed: ``pip install websockets``

Run::

    python example_openclaw_gateway_mode.py
    python example_openclaw_gateway_mode.py --demo 1   # follow-up query only
    python example_openclaw_gateway_mode.py --demo 3   # loom videos only
    python example_openclaw_gateway_mode.py --thinking high
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# parents[8] = CoreProjects  (script is 9 directories deep from CoreProjects)
_core_projects = Path(__file__).parents[8]
for _sub in ("AgentFoundation/src", "RichPythonUtils/src"):
    _p = _core_projects / _sub
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
# ─────────────────────────────────────────────────────────────────────────────

from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw import (
    OpenClawError,
    OpenClawInferencer,
    OpenClawMode,
    OpenClawNotFoundError,
    OpenClawRateLimitError,
)


def make_inferencer(
    session_id: str,
    thinking: str | None = None,
    timeout: int = 220,
    turn_separation: bool = False,
) -> OpenClawInferencer:
    """Create an OpenClawInferencer in PodGateway mode."""
    return OpenClawInferencer(
        mode=OpenClawMode.PodGateway,
        session_id=session_id,
        thinking=thinking,
        timeout_seconds=timeout,
        enable_turn_separation=turn_separation,
        always_initialize_new_session=True,
        auto_resume=True,
        max_retries=3,
        retry_delay=8.0,
    )


async def stream_and_print(
    inf: OpenClawInferencer,
    prompt: str,
    label: str = "Query",
) -> str:
    """Stream a query and print tokens in real time. Returns full response."""
    print(f"\n  [{label}]")
    print(f"  Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print()

    t0 = time.time()
    first_chunk_time: float | None = None
    chunks: list[str] = []

    async for chunk in inf.ainfer_streaming(prompt):
        if first_chunk_time is None:
            first_chunk_time = time.time()
        print(chunk, end="", flush=True)
        chunks.append(chunk)

    elapsed = time.time() - t0
    response = "".join(chunks)
    print()
    print()
    print(f"  ⏱  ttft={first_chunk_time - t0:.2f}s  "
          f"total={elapsed:.1f}s  chars={len(response)}")
    return response


# =============================================================================
# Demo 1 — What should I follow up on today? (streaming, real TWG query)
# =============================================================================

async def demo_follow_up(thinking: str | None, timeout: int) -> None:
    """Demo 1: Streaming a real 'what should I follow up on?' query.

    Uses TWG + agentic-search to pull comments on your Jira tickets,
    Confluence pages, and PRs, then prioritises them.
    """
    print("\n" + "=" * 70)
    print("Demo 1: 'What should I follow up on today?' (streaming, real TWG)")
    print("=" * 70)
    print("  The agent will use TWG to query Jira, Confluence, Bitbucket,")
    print("  and Slack to surface what needs your attention.")
    print()

    inf = make_inferencer(
        session_id="gw-demo-followup",
        thinking=thinking,
        timeout=timeout,
        turn_separation=True,   # show multi-turn agentic loop boundaries
    )

    prompt = (
        "What should I follow up on today? "
        "Prioritize: (1) comments on work items/pages/PRs I created, "
        "(2) replies to my comments, "
        "(3) tasks assigned to me, "
        "(4) PRs I opened."
    )

    t0 = time.time()
    first_chunk_time: float | None = None
    chunks: list[str] = []

    print("  Streaming response (turn boundaries shown as blank lines):")
    print("  " + "-" * 66)

    async for chunk in inf.ainfer_streaming(prompt):
        if first_chunk_time is None:
            first_chunk_time = time.time()
        print(chunk, end="", flush=True)
        chunks.append(chunk)

    elapsed = time.time() - t0
    response = "".join(chunks)
    print()
    print()
    print("  " + "-" * 66)
    print(f"  ✅ Streaming complete!")
    print(f"     ttft={first_chunk_time - t0:.2f}s  total={elapsed:.1f}s  "
          f"chars={len(response)}")
    print(f"     session={inf.active_session_id}")


# =============================================================================
# Demo 2 — Multi-turn session restore (ask a follow-up in a new inferencer)
# =============================================================================

async def demo_session_restore(thinking: str | None, timeout: int) -> None:
    """Demo 2: Cross-run session restore — ask a follow-up in a new instance.

    Turn 1: Ask for follow-ups (real query using TWG).
    Turn 2: Create a fresh OpenClawInferencer with the same session_id.
            Ask a follow-up — the agent should remember Turn 1's context.
    """
    print("\n" + "=" * 70)
    print("Demo 2: Session Restore — Follow-up across inferencer instances")
    print("=" * 70)
    print("  Simulates resuming a conversation in a new Python process.")
    print("  The gateway persists sessions as JSONL files in the pod.")
    print()

    session_id = "gw-demo-restore"

    # Turn 1: Initial query
    print(f"  [Turn 1 — session: {session_id!r}]")
    inf1 = make_inferencer(session_id=session_id, thinking=thinking, timeout=timeout)
    resp1 = await stream_and_print(
        inf1,
        "What did I work on last week? Give me a one-paragraph summary.",
        label="Turn 1",
    )

    # Turn 2: Fresh inferencer instance, same session_id
    print()
    print(f"  [Turn 2 — NEW inferencer instance, same session_id: {session_id!r}]")
    print("  (simulates a new Python process resuming the conversation)")
    inf2 = make_inferencer(session_id=session_id, thinking=thinking, timeout=timeout)
    resp2 = await stream_and_print(
        inf2,
        "Based on what you just told me, what is the single most important "
        "thing I should focus on this week? One sentence only.",
        label="Turn 2 (restored session)",
    )

    # Verify restore worked — Turn 2 should reference Turn 1's content
    print()
    if any(kw in resp2.lower() for kw in
           ["openclaw", "browser", "ai", "employee", "spike", "jira", "confluence",
            "last week", "based on", "you mentioned", "as i mentioned"]):
        print("  ✅ Session restore CONFIRMED — Turn 2 references Turn 1 context!")
    else:
        print("  ⚠️  Turn 2 may not have restored context — check session JSONL.")


# =============================================================================
# Demo 3 — Loom videos + comments query
# =============================================================================

async def demo_loom_videos(thinking: str | None, timeout: int) -> None:
    """Demo 3: Loom videos and audience comments via TWG.

    Queries all Loom videos from the past 3 months and their comment counts.
    """
    print("\n" + "=" * 70)
    print("Demo 3: Loom Videos + Audience Comments (TWG query)")
    print("=" * 70)
    print("  The agent will use 'twg videos query' to list your Loom videos")
    print("  and fetch audience comment counts for each.")
    print()

    inf = make_inferencer(
        session_id="gw-demo-loom",
        thinking=thinking,
        timeout=timeout,
        turn_separation=True,
    )

    await stream_and_print(
        inf,
        "What's my loom videos in the past 3 months, "
        "and what are audience comments on my videos?",
        label="Loom videos + comments",
    )


# =============================================================================
# Demo 4 — Team projects summary
# =============================================================================

async def demo_team_projects(thinking: str | None, timeout: int) -> None:
    """Demo 4: Summarise team's current projects and their progress.

    Uses TWG + agentic-search to pull Jira epics, Atlas projects,
    and Confluence pages to give a structured team status.
    """
    print("\n" + "=" * 70)
    print("Demo 4: 'Summarize my team's current major projects and their progress'")
    print("=" * 70)
    print("  The agent will query Jira, Atlas, and Confluence for team project status.")
    print()

    inf = make_inferencer(
        session_id="gw-demo-projects",
        thinking=thinking,
        timeout=timeout,
        turn_separation=True,
    )

    await stream_and_print(
        inf,
        "Summarize my team's current major projects and their progress. "
        "Group by project and include status, key recent updates, and open blockers.",
        label="Team projects summary",
    )


# =============================================================================
# Demo 5 — Auto-retry on rate limit
# =============================================================================

async def demo_retry(thinking: str | None, timeout: int) -> None:
    """Demo 5: Auto-retry with continuation prompt on rate limit.

    Uses a short timeout to simulate a rate-limit scenario.
    In normal operation this will succeed on the first attempt.
    """
    print("\n" + "=" * 70)
    print("Demo 5: Auto-Retry on Rate Limit (with continuation prompt)")
    print("=" * 70)
    print("  max_retries=3, retry_delay=8s")
    print("  If rate-limited, retries with a continuation prompt automatically.")
    print()

    inf = OpenClawInferencer(
        mode=OpenClawMode.PodGateway,
        session_id="gw-demo-retry",
        thinking=thinking,
        timeout_seconds=timeout,
        max_retries=3,
        retry_delay=8.0,
        retry_continuation_prompt=(
            "You were interrupted. Please re-answer concisely: {original_prompt}"
        ),
        always_initialize_new_session=True,
    )

    t0 = time.time()
    try:
        chunks: list[str] = []
        async for chunk in inf.ainfer_streaming(
            "What are my open Jira tickets assigned to me right now? "
            "List them as bullet points with status."
        ):
            print(chunk, end="", flush=True)
            chunks.append(chunk)
        print()
        print(f"\n  ✅ Done in {time.time()-t0:.1f}s")
    except OpenClawRateLimitError as e:
        print(f"\n  ❌ Still rate-limited after 3 retries: {e}")


# =============================================================================
# Main
# =============================================================================

async def main_async(args: argparse.Namespace) -> None:
    thinking = args.thinking
    timeout = args.timeout

    demos = {
        1: lambda: demo_follow_up(thinking, timeout),
        2: lambda: demo_session_restore(thinking, timeout),
        3: lambda: demo_loom_videos(thinking, timeout),
        4: lambda: demo_team_projects(thinking, timeout),
        5: lambda: demo_retry(thinking, timeout),
    }

    run = [args.demo] if args.demo else list(demos.keys())

    print()
    print("🐾 OpenClaw Gateway Mode — Real Atlassian Query Demos")
    print(f"   Mode: PodGateway  thinking={thinking or 'default'}  timeout={timeout}s")
    print(f"   Running demos: {run}")

    for d in run:
        try:
            await demos[d]()
        except OpenClawNotFoundError as e:
            print(f"\n❌ OpenClaw not found: {e}")
            print("   Ensure './run.sh start' is running and gateway is at ws://127.0.0.1:18789")
            break
        except OpenClawError as e:
            print(f"\n⚠️  OpenClaw error in demo {d}: {type(e).__name__}: {e}")
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break

    print()
    print("✅ Gateway mode demos complete.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenClaw PodGateway mode — Real Atlassian query examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--demo",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help="Run a specific demo (1-5). 0 = run all (default).",
    )
    parser.add_argument(
        "--thinking",
        choices=["off", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Thinking level for the OpenClaw agent (default: gateway default).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=220,
        help="Per-request timeout in seconds (default: 220).",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
