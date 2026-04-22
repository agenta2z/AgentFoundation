#!/usr/bin/env python3
"""OpenClaw CLI Mode Examples.

Demonstrates ``OpenClawInferencer`` with ``mode="cli"`` — Docker/kubectl
subprocess execution via ``openclaw agent --local --json``.

Features demonstrated:
  1. Simple one-shot sync query
  2. Multi-turn session (in-memory within a run)
  3. Custom Docker/kubectl targeting
  4. Async inference

Prerequisites:
  - OpenClaw Docker running: ``cd atlassian-packages/openclaw && ./run.sh start``
  - Docker available in PATH

Run::

    python example_openclaw_cli_mode.py
    python example_openclaw_cli_mode.py --session-id my-session --thinking medium
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_workspace = Path(__file__).parents[9]
for _sub in ("AgentFoundation/src", "RichPythonUtils/src"):
    _p = _workspace / _sub
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── Imports ───────────────────────────────────────────────────────────────────
from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw import (
    OpenClawInferencer,
    OpenClawError,
    OpenClawNotFoundError,
)


def demo_simple_query(session_id: str, thinking: str | None) -> None:
    """Demo 1: Simple one-shot sync query."""
    print("\n" + "=" * 60)
    print("Demo 1: Simple One-Shot CLI Query")
    print("=" * 60)

    inf = OpenClawInferencer(
        mode="cli",
        session_id=session_id,
        thinking=thinking,
        timeout_seconds=120,
    )

    prompt = "In one sentence, what is your name and what can you do?"
    print(f"Prompt: {prompt!r}")
    print("Running (blocking)...")

    t0 = time.time()
    result = inf(prompt)
    elapsed = time.time() - t0

    print(f"\nResponse ({elapsed:.1f}s):")
    print(f"  {result}")
    print(f"  session_id: {inf.active_session_id}")


def demo_multi_turn(session_id: str) -> None:
    """Demo 2: Multi-turn session (in-memory, CLI mode)."""
    print("\n" + "=" * 60)
    print("Demo 2: Multi-Turn CLI Session")
    print("=" * 60)
    print("Note: CLI mode sessions are in-memory per process — no cross-run restore.")

    inf = OpenClawInferencer(
        mode="cli",
        session_id=session_id,
        timeout_seconds=120,
        max_retries=2,
        retry_delay=5.0,
    )

    turns = [
        "My secret number is 42. Remember it.",
        "What is my secret number?",
    ]

    for i, prompt in enumerate(turns, 1):
        print(f"\nTurn {i}: {prompt!r}")
        t0 = time.time()
        # Each --local call is a new process → session not preserved across calls
        # For real multi-turn, use mode="gateway"
        result = inf(prompt)
        elapsed = time.time() - t0
        print(f"  Response ({elapsed:.1f}s): {result[:200]}")


def demo_custom_targeting() -> None:
    """Demo 3: Custom Docker/kubectl targeting."""
    print("\n" + "=" * 60)
    print("Demo 3: Custom Docker/kubectl Targeting")
    print("=" * 60)

    inf = OpenClawInferencer(
        mode="cli",
        docker_container="openshell-cluster-openshell",
        kubectl_namespace="openshell",
        kubectl_pod="atlassian-openclaw-gateway",
        kubectl_container="agent",
        openclaw_config_path="/sandbox/.openclaw/openclaw.json",
        openclaw_state_dir="/sandbox/.openclaw",
        session_id="custom-target-demo",
        timeout_seconds=90,
    )

    prompt = "Reply with exactly: CUSTOM_TARGET_OK"
    print(f"Prompt: {prompt!r}")
    t0 = time.time()
    result = inf(prompt)
    print(f"Response ({time.time()-t0:.1f}s): {result}")


async def demo_async_cli(session_id: str) -> None:
    """Demo 4: Async CLI inference."""
    print("\n" + "=" * 60)
    print("Demo 4: Async CLI Inference")
    print("=" * 60)

    inf = OpenClawInferencer(
        mode="cli",
        session_id=session_id,
        timeout_seconds=120,
    )

    prompt = "List three benefits of async programming in one sentence each."
    print(f"Prompt: {prompt!r}")
    t0 = time.time()
    result = await inf.ainfer(prompt)
    elapsed = time.time() - t0
    print(f"\nAsync response ({elapsed:.1f}s):")
    print(f"  {result[:400]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenClaw CLI mode examples")
    parser.add_argument("--session-id", default="cli-demo", help="Session ID")
    parser.add_argument("--thinking", default=None,
                        choices=["off", "minimal", "low", "medium", "high", "xhigh"],
                        help="Thinking level")
    parser.add_argument("--demo", type=int, default=0,
                        help="Run specific demo (1-4), 0=all")
    args = parser.parse_args()

    demos = {
        1: lambda: demo_simple_query(args.session_id, args.thinking),
        2: lambda: demo_multi_turn(args.session_id),
        3: demo_custom_targeting,
        4: lambda: asyncio.run(demo_async_cli(args.session_id)),
    }

    run = [args.demo] if args.demo else list(demos.keys())

    for d in run:
        try:
            demos[d]()
        except OpenClawNotFoundError as e:
            print(f"\n❌ OpenClaw not found: {e}")
            print("   Make sure to run: cd atlassian-packages/openclaw && ./run.sh start")
            sys.exit(1)
        except OpenClawError as e:
            print(f"\n⚠️  OpenClaw error in demo {d}: {e}")
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break

    print("\n✅ CLI mode demos complete.")


if __name__ == "__main__":
    main()
