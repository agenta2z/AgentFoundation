#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""MetaMate CLI Inferencer — Streaming vs Sync Demo.

Demonstrates three inference modes using the MetaMate CLI:
  1. Non-streaming (sync) — full response at once via ``inferencer(query)``
  2. Async streaming — line-by-line stdout via ``ainfer_streaming``
  3. Sync streaming — same as async but from synchronous code

The CLI inferencer launches ``buck run ...:query_metamate -- --query ...``
as a subprocess. The chunk granularity is one line per yield (subprocess
stdout). Deep-research mode (``--deep-research`` / ``agent_name=
SPACES_DEEP_RESEARCH_AGENT``) typically takes minutes; the demo uses
the regular agent by default.

Run (via buck — recommended):
    buck2 run @//mode/dbgo \\
      //_tony_dev/CoreProjects/AgentFoundation/examples/agent_foundation/common/inferencers/agentic_inferencers/external/metamate:example_metamate_cli_streaming

Run (direct Python — also works since the CLI inferencer subprocess uses buck):
    /usr/local/fbcode/platform010/bin/python3.12 \\
      examples/agent_foundation/common/inferencers/agentic_inferencers/external/metamate/example_metamate_cli_streaming.py

Prerequisites:
    - ``query_metamate`` Buck target reachable (built on demand by ``buck run``)
    - MetaMate backend reachable (the default API key works for internal users)
"""

import argparse
import asyncio
import os
import sys
import time

# Auto-add AgentFoundation/src and RichPythonUtils/src to path so direct
# ``python3 example_*.py`` invocation also works (mirrors devmate examples).
_script_dir = os.path.dirname(os.path.abspath(__file__))
_agent_foundation_root = os.path.normpath(
    os.path.join(_script_dir, "..", "..", "..", "..", "..", "..", "..")
)
_src_dir = os.path.join(_agent_foundation_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
_rich_utils_src = os.path.normpath(
    os.path.join(_agent_foundation_root, "..", "RichPythonUtils", "src")
)
if os.path.isdir(_rich_utils_src) and _rich_utils_src not in sys.path:
    sys.path.insert(0, _rich_utils_src)


def create_inferencer(args):
    """Create a MetamateCliInferencer with the given configuration."""
    from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
        MetamateCliInferencer,
    )

    return MetamateCliInferencer(
        agent_name=args.agent_name,
        deep_research=args.deep_research,
        timeout_seconds=args.timeout_seconds,
        idle_timeout_seconds=args.idle_timeout_seconds,
    )


# -- Demo 1: Non-streaming (sync with parsed metadata) --------------------


def demo_sync(inferencer, query: str) -> None:
    """Sync mode: blocks until full response returns."""
    print("=" * 70)
    print("MODE: Non-Streaming (Synchronous)")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    start = time.time()
    response = inferencer(query)
    elapsed = time.time() - start

    return_code = getattr(response, "return_code", None)
    output_text = getattr(response, "output", None) or str(response)

    print(f"Time:        {elapsed:.2f}s")
    print(f"Return code: {return_code}")
    print()
    print("Response:")
    print("-" * 60)
    body = output_text if output_text else "(empty)"
    display = body if len(body) < 2000 else body[:2000] + "..."
    print(display)
    print("-" * 60)
    print()


# -- Demo 2: Async streaming ----------------------------------------------


async def demo_async_streaming(inferencer, query: str) -> None:
    """Async streaming: prints subprocess stdout line-by-line as MetaMate emits."""
    print("=" * 70)
    print("MODE: Async Streaming")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    start = time.time()
    first_chunk_time = None
    char_count = 0
    line_count = 0

    print("Response (streaming):")
    print("-" * 60)

    async for chunk in inferencer.ainfer_streaming(query):
        if first_chunk_time is None:
            first_chunk_time = time.time()
        print(chunk, end="", flush=True)
        char_count += len(chunk)
        if "\n" in chunk:
            line_count += chunk.count("\n")

    elapsed = time.time() - start
    ttfc = (first_chunk_time - start) if first_chunk_time else elapsed

    print()
    print("-" * 60)
    print(f"Time:              {elapsed:.2f}s")
    print(f"Time to 1st chunk: {ttfc:.2f}s")
    print(f"Characters:        {char_count}")
    print(f"Lines:             {line_count}")
    print()


# -- Demo 3: Sync streaming -----------------------------------------------


def demo_sync_streaming(inferencer, query: str) -> None:
    """Sync streaming: same real-time output from synchronous code."""
    print("=" * 70)
    print("MODE: Sync Streaming")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    start = time.time()
    char_count = 0

    print("Response (streaming):")
    print("-" * 60)

    for chunk in inferencer.infer_streaming(query):
        print(chunk, end="", flush=True)
        char_count += len(chunk)

    elapsed = time.time() - start

    print()
    print("-" * 60)
    print(f"Time:        {elapsed:.2f}s")
    print(f"Characters:  {char_count}")
    print()


# -- Main -----------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="MetaMate CLI — Streaming vs Sync Demo"
    )
    parser.add_argument(
        "-q",
        "--query",
        default="What are the key differences between gRPC and Thrift in 3 bullet points?",
        help="Query to send to MetaMate",
    )
    parser.add_argument(
        "--agent-name",
        default=None,
        help=(
            "Optional MetaMate agent (e.g. METAMATE_GENERAL_AGENT, "
            "SPACES_DEEP_RESEARCH_AGENT). Default: server picks."
        ),
    )
    parser.add_argument(
        "--deep-research",
        action="store_true",
        default=False,
        help="Enable Deep Research mode (slower, more thorough output).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="Per-CLI-invocation timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--idle-timeout-seconds",
        type=int,
        default=300,
        help="Streaming idle timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "-e",
        "--examples",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Number of examples to run: 1=sync, 2=+async stream, 3=+sync stream",
    )
    args = parser.parse_args()

    if args.deep_research:
        # Bump timeouts: deep research can take several minutes.
        args.timeout_seconds = max(args.timeout_seconds, 600)
        args.idle_timeout_seconds = max(args.idle_timeout_seconds, 600)

    try:
        inferencer = create_inferencer(args)
    except Exception as e:
        print(f"Failed to create inferencer: {e}")
        return

    print()
    print("MetaMate CLI Inferencer Demo")
    print(f"   Agent:               {inferencer.agent_name or '(server default)'}")
    print(f"   Deep research:       {inferencer.deep_research}")
    print(f"   Timeout (seconds):   {inferencer.timeout_seconds}")
    print(f"   Idle timeout (s):    {inferencer.idle_timeout_seconds}")
    print()

    # Each mode uses a fresh inferencer — the CLI is single-turn so there
    # is no carry-over state between demos.
    demo_sync(create_inferencer(args), args.query)

    if args.examples >= 2:
        try:
            asyncio.run(demo_async_streaming(create_inferencer(args), args.query))
        except Exception as e:
            print(f"Async streaming demo failed: {e}")

    if args.examples >= 3:
        try:
            demo_sync_streaming(create_inferencer(args), args.query)
        except Exception as e:
            print(f"Sync streaming demo failed: {e}")

    print("All demos complete!")


if __name__ == "__main__":
    main()
