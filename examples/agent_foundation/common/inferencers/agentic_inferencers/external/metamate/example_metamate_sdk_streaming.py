#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""MetaMate SDK Inferencer — Streaming & Inference Modes Demo.

Demonstrates four inference modes using the MetaMate SDK:
  1. Async single call (``ainfer``) — full response at once
  2. Async streaming (``ainfer_streaming``) — text deltas via polling
  3. ``SDKInferencerResponse`` — structured result with session_id + tokens
  4. Sync single call (``__call__`` bridge) — for non-async code

Unlike the CLI inferencer (single-turn subprocess), the SDK speaks to the
MetaMate GraphQL backend directly via ``MetamateGraphQLClient`` and supports
multi-turn sessions. Streaming is poll-based: the SDK starts a conversation
via ``engine_start_v2`` and polls ``get_conversation_for_stream`` every few
seconds, yielding new text deltas as they accumulate.

Run (via buck — required because of ``//msl/metamate/cli:metamate_graphql`` dep):
    buck2 run @//mode/dbgo \\
      //_tony_dev/CoreProjects/AgentFoundation/examples/agent_foundation/common/inferencers/agentic_inferencers/external/metamate:example_metamate_sdk_streaming

Prerequisites:
    - MetaMate backend reachable
    - ``msl.metamate.cli.metamate_graphql`` available (Buck-only Meta dep —
      direct ``python3`` invocation will fail at import time)
"""

import argparse
import asyncio
import os
import sys
import time

# Auto-add AgentFoundation/src and RichPythonUtils/src to path (for direct
# Python — buck-run users don't hit this code because the package is on
# the binary's sys.path already).
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


def make_inferencer(args):
    """Create a fresh MetamateSDKInferencer."""
    from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
        MetamateSDKInferencer,
    )

    return MetamateSDKInferencer(
        agent_name=args.agent_name,
        total_timeout_seconds=args.total_timeout_seconds,
        idle_timeout_seconds=args.idle_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        auto_continue=True,
        max_continuations=5,
    )


# -- Demo 1: Async single call --------------------------------------------


async def demo_async_single(query: str, args) -> None:
    """ainfer: returns the full response text at once."""
    print("=" * 70)
    print("MODE 1: Async Single Call (ainfer)")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    inf = make_inferencer(args)

    start = time.time()
    result = await inf.ainfer(query)
    elapsed = time.time() - start

    output = str(result)
    print("Response:")
    print("-" * 60)
    print(output if len(output) < 2000 else output[:2000] + "...")
    print("-" * 60)
    print(f"Time:       {elapsed:.2f}s")
    print(f"Type:       {type(result).__name__}")
    print(f"Session ID: {inf.active_session_id}")
    print()


# -- Demo 2: Async streaming ----------------------------------------------


async def demo_async_streaming(query: str, args) -> None:
    """Async streaming: yields text deltas as MetaMate emits them."""
    print("=" * 70)
    print("MODE 2: Async Streaming (ainfer_streaming)")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    inf = make_inferencer(args)

    start = time.time()
    first_chunk_time = None
    char_count = 0
    chunk_count = 0

    print("Response (streaming):")
    print("-" * 60)

    async for chunk in inf.ainfer_streaming(query):
        if first_chunk_time is None:
            first_chunk_time = time.time()
        print(chunk, end="", flush=True)
        char_count += len(chunk)
        chunk_count += 1

    elapsed = time.time() - start
    ttfc = (first_chunk_time - start) if first_chunk_time else elapsed

    print()
    print("-" * 60)
    print(f"Time:              {elapsed:.2f}s")
    print(f"Time to 1st chunk: {ttfc:.2f}s")
    print(f"Chunks:            {chunk_count}")
    print(f"Characters:        {char_count}")
    print(f"Session ID:        {inf.active_session_id}")
    print()


# -- Demo 3: SDKInferencerResponse ----------------------------------------


async def demo_sdk_response(query: str, args) -> None:
    """SDKInferencerResponse: structured result with session_id and tokens."""
    print("=" * 70)
    print("MODE 3: SDKInferencerResponse (return_sdk_response=True)")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    inf = make_inferencer(args)

    start = time.time()
    response = await inf.ainfer(query, return_sdk_response=True)
    elapsed = time.time() - start

    content = getattr(response, "content", str(response))
    sid = getattr(response, "session_id", None)
    tokens = getattr(response, "tokens_received", None)

    print(f"Type:            {type(response).__name__}")
    print(f"Content:         {content[:300]}{'...' if len(content) > 300 else ''}")
    print(f"Session ID:      {sid}")
    print(f"Tokens received: {tokens}")
    print(f"Time:            {elapsed:.2f}s")
    print()


# -- Demo 4: Sync bridge --------------------------------------------------


def demo_sync_single(query: str, args) -> None:
    """Sync bridge: for non-async code (creates a fresh event loop)."""
    print("=" * 70)
    print("MODE 4: Sync Single Call (via _infer bridge)")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    inf = make_inferencer(args)

    start = time.time()
    result = inf(query)
    elapsed = time.time() - start

    output = str(result)
    print("Response:")
    print("-" * 60)
    print(output if len(output) < 2000 else output[:2000] + "...")
    print("-" * 60)
    print(f"Time:       {elapsed:.2f}s")
    print(f"Session ID: {inf.active_session_id}")
    print()


# -- Main -----------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="MetaMate SDK Inferencer — Streaming & Modes Demo"
    )
    parser.add_argument(
        "-q",
        "--query",
        default="Explain what a Python decorator is in 2 bullet points.",
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
        "--total-timeout-seconds",
        type=int,
        default=300,
        help="Total operation timeout (default: 300s).",
    )
    parser.add_argument(
        "--idle-timeout-seconds",
        type=int,
        default=180,
        help="Streaming idle timeout (default: 180s).",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=3.0,
        help="Poll interval for ``get_conversation_for_stream`` (default: 3.0).",
    )
    parser.add_argument(
        "-e",
        "--examples",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Number of examples to run: 1=async, 2=+streaming, 3=+sdk_response, 4=+sync",
    )
    args = parser.parse_args()

    print()
    print("MetaMate SDK Inferencer Demo")
    print(f"   Agent:           {args.agent_name or '(server default)'}")
    print(f"   Total timeout:   {args.total_timeout_seconds}s")
    print(f"   Idle timeout:    {args.idle_timeout_seconds}s")
    print(f"   Poll interval:   {args.poll_interval_seconds}s")
    print()

    # Demo 1: Async single call
    asyncio.run(demo_async_single(args.query, args))

    # Demo 2: Async streaming
    if args.examples >= 2:
        asyncio.run(demo_async_streaming(args.query, args))

    # Demo 3: SDKInferencerResponse
    if args.examples >= 3:
        asyncio.run(demo_sdk_response(args.query, args))

    # Demo 4: Sync bridge
    if args.examples >= 4:
        demo_sync_single(args.query, args)

    print("All demos complete!")


if __name__ == "__main__":
    main()
