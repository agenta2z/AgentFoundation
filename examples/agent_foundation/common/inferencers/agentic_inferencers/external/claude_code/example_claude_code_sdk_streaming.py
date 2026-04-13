#!/usr/bin/env python3
"""Claude Code SDK Inferencer -- Streaming & Inference Modes Demo.

Demonstrates four inference modes using the Claude Code SDK:
  1. Async single call (ainfer) -- full response at once via async with
  2. Async streaming (ainfer_streaming) -- real-time token output
  3. SDKInferencerResponse -- structured response with session_id + tool_uses
  4. Sync single call (_infer bridge) -- for non-async code

Run:
    python examples/agent_foundation/common/inferencers/agentic_inferencers/external/claude_code/example_claude_code_sdk_streaming.py

    # Customize:
    python ...example_claude_code_sdk_streaming.py -q "Explain recursion" -e 2

Prerequisites:
    - claude-agent-sdk package installed (pip install claude-agent-sdk)
    - Claude Code CLI authenticated on this machine
    - If running inside Claude Code (nested), set env: CLAUDECODE= (unset it)
"""

import argparse
import asyncio
import os
import sys
import time

# Auto-add AgentFoundation/src and RichPythonUtils/src to path
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


def _prepare_env():
    """Unset CLAUDECODE to bypass the nested-session guard.

    The Claude Code binary refuses to start when it detects CLAUDECODE=1 in
    the environment (nesting guard). Safe to clear for SDK subprocess usage.

    Note: ANTHROPIC_API_KEY is handled automatically by ClaudeCodeInferencer's
    ``prefer_subscription=True`` default — no need to touch os.environ.
    """
    if os.environ.pop("CLAUDECODE", None):
        print("  (Cleared CLAUDECODE env var to allow nested SDK usage)")


# -- Demo 1: Async single call (ainfer) --------------------------------------

async def demo_async_single(query: str, root_folder: str) -> None:
    """Async context manager + ainfer: full response at once."""
    print("=" * 70)
    print("MODE 1: Async Single Call (ainfer)")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
        ClaudeCodeInferencer,
    )

    start = time.time()

    async with ClaudeCodeInferencer(root_folder=root_folder, allowed_tools=[]) as inf:
        result = await inf.ainfer(query)

    elapsed = time.time() - start

    print("Response:")
    print("-" * 60)
    print(result)
    print("-" * 60)
    print(f"Time:       {elapsed:.2f}s")
    print(f"Type:       {type(result).__name__}")
    print()


# -- Demo 2: Async streaming (ainfer_streaming) -------------------------------

async def demo_async_streaming(query: str, root_folder: str) -> None:
    """Async streaming: prints text chunk-by-chunk as Claude generates."""
    print("=" * 70)
    print("MODE 2: Async Streaming (ainfer_streaming)")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
        ClaudeCodeInferencer,
    )

    start = time.time()
    first_chunk_time = None
    char_count = 0
    chunk_count = 0

    print("Response (streaming):")
    print("-" * 60)

    async with ClaudeCodeInferencer(root_folder=root_folder, allowed_tools=[]) as inf:
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
    print(f"Time:             {elapsed:.2f}s")
    print(f"Time to 1st chunk: {ttfc:.2f}s")
    print(f"Chunks:           {chunk_count}")
    print(f"Characters:       {char_count}")
    print()


# -- Demo 3: SDKInferencerResponse -------------------------------------------

async def demo_sdk_response(query: str, root_folder: str) -> None:
    """SDKInferencerResponse: structured result with metadata."""
    print("=" * 70)
    print("MODE 3: SDKInferencerResponse (return_sdk_response=True)")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
        ClaudeCodeInferencer,
    )

    start = time.time()

    async with ClaudeCodeInferencer(root_folder=root_folder, allowed_tools=[]) as inf:
        response = await inf.ainfer(query, return_sdk_response=True)

    elapsed = time.time() - start

    print(f"Type:       {type(response).__name__}")
    print(f"Content:    {response.content[:200]}{'...' if len(response.content) > 200 else ''}")
    print(f"Session ID: {response.session_id}")
    print(f"Tool uses:  {response.tool_uses}")
    print(f"str():      {str(response)[:80]}...")
    print(f"Time:       {elapsed:.2f}s")
    print()
    print("NOTE: SDKInferencerResponse.content returns the full text.")
    print("      str(response) also returns the text (for DualInferencer compat).")
    print("      session_id + tool_uses provide metadata not available in plain ainfer().")
    print()


# -- Demo 4: Sync bridge (_infer) --------------------------------------------

def demo_sync_single(query: str, root_folder: str) -> None:
    """Sync _infer bridge: for non-async code (pays reconnect cost per call)."""
    print("=" * 70)
    print("MODE 4: Sync Single Call (via _infer bridge)")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
        ClaudeCodeInferencer,
    )

    inferencer = ClaudeCodeInferencer(root_folder=root_folder, allowed_tools=[])

    start = time.time()
    result = inferencer(query)
    elapsed = time.time() - start

    print("Response:")
    print("-" * 60)
    print(result)
    print("-" * 60)
    print(f"Time:       {elapsed:.2f}s")
    print(f"Type:       {type(result).__name__}")
    print()
    print("NOTE: Sync bridge uses asyncio.run() internally. Each call starts a")
    print("      fresh event loop, so the SDK reconnects every time. For multi-call")
    print("      usage, prefer the async with pattern (Mode 1).")
    print()


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Claude Code SDK Inferencer -- Streaming & Modes Demo"
    )
    parser.add_argument(
        "-q", "--query",
        default="Explain what a Python decorator is in 2 bullet points.",
        help="Query to send to Claude",
    )
    parser.add_argument(
        "-r", "--root-folder",
        default=os.path.expanduser("~"),
        help="Working directory for Claude Code agent (default: home dir)",
    )
    parser.add_argument(
        "-e", "--examples",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Number of examples to run: 1=async, 2=+streaming, 3=+sdk_response, 4=+sync",
    )
    args = parser.parse_args()

    _prepare_env()

    print()
    print("Claude Code SDK Inferencer Demo")
    print(f"   Root folder: {args.root_folder}")
    print()

    # Demo 1: Async single call
    asyncio.run(demo_async_single(args.query, args.root_folder))

    # Demo 2: Async streaming
    if args.examples >= 2:
        asyncio.run(demo_async_streaming(args.query, args.root_folder))

    # Demo 3: SDKInferencerResponse
    if args.examples >= 3:
        asyncio.run(demo_sdk_response(args.query, args.root_folder))

    # Demo 4: Sync bridge
    if args.examples >= 4:
        demo_sync_single(args.query, args.root_folder)

    print("All demos complete!")


if __name__ == "__main__":
    main()
