#!/usr/bin/env python3
"""Devmate SDK Inferencer -- Streaming & Inference Modes Demo.

Demonstrates four inference modes using the Devmate SDK:
  1. Async single call (ainfer) -- full response at once
  2. Async streaming (ainfer_streaming) -- real-time text deltas via SDK events
  3. SDKInferencerResponse -- structured response with session_id + token count
  4. Sync single call (_infer bridge) -- for non-async code

Unlike ClaudeCodeSdkInferencer (persistent connection), DevmateSDKInferencer
creates a fresh SDK client per call. This means there is no ``async with``
context manager pattern — each call stands alone. Session continuity is
provided via ``auto_resume`` / ``previous_session_id``.

Run:
    /usr/local/fbcode/platform010/bin/python3.12 examples/agent_foundation/common/inferencers/agentic_inferencers/external/devmate/example_devmate_sdk_streaming.py

    # Customize:
    /usr/local/fbcode/platform010/bin/python3.12 examples/...example_devmate_sdk_streaming.py -q "Explain recursion" -e 2

Prerequisites:
    - devai.devmate_sdk Python package available (typically via Buck deps:
      //devai/devmate_sdk/python:devmate_python_sdk)
    - Devmate backend services reachable
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


def make_inferencer(target_path: str, model: str):
    """Create a fresh DevmateSDKInferencer (each call uses its own client)."""
    from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
        DevmateSDKInferencer,
    )

    return DevmateSDKInferencer(
        target_path=target_path,
        model_name=model,
    )


# -- Demo 1: Async single call (ainfer) --------------------------------------

async def demo_async_single(query: str, target_path: str, model: str) -> None:
    """ainfer: returns the full response text at once."""
    print("=" * 70)
    print("MODE 1: Async Single Call (ainfer)")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    inf = make_inferencer(target_path, model)

    start = time.time()
    result = await inf.ainfer(query)
    elapsed = time.time() - start

    print("Response:")
    print("-" * 60)
    print(result)
    print("-" * 60)
    print(f"Time:       {elapsed:.2f}s")
    print(f"Type:       {type(result).__name__}")
    sid = getattr(inf, "active_session_id", None)
    print(f"Session ID: {sid}")
    print()


# -- Demo 2: Async streaming (ainfer_streaming) -------------------------------

async def demo_async_streaming(query: str, target_path: str, model: str) -> None:
    """Async streaming: yields text deltas as the Devmate SDK emits events."""
    print("=" * 70)
    print("MODE 2: Async Streaming (ainfer_streaming)")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    inf = make_inferencer(target_path, model)

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
    sid = getattr(inf, "active_session_id", None)
    print(f"Session ID:        {sid}")
    print()


# -- Demo 3: SDKInferencerResponse -------------------------------------------

async def demo_sdk_response(query: str, target_path: str, model: str) -> None:
    """SDKInferencerResponse: structured result with session_id and tokens."""
    print("=" * 70)
    print("MODE 3: SDKInferencerResponse (return_sdk_response=True)")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    inf = make_inferencer(target_path, model)

    start = time.time()
    response = await inf.ainfer(query, return_sdk_response=True)
    elapsed = time.time() - start

    content = getattr(response, "content", str(response))
    sid = getattr(response, "session_id", None)
    tokens = getattr(response, "tokens_received", None)

    print(f"Type:            {type(response).__name__}")
    print(f"Content:         {content[:200]}{'...' if len(content) > 200 else ''}")
    print(f"Session ID:      {sid}")
    print(f"Tokens received: {tokens}")
    print(f"str():           {str(response)[:80]}...")
    print(f"Time:            {elapsed:.2f}s")
    print()
    print("NOTE: SDKInferencerResponse.content returns the full text.")
    print("      str(response) also returns the text (for DualInferencer compat).")
    print("      session_id + tokens_received provide metadata beyond plain ainfer().")
    print()


# -- Demo 4: Sync bridge (_infer) --------------------------------------------

def demo_sync_single(query: str, target_path: str, model: str) -> None:
    """Sync _infer bridge: for non-async code (creates a fresh event loop)."""
    print("=" * 70)
    print("MODE 4: Sync Single Call (via _infer bridge)")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    inf = make_inferencer(target_path, model)

    start = time.time()
    result = inf(query)
    elapsed = time.time() - start

    print("Response:")
    print("-" * 60)
    print(result)
    print("-" * 60)
    print(f"Time:       {elapsed:.2f}s")
    print(f"Type:       {type(result).__name__}")
    print()
    print("NOTE: Sync bridge uses _run_async internally. Since DevmateSDK")
    print("      creates a fresh client per call anyway, the per-call cost")
    print("      is similar for sync vs async modes.")
    print()


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Devmate SDK Inferencer -- Streaming & Modes Demo"
    )
    parser.add_argument(
        "-q", "--query",
        default="Explain what a Python decorator is in 2 bullet points.",
        help="Query to send to Devmate",
    )
    parser.add_argument(
        "-m", "--model",
        default="claude-sonnet-4-5",
        help="Devmate SDK model (default: claude-sonnet-4-5)",
    )
    parser.add_argument(
        "-r", "--root-folder",
        default=os.path.expanduser("~/fbsource"),
        help="Working directory for Devmate agent (default: ~/fbsource)",
    )
    parser.add_argument(
        "-e", "--examples",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Number of examples to run: 1=async, 2=+streaming, 3=+sdk_response, 4=+sync",
    )
    args = parser.parse_args()

    print()
    print("Devmate SDK Inferencer Demo")
    print(f"   Root folder: {args.target_path}")
    print(f"   Model:       {args.model}")
    print()

    # Demo 1: Async single call
    asyncio.run(demo_async_single(args.query, args.target_path, args.model))

    # Demo 2: Async streaming
    if args.examples >= 2:
        asyncio.run(demo_async_streaming(args.query, args.target_path, args.model))

    # Demo 3: SDKInferencerResponse
    if args.examples >= 3:
        asyncio.run(demo_sdk_response(args.query, args.target_path, args.model))

    # Demo 4: Sync bridge
    if args.examples >= 4:
        demo_sync_single(args.query, args.target_path, args.model)

    print("All demos complete!")


if __name__ == "__main__":
    main()
