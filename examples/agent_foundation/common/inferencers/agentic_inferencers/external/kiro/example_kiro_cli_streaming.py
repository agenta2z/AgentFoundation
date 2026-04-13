#!/usr/bin/env python3
"""Kiro CLI Inferencer — Streaming vs Sync Demo.

Demonstrates three inference modes using the Kiro CLI:
  1. Non-streaming (sync) — full response at once
  2. Async streaming — real-time line-by-line output via subprocess streaming
  3. Sync streaming — same as async but from synchronous code

Run:
    python examples/agent_foundation/common/inferencers/agentic_inferencers/external/kiro/example_kiro_cli_streaming.py

Prerequisites:
    - ``kiro-cli`` installed and in PATH
    - Authenticated (``kiro-cli login``)
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


def create_inferencer(args):
    """Create a KiroCliInferencer with the given configuration."""
    from agent_foundation.common.inferencers.agentic_inferencers.external.kiro import (
        KiroCliInferencer,
    )

    return KiroCliInferencer(
        target_path=args.target_path,
        model_name=args.model,
    )


# ── Demo 1: Non-streaming (sync) ─────────────────────────────────────────

def demo_sync(inferencer, query: str) -> None:
    """Sync mode: sends query, waits for full response."""
    print("=" * 70)
    print("MODE: Non-Streaming (Synchronous)")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    start = time.time()
    response = inferencer(query)
    elapsed = time.time() - start

    print(f"Success:     {response.success}")
    print(f"Time:        {elapsed:.2f}s")
    print(f"Return code: {response.return_code}")
    print()
    print("Response:")
    print("-" * 60)
    output = response.output if response.success else (response.error or "Unknown error")
    print(output)
    print("-" * 60)
    print()


# ── Demo 2: Async streaming (real-time line output) ──────────────────────

async def demo_async_streaming(inferencer, query: str) -> None:
    """Async streaming: prints text line-by-line as Kiro generates."""
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
    print(f"Time:             {elapsed:.2f}s")
    print(f"Time to 1st line: {ttfc:.2f}s")
    print(f"Characters:       {char_count}")
    print(f"Lines:            {line_count}")
    print()


# ── Demo 3: Sync streaming (for non-async code) ─────────────────────────

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


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Kiro CLI — Streaming vs Sync Demo"
    )
    parser.add_argument(
        "-q", "--query",
        default="Explain what a Python decorator is in 3 bullet points.",
        help="Query to send to Kiro",
    )
    parser.add_argument(
        "-m", "--model",
        default="auto",
        help="Model name/alias (default: auto). Examples: claude-sonnet-4.6, haiku, opus",
    )
    parser.add_argument(
        "-t", "--target-path",
        default="/tmp",
        help="Working directory for Kiro CLI (default: /tmp)",
    )
    parser.add_argument(
        "-e", "--examples",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Number of examples to run: 1=sync, 2=+async stream, 3=+sync stream",
    )
    args = parser.parse_args()

    try:
        inferencer = create_inferencer(args)
    except Exception as e:
        print(f"❌ Failed to create inferencer: {e}")
        return

    # Auth check
    if not inferencer.check_auth(timeout=15.0):
        print("❌ Not authenticated. Run 'kiro-cli login' first.")
        return

    print()
    print("🤖 Kiro CLI Inferencer Demo")
    print(f"   Model:       {inferencer.model_name}")
    print(f"   Working dir: {inferencer.working_dir}")
    print()

    # Demo 1: Sync
    demo_sync(inferencer, args.query)

    # Demo 2: Async streaming
    if args.examples >= 2:
        asyncio.run(demo_async_streaming(inferencer, args.query))

    # Demo 3: Sync streaming
    if args.examples >= 3:
        demo_sync_streaming(inferencer, args.query)

    print("✅ All demos complete!")


if __name__ == "__main__":
    main()
