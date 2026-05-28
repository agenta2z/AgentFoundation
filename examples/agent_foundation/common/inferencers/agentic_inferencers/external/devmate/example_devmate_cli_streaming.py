#!/usr/bin/env python3
"""Devmate CLI Inferencer — Streaming vs Sync Demo.

Demonstrates three inference modes using the Devmate CLI:
  1. Non-streaming (sync) — full response at once with metadata (session_id,
     trajectory_url, return_code)
  2. Async streaming — real-time line-by-line output via subprocess line
     streaming
  3. Sync streaming — same as async but from synchronous code

Run:
    /usr/local/fbcode/platform010/bin/python3.12 examples/agent_foundation/common/inferencers/agentic_inferencers/external/devmate/example_devmate_cli_streaming.py

Prerequisites:
    - `devmate` CLI available on PATH (e.g. `/usr/local/bin/devmate`)
    - A valid fbsource repo to use as the working directory
      (defaults to ~/fbsource)
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
    """Create a DevmateCliInferencer with the given configuration."""
    from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
        DevmateCliInferencer,
    )

    return DevmateCliInferencer(
        target_path=args.target_path,
        model_name=args.model,
        max_tokens=args.max_tokens,
        no_create_commit=True,
        config_name=args.config_name,
    )


# ── Demo 1: Non-streaming (sync with parsed metadata) ────────────────────

def demo_sync(inferencer, query: str) -> None:
    """Sync mode: sends query, waits for full response with metadata."""
    print("=" * 70)
    print("MODE: Non-Streaming (Synchronous)")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    start = time.time()
    response = inferencer(query)
    elapsed = time.time() - start

    success = getattr(response, "success", None)
    return_code = getattr(response, "return_code", None)
    session_id = getattr(response, "session_id", None)
    trajectory_url = getattr(response, "trajectory_url", None)
    output_text = getattr(response, "output", None) or str(response)

    print(f"Success:        {success}")
    print(f"Time:           {elapsed:.2f}s")
    print(f"Return code:    {return_code}")
    if session_id:
        print(f"Session ID:     {session_id}")
    if trajectory_url:
        print(f"Trajectory URL: {trajectory_url}")
    print()
    print("Response:")
    print("-" * 60)
    error_text = getattr(response, "error", None)
    body = output_text if success else (error_text or output_text or "Unknown error")
    print(body)
    print("-" * 60)
    print()


# ── Demo 2: Async streaming (real-time line output) ──────────────────────

async def demo_async_streaming(inferencer, query: str) -> None:
    """Async streaming: prints text line-by-line as Devmate generates."""
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
    print(f"Time to 1st line:  {ttfc:.2f}s")
    print(f"Characters:        {char_count}")
    print(f"Lines:             {line_count}")
    print()
    print("NOTE: Streaming mode filters out session header/footer lines by")
    print("      default. The session_id and trajectory_url are still")
    print("      available via inferencer.get_streaming_result() after")
    print("      streaming completes.")
    print()

    # Pull the parsed metadata from the streaming result
    parsed = inferencer.get_streaming_result()
    sid = parsed.get("session_id")
    turl = parsed.get("trajectory_url")
    print(f"Streaming Session ID:     {sid}")
    print(f"Streaming Trajectory URL: {turl}")
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
        description="Devmate CLI — Streaming vs Sync Demo"
    )
    parser.add_argument(
        "-q", "--query",
        default="Explain what a Python decorator is in 3 bullet points.",
        help="Query to send to Devmate",
    )
    parser.add_argument(
        "-m", "--model",
        default="claude-sonnet-4.5",
        help="Devmate model name (default: claude-sonnet-4.5)",
    )
    parser.add_argument(
        "-r", "--repo-path",
        default=os.path.expanduser("~/fbsource"),
        help="Repo working directory for Devmate (default: ~/fbsource)",
    )
    parser.add_argument(
        "--config-name",
        default="freeform",
        help="Devmate config name (default: freeform)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens for response (default: 4096 — keep small for demo)",
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

    print()
    print("🤖 Devmate CLI Inferencer Demo")
    print(f"   Model:        {inferencer.model_name}")
    print(f"   Repo path:    {inferencer.target_path}")
    print(f"   Config name:  {inferencer.config_name}")
    print(f"   Max tokens:   {inferencer.max_tokens}")
    print()

    # Demo 1: Sync
    demo_sync(inferencer, args.query)

    # Each streaming demo uses a fresh inferencer to avoid auto-resume coupling
    if args.examples >= 2:
        try:
            inferencer2 = create_inferencer(args)
            asyncio.run(demo_async_streaming(inferencer2, args.query))
        except Exception as e:
            print(f"⚠️  Async streaming demo failed: {e}")

    if args.examples >= 3:
        try:
            inferencer3 = create_inferencer(args)
            demo_sync_streaming(inferencer3, args.query)
        except Exception as e:
            print(f"⚠️  Sync streaming demo failed: {e}")

    print("✅ All demos complete!")


if __name__ == "__main__":
    main()
