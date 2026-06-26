#!/usr/bin/env python3
"""Codex CLI Inferencer — Sync, Streaming & Multi-Turn Demo.

Demonstrates the ``CodexCliInferencer`` (wraps the OpenAI ``codex exec`` CLI):
  1. Sync one-shot inference
  2. Async streaming
  3. Multi-turn session resume (``codex exec resume``)

Prerequisites:
    - ``codex`` CLI installed and authenticated (run ``codex login``;
      verify with ``codex login status``).

Usage:
    python example_codex_cli_streaming.py
    python example_codex_cli_streaming.py --mode sync --query "Explain a Python decorator in one sentence."
    python example_codex_cli_streaming.py --mode multi-turn
    python example_codex_cli_streaming.py --model gpt-5-codex --target-path /path/to/repo
"""

import argparse
import asyncio
import os
import sys
import tempfile
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


def create_inferencer(args: argparse.Namespace):
    """Create a CodexCliInferencer from CLI args."""
    from agent_foundation.common.inferencers.agentic_inferencers.external.codex import (
        CodexCliInferencer,
    )

    kwargs = {"target_path": args.target_path, "idle_timeout_seconds": 600}
    if args.model:
        kwargs["model_name"] = args.model
    return CodexCliInferencer(**kwargs)


def demo_sync(inferencer, query: str) -> None:
    """Non-streaming one-shot inference."""
    print("=" * 70)
    print("📦 Demo: Sync One-Shot Inference")
    print("=" * 70)
    print(f"  You:   {query}")

    start = time.time()
    result = inferencer(query)
    elapsed = time.time() - start

    print(f"  Codex: {result.output}")
    print(
        f"  [⏱ {elapsed:.1f}s | success={result.success} | "
        f"session={inferencer.active_session_id}]"
    )
    print()


async def demo_streaming(inferencer, query: str) -> None:
    """Streaming inference — prints the response as it arrives."""
    print("=" * 70)
    print("🌊 Demo: Streaming Inference")
    print("=" * 70)
    print(f"  You:   {query}")
    print("  Codex: ", end="", flush=True)

    start = time.time()
    total_chars = 0
    async for chunk in inferencer.ainfer_streaming(query):
        if chunk:
            print(chunk, end="", flush=True)
            total_chars += len(chunk)

    print()
    print(
        f"  [⏱ {time.time() - start:.1f}s | {total_chars} chars | "
        f"session={inferencer.active_session_id}]"
    )
    print()


async def demo_multi_turn(inferencer) -> None:
    """Multi-turn conversation via session resume."""
    print("=" * 70)
    print("🔄 Demo: Multi-Turn Session Resume")
    print("=" * 70)

    turns = [
        "My favorite programming language is Rust. Just acknowledge with 'noted'.",
        "What is my favorite programming language? Reply with just the language name.",
    ]
    for i, prompt in enumerate(turns, 1):
        print(f"  --- Turn {i} ---")
        print(f"  You:   {prompt}")
        if i == 1:
            result = await inferencer.ainfer(prompt, new_session=True)
        else:
            result = await inferencer.ainfer(prompt)
        print(f"  Codex: {result.output}")
        print(f"  [session={inferencer.active_session_id}]")
        print()

    print(f"  ✓ Completed {len(turns)} turns in session {inferencer.active_session_id}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Codex CLI Inferencer — Sync, Streaming & Multi-Turn Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["sync", "streaming", "multi-turn", "all"],
        default="all",
        help="Demo mode (default: all)",
    )
    parser.add_argument(
        "--query",
        default="Explain what a Python decorator is in one sentence.",
        help="Query to send",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Codex model for -m (default: codex's configured model)",
    )
    parser.add_argument(
        "--target-path",
        default=None,
        help="Working directory for the agent (default: temp dir)",
    )
    args = parser.parse_args()

    if args.target_path is None:
        args.target_path = tempfile.mkdtemp(prefix="codex_demo_")

    # Verify codex is installed
    import shutil

    if not shutil.which("codex"):
        print("❌ codex CLI not found. Install it, then run: codex login")
        sys.exit(1)

    print()
    print("🤖 Codex CLI Inferencer Demo")
    print(f"   Working dir: {args.target_path}")
    print()

    inferencer = create_inferencer(args)

    try:
        if args.mode in ("sync", "all"):
            demo_sync(inferencer, args.query)

        if args.mode in ("streaming", "all"):
            inferencer.reset_session()  # fresh conversation for the streaming demo
            asyncio.run(demo_streaming(inferencer, args.query))

        if args.mode in ("multi-turn", "all"):
            inferencer.reset_session()
            asyncio.run(demo_multi_turn(inferencer))

        print("🎉 Demo complete!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
