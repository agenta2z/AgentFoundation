#!/usr/bin/env python3
"""Rovo Dev CLI Inferencer — Streaming Demo.

Demonstrates streaming inference through the Rovo Dev CLI using
``acli rovodev legacy``. Shows three modes:

1. **Sync one-shot**: Send a prompt and get the full response.
2. **Streaming**: Stream the response token-by-token to the console.
3. **Streaming multi-turn**: Multiple turns in a single conversation
   with streaming output.

Prerequisites:
    - ``acli`` installed and in PATH (``brew install atlassian-cli``)
    - ``acli auth login`` completed (authenticated session)

Usage:
    python example_rovodev_streaming.py
    python example_rovodev_streaming.py --mode streaming --query "Explain decorators"
    python example_rovodev_streaming.py --mode sync --working-dir /path/to/repo
"""

import argparse
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


def create_inferencer(working_dir: str, output_file: str | None = None):
    """Create a RovoDevCliInferencer instance."""
    from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev import (
        RovoDevCliInferencer,
    )

    return RovoDevCliInferencer(
        working_dir=working_dir,
        output_file=output_file,
        idle_timeout_seconds=600,
        tool_use_idle_timeout_seconds=600,
    )


# =========================================================================
# Demo: Sync one-shot
# =========================================================================


def demo_sync(working_dir: str, query: str) -> None:
    """Non-streaming one-shot inference."""
    print("=" * 70)
    print("📦 Demo: Sync One-Shot Inference")
    print("=" * 70)
    print()

    out_file = os.path.join(working_dir, "rovodev_output.txt")
    inferencer = create_inferencer(working_dir, output_file=out_file)

    print(f"  Working dir: {working_dir}")
    print(f"  acli path:   {inferencer.acli_path}")
    print()
    print(f"  You: {query}")
    print(f"  Rovo Dev: ", end="", flush=True)

    start = time.time()
    result = inferencer(query)
    elapsed = time.time() - start

    print(result.output)
    print()
    print(f"  [⏱ {elapsed:.1f}s | success={result.success} | rc={result.return_code}]")
    print()


# =========================================================================
# Demo: Streaming
# =========================================================================


def demo_streaming(working_dir: str, query: str) -> None:
    """Streaming inference — prints tokens as they arrive."""
    print("=" * 70)
    print("🌊 Demo: Streaming Inference")
    print("=" * 70)
    print()

    inferencer = create_inferencer(working_dir)

    print(f"  Working dir: {working_dir}")
    print()
    print(f"  You: {query}")
    print(f"  Rovo Dev: ", end="", flush=True)

    start = time.time()
    total_chars = 0
    chunk_count = 0

    for chunk in inferencer.infer_streaming(query):
        if chunk.strip():
            print(chunk, end="", flush=True)
            total_chars += len(chunk)
            chunk_count += 1

    elapsed = time.time() - start
    print()
    print()
    print(f"  [⏱ {elapsed:.1f}s | {chunk_count} chunks | {total_chars} chars]")
    print()


# =========================================================================
# Demo: Streaming multi-turn
# =========================================================================


def demo_multi_turn_streaming(working_dir: str) -> None:
    """Multi-turn conversation with streaming output."""
    print("=" * 70)
    print("🔄 Demo: Multi-Turn Streaming Conversation")
    print("=" * 70)
    print()

    out_file = os.path.join(working_dir, "rovodev_output.txt")
    inferencer = create_inferencer(working_dir, output_file=out_file)

    turns = [
        "My favorite programming language is Rust. Just acknowledge this.",
        "What is my favorite programming language? Reply with just the language name.",
    ]

    for i, prompt in enumerate(turns, 1):
        print(f"  --- Turn {i} ---")
        print(f"  You: {prompt}")
        print(f"  Rovo Dev: ", end="", flush=True)

        start = time.time()

        if i == 1:
            result = inferencer.new_session(prompt)
        else:
            result = inferencer(prompt)

        elapsed = time.time() - start
        print(result.output)
        print(f"  [⏱ {elapsed:.1f}s | session={inferencer.active_session_id}]")
        print()

    print("=" * 70)
    print(f"✓ Completed {len(turns)} turns")
    print(f"  Session ID: {inferencer.active_session_id}")
    print("=" * 70)
    print()


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Rovo Dev CLI Inferencer — Streaming Demo",
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
        default="What is a Python decorator? Explain in 2 sentences.",
        help="Query to send",
    )
    parser.add_argument(
        "--working-dir",
        help="Working directory for the agent (default: temp dir)",
    )
    args = parser.parse_args()

    working_dir = args.working_dir or tempfile.mkdtemp(prefix="rovodev_demo_")

    # Verify acli is installed
    import shutil

    if not shutil.which("acli"):
        print("❌ acli not found. Install with: brew install atlassian-cli")
        sys.exit(1)

    print()
    print("🤖 Rovo Dev CLI Inferencer Demo")
    print(f"   Working dir: {working_dir}")
    print()

    try:
        if args.mode in ("sync", "all"):
            demo_sync(working_dir, args.query)

        if args.mode in ("streaming", "all"):
            demo_streaming(working_dir, args.query)

        if args.mode in ("multi-turn", "all"):
            demo_multi_turn_streaming(working_dir)

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
