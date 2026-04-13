#!/usr/bin/env python3
"""Rovo Dev CLI Inferencer — Non-Legacy Mode Streaming Demo.

Demonstrates inference through the Rovo Dev CLI using non-legacy (TUI) mode
(``acli rovodev <message>``). Shows three modes:

1. **Sync one-shot**: Send a prompt and get the full response.
   Output is captured cleanly via auto-injected ``--output-schema``.
2. **XML preservation**: Demonstrates that XML tags in LLM output are
   preserved (they would be eaten by Rich TUI in stdout, but the
   ``--output-schema`` JSON wrapper protects them).
3. **Streaming**: Stream the response token-by-token to the console.
   Note: streaming reads raw stdout, so output is noisier than sync.

Prerequisites:
    - ``acli`` installed and in PATH (``brew install atlassian-cli``)
    - ``acli auth login`` completed (authenticated session)

Usage:
    python example_rovodev_non_legacy_streaming.py
    python example_rovodev_non_legacy_streaming.py --mode sync
    python example_rovodev_non_legacy_streaming.py --mode xml
    python example_rovodev_non_legacy_streaming.py --mode streaming
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


def create_inferencer(working_dir: str):
    """Create a non-legacy RovoDevCliInferencer instance."""
    from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev import (
        RovoDevCliInferencer,
    )

    return RovoDevCliInferencer(
        working_dir=working_dir,
        enable_legacy=False,
        idle_timeout_seconds=600,
        tool_use_idle_timeout_seconds=600,
    )


# =========================================================================
# Demo: Sync one-shot
# =========================================================================


def demo_sync(working_dir: str, query: str) -> None:
    """Non-streaming one-shot inference (non-legacy mode)."""
    print("=" * 70)
    print("Demo: Sync One-Shot Inference (non-legacy)")
    print("=" * 70)
    print()

    inferencer = create_inferencer(working_dir)

    print(f"  Working dir: {working_dir}")
    print(f"  acli path:   {inferencer.acli_path}")
    print(f"  Mode:        non-legacy (enable_legacy=False)")
    print()
    print(f"  You: {query}")

    start = time.time()
    result = inferencer(query)
    elapsed = time.time() - start

    print(f"  Rovo Dev: {result.output}")
    print()
    print(f"  [{elapsed:.1f}s | success={result.success} | rc={result.return_code}]")
    print()


# =========================================================================
# Demo: XML tag preservation
# =========================================================================


def demo_xml(working_dir: str) -> None:
    """Demonstrate XML tag preservation in non-legacy mode."""
    print("=" * 70)
    print("Demo: XML Tag Preservation (non-legacy)")
    print("=" * 70)
    print()

    inferencer = create_inferencer(working_dir)

    query = "Reply with exactly this XML: <Result><Value>42</Value><Status>ok</Status></Result>"
    print(f"  You: {query}")

    start = time.time()
    result = inferencer(query)
    elapsed = time.time() - start

    print(f"  Rovo Dev: {result.output}")
    print()

    has_xml = "<Result>" in result.output and "<Value>" in result.output
    if has_xml:
        print(f"  [PASS] XML tags preserved in output [{elapsed:.1f}s]")
    else:
        print(f"  [FAIL] XML tags were stripped! [{elapsed:.1f}s]")
    print()


# =========================================================================
# Demo: Streaming
# =========================================================================


def demo_streaming(working_dir: str, query: str) -> None:
    """Streaming inference — prints tokens as they arrive."""
    print("=" * 70)
    print("Demo: Streaming Inference (non-legacy)")
    print("=" * 70)
    print()
    print("  Note: Streaming reads raw stdout, so output includes TUI noise.")
    print("  For clean output, use sync mode.")
    print()

    inferencer = create_inferencer(working_dir)

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
    print(f"  [{elapsed:.1f}s | {chunk_count} chunks | {total_chars} chars]")
    print()


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Rovo Dev CLI Inferencer — Non-Legacy Mode Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["sync", "xml", "streaming", "all"],
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

    working_dir = args.working_dir or tempfile.mkdtemp(prefix="rovodev_nonlegacy_demo_")

    # Verify acli is installed
    import shutil

    if not shutil.which("acli"):
        print("  acli not found. Install with: brew install atlassian-cli")
        sys.exit(1)

    print()
    print("Rovo Dev CLI Inferencer Demo (Non-Legacy Mode)")
    print(f"   Working dir: {working_dir}")
    print()

    try:
        if args.mode in ("sync", "all"):
            demo_sync(working_dir, args.query)

        if args.mode in ("xml", "all"):
            demo_xml(working_dir)

        if args.mode in ("streaming", "all"):
            demo_streaming(working_dir, args.query)

        print("Demo complete!")

    except KeyboardInterrupt:
        print("\n\n  Interrupted by user")
    except Exception as e:
        print(f"\n  Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
