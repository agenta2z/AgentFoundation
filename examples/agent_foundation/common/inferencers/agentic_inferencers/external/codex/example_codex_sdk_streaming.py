#!/usr/bin/env python3
"""Codex SDK Inferencer — Async, Streaming, SDK-Response & Multi-Turn Demo.

Demonstrates the ``CodexSdkInferencer`` (wraps the official ``openai-codex``
Python SDK — ``AsyncCodex``/``AsyncThread`` over the Codex app-server):
  1. Async single call (``ainfer``)
  2. Async streaming (``ainfer_streaming`` — token-level deltas)
  3. ``SDKInferencerResponse`` (``return_sdk_response=True`` -> content + session + tool_uses + tokens)
  4. Multi-turn on a held thread (session continuity)

Prerequisites:
    - ``pip install openai-codex``
    - Codex authenticated (``codex login``; verify with ``codex login status``)

Usage:
    python example_codex_sdk_streaming.py
    python example_codex_sdk_streaming.py --mode streaming --query "Explain recursion in one sentence."
    python example_codex_sdk_streaming.py --mode multi-turn
    python example_codex_sdk_streaming.py --model gpt-5-codex --sandbox read-only
"""

import argparse
import asyncio
import json
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
    from agent_foundation.common.inferencers.agentic_inferencers.external.codex import (
        CodexSdkInferencer,
    )

    kwargs = {"target_path": args.target_path, "sandbox_mode": args.sandbox}
    if args.model:
        kwargs["model_name"] = args.model
    # Optional: point the SDK at a specific codex app-server via a JSON object of
    # openai_codex.CodexConfig kwargs. Needed where the bundled codex can't
    # run/auth -- e.g. on a host that nests a seatbelt sandbox:
    #   CODEX_SDK_CONFIG_KWARGS='{"launch_args_override": ["/usr/local/bin/codex",
    #   "--dangerously-disable-osx-sandbox", "app-server", "--listen", "stdio://"]}'
    cfg = os.environ.get("CODEX_SDK_CONFIG_KWARGS")
    if cfg:
        kwargs["codex_config_kwargs"] = json.loads(cfg)
    return CodexSdkInferencer(**kwargs)


async def demo_async_single(inferencer, query: str) -> None:
    print("=" * 70)
    print("📦 Demo: Async Single Call (ainfer)")
    print("=" * 70)
    print(f"  You:   {query}")
    start = time.time()
    async with inferencer:
        result = await inferencer.ainfer(query)
    print(f"  Codex: {result}")
    print(f"  [⏱ {time.time() - start:.1f}s | session={inferencer.active_session_id}]")
    print()


async def demo_streaming(inferencer, query: str) -> None:
    print("=" * 70)
    print("🌊 Demo: Async Streaming (ainfer_streaming)")
    print("=" * 70)
    print(f"  You:   {query}")
    print("  Codex: ", end="", flush=True)
    start = time.time()
    chars = 0
    async with inferencer:
        async for chunk in inferencer.ainfer_streaming(query):
            if chunk:
                print(chunk, end="", flush=True)
                chars += len(chunk)
    print()
    print(f"  [⏱ {time.time() - start:.1f}s | {chars} chars | session={inferencer.active_session_id}]")
    print()


async def demo_sdk_response(inferencer, query: str) -> None:
    print("=" * 70)
    print("🧱 Demo: SDKInferencerResponse (return_sdk_response=True)")
    print("=" * 70)
    print(f"  You:   {query}")
    async with inferencer:
        resp = await inferencer.ainfer(query, return_sdk_response=True)
    print(f"  Type:       {type(resp).__name__}")
    print(f"  content:    {resp.content[:200]}")
    print(f"  session_id: {resp.session_id}")
    print(f"  tool_uses:  {resp.tool_uses}")
    print(f"  tokens:     {resp.tokens_received}")
    print()


async def demo_multi_turn(inferencer) -> None:
    print("=" * 70)
    print("🔄 Demo: Multi-Turn (held thread)")
    print("=" * 70)
    turns = [
        "My favorite programming language is Rust. Just acknowledge with 'noted'.",
        "What is my favorite programming language? Reply with just the language name.",
    ]
    async with inferencer:
        for i, prompt in enumerate(turns, 1):
            print(f"  --- Turn {i} ---")
            print(f"  You:   {prompt}")
            result = await inferencer.ainfer(prompt)
            print(f"  Codex: {result}")
            print(f"  [session={inferencer.active_session_id}]")
            print()
    print(f"  ✓ Completed {len(turns)} turns in one conversation")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Codex SDK Inferencer — Async/Streaming/Multi-Turn Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["async", "streaming", "sdk-response", "multi-turn", "all"],
        default="all",
        help="Demo mode (default: all)",
    )
    parser.add_argument(
        "--query",
        default="Explain what a Python decorator is in one sentence.",
        help="Query to send",
    )
    parser.add_argument("--model", default=None, help="Codex model (default: your Codex login's model)")
    parser.add_argument(
        "--sandbox",
        default="read-only",
        choices=["read-only", "workspace-write", "full-access"],
        help="Codex sandbox policy (default: read-only)",
    )
    parser.add_argument("--target-path", default=None, help="Working directory (default: temp dir)")
    args = parser.parse_args()

    if args.target_path is None:
        args.target_path = tempfile.mkdtemp(prefix="codex_sdk_demo_")

    try:
        import openai_codex  # noqa: F401
    except ImportError:
        print(
            "❌ openai-codex SDK not installed. It is currently in beta; install "
            "from the openai/codex repo:\n   pip install "
            "'git+https://github.com/openai/codex.git#subdirectory=sdk/python'"
        )
        sys.exit(1)

    print()
    print("🤖 Codex SDK Inferencer Demo")
    print(f"   Working dir: {args.target_path} | sandbox: {args.sandbox}")
    print()

    try:
        if args.mode in ("async", "all"):
            asyncio.run(demo_async_single(create_inferencer(args), args.query))
        if args.mode in ("streaming", "all"):
            asyncio.run(demo_streaming(create_inferencer(args), args.query))
        if args.mode in ("sdk-response", "all"):
            asyncio.run(demo_sdk_response(create_inferencer(args), args.query))
        if args.mode in ("multi-turn", "all"):
            asyncio.run(demo_multi_turn(create_inferencer(args)))
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
