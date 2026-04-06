#!/usr/bin/env python3
"""
Tutorial: Streaming with AgClaudeApiInferencer
==============================================

This script demonstrates streaming inference through the AI Gateway,
showing real-time token-by-token output in multiple modes.

Examples covered:
    1. Sync streaming bridge (infer_streaming)
    2. Async streaming (ainfer_streaming)
    3. Streaming vs sync comparison (time-to-first-token)
    4. Multi-turn streaming with set_messages()
    5. Streaming with different gateway modes

Prerequisites:
    At least one gateway mode must be available:
    - atlas CLI (for direct mode)
    - proximity ai-gateway (for proximity mode)
    - atlas slauth server --port 5000 (for slauth_server mode)

Run:
    PYTHONPATH=src python examples/agent_foundation/common/inferencers/api_inferencers/ag/example_ag_streaming.py
"""

import asyncio
import time

from agent_foundation.apis.ag import (
    AIGatewayClaudeModels,
    detect_available_mode,
)
from agent_foundation.common.inferencers.api_inferencers.ag.ag_claude_api_inferencer import (
    AgClaudeApiInferencer,
)


# ──────────────────────────────────────────────────────────────────────────────
# Example 1: Sync streaming via infer_streaming()
# ──────────────────────────────────────────────────────────────────────────────

def example_sync_streaming(mode: str):
    """Demonstrates the sync streaming bridge — works in regular (non-async) code."""
    inferencer = AgClaudeApiInferencer(
        model_id=str(AIGatewayClaudeModels.CLAUDE_45_SONNET),
        gateway_mode=mode,
    )

    prompt = "Write a haiku about streaming data."

    print("=" * 70)
    print(f"Example 1: Sync Streaming (infer_streaming) — mode={mode}")
    print("=" * 70)
    print(f"  Prompt: {prompt}")
    print(f"  Output: ", end="", flush=True)

    start = time.time()
    first_chunk_time = None
    chunk_count = 0

    for chunk in inferencer.infer_streaming(prompt):
        if first_chunk_time is None:
            first_chunk_time = time.time()
        print(chunk, end="", flush=True)
        chunk_count += 1

    elapsed_ms = int((time.time() - start) * 1000)
    ttft_ms = int((first_chunk_time - start) * 1000) if first_chunk_time else 0

    print()
    print(f"  Chunks : {chunk_count}")
    print(f"  TTFT   : {ttft_ms}ms")
    print(f"  Total  : {elapsed_ms}ms")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Example 2: Async streaming via ainfer_streaming()
# ──────────────────────────────────────────────────────────────────────────────

async def example_async_streaming(mode: str):
    """Demonstrates native async streaming — best for async applications."""
    inferencer = AgClaudeApiInferencer(
        model_id=str(AIGatewayClaudeModels.CLAUDE_45_SONNET),
        gateway_mode=mode,
    )

    prompt = "Explain what an API gateway does in three sentences."

    print("=" * 70)
    print(f"Example 2: Async Streaming (ainfer_streaming) — mode={mode}")
    print("=" * 70)
    print(f"  Prompt: {prompt}")
    print(f"  Output: ", end="", flush=True)

    start = time.time()
    first_chunk_time = None
    chunk_count = 0

    async for chunk in inferencer.ainfer_streaming(prompt):
        if first_chunk_time is None:
            first_chunk_time = time.time()
        print(chunk, end="", flush=True)
        chunk_count += 1

    elapsed_ms = int((time.time() - start) * 1000)
    ttft_ms = int((first_chunk_time - start) * 1000) if first_chunk_time else 0

    print()
    print(f"  Chunks : {chunk_count}")
    print(f"  TTFT   : {ttft_ms}ms")
    print(f"  Total  : {elapsed_ms}ms")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Example 3: Streaming vs Sync — side-by-side comparison
# ──────────────────────────────────────────────────────────────────────────────

async def example_streaming_vs_sync(mode: str):
    """Compares streaming and sync inference for the same prompt."""
    prompt = "What are three benefits of streaming APIs? Be concise."

    print("=" * 70)
    print(f"Example 3: Streaming vs Sync Comparison — mode={mode}")
    print("=" * 70)

    inferencer = AgClaudeApiInferencer(
        model_id=str(AIGatewayClaudeModels.CLAUDE_45_SONNET),
        gateway_mode=mode,
        max_tokens=300,
        temperature=0.3,
    )

    # Sync (non-streaming) inference
    print("  [Sync] ...")
    start = time.time()
    sync_result = inferencer(prompt)
    sync_ms = int((time.time() - start) * 1000)
    print(f"  [Sync] {sync_ms}ms — {sync_result[:80]}...")

    # Streaming inference
    print("  [Stream] ...")
    start = time.time()
    first_chunk_time = None
    stream_parts = []

    async for chunk in inferencer.ainfer_streaming(prompt):
        if first_chunk_time is None:
            first_chunk_time = time.time()
        stream_parts.append(chunk)

    stream_ms = int((time.time() - start) * 1000)
    ttft_ms = int((first_chunk_time - start) * 1000) if first_chunk_time else 0
    stream_result = "".join(stream_parts)

    print(f"  [Stream] {stream_ms}ms total, {ttft_ms}ms TTFT — {stream_result[:80]}...")
    print()
    print(f"  Summary:")
    print(f"    Sync total time          : {sync_ms}ms")
    print(f"    Stream time-to-first-token: {ttft_ms}ms")
    print(f"    Stream total time         : {stream_ms}ms")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Example 4: Multi-turn streaming with set_messages()
# ──────────────────────────────────────────────────────────────────────────────

async def example_multiturn_streaming(mode: str):
    """Demonstrates multi-turn conversation with streaming responses."""
    inferencer = AgClaudeApiInferencer(
        model_id=str(AIGatewayClaudeModels.CLAUDE_45_SONNET),
        gateway_mode=mode,
        system_prompt="You are a helpful assistant. Be concise.",
        max_tokens=200,
        temperature=0.5,
    )

    print("=" * 70)
    print(f"Example 4: Multi-turn Streaming — mode={mode}")
    print("=" * 70)

    conversation = []

    turns = [
        "What is an API gateway?",
        "How does Atlassian use one?",
        "What about rate limiting?",
    ]

    for i, user_msg in enumerate(turns, 1):
        conversation.append({"role": "user", "content": user_msg})
        inferencer.set_messages(conversation)

        print(f"  Turn {i} (user): {user_msg}")
        print(f"  Turn {i} (assistant): ", end="", flush=True)

        response_parts = []
        async for chunk in inferencer.ainfer_streaming(""):  # prompt ignored when messages are set
            print(chunk, end="", flush=True)
            response_parts.append(chunk)

        assistant_response = "".join(response_parts)
        conversation.append({"role": "assistant", "content": assistant_response})
        print()
        print()

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Example 5: Streaming with different gateway modes
# ──────────────────────────────────────────────────────────────────────────────

async def example_streaming_modes():
    """Demonstrates streaming across different gateway modes."""
    prompt = "What is 2+2? Answer with just the number."

    print("=" * 70)
    print("Example 5: Streaming Across Gateway Modes")
    print("=" * 70)

    for mode in ["auto", "direct", "proximity"]:
        inferencer = AgClaudeApiInferencer(
            model_id=str(AIGatewayClaudeModels.CLAUDE_45_SONNET),
            gateway_mode=mode,
            max_tokens=16,
            temperature=0.0,
        )

        try:
            start = time.time()
            parts = []
            first_chunk_time = None

            async for chunk in inferencer.ainfer_streaming(prompt):
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                parts.append(chunk)

            elapsed_ms = int((time.time() - start) * 1000)
            ttft_ms = int((first_chunk_time - start) * 1000) if first_chunk_time else 0
            result = "".join(parts)
            print(f"  [{mode:14s}] {elapsed_ms}ms total, {ttft_ms}ms TTFT — {result.strip()}")

        except Exception as e:
            print(f"  [{mode:14s}] FAILED: {e}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main — run all examples
# ──────────────────────────────────────────────────────────────────────────────

async def _async_main(mode: str):
    await example_async_streaming(mode)
    await example_streaming_vs_sync(mode)
    await example_multiturn_streaming(mode)
    await example_streaming_modes()


if __name__ == "__main__":
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        description="Tutorial: AgClaudeApiInferencer Streaming",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["direct", "proximity", "slauth_server", "auto"],
        default=None,
        help="Force a specific gateway mode. If omitted, auto-detects.",
    )
    parser.add_argument(
        "--example", "-e",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Run a specific example (1-5). If omitted, runs all.",
    )
    args = parser.parse_args()

    print()
    print("+" + "=" * 68 + "+")
    print("|   AgClaudeApiInferencer — Streaming Tutorial                       |")
    print("+" + "=" * 68 + "+")
    print()
    print(f"  User: {os.environ.get('AI_GATEWAY_USER_ID') or os.environ.get('USER', '(unknown)')}")

    # Detect mode
    if args.mode:
        mode = args.mode
        print(f"  Mode: {mode} (forced)")
    else:
        try:
            detected = detect_available_mode()
            mode = str(detected)
            print(f"  Mode: {mode} (auto-detected)")
        except RuntimeError as e:
            print(f"\n  Error: {e}")
            sys.exit(1)
    print()

    # Map example numbers to their (async) factory functions
    async_examples = {
        2: lambda: example_async_streaming(mode),
        3: lambda: example_streaming_vs_sync(mode),
        4: lambda: example_multiturn_streaming(mode),
        5: lambda: example_streaming_modes(),
    }

    try:
        if args.example == 1 or args.example is None:
            example_sync_streaming(mode)

        if args.example is None:
            asyncio.run(_async_main(mode))
        elif args.example in async_examples:
            asyncio.run(async_examples[args.example]())

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 70)
    print("Streaming examples completed.")
    print("=" * 70)
