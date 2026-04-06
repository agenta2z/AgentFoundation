#!/usr/bin/env python3
"""
Tutorial: Using AgClaudeApiInferencer with Different AI Gateway Modes
=====================================================================

This script demonstrates the three ways to connect to Claude models through
the Atlassian AI Gateway, plus the auto-detection mode that picks the best
available option automatically.

Gateway Modes:
    direct        - Shells out to `atlas slauth token` CLI. No local server needed.
    proximity     - Forwards to a local proximity proxy (port 29576). Proxy handles auth.
    slauth_server - Uses the AI Gateway SDK with a local SLAuth server (port 5000).
    auto          - Tries direct → proximity → slauth_server; falls back on failure.

Prerequisites (you only need ONE of these):
    Option A (direct):       Install atlas CLI — `atlas plugin install -n slauth`
    Option B (proximity):    Start proximity — `proximity ai-gateway`
    Option C (slauth_server): Start SLAuth server — `atlas slauth server --port 5000`

Run:
    PYTHONPATH=src python examples/agent_foundation/common/inferencers/ag_gateway_modes/example_ag_gateway_modes.py
"""

import time

from agent_foundation.apis.ag import (
    AIGatewayClaudeModels,
    GatewayMode,
    check_direct_available,
    check_proximity_available,
    check_slauth_server_available,
    detect_available_mode,
    generate_text,
)
from agent_foundation.common.inferencers.api_inferencers.ag.ag_claude_api_inferencer import (
    AgClaudeApiInferencer,
)


# ──────────────────────────────────────────────────────────────────────────────
# Example 1: Check which gateway modes are available on your machine
# ──────────────────────────────────────────────────────────────────────────────

def example_check_availability():
    direct_ok, direct_reason = check_direct_available()
    proximity_ok, proximity_reason = check_proximity_available()
    slauth_ok, slauth_reason = check_slauth_server_available()

    # region ── display ──
    print("=" * 70)
    print("Example 1: Gateway Mode Availability Check")
    print("=" * 70)
    print(f"  direct        : {'AVAILABLE' if direct_ok else 'UNAVAILABLE'}")
    if not direct_ok:
        print(f"                    Reason: {direct_reason}")
    print(f"  proximity     : {'AVAILABLE' if proximity_ok else 'UNAVAILABLE'}")
    if not proximity_ok:
        print(f"                    Reason: {proximity_reason}")
    print(f"  slauth_server : {'AVAILABLE' if slauth_ok else 'UNAVAILABLE'}")
    if not slauth_ok:
        print(f"                    Reason: {slauth_reason}")
    print()

    if not any([direct_ok, proximity_ok, slauth_ok]):
        print("  ⚠  No gateway mode is available. Set up one of:")
        print("       atlas plugin install -n slauth       (for direct mode)")
        print("       proximity ai-gateway                 (for proximity mode)")
        print("       atlas slauth server --port 5000      (for slauth_server mode)")
        return None

    detected = detect_available_mode()
    print(f"  Auto-detected : {detected}")
    print()
    # endregion
    return detected


# ──────────────────────────────────────────────────────────────────────────────
# Example 2: Basic text generation using generate_text() with explicit mode
# ──────────────────────────────────────────────────────────────────────────────

def example_basic_generate(mode: str):
    prompt = "What is the Atlassian AI Gateway? Answer in two sentences."

    start = time.time()
    response = generate_text(
        prompt_or_messages=prompt,
        model=AIGatewayClaudeModels.CLAUDE_46_OPUS,
        max_new_tokens=200,
        temperature=0.3,
        gateway_mode=mode,
    )
    elapsed_ms = int((time.time() - start) * 1000)

    # region ── display ──
    print("=" * 70)
    print(f"Example 2: Basic generate_text() — mode={mode}")
    print("=" * 70)
    print(f"  Prompt   : {prompt}")
    print(f"  Model    : {AIGatewayClaudeModels.CLAUDE_46_OPUS}")
    print(f"  Mode     : {mode}")
    print(f"  Latency  : {elapsed_ms}ms")
    print(f"  Response :")
    for line in response.splitlines():
        print(f"    {line}")
    print()
    # endregion
    return response


# ──────────────────────────────────────────────────────────────────────────────
# Example 3: Using the AgClaudeApiInferencer class (recommended for agents)
# ──────────────────────────────────────────────────────────────────────────────

def example_inferencer(mode: str):
    inferencer = AgClaudeApiInferencer(
        model_id=str(AIGatewayClaudeModels.CLAUDE_46_OPUS),
        gateway_mode=mode,
        max_retry=2,
    )

    prompt = "List three benefits of using an API gateway. Be concise."

    start = time.time()
    response = inferencer(prompt, max_new_tokens=300, temperature=0.3)
    elapsed_ms = int((time.time() - start) * 1000)

    # region ── display ──
    print("=" * 70)
    print(f"Example 3: AgClaudeApiInferencer — mode={mode}")
    print("=" * 70)
    print(f"  Model    : {inferencer.model_id}")
    print(f"  Mode     : {inferencer.gateway_mode}")
    print(f"  Latency  : {elapsed_ms}ms")
    print(f"  Response :")
    for line in response.splitlines():
        print(f"    {line}")
    print()
    # endregion
    return response


# ──────────────────────────────────────────────────────────────────────────────
# Example 4: Auto mode with runtime fallback
# ──────────────────────────────────────────────────────────────────────────────

def example_auto_mode():
    prompt = "What is 2+2? Answer with just the number."

    start = time.time()
    response = generate_text(
        prompt_or_messages=prompt,
        model=AIGatewayClaudeModels.CLAUDE_46_OPUS,
        max_new_tokens=16,
        temperature=0.0,
        gateway_mode="auto",
    )
    elapsed_ms = int((time.time() - start) * 1000)

    # region ── display ──
    print("=" * 70)
    print("Example 4: Auto Mode (detects best available, falls back on failure)")
    print("=" * 70)
    print(f"  Prompt   : {prompt}")
    print(f"  Latency  : {elapsed_ms}ms")
    print(f"  Response : {response}")
    print()
    # endregion
    return response


# ──────────────────────────────────────────────────────────────────────────────
# Example 5: Multi-turn conversation with system prompt
# ──────────────────────────────────────────────────────────────────────────────

def example_conversation(mode: str):
    system_prompt = "You are a helpful assistant who explains things simply."
    conversation = [
        "What is an API gateway?",
        "An API gateway is a server that acts as the single entry point for API requests.",
        "How does Atlassian use one?",
    ]

    start = time.time()
    response = generate_text(
        prompt_or_messages=conversation,
        model=AIGatewayClaudeModels.CLAUDE_46_OPUS,
        system=system_prompt,
        max_new_tokens=300,
        temperature=0.5,
        gateway_mode=mode,
    )
    elapsed_ms = int((time.time() - start) * 1000)

    # region ── display ──
    print("=" * 70)
    print(f"Example 5: Multi-turn Conversation — mode={mode}")
    print("=" * 70)
    print(f"  System   : {system_prompt}")
    print(f"  Turn 1   : {conversation[0]}")
    print(f"  Turn 2   : {conversation[1][:60]}...")
    print(f"  Turn 3   : {conversation[2]}")
    print(f"  Latency  : {elapsed_ms}ms")
    print(f"  Response :")
    for line in response.splitlines():
        print(f"    {line}")
    print()
    # endregion
    return response


# ──────────────────────────────────────────────────────────────────────────────
# Example 6: Comparing models side by side
# ──────────────────────────────────────────────────────────────────────────────

def example_compare_models(mode: str):
    prompt = "Explain recursion in one sentence."
    models = [
        (AIGatewayClaudeModels.CLAUDE_46_OPUS, "Opus 4.6"),
        (AIGatewayClaudeModels.CLAUDE_45_SONNET, "Sonnet 4.5"),
        (AIGatewayClaudeModels.CLAUDE_45_HAIKU, "Haiku 4.5"),
    ]

    results = []
    for model_enum, label in models:
        start = time.time()
        try:
            resp = generate_text(
                prompt_or_messages=prompt,
                model=model_enum,
                max_new_tokens=100,
                temperature=0.3,
                gateway_mode=mode,
            )
            elapsed_ms = int((time.time() - start) * 1000)
            results.append((label, str(model_enum), elapsed_ms, resp, None))
        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            results.append((label, str(model_enum), elapsed_ms, None, str(e)))

    # region ── display ──
    print("=" * 70)
    print(f"Example 6: Model Comparison — mode={mode}")
    print("=" * 70)
    print(f"  Prompt: {prompt}")
    print()
    for label, model_id, ms, resp, err in results:
        print(f"  [{label}] ({ms}ms)")
        if resp:
            print(f"    {resp[:120]}")
        else:
            print(f"    ERROR: {err[:120]}")
    print()
    # endregion
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main — run all examples
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        description="Tutorial: AgClaudeApiInferencer with different AI Gateway modes",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["direct", "proximity", "slauth_server", "auto"],
        default=None,
        help="Force a specific gateway mode. If omitted, auto-detects the best available.",
    )
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   AgClaudeApiInferencer — Multi-Mode AI Gateway Tutorial           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  User: {os.environ.get('AI_GATEWAY_USER_ID') or os.environ.get('USER', '(unknown)')}")
    print()

    # Step 1: Check what's available
    detected_mode = example_check_availability()
    if detected_mode is None and args.mode is None:
        sys.exit(1)

    mode = args.mode or str(detected_mode)
    if args.mode:
        print(f"  Forced mode: {args.mode}")
        print()

    # Step 2: Run examples with the detected mode
    try:
        example_basic_generate(mode)
        example_inferencer(mode)
        example_auto_mode()
        example_conversation(mode)
        example_compare_models(mode)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 70)
    print("All examples completed.")
    print("=" * 70)
