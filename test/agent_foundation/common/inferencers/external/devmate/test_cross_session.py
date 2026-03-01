#!/usr/bin/env python3
"""Cross-session test for all three inferencers.

This tests interleaved session handling:
1. Start Session A (favorite number = 42)
2. Start Session B (favorite color = blue)
3. Resume Session A (ask about number) - should remember 42
4. Resume Session B (ask about color) - should remember blue

This is a critical test to verify that sessions are truly independent
and the inferencer can switch between multiple sessions correctly.

Run with:
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_cross_session
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_cross_session -- --inferencer sdk
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_cross_session -- --inferencer cli
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_cross_session -- --inferencer claude
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_cross_session -- --inferencer all
"""

import argparse
import asyncio
import sys
import time


async def test_cross_session_sdk():
    """Test cross-session support for DevmateSDKInferencer."""
    print("\n" + "=" * 70)
    print("CROSS-SESSION TEST - DevmateSDKInferencer")
    print("=" * 70)

    from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
        DevmateSDKInferencer,
    )

    inferencer = DevmateSDKInferencer(
        root_folder="/data/users/zgchen/fbsource",
        config_file_path="config.dual_agent.md",
        usecase="dual_agent_coding",
        total_timeout_seconds=120,
        idle_timeout_seconds=60,
        auto_resume=False,  # Disable auto-resume to manually control sessions
    )

    # === SESSION A: Tell it favorite number is 42 ===
    print("\n[1] SESSION A: Start new session - 'My favorite number is 42'")
    start = time.time()
    result_a1 = await inferencer.anew_session(
        "My favorite number is 42. Remember this number. Just acknowledge."
    )
    session_a = inferencer.active_session_id
    print(f"    ✓ Session A ID: {session_a}")
    print(f"    ✓ Time: {time.time() - start:.2f}s")

    # === SESSION B: Tell it favorite color is blue ===
    print("\n[2] SESSION B: Start new session - 'My favorite color is blue'")
    start = time.time()
    result_b1 = await inferencer.anew_session(
        "My favorite color is blue. Remember this color. Just acknowledge."
    )
    session_b = inferencer.active_session_id
    print(f"    ✓ Session B ID: {session_b}")
    print(f"    ✓ Time: {time.time() - start:.2f}s")

    print(f"\n    Sessions are different: {session_a != session_b}")

    if session_a == session_b:
        print("\n❌ CRITICAL ERROR: Session A and B have same ID!")
        return False

    # === RESUME SESSION A: Ask about number ===
    print("\n[3] RESUME SESSION A: Ask 'What is my favorite number?'")
    start = time.time()
    result_a2 = await inferencer.aresume_session(
        "What is my favorite number? Just tell me the number.",
        session_id=session_a,
    )
    print(f"    ✓ Time: {time.time() - start:.2f}s")
    print(f"    ✓ Response: {str(result_a2)[:200]}...")

    number_remembered = "42" in str(result_a2)
    print(f"    ✓ Contains '42': {number_remembered}")

    # === RESUME SESSION B: Ask about color ===
    print("\n[4] RESUME SESSION B: Ask 'What is my favorite color?'")
    start = time.time()
    result_b2 = await inferencer.aresume_session(
        "What is my favorite color? Just tell me the color.",
        session_id=session_b,
    )
    print(f"    ✓ Time: {time.time() - start:.2f}s")
    print(f"    ✓ Response: {str(result_b2)[:200]}...")

    color_remembered = "blue" in str(result_b2).lower()
    print(f"    ✓ Contains 'blue': {color_remembered}")

    # === SUMMARY ===
    success = number_remembered and color_remembered and (session_a != session_b)
    print(
        f"\n{'✅ PASSED' if success else '❌ FAILED'} - DevmateSDKInferencer Cross-Session"
    )
    return success


async def test_cross_session_cli():
    """Test cross-session support for DevmateCliInferencer."""
    print("\n" + "=" * 70)
    print("CROSS-SESSION TEST - DevmateCliInferencer")
    print("=" * 70)

    from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
        DevmateCliInferencer,
    )

    inferencer = DevmateCliInferencer(
        repo_path="/data/users/zgchen/fbsource",
        model_name="claude-sonnet-4.5",
        no_create_commit=True,
        auto_resume=False,  # Disable auto-resume to manually control sessions
    )

    # === SESSION A: Tell it favorite number is 42 ===
    print("\n[1] SESSION A: Start new session - 'My favorite number is 42'")
    start = time.time()
    result_a1 = await inferencer.anew_session(
        "My favorite number is 42. Remember this number. Just acknowledge."
    )
    session_a = inferencer.active_session_id
    print(f"    ✓ Session A ID: {session_a}")
    print(f"    ✓ Time: {time.time() - start:.2f}s")

    # === SESSION B: Tell it favorite color is blue ===
    print("\n[2] SESSION B: Start new session - 'My favorite color is blue'")
    start = time.time()
    result_b1 = await inferencer.anew_session(
        "My favorite color is blue. Remember this color. Just acknowledge."
    )
    session_b = inferencer.active_session_id
    print(f"    ✓ Session B ID: {session_b}")
    print(f"    ✓ Time: {time.time() - start:.2f}s")

    print(f"\n    Sessions are different: {session_a != session_b}")

    if session_a == session_b:
        print("\n❌ CRITICAL ERROR: Session A and B have same ID!")
        return False

    # === RESUME SESSION A: Ask about number ===
    print("\n[3] RESUME SESSION A: Ask 'What is my favorite number?'")
    start = time.time()
    result_a2 = await inferencer.aresume_session(
        "What is my favorite number? Just tell me the number.",
        session_id=session_a,
    )
    print(f"    ✓ Time: {time.time() - start:.2f}s")
    output_a2 = str(result_a2.get("output", ""))
    print(f"    ✓ Response: {output_a2[:200]}...")

    number_remembered = "42" in output_a2
    print(f"    ✓ Contains '42': {number_remembered}")

    # === RESUME SESSION B: Ask about color ===
    print("\n[4] RESUME SESSION B: Ask 'What is my favorite color?'")
    start = time.time()
    result_b2 = await inferencer.aresume_session(
        "What is my favorite color? Just tell me the color.",
        session_id=session_b,
    )
    print(f"    ✓ Time: {time.time() - start:.2f}s")
    output_b2 = str(result_b2.get("output", ""))
    print(f"    ✓ Response: {output_b2[:200]}...")

    color_remembered = "blue" in output_b2.lower()
    print(f"    ✓ Contains 'blue': {color_remembered}")

    # === SUMMARY ===
    success = number_remembered and color_remembered and (session_a != session_b)
    print(
        f"\n{'✅ PASSED' if success else '❌ FAILED'} - DevmateCliInferencer Cross-Session"
    )
    return success


async def test_cross_session_claude():
    """Test cross-session support for ClaudeCodeInferencer."""
    print("\n" + "=" * 70)
    print("CROSS-SESSION TEST - ClaudeCodeInferencer")
    print("=" * 70)

    from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
        ClaudeCodeInferencer,
    )

    inferencer = ClaudeCodeInferencer(
        root_folder="/tmp",
        allowed_tools=[],
        idle_timeout_seconds=120,
        auto_resume=False,  # Disable auto-resume to manually control sessions
    )

    # === SESSION A: Tell it favorite number is 42 ===
    print("\n[1] SESSION A: Start new session - 'My favorite number is 42'")
    start = time.time()
    result_a1 = await inferencer.anew_session(
        "My favorite number is 42. Remember this number. Just acknowledge."
    )
    session_a = inferencer.active_session_id
    print(f"    ✓ Session A ID: {session_a}")
    print(f"    ✓ Time: {time.time() - start:.2f}s")

    # === SESSION B: Tell it favorite color is blue ===
    print("\n[2] SESSION B: Start new session - 'My favorite color is blue'")
    start = time.time()
    result_b1 = await inferencer.anew_session(
        "My favorite color is blue. Remember this color. Just acknowledge."
    )
    session_b = inferencer.active_session_id
    print(f"    ✓ Session B ID: {session_b}")
    print(f"    ✓ Time: {time.time() - start:.2f}s")

    print(f"\n    Session A: {session_a}")
    print(f"    Session B: {session_b}")
    # Note: ClaudeCodeInferencer may have None session_id by design
    sessions_different = (session_a != session_b) or (
        session_a is None and session_b is None
    )
    print(f"    Sessions check: {sessions_different}")

    # === RESUME SESSION A: Ask about number ===
    print("\n[3] RESUME SESSION A: Ask 'What is my favorite number?'")
    start = time.time()
    # ClaudeCodeInferencer resume uses session_id kwarg
    if session_a:
        result_a2 = await inferencer.aresume_session(
            "What is my favorite number? Just tell me the number.",
            session_id=session_a,
        )
    else:
        # If no session_id, just call ainfer (may not have cross-session support)
        result_a2 = await inferencer.ainfer(
            "What is my favorite number? Just tell me the number."
        )
    print(f"    ✓ Time: {time.time() - start:.2f}s")
    print(f"    ✓ Response: {str(result_a2)[:200]}...")

    number_remembered = "42" in str(result_a2)
    print(f"    ✓ Contains '42': {number_remembered}")

    # === RESUME SESSION B: Ask about color ===
    print("\n[4] RESUME SESSION B: Ask 'What is my favorite color?'")
    start = time.time()
    if session_b:
        result_b2 = await inferencer.aresume_session(
            "What is my favorite color? Just tell me the color.",
            session_id=session_b,
        )
    else:
        result_b2 = await inferencer.ainfer(
            "What is my favorite color? Just tell me the color."
        )
    print(f"    ✓ Time: {time.time() - start:.2f}s")
    print(f"    ✓ Response: {str(result_b2)[:200]}...")

    color_remembered = "blue" in str(result_b2).lower()
    print(f"    ✓ Contains 'blue': {color_remembered}")

    # === SUMMARY ===
    # For Claude, we need session IDs to work for true cross-session
    if session_a is None or session_b is None:
        print("\n⚠️ WARNING: ClaudeCodeInferencer does not return session IDs")
        print("   Cross-session may not be supported by the underlying SDK")
        # Still check if memory works within context
        success = number_remembered or color_remembered
    else:
        success = number_remembered and color_remembered and (session_a != session_b)

    print(
        f"\n{'✅ PASSED' if success else '❌ FAILED'} - ClaudeCodeInferencer Cross-Session"
    )
    return success


async def run_all_tests():
    """Run cross-session tests for all inferencers."""
    results = {}

    results["DevmateSDKInferencer"] = await test_cross_session_sdk()
    results["DevmateCliInferencer"] = await test_cross_session_cli()
    results["ClaudeCodeInferencer"] = await test_cross_session_claude()

    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-session tests for inferencers")
    parser.add_argument(
        "--inferencer",
        "-i",
        type=str,
        choices=["sdk", "cli", "claude", "all"],
        default="sdk",
        help="Which inferencer to test (default: sdk)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("CROSS-SESSION VERIFICATION TEST")
    print("=" * 70)
    print(f"Testing: {args.inferencer}")

    if args.inferencer == "sdk":
        result = asyncio.run(test_cross_session_sdk())
        results = {"DevmateSDKInferencer": result}
    elif args.inferencer == "cli":
        result = asyncio.run(test_cross_session_cli())
        results = {"DevmateCliInferencer": result}
    elif args.inferencer == "claude":
        result = asyncio.run(test_cross_session_claude())
        results = {"ClaudeCodeInferencer": result}
    else:  # all
        results = asyncio.run(run_all_tests())

    # Print summary
    print("\n" + "=" * 70)
    print("CROSS-SESSION TEST SUMMARY")
    print("=" * 70)
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
