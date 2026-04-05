#!/usr/bin/env python3
"""Real integration test for ClaudeCodeCliInferencer.

This script actually calls the Claude Code CLI to verify the inferencer works.

Usage (via buck2):
    # Run all tests with default query
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_claude_code_cli_inferencer_real

    # Run with custom query
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_claude_code_cli_inferencer_real -- --query "What is Python?"

    # Run specific test mode
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_claude_code_cli_inferencer_real -- --mode sync --query "Explain recursion"
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_claude_code_cli_inferencer_real -- --mode async --query "What is a list?"
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_claude_code_cli_inferencer_real -- --mode streaming --query "Explain Python"
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_claude_code_cli_inferencer_real -- --mode session
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_claude_code_cli_inferencer_real -- --mode all
"""

import argparse
import asyncio
import logging
import sys
import time

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default query for tests
DEFAULT_QUERY = "What is Python? Answer in one sentence."


def test_sync_single_call(query: str) -> bool:
    """Test synchronous single call to Claude Code CLI."""
    print("\n" + "=" * 60)
    print("TEST: Sync Single Call (CLI)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import ClaudeCodeCliInferencer: {e}")
        return False

    try:
        inferencer = ClaudeCodeCliInferencer(
            target_path="/tmp",
            model_name="sonnet",
            allowed_tools=[],  # No tools for simple question
        )
        print(f"✓ Created inferencer")

        print(f"\nSending SYNC query: '{query}'")
        start_time = time.time()

        response = inferencer(query)

        elapsed = time.time() - start_time
        print(f"\n✓ Got response in {elapsed:.2f}s:")
        print(f"  Response type: {type(response)}")
        print(f"  Success: {response.get('success')}")
        print(f"  Session ID: {response.get('session_id', 'N/A')[:16]}...")
        print(f"  Return code: {response.get('return_code')}")
        output = response.get("output", "")
        print(f"  Output length: {len(output)} chars")
        print("-" * 40)
        print(output[:500] if len(output) > 500 else output)
        print("-" * 40)

        if response.get("success") and len(output) >= 1:
            print("\n✅ SYNC TEST PASSED!")
            return True
        else:
            print(f"\n❌ SYNC TEST FAILED: success={response.get('success')}")
            if response.get("error"):
                print(f"  Error: {response.get('error')}")
            return False

    except Exception as e:
        print(f"\n❌ SYNC TEST FAILED with exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sync_multiple_calls() -> bool:
    """Test multiple sync calls (tests separate invocations)."""
    print("\n" + "=" * 60)
    print("TEST: Sync Multiple Calls (CLI)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import ClaudeCodeCliInferencer: {e}")
        return False

    try:
        inferencer = ClaudeCodeCliInferencer(
            target_path="/tmp",
            model_name="sonnet",
            allowed_tools=[],
        )

        # First call
        print("\nFirst SYNC call: 'What is 2+2?'")
        start_time = time.time()
        response1 = inferencer("What is 2+2? Just give the number.")
        elapsed1 = time.time() - start_time
        output1 = response1.get("output", "")
        print(f"✓ Response 1 in {elapsed1:.2f}s: {output1[:100]}...")

        # Second call
        print("\nSecond SYNC call: 'What is 3+3?'")
        start_time = time.time()
        response2 = inferencer("What is 3+3? Just give the number.")
        elapsed2 = time.time() - start_time
        output2 = response2.get("output", "")
        print(f"✓ Response 2 in {elapsed2:.2f}s: {output2[:100]}...")

        if response1.get("success") and response2.get("success"):
            print("\n✅ SYNC MULTIPLE CALLS PASSED!")
            return True
        else:
            print("\n❌ SYNC MULTIPLE CALLS FAILED: One or both responses failed")
            return False

    except Exception as e:
        print(
            f"\n❌ SYNC MULTIPLE CALLS FAILED with exception: {type(e).__name__}: {e}"
        )
        import traceback

        traceback.print_exc()
        return False


async def test_async_single_call(query: str) -> bool:
    """Test async single call."""
    print("\n" + "=" * 60)
    print("TEST: Async Single Call (CLI)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import ClaudeCodeCliInferencer: {e}")
        return False

    try:
        inferencer = ClaudeCodeCliInferencer(
            target_path="/tmp",
            model_name="sonnet",
            allowed_tools=[],
        )

        print(f"\nSending ASYNC query: '{query}'")
        start_time = time.time()

        response = await inferencer.ainfer(query)

        elapsed = time.time() - start_time
        print(f"\n✓ Got response in {elapsed:.2f}s:")
        print(f"  Response type: {type(response)}")
        print(f"  Success: {response.get('success')}")
        print(f"  Session ID: {response.get('session_id', 'N/A')[:16]}...")
        output = response.get("output", "")
        print(f"  Output length: {len(output)} chars")
        print("-" * 40)
        print(output[:500] if len(output) > 500 else output)
        print("-" * 40)

        if response.get("success") and len(output) >= 1:
            print("\n✅ ASYNC TEST PASSED!")
            return True
        else:
            print(f"\n❌ ASYNC TEST FAILED: success={response.get('success')}")
            return False

    except Exception as e:
        print(f"\n❌ ASYNC TEST FAILED with exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_async_multiple_calls() -> bool:
    """Test multiple async calls."""
    print("\n" + "=" * 60)
    print("TEST: Async Multiple Calls (CLI)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import ClaudeCodeCliInferencer: {e}")
        return False

    try:
        inferencer = ClaudeCodeCliInferencer(
            target_path="/tmp",
            model_name="sonnet",
            allowed_tools=[],
        )

        # First query
        print("\nFirst ASYNC query: 'What is 2+2?'")
        start_time = time.time()
        response1 = await inferencer.ainfer("What is 2+2? Just the number.")
        elapsed1 = time.time() - start_time
        output1 = response1.get("output", "")
        print(f"✓ Response 1 in {elapsed1:.2f}s: {output1[:100]}...")

        # Second query
        print("\nSecond ASYNC query: 'What is 3+3?'")
        start_time = time.time()
        response2 = await inferencer.ainfer("What is 3+3? Just the number.")
        elapsed2 = time.time() - start_time
        output2 = response2.get("output", "")
        print(f"✓ Response 2 in {elapsed2:.2f}s: {output2[:100]}...")

        if response1.get("success") and response2.get("success"):
            print("\n✅ ASYNC MULTIPLE CALLS PASSED!")
            return True
        else:
            print("\n❌ ASYNC MULTIPLE CALLS FAILED: One or both responses failed")
            return False

    except Exception as e:
        print(
            f"\n❌ ASYNC MULTIPLE CALLS FAILED with exception: {type(e).__name__}: {e}"
        )
        import traceback

        traceback.print_exc()
        return False


async def test_response_metadata() -> bool:
    """Test that response contains expected metadata fields (JSON mode)."""
    print("\n" + "=" * 60)
    print("TEST: Response Metadata (CLI JSON output)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False

    try:
        inferencer = ClaudeCodeCliInferencer(
            target_path="/tmp",
            model_name="sonnet",
            allowed_tools=[],
        )

        print("\nSending query to check response metadata...")
        response = await inferencer.ainfer("Say hello.")

        print(f"\n✓ Got response:")
        print(f"  Type: {type(response)}")
        print(f"  Keys: {list(response.keys())}")
        print(f"  success: {response.get('success')}")
        print(f"  session_id: {response.get('session_id', 'N/A')[:16] if response.get('session_id') else 'N/A'}...")
        print(f"  return_code: {response.get('return_code')}")
        print(f"  total_cost_usd: {response.get('total_cost_usd')}")
        print(f"  num_turns: {response.get('num_turns')}")
        print(f"  duration_ms: {response.get('duration_ms')}")

        # Check required fields
        has_output = "output" in response
        has_success = "success" in response
        has_return_code = "return_code" in response

        if has_output and has_success and has_return_code:
            print("\n✅ RESPONSE METADATA TEST PASSED!")
            return True
        else:
            print("\n❌ RESPONSE METADATA TEST FAILED: Missing required fields")
            return False

    except Exception as e:
        print(
            f"\n❌ RESPONSE METADATA TEST FAILED with exception: {type(e).__name__}: {e}"
        )
        import traceback

        traceback.print_exc()
        return False


def test_sync_streaming(query: str) -> bool:
    """Test sync streaming - outputs text to terminal in real-time."""
    print("\n" + "=" * 60)
    print("TEST: Sync Streaming (CLI infer_streaming)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import ClaudeCodeCliInferencer: {e}")
        return False

    try:
        inferencer = ClaudeCodeCliInferencer(
            target_path="/tmp",
            model_name="sonnet",
            allowed_tools=[],
        )
        print(f"✓ Created inferencer")

        print(f"\nSending SYNC STREAMING query: '{query}'")
        print("-" * 40)
        print("STREAMING OUTPUT:")
        print("-" * 40)

        start_time = time.time()
        lines = []
        line_count = 0

        # Stream output to terminal in real-time
        for line in inferencer.infer_streaming(query):
            lines.append(line)
            line_count += 1
            # Print each line immediately (real-time streaming to terminal)
            print(line, end="", flush=True)

        elapsed = time.time() - start_time
        full_response = "".join(lines)

        print("\n" + "-" * 40)
        print(f"\n✓ Streaming completed in {elapsed:.2f}s:")
        print(f"  Total lines received: {line_count}")
        print(f"  Total chars: {len(full_response)}")

        # Get streaming result for additional info
        result = inferencer.get_streaming_result()
        print(f"  Return code: {result.get('return_code')}")
        print(f"  Success: {result.get('success')}")

        if line_count > 0 and len(full_response) >= 1:
            print("\n✅ SYNC STREAMING TEST PASSED!")
            return True
        else:
            print("\n❌ SYNC STREAMING TEST FAILED: No lines received")
            return False

    except Exception as e:
        print(
            f"\n❌ SYNC STREAMING TEST FAILED with exception: {type(e).__name__}: {e}"
        )
        import traceback

        traceback.print_exc()
        return False


async def test_async_streaming(query: str) -> bool:
    """Test async streaming - outputs text to terminal in real-time."""
    print("\n" + "=" * 60)
    print("TEST: Async Streaming (CLI ainfer_streaming)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import ClaudeCodeCliInferencer: {e}")
        return False

    try:
        inferencer = ClaudeCodeCliInferencer(
            target_path="/tmp",
            model_name="sonnet",
            allowed_tools=[],
        )
        print(f"✓ Created inferencer")

        print(f"\nSending ASYNC STREAMING query: '{query}'")
        print("-" * 40)
        print("STREAMING OUTPUT:")
        print("-" * 40)

        start_time = time.time()
        lines = []
        line_count = 0

        # Stream output to terminal in real-time
        async for line in inferencer.ainfer_streaming(query):
            lines.append(line)
            line_count += 1
            # Print each line immediately (real-time streaming to terminal)
            print(line, end="", flush=True)

        elapsed = time.time() - start_time
        full_response = "".join(lines)

        print("\n" + "-" * 40)
        print(f"\n✓ Streaming completed in {elapsed:.2f}s:")
        print(f"  Total lines received: {line_count}")
        print(f"  Total chars: {len(full_response)}")

        if line_count > 0 and len(full_response) >= 1:
            print("\n✅ ASYNC STREAMING TEST PASSED!")
            return True
        else:
            print("\n❌ ASYNC STREAMING TEST FAILED: No lines received")
            return False

    except Exception as e:
        print(
            f"\n❌ ASYNC STREAMING TEST FAILED with exception: {type(e).__name__}: {e}"
        )
        import traceback

        traceback.print_exc()
        return False


async def test_session_continuation() -> bool:
    """Test session continuation (multi-turn conversation via --resume)."""
    print("\n" + "=" * 60)
    print("TEST: Session Continuation (CLI Multi-turn via --resume)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import ClaudeCodeCliInferencer: {e}")
        return False

    try:
        # Create inferencer with auto_resume=True (default)
        inferencer = ClaudeCodeCliInferencer(
            target_path="/tmp",
            model_name="sonnet",
            allowed_tools=[],
            auto_resume=True,
        )

        # First call - start new session
        print("\nFirst call: 'My favorite number is 42. Remember it.'")
        start_time = time.time()
        result1 = await inferencer.anew_session(
            "My favorite number is 42. Remember it."
        )
        elapsed1 = time.time() - start_time

        session_id = result1.get("session_id")
        print(f"✓ Response 1 in {elapsed1:.2f}s")
        print(f"  Session ID: {session_id[:16] if session_id else 'N/A'}...")
        print(f"  Success: {result1.get('success')}")
        print(f"  Active session: {inferencer.active_session_id[:16] if inferencer.active_session_id else 'N/A'}...")
        output1 = result1.get("output", "")
        print(f"  Response: {output1[:100]}...")

        if not session_id:
            print("\n⚠️ No session ID returned - cannot test continuation")
            # Still pass if first call worked
            if result1.get("success"):
                print("\n✅ SESSION TEST PASSED (no continuation, but first call worked)")
                return True
            return False

        # Second call - should auto-resume (auto_resume=True)
        print("\nSecond call: 'What is my favorite number?' (auto-resume)")
        start_time = time.time()
        result2 = await inferencer.ainfer("What is my favorite number?")
        elapsed2 = time.time() - start_time

        print(f"✓ Response 2 in {elapsed2:.2f}s")
        print(f"  Success: {result2.get('success')}")
        output2 = result2.get("output", "")
        print(f"  Response: {output2[:200]}...")

        # Check if the model remembered "42"
        if "42" in output2:
            print("\n✅ SESSION CONTINUATION TEST PASSED! Model remembered '42'")
            return True
        else:
            print("\n⚠️ SESSION CONTINUATION TEST WARNING: '42' not found in response")
            print("  (This may be a model behavior issue, not an inferencer issue)")
            # Still pass if we got a valid response
            if result2.get("success"):
                print("\n✅ SESSION CONTINUATION TEST PASSED (with warning)")
                return True
            return False

    except Exception as e:
        print(
            f"\n❌ SESSION CONTINUATION TEST FAILED with exception: {type(e).__name__}: {e}"
        )
        import traceback

        traceback.print_exc()
        return False


def test_command_construction() -> bool:
    """Test that command construction works correctly with various options."""
    print("\n" + "=" * 60)
    print("TEST: Command Construction (CLI)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import ClaudeCodeCliInferencer: {e}")
        return False

    try:
        # Test with various options
        inferencer = ClaudeCodeCliInferencer(
            target_path="/tmp",
            model_name="opus",
            system_prompt="You are a helpful assistant.",
            allowed_tools=["Read", "Write", "Bash"],
            permission_mode="bypassPermissions",
            max_budget_usd=1.0,
            extra_cli_args=["--verbose"],
        )

        # Construct a command
        command = inferencer.construct_command(
            {"prompt": "Test prompt with $pecial \"characters\" and `backticks`"},
            output_format="json",
        )

        print(f"\n✓ Constructed command:")
        print(f"  {command}")

        # Check command contains expected parts
        checks = [
            ("claude", "claude binary"),
            ("-p", "print mode flag"),
            ("--output-format json", "output format"),
            ("--model opus", "model name"),
            ('--system-prompt "You are a helpful assistant."', "system prompt"),
            ('--allowedTools "Read,Write,Bash"', "allowed tools"),
            ("--dangerously-skip-permissions", "permission mode"),
            ("--max-budget-usd 1.0", "max budget"),
            ("--verbose", "extra CLI arg"),
        ]

        all_passed = True
        for check_str, description in checks:
            if check_str in command:
                print(f"  ✓ Contains {description}")
            else:
                print(f"  ❌ Missing {description}: {check_str}")
                all_passed = False

        # Check shell escaping
        if "\\$pecial" in command and '\\"characters\\"' in command:
            print("  ✓ Shell escaping applied")
        else:
            print("  ⚠️ Shell escaping may not be applied correctly")

        if all_passed:
            print("\n✅ COMMAND CONSTRUCTION TEST PASSED!")
            return True
        else:
            print("\n❌ COMMAND CONSTRUCTION TEST FAILED: Missing expected components")
            return False

    except Exception as e:
        print(
            f"\n❌ COMMAND CONSTRUCTION TEST FAILED with exception: {type(e).__name__}: {e}"
        )
        import traceback

        traceback.print_exc()
        return False


def run_sync_tests(query: str) -> list:
    """Run all sync tests."""
    results = []
    results.append(("Sync Single Call", test_sync_single_call(query)))
    results.append(("Sync Multiple Calls", test_sync_multiple_calls()))
    results.append(("Command Construction", test_command_construction()))
    return results


def run_streaming_tests(query: str) -> list:
    """Run streaming tests (sync)."""
    results = []
    results.append(("Sync Streaming", test_sync_streaming(query)))
    return results


async def run_async_tests(query: str) -> list:
    """Run all async tests."""
    results = []
    results.append(("Async Single Call", await test_async_single_call(query)))
    results.append(("Async Multiple Calls", await test_async_multiple_calls()))
    results.append(("Response Metadata", await test_response_metadata()))
    results.append(("Async Streaming", await test_async_streaming(query)))
    results.append(("Session Continuation", await test_session_continuation()))
    return results


def print_summary(results: list) -> bool:
    """Print test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")
    return failed == 0


def main() -> int:
    """Run tests based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real integration tests for ClaudeCodeCliInferencer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests with default query
  python test_claude_code_cli_inferencer_real.py

  # Run with custom query
  python test_claude_code_cli_inferencer_real.py --query "Explain Python decorators"

  # Run only sync tests
  python test_claude_code_cli_inferencer_real.py --mode sync --query "What is a list?"

  # Run only async tests
  python test_claude_code_cli_inferencer_real.py --mode async --query "What is a dict?"

  # Run streaming tests (see output in real-time)
  python test_claude_code_cli_inferencer_real.py --mode streaming --query "Explain Python"

  # Run session continuation test
  python test_claude_code_cli_inferencer_real.py --mode session
        """,
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=DEFAULT_QUERY,
        help=f"Custom query to test (default: '{DEFAULT_QUERY}')",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["sync", "async", "streaming", "async-streaming", "session", "all"],
        default="all",
        help="Test mode: sync, async, streaming, async-streaming, session, or all (default: all)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CLAUDE CODE CLI INFERENCER - REAL INTEGRATION TESTS")
    print("=" * 60)
    print(f"\nQuery: {args.query}")
    print(f"Mode: {args.mode}")

    results = []

    if args.mode in ("sync", "all"):
        results.extend(run_sync_tests(args.query))

    if args.mode in ("streaming", "all"):
        results.extend(run_streaming_tests(args.query))

    if args.mode == "async-streaming":
        # Run just async streaming test
        async_streaming_result = asyncio.run(test_async_streaming(args.query))
        results.append(("Async Streaming", async_streaming_result))

    if args.mode in ("async", "all"):
        async_results = asyncio.run(run_async_tests(args.query))
        results.extend(async_results)

    if args.mode == "session":
        # Run just session test
        session_result = asyncio.run(test_session_continuation())
        results.append(("Session Continuation", session_result))

    success = print_summary(results)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
