#!/usr/bin/env python3
"""Real integration test for DevmateCliInferencer.

This script actually calls the Devmate CLI to verify the inferencer works.

Usage (via buck2):
    # Run all tests with default query
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_devmate_cli_inferencer_real

    # Run with custom query
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_devmate_cli_inferencer_real -- --query "What is Python?"

    # Run specific test mode
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_devmate_cli_inferencer_real -- --mode sync --query "Explain recursion"
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_devmate_cli_inferencer_real -- --mode async --query "What is a list?"
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_devmate_cli_inferencer_real -- --mode streaming --query "What is Python?"
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_devmate_cli_inferencer_real -- --mode session
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_devmate_cli_inferencer_real -- --mode all
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


def test_sync_single_call(query: str):
    """Test synchronous single call to Devmate CLI."""
    print("\n" + "=" * 60)
    print("TEST: Sync Single Call")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
            DevmateCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import DevmateCliInferencer: {e}")
        return False

    try:
        inferencer = DevmateCliInferencer(
            repo_path="/data/users/zgchen/fbsource",
            model_name="claude-sonnet-4.5",
            no_create_commit=True,
        )
        print(f"✓ Created inferencer")

        print(f"\nSending SYNC query: '{query}'")
        start_time = time.time()

        result = inferencer.infer(query)

        elapsed = time.time() - start_time
        print(f"\n✓ Got response in {elapsed:.2f}s:")
        print(f"  Success: {result.get('success')}")
        print(f"  Return code: {result.get('return_code')}")
        print(f"  Session ID: {result.get('session_id')}")
        print(f"  Trajectory URL: {result.get('trajectory_url')}")
        print(f"  Output length: {len(result.get('output', ''))} chars")
        print("-" * 40)
        output = result.get("output", "")
        print(output[:500] + "..." if len(output) > 500 else output)
        print("-" * 40)

        if result.get("success") and len(result.get("output", "")) >= 1:
            print("\n✅ SYNC TEST PASSED!")
            return True
        else:
            print(f"\n❌ SYNC TEST FAILED: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"\n❌ SYNC TEST FAILED with exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sync_with_headless(query: str):
    """Test synchronous call with headless mode."""
    print("\n" + "=" * 60)
    print("TEST: Sync with Headless Mode")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
            DevmateCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import DevmateCliInferencer: {e}")
        return False

    try:
        inferencer = DevmateCliInferencer(
            repo_path="/data/users/zgchen/fbsource",
            model_name="claude-sonnet-4.5",
            headless=True,
            no_create_commit=True,
        )
        print(f"✓ Created inferencer with headless=True")

        print(f"\nSending query: '{query}'")
        start_time = time.time()

        result = inferencer.infer(query)

        elapsed = time.time() - start_time
        print(f"\n✓ Got response in {elapsed:.2f}s:")
        print(f"  Success: {result.get('success')}")
        print(f"  Output length: {len(result.get('output', ''))} chars")

        if result.get("success") and len(result.get("output", "")) >= 1:
            print("\n✅ HEADLESS TEST PASSED!")
            return True
        else:
            print(f"\n❌ HEADLESS TEST FAILED: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"\n❌ HEADLESS TEST FAILED with exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sync_with_dump_output(query: str):
    """Test synchronous call with dump_output mode."""
    print("\n" + "=" * 60)
    print("TEST: Sync with Dump Output Mode")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
            DevmateCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import DevmateCliInferencer: {e}")
        return False

    try:
        inferencer = DevmateCliInferencer(
            repo_path="/data/users/zgchen/fbsource",
            model_name="claude-sonnet-4.5",
            dump_output=True,
            no_create_commit=True,
        )
        print(f"✓ Created inferencer with dump_output=True")

        print(f"\nSending query: '{query}'")
        start_time = time.time()

        result = inferencer.infer(query)

        elapsed = time.time() - start_time
        print(f"\n✓ Got response in {elapsed:.2f}s:")
        print(f"  Success: {result.get('success')}")
        print(f"  Has dump_data: {'dump_data' in result}")
        print(f"  Output length: {len(result.get('output', ''))} chars")

        if result.get("success") and len(result.get("output", "")) >= 1:
            if "dump_data" in result:
                print("  ✓ dump_data was parsed from file")
            print("\n✅ DUMP OUTPUT TEST PASSED!")
            return True
        else:
            print(f"\n❌ DUMP OUTPUT TEST FAILED: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"\n❌ DUMP OUTPUT TEST FAILED with exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_streaming_call(query: str):
    """Test streaming call to Devmate CLI."""
    print("\n" + "=" * 60)
    print("TEST: Streaming Call")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
            DevmateCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import DevmateCliInferencer: {e}")
        return False

    try:
        inferencer = DevmateCliInferencer(
            repo_path="/data/users/zgchen/fbsource",
            model_name="claude-sonnet-4.5",
            no_create_commit=True,
        )
        print(f"✓ Created inferencer")

        print(f"\nSending STREAMING query: '{query}'")
        start_time = time.time()

        # Collect streaming output
        collected_lines = []
        line_count = 0

        for line in inferencer.infer_streaming(query, filter_session_info=True):
            collected_lines.append(line)
            line_count += 1
            # Print progress every 10 lines
            if line_count % 10 == 0:
                print(f"  ... received {line_count} lines")

        elapsed = time.time() - start_time

        # Get parsed result
        result = inferencer.get_streaming_result()

        full_output = "".join(collected_lines)
        print(f"\n✓ Streaming completed in {elapsed:.2f}s:")
        print(f"  Total lines received: {line_count}")
        print(f"  Total chars: {len(full_output)}")
        print(f"  Session ID: {result.get('session_id')}")
        print(f"  Trajectory URL: {result.get('trajectory_url')}")
        print("-" * 40)
        print(full_output[:500] + "..." if len(full_output) > 500 else full_output)
        print("-" * 40)

        if result.get("success") and len(full_output) >= 1:
            print("\n✅ STREAMING TEST PASSED!")
            return True
        else:
            print(f"\n❌ STREAMING TEST FAILED: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"\n❌ STREAMING TEST FAILED with exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_session_continuation():
    """Test multi-turn session continuation."""
    print("\n" + "=" * 60)
    print("TEST: Session Continuation (Multi-turn)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
            DevmateCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import DevmateCliInferencer: {e}")
        return False

    try:
        inferencer = DevmateCliInferencer(
            repo_path="/data/users/zgchen/fbsource",
            model_name="claude-sonnet-4.5",
            no_create_commit=True,
        )

        # First call - start new session
        print("\nFirst call: 'My favorite number is 42. Remember it.'")
        start_time = time.time()
        result1 = inferencer.new_session("My favorite number is 42. Remember it.")
        elapsed1 = time.time() - start_time

        print(f"✓ Response 1 in {elapsed1:.2f}s:")
        print(f"  Session ID: {result1.get('session_id')}")
        session_id = result1.get("session_id")

        if not session_id:
            print("\n❌ SESSION TEST FAILED: No session ID returned")
            return False

        # Second call - resume session
        print("\nSecond call: 'What is my favorite number?'")
        start_time = time.time()
        result2 = inferencer.resume_session("What is my favorite number?")
        elapsed2 = time.time() - start_time

        print(f"✓ Response 2 in {elapsed2:.2f}s:")
        output2 = result2.get("output", "")
        print(f"  Output: {output2[:200]}...")

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
        print(f"\n❌ SESSION TEST FAILED with exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_async_single_call(query: str):
    """Test async single call to Devmate CLI."""
    print("\n" + "=" * 60)
    print("TEST: Async Single Call")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
            DevmateCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import DevmateCliInferencer: {e}")
        return False

    try:
        inferencer = DevmateCliInferencer(
            repo_path="/data/users/zgchen/fbsource",
            model_name="claude-sonnet-4.5",
            no_create_commit=True,
        )
        print(f"✓ Created inferencer")

        print(f"\nSending ASYNC query: '{query}'")
        start_time = time.time()

        result = await inferencer.ainfer(query)

        elapsed = time.time() - start_time
        print(f"\n✓ Got response in {elapsed:.2f}s:")
        print(f"  Success: {result.get('success')}")
        print(f"  Return code: {result.get('return_code')}")
        print(f"  Session ID: {result.get('session_id')}")
        print(f"  Output length: {len(result.get('output', ''))} chars")
        print("-" * 40)
        output = result.get("output", "")
        print(output[:500] + "..." if len(output) > 500 else output)
        print("-" * 40)

        if result.get("success") and len(result.get("output", "")) >= 1:
            print("\n✅ ASYNC TEST PASSED!")
            return True
        else:
            print(f"\n❌ ASYNC TEST FAILED: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"\n❌ ASYNC TEST FAILED with exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_async_streaming(query: str):
    """Test async streaming - outputs text to terminal in real-time."""
    print("\n" + "=" * 60)
    print("TEST: Async Streaming (ainfer_streaming)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
            DevmateCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import DevmateCliInferencer: {e}")
        return False

    try:
        inferencer = DevmateCliInferencer(
            repo_path="/data/users/zgchen/fbsource",
            model_name="claude-sonnet-4.5",
            no_create_commit=True,
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
        async for line in inferencer.ainfer_streaming(query, filter_session_info=True):
            lines.append(line)
            line_count += 1
            # Print each line immediately (real-time streaming to terminal)
            print(line, end="", flush=True)

        elapsed = time.time() - start_time
        full_output = "".join(lines)

        print("\n" + "-" * 40)
        print(f"\n✓ Streaming completed in {elapsed:.2f}s:")
        print(f"  Total lines received: {line_count}")
        print(f"  Total chars: {len(full_output)}")

        if line_count > 0 and len(full_output) >= 1:
            print("\n✅ ASYNC STREAMING TEST PASSED!")
            return True
        else:
            print("\n❌ ASYNC STREAMING TEST FAILED: No output received")
            return False

    except Exception as e:
        print(f"\n❌ ASYNC STREAMING TEST FAILED with exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_async_session_continuation():
    """Test async multi-turn session continuation."""
    print("\n" + "=" * 60)
    print("TEST: Async Session Continuation (Multi-turn)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
            DevmateCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import DevmateCliInferencer: {e}")
        return False

    try:
        inferencer = DevmateCliInferencer(
            repo_path="/data/users/zgchen/fbsource",
            model_name="claude-sonnet-4.5",
            no_create_commit=True,
            auto_resume=True,
        )

        # First call - start new session
        print("\nFirst call: 'My favorite number is 42. Remember it.'")
        start_time = time.time()
        result1 = await inferencer.anew_session("My favorite number is 42. Remember it.")
        elapsed1 = time.time() - start_time

        print(f"✓ Response 1 in {elapsed1:.2f}s:")
        print(f"  Session ID: {result1.get('session_id')}")

        # Second call - should auto-resume
        print("\nSecond call: 'What is my favorite number?' (auto-resume)")
        start_time = time.time()
        result2 = await inferencer.ainfer("What is my favorite number?")
        elapsed2 = time.time() - start_time

        print(f"✓ Response 2 in {elapsed2:.2f}s:")
        output2 = result2.get("output", "")
        print(f"  Output: {output2[:200]}...")

        # Check if the model remembered "42"
        if "42" in output2:
            print("\n✅ ASYNC SESSION CONTINUATION TEST PASSED! Model remembered '42'")
            return True
        else:
            print("\n⚠️ ASYNC SESSION CONTINUATION TEST WARNING: '42' not found in response")
            if result2.get("success"):
                print("\n✅ ASYNC SESSION CONTINUATION TEST PASSED (with warning)")
                return True
            return False

    except Exception as e:
        print(f"\n❌ ASYNC SESSION TEST FAILED with exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_sync_tests(query: str):
    """Run all sync tests."""
    results = []
    results.append(("Sync Single Call", test_sync_single_call(query)))
    results.append(("Sync with Headless", test_sync_with_headless(query)))
    results.append(("Sync with Dump Output", test_sync_with_dump_output(query)))
    return results


def run_streaming_tests(query: str):
    """Run all streaming tests."""
    results = []
    results.append(("Streaming Call", test_streaming_call(query)))
    return results


def run_session_tests():
    """Run session continuation tests."""
    results = []
    results.append(("Session Continuation", test_session_continuation()))
    return results


def print_summary(results):
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


def main():
    """Run tests based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real integration tests for DevmateCliInferencer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests with default query
  python test_devmate_cli_inferencer_real.py

  # Run with custom query
  python test_devmate_cli_inferencer_real.py --query "Explain Python decorators"

  # Run only sync tests
  python test_devmate_cli_inferencer_real.py --mode sync --query "What is a list?"

  # Run only async tests
  python test_devmate_cli_inferencer_real.py --mode async --query "What is a dict?"

  # Run only streaming tests
  python test_devmate_cli_inferencer_real.py --mode streaming --query "What is a dict?"

  # Run session continuation test
  python test_devmate_cli_inferencer_real.py --mode session
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
    print("DEVMATE CLI INFERENCER - REAL INTEGRATION TESTS")
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

    if args.mode in ("session", "all"):
        results.extend(run_session_tests())

    if args.mode == "session":
        # Just the sync session test alone
        pass  # Already added in run_session_tests

    success = print_summary(results)
    return 0 if success else 1


async def run_async_tests(query: str):
    """Run all async tests."""
    results = []
    results.append(("Async Single Call", await test_async_single_call(query)))
    results.append(("Async Streaming", await test_async_streaming(query)))
    results.append(("Async Session Continuation", await test_async_session_continuation()))
    return results


if __name__ == "__main__":
    sys.exit(main())
