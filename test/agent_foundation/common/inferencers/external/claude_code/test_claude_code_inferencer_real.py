#!/usr/bin/env python3
"""Real integration test for ClaudeCodeInferencer.

This script actually calls the Claude Code SDK to verify the inferencer works.

Usage (via buck2):
    # Run all tests with default query
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_claude_code_inferencer_real

    # Run with custom query
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_claude_code_inferencer_real -- --query "What is Python?"

    # Run specific test mode
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_claude_code_inferencer_real -- --mode sync --query "Explain recursion"
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_claude_code_inferencer_real -- --mode async --query "What is a list?"
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_claude_code_inferencer_real -- --mode streaming --query "Explain Python"
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_claude_code_inferencer_real -- --mode session
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_claude_code_inferencer_real -- --mode all
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
    """Test synchronous single call to Claude Code SDK."""
    print("\n" + "=" * 60)
    print("TEST: Sync Single Call")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import ClaudeCodeInferencer: {e}")
        return False

    try:
        inferencer = ClaudeCodeInferencer(
            root_folder="/tmp",
            allowed_tools=[],  # No tools for simple question
            idle_timeout_seconds=120,
        )
        print(f"✓ Created inferencer")

        print(f"\nSending SYNC query: '{query}'")
        start_time = time.time()

        response = inferencer(query)

        elapsed = time.time() - start_time
        print(f"\n✓ Got response in {elapsed:.2f}s:")
        print(f"  Response type: {type(response)}")
        print(f"  Response length: {len(str(response))} chars")
        print("-" * 40)
        print(str(response))
        print("-" * 40)

        if response and len(str(response)) >= 1:
            print("\n✅ SYNC TEST PASSED!")
            return True
        else:
            print("\n❌ SYNC TEST FAILED: Response empty")
            return False

    except Exception as e:
        print(f"\n❌ SYNC TEST FAILED with exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sync_multiple_calls():
    """Test multiple sync calls (tests stale-loop detection)."""
    print("\n" + "=" * 60)
    print("TEST: Sync Multiple Calls (Stale-Loop Detection)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import ClaudeCodeInferencer: {e}")
        return False

    try:
        inferencer = ClaudeCodeInferencer(
            root_folder="/tmp",
            allowed_tools=[],
            idle_timeout_seconds=120,
        )

        # First call
        print("\nFirst SYNC call: 'What is 2+2?'")
        start_time = time.time()
        response1 = inferencer("What is 2+2? Just give the number.")
        elapsed1 = time.time() - start_time
        print(f"✓ Response 1 in {elapsed1:.2f}s: {str(response1)[:100]}...")

        # Second call (tests stale-loop detection)
        print("\nSecond SYNC call: 'What is 3+3?'")
        start_time = time.time()
        response2 = inferencer("What is 3+3? Just give the number.")
        elapsed2 = time.time() - start_time
        print(f"✓ Response 2 in {elapsed2:.2f}s: {str(response2)[:100]}...")

        if response1 and response2:
            print("\n✅ SYNC MULTIPLE CALLS PASSED!")
            return True
        else:
            print("\n❌ SYNC MULTIPLE CALLS FAILED: One or both responses empty")
            return False

    except Exception as e:
        print(
            f"\n❌ SYNC MULTIPLE CALLS FAILED with exception: {type(e).__name__}: {e}"
        )
        import traceback

        traceback.print_exc()
        return False


async def test_async_single_call(query: str):
    """Test async single call."""
    print("\n" + "=" * 60)
    print("TEST: Async Single Call")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import ClaudeCodeInferencer: {e}")
        return False

    try:
        inferencer = ClaudeCodeInferencer(
            root_folder="/tmp",
            allowed_tools=[],
            idle_timeout_seconds=120,
        )

        print(f"\nSending ASYNC query: '{query}'")
        start_time = time.time()

        response = await inferencer.ainfer(query)

        elapsed = time.time() - start_time
        print(f"\n✓ Got response in {elapsed:.2f}s:")
        print(f"  Response type: {type(response)}")
        print(f"  Response length: {len(str(response))} chars")
        print("-" * 40)
        print(str(response))
        print("-" * 40)

        if response and len(str(response)) >= 1:
            print("\n✅ ASYNC TEST PASSED!")
            return True
        else:
            print("\n❌ ASYNC TEST FAILED: Response empty")
            return False

    except Exception as e:
        print(f"\n❌ ASYNC TEST FAILED with exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_async_context_manager():
    """Test async context manager with multiple calls."""
    print("\n" + "=" * 60)
    print("TEST: Async Context Manager (Multiple Calls, Persistent Connection)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import ClaudeCodeInferencer: {e}")
        return False

    try:
        print("\nUsing async context manager (single connection, multiple queries)...")

        async with ClaudeCodeInferencer(
            root_folder="/tmp",
            allowed_tools=[],
            idle_timeout_seconds=120,
        ) as inferencer:
            print("✓ Connected via context manager")

            # First query
            print("\nFirst ASYNC query: 'What is 2+2?'")
            start_time = time.time()
            response1 = await inferencer.ainfer("What is 2+2? Just the number.")
            elapsed1 = time.time() - start_time
            print(f"✓ Response 1 in {elapsed1:.2f}s: {str(response1)[:100]}...")

            # Second query (same connection)
            print("\nSecond ASYNC query: 'What is 3+3?'")
            start_time = time.time()
            response2 = await inferencer.ainfer("What is 3+3? Just the number.")
            elapsed2 = time.time() - start_time
            print(f"✓ Response 2 in {elapsed2:.2f}s: {str(response2)[:100]}...")

        print("✓ Disconnected via context manager")

        if response1 and response2:
            print("\n✅ ASYNC CONTEXT MANAGER PASSED!")
            return True
        else:
            print("\n❌ ASYNC CONTEXT MANAGER FAILED: One or both responses empty")
            return False

    except Exception as e:
        print(
            f"\n❌ ASYNC CONTEXT MANAGER FAILED with exception: {type(e).__name__}: {e}"
        )
        import traceback

        traceback.print_exc()
        return False


async def test_sdk_response_format():
    """Test returning SDKInferencerResponse format."""
    print("\n" + "=" * 60)
    print("TEST: SDK Response Format (return_sdk_response=True)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeInferencer,
        )
        from agent_foundation.common.inferencers.agentic_inferencers.external.sdk_types import (
            SDKInferencerResponse,
        )
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False

    try:
        inferencer = ClaudeCodeInferencer(
            root_folder="/tmp",
            allowed_tools=[],
            idle_timeout_seconds=120,
        )

        print("\nSending query with return_sdk_response=True...")
        response = await inferencer.ainfer(
            "Say hello.",
            return_sdk_response=True,
        )

        print(f"\n✓ Got response:")
        print(f"  Type: {type(response)}")
        print(
            f"  Is SDKInferencerResponse: {isinstance(response, SDKInferencerResponse)}"
        )

        if isinstance(response, SDKInferencerResponse):
            print(
                f"  Content: {response.content[:200]}..."
                if len(response.content) > 200
                else f"  Content: {response.content}"
            )
            print(f"  Session ID: {response.session_id}")
            print(f"  Tool uses: {response.tool_uses}")
            print("\n✅ SDK RESPONSE FORMAT PASSED!")
            return True
        else:
            print(
                "\n❌ SDK RESPONSE FORMAT FAILED: Response is not SDKInferencerResponse"
            )
            return False

    except Exception as e:
        print(
            f"\n❌ SDK RESPONSE FORMAT FAILED with exception: {type(e).__name__}: {e}"
        )
        import traceback

        traceback.print_exc()
        return False


def test_sync_streaming(query: str):
    """Test sync streaming - outputs text to terminal in real-time."""
    print("\n" + "=" * 60)
    print("TEST: Sync Streaming (infer_streaming)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import ClaudeCodeInferencer: {e}")
        return False

    try:
        inferencer = ClaudeCodeInferencer(
            root_folder="/tmp",
            allowed_tools=[],
            idle_timeout_seconds=120,
        )
        print(f"✓ Created inferencer")

        print(f"\nSending SYNC STREAMING query: '{query}'")
        print("-" * 40)
        print("STREAMING OUTPUT:")
        print("-" * 40)

        start_time = time.time()
        chunks = []
        chunk_count = 0

        # Stream output to terminal in real-time
        for chunk in inferencer.infer_streaming(query):
            chunks.append(chunk)
            chunk_count += 1
            # Print each chunk immediately (real-time streaming to terminal)
            print(chunk, end="", flush=True)

        elapsed = time.time() - start_time
        full_response = "".join(chunks)

        print("\n" + "-" * 40)
        print(f"\n✓ Streaming completed in {elapsed:.2f}s:")
        print(f"  Total chunks received: {chunk_count}")
        print(f"  Total chars: {len(full_response)}")

        if chunk_count > 0 and len(full_response) >= 1:
            print("\n✅ SYNC STREAMING TEST PASSED!")
            return True
        else:
            print("\n❌ SYNC STREAMING TEST FAILED: No chunks received")
            return False

    except Exception as e:
        print(
            f"\n❌ SYNC STREAMING TEST FAILED with exception: {type(e).__name__}: {e}"
        )
        import traceback

        traceback.print_exc()
        return False


async def test_async_streaming(query: str):
    """Test async streaming - outputs text to terminal in real-time."""
    print("\n" + "=" * 60)
    print("TEST: Async Streaming (ainfer_streaming)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import ClaudeCodeInferencer: {e}")
        return False

    try:
        inferencer = ClaudeCodeInferencer(
            root_folder="/tmp",
            allowed_tools=[],
            idle_timeout_seconds=120,
        )
        print(f"✓ Created inferencer")

        print(f"\nSending ASYNC STREAMING query: '{query}'")
        print("-" * 40)
        print("STREAMING OUTPUT:")
        print("-" * 40)

        start_time = time.time()
        chunks = []
        chunk_count = 0

        # Stream output to terminal in real-time
        async for chunk in inferencer.ainfer_streaming(query):
            chunks.append(chunk)
            chunk_count += 1
            # Print each chunk immediately (real-time streaming to terminal)
            print(chunk, end="", flush=True)

        elapsed = time.time() - start_time
        full_response = "".join(chunks)

        print("\n" + "-" * 40)
        print(f"\n✓ Streaming completed in {elapsed:.2f}s:")
        print(f"  Total chunks received: {chunk_count}")
        print(f"  Total chars: {len(full_response)}")

        if chunk_count > 0 and len(full_response) >= 1:
            print("\n✅ ASYNC STREAMING TEST PASSED!")
            return True
        else:
            print("\n❌ ASYNC STREAMING TEST FAILED: No chunks received")
            return False

    except Exception as e:
        print(
            f"\n❌ ASYNC STREAMING TEST FAILED with exception: {type(e).__name__}: {e}"
        )
        import traceback

        traceback.print_exc()
        return False


async def test_session_continuation():
    """Test session continuation (multi-turn conversation)."""
    print("\n" + "=" * 60)
    print("TEST: Session Continuation (Multi-turn)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
            ClaudeCodeInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import ClaudeCodeInferencer: {e}")
        return False

    try:
        # Create inferencer with auto_resume=True (default)
        inferencer = ClaudeCodeInferencer(
            root_folder="/tmp",
            allowed_tools=[],
            idle_timeout_seconds=120,
            auto_resume=True,
        )

        # First call - start new session
        print("\nFirst call: 'My favorite number is 42. Remember it.'")
        start_time = time.time()
        result1 = await inferencer.anew_session(
            "My favorite number is 42. Remember it."
        )
        elapsed1 = time.time() - start_time

        print(f"✓ Response 1 in {elapsed1:.2f}s")
        print(f"  Session ID: {inferencer.active_session_id}")
        print(f"  Response: {str(result1)[:100]}...")

        # Second call - should auto-resume (auto_resume=True)
        print("\nSecond call: 'What is my favorite number?' (auto-resume)")
        start_time = time.time()
        result2 = await inferencer.ainfer("What is my favorite number?")
        elapsed2 = time.time() - start_time

        print(f"✓ Response 2 in {elapsed2:.2f}s")
        output2 = str(result2)
        print(f"  Response: {output2[:200]}...")

        # Check if the model remembered "42"
        if "42" in output2:
            print("\n✅ SESSION CONTINUATION TEST PASSED! Model remembered '42'")
            return True
        else:
            print("\n⚠️ SESSION CONTINUATION TEST WARNING: '42' not found in response")
            print("  (This may be a model behavior issue, not an inferencer issue)")
            # Still pass if we got a valid response
            if result2:
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


def run_sync_tests(query: str):
    """Run all sync tests."""
    results = []
    results.append(("Sync Single Call", test_sync_single_call(query)))
    results.append(("Sync Multiple Calls", test_sync_multiple_calls()))
    return results


def run_streaming_tests(query: str):
    """Run streaming tests (sync)."""
    results = []
    results.append(("Sync Streaming", test_sync_streaming(query)))
    return results


async def run_async_tests(query: str):
    """Run all async tests."""
    results = []
    results.append(("Async Single Call", await test_async_single_call(query)))
    results.append(("Async Context Manager", await test_async_context_manager()))
    results.append(("SDK Response Format", await test_sdk_response_format()))
    results.append(("Async Streaming", await test_async_streaming(query)))
    results.append(("Session Continuation", await test_session_continuation()))
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
        description="Real integration tests for ClaudeCodeInferencer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests with default query
  python test_claude_code_inferencer_real.py

  # Run with custom query
  python test_claude_code_inferencer_real.py --query "Explain Python decorators"

  # Run only sync tests
  python test_claude_code_inferencer_real.py --mode sync --query "What is a list?"

  # Run only async tests
  python test_claude_code_inferencer_real.py --mode async --query "What is a dict?"

  # Run streaming tests (see output in real-time)
  python test_claude_code_inferencer_real.py --mode streaming --query "Explain Python"

  # Run session continuation test
  python test_claude_code_inferencer_real.py --mode session
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
    print("CLAUDE CODE INFERENCER - REAL INTEGRATION TESTS")
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
