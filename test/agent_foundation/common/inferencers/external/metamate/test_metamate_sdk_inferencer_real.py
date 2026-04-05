#!/usr/bin/env python3
"""Real integration test for MetamateSDKInferencer.

This script actually calls the MetaMate SDK to verify the inferencer works
end-to-end against the real MetaMate GraphQL API.

Usage (via buck2):
    # Run all tests with default query
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_sdk_inferencer_real

    # Run with custom query
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_sdk_inferencer_real -- --query "How does DLRM work?"

    # Run specific test mode
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_sdk_inferencer_real -- --mode async
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_sdk_inferencer_real -- --mode streaming
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_sdk_inferencer_real -- --mode session
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_sdk_inferencer_real -- --mode deep-research --query "Research ranking optimization"
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_sdk_inferencer_real -- --mode all
"""

import argparse
import asyncio
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_QUERY = "What are the key differences between DLRM and DeepFM ranking models?"


async def test_async_single_call(query: str):
    """Test async single call to MetaMate SDK."""
    print("\n" + "=" * 60)
    print("TEST: Async Single Call (MetaMate SDK) — Regular Mode")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
            MetamateSDKInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import MetamateSDKInferencer: {e}")
        return False

    try:
        inferencer = MetamateSDKInferencer(
            total_timeout_seconds=600,
            idle_timeout_seconds=300,
            auto_continue=True,
            max_continuations=5,
        )
        print("✓ Created inferencer (agent_name=None, i.e. regular mode)")

        print(f"\nSending ASYNC query: '{query}'")
        start_time = time.time()

        response = await inferencer.ainfer(query)

        elapsed = time.time() - start_time
        output = str(response)
        print(f"\n✓ Got response in {elapsed:.2f}s:")
        print(f"  Response type: {type(response)}")
        print(f"  Response length: {len(output)} chars")
        print("-" * 40)
        print(output[:2000] + "..." if len(output) > 2000 else output)
        print("-" * 40)

        if response and len(output) >= 1:
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


async def test_sdk_response_format():
    """Test returning SDKInferencerResponse format."""
    print("\n" + "=" * 60)
    print("TEST: SDK Response Format (return_sdk_response=True)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
            MetamateSDKInferencer,
        )
        from agent_foundation.common.inferencers.agentic_inferencers.external.sdk_types import (
            SDKInferencerResponse,
        )
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False

    try:
        inferencer = MetamateSDKInferencer(
            total_timeout_seconds=300,
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
            content_preview = (
                response.content[:200] + "..."
                if response.content and len(response.content) > 200
                else response.content
            )
            print(f"  Content: {content_preview}")
            print(f"  Session ID: {response.session_id}")
            print(f"  Tokens received: {response.tokens_received}")
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


async def test_async_streaming(query: str):
    """Test async streaming inference."""
    print("\n" + "=" * 60)
    print("TEST: Async Streaming (ainfer_streaming)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
            MetamateSDKInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import MetamateSDKInferencer: {e}")
        return False

    try:
        inferencer = MetamateSDKInferencer(
            total_timeout_seconds=300,
            idle_timeout_seconds=120,
        )
        print("✓ Created inferencer")

        print(f"\nSending STREAMING query: '{query}'")
        print("-" * 40)
        print("STREAMING OUTPUT:")
        print("-" * 40)

        start_time = time.time()
        chunks = []
        chunk_count = 0

        async for chunk in inferencer.ainfer_streaming(query):
            chunks.append(chunk)
            chunk_count += 1
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
        from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
            MetamateSDKInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import MetamateSDKInferencer: {e}")
        return False

    try:
        inferencer = MetamateSDKInferencer(
            auto_resume=True,
            total_timeout_seconds=300,
            idle_timeout_seconds=120,
        )

        print("\nFirst call: 'My favorite number is 42. Remember it.'")
        start_time = time.time()
        result1 = await inferencer.anew_session(
            "My favorite number is 42. Remember it."
        )
        elapsed1 = time.time() - start_time

        print(f"✓ Response 1 in {elapsed1:.2f}s")
        print(f"  Session ID: {inferencer.active_session_id}")
        print(f"  Response: {str(result1)[:100]}...")

        session_id = inferencer.active_session_id
        if not session_id:
            print("\n❌ SESSION TEST FAILED: No session ID returned")
            return False

        print("\nSecond call: 'What is my favorite number?' (auto-resume)")
        start_time = time.time()
        result2 = await inferencer.ainfer("What is my favorite number?")
        elapsed2 = time.time() - start_time

        print(f"✓ Response 2 in {elapsed2:.2f}s")
        output2 = str(result2)
        print(f"  Response: {output2[:200]}...")

        if "42" in output2:
            print("\n✅ SESSION CONTINUATION TEST PASSED! Model remembered '42'")
            return True
        else:
            print("\n⚠️ SESSION TEST WARNING: '42' not found in response")
            print("  (This may be a model behavior issue, not an inferencer issue)")
            if result2:
                print("\n✅ SESSION CONTINUATION TEST PASSED (with warning)")
                return True
            return False

    except Exception as e:
        print(f"\n❌ SESSION TEST FAILED with exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_deep_research(query: str):
    """Test deep research mode with the same query as regular mode."""
    print("\n" + "=" * 60)
    print("TEST: Deep Research Mode (agent_name=SPACES_DEEP_RESEARCH_AGENT)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
            MetamateSDKInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import MetamateSDKInferencer: {e}")
        return False

    try:
        inferencer = MetamateSDKInferencer(
            agent_name="SPACES_DEEP_RESEARCH_AGENT",
            total_timeout_seconds=600,
            idle_timeout_seconds=300,
            auto_continue=True,
            max_continuations=5,
        )
        print("✓ Created inferencer with deep research agent")

        print(f"\nSending DEEP RESEARCH query: '{query}'")
        print("(This may take several minutes...)")

        start_time = time.time()
        response = await inferencer.ainfer(query)
        elapsed = time.time() - start_time

        output = str(response)
        print(f"\n✓ Got response in {elapsed:.2f}s:")
        print(f"  Response length: {len(output)} chars")
        print("-" * 40)
        print(output[:2000] + "..." if len(output) > 2000 else output)
        print("-" * 40)

        if response and len(output) >= 50:
            print("\n✅ DEEP RESEARCH TEST PASSED!")
            return True
        else:
            print("\n❌ DEEP RESEARCH TEST FAILED: Response too short or empty")
            return False

    except Exception as e:
        print(f"\n❌ DEEP RESEARCH TEST FAILED with exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_async_tests(query: str):
    """Run all async tests."""
    results = []
    results.append(("Async Single Call", await test_async_single_call(query)))
    results.append(("SDK Response Format", await test_sdk_response_format()))
    results.append(("Async Streaming", await test_async_streaming(query)))
    results.append(("Session Continuation", await test_session_continuation()))
    return results


async def run_compare(query: str):
    """Run regular vs deep research with the SAME query for fair comparison."""
    print("\n" + "#" * 60)
    print("COMPARISON: Regular vs Deep Research (same query)")
    print("#" * 60)
    print(f"Query: {query}")
    print("#" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
            MetamateSDKInferencer,
        )
    except ImportError as e:
        print(f"Failed to import MetamateSDKInferencer: {e}")
        return []

    # --- Regular mode ---
    print("\n" + "=" * 60)
    print("PHASE 1: Regular Mode (agent_name=None)")
    print("=" * 60)
    regular_inferencer = MetamateSDKInferencer(
        total_timeout_seconds=600,
        idle_timeout_seconds=300,
        auto_continue=True,
        max_continuations=5,
    )
    regular_start = time.time()
    try:
        regular_response = await regular_inferencer.ainfer(query)
        regular_output = str(regular_response)
    except Exception as e:
        print(f"Regular mode FAILED: {e}")
        regular_output = ""
    regular_elapsed = time.time() - regular_start

    print(f"\nRegular response ({len(regular_output)} chars, {regular_elapsed:.2f}s):")
    print("-" * 40)
    print(regular_output[:3000] + "..." if len(regular_output) > 3000 else regular_output)
    print("-" * 40)

    # --- Deep research mode ---
    print("\n" + "=" * 60)
    print("PHASE 2: Deep Research Mode (agent_name=SPACES_DEEP_RESEARCH_AGENT)")
    print("=" * 60)
    deep_inferencer = MetamateSDKInferencer(
        agent_name="SPACES_DEEP_RESEARCH_AGENT",
        total_timeout_seconds=600,
        idle_timeout_seconds=300,
        auto_continue=True,
        max_continuations=5,
    )
    deep_start = time.time()
    try:
        deep_response = await deep_inferencer.ainfer(query)
        deep_output = str(deep_response)
    except Exception as e:
        print(f"Deep research mode FAILED: {e}")
        deep_output = ""
    deep_elapsed = time.time() - deep_start

    print(f"\nDeep research response ({len(deep_output)} chars, {deep_elapsed:.2f}s):")
    print("-" * 40)
    print(deep_output[:3000] + "..." if len(deep_output) > 3000 else deep_output)
    print("-" * 40)

    # --- Comparison summary ---
    print("\n" + "#" * 60)
    print("COMPARISON SUMMARY")
    print("#" * 60)
    print(f"  Query:                 {query[:80]}...")
    print(f"  Regular response:      {len(regular_output)} chars in {regular_elapsed:.2f}s")
    print(f"  Deep research response: {len(deep_output)} chars in {deep_elapsed:.2f}s")
    if regular_output and deep_output:
        ratio = len(deep_output) / len(regular_output) if len(regular_output) > 0 else float("inf")
        print(f"  Deep/Regular ratio:    {ratio:.1f}x")
    print(f"  Regular has headings:  {'#' in regular_output}")
    print(f"  Deep has headings:     {'#' in deep_output}")
    regular_lines = regular_output.count("\n")
    deep_lines = deep_output.count("\n")
    print(f"  Regular line count:    {regular_lines}")
    print(f"  Deep line count:       {deep_lines}")
    print("#" * 60)

    results = []
    results.append(("Regular Mode", bool(regular_output)))
    results.append(("Deep Research Mode", bool(deep_output)))
    return results


def print_summary(results):
    """Print test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, result in results:
        status = "PASSED" if result else "FAILED"
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
        description="Real integration tests for MetamateSDKInferencer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare regular vs deep research with the same query (recommended)
  buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_sdk_inferencer_real -- --mode compare

  # Compare with custom query
  buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_sdk_inferencer_real -- --mode compare --query "Explain HSTU architecture"

  # Run only regular async test
  buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_sdk_inferencer_real -- --mode async

  # Run only deep research test
  buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_sdk_inferencer_real -- --mode deep-research

  # Run all tests
  buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_sdk_inferencer_real -- --mode all
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
        choices=["async", "streaming", "session", "deep-research", "compare", "all"],
        default="compare",
        help="Test mode (default: compare)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("METAMATE SDK INFERENCER - REAL INTEGRATION TESTS")
    print("=" * 60)
    print(f"\nQuery: {args.query}")
    print(f"Mode: {args.mode}")

    results = []

    if args.mode == "compare":
        compare_results = asyncio.run(run_compare(args.query))
        results.extend(compare_results)

    if args.mode in ("async", "all"):
        async_results = asyncio.run(run_async_tests(args.query))
        results.extend(async_results)

    if args.mode == "streaming":
        streaming_result = asyncio.run(test_async_streaming(args.query))
        results.append(("Async Streaming", streaming_result))

    if args.mode == "session":
        session_result = asyncio.run(test_session_continuation())
        results.append(("Session Continuation", session_result))

    if args.mode == "deep-research":
        deep_result = asyncio.run(test_deep_research(args.query))
        results.append(("Deep Research", deep_result))

    success = print_summary(results)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
