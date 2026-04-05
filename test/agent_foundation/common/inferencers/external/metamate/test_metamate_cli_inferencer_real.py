#!/usr/bin/env python3
"""Real integration test for MetamateCliInferencer.

This script actually calls the query_metamate buck target to verify the
CLI inferencer works end-to-end.

Usage (via buck2):
    # Run all tests with default query
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_cli_inferencer_real

    # Run with custom query
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_cli_inferencer_real -- --query "How does DLRM work?"

    # Run specific test mode
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_cli_inferencer_real -- --mode sync
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_cli_inferencer_real -- --mode streaming
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_cli_inferencer_real -- --mode deep-research --query "Research ranking optimization"
    buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_cli_inferencer_real -- --mode all
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


def test_sync_single_call(query: str):
    """Test synchronous single call to MetaMate CLI."""
    print("\n" + "=" * 60)
    print("TEST: Sync Single Call (MetaMate CLI)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
            MetamateCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import MetamateCliInferencer: {e}")
        return False

    try:
        inferencer = MetamateCliInferencer(
            timeout_seconds=120,
            idle_timeout_seconds=120,
        )
        print("✓ Created inferencer")

        print(f"\nSending SYNC query: '{query}'")
        start_time = time.time()

        result = inferencer.infer(query)

        elapsed = time.time() - start_time
        output = str(result)
        print(f"\n✓ Got response in {elapsed:.2f}s:")
        print(f"  Response type: {type(result)}")
        print(f"  Output length: {len(output)} chars")
        print(f"  Return code: {result.get('return_code')}")
        print("-" * 40)
        print(output[:500] + "..." if len(output) > 500 else output)
        print("-" * 40)

        if output and len(output) >= 1:
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


async def test_async_single_call(query: str):
    """Test async single call to MetaMate CLI."""
    print("\n" + "=" * 60)
    print("TEST: Async Single Call (MetaMate CLI)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
            MetamateCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import MetamateCliInferencer: {e}")
        return False

    try:
        inferencer = MetamateCliInferencer(
            timeout_seconds=120,
            idle_timeout_seconds=120,
        )
        print("✓ Created inferencer")

        print(f"\nSending ASYNC query: '{query}'")
        start_time = time.time()

        result = await inferencer.ainfer(query)

        elapsed = time.time() - start_time
        output = str(result)
        print(f"\n✓ Got response in {elapsed:.2f}s:")
        print(f"  Output length: {len(output)} chars")
        print("-" * 40)
        print(output[:500] + "..." if len(output) > 500 else output)
        print("-" * 40)

        if output and len(output) >= 1:
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


def test_streaming_call(query: str):
    """Test streaming call to MetaMate CLI."""
    print("\n" + "=" * 60)
    print("TEST: Streaming Call (MetaMate CLI)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
            MetamateCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import MetamateCliInferencer: {e}")
        return False

    try:
        inferencer = MetamateCliInferencer(
            timeout_seconds=120,
            idle_timeout_seconds=120,
        )
        print("✓ Created inferencer")

        print(f"\nSending STREAMING query: '{query}'")
        print("-" * 40)
        print("STREAMING OUTPUT:")
        print("-" * 40)

        start_time = time.time()
        lines = []
        line_count = 0

        for line in inferencer.infer_streaming(query):
            lines.append(line)
            line_count += 1
            print(line, end="", flush=True)

        elapsed = time.time() - start_time
        full_output = "".join(lines)

        print("\n" + "-" * 40)
        print(f"\n✓ Streaming completed in {elapsed:.2f}s:")
        print(f"  Total lines received: {line_count}")
        print(f"  Total chars: {len(full_output)}")

        if line_count > 0 and len(full_output) >= 1:
            print("\n✅ STREAMING TEST PASSED!")
            return True
        else:
            print("\n❌ STREAMING TEST FAILED: No output received")
            return False

    except Exception as e:
        print(f"\n❌ STREAMING TEST FAILED with exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_deep_research(query: str):
    """Test deep research mode via CLI with the provided query."""
    print("\n" + "=" * 60)
    print("TEST: Deep Research Mode (MetaMate CLI)")
    print("=" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
            MetamateCliInferencer,
        )
    except ImportError as e:
        print(f"❌ Failed to import MetamateCliInferencer: {e}")
        return False

    try:
        inferencer = MetamateCliInferencer(
            deep_research=True,
            timeout_seconds=600,
            idle_timeout_seconds=300,
        )
        print("✓ Created inferencer with deep_research=True")

        print(f"\nSending DEEP RESEARCH query: '{query}'")
        print("(This may take several minutes...)")

        start_time = time.time()
        result = inferencer.infer(query)
        elapsed = time.time() - start_time

        output = str(result)
        print(f"\n✓ Got response in {elapsed:.2f}s:")
        print(f"  Output length: {len(output)} chars")
        print("-" * 40)
        print(output[:2000] + "..." if len(output) > 2000 else output)
        print("-" * 40)

        if output and len(output) >= 50:
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


def run_compare(query: str):
    """Run regular vs deep research with the SAME query for fair comparison."""
    print("\n" + "#" * 60)
    print("COMPARISON: Regular vs Deep Research (MetaMate CLI, same query)")
    print("#" * 60)
    print(f"Query: {query}")
    print("#" * 60)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.metamate import (
            MetamateCliInferencer,
        )
    except ImportError as e:
        print(f"Failed to import MetamateCliInferencer: {e}")
        return []

    # --- Regular mode ---
    print("\n" + "=" * 60)
    print("PHASE 1: Regular Mode (deep_research=False)")
    print("=" * 60)
    regular_inferencer = MetamateCliInferencer(
        timeout_seconds=600,
        idle_timeout_seconds=300,
    )
    print(f"Command: {regular_inferencer.construct_command(query)}")
    regular_start = time.time()
    try:
        regular_result = regular_inferencer.infer(query)
        regular_output = str(regular_result)
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
    print("PHASE 2: Deep Research Mode (deep_research=True)")
    print("=" * 60)
    deep_inferencer = MetamateCliInferencer(
        deep_research=True,
        timeout_seconds=600,
        idle_timeout_seconds=300,
    )
    print(f"Command: {deep_inferencer.construct_command(query)}")
    deep_start = time.time()
    try:
        deep_result = deep_inferencer.infer(query)
        deep_output = str(deep_result)
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
    print("COMPARISON SUMMARY (CLI)")
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
    results.append(("Regular Mode (CLI)", bool(regular_output)))
    results.append(("Deep Research Mode (CLI)", bool(deep_output)))
    return results


def run_sync_tests(query: str):
    """Run all sync tests."""
    results = []
    results.append(("Sync Single Call", test_sync_single_call(query)))
    return results


def run_streaming_tests(query: str):
    """Run streaming tests."""
    results = []
    results.append(("Streaming Call", test_streaming_call(query)))
    return results


async def run_async_tests(query: str):
    """Run all async tests."""
    results = []
    results.append(("Async Single Call", await test_async_single_call(query)))
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
        description="Real integration tests for MetamateCliInferencer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_cli_inferencer_real

  # Run with custom query
  buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_cli_inferencer_real -- --query "How does ranking work?"

  # Run deep research test
  buck2 run @//mode/dbgo //rankevolve/test/agentic_foundation:test_metamate_cli_inferencer_real -- --mode deep-research
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
        choices=["sync", "async", "streaming", "deep-research", "compare", "all"],
        default="compare",
        help="Test mode (default: compare)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("METAMATE CLI INFERENCER - REAL INTEGRATION TESTS")
    print("=" * 60)
    print(f"\nQuery: {args.query}")
    print(f"Mode: {args.mode}")

    results = []

    if args.mode == "compare":
        results.extend(run_compare(args.query))

    if args.mode in ("sync", "all"):
        results.extend(run_sync_tests(args.query))

    if args.mode in ("streaming", "all"):
        results.extend(run_streaming_tests(args.query))

    if args.mode in ("async", "all"):
        async_results = asyncio.run(run_async_tests(args.query))
        results.extend(async_results)

    if args.mode == "deep-research":
        results.append(("Deep Research", test_deep_research(args.query)))

    success = print_summary(results)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
