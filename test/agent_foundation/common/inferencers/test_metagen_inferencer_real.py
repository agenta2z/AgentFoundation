# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Real E2E integration test for MetagenApiInferencer (sync + async).

Tests the MetaGen API inferencer with dialog_completion mode for Claude models.

Usage (via buck2):
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_metagen_inferencer_real
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_metagen_inferencer_real -- --mode sync
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_metagen_inferencer_real -- --mode async
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_metagen_inferencer_real -- --query "Your question"
"""

import argparse
import asyncio
import sys
import time


def test_sync(inferencer, query: str) -> str:
    print(f"\n{'='*60}")
    print("TEST: Sync infer()")
    print(f"{'='*60}")
    print(f"  Model:  {inferencer.model_id}")
    print(f"  Key:    {inferencer.secret_key[:20]}...")
    print(f"  Query:  {query}")

    t0 = time.time()
    result = inferencer.infer(query)
    elapsed = time.time() - t0

    print(f"\n  Response ({elapsed:.1f}s):")
    print(f"  {result[:500]}")
    assert isinstance(result, str) and len(result) > 5, f"Bad result: {result!r}"
    print(f"\n  SYNC PASS")
    return result


async def test_async(inferencer, query: str) -> str:
    print(f"\n{'='*60}")
    print("TEST: Async ainfer()")
    print(f"{'='*60}")
    print(f"  Model:  {inferencer.model_id}")
    print(f"  Key:    {inferencer.secret_key[:20]}...")
    print(f"  Query:  {query}")

    t0 = time.time()
    result = await inferencer.ainfer(query)
    elapsed = time.time() - t0

    print(f"\n  Response ({elapsed:.1f}s):")
    print(f"  {result[:500]}")
    assert isinstance(result, str) and len(result) > 5, f"Bad result: {result!r}"
    print(f"\n  ASYNC PASS")
    return result


async def test_parallel_async(inferencer, queries: list) -> list:
    print(f"\n{'='*60}")
    print(f"TEST: Parallel async ainfer() x{len(queries)}")
    print(f"{'='*60}")

    t0 = time.time()
    results = await asyncio.gather(
        *[inferencer.ainfer(q) for q in queries]
    )
    elapsed = time.time() - t0

    for i, (q, r) in enumerate(zip(queries, results)):
        print(f"\n  [{i}] Q: {q}")
        print(f"      A: {r[:200]}")
        assert isinstance(r, str) and len(r) > 5, f"Bad result [{i}]: {r!r}"

    print(f"\n  Total time: {elapsed:.1f}s (avg {elapsed/len(queries):.1f}s/query)")
    print(f"\n  PARALLEL ASYNC PASS")
    return results


def main():
    parser = argparse.ArgumentParser(description="MetaGen API Inferencer E2E Test")
    parser.add_argument(
        "--mode", choices=["sync", "async", "parallel", "all"], default="all",
        help="Which test to run (default: all)",
    )
    parser.add_argument(
        "--query", default="What is Python? Answer in one sentence.",
        help="Query for sync/async tests",
    )
    args = parser.parse_args()

    from agent_foundation.common.inferencers.api_inferencers.metagen.metagen_api_inferencer import (
        MetagenApiInferencer,
    )

    print("Creating MetagenApiInferencer with defaults...")
    inferencer = MetagenApiInferencer()
    print(f"  model_id: {inferencer.model_id}")
    print(f"  secret_key: {inferencer.secret_key[:20]}...")

    if args.mode in ("sync", "all"):
        test_sync(inferencer, args.query)

    if args.mode in ("async", "all"):
        asyncio.run(test_async(inferencer, args.query))

    if args.mode in ("parallel", "all"):
        parallel_queries = [
            "What is Python? One sentence.",
            "What is Java? One sentence.",
            "What is Rust? One sentence.",
        ]
        asyncio.run(test_parallel_async(inferencer, parallel_queries))

    print(f"\n{'='*60}")
    print("ALL TESTS PASSED")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
