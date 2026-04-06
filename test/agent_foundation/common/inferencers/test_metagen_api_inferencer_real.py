

"""Real E2E integration tests for MetagenApiInferencer.

Tests sync, async, streaming, system_prompt, set_messages, and parallel calls
with real MetaGen API. Requires API key access.

Usage (via buck2):
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_metagen_api_inferencer_real
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_metagen_api_inferencer_real -- --mode sync
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_metagen_api_inferencer_real -- --mode async
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_metagen_api_inferencer_real -- --mode streaming
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_metagen_api_inferencer_real -- --mode all
"""

import argparse
import asyncio
import sys
import time
import traceback


def _make_inferencer(**kwargs):
    from agent_foundation.common.inferencers.api_inferencers.metagen.metagen_api_inferencer import (
        MetagenApiInferencer,
    )

    return MetagenApiInferencer(**kwargs)


# ---------------------------------------------------------------------------
# Sync Tests
# ---------------------------------------------------------------------------


def test_sync_simple_query():
    """Sync infer() with a simple factual question."""
    print("\n--- test_sync_simple_query ---")
    inf = _make_inferencer()
    print(f"  Model: {inf.model_id}")

    t0 = time.time()
    result = inf.infer("What is Python? Answer in one sentence.")
    elapsed = time.time() - t0

    print(f"  Response ({elapsed:.1f}s): {result[:300]}")
    assert isinstance(result, str) and len(result) > 10, f"Bad result: {result!r}"
    assert "python" in result.lower() or "programming" in result.lower(), (
        f"Response doesn't mention Python: {result!r}"
    )
    print("  PASS")


# ---------------------------------------------------------------------------
# Async Tests
# ---------------------------------------------------------------------------


async def test_async_simple_query():
    """Async ainfer() with a math question."""
    print("\n--- test_async_simple_query ---")
    inf = _make_inferencer()

    t0 = time.time()
    result = await inf.ainfer("What is 2+2? Just the number.")
    elapsed = time.time() - t0

    print(f"  Response ({elapsed:.1f}s): {result[:100]}")
    assert isinstance(result, str) and len(result) > 0, f"Bad result: {result!r}"
    assert "4" in result, f"Expected '4' in response: {result!r}"
    print("  PASS")


async def test_parallel_async():
    """Parallel async ainfer() calls."""
    print("\n--- test_parallel_async ---")
    inf = _make_inferencer()
    queries = [
        "What is Python? One sentence.",
        "What is Java? One sentence.",
        "What is Rust? One sentence.",
    ]

    t0 = time.time()
    results = await asyncio.gather(*[inf.ainfer(q) for q in queries])
    elapsed = time.time() - t0

    for i, (q, r) in enumerate(zip(queries, results)):
        print(f"  [{i}] Q: {q}")
        print(f"      A: {r[:150]}")
        assert isinstance(r, str) and len(r) > 5, f"Bad result [{i}]: {r!r}"

    print(f"  Total: {elapsed:.1f}s (avg {elapsed / len(queries):.1f}s/query)")
    print("  PASS")


# ---------------------------------------------------------------------------
# Streaming Tests
# ---------------------------------------------------------------------------


async def test_streaming():
    """Streaming ainfer_streaming() collects chunks into full response."""
    print("\n--- test_streaming ---")
    inf = _make_inferencer()

    t0 = time.time()
    chunks = []
    async for chunk in inf.ainfer_streaming("Tell me a short joke."):
        chunks.append(chunk)
    elapsed = time.time() - t0

    full = "".join(chunks)
    print(f"  Chunks: {len(chunks)}")
    print(f"  Response ({elapsed:.1f}s): {full[:300]}")
    assert len(chunks) > 0, "No chunks received"
    assert len(full) > 10, f"Response too short: {full!r}"
    print("  PASS")


async def test_system_prompt():
    """Streaming with system_prompt constraining the response."""
    print("\n--- test_system_prompt ---")
    inf = _make_inferencer(
        system_prompt="Always respond in exactly one word. No punctuation.",
    )

    t0 = time.time()
    chunks = []
    async for chunk in inf.ainfer_streaming("What is Python?"):
        chunks.append(chunk)
    elapsed = time.time() - t0

    full = "".join(chunks).strip()
    print(f"  Response ({elapsed:.1f}s): {full!r}")
    assert len(full) < 50, f"Response should be very short: {full!r}"
    print("  PASS")


async def test_set_messages_multi_turn():
    """Multi-turn conversation via set_messages."""
    print("\n--- test_set_messages_multi_turn ---")
    inf = _make_inferencer()
    inf.set_messages(
        [
            {"role": "user", "content": "Remember the number 42."},
            {"role": "assistant", "content": "I'll remember the number 42."},
            {
                "role": "user",
                "content": "What number did I ask you to remember? Just the number.",
            },
        ]
    )

    t0 = time.time()
    chunks = []
    async for chunk in inf.ainfer_streaming("placeholder"):
        chunks.append(chunk)
    elapsed = time.time() - t0

    full = "".join(chunks)
    print(f"  Response ({elapsed:.1f}s): {full[:100]}")
    assert "42" in full, f"Expected '42' in response: {full!r}"
    print("  PASS")


async def test_custom_temperature():
    """Low temperature should produce consistent results."""
    print("\n--- test_custom_temperature ---")
    inf = _make_inferencer(temperature=0.0, max_tokens=32)

    results = []
    for i in range(2):
        result = await inf.ainfer("What is 1+1? Just the number, nothing else.")
        results.append(result.strip())
        print(f"  Run {i + 1}: {results[-1]!r}")

    assert "2" in results[0], f"Expected '2' in response: {results[0]!r}"
    assert "2" in results[1], f"Expected '2' in response: {results[1]!r}"
    print("  PASS")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

SYNC_TESTS = [test_sync_simple_query]
ASYNC_TESTS = [test_async_simple_query, test_parallel_async]
STREAMING_TESTS = [
    test_streaming,
    test_system_prompt,
    test_set_messages_multi_turn,
    test_custom_temperature,
]


def run_tests(tests, label):
    print(f"\n{'=' * 60}")
    print(f"  {label} ({len(tests)} tests)")
    print(f"{'=' * 60}")

    passed, failed = 0, 0
    for test_fn in tests:
        try:
            if asyncio.iscoroutinefunction(test_fn):
                asyncio.run(test_fn())
            else:
                test_fn()
            passed += 1
        except Exception:
            failed += 1
            print(f"  FAIL: {test_fn.__name__}")
            traceback.print_exc()

    print(f"\n  Results: {passed} passed, {failed} failed")
    return failed


def main():
    parser = argparse.ArgumentParser(
        description="MetagenApiInferencer Real E2E Tests",
    )
    parser.add_argument(
        "--mode",
        choices=["sync", "async", "streaming", "all"],
        default="all",
        help="Which tests to run (default: all)",
    )
    args = parser.parse_args()

    total_failures = 0

    if args.mode in ("sync", "all"):
        total_failures += run_tests(SYNC_TESTS, "Sync Tests")

    if args.mode in ("async", "all"):
        total_failures += run_tests(ASYNC_TESTS, "Async Tests")

    if args.mode in ("streaming", "all"):
        total_failures += run_tests(STREAMING_TESTS, "Streaming Tests")

    print(f"\n{'=' * 60}")
    if total_failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILURES: {total_failures}")
    print(f"{'=' * 60}")

    sys.exit(1 if total_failures else 0)


if __name__ == "__main__":
    main()
