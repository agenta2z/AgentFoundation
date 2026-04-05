# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Real E2E integration test for BreakdownThenAggregateInferencer.

Tests the full diamond pipeline: breakdown -> parallel workers -> aggregate.
Supports multiple backends: MetaGen API, Devmate SDK, or Claude Code SDK.

Usage (via buck2):
    # Default: auto-detect best available backend
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_breakdown_then_aggregate_real

    # Use specific backend
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_breakdown_then_aggregate_real -- --backend metagen
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_breakdown_then_aggregate_real -- --backend devmate
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_breakdown_then_aggregate_real -- --backend claude_code

    # Custom query
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_breakdown_then_aggregate_real -- \
        --query "What are the key challenges in building reliable distributed systems?"

    # Control breakdown count
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_breakdown_then_aggregate_real -- --max-breakdown 3

    # Run specific test
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_breakdown_then_aggregate_real -- --mode simple
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_breakdown_then_aggregate_real -- --mode bta
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_breakdown_then_aggregate_real -- --mode all
"""

import argparse
import shutil
import sys
import tempfile
import time

DEFAULT_QUERY = (
    "What are the tradeoffs between microservice and monolith architectures "
    "for a startup building a social media platform?"
)


# =============================================================================
# Backend creation helpers
# =============================================================================


def _try_create_metagen_inferencer(**kwargs):
    """Try to create a MetagenApiInferencer. Returns (inferencer, name) or None."""
    try:
        from agent_foundation.common.inferencers.api_inferencers.metagen.metagen_api_inferencer import (
            MetagenApiInferencer,
        )

        inf = MetagenApiInferencer(**kwargs)
        return inf, "MetaGen"
    except Exception as e:
        print(f"  MetaGen not available: {type(e).__name__}: {e}")
        return None


def _try_create_devmate_inferencer(**kwargs):
    """Try to create a DevmateSDKInferencer. Returns (inferencer, name) or None."""
    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_sdk_inferencer import (
            DevmateSDKInferencer,
        )

        defaults = {
            "target_path": "/data/users/zgchen/fbsource",
            "config_file_path": "config.dual_agent.md",
            "usecase": "dual_agent_coding",
            "total_timeout_seconds": 300,
            "idle_timeout_seconds": 120,
        }
        # input_preprocessor is on InferencerBase, safe to pass through
        for k, v in kwargs.items():
            defaults[k] = v
        inf = DevmateSDKInferencer(**defaults)
        return inf, "Devmate"
    except Exception as e:
        print(f"  Devmate not available: {type(e).__name__}: {e}")
        return None


def _try_create_claude_code_inferencer(**kwargs):
    """Try to create a ClaudeCodeInferencer. Returns (inferencer, name) or None."""
    try:
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_inferencer import (
            ClaudeCodeInferencer,
        )

        defaults = {
            "target_path": "/data/users/zgchen/fbsource",
            "idle_timeout_seconds": 120,
            "allowed_tools": ["Read"],
            "enable_shell": False,
        }
        for k, v in kwargs.items():
            defaults[k] = v
        inf = ClaudeCodeInferencer(**defaults)
        return inf, "ClaudeCode"
    except Exception as e:
        print(f"  Claude Code not available: {type(e).__name__}: {e}")
        return None


def create_inferencer(backend: str, **kwargs):
    """Create an inferencer for the given backend.

    Args:
        backend: "metagen", "devmate", "claude_code", or "auto"
        **kwargs: Backend-specific arguments

    Returns:
        (inferencer, backend_name) tuple, or raises RuntimeError
    """
    creators = {
        "metagen": _try_create_metagen_inferencer,
        "devmate": _try_create_devmate_inferencer,
        "claude_code": _try_create_claude_code_inferencer,
    }

    if backend == "auto":
        for name in ["metagen", "devmate", "claude_code"]:
            result = creators[name](**kwargs)
            if result is not None:
                print(f"  Auto-selected backend: {result[1]}")
                return result
        raise RuntimeError("No backend available (tried MetaGen, Devmate, Claude Code)")

    result = creators[backend](**kwargs)
    if result is None:
        raise RuntimeError(f"Backend '{backend}' not available")
    return result


# =============================================================================
# Tests
# =============================================================================


def test_simple_query(query: str, backend: str = "auto"):
    """Test that the backend can handle a simple query."""
    print("\n" + "=" * 70)
    print(f"TEST: Simple Query (backend={backend})")
    print("=" * 70)

    try:
        inferencer, backend_name = create_inferencer(backend)
        print(f"  Backend: {backend_name}")
        print(f"  Query: {query[:80]}...")

        start = time.time()
        result = inferencer.infer(query)
        elapsed = time.time() - start

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Response length: {len(str(result))} chars")
        print("-" * 40)
        print(str(result)[:500])
        print("-" * 40)

        if result and len(str(result)) > 10:
            print(f"\n  PASSED: {backend_name} backend works")
            return True
        else:
            print("\n  FAILED: Response too short or empty")
            return False

    except Exception as e:
        print(f"\n  FAILED: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def _robust_breakdown_parser(raw_output, max_queries=10):
    """Parse numbered list from potentially noisy agentic output.

    Handles output wrapped in Thoughts:/Action: blocks, code fences,
    and other agentic formatting that parse_numbered_list may miss.
    """
    import re

    text = str(raw_output)
    queries = []

    # Try to find numbered items anywhere in the text
    for match in re.finditer(r'^\s*(\d+)[.)]\s+(.+)', text, re.MULTILINE):
        item = match.group(2).strip()
        # Skip items that look like agentic formatting
        if item and not item.startswith("```") and len(item) > 5:
            queries.append(item)

    if not queries:
        # Fallback: try bullet points
        for match in re.finditer(r'^\s*[-*]\s+(.+)', text, re.MULTILINE):
            item = match.group(1).strip()
            if item and len(item) > 5:
                queries.append(item)

    return queries[:max_queries]


def test_bta_full_pipeline(query: str, backend: str = "auto", max_breakdown: int = 3):
    """Test full BreakdownThenAggregateInferencer pipeline.

    Pipeline:
        1. Breakdown: Split query into sub-questions
        2. Workers: Research each sub-question in parallel
        3. Aggregate: Synthesize a unified proposal from all research
    """
    print("\n" + "=" * 70)
    print(f"TEST: Full BTA Pipeline (backend={backend}, max_breakdown={max_breakdown})")
    print("=" * 70)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
            BreakdownThenAggregateInferencer,
        )
    except ImportError as e:
        print(f"  FAIL: Cannot import BreakdownThenAggregateInferencer: {e}")
        return False

    tmpdir = tempfile.mkdtemp(prefix="bta_e2e_")
    try:
        # Verify backend works first
        breakdown_inf, backend_name = create_inferencer(backend)
        print(f"  Backend: {backend_name}")
        print(f"  Query: {query}")

        # --- Step 1: Create breakdown inferencer ---
        # Wrap it to prepend breakdown instructions
        breakdown_inferencer, _ = create_inferencer(
            backend,
            input_preprocessor=lambda q: (
                "You are a research planning assistant. "
                "Break down the following question into exactly "
                f"{max_breakdown} specific, focused sub-questions that together "
                "would comprehensively answer the original question.\n\n"
                "Return ONLY a numbered list (1. ..., 2. ..., 3. ...) "
                "with no preamble or explanation.\n\n"
                f"Question: {q}"
            ),
        )
        print("  Created breakdown_inferencer")

        # --- Step 2: Create worker factory ---
        workers_created = []

        def worker_factory(sub_query, index):
            """Create a worker for deep research on a sub-question."""
            worker, _ = create_inferencer(
                backend,
                input_preprocessor=lambda q: (
                    "You are a thorough research analyst. "
                    "Provide a detailed, well-structured analysis of this specific question. "
                    "Include concrete examples, data points, and actionable insights.\n\n"
                    f"Research question: {q}"
                ),
            )
            workers_created.append((index, sub_query))
            return worker

        print("  Created worker_factory")

        # --- Step 3: Create aggregator ---
        def aggregator_prompt_builder(worker_results, original_query=""):
            """Build a prompt that synthesizes all worker results."""
            parts = [
                "You are a senior strategist. Based on the following research findings, "
                "synthesize a comprehensive, actionable proposal that addresses the "
                "original question.\n\n"
                f"## Original Question\n{original_query}\n\n"
                "## Research Findings\n"
            ]
            for i, r in enumerate(worker_results):
                parts.append(f"### Finding {i + 1}\n{r}\n\n")
            parts.append(
                "## Instructions\n"
                "Synthesize these findings into a unified proposal with:\n"
                "1. Executive summary (2-3 sentences)\n"
                "2. Key recommendations (numbered list)\n"
                "3. Tradeoffs and considerations\n"
                "4. Suggested next steps\n"
            )
            return "".join(parts)

        aggregator_inferencer, _ = create_inferencer(backend)
        print("  Created aggregator_inferencer")

        # --- Step 4: Wire it all together via BTA ---
        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown_inferencer,
            worker_factory=worker_factory,
            aggregator_inferencer=aggregator_inferencer,
            aggregator_prompt_builder=aggregator_prompt_builder,
            breakdown_parser=lambda raw: _robust_breakdown_parser(raw, max_breakdown),
            max_breakdown=max_breakdown,
            checkpoint_dir=tmpdir,
        )
        print("  Created BreakdownThenAggregateInferencer")

        # --- Step 5: Run the full pipeline ---
        print("\n  Running full pipeline...")
        overall_start = time.time()

        result = bta.infer(query)

        overall_elapsed = time.time() - overall_start

        # --- Step 6: Validate results ---
        print(f"\n  Pipeline completed in {overall_elapsed:.2f}s")
        print(f"  Workers created: {len(workers_created)}")
        for idx, sq in workers_created:
            print(f"    Worker {idx}: {sq[:60]}...")
        print(f"  Result type: {type(result)}")
        print(f"  Result length: {len(str(result))} chars")
        print("\n" + "=" * 70)
        print("FINAL AGGREGATED PROPOSAL:")
        print("=" * 70)
        print(str(result))
        print("=" * 70)

        # Validation
        result_str = str(result)
        checks_passed = 0
        checks_total = 4

        if len(workers_created) > 0:
            print(f"\n  CHECK 1/4: Workers > 0 ({len(workers_created)}) - PASS")
            checks_passed += 1
        else:
            print(f"\n  CHECK 1/4: Workers > 0 ({len(workers_created)}) - FAIL")

        if len(workers_created) <= max_breakdown:
            print(
                f"  CHECK 2/4: Workers <= max ({len(workers_created)} <= {max_breakdown}) - PASS"
            )
            checks_passed += 1
        else:
            print(
                f"  CHECK 2/4: Workers <= max ({len(workers_created)} > {max_breakdown}) - FAIL"
            )

        if len(result_str) > 100:
            print(f"  CHECK 3/4: Result > 100 chars ({len(result_str)}) - PASS")
            checks_passed += 1
        else:
            print(f"  CHECK 3/4: Result > 100 chars ({len(result_str)}) - FAIL")

        if isinstance(result, str):
            print("  CHECK 4/4: Result is string - PASS")
            checks_passed += 1
        else:
            print(f"  CHECK 4/4: Result is {type(result).__name__} not str - FAIL")

        if checks_passed == checks_total:
            print(f"\n  PASSED: All {checks_total} checks passed")
            return True
        else:
            print(f"\n  FAILED: {checks_passed}/{checks_total} checks passed")
            return False

    except Exception as e:
        print(f"\n  FAILED: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_bta_no_aggregator(query: str, backend: str = "auto", max_breakdown: int = 2):
    """Test BTA pipeline without aggregator - returns raw worker results."""
    print("\n" + "=" * 70)
    print(f"TEST: BTA Without Aggregator (backend={backend})")
    print("=" * 70)

    try:
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
            BreakdownThenAggregateInferencer,
        )
    except ImportError as e:
        print(f"  FAIL: Cannot import: {e}")
        return False

    tmpdir = tempfile.mkdtemp(prefix="bta_noagg_")
    try:
        breakdown_inferencer, backend_name = create_inferencer(
            backend,
            input_preprocessor=lambda q: (
                f"Break this question into exactly {max_breakdown} sub-questions. "
                "Return ONLY a numbered list.\n\n"
                f"Question: {q}"
            ),
        )
        print(f"  Backend: {backend_name}")

        def worker_factory(sub_query, index):
            worker, _ = create_inferencer(
                backend,
                input_preprocessor=lambda q: (
                    f"Briefly answer this question in 2-3 sentences: {q}"
                ),
            )
            return worker

        bta = BreakdownThenAggregateInferencer(
            breakdown_inferencer=breakdown_inferencer,
            worker_factory=worker_factory,
            aggregator_inferencer=None,
            breakdown_parser=lambda raw: _robust_breakdown_parser(raw, max_breakdown),
            max_breakdown=max_breakdown,
            checkpoint_dir=tmpdir,
        )

        print(f"  Running BTA without aggregator...")
        start = time.time()
        result = bta.infer(query)
        elapsed = time.time() - start

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Result type: {type(result).__name__}")

        if isinstance(result, tuple):
            print(f"  Number of results: {len(result)}")
            for i, r in enumerate(result):
                print(f"    Result {i}: {str(r)[:100]}...")
            if len(result) > 0:
                print("\n  PASSED: Got tuple of worker results")
                return True
        elif isinstance(result, str) and len(result) > 10:
            print(f"  Single result: {result[:200]}...")
            print("\n  PASSED: Got single worker result")
            return True

        print("\n  FAILED: Unexpected result format")
        return False

    except Exception as e:
        print(f"\n  FAILED: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def print_summary(results):
    """Print test summary."""
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = 0
    failed = 0
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  [{status}] {name}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed out of {len(results)}")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Real E2E integration test for BreakdownThenAggregateInferencer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query", "-q", type=str, default=DEFAULT_QUERY,
        help="Query to test",
    )
    parser.add_argument(
        "--mode", "-m", type=str,
        choices=["simple", "bta", "bta-no-agg", "all"],
        default="all",
        help="Test mode (default: all)",
    )
    parser.add_argument(
        "--backend", "-b", type=str,
        choices=["metagen", "devmate", "claude_code", "auto"],
        default="auto",
        help="Backend inferencer (default: auto-detect)",
    )
    parser.add_argument(
        "--max-breakdown", type=int, default=3,
        help="Max sub-queries for breakdown (default: 3)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("BTA E2E INTEGRATION TEST")
    print("=" * 70)
    print(f"  Query: {args.query[:80]}...")
    print(f"  Mode: {args.mode}")
    print(f"  Backend: {args.backend}")
    print(f"  Max breakdown: {args.max_breakdown}")

    results = []

    if args.mode in ("simple", "all"):
        results.append(
            ("Simple Query", test_simple_query(args.query, args.backend))
        )

    if args.mode in ("bta-no-agg", "all"):
        results.append(
            (
                "BTA No Aggregator",
                test_bta_no_aggregator(args.query, args.backend, max_breakdown=2),
            )
        )

    if args.mode in ("bta", "all"):
        results.append(
            (
                "BTA Full Pipeline",
                test_bta_full_pipeline(
                    args.query, args.backend, max_breakdown=args.max_breakdown
                ),
            )
        )

    success = print_summary(results)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
