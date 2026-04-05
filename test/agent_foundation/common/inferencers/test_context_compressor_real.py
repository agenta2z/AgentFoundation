# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Real E2E integration tests for InferencerContextCompressor.

Uses real MetagenApiInferencer to compress action histories.
Verifies compression quality: shorter output, key info preserved.

Usage:
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_context_compressor_real
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_context_compressor_real -- --mode all
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_context_compressor_real -- --mode basic
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import traceback


def _make_compressor(**kwargs):
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.context_compressor import (
        InferencerContextCompressor,
    )
    from agent_foundation.common.inferencers.api_inferencers.metagen.metagen_api_inferencer import (
        MetagenApiInferencer,
    )

    inferencer = MetagenApiInferencer(temperature=0.1, max_tokens=512)
    return InferencerContextCompressor(inferencer=inferencer, **kwargs)


# ---------------------------------------------------------------------------
# Test 1: Basic compression — long action list → short summary
# ---------------------------------------------------------------------------
async def test_basic_compression():
    """Compress a long action history and verify it gets shorter."""
    print("\n--- test_basic_compression ---")
    compressor = _make_compressor()

    actions = []
    for i in range(15):
        actions.append(
            f"- task_{i}: Executed task {i} with parameters x={i * 10}, y={i * 5}. "
            f"Result: processed {i * 100} records, generated {i * 3} output files. "
            f"Status: completed successfully in {i + 1}.{i}s."
        )
    context = "\n".join(actions)
    print(f"  Original: {len(context)} chars, {len(actions)} actions")

    t0 = time.time()
    compressed = await compressor(context, max_length=600)
    elapsed = time.time() - t0

    print(f"  Compressed ({elapsed:.1f}s): {len(compressed)} chars")
    print(f"  Reduction: {(1 - len(compressed) / len(context)) * 100:.0f}%")
    print(f"  Output:\n    {compressed[:400]}")

    assert len(compressed) <= 600, f"Too long: {len(compressed)}"
    assert len(compressed) < len(context), "Should be shorter"
    assert len(compressed) > 20, "Should not be empty"
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 2: Preservation of key information (tool names, paths, metrics)
# ---------------------------------------------------------------------------
async def test_preserves_key_info():
    """Verify compression preserves critical information."""
    print("\n--- test_preserves_key_info ---")
    compressor = _make_compressor()

    context = "\n".join(
        [
            "- set_target_path: Set target path to /data/users/zgchen/ranking_v3",
            "- understand_codebase: Found 12 modules including train.py, evaluate.py, model.py. Framework: PyTorch 2.1.",
            "- task: Ran baseline. AUC=0.7823, NDCG@10=0.4512",
            "- task: Added feature normalization. Modified features.py and config.yaml.",
            "- task: Ran experiment A. AUC=0.7956 (+1.7%), NDCG@10=0.4689 (+3.9%)",
            "- task: Added attention mechanism to model. Modified model.py (85 lines added).",
            "- task: Ran experiment B. AUC=0.8102 (+3.6%), NDCG@10=0.4801 (+6.4%)",
            "- knowledge: Found documentation on distributed training with FSDP.",
            "- task: Set up distributed training with 8 GPUs. Modified train.py for FSDP.",
            "- task: Ran final evaluation. AUC=0.8234 (+5.3%), NDCG@10=0.4923 (+9.1%). Best results.",
        ]
    )

    t0 = time.time()
    compressed = await compressor(context, max_length=500)
    elapsed = time.time() - t0

    print(f"  Compressed ({elapsed:.1f}s): {len(compressed)} chars")
    print(f"  Output:\n    {compressed}")

    checks = {
        "target path": "ranking_v3" in compressed or "/data/users" in compressed,
        "baseline AUC": "0.78" in compressed or "0.7823" in compressed,
        "best AUC": "0.82" in compressed or "0.8234" in compressed,
        "tool types": "task" in compressed.lower(),
        "improvement trend": any(
            w in compressed.lower()
            for w in ["improv", "better", "increas", "+", "gain"]
        ),
    }
    for check_name, passed in checks.items():
        status = "OK" if passed else "MISSING"
        print(f"    {check_name}: {status}")

    pass_count = sum(checks.values())
    assert pass_count >= 3, f"Only {pass_count}/5 key info checks passed: {checks}"
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 3: Very short context (below max_length) — no LLM call needed
# ---------------------------------------------------------------------------
async def test_short_context_passthrough():
    """Context shorter than max_length should pass through unchanged."""
    print("\n--- test_short_context_passthrough ---")
    compressor = _make_compressor()

    short_context = "- task: Did one thing. Result: success."
    result = await compressor(short_context, max_length=1000)

    assert result == short_context, f"Should pass through unchanged, got: {result}"
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 4: Compression quality with AgenticDynamicContext integration
# ---------------------------------------------------------------------------
async def test_dynamic_context_integration():
    """Test compressor with real AgenticDynamicContext.to_text() output."""
    print("\n--- test_dynamic_context_integration ---")
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.context import (
        AgenticDynamicContext,
    )

    compressor = _make_compressor()

    ctx = AgenticDynamicContext()
    actions = [
        ("set_target_path", "Set to /data/users/zgchen/model_v3"),
        ("understand_codebase", "Analyzed 15 modules, PyTorch-based ranking model"),
        ("task", "Generated experiment proposal: add position bias correction"),
        ("task", "Implemented PositionBiasLayer in model.py (45 lines added)"),
        ("task", "Baseline eval: AUC=0.7823, NDCG@10=0.4512"),
        ("task", "Experiment eval: AUC=0.7956 (+1.7%), NDCG@10=0.4689 (+3.9%)"),
        ("knowledge", "Found 3 papers on feature interaction: DeepFM, DCN-v2, AutoInt"),
        ("task", "Implemented DCN-v2 cross network layer (60 lines in model.py)"),
        ("task", "Final eval: AUC=0.8102, NDCG@10=0.4801. Best results so far."),
    ]
    for tool, summary in actions:
        ctx.add_action(tool, summary)

    context_text = ctx.to_text()
    print(f"  Dynamic context text: {len(context_text)} chars")

    t0 = time.time()
    compressed = await compressor(context_text, max_length=400)
    elapsed = time.time() - t0

    print(f"  Compressed ({elapsed:.1f}s): {len(compressed)} chars")
    print(f"  Output: {compressed}")

    assert len(compressed) <= 400
    assert len(compressed) < len(context_text)

    # Simulate what the agentic loop would do after compression
    ctx.compress(compressed)
    assert ctx._compressed_history == compressed
    assert len(ctx._uncompressed_actions) == 0

    # Add new actions after compression
    ctx.add_action("task", "Added user embedding features")
    ctx.add_action("task", "Re-trained model. AUC=0.8234")

    combined = ctx.to_text()
    print(f"  After new actions: {len(combined)} chars")
    assert compressed in combined, "Compressed history should be in combined text"
    assert "user embedding" in combined, "New actions should appear"
    assert "0.8234" in combined, "New metrics should appear"
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 5: Fallback on error (simulate by using a failing inferencer)
# ---------------------------------------------------------------------------
async def test_fallback_on_error():
    """When LLM call fails, compressor should fallback to truncation."""
    print("\n--- test_fallback_on_error ---")
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.context_compressor import (
        InferencerContextCompressor,
    )

    class FailingInferencer:
        async def ainfer(self, prompt):
            raise RuntimeError("Simulated API failure")

    compressor = InferencerContextCompressor(
        inferencer=FailingInferencer(),
        fallback_on_error=True,
    )

    context = "x" * 500
    result = await compressor(context, max_length=200)

    assert len(result) <= 200, f"Fallback too long: {len(result)}"
    assert "truncated" in result, "Should indicate truncation"
    print(f"  Fallback result: {result[-50:]}")
    print("  PASS")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
TESTS = {
    "basic": test_basic_compression,
    "key_info": test_preserves_key_info,
    "passthrough": test_short_context_passthrough,
    "dynamic_context": test_dynamic_context_integration,
    "fallback": test_fallback_on_error,
}


def main():
    parser = argparse.ArgumentParser(
        description="InferencerContextCompressor Real E2E Tests"
    )
    parser.add_argument("--mode", choices=list(TESTS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    tests = TESTS if args.mode == "all" else {args.mode: TESTS[args.mode]}
    passed, failed = 0, 0

    for name, test_fn in tests.items():
        try:
            asyncio.run(test_fn())
            passed += 1
        except Exception:
            failed += 1
            print(f"  FAIL: {name}")
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
