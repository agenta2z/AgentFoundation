# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Real E2E integration tests for ConversationalInferencer agentic loop.

Uses real MetagenApiInferencer (real LLM calls) with:
- Mock tool executor (simulates tool execution without side effects)
- Real JinjaPromptRenderer (real template rendering)
- Real prior_context and dynamic_context

Usage:
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_conversational_inferencer_real
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_conversational_inferencer_real -- --mode all
    buck2 run --prefer-remote //rankevolve/test/agentic_foundation:test_conversational_inferencer_real -- --mode tool_call
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import traceback
from pathlib import Path

from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
    ToolExecutionResult,
)


def _make_base_inferencer():
    from agent_foundation.common.inferencers.api_inferencers.metagen.metagen_api_inferencer import (
        MetagenApiInferencer,
    )

    return MetagenApiInferencer(temperature=0.3, max_tokens=1024)


def _make_conversational_inferencer(**kwargs):
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
        ConversationalInferencer,
    )

    base = _make_base_inferencer()
    kwargs.setdefault("base_inferencer", base)
    return ConversationalInferencer(**kwargs)


def _make_prompt_renderer():
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.prompt_rendering import (
        JinjaPromptRenderer,
    )

    template_dir = (
        Path(__file__).resolve().parents[4] / "src" / "resources" / "prompt_templates"
    )
    return JinjaPromptRenderer(str(template_dir))


def _make_tool_registry():
    from rankevolve.src.resources.tools.models import ParameterDef, ToolDefinition

    return {
        "set_target_path": ToolDefinition(
            name="set_target_path",
            description="Set the target path for the current session",
            tool_type="Action",
            category="session",
            aliases=["target-path", "root"],
            parameters=[
                ParameterDef(
                    name="path",
                    type="string",
                    description="Target path",
                    required=True,
                )
            ],
        ),
        "task": ToolDefinition(
            name="task",
            description="Run a workflow task",
            tool_type="Action",
            category="workflow",
            aliases=[],
            parameters=[
                ParameterDef(
                    name="request",
                    type="string",
                    description="Task request",
                    required=True,
                )
            ],
        ),
    }


async def _mock_tool_executor(tool_name: str, arguments: dict) -> ToolExecutionResult:
    """Mock tool executor that simulates real tool responses."""
    if tool_name == "set_target_path":
        path = arguments.get("path", "/unknown")
        return ToolExecutionResult(
            result=f"Target path set to: {path}",
            context_updates={"target_path": path},
        )
    elif tool_name == "task":
        request = arguments.get("request", "")
        return ToolExecutionResult(
            result=f"Task completed. Request: '{request[:50]}'. 3 files modified."
        )
    else:
        return ToolExecutionResult(result=f"Mock result for {tool_name}: success")


# ---------------------------------------------------------------------------
# Test 1: Pure text conversation (no tools triggered)
# ---------------------------------------------------------------------------
async def test_pure_text():
    """Simple question → pure text response, no tool calls."""
    print("\n--- test_pure_text ---")
    inf = _make_conversational_inferencer(
        tool_registry=_make_tool_registry(),
        prompt_renderer=_make_prompt_renderer(),
        tool_executor=_mock_tool_executor,
    )
    inf.set_prior_context({"target_path": "/tmp/myproject", "model": "test"})

    t0 = time.time()
    result = await inf.run_agentic_loop("What is Python? Answer in one sentence.")
    elapsed = time.time() - t0

    print(f"  Response ({elapsed:.1f}s): {result.text[:300]}")
    print(f"  Iterations: {result.iterations_used}")
    assert isinstance(result.text, str) and len(result.text) > 10
    assert result.iterations_used == 1
    assert len(result.completed_actions) == 0
    assert result.last_rendered_prompt != ""
    assert result.last_template_source != ""
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 2: LLM triggers set_target_path tool
# ---------------------------------------------------------------------------
async def test_tool_call_set_target():
    """Ask LLM to set target path — should trigger set_target_path tool."""
    print("\n--- test_tool_call_set_target ---")
    inf = _make_conversational_inferencer(
        tool_registry=_make_tool_registry(),
        prompt_renderer=_make_prompt_renderer(),
        tool_executor=_mock_tool_executor,
    )
    inf.set_prior_context({"target_path": "", "model": "test"})

    t0 = time.time()
    result = await inf.run_agentic_loop(
        "Please set the target path to /data/users/zgchen/my_ranking_model"
    )
    elapsed = time.time() - t0

    print(f"  Response ({elapsed:.1f}s): {result.text[:300]}")
    print(f"  Iterations: {result.iterations_used}")
    print(
        f"  Completed actions: {[(a.tool, a.summary[:60]) for a in result.completed_actions]}"
    )

    if result.completed_actions:
        assert any(a.tool == "set_target_path" for a in result.completed_actions)
        print("  Tool was called: set_target_path")
    else:
        assert "/data/users/zgchen" in result.text
        print("  LLM responded with text (no tool call)")
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 3: Multi-turn with context accumulation
# ---------------------------------------------------------------------------
async def test_multi_turn_context():
    """Two turns — verify dynamic context accumulates."""
    print("\n--- test_multi_turn_context ---")
    inf = _make_conversational_inferencer(
        tool_registry=_make_tool_registry(),
        prompt_renderer=_make_prompt_renderer(),
        tool_executor=_mock_tool_executor,
    )
    inf.set_prior_context({"target_path": "/tmp/project", "model": "test"})

    # Turn 1
    t0 = time.time()
    r1 = await inf.run_agentic_loop("What can you help me with?")
    print(f"  Turn 1 ({time.time() - t0:.1f}s): {r1.text[:200]}")

    # Turn 2
    inf.set_messages(
        [
            {"role": "user", "content": "What can you help me with?"},
            {"role": "assistant", "content": r1.text},
            {"role": "user", "content": "How many tools do you have access to?"},
        ]
    )
    t0 = time.time()
    r2 = await inf.run_agentic_loop("How many tools do you have access to?")
    print(f"  Turn 2 ({time.time() - t0:.1f}s): {r2.text[:200]}")

    assert isinstance(r2.text, str) and len(r2.text) > 5
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 4: Prior context injection
# ---------------------------------------------------------------------------
async def test_prior_context():
    """Verify prior_context values appear in the rendered prompt."""
    print("\n--- test_prior_context ---")
    inf = _make_conversational_inferencer(
        tool_registry=_make_tool_registry(),
        prompt_renderer=_make_prompt_renderer(),
        tool_executor=_mock_tool_executor,
    )
    inf.set_prior_context(
        {
            "target_path": "/data/users/zgchen/special_model",
            "model": "claude-4-6-opus",
            "workflow_status": "Phase: idle | Status: ready",
            "workflow_description": "RankEvolve helps evolve ranking models.",
            "current_phase": "idle",
            "phase_status": "idle",
        }
    )

    result = await inf.run_agentic_loop("What is my current target path?")
    print(f"  Response: {result.text[:300]}")
    print(
        f"  Rendered prompt contains target_path: "
        f"{'/data/users/zgchen/special_model' in result.last_rendered_prompt}"
    )

    assert "/data/users/zgchen/special_model" in result.last_rendered_prompt
    assert (
        result.last_template_feed.get("target_path")
        == "/data/users/zgchen/special_model"
    )
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 5: AgenticResult fields completeness
# ---------------------------------------------------------------------------
async def test_result_fields():
    """Verify all AgenticResult fields are populated correctly."""
    print("\n--- test_result_fields ---")
    inf = _make_conversational_inferencer(
        tool_registry=_make_tool_registry(),
        prompt_renderer=_make_prompt_renderer(),
        tool_executor=_mock_tool_executor,
    )
    inf.set_prior_context({"target_path": "/tmp", "model": "test"})

    result = await inf.run_agentic_loop("Say hello.")

    assert isinstance(result.text, str), "text should be str"
    assert isinstance(result.raw_response, str), "raw_response should be str"
    assert isinstance(result.iterations_used, int), "iterations_used should be int"
    assert isinstance(result.completed_actions, list), (
        "completed_actions should be list"
    )
    assert (
        isinstance(result.last_rendered_prompt, str)
        and len(result.last_rendered_prompt) > 0
    )
    assert (
        isinstance(result.last_template_source, str)
        and len(result.last_template_source) > 0
    )
    assert (
        isinstance(result.last_template_feed, dict)
        and len(result.last_template_feed) > 0
    )
    assert result.has_conversation_tool is False
    print(f"  All fields validated. Response: {result.text[:100]}")
    print("  PASS")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
TESTS = {
    "pure_text": test_pure_text,
    "tool_call": test_tool_call_set_target,
    "multi_turn": test_multi_turn_context,
    "prior_context": test_prior_context,
    "result_fields": test_result_fields,
}


def main():
    parser = argparse.ArgumentParser(
        description="ConversationalInferencer Real E2E Tests"
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
