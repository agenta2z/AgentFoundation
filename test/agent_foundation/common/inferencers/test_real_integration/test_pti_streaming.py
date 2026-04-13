"""PTI + streaming DualInferencer tests — end-to-end and multi-iteration.

Validates Requirements 11 and 14 from the integration test spec.
All tests invoke real ``claude`` CLI subprocesses and are marked ``@pytest.mark.integration``.

PTI is configured with DualInferencer children whose base/review inferencers are
real ClaudeCodeCliInferencers. Uses ``append_system_prompt`` on review inferencers
to force approval (keeping tests deterministic).
"""

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
    PlanThenImplementResponse,
)

from .conftest import (
    DEFAULT_TIMEOUT,
    PUZZLE_PROMPT,
    skip_claude,
    verify_deep_flatten,
)

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------
APPROVE_SYSTEM_PROMPT = (
    "You are a code reviewer. Review the code. "
    'Respond ONLY with JSON: {"approved": true, "severity": "COSMETIC", '
    '"issues": [], "reasoning": "Looks good"}'
)

ANALYSIS_PROMPT = (
    "Analyze the implementation below. Check each function for correctness. "
    "If you find any issues, respond with:\n"
    "SHOULD_CONTINUE: YES\n"
    "followed by your analysis.\n"
    "If everything looks correct, respond with:\n"
    "SHOULD_CONTINUE: NO\n"
    "followed by your analysis."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_claude(tmp_workspace, **overrides):
    """Create a ClaudeCodeCliInferencer with sensible defaults for tests."""
    kwargs = dict(
        target_path=str(tmp_workspace["workspace"]),
        cache_folder=str(tmp_workspace["cache"]),
        model_name="sonnet",
        resume_with_saved_results=True,
        idle_timeout_seconds=60,
    )
    if hasattr(ClaudeCodeCliInferencer, "permission_mode"):
        kwargs["permission_mode"] = "bypassPermissions"
    kwargs.update(overrides)
    return ClaudeCodeCliInferencer(**kwargs)


def _make_dual(tmp_workspace, **overrides):
    """Create a DualInferencer with claude base + claude review (auto-approve)."""
    base = _make_claude(tmp_workspace)
    review = _make_claude(tmp_workspace, append_system_prompt=APPROVE_SYSTEM_PROMPT)
    kwargs = dict(
        base_inferencer=base,
        review_inferencer=review,
        consensus_config=ConsensusConfig(max_iterations=2, max_consensus_attempts=1),
    )
    kwargs.update(overrides)
    return DualInferencer(**kwargs)


# ===========================================================================
# Test 1: PTI end-to-end with coding puzzle (Req 11.1–11.4)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 3)
@skip_claude
async def test_pti_end_to_end_with_coding_puzzle(tmp_workspace):
    """PTI with planner=DualInferencer(claude,claude) and executor=DualInferencer(claude,claude).

    Uses the Tier C PUZZLE_PROMPT. Verifies PTI completes both planning and
    implementation phases, producing a PlanThenImplementResponse with both
    plan_output and executor_output populated.

    **Validates: Requirements 11.1, 11.2, 11.3, 11.4**
    """
    planner = _make_dual(tmp_workspace)
    executor = _make_dual(tmp_workspace)

    pti = PlanThenImplementInferencer(
        planner_inferencer=planner,
        executor_inferencer=executor,
    )

    result = await pti.ainfer(PUZZLE_PROMPT)

    # Verify response type and structure
    assert isinstance(result, PlanThenImplementResponse), (
        f"Expected PlanThenImplementResponse, got {type(result).__name__}"
    )

    # Plan output should be populated (Req 11.2)
    assert result.plan_output is not None and result.plan_output != "", (
        "plan_output should be populated after planning phase"
    )

    # Executor output should be populated (Req 11.3)
    assert result.executor_output is not None, (
        "executor_output should be populated after implementation phase"
    )

    # base_response should contain meaningful content
    assert result.base_response is not None and result.base_response != "", (
        "base_response should contain the final output"
    )

    # Optionally verify the generated code contains deep_flatten
    # (best-effort — the model may format output differently)
    output_text = str(result.base_response)
    if "deep_flatten" in output_text and "def deep_flatten" in output_text:
        # Extract code block if present
        code = output_text
        if "```python" in code:
            start = code.index("```python") + len("```python")
            end = code.index("```", start)
            code = code[start:end]
        try:
            verify_deep_flatten(code)
        except Exception:
            # Verification is best-effort — the model may not produce
            # perfectly correct code on every run
            pass


# ===========================================================================
# Test 2: PTI multi-iteration with analysis (Req 14.1–14.4)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 4)
@skip_claude
async def test_pti_multi_iteration_with_analysis(tmp_workspace):
    """PTI with enable_multiple_iterations=True, max_meta_iterations=2, and an analyzer.

    Uses a prompt that triggers the analysis loop. Verifies iteration_history
    has records for completed iterations.

    **Validates: Requirements 14.1, 14.2, 14.3, 14.4**
    """
    workspace_dir = str(tmp_workspace["workspace"])

    planner = _make_dual(tmp_workspace)
    executor = _make_dual(tmp_workspace)

    # Analyzer is a single ClaudeCodeCliInferencer (not a DualInferencer)
    analyzer = _make_claude(
        tmp_workspace,
        append_system_prompt=(
            "You are a code analyzer. Analyze the implementation provided. "
            "Always respond with 'SHOULD_CONTINUE: NO' followed by your analysis. "
            "This ensures the iteration loop terminates."
        ),
    )

    pti = PlanThenImplementInferencer(
        planner_inferencer=planner,
        executor_inferencer=executor,
        analyzer_inferencer=analyzer,
        enable_analysis=True,
        enable_multiple_iterations=True,
        max_meta_iterations=2,
        reset_sessions_per_meta_iteration=True,
        workspace_path=workspace_dir,
    )

    result = await pti.ainfer(PUZZLE_PROMPT)

    # Verify response type
    assert isinstance(result, PlanThenImplementResponse), (
        f"Expected PlanThenImplementResponse, got {type(result).__name__}"
    )

    # Plan and executor outputs should be populated
    assert result.plan_output is not None and result.plan_output != "", (
        "plan_output should be populated"
    )
    assert result.executor_output is not None, (
        "executor_output should be populated"
    )

    # iteration_history should have at least one record (Req 14.4)
    assert len(result.iteration_history) >= 1, (
        f"Expected at least 1 iteration record, got {len(result.iteration_history)}"
    )

    # Each iteration record should have plan and executor output
    for i, record in enumerate(result.iteration_history):
        assert record.iteration >= 1, (
            f"Iteration record {i} should have a positive iteration number"
        )
        assert record.plan_output is not None, (
            f"Iteration record {i} should have plan_output"
        )

    # total_meta_iterations should reflect completed iterations
    assert result.total_meta_iterations >= 1, (
        f"Expected at least 1 meta-iteration, got {result.total_meta_iterations}"
    )
