"""BTA resume tests — child->parent checkpoint promotion (real ``claude`` CLI).

Validates Requirements 20 and 21 from the integration test spec against the
generic promotion mechanism that retired BTA's hand-rolled
``breakdown_result.json``. All tests invoke real ``claude`` CLI subprocesses and
are marked ``@pytest.mark.integration``.

BTA is both an ``InferencerBase`` and a WorkGraph: a breakdown node decomposes
the request into sub-queries, N workers fan out, and an aggregator merges. The
durable resume state is now ONE generic artifact. The breakdown child emits a
``decomposed_subtasks`` JSON fence; the ``expected_extraction`` register
persists it to the child's ``outputs/decomposed_subtasks.json``; and because the
entry declares ``checkpoint_scope="parent"``, BTA promotes it up into the parent
workspace's ``checkpoints/breakdown/decomposed_subtasks.json`` the instant the
breakdown completes (before any worker can crash). On resume, that one promoted
file rebuilds the worker fan-out and restores the aggregator's guidance — the
breakdown LLM is never re-run, and ``breakdown_result.json`` is never written.

Direct construction (below) bypasses the YAML config layer that normally applies
``BREAKDOWN_TEMPLATE_DEFAULTS`` (via ``BTA.SLOT_DEFAULTS``) to the
``breakdown_inferencer`` slot, so these tests declare the same
``expected_extraction`` on the breakdown child explicitly and pass ``workspace=``
(the structured layout) rather than the legacy flat ``checkpoint_dir=`` — under
which the modern workspace/promotion machinery is inert (``self._workspace is
None``).
"""

import json
import os

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from attr import attrs

from .conftest import DEFAULT_TIMEOUT, skip_claude

# ---------------------------------------------------------------------------
# Prompt constants — elicit the ``decomposed_subtasks`` JSON fence consumed by
# BOTH the ``json_subtasks`` breakdown parser and the extraction register.
# ---------------------------------------------------------------------------
BREAKDOWN_PROMPT = (
    "Break the following request into exactly 2 focused, independent subtasks: "
    "'Explain what 1+1 equals and why' and 'Explain what 2+2 equals and why'."
)

BREAKDOWN_SYSTEM_PROMPT = (
    "You are a task decomposer. Respond with ONLY a Markdown fenced code block "
    "whose opening fence is exactly ```json decomposed_subtasks and whose body is "
    "a JSON object with two keys: a `subtasks` array (each element an object with a "
    "`description` string) and an `aggregation_guidance` string. Produce EXACTLY 2 "
    "subtasks. Wrap the whole response in <Response> and </Response> tags and "
    "output nothing else."
)

# The production breakdown extraction (mirrors ``BREAKDOWN_TEMPLATE_DEFAULTS``):
# persist the ``decomposed_subtasks`` fence to the breakdown child's
# ``outputs/decomposed_subtasks.json`` AND promote it up into the parent BTA's
# ``checkpoints/breakdown/`` (``checkpoint_scope="parent"``) as the durable resume
# state that replaced the hand-rolled ``breakdown_result.json``.
_BREAKDOWN_EXPECTED_EXTRACTION = [
    {
        "label": "decomposed_subtasks",
        "source": "response",
        "kind": "content",
        "fallback_to_source": True,
        "persist_to": "decomposed_subtasks.json",
        "checkpoint_scope": "parent",
    }
]


@attrs
class _FailIfCalledBreakdown(InferencerBase):
    """Breakdown sentinel proving the breakdown LLM is NOT re-run on resume: the
    promoted checkpoint must drive the fan-out rebuild. Any invocation fails the
    test. A real ``InferencerBase`` so ``workspace=`` propagation binds it cleanly."""

    def _infer(self, inference_input, inference_config=None, **kwargs):
        raise AssertionError(
            "breakdown must NOT run on resume — the promoted "
            "checkpoints/breakdown/decomposed_subtasks.json should rebuild the fan-out"
        )

    async def ainfer(self, *args, **kwargs):
        raise AssertionError("breakdown_inferencer.ainfer called on resume")

    def infer(self, *args, **kwargs):
        raise AssertionError("breakdown_inferencer.infer called on resume")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_claude(tmp_workspace, **overrides):
    """Create a ClaudeCodeCliInferencer with sensible defaults for tests."""
    kwargs = {
        "target_path": str(tmp_workspace["workspace"]),
        "cache_folder": str(tmp_workspace["cache"]),
        "model_name": "sonnet",
        "resume_with_saved_results": True,
        "idle_timeout_seconds": 60,
    }
    if hasattr(ClaudeCodeCliInferencer, "permission_mode"):
        kwargs["permission_mode"] = "bypassPermissions"
    kwargs.update(overrides)
    return ClaudeCodeCliInferencer(**kwargs)


def _make_breakdown(tmp_workspace):
    """A real breakdown inferencer wired to emit + promote the
    ``decomposed_subtasks`` fence. Direct construction skips the config layer that
    would apply ``BREAKDOWN_TEMPLATE_DEFAULTS``, so the extraction is declared
    explicitly (see module docstring)."""
    breakdown = _make_claude(
        tmp_workspace, append_system_prompt=BREAKDOWN_SYSTEM_PROMPT
    )
    breakdown.expected_extraction = _BREAKDOWN_EXPECTED_EXTRACTION
    return breakdown


def _make_worker_inferencers(tmp_workspace):
    """Return a worker_inferencers callable: (sub_query, index) -> ClaudeCodeCliInferencer."""

    def factory(sub_query, index):
        return _make_claude(tmp_workspace)

    return factory


def _bta_workspace(tmp_workspace):
    """The BTA's structured workspace. Its root holds ``children/`` and
    ``checkpoints/``; a first run and its resume twin share this one root, the way
    a real resume shares the run directory."""
    return InferencerWorkspace(root=str(tmp_workspace["checkpoint"]))


def _promoted_breakdown_path(ws):
    return ws.checkpoint_path(os.path.join("breakdown", "decomposed_subtasks.json"))


def _assert_promoted_breakdown_valid(ws):
    """The promoted ``decomposed_subtasks.json`` exists and parses to >=1 subtask."""
    promoted = _promoted_breakdown_path(ws)
    assert os.path.exists(promoted), (
        f"promoted breakdown checkpoint should exist at {promoted}"
    )
    with open(promoted, encoding="utf-8") as f:
        data = json.load(f)
    subtasks = data.get("subtasks") or data.get("decomposed_subtasks") or []
    assert len(subtasks) >= 1, "promoted checkpoint should carry at least 1 subtask"


def _assert_no_breakdown_result_json(ws):
    """The retired hand-rolled resume file must never be written, anywhere."""
    offenders = [
        os.path.join(root, "breakdown_result.json")
        for root, _dirs, files in os.walk(ws.root)
        if "breakdown_result.json" in files
    ]
    assert offenders == [], (
        f"retired breakdown_result.json must never appear: {offenders}"
    )


def _assert_result_nonempty(result):
    assert result is not None, "BTA should produce a result"
    # With no aggregator, result is a tuple of worker outputs or a single result.
    if isinstance(result, tuple):
        assert len(result) >= 1, "should have at least 1 worker result"
        for i, worker_result in enumerate(result):
            assert worker_result is not None, f"worker {i} result should not be None"
            assert str(worker_result).strip() != "", (
                f"worker {i} result should not be empty"
            )
    else:
        assert str(result).strip() != "", "result should not be empty"


# ===========================================================================
# Test 1: Breakdown promotion + resume rebuild (Req 20.1, 20.2, 20.3)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 3)
@skip_claude
async def test_breakdown_promotion_resume(tmp_workspace):
    """Run BTA -> breakdown completes and is promoted -> interrupt during workers
    -> verify the promoted ``decomposed_subtasks.json`` -> resume BTA -> verify the
    breakdown LLM is NOT re-run (fan-out rebuilt from the promoted checkpoint) ->
    workers run to a non-empty result.

    Short worker ``idle_timeout_seconds`` forces interruption in the worker phase,
    after the breakdown has completed and been promoted (promotion happens the
    moment the breakdown parses, before any worker is dispatched).

    **Validates: Requirements 20.1, 20.2, 20.3**
    """
    ws = _bta_workspace(tmp_workspace)

    def short_timeout_worker_inferencers(sub_query, index):
        return _make_claude(tmp_workspace, idle_timeout_seconds=2)

    bta1 = BreakdownThenAggregateInferencer(
        breakdown_inferencer=_make_breakdown(tmp_workspace),
        worker_inferencers=short_timeout_worker_inferencers,
        workspace=ws,
        breakdown_format="json_subtasks",
        resume_with_saved_results=True,
        max_breakdown=2,
    )

    # First run — breakdown completes + promotes, workers interrupted via idle_timeout.
    try:
        await bta1.ainfer(BREAKDOWN_PROMPT)
    except Exception:
        pass  # Interruption expected during the worker phase.

    # Breakdown state is durable in the promoted checkpoint (Req 20.1); the retired
    # hand-rolled file is gone.
    _assert_promoted_breakdown_valid(ws)
    _assert_no_breakdown_result_json(ws)

    # Resume: fresh BTA, same workspace root, a breakdown that fails if invoked —
    # proving the fan-out is rebuilt from the promoted checkpoint, not re-run
    # (Req 20.2, 20.3).
    bta2 = BreakdownThenAggregateInferencer(
        breakdown_inferencer=_FailIfCalledBreakdown(),
        worker_inferencers=_make_worker_inferencers(tmp_workspace),
        workspace=_bta_workspace(tmp_workspace),
        breakdown_format="json_subtasks",
        resume_with_saved_results=True,
        max_breakdown=2,
    )

    result = await bta2.ainfer(BREAKDOWN_PROMPT)

    _assert_result_nonempty(result)
    _assert_no_breakdown_result_json(ws)


# ===========================================================================
# Test 2: Worker resume from cache (Req 21.1, 21.4)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 3)
@skip_claude
async def test_worker_resume_from_cache(tmp_workspace):
    """Run BTA -> first worker completes, second worker interrupted -> resume ->
    the completed worker is served from its per-node checkpoint, the remaining
    worker runs fresh -> non-empty result. The breakdown is promoted once and
    reused on resume (a sentinel proves it is not re-run).

    **Validates: Requirements 21.1, 21.4**
    """
    ws = _bta_workspace(tmp_workspace)

    def mixed_timeout_worker_inferencers(sub_query, index):
        """First worker completes normally; second worker times out."""
        if index == 0:
            return _make_claude(tmp_workspace)
        return _make_claude(tmp_workspace, idle_timeout_seconds=2)

    bta1 = BreakdownThenAggregateInferencer(
        breakdown_inferencer=_make_breakdown(tmp_workspace),
        worker_inferencers=mixed_timeout_worker_inferencers,
        workspace=ws,
        breakdown_format="json_subtasks",
        resume_with_saved_results=True,
        max_breakdown=2,
    )

    # First run — first worker completes, second worker interrupted.
    try:
        await bta1.ainfer(BREAKDOWN_PROMPT)
    except Exception:
        pass  # Interruption expected from the second worker.

    _assert_promoted_breakdown_valid(ws)
    _assert_no_breakdown_result_json(ws)

    # Resume: same workspace root; the first worker is served from its checkpoint
    # (Req 21.1), the second executes fresh (Req 21.4). The breakdown is not
    # re-run — the promoted checkpoint drives the rebuild.
    bta2 = BreakdownThenAggregateInferencer(
        breakdown_inferencer=_FailIfCalledBreakdown(),
        worker_inferencers=_make_worker_inferencers(tmp_workspace),
        workspace=_bta_workspace(tmp_workspace),
        breakdown_format="json_subtasks",
        resume_with_saved_results=True,
        max_breakdown=2,
    )

    result = await bta2.ainfer(BREAKDOWN_PROMPT)

    _assert_result_nonempty(result)
    _assert_no_breakdown_result_json(ws)


# ===========================================================================
# Test 3: Corrupted promoted checkpoint (Req 20.4)
# ===========================================================================
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 2)
@skip_claude
async def test_corrupted_promoted_checkpoint(tmp_workspace):
    """Pre-create a corrupted promoted ``decomposed_subtasks.json`` -> run BTA ->
    the loader degrades to "no checkpoint" (``_load_promoted_breakdown`` catches
    the parse error and returns ``(None, None)``), so BTA re-runs the breakdown
    fresh and re-promotes a valid checkpoint (graceful degradation).

    **Validates: Requirement 20.4**
    """
    ws = _bta_workspace(tmp_workspace)

    # Pre-create a corrupted promoted checkpoint (invalid JSON).
    promoted = _promoted_breakdown_path(ws)
    os.makedirs(os.path.dirname(promoted), exist_ok=True)
    with open(promoted, "w", encoding="utf-8") as f:
        f.write("{corrupted json content that is not valid!!!")
    assert os.path.exists(promoted), "corrupted checkpoint file should exist"

    bta = BreakdownThenAggregateInferencer(
        breakdown_inferencer=_make_breakdown(tmp_workspace),
        worker_inferencers=_make_worker_inferencers(tmp_workspace),
        workspace=ws,
        breakdown_format="json_subtasks",
        resume_with_saved_results=True,
        max_breakdown=2,
    )

    result = await bta.ainfer(BREAKDOWN_PROMPT)

    _assert_result_nonempty(result)
    # The corrupted checkpoint was replaced by a valid re-promotion after the fresh
    # breakdown re-ran, and the retired file never appears.
    _assert_promoted_breakdown_valid(ws)
    _assert_no_breakdown_result_json(ws)
