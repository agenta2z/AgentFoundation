"""Real-CLI integration test — MultiFlowDualInferencer documents OpenStartup.

End-to-end test exercising the full Round 7 composition with REAL CLIs:
- Two parallel dynamic-mode flows: ClaudeCodeCli + KiroCli.
- Cross-flow visibility (`visible_flows="all"`).
- Aggregator integration with `<FinalPlan>` + `<Winner>` extraction.
- Rule-based dispatch: `review_default=kiro_cli` with `priority_pool=[claude_cli]`
  for self-review avoidance; `fixer_match_winner=True` for winner-as-fixer.
- Real `target_path` pointing at the OpenStartup repo.

The task: produce an architecture-overview document for OpenStartup. Loose
shape assertions only; content correctness is not verifiable in CI.

**Cost / runtime**: ~45-90 min, $15-30+ per run depending on token usage and
how many iterations consensus takes. Manual / pre-release validation only;
NEVER run in routine CI.
"""

import os
import re
from pathlib import Path

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
    DualInferencerResponse,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_dual_inferencer import (
    MultiFlowDualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    DEFAULT_AGGREGATOR_PROMPT_TEMPLATE,
)

# Reuse the conftest's CLI-availability skip markers.
from .conftest import (  # noqa: E402
    DEFAULT_TIMEOUT,
    KIRO_AVAILABLE,
    skip_claude,
)

# Kiro is only required when MFDUAL_REAL_NO_KIRO != "1".
skip_kiro_unless_bypassed = pytest.mark.skipif(
    not KIRO_AVAILABLE and os.environ.get("MFDUAL_REAL_NO_KIRO") != "1",
    reason="kiro-cli unavailable; set MFDUAL_REAL_NO_KIRO=1 to fall back to Claude",
)


# ---------------------------------------------------------------------------
# OpenStartup discovery (env override or default sibling-of-AgentFoundation)
# ---------------------------------------------------------------------------

OPENSTARTUP_PATH = Path(
    os.environ.get(
        "OPENSTARTUP_PATH",
        # default: AgentFoundation/test/.../test_real_integration → up 6 → CoreProjects/OpenStartup
        str(Path(__file__).resolve().parents[6] / "OpenStartup"),
    )
)

skip_openstartup = pytest.mark.skipif(
    not OPENSTARTUP_PATH.exists(),
    reason=f"OpenStartup repo not found at {OPENSTARTUP_PATH}",
)


# ---------------------------------------------------------------------------
# Templates and parsers (defined inline to keep the test self-contained)
# ---------------------------------------------------------------------------

MULTIFLOW_FOLLOWUP_W_DECISION_TEMPLATE = """\
Original task:
{{ input }}

Your previous draft:
{{ your_prev }}

{% if visible_plans %}
Other teams' latest drafts:
{% for idx, plan in visible_plans.items() %}
--- Flow {{ idx }} ---
{{ plan if plan else '(no draft yet)' }}

{% endfor %}
{% endif %}\
Refine your draft, integrating the strongest ideas from any other teams' drafts.
Decide whether your latest draft is comprehensive enough to stop iterating.

Output STRICTLY in this format:
<Plan>
...your refined draft...
</Plan>
<Decision>continue|stop</Decision>
"""

AGGREGATOR_W_WINNER_TEMPLATE = """\
Original task:
{{ input }}

Each team's final draft:
{% for idx, plan in worker_plans.items() %}
=== Flow {{ idx }} ===
{{ plan if plan else '(no output)' }}

{% endfor %}\
Produce the FINAL integrated documentation plan by INTEGRATING the best
parts of every team's draft above. DO NOT inspect the codebase further --
work strictly from the drafts provided. Apply critical thinking: if the
drafts conflict, choose the more concrete and better-justified option;
flag any unresolved ambiguity in the plan itself rather than guessing.

Then identify which flow contributed the strongest draft (by index,
0-based) based on depth, accuracy, and critical insight.

Output STRICTLY in this format:
<FinalPlan>
...the final integrated documentation plan...
</FinalPlan>
<Winner>flow_0|flow_1</Winner>
"""


def _parse_decision_stop(state, result):
    m = re.search(r"<Decision>\s*(stop|continue)\s*</Decision>", str(result), re.IGNORECASE)
    return m and m.group(1).lower() == "stop"


def _parse_winner_tag(s):
    m = re.search(r"<Winner>\s*flow_(\d+)\s*</Winner>", str(s))
    return int(m.group(1)) if m else None


def _parse_finalplan_tag(s):
    m = re.search(r"<FinalPlan>([\s\S]*?)</FinalPlan>", str(s))
    return m.group(1).strip() if m else str(s)


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


# Profile selector: choose via env var ``MFDUAL_REAL_PROFILE``.
#   "shallow" (default for first runs): max_dynamic_steps=2, max_iterations=1,
#       max_consensus_attempts=1. ~5-10 LLM calls, $1-3, 5-10 min.
#   "deep": max_dynamic_steps=6, max_iterations=3, max_consensus_attempts=2.
#       ~25+ LLM calls, $15-30+, 45-90 min. Pre-release validation.
_PROFILE = os.environ.get("MFDUAL_REAL_PROFILE", "shallow").lower()
_PROFILES = {
    "shallow": {"max_dynamic_steps": 2, "max_iterations": 1, "max_consensus_attempts": 1,
                "min_response_len": 200},
    "deep":    {"max_dynamic_steps": 6, "max_iterations": 3, "max_consensus_attempts": 2,
                "min_response_len": 500},
}


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(DEFAULT_TIMEOUT * 60)  # ~120 min upper bound
@skip_claude
@skip_kiro_unless_bypassed
@skip_openstartup
async def test_multi_flow_dual_documents_openstartup(tmp_path):
    """Real Claude+Kiro composition: produce architecture overview for OpenStartup.

    Profile is selected via env var ``MFDUAL_REAL_PROFILE``:
      - ``shallow`` (default): smoke wiring check, ~$1-3, 5-10 min.
      - ``deep``: full stress test (per plan §A.15), ~$15-30+, 45-90 min.

    Skipped automatically when ``claude``, ``kiro-cli``, or OpenStartup
    are unavailable.
    """
    if _PROFILE not in _PROFILES:
        pytest.skip(f"unknown MFDUAL_REAL_PROFILE={_PROFILE!r}; "
                    f"expected one of {sorted(_PROFILES)}")
    p = _PROFILES[_PROFILE]
    print(f"\n[mfdual-real] profile={_PROFILE} {p}")

    # Lazy-import the CLI inferencers so this module is collectable even when
    # the CLIs are absent — the @skip_claude / @skip_kiro decorators above
    # handle skipping cleanly.
    from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
        ClaudeCodeCliInferencer,
    )
    from agent_foundation.common.inferencers.agentic_inferencers.external.kiro.kiro_cli_inferencer import (
        KiroCliInferencer,
    )

    claude = ClaudeCodeCliInferencer(
        target_path=str(OPENSTARTUP_PATH),
        cache_folder=str(tmp_path / "claude_cache"),
        model_name="sonnet",
        permission_mode="bypassPermissions",
        idle_timeout_seconds=600,
        resume_with_saved_results=True,
    )
    # Second flow: Kiro by default. Set MFDUAL_REAL_NO_KIRO=1 to swap in a
    # second Claude (opus model) when Kiro is unavailable (quota exhausted, etc.).
    # The variable is named "kiro" throughout to keep wiring identical regardless.
    if os.environ.get("MFDUAL_REAL_NO_KIRO") == "1":
        kiro = ClaudeCodeCliInferencer(
            target_path=str(OPENSTARTUP_PATH),
            cache_folder=str(tmp_path / "claude2_cache"),
            model_name="opus",
            permission_mode="bypassPermissions",
            idle_timeout_seconds=600,
            resume_with_saved_results=True,
        )
        print("[mfdual-real] MFDUAL_REAL_NO_KIRO=1 -> using Claude opus instead of Kiro")
    else:
        kiro = KiroCliInferencer(
            target_path=str(OPENSTARTUP_PATH),
            cache_folder=str(tmp_path / "kiro_cache"),
            model_name="auto",
            trust_mode="all",
            idle_timeout_seconds=600,
            resume_with_saved_results=True,
        )
    # Aggregator: a third Claude instance (cheaper than introducing a
    # different CLI type).
    aggregator = ClaudeCodeCliInferencer(
        target_path=str(OPENSTARTUP_PATH),
        cache_folder=str(tmp_path / "agg_cache"),
        model_name="sonnet",
        permission_mode="bypassPermissions",
        idle_timeout_seconds=600,
    )

    DOC_TASK = (
        "You are creating a documentation PLAN for the OpenStartup codebase "
        "(located at the current working directory). Your output is the PLAN "
        "itself, NOT the documentation. Carefully and thoroughly investigate "
        "the codebase first; explore directory structure, read key entry "
        "points, sample representative modules, and trace at least one full "
        "request and one full data flow end-to-end before drafting.\n\n"
        "The plan must be comprehensive, deep, insightful, accurate, and "
        "demonstrate critical thinking. Specifically include:\n"
        "  1. Codebase landscape — top-level structure, languages, frameworks, "
        "build/runtime layout. Cite concrete file paths.\n"
        "  2. Proposed documentation outline — the table of contents you "
        "recommend, organized by audience (newcomer / contributor / operator). "
        "Justify each section's existence.\n"
        "  3. Per-section depth notes — for each section, what specifically "
        "needs to be documented, which files/modules to reference, and what "
        "level of detail is appropriate.\n"
        "  4. Integration surfaces & data models — external APIs consumed/"
        "exposed, persistence layer, message contracts. Be concrete.\n"
        "  5. Critical-thinking section — surface ambiguities, risky "
        "abstractions, undocumented invariants, areas where the code's "
        "behavior is non-obvious or where a doc would mislead if written "
        "naively. Identify the questions a documenter MUST answer (or escalate) "
        "before writing.\n"
        "  6. Effort & sequencing — estimate relative effort per section and "
        "propose a writing order with dependencies.\n"
        "Be concrete throughout; cite real file paths from this codebase. "
        "Avoid generic boilerplate that could apply to any project."
    )

    mfdi = MultiFlowDualInferencer(
        flow_configs=[
            {
                "input": DOC_TASK,
                "initial_inferencer": claude,
                "followup_inferencer": claude,
                "end_condition": _parse_decision_stop,
                "max_dynamic_steps": p["max_dynamic_steps"],
            },
            {
                "input": DOC_TASK,
                "initial_inferencer": kiro,
                "followup_inferencer": kiro,
                "end_condition": _parse_decision_stop,
                "max_dynamic_steps": p["max_dynamic_steps"],
            },
        ],
        visible_flows="all",
        multi_flow_aggregator_inferencer=aggregator,
        multi_flow_aggregator_prompt=AGGREGATOR_W_WINNER_TEMPLATE,
        multi_flow_followup_prompt=MULTIFLOW_FOLLOWUP_W_DECISION_TEMPLATE,
        multi_flow_winner_parser=_parse_winner_tag,
        multi_flow_response_parser=_parse_finalplan_tag,

        # Round 7 rule-based dispatch
        review_default=kiro,                  # default reviewer
        review_priority_pool=[claude],        # if Kiro wins, swap to Claude
        fixer_match_winner=True,              # fixer = winning flow's CLI

        consensus_config=ConsensusConfig(
            max_iterations=p["max_iterations"],
            max_consensus_attempts=p["max_consensus_attempts"],
        ),
        workspace=str(tmp_path / "mfdi_workspace"),
    )

    async with mfdi:
        result = await mfdi.ainfer(
            "Document the OpenStartup codebase architecture"
        )

    # ----- Loose shape assertions (real LLM output is non-deterministic) -----
    assert isinstance(result, DualInferencerResponse)
    assert isinstance(result.base_response, str)
    assert len(result.base_response.strip()) > p["min_response_len"], (
        f"profile={_PROFILE!r}: expected >{p['min_response_len']} chars, "
        f"got {len(result.base_response.strip())}"
    )
    # Winner was identified (LLM emitted <Winner>flow_X</Winner>)
    winner_idx = mfdi.base_inferencer.get_winner_flow_idx()
    assert winner_idx in (0, 1), f"unexpected winner_idx: {winner_idx}"
    # consensus_history populated
    assert len(result.consensus_history) >= 1

    # ----- Deterministic correctness checks (given a winner_idx) -----
    expected_reviewer = claude if winner_idx == 1 else kiro
    expected_fixer = claude if winner_idx == 0 else kiro
    # Self-review avoidance: reviewer is NOT the winner's CLI
    assert mfdi.review_inferencer is expected_reviewer, (
        f"self-review-avoidance broken: expected {expected_reviewer}, got "
        f"{mfdi.review_inferencer}"
    )
    # Winner-as-fixer
    assert mfdi.fixer_inferencer is expected_fixer, (
        f"winner-as-fixer broken: expected {expected_fixer}, got "
        f"{mfdi.fixer_inferencer}"
    )

    # ----- Workspace artifacts produced -----
    # Verify the workspace has SOME checkpoint content (BTA's expansion
    # checkpoints, worker final_results, aggregator stream cache, etc.).
    # We don't pin to a specific path because the actual checkpoint layout
    # depends on which optional features fire (DualInferencer step
    # checkpoints only land when `enable_checkpoint=True` AND a fix step
    # actually runs, etc.).
    workspace_root = tmp_path / "mfdi_workspace"
    assert workspace_root.exists(), f"workspace root missing: {workspace_root}"
    checkpoint_artifacts = list((workspace_root / "checkpoints").rglob("*"))
    children_artifacts = list((workspace_root / "children").rglob("*")) \
        if (workspace_root / "children").exists() else []
    artifact_count = len([p for p in checkpoint_artifacts if p.is_file()]) \
        + len([p for p in children_artifacts if p.is_file()])
    assert artifact_count > 0, (
        f"expected workspace to contain at least 1 artifact file under "
        f"checkpoints/ or children/, got 0 (workspace={workspace_root})"
    )
