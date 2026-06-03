"""Real-CLI integration test for `/task --agent-config <yaml>`.

Topology: outer Dual (review-and-fix) wrapping two distinct BTA instances,
each with PTI workers whose planner is MultiFlowDual and whose executor
is a Dual. Five inferencer types in their natural roles.

YAML lives at: src/openteam/server/resources/tools/task/topologies/
  breakdown-multiflow-plan-then-implement.yaml (the production path).
The test goes through `_run_topology()` -- the same path `/task` slash
uses -- exercising the full real loader (alias resolution, `_-prefix`
cascade, OmegaConf interpolation, `_partial_` auto-injection).

Two test functions:

  * ``test_yaml_smoke_instantiate`` -- unit-level (no LLM), ~5s. Catches
    alias / interpolation / cascade-injection regressions.
  * ``test_real_dual_outside_bta_pti_mfdual_dual`` -- @pytest.mark.integration
    real-CLI. Profile-parametrized via ``BMP_REAL_PROFILE`` env var
    (shallow / medium / deep). Default ``shallow``, ~30-60min, $30-60.

Cost gate: real-CLI test is skipped without ``-m integration`` and
without claude CLI on PATH.
"""

from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Paths and discovery
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
YAML_PATH = _HERE.parents[4] / "src" / "openteam" / "server" / "resources" / "tools" / "task" / "topologies" / "breakdown-multiflow-plan-then-implement.yaml"

# OpenStartup repo (the target_path for ClaudeCodeCli leaves to investigate)
OPENSTARTUP_PATH = Path(
    os.environ.get(
        "OPENSTARTUP_PATH",
        # default: this file at OpenStartup/test/openteam/resources/tools/task/
        # parents[0]=task, [1]=tools, [2]=resources, [3]=openteam,
        # [4]=test, [5]=OpenStartup
        str(_HERE.parents[4]),
    )
)
TEMPLATES_DIR = OPENSTARTUP_PATH / "src" / "openteam" / "server" / "resources" / "prompt_templates"


# ---------------------------------------------------------------------------
# Skip markers (inlined; this dir has no shared conftest yet)
# ---------------------------------------------------------------------------

def _cli_available(command: str) -> bool:
    try:
        result = subprocess.run(
            f"{command} --version",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


CLAUDE_AVAILABLE = _cli_available("claude")

skip_claude = pytest.mark.skipif(
    not CLAUDE_AVAILABLE,
    reason="claude CLI not available (not in PATH or non-zero exit)",
)
skip_openstartup = pytest.mark.skipif(
    not OPENSTARTUP_PATH.exists(),
    reason=f"OpenStartup repo not found at {OPENSTARTUP_PATH}",
)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

PROFILES = {
    # quick: tightest run for end-to-end verification (~10-30 min). Single
    # outer Dual iter, 2 workers per BTA, 1 inner iter, 1 flow step per
    # MFDual flow. Asserts only minimal completion semantics.
    "quick": {
        "outer_dual_iter": 0,           # No fixer pass — base PTI's output is final
        "bta_workers": 2,
        "mfdual_iter": 1,
        "inner_dual_iter": 1,
        "flow_dynamic_steps": 1,        # NEW: minimal per-flow followups
        "min_response_len": 100,
    },
    "shallow": {
        "outer_dual_iter": 1,
        "bta_workers": 2,
        "mfdual_iter": 1,
        "inner_dual_iter": 1,
        "flow_dynamic_steps": 2,
        "min_response_len": 200,
    },
    "medium": {
        "outer_dual_iter": 1,
        "bta_workers": 2,
        "mfdual_iter": 2,
        "inner_dual_iter": 2,
        "flow_dynamic_steps": 3,
        "min_response_len": 400,
    },
    "deep": {
        "outer_dual_iter": 3,
        "bta_workers": 3,
        "mfdual_iter": 2,
        "inner_dual_iter": 2,
        "flow_dynamic_steps": 3,
        "min_response_len": 600,
    },
}
_PROFILE = os.environ.get("BMP_REAL_PROFILE", "shallow")


# ---------------------------------------------------------------------------
# Default inferencer (overridable via env var BMP_DEFAULT_INFERENCER).
# Drives the YAML's `${_params.default_inferencer}` interpolations through the
# `_params.default_inferencer` override. Examples: "ClaudeCodeCLI" (default),
# "RovoDevCLI". Any value listed in registered_targets.py works.
# ---------------------------------------------------------------------------
_DEFAULT_INFERENCER = os.environ.get("BMP_DEFAULT_INFERENCER", "ClaudeCodeCLI")


# ---------------------------------------------------------------------------
# Proactive AI Platform — codebase-understanding task fixtures
# ---------------------------------------------------------------------------
# Target codebase to analyze + output location for the generated docs. Both
# overridable via env vars so the same test can point at different codebases.
PAI_CODEBASE_PATH = Path(
    os.environ.get(
        "PAI_CODEBASE_PATH",
        "/Users/tchen7/MyProjects/atlassian_packages/proactive-ai-platform",
    )
)
PAI_DOCS_OUTPUT_PATH = Path(
    os.environ.get(
        "PAI_DOCS_OUTPUT_PATH",
        # _HERE.parents (zero-indexed, walking UP from task/):
        #   [0]=tools, [1]=resources, [2]=openteam, [3]=test, [4]=OpenStartup,
        #   [5]=CoreProjects. Reference docs live under
        #   OpenStartup/_dev/pai_hack/codebase_understanding (i.e. parents[4]).
        str(_HERE.parents[4] / "_dev" / "pai_hack" / "codebase_understanding"),
    )
)


# ---------------------------------------------------------------------------
# Task prompt
# ---------------------------------------------------------------------------

DOC_TASK = r"""Create detailed, comprehensive documentation for the codebase at:

  C:\Users\yxinl\OneDrive\Projects\PythonProjects\CoreProjects\_dev\responsible-ai-api

The documentation must be:
- Comprehensive: cover the codebase's purpose, architecture, public API, key
  modules, and any operational concerns
- Structured: organized into clearly-named sections with consistent depth
- Both human-readable and machine-friendly: use proper headings, code blocks,
  cross-references, and a glossary; emit content that downstream tools (Sphinx,
  search indexers) can parse
- Accurate, informative, insightful, and reflective of critical thinking:
  verify claims against the source code, do not speculate; flag ambiguities
  rather than paper over them; bring out non-obvious design decisions
- Cross-referenced: where relevant, link sections together using Sphinx
  `:ref:` references and a shared anchor map, with sufficient citations

Output format: reStructuredText source files placed under `docs/source/`
(one .rst file per major section). Then build HTML output via Sphinx:
  - Place `conf.py` under `docs/source/` (alabaster or sphinx-rtd-theme)
  - Place `index.rst` under `docs/source/` with a toctree referencing each
    section in a sensible reading order
  - Run `sphinx-build -b html docs/source docs/build/html` and report any
    warnings or errors

The deliverable is the documentation itself, NOT a documentation plan or
outline. Produce the actual prose, code excerpts, and reference material
that a reader can consume directly.

Tool-use constraints (IMPORTANT — failure to honor these will cause the run to
hang for hours):
- Confine ALL filesystem operations to the target codebase directory shown
  above. Do NOT run `find /`, `find C:\`, `grep -r` over `c:/Users`,
  `OneDrive`, `Desktop`, `Projects`, or any other drive-root or large parent
  directory. On this Windows host, recursive scans rooted at `/` or
  `c:/Users/...` traverse hundreds of GB and take hours; they will exceed
  the per-tool idle timeout and stall the workflow.
- Use the target codebase path explicitly when listing or searching files:
  e.g., `find <repo_root>/src -name '*.py'`, NOT `find / -name '*.py'`.
- Prefer the `Read`/`Glob`/`Grep` tools scoped to a known subdirectory over
  shell `find`/`grep -r` whenever possible.
- If you cannot locate a file with a bounded search, ASK or report the
  ambiguity rather than escalating to a drive-wide scan.
"""


# ===========================================================================
# Smoke test (no LLM)
# ===========================================================================

def test_yaml_smoke_instantiate(tmp_path, monkeypatch):
    """Validate the YAML loads and instantiates the new Topology B-extended
    shape without LLM calls.

    Topology B-extended:
        Outer Dual { base = PTI { planner = Dual{base=BTA{workers=MFDual},
        review}, executor = BTA{workers=Dual} }, review, fixer = PTI }

    Catches alias-registration regressions, `_-prefix` cascade bugs (incl.
    `_logger: auto`, `_debug_mode: true`), OmegaConf interpolation bugs,
    type mismatches, and template_variables resolution in seconds --
    before the real-CLI test burns money.
    """
    import agent_foundation.common.configs.registered_targets  # noqa: F401
    from rich_python_utils.config_utils import load_config, instantiate

    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
        DualInferencer,
    )
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
        BreakdownThenAggregateInferencer,
    )
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
        PlanThenImplementInferencer,
    )
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_dual_inferencer import (
        MultiFlowDualInferencer,
    )
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
        MultiFlowInferencer,
    )
    from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
        ClaudeCodeCliInferencer,
    )
    from agent_foundation.common.inferencers.flow_parsers import (
        parse_winner_tag, parse_decision_stop, parse_finalplan_tag,
    )

    cfg = load_config(
        str(YAML_PATH),
        overrides={
            "_target_path": str(OPENSTARTUP_PATH),
            "templates_dir": str(TEMPLATES_DIR),
            "_params.workspace_root": str(tmp_path / "ws"),
        },
    )
    inferencer = instantiate(cfg)

    # ----- Top-level: outer Dual wrapping PTI base + PTI fixer -----
    assert isinstance(inferencer, DualInferencer), type(inferencer).__name__
    assert isinstance(inferencer.base_inferencer, PlanThenImplementInferencer), (
        f"base must be PTI (Topology B-extended); got {type(inferencer.base_inferencer).__name__}"
    )
    assert isinstance(inferencer.fixer_inferencer, PlanThenImplementInferencer), (
        f"fixer must be PTI (mirror of base); got {type(inferencer.fixer_inferencer).__name__}"
    )
    assert inferencer.base_inferencer is not inferencer.fixer_inferencer, (
        "base and fixer must be distinct PTI instances"
    )
    assert isinstance(inferencer.review_inferencer, ClaudeCodeCliInferencer)
    # Outer reviewer judges the deliverable, not a plan — uses implementation/review
    assert inferencer.review_inferencer.template_root_space == "implementation", (
        f"outer reviewer must judge deliverable, not plan; got "
        f"template_root_space={inferencer.review_inferencer.template_root_space!r}"
    )
    assert inferencer.review_inferencer.template_key == "review"

    # ----- PTI's plan stage: Dual{base=BTA{workers=MFDual}, review} -----
    base_pti = inferencer.base_inferencer
    plan_dual = base_pti.planner_inferencer
    assert isinstance(plan_dual, DualInferencer), (
        f"plan stage must be Dual (review-and-fix the integrated plan); got {type(plan_dual).__name__}"
    )
    plan_bta = plan_dual.base_inferencer
    assert isinstance(plan_bta, BreakdownThenAggregateInferencer), (
        f"plan Dual.base must be BTA (decompose planning); got {type(plan_bta).__name__}"
    )
    assert plan_bta.name == "plan_bta"
    assert isinstance(plan_dual.review_inferencer, ClaudeCodeCliInferencer)

    # plan BTA worker_factory yields MFDual instances (multi-perspective per sub-plan)
    plan_factory = plan_bta.worker_factory["__default__"]
    assert isinstance(plan_factory, functools.partial), (
        "plan BTA worker_factory.__default__ should be functools.partial"
    )
    sample_mfdual = plan_factory()
    assert isinstance(sample_mfdual, MultiFlowDualInferencer), (
        f"plan BTA worker must be MFDual; got {type(sample_mfdual).__name__}"
    )
    assert sample_mfdual.propagate_runtime_input is True, (
        "MFDual planner MUST have propagate_runtime_input=true so runtime "
        "subtask description reaches the flows (architectural fix)"
    )
    assert sample_mfdual.inject_upstream_artifacts is True, (
        "MFDual planner MUST have inject_upstream_artifacts=true so "
        "{{ upstream_artifacts }} slot in target wrappers is populated"
    )
    # MFDual auto-constructs an internal MultiFlow as base_inferencer
    assert isinstance(sample_mfdual.base_inferencer, MultiFlowInferencer)
    assert sample_mfdual.base_inferencer.propagate_runtime_input is True, (
        "MultiFlow inside MFDual must also have propagate_runtime_input=true (forwarded)"
    )
    assert sample_mfdual.base_inferencer.inject_upstream_artifacts is True, (
        "MultiFlow inside MFDual must also have inject_upstream_artifacts=true (forwarded)"
    )
    assert len(sample_mfdual.flow_configs) == 2, (
        f"expected 2 parallel flows by default; got {len(sample_mfdual.flow_configs)}"
    )
    # Parsers wired correctly
    assert sample_mfdual.multi_flow_winner_parser is parse_winner_tag, (
        "winner parser alias not resolved correctly"
    )
    assert sample_mfdual.multi_flow_response_parser is parse_finalplan_tag, (
        "response parser alias (FinalPlanParser) not resolved correctly"
    )
    # Per-flow end_condition is the decision-stop parser
    assert sample_mfdual.flow_configs[0]["end_condition"] is parse_decision_stop, (
        "flow end_condition alias (DecisionStopParser) not resolved correctly"
    )

    # plan BTA aggregator uses the SHARED `aggregation` task_* variants for
    # pure prose integration. Under Option A, NO structured-output addenda
    # fire here — the exec BTA's breakdown_inferencer is responsible for
    # decomposing the prose plan into per-section subtasks. The aggregator's
    # only job is faithful integration of upstream sub-plans.
    plan_agg = plan_bta.aggregator_inferencer
    assert isinstance(plan_agg, ClaudeCodeCliInferencer)
    assert plan_agg.template_root_space == "plan"
    # Refactor 12+13: BTA.SLOT_DEFAULTS now injects the full aggregation
    # triplet via template_version + None-keyed template_variables.
    # task_response_format will be present (None) but renders to the empty
    # string at feed time because the wrapper template never references
    # {{ task_response_format }} for the plan BTA aggregator. The structured-
    # output addenda (winner_pick / iteration_judgment) live on MFDual via
    # template_extra_feed flags, not on the plan BTA aggregator slot itself.
    assert plan_agg.template_version == "aggregation"
    assert "task_preamble" in plan_agg.template_variables
    assert "task_instructions" in plan_agg.template_variables
    assert "task_response_format" in plan_agg.template_variables
    # Plan BTA aggregator must NOT have any output-schema addendum flag set.
    # If any of these fire, decomposition logic has leaked back into the
    # plan stage (Option A violation).
    plan_agg_flags = plan_agg.template_extra_feed or {}
    for _flag in ("include_iteration_judgment", "include_winner_pick", "include_execution_subtasks"):
        assert not plan_agg_flags.get(_flag, False), (
            f"Plan BTA aggregator must not set {_flag!r} (Option A: aggregator "
            f"does pure prose integration; per-section structuring is the exec "
            f"BTA breakdown_inferencer's job)"
        )

    # Plan BTA opts into modern slot semantics via the class-level flag
    # (replaces the old aggregator_prompt_builder factory wiring). When True,
    # BTA's _build_agg_input internally pushes formatted worker outputs into
    # template_extra_feed["upstream_artifacts"], forwards the breakdown's
    # aggregation_guidance, and returns original_query as inference_input.
    assert plan_bta.inject_upstream_artifacts_to_aggregator is True, (
        "Plan BTA must set inject_upstream_artifacts_to_aggregator=True to "
        "wire the {{ upstream_artifacts }} slot (replaces the old "
        "UpstreamInjectingAggregatorPromptBuilder factory pattern)"
    )
    # Custom prompt_builder should NOT be set when the flag is used (the flag
    # IS the modern path; the factory was the legacy escape hatch).
    assert plan_bta.aggregator_prompt_builder is None, (
        "Plan BTA must not set both aggregator_prompt_builder AND the flag — "
        "the flag is the canonical pattern, the factory the deprecated one"
    )

    # Option B (breakdown → aggregator handoff): the new internal path
    # must plumb the breakdown's aggregation_guidance into the aggregator's
    # template_extra_feed. Verify by directly calling the helper that
    # _build_agg_input uses.
    plan_bta._last_aggregation_guidance = (
        "Treat subtask 1 as backbone; layer subtasks 2-3 onto it; preserve glossary."
    )
    plan_bta._inject_aggregator_extra_feed(
        worker_results=["worker 1 output", "worker 2 output"],
    )
    assert plan_bta.aggregator_inferencer.template_extra_feed.get(
        "aggregation_guidance"
    ) == (
        "Treat subtask 1 as backbone; layer subtasks 2-3 onto it; preserve glossary."
    ), (
        "_inject_aggregator_extra_feed must forward _last_aggregation_guidance "
        "into the aggregator's template_extra_feed['aggregation_guidance']"
    )
    # And the upstream_artifacts slot must be populated with formatted
    # worker outputs (matches the ### Result N format used by both the
    # legacy default and the new injection path).
    upstream = plan_bta.aggregator_inferencer.template_extra_feed.get("upstream_artifacts")
    assert upstream is not None and "### Result 1" in upstream and "### Result 2" in upstream, (
        f"upstream_artifacts must contain formatted ### Result N entries; got {upstream!r}"
    )
    # Stale aggregation_guidance must be dropped when the next breakdown
    # produces no guidance (else previous-call guidance leaks into the
    # current aggregator prompt).
    plan_bta._last_aggregation_guidance = None
    plan_bta._inject_aggregator_extra_feed(
        worker_results=["worker 1 output"],
    )
    assert "aggregation_guidance" not in plan_bta.aggregator_inferencer.template_extra_feed, (
        "Stale aggregation_guidance must be dropped when the next breakdown "
        "does not produce one"
    )

    # MFDual's final aggregator uses the SHARED structured-aggregation triplet.
    # Refactor 13: template_version drives expansion; per-key values are None.
    mfdual_agg = sample_mfdual.multi_flow_aggregator_inferencer
    assert isinstance(mfdual_agg, ClaudeCodeCliInferencer)
    assert mfdual_agg.template_version == "aggregation"
    assert "task_instructions" in mfdual_agg.template_variables
    assert mfdual_agg.template_extra_feed.get("include_winner_pick") is True

    # MFDual's per-flow followup_inferencer uses SHARED template with
    # include_iteration_judgment flag (asks the model to judge whether further
    # iteration adds value; the response-format addendum then specifies emit a
    # JSON `iteration_judgment` block with `decision: continue|stop`).
    flow_followup = sample_mfdual.flow_configs[0]["followup_inferencer"]
    assert isinstance(flow_followup, ClaudeCodeCliInferencer)
    assert flow_followup.template_version == "aggregation"
    assert "task_instructions" in flow_followup.template_variables
    assert flow_followup.template_extra_feed.get("include_iteration_judgment") is True

    # ----- PTI's exec stage: bare BTA{workers=Dual{base+review}} -----
    exec_bta = base_pti.executor_inferencer
    assert isinstance(exec_bta, BreakdownThenAggregateInferencer), (
        f"exec stage must be bare BTA (no surrounding Dual); got {type(exec_bta).__name__}"
    )
    assert exec_bta.name == "exec_bta"
    # Exec BTA also opts into modern slot semantics. Closes the historical
    # gap where exec_bta lacked a prompt_builder and worker outputs ended up
    # in the wrapper's {{ input }} slot (mislabeled as "Original User Request").
    assert exec_bta.inject_upstream_artifacts_to_aggregator is True, (
        "exec_bta must set inject_upstream_artifacts_to_aggregator=True so the "
        "exec aggregator receives upstream_artifacts + aggregation_guidance "
        "via template_extra_feed, with original BTA query in {{ input }}"
    )
    assert exec_bta.aggregator_prompt_builder is None
    # Exec breakdown uses the generic task_breakdown wrapper — no use-case-specific
    # task_instructions variant. Same generic-wrapper philosophy applies to the
    # exec workers and aggregator below.
    exec_breakdown = exec_bta.breakdown_inferencer
    assert exec_breakdown.template_root_space == "task_breakdown"
    assert not exec_breakdown.template_variables.get("task_instructions"), (
        "exec_breakdown must NOT set a task_instructions variant — the generic "
        "wrapper carries the cognitive load"
    )
    # Exec workers are Duals (per-section write+review+fix). They use ONLY the
    # generic implementation/main/initial.jinja2 wrapper — no task_instructions
    # variant. The LLM infers the deliverable shape from the subtask description.
    exec_factory = exec_bta.worker_factory["__default__"]
    sample_exec_dual = exec_factory()
    assert isinstance(sample_exec_dual, DualInferencer), (
        f"exec BTA worker must be Dual (per-section write+review+fix); got {type(sample_exec_dual).__name__}"
    )
    assert isinstance(sample_exec_dual.base_inferencer, ClaudeCodeCliInferencer)
    assert sample_exec_dual.base_inferencer.template_root_space == "implementation"
    assert not (sample_exec_dual.base_inferencer.template_variables or {}).get("task_instructions"), (
        "exec worker base must NOT set a task_instructions variant — the generic "
        "wrapper plus the subtask description carry all needed context"
    )
    # Exec aggregator: uses the implementation-namespace aggregation preamble
    # to consume {{ upstream_artifacts }} and {{ aggregation_guidance }}
    # (populated by BTA's inject_upstream_artifacts_to_aggregator path).
    # Refactor 12+13: BTA now defaults the full triplet via SLOT_DEFAULTS.
    # The newly-authored implementation/.../task_instructions/aggregation.jinja2
    # provides aggregator-role framing (NOT the worker-style default.jinja2),
    # so prompt content stays role-correct. Task-specific instructions still
    # ride in the breakdown's aggregation_guidance via the wrapper template's
    # {{ aggregation_guidance }} slot.
    exec_agg = exec_bta.aggregator_inferencer
    assert exec_agg.template_root_space == "implementation"
    assert exec_agg.template_version == "aggregation"
    assert "task_preamble" in exec_agg.template_variables
    assert "task_instructions" in exec_agg.template_variables
    # Verify the implementation-namespace aggregation task_instructions file
    # exists (authored to make the BTA flip semantically safe). load_variable
    # must return real aggregator framing, NOT the worker-style default.jinja2.
    _impl_vars = exec_agg.template_manager.load_variables(
        {"task_instructions": "aggregation", "task_preamble": "aggregation"},
        root_space="implementation",
    )
    impl_agg_instructions_text = _impl_vars["task_instructions"]
    assert "aggregating" in impl_agg_instructions_text.lower(), (
        "implementation/main/_variables/task_instructions/aggregation.jinja2 "
        "must contain aggregator-role framing, not worker-style instructions "
        "(which would be the default.jinja2 fallback)"
    )
    impl_agg_preamble_text = _impl_vars["task_preamble"]
    assert "upstream_artifacts" in impl_agg_preamble_text
    assert "aggregation_guidance" in impl_agg_preamble_text
    # Implementation aggregation is intentionally simpler than plan's:
    # no MFDual-specific addendum branches (workflows are non-overlapping
    # at execution time; no parallel exploration to judge or winner to pick).
    assert "include_iteration_judgment" not in impl_agg_preamble_text, (
        "implementation aggregation preamble must NOT include MFDual-specific "
        "branches — execution stage has no MFDual"
    )
    assert "include_winner_pick" not in impl_agg_preamble_text

    # ----- _-prefix cascade reached every leaf -----
    # _logger: auto → JsonLogger per leaf workspace; _debug_mode: true cascaded
    # too. Pick a deep leaf and verify all the cascade fields landed.
    deep_leaf = sample_mfdual.flow_configs[0]["initial_inferencer"]
    assert isinstance(deep_leaf, ClaudeCodeCliInferencer)
    # Model alias resolved
    assert deep_leaf.model_name in ("opus[1m]", "opus", "claude-opus-4-7[1m]"), (
        f"unexpected model_name on deep leaf: {deep_leaf.model_name!r}"
    )
    # _target_path cascade reached deep leaves
    assert deep_leaf.target_path == str(OPENSTARTUP_PATH), (
        f"_target_path cascade failed for deep leaf; got {deep_leaf.target_path!r}"
    )
    # _idle_timeout_seconds + _output_path cascades
    assert deep_leaf.idle_timeout_seconds == 600
    assert deep_leaf.output_path == "output.md"
    # _logger: auto either reached the leaf as the literal "auto" sentinel (still
    # deferred — workspace assignment hasn't happened for this fresh-from-factory
    # leaf), OR has already resolved to a concrete logger dict (if workspace
    # propagated). Both are acceptable; the cascade itself MUST have hit:
    assert deep_leaf.logger is not None, "_logger cascade missed deep leaf entirely"
    assert deep_leaf.logger == "auto" or not isinstance(deep_leaf.logger, str), (
        f"_logger value should be 'auto' or a resolved logger; got {deep_leaf.logger!r}"
    )
    # _debug_mode cascade reached deep leaf
    assert deep_leaf.debug_mode is True, "_debug_mode: true cascade did not reach deep leaf"

    # Verify per-leaf logger DID resolve where workspace was already assigned —
    # the plan BTA's aggregator (constructed at YAML instantiation, gets workspace
    # via _propagate_workspace_to_children before the test inspects it).
    assert plan_agg.logger is not None and not isinstance(plan_agg.logger, str), (
        f"plan BTA aggregator's _logger: auto should have resolved to a JsonLogger; "
        f"got {plan_agg.logger!r}"
    )

    # ----- Workspace propagation: outer Dual → PTI children → BTAs -----
    # After Refactor 1, the YAML carries only the top-level `workspace_root`.
    # The outer Dual creates its `_workspace` from that, then propagation
    # cascades through the inferencer tree via each `_workspace` setter.
    # Each child gets `parent_workspace.child("<attr_name>")`.
    assert base_pti._workspace is not None, "base PTI didn't receive propagated workspace"
    assert base_pti._workspace.root.replace("\\", "/").endswith(
        "/children/base_inferencer"
    ), f"base PTI workspace root: {base_pti._workspace.root!r}"
    fixer_pti = inferencer.fixer_inferencer
    assert fixer_pti._workspace is not None, "fixer PTI didn't receive propagated workspace"
    assert fixer_pti._workspace.root.replace("\\", "/").endswith(
        "/children/fixer_inferencer"
    ), f"fixer PTI workspace root: {fixer_pti._workspace.root!r}"
    # Plan BTA: under planner_inferencer (Dual) under base_inferencer (PTI)
    # under outer Dual. Path mirrors the topology.
    assert plan_bta._workspace is not None, (
        "plan_bta should receive a workspace via the propagation cascade"
    )
    assert plan_bta._workspace.root.replace("\\", "/").endswith(
        "/children/planner_inferencer/children/base_inferencer"
    ), f"plan_bta workspace root: {plan_bta._workspace.root!r}"
    # Exec BTA: directly under base_inferencer (PTI) under outer Dual,
    # via the executor_inferencer slot.
    assert exec_bta._workspace is not None, (
        "exec_bta should receive a workspace via the propagation cascade"
    )
    assert exec_bta._workspace.root.replace("\\", "/").endswith(
        "/children/executor_inferencer"
    ), f"exec_bta workspace root: {exec_bta._workspace.root!r}"

    # ----- Template rendering smoke checks -----
    # Generic task_breakdown wrapper (no use-case-specific task_instructions).
    # The breakdown LLM gets the integrated plan as `input` plus the wrapper's
    # universal decomposition guidance and produces subtasks; downstream
    # workers are self-sufficient (no args contract).
    rendered_breakdown = exec_breakdown.template_manager(
        "initial",
        active_template_root_space="task_breakdown",
        input="(integrated documentation plan in free-form prose here)",
        task_preamble="",
    )
    # The wrapper must carry the four-criteria self-check (Comprehensive,
    # Independent, Non-conflicting, Actionable) — the conflict-handling
    # criterion is what lets the LLM encode unavoidable conflicts as
    # subtask_dependencies rather than producing parallel-unsafe siblings.
    for _criterion in ("Comprehensive", "Independent", "Non-conflicting", "Actionable"):
        assert _criterion in rendered_breakdown, (
            f"task_breakdown wrapper must include the '{_criterion}' decomposition criterion"
        )
    assert "subtask_dependencies" in rendered_breakdown, (
        "task_breakdown wrapper must instruct the LLM how to encode "
        "subtask_dependencies (used both for logical ordering and conflict "
        "serialization)"
    )

    # The breakdown wrapper must instruct the LLM to also fill an
    # `aggregation_guidance` field. Without this, the breakdown's reasoning
    # about how to reconstruct the integrated artifact is lost between
    # breakdown and aggregator stages.
    assert "aggregation_guidance" in rendered_breakdown, (
        "task_breakdown wrapper must instruct the breakdown LLM to emit an "
        "`aggregation_guidance` field — that field carries reconstruction "
        "guidance forward to the aggregator (Option B for breakdown→aggregator "
        "handoff)"
    )
    assert "Aggregation Guidance" in rendered_breakdown, (
        "task_breakdown wrapper must include the 'Aggregation Guidance' "
        "instructional section that explains what the LLM should put in the "
        "`aggregation_guidance` field"
    )

    # task_preamble: aggregation must consume the breakdown's aggregation_guidance
    # via {% if aggregation_guidance %} so plan BTA aggregators see Option B
    # plumbing (the breakdown's reconstruction strategy reaches the aggregator).
    _plan_vars = plan_agg.template_manager.load_variables(
        {"task_preamble": "aggregation", "task_instructions": "aggregation",
         "task_response_format": "aggregation"},
        root_space="plan",
    )
    agg_preamble_text = _plan_vars["task_preamble"]
    assert "aggregation_guidance" in agg_preamble_text, (
        "task_preamble/aggregation.jinja2 must reference {{ aggregation_guidance }} "
        "so the breakdown's reconstruction guidance reaches the aggregator"
    )
    assert "{%- if aggregation_guidance %}" in agg_preamble_text or \
           "{% if aggregation_guidance %}" in agg_preamble_text, (
        "task_preamble must gate aggregation_guidance behind an `{% if %}` so "
        "breakdowns that don't emit guidance render cleanly"
    )

    # aggregation task_instructions: terse cognitive guidance (compare inputs,
    # integrate faithfully, maintain structure consistency, avoid hacky/ad-hoc).
    # NO structured-output addenda — those moved to task_response_format.
    int_agg_text = _plan_vars["task_instructions"]
    assert "consolidated artifact" in int_agg_text, (
        "aggregation task_instructions should mention 'consolidated artifact' (the integration goal)"
    )
    assert "DO NOT be hacky" in int_agg_text, (
        "aggregation task_instructions should include the rigor reminder"
    )
    # The {% if %} addenda for output schema have moved to task_response_format
    assert "include_execution_subtasks" not in int_agg_text, (
        "aggregation task_instructions should NOT contain output-schema addenda — "
        "those belong in task_response_format/aggregation.jinja2"
    )

    # Structured-output schema lives in task_response_format/aggregation.jinja2
    response_format_text = _plan_vars["task_response_format"]
    # Two surviving addenda under Option A: iteration_judgment (per-flow followup)
    # and winner_pick (MFDual final aggregator). No execution_breakdown — the
    # plan aggregator emits prose, exec BTA breakdown does decomposition.
    assert "include_iteration_judgment" in response_format_text
    assert "include_winner_pick" in response_format_text
    assert "include_execution_subtasks" not in response_format_text, (
        "Option A: include_execution_subtasks must be removed from "
        "task_response_format/aggregation.jinja2 — decomposition belongs "
        "to the exec BTA breakdown_inferencer, not the plan aggregator"
    )
    # JSON-fenced output schema for the surviving addenda.
    assert "```json iteration_judgment" in response_format_text
    assert "```json winner_pick" in response_format_text
    assert "```json execution_breakdown" not in response_format_text
    assert "winner_index" in response_format_text
    # Structured execution-subtask schema must NOT appear in plan response format
    assert "execution_subtasks" not in response_format_text
    assert "shared_context" not in response_format_text

    # Verify the per-step followup formatter produces clean text.
    # The formatter is now an inlined Python method on MultiFlowInferencer
    # (no longer a Jinja template under prompt_templates/). The original task
    # is intentionally NOT included — under inject_upstream_artifacts=True,
    # the wrapper template's {{ input }} slot carries it separately.
    formatted_followup = sample_mfdual.base_inferencer._format_followup_input(
        your_prev="prev draft",
        flow_idx=0,
        step_idx=2,
        visible_plans={1: "peer 1"},
    )
    assert "prev draft" in formatted_followup
    assert "peer 1" in formatted_followup
    assert "flow 0" in formatted_followup
    assert "step 1" in formatted_followup  # prev_step_idx = step_idx - 1
    # Critical: data formatter carries NO cognitive instructions (those live in
    # the followup_inferencer's task_instructions:aggregation wrapper).
    assert "Decision" not in formatted_followup, (
        "_format_followup_input should be a thin DATA FORMATTER only — "
        "cognitive instructions (incl. Decision tag) belong in the followup_inferencer's "
        "task_instructions (aggregation), not in this formatter"
    )
    # Original task must NOT be embedded (avoids duplication with wrapper's
    # {{ input }} slot that already carries it under inject_upstream_artifacts).
    assert "Original task" not in formatted_followup, (
        "_format_followup_input must not include the original task — "
        "the wrapper's {{ input }} slot supplies it separately under "
        "inject_upstream_artifacts=True (avoids prompt-level duplication)"
    )

    # Empty-peers branch: when the flow has no peer visibility, the formatter
    # emits a sensible placeholder instead of crashing.
    formatted_no_peers = sample_mfdual.base_inferencer._format_followup_input(
        your_prev="prev draft",
        flow_idx=0,
        step_idx=2,
        visible_plans={},
    )
    assert "No peer outputs visible yet" in formatted_no_peers


# ===========================================================================
# Real-CLI integration test
# ===========================================================================

@pytest.mark.integration
@skip_claude
@skip_openstartup
@pytest.mark.asyncio
@pytest.mark.timeout(60 * 60 * 6)  # 6h cap; covers deep profile worst-case
async def test_real_dual_outside_bta_pti_mfdual_dual(tmp_path, capfd):
    """Real-CLI run of the full composition through `_run_topology()`.

    Matches the canonical `/task` invocation path. Asserts wire-level
    correctness: topology completes, both BTA workspaces are populated
    (or at least the base; fixer is populated iff reviewer triggered fix),
    and the response is non-trivially long.

    Profile via ``BMP_REAL_PROFILE`` env var. See PROFILES dict above.
    """
    p = PROFILES[_PROFILE]
    print(f"\n[bmp-real] profile={_PROFILE!r}: {p}")

    # _run_topology is the exact path /task slash and agent dispatch use
    from agent_foundation.resources.tools.task.executor import _run_topology

    overrides = {
        "_params.default_inferencer": _DEFAULT_INFERENCER,
        "_target_path": str(OPENSTARTUP_PATH),
        "templates_dir": str(TEMPLATES_DIR),
        "consensus_config.max_iterations": p["outer_dual_iter"],
        "base_inferencer.max_breakdown": p["bta_workers"],
        "fixer_inferencer.max_breakdown": p["bta_workers"],
        "base_inferencer.worker_factory.__default__.planner_inferencer.consensus_config.max_iterations": p["mfdual_iter"],
        "base_inferencer.worker_factory.__default__.executor_inferencer.consensus_config.max_iterations": p["inner_dual_iter"],
        "fixer_inferencer.worker_factory.__default__.planner_inferencer.consensus_config.max_iterations": p["mfdual_iter"],
        "fixer_inferencer.worker_factory.__default__.executor_inferencer.consensus_config.max_iterations": p["inner_dual_iter"],
    }

    # Let `_run_topology` allocate its own working_dir (server/_runtime/tasks/
    # task_<id>_<ts>) and inject it as `_params.workspace_root` override. The
    # YAML's `${_params.workspace_root}` interpolation resolves against that
    # path, so context_updates["workspace_path"] aligns with the actual
    # workspace where artifacts land. Children appear under
    # workspace/children/base_inferencer/, workspace/children/fixer_inferencer/.
    result = await _run_topology(
        source=("file", str(YAML_PATH)),
        request=DOC_TASK,
        overrides=overrides,
        session_context={"working_dir": str(tmp_path / "task_ws")},
    )

    # ----- Topology completion -----
    assert result is not None, "_run_topology returned None"
    success = result.context_updates.get("success") is True
    assert success, f"topology reported failure: {result.result!r}"

    response_text = (result.result or "").strip()
    assert len(response_text) >= p["min_response_len"], (
        f"profile={_PROFILE!r}: expected >={p['min_response_len']} chars, "
        f"got {len(response_text)}"
    )

    # ----- Workspace artifacts -----
    workspace_path = result.context_updates.get("workspace_path")
    assert workspace_path, "context_updates missing workspace_path"
    workspace = Path(workspace_path)
    assert workspace.exists(), f"workspace dir missing: {workspace}"

    base_dir = workspace / "children" / "base_inferencer"
    fixer_dir = workspace / "children" / "fixer_inferencer"
    assert base_dir.exists(), (
        f"base BTA workspace missing under {workspace}; "
        f"contents: {list(workspace.iterdir())}"
    )

    # base BTA's children/ must have artifacts (the workers ran)
    base_children = list((base_dir / "children").rglob("*")) \
        if (base_dir / "children").exists() else []
    base_artifact_count = sum(1 for f in base_children if f.is_file())
    assert base_artifact_count > 0, (
        f"base BTA produced no children/ artifacts under {base_dir}"
    )

    # fixer BTA may or may not have run depending on reviewer's verdict.
    # NOTE: workspace propagation (InferencerBase._propagate_workspace_to_children)
    # creates SKELETON subdirs (artifacts/, checkpoints/, children/, etc.) at
    # construction time, so directory existence alone doesn't indicate the
    # fixer fired. Check for FILES specifically — if reviewer approved on first
    # pass, fixer has skeleton dirs but zero files.
    fixer_files = (
        [f for f in fixer_dir.rglob("*") if f.is_file()]
        if fixer_dir.exists() else []
    )
    fixer_fired = bool(fixer_files)
    if fixer_fired:
        fixer_children = list((fixer_dir / "children").rglob("*")) \
            if (fixer_dir / "children").exists() else []
        fixer_artifact_count = sum(1 for f in fixer_children if f.is_file())
        assert fixer_artifact_count > 0, (
            f"fixer BTA fired but produced no children/ artifacts under {fixer_dir}"
        )

    # ----- Observability summary (visible with -s) -----
    print(
        f"\n[bmp-real] outer_dual_iter={p['outer_dual_iter']} "
        f"fixer_fired={fixer_fired} "
        f"base_artifacts={base_artifact_count} "
        f"response_chars={len(response_text)}"
    )


# ===========================================================================
# PAI codebase-understanding task (RovoDevCLI default)
# ===========================================================================

def _build_pai_doc_request() -> str:
    """Construct the codebase-understanding prompt for proactive-ai-platform.

    Mirrors the user-facing request: produce comprehensive structured
    documentation under ``pai_hack/codebase_understanding`` following the
    format/style of ``rai_hack`` and ``convo_ai_hack`` reference folders.
    """
    rai_ref = _HERE.parents[4] / "_dev" / "rai_hack" / "codebase_understanding"
    convo_ref = _HERE.parents[4] / "_dev" / "convo_ai_hack" / "code_understanding"
    return f"""Perform deep codebase understanding on:

  {PAI_CODEBASE_PATH}

Output comprehensive, insightful, structured codebase documentation under:

  {PAI_DOCS_OUTPUT_PATH}

Follow the format and style of these reference documentation sets:
  - {rai_ref}
  - {convo_ref}

Both reference folders organize content into:
  README.md (top-level navigation), index.rst (Sphinx master),
  overviews/  (multi-axis matrix, architectural narrative, criticality dashboard),
  architecture/  (architecture-overview, request-lifecycle, module-catalog,
                  cross-cutting/ for goals/history/feature-flags/observability/...),
  modules/    (per-feature and per-platform-layer deep dives).

Special requirements:
  (1) BUSINESS CHAPTER — pull this year's business goals + metrics, technical
      goals + metrics related to the proactive-ai-platform codebase, and write
      a "business and technical goals" chapter under
      ``architecture/cross-cutting/01-business-and-technical-goals.rst``.
  (2) DEVELOPMENT HISTORY CHAPTER — pull previous merged Pull Requests to the
      main branch and their review comments; build a deep understanding of the
      development history and past business/technical decisions; write the
      result under ``architecture/cross-cutting/02-development-history.rst``.

Process directives:
  - Spawn as many parallel investigation agents as the topology allows.
  - Iterate as needed to converge on accuracy.
  - Make the plan/output deeply informative, careful, accurate.
  - Both human-readable and machine-followable (Sphinx-buildable .rst).
  - Apply critical thinking; verify claims against source code; flag
    ambiguities rather than paper over them.

Tool-use constraints (IMPORTANT — failure to honor these may stall the run):
  - Confine ALL filesystem operations to {PAI_CODEBASE_PATH} and
    {PAI_DOCS_OUTPUT_PATH}. Do NOT run unbounded recursive scans rooted at
    "/", "$HOME", "/Users/", or any other large parent directory.
  - Use the target codebase path explicitly when listing or searching files.
  - If you cannot locate a file with a bounded search, REPORT the ambiguity
    rather than escalating to a system-wide scan.

The deliverable is the documentation itself, NOT a documentation plan or
outline. Produce the actual prose, code excerpts, and reference material that
a reader can consume directly.
"""


@pytest.mark.integration
@skip_openstartup
@pytest.mark.asyncio
@pytest.mark.timeout(60 * 60 * 6)  # 6h cap
async def test_real_pai_codebase_understanding_with_rovodev(tmp_path, capfd):
    """Real-CLI run targeting proactive-ai-platform codebase with RovoDevCLI.

    Drives the same Topology-B-extended composition as the canonical
    integration test, but:
      - swaps the leaf inferencer to ``RovoDevCLI`` via
        ``_params.default_inferencer`` (acli rovodev legacy under the hood)
      - points the request at the proactive-ai-platform repo
      - lets _run_topology allocate its workspace (server/_runtime/tasks/)
        so the runtime folder is the standard place to monitor logs.

    Skip semantics:
      - Requires ``acli`` on PATH; skipped otherwise (acli is the rovodev CLI).
      - Requires the target codebase to exist at ``PAI_CODEBASE_PATH``.
    """
    # Pre-flight: acli + target codebase
    # _cli_available appends `--version` so pass just the binary name to avoid
    # building `acli --version --version`.
    if not _cli_available("acli"):
        pytest.skip("acli (Rovo Dev CLI) not available on PATH")
    if not PAI_CODEBASE_PATH.exists():
        pytest.skip(f"Target PAI codebase missing at {PAI_CODEBASE_PATH}")

    PAI_DOCS_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    p = PROFILES[_PROFILE]
    print(f"\n[pai-real] profile={_PROFILE!r}: {p}")
    print(f"[pai-real] default_inferencer={_DEFAULT_INFERENCER!r}")
    print(f"[pai-real] target codebase: {PAI_CODEBASE_PATH}")
    print(f"[pai-real] docs output: {PAI_DOCS_OUTPUT_PATH}")

    from agent_foundation.resources.tools.task.executor import _run_topology

    overrides = {
        # Critical: swap leaves to RovoDevCLI (or whatever caller chose)
        "_params.default_inferencer": "RovoDevCLI",
        # _target_path cascade points at the codebase to investigate. After
        # the axes refactoring, ALL terminal leaves (including RovoDevCli)
        # inherit target_path from TerminalInferencerBase — the cascade
        # reaches every leaf and sets their subprocess cwd via effective_cwd.
        "_target_path": str(PAI_CODEBASE_PATH),
        # Templates: still load from OpenStartup (the topology lives here).
        "templates_dir": str(TEMPLATES_DIR),
        # Profile knobs (cost-bound the run).
        "consensus_config.max_iterations": p["outer_dual_iter"],
        "base_inferencer.max_breakdown": p["bta_workers"],
        "fixer_inferencer.max_breakdown": p["bta_workers"],
        "base_inferencer.worker_factory.__default__.planner_inferencer.consensus_config.max_iterations": p["mfdual_iter"],
        "base_inferencer.worker_factory.__default__.executor_inferencer.consensus_config.max_iterations": p["inner_dual_iter"],
        "fixer_inferencer.worker_factory.__default__.planner_inferencer.consensus_config.max_iterations": p["mfdual_iter"],
        "fixer_inferencer.worker_factory.__default__.executor_inferencer.consensus_config.max_iterations": p["inner_dual_iter"],
        # NEW v1.7.2: flow_max_dynamic_steps cap (per-flow MFDual followups).
        "_params.flow_max_dynamic_steps": p["flow_dynamic_steps"],
    }

    request = _build_pai_doc_request()

    result = await _run_topology(
        source=("file", str(YAML_PATH)),
        request=request,
        overrides=overrides,
        session_context={"working_dir": str(tmp_path / "task_ws")},
    )

    # ----- Topology completion -----
    assert result is not None, "_run_topology returned None"
    success = result.context_updates.get("success") is True
    assert success, f"topology reported failure: {result.result!r}"

    response_text = (result.result or "").strip()
    assert len(response_text) >= p["min_response_len"], (
        f"profile={_PROFILE!r}: expected >={p['min_response_len']} chars, "
        f"got {len(response_text)}"
    )

    # ----- Workspace artifacts -----
    workspace_path = result.context_updates.get("workspace_path")
    assert workspace_path, "context_updates missing workspace_path"
    workspace = Path(workspace_path)
    assert workspace.exists(), f"workspace dir missing: {workspace}"

    base_dir = workspace / "children" / "base_inferencer"
    assert base_dir.exists(), (
        f"base BTA workspace missing under {workspace}; "
        f"contents: {list(workspace.iterdir())}"
    )

    base_children = list((base_dir / "children").rglob("*")) \
        if (base_dir / "children").exists() else []
    base_artifact_count = sum(1 for f in base_children if f.is_file())
    assert base_artifact_count > 0, (
        f"base BTA produced no children/ artifacts under {base_dir}"
    )

    print(
        f"\n[pai-real] response_chars={len(response_text)} "
        f"base_artifacts={base_artifact_count} "
        f"workspace={workspace}"
    )


# ===========================================================================
# Plan-only mode integration test
# ===========================================================================
#
# This test exercises `_run_topology(mode="plan", ...)`. The executor
# auto-swaps `source` from the full PTI YAML to the standalone planner
# (`breakdown-multiflow-plan.yaml`). We pass the FULL YAML path because:
#   1. It mirrors how the slash command and CLI invoke the executor
#   2. The swap logic is what we want to exercise (not bypass)
#   3. PMS10 preflight verifies the swap fires; this is the integration check
#
# We intentionally do NOT define a PLAN_YAML_PATH constant — the plan-only
# YAML name should never be referenced by name in this test, since the
# executor's swap is the canonical path. Adding a direct reference here
# would create a parallel code path that drifts from the production swap.


def _build_pai_plan_request() -> str:
    """Plan-only request: produce a PLAN for documenting the codebase."""
    return f"""Create a detailed, structured PLAN for documenting the codebase at:

  {PAI_CODEBASE_PATH}

The plan should cover:
  - What major modules and subsystems to document
  - Recommended documentation structure (chapters, sections)
  - Key architectural decisions to investigate
  - Cross-cutting concerns to cover (observability, error handling, etc.)
  - Recommended investigation order and dependencies between sections

Output: A comprehensive, actionable documentation plan that another agent
could follow to produce the actual documentation. This is a PLAN, not the
documentation itself.

Tool-use constraints:
  - Confine ALL filesystem operations to {PAI_CODEBASE_PATH}.
  - Do NOT run unbounded recursive scans rooted at "/", "$HOME", etc.
"""


@pytest.mark.integration
@skip_openstartup
@pytest.mark.asyncio
@pytest.mark.timeout(60 * 60 * 2)  # 2h cap — plan-only is much cheaper
async def test_plan_only_pai_codebase_understanding_with_rovodev(tmp_path, capfd):
    """Plan-only run: produce a structured PLAN for understanding the PAI
    codebase. Does NOT execute the documentation work.

    Exercises:
      - executor.py mode='plan' YAML swap to standalone planner topology
      - Standalone breakdown-multiflow-plan.yaml end-to-end
      - Dual{BTA{MFDual}} planner with lightweight leaf fixer
    """
    if not _cli_available("acli"):
        pytest.skip("acli (Rovo Dev CLI) not available on PATH")
    if not PAI_CODEBASE_PATH.exists():
        pytest.skip(f"Target PAI codebase missing at {PAI_CODEBASE_PATH}")

    p = PROFILES[_PROFILE]
    print(f"\n[pai-plan] profile={_PROFILE!r}: {p}")
    print(f"[pai-plan] default_inferencer={_DEFAULT_INFERENCER!r}")
    print(f"[pai-plan] target codebase: {PAI_CODEBASE_PATH}")

    from agent_foundation.resources.tools.task.executor import _run_topology

    overrides = {
        "_params.default_inferencer": _DEFAULT_INFERENCER,
        "_target_path": str(PAI_CODEBASE_PATH),
        "templates_dir": str(TEMPLATES_DIR),
        # Plan-only knobs — standalone planner is Dual{BTA{MFDual}}, NOT PTI.
        # No fixer.worker_factory.* overrides (fixer is a lightweight leaf).
        "_params.plan_max_breakdown": p["bta_workers"],
        "_params.flow_max_dynamic_steps": p["flow_dynamic_steps"],
        "_params.consensus_max_iterations": p["mfdual_iter"],
    }

    result = await _run_topology(
        source=("file", str(YAML_PATH)),
        request=_build_pai_plan_request(),
        overrides=overrides,
        mode="plan",
        session_context={"working_dir": str(tmp_path / "task_ws")},
    )

    # ----- Topology completion -----
    assert result is not None, "_run_topology returned None"
    success = result.context_updates.get("success") is True
    assert success, f"topology reported failure: {result.result!r}"

    response_text = (result.result or "").strip()
    assert len(response_text) >= p["min_response_len"], (
        f"profile={_PROFILE!r}: expected >={p['min_response_len']} chars, "
        f"got {len(response_text)}"
    )

    # ----- Workspace artifacts -----
    workspace_path = result.context_updates.get("workspace_path")
    assert workspace_path, "context_updates missing workspace_path"
    workspace = Path(workspace_path)
    assert workspace.exists(), f"workspace dir missing: {workspace}"

    # Standalone planner structure: Dual{BTA{MFDual}}
    # base_inferencer = BTA with MFDual workers
    base_dir = workspace / "children" / "base_inferencer"
    assert base_dir.exists(), (
        f"planner BTA workspace missing under {workspace}; "
        f"contents: {list(workspace.iterdir())}"
    )

    base_children = list((base_dir / "children").rglob("*")) \
        if (base_dir / "children").exists() else []
    base_artifact_count = sum(1 for f in base_children if f.is_file())
    assert base_artifact_count > 0, (
        f"planner BTA produced no children/ artifacts under {base_dir}"
    )

    print(
        f"\n[pai-plan] response_chars={len(response_text)} "
        f"base_artifacts={base_artifact_count} "
        f"workspace={workspace}"
    )


# ===========================================================================
# Test 4: Real-CLI subprocess test — invokes `python -m agent_foundation.resources.tools.task`
# ===========================================================================
#
# Why this exists (Hybrid Option C from PTI preflight plan v1.0):
#
# The other test methods (test_real_dual_outside_bta_pti_mfdual_dual,
# test_real_pai_codebase_understanding_with_rovodev,
# test_plan_only_pai_codebase_understanding_with_rovodev) all call
# `_run_topology(...)` directly. That bypasses 6 distinct code paths
# that live in the actual CLI module:
#
#   1. Argument parsing (argparse via tool.json)
#   2. Mode flag mutex (--plan / --execute / --full / --confirm)
#   3. --override key-value parsing
#   4. Topology name → YAML resolution
#   5. Environment variable handling
#   6. Subprocess exit code propagation
#
# This test spawns the actual CLI as a subprocess to validate ALL of those
# paths. It uses --plan mode and a minimal request to keep cost low (~$1-5).
#
# Per Hybrid Option C: this complements (does NOT replace) the in-process
# tests. The in-process tests give us fast structural feedback; this gives
# us end-to-end CLI validation.


def _build_minimal_cli_smoke_request() -> str:
    """Minimal request for CLI smoke test — fast, cheap, deterministic."""
    return (
        "Write a one-page plan for a simple Python script that prints "
        "'Hello, World!' to stdout. Include: 1) the script content, "
        "2) how to run it, 3) expected output. Keep it under 200 lines."
    )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(60 * 60 * 2)  # 2h cap
async def test_real_cli_subprocess_plan_mode(tmp_path, capfd):
    """Real-CLI test: spawn `python -m agent_foundation.resources.tools.task ...`
    as a subprocess to validate the full CLI module code path.

    Unlike the other integration tests in this file (which call
    `_run_topology()` directly), this test exercises:
      - Argument parsing via tool.json
      - Mode flag mutex (--plan flag handling)
      - --override key-value parsing
      - --agent-config topology resolution
      - Real subprocess stdout/stderr capture
      - Exit code propagation

    Uses --plan mode (cheaper, faster) with a minimal request to keep
    cost ~$1-5. Skipped if acli (Rovo Dev CLI) is not available.

    This is the canonical smoke test for the CLI itself.
    """
    if not _cli_available("acli"):
        pytest.skip("acli (Rovo Dev CLI) not available on PATH")

    import sys as _sys

    # Build the PYTHONPATH the way the production launcher script does.
    # _HERE is .../OpenStartup/test/openteam/resources/tools/task/
    # parents[4] = .../OpenStartup
    openstartup_root = _HERE.parents[4]
    coreprojects_root = openstartup_root.parent  # .../CoreProjects
    pythonpath_parts = [
        str(openstartup_root / "src"),
        str(coreprojects_root / "AgentFoundation" / "src"),
        str(coreprojects_root / "RichPythonUtils" / "src"),
        # OpenTeam (in rovoteam dir, sibling to CoreProjects)
        str(coreprojects_root.parent / "rovoteam" / "OpenTeam" / "src"),
    ]
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    pythonpath = ":".join(pythonpath_parts)

    env = {
        **os.environ,
        "PYTHONPATH": pythonpath,
        "OPENTEAM_RUNTIME_DIR": str(openstartup_root / "_runtime"),
    }

    # Use the plan-only YAML (cheaper, validated working on task-a755c721 +
    # task-b3d7ea5a). The full PTI YAML can be tested via a separate method
    # once the PTI preflight plan is complete.
    request = _build_minimal_cli_smoke_request()

    print(f"\n[real-cli-subprocess] cwd: {openstartup_root}")
    print(f"[real-cli-subprocess] PYTHONPATH parts: {len(pythonpath_parts)}")
    print(f"[real-cli-subprocess] request: {request[:80]}...")

    # Optional target_path override: lets the test point RovoDev's CLI subagent
    # sandbox at a specific codebase (e.g., conversational-ai-platform when the
    # request asks the agent to investigate that repo). When omitted, subagents
    # are sandboxed to the workspace directory only (the default behavior).
    # Pass via env var:  BMP_TARGET_PATH=/abs/path/to/codebase  pytest ...
    # Or via env var when using the __main__ launcher (--target-path=...).
    target_path = os.environ.get("BMP_TARGET_PATH", "").strip()

    # Optional request override (lets each run use a different prompt without
    # editing the source). Pass via env var:  BMP_REQUEST_FILE=/abs/path.txt
    request_override_file = os.environ.get("BMP_REQUEST_FILE", "").strip()
    if request_override_file and Path(request_override_file).is_file():
        request = Path(request_override_file).read_text(encoding="utf-8").strip()
        print(f"[real-cli-subprocess] request loaded from BMP_REQUEST_FILE "
              f"({len(request)} chars): {request_override_file}")

    cmd = [
        _sys.executable, "-m", "agent_foundation.resources.tools.task",
        "--plan",
        "--agent-config", "breakdown-multiflow-plan",
        "--override", f"_params.default_inferencer={_DEFAULT_INFERENCER}",
        "--override", "_params.plan_max_breakdown=2",  # minimal breakdown
        "--override", "_params.flow_max_dynamic_steps=1",  # minimal iteration
        "--override", "_params.consensus_max_iterations=1",
    ]
    if target_path:
        # Verify path exists before launching so we fail loudly instead of
        # silently producing speculative output from sandboxed subagents.
        if not Path(target_path).is_dir():
            raise AssertionError(
                f"BMP_TARGET_PATH={target_path!r} is not a directory. "
                f"Cannot proceed — subagents would be sandboxed to the "
                f"workspace and produce shallow speculative output."
            )
        cmd.extend(["--override", f"_target_path={target_path}"])
        print(f"[real-cli-subprocess] _target_path: {target_path}")
    else:
        print("[real-cli-subprocess] _target_path: (not set; subagents sandboxed "
              "to workspace only)")
    cmd.append(request)
    print(f"[real-cli-subprocess] cmd: {' '.join(cmd[:5])} ... (request elided)")

    log_path = tmp_path / "cli_subprocess.log"
    print(f"[real-cli-subprocess] log: {log_path}")

    # Spawn the actual CLI as a subprocess
    result = subprocess.run(
        cmd,
        cwd=str(openstartup_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=60 * 90,  # 90 minutes max
    )

    # Persist logs for debugging on failure
    log_path.write_text(
        f"=== ARGV ===\n{cmd}\n\n"
        f"=== RETURNCODE ===\n{result.returncode}\n\n"
        f"=== STDOUT ===\n{result.stdout}\n\n"
        f"=== STDERR ===\n{result.stderr}\n"
    )

    # ----- Exit code -----
    assert result.returncode == 0, (
        f"CLI subprocess failed (exit={result.returncode}). "
        f"Last stderr lines:\n{result.stderr[-3000:]}"
    )

    # ----- Workspace artifacts -----
    # The CLI allocates workspaces under
    # OpenStartup/src/openteam/server/_runtime/tasks/task_task-<hash>/
    runtime_tasks = (
        openstartup_root / "src" / "openteam" / "server" / "_runtime" / "tasks"
    )
    if runtime_tasks.exists():
        # Find the workspace created during THIS test (by mtime, most recent)
        all_workspaces = sorted(
            runtime_tasks.glob("task_task-*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        assert all_workspaces, (
            f"No task workspace created under {runtime_tasks}. "
            f"Last stdout:\n{result.stdout[-2000:]}"
        )
        latest_workspace = all_workspaces[0]
        print(f"[real-cli-subprocess] workspace: {latest_workspace}")

        # Verify the top-level output.md was produced
        top_output = latest_workspace / "outputs" / "output.md"
        assert top_output.exists(), (
            f"Expected top-level outputs/output.md missing at {top_output}. "
            f"Workspace contents: {list(latest_workspace.rglob('output.md'))[:10]}"
        )
        content = top_output.read_text()
        assert len(content) > 100, (
            f"Top-level output.md too short ({len(content)} chars). "
            f"Content preview: {content[:500]!r}"
        )
        print(f"[real-cli-subprocess] top-level output.md: {len(content)} chars")

    # ----- Regression checks: anomalies that prior runs had fixed -----
    full_log = result.stdout + result.stderr
    assert "NameError" not in full_log, (
        f"NameError detected in CLI output (regression of Bug B fix). "
        f"Excerpt: {full_log[full_log.find('NameError'):full_log.find('NameError') + 500]!r}"
    )
    sharing_warnings = full_log.count("share inferencer")
    assert sharing_warnings == 0, (
        f"Detected {sharing_warnings} 'share inferencer' warnings "
        f"(regression of Fix #11 LazyConfigFactory)"
    )
    print("[real-cli-subprocess] ✅ All regression checks passed")


# ===========================================================================
# Direct invocation: `python <this_file> [--pai] [--plan]`
# ===========================================================================

if __name__ == "__main__":
    """Run an integration test directly without pytest.

    Usage:
        python test_task_real_cli.py [MODE_FLAG] [OPTIONS]

    Mode flags (pick one; omitted = canonical):
        (none)              canonical real integration test
        --pai               PAI codebase-understanding test (RovoDevCLI default)
        --plan              plan-only PAI test (standalone planner topology)
        --cli-subprocess    real-CLI subprocess test (spawns `python -m
                            agent_foundation.resources.tools.task` as subprocess)

    Options (apply to --cli-subprocess):
        --target-path=PATH    Absolute path to the codebase that RovoDev's
                              subagents should sandbox into (e.g.,
                              /Users/tchen7/MyProjects/atlassian_packages/
                              conversational-ai-platform). Without this,
                              subagents are sandboxed to the workspace only
                              and produce shallow speculative output for any
                              request that references external paths.
        --request-file=PATH   Absolute path to a text file containing the
                              request body (replaces the default minimal smoke
                              request). Useful for re-running with a specific
                              prompt without editing the source.
    """
    import asyncio
    import sys
    import tempfile

    _OS = _HERE.parents[4]
    _CP = _OS.parent
    for _dep in (
        _OS / "src",
        _CP / "AgentFoundation" / "src",
        _CP / "RichPythonUtils" / "src",
    ):
        _p = str(_dep)
        if _p not in sys.path:
            sys.path.insert(0, _p)

    use_pai = "--pai" in sys.argv
    use_plan = "--plan" in sys.argv
    use_cli_subprocess = "--cli-subprocess" in sys.argv

    # Per-run config flags for the --cli-subprocess test method.
    # Forwarded via environment variables that the test reads at runtime.
    # Examples:
    #   --target-path=/Users/tchen7/MyProjects/atlassian_packages/conversational-ai-platform
    #   --request-file=/tmp/my_sop_request.txt
    for _arg in sys.argv[1:]:
        if _arg.startswith("--target-path="):
            os.environ["BMP_TARGET_PATH"] = _arg.split("=", 1)[1].strip()
        elif _arg.startswith("--request-file="):
            os.environ["BMP_REQUEST_FILE"] = _arg.split("=", 1)[1].strip()

    async def _main():
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            class _StubCapfd:
                def readouterr(self):
                    return type("R", (), {"out": "", "err": ""})()
            capfd = _StubCapfd()

            if use_cli_subprocess:
                print(f"[direct-run] mode=cli-subprocess default_inferencer={_DEFAULT_INFERENCER!r}")
                await test_real_cli_subprocess_plan_mode(tmp_path, capfd)
            elif use_plan:
                print(f"[direct-run] mode=plan default_inferencer={_DEFAULT_INFERENCER!r}")
                await test_plan_only_pai_codebase_understanding_with_rovodev(tmp_path, capfd)
            elif use_pai:
                print(f"[direct-run] mode=pai default_inferencer={_DEFAULT_INFERENCER!r}")
                await test_real_pai_codebase_understanding_with_rovodev(tmp_path, capfd)
            else:
                print(f"[direct-run] mode=canonical default_inferencer={_DEFAULT_INFERENCER!r}")
                await test_real_dual_outside_bta_pti_mfdual_dual(tmp_path, capfd)

    asyncio.run(_main())
