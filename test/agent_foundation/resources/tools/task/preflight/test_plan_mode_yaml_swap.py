"""Preflight tests for the `--plan` mode YAML swap (executor.py).

Background
----------
The full topology `breakdown-multiflow-plan-then-implement.yaml` wraps PTI in
an outer Dual that reviews the implementation deliverable. When `--plan` is
selected and the implementation is disabled, PTI returns "" (empty string),
causing the outer Dual to review an empty deliverable using
`template_root_space=implementation` review criteria. This wastes iterations
and applies the wrong review semantics.

Fix
---
When `mode == "plan"` AND the source YAML is the full PTI topology, the
executor swaps the source to the standalone `breakdown-multiflow-plan.yaml`
(which has its OWN outer Dual reviewing the PLAN itself with the correct
`_template_root_space=plan` criteria). The full PTI topology imports the
same standalone YAML via `_import_:`, so there's no drift risk.

These tests guard the swap logic in `executor._run_topology`.
"""
from __future__ import annotations

from pathlib import Path
import pytest

# ---------------------------------------------------------------------------
# Path discovery — these MUST exist or the swap can never succeed.
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_TOPOLOGIES_DIR = _HERE.parents[1] / "configs"   # NOT used here; for reference
# The PRODUCTION topologies dir (where the executor's swap looks):
_PROD_TOPOLOGIES = (
    _HERE.parents[5]  # OpenStartup/
    / "src" / "openteam" / "server" / "resources" / "tools" / "task" / "topologies"
)
_FULL_YAML = _PROD_TOPOLOGIES / "breakdown-multiflow-plan-then-implement.yaml"
_STANDALONE_YAML = _PROD_TOPOLOGIES / "breakdown-multiflow-plan.yaml"


# ---------------------------------------------------------------------------
# Static structural assertions — the swap is meaningful only if both YAMLs
# exist and have the right shape.
# ---------------------------------------------------------------------------


@pytest.mark.preflight
def test_PMS1_full_topology_yaml_exists():
    """The full PTI topology must exist in the production topologies dir."""
    assert _FULL_YAML.is_file(), (
        f"Full topology YAML not found at {_FULL_YAML}. "
        "If renamed, update the swap logic in executor.py too."
    )


@pytest.mark.preflight
def test_PMS2_standalone_planner_yaml_exists():
    """The standalone planner YAML must exist as a swap target."""
    assert _STANDALONE_YAML.is_file(), (
        f"Standalone planner YAML not found at {_STANDALONE_YAML}. "
        "Without it, the --plan mode swap falls back to the buggy "
        "enable_implementation=False path."
    )


@pytest.mark.preflight
def test_PMS3_full_topology_imports_standalone():
    """The full topology must `_import_:` the standalone planner — this is
    the single-source-of-truth guarantee that prevents drift between the
    two paths a user can take to plan-only execution."""
    full_text = _FULL_YAML.read_text()
    assert "_import_:" in full_text and "breakdown-multiflow-plan.yaml" in full_text, (
        "Full topology no longer imports breakdown-multiflow-plan.yaml. "
        "The swap and the import path could now diverge — fix one or both."
    )


@pytest.mark.preflight
def test_PMS4_standalone_has_outer_dual_with_plan_template_space():
    """The standalone planner must have an outer Dual with
    `_template_root_space: plan` — this is the WHOLE POINT of the swap.
    If someone removes the outer Dual or changes the template space, the
    --plan mode loses its review-the-plan semantics silently."""
    text = _STANDALONE_YAML.read_text()
    assert "_target_: Dual" in text, (
        "Standalone planner YAML must declare an outer Dual at root. "
        "Otherwise --plan loses its plan-quality review loop."
    )
    assert "_template_root_space: plan" in text, (
        "Standalone planner must set _template_root_space: plan so review "
        "uses plan-quality criteria (not implementation criteria)."
    )


@pytest.mark.preflight
def test_PMS5_standalone_has_no_PTI_wrapper():
    """The standalone planner must NOT contain a PTI wrapper. PTI lives in
    the full topology only — the standalone is for plan-only execution."""
    text = _STANDALONE_YAML.read_text()
    # The standalone uses `_target_: Dual` at root + `_target_: BTA` for
    # base_inferencer. PTI nowhere.
    assert "_target_: PTI" not in text, (
        "Standalone planner unexpectedly contains a PTI wrapper. "
        "If this is intentional, the swap logic in executor.py needs "
        "to be re-evaluated — `--plan` may now run the wrong topology."
    )


# ---------------------------------------------------------------------------
# Behavioral test — exercise the swap logic directly. This requires
# constructing the same inputs that _run_topology sees, but stopping
# BEFORE the actual instantiate (which would need a real LLM).
# ---------------------------------------------------------------------------


@pytest.mark.preflight
def test_PMS6_executor_swap_logic_present_for_plan_mode():
    """Source-level guard: the executor's `_run_topology` MUST contain the
    swap logic (i.e. it doesn't accidentally regress to a single-line
    `enable_implementation = False` toggle without the YAML swap)."""
    executor_path = (
        _HERE.parents[5] / "src" / "openteam" / "server"
        / "resources" / "tools" / "task" / "executor.py"
    )
    src = executor_path.read_text()
    assert 'if mode == "plan":' in src
    # Specific marker strings from the swap implementation.
    # If you rename these comments, update this test too.
    assert "swapping topology" in src, (
        "Expected --plan mode to log 'swapping topology' (the YAML swap). "
        "If the message changed, update the test. If the swap was removed, "
        "this is a regression — the outer Dual will review an empty "
        "deliverable when --plan is used."
    )
    assert "breakdown-multiflow-plan.yaml" in src, (
        "Expected --plan to reference the standalone planner YAML name."
    )


# ---------------------------------------------------------------------------
# Asymmetry test — `--execute` is intentionally NOT swapped (it needs the
# full PTI YAML to run implementation-only).
# ---------------------------------------------------------------------------


@pytest.mark.preflight
def test_PMS8_standalone_has_explicit_lightweight_fixer():
    """The standalone planner MUST declare a `fixer_inferencer:` block —
    without it, DualInferencer defaults fixer to base_inferencer (the same
    BTA{MFDual} object), causing fix iterations to re-run the FULL
    breakdown — ~10-20× more LLM calls than needed for typical incremental
    plan-quality feedback. This test locks in the lightweight-fixer design.
    """
    text = _STANDALONE_YAML.read_text()
    assert "fixer_inferencer:" in text, (
        "Standalone planner is missing explicit fixer_inferencer: block. "
        "Without it, fix iterations re-run the full BTA{MFDual} breakdown "
        "(~10-20× more LLM calls than needed). Re-add the leaf fixer."
    )
    # The fixer must use template_root_space=plan + template_key=followup.
    # We grep loosely (don't anchor to exact whitespace) — a deeper check
    # would need YAML parsing, which is overkill for this guard.
    assert "template_root_space: plan" in text
    assert "template_key: followup" in text, (
        "Fixer must use plan/main/followup.jinja2 — that's the template "
        "designed to receive prior plan + reviewer feedback."
    )
    # The fixer should be a single leaf, NOT a BTA wrapper.
    # Sanity check: no second `_target_: BTA` or similar nested structure
    # appears within ~10 lines after `fixer_inferencer:`.
    lines = text.split("\n")
    fixer_idx = next(
        i for i, ln in enumerate(lines) if ln.startswith("fixer_inferencer:")
    )
    fixer_block = "\n".join(lines[fixer_idx : fixer_idx + 10])
    assert "_target_: BTA" not in fixer_block, (
        "fixer_inferencer block contains BTA — likely heavyweight fixer. "
        "Should be single leaf to keep fix iterations cheap."
    )
    assert "_target_: PTI" not in fixer_block


@pytest.mark.preflight
def test_PMS9_followup_template_exists():
    """The fixer's `template_key: followup` MUST resolve to a real file —
    otherwise rendering crashes at the first fix iteration with
    TemplateNotFound."""
    followup_path = (
        _HERE.parents[5] / "src" / ".."
        / ".." / ".." / ".." / ".."  # arbitrary; we'll resolve via known-good path
    ).resolve()
    # Better: directly check the AgentFoundation prompt_templates
    af_followup = Path(
        "/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/"
        "agent_foundation/resources/prompt_templates/plan/main/followup.jinja2"
    )
    # Be tolerant of repo layout — check both standard locations
    candidates = [
        af_followup,
        _HERE.parents[5] / "src" / "openteam" / "server" / "resources"
        / "prompt_templates" / "plan" / "main" / "followup.jinja2",
    ]
    found = [p for p in candidates if p.is_file()]
    assert found, (
        f"plan/main/followup.jinja2 not found in any known location. "
        f"Standalone planner's fixer_inferencer references this template. "
        f"Searched: {[str(p) for p in candidates]}"
    )


@pytest.mark.preflight
def test_PMS10_run_topology_with_plan_mode_swaps_source(tmp_path, monkeypatch):
    """Behavioral test: invoke `_run_topology` with mode='plan' and assert
    the standalone planner topology is loaded (not the full PTI topology).

    This is the highest-fidelity preflight check — it actually exercises
    the swap logic in executor.py rather than just grepping source for
    expected strings.

    We monkey-patch `load_config` to capture which YAML path it receives
    and short-circuit before instantiation (which would need a real LLM).
    """
    import asyncio
    from agent_foundation.resources.tools.task import executor

    captured = {}

    def _fake_load_config(path, overrides=None):
        captured["path"] = str(path)
        captured["overrides"] = overrides
        # Raise a sentinel exception so _run_topology bails before
        # instantiation. The exception message is meaningless to us;
        # we only care about `captured` being populated.
        raise RuntimeError("__test_short_circuit__")

    monkeypatch.setattr(
        "rich_python_utils.config_utils.load_config",
        _fake_load_config,
    )

    full_yaml = _FULL_YAML
    assert full_yaml.is_file()

    async def _go():
        return await executor._run_topology(
            source=("file", str(full_yaml)),
            request="test request",
            session_context={"working_dir": str(tmp_path)},
            mode="plan",
        )

    result = asyncio.run(_go())
    # _run_topology catches exceptions and returns _error(...).
    # We don't care about result content — only that load_config saw the
    # SWAPPED path (standalone), not the original (full PTI YAML).
    assert "path" in captured, (
        "load_config was never called — _run_topology bailed earlier than "
        "expected. The swap path may not be exercised."
    )
    loaded_path = Path(captured["path"]).name
    assert loaded_path == "breakdown-multiflow-plan.yaml", (
        f"--plan mode did NOT swap source. load_config received "
        f"{loaded_path!r}, expected 'breakdown-multiflow-plan.yaml'. "
        "The swap logic in executor._run_topology has regressed."
    )
    # Sanity: enable_implementation should NOT be in overrides when swap fires
    # (the standalone has no PTI to receive it).
    assert "enable_implementation" not in (captured.get("overrides") or {}), (
        "When swap fires, executor should NOT also pass enable_implementation "
        "override — the standalone YAML has no PTI to receive it."
    )


@pytest.mark.preflight
def test_PMS11_run_topology_with_full_mode_does_NOT_swap(tmp_path, monkeypatch):
    """Inverse of PMS10: with mode='full' (or no mode), the source MUST
    remain the full PTI YAML — we shouldn't swap when --plan isn't set."""
    import asyncio
    from agent_foundation.resources.tools.task import executor

    captured = {}

    def _fake_load_config(path, overrides=None):
        captured["path"] = str(path)
        raise RuntimeError("__test_short_circuit__")

    monkeypatch.setattr(
        "rich_python_utils.config_utils.load_config",
        _fake_load_config,
    )

    async def _go():
        return await executor._run_topology(
            source=("file", str(_FULL_YAML)),
            request="test request",
            session_context={"working_dir": str(tmp_path)},
            mode="full",  # no swap should happen
        )

    asyncio.run(_go())
    loaded_path = Path(captured["path"]).name
    assert loaded_path == "breakdown-multiflow-plan-then-implement.yaml", (
        f"With mode='full', load_config should receive the full PTI YAML "
        f"unchanged. Got: {loaded_path!r}. Swap is firing when it should not."
    )


@pytest.mark.preflight
def test_PMS7_execute_mode_does_NOT_swap_yaml():
    """`--execute` legitimately needs the full PTI YAML — it skips planning
    and runs implementation. Verify the source still reflects this
    asymmetric design."""
    executor_path = (
        _HERE.parents[5] / "src" / "openteam" / "server"
        / "resources" / "tools" / "task" / "executor.py"
    )
    src = executor_path.read_text()
    # Find the --execute block
    assert 'if mode == "execute":' in src
    # It should ONLY toggle enable_planning, NOT swap source
    # (verify by line-locality: within ~10 lines of `if mode == "execute":`,
    # there's no source = ... reassignment)
    lines = src.split("\n")
    exec_idx = next(
        i for i, ln in enumerate(lines) if 'if mode == "execute":' in ln
    )
    block = "\n".join(lines[exec_idx : exec_idx + 10])
    assert "enable_planning" in block, "execute mode should toggle enable_planning"
    assert "source =" not in block, (
        "execute mode should NOT swap source — execute legitimately needs "
        "the full PTI YAML. If you're adding a swap, please document why."
    )
