"""AF-native real-CLI integration test for the task tool — PLAN and FULL/EXECUTE modes.

Spawns ``python -m agent_foundation.resources.tools.task ... --config pti`` as a subprocess
(the canonical AgentFoundation CLI path: cwd at the AF repo root, artifacts under
``AgentFoundation/_runtime/tasks/``) and asserts the three M7 fixes from the 2026-06-24
session landed in a real run.

Topology (``--config pti`` → ``default.yaml``): the root is **PTI** (plan-then-implement):

    PTI
    ├── planner_inferencer = Dual{ BTA{ MFDual workers }, review, lightweight-leaf fixer }
    │       (imported `_import_: breakdown-multiflow-plan` — the standalone planner)
    └── executor_inferencer = Dual{ BTA{ Dual workers }, review, lightweight-leaf fixer }

The three M7 fixes all live in the **planner** ``Dual{BTA{MFDual}}`` — which is shared by
both modes — so plan mode is a strict subset of full mode:

  1. **LWI flow-deliverable surfacing** — each MFDual flow surfaces its last dynamic step's
     output to ``flow_N/outputs/output.md`` (so the MFI aggregator can reference it).
  2. **Dynamic-step naming** — flow step children are named ``initial`` / ``round{NN}``,
     never ``step_N`` (the ctx-slot name must not leak on disk and break ``_finalize_output``).
  3. **Deferred-logger population** — every top BTA aggregator (constructed without a
     workspace → deferred logger) writes its OWN session ``.jsonl`` despite its workspace
     arriving only via M7 ctx-publish (``_ensure_ctx_workspace_logger``).

  * **plan mode** (``--plan --config pti``): the executor exercises executor.py's swap to the
    standalone ``breakdown-multiflow-plan.yaml``; the run root IS the planner Dual.
  * **full mode** (``--full --config pti``): the full PTI runs; the planner BTA lives one
    level deeper (``children/planner_inferencer/children/propose``) and the executor BTA
    adds a SECOND deferred top-aggregator plus an implementation artifact.

``_assert_three_fixes`` is fully ``rglob``-based (depth-agnostic), so the SAME assertions pass
for both layouts; ``_assert_full_completion`` adds the executor/implementation checks for full
mode only. Both helpers are callable against any completed run dir (see ``--assert-only`` in
``__main__``), so the structural checks can be validated without paying for a fresh LLM run.

The whole topology is forced to ClaudeCodeCLI (``_params.main_inferencer`` drives the planner
leaves; ``_params.default_inferencer`` drives the executor leaves), so the test depends only on
``claude``. Skipped without ``-m integration``, without ``claude`` on PATH, or without
``omegaconf`` importable. Run with an omegaconf-capable venv::

    PYTHONPATH=<AF>/src:<RPU>/src \\
      <venv-with-omegaconf>/bin/python -m pytest \\
      test/agent_foundation/resources/tools/task/test_task_real_cli.py -m integration -s

Profile via ``TASK_E2E_PROFILE`` env (minimal / shallow / medium; default minimal).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Paths (AF-native: _HERE.parents[4] == the AgentFoundation repo root)
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_AF_ROOT = _HERE.parents[4]                      # .../CoreProjects/AgentFoundation
_CP_ROOT = _AF_ROOT.parent                       # .../CoreProjects
_RPU_SRC = _CP_ROOT / "RichPythonUtils" / "src"
_CONFIGS_DIR = _AF_ROOT / "src" / "agent_foundation" / "resources" / "tools" / "task" / "configs"


# ---------------------------------------------------------------------------
# Skip gates
# ---------------------------------------------------------------------------

def _cli_available(command: str) -> bool:
    try:
        return subprocess.run(
            f"{command} --version", shell=True, capture_output=True, text=True, timeout=30
        ).returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _module_available(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


skip_no_claude = pytest.mark.skipif(
    not _cli_available("claude"), reason="claude CLI not on PATH"
)
skip_no_omegaconf = pytest.mark.skipif(
    not _module_available("omegaconf"),
    reason="omegaconf not importable in this interpreter (the config loader needs it)",
)
# Cost gate: the @pytest.mark.integration marker is decorative here (no conftest registers
# it), so it does NOT skip by default. This paid real-CLI e2e (plan ~15-30 min, full ~60 min,
# real claude spend) must therefore be opted into EXPLICITLY — even when claude + omegaconf are
# both present — so a broad `pytest` run never spawns it by accident.
skip_unless_e2e = pytest.mark.skipif(
    not os.environ.get("RUN_TASK_E2E"),
    reason="paid real-CLI e2e — set RUN_TASK_E2E=1 to run (plan ~15-30min, full ~60min)",
)


# ---------------------------------------------------------------------------
# Cost profiles (knobs map to default.yaml / breakdown-multiflow-plan.yaml _params)
# ---------------------------------------------------------------------------

PROFILES = {
    # minimal: tightest run that still exercises every fix — 2 planner subtasks,
    # 2 flows/worker, 2 steps/flow (→ initial+round01, exercises round{NN} naming),
    # 1 consensus iter. Plan ~15-30 min; full materially longer (adds executor fan-out).
    "minimal": {
        "plan_max_breakdown": 2,
        "exec_max_breakdown": 2,
        "flow_max_dynamic_steps": 2,
        "consensus_max_iterations": 1,
        "min_top_output_bytes": 100,
    },
    "shallow": {
        "plan_max_breakdown": 2,
        "exec_max_breakdown": 3,
        "flow_max_dynamic_steps": 2,
        "consensus_max_iterations": 1,
        "min_top_output_bytes": 200,
    },
    "medium": {
        "plan_max_breakdown": 3,
        "exec_max_breakdown": 4,
        "flow_max_dynamic_steps": 3,
        "consensus_max_iterations": 2,
        "min_top_output_bytes": 400,
    },
}
_PROFILE = os.environ.get("TASK_E2E_PROFILE", "minimal")


# A single self-contained request works for BOTH modes: plan mode produces the plan,
# full mode plans then implements it. Framed so the implementation is a small, bounded
# deliverable that lands in the run's _target_path sandbox (NOT the AF repo).
_TASK_REQUEST = (
    "Create a small, self-contained Python CLI utility `prune.py` that deletes files "
    "matching a glob, with a --dry-run flag that prints what WOULD be deleted without "
    "deleting, a --help, and an env-var fallback PRUNE_DRY_RUN. Include a one-line "
    "README.md explaining how to run it. Keep it minimal and confined to the target "
    "directory; do not touch anything outside it."
)


# ---------------------------------------------------------------------------
# Reusable structural assertions (callable against ANY completed run dir).
# rglob-from-root, depth-agnostic: the SAME logic passes for plan-only
# (Dual{BTA{MFDual}}, run-root IS the Dual) and full PTI (planner BTA two
# levels deeper at children/planner_inferencer/children/propose).
# ---------------------------------------------------------------------------

def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _is_flow_dir(d: Path) -> bool:
    """An LWI flow dir: named flow_* with a children/ holding initial / round{NN}."""
    ch = d / "children"
    if not (d.is_dir() and ch.is_dir()):
        return False
    return any(
        c.is_dir() and (c.name == "initial" or c.name.startswith("round"))
        for c in ch.iterdir()
    )


def _aggregators_that_ran(workspace: Path) -> list:
    """Every `aggregator` node that actually produced output (non-empty outputs/ —
    excludes construction-time skeleton dirs)."""
    out = []
    for a in workspace.rglob("aggregator"):
        if a.is_dir() and (a / "outputs").is_dir() and any(
            p.is_file() for p in (a / "outputs").rglob("*")
        ):
            out.append(a)
    return out


def _deliverable_candidates(workspace: Path, mode: str) -> list:
    """Paths where the final deliverable may land. Plan mode surfaces it to the run root;
    full (PTI) mode surfaces the implementation up the executor Dual to
    ``children/executor_inferencer/outputs/`` — and the PTI ``_finalize_outputs`` root copy
    keys off the SHORT role name ``executor`` while the on-disk dir is the attr name
    ``executor_inferencer``, so the root copy may not fire (observed empty on a real run).
    Accept the executor-outputs location as the implementation deliverable."""
    cand = [
        workspace / "outputs" / "output.md",
        workspace / "outputs" / "final_deliverables" / "output.md",
    ]
    if mode == "full":
        ex = workspace / "children" / "executor_inferencer" / "outputs"
        cand += [ex / "output.md", ex / "final_deliverables" / "output.md"]
    return cand


def _assert_three_fixes(workspace: Path) -> dict:
    """Assert the three M7 fixes in a completed run dir (the run-root, i.e.
    ``.../_runtime/tasks/<name>/<name>_<ts>_<hash>``). Structural-only (framework-created
    layout, deterministic given the topology + the fixes), never LLM content."""
    # --- discover LWI flow dirs anywhere in the tree (plan OR full depth) ---
    flow_dirs = [d for d in workspace.rglob("flow_*") if _is_flow_dir(d)]
    assert flow_dirs, f"no LWI flow dirs found under {workspace} (topology did not run flows)"

    # --- Fix 2: dynamic-step naming (initial / round{NN}, never step_N) ---
    naming_ok = 0
    for fd in flow_dirs:
        kids = {p.name for p in (fd / "children").iterdir() if p.is_dir()}
        leaked = sorted(c for c in kids if c.startswith("step_"))
        assert not leaked, (
            f"flow {fd.name} has ctx-slot-named children {leaked} on disk — dynamic-step "
            f"naming regression (must be initial/round{{NN}})"
        )
        if "initial" in kids or any(c.startswith("round") for c in kids):
            naming_ok += 1
    assert naming_ok > 0, f"no flow used initial/round{{NN}} naming; flows={[f.name for f in flow_dirs]}"

    # --- Fix 1: flow-deliverable surfacing (>=1 flow surfaced outputs/output.md) ---
    surfaced = sum(
        1 for fd in flow_dirs
        if (fd / "outputs" / "output.md").is_file()
        and (fd / "outputs" / "output.md").stat().st_size > 0
    )
    assert surfaced > 0, (
        f"no flow surfaced its last step's output to outputs/output.md (of {len(flow_dirs)} "
        f"flows) — LWI _finalize_output surfacing regression"
    )

    # --- Fix 1b (cascade): an aggregator input references the flow deliverable via
    #     (See file:)/(See outputs folder:) rather than embedding raw <Response> ---
    agg_inputs = []
    for a in workspace.rglob("aggregator"):
        if a.is_dir():
            agg_inputs += list(a.glob("logs/session/*.jsonl.parts/InferenceInput/*.txt"))
    marker_hits = sum(
        1 for f in agg_inputs
        if any(m in _read_text(f) for m in ("(See file:", "(See outputs folder:", "(See deliverables:"))
    )
    if agg_inputs:
        assert marker_hits > 0, (
            f"none of the {len(agg_inputs)} aggregator inputs referenced a flow deliverable "
            f"via (See file:)/(See outputs folder:) — surfacing→reference cascade regression"
        )

    # --- Fix 3: deferred-logger — every aggregator that RAN has a session .jsonl ---
    ran_aggs = _aggregators_that_ran(workspace)
    assert ran_aggs, f"no aggregator produced output under {workspace} (topology did not aggregate)"
    missing = [
        a for a in ran_aggs
        if not ((a / "logs" / "session").is_dir() and any((a / "logs" / "session").glob("*.jsonl")))
    ]
    assert not missing, (
        f"{len(missing)}/{len(ran_aggs)} aggregator(s) ran but wrote NO session .jsonl "
        f"(deferred-logger regression — logger never un-deferred under M7 ctx-publish): "
        f"{[str(a.relative_to(workspace)) for a in missing[:3]]}"
    )

    review_logs = list(workspace.rglob("review/logs/session/*.jsonl"))  # soft / reported
    return {
        "flows": len(flow_dirs),
        "naming_ok": naming_ok,
        "surfaced": surfaced,
        "agg_inputs": len(agg_inputs),
        "agg_reference_markers": marker_hits,
        "aggregators_ran": len(ran_aggs),
        "aggregators_with_session_log": len(ran_aggs) - len(missing),
        "review_session_logs": len(review_logs),
    }


def _assert_full_completion(workspace: Path, profile: dict) -> dict:
    """Full/execute-mode extras (on top of _assert_three_fixes): the executor stage ran
    and produced an implementation artifact. Gates on FILE CONTENT, never directory
    existence (PTI/Dual create full skeleton subdirs at construction time)."""
    # (1) Implementation artifact (root, or the executor-Dual outputs/ where the PTI
    #     full run actually surfaces it — see _deliverable_candidates).
    candidates = _deliverable_candidates(workspace, "full")
    sized = [c for c in candidates if c.is_file() and c.stat().st_size > profile["min_top_output_bytes"]]
    assert sized, (
        f"no non-empty implementation artifact at "
        f"{[str(c.relative_to(workspace)) for c in candidates]} "
        f"(> {profile['min_top_output_bytes']} bytes) — executor did not surface a deliverable"
    )
    root_surfaced = (workspace / "outputs" / "output.md").is_file() and (
        workspace / "outputs" / "output.md"
    ).stat().st_size > 0

    # (2) Executor stage actually ran (file count, not dir existence).
    exec_dir = workspace / "children" / "executor_inferencer"
    assert exec_dir.is_dir(), f"executor stage dir missing: {exec_dir} (not a full PTI run?)"
    exec_files = [f for f in exec_dir.rglob("*") if f.is_file()]
    assert len(exec_files) > 1, (
        f"executor stage produced only {len(exec_files)} file(s) — looks like a construction-time "
        f"skeleton, executor never fired"
    )
    exec_propose = exec_dir / "children" / "propose" / "children"
    exec_propose_has_files = exec_propose.is_dir() and any(f.is_file() for f in exec_propose.rglob("*"))
    assert exec_propose_has_files, f"executor BTA workers produced no artifacts under {exec_propose}"

    # (3) The executor BTA aggregator (2nd deferred top-aggregator) has a session log.
    exec_agg = exec_dir / "children" / "propose" / "children" / "aggregator"
    exec_agg_logs = 0
    if (exec_agg / "outputs").is_dir() and any(p.is_file() for p in (exec_agg / "outputs").rglob("*")):
        sess = exec_agg / "logs" / "session"
        exec_agg_logs = len(list(sess.glob("*.jsonl"))) if sess.is_dir() else 0
        assert exec_agg_logs > 0, (
            f"executor BTA aggregator ran but wrote no session .jsonl: {exec_agg} (deferred-logger regression)"
        )

    return {
        "impl_artifact": str(sized[0].relative_to(workspace)),
        "impl_artifact_bytes": max(c.stat().st_size for c in sized),
        "root_outputs_surfaced": root_surfaced,  # PTI _finalize_outputs short-name quirk: may be False
        "executor_files": len(exec_files),
        "executor_propose_has_files": exec_propose_has_files,
        "executor_aggregator_session_logs": exec_agg_logs,
    }


# ---------------------------------------------------------------------------
# Per-mode CLI command construction
# ---------------------------------------------------------------------------

def _build_cmd(mode: str, profile: dict, tool_name: str, target_path: Path) -> list:
    """Construct the AF task-CLI argv for ``mode``. ``--config pti`` (→ default.yaml) is used
    for BOTH modes so plan is a literal subset of full: ``--plan`` swaps to the standalone
    planner, ``--full`` runs the whole PTI. ``--tool-name`` + the request are appended last."""
    mode_flag = "--plan" if mode == "plan" else "--full"
    cmd = [
        sys.executable, "-m", "agent_foundation.resources.tools.task",
        mode_flag,
        "--config", "pti",
        "--override", "_params.main_inferencer=ClaudeCodeCLI",
        "--override", "_params.default_inferencer=ClaudeCodeCLI",
        "--override", f"_params.plan_max_breakdown={profile['plan_max_breakdown']}",
        "--override", f"_params.flow_max_dynamic_steps={profile['flow_max_dynamic_steps']}",
        "--override", f"_params.consensus_max_iterations={profile['consensus_max_iterations']}",
    ]
    if mode == "full":
        # exec_max_breakdown is VALID only on the full PTI (referenced by default.yaml); it is
        # inert/stripped on the plan-swapped standalone planner, so it is omitted in plan mode.
        # _target_path sandboxes the implementation file-writes away from the AF repo.
        cmd += [
            "--override", f"_params.exec_max_breakdown={profile['exec_max_breakdown']}",
            "--override", f"_target_path={target_path}",
        ]
    cmd += ["--tool-name", tool_name, _TASK_REQUEST]
    return cmd


# ---------------------------------------------------------------------------
# No-LLM smoke test — validates both config shapes (alias / _import_ / _params / fixer)
# ---------------------------------------------------------------------------

@skip_no_omegaconf
def test_configs_instantiate_smoke(tmp_path):
    """Instantiate both topologies (no LLM, ~5s) and assert their shape — catches
    alias / ``_import_`` / ``_params`` interpolation / lightweight-fixer regressions
    before any paid run. Exercises the SAME load_config + override-merge + resolve path
    the subprocess test drives, at zero cost."""
    import agent_foundation.common.configs.registered_targets  # noqa: F401
    from rich_python_utils.config_utils import instantiate, load_config

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

    base_overrides = {
        "_params.workspace_root": str(tmp_path / "ws"),
        "_params.main_inferencer": "ClaudeCodeCLI",
        "_params.default_inferencer": "ClaudeCodeCLI",
    }

    def _worker_from(bta):
        """Instantiate ONE BTA worker, tolerating both worker_inferencers shapes: a single
        LazyConfigFactory / partial / callable (the AF single-_target_ config, auto-wrapped by
        the RPU config walker) or a dict[task_type → factory]."""
        wi = bta.worker_inferencers
        factory = (
            (wi.get("__default__") or wi.get("_default") or next(iter(wi.values())))
            if isinstance(wi, dict) else wi
        )
        return factory()  # LazyConfigFactory / partial / callable → no-arg call

    # ----- (A) standalone planner: Dual{ BTA(plan_bta){ MFDual }, leaf fixer } -----
    plan_cfg = load_config(str(_CONFIGS_DIR / "breakdown-multiflow-plan.yaml"), overrides=dict(base_overrides))
    planner = instantiate(plan_cfg)
    assert isinstance(planner, DualInferencer), f"standalone planner root must be Dual; got {type(planner).__name__}"
    assert isinstance(planner.base_inferencer, BreakdownThenAggregateInferencer)
    assert planner.base_inferencer.name == "plan_bta"
    mfdual = _worker_from(planner.base_inferencer)
    assert isinstance(mfdual, MultiFlowDualInferencer), f"planner BTA worker must be MFDual; got {type(mfdual).__name__}"
    assert mfdual.propagate_runtime_input is True
    assert len(mfdual.flow_configs) == 2, f"expected 2 flows; got {len(mfdual.flow_configs)}"
    # lightweight-fixer design: the Dual fixer is a LEAF, not another BTA (guards the
    # costly-re-implementation regression).
    assert not isinstance(planner.fixer_inferencer, BreakdownThenAggregateInferencer), (
        "standalone planner fixer must be a lightweight leaf, not a BTA"
    )

    # ----- (B) full PTI: PTI{ planner=Dual{BTA{MFDual}}, executor=Dual{BTA(exec_bta){Dual}} } -----
    full_cfg = load_config(str(_CONFIGS_DIR / "default.yaml"), overrides=dict(base_overrides))
    pti = instantiate(full_cfg)
    assert isinstance(pti, PlanThenImplementInferencer), f"full root must be PTI; got {type(pti).__name__}"
    assert isinstance(pti.planner_inferencer, DualInferencer), "PTI.planner must be the imported Dual{BTA{MFDual}}"
    plan_bta = pti.planner_inferencer.base_inferencer
    assert isinstance(plan_bta, BreakdownThenAggregateInferencer) and plan_bta.name == "plan_bta"
    assert isinstance(_worker_from(plan_bta), MultiFlowDualInferencer)
    exec_dual = pti.executor_inferencer
    assert isinstance(exec_dual, DualInferencer), f"PTI.executor must be Dual; got {type(exec_dual).__name__}"
    exec_bta = exec_dual.base_inferencer
    assert isinstance(exec_bta, BreakdownThenAggregateInferencer) and exec_bta.name == "exec_bta"
    exec_worker = _worker_from(exec_bta)
    assert isinstance(exec_worker, DualInferencer), (
        f"exec BTA worker must be a Dual (write+review+fix), NOT MFDual; got {type(exec_worker).__name__}"
    )
    assert not isinstance(exec_dual.fixer_inferencer, BreakdownThenAggregateInferencer), (
        "executor Dual fixer must be a lightweight leaf, not a BTA"
    )


# ---------------------------------------------------------------------------
# Real-CLI subprocess integration test — parametrized over plan + full modes
# ---------------------------------------------------------------------------

@pytest.mark.integration
@skip_unless_e2e
@skip_no_claude
@skip_no_omegaconf
@pytest.mark.timeout(60 * 60 * 3)  # 3h cap (full mode adds the entire executor fan-out)
@pytest.mark.parametrize("mode", ["full", "plan"])
def test_task_cli_subprocess(tmp_path, mode):
    """Spawn the AF task CLI in ``mode`` over the PTI topology and assert the three M7
    fixes (both modes) + executor/implementation completion (full only). Exercises the
    real CLI module path: argparse (tool.json), --config alias resolution, the --plan
    topology swap, --override parsing, exit-code propagation."""
    profile = PROFILES[_PROFILE]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(_AF_ROOT / "src"), str(_RPU_SRC), os.environ.get("PYTHONPATH", "")]
        ),
    }
    env.pop("DEFAULT_MAIN_INFERENCER", None)  # claude-only is pinned via overrides; belt-and-suspenders

    target_path = tmp_path / "impl_target"
    target_path.mkdir()
    tool_name = f"pytest_task_{mode}_{int(time.time())}"
    cmd = _build_cmd(mode, profile, tool_name, target_path)

    print(f"\n[task-e2e:{mode}] profile={_PROFILE} cwd={_AF_ROOT}")
    print(f"[task-e2e:{mode}] cmd={' '.join(cmd[:10])} ... (request elided)")

    subprocess_timeout = 60 * 55 if mode == "plan" else 60 * 115
    result = subprocess.run(
        cmd, cwd=str(_AF_ROOT), env=env, capture_output=True, text=True, timeout=subprocess_timeout
    )

    log_path = tmp_path / f"cli_{mode}.log"
    log_path.write_text(
        f"=== ARGV ===\n{cmd}\n\n=== RETURNCODE ===\n{result.returncode}\n\n"
        f"=== STDOUT ===\n{result.stdout}\n\n=== STDERR ===\n{result.stderr}\n"
    )
    print(f"[task-e2e:{mode}] cli log: {log_path}")

    assert result.returncode == 0, (
        f"CLI subprocess ({mode}) failed (exit={result.returncode}). stderr tail:\n{result.stderr[-3000:]}"
    )

    # Regression markers from prior fixes (cheap, content-based).
    full_log = result.stdout + result.stderr
    assert "NameError" not in full_log, (
        f"NameError in CLI output ({mode}); excerpt: {full_log[full_log.find('NameError'):][:400]!r}"
    )
    sharing = full_log.count("share inferencer") + full_log.count("shared inferencer")
    assert sharing == 0, f"{sharing} inferencer-sharing warning(s) in CLI output ({mode})"

    # Locate THIS run's workspace (unique tool_name → one run dir).
    run_root = _AF_ROOT / "_runtime" / "tasks" / tool_name
    runs = sorted(run_root.glob(f"{tool_name}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    assert runs, f"no run workspace under {run_root}; stdout tail:\n{result.stdout[-2000:]}"
    workspace = runs[0]
    print(f"[task-e2e:{mode}] workspace={workspace}")

    deliverables = [
        c for c in _deliverable_candidates(workspace, mode)
        if c.is_file() and c.stat().st_size > profile["min_top_output_bytes"]
    ]
    assert deliverables, (
        f"no deliverable (> {profile['min_top_output_bytes']} bytes) for {mode} at "
        f"{[str(c.relative_to(workspace)) for c in _deliverable_candidates(workspace, mode)]}"
    )
    if mode == "full" and not (workspace / "outputs" / "output.md").is_file():
        print(
            f"[task-e2e:full] NOTE: PTI root outputs/output.md not surfaced; deliverable at "
            f"{deliverables[0].relative_to(workspace)} (executor-outputs fallback — "
            f"_finalize_outputs short-name 'executor' vs attr-name 'executor_inferencer')"
        )

    summary = _assert_three_fixes(workspace)
    print(f"[task-e2e:{mode}] ✅ three-fix assertions: {summary}")
    if mode == "full":
        full_summary = _assert_full_completion(workspace, profile)
        print(f"[task-e2e:{mode}] ✅ full-completion assertions: {full_summary}")


# ---------------------------------------------------------------------------
# Direct invocation (no pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Usage:
    #   python test_task_real_cli.py --assert-only <run_dir> [--full]   # re-check an existing run
    #   python test_task_real_cli.py [--mode full|plan]                 # run the subprocess test
    _argv = sys.argv[1:]
    if _argv and _argv[0] == "--assert-only":
        ws = Path(_argv[1]).resolve()
        print(f"[assert-only] workspace={ws}")
        print(f"[assert-only] three_fixes: {_assert_three_fixes(ws)}")
        if "--full" in _argv:
            print(f"[assert-only] full_completion: {_assert_full_completion(ws, PROFILES[_PROFILE])}")
    else:
        import tempfile

        _mode = "plan"
        if "--mode" in _argv:
            _mode = _argv[_argv.index("--mode") + 1]
        for _dep in (_AF_ROOT / "src", _RPU_SRC):
            if str(_dep) not in sys.path:
                sys.path.insert(0, str(_dep))
        with tempfile.TemporaryDirectory() as _td:
            print(f"[direct-run] mode={_mode}")
            test_task_cli_subprocess(Path(_td), _mode)
