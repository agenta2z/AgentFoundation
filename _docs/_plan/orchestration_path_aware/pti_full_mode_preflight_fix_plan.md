# Plan: PTI Full Plan-Then-Implement Mode Preflight & Fix Plan

**Status**: v1.0 ACTIVE | **Created**: 2026-05-15 18:09 | **Author**: Rovo Dev (synthesizing 3 parallel agent audits)

**Target topology**: `/Users/tchen7/MyProjects/CoreProjects/OpenStartup/src/openteam/server/resources/tools/task/topologies/breakdown-multiflow-plan-then-implement.yaml`

**Target test**: `/Users/tchen7/MyProjects/CoreProjects/OpenStartup/test/openteam/resources/tools/task/test_task_agent_config_brta_with_multiflow_pti.py`

---

## §0 Executive Summary

**Direct answer to "is it good to test out the full plan-implementation mode?"**

**NOT YET.** Three blocking gaps prevent a confident first end-to-end PTI run:

| # | Gap | Severity | Required before run |
|---|---|---|---|
| **G1** | YAML default inferencer is `ClaudeCodeCLI`, but our last 3 working SOP runs validated `RovoDevCLI`. Switching mid-stream invites silent failure (as happened with `task-2e912508` on May 14). | 🚨 HIGH | YES |
| **G2** | PTI's implementation phase has NEVER been end-to-end tested with any of our recent fixes (LazyConfigFactory, Bug A aggregator fix, unified `_finalize_output`, deep_mode/elegant_mode injection). PTI's own `_finalize_output` was NOT touched by those fixes. | 🚨 HIGH | YES — add 1 preflight test |
| **G3** | The `_template_root_space: implementation` cascade (Refactor 14, line 160) was NEVER tested in production. If the cascade fails, the executor renders the wrong template. | 🟠 MEDIUM | YES — verify in dry-run |

Plus 4 latent issues to address:

| # | Latent Issue | Severity | Decision |
|---|---|---|---|
| **L1** | PTI uses boundary collection (`is_deliverable_boundary=True`) — our recent unified `_finalize_output` work changed BTA but NOT PTI's boundary path. Mismatch could cause double-nesting at PTI level. | MEDIUM | Audit before run |
| **L2** | YAML comment line 84: "`propagate_runtime_input: true` breaks `/task --resume`." Run will be fine, but no resume testing this round. | LOW | Accept; document |
| **L3** | Sphinx-build integration is "implicit in templates, not explicit in config" (per audit). Could fail silently if templates don't actually invoke sphinx. | MEDIUM | Pre-run dry test |
| **L4** | `max_breakdown` injection bug (Bug C from prior plan) still unfixed — would silently waste tokens during executor BTA breakdown too. | LOW | Fix first OR accept | 

---

## §1 What's Different About Plan-Then-Implement Mode

### §1.1 Topology Comparison (Plan-Only vs Plan-Then-Implement)

| Layer | Plan-Only (`breakdown-multiflow-plan.yaml`) | Plan-Then-Implement (`breakdown-multiflow-plan-then-implement.yaml`) |
|---|---|---|
| **Root** | Dual{BTA{MFDual}} | **Outer Dual{PTI{Planner+Executor}}** |
| **Planner** | (root is the plan) | Imports plan-only YAML via `_import_` (line 153–155) |
| **Executor** | (none) | NEW: `BTA{Dual workers}` (line 158–180), max_breakdown=4 |
| **Outer Review** | Skipped (planning is terminal) | NEW: full implementation review (line 184–186) |
| **Fixer** | Light leaf fixer | NEW: full PTI fixer that re-runs plan+execute (line 191–197) |
| **Default inferencer** | RovoDevCLI ✅ (we updated 5/14) | **ClaudeCodeCLI** ❌ (untested with our fixes) |
| **Last successful E2E test** | task-b3d7ea5a (2026-05-14, 58min, 959-line deliverable) | **NEVER successfully tested end-to-end** |

### §1.2 Net New Code Paths Activated

| Code Path | File:Line | Tested Previously? |
|---|---|---|
| `PlanThenImplementInferencer.__init__` orchestration | `plan_then_implement_inferencer.py:252-422` | Unit tests only |
| `PTI._finalize_output()` with `_OUTPUT_MODE_MAP` | `plan_then_implement_inferencer.py:2400-2443` | `test_unified_finalize_output.py:424` (1 test only) |
| `PTI.execute()` two-phase orchestration | `plan_then_implement_inferencer.py:423-503` | NEVER end-to-end |
| Outer Dual reviewing implementation (not plan) template | `templates/implementation/review.jinja2` | NEVER |
| `_inherits_` pattern for fixer PTI | `breakdown-multiflow-plan-then-implement.yaml:191-197` | NEVER |
| `_template_root_space: implementation` cascade | `breakdown-multiflow-plan-then-implement.yaml:160` | NEVER |

### §1.3 What's Reused (Confirmed Working)

| Reused Component | Last Validated In |
|---|---|
| LazyConfigFactory (Fix #11) | task-b3d7ea5a ✅ |
| Bug A fix (aggregator filename) | task-a755c721 ✅ |
| Unified `_finalize_output` for BTA/MFDual | task-b3d7ea5a ✅ |
| deep_mode/elegant_mode injection | task-b3d7ea5a ✅ (32+40 injection sites) |
| Hierarchical workspace layout | task-b3d7ea5a ✅ |

---

## §2 Hard Evidence — Three Audit Findings

### §2.1 Finding 1 (Audit Agent #1 — YAML topology)

> **"Default inferencer differs: `ClaudeCodeCLI` (PTI YAML line 97) vs. `RovoDevCLI` (plan-only line 48). Suggests executor prefers Claude for code implementation."**

**Critical implication**: Our last 3 working runs ALL used RovoDevCLI. The first time we used ClaudeCodeCLI default (task-2e912508 May 14 00:55), the run **crashed in 12 seconds with empty outputs** because the LLM returned empty stdout/stderr. We never debugged ClaudeCodeCLI's empty-return mode.

### §2.2 Finding 2 (Audit Agent #2 — PTI code paths)

> **"PTI does implement `_finalize_output()` (lines 2400–2443) that symlinks child outputs per `output_mode` flags. It uses the deliverable boundary system with `is_deliverable_boundary=True` and `by_role` namespacing."**

**Critical implication**: PTI's finalization is a SEPARATE code path from BTA's that we just refactored. PTI still uses the boundary system. We don't know if the unified-finalize fixes propagated to PTI's path correctly. Need to verify before E2E run.

### §2.3 Finding 3 (Audit Agent #3 — Test coverage gaps)

> **"PTI's implementation phase lacks dedicated unit tests for execution behavior, error handling, and intermediate result synthesis. No tests for plan-phase to implementation-phase handoff."**

**Critical implication**: Even unit tests don't cover the plan→implement handoff. An E2E run is the FIRST test of this critical seam.

---

## §3 The Fix Plan — 3 Fixes + 4 Preflight Tests

### §3.1 Fix #1 — Align YAML Default Inferencer to RovoDevCLI

**File**: `breakdown-multiflow-plan-then-implement.yaml:97`

**Current**:
```yaml
default_inferencer: ClaudeCodeCLI
```

**Change to**:
```yaml
default_inferencer: RovoDevCLI
```

**Rationale**: Match the proven-working config from task-a755c721 and task-b3d7ea5a. We can switch back to ClaudeCodeCLI separately after debugging its empty-return mode.

**Effort**: 1 minute  
**Risk**: NONE — RovoDev is a strict superset (it can also write code via tools)

---

### §3.2 Fix #2 — Verify PTI's `_finalize_output()` Symlink Chain Honors Output Layout

**File**: `plan_then_implement_inferencer.py:2400-2443`

**Investigation**: Read PTI's `_finalize_output` to confirm:
1. It honors the new hierarchical layout (`children/plan/` and `children/implement/` instead of legacy paths)
2. It produces `outputs/output.md` symlink (so outer Dual can review it)
3. It promotes deliverables to `outputs/final_deliverables/` (so outer BTA can aggregate them)

**Expected fix scope**: Likely a 5-10 line update similar to what we did for BTA. If PTI is already correct, no code change needed — just write a verification test (§3.5).

**Effort**: 30 min investigation + 0-15 min fix  
**Risk**: MEDIUM (PTI's code path is untested at this layer)

---

### §3.3 Fix #3 — (DEFER) `max_breakdown` Injection (Bug C)

**File**: `breakdown_then_aggregate_inferencer.py:__attrs_post_init__`

**Decision**: **DEFER to follow-up plan.** Bug C causes 2 wasted LLM subtasks but does not break execution. For first PTI E2E test, accept the waste; fix in a separate plan.

**Rationale**: One change at a time. Fixing Bug C in the same plan as the PTI E2E test would conflate signal.

---

### §3.4 Preflight Test #1 — Dry-Run YAML Instantiation

**Goal**: Catch YAML/Hydra config errors before paying for a live LLM run.

**Command**:
```bash
cd /Users/tchen7/MyProjects/CoreProjects/OpenStartup && \
PYTHONPATH=src:../AgentFoundation/src:../RichPythonUtils/src:../../rovoteam/OpenTeam/src \
/opt/homebrew/anaconda3/bin/python -c "
from python_utils.config_utils import load_config, instantiate
cfg = load_config(
  'openteam.server.resources.tools.task.topologies',
  'breakdown-multiflow-plan-then-implement',
  hyperparam_overrides={'workspace_root': '/tmp/pti_dryrun'},
)
inferencer = instantiate(cfg.inferencer)
print('TYPE:', type(inferencer).__name__)
print('BASE TYPE:', type(inferencer.base_inferencer).__name__)
print('PLANNER TYPE:', type(inferencer.base_inferencer.planner_inferencer).__name__)
print('EXECUTOR TYPE:', type(inferencer.base_inferencer.executor_inferencer).__name__)
"
```

**Expected output**:
```
TYPE: DualInferencer
BASE TYPE: PlanThenImplementInferencer
PLANNER TYPE: DualInferencer  # wraps the plan-only BTA{MFDual}
EXECUTOR TYPE: BreakdownThenAggregateInferencer
```

**Pass criteria**: All 4 types match expected; no exceptions.

**Effort**: 5 min  
**Risk**: NONE — read-only

---

### §3.5 Preflight Test #2 — PTI `_finalize_output()` Symlink Test

**Goal**: Verify PTI's `_finalize_output()` produces the correct symlink chain WITHOUT running LLM inference.

**Add to**: `test/agent_foundation/common/inferencers/test_unified_finalize_output.py`

**Test sketch**:
```python
def test_pti_finalize_output_produces_correct_symlinks(tmp_path):
    """Verify PTI._finalize_output() symlinks plan + implementation to its own root."""
    ws = InferencerWorkspace(root=str(tmp_path / "pti_root"))
    
    # Create fake planner + executor workspaces with outputs
    planner_ws = ws.child("plan")
    executor_ws = ws.child("implement")
    for w, content in [(planner_ws, "Plan content"), (executor_ws, "Implementation content")]:
        os.makedirs(w.outputs_dir, exist_ok=True)
        Path(w.outputs_dir, "output.md").write_text(content)
    
    pti = _make_mock_pti(workspace=ws, output_mode=PTIOutputMode.PLAN_AND_IMPLEMENTATION)
    pti._finalize_output("")
    
    # Verify PTI's own output.md is a symlink (or copy) of implementation
    own_output = Path(ws.outputs_dir, "output.md")
    assert own_output.exists(), "PTI must produce outputs/output.md"
    assert own_output.read_text() == "Implementation content"
    
    # Verify plan content is preserved somewhere (e.g., final_deliverables/plan/)
    plan_in_deliverables = Path(ws.deliverables_dir, "plan", "output.md")
    assert plan_in_deliverables.exists(), "PTI must surface plan as deliverable"
```

**Pass criteria**: Both assertions pass.

**Effort**: 15 min  
**Risk**: Mock setup may need PTI-specific fixtures; could grow to 30 min

---

### §3.6 Preflight Test #3 — Verify `_template_root_space: implementation` Cascade

**Goal**: Confirm Refactor 14 (line 160) actually cascades `template_root_space="implementation"` to the executor's child inferencers.

**Method**: Add to the dry-run script (§3.4):
```python
exec_workers = inferencer.base_inferencer.executor_inferencer.worker_factory
# Invoke factory to instantiate a worker
w = exec_workers()  # if LazyConfigFactory
print("Worker template_root_space:", getattr(w.base_inferencer, "template_root_space", "<unset>"))
# Expected: "implementation"
```

**Pass criteria**: `template_root_space == "implementation"` on at least one worker.

**Effort**: 10 min  
**Risk**: NONE — read-only

---

### §3.7 Preflight Test #4 — Sphinx Toolchain Availability (Latent L3)

**Goal**: Confirm `sphinx-build` is callable in the environment, since YAML comment line 19 says executor "combines sections + sphinx-build HTML."

**Command**:
```bash
which sphinx-build && sphinx-build --version
```

**Pass criteria**: Returns version. If missing, document that HTML build will silently fail — OR install sphinx as preflight remediation.

**Effort**: 1 min  
**Risk**: NONE

---

## §4 Implementation Order

| Phase | Action | Time | Depends On |
|---|---|---|---|
| **0** | Run §3.4 dry-run | 5 min | None |
| **1** | Apply Fix #1 (default_inferencer=RovoDevCLI) | 1 min | §3.4 passes |
| **2** | Run §3.6 template_root_space verify | 10 min | Phase 1 |
| **3** | Run §3.7 sphinx check | 1 min | None |
| **4** | Apply Fix #2 (investigate PTI `_finalize_output`) | 30-45 min | Phase 0 |
| **5** | Write §3.5 PTI symlink unit test | 15-30 min | Phase 4 |
| **6** | Run existing test suite (regression check) | 5 min | Phase 5 |
| **7** | Launch E2E PTI run | 5 min setup | All above ✅ |
| **8** | Monitor + audit completion | 60-90 min runtime | — |

**Total preflight effort**: ~90 min before E2E run  
**Total E2E run cost**: ~$2-5 in LLM tokens (rough estimate; depends on plan complexity)

---

## §5 Acceptance Criteria

### §5.1 Preflight (Before E2E Launch)

| # | Criterion | Verification |
|---|---|---|
| **P1** | YAML instantiates without exception | §3.4 dry-run returns 4 expected types |
| **P2** | YAML default is `RovoDevCLI` | grep `default_inferencer: RovoDevCLI` in YAML |
| **P3** | PTI `_finalize_output()` produces own output.md | §3.5 unit test passes |
| **P4** | PTI `_finalize_output()` promotes plan as deliverable | §3.5 unit test passes |
| **P5** | `template_root_space=implementation` cascades to workers | §3.6 output |
| **P6** | sphinx-build available | §3.7 returns version |
| **P7** | Existing 124+ tests still pass | `pytest` exit 0 |

### §5.2 E2E Run (After Launch)

| # | Criterion | Where To Verify |
|---|---|---|
| **E1** | Workspace structure has `plan/` and `implement/` subtrees | `find workspace -maxdepth 5 -type d` |
| **E2** | Top-level `outputs/output.md` exists and is implementation | `cat */outputs/output.md` |
| **E3** | Top-level `outputs/final_deliverables/` has BOTH plan + impl | `ls -R */outputs/final_deliverables/` |
| **E4** | Aggregator inputs use `(See file: ...)` refs, NOT inlined | `grep "See file:" */aggregator/*InferenceInput*` |
| **E5** | Zero NameErrors, zero sharing warnings | `grep "NameError\|share inferencer" run log` |
| **E6** | Zero cross-worker symlinks | `find -L workspace -type l -not -path "*round_*"` |
| **E7** | Outer Dual review template = `implementation/review` | `grep template_key */review_inferencer/*` |
| **E8** | Run completes within 2h (vs 1h plan-only baseline) | wallclock |
| **E9** | Final deliverable is substantive (≥ 500 lines or has HTML) | `wc -l outputs/final_deliverables/*` |
| **E10** | Sphinx HTML built (if applicable) | `find -name '*.html'` in deliverables |

---

## §6 Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| **R1** | PTI `_finalize_output()` has unfixed boundary-system bug | HIGH | §3.5 unit test catches before E2E |
| **R2** | ClaudeCodeCLI default would crash silently | HIGH | Fix #1 swaps to RovoDevCLI |
| **R3** | `_template_root_space: implementation` cascade fails → wrong template | MEDIUM | §3.6 verification |
| **R4** | Sphinx not installed → silent HTML build failure | MEDIUM | §3.7 preflight |
| **R5** | Fixer PTI's `_inherits_` may share state with base | MEDIUM | LazyConfigFactory should prevent; verify in E1 |
| **R6** | `propagate_runtime_input: true` breaks resume | LOW | Documented; accept; no resume in this run |
| **R7** | Long runtime (>2h) exhausts patience/tokens | MEDIUM | Set max_breakdown low; can stop mid-run |
| **R8** | Outer review fails → fixer PTI re-runs → infinite cost | HIGH | YAML caps `consensus_max_iterations: 3` |

---

## §7 Open Questions

| # | Question | Investigation Needed |
|---|---|---|
| **OQ1** | Does PTI's `_finalize_output()` correctly handle nested `final_deliverables/`? | §3.5 unit test will tell us |
| **OQ2** | Does the fixer PTI (line 191-197) re-run the FULL plan+execute, or only the executor? | Read `_inherits_` semantics in code |
| **OQ3** | Will sphinx-build run from the executor's workspace or from a tool sidecar? | Check template content |
| **OQ4** | Does `deep_mode`/`elegant_mode` cascade through PTI to its planner+executor children? | Add an injection-count audit similar to task-b3d7ea5a |
| **OQ5** | What happens if executor finds 0 implementable sections? | Read PTI `_arun` for empty-input handling |

---

## §8 Out Of Scope (Deferred To Follow-Up Plans)

1. **Bug C — max_breakdown injection** — Real bug; defer per §3.3
2. **ClaudeCodeCLI empty-return debugging** — Need to understand why ClaudeCodeCLI returned empty in task-2e912508; separate plan
3. **PTI checkpoint/resume support with `propagate_runtime_input: true`** — Documented limitation; separate plan
4. **PTI unit test gaps** (per audit #3) — Important but not blocking for E2E; separate plan to add comprehensive unit tests

---

## §9 Provenance

| Version | Date | Author | Change |
|---|---|---|---|
| **v1.0** | 2026-05-15 18:10 | Rovo Dev | Initial plan after 3-agent parallel audit (YAML, PTI code paths, test coverage) |
