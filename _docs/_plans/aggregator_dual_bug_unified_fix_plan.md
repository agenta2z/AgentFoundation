# Unified Fix Plan — Aggregator Inlined Inputs + Repeated Aggregator Inference (v1.0)

**Status**: ACTIVE v1.0
**Authored**: 2026-05-12 22:17 (after stopping run `task-e3ae2732`)
**Run analyzed**: `/Users/tchen7/MyProjects/CoreProjects/OpenStartup/src/openteam/server/_runtime/tasks/task_task-e3ae2732_20260512_182441/`
**Prior plan superseded**: None — adds onto v3.2 unified `_finalize_output` plan

---

## §0 Executive Summary

Two correlated symptoms observed in `task-e3ae2732`:

1. **Symptom A** — Aggregator's `InferenceInput` shows 178-line bloated text with **0 file-path references** instead of expected ~50-line summary with `(See file: <path>)` references.
2. **Symptom B** — The same aggregator session received **7 InferenceInputs over 3 hours**, each with bloated text. Total runtime ballooned to 4+ hours.

**Root cause hierarchy**:

```
ROOT CAUSE (Symptom A):
└── output_path does NOT cascade to round01 inferencer at construction time
    ├── → resolve_output_path() returns None
    ├── → _finalize_output() no-op (early return at inferencer_base.py:822)
    ├── → outputs/output.md NOT written
    ├── → LWI's symlink at flow_X/outputs/output.md is DANGLING
    ├── → resolve_canonical_output_path() follows dangling → returns None
    ├── → _captured_paths = [None, None]
    └── → _format_worker_results_text falls to inlining <Response> text

ROOT CAUSE (Symptom B):
└── MFI's outer Dual review-fix loop runs aggregator MULTIPLE times
    ├── consensus_max_iterations=3 in YAML
    ├── Each Dual round: aggregator runs once → review → fix → repeat
    ├── Plus MFI's max_dynamic_steps=3 per flow
    └── Total: 7 aggregator invocations (3 Dual rounds × 2-3 MFI dynamic steps + initial)

AMPLIFICATION (Symptom A × Symptom B):
└── Each of the 7 invocations gets bloated 14KB input (instead of 1KB file refs)
    └── 7 × 14KB = 98KB context wasted per worker
        └── × 2 workers = 196KB total bloat
            └── → Slower LLM responses
                └── → 4+ hour total runtime instead of expected ~1 hour
```

---

## §1 Hard Evidence From Run `task-e3ae2732`

### Symptom A Evidence

| File | Size | Findings |
|---|---|---|
| `worker_0/.../aggregator/.../InferenceInput/20260512_184921_*.txt` | 14314 bytes / 178 lines | 0 `(See file:` refs, 2 `### Result N` sections inlined |
| `worker_0/.../flow_0/outputs/output.md` (symlink) | 56 bytes | mtime: 18:48 (created EARLY) |
| `worker_0/.../flow_0/children/round01/outputs/output.md` (target) | 22496 bytes | **Birth time: 21:46:26** (3 hours LATER!) |
| `worker_0/.../flow_0/children/round01/_runtime/raw/InferenceResponse/*.txt` | 68KB | mtime: 18:48:25 (raw response written immediately) |

**Conclusion**: The InferenceResponse was written at 18:48 but the `outputs/output.md` was NOT created until 21:46 — proving `_finalize_output` was a no-op for 3 hours.

### Symptom B Evidence

| InferenceInput | Time | Size | Content |
|---|---|---|---|
| `20260512_184921_*.txt` | 18:49 | 178 lines | "Flow 0: 742 lines, Flow 1: 181 lines" |
| `20260512_191453_*.txt` | 19:14 | 201 lines | "Flow 0: 792 lines, Flow 1: 920 lines" |
| `20260512_194513_*.txt` | 19:45 | 164 lines | (consolidated growing) |
| `20260512_201435_*.txt` | 20:14 | 190 lines | (continues growing) |
| `20260512_204200_*.txt` | 20:42 | 176 lines | (continues) |
| `20260512_211112_*.txt` | 21:11 | 181 lines | (continues) |
| `20260512_214953_*.txt` | 21:49 | 173 lines | "Flow 0: 1,051 lines, Flow 1: 1,055 lines" |

**Conclusion**: 7 invocations over 3 hours, ~25-30 min apart. Each subsequent input is COMPLETELY REGENERATED and inlines the new state of worker outputs.

### Verified Working — Hierarchical Layout + Symlinks

| Aspect | Status |
|---|---|
| Hierarchical layout (`flow_X/children/initial/`, `round01/`) | ✅ |
| LWI symlinks created | ✅ (created at construction) |
| Dual `propose/` semantic naming | ✅ |
| Cross-worker symlinks | ✅ NONE (Anomaly 6 eliminated) |
| Sharing-warnings | ✅ 0 (Fix #11 effective) |

---

## §2 Root Cause Analysis — Symptom A (Bloated Inputs)

### §2.1 The Cascade Failure

**File**: `inferencer_base.py:803-831` (`_finalize_output`)

```python
def _finalize_output(self, response):
    resolved = self.resolve_output_path()    # ← Returns None if output_path is None/relative
    if not resolved or not os.path.isabs(resolved):
        return response                       # ← Early return, NO write
    if self.has_local_access and os.path.isfile(resolved) and ...:
        return response
    # Framework writes
    ...
```

### §2.2 Why round01 Has No `output_path`

**File**: `linear_workflow_inferencer.py:220-244` (`_propagate_workspace_to_children`)

The LWI override propagates `_workspace` to followup children but **does NOT propagate `_output_path`**. So when round01 inferencer is constructed:
- `_workspace` is set to `flow_X/children/round01/` ✅
- `_output_path` is None ❌

Then `resolve_output_path()` calls:
```python
def resolve_output_path(self):
    if self._output_path is None:
        return None
    if os.path.isabs(self._output_path):
        return self._output_path
    if self._workspace is None:
        return None
    return os.path.join(self._workspace.outputs_dir, self._output_path)
```

→ Returns None → `_finalize_output` no-op.

### §2.3 Why It Eventually Works At 21:46

By 21:46, ONE of the following finally happened:
- A late iteration's `LazyConfigFactory` cascade injected `output_path=output.md` (cascade fix at `_lazy_config_factory.py:67`)
- A `switch_role()` call propagated `output_path` along with workspace
- A `_propagate_workspace_to_children` re-run was triggered with newly-set `output_path`

The exact trigger is undetermined but the empirical result is clear: round01 EVENTUALLY got `output_path` set, but only after 3 hours of bloated aggregator runs.

### §2.4 The Symlink Dangling Problem

**File**: `linear_workflow_inferencer.py:285-310` + `_symlink_child_output`

LWI creates the symlink `flow_X/outputs/output.md → flow_X/children/round01/outputs/output.md` when `_finalize_output` runs on the LWI itself. But the LWI's `_finalize_output` runs BEFORE its child's `_finalize_output` actually writes the target file (since the child's writes failed silently due to no `output_path`).

**Result**: Symlink exists at 18:48, target doesn't exist until 21:46.

---

## §3 Root Cause Analysis — Symptom B (7 Aggregator Invocations)

### §3.1 The MFI Dynamic Step Loop

**File**: `multi_flow_inferencer.py:1225-1248` (MFI._ainfer override)

MFI inherits `_build_subgraph_spec` from BTA. BTA's WorkGraph runs the aggregator each step. MFI's `dynamic_mode=True` allows multiple iterations.

YAML config:
```yaml
flow_max_dynamic_steps: 3   # per-flow MFDual followup iterations cap
```

But this is `flow_max_dynamic_steps` (per-flow LWI iterations), NOT for the aggregator. The aggregator's iteration count is governed by something else.

### §3.2 The Outer Dual Consensus Loop (Probable Cause)

**File**: `dual_inferencer.py` (consensus_max_iterations=3)

Looking at the workspace structure: Worker_0's `propose/` is itself a Dual (the MFDual's outer Dual). It has:
- propose (MFI) → review → fix → propose (MFI) → review → fix → ...

Each "fix" round potentially triggers a fresh MFI run, which means a fresh aggregator run.

**Estimated count**: 3 consensus iterations × 2-3 MFI re-runs = 6-9 aggregator invocations. Matches the observed 7.

### §3.3 Why Same JSONL Session?

The 7 InferenceInputs share ONE JSONL session file `RovoDevCliInferencer-74dbed09.jsonl`. This means:
- Aggregator instance is reused across iterations (single instance)
- Each iteration sends a fresh prompt (full `_build_agg_input` re-execution)
- RovoDev session persists across the iterations

This is by-design behavior for Dual review-fix loops with stable inferencer instances.

---

## §4 The Unified Fix

### §4.1 Fix A — Cascade `output_path` In LWI Workspace Propagation

**File**: `agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/linear_workflow_inferencer.py`

**Location**: `_propagate_workspace_to_children` method (lines ~220-244)

**Change**:

```python
def _propagate_workspace_to_children(self, parent_workspace):
    """Override to assign hierarchical workspaces to initial + followup."""
    if parent_workspace is None:
        return
    
    # Propagate workspace AND output_path to initial inferencer
    if self.initial_inferencer is not None:
        initial_ws = parent_workspace.child("initial")
        self.initial_inferencer._workspace = initial_ws
        # NEW: Cascade output_path so _finalize_output writes immediately
        if not getattr(self.initial_inferencer, "_output_path", None):
            self.initial_inferencer._output_path = self.output_path or "output.md"
    
    # Propagate workspace AND output_path to followup inferencer (if applicable)
    if self.followup_inferencer is not None:
        followup_ws = parent_workspace.child("round01")
        self.followup_inferencer._workspace = followup_ws
        # NEW: Cascade output_path
        if not getattr(self.followup_inferencer, "_output_path", None):
            self.followup_inferencer._output_path = self.output_path or "output.md"
    
    # Continue with base class propagation for other children
    super()._propagate_workspace_to_children(parent_workspace)
```

**Rationale**: Without explicit `output_path` cascade, round01 inferencer's `_finalize_output` is a no-op. With cascade, `outputs/output.md` is written on first iteration → symlink target valid → resolver returns valid path → aggregator gets file refs.

**Risk**: LOW. We only set `_output_path` if it's not already set (preserves explicit YAML overrides).

### §4.2 Fix B — Defensive Symlink Re-Resolution At Aggregator Build Time

**File**: `agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/breakdown_then_aggregate_inferencer.py`

**Location**: `_build_agg_input` closure (around lines 1655-1680)

**Change**: When `resolve_canonical_output_path` returns None, attempt to re-resolve by checking if the symlink target exists NOW (it might have been written between LWI and aggregator phases):

```python
def _build_agg_input(prompt_builder, worker_results, original_query):
    nonlocal _captured_paths
    if _worker_instances:
        from agent_foundation.common.inferencers.inferencer_workspace import (
            resolve_canonical_output_path,
        )
        new_paths = []
        for inst in _worker_instances:
            path = resolve_canonical_output_path(
                inst._workspace,
                filename=_bta_self.output_path or "output.md",
            )
            # NEW: Defensive fallback if path is None or dangling
            if path is None or not os.path.isfile(path):
                # Try common fallback locations
                ws_root = str(inst._workspace.root)
                candidates = [
                    os.path.join(ws_root, "outputs", "output.md"),
                    os.path.join(ws_root, "outputs", "final_deliverables", "output.md"),
                ]
                for cand in candidates:
                    if os.path.isfile(cand):
                        path = cand
                        break
            new_paths.append(path)
        _captured_paths = new_paths
    ...
```

**Rationale**: Belt-and-suspenders. Even if Fix A misses an edge case, this fallback ensures aggregator gets paths whenever they're recoverable.

**Risk**: LOW. Pure additive — only kicks in when primary resolver returns None.

### §4.3 Fix C — `_format_worker_results_text` Truncation Fallback

**File**: `breakdown_then_aggregate_inferencer.py` (around lines 645-660)

**Change**: When `path is None` AND we MUST inline, TRUNCATE the inlined content to a max length to bound the bloat:

```python
agg_has_local = (...)
paths = list(worker_output_paths or [])
parts = []
MAX_INLINED_CHARS = 2000  # Cap inlined content per worker
for idx, res in enumerate(worker_results):
    path = paths[idx] if idx < len(paths) else None
    if agg_has_local and path:
        parts.append(f"### Result {idx+1}\n(See file: {path})\n")
    else:
        text = str(res)[:MAX_INLINED_CHARS]
        if len(str(res)) > MAX_INLINED_CHARS:
            text += "\n... [TRUNCATED — see worker workspace for full output] ..."
        parts.append(f"### Result {idx+1}\n<Response>\n{text}\n</Response>\n")
```

**Rationale**: Even if Fixes A+B both fail to resolve paths, the inlined content is capped at ~2KB per worker instead of unbounded. Reduces 14KB inputs to ~5KB.

**Risk**: MEDIUM. Truncation may lose information. Mitigation: cap is generous (2000 chars ≈ 500 tokens), and aggregator prompt notes truncation explicitly.

### §4.4 Fix D — Investigate Aggregator Iteration Count (Symptom B Direct)

**Question**: Is 7 aggregator invocations correct for `consensus_max_iterations=3`?

**Investigation steps**:
1. Read `dual_inferencer.py` to count iterations per round
2. Verify outer Dual is correctly counting iterations
3. Check if there's a `max_iterations` cap on the MFI side that should bound aggregator runs separately

**Expected outcome**: Either confirm 7 is correct (consensus 3 × MFI internal 2-3) OR identify a runaway loop bug.

**If runaway**: Add Fix E — explicit `max_aggregator_invocations` cap on MFI/BTA.

**If correct**: Document the expected count + add monitoring/logging for visibility.

### §4.5 Fix E — Optional: Single-Pass Aggregator Mode

**Concept**: Allow MFI to be configured with `single_pass_aggregator=True` so the aggregator runs ONCE after all flows complete (not per Dual iteration). This decouples aggregator cost from review-fix iterations.

**Implementation**: Add YAML param, gate the aggregator invocation in MFI._ainfer.

**Risk**: HIGH (significant behavior change). Defer to follow-up plan.

---

## §5 Implementation Order

| Phase | Fix | Effort | Risk | Priority |
|---|---|---|---|---|
| 1 | Fix A — LWI output_path cascade | 15 min code + 30 min tests | LOW | 🔥 CRITICAL |
| 2 | Fix B — Defensive symlink re-resolution | 10 min code + 15 min tests | LOW | HIGH |
| 3 | Fix C — Truncation fallback | 5 min code + 10 min tests | MEDIUM | MEDIUM |
| 4 | Fix D — Iteration count investigation | 30 min investigation | NONE | HIGH (informs E) |
| 5 | Fix E — Single-pass mode | Defer | HIGH | DEFER |

**Total for Fixes A+B+C+D**: ~2 hours

---

## §6 Acceptance Criteria

After Fix A:
- [ ] **AC1**: round01 inferencer's `_output_path` is `"output.md"` (or YAML-set value) immediately after construction
- [ ] **AC2**: `outputs/output.md` exists in round01 workspace within 1 minute of round01 inferencer's first response
- [ ] **AC3**: LWI symlink at `flow_X/outputs/output.md` points to a real file (not dangling)

After Fix B:
- [ ] **AC4**: `_build_agg_input` returns valid paths for all workers whose output files exist on disk

After Fix C:
- [ ] **AC5**: When path is None for all workers, aggregator input is bounded to ~5KB (not 14KB+)

After Fix D:
- [ ] **AC6**: Documented expected aggregator invocation count for `consensus_max_iterations=3`
- [ ] **AC7**: If count is unexpectedly high, root cause is in §3 of follow-up plan

E2E (after all fixes):
- [ ] **AC8**: Live SOP run completes in <2 hours (vs 4+ hours pre-fix)
- [ ] **AC9**: Aggregator InferenceInputs contain `(See file: <path>)` references (NOT inlined `<Response>` text)
- [ ] **AC10**: Top-level deliverable produced and non-empty (existing behavior preserved)

---

## §7 Testing Strategy

### Unit Tests (in addition to existing 148 passing tests)

1. `test_lwi_propagates_output_path_to_initial`
2. `test_lwi_propagates_output_path_to_followup`
3. `test_lwi_preserves_explicit_output_path_override`
4. `test_finalize_output_writes_immediately_when_output_path_set`
5. `test_aggregator_recovers_path_via_fallback_when_primary_returns_none`
6. `test_inlined_content_truncated_at_max_chars`

### Integration Tests

7. `test_mfi_aggregator_input_uses_file_refs_when_paths_resolve`
8. `test_mfi_aggregator_falls_back_gracefully_when_paths_none`

### E2E Test

9. `test_sop_plan_run_completes_with_file_refs` — runs the actual SOP topology with mock LLM, verifies aggregator inputs contain file refs

---

## §8 Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Fix A cascade breaks existing leaf inferencers expecting None | LOW | Guard: only set if not already set + extensive unit tests |
| Fix B defensive fallback masks deeper bugs | MEDIUM | Add WARN log when fallback triggers; visibility into root cause |
| Fix C truncation loses information aggregator needs | MEDIUM | Generous 2000 chars ≈ 500 tokens; explicit "TRUNCATED" marker prompts agent to read files directly |
| Fix D investigation reveals aggregator runaway | LOW | Triggers Fix E (single-pass); adds opt-in cap |
| Symlinks created at construction may need re-creation if target overwritten | LOW | `_symlink_or_copy` already handles this case |

---

## §9 Open Questions

1. **OQ-1**: Is there an existing mechanism to cascade `output_path` that I'm missing? (E.g., `_DERIVED_FROM_WORKSPACE` infrastructure)
2. **OQ-2**: Should `output_path` cascade go through `LazyConfigFactory.injectables` instead of explicit propagation in LWI?
3. **OQ-3**: For Fix D — what's the EXPECTED count of aggregator invocations for `consensus_max_iterations=3 × max_dynamic_steps=3`? (Need to read Dual + MFI code carefully.)
4. **OQ-4**: Does the BTA outer aggregator (synthesizing worker_0 + worker_1) have the SAME bug? Need to inspect its inputs after a fixed run.

---

## §10 Provenance

- **v1.0 (2026-05-12 22:17)**: Initial plan after stopping run `task-e3ae2732`. Combines Symptom A (bloated inputs / output_path cascade) + Symptom B (7 aggregator invocations / iteration loop) into one unified plan. Hard evidence collected from 4+ deep investigation iterations of the actual workspace.
