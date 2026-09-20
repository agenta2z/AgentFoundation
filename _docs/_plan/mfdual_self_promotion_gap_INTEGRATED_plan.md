# MFDual Hygiene + Template Variable Population — INTEGRATED Fix Plan

**Status**: ACTIVE (v5.2 — Fix #9 RE-CORRECTED after deeper investigation. v5.1 cross-review was half-right (correctly located the gate at `file_based.py:755`) but proposed fix would BREAK existing intentional library contract. Real root cause: AgentFoundation `template_defaults.py` `REVIEW/FOLLOWUP_TEMPLATE_DEFAULTS` use plain `InferencerTemplateDefaults` instead of `InferencerTemplateVersionDefaults` like the proven aggregator pattern. ~20-min fix in 1 file (AgentFoundation). PLUS: 3 test gaps identified explaining why existing preflight didn't catch this.)
**Created**: 2026-05-10
**Combined sources**:
- `mfdual_self_promotion_gap_fix_plan.md` (v3.3, mine — superseded; archived as `_archive/mfdual_self_promotion_gap_fix_plan_v3.3.md`)
- `/Users/tchen7/.claude/plans/given-all-the-discussions-splendid-lantern.md` (alt — diagnoses adopted, structure from Plan A)
- `template_variable_population_UNIFIED_plan.md` v2.0 — INTEGRATED as Fix #9 (template-variable discovery decoupling)
**Severity**: 🔴 HIGH for Issues #1, #2, #4, #8, #9 (chain breaks, missing prompt content); 🟡 MEDIUM for #3 (operational); 🟢 LOW for #5/#6 (cosmetic)
**Scope**: 7 elegant code changes (no surgical patches), one architectural decoupling (#9), tests + live-run validation

---

## §1 Background — Empirical Evidence

For run `task_task-cbaf8f2b_20260510_033521` with shallow profile + SOP plan request:

| Layer | Symptom | Reality |
|---|---|---|
| Top-level fixer | `outputs/final_deliverables/output.md` (92 KB) + `.self_promoted` ✅ | Works correctly |
| Inner Dual fixer (in MFDual worker_0) | `outputs/output.md` (47 KB) but `final_deliverables/` EMPTY ❌ | Self-promotion did not fire |
| MFDual `outputs/output.md` | Only 2.2 KB (LLM TEXT response, not deliverable) ❌ | Inner content stranded |
| flow_X_followup/round_01 | 2 LLM calls in same workspace, both saying "stop" ❌ | Workspace stash reuse bug |
| Inner Dual reviewer | `round_01/review` symlinks to `flow_1_initial/` (no fresh `review_inferencer/`) ❌ | Identity-guard short-circuits |
| BTA aggregator input | `(See file: .../worker_0/outputs/output.md)` references the 2.2 KB summary, not the 47 KB deliverable ❌ | Wrong path resolution |
| Top-level outputs/ | No `output_manifest.json` ❌ | Provenance lost on surface |

The system "works" empirically only because the aggregator does redundant re-investigation. **Multiple coordinated gaps** in role-contract enforcement, workspace isolation, and surfacing semantics.

---

## §2 Issues Inventory & Verified Root Causes

The original Plan A had 8 issues (A-H); after rigorous re-diagnosis with Plan B's evidence, the corrected inventory is:

| Issue | Severity | Root cause (verified) | Fix | Effort |
|---|---|---|---|---|
| **#1** Reviewer identity-guard snapshot ordering | 🔴 HIGH | Snapshot of `_review_inferencer_original` taken at line 383 — AFTER the `reviewer_match_second` override at line 349-355. So snapshot equals current value, identity guard always short-circuits in `_reassign_role_workspace`. | Move snapshot BEFORE `reviewer_match_second` override | ~30 min |
| **#2** Aggregator references thin `outputs/output.md` instead of full `final_deliverables/output.md` | 🔴 HIGH | `_format_worker_results_text()` uses caller-provided `worker_output_paths`, which currently use `worker.resolve_output_path()` (returns `outputs/output.md`). Should use `resolve_canonical_output_path()` which Tier-1-checks `final_deliverables/`. | Replace path source in `_inject_aggregator_extra_feed` (or wherever paths are resolved before passed in) | ~30 min |
| **#3** Top-level outputs missing `output_manifest.json` | 🟡 MED | Two coupled gaps: (a) `surface_outputs_from()` only walks `final_deliverables/`, not parent `outputs/`; (b) Dual itself doesn't have `output_is_deliverable: true` so `_post_finalize_deliverable_and_manifest()` never runs at the orchestrator level. | Dual generates its OWN top-level manifest in `_finalize_response()` (with phase, active_proposer, iteration count, source workspace) | ~30 min |
| **#4** Two LLM calls in same `round_01` (workspace stash reuse) | 🟡 MED | LWI's per-round workspace stashes `_base_followup_workspace` once via `hasattr` check (line 536-538). When MFDual's fix-phase RE-INVOKES the winning flow's instance, stash persists → re-derives `round_01` → 2nd call lands in same workspace. | Reset `_base_followup_workspace` between consensus iterations (or include consensus iteration in round naming) | ~30 min |
| **#5** Followup rounds nested under `children/round_NN/` instead of flat `flow_X_followup_round_NN/` | 🟢 LOW | Uses `InferencerWorkspace.child()` API convention, not deliberate design. | Create at parent level: `parent_ws.child(f"flow_{i}_followup_round{N:02d}")`. Requires LWI wrapper to access parent MultiFlow's workspace. | ~45 min |
| **#6** `worker_N/` naming confusion (LWI checkpoint containers, not flow content) | 🟢 LOW | Pre-existing naming convention. `worker_N/` = LWI workflow orchestration containers; `flow_N_*` = actual flow workspaces. Misleading but architecturally accurate. | Rename `worker_N/` → `flow_N_workflow/` for clarity (deeper refactor) OR just document the convention | ~10 min (docs) / 1h (rename) |
| **#7** Followup `final_deliverables/` empty | ⚪ NO-OP | **CORRECT BY DESIGN.** Followup feeds aggregator via `_latest_per_flow`; aggregator surfaces. Followup self-promotion would be redundant; current path-aware Tier-2 fallback handles it. | No fix needed | 0 |
| **#8** **REINSTATED** Inner fixer's `output_is_deliverable=False` because winner-as-fixer mutation loses YAML's flag | 🔴 **HIGH** — DELIVERABLE CHAIN BROKEN | When `_select_reviewer_and_fixer` does `self.fixer_inferencer = winner`, it assigns the winning flow's instance. That instance came from `flow_configs` worker_factory and does NOT have `output_is_deliverable: true` set. So inner fixer's `output_is_deliverable` is `False`, self-promotion at `inferencer_base.py:754` skipped, `final_deliverables/` empty, surface_outputs_from no-ops, MFDual gets 2.2 KB summary instead of 47 KB content. **Fix #2's canonical resolver alone CAN'T reach the 47 KB because it's stranded 2 layers deep.** | After winner assignment, set `self.fixer_inferencer.output_is_deliverable = True` (and similarly for reviewer if applicable). Encode this in `_select_reviewer_and_fixer` so it's part of the role contract — when an instance assumes the fixer role, it inherits the role's deliverable contract. | ~30 min |
| **#9** **RE-CORRECTED v5.2** `REVIEW_TEMPLATE_DEFAULTS` and `FOLLOWUP_TEMPLATE_DEFAULTS` don't set `template_version` → cascade `version=""` → Pass 2 skipped (intentional library contract) → `default.jinja2` never loaded | 🔴 **HIGH** — PROMPT CONTENT MISSING | The `if version:` gate in `file_based.py:755` is INTENTIONAL — `test_pass2_skipped_when_version_empty` asserts the contract. Library design: deployment-level `template_version` (`enterprise`/`apac`) MUST be set for folder-based variable resolution. **Aggregator works** because `STRUCTURED_AGGREGATION_DEFAULTS` uses `InferencerTemplateVersionDefaults(template_version=VARIANT_AGGREGATION, variable_names=[...])` — per-inferencer template_version unblocks Pass 2. **Review/followup don't work** because `REVIEW_TEMPLATE_DEFAULTS`/`FOLLOWUP_TEMPLATE_DEFAULTS` use plain `InferencerTemplateDefaults` (no version, no variable_names). | Switch from `InferencerTemplateDefaults` to `InferencerTemplateVersionDefaults` with `template_version=VARIANT_DEFAULT` (`"default"`) and `variable_names=[VAR_TASK_PREAMBLE, VAR_TASK_INSTRUCTIONS, VAR_TASK_RESPONSE_FORMAT]`. Mirror the proven `STRUCTURED_AGGREGATION_DEFAULTS` pattern. NO RichPythonUtils changes. | ~20 min |

### Issue #9 — Template variables silently NOT populated (semantic conflation)

**Verified 2026-05-10 11:04** by grep on rendered fixer prompt — 0 occurrences of `task_instructions` despite 14 template references.

**Symptoms**:
- Templates `plan/main/{initial,review,followup}.jinja2`, `implementation/main/{initial,review,followup}.jinja2`, `task_breakdown/main/initial.jinja2` all reference `{{ task_instructions }}`
- All references inside `{% if task_instructions %}...{% endif %}` guards
- Variable is NOT populated at render time → guards silently skip → prompt missing entire instruction section
- Users see no error — just missing content

**Verified Root Cause** (`templated_inferencer_base.py:163`):
```python
if self.template_variables and self.template_manager:  # ← THE BUG
    resolved = self.template_manager.load_variables(...)
```

This conflates **TWO orthogonal concerns** into ONE condition:

| Concern | Should be | Is currently |
|---|---|---|
| **"Should I scan `_variables/` for defaults?"** | Always yes (when `template_root_space` set) | Gated on `template_variables` non-empty |
| **"What overrides do I apply?"** | Whatever's in `template_variables` | Same gate |

The `default.jinja2` filename convention (e.g., `_variables/task_instructions/default.jinja2` exists at 2300 bytes) PROVES the original design intent was "auto-load defaults; `template_variables` is for overrides."

But the code never implemented the auto-discovery side. Result:
- `template_variables = {}` → no loading at all (BUG)
- `template_variables = {task_preamble: aggregation}` → ONLY `task_preamble` loaded; `task_instructions`/`task_response_format` still NOT loaded (also BUG)

**The conflation is the bug.** `template_variables` should be PURELY about overrides, never about whether discovery happens.

Severity: 🔴 HIGH — silent failure; entire `task_instructions` section missing from fixer/reviewer/initial prompts.

Effort: ~2 hours code + tests (Fix #9 spec in §3).

---

### Issues from Plan A (v3.3) That Are Now Removed/Reframed

- **Old Issue A** (inner Dual fixer self-promotion fail) — was diagnosed as "fixer_match_winner mutation loses output_is_deliverable flag." This was 100% correct as a diagnosis. v4.0 retracted it as "folded into Issue #4" — that was a MISTAKE. Cross-review at v4.2 caught the regression: Fix #2 (canonical resolver) can NOT reach the 47 KB content because it's stranded 2 layers deep at `fixer_inferencer/outputs/output.md`. Tier 1 (MFDual's `final_deliverables/`) is empty because no self-promotion fired anywhere in the inner Dual → MFDual chain. **REINSTATED as Issue #8.** This is independent from and necessary alongside Fix #2.
- **Old Issue B** (reviewer not isolated) — replaced by precise Issue #1 (identity-guard snapshot ordering).
- **Old Issue C** (followup self-promote) — removed, retraced as correct-by-design (now Issue #7).
- **Old Issue G** (aggregator silently compensates) — replaced by precise Issue #2 (path resolution).
- **Old Issue H** (top-level manifest missing) — superseded by precise Issue #3 (Dual generates own manifest, not copy).

---

## §3 Detailed Fix Specs

### Fix #1 — Move identity-guard snapshot BEFORE `reviewer_match_second` override

**File**: `agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/multi_flow_dual_inferencer.py`

**Current (broken)** — verified line numbers in `__attrs_post_init__`:
```python
# Line 348-355: reviewer_match_second early override
if (self.review_inferencer is None and self.reviewer_match_second
    and len(self.flow_configs) > 1):
    self.review_inferencer = self.flow_configs[1].get("initial_inferencer")

# Line 357-364: fixer_match_winner placeholder
if (self.fixer_inferencer is None and self.fixer_match_winner
    and self.review_inferencer is not None):
    self.fixer_inferencer = self.review_inferencer

# Line 367: super() — DualInferencer's default-resolution
super().__attrs_post_init__()

# Line 382-383 (snapshot — AFTER all overrides AND DualInferencer defaults!):
self._fixer_inferencer_original = self.fixer_inferencer
self._review_inferencer_original = self.review_inferencer  # ← snapshot too late!
```

**Fixed** — snapshot at the VERY BEGINNING of `__attrs_post_init__`, before any override:
```python
def __attrs_post_init__(self):
    # FIRST: snapshot YAML-configured originals BEFORE any override
    # (whether MFDual's reviewer_match_second/fixer_match_winner, or
    # DualInferencer's super().__attrs_post_init__() defaults).
    self._fixer_inferencer_original = self.fixer_inferencer    # YAML value (likely None)
    self._review_inferencer_original = self.review_inferencer  # YAML value (likely None)

    # ... rest of existing logic (reviewer_match_second override at 348-355,
    #     fixer_match_winner placeholder at 357-364, super() at 367, etc.)
```

This way the identity guard correctly detects "the current value differs from the YAML-configured original" → reassigns workspace.

**Acceptance criterion**: After fix, `_reassign_role_workspace(loser_flow_instance, "review_inferencer")` no longer short-circuits because `loser_flow_instance is not None_or_yaml_value`. Fresh `review_inferencer/` workspace is created.

**Test path**: `test/agent_foundation/common/inferencers/agentic_inferencers/test_multi_flow_dual_inferencer.py` — add `test_reviewer_workspace_isolated_when_loser_assumes_role`.

### Fix #2 — Use `resolve_canonical_output_path()` for aggregator's worker references

**File**: `agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/breakdown_then_aggregate_inferencer.py`

**Find caller of `_format_worker_results_text`** (line 639 + 1678 + closures). The `worker_output_paths` argument is currently populated using `worker.resolve_output_path()` (returns `outputs/output.md`).

**Replace path source**:
```python
# BEFORE: worker_output_paths.append(worker.resolve_output_path("output.md"))
# AFTER:
from agent_foundation.common.inferencers.inferencer_workspace import resolve_canonical_output_path
path = resolve_canonical_output_path(
    workspace=worker._workspace,
    filename="output.md",
    fallback_strategy="first_match",  # Tier 1 final_deliverables/, Tier 2 outputs/
)
worker_output_paths.append(path)
```

This makes `(See file: ...)` references point to the FULL deliverable when available, falling back to `outputs/output.md` when not.

**Acceptance criterion**: After fix, aggregator's prompt's `(See file:)` references show `final_deliverables/output.md` (47 KB+) for workers that have deliverables.

**Test path**: `test/.../test_breakdown_then_aggregate_inferencer.py` — add `test_format_worker_results_uses_canonical_path`.

### Fix #3 — Dual generates its OWN top-level `output_manifest.json`

**File**: `agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/dual_inferencer.py`

**In `_finalize_response()`**, after `surface_outputs_from()` succeeds:
```python
# After surfacing the active proposer's deliverables to self._workspace:
state = self._state
counter = state.get("attempt_record", {}).get("iterations", [{}])[-1].get("counter_feedback")
active_label = "fixer" if counter is not None else "base"

manifest = {
    "source": type(self).__name__,
    "phase": getattr(self, "phase", None),
    "active_proposer": active_label,
    "total_iterations": len(state.get("attempt_record", {}).get("iterations", [])),
    "consensus_achieved": state.get("consensus_reached", False),
    "proposer_workspace": str(getattr(active, "_workspace", None).root) if active and getattr(active, "_workspace", None) else None,
    "deliverable_file": "output.md",
}

manifest_path = self._workspace.output_path("output_manifest.json")
import json
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2, default=str)
```

This gives the top-level workspace its OWN provenance record — orchestrator's perspective, not a copy of leaf's manifest.

**Why NOT just copy the leaf's manifest**: The leaf's manifest has fixer-relative paths and fixer-scoped provenance (which LLM call, what cache key). Top-level should answer "which proposer won, how many rounds did consensus take, where did the surfaced content come from."

**Acceptance criterion**: After fix, `task_*/outputs/output_manifest.json` exists with `active_proposer`, `total_iterations`, `proposer_workspace` fields.

**Test path**: `test/.../test_dual_inferencer.py` — add `test_finalize_response_emits_top_level_manifest`.

### Fix #4 — Reset `_base_followup_workspace` stash between consensus iterations

**File**: `agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/linear_workflow_inferencer.py`

**Current (line 536-538)**:
```python
if not hasattr(inf_instance, "_base_followup_workspace"):
    inf_instance._base_followup_workspace = ws
base_ws = inf_instance._base_followup_workspace
```

This stash persists across re-invocations — exactly what causes Issue #4.

**Adopted approach (v4.2 elegant) — derived-state auto-invalidation EXTENDS existing property setter** (per §8 Resolved Decisions #4):

The `_workspace` property setter on `InferencerBase` (already defined at lines 232-237) is **extended (not replaced)** to also auto-clear derived state when reassigned. ZERO cross-class coupling — no method to remember to call, no cross-class attribute knowledge, no explicit `delattr`. The semantic is "derived state always reflects current source state."

**EXISTING setter (lines 232-237 of `inferencer_base.py`)** — must be PRESERVED:
```python
@_workspace.setter
def _workspace(self, value):
    object.__setattr__(self, "_InferencerBase__workspace", value)  # name-mangled storage
    if value is not None:
        self._configure_for_workspace(value)         # working_dir, logger redirection
        self._propagate_workspace_to_children(value) # walk children, assign their workspaces
```

**EXTENSION to add** — auto-invalidation block, must be ADDITIVE (not a replacement):
```python
@_workspace.setter
def _workspace(self, value):
    # ────── EXISTING logic preserved ──────
    object.__setattr__(self, "_InferencerBase__workspace", value)
    if value is not None:
        self._configure_for_workspace(value)
        self._propagate_workspace_to_children(value)
    # ────── NEW: auto-invalidate derived state ──────
    # Note: walks the MRO via `getattr(type(self), ...)` so subclasses automatically
    # override `_DERIVED_FROM_WORKSPACE` via standard class-attribute resolution.
    for attr in getattr(type(self), '_DERIVED_FROM_WORKSPACE', ()):
        self.__dict__.pop(attr, None)
```

**Subclass extension via class-attribute override**:
```python
# In InferencerBase:
_DERIVED_FROM_WORKSPACE: tuple = ()  # base: no derived state

# In LinearWorkflowInferencer:
_DERIVED_FROM_WORKSPACE = ('_base_followup_workspace',)

# Future subclasses just override the tuple — no setter changes needed.
```

**Why this preserves existing behavior**: The new block is purely additive — it runs AFTER the existing `_configure_for_workspace` + `_propagate_workspace_to_children` calls. If a subclass has no `_DERIVED_FROM_WORKSPACE`, the loop is empty and behavior is identical to today.

**PLUS Fix #4b — per-consensus-iteration directory suffix** (necessary because auto-clearing alone would re-derive the SAME path and overwrite content):

```python
# In LWI's per-round directory derivation:
consensus_iter = state.get("consensus_iteration_id", 0)
suffix = f"_iter{consensus_iter}" if consensus_iter > 0 else ""
round_dir = f"round_{step_index:02d}{suffix}"
round_ws = base_ws.child(round_dir)
```

**How `consensus_iteration_id` is propagated** — must be specified explicitly:

DualInferencer already tracks consensus iterations via `state["attempt_record"]["iterations"]` (a list whose length = current iteration count). To make this available inside the LWI invoked as the fixer, DualInferencer's `_run_fix_attempt` (or wherever fix-phase calls into the fixer's `ainfer`) MUST inject the iteration counter into the fixer's local state BEFORE calling its `ainfer`:

```python
# In DualInferencer, before invoking fixer's ainfer for fix-phase:
fix_phase_state = self._build_state_for_role("fixer")  # existing API
fix_phase_state["consensus_iteration_id"] = len(self.state["attempt_record"]["iterations"])  # NEW
await self.fixer_inferencer.ainfer(fix_input, state=fix_phase_state)
```

**Mechanism**: pre-injection into the per-call `state` dict (NOT instance attribute, NOT extra_feed). Reasons:
- `state` dict is the canonical inter-inferencer communication channel for runtime values.
- Instance attribute would persist across unrelated calls, polluting state.
- `extra_feed` is for template rendering (orchestrator → leaf), not for orchestration metadata.

**Acceptance test**: verify Dual's `_run_fix_attempt` injects `consensus_iteration_id` and LWI's per-round derivation reads it correctly. (Test #14 verifies the LWI side; need to also test the Dual injection side.)

**Why NOT the older Option B (explicit `delattr`)**: Cross-class coupling — MFDual would need to know LWI's stash attribute name. Property-setter pattern is more general and doesn't leak abstractions.

**Acceptance criterion**: After fix, fix-phase re-invocation creates `round_02/` (or `round_NN_iterY/`), not reuses `round_01/`.

**Test path**: `test/.../test_linear_workflow_inferencer.py` — add `test_workspace_stash_reset_between_consensus_iterations`.

### Fix #8 — Role-contract inheritance (winner-as-fixer inherits `output_is_deliverable`)

**File**: `agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/multi_flow_dual_inferencer.py` — `_select_reviewer_and_fixer` method.

**The principle**: When an instance assumes a role (fixer/reviewer) at runtime, it inherits that role's contract — including the deliverable-promotion semantic. The YAML-declared `fixer_inferencer` block has `output_is_deliverable: true`, but when MFDual mutates `self.fixer_inferencer = winner`, the winning flow's instance comes from `flow_configs` with `output_is_deliverable: false`. Without inheritance, self-promotion at `inferencer_base.py:754` is skipped, breaking the entire deliverable chain.

**Current (broken)**:
```python
if chosen is not None:
    self.fixer_inferencer = chosen
elif self.fixer_match_winner and winner is not None:
    self.fixer_inferencer = winner
```

**Fixed**:
```python
if chosen is not None:
    self.fixer_inferencer = chosen
    self.fixer_inferencer.output_is_deliverable = True   # role-contract inheritance
elif self.fixer_match_winner and winner is not None:
    self.fixer_inferencer = winner
    self.fixer_inferencer.output_is_deliverable = True   # role-contract inheritance
```

**Why this is the elegant fix (not a hack)**: Encodes the principle "role assumption = contract inheritance." If MFDual ever adds more role-derived contracts (e.g., custom output filenames per role), they go in the same place. The mutation site is the single source of truth for role-contract handoff.

**Acceptance criterion**: After fix, `worker_0/children/fixer_inferencer/outputs/final_deliverables/output.md` exists (47 KB+ from inner fixer).

**Test path**: `test/agent_foundation/common/inferencers/agentic_inferencers/test_multi_flow_dual_inferencer.py` — add `test_winner_as_fixer_inherits_output_is_deliverable`.

**Why this UNBLOCKS Fix #2**: Without Fix #8, Fix #2's canonical resolver returns Tier 2 (2.2 KB summary). With Fix #8, the deliverable chain populates Tier 1 (47 KB content). The two fixes are complementary; Fix #8 is the prerequisite.

---

### Fix #9 (v5.2) — Switch `REVIEW_TEMPLATE_DEFAULTS` and `FOLLOWUP_TEMPLATE_DEFAULTS` to use `InferencerTemplateVersionDefaults`

**File**: `agent_foundation/common/inferencers/template_defaults.py` (~lines 290-307)

**The TRUE Root Cause** (verified with hard evidence 2026-05-10 11:33):

The library's `if version:` gate at `file_based.py:755` is **INTENTIONAL** — `test_pass2_skipped_when_version_empty` (lines 96-104) asserts: *"Even though `default.j2` exists, `version=""` means Pass 1+2 skipped."* The file header documents: *"`version=""` skips Pass 1 + Pass 2 (preserves existing semantics)."*

The library design: **`template_version`** is a deployment-level concept (e.g., `enterprise`, `apac`, `default`) that MUST be set for folder-based default-variant lookup. The bug is at the **AgentFoundation layer**: `REVIEW_TEMPLATE_DEFAULTS` and `FOLLOWUP_TEMPLATE_DEFAULTS` don't set it.

**Proof — why aggregator works**:
```python
STRUCTURED_AGGREGATION_DEFAULTS = InferencerTemplateVersionDefaults(  # Note: VERSION variant
    template_version=VARIANT_AGGREGATION,                              # truthy!
    variable_names=[VAR_TASK_PREAMBLE, VAR_TASK_INSTRUCTIONS, VAR_TASK_RESPONSE_FORMAT],
)
```
This sets `template_version="aggregation"` per-inferencer. Pass 1 finds `aggregation.jinja2` if exists; else Pass 2 finds `default.jinja2` (gate truthy). Variables load correctly.

**Proof — why fixer/reviewer don't work**:
```python
REVIEW_TEMPLATE_DEFAULTS = InferencerTemplateDefaults(   # ← no version!
    template_key=KEY_REVIEW,                              # ← no variable_names!
)

FOLLOWUP_TEMPLATE_DEFAULTS = InferencerTemplateDefaults( # ← same!
    template_key=KEY_FOLLOWUP,
)
```
No `template_version` → `version=""` cascade → Pass 1+2 skipped → `default.jinja2` invisible.

**The Fix** — mirror the aggregator pattern:
```python
REVIEW_TEMPLATE_DEFAULTS = InferencerTemplateVersionDefaults(   # ← changed
    template_key=KEY_REVIEW,
    template_version=VARIANT_DEFAULT,                            # ← new (= "default")
    variable_names=[VAR_TASK_PREAMBLE, VAR_TASK_INSTRUCTIONS, VAR_TASK_RESPONSE_FORMAT],
)

FOLLOWUP_TEMPLATE_DEFAULTS = InferencerTemplateVersionDefaults(  # ← changed
    template_key=KEY_FOLLOWUP,
    template_version=VARIANT_DEFAULT,                            # ← new
    variable_names=[VAR_TASK_PREAMBLE, VAR_TASK_INSTRUCTIONS, VAR_TASK_RESPONSE_FORMAT],
)
```

(Add `VARIANT_DEFAULT = "default"` constant to module if not present.)

**Why this is the elegant fix**:

| Comparison | v5.1 (cross-review) | v5.2 (correct) |
|---|---|---|
| Layer | RichPythonUtils library contract | AgentFoundation slot defaults |
| Breaks existing tests | YES (`test_pass2_skipped_when_version_empty`) | NO |
| Preserves library contract | NO | YES |
| Uses proven existing pattern | NO (invents new behavior) | YES (mirrors `STRUCTURED_AGGREGATION_DEFAULTS`) |
| Effort | 30 min + library risk | 20 min, AF-only |
| Future callers using `predefined_variables: true` directly | Would break their `version=""` semantic | Works as designed (set per-inferencer template_version when defaults wanted) |

### Why Existing Preflight Tests Didn't Catch This (3 test gaps)

The `test_preflight_template_variable_coverage.py` test exists with 4 stated coverage goals — but didn't catch this bug because:

| Gap | Detail | Fix |
|---|---|---|
| **Gap 1** — Mock TemplateManager bypasses real loader | `_RecordingTemplateManager.load_variables` returns `LOADED_VAR:{spec}` for whatever spec is passed; doesn't exercise the real Pass-1/Pass-2 file resolution chain | Add integration test using REAL `TemplateManager` against REAL `_variables/` filesystem |
| **Gap 2** — `setdefault` masks empty values | `test_path_aware_followup.py:465` does `feed.setdefault("task_preamble", "")`, suppressing the visibility of unset variables | Remove defensive setdefaults in tests; let absent variables fail loudly |
| **Gap 3** — `{% if var %}` guards mask silent skips | Templates wrap `{{ task_instructions }}` in `{% if task_instructions %}...{% endif %}` so empty values render nothing without raising | Add a "strict-rendering" preflight that asserts critical sections appear in the rendered prompt for the production wiring |

### Acceptance Criteria

- ✅ Fixer rendered prompt contains a non-empty "Task Instructions:" section (≥ 1 grep match)
- ✅ Reviewer rendered prompt contains "Task Instructions:" section
- ✅ Aggregator prompts unchanged (already worked)
- ✅ `template_variables: {task_preamble: aggregation}` STILL overrides the default
- ✅ Existing `test_pass2_skipped_when_version_empty` STILL passes (we don't touch RichPythonUtils)
- ✅ NEW: integration test using real `TemplateManager` proves variables load
- ✅ NEW: strict-rendering preflight test fails loudly if critical sections empty

### Test Path

- `AgentFoundation/test/.../test_template_defaults.py` — add `test_review_followup_defaults_use_version_defaults`
- `AgentFoundation/test/.../test_template_split_verification.py` (extend) — add `test_real_variable_loading_for_fixer_followup_renders_task_instructions`
- `AgentFoundation/test/.../test_dual_inferencer/test_strict_rendering_preflight.py` (NEW) — assert "Task Instructions:" appears in real fixer prompt

---

### [SUPERSEDED v5.1] Fix #9 — Remove `if version:` gate from Pass 2 (KEPT FOR REFERENCE, DO NOT IMPLEMENT)

**Status**: ❌ SUPERSEDED by v5.2. Would BREAK the existing intentional library contract test.

**File** (incorrectly identified): `RichPythonUtils/src/rich_python_utils/common_objects/variable_manager/file_based.py:754-761`

**The Real Root Cause** (verified with hard evidence 2026-05-10 11:29):

The TemplateManager's auto-discovery infrastructure ALREADY exists and works:
- YAML sets `predefined_variables: true` on TemplateManager (line 70 of `breakdown-multiflow-plan.yaml`)
- `template_manager.py:1798-1818` calls `loader.resolve_from_template(...)` after template resolution
- `VariableManager.resolve_from_template()` regex-scans the template for `{{ var }}` references
- For each found variable, calls `_resolve_variable("task_instructions", "plan", "main", version="")`
- `_find_variable_file("task_instructions", cascade_paths, version="")` is invoked

The bug is in `file_based.py` Pass 2 (lines 754-761):
```python
# ----- PASS 2: default search across all cascade levels -----
if version:  # ← BUG: skips when version="" (the common case)
    for cascade_path in cascade_paths:
        for path_variant in possible_paths:
            folder = cascade_path / path_variant
            resolved = self._find_in_variable_folder(folder, "default")
            if resolved is not None:
                return (resolved, variable_name)
```

When `version=""` (no explicit variant requested), Pass 2 is SKIPPED — so `default.jinja2` is never looked up. Pass 3 only does a flat-file search for `task_instructions.jinja2` (which doesn't exist; it's a folder).

**Why aggregator works** (proves diagnosis): `STRUCTURED_AGGREGATION_DEFAULTS` sets `template_version="aggregation"` → `version="aggregation"` (truthy) → Pass 1 finds `aggregation.jinja2`, or if not found, Pass 2 falls back to `default.jinja2`. The fixer/reviewer have NO `template_version` → `version=""` → Pass 2 skipped → bug.

**The Fix** — remove the `if version:` guard:
```python
# ----- PASS 2: default search across all cascade levels -----
# Always check default.jinja2 — it IS the no-version fallback.
# When version is set, this runs after Pass 1 (version-specific not found).
# When version is empty, this finds the canonical default.
for cascade_path in cascade_paths:
    for path_variant in possible_paths:
        folder = cascade_path / path_variant
        resolved = self._find_in_variable_folder(folder, "default")
        if resolved is not None:
            return (resolved, variable_name)
```

**Principle**: `default.jinja2` is semantically the "no-version specified" fallback. Pass 2 must run unconditionally so the convention works as documented. The agent that caught this got it exactly right: my earlier diagnosis was a workaround at the wrong layer.

**Why this is the elegant fix (not what I previously proposed)**:

| My v5.0 proposal (REJECTED) | The correct fix (v5.1) |
|---|---|
| Add `_load_default_variables()` helper in AgentFoundation | NO AgentFoundation changes |
| Add `discover_default_variables()` method to TemplateManager | NO TemplateManager changes |
| Cache filesystem scan per inferencer | Existing infrastructure already caches |
| Calls `load_variable(version="default")` to bypass the bug | Fixes the actual bug |
| ~2 hours of new code | 1-line change in 1 file |
| Adds parallel discovery path | Reuses existing `predefined_variables: true` path |
| Future callers using `predefined_variables: true` directly STILL broken | All callers benefit |

**Acceptance**:
- Fixer rendered prompt contains `task_instructions` content (≥ 1 grep match)
- Reviewer rendered prompt contains `task_instructions` content
- Aggregator prompts unchanged (already worked via Pass 2 with truthy version)
- `template_variables: {task_preamble: aggregation}` STILL overrides `task_preamble`
- `_variables/foo/default.jinja2` auto-loads `{{ foo }}` for all referencing templates without YAML changes

**Test path**: `RichPythonUtils/test/.../test_file_based.py` — add `test_pass_2_runs_when_version_empty`. Optionally add an end-to-end test in AgentFoundation: `test/.../test_dual_inferencer.py::test_dual_fixer_renders_with_task_instructions`.

---

### [SUPERSEDED] Original Fix #9 v5.0 — Decouple template-variable discovery from override gate (KEPT FOR REFERENCE)

**Status**: ❌ SUPERSEDED by v5.1 above. Was misdiagnosed at the wrong layer (AgentFoundation). The actual bug is in RichPythonUtils. The diagnosis below is preserved for historical reference; DO NOT IMPLEMENT.

**File** (incorrectly identified): `agent_foundation/common/inferencers/templated_inferencer_base.py` + companion in `rich_python_utils/string_utils/formatting/template_manager/template_manager.py`

**Current** (`templated_inferencer_base.py:163`):
```python
if self.template_variables and self.template_manager:  # BUGGY: conflated gate
    resolved = self.template_manager.load_variables(
        variable_specs=self.template_variables,
        ...
    )
    feed.update(resolved)
```

**Fixed**:
```python
if self.template_manager and self.template_root_space:
    # Step A (NEW): Always load defaults from _variables/<name>/default.jinja2
    feed.update(self._load_default_variables())

    # Step B (EXISTING, gate corrected): Apply explicit overrides
    if self.template_variables:
        resolved = self.template_manager.load_variables(
            variable_specs=self.template_variables,
            root_space=self.template_root_space,
            default_version=self.template_version or "",
        )
        feed.update(resolved)  # overrides win over defaults
```

**New helper method** (in same file):
```python
def _load_default_variables(self) -> dict:
    """Auto-load `_variables/<name>/default.jinja2` files under root_space.

    Implements the design intent: any `_variables/` subdirectory with a
    `default.jinja2` file becomes an auto-loaded variable. The variable
    name = directory name; value = file content. Cached per instance.

    Returns empty dict if no `_variables/` exists or template_manager
    doesn't support discovery (graceful degradation).
    """
    cache_attr = "_default_variables_cache"
    if hasattr(self, cache_attr):
        return getattr(self, cache_attr)

    defaults: dict = {}
    try:
        var_names = self.template_manager.discover_default_variables(
            root_space=self.template_root_space,
            template_type=self.template_type or "main",
        )
        for var_name in var_names:
            try:
                content = self.template_manager.load_variable(
                    var_name=var_name, version="default",
                    root_space=self.template_root_space,
                )
                defaults[var_name] = content
            except (FileNotFoundError, AttributeError):
                pass  # variant changed between scan and load
    except (AttributeError, NotImplementedError):
        pass  # template_manager doesn't support discovery yet

    object.__setattr__(self, cache_attr, defaults)
    return defaults
```

**Companion method** (in `template_manager.py`):
```python
def discover_default_variables(
    self, root_space: str, template_type: str = "main",
) -> list[str]:
    """Scan `<root_space>/<template_type>/_variables/*/default.jinja2`.

    Returns sorted list of variable names with default.jinja2 present.
    Empty list if no _variables/ dir found (graceful degradation).
    """
    candidates = [
        f"{root_space}/{template_type}/_variables",
        f"{root_space}/_variables",
        "_variables",
    ]
    found = set()
    for base in candidates:
        for search_path in self.search_paths:
            full = os.path.join(search_path, base)
            if not os.path.isdir(full):
                continue
            for entry in os.listdir(full):
                entry_path = os.path.join(full, entry)
                if (os.path.isdir(entry_path) and
                    os.path.isfile(os.path.join(entry_path, "default.jinja2"))):
                    found.add(entry)
    return sorted(found)
```

**Principle**: Decouple discovery (always-on) from override (driven by `template_variables`). `template_variables` becomes PURELY an override mechanism — empty means "use all defaults," non-empty means "for these specific variables, use this variant instead of default."

**Acceptance**:
- Fixer rendered prompt contains `task_instructions` content (≥ 1 grep match)
- Reviewer rendered prompt contains `task_instructions` content
- Aggregator prompts unchanged (no regression)
- `template_variables: {task_preamble: aggregation}` overrides ONLY `task_preamble`; `task_instructions` STILL loaded as default

**Test**: Multiple — see §7 Tests #15–#19.

---

### Fix #5 (LOW priority, optional) — Flat `flow_X_followup_round_NN/` layout

Defer unless adopted as part of broader workspace-naming refactor. Issue #5 is purely cosmetic.

### Fix #6 (LOW, optional) — Document `worker_N/` vs `flow_N_*` convention

Add a `README.md` in MFDual's children/ explaining the dual-namespace.

---

## §4 Implementation Order

**ORDER MATTERS** — Fix #8 must precede Fix #2 (Fix #2's acceptance depends on Fix #8 enabling the deliverable chain). Fix #9 is independent of MFDual fixes (separate subsystem) but MUST be tested before final SOP run since it affects all rendered prompts.

1. **Fix #1** — Reviewer snapshot ordering (workspace isolation prerequisite) — ~30 min
2. **Fix #8** — Role-contract inheritance (deliverable chain prerequisite) — ~30 min
3. **Fix #2** — Canonical path resolver in BTA aggregator (depends on #8) — ~1h
4. **Fix #4** — Workspace stash auto-invalidation (multi-iteration safety) — ~1h
5. **Fix #3** — Top-level Dual `output_manifest.json` — ~30 min
6. **Fix #9** (v5.2 corrected) — Switch `REVIEW/FOLLOWUP_TEMPLATE_DEFAULTS` to `InferencerTemplateVersionDefaults(template_version=VARIANT_DEFAULT, variable_names=[...])` mirroring `STRUCTURED_AGGREGATION_DEFAULTS` — ~20 min code + ~30 min tests (3 new tests addressing the test gaps that masked this bug)
7. **Fix #5** — Flat `flow_X_followup_round_NN/` layout — ~1h
8. **Fix #6** — Eliminate `worker_N/` shell directories OR document convention — ~1h (optional)
9. **Tests** (per §7) — ~2.5h
10. **Live SOP run validation** — ~1.5h

**Total**: ~10 hours (v5.2: Fix #9 simplified to ~50 min total — 20 min code + 30 min tests covering the 3 identified test gaps).

### Detailed Dependency Notes

1. **Fix #1** (reviewer snapshot ordering) — highest impact, smallest change. Independent.
2. **Fix #8** (winner-as-fixer role-contract inheritance) — **MUST come before Fix #2**, because Fix #2's acceptance criterion (Tier 1 hits at `worker_0/outputs/final_deliverables/`) only succeeds AFTER Fix #8 enables self-promotion in the inner fixer.
3. **Fix #2** (canonical path resolver migration) — depends on Fix #8 to populate Tier 1.
4. **Fix #4** (derived-state auto-invalidation + per-iter naming) — eliminates wasted LLM cycles. Independent of #1/#8/#2.
5. **Fix #3** (Dual top-level manifest) — provenance improvement, runs after deliverable chain works.
6. **Fix #5** (flat round layout) — improves debuggability.
7. **Fix #6** (eliminate worker_N/ pollution in MFDual) — cleanup.

**Total scope (elegant version, v4.2)**:
- Fix #1: 30 min (snapshot relocation)
- Fix #2: 1 hour (audit + migrate all canonical-path call sites)
- Fix #3: 30 min (Dual top-level manifest)
- Fix #4: 1 hour (derived-state auto-invalidation pattern)
- Fix #5: 1 hour (flat layout for followup rounds)
- Fix #6: 1 hour (eliminate redundant worker_N/ exec slots in MFDual)
- Tests: **3 hours** (19-test suite covering positive/negative/interaction/chain/regression/extensibility/resumability/E2E)

**~8.5 hours total** — investing the additional time to leave the code architecturally consistent AND comprehensively tested.

(v4.3 added: Fix #8 — role-contract inheritance — adds 30 min.)
(v4.5 expanded: 8-test → 19-test suite — adds 1 hour for negative/interaction/regression/E2E coverage.)

---

## §5 Acceptance Criteria

After all fixes:

- [ ] **Fix #1**: `worker_0/children/round_01/review` symlink targets a fresh `review_inferencer/` workspace (NOT `flow_1_initial/`)
- [ ] **Fix #8**: `worker_0/children/fixer_inferencer/outputs/final_deliverables/output.md` exists (47 KB+ from inner fixer) AND `worker_0/outputs/final_deliverables/output.md` exists (surfaced from inner fixer)
- [ ] **Fix #2**: Aggregator's `(See file: ...)` references point to `worker_0/outputs/final_deliverables/output.md` (47 KB Tier 1 hit) — relies on Fix #8
- [ ] **Fix #3**: `task_*/outputs/output_manifest.json` exists with Dual-level provenance (`active_proposer`, `total_iterations`, `proposer_workspace`, `consensus_achieved`)
- [ ] **Fix #4**: For runs where consensus iterates, `flow_X_followup_round_01/` AND `flow_X_followup_round_01_iter2/` exist as siblings (no overwrite)
- [ ] **Fix #5**: Round directories are flat siblings (`flow_X_followup_round_01/`), not nested under `children/round_01/`
- [ ] **Fix #6**: MFDual's workspace does NOT contain redundant `worker_N/` shells (eliminated)
- [ ] No regression in existing 214-test suite
- [ ] All new tests pass (1 per fix = **7 new tests minimum**)

---

## §6 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Fix #1 breaks YAML-configured reviewers (e.g. when YAML sets `review_inferencer: explicit_value`) | Low | Med | Test: verify YAML-configured reviewer still triggers `_reassign_role_workspace` correctly (only loser-as-reviewer assignment should now trigger fresh workspace) |
| Fix #2 breaks tests that assert specific `(See file:)` paths | Low | Low | Update test fixtures; behavior change is the GOAL |
| Fix #3 manifest schema conflicts with downstream consumers | Low | Low | Use additive JSON keys; downstream tools that parse manifest should ignore unknown keys |
| Fix #4 stash reset misses an edge case (e.g. concurrent flows) | Med | Med | Test with mock concurrent flows; document the consensus-iteration boundary |
| Cross-fix interaction (e.g. Fix #1 + Fix #4 both touch reviewer/fixer paths) | Low | Med | Implement in order, run integration test after each |

---

## §7 Test Plan

### v5.0 — 19-Test Suite (14 MFDual + 5 Template-Variable)

**Tests #15-#19 added in v5.0** for Fix #9 (template-variable discovery decoupling):

| # | Test | File | What it verifies |
|---|---|---|---|
| 15 | `test_defaults_auto_load_when_template_variables_empty` | `test/.../test_templated_inferencer_base.py` | Empty `template_variables` → defaults still load (`task_preamble`, `task_instructions`, `task_response_format`) |
| 16 | `test_explicit_overrides_win_over_defaults` | same | `{task_preamble: aggregation}` makes preamble = aggregation variant; `task_instructions` STILL default |
| 17 | `test_partial_overrides_dont_disable_other_defaults` | same | Overriding ONE variable doesn't suppress OTHER defaults |
| 18 | `test_discovery_cached_per_instance` | same | Two `ainfer()` calls trigger ONE filesystem scan |
| 19 | `test_dual_fixer_renders_with_task_instructions` | `test/.../test_dual_inferencer.py` | E2E: mock leaf, run propose→review→fix, verify rendered fixer prompt contains `task_instructions` (Fix #9 acceptance) |

### v4.5 — Original 14-Test Suite (preserved)

Per cross-review feedback, the original 8-test plan was insufficient. Expanded to 14 tests covering:
- Positive cases (the fix works)
- Negative cases (the fix doesn't overreach)
- Interaction cases (multiple fixes don't collide)
- Chain cases (end-to-end deliverable propagation)
- Resumability cases (mutations survive checkpoint reload)
- Regression cases (we don't break normal flows)

| # | Test | File | Fix | What it verifies |
|---|---|---|---|---|
| 1 | `test_reviewer_workspace_isolated_when_loser_assumes_role` | `test_multi_flow_dual_inferencer.py` | #1 | Reviewer gets fresh `review_inferencer/` workspace when loser is assigned |
| 2 | `test_yaml_configured_reviewer_keeps_workspace` | `test_multi_flow_dual_inferencer.py` | #1 | NEGATIVE: YAML-set reviewer NOT reassigned (identity guard correctly identifies "is YAML original") |
| 3 | `test_winner_as_fixer_inherits_output_is_deliverable` | `test_multi_flow_dual_inferencer.py` | #8 | After winner-as-fixer mutation, `fixer.output_is_deliverable == True` |
| 4 | `test_chosen_alias_fixer_inherits_output_is_deliverable` | `test_multi_flow_dual_inferencer.py` | #8 | Same for `chosen` (alias) branch — both mutation paths covered |
| 5 | `test_yaml_configured_fixer_keeps_original_flag` | `test_multi_flow_dual_inferencer.py` | #8 | NEGATIVE: when `fixer_match_winner=False`, original fixer's flag unchanged (no overreach) |
| 6 | `test_fixer_self_promotes_to_final_deliverables_after_fix8` | `test_multi_flow_dual_inferencer.py` | #8 | INTEGRATION at fixer level: with mock LLM, after Fix #8 fires, `fixer_inferencer/outputs/final_deliverables/output.md` exists |
| 7 | `test_deliverable_chain_inner_fixer_to_aggregator` | `test_multi_flow_dual_inferencer.py` | #8 + #2 | **CRITICAL CHAIN TEST**: full chain assertion — inner fixer self-promotes → inner Dual surfaces → MFDual surfaces → aggregator's path = full content (Tier 1 hit) |
| 8 | `test_fix1_and_fix8_compose_correctly` | `test_multi_flow_dual_inferencer.py` | #1 + #8 | INTERACTION: in `_select_reviewer_and_fixer`, both fresh-reviewer-workspace AND fixer-flag-mutation fire correctly together |
| 9 | `test_format_worker_results_uses_canonical_path` | `test_breakdown_then_aggregate_inferencer.py` | #2 | `(See file:)` points to canonical path; Tier 1 preferred over Tier 2 |
| 10 | `test_finalize_response_emits_top_level_manifest` | `test_dual_inferencer.py` | #3 | `output_manifest.json` exists with `active_proposer`, `total_iterations`, `proposer_workspace`, `consensus_achieved` |
| 11 | `test_workspace_property_setter_invalidates_derived_state` | `test_inferencer_base.py` | #4a | After `inf._workspace = new_ws`, `_base_followup_workspace` is cleared |
| 12 | `test_property_setter_preserves_normal_workspace_assignment` | `test_inferencer_base.py` | #4a | **REGRESSION GUARD**: normal `_workspace = ws` (no derived state present) works exactly as before — no breakage in non-consensus paths |
| 13 | `test_subclass_extends_DERIVED_FROM_WORKSPACE` | `test_inferencer_base.py` | #4a | EXTENSIBILITY: `LinearWorkflowInferencer._DERIVED_FROM_WORKSPACE` properly extends base tuple; new derived attrs in subclass are also auto-cleared |
| 14 | `test_round_dir_includes_consensus_iter_suffix` | `test_linear_workflow_inferencer.py` | #4b | When `state["consensus_iteration_id"] > 0`, round dir name includes `_iter{N}` suffix (e.g., `round_01_iter2/`) |
| 15 | `test_consensus_iteration_creates_separate_round_workspace` | `test_linear_workflow_inferencer.py` | #4 (4a+4b) | INTEGRATION: consensus iter 2's fix-phase creates `flow_X_followup_round_01_iter2/`, NOT overwrites `round_01/` |
| 16 | `test_followup_rounds_are_flat_siblings` | `test_linear_workflow_inferencer.py` | #5 | Round dirs are siblings (`flow_X_followup_round_01/`), NOT nested under `children/round_01/` |
| 17 | `test_mfdual_workspace_omits_worker_n_shells` | `test_multi_flow_dual_inferencer.py` | #6 | MFDual workspace lacks redundant `worker_N/` directories |
| 18 | `test_fix8_mutation_reapplies_on_resume` | `test_multi_flow_dual_inferencer.py` | #8 | RESUMABILITY: checkpoint resume RE-CONSTRUCTS inferencers from YAML (not pickle); the test verifies that `_select_reviewer_and_fixer` correctly re-fires during reconstruction, RE-APPLYING the `output_is_deliverable=True` mutation. So mutation persists by re-application, not by serialization. |
| 19 | `test_all_fixes_e2e_integration` | `test_multi_flow_dual_inferencer.py` | All | **AUTOMATED E2E** (replacing manual): mock inferencers, run full propose→review→fix cycle, assert workspace structure healthy at all levels (no `worker_N/`, flat round dirs, manifest exists, deliverable chain populated, role-contract inheritance fired). Replaces manual SOP run as the regression guard. |

**Total: 19 tests** (was 8 in v4.4). Effort: ~3 hours (was 2). Total plan effort: **~8.5 hours** (was 7.5).

### Why This Test Suite Is Sufficient (Honest Assessment)

| Coverage | Without v4.5 | With v4.5 |
|---|---|---|
| Positive case per fix | ✅ | ✅ |
| Negative cases (no overreach) | ❌ | ✅ (#2, #5, #12) |
| Interaction cases | ❌ | ✅ (#7, #8, #15) |
| Chain assertion | ❌ | ✅ (#7) |
| Regression guards | ❌ | ✅ (#12) |
| Extensibility | ❌ | ✅ (#13) |
| Resumability | ❌ | ✅ (#18) |
| Automated E2E | ❌ (manual) | ✅ (#19) |

---

## §8 Resolved Decisions (Elegant-First Principles, Ratified v4.2)

Each decision below adheres to: **architectural correctness > tactical convenience**, **once-and-correct > deferred-and-iterative**, **zero-coupling > explicit-call-discipline**.

1. **(Fix #1) Snapshot YAML-configured value at the TOP of `__attrs_post_init__`** — BEFORE any MFDual override AND BEFORE `super().__attrs_post_init__()`. ✅ Already principled.

2. **(Fix #2) Migrate ALL inferencer-to-inferencer path-resolution call sites to `resolve_canonical_output_path`** — audit every place that resolves another inferencer's output path. The function is the canonical API; bypassing it is what created BTA's bug. Probably 3-5 sites total. NOT just BTA→aggregator.

3. **(Fix #3) Light manifest with separation of concerns** — `active_proposer`, `total_iterations`, `proposer_workspace`, `consensus_achieved`. `round_log.jsonl` and `output_manifest.json` are complementary artifacts; bundling violates SoC. ✅ Already principled.

4. **(Fix #4) Derived-state auto-invalidation pattern + per-consensus-iteration directory naming** — combines TWO sub-fixes:
   - **(4a) Auto-invalidation**: when `_workspace` is reassigned on an inferencer, ALL derived state (including `_base_followup_workspace`) is automatically invalidated via a property setter. ZERO cross-class coupling.
   - **(4b) Per-consensus-iteration naming**: round directory names include consensus iteration ID when > 0 (e.g., `round_01` for propose-phase, `round_01_iter2` for fix-phase consensus iteration 2). Otherwise auto-invalidation alone would re-derive the SAME path and overwrite. This requires `consensus_iteration_id` in state, which Dual must propagate.

   ```python
   # In LWI's per-round directory derivation:
   consensus_iter = state.get("consensus_iteration_id", 0)
   suffix = f"_iter{consensus_iter}" if consensus_iter > 0 else ""
   round_dir = f"round_{step_index:02d}{suffix}"
   round_ws = base_ws.child(round_dir)
   ```

   ```python
   # InferencerBase or LWI mixin
   _DERIVED_FROM_WORKSPACE = ('_base_followup_workspace',)  # extensible

   @property
   def _workspace(self):
       return self.__dict__.get('_workspace_value')

   @_workspace.setter
   def _workspace(self, value):
       if value is not self.__dict__.get('_workspace_value'):
           for attr in self._DERIVED_FROM_WORKSPACE:
               self.__dict__.pop(attr, None)
       self.__dict__['_workspace_value'] = value
   ```

5. **(Fix #5) Flat layout `flow_X_followup_round_NN/` as siblings** — INCLUDE in this PR. Eliminates the misleading `children/round_01/` nesting that suggests round_01 has independent children when it's just one followup invocation. Linearly grep'able + visually clear.

6. **(Fix #6) Eliminate `worker_N/` redundant exec slots in MFDual workspace** — INCLUDE in this PR. MFDual inherits BTA's `worker_N/` directories as side-effect of class hierarchy but never uses them (its actual execution happens in `flow_N_initial/` etc.). Override workspace creation in MFDual to skip the redundant `worker_N/` shells. The dual-namespace pollution disappears.

### Why "Defer #5/#6" Was Wrong

The "surgical fix only the bugs" philosophy creates technical debt. The codebase has now been through 12+ rounds of plans/audits because of accumulated band-aids. ELEGANT means: every fix in this plan should leave the code more orthogonal, more discoverable, and more architecturally consistent than before. Skipping cosmetic fixes that compound onto the bug fixes is false economy.

---

## §10 Provenance

- 2026-05-10 09:37 — v1: initial draft after discovering MFDual self-promotion gap (3-agent investigation)
- 2026-05-10 09:46 — v2: REVISED root cause — discovered `fixer_match_winner: true` mutation hypothesis (later refined)
- 2026-05-10 09:57 — v3: expanded to 5 issues (B-F)
- 2026-05-10 10:02 — v3.1: added Issue G (aggregator silent compensation)
- 2026-05-10 10:05 — v3.2: added Issue H (top-level manifest missing)
- 2026-05-10 10:11 — v3.3: VERIFIED Issue D — both calls said "stop" but LWI ran both anyway
- 2026-05-10 10:30 — **v4.0 INTEGRATED**: replaced root-cause hypotheses with Plan B's verified diagnoses (identity-guard snapshot ordering, workspace stash reuse, Dual generates own manifest, BTA path resolution). Plan B's diagnostic precision exceeds Plan A's; Plan A's structural format (acceptance criteria, risks, ordering) preserved. Old Plan A v3.3 archived for reference.
- 2026-05-10 10:59 — **v4.6 (CRITICAL property-setter bug fix + 2 under-specifications)**: Cross-review caught 3 bugs in v4.5 plan:
  - **CRITICAL**: My §3 Fix #4 property-setter sketch REPLACED the existing setter at `inferencer_base.py:232-237`, dropping `_configure_for_workspace()` and `_propagate_workspace_to_children()`. This would have broken EVERY inferencer's workspace assignment. Corrected to be PURELY ADDITIVE — extends existing setter with auto-invalidation block AFTER existing logic.
  - **Under-specified**: Fix #4b's `consensus_iteration_id` propagation didn't specify HOW Dual injects it into LWI's state. Now specified: pre-injection into per-call `state` dict in `_run_fix_attempt` (not instance attr, not extra_feed). Rationale documented.
  - **Test #18 wrong mechanism**: was testing pickle persistence, but checkpoint resume re-constructs from YAML. Corrected to test that `_select_reviewer_and_fixer` re-fires during reconstruction, re-applying the mutation.

- 2026-05-10 10:56 — **v4.5 (test suite expansion)**: Cross-review agent (correctly) identified 7 testing gaps:
  - No chain test (full deliverable propagation) — ADDED test #7
  - No interaction test for Fix #1 + #8 (both mutate _select_reviewer_and_fixer) — ADDED test #8
  - No regression guard for Fix #4's property setter (touches ALL workspaces, could break codebase) — ADDED test #12
  - No negative test for Fix #8 (overreach to YAML-configured fixers) — ADDED test #5
  - No resumability test (Python attr mutation across checkpoint reload) — ADDED test #18
  - Integration was "manual or scripted" (skips, no regression catch) — REPLACED with automated test #19
  - Single Fix #8 test missed `chosen` alias branch — ADDED test #4
  Plus my own additions:
  - Test #13 (subclass extension of `_DERIVED_FROM_WORKSPACE`) — without this, future derived attrs silently break
  - Test #15 (consensus iteration creates separate workspace) — explicit chain test for Fix #4a + #4b together
  Total: 8 → 19 tests; effort 7.5h → 8.5h.

- 2026-05-10 10:54 — **v4.4 (Fix #8 fully wired in + Fix #4 reconciled)**: Cross-review agent identified 3 gaps + 1 inconsistency, all valid:
  - Gap 1: Added Fix #8 detailed spec to §3 with code snippets, acceptance criterion, test path, and explanation of why it unblocks Fix #2.
  - Gap 2: Updated §4 implementation order to put Fix #8 BEFORE Fix #2 (dependency: Fix #8 populates Tier 1 that Fix #2 expects to find). Also added Fix #5 + #6 to the order list (were missing).
  - Gap 3: Added Fix #8 acceptance criterion + test name to §5 and §7. Also added missing acceptance criteria for #5 + #6 and tests for #4a/#4b/#5/#6 (test count: 4 → 8).
  - Gap 4 (inconsistency): Reconciled §3 Fix #4 spec (was showing Option B "explicit delattr") with §8 Resolved Decisions (auto-invalidation via property setter). §3 now shows the elegant property-setter pattern + Fix #4b per-iteration naming.

- 2026-05-10 10:51 — **v4.3 (REGRESSION CAUGHT — Issue #8 reinstated)**: Cross-review agent (correctly) flagged that Fix #2 alone is INSUFFICIENT. The 47 KB inner-fixer output is stranded 2 layers deep at `fixer_inferencer/outputs/output.md`; Fix #2's canonical resolver returns 2.2 KB summary from MFDual's `outputs/output.md` (Tier 2 hit on WRONG file). Old Plan A v3.3's Issue A (winner-as-fixer mutation loses `output_is_deliverable=True`) was correct and has been REINSTATED as Issue #8. The role-contract-inheritance principle: when an instance assumes a role (fixer/reviewer), it inherits that role's contract (including deliverable-promotion semantics). Implementation: in `_select_reviewer_and_fixer` after `self.fixer_inferencer = winner`, also set `self.fixer_inferencer.output_is_deliverable = True`. This unlocks the deliverable chain so that Fix #2 can find the 47 KB at Tier 1. Effort: +30 min, total ~7.5h.

- 2026-05-10 10:45 — **v4.2 (elegant-no-hack philosophy)**: Re-evaluated all 5 resolved decisions through "architectural correctness > tactical convenience" lens. Upgraded Fix #2 from "BTA-only" to "audit + migrate all call sites to canonical resolver". Upgraded Fix #4 from "method API + explicit call" to "derived-state auto-invalidation via property setter (zero-coupling)". Promoted Fix #5 + #6 from "deferred cosmetic" to "included in this PR" — eliminating technical debt now is cheaper than accumulating more. Effort estimate raised to ~7 hours.

- 2026-05-10 10:44 — **v4.1 (cross-review corrections)**:
  - Fix #1 spec REFINED — snapshot must move to the VERY TOP of `__attrs_post_init__`, BEFORE both MFDual's reviewer_match_second/fixer_match_winner overrides AND DualInferencer's super().__attrs_post_init__() (which also has default-resolution logic). Hard-verified line numbers: 348-355 (reviewer override), 357-364 (fixer placeholder), 367 (super), 382-383 (current snapshot location).
  - Fix #2 — VERIFIED `resolve_canonical_output_path` exists at `inferencer_workspace.py:351` and is already used by `multi_flow_inferencer.py:582-600`. Cross-review agent's "API doesn't exist" was wrong (failed to find it).
  - Fix #4 — VERIFIED inf_instance lifecycle: `self.fixer_inferencer = winner` (line found in `_select_reviewer_and_fixer`) confirms the SAME Python object is reused across propose→fix phases, so `_base_followup_workspace` stash DOES persist (Plan B's diagnosis confirmed).
  - 5 open questions resolved: Q1=YAML value, Q2=BTA-only, Q3=light, Q4=method API on LWI, Q5=defer cosmetic fixes.
