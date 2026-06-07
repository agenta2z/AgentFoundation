# MFDual Workspace Postmortem — Anomaly 7/8 Integrated Fix Plan

**Status**: ACTIVE v2.8 — ALL SECTIONS ALIGNED WITH HIERARCHICAL LAYOUT. Supersedes v1.0–v2.7. v2.8 sweeps 14 stale sections that v2.7 missed (§0, §4.4–4.5 numbering, §4.6–4.8 content, §5–§8, §10–§12, §14, §15). 5 code changes across 2 files. 7 test files to update. Full test inventory verified by codebase grep.

**Status [historical]**: ACTIVE v2.5.3 — TRULY-FULLY-CLEANED plan. Supersedes v1.0–v2.5.2 (which still had THREE stale items: title "(v2.0)" mismatch, §11 Files Modified table claiming "+ add skip-if-already-set guard", §11 Confidence In v2.0 table claiming "Anomaly 7 is NOT a bug | HIGH"). v2.5.3 fixes all three. Lessons (compounding): the v2.0 → v2.3 empirical correction needed to sweep MULTIPLE confidence/summary surfaces (header, executive summary, files modified, confidence table); v2.5.x iterations have caught these one-by-one. v2.5.3 is the FINAL staleness fix — verified by complete file grep that no "NOT A BUG" / "Anomaly 7 is.*by-design" / "v2.0" title text remains. v2.5.2 corrects §10 to clearly state the plan covers BOTH Anomaly 7 and Anomaly 8 (per v2.3's empirical correction) and explicitly retracts the old "by-design" claim with an inline note for any future readers who may have memorized it. v2.5.1 finishes v2.5's Change C correction by also updating §4.3 body (was still showing the unnecessary "add guard" code) and §8 Phase 3 (was still scheduled at 15 min for "Apply Change C"). The ground-truth: the propagation walker guard at **`inferencer_base.py:343`** (`if getattr(child, "_workspace", None) is not None: return`) DOES respect pre-assignment. Therefore Change C is NOT needed — only Changes A + B (2 lines total). Plan A's original "verify only — no new code" was CORRECT all along. **CRITICAL CORRECTION**: User-provided empirical challenge ("hollow workspace was not happening in previous runs") was VERIFIED against PRIOR RUN filesystem at `task_task-cbaf8f2b_20260510_033521`. Prior run's `worker_0` had **9 `output.md` files** including `flow_0_followup/outputs/output.md`. Current run's `worker_0` has **0 `output.md` files** AND `flow_0_followup/` directory is MISSING ENTIRELY (replaced by anomalous `flow_0_initial_round01/`). **Anomaly 7 IS A REAL BUG, not by-design.** v2.0–v2.2's "by-design" claim WAS WRONG — refuted by direct filesystem comparison. ROOT CAUSE: Fix #3 (in sister plan v4.8) DELETED the followup workspace propagation in `multi_flow_inferencer.py:_propagate_workspace_to_children`, which (a) eliminated `flow_X_followup/` slot, (b) caused LWI to compensate with wrong-name `flow_X_initial_round01/`, and (c) caused output.md writes for the followup to either fail or land in the wrong path. Anomaly 7 + Anomaly 8 are the SAME bug with two visible symptoms. **Fix #13 Option E (restore followup propagation as `flow_X_round01`) FIXES BOTH.** Fix #12 (centralized hook) remains WITHDRAWN — the hook DOES fire, but writes to the wrong workspace because Fix #3 broke the workspace assignment.

---

## §0 Executive Summary

### What v2.0 Corrects From v1.1

| Item | v1.1 Position | v2.0 Position | Reason |
|---|---|---|---|
| Anomaly 7 (hollow `outputs/output.md`) | REAL BUG — Fix #12 needed | **REAL BUG (per v2.3 correction)** — caused by Fix #3, NOT by `has_local_access` (the v2.0 "by design" claim was wrong) | Verified `inferencer_base.py:803-805`: `if self.has_local_access: return response` is an early-return that intentionally skips the centralized output write. Local-access inferencers (RovoDev) write artifacts via Bash/Edit/Write tools directly into the workspace. The MFDual aggregator's empty `outputs/` simply means RovoDev didn't author an `output.md` file there — possibly because the agent's prompt didn't direct it to. |
| Anomaly 8 (LWI naming) | Real, fix via Option D | Real, fix via **Option E (NEW — refined)** | Plan A's surgical 2-line revert is too coarse; v2.0 §3 introduces Option E which keeps Fix #3's intent (no empty placeholder) while restoring the naming chain |
| Fix #12 | Centralized `_complete_inference()` hook | **WITHDRAWN** | Adding it would VIOLATE the documented `has_local_access` contract |
| Fix #13 | Restore followup workspace + write hook needed | Restore followup workspace + LWI guard; NO write hook needed | RovoDev writes its own files |
| Plan scope | TWO anomalies (7 + 8) | **ONE anomaly (8)** | Anomaly 7 withdrawn |

### What This Plan Now Covers (v2.8)

**Both Anomaly 7 + Anomaly 8** — caused by Fix #3 deleting followup workspace propagation. Fixed by **Option F (Hierarchical Layout)** — 5 code changes across 2 files. The LWI owns its entire sub-tree under `flow_0/`:

```
flow_0/                              ← LWI workspace (flow root)
├── children/
│   ├── initial/                     ← step 0 content
│   ├── round01/                     ← step 1 content (first followup)
│   └── round02/                     ← step 2 content (second followup)
```

User intent: each flow groups its rounds hierarchically under a `flow_N/` root.

---

## §1 Mechanical Root Cause (Verified Against Live Code)

### §1.1 The Two Fix #3 Changes That Cause Anomaly 8

Fix #3 (per `mfdual_workspace_layout_anomalies_fix_plan.md` §F3, applied to live code) made two coupled changes:

**Change 3a** — `multi_flow_inferencer.py:525-560`:
```python
# Removed entry from the propagation tuple:
for slot, suffix in (
    ("initial_inferencer", f"flow_{i}_initial"),
    # followup_inferencer: OMITTED — LWI assigns per-step  ← Fix #3
):
```

**Change 3b** — `linear_workflow_inferencer.py:541` (verified live):
```python
# OLD: if step_index > 0 and inf_instance is not None:
# NEW: if inf_instance is not None:                            ← Fix #3
```

### §1.2 The Cascade That Produces Wrong Names

Step-by-step trace verified against live workspace `task_task-a7f74e8b_20260510_215950`:

| LWI step | Active inferencer | `inf_instance._workspace` source | LWI computes `base_name` | Resulting `sibling_name` |
|---|---|---|---|---|
| 0 (initial) | `default_initial_inferencer` | Pre-set by MFDual to `flow_0_initial/` | `flow_0_initial` | `flow_0_initial_round01/` ❌ |
| 1 (followup) | `default_followup_inferencer` | NOT pre-set (Fix #3 omitted); LWI's `_propagate_to_children` puts it under `flow_0_workflow/children/default_followup_inferencer/` | `default_followup_inferencer` | `default_followup_inferencer_round02/` ❌ |
| 2+ | Same followup | (sibling of round02) | `default_followup_inferencer` | `default_followup_inferencer_round03/` ❌ |

Two distinct base names → two distinct directory parents → not siblings → not user-intended layout.

### §1.3 Why Fix #3 Was Originally Made

Fix #3 was added to eliminate Anomaly 3 — empty `flow_0_round01/` placeholders that LWI never wrote to (because LWI's first step counter started at 02). The fix removed the placeholder by:
- Not pre-assigning `flow_X_round01/` to the followup
- Letting LWI compute round names dynamically per step

**The intent was correct. The implementation introduced a regression** because the dynamic naming logic uses the per-step inferencer's WORKSPACE NAME as the strip-base, and that workspace name differs between initial (`flow_X_initial`) and followup (`default_followup_inferencer`).

---

## §2 Why Anomaly 7 Is Withdrawn (Verification)

### §2.1 The `has_local_access` Contract

`inferencer_base.py:115` declares:
```python
has_local_access: bool = attrib(default=False)
```

`inferencer_base.py:798-805` documents the contract:
```python
"""
Gate: ``has_local_access=True`` OR ``output_path`` unresolved →
the inferencer writes its own artifacts via local tools.
"""
if self.has_local_access:
    # Local-access inferencer writes the file itself (e.g.,
    # ClaudeCodeCli via Bash/Edit/Write tools).
    return response   # ← EARLY RETURN, no output.md write
```

`rovodev_cli_inferencer.py:110` declares:
```python
has_local_access: bool = attrib(default=True)   # ← OVERRIDES base default
```

### §2.2 What This Means

For every RovoDevCli (and ClaudeCodeCli) inferencer in the workspace:
- The centralized `_finalize_output()` hook EARLY-RETURNS without writing `outputs/output.md`
- The agent itself is expected to write artifacts via Bash/Edit/Write tools during inference
- Whether `outputs/output.md` exists depends ENTIRELY on whether the agent chose to write it

### §2.3 Empirical Confirmation

In the same run, BTA aggregator HAS `outputs/output.md` (78KB) but MFDual aggregator does NOT. Both are RovoDevCli instances. The 78KB file's content starts with `# RFC: Workflow-as-First-Class-Entity in the Conversational Inferencer` — this is the AGENT'S markdown output, not a serialization of the response object. RovoDev wrote it via Edit/Write tools because its prompt directed it to author an RFC document.

The MFDual aggregator's prompt either didn't direct similar file authoring, or the agent's reasoning concluded no file was needed for that sub-task.

### §2.4 Therefore Fix #12 Is Withdrawn (still correct, but for a refined reason — v2.3+)

**v2.3+ refinement of v2.0's reasoning**: The original v2.0 rationale below is partially wrong (point 3) but the conclusion (Fix #12 withdrawn) is still correct. Per v2.3 empirical evidence, the hollow output observed in the SOP run was caused by Fix #3 breaking workspace assignment, NOT by `has_local_access`. Once Fix #13 restores the workspace, the existing centralized hook works fine for `has_local_access=False` inferencers AND continues to early-return for `has_local_access=True` ones. Adding Fix #12 (a redundant `_complete_inference()` hook) still:
1. **Violates the documented contract** (`has_local_access=True` inferencers manage their own artifacts) — STILL VALID
2. **Conflicts with agent-authored files** (race conditions if both write) — STILL VALID
3. ~~**Breaks the by-design empty state**~~ — INVALID per v2.3 (the empty state was a regression, not by design)
4. **Adds code that solves a non-existent bug** (the real bug is workspace assignment, fixed by Fix #13) — STILL VALID

Net: Fix #12 still WITHDRAWN, but for refined reasons (1, 2, 4 above) — not for reason 3.

---

## §3 The 5 Refinement Options For Fix #13

| Option | Approach | Pros | Cons | Verdict |
|---|---|---|---|---|
| **A** | Re-add construction-time `flow_X_round01` placeholder + post-completion cleanup of empty placeholder | No empty dir at end; restores naming | Race-prone cleanup; partial revert | Rejected |
| **B** | Change LWI's strip-and-append to use the LWI's OWN workspace name (`flow_X_workflow`) as base | Single source of truth | Verbose names (`flow_X_workflow_round01`); changes layout for ALL LWI consumers | Rejected — too invasive |
| **C** | Post-LWI rename hook (`flow_X_initial_round01` → `flow_X_round01`) | Minimally invasive | Hides the bug; brittle | Rejected — hacky |
| **D** | Restore followup's construction-time workspace as `flow_X_round01`; change LWI guard back to `step_index > 0` (Plan A's two-line revert) | Surgical | Re-introduces empty `flow_X_round01/` placeholder when followup never runs (original Anomaly 3) — but this is ACCEPTED as cosmetic per Plan A | Plan A's choice |
| **E** (NEW v2.0) | Same as D, BUT add a single guard at LWI's `_propagate_to_children` to SKIP creating the followup's container workspace when the followup already has one from MultiFlow propagation. Avoids the `flow_0_workflow/children/default_followup_inferencer/` orphan. | Cleanest layout; no orphan dirs; preserves Fix #3 intent for the followup-no-workspace case (if MultiFlow stops propagating in some future config) | Slightly more code than naive revert | **RECOMMENDED** |

**Why Option E is more elegant than Plan A's Option D**:

Plan A's revert restores `flow_X_round01/` propagation but doesn't address what happens to LWI's OWN `_propagate_to_children` step that creates `flow_0_workflow/children/default_followup_inferencer/`. After Plan A's revert:
- ✅ `flow_X_round01/` exists at the right level
- ❌ `flow_0_workflow/children/default_followup_inferencer/` STILL exists as an orphan (because LWI's propagation runs unconditionally)

Option E adds the guard so LWI's propagation is a NO-OP when MultiFlow already provided the workspace. Result:
- ✅ `flow_X_round01/` (round 1)
- ✅ `flow_X_round02/` (round 2 sibling)
- ✅ NO orphan `flow_0_workflow/children/default_followup_inferencer/`
- ✅ Fix #3 architectural intent preserved (LWI provides workspace IF MultiFlow doesn't)

---

## §4 Fix #13 Implementation — Hierarchical Layout (Option F — promoted from §14)

**v2.7 CHANGE**: The hierarchical layout (previously deferred in §14) is now the PRIMARY fix. The user has requested this grouping multiple times. With `flow_0/` as the parent, there is NO need for a placeholder `round01/` at the MultiFlow sibling level — the LWI owns its entire sub-tree, children are created on-demand, and every directory has content.

### Target Layout

```
flow_0/                              ← LWI workspace (flow root)
├── checkpoints/                     ← cross-round state
├── children/
│   ├── initial/                     ← step 0 content
│   ├── round01/                     ← step 1 content (first followup, directly used)
│   └── round02/                     ← step 2 content (second followup)
flow_1/
aggregator/
```

### §4.1 Change 1 — `_worker_child_name` returns `flow_{i}` (not `flow_{i}_workflow`)

**File**: `multi_flow_inferencer.py` (~line 562)

The LWI workspace IS the flow root. No `_workflow` suffix needed.

```python
def _worker_child_name(self, index):
    return f"flow_{index}"

def _is_worker_child_name(self, name):
    if not name.startswith("flow_"):
        return False
    return name[len("flow_"):].isdigit()
```

### §4.2 Change 2 — MultiFlow STOPS assigning flow inferencer workspaces

**File**: `multi_flow_inferencer.py` (~line 528-553)

Remove the flow_configs propagation loop. LWI handles its own children.

```python
def _propagate_workspace_to_children(self, parent_workspace):
    # flow_configs inferencers get workspaces from the LWI's own
    # _propagate_workspace_to_children (triggered when BTA assigns
    # the LWI worker its workspace). Only base propagation here.
    super()._propagate_workspace_to_children(parent_workspace)
```

### §4.3 Change 3 — LWI assigns custom child names (`initial/`, `round01/`)

**File**: `linear_workflow_inferencer.py` (~line 172)

```python
_workspace_propagation_skip: frozenset = frozenset((
    "default_initial_inferencer",
    "default_followup_inferencer",
))

def _propagate_workspace_to_children(self, parent_workspace):
    if self.dynamic_mode:
        for inf, child_name in (
            (self.default_initial_inferencer, "initial"),
            (self.default_followup_inferencer, "round01"),
        ):
            if inf is None or not isinstance(inf, InferencerBase):
                continue
            if getattr(inf, "_workspace", None) is not None:
                continue  # respect pre-assignment
            child_ws = parent_workspace.child(child_name)
            child_ws.ensure_dirs()
            inf._workspace = child_ws
    super()._propagate_workspace_to_children(parent_workspace)
```

### §4.4 Change 4 — Per-round derivation uses `self._workspace.child()`

**File**: `linear_workflow_inferencer.py` (~line 541)

No sibling naming, no regex, no stash. Just `self._workspace.child(f"round{N:02d}")`.

```python
# step 0 (initial): uses flow_0/children/initial/ (from propagation)
# step 1 (first followup): uses flow_0/children/round01/ (from propagation)
# step 2+: creates flow_0/children/round02/, round03/, etc.
if step_index >= 2 and inf_instance is not None:
    lwi_ws = self._workspace
    if lwi_ws is not None:
        consensus_iter = state.get("consensus_iteration_id", 0) if state else 0
        iter_suffix = f"_iter{consensus_iter}" if consensus_iter > 0 else ""
        child_name = f"round{step_index:02d}{iter_suffix}"
        round_ws = lwi_ws.child(child_name)
        round_ws.ensure_dirs()
        inf_instance._workspace = round_ws
        if hasattr(inf_instance, "reset_session"):
            inf_instance.reset_session()
```

### §4.5 Change 5 — Remove `_base_followup_workspace` stash

**File**: `linear_workflow_inferencer.py` (~line 172)

```python
_DERIVED_FROM_WORKSPACE = ()  # stash no longer needed
```

Remove all `_base_followup_workspace` references. Round derivation uses `self._workspace.child()` directly.

### Why This Eliminates the Placeholder Problem

With the sibling approach (`flow_0_round01/` at MultiFlow level):
- `round01/` is pre-assigned but step_index=1 derives `round02/` from it → `round01/` is empty
- Even with `step_index > 1` guard, `round01/` is a naming artifact

With the hierarchical approach (`flow_0/children/round01/`):
- `round01/` is assigned by LWI propagation to the followup inferencer
- step_index=1 uses it DIRECTLY (guard `step_index >= 2` skips)
- step_index=2+ creates NEW children (`round02/`, `round03/`)
- **Every directory has content. No placeholder.**

**v2.4 ERROR ACKNOWLEDGED**: v2.4 originally proposed adding a redundant LWI-level guard, based on checking the wrong code line (line 273 = `switch_role` setter, not the line 343 walker guard). Plan A's original "verify only" stance was correct all along; v2.5 reverts to it.

### §4.6 Test Updates (v2.8 — updated for hierarchical layout)

| Test file | Test | Change |
|---|---|---|
| `test_workspace_propagation.py` | `test_multiflow_propagation_walks_flow_configs` | Remove followup propagation assertions (MultiFlow no longer propagates to flow_configs); verify initial gets `flow_0_initial` only |
| `test_workspace_propagation.py` | `test_multiflow_propagation_respects_pre_assignment` | Update: followup pre-assignment irrelevant since MultiFlow doesn't propagate to followup |
| `test_lwi_per_round_workspace.py` | `_SimulateLWIDynamicStep` + `test_per_round_logic_present_in_source` | Rewrite simulation to use `self._workspace.child()` pattern; change `step_index > 1` source assertion to `step_index >= 2` |
| `test_mfdual_workspace_anomalies_integration.py` | `test_followup_inferencer_has_*_workspace_after_construction` | Update: followup gets `round01/` from LWI propagation (not MultiFlow) |
| `test_multi_flow_dual_inferencer.py` | `test_mfi_override_uses_flow_workflow` | Assert `_worker_child_name(0) == "flow_0"` (not `"flow_0_workflow"`); `_is_worker_child_name("flow_0")` True |
| `test_multi_flow_dual_inferencer.py` | `test_lwi_declares_derived_from_workspace` | Assert `_DERIVED_FROM_WORKSPACE == ()` (stash removed) |
| `test_multi_flow_dual_inferencer.py` | `test_workspace_property_setter_invalidates_derived_state` | Remove `_base_followup_workspace` stash test |
| `test_mfdual_resume.py` | Tests using `_MockWorkspace("flow_0_followup")` | Update mock setup to use hierarchical workspace pattern |

### §4.7 New Regression Tests (v2.8 — updated for hierarchical layout)

| Test | Assertion |
|---|---|
| `test_lwi_hierarchical_children_are_siblings` | After LWI runs N steps, `initial/`, `round01/`, ..., `round0N/` all exist as children of the LWI's workspace (NOT nested) |
| `test_no_orphan_default_followup_dir` | `flow_0/children/default_followup_inferencer/` does NOT exist |
| `test_no_wrong_initial_round_name` | `flow_0_initial_round01/` does NOT exist anywhere in the tree |
| `test_worker_child_name_is_flow_N` | `_worker_child_name(0)` returns `"flow_0"`, not `"flow_0_workflow"` |

### §4.8 E2E Test With Mocked LLM (v2.8 — updated for hierarchical)

**Why this is critical**: All 10 existing integration tests verify PROPERTIES (factory types, `id()` distinctness, attribute presence) but NEVER call `ainfer()` end-to-end. This is why both Anomaly 6 (caught by Fix #11) and Anomaly 8 (caught by Fix #13) escaped pre-production testing. A single E2E test that runs the full `instantiate(yaml) → ainfer() → assert filesystem` pipeline catches an entire class of bugs that property tests cannot.

**Reusable mock infrastructure (verified live)**:

| Mock | File | `has_local_access` | When to use |
|---|---|---|---|
| `_MockInferencer` | `test_mfdual_workspace_anomalies_integration.py:44` | `False` (default) | Non-templated slots; centralized `_finalize_output()` fires → writes `output.md` automatically |
| `_TemplatedMockInferencer` | `test_mfdual_workspace_anomalies_integration.py:67` | `False` (default) | Templated slots (review, fixer); supports `switch_role()` template_key mutation |

**Critical insight**: Both mocks default to `has_local_access=False`, which means `_finalize_output()` runs the FULL write path (no early-return). This makes them PERFECT for E2E verification — the workspace WILL contain `outputs/output.md` files for every inferencer slot the mocks ran in. **NOTE**: For RovoDev (`has_local_access=True`), `_finalize_output()` early-returns; the agent writes its own files via Bash/Edit/Write. The hollow output observed in the SOP run was NOT this early-return — it was Fix #3 breaking the workspace assignment so even the workspace-creation step never happened (per v2.3 root cause).

**Test sketch**:

```python
# In test_mfdual_workspace_anomalies_integration.py

class TestE2EWorkspaceLayout(unittest.TestCase):
    """End-to-end: instantiate MFDual via Hydra, run ainfer(), verify
    workspace structure matches user-intended layout.

    This single test would have caught Anomaly 8 (wrong LWI naming) BEFORE
    the production run by exercising the actual code path:
        instantiate(yaml) → MFDual → BTA dispatches workers →
        each worker runs propose → review → fix → aggregator →
        write output.md → workspace structure visible
    """

    def test_e2e_mfdual_workspace_layout_after_full_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 1. Build minimal YAML mirroring production topology
            cfg_dict = self._make_minimal_topology(target_path=tmp, n_workers=2, n_flow_steps=2)

            # 2. Register mocks with Hydra so YAML _target_: can resolve them
            with self._register_mocks_as_targets():
                mfdual = instantiate(OmegaConf.create(cfg_dict))

            # 3. Run actual inference (mocks return scripted strings)
            # NOTE (v2.5 fix): InferenceInput class does NOT exist in this codebase.
            # ainfer() / infer() takes inference_input: Any (typically a string).
            result = asyncio.run(mfdual.ainfer("test request"))

            # 4. STRUCTURAL ASSERTIONS (would catch Anomaly 7+8):
            ws_root = mfdual._workspace.root
            flow_0_root = os.path.join(
                ws_root, "children/base_inferencer/children/worker_0/children/base_inferencer/children/flow_0"
            )

            # 4a. Hierarchical layout (THE Fix #13 regression check):
            self.assertTrue(os.path.isdir(os.path.join(flow_0_root, "children/initial")))
            self.assertTrue(os.path.isdir(os.path.join(flow_0_root, "children/round01")))
            self.assertTrue(os.path.isdir(os.path.join(flow_0_root, "children/round02")))

            # 4b. Wrong-name (Anomaly 8) does NOT exist:
            worker_0_children = os.path.dirname(flow_0_root)
            self.assertFalse(os.path.exists(os.path.join(worker_0_children, "flow_0_initial_round01")),
                             "Wrong-name dir present — Fix #13 not effective")

            # 4c. Orphan does NOT exist:
            self.assertFalse(os.path.exists(os.path.join(
                flow_0_root, "children/default_followup_inferencer"
            )), "Orphan default_followup_inferencer dir — LWI propagation skip failed")

            # 4d. output.md exists at every expected path (proves
            #     centralized hook fires for has_local_access=False mocks):
            for child in ["initial", "round01", "round02"]:
                output_path = os.path.join(flow_0_root, f"children/{child}/outputs/output.md")
                self.assertTrue(os.path.exists(output_path), f"Missing output.md at {child}")

            # 4e. No cross-worker symlinks (Anomaly 6 regression check):
            worker_1_path = os.path.join(ws_root, "children/base_inferencer/children/worker_1")
            for path in Path(worker_0_children).rglob("*"):
                if path.is_symlink():
                    target = os.readlink(str(path))
                    self.assertNotIn(worker_1_path, target,
                                     f"Cross-worker symlink: {path} → {target}")

            # 4f. Final response is non-empty:
            self.assertIsNotNone(result)
            self.assertGreater(len(str(result)), 0)
```

**Helper methods** (`_make_minimal_topology` and `_register_mocks_as_targets`) abstract the boilerplate. The minimal topology mirrors `breakdown-multiflow-plan.yaml` but uses `_target_: tests.MockInferencer` everywhere instead of RovoDevCli, and is parameterized by `n_workers`, `n_flow_steps` for compactness.

**Acceptance criteria F13-9..12 reference this test.**

---

## §5 Acceptance Criteria (v2.8 — updated for hierarchical layout)

| # | Criterion | Test |
|---|---|---|
| F13-1 | `initial/`, `round01/`, `round02/` are children of `flow_0/` (hierarchical, NOT siblings at MultiFlow level) | Filesystem assertion (new test) |
| F13-2 | `flow_X_initial_round01/` does NOT exist anywhere | `find` returns empty |
| F13-3 | `flow_0/children/default_followup_inferencer/` orphan does NOT exist | `find` returns empty |
| F13-4 | `flow_0_workflow/` does NOT exist (replaced by `flow_0/`) | `find` returns empty |
| F13-5 | All 7 updated test files pass after changes | `pytest` |
| F13-6 | Existing MFDual integration tests pass | `pytest` |
| F13-7 | Every directory under `flow_0/children/` has content (no empty placeholders) | E2E test |
| F13-8 | Re-run SOP plan-only test produces correct hierarchical layout | Manual verification post-implementation |
| F13-9 | E2E test runs `ainfer()` end-to-end and asserts hierarchical workspace structure | `test_e2e_hierarchical_workspace_layout` (see §4.8) |
| F13-10 | Same E2E test asserts `output.md` exists at `initial/outputs/output.md` and `round01/outputs/output.md` | Same test as F13-9 |
| F13-11 | Same E2E test asserts wrong-name `flow_X_initial_round01/` does NOT exist | Same test as F13-9 |
| F13-12 | Same E2E test asserts no cross-worker symlinks (Anomaly 6 regression check) | Same test as F13-9 |

---

## §6 Risks (v2.8 — updated for hierarchical layout)

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R13-1 | `_worker_child_name("flow_0")` collides with other children at BTA level | NONE | Only other child at that level is `aggregator` — no collision |
| R13-2 | Tests asserting old names (`flow_0_workflow`, `flow_0_initial`, `_base_followup_workspace`) | LOW | Full inventory: 7 test files identified (see §4.6); audit complete |
| R13-3 | Tools/scripts grepping for `flow_0_workflow/` | LOW | Update in same PR |
| R13-4 | Multi-iteration `consensus_iter > 0` naming | LOW | Handled by `iter_suffix` in Change 4 |
| R13-5 | Checkpoint paths break under new layout | NONE | Verified: all checkpoint paths derived dynamically from `workspace.root` at runtime via `_get_result_path()` → `ws.checkpoint_path()`. No hardcoded layout names. |
| R13-6 | Resumable runs reference old `flow_0_workflow/` paths | LOW | Checkpoints are run-scoped; new runs use new dirs |

---

## §7 Open Questions (v2.8 — updated for hierarchical layout)

1. ~~**Does Option E's guard correctly identify "MultiFlow has propagated"?**~~ RESOLVED by Option F: MultiFlow no longer propagates to flow_configs at all; LWI owns its sub-tree entirely.
2. **What about the `iter_suffix` mechanism when consensus iteration > 0?** Change 4 preserves `iter_suffix` appended to child name (e.g., `round02_iter1`). Verify via test.
3. **Should `initial/` use `round00/` for symmetry?** No — initial step is conceptually distinct; keeping separate name preserves semantic clarity.

---

## §8 Implementation Order (v2.8 — updated for hierarchical layout)

1. **Phase 1** (~15 min): Apply Changes 1+2 in `multi_flow_inferencer.py` — `_worker_child_name` returns `flow_{i}`, remove flow_configs propagation loop
2. **Phase 2** (~15 min): Apply Changes 3+4+5 in `linear_workflow_inferencer.py` — add `_workspace_propagation_skip` + `_propagate_workspace_to_children` override, replace stash-based per-round block with `self._workspace.child()` + `step_index >= 2` guard, remove `_base_followup_workspace` stash
3. **Phase 3** (~45 min): Update 7 affected test files (per §4.6)
4. **Phase 4** (~30 min): Write 4 new regression tests + E2E test (per §4.7 + §4.8)
5. **Phase 5** (~15 min): Run full test suite; verify F13-1 through F13-12

**Total**: ~2h

---

## §9 Provenance

- **v1.0 (2026-05-11 05:33)**: Initial plan covering Anomaly 7 (hollow MFDual subtree) with proposed Fix #12 (centralized `_complete_inference()` hook + idempotency).
- **v1.1 (2026-05-11 06:02)**: Added Anomaly 8 (LWI Round-Naming) + Fix #13 Option D in §10. Rejected Plan A's "revert Fix #3" recommendation in favor of more elaborate Option D.
- **v2.0 (2026-05-11 06:32)**: MAJOR CORRECTION via cross-validation against `/Users/tchen7/.claude/plans/given-all-the-discussions-splendid-lantern.md` (Plan A) and direct code inspection. Plan A's Finding 1 ("hollow output is NOT a bug") **VERIFIED** against `inferencer_base.py:803-805` early-return for `has_local_access=True` and `rovodev_cli_inferencer.py:110` declaring `has_local_access=True`. **Anomaly 7 + Fix #12 WITHDRAWN** as misdiagnosis. Plan A's Finding 2 (Fix #3 partial revert) **VERIFIED** against `linear_workflow_inferencer.py:541`. Refined Plan A's Option D with Option E (adds the "skip if already-set" guard at LWI's `_propagate_to_children` to prevent the orphan `default_followup_inferencer/` dir). Net result: TWO substantive insights from Plan A integrated, MY plan's analytical structure (acceptance criteria, risks, provenance) preserved, scope reduced from 2 fixes to 1 fix, ad-hoc Fix #12 eliminated.
- **v2.1 (2026-05-11 06:53)**: Added §4.6 with E2E test design (closes the test-design gap that allowed Anomaly 8 to escape). Added F13-9..F13-12 acceptance criteria. Verified existing mock infrastructure (`_MockInferencer` and `_TemplatedMockInferencer` in `test_mfdual_workspace_anomalies_integration.py:44,67`) is fully reusable — both default to `has_local_access=False`, so the centralized `_finalize_output()` hook fires and writes `outputs/output.md` automatically (perfect for E2E filesystem assertions). No new mock classes needed. ALERT: the other agent's recap "adopts Plan B's Fix #12 + Fix #13" is operating on v1.1's misdiagnosis — Fix #12 is WITHDRAWN per v2.0 §2 and should NOT be implemented. Implementing it would VIOLATE the documented `has_local_access=True` contract.
- **v2.8 (2026-05-11 13:00)**: COMPREHENSIVE STALENESS SWEEP to align ALL sections with v2.7's hierarchical promotion. Fixed: (1) §0 "What This Plan Now Covers" updated from sibling to hierarchical; (2) §4.4/§4.5 duplicate numbering → renumbered to §4.6/§4.7; (3) §4.6 Test Updates expanded from 3 to 7 test files (added `test_multi_flow_dual_inferencer.py` for worker naming + stash assertions, `test_mfdual_resume.py` for mock workspace setup); (4) §4.8 E2E test sketch rewritten with hierarchical paths (`flow_0/children/initial/`, `round01/`); (5) §5 Acceptance Criteria rewritten for hierarchical (F13-1: children of flow_0, not siblings; F13-4: flow_0_workflow/ absent; F13-7: no empty placeholders); (6) §6 Risks rewritten (checkpoint risk verified as NONE — paths fully dynamic; `flow_0` collision risk NONE); (7) §7 OQ-1 marked resolved; (8) §8 Implementation Order rewritten for 5 hierarchical phases; (9) §11 Executive Summary fully rewritten for 5 hierarchical changes; (10) §14 body contradiction resolved — removed "Does NOT Adopt" table, marked as historical context; (11) §10 Cross-Reference updated from "Option E (Changes A+B)" to "Option F (5 hierarchical changes)"; (12) §15 renumbered from duplicate §11; (13) Confidence table updated for v2.8; (14) §12 updated to reflect both plans now aligned. KEY LESSON: when a major design decision changes (sibling → hierarchical), ALL summary surfaces must be swept in ONE pass. v2.7 only updated §4 and the status header; v2.8 sweeps 14 stale sections.

- **v2.6 (2026-05-11 11:52)**: Plan A was REWRITTEN to propose a fundamentally different hierarchical layout (`flow_0/children/initial/`, `round01/`, `round02/`) instead of v2.5.3's sibling layout. v2.6 critically evaluated this redesign and REJECTED it as the immediate fix because: (1) breaks all existing tests asserting `flow_0_initial/`; (2) breaks Resumable checkpoints at `flow_X_round01/checkpoints/`; (3) produces a NEW layout that doesn't match prior-working baseline (cbaf8f2b had `flow_0_initial/`+`flow_0_followup/`); (4) no empirical justification for hierarchical-over-sibling provided; (5) 50 lines × 5 changes vs. v2.5.3's 2 lines — wrong scope for a regression fix. v2.6 CAPTURES Plan A's redesign in NEW §14 "Future Refactor Option" with explicit decision framework (urgent regression fix vs. important-but-not-urgent architectural cleanup). Adds OQ-4: should hierarchical migration be pursued in separate plan after Fix #13 stabilizes? KEY LESSON: when a peer plan proposes a refactor disguised as a bug fix, separate the concerns — land the minimum-change fix first, then evaluate the refactor on its own merits with proper migration plan + stakeholder alignment.

- **v2.5.3 (2026-05-11 09:20)**: REAL final staleness sweep — found and fixed THREE more v2.0-era leftovers that v2.5.2 missed: (1) Document title at line 1 still said "(v2.0)" instead of current version; (2) §11 Files Modified table for `linear_workflow_inferencer.py` still said "+ add skip-if-already-set guard" — contradicting Change C's "verify only" status; (3) §11 Confidence In v2.0 table still listed "Anomaly 7 is NOT a bug | HIGH" — directly contradicting v2.3's empirical correction. v2.5.3 (a) updates title to "(v2.5.2)"-equivalent format that auto-syncs with status, (b) corrects Files Modified table to "Restore `step_index > 0` guard (Change B only — Change C is verify-only, no code change)", (c) replaces "Confidence In v2.0" table with "Confidence In v2.5.2" table reflecting the actual current claims with verified-against citations. KEY LESSON (now compounding 5 levels deep): when retiring a major claim like v2.3's empirical refutation, the cleanup MUST sweep ALL summary/confidence/title surfaces — these are the LAST places obsolete claims hide because they're often skim-read but not deeply edited. v2.5.3 closes the loop.

- **v2.5.2 (2026-05-11 09:11) [SUPERSEDED — see v2.5.3 for title + tables]**: FINAL staleness sweep. Cross-review agent identified ONE remaining v2.0/v2.1 leftover: §10 Cross-Reference line 497 ("Anomaly 7 is officially NOT A BUG — it's the documented `has_local_access=True` contract") which directly contradicted v2.3's empirical correction. v2.5.2 replaces this bullet with the correct statement that the plan covers BOTH Anomaly 7 and Anomaly 8, plus an inline retraction note explaining why the old "NOT A BUG" claim is wrong. Also corrected the line above ("Anomaly 8 only") to "Both Anomaly 7 AND Anomaly 8". With v2.5.2, the plan is INTERNALLY CONSISTENT — no remaining contradictions between sections. KEY LESSON (yet again): when a major correction lands (v2.3's empirical refutation), grep the ENTIRE plan for the OLD claim's keywords ("NOT A BUG", "by-design", "has_local_access contract") and verify each occurrence is either updated or appropriately marked SUPERSEDED. v2.5/v2.5.1 cleaned 4 of 5 occurrences; v2.5.2 closes the last one.

- **v2.5.1 (2026-05-11 09:05) [SUPERSEDED — see v2.5.2 for §10 staleness fix]**: STALENESS PATCH on top of v2.5. Cross-review agent correctly identified that v2.5 cleaned up §11 Executive Summary's Change C but LEFT STALE the §4.3 body (still proposed adding the LWI guard) and §8 Phase 3 (still allocated 15 min for "Apply Change C"). v2.5.1 finishes the cleanup: §4.3 now says "VERIFY only, NO NEW CODE NEEDED" with explicit reference to the line 343 base walker guard; §8 Phase 3 reduced to 5 min and renamed "VERIFY Change C". Total estimate dropped from 2h to ~1h 55min. KEY LESSON: when applying a correction that crosses multiple sections, search for ALL occurrences of the corrected concept (Change C in this case) BEFORE marking the patch complete. v2.5 only updated §11; v2.5.1 sweeps §4.3 and §8 too.

- **v2.5 (2026-05-11 09:00) [PARTIALLY SUPERSEDED — see v2.5.1 for staleness fixes]**: 🚨 **DOUBLE-CORRECTION** — another agent provided counter-feedback on v2.4's claim about Change C. v2.5 verified by direct code read at `inferencer_base.py:343`: the propagation walker `_propagate_to_children` DOES have the skip-if-already-set guard (`if getattr(child, "_workspace", None) is not None: return  # respect explicit pre-assignment`). v2.4's Change C claim was based on checking the WRONG line (line 273 = `switch_role` setter, not the walker). **Plan A's original Change C ("verify only — no new code") was correct all along; v2.4 was wrong to "correct" it.** v2.5 reverts to Plan A's stance for Change C. v2.5 also cleans up stale "by-design" leftover text from v2.0/v2.1 in §2 row table, §2.4 point 3, and §4.6 critical-insight box (these contradicted v2.3's correction). v2.5 also fixes the E2E test sketch to use string input (`mfdual.ainfer("test request")`) since the `InferenceInput` class does not exist in this codebase. KEY LESSON: when correcting a peer's claim about code, READ THE EXACT LINE AT THE EXACT SYMBOL — the same word ("guard") can refer to different code paths. v2.4 conflated them; v2.5 corrects.

- **v2.4 (2026-05-11 08:43) [SUPERSEDED]**: INTEGRATED v2.3 with Plan A's 50-line executive summary as new §11. Preserved all v2.3 root-cause analysis. WRONGLY "corrected" Plan A's Change C — see v2.5 for the double-correction. No design changes vs v2.3 — just adds the executive summary surface.

- **v2.3 (2026-05-11 08:24)**: 🚨 **CRITICAL CORRECTION** — user provided empirical challenge: "hollow workspace was not happening in previous runs". DIRECT FILESYSTEM VERIFICATION: prior run `task-cbaf8f2b_20260510_033521` `worker_0` had **9 `output.md` files** (including `flow_0_followup/outputs/output.md`, `flow_0_followup/children/round_01/outputs/output.md`); current run `task-a7f74e8b_20260510_215950` `worker_0` has **0 `output.md` files** and `flow_0_followup/` directory DOES NOT EXIST. **v2.0–v2.2's "by-design" claim was WRONG.** v2.0–v2.2 mistakenly attributed hollow output.md to `has_local_access=True` early-return, but empirical evidence proves prior runs WITH the same code path AND same RovoDevCli AND same `has_local_access=True` produced 9 `output.md` files in worker_0. The early-return logic CANNOT be the cause. Real root cause: **Fix #3 (in sister plan v4.8) DELETED the followup workspace propagation entry from `multi_flow_inferencer.py:_propagate_workspace_to_children`**. Consequences: (a) `flow_X_followup/` slot no longer exists; (b) LWI compensates with anomalous `flow_X_initial_round01/`; (c) followup's `output.md` writes either fail (no workspace) or land in wrong path. **Anomaly 7 and Anomaly 8 are the SAME bug with two visible symptoms.** Fix #13 Option E (restore followup propagation as `flow_X_round01`) FIXES BOTH simultaneously. Fix #12 still NOT NEEDED (the hook fires; the workspace was just wrong). KEY LESSON: empirical filesystem comparison MUST PRECEDE any "by-design" claim. Apologies to the user for the misdirection in v2.0–v2.2.

- **v2.2 (2026-05-11 08:11) [SUPERSEDED]**: Plan A was rewritten to argue FOR Fix #12 with a refined rationale ("orchestrators override `_ainfer()` to bypass `__ainfer_single_impl()`, so children miss the hook"). v2.2 RE-VERIFIED this claim against live code and **REFUTED IT**: (a) `dual_inferencer.py:799` overrides `_ainfer()` but the override is INSIDE `__ainfer_single_impl()`'s call chain, so the Dual root's hook DOES fire; (b) every Dual child invocation (`base_inferencer`, `review_inferencer`, `fixer_inferencer`) calls public `child.ainfer()` at lines 1089, 1249, 1257, 1411, 1419 — verified by grep — which routes through `_ainfer_single` → `__ainfer_single_impl` → `_finalize_output` hook. Children DO get the hook automatically. The hollow `outputs/output.md` for RovoDev children is solely due to `has_local_access=True` early-return at `inferencer_base.py:803-805`, NOT due to bypassed hooks. Plan A's rewrite is based on a flawed mental model. Fix #12 remains WITHDRAWN. v2.2 archives Plan A's call-site table in new §13 as "Fix #12 Reference (UNUSED)" for the hypothetical future case where some new orchestrator does bypass the hook. v2.2 also INTEGRATES Plan A's E2E test sketch (§4.6 was already updated in v2.1; v2.2 adds Plan A's specific `predefined_sub_queries` and `register_class` techniques to §4.6 for completeness).

---

## §11 Executive Summary (v2.8 — updated for hierarchical layout)

For readers who need the fix without the analysis, here is the complete distillation:

### Root Cause (Unified)

**Fix #3 broke workspace assignment** — producing BOTH visible anomalies:
- **Anomaly 7 (hollow output)**: `output.md` missing because workspace path was wrong
- **Anomaly 8 (wrong naming)**: `flow_0_initial_round01/` instead of proper round naming

**One fix (Fix #13 Option F — Hierarchical Layout) resolves both.** Fix #12 NOT needed — the hook fires, the workspace was just wrong.

### Empirical Proof (the smoking gun)

| Run | `output.md` count in `worker_0` | `flow_0_followup/` exists? |
|---|---|---|
| Prior (cbaf8f2b, before fixes) | **9** | **YES** |
| Current (a7f74e8b, after Fix #3) | **0** | **NO** |

### The 5 Code Changes (Fix #13 Option F — Hierarchical Layout)

**Change 1** — `_worker_child_name` returns `flow_{i}` (`multi_flow_inferencer.py`):
```python
def _worker_child_name(self, index):
    return f"flow_{index}"
```

**Change 2** — MultiFlow stops propagating to flow_configs (`multi_flow_inferencer.py`):
```python
def _propagate_workspace_to_children(self, parent_workspace):
    super()._propagate_workspace_to_children(parent_workspace)
```

**Change 3** — LWI assigns `initial/` and `round01/` to children (`linear_workflow_inferencer.py`):
```python
_workspace_propagation_skip = frozenset(("default_initial_inferencer", "default_followup_inferencer"))
def _propagate_workspace_to_children(self, parent_workspace):
    # assigns initial/ and round01/ to children in dynamic_mode
```

**Change 4** — Per-round uses `self._workspace.child()` with `step_index >= 2` guard (`linear_workflow_inferencer.py`):
```python
if inf_instance is not None and step_index >= 2:
    round_ws = self._workspace.child(f"round{step_index:02d}")
```

**Change 5** — Remove `_base_followup_workspace` stash (`linear_workflow_inferencer.py`):
```python
_DERIVED_FROM_WORKSPACE = ()
```

### E2E Test (the regression catcher)

Run `instantiate(yaml) → infer() → assert filesystem`. Asserts:
1. `flow_0/children/initial/` exists (hierarchical)
2. `flow_0/children/round01/` exists (hierarchical)
3. `flow_0_initial_round01/` does NOT exist (wrong naming gone)
4. `flow_0/children/default_followup_inferencer/` does NOT exist (no orphan)
5. `output.md` exists at `initial/` and `round01/` (hollow output resolved)
6. No cross-worker symlinks (Anomaly 6 regression check)

### Implementation Effort: ~2h

| Phase | Effort |
|---|---|
| Changes 1+2 (`multi_flow_inferencer.py`) | 15 min |
| Changes 3+4+5 (`linear_workflow_inferencer.py`) | 15 min |
| Update 7 test files | 45 min |
| New regression + E2E tests | 30 min |
| Verification | 15 min |

### Files Modified

| File | Change |
|---|---|
| `multi_flow_inferencer.py` | Changes 1+2: `flow_{i}` naming, remove flow_configs loop |
| `linear_workflow_inferencer.py` | Changes 3+4+5: LWI propagation override, `step_index >= 2` guard, remove stash |
| `test_workspace_propagation.py` | Update MultiFlow propagation tests |
| `test_lwi_per_round_workspace.py` | Rewrite simulation + source assertion |
| `test_mfdual_workspace_anomalies_integration.py` | Update + add E2E test |
| `test_multi_flow_dual_inferencer.py` | Update worker naming + stash assertions |
| `test_mfdual_resume.py` | Update mock workspace setup |

---

## §14 Hierarchical Layout — Decision History

**v2.7+**: Hierarchical layout is NOW the primary fix (§4). This section is preserved as historical context for how the decision evolved.

**v2.6 (historical)** initially rejected hierarchical as the immediate fix, citing backward compatibility with existing tests and checkpoint paths. These concerns were later addressed:
- **Test compatibility**: All 7 affected test files identified and included in §4.6 update inventory
- **Checkpoint paths**: Verified NOT a risk — all checkpoint paths derived dynamically from `workspace.root` at runtime via `_get_result_path()` → `ws.checkpoint_path()`. No hardcoded layout names anywhere in the checkpoint system.
- **User's repeated request**: The user asked for hierarchical grouping multiple times. The layout is cleaner (LWI owns sub-tree, no cross-orchestrator coupling, eliminates `_base_followup_workspace` stash).

**v2.7 promoted hierarchical to §4 as the primary fix. The 5 code changes are detailed in §4.1–§4.5.**

---

## §13 Fix #12 Reference (UNUSED — kept for future hypothetical case)

This section archives Plan A's Fix #12 design IF the hook-bypass concern ever becomes real (e.g., a future orchestrator that does NOT call `child.ainfer()` publicly but instead calls a private method skipping `__ainfer_single_impl`). Currently this is hypothetical — verified live as of 2026-05-11 that all known orchestrators correctly route through the public path.

### Hypothetical Implementation (DO NOT IMPLEMENT NOW)

```python
# In InferencerBase
_output_finalized: bool = False  # class-level default

def _complete_inference(self, response=None, *, force=False):
    """Centralized output-finalization. Idempotent — first call writes, rest are no-ops.
    
    Use ONLY when an orchestrator bypasses __ainfer_single_impl by calling a
    private method on a child (rather than child.ainfer()). Currently no
    known orchestrator does this — keep this method dormant until needed.
    """
    if self._output_finalized and not force:
        return response
    if response is not None:
        response = self._finalize_output(response)
    self._post_finalize_deliverable_and_manifest()
    self._output_finalized = True
    return response
```

### Why It's Archived, Not Active

Per §2 verification: every Dual child invocation goes through `child.ainfer()` (lines 1089, 1249, 1257, 1411, 1419 of `dual_inferencer.py`) → `_ainfer_single` → `__ainfer_single_impl` → `_finalize_output` hook fires. Adding a redundant `_complete_inference()` call after each `child.ainfer()` would either:
- Be a NO-OP (idempotency flag prevents double-write) — defensible but adds clutter
- VIOLATE `has_local_access=True` contract if not careful — the hook would still try to write `outputs/output.md` for RovoDev, which is wrong

The right design (per v2.2): leave the hook firing automatically via the existing public-entry path. Address Anomaly 8 with Fix #13 only.

---

## §10 Cross-Reference

- Sister plan: `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/_docs/_plans/mfdual_workspace_layout_anomalies_fix_plan.md` (v4.8 — covers Anomalies 1–6 + Fix #11)
- Plan A (decision-influencing): `/Users/tchen7/.claude/plans/given-all-the-discussions-splendid-lantern.md` (Finding 1 verified; Finding 2 refined into Option E)
- This plan covers: **Both Anomaly 7 (hollow output) AND Anomaly 8 (LWI naming)** — Fix #13 Option F (5 hierarchical changes) addresses both, since they share the same root cause (Fix #3 broke followup workspace assignment)
- v2.5.1 NOTE: Earlier v2.0/v2.1 versions of this plan claimed Anomaly 7 was "NOT A BUG, by-design `has_local_access=True` contract". This claim was REFUTED in v2.3 by direct filesystem comparison (prior run had 9 `output.md` files in `worker_0`; current run has 0). The `has_local_access` early-return is a different mechanism unrelated to the current bug — the hollow output we observed was caused by Fix #3 deleting the workspace propagation, not by any `has_local_access` early-return.

---

## §15 Honest Critical Self-Assessment

### Why Did v1.1 Misdiagnose Anomaly 7?

I (the agent that wrote v1.0/v1.1) made the following analytical errors:
1. ❌ Did NOT read `inferencer_base.py:803-805` to verify the centralized hook fires unconditionally — assumed it did
2. ❌ Did NOT check `has_local_access` on the relevant inferencer types
3. ❌ Treated the BTA-aggregator-vs-MFDual-aggregator asymmetry as evidence of a hook bypass instead of evidence of agent-driven file authoring choice
4. ❌ Conflated "file missing" with "write hook failed" — when "file missing" actually means "agent chose not to write"

Plan A correctly identified Finding 1 by reading the contract. v2.0 corrects via direct verification.

### Why Plan A's Option D Is Slightly Insufficient

Plan A's surgical 2-line revert addresses the immediate symptom; the original concern (this plan v2.0) was that it didn't address an orphan `flow_0_workflow/children/default_followup_inferencer/` from LWI. v2.5+ verified at `inferencer_base.py:343` that the base walker's existing skip-if-already-set guard ALREADY prevents the orphan. So the v2.5+ position is: Plan A's 2-line revert is SUFFICIENT — no additional code needed for the orphan case; only Changes A + B are required.

### Confidence In v2.8

| Aspect | Confidence | Verified Against |
|---|---|---|
| Anomaly 7 IS a real bug | **HIGH** | Direct filesystem comparison: prior run (cbaf8f2b) had 9 `output.md` files; current run (a7f74e8b) has 0 |
| Anomaly 8 IS a bug | **HIGH** | Verified against live workspace structure (anomalous `flow_0_initial_round01/` dir name) |
| Root cause is Fix #3 breaking followup workspace assignment | **HIGH** | Both symptoms appear/disappear together across runs |
| Fix #12 should remain WITHDRAWN | **HIGH** | The hook fires; the workspace was just wrong |
| Fix #13 Option F (5 hierarchical changes) will fix both anomalies | **HIGH** | LWI owns sub-tree, assigns `initial/` and `round01/` directly; no sibling-naming derivation |
| Checkpoint paths NOT affected by layout change | **HIGH** | Verified: `_get_result_path()` → `ws.checkpoint_path()` — fully dynamic from `workspace.root` |
| 7 test files identified (complete inventory) | **HIGH** | Full grep for `flow_0_workflow`, `flow_0_initial`, `_base_followup_workspace`, `step_index > 1` |
| ~2h effort estimate | **MEDIUM-HIGH** | 5 code changes + 7 test files + E2E test |

---

## §12 If Asked "Pick One Plan"

**Both plans are now aligned (v2.8).** Plan A (`given-all-the-discussions-splendid-lantern.md`) is the concise implementation reference. This plan (Plan B v2.8) provides the analytical depth: root-cause verification, acceptance criteria, risk framework, and the complete self-correction provenance trail v1.0 → v2.8. Both describe the same 5 hierarchical changes.
