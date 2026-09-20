# MFDual Workspace Layout Anomalies — Fix Plan

**Status**: ACTIVE v4.8 — Targeted critical-thinking pass on agent feedback applied (verified 2026-05-10 20:07). Two valid feedback items applied: (FB2) Fix #2's sketch was rewriting `surface_outputs_from`'s API — replaced with single-line `dirs[:]` `os.walk` prune that preserves the live `namespace`/`skip_existing` kwargs and the `deliverables_dir` attribute (NOT `outputs_dir`); (FB3) Fix #3's `("followup_inferencer", None)` placeholder risked `parent.child(None)` crash at the consumer — added Option A.1 (omit entry entirely, recommended) and Option A.2 (explicit None-guard at consumer) with crash explanation. Five other feedback items REJECTED after critical-thinking verification (FB1 line citations bogus; FB4 §3 Fix #5 Part B IS already v4.6-converted; FB5 `import time` IS present at line 1779; FB6/FB7 already covered by existing risk/open-question rows). Plan retains v4.7's RE-INTEGRATED foundations: **the canonical `_instantiate.py` lives at `/Users/tchen7/MyProjects/CoreProjects/RichPythonUtils/src/rich_python_utils/config_utils/_instantiate.py` (NOT `rovoteam/PythonUtils/...`)** — verified by grep of `AgentFoundation/src/` imports which use `from rich_python_utils.config_utils import instantiate` (registered_targets.py:8, factories.py:16, mock_bta_components.py:10). The `rovoteam/PythonUtils/` copy is a STALE OLDER MIRROR (28 KB Apr 29 vs the live RichPythonUtils 39 KB May 4). All v4.4-v4.6 line-number references for Fix #9 (`_instantiate.py:959` for `_partial_: true` injection, `_filter_attrs_keys:944-972`) are WRONG by ~250 lines because they referenced the stale mirror. Verified live line numbers (2026-05-10 19:58): `_FACTORY_MARKER` at line 35, `_ImportFactory` class at line 38, `_filter_attrs_keys` at line 881, `*_factory` handling block at lines 944-972, the buggy `val["_partial_"] = True` at line 959 (single-factory case) and 972 (dict-of-factories case), the gated `_factory_configs.append(...)` at lines 957 and 970 (gated on `_FACTORY_MARKER in val`). Mechanical root-cause attribution (§2.8) is otherwise UNCHANGED — the bug mechanic is identical, only the file path and line numbers were wrong. Retains all v4.6 design decisions (single boolean attrib for sharing detection, WARN-only, base-class helper, layered switch_role, audit hardening, snapshot-at-phase-time semantics, defense-in-depth model). Adds v4.6's BTA call-site correction (workers built per-call in `_build_subgraph_spec()` at lines 1510-1512, validation call at end of that method) — also verified.

**v4.4 (refined by v4.5)** — Pinpoints the MECHANICAL ROOT CAUSE in `python_utils/config_utils/_instantiate.py:711` (vanilla `_partial_: true` injection produces `functools.partial` with EAGERLY-INSTANTIATED nested children baked into `keywords`, shared across every `factory()` call). Introduces **Fix #9: `LazyConfigFactory`** — a generalization of the existing `_ImportFactory` that becomes the universal mechanism for ALL `*_factory`-suffix attrs fields, NOT just `_import_:`-tagged ones. Each `factory()` call deep-copies the captured config and re-runs `instantiate()`, producing a completely fresh sub-tree with NO shared inner instances. **Fix #10**: subsumes existing `_ImportFactory` as a deprecated alias of `LazyConfigFactory` (zero-breaking-change migration). BTA call-sites updated with `isinstance(x, (functools.partial, LazyConfigFactory))` recognition pattern (the explicit, debuggable form chosen over silent kwarg-filtering). Adds §2.8 (mechanical root cause walkthrough) + §2.9 (BTA recognition gap) documenting WHY no current orchestrator can uniformly distinguish factory types. v4.4 explicitly REJECTS the "deep-copy `flow_configs` instances at BTA" alternative as the wrong layer (instance-level deep-copy breaks workspace propagation, attrs validators, lock semantics; config-level re-instantiation is the principled fix that already lives in the codebase).

**v4.3 (refined by v4.4)** — Refined v4.2's attribution by explicitly identifying the SHARING SUBSTRATE: cross-run comparison (prior run `task-cbaf8f2b_20260510_033521` at 03:35 vs recent run `task-7522157d_20260510_133627` at 13:36) proves that the cross-worker symlink anomaly is **PRE-EXISTING** and was only made VISIBLE by the prior `mfdual_self_promotion_gap_INTEGRATED_plan.md` integration of `fixer_match_winner` + `reviewer_match_second`. The actual sharing happens at the **flow-pool inferencer layer** (`flow_configs[i].initial_inferencer` / `followup_inferencer`), NOT at the top-level role-slot layer. v4.3 corrects §2.6's narrative (sharing at flow-pool layer; role-slot becomes a downstream alias), expands Fix #5 to deep-walk `flow_configs` slots, and adds §2.7 documenting the prior-run evidence + the "exposed-not-introduced" causality chain. The fix mechanics (Fix #7, Fix #8, layered `switch_role()`) remain unchanged — they correctly target the symptom and the timing race regardless of which layer caused the sharing.

**v4.2 (refined by v4.3)** — Added Anomaly 6 (Cross-Worker / Role-Inverted Audit Symlinks — CRITICAL) + two coordinated fixes (Fix #7 audit-symlink hardening; Fix #8 snapshot-at-phase-time semantics). Anomaly 6 provides direct hard-evidence proof of the instance-sharing hypothesis from Anomaly 1 — symlinks under `worker_1/children/round_01/` cross-link to `worker_0`'s slots and role-invert (`fix → review_inferencer`, `review → fixer_inferencer`). v4.1's layered `switch_role()` makes per-call mutation atomic but does NOT prevent two concurrent workers from racing on the SAME shared instance — Fix #7 makes the audit layer DETECT and LOUDLY REPORT cross-MFDual leakage; Fix #8 changes audit semantics from "snapshot at audit time" to "snapshot at phase-execution time" so audit symlinks reflect where work ACTUALLY ran, not where the instance currently lives. Together with Fix #5's BTA sharing detection (which prevents the bug at construction time), this provides defense in depth.

**v4.1 (preserved as foundational)** — Refines v4.0's `switch_role()` API into a **layered design** that respects the documented `InferencerBase` ↔ `TemplatedInferencerBase` inheritance boundary. `InferencerBase.switch_role()` handles ONLY base-level attribs (`workspace`, `output_is_deliverable`, `is_deliverable_boundary`, audit trail, session reset); `TemplatedInferencerBase.switch_role()` overrides with `super()` and adds template-related attribs (`template_key`, `template_root_space`, `template_extra_feed`, `template_variables`, `template_version`, `modes`). This removes the `hasattr` defensive guards (attribs definitely exist where they're declared), makes typo'd kwargs raise `TypeError` instead of silently dropping, and mirrors the existing `_propagate_to_children()` precedent that already splits responsibilities the same way. Caller code is UNCHANGED — Python's MRO routes `inferencer.switch_role(template_key=..., workspace=...)` to the correct layer transparently.

**v4.0 (superseded)** bundled all role-relevant attribs onto `InferencerBase.switch_role()` with `hasattr` guards. While operationally correct, this violated the deliberate inheritance boundary documented at `inferencer_base.py:184-198` ("Template fields ... live on TemplatedInferencerBase. Orchestrators ... inherit from InferencerBase directly and don't carry template state"). v4.1 fixes this without changing call-site ergonomics.

**v3.0 (superseded)** previously: Added Anomaly 5 (Role-Mutation Template Key Drift — CRITICAL) + Fix #6 (Hybrid Option A+C). When MFDual reuses winning flow inferencer as fixer (and loser as reviewer), the inferencer's `template_key` is NOT updated, causing it to render `initial.jinja2` instead of `review.jinja2`/`followup.jinja2`. All consensus iterations were silently broken.
**Severity**: 🟡 Medium (debugging/inspection clarity; no data loss in production output)
**Effort**: ~3.5 hours code + ~1 hour tests = **~4.5 hours**
**Sources**: User-reported anomalies in run `task_task-7522157d_20260510_133627`

---

## §1 The Anomalies — Hard Evidence

### Workspace Tree Observed (worker_0's inner BTA → MFDual)

```
worker_0/children/base_inferencer/children/   ← MFDual workspace (BTA-like for MFDual's flow workers)
├── breakdown/                  ✅ has content (5.99 KB output.md)
├── flow_0_initial/             ✅ has content (39.6 KB output.md, 13:49)
├── flow_0_round01/             🚨 EMPTY (no logs, no outputs, no checkpoints)
├── flow_0_round02/             ✅ has content (96.4 KB output.md, 13:56) — TWO inference inputs
├── flow_0_workflow/            🆕 LWI state container (182 KB final_result.json)
├── flow_1_initial/             ✅ has content (23.3 KB output.md, 13:56)
├── flow_1_round01/             🚨 EMPTY
├── flow_1_round02/             ✅ has content (74.3 KB output.md, 14:04)
├── flow_1_workflow/            🆕 LWI state container
└── aggregator/                 🚨 EMPTY (logs/checkpoints/outputs all 0 bytes)
```

### Final Deliverables Path Observed (suspicious)

```
worker_0/children/base_inferencer/outputs/
├── output.md                                                     (82 KB, 14:13) ✅
├── output_manifest.json                                          (8.7 KB, 14:14) ✅
├── workers/                                                       ✅
└── final_deliverables/                                            ✅ Layer 1
    ├── output.md                                                  (82 KB) ✅
    ├── output_manifest.json                                       ✅
    ├── workers/                                                   ✅
    └── final_deliverables/                                        🚨 Layer 2 (DOUBLE!)
        ├── output.md                                              (24 KB — STALE 14:03)
        └── .self_promoted                                         (0 bytes, 14:05)
```

---

## §2 Root-Cause Analyses

### 🔴 Anomaly 1 [CRITICAL — DIAGNOSIS FINALIZED]: Aggregator Inferencer Instance Shared Across Workers via Hydra/Factory

**Diagnostic journey**:
- Round 1: I claimed "winner-pick bypass — acceptable" → WRONG
- Round 2: I claimed "shared with flow_0_initial → workspace skip" → mostly RIGHT but uuid evidence was confused
- Round 3: User pushed back with YAML evidence showing distinct config blocks → forced re-investigation
- Round 4: Hard evidence reveals: only 4 RovoDevCli inferencer UUIDs exist for worker_0, vs 5 for worker_1 → confirms shared/missing aggregator

**Hard Evidence (worker_0 vs worker_1 asymmetry)**:

| | worker_0 | worker_1 |
|---|---|---|
| Aggregator workspace | EMPTY (no logs, cache, outputs) | ✅ Has logs + cache + 2 InferenceInputs |
| Aggregator output | NONE in canonical dir | ✅ 82 KB at canonical path |
| Aggregator inferencer uuid | `bfd43b17` (SAME as flow_0_initial!) | `c5f94c73` (FRESH) |
| Cache location | `flow_0_initial/_runtime/inferencer_cache/RovoDevCliInferencer-bfd43b17_*` | `aggregator/_runtime/inferencer_cache/RovoDevCliInferencer-c5f94c73_*` |

**The smoking guns**:
1. Worker_0 has only 4 RovoDevCli inferencer UUIDs (bfd43b17, e98c87d0, ce28e3b5, 3b05c637) — all accounted for as flow inferencers + reused fixer/reviewer.
2. Worker_1 has 5 RovoDevCli UUIDs — including c5f94c73 ONLY at the canonical `aggregator/` workspace.
3. Worker_0's `artifacts/round00_output.md` IS aggregator output (4369 bytes containing `winner_pick` JSON block + ranking) — proving aggregator DID produce output, just not in the canonical workspace.
4. The aggregator inference for worker_0 was likely executed by reusing one of the 4 existing inferencer instances; workspace assignment was skipped because `_workspace` was already bound to a flow workspace.

**Why worker_0 vs worker_1 differ**: Hydra factory construction shared the `multi_flow_aggregator_inferencer` config across BTA worker invocations. Worker_0's MFDual was constructed FIRST, bound the aggregator to a flow workspace via shared instance. Worker_1's MFDual was constructed SECOND, possibly recreating the aggregator instance fresh.

**Workspace propagation skip logic at fault** (`inferencer_base.py` workspace propagation):
```python
# When MFDual tries to assign aggregator workspace:
if getattr(child, "_workspace", None) is not None:
    return  # ← Skipped because flow_0_initial's workspace was already set
```

**This is the SAME bug class as Fix #1 (reviewer/fixer workspace isolation)** — but the aggregator slot was NOT included in the role-mutation workspace-reassignment fix.

**Why worker_1 was different**: Likely a race condition — worker_1's aggregator slot got a freshly-constructed inferencer (`c5f94c73`) before any flow inferencer had set its workspace. Worker_0's aggregator slot bound LATER, after `bfd43b17` was already running as `flow_0_initial`.

**Where in code**:
- `multi_flow_dual_inferencer.py` constructs MFI with `aggregator_inferencer=self.multi_flow_aggregator_inferencer`. The shared instance is reused across MFDual instances.
- `_select_reviewer_and_fixer` calls `_reassign_role_workspace()` for reviewer + fixer, but NOT for aggregator
- Result: aggregator inherits a stale workspace from whichever role the shared instance previously played

**Fix**: Apply the same `_reassign_role_workspace()` pattern to the aggregator slot before invoking aggregator inference. Mirror Fix #1's pattern exactly.

**Priority**: 🔴 CRITICAL — content went to wrong directory, debugging is broken, manifest at canonical path is missing.

### 🚨 Anomaly 2: Double `final_deliverables/final_deliverables/` Nesting

**Root cause**: The MFDual surfaces winner-flow content via `surface_outputs_from(winner_ws)` — but the winner workspace already has its OWN `outputs/final_deliverables/` (because `output_is_deliverable=True` self-promotes). When `surface_outputs_from` copies the entire `outputs/` tree (including the nested `final_deliverables/` subdir), you get **layered nesting**.

Hard evidence:
- Inner `final_deliverables/output.md` is **24 KB** (stale early flow content from 14:03)
- Outer `final_deliverables/output.md` is **82 KB** (later aggregated content from 14:13)

The 24 KB file was the ORIGINAL winner-flow's self-promoted deliverable. The 82 KB came from a LATER aggregation/fix step. They got stacked.

**Where in code**:
- `inferencer_workspace.py:130 surface_outputs_from()` copies `outputs/` recursively, including pre-existing `final_deliverables/` subdir
- Should detect when source already has `final_deliverables/` and either flatten (overwrite at correct level) or skip the inner copy

**Fix**: In `surface_outputs_from`, when `skip_existing=True` AND source has `outputs/final_deliverables/`, the destination's `final_deliverables/` shouldn't get a redundant inner copy.

### 🚨 Anomaly 3: `flow_0_round01` and `flow_1_round01` are EMPTY

**Root cause**: The LWI follow-up workspace naming starts at `_round01` (assigned at construction in `_propagate_workspace_to_children`). But when LWI runs followup steps, **per-round workspace** is computed **per-step** (`_round02` for step 1, `_round03` for step 2, etc.). Step 0's followup result goes into `_round02`, not `_round01`.

Looking at LWI:533-538:
```
<parent>/flow_X_round01/  (step 0 — assigned at construction)  ← NEVER USED
<parent>/flow_X_round02/  (step 1)                              ← actual step 0 output
<parent>/flow_X_round03/  (step 2)                              ← actual step 1 output (if any)
```

This is an **off-by-one naming bug**. The first followup step's workspace should be `_round01`, not `_round02`. Or alternately: `_round01/` should be deleted because it's only a "construction-time placeholder" that's never used.

**Where in code**:
- `multi_flow_inferencer.py:533` assigns `followup_inferencer` → `flow_{i}_round01` at construction (placeholder)
- `linear_workflow_inferencer.py:533` per-step workspace computed as `flow_X_round{step+2}` (off by one OR the construction placeholder is throwaway)

### 🚨 Anomaly 4: `flow_X_workflow/` Appears (Surprise — Not Discussed in Plan)

**Root cause**: This is **Fix #6 from the prior MFDual plan** (commit confirmed at `multi_flow_inferencer.py:559-582`). The override renames BTA's worker container from `worker_N` to `flow_N_workflow` to clarify "this dir holds LWI orchestration state, not actual flow inferences."

**Reading the docstring at lines 568-581**:
> "per-flow work happens in `flow_N_initial/` and ... so renaming the storage container to `flow_N_workflow` makes the workspace layout self-documenting"

This was **deliberately chosen** as the implementation approach (vs. the user's proposal of `worker_N/flow_X_initial/...` nesting).

**Disagreement with implemented choice**: The user wanted **everything for flow N to live UNDER worker_N**:
```
worker_0/                       ← user's proposal
  ├── flow_0_initial/
  ├── flow_0_round01/
  └── flow_0_workflow/          (orchestration state)
worker_1/
  ├── flow_1_initial/
  └── ...
```

But the implementation chose **flat sibling** layout:
```
flow_0_initial/                 ← actual chosen
flow_0_round01/
flow_0_workflow/                ← just for state
flow_1_initial/
...
```

The user is right that nested-under-worker is more intuitive and prevents the `flow_X_workflow/` "stray" feel.

---

## §2.5 — 🔴 Anomaly 5 [CRITICAL]: Role-Mutation Template Key Drift

**Discovered**: 2026-05-10 17:42 — User inspection of reviewer + fixer rendered prompts.

**Symptom**: Reviewer prompt and fixer prompt **both render `initial.jinja2`** instead of their canonical `review.jinja2` / `followup.jinja2` templates.

**Hard Evidence (from run `task_task-7522157d_20260510_133627`)**:

| Inferencer | Expected template | Actual rendered template | File observed |
|---|---|---|---|
| `review_inferencer` (worker_0) | `plan/main/review.jinja2` (verdict format) | 🚨 `plan/main/initial.jinja2` ("create comprehensive artifact") | `RovoDevCliInferencer-e98c87d0.jsonl.parts/InferenceInput/20260510_141434_*.txt` |
| `fixer_inferencer` (worker_0) | `plan/main/followup.jinja2` (cp-first + ProposedDocument + ReviewerFeedback) | 🚨 `plan/main/initial.jinja2` (no ReviewerFeedback, no prior_output_path, no cp directive) | `RovoDevCliInferencer-bfd43b17.jsonl.parts/InferenceInput/20260510_142432_*.txt` |

**Tell-tale phrases proving template misuse**:
- Reviewer prompt contains: *"You are tasked with creating a comprehensive artifact"* + *"Your Task: Create a comprehensive, actionable, implementable plan"* — these are the `initial.jinja2` opening, NOT the reviewer's expected verdict-evaluation framing
- Fixer prompt is missing: `<ProposedDocument>`, `<ReviewerFeedback>`, `prior_output_path`, the cp-first directive, the soft NOTES guidance
- Both prompts contain: the original user `<UserRequest>` tag with full description+todos (correct for `initial`, redundant for `review`/`followup`)

**Root cause**: When MFDual's `_select_reviewer_and_fixer()` mutates `self.fixer_inferencer = winner_flow_inferencer` (and `self.review_inferencer = loser_flow_inferencer`), the assigned instance retains its **original `template_key="initial"`** from when it was constructed as a flow inferencer. The mutation rebinds workspace via `_reassign_role_workspace()` (Fix #1 from prior session) but does **NOT** rebind `template_key` or `template_root_space`.

So when the orchestrator calls `role_inferencer.ainfer(input)`, the leaf renders `plan/main/initial.jinja2` — its original template — instead of the role-appropriate `plan/main/review.jinja2` or `plan/main/followup.jinja2`.

**Why all our Phase 5 work didn't catch this**:
- Phase 5 leaf-owned templates correctly cascade `template_key` at **construction time** via SLOT_DEFAULTS based on the slot the inferencer is BORN into
- `flow_configs[i].initial_inferencer` slot → SLOT_DEFAULTS cascade gives `template_key="initial"` ✅
- `review_inferencer` slot → SLOT_DEFAULTS cascade gives `template_key="review"` ✅
- `fixer_inferencer` slot → SLOT_DEFAULTS cascade gives `template_key="followup"` ✅
- BUT: `_select_reviewer_and_fixer()` reassigns slots AT RUNTIME (after construction), and SLOT_DEFAULTS only fire at construction → NO cascade refresh → wrong template renders

**Why severity is CRITICAL**:
- Every consensus iteration of every MFDual since Phase 5 has silently rendered wrong templates
- The "review" was actually a re-investigation (the loser-flow inferencer just re-ran the initial task)
- The "fix" never received `prior_output_path`, never got the cp-first directive, never saw `<ReviewerFeedback>`
- The 73 KB "consensus-iterated final plan" we got is actually 3 separate `initial.jinja2` runs, not a propose→review→fix loop
- Phase 0 path-aware infrastructure (`prior_output_path` plumbing) is being SKIPPED for these flows because `prior_output_path` only appears in `followup.jinja2`, which never renders for the runtime-mutated fixer

**Where in code**:
- `multi_flow_dual_inferencer.py` `_select_reviewer_and_fixer()` (around line 660) — performs `self.fixer_inferencer = winner` mutation
- `_reassign_role_workspace()` (around line 390) — rebinds workspace correctly but ignores template_key/template_root_space
- The fix needs to extend `_reassign_role_workspace()` to ALSO refresh template config

---

## §2.6 — 🔴 Anomaly 6 [CRITICAL]: Cross-Worker / Role-Inverted Audit Symlinks

**Discovered**: 2026-05-10 18:28 — User inspection of `worker_*/children/round_01/` symlinks.

**Hard Evidence (from run `task_task-7522157d_20260510_133627`)**:

```
# worker_1/children/round_01/
fix             →  worker_0/children/review_inferencer    ← CROSS-WORKER + ROLE-INVERTED
review          →  worker_1/children/fixer_inferencer     ← ROLE-INVERTED (within worker)
review_dispatch →  worker_1/children/fixer_inferencer     ← ROLE-INVERTED (within worker)
propose         →  worker_1/children/base_inferencer      ← correct

# worker_0/children/round_01/
fix             →  worker_0/children/fixer_inferencer     ← correct
review          →  worker_0/children/review_inferencer    ← correct
review_dispatch →  worker_0/children/review_inferencer    ← correct
propose         →  worker_0/children/base_inferencer      ← correct
```

**The asymmetry alone proves a bug**: worker_0's symlinks are clean; worker_1's are mangled with cross-worker leakage and role inversion.

**Source code**: `dual_inferencer.py:712-757` (`_record_round_audit()`):

```python
def _record_round_audit(self, round_idx, phase, inferencer, extra=None):
    ...
    nav_dir = os.path.join(children_dir, f"round_{round_idx:02d}")
    link_path = os.path.join(nav_dir, phase)
    target = str(inferencer._workspace.root)        # ← TARGET is whatever the inferencer's _workspace points to AT AUDIT TIME
    if os.path.islink(link_path):
        os.unlink(link_path)                        # ← silently overwrite
    os.symlink(target, link_path, target_is_directory=True)
```

Three mechanically-distinct bugs compose to produce the observed symptom:

### Bug 6a — Instance sharing at the FLOW-POOL layer (NOT at the role-slot layer) [v4.3 REFINED]

**v4.2 narrative (incomplete)**: "the same Python instance is reachable as `worker_1.fixer_inferencer` AND `worker_0.review_inferencer`."

**v4.3 corrected narrative**: The sharing is actually at the **flow-pool inferencer layer** (`flow_configs[i].initial_inferencer` / `followup_inferencer`). The role-slot symptom is a downstream alias created by the integrated plan's `fixer_match_winner=True` + `reviewer_match_second=True` features, which assign the WINNING and RUNNER-UP **flow** inferencers to MFDual's role slots:

```
PRE-EXISTING: flow_configs[0].initial_inferencer ─┐
              flow_configs[1].initial_inferencer ─┼─ all bind to the SAME
              flow_configs[0].followup_inferencer ┤  Python InferencerBase
              flow_configs[1].followup_inferencer ┘  instance (YAML wires
                                                     them as singletons,
                                                     not factories)

EXPOSED by integrated plan:
  worker_0:
    self.fixer_inferencer  = winner_flow_inferencer  (= shared singleton)
    self.review_inferencer = runner_up_flow_inferencer (= shared singleton)
  worker_1:
    self.fixer_inferencer  = winner_flow_inferencer  (= same shared singleton)
    self.review_inferencer = runner_up_flow_inferencer (= same shared singleton)

  → All four "role-slot" handles point to the SAME object across both workers.
  → _reassign_role_workspace() runs on it; LAST writer wins.
  → Audit symlinks reflect last-writer state.
```

**Why v4.3 cares about the distinction**:
- Fix #5's sharing-detection scope must reach into `flow_configs[*].initial_inferencer` AND `flow_configs[*].followup_inferencer` (NOT just the top-level role slots) — already partly anticipated by Fix #5's recursive `_collect_descendant_ids` design but should be explicit about flow_configs traversal.
- The fix at the BTA/Hydra factory layer needs to ensure flow_configs entries get `_factory` suffix or Hydra's `_partial_: true` so each worker MFDual's `flow_configs[i].initial_inferencer` is a fresh instance.
- `inferencer_pool` entries (used for alias resolution by `reviewer_match_second`/`fixer_match_winner`) are ALSO susceptible — must be deep-walked too.

**Causality chain (the integrated plan's role)**: The plan did NOT introduce instance sharing — that's a pre-existing YAML factory wiring concern. The plan introduced the `reviewer_match_second` + `fixer_match_winner` features that PROMOTE shared flow inferencers into MFDual's role slots, which the audit layer then symlinks. **Sharing was always there; v3+ features made it loudly visible.** See §2.7 for the cross-run evidence proving this.

### Bug 6b — Within-worker role aliasing (`review_inferencer is fixer_inferencer`)

For both `worker_1/round_01/review` AND `worker_1/round_01/fix` to symlink to the SAME directory (`worker_1/.../fixer_inferencer`), worker_1's `self.review_inferencer` and `self.fixer_inferencer` must be the same instance at audit time. This happens when `_select_reviewer_and_fixer()` ends up assigning the same winner instance to both slots — possible when `inferencer_pool` has only one templated instance, when `review_default == winner` and `priority_pool` is exhausted (line 558 of `multi_flow_dual_inferencer.py` — explicit "self-review" fallback), or when the loser-as-reviewer path collapses for any other reason. The `_reassign_role_workspace()` then runs TWICE on the same instance with two different role names — the LAST call wins.

### Bug 6c — `_record_round_audit()` audit-time snapshot is unstable

`_record_round_audit("review", self.review_inferencer)` reads `inferencer._workspace.root` AT THE MOMENT THE AUDIT RUNS, not at the moment the phase ACTUALLY EXECUTED. Because:
- Worker_0 and worker_1 run concurrently (BTA dispatches in parallel via `MPTarget`).
- They share the same `review_inferencer` instance (Bug 6a).
- They each call `_reassign_role_workspace(self.review_inferencer, "review_inferencer")` which mutates the SAME `_workspace` slot of the SAME instance.
- The audit symlink, recorded later, reads whatever the latest mutator left there.

So the audit symlink is not a "where this phase ran" record; it's a "where this instance currently lives" record — those diverge precisely because of Bug 6a.

### Severity

**🔴 CRITICAL** — same severity tier as Anomaly 5. Reasons:

1. **The symlinks are the diagnostic surface developers rely on for postmortem inspection.** When they lie, every subsequent debugging session is poisoned.
2. **The cross-worker symlink reveals data-corruption potential**: if reviewer wrote to its workspace BUT another worker's `_reassign_role_workspace()` had moved that workspace mid-run, output may have been written to the wrong directory entirely. The audit symlinks suggest this happened — the "review output" we'd find at worker_1's `round_01/review` target is actually in worker_1's fixer slot, not its review slot.
3. **The exact same instance-sharing causes Anomaly 5's template_key drift**: worker_1 mutates the shared instance's `template_key` to "followup" for fixer use, then worker_0 mutates the SAME instance's `template_key` to "review" for reviewer use — whichever runs `ainfer()` LAST renders with the wrong template, even WITH v4.1's `switch_role()` (because `switch_role` is atomic per-call, but two concurrent callers race the assignment).

### Why v4.1's `switch_role()` Alone Does NOT Solve This

The v4.1 plan correctly fixes the per-call atomicity of role mutation. But it does NOT prevent two concurrent callers from racing each other on the SAME shared instance. The audit symlinks would still be wrong because:

```
T=0   worker_0._reassign_role_workspace(shared_instance, "review_inferencer")
       → switch_role(workspace=worker_0/review_inferencer, template_key="review")
       → shared_instance._workspace.root = worker_0/review_inferencer/

T=1   worker_1._reassign_role_workspace(shared_instance, "fixer_inferencer")
       → switch_role(workspace=worker_1/fixer_inferencer, template_key="followup")
       → shared_instance._workspace.root = worker_1/fixer_inferencer/

T=2   worker_0._record_round_audit("review", shared_instance)
       → reads shared_instance._workspace.root  → worker_1/fixer_inferencer/   ← WRONG (last-writer-wins)
       → symlinks worker_0/round_01/review → worker_1/fixer_inferencer/        ← cross-worker, role-inverted

T=3   worker_1._record_round_audit("fix", shared_instance)
       → reads shared_instance._workspace.root  → worker_1/fixer_inferencer/   ← also reads same value
       → symlinks worker_1/round_01/fix → worker_1/fixer_inferencer/           ← happens to look "right" for worker_1's fixer
```

In other words: v4.1 makes mutation atomic per-call but doesn't address the racing-shared-instance problem. **Fix #5 (sharing detection) + new Fix #7 + new Fix #8 below are the layered solution.**

---

---

## §2.7 — Cross-Run Evidence: The Bug Was Pre-Existing, Made Visible By Integrated Plan [v4.3 NEW]

**Comparison**: Two runs separated by ~10 hours, BOTH on the same code state (no commits between them — git agent confirmed last commit was `ef49c7e` at 2026-05-09 18:11:33; both runs occurred within the same working-tree window):

| Aspect | Prior run `task-cbaf8f2b_20260510_033521` (03:35) | Recent run `task-7522157d_20260510_133627` (13:36) |
|---|---|---|
| **worker_0 round_01/propose** | → worker_0/.../base_inferencer ✅ | → worker_0/.../base_inferencer ✅ |
| **worker_0 round_01/fix** | → worker_0/.../fixer_inferencer ✅ | → worker_0/.../fixer_inferencer ✅ |
| **worker_0 round_01/review** | → worker_0/.../**flow_1_initial** ⚠️ | → worker_0/.../review_inferencer ✅ |
| **worker_0 round_01/review_dispatch** | → worker_0/.../**flow_1_initial** ⚠️ | → worker_0/.../review_inferencer ✅ |
| **worker_1 round_01/propose** | → worker_1/.../base_inferencer ✅ | → worker_1/.../base_inferencer ✅ |
| **worker_1 round_01/fix** | → **worker_0**/.../base_inferencer/.../**flow_1_initial** 🔴 cross-worker + flow-named | → **worker_0**/.../**review_inferencer** 🔴 cross-worker + role-named |
| **worker_1 round_01/review** | → **worker_0**/.../**fixer_inferencer** 🔴 cross-worker + role-inverted | → worker_1/.../**fixer_inferencer** 🔴 within-worker role-inverted |
| **worker_1 round_01/review_dispatch** | → worker_1/.../review_inferencer ✅ | → worker_1/.../**fixer_inferencer** 🔴 within-worker role-inverted |

### The Three Decisive Observations

**Observation 1 — Cross-worker leakage exists in BOTH runs**

worker_1's `fix` symlink in BOTH runs points into worker_0's tree. The pre-existing run is unambiguous evidence that **the cross-worker instance sharing is NOT introduced by anything between the two runs** (no commits happened between them; the working tree was the same).

**Observation 2 — Symlink TARGETS shifted from "flow-named" to "role-named"**

| Slot | Prior-run target | Recent-run target |
|---|---|---|
| worker_0/round_01/review | `flow_1_initial` | `review_inferencer` |
| worker_0/round_01/review_dispatch | `flow_1_initial` | `review_inferencer` |
| worker_1/round_01/fix | `flow_1_initial` | `review_inferencer` |
| worker_1/round_01/review | `fixer_inferencer` | `fixer_inferencer` |

The shift from flow-named (`flow_1_initial/`) to role-named (`review_inferencer/` / `fixer_inferencer/`) targets between the two runs IS evidence of the `_reassign_role_workspace()` helper being effective. In the prior run, the shared instance's `_workspace.root` was still set to its CONSTRUCTION-TIME workspace (`flow_1_initial/` from MFI's flow construction). In the recent run, the helper rebound it to a canonical role-named slot.

**Observation 3 — Within-worker role aliasing only appears in the RECENT run**

Prior run: worker_1's `review` and `review_dispatch` point to DIFFERENT places (`fixer_inferencer/` vs `review_inferencer/`). Recent run: worker_1's `review` and `review_dispatch` point to the SAME place (`fixer_inferencer/`). The recent run's stronger aliasing is consistent with `reviewer_match_second` being more actively used (or different rankings producing different runner-up resolutions per call).

### Causal Decomposition (refined v4.3)

| Bug | Pre-existing? | Affected by integrated plan? | Affected by `_reassign_role_workspace()` helper? |
|---|---|---|---|
| **Bug 6a — flow-pool instance sharing across workers** | ✅ YES (visible in prior run) | ❌ NO | ❌ NO |
| **Bug 6b — within-worker role aliasing** (`review_inferencer is fixer_inferencer`) | ⚠️ STRONGER NOW | ✅ EXPOSED via `reviewer_match_second` + `fixer_match_winner` | ❌ NO |
| **Bug 6c — audit-time vs phase-time snapshot drift** | ✅ YES (intrinsic to current `_record_round_audit()`) | ❌ NO | ❌ NO |
| **Symlink targets switched flow-named → role-named** | n/a (a SIDE EFFECT of) | ❌ NO | ✅ YES (helper rebinds workspace) |

### Critical Finding

**The integrated plan did NOT introduce these bugs. It made them MORE VISIBLE by:**
1. Adding `reviewer_match_second=True` and `fixer_match_winner=True` features that PROMOTE flow-pool inferencers into MFDual's named role slots → the audit layer's role-named symlinks now reflect those references → cross-worker sharing of flow-inferencers becomes loudly visible at the role-symlink layer.

**`_reassign_role_workspace()` (added by a parallel session, not the integrated plan) made the targets cleaner-looking** by rebinding to canonical role-named slots → which paradoxically made the WITHIN-WORKER aliasing MORE confusing (because two roles now map to the same role-named slot, instead of one role pointing to a flow-named slot).

**The actual root cause** — flow inferencers being shared instances across workers because YAML wires them as singletons, NOT as factories with `_factory` suffix — is **outside both plans' scope**. It is the BTA / Hydra factory-wiring layer that needs the structural fix; this fix plan's Fix #5 (deep-walk sharing detection) + Fix #7 (audit diagnostics) + Fix #8 (snapshot semantics) provide the LAYERED defense, but the GROUND TRUTH remediation is at the YAML topology layer.

### Why This Matters For Implementation Order

1. **Fix #5 Part A** (BTA worker diagnostic logging) is even MORE valuable in light of v4.3 — it should log NOT JUST top-level worker IDs but ALSO each worker's `flow_configs[i].initial_inferencer` ID and `flow_configs[i].followup_inferencer` ID. This will make the prior-run sharing visible in logs, not just in symlinks.
2. **Fix #5 Part B** (strict-mode sharing detection) must deep-walk flow_configs (already conceptually in `_collect_descendant_ids` per the original Fix #5 sketch — needs explicit confirmation that flow_configs are walked).
3. **YAML topology audit** — separate from this plan — needs to verify that all flow_configs entries either use `_factory` suffix OR Hydra's `_partial_: true` so each worker's MFDual gets a fresh per-flow instance.
4. **The integrated plan's `reviewer_match_second` / `fixer_match_winner` are CORRECT FEATURES, not bugs**. They should not be reverted; instead, the underlying flow-pool instance sharing must be fixed at the topology / factory layer.

---

---

## §2.8 — Mechanical Root Cause: `_filter_attrs_keys` Eager Instantiation [v4.4 NEW]

**Discovered**: 2026-05-10 18:42-19:00, after another agent's diagnosis was cross-verified against the actual `_walk` and `_filter_attrs_keys` implementation.

**Source code site**: `/Users/tchen7/MyProjects/CoreProjects/RichPythonUtils/src/rich_python_utils/config_utils/_instantiate.py:944-972`.

### The Mechanical Walkthrough (verified against direct code reads)

When the YAML below is parsed:

```yaml
worker_factory:
  _target_: MultiFlowDual
  flow_configs:
    - initial_inferencer:
        _target_: ${_params.default_inferencer}
      followup_inferencer:
        _target_: ${_params.default_inferencer}
    - initial_inferencer:
        _target_: ${_params.default_inferencer}
      followup_inferencer:
        _target_: ${_params.default_inferencer}
```

`_filter_attrs_keys` (lines 697-711) processes the `_factory`-suffix attrs field `worker_factory`:

```python
for a in attr.fields(cls):
    if not a.name.endswith("_factory") or a.name not in node:
        continue
    val = node[a.name]
    if not isinstance(val, dict):
        continue
    if "_target_" in val:
        if _FACTORY_MARKER in val:                 # ← only TRUE for _import_-tagged
            raw = copy.deepcopy(val)
            del raw[_FACTORY_MARKER]
            del val[_FACTORY_MARKER]
            if _factory_configs is not None:
                _factory_configs.append((a.name, None, raw, _injectables or {}))
        val["_partial_"] = True                    # ← unconditional injection
```

**The bug is in the conditional**: `_factory_configs.append(...)` only fires when `_FACTORY_MARKER` is present, which only happens when `_import_:` was used to load the factory's body from another YAML. For inline `_target_:` factories (the dominant pattern in BTA configs per §2.7's evidence), the recording is SKIPPED, leaving only `val["_partial_"] = True` — which produces a vanilla `functools.partial`.

### Why Vanilla `functools.partial` Causes Cross-Worker Sharing

The walker continues recursing into `flow_configs[*].initial_inferencer` and `flow_configs[*].followup_inferencer` (lines 396-414 of `_instantiate.py`). Hydra's `instantiate()` is then called on the WHOLE pre-processed tree (line 219). Hydra's recursion is depth-first: ALL `_target_` blocks are EAGERLY INSTANTIATED into concrete objects. The outer MFDual node, having `_partial_: true`, is wrapped in `functools.partial(MultiFlowDual, **resolved_kwargs)` — but `resolved_kwargs` already contains the FULLY-INSTANTIATED `flow_configs` list (with concrete `initial_inferencer` / `followup_inferencer` instances).

The `functools.partial` therefore captures these instances by reference in its `keywords` dict. Subsequent calls to the partial (`worker_factory()`) construct a NEW outer `MultiFlowDual` but pass it the SAME `flow_configs` list with the SAME inner instances. **Cross-worker instance sharing is born here.**

### Why The Existing `_ImportFactory` Path Is Correct (And Why It's Underused)

`_ImportFactory.__call__` (lines 49-60 of `_instantiate.py`) does the right thing:

```python
def __call__(self, **_kwargs):
    config = copy.deepcopy(self._config_dict)            # ← deep-copy CONFIG, not instances
    for k, v in self._injectables.items():
        injectable_key = f"_{k}"
        if injectable_key in config:
            config[injectable_key] = copy.deepcopy(v)
    instance = instantiate(OmegaConf.create(config))     # ← re-runs full walker + Hydra
    if self.template_extra_feed and hasattr(instance, "template_extra_feed"):
        instance.template_extra_feed.update(self.template_extra_feed)
    return instance
```

Each call deep-copies the CONFIG DICT (cheap, no fragile attrs/lock state) and re-runs `instantiate()`, producing a freshly-walked, freshly-Hydra-instantiated sub-tree. **No instance sharing across calls — by construction.**

The asymmetry is the bug: today this correctness is gated on `_import_:` directive usage. Per §2.7's YAML topology agent: NO BTA YAML in the codebase uses `_import_:` for factory bodies. So the `_ImportFactory` path is essentially DEAD CODE in production today.

### Why "Deep-Copy Instances At BTA" Is The WRONG Layer (REJECTED)

A naive alternative is to `copy.deepcopy(flow_configs)` inside BTA's `_build_subgraph_spec` after each `factory()` call. v4.4 explicitly REJECTS this:

| Concern | Why deep-copy at BTA fails |
|---|---|
| **`InferencerWorkspace` references** | Workspace cascade depends on SHARED workspace identity (parent-child propagation); deep-copy duplicates workspace objects → propagation breaks silently |
| **`template_manager` references** | Designed as a SHARED singleton (cache hit-rate matters); deep-copy creates per-worker template managers → cache-thrash + duplicate template loads |
| **`_logger` instances** | Python loggers SHOULD be shared (one per logical channel); deep-copy creates duplicate logger references |
| **`asyncio.Lock` / `threading.Lock`** | Locks are NOT deep-copyable in general; they're picklable per Python's recent versions but semantically should be SHARED to provide mutual exclusion. Deep-copy creates per-instance locks → mutual exclusion BROKEN |
| **`attrs` validators** | Some attrs classes have `__attrs_post_init__` side effects (e.g., MFDual snapshots `_review_inferencer_original`); deep-copy bypasses these → state inconsistencies |
| **Weakref-tracked registries** | Several inferencer subclasses register themselves in module-level weakref dicts; deep-copy may duplicate registrations or leave stale entries |
| **Reproducing the YAML topology in code** | Deep-copy must clone the EXACT instance structure including parent links; effectively re-implements the walker manually |

**The principled fix is at the CONFIG layer (re-instantiate from saved config per call) — which is exactly what `_ImportFactory` already does**. Deep-copying instances at BTA would create a different bug class (broken workspace propagation, lock duplication, etc.) and is fundamentally fighting the framework instead of using it correctly.

---

## §2.9 — BTA Recognition Gap: No Formal Factory Protocol [v4.4 NEW]

**Source code site**: `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/breakdown_then_aggregate_inferencer.py:1483-1511`.

Today's BTA distinguishes factory-shaped from non-factory-shaped callables via `isinstance(x, functools.partial)`:

```python
if isinstance(self.worker_factory, dict):
    # heterogeneous (per-task-type)
    factory_entry = self.worker_factory.get(task_type, self.worker_factory.get("__default__"))
    if isinstance(factory_entry, str):
        factory_entry = self.worker_factory.get(factory_entry)
    factory = (factory_entry["factory"]
               if isinstance(factory_entry, dict) and "factory" in factory_entry
               else factory_entry)
    if isinstance(factory, functools.partial):
        worker = factory()                                      # no args
    else:
        worker = factory(sub_query=query_str, index=i)          # WITH args
else:
    # homogeneous
    if isinstance(self.worker_factory, functools.partial):
        worker = self.worker_factory()                          # no args
    else:
        worker = self.worker_factory(sub_query=query_str, index=i)
```

### Why This Recognition Pattern Is Fragile

| Property | Today's mechanism |
|---|---|
| Formal "I am a factory" marker | ❌ NONE — only structural `isinstance(x, functools.partial)` |
| Detection of `_ImportFactory` as a factory | ❌ NONE — `_ImportFactory` would fall into the bare-callable branch and receive `sub_query=` / `index=` kwargs that its `__call__(**_kwargs)` silently ignores → silent kwarg drop |
| Discoverability | ❌ Two scattered `isinstance` branches duplicated for heterogeneous + homogeneous code paths |
| Forward-compat with new factory types | ❌ Adding `LazyConfigFactory` requires touching every `isinstance` site |
| Debuggability | ⚠️ `repr(functools.partial(MultiFlowDual, ...))` shows the partial but not WHY it's a factory; `_ImportFactory.__repr__` is informative but never reached by current code |

### The v4.4 Solution (Detailed In Fix #9)

Introduce a `LazyConfigFactory` marker class that:

1. **Subsumes** `_ImportFactory` (which becomes a deprecated alias).
2. **Extends** the recognized factory-types tuple in BTA: `isinstance(x, (functools.partial, LazyConfigFactory))`.
3. **Documents** the contract: factories are "fully bound, call with no args"; bare callables are "need runtime context (sub_query, index)".
4. **Pairs** with the existing `_factory`-suffix attrs convention so YAML authors don't need new opt-in syntax.

Fix #9 makes the WALKER produce `LazyConfigFactory` for ALL `_factory` fields (not just `_import_`-tagged ones), and Fix #10 makes existing `_ImportFactory` callers seamlessly migrate via deprecated-alias sema## §2.10 — File-Path Correction: `RichPythonUtils` is the Live Codebase [v4.7 NEW]se [v4.7 NEW]

**The error in v4.4-v4.6**: All references to the Hydra walker / `_instantiate.py` pointed to `/Users/tchen7/MyProjects/CoreProjects/RichPythonUtils/src/rich_python_utils/config_utils/_instantiate.py`. This file EXISTS but is a **stale older mirror** (28 KB, last modified Apr 29, 2026). The line numbers cited (e.g., `_partial_: true` at line 959, `_filter_attrs_keys:944-972`) are from this stale copy.

**The actual live codebase**: `/Users/tchen7/MyProjects/CoreProjects/RichPythonUtils/src/rich_python_utils/config_utils/_instantiate.py` (39 KB, last modified May 4, 2026).

**Verification**: `AgentFoundation/src/` imports `from rich_python_utils.config_utils import instantiate`:

| Importer | Line | Confirmation |
|---|---|---|
| `agent_foundation/common/configs/registered_targets.py` | 8 | `from rich_python_utils.config_utils import register_alias` |
| `agent_foundation/common/configs/factories.py` | 16 | `from rich_python_utils.config_utils import instantiate, load_config` |
| `agent_foundation/common/inferencers/mock_inferencers/mock_bta_components.py` | 10 | `from rich_python_utils.config_utils import load_config, instantiate` |

Zero AgentFoundation imports reference `python_utils.config_utils`. The `rovoteam/PythonUtils/` directory is a historical mirror (likely from an earlier Atlassian internal split-out) that no live code consumes.

### Verified Live Line Numbers (`RichPythonUtils/.../_instantiate.py`, 2026-05-10 19:58)

| Symbol | Live line | Stale (v4.4-v4.6) | Δ |
|---|---|---|---|
| `_FACTORY_MARKER` constant | 35 | not cited | n/a |
| `class _ImportFactory:` | 38 | not cited | n/a |
| `_apply_import_factory()` function | 356 | not cited | n/a |
| `def _filter_attrs_keys(...):` | 881 | 697 | +184 |
| `*_factory` handling block (single-factory case) | 944-960 | 697-711 | +247 |
| `val["_partial_"] = True` (single) | 959 | 711 | +248 |
| `*_factory` handling block (dict-of-factories case) | 961-972 | 712-724 | +249 |
| `val["_partial_"] = True` (dict, per-entry `v["_partial_"] = True`) | 972 | 724 | +248 |
| `_factory_configs.append(...)` (single, gated on `_FACTORY_MARKER`) | 957 | not cited | n/a |
| `_factory_configs.append(...)` (dict, gated on `_FACTORY_MARKER`) | 970 | not cited | n/a |

### What Changes Mechanically (Bug + Fix Description Now Pointing At Live Code)

The bug mechanic is unchanged: untagged inline `_target_:` factories ONLY get `val["_partial_"] = True` (lines 959, 972), creating vanilla `functools.partial` with eagerly-instantiated nested children. The fix (Fix #9) is to make `_factory_configs.append(...)` UNCONDITIONAL — drop the `if _FACTORY_MARKER in val:` gate at lines 956-958 and 968-970 so EVERY `*_factory` field captures the raw config for post-Hydra `LazyConfigFactory` replacement.

**Plan A's diagnosis is technically PRECISE on this point** — Plan A correctly cites "lines ~944-970" matching the live file. v4.7 adopts Plan A's correct location and corrects all v4.4-v4.6 line references throughout this plan accordingly.

### Implications For Implementation

- All Fix #9 / Fix #10 file-path references in §3 (Fix #9 + Fix #10), §5 (Implementation Order), §6 (Acceptance Criteria), §7 (Risks) MUST point to `RichPythonUtils/.../_instantiate.py`. The pre-flight grep audit must run against the live tree, NOT the stale mirror.
- Cyclic-import risk (v4.4 risk #18) is RE-VERIFIED against the live file: `_apply_import_factory` at line 356 already does the right thing for `_import_:`-tagged blocks; the same module already imports `instantiate` from itself for that purpose, so adding `LazyConfigFactory` to the same module does not introduce a NEW cyclic-import risk beyond what already exists.
- The acceptance criterion "BTA's `isinstance(x, (functools.partial, LazyConfigFactory))` accepts BOTH legacy and new factory types" (v4.4 row 25) must be tested against `RichPythonUtils.config_utils.LazyConfigFactory`, NOT `python_utils.config_utils.LazyConfigFactory`.

### How This Was Missed In v4.4-v4.6

The earlier investigation invoked an Explore subagent that found `_instantiate.py` in BOTH locations (the two `find_and_replace_code` results both turned up matches). The agent's brief response listed only `python_utils/...` as the citation source, and v4.4 took that at face value without cross-checking imports in the consumer (`AgentFoundation`). v4.7 adds an audit checkpoint: any line-number reference to `_instantiate.py` MUST be paired with a grep proving the consumer imports the SAME path.

---

## §3 The Fixes (v4.7: 10 fixes — #1 through #10; file paths corrected)

### Fix #1 — Aggregator Workspace Reassignment (mirrors Fix #1 from prior plan)

**File**: `multi_flow_inferencer.py` (or `multi_flow_dual_inferencer.py` — wherever aggregator inference is invoked)

**Change**: Before invoking aggregator inference, reassign the aggregator inferencer's workspace to the canonical `aggregator/` slot — using the same `_reassign_role_workspace()` helper that Fix #1 uses for reviewer/fixer. This handles the case where the aggregator inferencer instance was previously bound to a flow workspace.

```python
# In MFI _build_aggregator_input or _run_aggregator:
# Before any aggregator.ainfer() call:
if self.aggregator_inferencer is not None:
    self._reassign_role_workspace(
        role_inferencer=self.aggregator_inferencer,
        role_name="aggregator",
        parent_workspace=self._workspace,
    )
```

The helper:
1. Detects that `aggregator_inferencer._workspace` is set to something other than the canonical `aggregator/` slot
2. Stashes the old workspace (for potential debugging)
3. Reassigns `_workspace` to `parent_workspace.child("aggregator")`
4. Invalidates derived state via the `_workspace` property setter

**Why this is the correct fix**:
- It's the SAME bug class as the reviewer/fixer mutation issue (Fix #1 from the prior plan)
- The pattern is already established and tested
- It restores the canonical workspace layout
- Aggregator output ends up in the right place
- Manifest, logs, cache, and final_deliverables all populate the canonical paths

**Effort**: ~45 min code + ~30 min tests (mirror existing reviewer/fixer tests for aggregator slot)

### Fix #2 — Eliminate double `final_deliverables/final_deliverables/` nesting [v4.8 CORRECTED — was rewriting the API]

**File**: `inferencer_workspace.py` `surface_outputs_from()` (lines 130-191)

**Live API** (verified 2026-05-10 20:06):
```python
def surface_outputs_from(
    self,
    source_workspace: "InferencerWorkspace",
    *,
    namespace: "Optional[str]" = None,
    skip_existing: bool = True,
) -> "List[str]":
    """Copy a source workspace's deliverables into this workspace's deliverables_dir."""
```

The live implementation walks `source_workspace.deliverables_dir` (NOT `outputs_dir`) and supports an optional `namespace` parameter. Earlier plan revisions (pre-v4.8) had a sketch that DROPPED the `namespace` parameter and referenced the wrong attribute (`outputs_dir`) — this would have been an unintentional API break. v4.8 corrects this.

**Change** (single-line, non-breaking — adopts Plan A's elegant fix): During the existing `os.walk(deliverables_dir)` traversal, prune any subdirectory named `final_deliverables` so the recursive copy never descends into a nested `final_deliverables/final_deliverables/` chain.

```python
# In the existing surface_outputs_from(), inside the os.walk loop:
for root_dir, dirs, files in os.walk(src_root):
    # NEW (v4.8): prevent double-nesting final_deliverables/final_deliverables/
    dirs[:] = [d for d in dirs if d != "final_deliverables"]
    for f in files:
        # ... existing copy logic UNCHANGED ...
```

**Why a single-line prune is the correct fix**:
- Preserves the live signature (`namespace`, `skip_existing` kwargs untouched)
- Preserves the live attribute (`deliverables_dir`, NOT `outputs_dir`)
- Preserves the existing return value (list of relative paths copied)
- The mutation `dirs[:] = ...` is the canonical `os.walk` idiom for pruning traversal — well-understood, well-tested
- Zero behavior change for healthy cases (no nested `final_deliverables/`); only blocks the pathological recursion

**Acceptance criteria**:
- F2-1: A source workspace whose `deliverables_dir` contains `final_deliverables/output.md` is copied to dest's `deliverables_dir/output.md` (NOT `deliverables_dir/final_deliverables/final_deliverables/output.md`).
- F2-2: A source workspace WITHOUT a nested `final_deliverables/` produces identical output to pre-fix behavior.
- F2-3: The `namespace` parameter still works (e.g., `surface_outputs_from(src, namespace="archive/")` writes to `dest/deliverables_dir/archive/...`).
- F2-4: The `skip_existing` parameter still works (existing files are not overwritten).
- F2-5: Return value is a non-empty list of relative paths (matches pre-fix contract).

**Effort**: ~30 min code (single-line + tests for nested case) + ~30 min for the 5 acceptance criteria above. Total ~1 hour (was ~75 min in earlier estimate; the simplification reflects the smaller delta).

### Fix #3 — Remove `flow_X_round01` placeholder OR rename per-step output to start at `_round01`

**Two options**:

**Option A (cleaner)**: Don't pre-assign `_round01` placeholder at construction. Have LWI's per-step naming start from `_round01` (step 0's actual output dir).

**File**: `multi_flow_inferencer.py:533` + `linear_workflow_inferencer.py:533-538`

```python
# multi_flow_inferencer.py BEFORE:
("followup_inferencer", f"flow_{i}_round01"),  # ← placeholder, never used

# AFTER (v4.8 CORRECTED — was at risk of `parent.child(None)` crash):
# OPTION A.1 (PREFERRED): Omit the entry entirely from the propagation list
# so the consumer loop never receives a None suffix:
[
    ("base_inferencer", f"flow_{i}_initial"),
    # ("followup_inferencer", ...) entry omitted entirely — LWI computes per-step
]

# OPTION A.2 (alternative): Keep tuple shape (handy for unit-test parity) but
# add an explicit skip-on-None guard at the consumer:
[
    ("base_inferencer", f"flow_{i}_initial"),
    ("followup_inferencer", None),  # marker: "LWI computes this per-step"
]
# At the consumer (the propagation loop):
for attr_name, suffix in propagation_entries:
    if suffix is None:
        # Explicit "skip — populated elsewhere" marker; do NOT call child(None)
        continue
    inferencer = getattr(self, attr_name, None)
    if inferencer is None:
        continue
    inferencer._workspace = parent_workspace.child(suffix)
```

**Why the explicit guard matters**: `InferencerWorkspace.child(name: str)` requires a non-None string (verified at `inferencer_workspace.py:267`). Passing `None` would crash with `TypeError: can only join str (not "NoneType") to str` deep inside the workspace path resolution — a confusing error far from the propagation site.

**Recommendation**: Use Option A.1 (omit entirely). It's strictly simpler than A.2's marker+guard pair and removes the shape-mismatch trap altogether. Option A.2 is documented for codebases where unit tests assert tuple-list length (in which case the explicit `None` marker preserves the contract while the guard prevents the crash).

```python
# linear_workflow_inferencer.py: per-step workspace computation
def _per_step_workspace(self, step_index):
    parent = self._base_followup_workspace
    # Step 0 → flow_X_round01 (was _round02 before fix)
    # Step 1 → flow_X_round02 (was _round03 before fix)
    return parent.child(f"{base_name}_round{step_index + 1:02d}")
```

**Option B (less disruptive)**: Delete the empty `_round01` dir after construction if it remains empty.

**Recommendation**: **Option A** — root-cause fix; aligns with "no surprises" philosophy.

**Effort**: ~1 hour code + ~30 min tests (need test for first-step naming)

### Fix #6 — Role-Mutation Template Refresh [CRITICAL — Anomaly 5] — v4.1: Subsumed by layered `switch_role()`

**Status**: v4.1 restructure — the parallel template-mutation logic in `_reassign_role_workspace()` is REMOVED; the orchestrator instead invokes the new layered `switch_role()` API documented in **§4 Architectural Refactor**. The bug fix is now ONE call from `_step_propose_impl` that, via Python's MRO, dispatches to `TemplatedInferencerBase.switch_role()` (which applies template attribs and calls `super().switch_role()` for base-level attribs in one atomic transition with one merged audit entry).

**Files**: `multi_flow_dual_inferencer.py` (call-site migration only — all attribute-mutation mechanics live on `InferencerBase`).

**Approach**: Hybrid Option A (explicit MFDual-level override) + Option C (SLOT_DEFAULTS fallback) is preserved as the *resolution policy* on the orchestrator side; the *application* of resolved values is delegated to `switch_role()`. This separation of concerns is the v4.0 contribution.

#### New attribs on MFDual (unchanged from v3.0)

```python
# Optional MFDual-level overrides (Option A — power-user customization).
# When None, fall back to canonical SLOT_DEFAULTS (Option C).
review_template_key: Optional[str] = attrib(default=None)
review_template_root_space: Optional[str] = attrib(default=None)
followup_template_key: Optional[str] = attrib(default=None)
followup_template_root_space: Optional[str] = attrib(default=None)
```

#### MFDual.`_resolve_role_template()` — value resolver only (no mutation)

```python
def _resolve_role_template(
    self, role_name: str
) -> tuple[Optional[str], Optional[str]]:
    """Resolve (template_key, template_root_space) for a runtime role assignment.

    Three-tier resolution (first non-None wins):
      1. Explicit MFDual override (review_template_key / followup_template_key)
      2. Canonical SLOT_DEFAULTS (REVIEW_TEMPLATE_DEFAULTS / FOLLOWUP_TEMPLATE_DEFAULTS)
      3. None (caller preserves original — backward compat via switch_role())

    Pure function — does NOT mutate any inferencer. Mutation is delegated
    to `InferencerBase.switch_role()` so the policy lives here and the
    mechanism lives on the base class.
    """
    from agent_foundation.common.inferencers.template_defaults import (
        REVIEW_TEMPLATE_DEFAULTS, FOLLOWUP_TEMPLATE_DEFAULTS,
    )
    if role_name in ("reviewer", "review_inferencer"):
        explicit_key = self.review_template_key
        explicit_root = self.review_template_root_space
        defaults = REVIEW_TEMPLATE_DEFAULTS
    elif role_name in ("fixer", "fixer_inferencer"):
        explicit_key = self.followup_template_key
        explicit_root = self.followup_template_root_space
        defaults = FOLLOWUP_TEMPLATE_DEFAULTS
    else:
        return (None, None)

    return (
        explicit_key or (defaults.template_key if defaults else None),
        explicit_root or (defaults.template_root_space if defaults else None),
    )
```

#### `_reassign_role_workspace()` — REPLACED by single `switch_role()` call

```python
def _reassign_role_workspace(self, inferencer, role_name: str) -> None:
    """v4.0: Thin wrapper that prepares (workspace, template_key,
    template_root_space) and forwards them to switch_role().

    Preserved as a name for backward compatibility with existing call sites
    AND because the identity-guard policy ("don't clobber YAML-configured
    originals") is MFDual-specific — it doesn't belong on InferencerBase.

    The MECHANICS of attribute application — workspace setter cascade, cache
    invalidation, audit logging, parallel-safety guard — all live on
    InferencerBase.switch_role().
    """
    if inferencer is None or self._workspace is None:
        return
    # Identity guard (preserved from prior session): never re-bind a YAML-
    # configured original; switch_role() is opt-in per role transition.
    original = getattr(self, f"_{role_name}_original", None)
    if inferencer is original:
        return

    role_ws = self._workspace.child(role_name)
    role_ws.ensure_dirs()
    new_key, new_root = self._resolve_role_template(role_name)

    inferencer.switch_role(
        new_role=role_name,
        workspace=role_ws,
        template_key=new_key,
        template_root_space=new_root,
        # output_is_deliverable / template_extra_feed: NOT supplied here —
        # MFDual sets output_is_deliverable separately at fixer dispatch
        # time (see _select_reviewer_and_fixer); switch_role() preserves
        # any value not explicitly provided. (See §4 "implicit-preserve"
        # semantics.)
    )
    # `reset_session` semantics are preserved by switch_role()'s
    # post-hook protocol — see §4 "Subclass extensibility".
```

#### Edge cases handled (still relevant; v4.0 mechanics preserve all v3.0 guarantees)

| Case | Behavior |
|---|---|
| Standard MFDual, no overrides | `_resolve_role_template` returns canonical `"review"` / `"followup"` → `switch_role(template_key=...)` applies them — fixes the bug |
| MFDual with `review_template_key: "custom_review"` | Resolver returns `"custom_review"` (Option A power) — `switch_role()` applies as-is |
| Inferencer without `template_key` attrib (non-templated leaf) | `switch_role()`'s `hasattr` guard skips silently — no error |
| Original `review_inferencer` configured at construction | Identity guard in `_reassign_role_workspace` short-circuits BEFORE `switch_role()` is called → template stays as YAML-configured |
| Explicit empty string override (`review_template_key: ""`) | Falls through to defaults via `or` (consistent with v3.0) |
| Future role-relevant attribute added (e.g., `output_is_deliverable` reset on every transition) | Add it as a `switch_role()` kwarg; ALL orchestrators benefit automatically — no per-orchestrator helper change |

#### Why v4.0 is the elegant proper fix (delta over v3.0)

| v3.0 | v4.0 |
|---|---|
| Mutation logic inside `_reassign_role_workspace()` (MFDual-specific) | Mutation logic on `InferencerBase.switch_role()` (universally reusable) |
| Each orchestrator must replicate the `hasattr` + setter pattern | Single source of truth — base method enforces uniform application |
| Adding new role-relevant attribute = touching every orchestrator | Adding new role-relevant attribute = adding ONE kwarg to `switch_role()` |
| No audit trail of role transitions | `_role_history` audit list on every inferencer (postmortem-friendly) |
| Subsumes Fix #6 only; no symmetry across DualInferencer | DualInferencer's similar swap can call the same method (free) |

#### Acceptance criteria (unchanged from v3.0; tests assert post-conditions on the inferencer state)

| # | Criterion | Test |
|---|---|---|
| 1 | After winner-as-fixer assignment, `fixer_inferencer.template_key == "followup"` | Unit: `test_winner_as_fixer_inherits_followup_template` |
| 2 | After loser-as-reviewer assignment, `review_inferencer.template_key == "review"` | Unit: `test_loser_as_reviewer_inherits_review_template` |
| 3 | When MFDual sets `followup_template_key: "custom"`, fixer uses `"custom"` | Unit: `test_explicit_override_wins_over_defaults` |
| 4 | YAML-configured original reviewer/fixer NOT touched (identity guard) | Unit: `test_original_role_inferencer_preserved` |
| 5 | After end-to-end run, fixer's actual rendered prompt contains `<ReviewerFeedback>` and `prior_output_path` | Integration: SOP plan run + grep |
| 6 | After end-to-end run, reviewer's actual rendered prompt contains `<ProposedDocument>` and verdict format | Integration: SOP plan run + grep |
| 7 | **(v4.0)** `inferencer._role_history[-1]["to_role"] == "fixer_inferencer"` after winner-as-fixer assignment | Unit: `test_switch_role_records_audit_trail` |
| 8 | **(v4.0)** Calling `switch_role()` with `template_key=None` preserves the existing `template_key` | Unit: `test_switch_role_preserve_when_kwarg_none` |
| 9 | **(v4.0)** `switch_role()` cascades workspace via the existing `_workspace` setter (cache invalidation + child propagation) | Unit: `test_switch_role_workspace_cascades_via_setter` |

#### Open questions for implementation (carried forward)

1. **Explicit empty string `""` semantics**: Treat as "fall through to defaults" or "explicitly preserve original"? Recommended: fall through (consistent with `or` pattern).
2. **Should we also reset `template_extra_feed`?** v4.0 enables this via the `template_extra_feed` kwarg on `switch_role()` — orchestrator chooses by passing or omitting; current MFDual omits.
3. **What if `REVIEW_TEMPLATE_DEFAULTS.template_key` is None in a future codebase change?** `switch_role()` gracefully preserves the existing value (no mutation). Same safe default.

#### Effort: subsumed under §4 (~30 min for the call-site migration AFTER §4 lands)

---

### Fix #5 — BTA Worker Instance Diagnostics + Sharing Detection [NEW — HIGH-VALUE INFRASTRUCTURE]

**Motivation**: This investigation thread spent 5+ rounds speculating about whether worker_0 and worker_1 share a Python instance for the MultiFlowDual or its children (aggregator, flow inferencers). Hard evidence to answer this was missing because BTA does not log instance identity. Without this evidence, all diagnoses for Anomaly 1 remain unverifiable.

This fix adds two complementary pieces:

#### Part A — Diagnostic Logging (zero behavior change, IMMEDIATE value)

**File**: `breakdown_then_aggregate_inferencer.py`

**Change**: After each `worker = factory(...)` call in `_make_worker_iter`, log the produced instance's identity:

```python
# After: worker = factory(...) (around line 1510)
_logger.info(
    "BTA[%s] worker[%d] type=%s id=0x%x has_workspace=%s",
    getattr(self, "name", "?"),
    i,
    type(worker).__name__,
    id(worker),
    getattr(worker, "_workspace", None) is not None,
)

# Optionally, recursively log key children (aggregator, flow inferencers):
for child_attr in ("aggregator_inferencer", "multi_flow_aggregator_inferencer", "base_inferencer"):
    child = getattr(worker, child_attr, None)
    if child is not None:
        _logger.info(
            "  ↳ %s.%s id=0x%x type=%s",
            type(worker).__name__, child_attr,
            id(child), type(child).__name__,
        )
```

**Effect**: A single log line per worker tells us definitively whether instances are shared. This investigation thread (5+ rounds of speculation) would have ended in 30 seconds with this in place.

#### Part B — Recursive Sharing Detection (warn-only, simple boolean toggle) [v4.6 INTEGRATED]

**v4.6 design rationale (integration of v4.5 + integration-memo simplification)**:

The original (pre-v4.5) Part B sketch checked ONLY top-level `id(worker)` — which would PASS for the actual Anomaly 6 sharing pattern (where each freshly-constructed outer MFDual has shared `flow_configs[*].initial_inferencer` children). v4.5 correctly identified this and made the detection RECURSIVE, but over-engineered the public API with two enums (`SharingScope` / `SharingPolicy`) producing 6 combinations, several of which had no use case. v4.6 integrates the v4.5 architectural correctness (recursive walk reusing the existing `_iter_child_inferencers()` infrastructure that BTA / MultiFlow / Dual / MFDual already override) with the integration-memo's simplification philosophy:

| v4.5 design | v4.6 integrated design | Rationale |
|---|---|---|
| `SharingScope` enum (NONE / TOP_LEVEL / RECURSIVE) | DROPPED | TOP_LEVEL is dead semantic surface (no use case after Fix #9); NONE collapses to the boolean's False state |
| `SharingPolicy` enum (STRICT / WARN) | DROPPED, WARN-only | STRICT-by-default risks breaking legitimate-sharing test fixtures during rollout; YAGNI on escalation until WARN tells us it's needed |
| `_sharing_anomaly.diagnostic.txt` file | DROPPED | `_logger.warning()` is the codebase's standard observability surface; file proliferation is noise in CI/test loops |
| `_verify_worker_sharing()` method | RENAMED to `_validate_worker_isolation()` | "Isolation" is the property being verified (per integration-memo); "sharing" is the failure mode |
| `worker_sharing_check_scope` + `worker_sharing_policy` attribs | SINGLE attrib `worker_isolation_check: bool = True` | One toggle, sensible default, future-extensible |
| Call site at "end of `_make_worker_iter()`" (which doesn't exist) | Call site at end of `_build_subgraph_spec()` (verified at line 1804) | Critical correction: BTA has no `self.workers` attribute |

Retains v4.5's two architecturally-correct decisions: (1) the base-class helper `InferencerBase._collect_all_descendant_inferencers()` (NOT static-on-BTA — helper is universal so lives on the type that defines `_iter_child_inferencers`), (2) reuses the established `_iter_child_inferencers()` overrides without modifying any orchestrator (audited: BTA `breakdown_then_aggregate_inferencer.py:1178`, MultiFlow `multi_flow_inferencer.py:991`, Dual `dual_inferencer.py:1943`, MFDual `multi_flow_dual_inferencer.py:632`).

**Future escalation path** (documented for clarity, NOT implemented in v4.6): If WARN-mode logs reveal that legitimate sharing is rare and accidental sharing is common, a future revision MAY add `worker_isolation_strict: bool = False` to escalate WARN → raise. Until that signal exists, this is YAGNI.

##### §F5B.1 — New base-class helper: `InferencerBase._collect_all_descendant_inferencers`

**Location**: `inferencer_base.py`, immediately after `_iter_child_inferencers` (line 570) and BEFORE `pre_retry` (line 589), so the recursive helper sits between the per-layer iterator and the pre-retry consumer that already uses the same `_seen` cycle-safety pattern.

```python
def _collect_all_descendant_inferencers(
    self, _seen: Optional[Set[int]] = None,
) -> Iterator["InferencerBase"]:
    """Recursively yield self + all descendant inferencers, deduped by id().

    Mirrors the recursion + cycle-safety pattern of ``pre_retry`` (line 589),
    but exposes a pure generator over the descendant set rather than calling
    a per-instance hook.

    Cycle safety: An ``_seen: set[int]`` accumulator (using ``id()`` of each
    visited inferencer) prevents infinite loops when the descendant graph
    contains cycles or self-references. The caller MAY pass ``_seen`` to
    continue an existing traversal (e.g., when sweeping multiple roots);
    if ``_seen`` is None a fresh set is created.

    Yields ``self`` FIRST, then recursively yields children of each
    direct child via ``_iter_child_inferencers()``. Children that have
    already been yielded (by id) are skipped.

    Used by:
      - ``BreakdownThenAggregateInferencer._verify_worker_sharing()`` (Fix #5
        Part B in v4.5) — recursive cross-worker sharing detection.
      - Any future tool that needs "all instances reachable from this root".
    """
    if _seen is None:
        _seen = set()
    if id(self) in _seen:
        return
    _seen.add(id(self))
    yield self
    for child in self._iter_child_inferencers():
        # Re-uses _seen so cross-tree visits are properly deduped when the
        # caller invokes this helper repeatedly with shared accumulator.
        yield from child._collect_all_descendant_inferencers(_seen=_seen)
```

##### §F5B.2 — BTA attrib (replaces v4.5's two-enum design)

**Location**: in BTA's attrs class declaration block (alongside other BTA configuration attribs like `aggregator_inferencer`).

```python
worker_isolation_check: bool = attrib(
    default=True,
    kw_only=True,
    metadata={
        "description": (
            "When True (default), validate after each worker construction "
            "that no inferencer instance is shared across workers in the "
            "same BTA dispatch (recursive walk via "
            "InferencerBase._collect_all_descendant_inferencers). "
            "Sharing is logged as a WARNING (not raised) — set False to "
            "skip the check entirely (e.g., for advanced setups that "
            "intentionally share stateless inferencers)."
        ),
    },
)
```

**Back-compat**: There is no prior `allow_worker_sharing` attrib in the codebase (audit verified 2026-05-10 19:36 — zero references found). v4.6 introduces the new `worker_isolation_check` cleanly without back-compat aliasing.

##### §F5B.3 — Validation method on BTA (replaces v4.5's `_verify_worker_sharing`)

```python
def _validate_worker_isolation(
    self,
    workers: Sequence["InferencerBase"],
) -> None:
    """v4.6: Detect shared inferencer instances across BTA workers and warn.

    Called from the END of ``_build_subgraph_spec()`` (after the worker
    construction loop at lines 1499-1512), receiving the locally-built
    list of worker instances. Recursively walks each worker's descendant
    tree via ``InferencerBase._collect_all_descendant_inferencers()``
    (which mirrors ``pre_retry``'s ``_seen``-based cycle-safety pattern
    at ``inferencer_base.py:589``) and logs a warning for each
    cross-worker collision detected.

    Args:
        workers: The list of worker instances built during this BTA
            dispatch's ``_build_subgraph_spec`` call. Order matches
            sub-query order; indices are reported in warning messages.

    Behavior:
        - If ``self.worker_isolation_check`` is False → skip entirely.
        - Otherwise, build a single ``seen: dict[int, int]`` mapping
          ``id(inferencer) → first_worker_index`` across the union of
          all descendant trees. On each duplicate ``id``, emit ONE
          ``_logger.warning`` with worker indices, instance id, type,
          and remediation hint (root cause is almost always missing
          LazyConfigFactory wiring, i.e., Fix #9 not yet landed for
          a particular config field).

    Why warn-only (not raise): During the phased rollout of Fix #9,
        legitimate-sharing test fixtures may exist; raising would
        break them. The codebase's standard observability path
        (``_logger.warning``) is sufficient. If WARN logs accumulate
        in production after Fix #9 lands, a future revision MAY
        introduce ``worker_isolation_strict: bool`` to escalate.
        That escalation is YAGNI today.
    """
    if not self.worker_isolation_check:
        return

    seen: dict[int, int] = {}  # id → first_worker_index
    for i, worker in enumerate(workers):
        if not isinstance(worker, InferencerBase):
            # Defensive: if a future worker_factory returns a non-
            # InferencerBase wrapper (e.g., a callable wrapper), skip
            # it but log so the gap is visible.
            _logger.debug(
                "BTA._validate_worker_isolation: worker[%d] type=%s is "
                "not an InferencerBase; skipping descendant walk.",
                i, type(worker).__name__,
            )
            continue
        for inf in worker._collect_all_descendant_inferencers():
            iid = id(inf)
            if iid in seen and seen[iid] != i:
                _logger.warning(
                    "BTA[%s] worker[%d] shares inferencer instance with "
                    "worker[%d]: id=0x%x type=%s. Likely cause: "
                    "missing LazyConfigFactory wiring for a `*_factory` "
                    "field (Fix #9). Set worker_isolation_check=False "
                    "if sharing is intentional.",
                    getattr(self, "name", type(self).__name__),
                    i, seen[iid], iid, type(inf).__name__,
                )
            else:
                seen[iid] = i
```

##### §F5B.4 — Call site (corrected for v4.6)

**Critical correction over v4.5**: BTA does NOT have `_make_worker_iter()` or `self.workers`. Workers are constructed per-inference-call inside `_build_subgraph_spec()` at lines 1499-1512. The validation must happen there, AFTER the worker-construction loop and BEFORE the SubgraphSpec is returned.

```python
# In _build_subgraph_spec (around line 1499-1512):
workers: list["InferencerBase"] = []
for i, query_str in enumerate(sub_queries):
    if isinstance(self.worker_factory, dict):
        # ... existing dict-of-factories dispatch ...
    elif isinstance(factory, (functools.partial, LazyConfigFactory)):  # v4.4 Fix #9
        worker = factory()
    else:
        worker = factory(sub_query=query_str, index=i)
    workers.append(worker)

# v4.6 Fix #5 Part B: validate worker isolation across the just-built tree.
self._validate_worker_isolation(workers)

# ... continue with SubgraphSpec construction (line 1513+) ...
```

##### §F5B.5 — Why this fits the codebase's architectural philosophy

| Existing principle | How v4.6 Fix #5 Part B aligns |
|---|---|
| "Diagnose first, escalate later" (BTA's existing `_normalize_aggregator_output` warn-then-fail logic) | WARN-only in v1; future-extensible to strict via `worker_isolation_strict` |
| `*_factory` suffix infrastructure | The warning message names the likely root cause (missing LazyConfigFactory wiring) |
| Type-safe iteration via base-class helper | `_collect_all_descendant_inferencers()` is type-annotated; no `hasattr` defensiveness needed |
| `pre_retry` recursion + `_seen` cycle-safety pattern (`inferencer_base.py:589`) | Helper replicates the established pattern verbatim |
| Existing `_iter_child_inferencers()` overrides on BTA / MultiFlow / Dual / MFDual | Recursive walk uses these overrides without modifying any orchestrator |
| Standard observability via `_logger.warning` (no separate diagnostic-file conventions) | One log line per collision; integrates with existing log aggregation |

##### §F5B.6 — Acceptance criteria for v4.6

| # | Criterion | Test |
|---|---|---|
| F5B-1 | `_collect_all_descendant_inferencers()` on a leaf inferencer yields exactly `[self]` | Unit: `test_collect_descendants_leaf_yields_self` |
| F5B-2 | `_collect_all_descendant_inferencers()` on an MFDual yields all flow_configs entries (initial + followup) plus aggregator, deduped by id | Unit: `test_collect_descendants_mfdual_yields_full_tree` |
| F5B-3 | Circular reference (A.children=[B], B.children=[A]) terminates via `_seen`; each instance yielded exactly once | Unit: `test_collect_descendants_handles_cycles` |
| F5B-4 | Self-reference (A.children=[A]) terminates; A yielded exactly once | Unit: `test_collect_descendants_handles_self_reference` |
| F5B-5 | BTA invoking `_build_subgraph_spec` with two sub-queries whose workers share `flow_configs[0].initial_inferencer` produces ONE `_logger.warning` per collision (not raise) under default `worker_isolation_check=True` | Unit: `test_recursive_sharing_logs_warning_per_collision` |
| F5B-6 | Warning message contains: BTA name, both worker indices, instance id (hex), instance type name, remediation hint mentioning LazyConfigFactory / Fix #9 | Unit: `test_warning_message_format_and_content` |
| F5B-7 | Setting `worker_isolation_check=False` skips the check entirely (no descendant walk, no warnings, no perf cost) | Unit: `test_worker_isolation_check_false_skips_entirely` |
| F5B-8 | After Fix #9 (LazyConfigFactory) lands, default `worker_isolation_check=True` produces ZERO warnings for the SOP plan integration test (no false positives) | Integration: SOP plan run with default policy; assert log contains zero "shares inferencer instance" warnings |
| F5B-9 | When MULTIPLE shared instances exist across workers, ALL collisions are logged (one warning per collision); no batching that hides collisions | Unit: `test_multiple_collisions_all_logged` |
| F5B-10 | Worker indices in the warning message are correct: `worker[i]` is the LATER detection; `worker[prev_i]` is the FIRST detection | Unit: `test_collision_worker_indices_correctly_attributed` |
| F5B-11 | A non-`InferencerBase` worker (e.g., a Mock or callable wrapper from a future factory variant) is gracefully skipped with a `_logger.debug` message — does NOT crash the validator | Unit: `test_non_inferencer_worker_gracefully_skipped` |

##### §F5B.7 — Effort breakdown (v4.6 simplified)

| Item | Estimate |
|---|---|
| Add `_collect_all_descendant_inferencers` helper to `InferencerBase` + tests F5B-1..F5B-4 | 30 min |
| Add `worker_isolation_check: bool` attrib to BTA | 5 min |
| Implement `_validate_worker_isolation()` method on BTA | 20 min |
| Wire into `_build_subgraph_spec()` after the worker-construction loop (around line 1512) | 5 min |
| Tests F5B-5..F5B-11 (7 unit tests + 1 integration) | 1.25 hours |
| Update Fix #5 docstring + add cross-reference to LazyConfigFactory (Fix #9) and `_iter_child_inferencers` | 10 min |
| Audit grep for any pre-existing `worker_isolation_*` callers in tests/configs (back-compat sweep) | 5 min |
| **Total Fix #5 Part B v4.6 simplified** | **~2.25 hours** (was ~3h in v4.5; the −45 min savings come from dropping enum proliferation + diagnostic-file plumbing while keeping the architecturally-correct recursive base helper) |

##### §F5B.8 — Phased rollout (v4.6)

| Phase | Configuration | Use case |
|---|---|---|
| 1. Land Fix #9 (LazyConfigFactory) FIRST | n/a | Eliminates the root cause structurally. After this, default `worker_isolation_check=True` should produce zero warnings. |
| 2. Land Fix #5 Part B v4.6 with default `worker_isolation_check=True` | Default | Becomes the regression-detection log line. Zero warnings = healthy baseline. |
| 3. If a legitimate-sharing test fixture is discovered | Set `worker_isolation_check=False` on that fixture only | Surgical opt-out. |
| 4. If WARN logs accumulate in production over a release window | OPTIONAL future PR adds `worker_isolation_strict: bool = False` | Escalate to raise if signal warrants — strictly YAGNI today. |

**Caveat (carried forward from v4.4)**: With Fix #9 in place, this check should never fire in healthy runs — it serves as **regression-detection insurance** against future code paths that bypass the `LazyConfigFactory` pathway (e.g., directly mutating `_factory_configs` in tests, or constructing a BTA with manually-injected pre-instantiated workers). The warning message tells the user exactly what likely went wrong.

---

### Fix #7 — Audit Symlink Hardening [v4.2 NEW — for Anomaly 6]

**File**: `src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/dual_inferencer.py` `_record_round_audit()` (lines 712-757).

**Goal**: Make the audit layer DETECT pathological symlink targets (cross-MFDual leakage, within-worker role aliasing, duplicate-phase-overwrite races) and emit LOUD diagnostic markers — so postmortem inspection sees the bug at the audit symlink itself, not 5 levels deep into a wrong-target directory.

**Approach**: Three layered checks, each adding a sibling diagnostic file (`.diagnostic.txt`) WITHOUT removing the symlink itself (preserves backward compat for tooling that follows the symlink).

```python
def _record_round_audit(self, round_idx, phase, inferencer, extra=None):
    if not self.enable_round_audit or self._workspace is None:
        return
    if getattr(inferencer, "_workspace", None) is None:
        return
    try:
        # ... existing log_entry + outputs_dir code unchanged ...

        children_dir = getattr(self._workspace, "children_dir", None)
        if not children_dir:
            return
        nav_dir = os.path.join(children_dir, f"round_{round_idx:02d}")
        os.makedirs(nav_dir, exist_ok=True)
        link_path = os.path.join(nav_dir, phase)
        target = str(inferencer._workspace.root)

        # ───── v4.2 Check 1: Cross-MFDual leakage detection ─────
        # The target SHOULD live under self._workspace.root (this MFDual's tree).
        # If it doesn't, the inferencer's _workspace was mutated by a SIBLING
        # MFDual (instance sharing — Anomaly 1/6). Emit a loud diagnostic.
        my_root = str(self._workspace.root).rstrip("/") + "/"
        target_norm = target.rstrip("/") + "/"
        if not target_norm.startswith(my_root):
            diagnostic = (
                f"# AUDIT ANOMALY [cross-MFDual leakage] round={round_idx} phase={phase}\n"
                f"# Inferencer instance id=0x{id(inferencer):x} type={type(inferencer).__name__}\n"
                f"# Symlink target lives OUTSIDE this MFDual's workspace tree:\n"
                f"#   this MFDual root: {self._workspace.root}\n"
                f"#   target           : {target}\n"
                f"# This typically means the same Python instance is shared with\n"
                f"# another worker's MFDual (BTA factory misconfiguration — see Fix #5).\n"
                f"# The instance's _workspace was last mutated by the OTHER MFDual.\n"
                f"# Audit symlink points to wherever the instance currently lives,\n"
                f"# which is NOT where THIS round's {phase} phase actually ran.\n"
            )
            with open(os.path.join(nav_dir, f"{phase}.LEAKAGE.diagnostic.txt"), "w") as f:
                f.write(diagnostic)
            logger.error(
                "Round audit: cross-MFDual leakage detected at %s/round_%02d/%s "
                "(target=%s lives outside %s)",
                children_dir, round_idx, phase, target, self._workspace.root,
            )

        # ───── v4.2 Check 2: Within-worker role aliasing detection ─────
        # If a previous symlink in this round_NN/ already points to the SAME
        # target with a DIFFERENT phase name, two roles are aliased to the same
        # instance (Bug 6b). Emit a diagnostic listing the conflict.
        if os.path.isdir(nav_dir):
            for existing_name in os.listdir(nav_dir):
                if existing_name == phase:
                    continue
                if existing_name.endswith(".diagnostic.txt") or existing_name.endswith(".pointer.txt"):
                    continue
                existing_path = os.path.join(nav_dir, existing_name)
                if os.path.islink(existing_path):
                    existing_target = os.readlink(existing_path)
                    if existing_target.rstrip("/") == target.rstrip("/"):
                        diagnostic = (
                            f"# AUDIT ANOMALY [within-worker role aliasing] round={round_idx}\n"
                            f"# Phase '{phase}' and phase '{existing_name}' BOTH symlink to:\n"
                            f"#   {target}\n"
                            f"# This means self.{phase}_inferencer IS self.{existing_name}_inferencer\n"
                            f"# (same Python instance assigned to two roles).\n"
                            f"# Likely cause: _select_reviewer_and_fixer() resolved both slots to\n"
                            f"# the same instance (e.g., review_default==winner with empty pool, OR\n"
                            f"# inferencer_pool has only one templated CLI).\n"
                            f"# Effect: the LAST _reassign_role_workspace() call wins; the role\n"
                            f"# assigned earlier silently writes into the LATER role's workspace.\n"
                        )
                        with open(os.path.join(nav_dir, f"{phase}_vs_{existing_name}.ALIASING.diagnostic.txt"), "w") as f:
                            f.write(diagnostic)
                        logger.error(
                            "Round audit: within-worker role aliasing at round_%02d "
                            "(phase=%s aliases existing phase=%s, both → %s)",
                            round_idx, phase, existing_name, target,
                        )

        # ───── v4.2 Check 3: Duplicate-phase overwrite warning ─────
        # The original code silently re-creates the symlink. Detect when this happens
        # AND the target changed — that's a bug (audit shouldn't reuse phase names within
        # a round; if it does and target differs, the LATER value silently wins).
        if os.path.islink(link_path):
            old_target = os.readlink(link_path)
            if old_target.rstrip("/") != target.rstrip("/"):
                diagnostic = (
                    f"# AUDIT ANOMALY [phase overwrite with different target] round={round_idx} phase={phase}\n"
                    f"# Previous symlink target: {old_target}\n"
                    f"# New      symlink target: {target}\n"
                    f"# This means _record_round_audit() was called twice for the same\n"
                    f"# (round, phase) with two different inferencer instances — a logic bug.\n"
                )
                with open(os.path.join(nav_dir, f"{phase}.OVERWRITE.diagnostic.txt"), "w") as f:
                    f.write(diagnostic)
                logger.error(
                    "Round audit: duplicate-phase overwrite at round_%02d/%s (was→%s now→%s)",
                    round_idx, phase, old_target, target,
                )
            os.unlink(link_path)
        try:
            os.symlink(target, link_path, target_is_directory=True)
        except (OSError, NotImplementedError):
            pointer = os.path.join(nav_dir, f"{phase}.pointer.txt")
            with open(pointer, "w") as f:
                f.write(f"# Workspace pointer\n# Target: {target}\n")
    except Exception as exc:
        logger.warning("Round audit for %s/%s failed: %s", round_idx, phase, exc)
```

**Behavior**:
- The symlink itself is still created (preserves backward compat).
- Three sibling diagnostic files are written when anomalies are detected:
  - `<phase>.LEAKAGE.diagnostic.txt` — cross-MFDual leakage
  - `<phase>_vs_<other_phase>.ALIASING.diagnostic.txt` — within-worker role aliasing
  - `<phase>.OVERWRITE.diagnostic.txt` — duplicate-phase overwrite with different target
- Each anomaly also logs at `ERROR` level so it surfaces in stderr / alerting.
- All checks fail-safe: any exception in the diagnostic logic is caught and logged as a warning, not propagated (matches the existing `_record_round_audit` fail-safe contract at line 715).

**Effort**: ~45 min code + ~30 min tests (synthetic test fixtures that pass shared-instance MFDual or aliased reviewer/fixer to verify each diagnostic fires).

### Fix #8 — Snapshot-at-Phase-Execution-Time Audit Semantics [v4.2 NEW — for Anomaly 6 Bug 6c]

**File**: same — `_record_round_audit()` + the three call-sites in `dual_inferencer.py:1088`, `1269`, `1411` + `multi_flow_dual_inferencer.py:678`.

**Goal**: Decouple "where the inferencer's `_workspace` points NOW" from "where this phase actually ran". Snapshot the workspace AT THE PHASE-EXECUTION moment, not at audit-record time.

**Why this matters even with Fix #5 + Fix #7 in place**:
- Fix #5 prevents instance sharing at construction time → eliminates the root cause.
- Fix #7 detects the symptom and writes diagnostics → user sees the bug clearly.
- BUT: even in the absence of sharing, if `switch_role()` runs between phase-execution and audit-record (e.g., `_reassign_role_workspace` for the NEXT round of the SAME MFDual), the audit symlink would still snapshot the wrong path. Fix #8 hardens against this independent timing class.

**Approach**: Capture `inferencer._workspace.root` EAGERLY (immediately after the phase completes), pass the captured path into `_record_round_audit()` as an explicit argument. Audit no longer dereferences the live `_workspace` slot.

```python
# NEW signature:
def _record_round_audit(
    self, round_idx, phase, inferencer,
    *,
    workspace_root_at_phase: Optional[str] = None,   # NEW: explicit snapshot
    extra=None,
):
    """... if workspace_root_at_phase is None, fall back to live read (back-compat)."""
    ...
    target = workspace_root_at_phase or str(inferencer._workspace.root)
    ...
```

```python
# CALL-SITE migration in _step_review_impl (dual_inferencer.py:1269):
# BEFORE:
self._record_round_audit(total_iters, "review", self.review_inferencer)

# AFTER:
review_ws_at_phase = (
    str(self.review_inferencer._workspace.root)
    if getattr(self.review_inferencer, "_workspace", None)
    else None
)
# ... await self.review_inferencer.ainfer(...) ...
self._record_round_audit(
    total_iters, "review", self.review_inferencer,
    workspace_root_at_phase=review_ws_at_phase,
)
```

Same pattern at the propose, fix, and review_dispatch call sites.

**Edge cases**:

| Case | Behavior |
|---|---|
| Single-instance, no sharing, no concurrent mutation | Eager snapshot == live read — no observable difference; cost is one extra string copy per round per phase |
| Concurrent sibling MFDual mutates the shared instance between phase and audit | Audit snapshot preserves the CORRECT path the phase ACTUALLY used; symlink is accurate |
| `switch_role()` for the NEXT round runs before this round's audit completes | Same as above — snapshot preserves history |
| Inferencer's `_workspace` was None at phase time | `workspace_root_at_phase` is None → fall through to live read → same as today |
| Caller forgot to pass `workspace_root_at_phase` (e.g., a future call site) | Falls through to live read with same behavior as today (fail-safe) — but Fix #7's Check 1 would still catch the resulting wrong target if instance sharing was active |

**Why this is the correct semantic**:

The audit's purpose is to record **what happened during a phase**. The phase-execution moment is the only moment when "where this work ran" is unambiguously knowable. Reading the workspace AT AUDIT TIME is an optimisation that works ONLY when no concurrent mutation occurs — an assumption violated by Anomaly 6's evidence.

**Composition with Fix #7**: Fix #7's Check 1 (cross-MFDual leakage) becomes more precise — it now compares the SNAPSHOT path against `self._workspace.root`, which is the MFDual's own tree. If the snapshot lives outside that tree, that's because the inferencer was ALREADY in a foreign workspace at phase-execution time (= active sharing during the phase) — a stronger and earlier-detected signal than today's "moved between phase and audit" signal.

**Effort**: ~30 min code (4 call-site migrations + signature change + back-compat fall-through) + ~30 min tests = **~1 hour**.

### Fix #9 — `LazyConfigFactory` Universal Re-Instantiation Protocol [v4.4 NEW — for §2.8 root cause]

**Goal**: Replace the current asymmetric handling (`functools.partial` for inline `_target_:` `_factory` fields vs `_ImportFactory` for `_import_:`-tagged ones) with a UNIVERSAL `LazyConfigFactory` mechanism that produces fresh sub-trees on every `factory()` call. Pairs with §2.7's identification that the "real" sharing happens at `flow_configs[*].initial_inferencer` / `followup_inferencer` and §2.8's mechanical pinpoint of `_filter_attrs_keys:711`.

**Files**:
1. NEW: `/Users/tchen7/MyProjects/CoreProjects/RichPythonUtils/src/rich_python_utils/config_utils/_lazy_config_factory.py` — the new class.
2. MODIFY: `/Users/tchen7/MyProjects/CoreProjects/RichPythonUtils/src/rich_python_utils/config_utils/_instantiate.py` — walker change at lines 944-972 + `_apply_lazy_factory` rename + `_ImportFactory` deprecation alias.
3. MODIFY: `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/breakdown_then_aggregate_inferencer.py` — BTA `isinstance` recognition update at lines 1503-1511.

#### §F9.1 — `LazyConfigFactory` class (new file)

```python
"""LazyConfigFactory — re-instantiate from stored config on each call.

The canonical mechanism for `*_factory`-suffix attrs fields. Each call
deep-copies the captured raw config dict, re-applies parent injectables,
and re-runs `instantiate()`, producing a completely fresh sub-tree with
NO shared inner instances across calls.

Subsumes the older `_ImportFactory` (which is now a deprecated alias —
see Fix #10).

Recognition contract: orchestrators detect factory-shaped callables via
`isinstance(x, (functools.partial, LazyConfigFactory))`. Both types
satisfy the "fully bound, call with no args" semantic.
"""
from __future__ import annotations

import copy
import logging
import warnings
from typing import Any, Dict, Optional

_logger = logging.getLogger(__name__)


class LazyConfigFactory:
    """Lazy factory that re-instantiates from stored YAML/dict config on each call.

    Attributes:
        template_extra_feed: Required public attribute (default empty dict)
            for `_for_each_child_inferencer` duck-typing. When the caller
            populates this dict before invoking the factory, the resulting
            instance receives the values via `instance.template_extra_feed.update(...)`.
            This preserves the existing `_ImportFactory` integration with
            template-state propagation.

    Note:
        ``__call__`` accepts NO positional args and NO arbitrary kwargs —
        the contract is "factory is fully bound; call with empty parens".
        This eliminates the silent-kwarg-drop hazard that the old
        ``_ImportFactory.__call__(**_kwargs)`` signature had (which would
        silently absorb e.g. ``sub_query=`` / ``index=`` from BTA's
        bare-callable branch — a real bug for heterogeneous workers if
        the recognition isinstance check were ever bypassed).
    """

    __slots__ = ("_config_dict", "_injectables", "template_extra_feed")

    def __init__(
        self,
        *,
        config_dict: Dict[str, Any],
        injectables: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(config_dict, dict):
            raise TypeError(
                f"LazyConfigFactory.config_dict must be a dict, got {type(config_dict).__name__}"
            )
        self._config_dict = config_dict
        self._injectables = injectables or {}
        # MUST be a public attribute, MUST default to empty dict — required
        # for `_for_each_child_inferencer` duck-typing in
        # TemplatedInferencerBase._propagate_to_children.
        self.template_extra_feed: dict = {}

    def __call__(self) -> Any:
        """Instantiate a fresh sub-tree from the captured config.

        STRICT no-args signature: passing positional or keyword args raises
        TypeError. This is intentional — factories are "fully bound", and
        accepting kwargs would silently mask caller bugs (e.g., passing
        ``sub_query=`` to a factory whose target doesn't accept it).
        """
        # Local import to avoid cyclic dependency: LazyConfigFactory is used
        # by `instantiate()`, but `instantiate()` is also what we call here.
        from rich_python_utils.config_utils._instantiate import instantiate
        from omegaconf import OmegaConf

        config = copy.deepcopy(self._config_dict)
        # Re-apply parent injectables (e.g., `_template_manager`) — NOT
        # deep-copied because injectables are intended to be SHARED across
        # all instances (cf. TemplateManager cache hit-rate, logger identity).
        for k, v in self._injectables.items():
            injectable_key = f"_{k}"
            if injectable_key in config:
                config[injectable_key] = v
        instance = instantiate(OmegaConf.create(config))
        # Forward template_extra_feed staged by caller (preserves
        # _ImportFactory's existing duck-typing contract).
        if self.template_extra_feed and hasattr(instance, "template_extra_feed"):
            instance.template_extra_feed.update(self.template_extra_feed)
        return instance

    @property
    def target(self) -> str:
        """Return the `_target_` string from the captured config (introspection helper)."""
        return self._config_dict.get("_target_", "<unknown>")

    def __repr__(self) -> str:
        return f"LazyConfigFactory(target={self.target!r})"
```

#### §F9.2 — Walker change in `_filter_attrs_keys`

**Location**: `/Users/tchen7/MyProjects/CoreProjects/RichPythonUtils/src/rich_python_utils/config_utils/_instantiate.py:944-972`

**Diff** (the SINGLE structural change — make `_factory_configs.append(...)` UNCONDITIONAL for `_factory`-suffix fields):

```python
# Auto-partial / auto-factory for *_factory fields.
# v4.4 change: ALWAYS record _factory_configs entry so post-Hydra
# replacement with LazyConfigFactory always fires (not just for
# _import_-tagged blocks). This makes EVERY *_factory field produce
# fresh sub-trees per call.
for a in attr.fields(cls):
    if not a.name.endswith("_factory") or a.name not in node:
        continue
    val = node[a.name]
    if not isinstance(val, dict):
        continue
    if "_target_" in val:
        # Single factory: worker_factory: {_target_: ..., ...}
        # v4.4: ALWAYS capture raw + record (was only when _FACTORY_MARKER present)
        raw = copy.deepcopy(val)
        if _FACTORY_MARKER in val:
            del raw[_FACTORY_MARKER]
            del val[_FACTORY_MARKER]
        if _factory_configs is not None:
            _factory_configs.append((a.name, None, raw, _injectables or {}))
        val["_partial_"] = True
    else:
        # Dict of factories: worker_factory: {type1: {_target_: ...}, ...}
        for k, v in list(val.items()):
            if k.startswith("_") and k not in _DATA_KEYS:
                continue
            if isinstance(v, dict) and "_target_" in v:
                # v4.4: ALWAYS capture raw + record
                raw = copy.deepcopy(v)
                if _FACTORY_MARKER in v:
                    del raw[_FACTORY_MARKER]
                    del v[_FACTORY_MARKER]
                if _factory_configs is not None:
                    _factory_configs.append((a.name, k, raw, _injectables or {}))
                v["_partial_"] = True
```

#### §F9.3 — `_apply_import_factory` rename + LazyConfigFactory wrapping

**Location**: `_instantiate.py:225-234`

```python
# v4.4: rename to _apply_lazy_factory + use LazyConfigFactory class
def _apply_lazy_factory(obj, field_name, child_key, raw_config, injectables=None):
    """Replace the Hydra-created ``functools.partial`` with a ``LazyConfigFactory``.

    Called by ``instantiate()`` for every ``*_factory`` field whose raw
    config was captured by ``_filter_attrs_keys``. Replaces the eager-
    children partial with a config-deep-copying factory that produces
    fresh sub-trees per call.
    """
    from rich_python_utils.config_utils._lazy_config_factory import LazyConfigFactory
    container = getattr(obj, field_name, None)
    if container is None:
        return
    factory = LazyConfigFactory(config_dict=raw_config, injectables=injectables)
    if child_key is None:
        setattr(obj, field_name, factory)
    elif isinstance(container, dict):
        container[child_key] = factory
```

And update the call site at `_instantiate.py:220-222`:

```python
result = _hydra_instantiate(config, _convert_=_convert_, **kwargs)
for field_name, child_key, raw_config, injectables in factory_configs:
    _apply_lazy_factory(result, field_name, child_key, raw_config, injectables)
return result
```

#### §F9.4 — BTA recognition update

**Location**: `breakdown_then_aggregate_inferencer.py:1503-1511`

```python
# v4.4 BEFORE:
if isinstance(factory, functools.partial):
    worker = factory()
else:
    worker = factory(sub_query=query_str, index=i)

# v4.4 AFTER:
from rich_python_utils.config_utils._lazy_config_factory import LazyConfigFactory
# (import at top of module, not inline)

if isinstance(factory, (functools.partial, LazyConfigFactory)):
    # Fully-bound factory — call with no args.
    worker = factory()
else:
    # Bare callable — needs runtime context.
    worker = factory(sub_query=query_str, index=i)
```

Same pattern at the homogeneous branch (lines 1508-1511) and at any other orchestrator that uses `*_factory` fields. Single audit grep `rg "isinstance.*functools\.partial"` enumerates the migration sites.

#### §F9.5 — Acceptance criteria for Fix #9

| # | Criterion | Test |
|---|---|---|
| F9-1 | After Fix #9, two consecutive `worker_factory()` calls produce DISTINCT outer instances AND DISTINCT nested `flow_configs[i]['initial_inferencer']` instances | Unit: `test_lazy_config_factory_produces_distinct_nested_instances` |
| F9-2 | `factory()` deep-copies `config_dict` per call (mutating the result's nested dict does not affect the next call's config) | Unit: `test_lazy_config_factory_isolates_config_mutation` |
| F9-3 | Injectables (`_template_manager` etc.) are NOT deep-copied; the same parent reference is reused per call | Unit: `test_lazy_config_factory_shares_injectables` |
| F9-4 | `factory()` raises `TypeError` if called with positional args OR kwargs (strict no-args contract) | Unit: `test_lazy_config_factory_rejects_args_and_kwargs` |
| F9-5 | `template_extra_feed` populated on the factory BEFORE call propagates to the resulting instance via `update()` | Unit: `test_lazy_config_factory_propagates_template_extra_feed` |
| F9-6 | After Fix #9, BTA `worker_0 = self.worker_factory()` and `worker_1 = self.worker_factory()` produce DISTINCT MFDual instances with DISTINCT flow_configs sub-trees | Unit: `test_bta_workers_have_distinct_flow_configs_after_lazy_factory` |
| F9-7 | Cross-worker symlinks (Anomaly 6) DISAPPEAR — no `worker_N/round_NN/<phase>` symlinks resolve into a sibling worker's tree | Integration: SOP plan run + assert no cross-worker symlinks under any `worker_*/children/round_*` |
| F9-8 | `LazyConfigFactory.__repr__` returns `f"LazyConfigFactory(target='...')"` with the captured `_target_` (debuggability) | Unit: `test_lazy_config_factory_repr` |
| F9-9 | `inspect.signature(LazyConfigFactory.__call__)` shows `(self)` — strict no-args; documented as the contract | Unit: `test_lazy_config_factory_strict_signature` |
| F9-10 | YAML configs that previously used `_import_:` directives still produce LazyConfigFactory instances (back-compat — see Fix #10) | Unit: `test_import_directive_still_yields_factory` |

#### §F9.6 — Effort breakdown

| Item | Estimate |
|---|---|
| Create `LazyConfigFactory` class + tests F9-1..F9-5, F9-8, F9-9 | 1.5 hours |
| Walker change in `_filter_attrs_keys` + `_apply_lazy_factory` rename | 30 min |
| BTA `isinstance` migration (homogeneous + heterogeneous branches + import) | 20 min |
| Audit grep for other `*_factory` consumers (ConversationalDual, etc.); migrate same way | 30 min |
| Tests F9-6, F9-7 (BTA + integration) | 45 min |
| Code review pass on edge cases (interpolation resolution, OmegaConf round-trip, slots=False compatibility) | 30 min |
| **Total Fix #9** | **~4 hours** |

---

### Fix #10 — `_ImportFactory` Deprecated Alias for Back-Compat [v4.4 NEW — for Fix #9 migration safety]

**Goal**: Make Fix #9 a zero-breaking-change migration by keeping `_ImportFactory` as a deprecated subclass of `LazyConfigFactory`. All existing imports (`from rich_python_utils.config_utils._instantiate import _ImportFactory` — though no external consumers exist per audit) keep working with a deprecation warning on instantiation.

**Audit (verified 2026-05-10 19:13)**: `rg "_ImportFactory|ImportFactory"` across `/Users/tchen7/MyProjects/CoreProjects/AgentFoundation/src/` and `/Users/tchen7/MyProjects/rovoteam/PythonUtils/src/` returns:
- `_instantiate.py` (8 references — all internal: class def + repr + 6 docstring/comment mentions)
- `breakdown_then_aggregate_inferencer.py:785` (ONE reference — in a docstring comment, not an import)

**No external Python imports of `_ImportFactory` exist anywhere in the repos**. Subsumption is safe; the alias is for forward-defense (in case external consumers exist outside the searched paths or in user scripts).

**File**: `/Users/tchen7/MyProjects/CoreProjects/RichPythonUtils/src/rich_python_utils/config_utils/_instantiate.py` — replace the existing `class _ImportFactory:` block (lines 38-65) with:

```python
# v4.4: _ImportFactory is now a deprecated alias of LazyConfigFactory.
# Kept for backward compatibility with any external consumers (none found
# in the codebase audit, but defensive). Will be removed in a future
# release once all callers migrate to the new name.
from rich_python_utils.config_utils._lazy_config_factory import LazyConfigFactory


class _ImportFactory(LazyConfigFactory):
    """Deprecated alias for LazyConfigFactory.

    Use ``LazyConfigFactory`` directly. The old name is kept for one
    release cycle to avoid breaking external imports.
    """

    def __init__(self, config_dict: dict, injectables: dict | None = None) -> None:
        warnings.warn(
            "_ImportFactory is deprecated; use LazyConfigFactory instead. "
            "The class will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Note: positional `config_dict` for back-compat; new code MUST use
        # `LazyConfigFactory(config_dict=..., injectables=...)` keyword form.
        super().__init__(config_dict=config_dict, injectables=injectables)
```

#### §F10.1 — Acceptance criteria for Fix #10

| # | Criterion | Test |
|---|---|---|
| F10-1 | `from rich_python_utils.config_utils._instantiate import _ImportFactory` still works (no ImportError) | Unit: `test_import_factory_alias_importable` |
| F10-2 | `_ImportFactory(config_dict, injectables)` (positional first arg, back-compat) succeeds and returns a `LazyConfigFactory`-instance | Unit: `test_import_factory_alias_constructs` |
| F10-3 | Constructing `_ImportFactory(...)` emits a `DeprecationWarning` with stacklevel=2 (so the warning points to the CALLER, not the alias body) | Unit: `test_import_factory_emits_deprecation_warning` |
| F10-4 | `isinstance(_ImportFactory(...), LazyConfigFactory)` returns True (subclass relationship preserved for orchestrator recognition) | Unit: `test_import_factory_is_lazyconfigfactory_subclass` |
| F10-5 | The `__call__` behavior is identical between `_ImportFactory(...)` and `LazyConfigFactory(...)` (same fresh-instance semantics) | Unit: `test_import_factory_call_behaves_like_lazyconfigfactory` |

#### §F10.2 — Effort breakdown

| Item | Estimate |
|---|---|
| Replace `_ImportFactory` class body with deprecated subclass | 10 min |
| Tests F10-1..F10-5 | 20 min |
| Update `_instantiate.py` docstrings/comments referencing `_ImportFactory` to point to `LazyConfigFactory` instead | 10 min |
| **Total Fix #10** | **~40 min** |

---

### Fix #4 — Adopt user's proposed `worker_N/flow_*` nesting

**File**: `multi_flow_inferencer.py:559-586` (revert/replace Fix #6 from prior plan)

**Change**: Per-flow layout becomes:
```
worker_0/                       ← BTA's standard worker container (NOT renamed)
  ├── _workflow/                ← LWI orchestration state (was flow_0_workflow at parent level)
  ├── flow_0_initial/           ← actually moved one level deeper
  ├── flow_0_round01/
  └── flow_0_round02/
```

OR (simpler):
```
worker_0/                       ← contains EVERYTHING for flow 0
  ├── checkpoints/              ← LWI state goes here directly
  ├── final_result.json
  └── children/
      ├── flow_0_initial/
      ├── flow_0_round01/
      └── flow_0_round02/
```

**This is more invasive** — touches BTA's worker dispatch logic. Effort: ~1.5 hours code + ~1 hour tests.

**Risk**: Existing tests/checkpoints may reference the flat sibling layout; would need to also update LWI per-step naming.

---

## §4 Architectural Refactor — Layered `switch_role()` API across `InferencerBase` + `TemplatedInferencerBase` [v4.1]

**Status**: REVISED in v4.1 (supersedes v4.0's single-method design). Formalizes the recurring "rebind a leaf inferencer to a new orchestration role" pattern into a **layered, MRO-routed** contract. Fix #6 is restructured to USE this API rather than to extend MFDual's helper.

**Why layered**: The codebase deliberately splits responsibilities across `InferencerBase` (workspace, deliverable semantics, lifecycle) and `TemplatedInferencerBase` (template rendering state). The comment block at `inferencer_base.py:184-198` documents this boundary explicitly. v4.0 violated it by referencing `template_key` from the base method (guarded by `hasattr`); v4.1 mirrors the established `_propagate_to_children()` pattern that already splits the same way:

```
InferencerBase._propagate_to_children()       → no-op stub
TemplatedInferencerBase._propagate_to_children() → propagates template_extra_feed + modes
```

We replicate that pattern for `switch_role()`.

**Why now**: Anomaly 5 exposed that the codebase has at least three implicit mutation sites for role transitions (workspace via `_reassign_role_workspace`, `template_key` per Fix #6, `output_is_deliverable` in `_select_reviewer_and_fixer` lines 577/580), each currently expressed as direct attribute writes. Adding a fourth (e.g., `template_extra_feed` reset) would require chasing N call sites. The layered `switch_role()` collapses these into one named, documented, testable contract per layer.

### §4.0 MRO confirmation (foundational invariant for the design)

Verified across the codebase:

| Class | MRO position | Has template attribs? |
|---|---|---|
| `InferencerBase(Debuggable, Resumable, ABC)` | base | ❌ NO |
| `TemplatedInferencerBase(InferencerBase)` | mixin layer | ✅ YES (declares them) |
| `ApiInferencerBase(TemplatedInferencerBase)` | leaf-base | ✅ YES (inherited) |
| `RemoteInferencerBase(TemplatedInferencerBase)` | leaf-base | ✅ YES (inherited) |
| `StreamingInferencerBase(TemplatedInferencerBase)` (via Remote) | leaf-base | ✅ YES (inherited) |
| `TerminalInferencerBase(TemplatedInferencerBase)` | leaf-base | ✅ YES (inherited) |
| `ClaudeApiInferencer(ApiInferencerBase)` | leaf | ✅ YES |
| `OpenaiApiInferencer(ApiInferencerBase)` | leaf | ✅ YES |
| `BedrockInferencer(RemoteInferencerBase)` | leaf | ✅ YES |
| `HttpRequestInferencer(RemoteInferencerBase)` | leaf | ✅ YES |
| `ToolAsInferencer(StreamingInferencerBase)` | leaf | ✅ YES |
| `BreakdownThenAggregateInferencer(InferencerBase, WorkGraph)` | orchestrator | ❌ NO |
| `LinearWorkflowInferencer(InferencerBase, Workflow)` | orchestrator | ❌ NO |
| `MultiFlowDualInferencer(DualInferencer)` → ultimately `InferencerBase` | orchestrator | ❌ NO |

**Critical invariant**: every leaf MFDual mutates as winner/loser is a `TemplatedInferencerBase` descendant (it has to be — it renders LLM prompts via `template_manager`). So the layered design's MRO routing always reaches the templated layer for those calls. Confirmed by inspection of all CLI/API/streaming leaves used in flow_configs.

### §4.1 Phase 1a — Add `InferencerBase.switch_role()` (base layer ONLY)

**File**: `src/agent_foundation/common/inferencers/inferencer_base.py`

**Location**: Immediately after the `_workspace` property setter (~line 240) — co-located with the other workspace-mutation contract methods. Class-level constant `_ROLE_RELEVANT_ATTRS` placed near `_DERIVED_FROM_WORKSPACE` and `_workspace_propagation_skip` for discoverability.

**Scope**: Only attributes ACTUALLY DECLARED on `InferencerBase`:
- `_workspace` (private property at line 227)
- `output_is_deliverable` (line 122)
- `is_deliverable_boundary` (line 182)
- Lifecycle hook: `reset_session()` (defined on `Resumable` mixin, present on every InferencerBase descendant)
- Audit-trail accumulator: `_role_history`

```python
# Class-level enumeration of attributes a role transition may affect AT
# THIS LAYER. Subclasses MUST extend this tuple (cumulatively, via
# `BaseClass._ROLE_RELEVANT_ATTRS + (...)`) so that documentation +
# linting + introspection see the full set at any layer in the MRO.
# This is the ONE place to maintain when adding a new role-relevant
# field. Single source of truth per layer.
_ROLE_RELEVANT_ATTRS: tuple = (
    "output_is_deliverable",
    "is_deliverable_boundary",
)

# Per-instance audit trail of role transitions. Lazy-initialised in
# switch_role() to avoid attrs storage requirements (slots=False on
# subclasses already; this matches Resumable's lazy state pattern).
# Each entry: {"to_role": str, "at": float, "changes": dict[str, Any]}
# _role_history: list[dict] (created on first switch_role call)

def switch_role(
    self,
    new_role: str,
    *,
    workspace: Optional["InferencerWorkspace"] = None,
    output_is_deliverable: Optional[bool] = None,
    is_deliverable_boundary: Optional[bool] = None,
    reset_session: bool = True,
) -> None:
    """Transition this inferencer to a new orchestration role — base layer.

    The canonical, documented mechanism for orchestrators to rebind a leaf
    inferencer's role-relevant attributes when reusing the instance across
    roles (e.g., MFDual reusing the winning flow inferencer as fixer, or
    DualInferencer swapping reviewer/fixer mid-run).

    LAYERING NOTE: This base implementation handles ONLY attributes
    declared on ``InferencerBase``. Template-related attributes
    (``template_key``, ``template_root_space``, ``template_extra_feed``,
    ``template_variables``, ``template_version``, ``modes``) are handled
    by ``TemplatedInferencerBase.switch_role()`` which overrides this
    method, applies its own attribs, then calls ``super().switch_role()``
    to inherit base behavior. Caller code is unchanged — Python's MRO
    routes ``inferencer.switch_role(template_key=..., workspace=...)`` to
    the deepest applicable layer transparently. If an orchestrator passes
    ``template_key=`` to an inferencer that does NOT inherit from
    ``TemplatedInferencerBase``, Python raises ``TypeError: switch_role()
    got an unexpected keyword argument 'template_key'`` — a LOUD failure
    consistent with the codebase's "no silent failure" philosophy.

    Each kwarg follows **explicit-or-preserve** semantics:
      - If provided (not None): apply via the appropriate write path.
      - If None (the default): preserve the existing attribute value.

    Callers can be SURGICAL — pass only what changes for the role;
    everything else stays intact. Mirrors Python's ``dict.update``-style
    conventions.

    The MUTATION ORDER is fixed and documented:
      1. ``workspace`` — assigned via the property setter so the existing
         workspace cascade fires (cache invalidation + child propagation
         + ``_configure_for_workspace`` + logger redirection). MUST happen
         FIRST so subsequent assignments operate against the new workspace.
      2. ``output_is_deliverable`` — affects ``_post_finalize_deliverable_and_manifest``.
      3. ``is_deliverable_boundary`` — affects deliverable surfacing.
      4. (template attribs handled by TemplatedInferencerBase override
         BEFORE this base method is reached, since the override wraps
         ``super().switch_role()`` — see TemplatedInferencerBase below.)
      5. ``reset_session`` — if True (default) AND the inferencer exposes
         ``reset_session()``, invoke it AFTER all attribute writes so the
         leaf starts the new role with a clean session.

    NOTE: No ``hasattr`` defensive guards here — every attribute named in
    this method's signature is GUARANTEED present on every descendant of
    ``InferencerBase`` (declared at base level). The absence of guards is
    a feature: AttributeError on a "missing" attribute would indicate a
    real architectural breakage, not an expected variant case.

    Args:
        new_role: Identifier of the role being assumed (e.g.,
            ``"fixer_inferencer"``, ``"review_inferencer"``,
            ``"aggregator_inferencer"``). Used in the audit log and
            (optionally) by subclass overrides for role-conditional logic.
        workspace: New ``InferencerWorkspace`` to bind, or None to preserve.
        output_is_deliverable: Override for the deliverable flag, or None
            to preserve.
        is_deliverable_boundary: Override for the boundary flag, or None
            to preserve.
        reset_session: If True (default), call ``reset_session()`` on this
            inferencer after attribute application so the leaf does not
            carry session state across roles. Set False for orchestrators
            that want to retain session continuity (rare).

    Audit trail: Records the transition in ``self._role_history`` with
    timestamp + the dict of explicitly-changed attributes. Use this for
    post-mortem debugging — the absence of an expected entry is a strong
    signal that an orchestrator is bypassing the canonical pathway.
    Templated-layer overrides MUST forward their own changes to this
    audit by passing them through ``super().switch_role()``-side hooks
    (see TemplatedInferencerBase impl below for the merge protocol).

    Parallel-safety: ``switch_role()`` is NOT safe to call concurrently
    with an in-flight ``ainfer()`` on the same inferencer. Orchestrators
    are responsible for serialization (MFDual's ``_step_propose_impl``
    already serializes propose-then-dispatch; DualInferencer's consensus
    loop is single-threaded per inferencer). Future hardening: optional
    ``assert not self._currently_inferring`` guard.
    """
    import time

    # ── 1. Workspace (FIRST — triggers cascade) ──
    if workspace is not None:
        self._workspace = workspace  # property setter handles cascade

    # ── 2-3. Base-level role-relevant attribute application ──
    explicit_changes: dict = {}
    pending = {
        "output_is_deliverable": output_is_deliverable,
        "is_deliverable_boundary": is_deliverable_boundary,
    }
    for attr_name, value in pending.items():
        if value is None:
            continue  # implicit-preserve
        # NO hasattr guard — every descendant has these attribs (base-declared).
        setattr(self, attr_name, value)
        explicit_changes[attr_name] = value

    # ── 5. Optional session reset ──
    # `reset_session` is on Resumable (mixed into InferencerBase); always present.
    if reset_session:
        self.reset_session()

    # ── Audit trail (MERGE-FRIENDLY for templated-layer overrides) ──
    history = getattr(self, "_role_history", None)
    if history is None:
        history = []
        # Avoid attrs validation fights — direct __dict__ assignment.
        object.__setattr__(self, "_role_history", history)
    base_changes = {
        **({"workspace": workspace} if workspace is not None else {}),
        **explicit_changes,
        "reset_session": reset_session,
    }
    # If a templated-layer override stashed its own changes via the
    # _pending_role_changes attribute (set by the override BEFORE calling
    # super().switch_role()), merge them into THIS audit entry so the
    # caller sees ONE entry per logical transition, not two.
    pending_template_changes = getattr(self, "_pending_role_changes", None)
    if pending_template_changes:
        base_changes.update(pending_template_changes)
        # Clear the buffer so it doesn't leak into the next call.
        object.__setattr__(self, "_pending_role_changes", None)
    history.append({
        "to_role": new_role,
        "at": time.time(),
        "changes": base_changes,
    })

    import logging as _logging
    _logging.getLogger(__name__).info(
        "InferencerBase[%s] switch_role(new_role=%s) applied changes: %s",
        type(self).__name__,
        new_role,
        sorted(history[-1]["changes"].keys()),
    )
```

### §4.2 Phase 1b — Add `TemplatedInferencerBase.switch_role()` (template layer)

**File**: `src/agent_foundation/common/inferencers/templated_inferencer_base.py`

**Location**: After `_propagate_to_children()` (around line 365) — co-located with the other template-state cascade methods, mirroring the spatial pattern of the analogous `_propagate_to_children` split.

**Scope**: Only attributes DECLARED on `TemplatedInferencerBase` (lines 91-114):
- `template_key`, `template_root_space`, `template_extra_feed`
- `template_variables`, `template_version`, `modes`

```python
# Cumulative extension of base's _ROLE_RELEVANT_ATTRS — adds template
# attributes. Used by introspection/documentation/lint tooling to enumerate
# the FULL role-relevant attrib set at any layer of the MRO.
_ROLE_RELEVANT_ATTRS: tuple = InferencerBase._ROLE_RELEVANT_ATTRS + (
    "template_key",
    "template_root_space",
    "template_extra_feed",
    "template_variables",
    "template_version",
    "modes",
)

def switch_role(
    self,
    new_role: str,
    *,
    template_key: Optional[str] = None,
    template_root_space: Optional[str] = None,
    template_extra_feed: Optional[dict] = None,
    template_variables: Optional[dict] = None,
    template_version: Optional[str] = None,
    modes: Optional[dict] = None,
    **base_kwargs,
) -> None:
    """Transition this templated inferencer to a new orchestration role.

    Extends ``InferencerBase.switch_role()`` with template-state attribs.
    All template-related kwargs follow the same explicit-or-preserve
    semantics as the base method.

    APPLICATION PROTOCOL:
      1. Template attribs are applied FIRST (in this override) so the
         workspace cascade triggered by base's ``super().switch_role()``
         operates against the new template config.
         RATIONALE: ``_workspace`` setter calls ``_configure_for_workspace``
         + ``_propagate_workspace_to_children``; while neither directly
         consults ``template_key`` today, applying templates BEFORE
         workspace cascade keeps the order intuitive and future-proofs
         against any future template-aware workspace setup.
      2. Pending template changes are stashed on ``_pending_role_changes``
         so base's audit-trail logic can MERGE them into a single audit
         entry rather than emitting two (one per layer).
      3. ``super().switch_role(**base_kwargs)`` runs base-layer logic
         (workspace assignment, base-level attribs, reset_session, audit).
         The merge protocol means the audit reflects the FULL transition
         atomically.

    NOTE: NO ``hasattr`` guards on template attribs — every descendant of
    ``TemplatedInferencerBase`` has them (declared at this layer).
    AttributeError here would indicate real breakage. Loud failure mode is
    the codebase convention (cf. Phase 5 leaf rendering, default.jinja2
    strict resolution, etc.).

    NOTE: Mistyped kwargs (e.g., ``template_KKey=...``) raise
    ``TypeError: switch_role() got an unexpected keyword argument`` —
    Python's standard kwargs-validation behavior. This is STRICTLY
    BETTER than v4.0's hasattr-based silent-skip: typos are caught at
    call time, not 5 hours later in production after templates render
    wrong.

    Args:
        new_role: Identifier of the role being assumed.
        template_key: New template variant key (e.g., ``"review"``,
            ``"followup"``), or None to preserve.
        template_root_space: New template namespace (e.g., ``"plan"``),
            or None to preserve.
        template_extra_feed: Replacement dict for runtime feed overrides,
            or None to preserve. Pass ``{}`` to explicitly clear.
        template_variables: Replacement dict for variable specs (variant
            selectors), or None to preserve.
        template_version: Override for variant fallback version, or None
            to preserve.
        modes: Replacement dict for mode flags (e.g., ``deep_mode``,
            ``elegant_mode``), or None to preserve.
        **base_kwargs: Forwarded to ``InferencerBase.switch_role()``
            (``workspace``, ``output_is_deliverable``,
            ``is_deliverable_boundary``, ``reset_session``).
    """
    # ── 1. Apply template-layer attribs FIRST ──
    template_changes: dict = {}
    pending = {
        "template_root_space": template_root_space,
        "template_key": template_key,
        "template_extra_feed": template_extra_feed,
        "template_variables": template_variables,
        "template_version": template_version,
        "modes": modes,
    }
    for attr_name, value in pending.items():
        if value is None:
            continue  # implicit-preserve
        # NO hasattr guard — every TemplatedInferencerBase descendant has these.
        setattr(self, attr_name, value)
        template_changes[attr_name] = value

    # ── 2. Stash pending changes for base's audit-trail merge ──
    # Use object.__setattr__ to sidestep attrs validation; matches the
    # _role_history initialization pattern.
    if template_changes:
        object.__setattr__(self, "_pending_role_changes", template_changes)

    # ── 3. Forward to base ──
    super().switch_role(new_role, **base_kwargs)
```

### §4.3 Phase 2 — Migrate existing call sites in the SAME commit

The whole point of v4.1 is to land **base + template methods + all current callers** atomically. No transitional state where the helper exists but call sites still hand-mutate.

#### Migration #1 — `MultiFlowDualInferencer._reassign_role_workspace()`

**File**: `src/agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/multi_flow_dual_inferencer.py` (~line 389-418).

**Replacement**: see Fix #6 above — the helper becomes a thin wrapper that prepares (workspace, template_key, template_root_space) via `_resolve_role_template()` and forwards to `switch_role()`. Identity-guard policy stays MFDual-local; mutation mechanics live on the base.

#### Migration #2 — `MultiFlowDualInferencer._select_reviewer_and_fixer()` direct mutations

**File**: same file, lines 577 + 580.

```python
# BEFORE (v3.0):
self.fixer_inferencer = chosen
self.fixer_inferencer.output_is_deliverable = True
# ... or ...
self.fixer_inferencer = winner
self.fixer_inferencer.output_is_deliverable = True
```

The `output_is_deliverable = True` write is a role-relevant mutation (only the fixer's output is the deliverable in MFDual). After v4.0 lands, the canonical path is:

```python
# AFTER (v4.0): _select_reviewer_and_fixer assigns the SLOT only;
# `_step_propose_impl` then drives the canonical role transition,
# which carries `output_is_deliverable=True` through switch_role().
self.fixer_inferencer = chosen   # slot binding (orchestrator bookkeeping)
# (no direct attribute mutation here — _reassign_role_workspace
#  in _step_propose_impl forwards output_is_deliverable=True via
#  switch_role() once the slot is settled)
```

Then in `_step_propose_impl`'s reassignment block:

```python
# AFTER (v4.0):
self._reassign_role_workspace(
    self.review_inferencer, "review_inferencer",
    output_is_deliverable=False,        # reviewers never deliver
)
self._reassign_role_workspace(
    self.fixer_inferencer, "fixer_inferencer",
    output_is_deliverable=True,         # fixer carries the deliverable
)
```

**Note**: `_reassign_role_workspace()` v4.0 signature accepts the kwarg and forwards it to `switch_role()`. The session-reset loop that follows the reassignment block (lines 664-669) becomes redundant — `switch_role(reset_session=True)` already handles it — and is removed in the same commit.

#### Migration #3 — Optional symmetric uptake on `DualInferencer`

`DualInferencer` does not currently mutate `template_key` (its slots are YAML-configured), but if a future code path adds runtime swap of reviewer/fixer there, it will use the same `switch_role()` API for free. No action required NOW; documented as a forward-compatibility win.

#### Migration #4 — Aggregator workspace reassignment (Fix #1 of this plan)

Fix #1's aggregator workspace rebind (`MultiFlowInferencer._build_aggregator_input` / `_run_aggregator`) ALSO calls `switch_role(new_role="aggregator_inferencer", workspace=parent.child("aggregator"))`. No template change needed there (aggregator's `STRUCTURED_AGGREGATION_DEFAULTS` is constructed once and stays valid), but routing through the canonical API gets the audit trail "for free" and standardizes the call shape with reviewer/fixer.

### §4.4 Subclass extensibility (forward-compat) — applies at BOTH layers

The layered design composes naturally for any further subclass that needs to add role-relevant attribs. The pattern is uniform — extend `_ROLE_RELEVANT_ATTRS` cumulatively from the parent layer's tuple, declare typed kwargs in the override, apply them, then call `super().switch_role(...)`:

```python
# Example: a hypothetical DualInferencer-level role-relevant attrib
class DualInferencer(InferencerBase):
    _ROLE_RELEVANT_ATTRS = InferencerBase._ROLE_RELEVANT_ATTRS + ()
    # No DualInferencer-specific role-relevant attribs today; placeholder.

# Example: orchestrator-level subclass that wants role-conditional setup
# (e.g., clear flow_results when role transitions to fixer). Note that
# orchestrators inherit from InferencerBase directly (NOT
# TemplatedInferencerBase), so their override calls
# `InferencerBase.switch_role()` via super() — the templated layer is NOT
# in their MRO.
class MultiFlowDualInferencer(DualInferencer):
    _ROLE_RELEVANT_ATTRS = DualInferencer._ROLE_RELEVANT_ATTRS + ()

    def switch_role(self, new_role, *, clear_flow_results: bool = False, **kwargs):
        super().switch_role(new_role, **kwargs)
        # Subclass-specific post-step (example — not implemented today):
        # if clear_flow_results and hasattr(self, "_flow_results"):
        #     self._flow_results.clear()
```

The `_pending_role_changes` audit-merge protocol established at the templated layer also serves any FUTURE intermediate layer that wants to contribute to the audit entry without emitting its own — just stash a dict and let base merge it.

### §4.5 Acceptance criteria for §4 (revised for v4.1 layered design)

| # | Criterion | Test |
|---|---|---|
| §4-1 | `InferencerBase.switch_role(workspace=ws)` triggers the same cascade as direct `_workspace = ws` (cache invalidation + child propagation + logger redirect) | Unit: `test_switch_role_workspace_cascades_via_setter` |
| §4-2 | `TemplatedInferencerBase.switch_role(template_key=None, template_root_space=None)` preserves existing values | Unit: `test_switch_role_implicit_preserve_templates` |
| §4-3 | `TemplatedInferencerBase.switch_role(template_extra_feed={})` clears the dict (explicit empty != None) | Unit: `test_switch_role_explicit_clear_extra_feed` |
| §4-4 | **(v4.1)** `inferencer.switch_role(template_key="x")` on a NON-templated `InferencerBase` raises `TypeError: unexpected keyword argument 'template_key'` (loud failure, not silent skip) | Unit: `test_switch_role_typeerror_on_template_kwarg_to_base_only` |
| §4-5 | `_role_history` accumulates ONE entry per logical call even when both layers contribute changes (no double-entry per layered call) | Unit: `test_switch_role_audit_trail_single_entry_per_call` |
| §4-6 | `switch_role(reset_session=False)` does NOT call `reset_session()` even when present | Unit: `test_switch_role_reset_session_opt_out` |
| §4-7 | Templated override's `super().switch_role(...)` inherits the base audit + merges its own template changes into the SAME entry via `_pending_role_changes` | Unit: `test_switch_role_layered_audit_merge` |
| §4-8 | After v4.1 migration, NO occurrence of direct `inferencer.template_key = ...` outside `TemplatedInferencerBase.switch_role()` (lint/grep check) | Repo grep: `rg "\\.template_key\\s*=\\s*"` returns only the `switch_role()` body in `templated_inferencer_base.py` |
| §4-9 | **(v4.1)** `TemplatedInferencerBase._ROLE_RELEVANT_ATTRS` includes ALL of `InferencerBase._ROLE_RELEVANT_ATTRS` + 6 template attribs (cumulative) | Unit: `test_role_relevant_attrs_cumulative` |
| §4-10 | **(v4.1)** Mistyped kwarg (e.g., `template_KKey="x"`) raises `TypeError` from Python's standard kwargs validation | Unit: `test_switch_role_typo_kwarg_raises_typeerror` |
| §4-11 | **(v4.1)** `_pending_role_changes` is cleared after base consumes it (no leakage into the next call) | Unit: `test_pending_role_changes_buffer_lifecycle` |
| §4-12 | **(v4.1)** Calling base `InferencerBase.switch_role()` directly (e.g., on a BTA worker that's not templated) works correctly with NO template kwargs accepted in its signature | Unit: `test_base_switch_role_signature_excludes_template_kwargs` |

### §4.6 Effort breakdown for §4 (revised for v4.1 layered design)

| Item | Estimate |
|---|---|
| `InferencerBase.switch_role()` + `_ROLE_RELEVANT_ATTRS` (base layer) | 30 min |
| `TemplatedInferencerBase.switch_role()` + cumulative `_ROLE_RELEVANT_ATTRS` (template layer) | 20 min |
| `_role_history` audit trail + `_pending_role_changes` merge protocol + logging | 25 min |
| Migration #1 — MFDual `_reassign_role_workspace()` thin-wrapper | 20 min |
| Migration #2 — MFDual `_select_reviewer_and_fixer()` mutation removal + `_step_propose_impl` kwarg threading | 20 min |
| Migration #4 — Fix #1 aggregator routed through `switch_role()` | 10 min |
| Unit tests (acceptance §4-1 .. §4-12 + Fix #6 rows 7-9) | 1.5 hours |
| Integration test (full SOP plan run, assert correct templates rendered) | 30 min |
| Repo-wide grep + cleanup (acceptance §4-8) | 15 min |
| **Total §4** | **~4 hours** (was ~3.5h in v4.0; +30 min for second layer + audit-merge protocol + extra tests) |

---

## §5 Implementation Order [v4.4 RESTRUCTURED]

v4.4 promotes Fix #9 to FIRST priority because it eliminates the ROOT CAUSE of Anomaly 6 (cross-worker instance sharing). Once landed, Fixes #7/#8 become "defensive observability" rather than "load-bearing for correctness".

1. **Fix #9** (`LazyConfigFactory` universal re-instantiation) — **4 hours** — 🔴 **LAND FIRST**. Eliminates the `_filter_attrs_keys:711` root cause; produces fresh sub-trees per `factory()` call. Anomaly 6's cross-worker symlinks STOP appearing after this lands.
2. **Fix #10** (`_ImportFactory` deprecated alias) — **40 min** — atomic with Fix #9; together they form a single PR.
3. **§4 (layered `switch_role()` API + Phase-2 migrations) — 4 hours** — FOUNDATIONAL atomic-mutation primitive. Even with Fix #9 eliminating sharing, the per-call atomicity of role transitions remains the right contract for any future code that mutates role-relevant attribs.
4. **Fix #5 Part A** (BTA worker diagnostic logging) — 15 min (expanded from 10 min — now logs `flow_configs[*].initial_inferencer` IDs too per v4.3 refinement) — observability; verifies Fix #9's effectiveness in production runs.
5. **Fix #7** (audit symlink hardening) — 1.25 hours — DEFENSIVE diagnostic layer. With Fix #9 in place, Fix #7's three diagnostic checks SHOULD never fire in healthy runs — but if a regression reintroduces sharing, Fix #7 catches it loudly.
6. **Fix #8** (snapshot-at-phase-time audit semantics) — 1 hour — DEFENSIVE timing-race protection. Independent of sharing; protects against any future code that mutates `_workspace` between phase-execution and audit.
7. **Fix #1** (aggregator marker / workspace rebind via `switch_role()`) — 30 min — lowest risk
8. **Fix #2** (no double final_deliverables) — 45 min — medium risk
9. **Fix #3** (eliminate _round01 phantom) — 1 hour — medium risk
10. **Fix #5 Part B v4.6 INTEGRATED** (recursive descendant-walk sharing detection — single boolean attrib, WARN-only, base-class helper, called from `_build_subgraph_spec()`) — **2.25 hours** (was 3h in v4.5; the −45 min savings come from dropping enum proliferation + diagnostic-file plumbing). Adds `InferencerBase._collect_all_descendant_inferencers()` base helper (mirrors `pre_retry`'s `_seen` cycle-safety pattern). Default `worker_isolation_check=True`. With Fix #9 in place this should produce ZERO warnings, but if a future code path adds a shared-instance regression at ANY depth, Part B logs ONE warning per collision with full context (BTA name, both worker indices, instance id, type, remediation hint). Future-extensible via OPTIONAL `worker_isolation_strict: bool` if WARN logs warrant escalation — strictly YAGNI today.
11. **Fix #4** (worker_N nesting) — 1.5 hours — highest risk
12. **Fix #6** is fully expressed by §4's Migration #1+#2 — no separate code step

**Total**: ~16.2 hours code + tests (was ~16.95h in v4.5; the −0.75h is the v4.6 simplification of Fix #5 Part B from enum-and-diagnostic-file design to single-boolean WARN-only design).

**Defense-in-depth ordering rationale (v4.4 layered correctness model)**:
- **Layer 0 (Fix #9 + Fix #10)**: ELIMINATES the bug class at the parser layer. Fresh sub-trees per `factory()` call mean cross-worker sharing is structurally impossible.
- **Layer 1 (§4 layered `switch_role()`)**: Makes per-call role-attribute mutation atomic + auditable. Necessary for future code that swaps roles within ONE worker (e.g., self-promotion).
- **Layer 2 (Fix #5 Part A)**: Logs instance IDs to verify Layer 0's effectiveness in production. Cheap observability.
- **Layer 3 (Fix #7)**: Detects ANY return of cross-worker leakage at the audit layer. Defensive lint for the audit substrate.
- **Layer 4 (Fix #8)**: Decouples audit snapshot from live state. Defensive against any future timing race.
- **Layer 5 (Fix #5 Part B)**: Raises at construction if a regression reintroduces sharing. Preventive belt-and-suspenders.

Each layer above Layer 0 is INSURANCE — they should never fire after Fix #9 lands, but if they do, they catch the regression before user-visible damage.

---

## §6 Acceptance Criteria

| # | Criterion | Test |
|---|---|---|
| 1 | Empty aggregator dir contains `.skipped` marker explaining winner-pick strategy | `test_aggregator_skipped_marker_when_winner_pick` |
| 2 | After winner self-promotes, MFDual deliverables show ONE level of `final_deliverables/`, not two | `test_no_double_final_deliverables_nesting` |
| 3 | After Fix #3: NO empty `_round01` placeholder exists; first followup step writes to `_round01` | `test_first_followup_step_uses_round01_dir` |
| 4 | After Fix #4: `flow_X_workflow/` no longer appears at MFDual level; LWI state lives at `worker_X/checkpoints/` | `test_no_flow_workflow_at_mfdual_level` |
| 5 | **(v4.x)** Fixer renders `followup.jinja2` (NOT `initial.jinja2`) after winner-as-fixer assignment | Integration: SOP plan run + grep for `<ReviewerFeedback>` block in fixer's rendered prompt |
| 6 | **(v4.x)** Reviewer renders `review.jinja2` (NOT `initial.jinja2`) after loser-as-reviewer assignment | Integration: SOP plan run + grep for verdict-evaluation framing |
| 7 | **(v4.x)** `inferencer._role_history` is non-empty after any role transition; first entry has correct `to_role` | Unit: `test_switch_role_audit_trail_recorded` |
| 8 | **(v4.x)** Repo grep `rg "\\.template_key\\s*=\\s*"` returns ONLY the `TemplatedInferencerBase.switch_role()` body — no other direct mutations exist | CI lint check |
| 9 | **(v4.1)** `MultiFlowDualInferencer.switch_role()` (orchestrator override) cleanly composes via `super()` → `InferencerBase.switch_role()` (NOT TemplatedInferencerBase, since orchestrators don't inherit from it) | Unit: `test_orchestrator_subclass_super_inherits_audit` |
| 10 | **(v4.1)** A leaf-layer call `inferencer.switch_role(template_key="x", workspace=ws, output_is_deliverable=True)` produces ONE audit entry with all three changes merged via `_pending_role_changes` | Unit: `test_layered_call_produces_single_merged_audit_entry` |
| 11 | **(v4.1)** Calling `switch_role(template_key="x")` on an instance whose MRO does NOT include `TemplatedInferencerBase` raises `TypeError: switch_role() got an unexpected keyword argument 'template_key'` | Unit: `test_switch_role_typeerror_on_template_kwarg_to_non_templated` |
| 12 | **(v4.2)** When two MFDual workers share a Python inferencer instance, EACH worker's `round_NN/<phase>.LEAKAGE.diagnostic.txt` is written with both the MFDual root path AND the foreign target path (Fix #7 Check 1) | Unit: `test_audit_emits_leakage_diagnostic_when_target_outside_mfdual_root` |
| 13 | **(v4.2)** When `review_inferencer is fixer_inferencer` within one MFDual, after both phases run audit, a `<phase>_vs_<other_phase>.ALIASING.diagnostic.txt` exists in `round_NN/` (Fix #7 Check 2) | Unit: `test_audit_emits_aliasing_diagnostic_when_two_phases_share_target` |
| 14 | **(v4.2)** When `_record_round_audit()` is called twice with same (round, phase) but different inferencer targets, `<phase>.OVERWRITE.diagnostic.txt` is written (Fix #7 Check 3) | Unit: `test_audit_emits_overwrite_diagnostic_on_same_phase_target_change` |
| 15 | **(v4.2)** Healthy run (no sharing, no aliasing) produces ZERO `*.diagnostic.txt` files — diagnostics are silent in the absence of pathology (no false positives) | Integration: SOP plan run + assert no `*.diagnostic.txt` exists under any `worker_*/round_*/` |
| 16 | **(v4.2)** Symlink itself is still created when diagnostics fire (back-compat: existing tooling that follows the link still works) | Unit: `test_audit_symlink_still_created_alongside_diagnostic` |
| 17 | **(v4.2)** When `workspace_root_at_phase=` is passed explicitly, audit symlink target uses the snapshot, NOT the live `inferencer._workspace.root` (Fix #8) | Unit: `test_audit_uses_explicit_workspace_snapshot_when_provided` |
| 18 | **(v4.2)** When `workspace_root_at_phase=None` is passed (back-compat), audit falls through to live read (Fix #8 fail-safe) | Unit: `test_audit_falls_back_to_live_read_when_no_snapshot` |
| 19 | **(v4.2)** Concurrent test: two workers share one instance, run review on worker_0, swap `_workspace` via worker_1's `switch_role()`, then audit on worker_0 — symlink target equals snapshot path (worker_0's review_inferencer/), NOT the moved-to path (Fix #8 main scenario) | Unit (asyncio): `test_audit_snapshot_immune_to_concurrent_workspace_swap` |
| 20 | **(v4.2)** Diagnostic files are written even if `os.symlink` raises (e.g., on Windows) — pointer-fallback ALSO gets the diagnostic | Unit: `test_diagnostic_written_even_when_symlink_raises` |
| 21 | **(v4.4)** After Fix #9, `id(worker_0.flow_configs[0]['initial_inferencer']) != id(worker_1.flow_configs[0]['initial_inferencer'])` — flow-pool inferencers are no longer shared across BTA workers | Unit: `test_bta_workers_have_distinct_flow_pool_instances_after_lazy_factory` |
| 22 | **(v4.4)** After Fix #9, the integration test from row 15 produces ZERO cross-worker symlinks AND ZERO `*.diagnostic.txt` files — Layer 0 has eliminated the root cause | Integration: same as row 15, asserts both conditions |
| 23 | **(v4.4)** `LazyConfigFactory` is the ONLY type produced for `*_factory`-suffix attrs fields (no vanilla `functools.partial` for inline `_target_` `_factory` fields) | Unit: `test_walker_produces_lazy_config_factory_for_all_factory_fields` |
| 24 | **(v4.4)** `_ImportFactory` import + construction continues to work with `DeprecationWarning` — zero breaking change for any external consumers | Unit: `test_import_factory_alias_back_compat` |
| 25 | **(v4.4)** BTA's `isinstance(x, (functools.partial, LazyConfigFactory))` accepts BOTH legacy and new factory types — heterogeneous mix in `worker_factory: dict` works | Unit: `test_bta_recognizes_both_factory_types_in_dict` |
| 26 | **(v4.4)** OmegaConf interpolations (`${_params.default_inferencer}`) are RE-RESOLVED on each `LazyConfigFactory()` call (so override changes propagate to subsequent instances) | Unit: `test_lazy_config_factory_re_resolves_interpolations_per_call` |
| 27 | **(v4.4)** `LazyConfigFactory` correctly handles nested `*_factory` fields (a factory whose body itself contains `*_factory` sub-fields produces fresh sub-tree-with-fresh-sub-factories per call) | Unit: `test_lazy_config_factory_recursive_factories` |
| 28 | **(v4.6)** `_collect_all_descendant_inferencers()` on a leaf yields exactly `[self]`; on an MFDual yields all flow_configs entries + aggregator deduped by id | Unit: `test_collect_descendants_yields_correct_set` |
| 29 | **(v4.6)** Cycle in descendant graph (A.children=[B], B.children=[A]) terminates via `_seen`; each instance yielded once | Unit: `test_collect_descendants_handles_cycles` |
| 30 | **(v4.6)** Self-reference (A.children=[A]) terminates; A yielded once | Unit: `test_collect_descendants_handles_self_reference` |
| 31 | **(v4.6)** BTA `_build_subgraph_spec` with two sub-queries whose workers share `flow_configs[0].initial_inferencer` produces ONE `_logger.warning` per collision under default `worker_isolation_check=True` (does NOT raise) | Unit: `test_recursive_sharing_logs_warning_per_collision` |
| 32 | **(v4.6)** Warning message contains: BTA name, both worker indices, instance id (hex), instance type name, remediation hint mentioning LazyConfigFactory / Fix #9 | Unit: `test_warning_message_format_and_content` |
| 33 | **(v4.6)** Setting `worker_isolation_check=False` skips the validation entirely (no descendant walk, no warnings, no perf cost) | Unit: `test_worker_isolation_check_false_skips_entirely` |
| 34 | **(v4.6)** After Fix #9 (LazyConfigFactory) lands, default `worker_isolation_check=True` produces ZERO warnings for the SOP plan integration test (no false positives) | Integration: SOP plan run with default policy; assert log contains zero "shares inferencer instance" warnings |
| 35 | **(v4.6)** When MULTIPLE shared instances exist across workers, ALL collisions are logged (one warning per collision); no batching that hides collisions | Unit: `test_multiple_collisions_all_logged` |
| 36 | **(v4.6)** Worker indices in the warning message are correct: `worker[i]` is the LATER detection; `worker[prev_i]` is the FIRST detection | Unit: `test_collision_worker_indices_correctly_attributed` |
| 37 | **(v4.6)** A non-`InferencerBase` worker (Mock or callable wrapper from a future factory variant) is gracefully skipped with a `_logger.debug` message — does NOT crash the validator | Unit: `test_non_inferencer_worker_gracefully_skipped` |
| 38 | **(v4.7)** All `_instantiate.py` modifications are made in `RichPythonUtils/src/rich_python_utils/config_utils/_instantiate.py` (the live file consumed by AgentFoundation), NOT in the stale `rovoteam/PythonUtils/...` mirror | Pre-flight: CI lint check that reads modified-files list and refuses any commit touching `python_utils/config_utils/` |
| 39 | **(v4.7)** Walker change targets the EXACT lines of the live file: `_filter_attrs_keys` at line 881, `*_factory` block at lines 944-972, `_FACTORY_MARKER` gate removal at lines 956-958 + 968-970 | Unit/integration: post-change diff must touch only these line ranges in the live file |
| 40 | **(v4.7)** `LazyConfigFactory` is importable as `from rich_python_utils.config_utils import LazyConfigFactory` (re-exported from `__init__.py` if needed) | Unit: `test_lazy_config_factory_importable_from_rich_python_utils` |
| 41 | **(v4.7)** `_ImportFactory` deprecation alias is importable as `from rich_python_utils.config_utils._instantiate import _ImportFactory` (preserves the same dotted path the existing `_apply_import_factory` site uses internally) | Unit: `test_import_factory_alias_importable_from_rich_python_utils` |

---

## §7 Risks

1. **Fix #2 interaction with `skip_existing`**: Need careful test to ensure pre-existing `final_deliverables/` files at destination ARE preserved, only nested directory structure is flattened
2. **Fix #3 disruption to in-flight runs**: Anyone with checkpoints from a `_round02`-naming run won't be able to resume after fix
3. **Fix #4 invasive — may break BTA's per-worker-completed markers**: Worker dispatch logic deeply assumes flat `worker_N` layout
4. **(v4.x) `switch_role()` parallel-safety**: NOT safe under concurrent `ainfer()` on the same inferencer. All current callers serialize naturally (MFDual `_step_propose_impl` is await-then-dispatch; BTA workers are independent instances), but FUTURE callers must check this. Mitigation: documented in the docstring; future hardening is an opt-in `_currently_inferring` flag (deferred).
5. **(v4.x) Audit-trail memory growth**: `_role_history` is unbounded per inferencer. In long-lived processes with many consensus iterations, it accumulates. Mitigation: cap to last N entries (e.g., 50) via a deque if profiling shows it matters; not pre-emptively optimised.
6. **(v4.x) `reset_session=True` default may surprise**: Callers migrating from direct mutation may not expect the session reset. Mitigation: docstring spells it out; the v4.x migration of `_step_propose_impl` deletes the redundant manual `reset_session` loop in the same commit so behavior is unchanged externally.
7. **(v4.x) `attrs` interaction**: `_role_history` AND `_pending_role_changes` are set via `object.__setattr__` to bypass attrs validators; this is consistent with the existing `_workspace` private-name pattern (`_InferencerBase__workspace`). Verified compatible.
8. **(v4.1) Audit-merge protocol coupling**: The templated layer relies on `_pending_role_changes` being CONSUMED-AND-CLEARED by base. If a future intermediate layer (between `TemplatedInferencerBase` and `InferencerBase`) ALSO uses the same buffer without clearing, audit entries can leak across calls. Mitigation: §4.5 acceptance §4-11 explicitly tests buffer lifecycle; the protocol is documented in the templated docstring; future intermediate layers must follow the same stash-and-let-base-merge convention OR establish a per-layer namespaced buffer if they need independent audits.
9. **(v4.1) MRO breakage if a future `InferencerBase` subclass adds a kwarg overlapping with `TemplatedInferencerBase`'s template kwargs**: `TypeError` would result. Mitigation: `_ROLE_RELEVANT_ATTRS` cumulative tuple makes the namespace owned by each layer explicit and discoverable; CI grep on `_ROLE_RELEVANT_ATTRS` declarations would catch overlaps in code review.
10. **(v4.2) Diagnostic-file proliferation**: pathological runs may emit MANY `*.diagnostic.txt` files (one per affected round × phase × anomaly type). Mitigation: filenames are deterministic so re-runs overwrite rather than accumulate; tooling can `find . -name '*.diagnostic.txt'` to enumerate. Acceptable trade-off — silent corruption is strictly worse than verbose diagnostics.
11. **(v4.2) Fix #7 Check 1 false positive on intentional cross-MFDual references**: A future use case might WANT one MFDual to reference another's workspace (e.g., shared evaluation harness). Mitigation: introduce `enable_cross_mfdual_audit_check: bool = attrib(default=True)` opt-out if such a case emerges; today no such caller exists.
12. **(v4.2) Fix #8 call-site coverage**: If a NEW `_record_round_audit()` call is added in a future phase without supplying `workspace_root_at_phase=`, that phase will silently revert to live-read semantics (Fix #8 fail-safe). Mitigation: deprecation warning when the kwarg is missing AND `enable_audit_snapshot_strict=True` (a new opt-in flag, off by default for back-compat); CI lint rule that flags new `_record_round_audit(` calls without the kwarg in code review.
13. **(v4.2) Race-window between phase-execution and audit-record still exists for the JSONL `round_log.jsonl` write** (which uses `inferencer._workspace.root` at log time, not the snapshot). The user-visible symptom (symlinks) is fixed by Fix #8, but the JSONL log retains the live-read race. Mitigation: extend Fix #8 to ALSO use `workspace_root_at_phase` for the JSONL `inferencer_workspace` field (~5 line change). Folded into Fix #8 effort.
14. **(v4.4) Performance overhead of LazyConfigFactory**: Each `factory()` call now runs the full `instantiate()` pipeline (alias resolution + walker + Hydra recursion + attrs filtering). For BTA with N workers, cost is O(N × tree_size) instead of O(tree_size + N). For typical N=2-8 BTA workers and tree_size ~15 inferencers, this is ~30-200ms total — negligible. Mitigation: profile first run after Fix #9 lands; if a hot path is impacted, cache the parsed `OmegaConf.create(config)` result in `LazyConfigFactory.__init__` (one-time cost) and only re-instantiate per call (deeper savings).
15. **(v4.4) OmegaConf interpolation resolution timing**: `${_params.default_inferencer}` is resolved at `OmegaConf.create()` time. After `copy.deepcopy(self._config_dict)`, the interpolation strings are still present in the dict (deep-copy preserves strings); `OmegaConf.create(config)` re-resolves them on each call. If `_params.default_inferencer` was MUTATED between two `factory()` calls, the second call would pick up the mutation. Mitigation: this is correct behavior (callers expect override-then-call to be reflected); documented as expected.
16. **(v4.4) Test fixtures that depend on shared instances**: Any test asserting `worker_0.flow_configs[0]['initial_inferencer'] is worker_1.flow_configs[0]['initial_inferencer']` would FAIL after Fix #9. Audit (verified 2026-05-10 19:13): no such test exists in `test_breakdown_then_aggregate.py` or `test_multi_flow_dual_inferencer.py`. Mitigation: pre-flight grep for `is.*flow_configs|flow_configs.*is|id(.*flow_configs` before merge.
17. **(v4.4) `__slots__` interaction with template_extra_feed mutation**: `LazyConfigFactory.__slots__ = ("_config_dict", "_injectables", "template_extra_feed")` allows mutation of `template_extra_feed` (it's a slot) but PREVENTS adding arbitrary attributes (e.g., `factory.foo = "x"` would raise `AttributeError`). This is intentional (matches the strict no-args contract spirit). Mitigation: documented in `LazyConfigFactory` docstring; if any caller tries to attach metadata, they should subclass.
18. **(v4.4) Cyclic import risk**: `LazyConfigFactory.__call__` imports `instantiate` from `_instantiate.py`, which imports `LazyConfigFactory` from `_lazy_config_factory.py`. The import in `__call__` is lazy (inside the method) per F9.1 to break the cycle. Mitigation: tested by F9-1 (which exercises the full call path); if anyone moves the import to module-level, the cycle would surface immediately at import time.
19. **(v4.5) Recursive walk perf overhead at scale**: `_collect_all_descendant_inferencers()` runs on every BTA worker construction. For typical N=2-8 workers × tree_size ~15 inferencers, cost is O(N × tree_size) = ~120 dict.add/in operations per BTA. Negligible. Mitigation: profile if BTA is constructed in a tight loop (e.g., per-request); if so, add `@functools.cache` on a per-worker basis. Not pre-emptively optimised.
20. **(v4.5) `_iter_child_inferencers()` coverage gap risk**: If a future orchestrator adds a NEW child-inferencer slot but forgets to override `_iter_child_inferencers()`, the recursive walk will miss those children and Fix #5 Part B will silently underreport sharing. Mitigation: when adding a new child slot, also extend the orchestrator's `_iter_child_inferencers()` override (already a established pattern per the agent's audit of MFDual:632, MFI:991, Dual:1943, BTA:1178). Recommend a code review checklist item.
21. **(v4.5) False-positive risk if `_iter_child_inferencers()` ever yields a singleton like `_template_manager`**: Today, the overrides yield only INFERENCER children (not template_manager, logger, workspace). If a future override accidentally yields a non-inferencer or a deliberately-shared singleton, all workers would "share" it and trigger false-positives. Mitigation: type-check `isinstance(inf, InferencerBase)` in the recursive helper before yielding; rejected as unnecessary today (the existing overrides are correct), but documented as a guard worth adding if a regression appears.
22. **(v4.6) Helper placement on `InferencerBase`** vs. as a static method on BTA (rejected alternative from integration-memo): The integration-memo's static `_collect_tree(root)` on BTA was simpler at the call site but architecturally inferior — `_iter_child_inferencers` is defined on `InferencerBase`, so the recursive walk over those children belongs on the same type. Risk: any future orchestrator that needs the same capability would need the helper, and a static-on-BTA design forces re-implementation. v4.6 puts the helper on `InferencerBase` as `_collect_all_descendant_inferencers`, mirroring `pre_retry`'s pattern and avoiding `hasattr` defensive checks. Mitigation: none needed — the placement IS the mitigation.
23. **(v4.6) WARN-only-in-v1 misses real bugs**: A serious shared-instance bug today produces only a log line, not a test failure. Risk: developers ignore the warning, the bug ships. Mitigation: (a) add a CI grep that fails the build if `"shares inferencer instance"` appears in test logs; (b) the OPTIONAL future `worker_isolation_strict: bool` provides escalation when ready. Acceptance criterion F5B-8 explicitly verifies post-Fix-#9 zero-warning baseline as a tight signal.
24. **(v4.6) `worker_isolation_strict` future-extension cost**: If we later need strict mode, adding a second attrib + branching in `_validate_worker_isolation` is a 5-line change. Acceptable. Mitigation: documented in §F5B.8 phase 4 as the explicit escalation path.
25. **(v4.6) Naming collision risk on `worker_isolation_check`**: Audit (2026-05-10 19:36) confirms zero existing references to this name in `AgentFoundation/src/` or any YAML configs under `CoreProjects/`. Mitigation: name claimed; if a future feature needs the name for something else, rename via a one-shot grep+replace (the attrib is on a single class).
26. **(v4.7) Stale-mirror divergence risk**: The `rovoteam/PythonUtils/` and `CoreProjects/RichPythonUtils/` `_instantiate.py` files differ by ~10 KB (28 KB vs 39 KB). If a developer accidentally implements Fix #9 in the stale mirror, the live consumer (AgentFoundation) will see no behavioral change while tests against the stale tree pass. Mitigation: (a) CI lint preventing commits that touch `rovoteam/PythonUtils/config_utils/`; (b) a top-of-file comment in the stale `_instantiate.py` saying "DO NOT EDIT — historical mirror; live source is `CoreProjects/RichPythonUtils/`"; (c) acceptance criterion 38 codifies the path requirement.
27. **(v4.7) Acceptance-criterion-38 false-positive risk**: A blanket "no commits to `python_utils/`" CI lint may block legitimate work on the historical mirror (e.g., backporting unrelated fixes for an external Atlassian-internal consumer). Mitigation: lint should warn-and-require-explicit-bypass-comment rather than hard-block; reviewer acks the bypass.
28. **(v4.7) Hidden references in tests/configs may still point to stale path**: Even after this plan corrects the documentation, a test file or YAML config may contain hardcoded references to `python_utils.config_utils`. Audit: pre-flight `grep -r "python_utils\." /Users/tchen7/MyProjects/CoreProjects/AgentFoundation/` (2026-05-10 20:00) returned no live imports. Mitigation: re-run the audit immediately before merging Fix #9.

---

## §8 Open Questions

1. For Fix #4: should we use the simpler "everything under worker_N/checkpoints + worker_N/children/flow_*" layout, or the more elaborate "worker_N/_workflow + worker_N/flow_*"?
2. Should Fix #3 alone be sufficient, or does the user actually want Fix #4's full nesting?
3. Is the `aggregator/` empty placeholder bothering more than just visually? (No data loss; just clutter)
4. **(v4.x)** Should `switch_role()` enforce a `from_role` argument too (so transitions are auditable as `from→to` pairs, not just `to`)? Current design records only `to_role` — `from_role` would require either tracking the previous role on the inferencer (state) OR requiring callers to pass it (boilerplate). Defer: callers can grep `_role_history[-2]["to_role"]` if needed.
5. **(v4.x)** Should `_ROLE_RELEVANT_ATTRS` be runtime-enforced (i.e., `switch_role(**kwargs)` rejects keys not in the tuple)? v4.1 design uses explicit kwargs — adding a new field requires editing the signature, which is the discoverability point. The tuple is documentation/lint-target only and serves cumulative MRO introspection. If we move to `**kwargs` dispatch later, the tuple becomes the validation set.
6. **(v4.x)** Should aggregator's `switch_role()` call (Migration #4) supply `template_key`/`template_root_space`? Today the aggregator's `STRUCTURED_AGGREGATION_DEFAULTS` is set at construction and never mutated, so passing them as None (preserve) is correct — but we could also pass them explicitly for symmetry with reviewer/fixer. Defer until a use case emerges.
7. **(v4.1)** Should `_pending_role_changes` be a per-layer-namespaced dict (e.g., `_pending_role_changes["templated"] = {...}`) instead of a flat dict? Current flat-dict design relies on the convention "only ONE intermediate layer between base and the caller will ever stash here per call". Adding a third layer would force a richer protocol. Defer until we actually have a third intermediate layer.
8. **(v4.1)** Should the `is_deliverable_boundary` kwarg actually be on the BASE layer? It's declared on `InferencerBase` (line 182), so YES per the layering principle — but it's semantically tied to deliverable surfacing, which is more a leaf concern than a base concern. v4.1 leaves it on base for consistency with where the attrib is declared; revisit if a use case shows it should move.
9. **(v4.2)** Should Fix #7 diagnostics be promoted to ERRORS (raise) rather than file + log markers? Pros: louder, harder to ignore. Cons: would break runs that have benign sharing (e.g., test fixtures). Recommendation: keep file + log for now; add an opt-in `audit_strict_on_anomaly: bool = attrib(default=False)` to raise on detection in CI.
10. **(v4.2)** Should the LEAKAGE / ALIASING / OVERWRITE diagnostics also be aggregated into a top-level `_audit_anomalies.json` at the MFDual root for easier post-run scanning? Today they're scattered across `round_NN/` directories. A summary file would make `find . -name '*.diagnostic.txt'` redundant.
11. **(v4.2)** Should Fix #8 SNAPSHOT include MORE state than just the workspace root (e.g., template_key, output_is_deliverable as they were at phase-execution time)? Today only the symlink target benefits from the snapshot; if we ever extend the JSONL log to include those fields, they'd suffer the same race. Defer until a use case emerges; the architecture supports adding more snapshot fields by extending the kwarg set.
12. **(v4.2)** Anomaly 6's evidence shows that v4.1's `switch_role()` is necessary but not sufficient — should the plan promote Fix #5 strict-mode to MANDATORY (default-on) rather than warn-mode-then-strict-mode? Faster eradication of the bug class but breaks anyone who today relies on benign sharing. Recommendation: keep the phased rollout (warn → strict over one release) for safety.
13. **(v4.4)** Should `LazyConfigFactory` be made the universal config-instantiation primitive — i.e., should ALL nested `_target_` blocks (not just `_factory`-suffix fields) become `LazyConfigFactory` by default? This would eliminate ALL eager instantiation across the codebase, but would break the (intentional) shared-singleton pattern for `_template_manager`, loggers, etc. Recommendation: keep the `_factory`-suffix opt-in convention; the suffix IS the discoverable signal.
14. **(v4.4)** Should `LazyConfigFactory.__call__` accept an OPTIONAL `**overrides` kwarg that injects per-call config overrides before re-instantiation (e.g., `worker_factory(template_key="custom")`)? Pros: powerful per-call customization. Cons: violates the "fully bound, call with no args" contract; orchestrators would need richer recognition logic. Recommendation: defer; if needed, add an explicit method like `factory.with_overrides(template_key="x")()`.
15. **(v4.4)** Should the deprecation warning on `_ImportFactory` be REMOVED in this same release (cold migration) since no external consumers exist? Pros: cleaner code immediately. Cons: defensive — external repos checked out at older commits might import it. Recommendation: keep alias for ONE release cycle (Fix #10's stance); remove in v4.5.
16. **(v4.4)** Should `LazyConfigFactory` provide a `clone()` method that returns a copy with fresh `template_extra_feed`? Today's design forces callers to construct a new factory if they want isolated `template_extra_feed`. Likely YES if any caller pattern emerges. Defer to first concrete use case.
17. **(v4.4)** Should the walker produce a different factory class for `dict`-typed `worker_factory` (heterogeneous workers) vs single `_target_:` (homogeneous)? Today both go through the same `LazyConfigFactory` path. The orchestrator's `isinstance(x, (functools.partial, LazyConfigFactory))` check works for both, but the dict-of-factories case has additional structure (per-task-type lookup) that BTA handles separately. Recommendation: keep one class; orchestrators handle the dict-routing layer.
18. **(v4.5)** Should `_collect_all_descendant_inferencers()` accept a filter callback (e.g., `predicate=lambda inf: not isinstance(inf, ToolAsInferencer)`) so callers can scope the traversal? Today it yields ALL descendants; callers filter on the consumption side. Defer until a use case emerges; the API is small enough to extend non-breakingly.
19. **(v4.5)** Should `SharingScope` have a fourth level `RECURSIVE_INCLUDING_INJECTABLES` that ALSO checks `_template_manager` / `_logger` for sharing? Today injectables are intentionally shared singletons (see §F9.5 F9-3) so cross-worker sharing of them is EXPECTED. Adding the level would be misleading. Recommendation: never add this level; it would invert the v4.5 invariant.
20. **(v4.5)** Should the `_sharing_anomaly.diagnostic.txt` file (WARN policy) be APPEND-mode rather than truncate? Today it's truncate (per WARN's `open(path, "w")`), so a re-run overwrites the previous report. Pros of append: full history. Cons: file grows unbounded. Recommendation: keep truncate; if append-history is desired, add a separate `_sharing_anomaly_history.jsonl` file (deferred).
21. **(v4.5)** Should Fix #5 Part B be moved to BASE class (so DualInferencer / MultiFlowDualInferencer / any orchestrator with worker-children gets it for free)? Today it's BTA-specific. Pros: universal protection. Cons: most orchestrators don't construct multiple workers per call (BTA is the canonical multi-worker pattern). Recommendation: keep BTA-specific; if another multi-worker orchestrator emerges, it can call `_verify_worker_sharing()` itself or we promote the method then.
22. **(v4.6)** Should the recursive helper be exposed publicly (`InferencerBase.collect_all_descendant_inferencers` without leading underscore) given that it's a useful introspection tool for users? Today it's underscored ("internal API"). Defer; user-facing API graduation is a separate decision.
23. **(v4.6)** Should the warning message in `_validate_worker_isolation` use structured logging (e.g., `extra={"shared_id": iid, "worker_a": i, "worker_b": prev}`) so log aggregators can index it? Today it's plain string. Recommendation: add `extra=` kwarg if the standard `_logger` supports it; non-blocking.
24. **(v4.6)** Should `_validate_worker_isolation` ALSO check for sharing of the `aggregator_inferencer` against any worker descendant (cross-pool sharing of the multi-flow aggregator with a per-worker inferencer)? Today the helper walks workers independently; aggregator sharing is a separate dimension. If it occurs in practice, add a second pass that walks `[self.aggregator_inferencer]` against the same `seen` dict. Defer until WARN logs reveal it.
25. **(v4.6)** Should the helper accept an OPTIONAL `predicate: Callable[[InferencerBase], bool]` so callers can scope the traversal (e.g., skip leaves that are intentional singletons)? v4.5 raised this same question (#18) — defer until a use case emerges; the API is small enough to extend non-breakingly.
26. **(v4.7)** Should the stale `rovoteam/PythonUtils/` mirror be DELETED entirely as part of Fix #9, given that the live AgentFoundation consumer never imports from it? Pros: zero divergence risk. Cons: external Atlassian-internal consumers might depend on it (cannot verify from local checkout). Recommendation: defer deletion; add the "DO NOT EDIT" header (per risk 26 mitigation) and revisit after one release cycle.
27. **(v4.7)** Should `LazyConfigFactory` be re-exported from `rich_python_utils/config_utils/__init__.py` as a top-level public symbol, OR kept module-private at `_instantiate.py` level (matching `_ImportFactory`'s current visibility)? Plan A's example uses `from python_utils.config_utils import ...` (top-level). The codebase audit shows `__init__.py` already exports `instantiate` and `load_config` publicly; consistency suggests `LazyConfigFactory` should also be top-level public. Recommendation: top-level re-export.
28. **(v4.7)** Plan A's Fix #6 example (`MFDual._reassign_role_workspace`) calls `inferencer.switch_role(new_role=role_name, workspace=role_ws, template_key=new_key, template_root_space=new_root)` — passing template kwargs directly to the (potentially) base-only inferencer. The MRO design in v4.1 routes these correctly through `TemplatedInferencerBase`, but only IF the inferencer is a `TemplatedInferencerBase` descendant. For a hypothetical raw `InferencerBase` worker, this would `TypeError`. Today MFDual's flow inferencers are ALL templated (verified earlier in v4.1's MRO confirmation table) so this is moot. Open question: should `_reassign_role_workspace` filter kwargs based on inferencer type? Recommendation: NO — the TypeError IS the desired loud failure if the invariant is violated.

---

## §9 Provenance

- v1.0 (2026-05-10 16:32): Initial plan after observed anomalies in run `task_task-7522157d_20260510_133627`
- Diagnoses based on hard evidence from workspace inspection + code reads
- v3.0 (2026-05-10 17:42): Added Anomaly 5 (Role-Mutation Template Key Drift — CRITICAL) + Fix #6 (Hybrid Option A+C); root-caused `template_key` not refreshed when MFDual reuses winning flow inferencer as fixer.
- v4.0 (2026-05-10 18:17): Promoted Fix #6's ad-hoc mutation extension into a first-class `InferencerBase.switch_role()` API. Added new §4 "Architectural Refactor" capturing Phase 1 (base method + `_ROLE_RELEVANT_ATTRS` constant + `_role_history` audit trail) and Phase 2 (migrations of `_reassign_role_workspace`, `_select_reviewer_and_fixer` direct mutations, and Fix #1 aggregator workspace rebind) as a single atomic commit. Restructured Fix #6 as a thin caller of the new API. Renumbered Implementation Order → §5, Acceptance Criteria → §6, Risks → §7, Open Questions → §8, Provenance → §9. Added 5 v4.0-specific acceptance rows + 4 v4.0-specific risks + 3 v4.0-specific open questions.
- v4.1 (2026-05-10 18:24): Refined v4.0's single-method `switch_role()` into a **layered design** that respects the documented `InferencerBase` ↔ `TemplatedInferencerBase` boundary (per `inferencer_base.py:184-198`). Added §4.0 MRO confirmation table; split §4.1 into Phase 1a (base layer, only `workspace`/`output_is_deliverable`/`is_deliverable_boundary`) and §4.2 Phase 1b (template layer, all template attribs); introduced `_pending_role_changes` audit-merge protocol so layered calls produce ONE audit entry; removed `hasattr` defensive guards (attribs definitely exist where they're declared); replaced silent-skip with TypeError-on-unknown-kwarg for typo safety; updated `_ROLE_RELEVANT_ATTRS` to cumulative-tuple convention; added 4 new v4.1-specific acceptance rows (§4-9..§4-12) + 2 new v4.1-specific risks (audit-merge protocol coupling, MRO breakage) + 2 new v4.1-specific open questions. Effort revised from ~3.5h → ~4h (+30 min for second layer + audit-merge protocol). Caller code is UNCHANGED — Python's MRO handles dispatch transparently.
- v4.2 (2026-05-10 18:32): Added Anomaly 6 (Cross-Worker / Role-Inverted Audit Symlinks — CRITICAL) with hard-evidence proof of the instance-sharing hypothesis from Anomaly 1 (worker_1's `round_01/fix → worker_0/.../review_inferencer` cross-link; worker_1's `round_01/review` and `round_01/fix` BOTH symlinking to `worker_1/.../fixer_inferencer` — within-worker role aliasing). Decomposed into Bug 6a (cross-worker instance sharing), Bug 6b (within-worker role aliasing), Bug 6c (audit-time vs phase-time snapshot drift). Added Fix #7 (audit symlink hardening — three checks emit `.LEAKAGE.diagnostic.txt`, `.ALIASING.diagnostic.txt`, `.OVERWRITE.diagnostic.txt` sibling files when pathology detected) + Fix #8 (snapshot-at-phase-time semantics — `_record_round_audit()` accepts explicit `workspace_root_at_phase=` kwarg captured eagerly at phase execution; 4 call-site migrations). Updated §5 Implementation Order to interleave Fix #5 Part A → Fix #7 → Fix #8 EARLY for diagnostic value; promoted Fix #5 Part B to AFTER Fix #7's baseline confirmation. Added 9 v4.2-specific acceptance rows (§6 rows 12-20) + 4 v4.2-specific risks (rows 10-13) + 4 v4.2-specific open questions (rows 9-12). Effort revised from ~7.75h → ~10.75h (+3h for audit-layer defense in depth). Defense-in-depth rationale: v4.1's per-call atomicity is necessary but insufficient when instances are shared AND mutated concurrently — Anomaly 6's symlink evidence directly proves this.
- v4.3 (2026-05-10 18:39): REFINED ATTRIBUTION based on cross-run comparison evidence between prior run `task-cbaf8f2b_20260510_033521` (03:35, May 10) and recent run `task-7522157d_20260510_133627` (13:36, May 10). Both runs share the same code state (no git commits between them) yet exhibit cross-worker symlink leakage (`worker_1/.../fix → worker_0/...`) — proves cross-worker sharing is PRE-EXISTING. Attribution corrections: (a) §2.6 Bug 6a narrative refined — sharing is at the FLOW-POOL layer (`flow_configs[i].initial_inferencer` / `followup_inferencer`), NOT the top-level role-slot layer; the role-slot symptom is a downstream alias created by `mfdual_self_promotion_gap_INTEGRATED_plan.md`'s `fixer_match_winner` + `reviewer_match_second` features. (b) Added new §2.7 "Cross-Run Evidence" with side-by-side symlink comparison table proving (i) cross-worker sharing was pre-existing, (ii) the integrated plan EXPOSED but didn't INTRODUCE it, (iii) `_reassign_role_workspace()` made the SYMLINK TARGETS look cleaner (canonical role names) which paradoxically makes within-worker aliasing more visually confusing. (c) Documented the actual root-cause layer (YAML topology / Hydra factory wiring of `flow_configs[*]` entries — outside this plan's scope; needs separate topology audit). (d) Refined Fix #5's scope guidance: deep-walk MUST traverse `flow_configs[*].initial_inferencer` and `flow_configs[*].followup_inferencer` AND `inferencer_pool` entries (per the integrated plan's `reviewer_match_second` resolver). The `reviewer_match_second` and `fixer_match_winner` features are CORRECT features and should NOT be reverted; only the underlying YAML topology must be hardened. Effort + acceptance criteria + fix mechanics unchanged from v4.2 — v4.3 is purely an attribution / scope refinement.
- v4.8 (2026-05-10 20:07): TARGETED CRITICAL-THINKING PASS on external-agent feedback (7 items submitted; 2 valid, 5 rejected after ground-truth verification). **VALID & APPLIED**: (FB2) Fix #2's sketch was unintentionally rewriting `surface_outputs_from`'s live API — verified at `inferencer_workspace.py:130-191` the live signature has `namespace: Optional[str]` kwarg + uses `deliverables_dir` (NOT `outputs_dir`) — replaced the multi-line speculative sketch with Plan A's elegant single-line `dirs[:] = [d for d in dirs if d != "final_deliverables"]` `os.walk` prune that preserves all live API contracts. Added 5 new acceptance criteria (F2-1 through F2-5). Effort revised from ~75 min → ~1h. (FB3) Fix #3's `("followup_inferencer", None)` tuple risked `parent.child(None)` crash at the propagation consumer — verified at `inferencer_workspace.py:267` that `child(name: str)` has no None tolerance — added Option A.1 (PREFERRED: omit entry entirely) and Option A.2 (explicit None-guard `if suffix is None: continue` at consumer) with crash explanation pointing to the exact `TypeError: can only join str` failure mode. **REJECTED with documented reasoning**: (FB1) "Wrong file paths at lines 1313-1565" — bogus line citations; verified line 1315 is `import copy`, 1396 is `return f"LazyConfigFactory(...)"`, 1481 is acceptance text. The remaining `rovoteam/PythonUtils` refs in v4.8 are intentional (§2.10 educational-text describing the OLD wrong path; v4.7 acceptance #38 about CI lint that BLOCKS this path; v4.7 risk #26 about divergence; v4.7 open-question #26; v4.4/v4.7 historical provenance). (FB4) "§3 Fix #5 Part B not v4.6 updated" — verified §3 Fix #5 Part B IS fully v4.6-converted: header reads "[v4.6 INTEGRATED]", contains the v4.5-vs-v4.6 comparison table, uses `worker_isolation_check: bool = True`, calls `_validate_worker_isolation` via `_collect_all_descendant_inferencers()`. The agent likely confused Part A's diagnostic-logging hand-enumeration (which is FINE for logging) with Part B's sharing detection. (FB5) "`import time` missing in switch_role example" — verified line 1779 has `import time` 5 lines before the `time.time()` usage at line 1825. (FB6) "Fix #4 invasiveness" — already labeled "highest risk" in §5 Implementation Order entry 11; recommendation to defer is consistent with existing positioning. (FB7) "`_pending_role_changes` buffer pattern fragility" — already covered by §4-11 acceptance criterion + §7 risk #4. **Net delta**: ~+50 lines (Fix #2 expansion + Fix #3 expansion + this provenance entry); **no API changes** to the existing fix designs; **no new fixes**; effort estimates unchanged (~16.2h total). v4.8 is a CORRECTNESS PATCH layer over v4.7, not a redesign. **Honest assessment**: the feedback agent's process was good (critical-thinking review of a long plan) but its specific claims were 2/7 accurate. The ~71% rejection rate validates the user's instruction to "not blindly trust feedback" — applying all 7 items would have caused regressions on FB1 (would have re-edited correct paths into wrong-versions) and FB4 (would have re-applied the v4.5 design that v4.6 explicitly superseded).

- v4.7 (2026-05-10 20:00): MAJOR FILE-PATH CORRECTION based on a re-comparison with Plan A (`/Users/tchen7/.claude/plans/given-all-the-discussions-splendid-lantern.md`) which had been substantively REWRITTEN between v4.6 and v4.7 (47 lines → 253 lines; from a decision-memo into a full standalone integrated plan). Plan A's substantive correction over Plan B: **the canonical `_instantiate.py` file is at `/Users/tchen7/MyProjects/CoreProjects/RichPythonUtils/src/rich_python_utils/config_utils/_instantiate.py`, NOT `/Users/tchen7/MyProjects/rovoteam/PythonUtils/src/python_utils/config_utils/_instantiate.py`** — verified by direct grep of `AgentFoundation/src/` imports (registered_targets.py:8, factories.py:16, mock_bta_components.py:10 all import `from rich_python_utils.config_utils import ...`; zero live imports reference `python_utils.config_utils`). The `rovoteam/PythonUtils/` copy is a stale older mirror (28 KB Apr 29 vs the live RichPythonUtils 39 KB May 4). Verified live line numbers: `_FACTORY_MARKER` at line 35, `_ImportFactory` at line 38, `_apply_import_factory` at line 356, `_filter_attrs_keys` at line 881, `*_factory` handling block at lines 944-972, the buggy `val["_partial_"] = True` at lines 959 and 972, the gated `_factory_configs.append(...)` at lines 957 and 970 (gated on `_FACTORY_MARKER in val`). All v4.4-v4.6 line-number references were OFF BY ~250 lines because they cited the stale mirror. Added new **§2.10 "File-Path Correction: RichPythonUtils is the Live Codebase"** with the verification table, the live-line-number table, the bug+fix mechanic re-described against the live file, and an "audit checkpoint" requirement (any future line-number reference to `_instantiate.py` MUST be paired with a grep proving the consumer imports the SAME path). Bulk-corrected ALL 6 stale references throughout the plan: 4 in Fix #9 (file paths + line ranges), 1 in Fix #10 (audit grep paths), 1 in §2.8 (mechanical root cause source-code site). Preserved historical-context blocks (the v4.4 provenance bullet + the §2.10 educational text) which intentionally cite the OLD wrong path. Added 4 v4.7 acceptance rows (§6 rows 38-41: target-file path, line-range fidelity, top-level import, alias import) + 3 v4.7 risks (§7 rows 26-28: stale-mirror divergence, CI lint false positives, hidden test references) + 3 v4.7 open questions (§8 rows 26-28: stale mirror deletion, public top-level re-export, Plan A's switch_role kwarg routing). Mechanical attribution and design decisions otherwise UNCHANGED — v4.7 is a path/line-number correction layer over v4.6's design. Effort estimates UNCHANGED (~16.2 hours total). **Honest assessment**: Plan B (v4.4-v4.6) had a real factual error that would have caused implementation-time confusion and possibly a wrong-tree commit. Plan A's expansion caught it because Plan A's author re-verified against the live tree before promoting to standalone. v4.7 takes the BEST OF BOTH: Plan B's architectural depth + Plan A's path/line-number precision. If we only pick one plan, the answer is now NUANCED — see end-of-plan note.

- v4.6 (2026-05-10 19:55): INTEGRATED v4.5's deep architectural foundations with the integration-memo simplification (`/Users/tchen7/.claude/plans/given-all-the-discussions-splendid-lantern.md`) of Fix #5 Part B. Three categories of changes: (1) **Critical correction**: BTA does NOT have `self.workers` or `_make_worker_iter()` — verified 2026-05-10 19:52 against `breakdown_then_aggregate_inferencer.py:1408-1804`. Workers are constructed PER-INFERENCE-CALL inside `_build_subgraph_spec()` via `worker_factory()` calls at lines 1510-1512. The `_validate_worker_isolation()` call site is therefore at the END of `_build_subgraph_spec()`, receiving the locally-collected list of worker instances. (2) **API simplification**: dropped v4.5's `SharingScope` (NONE / TOP_LEVEL / RECURSIVE) and `SharingPolicy` (STRICT / WARN) enums in favor of a SINGLE `worker_isolation_check: bool = True` attrib on BTA. Dropped the `_sharing_anomaly.diagnostic.txt` file convention in favor of standard `_logger.warning`. Renamed `_verify_worker_sharing()` → `_validate_worker_isolation()` (the integration-memo's naming, which describes the property being verified rather than the failure mode). (3) **Architectural retentions from v4.5**: kept the base-class helper `InferencerBase._collect_all_descendant_inferencers()` placement (NOT static-on-BTA per the integration-memo's `_collect_tree`); kept the `pre_retry`-mirrored `_seen` cycle-safety pattern; kept reuse of existing `_iter_child_inferencers()` overrides on BTA / MultiFlow / Dual / MFDual; kept all 4 cycle/self-ref/leaf/MFDual correctness tests (F5B-1 through F5B-4). Trimmed acceptance criteria from 12 → 11 (dropped enum-permutation tests; added `non_inferencer_worker_gracefully_skipped` test). Updated 1 v4.5 risk and added 4 v4.6-specific risks (helper placement, WARN-only-in-v1 misses real bugs, future strict-mode extensibility, naming collision risk). Added 4 v4.6-specific open questions. Effort revised from ~3h → ~2.25h (the −45 min reflects the simpler API). Total plan effort: ~16.95h → ~16.2h. **Honest assessment carried forward**: the integration-memo correctly identified that v4.5 was over-engineered for v1 needs (STRICT-by-default risks breaking legitimate-sharing fixtures; enum proliferation creates 6 combinations with several no-use-case states); v4.5 correctly identified that the integration-memo's static-on-BTA + `hasattr`-defensive-checks were architecturally inferior. v4.6 takes the BEST OF BOTH: simple v1 surface (boolean + WARN-only), correct architectural placement (base-class helper + type-safe iteration), correct call-site (corrected to `_build_subgraph_spec` per ground-truth code audit), and a documented escalation path (`worker_isolation_strict` for future) without implementing it today.

- v4.5 (2026-05-10 19:33): EXPANDED Fix #5 Part B from a top-level-only `id(worker)` check into a RECURSIVE descendant-walk that catches sharing at ANY depth in the worker tree (the actual Anomaly 6 sharing site is at `flow_configs[*].initial_inferencer`, NOT at `id(worker)`). Added new base-class helper `InferencerBase._collect_all_descendant_inferencers()` that mirrors `pre_retry`'s recursive `_seen` cycle-safety pattern (verified at `inferencer_base.py:589`). Replaced the boolean `allow_worker_sharing` attrib with two ORTHOGONAL enums: `SharingScope` (`NONE` / `TOP_LEVEL` / `RECURSIVE`, default `RECURSIVE`) and `SharingPolicy` (`STRICT` / `WARN`, default `STRICT`). Reuses the existing `_iter_child_inferencers()` overrides on BTA / MultiFlow / Dual / MFDual (verified to cover all relevant child slots: aggregator, flow_configs entries, base/review/fixer inferencers, candidate pools). Added 10 v4.5 acceptance rows (§6 rows 28-37) covering leaf/MFDual/cycle/self-ref correctness, three-axis policy combinations, no-false-positive baseline, multi-collision reporting, independent overridability. Added 4 v4.5 risks (§7 rows 19-22: recursive walk perf, future orchestrator coverage gap, false-positive risk if helper yields non-inferencers, `allow_worker_sharing` removal back-compat). Added 5 v4.5 open questions (§8 rows 18-22). Effort revised from 50 min → 3 hours for Fix #5 Part B (the +2.25h pays for recursive correctness + 8 new test cases + new base-class helper). Total plan effort: ~14.7h → ~16.95h. Critical insight rejected: a single three-level enum would NAME the buggy state ("AllowSharedChildInferencers" IS Anomaly 6) as a permitted level — orthogonal axes are strictly better because behavior and scope are separate concerns.

- v4.4 (2026-05-10 19:15): PINPOINTED MECHANICAL ROOT CAUSE in `python_utils/config_utils/_instantiate.py:711` (vanilla `_partial_: true` injection produces `functools.partial` with EAGERLY-INSTANTIATED nested children baked into `keywords`, shared across every `factory()` call) — verified by direct code reads of `_walk` (lines 396-414), `_filter_attrs_keys` (lines 697-724), Hydra's recursive instantiation semantics, and `_ImportFactory.__call__` (lines 49-60 — already does the right thing but is gated on `_import_:` directive usage). Added §2.8 "Mechanical Root Cause" with full walkthrough + explicit rejection of "deep-copy at BTA" alternative (lists 7 fragile concerns: workspace propagation, template_manager singleton, logger sharing, lock semantics, attrs validators, weakref registries, manual topology reproduction). Added §2.9 "BTA Recognition Gap" documenting that today's `isinstance(x, functools.partial)` is the ONLY structural recognition mechanism — no formal protocol; `_ImportFactory` is silently miscategorized as bare-callable. Added Fix #9 (`LazyConfigFactory` universal re-instantiation protocol) — new class at `_lazy_config_factory.py` with strict no-args `__call__`, `__slots__`-protected attribs, `template_extra_feed` duck-typing for child propagation, OmegaConf re-resolution per call, NOT-deep-copied injectables (preserves singleton sharing). Walker change in `_filter_attrs_keys`: make `_factory_configs.append(...)` UNCONDITIONAL for `_factory`-suffix fields (was only on `_FACTORY_MARKER`). `_apply_import_factory` renamed to `_apply_lazy_factory`. BTA `isinstance(x, (functools.partial, LazyConfigFactory))` recognition update at lines 1503-1511. Added Fix #10 (`_ImportFactory` deprecated alias subclass) — zero-breaking-change migration verified by audit (no external imports found). Added 7 v4.4 acceptance rows (§6 rows 21-27), 5 v4.4 risks (§7 rows 14-18: perf overhead, interpolation timing, test fixture audit, slots interaction, cyclic import), 5 v4.4 open questions (§8 rows 13-17). Restructured §5 implementation order: Fix #9 + #10 promoted to FIRST priority (eliminates root cause); §4 layered switch_role() and Fixes #5/#7/#8 demoted to "DEFENSIVE INSURANCE LAYERS" (should never fire after Fix #9 lands; catch regressions). Total effort revised from ~10.75h → ~14.7h (+4h for Fix #9 architectural primitive). Defense-in-depth model formalized as 6 layers (Layer 0 root-cause elimination → Layer 5 strict-mode regression guard).
depth model formalized as 6 layers (Layer 0 root-cause elimination → Layer 5 strict-mode regression guard).
