# MFDual Self-Promotion Gap — Fix Plan

**Status**: DRAFT v3 — expanded scope (5 related issues, all variations of the same role-contract gap)
**Created**: 2026-05-10
**Severity**: 🟡 MEDIUM — runs work end-to-end empirically, but architectural inconsistency leaves rich content stranded at MFDual level (2.2 KB summary surfaces upward instead of full 47 KB deliverable)
**Scope**: 1 file change (`multi_flow_dual_inferencer.py` _ainfer extension), or 1 YAML change (worker_factory)

---

## §1 The Anomaly — Hard Evidence from Run `task_task-cbaf8f2b_20260510_033521`

For an MFDual at `children/base_inferencer/children/worker_0/`:

| Path | Size | What it contains |
|---|---|---|
| `worker_0/outputs/output.md` | **2,205 bytes** | LLM TEXT response — a SUMMARY |
| `worker_0/outputs/final_deliverables/` | **EMPTY** | Should contain the deliverable |
| `worker_0/children/fixer_inferencer/outputs/output.md` | **47,830 bytes** | The actual rich deliverable |
| `worker_0/children/fixer_inferencer/outputs/final_deliverables/` | **EMPTY** | Should contain self-promoted copy |
| `worker_0/children/base_inferencer/outputs/final_deliverables/output.md` | 61,967 bytes | Inner BTA aggregator's full output (correctly promoted) |

The chain breaks in TWO places:
1. **Inner Dual's fixer_inferencer/outputs/final_deliverables/ is EMPTY** — fixer wrote `output.md` but didn't self-promote
2. **MFDual's outputs/final_deliverables/ is EMPTY** — pass-through surfacing didn't fire because (1) had nothing to surface

End result: MFDual's `outputs/output.md` only has the LLM's text summary (2.2 KB), not the rich deliverable (47 KB).

---

## §2 ROOT CAUSE (verified post-investigation 2026-05-10 09:45)

### Definitive Mechanism: `fixer_match_winner: true` Mutation Loses `output_is_deliverable`

**Hard evidence from verification**:

1. **Top-level fixer** (`children/fixer_inferencer/`) — has `output.md` (92 KB) + `final_deliverables/output.md` + `.self_promoted` marker + `output_manifest.json`. **Self-promotion FIRED correctly.**
2. **Inner fixer** (`worker_0/children/fixer_inferencer/`) — has `output.md` (47 KB) + EMPTY `final_deliverables/` + NO marker + NO manifest. **Self-promotion DID NOT FIRE.**
3. **Both fixer workspaces have `final_deliverables/` subdir created** — meaning `use_final_deliverables_folder=True` propagated correctly via `child()`. So the workspace flag is NOT the issue.

**The mechanism**:

```yaml
worker_factory:
  __default__:
    _target_: MultiFlowDual
    fixer_match_winner: true            # ← THE MECHANISM
    flow_configs:
      flow_0:
        initial_inferencer:
          _target_: ${_params.default_inferencer}
          # NOTE: NO output_is_deliverable: true ← THE GAP
      flow_1:
        initial_inferencer:
          _target_: ${_params.default_inferencer}
          # NOTE: NO output_is_deliverable: true ← THE GAP

    fixer_inferencer:                   # ← only DECLARED here
      _target_: ${_params.default_inferencer}
      output_is_deliverable: true       # ← only set here, but mutated away at runtime
```

When MFDual completes propose phase, `fixer_match_winner: true` mutates `self.fixer_inferencer` to be the **winning flow's `initial_inferencer` instance** (NOT the YAML-declared `fixer_inferencer`). This mutated instance has `output_is_deliverable=False` (the attribute default; never set in YAML for `flow_configs.*.initial_inferencer`).

Then when the inner Dual's fix phase writes its output:
```python
# inferencer_base.py:754
if self.output_is_deliverable:           # ← False on the mutated instance
    fd = getattr(ws, "deliverables_dir", None)
    if fd is not None:                   # ← True (workspace flag was correct)
        # copy + write marker + emit_manifest    # ← NEVER RUNS
```

The `if self.output_is_deliverable: ...` gate is `False`, so:
- `output.md` IS written (this happens regardless)
- BUT `final_deliverables/output.md` is NEVER copied
- `.self_promoted` marker is NEVER created
- `output_manifest.json` is NEVER emitted

Then the inner Dual's `_finalize_response` calls `surface_outputs_from(active_ws=fixer_ws)` — but `fixer_ws.has_deliverables` is `False` (empty folder), so the surfacing silently no-ops.

This cascades up: MFDual's `_finalize_response` finds nothing to surface either; outer BTA sees the 2.2 KB summary instead of the 47 KB deliverable.

### Original (incorrect) hypothesis — Two Coupled Layers

### Layer A — The fixer leaf in MFDual's worker_factory has `output_is_deliverable: true` BUT no `final_deliverables/` was created

**Location**: `/Users/tchen7/MyProjects/CoreProjects/OpenStartup/src/openteam/server/resources/tools/task/topologies/breakdown-multiflow-plan.yaml`

**The fixer config** (in worker_factory, the MFDual's lightweight reviewer/fixer):
```yaml
worker_factory:
  __default__:
    _target_: MultiFlowDual
    # ...
    fixer_inferencer:                    # ← inner Dual's fixer
      _target_: ${_params.default_inferencer}
      output_is_deliverable: true        # ← SET on this leaf
```

**But empirically**: `worker_0/children/fixer_inferencer/outputs/final_deliverables/` is empty. So either:
- (a) `output_is_deliverable` is NOT being respected by the leaf inferencer (RovoDevCli) → bug in `_process_response`/`_promote_to_deliverables`, OR
- (b) The leaf wrote to `outputs/output.md` but the deliverable-promotion code path was skipped (e.g. file didn't exist when inspected, race condition), OR
- (c) The flag is set but RovoDevCli doesn't implement self-promotion (the inferencer subclass has to honor it)

**Need to verify**: which of (a)/(b)/(c) is the actual cause.

### Layer B — `_active_proposer()` returns `base_inferencer` (the inner BTA), not `fixer_inferencer`

**Location**: `dual_inferencer.py:564-606` (used by `_finalize_response` line 555-562)

`_active_proposer()` reads `state["attempt_record"]["iterations"][-1].counter_feedback`:
- `None` → returns `base_inferencer`
- non-`None` → returns `fixer_inferencer`

For an MFDual run with `consensus_max_iterations=1` (the shallow profile):
- The inner Dual ran 1 round: propose → review → fix
- After fix, the consensus exited with `counter_feedback=None` (the fix was accepted)
- So `_active_proposer()` returned `base_inferencer` (NOT the fixer)
- `surface_outputs_from(base_inferencer.workspace, ...)` ran — and the inner BTA's deliverable (61 KB) was potentially copied
- But the run output shows MFDual's `final_deliverables/` is EMPTY, so this didn't happen either

**Verify**: read state["attempt_record"] from `/Users/tchen7/MyProjects/CoreProjects/.../worker_0/checkpoints/...` to confirm what `_active_proposer()` actually returned.

### Layer C — MFDual itself doesn't have `output_is_deliverable: true`

**Location**: YAML worker_factory section

The MFDual orchestrator sets `output_is_deliverable: true` on `multi_flow_aggregator_inferencer` (its child) but not on itself. So when MFDual's `_ainfer` returns, the `_process_response` path does NOT promote MFDual's output to deliverables.

This is by design — MFDual is supposed to be a transparent pass-through, NOT a self-promoter. The pass-through is supposed to happen via `surface_outputs_from()` in `_finalize_response()`.

But `surface_outputs_from()` only fires if the active proposer's workspace has deliverables — and as shown in Layer A/B, neither has them populated correctly.

---

## §3 Pre-Implementation Verification (MUST DO FIRST)

Before writing any code, run these checks against the existing run to identify which layer is actually broken:

### Check 1: Is `output_is_deliverable` honored by the inferencer used in MFDual's fixer?

```bash
WS=/Users/tchen7/MyProjects/CoreProjects/OpenStartup/src/openteam/server/_runtime/tasks/task_task-cbaf8f2b_20260510_033521
# 1. Confirm yaml sets output_is_deliverable: true
grep -A 3 "fixer_inferencer:" /Users/tchen7/MyProjects/CoreProjects/OpenStartup/src/openteam/server/resources/tools/task/topologies/breakdown-multiflow-plan.yaml | head -20

# 2. Check that the leaf actually saw output_is_deliverable=true at runtime
grep -r "output_is_deliverable" $WS/children/base_inferencer/children/worker_0/children/fixer_inferencer/logs/ 2>/dev/null | head

# 3. Check if other Dual fixer leaves (top-level) ALSO have empty final_deliverables/
ls -la $WS/children/fixer_inferencer/outputs/final_deliverables/  # top-level fixer deliverables
```

If the top-level `children/fixer_inferencer/outputs/final_deliverables/` is also empty, then `output_is_deliverable` is broken globally and the fix is in `_process_response`. If only MFDual's inner fixer is broken, the fix is more localized.

### Check 2: What is the actual state at finalization?

```bash
# Look at the checkpoint state
find $WS/children/base_inferencer/children/worker_0/checkpoints -name "*.json" 2>/dev/null | xargs grep -l "attempt_record" | head -3
```

Read `attempt_record["iterations"]` for the inner Dual. Confirm `counter_feedback` value at the last iteration.

### Check 3: Does the outer-level Dual's surface_outputs_from work because the YAML sets it explicitly?

The TOP-level Dual also doesn't set `output_is_deliverable: true` on itself. Yet it produced a valid 92 KB final deliverable. Why does TOP work but MFDual doesn't?

**Hypothesis**: The top-level fixer is `${_params.default_inferencer}` with `output_is_deliverable: true` (line 163 of YAML). When it ran, RovoDevCli somehow DID populate `final_deliverables/`. But MFDual's inner fixer is the SAME inferencer kind with the SAME flag — yet inner fixer's `final_deliverables/` is empty.

Could be a **`_workspace.use_final_deliverables_folder`** misconfiguration on inner workspaces, OR the leaf's promotion logic uses a different criterion.

---

## §4 The Fix — Updated Options Post-Verification

### Option α (RECOMMENDED) — Set `output_is_deliverable=True` at the mutation site

In MFDual's `_select_reviewer_and_fixer` (or the `fixer_match_winner` mutation path), after assigning `self.fixer_inferencer = winning_flow_instance`, also set:

```python
# multi_flow_dual_inferencer.py — wherever fixer_match_winner mutates self.fixer_inferencer
self.fixer_inferencer = winning_flow_instance
self.fixer_inferencer.output_is_deliverable = True   # ← NEW: inherit fixer-role contract
# Optionally also: self.fixer_inferencer.output_manifest_index = True (if applicable)
```

**Symmetric for reviewer if reviewer should also produce deliverables** (typically not — reviewer outputs are JSON verdicts).

**Pros**:
- Fixes the root cause at the exact code site that introduces the bug
- One-line change (or two-line)
- Architecturally correct: "the fixer role implies fixer contract"
- Composes correctly with Phase B's `_reassign_role_workspace()` (workspace already isolated; this just adds the missing flag)

**Cons**:
- Mutates an attribute on a Python instance whose original value (from YAML) might be `False` — slightly surprising to users debugging
- Should be paired with a comment explaining the role-contract inheritance

### Option β (alternative) — Extend `_reassign_role_workspace()` to also set role-contract attributes

```python
# multi_flow_dual_inferencer.py — _reassign_role_workspace
def _reassign_role_workspace(self, inferencer, role_name: str) -> None:
    # ... existing code ...
    if role_name == "fixer_inferencer":
        inferencer.output_is_deliverable = True
```

**Pros**: Centralizes role-contract enforcement in one helper.
**Cons**: Slightly broader scope — touches the workspace-isolation helper.

### Option γ (band-aid, deferred) — Defensive override in MFDual `_finalize_response`

The original Option A from §4. Should be considered ONLY as a defense-in-depth measure if Option α/β can't be applied.

### Original (also valid as fallback) — Three Options

### Option A — MFDual explicitly self-promotes the inner Dual's winning content (RECOMMENDED)

Add an override of `_finalize_response` in `MultiFlowDualInferencer` that explicitly surfaces the FIRST available deliverable from a known list of candidate workspaces:
1. `children/fixer_inferencer/outputs/final_deliverables/` (inner Dual's fixer output)
2. `children/aggregator/outputs/final_deliverables/` (inner BTA's aggregator output)
3. `children/base_inferencer/outputs/final_deliverables/` (inner BTA root, which collects from aggregator)

```python
# multi_flow_dual_inferencer.py
def _finalize_response(self):
    # First: run inherited Dual finalization (handles last-round-artifact copy
    # and the conventional surface_outputs_from chain).
    super()._finalize_response()

    # If the parent did NOT populate our final_deliverables/, fall back to
    # surfacing from a known-good candidate inside our subtree. This is the
    # MFDual-specific pass-through that the generic Dual logic misses
    # because the active proposer (an inner Dual) didn't itself self-promote.
    if self._workspace is None or self._workspace.deliverables_dir is None:
        return
    if self._workspace.has_deliverables:
        return  # already populated

    candidates = []
    if self.fixer_inferencer is not None:
        ws = getattr(self.fixer_inferencer, "_workspace", None)
        if ws is not None:
            candidates.append(ws)
    if self.base_inferencer is not None:
        ws = getattr(self.base_inferencer, "_workspace", None)
        if ws is not None:
            candidates.append(ws)
    for ws in candidates:
        if ws.has_deliverables:
            self._workspace.surface_outputs_from(ws, skip_existing=True)
            break
```

**Pros**: Local change, no YAML change, defensive (only kicks in when conventional path didn't work).
**Cons**: Adds a special-case to MFDual; doesn't address the root cause if the root cause is in the leaf inferencer's promotion logic.

### Option B — Fix `_active_proposer()` to handle MFDual's inner Dual case

If the bug is `_active_proposer()` returning the wrong inferencer for MFDual's inner Dual layer, fix the state-reading logic. But this requires us to KNOW that's the bug — Pre-Implementation Verification §3 Check 2 will tell us.

**Pros**: Fixes the root cause if `_active_proposer` is the issue.
**Cons**: Risky — could change behavior for the top-level Dual too.

### Option C — Set `output_is_deliverable: true` on MFDual itself + change Dual to also self-write

Add `output_is_deliverable: true` to the MFDual config in YAML:
```yaml
worker_factory:
  __default__:
    _target_: MultiFlowDual
    output_is_deliverable: true   # ← NEW
    # ...
```

Then make MFDual's `_finalize_response` write its own combined output to `outputs/final_deliverables/output.md` (e.g. using the inner aggregator's content as the basis).

**Pros**: Architecturally consistent — every layer that should produce a deliverable explicitly says so.
**Cons**: Changes the conceptual model — MFDual becomes a "boundary" rather than a "transparent pass-through". May conflict with how parent BTA reads worker_results.

---

## §5 Recommended Approach

**1. Run Pre-Implementation Verification §3 checks first** to determine which layer is broken (A: leaf promotion / B: _active_proposer / both / something else).

**2. If Layer A is broken** (leaf doesn't self-promote despite `output_is_deliverable: true`):
   - Investigate `_process_response` and the promotion code path
   - Fix the global mechanism, NOT just for MFDual

**3. If Layer B is broken** (`_active_proposer` returns wrong inferencer):
   - Fix `_active_proposer` to correctly identify the active proposer for MFDual's inner Dual
   - Add tests that specifically exercise the MFDual path

**4. If both layers work but MFDual still doesn't self-promote**:
   - Apply Option A (explicit MFDual `_finalize_response` override)

---

## §6 Test Plan

1. **Unit test**: Mock an MFDual scenario where the inner fixer wrote `outputs/final_deliverables/output.md`. Verify the MFDual's `_finalize_response` correctly surfaces it to MFDual's own `final_deliverables/`.

2. **Unit test**: Mock an MFDual scenario where the inner aggregator wrote (no fix needed, base accepted). Verify the same.

3. **Integration test**: Run a small MFDual end-to-end with mock inferencer and verify the outer BTA receives the FULL inner content (not just a 2 KB summary).

4. **Regression test**: Verify the top-level Dual still produces correct final_deliverables after the change.

---

## §7 Open Questions

1. Is the actual root cause in Layer A (leaf inferencer doesn't self-promote) or Layer B (_active_proposer returns wrong child)?
2. Does the top-level Dual's success rely on some specific chain that MFDual doesn't have?
3. Should MFDual be a "boundary" (set `output_is_deliverable: true` on itself) or remain a "transparent pass-through"?
4. What happens for MFDual when ALL flows are equal and there's no clear winner — should the aggregator's output be the deliverable instead?

---

## §8 Acceptance Criteria

- [ ] Run `test_task_agent_config_brta_with_multiflow_pti.py` end-to-end and verify:
  - [ ] Each inner MFDual's `outputs/final_deliverables/output.md` exists and is the full content (not a 2 KB summary)
  - [ ] The outer BTA's worker_results aggregation receives the FULL inner content
  - [ ] The top-level final deliverable is preserved at >= the previous run's quality
- [ ] No regression in existing 214-test suite
- [ ] New unit tests for MFDual self-promotion pass

---

## §9 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Fix breaks top-level Dual self-promotion | Low | High | Option A is defensive (only kicks in when parent failed); regression test |
| Root cause is elsewhere (e.g. RovoDevCli's promotion logic) | Medium | Medium | Pre-implementation verification §3 catches this before code change |
| MFDual's promoted content disagrees with what outer BTA's text-based aggregation produced | Low | Low | Both should reach the same conclusions; if they diverge, the deliverable is more authoritative |

---

## §11 ADDITIONAL ISSUES — Found 2026-05-10 09:57

The same root cause (`output_is_deliverable` not set on dynamically-assigned role instances) manifests in **multiple places**. Each variant deserves its own micro-fix:

### Issue A — Inner Dual fixer (winner-as-fixer) doesn't self-promote
**Already covered above** (§2-§4). Root cause: `fixer_match_winner: true` mutation loses `output_is_deliverable`.

### Issue B — Inner Dual reviewer (loser-as-reviewer) reuses `flow_X_initial/` workspace ❌ Phase B GAP
**Evidence**: `round_01/review` symlink points to `flow_1_initial/` (the loser flow's race workspace). NO fresh `review_inferencer/` workspace exists.
**Impact**: Review-phase artifacts mix with propose-phase artifacts in `flow_1_initial/` (same hygiene problem Phase B was supposed to solve for fixer; was applied to fixer but skipped for reviewer).
**Fix**: Extend Phase B's `_reassign_role_workspace()` to ALSO be called for the reviewer (when `reviewer_match_second: true`).
**Effort**: ~30 min — symmetric to Phase B's fixer code path.

### Issue C — Followup-round inferencers don't self-promote
**Evidence**: `flow_0_followup/children/round_01/outputs/` has `output.md` (62 KB) but `final_deliverables/` is EMPTY. Same for flow_1.
**Root cause**: Same as Issue A — followup inferencer instances don't have `output_is_deliverable=True` (defaulted False).
**Impact**: LWI's last-round content cannot be surfaced upward as a deliverable; only the LLM TEXT response (smaller summary) is visible to parent.
**Fix**: When LWI launches a followup round inferencer, set `output_is_deliverable=True` on it (only for the last-round semantically; OR for all rounds since later rounds will overwrite).

**Open question**: Should EVERY followup round have `output_is_deliverable=True`, or only the LAST round? Argument for "every round": resume-safety (a round that doesn't know it's the last shouldn't omit promotion). Argument for "only last": avoid spurious manifests.
**Recommendation**: Every round (defensive) — the cost is just a marker file + manifest write per round.
**Effort**: ~30 min.

### Issue D — Two inference calls in a single followup round (VERIFIED 2026-05-10 10:11)

**Evidence — Both calls produced `"decision": "stop"`**:
| Call | Time | Output Size | Decision JSON |
|---|---|---|---|
| 1st | 04:00:22 | 45,251 bytes | `"decision": "stop", "reason": "...integration added significant value..."` |
| 2nd | 04:05:23 | 44,157 bytes | `"decision": "stop", "reason": "...complementary rather than redundant..."` |

So the 1st response said "stop", but LWI invoked a 2nd followup anyway.

**Verified root cause path traced through code**:
1. YAML config: `flow_configs.*.iteration_judgment: true` ✅ correctly set (line 125, 131)
2. MultiFlow's `iteration_judgment: true` should set `end_condition = parse_decision_stop` via `setdefault` (multi_flow_inferencer.py:388-390)
3. `parse_decision_stop()` correctly extracts the `iteration_judgment` JSON block and returns `True` for `decision: stop` (flow_parsers.py:83)

**So the wiring SHOULD work but DOESN'T. Hypothesis: one of**:
- (a) The `setdefault` in MultiFlow doesn't fire because `end_condition` was set elsewhere first to a no-op (e.g. by initial_inferencer's flow config)
- (b) `parse_decision_stop` IS called but `result` is not the LLM text response (it's `InferenceResponse` object whose `.output` field is empty)
- (c) `result` parameter is the LWI's internal step-result, not the raw LLM text — so `_extract_json_block` doesn't find the block

Most likely (b) or (c) — the parser was wired but receives wrong-typed input.

**Required next investigation**:
- Add print/logging to `parse_decision_stop` and re-run minimal MFDual scenario to see what `result` actually contains
- Check what `actual_result` is in LWI's loop (line ~ "5. Check termination") — is it a string, an InferenceResponse, or what?

**Fix candidates** (depend on investigation):
- (b/c) Fix `parse_decision_stop` to handle both string and InferenceResponse types defensively
- (a) Audit MultiFlow's `setdefault` to ensure it fires for both flow_configs entries

**Severity REVISED to HIGH**: this means EVERY MFDual run currently runs `max_dynamic_steps` followup rounds regardless of LLM's stop decision. Wasted LLM time = (max_steps - 1) × LLM call cost per flow per round.

### Issue E — Workspace layout: nested `round_NN/` vs flat `_roundNN` suffix (cosmetic but readable)
**Current**: `flow_0_followup/children/round_01/...` (1 round nested in `children/`)
**Proposed**: `flow_0_followup_round01/...` (sibling at parent level)
**Pros of flat**: linearly grep'able, no extra nesting layer, clearer "this is round N"
**Cons of flat**: breaks the convention that `children/` is where workspaces live; loses the natural "this round belongs to this followup" parent-child relationship
**Recommendation**: Keep current nested layout, BUT add a `manifest.json` in `flow_X_followup/` that lists round_NN dirs in order. Avoids breaking existing tooling while improving discoverability.
**Effort**: ~20 min (add manifest emission).

### Issue G — Aggregator silently compensates for broken upstream via independent re-investigation (HIDDEN dependency)

**Evidence**: From `aggregator/logs/session/.../InferenceInput/*.txt`:
- The aggregator's input is only **9 KB** containing 2 path references: `(See file: .../worker_0/outputs/output.md)` and `(See file: .../worker_1/outputs/output.md)`
- The actual worker output.md files at those paths are **2.2 KB and 2.7 KB** (LLM TEXT summaries, NOT the rich 47 KB+56 KB content that's in `worker_0/children/fixer_inferencer/outputs/output.md`)

**From `aggregator/logs/session/.../InferenceResponse/`**:
1. Agent tried `open_files(...)` with the worker output paths → REFUSED by RovoDevCli sandbox: *"Path is outside the workspace directory and whitelisted paths"*
2. Agent fell back to `bash cat ...` → succeeded, got 2.2+2.7 KB
3. Agent ALSO did its own end-to-end investigation: read `sop.jinja2`, conversation prompt templates, designed from scratch
4. Final aggregator output (79 KB) is mostly the agent's own work, only loosely informed by upstream

**Why this matters**:
- The current run "works" only because the aggregator implicitly compensates for the broken self-promotion by doing redundant investigation
- This redundancy DOUBLES the cost (~30 min of agentic work that should have been pre-computed by upstream workers)
- Fragile: if `bash cat` fallback failed (e.g., on a different system / path), aggregator would have NO upstream content to aggregate
- Misleading: the surface 79 KB output suggests "everything is fine" but the actual aggregation discipline (synthesizing N upstream artifacts) is NOT happening

**Fix**: Once Issue A + C are fixed, the worker output.md will be the FULL content (47 KB + 56 KB), `(See file:)` will reference rich content, and aggregator can do TRUE aggregation without redundant re-investigation.

**Secondary fix**: Add `(See file:)` path access to RovoDevCli's whitelist OR auto-include workspace-children paths. Today's bash fallback is a hidden brittleness.

**Effort**: Issue A + C fix is sufficient for the primary symptom; the whitelist fix is separate work.

### Issue H — Top-level outputs/ has no `output_manifest.json` (provenance hidden deeper)

**Evidence**: `task_task-cbaf8f2b_*/outputs/` contains:
- `output.md` (5.3 KB summary from Dual's text response)
- `final_deliverables/output.md` (92 KB — surfaced from fixer)
- `final_deliverables/.self_promoted` (marker)
- `round_log.jsonl`
- ❌ NO `output_manifest.json`

But `children/fixer_inferencer/outputs/output_manifest.json` exists (3.9 KB) — the provenance is buried one level deeper.

**Root cause**: The manifest is emitted by leaves at `outputs/output_manifest.json` (NOT inside `outputs/final_deliverables/`). When `surface_outputs_from()` walks `source.deliverables_dir`, it only copies files INSIDE that directory — not the manifest at the parent level.

**Code reference**:
- `inferencer_base.py:765-766`: `_emit_output_manifest(resolved)` writes to `outputs/output_manifest.json` (peer of `final_deliverables/`, not inside it)
- `inferencer_workspace.py:178`: `os.walk(src_root)` walks ONLY `deliverables_dir/` contents

**Impact**: For someone inspecting the run's surfaced output, the provenance ("which leaf produced this? how many LLM calls?") is invisible without walking the workspace tree.

**Fix options**:
1. **Option α**: Emit manifest INTO `deliverables_dir/output_manifest.json` (when `output_is_deliverable=True`). Then `surface_outputs_from()` will copy it automatically.
2. **Option β**: Have `surface_outputs_from()` ALSO copy `output_manifest.json` from source's `outputs/` (sibling of deliverables_dir) when present.
3. **Option γ**: Have orchestrator's `_finalize_response` emit its own top-level manifest summarizing the surfaced files + their source.

**Recommendation**: Option α is simplest and most direct — promote the manifest to "deliverable-grade metadata" by placing it inside the deliverable directory.

**Effort**: ~15 min (one-line change in `_emit_output_manifest`).

### Issue F — Dual-namespace pollution: BTA's worker_X/ alongside MFDual's flow_X_*/
**Current**: Inner BTA's `children/` has BOTH worker_0/, worker_1/ (BTA's exec slots, mostly empty husks) AND flow_0_initial/, flow_0_followup/, flow_1_initial/, flow_1_followup/ (MFDual's flow workspaces)
**Honest assessment**: Confusing but architecturally accurate (MFDual inherits BTA's worker_factory pattern but layers its own flow naming on top). Refactoring would require either:
- Removing BTA's worker_X/ exec slots when MFDual is the orchestrator (breaks BTA contract)
- Relocating flow_X_* under worker_X/ (changes long-standing layout)
**Recommendation**: Document the dual-namespace as intentional (MFDual = "BTA + flow naming layer") + add a `README.md` in `children/` explaining the convention. Refactoring is out of scope.
**Effort**: ~10 min (docstring + README).

---

## §12 Updated Implementation Order

Recommended order (do issues in dependency order):

1. **Issue A fix** (Option α — `output_is_deliverable=True` at fixer mutation site) — ~30 min
2. **Issue C fix** (followup-round self-promotion) — ~30 min
3. **Issue B fix** (Phase B for reviewer) — ~30 min
4. **Issue D investigation** (deferred until A/C done; impact unclear) — ~1 hour
5. **Issue E + F docs** (cosmetic) — ~30 min

**Total scope**: ~3 hours of code/test + 1 hour investigation.

---

## §13 Acceptance Criteria (UPDATED)

- [ ] After fix, `worker_0/children/fixer_inferencer/outputs/final_deliverables/output.md` exists and matches `outputs/output.md` content (Issue A)
- [ ] After fix, `worker_0/children/round_01/review` symlink points to a fresh `review_inferencer/` workspace, NOT to `flow_1_initial/` (Issue B)
- [ ] After fix, every `flow_X_followup/children/round_NN/outputs/final_deliverables/output.md` exists (Issue C)
- [ ] After fix, MFDual's `outputs/final_deliverables/output.md` is the full content (47+ KB), not the 2.2 KB summary (combined effect of A + C)
- [ ] After investigation, Issue D root cause is documented (and fixed if a real bug)
- [ ] Documentation for Issues E + F added
- [ ] No regression in existing 214-test suite

---

## §10 Provenance

- 2026-05-10 09:37 — Initial draft based on empirical evidence from run `task_task-cbaf8f2b_20260510_033521` and 3-agent investigation
- 2026-05-10 09:46 — v2: root cause REVISED post-verification — discovered the actual mechanism is `fixer_match_winner: true` mutation losing `output_is_deliverable` (NOT a workspace-flag issue). Added Options α/β as the proper fix.
- 2026-05-10 09:57 — v3: expanded scope — found 5 related issues (B-F) all variations of the same role-contract gap or workspace-hygiene gap. Updated implementation order and acceptance criteria.
- 2026-05-10 10:02 — v3.1: added Issue G — aggregator silently compensates for broken upstream via redundant re-investigation. Confirmed by reading aggregator's actual InferenceInput (9 KB) and InferenceResponse (sandbox-refused open_files → bash cat fallback → independent investigation). The current "successful" run hides the broken self-promotion behind expensive redundant work.
- 2026-05-10 10:05 — v3.2: added Issue H — top-level `outputs/` has no `output_manifest.json` because the manifest is emitted at `outputs/output_manifest.json` (sibling of `final_deliverables/`, not inside it). `surface_outputs_from()` walks only `deliverables_dir/` contents, so the manifest doesn't propagate up. Recommend Option α: emit manifest INSIDE `deliverables_dir/` when `output_is_deliverable=True`.
- 2026-05-10 10:11 — v3.3: VERIFIED Issue D root cause — BOTH calls' responses contain `"decision": "stop"` correctly. The wiring (`iteration_judgment: true` → `end_condition = parse_decision_stop`) exists but doesn't fire as expected. Most likely cause: `parse_decision_stop` receives an `InferenceResponse` object (not raw string) and `_extract_json_block` fails to find the JSON. Severity raised to HIGH — every MFDual run wastes max_steps-1 LLM calls per flow per round.
