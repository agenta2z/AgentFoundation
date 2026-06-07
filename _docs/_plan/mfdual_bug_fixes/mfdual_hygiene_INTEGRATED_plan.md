# MFDual Hygiene Fix Plan — INTEGRATED (A+B+C+D)

**Status**: Ready to implement. Combines Plan A's structure (phased rollout, open questions, provenance, empirical evidence) with Plan B (`splendid-lantern`)'s 4 critical contributions:

1. ✅ **Plan B's lock-step execution model** for Part C (replaces Plan A's hack-on-judgments approach; kills async-race + deadlock caveats by construction)
2. ✅ **Plan B's Fix 2** promoted to **Part D — Per-Round Followup Workspace** (a real bug Plan A missed)
3. ✅ **Plan B's own-flow path inclusion** in Part A (peer-visibility now covers self-reference too)
4. ✅ **Plan B's risk table format** added to Part B (debugability win)

This plan supersedes both `mfdual_peer_visibility_path_aware_fix_plan.md` (Plan A) and `_plan_splendid_lantern.md` (Plan B). Both source plans should be archived after this is approved.

---

## Plan Scope — FOUR Coordinated Fixes (All Ready)

| Part | Item | Root Cause | Severity | Effort |
|---|---|---|---|---|
| **A** | Peer-visibility path-aware (followup can't see peer flows' file artifacts) | `_format_followup_input` only embeds LLM summaries, never file paths | 🔴 High | ~3h |
| **B** | Workspace isolation (winner-as-fixer / loser-as-reviewer reuses race-phase workspace) | `_select_reviewer_and_fixer()` reassigns Python instance but skips workspace reassignment due to `inferencer_base.py:284` guard | 🟡 Medium | ~2.5h |
| **C** | Coordinated stop mode (lock-step execution; all flows must agree) | Today's `asyncio.gather()` race produces asymmetric peer-visibility; `_all_judgments` exists but isn't used for stop coordination | 🟡 Medium | ~3h (Plan B's lock-step is simpler than Plan A's barrier-sync) |
| **D** | Per-round followup workspace (LWI's dynamic mode reuses one dir for all rounds) | `flow_N_followup/` workspace shared across multiple followup steps; outputs overwrite each round | 🟡 Medium | ~2h |

**Combined effort**: ~10.5h active engineering for all four.

**Order of implementation** (dependencies):
1. Part A first (highest user-facing impact; standalone)
2. Part B second (independent of A; clean separation needed before D makes sense)
3. Part D third (per-round subdirs require B's clean workspace separation as substrate)
4. Part C last (integrates A + B + D; lock-step mode benefits from all the workspace clarity)

---

## Empirical Evidence — Run `task_task-0434bc68_20260509_165444`

These observations from a real production run motivate every fix in this plan:

| Observation | Severity | Maps to |
|---|---|---|
| `flow_0_followup` response says *"Only one upstream input (Flow 1 had no output)"* — but flow_1 actually produced 100KB output | 🔴 Critical | Part A |
| `round_01/fix → flow_0_initial/` symlink (winner reused, no separate `fixer_inferencer/` dir) | 🟡 Medium | Part B |
| `flow_0_followup/` had 2 InferenceInputs from 2 rounds — no per-round subdirs; `outputs/output.md` overwritten each round | 🟡 Medium | Part D |
| Both flows emitted `"decision": "stop"` independently with NO cross-flow consultation | 🟡 Medium | Part C |

These are not hypothetical concerns — every fix in this plan addresses a failure pattern observed in real-run artifacts.

---

# PART A — Peer-Visibility Path-Aware Fix

## §A1 The Bug

### Symptom

When MFDual's followup step runs, the prompt embeds peer flows' outputs as inline `<Response>` text — but the LLM has no way to read the FULL artifact (`output.md`) on disk. So followups see a summary/excerpt, not the actual document.

### Root Cause

`MultiFlowInferencer._format_followup_input()` collects peer flows' outputs from `_latest_per_flow` (a dict of latest-step results). These are LLM response strings, not file paths. The full file lives at `flow_N_initial/outputs/output.md` (often 40-100 KB) but is never surfaced to the followup prompt.

This is the SAME bug class as DualInferencer's empty `<ProposedDocument>` (fixed in Phase 0 by `_resolve_prior_proposer_output_path()`).

### Concrete Run Evidence

From `flow_0_followup` response file at line 1409 in run `task_task-0434bc68_20260509_165444`:

```json
{
  "decision": "stop",
  "reason": "Only one upstream input (Flow 1 had no output)..."
}
```

flow_0 stopped because it thought flow_1 had no output. But `flow_1_initial/outputs/output.md` was 100 KB on disk. The followup just couldn't see it.

---

## §A2 Proposed Fix — Path-Aware Peer Visibility (with own-flow path)

### Two-Tier Resolution Pattern (reused from Phase 0)

For each visible peer flow (and the flow's OWN previous step), resolve a path using:

1. **First tier**: `peer_ws.deliverable_path("output.md")` — preferred (the surfaced/promoted artifact)
2. **Second tier**: `peer_ws.output_path("output.md")` — fallback (raw output)
3. **None tier**: skip path injection (flow may not have produced an output yet)

### Template Block Pattern

In the followup prompt, after each `<Response>` block, inject:

```
The full peer artifact is available at:
  `/path/to/flow_N_initial/outputs/output.md`
```

For the flow's OWN previous step (a contribution from Plan B I missed initially):
```
Your previous output is available at:
  `/path/to/flow_X_initial/outputs/output.md`
```

This is critical: WITHOUT this, even if peers are path-aware, the flow can't easily re-read what it just produced (the prompt only contains a summary).

---

## §A3 Phased Plan

### Phase A1 — Helper Function (~30 min)

```python
# In multi_flow_inferencer.py
def _resolve_flow_output_path(flow_inferencer) -> Optional[str]:
    """Two-tier resolution: deliverable → output → None.
    Returns absolute path to output.md if it exists; None otherwise."""
    ws = getattr(flow_inferencer, "_workspace", None)
    if ws is None:
        return None
    # Tier 1: deliverable
    if hasattr(ws, "has_deliverables") and ws.has_deliverables:
        candidate = ws.deliverable_path("output.md")
        if candidate and os.path.isfile(candidate):
            return candidate
    # Tier 2: output
    out = ws.output_path("output.md")
    if os.path.isfile(out):
        return out
    return None
```

Tests:
- `test_resolve_flow_output_path_picks_deliverable_first`
- `test_resolve_flow_output_path_falls_back_to_output`
- `test_resolve_flow_output_path_returns_none_when_no_output`

### Phase A2 — Modify `_format_followup_input` (~1.5h)

Extend the function to accept and use a `visible_plan_paths: Dict[int, str]` parameter (peer flow_idx → absolute path). For each visible peer's `<Response>` block, append the path-aware footer.

**ALSO** include the own-flow path so the flow can re-read its previous output.

### Phase A3 — Wire `visible_plan_paths` Into the Caller (~30 min)

In the dynamic_input_builder wrapper, build the `visible_plan_paths` dict using `_resolve_flow_output_path()` on each visible flow's initial inferencer.

### Phase A4 — Documentation & Acceptance (~30 min)

- Add docstring example
- Add acceptance test: launch a 2-flow MFDual run; assert followup prompt for flow_0 contains literal substring `"flow_1_initial/outputs/output.md"`

---

## §A4 Risks (Plan B's table format)

| Risk | Mitigation |
|---|---|
| Peer flow not yet produced output → path doesn't exist | Two-tier resolver returns None; caller skips path injection (no block emitted) |
| Path is on shared filesystem but not readable by LLM tool | Verify in acceptance test that `output.md` exists at the resolved path |
| Backward compat: existing callers don't pass `visible_plan_paths` | Default param to `{}`; behavior preserved if dict is empty |
| Deliverable surfacing changes path between rounds | Re-resolve on every followup call (don't cache) |

---

## §A5 Open Questions — Part A

1. **Q**: Should we also include `output_path("aggregator_summary.md")` if it exists, alongside `output.md`?
   - **Recommendation**: NO — keep simple. Only `output.md` is the canonical artifact.
2. **Q**: Should the path-aware block be opt-in via flow_config flag?
   - **Recommendation**: NO — always emit when path resolves; backward compat is not at risk (just adds info).
3. **Q**: Should we surface ALL prior step outputs of the own flow (step 0, 1, 2, ...) or only the most recent?
   - **Recommendation**: Most recent only. Including all is noise; LLM can resolve via path if it wants history.
4. **Q**: Should we include byte size in the path block? E.g., `("100 KB")` ?
   - **Recommendation**: YES — gives LLM signal about depth. One extra line.

---

# PART B — Workspace Isolation Fix (winner-as-fixer / loser-as-reviewer)

## §B1 The Bug

### Symptom

When `fixer_match_winner=true`, `_select_reviewer_and_fixer()` reassigns `self.fixer_inferencer` to the winning flow's `initial_inferencer` Python instance. The instance keeps its original `_workspace` (`flow_0_initial/`). Fix-phase artifacts (session logs, cache, outputs) accumulate in the SAME directory as the propose phase.

### Why The Existing Workspace Propagation Doesn't Help

`inferencer_base.py:284`:
```python
def _propagate_workspace_to_children(self):
    for child in self._iter_child_inferencers():
        if getattr(child, "_workspace", None) is not None:
            return  # SKIP — child already has a workspace
        ...
```

This guard is correct for normal initialization but accidentally protects the stale (race-phase) workspace from being overwritten when role assignment happens.

### Concrete Run Evidence

| Symlink | Points to | Should point to |
|---|---|---|
| `round_01/propose` | `flow_0_initial/` (= base_inferencer) | ✅ correct |
| `round_01/fix` | `flow_0_initial/` (winner's race-phase dir) | ❌ should be `fixer_inferencer/` |
| `round_01/review` | `flow_1_initial/` (loser's race-phase dir) | ❌ should be `review_inferencer/` |

### Why This Is Not Just Cosmetic

1. **Debug clarity**: Mixed timestamps in `flow_0_initial/logs/session/` make it impossible to tell which inference belongs to propose vs fix
2. **Session continuity is wrong by design**: The role changed (race-flow → fix-surgeon); the conversation context from race-phase is irrelevant or harmful
3. **Output overwriting**: Both phases write to `flow_0_initial/outputs/output.md`; second one overwrites first
4. **Deliverable surfacing confusion**: `_finalize_response`'s `_active_proposer().has_deliverables` check sees the race-phase deliverable, not the fix-phase one

---

## §B2 Proposed Fix — Role-Based Workspace Reassignment

### Architectural Principle

> *"Role change = new session + new workspace. When an inferencer moves from 'propose-phase explorer' to 'fix-phase surgeon,' its prior conversation context becomes irrelevant or actively harmful (broad exploration vs. surgical edits). A fresh workspace and session ensure clean phase separation both logically (the LLM starts with clear fix instructions, no propose-phase baggage) and physically (artifacts are unambiguous)."*

### Two-Step Fix at Each Reassignment Site

After `_select_reviewer_and_fixer()` (or wherever the assignment happens), for each role (`fixer_inferencer`, `review_inferencer`):

1. **Assign role-named workspace** (overwrite `_workspace`, even if already set):
   ```python
   role_ws = self._workspace.child("fixer_inferencer")
   role_ws.ensure_dirs()
   self.fixer_inferencer._workspace = role_ws  # triggers _configure_for_workspace
   ```

2. **Reset session** so prior conversation context doesn't leak into new role:
   ```python
   if hasattr(self.fixer_inferencer, "reset_session"):
       self.fixer_inferencer.reset_session()
   ```

### What `_workspace = role_ws` Triggers (Detailed)

Via the property setter at `inferencer_base.py:246`:
- `_configure_for_workspace()` reconfigures `working_dir`, `cache_folder`, logger paths
- `ensure_dirs()` creates fresh `logs/`, `outputs/`, `artifacts/` directories
- Clean separation: race-phase artifacts stay in `flow_N_initial/`, role-phase artifacts go to `fixer_inferencer/` or `review_inferencer/`

### Result Workspace Layout (Post-Fix)

```
worker_0/children/
  base_inferencer/children/
    flow_0_initial/          ← propose phase ONLY (clean)
    flow_0_followup/         ← propose phase ONLY (covered by Part D for per-round)
    flow_1_initial/          ← propose phase ONLY (clean)
    flow_1_followup/         ← propose phase ONLY
  fixer_inferencer/          ← NEW: fix phase ONLY (clean)
  review_inferencer/         ← NEW: review phase ONLY (clean)
  round_01/
    propose → base_inferencer/
    review  → review_inferencer/    ← now points to dedicated dir
    fix     → fixer_inferencer/     ← now points to dedicated dir
```

---

## §B3 Phased Plan

### Phase B1 — Helper for Role Reassignment (~30 min)

```python
# In multi_flow_dual_inferencer.py
def _reassign_role_workspace(self, inferencer, role_name: str) -> None:
    """Force a fresh workspace + session because role has changed.

    The instance is the same Python object (e.g., flow_0's initial_inferencer
    repurposed as fixer_inferencer), but its conversational context, prompts,
    and outputs now belong to a different phase. Race-phase artifacts must
    stay in flow_N_initial/ — role-phase artifacts go to <role_name>/.

    Identity guard (from Plan B's Fix 1): skip reassignment when the inferencer
    is the role's ORIGINALLY-CONFIGURED instance (i.e., not a runtime swap).
    Prevents clobbering a legitimately-configured fixer_inferencer's workspace.

    role_name: 'fixer_inferencer' or 'review_inferencer'
    """
    if inferencer is None or self._workspace is None:
        return
    # Identity guard: don't reassign if the instance was already the configured one
    original = getattr(self, f"_{role_name}_original", None)
    if inferencer is original:
        return
    role_ws = self._workspace.child(role_name)
    role_ws.ensure_dirs()
    inferencer._workspace = role_ws  # triggers _configure_for_workspace
    if hasattr(inferencer, "reset_session"):
        inferencer.reset_session()
```

**Note**: The `_<role>_original` attributes (e.g., `_fixer_inferencer_original`, `_review_inferencer_original`) need to be captured at MFDual construction time (`__attrs_post_init__`) — see Open Question §B5.Q6.

### Phase B2 — Wire Reassignment Into MFDual (~1h)

Find every site where `self.fixer_inferencer` or `self.review_inferencer` is reassigned. Per code inspection: **3 sites in `multi_flow_dual_inferencer.py`**:
- Line 364: reviewer-as-fixer fallback
- Line 533: winner-as-fixer (the common case)
- Line 535: runner-up-as-reviewer

After each assignment, call `self._reassign_role_workspace(...)`.

### Phase B3 — End-to-End Verification (~30 min)

Add acceptance test that launches a 2-flow MFDual run and verifies:
- `worker_X/fixer_inferencer/outputs/output.md` exists
- `worker_X/review_inferencer/logs/session/` exists with one session log
- `worker_X/flow_0_initial/logs/session/` contains ONLY propose-phase timestamps (no fix-phase mix)
- `round_01/fix → fixer_inferencer/` (not `→ flow_X_initial/`)

### Phase B4 — Documentation (~30 min)

Update MFDual class docstring with the role-vs-flow workspace separation principle.

---

## §B4 Risks (Plan B's table format — adopted)

| Risk | Mitigation |
|---|---|
| Checkpoint resume references old workspace path | `reset_session()` already clears session state; checkpoint lives at Dual level, not leaf level |
| `_configure_for_workspace` side effects | This IS the standard mechanism — used at construction for every inferencer. Not a novel code path. |
| Deliverable surfacing (`_finalize_response`) | `_active_proposer()._workspace` now returns `fixer_inferencer/` workspace → `has_deliverables` check works correctly |
| Output path resolution | `resolve_output_path()` uses new workspace → fixer writes to `fixer_inferencer/outputs/output.md` → correct |
| Multiple Dual rounds reassigning the same instance | First round creates `fixer_inferencer/`. Round 2 also wants `fixer_inferencer/`. Need to either (a) sub-version `fixer_inferencer/round_02/` or (b) accept overwrite. **See Open Question §B5.Q1** |
| **Resumability after workspace reassignment** | Inferencer cache lookup is HASH-keyed (per `inferencer_base.py:843-861`), not path-keyed. So workspace mutations don't break cache hits — same input still finds same cached output. Mid-fix resume re-does fix from scratch (wasted work) but produces CORRECT output. NOT a regression vs today. |

---

## §B5 Open Questions — Part B

1. **Q (CRITICAL)**: How to handle multi-round Dual where the same instance is reassigned across rounds? Round 1 creates `fixer_inferencer/round_01/`; round 2 wants `fixer_inferencer/round_02/`?
   - **Recommendation**: For Phase B2, always overwrite (simplicity). For multi-round, use Part D's per-round subdirectory pattern (`fixer_inferencer/round_NN/`) — defer to Part D.
2. **Q**: Should we PREVENT the workspace reassignment if the role is unchanged across rounds (idempotent)?
   - **Recommendation**: YES — guard with `if inferencer._workspace.path != role_ws.path: ...`. Avoids repeated `reset_session()` clearing useful continuity.
3. **Q**: Should `reset_session()` be called even when the workspace is already set to the role-name?
   - **Recommendation**: NO — only on actual workspace change. Session continuity is valuable when role + workspace are identical.
4. **Q**: What if the role is `None` (review or fix not configured)?
   - **Recommendation**: Helper returns early on `inferencer is None`. Already handled.
5. **Q**: Race condition: another async task is reading from the old workspace while we reassign?
   - **Recommendation**: Reassignment happens synchronously in `_step_propose_impl` before next phase starts. No concurrent reader.

6. **Q (NEW from Plan B's Fix 1)**: Where do we capture `_fixer_inferencer_original` and `_review_inferencer_original` for the identity guard?
   - **Recommendation**: In MFDual's `__attrs_post_init__`, snapshot the configured `fixer_inferencer` / `review_inferencer` references BEFORE any runtime reassignment. These references are immutable for the lifetime of the MFDual instance.

---

# PART C — Coordinated Stop (Lock-Step Execution Model)

## §C1 The Bug Class (Quality, Not Correctness)

Today, each flow's `LinearWorkflowInferencer` (LWI) loop runs INDEPENDENTLY via `asyncio.gather()`:
```python
should_stop = end_condition(state, result) OR step_count >= max_dynamic_steps
if should_stop: return result
```
Each flow's `end_condition` only sees ITS OWN state. So flows can stop at different times, producing the asymmetric-quality issue documented in §A's empirical evidence.

### The Quality Concern

When flow_0 stops at step 1 but flow_1 keeps refining through step 5:
- flow_0 sat idle for 4 step-times → no opportunity to react to flow_1's late insights
- flow_1's late steps see flow_0's stale step-1 output → no fresh peer feedback either
- Aggregator picks from a pool of mixed maturity (1 round vs 5 rounds)

**Plan A's original Part C** proposed augmenting `end_condition` to check `_all_judgments` — but this approach has a critical async-race caveat (the judgment-check might happen before peers have judged) requiring barrier-sync as a separate mitigation.

**Plan B's approach is fundamentally more elegant**: instead of patching the per-flow loop, restructure execution itself to be lock-step. This makes the bug class IMPOSSIBLE by construction.

---

## §C2 Proposed Fix — Lock-Step Coordinated Mode (from Plan B)

### Architectural Shift

Add a new opt-in execution mode `coordinated_stop: bool = False`. When `True`, MultiFlowInferencer runs flows **lock-step**: one step per "round," gather all flows' results, vote unanimous before proceeding.

### Why Lock-Step Eliminates Plan A's Caveats

| Plan A Caveat | Plan B Resolution |
|---|---|
| C1: Async race (judgment-check might miss peer judgments) | ✅ Lock-step IS the barrier — flow N's judgment isn't checked until after `gather()` returns N results |
| C2: Deadlock risk (one stubborn flow blocks all) | ✅ Eliminated — each step is a sync gather; flows can't be "blocked waiting" because next step doesn't start until ALL finish current step |
| C3: Cost increase | ⚠️ Same in both approaches (early-stoppers do extra work). Mitigated by opt-in default. |
| C4: Quality tradeoff | Same — both modes have legitimate use cases |

### Code Sketch (from Plan B, refined)

```python
class MultiFlowInferencer(BreakdownThenAggregateInferencer):
    coordinated_stop: bool = attrib(default=False)

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        if not self.coordinated_stop:
            # Existing path: independent execution via BTA WorkGraph
            return await super()._ainfer(inference_input, inference_config, **kwargs)
        # NEW: lock-step coordinated execution
        return await self._ainfer_coordinated(inference_input, inference_config, **kwargs)

    async def _ainfer_coordinated(self, inference_input, inference_config, **kwargs):
        """Run all flows lock-step; stop only when ALL flows agree."""
        max_steps = max(cfg.get("max_dynamic_steps", 1) for cfg in self.flow_configs)
        n_flows = len(self.flow_configs)
        step_results = [None] * n_flows

        for step in range(max_steps):
            # Run ONE step for each flow in parallel (this IS the barrier)
            step_results = await asyncio.gather(*[
                self._run_flow_single_step(flow_idx, step, inference_input, prior=step_results[flow_idx])
                for flow_idx in range(n_flows)
            ])

            # Update visibility for next step's prompts
            for idx, result in enumerate(step_results):
                self._latest_per_flow[idx] = self._coerce_to_text(result)

            # Coordinated stop check (skip after step 0 — initial always runs)
            if step > 0:
                judgments = [self._evaluate_stop(idx, step_results[idx]) for idx in range(n_flows)]
                if all(judgments):  # unanimous stop
                    break

        # Proceed to aggregation with collected per-flow final results
        return await self._run_aggregator_with_flow_outputs(step_results)
```

### Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Default value | `coordinated_stop=False` | Backward compatible; independent mode stays |
| Granularity | Per-step sync, not post-hoc | Matches "all flows agree at decision point" intent |
| Voting rule | Unanimous (ALL must say "stop") | Conservative — no flow loses work prematurely |
| Step 0 exempted | Don't check stop after initial | Initial always runs once; coordination starts at followup |
| `max_steps` calc | `max()` across flow_configs | Allow flows with different caps; longest cap wins |

---

## §C3 Phased Plan — Part C

### Phase C1 — Add `coordinated_stop` Attribute (~15 min)

Add `attrib(default=False)` to `MultiFlowInferencer`. Update YAML schema docs.

### Phase C2 — Implement `_ainfer_coordinated` Path (~1.5h)

- Branch in `_ainfer` based on `coordinated_stop`
- Implement `_ainfer_coordinated` per the sketch above
- Implement `_run_flow_single_step(flow_idx, step, prior)` helper that runs ONE step of ONE flow
- Implement `_evaluate_stop(flow_idx, result)` — calls `parse_decision_stop(state, result)` per-flow
- Implement `_run_aggregator_with_flow_outputs(results)` — feeds collected results into the aggregator

### Phase C3 — Tests (~45 min)

- `test_coordinated_stop_unanimous_continues` — 2 flows both say "continue" → step 2 runs
- `test_coordinated_stop_split_decision` — flow_0 says "stop", flow_1 says "continue" → step 2 runs (both)
- `test_coordinated_stop_unanimous_stops` — 2 flows both say "stop" → loop breaks
- `test_coordinated_stop_max_step_cap` — neither says "stop" → loop hits max_dynamic_steps, exits
- `test_coordinated_stop_default_false` — backward compat: existing YAMLs continue using independent mode

### Phase C4 — Documentation (~30 min)

- Add `coordinated_stop` to MultiFlowInferencer docstring with tradeoff matrix
- Add example YAML showing both modes
- Add Compass docs entry

---

## §C4 When To Enable `coordinated_stop` In Production

| Use Case | Enable? |
|---|---|
| Quality > speed planning topologies | ✅ Yes |
| `max_dynamic_steps` ≥ 3 (where divergence likely) | ✅ Yes |
| Cost-sensitive runs | ❌ No (extra LLM calls for early-stoppers) |
| Exploration / non-critical tasks | ❌ No (independence is faster) |

**Default value is `False`** — `breakdown-multiflow-plan.yaml` does NOT switch to coordinated mode without explicit per-YAML opt-in.

---

## §C5 Risks (Plan B-style table)

| Risk | Mitigation |
|---|---|
| Performance regression in independent mode | Branching on `coordinated_stop` flag — independent code path unchanged |
| `_run_flow_single_step` doesn't exist as a helper today | New helper needed (~1h). LWI runs sequences via `expansion_step_registry`, not as discrete callables. **Recommended approach**: bypass LWI for coordinated mode — call the per-step inferencer directly: `cfg = self.flow_configs[flow_idx]; inferencer = cfg["initial_inferencer"] if step_idx == 0 else cfg["followup_inferencer"]; await inferencer.ainfer(step_input, ...)`. This is consistent with coordinated mode being a deliberate departure from BTA's WorkGraph orchestration. |
| Aggregator expects different input shape than `step_results` | `_run_aggregator_with_flow_outputs` adapts; existing `_run_aggregator` callable from inside |
| Multi-flow with different `max_dynamic_steps` per flow | Use `max()` across configs; flows that hit their own cap early just return their last result for subsequent gathers |
| Lock-step makes wall-clock = max(slowest flow per step) × N steps | Acceptable cost of coordination; documented tradeoff |

---

### PR Strategy Note (from Plan B)

**Scope warning**: Part C is a materially LARGER change than Parts A, B, D — it adds a new execution mode (lock-step vs. independent). To reduce blast radius:

- **Recommended PR sequence**: Parts A + B + D in PR #1; Part C in PR #2
- **Rationale**: A + B + D are clearly bug fixes. C adds new functionality. Keeping them separate makes review and rollback easier.
- **Optional**: Land C as opt-in (default False) so production runs are unaffected until per-YAML adoption

## §C6 Open Questions — Part C

1. **Q**: Should `coordinated_stop` be `bool` or an enum (`independent`, `coordinated_unanimous`, `coordinated_majority`)?
   - **Recommendation**: Start with `bool` (simpler). Enum can come later if `majority` use case arises.
2. **Q**: How should a flow that already stopped behave in subsequent step gathers?
   - **Recommendation**: Return its last result unchanged — `_run_flow_single_step` short-circuits if `step >= flow's_max_steps`.
3. **Q**: Does coordinated mode interact with Part D (per-round followup workspace)?
   - **Recommendation**: YES — round_NN naming applies in BOTH modes. Phase D2 handles this.
4. **Q**: Should there be a per-step timeout to prevent one slow flow from blocking forever?
   - **Recommendation**: Defer to v2; existing per-step LLM timeouts already provide bounded waits.
5. **Q**: Should `_evaluate_stop` defer to `flow_config[i]['end_condition']` if specified, or always use `parse_decision_stop`?
   - **Recommendation**: Defer to flow_config's `end_condition` if set; fall back to `parse_decision_stop`. Backward compat preserved.

---

# PART D — Per-Round Followup Workspace (NEW; from Plan B's Fix 2)

## §D1 The Bug

### Symptom

When LWI's dynamic mode runs followup multiple times, all rounds share the SAME `flow_N_followup/` workspace. Each round overwrites `outputs/output.md`. Session logs from rounds 1, 2, 3 are interleaved in `logs/session/`. Only timestamps distinguish them.

### Concrete Run Evidence

In run `task_task-0434bc68_20260509_165444`, `flow_0_followup/` had:
- 2 InferenceInputs (one per round) in the SAME `RovoDevCliInferencer-XXX.jsonl.parts/InferenceInput/` directory
- 2 InferenceResponses in the same parts directory
- ONE `outputs/output.md` (round 2 overwrote round 1)

### Why Plan A Missed This

Plan A focused on the Dual-level workspace reuse (winner-as-fixer). Plan B noticed the LWI-level reuse separately. Both are real bugs at different layers.

---

## §D2 Proposed Fix — Per-Round Subdirs in Followup Workspace

### Pattern (from Plan B)

In `linear_workflow_inferencer.py`, in the dynamic step wrapper, when running step N (N ≥ 1) of a followup_inferencer, assign a round-specific child workspace:

```python
# In _build_dynamic_step_wrapper, before calling followup_inferencer.ainfer():
#
# CRITICAL: stash the BASE workspace once and never mutate it.
# If we read followup_inferencer._workspace as parent each round, round 2 would
# read round_01/ (already mutated by round 1) and create round_01/children/round_02/
# (NESTED instead of SIBLING). This pattern mirrors PTI's _current_base_workspace
# (plan_then_implement_inferencer.py:419, 1515, 1562, 1802) which uses an init=False
# attribute to preserve the base across iterations.
if step_index > 0 and followup_inferencer is not None:
    ws = getattr(followup_inferencer, "_workspace", None)
    if ws is not None:
        # Stash base workspace ONCE — survives across rounds
        if not hasattr(followup_inferencer, "_base_followup_workspace"):
            followup_inferencer._base_followup_workspace = ws
        base_ws = followup_inferencer._base_followup_workspace
        # Always derive round subdir from BASE, not from current (possibly mutated) _workspace
        round_ws = base_ws.child(f"round_{step_index:02d}")
        round_ws.ensure_dirs()
        followup_inferencer._workspace = round_ws
        if hasattr(followup_inferencer, "reset_session"):
            followup_inferencer.reset_session()
```

**Why the stash matters** — naive trace without stash:
- Round 1: `parent_ws = flow_0_followup/` → creates `flow_0_followup/children/round_01/` ✅
- Round 2: `parent_ws = round_01/` (mutated!) → creates `round_01/children/round_02/` ❌ NESTED
- Round 3: `parent_ws = round_02/` → creates `round_02/children/round_03/` ❌ DEEPER

With the stash, every round reads the original `flow_0_followup/`, producing the correct flat:
- `flow_0_followup/children/round_01/`
- `flow_0_followup/children/round_02/`
- `flow_0_followup/children/round_03/`

**Credit**: This bug (and the stash pattern) was caught by an external review agent on 2026-05-09 22:23, who pointed to PTI's `_current_base_workspace` as the precedent.

### Result Workspace Layout

```
base_inferencer/children/
  flow_0_initial/                 ← step 0 (propose)
  flow_0_followup/                ← base followup ws (shared config)
    round_01/                     ← followup round 1
      logs/, outputs/, artifacts/
    round_02/                     ← followup round 2
      logs/, outputs/, artifacts/
  flow_1_initial/
  flow_1_followup/
    round_01/
    round_02/
```

### Synergy with Part B

The same `child(f"round_{NN:02d}")` pattern can be reused for multi-round Dual fixer/reviewer (Open Question §B5.Q1). One unified per-round-subdir convention across all multi-round contexts.

---

## §D3 Phased Plan — Part D

### Phase D1 — Identify Round Index in LWI Wrapper (~30 min)

Verify `step_index` is available in the dynamic step wrapper closure. If not, plumb it from the caller.

### Phase D2 — Add Per-Round Workspace Assignment (~45 min)

Implement the snippet above in `_build_dynamic_step_wrapper`. Gate by `step_index > 0` (round 0 = initial, no subdir needed).

### Phase D3 — Tests (~30 min)

- `test_per_round_workspace_created_for_each_followup_step` — assert `round_01/`, `round_02/` exist after a 2-step followup run
- `test_outputs_not_overwritten_across_rounds` — assert `round_01/outputs/output.md` and `round_02/outputs/output.md` both exist with different content
- `test_session_reset_between_rounds` — assert no conversation state leak from round 1 to round 2

### Phase D4 — Update Symlink Audit (~15 min)

If round audit symlinks (e.g., `worker_X/round_01/followup_step_01 → flow_0_followup/round_01/`) are useful for debug, add them in `_create_round_audit_links`.

---

## §D4 Risks

| Risk | Mitigation |
|---|---|
| `step_index` not available in wrapper closure | Plumb from caller (small refactor) |
| `reset_session()` breaks intended session continuity within a flow | This is INTENTIONAL — each round is logically a separate refinement; session reset matches the role-change principle |
| Workspace explosion (10 rounds × 4 flows × 2 workers = 80 dirs) | Acceptable; provides debug clarity. Cleanup is the workspace's job, not the runtime's. |
| Aggregator can't find "the final round's output" | Aggregator already uses `_latest_per_flow` (in-memory); workspace dirs are for audit, not aggregation |

---

## §D5 Open Questions — Part D

0. **Q (CRITICAL — RESOLVED 2026-05-09 22:23)**: Should `parent_ws` be the inferencer's CURRENT `_workspace` or a stashed BASE workspace?
   - **Answer**: STASHED BASE. Naive trace shows nested round dirs (`round_01/children/round_02/`) if we read current. Use `_base_followup_workspace` attribute stashed on first call. Mirrors PTI's `_current_base_workspace` pattern.

1. **Q**: Should round_00 (the initial) also be a subdir for symmetry?
   - **Recommendation**: NO — initial is in `flow_N_initial/`, separate inferencer instance. Round subdir naming starts from followup round 1.
2. **Q**: Should the round subdir use 0-indexed or 1-indexed numbering?
   - **Recommendation**: 1-indexed (`round_01`, `round_02`). Matches human intuition; matches Plan B's spec.
3. **Q**: What about runs where `iteration_judgment=False` and `max_dynamic_steps=1`?
   - **Recommendation**: Just `round_01/` exists. No extra cost.

---

## §6 Combined Sequencing & Schedule

### Recommended Order

| # | Part | Effort | Dependency |
|---|---|---|---|
| 1 | Part A | ~3h | Standalone |
| 2 | Part B | ~2.5h | Standalone (parallel-safe with A) |
| 3 | Part D | ~2h | Builds on Part B's clean workspace separation |
| 4 | Part C | ~3h | Builds on A + B + D for clean coordinated mode |

**Total**: ~10.5h active engineering. Recommended split: A + B in one session (~5.5h); D + C in a second session (~5h).

### Per-Part Acceptance Gates

After Part A: re-run shallow test → verify followup prompt contains literal peer-path strings
After Part B: re-run shallow test → verify `fixer_inferencer/` and `review_inferencer/` dirs exist with clean separation
After Part D: re-run shallow test → verify `flow_X_followup/round_01/`, `round_02/` dirs exist with non-overwritten outputs
After Part C: enable `coordinated_stop: true` in test YAML → verify lock-step execution log + unanimous stop

### Files to Modify (Quick Reference — adopted from Plan B)

| File | Parts | Changes |
|------|-------|---------|
| `multi_flow_dual_inferencer.py` | B | Add `_reassign_role_workspace()` helper; call after winner/loser assignment in `_step_propose_impl` (~line 612 area + lines 364, 533, 535); snapshot `_<role>_original` in `__attrs_post_init__` |
| `linear_workflow_inferencer.py` | D | Add per-round workspace assignment in `_build_dynamic_step_wrapper` with `_base_followup_workspace` stash (PTI pattern) |
| `multi_flow_inferencer.py` | A, C | Path resolution helper `_resolve_flow_output_path()`; extend `_format_followup_input` to inject peer/own paths; add `coordinated_stop` attribute + `_ainfer_coordinated()` lock-step path |
| `breakdown-multiflow-plan.yaml` | (verification only) | No YAML changes required; opt-in `coordinated_stop: true` only if user wants lock-step mode |

### Final QA Checklist (consolidated end-to-end verification)

1. ✅ Re-run shallow test on `test_task_agent_config_brta_with_multiflow_pti.py`
2. ✅ **Part A**: Followup prompt's `<Response>` blocks have `The full peer artifact is available at: <path>` footers; own-flow path included
3. ✅ **Part B**: `worker_X/fixer_inferencer/` and `worker_X/review_inferencer/` dirs exist; `flow_N_initial/logs/session/` contains ONLY propose-phase entries (no fix-phase mix)
4. ✅ **Part D**: `flow_X_followup/round_01/`, `round_02/` exist with separate `outputs/output.md` files (not overwritten); `flow_X_followup/round_01/children/round_02/` does NOT exist (no nesting)
5. ✅ **Part C** (if enabled): Session logs show lock-step execution; unanimous stop check fires after each followup step
6. ✅ Round audit symlinks: `round_NN/fix → fixer_inferencer/`, `round_NN/review → review_inferencer/` (NOT `→ flow_X_initial/`)
7. ✅ Deliverable surfacing: fixer's output at `fixer_inferencer/outputs/final_deliverables/output.md` AND surfaced upward to top-level Dual's deliverables
8. ✅ Resumability: hash-keyed cache lookup unaffected; mid-fix resume produces correct output

---

### Resumability Test Suite (NEW — Part E)

**Status**: Ready  
**Effort**: ~3h  
**File**: `test/agent_foundation/common/inferencers/test_dual_inferencer/test_mfdual_resume.py`  
**Mirrors**: `test_dual_inferencer_resume.py` (866 lines, 6 Tiers) for MFDual specifically

#### Resumability Verdicts (Verified Against Code)

| Part | Resumability Impact | Severity | Evidence |
|---|---|---|---|
| **A** (peer paths) | None (only changes prompt content) | ✅ Safe | No state mutation |
| **B** (workspace isolation) | None | ✅ Safe | `inferencer_base.py:843-861` cache is hash-keyed; pre-existing artifacts in `flow_N_initial/` remain accessible after resume |
| **D** (per-round subdirs) | None | ✅ Safe | `linear_workflow_inferencer.py:348-365` — LWI dynamic mode ALREADY has `resume_with_saved_results=False`; only `_load_final_result()` matters and it's workspace-root-relative |
| **C** (coordinated_stop) | **Wasted work on resume but CORRECT output** | 🟡 Documented behavior | Lock-step outer loop is custom Python (not BTA WorkGraph), so per-worker-completed checkpoints don't apply. BUT per-LLM-call cache (hash-keyed, `inferencer_base.py:843-861`) is independent of orchestration — already-completed inferences are cache-hits on resume. Outer loop re-evaluates lock-step decisions (wasted iterations) but produces correct final output. NOT a hard regression vs today's mid-fix-resume behavior class. |

#### Test Tier Structure (mirrors `test_dual_inferencer_resume.py`)

**Tier 1 — Backward Compatibility (6 tests)**
- `test_resume_existing_run_with_pre_part_b_artifacts` — Old workspace layout (mixed-content `flow_N_initial/`) still loads correctly
- `test_resume_after_complete_run_finds_final_result_json` — `_load_final_result()` works after Part B/D changes
- `test_part_b_does_not_break_bta_workgraph_resume` — BTA's `resume_with_saved_results` still finds `worker_X/` checkpoints
- `test_part_d_does_not_break_lwi_final_result_load` — LWI's `final_result.json` still loadable
- `test_no_op_when_resume_disabled` — Disabling resume produces same workspace as fresh run
- `test_legacy_yaml_without_part_c_runs_unchanged` — Default mode (independent) preserves all existing semantics

**Tier 2 — Checkpoint Normal Completion (5 tests)**
- `test_part_b_creates_fixer_workspace_with_checkpoint` — `fixer_inferencer/checkpoints/final_result.json` exists after fix completes
- `test_part_b_creates_review_workspace_with_checkpoint` — Same for `review_inferencer/`
- `test_part_d_each_round_has_separate_checkpoint` — `flow_N_followup/round_01/checkpoints/`, `round_02/checkpoints/` exist independently
- `test_identity_guard_prevents_clobbering_original_fixer` — When configured fixer == original, workspace not reassigned
- `test_old_workspace_artifacts_preserved_after_role_change` — After winner-as-fixer, `flow_N_initial/` artifacts remain readable

**Tier 3 — Resume from Crash (8 tests)** [Most critical]
- `test_resume_after_crash_in_propose_phase` — Crash mid-flow_0_initial → resume re-runs propose, finds fix-phase incomplete → completes
- `test_resume_after_crash_in_fix_phase_with_part_b_workspace` — Crash mid-fixer_inferencer/ → resume sees partial fix data → re-runs fix from scratch (correct, wasted work documented)
- `test_resume_after_crash_in_round_2_followup` — Crash in `round_02/` → resume re-runs `round_02/` (round_01 results preserved)
- `test_resume_after_crash_finds_correct_active_proposer` — Dual's `_active_proposer()` still works after partial-fix resume
- `test_resume_with_part_d_does_not_create_nested_round_dirs` — After crash + resume, NO `round_01/children/round_02/` (the nesting bug from Round 7)
- `test_session_id_cleared_on_resume_after_role_change` — `reset_session()` propagates correctly
- `test_resume_does_not_double_count_iterations` — `consensus_max_iterations` counter respects pre-crash state
- `test_part_c_coordinated_mode_resume_via_cache_hits` — Opt-in `coordinated_stop=true` resumes correctly via per-LLM-call cache hits (re-runs lock-step decisions but produces correct output)

**Tier 4 — State Restoration (4 tests)**
- `test_round_audit_symlinks_recreated_on_resume` — `round_NN/{propose,review,fix}` symlinks correct after resume
- `test_deliverable_surfacing_after_resume` — Fixer's deliverable surfaces correctly after partial-state resume
- `test_round_log_jsonl_consistent_after_resume` — `round_log.jsonl` doesn't have duplicate entries
- `test_manifest_consistent_after_resume` — `*_manifest.json` integrity preserved

**Tier 5 — Multi-Attempt (3 tests)**
- `test_multiple_resumes_converge_to_same_final_state` — N resumes from M crashes → same final output
- `test_resume_then_continue_to_round_2` — Crash in round_1, resume, complete round_2 successfully
- `test_resume_handles_partial_workspace_creation` — Crash mid-`ensure_dirs()` → resume completes safely

**Tier 6 — Edge Cases (4 tests)**
- `test_resume_with_only_one_winner_no_fixer_assignment` — When fix not needed, workspace unchanged
- `test_resume_with_changed_yaml_between_attempts` — YAML changes between crash and resume → graceful handling
- `test_resume_when_fixer_workspace_dir_already_exists_from_prior_run` — Reuse existing dir, don't crash
- `test_resume_when_part_d_round_dir_already_exists_with_partial_outputs` — Don't lose partial outputs

**Total: ~30 resumability tests** (mirrors test_dual_inferencer_resume.py's coverage depth)

---

## §7 Out-of-Scope

- **MFDual aggregator improvements** (separate concern; aggregator works correctly today)
- **Cross-worker (BTA) coordination** (this plan only fixes WITHIN one MFDual; cross-worker stays via BTA's existing dispatch)
- **Per-step LLM timeout** (Open Question §C6.Q4 deferred to v2)
- **Workspace cleanup / pruning** (out of runtime's responsibility)

---

## §8 Provenance — Integrated Plan

| Date | Event |
|---|---|
| 2026-05-08 | Plan A v1 drafted (peer-visibility + workspace isolation only) |
| 2026-05-09 21:33 | Plan A's Part C (consensus stop) added as deferred design |
| 2026-05-09 21:35 | User direction: "Part C cannot be deferred, fix together" → Part C promoted in Plan A |
| 2026-05-09 21:46 | User asked to compare Plan A with `_plan_splendid_lantern.md` (Plan B) |
| 2026-05-09 21:53 | Comparison completed; identified 4 critical Plan B contributions Plan A missed:<br>1. Plan B's lock-step model is more elegant than Plan A's barrier-sync approach for Part C<br>2. Plan B caught a missed bug (per-round followup workspace) — promoted to Part D<br>3. Plan B includes own-flow path in Part A — added<br>4. Plan B's risks-table format adopted across Parts B + C |
| 2026-05-09 21:55 | User direction: "Write the integrated final plan combining Plan A's spine + Plan B's 4 critical contributions" → THIS plan |

### Critical-Thinking Notes

| Plan B claim | Verdict | Evidence |
|---|---|---|
| "Lock-step kills async-race + deadlock by construction" | ✅ TRUE | `asyncio.gather()` IS a synchronization barrier; flows can't proceed until all peers finish current step |
| "Per-round subdir is needed; LWI overwrites outputs" | ✅ TRUE | Verified in run logs — `flow_0_followup/outputs/output.md` is single-file, mtime = round 2's mtime |
| "Own-flow path is necessary, not just peers" | ✅ TRUE | Without own path, flow can't read what it just wrote — only summary in prompt |
| "Plan A's barrier-sync approach is hacky" | ⚠️ PARTIALLY TRUE | Plan A's approach works but adds 1.5h for barrier; Plan B avoids both |

### Plan B claims I REJECTED

| Plan B claim | Verdict | Why |
|---|---|---|
| Plan B's "(~line 612)" line number for `_step_propose_impl` | ⚠️ NEEDS VERIFICATION | Should verify against actual code; Plan A says lines 364, 533, 535. Phase B2 enumerates all 3 sites. |
| Plan B's single-mention of `_step_propose_impl` | ❌ INCOMPLETE | There are 3 sites, not 1; Plan A's enumeration is more rigorous. Adopted Plan A's enumeration. |

---

## §9 Out-of-Scope (deferred)

- Coordinated stop with `majority` voting (only `unanimous` for v1)
- Per-step timeouts in coordinated mode
- Async barrier optimizations (none needed since lock-step IS the barrier)
- Multi-MFDual cross-coordination

---

---

# ~~PART E — RETRACTED~~ (2026-05-09 22:16)

**Status**: REMOVED FROM PLAN. Initial assessment was based on incomplete file system inspection. Re-verification reveals:

| Original Claim | Hard Truth |
|---|---|
| "Both workers' aggregator workspaces are empty" | ❌ Wrong — worker_1's aggregator IS populated (88 KB output) |
| "Manifest references missing files" | ❌ Wrong — files DO exist; I checked wrong location |
| "Cross-worker manifest pollution" | ❌ Wrong — worker_0 simply has no aggregator artifacts; manifest correctly attributes content origin to worker_1 |
| "BTA instances are shared across workers" | ❌ Wrong — verified `worker_factory` creates distinct instances per worker (`MultiFlowDualInferencer-35c5b6bc` vs `-19ee607f`) |

**Real question that remains**: WHY did worker_0's aggregator not run while worker_1's did? Two possibilities:
1. **Intentional**: Aggregator output is shared when both workers' sub-tasks produced equivalent outputs (winner-pick at outer BTA level)
2. **Bug**: worker_0's aggregator was silently skipped

This needs separate investigation, NOT a Part E in this plan. If it turns out to be a bug, file as a separate plan.

<!--
ORIGINAL PART E BODY REMOVED 2026-05-09 22:18 — accessible via git history.
The body was based on incomplete evidence and would have led to wasted refactor work.
Honesty record preserved in this stub above.
-->

---

---

<!-- ORPHANED PART E BODY (lines below) RETAINED IN GIT HISTORY ONLY — REMOVED FROM PLAN
     The Layer 1/Layer 2 analysis below was based on incomplete evidence (worker_1's
     aggregator workspace was actually populated; we just hadn't checked deep enough).
     Retracted 2026-05-09 22:16 with corrective notes inserted at §Part E RETRACTED stub above.
     Honest history preserved; not suitable for active plan content.
-->
<!-- BEGIN_RETRACTED_BODY

### Layer 1: Cross-Worker Manifest Pollution

**worker_0's `output_manifest.json`** at:
`worker_0/children/base_inferencer/outputs/final_deliverables/output_manifest.json`

Has `output.workspace_root` field pointing to:
`worker_1/.../aggregator` (the WRONG worker)

This means: worker_0's deliverable manifest claims its content was produced by worker_1's aggregator. Either (a) worker_0's manifest was mis-stamped at write time, or (b) worker_0 silently reused worker_1's aggregated output without attribution.

### Layer 2: Aggregator Artifacts Missing on Disk

The same manifest's `contributors` array references concrete files like:
`worker_1/.../aggregator/logs/session/RovoDevCliInferencer-6796bd3c.jsonl.parts/InferenceResponse/20260508_090006_output_dfc8a37a.txt` (88 KB)

But the actual `aggregator/outputs/output.md` (the surfaced deliverable) does NOT exist on disk in either worker_0 or worker_1's aggregator workspace. The 88 KB inference response files referenced in the manifest also don't exist (verified: `aggregator/logs/session/` is empty in both workers).

### Concrete Run Evidence

```
worker_0/aggregator/outputs/   →  EMPTY (0 files)
worker_1/aggregator/outputs/   →  EMPTY (0 files)
worker_0/manifest workspace_root → worker_1/.../aggregator  ❌ WRONG
worker_1/manifest workspace_root → worker_1/.../aggregator  (matches itself)
manifest contributors           → list 14 files in worker_1/aggregator/logs/  ❌ NONE EXIST
```

This is **systemic manifest corruption**, not debug-clarity.

### Why This Wasn't Caught Earlier

- The deliverable content (`output.md`) IS surfaced correctly to `base_inferencer/outputs/final_deliverables/output.md`
- The end user sees the right content
- The manifest is consulted for AUDIT only, not for serving deliverables
- So: silent corruption in the audit trail; no functional symptom

### Why This Matters Even Though "It Works"

1. **Audit integrity** — manifest is the source of truth for "where did this come from?"; if it lies, debugging is broken
2. **Deliverable provenance** — downstream consumers may rely on `workspace_root` for resolution; getting wrong path causes broken links
3. **Bug masking** — if the aggregator silently failed or got short-circuited (which seems to be happening here), the manifest's corruption hides this from observability

---

## §E2 Investigation Plan (BEFORE FIX)

This bug needs INVESTIGATION first because root cause is unclear. Possible mechanisms:

| Hypothesis | Evidence For | Evidence Against |
|---|---|---|
| H1: BTA's `_publish_aggregator_response` writes manifest BEFORE moving artifacts; artifacts get cleaned up | `.self_promoted` marker exists — suggests manifest write happened | If artifacts existed at write time, they should still be there |
| H2: Aggregator output is INHERITED from a sibling worker (not run independently) | worker_0's manifest points to worker_1 → suggests sharing | YAML doesn't define cross-worker sharing |
| H3: Both workers share the SAME aggregator inferencer instance, which was last-assigned to worker_1's workspace | Plausible Python instance reuse | Need to verify with code inspection |
| H4: Manifest paths are computed from a shared variable that was last updated by worker_1 | Strongly matches symptom | Most likely root cause |

**Most likely root cause**: `_publish_aggregator_response` writes the manifest using `self.aggregator_inferencer._workspace.path` — a Python attribute that gets reassigned between worker_0 and worker_1's processing, so by the time both manifests are written, BOTH point to the LAST workspace assigned (worker_1's).

Combined with Part B (workspace isolation): once Part B is implemented, the manifest will be stamped with role-named workspace, eliminating this cross-worker pollution as a side effect.

---

## §E3 Phased Plan — Part E

### Phase E1 — Verify Root Cause (~30 min)

Read `breakdown_then_aggregate_inferencer.py:_publish_aggregator_response` (around line 1042). Verify whether the manifest's `workspace_root` is computed from `self.aggregator_inferencer._workspace.path` (a Python attribute, mutable across workers) vs from a per-call argument (immutable).

Test by adding a print statement at manifest-write time logging `self.aggregator_inferencer._workspace.path` — confirm it's the same value across workers.

### Phase E2 — Capture the Aggregator Artifacts (~1h)

Investigate where the 88 KB aggregator response actually lives. Hypothesis: it was written to a TEMP location and cleaned up after surfacing. Find the actual disk path.

If artifacts are being deleted: stop deleting them OR copy to per-worker aggregator workspace before deletion.

### Phase E3 — Fix Manifest Attribution (~1h)

Two options:

**Option E3a (Quick Fix)**: Compute `workspace_root` from the WORKER's workspace, not the aggregator inferencer's `_workspace`:
```python
manifest["output"]["workspace_root"] = self._workspace.path  # the BTA's own ws
```

**Option E3b (Proper Fix, Synergy with Part B)**: Apply Part B's role-workspace pattern to the aggregator too — give each worker its own aggregator workspace via `_assign_role_workspace(self.aggregator_inferencer, "aggregator")`. Then `aggregator._workspace.path` is correct per-worker.

**Recommendation**: E3b. Same mechanism as Part B; restores per-worker artifact integrity at the same time as fixing manifest attribution.

### Phase E4 — Add Audit Tests (~30 min)

Acceptance tests:
- `test_aggregator_workspace_per_worker` — assert each worker's `aggregator/outputs/output.md` exists
- `test_manifest_workspace_root_matches_worker` — assert worker_N's manifest references worker_N's aggregator (not worker_M's)
- `test_manifest_contributors_paths_exist` — assert every file path listed in `contributors` actually exists on disk

---

## §E4 Risks

| Risk | Mitigation |
|---|---|
| Investigation may reveal even deeper bugs | Time-box Phase E1 to 30 min; if root cause isn't clear, escalate to a separate plan |
| Fix may regress aggregator behavior in other YAMLs (BTA without MFDual) | Test against `pti.yaml` and `bta-dual.yaml` after fix |
| Manifest schema change breaks downstream consumers | Manifest schema_version is "1.0"; bump to "1.1" if structure changes; otherwise just fix values |
| Synergy with Part B is required | Document explicit dependency: Phase E3b needs Part B's helper |

---

## §E5 Open Questions — Part E

1. **Q (CRITICAL)**: Are the 14 files referenced in `contributors` (totaling ~190 KB) actually missing, or were they moved elsewhere?
   - **Action**: Phase E1 must resolve this empirically before designing the fix.
2. **Q**: If artifacts ARE being deleted, by what mechanism?
   - **Hypothesis**: Workspace cleanup at MFDual finalization. Need to read finalize-deliverable code path.
3. **Q**: Does the same bug affect non-MFDual BTA aggregators (e.g., outer BTA)?
   - **Action**: Verify by checking outer `base_inferencer/outputs/output_manifest.json` workspace_root field.
4. **Q**: Should manifest validation be added as a runtime invariant (raise if path doesn't exist)?
   - **Recommendation**: NO — would be too aggressive. Add as audit warning instead (loud log).

---

## §E6 Provenance — Part E

| Date | Event |
|---|---|
| 2026-05-09 22:07 | User asked: "why is `aggregator/outputs/final_deliverables/` empty but `base_inferencer/outputs` has artifacts?" |
| 2026-05-09 22:09 | Initial assessment claimed "self-promotion = expected behavior" |
| 2026-05-09 22:09 | Hard evidence revealed: worker_0's manifest points to worker_1's aggregator workspace (cross-worker pollution); 88 KB contributors don't exist on disk |
| 2026-05-09 22:10 | **Honest correction**: This is NOT just debug-clarity, it's two layered bugs (E1: cross-worker manifest pollution, E2: missing aggregator artifacts) |
| 2026-05-09 22:10 | Added Part E to the integrated plan with INVESTIGATION-FIRST approach (root cause unclear) |

---

## §10 Combined Sequencing (Part E retracted — 4 parts only)

| # | Part | Effort | Dependency |
|---|---|---|---|
| 1 | Part A | ~3h | Standalone |
| 2 | Part B | ~2.5h | Standalone (parallel-safe with A) |
| 3 | Part D | ~2h | Builds on Part B |
| 4 | Part C | ~3h | Builds on A + B + D |

**Total**: ~10.5h (Part E retracted). Recommended split: A + B in one session (~5.5h); D + C in a second session (~5h).

---

END_RETRACTED_BODY -->

---

**END OF INTEGRATED PLAN**

Total: **4 fixes**, ~10.5h implementation + ~3h resumability tests = ~13.5h. Combines Plan A's structural rigor (phased plans, open questions, provenance, empirical evidence, 30 resumability tests across 6 tiers) with Plan B's elegance (lock-step over barrier-sync, identity guard, files-to-modify table, PR strategy) and missed-bug catches (Part D, own-flow path, nesting bug fix).

**Note**: Part E was added then retracted (2026-05-09 22:16) after re-verification revealed initial evidence was incomplete. See "Part E RETRACTED" section above for honesty record. Retracted body wrapped in HTML comment for cleanliness; full content available in git history.


