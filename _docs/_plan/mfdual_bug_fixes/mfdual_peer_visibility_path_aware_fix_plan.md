# MFDual Hygiene Fix Plan: Peer-Visibility Path-Aware + Workspace Isolation

**Created**: 2026-05-09
**Updated**: 2026-05-09 — added Part B (Workspace Isolation) after discovering the winner-as-fixer reuse pattern leaks artifacts across phases
**Status**: DRAFT — pending review
**Owner**: TBD
**Related**: `dual_inferencer_path_aware_followup_INTEGRATED_plan.md` (the analogous Phase-0 fix for DualInferencer)

---

## Plan Scope — THREE Coordinated Fixes (All Ready to Ship)

This plan covers **three related fixes** that together restore MFDual hygiene; user direction (2026-05-09 21:35) is to ship all three together rather than defer Part C:

| Part | Item | Root Cause | Severity | Status |
|---|---|---|---|---|
| **Part A** | Peer-visibility (followup can't see peer flows' real artifacts) | `_format_followup_input` only receives LLM summaries, not file paths | 🔴 High — direct prompt-quality impact | **Ready** |
| **Part B** | Workspace isolation (winner-as-fixer reuses race-phase workspace) | Inner Dual reuses Python instance + workspace from MFDual flow when `fixer_match_winner=true` / `reviewer_match_second=true` | 🟡 Medium — debug clarity, not correctness | **Ready** |
| **Part C** | Consensus stop mode (all flows must agree before stopping) | Today's per-flow independent stop creates asymmetric peer-visibility quality issue | 🟡 Medium — quality issue (now treated as bug class because the 3 fixes together restore the elegant MFDual semantic) | **Ready** |

**Combined effort**: ~10 hours active engineering for all three. Each part can technically ship independently; coordinated shipping is preferred per user direction.

---

# PART A — Peer-Visibility Path-Aware Fix

## §A1 Background — The Bug

### Symptom

In `MultiFlowDualInferencer`, when a flow enters its `followup` step, the prompt is supposed to include peer flows' outputs (via `visible_flows: all` in YAML) so each flow can synthesize/cross-pollinate from peers. **But peer outputs are NOT actually visible.**

### Hard Empirical Evidence (from run `task_task-0434bc68_20260509_165444`)

| Item | Size |
|---|---|
| `flow_0_initial/outputs/output.md` (peer's actual artifact) | 44,670 bytes |
| `flow_1_initial/outputs/output.md` (peer's actual artifact) | 44,831 bytes |
| **Total peer artifacts that should be visible** | **~89 KB** |
| `flow_0_followup`'s **entire** prompt (including instructions, peer summaries, own prior) | **16,093 bytes** |

The followup prompt is **smaller than ONE peer's output** — it cannot possibly contain both peers' full work.

**Direct verification**: Grepping a distinctive line from `flow_1_initial/output.md` in `flow_0_followup`'s prompt returns **zero matches**. The peer's real artifact content is mechanically absent from the prompt.

### Root Cause

In `multi_flow_inferencer.py` (`_format_followup_input` method, ~line 557-595):

```python
def _format_followup_input(
    self,
    *,
    your_prev: str,
    flow_idx: int,
    step_idx: int,
    visible_plans: Dict[int, Optional[str]],  # ← receives LWI step results
) -> str:
    peer_blocks = "\n\n".join(
        f"{_FOLLOWUP_PEER_BLOCK_HEADER.format(idx=idx)}\n"
        f"{plan or _FOLLOWUP_PEER_EMPTY_PLACEHOLDER}"
        for idx, plan in visible_plans.items()
    )
```

`visible_plans` is `Dict[int, str]` where the strings come from each LWI step's `result.response_str` — which is the LLM's SHORT `<Response>` tag content (a 28-31 line LLM-generated summary), NOT the actual artifact file content.

This is **structurally identical** to the original DualInferencer bug we fixed in Phase 0 (where `state["base_output_str"]` captured the LLM's text response, not the file deliverable).

---

## §2 Proposed Fix — Path-Aware Peer Visibility

Mirror Phase 0's pattern: instead of inlining peer text content (which is just an LLM summary), pass **file PATHS** to peer outputs and let the followup inferencer read them via file tools.

### Two-Tier Resolution (Same as Phase 0)

For each peer flow, the path is resolved to:
1. **First preference**: `<peer_flow_workspace>/outputs/final_deliverables/output.md` (if has_deliverables)
2. **Fallback**: `<peer_flow_workspace>/outputs/output.md` (if exists)
3. **Else**: keep the existing summary text behavior (backward compatible)

### Template Block Pattern

```
=== Flow {idx} ===
{% if peer_path %}
The full artifact from this peer is on disk at:
  `{{ peer_path }}`

To incorporate ideas from this peer, **read the file** to see the full content.
A short summary is below for orientation:

<Response>
{{ peer_summary }}
</Response>
{% else %}
{{ peer_summary or "(No peer output available)" }}
{% endif %}
```

This preserves backward compatibility (summary still shown) while making the FULL artifact discoverable.

---

## §3 Phased Plan

### Phase 1 — Helper to Resolve Peer Output Paths

**File**: `multi_flow_inferencer.py`

**Add method** (mirrors `DualInferencer._resolve_prior_proposer_output_path`):

```python
def _resolve_peer_output_path(self, peer_flow_workspace) -> Optional[str]:
    """Two-tier resolution for a peer flow's output path.

    Returns:
      str: Absolute path to peer's most-recent output artifact, or None if unavailable.
    """
    if peer_flow_workspace is None:
        return None
    # Tier 1: deliverables_dir (if non-empty)
    if (peer_flow_workspace.deliverables_dir is not None
            and peer_flow_workspace.has_deliverables):
        candidate = peer_flow_workspace.deliverables_dir / "output.md"
        if candidate.is_file():
            return str(candidate)
    # Tier 2: outputs/output.md
    candidate = peer_flow_workspace.outputs_dir / "output.md"
    if candidate.is_file():
        return str(candidate)
    return None
```

**Tests**: 4 unit tests covering both tiers + missing files + None workspace.

### Phase 2 — Modify `_format_followup_input` Signature & Behavior

**Change**: Accept an additional `visible_plan_paths: Dict[int, Optional[str]]` parameter that pairs with each `visible_plans` entry.

**Code**:
```python
def _format_followup_input(
    self,
    *,
    your_prev: str,
    flow_idx: int,
    step_idx: int,
    visible_plans: Dict[int, Optional[str]],
    visible_plan_paths: Optional[Dict[int, Optional[str]]] = None,
) -> str:
    visible_plan_paths = visible_plan_paths or {}
    peer_blocks = []
    for idx, plan in visible_plans.items():
        path = visible_plan_paths.get(idx)
        if path:
            block = _FOLLOWUP_PEER_BLOCK_PATH_AWARE.format(
                idx=idx, path=path, summary=plan or "(no summary)"
            )
        else:
            block = _FOLLOWUP_PEER_BLOCK_TEXT_ONLY.format(
                idx=idx, plan=plan or _FOLLOWUP_PEER_EMPTY_PLACEHOLDER
            )
        peer_blocks.append(block)
    return ...  # same assembly as before, just with new peer_blocks
```

**New constants**:
```python
_FOLLOWUP_PEER_BLOCK_PATH_AWARE = """=== Flow {idx} ===
The full artifact from this peer is on disk at:
  `{path}`

To incorporate ideas from this peer, you may read the file with file-read tools to see the full content.
A short summary is below for orientation:

<Response>
{summary}
</Response>
"""

_FOLLOWUP_PEER_BLOCK_TEXT_ONLY = """=== Flow {idx} ===
{plan}
"""
```

### Phase 3 — Wire `visible_plan_paths` Into the Caller

**File**: `multi_flow_inferencer.py` (the loop that calls `_format_followup_input`)

Find the call site of `_format_followup_input(...)` and inject the path map:

```python
# Build the path map for visible peers
visible_plan_paths = {
    idx: self._resolve_peer_output_path(self._flow_workspaces.get(idx))
    for idx in visible_plans.keys()
}

formatted = self._format_followup_input(
    your_prev=your_prev,
    flow_idx=flow_idx,
    step_idx=step_idx,
    visible_plans=visible_plans,
    visible_plan_paths=visible_plan_paths,  # ← new
)
```

**Tests**: 3 integration tests:
- Path-aware block fires when peer's output.md exists
- Falls back to text-only when peer's output.md missing
- Both peers visible simultaneously

### Phase 4 — Documentation & Acceptance

**Update**:
- Inline docstring on `_format_followup_input` to document the new parameter
- This plan's `§7` Open Questions section if any new edge cases emerge

**Acceptance Criteria** (verifiable post-implementation):
1. Running `breakdown-multiflow-plan.yaml` shallow produces a `flow_X_followup` prompt where peer paths appear (greppable)
2. Prompt size grows by ~2-300 bytes per peer (just the path block + short summary), NOT by peer's full content size
3. The followup LLM (with file-read capability like RovoDevCli) can demonstrate awareness of peer's full content (verifiable via inspection of fixer output)
4. No regression in existing `multi_flow_inferencer` tests

---

## §4 Sequencing & Schedule

| Phase | Effort | Risk |
|---|---|---|
| 1 (helper) | 30 min | 🟢 Low (additive, isolated) |
| 2 (template + signature) | 1 hour | 🟡 Med (signature change, but new param has default) |
| 3 (wire into caller) | 30 min | 🟡 Med (need to find correct workspace lookup) |
| 4 (docs + tests) | 1 hour | 🟢 Low |
| **Total** | **~3 hours** | |

---

## §5 Risk Assessment

### Mitigated
- ✅ Backward compat preserved: `visible_plan_paths` defaults to None → behaves like today
- ✅ Two-tier path resolution + None fallback: never crashes if peer's output is missing
- ✅ Summary still shown as orientation: LLM gets context even before reading the file
- ✅ Path-aware block is opt-in (only fires if path resolves)

### Residual
- ⚠️ Some leaf inferencers may not have file-read tools — they'll see the path but can't act on it. Acceptable: they still see the summary as today.
- ⚠️ If the path resolution depends on `_flow_workspaces` being a dict that doesn't exist in current code — Phase 3 will need to identify the correct attribute name.

### Out of Scope
- ❌ Same fix for OWN previous step's content (`your_prev`) — that's a separate bug if it has the same pattern; defer to follow-up.
- ❌ Path-aware peer visibility in DualInferencer — already done in Phase 0.
- ❌ Caching peer artifacts (e.g., copying to a shared location) — not needed; symlinks already exist via the round_NN/ structure.

---

## §6 Open Questions

1. **Q**: What's the exact attribute name for accessing peer flow workspaces from `MultiFlowDualInferencer`? (`_flow_workspaces`? `flow_inferencers[idx]._workspace`?)
   - **Action**: Phase 3 implementation will inspect the code to determine.

2. **Q**: Should we also include `prior_output_path` for the OWN previous step (so the LLM can read its own full prior work, not just summary)?
   - **Recommendation**: Yes, in a follow-up; this same bug almost certainly affects own previous step too. Defer to keep this plan focused.

3. **Q**: Does the LLM reliably call file-read tools when given a path? (Or does it just glance at the summary and move on?)
   - **Recommendation**: Same as Phase 0 — the prompt should make the path prominent and guide explicit reading. We learned in Phase 0 that strong directives ("MUST read the file before...") work much better than soft suggestions.

4. **Q**: Should this be folded into the broader leaf-owned-template-rendering refactor (`leaf_owned_template_rendering_INTEGRATED_plan.md`)?
   - **Recommendation**: NO. This is a tactical bug fix that should ship independently. The refactor is architectural and longer-horizon. Cross-reference but don't couple.

---

## §7 Provenance

- **2026-05-09 19:25** — User identified peer outputs missing from followup prompt
- **2026-05-09 19:40** — 3 parallel agents investigated the workspace; bug confirmed by hard empirical comparison
- **2026-05-09 19:42** — This plan drafted

**Verified by**:
- File size comparison: prompt (16 KB) vs peer artifacts (44+45 KB)
- Direct grep: distinctive peer line not found in prompt
- Code inspection: `_format_followup_input` only consumes `Dict[int, str]` summaries

---

# PART B — Workspace Isolation Fix (winner-as-fixer / loser-as-reviewer reuse)

## §B1 Background — The Bug

### Symptom

When MFDual is composed with `winner_pick=true + fixer_match_winner=true + reviewer_match_second=true`, the inner Dual's `fixer_inferencer` and `review_inferencer` are not constructed fresh — they are **reassigned to the existing Python instances** of the winning/runner-up race flows. Those instances retain their original `_workspace` attribute pointing at the race-phase workspace dir (e.g., `flow_0_initial/`). Result:

- ❌ The fixer phase writes to `flow_0_initial/logs/session/...` instead of `fixer_inferencer/logs/...`
- ❌ The same RovoDevCli `--resume` session continues from the propose-phase chat, mixing two unrelated phases of conversation
- ❌ Debugging "what did the fixer see/do?" requires sifting timestamp-mixed artifacts in a directory whose name says "initial"
- ❌ No `fixer_inferencer/` or `review_inferencer/` directory ever appears under the inner Dual workspace

### Hard Empirical Evidence (run `task_task-0434bc68_20260509_165444`)

`worker_0/children/round_01/` symlinks reveal the reuse:
```
fix             → flow_0_initial/    ← winner reused as fixer
review          → flow_1_initial/    ← runner-up reused as reviewer
review_dispatch → flow_1_initial/
propose         → base_inferencer/   ← MFDual itself
```

`flow_0_initial/logs/session/<sessionid>.jsonl.parts/InferenceInput/` shows 4 files:
```
20260509_170235_*.txt   (twice)   ← propose phase: race attempt 0 initial step
20260509_173957_*.txt              ← FIX phase: same instance reused as fixer (iteration 1)
20260509_174644_*.txt              ← FIX phase: same instance reused as fixer (iteration 2)
```

The same RovoDevCli session writes both phases. Phases are only distinguishable by clock skew across files.

### Root Cause

In `multi_flow_dual_inferencer.py` (verified):

| Line | Code | Effect |
|---|---|---|
| 533 | `self.fixer_inferencer = chosen` | Reuses winner's flow-instance as fixer |
| 535 | `self.fixer_inferencer = winner` | Same — alternative branch |
| 364 | `self.fixer_inferencer = self.review_inferencer` | Reuses reviewer-instance as fixer |

Then in `inferencer_base.py` line 284:
```python
if getattr(child, "_workspace", None) is not None:
    return  # SKIP propagation — child already has workspace
```

The reused instance's `_workspace` is **already set** (to the race-phase dir). The propagation skip means the inner Dual never assigns the proper `fixer_inferencer/` workspace path. The instance happily writes to its old workspace under its new role.

### Why This Is Not Just Cosmetic

| Concern | Impact |
|---|---|
| Debug clarity | High — humans can't easily tell which artifacts belong to which phase |
| LLM session contamination | Medium — RovoDevCli `--resume` continues the propose-phase chat into the fix-phase, potentially carrying stale context inappropriate for the new role |
| Round-symlink integrity | Medium — `round_NN/fix → flow_0_initial` is technically correct (that IS where the fixer wrote), but mixes propose+fix artifacts into the same target |
| Breaks the architectural invariant | High — workspace should reflect role, not instance identity (per @user 2026-05-09) |

---

## §B2 Proposed Fix — Role-Based Workspace Reassignment

### Architectural Principle (per @user)

> *"When the role changes, it should be a new workspace AND a new session by definition. The propose-phase context is about exploring; the fix-phase context is about narrow surgical edits. Continuity here is harmful, not helpful."*

### Two-Step Fix

**Step 1**: When `multi_flow_dual_inferencer` reassigns a flow instance to a new role (`fixer_inferencer` or `review_inferencer`), **explicitly reset** that instance's `_workspace` to None and force a new session ID.

**Step 2**: When `_propagate_to_children` runs at construction time, the now-`_workspace=None` instance will be properly assigned the role-appropriate workspace (`fixer_inferencer/` or `review_inferencer/` under the inner Dual's workspace).

### Code Sketch

```python
# multi_flow_dual_inferencer.py — at every reassignment site (lines 364, 533, 535)

# BEFORE (today):
self.fixer_inferencer = chosen

# AFTER (proposed):
self.fixer_inferencer = chosen
# Force new workspace + session because role has changed:
# the instance is the same Python object, but its conversational
# context, prompts, and outputs now belong to a different phase.
chosen._workspace = None  # next _propagate_to_children call will assign fixer_inferencer/
if hasattr(chosen, "_session_id"):
    chosen._session_id = None  # force RovoDevCli to start a new --session, not --resume
```

### Why This Works (Verified Against Existing Mechanism)

`inferencer_base.py:284-299` (existing logic, unchanged):
```python
if getattr(child, "_workspace", None) is not None:
    return  # skip — already set
# ELSE assign role-appropriate workspace:
child._workspace = self._workspace.child(role_name)
```

Setting `_workspace = None` makes the propagation guard fall through, and the proper `fixer_inferencer/` (or `review_inferencer/`) child workspace is assigned by the existing code path. **No changes needed to inferencer_base.py.**

---

## §B3 Phased Plan — Part B

### Phase B1 — Helper for Role Reassignment

**File**: `multi_flow_dual_inferencer.py`

**Add method**:
```python
def _reassign_for_new_role(self, instance, new_role_name: str) -> None:
    """Reset instance's workspace and session for a role change.

    When MFDual reuses a race-flow's Python instance for a new role
    (fixer or reviewer), the instance's prior workspace and session
    state are tied to its OLD role. We must reset both so:

    1. _propagate_to_children assigns a fresh role-appropriate workspace
    2. The leaf inferencer (e.g., RovoDevCli) starts a new conversation
       session, not --resume from the propose-phase chat.

    Args:
        instance: The inferencer to be reused under a new role.
        new_role_name: For logging/debugging only — does not assign workspace.
    """
    instance._workspace = None
    if hasattr(instance, "_session_id"):
        instance._session_id = None
    # Optional: archive prior workspace path for traceability:
    if hasattr(instance, "_prior_workspaces"):
        instance._prior_workspaces.append((new_role_name, "<prior_path>"))
```

**Tests** (in `test/agent_foundation/.../test_multi_flow_dual_inferencer/test_role_workspace_reassignment.py`):
- 3 unit tests covering: instance has `_workspace`, instance has no `_workspace`, instance has `_session_id`

### Phase B2 — Wire Reassignment Into MFDual

**File**: `multi_flow_dual_inferencer.py`

**Modify** lines 364, 533, 535 — each reassignment site:
```python
# Line 533 (winner-as-fixer):
self._reassign_for_new_role(chosen, new_role_name="fixer_inferencer")
self.fixer_inferencer = chosen

# Line 364 (reviewer-as-fixer fallback):
self._reassign_for_new_role(self.review_inferencer, new_role_name="fixer_inferencer")
self.fixer_inferencer = self.review_inferencer

# Similarly for runner-up-as-reviewer reassignment (find that site)
self._reassign_for_new_role(runner_up, new_role_name="review_inferencer")
self.review_inferencer = runner_up
```

**Tests** (3 integration tests):
- After construction with `fixer_match_winner=true`, fixer's workspace path ends in `fixer_inferencer`, not `flow_X_initial`
- Same for reviewer with `reviewer_match_second=true`
- Race-phase workspace (`flow_X_initial`) remains intact and contains ONLY race-phase artifacts

### Phase B3 — End-to-End Verification

**Acceptance criteria** (verifiable post-implementation by re-running the SOP plan test):

1. ✅ `worker_0/children/fixer_inferencer/` directory EXISTS (today: doesn't exist)
2. ✅ `worker_0/children/review_inferencer/` directory EXISTS (today: doesn't exist)
3. ✅ `flow_0_initial/logs/session/.../InferenceInput/` has only 1-2 files (race phase only — today: 4 files mixing race + fix)
4. ✅ `fixer_inferencer/logs/session/.../InferenceInput/` has the fix-phase prompts (today: nowhere)
5. ✅ `round_NN/fix` symlink points at `fixer_inferencer/`, not `flow_X_initial/` (today: points at race-phase dir)
6. ✅ Each session log corresponds to one phase only (no timestamp-spanning across phases)

### Phase B4 — Documentation

- Inline docstring on the new `_reassign_for_new_role` helper explaining the architectural principle
- Update MFDual class docstring with a note: *"When `fixer_match_winner` or `reviewer_match_second` is true, the reused instance is given a fresh workspace and session to reflect its new role."*
- This plan's `§B5 Open Questions` section (below)

---

## §B4 Sequencing & Schedule — Part B

| Phase | Effort | Risk |
|---|---|---|
| B1 (helper) | 30 min | 🟢 Low (additive) |
| B2 (wire reassignment) | 1 hour | 🟡 Med (touches MFDual core; must verify all 3 sites) |
| B3 (E2E test) | 30 min | 🟢 Low (re-run existing test + diff workspace structure) |
| B4 (docs) | 30 min | 🟢 Low |
| **Total Part B** | **~2.5 hours** | |

**Combined Part A + Part B**: ~5.5 hours active engineering.

---

## §B5 Risk Assessment — Part B

### Mitigated
- ✅ Backward compat: only affects MFDual when `fixer_match_winner` or `reviewer_match_second` is true (explicit opt-in via YAML)
- ✅ Existing `_propagate_to_children` mechanism unchanged — relies on the well-tested `_workspace=None → assign` path
- ✅ Race-phase artifacts remain intact (instance keeps writing to its NEW workspace; old workspace is frozen)
- ✅ No changes to inferencer_base.py — the fix lives in MFDual where the reuse pattern lives

### Residual
- ⚠️ Some leaf inferencers may not have `_session_id` — the helper handles this with `hasattr` check; safe.
- ⚠️ If `_propagate_to_children` is not re-invoked after the reassignment, the new workspace won't be assigned. Need to verify the construction order: does MFDual's reassignment happen BEFORE `_propagate_to_children`? **If not, we need to call `_propagate_to_children` explicitly after reassignment, or restructure the construction order.**
- ⚠️ The `_session_id` attribute is RovoDevCli-specific. A more architecturally clean approach would be a method on the leaf base class (`reset_session()`) that each leaf implements as appropriate. Worth considering for future generalization.

### Out of Scope
- ❌ Changing the `winner_pick + fixer_match_winner + reviewer_match_second` semantics themselves (that's a separate design discussion)
- ❌ Eliminating instance reuse entirely (would lose the "winner has best context for the fix" benefit at the conceptual level — though we're already discarding chat-history continuity, so this is debatable)
- ❌ Restructuring the BTA inheritance pattern in MFDual (separate refactor; out of scope)

---

## §B6 Open Questions — Part B

1. **Q**: Is `_propagate_to_children` re-invoked AFTER MFDual reassigns instances? If not, we need to call it explicitly post-reassignment.
   - **Action**: Phase B2 implementation must verify construction-order; may require a small `_propagate_to_children()` call after reassignment.

2. **Q**: Should the reset of `_session_id` be generic via a `reset_session()` method on the leaf base class, instead of hardcoded `_session_id = None`?
   - **Recommendation**: Yes, but as follow-up. Hardcoded `hasattr` check is fine for v1.

3. **Q**: Should we KEEP a back-pointer in the new workspace to the prior workspace (for traceability)?
   - **Recommendation**: Yes — write a small `prior_workspace.txt` file in the new workspace dir noting the original race-phase workspace path. Helps debugging.

4. **Q**: Should the `round_NN/fix` symlink point at `fixer_inferencer/` (which would now be the canonical fix-phase location) — does the existing symlink-creation logic auto-update?
   - **Action**: Phase B3 verification will confirm; expected yes since symlinks are based on `self.fixer_inferencer._workspace.path`.

5. **Q**: Are there OTHER places (besides MFDual) that do "instance reuse with role change" without resetting workspace?
   - **Recommendation**: Audit. Likely candidates: PTI's plan-then-implement reuse, any inferencer with `_match_X` flags. Defer audit to a follow-up.

---

## §B7 Provenance — Part B

- **2026-05-09 20:30** — User identified `worker_0`/`worker_1` (under inner MFDual base_inferencer) as architecturally muddled
- **2026-05-09 20:40** — Hard evidence from workspace symlinks (`round_01/fix → flow_0_initial`) confirmed winner-as-fixer reuse pattern is real
- **2026-05-09 20:53** — User pushed back on "session continuity" concern: *"role changes mean new sessions"*
- **2026-05-09 20:58** — User accepted the elegant fix: reset workspace + session on role change
- **2026-05-09 21:00** — Part B drafted in this plan

**Verified by**:
- Code inspection: 3 reassignment sites in `multi_flow_dual_inferencer.py` (lines 364, 533, 535)
- Workspace inspection: timestamps in `flow_0_initial/.../InferenceInput/` span propose + fix phases
- Symlink inspection: `round_01/fix → flow_0_initial` (not `→ fixer_inferencer/`)

---

# PART C — Consensus Stop Mode (READY — Coordinated With Parts A + B)

**Status**: Ready to implement (promoted from "deferred" per user direction 2026-05-09 21:35). The 4 caveats below are NOT reasons to skip — they are design constraints to respect during implementation. The "Recommendation" section is updated to reflect "ship now with mitigations" rather than "defer."

## §C1 Background — The Quality Tradeoff (Not a Bug)

### Today's Behavior (Verified)

In MFDual, each parallel flow has its own `LinearWorkflowInferencer` (LWI) loop with an **independent stop condition**:

```python
# linear_workflow_inferencer.py:544-549
should_stop = False
if self.end_condition is not None and self.end_condition(state, actual_result):
    should_stop = True
if state["dynamic_step_count"] >= self.max_dynamic_steps:
    should_stop = True
if should_stop:
    return actual_result
```

The `iteration_judgment: true` toggle wires `end_condition = parse_decision_stop`, which extracts the LLM's own JSON judgment and stops THIS flow when the LLM emits `"decision": "stop"`.

**There is no cross-flow coordination on the stop decision.** Flows stop independently, run in `asyncio.gather()`, and the aggregator sees each flow's final output (whatever it produced when it personally decided to stop).

### The Quality Concern (Not a Correctness Bug)

When flows stop at different times:
- **Early-stopper** sees only peers' early-state outputs at the time of its stop decision; misses late-stage peer refinements
- **Late-stopper** sees the early-stopper's stale step-N output throughout its remaining iterations; doesn't get fresh peer feedback either
- **Aggregator** picks a winner from a pool with mixed maturity: some flows finished after 1 round, others after 5 rounds

In `winner_pick + fixer_match_winner` workflows (the common case), the picked winner may not be the BEST IDEA — just the most-developed one. A flow that stopped early might have had a superior insight, just under-developed.

### Why This Is "Quality" Not "Correctness"

- ✅ The system produces valid output (no crash, no wrong-shape data)
- ✅ Reviewer/fixer downstream can correct quality issues
- ✅ With small `max_dynamic_steps` (2-3 in shallow profile), the divergence is bounded
- ⚠️ With large `max_dynamic_steps` (10 default), divergence could be large — this is where the issue would manifest most

---

## §C2 Proposed Design — Configurable Stop Modes

### Surprise: Half the Infrastructure Already Exists

Hard-evidence findings from code inspection:

| Mechanism | Where | Purpose |
|---|---|---|
| `_latest_per_flow: Dict[int, Any]` | `multi_flow_inferencer.py:330` | Each flow's latest step output, visible to all flows |
| `_all_judgments: List[Tuple[int, int, str]]` | `multi_flow_inferencer.py:331` | Every flow's judgment at every step (flow_idx, step_idx, judgment_text) |
| `all_judgments_summary` template variable | `multi_flow_inferencer.py:108-110` | Already rendered into followup template |

So the infrastructure to make a flow's stop decision aware of peers' judgments **is already present**. We just don't expose a stop-mode that uses it.

### Proposed Stop-Mode Enum

Add a new optional field to each `flow_configs[i]`:

```yaml
flow_configs:
  - max_dynamic_steps: 5
    iteration_judgment: true
    stop_mode: "individual"      # ← TODAY: each flow stops independently (DEFAULT for backward compat)
    # OR
    stop_mode: "consensus_all"   # ← NEW: stop only when ALL flows have said "stop" (with cap fallback)
    # OR (future)
    stop_mode: "consensus_majority"  # ← FUTURE: stop when MAJORITY of flows say "stop"
```

### Implementation Sketch (~30 LOC for `consensus_all`)

```python
def _make_consensus_stop_condition(outer, flow_idx):
    """Create an end_condition that stops THIS flow only when:
       1. THIS flow's LLM said "stop", AND
       2. All OTHER flows have most-recently said "stop" too.

    Falls through to max_dynamic_steps cap if consensus never reached.
    """
    def _consensus_stop(state, result):
        # Step 1: has THIS flow's LLM said stop?
        my_judgment = parse_decision_stop(state, result)
        if not my_judgment:
            return False  # this flow itself wants to continue

        # Step 2: check all peers' most-recent judgments
        n_flows = len(outer.flow_configs)
        latest_judgments_per_peer = {}
        for (fi, si, judg) in outer._all_judgments:
            if fi == flow_idx:
                continue
            if fi not in latest_judgments_per_peer or si > latest_judgments_per_peer[fi][0]:
                latest_judgments_per_peer[fi] = (si, judg)

        # Step 3: if any peer hasn't reported yet, keep going (peer might disagree)
        if len(latest_judgments_per_peer) < n_flows - 1:
            return False

        # Step 4: all peers have reported — do they all want to stop?
        all_stop = all("stop" in j.lower() for (_, j) in latest_judgments_per_peer.values())
        return all_stop  # only stop if EVERYONE wants to

    return _consensus_stop
```

Wired in `__attrs_post_init__` when `stop_mode == "consensus_all"`:
```python
if cfg.get("stop_mode") == "consensus_all":
    cfg.setdefault("end_condition", _make_consensus_stop_condition(self, flow_idx))
```

---

## §C3 The 4 Honest Caveats

These are the reasons Part C is **DEFERRED** rather than ready-to-implement.

### Caveat 1 — Asyncio Race Condition (CRITICAL)

`asyncio.gather()` runs flows concurrently. The order each flow reaches each step is **non-deterministic**. So:

| Time | flow_0 state | flow_1 state |
|---|---|---|
| t=0 | step 0 done → emits "stop" → checks `_all_judgments` | (still in step 0 LLM call) |
| t=1 | sees only its own judgment in `_all_judgments` → blocks | step 0 done → emits "stop" |
| t=2 | retried check → now sees both → stops | retried check → both stopped → stops |

**But what if flow_0 doesn't retry?** Today's `end_condition` is called ONCE per step. If it returns False, the flow proceeds to the next step. There's no "wait and recheck" mechanism.

**Fix needed**: Add a barrier-sync (asyncio.Event per step) that lets all flows reach the judgment-check point before any flow's stop decision is finalized. **This is a non-trivial code change** (1-1.5h additional).

### Caveat 2 — Deadlock Risk (NEEDS CAREFUL THINKING)

If consensus_all is misconfigured:
- Flow_0 emits "stop" at step 1 (truly done)
- Flow_1 NEVER emits "stop" (LLM keeps wanting more iteration)
- → Flow_0 sits idle waiting for consensus, hits `max_dynamic_steps` cap eventually, but consumes wall-clock time doing nothing

**Mitigation**: `max_dynamic_steps` cap is the ultimate stop (already exists). But the wasted cycles are real cost in $ and wall-clock.

### Caveat 3 — Cost Increase (REAL)

Today: total wall-clock = max(flow durations) where each flow can stop independently
Consensus: total wall-clock = max(flow durations) where each flow runs to consensus or cap

**For a 2-flow setup where flow_0 normally stops at step 1 and flow_1 normally stops at step 5**:
- Today: ~5 step-times wall-clock (flow_1 dominates)
- Consensus: ~5 step-times wall-clock (same — flow_1 still dominates)
- **But flow_0 now does 4 extra LLM calls** (step 2, 3, 4, 5) waiting for flow_1 to consent → 4 extra LLM costs

For 4-flow setup where 3 stop early and 1 keeps going: 3 × extra-steps = 9-12 extra LLM calls per cycle. **Significant cost increase in production.**

### Caveat 4 — The Quality Tradeoff Is Subtle (NOT UNIVERSALLY BETTER)

| Concern | Today (individual) | Consensus (all-agree) |
|---|---|---|
| Early-stopper misses late peer insights | ✅ Yes | ❌ Mitigated (early-stopper continues alongside late-stopper) |
| Late-stopper sees stale early-stopper output | ✅ Yes | ⚠️ Partially (early-stopper now produces fresh output too) |
| Wall-clock time | ✅ Faster (flows can exit early) | ❌ Slower (all flows wait for slowest) |
| LLM cost | ✅ Lower | ❌ Higher (early-stoppers do extra work) |
| Stop-decision quality | ⚠️ Each flow decides alone | ✅ Cross-validated by peers |
| Risk of forced over-iteration | ✅ Low | ❌ Higher (one stubborn flow drags everyone) |

**Neither is universally better.** Today's design favors speed and low cost; consensus favors stop-decision quality at a real cost.

---

## §C4 Estimated Effort (If Eventually Implemented)

| Component | Effort | Risk |
|---|---|---|
| Define `stop_mode` enum + YAML schema | 30 min | 🟢 Low |
| `_make_consensus_stop_condition` helper | 1 hour | 🟢 Low (additive) |
| **Async barrier-sync (asyncio.Event per step) — RESOLVES Caveat 1** | **1.5 hours** | 🔴 **High** (concurrency, possible deadlocks) |
| Tests (consensus, deadlock prevention, fallback to cap, mixed stop_modes) | 1 hour | 🟡 Med |
| Documentation + design rationale | 30 min | 🟢 Low |
| **Total Part C** | **~4.5 hours** | |

---

## §C5 Implementation Approach — Ship Now With Caveat Mitigations

Per user direction, Part C ships together with Parts A + B. The 4 caveats from §C3 are addressed via these explicit design choices:

| Caveat | Mitigation in This Plan |
|---|---|
| **C1 — Async race condition** | Implement an explicit `asyncio.Event` per step: each flow signals its judgment-emitted event after step N completes; consensus check awaits N events from all peers before deciding. Adds 1.5h to estimate but resolves the race correctness concern. |
| **C2 — Deadlock risk** | `max_dynamic_steps` cap remains the ultimate stop. Additionally, add a per-step `consensus_timeout` (default = 0 = wait indefinitely up to cap; configurable for time-bounded experiments). |
| **C3 — Cost increase** | Make `consensus_all` opt-in, NOT default. `breakdown-multiflow-plan.yaml` should NOT switch to consensus_all without separate evaluation; this plan adds the *capability*, not changes the default. Cost remains opt-in. |
| **C4 — Subtle quality tradeoff** | Document both modes' tradeoffs in inline docstrings + Compass docs so users can make informed choice. Pre-flight test asserts both modes produce valid runs. |

**Default `stop_mode` remains `"individual"`** — backward compatibility is preserved. Adopting `consensus_all` is a separate per-YAML decision.

**When to ENABLE consensus_all in a YAML:**
- Quality > speed (production planning topologies, not exploration)
- `max_dynamic_steps` ≥ 3 (where divergence is more likely)
- Acceptance of the cost increase (well-understood + budgeted)

---

## §C6 Open Questions — Part C (Tracked for Future Implementation)

1. **Q**: Should `consensus_majority` be added alongside `consensus_all`? Or just binary all-agree?
   - **Recommendation**: Just `consensus_all` first. `consensus_majority` adds complexity (what's "majority" with 2 flows?) — only add when there's a 3+ flow use case.

2. **Q**: Should we add a TIMEOUT for consensus to prevent unbounded waits?
   - **Recommendation**: `max_dynamic_steps` already serves as the ultimate timeout. No additional timeout needed for v1.

3. **Q**: Should the followup template show the LLM that "we're in consensus_all mode, your stop request will only be honored if all peers agree"?
   - **Recommendation**: YES — add a `stop_mode_hint` template variable so the LLM doesn't get frustrated thinking its stop request is being ignored.

4. **Q**: How should we handle a peer flow that's STILL RUNNING (no judgment emitted yet) when this flow's `end_condition` is called?
   - **Recommendation**: Treat absence of judgment as "wants to continue" → don't stop. (Conservative; matches Caveat 1's barrier semantics.)

5. **Q**: Should we expose stop_mode as a CLASS-level attribute (one mode for all flows) instead of per-flow_config?
   - **Recommendation**: Per-flow for flexibility — but allow class-level default that flow_configs can override. Pythonic pattern.

---

## §C7 Provenance — Part C

- **2026-05-09 21:17** — User asked: *"What happens when 2 flows decide to stop differently?"*
- **2026-05-09 21:26** — Verified today's behavior: per-flow independent stops via `asyncio.gather()`; identified asymmetric-quality concern
- **2026-05-09 21:27** — User asked: *"Can we add a mode that requires all flows to agree to stop?"*
- **2026-05-09 21:33** — Discovered `_latest_per_flow` and `_all_judgments` infrastructure already exists; identified 4 caveats; initially recommended defer
- **2026-05-09 21:34** — Part C drafted as deferred design proposal in this plan
- **2026-05-09 21:35** — User direction: *"Part C cannot be deferred, we need to fix together"* → Part C promoted to "Ready" with caveat mitigations baked into §C5
- **2026-05-09 21:36** — Plan updated: scope table, §C status header, §C5 Implementation Approach, this provenance entry

**Verified by**:
- Code inspection: `multi_flow_inferencer.py:330-331` (state collection), :108-110 (template visibility), `linear_workflow_inferencer.py:544-549` (per-flow stop)
- Architecture analysis: 4 caveats grounded in concurrency / cost / design tradeoffs
- Comparison to today's design: not universally better, real costs identified
