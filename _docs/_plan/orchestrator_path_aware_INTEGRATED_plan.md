# Orchestrator Path-Aware Outcome Passing — INTEGRATED Fix Plan

**Status**: Ready for implementation
**Severity**: Mixed — see §5 per-fix table
**Estimated effort**: ~6.5h total
**Author**: Synthesized from Plans A + B with hard-evidence corrections (Round 1)
**Date**: 2026-05-10
**Supersedes**: `lwi_reflective_path_aware_fix_plan.md` (Plan A) + `orchestrator_path_aware_outcome_passing_plan.md` (Plan B)

---

## 1. Why This Plan Exists

After landing the MFDual hygiene fixes (Parts A + B + C + D, 63 tests), a systematic audit of `agent_foundation/common/inferencers/` identified the **same Phase-0-style bug class** in additional orchestrators: downstream stages receive only LLM TEXT, not file PATH, causing **regeneration drift** when the previous stage produced a large file deliverable.

Two earlier plans (A and B) each captured part of the picture but had gaps:

| Plan | Strengths | Critical Gaps |
|---|---|---|
| **A** (LWI + Reflective focus) | Detailed Q&A, risks, provenance; technical claims all VERIFIED | Misses BTA, MultiFlow, PTI; `inspect.signature` re-runs every step (perf tax) |
| **B** (Orchestrator uniformity) | Broader systemic scope; clean template-driven design | Misses ReflectiveInferencer; stale line numbers; some "fixes" target already-correct code; no Q&A discipline |

This integrated plan **combines both** with hard-evidence corrections (e.g. BTA's text-formatter already includes `(Full output at: <path>)` so its bug is *exposing as separate feed key*, not *adding the path*).

---

## 2. The Core Principle (Architectural North Star)

> **Orchestrators MUST pass downstream nodes both (a) the LLM TEXT response AND (b) the FILE PATH to any deliverable**, so the downstream node's prompt template can reference the path for `cp`-style preservation (avoiding regeneration drift).

The path SHOULD be exposed as a **separate, structured `template_extra_feed` key** (matching DualInferencer's `prior_output_path` pattern). Path embedded in prose text is a fallback, not the primary mechanism.

### Verification Status (Hard Code Reads — Round 1)

| Orchestrator | Current State | Verdict |
|---|---|---|
| **`DualInferencer`** | ✅ Already correct — Phase 0 added `prior_output_path` as structured feed key | Reference implementation |
| **`MultiFlowDualInferencer`** | ✅ Already correct — Part A added peer + own-flow paths | Reference implementation |
| **`BreakdownThenAggregateInferencer`** (BTA) | ⚠️ Mixed — `_format_worker_results_text` includes `(Full output at: <path>)` in TEXT (line 612), and structured `deliverables_promoted` etc. injected into `template_extra_feed` (line 269), but **per-worker `worker_output_paths` list NOT exposed as a separate feed key** for LLM template use | Fix 1 |
| **`MultiFlowInferencer`** (aggregator) | 🔴 BUG — `worker_output_paths` computed at line 720 but NEVER injected into `template_extra_feed`; only TEXT (`upstream_artifacts`) at line 740 | Fix 2 |
| **`PlanThenImplementInferencer`** (PTI) | 🔴 BUG — `_build_executor_input` at lines 525-578 embeds `plan_file_path` ONLY in prose ("The full approved plan is at: `<path>`" at line 557); no structured `template_extra_feed["plan_output_path"]` | Fix 3 |
| **`LinearWorkflowInferencer`** (LWI) | 🔴 BUG — `dynamic_input_builder` at line 519 receives only TEXT; no path mechanism at all | Fix 4 |
| **`ReflectiveInferencer`** | 🔴 BUG — Three `_build_reflection_input_*` methods (lines 167-211) pass `state["base_response"]` (text only); no `base_response_path` in state | Fix 5 |
| `ConversationalInferencer` | ✅ Acceptable — by design, file paths are tool-layer concern | No fix needed |

---

## 3. Fixes — Detailed Plan

### Fix 1 — BTA Aggregator: Expose Per-Worker Paths as Structured Feed Key

**File**: `breakdown_then_aggregate_inferencer.py`

**Current state** (verified 2026-05-10):
- `_format_worker_results_text` (line 582-612): includes `(Full output at: <path>)` in formatted TEXT ✅
- `_inject_aggregator_extra_feed` (line 615-670): injects `deliverables_promoted`, `deliverables_with_conflicts`, `conflicts_grouped_by_parent`, `deliverables_dst`, `worker_summaries` into `template_extra_feed` (line 269)
- **Missing**: a clean per-worker `worker_output_paths` list mapped to worker indices, exposed as a top-level `template_extra_feed` key for templates to use directly

**Bug Pattern**: Templates that want to do `{% for path in worker_output_paths %}{{ path }}{% endfor %}` cannot — the path data is only inside formatted text and inside conflict-resolution structures.

**Fix Sketch**:
```python
# In _inject_aggregator_extra_feed (~line 269 area), add:
agg_inf.template_extra_feed.update({
    # ... existing keys ...
    "worker_output_paths": list(worker_output_paths or []),  # NEW: per-worker paths
})
```

**Severity**: 🟡 **Medium** — text format already includes paths; this is about *structured access* for cleaner templates.

**Effort**: ~45min (1 line of code + test)

### Fix 2 — MultiFlow Aggregator: Inject `worker_output_paths`

**File**: `multi_flow_inferencer.py`

**Current state** (verified 2026-05-10):
- Aggregator builder (`_make_default_aggregator_prompt_builder`, lines 681-748):
  - Line 720: `"worker_output_paths": list(worker_output_paths or [])` is computed in the local `feed` dict
  - Line 740: `target.template_extra_feed["upstream_artifacts"] = rendered` — only the rendered TEXT is exposed
  - **Missing**: `target.template_extra_feed["worker_output_paths"] = ...` injection alongside

**Bug Pattern**: Aggregator template author has no way to access per-flow paths. Same regeneration drift pattern as Phase 0.

**Fix Sketch**:
```python
# At line 740, after the upstream_artifacts injection:
target.template_extra_feed["upstream_artifacts"] = rendered
target.template_extra_feed["worker_output_paths"] = list(worker_output_paths or [])  # NEW
```

**Severity**: 🔴 **High** — aggregator is the consensus-resolution stage; drift here loses real work.

**Effort**: ~45min (1 line of code + test)

### Fix 3 — PTI Executor: Pass `plan_output_path` as Structured Feed

**File**: `plan_then_implement_inferencer.py`

**Current state** (verified 2026-05-10 — Plan B's stale line numbers corrected):
- `_build_executor_input` is at **lines 525-578** (NOT 553-561 as Plan B claimed)
- Line 553-557: `plan_file_path` embedded ONLY in prose text:
  ```python
  if plan_file_path:
      prompt_parts.append(
          f"The full approved plan is at: `{plan_file_path}`\n"
          f"Read that file to understand the complete plan...\n"
      )
  ```
- No `template_extra_feed` injection of the structured path

**Bug Pattern**: Executor LLM must parse prose to find path. If text format changes, parsing breaks. Implementation drift risk if executor regenerates portions of the plan rather than reading it.

**Fix Sketch**:
```python
# In _build_executor_input (~line 553), in addition to the existing prose:
if plan_file_path:
    # Existing prose remains as fallback for non-templated executors
    prompt_parts.append(
        f"The full approved plan is at: `{plan_file_path}`\n..."
    )
    # NEW: structured injection for templated executors
    if hasattr(self, "executor_inferencer") and self.executor_inferencer is not None:
        if hasattr(self.executor_inferencer, "template_extra_feed"):
            if self.executor_inferencer.template_extra_feed is None:
                self.executor_inferencer.template_extra_feed = {}
            self.executor_inferencer.template_extra_feed["plan_output_path"] = plan_file_path
```

**Severity**: 🟡 **Medium** — current prose works for most executors; structured access is for cleaner templates.

**Effort**: ~1h (code + verify executor template can use it + test)

### Fix 4 — LWI: Path-Aware via State Channel (No Signature Dispatch)

**File**: `linear_workflow_inferencer.py`

**Current state** (verified 2026-05-10):
- Line 168: `dynamic_input_builder: Optional[Callable[[dict, Any], Any]] = attrib(default=None)`
- Lines 517-520: `inp = self.dynamic_input_builder(state, prev)` — `prev` is text only
- Lines 559-562: `state["dynamic_step_results"].append(actual_result)` — text only
- **Missing**: parallel `dynamic_step_output_paths` list + path argument to builder

**Bug Pattern**: Custom callbacks cannot access prior step's file path. Same drift risk as Phase 0.

#### Self-Audit (2026-05-10): Why The State-Channel Approach (vs `inspect.signature` Dispatch)

Earlier drafts proposed an `inspect.signature`-based dispatch that detected whether the builder accepted a 3rd `prior_output_path` argument and called it accordingly. **This was rejected as ad-hoc** for these reasons:

| Reason | Detail |
|---|---|
| Inconsistent with the rest of the codebase | Every other orchestrator (BTA, MFI, PTI, Reflective, MFDual) uses `template_extra_feed` as the structured-data channel. Signature-based dispatch is unique to LWI in this plan. |
| API archaeology, not API design | Inspecting callable signatures to choose arity is a workaround, not a principled mechanism. |
| `state` already carries everything | The builder already receives the full `state` dict — adding `state["dynamic_step_output_paths"]` requires no signature change at all. |
| Edge cases (callable classes, `*args`, `partial`) need defensive try/except | Adds complexity for marginal gain. |

**The elegant fix**: Just store paths in `state["dynamic_step_output_paths"]` and let any builder read them. **No signature change required.** Backward-compat is automatic — existing 2-arg builders continue to work without modification, and they can opt in to path-aware behavior whenever they're updated.

#### The Fix

```python
# After existing line 561 (state["dynamic_step_results"].append(actual_result)):
if "dynamic_step_output_paths" not in state:
    state["dynamic_step_output_paths"] = []
prev_path = resolve_canonical_output_path(
    getattr(inf_instance, "_workspace", None),
    deliverables_fallback="first_match",   # LWI prior step typically a leaf or BTA-style result
)  # uses Phase 0a helper; returns absolute path or None (Tier 2 catches leaf CLI inferencers)
state["dynamic_step_output_paths"].append(prev_path)
```

**Step 4b — No dispatch change needed** (line 519 stays exactly as-is):
```python
if self.dynamic_input_builder is not None:
    inp = self.dynamic_input_builder(state, prev)  # SAME signature, no perf hit
```

The builder accesses paths via `state["dynamic_step_output_paths"][-1]` (latest) or `state["dynamic_step_output_paths"]` (full list). The `prev` argument remains the latest text result for backward compatibility (and is also reachable via `state["dynamic_step_results"][-1]`).

**Step 4c — Update docstring on `dynamic_input_builder` and `state` contract**:
```python
dynamic_input_builder: Optional[Callable[[dict, Any], Any]] = attrib(default=None)
"""Optional callable that builds the input for the next dynamic step.

Signature: ``(state, prev) -> input``  — UNCHANGED from prior versions.

The ``state`` dict carries (in addition to user keys):
  * ``dynamic_step_results``: list of prior step text outputs (str)
  * ``dynamic_step_output_paths``: list of prior step deliverable paths
    (str or None) — parallel to ``dynamic_step_results``. Use the latest
    path (``state["dynamic_step_output_paths"][-1]``) for ``cp``-style
    file preservation in path-aware refinement workflows.
  * ``dynamic_step_count``: int, len of ``dynamic_step_results``

The ``prev`` argument is the latest text result (= ``dynamic_step_results[-1]``),
preserved for builders that only need text.
"""
```

**Why this is elegant** (consistent with the rest of the codebase):
- **Zero API change** — backward-compat is automatic; no `inspect.signature` overhead at all
- **Builder authors opt in by reading state** — same pattern users already follow for `state["dynamic_step_count"]` etc.
- **No new dispatch path to maintain** — the codebase has ONE dispatch path through `dynamic_input_builder`

#### Alternative Considered: Callable Wrapper at Construction Time

Agent 1 (cross-review Round 2) suggested wrapping `dynamic_input_builder` once at construction with `lambda state, prev: user_builder(state, prev)`. This was considered and **rejected** for these reasons:

| Aspect | Wrapper at construction | State-channel (chosen) |
|---|---|---|
| Where the path goes | Wrapped into the call args | Already in state dict |
| Visibility to builder | Implicit (must inspect args) | Explicit (`state["dynamic_step_output_paths"]`) |
| Number of dispatch paths | 1 (wrapped) | 1 (unwrapped) |
| New mechanism to maintain | Wrapping logic at construction | None — uses existing state channel |
| Discoverability | Builder author must read code that wraps | Builder author sees state contract in docstring |

**State-channel wins on simplicity** — no new mechanism is introduced; the existing state dict gains one parallel list. Wrapping introduces a meta-layer that has to be remembered when debugging.

**Severity**: 🔴 **High** for affected users; limited blast radius (only topologies with custom `dynamic_input_builder` callbacks).

**Effort**: ~1h (was 1.5h with inspect dispatch — saved 30min by removing complexity)

### Fix 5 — ReflectiveInferencer: Path-Aware Reflection Input

**File**: `reflective_inferencer.py`

**Current state** (verified 2026-05-10):
- Lines 167-194: three `_build_reflection_input_*` methods all pass `state["base_response"]` or `reflections[-1].response` (text only)
- Line 270: `_process_reflection_input(self, inference_input, reflection_input, inference_config)` — no path parameter
- **Missing**: `state["base_response_path"]` and `state["reflection_paths"]` accumulation

**Bug Pattern**: Same Phase-0-style drift on file deliverables.

**Fix Sketch**:

**Step 5a — Surface path into state** (where `base_response` is set):
```python
state["base_response"] = base_result_text
state["base_response_path"] = self._resolve_inferencer_output_path(self.base_inferencer)  # see Phase 0a
```

**Step 5b — Update three builders**:
```python
def _build_reflection_input_sequential(self, state):
    reflections = state.get("all_reflections", [])
    if reflections:
        reflection_input = reflections[-1].response
        reflection_input_path = (state.get("reflection_paths") or [None])[-1]
    else:
        reflection_input = state["base_response"]
        reflection_input_path = state.get("base_response_path")
    return self._process_reflection_input(
        inference_input=state["original_input"],
        reflection_input=reflection_input,
        reflection_input_path=reflection_input_path,  # NEW
        inference_config=state.get("_inference_config", {}),
    )
# Apply same pattern to _separate and _integrate
```

**Step 5c — Update `_process_reflection_input` signature**:
```python
def _process_reflection_input(
    self,
    inference_input,
    reflection_input,
    inference_config,
    reflection_input_path=None,  # NEW, backward compat
):
    """If reflection_input_path is provided, surface it as a structured feed
    key on the reflection inferencer's template_extra_feed."""
    if reflection_input_path is not None and self.reflection_inferencer is not None:
        if hasattr(self.reflection_inferencer, "template_extra_feed"):
            if self.reflection_inferencer.template_extra_feed is None:
                self.reflection_inferencer.template_extra_feed = {}
            self.reflection_inferencer.template_extra_feed["reflection_input_path"] = reflection_input_path
    return self.reflection_prompt_formatter(
        feed={
            self.reflection_prompt_placeholder_inferencer_input: inference_input,
            self.reflection_prompt_placeholder_inferencer_response: reflection_input,
        },
        post_process=partial(unescape_xml, unescape_for_html=True),
        **inference_config,
    )
```

**Severity**: 🟡 **Medium** — narrow use case (ReflectiveInferencer reflecting on file deliverables is uncommon).

**Effort**: ~1h (code + tests)

---

## 4. Phase 0a — Shared Path-Resolution Helper (Promoted from Open Question)

**File**: `inferencer_workspace.py` (or new `inferencer_path_helpers.py`)

To avoid duplicating workspace→path logic across MFDual + LWI + Reflective + BTA + PTI:

```python
def resolve_canonical_output_path(
    workspace: Optional["InferencerWorkspace"],
    *,
    filename: str = "output.md",
    deliverables_fallback: str = "first_match",
) -> Optional[str]:
    """Returns the ABSOLUTE on-disk path to an inferencer's canonical
    output file, or None.

    Implements the **THREE-tier resolution** used by DualInferencer's
    ``_resolve_prior_proposer_output_path`` (the reference implementation),
    so it works for BOTH orchestrators (which have ``final_deliverables/``)
    AND leaf CLI inferencers (which write only to ``outputs/output.md``).

    **Critical**: Without Tier 2, this helper returns ``None`` for the most
    common production case (RovoDevCli, ClaudeCodeCli) which write to
    ``outputs/output.md`` but typically don't promote to
    ``final_deliverables/``. Tier 2 ensures leaf inferencers also resolve.

    Tiers (in order):
      Tier 1 — Deliverable file (preferred for orchestrators):
        If workspace.has_deliverables, try ``final_deliverables/<filename>``.
        On miss, apply ``deliverables_fallback`` policy.
      Tier 2 — Outputs file (canonical for leaf inferencers):
        Try ``outputs/<filename>`` directly. This catches CLI inferencers
        and any case where the deliverable hasn't been promoted yet.
      Tier 3 — None: no usable file exists.

    Parameters
    ----------
    workspace : InferencerWorkspace or None
        The inferencer's ``_workspace``. ``None`` returns ``None``.
    filename : str
        Preferred filename (default ``"output.md"``).
    deliverables_fallback : {"first_match", "alphabetical_scan", "none"}
        Behavior WITHIN Tier 1 when the preferred filename is missing:
          * ``"first_match"`` (DEFAULT, MFDual semantics):
            return ``deliverable_paths()[0]``
          * ``"alphabetical_scan"`` (DualInferencer semantics):
            return first non-dotfile deliverable in sorted order
            (filters ``.self_promoted`` etc.)
          * ``"none"``: skip Tier 1 fallback; proceed to Tier 2 immediately
        (Note: this controls fallback WITHIN Tier 1 only. Tier 2 always runs
        when Tier 1 produces no result.)

    Returns
    -------
    Optional[str]
        ABSOLUTE filesystem path (CWD-independent; safe for resume; safe
        for shell ``cp``), or ``None`` if no usable output file exists.

    Notes
    -----
    * **Returns absolute paths via ``os.path.abspath`` (NOT ``os.path.realpath``)**:
      Symlinks are PRESERVED, not resolved. This matches downstream usage —
      templates do ``cp '{{ prior_output_path }}' '{{ output_path }}'`` and
      callers expect to operate on the alias the orchestrator captured, not
      its symlink target. Use ``os.path.realpath()`` only if you need the
      true on-disk identity (rare; not done here by design).
    * Returns ``None`` (not ``""``) so callers can branch cleanly. All
      orchestrators in this plan inject ``None`` (not empty string) into
      ``template_extra_feed`` for "no deliverable" — sentinel consistency.
    * **TOCTOU caveat**: ``os.path.isfile`` checks happen at call time. A
      file deleted between this call and consumption returns a stale path
      reference. Long-running consumers should re-validate before use.
      (Out of scope for this helper; Plan §8 acknowledges as caller
      responsibility.)
    * **Filename contract**: Caller MUST pass a non-empty, non-absolute
      filename (e.g. ``"output.md"``, NOT ``""``, NOT ``"/abs/path"``).
      No validation here; passing invalid input may yield surprising
      ``os.path.join`` semantics (e.g. ``""`` returns the directory itself).
    * Does NOT escape for shell — templates MUST single-quote: ``'{{ p }}'``
    * Never raises — all exceptions caught and treated as "not found"
    """
    if workspace is None:
        return None

    # === Tier 1: deliverable file ===
    if getattr(workspace, "has_deliverables", False):
        deliverables_dir = getattr(workspace, "deliverables_dir", None)
        if deliverables_dir:
            candidate = os.path.join(str(deliverables_dir), filename)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

            if deliverables_fallback != "none":
                try:
                    deliverable_paths_fn = getattr(workspace, "deliverable_paths", None)
                    deliverable_paths = (
                        deliverable_paths_fn() if callable(deliverable_paths_fn) else []
                    )
                except Exception:
                    deliverable_paths = []

                if deliverable_paths:
                    if deliverables_fallback == "alphabetical_scan":
                        non_dotfiles = sorted(
                            p for p in deliverable_paths
                            if not os.path.basename(p).startswith(".")
                        )
                        if non_dotfiles:
                            chosen = non_dotfiles[0]
                        else:
                            chosen = None
                    else:  # "first_match"
                        chosen = deliverable_paths[0]

                    if chosen:
                        # deliverable_paths() may return basenames or full paths
                        if not os.path.isabs(chosen):
                            try:
                                chosen = workspace.deliverable_path(chosen)
                            except Exception:
                                chosen = None
                        if chosen and os.path.isfile(chosen):
                            return os.path.abspath(chosen)

    # === Tier 2: outputs/<filename> (CRITICAL for leaf CLI inferencers) ===
    try:
        out_path = (
            workspace.output_path(filename)
            if hasattr(workspace, "output_path")
            else None
        )
    except Exception:
        out_path = None
    if out_path and os.path.isfile(out_path):
        return os.path.abspath(out_path)

    # === Tier 3: nothing on disk ===
    return None
```

**Function renamed**: `resolve_canonical_deliverable_path` → `resolve_canonical_output_path` to accurately reflect that it covers BOTH `final_deliverables/` AND `outputs/`. The old name was misleading — it suggested deliverables-only.

#### Per-Caller Strategy Mapping (resolves Agent 2 "behavioral divergence" finding)

| Caller | Strategy | Rationale |
|---|---|---|
| `MFDual._resolve_flow_output_path` (Fix 6 retrofit) | `"first_match"` | Preserves existing MFDual behavior exactly |
| `DualInferencer._resolve_prior_proposer_output_path` (Phase 0a-extra retrofit) | `"alphabetical_scan"` | Preserves existing DualInferencer dotfile-aware behavior exactly |
| BTA / MFI / PTI / Reflective (new callers, Fixes 1-5) | `"first_match"` | Matches the more common case |

**This explicit strategy parameter makes the retrofit truly behavior-preserving** — neither MFDual nor DualInferencer changes behavior because each passes the strategy that matches its current logic.

Then **retrofit** MFDual's `_resolve_flow_output_path` (line 575) and DualInferencer's `_resolve_prior_proposer_output_path` (line 607) to use this helper. **Phase 0a runs FIRST** before any of Fix 1-5.

---

## 5. PR-Split Strategy

To enable independent review and clean blame analysis if regressions appear, this plan ships as **TWO separate PRs**:

### PR-1 — "Path resolution shared helper" (PURE ADDITION — ZERO retrofit)

| # | Fix | Severity | Effort | Risk |
|---|---|---|---|---|
| 0a | Add `resolve_canonical_output_path()` (3-tier) to `inferencer_workspace.py` | — | ~30min | **Zero** (new function, NO callers yet) |
| 0a-tests | 10 unit tests covering: (1) None workspace, (2) no-deliverables AND no outputs/output.md, (3) Tier 1 deliverable exists, (4) Tier 1 custom filename, (5) Tier 1 "first_match" fallback, (6) Tier 1 "alphabetical_scan" fallback (incl. dotfile filter), (7) Tier 1 "none" fallback skips to Tier 2, (8) **Tier 2 ONLY: workspace has no deliverables but outputs/output.md exists (CRITICAL leaf-CLI scenario)**, (9) **Tier 1→Tier 2 cascade: deliverables_dir empty after has_deliverables=True, falls through to Tier 2**, (10) absolute-path guarantee | — | ~40min | Zero |
| **PR-1 total** | | | **~1h** | |

**Acceptance**: ALL existing tests pass without modification. Zero behavior change anywhere. The helper exists but is unused.

**Critical change from earlier draft (Agent 2 finding)**: Originally PR-1 included retrofitting MFDual + DualInferencer "as a pure refactor". Audit showed that's NOT pure — MFDual and DualInferencer have **semantically different** edge-case logic (MFDual: first match; DualInferencer: filtered alphabetical scan). The retrofit IS behavior-preserving but only because the new helper takes a `fallback_strategy` parameter — that's not "pure refactor", that's "carefully matched behavior". To eliminate any risk of unintended divergence, **retrofit moves to PR-2** alongside the new feature additions.

### PR-2 — "Orchestrator path-aware feed injection + retrofit" (BEHAVIOR ADDITIONS + RETROFIT)

Built on top of PR-1. Each fix uses the helper from PR-1.

| # | Fix | Severity | Effort | Risk |
|---|---|---|---|---|
| 6a | Retrofit MFDual `_resolve_flow_output_path` to delegate to helper with `"first_match"` strategy | — | ~30min | Low (existing 11 Part-A tests must still pass; behavior-equivalent by construction) |
| 6b | Retrofit DualInferencer `_resolve_prior_proposer_output_path` to delegate to helper with `"alphabetical_scan"` strategy | — | ~30min | Low (existing Phase 0 tests must still pass; behavior-equivalent by construction) |
| 1 | BTA aggregator: inject `worker_output_paths` into `template_extra_feed` (use `setdefault` to avoid silent overwrite) | 🟡 Med | ~45min | Low (additive) |
| 2 | MultiFlow aggregator: inject `worker_output_paths` into `template_extra_feed` | 🔴 High | ~45min | Low (additive) |
| 3 | PTI executor: inject `plan_output_path` into `template_extra_feed` | 🟡 Med | ~1h | Low (additive; prose fallback preserved) |
| 4 | LWI: add `dynamic_step_output_paths` to state (no API change) | 🔴 High | ~1h | Low (state addition only) |
| 5 | ReflectiveInferencer: inject `reflection_input_path` via `_process_reflection_input` | 🟡 Med | ~1h | Low (default-None param) |
| **PR-2 total** | | | **~5.5h** | |

**Acceptance**: All existing tests pass (including 11 Part-A + all Phase 0 + everything else). New tests cover each fix's happy path + None handling + backward compat + key-collision guard.

### Combined Effort

| Stage | Effort |
|---|---|
| PR-1 (pure addition) | ~1h |
| PR-2 (behavior + retrofit) | ~5.5h |
| Docs & cross-references | ~30min |
| **Total** | **~7h** |

---

## 6. Files to Modify

| File | Fix | Change Summary |
|---|---|---|
| `inferencer_workspace.py` (or new `inferencer_path_helpers.py`) | Phase 0a | Add `resolve_canonical_output_path()` 3-tier helper (Tier 1: deliverables, Tier 2: outputs/, Tier 3: None) |
| `breakdown_then_aggregate_inferencer.py` (~line 269) | Fix 1 | Add `worker_output_paths` to `template_extra_feed.update()` block |
| `multi_flow_inferencer.py` (~line 740) | Fix 2 | Inject `worker_output_paths` into `template_extra_feed` |
| `plan_then_implement_inferencer.py` (~line 553) | Fix 3 | Add `template_extra_feed["plan_output_path"]` injection |
| `linear_workflow_inferencer.py` (~line 561) | Fix 4 | Add `state["dynamic_step_output_paths"]` accumulation (NO signature change) |
| `reflective_inferencer.py` (~lines 167-211, 270) | Fix 5 | Add `base_response_path`/`reflection_paths` + path arg to `_process_reflection_input` |
| `multi_flow_inferencer.py` (`_resolve_flow_output_path` ~line 575) | Fix 6 | Refactor to use Phase 0a helper |

### New Test Files
| File | Tests |
|---|---|
| `test_path_helpers.py` | 10 tests for `resolve_canonical_output_path`: None workspace, no deliverables, deliverable exists, custom filename, "first_match" Tier 1 fallback, "alphabetical_scan" Tier 1 fallback (with dotfile filtering), "none" Tier 1 fallback, **Tier 2 outputs/ resolution (CRITICAL — leaf CLI inferencers)**, **Tier 1 → Tier 2 cascade**, absolute-path guarantee |
| `test_bta_aggregator_path_injection.py` | 3 tests (paths injected, empty list when no workers, backward-compat) |
| `test_mfi_aggregator_path_injection.py` | 3 tests (paths injected, empty list, backward-compat) |
| `test_pti_executor_path_injection.py` | 3 tests (path injected when present, prose still present, None handling) |
| `test_lwi_path_aware_dynamic_input.py` | 5 tests (state contains paths after each step, latest path accessible, full list accessible, None when no deliverable, builder ignoring paths still works) |
| `test_reflective_path_aware_input.py` | 4 tests (sequential, separate, integrate modes; path None handling) |

---

## 7. Acceptance Criteria

### Universal (all fixes)
- [ ] Phase 0a helper has 4 unit tests
- [ ] Each fix is **strictly additive** — existing tests pass without modification
- [ ] Each fix has at least 3 unit tests covering: happy path, None/missing path, backward compat
- [ ] No silent failures: `path == None` is documented "no deliverable available" (silent OK); misuse raises loudly
- [ ] All `template_extra_feed` injections check `if hasattr(target, "template_extra_feed")` first

### Per-fix
| Fix | Specific criterion |
|---|---|
| 1 | BTA `template_extra_feed["worker_output_paths"]` ALWAYS SET to a list of (worker_idx → path-or-None) on EVERY `_inject_aggregator_extra_feed` call (uses `template_extra_feed.update({...})` which already overwrites the key — verified). Empty list `[]` when no workers have deliverables (NOT missing key, NOT stale prior-iteration value). |
| 2 | MFI `template_extra_feed["worker_output_paths"]` ALWAYS SET (unconditional `=`); empty list when no flow has deliverables. |
| 3 | PTI `executor_inferencer.template_extra_feed["plan_output_path"]` ALWAYS SET via unconditional `=`; **uses `None` (not `""`) as sentinel** for "no deliverable" — consistent with helper return type. Templates use `{% if plan_output_path %}` (None is falsy in Jinja); downstream code MUST use `if plan_output_path:` (truthiness) NOT `is not None` (would treat None as truthy-test). |
| 4 | `state["dynamic_step_output_paths"]` is a parallel append-only list (1:1 with `dynamic_step_results`). Each step appends its OWN prior-step path or None. Builders read via `state.get("dynamic_step_output_paths", [])` (NOT `state["..."]` — required for backward-compat with checkpoints created before this fix; matches existing pattern at LWI line 517 for `dynamic_step_results`). Builders MUST also call `os.path.isfile(path)` before consuming the path (cross-machine resume produces stale absolute paths from machine A on machine B; missing-file means treat as None and degrade gracefully — no crash). Values are absolute paths or None (no empty strings). |
| 5 | All three reflection modes (sequential, separate, integrate) plumb path; `reflection_input_path=None` explicit when no deliverable. State key `state.get("base_response_path")` accessed defensively (NOT `state["..."]`); same `os.path.isfile()` validation before consumption for cross-machine resume safety. |
| 6 | MFDual `_resolve_flow_output_path` retrofitted to delegate to `resolve_canonical_output_path` (with `deliverables_fallback="first_match"`); existing 11 Part-A tests must pass. |
| **GLOBAL** | **Sentinel pattern**: All path-aware injections use `None` for "no path" (NOT `""`); rationale: aligns with helper return type `Optional[str]`; Jinja treats both as falsy so templates work either way; Python code checking `is None` semantically clearer. |
| **GLOBAL** | **Assignment pattern**: All injections use unconditional `=` (overwrite) NOT `setdefault`; rationale: `template_extra_feed` is owned by the orchestrator (not user-extensible at this key namespace); stale-path prevention is more important than user-override preservation for these reserved keys. |

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Builder API change risks breakage | ✅ ELIMINATED — Fix 4 uses state-channel; no API change, no `inspect.signature` at all |
| `inspect.signature()` perf overhead | ✅ ELIMINATED — no signature inspection used |
| `template_extra_feed` may be `None` on some inferencers | All injections check `if target.template_extra_feed is None: target.template_extra_feed = {}` |
| Phase 0a helper extraction touches MFDual which already shipped Part A | ✅ Existing 11 Part-A tests must still pass; retrofit uses `"first_match"` strategy = behavior-equivalent |
| `dynamic_step_output_paths` could grow unbounded | Same growth pattern as `dynamic_step_results` (already-bounded by `flow_dynamic_steps`) |
| **Multi-round overwrite** (Agent 4 finding): consensus_max_iterations > 1 → round 2's `template_extra_feed["worker_output_paths"]` overwrites round 1's | Documented as **intentional**: each round's aggregator should see the LATEST workers' paths. Add explicit `# overwrite per round` comment + 1 test that verifies round-2 sees round-2 paths |
| **Template extra feed key collision** (Agent 4 finding): downstream template may already have `worker_output_paths` key from another source | Use `template_extra_feed.setdefault(key, value)` for safe-default semantics. If user explicitly set the key, respect their intent. Document the contract |
| **Cache hash interaction** (Agent 4 finding): adding new keys to `template_extra_feed` changes the hash and may bust resume cache | Verified: cache hash already includes full `template_extra_feed` (else cache would be wrong today). Adding new keys correctly produces a different hash for the new feature; resuming a run from BEFORE the feature shipped will rebuild from scratch (acceptable one-time cost). Add 1 test asserting "same input + same feed = same cache hit" |
| **Workspace symlink aliasing** (Agent 4 finding): if Inferencer A captures path X and Inferencer B writes to X via symlink, A's path now points to B's content | Out of scope for this plan (broader workspace-isolation issue handled by MFDual hygiene plan Part B). Document as known limitation; mitigated by Part B's per-role workspace separation |
| **Path encoding for shell** (Agent 4 finding): paths with spaces/unicode/quotes break shell `cp` commands in templates | Templates MUST use single-quote wrapping: `cp '{{ prior_output_path }}' '{{ output_path }}'`. Document in template authoring guide; add to template review checklist |
| **Stale path persistence** (Round-4+5 audit findings): if any Fix sets `template_extra_feed["..._path(s)"]` only when truthy/present, a stale value from prior iteration persists across multi-round invocations (PTI executor consensus, BTA aggregator re-aggregation, MFI re-aggregation). | ✅ FIXED in §6 — **GLOBAL acceptance criteria** mandate unconditional `=` assignment with `None` (not `""`) as the no-path sentinel for ALL of Fix 1-5. PR-2 tests must include the multi-iteration stale-overwrite scenario for each fix. Fix 4 (LWI state) is naturally per-step (append-only list), no stale risk. |
| **Helper missing Tier 2 outputs/ fallback** (Round-4 audit finding — CRITICAL): leaf CLI inferencers (RovoDevCli, ClaudeCodeCli) write only to `outputs/output.md`, not `final_deliverables/`. Without Tier 2, helper returns `None` for the most common production case | ✅ FIXED in §4 — helper renamed to `resolve_canonical_output_path` with explicit Tier 2 (`outputs/<filename>`) fallback. Matches DualInferencer reference implementation exactly |
| **Resume backward compat — KeyError on old checkpoints** (Round-6 audit finding): If Fix 4 / Fix 5 use `state["dynamic_step_output_paths"]` (direct dict access), resuming a checkpoint from BEFORE the fix shipped would crash with `KeyError`. | ✅ FIXED in §6 — acceptance criteria explicitly require `state.get("dynamic_step_output_paths", [])` with `[]` default; matches existing pattern at LWI line 517 |
| **Cross-machine resume — stale absolute paths** (Round-6 audit finding): Absolute paths captured on machine A point to non-existent paths on machine B during cross-machine resume. Builders consuming raw paths would either crash or pass invalid paths to LLM templates. | ✅ FIXED in §6 — acceptance criteria require builders to call `os.path.isfile(path)` before use; missing-file → treat as None (graceful degradation, no crash). Documented as cross-machine-safe contract. |
| Templates that don't use the new `worker_output_paths` etc. — no change in behavior | ✅ All injections are `template_extra_feed` additions; templates that don't reference them ignore them |
| PTI's executor template may not exist or may need updating | Fix 3 keeps prose as fallback — works whether or not template variable is consumed |
| Stale line numbers in this plan if code changes during implementation | Verify line numbers at start of each fix; the *semantic* anchors (function names) are the source of truth |

---

## 9. Open Questions

| # | Question | Recommendation |
|---|---|---|
| Q1 | Should `worker_output_paths` be a list or a dict-keyed-by-worker-index? | **List** — matches existing `worker_results` shape; index = worker index |
| Q2 | Should ReflectiveInferencer's `iter_infer` (IntegrateAll mode) also collect paths? | **Yes** — for consistency; minor extra effort |
| Q3 | Should the canonical deliverable filename be configurable per orchestrator, or always `"output.md"`? | **Always `"output.md"`** — matches the convention. Make `filename` arg available but default everywhere. |
| Q4 | Should Phase 0a helper live in `inferencer_workspace.py` or a new `inferencer_path_helpers.py` file? | **`inferencer_workspace.py`** — close to `has_deliverables` and `deliverables_dir` properties; avoids new file proliferation |
| Q5 | Should we also update prompt templates (`plan/main/followup.jinja2`, `aggregator/*.jinja2`) to use the new structured paths? | **Phase 2 (separate PR)** — current text-embedded paths still work; structured access is opt-in |
| Q6 | (RESOLVED) Should `_builder_accepts_path` detection handle `**kwargs`-only signatures? | OBSOLETE — Fix 4 now uses state-channel; no signature dispatch exists |
| Q7 | Should we add a deprecation warning for `dynamic_input_builder` callbacks that don't accept path arg? | **No** — text-only is a valid use case (e.g. summarization steps); not all users need paths |

---

## 10. Out of Scope (Deferred)

- **`ConversationalInferencer` message-history paths**: by-design text-only; tool-layer manages artifacts
- **Template authoring** (`plan/main/followup.jinja2` updating to use `{{ worker_output_paths }}` etc.): Phase 2 work; current text-embedded paths still work
- **Removing prose-embedded paths** from BTA/PTI text formatters: keep as fallback for non-templated downstream
- **`inferencer_base.resolve_output_path` silent fallback** (claimed bug in audit Round 1): Verified safe by audit Round 2 — documented behavior for inferencers without workspaces

---

## 11. Implementation Order

### PR-1 (Pure Addition — ship first, verify ZERO behavior change anywhere)
1. **Phase 0a** (~30min): Add `resolve_canonical_output_path` (3-tier) helper to `inferencer_workspace.py`
2. **Phase 0a-tests** (~30min): 8 unit tests covering all parameters and fallback strategies

**PR-1 Acceptance**: ALL existing tests pass without modification. Helper exists, helper has tests, helper is unused. Zero risk of regression because no production code calls it yet.

### PR-2 (Retrofit + Behavior Additions — ship after PR-1 merges)
3. **Fix 6a** (~30min): Retrofit MFDual `_resolve_flow_output_path` to delegate with `"first_match"` strategy; verify all 11 Part-A tests still pass
4. **Fix 6b** (~30min): Retrofit DualInferencer `_resolve_prior_proposer_output_path` to delegate with `"alphabetical_scan"` strategy; verify all Phase 0 tests still pass
5. **Fix 1** (~45min): BTA aggregator inject `worker_output_paths` (with `setdefault` collision guard) + 4 tests (happy path + None + collision + multi-round)
6. **Fix 2** (~45min): MFI aggregator inject `worker_output_paths` + 4 tests
7. **Fix 3** (~1h): PTI executor inject `plan_output_path` + 4 tests (path injected + prose preserved + None handling + cache-hash-stable)
8. **Fix 4** (~1h): LWI add `state["dynamic_step_output_paths"]` + 5 tests
9. **Fix 5** (~1h): ReflectiveInferencer inject `reflection_input_path` + 4 tests

**PR-2 Acceptance**: All existing tests pass (including 11 Part-A + Phase 0 + everything). New tests cover happy path + None + collision + multi-round + cache-hash stability.

**Total: PR-1 (1h) + PR-2 (5.5h) + docs (30min) = ~7h**

---

## 12. Provenance & Verification History

This plan was generated through:

1. **Initial audit** (2026-05-09): 4-agent parallel scan of all `agent_foundation/common/inferencers/` for Phase-0-style bugs
2. **Plan A draft** (LWI + Reflective focus, 373 lines)
3. **Plan B draft** (Orchestrator uniformity focus, 166 lines)
4. **3-agent comparison** (2026-05-10): plan-comparison + verify-A-claims + verify-B-claims
5. **Hard-evidence corrections** (2026-05-10):
   - Plan B claimed BTA "loses" structured paths → FALSE; BTA already injects `deliverables_promoted` etc., just missing per-worker `worker_output_paths`
   - Plan B claimed PTI line 553-561 → FALSE; actual lines 525-578
   - Plan A's `inspect.signature` per-call → FIXED; now detected once at `attrs_post_init`
   - Plan A missed BTA/MultiFlow/PTI → FIXED; integrated as Fixes 1-3
   - Plan B missed ReflectiveInferencer → FIXED; integrated as Fix 5
6. **Phase 0a helper** promoted from Plan A's Q3 to a Phase 0 deliverable for proper DRY
7. **Self-audit (Round 2, 2026-05-10 00:13)**: Identified two ad-hoc patterns in earlier draft and fixed them:
   - **Hack #1**: `inspect.signature()` dispatch in Fix 4 was inconsistent with the rest of the codebase — REPLACED with elegant state-channel approach (`state["dynamic_step_output_paths"]`)
   - **Hack #2**: Bundling Phase 0a refactor + Fixes 1-5 in one PR muddied blame analysis — SPLIT into PR-1 (pure refactor) + PR-2 (behavior additions)
8. **Cross-review (Round 3, 2026-05-10 00:23)**: 4 parallel agents (state-channel critic, PR-split feasibility, bug-vs-not-bug audit, edge-case stress-test). Honest verdicts:
   - Agent 2 caught **TRUE issue**: PR-1's "pure refactor" claim was false — MFDual + DualInferencer have semantically different fallback logic. **FIXED**: PR-1 is now PURE ADDITION (zero retrofit); retrofit moved to PR-2 with explicit `fallback_strategy` parameter
   - Agent 4 caught **5 TRUE issues**: undefined helper signature; multi-round overwrite; key collision; cache hash; symlink aliasing; path encoding. **ALL ADDRESSED** in §4 (signature) and §8 (mitigations)
   - Agent 1 (state-channel critique): consistency claim REJECTED (state-channel is the right layer for callbacks); wrapper alternative ACKNOWLEDGED in Fix 4 alternatives section
   - Agent 3 (gold-plating claim): REJECTED — confused prose embedding with structured access
9. **Final verification**: §4 helper signature now explicit (parameters, return, fallback strategies, absolute paths, never-raises contract) + §8 risks include all 5 Agent 4 mitigations + §5 PR-split corrected
10. **Cross-review (Round 4, 2026-05-10 00:32)**: External agent caught **2 TRUE issues**:
    - **CRITICAL**: Helper was missing Tier 2 (`outputs/<filename>`) fallback. Verified against DualInferencer reference implementation (lines 651-680): the reference has THREE tiers, not just one. Without Tier 2, helper returns `None` for **leaf CLI inferencers** (RovoDevCli, ClaudeCodeCli) which write only to `outputs/`, not `final_deliverables/` — the most common production case. **FIXED**: helper renamed `resolve_canonical_output_path` (more accurate name) and rewritten with explicit 3-tier resolution matching DualInferencer behavior exactly. Test count increased from 8 → 10 to cover Tier 2 + cascade.
    - **MEDIUM**: PTI Fix 3 stale-path bug — if `template_extra_feed["plan_output_path"]` set conditionally, stale value persists across iterations. **FIXED**: added explicit acceptance criterion requiring unconditional `=` assignment with empty-string default.
    - These bugs would have caused the implementation to silently produce empty paths in production. **Critical-thinking discipline working as designed.**
11. **Cross-review (Round 5, 2026-05-10 00:37)**: Four parallel agents audited Round-4 fixes. **5 valid issues caught + 4 false positives correctly rejected**:
    - **VALID — Test count mismatch (Agent A1)**: Acceptance said "4 tests" but spec said "10 tests". **FIXED**: explicit list of 10 tests with critical Tier 2 + cascade scenarios named.
    - **VALID — Fix 4 strategy unspecified (Agent A1)**: LWI helper call had no `deliverables_fallback`. **FIXED**: added explicit `deliverables_fallback="first_match"` to LWI Fix 4 call.
    - **VALID — Stale-path scope (Agent A2)**: Stale-path bug applies to ALL fixes, not just Fix 3. Verified BTA `update({...})` and PTI executor multi-iteration both at risk. **FIXED**: added GLOBAL acceptance criteria for unconditional `=` pattern across Fix 1-5.
    - **VALID — None vs "" sentinel (Agent A2)**: Plan said `or ""` for PTI but helper returns `None`. **FIXED**: GLOBAL sentinel = `None` for consistency; documented Jinja vs Python truthiness implications.
    - **VALID — abspath vs realpath documentation (Agent A4)**: Helper docstring didn't justify the choice. **FIXED**: added explicit "abspath preserves symlinks for `cp` semantics" rationale + TOCTOU caveat + filename contract.
    - REJECTED — Race condition / shared mutable dict (Agent A2): correctly out-of-scope; sequential execution within an inferencer; documented in §10 limitations.
    - REJECTED — Helper rename critique (Agent A3): independently confirmed rename is good, no changes needed.
    - REJECTED — Edge cases #1-5, #7-8 (Agent A4): independently verified all are handled by existing defensive code (try/except, hasattr, isfile checks).
    - REJECTED — Empty filename guard (Agent A4): caller responsibility; documented in helper contract; not worth runtime check overhead.
    - **5 catches across 5 rounds × 4 agents demonstrates that critical-thinking discipline catches real bugs that would have shipped silently broken paths to production.**
12. **Cross-review (Round 6, 2026-05-10 00:49)**: External agent verified resumability across all 6 fixes. **2 valid issues caught**:
    - **VALID — Resume backward compat (Recommendation 1)**: Plan didn't specify defensive `state.get(key, [])` reads. Without this, resuming a pre-Fix-4 checkpoint would raise `KeyError`. Verified existing LWI line 517 ALREADY uses this pattern for `dynamic_step_results`. **FIXED**: acceptance criteria #4 + #5 now mandate `state.get(...)` with default; risks table updated.
    - **VALID — Cross-machine resume safety (Recommendation 2)**: Plan didn't address that absolute paths captured on machine A are stale on machine B. Without `os.path.isfile()` validation, builders pass invalid paths to LLM templates. **FIXED**: acceptance criteria #4 + #5 require pre-consumption file-existence validation with graceful degradation to None.
    - Resumability table from Round 6 (Fix 1-3 inherently safe via runtime re-derivation; Fix 4-5 safe with defensive read + isfile validation; Fix 6 zero impact) confirms full resume coverage. The agent claimed "no remaining issues" but missed that the defensive patterns must be PLAN-LEVEL contracts (not implementation details) — applied that disagreement.
    - **15 valid bugs caught total across 6 rounds**, 7 false positives correctly rejected. Plan is now resume-safe by contract, not by accident.

### Rejected Claims (False Positives)

| Claim Source | Claim | Why Rejected |
|---|---|---|
| Audit Round 1 | `inferencer_base.resolve_output_path` silent fallback bug | Verified safe — documented behavior for non-workspace inferencers |
| Audit Round 1 | `_for_each_child_inferencer` shared mutable dict bug | Local shallow copy; no cache collision possible |
| Audit Round 1 | `_get_result_path` checkpoint None bug | `or self.output_path` fallback at line 839; ValueError raised if both None |
| Plan B Fix 1 framing | BTA "loses paths in text format" | BTA's text already includes `(Full output at: <path>)`; bug is *no separate feed key*, not *path loss* |
| Plan B Fix 3 line number | "lines 553-561" | Actual `_build_executor_input` at lines 525-578 |
| ConversationalInferencer audit | "missing per-turn paths" | By design — tool layer concern, not message history |

---

**End of integrated plan.**
