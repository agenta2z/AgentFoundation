# LWI + ReflectiveInferencer Path-Aware Fix Plan

**Status**: Ready for implementation
**Severity**: Mixed (LWI: HIGH for custom-builder users; Reflective: MEDIUM)
**Estimated effort**: ~2 hours total
**Author**: Synthesized from systematic audit of all inferencers under `agent_foundation/common/inferencers/`
**Date**: 2026-05-09

---

## 1. Background

After landing the MFDual hygiene fixes (Parts A + B + C + D), I performed a systematic audit of all other inferencers in `agent_foundation/common/inferencers/` to check whether the **same bug class** (downstream stages receive only LLM TEXT, not file PATH, causing regeneration drift on large file deliverables — the original Phase 0 bug pattern) appears elsewhere.

### Audit Summary

| Inferencer | Verdict |
|---|---|
| `DualInferencer` | ✅ Already fixed by Phase 0 (path-aware followup) |
| `MultiFlowDualInferencer` | ✅ Already fixed by Part A (peer + own-flow paths) |
| `MultiFlowInferencer` | ✅ Already fixed by Part A |
| `BreakdownThenAggregateInferencer` | ✅ Already correct — `_format_worker_results_text` includes `(See file: <path>)` or `(Full output at: <path>)` |
| `PlanThenImplementInferencer` | ✅ Already correct — `_build_executor_input` includes `plan_file_path` |
| `ConversationalInferencer` | ✅ Acceptable — by design, file paths are tool-layer concern, not message-history concern |
| **`LinearWorkflowInferencer`** | 🔴 **HIGH bug** — `dynamic_input_builder` receives only TEXT, no file path |
| **`ReflectiveInferencer`** | 🟡 **MEDIUM bug** — reflection-input uses TEXT only, no file path |

### Common Bug Pattern (Both)

```python
# Step N produces a large file deliverable + a short text response
state["step_N_response"] = "<short LLM text>"        # ✅ stored
# state["step_N_file_path"] = "<path to artifact>"   # ❌ NOT stored

# Step N+1's input builder receives only the text
inp = builder(state, prev=state["step_N_response"])  # ❌ no path
```

Same pattern as the original Phase 0 bug in `DualInferencer`. When the LLM in step N+1 is told to "refine the previous output," it has only the short text summary, not the actual file content. Result: regeneration drift (smaller, lossy output).

---

## 2. Bug 1 — LinearWorkflowInferencer dynamic_input_builder

### Evidence

**File**: `agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/linear_workflow_inferencer.py`

**Bug location** (lines 517–520):
```python
prev_results = state.get("dynamic_step_results", [])
prev = prev_results[-1] if prev_results else state.get("input")
if self.dynamic_input_builder is not None:
    inp = self.dynamic_input_builder(state, prev)  # ← `prev` is text only
```

**Where `dynamic_step_results` is populated** (lines 559–562):
```python
if "dynamic_step_results" not in state:
    state["dynamic_step_results"] = []
state["dynamic_step_results"].append(actual_result)  # ← actual_result is text
state["dynamic_step_count"] = len(state["dynamic_step_results"])
```

`actual_result` is unpacked from `_resolve_next_inferencer` (line 555) and is the LLM's text response. **No corresponding `dynamic_step_paths` is maintained.**

### Severity Assessment

**HIGH for affected users**, but limited blast radius:

| User type | Affected? |
|---|---|
| Users who use LWI's default text passing (no custom builder) | ❌ Not affected — text-only is the documented contract |
| Users who plug in `dynamic_input_builder` callback for path-aware refinement | 🔴 **Affected** — they cannot access prior step's file path |
| MFDual followup formatter (`_format_followup_input`) | ✅ Not affected — uses `_resolve_flow_output_path()` directly, bypassing this code path |

Real-world impact: any topology that needs to do **path-aware refinement** in dynamic mode (e.g. iterative document refinement where each step writes a file to disk) is broken in subtle ways — the refinement step regenerates content rather than incrementally edits.

### Fix Sketch

**Approach**: Extend `dynamic_step_results` with paths, plumb to builder, keep backward compat.

**Step 1**: Maintain a parallel `dynamic_step_output_paths` list in state.
```python
# In _build_dynamic_step_wrapper, after appending result (line 561):
if "dynamic_step_output_paths" not in state:
    state["dynamic_step_output_paths"] = []
# Resolve path from inf_instance's workspace
prev_path = None
inf_ws = getattr(inf_instance, "_workspace", None)
if inf_ws is not None and getattr(inf_ws, "has_deliverables", False):
    deliverables = getattr(inf_ws, "deliverables_dir", None)
    if deliverables:
        # Look for the canonical output.md
        candidate = os.path.join(str(deliverables), "output.md")
        if os.path.exists(candidate):
            prev_path = candidate
state["dynamic_step_output_paths"].append(prev_path)  # may be None
```

**Step 2**: Pass path to builder via inspectable signature (backward compat).
```python
# Lines 519-520:
if self.dynamic_input_builder is not None:
    # Backward-compatible: only pass prev_path if builder accepts 3 positional args
    import inspect
    sig = inspect.signature(self.dynamic_input_builder)
    n_params = len([p for p in sig.parameters.values()
                    if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  inspect.Parameter.POSITIONAL_ONLY)])
    if n_params >= 3:
        prev_path = (state.get("dynamic_step_output_paths") or [None])[-1]
        inp = self.dynamic_input_builder(state, prev, prev_path)
    else:
        # Legacy 2-arg builder — call as before
        inp = self.dynamic_input_builder(state, prev)
```

**Step 3**: Update type hint and docstring.
```python
# Line 168:
dynamic_input_builder: Optional[Callable[..., Any]] = attrib(default=None)
"""Optional callable. Two signatures supported:

- ``(state, prev) -> input``  — legacy, text-only (backward compat)
- ``(state, prev, prev_path) -> input``  — path-aware; ``prev_path`` is the
  filesystem path to the prior step's deliverable (or ``None`` if not available).
  Use this signature when the next step needs to reference the prior step's
  file artifact (e.g. to ``cp`` it before incremental editing).
"""
```

### Why Not Just Add a Required New Argument?

Real-world users have existing `dynamic_input_builder` callbacks. Forcing a signature change would break them. The `inspect`-based dispatch is the same pattern Python uses for `__init_subclass__` and similar APIs — it's idiomatic and tested.

---

## 3. Bug 2 — ReflectiveInferencer reflection-input

### Evidence

**File**: `agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/reflective_inferencer.py`

**Bug locations** (lines 168–198):
```python
def _build_reflection_input_sequential(self, state):
    reflections = state.get("all_reflections", [])
    if reflections:
        reflection_input = reflections[-1].response  # ← text only
    else:
        reflection_input = state["base_response"]    # ← text only
    return self._process_reflection_input(
        inference_input=state["original_input"],
        reflection_input=reflection_input,           # ← passed as text
        inference_config=state.get("_inference_config", {}),
    )

def _build_reflection_input_separate(self, state):
    return self._process_reflection_input(
        inference_input=state["original_input"],
        reflection_input=state["base_response"],     # ← text only
        ...
    )

def _build_reflection_input_integrate(self, state):
    return self._process_reflection_input(
        inference_input=state["original_input"],
        reflection_input=state["collected_responses"],  # ← text only
        ...
    )
```

### Severity Assessment

**MEDIUM** — narrower use case than LWI:

| Scenario | Affected? |
|---|---|
| ReflectiveInferencer reflecting on short-text base outputs | ❌ Not affected — text suffices |
| ReflectiveInferencer reflecting on file-deliverable base outputs (e.g. plan refinement) | 🟡 **Affected** — same drift risk |
| Existing production topologies | ⚠️ Need to audit — likely only `*-then-reflect` patterns; few in current codebase |

### Fix Sketch

**Approach**: Same as LWI — pass an optional `reflection_input_path` alongside `reflection_input`.

**Step 1**: Surface `base_inferencer` workspace's deliverable path into state.
```python
# Where state["base_response"] is set (need to find this in the dispatch logic):
state["base_response"] = base_result_text
base_ws = getattr(self.base_inferencer, "_workspace", None)
state["base_response_path"] = None
if base_ws is not None and getattr(base_ws, "has_deliverables", False):
    deliverables = getattr(base_ws, "deliverables_dir", None)
    if deliverables:
        candidate = os.path.join(str(deliverables), "output.md")
        if os.path.exists(candidate):
            state["base_response_path"] = candidate
```

**Step 2**: Plumb the path through the three reflection-input builders.
```python
def _build_reflection_input_sequential(self, state):
    reflections = state.get("all_reflections", [])
    if reflections:
        reflection_input = reflections[-1].response
        reflection_input_path = state.get("reflection_paths", [None])[-1]
    else:
        reflection_input = state["base_response"]
        reflection_input_path = state.get("base_response_path")
    return self._process_reflection_input(
        inference_input=state["original_input"],
        reflection_input=reflection_input,
        reflection_input_path=reflection_input_path,  # ← NEW
        inference_config=state.get("_inference_config", {}),
    )
# Apply same pattern to _separate and _integrate variants
```

**Step 3**: Update `_process_reflection_input` to accept and use the path.
```python
def _process_reflection_input(
    self,
    inference_input,
    reflection_input,
    reflection_input_path=None,  # ← NEW, backward compat
    inference_config=None,
):
    """If reflection_input_path is provided, downstream prompt template can
    reference it for cp-style preservation (avoids regeneration drift)."""
    # Existing logic, plus inject reflection_input_path into template_extra_feed
    if reflection_input_path is not None and self.reflection_inferencer is not None:
        if not hasattr(self.reflection_inferencer, "template_extra_feed"):
            return existing_logic(...)
        if self.reflection_inferencer.template_extra_feed is None:
            self.reflection_inferencer.template_extra_feed = {}
        self.reflection_inferencer.template_extra_feed["reflection_input_path"] = reflection_input_path
    return existing_logic(...)
```

**Step 4**: Update reflection prompt template to use the path (optional, follow-up).
```jinja2
{# In whatever template the reflection_inferencer uses, add: #}
{% if reflection_input_path %}
The previous version is saved on disk at: `{{ reflection_input_path }}`

To preserve content and avoid regeneration drift, your **FIRST tool action MUST be**
to copy the previous file to your output path before any edit.
{% endif %}
```

(Same pattern as the Phase 0 fix in `plan/main/followup.jinja2`.)

---

## 4. Acceptance Criteria

### Bug 1 (LWI)
- [ ] `dynamic_step_output_paths` list maintained in state alongside `dynamic_step_results`
- [ ] 3-arg `dynamic_input_builder` signature supported (backward compat for 2-arg)
- [ ] Path is `None` when prior step has no file deliverable (not raise)
- [ ] Test: 2-arg builder works unchanged
- [ ] Test: 3-arg builder receives correct path
- [ ] Test: 3-arg builder receives `None` when prior step has no deliverable
- [ ] Test: works with both static and dynamic mode steps

### Bug 2 (Reflective)
- [ ] `state["base_response_path"]` set when base_inferencer has deliverable
- [ ] `state["reflection_paths"]` accumulated for sequential mode
- [ ] `_process_reflection_input` accepts optional `reflection_input_path`
- [ ] `template_extra_feed["reflection_input_path"]` populated when path available
- [ ] Test: existing reflection paths unchanged when no file deliverable
- [ ] Test: path appears in extra_feed when base_inferencer produces deliverable
- [ ] Test: sequential mode chains paths correctly across reflections

### Both
- [ ] Zero regressions in existing test suites
- [ ] No silent failures (path == None is silent OK; misuse raises loudly)
- [ ] Documentation updated for new signature and field

---

## 5. Files to Modify

### Bug 1 (LWI)
| File | Change |
|---|---|
| `linear_workflow_inferencer.py` (~line 168, 517-520, 559-562) | Add `dynamic_step_output_paths` plumbing + `inspect`-based 3-arg dispatch |

### Bug 2 (Reflective)
| File | Change |
|---|---|
| `reflective_inferencer.py` (~line 168-211, 271+) | Add `base_response_path`, `reflection_paths`, plumb to `_process_reflection_input` |

### New Tests
| File | Tests |
|---|---|
| `test/agent_foundation/common/inferencers/test_lwi_path_aware_dynamic_input.py` | 4-6 tests |
| `test/agent_foundation/common/inferencers/test_reflective_path_aware_input.py` | 4-6 tests |

### Templates (Optional Phase 2)
| File | Change |
|---|---|
| `resources/prompt_templates/reflection/main/followup.jinja2` (if exists) | Add path-aware block (mirrors `plan/main/followup.jinja2`) |

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Existing 2-arg `dynamic_input_builder` callbacks would break with mandatory new arg | ✅ Use `inspect.signature` to dispatch (3-arg new, 2-arg legacy) |
| Resolving file path on every step adds latency | ✅ Negligible — single `os.path.exists()` per step |
| `dynamic_step_output_paths` could grow unbounded for long-running flows | ✅ Same pattern as `dynamic_step_results` (already grows; same bound) |
| ReflectiveInferencer's `_process_reflection_input` is called from multiple places (lines 175, 183, 191, 211, 304, 353) | ✅ Default `reflection_input_path=None` keeps all sites backward-compat |
| Template doesn't yet exist for reflection followup | ✅ Mark Phase 2 as optional; Phase 1 just makes the path AVAILABLE in extra_feed for any future template |

---

## 7. Implementation Order

1. **Phase 1 (Bug 1, ~1h)**: LWI dynamic_input_builder path-aware
   - Helper: extract path resolution to private method `_resolve_step_output_path`
   - Modify wrapper: maintain `dynamic_step_output_paths`
   - Modify dispatch: `inspect`-based 3-arg detection
   - Tests: 4-6 unit tests

2. **Phase 2 (Bug 2, ~30min)**: Reflective `_process_reflection_input` path-aware
   - Surface `base_response_path` in state
   - Plumb through 3 builder variants
   - Update `_process_reflection_input` signature
   - Tests: 4-6 unit tests

3. **Phase 3 (Optional, ~30min)**: Update reflection followup template (if exists)
   - Add path-aware block matching `plan/main/followup.jinja2`

---

## 8. Open Questions

| # | Question | Recommendation |
|---|---|---|
| Q1 | Should `dynamic_input_builder` use `inspect`-based dispatch (3-arg auto-detect) or a separate explicit `path_aware_dynamic_input_builder` attribute? | Inspect-based — less API surface, cleaner per-call flexibility |
| Q2 | Should we add the same path-aware support to `ReflectiveInferencer.iter_infer` (collected_responses) for IntegrateAll mode? | Yes — for consistency; minor extra effort |
| Q3 | Should the path resolution use `_resolve_flow_output_path` from MFDual or duplicate the logic in LWI? | Extract to a shared helper in `inferencer_workspace.py` (e.g. `resolve_canonical_deliverable_path(workspace)`) — DRY across MFDual + LWI + Reflective |
| Q4 | If we extract a shared helper (Q3), should we also retrofit MFDual's `_resolve_flow_output_path` to use it? | Yes — small refactor; keeps one source of truth |
| Q5 | Where should the path-aware reflection block live in the prompt template? | Match `plan/main/followup.jinja2` line 11-15 structure for consistency |

---

## 9. Out of Scope (Deferred)

- **ConversationalInferencer message-history paths**: by-design text-only; tool-layer manages artifacts
- **BTA aggregator path passing**: already correct (`_format_worker_results_text`)
- **PTI plan-to-impl path passing**: already correct (`_build_executor_input`)
- **`inferencer_base.py:1200-1202` silent fallback**: verified safe; documented as "simple API inferencers never set workspace"

---

## 10. Provenance

This plan was generated after a 4-agent parallel audit of all inferencers in `agent_foundation/common/inferencers/`. The two bugs identified here passed:
- Round 1: claim-by-claim verification with hard code reads
- Round 2: scope-impact analysis (who actually triggers the bug?)
- Round 3: solution-design check (is the fix backward-compatible?)

False-positive claims that were rejected:
- BTA aggregator TEXT-only — already correct (line 612)
- PTI plan-to-impl path missing — already correct (line 553)
- `inferencer_base.resolve_output_path` silent fallback — documented safe behavior
- `_for_each_child_inferencer` shared mutable dict — local copy, no collision
- `_get_result_path` checkpoint None — `or self.output_path` fallback present
