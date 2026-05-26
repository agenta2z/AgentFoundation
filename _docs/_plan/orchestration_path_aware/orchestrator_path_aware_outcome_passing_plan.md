# Orchestrator Path-Aware Outcome Passing — Fix Plan

**Status**: Ready to implement
**Created**: 2026-05-10
**Related**: Phase 0 DualInferencer fix (shipped), MFDual hygiene plan (shipped)
**Scope**: BTA aggregator, MultiFlow aggregator, PTI executor, LWI step-to-step

---

## Problem

DualInferencer is the ONLY orchestrator that correctly passes structured file paths (`prior_output_path` as a separate feed dict key) to downstream nodes. All other orchestrators embed paths in formatted text strings or don't pass them at all. Downstream LLMs receive text summaries and regenerate from scratch instead of reading the actual file artifacts.

### Empirical Evidence

From the MFDual hygiene audit (2026-05-09):
- BTA aggregator's `worker_output_paths` list is COLLECTED (line 1435-1508) but CONVERTED to text in `_format_worker_results_text()` — structured paths lost
- MultiFlow aggregator has `worker_output_paths` in feed dict but `upstream_artifacts` is rendered TEXT
- PTI executor receives plan path embedded in prose (`"The full plan is at: {path}"`) — unstructured
- LWI's `dynamic_input_builder` has NO path mechanism at all

### The Correct Pattern (from DualInferencer, verified working)

```python
# Structured: downstream gets a dict key with the path
feed["prior_output_path"] = "/path/to/output.md"

# NOT: path buried in prose text
text += f"\n(See file: {path})"  # unstructured, fragile
```

---

## Fix 1: BTA Aggregator — Pass `worker_output_paths` as structured feed

**File:** `breakdown_then_aggregate_inferencer.py`

### Current Bug (lines 638-641)

`worker_output_paths` is collected during worker execution but then `_format_worker_results_text()` converts everything to a formatted text string. The structured path list is lost when injected into `template_extra_feed["upstream_artifacts"]`.

### Fix

After injecting the text `upstream_artifacts`, ALSO inject the structured paths:

```python
# At the injection site (~line 641):
target.template_extra_feed["upstream_artifacts"] = rendered_text  # existing
target.template_extra_feed["worker_output_paths"] = list(worker_output_paths)  # NEW
```

The aggregator template can then use `{{ worker_output_paths }}` to give the LLM structured access to each worker's output file.

### Effort: ~1h (code change + test)

---

## Fix 2: MultiFlow Aggregator — Same pattern

**File:** `multi_flow_inferencer.py`

### Current Bug (lines 701-740)

`worker_output_paths` exists in the feed dict (line 720) but is never injected into `template_extra_feed`. The aggregator only sees the rendered text from `worker_plans`.

### Fix

Same as Fix 1 — inject `worker_output_paths` alongside `upstream_artifacts`:

```python
# At the injection site (~line 740):
target.template_extra_feed["upstream_artifacts"] = rendered  # existing
target.template_extra_feed["worker_output_paths"] = list(worker_output_paths or [])  # NEW
```

### Effort: ~1h

---

## Fix 3: PTI Executor — Pass plan path as structured feed key

**File:** `plan_then_implement_inferencer.py`

### Current Bug (lines 553-561)

The plan file path is embedded in a text string:
```python
f"The full approved plan is at: `{plan_file_path}`\nRead that file..."
```

This is unstructured — the executor LLM must parse the prose to find the path. If the text format changes, parsing breaks.

### Fix

Pass `plan_file_path` as a separate feed key via `extra_feed` or `template_extra_feed`:

```python
# When calling executor_inferencer.ainfer():
executor_inferencer.template_extra_feed["plan_output_path"] = plan_file_path or ""
```

The executor template can then use `{{ plan_output_path }}` directly. Keep the existing text reference as a fallback for non-templated executors.

### Effort: ~1.5h (need to verify executor template exists and add the variable)

---

## Fix 4: LWI Step-to-Step — Expose prior step's output path

**File:** `linear_workflow_inferencer.py`

### Current Bug (lines 519-522)

`dynamic_input_builder` receives `(state, prev_result)` where `prev_result` is the previous step's text output. No file path is available. The builder cannot tell the LLM where the prior step's output lives on disk.

### Fix

Extend the `dynamic_input_builder` callback signature to include the prior step's output path:

```python
# In _build_dynamic_step_wrapper, when calling dynamic_input_builder:
prev_path = None
if inf_instance is not None:
    prev_ws = getattr(inf_instance, "_workspace", None)
    if prev_ws is not None:
        candidate = prev_ws.output_path("output.md")
        if os.path.isfile(candidate):
            prev_path = candidate

# Pass to builder (backward-compatible via **kwargs or optional param):
if self.dynamic_input_builder is not None:
    inp = self.dynamic_input_builder(state, prev, prior_output_path=prev_path)
```

Existing builders that don't accept `prior_output_path` are unaffected (it's a keyword arg they can ignore). New builders can use it.

### Effort: ~1.5h (signature change + verify no callers break)

---

## Files to Modify

| File | Fix | Change |
|------|-----|--------|
| `breakdown_then_aggregate_inferencer.py` | 1 | Inject `worker_output_paths` into aggregator's `template_extra_feed` |
| `multi_flow_inferencer.py` | 2 | Same pattern for MultiFlow's aggregator |
| `plan_then_implement_inferencer.py` | 3 | Pass `plan_output_path` as structured feed key to executor |
| `linear_workflow_inferencer.py` | 4 | Extend `dynamic_input_builder` with `prior_output_path` kwarg |

## Sequencing

| # | Fix | Effort | Risk |
|---|-----|--------|------|
| 1 | BTA Aggregator | ~1h | Low — additive, existing text path preserved |
| 2 | MultiFlow Aggregator | ~1h | Low — same pattern as Fix 1 |
| 3 | PTI Executor | ~1.5h | Medium — need to verify template variable propagation |
| 4 | LWI Step-to-Step | ~1.5h | Medium — callback signature change, verify no callers break |
| **Total** | | **~5h** | |

## Verification

1. **Fix 1:** Run BTA topology → verify aggregator's `template_extra_feed` contains `worker_output_paths` list
2. **Fix 2:** Run MultiFlow topology → same check for MultiFlow's aggregator
3. **Fix 3:** Run PTI topology → verify executor's feed has `plan_output_path` as separate key
4. **Fix 4:** Run LWI with custom `dynamic_input_builder` → verify `prior_output_path` kwarg received
5. **Regression:** All existing tests pass (no signature breakage)
