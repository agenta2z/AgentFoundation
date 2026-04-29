# Plan: `BreakdownThenAggregateInferencer` — Skip-Breakdown / Predefined Sub-Queries

**Date:** 2026-04-16 (revised — integrated from two agent plans)
**File:** `agent_foundation/common/inferencers/agentic_inferencers/flow_inferencers/breakdown_then_aggregate_inferencer.py`

---

## 1. Problem Statement

`BreakdownThenAggregateInferencer` (BTA) always requires a `breakdown_inferencer` to decompose a query into sub-queries at runtime. Two use cases need a bypass:

1. **Caller-supplied sub-queries**: The caller already has a list (external planner, prior run, YAML config) and wants BTA to skip LLM breakdown entirely.
2. **Single-query repetition / parallel sampling**: The caller has one query and wants to run it N times in parallel (ensemble, diverse perspectives), with N derived from the class config.

---

## 2. Critical Comparison of the Two Agent Plans

### 2.1 Point of Agreement — naming the flag

Both plans agree on `disable_breakdown` vs `predefined_sub_queries` as the mechanism for skipping. The key disagreement is **where the sub-query list lives**: class attribute vs call-time kwarg. See section 2.2.

### 2.2 CORE DISAGREEMENT — class attribute vs call-time kwarg

| | Plan A (my plan) | Plan B (other agent) |
|---|---|---|
| **Skip mechanism** | `disable_breakdown: bool` attrib + `sub_queries` kwarg to `infer()` | `predefined_sub_queries` attrib (str or List) |
| **Sub-query list** | Passed at **call time** via `infer(..., sub_queries=[...])` | Stored on the **object** at construction time |
| **Single-string auto-repeat** | No — infers N from `max_concurrency` only | Yes — string value triggers replication |
| **Fallback N** | `max_concurrency or 1` | `max_breakdown or max_concurrency or 3` |

**Critical analysis:**

Plan A (call-time kwarg) argument: sub-queries are per-call dynamic data, not object config.

Plan B (class attribute) argument: consistent with the rest of BTA's design — ALL phase-control knobs (`breakdown_only`, `disable_aggregator`, `max_breakdown`, `breakdown_format`, `worker_factory`) are class attributes, not call-time kwargs. The BTA object IS the configured pipeline; callers just pass the top-level query to `infer()`. YAML configs define the entire pipeline statically, including predefined sub-queries. This is the dominant usage pattern.

**Verdict: Plan B's `predefined_sub_queries` class attribute is architecturally correct for BTA.**

Rationale:
- Every other phase-control knob in BTA is a class attribute
- YAML-driven usage (the primary use case) cannot use call-time kwargs
- Object reuse with *different* sub-queries each call is NOT the primary use case — if you need that, you instantiate a new BTA or use a different pattern
- The user's own words: "inferencer accepts a list as predefined breakdown" — "predefined" is a class-level concept

However, Plan B's design still has issues we will fix (see section 2.3 onwards).

### 2.3 ISSUE — Plan B puts `predefined_sub_queries` BEFORE checkpoint (Step -1)

Plan B inserts the predefined block *before* the checkpoint check, making it highest priority and overriding `resume_with_saved_results`. This is **wrong**:

- `resume_with_saved_results` is an explicit opt-in to resume from a prior run
- A checkpoint from a prior run is already computed work that should take precedence over reconfiguration
- Plan A correctly puts skip-breakdown AFTER the checkpoint (Step 0b)
- Consistent with all other BTA modes: checkpoint always wins

**Verdict: Plan A's ordering (checkpoint first, predefined second) is correct.**

### 2.4 ISSUE — Plan B's single-string fallback N formula

Plan B: `num_workers = self.max_breakdown or self.max_concurrency or 3`

Problems:
- `max_breakdown` is a *cap*, not a target worker count. Using it as a target conflates two different semantics. `max_breakdown=5` means "don't exceed 5 sub-queries" — it doesn't mean "I want 5 workers."
- The hardcoded fallback of `3` is arbitrary and surprising. A caller with `max_concurrency=None` and `max_breakdown=None` gets 3 workers with no explanation.

Plan A: `num_workers = self.max_concurrency or 1`

Problems:
- `max_concurrency` is a concurrency *throttle* for async, not a target worker count. Using it as the worker count conflates semantics too — though it is a better proxy than `max_breakdown`.
- Also has the deadlock warning: with an aggregator, `max_concurrency` must be > num_workers (per the class docstring warning).

**Better formula — use a dedicated attribute:**

Neither plan's formula is ideal. The cleanest solution is a **dedicated `num_workers` attribute** for the auto-repeat case. However, since the user asked to minimise new API surface, the pragmatic best choice from existing attributes is:

`num_workers = self.max_breakdown or self.max_concurrency or 1`

With `max_breakdown` as the **primary** source (it's the most natural "how many workers do I want" config in BTA) and explicit `1` as the fallback (never silent magic number). The caller who wants N workers sets `max_breakdown=N`. This is actually semantically sound for the single-string case: "run this one query, broken into max_breakdown parallel instances."

**Verdict: Integrated formula: `max_breakdown or max_concurrency or 1`. Fallback is 1, not 3.**

### 2.5 NAMING — `disable_breakdown` vs `predefined_sub_queries`

Plan A: `disable_breakdown: bool` (verb pattern matching `disable_aggregator`)
Plan B: `predefined_sub_queries: Optional[Union[str, List]]` (noun, descriptive)

Critical analysis:
- `disable_aggregator` and `breakdown_only` are pure behavioural flags that say *what to do*
- `predefined_sub_queries` is both a flag AND data — it says *what to use*
- The user's request: "skip breakdown (e.g. through an attribute `disable_breakdown`)" — user explicitly mentioned this naming
- BUT: having two separate attributes (`disable_breakdown=True` + data passed elsewhere) is clunkier than one unified attribute that both signals the skip AND carries the data
- Plan B's single attribute is more ergonomic: `predefined_sub_queries=["a","b"]` is self-explanatory; `predefined_sub_queries=None` means "don't skip"

**Verdict: Plan B's unified `predefined_sub_queries` attribute wins on ergonomics. The presence of a non-None value implicitly disables breakdown — no separate boolean needed.**

### 2.6 TYPE — `Optional[Union[str, List]]`

Plan B allows a single string as the value, which auto-replicates. This is a nice ergonomic touch for the auto-repeat use case. Plan A only supported lists.

**Verdict: Plan B's `Union[str, List]` typing is better — keep it.**

### 2.7 ISSUE — `breakdown_only` interaction

Plan B doesn't address `breakdown_only=True` + `predefined_sub_queries` being set simultaneously.

When `predefined_sub_queries` is set, there is no breakdown output to return. `breakdown_only=True` is meaningless/contradictory. Must be explicitly handled.

**Verdict: When `predefined_sub_queries` is set and `breakdown_only=True`, log a warning and proceed normally (ignore `breakdown_only`).**

### 2.8 ISSUE — `_infer` vs `_ainfer` asymmetry in Plan B

Plan B shows identical code for `_infer` and `_ainfer` but the actual `_ainfer` has the `use_async=True / try/finally` guard around `_build_diamond_graph` (lines 897–907, 970–980). Plan B's pseudocode omits this for `_ainfer`. This would be a bug — async workers would be built as sync functions.

**Verdict: Must include the `use_async=True / try/finally` guard in `_ainfer` Step 0b (Plan A correctly identified this).**

### 2.9 ISSUE — `breakdown_inferencer` null-guard

Both plans note this but plan B doesn't add the null-guard `ValueError`. Currently `breakdown_inferencer=None` with no predefined sub-queries gives a cryptic `AttributeError` at line 846.

**Verdict: Add explicit `ValueError` guard (Plan A's suggestion).**

### 2.10 ISSUE — `breakdown_only` in `_infer` is checked AFTER cap (Step 3) in `_ainfer` but NOT in `_infer`

Looking at the actual code: in `_ainfer`, `breakdown_only` is checked at Step 3b (line 947), after parsing and capping. In `_infer`, there is NO `breakdown_only` check at all! This is a pre-existing bug/asymmetry unrelated to our change — but we should not make it worse.

### 2.11 PLAN B's `return inference_input` when sub_queries empty after cap

Plan B returns `inference_input` when the sub-queries list is empty after cap. Plan A returns `""`. The actual code returns `""` in checkpoint resume path (line 832) and `raw_output` in the normal path (line 870). Since there's no `raw_output` in the predefined path, `""` is consistent with the checkpoint resume path and is correct.

**Verdict: Return `""` when sub_queries is empty after cap (matches checkpoint resume path).**

---

## 3. Integrated Design Decision

### Single new class attribute:

```python
predefined_sub_queries: Optional[Union[str, List]] = attrib(default=None, kw_only=True)
```

- `None` (default): normal breakdown phase runs — **zero behaviour change**
- `List[str]` or `List[dict]`: skip breakdown, use list directly as `sub_queries`
- `str`: skip breakdown, replicate the string `N` times where `N = max_breakdown or max_concurrency or 1`

**No separate `disable_breakdown` boolean.** The presence of a non-None value is the flag.

### Placement:

After `disable_aggregator` (line 180), before `start_nodes`:

```python
breakdown_only: bool = attrib(default=False)
disable_aggregator: bool = attrib(default=False)
# When set, skips the LLM breakdown phase and uses these as sub_queries directly.
# See class docstring "Predefined sub-queries mode" for full semantics.
predefined_sub_queries: Optional[Union[str, List]] = attrib(default=None, kw_only=True)  # ← NEW
```

### Auto-repeat N formula:

```python
num_workers = self.max_breakdown or self.max_concurrency or 1
```

---

## 4. Behaviour Specification

### When `predefined_sub_queries` is not None:

The entire LLM breakdown phase (Steps 1–2b) is skipped. Sub-queries are:

| Value | Behaviour |
|---|---|
| `["q1", "q2", ...]` | Use list directly as `sub_queries` |
| `[{"query": "...", "args": {...}}, ...]` | Use structured list (heterogeneous dispatch) |
| `"single query string"` | Replicate N times: `N = max_breakdown or max_concurrency or 1` |
| `[]` (empty list) | Return `""` immediately (consistent with checkpoint resume path) |

### Priority order (highest first):

1. **`resume_with_saved_results` checkpoint** (Step 0 — unchanged, always wins)
2. **`predefined_sub_queries`** (Step 0b — NEW)
3. **Normal LLM breakdown** (Step 1 — unchanged)

### Interaction with existing flags:

| Combo | Behaviour |
|---|---|
| `predefined_sub_queries=X, breakdown_only=True` | Log warning, ignore `breakdown_only`, proceed normally |
| `predefined_sub_queries=X, disable_aggregator=True` | Workers run, aggregation skipped — existing path handles this |
| `predefined_sub_queries=X, max_breakdown=N` | Cap still applied after resolving list |
| `predefined_sub_queries=X, resume_with_saved_results=True` + checkpoint exists | Checkpoint wins (Step 0 runs first) |
| `predefined_sub_queries=None` (default) | Existing behaviour — completely unchanged |

### `breakdown_inferencer` nullability:

- `predefined_sub_queries` set: `breakdown_inferencer` may be `None` — not called
- `predefined_sub_queries=None`: `breakdown_inferencer=None` raises `ValueError` with helpful message (currently: silent `AttributeError`)

---

## 5. New Attribute

```python
# === Predefined sub-queries (skip breakdown) ===
# When set, skips the LLM-driven breakdown phase entirely.
# Accepts:
#   - List[str]: each string becomes a worker query.
#   - List[dict]: each dict has "query" and optional "args" fields
#     (same format as produced by breakdown + json_subtasks parsing).
#     Enables heterogeneous worker dispatch when task_type_arg_name is set.
#   - str: single query — replicated to N workers where
#     N = max_breakdown or max_concurrency or 1.
#     Useful for parallel sampling / diverse perspectives on one query.
# When None (default): normal LLM breakdown phase runs.
# breakdown_inferencer is not required when predefined_sub_queries is set.
# Note: resume_with_saved_results checkpoint takes priority over this field.
predefined_sub_queries: Optional[Union[str, List]] = attrib(default=None, kw_only=True)
```

**Placement:** After `disable_aggregator` at line 180, before `start_nodes`.

---

## 6. Helper Method: `_resolve_predefined_sub_queries()`

Add a private method to keep `_infer`/`_ainfer` clean:

```python
def _resolve_predefined_sub_queries(self) -> List:
    """Resolve predefined_sub_queries into a list of sub-queries.

    Called only when self.predefined_sub_queries is not None.

    Returns:
        List of sub-queries (strings or dicts) to pass to _build_diamond_graph.
    """
    psq = self.predefined_sub_queries
    if isinstance(psq, str):
        # Auto-repeat mode: replicate string query N times
        n = self.max_breakdown or self.max_concurrency or 1
        _logger.info(
            "predefined_sub_queries: auto-repeating single query x%d "
            "(max_breakdown=%s, max_concurrency=%s)",
            n, self.max_breakdown, self.max_concurrency,
        )
        return [psq] * n
    elif isinstance(psq, list):
        _logger.info(
            "predefined_sub_queries: using caller-supplied list of %d sub_queries",
            len(psq),
        )
        return list(psq)
    else:
        # Unexpected type — coerce to single-item list with warning
        _logger.warning(
            "predefined_sub_queries: unexpected type %s, coercing to string",
            type(psq).__name__,
        )
        return [str(psq)]
```

**Placement:** Near `_load_breakdown_checkpoint` / `_save_breakdown_checkpoint` (lines 707–753).

---

## 7. Changes to `_infer()`

**Current structure:**
```
Step 0:  Load breakdown checkpoint  → if found, skip to graph
Step 1:  breakdown_inferencer.infer(...)
Step 2:  Parse breakdown output
Step 2b: Save breakdown checkpoint
Step 3:  Apply max_breakdown cap
Step 4:  Build diamond graph
Step 5:  Run graph
```

**New structure — insert Step 0b after checkpoint, before breakdown:**

```python
def _infer(self, inference_input, inference_config=None, **kwargs):
    """Core inference: breakdown → build graph → run graph."""

    # Step 0: Check for saved breakdown checkpoint (highest priority — unchanged)
    sub_queries = self._load_breakdown_checkpoint()
    if sub_queries is not None:
        # [existing checkpoint resume code — unchanged]
        ...

    # Step 0b: Predefined sub-queries — skip breakdown entirely — NEW
    if self.predefined_sub_queries is not None:
        if self.breakdown_only:
            _logger.warning(
                "predefined_sub_queries is set but breakdown_only=True — "
                "breakdown_only ignored (no LLM breakdown to stop after)."
            )
        sub_queries = self._resolve_predefined_sub_queries()
        # Apply max_breakdown cap (consistent with checkpoint resume path)
        if self.max_breakdown is not None and len(sub_queries) > self.max_breakdown:
            sub_queries = sub_queries[: self.max_breakdown]
        if not sub_queries:
            return ""
        self._build_diamond_graph(
            sub_queries,
            inference_config=inference_config,
            _original_query=inference_input,
            **kwargs,
        )
        result = WorkGraph._run(self, inference_input, **kwargs)
        if isinstance(result, tuple) and len(result) == 1:
            result = result[0]
        self._finalize_response(result)
        return result

    # Step 1: Breakdown (existing — with improved null-guard)
    if self.breakdown_inferencer is None:
        raise ValueError(
            "breakdown_inferencer must be set when predefined_sub_queries is None. "
            "Either provide a breakdown_inferencer or set predefined_sub_queries."
        )
    raw_output = self.breakdown_inferencer.infer(
        inference_input, inference_config=inference_config
    )
    # [Steps 2–5 unchanged]
```

---

## 8. Changes to `_ainfer()`

Mirror of `_infer()`. Two critical requirements:
1. **`use_async=True / try/finally` guard** around `_build_diamond_graph` (lines 970–980 pattern) — without this, async workers are built as sync functions (silent bug)
2. **Must include `enable_checkpoint_results_review` block** (lines 988–1001) — the predefined path should NOT skip the interactive results review, since a re-run would correctly re-use the same predefined sub-queries. Skip only `enable_checkpoint_sub_query_selection` (the user already chose their sub-queries deliberately).

```python
async def _ainfer(self, inference_input, inference_config=None, **kwargs):
    """Async core inference: breakdown → build graph → run graph."""

    # Step 0: Check for saved breakdown checkpoint (unchanged)
    sub_queries = self._load_breakdown_checkpoint()
    if sub_queries is not None:
        # [existing code — unchanged, already has use_async guard]
        ...

    # Step 0b: Predefined sub-queries — skip breakdown entirely — NEW
    if self.predefined_sub_queries is not None:
        if self.breakdown_only:
            _logger.warning(
                "predefined_sub_queries is set but breakdown_only=True — "
                "breakdown_only ignored (no LLM breakdown to stop after)."
            )
        sub_queries = self._resolve_predefined_sub_queries()
        if self.max_breakdown is not None and len(sub_queries) > self.max_breakdown:
            sub_queries = sub_queries[: self.max_breakdown]
        if not sub_queries:
            return ""
        # NOTE: skip enable_checkpoint_sub_query_selection — user already chose sub-queries.
        # CRITICAL: set use_async=True so _build_diamond_graph creates async worker fns
        old_use_async = getattr(self, "use_async", False)
        self.use_async = True
        try:
            self._build_diamond_graph(
                sub_queries,
                inference_config=inference_config,
                _original_query=inference_input,
                **kwargs,
            )
        finally:
            self.use_async = old_use_async  # always restore
        result = await WorkGraph._arun(self, inference_input, **kwargs)
        if isinstance(result, tuple) and len(result) == 1:
            result = result[0]
        # Step 5b: Interactive results review — keep this even in predefined mode.
        # A re-run will correctly re-use the same predefined_sub_queries.
        if self.enable_checkpoint_results_review and self.interactive:
            # TODO: interactive_checkpoint module does not exist at agent_foundation.ui — needs separate migration
            from agent_foundation.ui.interactive_checkpoint import checkpoint_results_review
            result_str = str(result)[:2000]
            cp_result = await checkpoint_results_review(
                self.interactive, result_str, default_action="approve"
            )
            if cp_result.action == "rerun":
                return await self._ainfer(inference_input, inference_config, **kwargs)
        self._finalize_response(result)
        return result

    # Step 1: Breakdown (existing — with improved null-guard)
    if self.breakdown_inferencer is None:
        raise ValueError(
            "breakdown_inferencer must be set when predefined_sub_queries is None. "
            "Either provide a breakdown_inferencer or set predefined_sub_queries."
        )
    if hasattr(self.breakdown_inferencer, "ainfer"):
        raw_output = await self.breakdown_inferencer.ainfer(
            inference_input, inference_config=inference_config
        )
    else:
        raw_output = self.breakdown_inferencer.infer(
            inference_input, inference_config=inference_config
        )
    # [Steps 2–5 unchanged]
```

---

## 9. Changes to `__attrs_post_init__()`

No changes needed. The existing guard already handles `breakdown_inferencer=None`:

```python
if self._workspace is not None and self.breakdown_inferencer is not None:
    breakdown_ws = self._workspace.child("breakdown")
    ...
```

When `predefined_sub_queries` is set and `breakdown_inferencer=None`, this block is correctly skipped.

---

## 10. Changes to Class Docstring

Add a new section after the existing `Concurrency control:` block:

```
Predefined sub-queries mode:
    Set ``predefined_sub_queries`` to bypass the LLM-driven breakdown phase.
    Sub-queries are resolved as follows:

    - ``List[str]`` or ``List[dict]``: used directly as sub_queries.
      ``breakdown_inferencer`` is not required.
    - ``str`` (single query): replicated to N workers where
      ``N = max_breakdown or max_concurrency or 1``.
      Useful for parallel sampling or diverse perspectives on one query.

    ``max_breakdown`` still caps the resolved sub-query list.
    A saved checkpoint (``resume_with_saved_results``) takes priority and
    overrides ``predefined_sub_queries`` when found (checkpoint is loaded first).
    Setting ``breakdown_only=True`` alongside ``predefined_sub_queries`` is
    contradictory — ``breakdown_only`` will be ignored with a warning.
```

---

## 11. Complete List of Files to Modify

| # | File | Change |
|---|------|--------|
| 1 | `breakdown_then_aggregate_inferencer.py` | Add 1 attrib, 1 helper method, update `_infer`, `_ainfer`, docstring |

**One file only.**

---

## 12. Backward Compatibility

| Existing usage | Impact |
|---|---|
| Any BTA without `predefined_sub_queries` | ✅ Zero change — `None` by default, all existing paths taken |
| `breakdown_inferencer=None` with `predefined_sub_queries=None` | ⚠️ Improves from silent `AttributeError` → clear `ValueError` with message |
| YAML configs without `predefined_sub_queries` key | ✅ No change |
| `max_breakdown` / `max_concurrency` without new field | ✅ No change |

---

## 13. Usage Examples

### Example A — Predefined list (no LLM breakdown):

```python
bta = BreakdownThenAggregateInferencer(
    predefined_sub_queries=[
        "Research PM frameworks (PMI, SAFe, Agile)",
        "Research cross-functional collaboration patterns",
        "Research success metrics and KPIs for PMs",
    ],
    worker_factory=lambda sub_query, index: ResearchWorker(...),
    aggregator_inferencer=SynthesisInferencer(...),
    max_concurrency=3,
)
result = await bta.ainfer("Research the Program Manager role")
```

### Example B — Auto-repeat single query (parallel sampling / diverse perspectives):

```python
bta = BreakdownThenAggregateInferencer(
    predefined_sub_queries="Research Program Manager role responsibilities",
    max_breakdown=5,   # → 5 workers all get the same query
    worker_factory=functools.partial(ResearchWorker),
    aggregator_inferencer=VotingAggregator(...),
)
result = await bta.ainfer("Research PM role")
# 5 parallel workers, same query, aggregator synthesises diverse outputs
```

### Example C — Auto-repeat with max_concurrency fallback:

```python
bta = BreakdownThenAggregateInferencer(
    predefined_sub_queries="What is the root cause of this bug?",
    # max_breakdown=None, so falls back to max_concurrency
    max_concurrency=4,   # → 4 workers
    worker_factory=functools.partial(DebugWorker),
    aggregator_inferencer=ConsensusAggregator(...),
)
```

### Example D — Structured heterogeneous list:

```python
bta = BreakdownThenAggregateInferencer(
    predefined_sub_queries=[
        {"query": "Analyse security", "args": {"task_type": "security_review"}},
        {"query": "Analyse performance", "args": {"task_type": "perf_review"}},
    ],
    task_type_arg_name="task_type",
    worker_factory={
        "security_review": SecurityWorker,
        "perf_review": PerfWorker,
    },
    aggregator_inferencer=FinalReportInferencer(...),
)
```

### Example E — YAML config:

```yaml
_target_: BreakdownThenAggregateInferencer
predefined_sub_queries:
  - "Research pricing strategy"
  - "Research competitive landscape"
  - "Research customer segments"
max_concurrency: 3
worker_factory:
  _target_: ...
aggregator_inferencer:
  _target_: ...
```

### Example F — YAML single-string auto-repeat:

```yaml
_target_: BreakdownThenAggregateInferencer
predefined_sub_queries: "Research Program Manager role"
max_breakdown: 5  # → 5 parallel workers
worker_factory:
  _target_: ...
```

---

## 14. Implementation Order

```
Step 1 — Add attrib
    predefined_sub_queries: Optional[Union[str, List]] = attrib(default=None, kw_only=True)
    After disable_aggregator (line 180), before start_nodes.
    Ensure Union is imported from typing (already imported at line 14).

Step 2 — Add helper method _resolve_predefined_sub_queries()
    Insert near _load_breakdown_checkpoint() / _save_breakdown_checkpoint() (lines 707–753).
    Private, well-documented, handles str / list / unexpected types.

Step 3 — Update _infer()
    Insert Step 0b block after the checkpoint resume block (after line 843).
    Add breakdown_only warning.
    Add null-guard ValueError before Step 1 (line 846).

Step 4 — Update _ainfer()
    Mirror of Step 3.
    CRITICAL: include use_async=True / try/finally guard around _build_diamond_graph
    (copy the exact pattern from lines 970–980).
    Add breakdown_only warning.
    Add null-guard ValueError before Step 1 (line 914).
    Include enable_checkpoint_results_review block (lines 988–1001) AFTER WorkGraph._arun
    — rerun correctly re-uses predefined_sub_queries via recursive _ainfer call.
    Skip enable_checkpoint_sub_query_selection (add comment explaining why).

Step 5 — Update class docstring
    Add "Predefined sub-queries mode:" section after "Concurrency control:" block.

Step 6 — Smoke tests
    A: predefined list → workers get correct queries, no LLM call
    B: single string, max_breakdown=5 → 5 workers with same query
    C: single string, max_breakdown=None, max_concurrency=3 → 3 workers
    D: single string, both None → 1 worker (fallback)
    E: default None → existing behaviour unchanged
    F: checkpoint + predefined → checkpoint wins
    G: predefined + breakdown_only=True → warning logged, proceeds normally
    H: predefined_sub_queries=None, breakdown_inferencer=None → clear ValueError
    I: predefined + enable_checkpoint_results_review=True → results review runs, rerun re-uses predefined_sub_queries
    J: predefined + enable_checkpoint_sub_query_selection=True → selection step skipped (not interactive trimming)
```

---

## 15. Critical-Thinking Double-Check

| Risk | Analysis | Verdict |
|---|---|---|
| **Checkpoint priority** | Step 0 runs before Step 0b. A saved checkpoint overrides `predefined_sub_queries`. Correct — checkpoint is explicit opt-in to resume saved work, always wins. | ✅ Correct |
| **`breakdown_only` + predefined** | `breakdown_only` checks for a `raw_output` to return — there is none in predefined mode. Must warn and ignore `breakdown_only`. Checked in both `_infer` and `_ainfer`. | ✅ Handled |
| **`use_async` flag in `_ainfer`** | Without `use_async=True`, `_build_diamond_graph` builds *sync* worker closures even in `_ainfer`. This is already the pattern at lines 897–907 and 970–980 — must copy exactly. Plan B's pseudocode omitted this — would be a silent bug. | ✅ Must include |
| **`max_breakdown or max_concurrency or 1` semantics** | `max_breakdown` used as target count for single-string case. Semantically dual-use but pragmatically the most natural existing knob. The `or 1` fallback prevents silent surprises (no magic `3`). | ✅ Acceptable |
| **`Union` import** | `Union` is already imported at line 14: `from typing import Any, Callable, Dict, List, Optional, Union`. No import change needed. | ✅ Safe |
| **`kw_only=True`** | Required to avoid `TypeError: non-keyword-only argument follows keyword-only argument` from `attrs` since there are already `kw_only=True` attribs above. Consistent with other optional attribs like `breakdown_format`, `task_type_arg_name`. | ✅ Required |
| **Empty list `[]`** | `_resolve_predefined_sub_queries` returns `[]`. After cap, still `[]`. The `if not sub_queries: return ""` guard handles it. Return `""` matches checkpoint resume path at line 832. | ✅ Consistent |
| **Type coercion for unexpected types** | Log a warning and coerce to `[str(psq)]`. Safe fallback, doesn't crash. | ✅ Safe |
| **No breakdown checkpoint saved** | When `predefined_sub_queries` is set, `_save_breakdown_checkpoint` is NOT called. Correct — the predefined list is stable/deterministic from the caller's side, not an LLM output. | ✅ Correct |
| **`breakdown_inferencer` null-guard** | Replaces silent `AttributeError` at line 846 with clear `ValueError`. Technically a breaking change but strictly better — no one depends on `AttributeError` semantics. | ✅ Improvement |
| **`_infer` missing `breakdown_only` check** | Pre-existing asymmetry (only `_ainfer` has it at line 947). Our change adds a warning for `breakdown_only + predefined` in both — we do NOT fix the pre-existing asymmetry (out of scope, avoid unrelated changes). | ✅ Scoped correctly |
| **Thread safety of `use_async` mutation** | Existing concern (flag is set on `self`). Step 0b copies the same `try/finally` pattern from checkpoint resume — no new risk. | ✅ Same as existing |
| **YAML `kw_only` compatibility** | Hydra/OmegaConf passes all fields as kwargs — `kw_only=True` is transparent. Consistent with how `breakdown_format`, `task_type_arg_name`, `expand_todos_to_workers` are already handled. | ✅ Compatible |
| **`enable_checkpoint_results_review` in predefined path** | The predefined path in `_ainfer` must include the interactive results review block (lines 988–1001). A rerun correctly re-uses `predefined_sub_queries`. Skipping it would silently break the interactive rerun feature. `_infer` has no such block (pre-existing) — no change needed there. | ✅ Included in Step 0b |
| **`enable_checkpoint_sub_query_selection` in predefined path** | Must be SKIPPED in predefined path — user explicitly provided their sub-queries; interactive trimming would be contradictory and confusing. A comment in the code makes this intentional. | ✅ Skipped with comment |
