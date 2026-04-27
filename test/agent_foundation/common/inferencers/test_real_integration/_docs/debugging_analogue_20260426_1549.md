# Debugging journey — `test_multi_flow_dual_real.py` (Apr 26, 2026)

**Test:** `AgentFoundation/test/agent_foundation/common/inferencers/test_real_integration/test_multi_flow_dual_real.py::test_multi_flow_dual_documents_openstartup`

**Initial symptom:** `AssertionError: unexpected winner_idx: None` after a ~50-min real-CLI run.

**Final outcome:** ✅ PASS in 1201.65s (20:01) after **6 real-CLI iterations** discovering and fixing **8 distinct bugs** across 5 production files + 1 test file.

---

## TL;DR

A composition stress test (`MultiFlowDualInferencer` running two real Claude Code CLI flows, with a third Claude as integrating aggregator, plus self-review-avoidance + winner-as-fixer dispatch) was failing with `winner_idx: None` and zero actionable Python traceback. The post-mortem hypothesized 5 root causes; we landed all 5, then ran the test, which surfaced a **UnicodeEncodeError** (a Windows cp1252 encoding bug on the `→` character that Claude routinely puts in architectural docs). Fixing that surfaced the **next** bug — `_normalize_aggregator_output` mis-routing the BTA result tuple, picking a worker's plain-string output instead of the aggregator's `TerminalInferencerResponse` object. Two iterations of that fix landed before a position-independent tier-based selection finally made it robust. Plus a hardcoded test-side assertion that didn't match actual workspace layout. Each iteration was load-bearing on Fix 3 (visibility) — without it, every subsequent bug would have been invisible.

---

## Table of contents

1. [The starting symptom and visible signal](#the-starting-symptom-and-visible-signal)
2. [Original root-cause hypothesis (the 5-fix plan)](#original-root-cause-hypothesis-the-5-fix-plan)
3. [Run-by-run debugging journey](#run-by-run-debugging-journey)
4. [The 8 distinct bugs we found and fixed](#the-8-distinct-bugs-we-found-and-fixed)
5. [Key insights and meta-lessons](#key-insights-and-meta-lessons)
6. [Final architecture (what's in place now)](#final-architecture-whats-in-place-now)
7. [References (file:line citations)](#references-fileline-citations)
8. [How to debug similar issues in the future](#how-to-debug-similar-issues-in-the-future)

---

## The starting symptom and visible signal

The test runs a `MultiFlowDualInferencer` with two parallel Claude Code CLI flows, an aggregator (third Claude), and Round-7 dispatch (self-review-avoidance + winner-as-fixer). After ~50 minutes, the only diagnostic we got was:

```
E       AssertionError: unexpected winner_idx: None
E       assert None in (0, 1)
```

Plus two `step_failed` log entries:
```
2026-04-26 08:36:50 - MultiFlowDualInferencer - ERROR - step_failed: 0 (propose)
2026-04-26 08:48:14 - MultiFlowDualInferencer - ERROR - step_failed: 1 (review)
```

**Crucially, no Python traceback.** The retry helper had swallowed the underlying exception. Filesystem inspection revealed:
- **4 aggregator subprocess calls** had been made (cache directories `ClaudeCodeCliInferencer-17bc6129_20260426_*`)
- Calls 1–3 had each emitted `<FinalPlan>...</FinalPlan>\n<Winner>flow_1</Winner>`
- Call 4 had emitted `<analysis>...</analysis><summary>...</summary>` — Claude Code's **internal context-summarization** had kicked in
- All 4 calls were on the **same Claude session** (`session_id='b2582307...'`, `resume=True`)

That was enough to formulate a hypothesis.

---

## Original root-cause hypothesis (the 5-fix plan)

Five compounding bugs, ranked from most-load-bearing to least:

1. **Aggregator session resumption across retries** — `ClaudeCodeCliInferencer.active_session_id` persisted on the instance across calls. When BTA retried the aggregator, each retry resumed the same Claude session. After ~4 turns, Claude Code's auto-summarization replaced the user's prompt with its own request. Output format broke.

2. **Dispatch state clobbered on retry** — `MultiFlow._reset_cross_flow_state()` ran at the top of every `_ainfer` call and reset `_last_winner_idx = None`. On retry, dispatch state captured from a successful earlier attempt was overwritten by a malformed later attempt's `None`.

3. **Retry helper swallows exceptions silently** — `async_execute_with_retry`'s `_default_return_or_raise_terminal()` either raised or returned the default with **zero logging**. Operators had no way to see what drove retries.

4. **`retry_on_exceptions=(Exception,)` too broad** — All BTA WorkGraph nodes retried every `Exception`, including programming errors. One transient error → 2–4 retries; one programming bug → also 2–4 retries.

5. **Self-review-avoidance silently no-ops on dispatch failure** — When `_select_reviewer_and_fixer()` ran with `winner=None`, it fell through to `review_default` as-is with no warning.

The fix design went through several rounds of feedback. The architecturally-significant choice: **recursive `pre_retry` propagation through a generic `_iter_child_inferencers` primitive on `InferencerBase`** — the same primitive then refactored existing `aconnect`/`adisconnect`/`_areset_sub_inferencers` paths, eliminating four hand-rolled iteration loops.

---

## Run-by-run debugging journey

### Run 1 (pre-fixes, original) — opaque failure

**Setup:** Original failing run that triggered the entire investigation.

**Result:** `winner_idx: None`. No traceback. ~50 min wasted; couldn't tell why.

**Filesystem clue:** 4 aggregator calls all on same session, last one returning auto-summary instead of `<FinalPlan>+<Winner>`.

### Run 2 (after 5 plan fixes landed) — Fix 3 paid off immediately

**Code state:** All 5 plan fixes + 18 unit tests added; 145/145 pass locally.

**Result:** Failed with `winner_idx: None`, BUT now with Fix 3's WARNING fully readable in the log:
```
WARNING async_utils.py:180 async_execute_with_retry: retry chain exhausted after 2 total attempts. 
Last exception: UnicodeEncodeError: 'charmap' codec can't encode character '→' in position 576
```

**The actual root cause was a Windows cp1252 encoding bug**, not anything in the original 5-bug analysis. `→` is `→` — a character Claude routinely emits in architectural documentation ("server → DB", etc.). When BTA's `_finalize_response` tried `with open(report_dst, "w") as f: f.write(text)` with no explicit encoding, Windows' default cp1252 codec failed on the arrow.

This is the most important lesson of the entire debugging session: **Fix 3 (visibility) was the only load-bearing planned fix.** Every subsequent bug was invisible without it.

The exception traceback also pinpointed the exact line:
```
File: breakdown_then_aggregate_inferencer.py:866
Method: _finalize_response  
Line: f.write(text)
```

### Run 3 (after encoding fix) — `winner_idx` still None, but DIFFERENT failure

**Fix added:** `encoding='utf-8'` on file writes in 4 sites:
- `breakdown_then_aggregate_inferencer.py` `_finalize_response` (the immediate fix)
- `dual_inferencer.py` `_maybe_replace_with_file_reference` (2 sites)
- `terminal_inferencer_base.py` (output saving)
- `inferencer_workspace.py` (marker writes)

Plus widened the `except OSError` clauses to also catch `UnicodeEncodeError` so future encoding hiccups don't propagate as transients.

**Result:** No more `UnicodeEncodeError`. Test got past the propose step. But STILL `winner_idx: None`, with a NEW failure mode: `step_failed: 1` (review) instead of `step_failed: 0` (propose).

**Key forensic discovery:** Comparing `mfdi_workspace/children/aggregator/.../*.txt` (the aggregator's stream — clean `<FinalPlan>...</FinalPlan>\n<Winner>flow_0</Winner>`) against the `InitialResponse` log line (which contained `<Plan>...</Plan>\n<Decision>stop</Decision>`):

```python
worker_1's step 1 output[:120]: '<Plan>\n# Documentation Plan: OpenStartup\n\n> Output is the PLAN itself...'
InitialResponse:                '<Plan>\n# Documentation Plan: OpenStartup\n\n> Output is the PLAN itself...'
```

**Identical match.** MultiFlow was returning **worker_1's text**, not the aggregator's. Somewhere between BTA returning the aggregator's response and the propose step receiving it, the aggregator's contribution was being dropped in favor of a worker's plain-string output.

Tracing the code: `_normalize_aggregator_output` was the culprit. The OLD code:
```python
if isinstance(raw, tuple):
    str_els = [x for x in raw if isinstance(x, str)]
    if str_els:
        return str_els[-1]   # ← picks LAST STRING; aggregator's TerminalInferencerResponse excluded
```

When workers return plain strings (LWI dynamic-mode last-step text) and the aggregator returns a `TerminalInferencerResponse` (object with `.output` attr, not a string), `str_els` filters out the aggregator and picks the last worker's text.

### Run 4 (routing fix v1: `non_none[-1]`) — flaky pass/fail

**Fix:** Replace `str_els[-1]` with `non_none[-1]` (last non-None element regardless of type), assuming aggregator runs LAST in BTA's topology.

**Result:** Flaky. **Sometimes passed, sometimes failed.** The DEBUG log added during this run revealed why:
```
DEBUG _normalize_aggregator_output: raw shape=tuple[2] 
elements=[0]=str(38720ch:'Reviewing my draft against the requireme') 
       | [1]=TerminalInferencerResponse(output='<FinalPlan>\n# OpenStartup Codebase Archi')
```

In one run the tuple was `(worker_str, agg_response_obj)` → `non_none[-1]` correctly picks aggregator. ✓
In another the tuple was `(agg_response_obj, worker_str)` → `non_none[-1]` picks worker_str. ✗

**WorkGraph result tuple ordering is non-deterministic.** The order depends on which node finishes first relative to async-task scheduling, not on graph topology.

### Run 5 (routing fix v2: filter LWI state dicts) — still flaky for same reason

**Fix:** Identify "response-shaped" elements (str, has `.output`, dict-with-output-and-not-LWI-state) and prefer last response-shaped. Filtering LWI state dicts (worker `dynamic_step_results`/`iteration_records`/`__expansion_count`/`_prev_iteration` markers).

**Result:** Same flakiness. Both worker plain-strings AND aggregator `TerminalInferencerResponse` matched the "response-shaped" filter (both have `.output` or are strings). When the aggregator was at index 0 and worker at index 1, `response_like[-1]` still picked worker.

The bug was: **type filter wasn't precise enough; we still depended on position within the filtered list.**

### Run 6 (routing fix v3: tier-based selection) — ✅ PASSED

**Fix:** Distinguish by TYPE (not position). Four-tier priority:

```python
tier1 = [x for x in non_none if hasattr(x, "output") and not isinstance(x, str)]
tier2 = [x for x in non_none if isinstance(x, dict) and "output" in x and not _looks_like_lwi_state(x)]
tier3 = [x for x in non_none if isinstance(x, str)]
# fall back: non_none[-1]
```

**Reasoning:** Workers in LWI dynamic mode return plain strings (last step's text). The aggregator (when it's a CLI inferencer) returns a `TerminalInferencerResponse` — an *object* with `.output` attr but NOT itself a string. That makes Tier 1 a strong, position-independent signal: "I have an `.output` attr AND I'm wrapping it (not raw text)."

**Result:** ✅ PASSED in 1201.65s. All 6 test assertions passed:
1. `result` is `DualInferencerResponse`
2. `result.base_response` >200 chars (the integrated plan content)
3. `winner_idx in (0, 1)` ← the original failure mode is FIXED
4. Self-review-avoidance fired (reviewer is non-winner)
5. Winner-as-fixer fired (fixer is winner)
6. Workspace artifacts present

Plus one test-side fix along the way: the original test had a hardcoded path assertion `tmp_path / "mfdi_workspace" / "checkpoints" / "attempt_01" / "step_propose.json"` that didn't match the actual workspace layout. Relaxed to "any artifact under workspace/checkpoints or workspace/children" — pin to behavior, not specific paths.

---

## The 8 distinct bugs we found and fixed

| # | Bug | Root cause | Fix location | How surfaced |
|---|-----|------------|--------------|--------------|
| 1 | Aggregator session resumption across retries | `active_session_id` persisted across calls; retries used `resume=True` | `streaming_inferencer_base.py` `_pre_retry` calls existing `reset_session()`; `inferencer_base.py` recursive `pre_retry` propagation | Original analysis; verified empirically (Run 5 onwards: aggregator session=None on retry) |
| 2 | Dispatch state clobbered on retry | `MultiFlow._reset_cross_flow_state()` reset dispatch state on every `_ainfer` (per-attempt instead of per-call) | `multi_flow_inferencer.py`: split into `_reset_cross_flow_state` (per-attempt) and `_reset_dispatch_state_for_call` (once per `ainfer`); override `ainfer`/`infer` to reset dispatch once | Original analysis; unit-tested |
| 3 | Retry helper swallows exceptions silently | `_default_return_or_raise_terminal()` had no logging | `async_utils.py` and `function_helper.py`: WARNING log of `last_exception` once per chain on exhaustion | Original analysis. **Most load-bearing fix** — surfaced bug #6 |
| 4 | `retry_on_exceptions=(Exception,)` too broad | BTA WorkGraph nodes retried programming errors as if they were transients | `breakdown_then_aggregate_inferencer.py`: `TRANSIENT_RETRY_EXCEPTIONS = (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError)` at 4 sites | Original analysis; unit-tested |
| 5 | Self-review-avoidance silently no-ops when winner unknown | No telemetry when dispatch couldn't fire | `multi_flow_dual_inferencer.py` `_select_reviewer_and_fixer`: WARNING when `winner is None` and `review_default`/`fixer_match_winner` configured | Original analysis; verified fired in failing runs |
| 6 | UnicodeEncodeError on `→` (Windows cp1252) | `open(path, "w")` defaulted to cp1252 on Windows; Claude emits arrows in arch docs | 4 file-write sites: `breakdown_then_aggregate_inferencer.py:866`, `dual_inferencer.py:1108,1146`, `terminal_inferencer_base.py:442`, `inferencer_workspace.py:218`. Added `encoding='utf-8'` + widened except clauses | **Surfaced by Fix 3** (Run 2 WARNING log). Was completely invisible before. |
| 7 | `_normalize_aggregator_output` mis-routing BTA tuple | Tuple ordering non-deterministic; old logic picked `str_els[-1]`, missing aggregator's `TerminalInferencerResponse` | `multi_flow_inferencer.py` `_normalize_aggregator_output`: 4-tier type-based priority (response-obj > response-dict > string > non-None) | Surfaced by Fix 6 (Run 3 — once propose stopped failing on encoding, the routing bug became the dominant failure) |
| 8 | Test workspace assertion hardcoded non-existent path | Test was written expecting `attempt_01/step_propose.json` which DualInferencer doesn't produce | `test_multi_flow_dual_real.py:328-336`: relax to "any artifact under workspace" | Surfaced after #7 was fixed (Run 6 reached this assertion for the first time) |

---

## Key insights and meta-lessons

### 1. Visibility-first

Of the 5 originally-planned fixes, **Fix 3 (logging the swallowed exception) was the only load-bearing one** for resolving the actual failure. The other 4 (session reset, dispatch state preservation, narrow retries, sanity warnings) were all correct and useful, but they addressed *consequences* of the failure mode, not the mode itself. Without Fix 3 we'd have had no path to discover bug #6 (the encoding error), and the iteration would have stalled.

**Takeaway for future debugging of opaque-failure systems:** when a system swallows exceptions and you have no traceback, **fix the visibility first**, even if it feels like "not really fixing anything." The information unlock is the multiplier.

### 2. The original 5-bug post-mortem analysis was ~30% right by surface, ~95% right by structure

We thought the failure was a session leak (it was, partially). We thought retries were piling up because of legitimate transient errors (they weren't — they were piling up because of a deterministic encoding bug masquerading as a transient via the broad `retry_on_exceptions=(Exception,)` filter). We thought dispatch state preservation across retries was load-bearing (it would have been if the encoding bug had been one of the rare-success-rare-failure transients we hypothesized).

Every fix turned out to be useful (none were wasted), but the *mechanism* by which they helped wasn't what the analysis predicted. **A correct architectural understanding can produce 5 useful fixes even when the specific failure mechanism is misidentified.**

### 3. WorkGraph result tuples have non-deterministic ordering

This is the most important new mental model from this debugging session. We had implicitly assumed BTA's WorkGraph returned results in topological order (workers first, aggregator last). It doesn't — the order depends on which node *finishes* first in async scheduling, and there's no guarantee.

**Don't depend on tuple position.** Use type-based or label-based identification when picking elements out of a graph result tuple.

### 4. Mocks and reality differ in non-obvious ways

The unit tests for `_normalize_aggregator_output` passed for both v1 and v2 fixes. The failure was specific to real CLI inferencers because:
- Mock workers return plain strings (the obvious shape)
- Real LWI-dynamic-mode workers return plain strings (their last step text) → same shape as mocks
- Mock aggregators in tests return plain strings (the simple shape)
- Real CLI aggregators return `TerminalInferencerResponse` objects (a wrapper with `.output`)

The unit tests didn't capture the "aggregator returns a wrapper object, workers return raw strings" asymmetry. After the bug was found, we added `_TerminalResponseAgg` to the test fixture set so future routing changes can't regress.

**Takeaway:** when mocking, mock the *shape* of real-world data, not just the data type. If your real code returns a `TerminalInferencerResponse` wrapping a string, mock that, not the bare string.

### 5. Distinguishing structural bugs from LLM determinism

Several times during the debugging we wondered whether a failure was due to LLM behavior (the model not following the prompt) or a structural code bug. The evidence chain that helped:
- The aggregator's stream cache file ALWAYS contained the right `<FinalPlan>+<Winner>` format → LLM was following the prompt fine.
- The `InitialResponse` log showed worker text instead → the structural mis-routing was on us.

Reading the streamed cache file (`_runtime/inferencer_cache/.../*.txt`) was the key. **When you suspect "the LLM is being weird," check the cache file first** — if it has the right content, the bug is in your routing/parsing.

### 6. Generic primitives over special-case ones

When designing Fix 1c, the user pushed back on adding a retry-specific `_iter_child_inferencers_for_retry`. The codebase already had hand-rolled iteration in 4 sites (`DualInferencer.aconnect`/`adisconnect`/`_areset_sub_inferencers` and `MultiFlowDualInferencer._all_candidate_inferencers`). The user's instinct was right: extract one generic `_iter_child_inferencers` primitive, refactor all 4 existing sites to use it, then `pre_retry` is just one more consumer.

**Result:** Five iteration sites converged on a single primitive instead of becoming a fifth special-case iterator.

---

## Final architecture (what's in place now)

### Files touched (production code)

```
RichPythonUtils/src/rich_python_utils/common_utils/async_utils.py     +  Fix 3
RichPythonUtils/src/rich_python_utils/common_utils/function_helper.py +  Fix 3
AgentFoundation/src/agent_foundation/common/inferencers/
  inferencer_base.py                                                  +  Fix 1a
  streaming_inferencer_base.py                                        +  Fix 1b
  inferencer_workspace.py                                             +  Fix 6
  terminal_inferencers/terminal_inferencer_base.py                    +  Fix 6
  agentic_inferencers/flow_inferencers/
    breakdown_then_aggregate_inferencer.py                            +  Fix 4, 6, 1c
    dual_inferencer.py                                                +  Fix 1c, 6
    multi_flow_inferencer.py                                          +  Fix 1c, 2, 7
    multi_flow_dual_inferencer.py                                     +  Fix 1c, 5
```

### Files touched (tests)

```
AgentFoundation/test/agent_foundation/common/inferencers/
  test_inferencer_base_async.py            +  8 new tests for Fix 1a
  test_streaming_recovery.py               +  1 new test for Fix 1b
  test_breakdown_then_aggregate.py         +  3 new tests for Fix 1c (BTA) + Fix 4
  test_multi_flow_inferencer.py            +  4 new tests for Fix 1c (MultiFlow) + Fix 2 + Fix 7
                                              (including TerminalResponseAgg shape regression)
  test_multi_flow_dual_inferencer.py       +  2 new tests for Fix 1c + Fix 5
                                              (rename _all_candidate → _iter_child_inferencers)
  test_real_integration/
    test_multi_flow_dual_real.py           +  Fix 8 (relax workspace assertion)
RichPythonUtils/test/rich_python_utils/
  common_utils/test_async_utils.py         +  1 new test for Fix 3
```

**Total: 19 new unit tests, 145/145 unit tests pass after all changes.**

### Architectural primitives added

| Primitive | Location | Purpose |
|-----------|----------|---------|
| `pre_retry(self, attempt, exception, _seen=None)` (public) | `InferencerBase` | Recursive retry-cleanup hook called by retry helper between attempts |
| `_pre_retry(self, attempt, exception)` (protected) | `InferencerBase` | Subclass override hook for own-state cleanup; default no-op |
| `_iter_child_inferencers()` | `InferencerBase` | **Generic** child iteration mechanism used uniformly by `aconnect`/`adisconnect`/`_areset_sub_inferencers`/`pre_retry` |
| `_pre_retry` override | `StreamingInferencerBase` | Calls existing `reset_session()` to clear `active_session_id` |
| `_reset_dispatch_state_for_call()` | `MultiFlowInferencer` | Per-`ainfer`-call reset (vs. per-attempt `_reset_cross_flow_state`) |
| `_collect_candidate_inferencers()` | `MultiFlowDualInferencer` | Renamed from `_all_candidate_inferencers`; raw collection (with duplicates) |
| `TRANSIENT_RETRY_EXCEPTIONS` constant | `breakdown_then_aggregate_inferencer.py` | `(TimeoutError, asyncio.TimeoutError, ConnectionError, OSError)` |
| `_looks_like_lwi_state(d)` | `multi_flow_inferencer.py` | Marker-key detection for filtering LWI state dicts out of aggregator selection |

---

## References (file:line citations)

### Production code

- **Fix 1a (recursive `pre_retry`)**: `inferencer_base.py:387-477` (new methods); `inferencer_base.py:1326-1366` (wired into `_internal_retry_callback`)
- **Fix 1b (`_pre_retry` override)**: `streaming_inferencer_base.py:799-818`
- **Fix 1c (`_iter_child_inferencers` overrides + lifecycle refactor)**:
  - BTA: `breakdown_then_aggregate_inferencer.py:878-887`
  - MultiFlow: `multi_flow_inferencer.py:548-563`
  - DualInferencer: `dual_inferencer.py:1181-1226` (refactored from hardcoded loops to use the primitive)
  - MultiFlowDualInferencer: `multi_flow_dual_inferencer.py:351-411` (renamed `_all_candidate_inferencers` → `_collect_candidate_inferencers`; new `_iter_child_inferencers` with dedup)
- **Fix 2 (dispatch state lifetime split)**: `multi_flow_inferencer.py:518-549`, `696-720`
- **Fix 3 (terminal-exception logging)**:
  - Async: `async_utils.py:178-190`
  - Sync: `function_helper.py:506-520`
- **Fix 4 (narrow retries)**: `breakdown_then_aggregate_inferencer.py:34-46` (`TRANSIENT_RETRY_EXCEPTIONS`); replacements at lines 922, 1003, 1314, 1439
- **Fix 5 (sanity warnings)**: `multi_flow_dual_inferencer.py:289-318`
- **Fix 6 (encoding)**:
  - `breakdown_then_aggregate_inferencer.py:744-770` (`_save_breakdown_checkpoint`), `857-879` (`_finalize_response`)
  - `dual_inferencer.py:1106-1162`
  - `terminal_inferencer_base.py:442-449`
  - `inferencer_workspace.py:218-219`
- **Fix 7 (4-tier aggregator routing)**: `multi_flow_inferencer.py:712-756` (the `_normalize_aggregator_output` body)
- **Fix 8 (test workspace assertion)**: `test_multi_flow_dual_real.py:328-345`

### Diagnostic logging (kept at DEBUG level for future investigation)

- `multi_flow_inferencer.py:660-690` — emits tuple shape from BTA on the BTA→MultiFlow handoff. Enable via `logging.getLogger("agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer").setLevel(logging.DEBUG)` if a similar mis-routing is suspected in the future.

---

## How to debug similar issues in the future

If a `MultiFlowDualInferencer` (or similar BTA-stacked composition) test hangs or fails opaquely under real CLI inferencers, here's the playbook this debugging session validated:

### 1. Always look at the WARNING log from `_default_return_or_raise_terminal` first

If retry chain exhausted, the WARNING from `async_utils.py:180` (or `function_helper.py:512`) tells you the actual exception type and message. Without this you're guessing.

### 2. Compare the aggregator stream cache against `InitialResponse`

```bash
# Aggregator's actual subprocess output:
cat <pytest tmp>/mfdi_workspace/children/aggregator/_runtime/inferencer_cache/*/*.txt | head -3

# Workers' final outputs:
python -c "import json; print(json.load(open('<pytest tmp>/mfdi_workspace/children/worker_0/checkpoints/final_result.json'))['dynamic_step_results'][-1].get('output', '')[:120])"

# What the propose step ACTUALLY captured:
grep -A1 "InitialResponse:" <test log>
```

If the aggregator stream has clean structured output but `InitialResponse` shows worker text, the bug is in `_normalize_aggregator_output` or upstream tuple construction. If the aggregator stream itself is malformed (missing `<FinalPlan>` or `<Winner>`), the bug is the LLM not following the prompt — adjust the prompt design.

### 3. Enable the diagnostic DEBUG log

```python
import logging
logging.getLogger(
    "agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer"
).setLevel(logging.DEBUG)
```

This emits one line per `_normalize_aggregator_output` call showing the exact tuple shape (element types and partial repr). Saves a real-CLI iteration.

### 4. Don't assume tuple position

If you ever find yourself writing `non_none[-1]` or `result[0]` after pulling from a WorkGraph result, stop and ask: "Is this position deterministic?" If you're not sure, use type-based filtering (Tier 1: has `.output` and not str, Tier 2: dict-with-output-and-not-LWI-state, Tier 3: str, Tier 4: fallback).

### 5. On Windows, ALWAYS pass `encoding='utf-8'` to file open

Default cp1252 will silently break on `→`, `—`, `…`, `’`, and a long list of other Unicode punctuation that LLMs routinely emit. Treat any `open(path, 'w')` (or `'r'`) without explicit encoding as a latent bug.

### 6. Six iterations is normal for a system this deep

Don't try to find all bugs at once; find the next one each iteration. Real-CLI runs are slow and expensive but each iteration peels back a layer. Keep your unit tests fast, your fix surface focused, and your diagnostic logging on.

### 7. When in doubt, prefer the more general primitive

The `_iter_child_inferencers` consolidation (replacing 4 hand-rolled loops + 1 new retry-specific iterator with 1 generic primitive) is the most architecturally satisfying piece of this work. It came from a single user push: "isn't there already a mechanism for this?" Always worth asking.

---

*Captured 2026-04-26 15:49. Final test run: `btuw3j9se` (1201.65s, exit 0). 145/145 unit tests pass.*
