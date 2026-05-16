# MFDual Dispatch Audit Analysis

## Summary
**VERDICT: This is a BUG. Audit entries are missing for rounds where the reviewer doesn't change.**

The dispatch audit only fires when `self.review_inferencer is not prev_reviewer` (line 625), which means:
- ✅ Round 1: Audit FIRES (None → some_inferencer)
- ❌ Round 2+: Audit SKIPPED if same reviewer (inferencer → same_inferencer)

---

## Detailed Findings

### A. What is `prev_reviewer` set to BEFORE `_select_reviewer_and_fixer` is called?

**Line 608** (in `_step_propose_impl`):
```python
prev_reviewer = self.review_inferencer
```

**Trace:**
- On **Round 1**: `self.review_inferencer` starts at its initial value from DualInferencer class definition (line 158 in dual_inferencer.py): `review_inferencer: InferencerBase = attrib(default=None)`
  - So `prev_reviewer = None`
  
- On **Round 2+**: `self.review_inferencer` is whatever was set in the previous round's `_select_reviewer_and_fixer()` call
  - So `prev_reviewer` holds the **previous round's reviewer** (persistent across rounds)

**Key Finding:** `prev_reviewer` is the previous round's reviewer, NOT just an initial value. It persists because `self.review_inferencer` is mutated in place by `_select_reviewer_and_fixer()`.

---

### B. Round 1 Behavior

**Expected:**
- `prev_reviewer = None` (from line 608)
- `_select_reviewer_and_fixer()` called (line 611)
- Sets `self.review_inferencer` to some flow inferencer via one of these paths (lines 415-517):
  1. LLM-driven alias resolution (line 477)
  2. `reviewer_match_second` → runner-up (line 484)
  3. `review_default` → rule-based dispatch (lines 492-516)

- Line 625 condition: `self.review_inferencer is not prev_reviewer`
  - Compares: (some_inferencer) is not (None) → **TRUE**
  - **Audit FIRES ✓**

---

### C. Round 2+ Behavior (The Bug)

**Scenario: If same reviewer serves consecutive rounds**

Example:
- Round 1: Winner = flow_0, reviewer = flow_1 (set via `review_default` or `reviewer_match_second`)
  - `prev_reviewer = None`
  - After `_select_reviewer_and_fixer()`: `self.review_inferencer = flow_1_inferencer_object`
  - Condition: flow_1 is not None → **TRUE** → audit fires ✓

- Round 2: Winner = flow_0 again, reviewer = flow_1 (same logic, same winner → same reviewer)
  - `prev_reviewer = flow_1_inferencer_object` (captured at line 608)
  - After `_select_reviewer_and_fixer()`: `self.review_inferencer = flow_1_inferencer_object` (same object)
  - Condition: flow_1 is not flow_1 → **FALSE** → **audit SKIPPED ❌**

- Round 3: Winner = flow_1, reviewer = flow_0 (different winner → different reviewer)
  - `prev_reviewer = flow_1_inferencer_object` (captured at line 608)
  - After `_select_reviewer_and_fixer()`: `self.review_inferencer = flow_0_inferencer_object`
  - Condition: flow_0 is not flow_1 → **TRUE** → audit fires ✓

**Pattern:** Audit only fires when the reviewer **changes**, not every round.

---

### D. Is This Intended Behavior?

**NO, this is a BUG.** 

Evidence:
1. **The dispatch state is being captured every round** (lines 621-624):
   ```python
   dispatch_extra = {"mfdual_dispatch": {
       "winner_idx": mfi._last_winner_idx,
       "ranking": mfi._last_ranking,
   }}
   ```
   This is computed fresh each round, but only recorded when reviewer changes.

2. **Round audit is meant to be comprehensive** (DualInferencer docstring, line 164):
   ```python
   enable_round_audit: bool = attrib(default=True)
   """Emits per-round audit: outputs/round_log.jsonl + children/round_NN/ nav links."""
   ```
   The intent is to emit an entry **per round**, not per reviewer-change event.

3. **The `_record_round_audit()` method accepts `round_idx`** (line 448), implying it's meant to be called once per round.

4. **Comparison with review/fix dispatch:** The code only checks reviewer, not fixer. If `fixer_match_winner=True` and the winner stays the same, the fixer audit entry is also skipped—but this is also suspicious.

---

### E. Is `prev_reviewer` reset between propose/review/fix cycles?

**NO, it is NOT reset.** 

**Trace:**
- `prev_reviewer` is a **local variable** captured at line 608, fresh for each `_step_propose_impl()` call
- But `self.review_inferencer` is **mutated in place** by `_select_reviewer_and_fixer()` (line 611)
- This mutation **persists** across rounds because `self.review_inferencer` is an instance attribute
- The next round's `prev_reviewer` (line 608) will read the updated `self.review_inferencer`

So the flow is:
```
Round 1: prev_reviewer = None → _select... → self.review_inferencer = flow_1
Round 2: prev_reviewer = flow_1 → _select... → self.review_inferencer = flow_1
Round 3: prev_reviewer = flow_1 → _select... → self.review_inferencer = flow_0
```

**Not reset between cycles.** Persistent across rounds as expected.

---

### F. Round Index and `state['total_iterations']` Initialization

**Line 626:**
```python
round_idx = (self._state.get("total_iterations", 0) + 1) if isinstance(getattr(self, "_state", None), dict) else 1
```

**Analysis:**
- `.get("total_iterations", 0)` returns 0 if key doesn't exist
- `0 + 1 = 1` for the first call
- This appears **correct** for round numbering

**However, where is `total_iterations` set?** 
- The code uses `.get(..., 0)` as a fallback, suggesting initialization elsewhere
- The condition `isinstance(getattr(self, "_state", None), dict)` is defensive, defaulting to `round_idx = 1` if `_state` is missing
- This looks like **defensive programming**, but the actual source of `total_iterations` wasn't traced in this analysis

---

## Code References

| Item | File | Lines |
|------|------|-------|
| Audit condition (bug location) | multi_flow_dual_inferencer.py | 625 |
| `prev_reviewer` capture | multi_flow_dual_inferencer.py | 608 |
| `_select_reviewer_and_fixer()` | multi_flow_dual_inferencer.py | 415-517 |
| Initial value of `review_inferencer` | dual_inferencer.py | 158 |
| `_record_round_audit()` | dual_inferencer.py | 448-487 |
| `enable_round_audit` doc | dual_inferencer.py | 163-164 |

---

## Recommended Fix

Change line 625 from:
```python
if self.review_inferencer is not prev_reviewer:
```

To something like:
```python
if self.review_inferencer is not None:  # or always audit
```

Or add a flag to control whether to audit only on changes or every round. The current condition loses audit data whenever a reviewer is reused across rounds, which is likely common in multi-flow scenarios with 2-3 reviewers.

---

## Impact Assessment

**Severity: MEDIUM**

- ✅ Doesn't break functionality
- ❌ Causes silent data loss in audit logs
- ❌ Makes it hard to trace dispatch decisions across all rounds
- ✅ Only affects observability/debugging, not runtime behavior
- ⚠️ Especially problematic if you want to correlate "which reviewer handled which winner" across a full multi-round consensus loop
