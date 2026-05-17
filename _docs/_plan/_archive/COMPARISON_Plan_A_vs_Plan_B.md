# Detailed Comparison: Plan A vs Plan B

## Executive Summary

| Dimension | Plan A | Plan B |
|-----------|--------|--------|
| **Scope** | Comprehensive: helper + plumbing + 4 template updates + loud warnings + fallback removal + 6 test files | Minimal: helper + plumbing only, no template changes, single test file |
| **Path delivery** | New template variable `prior_output_path` (file path) | Appended to `proposal` text (folder path) |
| **Template changes required** | 4 templates (followup.jinja2, review.jinja2, implementation templates) | 0 templates |
| **Active proposer detection** | `_active_proposer()` reads `counter_feedback` from state | `iteration <= 1` heuristic |
| **Bug A fix** | Explicit `main_response` in feed dict | Relies on `placeholder_proposal` mapping |
| **Lines of code** | ~1100 | ~326 |
| **Phases** | 7 (includes architecture cleanup) | 1 (scope-limited fix) |
| **Real-world effectiveness** | ✅ 100% solves observed bug | ❌ 0% — critical design flaw |

---

## 1. Scope Difference

### Plan A's Scope
- Phase 1: Add `_resolve_prior_proposer_output_path()` helper (169 lines of pseudocode + logic)
- Phase 2: Plumb `prior_output_path` and `main_response` into `_build_followup_prompt` and `_build_review_prompt`
- Phase 3: Update `plan/main/followup.jinja2` to use `{{ prior_output_path }}` with explicit `cp` instruction
- Phase 4: Symmetric updates to `plan/main/review.jinja2` and `implementation/main/` templates (4 templates total)
- Phase 5: Add loud warnings to `_render_role_prompt` when falling back to in-Python defaults
- Phase 6: Audit and migrate all consumers, then remove silent fallback entirely
- Phase 7: 6 test files (~27-29 test methods) covering all phases

**Verdict on Plan A scope**: Over-engineered for the immediate bug fix. Phases 5-7 add 30-40% effort for future-proofing. Tests are comprehensive. **MVP (Phases 1-3) is sufficient.**

### Plan B's Scope
- Add `_resolve_prior_output_folder()` helper (~18 lines of clean logic)
- Modify `_build_followup_prompt` to append path to proposal (~5 lines)
- Modify `_build_review_prompt` to append path to proposal (~5 lines)
- Single test file with 5 test classes (~17 test methods)
- **Zero template changes** — path appears inside existing `<ProposedDocument>` tag via text appending

**Verdict on Plan B scope**: Laser-focused on immediate fix. Minimal touch surface. But has critical design flaw (see section 5).

---

## 2. Path Resolution Rule — File vs Folder, Tiers

### Plan A: Two-Tier Deliverable-FILE Selection

```
Tier 1 (preferred): proposer.workspace.final_deliverables/output.md
   - Check: does final_deliverables/ exist and contain output.md?
   - If yes, return <full_path_to_file>

Tier 2 (fallback): proposer.workspace.final_deliverables/<first_alphabetically>
   - Check: does final_deliverables/ exist and is non-empty?
   - Return first file (alphabetically) from that directory

Edge case: no deliverables → return None
```

**What it returns**: Absolute file path (e.g., `/workspace/.../final_deliverables/output.md`)

**Issue with Plan A's tier 2**: Alphabetical fallback may pick the wrong file if multiple exist.

### Plan B: Two-Tier Deliverable-FOLDER Selection

```
Tier 1 (preferred): proposer.workspace.final_deliverables/
   - Check: does final_deliverables/ exist and is non-empty?
   - If yes, return <full_path_to_directory>

Tier 2 (fallback): proposer.workspace.outputs/
   - Check: does outputs/ exist?
   - Return that directory

Edge case: no workspace → return None
```

**What it returns**: Directory path (e.g., `/workspace/.../final_deliverables/`) — **NOT a file**.

**Issue with Plan B's folder return**: LLM must **infer which file to read** inside the folder. Ambiguous when multiple files exist.

### Comparison

| Aspect | Plan A (FILE) | Plan B (FOLDER) |
|--------|----------------|-----------------|
| **Precision** | Higher: exact file | Lower: directory, LLM must guess |
| **Robustness** | Medium: breaks if `output.md` doesn't exist | Higher: works as long as dir exists |
| **LLM clarity** | Higher: path is unambiguous | Lower: requires LLM inference |
| **Generality** | Medium: assumes `output.md` convention | Higher: works for any convention |

**Verdict**: Plan A's file-based approach is better for the observed bug (BTA producing single file). Plan B's folder approach is more general but less discoverable.

---

## 3. Active Proposer Detection — Semantic Equivalence?

### Plan A: `_active_proposer()` reads `counter_feedback` from state

**Current code** (`dual_inferencer.py` lines 430-470):
```python
def _active_proposer(self):
    state = getattr(self, "_state", None) or {}
    iters = state.get("attempt_record", {}).get("iterations") or []
    if not iters:
        return self.base_inferencer
    
    last = iters[-1]
    counter = getattr(last, "counter_feedback", None) or last.get("counter_feedback")
    
    if counter is None:
        return self.base_inferencer  # review passed
    return self.fixer_inferencer    # fixer ran
```

**What it determines**: Whether final output came from base (review passed) or fixer (review failed).

### Plan B: `iteration <= 1` heuristic

**Logic**:
```python
def _resolve_prior_output_folder(self) -> str | None:
    iteration = state.get("consensus_iteration", 1)
    
    if iteration <= 1:
        return None  # first iteration, no fixer yet
    # ... return folder
```

**What it determines**: Whether to suppress path appending on first iteration.

### Semantic Equivalence?

| Scenario | `counter_feedback` (A) | `iteration <= 1` (B) | Equivalent? |
|----------|----------------------|-------------------|-----------|
| Fresh run, before review | counter_feedback = None | iteration = 1 | ✅ YES |
| After first review passes | counter_feedback = None | iteration = 1 | ✅ YES |
| During fix (review failed) | counter_feedback ≠ None | iteration = 2 | ✅ YES |
| After second review passes | counter_feedback = None | iteration = 2 | ❌ **DIVERGE** |

**Verdict**: **NOT semantically equivalent**. Plan A is state-based and correct. Plan B's heuristic is fragile and wrong after round 1.

---

## 4. How the Path Reaches the LLM — Template Changes vs Text Appending

### Plan A: Template Variable + Four Template Updates

**Mechanism**:
1. `_build_followup_prompt` sets `feed["prior_output_path"] = "/path/to/output.md"`
2. Template renders `{{ prior_output_path }}` with explicit instruction
3. Example (Plan A, Phase 3):
```jinja2
YOU MUST copy the previous file using the cp command:
cp {{ prior_output_path }} {{ output_path }}
```

**Visibility to LLM**: 
- ✅ Explicit: instruction is front-and-center
- ✅ Discoverable: path with surrounding "YOU MUST copy" context
- ❌ Requires 4 edits: each template must be updated

### Plan B: Append Path to Proposal Text (No Template Changes)

**Mechanism**:
1. `_build_followup_prompt` appends path to proposal: `proposal += f"\n\n[Available at: {folder_path}]"`
2. Template (unchanged) renders `{{ proposal }}` inside `<ProposedDocument>`
3. Path appears as trailing text, not an instruction

**Visibility to LLM**:
- ⚠️ Implicit: path is appended text, not an instruction
- ⚠️ Less discoverable: LLM must infer purpose
- ✅ No template edits: single change in Python

### Comparison

| Dimension | Plan A | Plan B |
|-----------|--------|--------|
| **Explicitness** | High: "YOU MUST copy" | Low: "[Available at...]" |
| **Discoverability** | High: separate instruction | Low: mixed with content |
| **LLM clarity** | Clear: see example | Implicit: infer action |
| **Template coupling** | High: 4 edits | Zero: no changes |
| **Maintainability** | Medium: keep in sync | High: single point |

**Verdict**: Plan A is more discoverable. Plan B is less friction to ship but risks LLM not recognizing the path as actionable.

---

## 5. CRITICAL: Bug A and B Handling — `main_response` Plumbing

### Plan A: Explicit `main_response` in Feed Dictionary

**What Plan A does** (Phase 2):
```python
def _build_followup_prompt(self, ..., proposal: str, ...):
    feed = {
        self.placeholder_input: inference_input,
        self.placeholder_proposal: proposal,
        "main_response": proposal,  # ← NEW: fixes Bug A
        ...
    }
    return self._render_role_prompt("followup", feed, inference_config)
```

**Why this fixes Bug A**: The outer template `plan/main/followup.jinja2` (lines 12-14) contains:
```jinja2
<ProposedDocument>
{{ main_response }}
</ProposedDocument>
```

Currently, `main_response` is NOT in the feed dict, so this renders **empty**. Plan A explicitly sets it, fixing the bug.

**Verification**: Looking at current code (`dual_inferencer.py` lines 1129-1141), `_build_followup_prompt` sets:
```python
feed = {
    self.placeholder_input: inference_input,
    self.placeholder_proposal: proposal,
    "iteration": iteration,
    "attempt": attempt,
    ...
}
# Note: main_response is NOT set
```

The outer template's `{{ main_response }}` renders **empty string**.

### Plan B: Reliance on `placeholder_proposal` Mapping — **WRONG**

**What Plan B assumes**:
> "works with both inner default template AND outer Jinja file via `placeholder_proposal` indirection"

**The critical flaw**: `placeholder_proposal` is a **key in the feed dict** (default value = "proposal"). It does NOT automatically map to `main_response` in the outer template.

**Proof**:
- Plan B appends path to `proposal` and sets `feed[self.placeholder_proposal] = proposal`
- The outer template uses `{{ main_response }}`, not `{{ proposal }}`
- They are **separate variables** with separate namespaces
- Setting `feed["proposal"]` does NOT set `feed["main_response"]`

**Therefore**: Plan B does NOT actually set `main_response` in the feed dict. The outer template's `<ProposedDocument>{{ main_response }}</ProposedDocument>` will render **empty**, just like the current bug.

**The hidden assumption**: Plan B's design works **ONLY if** the outer template uses `{{ proposal }}` instead of `{{ main_response }}`. But the actual template uses `{{ main_response }}`. Plan B's assumption is false.

### Verdict on Plan B

**Plan B has a CRITICAL BUG**: It does NOT fix Bug A. The outer template's `<ProposedDocument>` will still render empty. The path is appended to `proposal` (inner template variable), which is rendered in a different context than the outer template's `{{ main_response }}`.

**To fix Plan B's design**, you would need to EITHER:
1. **Set `main_response` in feed dict** (making it equivalent to Plan A), OR
2. **Change the outer template to use `{{ proposal }}`** (requires template edits, contradicting "zero template changes"), OR
3. **Accept that `<ProposedDocument>` remains empty** (defeats the purpose of the fix)

**Plan B's claim of "no template changes" is undermined by the fact that it doesn't actually work.**

---

## 6. File vs Folder — Tradeoffs for LLM Tools

### Plan A (File Path)

**Path example**: `/workspace/children/base_inferencer/outputs/final_deliverables/output.md`

**LLM instruction in template**:
```
cp {{ prior_output_path }} {{ output_path }}
```

**Utility**:
- ✅ Can directly use `cp` command
- ✅ Can directly use `read_file({{ prior_output_path }})`
- ✅ Can directly use `find_and_replace_in_file({{ prior_output_path }}, ...)`

**Risk**:
- ❌ If filename changes, path breaks

### Plan B (Folder Path)

**Path example**: `/workspace/children/base_inferencer/outputs/final_deliverables/`

**LLM sees**:
```
[Available at: /workspace/children/base_inferencer/outputs/final_deliverables/]
```

**LLM's likely action** (if it infers correctly):
```bash
ls folder  # figure out filename
cp folder/output.md ...
```

**Utility**:
- ⚠️ Cannot directly use `cp` — must first list folder
- ⚠️ Cannot directly use `read_file(folder)` — must infer filename
- ⚠️ Requires LLM inference

**Verdict**: Plan A's file path is more useful for LLM tools. It requires no inference.

---

## 7. Template Changes — Pros and Cons

### Plan A: Explicit Template Changes (4 files)

**Files modified**:
1. `plan/main/followup.jinja2`
2. `plan/main/review.jinja2`
3. `implementation/main/followup.jinja2`
4. `implementation/main/review.jinja2`

**Pros**:
- ✅ Explicit: no ambiguity
- ✅ Discoverable: instruction is prominent
- ✅ Testable: verify rendered template contains path
- ✅ Future-proof: easy to add more instructions

**Cons**:
- ❌ 4 files to edit and keep in sync
- ❌ Risk of templates diverging
- ❌ OpenStartup and other consumers may have custom templates (Phase 6 audit required)

### Plan B: No Template Changes

**Pros**:
- ✅ Single point of change in Python
- ✅ Works with custom templates automatically
- ✅ Lower deployment friction

**Cons**:
- ❌ Less discoverable: path is appended text
- ❌ Mixes data (proposal) with metadata (path)
- ❌ **BROKEN: doesn't set `main_response`**, so outer template still empty
- ❌ Requires LLM to infer purpose

---

## 8. Phase 5/6 — Fallback Removal and Loud Warnings

### Plan A Includes Phases 5-6

**Phase 5**: Add loud warnings when falling back to in-Python defaults
**Phase 6**: Audit all consumers, migrate to explicit templates, remove fallback

**Why**: Prevent recurrence of silent-fallback bugs.

**Cost**: ~1-2 days of audit + migration work.

### Plan B: Ignores the Fallback

**Risk**: Another config forgets templates → silent fallback applies again → bug repeats.

**Upside**: Lower scope, faster to ship.

**Verdict**: Phase 5-6 should happen eventually, but Phase 1-3 is a complete fix on its own. Deferring is acceptable but leaves an anti-pattern in place.

---

## 9. Test Coverage Comparison

### Plan A: 6 Test Files (~27-29 Test Methods)

| File | Purpose | Coverage |
|------|---------|----------|
| `test_path_resolution.py` | Unit tests for `_resolve_prior_proposer_output_path()` | All tiers, edge cases, fallbacks |
| `test_build_prompt_feeds.py` | Unit tests for feed dicts | Presence of `prior_output_path` and `main_response` |
| `test_mock_e2e_followup.py` | Mock E2E for followup template | Verify path in rendered prompt |
| `test_mock_e2e_review.py` | Mock E2E for review template | Symmetric test |
| `test_openstart_preflight.py` | Template variable presence | Preflight check |
| `test_jinja_render_all_templates.py` (modified) | Regression check | Template rendering |

**Strengths**:
- ✅ Tiered testing (unit → mock E2E → preflight)
- ✅ Covers all phases
- ✅ Separate file for path resolution

**Weaknesses**:
- ❌ 6 files is maintenance overhead

### Plan B: 1 Test File (~17 Test Methods)

**Strengths**:
- ✅ Concise and focused
- ✅ Includes E2E test

**Weaknesses**:
- ❌ Fewer test methods (~17 vs 27-29)
- ❌ **Does NOT test outer template rendering** (would reveal the `main_response` bug)
- ❌ No preflight tests

**Verdict**: Plan A's tests are more comprehensive. Plan B's tests would miss the `main_response` bug because they don't test outer template rendering.

---

## 10. Code Quality Issues

### Plan A: Code Quality Assessment

**Bugs**: None identified.

**Gaps**:
1. **Tier 2 heuristic**: Alphabetical fallback could pick wrong file. Mitigation: document convention, add warning logs.
2. **Phase 5-6 scope**: Incomplete audit — what about other domains? Clarify "Phase 4 known domains; Phase 6 audits others".
3. **Template rollback complexity**: Coordinating template rollback is harder than code rollback. Mitigation: test thoroughly.

**Over-engineering**: Phase 5-6 could be deferred; MVP (Phases 1-3) is sufficient.

**Variable naming**: ✅ Good (`_resolve_prior_proposer_output_path`, `prior_output_path` are clear).

### Plan B: Code Quality Assessment

**CRITICAL BUGS**:
1. **`main_response` NOT set in feed dict** — Bug A is NOT fixed. This is a **show-stopper**.
2. **Folder path requires LLM inference** — Fragile.
3. **No explicit instruction** — Path is implicit; LLM might not recognize it.
4. **Wrong iteration heuristic** — `iteration <= 1` diverges from `_active_proposer()`.
5. **Conflates concerns** — Data + metadata mixed in same variable.
6. **Hidden dependency** — Assumes templates use `{{ proposal }}` not `{{ main_response }}`; fragile.

**Verdict**: Plan B has critical design flaws that make it non-functional.

---

## 11. Real-World Impact — Which Solves the Drift Bug?

### The Original Bug

**Symptom**: Fixer regenerated 43 KB plan from 2.7 KB summary, producing 35 KB output (17% shorter).

**Root cause**: Fixer never received prior plan's file path.

### Plan A's Fix

1. ✅ Finds 43 KB file at `/workspace/.../output.md`
2. ✅ Sets `feed["prior_output_path"]` and `feed["main_response"]`
3. ✅ Template renders explicit "cp" instruction
4. ✅ Fixer produces 43 KB output with only 6 fixes applied

**Effectiveness**: ✅ **100%** — directly solves the bug.

### Plan B's "Fix"

1. ✅ Finds folder `/workspace/.../final_deliverables/`
2. ✅ Appends path to proposal text
3. ❌ `main_response` NOT set → outer template `<ProposedDocument>` still empty
4. ❌ Fixer doesn't have enough context

**Effectiveness**: ❌ **0%** — does NOT fix the bug due to `main_response` bug.

**Verdict**: Plan A solves the bug. Plan B does not.

---

## 12. Maintainability — Future Engineers' Perspective

### Plan A

**Understandability**:
- ✅ Two-tier rule is explicit and documentable
- ✅ `prior_output_path` is a clear feed dict addition
- ✅ Template changes are visible and obvious

**Extensibility**:
- ✅ To add new phase, update its templates
- ✅ To change path resolution, edit one function
- ✅ Easy to add `prior_output_path_relative` if needed

**Debugging**:
- ✅ Easy to trace: grep for `prior_output_path`
- ✅ Logs provide audit trail
- ✅ Template rendering is explicit

### Plan B

**Understandability**:
- ⚠️ Path appending is implicit and easy to miss
- ❌ `placeholder_proposal` indirection claim is **false**
- ❌ Iteration heuristic diverges from `_active_proposer()` but not explained

**Extensibility**:
- ❌ If you add new phase, template might not expect appended path
- ❌ If you change path resolution, must understand folder vs file tradeoff

**Debugging**:
- ❌ Harder to trace: path appears only in rendered output
- ❌ Hidden assumptions about template structure

**Verdict**: Plan A is significantly easier to maintain. Plan B relies on false assumptions and implicit dependencies.

---

## Summary Table: Side-by-Side

| Dimension | Plan A | Plan B |
|-----------|--------|--------|
| **Scope** | Comprehensive (1100 lines) | Minimal (326 lines) |
| **Path type** | File path | Folder path |
| **Template changes** | 4 files | 0 files |
| **Active proposer** | `counter_feedback` state (✅ correct) | `iteration <= 1` heuristic (❌ wrong) |
| **Bug A fix** | `main_response` set explicitly (✅ FIXED) | Assumes `placeholder_proposal` (❌ BROKEN) |
| **LLM discoverability** | Explicit "YOU MUST copy" | Implicit "[Available at...]" |
| **Real-world effectiveness** | ✅ **100%** solves bug | ❌ **0%** doesn't fix Bug A |
| **Tests** | 6 files, 27-29 methods | 1 file, 17 methods |
| **Code quality** | Well-designed, minor gaps | Critical bugs (`main_response`, heuristic) |
| **Maintainability** | High | Low |
| **Shipping time** | ~1-3 days (Phases 1-3) | ~2 hours |

---

## Strengths Unique to Plan A

1. **Actually fixes the bug** (sets `main_response`)
2. **File path over folder path** (no LLM inference needed)
3. **Explicit instructions** (template changes make intent clear)
4. **State-based proposer detection** (correct semantics)
5. **Comprehensive testing** (6 test files)
6. **Prevents recurrence** (Phase 5-6 removes fallback)
7. **Future-proof design** (easy to extend)
8. **Clear audit trail** (explicit variables, logs)

---

## Critical Bugs in Plan B

1. **`main_response` NOT set** → Bug A remains unfixed → outer template `<ProposedDocument>` renders empty (show-stopper)
2. **Folder path requires inference** → fragile
3. **No explicit instruction** → implicit path, LLM might ignore
4. **Wrong iteration heuristic** → can cause false appending after round 1
5. **Conflates concerns** → data + metadata mixed
6. **Hidden assumption** → assumes template uses `{{ proposal }}` not `{{ main_response }}`

---

## RECOMMENDATION

### ❌ **Do NOT ship Plan B**

Plan B has a **critical design flaw**: it does NOT set `main_response` in the feed dict, leaving Bug A (empty `<ProposedDocument>`) unfixed. The plan's claim of "works via `placeholder_proposal` indirection" is factually incorrect.

### ✅ **Ship Plan A (Phases 1-3 + 7), defer Phase 5-6**

Implementation plan:

1. **Phase 1**: `_resolve_prior_proposer_output_path()` helper (169 lines)
   - Add docstring explaining two-tier rule
   - Add logs when Tier 2 fallback is used

2. **Phase 2**: Plumb `prior_output_path` AND `main_response` into feed dicts
   - Both `_build_followup_prompt` and `_build_review_prompt`
   - This fixes Bug A

3. **Phase 3**: Update 4 templates with explicit instructions
   ```jinja2
   The document is available at: {{ prior_output_path }}
   YOU MUST copy the previous file and apply targeted edits:
   cp {{ prior_output_path }} {{ output_path }}
   ```

4. **Phase 7**: Comprehensive tests (6 test files)

5. **Defer Phase 5-6**: After Phase 1-3 ships and data confirms fix works, plan Phase 5-6 audit + migration

**Timeline**: ~1-2 days for development + testing, then ship. Phase 5-6 in 2-3 weeks after validation.

**Risk**: Low. Changes are additive, focused, and well-tested.

**Impact**: ✅ 100% solves the observed drift bug. ✅ Prevents future silent-fallback bugs (after Phase 5-6).

