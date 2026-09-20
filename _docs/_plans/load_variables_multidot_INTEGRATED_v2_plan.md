# `TemplateManager.load_variables` Multi-Dot Support — Integrated v2 Plan

**Author:** Tony Chen (integrating Rovo Dev v1 + Claude's pinpoint plan)
**Date drafted:** 2026-05-17 00:39
**Status:** Ready for review and implementation
**Supersedes:**
- `load_variables_multidot_enhancement_plan.md` (Rovo Dev v1, 441 lines)
- `/Users/tchen7/.claude/plans/let-s-create-an-integrated-lively-pearl.md` (Claude, 124 lines)

> **Why v2 exists.** v1 and Claude's plan agree on the architecture (fix `load_variables`'s first-dot-split bug at the source; remove the `_inject_mode_flags_and_content` workaround). v2 adopts Claude's cleaner `_store` helper, Claude's `_propagate_to_children` check, and Claude's broader verification test list — while keeping v1's risk register, permanent regression test, cross-repo PR discipline, and acceptance criteria. v2 also catches one latent bug in Claude's inline merge logic (shallow `update` vs needed deep merge) and one over-engineering in v1's `_store_nested` helper (15-line TypeError branch reduced to a 2-line assertion).

> **If forced to pick ONE plan today:** v1 (mine), because Claude's plan lacks the cross-repo PR discipline and the permanent regression test that prevents the workaround from coming back. But v2 is strictly better than either.

---

## 1. The bug — precise diagnosis (verified against current code)

**File:** `RichPythonUtils/src/rich_python_utils/string_utils/formatting/template_manager/template_manager.py`
**Buggy line:** 716

```python
# Current code (RichPythonUtils):
if "." in raw_key:
    base_var, nested_key = raw_key.split(".", 1)   # ← maxsplit=1 is the bug
else:
    base_var, nested_key = raw_key, None
```

**Symptom for `"instructions.modes.deep_mode"`:**
- Splits into `base_var="instructions"`, `nested_key="modes.deep_mode"`.
- Passes `nested_key` as `version` to `_cascade_load_variable`.
- Looks for a file literally named `modes.deep_mode.jinja2` (dot in filename) at `_variables/instructions/`.
- Not found → returns `None` → variable disappears from feed → Jinja2 renders empty.

**The downstream `_find_variable_file` in `FileBasedVariableManager` (lines 712–718) already converts multi-dot keys to slashes correctly**, but this code path is unreachable because `load_variables` has already mangled the input.

**The workaround that exists today:** `TemplatedInferencerBase._inject_mode_flags_and_content` (lines 213–276 of `templated_inferencer_base.py`) bypasses `load_variables` entirely and calls `_cascade_load_variable("instructions/modes", mode_name, ...)` with a pre-slashed path. The code carries this self-incriminating comment:

> *"load_variables() in TemplateManager only splits on the FIRST dot (so 'instructions.modes.deep_mode' resolves wrong). We use _cascade_load_variable directly for proper nested access."*

This workaround handles `modes` only. The user's new `instructions.behavior.*` use case hits the same bug, and adding `_inject_behavior_content` would be the hacky-and-ad-hoc direction the user explicitly rejected. **The correct fix is at the source: make `load_variables` handle N-dot keys properly, then delete the workaround.**

---

## 2. Target design

### 2.1 Key shape contract for `load_variables`

| Key shape | Resolves to file | Feed dict shape |
|---|---|---|
| `"task_preamble"` | `_variables/task_preamble.jinja2` (via default_version) | `{"task_preamble": "<content>"}` |
| `"notes.local_search_efficiency"` | `_variables/notes/local_search_efficiency.jinja2` | `{"notes": {"local_search_efficiency": "<content>"}}` |
| `"instructions.modes.deep_mode"` | `_variables/instructions/modes/deep_mode.jinja2` | `{"instructions": {"modes": {"deep_mode": "<content>"}}}` |
| `"instructions.behavior.file_reading_fallback"` | `_variables/instructions/behavior/file_reading_fallback.jinja2` | `{"instructions": {"behavior": {"file_reading_fallback": "<content>"}}}` |
| `"a.b.c.d.e"` | `_variables/a/b/c/d/e.jinja2` | 5-level nested dict |

**Rule:** All dots become path separators. The last segment is the file stem.

### 2.2 Sibling-resolution is unchanged

When a variable file (e.g., `file_reading_fallback_for_review.jinja2`) references `{{ file_reading_fallback }}` internally, the existing `_resolve_variable` logic in `file_based.py:1024-1038` does sibling-first resolution using `current_level_path`. **This already works** and this plan does NOT touch it.

---

## 3. The fix — three precise edits

### 3.1 Edit A — `RichPythonUtils/template_manager.py:711-720`

**OLD:**
```python
for raw_key, spec in variable_specs.items():
    # Dot-key: "notes.local_search_efficiency" → nested dict output
    # for Jinja2 {{ notes.local_search_efficiency }} access.
    # Splits into var_name="notes" (directory) + nested_key="local_search_efficiency" (file).
    if "." in raw_key:
        base_var, nested_key = raw_key.split(".", 1)
    else:
        base_var, nested_key = raw_key, None
```

**NEW:**
```python
for raw_key, spec in variable_specs.items():
    # Dot-key: ALL dots become path separators; LAST segment is the file stem.
    # Result is a deeply-nested dict for Jinja2 attribute traversal.
    #   "notes.local_search_efficiency"
    #     → var_dir="notes", file_stem="local_search_efficiency"
    #     → file: _variables/notes/local_search_efficiency.jinja2
    #     → feed: {"notes": {"local_search_efficiency": "<content>"}}
    #   "instructions.modes.deep_mode"
    #     → var_dir="instructions/modes", file_stem="deep_mode"
    #     → file: _variables/instructions/modes/deep_mode.jinja2
    #     → feed: {"instructions": {"modes": {"deep_mode": "<content>"}}}
    if "." in raw_key:
        parts = raw_key.split(".")
        var_dir = "/".join(parts[:-1])  # directory path under _variables/
        file_stem = parts[-1]            # file stem (no .jinja2)
        nested_path = parts              # for _store
    else:
        var_dir = raw_key
        file_stem = None                 # use default_version
        nested_path = [raw_key]
```

### 3.2 Edit B — replace `_store` with N-level `_store` (Claude's clean version + 1 defensive assertion)

**OLD:**
```python
def _store(base: str, nested_key: Optional[str], content: Any) -> None:
    if nested_key is not None:
        result.setdefault(base, {})[nested_key] = content
    else:
        result[base] = content
```

**NEW:**
```python
def _store(parts: List[str], content: Any) -> None:
    """Store ``content`` at arbitrary nesting depth: result[parts[0]]...[parts[-1]].

    Intermediate dicts are created on demand. Defensive assertion guards
    against silent string-vs-dict shadowing if a caller mixes flat and
    nested keys that share a prefix (e.g., {"a": "x", "a.b": "y"}).
    """
    d = result
    for part in parts[:-1]:
        nxt = d.setdefault(part, {})
        assert isinstance(nxt, dict), (
            f"load_variables: cannot nest key path {'.'.join(parts)!r}; "
            f"segment {part!r} is already bound to a non-dict value "
            f"({type(nxt).__name__}). Don't mix flat and nested keys "
            f"that share a prefix."
        )
        d = nxt
    d[parts[-1]] = content
```

> **Why both elegant AND safe:** Claude's plan used 3 lines (`d.setdefault(part, {})`) which is clean but would silently corrupt if `parts[0]` already maps to a string. v1 used a 15-line if/elif/else with explicit TypeError. v2 keeps Claude's clean shape and adds ONE defensive assertion — total 5 functional lines.

### 3.3 Edit C — rewire the resolution-loop body

Wherever the existing loop calls `_cascade_load_variable(base_var, version, ...)` and `_store(base_var, nested_key, ...)`, swap to:

- `_cascade_load_variable(var_dir, file_stem or default_version, root_space, tmpl_type)`
- `_store(nested_path, content)`

All value semantics (`"@strict"`, `"=literal"`, plain, `None`) route through `_store(nested_path, ...)` with the full path. **No change to value-prefix semantics.**

### 3.4 Update `load_variables` docstring

Update the docstring to describe N-level support; the current docstring (lines 656–706) documents only flat and 2-level dot keys. Add an example for 3-level, and add the warning about flat-vs-nested prefix collisions.

---

## 4. Remove the workaround in AgentFoundation

**File:** `AgentFoundation/src/agent_foundation/common/inferencers/templated_inferencer_base.py`

### 4.1 Delete `_inject_mode_flags_and_content` (lines 213–276)

The entire method. Plus the self-incriminating comment block at lines 198–205.

### 4.2 Replace the call site in `_build_template_feed` (lines 198–205)

**OLD:**
```python
# Mode handling: for each declared mode, set enable_<name> bool
# and (if enabled) load instruction content into nested dict for
# {{ instructions.modes.<name> }} Jinja2 access.
# Note: load_variables() in TemplateManager only splits on the
# FIRST dot (so "instructions.modes.deep_mode" resolves wrong).
# We use _cascade_load_variable directly for proper nested access.
if self.modes:
    self._inject_mode_flags_and_content(feed)
```

**NEW:**
```python
# Mode handling: derive enable_<name> flags unconditionally (so
# {%- if enable_X %} can check) and load instruction content for
# enabled modes via load_variables (which now handles multi-dot
# keys natively — see RichPythonUtils load_variables fix).
if self.modes:
    for mode_name, enabled in self.modes.items():
        feed[f"enable_{mode_name}"] = bool(enabled)
    enabled_specs = {
        f"instructions.modes.{name}": None
        for name, enabled in self.modes.items()
        if enabled
    }
    if enabled_specs and self.template_manager:
        try:
            loaded = self.template_manager.load_variables(
                variable_specs=enabled_specs,
                root_space=self.template_root_space or "",
            )
        except FileNotFoundError:
            logger.debug(
                "Mode enabled but no instruction file found at "
                "_variables/instructions/modes/{name}.jinja2 — "
                "{{ instructions.modes.<name> }} will render empty."
            )
            loaded = {}
        _deep_merge_into(feed, loaded)
```

### 4.3 Add `_deep_merge_into` helper (module-private)

> **Why this exists:** Claude's plan inlined the merge as `existing.update(v)`, which is a *shallow* merge. If `feed["instructions"]` already exists (from another caller in the future, or from a propagated parent feed), `update` would replace `feed["instructions"]["modes"]` wholesale and lose any other sub-namespace (e.g., `feed["instructions"]["behavior"]`). The recursive helper avoids that latent corruption.

```python
def _deep_merge_into(target: dict, source: dict) -> None:
    """Recursively merge ``source`` into ``target``.

    For each key in ``source``:
      - If both ``target[key]`` and ``source[key]`` are dicts, recurse.
      - Otherwise, ``source[key]`` overwrites ``target[key]``.

    Used to fold load_variables() output into the feed without losing
    sibling sub-namespaces. Example: if feed["instructions"]["behavior"]
    is already populated and loaded contains
    {"instructions": {"modes": {...}}}, the merged feed has BOTH
    instructions.behavior AND instructions.modes intact.
    """
    for k, v in source.items():
        existing = target.get(k)
        if isinstance(existing, dict) and isinstance(v, dict):
            _deep_merge_into(existing, v)
        else:
            target[k] = v
```

### 4.4 `_propagate_to_children` check (from Claude — real gap in v1)

The `modes` dict is propagated to children via `_propagate_dict_attr_to_children(self.modes, "modes")`. This propagates the *dict*, not the loaded *content*. So children re-load their own mode content via the new `load_variables` path. **This is unchanged behavior** and requires no edit — but the plan must explicitly verify it.

**Verification step:** Phase 6 includes asserting that a child inferencer with `modes={"deep_mode": True}` (propagated from parent) renders the deep-mode instruction text correctly.

---

## 5. Test plan

### 5.1 Phase 0 — RED tests (pin contract before any source edit)

**File (NEW):** `RichPythonUtils/test/string_utils/formatting/template_manager/test_load_variables_multidot.py`

8 tests, ALL `xfail(strict=True)` before the fix:

| # | Test | What it pins | xfail until |
|---|---|---|---|
| 1 | `test_flat_key_unchanged` | Backward compat: `{"task_preamble": "default"}` | Always PASS (regression guard) |
| 2 | `test_two_level_dotted_key_unchanged` | `{"notes.local_search_efficiency": ""}` → nested 2-level | Always PASS (regression guard) |
| 3 | `test_three_level_dotted_key` | `{"instructions.modes.deep_mode": ""}` → nested 3-level | After Edit A+B+C |
| 4 | `test_four_level_dotted_key` | `{"a.b.c.d": ""}` → nested 4-level | After Edit A+B+C |
| 5 | `test_multiple_keys_share_root_merge` | `{"instructions.modes.deep_mode": "", "instructions.behavior.file_reading_fallback": ""}` produces single `instructions` dict with both sub-namespaces | After Edit A+B+C |
| 6 | `test_flat_then_nested_raises_clear_error` | `{"instructions": "literal", "instructions.modes.x": ""}` raises `AssertionError` with documented message | After Edit B |
| 7 | `test_strict_prefix_still_raises_on_missing` | `{"instructions.modes.nonexistent": "@strict"}` raises `FileNotFoundError` | After Edit A+B+C |
| 8 | `test_literal_prefix_still_skips_file` | `{"instructions.modes.x": "=hello world"}` → `{"instructions": {"modes": {"x": "hello world"}}}` (no filesystem read) | After Edit A+B+C |

### 5.2 AgentFoundation integration — existing tests (Claude's list)

These already exist or should:
1. **`test_behavior_variable_injection.py`** (156 lines, exists, FAILS today) — the 3 tests inside (`test_base_fallback_variable_loads`, `test_review_fallback_variable_loads`, `test_followup_fallback_variable_loads`) all GREEN after the fix. This is the strongest user-perspective signal.
2. **Existing mode tests** — find them via `grep -rn "deep_mode\|elegant_mode" CoreProjects/AgentFoundation/test/`. All must continue to GREEN (regression guard).
3. **`test_variable_two_pass_search.py`** (in RichPythonUtils) — Claude flagged this. Existing dot-to-slash tests must continue to GREEN.
4. **Sibling resolution test** — verify `file_reading_fallback_for_review.jinja2` containing `{{ file_reading_fallback }}` continues to resolve the sibling file. This is the sibling-resolution path in `file_based.py:1024-1038`, untouched by this plan.

### 5.3 Permanent regression test in AgentFoundation (NEW — v1's addition)

**File (NEW):** `CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers/test_templated_inferencer_no_workaround.py`

```python
def test_inject_mode_flags_and_content_workaround_is_removed():
    """Prevent regression: the _inject_mode_flags_and_content workaround
    must NOT come back. Mode injection should go through
    TemplateManager.load_variables with proper multi-dot support.
    See plan §4.

    If this test fails, someone re-introduced the hack — push back on
    the PR and direct them to fix load_variables instead.
    """
    from agent_foundation.common.inferencers.templated_inferencer_base import (
        TemplatedInferencerBase,
    )
    assert not hasattr(TemplatedInferencerBase, "_inject_mode_flags_and_content"), (
        "_inject_mode_flags_and_content was re-added to TemplatedInferencerBase. "
        "This is a workaround for a load_variables bug that has been FIXED in "
        "RichPythonUtils. Use load_variables({'instructions.modes.X': None}) "
        "instead. See _docs/_plans/load_variables_multidot_INTEGRATED_v2_plan.md §4."
    )


def test_deep_mode_renders_via_load_variables_only():
    """End-to-end: a class with modes={'deep_mode': True} produces a
    feed where instructions.modes.deep_mode is the file content,
    going through load_variables (not the deleted workaround)."""
    # ...uses an in-process TemplatedInferencerBase subclass + a temp
    # _variables/instructions/modes/deep_mode.jinja2 fixture...
```

### 5.4 Sibling-resolution smoke test

Add one explicit smoke test (small, focused) that confirms the sibling-first resolution still works:

```python
def test_sibling_resolution_within_variable_file_still_works():
    """file_reading_fallback_for_review.jinja2 contains
    {{ file_reading_fallback }} and must resolve the sibling file
    in the same _variables/instructions/behavior/ directory."""
    # render the template and assert the inner content is present
```

---

## 6. Phased rollout

| Phase | What | Files | Risk | Reversible? |
|---|---|---|---|---|
| 0 | Pre-flight: confirm `test_behavior_variable_injection.py` FAILS today; write 8 RED tests | 1 new test file | none | n/a |
| 1 | RichPythonUtils Edit A (split rule) | 1 file | medium (cross-repo) | yes |
| 2 | RichPythonUtils Edit B (`_store` with assertion) | same file | low | yes |
| 3 | RichPythonUtils Edit C (rewire loop body) + docstring update | same file | low | yes |
| 4 | RichPythonUtils: 8 unit tests turn GREEN | new test file | none | yes |
| 5 | AgentFoundation: delete `_inject_mode_flags_and_content` + add `_deep_merge_into` helper + rewrite mode-handling block | 1 file | medium | yes |
| 6 | AgentFoundation: existing `test_behavior_variable_injection.py` GREEN; permanent regression test added | 1 new test file | trivial | yes |
| 7 | Smoke: render representative templates with modes enabled; assert no `{{ ... }}` literals leak through | 1 smoke test | trivial | yes |

### 6.1 Cross-repo PR ordering (v1's discipline — Claude omitted)

- **PR-A (RichPythonUtils):** Phases 0–4. Lands first. Strictly backward-compatible. Tag a new RichPythonUtils version.
- **PR-B (AgentFoundation):** Phases 5–7. Depends on PR-A's tag. Lands after AgentFoundation's pyproject.toml is bumped to the new RichPythonUtils version.

**Do NOT bundle.** Cross-repo bundles are hard to revert and harder to bisect.

### 6.2 Rollback strategy

| Revert | Consequence |
|---|---|
| PR-B only (AgentFoundation) | Workaround comes back; mode injection works via the old path; `instructions.behavior.*` references still fail (pre-bug state). |
| PR-A only (after PR-B has landed) | `load_variables` returns to first-dot split; AgentFoundation's removed workaround can't compensate; modes silently render empty. **DO NOT revert PR-A while PR-B is live.** |
| Both | Full pre-state. Safe. |

If hot rollback needed, revert PR-B first, then PR-A. CI must enforce this ordering by version pinning.

---

## 7. Risk register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Existing caller depends on the FIRST-dot-split bug (e.g., a key `"foo.bar.dots.in.it"` that today resolves to a file literally named `bar.dots.in.it.jinja2`) | 🟡 Medium | Pre-flight grep across AgentFoundation + RichPythonUtils for all `load_variables(` callers; inspect each `variable_specs=` dict. Investigation done: only flat and 2-level dotted keys exist. New behavior is a strict superset for those. |
| 2 | The new `_store` assertion fires on flat-vs-nested prefix collision (e.g., `{"a": "x", "a.b.c": ""}`) | 🟢 Low | NEW error mode but no existing caller mixes flat and nested under the same root. Error message points to the plan §3.2 line. Test #6 pins it. |
| 3 | Subclass overrides `_inject_mode_flags_and_content` | 🟡 Medium | Pre-flight grep: `def _inject_mode_flags_and_content` should appear only in `templated_inferencer_base.py`. Verify; migrate any overrides in the same PR. |
| 4 | Claude's inline `existing.update(v)` shallow-merge would lose sibling namespaces if `feed["instructions"]` already exists from another source | 🟢 Low (today; latent) | v2 uses recursive `_deep_merge_into` helper instead. Latent bug closed. |
| 5 | Cross-repo coordination — PR-A merges, PR-B doesn't get rebased onto new RichPythonUtils version | 🟡 Medium | Pin RichPythonUtils version in AgentFoundation's pyproject.toml as part of PR-B. CI catches mismatch via new tests. |
| 6 | A future contributor re-adds `_inject_mode_flags_and_content` for some local issue | 🟢 Low | Permanent regression test §5.3 fails the build immediately and points at this plan. |
| 7 | Performance: recursive deep-merge has O(N) per key | 🟢 Low | N is bounded by template key dot count (typically 2–3). Negligible. |
| 8 | `load_variables` docstring no longer documents the new behavior | 🟢 Low | Edit C includes docstring update. Acceptance criterion below. |

---

## 8. Acceptance criteria

**PR-A (RichPythonUtils) is mergeable when:**
- ☐ All 8 new unit tests in §5.1 pass.
- ☐ Existing RichPythonUtils test suite passes (no regression).
- ☐ `load_variables` docstring updated with N-level example + prefix-collision warning.
- ☐ `test_variable_two_pass_search.py` continues to PASS.

**PR-B (AgentFoundation) is mergeable when:**
- ☐ `_inject_mode_flags_and_content` removed from `templated_inferencer_base.py`.
- ☐ `_deep_merge_into` helper added (module-private).
- ☐ All 3 tests in `test_behavior_variable_injection.py` GREEN.
- ☐ Permanent regression test (§5.3) in CI.
- ☐ Sibling-resolution smoke test (§5.4) in CI.
- ☐ `grep -rn "_inject_mode_flags_and_content" CoreProjects/AgentFoundation/` returns zero matches.
- ☐ Manual smoke: render `plan/main/initial.jinja2` with `modes={"deep_mode": True}` and assert the output contains the deep-mode instruction text (NOT literal `{{ instructions.modes.deep_mode }}`).
- ☐ AgentFoundation's pyproject.toml bumped to the new RichPythonUtils version.

---

## 9. Comparison with source plans

| Aspect | v1 (Rovo Dev) | Claude | v2 (this plan) |
|---|---|---|---|
| Architecture (fix at source + remove workaround) | ✅ | ✅ | ✅ |
| `_store` helper | 15-line TypeError branch (over-engineered) | 3-line setdefault (silent shadowing risk) | **5-line setdefault + 1 assertion (clean + safe)** |
| `_deep_merge_into` correctness | ✅ Recursive (correct) | ❌ Shallow `update` (latent multi-call bug) | **Recursive (v1's version)** |
| `_propagate_to_children` check | ❌ Not addressed | ✅ Mentioned and verified unchanged | **Adopt Claude's explicit check** |
| Verification test list | Only `test_behavior_variable_injection.py` | Adds `test_variable_two_pass_search.py` + sibling resolution + backward-compat for `notes.X` | **Adopt Claude's broader list** |
| Phase 0 RED tests | ✅ 8 tests | ❌ Verification afterthought | **Adopt v1's RED-first** |
| Risk register | ✅ 8 risks | ❌ Missing | **Adopt v1's; add Claude's `_propagate_to_children` as risk 0** |
| Cross-repo PR ordering | ✅ Explicit "do NOT bundle" + rollback matrix | ❌ Missing | **Adopt v1's** |
| Permanent regression test (workaround-must-not-return) | ✅ §5.3 | ❌ Missing | **Adopt v1's** |
| Acceptance criteria checkboxes | ✅ Per PR | ❌ Missing | **Adopt v1's** |
| Phased rollout | ✅ 6 phases | ❌ 3-change list | **Adopt v1's (extended to 7 with sibling smoke)** |
| Effort estimate | ✅ | ❌ | **Adopt v1's, ~13 hours** |
| Length | 441 lines | 124 lines | **~330 lines (target — pinpoint + operational)** |

### 9.1 If forced to pick ONE plan today, which?

**v1 (mine, the 441-line plan).**

Reasons:
- Both plans have the right architecture, but Claude's lacks:
  - The cross-repo PR ordering discipline (cross-repo bundles are a real revert hazard)
  - The permanent regression test (without it, a future contributor will re-add the workaround the next time someone hits the same bug class)
  - The risk register with severity ratings
  - Acceptance criteria per PR
- Claude's plan has a latent bug (shallow `update` in the inline merge) that would silently lose sibling sub-namespaces if `feed["instructions"]` ever has more than one sub-key
- Claude's `_store` helper is cleaner than mine, but mine is correct (with the over-engineering being safety, not error)

If you only had Claude's plan and shipped from it, you'd:
- Land the fix
- Lose `instructions.behavior` sub-namespace if it ever co-existed with `instructions.modes` in the feed (latent shallow-merge bug)
- Have nothing preventing the workaround's return
- Have to figure out cross-repo coordination yourself

If you only had v1 and shipped from it:
- Same architecture, same outcome
- 15-line over-engineered `_store_nested` instead of 5
- Missed the `_propagate_to_children` verification (would discover during review)
- Missed `test_variable_two_pass_search.py` in the verification list

**v2 = v1's operational rigor + Claude's cleaner `_store` + Claude's `_propagate_to_children` check + Claude's broader verification list + the deep-merge correctness fix.**

---

## 10. What this plan deliberately does NOT do

- ❌ Does NOT rename `load_variables`.
- ❌ Does NOT change `@strict` / `=literal` / plain value semantics.
- ❌ Does NOT change `_find_variable_file` or `_resolve_variable` (already correct).
- ❌ Does NOT change sibling-first resolution for nested variable refs inside variable files.
- ❌ Does NOT introduce a feature flag (the fix is strictly backward-compatible).
- ❌ Does NOT touch any prompt template files (the user already updated them).
- ❌ Does NOT add `_inject_behavior_content` or any analogous workaround for `instructions.behavior.*` (the whole point is to NOT proliferate workarounds).

---

## 11. Design principles applied

1. **Fix at the source.** The bug is in RichPythonUtils. AgentFoundation's workaround is at the wrong layer.
2. **Strict backward compatibility.** The new behavior is a superset.
3. **Eliminate workarounds atomically with the fix.** Pin their absence with a permanent regression test.
4. **Clear errors over silent corruption.** Assertion message points to the plan section.
5. **Test the user-facing path.** `test_behavior_variable_injection.py` becomes the strongest validation signal.
6. **Cross-repo discipline.** Two PRs, ordered, not bundled.
7. **Elegant over clever.** Claude's `_store` is cleaner than v1's; adopt it. v1's `_deep_merge_into` is recursively correct; adopt it. Best of both, no ego.
8. **Pin the architecture with permanent tests.** The "workaround-must-not-return" test fails the build if anyone tries to re-add the hack.

---

## 12. Estimated effort

| Phase | Implementation | Tests | Review | Total (h) |
|---|---|---|---|---|
| 0 (RED tests) | 0 | 2 | 0.5 | 2.5 |
| 1–3 (RichPythonUtils fix) | 1.5 | 0 | 1 | 2.5 |
| 4 (RichPythonUtils tests GREEN) | 0 | 0.5 | 0.5 | 1 |
| 5 (AgentFoundation removal) | 2 | 0 | 1.5 | 3.5 |
| 6 (existing tests GREEN + permanent regression) | 0.5 | 1.5 | 1 | 3 |
| 7 (sibling smoke + manual smoke) | 0 | 1 | 0.5 | 1.5 |
| **Total** | **4** | **5** | **5** | **14** |

Roughly **1.5–2 engineer-days** total across both repos.

---

*End of v2 integrated plan. Reviewers: please challenge §3 (the exact code edits — especially the `_store` assertion line), §4.3 (the deep-merge helper's necessity), §6.1 (cross-repo PR ordering), and §9.1 ("if forced to one, pick v1") most carefully.*
