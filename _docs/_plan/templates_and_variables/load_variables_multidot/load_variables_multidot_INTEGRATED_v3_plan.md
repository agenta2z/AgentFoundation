# `TemplateManager.load_variables` Multi-Dot + Composition — Integrated v3 Plan

**Author:** Tony Chen (integrating Rovo Dev v1 + Cursor + Claude)
**Date drafted:** 2026-05-17 00:46
**Status:** Ready for review and implementation
**Supersedes:**
- v1 (Rovo Dev, 441 lines) — `load_variables_multidot_enhancement_plan.md`
- v2 (Rovo Dev integration of Claude#1, 365 lines) — `load_variables_multidot_INTEGRATED_v2_plan.md`
- Cursor plan (226 lines) — `/Users/tchen7/.cursor/plans/load_variables_multi-dot_enhancement_98c574e9.plan.md`
- Claude plan (14 lines, formally yields) — `/Users/tchen7/.claude/plans/let-s-create-an-integrated-lively-pearl.md`

> **Why v3 exists.** Cursor's plan caught three real correctness issues that v1/v2 missed: (1) `_cascade_load_variable` does raw `read_text()` with NO composition, so the user's `{{ file_reading_fallback }}` reference inside `file_reading_fallback_for_review.jinja2` would NOT resolve today; (2) the workaround hard-codes `tmpl_type="main"`, which is a latent bug for non-main inferencers; (3) a real test stub (`test_preflight_template_variable_coverage.py:148-153`) replicates the workaround and must be migrated. v3 integrates Cursor's architectural depth with v1's operational discipline (risk register, cross-repo PR ordering, permanent regression test, acceptance criteria). Claude formally yielded to v1; v3 is strictly better than any of the four input plans.

> **If forced to pick ONE plan today:** **Cursor's plan** — it catches the most correctness bugs. v1 is operationally more rigorous but architecturally incomplete. v2 is a refinement of v1, still architecturally incomplete. Claude yields. (Full reasoning in §11.)

---

## 1. Verified empirical claims (all independently confirmed against current code)

| # | Claim | Verified by |
|---|---|---|
| 1 | `TemplateManager.load_variables` splits on FIRST dot only (`raw_key.split(".", 1)` at line 716) | Read of `template_manager.py:711-720` |
| 2 | `FileBasedVariableManager._find_variable_file` correctly converts ALL dots to slashes (`variable_name.replace(".", "/")` at line 712-718) — but unreachable for dotted-key path because `load_variables` mangles first | Read of `file_based.py:711-719` |
| 3 | `TemplatedInferencerBase._inject_mode_flags_and_content` (lines 213-276) is a workaround that bypasses `load_variables` | Read of `templated_inferencer_base.py:213-276` |
| 4 | **The workaround hard-codes `"main"` as `tmpl_type`** (line 245) — silently wrong for non-main inferencers | Grep of `templated_inferencer_base.py` line 245: literal `"main"` passed as 4th positional arg |
| 5 | **`_cascade_load_variable` returns raw `read_text(...)`** with NO Jinja2 rendering or nested variable substitution (lines 646, 652) — so `{{ X }}` inside variable file contents is returned LITERAL | Grep of `template_manager.py:593-660` — only `read_text` calls, no rendering |
| 6 | **`test_preflight_template_variable_coverage.py:148-153`** has a `_inject_mode_flags` stub that calls `_cascade_load_variable` directly — will break when workaround is removed | Grep of the test file, line numbers verified |
| 7 | No callers in AgentFoundation or RichPythonUtils currently use dotted `template_variables=` keys at construction time | Grep for `template_variables` + dot patterns: only the workaround uses `instructions/modes` internally |

> **The most important verification:** Claim #5 (no composition) means v1/v2's assumption that "sibling resolution already works for `{{ file_reading_fallback }}` inside `file_reading_fallback_for_review.jinja2`" was **WRONG**. Cursor's plan correctly identifies that composition must be added — the user's stated use case cannot work without it.

---

## 2. The three real problems (not just one)

### 2.1 Problem A — Multi-dot key splitting (v1/v2/Cursor all agree)

`load_variables` splits on FIRST dot, can't reach 3+ level paths. Fix: split on ALL dots, last segment is the file stem.

### 2.2 Problem B — No composition in `_cascade_load_variable` (Cursor's catch)

When `_cascade_load_variable` loads `file_reading_fallback_for_review.jinja2`, it returns the file content **as a raw string**:

```text
{{ file_reading_fallback }} If BOTH fail for a referenced path, raise a MAJOR `verification_gap` issue...
```

The `{{ file_reading_fallback }}` is returned LITERAL. Jinja2 rendering of the OUTER template might or might not re-process it (depending on context dict shape — if `file_reading_fallback` is not in scope, it renders as empty string). The user's stated design — sibling variable files composing each other — does NOT work today.

**Fix:** add a `_compose_variable_content` helper that recursively resolves `{{ X }}` references via sibling-first then cascade, with cycle detection and max-depth cap matching `FileBasedVariableManager` semantics.

### 2.3 Problem C — Hard-coded `tmpl_type="main"` in the workaround

`_inject_mode_flags_and_content` at line 245 passes literal `"main"` as the 4th argument. For any inferencer whose `active_template_type` is NOT `"main"`, modes silently load from `_variables/instructions/modes/<name>.jinja2` under the `main` type — wrong directory cascade. Latent bug today (all current inferencers use `main`); explicit bug the moment a non-main inferencer adds modes.

**Fix:** the workaround removal already eliminates this — but the v3 plan calls it out so the test migration (§4.4) doesn't re-introduce the same hard-coding.

---

## 3. Target design

### 3.1 Key shape contract (unchanged from v2)

| Key shape | Resolves to file | Feed dict shape |
|---|---|---|
| `"task_preamble"` | `_variables/task_preamble.jinja2` | `{"task_preamble": "<content>"}` |
| `"notes.local_search_efficiency"` | `_variables/notes/local_search_efficiency.jinja2` | `{"notes": {"local_search_efficiency": "<content>"}}` |
| `"instructions.modes.deep_mode"` | `_variables/instructions/modes/deep_mode.jinja2` | `{"instructions": {"modes": {"deep_mode": "<content>"}}}` |
| `"instructions.behavior.file_reading_fallback"` | `_variables/instructions/behavior/file_reading_fallback.jinja2` | `{"instructions": {"behavior": {"file_reading_fallback": "<content>"}}}` |
| `"a.b.c.d.e"` | `_variables/a/b/c/d/e.jinja2` | 5-level nested dict |

**Rule:** All dots become path separators. Last segment is the file stem.

### 3.2 Composition contract (Cursor's addition — critical)

When a variable file content contains `{{ X }}` references, those references must resolve at load time, BEFORE being injected into the outer template's feed dict. Resolution order:

1. **Sibling-first** — look in the same directory as the current file (e.g., `_variables/instructions/behavior/X.jinja2`).
2. **Cascade fallback** — walk up the cascade paths (`<space>/main` → `<space>` → global) for `X.jinja2`.
3. **Cycle detection** — track visited file paths; raise `CircularReferenceError` if revisited.
4. **Max depth cap** — default 10 (match `FileBasedVariableManager` semantics).
5. **Unresolved reference** — if `X` is not findable in any of (1)–(2), leave it literal so the outer Jinja2 render can fall back to the feed dict (e.g., template-author-provided variables).

This is the same algorithm `FileBasedVariableManager._resolve_variable` already implements (`file_based.py:1024-1038`); the fix is to **call into it** from the `load_variables` path instead of returning raw `read_text` output.

### 3.3 Sibling-resolution mechanism reuse

Rather than reimplementing composition, reuse `FileBasedVariableManager._resolve_variable` with `current_level_path` set to the loaded file's path. Two architectural options:

- **Option A (Cursor's):** Extract `_cascade_load_variable_path` that returns `(content, file_path)`; the existing `_cascade_load_variable` becomes a thin wrapper returning only content (preserves the public API + monkeypatch tests).
- **Option B (alternative):** Add a `compose=True` parameter to `_cascade_load_variable` that triggers the recursive resolution.

**v3 picks Option A** because:
- Preserves the existing `_cascade_load_variable(path, name, root_space, tmpl_type) -> str` signature exactly.
- Monkeypatch tests (`test_templated_inferencer_modes.py:567-581`, `test_preflight_template_variable_coverage.py:148-153`) keep working without modification (they monkeypatch the wrapper, which now delegates).
- Composition is enabled by default for `load_variables` callers (the user-facing API) and disabled by default for direct `_cascade_load_variable` callers (the internal path used by tests).

---

## 4. The fix — four precise edits

### 4.1 Edit A — RichPythonUtils: multi-dot split in `load_variables` (lines 711-720)

**OLD:**
```python
for raw_key, spec in variable_specs.items():
    if "." in raw_key:
        base_var, nested_key = raw_key.split(".", 1)   # ← bug
    else:
        base_var, nested_key = raw_key, None
```

**NEW:**
```python
for raw_key, spec in variable_specs.items():
    # Dot-key: ALL dots become path separators; LAST segment is the file stem.
    # Produces a deeply-nested dict for Jinja2 attribute traversal.
    #   "notes.local_search_efficiency"
    #     → var_dir="notes", file_stem="local_search_efficiency", nested_path=["notes", "local_search_efficiency"]
    #   "instructions.modes.deep_mode"
    #     → var_dir="instructions/modes", file_stem="deep_mode", nested_path=["instructions","modes","deep_mode"]
    if "." in raw_key:
        parts = raw_key.split(".")
        var_dir = "/".join(parts[:-1])
        file_stem = parts[-1]
        nested_path = parts
    else:
        var_dir = raw_key
        file_stem = None    # use default_version
        nested_path = [raw_key]
```

### 4.2 Edit B — RichPythonUtils: replace `_store` with N-level `_store_nested`

**OLD:**
```python
def _store(base: str, nested_key: Optional[str], content: Any) -> None:
    if nested_key is not None:
        result.setdefault(base, {})[nested_key] = content
    else:
        result[base] = content
```

**NEW** (combines Cursor's clean setdefault chain with v1's defensive assertion):
```python
def _store_nested(parts: List[str], content: Any) -> None:
    """Store ``content`` at arbitrary nesting depth: result[parts[0]]...[parts[-1]].

    Intermediate dicts are created on demand. Defensive assertion guards
    against silent string-vs-dict shadowing if a caller mixes flat and
    nested keys that share a prefix (e.g., {"a": "x", "a.b": "y"}).
    """
    d = result
    for part in parts[:-1]:
        nxt = d.setdefault(part, {})
        assert isinstance(nxt, dict), (
            f"load_variables: cannot nest key {'.'.join(parts)!r}; "
            f"segment {part!r} is bound to a non-dict ({type(nxt).__name__}). "
            f"Don't mix flat and nested keys that share a prefix."
        )
        d = nxt
    d[parts[-1]] = content
```

Update all `_cascade_load_variable(base_var, version, ...)` calls in the loop body to `_cascade_load_variable_path(var_dir, file_stem or default_version, ...)`, and all `_store(base_var, nested_key, ...)` calls to `_store_nested(nested_path, content)`. Value semantics (`"@strict"`, `"=literal"`, plain, `None`) are unchanged.

### 4.3 Edit C — RichPythonUtils: extract `_cascade_load_variable_path` + add `_compose_variable_content`

**NEW helper 1** (returns content + file path so composition knows where the file lived):

```python
def _cascade_load_variable_path(
    self,
    var_name: str,
    version: str,
    root_space: str,
    tmpl_type: str,
) -> Tuple[Optional[str], Optional[Path]]:
    """Same cascade resolution as _cascade_load_variable, but also returns
    the resolved file path (or None if not found).

    The path is needed for sibling-first composition of nested {{ X }}
    references inside the loaded content.
    """
    # ... existing cascade body, but capture `resolved` path and return ...
    # Returns (None, None) if not found.
```

**Make `_cascade_load_variable` a thin wrapper** (preserves existing public API + monkeypatch tests):

```python
def _cascade_load_variable(
    self,
    var_name: str,
    version: str,
    root_space: str,
    tmpl_type: str,
) -> Optional[str]:
    """Backward-compatible wrapper. Returns raw content (no composition).
    Use load_variables() for composition. Use _cascade_load_variable_path()
    when you need the file path."""
    content, _ = self._cascade_load_variable_path(var_name, version, root_space, tmpl_type)
    return content
```

**NEW helper 2** — composition (reuses `FileBasedVariableManager._resolve_variable` semantics):

```python
def _compose_variable_content(
    self,
    content: str,
    current_file_path: Path,
    root_space: str,
    tmpl_type: str,
    visited: Optional[Set[Path]] = None,
    max_depth: int = 10,
) -> str:
    """Recursively resolve {{ X }} references inside `content`.

    Resolution order for each X:
      1. Sibling lookup: same directory as current_file_path.
      2. Cascade lookup: walk cascade paths.
      3. If unresolved: leave literal (outer Jinja2 render may resolve it).

    Cycle detection via `visited`. Bounded recursion at `max_depth`.
    Reuses FileBasedVariableManager._resolve_variable internally.
    """
    if visited is None:
        visited = set()
    visited.add(current_file_path)
    if len(visited) > max_depth:
        raise MaxDepthExceededError(
            f"Variable composition exceeded max_depth={max_depth} at {current_file_path}"
        )
    # Extract {{ X }} references where X is a bare identifier (no dots, no filters).
    # For each, look up sibling-then-cascade; if found, recurse; substitute result.
    # Use a simple regex (Jinja2 syntax is well-defined; pattern: {{\s*([A-Za-z_]\w*)\s*}}).
    # Returns the substituted content.
    # ... see Cursor plan §1.C for the implementation skeleton ...
```

Update the body of `load_variables` to call composition after loading:

```python
content, file_path = self._cascade_load_variable_path(var_dir, file_stem or default_version, root_space, tmpl_type)
if content is not None and file_path is not None:
    content = self._compose_variable_content(content, file_path, root_space, tmpl_type)
_store_nested(nested_path, content if content is not None else "")
```

### 4.4 Edit D — AgentFoundation: delete the workaround + replace with `load_variables` call

**Delete** `_inject_mode_flags_and_content` from `templated_inferencer_base.py` (lines 213-276) and its call site (lines 198-205).

**Replace** with this body in `_build_template_feed` (combines Cursor's unified-specs approach with v1's deep-merge correctness):

```python
# Build effective specs: user-declared template_variables + per-enabled-mode entries.
# enable_<name> flags are always set (so {%- if enable_X %} can short-circuit even when False).
effective_specs: dict = dict(self.template_variables or {})
for mode_name, enabled in (self.modes or {}).items():
    feed[f"enable_{mode_name}"] = bool(enabled)
    if enabled:
        # setdefault: don't clobber an explicit user-supplied spec for the same key.
        effective_specs.setdefault(f"instructions.modes.{mode_name}", None)

if effective_specs and self.template_manager and hasattr(self.template_manager, "load_variables"):
    try:
        resolved = self.template_manager.load_variables(
            variable_specs=effective_specs,
            root_space=self.template_root_space or "",
            # CRITICAL: do NOT hardcode "main" — let TemplateManager use
            # its active_template_type (fixes the latent bug at the OLD
            # line 245 that passed literal "main").
        )
    except FileNotFoundError as e:
        logger.debug("Variable not found, degrading gracefully: %s", e)
        resolved = {}
    _deep_merge_into(feed, resolved)
```

**Add** the `_deep_merge_into` helper (recursive — closes Claude#1's latent shallow-merge bug):

```python
def _deep_merge_into(target: dict, source: dict) -> None:
    """Recursively merge source into target. Dicts merge; non-dict leaves overwrite.
    Used to fold load_variables output into the feed without losing sibling sub-namespaces."""
    for k, v in source.items():
        existing = target.get(k)
        if isinstance(existing, dict) and isinstance(v, dict):
            _deep_merge_into(existing, v)
        else:
            target[k] = v
```

> **Why `_deep_merge_into` and not Cursor's `feed.update(resolved)`?** Cursor's `feed.update(resolved)` shallow-overwrites `feed["instructions"]` if it already exists from `template_variables` (the user could pass both `template_variables={"instructions.behavior.X": None}` AND have modes enabled, producing two separate `instructions.*` sub-namespaces in `resolved`'s output). Deep-merge is the correct semantics; `dict.update` is the latent footgun.

---

## 5. Test plan

### 5.1 Phase 0 — RED tests (pin contract before any source edit)

**File (NEW):** `RichPythonUtils/test/string_utils/formatting/template_manager/test_load_variables_multidot.py`

`TestMultiLevelDotKeys` class — 8 tests, `xfail(strict=True)` before fix:

| # | Test | What it pins |
|---|---|---|
| 1 | `test_flat_key_unchanged` | Backward compat: `{"task_preamble": "default"}` resolves identically |
| 2 | `test_two_level_dotted_key_unchanged` | Backward compat: `{"notes.local_search_efficiency": ""}` → 2-level nested dict |
| 3 | `test_three_level_dotted_key` | `{"instructions.modes.deep_mode": ""}` → 3-level nested dict |
| 4 | `test_four_level_dotted_key` | `{"a.b.c.d": ""}` → 4-level nested dict |
| 5 | `test_multiple_keys_share_intermediates` | `{"a.b.c1": "", "a.b.c2": ""}` produces `{"a": {"b": {"c1": ..., "c2": ...}}}` |
| 6 | `test_flat_then_nested_raises_assertion` | `{"a": "literal", "a.b": ""}` raises `AssertionError` with documented message |
| 7 | `test_strict_prefix_still_raises_on_missing` | `{"instructions.modes.nonexistent": "@strict"}` raises `FileNotFoundError` |
| 8 | `test_literal_prefix_still_skips_file` | `{"instructions.modes.x": "=hello"}` → `{"instructions": {"modes": {"x": "hello"}}}` (no FS read) |

`TestNestedVariableComposition` class — 5 tests, `xfail(strict=True)` before fix (Cursor's catch — sibling composition does NOT work today):

| # | Test | What it pins |
|---|---|---|
| 9 | `test_sibling_reference_resolves` | `_variables/x/parent.jinja2` contains `{{ sibling }}`; `_variables/x/sibling.jinja2` exists. `load_variables({"x.parent": None})` returns content with sibling expanded (no literal `{{ sibling }}`) |
| 10 | `test_cascade_reference_when_no_sibling` | Parent file references `{{ shared }}` with no same-folder sibling; resolved via cascade walk |
| 11 | `test_unresolved_reference_left_literal` | `{{ unknown_var }}` with no sibling and no cascade match is left literal (outer template can fall back) |
| 12 | `test_circular_reference_detected` | A→B→A raises `CircularReferenceError` (or framework equivalent) |
| 13 | `test_max_recursion_depth_enforced` | Chain longer than max_depth raises `MaxDepthExceededError` |

### 5.2 AgentFoundation integration tests (should GREEN after Edit D)

- `test_behavior_variable_injection.py` (156 lines, FAILS today) — 3 fixture-level tests + the `TestNestedVariableExpansion` + `TestFullTemplateRendering` classes all GREEN after Edit C (composition) lands.
- Existing mode tests (`test_templated_inferencer_modes.py` M1–M9): all stay GREEN. M5 monkeypatches `_cascade_load_variable` — preserved via wrapper (§4.3 Option A).
- `test_preflight_template_variable_coverage.py` `_inject_mode_flags` stub at lines 148-153: see §9 migration.

### 5.3 Permanent regression test in AgentFoundation (NEW)

**File (NEW):** `CoreProjects/AgentFoundation/test/agent_foundation/common/inferencers/test_templated_inferencer_no_workaround.py`

```python
def test_inject_mode_flags_and_content_workaround_is_removed():
    """Prevent regression: the _inject_mode_flags_and_content workaround
    must NOT come back. Mode injection should go through
    TemplateManager.load_variables with proper multi-dot support.
    See plan §4.4. If this test fails, someone re-added the hack — push
    back on the PR and direct them to fix load_variables instead."""
    from agent_foundation.common.inferencers.templated_inferencer_base import (
        TemplatedInferencerBase,
    )
    assert not hasattr(TemplatedInferencerBase, "_inject_mode_flags_and_content"), (
        "_inject_mode_flags_and_content was re-added to TemplatedInferencerBase. "
        "Mode injection should go through load_variables() which now supports "
        "multi-dot keys. See _docs/_plans/load_variables_multidot_INTEGRATED_v3_plan.md §4."
    )

def test_no_hardcoded_tmpl_type_main_in_mode_injection():
    """The OLD workaround hard-coded tmpl_type='main' (line 245). The
    replacement must let TemplateManager use active_template_type.
    Regression guard: scan _build_template_feed source for the literal."""
    import inspect
    from agent_foundation.common.inferencers.templated_inferencer_base import (
        TemplatedInferencerBase,
    )
    src = inspect.getsource(TemplatedInferencerBase._build_template_feed)
    # Guard against the specific anti-pattern from the deleted workaround:
    assert 'tmpl_type="main"' not in src and "tmpl_type='main'" not in src, (
        "Mode injection re-introduced hard-coded tmpl_type='main' — this was "
        "a latent bug in the old workaround. Use active_template_type via load_variables."
    )
```

### 5.4 RichPythonUtils full-suite regression

Run these existing test files; all must continue GREEN:
- `test_template_manager_load_variable.py`
- `test_cross_root_variable_lookup.py`
- `test_predefined_variables_integration.py`
- `test_variable_two_pass_search.py`
- `test_variable_manager.py`

### 5.5 OpenStartup integration tests

Run these to confirm BRTA / MultiFlow / DualInferencer integration still resolves variables correctly:
- `test_task_agent_config_brta_with_multiflow_pti.py`
- `test_template_split_integration.py`

---

## 6. Phased rollout

| Phase | What | Files | Risk | Reversible? |
|---|---|---|---|---|
| 0 | RED tests in RichPythonUtils (13 tests, all `xfail(strict=True)`) | 1 new test file | none | n/a |
| 1 | RichPythonUtils Edit A (multi-dot split) | 1 file | medium | yes |
| 2 | RichPythonUtils Edit B (`_store_nested`) | same file | low | yes |
| 3 | RichPythonUtils Edit C (extract `_cascade_load_variable_path` + add `_compose_variable_content`) | same file | medium-high (composition is new behavior) | yes |
| 4 | RichPythonUtils: 13 unit tests GREEN; full-suite regression GREEN | new + existing test files | low | yes |
| 5 | AgentFoundation Edit D (delete workaround + add `_deep_merge_into` + replacement call site) | 1 file | medium | yes |
| 6 | AgentFoundation: migrate `test_preflight_template_variable_coverage.py` stub (§9) | 1 test file | low | yes |
| 7 | AgentFoundation: permanent regression tests added (`test_templated_inferencer_no_workaround.py`) | 1 new test file | trivial | yes |
| 8 | AgentFoundation: `test_behavior_variable_injection.py` GREEN; mode tests M1-M9 GREEN | n/a | n/a | n/a |
| 9 | OpenStartup integration tests GREEN | n/a | n/a | n/a |

### 6.1 Cross-repo PR ordering

- **PR-A (RichPythonUtils):** Phases 0–4. Lands first. Strictly backward-compatible. Tag a new RichPythonUtils version.
- **PR-B (AgentFoundation):** Phases 5–9. Depends on PR-A's tag. Bumps `pyproject.toml` to new RichPythonUtils version.

**Do NOT bundle.** Cross-repo bundles are hard to revert and bisect.

### 6.2 Rollback strategy

| Revert | Consequence |
|---|---|
| PR-B only | Workaround comes back; modes work via old path; `instructions.behavior.*` references fail (pre-bug state). Safe. |
| PR-A only AFTER PR-B has landed | AgentFoundation has no workaround to compensate; modes render empty. **Do not revert PR-A while PR-B is live.** |
| Both (in order: PR-B first, then PR-A) | Full pre-state. Safe. |

---

## 7. Risk register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Existing caller uses a 2-level dot key where the *version* contains a literal dot (e.g., `"notes.v1.2"` meaning "notes folder, version v1.2") — new split changes semantics | 🟡 Medium | **Cursor's catch.** Pre-flight grep for all `load_variables(` callers + all `template_variables=` dicts across AgentFoundation + OpenStartup + RichPythonUtils. Investigation found: only flat and 2-level keys; no caller uses dotted version strings. |
| 2 | `_compose_variable_content` (Edit C) is genuinely new behavior — could change output of existing 2-level variable files that happen to contain literal `{{ X }}` patterns the author didn't expect to be expanded | 🟡 Medium | **Cursor's risk.** Grep existing variable files for `{{ ` patterns. Today only the user's new `file_reading_fallback_for_review.jinja2` and `file_reading_fallback_for_followup.jinja2` contain such patterns; both expect expansion (the whole point of this plan). |
| 3 | `_store_nested` assertion fires on flat-vs-nested prefix collision | 🟢 Low | NEW error mode; no existing caller mixes flat + nested under shared prefix. Test #6 pins it. |
| 4 | Subclass overrides `_inject_mode_flags_and_content` | 🟡 Medium | Pre-flight grep: only `templated_inferencer_base.py` defines it. Test stub in `test_preflight_template_variable_coverage.py` replicates but does not subclass. Migrate stub (§9). |
| 5 | Hard-coded `tmpl_type="main"` was masking a latent bug for non-main inferencers; removing it changes behavior for those | 🟢 Low | No inferencer today uses non-`main` `active_template_type` AND uses modes. The fix is correct; latent bug closed. |
| 6 | Test monkeypatches of `_cascade_load_variable` break if signature changes | 🟢 Low | **Option A architecture preserves signature.** `_cascade_load_variable` stays `(var_name, version, root_space, tmpl_type) -> Optional[str]`. Tests work unmodified. |
| 7 | Composition recursion deep-stacks or slow | 🟢 Low | `max_depth=10` cap + cycle detection (matching `FileBasedVariableManager` semantics). Pinned by tests #12, #13. |
| 8 | `_deep_merge_into` recursion on adversarial dict (deeply nested) | 🟢 Low | Bounded by `template_variables` key depth — typically ≤ 4. Negligible. |
| 9 | Cross-repo coordination: PR-A merges, PR-B doesn't get rebased onto new RichPythonUtils version | 🟡 Medium | Pin RichPythonUtils version in `pyproject.toml` as part of PR-B. CI catches mismatch via new tests. |
| 10 | Future contributor re-adds `_inject_mode_flags_and_content` or hardcodes `tmpl_type="main"` | 🟢 Low | Permanent regression tests §5.3 catch both at CI time. |

---

## 8. Acceptance criteria

**PR-A (RichPythonUtils) mergeable when:**
- ☐ All 13 new unit tests in §5.1 pass.
- ☐ Existing RichPythonUtils test suite (§5.4) passes — zero regressions.
- ☐ `load_variables` docstring updated with N-level example + composition note + prefix-collision warning.
- ☐ `_cascade_load_variable` public signature unchanged (verified by grep).
- ☐ New `_cascade_load_variable_path` and `_compose_variable_content` have docstrings citing this plan section.

**PR-B (AgentFoundation) mergeable when:**
- ☐ `_inject_mode_flags_and_content` removed from `templated_inferencer_base.py`.
- ☐ `_deep_merge_into` helper added (module-private).
- ☐ Replacement uses `load_variables` and does NOT hardcode `tmpl_type="main"`.
- ☐ All 3 tests in `test_behavior_variable_injection.py` GREEN.
- ☐ Mode tests M1-M9 in `test_templated_inferencer_modes.py` GREEN.
- ☐ `test_preflight_template_variable_coverage.py` stub migrated (§9) and GREEN.
- ☐ Both permanent regression tests in §5.3 in CI.
- ☐ `grep -rn "_inject_mode_flags_and_content" CoreProjects/AgentFoundation/` returns zero matches.
- ☐ `grep -rn 'tmpl_type *= *["\\\']main["\\\']' CoreProjects/AgentFoundation/src/` returns zero matches (or only justified ones).
- ☐ `pyproject.toml` bumped to new RichPythonUtils version.
- ☐ OpenStartup integration tests (§5.5) GREEN.

---

## 9. Test stub migration (Cursor's catch)

**File:** `test/agent_foundation/common/inferencers/test_dual_inferencer/test_preflight_template_variable_coverage.py`

Lines 148-159 contain:
```python
def _inject_mode_flags(self, feed):
    """Simulate _inject_mode_flags_and_content."""
    for name, enabled in self.modes.items():
        feed[f"enable_{name}"] = bool(enabled)
        if enabled:
            content = self.template_manager._cascade_load_variable(
                ...
            )
```

This stub replicates the workaround inside a test scaffold. After workaround removal it should either:

**Option 1 (preferred):** Delete the stub and let the test use the production `_build_template_feed` (which now uses `load_variables`).

**Option 2:** Replace the stub body with the same `effective_specs` + `load_variables` pattern used in Edit D, so the test scaffold mirrors production behavior.

The plan picks **Option 1** because mirroring production via the actual path produces a stronger test (no parallel implementation drift).

---

## 10. What this plan deliberately does NOT do

- ❌ Does NOT rename `load_variables`.
- ❌ Does NOT change `@strict` / `=literal` / plain value semantics.
- ❌ Does NOT change `FileBasedVariableManager._find_variable_file` or `_resolve_variable` (already correct).
- ❌ Does NOT change `_cascade_load_variable`'s public signature (Option A preserves it via wrapper).
- ❌ Does NOT introduce a feature flag (fix is strictly backward-compatible).
- ❌ Does NOT touch any prompt template files.
- ❌ Does NOT add an `_inject_behavior_content` or any analogous per-namespace workaround.
- ❌ Does NOT add a new variable-spec prefix character. Composition is the default behavior of `load_variables`, opt-out is unnecessary today.

---

## 11. Comparison + "if forced to pick one"

| Aspect | v1 (mine, 441L) | v2 (mine+Claude#1, 365L) | Cursor (226L) | Claude#2 (14L) | **v3** |
|---|---|---|---|---|---|
| Multi-dot split fix | ✅ | ✅ | ✅ | yields | ✅ |
| `_store` correctness | over-engineered 15L TypeError | 5L w/ assertion | clean 3L (silent shadow) | yields | **5L w/ assertion (v2's)** |
| **`_compose_variable_content` (composition)** | ❌ missed | ❌ missed | ✅ caught | yields | ✅ adopt Cursor's |
| **`_cascade_load_variable_path` extraction (BC for monkeypatches)** | ❌ missed | ❌ missed | ✅ caught | yields | ✅ adopt Cursor's |
| **Hard-coded `tmpl_type="main"` bug** | ❌ missed (carried over) | ❌ missed | ✅ caught | yields | ✅ adopt Cursor's fix |
| **`test_preflight_template_variable_coverage.py` stub migration** | ❌ missed | ❌ missed | ✅ caught | yields | ✅ adopt Cursor's plan §3.C |
| `_deep_merge_into` (deep) vs `dict.update` (shallow) | ✅ deep | ✅ deep | ❌ shallow (latent bug) | yields | ✅ deep (mine) |
| Phase 0 RED tests | ✅ 8 | ✅ 8 | ❌ "add tests" only | yields | **13 (8 multidot + 5 composition)** |
| Risk register | ✅ 8 risks | ✅ 8 risks | ✅ 4 risks (good ones) | yields | **10 risks (merged)** |
| Cross-repo PR ordering | ✅ explicit | ✅ explicit | ✅ "rollout order" 3 steps | yields | ✅ explicit (mine + rollback matrix) |
| Permanent regression test (workaround-must-not-return) | ✅ | ✅ | ❌ | yields | ✅ + hardcoded-main guard |
| Acceptance criteria checkboxes | ✅ | ✅ | ❌ | yields | ✅ extended |
| Cited specific existing tests by name | partial | partial | ✅ by file:line | yields | ✅ adopt Cursor's citations |
| Length | 441L (verbose) | 365L | 226L (lean) | 14L (yield) | **~500L (covers more, denser)** |

### 11.1 If forced to pick ONE plan today — **Cursor's plan**, not mine.

This is a different answer than I gave in the previous round. Why I changed:
1. **Cursor catches 3 real correctness issues** that v1 and v2 missed:
   - `_cascade_load_variable` does raw `read_text()` — no composition. The user's `{{ file_reading_fallback }}` inside `file_reading_fallback_for_review.jinja2` does NOT work today. v1/v2 incorrectly assumed sibling resolution "already works." It doesn't.
   - Hard-coded `tmpl_type="main"` in the workaround that v1/v2's replacement would have carried over.
   - Test stub at `test_preflight_template_variable_coverage.py:148-153` that needs migration.
2. **Architectural correctness > operational rigor.** A plan that's operationally rigorous but produces an incomplete fix wastes more time than a leaner plan that gets the architecture right.
3. **Cursor's `_cascade_load_variable_path` + thin-wrapper pattern is genuinely elegant.** Preserves monkeypatch back-compat without ceremony.

What Cursor's plan lacks that v1/v2 has:
- Permanent regression test (workaround-must-not-return).
- Acceptance criteria checkboxes.
- Risk register severity ratings.
- Deep-merge correctness (Cursor uses `feed.update(resolved)` shallow — latent bug).

If you only shipped Cursor's plan, you'd:
- Get the correct architecture and all the right code edits.
- Be vulnerable to the workaround being re-added later (no regression guard).
- Have a latent shallow-merge bug that triggers the first time `instructions.modes` and `instructions.behavior` co-exist in the feed.

If you only shipped v1, you'd:
- Get the operational discipline.
- Land an incomplete fix that doesn't solve the user's stated `instructions.behavior.*` use case (because composition is missing).
- Carry the hard-coded `tmpl_type="main"` latent bug.
- Need a second round of work.

**Cursor's plan ships a correct fix; v1 ships a half-fix with discipline. Correctness wins.**

**But you don't have to pick.** v3 = Cursor's architecture + Cursor's composition + Cursor's back-compat wrapper + Cursor's test stub migration + v1's risk register + v1's cross-repo PR discipline + v1's permanent regression test + v1's deep-merge correctness. Strictly better than any single input plan.

---

## 12. Design principles applied

1. **Fix at the source.** Bug is in RichPythonUtils.
2. **Strict backward compatibility.** Multi-dot is a superset of single-dot. `_cascade_load_variable` signature unchanged.
3. **Eliminate workarounds atomically with the fix.** Pin absence with permanent tests.
4. **Composition reuses existing semantics** (`FileBasedVariableManager._resolve_variable`) — don't reinvent.
5. **Wrapper preserves test monkeypatches.** Architectural change with zero test migration burden in RichPythonUtils.
6. **Clear errors over silent corruption.** `_store_nested` assertion + `_compose_variable_content` cycle/depth errors.
7. **Cross-repo discipline.** PR-A then PR-B, ordered, not bundled.
8. **Elegant over clever.** Adopt Cursor's clean 3-line setdefault; add ONE assertion. Adopt v1's recursive `_deep_merge_into` over Cursor's shallow `update`. No ego.
9. **Pin the architecture with permanent tests.** Two regression guards: workaround-must-not-return + no-hardcoded-tmpl-type-main.

---

## 13. Estimated effort

| Phase | Impl | Tests | Review | Total (h) |
|---|---|---|---|---|
| 0 (RED tests) | 0 | 3 | 1 | 4 |
| 1-3 (RichPythonUtils fix incl. composition) | 4 | 0 | 2 | 6 |
| 4 (RichPythonUtils tests GREEN + regression) | 0 | 2 | 1 | 3 |
| 5 (AgentFoundation removal + replacement) | 2 | 0 | 1.5 | 3.5 |
| 6 (stub migration) | 0.5 | 0 | 0.5 | 1 |
| 7 (permanent regression tests) | 0.5 | 1 | 0.5 | 2 |
| 8 (existing tests GREEN) | 0 | 1 | 0.5 | 1.5 |
| 9 (OpenStartup integration) | 0 | 1 | 0.5 | 1.5 |
| **Total** | **7** | **8** | **7.5** | **22.5** |

Roughly **3 engineer-days**, distributed across both repos. Larger than v1's 14h estimate because composition (Edit C) is genuinely new behavior that needs its own test class and careful regression coverage.

---

*End of v3 integrated plan. Reviewers: please challenge §3.3 (the wrapper architecture — is composition the right default for `load_variables`?), §4.3 (the `_compose_variable_content` algorithm — is regex-based extraction safe or should we use a proper Jinja2 AST?), §6.1 (cross-repo PR ordering), §11.1 ("Cursor's plan over v1 if forced to one"), and §7 risk #2 (existing variable files containing literal `{{ X }}` patterns).*

